"""WEEK 13 - Final validation tests for Moon Ruler.

Run:
    python week13_tests.py

The script does not modify the real dataset or user accounts.
"""
from pathlib import Path
import json
import tempfile

import numpy as np
import pandas as pd

from week10_dataset import DATA_PATH, REQUIRED_COLUMNS, build_feature_matrix, load_books
from week11_similarity import ContentRecommender, build_user_vector, cosine_similarity_matrix, cosine_to_vector, recommend

BASE_DIR = Path(__file__).parent
APP_PATH = BASE_DIR / "week12_app.py"
USERS_PATH = BASE_DIR / "data" / "users.json"

results = []


def check(name, condition, detail=""):
    passed = bool(condition)
    results.append((name, passed, detail))
    print(f"{'PASS' if passed else 'FAIL'} | {name}" + (f" | {detail}" if detail else ""))
    return passed


def main():
    df = load_books()
    X, feature_names = build_feature_matrix(df)

    # 1. Dataset integrity
    check("Dataset contains exactly 1,000 books", len(df) == 1000, f"rows={len(df)}")
    check("Required columns are present", all(c in df.columns for c in REQUIRED_COLUMNS))
    check("No missing required values", not df[REQUIRED_COLUMNS].isna().any().any())
    check("No duplicate title/author pairs", df.duplicated(["title", "author"]).sum() == 0)
    check("Ratings remain within 0-5", df["rating"].between(0, 5).all())
    check(
        "Category balance is 500 Fiction / 500 Academics",
        df["category"].value_counts().to_dict() == {"Fiction": 500, "Academics": 500},
        str(df["category"].value_counts().to_dict()),
    )
    genre_counts = df["genre"].value_counts()
    check("Ten genres contain 100 books each", len(genre_counts) == 10 and genre_counts.eq(100).all())
    check("Every book has at least one tag", df["tags"].map(bool).all())

    # 2. Feature matrix / cosine math
    check("Feature matrix has one row per book", X.shape[0] == len(df), f"shape={X.shape}")
    check("Feature matrix contains only finite numbers", np.isfinite(X).all())
    sim = cosine_similarity_matrix(X)
    check("Cosine matrix is 1000 x 1000", sim.shape == (1000, 1000), f"shape={sim.shape}")
    check("Cosine self-similarity is 1.0", np.allclose(np.diag(sim), 1.0, atol=1e-8))
    engine = ContentRecommender(df, X, feature_names)
    check("OOP ContentRecommender class is available", engine.__class__.__name__ == "ContentRecommender")
    check("ContentRecommender owns a valid similarity matrix", engine.similarity_matrix.shape == (1000, 1000))

    # 3. Recommendation behavior
    uv = build_user_vector(df, X, feature_names, genres=["Science Fiction"])
    picks = recommend(df, X, uv, category="Fiction", top_n=10)
    check(
        "Cold-start Science Fiction preference returns Science Fiction",
        len(picks) == 10 and picks["genre"].eq("Science Fiction").all(),
        f"genres={picks['genre'].value_counts().to_dict()}",
    )

    hp = "Harry Potter and the Sorcerer's Stone"
    uv = build_user_vector(df, X, feature_names, ratings={hp: 5}, genres=["Fantasy"])
    picks = recommend(df, X, uv, exclude_titles=[hp], category="Fiction", top_n=10)
    check(
        "Fantasy + 5-star Harry Potter keeps recommendations in Fantasy",
        len(picks) == 10 and picks["genre"].eq("Fantasy").all(),
    )
    check("Already-rated books are excluded", hp not in set(picks["title"]))
    check("Recommendation shelf is author-diverse", picks["author"].nunique() >= 8, f"unique_authors={picks['author'].nunique()}")

    uv = build_user_vector(df, X, feature_names, ratings={"Clean Code": 5}, genres=["STEM"])
    picks = recommend(df, X, uv, exclude_titles=["Clean Code"], category="Academics", top_n=10)
    computing_tags = {"programming", "computer-science", "software-engineering", "algorithms", "databases", "data-systems"}
    computing_matches = sum(bool(set(tags) & computing_tags) for tags in picks["tags"])
    check(
        "Programming preference surfaces multiple computing-related STEM books",
        computing_matches >= 4,
        f"computing_matches={computing_matches}/10",
    )

    # Negative feedback should reduce the similarity of books like a 1-star seed.
    uv_genre = build_user_vector(df, X, feature_names, genres=["Science Fiction"])
    uv_dislike = build_user_vector(df, X, feature_names, ratings={"Dune": 1}, genres=["Science Fiction"])
    sf_mask = (df["genre"] == "Science Fiction").to_numpy()
    before = float(cosine_to_vector(X, uv_genre)[sf_mask].mean())
    after = float(cosine_to_vector(X, uv_dislike)[sf_mask].mean())
    check("1-star negative feedback lowers similar-book scores", after < before, f"before={before:.3f}, after={after:.3f}")

    # 4. User/account storage sanity (read-only)
    users = json.loads(USERS_PATH.read_text(encoding="utf-8")) if USERS_PATH.exists() else {}
    check("users.json is valid JSON", isinstance(users, dict))
    sensitive_plaintext = any("password" in acct for acct in users.values() if isinstance(acct, dict))
    check("No plaintext password field is stored", not sensitive_plaintext)
    creds_ok = all("salt" in acct and "hash" in acct for acct in users.values() if isinstance(acct, dict))
    check("Stored accounts use salt + hash credentials", creds_ok)

    # 5. Admin-style CSV round-trip on a temporary copy only.
    raw = pd.read_csv(DATA_PATH)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "books_test.csv"
        raw.to_csv(tmp, index=False)
        test_row = pd.DataFrame([{
            "title": "Moon Ruler Week 13 Test Book",
            "author": "QA Tester",
            "category": "Fiction",
            "genre": "Science Fiction",
            "tags": "space|future|technology",
            "rating": 4.2,
        }])
        updated = pd.concat([pd.read_csv(tmp), test_row], ignore_index=True)
        updated.to_csv(tmp, index=False)
        loaded = load_books(tmp)
        add_ok = len(loaded) == 1001 and "Moon Ruler Week 13 Test Book" in set(loaded["title"])
        check("Admin-style add operation survives CSV reload", add_ok)
        updated = updated[updated["title"] != "Moon Ruler Week 13 Test Book"]
        updated.to_csv(tmp, index=False)
        deleted = load_books(tmp)
        check("Admin-style delete operation survives CSV reload", len(deleted) == 1000)

    # 6. Final app source checks
    app_text = APP_PATH.read_text(encoding="utf-8")
    check("Moon Ruler branding is present", "Moon Ruler" in app_text)
    check("Old Book of SEYA branding is absent", "Book of SEYA" not in app_text and "BOOK OF SEYA" not in app_text)
    for route in ["dashboard", "interests", "recommendations", "profile_page", "admin"]:
        check(f"Route exists: {route}", f'"{route}"' in app_text)
    check("Recommendations page contains See more", '"See more"' in app_text)
    check("Admin panel contains Book Management", "Book Management" in app_text)
    check("Admin panel contains User Management", "User Management" in app_text)
    check("Book Detail route exists", '"book_detail"' in app_text and "def page_book_detail" in app_text)
    check("How It Works route exists", '"about"' in app_text and "def page_about" in app_text)
    check("Read / unread state is implemented", "def toggle_read" in app_text and "read_books" in app_text)
    check("Book Detail supports save / unsave", "♡ Save" in app_text and "♥ Unsave" in app_text)
    check("Book Detail supports 1-5 star rating", 'st.radio("Your rating", options=[1,2,3,4,5]' in app_text)
    check("Book Detail shows similar books", "Similar books" in app_text and "content similarity" in app_text)
    check("Admin panel shows dataset statistics", 'metric("Books"' in app_text and "Dataset distribution" in app_text)
    check("Proposal vs Final System view is present", "Proposal vs Final System" in app_text and "Original proposal" in app_text)
    check("Recommendation diagnostics are present", "Recommendation diagnostics" in app_text and "Ranking method" in app_text)
    check("Admin role has a durable registry", "ADMINS_PATH" in app_text and "set_admin_role" in app_text and "effective_role" in app_text)
    check("Admin role survives local project-folder upgrades", "USER_ADMIN_REGISTRY_PATH" in app_text and '.moon_ruler' in app_text)
    check("Dashboard uses professional Material vector icons", 'if return_page == "dashboard"' in app_text and ':material/favorite_border:' in app_text and ':material/favorite:' in app_text and ':material/star_border:' in app_text and ':material/star:' in app_text)
    check("Preset profile avatars are implemented", "avatar_preset_" in app_text and "avatar_value" in app_text)
    check("Custom JPG/PNG profile upload is implemented", "file_uploader" in app_text and "image_file_to_data_uri" in app_text)
    check("All genre cards have visual icons", "GENRE_ICONS" in app_text and "render_genre_badge" in app_text)
    check("Book-cover lookup with offline fallback is implemented", "get_book_cover" in app_text and "fallback_cover_data_uri" in app_text)
    check("Recommendation details live inside the book card and interests use visual grids", "Details  ▷" in app_text and "Saved books" in app_text and "Reading History & Ratings" in app_text)
    check("Recommendation cards omit the old because-you-liked prefix", "Because you liked…" not in app_text)
    check("Book Detail header includes a cover image", "hero_cover" in app_text and "get_book_cover(row[\"title\"], row[\"author\"])" in app_text)

    passed = sum(ok for _, ok, _ in results)
    total = len(results)
    print("\n" + "=" * 72)
    print(f"WEEK 13 RESULT: {passed}/{total} checks passed")
    print("STATUS:", "READY FOR FINAL DEMO" if passed == total else "REVIEW FAILED CHECKS BEFORE DEMO")
    print("=" * 72)
    raise SystemExit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
