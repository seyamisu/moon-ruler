"""WEEK 10 - Dataset preparation and feature vectors for Moon Ruler."""
from pathlib import Path
import re
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "books.csv"
REQUIRED_COLUMNS = ["title", "author", "category", "genre", "tags", "rating"]
WEIGHTS = {"category": 1.0, "genre": 3.0, "tags": 2.0, "author": 1.5}

# Week 13 refinement: add high-signal semantic tags from title keywords.
# These rules do not replace the researched CSV tags; they only enrich them so
# books with very generic base tags can still be distinguished meaningfully.
SEMANTIC_TITLE_RULES = [
    # Computing / STEM
    (r"\bcode\b|\bprogrammer\b|\bprogramming\b|\bsoftware\b|\brefactoring\b", ["programming", "software-engineering"]),
    (r"\balgorithm(s)?\b|\bdata structures?\b", ["algorithms", "computer-science"]),
    (r"\bmachine learning\b|\bdeep learning\b|\bartificial intelligence\b", ["machine-learning", "artificial-intelligence"]),
    (r"\bcomputer networks?\b|\bnetworking\b", ["computer-networks", "systems"]),
    (r"\boperating systems?\b", ["operating-systems", "systems"]),
    (r"\bdatabase(s)?\b|\bdata-intensive\b", ["databases", "data-systems"]),
    (r"\bjava\b", ["java", "programming"]),
    (r"\bstatistics?\b|\bstatistical\b|\bprobability\b", ["statistics", "probability"]),
    (r"\bcalculus\b|\blinear algebra\b|\balgebra\b|\bmathematics\b|\bproof\b|\bprove\b", ["mathematics", "mathematical-reasoning"]),
    (r"\bphysics\b|\bquantum\b|\belectrodynamics\b|\bmechanics\b", ["physics", "mathematics"]),
    (r"\bcosmos\b|\bastrophysics\b|\bpale blue dot\b|\bbrief history of time\b|\belegant universe\b", ["astronomy", "physics"]),
    (r"\bdouble helix\b", ["genetics", "biology"]),
    (r"\bchemistry\b|\bbiochemistry\b", ["chemistry", "laboratory-science"]),
    (r"\bbiology\b|\bgenetics\b|\bevolution\b|\bgene\b|\bcell\b", ["biology", "life-science"]),
    (r"\bcircuit(s)?\b|\belectronics\b|\bsignals? and systems\b|\bcontrol systems?\b", ["electrical-engineering", "systems"]),
    (r"\bthermodynamics\b|\bfluid mechanics\b|\bheat and mass transfer\b|\bmaterials science\b", ["mechanical-engineering", "engineering"]),

    # Business / economics
    (r"\binvest(or|ing|ment)?\b|\bstocks?\b|\bwall street\b|\bsecurity analysis\b", ["investing", "finance"]),
    (r"\beconomics?\b|\bmarkets?\b", ["economics", "markets"]),
    (r"\bmarketing\b|\badvertising\b|\bpositioning\b", ["marketing", "strategy"]),
    (r"\bleadership\b|\bmanagement\b|\bmanager\b|\bexecutive\b", ["leadership", "management"]),
    (r"\bstartup\b|\bentrepreneur\b|\binnovation\b", ["entrepreneurship", "innovation"]),
    (r"\bmoney\b|\bfinance\b|\bfinancial\b", ["finance", "decision-making"]),

    # Medicine / health
    (r"\banatomy\b|\bhistology\b|\bembryology\b", ["anatomy", "clinical"]),
    (r"\bphysiology\b", ["physiology", "clinical"]),
    (r"\bpathology\b|\bpathologic\b", ["pathology", "disease"]),
    (r"\bpharmacology\b|\bpsychopharmacology\b", ["pharmacology", "medicine"]),
    (r"\bepidemiology\b|\bpublic health\b", ["epidemiology", "public-health"]),
    (r"\bneuro(logy|science|anatomy)?\b", ["neuroscience", "medicine"]),
    (r"\bsurgery\b|\boperative\b", ["surgery", "clinical"]),

    # Fiction cues
    (r"\bdragon(s)?\b", ["dragons", "magic"]),
    (r"\bwitch\b|\bwizard\b|\bmagic\b|\benchanted\b", ["magic", "supernatural"]),
    (r"\bking\b|\bqueen\b|\bcrown\b|\bkingdom\b", ["royalty", "kingdoms"]),
    (r"\bmurder\b|\bdetective\b|\bcase\b|\bwitness\b|\bcrime\b", ["crime", "investigation"]),
    (r"\bsecret\b|\bmissing\b|\bsilent\b|\bguest\b", ["secrets", "suspense"]),
    (r"\bspace\b|\bmars\b|\bplanet\b|\borbit\b|\bsolar\b", ["space", "exploration"]),
    (r"\brobot(s)?\b|\bandroid(s)?\b|\bartificial intelligence\b", ["robots", "artificial-intelligence"]),
    (r"\btime machine\b|\btime traveler\b|\btime travel\b", ["time-travel", "science-fiction"]),
    (r"\blove\b|\bkiss\b|\bbride\b|\bwedding\b|\bheart\b", ["love", "relationships"]),
]


# Remove a small set of known false-positive tags left by the earlier automatic
# title matcher (for example, "grokking" accidentally triggering "king").
NOISY_TAGS_BY_GENRE = {
    "STEM": {"art", "aesthetics", "kingdoms", "power"},
    "Business & Economics": {"art", "aesthetics", "kingdoms", "war", "conflict", "love", "relationships"},
    "Social Sciences": {"kingdoms", "secrets", "suspense", "art", "aesthetics"},
    "Medicine & Health": {"kingdoms", "power", "art", "aesthetics", "love", "relationships", "war", "conflict", "space", "survival", "exploration", "literature", "criticism", "detective", "investigation"},
    "Romance": {"art", "aesthetics", "dragons", "magic"},
    "Mystery & Thriller": {"art", "aesthetics", "dragons", "magic", "programming", "software-engineering"},
    "Science Fiction": {"art", "aesthetics"},
    "Historical & Literary Fiction": {"programming", "software-engineering"},
}

def _sanitize_tags(genre, tags):
    blocked = NOISY_TAGS_BY_GENRE.get(str(genre), set())
    return [t for t in tags if t not in blocked]

def _enrich_tags(title, tags):
    low = str(title).lower()
    out = list(tags)
    for pattern, extras in SEMANTIC_TITLE_RULES:
        if re.search(pattern, low):
            for tag in extras:
                if tag not in out:
                    out.append(tag)
    return out


def load_books(path=DATA_PATH):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"books.csv is missing columns: {missing}")

    for col in ["title", "author", "category", "genre"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    df["tags"] = (
        df["tags"].fillna("").astype(str)
        .apply(lambda s: [t.strip().lower() for t in s.split("|") if t.strip()])
    )
    df["tags"] = [
        _sanitize_tags(genre, _enrich_tags(title, tags))
        for title, genre, tags in zip(df["title"], df["genre"], df["tags"])
    ]
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0).clip(0, 5)
    df = df.drop_duplicates(subset=["title", "author"]).reset_index(drop=True)

    # Stable taxonomy used throughout the V2 application.
    allowed_categories = {"Fiction", "Academics"}
    if not set(df["category"]).issubset(allowed_categories):
        bad = sorted(set(df["category"]) - allowed_categories)
        raise ValueError(f"Unexpected categories in dataset: {bad}")
    return df


def build_feature_matrix(df):
    feature_names, columns = [], []

    def add_group(prefix, values_per_book, weight):
        vocabulary = sorted({v for values in values_per_book for v in values if v})
        for term in vocabulary:
            feature_names.append(f"{prefix}:{term}")
            columns.append(np.array([weight if term in values else 0.0 for values in values_per_book]))

    add_group("category", [[c] for c in df["category"]], WEIGHTS["category"])
    add_group("genre", [[g] for g in df["genre"]], WEIGHTS["genre"])
    add_group("tags", list(df["tags"]), WEIGHTS["tags"])
    add_group("author", [[a] for a in df["author"]], WEIGHTS["author"])

    matrix = np.column_stack(columns) if columns else np.empty((len(df), 0))
    return matrix, feature_names


def main():
    df = load_books()
    matrix, feature_names = build_feature_matrix(df)
    print("=== WEEK 10: DATASET SUMMARY ===")
    print(f"Books loaded: {len(df)}")
    print("\nBooks per category:")
    print(df["category"].value_counts().to_string())
    print("\nBooks per genre:")
    print(df["genre"].value_counts().sort_index().to_string())
    print(f"\nFeature matrix shape: {matrix.shape} (books x features)")
    print(f"Unique features: {len(feature_names)}")


if __name__ == "__main__":
    main()
