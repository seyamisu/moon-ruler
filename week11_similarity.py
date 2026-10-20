"""WEEK 11 - Cosine similarity and recommendation ranking for Moon Ruler."""
import numpy as np
from week10_dataset import WEIGHTS, build_feature_matrix, load_books

LIKED_THRESHOLD = 4


def cosine_similarity_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    unit = X / norms
    return unit @ unit.T


def cosine_to_vector(X, vector):
    norms = np.linalg.norm(X, axis=1)
    norms[norms == 0] = 1
    vector_norm = np.linalg.norm(vector) or 1
    return (X @ vector) / (norms * vector_norm)


def build_user_vector(df, X, feature_names, ratings=None, favorites=None, genres=None):
    ratings = ratings or {}
    favorites = favorites or []
    genres = genres or []
    title_to_index = {title: i for i, title in enumerate(df["title"])}
    vector = np.zeros(X.shape[1])
    total = 0.0

    # Strongest signal: explicit ratings.
    # 4-5 stars pull the taste vector toward similar books.
    # 1-2 stars gently push the taste vector away from similar books.
    # 3 stars is treated as neutral evidence.
    for title, rating in ratings.items():
        if title not in title_to_index:
            continue
        if rating >= LIKED_THRESHOLD:
            weight = rating / 5
            vector += weight * X[title_to_index[title]]
            total += abs(weight)
        elif rating <= 2:
            weight = 0.35 if rating == 2 else 0.55
            vector -= weight * X[title_to_index[title]]
            total += abs(weight)

    # Saved/favorite books are a softer signal than a 5-star rating.
    for title in favorites:
        if title in title_to_index:
            vector += 0.75 * X[title_to_index[title]]
            total += 0.75

    # Genre preferences solve the cold-start problem before the user has ratings.
    for genre in genres:
        name = f"genre:{genre}"
        if name in feature_names:
            seed = np.zeros(X.shape[1])
            seed[feature_names.index(name)] = WEIGHTS["genre"]
            vector += 0.7 * seed
            total += 0.7

    return None if total == 0 else vector / total


def _filter(result, exclude_titles=(), category=None, genre=None):
    if exclude_titles:
        result = result[~result["title"].isin(list(exclude_titles))]
    if category:
        result = result[result["category"] == category]
    if genre:
        result = result[result["genre"] == genre]
    return result


def _diversify(result, top_n, max_per_author=1):
    """Keep recommendation shelves from being dominated by a single author."""
    selected, author_counts = [], {}
    for idx, row in result.iterrows():
        author = row["author"]
        if author_counts.get(author, 0) >= max_per_author:
            continue
        selected.append(idx)
        author_counts[author] = author_counts.get(author, 0) + 1
        if len(selected) >= top_n:
            break
    if len(selected) < top_n:
        for idx in result.index:
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= top_n:
                break
    return result.loc[selected]


def recommend(df, X, user_vector, exclude_titles=(), category=None, genre=None, top_n=5, diversify=True):
    result = df.copy()
    result["score"] = cosine_to_vector(X, user_vector)
    result = _filter(result, exclude_titles, category, genre)
    result = result.sort_values(["score", "rating", "title"], ascending=[False, False, True])
    return _diversify(result, top_n) if diversify else result.head(top_n)


def default_recommendations(df, exclude_titles=(), category=None, genre=None, top_n=5):
    """Stable, diverse cold-start shelf when no personalized vector is available."""
    result = _filter(df.copy(), exclude_titles, category, genre)
    result = result.sort_values(["rating", "title"], ascending=[False, True])
    return _diversify(result, top_n)




class ContentRecommender:
    """Object-oriented facade for Moon Ruler's content-based recommender.

    The class keeps the mathematical implementation in the tested helper
    functions below while giving the project a clear recommender object that
    owns the catalog feature matrix, feature names, similarity matrix, user
    vector construction, and ranking operations.
    """

    def __init__(self, df, X, feature_names):
        self.df = df
        self.X = X
        self.feature_names = feature_names
        self._similarity_matrix = None

    @property
    def similarity_matrix(self):
        if self._similarity_matrix is None:
            self._similarity_matrix = cosine_similarity_matrix(self.X)
        return self._similarity_matrix

    def build_user_vector(self, ratings=None, favorites=None, genres=None):
        return build_user_vector(
            self.df, self.X, self.feature_names,
            ratings=ratings, favorites=favorites, genres=genres,
        )

    def recommend(self, user_vector, exclude_titles=(), category=None, genre=None, top_n=5, diversify=True):
        return recommend(
            self.df, self.X, user_vector, exclude_titles=exclude_titles,
            category=category, genre=genre, top_n=top_n, diversify=diversify,
        )

    def default_recommendations(self, exclude_titles=(), category=None, genre=None, top_n=5):
        return default_recommendations(
            self.df, exclude_titles=exclude_titles, category=category,
            genre=genre, top_n=top_n,
        )

    def similar_books(self, title, top_n=6, same_category=True):
        matches = self.df.index[self.df["title"] == title].tolist()
        if not matches:
            return self.df.iloc[0:0].copy()
        idx = int(matches[0])
        result = self.df.copy()
        result["score"] = self.similarity_matrix[idx]
        result = result[result.index != idx]
        if same_category:
            result = result[result["category"] == self.df.iloc[idx]["category"]]
        return result.sort_values(["score", "rating", "title"], ascending=[False, False, True]).head(top_n)


def main():
    df = load_books()
    X, feature_names = build_feature_matrix(df)
    sim = cosine_similarity_matrix(X)
    print("=== WEEK 11: SIMILARITY DEMO ===")
    seed = "The Martian"
    if seed in set(df["title"]):
        i = int(df.index[df["title"] == seed][0])
        order = np.argsort(sim[i])[::-1]
        print(f"\nBooks most similar to '{seed}':")
        for j in [x for x in order if x != i][:5]:
            print(f"  {sim[i][j]:.2f}  {df.iloc[j]['title']} ({df.iloc[j]['genre']})")

    uv = build_user_vector(df, X, feature_names,
        ratings={"Dune": 5, "Harry Potter and the Sorcerer's Stone": 4},
        genres=["Science Fiction", "Fantasy"])
    print("\nDemo personalized picks:")
    for _, row in recommend(df, X, uv, exclude_titles=["Dune", "Harry Potter and the Sorcerer's Stone"], category="Fiction", top_n=5).iterrows():
        print(f"  {row['score']:.2f}  {row['title']} ({row['genre']})")


if __name__ == "__main__":
    main()
