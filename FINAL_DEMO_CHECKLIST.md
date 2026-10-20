# Moon Ruler — Final Demo Checklist

## Before class
- [ ] Run `pip install -r requirements.txt` if dependencies are not installed.
- [ ] Run `python week13_tests.py` and confirm **54/54 checks passed**.
- [ ] Run `streamlit run week12_app.py`.
- [ ] Prepare one reader demo account with 2–3 selected genres.
- [ ] Give that reader at least one 5★ rating, one low rating, and one saved book.
- [ ] Prepare one admin account with `python grant_admin.py <username>` once; confirm the role remains after restarting the app.
- [ ] Keep the final PowerPoint open separately.

## Recommended live demo order
1. Login as the prepared reader.
2. Show Dashboard and explain the three-book preview shelves.
3. Open Recommendations and point out **Mode**, **Taste strength**, **Rated**, and **Read excluded**.
4. Open **Recommendation diagnostics** and explain the ranking pipeline.
5. Use **See more**.
6. Open a **Book Detail** page.
7. Demonstrate **Save for later**, **Mark as read**, and a **1–5★ rating**.
8. Point out the **book cover**, integrated **View details**, **match percentage**, **why recommended**, and **Similar books**.
9. Open **My Interests** and show genre icons, the **Saved Books** cover grid, and **Reading History & Ratings** grid.
10. Open **How It Works → Proposal vs Final System** and show that the final app still follows the proposal.
11. Open **How It Works** and briefly explain the algorithm, then show **Proposal vs Final System**.
12. Show Profile, change a preset avatar or upload a JPG/PNG, and point out the locked main category.
13. Login as admin.
14. Show Admin statistics, Book Management, and User Management.

## Key explanation to memorize
- Genre weight = 3.0
- Tags weight = 2.0
- Author weight = 1.5
- Category weight = 1.0
- Saved books = softer positive signal
- 4–5★ = positive recommendation signal
- 3★ = neutral
- 1–2★ = negative signal
- Cosine similarity ranks books by vector direction / feature overlap
- Already-read books are excluded from recommendation candidates
- `ContentRecommender` provides the object-oriented structure stated in the proposal

## Proposal answer to memorize
> Moon Ruler keeps the original content-based recommendation scope: it uses genre, author, and tags with cosine similarity, learns from books the reader marks or rates, explains recommendation evidence, works local-first, and uses NumPy, Pandas, and an object-oriented recommender structure. The additional admin, profile, negative-feedback, and book-detail features are enhancements, not a change of project direction.

## Backup plan
If live interaction fails, use screenshots in the presentation and show `week13_test_output.txt` with **50/50 PASS**. Do not edit core code during the presentation.
