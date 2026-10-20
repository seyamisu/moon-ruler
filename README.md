# Moon Ruler — Week 13.4 UI & Persistence Upgrade

Content-based book recommendation system for ECE261.

## Run

```bash
pip install -r requirements.txt
python week13_tests.py
streamlit run week12_app.py
```

The current automated validation result is **54/54 checks passed**.

## Reader pages

- **Dashboard** — overview of interests, recommendation shelves, search, and profile
- **My Interests** — visual genre icons, a Saved Books cover grid, and a Reading History & Ratings cover grid
- **Recommendations** — cover-first recommendation cards, integrated View Details action, 3 books initially per genre, See more / Show less, personalization metrics, and diagnostics
- **Book Detail** — metadata, match score, why recommended, Save, Read/Unread, 1–5★ rating, and Similar Books
- **Profile** — reader information, preset avatars, JPG/PNG profile-picture upload, and recommendation settings
- **How It Works** — two instructor-facing tabs: algorithm explanation and Proposal vs Final System

## Proposal alignment

The final implementation keeps the original scope:

- local-first / offline-capable content-based recommendation
- user read/liked/rating feedback
- genre, author, and tag comparison
- cosine similarity ranking
- explanation of recommendation evidence
- NumPy for vector math, Pandas for data handling, and a `ContentRecommender` class for the recommender structure
- Week 13 testing and recommendation refinement

Enhancements such as category weighting, separate Save/Read/Rate states, negative feedback, dedicated pages, Similar Books, and admin tools extend the proposal without changing the core method.

## Admin Panel

### Overview
- total books
- total users
- Fiction count
- Academics count
- dataset distribution by genre

### Book Management
- add
- edit
- delete with confirmation
- cache refresh after catalog changes

### User Management
- safe account table (no password hashes/salts displayed)
- search
- promote/demote admin with persistent `data/admins.json` registry
- reset profile/preferences
- delete test accounts with confirmation

## Reader feedback signals

Moon Ruler distinguishes:
- **Saved** — mild positive interest; the book can still be recommended until read
- **Read** — excluded from future recommendation candidates
- **Rated 4–5★** — positive content signal
- **Rated 3★** — neutral
- **Rated 1–2★** — negative content signal

## Recommendation model

Weighted content features:
- category = 1.0
- genre = 3.0
- tags = 2.0
- author = 1.5

A user taste vector is built from genres, saved books, and ratings. Candidate books are ranked using cosine similarity, then filtered by the locked main category and shelf genre, with author diversity applied.

## Dataset

- 1,000 books
- 500 Fiction
- 500 Academics
- 10 genres
- 100 books per genre

Required fields: `title, author, category, genre, tags, rating`

## Current limitation

The application still uses local `books.csv` and `users.json` storage. This is appropriate for the classroom/final demo and local testing, but persistent public deployment should eventually use a cloud database.


## Week 13.4 UI upgrade

- Admin roles now persist in a separate `data/admins.json` registry and are synchronized with `users.json`.
- Run `python grant_admin.py <username>` **once**; normal app restarts and profile resets no longer remove that role.
- Profile pictures support eight presets or a custom JPG/PNG up to 2 MB.
- All ten genres have distinct visual icons.
- Recommendation cards request book covers from Open Library when online and use an offline Moon Ruler fallback cover when unavailable.
- The Details action is inside the recommendation card.
- My Interests now follows the saved-books / reading-history card-grid direction from the UI reference.


## Week 13.4 UI polish
- Recommendation cards now use clear Details / Save / Rate controls.
- Removed the old “Because you liked…” line from recommendation cards.
- Rate opens Book Detail instead of silently assigning a 5-star rating.
- Book Detail now shows the cover beside the title and uses a compact Save / Read action row.
- Rating uses a visible 1–5 star choice with a Save / update rating action.


## Week 13.6 dashboard + admin persistence fix
- Dashboard recommendation cards use compact heart/save and star/rate icons.
- The full Recommendations page keeps descriptive action labels.
- Admin usernames are synchronized to both `data/admins.json` and `~/.moon_ruler/admins.json`.
- The user-level registry lets admin status survive extracting a newer Moon Ruler project folder on the same computer.
- After upgrading from an older build, run `python grant_admin.py <username>` one final time to seed the user-level registry.


## Week 13.6 UI polish
- Dashboard Save and Rate actions now use Streamlit Material Symbols (SVG/vector icons), not emoji or Unicode heart/star glyphs.
- Icons use fixed 46 px hit areas and 23 px vector marks to prevent clipping.

## Public test deployment (Streamlit Community Cloud)

This package is safe to upload to a GitHub repository for a short public test. `data/users.json` starts empty so local test-account hashes are not published.

1. Push the project files to GitHub.
2. In Streamlit Community Cloud choose **Create app**.
3. Select the repository and branch, then set the entrypoint to `week12_app.py`.
4. In **Advanced settings > Secrets**, add:

```toml
admin_usernames = ["your_admin_username"]
```

5. Deploy, create/register that username in Moon Ruler, and the app will recognize it as an admin.

### Important test limitation

The current user/account activity and admin book edits are still stored in local JSON/CSV files. Community Cloud can recreate the app filesystem, so those runtime changes are not guaranteed to persist through restarts or redeployments. This is appropriate for public browser testing and the class demo, but a persistent database is recommended before production use.
