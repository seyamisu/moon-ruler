# Moon Ruler — Week 13 Test Report

**Automated result:** 54/54 checks passed  
**Status:** READY FOR FINAL DEMO

Week 13.6 additionally validates local admin-role persistence across project-folder upgrades and compact heart/star dashboard actions.

```text
PASS | Dataset contains exactly 1,000 books | rows=1000
PASS | Required columns are present
PASS | No missing required values
PASS | No duplicate title/author pairs
PASS | Ratings remain within 0-5
PASS | Category balance is 500 Fiction / 500 Academics | {'Fiction': 500, 'Academics': 500}
PASS | Ten genres contain 100 books each
PASS | Every book has at least one tag
PASS | Feature matrix has one row per book | shape=(1000, 710)
PASS | Feature matrix contains only finite numbers
PASS | Cosine matrix is 1000 x 1000 | shape=(1000, 1000)
PASS | Cosine self-similarity is 1.0
PASS | OOP ContentRecommender class is available
PASS | ContentRecommender owns a valid similarity matrix
PASS | Cold-start Science Fiction preference returns Science Fiction | genres={'Science Fiction': 10}
PASS | Fantasy + 5-star Harry Potter keeps recommendations in Fantasy
PASS | Already-rated books are excluded
PASS | Recommendation shelf is author-diverse | unique_authors=10
PASS | Programming preference surfaces multiple computing-related STEM books | computing_matches=4/10
PASS | 1-star negative feedback lowers similar-book scores | before=0.550, after=-0.372
PASS | users.json is valid JSON
PASS | No plaintext password field is stored
PASS | Stored accounts use salt + hash credentials
PASS | Admin-style add operation survives CSV reload
PASS | Admin-style delete operation survives CSV reload
PASS | Moon Ruler branding is present
PASS | Old Book of SEYA branding is absent
PASS | Route exists: dashboard
PASS | Route exists: interests
PASS | Route exists: recommendations
PASS | Route exists: profile_page
PASS | Route exists: admin
PASS | Recommendations page contains See more
PASS | Admin panel contains Book Management
PASS | Admin panel contains User Management
PASS | Book Detail route exists
PASS | How It Works route exists
PASS | Read / unread state is implemented
PASS | Book Detail supports save / unsave
PASS | Book Detail supports 1-5 star rating
PASS | Book Detail shows similar books
PASS | Admin panel shows dataset statistics
PASS | Proposal vs Final System view is present
PASS | Recommendation diagnostics are present
PASS | Admin role has a durable registry
PASS | Admin role survives local project-folder upgrades
PASS | Dashboard uses heart and star icon actions
PASS | Preset profile avatars are implemented
PASS | Custom JPG/PNG profile upload is implemented
PASS | All genre cards have visual icons
PASS | Book-cover lookup with offline fallback is implemented
PASS | Recommendation details live inside the book card and interests use visual grids
PASS | Recommendation cards omit the old because-you-liked prefix
PASS | Book Detail header includes a cover image

========================================================================
WEEK 13 RESULT: 54/54 checks passed
STATUS: READY FOR FINAL DEMO
========================================================================

```
