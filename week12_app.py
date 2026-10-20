"""WEEK 12 - Moon Ruler: Figma-faithful Streamlit UI, revised layout."""
from __future__ import annotations

import base64
import hashlib
import html
import json
import re
import secrets
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from week10_dataset import build_feature_matrix, load_books
from week11_similarity import (
    ContentRecommender,
    LIKED_THRESHOLD,
    build_user_vector,
    cosine_similarity_matrix,
    default_recommendations,
    recommend,
)

BASE_DIR = Path(__file__).parent
USERS_PATH = BASE_DIR / "data" / "users.json"
ADMINS_PATH = BASE_DIR / "data" / "admins.json"
USER_ADMIN_REGISTRY_PATH = Path.home() / ".moon_ruler" / "admins.json"
BOOKS_PATH = BASE_DIR / "data" / "books.csv"
DATA_DIR = BASE_DIR / "data"

# Ensure writable runtime state files have a parent directory even if a deployment
# accidentally omits the state files. The catalog itself must still be shipped.
DATA_DIR.mkdir(parents=True, exist_ok=True)
if not USERS_PATH.exists():
    USERS_PATH.write_text("{}\n", encoding="utf-8")
if not ADMINS_PATH.exists():
    ADMINS_PATH.write_text(json.dumps({"admins": []}, indent=2) + "\n", encoding="utf-8")

TIERS = {
    "New to the Shelf": {"blurb": "Just starting out", "max_genres": 0},
    "Page Turner": {"blurb": "Reads occasionally", "max_genres": 3},
    "Bookworm": {"blurb": "Reads regularly and extensively", "max_genres": 5},
}
GENRE_ICONS = {
    "Fantasy": "🐉", "Romance": "💕", "Mystery & Thriller": "🔎",
    "Science Fiction": "🚀", "Historical & Literary Fiction": "🕰️",
    "STEM": "🧪", "Business & Economics": "📈", "Social Sciences": "🌐",
    "Humanities & Arts": "🎨", "Medicine & Health": "⚕️",
}
OLD_GENRE_MAP = {
    "Sci-Fi": "Science Fiction", "Mystery": "Mystery & Thriller",
    "Historical": "Historical & Literary Fiction", "Mathematics": "STEM",
    "Programming": "STEM", "Data Science": "STEM",
    "Psychology": "Social Sciences", "Horror": "Mystery & Thriller",
}
GENDERS = ["Female", "Male", "Non-binary", "Prefer not to say"]

st.set_page_config(page_title="Moon Ruler", page_icon="🌙", layout="wide", initial_sidebar_state="collapsed")


# -----------------------------------------------------------------------------
# Data + persistence
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_data():
    df = load_books()
    X, feature_names = build_feature_matrix(df)
    engine = ContentRecommender(df, X, feature_names)
    return df, X, feature_names, engine.similarity_matrix


def _migrate_user(user: dict) -> tuple[dict, bool]:
    changed = False
    genres = []
    for genre in user.get("genres", []):
        mapped = OLD_GENRE_MAP.get(genre, genre)
        if mapped not in genres:
            genres.append(mapped)
        changed |= mapped != genre
    user["genres"] = genres
    for field, default in [
        ("favorites", []),
        ("ratings", {}),
        ("read_books", list(user.get("ratings", {}).keys())),
        ("profile_done", False),
        ("role", "reader"),
        ("avatar_type", "preset"),
        ("avatar_value", "🌙"),
    ]:
        if field not in user:
            user[field] = default
            changed = True
    return user, changed


def _read_admin_registry(path):
    if not path.exists():
        return set()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw = raw.get("admins", [])
        return {str(u).strip().lower() for u in raw if str(u).strip()}
    except Exception:
        return set()


def _secret_admins():
    """Read optional admin usernames from Streamlit Secrets for cloud deployment."""
    try:
        raw = st.secrets.get("admin_usernames", [])
    except Exception:
        return set()
    if isinstance(raw, str):
        raw = [raw]
    return {str(u).strip().lstrip("@").lower() for u in raw if str(u).strip()}


def load_admins():
    """Load admin usernames from project, local-machine, and cloud-secret registries.

    Local development persists admins in data/admins.json and ~/.moon_ruler/admins.json.
    Streamlit Community Cloud can bootstrap admins from the `admin_usernames` secret.
    """
    admins = (
        _read_admin_registry(ADMINS_PATH)
        | _read_admin_registry(USER_ADMIN_REGISTRY_PATH)
        | _secret_admins()
    )
    return admins


def save_admins(admins):
    payload = json.dumps({"admins": sorted(set(admins))}, indent=2, ensure_ascii=False)
    ADMINS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ADMINS_PATH.write_text(payload, encoding="utf-8")
    try:
        USER_ADMIN_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        USER_ADMIN_REGISTRY_PATH.write_text(payload, encoding="utf-8")
    except OSError:
        # Project-level registry still works on restricted environments.
        pass


def load_users():
    if not USERS_PATH.exists():
        return {}
    users = json.loads(USERS_PATH.read_text(encoding="utf-8"))
    admins = load_admins()
    changed = False
    registry_changed = False
    for key, user in users.items():
        users[key], did_change = _migrate_user(user)
        changed |= did_change
        if users[key].get("role") == "admin" and key not in admins:
            admins.add(key)
            registry_changed = True
        if key in admins and users[key].get("role") != "admin":
            users[key]["role"] = "admin"
            changed = True
    if registry_changed:
        save_admins(admins)
    if changed:
        save_users(users)
    return users


def save_users(users):
    USERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    USERS_PATH.write_text(json.dumps(users, indent=2, ensure_ascii=False), encoding="utf-8")


def set_admin_role(username, make_admin=True):
    username = username.strip().lstrip("@").lower()
    users = load_users()
    if username not in users:
        raise KeyError(username)
    admins = load_admins()
    if make_admin:
        admins.add(username)
        users[username]["role"] = "admin"
    else:
        admins.discard(username)
        users[username]["role"] = "reader"
    save_admins(admins)
    save_users(users)


def effective_role(username, user=None):
    if username in load_admins():
        return "admin"
    return (user or {}).get("role", "reader")


def hash_password(password, salt):
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def current_user():
    return load_users()[st.session_state.username]


def update_user(**changes):
    users = load_users()
    users[st.session_state.username].update(changes)
    save_users(users)


def age_from(iso_date):
    born, today = date.fromisoformat(iso_date), date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def genres_for_scope(df, scope):
    categories = ["Fiction", "Academics"] if scope == "Free Reader" else [scope]
    return [g for g in df["genre"].drop_duplicates() if df.loc[df["genre"] == g, "category"].iloc[0] in categories]


def avatar_markup(user, size=82):
    """Return safe HTML for a preset emoji or uploaded JPG/PNG avatar."""
    if user.get("avatar_type") == "custom" and str(user.get("avatar_value", "")).startswith("data:image/"):
        src = html.escape(user["avatar_value"], quote=True)
        return f'<img class="avatar-img" src="{src}" alt="Profile picture" style="width:{size}px;height:{size}px">'
    avatar = html.escape(str(user.get("avatar_value", "🌙")))
    return f'<div class="avatar-ring" style="width:{size}px;height:{size}px">{avatar}</div>'


def image_file_to_data_uri(uploaded):
    if uploaded is None:
        return None
    if uploaded.type not in {"image/png", "image/jpeg"}:
        raise ValueError("Please upload a JPG or PNG image.")
    data = uploaded.getvalue()
    if len(data) > 2 * 1024 * 1024:
        raise ValueError("Profile pictures must be 2 MB or smaller.")
    return f"data:{uploaded.type};base64,{base64.b64encode(data).decode('ascii')}"


def fallback_cover_data_uri(title, author=""):
    safe_title = html.escape(str(title)[:36])
    safe_author = html.escape(str(author)[:28])
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="240" height="360" viewBox="0 0 240 360">'
        '<rect width="240" height="360" rx="10" fill="#E7C77E"/>'
        '<rect x="8" y="8" width="224" height="344" rx="8" fill="none" stroke="#76543A" stroke-width="3"/>'
        f'<text x="120" y="135" text-anchor="middle" font-family="Georgia,serif" font-size="18" fill="#2F2018">{safe_title}</text>'
        f'<text x="120" y="235" text-anchor="middle" font-family="Georgia,serif" font-size="12" fill="#6D5138">{safe_author}</text>'
        '<text x="120" y="325" text-anchor="middle" font-family="Georgia,serif" font-size="16" fill="#76543A">MOON RULER</text>'
        '</svg>'
    )
    return "data:image/svg+xml;utf8," + quote(svg)


@st.cache_data(ttl=60 * 10, show_spinner=False)
def cover_service_available():
    """Probe Open Library once so offline use does not wait on every book card."""
    try:
        req = Request("https://openlibrary.org/", headers={"User-Agent": "MoonRuler/1.0"})
        with urlopen(req, timeout=1.2) as response:
            return 200 <= getattr(response, "status", 200) < 500
    except Exception:
        return False


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def get_book_cover(title, author=""):
    """Resolve a cover through Open Library and fall back to a local title card."""
    if not cover_service_available():
        return fallback_cover_data_uri(title, author)
    try:
        params = urlencode({"title": str(title), "author": str(author), "limit": 1, "fields": "cover_i,title,author_name"})
        req = Request("https://openlibrary.org/search.json?" + params, headers={"User-Agent": "MoonRuler/1.0"})
        with urlopen(req, timeout=2.5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        docs = payload.get("docs", [])
        if docs and docs[0].get("cover_i"):
            return f"https://covers.openlibrary.org/b/id/{docs[0]['cover_i']}-M.jpg"
    except Exception:
        pass
    return fallback_cover_data_uri(title, author)


def render_genre_badge(genre):
    icon = GENRE_ICONS.get(genre, "📚")
    return f'<div class="genre-badge"><div class="genre-badge-icon">{icon}</div><div class="genre-badge-title">{html.escape(genre)}</div></div>'


# -----------------------------------------------------------------------------
# Recommendation helpers
# -----------------------------------------------------------------------------
def user_taste(df, X, feature_names, user):
    engine = ContentRecommender(df, X, feature_names)
    return engine.build_user_vector(
        ratings=user.get("ratings", {}),
        favorites=user.get("favorites", []),
        genres=user.get("genres", []),
    )


def why_recommended(row, df, sim, user, personalized=True):
    """Explain the strongest evidence behind a recommendation in plain language."""
    if not personalized:
        return f"Discovery pick from {row['genre']}; no personal ratings are needed yet."

    liked = [t for t, r in user.get("ratings", {}).items() if r >= LIKED_THRESHOLD]
    liked += user.get("favorites", [])
    title_to_index = {t: i for i, t in enumerate(df["title"])}
    liked_idx = [title_to_index[t] for t in dict.fromkeys(liked) if t in title_to_index]

    if liked_idx:
        seed_idx = max(liked_idx, key=lambda i: sim[row.name][i])
        seed = df.iloc[seed_idx]
        evidence = []
        if seed["genre"] == row["genre"]:
            evidence.append(f"same genre as {seed['title']}")
        if seed["author"] == row["author"]:
            evidence.append(f"same author ({row['author']})")
        shared = sorted(set(seed["tags"]) & set(row["tags"]))
        if shared:
            evidence.append("shared themes: " + ", ".join(shared[:3]))
        if evidence:
            return "Because you liked or saved " + seed["title"] + ": " + "; ".join(evidence)

    if row["genre"] in user.get("genres", []):
        tag_hint = ", ".join(list(row["tags"])[:2])
        return f"Matches your {row['genre']} interest" + (f" · themes: {tag_hint}" if tag_hint else "")
    return "Fits the combined pattern of your genres, saved books, and ratings."


def find_books(df, query):
    q = query.strip().lower()
    if not q:
        return df.iloc[0:0]
    mask = (
        df["title"].str.lower().str.contains(q, regex=False)
        | df["author"].str.lower().str.contains(q, regex=False)
        | df["genre"].str.lower().str.contains(q, regex=False)
        | df["tags"].apply(lambda tags: any(q in t for t in tags))
    )
    return df[mask]


def profile_strength(user):
    signals = len(user.get("genres", [])) + len(user.get("favorites", [])) + len(user.get("read_books", [])) + len(user.get("ratings", {})) * 2
    return min(100, 15 + signals * 9)


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
def inject_css():
    st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');
:root{--paper:#F5EFE5;--paper2:#FBF7F0;--ink:#3A2921;--brown:#765038;--brown2:#55382A;--gold:#DFB55D;--gold2:#E9CC8B;--line:#5E4637;--muted:#9A8C80;}
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:#262626;color:var(--ink)}
.block-container{max-width:1390px;padding:1.0rem 1.2rem 1.4rem}
#MainMenu,footer,header[data-testid="stHeader"]{visibility:hidden}
h1,h2,h3,h4{font-family:'Cormorant Garamond',serif!important;color:var(--ink)!important}

/* Book shell */
.st-key-book-login,.st-key-book-profile,.st-key-book-genres{
 background:linear-gradient(90deg,var(--paper) 0 49.25%,var(--line) 49.35% 49.48%,#F0E7DC 49.58% 50.42%,var(--line) 50.52% 50.65%,var(--paper) 50.75% 100%);
 border:1px solid #36261F;min-height:640px;padding:2.05rem 2.7rem 1.7rem;box-shadow:0 16px 50px rgba(0,0,0,.28);overflow:hidden
}
.st-key-book-dashboard{background:var(--paper);border:1px solid #36261F;min-height:690px;padding:1.2rem 1.35rem 1.5rem;box-shadow:0 16px 50px rgba(0,0,0,.28)}
.st-key-book-login [data-testid="stHorizontalBlock"],.st-key-book-profile [data-testid="stHorizontalBlock"],.st-key-book-genres [data-testid="stHorizontalBlock"]{align-items:flex-start}

/* Editorial text */
.figma-title{font-family:'Cormorant Garamond',serif;font-size:3rem;line-height:.88;letter-spacing:.03em;text-align:center;color:var(--ink);margin:0}
.figma-rule{height:1.5px;background:var(--ink);width:86%;margin:1rem auto}
.figma-quote{font-family:'Cormorant Garamond',serif;font-size:2.05rem;line-height:1.12;color:var(--ink);max-width:240px;margin:11rem auto 0}
.figma-section{font-family:'Cormorant Garamond',serif;font-size:1.72rem;font-weight:600;line-height:1.05;margin:0 0 .85rem;color:var(--ink)}
.figma-small{font-family:'Inter',sans-serif;font-size:.82rem;color:var(--muted);line-height:1.45}
.choice-title{font-family:'Cormorant Garamond',serif;font-size:1.08rem;font-weight:700;text-transform:uppercase;letter-spacing:.02em}
.choice-sub{font-size:.76rem;color:#7D6E61;margin-top:2px}

/* Inputs */
[data-testid="stTextInput"] input,[data-testid="stDateInput"] input,[data-baseweb="select"]>div,[data-testid="stSelectbox"]>div>div{
 background:rgba(255,255,255,.34)!important;border:1px solid #9C8A7B!important;border-radius:7px!important;min-height:43px;color:var(--ink)!important;box-shadow:none!important
}
[data-testid="stTextInput"] input:focus,[data-testid="stDateInput"] input:focus{border-color:var(--brown)!important}

/* Buttons become the Figma cards. */
.stButton>button,[data-testid="stFormSubmitButton"]>button{
 border:1px solid var(--brown)!important;border-radius:8px!important;background:transparent!important;color:var(--brown2)!important;font-weight:500!important;min-height:40px;box-shadow:none!important;transition:.12s ease
}
.stButton>button:hover,[data-testid="stFormSubmitButton"]>button:hover{background:#EFE0C7!important;border-color:var(--brown)!important;color:var(--brown2)!important}
.stButton>button[kind="primary"]{background:var(--gold2)!important;border-color:#9B703D!important;color:#3A2921!important;box-shadow:none!important}
.stButton>button[kind="primary"]:hover{background:var(--gold)!important}

/* compact tabs on cover */
button[data-baseweb="tab"]{font-family:'Cormorant Garamond',serif!important;font-weight:700!important;color:var(--brown)!important;font-size:1rem!important;padding-left:0!important;padding-right:1.2rem!important}
div[data-baseweb="tab-highlight"]{background:var(--brown)!important}

/* Choice card grids */
.tier-card-note,.scope-note,.genre-note{margin:-.45rem 0 .8rem .15rem}
.genre-count{font-size:.78rem;color:var(--muted);margin-top:.15rem}

/* Expanders */
div[data-testid="stExpander"]{background:rgba(255,255,255,.22);border:1px solid #CBBBAE;border-radius:8px}


/* Profile tier cards */
.st-key-tier-stack .stButton{margin-bottom:.72rem}
.st-key-tier-stack .stButton>button{min-height:70px!important;padding:.7rem 1rem!important;justify-content:flex-start!important;text-align:left!important}
.st-key-tier-stack .stButton>button p{white-space:pre-line!important;line-height:1.15!important;text-align:left!important;font-family:'Cormorant Garamond',serif!important;font-size:1.02rem!important}

/* Figma-like scope cards */
.st-key-scope-top .stButton>button{min-height:88px!important}
.st-key-scope-top .stButton>button p,.st-key-scope-free .stButton>button p{white-space:pre-line!important;line-height:1.05!important;font-family:'Cormorant Garamond',serif!important;font-size:1.05rem!important}
.st-key-scope-free{margin-top:.65rem}
.st-key-scope-free .stButton>button{min-height:56px!important}

/* Genre tiles */
.st-key-genre-grid .stButton>button{min-height:76px!important;padding:.5rem .55rem!important}
.st-key-genre-grid .stButton>button p{white-space:pre-line!important;line-height:1.08!important;text-align:center!important;font-family:'Cormorant Garamond',serif!important;font-size:.92rem!important;overflow-wrap:anywhere!important}

/* Page navigation */
.st-key-profile-nav,.st-key-genres-nav{margin-top:8.5rem}
.st-key-genres-nav.bookworm{margin-top:2.5rem}
.st-key-profile-nav .stButton>button,.st-key-genres-nav .stButton>button{min-height:38px!important;font-size:.9rem!important}

/* Compact recommendation cards */
.rec-card{background:var(--gold);border:1px solid #8A6031;border-radius:8px;padding:9px 11px;margin-bottom:8px}
.rec-top{display:flex;gap:10px;align-items:flex-start;justify-content:space-between}
.rec-card-title{font-family:'Cormorant Garamond',serif;font-weight:700;font-size:.96rem;line-height:1.1;color:var(--ink)}
.rec-card-meta{font-size:.68rem;color:#5A412B;white-space:nowrap;padding-top:1px}
.rec-card-why{font-family:'Cormorant Garamond',serif;font-style:italic;color:#6D5138;font-size:.78rem;line-height:1.25;margin-top:3px}
[class*='st-key-rec-actions-'] [data-testid='stHorizontalBlock']{gap:.35rem}
[class*='st-key-rec-actions-'] .stButton>button{min-height:30px!important;padding:.15rem .45rem!important;font-size:.78rem!important;border-radius:6px!important}

/* Dashboard vector icon actions — Material Symbols stay crisp and cannot be cropped like font glyphs. */
[class*="st-key-dash_save_"] .stButton>button,
[class*="st-key-dash_rate_"] .stButton>button{
 width:46px!important;min-width:46px!important;max-width:46px!important;
 height:46px!important;min-height:46px!important;padding:0!important;
 border-radius:11px!important;display:flex!important;align-items:center!important;justify-content:center!important;
 overflow:visible!important;
}
[class*="st-key-dash_save_"] .stButton>button p,
[class*="st-key-dash_rate_"] .stButton>button p{font-size:0!important;line-height:0!important;margin:0!important;}
[class*="st-key-dash_save_"] .stButton>button [data-testid="stIconMaterial"],
[class*="st-key-dash_rate_"] .stButton>button [data-testid="stIconMaterial"]{
 font-size:23px!important;width:23px!important;height:23px!important;line-height:23px!important;
 color:var(--brown2)!important;overflow:visible!important;
}

/* Dashboard density */
.st-key-book-dashboard{min-height:650px;padding:1.1rem 1.35rem 1.25rem}
.st-key-book-dashboard .figma-section{margin-bottom:.7rem}
.profile-card{min-height:440px}

/* Dashboard */
.dashboard-search [data-testid="stTextInput"] input{background:var(--paper2)!important;border-radius:12px!important;height:52px;font-size:.98rem}
.interest-card{background:var(--gold2);border:1px solid #977044;border-radius:8px;padding:11px 13px;min-height:68px;margin-bottom:11px}
.interest-title{font-family:'Cormorant Garamond',serif;text-transform:uppercase;font-size:1.05rem;font-weight:700}
.interest-line{font-family:'Cormorant Garamond',serif;font-size:.9rem;color:#5F4634;margin-top:2px}
.rec-title{font-family:'Cormorant Garamond',serif;text-transform:uppercase;font-size:1.15rem;font-weight:700;margin:.1rem 0 .35rem}
.rec-row{display:grid;grid-template-columns:1fr auto;gap:8px;align-items:center;background:var(--gold);border:1px solid #8A6031;border-radius:7px;padding:8px 10px;margin-bottom:5px;min-height:39px}
.rec-book{font-family:'Cormorant Garamond',serif;font-weight:700;font-size:.9rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.rec-meta{font-size:.68rem;color:#5A412B;white-space:nowrap}
.rec-why{font-family:'Cormorant Garamond',serif;font-style:italic;color:#6D5138;font-size:.8rem;margin:0 0 7px 2px}
.genre-divider{height:1px;background:#716357;opacity:.45;margin:8px 0 9px}
.profile-card{background:#FFFDFC;border:1px solid #70675F;border-radius:26px;padding:20px 18px;min-height:500px;text-align:center}
.avatar-ring{width:82px;height:82px;border:2px solid var(--brown);border-radius:50%;margin:0 auto 8px;display:flex;align-items:center;justify-content:center;font-size:2.1rem;color:var(--brown)}
.profile-name{font-family:'Cormorant Garamond',serif;font-size:1.85rem;line-height:1;margin-top:.35rem}
.profile-handle{color:#8F8A86;font-size:.88rem;margin-bottom:.8rem}
.profile-pill{display:inline-block;border:1px solid #6C6159;border-radius:9px;padding:5px 11px;margin:3px;font-size:.76rem}
.profile-tier{background:#DDA83A;border:1px solid #93621E;border-radius:9px;padding:7px;margin:14px 0 10px;font-size:.88rem}
.profile-rule{height:1px;background:#766A62;margin:15px 0}

/* suppress Streamlit label clutter */
[data-testid="stWidgetLabel"] p{font-size:.78rem;color:#69584D}

@media(max-width:900px){
 .st-key-book-login,.st-key-book-profile,.st-key-book-genres{background:var(--paper);padding:1.2rem;min-height:auto}
 .figma-quote{margin:2rem 0;font-size:1.7rem}.figma-title{font-size:2.35rem}
}

.st-key-profile-panel{background:#FFFDFC!important;border:1px solid #70675F!important;border-radius:26px!important;padding:1rem!important;overflow:hidden!important}
.st-key-profile-panel [data-testid="stVerticalBlock"]{gap:.55rem!important}
.profile-inner{text-align:center}
.profile-scope-label{font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:.25rem}
.profile-scope{border:1px solid #8A715F;border-radius:9px;padding:.45rem .55rem;font-family:'Cormorant Garamond',serif;font-size:1rem;background:#F8F1E6}
.locked-scope-card{border:1px solid #9B7552;background:#E9CC8B;border-radius:10px;min-height:112px;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;font-family:'Cormorant Garamond',serif;font-size:1.15rem;letter-spacing:.02em;margin-bottom:1rem}
.locked-scope-card small{font-family:'Inter',sans-serif;font-size:.68rem;color:#7D6A5A;margin-top:.35rem}
.locked-scope-icon{font-size:1.25rem;margin-bottom:.2rem}

/* V5 multi-page navigation */
.app-nav{display:flex;gap:.45rem;flex-wrap:wrap;margin:.15rem 0 1rem}
.page-shell{background:var(--paper);border:1px solid #36261F;min-height:650px;padding:1.2rem 1.35rem 1.5rem;box-shadow:0 16px 50px rgba(0,0,0,.28)}
.page-title{font-family:'Cormorant Garamond',serif;font-size:2rem;font-weight:600;margin:0 0 .15rem;color:var(--ink)}
.page-sub{font-size:.82rem;color:var(--muted);margin-bottom:1rem}
.section-link .stButton>button{font-family:'Cormorant Garamond',serif!important;font-size:1.15rem!important;font-weight:700!important;text-align:left!important;justify-content:flex-start!important;border:none!important;padding:.1rem 0!important;min-height:28px!important;background:transparent!important}
.admin-card{border:1px solid #C5B4A5;border-radius:10px;background:#FBF7F0;padding:1rem;margin-bottom:1rem}
.admin-table-note{font-size:.78rem;color:var(--muted);margin:.4rem 0 .8rem}
.stat-card{border:1px solid #CBBBAE;border-radius:10px;background:#FFFDFC;padding:.85rem 1rem;text-align:center}
.stat-number{font-family:'Cormorant Garamond',serif;font-size:1.7rem;font-weight:700;color:var(--brown2)}
.stat-label{font-size:.72rem;color:var(--muted)}

/* Week 13.3 visual cards */
.avatar-img{border-radius:50%;object-fit:cover;border:2px solid var(--brown);margin:0 auto 8px;display:block;background:#FFFDFC}
.genre-badge{background:linear-gradient(180deg,#E9CC8B,#E4BF70);border:1px solid #977044;border-radius:12px;padding:14px 10px;text-align:center;min-height:106px;display:flex;flex-direction:column;align-items:center;justify-content:center}
.genre-badge-icon{font-size:2rem;line-height:1;margin-bottom:8px}
.genre-badge-title{font-family:'Cormorant Garamond',serif;font-size:1rem;font-weight:700;line-height:1.05}
[class*='st-key-rec-book-card-']{background:var(--gold)!important;border:1px solid #8A6031!important;border-radius:16px!important;padding:.65rem .75rem!important;margin-bottom:.65rem!important}
[class*='st-key-rec-book-card-'] [data-testid='stImage'] img{border-radius:5px;box-shadow:0 2px 8px rgba(62,43,30,.16)}
.cover-title{font-family:'Cormorant Garamond',serif;font-size:1.12rem;font-weight:700;line-height:1.05;margin-bottom:.15rem}
.cover-meta{font-size:.72rem;color:#654A35;margin-bottom:.25rem}
.cover-reason{font-family:'Cormorant Garamond',serif;font-style:italic;font-size:.82rem;color:#684D38;line-height:1.2}
.match-badge{font-family:'Cormorant Garamond',serif;font-weight:700;text-align:right;font-size:1.02rem;padding-top:.15rem;white-space:nowrap}
[class*='st-key-rec-book-card-'] .stButton>button{min-height:34px!important;padding:.2rem .5rem!important;font-size:.84rem!important;border-radius:8px!important}
.book-detail-title{margin-top:.2rem!important}
.book-grid-title{font-family:'Cormorant Garamond',serif;font-weight:700;font-size:.9rem;line-height:1.05;min-height:36px;text-align:center}
.book-grid-rating{font-family:'Cormorant Garamond',serif;font-weight:700;font-size:.9rem;color:#76543A;text-align:center}
.interests-panel-title{font-family:'Cormorant Garamond',serif;font-size:1.7rem;font-weight:700;text-align:center;margin:.4rem 0 .7rem}


/* ============================================================
   MOON RULER — POLISHED MOTION SYSTEM
   CSS-only: safe for Streamlit Community Cloud
   ============================================================ */

/* Page entrance: subtle editorial reveal */
.st-key-book-dashboard,
.page-shell,
.st-key-book-login,
.st-key-book-profile,
.st-key-book-genres {
    animation: moonPageIn .42s cubic-bezier(.2,.7,.25,1) both;
}
@keyframes moonPageIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* General interactive controls */
.stButton > button,
[data-testid="stFormSubmitButton"] > button {
    transition:
        transform .18s ease,
        box-shadow .18s ease,
        background-color .18s ease,
        border-color .18s ease,
        color .18s ease !important;
    will-change: transform;
}
.stButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 7px 16px rgba(74,50,34,.10) !important;
}
.stButton > button:active,
[data-testid="stFormSubmitButton"] > button:active {
    transform: translateY(0) scale(.985);
    box-shadow: 0 3px 8px rgba(74,50,34,.08) !important;
}

/* Top navigation: polished hover and active state */
[class*="st-key-nav_"] .stButton > button,
.st-key-book-dashboard > div:first-child .stButton > button {
    position: relative;
    overflow: hidden;
}
[class*="st-key-nav_"] .stButton > button::after,
.st-key-book-dashboard > div:first-child .stButton > button::after {
    content: "";
    position: absolute;
    left: 14%;
    right: 14%;
    bottom: 5px;
    height: 1px;
    background: var(--brown);
    transform: scaleX(0);
    transform-origin: center;
    transition: transform .22s ease;
    opacity: .75;
}
[class*="st-key-nav_"] .stButton > button:hover::after,
.st-key-book-dashboard > div:first-child .stButton > button:hover::after {
    transform: scaleX(1);
}

/* Recommendation cards: lift + depth */
[class*='st-key-rec-book-card-'] {
    transition:
        transform .24s cubic-bezier(.2,.7,.25,1),
        box-shadow .24s ease,
        filter .24s ease;
    animation: moonCardIn .38s ease both;
    will-change: transform;
}
[class*='st-key-rec-book-card-']:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 26px rgba(78,53,34,.14) !important;
    filter: saturate(1.02);
}
@keyframes moonCardIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Book covers: gentle zoom only inside recommendation cards */
[class*='st-key-rec-book-card-'] [data-testid='stImage'] img {
    transition: transform .28s ease, box-shadow .28s ease, filter .28s ease;
    transform-origin: center;
}
[class*='st-key-rec-book-card-']:hover [data-testid='stImage'] img {
    transform: scale(1.045);
    box-shadow: 0 7px 16px rgba(62,43,30,.20) !important;
    filter: contrast(1.02);
}

/* Dashboard heart/star vector actions */
[class*="st-key-dash_save_"] .stButton > button,
[class*="st-key-dash_rate_"] .stButton > button {
    transition:
        transform .16s ease,
        background-color .16s ease,
        box-shadow .16s ease !important;
}
[class*="st-key-dash_save_"] .stButton > button:hover,
[class*="st-key-dash_rate_"] .stButton > button:hover {
    transform: translateY(-1px) scale(1.07);
    background: #F6E9D4 !important;
    box-shadow: 0 5px 12px rgba(80,55,35,.11) !important;
}
[class*="st-key-dash_save_"] .stButton > button:active,
[class*="st-key-dash_rate_"] .stButton > button:active {
    transform: scale(.93);
}
[class*="st-key-dash_save_"] .stButton > button [data-testid="stIconMaterial"],
[class*="st-key-dash_rate_"] .stButton > button [data-testid="stIconMaterial"] {
    transition: transform .16s ease, color .16s ease;
}
[class*="st-key-dash_save_"] .stButton > button:hover [data-testid="stIconMaterial"],
[class*="st-key-dash_rate_"] .stButton > button:hover [data-testid="stIconMaterial"] {
    transform: scale(1.08);
}

/* Interest shelf cards */
.interest-card {
    transition: transform .22s ease, box-shadow .22s ease, filter .22s ease;
}
.interest-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 18px rgba(80,55,35,.10);
    filter: brightness(1.018);
}

/* Genre identity cards */
.genre-badge {
    transition: transform .22s ease, box-shadow .22s ease, filter .22s ease;
}
.genre-badge:hover {
    transform: translateY(-3px);
    box-shadow: 0 9px 20px rgba(80,55,35,.11);
    filter: brightness(1.02);
}
.genre-badge-icon {
    transition: transform .24s ease;
}
.genre-badge:hover .genre-badge-icon {
    transform: scale(1.07);
}

/* Profile card: very subtle depth */
.profile-card,
.st-key-profile-panel {
    transition: transform .26s ease, box-shadow .26s ease;
}
.profile-card:hover,
.st-key-profile-panel:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 28px rgba(76,52,37,.09);
}
.avatar-ring,
.avatar-img {
    transition: transform .26s ease, box-shadow .26s ease;
}
.profile-card:hover .avatar-ring,
.profile-card:hover .avatar-img,
.st-key-profile-panel:hover .avatar-ring,
.st-key-profile-panel:hover .avatar-img {
    transform: scale(1.025);
}

/* Search and text inputs */
[data-testid="stTextInput"] input,
[data-testid="stDateInput"] input,
[data-baseweb="select"] > div,
[data-testid="stSelectbox"] > div > div {
    transition: border-color .18s ease, box-shadow .18s ease, background-color .18s ease;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stDateInput"] input:focus {
    box-shadow: 0 0 0 3px rgba(118,80,56,.10) !important;
}

/* Expander polish */
div[data-testid="stExpander"] {
    transition: box-shadow .2s ease, border-color .2s ease;
}
div[data-testid="stExpander"]:hover {
    border-color: #A98A70;
    box-shadow: 0 6px 14px rgba(80,55,35,.06);
}

/* Respect operating-system accessibility preference */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: .001ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: .001ms !important;
        scroll-behavior: auto !important;
    }
}

</style>
""", unsafe_allow_html=True)


def go(page):
    st.session_state.page = page
    st.rerun()


def logout():
    st.session_state.username = None
    st.session_state.draft = {}
    go("login")


def toggle_favorite(title):
    user = current_user()
    favorites = list(user.get("favorites", []))
    favorites.remove(title) if title in favorites else favorites.append(title)
    update_user(favorites=favorites)


def save_rating(title, rating):
    user = current_user()
    ratings = dict(user.get("ratings", {}))
    ratings[title] = int(rating)
    read_books = list(user.get("read_books", []))
    if title not in read_books:
        read_books.append(title)
    update_user(ratings=ratings, read_books=read_books)


def toggle_read(title):
    user = current_user()
    read_books = list(user.get("read_books", []))
    if title in read_books:
        read_books.remove(title)
        ratings = dict(user.get("ratings", {}))
        ratings.pop(title, None)
        update_user(read_books=read_books, ratings=ratings)
    else:
        read_books.append(title)
        update_user(read_books=read_books)


def open_book(title, return_page="recommendations"):
    st.session_state.selected_book = title
    st.session_state.book_return_page = return_page
    go("book_detail")


def book_brand():
    st.markdown('<div class="figma-rule"></div><div class="figma-title">MOON<br>RULER</div><div class="figma-rule"></div>', unsafe_allow_html=True)


def set_draft_choice(field, value):
    st.session_state.draft[field] = value
    st.rerun()


def choice_button(label, selected, key, help_text=None):
    clicked = st.button(label, key=key, type="primary" if selected else "secondary", use_container_width=True, help=help_text)
    return clicked


def page_login():
    with st.container(key="book-login"):
        left, right = st.columns([1, 1], gap="large")
        with left:
            st.markdown('<div class="figma-quote">Depends on<br>which side<br>of the sea<br>you’re from.</div>', unsafe_allow_html=True)
        with right:
            book_brand()
            sign_in, sign_up = st.tabs(["Sign in", "Create character"])
            with sign_in:
                with st.form("signin"):
                    name = st.text_input("Character Name", placeholder="@username")
                    password = st.text_input("Truth not told", type="password", placeholder="password")
                    submitted = st.form_submit_button("Open the book  ▷", use_container_width=True)
                if submitted:
                    users = load_users(); key = name.strip().lstrip("@").lower(); user = users.get(key)
                    if user and hash_password(password, user["salt"]) == user["hash"]:
                        st.session_state.username = key
                        go("dashboard" if user.get("profile_done") else "profile")
                    else:
                        st.error("That name and secret do not match any character.")
            with sign_up:
                with st.form("signup"):
                    new_name = st.text_input("Choose a character name", placeholder="@username")
                    new_pw = st.text_input("Choose your truth not told", type="password")
                    confirm = st.text_input("Repeat it", type="password")
                    created = st.form_submit_button("Begin my story  ▷", use_container_width=True)
                if created:
                    key = new_name.strip().lstrip("@").lower(); users = load_users()
                    if not re.fullmatch(r"[a-z0-9_.]{3,20}", key): st.error("Use 3–20 letters, numbers, dots, or underscores.")
                    elif key in users: st.error("That character name is already taken.")
                    elif len(new_pw) < 6: st.error("Your secret needs at least 6 characters.")
                    elif new_pw != confirm: st.error("The two secrets do not match.")
                    else:
                        salt = secrets.token_hex(8)
                        users[key] = {"salt": salt, "hash": hash_password(new_pw, salt), "ratings": {}, "favorites": [], "read_books": [], "genres": [], "profile_done": False, "role": "reader"}
                        save_users(users); st.session_state.username = key; st.session_state.draft = {}; go("profile")


def page_profile():
    draft = st.session_state.draft
    tier = draft.get("tier", "Page Turner")
    with st.container(key="book-profile"):
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown('<div class="figma-section">Your character</div>', unsafe_allow_html=True)
            name = st.text_input("Name", value=draft.get("name", ""), placeholder="Name", key="profile_name", label_visibility="collapsed")
            c1, c2 = st.columns(2)
            birthdate = c1.date_input("Birthdate", value=date.fromisoformat(draft["birthdate"]) if draft.get("birthdate") else None, min_value=date(1940,1,1), max_value=date.today(), format="YYYY/MM/DD", key="profile_birthdate", label_visibility="collapsed")
            gender = c2.selectbox("Gender", GENDERS, index=GENDERS.index(draft["gender"]) if draft.get("gender") in GENDERS else None, placeholder="Gender", key="profile_gender", label_visibility="collapsed")
            st.markdown('<div class="figma-small" style="margin-top:.9rem">Create the reader profile Moon Ruler will use to shape recommendations.</div>', unsafe_allow_html=True)
        with right:
            st.markdown('<div class="figma-section">What kind of reader are you?</div>', unsafe_allow_html=True)
            inner_left, tier_col, inner_right = st.columns([.08,.84,.08])
            with tier_col:
                with st.container(key="tier-stack"):
                    for t in TIERS:
                        label = f"{t.upper()}\n{TIERS[t]['blurb']}"
                        if choice_button(label, tier == t, f"tier_{t}"):
                            draft["tier"] = t
                            st.rerun()
        with st.container(key="profile-nav"):
            back, spacer, forward = st.columns([1.0, 2.7, 1.0])
            if back.button("◁  Back", use_container_width=True):
                logout()
            if forward.button("Next page  ▷", use_container_width=True):
                if not name.strip(): st.error("Every character needs a name.")
                elif not birthdate: st.error("Every character needs a birthdate.")
                elif not gender: st.error("Please select a gender.")
                else:
                    draft.update(name=name.strip(), birthdate=birthdate.isoformat(), gender=gender, tier=tier)
                    go("genres")


def page_genres():
    df, *_ = get_data()
    draft = st.session_state.draft
    draft.setdefault("ratings", {})
    tier = draft.get("tier", "Page Turner")
    max_genres = TIERS[tier]["max_genres"]
    scope = draft.get("scope", "Fiction")
    chosen = list(draft.get("genres", []))
    editing = bool(st.session_state.get("editing_preferences", False))

    # After onboarding, the chosen main scope is locked for this account.
    # Free Reader remains the special cross-category mode.
    if editing:
        saved_scope = current_user().get("scope", scope)
        scope = saved_scope
        draft["scope"] = saved_scope

    valid = genres_for_scope(df, scope)
    chosen = [g for g in chosen if g in valid]
    draft["genres"] = chosen

    with st.container(key="book-genres"):
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown('<div class="figma-section">Which side of the sea are you from?</div>', unsafe_allow_html=True)
            if editing:
                locked_label = "FREE READER" if scope == "Free Reader" else scope.upper()
                locked_icon = "⊞" if scope == "Free Reader" else ("✒" if scope == "Fiction" else "✎")
                st.markdown(
                    f'<div class="locked-scope-card"><div class="locked-scope-icon">{locked_icon}</div>'
                    f'<div>{html.escape(locked_label)}</div><small>Main category locked for this profile</small></div>',
                    unsafe_allow_html=True,
                )
            else:
                with st.container(key="scope-top"):
                    c1, c2 = st.columns(2, gap="small")
                    for col, s, label in [(c1,"Fiction","✒\nFICTION"),(c2,"Academics","✎\nACADEMICS")]:
                        with col:
                            if choice_button(label, scope == s, f"scope_{s}"):
                                draft["scope"] = s; draft["genres"] = []; st.rerun()
                with st.container(key="scope-free"):
                    if choice_button("⊞  FREE READER", scope == "Free Reader", "scope_Free Reader"):
                        draft["scope"] = "Free Reader"; draft["genres"] = []; st.rerun()

            if tier == "Bookworm":
                with st.expander("Add books you've already read"):
                    title = st.selectbox("Book", df["title"].tolist(), key="onboard_book")
                    stars = st.slider("Your rating", 1, 5, 4, key="onboard_stars")
                    if st.button("Add to my shelf", key="onboard_add"):
                        draft["ratings"][title] = stars
                        st.toast(f"Added {title} at {stars}★")

        with right:
            if max_genres == 0:
                st.markdown('<div class="figma-section">Start with discovery</div><div class="figma-small">New to the Shelf does not require genre choices yet. Moon Ruler will learn from what you save and rate.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="figma-section">Choose up to {max_genres} selected genre/s</div>', unsafe_allow_html=True)
                genres = genres_for_scope(df, scope)
                with st.container(key="genre-grid"):
                    cols = st.columns(3, gap="small")
                    for i, genre in enumerate(genres):
                        selected = genre in chosen
                        with cols[i % 3]:
                            label = f"{GENRE_ICONS.get(genre,'•')}\n{genre.upper()}"
                            if choice_button(label, selected, f"genre_btn_{genre}"):
                                if selected:
                                    chosen.remove(genre)
                                elif len(chosen) < max_genres:
                                    chosen.append(genre)
                                else:
                                    st.toast(f"You can choose up to {max_genres} genres.")
                                draft["genres"] = chosen
                                st.rerun()
                st.markdown(f'<div class="genre-count">{len(chosen)} of {max_genres} chosen</div>', unsafe_allow_html=True)

        with st.container(key="genres-nav"):
            back, spacer, finish = st.columns([1.0, 2.7, 1.0])
            if back.button("◁  Cancel" if editing else "◁  Previous", use_container_width=True):
                if editing:
                    st.session_state.editing_preferences = False
                    st.session_state.draft = {}
                    go("dashboard")
                else:
                    draft["scope"] = scope
                    go("profile")
            finish_label = "Save interests  ▷" if editing else "Enter library  ▷"
            if finish.button(finish_label, use_container_width=True):
                if editing:
                    update_user(genres=chosen)
                    st.session_state.editing_preferences = False
                    st.session_state.draft = {}
                    go("dashboard")
                else:
                    update_user(
                        name=draft["name"], birthdate=draft["birthdate"], gender=draft["gender"],
                        tier=tier, scope=scope, active_category="Academics" if scope == "Academics" else "Fiction",
                        genres=chosen, ratings=draft["ratings"], profile_done=True,
                    )
                    go("dashboard")


def mini_recommendation_row(row, reason, favorites, ratings, key_prefix, return_page="recommendations"):
    score = row.get("score", None)
    score_text = f"{int(round(float(score)*100))}% match" if score is not None and pd.notna(score) else row["genre"]
    card_key = re.sub(r"[^a-zA-Z0-9_-]", "_", f"{key_prefix}_{row.name}")
    with st.container(border=True, key=f"rec-book-card-{card_key}"):
        cover_col, info_col = st.columns([1.0, 5.8], gap="medium")
        with cover_col:
            st.image(get_book_cover(row["title"], row["author"]), use_container_width=True)
        with info_col:
            title_col, score_col = st.columns([4.5, 1.35], gap="small")
            with title_col:
                st.markdown(f'<div class="cover-title">{html.escape(row["title"])}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="cover-meta">{html.escape(row["author"])} · {GENRE_ICONS.get(row["genre"], "📚")} {html.escape(row["genre"])}</div>', unsafe_allow_html=True)
            with score_col:
                st.markdown(f'<div class="match-badge">{html.escape(score_text)}</div>', unsafe_allow_html=True)

            if return_page == "dashboard":
                # Dashboard uses Streamlit's Material Symbols (vector icons), not emoji/text glyphs.
                details_col, save_col, rate_col, spacer_col = st.columns([1.55, .78, .78, 2.9], gap="small")
                with details_col:
                    if st.button("Details  ▷", key=f"details_{card_key}", use_container_width=True):
                        open_book(row["title"], return_page)
                with save_col:
                    saved = row["title"] in favorites
                    save_icon = ":material/favorite:" if saved else ":material/favorite_border:"
                    if st.button("Save", icon=save_icon, key=f"dash_save_{card_key}", help="Remove from Saved Books" if saved else "Save for later"):
                        toggle_favorite(row["title"]); st.rerun()
                with rate_col:
                    rated = row["title"] in ratings
                    rate_icon = ":material/star:" if rated else ":material/star_border:"
                    rate_help = f"Your rating: {int(ratings[row['title']])}/5 — click to update" if rated else "Rate this book"
                    if st.button("Rate", icon=rate_icon, key=f"dash_rate_{card_key}", help=rate_help):
                        open_book(row["title"], return_page)
            else:
                # Full Recommendations page keeps descriptive labels.
                details_col, save_col, rate_col, spacer_col = st.columns([1.55, 1.25, 1.15, 2.3], gap="small")
                with details_col:
                    if st.button("Details  ▷", key=f"details_{card_key}", use_container_width=True):
                        open_book(row["title"], return_page)
                with save_col:
                    saved = row["title"] in favorites
                    if st.button("♥ Saved" if saved else "♡ Save", key=f"fav_{card_key}", help="Save or remove from Saved Books", use_container_width=True):
                        toggle_favorite(row["title"]); st.rerun()
                with rate_col:
                    rating_label = f"★ {int(ratings[row['title']])}/5" if row["title"] in ratings else "☆ Rate"
                    if st.button(rating_label, key=f"rate_{card_key}", help="Open this book to add or update your rating", use_container_width=True):
                        open_book(row["title"], return_page)


def logged_in_nav(active):
    user = current_user()
    labels = [("dashboard","Dashboard"),("interests","My Interests"),("recommendations","Recommendations"),("profile_page","Profile"),("about","How It Works")]
    if effective_role(st.session_state.username, user) == "admin":
        labels.append(("admin","Admin Panel"))
    cols = st.columns(len(labels))
    for c,(page,label) in zip(cols,labels):
        if c.button(label, key=f"nav_{page}", type="primary" if active == page else "secondary", use_container_width=True):
            go(page)


def current_category(user):
    if user.get("scope") in ("Fiction", "Academics"):
        return user["scope"]
    return user.get("active_category", "Fiction")


def prepare_interest_edit(user):
    st.session_state.draft = {
        "name": user.get("name", ""),
        "birthdate": user.get("birthdate"),
        "gender": user.get("gender"),
        "tier": user.get("tier", "Page Turner"),
        "scope": user.get("scope", "Fiction"),
        "genres": list(user.get("genres", [])),
        "ratings": dict(user.get("ratings", {})),
    }
    st.session_state.editing_preferences = True
    go("genres")


def _saved_book_card(row, favorites, ratings, key_prefix):
    with st.container(border=True, key=f"saved-card-{key_prefix}-{row.name}"):
        st.markdown(f'<div class="book-grid-title">{html.escape(row["title"])}</div>', unsafe_allow_html=True)
        st.image(get_book_cover(row["title"], row["author"]), use_container_width=True)
        x,h,star=st.columns(3)
        if x.button("✕", key=f"saved_remove_{key_prefix}_{row.name}", help="Remove from saved", use_container_width=True):
            toggle_favorite(row["title"]); st.rerun()
        if h.button("♥", key=f"saved_heart_{key_prefix}_{row.name}", help="Saved", use_container_width=True):
            toggle_favorite(row["title"]); st.rerun()
        if star.button("★" if row["title"] in ratings else "☆", key=f"saved_rate_{key_prefix}_{row.name}", help="Quick 5-star rating", use_container_width=True):
            save_rating(row["title"], 5); st.rerun()
        if st.button("Details", key=f"saved_details_{key_prefix}_{row.name}", use_container_width=True):
            open_book(row["title"], "interests")


def _history_book_card(row, rating, key_prefix):
    with st.container(border=True, key=f"history-card-{key_prefix}-{row.name}"):
        st.markdown(f'<div class="book-grid-title">{html.escape(row["title"])}</div>', unsafe_allow_html=True)
        st.image(get_book_cover(row["title"], row["author"]), use_container_width=True)
        st.markdown(f'<div class="book-grid-rating">{rating if rating is not None else "—"}★</div>', unsafe_allow_html=True)
        x,details=st.columns([1,2])
        if x.button("✕", key=f"history_remove_{key_prefix}_{row.name}", help="Remove from reading history", use_container_width=True):
            toggle_read(row["title"]); st.rerun()
        if details.button("Details", key=f"history_details_{key_prefix}_{row.name}", use_container_width=True):
            open_book(row["title"], "interests")


def page_interests():
    df, *_ = get_data(); user=current_user(); category=current_category(user)
    ratings=user.get("ratings",{}); favorites=user.get("favorites",[]); read_books=user.get("read_books",[])
    title_set=set(df["title"])
    with st.container(key="book-dashboard"):
        logged_in_nav("interests")
        st.markdown('<div class="page-title">My Interests</div><div class="page-sub">Your preferred genres, saved shelf, and reading history.</div>', unsafe_allow_html=True)
        st.markdown('### Preferred genres')
        genres=[g for g in user.get("genres",[]) if g in genres_for_scope(df, category)]
        if genres:
            cols=st.columns(min(5, len(genres)))
            for i,g in enumerate(genres):
                cols[i%len(cols)].markdown(render_genre_badge(g),unsafe_allow_html=True)
        else:
            st.info("No preferred genres selected yet.")
        if st.button("Edit preferred genres", key="interests_edit"):
            prepare_interest_edit(user)
        st.markdown('<div style="height:.5rem"></div>', unsafe_allow_html=True)
        saved_panel, history_panel = st.columns([1.04,1], gap="large")
        with saved_panel:
            st.markdown('<div class="interests-panel-title">Saved books</div>', unsafe_allow_html=True)
            saved_rows=[df[df["title"]==t].iloc[0] for t in favorites if t in title_set]
            if saved_rows:
                cols=st.columns(3)
                for i,row in enumerate(saved_rows):
                    with cols[i%3]:
                        _saved_book_card(row, favorites, ratings, "grid")
            else:
                st.caption("No saved books yet. Use ♡ on a recommendation to build this shelf.")
        with history_panel:
            st.markdown('<div class="interests-panel-title">Reading History & Ratings</div>', unsafe_allow_html=True)
            with st.expander("▹ Rate a book you've read / update a rating"):
                available = df[df["category"] == category]["title"].tolist()
                rate_title = st.selectbox("Book", available, key="interests_rate_title")
                current_value = int(ratings.get(rate_title, 4))
                rate_value = st.slider("Your rating", 1, 5, current_value, key="interests_rate_value")
                st.caption("1–2★ reduces similar recommendations, 3★ is neutral, and 4–5★ strengthens similar recommendations.")
                if st.button("Save rating", key="interests_save_rating"):
                    save_rating(rate_title, rate_value); st.success(f"Saved {rate_value}★ for {rate_title}."); st.rerun()
            history_rows=[df[df["title"]==t].iloc[0] for t in read_books if t in title_set]
            if history_rows:
                cols=st.columns(2)
                for i,row in enumerate(history_rows):
                    with cols[i%2]:
                        _history_book_card(row, ratings.get(row["title"]), "grid")
            else:
                st.caption("No reading history yet. Mark a book as read from its details page.")


def page_recommendations():
    df,X,feature_names,sim=get_data(); user=current_user(); category=current_category(user)
    engine=ContentRecommender(df,X,feature_names)
    taste=user_taste(df,X,feature_names,user); personalized=taste is not None
    ratings=user.get("ratings",{}); favorites=user.get("favorites",[]); read_titles=set(user.get("read_books",[])) | set(ratings)
    st.session_state.setdefault("rec_limits", {})
    with st.container(key="book-dashboard"):
        logged_in_nav("recommendations")
        st.markdown('<div class="page-title">Recommendations</div><div class="page-sub">Three books are shown first for each genre. Use “See more” to expand a shelf.</div>', unsafe_allow_html=True)

        # Presentation-friendly summary of the signals currently affecting ranking.
        m1,m2,m3,m4=st.columns(4)
        m1.metric("Mode", "Personalized" if personalized else "Discovery")
        m2.metric("Taste strength", f"{profile_strength(user)}%")
        m3.metric("Rated", len(ratings))
        m4.metric("Read excluded", len(read_titles))
        with st.expander("Recommendation diagnostics — useful during instructor review"):
            st.write(f"**Active category:** {category}")
            st.write(f"**Selected genres:** {', '.join(user.get('genres', [])) or 'None yet'}")
            st.write(f"**Saved books:** {len(favorites)} · **Read books:** {len(user.get('read_books', []))} · **Rated books:** {len(ratings)}")
            st.write("**Ranking method:** weighted category/genre/tag/author vectors → user taste vector → cosine similarity → filtering → author diversity.")
            st.caption("Feedback meaning: Save = mild interest · 4–5★ = positive · 3★ = neutral · 1–2★ = negative · Read = excluded from future candidates.")

        genres=[g for g in user.get("genres",[]) if g in genres_for_scope(df,category)]
        if not genres:
            genres=list(df[df["category"]==category]["genre"].value_counts().index[:3])
        for genre in genres:
            st.markdown(f'<div class="rec-title">{html.escape(genre)}</div>',unsafe_allow_html=True)
            limit=int(st.session_state.rec_limits.get(genre,3))
            picks=engine.recommend(taste,read_titles,category=category,genre=genre,top_n=limit) if personalized else engine.default_recommendations(read_titles,category=category,genre=genre,top_n=limit)
            for _,row in picks.iterrows():
                mini_recommendation_row(row, why_recommended(row,df,sim,user,personalized), favorites, ratings, f"full_{genre}", "recommendations")
            total_available=len(df[(df["category"]==category)&(df["genre"]==genre)&(~df["title"].isin(read_titles))])
            more_col,less_col,_=st.columns([1,1,4])
            if limit < total_available and more_col.button("See more", key=f"more_{genre}"):
                st.session_state.rec_limits[genre]=min(limit+3,total_available); st.rerun()
            if limit > 3 and less_col.button("Show less", key=f"less_{genre}"):
                st.session_state.rec_limits[genre]=3; st.rerun()
            st.markdown('<div class="genre-divider"></div>',unsafe_allow_html=True)


def page_profile_page():
    user=current_user(); category=current_category(user)
    with st.container(key="book-dashboard"):
        logged_in_nav("profile_page")
        st.markdown('<div class="page-title">Profile</div><div class="page-sub">Your reader identity and recommendation settings.</div>',unsafe_allow_html=True)
        left,right=st.columns([1.1,1.5],gap="large")
        with left:
            st.markdown(
                '<div class="profile-card">' + avatar_markup(user) +
                f'<div class="profile-name">{html.escape(user.get("name","Reader"))}</div>'
                f'<div class="profile-handle">@{html.escape(st.session_state.username)}</div>'
                f'<span class="profile-pill">{html.escape(user.get("gender","—"))}</span>'
                f'<span class="profile-pill">{age_from(user["birthdate"]) if user.get("birthdate") else "—"} years old</span>'
                f'<div class="profile-tier">{html.escape(user.get("tier","Reader"))}</div>'
                f'<div class="figma-small">Taste profile strength: {profile_strength(user)}%</div>'
                f'<div class="profile-rule"></div><div class="profile-scope-label">Main category</div><div class="profile-scope">{html.escape(user.get("scope","Fiction"))}</div></div>',
                unsafe_allow_html=True)
        with right:
            st.markdown('### Profile picture')
            presets=["🌙","📚","🦉","🐈","⭐","🌸","🪐","🧙"]
            pcols=st.columns(8)
            for i,preset in enumerate(presets):
                if pcols[i].button(preset, key=f"avatar_preset_{i}", use_container_width=True):
                    update_user(avatar_type="preset", avatar_value=preset); st.rerun()
            uploaded=st.file_uploader("Upload custom JPG or PNG", type=["jpg","jpeg","png"], key="profile_avatar_upload")
            if uploaded is not None and st.button("Use uploaded picture", key="profile_avatar_save"):
                try:
                    update_user(avatar_type="custom", avatar_value=image_file_to_data_uri(uploaded)); st.success("Profile picture updated."); st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
            st.caption("Custom pictures are stored with your local account data. Maximum file size: 2 MB.")
            st.markdown('### Account settings')
            st.write(f"**Name:** {user.get('name','—')}")
            st.write(f"**Reader tier:** {user.get('tier','—')}")
            st.write(f"**Main category:** {user.get('scope','—')}")
            st.write(f"**Selected genres:** {', '.join(user.get('genres',[])) or 'None'}")
            if user.get("scope") == "Free Reader":
                st.markdown('### Active library')
                c1,c2=st.columns(2)
                if c1.button("Fiction",type="primary" if category=="Fiction" else "secondary",use_container_width=True,key="profilepage_fiction"):
                    update_user(active_category="Fiction"); st.rerun()
                if c2.button("Academics",type="primary" if category=="Academics" else "secondary",use_container_width=True,key="profilepage_academics"):
                    update_user(active_category="Academics"); st.rerun()
            if st.button("Edit interests",use_container_width=True,key="profilepage_edit"):
                prepare_interest_edit(user)
            if st.button("Log out / Switch Account",use_container_width=True,key="profilepage_logout"):
                logout()


def page_book_detail():
    df,X,feature_names,sim=get_data(); user=current_user()
    title=st.session_state.get("selected_book")
    return_page=st.session_state.get("book_return_page","recommendations")
    with st.container(key="book-dashboard"):
        logged_in_nav("")
        if not title or title not in set(df["title"]):
            st.error("That book is no longer available in the catalog.")
            if st.button("Go back"):
                go(return_page)
            return
        row=df[df["title"]==title].iloc[0]
        row_idx=int(row.name)
        category=current_category(user)
        taste=user_taste(df,X,feature_names,user)
        match=None
        if taste is not None:
            from week11_similarity import cosine_to_vector
            match=float(cosine_to_vector(X,taste)[row_idx])
        topbar,_=st.columns([1,5])
        if topbar.button("◁ Back", key="book_detail_back"):
            go(return_page)
        hero_cover, hero_text = st.columns([0.85, 5.15], gap="medium")
        with hero_cover:
            st.image(get_book_cover(row["title"], row["author"]), use_container_width=True)
        with hero_text:
            st.markdown(f'<div class="page-title book-detail-title">{html.escape(row["title"])}</div><div class="page-sub">{html.escape(row["author"])} · {GENRE_ICONS.get(row["genre"], "📚")} {html.escape(row["genre"])}</div>',unsafe_allow_html=True)
        left,right=st.columns([1.25,1],gap="large")
        with left:
            st.markdown("### Book details")
            m1,m2,m3=st.columns(3)
            m1.metric("Category", row["category"])
            m2.metric("Catalog rating", f"{float(row['rating']):.1f}★")
            m3.metric("Your match", f"{max(0,int(round(match*100)))}%" if match is not None else "—")
            st.write("**Themes / tags:** " + ", ".join(row["tags"]))
            st.info(why_recommended(row,df,sim,user,personalized=taste is not None))
        with right:
            st.markdown("### Your actions")
            favorites=user.get("favorites",[])
            read_books=user.get("read_books",[])
            ratings=user.get("ratings",{})
            save_col, read_col = st.columns(2, gap="small")
            with save_col:
                if st.button("♥ Unsave" if title in favorites else "♡ Save",use_container_width=True,key="detail_save"):
                    toggle_favorite(title); st.rerun()
            with read_col:
                if st.button("✓ Unread" if title in read_books else "✓ Mark read",use_container_width=True,key="detail_read"):
                    toggle_read(title); st.rerun()
            current_rating=int(ratings.get(title,0))
            rating_value=st.radio("Your rating", options=[1,2,3,4,5], index=(current_rating-1 if current_rating else 3), horizontal=True, key="detail_rating", format_func=lambda x: f"{x}★")
            if st.button("Save / update rating",use_container_width=True,key="detail_save_rating"):
                save_rating(title,rating_value); st.success(f"Saved {rating_value}★") ; st.rerun()
            st.caption("1–2★ reduces similar recommendations, 3★ is neutral, and 4–5★ strengthens similar recommendations.")

        st.markdown("### Similar books")
        order=sim[row_idx].argsort()[::-1]
        shown=0
        cols=st.columns(3)
        for j in order:
            if j==row_idx: continue
            candidate=df.iloc[int(j)]
            if candidate["category"] != row["category"]: continue
            with cols[shown%3]:
                st.markdown(f'<div class="interest-card"><div class="interest-title">{html.escape(candidate["title"])}</div><div class="figma-small">{html.escape(candidate["author"])} · {html.escape(candidate["genre"])}</div><div class="figma-small">{int(round(float(sim[row_idx][j])*100))}% content similarity</div></div>',unsafe_allow_html=True)
                if st.button("Open",key=f"similar_{int(j)}",use_container_width=True):
                    st.session_state.selected_book=candidate["title"]; st.rerun()
            shown+=1
            if shown>=6: break


def page_about():
    with st.container(key="book-dashboard"):
        logged_in_nav("about")
        st.markdown('<div class="page-title">How Moon Ruler Works</div><div class="page-sub">Algorithm transparency and proposal alignment.</div>',unsafe_allow_html=True)
        how_tab, proposal_tab = st.tabs(["How It Works", "Proposal vs Final System"])

        with how_tab:
            st.markdown("### 1. Every book becomes a feature vector")
            st.write("Moon Ruler represents each book using **category, genre, tags, and author**. The current weights are: genre **3.0**, tags **2.0**, author **1.5**, and category **1.0**.")
            st.markdown("### 2. The reader builds a taste vector")
            st.write("Selected genres provide the cold-start signal. Saved books add a softer positive signal. Ratings are stronger: **4–5★ pull recommendations toward similar books, 3★ is neutral, and 1–2★ push the taste vector away from similar books**.")
            st.markdown("### 3. Cosine similarity ranks candidates")
            st.latex(r"\text{cosine}(A,B)=\frac{A\cdot B}{\|A\|\|B\|}")
            st.write("The app compares the user's taste vector with each candidate book. Higher cosine similarity means stronger content overlap.")
            st.markdown("### 4. Filtering and diversity make the shelf practical")
            st.write("Moon Ruler removes books already marked as read, respects the profile's locked main category, filters by shelf genre, then limits repeated authors so one writer does not dominate the results.")
            st.markdown("### 5. Object-oriented recommender structure")
            st.write("The core recommendation operations are grouped in a **ContentRecommender** class while NumPy handles vector math and Pandas manages the catalog. This matches the technical direction stated in the original proposal without changing the underlying cosine-similarity formula.")
            st.markdown("### Why content-based filtering?")
            st.write("It works offline/local-first with a project-sized dataset, is mathematically explainable, and does not require thousands of users. Collaborative filtering can be added later after enough real user activity exists.")

        with proposal_tab:
            st.markdown("### Original proposal → implemented system")
            alignment = pd.DataFrame([
                ["Works offline / without a paid recommendation service", "Local-first Python + Streamlit app with CSV/JSON storage", "Implemented"],
                ["User rates or marks books already read and liked", "Separate Read, Save, and 1–5★ Rating signals", "Enhanced"],
                ["Compare genre, author, and tags", "Weighted category, genre, tags, and author feature vectors", "Enhanced"],
                ["Use cosine similarity", "Cosine similarity ranks candidate books against the user taste vector", "Implemented"],
                ["Explain why each title was suggested", "Shared genre, author, tags, selected interests, and match percentage", "Implemented"],
                ["Week 13: test and refine recommendation quality", "Automated validation plus manual demo checklist and feedback refinement", "Implemented"],
                ["Use NumPy, Pandas, and OOP", "NumPy vector math, Pandas catalog handling, ContentRecommender class", "Implemented"],
            ], columns=["Proposed", "Final Moon Ruler", "Status"])
            st.dataframe(alignment, use_container_width=True, hide_index=True)
            st.success("The final system preserves the original content-based recommendation scope. New pages, admin tools, negative feedback, and book details are extensions rather than a change of project direction.")



def admin_read_books():
    return pd.read_csv(BOOKS_PATH, dtype=str).fillna("")


def admin_write_books(df):
    cols=["title","author","category","genre","tags","rating"]
    df=df[cols].copy()
    df.to_csv(BOOKS_PATH,index=False,encoding="utf-8-sig")
    get_data.clear()


def user_admin_rows(users):
    """Return non-sensitive account metadata for the admin table."""
    rows = []
    for username, account in sorted(users.items()):
        rows.append({
            "username": username,
            "name": account.get("name", "—"),
            "role": effective_role(username, account),
            "profile": "Complete" if account.get("profile_done") else "Incomplete",
            "tier": account.get("tier", "—"),
            "scope": account.get("scope", "—"),
            "genres": len(account.get("genres", [])),
            "read": len(account.get("read_books", [])),
            "rated": len(account.get("ratings", {})),
            "saved": len(account.get("favorites", [])),
        })
    return pd.DataFrame(rows)


def reset_user_profile(username):
    """Reset onboarding and reading preferences while preserving login credentials and role."""
    users = load_users()
    if username not in users:
        return False
    role = users[username].get("role", "reader")
    salt = users[username].get("salt")
    pw_hash = users[username].get("hash")
    users[username] = {
        "salt": salt,
        "hash": pw_hash,
        "ratings": {},
        "favorites": [],
        "read_books": [],
        "genres": [],
        "profile_done": False,
        "role": role,
        "avatar_type": users[username].get("avatar_type", "preset"),
        "avatar_value": users[username].get("avatar_value", "🌙"),
    }
    save_users(users)
    return True


def page_admin():
    user = current_user()
    if effective_role(st.session_state.username, user) != "admin":
        st.error("Admin access required.")
        return

    with st.container(key="book-dashboard"):
        logged_in_nav("admin")
        st.markdown(
            '<div class="page-title">Admin Panel</div>'
            '<div class="page-sub">Manage the book catalog and reader accounts.</div>',
            unsafe_allow_html=True,
        )

        overview_books = admin_read_books()
        overview_users = load_users()
        a1,a2,a3,a4 = st.columns(4)
        a1.metric("Books", len(overview_books))
        a2.metric("Users", len(overview_users))
        a3.metric("Fiction", int((overview_books["category"] == "Fiction").sum()))
        a4.metric("Academics", int((overview_books["category"] == "Academics").sum()))
        with st.expander("Dataset distribution"):
            dist = overview_books.groupby(["category","genre"]).size().reset_index(name="books")
            st.dataframe(dist, use_container_width=True, hide_index=True)

        books_tab, users_tab = st.tabs(["Book Management", "User Management"])

        # ------------------------------------------------------------------
        # Book management
        # ------------------------------------------------------------------
        with books_tab:
            books = admin_read_books()
            st.caption(
                f"Dataset currently contains {len(books)} books. Changes are written directly "
                "to data/books.csv and the recommender cache is refreshed."
            )
            add_tab, edit_tab, delete_tab = st.tabs(["Add book", "Edit book", "Delete book"])

            with add_tab:
                with st.form("admin_add_book"):
                    title = st.text_input("Title")
                    author = st.text_input("Author")
                    category = st.selectbox("Category", ["Fiction", "Academics"])
                    valid_genres = [g for g in genres_for_scope(load_books(), category)]
                    genre = st.selectbox("Genre", valid_genres)
                    tags = st.text_input("Tags", placeholder="tag-one|tag-two|tag-three")
                    rating = st.number_input("Rating", min_value=0.0, max_value=5.0, value=4.0, step=0.1)
                    submit = st.form_submit_button("Add book")
                if submit:
                    if not title.strip() or not author.strip() or not tags.strip():
                        st.error("Title, author, and tags are required.")
                    elif ((books["title"].str.casefold() == title.strip().casefold()) &
                          (books["author"].str.casefold() == author.strip().casefold())).any():
                        st.error("That title and author already exist.")
                    else:
                        new = pd.DataFrame(
                            [[title.strip(), author.strip(), category, genre, tags.strip().lower(), f"{rating:.1f}"]],
                            columns=books.columns,
                        )
                        admin_write_books(pd.concat([books, new], ignore_index=True))
                        st.success("Book added.")
                        st.rerun()

            with edit_tab:
                labels = [f"{r.title} — {r.author}" for r in books.itertuples()]
                choice = st.selectbox(
                    "Choose book", range(len(labels)), format_func=lambda i: labels[i], key="admin_edit_choice"
                )
                row = books.iloc[int(choice)]
                with st.form("admin_edit_book"):
                    title = st.text_input("Title", value=row["title"])
                    author = st.text_input("Author", value=row["author"])
                    category = st.selectbox(
                        "Category", ["Fiction", "Academics"],
                        index=["Fiction", "Academics"].index(row["category"])
                        if row["category"] in ["Fiction", "Academics"] else 0,
                    )
                    genre_options = genres_for_scope(load_books(), category)
                    genre = st.selectbox(
                        "Genre", genre_options,
                        index=genre_options.index(row["genre"]) if row["genre"] in genre_options else 0,
                    )
                    tags = st.text_input("Tags", value=row["tags"])
                    try:
                        rv = float(row["rating"])
                    except Exception:
                        rv = 4.0
                    rating = st.number_input("Rating", min_value=0.0, max_value=5.0, value=rv, step=0.1)
                    submit = st.form_submit_button("Save changes")
                if submit:
                    books.loc[int(choice), ["title", "author", "category", "genre", "tags", "rating"]] = [
                        title.strip(), author.strip(), category, genre, tags.strip().lower(), f"{rating:.1f}"
                    ]
                    admin_write_books(books)
                    st.success("Book updated.")
                    st.rerun()

            with delete_tab:
                labels = [f"{r.title} — {r.author}" for r in books.itertuples()]
                choice = st.selectbox(
                    "Choose book to delete", range(len(labels)), format_func=lambda i: labels[i],
                    key="admin_delete_choice"
                )
                row = books.iloc[int(choice)]
                st.warning(f"Delete **{row['title']}** by {row['author']}? This cannot be undone from the app.")
                confirm_book = st.checkbox("I understand this will permanently remove the book.", key="confirm_delete_book")
                if st.button(
                    "Delete book", type="primary", key="admin_delete_button", disabled=not confirm_book
                ):
                    books = books.drop(index=int(choice)).reset_index(drop=True)
                    admin_write_books(books)
                    st.success("Book deleted.")
                    st.rerun()

        # ------------------------------------------------------------------
        # User management
        # ------------------------------------------------------------------
        with users_tab:
            users = load_users()
            st.caption(
                f"{len(users)} account(s) found. Password hashes and salts are intentionally hidden from this panel."
            )

            query = st.text_input(
                "Search users", placeholder="Search username, display name, tier, or category",
                key="admin_user_search"
            ).strip().casefold()
            table = user_admin_rows(users)
            if query and not table.empty:
                mask = table.astype(str).apply(
                    lambda col: col.str.casefold().str.contains(query, regex=False)
                ).any(axis=1)
                table = table[mask]
            st.dataframe(table, use_container_width=True, hide_index=True)

            if not users:
                st.info("There are no user accounts to manage.")
                return

            usernames = sorted(users)
            selected = st.selectbox(
                "Select an account", usernames,
                format_func=lambda u: f"@{u} — {users[u].get('name', 'No profile yet')}",
                key="admin_user_choice",
            )
            account = users[selected]
            is_self = selected == st.session_state.username

            c1, c2, c3 = st.columns(3)
            c1.metric("Role", effective_role(selected, account).title())
            c2.metric("Rated books", len(account.get("ratings", {})))
            c3.metric("Saved books", len(account.get("favorites", [])))

            st.markdown(
                f"**Profile:** {'Complete' if account.get('profile_done') else 'Incomplete'}  |  "
                f"**Tier:** {account.get('tier', '—')}  |  "
                f"**Main category:** {account.get('scope', '—')}"
            )
            genres = account.get("genres", [])
            st.write("**Genres:** " + (", ".join(genres) if genres else "None"))

            if is_self:
                st.info("For safety, you cannot delete, reset, or demote the account you are currently using.")

            role_col, reset_col, delete_col = st.columns(3)

            # Promote / demote role
            with role_col:
                st.markdown("#### Account role")
                if effective_role(selected, account) == "admin":
                    if st.button(
                        "Demote to reader", use_container_width=True, key="admin_demote_user", disabled=is_self
                    ):
                        set_admin_role(selected, False)
                        st.success(f"@{selected} is now a reader.")
                        st.rerun()
                else:
                    if st.button(
                        "Promote to admin", use_container_width=True, key="admin_promote_user"
                    ):
                        set_admin_role(selected, True)
                        st.success(f"@{selected} is now an admin.")
                        st.rerun()

            # Reset profile/preferences
            with reset_col:
                st.markdown("#### Reset profile")
                st.caption("Keeps the login and role, but clears onboarding, genres, ratings, and saved books.")
                confirm_reset = st.checkbox(
                    "Confirm reset", key=f"confirm_reset_{selected}", disabled=is_self
                )
                if st.button(
                    "Reset account", use_container_width=True, key="admin_reset_user",
                    disabled=is_self or not confirm_reset
                ):
                    reset_user_profile(selected)
                    st.success(f"@{selected} was reset and will go through onboarding again.")
                    st.rerun()

            # Delete account
            with delete_col:
                st.markdown("#### Delete account")
                st.caption("Permanently removes this account from data/users.json.")
                confirm_text = st.text_input(
                    f"Type {selected} to confirm", key=f"delete_confirm_text_{selected}", disabled=is_self
                )
                can_delete = (not is_self) and confirm_text.strip() == selected
                if st.button(
                    "Delete account", type="primary", use_container_width=True,
                    key="admin_delete_user", disabled=not can_delete
                ):
                    latest = load_users()
                    if selected in latest:
                        del latest[selected]
                        save_users(latest)
                        st.success(f"@{selected} was deleted.")
                        st.rerun()

def dashboard_profile_panel(user, category):
    with st.container(border=True, key="profile-panel"):
        profile_html = (
            '<div class="profile-inner">' + avatar_markup(user) +
            f'<div class="profile-name">{html.escape(user.get("name","Reader"))}</div>'
            f'<div class="profile-handle">@{html.escape(st.session_state.username)}</div>'
            '<div>'
            f'<span class="profile-pill">{html.escape(user.get("gender","—"))}</span>'
            f'<span class="profile-pill">{age_from(user["birthdate"]) if user.get("birthdate") else "—"} years old</span></div>'
            f'<div class="profile-tier">{html.escape(user.get("tier","Reader"))}</div>'
            f'<div class="figma-small">Taste profile strength: {profile_strength(user)}%</div>'
            '<div class="profile-rule"></div>'
            f'<div class="profile-scope-label">Main category</div><div class="profile-scope">{html.escape(user.get("scope","Fiction"))}</div></div>'
        )
        st.markdown(profile_html, unsafe_allow_html=True)
        if st.button("Open profile  ▷",use_container_width=True,key="dash_open_profile"):
            go("profile_page")


def page_dashboard():
    df,X,feature_names,sim=get_data(); user=current_user(); category=current_category(user)
    if user.get("scope") in ("Fiction","Academics") and user.get("active_category") != category:
        update_user(active_category=category)
    taste=user_taste(df,X,feature_names,user); personalized=taste is not None
    ratings=user.get("ratings",{}); favorites=user.get("favorites",[]); read_titles=set(user.get("read_books",[])) | set(ratings)
    with st.container(key="book-dashboard"):
        logged_in_nav("dashboard")
        search=st.text_input("Search",placeholder="⌕  Search Book / Author / Genre",label_visibility="collapsed",key="dash_search")
        if search.strip():
            found=find_books(df,search).head(20)
            st.markdown('<div class="figma-section">Search Results</div>',unsafe_allow_html=True)
            if found.empty: st.info("No books matched that search.")
            for _,row in found.iterrows():
                a,b=st.columns([6,1])
                a.markdown(f"**{row['title']}** — {row['author']} · {row['genre']}")
                if b.button("Details", key=f"search_details_{row.name}"):
                    open_book(row["title"], "dashboard")
            return
        left,middle,right=st.columns([1.05,1.45,1.05],gap="large")
        with left:
            if st.button("My Interests  ▷",key="dash_open_interests",use_container_width=True): go("interests")
            interests=[g for g in user.get("genres",[]) if g in genres_for_scope(df,category)]
            if not interests: interests=list(df[df["category"]==category]["genre"].value_counts().index[:3])
            for genre in interests[:4]:
                mine=df[(df["genre"]==genre)&(df["title"].isin(set(ratings)|set(favorites)))]
                lines="".join(f'<div class="interest-line">{("♥" if t in favorites else "★"+str(ratings.get(t,"")))} {html.escape(t)}</div>' for t in mine["title"].head(3)) or '<div class="interest-line">Nothing on this shelf yet</div>'
                st.markdown(f'<div class="interest-card"><div class="interest-title">{html.escape(genre)}</div>{lines}</div>',unsafe_allow_html=True)
        with middle:
            if st.button("Our Recommendations  ▷",key="dash_open_recommendations",use_container_width=True): go("recommendations")
            shown=[g for g in user.get("genres",[]) if g in genres_for_scope(df,category)]
            if not shown: shown=list(df[df["category"]==category]["genre"].value_counts().index[:3])
            engine=ContentRecommender(df,X,feature_names)
            for genre in shown[:3]:
                st.markdown(f'<div class="rec-title">{html.escape(genre)}</div>',unsafe_allow_html=True)
                picks=engine.recommend(taste,read_titles,category=category,genre=genre,top_n=3) if personalized else engine.default_recommendations(read_titles,category=category,genre=genre,top_n=3)
                for _,row in picks.iterrows(): mini_recommendation_row(row,why_recommended(row,df,sim,user,personalized),favorites,ratings,f"dash_{genre}", "dashboard")
                st.markdown('<div class="genre-divider"></div>',unsafe_allow_html=True)
        with right:
            dashboard_profile_panel(user,category)


st.session_state.setdefault("page", "login")
st.session_state.setdefault("username", None)
st.session_state.setdefault("draft", {})
st.session_state.setdefault("editing_preferences", False)
st.session_state.setdefault("rec_limits", {})
st.session_state.setdefault("selected_book", None)
st.session_state.setdefault("book_return_page", "recommendations")
inject_css()
page = st.session_state.page
if page != "login" and (not st.session_state.username or st.session_state.username not in load_users()):
    page = "login"
ROUTES={"login":page_login,"profile":page_profile,"genres":page_genres,"dashboard":page_dashboard,"interests":page_interests,"recommendations":page_recommendations,"profile_page":page_profile_page,"book_detail":page_book_detail,"about":page_about,"admin":page_admin}
ROUTES[page]()
