"""Persistently promote or demote an existing Moon Ruler account.

Usage:
  python grant_admin.py username
  python grant_admin.py username --remove

Admin usernames are stored in data/admins.json as well as data/users.json,
so the role survives normal app restarts and profile resets.
"""
import argparse, json
from pathlib import Path

BASE_DIR = Path(__file__).parent
USERS_PATH = BASE_DIR / "data" / "users.json"
ADMINS_PATH = BASE_DIR / "data" / "admins.json"
USER_ADMIN_REGISTRY_PATH = Path.home() / ".moon_ruler" / "admins.json"

parser = argparse.ArgumentParser()
parser.add_argument("username")
parser.add_argument("--remove", action="store_true")
args = parser.parse_args()
username = args.username.strip().lstrip("@").lower()

if not USERS_PATH.exists():
    raise SystemExit("users.json does not exist yet. Create the account in the app first.")
users = json.loads(USERS_PATH.read_text(encoding="utf-8"))
if username not in users:
    raise SystemExit(f"Account @{username} was not found. Create it in the app first.")

admins = set()
for registry in (ADMINS_PATH, USER_ADMIN_REGISTRY_PATH):
    if registry.exists():
        try:
            raw = json.loads(registry.read_text(encoding="utf-8"))
            admins |= set(raw.get("admins", raw if isinstance(raw, list) else []))
        except Exception:
            pass

if args.remove:
    admins.discard(username)
    users[username]["role"] = "reader"
else:
    admins.add(username)
    users[username]["role"] = "admin"

USERS_PATH.write_text(json.dumps(users, indent=2, ensure_ascii=False), encoding="utf-8")
payload = json.dumps({"admins": sorted(admins)}, indent=2, ensure_ascii=False)
ADMINS_PATH.parent.mkdir(parents=True, exist_ok=True)
ADMINS_PATH.write_text(payload, encoding="utf-8")
try:
    USER_ADMIN_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    USER_ADMIN_REGISTRY_PATH.write_text(payload, encoding="utf-8")
except OSError:
    pass
print(f"@{username} is now a {users[username]['role']}. Admin status is stored in the project and in ~/.moon_ruler/admins.json for local upgrade persistence.")
