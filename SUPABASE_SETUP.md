# Moon Ruler — Persistent Account Setup

This build keeps the existing local JSON storage for development, but automatically switches to Supabase when the three Supabase secrets are configured in Streamlit Community Cloud.

## 1. Create a Supabase project
Go to Supabase and create a new project. Wait for the database to finish provisioning.

## 2. Create the Moon Ruler table
Open **SQL Editor**, paste the contents of `SUPABASE_SETUP.sql`, and click **Run**.

## 3. Get the connection values
In Supabase, open the project's **Connect / API Keys** area and copy:
- Project URL
- A server-side **secret key** (`sb_secret_...`)

Do not put the secret key in GitHub.

## 4. Add Streamlit Secrets
Open your deployed app -> **Manage app -> Settings -> Secrets** and add:

```toml
admin_usernames = ["your_username"]
supabase_url = "https://YOUR_PROJECT_REF.supabase.co"
supabase_secret_key = "sb_secret_REPLACE_ME"
supabase_table = "moon_ruler_users"
```

Save, then reboot the Streamlit app.

## 5. Create your Moon Ruler account once more
Accounts that existed only in Streamlit's temporary filesystem cannot be recovered automatically after a redeploy. Create/sign up once after enabling Supabase. From then on, accounts, profiles, selected genres, saved books, reading history, ratings, uploaded avatar data, and user roles stored in the user record persist across app updates.

## Local development
If the Supabase secrets are absent, Moon Ruler continues using `data/users.json` exactly as before.
