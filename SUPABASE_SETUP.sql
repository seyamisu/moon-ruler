-- Moon Ruler persistent account storage
-- Run this once in Supabase -> SQL Editor.

create table if not exists public.moon_ruler_users (
    username text primary key,
    data jsonb not null default '{}'::jsonb,
    updated_at timestamptz not null default timezone('utc', now())
);

-- The app uses a server-side Supabase secret key stored only in Streamlit Secrets.
-- Do not expose this table to public browser clients.
alter table public.moon_ruler_users enable row level security;
revoke all on table public.moon_ruler_users from anon, authenticated;
grant all on table public.moon_ruler_users to service_role;
