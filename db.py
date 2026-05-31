"""Supabase profiles CRUD.

Uses the Supabase REST API (supabase-py) with the service role key so RLS
is bypassed and all rows are accessible from the backend.

Table schema (run once in Supabase SQL editor):

    create table if not exists profiles (
        name            text primary key,
        university      text,
        nationality     text,
        visa_status     text,
        field_of_study  text,
        post_study_plan text,
        time_in_usa     text,
        updated_at      timestamptz default now()
    );
"""
from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

_TABLE = "profiles"
_client: Client | None = None


def _get_client() -> Client:
    global _client
    if _client is None:
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_SERVICE_KEY"]
        _client = create_client(url, key)
    return _client


def get_profile(name: str) -> Optional[dict]:
    """Fetch profile by name (case-insensitive). Returns None if not found."""
    response = (
        _get_client()
        .table(_TABLE)
        .select("*")
        .eq("name", name.strip().lower())
        .maybe_single()
        .execute()
    )
    return response.data


def upsert_profile(profile: dict) -> None:
    """Insert or update a profile. Keyed by lowercased name."""
    row = {k: v for k, v in profile.items()}
    row["name"] = row["name"].strip().lower()
    (
        _get_client()
        .table(_TABLE)
        .upsert(row, on_conflict="name")
        .execute()
    )
