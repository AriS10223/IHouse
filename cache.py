"""Redis profile cache (Upstash HTTP API).

Read path:  get_profile(name) → Redis HIT → return
                               → MISS    → Supabase → populate Redis → return
Write path: set_profile(profile) → write Supabase → write Redis (write-through)

Profile TTL: 24 hours. Invalidated immediately on any write.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from dotenv import load_dotenv
from upstash_redis import Redis

import db

load_dotenv()

_PROFILE_TTL = 86_400  # 24 hours
_redis: Redis | None = None


def _get_redis() -> Redis:
    global _redis
    if _redis is None:
        _redis = Redis(
            url=os.environ["UPSTASH_REDIS_URL"],
            token=os.environ["UPSTASH_REDIS_TOKEN"],
        )
    return _redis


def _profile_key(name: str) -> str:
    return f"profile:{name.strip().lower()}"


def get_profile(name: str) -> Optional[dict]:
    """Return profile from Redis if cached, otherwise fetch from Supabase and cache it."""
    key = _profile_key(name)
    try:
        cached = _get_redis().get(key)
        if cached is not None:
            return json.loads(cached)
    except Exception as exc:
        print(f"[cache] Redis read failed, falling back to Supabase: {exc}")

    profile = db.get_profile(name)
    if profile:
        try:
            _get_redis().set(key, json.dumps(profile), ex=_PROFILE_TTL)
        except Exception as exc:
            print(f"[cache] Redis write failed (non-fatal): {exc}")
    return profile


def set_profile(profile: dict) -> None:
    """Write profile to Supabase then update Redis (write-through)."""
    db.upsert_profile(profile)
    key = _profile_key(profile["name"])
    try:
        _get_redis().set(key, json.dumps(profile), ex=_PROFILE_TTL)
    except Exception as exc:
        print(f"[cache] Redis write failed (non-fatal): {exc}")
