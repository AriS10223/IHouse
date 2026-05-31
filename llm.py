"""Groq LLM client with W&B Weave auto-tracing.

IMPORTANT: weave.init() must be called in main.py BEFORE this module
is used to make API calls.  The lazy _client singleton ensures the
OpenAI client is created after Weave has patched the SDK.
"""
from __future__ import annotations

import os
import weave
from openai import OpenAI

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Return (or lazily create) the Groq-backed OpenAI client.

    Creating after weave.init() ensures Weave's auto-patch is in place,
    so every chat.completions.create() call is traced automatically.
    """
    global _client
    if _client is None:
        _client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ["GROQ_API_KEY"],
        )
    return _client


@weave.op(name="llm_chat")
def chat(
    system: str,
    user: str,
    model: str | None = None,
    max_tokens: int | None = None,
) -> str:
    """Make a single chat completion call and return the text response.

    Decorated with @weave.op so every call is a named trace node with
    token counts and latency recorded automatically.
    """
    from config import FAST_MODEL, MAX_TOKENS as DEFAULT_MAX_TOKENS

    model = model or FAST_MODEL
    max_tokens = max_tokens or DEFAULT_MAX_TOKENS

    response = get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return response.choices[0].message.content or ""
