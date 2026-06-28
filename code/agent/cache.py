"""Tiny on-disk response cache.

Keyed by a hash of (provider, model, thinking budget, full text prompt, and the
bytes of every image). Identical re-runs cost nothing and are instant, which
matters for iterating on the sample set and for safe retries.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Optional

import config


def make_key(provider: str, model: str, thinking: int, text_prompt: str,
             image_bytes: List[bytes]) -> str:
    h = hashlib.sha256()
    h.update(provider.encode())
    h.update(model.encode())
    h.update(str(thinking).encode())
    h.update(text_prompt.encode("utf-8"))
    for b in image_bytes:
        h.update(hashlib.sha256(b).digest())
    return h.hexdigest()


def _path_for(key: str) -> Path:
    return config.CACHE_DIR / f"{key}.json"


def get(key: str) -> Optional[dict]:
    if not config.USE_CACHE:
        return None
    p = _path_for(key)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def put(key: str, value: dict) -> None:
    if not config.USE_CACHE:
        return
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _path_for(key).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
