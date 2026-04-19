import hashlib
import json
from pathlib import Path

from diskcache import Cache

from .config import CACHE_DIR, CACHE_ENABLED

_cache: Cache | None = None


def cache() -> Cache:
    global _cache
    if _cache is None:
        Path(CACHE_DIR).mkdir(exist_ok=True)
        _cache = Cache(CACHE_DIR)
    return _cache


def key_for(model: str, system: str, prompt: str, schema_json: str, temperature: float) -> str:
    payload = json.dumps(
        {"model": model, "system": system, "prompt": prompt, "schema": schema_json, "temp": temperature},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def get(key: str):
    if not CACHE_ENABLED:
        return None
    return cache().get(key)


def put(key: str, value) -> None:
    if not CACHE_ENABLED:
        return
    cache()[key] = value
