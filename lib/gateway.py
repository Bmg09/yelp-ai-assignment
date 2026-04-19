import json
import re
from typing import TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from . import cache as cache_mod
from .config import ACTIVE_CLASSIFIER, GATEWAY_URL, JUDGE, gateway_key

T = TypeVar("T", bound=BaseModel)

_client: AsyncOpenAI | None = None

_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]+?)\s*```", re.IGNORECASE)


def client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=gateway_key(), base_url=GATEWAY_URL, max_retries=2)
    return _client


def _extract_json(text: str) -> str | None:
    m = _JSON_FENCE.search(text)
    if m:
        return m.group(1).strip()
    text = text.strip()
    first = min((i for i in (text.find("{"), text.find("[")) if i != -1), default=-1)
    last = max(text.rfind("}"), text.rfind("]"))
    if first == -1 or last == -1 or last <= first:
        return None
    return text[first : last + 1]


async def _call_openai_parse(model: str, schema: type[T], system: str, prompt: str, temperature: float):
    return await client().chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        response_format=schema,
        temperature=temperature,
        extra_body={"providerOptions": {"gateway": {"caching": "auto"}}},
    )


async def _call_generic_json(model: str, schema: type[T], system: str, prompt: str, temperature: float):
    schema_hint = json.dumps(schema.model_json_schema(), indent=2)
    sys_with_schema = (
        f"{system}\n\nReturn ONLY a single JSON object matching this schema. No prose, no markdown fences.\n\nSchema:\n{schema_hint}"
    )
    return await client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_with_schema},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        extra_body={"providerOptions": {"gateway": {"caching": "auto"}}},
    )


async def classify_json(
    schema: type[T],
    system: str,
    prompt: str,
    *,
    model: str = ACTIVE_CLASSIFIER,
    temperature: float = 0.0,
    cache: bool = True,
) -> tuple[T | None, dict]:
    schema_json = json.dumps(schema.model_json_schema(), sort_keys=True)
    use_cache = cache and temperature == 0
    key = cache_mod.key_for(model, system, prompt, schema_json, temperature) if use_cache else ""

    if use_cache:
        hit = cache_mod.get(key)
        if hit is not None:
            return schema.model_validate(hit["object"]), {
                "cached": True,
                "usage": hit.get("usage"),
                "model": model,
            }

    is_openai = model.startswith("openai/")

    try:
        if is_openai:
            resp = await _call_openai_parse(model, schema, system, prompt, temperature)
            msg = resp.choices[0].message
            if msg.refusal:
                return None, {"cached": False, "refusal": msg.refusal, "model": model}
            obj = msg.parsed
            raw_text = msg.content
        else:
            resp = await _call_generic_json(model, schema, system, prompt, temperature)
            raw_text = resp.choices[0].message.content or ""
            payload = _extract_json(raw_text)
            if payload is None:
                return None, {"cached": False, "error": "no_json_found", "raw": raw_text[:200], "model": model}
            try:
                obj = schema.model_validate_json(payload)
            except ValidationError as ve:
                return None, {"cached": False, "error": f"validation: {str(ve)[:150]}", "raw": payload[:200], "model": model}
    except Exception as e:
        return None, {"cached": False, "error": str(e)[:200], "model": model}

    if obj is None:
        return None, {"cached": False, "error": "no parsed object", "raw": (raw_text or "")[:200], "model": model}

    usage = resp.usage.model_dump() if resp.usage else None
    if use_cache:
        cache_mod.put(key, {"object": obj.model_dump(), "usage": usage})

    return obj, {"cached": False, "usage": usage, "model": model}


async def judge_score(schema: type[T], system: str, prompt: str) -> tuple[T | None, dict]:
    return await classify_json(schema, system, prompt, model=JUDGE)
