"""Pluggable vision-language model client.

One entry point, `analyze(...)`, returns a parsed dict + token usage. Providers
(gemini / anthropic / openai) are swappable via EVIDENCE_PROVIDER so the
evaluation can compare configurations on the same data.

The Gemini path uses native structured output (response_schema) + extended
thinking. The Anthropic/OpenAI paths use JSON-mode prompting and tolerant
parsing, which keeps the system portable for the model-comparison requirement.
"""
from __future__ import annotations

import json
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import config
from agent import cache
from agent.data import LoadedImage
from agent.schema import ReviewResponse


@dataclass
class Usage:
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    thoughts_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.calls += other.calls
        self.cache_hits += other.cache_hits
        self.prompt_tokens += other.prompt_tokens
        self.output_tokens += other.output_tokens
        self.thoughts_tokens += other.thoughts_tokens
        self.total_tokens += other.total_tokens


@dataclass
class VlmResult:
    data: dict
    usage: Usage = field(default_factory=Usage)
    raw_text: str = ""
    model: str = ""


# ---------------------------------------------------------------- rate limiting
class _RateLimiter:
    """Spaces request *starts* at least `min_interval` apart, thread-safe.

    With REQUESTS_PER_MINUTE=5 this enforces a 12s gap so concurrent workers
    never burst past a strict free-tier RPM cap.
    """

    def __init__(self, rpm: float):
        self.min_interval = 60.0 / rpm if rpm > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.min_interval
            delay = start - now
        if delay > 0:
            time.sleep(delay)


_limiter = _RateLimiter(config.REQUESTS_PER_MINUTE)


def _parse_retry_delay(msg: str) -> Optional[float]:
    """Pull the server-suggested wait (seconds) out of a 429 message."""
    for pat in (r"retry in ([0-9.]+)s", r"retryDelay'?:?\s*'?([0-9.]+)s"):
        m = re.search(pat, msg)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None


# ----------------------------------------------------- multi-model failover
# Models whose daily free-tier quota is exhausted this run -> skip them.
_exhausted_models: set = set()
_models_lock = threading.Lock()
_model_call_counts: Counter = Counter()  # which model produced each live result


def _is_daily_quota_error(msg: str) -> bool:
    m = msg.lower().replace("_", "").replace("-", "")
    return "perday" in m  # e.g. GenerateRequestsPerDayPerProjectPerModel-FreeTier


def model_usage_summary() -> dict:
    return dict(_model_call_counts)


# --------------------------------------------------------------- JSON helpers
def _extract_json(text: str) -> dict:
    """Best-effort: parse strict JSON, else the first balanced {...} block."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text).rstrip("`").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except Exception:
                        break
    raise ValueError("Model did not return parseable JSON")


# ------------------------------------------------------------------ public API
def analyze(system: str, text_prompt: str, images: List[LoadedImage]) -> VlmResult:
    image_bytes = [img.data for img in images if img.exists]
    candidates = config.MODEL_POOL or [config.MODEL]
    base = system + "\n" + text_prompt

    # Cache check across all candidate models (results are model-agnostic in
    # shape, so a row cached under any model is reused).
    for m in candidates:
        key = cache.make_key(config.PROVIDER, m, config.THINKING_BUDGET, base, image_bytes)
        cached = cache.get(key)
        if cached is not None:
            return VlmResult(data=cached["data"], usage=Usage(cache_hits=1),
                             raw_text=cached.get("raw_text", ""), model=m)

    # Live call, rolling to the next model when one's daily quota is exhausted.
    last_err: Optional[Exception] = None
    for m in candidates:
        with _models_lock:
            if m in _exhausted_models:
                continue
        try:
            result = _analyze_with_retry(system, text_prompt, images, m)
        except Exception as e:  # noqa: BLE001
            last_err = e
            if _is_daily_quota_error(str(e)):
                with _models_lock:
                    _exhausted_models.add(m)
                continue  # try the next model in the pool
            raise
        key = cache.make_key(config.PROVIDER, m, config.THINKING_BUDGET, base, image_bytes)
        cache.put(key, {"data": result.data, "raw_text": result.raw_text})
        with _models_lock:
            _model_call_counts[m] += 1
        result.model = m
        return result
    raise RuntimeError(f"All candidate models exhausted/failed: {last_err}")


def analyze_model(system: str, text_prompt: str, images: List[LoadedImage],
                  model: str) -> VlmResult:
    """Judge with ONE specific model (used by self-consistency voting). Cached."""
    image_bytes = [img.data for img in images if img.exists]
    base = system + "\n" + text_prompt
    key = cache.make_key(config.PROVIDER, model, config.THINKING_BUDGET, base, image_bytes)
    cached = cache.get(key)
    if cached is not None:
        return VlmResult(data=cached["data"], usage=Usage(cache_hits=1),
                         raw_text=cached.get("raw_text", ""), model=model)
    result = _analyze_with_retry(system, text_prompt, images, model)
    cache.put(key, {"data": result.data, "raw_text": result.raw_text})
    with _models_lock:
        _model_call_counts[model] += 1
    result.model = model
    return result


def _analyze_with_retry(system: str, text_prompt: str, images: List[LoadedImage],
                        model: str) -> VlmResult:
    last_err: Optional[Exception] = None
    for attempt in range(config.MAX_RETRIES):
        _limiter.acquire()  # pace request starts under the RPM cap
        try:
            if config.PROVIDER == "gemini":
                return _analyze_gemini(system, text_prompt, images, model)
            if config.PROVIDER == "anthropic":
                return _analyze_anthropic(system, text_prompt, images, model)
            if config.PROVIDER == "openai":
                return _analyze_openai(system, text_prompt, images, model)
            raise ValueError(f"Unknown provider: {config.PROVIDER}")
        except Exception as e:  # noqa: BLE001 - retry transient API/rate errors
            last_err = e
            # A daily-quota error won't clear by retrying; surface it so the
            # caller can switch models immediately.
            if _is_daily_quota_error(str(e)):
                raise
            msg = str(e).lower()
            transient = any(t in msg for t in ("429", "rate", "quota", "overloaded",
                                               "503", "500", "timeout", "deadline", "unavailable"))
            if attempt == config.MAX_RETRIES - 1 or not transient:
                break
            # Honor the server's suggested delay, else exponential backoff.
            server_delay = _parse_retry_delay(str(e))
            backoff = config.RETRY_BASE_DELAY * (2 ** attempt)
            time.sleep(max(server_delay or 0.0, backoff) + 0.5)
    raise RuntimeError(f"VLM call failed after retries: {last_err}")


# --------------------------------------------------------------------- Gemini
_gemini_clients: dict = {}
_exhausted_model_key: set = set()  # (model, key) pairs whose daily quota is gone


def _get_gemini_client(key: str):
    if key not in _gemini_clients:
        from google import genai
        _gemini_clients[key] = genai.Client(api_key=key)
    return _gemini_clients[key]


def _analyze_gemini(system: str, text_prompt: str, images: List[LoadedImage],
                    model: str) -> VlmResult:
    from google.genai import types

    keys = config.gemini_keys()
    if not keys:
        raise RuntimeError("No Gemini API key set (GEMINI_API_KEY / EVIDENCE_GEMINI_KEYS)")

    parts = []
    for img in images:
        if img.exists:
            parts.append(types.Part.from_bytes(data=img.data, mime_type=img.mime_type))
    parts.append(types.Part(text=text_prompt))

    cfg = types.GenerateContentConfig(
        system_instruction=system,
        temperature=config.TEMPERATURE,
        max_output_tokens=config.MAX_OUTPUT_TOKENS,
        response_mime_type="application/json",
        response_schema=ReviewResponse,
    )
    if config.THINKING_BUDGET != 0:
        cfg.thinking_config = types.ThinkingConfig(thinking_budget=config.THINKING_BUDGET)

    # Rotate keys: each project has its own daily quota, so a daily-quota 429 on
    # one key just moves us to the next. Only when ALL keys are exhausted for this
    # model do we surface the daily error (so the caller can fail over to another model).
    last_daily: Optional[Exception] = None
    for key in keys:
        with _models_lock:
            if (model, key) in _exhausted_model_key:
                continue
        try:
            resp = _get_gemini_client(key).models.generate_content(
                model=model, contents=parts, config=cfg)
        except Exception as e:  # noqa: BLE001
            if _is_daily_quota_error(str(e)):
                with _models_lock:
                    _exhausted_model_key.add((model, key))
                last_daily = e
                continue
            raise  # transient/other -> handled by _analyze_with_retry
        parsed = getattr(resp, "parsed", None)
        if parsed is not None:
            data = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)
        else:
            data = _extract_json(resp.text)
        um = resp.usage_metadata
        usage = Usage(
            calls=1,
            prompt_tokens=getattr(um, "prompt_token_count", 0) or 0,
            output_tokens=getattr(um, "candidates_token_count", 0) or 0,
            thoughts_tokens=getattr(um, "thoughts_token_count", 0) or 0,
            total_tokens=getattr(um, "total_token_count", 0) or 0,
        )
        return VlmResult(data=data, usage=usage, raw_text=resp.text or "")

    raise last_daily or RuntimeError(f"All Gemini keys exhausted for {model}")


# ------------------------------------------------------------------ Anthropic
def _analyze_anthropic(system: str, text_prompt: str, images: List[LoadedImage],
                       model: str) -> VlmResult:
    import base64

    import anthropic

    key = config.api_key_for("anthropic")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    client = anthropic.Anthropic(api_key=key)

    content = []
    for img in images:
        if img.exists:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": img.mime_type,
                           "data": base64.b64encode(img.data).decode()},
            })
    schema_hint = ("\n\nReturn ONLY a JSON object with keys: extracted_claim, claimed_issue_type, "
                   "claimed_object_part, per_image[], evidence_standard_met, evidence_standard_met_reason, "
                   "issue_type, object_part, claim_status, claim_status_justification, "
                   "supporting_image_ids[], valid_image, severity, risk_flags[].")
    content.append({"type": "text", "text": text_prompt + schema_hint})

    msg = client.messages.create(
        model=model, max_tokens=config.MAX_OUTPUT_TOKENS,
        system=system, temperature=config.TEMPERATURE,
        messages=[{"role": "user", "content": content}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    usage = Usage(calls=1,
                  prompt_tokens=msg.usage.input_tokens, output_tokens=msg.usage.output_tokens,
                  total_tokens=msg.usage.input_tokens + msg.usage.output_tokens)
    return VlmResult(data=_extract_json(text), usage=usage, raw_text=text)


# --------------------------------------------------------------------- OpenAI
def _analyze_openai(system: str, text_prompt: str, images: List[LoadedImage],
                    model: str) -> VlmResult:
    import base64

    from openai import OpenAI

    key = config.api_key_for("openai")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=key)

    content = [{"type": "text", "text": text_prompt}]
    for img in images:
        if img.exists:
            b64 = base64.b64encode(img.data).decode()
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:{img.mime_type};base64,{b64}"}})

    resp = client.chat.completions.create(
        model=model, temperature=config.TEMPERATURE,
        max_tokens=config.MAX_OUTPUT_TOKENS,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": content}],
    )
    text = resp.choices[0].message.content or ""
    u = resp.usage
    usage = Usage(calls=1, prompt_tokens=u.prompt_tokens, output_tokens=u.completion_tokens,
                  total_tokens=u.total_tokens)
    return VlmResult(data=_extract_json(text), usage=usage, raw_text=text)
