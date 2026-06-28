"""Central configuration: paths, model selection, runtime knobs.

Secrets are read from environment variables only (loaded from code/.env if
present). Nothing here is hardcoded that you would not want in a public repo.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent

# Load local .env (gitignored). Safe no-op if the file is absent.
load_dotenv(CODE_DIR / ".env")

# ---------------------------------------------------------------- data paths
DATASET_DIR = REPO_ROOT / "dataset"
IMAGES_DIR = DATASET_DIR  # image_paths in the CSVs are relative to dataset/
CLAIMS_CSV = DATASET_DIR / "claims.csv"
SAMPLE_CLAIMS_CSV = DATASET_DIR / "sample_claims.csv"
USER_HISTORY_CSV = DATASET_DIR / "user_history.csv"
EVIDENCE_REQUIREMENTS_CSV = DATASET_DIR / "evidence_requirements.csv"
DEFAULT_OUTPUT_CSV = DATASET_DIR / "output.csv"

CACHE_DIR = CODE_DIR / ".cache"

# ---------------------------------------------------------------- model config
PROVIDER = os.environ.get("EVIDENCE_PROVIDER", "gemini").strip().lower()
MODEL = os.environ.get("EVIDENCE_MODEL", "gemini-3.5-flash").strip()

# Optional ordered failover pool (comma-separated). When set, the client uses the
# first model and automatically rolls to the next when one hits its daily quota.
# Lets the free tier (20 req/day PER MODEL) cover a batch larger than 20.
MODEL_POOL = [m.strip() for m in os.environ.get("EVIDENCE_MODEL_POOL", "").split(",") if m.strip()]

# Optional self-consistency voting: if 2+ models are listed, each claim is judged
# by all of them and claim_status is decided by majority vote; disagreement is
# surfaced as manual_review_required. Triples+ the calls, so off by default.
VOTE_MODELS = [m.strip() for m in os.environ.get("EVIDENCE_VOTE_MODELS", "").split(",") if m.strip()]
THINKING_BUDGET = int(os.environ.get("EVIDENCE_THINKING_BUDGET", "-1"))
TEMPERATURE = float(os.environ.get("EVIDENCE_TEMPERATURE", "0"))
# Must comfortably hold THINKING tokens + the JSON answer (thinking counts toward
# the output budget on Gemini 2.5+, so too-small a value truncates the JSON).
MAX_OUTPUT_TOKENS = int(os.environ.get("EVIDENCE_MAX_OUTPUT_TOKENS", "12000"))

# ---------------------------------------------------------------- runtime knobs
# Concurrency + retry to respect provider RPM/TPM limits without being slow.
# Gemini free tier is strict (e.g. gemini-3.5-flash = 5 RPM). The rate limiter
# paces request *starts* to stay under REQUESTS_PER_MINUTE; workers overlap the
# (slow) thinking latency within that budget.
MAX_WORKERS = int(os.environ.get("EVIDENCE_MAX_WORKERS", "3"))
# Stay UNDER the hard cap: free-tier gemini-3.5-flash is 5/min, but a retry is
# itself a request, so pacing at exactly 5 lets a single 429 cascade. 4/min
# (15s spacing) leaves headroom for the occasional retry. Raise on paid tiers.
REQUESTS_PER_MINUTE = float(os.environ.get("EVIDENCE_RPM", "4"))
MAX_RETRIES = int(os.environ.get("EVIDENCE_MAX_RETRIES", "6"))
RETRY_BASE_DELAY = float(os.environ.get("EVIDENCE_RETRY_BASE_DELAY", "2.0"))
USE_CACHE = os.environ.get("EVIDENCE_USE_CACHE", "1") not in ("0", "false", "False")

# Downscale very large images before upload to save tokens/latency.
MAX_IMAGE_EDGE = int(os.environ.get("EVIDENCE_MAX_IMAGE_EDGE", "1024"))


def api_key_for(provider: str) -> str | None:
    return {
        "gemini": os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
        "anthropic": os.environ.get("ANTHROPIC_API_KEY"),
        "openai": os.environ.get("OPENAI_API_KEY"),
    }.get(provider)


# Multiple Gemini keys (from DIFFERENT projects = independent 20/day quotas).
# Set EVIDENCE_GEMINI_KEYS=key1,key2,... to rotate when a key's daily quota for a
# model is exhausted. Falls back to the single GEMINI_API_KEY.
def gemini_keys() -> list[str]:
    multi = [k.strip() for k in os.environ.get("EVIDENCE_GEMINI_KEYS", "").split(",") if k.strip()]
    if multi:
        return multi
    single = api_key_for("gemini")
    return [single] if single else []
