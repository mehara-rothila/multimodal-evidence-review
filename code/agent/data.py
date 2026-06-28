"""Loading and light preprocessing of the dataset: claims, user history,
evidence requirements, and the images themselves.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import config


# ----------------------------------------------------------------- data classes
@dataclass
class Claim:
    user_id: str
    image_paths: str            # raw, semicolon-separated, as in the CSV
    user_claim: str
    claim_object: str
    image_rel_paths: List[str] = field(default_factory=list)
    # When sample_claims.csv is loaded, the expected outputs live here.
    expected: Optional[Dict[str, str]] = None


@dataclass
class UserHistory:
    user_id: str
    past_claim_count: int
    accept_claim: int
    manual_review_claim: int
    rejected_claim: int
    last_90_days_claim_count: int
    history_flags: str
    history_summary: str


@dataclass
class LoadedImage:
    image_id: str               # filename without extension, e.g. "img_1"
    rel_path: str               # e.g. "images/test/case_001/img_1.jpg"
    data: bytes
    mime_type: str
    exists: bool = True


# ---------------------------------------------------------------- path helpers
def parse_image_paths(raw: str) -> List[str]:
    return [p.strip() for p in (raw or "").split(";") if p.strip()]


def image_id_from_path(rel_path: str) -> str:
    return Path(rel_path).stem


_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


def _mime_for(path: Path) -> str:
    return _MIME.get(path.suffix.lower(), "image/jpeg")


# ------------------------------------------------------------------- CSV loaders
def _read_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_claims(path: Path, with_expected: bool = False) -> List[Claim]:
    claims: List[Claim] = []
    for row in _read_rows(path):
        c = Claim(
            user_id=row.get("user_id", "").strip(),
            image_paths=row.get("image_paths", "").strip(),
            user_claim=row.get("user_claim", "").strip(),
            claim_object=row.get("claim_object", "").strip().lower(),
        )
        c.image_rel_paths = parse_image_paths(c.image_paths)
        if with_expected:
            c.expected = {k: (row.get(k) or "").strip() for k in row}
        claims.append(c)
    return claims


def load_user_history(path: Path) -> Dict[str, UserHistory]:
    def _int(v: str) -> int:
        try:
            return int(str(v).strip())
        except (ValueError, TypeError):
            return 0

    out: Dict[str, UserHistory] = {}
    for row in _read_rows(path):
        uid = row.get("user_id", "").strip()
        if not uid:
            continue
        out[uid] = UserHistory(
            user_id=uid,
            past_claim_count=_int(row.get("past_claim_count")),
            accept_claim=_int(row.get("accept_claim")),
            manual_review_claim=_int(row.get("manual_review_claim")),
            rejected_claim=_int(row.get("rejected_claim")),
            last_90_days_claim_count=_int(row.get("last_90_days_claim_count")),
            history_flags=(row.get("history_flags") or "none").strip(),
            history_summary=(row.get("history_summary") or "").strip(),
        )
    return out


def load_evidence_requirements(path: Path) -> List[Dict[str, str]]:
    return _read_rows(path)


def requirements_for_object(reqs: List[Dict[str, str]], claim_object: str) -> List[Dict[str, str]]:
    """Rules that apply to this object (its own + the `all` rules)."""
    return [r for r in reqs if r.get("claim_object") in (claim_object, "all")]


# --------------------------------------------------------------- image loading
def load_images(claim: Claim) -> List[LoadedImage]:
    """Load (and optionally downscale) each image referenced by the claim.

    Missing files are returned with exists=False so the agent can flag them
    rather than crash.
    """
    images: List[LoadedImage] = []
    for rel in claim.image_rel_paths:
        abs_path = (config.IMAGES_DIR / rel).resolve()
        image_id = image_id_from_path(rel)
        if not abs_path.exists():
            images.append(LoadedImage(image_id, rel, b"", "image/jpeg", exists=False))
            continue
        data = abs_path.read_bytes()
        mime = _mime_for(abs_path)
        data, mime = _maybe_downscale(data, mime)
        images.append(LoadedImage(image_id, rel, data, mime, exists=True))
    return images


def _maybe_downscale(data: bytes, mime: str) -> tuple[bytes, str]:
    """Shrink large images to MAX_IMAGE_EDGE to cut tokens/latency. Best-effort."""
    try:
        from PIL import Image
    except ImportError:
        return data, mime
    try:
        img = Image.open(io.BytesIO(data))
        longest = max(img.size)
        if longest <= config.MAX_IMAGE_EDGE:
            return data, mime
        scale = config.MAX_IMAGE_EDGE / longest
        new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
        img = img.convert("RGB").resize(new_size, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)
        return buf.getvalue(), "image/jpeg"
    except Exception:
        return data, mime
