"""Output contract: allowed values, the model's structured response shape, and
normalization that forces every field into a legal value.

Keeping the allowed values in one place means the prompt, the structured-output
schema, and the post-processing validator can never drift apart.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

# --------------------------------------------------------------- output columns
# Exact order required by problem_statement.md.
OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

INPUT_COLUMNS = ["user_id", "image_paths", "user_claim", "claim_object"]

# --------------------------------------------------------------- allowed values
CLAIM_OBJECTS = ["car", "laptop", "package"]

CLAIM_STATUS = ["supported", "contradicted", "not_enough_information"]

ISSUE_TYPES = [
    "dent", "scratch", "crack", "glass_shatter", "broken_part", "missing_part",
    "torn_packaging", "crushed_packaging", "water_damage", "stain", "none", "unknown",
]

SEVERITY = ["none", "low", "medium", "high", "unknown"]

RISK_FLAGS = [
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
    "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
    "claim_mismatch", "possible_manipulation", "non_original_image",
    "text_instruction_present", "user_history_risk", "manual_review_required",
]

OBJECT_PARTS = {
    "car": [
        "front_bumper", "rear_bumper", "door", "hood", "windshield", "side_mirror",
        "headlight", "taillight", "fender", "quarter_panel", "body", "unknown",
    ],
    "laptop": [
        "screen", "keyboard", "trackpad", "hinge", "lid", "corner", "port",
        "base", "body", "unknown",
    ],
    "package": [
        "box", "package_corner", "package_side", "seal", "label", "contents",
        "item", "unknown",
    ],
}

# Union of every part token, used to constrain the model's structured output.
ALL_OBJECT_PARTS = sorted({p for parts in OBJECT_PARTS.values() for p in parts})

# Risk flags that are derived deterministically from user history, not vision.
HISTORY_RISK_FLAGS = {"user_history_risk", "manual_review_required"}

# Visual risk flags that should trigger human review when present.
REVIEW_TRIGGER_FLAGS = {
    "wrong_object", "claim_mismatch", "possible_manipulation",
    "non_original_image", "text_instruction_present",
}


# --------------------------------------------------------- structured response
# What we ask the vision model to return. Enums guarantee legal values for the
# constrained fields; free-text fields carry the image-grounded reasoning.
class PerImage(BaseModel):
    image_id: str
    shows_claimed_object: bool = Field(description="Does this image show the claimed object at all?")
    quality_issues: List[str] = Field(default_factory=list, description="e.g. blurry, glare, cropped, dark, wrong_angle")
    visible_issue_type: str = Field(default="unknown", description="issue visible IN THIS image, from the issue_type list")
    visible_object_part: str = Field(default="unknown", description="part visible IN THIS image")
    notes: str = Field(default="", description="one short grounded observation")


class ReviewResponse(BaseModel):
    extracted_claim: str = Field(description="the actual damage claim, paraphrased from the conversation")
    claimed_issue_type: str = Field(description="issue the user is claiming")
    claimed_object_part: str = Field(description="part the user is claiming")
    per_image: List[PerImage]
    evidence_standard_met: bool
    evidence_standard_met_reason: str
    issue_type: str = Field(description="final visible issue type, from the allowed list")
    object_part: str = Field(description="final relevant part, from the allowed list for this object")
    claim_status: str = Field(description="supported | contradicted | not_enough_information")
    claim_status_justification: str
    supporting_image_ids: List[str] = Field(default_factory=list)
    valid_image: bool
    severity: str = Field(description="none | low | medium | high | unknown")
    risk_flags: List[str] = Field(default_factory=list, description="visual risk flags only; history flags are added later")


# ------------------------------------------------------------------ normalizers
def _closest(value: str, allowed: List[str], default: str) -> str:
    """Snap a model value to the nearest legal token (exact, then substring)."""
    if value is None:
        return default
    v = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    if v in allowed:
        return v
    for a in allowed:
        if a == v:
            return a
    # substring / containment fallback (e.g. "shattered_glass" -> "glass_shatter")
    for a in allowed:
        if a != "unknown" and a != "none" and (a in v or v in a):
            return a
    return default


def norm_issue_type(value: str) -> str:
    return _closest(value, ISSUE_TYPES, "unknown")


def norm_object_part(value: str, claim_object: str) -> str:
    allowed = OBJECT_PARTS.get(claim_object, ALL_OBJECT_PARTS)
    return _closest(value, allowed, "unknown")


def norm_claim_status(value: str) -> str:
    return _closest(value, CLAIM_STATUS, "not_enough_information")


def norm_severity(value: str) -> str:
    return _closest(value, SEVERITY, "unknown")


def norm_risk_flags(values: List[str]) -> List[str]:
    out: List[str] = []
    for v in values or []:
        snapped = _closest(v, RISK_FLAGS, "")
        if snapped and snapped != "none" and snapped not in out:
            out.append(snapped)
    return out


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes", "y")
