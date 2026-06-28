"""Per-claim orchestration: build prompt -> call VLM -> normalize -> apply the
deterministic history/evidence rule layer -> emit one output row.

The rule layer is intentionally separate from the model so that:
- user history only ADDS risk_flags / triggers manual review; it never flips a
  visual decision (per the problem statement);
- every emitted field is guaranteed to be a legal token, regardless of model output.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Set, Tuple

import config
from agent import prompt as prompt_mod
from agent import schema, vlm
from agent.data import Claim, UserHistory, load_images
from agent.vlm import Usage, VlmResult


def _history_flag_set(h: UserHistory) -> Set[str]:
    """The explicit flags recorded in user_history (the gold labels key off these
    directly, e.g. user_history_risk / manual_review_required - NOT off raw
    rejected/last-90 counts, which would over-flag)."""
    raw = (h.history_flags or "none").lower()
    return {f.strip() for f in re.split(r"[;,/]", raw) if f.strip() and f.strip() != "none"}


def _to_row(claim: Claim, fields: Dict) -> Dict[str, str]:
    """Serialize a normalized field dict into the exact output schema (strings)."""
    def join(ids: List[str]) -> str:
        return ";".join(ids) if ids else "none"

    def flags(fl: List[str]) -> str:
        return ";".join(fl) if fl else "none"

    return {
        "user_id": claim.user_id,
        "image_paths": claim.image_paths,
        "user_claim": claim.user_claim,
        "claim_object": claim.claim_object,
        "evidence_standard_met": "true" if fields["evidence_standard_met"] else "false",
        "evidence_standard_met_reason": fields["evidence_standard_met_reason"],
        "risk_flags": flags(fields["risk_flags"]),
        "issue_type": fields["issue_type"],
        "object_part": fields["object_part"],
        "claim_status": fields["claim_status"],
        "claim_status_justification": fields["claim_status_justification"],
        "supporting_image_ids": join(fields["supporting_image_ids"]),
        "valid_image": "true" if fields["valid_image"] else "false",
        "severity": fields["severity"],
    }


def _no_images_row(claim: Claim, history: UserHistory | None) -> Dict[str, str]:
    risk = ["damage_not_visible", "manual_review_required"]
    if history is not None and "user_history_risk" in _history_flag_set(history):
        risk.insert(0, "user_history_risk")
    fields = {
        "evidence_standard_met": False,
        "evidence_standard_met_reason": "No usable image was submitted, so the claim cannot be evaluated visually.",
        "risk_flags": risk,
        "issue_type": "unknown",
        "object_part": "unknown",
        "claim_status": "not_enough_information",
        "claim_status_justification": "No usable images were provided to verify the claim.",
        "supporting_image_ids": [],
        "valid_image": False,
        "severity": "unknown",
    }
    return _to_row(claim, fields)


def error_row(claim: Claim, history: UserHistory | None, message: str) -> Dict[str, str]:
    """Safe fallback when a claim cannot be processed (e.g. model/parse failure).
    Routes the claim to manual review rather than crashing the batch."""
    risk = ["manual_review_required"]
    if history is not None and "user_history_risk" in _history_flag_set(history):
        risk.insert(0, "user_history_risk")
    fields = {
        "evidence_standard_met": False,
        "evidence_standard_met_reason": "Automated review could not complete; routed to manual review.",
        "risk_flags": risk,
        "issue_type": "unknown",
        "object_part": "unknown",
        "claim_status": "not_enough_information",
        "claim_status_justification": f"Automated analysis failed ({message[:80]}); manual review required.",
        "supporting_image_ids": [],
        "valid_image": False,
        "severity": "unknown",
    }
    return _to_row(claim, fields)


def review_claim(claim: Claim, history: UserHistory | None,
                 requirements: List[dict]) -> Tuple[Dict[str, str], Usage]:
    images = load_images(claim)
    usable = [im for im in images if im.exists]
    valid_ids = {im.image_id for im in images}

    # Short-circuit: nothing to look at -> deterministic NEI (saves a model call).
    if not usable:
        return _no_images_row(claim, history), Usage()

    text_prompt = prompt_mod.build_text_prompt(claim, history, requirements, images)
    result, usage_obj, vote_disagreement = _judge(text_prompt, images)
    raw = result.data

    # --- normalize model output to legal tokens ---
    claim_status = schema.norm_claim_status(raw.get("claim_status"))
    issue_type = schema.norm_issue_type(raw.get("issue_type"))
    object_part = schema.norm_object_part(raw.get("object_part"), claim.claim_object)
    severity = schema.norm_severity(raw.get("severity"))
    evidence_met = schema.as_bool(raw.get("evidence_standard_met"))
    valid_image = schema.as_bool(raw.get("valid_image"))
    risk_flags = schema.norm_risk_flags(raw.get("risk_flags", []))

    # supporting image ids must reference images that were actually submitted
    support = [i for i in (raw.get("supporting_image_ids") or []) if i in valid_ids]

    # --- consistency guards (align evidence with the verdict; never invent damage) ---
    if issue_type == "none" and severity not in ("none", "unknown"):
        severity = "none"
    # A substantive verdict (supported/contradicted) means the claimed part was
    # visible enough to judge -> evidence standard was met. NEI means it was not.
    evidence_met = claim_status != "not_enough_information"
    if claim_status in ("supported", "contradicted") and not support:
        # cite the image(s) the model said actually show the claimed object
        support = [pi.get("image_id") for pi in raw.get("per_image", [])
                   if pi.get("shows_claimed_object") and pi.get("image_id") in valid_ids]
    if claim_status == "not_enough_information":
        support = []

    # --- deterministic history / review rule layer ---
    # History only ADDS risk flags; it never changes the visual verdict.
    history_wants_review = False
    if history is not None:
        hflags = _history_flag_set(history)
        if "user_history_risk" in hflags and "user_history_risk" not in risk_flags:
            risk_flags.append("user_history_risk")
        history_wants_review = bool(hflags & {"user_history_risk", "manual_review_required"})

    # Manual review is driven by trust/mismatch flags or a history review flag -
    # NOT by ordinary quality issues (blur / wrong angle) alone.
    needs_review = (any(f in schema.REVIEW_TRIGGER_FLAGS for f in risk_flags)
                    or history_wants_review or vote_disagreement)
    if needs_review and "manual_review_required" not in risk_flags:
        risk_flags.append("manual_review_required")

    fields = {
        "evidence_standard_met": evidence_met,
        "evidence_standard_met_reason": (raw.get("evidence_standard_met_reason") or "").strip()
                                        or "Evidence sufficiency assessed from the submitted images.",
        "risk_flags": risk_flags,
        "issue_type": issue_type,
        "object_part": object_part,
        "claim_status": claim_status,
        "claim_status_justification": (raw.get("claim_status_justification") or "").strip()
                                      or "Decision based on the submitted images.",
        "supporting_image_ids": support,
        "valid_image": valid_image,
        "severity": severity,
    }
    return _to_row(claim, fields), usage_obj


def _judge(text_prompt: str, images) -> Tuple[VlmResult, Usage, bool]:
    """Single call, or self-consistency majority vote across config.VOTE_MODELS.
    Returns (representative result, aggregate usage, disagreement flag)."""
    sys_prompt = prompt_mod.SYSTEM_INSTRUCTIONS
    if len(config.VOTE_MODELS) >= 2:
        results: List[VlmResult] = []
        for m in config.VOTE_MODELS:
            try:
                results.append(vlm.analyze_model(sys_prompt, text_prompt, images, m))
            except Exception:  # a model being down shouldn't break the vote
                continue
        if results:
            statuses = [schema.norm_claim_status(r.data.get("claim_status")) for r in results]
            winner = Counter(statuses).most_common(1)[0][0]
            rep = next(r for r, s in zip(results, statuses) if s == winner)
            agg = Usage()
            for r in results:
                agg.add(r.usage)
            return rep, agg, len(set(statuses)) > 1
    result = vlm.analyze(sys_prompt, text_prompt, images)
    return result, result.usage, False
