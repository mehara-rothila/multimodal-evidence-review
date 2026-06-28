"""Metrics for comparing predictions against sample_claims.csv expected labels."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

_SEV_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3}


def _split_set(value: str) -> set:
    if value is None:
        return set()
    return {v.strip() for v in str(value).split(";") if v.strip() and v.strip() != "none"}


def _set_prf(pred: str, gold: str) -> Tuple[float, float, float]:
    p, g = _split_set(pred), _split_set(gold)
    if not p and not g:
        return 1.0, 1.0, 1.0
    tp = len(p & g)
    prec = tp / len(p) if p else (1.0 if not g else 0.0)
    rec = tp / len(g) if g else (1.0 if not p else 0.0)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def _eq(a: str, b: str) -> bool:
    return (a or "").strip().lower() == (b or "").strip().lower()


def compute(preds: List[Dict[str, str]], golds: List[Dict[str, str]]) -> Dict:
    """Return a metrics dict over aligned prediction/gold rows."""
    n = len(preds)
    exact_fields = ["claim_status", "evidence_standard_met", "valid_image",
                    "issue_type", "object_part", "severity"]
    correct = {f: 0 for f in exact_fields}
    sev_within1 = 0
    setf = {f: [0.0, 0.0, 0.0] for f in ["risk_flags", "supporting_image_ids"]}
    confusion = defaultdict(int)  # (gold_status, pred_status) -> count

    for p, g in zip(preds, golds):
        for f in exact_fields:
            if _eq(p.get(f), g.get(f)):
                correct[f] += 1
        # severity adjacency
        ps, gs = p.get("severity", ""), g.get("severity", "")
        if ps in _SEV_ORDER and gs in _SEV_ORDER and abs(_SEV_ORDER[ps] - _SEV_ORDER[gs]) <= 1:
            sev_within1 += 1
        elif _eq(ps, gs):
            sev_within1 += 1
        for f in setf:
            prec, rec, f1 = _set_prf(p.get(f), g.get(f))
            setf[f][0] += prec
            setf[f][1] += rec
            setf[f][2] += f1
        confusion[(g.get("claim_status", ""), p.get("claim_status", ""))] += 1

    return {
        "n": n,
        "accuracy": {f: round(correct[f] / n, 4) for f in exact_fields},
        "severity_within_1": round(sev_within1 / n, 4),
        "set_f1": {f: {"precision": round(setf[f][0] / n, 4),
                       "recall": round(setf[f][1] / n, 4),
                       "f1": round(setf[f][2] / n, 4)} for f in setf},
        "claim_status_confusion": {f"{k[0]}->{k[1]}": v for k, v in sorted(confusion.items())},
    }


def format_report(metrics: Dict, label: str = "") -> str:
    a = metrics["accuracy"]
    lines = [f"### Metrics {label}".rstrip(), "",
             f"- rows evaluated: {metrics['n']}",
             f"- **claim_status accuracy: {a['claim_status']:.1%}**  (primary metric)",
             f"- evidence_standard_met accuracy: {a['evidence_standard_met']:.1%}",
             f"- valid_image accuracy: {a['valid_image']:.1%}",
             f"- issue_type accuracy: {a['issue_type']:.1%}",
             f"- object_part accuracy: {a['object_part']:.1%}",
             f"- severity accuracy: {a['severity']:.1%}  (within-1: {metrics['severity_within_1']:.1%})",
             f"- risk_flags F1: {metrics['set_f1']['risk_flags']['f1']:.3f}  "
             f"(P {metrics['set_f1']['risk_flags']['precision']:.3f} / "
             f"R {metrics['set_f1']['risk_flags']['recall']:.3f})",
             f"- supporting_image_ids F1: {metrics['set_f1']['supporting_image_ids']['f1']:.3f}",
             "",
             "claim_status confusion (gold -> pred):"]
    for k, v in metrics["claim_status_confusion"].items():
        lines.append(f"  - {k}: {v}")
    return "\n".join(lines)
