"""Evaluation harness.

Runs the agent on dataset/sample_claims.csv (which carries expected labels) and
reports per-field accuracy, set-F1 for multi-valued fields, and a claim_status
confusion matrix. Supports overriding the model/thinking budget so two configs
can be compared on identical data.

Examples:
    python code/evaluation/main.py
    python code/evaluation/main.py --model gemini-2.5-flash --tag flash25
    python code/evaluation/main.py --thinking 0 --tag no-thinking
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make code/ importable

import config
from agent.data import load_claims
from agent.runner import run_claims, write_output_csv
from evaluation import metrics as M


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate the agent on labeled samples")
    ap.add_argument("--input", default=str(config.SAMPLE_CLAIMS_CSV))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=config.MAX_WORKERS)
    ap.add_argument("--model", default=None, help="override EVIDENCE_MODEL")
    ap.add_argument("--thinking", type=int, default=None, help="override thinking budget")
    ap.add_argument("--tag", default=None, help="label for output files")
    args = ap.parse_args()

    if args.model:
        config.MODEL = args.model
        config.MODEL_POOL = []  # a comparison run pins one model, no failover
    if args.thinking is not None:
        config.THINKING_BUDGET = args.thinking
    tag = args.tag or f"{config.PROVIDER}_{config.MODEL}".replace("/", "-")

    claims = load_claims(Path(args.input), with_expected=True)
    if args.limit:
        claims = claims[: args.limit]
    golds = [c.expected for c in claims]

    print(f"Eval config: provider={config.PROVIDER} model={config.MODEL} "
          f"thinking={config.THINKING_BUDGET} | {len(claims)} samples\n")
    preds, usage = run_claims(claims, max_workers=args.workers)

    metrics = M.compute(preds, golds)
    report = M.format_report(metrics, label=f"[{tag}]")
    print("\n" + report)

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    write_output_csv(preds, out_dir / f"predictions_{tag}.csv")
    (out_dir / f"metrics_{tag}.json").write_text(
        json.dumps({"tag": tag, "model": config.MODEL,
                    "thinking_budget": config.THINKING_BUDGET,
                    "usage": usage.__dict__, "metrics": metrics}, indent=2),
        encoding="utf-8")
    print(f"\nSaved -> {out_dir / f'metrics_{tag}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
