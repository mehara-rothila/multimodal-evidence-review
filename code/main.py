"""Entry point: run the evidence-review agent over a claims CSV and write output.csv.

Usage:
    python code/main.py                       # dataset/claims.csv -> dataset/output.csv
    python code/main.py --limit 5             # quick smoke run on the first 5 rows
    python code/main.py --input dataset/sample_claims.csv --output sample_predictions.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python code/main.py` from the repo root by making code/ importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from agent.data import load_claims
from agent.runner import run_claims, write_output_csv


def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-modal evidence review agent")
    ap.add_argument("--input", default=str(config.CLAIMS_CSV), help="input claims CSV")
    ap.add_argument("--output", default=str(config.DEFAULT_OUTPUT_CSV), help="output CSV path")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N rows (0 = all)")
    ap.add_argument("--workers", type=int, default=config.MAX_WORKERS, help="concurrent workers")
    args = ap.parse_args()

    claims = load_claims(Path(args.input))
    if args.limit:
        claims = claims[: args.limit]

    print(f"Provider={config.PROVIDER} model={config.MODEL} "
          f"thinking_budget={config.THINKING_BUDGET} workers={args.workers}")
    print(f"Reviewing {len(claims)} claims from {args.input} ...\n")

    rows, usage = run_claims(claims, max_workers=args.workers)
    write_output_csv(rows, Path(args.output))

    print(f"\nWrote {len(rows)} rows -> {args.output}")
    from agent.vlm import model_usage_summary
    mu = model_usage_summary()
    if mu:
        print("Live calls per model:", ", ".join(f"{k}={v}" for k, v in mu.items()))
    _print_cost(usage)
    return 0


def _print_cost(usage) -> None:
    # Rough pricing knobs (USD per 1M tokens). Adjust in the report as needed.
    in_price = 0.30
    out_price = 2.50
    billable_out = usage.output_tokens + usage.thoughts_tokens
    cost = usage.prompt_tokens / 1e6 * in_price + billable_out / 1e6 * out_price
    print(f"Tokens: in={usage.prompt_tokens} out={usage.output_tokens} "
          f"thinking={usage.thoughts_tokens} | est. cost ~${cost:.4f} "
          f"(assuming ${in_price}/M in, ${out_price}/M out)")


if __name__ == "__main__":
    raise SystemExit(main())
