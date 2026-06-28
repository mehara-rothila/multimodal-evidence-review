"""Batch execution: run the reviewer over many claims with bounded concurrency,
accumulate token usage, and write the output CSV in the exact required schema.
"""
from __future__ import annotations

import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import config
from agent import reviewer, schema
from agent.data import (Claim, load_evidence_requirements, load_user_history,
                        requirements_for_object)
from agent.vlm import Usage


def run_claims(claims: List[Claim], max_workers: int | None = None,
               progress: bool = True) -> Tuple[List[Dict[str, str]], Usage]:
    history_map = load_user_history(config.USER_HISTORY_CSV)
    all_reqs = load_evidence_requirements(config.EVIDENCE_REQUIREMENTS_CSV)
    workers = max_workers or config.MAX_WORKERS

    rows: List[Dict[str, str]] = [None] * len(claims)  # type: ignore[list-item]
    total_usage = Usage()
    done = 0
    start = time.time()

    def work(idx: int, claim: Claim):
        reqs = requirements_for_object(all_reqs, claim.claim_object)
        history = history_map.get(claim.user_id)
        try:
            return idx, reviewer.review_claim(claim, history, reqs)
        except Exception as e:  # one bad claim must never sink the whole batch
            print(f"  [warn] claim {idx} ({claim.user_id}) failed: {str(e)[:90]}", flush=True)
            return idx, (reviewer.error_row(claim, history, str(e)), Usage())

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(work, i, c) for i, c in enumerate(claims)]
        for fut in as_completed(futures):
            idx, (row, usage) = fut.result()
            rows[idx] = row
            total_usage.add(usage)
            done += 1
            if progress:
                tag = "cache" if usage.cache_hits else f"{usage.total_tokens}tok"
                print(f"  [{done}/{len(claims)}] {row['user_id']:>8} {row['claim_object']:<8}"
                      f" -> {row['claim_status']:<22} ({tag})", flush=True)

    if progress:
        dt = time.time() - start
        print(f"\nProcessed {len(claims)} claims in {dt:.1f}s "
              f"({total_usage.calls} model calls, {total_usage.cache_hits} cache hits, "
              f"{total_usage.total_tokens} tokens).")
    return rows, total_usage


def write_output_csv(rows: List[Dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=schema.OUTPUT_COLUMNS,
                                quoting=csv.QUOTE_ALL, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
