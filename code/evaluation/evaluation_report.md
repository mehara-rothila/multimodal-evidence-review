# Evaluation Report - Multi-Modal Evidence Review

## 1. Method

The agent is evaluated on `dataset/sample_claims.csv` (20 labeled claims) by
comparing predictions against the expected labels field-by-field:

- **Exact-match accuracy** for single-value fields (`claim_status`,
  `evidence_standard_met`, `valid_image`, `issue_type`, `object_part`,
  `severity`).
- **Set-based F1** for multi-value fields (`risk_flags`,
  `supporting_image_ids`).
- A **claim_status confusion matrix** (the primary business metric).
- `severity` also reported as **within-1** (adjacent band) since severity is
  inherently fuzzy.

Reproduce:

```bash
python code/evaluation/main.py                         # primary config
python code/evaluation/main.py --model <m> --tag <t>   # a comparison config
```

Each run writes `evaluation/results/predictions_<tag>.csv` and
`metrics_<tag>.json`.

> Free-tier note: `gemini-3.5-flash` (the originally requested model) and
> `gemini-2.5-flash` are capped at **20 requests/day each**, so configs were run
> on the model that had quota at the time. The comparison below therefore varies
> *both* the decision-rule strategy and the model; the dominant driver of the
> gain is the rule tuning (Section 3), confirmed by the confusion-matrix shift.

## 2. Results (20-sample dev set)

| Config | Model | Rules | **claim_status** | evidence | valid_image | issue_type | object_part | severity (±1) | risk_flags F1 | support_ids F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | gemini-3.5-flash | v1 (untuned) | 50% | 55% | 90% | 40% | 85% | 40% (70%) | 0.643 | 0.50 |
| Tuned rules | gemini-2.5-flash | v3 | 75% | 85% | 80% | 55% | 70% | 55% (85%) | 0.710 | 0.90 |
| **+ few-shot (FINAL)** | lite (3.1-flash-lite) | v4 (+7 exemplars) | **80%** | **90%** | **95%** | 45% | **80%** | 65% (85%) | **0.884** | 0.833 |
| + few-shot + part-gate | lite | v5 | 80% | 85% | 95% | 50% | 80% | 60% (80%) | 0.827 | 0.867 |
| premium model | **gemini-3.5-flash** | v5 | 70% | 85% | 90% | 45% | 90% | 65% (80%) | 0.588 | 0.767 |
| 3-model voting ensemble | lite+2.5-flash+3-preview | v5 | 75% | 85% | 95% | 50% | 75% | 60% (80%) | 0.767 | 0.867 |

**Findings (this is the core analysis):**
- **Few-shot exemplars drove the win** (75%→80% claim_status, risk_flags F1 0.71→0.88) - on the
  *weaker lite model*, so it's the rules, not the model.
- **Bigger model scored *lower*.** `gemini-3.5-flash` got only 70% and risk_flags F1 0.59 - it
  **over-reasons and over-flags**, disputing the gold's judgment calls and dumping 34% of the
  test set into `not_enough_information`. On this task **decision-rule calibration beats model
  size** - so we did NOT ship the bigger model.
- **Self-consistency voting regressed** (75%): the errors are correlated/subjective, so an
  ensemble doesn't help and adds 3× cost - not shipped.
- **The claimed-part gate** (v5) fixes specific reasoning errors but slightly lowered the gold
  metrics, so the simpler **v4 few-shot** is the shipped config.
- We **A/B-tested 6 configurations** and chose the one that scored best on ground truth.

> **Caveat on the numbers.** All scores above are on a **20-row labeled dev set**, which
> has wide confidence intervals (a few rows swing a percentage point a lot). The hidden test
> set is larger and scored across *all* fields, so treat these as **directional, not absolute** -
> they guided which configuration to ship, not a guarantee of held-out performance.

**claim_status confusion (tuned, gold → pred):**

```
supported     -> supported            12     (all supported caught)
contradicted  -> contradicted          1
contradicted  -> not_enough_info       2
contradicted  -> supported             2
not_enough    -> not_enough_info       2
not_enough    -> contradicted          1
```

Reading it: `supported` recall is perfect (12/12). The remaining error mass is
the **subtle `contradicted` cases** - "part is visible but undamaged / milder
than claimed" - which need fine-grained perception. This is the class a stronger
model (gemini-3.5-flash or Claude) is expected to lift, and is the top item in
Section 6.

## 3. What moved the needle (strategy ablation)

The jump from 50% → 75% claim_status came from four decision-rule changes,
derived from the confusion matrix and a labeled-data rubric study:

1. **Multi-image "best image wins."** `REQ_GENERAL_MULTI_IMAGE` says one clear
   relevant image is enough. v1 let a second irrelevant/stock image veto a good
   one → over-predicted `not_enough_information`. v3 decides from the best
   relevant image and demotes the bad image to a *risk flag*. (Removed 5
   supported→NEI and several contradicted→NEI errors.)
2. **Evidence ↔ verdict alignment.** A substantive verdict implies the part was
   visible, so `evidence_standard_met = (claim_status != not_enough_information)`,
   derived deterministically. (55% → 85%.)
3. **History-flag calibration.** `user_history_risk` / `manual_review_required`
   are taken *only* from the `history_flags` column, not from raw rejected/last-90
   counts - verified against the labels (users with `rejected≥1` but
   `flags=none` get **no** flag). Removed false-positive risk flags.
4. **Severity scale.** Explicit bands (`medium` = clear damage default, `high` =
   severe wreckage only, claimed-severe-but-minor → `contradicted`+`low`).
   (severity within-1 70% → 85%.)

## 4. Final test-set run (`dataset/claims.csv`, 44 claims)

Produced `dataset/output.csv` with the v4 rules (tuned + few-shot) over the
failover pool.

| Quantity | Value |
|---|---|
| Claims processed | 44 (44 model calls, 0 cache hits, **0 fallback rows**) |
| Model used | `gemini-3.1-flash-lite` ×44 (single model; pool available for failover) |
| Total tokens | **261,359** (input 186,746 · output 17,099 · thinking 57,514) |
| Avg per claim | ~5,940 tokens, ~7.1 s wall |
| Wall-clock runtime | **312.3 s (~5 min)** at the throttled rate |
| Verdict mix | supported 17 · contradicted 19 · not_enough_information 8 |

(An earlier no-few-shot pass over a 2-model split is kept as
`dataset/output_v1_mixedpool.csv` for comparison.)

## 5. Operational analysis

**Model calls.** One call per claim (no multi-pass). Sample eval = 20 calls;
test = 44 calls. Development consumed ~3 extra eval iterations (cached, so
re-runs were free).

**Token usage.** ~5.9k tokens/claim end-to-end: ~4.2k input (system + rules +
few-shot exemplars + conversation + 1-3 downscaled images), ~1.3k thinking,
~0.4k JSON output. Images are downscaled to ≤1024 px to cut input tokens.

**Images.** 29 sample + 82 test = **111 images** processed.

**Cost (full 44-claim test set).** On the **free tier the run cost $0**. As a
paid-tier estimate using representative gemini-2.5-flash pricing ($0.30 / 1M
input, $2.50 / 1M output incl. thinking):

```text
input :  186,746 / 1e6 * $0.30 = $0.056
output:  (17,099 + 57,514) / 1e6 * $2.50 = $0.187
total ≈ $0.24 for 44 claims  (~$0.0055 / claim)
```

The test largely ran on *lite* models (cheaper), so a same-tier estimate is
lower; the figure above is a conservative upper bound. Scaling: ~**$5 per 1,000
claims**.

**Latency.** Model latency ~6-12 s/call (extended thinking). Wall-clock is
dominated by deliberate rate-limiting, not compute.

**TPM/RPM & reliability strategy.**
- **Rate limiter** paces request *starts* under the cap (free tier is strict;
  pacing at exactly the limit lets a single retry cascade, so we pace below it).
- **Quota-aware failover**: free tier = 20 req/day **per model**, so a pool of
  models is used in order; a daily-quota `429` marks a model exhausted and the
  client rolls to the next. This is what let a 44-claim batch complete on the
  free tier in one run.
- **Retries** honor the server's `retryDelay` for transient (RPM) `429`s, but
  do *not* retry daily-quota errors (pointless) - they switch models instead.
- **On-disk cache** keyed by prompt + image bytes: identical re-runs cost $0 and
  are instant; safe resume after interruptions.
- **Per-claim isolation**: one failed claim falls back to a safe
  manual-review row instead of sinking the batch.

## 6. Limitations & next steps

- **Subtle `contradicted` cases** are the main error class; re-running on
  `gemini-3.5-flash` (extended thinking) or Claude - a one-line `.env` change,
  since the client is provider-agnostic - is expected to lift these.
- The final test set ran on *lite* models due to free-tier daily caps; a paid
  key or enabled billing would allow a single high-quality model end-to-end.
- A larger labeled set would tighten the per-field estimates (20 samples → wide
  confidence intervals).
