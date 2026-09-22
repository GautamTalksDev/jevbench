# Pre-registration — exp1_difficulty_calibration

**Status:** LOCKED (amended — see Amendments)
**Experiment:** `exp1_difficulty_calibration`
**Task:** `datasets/support_tickets/`
**Pinned model:** `jev-1.13.0`
**Serving path:** `native`
**Locked at (UTC):** 2026-09-20T02:59:57+00:00
**Amendments dated (UTC):** 2026-09-22

This document locks hypotheses, metrics, and decision rules **before**
scored API calls. Analyses that diverge are labelled exploratory.

---

## Publication commitment (pre-committed)

> **We will publish the result regardless of direction, including a null or a
> result supporting TypeSafe's claims.**

This sentence is locked before any paid call. A null or an inconclusive
interval under the powered ΔECE design is a publishable outcome. Running
EXP-1 underpowered and dressing an inconclusive interval as a finding is not.

---

## Content hashes (binding)

Scored runs **must refuse** if either hash changes.

| File | SHA-256 |
|---|---|
| `datasets/support_tickets/items.jsonl` | `8053abd2c46cd892c4f48e870bf74bfa91427a78f858c4b1c0549c864de64c23` |
| `datasets/support_tickets/labels.jsonl` | `9705aa795ff798ccaa5f8ebded8b27d2caa647245966fce803f70a3839fe041c` |

Machine-readable lock: `preregistration.lock.json`

- Items currently locked: **8** (demo corpus)
- Labels: **8**
- Tier counts: `{'trivial': 2, 'easy': 2, 'hard': 2, 'ambiguous': 2}`

**Labeling obligation (Amendment 4):** before the first paid call, grow the
corpus to **n ≥ 200 per pooled stratum** (easy = trivial∪easy, hard =
hard∪ambiguous), equal n, as required by `results/power_analysis.json`.
Re-lock content hashes after labeling; do not score until hashes match the
amended sample-size rule.

---

## Description

Flagship difficulty-stratified calibration study. Same label schema across
four tiers (trivial, easy, hard, ambiguous). **Primary endpoint (amended):**
whether top-label ΔECE between pooled hard and pooled easy strata differs
from zero. The four-tier ECE-vs-accuracy slope is retained as a
**descriptive** figure only.

---

## Hypotheses & falsification rules

### H1 — primary (amended)

**Statement:** Top-label ΔECE = ECE(hard∪ambiguous) − ECE(trivial∪easy) is
distinguishable from zero in the direction implied by the literature gap
(hard less calibrated), with equal n per pooled stratum and occupancy
reported on every ECE.

**Falsified when:** The stratified two-sample bootstrap 95% CI on ΔECE
includes zero (inconclusive — not equivalence), or the powered sample fails
to exclude zero at the pre-specified effect size. Never interpret an
underpowered inconclusive interval as a substantive finding.

### H1a — descriptive (original slope; retained)

**Statement:** ECE is approximately flat across difficulty tiers after
accounting for bin occupancy (calibration is independent of accuracy).

**Falsified when:** The bootstrap CI (10k paired resamples) on the slope of
ECE vs tier accuracy excludes zero in the positive direction (ECE rises as
accuracy falls), with adequate occupancy in every tier; then prefer H1b.
**Note:** the slope has two residual df and is underpowered by construction;
it is not the primary endpoint (Amendment 1).

### H1b — descriptive (original slope; retained)

**Statement:** ECE rises as tier accuracy falls (calibration tracks accuracy).

**Falsified when:** The bootstrap CI on the ECE-vs-accuracy slope includes
zero and every tier has adequate bin occupancy; then prefer H1a. An interval
crossing zero is inconclusive, not equivalence.

---

## Metrics

- **Primary:** top-label ΔECE (hard∪ambiguous vs trivial∪easy), uniform and
  quantile binning (M=10), stratified two-sample bootstrap CI (10 000
  resamples for scored analysis), equal n enforced
- ECE (uniform / quantile) with per-bin occupancy — netcal-cross-checked
- Bias floor / null band: `results/bias_floor.json` (Amendment 2)
- MCE, Brier, reliability diagrams per tier
- Accuracy per tier
- **Descriptive:** ECE-vs-accuracy OLS slope across four tiers (not powered)
- Automation@0.90 accuracy
- Latency p50/p95/p99 (never mean)
- Paired bootstrap CIs when comparing two models on the **same** items

---

## Decision rules

- Report **ΔECE** (with CI, equal-n flag, occupancy) as the primary result.
- Retain the ECE-vs-accuracy curve as a descriptive figure only.
- Never report ECE without bin occupancy.
- Calibration metrics use probabilities/noul only — never confidence.
- Interval crossing zero ⇒ inconclusive, not evidence of equivalence.
- Do not run EXP-1 underpowered; if labeling capacity is below the power
  floor, prefer a larger pre-specified detectable effect or add domains —
  do not publish an underpowered inconclusive as a finding.

---

## Sample size

- **n per pooled stratum:** **200** (from `results/power_analysis.json`,
  Amendment 4) — power ≥ 0.80 to detect ΔECE = 0.09 (Amendment 3)
- **Total items to label (two pooled strata):** ≥ 400, equal n
- **repeats:** 3 (scored EXP-1)
- **Bootstrap (scored):** n_boot = 10 000
- **notes:** Demo lock still has 8 items; paper protocol requires meeting
  the power floor before the first paid call. Mean null FPR in the power
  analysis was ~0.07 (near 0.05); if FPR were materially above 0.05, stop
  and fix the estimator before spending.

---

## Stopping rule

Stop after the pre-registered powered sample is scored with the stated
repeats. Do not add items because interim ΔECE looks flat or rising. New
items require a new preregistration version and exploratory labelling of
prior runs.

---

## Contamination guard

1. Labels are fixed at lock time (hash above); re-lock after labeling to n.
2. `jevbench verify-labels` fails if `labels.jsonl` changes after
   the first scored run under `runs/`.
3. Adjudicating a label after seeing model output is a protocol
   breach — the tool makes accidental contamination fail CI.

---

## Amendments

All amendments below were made on the basis of **simulated and fixture data
containing no measurements of Jev**. Revising a design before seeing any real
scored data is correct practice; revising after is not. The git history is
what distinguishes them: these amendments are committed **before** any paid
API call and before any scored `runs/<id>/` artifact (the only JSONL under
`runs/` at amendment time is `runs/offline_fixture/`, an offline demo pack
explicitly marked not scored).

| Amendment | Date (UTC) | Binding commit |
|---|---|---|
| 1–4 (this section) | 2026-09-22 | _filled in immediately after this commit; see below_ |

### Amendment 1 — primary endpoint → ΔECE

**Change:** EXP-1 primary endpoint changed from OLS slope of ECE on accuracy
across four tiers, to **ΔECE** between pooled hard and pooled easy strata
with a **stratified two-sample bootstrap** (disjoint items — never paired).

**Reason:** the slope has two residual degrees of freedom and is underpowered
regardless of sample size. Discovered via an offline fixture containing **no
Jev API data**. The slope is retained as a descriptive figure.

### Amendment 2 — equal n per stratum

**Change:** equal n per stratum is now required, with documented
`subsample_to_equal_n` and a published bias floor
(`results/bias_floor.json` / `null_delta_ece_band`).

**Reason:** ECE bias varies with n; unequal strata would produce a positive
ΔECE from sample size alone — exactly the signal sought, for the wrong reason.

### Amendment 3 — effect size ΔECE = 0.09

**Change:** detectable effect size **ΔECE = 0.09** is pre-specified.

**Reason:** anchored to the gap between published benchmark results
(~ECE 0.05–0.07 at ~92% accuracy vs ~0.154 at ~63% accuracy), not chosen
ad hoc. See `jevbench/power.py` module docstring and
`results/power_analysis.json`.

### Amendment 4 — n from power analysis

**Change:** n per stratum set by the offline power analysis in
`results/power_analysis.json` (**chosen n = 200** per stratum; power ≥ 0.80
at ΔECE = 0.09; mean null FPR ≈ 0.074).

**Reason:** sample size is a design input fixed before seeing Jev
measurements. If labeling capacity cannot reach this n, that is a finding
about the study — options in order: keep two pooled strata (already the
plan), accept a larger pre-specified detectable effect, or add domains.
Do not proceed underpowered.

### Amendment commit hash (binding timestamp)

```
AMENDMENT_COMMIT=<filled by follow-up one-line commit after this file lands>
```

Until that line is replaced with a 40-character git SHA, treat the commit
that introduced this Amendments section as the binding timestamp. Verify with:

```bash
git log --oneline -- PREREGISTRATION.md
# confirm no scored runs/:  find runs -name raw.jsonl ! -path 'runs/offline_fixture/*'
```

---

## AI-usage statement

> Large language models were used for literature search, prose drafting,
> and code scaffolding under the author's direction. All experimental
> design, claims, analysis, and errors are the author's own. Every
> factual claim was verified against the cited primary source.
