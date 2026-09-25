# JEV TEARDOWN. Build Spec

Companion to: `jev-teardown-script.md`  
Scope: what to build, Cursor prompts, verified API surface  
Verified: 19 September 2026 · **Part 5 refinements from FINAL_PLAN applied**

Authoritative API surface lives in repo root: [`API_REFERENCE.md`](../API_REFERENCE.md).  
Constitution: [`CONSTITUTION.md`](../CONSTITUTION.md) (includes rules 8 to 10). 
Architecture: [`ARCHITECTURE.md`](../ARCHITECTURE.md).

---

## What to build / not build

| Tier | Build? |
|---|---|
| TypeSafe Playground | Use for exploration; never cite as evidence |
| **jevbench** headless harness | **YES**, the credential |
| **jev-arena** small local replay UI | YES, small; never produces unscored numbers |

**Do not build:** auth, database, hosted service, realtime collab, custom charting libs.

**Rule:** Arena only reads harness JSONL. Live mode → `runs/demo/` labelled DEMO.

---

## PROMPT 0. Constitution (done at scaffold)

See `CONSTITUTION.md`. Setup targets: `pyproject.toml` (Python 3.11, deps including
**netcal**, **mapie**), directory tree, Makefile (`install`, `test`, `reproduce`,
`charts`, `lint-claims`), MIT LICENSE.

Stop after scaffold before clients, then proceed prompt-by-prompt.

---

## PROMPT 4. Metrics (REPLACE, libraries only)

```
Build jevbench/metrics.py.

CRITICAL: do NOT hand-roll calibration mathematics.
Dependencies: netcal, mapie, scikit-learn, scipy.

ARCHITECTURE: thin wrappers + validation suite.

1. CALIBRATION — wrap netcal.
   - ECE, MCE via netcal.metrics
   - Reliability curves via netcal
   - ONLY original: ece_with_occupancy() → netcal ECE + per-bin counts
   - Support uniform AND quantile binning; report both

2. RISK CONTROL — wrap MAPIE (EXPERIMENT 7).
   - Conformal prediction + Learn-Then-Test on COMPOSED multi-question decisions
   - Read current MAPIE LLM-as-Judge / risk-control docs before coding
   - Run exchangeability checks; if they fail, THAT IS A RESULT — report, don't pretend

3. VALIDATION — tests/test_agreement.py
   - Synthetic calibrated + miscalibrated fixtures
   - Agree with netcal/sklearn to 1e-9
   - Output goes in paper appendix

4. OUR CODE only where libraries don't cover:
   - automation_at_accuracy(probs, labels, target=0.90)
   - risk_coverage_curve, aurc
   - primitive_disagreement_rate
   - negation_sum_distribution
   - stability metrics (stability.py)
   - paired_bootstrap(a, b, statistic, n=10_000)
   - mcnemar

5. GUARD: confidence arrays → TypeError on calibration functions.
   Calibration runs on probabilities/noul ONLY.
```

---

## PROMPT 8. Stability EXP-6 (NEW)

```
Build jevbench/stability.py + experiments/exp6_stability.yaml.

MEASURE SEPARATELY:
  - LABEL stability (argmax / threshold flips)
  - CONFIDENCE stability (value drift)
  - DISTRIBUTION stability (TV + KL per item)

STRATIFY BY: difficulty tier, |p - threshold|, serving path, resolved model version.

PROTOCOL:
  - N >= 300, >= 10 repeats, identical request bytes
  - Interleave repeats over time (not back-to-back)
  - Log model field every call; mid-run alias change is a finding
  - Same protocol on temp-0 LLM baseline

OUTPUT: flip rate + CI, confidence drift, boundary-distance plot,
explicit statement which prior result our data supports and when.
```

---

## PROMPT 9. Paper artifact (NEW)

```
Build paper/ (main.md, RELATED_WORK.md, figures→results symlink,
appendix/{validation,preregistration,label_guide}.md, CITATION.cff).

Rules enforced by scripts/lint_claims.py:
- Every Jev fact cites primary source or our measurement
- Novelty = "we are not aware..." → RELATED_WORK.md
- Every number → run_id + raw.jsonl line
- Architecture / RLCD internals = UNKNOWN (lint fails on asserts)
- Limitations written BEFORE Results
```

---

## Remaining prompts (order)

After metrics + stability:

1. Clients: `jev.py`, `adapter.py`, `trivial.py` (prefill later for EXP-3)
2. `runner.py` + manifest/raw/scored schema
3. EXP-1..5 YAML + analysis scripts
4. `charts.py` (video resolution)
5. Arena replay UI (budget half day; cut if over)

---

## Hand-write these two

`metrics.py` and `stability.py` are where a subtle bug silently invalidates the
paper. Prefer authoring them directly rather than via autocomplete prompts.
