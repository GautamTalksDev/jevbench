# JEV TEARDOWN — Research dossier, test plan & video script

**Subject:** Jev / System One — TypeSafe AI  
**Research cutoff:** 19 September 2026  
**Full dossier:** pasted into the planning session that created this repo.
**Superseding plan:** [`FINAL_PLAN.md`](FINAL_PLAN.md) (paper first; no product).

## Lane (open)

Difficulty-stratified calibration (ECE vs accuracy across tiers) + confidence-vs-label
stability contradiction + workflow conformal risk control. Not another me-too ECE blog.

## Non-negotiable test rules

1. Pre-register before first API call  
2. Labels before model output  
3. Pin `jev-1.13.0`; log resolved `model`  
4. Non-AI baseline on every task  
5. ≥3 repeats (EXP-6: ≥10 interleaved)  
6. Commit raw JSONL  
7. Latency: p50/p95/p99 + geography + serving path  
8. Bootstrap 10k; interval crossing zero = inconclusive  

## Experiments

| # | Name |
|---|---|
| 1 | Difficulty-stratified calibration (flagship) |
| 2 | Decomposition vs direct |
| 3 | Moat control (prefill + GLiClass) |
| 4 | Structural invariants at scale |
| 5 | Adversarial & context rot |
| 6 | Stability: confidence vs label |
| 7 | Workflow-level conformal risk control |

## Video

~38–44 min, chaptered; **repo + Zenodo DOI ship before the video**.  
Preferred title if EXP-1 lands: *Five People Benchmarked Jev And Got Different Answers. I Found Out Why.*

## Primary sources (verify before asserting)

- https://typesafe.ai/blog/introducing-system-one-models-and-jev  
- https://typesafe.ai/blog/bitterest-lesson  
- https://docs.typesafe.ai  
- https://docs.typesafe.ai/model-jaggedness/jev-1.13  
- https://news.ycombinator.com/item?id=49717558  
- https://github.com/typesafe-ai/system-one-adapter-python  
- Benchmarks: ickma2311, anisselbd, anessbelbati, themsquared, SamuelSacco  

## Open questions (answer to unlock Day 1)

1. Task domain for EXP-1 (4 tiers, ≥300 labels)  
2. Public repo under your name?  
3. Baseline LLM (Haiku/Flash-class recommended)  
4. Serving path (native recommended)  
5. Local open-weight + logprobs for EXP-3?  
6. Author `metrics.py` / `stability.py` directly? (recommended: yes)
