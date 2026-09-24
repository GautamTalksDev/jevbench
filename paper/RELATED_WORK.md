# Related work search log

**Purpose:** Every novelty claim in `main.md` must point here.
Phrase novelty as: *"we are not aware of published work doing X"* — never
*"nobody has done this."*

**Search date:** 2026-09-24
**Searcher:** Amendment 9 (PROMPT Q)

## Databases / venues checked

- [x] Google Scholar / web search
- [x] Semantic Scholar (via arXiv/ACL)
- [x] arXiv (cs.LG, cs.CL, stat.ML)
- [ ] Zenodo
- [ ] OpenReview
- [x] ACL Anthology
- [x] GitHub issues / independent Jev benches

## Search terms (C1 / difficulty-stratified calibration)

- Does Jev's calibration survive difficulty, or only track accuracy
- jev-phishing-bench ECE 0.154
- ChaosNLI Jev calibration
- Choice vs Noul calibration Jev
- Stop Measuring Calibration When Humans Disagree

## Found (cite in paper)

| Work | Relevance | Notes |
|---|---|---|
| SamuelSacco/jev-exploration#1 (2026-09-17) | Origin of the flagship question | **Verified.** Issue titled "Does Jev's calibration survive difficulty, or only track accuracy?"; proposed ECE next to noise floor. Closed 2026-09-18 after a synthetic email-tier run concluding the calibration *curve* holds while mass relocates (different design from ChaosNLI). Credit as the origin of the question; our contribution is the pre-registered, equal-n, bias-corrected ChaosNLI test with Choice+Noul arms. |
| anisselbd/jev-phishing-bench | Low-accuracy ECE anchor | **Verified** via jevbench.xyz: Jev 62.6% accuracy, ECE 0.154 on 2,000 emails. |
| Baan, Aziz, Plank & Fernandez, EMNLP 2022 (arXiv:2210.16133) | Soft scoring / human disagreement | **Verified.** Shows ECE vs majority is theoretically problematic under inherent disagreement; ChaosNLI case study; argues divergence-to-humans metrics; examines Wang et al.'s claim that ECE substitutes for divergence. Motivates JSD/TVD secondary. |
| Kumar, Liang & Ma, NeurIPS 2019 (arXiv:1909.10155) | Plugin CE bias | Verified Amendment 8. |
| Roelofs et al., AISTATS 2022 (arXiv:2012.08668) | ECE_bin bias under calibration | Verified Amendment 8. |
| TypeSafe / jevaiguide Noul vs Choice docs | Primitive confound | **Verified.** Official docs: Choice and Noul are not interchangeable; same question can return different probabilities. Motivates dual-arm EXP-1 even without a third-party audit repo. |
| Alex Molas, "Jev can't be calibrated" (2026-09-23) | Primitive confound (secondary) | Notes a recent experiment where Noul calibrated better than Choice; does not name a repo. Cite only as a pointer, not for specific numbers. |
| MAPIE (scikit-learn-contrib) | Calibration + LTT risk control | Use, do not reimplement |
| netcal | ECE / recalibration | Use |
| Deferred Crispification (Doan Ngoc, Zenodo 22801506) | Position paper; not empirical refutation of Jev | Cite honestly |
| Jev in Practice / daf-jev (Zenodo 22816188) | Tools paper; confidence self-consistency claim | C2 target |
| Public benches: ickma2311, anisselbd, anessbelbati, themsquared, SamuelSacco | Prior measurements | Credit |

## Searched, not verified at Amendment 9 (do not cite numbers)

| Claim | Search result |
|---|---|
| jev-frontier-bench JSD 0.149 vs uniform 0.127 on ChaosNLI | No matching primary repo located 2026-09-24; 0.149 appears in ChaosNLI XLNet binned tables, not a Jev frontier bench. |
| Hugging Face `jev-bench` "crisp calibrated / disagree not" | `Praveenrajus/jev-bench` is ARC-Challenge MCQ, not ChaosNLI. |
| `jev-calibration-audit` Choice step-function / Noul 0.015–0.027 / 4–6× | No primary repo matching that name and those numbers located; dual arm motivated by TypeSafe docs instead. |
| ECE 0.05–0.07 at ~91.7–92% accuracy | Not re-traced to a primary repo; phishing ECE 0.154 @ 62.6% is the verified low-accuracy end. Threshold 0.09 stays pre-registered with disclosed justification. |

## Not found (as of search date)

_Document absences here after the search. Novelty claims may only reference this section._

Headline novelty is **not** "calibration degrades where people disagree" as a first observation. It is the first **pre-registered, equal-n, bias-corrected** test of that claim on ChaosNLI, plus the methods finding that naive stratified ECE comparisons are biased toward exactly that conclusion (Amendments 7–8).

## Prior Jev benchmarks to credit by name

- https://github.com/ickma2311/jev-baselines-eval
- https://github.com/anisselbd/jev-phishing-bench
- https://github.com/anessbelbati/jev-rerank-bench
- https://github.com/themsquared/jev-benchmark
- https://github.com/SamuelSacco/jev-exploration
