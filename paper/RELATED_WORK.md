# Related work search log

**Purpose:** Every novelty claim in `main.md` must point here.
Phrase novelty as: *"we are not aware of published work doing X"* — never
*"nobody has done this."*

**Search date:** _TBD_
**Searcher:** _TBD_

## Databases / venues checked

- [ ] Google Scholar
- [ ] Semantic Scholar
- [ ] arXiv (cs.LG, cs.CL, stat.ML)
- [ ] Zenodo
- [ ] OpenReview
- [ ] ACL Anthology

## Search terms (C3 / workflow risk control)

- conformal risk control structured decisions
- selective prediction compositional workflows
- Learn-Then-Test multi-question
- calibration of composed classifiers
- conformal prediction System One / typed decisions
- MAPIE LLM-as-Judge risk control

## Search terms (C1 / difficulty-stratified calibration)

- difficulty stratified calibration
- ECE vs accuracy neural classifiers
- reliability diagrams class difficulty

## Search terms (C2 / stability)

- prediction stability confidence vs label
- run-to-run variance calibrated classifiers
- Jev consistency / self-consistency

## Found (cite in paper)

| Work | Relevance | Notes |
|---|---|---|
| MAPIE (scikit-learn-contrib) | Calibration + LTT risk control | Use, do not reimplement |
| netcal | ECE / recalibration | Use |
| Kumar, Liang & Ma, NeurIPS 2019 (arXiv:1909.10155) | Plugin binned CE estimators are biased; debiased estimator improves sample complexity for *squared* CE (O(B)→O(√B)); App. G heuristic for ℓ1 ECE | Verified against PDF/HTML 2026-09-23. We cite for bias of plugin ECE, not as claiming our parametric E0 is their estimator. |
| Roelofs, Cain, Shlens & Mozer, AISTATS 2022 (arXiv:2012.08668) | Equal-width ECE_bin biased even under perfect calibration (BBC); equal-mass lower bias; ECE_debias / ECE_sweep recommended | Verified against PMLR/arXiv 2026-09-23. Primary endpoint stays plugin ECE with parametric E0 correction; always report raw beside corrected. |
| Deferred Crispification (Doan Ngoc, Zenodo 22801506) | Position paper; not empirical refutation of Jev | Cite honestly |
| Jev in Practice / daf-jev (Zenodo 22816188) | Tools paper; confidence self-consistency claim | C2 target |
| Public benches: ickma2311, anisselbd, anessbelbati, themsquared, SamuelSacco | Prior measurements | Credit |

## Not found (as of search date)

_Document absences here after the search. Novelty claims may only reference this section._

## Prior Jev benchmarks to credit by name

- https://github.com/ickma2311/jev-baselines-eval
- https://github.com/anisselbd/jev-phishing-bench
- https://github.com/anessbelbati/jev-rerank-bench
- https://github.com/themsquared/jev-benchmark
- https://github.com/SamuelSacco/jev-exploration
