# Related work search log

**Purpose:** Every novelty claim in `main.md` must point here.
Phrase novelty as: *"we are not aware of published work doing X"* — never
*"nobody has done this."*

**Search date:** 2026-09-24 (PROMPT R reconciliation)
**Searcher:** Amendment 9 / PROMPT R

## Databases / venues checked

- [x] Google Scholar / web search
- [x] Semantic Scholar (via arXiv/ACL)
- [x] arXiv (cs.LG, cs.CL, stat.ML)
- [ ] Zenodo
- [ ] OpenReview
- [x] ACL Anthology
- [x] GitHub issues / independent Jev benches (READMEs read before citing)

## Search terms (C1 / difficulty-stratified calibration)

- Does Jev's calibration survive difficulty, or only track accuracy
- jev-phishing-bench ECE 0.154
- ChaosNLI Jev calibration
- Choice vs Noul calibration Jev
- Stop Measuring Calibration When Humans Disagree
- manjunathshiva/jev-frontier-bench
- Praveenrajus/jev-bench
- MohitSV/jev-calibration-audit
- OmarMujahid/jev-decision-bench

## Novelty (defensible claim only)

We are not aware of a **pre-registered, equal-n, bias-corrected, entropy-stratified**
test of soft ΔECE on ChaosNLI for Jev, with the search logged here. That is the
claim. Dual primitives (Choice + Noul) are **not** novel — see
`MohitSV/jev-calibration-audit` below.

---

## Prior answer to the same question (cite prominently)

### SamuelSacco/jev-exploration — synthetic email-tier run (2026-09-18)

**This is a prior answer to "does calibration survive difficulty?"** It is not
our ChaosNLI design, but it already answered the question on a different
corpus (synthetic email tiers + spam-eval transfer).

| Evidence | Permalink |
|---|---|
| Closing comment on issue #1 | https://github.com/SamuelSacco/jev-exploration/issues/1#issuecomment-5723851806 |
| Committed write-up | https://github.com/SamuelSacco/jev-exploration/blob/6161684f4917691deef056bc3efb6d2a9f810cd0/lab/tiers/FINDINGS.md |
| Commit that landed FINDINGS.md | https://github.com/SamuelSacco/jev-exploration/commit/6161684f4917691deef056bc3efb6d2a9f810cd0 |

**API state (fetched 2026-09-24 via `GET /repos/SamuelSacco/jev-exploration/issues/1`):**
`state=closed`, `closed_at=2026-09-18T01:52:33Z`, `closed_by=SamuelSacco`.

**Closing comment (verbatim gist):** accuracy tracks difficulty (97.5% → 90.0%,
non-overlapping intervals); ECE does not — bootstrap intervals on tier
differences include zero; the calibration *function* is invariant and the ECE
rise is mass relocating into the badly-calibrated middle. Full numbers in
`FINDINGS.md` (run `20260918T013027Z`, `jev-1.13.0`, 800 items, 4 tiers, 3
passes).

**UI conflict note:** a browser view of the same issue has been reported as
still showing **Open** with an empty activity section. The binding evidence we
cite is the closing-comment permalink and the committed `FINDINGS.md` above —
not a secondary tool summary. If a reviewer’s HTML view disagrees with the
API, reconcile against those two permalinks.

Credit issue #1 as the **origin of the flagship question** (opened 2026-09-17).
Our contribution is the pre-registered ChaosNLI test, not inventing the question.

---

## Found (cite in paper) — READMEs read before citing

| Work | Relevance | Notes (from primary README / FINDINGS only) |
|---|---|---|
| SamuelSacco #1 + FINDINGS.md | Prior answer to the same question | Permalinks above. Synthetic email tiers, not ChaosNLI. |
| [manjunathshiva/jev-frontier-bench](https://github.com/manjunathshiva/jev-frontier-bench) | ChaosNLI human-agreement arm | **Read 2026-09-24.** Jev vs five frontier LLMs, 200 decisions, OpenRouter 2026-09-19. On ChaosNLI ambiguity (50 items): JSD to 100-annotator split — Claude Fable 5.1 **0.043**, MiniMax **0.107**, Jev **0.149**; uniform 1/3 baseline **0.127** (beats Jev). Calls Omar Mujahid’s jev-decision-bench “the most thorough earlier comparison.” |
| [OmarMujahid/jev-decision-bench](https://github.com/OmarMujahid/jev-decision-bench) | Broad task bench (not ChaosNLI-stratified) | **Read 2026-09-24.** 49 tasks, 8 225 items, `jev-1.13.0`, 2026-09-18. Jev led/tied/trailed **29 / 13 / 7** vs gpt-5.6-luna; ECE **0.07** (Jev) vs 0.18 / 0.14 (luna / luna-low). Not a difficulty-stratified ChaosNLI ΔECE test. |
| [MohitSV/jev-calibration-audit](https://github.com/MohitSV/jev-calibration-audit) | Choice vs Noul + ChaosNLI votes | **Read 2026-09-24.** Exact-target audit of RLCD probabilities. Main findings (README): with no evidence, Choice puts a fair coin at **0.83–0.93** on “heads”; Choice is a **step function** on stated probabilities; Noul tracks the same targets within **0.015–0.027**; on human disagreement, normalized Noul is **4–6×** closer to the vote distribution zero-shot; Choice ranks disagreement and matches Noul after one per-task temperature. ChaosNLI + DICES-350 used as checks. **This already compares Choice against Noul on ChaosNLI vote distributions — dual primitives are not our novelty.** |
| [Praveenrajus/jev-bench](https://huggingface.co/datasets/Praveenrajus/jev-bench) (HF) | Broad System-One bench incl. ChaosNLI | **Read card 2026-09-24.** Multi-config Hub dataset (banking77, mmlu, arc_challenge, chaosnli, …). ChaosNLI listed as calibration-gold (soft_label = human vote shares). README claim: crisp/grounded decisions land accurate-and-calibrated; ordinal ratings and human-disagreement items do not; Jev confidence on ChaosNLI is flat. Cite as a prior ChaosNLI-on-Jev measurement surface, not as our stratified ΔECE design. |
| anisselbd/jev-phishing-bench | Low-accuracy ECE anchor | **Verified** via jevbench.xyz: Jev 62.6% accuracy, ECE 0.154 on 2,000 emails. |
| Baan, Aziz, Plank & Fernandez, EMNLP 2022 (arXiv:2210.16133) | Soft scoring / human disagreement | **Verified.** ECE vs majority problematic under inherent disagreement; motivates JSD/TVD secondary. |
| Kumar, Liang & Ma, NeurIPS 2019 (arXiv:1909.10155) | Plugin CE bias | Verified Amendment 8. |
| Roelofs et al., AISTATS 2022 (arXiv:2012.08668) | ECE_bin bias under calibration | Verified Amendment 8. |
| TypeSafe / jevaiguide Noul vs Choice docs | Primitive confound | Official docs: Choice and Noul are not interchangeable. |
| facebook/bart-large-mnli | Supervised in-domain reference only | Fine-tuned on MultiNLI. ChaosNLI-MNLI items are in-domain — **not** a zero-shot baseline. Report MNLI vs SNLI separately; never in the same table as Jev (Amendment 11). |
| knowledgator/gliclass-base-v1.0 | Local zero-shot-style control | Trained on synthetic zero-shot mix; card does not list MNLI/SNLI. Disclose synthetic training; do not silently substitute BART. |
| Deferred Crispification (Doan Ngoc, Zenodo 22801506) | Position paper | Cite honestly |
| Jev in Practice / daf-jev (Zenodo 22816188) | Tools paper | C2 target |
| Public benches: ickma2311, anisselbd, anessbelbati, themsquared, SamuelSacco | Prior measurements | Credit |

## Effect-size justification (0.09)

Verified low-accuracy public end: phishing ECE **0.154 @ 62.6%**. An earlier draft’s
“~ECE 0.05–0.07 at ~92% accuracy” was **not** re-traced to a primary repo;
threshold **0.09 stays** with that disclosure. Frontier-bench / decision-bench
ECEs above are different designs (not entropy-stratified soft ΔECE) and do not
re-justify the threshold by themselves.

## Not found (as of search date)

_Document absences here after the search. Novelty claims may only reference this section._

- No other pre-registered **equal-n + bias-corrected** ChaosNLI soft-ΔECE
  stratified test located.
- Dual-primitive comparison on ChaosNLI **was** found (`jev-calibration-audit`) —
  do not claim that as new.

## Prior Jev benchmarks to credit by name

- https://github.com/ickma2311/jev-baselines-eval
- https://github.com/anisselbd/jev-phishing-bench
- https://github.com/anessbelbati/jev-rerank-bench
- https://github.com/themsquared/jev-benchmark
- https://github.com/SamuelSacco/jev-exploration
- https://github.com/manjunathshiva/jev-frontier-bench
- https://github.com/OmarMujahid/jev-decision-bench
- https://github.com/MohitSV/jev-calibration-audit
- https://huggingface.co/datasets/Praveenrajus/jev-bench
