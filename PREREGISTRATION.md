# Pre-registration — exp1_difficulty_calibration

**Status:** LOCKED (amended — see Amendments)
**Experiment:** `exp1_difficulty_calibration`
**Task:** `datasets/chaosnli/`
**Pinned model:** `jev-1.13.0`
**Serving path:** `native`
**Locked at (UTC):** 2026-09-20T02:59:57+00:00
**Amendments dated (UTC):** 2026-09-22; 2026-09-23 (Amendments 6–8); 2026-09-24 (Amendments 9–11)

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
| `datasets/chaosnli/items.jsonl` | `1fbb1ab363004139b82431f8bf378ba4d43be57cd995c7e64d231770b1c71a19` |
| `datasets/chaosnli/labels.jsonl` | `efadb35ac7472d18d25ee325b0582d4632a393ea7e024b7ff09c5ce164fe84ad` |

Supporting lock files (not hashed by the runner, but fixed before any call):

| File | SHA-256 |
|---|---|
| `datasets/chaosnli/universe.jsonl` | `afde068d643bf5e6d915d428849adf27938cb827badd2d023837719af129e3ca` |
| `datasets/chaosnli/thresholds.json` | `145b2434eb95f9e1322e98ed7eeca4caaa60c6e888ab3d01963959ce09297e67` |
| `datasets/chaosnli/paraphrases.sha256` | `945d2ae0eb8dd0d5ce21b2f2a20095ea97ff4f28e37997aebe86943a84c5f590` (of local `paraphrases.jsonl`) |
| `datasets/chaosnli/paraphrase_ids.json` | (IDs only — see file) |

Machine-readable lock: `preregistration.lock.json`

- Items locked: **1600** (1500 primary + 100 frozen paraphrases)
- Labels: **1600**
- Primary tier counts: `{'easy': 750, 'hard': 750}` (paraphrases add 50 per tier and are excluded from ΔECE)
- Entropy thresholds (locked before any Jev call): `q_easy = 0.7357948629753385`, `q_hard = 1.1625318905614013`

**No hand-labelling kappa gate.** Agreement is the ChaosNLI 100-annotator
distribution. Entropy is the difficulty measure. Soft scoring uses
`label_dist`; hard scoring uses the majority in `label`.

---

## Description

Difficulty-stratified calibration on **ChaosNLI SNLI + MNLI** (3,113 items).
αNLI is excluded: it is a 2-way abductive task (observation-start / two
hypotheses / observation-end), not 3-way entailment–neutral–contradiction;
entropy is not on the same scale. Easy and hard are the lowest and highest
quartiles of per-item Shannon entropy of the 100-annotator label
distribution, then reduced to **n = 750** per stratum by
`subsample_to_equal_n` (seed 20260923).

**Primary endpoint:** soft top-label ΔECE =
ECE(high-entropy) − ECE(low-entropy). Soft correctness is the share of
annotators who chose the model's argmax; ECE bin accuracy is the mean of
that share. Hard scoring (argmax vs majority) is always reported beside it.

Sentence text is **not** in the repository. ChaosNLI is CC BY-NC 4.0; SNLI
is CC BY-SA 4.0; MNLI has mixed terms. This MIT tree ships item IDs +
annotator counts + a fetch script (`scripts/fetch_chaosnli.py`).

---

## Hypotheses & falsification rules

### H1 — primary (amended: soft ΔECE on ChaosNLI; Amendment 9 verdict)

**Statement:** Soft top-label ΔECE = ECE(high-entropy) − ECE(low-entropy) is
distinguishable from zero in the literature direction (hard less
calibrated), with equal n = 750 per stratum and occupancy reported on every
ECE. Primary inference uses the **parametric bias-corrected** test
(Amendments 8–9), not the bootstrap interval alone.

**Falsified when / verdict (Amendment 9):**
- **tracks** = corrected ΔECE ≥ 0.09 AND parametric p < 0.05
- **holds** = one-sided test rejects ΔECE ≥ 0.09 at level 0.05
  (simulate under true ΔECE = 0.09) AND corrected ΔECE ≤ 0.02
- else **inconclusive**

Run identical analysis on **Choice** and **normalized three-Noul** arms
from the same call; report both; neither is "the" result. Never interpret
an underpowered inconclusive as a substantive finding.

### H1a — descriptive (retained)

**Statement:** ECE is approximately flat across difficulty after accounting
for bin occupancy.

**Falsified when:** Descriptive only. Never the powered claim.

### H1b — descriptive (retained)

**Statement:** ECE rises as accuracy falls (calibration tracks accuracy).

**Falsified when:** Descriptive only. Interval crossing zero is inconclusive.

---

## Metrics

- **Primary:** soft top-label ΔECE (high-entropy vs low-entropy), uniform and
  quantile binning (M=10), stratified two-sample bootstrap CI (10 000
  resamples), equal n enforced
- **Reported:** hard top-label ΔECE (argmax vs majority). **Hard scoring puts
  label noise in the hard stratum only, which inflates ΔECE in the direction
  of the hypothesis.** Soft scoring is primary for that reason.
- ECE (uniform / quantile) with per-bin occupancy — netcal-cross-checked
- Bias floor / null band: `results/bias_floor.json`
- MCE, Brier, reliability diagrams per stratum
- Accuracy per stratum
- Contamination: original vs frozen paraphrase accuracy (~100 items), every arm
- Latency p50/p95/p99 (never mean)
- Paired bootstrap CIs when comparing two models on the **same** items

---

## Decision rules

- Primary verdict uses the **Amendment 9 parametric rule** on corrected soft
  ΔECE (Choice arm stamped on the certificate; Noul arm reported beside it).
- Always report raw and corrected ΔECE. The certificate page displays
  `result.verdict` from the harness — it must not recompute a zone from the
  interval.
- Always report hard ΔECE beside soft, with the label-noise caveat above.
- Secondary: JSD and TVD from Jev's distribution to the human distribution,
  per stratum, both arms, compared to a uniform-guess baseline.
- Never report ECE without bin occupancy.
- Calibration metrics use probabilities only — never confidence.
- Do not run EXP-1 underpowered relative to `results/power_asymmetric.json`
  / `results/soft_bias_correction.json`.
- Report the BCa coverage diagnostic in `results/bca_diagnostic.json`: the
  data support the **ΔECE statistic** explanation (mean coverage near
  nominal; ΔECE BCa undercovers). Do not claim a 95% ΔECE interval.
- A large accuracy drop from original wording to the frozen paraphrase,
  reported for every arm, is a memorisation signal — not a calibration finding.

---

## Sample size

- **n per stratum:** **750** (entropy pools → `subsample_to_equal_n`)
- **Total primary items:** 1500
- **repeats:** 3 (scored EXP-1; feeds EXP-6)
- **Bootstrap (scored):** n_boot = 10 000; report **BCa and percentile**
- **Asymmetric-noise power (n=750, 400 trials):** soft power = 1.00; hard
  power ≈ 1.00 (`results/power_asymmetric.json`). Soft scoring remains
  primary. Soft null is the **Beta–Binomial** null of Amendment 7 plus the
  **parametric ECE bias correction** of Amendment 8
  (`results/soft_bias_correction.json`). Operating κ = 20. Raw soft null
  FPR ≈ 0.10–0.11 was structural (upper-tail only; null mean raw
  ΔECE ≈ **0.0033**); corrected parametric p-value FPR ≈ **0.051** with
  power 1.00. Always report raw and corrected ΔECE. Hard null FPR =
  **0.085** (hard scoring puts label noise only in the hard stratum — an
  artifact). v1/v2 soft-null history: see Amendments 7–8.
- **BCa diagnostic (5000 trials):** stratum-mean coverage ≈ 95.0% (percentile
  and BCa); ΔECE coverage percentile ≈ 92.3%, BCa ≈ 89.2%. Supported
  explanation: `delta_ece_statistic` (ECE binning non-smoothness, not a bug).

---

## Stopping rule

Stop after the pre-registered powered sample is scored with the stated
repeats. Do not add items because interim ΔECE looks flat or rising. Do not
move the entropy thresholds after seeing model output. New items require a
new preregistration version and exploratory labelling of prior runs.

---

## Contamination guard

1. Labels and strata are fixed at lock time (hashes above).
2. `jevbench verify-labels` fails if `labels.jsonl` changes after the first
   scored run under `runs/`.
3. Adjudicating a label after seeing model output is a protocol breach.
4. SNLI / MNLI have been public since 2015 / 2018. Compare every arm's
   accuracy on ChaosNLI items vs the frozen paraphrase subset (~100 items,
   reworded before any run). A large drop suggests memorisation.

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
| 1–4 (this section) | 2026-09-22 | `70be25f7baf91daa748cec11c38f26f17315b17c` |
| 5 (F1 n=750; BCa; kappa) | 2026-09-22 | recorded in-file before paid calls |
| 6 (ChaosNLI) | 2026-09-23 | see `AMENDMENT_6_COMMIT` below |
| 7 (soft null Beta–Binomial) | 2026-09-23 | see `AMENDMENT_7_COMMIT` below |
| 8 (structural ECE bias correction) | 2026-09-24 | see `AMENDMENT_8_COMMIT` below |
| 9 (Choice+Noul; parametric verdict; JSD/TVD) | 2026-09-24 | see `AMENDMENT_9_COMMIT` below |
| 10 (secondaries S1–S3; local baselines; budget) | 2026-09-24 | see `AMENDMENT_10_COMMIT` below |
| 11 (BART reference; local R=1; WSL2 memory) | 2026-09-24 | see `AMENDMENT_11_COMMIT` below |

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

**Reason (Amendment 9 re-trace):** the verified low-accuracy public anchor is
**jev-phishing-bench** (62.6% accuracy, ECE **0.154**). SamuelSacco issue #1
contrasted that with jev-spam-eval at 98.3% accuracy (well calibrated at the
extremes). An earlier draft also cited "~ECE 0.05–0.07 at ~92%"; that
specific high-accuracy ECE was **not re-traced to a primary repo** and is
withdrawn as justification. The threshold stays 0.09 (locked before any Jev
call); the justification text is the honest one. See `jevbench/power.py`
module docstring and `results/power_analysis.json`.

### Amendment 4 — n from power analysis

**Change:** n per stratum set by the offline power analysis in
`results/power_analysis.json` (**chosen n = 200** per stratum; power ≥ 0.80
at ΔECE = 0.09; mean null FPR ≈ 0.074).

**Reason:** sample size is a design input fixed before seeing Jev
measurements. If labeling capacity cannot reach this n, that is a finding
about the study — options in order: keep two pooled strata (already the
plan), accept a larger pre-specified detectable effect, or add domains.
Do not proceed underpowered.

### Amendment 5 — F1 hardened power → n = 750; BCa; kappa gate

**Change:** After adding label_noise / difficulty_sd / tier_leakage to the
simulator (Prompt F1), the pessimistic corner requires **n = 750** per
stratum for power ≥ 0.80 at ΔECE = 0.09. Scored analysis reports **BCa and
percentile** intervals. A 15% double-label agreement pass with Cohen's
κ ≥ 0.6 is required before paid runs; measured disagreement is fed back
into the F1 sweep.

**Reason:** the clean-simulator n=200 answered an easier question than the
study runs. Coverage study (`results/coverage.json`) shows empirical
coverage ~92% — do not claim 95%. If labelling capacity stays below 750,
the study is **underpowered** and must not be scored as a finding.

### Amendment 6 — switch EXP-1 to ChaosNLI (soft scoring primary)

**Change:**

1. **Dataset.** EXP-1 task is `datasets/chaosnli/` (ChaosNLI SNLI + MNLI =
   3,113 items). αNLI is excluded because it is a different task format
   (2-way abductive, not 3-way NLI). Sentence text is not redistributed;
   the repo ships item IDs, annotator counts, entropy, and
   `scripts/fetch_chaosnli.py` (ChaosNLI CC BY-NC 4.0; SNLI CC BY-SA 4.0;
   MNLI mixed terms — incompatible with shipping text under this MIT tree).
2. **Difficulty.** Per-item entropy of the 100-annotator distribution.
   Easy = lowest entropy quartile (`entropy ≤ q_easy`); hard = highest
   (`entropy ≥ q_hard`). Thresholds fixed from population quantiles
   **before any Jev call** and committed in
   `datasets/chaosnli/thresholds.json`. Equal n = 750 via
   `subsample_to_equal_n` (seed 20260923).
3. **Scoring.** Soft scoring is primary (`correct_i` = annotator share on
   the model's argmax; bin accuracy = mean soft correctness). Hard scoring
   (argmax vs majority) is always reported. Hard scoring puts label noise
   in the hard stratum only, which inflates ΔECE in the direction of the
   hypothesis.
4. **Simulator.** Asymmetric label noise (flip rate = f(entropy)) re-run at
   n=750 under both scoring rules (`results/power_asymmetric.json`).
5. **BCa diagnostic.** 5000-trial null on the mean of a stratum vs ΔECE,
   percentile and BCa side by side (`results/bca_diagnostic.json`). The
   data support the **ΔECE statistic** explanation: mean coverage is near
   nominal; ΔECE BCa undercovers.
6. **Contamination.** SNLI/MNLI public since 2015/2018. Frozen paraphrase
   subset (~100 items) locked before any run; compare accuracy for all arms.
7. **Kappa gate removed** for this task: the 100-annotator distribution
   replaces double-labelling. The hand-labelling ceiling of 500 no longer
   applies.

**Reason:** self-labelled support tickets could not reach the powered n
without months of labelling. ChaosNLI already supplies 100 labels per item
and an entropy that is an external difficulty measure, fixed before any
model call.

**No Jev output had been observed.** This amendment, the entropy
thresholds, the paraphrase freeze, and the soft/hard scoring rules were
written from ChaosNLI metadata and offline simulators only. Verify with:

```bash
git log --oneline -- PREREGISTRATION.md
find runs -name raw.jsonl ! -path 'runs/offline_fixture/*'
```

### Amendment 7 — soft-scoring null → Beta–Binomial (κ sweep)

**Change:**

1. **Definition.** Under soft scoring, correctness **is** the annotator
   share of the model's pick — there is no Bernoulli label draw.
   Calibration means \(E[\mathrm{share}\mid p]=p\), **not** share \(= p\)
   per item. The soft null therefore draws, per item:
   - \(p_i\) from the certificate specimen top-probability shape (seed
     20260919), with **separate** easy and hard pools of 750 (same
     generative sketch as `arena/index.html` `specimen()`; not Jev
     output);
   - true share \(s_i\sim\mathrm{Beta}(\kappa p_i,\kappa(1-p_i))\);
   - observed share \(=\mathrm{Binomial}(100,s_i)/100\);
   - soft correctness = observed share; reported confidence = \(p_i\)
     under the null.
2. **κ range.** Sweep \(\kappa\in\{5,10,20,50,200\}\) over 2000 trials
   each (SE ≈ 0.007); see `results/soft_null_kappa.json`.
3. **Operating κ.** **κ = 20** (midpoint of the realistic band 10–50).
   Empirical null FPR ≈ **0.101** at κ=20 (and ≈ 0.10–0.11 across the
   band) — outside [0.03, 0.08]; reported honestly under the same rule
   as hard-scoring ~92% coverage. Power at n=750 for ΔECE=0.09 is
   **1.00** at every κ, including the pessimistic realistic end.
4. **Proof of cause.** \(\kappa\to\infty\) with no binomial step
   reproduces FPR = 0.000, confirming the v2 constant-confidence soft
   null was **degenerate** (share locked to \(p\) per item ⇒ ECE≈0 with
   ~no variance in both strata), not that the ECE estimator is
   conservative. Methods one-liner: when soft correctness equals
   reported confidence with no item-level scatter, the soft ΔECE null is
   degenerate and the false-positive rate is exactly zero.

**Reason:** Amendment 6's soft-calibrated null set top probability equal
to the expected share with constant confidence, which forced FPR=0.000
and blocked a trustworthy soft primary endpoint. Item-level Beta–
Binomial scatter restores a non-degenerate null; separate easy/hard
specimen \(p\)-pools match the EXP-1 design (different confidence shapes
across strata, both calibrated under the null).

**No Jev output observed.** Offline simulators and the certificate
specimen generator only. Verify with:

```bash
git log --oneline -- PREREGISTRATION.md
find runs -name raw.jsonl ! -path 'runs/offline_fixture/*'
```

### Amendment 8 — parametric correction for structural ECE bias (PROMPT P)

**Change:**

1. **Diagnosis (κ=20, 2000 trials).** Raw soft null FPR ≈ **0.112** is
   **entirely upper-tail** (lower-tail = 0.000): the bias favours H1.
   Null mean raw ΔECE ≈ **0.0033**. Place this number next to the
   detectable effect (0.09) and the "holds" zone (≤0.02) in the paper.
   Flat FPR across κ showed the inflation is structural (bin occupancy
   differs between easy/hard confidence shapes), not item-level scatter.
2. **Correction.** At the observed top probabilities of each stratum,
   estimate \(E0=\mathbb{E}[\mathrm{ECE}\mid p]\) by 2000 calibrated
   Beta–Binomial soft draws. Report
   \(\mathrm{corrected\ }\Delta\mathrm{ECE}=(ECE_h-E0_h)-(ECE_e-E0_e)\)
   **beside** raw ΔECE — never corrected alone
   (`results/soft_bias_correction.json`).
3. **Test.** Primary: two-sided p-value vs the simulated null
   distribution of ΔECE at the observed probabilities (correct size by
   construction). Interval: percentile bootstrap of corrected ΔECE
   (E0 fixed at observed \(p\); empirically conservative). At κ=20,
   p-value FPR ≈ **0.051** ∈ [0.03, 0.08]; power = 1.00 at ΔECE=0.09.
4. **Sensitivity.** Corrected null mean ≈ 0 and FPR ≈ 0.05 across
   κ∈{10,20,50}; alternative corrected means ≈ 0.09. Verdict is **not**
   κ-dependent under this simulator; still report the κ table because κ
   is not identifiable from calibrated data alone.
5. **Related work (verified before citing).** Kumar, Liang & Ma
   (NeurIPS 2019): plugin binned CE estimators are biased; debiased
   estimator improves sample complexity for *squared* CE. Roelofs et al.
   (AISTATS 2022): equal-width ECE_bin biased even under perfect
   calibration; equal-mass / ECE_debias / ECE_sweep reduce bias. We keep
   plugin ECE as the endpoint and subtract parametric \(E0\); see
   `paper/RELATED_WORK.md`.

**Reason:** Amendment 7 made the soft null non-degenerate, but equal-n
does not cancel ECE's dependence on how confidence mass is spread across
bins. Easy packs into a few high-\(p\) bins; hard spreads. Under a true
null, \(E[\Delta\mathrm{ECE}]\neq 0\), and the excess FPR is one-sided
toward H1. The "tracks accuracy" verdict (≥0.09) is mostly protected; the
interval and the margin to the ≤0.02 "holds" zone are not.

**No Jev output observed.** Offline only.

### Amendment 9 — Choice+Noul arms; parametric verdict; JSD/TVD (PROMPT Q)

**Change:**

1. **Primitive arms.** EXP-1 asks, in the **same** System One call as the
   Choice `relation` question, three Nouls
   (`noul_entailment` / `noul_neutral` / `noul_contradiction`), normalized
   to sum to 1. Run the identical soft ΔECE + bias-corrected analysis on
   both arms. Report both; neither is "the" result. Motivation: TypeSafe's
   own docs that Choice and Noul are not interchangeable, and
   `MohitSV/jev-calibration-audit` (README read 2026-09-24) already comparing
   Choice vs Noul against ChaosNLI vote distributions. Dual primitives are
   **not** novelty — they are a confound control. See `paper/RELATED_WORK.md`.
2. **Verdict rule** (primary = parametric corrected test):
   - tracks = corrected ΔECE ≥ 0.09 AND p < 0.05
   - holds = one-sided reject ΔECE ≥ 0.09 at 0.05 (simulate under true
     ΔECE = 0.09) AND corrected ΔECE ≤ 0.02
   - else inconclusive
   Size re-verified in `results/verdict_size.json` (p-value FPR ≈ 0.04;
   one-sided reject-≥0.09 size at the boundary ≈ 0.05; tracks under null
   ≈ 0; holds under true Δ=0.09 ≈ 0). **Composite tracks power at exactly
   ΔECE=0.09 is ~0.5 by construction** (requiring corrected ≥ 0.09 when the
   true mean is 0.09 is a median split); the parametric p-value test itself
   has power ≈ 1.0 at that effect. Do not claim 80% power for the composite
   tracks gate at the boundary. The certificate page displays
   `result.verdict` from the harness and must not recompute a zone from the
   interval (`result.p_value`, `result.delta_raw`, `result.delta_corrected`
   stamped beside it).
3. **Secondary.** JSD and TVD from Jev's distribution to the human
   distribution, per stratum, both arms, vs a uniform-guess baseline
   (motivated by Baan et al., EMNLP 2022 / arXiv 2210.16133 — verified).
4. **Related work (PROMPT R permalinks).** SamuelSacco #1 is the origin of
   the question; the synthetic email-tier answer lives at closing comment
   https://github.com/SamuelSacco/jev-exploration/issues/1#issuecomment-5723851806
   and `lab/tiers/FINDINGS.md` @ `6161684f`. Primary ChaosNLI-on-Jev sources
   read before citing: manjunathshiva/jev-frontier-bench, OmarMujahid/
   jev-decision-bench, MohitSV/jev-calibration-audit, HF Praveenrajus/jev-bench
   — see `paper/RELATED_WORK.md`. Effect-size 0.09: verified low-accuracy end
   is phishing ECE 0.154 @ 62.6%; the earlier "~0.05–0.07 at ~92%" figure was
   not re-traced — threshold stays with disclosure.
5. **Composite tracks power curve.** At true ΔECE = 0.09, composite tracks
   power ≈ 0.52 by construction (median split). Curve across
   {0.09, 0.11, 0.13, 0.15} in `results/tracks_power_curve.json`.

**Reason:** The landscape moved while the harness was built. The core
question already has a prior synthetic answer (SamuelSacco FINDINGS). Descriptive
ChaosNLI-on-Jev reports and Choice-vs-Noul audits exist. Running Choice alone
would confound model with primitive. The bootstrap interval's near-100%
coverage made the old interval-based "holds" rule asymmetrically hard. This
amendment repositions the paper as the first **pre-registered, equal-n,
bias-corrected stratified** test we are aware of (search documented) — not
as the first dual-primitive or first ChaosNLI-on-Jev look.

**No Jev output observed.** Offline only.

### Amendment 10 — secondary analyses; local baselines; budget guard (PROMPT S)

**Change:**

1. **Baselines (zero paid cost).** Drop Haiku / OpenAI adapter baselines.
   Baselines are local open-weight via `clients/prefill.py` (logprob-derived,
   single constrained token): **Qwen2.5-1.5B-Instruct**, plus **GLiClass**.
   A ~7B instruct baseline is **not** used on the study host (WSL2 x86_64,
   Intel i7-13620H, ~10 GB RAM, no NVIDIA GPU). Frontier LLM comparison is
   **out of scope**; cite `manjunathshiva/jev-frontier-bench`. Remove
   `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` from the environment; only
   `TYPESAFE_API_KEY` is needed for scored Jev calls.
2. **Repeats.** EXP-1 uses **R = 10** repeats per item (feeds S1/S2).
3. **Secondary analyses (no extra calls; exploratory unless named primary):**
   - **S1 Ambiguity detection.** AUROC and Spearman of three Jev signals
     against human vote entropy: (1) \(1 -\) top probability, (2) JSD between
     Choice and normalised three Nouls in the same call, (3) repeat
     instability (flip rate + mean TVD across R=10). Paired bootstrap
     comparisons between signals. Same where defined for local baselines.
   - **S2 Stability reconciliation.** Flip rate vs human entropy, not only
     vs distance from the decision boundary.
   - **S3 Temperature scaling.** Fit one temperature on a stratified random
     half; evaluate ECE and JSD on the other half; swap halves; average.
     Report Choice and Noul arms separately.
4. **Budget guard (code).** `budget.json` ledger appends every call's
   `input_tokens` and computed USD. Hard caps: pilot **$0.05** / run,
   EXP-1 **$1.00** / run, global spendable **$2.50** (50% of a $5 credit
   reserved and untouchable). Runner refuses to START if projected cost
   exceeds remaining room; STOPS mid-run on hitting a cap (resumable
   `raw.jsonl`). Max 3 retries; **never retry HTTP 4xx**. Dry-run prints
   projected cost; live runs require `--confirm`.
5. **Key hygiene.** `.env` gitignored; gitleaks pre-commit + CI; rotate
   TypeSafe key after the study. Never paste keys into chat or commits.

**Reason:** Paid frontier baselines are redundant with published
jev-frontier-bench and burn the credit that should protect a re-run.
Ambiguity-detection secondaries use signals already in the EXP-1 calls and
answer a sharper, pre-registered question. A hard spend guard prevents a
retry loop from consuming the reserve.

**No Jev output observed at amendment time.** Offline only. Any later code
change prompted by pilot output on **excluded mid-entropy items only** is
still this amendment's intent; record the pilot run id in the commit message.

### Amendment 11 — baseline labelling; local repeats=1; WSL2 memory (PROMPT S follow-up)

**Change:**

1. **BART-large-MNLI is a supervised in-domain reference, not a zero-shot
   baseline.** It is fine-tuned on MultiNLI; ChaosNLI-MNLI items come from
   MNLI's development set. New client arm `bart_mnli_ref` (`role:
   supervised_in_domain_reference`). Report MNLI and SNLI items separately.
   **Do not** put BART in the same comparison table as Jev. GLiClass stays a
   separate arm: trained on synthetic zero-shot mix
   (`MoritzLaurer/synthetic_zeroshot_mixtral_v0.1`). **The model card doesn't
   list MNLI or SNLI** (verified) — that is not a claim that it was not
   trained on them. Never silently fall back from GLiClass to BART.
2. **Local repeats = 1.** Prefill / GLiClass / BART / trivial use
   `repeats: 1` (greedy scoring is expected-deterministic). Jev keeps R=10
   for S1/S2. Determinism check: 50 mid-entropy items × 3 repeats
   (`scripts/run_local_determinism.py` → `results/local_determinism.json`).
3. **WSL2 memory.** Runner processes **one client at a time** and calls
   `release()` between local models. Load Qwen2.5-1.5B and BART never
   together. If OOM, raise WSL2 memory in `.wslconfig` rather than shrinking
   the model. Expect multi-hour CPU overnight for 1,500 local items.

**Reason:** Mislabeling BART as zero-shot would overclaim. Repeating
deterministic locals ×10 wastes hours. Sequential load avoids OOM on ~10 GB.

**No Jev output observed.** Offline only.

### Amendment commit hash (binding timestamp)

```
AMENDMENT_COMMIT=70be25f7baf91daa748cec11c38f26f17315b17c
AMENDMENT_6_COMMIT=bf03af91ecd08efa927cea9d1a8eb5adfc941634
AMENDMENT_7_COMMIT=3891afac56818bed1dc64850d9775e65d64f613c
AMENDMENT_8_COMMIT=9828641f73666c51a6ced232726049b7ff19e141
AMENDMENT_9_COMMIT=6011c75818f851ea99b495776edb7f2be0a8933d
AMENDMENT_10_COMMIT=755be90731f71dcfdd503fc9668af6a145b90f71
AMENDMENT_11_COMMIT=49c13f7debcede5d3f7f27c05551c906aa0ede27
```

Verify with:

```bash
git log --oneline -- PREREGISTRATION.md
# confirm no scored runs/:  find runs -name raw.jsonl ! -path 'runs/offline_fixture/*'
```

---

## AI-usage statement

> Large language models were used for literature search, prose drafting,
> and code scaffolding under the author's direction. All experimental
> design, claims, analysis, and errors are the author's own. Every
> factual claim was verified against the cited primary source. The ~100
> contamination paraphrases were **hand-authored** for this study and frozen
> before any model run. Sentence text is not redistributed; only
> `paraphrases.sha256` and item IDs are public (see `paper/METHODS_NOTES.md`
> and `docs/HISTORY_REWRITE.md`).
