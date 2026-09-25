# FINAL PLAN. Paper first (supersedes product pitch)

**Verified:** 19 September 2026  
**Companions:** `jev-teardown-script.md`, `jev-build-spec.md`

## Retraction

Do **not** build a calibrate→gate→escalate→drift product. MAPIE/netcal cover
calibration; Argilla/Label Studio/etc. cover review queues; `daf-jev`
(Zenodo 22816188) already ships the toolkit pitch.

**The research paper is the play.** Gap: independent, pre-registered,
difficulty-stratified calibration + resolution of the stability contradiction
+ workflow-level conformal risk control.

## Contributions

| ID | Question |
|---|---|
| C1 | Does calibration survive difficulty, or track accuracy? |
| C2 | Confidence stability vs label stability (published contradiction) |
| C3 | Distribution-free risk control on composed multi-question workflows |
| C4 | Artifact: protocol, labels, JSONL, harness, Zenodo DOI |

## Experiments

1 to 5 as in teardown; **6** stability (new); **7** MAPIE Learn-Then-Test risk
control (new).

## Publishing

Zenodo first (CC BY 4.0, concept DOI). arXiv when endorser materialises.
Email TypeSafe short, no ask. Discord / `nathan@typesafe.ai` / HN
`CompleteSkeptic`.

## Sequence (days 1 to 7)

Related-work + PREREGISTRATION + labels → harness/metrics → EXP-1/6 →
EXP-2/3 → EXP-4/5/7 → analysis/draft → **publish repo + DOI** → video after.

## Blocking decisions (still open)

1. Task domain (4 tiers, ≥300 self-labelled items)
2. Baseline LLM (Haiku/Flash-class)
3. Local open-weight + logprobs for EXP-3?
4. Serving path for scored runs (native preferred)
5. Write `metrics.py` / `stability.py` by hand (recommended: yes)

## Claim sticky note

No architecture/RLCD internals · no bare "nobody has done X" · no unverified
Almeida "exactly right!" quote · limitations before results · AI-usage statement.
