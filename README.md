# jevbench

**Paper:** [doi.org/10.5281/zenodo.22971492](https://doi.org/10.5281/zenodo.22971492)  
**Preregistration:** [doi.org/10.5281/zenodo.22971413](https://doi.org/10.5281/zenodo.22971413)

> **About the red CI marks.** GitHub Actions on this repository are currently blocked by an account
> billing issue, so workflow runs fail within seconds, before any code executes. This is not a test
> failure. At commit `4291542` the full suite passes locally (184 tests). To check for yourself, run
> `pytest -q` and `make reproduce` (no API key needed). The note will be removed once CI runs again.


Independent study. Not affiliated with or endorsed by TypeSafe AI.

This repository is a pre-registered test of whether TypeSafe Jev stays calibrated when natural-language inference items get harder. Expected calibration error (ECE) is the gap between predicted confidence and how often that confidence is right. The study compares ECE on easy items with ECE on hard items.

## Status

Preregistered. EXP-1 scored run is complete (`runs/` is local; committed summaries live under `results/exp1*.json`). Primary Choice-arm Amendment 9 verdict in `results/exp1.json`: **tracks**. Offline numbers under `results/harness_fixture/` remain a pipeline check only, not a Jev finding.

The offline numbers under `results/harness_fixture/` come from a committed fixture. They check that the pipeline is deterministic. They are not a finding about Jev.

## The question in plain English

When an item is harder for humans, does Jev's confidence still match the chance it is right?

A stratum is a difficulty group. This study uses two strata, easy and hard, cut from ChaosNLI by annotator disagreement. Delta ECE (written ΔECE) is ECE on the hard stratum minus ECE on the easy stratum. A bootstrap is a way to redraw the sample many times and see how much a number moves. The pre-registered rule looks at the corrected ΔECE, not at a story written after the numbers appear.

## How it works

The harness calls a pinned model, writes every response to JSONL, and computes metrics later, offline. Primary scoring is soft. Soft correctness is the share of ChaosNLI annotators who picked the model's top label. Hard scoring (top label versus the majority label) is reported beside it. Sentence text is not stored in this repository.

The two strata:

- Easy: the lowest quartile of per-item Shannon entropy of the 100-annotator label distribution, then cut to 750 items.
- Hard: the highest quartile, also cut to 750 items.

Equal counts matter. A larger hard set could move ΔECE even if calibration did not change. Paraphrases (50 extra items per stratum) are a contamination check and are excluded from ΔECE. The pinned model is `jev-1.13.0`. The serving path for scored calls is `native`.

## Why you can trust it

The analysis was written down before any scored Jev call. That document is the preregistration: [`PREREGISTRATION.md`](PREREGISTRATION.md), with dataset hashes in [`preregistration.lock.json`](preregistration.lock.json). Changing the locked item or label bytes makes a scored run refuse to start.

Hashes: item and label files are pinned by SHA-256. Citation checks confirm that commit ids named in the docs exist in this repository.

Equal-n: easy and hard use the same count, so a sample-size gap cannot imitate a calibration gap.

Bias correction: binned ECE is biased even when a model is perfectly calibrated. The preregistration requires a parametric correction and says to publish the raw number beside the corrected one.

Publish-regardless: the locked text says the result will be published whichever way it goes, including a null and including a result that supports TypeSafe's claims.

What you should not trust yet:

- The fixture finding is not EXP-1. Do not cite it as a product claim.
- The fixture has 44 Jev rows. Bootstrap intervals on that pack are wide.
- Support-ticket labels in `datasets/support_tickets/` are author-assigned, with a written guide and a disputed file. They are not the ChaosNLI 100-annotator labels used for EXP-1.
- Latency and cost from a path other than `native` are not interchangeable with the scored path.
- ECE without bin counts is not interpretable here. Empty high-confidence bins are a warning.
- Jev's `confidence` field is sharpness, not a probability of being correct. Metrics refuse to treat it as one.
- The fixture baseline is simulated. It is not a production invoice for another model.
- Arena replay does not write a scored run. A specimen stamp means the gauge is synthetic.
- Chart PNG hashes match Linux CI (Agg backend, DejaVu). macOS pixels can differ. `make check-repro` on Ubuntu is the check that counts.

## Quick start

Offline. No API key. No network.

```bash
make install
make reproduce
make check-repro
```

`make reproduce` rebuilds metrics and charts from committed JSONL. CI fails if `results/harness_fixture/` drifts from [`results/SHA256SUMS`](results/SHA256SUMS). Source rows: [`runs/offline_fixture/raw.jsonl`](runs/offline_fixture/raw.jsonl).

A skeptic's checklist is in [`docs/HOW_TO_VERIFY.md`](docs/HOW_TO_VERIFY.md). Terms are in [`docs/GLOSSARY.md`](docs/GLOSSARY.md).

## Run with Jev

You need your own key in `.env` (see [`.env.example`](.env.example)). The key is never printed and never written into a run file. CI does not receive it.

The EXP-1 hard cap is $1.00. Estimate before you pay:

```bash
jevbench run experiments/exp1_difficulty_calibration.yaml --dry-run
```

EXP-1 has been scored in this tree. A public DOI for the preregistration is still recommended before treating the write-up as archival; this tree does not mint that DOI for you.

Hypothetical replay of fixture tokens at the `2026-09-19` price snapshot is recorded in `results/harness_fixture/metrics.json` under `cost.hypothetical_live_usd_if_replayed`. `make reproduce` itself costs $0.00.

## Repo map

| Folder | What's in it |
|---|---|
| `jevbench/` | Harness: clients, metrics, runner, preregistration lock checks |
| `experiments/` | Experiment YAML, including the locked EXP-1 spec |
| `datasets/chaosnli/` | Item ids, annotator counts, entropy, stratum. No sentence text |
| `datasets/support_tickets/` | Small labelled ticket set used by the offline fixture |
| `runs/offline_fixture/` | Committed responses the offline rebuild reads |
| `results/` | Charts, power notes, and the fixture checksum file |
| `results/harness_fixture/` | Regenerated fixture metrics. Not an EXP-1 result |
| `arena/` | Static certificate page. Replay only |
| `paper/` | Methods notes and the related-work search log |
| `docs/` | Glossary, FAQ, verification steps |
| `tests/` | Pytest suite, including supply-chain checks |
| `.github/` | CI workflows, Dependabot, secret scan |

## Data licence

ChaosNLI is CC BY-NC 4.0. SNLI is CC BY-SA 4.0. MNLI has mixed terms. Sentence text is not in this repository. The fetch script is [`scripts/fetch_chaosnli.py`](scripts/fetch_chaosnli.py). It checks pinned SHA-256 values and writes a gitignored cache.

## Prior work

Credit these benches by name:

- [jev-baselines-eval](https://github.com/ickma2311/jev-baselines-eval) (ickma2311)
- [jev-phishing-bench](https://github.com/anisselbd/jev-phishing-bench) (anisselbd)
- [jev-rerank-bench](https://github.com/anessbelbati/jev-rerank-bench) (anessbelbati)
- [jev-benchmark](https://github.com/themsquared/jev-benchmark) (themsquared)
- [jev-exploration](https://github.com/SamuelSacco/jev-exploration) (SamuelSacco)
- [jev-frontier-bench](https://github.com/manjunathshiva/jev-frontier-bench) (manjunathshiva)
- [jev-decision-bench](https://github.com/OmarMujahid/jev-decision-bench) (OmarMujahid)
- [jev-calibration-audit](https://github.com/MohitSV/jev-calibration-audit) (MohitSV)
- [jev-bench](https://huggingface.co/datasets/Praveenrajus/jev-bench) (Praveenrajus)

Position and tools papers: Deferred Crispification (Zenodo 22801506) and Jev in Practice / daf-jev (Zenodo 22816188). The search log is [`paper/RELATED_WORK.md`](paper/RELATED_WORK.md). Samuel Sacco's issue #1 is the origin of the flagship question. This study does not claim to have invented it.

## How to cite

Use [`CITATION.cff`](CITATION.cff).

## Security

This is a static repository. There is no login and no database. CI is read-only, third-party actions are pinned, and the Jev API key is not a workflow secret. The certificate page accepts only a same-origin `data/*.json` path. Details: [`SECURITY.md`](SECURITY.md), [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md), [`.github/workflows/security.yml`](.github/workflows/security.yml).

## Contributing

Bug reports, replication runs, and docs are welcome. Changes to the preregistered analysis, the frozen files, or results after unblinding are not merged as ordinary pull requests. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Licence

MIT. See [`LICENSE`](LICENSE).

The certificate page is a lab instrument: [`arena/UI_SPEC.md`](arena/UI_SPEC.md), [`arena/PUBLISH.md`](arena/PUBLISH.md). Do not share an old hosted link that still shows the pre-Amendment 9 interval verdict. Project rules: [`CONSTITUTION.md`](CONSTITUTION.md).
