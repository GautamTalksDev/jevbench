# Contributing

This repo is meant to outlive one video. The useful fork is: add a task, pin a
newer Jev version, re-run the harness, and keep the same metrics.

## What to add

1. **A task** under `datasets/<name>/` with `items.jsonl`, `labels.jsonl`,
   [`LABEL_GUIDE.md`](datasets/support_tickets/LABEL_GUIDE.md), and
   [`DISPUTED.md`](datasets/support_tickets/DISPUTED.md).
2. **An experiment YAML** under `experiments/` that points at that task and
   pins `jev-<version>` (never `jev-latest` for scored runs).
3. **Pre-registration** — `jevbench preregister <exp>` before any scored call.
4. **Dry-run cost** — `jevbench run … --dry-run` before you spend.

## What not to do

- Do not cite Arena Live / playground numbers as scored results.
- Do not compute ECE by hand — use `jevbench.metrics` (netcal wrappers).
- Do not pass `confidence` into calibration functions (raises `TypeError`).
- Do not write scored artifacts from the Arena UI into `runs/<id>/`.

## Reproduce before you open a PR

```bash
make install
make test
make reproduce
make check-repro
```

CI fails if `results/` drifts after `make reproduce` (bit-for-bit vs the
committed tree). If your change *intentionally* updates goldens, regenerate
with `make reproduce` and commit the new `results/` artifacts.

## Rerunning against a newer Jev

1. Update the pinned model in the experiment YAML and `PREREGISTRATION.md`.
2. Re-lock hashes only if the **dataset** changed — not because the model did.
3. Run scored calls with the native serving path stated in the prereg (or
   measure both paths and label them).
4. Leave the offline fixture (`runs/offline_fixture/`) alone unless you are
   refreshing the demo pack — that fixture is for harness CI, not claims.

## Code style

- Prefer libraries over reimplementation (`netcal`, `mapie`, sklearn).
- Latency: report p50 / p95 / p99 only.
- Score floats stay floats — never cast System One Score expectations to int.

Questions and task proposals: open an issue. Cross-links to prior Jev benches
are welcome — credit is cheap and compounds.
