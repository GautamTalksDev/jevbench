# Contributing

Independent study. Not affiliated with or endorsed by TypeSafe AI.

## Welcome

- Bug reports against the harness, the arena page, or the docs
- Replication runs that name the commit, the serving path, n, the cost, and any deviation
- Documentation that keeps the facts and the status line (preregistered, EXP-1 not yet run)

One topic per pull request. Tests must pass (`make test`).

## Not merged as an ordinary pull request

These go through the public amendment process below:

- Any change to the preregistered analysis (hypotheses, metrics, decision rules, sample size)
- Any change to a frozen file: `PREREGISTRATION.md`, `preregistration.lock.json`, or `experiments/exp1_difficulty_calibration.yaml`
- Any change to results after unblinding

A pull request that edits those files to "fix" a number after a run will be closed.

## Public amendment process

1. Open an issue that states the reason, what would change, and why the locked rule cannot stand.
2. Wait for the discussion to be public. Do not edit the frozen files in that issue's pull request.
3. If the maintainer accepts the amendment, they commit it themselves: update `PREREGISTRATION.md` with a dated amendment, and update `EXPECTED_PREREG_SHA256` in `scripts/verify_prereg_integrity.py` in the same commit.
4. Dataset byte changes are a new protocol. Prior scored runs stay exploratory. They are not quietly reused.

Until EXP-1 has been run, there are no results to unblind.

## Local check

```bash
make test
python3 scripts/verify_citations.py
python3 scripts/check_prose_dashes.py
```

Code of conduct: [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). Security reports: [`SECURITY.md`](SECURITY.md).
