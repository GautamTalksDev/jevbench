# How to verify

You do not need an API key. You do need git and Python 3.11 or newer.

## 1. Confirm the preregistration hash

The lock file does not store a digest of `PREREGISTRATION.md` (that file is frozen, so the digest lives in the checker). Recompute both.

```bash
python3 scripts/verify_prereg_integrity.py
sha256sum PREREGISTRATION.md
```

The script exits 0 only if `PREREGISTRATION.md` still matches the published digest and `datasets/chaosnli/items.jsonl` and `labels.jsonl` still match `preregistration.lock.json`.

You can also hash the markdown yourself and compare it to `EXPECTED_PREREG_SHA256` in [`scripts/verify_prereg_integrity.py`](../scripts/verify_prereg_integrity.py).

## 2. Confirm cited commits exist

Docs name commit ids for amendments and history rewrites. This fails if one of those ids is not in the repository:

```bash
python3 scripts/verify_citations.py
```

## 3. Reproduce the harness offline

```bash
make install
make reproduce
make check-repro
git status --porcelain -- results/harness_fixture results/SHA256SUMS
```

The last command should print nothing. If it prints a path, the regenerated fixture does not match the committed checksums.

`results/harness_fixture/README.md` states what that folder is. It is a pipeline check, not EXP-1.
