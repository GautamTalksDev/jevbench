# Preregistration post-binding changelog

**Binding commit for Amendment 11:** `ed4655f9cc5dbf0bd7ea90b5929a72794d5a9495`

**SHA-256 of `PREREGISTRATION.md` at that commit:**
`8f84dde26b39612c476f8dafffc6433e481d7e1b9c2934fb975db6fbace63b32`

**SHA-256 of `PREREGISTRATION.md` at HEAD (tonight's Zenodo freeze candidate):**
`e19aecb696d780d16d05363b34d850b9a58282097cd84b0b207df7eb77146014`

**First pilot call:** `2026-09-24T04:11:33Z` (file `pilot/full/pilot_20260924T041133Z.jsonl`).
The pilot used excluded items and recorded tokens and errors, not accuracy.
No EXP-1 scored run exists.

**Classification rule used here:**

- *Clerical:* wording, disclosure, permalinks, pin refreshes after a history rewrite, or a content-hash update that only removes redistributed sentence text while leaving labels, n, strata membership, hypotheses, thresholds, tests, and the verdict rule alone.
- *Substantive:* any change to hypotheses, thresholds, strata definition, n, statistical tests, or the verdict rule.

**Result of this review:** every listed commit is classified *clerical*. None changed the locked analysis rule. Every listed commit is *after* the first pilot timestamp. That timing is disclosed, not hidden.

The machine-readable lock (`preregistration.lock.json`) also moved once, in `4ab5c91`, when `items_sha256` was updated to match the stripped `items.jsonl`. `labels_sha256`, `n_items`, `tier_counts`, and `locked_at` stayed the same. Updating the digest without bumping `locked_at` is itself disclosed here.

Tonight's Zenodo deposit is the true public freeze: current file, both SHA-256 values, the commit hashes below, and this changelog. The paper should say: bound at `ed4655f9`; N clerical edits followed, all listed; frozen by DOI before any EXP-1 call.

| Commit | ISO timestamp (commit author date, UTC) | vs first pilot | Classification | One-line reason |
|---|---|---|---|---|
| `690c169` | 2026-09-24T05:24:40Z | after | clerical | Records `AMENDMENT_11_COMMIT` (a file cannot contain its own commit hash until a follow-up commit). |
| `9ae9aa1` | 2026-09-24T05:29:46Z | after | clerical | GLiClass wording: model card does not list MNLI or SNLI; that is not a claim it was untrained on them. Arm assignment unchanged. |
| `4ab5c91` | 2026-09-24T19:18:36Z | after | clerical | Strips paraphrase sentence text from the public tree; rehashes `items.jsonl` (`38c8c5b2…` to `1fbb1ab3…`); points at `paraphrases.sha256` instead of shipping text; labels, n, and strata unchanged. Lock `items_sha256` updated in the same commit. Verified content-preserving: hydrated text equals pre-rewrite text for 100/100 paraphrase items (and 20/20 sampled primary items); see `tests/test_hydration_integrity.py`. |

After the history rewrite, commit `ed4655f9` contains the placeholder `items.jsonl` (`1fbb1ab3…`) while its lock records `38c8c5b2…`. The pre-rewrite file is preserved in the local backup; its hash and a field-by-field comparison (1,600 IDs identical; only 100 paraphrase-role state fields differ) are recorded here.
| `2ca7762` | 2026-09-24T19:21:21Z | after | clerical | Refreshes Amendment 6 to 11 pin hashes to post-history-rewrite ids; points at `docs/HISTORY_REWRITE.md`. |
| `6b0c88b` | 2026-09-24T19:26:10Z | after | clerical | Corrects paraphrase authorship from hand-authored to Cursor-generated. |
| `e703d1d` | 2026-09-24T19:38:31Z | after | clerical | Expands the FINDINGS.md permalink; refuses to invent a paraphrase model name; adds the limitation that the contamination control is directional, not dispositive. |
| `733cb7c` | 2026-09-25 (local) | after | non-analytic safety guard | Runner refuses unhydrated or hash-mismatched paraphrase inputs before any client call; no effect on valid inputs; added before any EXP-1 call. |
| `20ee218` | 2026-09-25 | after | non-analytic | `--clients` split for scheduling; Qwen float32 for CPU performance; same model, same prompts, logprob method unchanged. PrefillClient now caches the loaded model until release() (previously reloaded per call); outputs unchanged, performance only. |
| _(pending)_ | 2026-09-25 | after | non-analytic | CLI loads `.env` before live runs (smoke already did); resume ignores error rows so failed keys are retried. No change to prompts, models, or scoring. |

Full line-level drift: `/tmp/prereg_drift.diff` (also copied to the Windows Desktop as `prereg_drift.diff`). Command that produced it:

```bash
git --no-pager diff ed4655f9 HEAD -- PREREGISTRATION.md
```

Nothing in that diff changes H1, the soft ΔECE endpoint, the 0.09 / 0.02 gates, the parametric verdict, n = 750 per stratum, or the entropy thresholds.
