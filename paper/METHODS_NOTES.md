# Methods notes (pre-EXP-1)

## Contamination paraphrases

The EXP-1 contamination control uses **100 frozen paraphrases** (50 easy /
50 hard) of ChaosNLI analysis-set items. They are **not** a new gold label:
each paraphrase keeps its source item's annotator distribution.

**Authorship.** The paraphrases were **authored by hand for this study**
(human rewrite of the premise and hypothesis). They were **not** generated
by an LLM. They were frozen before any scored Jev call
(seed `20260924`; see `datasets/chaosnli/thresholds.json`).

**What is public.** This repository does **not** redistribute paraphrase
sentence text (cautious reading of SNLI/MNLI source licenses — the
paraphrases are rewrites of those sentences). Public artifacts:

- `datasets/chaosnli/paraphrases.sha256` — SHA-256 of the exact frozen
  `paraphrases.jsonl` bytes
- `datasets/chaosnli/paraphrase_ids.json` — the 100 source item IDs and
  tier membership

**What stays local (gitignored).**  
`datasets/chaosnli/local/paraphrases.jsonl` — the sentence text. Verify:

```bash
cd datasets/chaosnli && sha256sum -c paraphrases.sha256
```

**Runtime.** `jevbench.chaosnli.hydrate_dataset` loads local paraphrase
text into memory for runs; `logged_state` strips it from committed logs.
