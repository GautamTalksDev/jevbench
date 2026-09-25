# History rewrite (pre-public)

**When:** 2026-09-24 (UTC).
**Why:** Remove frozen-paraphrase sentence text from git history before any
public push. Paraphrases are AI-generated rewrites of ChaosNLI/SNLI/MNLI
sentences (specific Cursor model undetermined); publishing them would undo
the "no sentence text in the repo" decision.

**Credibility:** This rewrite happened **before any public push** and **before
any scored EXP-1 Jev data**. Local commit hashes cited in earlier amendment
notes had no public value yet. The public proof of pre-registration is the
first push plus a Zenodo DOI, both before EXP-1.

**What changed in history:**
- Deleted `datasets/chaosnli/paraphrases.jsonl` from every commit.
- Rewrote paraphrase rows in `datasets/chaosnli/items.jsonl` to drop
  `premise`/`hypothesis` (`text_status: local_paraphrase_required`).

**What is public now:** `paraphrases.sha256` (SHA-256 of the exact frozen
file) and `paraphrase_ids.json`. Text remains only at gitignored
`datasets/chaosnli/local/paraphrases.jsonl`.

**Backup (local):**
`/home/gautamtalksdev/projects/jevbench_backup_pre_history_rewrite_20260924T191456Z`

## Old → new commit hashes

| Subject | Old | New |
|---|---|---|
| prereg: amend EXP-1 endpoint, equal-n, effect size, and powered n | `70be25f7baf91daa748cec11c38f26f17315b17c` | `70be25f7baf91daa748cec11c38f26f17315b17c` |
| prereg: record binding amendment commit hash | `9579c9c9b4f3d523c2ab596be2dde46671855d30` | `9579c9c9b4f3d523c2ab596be2dde46671855d30` |
| Switch EXP-1 to ChaosNLI with entropy strata and soft scoring. | `fac0ce96483a75d10671a5bc426e9c351ead8b40` | `5d3ecdcee949b847fa624e9753462b7ef23cdea4` |
| prereg: Amendment 6, switch EXP-1 to ChaosNLI (soft scoring primary) | `bf03af91ecd08efa927cea9d1a8eb5adfc941634` | `8f15463287ee0e923852ba175269bb4499923aee` |
| prereg: record Amendment 6 binding commit hash | `04da9f62ad950360ccb09d951ae5a2ce5003a0bf` | `52cacfe17348576ee3e8553e7daf4c7b30ecb2aa` |
| Ship EXP-1 certificate page and soft-calibrated asymmetric null. | `4b3a5173d9922c77faed0fc1a330d11f3149b2a7` | `01fecaa7c3748d8bc346ed32e2b403d712e2d415` |
| prereg: Amendment 7. Beta-Binomial soft null (PROMPT O) | `3891afac56818bed1dc64850d9775e65d64f613c` | `6a3bbef149f4d6536f6f9cfe2e28a14055e30144` |
| prereg: record Amendment 7 binding commit hash | `1cac0bcc401e723b3ebfff705ecffdb75deb8235` | `a909d9326dc7b2e2705265a6b336e8949413c039` |
| prereg: Amendment 8, parametric correction for structural ECE bias | `9828641f73666c51a6ced232726049b7ff19e141` | `4537e87220a0a324d4c290c0a5f0d231a5cfd90e` |
| prereg: record Amendment 8 binding commit hash | `18250ad8a0c6bea26c260ef007360bbb932860f0` | `d908249895eb87847c89f5c03050bb387f86adbb` |
| prereg: Amendment 9. Choice+Noul arms, parametric verdict, JSD/TVD | `6011c75818f851ea99b495776edb7f2be0a8933d` | `66037a7624df2b8fcd2860eeecf9f88b09546305` |
| prereg: record Amendment 9 binding commit hash | `46b09274dd3007c5a6795c71bc487eb84369b11e` | `bf65aa4af0d0f883f436323cbdc8e42a62e68888` |
| cert: pin prereg commit stamp to Amendment 9 | `e7f93948b27ecd5c88869754bb98ebc759101509` | `a9089b54fa8b266c838e13a20023cfe7420676ee` |
| prereg: Amendment 10, secondaries S1 to S3, local baselines, budget guard | `755be90731f71dcfdd503fc9668af6a145b90f71` | `1e64d723440ce27a3b67ac699380b180a5263908` |
| prereg: record Amendment 10 binding commit hash | `b039b5e20d6031c646c874f06b6279c1e1407bac` | `b49f0f11b720bd89953b9be5f250fe657012d74a` |
| fix: load GLiClass via package or BART-MNLI fallback | `b3b0ab116fec44760abe7d8854eb3005522daf24` | `2a489733efe663e7830754b206d9858f4cd584d1` |
| prereg: Amendment 11. BART as supervised reference; local R=1 | `49c13f7debcede5d3f7f27c05551c906aa0ede27` | `ed4655f9cc5dbf0bd7ea90b5929a72794d5a9495` |
| prereg: record Amendment 11 binding commit hash | `1b3ad58b2aae4726b2004fcaea0f62b34a415f65` | `690c169cab29b904191d4dd59f2e6542061fa545` |
| docs: GLiClass wording, model card doesn't list MNLI/SNLI | `18dbb86f436b0e7ee0b7bd277d111f6cbd1c0f37` | `9ae9aa1473ede8a229f3e827b95795a309d94510` |
| prepub: strip paraphrase text; Pages manual; disclaimer; runs ignore | `bdbba1d657559f8d56e28f81405515b6e20d0c4b` | `4ab5c912e668de2c3e6d8ba9e4e78222ed7fcd72` |

