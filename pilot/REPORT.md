# Pilot REPORT — PROMPT S / R

**Run id:** `pilot_20260924T190111Z`  
**UTC:** 2026-09-24T19:01:36.020154+00:00  
**Pinned Jev:** `jev-1.13.0` native `api.typesafe.ai`  
**Local baselines:** prefill `Qwen/Qwen2.5-1.5B-Instruct`, GLiClass `knowledgator/gliclass-base-v1.0`  
**Items:** 20 mid-entropy only (seed `20260924`)  
**Outputs analysed for ΔECE:** NO  
**Credit expiry (fill from console):** unknown (console: "See credit details for expiry.")  
**Console balance after pilot:** **$5.00** (screenshot; display unchanged — pilot spend $0.000494 is below console precision)  
**Console balance Δ vs ledger:** not comparable on pilot (console still shows $5.00; ledger Δ = $0.000494). Defer ledger-vs-console check to EXP-1 (~$0.05 pause).

## Item guard

- Pilot ∩ analysis: **0** (must be 0)
- Assert passed: **True**

```
109443n, 110061n, 115173n, 116176c, 1368082221.jpg#4r1c, 21147e, 23280e, 2370481277.jpg#0r1n, 24789e, 27587n, 2904739724.jpg#0r1e, 372570543.jpg#0r1e, 3822535114.jpg#2r1e, 3970646119.jpg#0r1n, 448658518.jpg#0r1c, 4637931300.jpg#2r1c, 4695720306.jpg#3r1n, 476477265.jpg#0r1n, 507035997.jpg#2r1c, 83145n
```

## Budget

- Pilot cap: **$0.05**
- Ledger spent this run: **$0.000494**
- Global remaining: **$2.4995**

## Checks

### 1. Resolved `model` on every Jev row
- OK: **True** — `['jev-1.13.0']`

### 2. Choice vs Noul shapes
- Choice all 3 labels: **True**
- Noul, no confidence: **True**

### 3. Noul near-zero sum
- Floor `1e-06`; triggers: **0** / 20

### 4. Probability source
- Amendment 10: paid adapter baselines removed. Prefill probabilities are logprob-derived (TransformersPrefillBackend / single constrained token). GLiClass scores are classifier logits→softmax, not verbalised.

### 5. Tokens vs cost.py
- Jev mean in/out: **588.2 / 103.5**
- Prior DEFAULT_PILOT jev: {'input_tokens': 180.0, 'output_tokens': 24.0}
- Defaults updated: **True**

### 6. Latency
- Jev p50/p95: **125 / 266** ms
- Prefill p50/p95: **nan / nan** ms
- GLiClass p50/p95: **148 / 208** ms
- Network floor api.typesafe.ai p50/p95: **63 / 208** ms

### 7. Errors (max 3 retries; never 4xx)
- `{}`

### 8. Baseline model pins
- Prefill resolved: `[]`
- GLiClass resolved: `['knowledgator/gliclass-base-v1.0']`

## Blockers

- None

## Amendment 10 gate

Code changes after this pilot prompted by Jev output → cite this pilot run id and mid-entropy-only items.

Raw (stripped, safe to commit): `pilot/stripped/pilot_20260924T190111Z.jsonl`  
Raw (full, gitignored — has request bodies): `pilot/full/pilot_20260924T190111Z.jsonl`  
Rebuild/verify: `.venv/bin/python scripts/rebuild_requests.py`
