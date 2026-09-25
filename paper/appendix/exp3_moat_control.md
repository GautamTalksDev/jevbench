# EXP-3. Moat control (methods note)

## Honest claim

> On identical items, is Jev's top-label ECE better than raw softmaxed
> logprobs from an open-weight model that prefills JSON and emits one
> constrained token?

That is the deciding question. Accuracy is secondary. Latency is reported
only for completeness and is **not** a fair comparison.

Motivated by Sean Goedecke's published argument that prefilling the JSON
opening and generating one constrained token recovers most of the speed,
parallelism and probabilities without RLCD.

## Arms

| Arm | System | Role |
|---|---|---|
| Jev | `jev-1.13.0` | treatment |
| Prefill | local open-weight + vLLM/MLX, one constrained token | Goedecke control |
| GLiClass | `knowledgator/gliclass` zero-shot | off-the-shelf classifier |
| Adapter | frontier LLM, full structured output | baseline |

## Prefill protocol

1. Prompt: state + question + option list (as sentinels).
2. Prefill assistant turn with `{"choice": "`.
3. Generate exactly one token, constrained to the sentinel set.
4. Softmax logprobs over that set → probabilities.
5. Map into the shared `Decision` shape.

**Tokenizer trap.** Option strings are usually not single tokens. Map each
option to a distinct sentinel (`"1"`…`"N"`), assert every sentinel is
exactly one token in *this* tokenizer (fail loudly), and record the mapping
in the run manifest. Across repeats, shuffle the sentinel↔option assignment
(order-permutation control). If accuracy or ECE moves, that is a finding
about the **control**, not about Jev.

Logprob access is required. A hosted API that hides logprobs cannot run
this experiment.

## Deciding metric

Paired **ΔECE = ECE(prefill) − ECE(jev)** on identical items, with BCa and
percentile intervals (`n_boot=10000`).

- CI entirely above 0 → prefill worse calibrated → evidence the moat (RLCD) is real.
- CI entirely below 0 → Goedecke's reading supported on this corpus.
- Crossing 0 → inconclusive.

## Latency confound (do not hide)

Local GPU inference and a hosted API are not comparable: network RTT,
batching, cold start and queueing differ. Report wall-clock flagged unfair,
plus compute-only generation time for the local model. **Do not use EXP-3
for video latency claims**, those come from EXP-1's matched serving path.

## Gate

Sentinel single-token assertion must pass before scored runs (Prompt I).
Paid scoring follows the F→G→H→I order: do not start until F1's sigmoid
power curve and labelling κ gates clear.
