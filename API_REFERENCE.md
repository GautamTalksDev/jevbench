# API Reference — TypeSafe Jev / System One

**Authoritative for this repo.** Do not infer, guess, or "correct" any API shape
from model memory. Jev launched 15 September 2026; it is not in training data.
If something is missing here, add a TODO and ask — do not invent it.

Verified: 19 September 2026 · SDK line: 0.6.0 (Python and JavaScript)

---

## Install & environment

```bash
pip install typesafe-sdk        # requires Python >= 3.10
# or: uv add typesafe-sdk
```

| Var | Purpose |
|---|---|
| `TYPESAFE_API_KEY` | Auth. Read automatically by the client. |
| `TYPESAFE_BASE_URL` | Override API root. Default `https://api.typesafe.ai` |
| `TYPESAFE_DEFAULT_MODEL` | Default model. Client defaults to `jev-latest`. |

Endpoints: `POST /v1/systemone` (evaluate), `GET /v1/models` (list).

**Pinned model for scored runs:** `jev-1.13.0` (never the moving `jev-latest` alias).
Always log the response `model` field (resolved versioned ID).

---

## Request

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

client = TypeSafeClient()   # reads TYPESAFE_API_KEY

response = client.system_one(
    state="I was charged twice for order A-104. Please refund the duplicate.",
    questions={
        "department": Choice(
            instructions="Which team should handle this?",
            criteria={
                "billing":   "Charges, invoices, refunds",
                "technical": "Bugs and integration problems",
                "other":     None,          # None = no description, still a valid option
            },
        ),
        "refund_requested": Noul(
            instructions="Does the customer ask for money back?",
        ),
        "urgency": Score(
            instructions="How urgent is this ticket?",
            criteria=["can wait", "this week", "today"],   # ORDERED list, not a dict
        ),
    },
)
```

### Critical shape notes

- `Choice.criteria` is a **dict** — option name → description (or `None`).
- `Score.criteria` is an **ordered, non-empty sequence** (list). Order is the rubric. Sent as an array on the wire.
- `state` accepts a string, a JSON object, or an array of text values.
- `instructions` and description accept JSON structure, not just strings. Omitted vs explicit null are distinct.
- All questions in one call are evaluated **in parallel and in isolation** against the same state.

---

## Response

```json
{
  "model": "jev-latest",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "probabilities": { "billing": 0.84, "technical": 0.159, "sales": 0.001 },
      "confidence": 0.596
    },
    "frustration": {
      "type": "score",
      "score": 1.035,
      "legend": { "0": "Calm, just stating facts", "1": "Frustrated but civil", "2": "Very angry, strong language" },
      "confidence": 0.842
    },
    "is_urgent": { "type": "noul", "noul": 0.999 }
  },
  "usage": { "input_tokens": 312, "output_tokens": 48 }
}
```

### Four things that break naive code

1. **`score` is a FLOAT, not an index.** `1.035` is an expectation across levels with a legend mapping integer keys to labels. Do not round it and do not interpolate a magnitude from it — TypeSafe's jaggedness page warns score levels are weak in numerical calibration. You may threshold the expectation. Treat the probability distribution over levels as the real output.
2. **`noul` has no `confidence` field.** Only Choice and Score do. Any code assuming a uniform answer shape will crash.
3. **`output_tokens` is reported but free.** Don't put it in the cost calculation. Do log it.
4. **`model` in the response is the resolved versioned ID.** Log it on every call — `jev-latest` is a moving alias and this field is provenance.

### Access patterns

```python
response.answers["department"].choice       # by key
response.nouls["refund_requested"].noul     # by primitive type
response.choices["department"].probabilities
response.scores["urgency"].score
response.usage.input_tokens
```

Also available: `AsyncTypeSafeClient`, configurable `RetryPolicy`, typed exceptions
(`RateLimitError`, `APITimeoutError`, `APIConnectionError`, `BadRequestError`,
`InternalServerError`, …).

---

## `confidence` vs `probabilities` (calibration guard)

Jev returns BOTH `probabilities` (a distribution) and `confidence` (a statistic
derived from the **shape** of that distribution). `confidence` is a **sharpness**
measure, not a correctness estimate. Calibration metrics run on
`probabilities` / `noul` ONLY. Raise `TypeError` with an explanatory message if a
confidence array reaches a calibration function.

---

## Serving paths

| Path | Notes |
|---|---|
| Native `api.typesafe.ai` | Use for all scored latency. |
| OpenRouter Decisions API (`openrouter.ai/api/alpha/decisions`) | Same System One body shape. SDK friction: `base_url` override still appends `/v1/systemone`. |
| Vercel AI Gateway / AI SDK | Usable without waitlist if you have a Vercel account. |

If Jev runs through a gateway and the baseline through a native API, latency
differences are confounded by path. State geography and path on every scored run.

---

## Fair-baseline adapter

TypeSafe publishes `system-one-adapter-python`:

```bash
pip install 'system-one-adapter[openai]'
pip install 'system-one-adapter[anthropic]'
```

```python
from system_one_adapter import SystemOneAdapterClient, Noul, Choice, Score

client = SystemOneAdapterClient(
    structured_outputs=True,
    llm_answer_mode="probabilities",  # or "discrete"
    normalize_probabilities=True,
)

response = client.system_one(
    state="This book was a delight to read.",
    questions={"positive": Noul(instructions="The book review is positive.")},
    provider="openai",
    model="gpt-4o-mini",
)
```

Use their adapter for EXP-2/EXP-6 LLM baselines. Ablate
`llm_answer_mode="probabilities"` vs `"discrete"`.

---

## Things to verify live (first 10 minutes)

- [ ] Choice cardinality (launch blog: 255; HN comment: 10). Binary-search.
- [ ] Context limit: `models.md` says 64k total / 32k state+longest question; tracker claims `primitives.md` says ~32k total.
- [ ] `jev-preview` vs `jev-latest` both resolve to `jev-1.13.0` via response `model`.
- [ ] `GET /v1/models` lists aliases only? Versioned IDs accepted anyway?
- [ ] Rate-limit `Retry-After` header presence on 429.
