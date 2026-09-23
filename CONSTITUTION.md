# Project Constitution

These rules override any instinct. Read `API_REFERENCE.md` before writing client code.
If something isn't documented there, add a TODO and ask — do not invent it.

1. **SCIENCE IS HEADLESS.** The harness is CLI + files. No web server, no
   database, no auth, no ORM. Results are JSONL on disk.

2. **RAW RESPONSES ARE SACRED.** Every API response is written verbatim to
   `raw.jsonl` before any parsing, scoring, or transformation. Never mutate
   `raw.jsonl`. Scoring reads it and writes elsewhere.

3. **PROVENANCE ON EVERY ROW.** Each record carries: `run_id`, `item_id`,
   resolved model ID (from the response `model` field, NOT the requested alias),
   serving path, request timestamp, wall-clock latency in ms, attempt number,
   and the git SHA of the code that produced it.

4. **DETERMINISTIC REPLAY.** Every metric and chart must be regenerable from
   committed JSONL with zero network access. `make reproduce` runs offline.

5. **NO SILENT FAILURES.** A failed call is recorded as a failed call with its
   error class. It is never dropped, retried-into-silence, or averaged away.

6. **NO MEANS FOR LATENCY.** Report p50/p95/p99. If you write `.mean()` on a
   latency series, you have made a mistake.

7. **INTERVALS OR NOTHING.** Any comparative claim ships with a bootstrap
   interval (10,000 paired resamples). An interval crossing zero is
   **INCONCLUSIVE**, never "equivalent".

8. **LIBRARIES OVER REIMPLEMENTATION.** Calibration and conformal prediction
   mathematics come from netcal and MAPIE. Our code wraps and validates
   against them. A hand-rolled ECE in a paper about calibration is an
   unforced error.

9. **CONTRADICTIONS ARE FINDINGS.** When our data disagrees with a published
   result, we report the disagreement and the conditions under which each
   holds. We never quietly pick the one that flatters us.

10. **UNKNOWN IS A VALID ANSWER.** Jev's architecture, training data and RLCD
    method are undisclosed. Any code comment, docstring, variable name or
    paper sentence that asserts them is a bug.
