"""Canonical request-body hashing for public (stripped) run logs.

Public logs keep responses + item IDs + ``request_body_sha256``. The premise /
hypothesis text stays out of the committed artifact. ``scripts/rebuild_requests.py``
re-fetches ChaosNLI text, rebuilds the body, and checks the hash.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


REQUEST_BODY_SCHEMA = "jevbench.request_body.v1"


def questions_to_canon(questions: dict[str, Any]) -> dict[str, Any]:
    """Stable JSON view of Choice/Noul/Score question objects or yaml dicts."""
    out: dict[str, Any] = {}
    for key in sorted(questions.keys()):
        q = questions[key]
        if hasattr(q, "instructions"):
            kind = type(q).__name__.replace("Question", "").lower()
            entry: dict[str, Any] = {
                "kind": kind,
                "instructions": q.instructions,
            }
            criteria = getattr(q, "criteria", None)
            if criteria is not None:
                if isinstance(criteria, dict):
                    entry["criteria"] = {str(k): criteria[k] for k in sorted(criteria)}
                else:
                    entry["criteria"] = list(criteria)
            out[key] = entry
            continue
        if isinstance(q, dict):
            entry = {k: q[k] for k in sorted(q) if k != "name"}
            out[key] = entry
            continue
        raise TypeError(f"cannot canonise question {key!r}: {type(q)!r}")
    return out


def build_request_body(
    *,
    item_id: str,
    client: str,
    model: str,
    state: dict[str, Any],
    questions: dict[str, Any],
) -> dict[str, Any]:
    """Full request body (includes sentence text). Never commit this object."""
    # Only the fields that affect the model call.
    state_canon = {
        k: state[k]
        for k in sorted(state.keys())
        if k in ("premise", "hypothesis", "pair_id", "source")
    }
    return {
        "schema": REQUEST_BODY_SCHEMA,
        "item_id": str(item_id),
        "client": str(client),
        "model": str(model),
        "state": state_canon,
        "questions": questions_to_canon(questions),
    }


def dumps_canonical(body: dict[str, Any]) -> bytes:
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def request_body_sha256(body: dict[str, Any]) -> str:
    return hashlib.sha256(dumps_canonical(body)).hexdigest()


def strip_row_for_public(row: dict[str, Any], *, request_body_sha256_hex: str) -> dict[str, Any]:
    """Keep responses / ids / hash; drop any request-side sentence text."""
    out = {
        "item_id": row.get("item_id"),
        "client": row.get("client"),
        "model_requested": row.get("model_requested"),
        "request_body_sha256": request_body_sha256_hex,
        "request_body_schema": REQUEST_BODY_SCHEMA,
        "summary": row.get("summary"),
        "decisions": row.get("decisions"),
        "est_cost_usd": row.get("est_cost_usd"),
        "stratum": row.get("stratum"),
        "outputs_analysed": row.get("outputs_analysed", False),
    }
    # Defensive: never leave premise/hypothesis in nested blobs.
    blob = json.dumps(out, ensure_ascii=False)
    if '"premise"' in blob or '"hypothesis"' in blob:
        raise ValueError(
            f"stripped row for {row.get('item_id')} still contains premise/hypothesis"
        )
    return out
