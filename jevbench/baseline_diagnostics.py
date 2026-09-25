"""Part A — baseline label-mapping diagnostics (report only; no fixes).

Implements the offline diagnostic promised before any baseline number enters
the paper. Does not change client code or re-score raw.jsonl.
"""

from __future__ import annotations

import inspect
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from jevbench.chaosnli import LABEL_ORDER
from jevbench.clients import bart_mnli as bart_mod
from jevbench.clients import gliclass as gliclass_mod
from jevbench.clients import prefill as prefill_mod
from jevbench.clients import trivial as trivial_mod
from jevbench import exp1 as exp1_mod
from jevbench.exp1 import _client_pass0, normalize_records

CLIENTS = (
    "jev",
    "prefill_qwen15",
    "gliclass",
    "bart_mnli_ref",
    "trivial",
)


def load_source_by_id(items_path: Path) -> dict[str, str]:
    """Map item id → snli|mnli from committed items.jsonl state.source."""
    import json

    out: dict[str, str] = {}
    with items_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            iid = str(row["id"])
            state = row.get("state") or {}
            src = state.get("source")
            if src in ("snli", "mnli"):
                out[iid] = str(src)
    return out


def _prob_vector(probs: dict[str, float]) -> list[float]:
    return [float(probs.get(lab, 0.0)) for lab in LABEL_ORDER]


def _confusion(y_true: list[str], y_pred: list[str]) -> list[list[int]]:
    idx = {lab: i for i, lab in enumerate(LABEL_ORDER)}
    mat = [[0 for _ in LABEL_ORDER] for _ in LABEL_ORDER]
    for t, p in zip(y_true, y_pred, strict=True):
        if t not in idx or p not in idx:
            continue
        mat[idx[t]][idx[p]] += 1
    return mat


def _client_block(
    rows: list[dict[str, Any]],
    *,
    client: str,
    source_by_id: dict[str, str],
) -> dict[str, Any]:
    items = [
        r
        for r in _client_pass0(rows, client)
        if r.get("role", "primary") != "paraphrase"
        and isinstance(r.get("probabilities"), dict)
        and r.get("label") is not None
    ]
    y_true = [str(r["label"]) for r in items]
    y_pred = []
    tops: list[float] = []
    for r in items:
        probs = r["probabilities"]
        choice = str(r.get("choice") or max(probs, key=probs.get))
        y_pred.append(choice)
        tops.append(float(max(probs.values())) if probs else float("nan"))

    freq = Counter(y_pred)
    acc = float(np.mean([1.0 if a == b else 0.0 for a, b in zip(y_true, y_pred)]))

    def _split(src: str) -> dict[str, Any]:
        sel = [
            (t, p, top)
            for r, t, p, top in zip(items, y_true, y_pred, tops, strict=True)
            if source_by_id.get(str(r["item_id"])) == src
        ]
        if not sel:
            return {"n": 0, "accuracy": None}
        yt, yp, tp = zip(*sel)
        return {
            "n": len(sel),
            "accuracy": float(np.mean([1.0 if a == b else 0.0 for a, b in zip(yt, yp)])),
            "predicted_label_frequencies": dict(Counter(yp)),
            "mean_top_prob": float(np.mean(tp)),
            "confusion_matrix": {
                "order": list(LABEL_ORDER),
                "rows_majority_cols_predicted": _confusion(list(yt), list(yp)),
            },
        }

    return {
        "client": client,
        "n_primary": len(items),
        "accuracy_vs_majority": acc,
        "predicted_label_frequencies": {lab: int(freq.get(lab, 0)) for lab in LABEL_ORDER},
        "mean_top_prob": float(np.nanmean(tops)) if tops else None,
        "confusion_matrix": {
            "order": list(LABEL_ORDER),
            "rows_majority_cols_predicted": _confusion(y_true, y_pred),
            "note": "rows = majority label; cols = predicted; order entailment/neutral/contradiction",
        },
        "by_source": {
            "snli": _split("snli"),
            "mnli": _split("mnli"),
        },
        "label_order_for_probability_dict": list(LABEL_ORDER),
        "observed_probability_key_orders_sample": [
            list(r["probabilities"].keys()) for r in items[:3]
        ],
    }


def _code_excerpt(mod: Any, needle: str, *, context: int = 2) -> dict[str, Any]:
    src = inspect.getsource(mod)
    path = inspect.getsourcefile(mod) or getattr(mod, "__file__", "?")
    lines = src.splitlines()
    hits: list[dict[str, Any]] = []
    for i, line in enumerate(lines):
        if needle in line:
            lo = max(0, i - context)
            hi = min(len(lines), i + context + 1)
            hits.append(
                {
                    "match_line_index_in_module_source": i + 1,
                    "text": "\n".join(lines[lo:hi]),
                }
            )
            if len(hits) >= 3:
                break
    return {"module": str(path), "needle": needle, "excerpts": hits}


def label_mapping_report() -> dict[str, Any]:
    """Static report of how each client builds label→prob mappings (no model load)."""
    # Prefer reading HF config from cache if present; else record the expected id2label.
    bart_id2label: dict[str, Any] = {
        "model_id": bart_mod.DEFAULT_MODEL,
        "note": (
            "facebook/bart-large-mnli typical id2label is "
            "{0: 'contradiction', 1: 'neutral', 2: 'entailment'}. "
            "This study does NOT use that head directly: BartMnliClient calls "
            "transformers zero-shot-classification with candidate_labels = "
            "list(question.criteria.keys())."
        ),
        "attempted_load": False,
        "id2label": None,
    }
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(bart_mod.DEFAULT_MODEL)
        bart_id2label["attempted_load"] = True
        bart_id2label["id2label"] = dict(getattr(cfg, "id2label", {}) or {})
    except Exception as exc:  # noqa: BLE001
        bart_id2label["load_error"] = f"{type(exc).__name__}: {exc}"

    return {
        "prereg_clause": "Amendment 10/11 baseline labelling; diagnostic only (Step 15 A)",
        "chaosnli_label_order": list(LABEL_ORDER),
        "experiment_yaml_criteria_key_order": [
            "entailment",
            "neutral",
            "contradiction",
        ],
        "clients": {
            "jev": {
                "probability_dict_keys": (
                    "Whatever the API returns; flatten prefers Choice "
                    "decision probabilities keyed by label name."
                ),
                "code": _code_excerpt(exp1_mod, "Prefer the Choice decision"),
            },
            "prefill_qwen15": {
                "protocol": (
                    "Options mapped to digit/letter SENTINELS via "
                    "build_sentinel_mapping(list(criteria.keys()), pass_idx). "
                    "Prefill scores logprobs of sentinel tokens "
                    f"{prefill_mod._SENTINEL_ALPHABET[:3]!r} for the three "
                    "labels in criteria key order, then maps sentinel→label."
                ),
                "tokens_scored_for_labels_pass0_unpermuted": {
                    "entailment": "1",
                    "neutral": "2",
                    "contradiction": "3",
                },
                "note": (
                    "pass_idx>0 permutes sentinel↔label; EXP-1 local arms use "
                    "repeats=1 so pass_idx=0 → unpermuted 1/2/3."
                ),
                "code_build_sentinel_mapping": _code_excerpt(
                    prefill_mod, "def build_sentinel_mapping"
                ),
                "code_options_for": _code_excerpt(
                    prefill_mod, "return list(question.criteria.keys())"
                ),
            },
            "gliclass": {
                "labels_passed_to_pipeline": "list(question.criteria.keys())",
                "classification_type": "multi-label (then renormalised over the three)",
                "state_text": "str(request.state) when state is a dict",
                "code_labels": _code_excerpt(
                    gliclass_mod, "labels = list(question.criteria.keys())"
                ),
                "code_state": _code_excerpt(
                    gliclass_mod, "text = request.state if isinstance"
                ),
            },
            "bart_mnli_ref": {
                "hf_config_id2label": bart_id2label,
                "runtime_labels": (
                    "zero-shot-classification candidate_labels = "
                    "list(question.criteria.keys()) — NOT the MNLI head id2label"
                ),
                "state_text": "str(request.state) when state is a dict",
                "code_pipeline": _code_excerpt(
                    bart_mod, 'pipe(text, candidate_labels=labels'
                ),
                "code_state": _code_excerpt(
                    bart_mod, "text = request.state if isinstance"
                ),
            },
            "trivial": {
                "strategy": "majority_class='neutral' for ChaosNLI choice (runner default)",
                "probability_dict": "one-hot on the chosen label",
                "code": _code_excerpt(
                    trivial_mod, 'meta["strategy"] = "majority_class"'
                ),
            },
        },
        "diagnostic_hypothesis": (
            "If BART (MNLI-supervised) and Qwen fall at or below a constant "
            "majority guess on in-domain items, suspect label-name / sentinel "
            "mapping or state serialisation — not model competence. This file "
            "reports only; it does not fix clients."
        ),
    }


def run_baseline_diagnostics(
    *,
    raw_path: Path,
    items_path: Path,
    labels_by_id: dict[str, str],
    label_dist_by_id: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    from jevbench.exp1 import load_raw_records

    raw = load_raw_records(raw_path)
    rows = normalize_records(
        raw, labels_by_id=labels_by_id, label_dist_by_id=label_dist_by_id
    )
    source_by_id = load_source_by_id(items_path)
    arms = {
        c: _client_block(rows, client=c, source_by_id=source_by_id) for c in CLIENTS
    }
    return {
        "schema": "jevbench.baseline_diagnostics.v1",
        "prereg_clause": (
            "Step 15 A / Amendments 10–11 — baseline diagnostics (report only; "
            "no client fixes)"
        ),
        "raw": str(raw_path),
        "label_order": list(LABEL_ORDER),
        "arms": arms,
        "label_mapping": label_mapping_report(),
    }
