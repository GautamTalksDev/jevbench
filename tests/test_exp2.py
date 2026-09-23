"""EXP-2 nested CV, fairness framing, and label-sweep tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jevbench.exp2 import (
    FAIRNESS_NOTE,
    HONEST_CLAIM,
    ItemRow,
    nested_cv_supervised,
    run_exp2_analysis,
    run_label_sweep,
)
from jevbench.experiment import load_experiment


def _synth_rows(n: int = 120, seed: int = 0) -> list[ItemRow]:
    rng = np.random.default_rng(seed)
    labels = ["billing", "technical", "other"]
    tiers = ["trivial", "easy", "hard", "ambiguous"]
    rows: list[ItemRow] = []
    for i in range(n):
        lab = labels[i % 3]
        tier = tiers[i % 4]
        # Signals correlated with label + noise
        base = {
            "billing": np.array([0.8, 0.1, 0.1]),
            "technical": np.array([0.1, 0.8, 0.1]),
            "other": np.array([0.1, 0.1, 0.8]),
        }[lab]
        noise = rng.normal(0, 0.08, 3)
        sig = np.clip(base + noise, 0.01, 0.99)
        sig = sig / sig.sum()
        feats_jev = {
            "signal_billing": float(sig[0]),
            "signal_technical": float(sig[1]),
            "signal_other": float(sig[2]),
            "signal_multi_issue": float(rng.uniform(0, 0.4)),
            "signal_urgency_framing": float(rng.uniform(0, 0.4)),
        }
        # LLM slightly worse
        sig_l = np.clip(sig + rng.normal(0, 0.12, 3), 0.01, 0.99)
        sig_l = sig_l / sig_l.sum()
        feats_llm = {
            "signal_billing": float(sig_l[0]),
            "signal_technical": float(sig_l[1]),
            "signal_other": float(sig_l[2]),
            "signal_multi_issue": float(rng.uniform(0, 0.5)),
            "signal_urgency_framing": float(rng.uniform(0, 0.5)),
        }
        choice_j = labels[int(np.argmax(sig))]
        choice_l = labels[int(np.argmax(sig_l))]
        rows.append(
            ItemRow(
                item_id=f"syn-{i:03d}",
                tier=tier,
                label=lab,
                features={"jev": feats_jev, "adapter": feats_llm},
                verdict_choice={"jev": choice_j, "adapter": choice_l, "trivial": "other"},
                verdict_probs={
                    "jev": {
                        "billing": float(sig[0]),
                        "technical": float(sig[1]),
                        "other": float(sig[2]),
                    },
                    "adapter": {
                        "billing": float(sig_l[0]),
                        "technical": float(sig_l[1]),
                        "other": float(sig_l[2]),
                    },
                    "trivial": {"billing": 0.1, "technical": 0.1, "other": 0.8},
                },
                latency_ms={"jev": 50.0, "adapter": 300.0, "trivial": 1.0},
                input_tokens={"jev": 200, "adapter": 500, "trivial": 0},
                output_tokens={"jev": 20, "adapter": 80, "trivial": 0},
                cost_usd={"jev": 0.00001, "adapter": 0.0004, "trivial": 0.0},
            )
        )
    return rows


def test_exp2_yaml_loads():
    root = Path(__file__).resolve().parents[1]
    spec = load_experiment(root, "exp2_decomposition")
    assert spec.name == "exp2_decomposition"
    assert "verdict" in spec.questions
    assert "signal_billing" in spec.questions
    assert len([k for k in spec.questions if k.startswith("signal_")]) == 5
    arms = (spec.extras or {}).get("arms") or {}
    assert set(arms) >= {"A", "B", "C", "D", "E"}
    assert arms["A"]["mode"] == "zero_shot"
    assert arms["B"]["mode"] == "supervised"
    assert HONEST_CLAIM.split()[0] == "Jev"


def test_nested_cv_ece_is_outer_only():
    rows = _synth_rows(96)
    res = nested_cv_supervised(rows, "jev", seed=1)
    assert "error" not in res
    assert "ece_outer_held_out" in res["metrics"]
    assert "in_sample" not in str(res["metrics"]).lower()
    assert res["metrics"]["ece_note"].startswith("ECE measured on concatenated outer")
    assert len(res["fold_accuracy"]) >= 2
    assert res["fold_accuracy_std"] >= 0.0


def test_label_sweep_runs():
    rows = _synth_rows(100)
    sweep = run_label_sweep(rows, client="jev", Ns=(25, 50), seed=2)
    assert sweep["arm"] == "B"
    done = [p for p in sweep["points"] if not p.get("skipped")]
    assert len(done) >= 1
    assert "accuracy_mean" in done[0]


def test_fairness_note_in_payload(tmp_path: Path):
    """Write a tiny synthetic raw.jsonl and ensure framing lands in exp2.json."""
    import json

    rows = _synth_rows(60, seed=3)
    # Flatten to fixture-like raw for jev + adapter
    raw_lines = []
    for r in rows:
        for client in ("jev", "adapter", "trivial"):
            raw_lines.append(
                json.dumps(
                    {
                        "item_id": r.item_id,
                        "client": client,
                        "tier": r.tier,
                        "label": r.label,
                        "choice": r.verdict_choice[client],
                        "probabilities": r.verdict_probs[client],
                        "latency_ms": r.latency_ms[client],
                        "input_tokens": r.input_tokens[client],
                        "output_tokens": r.output_tokens[client],
                        "est_cost_usd": r.cost_usd[client],
                    }
                )
            )
    raw = tmp_path / "raw.jsonl"
    raw.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
    out = tmp_path / "exp2.json"
    payload = run_exp2_analysis(
        raw_path=raw,
        out_path=out,
        seed=3,
        n_boot=200,
        write_chart=False,
        llm_client="adapter",
    )
    assert FAIRNESS_NOTE[:40] in payload["fairness_note"]
    assert payload["comparable_pairs"]["zero_shot"] == "A vs C"
    assert payload["comparable_pairs"]["supervised"] == "B vs D"
    assert "B_minus_A" in payload["comparisons_paired_vs_A"]
    assert "C_minus_A" in payload["comparisons_paired_vs_A"]
    assert out.is_file()
