"""EXP-3 moat control tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jevbench.exp3 import LATENCY_NOTE, run_exp3_analysis
from jevbench.experiment import load_experiment


def test_exp3_yaml_loads():
    root = Path(__file__).resolve().parents[1]
    spec = load_experiment(root, "exp3_moat_control")
    assert spec.name == "exp3_moat_control"
    types = {c.type for c in spec.clients}
    assert "jev" in types and "prefill" in types
    assert "gliclass" in types and "adapter" in types
    assert spec.effective_repeats() == 3
    assert "ECE" in " ".join(spec.metrics) or "delta_ece" in " ".join(spec.metrics).lower()


def _synth_raw(tmp_path: Path, n: int = 40) -> Path:
    labels = ["billing", "technical", "other"]
    lines = []
    for i in range(n):
        lab = labels[i % 3]
        # Jev better calibrated: probs closer to one-hot on true label
        j = np.array([0.1, 0.1, 0.1])
        j[labels.index(lab)] = 0.8
        j = j / j.sum()
        # Prefill overconfident / miscalibrated
        p = np.array([0.05, 0.05, 0.05])
        p[labels.index(lab)] = 0.9
        # Flip some prefill mass away from truth → higher ECE
        if i % 5 == 0:
            p = np.array([0.7, 0.2, 0.1])
        p = p / p.sum()
        for client, probs, lat in (
            ("jev", j, 40.0),
            ("prefill", p, 12.0),
            ("adapter", j * 0.9 + 0.033, 300.0),
            ("gliclass", np.ones(3) / 3, 80.0),
        ):
            probs = probs / probs.sum()
            choice = labels[int(np.argmax(probs))]
            lines.append(
                json.dumps(
                    {
                        "item_id": f"m-{i:03d}",
                        "client": client,
                        "pass": 0,
                        "tier": ["trivial", "easy", "hard", "ambiguous"][i % 4],
                        "label": lab,
                        "choice": choice,
                        "probabilities": {
                            "billing": float(probs[0]),
                            "technical": float(probs[1]),
                            "other": float(probs[2]),
                        },
                        "latency_ms": lat,
                        "decisions": [
                            {
                                "question_key": "department",
                                "kind": "choice",
                                "value": choice,
                                "probabilities": {
                                    "billing": float(probs[0]),
                                    "technical": float(probs[1]),
                                    "other": float(probs[2]),
                                },
                                "latency_ms": lat,
                                "raw": {
                                    "latency": {
                                        "wall_clock_ms": lat,
                                        "compute_only_ms": lat * 0.5,
                                        "fair_comparison": False,
                                    }
                                },
                            }
                        ],
                    }
                )
            )
        # pass 1 for prefill permutation stability check
        lines.append(
            json.dumps(
                {
                    "item_id": f"m-{i:03d}",
                    "client": "prefill",
                    "pass": 1,
                    "tier": ["trivial", "easy", "hard", "ambiguous"][i % 4],
                    "label": lab,
                    "choice": lab,
                    "probabilities": {
                        "billing": float(j[0]),
                        "technical": float(j[1]),
                        "other": float(j[2]),
                    },
                    "latency_ms": 12.0,
                }
            )
        )
    path = tmp_path / "raw.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_exp3_analysis_deciding_metric(tmp_path: Path):
    raw = _synth_raw(tmp_path)
    out = tmp_path / "exp3.json"
    payload = run_exp3_analysis(
        raw_path=raw, out_path=out, n_boot=300, seed=1, adapter_client="adapter"
    )
    assert out.is_file()
    d = payload["paired_delta_ece_prefill_minus_jev"]
    assert "point" in d
    assert "ci_bca" in d and "ci_percentile" in d
    assert LATENCY_NOTE[:20] in payload["latency_note"]
    assert payload["arms"]["prefill"]["latency"]["wall_clock_ms"]["fair_comparison"] is False
    assert "verdict" in payload
