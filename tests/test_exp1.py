"""EXP-1 labelling agreement + contamination-guard tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jevbench.agreement import cohen_kappa, compute_agreement, select_agreement_subset
from jevbench.experiment import load_experiment
from jevbench.exp1 import normalize_records, run_exp1_analysis


def test_cohen_kappa_perfect():
    k, po, pe = cohen_kappa(["a", "b", "a"], ["a", "b", "a"])
    assert k == pytest.approx(1.0)
    assert po == pytest.approx(1.0)


def test_cohen_kappa_chance():
    # All labeler1 = a, half labeler2 = a → low kappa
    y1 = ["a"] * 10
    y2 = ["a"] * 5 + ["b"] * 5
    k, po, _ = cohen_kappa(y1, y2)
    assert po == pytest.approx(0.5)
    assert k < 0.1


def test_select_agreement_subset_deterministic():
    ids = [f"i{i}" for i in range(100)]
    a = select_agreement_subset(ids, fraction=0.15, seed=1)
    b = select_agreement_subset(ids, fraction=0.15, seed=1)
    assert a == b
    assert len(a) == 15


def test_compute_agreement_from_tmp(tmp_path: Path):
    (tmp_path / "labels.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "id": f"x{i}",
                    "label": "billing" if i % 2 == 0 else "technical",
                    "labeler": "a",
                    "labeled_at": "2026-09-22T00:00:00+00:00",
                }
            )
            for i in range(10)
        )
        + "\n",
        encoding="utf-8",
    )
    # Pass2 agrees on 9/10
    rows = []
    for i in range(10):
        lab = "billing" if i % 2 == 0 else "technical"
        if i == 3:
            lab = "other"
        rows.append(
            {
                "id": f"x{i}",
                "label": lab,
                "labeler": "b",
                "labeled_at": "2026-09-22T01:00:00+00:00",
            }
        )
    (tmp_path / "labels_pass2.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    res = compute_agreement(tmp_path, kappa_floor=0.6)
    assert res.n_paired == 10
    assert res.passes_floor


def test_label_cli_forbids_prediction_flags():
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            sys.executable,
            str(root / "tools" / "label.py"),
            "--labeler",
            "t",
            "--run-dir",
            "runs/x",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "forbidden" in (proc.stderr + proc.stdout).lower()


def test_exp1_yaml_loads_with_n_per_stratum():
    root = Path(__file__).resolve().parents[1]
    spec = load_experiment(root, "exp1_difficulty_calibration")
    assert spec.sample_size is not None
    assert spec.sample_size.n_per_stratum == 750
    assert spec.model == "jev-1.13.0"
    assert spec.serving_path == "native"
    assert spec.effective_repeats() == 3


def test_exp1_analysis_smoke_on_fixture(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    raw = root / "runs" / "offline_fixture" / "raw.jsonl"
    if not raw.is_file():
        pytest.skip("offline fixture missing")
    out = tmp_path / "exp1.json"
    payload = run_exp1_analysis(
        raw_path=raw,
        out_path=out,
        jev_client="jev",
        baseline_client="adapter",
        n_boot=200,
        scored=False,
    )
    assert out.is_file()
    assert "verdict" in payload
    assert payload["jev"]["primary_delta_ece"]["point"] == payload["jev"][
        "primary_delta_ece"
    ]["point"]
    assert "ci_bca" in payload["jev"]["primary_delta_ece"]
    assert "ci_percentile" in payload["jev"]["primary_delta_ece"]


def test_normalize_fixture_records():
    root = Path(__file__).resolve().parents[1]
    raw = root / "runs" / "offline_fixture" / "raw.jsonl"
    rows = [
        json.loads(line)
        for line in raw.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:5]
    norm = normalize_records(rows)
    assert len(norm) == 5
    assert "probabilities" in norm[0]
