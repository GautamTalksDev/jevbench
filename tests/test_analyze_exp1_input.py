"""analyze-exp1 input selection + null verdict (no statistical-method changes)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from jevbench.exp1 import build_verdict, no_verdict_reason


ROOT = Path(__file__).resolve().parents[1]


def _load_analyze_mod():
    path = ROOT / "scripts" / "run_exp1_analyze.py"
    spec = importlib.util.spec_from_file_location("run_exp1_analyze", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_refuse_offline_fixture_manifest() -> None:
    mod = _load_analyze_mod()
    man = json.loads(
        (ROOT / "runs" / "offline_fixture" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    with pytest.raises(RuntimeError, match="chaosnli|scored|offline"):
        mod.assert_exp1_manifest(man, run_id="offline_fixture")


def test_resolve_rejects_offline_fixture_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_analyze_mod()
    runs = tmp_path / "runs"
    # Copy-shaped offline fixture under runs/offline_fixture
    of = runs / "offline_fixture"
    of.mkdir(parents=True)
    (of / "manifest.json").write_text(
        json.dumps(
            {
                "task": "support_tickets",
                "scored": False,
                "run_id": "offline_fixture",
            }
        ),
        encoding="utf-8",
    )
    (of / "raw.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="chaosnli|scored"):
        mod.resolve_exp1_run(tmp_path, "offline_fixture")


def test_latest_exp1_skips_aborted_and_unscored(tmp_path: Path) -> None:
    mod = _load_analyze_mod()
    runs = tmp_path / "runs"
    runs.mkdir()

    def _write(name: str, *, task: str, scored: bool | None) -> None:
        d = runs / name
        d.mkdir()
        man = {"task": task, "run_id": name}
        if scored is not None:
            man["scored"] = scored
        (d / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
        (d / "raw.jsonl").write_text(
            json.dumps(
                {
                    "item_id": "x",
                    "tier": "easy",
                    "client": "jev",
                    "role": "primary",
                    "label": "entailment",
                    "probabilities": {"entailment": 1.0, "neutral": 0.0, "contradiction": 0.0},
                }
            )
            + "\n",
            encoding="utf-8",
        )

    _write(
        "_aborted_nokey_exp1_difficulty_calibration_20260925T065814Z_6d009262",
        task="chaosnli",
        scored=None,
    )
    _write(
        "exp1_difficulty_calibration_20260925T060000Z_old",
        task="chaosnli",
        scored=None,
    )
    _write(
        "exp1_difficulty_calibration_20260925T070159Z_new",
        task="chaosnli",
        scored=None,
    )
    _write("exp1_bogus_support", task="support_tickets", scored=False)

    run_id, raw, man = mod.resolve_exp1_run(tmp_path, "latest-exp1")
    assert run_id == "exp1_difficulty_calibration_20260925T070159Z_new"
    assert man["task"] == "chaosnli"
    assert raw.name == "raw.jsonl"


def test_verdict_null_when_delta_or_p_missing() -> None:
    payload = {
        "gates": {},
        "jev": {
            "delta_corrected": None,
            "p_value": None,
            "verdict": None,
        },
        "jev_noul": {},
    }
    assert build_verdict(payload) is None
    reason = no_verdict_reason(payload)
    assert "delta_corrected" in reason or "p_value" in reason
    # Must not fake Amendment 9 "inconclusive"
    assert build_verdict(payload) != "inconclusive"


def test_verdict_null_when_only_p_missing() -> None:
    payload = {
        "gates": {},
        "jev": {
            "delta_corrected": 0.1,
            "p_value": None,
            "verdict": "tracks",
        },
        "jev_noul": {},
    }
    assert build_verdict(payload) is None


def test_cli_analyze_exp1_requires_run() -> None:
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "jevbench", "analyze-exp1", "--quick"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    blob = (proc.stderr + proc.stdout).lower()
    assert "run" in blob or "required" in blob or "missing" in blob


def test_script_refuses_missing_run_flag() -> None:
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_exp1_analyze.py"), "--quick"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "--run" in (proc.stderr + proc.stdout)


def test_preflight_strata_refuse() -> None:
    mod = _load_analyze_mod()
    rows = [
        {
            "item_id": f"e{i}",
            "tier": "easy",
            "role": "primary",
            "probabilities": {"a": 1.0},
            "label_dist": [1.0, 0.0, 0.0],
        }
        for i in range(10)
    ] + [
        {
            "item_id": f"h{i}",
            "tier": "hard",
            "role": "primary",
            "probabilities": {"a": 1.0},
            "label_dist": [1.0, 0.0, 0.0],
        }
        for i in range(10)
    ]
    with pytest.raises(RuntimeError, match="750"):
        mod.preflight_print_and_check(
            run_id="fake",
            raw_path=Path("x"),
            rows=rows,
            n_records=20,
        )
