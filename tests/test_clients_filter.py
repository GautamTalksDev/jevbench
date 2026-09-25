"""--clients filter and analyze completeness."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest
import yaml

from jevbench.clients.base import Decision, SystemOneRequest
from jevbench.dataset import load_dataset
from jevbench.experiment import load_experiment
from jevbench.prereg import datasets_root, preregister
from jevbench.runner import (
    Runner,
    RunnerConfig,
    dry_run_estimate,
    resolve_client_specs,
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "datasets").mkdir(parents=True)
    (root / "experiments").mkdir()
    (root / "runs").mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='jevbench'\n", encoding="utf-8")
    src = Path(__file__).resolve().parents[1] / "datasets" / "support_tickets"
    shutil.copytree(src, root / "datasets" / "support_tickets")
    exp = {
        "name": "exp1_difficulty_calibration",
        "status": "ready",
        "model": "jev-1.13.0",
        "task": "support_tickets",
        "serving_path": "native",
        "pricing_snapshot_date": "2026-09-19",
        "geography_note": "test-lab",
        "concurrency": 2,
        "repeats": 1,
        "description": "runner test",
        "questions": {
            "department": {
                "kind": "choice",
                "instructions": "Which team?",
                "criteria": {
                    "billing": "money",
                    "technical": "bugs",
                    "other": None,
                },
            }
        },
        "clients": [
            {"name": "jev", "type": "jev", "model": "jev-1.13.0"},
            {"name": "trivial", "type": "trivial"},
        ],
        "hypotheses": [
            {"id": "H1a", "statement": "flat", "falsified_when": "slope up"}
        ],
        "metrics": ["ECE"],
        "decision_rules": ["report curve"],
        "sample_size": {"min_items": 8, "per_tier": 2, "repeats": 1},
        "stopping_rule": "stop",
    }
    (root / "experiments" / "exp1_difficulty_calibration.yaml").write_text(
        yaml.safe_dump(exp), encoding="utf-8"
    )
    preregister(root, "exp1_difficulty_calibration")
    return root


class _RecClient:
    def __init__(self, name: str) -> None:
        self.serving_path = "native"
        self.name = name
        self.calls: list[str] = []

    def decide(self, request: SystemOneRequest) -> list[Decision]:
        self.calls.append(request.item_id)
        return [
            Decision(
                item_id=request.item_id,
                question_key="department",
                kind="choice",
                value="billing",
                probabilities={"billing": 1.0, "technical": 0.0, "other": 0.0},
                confidence=1.0,
                latency_ms=1.0,
                input_tokens=1,
                output_tokens=1,
                resolved_model="mock",
                serving_path=self.serving_path,
                attempt=1,
                error=None,
                raw={},
            )
        ]


def test_resolve_client_specs_unknown_raises(repo: Path) -> None:
    spec = load_experiment(repo, "exp1_difficulty_calibration")
    with pytest.raises(ValueError, match="unknown client"):
        resolve_client_specs(spec, ["no_such_client"])


def test_dry_run_respects_clients(repo: Path) -> None:
    spec = load_experiment(repo, "exp1_difficulty_calibration")
    ds = load_dataset(datasets_root(repo), spec.task)
    selected = resolve_client_specs(spec, ["jev"])
    estimate = dry_run_estimate(
        spec=spec, dataset=ds, repeats=1, clients=selected
    )
    assert list(estimate["by_client"]) == ["jev"]
    assert estimate["clients_this_invocation"] == ["jev"]


def test_resume_appends_new_client_without_resend(repo: Path) -> None:
    built: dict[str, _RecClient] = {}

    def factory(cspec):  # noqa: ANN001
        c = _RecClient(cspec.name)
        built[cspec.name] = c
        return c

    rdir = Runner(
        RunnerConfig(
            repo_root=repo,
            experiment="exp1_difficulty_calibration",
            repeats=1,
            concurrency=1,
            client_factory=factory,
            client_names=["jev"],
            progress=False,
            skip_budget_guard=True,
            confirm=True,
        )
    ).run()
    assert isinstance(rdir, Path)
    manifest = json.loads((rdir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["clients_this_invocation"] == ["jev"]
    n_after_jev = len(
        [ln for ln in (rdir / "raw.jsonl").read_text().splitlines() if ln.strip()]
    )
    assert built["jev"].calls
    built.clear()

    Runner(
        RunnerConfig(
            repo_root=repo,
            experiment="exp1_difficulty_calibration",
            repeats=1,
            concurrency=1,
            client_factory=factory,
            client_names=["trivial"],
            resume_run_id=rdir.name,
            progress=False,
            skip_budget_guard=True,
            confirm=True,
        )
    ).run()
    manifest2 = json.loads((rdir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest2["clients_this_invocation"] == ["trivial"]
    assert len(manifest2["client_invocations"]) == 2
    assert "jev" not in built
    assert "trivial" in built
    n_after = len(
        [ln for ln in (rdir / "raw.jsonl").read_text().splitlines() if ln.strip()]
    )
    assert n_after > n_after_jev


def test_analyze_completeness_gate(tmp_path: Path) -> None:
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_exp1_analyze.py"
    spec = importlib.util.spec_from_file_location("run_exp1_analyze", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    raw = tmp_path / "raw.jsonl"
    raw.write_text("{}\n", encoding="utf-8")
    # Real EXP-1 YAML expects thousands of keys — empty raw must fail.
    with pytest.raises(RuntimeError, match="incomplete"):
        mod.assert_clients_complete(
            raw,
            repo_root=Path(__file__).resolve().parents[1],
            experiment="exp1_difficulty_calibration",
            client_names=["jev"],
        )
