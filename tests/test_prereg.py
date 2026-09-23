"""Offline tests for dataset loader, preregister, and verify-labels."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from jevbench.dataset import (
    TIERS,
    balanced_sample,
    load_dataset,
    stratify_by_tier,
    write_jsonl,
)
from jevbench.experiment import load_experiment
from jevbench.prereg import (
    LabelContaminationError,
    PreregistrationError,
    assert_hashes_unchanged,
    load_lock,
    preregister,
    verify_labels,
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Minimal repo copy: datasets + one experiment YAML."""
    root = tmp_path / "repo"
    (root / "datasets").mkdir(parents=True)
    (root / "experiments").mkdir()
    (root / "runs").mkdir()

    # Copy demo task from the real repo when available; else synthesize
    src = Path(__file__).resolve().parents[1] / "datasets" / "support_tickets"
    dest = root / "datasets" / "support_tickets"
    if src.is_dir():
        shutil.copytree(src, dest)
    else:
        dest.mkdir()
        write_jsonl(
            dest / "items.jsonl",
            [
                {"id": "a", "state": "refund please", "tier": "trivial"},
                {"id": "b", "state": "app crash", "tier": "easy"},
                {"id": "c", "state": "mixed", "tier": "hard"},
                {"id": "d", "state": "???", "tier": "ambiguous"},
            ],
        )
        write_jsonl(
            dest / "labels.jsonl",
            [
                {
                    "id": x,
                    "label": "billing",
                    "labeler": "t",
                    "labeled_at": "2026-09-19T00:00:00+00:00",
                }
                for x in "abcd"
            ],
        )
        (dest / "LABEL_GUIDE.md").write_text("# guide\n", encoding="utf-8")
        (dest / "DISPUTED.md").write_text("# disputed\n", encoding="utf-8")

    (root / "experiments" / "exp1_difficulty_calibration.yaml").write_text(
        """
name: exp1_difficulty_calibration
status: ready
model: jev-1.13.0
task: support_tickets
description: Test experiment.
hypotheses:
  - id: H1a
    statement: ECE flat across tiers.
    falsified_when: Slope CI excludes zero upward.
metrics:
  - ECE with occupancy
decision_rules:
  - Report ECE-vs-accuracy curve.
sample_size:
  min_items: 4
  per_tier: 1
  repeats: 3
stopping_rule: Stop at pre-registered N.
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text("[project]\nname='jevbench'\n", encoding="utf-8")
    return root


def test_load_dataset_and_stratify(repo: Path):
    ds = load_dataset(repo / "datasets", "support_tickets")
    assert len(ds) >= 4
    groups = stratify_by_tier(ds)
    assert set(groups) == set(TIERS)
    for tier in TIERS:
        assert all(x.tier == tier for x in groups[tier])


def test_balanced_sample(repo: Path):
    ds = load_dataset(repo / "datasets", "support_tickets")
    sample = balanced_sample(ds, n_per_tier=1, seed=42)
    assert len(sample) == 4
    assert {x.tier for x in sample} == set(TIERS)
    # deterministic
    sample2 = balanced_sample(ds, n_per_tier=1, seed=42)
    assert [x.id for x in sample] == [x.id for x in sample2]


def test_balanced_sample_rejects_underfilled_tier(repo: Path):
    ds = load_dataset(repo / "datasets", "support_tickets")
    with pytest.raises(ValueError, match="need"):
        balanced_sample(ds, n_per_tier=10_000, seed=0)


def test_preregister_writes_md_and_lock(repo: Path):
    md, lock_file, lock = preregister(repo, "exp1_difficulty_calibration")
    assert md.is_file()
    assert lock_file.is_file()
    text = md.read_text(encoding="utf-8")
    assert "LOCKED" in text
    assert lock.items_sha256 in text
    assert lock.labels_sha256 in text
    assert "Falsified when" in text or "falsified" in text.lower()
    loaded = load_lock(repo)
    assert loaded is not None
    assert loaded.items_sha256 == lock.items_sha256
    assert_hashes_unchanged(repo)


def test_preregister_refuses_stub(repo: Path):
    (repo / "experiments" / "stub_exp.yaml").write_text(
        "name: stub_exp\nstatus: stub\nmodel: jev-1.13.0\ntask: support_tickets\n",
        encoding="utf-8",
    )
    with pytest.raises(PreregistrationError, match="stub"):
        preregister(repo, "stub_exp")


def test_hash_change_refuses_run_check(repo: Path):
    preregister(repo, "exp1_difficulty_calibration")
    labels = repo / "datasets" / "support_tickets" / "labels.jsonl"
    # Tamper: append a blank line still changes? empty line might not if we strip —
    # append a comment-like change by rewriting a labeler field
    rows = [json.loads(line) for line in labels.read_text().splitlines() if line.strip()]
    rows[0]["labeler"] = "contaminated"
    write_jsonl(labels, rows)
    with pytest.raises(PreregistrationError, match="hash mismatch"):
        assert_hashes_unchanged(repo)


def test_verify_labels_ok_before_runs(repo: Path):
    preregister(repo, "exp1_difficulty_calibration")
    verify_labels(repo)  # no scored runs yet


def test_verify_labels_fails_after_run_if_labels_change(repo: Path):
    _, _, lock = preregister(repo, "exp1_difficulty_calibration")
    run = repo / "runs" / "run-001"
    run.mkdir()
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "started_at": "2026-09-19T20:00:00+00:00",
                "task": "support_tickets",
                "labels_sha256": lock.labels_sha256,
                "items_sha256": lock.items_sha256,
            }
        ),
        encoding="utf-8",
    )
    (run / "raw.jsonl").write_text("{}\n", encoding="utf-8")

    verify_labels(repo)  # still ok

    labels = repo / "datasets" / "support_tickets" / "labels.jsonl"
    rows = [json.loads(line) for line in labels.read_text().splitlines() if line.strip()]
    rows[0]["label"] = "technical"  # post-hoc adjudication
    write_jsonl(labels, rows)

    with pytest.raises(LabelContaminationError, match="CONTAMINATION"):
        verify_labels(repo)


def test_load_experiment_exp1_from_real_repo():
    root = Path(__file__).resolve().parents[1]
    if not (root / "experiments" / "exp1_difficulty_calibration.yaml").is_file():
        pytest.skip("not in full repo")
    spec = load_experiment(root, "exp1_difficulty_calibration")
    assert spec.task == "chaosnli"
    assert spec.status == "ready"
    assert len(spec.hypotheses) >= 1
    assert any("soft" in m.lower() for m in spec.metrics)
