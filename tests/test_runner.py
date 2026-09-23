"""Offline tests for the runner — no network."""

from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path

import pytest
import yaml

from jevbench.clients.base import (
    ChoiceQuestion,
    Decision,
    SystemOneRequest,
)
from jevbench.clients.trivial import (
    KeywordRule,
    TrivialClient,
    TrivialClientConfig,
    TrivialQuestionSpec,
)
from jevbench.dataset import write_jsonl
from jevbench.experiment import ClientSpec, load_experiment
from jevbench.prereg import preregister
from jevbench.rate_limit import RateLimitConfig, TokenBucketLimiter
from jevbench.runner import (
    Runner,
    RunnerConfig,
    dry_run_estimate,
    load_completed_keys,
    work_units,
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
        "repeats": 2,
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
            {
                "name": "adapter_baseline",
                "type": "adapter",
                "provider": "openai",
                "model": "gpt-4o-mini",
                "llm_answer_mode": "probabilities",
            },
        ],
        "hypotheses": [
            {
                "id": "H1a",
                "statement": "flat",
                "falsified_when": "slope up",
            }
        ],
        "metrics": ["ECE"],
        "decision_rules": ["report curve"],
        "sample_size": {"min_items": 8, "per_tier": 2, "repeats": 2},
        "stopping_rule": "stop",
        "rate_limit": {"headroom": 0.8},
    }
    (root / "experiments" / "exp1_difficulty_calibration.yaml").write_text(
        yaml.safe_dump(exp), encoding="utf-8"
    )
    preregister(root, "exp1_difficulty_calibration")
    return root


class SequencedFakeClient:
    """Records call order; optionally asserts manifest already exists."""

    def __init__(self, name: str, serving_path: str = "native", manifest_path: Path | None = None):
        self.name = name
        self.serving_path = serving_path
        self.manifest_path = manifest_path
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def decide(self, request: SystemOneRequest) -> list[Decision]:
        if self.manifest_path is not None:
            assert self.manifest_path.is_file(), "manifest must exist before first call"
        with self._lock:
            self.calls.append(request.item_id)
        return [
            Decision(
                item_id=request.item_id,
                question_key=key,
                kind="choice",
                value="billing",
                probabilities={"billing": 0.9, "technical": 0.05, "other": 0.05},
                confidence=0.8,
                latency_ms=1.0,
                input_tokens=100,
                output_tokens=10,
                resolved_model=f"fake-{self.name}",
                serving_path=self.serving_path,
                attempt=1,
                error=None,
                raw={"fake": True},
            )
            for key in request.questions
        ]


def _factory(manifest_path: Path | None = None):
    clients: dict[str, SequencedFakeClient] = {}

    def make(cspec: ClientSpec):
        if cspec.type == "trivial":
            return TrivialClient(
                TrivialClientConfig(
                    specs={
                        "department": TrivialQuestionSpec(
                            keywords=(KeywordRule("billing", (r"refund", r"charg")),),
                            majority_class="other",
                        )
                    }
                )
            )
        c = SequencedFakeClient(cspec.name, manifest_path=manifest_path)
        clients[cspec.name] = c
        return c

    make.clients = clients  # type: ignore[attr-defined]
    return make


def test_dry_run_splits_jev_and_baselines(repo: Path):
    spec = load_experiment(repo, "exp1_difficulty_calibration")
    from jevbench.dataset import load_dataset
    from jevbench.prereg import datasets_root

    ds = load_dataset(datasets_root(repo), spec.task)
    est = dry_run_estimate(spec=spec, dataset=ds, repeats=2)
    assert est["split_usd"]["total"] == pytest.approx(
        est["split_usd"]["jev"] + est["split_usd"]["baselines"], abs=1e-5
    )
    assert est["split_usd"]["baselines"] > est["split_usd"]["jev"]
    assert "adapter_baseline" in est["by_client"]
    assert est["by_client"]["jev"]["est_output_tokens"] == 0

    result = Runner(
        RunnerConfig(
            repo_root=repo,
            experiment="exp1_difficulty_calibration",
            dry_run=True,
            progress=False,
        )
    ).run()
    assert isinstance(result, dict)
    assert "split_usd" in result


def test_manifest_written_before_any_call(repo: Path):
    # We don't know run_id ahead of time — use a factory that checks any
    # manifest under runs/ exists once decide is entered... Better: spy on
    # first call by wrapping after creating run via a custom approach.
    # Instead: run with concurrency=1 and a client that checks runs/*/manifest.json
    seen_manifest = {"ok": False}

    class GuardClient(SequencedFakeClient):
        def decide(self, request: SystemOneRequest) -> list[Decision]:
            runs = list((repo / "runs").glob("*/manifest.json"))
            assert runs, "manifest.json must exist before first API call"
            seen_manifest["ok"] = True
            return super().decide(request)

    def factory(cspec: ClientSpec):
        if cspec.type == "trivial":
            return TrivialClient(TrivialClientConfig(specs={}))
        return GuardClient(cspec.name)

    rdir = Runner(
        RunnerConfig(
            repo_root=repo,
            experiment="exp1_difficulty_calibration",
            repeats=1,
            concurrency=1,
            client_factory=factory,
            progress=False,
        )
    ).run()
    assert isinstance(rdir, Path)
    assert seen_manifest["ok"]
    manifest = json.loads((rdir / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["git_sha"]
    assert manifest["pricing_snapshot_date"] == "2026-09-19"
    assert manifest["labels_sha256"]
    assert "ECE" not in json.dumps(manifest.get("stats", {}))  # no metrics
    # raw streamed
    lines = (rdir / "raw.jsonl").read_text().strip().splitlines()
    assert len(lines) == 8 * 1 * 3  # items * repeats * clients


def test_repeat_passes_are_separate(repo: Path):
    factory = _factory()
    rdir = Runner(
        RunnerConfig(
            repo_root=repo,
            experiment="exp1_difficulty_calibration",
            repeats=2,
            concurrency=2,
            client_factory=factory,
            progress=False,
        )
    ).run()
    assert isinstance(rdir, Path)
    rows = [json.loads(l) for l in (rdir / "raw.jsonl").read_text().splitlines() if l.strip()]
    passes = {r["pass"] for r in rows}
    assert passes == {0, 1}
    # identical item+client appears twice (once per pass)
    keys = [(r["pass"], r["item_id"], r["client"]) for r in rows]
    assert len(keys) == len(set(keys))


def test_resume_skips_completed(repo: Path):
    factory = _factory()
    rdir = Runner(
        RunnerConfig(
            repo_root=repo,
            experiment="exp1_difficulty_calibration",
            repeats=1,
            concurrency=1,
            client_factory=factory,
            progress=False,
        )
    ).run()
    assert isinstance(rdir, Path)
    raw = rdir / "raw.jsonl"
    all_rows = raw.read_text().splitlines()
    # Truncate to simulate crash after 5 calls
    raw.write_text("\n".join(all_rows[:5]) + "\n", encoding="utf-8")
    # Reset status
    man = json.loads((rdir / "manifest.json").read_text())
    man["status"] = "running"
    man["ended_at"] = None
    (rdir / "manifest.json").write_text(json.dumps(man), encoding="utf-8")

    done = load_completed_keys(raw)
    assert len(done) == 5

    factory2 = _factory()
    rdir2 = Runner(
        RunnerConfig(
            repo_root=repo,
            experiment="exp1_difficulty_calibration",
            repeats=1,
            concurrency=1,
            resume_run_id=rdir.name,
            client_factory=factory2,
            progress=False,
        )
    ).run()
    assert rdir2 == rdir
    rows = [json.loads(l) for l in raw.read_text().splitlines() if l.strip()]
    assert len(rows) == 8 * 1 * 3
    assert len(rows) == 24


def test_runner_refuses_without_lock(tmp_path: Path):
    root = tmp_path / "nolock"
    # minimal broken setup
    (root / "datasets" / "support_tickets").mkdir(parents=True)
    write_jsonl(
        root / "datasets" / "support_tickets" / "items.jsonl",
        [{"id": "a", "state": "x", "tier": "trivial"}],
    )
    write_jsonl(
        root / "datasets" / "support_tickets" / "labels.jsonl",
        [{"id": "a", "label": "other", "labeler": "t", "labeled_at": "2026-09-19T00:00:00+00:00"}],
    )
    (root / "datasets" / "support_tickets" / "LABEL_GUIDE.md").write_text("g", encoding="utf-8")
    (root / "datasets" / "support_tickets" / "DISPUTED.md").write_text("d", encoding="utf-8")
    (root / "experiments").mkdir()
    (root / "runs").mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (root / "experiments" / "e.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "e",
                "status": "ready",
                "task": "support_tickets",
                "questions": {
                    "department": {
                        "kind": "choice",
                        "instructions": "t",
                        "criteria": {"other": None},
                    }
                },
                "clients": [{"name": "trivial", "type": "trivial"}],
            }
        ),
        encoding="utf-8",
    )
    from jevbench.prereg import PreregistrationError

    with pytest.raises(PreregistrationError):
        Runner(
            RunnerConfig(
                repo_root=root,
                experiment="e",
                progress=False,
            )
        ).run()


def test_work_units_and_rate_limiter():
    from jevbench.dataset import Item, Label, LabeledItem

    items = [
        LabeledItem(
            item=Item(id="i1", state="a", tier="trivial"),
            label=Label(id="i1", label="x", labeler="t", labeled_at="2026-09-19T00:00:00+00:00"),
        )
    ]
    clients = [ClientSpec(name="jev", type="jev"), ClientSpec(name="t", type="trivial")]
    units = work_units(items, clients, repeats=3)
    assert len(units) == 6
    assert units[0].key == "0:i1:jev"

    limiter = TokenBucketLimiter(
        RateLimitConfig(tokens_per_sec=1_000_000, requests_per_min=60_000)
    )
    waited = limiter.acquire(tokens=10, requests=1)
    assert waited == 0.0


def test_load_experiment_has_runner_fields():
    root = Path(__file__).resolve().parents[1]
    spec = load_experiment(root, "experiments/exp1_difficulty_calibration.yaml")
    assert spec.questions
    assert spec.clients
    assert spec.pricing_snapshot_date == "2026-09-19"
    assert "department" in spec.question_map()
    assert isinstance(spec.question_map()["department"], ChoiceQuestion)
