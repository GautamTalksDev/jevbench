"""Request-body hygiene and runner refusal when ChaosNLI hydration is unsafe."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from jevbench.clients.base import Decision, SystemOneRequest
from jevbench.prereg import PreregistrationError, preregister
from jevbench.request_canon import build_request_body
from jevbench.runner import Runner, RunnerConfig

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = (
    "text_status",
    "local_paraphrase_required",
    "fetch_required",
    "frozen_before_any_model_run",
)


def _questions() -> dict:
    return {
        "label": {
            "kind": "choice",
            "instructions": "Pick one.",
            "criteria": {"a": "A", "b": "B", "c": None},
        }
    }


def test_request_body_omits_placeholder_metadata() -> None:
    """Hydration-mocked placeholders must not leak status keys into the body."""
    placeholders = [
        {
            "id": "p1::paraphrase",
            "state": {
                "pair_id": "p1",
                "source": "mnli",
                "source_item_id": "p1",
                "text_status": "local_paraphrase_required",
                "frozen_before_any_model_run": True,
            },
        },
        {
            "id": "p2",
            "state": {
                "pair_id": "p2",
                "source": "snli",
                "text_status": "fetch_required",
            },
        },
    ]
    # Mock hydration: fill sentences the way hydrate_dataset would, in memory.
    hydrated = []
    for row in placeholders:
        state = dict(row["state"])
        state["premise"] = "Premise text."
        state["hypothesis"] = "Hypothesis text."
        if state.get("text_status") == "local_paraphrase_required":
            state["text_status"] = "local_paraphrase_hydrated"
        else:
            state["text_status"] = "fetched"
        hydrated.append((row["id"], state))

    for item_id, state in hydrated:
        body = build_request_body(
            item_id=item_id,
            client="jev",
            model="jev-1.13.0",
            state=state,
            questions=_questions(),
        )
        blob = json.dumps(body, ensure_ascii=False)
        for token in FORBIDDEN:
            assert token not in blob, f"{item_id} body still contains {token!r}"
        assert body["state"]["premise"] == "Premise text."
        assert body["state"]["hypothesis"] == "Hypothesis text."


def _tiny_chaosnli_repo(tmp_path: Path, *, paraphrase_text: bool = True) -> Path:
    """Minimal chaosnli task the runner can load, with a lock."""
    root = tmp_path / "repo"
    task = root / "datasets" / "chaosnli"
    (task / "local").mkdir(parents=True)
    (task / "cache").mkdir(parents=True)
    (root / "experiments").mkdir(parents=True)
    (root / "runs").mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='t'\n", encoding="utf-8")

    (task / "LABEL_GUIDE.md").write_text("# guide\n", encoding="utf-8")
    (task / "DISPUTED.md").write_text("# none\n", encoding="utf-8")
    items = [
        {
            "id": "a1",
            "role": "primary",
            "tier": "easy",
            "state": {"pair_id": "a1", "source": "mnli", "text_status": "fetch_required"},
        },
        {
            "id": "a1::paraphrase",
            "role": "paraphrase",
            "tier": "easy",
            "state": {
                "pair_id": "a1",
                "source": "mnli",
                "source_item_id": "a1",
                "text_status": "local_paraphrase_required",
                "frozen_before_any_model_run": True,
            },
        },
        {
            "id": "b1",
            "role": "primary",
            "tier": "hard",
            "state": {"pair_id": "b1", "source": "snli", "text_status": "fetch_required"},
        },
    ]
    labels = []
    for row in items:
        labels.append(
            {
                "id": row["id"],
                "label": "entailment",
                "labeler": "test",
                "labeled_at": "2026-09-23T00:00:00+00:00",
                "label_dist": [1.0, 0.0, 0.0],
                "label_count": [100, 0, 0],
            }
        )
    (task / "items.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in items), encoding="utf-8"
    )
    (task / "labels.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in labels), encoding="utf-8"
    )
    (task / "cache" / "text.jsonl").write_text(
        json.dumps({"id": "a1", "premise": "P-a", "hypothesis": "H-a", "source": "mnli"})
        + "\n"
        + json.dumps({"id": "b1", "premise": "P-b", "hypothesis": "H-b", "source": "snli"})
        + "\n",
        encoding="utf-8",
    )
    para_line = json.dumps(
        {"id": "a1", "premise": "Para-P", "hypothesis": "Para-H", "tier": "easy"}
    ) + "\n"
    para_path = task / "local" / "paraphrases.jsonl"
    if paraphrase_text:
        para_path.write_text(para_line, encoding="utf-8")
    else:
        para_path.write_text("", encoding="utf-8")
    digest = hashlib.sha256(para_path.read_bytes()).hexdigest()
    (task / "paraphrases.sha256").write_text(
        f"{digest}  local/paraphrases.jsonl\n", encoding="utf-8"
    )

    exp = {
        "name": "exp_hydrate_guard",
        "status": "ready",
        "model": "jev-1.13.0",
        "task": "chaosnli",
        "serving_path": "native",
        "pricing_snapshot_date": "2026-09-19",
        "geography_note": "test",
        "concurrency": 1,
        "repeats": 1,
        "description": "hydration guard",
        "hypotheses": [
            {
                "id": "H1",
                "statement": "test",
                "falsified_when": "never in this fixture",
            }
        ],
        "metrics": ["ece"],
        "decision_rules": ["fixture"],
        "sample_size": {"min_items": 1, "repeats": 1},
        "stopping_rule": "fixture",
        "questions": _questions(),
        "clients": [{"name": "jev", "type": "jev", "model": "jev-1.13.0"}],
    }
    (root / "experiments" / "exp_hydrate_guard.yaml").write_text(
        yaml.safe_dump(exp), encoding="utf-8"
    )
    preregister(root, "exp_hydrate_guard")
    return root


class _RecordingClient:
    def __init__(self) -> None:
        self.serving_path = "native"
        self.calls: list[SystemOneRequest] = []

    def decide(self, request: SystemOneRequest) -> list[Decision]:
        self.calls.append(request)
        out: list[Decision] = []
        for key, q in request.questions.items():
            kind = "choice"
            if hasattr(q, "instructions") and type(q).__name__.startswith("Noul"):
                kind = "noul"
            out.append(
                Decision(
                    item_id=request.item_id,
                    question_key=key,
                    kind=kind,  # type: ignore[arg-type]
                    value="a",
                    probabilities={"a": 1.0, "b": 0.0, "c": 0.0},
                    confidence=1.0,
                    latency_ms=1.0,
                    input_tokens=1,
                    output_tokens=1,
                    resolved_model="jev-1.13.0",
                    serving_path=self.serving_path,
                    attempt=1,
                    error=None,
                    raw={},
                )
            )
        return out


def test_runner_raises_when_paraphrase_hydration_fails(tmp_path: Path) -> None:
    root = _tiny_chaosnli_repo(tmp_path, paraphrase_text=False)
    # Empty paraphrase file still has a matching sha for that empty content.
    # Hydration cannot fill the paraphrase item.
    client = _RecordingClient()

    def factory(_spec):  # noqa: ANN001
        return client

    runner = Runner(
        RunnerConfig(
            repo_root=root,
            experiment="exp_hydrate_guard",
            client_factory=factory,
            skip_budget_guard=True,
            progress=False,
            confirm=True,
        )
    )
    with pytest.raises(PreregistrationError, match="not hydrated"):
        runner.run()
    assert len(client.calls) == 0, (
        f"expected ZERO client calls before refuse, got {len(client.calls)}"
    )


def test_runner_raises_when_paraphrase_sha256_mismatches(tmp_path: Path) -> None:
    root = _tiny_chaosnli_repo(tmp_path, paraphrase_text=True)
    # Corrupt the pin without changing the local file bytes.
    pin = root / "datasets" / "chaosnli" / "paraphrases.sha256"
    pin.write_text(
        "0" * 64 + "  local/paraphrases.jsonl\n",
        encoding="utf-8",
    )
    client = _RecordingClient()

    def factory(_spec):  # noqa: ANN001
        return client

    runner = Runner(
        RunnerConfig(
            repo_root=root,
            experiment="exp_hydrate_guard",
            client_factory=factory,
            skip_budget_guard=True,
            progress=False,
            confirm=True,
        )
    )
    with pytest.raises(PreregistrationError, match="does not match"):
        runner.run()
    assert len(client.calls) == 0, (
        f"expected ZERO client calls before refuse, got {len(client.calls)}"
    )
