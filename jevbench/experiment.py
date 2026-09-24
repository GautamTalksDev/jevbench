"""Experiment YAML schema and loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from jevbench.clients.base import (
    ChoiceQuestion,
    NoulQuestion,
    Question,
    ScoreQuestion,
)


class Hypothesis(BaseModel):
    id: str
    statement: str
    falsified_when: str  # explicit decision rule — what result kills this H


class SampleSize(BaseModel):
    min_items: int = Field(ge=1)
    per_tier: int | None = Field(default=None, ge=1)
    n_per_stratum: int | None = Field(default=None, ge=1)
    repeats: int = Field(default=3, ge=1)
    notes: str | None = None


class QuestionSpec(BaseModel):
    """One System One question declared in the experiment YAML."""

    kind: Literal["noul", "choice", "score"]
    instructions: Any
    criteria: dict[str, str | None] | list[str] | None = None

    def to_question(self) -> Question:
        if self.kind == "noul":
            return NoulQuestion(instructions=self.instructions)
        if self.kind == "choice":
            if not isinstance(self.criteria, dict) or not self.criteria:
                raise ValueError("choice questions require criteria: dict")
            return ChoiceQuestion(instructions=self.instructions, criteria=self.criteria)
        if self.kind == "score":
            if not isinstance(self.criteria, list) or not self.criteria:
                raise ValueError("score questions require criteria: ordered list")
            return ScoreQuestion(instructions=self.instructions, criteria=list(self.criteria))
        raise ValueError(f"unknown kind {self.kind!r}")


class ClientSpec(BaseModel):
    """One client arm in the experiment."""

    name: str
    type: Literal["jev", "adapter", "prefill", "trivial", "gliclass", "bart_mnli"]
    model: str | None = None
    serving_path: str | None = None  # overrides experiment serving_path for jev
    # Per-client repeat override (Amendment 11): local greedy models use 1.
    repeats: int | None = Field(default=None, ge=1)
    # Paper role: e.g. "supervised_in_domain_reference" for BART-MNLI.
    role: str | None = None
    # adapter
    provider: Literal["openai", "anthropic"] | None = None
    llm_answer_mode: Literal["probabilities", "discrete"] = "probabilities"
    # prefill
    prefill_backend: str = "transformers"
    prefill_base_url: str | None = None
    # trivial — optional keyed specs left to extras / future YAML
    extras: dict[str, Any] = Field(default_factory=dict)


class RateLimitSpec(BaseModel):
    """Optional override; defaults are 80% of published ceilings."""

    headroom: float = Field(default=0.80, gt=0.0, le=1.0)
    tokens_per_sec: float | None = None
    requests_per_min: float | None = None


class ExperimentSpec(BaseModel):
    name: str
    status: Literal["stub", "draft", "locked", "ready"] = "draft"
    model: str = "jev-1.13.0"
    task: str  # datasets/<task>/
    description: str = ""
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    decision_rules: list[str] = Field(default_factory=list)
    sample_size: SampleSize | None = None
    stopping_rule: str = (
        "Stop after the pre-registered sample size is reached. "
        "Do not collect additional items because interim results look weak or strong."
    )
    serving_path: str = "native"
    baseline_model: str | None = None
    # Runner fields
    questions: dict[str, QuestionSpec] = Field(default_factory=dict)
    clients: list[ClientSpec] = Field(default_factory=list)
    repeats: int | None = Field(default=None, ge=1)
    concurrency: int = Field(default=1, ge=1)
    pricing_snapshot_date: str = "2026-09-19"
    geography_note: str = (
        "Record geography of the machine making scored calls. "
        "TypeSafe published latency numbers come from US West Coast laptops."
    )
    rate_limit: RateLimitSpec = Field(default_factory=RateLimitSpec)
    extras: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _name_ok(cls, v: str) -> str:
        if not v or "/" in v or "\\" in v:
            raise ValueError(f"invalid experiment name: {v!r}")
        return v

    def effective_repeats(self) -> int:
        if self.repeats is not None:
            return self.repeats
        if self.sample_size is not None:
            return self.sample_size.repeats
        return 1

    def question_map(self) -> dict[str, Question]:
        if not self.questions:
            raise ValueError(
                f"experiment {self.name!r} has no questions — "
                "declare questions: in the YAML before running"
            )
        return {k: q.to_question() for k, q in self.questions.items()}


def experiments_dir(repo_root: Path) -> Path:
    return repo_root / "experiments"


def resolve_experiment_path(repo_root: Path, experiment: str) -> Path:
    """Accept bare name, path with/without .yaml, or absolute path."""
    p = Path(experiment)
    if p.is_file():
        return p
    cand = experiments_dir(repo_root) / experiment
    if cand.is_file():
        return cand
    if not experiment.endswith((".yaml", ".yml")):
        for ext in (".yaml", ".yml"):
            cand = experiments_dir(repo_root) / f"{experiment}{ext}"
            if cand.is_file():
                return cand
    raise FileNotFoundError(
        f"experiment not found: {experiment!r} (looked under {experiments_dir(repo_root)})"
    )


def load_experiment(repo_root: Path, experiment: str) -> ExperimentSpec:
    path = resolve_experiment_path(repo_root, experiment)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected mapping at top level")
    known = set(ExperimentSpec.model_fields)
    extras = {k: v for k, v in data.items() if k not in known}
    core = {k: v for k, v in data.items() if k in known}
    if extras:
        core["extras"] = {**(core.get("extras") or {}), **extras}
    if "name" not in core:
        core["name"] = path.stem
    return ExperimentSpec.model_validate(core)
