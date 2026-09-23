"""Pre-registration lock: hypotheses + dataset content hashes.

Once ``jevbench preregister`` writes the lock, scored runs must refuse if
``items.jsonl`` / ``labels.jsonl`` bytes change. ``verify-labels`` additionally
fails if labels change after the first scored run exists — the contamination
guard this project exists to enforce.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jevbench.dataset import dataset_hashes, load_dataset, sha256_file
from jevbench.experiment import ExperimentSpec, load_experiment

LOCK_PATH_REL = Path("preregistration.lock.json")
PREREG_MD_REL = Path("PREREGISTRATION.md")


@dataclass
class PreregistrationLock:
    experiment: str
    task: str
    model: str
    items_sha256: str
    labels_sha256: str
    locked_at: str
    experiment_yaml: str
    preregistration_md: str
    tier_counts: dict[str, int]
    n_items: int
    n_labels: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreregistrationLock:
        return cls(**{k: data[k] for k in cls.__dataclass_fields__})


class PreregistrationError(RuntimeError):
    """Hash mismatch or missing lock — do not run scored experiments."""


class LabelContaminationError(RuntimeError):
    """labels.jsonl changed after the first scored run — protocol breach."""


def lock_path(repo_root: Path) -> Path:
    return repo_root / LOCK_PATH_REL


def prereg_md_path(repo_root: Path) -> Path:
    return repo_root / PREREG_MD_REL


def load_lock(repo_root: Path) -> PreregistrationLock | None:
    path = lock_path(repo_root)
    if not path.is_file():
        return None
    return PreregistrationLock.from_dict(json.loads(path.read_text(encoding="utf-8")))


def write_lock(repo_root: Path, lock: PreregistrationLock) -> Path:
    path = lock_path(repo_root)
    path.write_text(
        json.dumps(lock.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def datasets_root(repo_root: Path) -> Path:
    return repo_root / "datasets"


def build_lock(
    repo_root: Path,
    spec: ExperimentSpec,
    *,
    experiment_yaml_rel: str,
) -> PreregistrationLock:
    ds = load_dataset(datasets_root(repo_root), spec.task)
    hashes = dataset_hashes(ds.root)
    return PreregistrationLock(
        experiment=spec.name,
        task=spec.task,
        model=spec.model,
        items_sha256=hashes["items_sha256"],
        labels_sha256=hashes["labels_sha256"],
        locked_at=datetime.now(UTC).replace(microsecond=0).isoformat(),
        experiment_yaml=experiment_yaml_rel,
        preregistration_md=str(PREREG_MD_REL),
        tier_counts=ds.tier_counts(),
        n_items=len(ds.items),
        n_labels=len(ds.labels),
    )


def render_preregistration_md(
    spec: ExperimentSpec,
    lock: PreregistrationLock,
) -> str:
    hyps = spec.hypotheses or []
    metrics = spec.metrics or ["(none listed in experiment YAML)"]
    rules = spec.decision_rules or []
    ss = spec.sample_size

    lines: list[str] = [
        f"# Pre-registration — {spec.name}",
        "",
        "**Status:** LOCKED",
        f"**Experiment:** `{spec.name}`",
        f"**Task:** `datasets/{spec.task}/`",
        f"**Pinned model:** `{spec.model}`",
        f"**Serving path:** `{spec.serving_path}`",
        f"**Locked at (UTC):** {lock.locked_at}",
        "",
        "This document locks hypotheses, metrics, and decision rules **before**",
        "scored API calls. Analyses that diverge are labelled exploratory.",
        "",
        "---",
        "",
        "## Content hashes (binding)",
        "",
        "Scored runs **must refuse** if either hash changes.",
        "",
        "| File | SHA-256 |",
        "|---|---|",
        f"| `datasets/{spec.task}/items.jsonl` | `{lock.items_sha256}` |",
        f"| `datasets/{spec.task}/labels.jsonl` | `{lock.labels_sha256}` |",
        "",
        f"Machine-readable lock: `{LOCK_PATH_REL}`",
        "",
        f"- Items: **{lock.n_items}**",
        f"- Labels: **{lock.n_labels}**",
        f"- Tier counts: `{lock.tier_counts}`",
        "",
        "---",
        "",
        "## Description",
        "",
        spec.description.strip() or "_(none)_",
        "",
        "---",
        "",
        "## Hypotheses & falsification rules",
        "",
    ]

    if not hyps:
        lines.append("_No hypotheses in experiment YAML — add them before locking._")
        lines.append("")
    for h in hyps:
        lines.extend(
            [
                f"### {h.id}",
                "",
                f"**Statement:** {h.statement}",
                "",
                f"**Falsified when:** {h.falsified_when}",
                "",
            ]
        )

    lines.extend(
        [
            "---",
            "",
            "## Metrics",
            "",
        ]
    )
    for m in metrics:
        lines.append(f"- {m}")
    lines.append("")

    lines.extend(
        [
            "---",
            "",
            "## Decision rules",
            "",
        ]
    )
    if not rules:
        lines.append("_See falsification clauses on each hypothesis._")
        lines.append("")
    for r in rules:
        lines.append(f"- {r}")
    lines.append("")

    lines.extend(["---", "", "## Sample size", ""])
    if ss is None:
        lines.append("_Not specified in experiment YAML._")
        lines.append("")
    else:
        lines.append(f"- **min_items:** {ss.min_items}")
        if ss.per_tier is not None:
            lines.append(f"- **per_tier:** {ss.per_tier}")
        lines.append(f"- **repeats:** {ss.repeats}")
        if ss.notes:
            lines.append(f"- **notes:** {ss.notes}")
        lines.append("")

    lines.extend(
        [
            "---",
            "",
            "## Stopping rule",
            "",
            spec.stopping_rule.strip(),
            "",
            "---",
            "",
            "## Contamination guard",
            "",
            "1. Labels are fixed at lock time (hash above).",
            "2. `jevbench verify-labels` fails if `labels.jsonl` changes after",
            "   the first scored run under `runs/`.",
            "3. Adjudicating a label after seeing model output is a protocol",
            "   breach — the tool makes accidental contamination fail CI.",
            "",
            "---",
            "",
            "## AI-usage statement",
            "",
            "> Large language models were used for literature search, prose drafting,",
            "> and code scaffolding under the author's direction. All experimental",
            "> design, claims, analysis, and errors are the author's own. Every",
            "> factual claim was verified against the cited primary source.",
            "",
        ]
    )
    if spec.baseline_model:
        lines.extend(
            [
                "---",
                "",
                "## Baseline",
                "",
                f"- LLM baseline model: `{spec.baseline_model}`",
                "",
            ]
        )
    return "\n".join(lines)


def assert_hashes_unchanged(repo_root: Path, lock: PreregistrationLock | None = None) -> None:
    """Raise ``PreregistrationError`` if items/labels diverge from the lock."""
    lock = lock or load_lock(repo_root)
    if lock is None:
        raise PreregistrationError(
            f"No {LOCK_PATH_REL} — run `jevbench preregister <experiment>` "
            "before scored runs."
        )
    task_root = datasets_root(repo_root) / lock.task
    items = task_root / "items.jsonl"
    labels = task_root / "labels.jsonl"
    if not items.is_file() or not labels.is_file():
        raise PreregistrationError(
            f"Locked task files missing under {task_root}"
        )
    current_items = sha256_file(items)
    current_labels = sha256_file(labels)
    problems: list[str] = []
    if current_items != lock.items_sha256:
        problems.append(
            f"items.jsonl hash mismatch\n"
            f"  locked:  {lock.items_sha256}\n"
            f"  current: {current_items}"
        )
    if current_labels != lock.labels_sha256:
        problems.append(
            f"labels.jsonl hash mismatch\n"
            f"  locked:  {lock.labels_sha256}\n"
            f"  current: {current_labels}"
        )
    if problems:
        raise PreregistrationError(
            "Dataset content changed after pre-registration. "
            "Scored runs are refused.\n\n" + "\n\n".join(problems)
        )


def list_scored_runs(repo_root: Path) -> list[Path]:
    """Scored run directories under runs/ (excludes demo/)."""
    runs = repo_root / "runs"
    if not runs.is_dir():
        return []
    out: list[Path] = []
    for child in sorted(runs.iterdir()):
        if not child.is_dir():
            continue
        if child.name == "demo":
            continue
        # A scored run has manifest.json (written by the runner)
        if (child / "manifest.json").is_file() or (child / "raw.jsonl").is_file():
            out.append(child)
    return out


def earliest_scored_run(repo_root: Path) -> Path | None:
    runs = list_scored_runs(repo_root)
    if not runs:
        return None

    def sort_key(p: Path) -> tuple:
        manifest = p / "manifest.json"
        if manifest.is_file():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                ts = data.get("started_at") or data.get("created_at") or ""
                return (0, ts, p.name)
            except (json.JSONDecodeError, OSError):
                pass
        return (1, p.stat().st_mtime, p.name)

    return min(runs, key=sort_key)


def labels_hash_from_run(run_dir: Path) -> str | None:
    manifest = run_dir / "manifest.json"
    if not manifest.is_file():
        return None
    data = json.loads(manifest.read_text(encoding="utf-8"))
    return data.get("labels_sha256") or (data.get("dataset") or {}).get("labels_sha256")


def verify_labels(repo_root: Path) -> None:
    """Fail loudly if labels.jsonl changed after the first scored run.

    This is the contamination guard: adjudicating labels after seeing model
    output must be impossible to do by accident.
    """
    lock = load_lock(repo_root)
    first = earliest_scored_run(repo_root)

    if first is None:
        # No scored runs yet — still enforce lock if present
        if lock is not None:
            assert_hashes_unchanged(repo_root, lock)
        return

    # Determine reference labels hash: prefer first run manifest, else lock
    ref_hash = labels_hash_from_run(first)
    task: str | None = None
    if lock is not None:
        task = lock.task
        if ref_hash is None:
            ref_hash = lock.labels_sha256
    if ref_hash is None:
        raise LabelContaminationError(
            f"First scored run {first.name} has no labels_sha256 in manifest, "
            f"and no {LOCK_PATH_REL}. Cannot verify label integrity."
        )

    if task is None:
        # Try to read task from first manifest
        manifest = first / "manifest.json"
        if manifest.is_file():
            data = json.loads(manifest.read_text(encoding="utf-8"))
            task = data.get("task") or (data.get("dataset") or {}).get("task")
    if task is None:
        raise LabelContaminationError(
            "Cannot locate task name to verify labels.jsonl "
            f"(run={first.name}, lock={'yes' if lock else 'no'})"
        )

    labels_path = datasets_root(repo_root) / task / "labels.jsonl"
    if not labels_path.is_file():
        raise LabelContaminationError(f"labels.jsonl missing: {labels_path}")

    current = sha256_file(labels_path)
    if current != ref_hash:
        raise LabelContaminationError(
            "LABEL CONTAMINATION DETECTED\n\n"
            f"labels.jsonl was modified after the first scored run "
            f"(`runs/{first.name}/`).\n"
            "Adjudicating labels after seeing model output is the exact "
            "contamination this project criticises. Restore labels.jsonl to "
            "the pre-run bytes (or re-preregister and treat prior runs as "
            "exploratory — do not silently continue).\n\n"
            f"  reference (run/lock): {ref_hash}\n"
            f"  current:              {current}\n"
            f"  path:                 {labels_path}"
        )

    # Also enforce full lock (items + labels) when present
    if lock is not None:
        assert_hashes_unchanged(repo_root, lock)


def preregister(
    repo_root: Path,
    experiment: str,
    *,
    force: bool = False,
) -> tuple[Path, Path, PreregistrationLock]:
    """Load experiment YAML, lock dataset hashes, write PREREGISTRATION.md."""
    from jevbench.experiment import resolve_experiment_path

    spec = load_experiment(repo_root, experiment)
    if spec.status == "stub":
        raise PreregistrationError(
            f"Experiment {spec.name!r} is still status=stub. "
            "Fill hypotheses, metrics, sample_size, and task before locking."
        )

    yaml_path = resolve_experiment_path(repo_root, experiment)
    try:
        yaml_rel = str(yaml_path.relative_to(repo_root))
    except ValueError:
        yaml_rel = str(yaml_path)

    existing = load_lock(repo_root)
    if existing is not None and not force:
        # Allow re-preregister only if hashes still match (refresh markdown)
        try:
            assert_hashes_unchanged(repo_root, existing)
        except PreregistrationError:
            raise PreregistrationError(
                f"{LOCK_PATH_REL} already exists and dataset hashes changed. "
                "Refusing to overwrite (would erase the timestamped protocol). "
                "Pass --force only if you intentionally start a new protocol "
                "and will treat prior runs as exploratory."
            ) from None

    if existing is not None and list_scored_runs(repo_root) and not force:
        # Re-emitting markdown is fine if hashes match; changing lock is not
        pass

    lock = build_lock(repo_root, spec, experiment_yaml_rel=yaml_rel)
    if (
        existing is not None
        and list_scored_runs(repo_root)
        and (
            lock.items_sha256 != existing.items_sha256
            or lock.labels_sha256 != existing.labels_sha256
        )
        and not force
    ):
        raise PreregistrationError(
            "Scored runs already exist; cannot change dataset hashes. "
            "Use --force only to deliberately void prior runs."
        )

    md = render_preregistration_md(spec, lock)
    md_path = prereg_md_path(repo_root)
    md_path.write_text(md, encoding="utf-8")
    write_lock(repo_root, lock)
    return md_path, lock_path(repo_root), lock
