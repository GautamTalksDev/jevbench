"""Dataset models, JSONL I/O, and tier-stratified sampling."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, field_validator

Tier = Literal["trivial", "easy", "hard", "ambiguous"]
TIERS: tuple[Tier, ...] = ("trivial", "easy", "hard", "ambiguous")


class Item(BaseModel):
    id: str
    state: str | dict[str, Any] | list[Any]
    tier: Tier
    # paraphrase rows are a contamination control and are not in the ΔECE sample
    role: Literal["primary", "paraphrase"] = "primary"

    @field_validator("tier")
    @classmethod
    def _tier_ok(cls, v: str) -> str:
        if v not in TIERS:
            raise ValueError(f"tier must be one of {TIERS}, got {v!r}")
        return v


class Label(BaseModel):
    id: str
    label: str | float | int | bool
    labeler: str
    labeled_at: str  # ISO-8601
    note: str | None = None
    # ChaosNLI: annotator distribution in label-column order. Absent on other tasks.
    label_dist: list[float] | None = None
    label_count: list[int] | None = None

    @field_validator("labeled_at")
    @classmethod
    def _ts_ok(cls, v: str) -> str:
        # Accept Z or offset; reject empty
        if not v or not v.strip():
            raise ValueError("labeled_at must be a non-empty ISO-8601 timestamp")
        return v


@dataclass(frozen=True)
class LabeledItem:
    item: Item
    label: Label

    @property
    def id(self) -> str:
        return self.item.id

    @property
    def tier(self) -> Tier:
        return self.item.tier


@dataclass
class Dataset:
    """One task directory: items + labels joined on id."""

    task: str
    root: Path
    items: list[Item]
    labels: list[Label]
    by_id: dict[str, LabeledItem]

    def __len__(self) -> int:
        return len(self.by_id)

    def tiers(self) -> dict[Tier, list[LabeledItem]]:
        out: dict[Tier, list[LabeledItem]] = {t: [] for t in TIERS}
        for li in self.by_id.values():
            out[li.tier].append(li)
        return out

    def tier_counts(self) -> dict[str, int]:
        return {t: len(xs) for t, xs in self.tiers().items()}


def sha256_file(path: Path) -> str:
    """SHA-256 hex digest of file bytes (exact; any edit changes the hash)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON — {exc}") from exc
            if not isinstance(obj, dict):
                raise TypeError(
                    f"{path}:{lineno}: expected object, got {type(obj).__name__}"
                )
            rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def task_dir(datasets_root: Path, task: str) -> Path:
    return datasets_root / task


def required_task_files(root: Path) -> dict[str, Path]:
    return {
        "items": root / "items.jsonl",
        "labels": root / "labels.jsonl",
        "label_guide": root / "LABEL_GUIDE.md",
        "disputed": root / "DISPUTED.md",
    }


def validate_task_layout(root: Path) -> list[str]:
    """Return list of human-readable problems (empty ⇒ ok)."""
    problems: list[str] = []
    files = required_task_files(root)
    for name, path in files.items():
        if not path.is_file():
            problems.append(f"missing {name}: {path}")
    return problems


def load_dataset(datasets_root: Path, task: str) -> Dataset:
    root = task_dir(datasets_root, task)
    problems = validate_task_layout(root)
    if problems:
        raise FileNotFoundError(
            "task directory incomplete:\n  " + "\n  ".join(problems)
        )

    items = [Item.model_validate(r) for r in read_jsonl(root / "items.jsonl")]
    labels = [Label.model_validate(r) for r in read_jsonl(root / "labels.jsonl")]

    item_ids = {it.id for it in items}
    label_ids = {lb.id for lb in labels}
    if len(item_ids) != len(items):
        raise ValueError(f"{task}: duplicate item ids")
    if len(label_ids) != len(labels):
        raise ValueError(f"{task}: duplicate label ids")

    missing_labels = item_ids - label_ids
    orphan_labels = label_ids - item_ids
    if missing_labels:
        sample = sorted(missing_labels)[:5]
        raise ValueError(
            f"{task}: {len(missing_labels)} items without labels "
            f"(e.g. {sample})"
        )
    if orphan_labels:
        sample = sorted(orphan_labels)[:5]
        raise ValueError(
            f"{task}: {len(orphan_labels)} labels without items "
            f"(e.g. {sample})"
        )

    by_id = {
        it.id: LabeledItem(item=it, label=next(lb for lb in labels if lb.id == it.id))
        for it in items
    }
    # Rebuild labels list aligned to items order for stable hashing consumers
    labels_aligned = [by_id[it.id].label for it in items]
    return Dataset(
        task=task,
        root=root,
        items=items,
        labels=labels_aligned,
        by_id=by_id,
    )


def dataset_hashes(root: Path) -> dict[str, str]:
    files = required_task_files(root)
    return {
        "items_sha256": sha256_file(files["items"]),
        "labels_sha256": sha256_file(files["labels"]),
    }


def stratify_by_tier(
    dataset: Dataset,
    *,
    tiers: Sequence[Tier] | None = None,
) -> dict[Tier, list[LabeledItem]]:
    """Group labeled items by difficulty tier."""
    wanted = set(tiers) if tiers is not None else set(TIERS)
    out: dict[Tier, list[LabeledItem]] = {t: [] for t in TIERS if t in wanted}
    for li in dataset.by_id.values():
        if li.tier in out:
            out[li.tier].append(li)
    for items in out.values():
        items.sort(key=lambda x: x.id)
    return out


def balanced_sample(
    dataset: Dataset,
    *,
    n_per_tier: int,
    seed: int = 0,
    tiers: Sequence[Tier] | None = None,
    replace: bool = False,
) -> list[LabeledItem]:
    """Draw the same number of items from each tier.

    Raises ``ValueError`` if a tier has fewer than ``n_per_tier`` items and
    ``replace`` is False — under-filled hard tiers silently inflate ECE.
    """
    import random

    if n_per_tier < 1:
        raise ValueError("n_per_tier must be >= 1")

    rng = random.Random(seed)
    groups = stratify_by_tier(dataset, tiers=tiers)
    sample: list[LabeledItem] = []
    for tier, items in groups.items():
        if not items:
            raise ValueError(f"tier {tier!r} is empty — cannot balance")
        if len(items) < n_per_tier and not replace:
            raise ValueError(
                f"tier {tier!r} has {len(items)} items; need {n_per_tier}. "
                "Either collect more labels or lower n_per_tier."
            )
        if replace:
            picked = [items[rng.randrange(len(items))] for _ in range(n_per_tier)]
        else:
            pool = list(items)
            rng.shuffle(pool)
            picked = pool[:n_per_tier]
        sample.extend(picked)
    sample.sort(key=lambda x: (TIERS.index(x.tier), x.id))
    return sample


def iter_disputed_ids(disputed_md: Path) -> Iterator[str]:
    """Best-effort: lines like ``- id: foo`` or ``### foo`` in DISPUTED.md."""
    if not disputed_md.is_file():
        return
    text = disputed_md.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- id:"):
            yield stripped.split(":", 1)[1].strip().strip("`")
        elif stripped.startswith("### "):
            yield stripped[4:].strip().strip("`")


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
