#!/usr/bin/env python3
"""Minimal local CLI labeller for jevbench tasks.

Contamination guard
-------------------
This tool is structurally incapable of showing model predictions:

- It never imports ``jevbench.clients``, ``jevbench.runner``, or anything
  under ``runs/``.
- It never accepts a ``--run``, ``--predictions``, or similar flag.
- The only text shown is the item ``state`` from the queue / items file.

Append-only writes: primary labels go to ``labels.jsonl``; a second labeler
pass goes to ``labels_pass2.jsonl`` (never overwrites primary).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Allowed imports: stdlib + local path helpers only. Do NOT import clients /
# runner / metrics that could load predictions.
REPO = Path(__file__).resolve().parents[1]
ALLOWED_LABELS = ("billing", "technical", "other")
ALLOWED_TIERS = ("trivial", "easy", "hard", "ambiguous")

# Hard ban: if these strings appear in argv as flags, refuse.
_FORBIDDEN_FLAGS = (
    "--run",
    "--run-dir",
    "--predictions",
    "--raw",
    "--model-output",
    "--probs",
    "--confidence",
    "--manifest",
)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{lineno}: expected object")
            rows.append(obj)
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(r["id"]) for r in rows if "id" in r}


def _state_text(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, indent=2)


def _prompt(msg: str) -> str:
    sys.stdout.write(msg)
    sys.stdout.flush()
    return sys.stdin.readline().strip()


def _assert_no_forbidden_argv(argv: list[str]) -> None:
    for a in argv:
        key = a.split("=", 1)[0]
        if key in _FORBIDDEN_FLAGS:
            raise SystemExit(
                f"Refused: {key} is forbidden. This labeller cannot load "
                "model output — that is the contamination guard."
            )


def next_queue_item(
    *,
    queue: list[dict[str, Any]],
    labeled_ids: set[str],
    pass2_ids: set[str],
    pass_id: int,
) -> dict[str, Any] | None:
    for row in queue:
        iid = str(row["id"])
        if pass_id == 1 and iid not in labeled_ids:
            return row
        if pass_id == 2 and iid in labeled_ids and iid not in pass2_ids:
            return row
    return None


def label_one(
    *,
    task_dir: Path,
    labeler: str,
    pass_id: int,
    item: dict[str, Any],
    note: str | None,
) -> None:
    iid = str(item["id"])
    state = item.get("state", "")
    print()
    print("=" * 72)
    print(f"id: {iid}")
    print("-" * 72)
    print(_state_text(state))
    print("=" * 72)
    print(f"labels: {', '.join(ALLOWED_LABELS)}")
    print(f"tiers:  {', '.join(ALLOWED_TIERS)}")
    print("(Read datasets/.../LABEL_GUIDE.md before assigning tier.)")
    print()

    while True:
        lab = _prompt("department label> ").lower()
        if lab in ALLOWED_LABELS:
            break
        print(f"  invalid; choose one of {ALLOWED_LABELS}")

    while True:
        tier = _prompt("tier> ").lower()
        if tier in ALLOWED_TIERS:
            break
        print(f"  invalid; choose one of {ALLOWED_TIERS}")

    if note is None:
        note_in = _prompt("note (optional, Enter to skip)> ")
        note = note_in if note_in else None

    ts = _utc_now()
    label_row: dict[str, Any] = {
        "id": iid,
        "label": lab,
        "labeler": labeler,
        "labeled_at": ts,
    }
    if note:
        label_row["note"] = note

    if pass_id == 1:
        # Append label; upsert item with tier (item may already exist without us)
        items_path = task_dir / "items.jsonl"
        labels_path = task_dir / "labels.jsonl"
        existing_items = _read_jsonl(items_path)
        by_id = {str(r["id"]): r for r in existing_items}
        by_id[iid] = {"id": iid, "state": state, "tier": tier}
        # Rewrite items in stable id order (tier is an item property)
        ordered = [by_id[k] for k in sorted(by_id)]
        with items_path.open("w", encoding="utf-8") as f:
            for row in ordered:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        _append_jsonl(labels_path, label_row)
        # Remove from queue if present
        queue_path = task_dir / "queue.jsonl"
        if queue_path.is_file():
            remaining = [r for r in _read_jsonl(queue_path) if str(r["id"]) != iid]
            with queue_path.open("w", encoding="utf-8") as f:
                for row in remaining:
                    f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        print(f"wrote primary label → {labels_path.name}; tier on items.jsonl")
    else:
        # Pass 2: labels only (tier already fixed on the item — do not change)
        pass2 = task_dir / "labels_pass2.jsonl"
        # Still record the second labeler's tier judgment for agreement diagnostics
        label_row["tier_judgment"] = tier
        _append_jsonl(pass2, label_row)
        print(f"wrote pass-2 label → {pass2.name} (items.jsonl tier unchanged)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--task",
        default="support_tickets",
        help="Task name under datasets/ (default: support_tickets)",
    )
    p.add_argument(
        "--labeler",
        required=True,
        help="Labeler id (required). Pass-2 must differ from the primary labeler.",
    )
    p.add_argument(
        "--pass",
        dest="pass_id",
        type=int,
        choices=(1, 2),
        default=1,
        help="1 = primary labels.jsonl; 2 = labels_pass2.jsonl agreement pass",
    )
    p.add_argument(
        "--id",
        dest="item_id",
        default=None,
        help="Label a specific item id (from queue or items)",
    )
    p.add_argument(
        "--note",
        default=None,
        help="Optional note (otherwise prompted)",
    )
    p.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of items to label in this session (default 1)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    _assert_no_forbidden_argv(argv)
    args = build_parser().parse_args(argv)

    task_dir = REPO / "datasets" / args.task
    if not task_dir.is_dir():
        print(f"task directory missing: {task_dir}", file=sys.stderr)
        return 2
    guide = task_dir / "LABEL_GUIDE.md"
    if not guide.is_file():
        print(f"missing LABEL_GUIDE.md at {guide}", file=sys.stderr)
        return 2

    queue_path = task_dir / "queue.jsonl"
    items_path = task_dir / "items.jsonl"
    labels_path = task_dir / "labels.jsonl"
    pass2_path = task_dir / "labels_pass2.jsonl"

    queue = _read_jsonl(queue_path)
    items = _read_jsonl(items_path)
    labels = _read_jsonl(labels_path)
    pass2 = _read_jsonl(pass2_path)

    labeled_ids = _ids(labels)
    pass2_ids = _ids(pass2)

    # Candidates for pass 1: queue rows, else items missing labels
    if not queue:
        queue = [r for r in items if str(r["id"]) not in labeled_ids]
    # Pass 2 candidates: already labelled items not yet in pass2
    if args.pass_id == 2:
        primary_by_id = {str(r["id"]): r for r in labels}
        item_by_id = {str(r["id"]): r for r in items}
        queue = []
        for iid in sorted(primary_by_id):
            if iid in pass2_ids:
                continue
            # Second labeler must differ
            if primary_by_id[iid].get("labeler") == args.labeler:
                continue
            if iid in item_by_id:
                queue.append(item_by_id[iid])

    n_done = 0
    while n_done < args.count:
        if args.item_id:
            pool = _read_jsonl(queue_path) + items
            match = next((r for r in pool if str(r["id"]) == args.item_id), None)
            if match is None:
                print(f"id not found: {args.item_id}", file=sys.stderr)
                return 3
            item = match
            args.item_id = None  # only once
        else:
            item = next_queue_item(
                queue=queue,
                labeled_ids=labeled_ids,
                pass2_ids=pass2_ids,
                pass_id=args.pass_id,
            )
            if item is None:
                print("queue empty for this pass.")
                break
        label_one(
            task_dir=task_dir,
            labeler=args.labeler,
            pass_id=args.pass_id,
            item=item,
            note=args.note,
        )
        iid = str(item["id"])
        labeled_ids.add(iid)
        if args.pass_id == 2:
            pass2_ids.add(iid)
        queue = [r for r in queue if str(r["id"]) != iid]
        n_done += 1

    print(f"session complete: labelled {n_done} item(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
