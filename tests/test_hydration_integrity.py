"""Local-only: hydrated ChaosNLI text matches the pre-history-rewrite backup."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from jevbench.chaosnli import hydrate_dataset
from jevbench.dataset import load_dataset
from jevbench.request_canon import build_request_body
from jevbench.runner import state_to_text

ROOT = Path(__file__).resolve().parents[1]
BACKUP_ITEMS = Path(
    "/home/gautamtalksdev/projects/jevbench_backup_pre_history_rewrite_20260924T191456Z"
    "/datasets/chaosnli/items.jsonl"
)
BACKUP_CACHE = BACKUP_ITEMS.parent / "cache" / "text.jsonl"
LOCAL_PARAPHRASES = ROOT / "datasets" / "chaosnli" / "local" / "paraphrases.jsonl"
CACHE = ROOT / "datasets" / "chaosnli" / "cache" / "text.jsonl"
SEED = 20260925


def _load_backup_by_id(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            out[str(row["id"])] = row
    return out


def _request_state(state: dict) -> dict:
    """Fields that enter the canonical request body (runner request path)."""
    body = build_request_body(
        item_id="probe",
        client="probe",
        model="probe",
        state=state,
        questions={},
    )
    return body["state"]


def test_paraphrase_hydration_matches_backup_state() -> None:
    if not BACKUP_ITEMS.is_file() or not LOCAL_PARAPHRASES.is_file():
        pytest.skip("backup items.jsonl or local paraphrases.jsonl absent")

    backup = _load_backup_by_id(BACKUP_ITEMS)
    dataset = load_dataset(ROOT / "datasets", "chaosnli")
    # Same call the runner makes before building SystemOneRequest.
    hydrate_dataset(dataset, dataset.root / "cache" / "text.jsonl")

    para = [li for li in dataset.by_id.values() if li.item.role == "paraphrase"]
    assert len(para) == 100

    n_equal = 0
    mismatches: list[str] = []
    for li in para:
        bak = backup[li.id]
        assert isinstance(li.item.state, dict)
        assert isinstance(bak["state"], dict)
        # Content that the request path sends (premise / hypothesis text).
        got = _request_state(li.item.state)
        expected = _request_state(bak["state"])
        # Also compare the full serialized state text the runner estimates tokens on,
        # after aligning the only renamed status field.
        got_full = dict(li.item.state)
        exp_full = dict(bak["state"])
        # text_status is renamed by hydrate; content-preserving means the sentences.
        text_equal = (
            got == expected
            and got_full.get("premise") == exp_full.get("premise")
            and got_full.get("hypothesis") == exp_full.get("hypothesis")
        )
        if text_equal:
            n_equal += 1
        else:
            mismatches.append(
                f"{li.id}: got={got!r} expected={expected!r} "
                f"state_to_text_got={state_to_text(got_full)[:120]!r}"
            )

    print(f"paraphrase hydration n_equal={n_equal}/100")
    assert n_equal == 100, (
        f"paraphrase hydration content mismatch: {n_equal}/100\n"
        + "\n".join(mismatches[:5])
    )


def test_primary_hydration_matches_backup_via_fetch() -> None:
    if not BACKUP_ITEMS.is_file() or not CACHE.is_file() or not BACKUP_CACHE.is_file():
        pytest.skip("backup path or local ChaosNLI cache absent")

    backup_items = _load_backup_by_id(BACKUP_ITEMS)
    backup_text = _load_backup_by_id(BACKUP_CACHE)

    dataset = load_dataset(ROOT / "datasets", "chaosnli")
    hydrate_dataset(dataset, dataset.root / "cache" / "text.jsonl")

    primary_ids = sorted(
        li.id for li in dataset.by_id.values() if li.item.role == "primary"
    )
    rng = random.Random(SEED)
    sample = rng.sample(primary_ids, 20)

    n_equal = 0
    mismatches: list[str] = []
    for item_id in sample:
        assert item_id in backup_items
        li = dataset.by_id[item_id]
        assert isinstance(li.item.state, dict)
        assert li.item.state.get("text_status") == "fetched"
        bak_text = backup_text[item_id]
        got = _request_state(li.item.state)
        expected = {
            "hypothesis": bak_text["hypothesis"],
            "pair_id": backup_items[item_id]["state"].get("pair_id", item_id),
            "premise": bak_text["premise"],
            "source": backup_items[item_id]["state"].get("source"),
        }
        # Drop keys that are None if source missing from expected construction
        expected = {k: v for k, v in expected.items() if v is not None}
        if got == expected or (
            got.get("premise") == bak_text["premise"]
            and got.get("hypothesis") == bak_text["hypothesis"]
        ):
            n_equal += 1
        else:
            mismatches.append(f"{item_id}: got={got!r} expected_text={bak_text!r}")

    print(f"primary hydration n_equal={n_equal}/20")
    assert n_equal == 20, (
        f"primary hydration content mismatch: {n_equal}/20\n"
        + "\n".join(mismatches[:5])
    )
