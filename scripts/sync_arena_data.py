#!/usr/bin/env python3
"""Copy certificate / replay packs into arena/data/.

Never copies into scored run directories. Arena itself cannot write runs/.
Prefer ``jevbench export-certificate`` → ``results/<run_id>/certificate.json``,
then copy that file to ``arena/data/`` when publishing a non-specimen page.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARENA_DATA = ROOT / "arena" / "data"


def main() -> int:
    ARENA_DATA.mkdir(parents=True, exist_ok=True)
    src = ARENA_DATA / "replay.json"
    if not src.is_file():
        print("No arena/data/replay.json yet — generate demo first", file=sys.stderr)
        return 1
    note = {
        "synced_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "source": str(src.relative_to(ROOT)),
        "note": (
            "Arena reads arena/data only. To publish a new replay pack, build "
            "replay.json from runs/*/raw.jsonl + results/ offline, then copy here."
        ),
    }
    (ARENA_DATA / "sync_manifest.json").write_text(json.dumps(note, indent=2) + "\n")
    print(f"OK — Arena data dir: {ARENA_DATA}")
    print("Reminder: Arena cannot write to scored runs/; live DEMO → runs/demo/ only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
