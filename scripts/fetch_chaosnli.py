#!/usr/bin/env python3
"""Fetch ChaosNLI v1.0 SNLI + MNLI text into a gitignored cache.

The repository ships item ids, annotator counts, and entropy — not the
sentences. ChaosNLI is CC BY-NC 4.0. The sentences are SNLI (CC BY-SA 4.0)
and MNLI (mixed terms). This MIT tree does not redistribute them.

αNLI is not downloaded. It is a 2-way abductive task; see
jevbench.chaosnli.ALPHANLI_EXCLUSION.

The official Dropbox zip was deleted as of 2026-09-23. This script tries
that URL, then pinned mirrors, and refuses any file whose sha256 is not
the v1.0 pin.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jevbench.chaosnli import (  # noqa: E402
    DROPBOX_ZIP,
    MIRROR_MNLI,
    MIRROR_SNLI,
    MNLI_SHA256,
    SNLI_SHA256,
    cache_text_rows,
    load_population,
    verify_download,
)
from jevbench.dataset import write_jsonl  # noqa: E402

CACHE = ROOT / "datasets" / "chaosnli" / "cache"
UA = "jevbench-fetch/0.1 (research; ChaosNLI v1.0 pin)"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    if data[:1] == b"<" or data[:15].lower().startswith(b"<!doctype"):
        raise RuntimeError(f"{url} returned HTML, not the dataset")
    dest.write_bytes(data)


def _from_zip(blob: bytes, name: str) -> bytes | None:
    try:
        zf = zipfile.ZipFile(io.BytesIO(blob))
    except zipfile.BadZipFile:
        return None
    for info in zf.infolist():
        if info.filename.endswith(name) and not info.filename.startswith("__MACOSX"):
            return zf.read(info)
    return None


def _ensure(name: str, sha: str, mirror: str, zip_member: str) -> Path:
    dest = CACHE / name
    if dest.is_file() and _sha256(dest) == sha:
        print(f"cache ok {dest.name}")
        return dest
    CACHE.mkdir(parents=True, exist_ok=True)
    zip_path = CACHE / "chaosNLI_v1.0.zip"
    if not zip_path.is_file():
        try:
            print(f"trying official zip {DROPBOX_ZIP}")
            _download(DROPBOX_ZIP, zip_path)
        except Exception as exc:  # network, HTML interstitial, deleted file
            print(f"official zip unavailable: {exc}")
            zip_path.unlink(missing_ok=True)
    if zip_path.is_file():
        member = _from_zip(zip_path.read_bytes(), zip_member)
        if member is not None:
            dest.write_bytes(member)
    if not dest.is_file() or _sha256(dest) != sha:
        print(f"mirror {mirror}")
        _download(mirror, dest)
    verify_download(dest, sha, sum(1 for line in dest.read_text(encoding="utf-8").splitlines() if line.strip()))
    print(f"pinned {dest.name} {sha[:16]}")
    return dest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--write-cache",
        action="store_true",
        default=True,
        help="Write datasets/chaosnli/cache/text.jsonl (gitignored).",
    )
    args = p.parse_args()
    snli = _ensure("chaosNLI_snli.jsonl", SNLI_SHA256, MIRROR_SNLI, "chaosNLI_snli.jsonl")
    mnli = _ensure("chaosNLI_mnli_m.jsonl", MNLI_SHA256, MIRROR_MNLI, "chaosNLI_mnli_m.jsonl")
    records = load_population(snli, mnli)
    if args.write_cache:
        text_path = CACHE / "text.jsonl"
        write_jsonl(text_path, cache_text_rows(records))
        print(f"wrote {text_path} ({len(records)} items, gitignored)")
    # Touch a marker so a reader sees why the directory exists.
    (CACHE / "README.txt").write_text(
        "Local sentence cache. Not part of the git tree.\n"
        "SNLI text is CC BY-SA 4.0. MNLI text has mixed source terms.\n"
        "ChaosNLI annotations are CC BY-NC 4.0 (Nie, Zhou, Bansal 2020).\n"
        "Do not commit this directory.\n",
        encoding="utf-8",
    )
    print(json.dumps({"n": len(records), "snli": str(snli), "mnli": str(mnli)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
