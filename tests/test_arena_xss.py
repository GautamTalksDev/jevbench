"""Arena certificate page must not execute markup from a loaded JSON file."""

from __future__ import annotations

import json
import shutil
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = "<img src=x onerror=alert(1)>"

pytest.importorskip("playwright.sync_api")


def _certificate() -> dict:
    return {
        "schema": "jevbench.certificate.v1",
        "meta": {
            "run_id": PAYLOAD,
            "model_resolved": PAYLOAD,
            "binning": PAYLOAD,
            "interval_method": PAYLOAD,
            "prereg_commit": PAYLOAD,
            "n_bins": 10,
            "coverage_empirical": 0.5,
            "synthetic": True,
        },
        "result": {
            "delta_ece": 0.1,
            "delta_raw": 0.1,
            "delta_corrected": 0.1,
            "ci_low": 0.0,
            "ci_high": 0.2,
            "ece_easy": 0.01,
            "ece_hard": 0.11,
            "verdict": "tracks",
        },
        "items": [
            {
                "id": PAYLOAD,
                "stratum": PAYLOAD,
                "entropy": 0.2,
                "human": [0.0, 0.1, 0.9],
                "probs": [0.99, 0.005, 0.005],
                "confidence": 0.99,
                "latency_ms": 10,
            }
        ],
    }


def test_arena_rejects_script_in_certificate(tmp_path: Path) -> None:
    from playwright.sync_api import sync_playwright

    site = tmp_path / "arena"
    shutil.copytree(ROOT / "arena", site, ignore=shutil.ignore_patterns("data"))
    data = site / "data"
    data.mkdir()
    (data / "xss.json").write_text(json.dumps(_certificate()), encoding="utf-8")

    handler = partial(SimpleHTTPRequestHandler, directory=str(site))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    dialogs: list[str] = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page()
            page.on("dialog", lambda dialog: dialogs.append(dialog.message))
            page.goto(f"http://127.0.0.1:{port}/index.html?run=data/xss.json")
            page.wait_for_function("window.__CERTIFICATE_READY__ === true")
            body = page.inner_text("body")
            assert PAYLOAD in body
            assert page.locator("img").count() == 0
            browser.close()
    finally:
        server.shutdown()
    assert dialogs == []
