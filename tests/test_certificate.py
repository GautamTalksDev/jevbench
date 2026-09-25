"""Certificate export + headless Arena match check."""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from jevbench.certificate import (
    build_specimen,
    export_certificate,
    match_check,
    write_arena_specimen,
)

ROOT = Path(__file__).resolve().parents[1]


def test_specimen_match_and_cw_count() -> None:
    doc = build_specimen()
    check = match_check(doc)
    assert check["ok"], check
    assert check["symbol"] == "✓"
    assert abs(doc["result"]["delta_ece"] - 0.167) < 0.01
    assert doc["meta"]["synthetic"] is True
    assert doc["result"]["delta_corrected"] != doc["result"]["delta_raw"]
    cw = [
        it
        for it in doc["items"]
        if max(it["probs"]) >= 0.75
        and it["human"][int(max(range(3), key=lambda i: it["probs"][i]))] < 0.35
    ]
    assert len(cw) == 85


def test_specimen_delta_corrected_differs_from_raw() -> None:
    doc = build_specimen()
    assert doc["result"]["delta_corrected"] != doc["result"]["delta_raw"]


def test_export_specimen_cli_path(tmp_path: Path) -> None:
    (tmp_path / "results").mkdir()
    path = export_certificate(tmp_path, "specimen-synthetic", synthetic=True)
    assert path.name == "certificate.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert match_check(doc)["ok"]
    assert doc["result"]["delta_corrected"] != doc["result"]["delta_raw"]
    assert match_check(doc)["abs_delta"]["delta_ece"] == 0.0


def test_write_arena_specimen(tmp_path: Path) -> None:
    path = write_arena_specimen(tmp_path)
    assert path.is_file()
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert match_check(doc)["symbol"] == "✓"
    assert doc["meta"]["synthetic"] is True


def test_committed_specimen_synthetic_flag() -> None:
    for rel in (
        "arena/data/specimen.json",
        "results/specimen-synthetic/certificate.json",
    ):
        doc = json.loads((ROOT / rel).read_text(encoding="utf-8"))
        assert doc["meta"]["synthetic"] is True, rel


def test_export_requires_delta_corrected(tmp_path: Path) -> None:
    run_id = "run-missing-corr"
    runs = tmp_path / "runs" / run_id
    runs.mkdir(parents=True)
    (runs / "raw.jsonl").write_text(
        json.dumps(
            {
                "item_id": "x",
                "client": "jev",
                "probs": [0.8, 0.1, 0.1],
                "human": [0.7, 0.2, 0.1],
                "stratum": "easy",
                "choice": "entailment",
                "latency_ms": 10,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "exp1.json").write_text(
        json.dumps(
            {
                "jev": {
                    "primary_delta_ece": {
                        "soft": {
                            "delta_ece": 0.1,
                            "ci_low": 0.0,
                            "ci_high": 0.2,
                            "ece_easy": 0.05,
                            "ece_hard": 0.15,
                        }
                    },
                    "delta_raw": 0.1,
                    "p_value": 0.01,
                    "verdict": "tracks",
                },
                "result": {
                    "delta_raw": 0.1,
                    # delta_corrected intentionally absent
                    "p_value": 0.01,
                    "verdict": "tracks",
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="delta_corrected"):
        export_certificate(tmp_path, run_id)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def arena_server(tmp_path_factory):
    """Serve a temp copy of arena/ so tests never rewrite committed specimen.json."""
    base = tmp_path_factory.mktemp("arena_serve")
    arena = base / "arena"
    shutil.copytree(
        ROOT / "arena",
        arena,
        ignore=shutil.ignore_patterns(".git", "__pycache__"),
    )
    write_arena_specimen(base)
    port = _free_port()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(arena), **kwargs)

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}/"
    for _ in range(50):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.05)
    yield url
    server.shutdown()


def test_page_headless_match_check(arena_server: str) -> None:
    """Load the certificate page headlessly; assert match check shows ✓."""
    script = r"""
const { chromium } = require('playwright');
(async () => {
  const url = process.argv[2];
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  await page.goto(url, { waitUntil: 'networkidle', timeout: 60000 });
  await page.waitForFunction(() => window.__CERTIFICATE_READY__ === true, null, { timeout: 60000 });
  const match = await page.evaluate(() => window.__CERTIFICATE_MATCH__);
  const ok = await page.evaluate(() => window.JevbenchCertificate.matchStatus(window.JevbenchCertificate.getDoc()).ok);
  console.log(JSON.stringify({ match, ok }));
  await browser.close();
  if (match !== '✓' || !ok) process.exit(2);
})().catch((e) => { console.error(e); process.exit(1); });
"""
    _run_playwright(arena_server, script, "headless_match.cjs")


def test_gauge_needle_equals_delta_corrected(arena_server: str) -> None:
    """Prompt N: gauge needle must track result.delta_corrected, not raw ΔECE."""
    script = r"""
const { chromium } = require('playwright');
(async () => {
  const url = process.argv[2];
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  await page.goto(url, { waitUntil: 'networkidle', timeout: 60000 });
  await page.waitForFunction(() => window.__CERTIFICATE_READY__ === true, null, { timeout: 60000 });
  const payload = await page.evaluate(() => {
    const doc = window.JevbenchCertificate.getDoc();
    const needle = window.JevbenchCertificate.gaugeNeedleDelta();
    const corrected = Number(doc.result.delta_corrected != null ? doc.result.delta_corrected : doc.result.delta_ece);
    return {
      needle,
      corrected,
      raw: Number(doc.result.delta_raw != null ? doc.result.delta_raw : doc.result.delta_ece),
      absDiff: Math.abs(needle - corrected),
    };
  });
  console.log(JSON.stringify(payload));
  await browser.close();
  if (!(Number.isFinite(payload.needle) && payload.absDiff <= 1e-9)) process.exit(2);
})().catch((e) => { console.error(e); process.exit(1); });
"""
    out = _run_playwright(arena_server, script, "headless_gauge.cjs")
    payload = json.loads(out.strip().splitlines()[-1])
    assert payload["absDiff"] <= 1e-9
    assert abs(payload["needle"] - payload["corrected"]) <= 1e-9


def _run_playwright(arena_server: str, script: str, filename: str) -> str:
    # Install playwright on demand into a cache dir under the repo (dev-only).
    node_dir = ROOT / ".cache" / "certificate-playwright"
    node_dir.mkdir(parents=True, exist_ok=True)
    pkg = node_dir / "package.json"
    if not pkg.is_file():
        pkg.write_text('{"name":"jevbench-cert-test","private":true}\n', encoding="utf-8")
    if not (node_dir / "node_modules" / "playwright").is_dir():
        subprocess.run(
            ["npm", "install", "--no-save", "playwright@1.48.0"],
            cwd=node_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["npx", "playwright", "install", "chromium"],
            cwd=node_dir,
            check=True,
            capture_output=True,
            text=True,
        )

    script_path = node_dir / filename
    script_path.write_text(script, encoding="utf-8")
    proc = subprocess.run(
        ["node", str(script_path), arena_server],
        cwd=node_dir,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"headless failed (code={proc.returncode})\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
    return proc.stdout
