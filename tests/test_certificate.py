"""Certificate export + headless Arena match check."""

from __future__ import annotations

import json
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
    cw = [
        it
        for it in doc["items"]
        if max(it["probs"]) >= 0.75
        and it["human"][int(max(range(3), key=lambda i: it["probs"][i]))] < 0.35
    ]
    assert len(cw) == 85


def test_export_specimen_cli_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Use real repo root so arena/data update is optional; write under results/
    path = export_certificate(ROOT, "specimen-synthetic", synthetic=True)
    assert path.name == "certificate.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert match_check(doc)["ok"]
    assert doc["result"]["delta_ece"] == doc["result"]["delta_ece"]  # harness stamp
    # Exporter must stamp recomputed-from-items values for specimen only
    assert match_check(doc)["abs_delta"]["delta_ece"] == 0.0


def test_write_arena_specimen() -> None:
    path = write_arena_specimen(ROOT)
    assert path.is_file()
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert match_check(doc)["symbol"] == "✓"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def arena_server():
    """Serve arena/ so the certificate page can fetch data/specimen.json."""
    write_arena_specimen(ROOT)
    port = _free_port()
    arena = ROOT / "arena"

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(arena), **kwargs)

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}/"
    # Wait until accepting
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

    script_path = node_dir / "headless_match.mjs"
    # CommonJS require — use .cjs
    script_path = node_dir / "headless_match.cjs"
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
            f"headless match failed (code={proc.returncode})\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["match"] == "✓"
    assert payload["ok"] is True
