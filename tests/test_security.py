"""Supply-chain and secret-handling checks. No statistical assertions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def test_workflows_only_reference_github_token() -> None:
    offenders: list[str] = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        if "pull_request_target" in text or "workflow_run" in text:
            offenders.append(f"{path.name}: untrusted trigger")
        for line in text.splitlines():
            if "secrets." in line and "secrets.GITHUB_TOKEN" not in line:
                offenders.append(f"{path.name}: {line.strip()}")
    assert offenders == []


def test_preregistration_integrity_script() -> None:
    import runpy

    ns = runpy.run_path(str(ROOT / "scripts" / "verify_prereg_integrity.py"))
    assert ns["main"]() == 0


def test_api_key_missing_error_omits_value(monkeypatch: pytest.MonkeyPatch) -> None:
    from jevbench.envload import require_typesafe_api_key

    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY is missing") as exc:
        require_typesafe_api_key()
    assert "sk-" not in str(exc.value)


def test_api_key_not_written_to_fixture_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from jevbench.runner import assert_run_text_has_no_api_key, write_manifest

    secret = "jevbench-test-key-not-a-real-secret"
    monkeypatch.setenv("TYPESAFE_API_KEY", secret)
    fixture = ROOT / "runs" / "offline_fixture"
    blob = "\n".join(
        p.read_text(encoding="utf-8", errors="replace") for p in fixture.rglob("*") if p.is_file()
    )
    assert secret not in blob

    clean = tmp_path / "manifest.json"
    write_manifest(clean, {"run_id": "fixture", "status": "offline"})
    written = clean.read_text(encoding="utf-8")
    assert secret not in written
    assert_run_text_has_no_api_key(written)

    poisoned = {"run_id": "bad", "note": secret}
    with pytest.raises(RuntimeError, match="never stored"):
        write_manifest(tmp_path / "bad.json", poisoned)


def test_env_example_has_no_real_key() -> None:
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        _name, _, value = stripped.partition("=")
        assert value.strip().strip("'\"") == ""


def test_workflows_parse() -> None:
    for path in WORKFLOWS.glob("*.yml"):
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert doc["permissions"]["contents"] == "read"
        dumped = json.dumps(doc)
        assert "pull_request_target" not in dumped
