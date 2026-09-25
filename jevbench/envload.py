"""Load local environment files without ever printing secret values."""

from __future__ import annotations

import os
from pathlib import Path

KEY_NAME = "TYPESAFE_API_KEY"


def load_dotenv(path: Path) -> None:
    """Set missing/empty variables from a .env file. Values are not logged."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        name = name.strip()
        value = value.strip().strip("'").strip('"')
        if not name:
            continue
        # Overwrite empty placeholders so a blank export cannot block .env.
        if not (os.environ.get(name) or "").strip():
            os.environ[name] = value


def require_typesafe_api_key(dotenv_path: Path | None = None) -> str:
    """Return the Jev API key or raise a message that does not include it."""
    if dotenv_path is not None:
        load_dotenv(dotenv_path)
    key = os.environ.get(KEY_NAME, "").strip()
    if not key:
        raise RuntimeError(
            f"{KEY_NAME} is missing. Copy .env.example to .env and set the key there. "
            "The key value is never printed or written to run files."
        )
    return key
