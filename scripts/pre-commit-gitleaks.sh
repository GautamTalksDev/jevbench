#!/usr/bin/env bash
# Pre-commit secret scan (Amendment 10). Requires gitleaks on PATH.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
if ! command -v gitleaks >/dev/null 2>&1; then
  echo "gitleaks not installed — install from https://github.com/gitleaks/gitleaks" >&2
  echo "or:  go install github.com/gitleaks/gitleaks/v8@latest" >&2
  exit 1
fi
gitleaks protect --staged --config gitleaks.toml --verbose
