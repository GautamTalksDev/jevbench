.PHONY: install test reproduce check-repro charts power lint-claims verify-labels check-lock verify-citations clean

install:
	pip install -e ".[dev]"

test:
	python3 -m pytest -q

verify-citations:
	python3 scripts/verify_citations.py

# Offline: regenerate every number and chart from committed JSONL (< 2 min, no network).
reproduce:
	python3 -m jevbench.reproduce

# Fail if any metric or chart diverges from results/SHA256SUMS.
check-repro:
	python3 -m jevbench.reproduce --check

# Offline power analysis — run before any paid call. Prints the budget decision.
power:
	python3 scripts/run_power.py

charts:
	python3 -m jevbench.charts

lint-claims:
	python3 scripts/lint_claims.py paper/main.md

verify-labels:
	python3 -m jevbench.cli verify-labels

check-lock:
	python3 -m jevbench.cli check-lock

clean:
	rm -rf results/harness_fixture/charts results/harness_fixture/metrics.json results/harness_fixture/finding.json results/SHA256SUMS
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
