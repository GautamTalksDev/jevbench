"""Versioned per-provider pricing for dry-run cost estimates.

Rules:
- Every rate has a snapshot date; never silently edit an old snapshot in place.
- Jev output tokens are free — never bill them. Baseline output is NOT free.
- Never apply one provider's rate to another; ``model`` is required for
  non-Jev clients so pricing cannot silently default to the wrong table.
- ``estimate()`` prefers measured pilot token counts over guesses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ClientType = Literal["jev", "adapter", "trivial", "prefill", "gliclass", "bart_mnli"]

# Snapshot → model_id → {input, output} USD / Mtok
# Jev key is always "jev". Adapter keys are provider/model ids.
_SNAPSHOTS: dict[str, dict[str, dict[str, float]]] = {
    "2026-09-19": {
        "jev": {"input": 0.042, "output": 0.0},
        "anthropic/claude-haiku-3.5": {"input": 0.80, "output": 4.00},
        "anthropic/claude-haiku": {"input": 0.80, "output": 4.00},
        "anthropic/claude-haiku-4-5": {"input": 1.00, "output": 5.00},
        "anthropic/claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
        "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "openai/gpt-4o": {"input": 2.50, "output": 10.00},
        "google/gemini-1.5-flash": {"input": 0.075, "output": 0.30},
        "trivial": {"input": 0.0, "output": 0.0},
        "prefill": {"input": 0.0, "output": 0.0},
        "gliclass": {"input": 0.0, "output": 0.0},
        "bart_mnli": {"input": 0.0, "output": 0.0},
    },
}

# Default pilot means (from a notional 20-call structured-output pilot).
# Replace with measured counts before claiming a budget in the README.
DEFAULT_PILOT: dict[str, dict[str, float]] = {
    "jev": {"input_tokens": 180.0, "output_tokens": 24.0},
    # Structured JSON + system prompt overhead — ~500 in / ~80 out is realistic
    "openai/gpt-4o-mini": {"input_tokens": 500.0, "output_tokens": 80.0},
    "anthropic/claude-haiku": {"input_tokens": 500.0, "output_tokens": 80.0},
}


@dataclass(frozen=True)
class Price:
    model_id: str
    snapshot_date: str
    input_usd_per_mtok: float
    output_usd_per_mtok: float

    def cost_usd(self, input_tokens: float, output_tokens: float) -> float:
        return (
            float(input_tokens) / 1_000_000.0 * self.input_usd_per_mtok
            + float(output_tokens) / 1_000_000.0 * self.output_usd_per_mtok
        )


@dataclass(frozen=True)
class CostSplit:
    """Printed budget line — Jev / baseline / total, never collapsed."""

    n_calls_jev: int
    n_calls_baseline: int
    jev_input_tokens: float
    jev_output_tokens: float
    baseline_input_tokens: float
    baseline_output_tokens: float
    jev_usd: float
    baseline_usd: float
    total_usd: float
    jev_model: str
    baseline_model: str
    snapshot_date: str
    pilot_source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_calls_jev": self.n_calls_jev,
            "n_calls_baseline": self.n_calls_baseline,
            "jev_input_tokens": self.jev_input_tokens,
            "jev_output_tokens": self.jev_output_tokens,
            "baseline_input_tokens": self.baseline_input_tokens,
            "baseline_output_tokens": self.baseline_output_tokens,
            "jev_usd": round(self.jev_usd, 4),
            "baseline_usd": round(self.baseline_usd, 4),
            "total_usd": round(self.total_usd, 4),
            "jev_model": self.jev_model,
            "baseline_model": self.baseline_model,
            "snapshot_date": self.snapshot_date,
            "pilot_source": self.pilot_source,
            "split_line": (
                f"Jev ${self.jev_usd:.4f} + baseline({self.baseline_model}) "
                f"${self.baseline_usd:.4f} = ${self.total_usd:.4f}"
            ),
        }


def available_snapshot_dates() -> list[str]:
    return sorted(_SNAPSHOTS)


def get_snapshot(date: str) -> dict[str, dict[str, float]]:
    if date not in _SNAPSHOTS:
        known = ", ".join(available_snapshot_dates()) or "(none)"
        raise KeyError(f"unknown pricing snapshot {date!r}; known: {known}")
    return dict(_SNAPSHOTS[date])


def _normalize_model_id(model: str) -> str:
    return model.strip().lower().replace("_", "-")


def resolve_price(
    snapshot_date: str,
    client_type: ClientType | str,
    model: str | None = None,
) -> Price:
    """Resolve a price row. ``model`` is required for adapter clients."""
    snap = get_snapshot(snapshot_date)
    ct = str(client_type)

    if ct == "jev":
        # Hard guard: Jev rate only for Jev calls
        row = snap["jev"]
        return Price("jev", snapshot_date, row["input"], row["output"])

    if ct == "trivial":
        row = snap["trivial"]
        return Price("trivial", snapshot_date, row["input"], row["output"])

    if ct == "prefill":
        row = snap["prefill"]
        return Price("prefill", snapshot_date, row["input"], row["output"])

    if ct == "gliclass":
        row = snap.get("gliclass") or {"input": 0.0, "output": 0.0}
        return Price("gliclass", snapshot_date, row["input"], row["output"])

    if ct == "bart_mnli":
        row = snap.get("bart_mnli") or {"input": 0.0, "output": 0.0}
        return Price("bart_mnli", snapshot_date, row["input"], row["output"])

    if ct == "adapter":
        if not model or not str(model).strip():
            raise ValueError(
                "resolve_price(adapter) requires an explicit model id "
                "(e.g. 'openai/gpt-4o-mini'). Silent defaults are forbidden."
            )
        mid = _normalize_model_id(model)
        # Exact key
        for key, row in snap.items():
            if key in ("jev", "trivial", "prefill", "gliclass"):
                continue
            if _normalize_model_id(key) == mid or mid.endswith(
                _normalize_model_id(key).split("/", 1)[-1]
            ):
                # Never return the Jev row for an adapter
                assert key != "jev"
                return Price(key, snapshot_date, row["input"], row["output"])
        # Fuzzy family match (still never Jev)
        for key, row in snap.items():
            if key in ("jev", "trivial", "prefill", "gliclass"):
                continue
            kn = _normalize_model_id(key)
            if "haiku" in mid and "haiku" in kn:
                return Price(key, snapshot_date, row["input"], row["output"])
            if "4o-mini" in mid and "4o-mini" in kn:
                return Price(key, snapshot_date, row["input"], row["output"])
            if "flash" in mid and "flash" in kn:
                return Price(key, snapshot_date, row["input"], row["output"])
        raise KeyError(
            f"no price for adapter model={model!r} in snapshot {snapshot_date!r}; "
            f"known: {[k for k in snap if k not in ('jev', 'trivial', 'prefill', 'gliclass')]}"
        )

    raise KeyError(f"no price for client_type={client_type!r}")


def assert_jev_rate_only_on_jev(price: Price, client_type: str) -> None:
    """Fail loudly if the Jev rate table is attached to a non-Jev client."""
    if client_type != "jev" and price.model_id == "jev":
        raise AssertionError(
            "Jev pricing was resolved for a non-Jev client — cost model bug."
        )
    if client_type == "jev" and price.model_id != "jev":
        raise AssertionError(
            f"Jev client resolved to non-Jev price row {price.model_id!r}."
        )


def estimate(
    *,
    snapshot_date: str,
    n_items: int,
    repeats: int,
    baseline_model: str,
    jev_input_tokens: float | None = None,
    jev_output_tokens: float | None = None,
    baseline_input_tokens: float | None = None,
    baseline_output_tokens: float | None = None,
    pilot_source: str = "default_pilot_means",
) -> CostSplit:
    """Estimate spend from pilot (or measured) per-call token counts.

    Pass measured means from a 20-call pilot when available. Defaults come
    from ``DEFAULT_PILOT`` and are labelled as such — not invoice truth.
    """
    if n_items < 1 or repeats < 1:
        raise ValueError("n_items and repeats must be >= 1")

    jev_pilot = DEFAULT_PILOT["jev"]
    # Pick baseline pilot row by family
    b_key = "openai/gpt-4o-mini"
    bn = _normalize_model_id(baseline_model)
    if "haiku" in bn:
        b_key = "anthropic/claude-haiku"
    elif "flash" in bn:
        b_key = "openai/gpt-4o-mini"  # similar class; override when measured
    base_pilot = DEFAULT_PILOT.get(b_key, DEFAULT_PILOT["openai/gpt-4o-mini"])

    ji = float(jev_input_tokens if jev_input_tokens is not None else jev_pilot["input_tokens"])
    jo = float(
        jev_output_tokens if jev_output_tokens is not None else jev_pilot["output_tokens"]
    )
    bi = float(
        baseline_input_tokens
        if baseline_input_tokens is not None
        else base_pilot["input_tokens"]
    )
    bo = float(
        baseline_output_tokens
        if baseline_output_tokens is not None
        else base_pilot["output_tokens"]
    )

    n_calls = n_items * repeats
    jev_price = resolve_price(snapshot_date, "jev")
    base_price = resolve_price(snapshot_date, "adapter", baseline_model)
    assert_jev_rate_only_on_jev(jev_price, "jev")
    assert_jev_rate_only_on_jev(base_price, "adapter")

    # Jev: bill input only (output free by contract)
    jev_usd = n_calls * jev_price.cost_usd(ji, 0.0)
    # Baseline: bill input AND output
    baseline_usd = n_calls * base_price.cost_usd(bi, bo)

    return CostSplit(
        n_calls_jev=n_calls,
        n_calls_baseline=n_calls,
        jev_input_tokens=n_calls * ji,
        jev_output_tokens=n_calls * jo,  # recorded but not billed
        baseline_input_tokens=n_calls * bi,
        baseline_output_tokens=n_calls * bo,
        jev_usd=jev_usd,
        baseline_usd=baseline_usd,
        total_usd=jev_usd + baseline_usd,
        jev_model="jev",
        baseline_model=base_price.model_id,
        snapshot_date=snapshot_date,
        pilot_source=pilot_source,
    )


def snapshot_as_dict(date: str) -> dict[str, Any]:
    snap = get_snapshot(date)
    return {
        "date": date,
        "prices_usd_per_mtok": {
            k: {"input": v["input"], "output": v["output"]} for k, v in snap.items()
        },
        "notes": (
            "Jev output tokens are free and must never be billed. "
            "Baseline output tokens are billed. Adapter requires an explicit "
            "model id — no silent default to Jev rates."
        ),
    }


def estimate_tokens_from_text(text: str) -> int:
    """Rough token estimate (chars/4). Dry-run only — not a billing claim."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)
