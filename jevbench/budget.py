"""Hard spend guard for paid API calls (Amendment 10 / PROMPT S).

Ledger: ``budget.json`` at the repo root (gitignored runtime state may live
under ``runs/``; the schema file ``budget.schema.json`` is committed).

Rules
-----
- Every billed call appends ``input_tokens`` and computed USD cost.
- Hard caps: pilot $0.05 / run, EXP-1 $1.00 / run, global $2.50.
- 50% of the $5 credit ($2.50) is reserved and never allocated to the global
  spendable cap — so a re-run after a bug is always possible.
- Runner refuses to START if projected cost exceeds remaining room, and STOPS
  mid-run on hitting the cap (resumable checkpoint via raw.jsonl).
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from jevbench.cost import resolve_price

RunKind = Literal["pilot", "exp1", "other"]

# Spendable half of a $5 credit. The other $2.50 is untouchable reserve.
GLOBAL_CAP_USD = 2.50
RUN_CAPS_USD: dict[RunKind, float] = {
    "pilot": 0.05,
    "exp1": 1.00,
    "other": 0.25,
}

LEDGER_NAME = "budget.json"


class BudgetExceeded(RuntimeError):
    """Raised when a run would breach a hard spend cap."""


@dataclass
class LedgerEntry:
    utc: str
    run_id: str
    run_kind: RunKind
    client: str
    model: str
    item_id: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    note: str = ""


@dataclass
class BudgetLedger:
    """Append-only spend ledger with hard caps."""

    path: Path
    global_cap_usd: float = GLOBAL_CAP_USD
    run_caps_usd: dict[str, float] = field(
        default_factory=lambda: dict(RUN_CAPS_USD)
    )
    entries: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def load(cls, repo_root: Path) -> BudgetLedger:
        path = repo_root / LEDGER_NAME
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                path=path,
                global_cap_usd=float(data.get("global_cap_usd", GLOBAL_CAP_USD)),
                run_caps_usd=dict(data.get("run_caps_usd") or RUN_CAPS_USD),
                entries=list(data.get("entries") or []),
            )
        return cls(path=path)

    def to_dict(self) -> dict[str, Any]:
        spent = self.spent_usd()
        return {
            "schema": "jevbench.budget.v1",
            "global_cap_usd": self.global_cap_usd,
            "reserve_usd": 2.50,  # untouchable half of $5
            "run_caps_usd": dict(self.run_caps_usd),
            "spent_usd": round(spent, 6),
            "remaining_usd": round(max(0.0, self.global_cap_usd - spent), 6),
            "entries": list(self.entries),
            "updated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        }

    def save(self) -> None:
        with self._lock:
            self.path.write_text(
                json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    def spent_usd(self) -> float:
        return float(sum(float(e.get("cost_usd") or 0.0) for e in self.entries))

    def spent_for_run(self, run_id: str) -> float:
        return float(
            sum(
                float(e.get("cost_usd") or 0.0)
                for e in self.entries
                if e.get("run_id") == run_id
            )
        )

    def remaining_global(self) -> float:
        return max(0.0, self.global_cap_usd - self.spent_usd())

    def run_cap(self, kind: RunKind) -> float:
        return float(self.run_caps_usd.get(kind, RUN_CAPS_USD.get(kind, 0.25)))

    def assert_can_start(
        self, *, kind: RunKind, projected_usd: float, run_id: str
    ) -> None:
        """Refuse to start if projected cost exceeds remaining room."""
        run_cap = self.run_cap(kind)
        remaining = self.remaining_global()
        if projected_usd > run_cap + 1e-9:
            raise BudgetExceeded(
                f"Projected ${projected_usd:.4f} exceeds {kind} run cap "
                f"${run_cap:.2f}. Reduce items/repeats or raise the cap only "
                f"via Amendment."
            )
        if projected_usd > remaining + 1e-9:
            raise BudgetExceeded(
                f"Projected ${projected_usd:.4f} exceeds remaining global "
                f"budget ${remaining:.4f} (cap ${self.global_cap_usd:.2f}; "
                f"spent ${self.spent_usd():.4f}). 50% reserve is untouchable."
            )

    def record(
        self,
        *,
        run_id: str,
        run_kind: RunKind,
        client: str,
        model: str,
        item_id: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        note: str = "",
    ) -> None:
        entry = LedgerEntry(
            utc=datetime.now(UTC).replace(microsecond=0).isoformat(),
            run_id=run_id,
            run_kind=run_kind,
            client=client,
            model=model,
            item_id=item_id,
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            cost_usd=round(float(cost_usd), 8),
            note=note,
        )
        with self._lock:
            self.entries.append(asdict(entry))
            # Persist after every call so a crash keeps the ledger honest.
            self.path.write_text(
                json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    def check_after_call(self, *, run_id: str, kind: RunKind) -> None:
        """Stop mid-run if a hard cap is hit."""
        run_spent = self.spent_for_run(run_id)
        run_cap = self.run_cap(kind)
        if run_spent > run_cap + 1e-9:
            raise BudgetExceeded(
                f"Run {run_id} hit {kind} cap: spent ${run_spent:.4f} > "
                f"${run_cap:.2f}. Checkpoint is resumable via raw.jsonl."
            )
        if self.spent_usd() > self.global_cap_usd + 1e-9:
            raise BudgetExceeded(
                f"Global budget exhausted: spent ${self.spent_usd():.4f} > "
                f"cap ${self.global_cap_usd:.2f}."
            )


def classify_run_kind(experiment_name: str) -> RunKind:
    name = experiment_name.lower()
    if "pilot" in name:
        return "pilot"
    if "exp1" in name or "difficulty" in name:
        return "exp1"
    return "other"


def cost_for_call(
    *,
    snapshot_date: str,
    client_type: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """USD for one call. Local clients (prefill/gliclass/trivial) are $0."""
    if client_type in ("prefill", "gliclass", "trivial", "bart_mnli", "bart_mnli_nli"):
        return 0.0
    price = resolve_price(snapshot_date, client_type, model)
    if client_type == "jev":
        return price.cost_usd(input_tokens, 0)
    return price.cost_usd(input_tokens, output_tokens)
