"""Label agreement: Cohen's kappa on a double-labelled subset.

If kappa < 0.6, labels are too noisy to trust a ΔECE claim — tighten
LABEL_GUIDE.md and re-label. Do not proceed and hope.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from jevbench.dataset import read_jsonl


@dataclass(frozen=True)
class AgreementResult:
    n_paired: int
    n_agree: int
    observed_agreement: float
    expected_agreement: float
    kappa: float
    disagreement_rate: float
    labeler_primary: str | None
    labeler_secondary: str | None
    kappa_floor: float
    passes_floor: bool
    paired_ids: tuple[str, ...]
    disagree_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def cohen_kappa(y1: list[str], y2: list[str]) -> tuple[float, float, float]:
    """Return (kappa, observed_agreement, expected_agreement)."""
    if len(y1) != len(y2):
        raise ValueError("y1 and y2 length mismatch")
    n = len(y1)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    agree = sum(a == b for a, b in zip(y1, y2))
    p_o = agree / n
    labels = sorted(set(y1) | set(y2))
    # Expected agreement under independence
    p_e = 0.0
    for lab in labels:
        p1 = sum(1 for a in y1 if a == lab) / n
        p2 = sum(1 for b in y2 if b == lab) / n
        p_e += p1 * p2
    if abs(1.0 - p_e) < 1e-12:
        kappa = 1.0 if p_o >= 1.0 - 1e-12 else 0.0
    else:
        kappa = (p_o - p_e) / (1.0 - p_e)
    return float(kappa), float(p_o), float(p_e)


def select_agreement_subset(
    primary_ids: list[str],
    *,
    fraction: float = 0.15,
    seed: int = 20260922,
) -> list[str]:
    """Deterministic random 15% subset of primary label ids."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    ids = sorted(primary_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    k = max(1, int(round(len(ids) * fraction))) if ids else 0
    return sorted(ids[:k])


def compute_agreement(
    task_dir: Path,
    *,
    kappa_floor: float = 0.6,
    primary_name: str = "labels.jsonl",
    secondary_name: str = "labels_pass2.jsonl",
) -> AgreementResult:
    """Pair primary vs pass-2 labels on shared ids; compute Cohen's κ."""
    primary = read_jsonl(task_dir / primary_name)
    secondary_path = task_dir / secondary_name
    if not secondary_path.is_file():
        return AgreementResult(
            n_paired=0,
            n_agree=0,
            observed_agreement=float("nan"),
            expected_agreement=float("nan"),
            kappa=float("nan"),
            disagreement_rate=float("nan"),
            labeler_primary=None,
            labeler_secondary=None,
            kappa_floor=kappa_floor,
            passes_floor=False,
            paired_ids=(),
            disagree_ids=(),
        )

    secondary = read_jsonl(secondary_path)
    p_by = {str(r["id"]): r for r in primary}
    s_by = {str(r["id"]): r for r in secondary}
    common = sorted(set(p_by) & set(s_by))
    y1 = [str(p_by[i]["label"]) for i in common]
    y2 = [str(s_by[i]["label"]) for i in common]
    kappa, p_o, p_e = cohen_kappa(y1, y2)
    disagree = [i for i, a, b in zip(common, y1, y2) if a != b]
    lab1 = p_by[common[0]].get("labeler") if common else None
    lab2 = s_by[common[0]].get("labeler") if common else None
    # Prefer reporting distinct labeler sets
    labs1 = {str(p_by[i].get("labeler")) for i in common}
    labs2 = {str(s_by[i].get("labeler")) for i in common}
    return AgreementResult(
        n_paired=len(common),
        n_agree=len(common) - len(disagree),
        observed_agreement=p_o,
        expected_agreement=p_e,
        kappa=kappa,
        disagreement_rate=(len(disagree) / len(common)) if common else float("nan"),
        labeler_primary=",".join(sorted(labs1)) if labs1 else lab1,
        labeler_secondary=",".join(sorted(labs2)) if labs2 else lab2,
        kappa_floor=kappa_floor,
        passes_floor=bool(common) and kappa >= kappa_floor,
        paired_ids=tuple(common),
        disagree_ids=tuple(disagree),
    )


def write_agreement_report(result: AgreementResult, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "jevbench.label_agreement.v1",
        **result.to_dict(),
        "interpretation": (
            "PASS — kappa meets floor; proceed to power re-check / scoring"
            if result.passes_floor
            else (
                "FAIL — kappa below floor. Tighten LABEL_GUIDE.md, re-label, "
                "and do not proceed to paid runs."
                if result.n_paired > 0
                else "NO DATA — labels_pass2.jsonl missing or empty overlap."
            )
        ),
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def measured_label_noise(result: AgreementResult) -> float:
    """Map disagreement rate → simulator ``label_noise`` for F1 re-check."""
    if result.n_paired < 1 or result.disagreement_rate != result.disagreement_rate:
        return float("nan")
    return float(result.disagreement_rate)
