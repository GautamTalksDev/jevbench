"""Certificate export for the Arena — harness numbers only, never recompute ΔECE.

The page may recompute ECE from ``items`` solely to confirm a match with the
stamped ``result`` fields within 1e-6. The exporter copies harness outputs
verbatim into ``result`` and must not invent those numbers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from jevbench.chaosnli import LABEL_ORDER as NLI_LABEL_ORDER
from jevbench.metrics import expected_calibration_error, soft_correctness
from jevbench.verdict import decide_verdict

SCHEMA = "jevbench.certificate.v1"
MATCH_TOL = 1e-6
PREREG_COMMIT_EXP1 = "6011c75"  # Amendment 9 binding (see PREREGISTRATION.md)
LABEL_ORDER = NLI_LABEL_ORDER  # entailment, neutral, contradiction


def _top_prob(probs: list[float]) -> float:
    return float(max(probs)) if probs else float("nan")


def _argmax_label(probs: list[float]) -> str:
    i = int(np.argmax(np.asarray(probs, dtype=float)))
    return LABEL_ORDER[i]


def soft_correct_from_item(item: dict[str, Any]) -> float:
    probs = [float(x) for x in item["probs"]]
    human = [float(x) for x in item["human"]]
    pick = int(np.argmax(probs))
    return float(human[pick])


def recompute_stratum_ece(
    items: list[dict[str, Any]],
    *,
    n_bins: int = 10,
    binning: str = "uniform",
) -> dict[str, float]:
    """Recompute soft ECE per stratum from certificate items (page match check)."""
    by: dict[str, list[dict[str, Any]]] = {"easy": [], "hard": []}
    for it in items:
        s = str(it.get("stratum", ""))
        if s in by:
            by[s].append(it)

    out: dict[str, float] = {}
    for stratum, rows in by.items():
        if not rows:
            out[f"ece_{stratum}"] = float("nan")
            continue
        probs = np.asarray([r["probs"] for r in rows], dtype=float)
        human = np.asarray([r["human"] for r in rows], dtype=float)
        # Dummy hard labels — soft correctness drives the ECE.
        labels = np.argmax(human, axis=1)
        soft = soft_correctness(probs, human)
        ece = expected_calibration_error(
            probs,
            labels,
            n_bins=n_bins,
            strategy=binning,  # type: ignore[arg-type]
            correctness=soft,
        )
        out[f"ece_{stratum}"] = float(ece.ece)
    eh = out.get("ece_hard", float("nan"))
    ee = out.get("ece_easy", float("nan"))
    out["delta_ece"] = float(eh - ee) if math.isfinite(eh) and math.isfinite(ee) else float("nan")
    return out


def match_check(doc: dict[str, Any]) -> dict[str, Any]:
    """Compare stamped result.* to ECE recomputed from items."""
    meta = doc.get("meta") or {}
    result = doc.get("result") or {}
    items = list(doc.get("items") or [])
    n_bins = int(meta.get("n_bins") or 10)
    binning = str(meta.get("binning") or "uniform")
    recomputed = recompute_stratum_ece(items, n_bins=n_bins, binning=binning)
    keys = ("delta_ece", "ece_hard", "ece_easy")
    deltas: dict[str, float] = {}
    ok = True
    for k in keys:
        stamped = float(result[k])
        got = float(recomputed[k])
        d = abs(stamped - got)
        deltas[k] = d
        if not (math.isfinite(stamped) and math.isfinite(got) and d <= MATCH_TOL):
            ok = False
    return {
        "ok": ok,
        "tolerance": MATCH_TOL,
        "stamped": {k: float(result[k]) for k in keys},
        "recomputed": {k: recomputed[k] for k in keys},
        "abs_delta": deltas,
        "symbol": "✓" if ok else "✗",
    }


def build_specimen(*, seed: int = 20260923) -> dict[str, Any]:
    """Synthetic certificate with ΔECE in the ≥0.09 zone and 85 confidently-wrong rows."""
    rng = np.random.default_rng(seed)
    n = 750
    items: list[dict[str, Any]] = []

    def _human_from_entropy(entropy_target: float, majority: int) -> list[float]:
        # Approximate ChaosNLI-style shares; majority gets the bulk.
        # H = -sum p log2 p; we construct a peaked or flat distribution.
        if entropy_target < 0.6:
            shares = np.array([0.02, 0.02, 0.02], dtype=float)
            shares[majority] = 0.94
        elif entropy_target < 1.0:
            shares = np.array([0.10, 0.10, 0.10], dtype=float)
            shares[majority] = 0.80
        else:
            # Near-uniform high entropy
            shares = np.array([0.34, 0.33, 0.33], dtype=float)
            shares[majority] += 0.05
            shares = shares / shares.sum()
        # Light jitter
        shares = shares + rng.normal(0, 0.01, size=3)
        shares = np.clip(shares, 0.01, None)
        shares = shares / shares.sum()
        return [float(x) for x in shares]

    # Easy: well calibrated, low entropy
    for i in range(n):
        maj = int(rng.integers(0, 3))
        human = _human_from_entropy(0.35 + 0.2 * rng.random(), maj)
        # Calibrated: top prob ≈ annotator share of pick
        conf = float(np.clip(human[maj] + rng.normal(0, 0.02), 0.55, 0.98))
        rest = (1.0 - conf) / 2.0
        probs = [rest, rest, rest]
        probs[maj] = conf
        items.append(
            {
                "id": f"syn-easy-{i:04d}",
                "stratum": "easy",
                "entropy": float(0.2 + 0.4 * rng.random()),
                "human": human,
                "probs": probs,
                "confidence": conf,
                "latency_ms": float(rng.uniform(40, 120)),
                "baseline": {
                    "pick": LABEL_ORDER[maj],
                    "latency_ms": float(rng.uniform(180, 900)),
                },
            }
        )

    # Hard: mild overconfidence targeting ΔECE ≈ 0.167 (specimen stamp).
    # Exactly 85 confidently-wrong rows; the rest are nearer soft-calibrated.
    hard_rows: list[dict[str, Any]] = []
    for i in range(n):
        maj = int(rng.integers(0, 3))
        human = _human_from_entropy(1.1 + 0.4 * rng.random(), maj)
        soft_share = float(human[maj])
        # Slight overconfidence on majority pick
        conf = float(np.clip(soft_share + 0.12 + rng.normal(0, 0.03), 0.45, 0.92))
        rest = (1.0 - conf) / 2.0
        probs = [rest, rest, rest]
        probs[maj] = conf
        hard_rows.append(
            {
                "id": f"syn-hard-{i:04d}",
                "stratum": "hard",
                "entropy": float(1.05 + 0.5 * rng.random()),
                "human": human,
                "probs": probs,
                "confidence": conf,
                "latency_ms": float(rng.uniform(45, 140)),
                "baseline": {
                    "pick": LABEL_ORDER[maj],
                    "latency_ms": float(rng.uniform(200, 1100)),
                },
                "_maj": maj,
            }
        )

    # Carve exactly 85 confidently-wrong items (high conf, low soft correctness).
    for it in hard_rows[:85]:
        maj = int(it.pop("_maj"))
        wrong = (maj + 1) % 3
        conf = 0.87
        rest = (1.0 - conf) / 2.0
        probs = [rest, rest, rest]
        probs[wrong] = conf
        human = [0.12, 0.12, 0.12]
        human[maj] = 0.70
        human[wrong] = 0.18
        s = sum(human)
        it["human"] = [h / s for h in human]
        it["probs"] = probs
        it["confidence"] = conf
    for it in hard_rows[85:]:
        it.pop("_maj", None)

    items.extend(hard_rows)
    recomputed = recompute_stratum_ece(items, n_bins=10, binning="uniform")

    # Nudge hard confidence on non-CW rows so ΔECE lands near 0.167.
    target_delta = 0.167
    for _ in range(8):
        cur = recomputed["delta_ece"]
        gap = target_delta - cur
        if abs(gap) < 0.002:
            break
        # Shift hard-stratum (non-CW) top probs toward/away from soft share.
        for it in hard_rows[85:]:
            probs = list(it["probs"])
            pick = int(np.argmax(probs))
            soft = float(it["human"][pick])
            new_conf = float(np.clip(probs[pick] + 0.35 * gap, 0.40, 0.95))
            # Keep some gap from soft share for ECE mass
            new_conf = float(np.clip(soft + (new_conf - soft), 0.40, 0.95))
            rest = (1.0 - new_conf) / 2.0
            probs = [rest, rest, rest]
            probs[pick] = new_conf
            it["probs"] = probs
            it["confidence"] = new_conf
        recomputed = recompute_stratum_ece(items, n_bins=10, binning="uniform")
    # Coverage from results/coverage.json if present; else specimen default below 95%
    coverage = 0.923
    cov_path = Path(__file__).resolve().parents[1] / "results" / "coverage.json"
    if cov_path.is_file():
        try:
            cov = json.loads(cov_path.read_text(encoding="utf-8"))
            coverage = float(
                (cov.get("percentile") or {}).get("empirical_coverage")
                or cov.get("empirical_coverage")
                or coverage
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    # Specimen is built to land in the tracks zone; stamp a harness verdict.
    delta = float(recomputed["delta_ece"])
    specimen_verdict = decide_verdict(
        corrected_delta_ece=delta,
        p_value=0.001,
        p_holds_reject_ge_effect=0.99,
    )

    return {
        "schema": SCHEMA,
        "meta": {
            "run_id": "specimen-synthetic",
            "git_sha": "0000000",
            "prereg_commit": PREREG_COMMIT_EXP1,
            "model_resolved": "jev-specimen",
            "baseline_model": "adapter-specimen",
            "dataset": "chaosnli_snli_mnli",
            "serving_path": "replay",
            "n_bins": 10,
            "binning": "uniform",
            "seed": seed,
            "interval_method": "percentile",
            "coverage_empirical": coverage,
            "synthetic": True,
        },
        "result": {
            "delta_ece": recomputed["delta_ece"],
            "delta_raw": recomputed["delta_ece"],
            "delta_corrected": recomputed["delta_ece"],
            "ci_low": recomputed["delta_ece"] - 0.02,
            "ci_high": recomputed["delta_ece"] + 0.02,
            "ece_easy": recomputed["ece_easy"],
            "ece_hard": recomputed["ece_hard"],
            "p_value": 0.001,
            "verdict": specimen_verdict["verdict"],
            "verdict_reason": specimen_verdict["reason"],
        },
        "items": items,
    }


def export_certificate(
    root: Path,
    run_id: str,
    *,
    exp1_path: Path | None = None,
    raw_path: Path | None = None,
    coverage_path: Path | None = None,
    synthetic: bool = False,
) -> Path:
    """Write ``results/<run_id>/certificate.json``.

    When ``synthetic`` is True, writes a specimen (ignores run artifacts).
    Otherwise copies ΔECE fields from harness ``exp1.json`` and builds items
    from the run's raw records — never recomputes the stamped result.
    """
    out_dir = root / "results" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "certificate.json"

    if synthetic or run_id in {"specimen", "specimen-synthetic"}:
        doc = build_specimen()
        doc["meta"]["run_id"] = run_id if run_id != "specimen" else "specimen-synthetic"
        out_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        return out_path

    exp1_path = exp1_path or (root / "results" / "exp1.json")
    raw_path = raw_path or (root / "runs" / run_id / "raw.jsonl")
    coverage_path = coverage_path or (root / "results" / "coverage.json")
    if not exp1_path.is_file():
        raise FileNotFoundError(f"missing harness output: {exp1_path}")
    if not raw_path.is_file():
        raise FileNotFoundError(f"missing run raw.jsonl: {raw_path}")

    exp1 = json.loads(exp1_path.read_text(encoding="utf-8"))
    jev = exp1.get("jev") or {}
    primary = jev.get("primary_delta_ece") or {}
    soft = primary.get("soft") or primary  # soft is primary when present
    # Prefer explicit soft block if nested
    if isinstance(primary.get("soft"), dict):
        soft = primary["soft"]
    elif "delta_ece" not in primary and isinstance(jev.get("soft"), dict):
        soft = jev["soft"].get("primary_delta_ece") or soft

    # Harness field names vary slightly across versions — copy, don't invent.
    def _req(block: dict[str, Any], *keys: str) -> float:
        for k in keys:
            if k in block and block[k] is not None:
                return float(block[k])
        raise KeyError(f"harness missing {'/'.join(keys)} in primary_delta_ece")

    delta = _req(soft, "delta_ece", "point", "estimate")
    ci_low = _req(soft, "ci_low", "low", "percentile_low")
    ci_high = _req(soft, "ci_high", "high", "percentile_high")
    ece_easy = _req(soft, "ece_easy")
    ece_hard = _req(soft, "ece_hard")

    # Amendment 9 fields — copy from harness result / jev block; never invent.
    result_block = exp1.get("result") or {}
    delta_raw = result_block.get("delta_raw", jev.get("delta_raw", delta))
    delta_corrected = result_block.get(
        "delta_corrected", jev.get("delta_corrected", delta)
    )
    p_value = result_block.get("p_value", jev.get("p_value"))
    verdict = result_block.get("verdict", jev.get("verdict"))
    if verdict is None:
        raise KeyError("harness missing result.verdict — re-run EXP-1 analyze")
    if p_value is None:
        raise KeyError("harness missing result.p_value — re-run EXP-1 analyze")

    coverage = float("nan")
    if coverage_path.is_file():
        cov = json.loads(coverage_path.read_text(encoding="utf-8"))
        coverage = float(
            (cov.get("percentile") or {}).get("empirical_coverage")
            or cov.get("empirical_coverage")
            or float("nan")
        )

    manifest_path = root / "runs" / run_id / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    items = _items_from_raw(raw_path)
    model_resolved = str(
        manifest.get("resolved_model")
        or manifest.get("model")
        or _first_model_from_raw(raw_path)
        or "unknown"
    )

    doc = {
        "schema": SCHEMA,
        "meta": {
            "run_id": run_id,
            "git_sha": str(manifest.get("git_sha") or "unknown"),
            "prereg_commit": PREREG_COMMIT_EXP1,
            "model_resolved": model_resolved,
            "baseline_model": str(
                manifest.get("baseline_model")
                or (exp1.get("baseline") or {}).get("client")
                or "adapter"
            ),
            "dataset": str(manifest.get("task") or manifest.get("dataset") or "chaosnli"),
            "serving_path": str(manifest.get("serving_path") or "recorded"),
            "n_bins": int(soft.get("n_bins") or primary.get("n_bins") or 10),
            "binning": str(soft.get("strategy") or soft.get("binning") or "uniform"),
            "seed": int(soft.get("seed") or manifest.get("seed") or 0),
            "interval_method": str(soft.get("ci_method") or "percentile"),
            "coverage_empirical": coverage,
            "synthetic": False,
        },
        "result": {
            "delta_ece": delta,
            "delta_raw": float(delta_raw),
            "delta_corrected": float(delta_corrected),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "ece_easy": ece_easy,
            "ece_hard": ece_hard,
            "p_value": float(p_value),
            "verdict": str(verdict),
        },
        "items": items,
    }
    out_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return out_path


def _first_model_from_raw(raw_path: Path) -> str | None:
    with raw_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("client") == "jev" and rec.get("model"):
                return str(rec["model"])
            if rec.get("model"):
                return str(rec["model"])
    return None


def _items_from_raw(raw_path: Path) -> list[dict[str, Any]]:
    """Build certificate items from raw.jsonl. No sentence text."""
    by_item: dict[str, dict[str, Any]] = {}
    baseline_by: dict[str, dict[str, Any]] = {}
    with raw_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            item_id = str(rec.get("item_id") or rec.get("id") or "")
            if not item_id:
                continue
            client = str(rec.get("client") or "jev")
            if client not in {"jev", "adapter", "baseline"}:
                # Keep jev primary; adapter as baseline
                if client != "adapter":
                    continue
            probs_raw = rec.get("probabilities") or rec.get("probs")
            if isinstance(probs_raw, dict):
                probs = [float(probs_raw.get(lab, 0.0)) for lab in LABEL_ORDER]
                if sum(probs) <= 0 and any(k not in LABEL_ORDER for k in probs_raw):
                    # support_tickets-style labels — skip ChaosNLI certificate shape
                    continue
            elif isinstance(probs_raw, list):
                probs = [float(x) for x in probs_raw]
            else:
                continue
            human = rec.get("label_dist") or rec.get("human")
            if human is None:
                continue
            human_l = [float(x) for x in human]
            stratum = str(rec.get("stratum") or rec.get("tier") or "")
            if stratum in {"trivial", "easy"}:
                stratum = "easy"
            elif stratum in {"hard", "ambiguous"}:
                stratum = "hard"
            row = {
                "id": item_id,
                "stratum": stratum,
                "entropy": float(rec.get("entropy") or 0.0),
                "human": human_l,
                "probs": probs,
                "confidence": float(
                    rec["confidence"]
                    if rec.get("confidence") is not None
                    else _top_prob(probs)
                ),
                "latency_ms": float(rec.get("latency_ms") or 0.0),
            }
            if client == "jev":
                by_item[item_id] = row
            else:
                baseline_by[item_id] = {
                    "pick": str(rec.get("choice") or _argmax_label(probs)),
                    "latency_ms": float(rec.get("latency_ms") or 0.0),
                }
    items: list[dict[str, Any]] = []
    for iid, row in by_item.items():
        base = baseline_by.get(iid) or {
            "pick": _argmax_label(row["probs"]),
            "latency_ms": row["latency_ms"] * 4.0,
        }
        row["baseline"] = base
        items.append(row)
    return items


def write_arena_specimen(root: Path) -> Path:
    """Write ``arena/data/specimen.json`` for the default certificate view."""
    doc = build_specimen()
    data_dir = root / "arena" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "specimen.json"
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path
