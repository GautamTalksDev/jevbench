"""Offline reproduction: metrics + charts from committed JSONL.

No network. No API key. Regenerates every number and chart under
``results/harness_fixture/`` from ``runs/offline_fixture/raw.jsonl`` in under
two minutes. These are harness-check goldens only — not study results.

Bit-for-bit check: ``make check-repro`` verifies ``results/SHA256SUMS``.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np

# Deterministic Agg backend before any pyplot import via charts.
mpl.use("Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/jevbench-mpl")
mpl.rcParams["svg.hashsalt"] = "jevbench-repro-v1"

# Fixture-only SVG date freeze (2026-09-19). Set inside reproduce(), not at import.
_REPRO_SOURCE_DATE_EPOCH = "1789776000"

from jevbench.charts import (  # noqa: E402
    ece_vs_accuracy_by_tier,
    make_context,
    reliability_diagram,
)
from jevbench.cost import resolve_price  # noqa: E402
from jevbench.metrics import (  # noqa: E402
    BootstrapCI,
    calibration_by_tier,
    paired_bootstrap,
)

REPO = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO / "runs" / "offline_fixture"
RAW_JSONL = FIXTURE_DIR / "raw.jsonl"
MANIFEST = FIXTURE_DIR / "manifest.json"
RESULTS = REPO / "results"
HARNESS = RESULTS / "harness_fixture"
CHARTS = HARNESS / "charts"
METRICS_PATH = HARNESS / "metrics.json"
FINDING_PATH = HARNESS / "finding.json"
SUMS_PATH = RESULTS / "SHA256SUMS"

LABEL_ORDER = ("billing", "technical", "other")
TIER_ORDER = ("trivial", "easy", "hard", "ambiguous")
CLIENT = "jev"
N_BOOT = 10_000
BOOT_SEED = 20_260_919
PRICE_SNAPSHOT = "2026-09-19"


def _round_floats(obj: Any, ndigits: int = 12) -> Any:
    if isinstance(obj, float):
        if np.isnan(obj):
            return None
        if np.isinf(obj):
            return obj
        return round(float(obj), ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_floats(v, ndigits) for v in obj]
    if isinstance(obj, (np.floating,)):
        return _round_floats(float(obj), ndigits)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def load_records(path: Path = RAW_JSONL) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing committed fixture: {path}")
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _prob_row(probabilities: dict[str, float]) -> list[float]:
    return [float(probabilities.get(lab, 0.0)) for lab in LABEL_ORDER]


def _label_index(label: str) -> int:
    return LABEL_ORDER.index(label)


def group_by_tier(
    records: list[dict[str, Any]], *, client: str = CLIENT
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, Any]]]:
    by_tier_probs: dict[str, list[list[float]]] = defaultdict(list)
    by_tier_labels: dict[str, list[int]] = defaultdict(list)
    kept: list[dict[str, Any]] = []
    for rec in records:
        if rec.get("client") != client:
            continue
        tier = rec["tier"]
        by_tier_probs[tier].append(_prob_row(rec["probabilities"]))
        by_tier_labels[tier].append(_label_index(rec["label"]))
        kept.append(rec)
    probs = {t: np.asarray(by_tier_probs[t], dtype=float) for t in TIER_ORDER if t in by_tier_probs}
    labels = {
        t: np.asarray(by_tier_labels[t], dtype=int) for t in TIER_ORDER if t in by_tier_labels
    }
    return probs, labels, kept


def _ece_occupancy_payload(ece) -> dict[str, Any]:
    return {
        "ece": ece.ece,
        "mce": ece.mce,
        "n": ece.n,
        "n_bins": ece.n_bins,
        "strategy": ece.strategy,
        "max_bin_fraction": ece.max_bin_fraction,
        "bin_counts": list(ece.bin_counts),
        "bin_mean_confidence": list(ece.bin_mean_confidence),
        "bin_accuracy": list(ece.bin_accuracy),
        "bin_edges": list(ece.bin_edges),
        "bins": [
            {
                "index": b.index,
                "lo": b.lo,
                "hi": b.hi,
                "count": b.count,
                "fraction": b.fraction,
                "mean_confidence": b.mean_confidence,
                "accuracy": b.accuracy,
                "abs_gap": b.abs_gap,
            }
            for b in ece.bins
        ],
    }


def slope_from_arrays(
    probs_by_tier: dict[str, np.ndarray],
    labels_by_tier: dict[str, np.ndarray],
) -> float:
    cal = calibration_by_tier(probs_by_tier, labels_by_tier, tier_order=TIER_ORDER)
    return float(cal.slope_uniform)


def bootstrap_slope_ci(
    kept: list[dict[str, Any]],
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> BootstrapCI:
    """Item-level bootstrap CI on the ECE-vs-accuracy OLS slope."""
    n = len(kept)
    if n == 0:
        return BootstrapCI(
            point=float("nan"),
            low=float("nan"),
            high=float("nan"),
            n_resamples=n_boot,
        )

    rng = np.random.default_rng(seed)

    def _slope_of_indices(idxs: np.ndarray) -> float:
        by_p: dict[str, list[list[float]]] = defaultdict(list)
        by_y: dict[str, list[int]] = defaultdict(list)
        for i in idxs:
            rec = kept[int(i)]
            by_p[rec["tier"]].append(_prob_row(rec["probabilities"]))
            by_y[rec["tier"]].append(_label_index(rec["label"]))
        if any(t not in by_p or len(by_p[t]) < 2 for t in TIER_ORDER):
            return float("nan")
        probs = {t: np.asarray(by_p[t], dtype=float) for t in TIER_ORDER}
        labels = {t: np.asarray(by_y[t], dtype=int) for t in TIER_ORDER}
        return slope_from_arrays(probs, labels)

    point = _slope_of_indices(np.arange(n))
    samples: list[float] = []
    for _ in range(n_boot):
        draw = rng.integers(0, n, size=n)
        s = _slope_of_indices(draw)
        if np.isfinite(s):
            samples.append(s)
    if not samples:
        return BootstrapCI(
            point=point, low=float("nan"), high=float("nan"), n_resamples=n_boot
        )
    arr = np.asarray(samples, dtype=float)
    low, high = np.quantile(arr, [0.025, 0.975])
    return BootstrapCI(
        point=float(point), low=float(low), high=float(high), n_resamples=n_boot
    )


def estimate_fixture_cost_usd(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Actual dollars for this offline fixture: $0 (no API). Also estimate if it were live."""
    by_client: dict[str, float] = defaultdict(float)
    for rec in records:
        client = rec["client"]
        model = rec.get("model")
        try:
            price = resolve_price(PRICE_SNAPSHOT, client, model)
        except Exception:
            # adapter without model → gpt-4o-mini default
            price = resolve_price(PRICE_SNAPSHOT, client, "gpt-4o-mini")
        by_client[client] += price.cost_usd(
            int(rec.get("input_tokens") or 0), int(rec.get("output_tokens") or 0)
        )
    hypothetical = sum(by_client.values())
    return {
        "actual_usd": 0.0,
        "note": "Offline fixture — no API calls. actual_usd is always 0 for make reproduce.",
        "hypothetical_live_usd_if_replayed": round(hypothetical, 6),
        "by_client_hypothetical_usd": {k: round(v, 6) for k, v in sorted(by_client.items())},
        "pricing_snapshot": PRICE_SNAPSHOT,
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_round_floats(data), indent=2, sort_keys=True, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def write_checksums(paths: list[Path], out: Path = SUMS_PATH) -> None:
    lines: list[str] = []
    for p in sorted(paths, key=lambda x: x.as_posix()):
        rel = p.relative_to(RESULTS).as_posix()
        lines.append(f"{sha256_file(p)}  {rel}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_checksums(sums_path: Path = SUMS_PATH) -> None:
    if not sums_path.is_file():
        raise SystemExit(f"missing {sums_path}")
    failed = 0
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, name = line.partition("  ")
        path = RESULTS / name
        if not path.is_file():
            print(f"MISSING {name}")
            failed += 1
            continue
        got = sha256_file(path)
        if got != digest:
            print(f"MISMATCH {name}\n  expect {digest}\n  got    {got}")
            failed += 1
        else:
            print(f"ok  {name}")
    if failed:
        raise SystemExit(f"check-repro failed: {failed} file(s)")
    print("check-repro: all checksums match")


def reproduce(*, check: bool = False) -> dict[str, Any]:
    t0 = time.perf_counter()
    if check:
        verify_checksums()
        return {}

    prior_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    os.environ["SOURCE_DATE_EPOCH"] = _REPRO_SOURCE_DATE_EPOCH
    try:
        return _reproduce_body(t0=t0)
    finally:
        if prior_epoch is None:
            os.environ.pop("SOURCE_DATE_EPOCH", None)
        else:
            os.environ["SOURCE_DATE_EPOCH"] = prior_epoch


def _reproduce_body(*, t0: float) -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.is_file() else {}
    records = load_records()
    probs, labels, kept = group_by_tier(records)
    if set(probs) != set(TIER_ORDER):
        raise RuntimeError(f"fixture missing tiers: have {sorted(probs)}, need {list(TIER_ORDER)}")

    cal = calibration_by_tier(probs, labels, tier_order=TIER_ORDER)
    slope_ci = bootstrap_slope_ci(kept)

    # Paired accuracy delta jev vs adapter (same items) for a secondary interval
    jev_by_id = {r["item_id"]: r for r in records if r["client"] == "jev"}
    ad_by_id = {r["item_id"]: r for r in records if r["client"] == "adapter"}
    common = sorted(set(jev_by_id) & set(ad_by_id))
    jev_correct = np.asarray(
        [1.0 if jev_by_id[i]["choice"] == jev_by_id[i]["label"] else 0.0 for i in common]
    )
    ad_correct = np.asarray(
        [1.0 if ad_by_id[i]["choice"] == ad_by_id[i]["label"] else 0.0 for i in common]
    )
    acc_delta_ci = paired_bootstrap(
        jev_correct,
        ad_correct,
        statistic=lambda a, b: float(np.mean(a) - np.mean(b)),
        n=N_BOOT,
        seed=BOOT_SEED,
    )

    cost = estimate_fixture_cost_usd(records)

    by_tier_payload = {}
    for tc in cal.tiers:
        by_tier_payload[tc.tier] = {
            "n": tc.n,
            "accuracy": tc.accuracy,
            "brier": tc.brier,
            "ece_uniform": _ece_occupancy_payload(tc.ece_uniform),
            "ece_quantile": _ece_occupancy_payload(tc.ece_quantile),
        }

    metrics = {
        "schema": "jevbench.metrics.v1",
        "source": {
            "fixture": str(RAW_JSONL.relative_to(REPO)),
            "run_id": manifest.get("run_id", "offline_fixture"),
            "resolved_model": manifest.get("resolved_model", "jev-1.13.0"),
            "scored": False,
            "status": "offline_fixture",
        },
        "client": CLIENT,
        "label_order": list(LABEL_ORDER),
        "tier_order": list(TIER_ORDER),
        "calibration_by_tier": {
            "tier_names": list(cal.tier_names),
            "accuracies": list(cal.accuracies),
            "ece_uniform": list(cal.ece_uniform),
            "ece_quantile": list(cal.ece_quantile),
            "slope_uniform": cal.slope_uniform,
            "slope_quantile": cal.slope_quantile,
            "note": cal.note,
            "by_tier": by_tier_payload,
        },
        "slope_uniform_bootstrap_ci": {
            **slope_ci.to_dict(),
            "n_items": len(kept),
            "seed": BOOT_SEED,
            "level": 0.95,
        },
        "accuracy_delta_jev_minus_adapter_ci": {
            **acc_delta_ci.to_dict(),
            "n_items": len(common),
            "seed": BOOT_SEED,
            "level": 0.95,
        },
        "cost": cost,
        "n_records": len(records),
        "n_jev_items": len(kept),
    }
    write_json(METRICS_PATH, metrics)

    # Headline finding for README / consumers
    finding = {
        "headline": (
            f"On the committed offline fixture (n={len(kept)} Jev items, not a scored live run), "
            f"the OLS slope of uniform ECE vs tier accuracy is {slope_ci.point:.4f} "
            f"(95% bootstrap CI [{slope_ci.low:.4f}, {slope_ci.high:.4f}], "
            f"{N_BOOT} resamples, seed={BOOT_SEED}). "
            "Inspect per-tier occupancy before interpreting ECE; this pack is a harness check, "
            "not the EXP-1 claim."
        ),
        "slope_point": slope_ci.point,
        "slope_ci_low": slope_ci.low,
        "slope_ci_high": slope_ci.high,
        "n": len(kept),
        "chart": "results/harness_fixture/charts/ece_vs_accuracy_by_tier_1080p.png",
        "actual_cost_usd": 0.0,
    }
    write_json(FINDING_PATH, finding)

    # Charts
    CHARTS.mkdir(parents=True, exist_ok=True)
    ctx = make_context(
        run_id=manifest.get("run_id", "offline_fixture"),
        resolved_model=manifest.get("resolved_model", "jev-1.13.0"),
        out_dir=CHARTS,
        transparent=False,
        extra_footer="offline fixture · not a scored run",
    )
    chart_paths: list[Path] = []
    money = ece_vs_accuracy_by_tier(
        ctx,
        tier_names=list(cal.tier_names),
        accuracies=list(cal.accuracies),
        eces=list(cal.ece_uniform),
        strategy="uniform",
        stem="ece_vs_accuracy_by_tier",
    )
    chart_paths.extend(money.values())

    # One reliability diagram per tier (hard is the money close-up)
    for tc in cal.tiers:
        paths = reliability_diagram(
            ctx,
            ece=tc.ece_uniform,
            tier_label=tc.tier,
            stem="reliability",
            title="Reliability",
        )
        chart_paths.extend(paths.values())

    # Checksum metrics + SVG (cross-platform friendlier) + PNGs (CI Linux bit-for-bit)
    hashed = [METRICS_PATH, FINDING_PATH] + [
        p for p in chart_paths if p.suffix in {".svg", ".png"}
    ]
    # Deduplicate
    uniq: list[Path] = []
    seen: set[str] = set()
    for p in hashed:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    write_checksums(uniq)

    elapsed = time.perf_counter() - t0
    summary = {
        "elapsed_s": round(elapsed, 3),
        "metrics": str(METRICS_PATH.relative_to(REPO)),
        "finding": finding["headline"],
        "charts": len(chart_paths),
    }
    print(json.dumps(summary, indent=2))
    if elapsed > 120:
        print(f"WARNING: reproduce took {elapsed:.1f}s (>120s budget)")
    return summary


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Offline jevbench reproduce")
    p.add_argument(
        "--check",
        action="store_true",
        help="Verify results/SHA256SUMS against on-disk artifacts (no regen)",
    )
    args = p.parse_args()
    reproduce(check=args.check)


if __name__ == "__main__":
    main()
