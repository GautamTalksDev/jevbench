"""EXP-1 verdict rule (Amendment 9) — parametric corrected tests.

tracks = corrected ΔECE >= 0.09 AND p < 0.05
         (p = two-sided parametric null at the observed probabilities)

holds  = one-sided test rejects H0: ΔECE >= 0.09 at level 0.05
         (simulate under true ΔECE = 0.09) AND corrected ΔECE <= 0.02

else inconclusive.

The certificate page must display ``result.verdict`` from the harness —
never recompute a zone from the interval.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from jevbench.power import (
    POWER_THRESHOLD,
    SOFT_NULL_OPERATING_KAPPA,
    TARGET_DELTA_ECE,
    _calibrate_soft_alt_shift_corrected,
    soft_delta_ece_corrected,
    simulate_soft_beta_binomial_stratum,
    specimen_top_prob_pools,
)

HOLDS_THRESHOLD = 0.02
TRACKS_THRESHOLD = TARGET_DELTA_ECE  # 0.09
VerdictLabel = Literal["tracks", "holds", "inconclusive"]


def decide_verdict(
    *,
    corrected_delta_ece: float,
    p_value: float,
    p_holds_reject_ge_effect: float,
) -> dict[str, Any]:
    """Apply the Amendment 9 decision rule to one arm's primary test."""
    corr = float(corrected_delta_ece)
    p = float(p_value)
    p_hold = float(p_holds_reject_ge_effect)

    tracks_ok = corr >= TRACKS_THRESHOLD and p < 0.05
    holds_ok = p_hold < 0.05 and corr <= HOLDS_THRESHOLD

    if tracks_ok and holds_ok:
        # Should be nearly impossible; prefer tracks only if both somehow fire.
        label: VerdictLabel = "inconclusive"
        reason = (
            "Both tracks and holds conditions fired — treat as inconclusive "
            "and inspect the numbers."
        )
    elif tracks_ok:
        label = "tracks"
        reason = (
            f"corrected ΔECE={corr:.4f} ≥ {TRACKS_THRESHOLD} and "
            f"parametric p={p:.4f} < 0.05."
        )
    elif holds_ok:
        label = "holds"
        reason = (
            f"one-sided test rejects ΔECE≥{TRACKS_THRESHOLD} "
            f"(p={p_hold:.4f}<0.05) and corrected ΔECE={corr:.4f} "
            f"≤ {HOLDS_THRESHOLD}."
        )
    else:
        label = "inconclusive"
        reason = (
            f"corrected ΔECE={corr:.4f}, p={p:.4f}, "
            f"p_holds={p_hold:.4f} — neither tracks nor holds."
        )

    return {
        "verdict": label,
        "reason": reason,
        "corrected_delta_ece": corr,
        "p_value": p,
        "p_holds_reject_ge_effect": p_hold,
        "tracks_threshold": TRACKS_THRESHOLD,
        "holds_threshold": HOLDS_THRESHOLD,
        "rule": (
            "tracks = corrected≥0.09 AND p<0.05; "
            "holds = reject Δ≥0.09 (one-sided, sim under 0.09) AND corrected≤0.02; "
            "else inconclusive"
        ),
    }


def p_holds_one_sided(
    observed_corrected: float,
    null_corrected_at_effect: np.ndarray,
) -> float:
    """Left-tailed p for H0: ΔECE ≥ 0.09 using sims under true ΔECE = 0.09."""
    sims = np.asarray(null_corrected_at_effect, dtype=float)
    if sims.size < 1:
        return float("nan")
    return float(np.mean(sims <= float(observed_corrected)))


def simulate_corrected_delta_ece(
    *,
    setting: Literal["null", "alternative"],
    n: int = 750,
    kappa: float = SOFT_NULL_OPERATING_KAPPA,
    n_trials: int = 500,
    n_boot: int = 400,
    n_e0: int = 400,
    n_null_pval: int = 400,
    seed: int = 20260924,
    conf_shift: float | None = None,
) -> dict[str, Any]:
    """Draw corrected ΔECE (and p) under null or ΔECE≈0.09 alternative."""
    p_easy, p_hard = specimen_top_prob_pools()
    if conf_shift is None:
        conf_shift = (
            0.0
            if setting == "null"
            else _calibrate_soft_alt_shift_corrected(
                p_easy, p_hard, kappa=kappa, n=n, target=TARGET_DELTA_ECE, seed=seed + 3
            )
        )
    rng = np.random.default_rng(seed)
    corr_pts: list[float] = []
    raw_pts: list[float] = []
    p_vals: list[float] = []
    for _ in range(n_trials):
        trial = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        hard = simulate_soft_beta_binomial_stratum(
            n,
            trial,
            kappa=kappa,
            p_pool=p_hard,
            conf_shift=float(conf_shift) if setting == "alternative" else 0.0,
        )
        easy = simulate_soft_beta_binomial_stratum(
            n, trial, kappa=kappa, p_pool=p_easy, conf_shift=0.0
        )
        iv = soft_delta_ece_corrected(
            hard.conf,
            hard.soft_correct,
            easy.conf,
            easy.soft_correct,
            kappa=kappa,
            n_boot=n_boot,
            n_e0=n_e0,
            n_null_pval=n_null_pval,
            rng=trial,
        )
        corr_pts.append(float(iv["corrected_delta_ece"]))
        raw_pts.append(float(iv["raw_delta_ece"]))
        p_vals.append(float(iv["p_value"]))
    return {
        "setting": setting,
        "conf_shift": float(conf_shift),
        "corrected": np.asarray(corr_pts, dtype=float),
        "raw": np.asarray(raw_pts, dtype=float),
        "p_value": np.asarray(p_vals, dtype=float),
        "mean_corrected": float(np.mean(corr_pts)),
        "mean_raw": float(np.mean(raw_pts)),
    }


def verify_verdict_size(
    *,
    n_trials: int = 1000,
    n: int = 750,
    seed: int = 20260924,
    quick: bool = False,
) -> dict[str, Any]:
    """Re-verify size for tracks and holds tests by simulation (PROMPT Q)."""
    if quick:
        n_trials = 80
        n_boot = n_e0 = n_null = 200
    else:
        n_boot = 1000
        n_e0 = 1000
        n_null = 1000

    null = simulate_corrected_delta_ece(
        setting="null",
        n=n,
        n_trials=n_trials,
        n_boot=n_boot,
        n_e0=n_e0,
        n_null_pval=n_null,
        seed=seed,
    )
    alt = simulate_corrected_delta_ece(
        setting="alternative",
        n=n,
        n_trials=n_trials,
        n_boot=n_boot,
        n_e0=n_e0,
        n_null_pval=n_null,
        seed=seed + 17,
    )

    # Reference distribution for holds one-sided test (true ΔECE = 0.09).
    effect_ref = alt["corrected"]

    tracks_null = 0
    holds_null = 0
    p05_null = 0
    tracks_alt = 0
    holds_alt = 0
    reject_ge_at_boundary = 0

    for i in range(n_trials):
        p_hold_n = p_holds_one_sided(float(null["corrected"][i]), effect_ref)
        v_n = decide_verdict(
            corrected_delta_ece=float(null["corrected"][i]),
            p_value=float(null["p_value"][i]),
            p_holds_reject_ge_effect=p_hold_n,
        )
        tracks_null += int(v_n["verdict"] == "tracks")
        holds_null += int(v_n["verdict"] == "holds")
        p05_null += int(float(null["p_value"][i]) < 0.05)

        p_hold_a = p_holds_one_sided(float(alt["corrected"][i]), effect_ref)
        v_a = decide_verdict(
            corrected_delta_ece=float(alt["corrected"][i]),
            p_value=float(alt["p_value"][i]),
            p_holds_reject_ge_effect=p_hold_a,
        )
        tracks_alt += int(v_a["verdict"] == "tracks")
        holds_alt += int(v_a["verdict"] == "holds")
        # Size of one-sided reject-≥0.09 at the boundary (leave-one-out-ish:
        # compare each alt draw to the other alt draws).
        others = np.concatenate([effect_ref[:i], effect_ref[i + 1 :]]) if n_trials > 1 else effect_ref
        p_bound = p_holds_one_sided(float(alt["corrected"][i]), others)
        reject_ge_at_boundary += int(p_bound < 0.05)

    fpr_p = p05_null / n_trials
    fpr_tracks = tracks_null / n_trials
    fpr_holds = holds_null / n_trials
    size_holds_boundary = reject_ge_at_boundary / n_trials
    power_tracks = tracks_alt / n_trials

    return {
        "schema": "jevbench.verdict_size.v1",
        "n_trials": n_trials,
        "n_per_stratum": n,
        "seed": seed,
        "null": {
            "mean_corrected": null["mean_corrected"],
            "fpr_pvalue": fpr_p,
            "fpr_tracks": fpr_tracks,
            "rate_holds_when_true_delta_0": fpr_holds,
        },
        "alternative_at_0_09": {
            "mean_corrected": alt["mean_corrected"],
            "power_tracks": power_tracks,
            "rate_holds": holds_alt / n_trials,
            "size_one_sided_reject_ge_0_09": size_holds_boundary,
            "power_pvalue": float(np.mean(alt["p_value"] < 0.05)),
        },
        "acceptance": {
            "pvalue_fpr_ok": 0.03 <= fpr_p <= 0.08,
            "holds_boundary_size_ok": 0.03 <= size_holds_boundary <= 0.08,
            "tracks_null_fpr_low": fpr_tracks <= 0.01,
            "holds_at_effect_rare": (holds_alt / n_trials) <= 0.05,
            "pvalue_power_at_0_09_ok": float(np.mean(alt["p_value"] < 0.05))
            >= POWER_THRESHOLD,
            "tracks_power_at_boundary_note": (
                f"Composite tracks power at true ΔECE=0.09 is {power_tracks:.3f}. "
                "Requiring corrected≥0.09 when the true mean is 0.09 yields ~50% "
                "power by construction (median split). The parametric p-value "
                "test itself retains high power; the paper must not claim 80% "
                "power for the composite tracks gate at exactly 0.09."
            ),
            "note": (
                "Size checks: parametric p FPR under null; one-sided reject-≥0.09 "
                "size under true Δ=0.09; tracks under null rare; holds under "
                "true Δ=0.09 rare. When true Δ=0, holds is the correct verdict "
                "(not a false positive)."
            ),
        },
        "jev_data_observed": False,
    }


def verdict_copy(label: VerdictLabel, *, synthetic: bool = False) -> tuple[str, str]:
    """Title and lede for the certificate page (display only)."""
    if synthetic and label == "tracks":
        return (
            "Specimen: tracks accuracy (not a finding)",
            "This synthetic run lands in the tracks verdict so you can see the "
            "page. It means nothing about Jev.",
        )
    if label == "tracks":
        return (
            "Calibration tracks accuracy",
            "Corrected ΔECE reaches the pre-registered 0.09 and the parametric "
            "test rejects the calibrated null at 0.05.",
        )
    if label == "holds":
        return (
            "Calibration holds across difficulty",
            "Corrected ΔECE stays ≤ 0.02 and a one-sided test rejects "
            "ΔECE ≥ 0.09 at level 0.05.",
        )
    return (
        "Inconclusive on the pre-registered rule",
        "Neither the tracks nor the holds test clears its Amendment 9 gate.",
    )
