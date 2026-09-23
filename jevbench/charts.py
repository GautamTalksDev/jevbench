"""Video-ready charts from results / arrays.

Exports PNG (1080p + 1440p) and SVG. Colour tokens from ``arena/tokens.json``
so charts and Arena share one product look. Colour-blind safe: hue is always
paired with marker shape, dash pattern, or direct labels.

Every figure embeds ``run_id`` and resolved model version in a footer so
screenshots carry provenance.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from jevbench.metrics import ECEResult, ReliabilityCurve

SizeName = Literal["1080p", "1440p"]

_DASH_MAP = {
    "solid": "-",
    "dashed": "--",
    "dashdot": "-.",
    "dotted": ":",
}


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[1]


def default_tokens_path() -> Path:
    return repo_root_from_here() / "arena" / "tokens.json"


def load_tokens(path: Path | None = None) -> dict[str, Any]:
    p = path or default_tokens_path()
    data = json.loads(p.read_text(encoding="utf-8"))
    ref = data.get("reference") or {}
    if isinstance(ref.get("dash"), list):
        ref["dash"] = tuple(ref["dash"])
        data["reference"] = ref
    return data


@dataclass
class ChartProvenance:
    run_id: str
    resolved_model: str
    extra: str = ""

    def footer_text(self) -> str:
        bits = [f"run_id={self.run_id}", f"model={self.resolved_model}"]
        if self.extra:
            bits.append(self.extra)
        return "  ·  ".join(bits)


@dataclass
class ChartStyle:
    tokens: dict[str, Any]
    transparent: bool = False
    size: SizeName = "1080p"

    @property
    def typography(self) -> dict[str, Any]:
        return self.tokens["typography"]

    @property
    def brand(self) -> dict[str, str]:
        if self.transparent:
            o = self.tokens["overlay"]
            return {
                "ink": o["ink"],
                "paper": o["paper"],
                "muted": o["muted"],
                "hairline": o["hairline"],
                "accent": o["accent"],
                "accent_soft": self.tokens["brand"].get("accent_soft", o["accent"]),
                "danger": self.tokens["brand"]["danger"],
                "warn": self.tokens["brand"]["warn"],
            }
        return dict(self.tokens["brand"])

    def series(self, series_id: str) -> dict[str, Any]:
        for s in self.tokens["series"]:
            if s["id"] == series_id:
                return s
        return self.tokens["series"][0]

    def series_or_index(self, name: str, index: int) -> dict[str, Any]:
        sid = name.lower().replace(" ", "_").split("_")[0]
        for s in self.tokens["series"]:
            if s["id"] == sid or s["id"] in name.lower():
                return s
        return self.tokens["series"][index % len(self.tokens["series"])]

    def tier(self, tier_id: str) -> dict[str, Any]:
        for t in self.tokens["tiers"]:
            if t["id"] == tier_id:
                return t
        return self.tokens["tiers"][0]

    def linestyle(self, dash: Any) -> Any:
        if isinstance(dash, (list, tuple)):
            return (0, tuple(float(x) for x in dash))
        return _DASH_MAP.get(str(dash), "-")

    def figsize_inches(self) -> tuple[float, float]:
        w, h = self.tokens["export"]["sizes"][self.size]
        dpi = float(self.tokens["export"]["dpi_base"])
        return (w / dpi, h / dpi)

    def px_to_pt(self, px: float) -> float:
        dpi = float(self.tokens["export"]["dpi_base"])
        min_px = float(self.typography["min_px"])
        return max(px, min_px) * 72.0 / dpi

    def apply_rc(self) -> None:
        ty = self.typography
        brand = self.brand
        mpl.rcParams.update(
            {
                "font.family": ty.get("family", "DejaVu Sans"),
                "font.size": self.px_to_pt(ty["label_px"]),
                "axes.titlesize": self.px_to_pt(ty["title_px"]),
                "axes.labelsize": self.px_to_pt(ty["label_px"]),
                "xtick.labelsize": self.px_to_pt(ty["tick_px"]),
                "ytick.labelsize": self.px_to_pt(ty["tick_px"]),
                "legend.fontsize": self.px_to_pt(ty["label_px"]),
                "axes.edgecolor": brand["hairline"],
                "axes.labelcolor": brand["ink"],
                "axes.titlecolor": brand["ink"],
                "xtick.color": brand["muted"],
                "ytick.color": brand["muted"],
                "text.color": brand["ink"],
                "figure.facecolor": "none" if self.transparent else brand["paper"],
                "axes.facecolor": "none" if self.transparent else brand["paper"],
                "savefig.facecolor": "none" if self.transparent else brand["paper"],
                "savefig.transparent": self.transparent,
                "axes.grid": True,
                "grid.color": self.tokens["overlay"]["grid"]
                if self.transparent
                else brand["hairline"],
                "grid.alpha": 0.45,
                "grid.linestyle": ":",
            }
        )


@dataclass
class ChartContext:
    provenance: ChartProvenance
    style: ChartStyle
    out_dir: Path
    stem_prefix: str = ""

    def new_figure(self) -> Figure:
        self.style.apply_rc()
        return plt.figure(
            figsize=self.style.figsize_inches(),
            dpi=self.style.tokens["export"]["dpi_base"],
        )

    def add_footer(self, fig: Figure) -> None:
        brand = self.style.brand
        footer_pt = self.style.px_to_pt(self.style.typography["footer_px"])
        fig.text(
            0.5,
            0.015,
            self.provenance.footer_text(),
            ha="center",
            va="bottom",
            fontsize=footer_pt,
            color=brand["muted"],
            family=self.style.typography.get("family", "DejaVu Sans"),
        )

    def save(self, fig: Figure, stem: str) -> dict[str, Path]:
        self.add_footer(fig)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{self.stem_prefix}{stem}" if self.stem_prefix else stem
        paths: dict[str, Path] = {}
        dpi = float(self.style.tokens["export"]["dpi_base"])

        # SVG — vector; allow tight trim
        svg_path = self.out_dir / f"{prefix}.svg"
        fig.savefig(
            svg_path,
            format="svg",
            bbox_inches="tight",
            pad_inches=0.35,
            transparent=self.style.transparent,
        )
        paths["svg"] = svg_path

        # Raster at exact video pixel sizes (no tight crop — phones need the frame)
        for size_name, (w, h) in self.style.tokens["export"]["sizes"].items():
            fig.set_size_inches(w / dpi, h / dpi)
            png_path = self.out_dir / f"{prefix}_{size_name}.png"
            fig.savefig(
                png_path,
                format="png",
                dpi=dpi,
                facecolor=fig.get_facecolor(),
                transparent=self.style.transparent,
            )
            paths[size_name] = png_path
        plt.close(fig)
        return paths


def make_context(
    *,
    run_id: str,
    resolved_model: str,
    out_dir: Path | str,
    transparent: bool = False,
    size: SizeName = "1080p",
    tokens_path: Path | None = None,
    extra_footer: str = "",
    stem_prefix: str = "",
) -> ChartContext:
    tokens = load_tokens(tokens_path)
    style = ChartStyle(tokens=tokens, transparent=transparent, size=size)
    return ChartContext(
        provenance=ChartProvenance(
            run_id=run_id, resolved_model=resolved_model, extra=extra_footer
        ),
        style=style,
        out_dir=Path(out_dir),
        stem_prefix=stem_prefix,
    )


def _ref_style(ctx: ChartContext) -> tuple[str, Any, float]:
    brand = ctx.style.brand
    ref = ctx.style.tokens["reference"]
    color = brand["muted"] if ctx.style.transparent else ref["color"]
    return color, ctx.style.linestyle(ref["dash"]), float(ref["alpha"])


# ---------------------------------------------------------------------------
# 1. Reliability + occupancy
# ---------------------------------------------------------------------------


def reliability_diagram(
    ctx: ChartContext,
    *,
    ece: ECEResult | None = None,
    curve: ReliabilityCurve | None = None,
    title: str = "Reliability diagram",
    tier_label: str | None = None,
    stem: str = "reliability_diagram",
) -> dict[str, Path]:
    """Reliability curve with bin-occupancy histogram on the shared x-axis."""
    if ece is None and curve is None:
        raise ValueError("pass ece= or curve=")
    if curve is None and ece is not None:
        curve = ReliabilityCurve(
            bin_mean_confidence=ece.bin_mean_confidence,
            bin_accuracy=ece.bin_accuracy,
            bin_counts=ece.bin_counts,
            bin_edges=ece.bin_edges,
            strategy=ece.strategy,
            n_bins=ece.n_bins,
        )
    assert curve is not None

    fig = ctx.new_figure()
    brand = ctx.style.brand
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax: Axes = fig.add_subplot(gs[0])
    ax_occ: Axes = fig.add_subplot(gs[1], sharex=ax)

    conf = np.asarray(curve.bin_mean_confidence, dtype=float)
    acc = np.asarray(curve.bin_accuracy, dtype=float)
    counts = np.asarray(curve.bin_counts, dtype=float)
    edges = np.asarray(curve.bin_edges, dtype=float)
    mask = np.isfinite(conf) & np.isfinite(acc) & (counts > 0)

    ref_c, ref_ls, ref_a = _ref_style(ctx)
    ax.plot(
        [0, 1],
        [0, 1],
        color=ref_c,
        linestyle=ref_ls,
        linewidth=2.0,
        alpha=ref_a,
        label="perfect calibration (y = x)",
        zorder=1,
    )
    series = ctx.style.series("jev")
    ax.plot(
        conf[mask],
        acc[mask],
        color=series["color"],
        marker=series["marker"],
        markersize=14,
        linestyle=ctx.style.linestyle(series["dash"]),
        linewidth=2.5,
        label=f"empirical ({curve.strategy})",
        zorder=3,
    )
    for x, y, n in zip(conf[mask], acc[mask], counts[mask], strict=True):
        ax.annotate(
            f"n={int(n)}",
            (x, y),
            textcoords="offset points",
            xytext=(6, 6),
            color=brand["muted"],
            fontsize=ctx.style.px_to_pt(22),
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy in bin")
    ttl = title if tier_label is None else f"{title} — {tier_label}"
    if ece is not None:
        ttl += f"   ECE={ece.ece:.3f}"
    ax.set_title(ttl)
    ax.legend(loc="upper left", frameon=not ctx.style.transparent)

    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges) * 0.9
    ax_occ.bar(
        centers,
        counts,
        width=widths,
        color=series["color"],
        edgecolor=brand["ink"],
        linewidth=1.0,
        alpha=0.75,
        hatch="//",
        label="bin occupancy",
    )
    ax_occ.set_xlabel("Predicted probability")
    ax_occ.set_ylabel("Count")
    ax_occ.set_xlim(0, 1)
    ax_occ.legend(loc="upper right", frameon=not ctx.style.transparent)
    ax_occ.text(
        0.01,
        0.95,
        "occupancy (required — not optional)",
        transform=ax_occ.transAxes,
        va="top",
        ha="left",
        color=brand["muted"],
        fontsize=ctx.style.px_to_pt(22),
    )

    fig.subplots_adjust(bottom=0.12, top=0.9, left=0.1, right=0.96)
    out_stem = stem if tier_label is None else f"{stem}_{tier_label}"
    return ctx.save(fig, out_stem)


# ---------------------------------------------------------------------------
# 2. ECE vs accuracy by tier (money chart)
# ---------------------------------------------------------------------------


def ece_vs_accuracy_by_tier(
    ctx: ChartContext,
    *,
    tier_names: Sequence[str],
    accuracies: Sequence[float],
    eces: Sequence[float],
    strategy: str = "uniform",
    stem: str = "ece_vs_accuracy_by_tier",
) -> dict[str, Path]:
    fig = ctx.new_figure()
    ax = fig.add_subplot(111)
    brand = ctx.style.brand
    xs = np.asarray(accuracies, dtype=float)
    ys = np.asarray(eces, dtype=float)
    flat = float(np.nanmean(ys)) if np.isfinite(ys).any() else 0.0

    ref_c, ref_ls, ref_a = _ref_style(ctx)
    ax.axhline(
        flat,
        color=ref_c,
        linestyle=ref_ls,
        linewidth=2.0,
        alpha=max(ref_a, 0.7),
        label=f"flat reference (mean ECE={flat:.3f}) — calibration independent of accuracy",
        zorder=1,
    )
    for name, x, y in zip(tier_names, xs, ys, strict=True):
        t = ctx.style.tier(str(name))
        ax.scatter(
            [x],
            [y],
            s=280,
            c=t["color"],
            marker=t["marker"],
            edgecolors=brand["ink"],
            linewidths=1.5,
            zorder=3,
            label=f"{t['label']}  marker={t['marker']}",
        )
        ax.annotate(
            str(name),
            (x, y),
            textcoords="offset points",
            xytext=(10, 8),
            color=brand["ink"],
            fontweight="bold",
        )

    ax.set_xlabel("Accuracy")
    ax.set_ylabel(f"ECE ({strategy})")
    ax.set_title("ECE vs accuracy by difficulty tier")
    ax.legend(loc="best", frameon=not ctx.style.transparent)
    fig.subplots_adjust(bottom=0.14, top=0.9, left=0.1, right=0.96)
    return ctx.save(fig, stem)


# ---------------------------------------------------------------------------
# 3. Three-arm comparison
# ---------------------------------------------------------------------------


def three_arm_comparison(
    ctx: ChartContext,
    *,
    arms: Sequence[str],
    accuracy: Sequence[float],
    cost: Sequence[float],
    latency_p50: Sequence[float],
    accuracy_err: Sequence[float] | None = None,
    cost_err: Sequence[float] | None = None,
    latency_err: Sequence[float] | None = None,
    stem: str = "three_arm_comparison",
) -> dict[str, Path]:
    fig = ctx.new_figure()
    brand = ctx.style.brand
    panels = [
        ("Accuracy", accuracy, accuracy_err),
        ("Cost / 1k calls ($)", cost, cost_err),
        ("Latency p50 (ms)", latency_p50, latency_err),
    ]
    axes = fig.subplots(1, 3)
    markers = ["o", "s", "D", "^"]

    for ax, (title, vals, errs) in zip(axes, panels, strict=True):
        for i, (arm, v) in enumerate(zip(arms, vals, strict=True)):
            s = ctx.style.series_or_index(arm, i)
            err = None if errs is None else errs[i]
            ax.bar(
                i,
                v,
                width=0.55,
                color=s["color"],
                edgecolor=brand["ink"],
                hatch=s.get("hatch") or "",
                yerr=err,
                capsize=8,
                error_kw={"elinewidth": 2, "capthick": 2, "ecolor": brand["ink"]},
            )
            ax.scatter(
                [i],
                [v],
                marker=markers[i % len(markers)],
                s=140,
                c=brand["ink"],
                zorder=5,
            )
            ax.annotate(
                arm,
                (i, v),
                textcoords="offset points",
                xytext=(0, 14),
                ha="center",
                color=brand["ink"],
            )
        ax.set_xticks([])
        ax.set_title(title)
        ax.set_xlim(-0.6, len(arms) - 0.4)

    fig.suptitle("Three-arm comparison (bootstrap intervals)", y=0.98)
    fig.subplots_adjust(bottom=0.14, top=0.86, left=0.07, right=0.98, wspace=0.35)
    return ctx.save(fig, stem)


# ---------------------------------------------------------------------------
# 4. Latency distribution
# ---------------------------------------------------------------------------


def latency_distribution(
    ctx: ChartContext,
    *,
    series_latencies: dict[str, ArrayLike],
    stem: str = "latency_distribution",
    kind: Literal["ecdf", "violin"] = "ecdf",
) -> dict[str, Path]:
    """ECDF or violin with p50/p95/p99 marked. Never a bar chart of means."""
    fig = ctx.new_figure()
    ax = fig.add_subplot(111)
    brand = ctx.style.brand

    for i, (name, vals) in enumerate(series_latencies.items()):
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            continue
        s = ctx.style.series_or_index(name, i)
        color, marker = s["color"], s["marker"]
        dash = ctx.style.linestyle(s["dash"])
        p50, p95, p99 = (float(x) for x in np.percentile(arr, [50, 95, 99]))

        if kind == "ecdf":
            xs = np.sort(arr)
            ys = np.arange(1, len(xs) + 1) / len(xs)
            ax.plot(
                xs,
                ys,
                color=color,
                linestyle=dash,
                linewidth=2.5,
                label=f"{name}  p50={p50:.0f}  p95={p95:.0f}  p99={p99:.0f}",
            )
            for p, pv in [(p50, 0.5), (p95, 0.95), (p99, 0.99)]:
                ax.scatter(
                    [p],
                    [pv],
                    marker=marker,
                    s=110,
                    c=color,
                    edgecolors=brand["ink"],
                    zorder=4,
                )
        else:
            parts = ax.violinplot(
                [arr],
                positions=[i],
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body in parts["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor(brand["ink"])
                body.set_alpha(0.7)
            ax.scatter(
                [i, i, i],
                [p50, p95, p99],
                marker=marker,
                s=130,
                c=brand["ink"],
                zorder=5,
            )
            ax.annotate(
                f"{name}\np50={p50:.0f} p95={p95:.0f} p99={p99:.0f}",
                (i, p99),
                textcoords="offset points",
                xytext=(8, 8),
                color=brand["ink"],
            )

    if kind == "ecdf":
        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("ECDF")
        ax.set_ylim(0, 1.05)
        for y, lab in [(0.5, "p50"), (0.95, "p95"), (0.99, "p99")]:
            ax.axhline(y, color=brand["hairline"], linestyle=":", linewidth=1.2)
            ax.text(
                0.01,
                y + 0.015,
                lab,
                transform=ax.get_yaxis_transform(),
                color=brand["muted"],
            )
        ax.legend(loc="lower right", frameon=not ctx.style.transparent)
    else:
        ax.set_xticks(range(len(series_latencies)))
        ax.set_xticklabels(list(series_latencies.keys()))
        ax.set_ylabel("Latency (ms)")

    ax.set_title("Latency distribution (p50 / p95 / p99 — never means)")
    fig.subplots_adjust(bottom=0.14, top=0.9, left=0.1, right=0.96)
    return ctx.save(fig, stem)


# ---------------------------------------------------------------------------
# 5. Cost per 1000 correct
# ---------------------------------------------------------------------------


def cost_per_1000_correct(
    ctx: ChartContext,
    *,
    arms: Sequence[str],
    cost_per_1000: Sequence[float],
    errors: Sequence[float] | None = None,
    stem: str = "cost_per_1000_correct",
) -> dict[str, Path]:
    """Buyer's metric — almost nobody reports it."""
    fig = ctx.new_figure()
    ax = fig.add_subplot(111)
    brand = ctx.style.brand
    markers = ["o", "s", "D", "^"]

    for i, (arm, c) in enumerate(zip(arms, cost_per_1000, strict=True)):
        s = ctx.style.series_or_index(arm, i)
        err = None if errors is None else errors[i]
        ax.bar(
            i,
            c,
            width=0.55,
            color=s["color"],
            edgecolor=brand["ink"],
            hatch=s.get("hatch") or "",
            yerr=err,
            capsize=10,
            error_kw={"elinewidth": 2.5, "capthick": 2.5, "ecolor": brand["ink"]},
        )
        ax.scatter([i], [c], marker=markers[i % 4], s=160, c=brand["ink"], zorder=5)
        ax.annotate(
            f"{arm}\n${c:.4f}",
            (i, c),
            textcoords="offset points",
            xytext=(0, 16),
            ha="center",
            color=brand["ink"],
            fontweight="bold",
        )

    ax.set_xticks([])
    ax.set_ylabel("USD per 1,000 correct decisions")
    ax.set_title("Cost per 1,000 correct — the buyer's metric")
    ax.set_xlim(-0.6, len(arms) - 0.4)
    fig.subplots_adjust(bottom=0.14, top=0.9, left=0.12, right=0.96)
    return ctx.save(fig, stem)


# ---------------------------------------------------------------------------
# 6. Selective automation curve
# ---------------------------------------------------------------------------


def selective_automation_curve(
    ctx: ChartContext,
    *,
    coverage: Sequence[float],
    accuracy: Sequence[float],
    target_accuracy: float = 0.90,
    automatable_fraction: float | None = None,
    stem: str = "selective_automation_curve",
) -> dict[str, Path]:
    fig = ctx.new_figure()
    ax = fig.add_subplot(111)
    brand = ctx.style.brand
    series = ctx.style.series("jev")
    cov = np.asarray(coverage, dtype=float)
    acc = np.asarray(accuracy, dtype=float)
    order = np.argsort(cov)
    cov, acc = cov[order], acc[order]

    ax.plot(
        cov,
        acc,
        color=series["color"],
        marker=series["marker"],
        markersize=10,
        linestyle=ctx.style.linestyle(series["dash"]),
        linewidth=2.5,
        label="selective accuracy vs coverage",
    )
    _, ref_ls, _ = _ref_style(ctx)
    ax.axhline(
        target_accuracy,
        color=brand["warn"] if not ctx.style.transparent else brand["accent"],
        linestyle=ref_ls,
        linewidth=2.2,
        alpha=0.85,
        label=f"target accuracy = {target_accuracy:.0%}",
    )

    if automatable_fraction is None:
        # Largest coverage with accuracy >= target
        ok = acc >= target_accuracy - 1e-12
        automatable_fraction = float(cov[ok].max()) if ok.any() else 0.0

    ax.axvline(
        automatable_fraction,
        color=brand["accent"],
        linestyle="--",
        linewidth=2.0,
        label=f"automatable fraction = {automatable_fraction:.1%}",
    )
    ax.scatter(
        [automatable_fraction],
        [target_accuracy],
        s=220,
        marker="*",
        c=brand["accent"],
        edgecolors=brand["ink"],
        zorder=5,
    )
    ax.annotate(
        f"{automatable_fraction:.1%} automatable @ {target_accuracy:.0%}",
        (automatable_fraction, target_accuracy),
        textcoords="offset points",
        xytext=(12, -24),
        color=brand["ink"],
        fontweight="bold",
    )

    ax.set_xlim(0, 1.02)
    ax.set_ylim(min(0.5, float(np.nanmin(acc)) - 0.05), 1.02)
    ax.set_xlabel("Coverage (fraction automated)")
    ax.set_ylabel("Accuracy on automated set")
    ax.set_title("Selective automation curve")
    ax.legend(loc="lower left", frameon=not ctx.style.transparent)
    fig.subplots_adjust(bottom=0.14, top=0.9, left=0.1, right=0.96)
    return ctx.save(fig, stem)


# ---------------------------------------------------------------------------
# 7. Invariant scatter (Noul vs Choice)
# ---------------------------------------------------------------------------


def invariant_scatter(
    ctx: ChartContext,
    *,
    noul_probs: ArrayLike,
    choice_probs: ArrayLike,
    stem: str = "invariant_scatter",
) -> dict[str, Path]:
    """Noul vs Choice for the same question; distance from y=x is the story."""
    fig = ctx.new_figure()
    ax = fig.add_subplot(111)
    brand = ctx.style.brand
    x = np.asarray(noul_probs, dtype=float)
    y = np.asarray(choice_probs, dtype=float)
    series = ctx.style.series("jev")

    ref_c, ref_ls, ref_a = _ref_style(ctx)
    ax.plot(
        [0, 1],
        [0, 1],
        color=ref_c,
        linestyle=ref_ls,
        linewidth=2.0,
        alpha=ref_a,
        label="agreement line (y = x)",
        zorder=1,
    )
    ax.scatter(
        x,
        y,
        s=90,
        c=series["color"],
        marker=series["marker"],
        edgecolors=brand["ink"],
        linewidths=0.8,
        alpha=0.85,
        label="item (Noul vs Choice)",
        zorder=3,
    )
    # Highlight farthest point
    dist = np.abs(x - y)
    if len(dist):
        i = int(np.nanargmax(dist))
        ax.scatter(
            [x[i]],
            [y[i]],
            s=220,
            facecolors="none",
            edgecolors=brand["danger"],
            linewidths=2.5,
            marker="o",
            label=f"max |Δ|={dist[i]:.2f}",
            zorder=4,
        )
        ax.annotate(
            f"|Δ|={dist[i]:.2f}",
            (x[i], y[i]),
            textcoords="offset points",
            xytext=(10, 10),
            color=brand["danger"],
            fontweight="bold",
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Noul probability")
    ax.set_ylabel("Choice P(positive)")
    ax.set_title("Primitive invariant — Noul vs Choice")
    ax.legend(loc="upper left", frameon=not ctx.style.transparent)
    ax.set_aspect("equal", adjustable="box")
    fig.subplots_adjust(bottom=0.14, top=0.9, left=0.1, right=0.96)
    return ctx.save(fig, stem)


# ---------------------------------------------------------------------------
# 8. Context rot curve
# ---------------------------------------------------------------------------


def context_rot_curve(
    ctx: ChartContext,
    *,
    state_tokens: Sequence[float],
    accuracy: Sequence[float],
    accuracy_err: Sequence[float] | None = None,
    series_name: str = "jev",
    stem: str = "context_rot_curve",
) -> dict[str, Path]:
    fig = ctx.new_figure()
    ax = fig.add_subplot(111)
    brand = ctx.style.brand
    s = ctx.style.series(series_name)
    xs = np.asarray(state_tokens, dtype=float)
    ys = np.asarray(accuracy, dtype=float)
    yerr = None if accuracy_err is None else np.asarray(accuracy_err, dtype=float)

    ax.errorbar(
        xs,
        ys,
        yerr=yerr,
        color=s["color"],
        marker=s["marker"],
        markersize=14,
        linestyle=ctx.style.linestyle(s["dash"]),
        linewidth=2.5,
        capsize=8,
        ecolor=brand["ink"],
        label=f"{series_name} accuracy vs state size",
    )
    for x, y in zip(xs, ys, strict=True):
        ax.annotate(
            f"{int(x)} tok\n{y:.2f}",
            (x, y),
            textcoords="offset points",
            xytext=(8, 8),
            color=brand["ink"],
        )

    ax.set_xlabel("State size (tokens)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Context rot — accuracy vs irrelevant state size")
    ax.legend(loc="best", frameon=not ctx.style.transparent)
    fig.subplots_adjust(bottom=0.14, top=0.9, left=0.1, right=0.96)
    return ctx.save(fig, stem)


# ---------------------------------------------------------------------------
# 9. Boundary flip-rate (EXP-6)
# ---------------------------------------------------------------------------


def boundary_flip_curve(
    ctx: ChartContext,
    *,
    decile_centers: Sequence[float],
    flip_rates: Sequence[float],
    confidence_sds: Sequence[float] | None = None,
    stem: str = "boundary_flip_curve",
) -> dict[str, Path]:
    """Flip rate vs boundary distance — the EXP-6 reconciliation plot."""
    fig = ctx.new_figure()
    ax = fig.add_subplot(111)
    brand = ctx.style.brand
    xs = np.asarray(decile_centers, dtype=float)
    ys = np.asarray(flip_rates, dtype=float)
    ax.plot(
        xs,
        ys,
        color=ctx.style.series("jev")["color"],
        marker="o",
        markersize=12,
        linewidth=2.5,
        label="flip rate",
    )
    if confidence_sds is not None:
        ax2 = ax.twinx()
        ax2.plot(
            xs,
            confidence_sds,
            color=ctx.style.series("adapter")["color"],
            marker="s",
            linestyle="--",
            linewidth=2.0,
            label="mean confidence SD",
        )
        ax2.set_ylabel("Mean confidence SD")
        ax2.tick_params(colors=brand["muted"])
    ax.set_xlabel("|mean confidence − decision threshold|")
    ax.set_ylabel("Flip rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("EXP-6 — label flips vs boundary distance")
    ax.legend(loc="upper right", frameon=not ctx.style.transparent)
    fig.subplots_adjust(bottom=0.14, top=0.9, left=0.1, right=0.9)
    return ctx.save(fig, stem)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _demo_all(ctx: ChartContext) -> dict[str, dict[str, Path]]:
    """Smoke-render every chart from synthetic data (offline / CI)."""
    from jevbench.metrics import expected_calibration_error

    rng = np.random.default_rng(0)
    probs = rng.uniform(0.05, 0.95, 400)
    labels = (rng.uniform(0, 1, 400) < probs).astype(int)
    ece = expected_calibration_error(probs, labels, n_bins=10)

    out: dict[str, dict[str, Path]] = {}
    out["reliability"] = reliability_diagram(ctx, ece=ece, tier_label="easy")
    out["ece_vs_acc"] = ece_vs_accuracy_by_tier(
        ctx,
        tier_names=["trivial", "easy", "hard", "ambiguous"],
        accuracies=[0.95, 0.88, 0.72, 0.58],
        eces=[0.04, 0.06, 0.11, 0.16],
    )
    out["three_arm"] = three_arm_comparison(
        ctx,
        arms=["jev", "adapter", "trivial"],
        accuracy=[0.87, 0.84, 0.71],
        accuracy_err=[0.02, 0.025, 0.03],
        cost=[0.12, 4.5, 0.0],
        cost_err=[0.01, 0.4, 0.0],
        latency_p50=[90, 420, 1],
        latency_err=[10, 40, 0.2],
    )
    out["latency"] = latency_distribution(
        ctx,
        series_latencies={
            "jev": rng.normal(90, 20, 200).clip(20),
            "adapter": rng.normal(400, 80, 200).clip(50),
        },
        kind="ecdf",
    )
    out["cost_1k"] = cost_per_1000_correct(
        ctx,
        arms=["jev", "adapter", "trivial"],
        cost_per_1000=[0.15, 6.2, 0.0],
        errors=[0.02, 0.5, 0.0],
    )
    cov = np.linspace(0.2, 1.0, 20)
    acc = 0.98 - 0.15 * (cov - 0.2)
    out["selective"] = selective_automation_curve(
        ctx, coverage=cov, accuracy=acc, target_accuracy=0.90
    )
    out["invariant"] = invariant_scatter(
        ctx,
        noul_probs=rng.uniform(0.1, 0.9, 80),
        choice_probs=rng.uniform(0.1, 0.9, 80),
    )
    out["context_rot"] = context_rot_curve(
        ctx,
        state_tokens=[1000, 4000, 16000, 32000],
        accuracy=[0.91, 0.88, 0.79, 0.71],
        accuracy_err=[0.02, 0.02, 0.03, 0.03],
    )
    out["boundary_flip"] = boundary_flip_curve(
        ctx,
        decile_centers=[0.02, 0.08, 0.15, 0.25, 0.4],
        flip_rates=[0.55, 0.35, 0.2, 0.08, 0.02],
        confidence_sds=[0.01, 0.012, 0.015, 0.02, 0.018],
    )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="jevbench video charts")
    parser.add_argument(
        "--out",
        type=Path,
        default=repo_root_from_here() / "results" / "charts",
        help="Output directory",
    )
    parser.add_argument("--run-id", default="demo-run")
    parser.add_argument("--model", default="jev-1.13.0")
    parser.add_argument(
        "--transparent",
        action="store_true",
        help="Transparent background for video overlays",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Render all chart types from synthetic data",
    )
    parser.add_argument("--tokens", type=Path, default=None)
    args = parser.parse_args(argv)

    ctx = make_context(
        run_id=args.run_id,
        resolved_model=args.model,
        out_dir=args.out,
        transparent=args.transparent,
        tokens_path=args.tokens,
    )
    paths = _demo_all(ctx)
    for name, pmap in paths.items():
        print(f"{name}: " + ", ".join(f"{k}={v.name}" for k, v in pmap.items()))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
