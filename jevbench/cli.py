"""jevbench CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="jevbench",
    help="Pre-registered calibration & stability study of TypeSafe Jev.",
    no_args_is_help=True,
)
console = Console(stderr=True)


def repo_root() -> Path:
    """Walk up from CWD to find the repo root (pyproject + datasets/)."""
    here = Path.cwd().resolve()
    for cand in [here, *here.parents]:
        if (cand / "pyproject.toml").is_file() and (cand / "datasets").is_dir():
            return cand
    # Fallback: package parent
    return Path(__file__).resolve().parents[1]


@app.command("preregister")
def preregister_cmd(
    experiment: str = typer.Argument(
        ...,
        help="Experiment name or path (e.g. exp1_difficulty_calibration)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite lock even if hashes changed (voids prior protocol).",
    ),
) -> None:
    """Lock hypotheses + dataset hashes into PREREGISTRATION.md."""
    from jevbench.prereg import PreregistrationError, preregister

    root = repo_root()
    try:
        md_path, lock_file, lock = preregister(root, experiment, force=force)
    except (PreregistrationError, FileNotFoundError, ValueError) as exc:
        console.print(f"[red]preregister failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]Locked[/green] experiment [bold]{lock.experiment}[/bold]")
    console.print(f"  task:    datasets/{lock.task}/")
    console.print(f"  items:   {lock.n_items}  sha256={lock.items_sha256[:12]}…")
    console.print(f"  labels:  {lock.n_labels}  sha256={lock.labels_sha256[:12]}…")
    console.print(f"  wrote:   {md_path.relative_to(root)}")
    console.print(f"  wrote:   {lock_file.relative_to(root)}")
    table = Table("tier", "n")
    for tier, n in lock.tier_counts.items():
        table.add_row(tier, str(n))
    console.print(table)


@app.command("verify-labels")
def verify_labels_cmd() -> None:
    """Fail if labels.jsonl changed after the first scored run (or lock)."""
    from jevbench.prereg import LabelContaminationError, PreregistrationError, verify_labels

    root = repo_root()
    try:
        verify_labels(root)
    except (LabelContaminationError, PreregistrationError) as exc:
        console.print(f"[red bold]verify-labels FAILED[/red bold]\n{exc}")
        raise typer.Exit(code=1) from exc

    console.print("[green]verify-labels OK[/green] — labels match lock / first run.")


@app.command("check-lock")
def check_lock_cmd() -> None:
    """Refuse-friendly check: dataset hashes still match preregistration.lock.json."""
    from jevbench.prereg import PreregistrationError, assert_hashes_unchanged, load_lock

    root = repo_root()
    lock = load_lock(root)
    try:
        assert_hashes_unchanged(root, lock)
    except PreregistrationError as exc:
        console.print(f"[red]check-lock FAILED[/red]\n{exc}")
        raise typer.Exit(code=1) from exc
    console.print("[green]check-lock OK[/green]")


@app.command("sample")
def sample_cmd(
    task: str = typer.Argument(..., help="Task name under datasets/"),
    n_per_tier: int = typer.Option(10, "--n-per-tier", min=1),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Print a balanced tier-stratified sample (for smoke checks)."""
    from jevbench.dataset import balanced_sample, load_dataset
    from jevbench.prereg import datasets_root

    root = repo_root()
    try:
        ds = load_dataset(datasets_root(root), task)
        sample = balanced_sample(ds, n_per_tier=n_per_tier, seed=seed)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]sample failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    for li in sample:
        console.print(f"{li.tier:10}  {li.id}  label={li.label.label!r}")


@app.command("run")
def run_cmd(
    experiment: str = typer.Argument(
        ...,
        help="Experiment YAML path or name (e.g. experiments/exp1_difficulty_calibration.yaml)",
    ),
    repeat: int | None = typer.Option(
        None,
        "--repeat",
        min=1,
        help="Override YAML repeats — identical request set, separate passes.",
    ),
    concurrency: int | None = typer.Option(
        None,
        "--concurrency",
        min=1,
        help="Override YAML concurrency.",
    ),
    resume: str | None = typer.Option(
        None,
        "--resume",
        help="Resume an existing run_id under runs/.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Estimate tokens and cost; spend nothing. Prints budget check.",
    ),
    confirm: bool = typer.Option(
        False,
        "--confirm",
        help="Required for live runs after reviewing --dry-run projection.",
    ),
    geography: str | None = typer.Option(
        None,
        "--geography",
        help="Override geography note recorded in the manifest.",
    ),
) -> None:
    """Execute an experiment: manifest first, stream raw.jsonl, no metrics."""
    from jevbench.budget import BudgetExceeded
    from jevbench.prereg import PreregistrationError
    from jevbench.runner import Runner, RunnerConfig

    root = repo_root()
    cfg = RunnerConfig(
        repo_root=root,
        experiment=experiment,
        repeats=repeat,
        concurrency=concurrency,
        resume_run_id=resume,
        dry_run=dry_run,
        confirm=confirm,
        geography_note=geography,
    )
    try:
        result = Runner(cfg).run()
    except (PreregistrationError, FileNotFoundError, ValueError, BudgetExceeded) as exc:
        console.print(f"[red]run failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if dry_run:
        return
    console.print(f"[dim]run dir:[/dim] {result}")


@app.command("charts")
def charts_cmd(
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Output directory (default: results/charts)"),
    ] = None,
    run_id: Annotated[str, typer.Option("--run-id")] = "demo-run",
    model: Annotated[str, typer.Option("--model")] = "jev-1.13.0",
    transparent: Annotated[
        bool,
        typer.Option(
            "--transparent",
            help="Transparent background for overlaying on video footage",
        ),
    ] = False,
) -> None:
    """Render video-sized chart pack (1080p + 1440p PNG, SVG)."""
    from jevbench.charts import _demo_all, make_context

    root = repo_root()
    out_dir = (out if out is not None else Path("results/charts"))
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    ctx = make_context(
        run_id=run_id,
        resolved_model=model,
        out_dir=out_dir,
        transparent=transparent,
    )
    paths = _demo_all(ctx)
    for name, pmap in paths.items():
        console.print(
            f"[green]{name}[/green]  "
            + "  ".join(f"{k}={v.name}" for k, v in pmap.items())
        )
    console.print(f"[dim]wrote → {out_dir}[/dim]")


@app.command("analyze-exp1")
def analyze_exp1_cmd(
    raw: Path | None = typer.Option(
        None,
        "--raw",
        help="raw.jsonl path (default: runs/offline_fixture/raw.jsonl)",
    ),
    quick: bool = typer.Option(False, "--quick", help="n_boot=500 smoke"),
    scored: bool = typer.Option(False, "--scored"),
) -> None:
    """Run EXP-1 ΔECE analysis → results/exp1.json."""
    import subprocess
    import sys

    root = repo_root()
    cmd = [sys.executable, str(root / "scripts" / "run_exp1_analyze.py")]
    if raw is not None:
        cmd.extend(["--raw", str(raw)])
    if quick:
        cmd.append("--quick")
    if scored:
        cmd.append("--scored")
    raise typer.Exit(code=subprocess.call(cmd))


@app.command("label-agreement")
def label_agreement_cmd(
    task: str = typer.Option("support_tickets", "--task"),
    recheck_power: bool = typer.Option(False, "--recheck-power"),
) -> None:
    """Cohen's kappa on double-labelled subset; optional F1 noise re-check."""
    import subprocess
    import sys

    root = repo_root()
    cmd = [
        sys.executable,
        str(root / "scripts" / "run_agreement.py"),
        "--task",
        task,
    ]
    if recheck_power:
        cmd.append("--recheck-power")
    raise typer.Exit(code=subprocess.call(cmd))


@app.command("analyze-exp2")
def analyze_exp2_cmd(
    raw: Path | None = typer.Option(
        None,
        "--raw",
        help="raw.jsonl path (default: runs/offline_fixture/raw.jsonl)",
    ),
    quick: bool = typer.Option(False, "--quick", help="n_boot=500 smoke"),
) -> None:
    """Run EXP-2 decomposition analysis → results/exp2.json."""
    import subprocess
    import sys

    root = repo_root()
    cmd = [sys.executable, str(root / "scripts" / "run_exp2_analyze.py")]
    if raw is not None:
        cmd.extend(["--raw", str(raw)])
    if quick:
        cmd.append("--quick")
    raise typer.Exit(code=subprocess.call(cmd))


@app.command("analyze-exp3")
def analyze_exp3_cmd(
    raw: Path | None = typer.Option(None, "--raw"),
    quick: bool = typer.Option(False, "--quick"),
) -> None:
    """Run EXP-3 moat analysis → results/exp3.json (deciding metric: ΔECE)."""
    import subprocess
    import sys

    root = repo_root()
    cmd = [sys.executable, str(root / "scripts" / "run_exp3_analyze.py")]
    if raw is not None:
        cmd.extend(["--raw", str(raw)])
    if quick:
        cmd.append("--quick")
    raise typer.Exit(code=subprocess.call(cmd))


@app.command("export-certificate")
def export_certificate_cmd(
    run_id: Annotated[str, typer.Argument(help="Run id under runs/ (or 'specimen')")],
    synthetic: bool = typer.Option(
        False,
        "--synthetic",
        help="Write a synthetic specimen (ignores run artifacts).",
    ),
) -> None:
    """Write results/<run_id>/certificate.json for the Arena certificate page.

    Copies ΔECE fields from the harness. Never recomputes them in the exporter.
    """
    from jevbench.certificate import export_certificate, match_check, write_arena_specimen
    import json

    root = repo_root()
    try:
        path = export_certificate(root, run_id, synthetic=synthetic)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        console.print(f"[red]export-certificate failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    doc = json.loads(path.read_text(encoding="utf-8"))
    check = match_check(doc)
    if doc["meta"].get("synthetic"):
        write_arena_specimen(root)
    console.print(f"[green]wrote[/green] {path.relative_to(root)}")
    console.print(
        f"  match check {check['symbol']}  "
        f"ΔECE={doc['result']['delta_ece']:.6f}  "
        f"synthetic={doc['meta'].get('synthetic')}"
    )
    if not check["ok"]:
        console.print(
            "[red]Page ECE from items would disagree with stamped result.[/red]"
        )
        raise typer.Exit(code=1)


def main(argv: list[str] | None = None) -> None:
    app(args=argv)


if __name__ == "__main__":
    app()
