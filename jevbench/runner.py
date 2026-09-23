"""Scored-run executor.

Records only — never computes metrics. Manifest is written BEFORE any call.
raw.jsonl is streamed per call so a crash mid-run leaves a resumable prefix.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from jevbench.chaosnli import hydrate_dataset, logged_state, missing_primary_text
from jevbench.clients.base import DecisionClient, SystemOneRequest
from jevbench.cost import (
    estimate_tokens_from_text,
    resolve_price,
    snapshot_as_dict,
)
from jevbench.dataset import Dataset, LabeledItem, load_dataset, sha256_file
from jevbench.experiment import ClientSpec, ExperimentSpec, load_experiment
from jevbench.prereg import (
    PreregistrationError,
    assert_hashes_unchanged,
    datasets_root,
    load_lock,
    verify_labels,
)
from jevbench.rate_limit import (
    DEFAULT_HEADROOM,
    PUBLISHED_REQUESTS_PER_MIN,
    PUBLISHED_TOKENS_PER_SEC,
    RateLimitConfig,
    TokenBucketLimiter,
)

console = Console(stderr=True)

PACKAGE_NAMES = (
    "jevbench",
    "typesafe-sdk",
    "system-one-adapter",
    "httpx",
    "pydantic",
    "numpy",
    "scikit-learn",
    "netcal",
    "mapie",
)


@dataclass(frozen=True)
class WorkUnit:
    """One API call identity — used for resume dedup."""

    pass_idx: int
    item_id: str
    client_name: str

    @property
    def key(self) -> str:
        return f"{self.pass_idx}:{self.item_id}:{self.client_name}"


@dataclass
class RunStats:
    calls_done: int = 0
    errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    spend_usd: float = 0.0
    spend_by_client: dict[str, float] = field(default_factory=dict)
    t0: float = field(default_factory=time.perf_counter)

    def note_call(
        self,
        *,
        client_name: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        error: bool,
    ) -> None:
        self.calls_done += 1
        if error:
            self.errors += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.spend_usd += cost
        self.spend_by_client[client_name] = (
            self.spend_by_client.get(client_name, 0.0) + cost
        )

    @property
    def calls_per_sec(self) -> float:
        elapsed = max(time.perf_counter() - self.t0, 1e-9)
        return self.calls_done / elapsed


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def git_sha(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def new_run_id(experiment: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{experiment}_{stamp}_{uuid.uuid4().hex[:8]}"


def runs_dir(repo_root: Path) -> Path:
    return repo_root / "runs"


def run_path(repo_root: Path, run_id: str) -> Path:
    return runs_dir(repo_root) / run_id


def state_to_text(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def build_questions_token_estimate(spec: ExperimentSpec) -> int:
    total = 0
    for key, q in spec.questions.items():
        blob = json.dumps(
            {"key": key, "kind": q.kind, "instructions": q.instructions, "criteria": q.criteria},
            ensure_ascii=False,
        )
        total += estimate_tokens_from_text(blob)
    return total


def build_client(
    cspec: ClientSpec,
    *,
    experiment: ExperimentSpec,
    transport: Callable[..., Any] | None = None,
) -> DecisionClient:
    """Construct a live client. Tests inject via ``client_factory`` instead."""
    if cspec.type == "jev":
        from jevbench.clients.jev import JevClient, JevClientConfig

        path = cspec.serving_path or experiment.serving_path
        cfg = JevClientConfig(
            model=cspec.model or experiment.model,
            serving_path=path if path in ("native", "gateway") else "native",
            base_url=cspec.extras.get("base_url"),
        )
        return JevClient(cfg, transport=transport)
    if cspec.type == "adapter":
        from jevbench.clients.adapter import AdapterClient, AdapterClientConfig

        cfg = AdapterClientConfig(
            provider=cspec.provider or "openai",
            model=cspec.model or experiment.baseline_model or "gpt-4o-mini",
            llm_answer_mode=cspec.llm_answer_mode,
        )
        return AdapterClient(cfg, transport=transport)
    if cspec.type == "prefill":
        from jevbench.clients.prefill import PrefillClient, PrefillClientConfig

        cfg = PrefillClientConfig(
            model=cspec.model or "local-open-weight",
            backend=cspec.prefill_backend,
            base_url=cspec.prefill_base_url,
            # Tests / dry fixtures inject backends; live runs assert tokens
            require_tokenizer_assert=bool(
                cspec.extras.get("require_tokenizer_assert", True)
            ),
        )
        return PrefillClient(cfg)
    if cspec.type == "gliclass":
        from jevbench.clients.gliclass import GLiClassClient, GLiClassClientConfig

        return GLiClassClient(
            GLiClassClientConfig(
                model_id=cspec.model or "knowledgator/gliclass-base-v1.0",
                resolved_model=cspec.model,
            )
        )
    if cspec.type == "trivial":
        from jevbench.clients.trivial import (
            KeywordRule,
            TrivialClient,
            TrivialClientConfig,
            TrivialQuestionSpec,
        )

        # Default floor: department triage for support_tickets, majority for NLI
        specs = {}
        for qkey, qspec in experiment.questions.items():
            if qspec.kind == "choice" and isinstance(qspec.criteria, dict):
                keys = set(qspec.criteria)
                if keys >= {"entailment", "neutral", "contradiction"}:
                    specs[qkey] = TrivialQuestionSpec(majority_class="neutral")
                else:
                    specs[qkey] = TrivialQuestionSpec(
                        keywords=(
                            KeywordRule(
                                "billing",
                                (r"refund", r"charg", r"invoice", r"VAT", r"seat"),
                            ),
                            KeywordRule(
                                "technical",
                                (r"crash", r"bug", r"password", r"webhook", r"deploy"),
                            ),
                        ),
                        majority_class="other",
                    )
            elif qspec.kind == "noul":
                specs[qkey] = TrivialQuestionSpec(
                    regex=r"refund|urgent|crash", noul_default=0.0
                )
            elif qspec.kind == "score":
                specs[qkey] = TrivialQuestionSpec(
                    majority_class=qspec.criteria[0]
                    if isinstance(qspec.criteria, list)
                    else 0
                )
        return TrivialClient(TrivialClientConfig(specs=specs))
    raise ValueError(f"unknown client type {cspec.type!r}")


def work_units(
    items: list[LabeledItem],
    clients: list[ClientSpec],
    repeats: int,
) -> list[WorkUnit]:
    units: list[WorkUnit] = []
    for pass_idx in range(repeats):
        for item in items:
            for client in clients:
                units.append(
                    WorkUnit(pass_idx=pass_idx, item_id=item.id, client_name=client.name)
                )
    return units


def load_completed_keys(raw_path: Path) -> set[str]:
    done: set[str] = set()
    if not raw_path.is_file():
        return done
    with raw_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = row.get("unit_key")
            if key:
                done.add(str(key))
            else:
                # Backward-compatible reconstruct
                done.add(
                    f"{row.get('pass', 0)}:{row.get('item_id')}:{row.get('client')}"
                )
    return done


def append_raw(raw_path: Path, record: dict[str, Any], lock: threading.Lock) -> None:
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with lock, raw_path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def build_manifest(
    *,
    run_id: str,
    repo_root: Path,
    spec: ExperimentSpec,
    dataset: Dataset,
    repeats: int,
    concurrency: int,
    geography_note: str,
) -> dict[str, Any]:
    lock = load_lock(repo_root)
    hashes = {
        "items_sha256": sha256_file(dataset.root / "items.jsonl"),
        "labels_sha256": sha256_file(dataset.root / "labels.jsonl"),
    }
    return {
        "run_id": run_id,
        "status": "running",
        "started_at": utc_now_iso(),
        "ended_at": None,
        "experiment": spec.name,
        "task": spec.task,
        "model_requested": spec.model,
        "serving_path": spec.serving_path,
        "geography_note": geography_note,
        "pricing_snapshot_date": spec.pricing_snapshot_date,
        "pricing": snapshot_as_dict(spec.pricing_snapshot_date),
        "git_sha": git_sha(repo_root),
        "package_versions": package_versions(),
        "platform": {
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "node": platform.node(),
        },
        "dataset": {
            "task": spec.task,
            "n_items": len(dataset),
            "tier_counts": dataset.tier_counts(),
            **hashes,
        },
        "items_sha256": hashes["items_sha256"],
        "labels_sha256": hashes["labels_sha256"],
        "preregistration_lock": lock.to_dict() if lock else None,
        "clients": [c.model_dump() for c in spec.clients],
        "questions": {k: q.model_dump() for k, q in spec.questions.items()},
        "repeats": repeats,
        "concurrency": concurrency,
        "rate_limit": {
            "headroom": spec.rate_limit.headroom,
            "tokens_per_sec": spec.rate_limit.tokens_per_sec
            or PUBLISHED_TOKENS_PER_SEC * (spec.rate_limit.headroom or DEFAULT_HEADROOM),
            "requests_per_min": spec.rate_limit.requests_per_min
            or PUBLISHED_REQUESTS_PER_MIN * (spec.rate_limit.headroom or DEFAULT_HEADROOM),
            "published_tokens_per_sec": PUBLISHED_TOKENS_PER_SEC,
            "published_requests_per_min": PUBLISHED_REQUESTS_PER_MIN,
        },
        "resolved_models_seen": [],
        "notes": (
            "Runner records only. Metrics are a separate step. "
            "raw.jsonl is append-only and resumable."
        ),
    }


def dry_run_estimate(
    *,
    spec: ExperimentSpec,
    dataset: Dataset,
    repeats: int | None = None,
) -> dict[str, Any]:
    """Estimate tokens and cost for Jev vs baselines — spend nothing."""
    n_repeats = repeats if repeats is not None else spec.effective_repeats()
    q_tokens = build_questions_token_estimate(spec)
    items = list(dataset.by_id.values())
    clients = spec.clients or [
        ClientSpec(name="jev", type="jev", model=spec.model),
    ]

    by_client: dict[str, dict[str, float | int | str]] = {}
    total_jev = 0.0
    total_baseline = 0.0
    total_calls = 0

    for cspec in clients:
        input_tok = 0
        output_tok = 0  # unknown; assume ~50 for structured answers dry-run
        for item in items:
            state_tok = estimate_tokens_from_text(state_to_text(item.item.state))
            # One call carries state + all questions
            input_tok += state_tok + q_tokens
            output_tok += 50
        input_tok *= n_repeats
        output_tok *= n_repeats
        n_calls = len(items) * n_repeats
        total_calls += n_calls
        price = resolve_price(
            spec.pricing_snapshot_date,
            cspec.type,
            cspec.model or spec.model,
        )
        cost = price.cost_usd(input_tok, output_tok if cspec.type != "jev" else 0)
        # Jev output free — force output cost 0
        if cspec.type == "jev":
            cost = price.cost_usd(input_tok, 0)
            total_jev += cost
        else:
            total_baseline += cost
        by_client[cspec.name] = {
            "type": cspec.type,
            "model": cspec.model or spec.model,
            "calls": n_calls,
            "est_input_tokens": input_tok,
            "est_output_tokens": output_tok if cspec.type != "jev" else 0,
            "est_cost_usd": round(cost, 6),
        }

    return {
        "experiment": spec.name,
        "task": spec.task,
        "n_items": len(items),
        "repeats": n_repeats,
        "total_calls": total_calls,
        "pricing_snapshot_date": spec.pricing_snapshot_date,
        "by_client": by_client,
        "split_usd": {
            "jev": round(total_jev, 6),
            "baselines": round(total_baseline, 6),
            "total": round(total_jev + total_baseline, 6),
        },
        "warning": (
            "Estimates use chars/4 tokenization. Jev is nearly free; "
            "LLM baselines will dominate the bill."
        ),
    }


def print_dry_run(estimate: dict[str, Any]) -> None:
    console.print(f"[bold]Dry-run[/bold] — {estimate['experiment']} / {estimate['task']}")
    console.print(
        f"  items={estimate['n_items']}  repeats={estimate['repeats']}  "
        f"calls={estimate['total_calls']}  "
        f"pricing={estimate['pricing_snapshot_date']}"
    )
    table = Table("client", "type", "calls", "est input tok", "est $")
    for name, row in estimate["by_client"].items():
        table.add_row(
            name,
            str(row["type"]),
            str(row["calls"]),
            f"{int(row['est_input_tokens']):,}",
            f"${float(row['est_cost_usd']):.4f}",
        )
    console.print(table)
    split = estimate["split_usd"]
    console.print(
        f"  [cyan]Jev[/cyan] ${split['jev']:.4f}  |  "
        f"[yellow]baselines[/yellow] ${split['baselines']:.4f}  |  "
        f"[bold]total[/bold] ${split['total']:.4f}"
    )
    console.print(f"  [dim]{estimate['warning']}[/dim]")


@dataclass
class RunnerConfig:
    repo_root: Path
    experiment: str
    repeats: int | None = None
    concurrency: int | None = None
    resume_run_id: str | None = None
    dry_run: bool = False
    geography_note: str | None = None
    skip_lock_check: bool = False  # tests / emergency only
    # Inject clients for offline tests: name -> DecisionClient
    client_factory: Callable[[ClientSpec], DecisionClient] | None = None
    progress: bool = True


class Runner:
    """Execute an experiment YAML into runs/<run_id>/{manifest,raw}.jsonl."""

    def __init__(self, config: RunnerConfig) -> None:
        self.config = config
        self.repo_root = config.repo_root
        self.spec = load_experiment(config.repo_root, config.experiment)
        self.dataset = load_dataset(datasets_root(config.repo_root), self.spec.task)
        if self.spec.task == "chaosnli":
            hydrate_dataset(self.dataset, self.dataset.root / "cache" / "text.jsonl")
        self.repeats = config.repeats if config.repeats is not None else self.spec.effective_repeats()
        self.concurrency = (
            config.concurrency if config.concurrency is not None else self.spec.concurrency
        )
        self.geography = config.geography_note or self.spec.geography_note
        self._raw_lock = threading.Lock()
        self._manifest_lock = threading.Lock()
        self._stats = RunStats()
        self._resolved_models: set[str] = set()
        self._prefill_manifest: dict[str, Any] = {}

    def _guard(self) -> None:
        if self.config.skip_lock_check:
            return
        if self.config.dry_run:
            # Dry-run may proceed without a lock, but warn
            lock = load_lock(self.repo_root)
            if lock is None:
                console.print(
                    "[yellow]warning:[/yellow] no preregistration.lock.json — "
                    "dry-run only; scored runs will refuse"
                )
            else:
                assert_hashes_unchanged(self.repo_root, lock)
            self._check_sample_size(warn_only=True)
            self._require_chaosnli_text(warn_only=True)
            return
        lock = load_lock(self.repo_root)
        if lock is None:
            raise PreregistrationError(
                "No preregistration.lock.json — run "
                "`jevbench preregister <experiment>` before scored runs."
            )
        if lock.task != self.spec.task:
            raise PreregistrationError(
                f"Lock task {lock.task!r} != experiment task {self.spec.task!r}"
            )
        assert_hashes_unchanged(self.repo_root, lock)
        verify_labels(self.repo_root)
        self._require_chaosnli_text(warn_only=False)
        self._check_sample_size(warn_only=False)

    def _require_chaosnli_text(self, *, warn_only: bool) -> None:
        if self.spec.task != "chaosnli":
            return
        missing = missing_primary_text(self.dataset)
        if not missing:
            return
        msg = (
            f"{len(missing)} ChaosNLI items have no local sentence text. "
            "Run scripts/fetch_chaosnli.py. Premises and hypotheses are not "
            "stored in the repository."
        )
        if warn_only:
            console.print(f"[yellow]warning:[/yellow] {msg}")
        else:
            raise PreregistrationError(msg)

    def _check_sample_size(self, *, warn_only: bool) -> None:
        """Refuse underpowered scored runs (Prompt F/G gate)."""
        ss = self.spec.sample_size
        if ss is None or ss.n_per_stratum is None:
            return
        need = int(ss.n_per_stratum)
        tiers = self.dataset.tiers()
        # Paraphrase rows are a contamination control, not the ΔECE sample.
        def _primary_count(tier: str) -> int:
            return sum(
                1
                for li in tiers.get(tier, [])  # type: ignore[arg-type]
                if getattr(li.item, "role", "primary") == "primary"
            )

        n_hard = _primary_count("hard") + _primary_count("ambiguous")
        n_easy = _primary_count("trivial") + _primary_count("easy")
        msg = (
            f"Sample size gate: need ≥{need} per pooled stratum; "
            f"have hard∪ambiguous={n_hard}, trivial∪easy={n_easy}."
        )
        if n_hard >= need and n_easy >= need:
            return
        underpowered_note = None
        pa = self.repo_root / "results" / "power_analysis.json"
        if pa.is_file():
            try:
                import json

                payload = json.loads(pa.read_text(encoding="utf-8"))
                if payload.get("underpowered"):
                    underpowered_note = payload.get("underpowered_note")
            except (OSError, json.JSONDecodeError, TypeError):
                pass
        full = msg
        if underpowered_note:
            full = f"{msg} Power analysis: {underpowered_note}"
        if warn_only:
            console.print(f"[yellow]warning:[/yellow] {full}")
        else:
            raise PreregistrationError(
                full
                + " Do not score underpowered — grow labels or pre-specify "
                "a larger detectable effect size."
            )

    def _rate_limiter(self) -> TokenBucketLimiter:
        rl = self.spec.rate_limit
        headroom = rl.headroom
        return TokenBucketLimiter(
            RateLimitConfig(
                tokens_per_sec=rl.tokens_per_sec
                or PUBLISHED_TOKENS_PER_SEC * headroom,
                requests_per_min=rl.requests_per_min
                or PUBLISHED_REQUESTS_PER_MIN * headroom,
            )
        )

    def _clients(self) -> dict[str, tuple[ClientSpec, DecisionClient]]:
        if not self.spec.clients:
            raise ValueError(
                f"experiment {self.spec.name!r} has no clients — declare clients: in YAML"
            )
        out: dict[str, tuple[ClientSpec, DecisionClient]] = {}
        for cspec in self.spec.clients:
            if self.config.client_factory is not None:
                client = self.config.client_factory(cspec)
            else:
                client = build_client(cspec, experiment=self.spec)
            out[cspec.name] = (cspec, client)
        return out

    def run(self) -> Path | dict[str, Any]:
        self._guard()
        if not self.spec.questions:
            raise ValueError(
                f"experiment {self.spec.name!r} has no questions — cannot run"
            )
        if self.config.dry_run:
            estimate = dry_run_estimate(
                spec=self.spec, dataset=self.dataset, repeats=self.repeats
            )
            print_dry_run(estimate)
            return estimate

        items = sorted(self.dataset.by_id.values(), key=lambda x: x.id)
        units = work_units(items, self.spec.clients, self.repeats)

        if self.config.resume_run_id:
            run_id = self.config.resume_run_id
            rdir = run_path(self.repo_root, run_id)
            if not rdir.is_dir():
                raise FileNotFoundError(f"run not found: {rdir}")
            manifest_path = rdir / "manifest.json"
            raw_path = rdir / "raw.jsonl"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            done = load_completed_keys(raw_path)
            console.print(
                f"[cyan]Resuming[/cyan] {run_id} — {len(done)} calls already recorded"
            )
        else:
            run_id = new_run_id(self.spec.name)
            rdir = run_path(self.repo_root, run_id)
            rdir.mkdir(parents=True, exist_ok=False)
            manifest_path = rdir / "manifest.json"
            raw_path = rdir / "raw.jsonl"
            raw_path.touch()
            # Manifest FIRST — before any call
            manifest = build_manifest(
                run_id=run_id,
                repo_root=self.repo_root,
                spec=self.spec,
                dataset=self.dataset,
                repeats=self.repeats,
                concurrency=self.concurrency,
                geography_note=self.geography,
            )
            write_manifest(manifest_path, manifest)
            done = set()

        pending = [u for u in units if u.key not in done]
        clients = self._clients()
        # EXP-3 gate: prefill sentinel single-token assertion before scoring
        for name, (cspec, client) in clients.items():
            ensure = getattr(client, "ensure_tokenizer_ready", None)
            if callable(ensure) and not self.config.dry_run:
                try:
                    ensure()
                except Exception as exc:
                    raise PreregistrationError(
                        f"Prefill sentinel token assertion failed for client "
                        f"{name!r}: {exc}. EXP-3 cannot score until every "
                        "sentinel is exactly one token in THIS tokenizer."
                    ) from exc
        limiter = self._rate_limiter()
        questions = self.spec.question_map()
        item_by_id = {li.id: li for li in items}
        git = git_sha(self.repo_root)

        def execute(unit: WorkUnit) -> dict[str, Any]:
            cspec, client = clients[unit.client_name]
            item = item_by_id[unit.item_id]
            # Pre-estimate tokens for the limiter (state + questions)
            est_tokens = estimate_tokens_from_text(
                state_to_text(item.item.state)
            ) + build_questions_token_estimate(self.spec)
            limiter.acquire(tokens=est_tokens, requests=1)

            request = SystemOneRequest(
                item_id=item.id,
                state=item.item.state,
                questions=questions,
                model=cspec.model or self.spec.model,
                pass_idx=unit.pass_idx,
            )
            ts = utc_now_iso()
            decisions = client.decide(request)
            # Capture prefill sentinel mapping (methodological — for manifest)
            mapper = getattr(client, "sentinel_mappings_for_manifest", None)
            if callable(mapper):
                with self._manifest_lock:
                    self._prefill_manifest[f"pass_{unit.pass_idx}"] = mapper()
            # Actual tokens from first decision (shared across questions in one call)
            in_tok = max((d.input_tokens for d in decisions), default=0)
            out_tok = max((d.output_tokens for d in decisions), default=0)
            err = any(d.error for d in decisions)
            for d in decisions:
                if d.resolved_model:
                    self._resolved_models.add(d.resolved_model)

            price = resolve_price(
                self.spec.pricing_snapshot_date,
                cspec.type,
                cspec.model or self.spec.model,
            )
            cost = price.cost_usd(in_tok, 0 if cspec.type == "jev" else out_tok)

            record = {
                "run_id": run_id,
                "unit_key": unit.key,
                "pass": unit.pass_idx,
                "item_id": unit.item_id,
                "tier": item.tier,
                "role": item.item.role,
                "source_item_id": (
                    item.item.state.get("source_item_id")
                    if isinstance(item.item.state, dict)
                    else None
                ),
                "client": unit.client_name,
                "client_type": cspec.type,
                "timestamp": ts,
                "git_sha": git,
                "serving_path": getattr(client, "serving_path", self.spec.serving_path),
                "model_requested": request.model,
                "request": {
                    "state": logged_state(item.item.state),
                    "questions": {k: q.model_dump() for k, q in self.spec.questions.items()},
                    "model": request.model,
                },
                "decisions": [d.to_dict() for d in decisions],
                "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
                "est_cost_usd": cost,
                "error": err,
            }
            append_raw(raw_path, record, self._raw_lock)
            self._stats.note_call(
                client_name=unit.client_name,
                input_tokens=in_tok,
                output_tokens=out_tok,
                cost=cost,
                error=err,
            )
            return record

        total = len(pending)
        if total == 0:
            console.print("[green]Nothing to do — all units already in raw.jsonl[/green]")
            manifest["status"] = "completed"
            manifest["ended_at"] = utc_now_iso()
            write_manifest(manifest_path, manifest)
            return rdir

        progress_ctx = (
            Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TextColumn("• {task.fields[rate]} calls/s"),
                TextColumn("• ${task.fields[spend]:.4f}"),
                TextColumn("• err={task.fields[errors]}"),
                console=console,
                disable=not self.config.progress,
            )
            if self.config.progress
            else None
        )

        def _run_pool() -> None:
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                futures = {pool.submit(execute, u): u for u in pending}
                if progress_ctx is not None:
                    task_id = progress_ctx.add_task(
                        f"run {run_id}",
                        total=total,
                        rate="0.0",
                        spend=0.0,
                        errors=0,
                    )
                    for fut in as_completed(futures):
                        try:
                            fut.result()
                        except Exception as exc:
                            console.print(f"[red]worker crashed:[/red] {exc}")
                            raise
                        progress_ctx.update(
                            task_id,
                            advance=1,
                            rate=f"{self._stats.calls_per_sec:.2f}",
                            spend=self._stats.spend_usd,
                            errors=self._stats.errors,
                        )
                else:
                    for fut in as_completed(futures):
                        fut.result()

        if progress_ctx is not None:
            with progress_ctx:
                _run_pool()
        else:
            _run_pool()

        # Finalize manifest (still no metrics)
        with self._manifest_lock:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["status"] = "completed"
            manifest["ended_at"] = utc_now_iso()
            manifest["resolved_models_seen"] = sorted(self._resolved_models)
            manifest["stats"] = {
                "calls_done": self._stats.calls_done,
                "errors": self._stats.errors,
                "input_tokens": self._stats.input_tokens,
                "output_tokens": self._stats.output_tokens,
                "spend_usd": round(self._stats.spend_usd, 6),
                "spend_by_client": {
                    k: round(v, 6) for k, v in self._stats.spend_by_client.items()
                },
                "calls_per_sec": round(self._stats.calls_per_sec, 4),
            }
            if self._prefill_manifest:
                manifest["prefill_control"] = dict(self._prefill_manifest)
                manifest["prefill_latency_note"] = (
                    "Prefill wall-clock latency is NOT a fair comparison vs "
                    "hosted APIs. Prefer compute_only_ms. Do not use EXP-3 for "
                    "video latency claims."
                )
            write_manifest(manifest_path, manifest)

        console.print(
            f"[green]Done[/green] {run_id} — "
            f"{self._stats.calls_done} calls, "
            f"{self._stats.errors} errors, "
            f"${self._stats.spend_usd:.4f} est. "
            f"(raw → {raw_path})"
        )
        return rdir
