"""ChaosNLI (SNLI + MNLI) population for EXP-1.

Sentence text is not stored in this package. ChaosNLI itself is CC BY-NC 4.0,
which allows non-commercial sharing with attribution, but the premises and
hypotheses are SNLI (CC BY-SA 4.0, share-alike) and MNLI (mixed OANC and
fiction terms, no single redistribution grant). This repository is MIT.
Shipping the sentences here would hand them out under terms their licenses
do not allow. Committed artifacts are item ids, annotator counts, entropy,
and stratum membership. A fetch script materialises the text locally.

αNLI is not part of the population. See :data:`ALPHANLI_EXCLUSION`.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from jevbench.metrics import subsample_to_equal_n

LABEL_ORDER = ("entailment", "neutral", "contradiction")
CODE_TO_LABEL = {"e": "entailment", "n": "neutral", "c": "contradiction"}
LABEL_TO_INDEX = {lab: i for i, lab in enumerate(LABEL_ORDER)}

SUBSAMPLE_SEED = 20260923
PARAPHRASE_SEED = 20260924
TARGET_N = 750
N_PARAPHRASE_PER_STRATUM = 50
LABELER = "chaosnli-v1.0-100"
LABELED_AT = "2020-09-29T00:00:00+00:00"

# Full sha256 of the v1.0 jsonl files. Prefixes 99f9015ddda7d85f and
# 8eb49b589488e7b1 match an independent acquisition table for ChaosNLI v1.0.
SNLI_SHA256 = "99f9015ddda7d85f66a087452bc30d53974314fe27e7d589e2f41ad44bd509c1"
MNLI_SHA256 = "8eb49b589488e7b1a9ec95fb8a864bd50f0509b4f808eed0aed3f8167617664d"
SNLI_N = 1514
MNLI_N = 1599
POPULATION_N = SNLI_N + MNLI_N  # 3113

ALPHANLI_EXCLUSION = (
    "αNLI is a different task. The input is an observation-start, two "
    "hypotheses, and an observation-end, and the label is which hypothesis "
    "better explains the transition — a 2-way abductive choice. SNLI and MNLI "
    "are 3-way entailment / neutral / contradiction on one premise–hypothesis "
    "pair. Entropy is not on the same scale (the maximum is log2(2) = 1 for "
    "αNLI and log2(3) ≈ 1.585 for SNLI/MNLI), and one argmax accuracy would "
    "mix two decision problems. ChaosNLI-α (1,532 items) is excluded. The "
    "population is the SNLI subset (1,514) plus the MNLI-matched subset "
    "(1,599), 3,113 items."
)

# Official Dropbox object returned "File Deleted" on 2026-09-23. The fetch
# script still tries it, then these mirrors, and refuses anything whose
# sha256 is not the pin above.
DROPBOX_ZIP = "https://www.dropbox.com/s/h4j7dqszmpt2679/chaosNLI_v1.0.zip?dl=1"
MIRROR_SNLI = (
    "https://raw.githubusercontent.com/corneliagru/label-variation-nli/"
    "main/data/raw/chaosNLI_snli.jsonl"
)
MIRROR_MNLI = (
    "https://raw.githubusercontent.com/m-sjollema/languageandainli/"
    "c853233bbf8370e26230ae1ebc8f27f258759a1f/chaosNLI_mnli_m.jsonl"
)


def shannon_entropy(counts: list[int] | np.ndarray) -> float:
    """Base-2 Shannon entropy of a count vector. ``0 log 0`` is 0."""
    c = np.asarray(counts, dtype=float)
    total = float(c.sum())
    if total <= 0:
        raise ValueError("label counts must sum to a positive number")
    p = c / total
    terms = np.zeros_like(p)
    nz = p > 0
    terms[nz] = p[nz] * np.log2(p[nz])
    return float(-terms.sum())


def load_chaosnli_jsonl(path: Path, source: str) -> list[dict[str, Any]]:
    """Load one ChaosNLI jsonl. Keeps premise/hypothesis only in memory."""
    if source not in ("snli", "mnli"):
        raise ValueError(f"source must be snli or mnli, got {source!r}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON — {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{lineno}: expected a JSON object")
            if "obs1" in obj or "hyp1" in obj:
                raise ValueError(
                    f"{path}:{lineno}: not a 3-way SNLI/MNLI row "
                    f"(αNLI is excluded). {ALPHANLI_EXCLUSION}"
                )
            if "label_count" not in obj:
                raise ValueError(f"{path}:{lineno}: missing label_count")
            counts = [int(x) for x in obj["label_count"]]
            if len(counts) != 3 or sum(counts) != 100:
                raise ValueError(f"{path}:{lineno}: expected 100 labels over 3 classes")
            example = obj.get("example") or {}
            if "premise" not in example or "hypothesis" not in example:
                raise ValueError(f"{path}:{lineno}: missing premise/hypothesis")
            entropy = shannon_entropy(counts)
            if "entropy" not in obj:
                raise ValueError(f"{path}:{lineno}: missing entropy")
            given = float(obj["entropy"])
            if abs(entropy - given) > 1e-9:
                raise ValueError(
                    f"{path}:{lineno}: recomputed entropy {entropy} != field {given}"
                )
            if "majority_label" not in obj:
                raise ValueError(f"{path}:{lineno}: missing majority_label")
            majority = CODE_TO_LABEL[str(obj["majority_label"])]
            rows.append(
                {
                    "id": str(obj["uid"]),
                    "source": source,
                    "label_count": counts,
                    "label_dist": [c / 100.0 for c in counts],
                    "entropy": entropy,
                    "majority": majority,
                    "premise": str(example["premise"]),
                    "hypothesis": str(example["hypothesis"]),
                }
            )
    return rows


def load_population(snli_path: Path, mnli_path: Path) -> list[dict[str, Any]]:
    snli = load_chaosnli_jsonl(snli_path, "snli")
    mnli = load_chaosnli_jsonl(mnli_path, "mnli")
    if len(snli) != SNLI_N or len(mnli) != MNLI_N:
        raise ValueError(
            f"expected {SNLI_N} SNLI and {MNLI_N} MNLI rows, got {len(snli)} and {len(mnli)}"
        )
    ids = [r["id"] for r in snli + mnli]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate uids across SNLI and MNLI")
    return snli + mnli


def quantile_thresholds(entropy: np.ndarray) -> tuple[float, float]:
    """Easy/hard cutoffs: linear 25th and 75th percentiles of entropy.

    Fixed from the population before any model call. Easy is
    ``entropy <= q25`` (lowest-entropy items). Hard is ``entropy >= q75``
    (highest). The middle half is not in the primary contrast.
    """
    q_easy = float(np.quantile(entropy, 0.25, method="linear"))
    q_hard = float(np.quantile(entropy, 0.75, method="linear"))
    if not q_easy < q_hard:
        raise ValueError(f"degenerate quantiles q25={q_easy} q75={q_hard}")
    return q_easy, q_hard


def assign_stratum(entropy: float, q_easy: float, q_hard: float) -> str:
    if entropy <= q_easy:
        return "easy"
    if entropy >= q_hard:
        return "hard"
    return "mid"


def balanced_analysis_ids(
    records: list[dict[str, Any]],
    *,
    q_easy: float,
    q_hard: float,
    target_n: int = TARGET_N,
    seed: int = SUBSAMPLE_SEED,
) -> dict[str, Any]:
    """Pools from entropy quantiles, then ``subsample_to_equal_n`` to ``target_n``."""
    easy = sorted((r for r in records if r["entropy"] <= q_easy), key=lambda r: r["id"])
    hard = sorted((r for r in records if r["entropy"] >= q_hard), key=lambda r: r["id"])
    if len(easy) < target_n or len(hard) < target_n:
        raise ValueError(
            f"pools easy={len(easy)} hard={len(hard)} cannot supply n={target_n}"
        )
    # Dummy arrays: subsample only needs aligned ids. Order is id-sorted.
    sub = subsample_to_equal_n(
        np.zeros(len(hard)),
        np.zeros(len(hard), dtype=int),
        [r["id"] for r in hard],
        np.zeros(len(easy)),
        np.zeros(len(easy), dtype=int),
        [r["id"] for r in easy],
        seed=seed,
        target_n=target_n,
    )
    # subsample names the first stratum "a" (hard) and the second "b" (easy).
    return {
        "easy_ids": list(sub.item_ids_b),
        "hard_ids": list(sub.item_ids_a),
        "discarded_ids": list(sub.discarded_item_ids),
        "n_easy_pool": len(easy),
        "n_hard_pool": len(hard),
        "target_n": sub.target_n,
        "seed": seed,
    }


def paraphrase_ids(
    easy_ids: list[str],
    hard_ids: list[str],
    *,
    n_per: int = N_PARAPHRASE_PER_STRATUM,
    seed: int = PARAPHRASE_SEED,
) -> dict[str, list[str]]:
    """Stratified paraphrase subset. Ids only — text is written later, still before any run."""
    rng = np.random.default_rng(seed)

    def _pick(ids: list[str]) -> list[str]:
        ordered = sorted(ids)
        if len(ordered) < n_per:
            raise ValueError(f"need {n_per} ids, have {len(ordered)}")
        chosen = rng.choice(len(ordered), size=n_per, replace=False)
        return sorted(ordered[i] for i in chosen)

    return {"easy": _pick(easy_ids), "hard": _pick(hard_ids)}


def public_record(row: dict[str, Any], stratum: str, *, in_analysis: bool) -> dict[str, Any]:
    """Drop premise and hypothesis. Counts and entropy stay (not the sentences)."""
    return {
        "entropy": row["entropy"],
        "id": row["id"],
        "in_analysis": in_analysis,
        "label_count": list(row["label_count"]),
        "label_dist": list(row["label_dist"]),
        "majority_label": row["majority"],
        "source": row["source"],
        "stratum": stratum,
    }


def primary_item(row: dict[str, Any], tier: str) -> dict[str, Any]:
    return {
        "id": row["id"],
        "role": "primary",
        "state": {
            "pair_id": row["id"],
            "source": row["source"],
            "text_status": "fetch_required",
        },
        "tier": tier,
    }


def primary_label(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "label": row["majority"],
        "label_count": list(row["label_count"]),
        "label_dist": list(row["label_dist"]),
        "labeled_at": LABELED_AT,
        "labeler": LABELER,
        "note": (
            "Majority of 100 ChaosNLI annotators. label_dist order is "
            "entailment, neutral, contradiction. Soft scoring uses label_dist; "
            "hard scoring uses label."
        ),
    }


def paraphrase_item(
    row: dict[str, Any], premise: str, hypothesis: str, *, tier: str
) -> dict[str, Any]:
    if premise.strip() == row["premise"].strip() or hypothesis.strip() == row["hypothesis"].strip():
        raise ValueError(f"{row['id']}: paraphrase copies the source sentence")
    if tier not in ("easy", "hard"):
        raise ValueError(f"paraphrase tier must be easy or hard, got {tier!r}")
    return {
        "id": f"{row['id']}::paraphrase",
        "role": "paraphrase",
        "state": {
            "hypothesis": hypothesis,
            "pair_id": row["id"],
            "premise": premise,
            "source": row["source"],
            "source_item_id": row["id"],
            "text_status": "frozen_paraphrase",
            "frozen_before_any_model_run": True,
        },
        "tier": tier,
    }


def build_thresholds(
    records: list[dict[str, Any]],
    balance: dict[str, Any],
    para: dict[str, list[str]],
) -> dict[str, Any]:
    entropy = np.asarray([r["entropy"] for r in records], dtype=float)
    q_easy, q_hard = quantile_thresholds(entropy)
    return {
        "alphanli_excluded": True,
        "alphanli_exclusion": ALPHANLI_EXCLUSION,
        "easy_rule": "entropy <= q_easy (lowest-entropy quartile, ties included)",
        "entropy_definition": (
            "Base-2 Shannon entropy of the 100-annotator count vector "
            "(entailment, neutral, contradiction). 0 log 0 = 0. Recomputed "
            "from label_count; matches the ChaosNLI entropy field."
        ),
        "hard_rule": "entropy >= q_hard (highest-entropy quartile, ties included)",
        "mid_excluded_from_primary": True,
        "n_easy_pool": balance["n_easy_pool"],
        "n_hard_pool": balance["n_hard_pool"],
        "n_paraphrase_per_stratum": N_PARAPHRASE_PER_STRATUM,
        "n_population": len(records),
        "numpy_quantile_method": "linear",
        "paraphrase_ids": para,
        "paraphrase_seed": PARAPHRASE_SEED,
        "q_easy": q_easy,
        "q_hard": q_hard,
        "sources": {"mnli": MNLI_N, "snli": SNLI_N},
        "subsample": "subsample_to_equal_n",
        "subsample_seed": balance["seed"],
        "target_n_per_stratum": balance["target_n"],
        "text_redistributed": False,
    }


def assert_no_source_sentences(rows: list[dict[str, Any]]) -> None:
    """Primary and universe rows must not carry premise/hypothesis text."""
    for row in rows:
        blob = json.dumps(row)
        if '"premise"' in blob or '"hypothesis"' in blob:
            if row.get("role") == "paraphrase":
                continue
            raise ValueError(f"source sentence leaked into committed row {row.get('id')}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_download(path: Path, expected_sha256: str, expected_n: int) -> None:
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise ValueError(
            f"{path} sha256 {digest} != pinned {expected_sha256}. "
            "Refusing to build strata from an unexpected file."
        )
    n = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if n != expected_n:
        raise ValueError(f"{path} has {n} rows, expected {expected_n}")


def cache_text_rows(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Local cache only. Gitignored. Not a redistribution in the repo."""
    return [
        {
            "hypothesis": r["hypothesis"],
            "id": r["id"],
            "premise": r["premise"],
            "source": r["source"],
        }
        for r in sorted(records, key=lambda r: r["id"])
    ]


def hydrate_dataset(dataset: Any, cache_path: Path) -> int:
    """Fill premise/hypothesis on in-memory primary items. Does not touch the hashed file."""
    if not cache_path.is_file():
        return 0
    by_id: dict[str, dict[str, str]] = {}
    with cache_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            by_id[str(obj["id"])] = obj
    filled = 0
    for li in dataset.by_id.values():
        state = li.item.state
        if not isinstance(state, dict):
            continue
        if state.get("text_status") != "fetch_required":
            continue
        hit = by_id.get(li.id)
        if hit is None:
            continue
        state["premise"] = hit["premise"]
        state["hypothesis"] = hit["hypothesis"]
        state["text_status"] = "fetched"
        filled += 1
    return filled


def logged_state(state: Any) -> Any:
    """Drop fetched SNLI/MNLI sentences from run logs. Paraphrases are ours and stay."""
    if isinstance(state, dict) and state.get("text_status") == "fetched":
        return {
            "pair_id": state.get("pair_id"),
            "source": state.get("source"),
            "text_status": "fetched_not_logged",
        }
    return state


def missing_primary_text(dataset: Any) -> list[str]:
    missing: list[str] = []
    for li in dataset.by_id.values():
        if getattr(li.item, "role", "primary") != "primary":
            continue
        state = li.item.state
        if isinstance(state, dict) and state.get("text_status") == "fetch_required":
            missing.append(li.id)
    return missing


def contamination_table(
    rows: list[dict[str, Any]],
    *,
    arms: list[str],
) -> dict[str, Any]:
    """Original vs paraphrase accuracy for every arm.

    Accuracy here is the hard majority match on the frozen ~100-item subset.
    A large drop on the paraphrase, relative to the same items in their
    original wording, is the memorisation signal. SNLI has been public since
    2015 and MNLI since 2018, so a baseline LLM can have seen the sentences.
    """
    by_client: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_client.setdefault(str(row.get("client", "")), []).append(row)

    per_arm: dict[str, Any] = {}
    for arm in arms:
        arm_rows = by_client.get(arm, [])
        original: dict[str, dict[str, Any]] = {}
        paraphrased: dict[str, dict[str, Any]] = {}
        for row in arm_rows:
            if row.get("role") == "paraphrase" or str(row.get("item_id", "")).endswith("::paraphrase"):
                src = str(row.get("source_item_id") or str(row.get("item_id", "")).split("::paraphrase")[0])
                paraphrased[src] = row
            else:
                original[str(row["item_id"])] = row
        common = sorted(set(original) & set(paraphrased))
        if not common:
            per_arm[arm] = {
                "n": 0,
                "original_accuracy": None,
                "paraphrase_accuracy": None,
                "drop": None,
                "status": "unscored",
            }
            continue

        def _hit(row: dict[str, Any]) -> float:
            return 1.0 if str(row.get("choice")) == str(row.get("label")) else 0.0

        orig = float(np.mean([_hit(original[i]) for i in common]))
        para = float(np.mean([_hit(paraphrased[i]) for i in common]))
        per_arm[arm] = {
            "n": len(common),
            "original_accuracy": orig,
            "paraphrase_accuracy": para,
            "drop": orig - para,
            "status": "scored",
        }
    return {
        "schema": "jevbench.contamination.v1",
        "public_since": {"snli": 2015, "mnli": 2018},
        "interpretation": (
            "A large drop from original wording to the frozen paraphrase, "
            "on the same items, suggests the arm memorised SNLI/MNLI surface "
            "forms rather than solving the inference. Report every arm. "
            "This comparison is a contamination control, not the primary ΔECE endpoint."
        ),
        "arms": per_arm,
        "n_paraphrase_frozen": N_PARAPHRASE_PER_STRATUM * 2,
    }


def entropy_ceiling() -> float:
    return float(math.log2(3))
