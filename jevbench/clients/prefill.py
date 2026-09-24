"""Poor man's System One — single-token constrained decoding control (EXP-3).

Motivated by Sean Goedecke's published argument that prefilling the JSON
opening and generating one constrained token recovers most of the speed /
parallelism / probabilities without RLCD.

Protocol
--------
1. Build the prompt: state + question + option list (as sentinel tokens).
2. Prefill the assistant turn with the JSON prefix up to the answer, e.g.
   ``{"choice": "``.
3. Generate EXACTLY ONE token, constrained to the option sentinel set.
4. Softmax the logprobs over that constrained set → probabilities.
5. Map into the standard ``Decision`` dataclass.

TOKENIZER TRAP (methodological — not an implementation detail)
--------------------------------------------------------------
Option strings are usually NOT single tokens, and leading-space variants
tokenize differently. Map each option to a distinct single sentinel
(``"1"``..``"N"``) and translate back.

- ASSERT at startup that every sentinel is exactly one token in THIS
  tokenizer. Fail loudly if not.
- Record the sentinel mapping in the run manifest.
- Order-permutation control: shuffle the sentinel→option assignment across
  repeats (``pass_idx``). If accuracy moves, the model has positional or
  numeric bias — a finding about the CONTROL, not about Jev. Report it.

Latency confound
----------------
Local GPU vs hosted API are NOT comparable. Report wall-clock for
completeness (flagged unfair) and compute-only generation time as the
closest honest analogue. Do NOT use EXP-3 for video latency claims.
"""

from __future__ import annotations

import logging
import math
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from jevbench.clients.base import (
    SENTINEL_DOC_KEY,
    ChoiceQuestion,
    Decision,
    NoulQuestion,
    Question,
    ScoreQuestion,
    SystemOneRequest,
    error_decision,
    question_kind,
)

logger = logging.getLogger(__name__)

# Digit sentinels first, then letters — keeps prompts readable for small N.
_SENTINEL_ALPHABET = "123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Prefill string — Goedecke-style JSON opening up to the answer value.
JSON_CHOICE_PREFIX = '{"choice": "'
JSON_NOUL_PREFIX = '{"answer": "'


class TokenizerLike(Protocol):
    """Minimal tokenizer surface for the single-token assertion."""

    def encode(self, text: str, add_special_tokens: bool = False) -> Sequence[int]:
        ...


class PrefillBackend(Protocol):
    """Minimal backend: return logprobs for a constrained one-token completion."""

    def complete_one_token(
        self,
        *,
        messages: list[dict[str, str]],
        allowed_tokens: list[str],
        model: str,
    ) -> PrefillCompletion:
        ...

    def get_tokenizer(self) -> TokenizerLike | None:
        """Return a tokenizer for the sentinel assertion, or None if unavailable."""
        ...


@dataclass
class PrefillCompletion:
    """Raw backend result for one constrained token."""

    token: str
    # log-probability for each allowed token (missing → treated as -inf)
    logprobs: dict[str, float]
    resolved_model: str
    raw: dict[str, Any] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 1
    # Generation time excluding HTTP/transport — closest honest local analogue
    compute_only_ms: float | None = None


@dataclass
class PrefillClientConfig:
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    backend: str = "transformers"  # "transformers" | "vllm" | "mlx" | "inject"
    base_url: str | None = None  # e.g. http://127.0.0.1:8000/v1 for vLLM
    api_key: str = "EMPTY"
    serving_path: str = "prefill"
    noul_true_label: str = "yes"
    noul_false_label: str = "no"
    temperature: float = 0.0
    device: str = "cpu"
    # Fail startup unless sentinels are verified single-token (or explicitly skipped
    # for injected test backends that cannot provide a tokenizer).
    require_tokenizer_assert: bool = True
    json_choice_prefix: str = JSON_CHOICE_PREFIX
    json_noul_prefix: str = JSON_NOUL_PREFIX
    # Seed mixed with pass_idx for order-permutation control
    permutation_seed: int = 20260922


class TransformersPrefillBackend:
    """Local HuggingFace transformers one-token logprobs (CPU/GPU).

    Used when no NVIDIA GPU / vLLM is available. Logprobs are derived from the
    next-token distribution — not verbalised by the model.
    """

    def __init__(
        self,
        *,
        model_id: str,
        device: str = "cpu",
        tokenizer: TokenizerLike | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self._tokenizer = tokenizer
        self._model: Any = None
        self._tok: Any = None

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None and self._tok is not None:
            return self._tok, self._model
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers+torch required for backend='transformers'. "
                "pip install transformers accelerate"
            ) from exc
        self._tok = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        self._model.to(self.device)
        self._model.eval()
        return self._tok, self._model

    def get_tokenizer(self) -> TokenizerLike | None:
        if self._tokenizer is not None:
            return self._tokenizer
        tok, _ = self._load()
        return tok

    def complete_one_token(
        self,
        *,
        messages: list[dict[str, str]],
        allowed_tokens: list[str],
        model: str,
    ) -> PrefillCompletion:
        import torch

        tok, mdl = self._load()
        # Build a flat prompt; prefer chat template when available.
        if hasattr(tok, "apply_chat_template"):
            # Drop the trailing assistant prefill into the template carefully:
            # append JSON prefix as the start of the assistant message.
            chat = [m for m in messages if m["role"] != "assistant"]
            prefix = next(
                (m["content"] for m in messages if m["role"] == "assistant"), ""
            )
            prompt = tok.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
            prompt = prompt + prefix
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

        inputs = tok(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        t0 = time.perf_counter()
        with torch.no_grad():
            out = mdl(**inputs)
            logits = out.logits[0, -1, :]
        compute_ms = (time.perf_counter() - t0) * 1000.0

        logprobs: dict[str, float] = {}
        for sent in allowed_tokens:
            ids = tok.encode(sent, add_special_tokens=False)
            if len(ids) != 1:
                logprobs[sent] = float("-inf")
                continue
            logprobs[sent] = float(torch.log_softmax(logits, dim=-1)[ids[0]].item())

        # Pick MAP among allowed
        best = max(allowed_tokens, key=lambda s: logprobs.get(s, float("-inf")))
        return PrefillCompletion(
            token=best,
            logprobs=logprobs,
            resolved_model=model or self.model_id,
            raw={"backend": "transformers", "model_id": self.model_id},
            input_tokens=int(inputs["input_ids"].numel()),
            output_tokens=1,
            compute_only_ms=compute_ms,
        )


class SentinelTokenError(RuntimeError):
    """Raised when a sentinel is not exactly one token in the active tokenizer."""


def build_sentinel_mapping(
    options: list[str],
    *,
    pass_idx: int = 0,
    permutation_seed: int = 20260922,
) -> dict[str, str]:
    """Map option label → single sentinel token.

    Always builds a full mapping so the manifest documents the constrained set.
    Across repeats (``pass_idx``), the assignment is shuffled — order-permutation
    control for positional / numeric bias in the open-weight model.
    """
    if len(options) > len(_SENTINEL_ALPHABET):
        raise ValueError(
            f"prefill supports at most {len(_SENTINEL_ALPHABET)} options; "
            f"got {len(options)}"
        )
    sentinels = list(_SENTINEL_ALPHABET[: len(options)])
    if pass_idx != 0 or permutation_seed:
        rng = random.Random(permutation_seed + int(pass_idx) * 1_000_003)
        rng.shuffle(sentinels)
    mapping: dict[str, str] = {}
    for opt, sent in zip(options, sentinels, strict=True):
        mapping[opt] = sent
    return mapping


def invert_mapping(mapping: dict[str, str]) -> dict[str, str]:
    return {v: k for k, v in mapping.items()}


def assert_sentinels_are_single_tokens(
    tokenizer: TokenizerLike,
    mapping: dict[str, str],
    *,
    tokenizer_name: str = "unknown",
) -> dict[str, int]:
    """ASSERT every sentinel is exactly one token. Fail loudly if not.

    Returns ``{sentinel: token_id}`` for the run manifest.
    """
    token_ids: dict[str, int] = {}
    failures: list[str] = []
    for opt, sent in mapping.items():
        ids = list(tokenizer.encode(sent, add_special_tokens=False))
        if len(ids) != 1:
            failures.append(
                f"option={opt!r} sentinel={sent!r} → {len(ids)} tokens {ids}"
            )
        else:
            token_ids[sent] = int(ids[0])
        # Also check common leading-space variant that often splits differently
        spaced = f" {sent}"
        ids_sp = list(tokenizer.encode(spaced, add_special_tokens=False))
        if len(ids_sp) != 1:
            # Leading-space must also be single-token OR we document that we
            # constrain the unspaced form only (JSON prefix ends with `"` so
            # the next token is unspaced). Warn into failures only if unspaced
            # already failed; otherwise record as note via logger.
            logger.debug(
                "leading-space sentinel %r tokenizes to %d ids (unspaced used)",
                sent,
                len(ids_sp),
            )
    if failures:
        raise SentinelTokenError(
            "Prefill sentinel single-token assertion FAILED for tokenizer "
            f"{tokenizer_name!r}. Option strings must map to distinct single "
            "tokens — this is a methodological gate for EXP-3, not a soft "
            "warning.\n  " + "\n  ".join(failures)
        )
    return token_ids


def softmax_logprobs(logprobs: dict[str, float], allowed: list[str]) -> dict[str, float]:
    """Softmax over the constrained set only."""
    vals = []
    for t in allowed:
        lp = logprobs.get(t, float("-inf"))
        vals.append(lp)
    max_lp = max(vals) if vals and math.isfinite(max(vals)) else 0.0
    exps = [math.exp(v - max_lp) if math.isfinite(v) else 0.0 for v in vals]
    total = sum(exps) or 1.0
    return {t: e / total for t, e in zip(allowed, exps, strict=True)}


def _state_to_text(state: str | dict[str, Any] | list[Any]) -> str:
    if isinstance(state, str):
        return state
    import json

    return json.dumps(state, ensure_ascii=False)


def _choice_prompt(
    state_text: str,
    instructions: Any,
    mapping: dict[str, str],
    *,
    json_prefix: str,
) -> list[dict[str, str]]:
    lines = ["Options (the next token must be exactly one sentinel):"]
    # Stable display order by sentinel for readability; mapping already permuted
    for sent, opt in sorted((s, o) for o, s in mapping.items()):
        lines.append(f'  {sent} = {opt}')
    user = (
        f"State:\n{state_text}\n\n"
        f"Question:\n{instructions}\n\n"
        + "\n".join(lines)
        + "\n\nRespond with a JSON object whose \"choice\" value is one sentinel."
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a classifier. Continue the JSON with exactly one "
                "sentinel token, then stop."
            ),
        },
        {"role": "user", "content": user},
        # Prefill assistant turn with JSON opening up to the answer value
        {"role": "assistant", "content": json_prefix},
    ]


class CharTokenizer:
    """Test / fallback tokenizer: each character is one token."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]


class HttpVllmBackend:
    """OpenAI-compatible completions with logprobs (vLLM). Logprobs REQUIRED."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "EMPTY",
        transport: Callable[..., Any] | None = None,
        tokenizer: TokenizerLike | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._transport = transport
        self._tokenizer = tokenizer

    def get_tokenizer(self) -> TokenizerLike | None:
        if self._tokenizer is not None:
            return self._tokenizer
        # Best-effort: try transformers AutoTokenizer from served model name later
        return None

    def set_tokenizer(self, tokenizer: TokenizerLike) -> None:
        self._tokenizer = tokenizer

    def complete_one_token(
        self,
        *,
        messages: list[dict[str, str]],
        allowed_tokens: list[str],
        model: str,
    ) -> PrefillCompletion:
        if self._transport is not None:
            return self._transport(
                messages=messages, allowed_tokens=allowed_tokens, model=model
            )
        import httpx

        prompt_parts: list[str] = []
        for m in messages:
            role, content = m["role"], m["content"]
            prompt_parts.append(f"<|{role}|>\n{content}")
        prompt = "\n".join(prompt_parts)

        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": max(5, len(allowed_tokens)),
            "guided_choice": allowed_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        t_compute0 = time.perf_counter()
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                f"{self.base_url}/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
        # Wall includes transport; vLLM may expose generation time in raw —
        # fall back to full request duration marked as approximate compute.
        compute_only_ms = (time.perf_counter() - t_compute0) * 1000.0
        if isinstance(data.get("timings"), dict) and "generation_ms" in data["timings"]:
            compute_only_ms = float(data["timings"]["generation_ms"])

        choice0 = data["choices"][0]
        text = (choice0.get("text") or "").strip()
        token = text if text in allowed_tokens else (text[:1] if text else allowed_tokens[0])
        logprob_info = choice0.get("logprobs") or {}
        top: dict[str, float] = {}
        top_list = logprob_info.get("top_logprobs") or []
        if top_list and isinstance(top_list[0], dict):
            top = {str(k): float(v) for k, v in top_list[0].items()}
        elif isinstance(logprob_info.get("top_logprobs"), dict):
            top = {str(k): float(v) for k, v in logprob_info["top_logprobs"].items()}
        if not top:
            raise RuntimeError(
                "vLLM response contained no logprobs — EXP-3 requires logprob "
                "access. A hosted API that hides logprobs cannot run this experiment."
            )
        for t in allowed_tokens:
            top.setdefault(t, float("-inf"))
        usage = data.get("usage") or {}
        return PrefillCompletion(
            token=token,
            logprobs=top,
            resolved_model=data.get("model", model),
            raw=data,
            input_tokens=int(usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage.get("completion_tokens", 1) or 1),
            compute_only_ms=compute_only_ms,
        )


class PrefillClient:
    """Goedecke-style constrained one-token control client (EXP-3)."""

    def __init__(
        self,
        config: PrefillClientConfig | None = None,
        *,
        backend: PrefillBackend | None = None,
        tokenizer: TokenizerLike | None = None,
    ) -> None:
        self.config = config or PrefillClientConfig()
        self._backend = backend
        self._tokenizer = tokenizer
        self._last_sentinel_mappings: dict[str, dict[str, str]] = {}
        self._last_token_ids: dict[str, dict[str, int]] = {}
        self._asserted_fingerprints: set[str] = set()
        self._startup_asserted = False

    @property
    def serving_path(self) -> str:
        return self.config.serving_path

    def sentinel_mappings_for_manifest(self) -> dict[str, Any]:
        """Return mapping recorded during the last ``decide`` call for manifests."""
        return {
            SENTINEL_DOC_KEY: dict(self._last_sentinel_mappings),
            "prefill_sentinel_token_ids": dict(self._last_token_ids),
            "prefill_json_choice_prefix": self.config.json_choice_prefix,
            "prefill_order_permutation": True,
            "prefill_latency_note": (
                "wall_clock_ms is NOT a fair comparison vs hosted APIs "
                "(RTT/batching/cold-start differ). Prefer compute_only_ms for "
                "the local model. Do not use EXP-3 for video latency claims."
            ),
        }

    def ensure_tokenizer_ready(self, options_probe: list[str] | None = None) -> None:
        """Run the single-token assertion once at startup / first use."""
        if self._startup_asserted and not self.config.require_tokenizer_assert:
            return
        tok = self._resolve_tokenizer()
        if tok is None:
            if self.config.require_tokenizer_assert:
                raise SentinelTokenError(
                    "PrefillClient requires a tokenizer to assert sentinels are "
                    "single tokens. Pass tokenizer=... or use a backend that "
                    "implements get_tokenizer(). Logprob-capable local vLLM/MLX "
                    "only — hosted APIs that hide logprobs cannot run EXP-3."
                )
            return
        probe = options_probe or ["billing", "technical", "other"]
        mapping = build_sentinel_mapping(
            probe, pass_idx=0, permutation_seed=self.config.permutation_seed
        )
        ids = assert_sentinels_are_single_tokens(
            tok, mapping, tokenizer_name=self.config.model
        )
        self._last_token_ids["_startup_probe"] = ids
        self._startup_asserted = True

    def _resolve_tokenizer(self) -> TokenizerLike | None:
        if self._tokenizer is not None:
            return self._tokenizer
        backend = self._get_backend()
        getter = getattr(backend, "get_tokenizer", None)
        if callable(getter):
            return getter()
        return None

    def _get_backend(self) -> PrefillBackend:
        if self._backend is not None:
            return self._backend
        if self.config.backend == "transformers":
            return TransformersPrefillBackend(
                model_id=self.config.model,
                device=self.config.device,
                tokenizer=self._tokenizer,
            )
        if self.config.backend == "vllm":
            if not self.config.base_url:
                raise ValueError(
                    "PrefillClientConfig.base_url required for vllm backend"
                )
            return HttpVllmBackend(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                tokenizer=self._tokenizer,
            )
        if self.config.backend == "mlx":
            raise NotImplementedError(
                "MLX backend: inject a PrefillBackend that calls mlx_lm with "
                "constrained decoding + logprobs, or use backend='transformers' "
                "on CPU / 'vllm' against a local server. Logprob access is "
                "REQUIRED for EXP-3."
            )
        raise ValueError(
            f"Unknown backend {self.config.backend!r}; pass an injected "
            "PrefillBackend for tests"
        )

    def _options_for(self, question: Question) -> list[str]:
        if isinstance(question, ChoiceQuestion):
            return list(question.criteria.keys())
        if isinstance(question, NoulQuestion):
            return [self.config.noul_true_label, self.config.noul_false_label]
        if isinstance(question, ScoreQuestion):
            return list(question.criteria)
        raise TypeError(type(question))

    def _assert_mapping(self, mapping: dict[str, str], question_key: str) -> None:
        tok = self._resolve_tokenizer()
        if tok is None:
            if self.config.require_tokenizer_assert:
                raise SentinelTokenError(
                    f"Cannot assert single-token sentinels for {question_key!r}: "
                    "no tokenizer available."
                )
            return
        fp = f"{question_key}:{sorted(mapping.items())}"
        if fp in self._asserted_fingerprints:
            return
        ids = assert_sentinels_are_single_tokens(
            tok, mapping, tokenizer_name=self.config.model
        )
        self._last_token_ids[question_key] = ids
        self._asserted_fingerprints.add(fp)

    def decide(self, request: SystemOneRequest) -> list[Decision]:
        backend = self._get_backend()
        state_text = _state_to_text(request.state)
        self._last_sentinel_mappings = {}
        decisions: list[Decision] = []
        pass_idx = int(getattr(request, "pass_idx", 0) or 0)

        for key, question in request.questions.items():
            kind = question_kind(question)
            try:
                options = self._options_for(question)
                mapping = build_sentinel_mapping(
                    options,
                    pass_idx=pass_idx,
                    permutation_seed=self.config.permutation_seed,
                )
                self._assert_mapping(mapping, key)
                self._last_sentinel_mappings[key] = mapping
                inv = invert_mapping(mapping)
                sentinels = list(mapping.values())
                instructions = getattr(question, "instructions", "")
                prefix = (
                    self.config.json_noul_prefix
                    if kind == "noul"
                    else self.config.json_choice_prefix
                )
                messages = _choice_prompt(
                    state_text, instructions, mapping, json_prefix=prefix
                )

                t0 = time.perf_counter()
                completion = backend.complete_one_token(
                    messages=messages,
                    allowed_tokens=sentinels,
                    model=request.model or self.config.model,
                )
                wall_ms = (time.perf_counter() - t0) * 1000.0
                compute_ms = (
                    float(completion.compute_only_ms)
                    if completion.compute_only_ms is not None
                    else wall_ms
                )

                sent_probs = softmax_logprobs(completion.logprobs, sentinels)
                opt_probs = {inv[s]: p for s, p in sent_probs.items()}
                chosen_sent = completion.token.strip()
                if chosen_sent not in inv:
                    chosen_sent = max(sent_probs, key=sent_probs.get)
                chosen_opt = inv[chosen_sent]

                common_raw = {
                    "completion": completion.raw,
                    SENTINEL_DOC_KEY: mapping,
                    "chosen_sentinel": chosen_sent,
                    "pass_idx": pass_idx,
                    "json_prefix": prefix,
                    "latency": {
                        "wall_clock_ms": wall_ms,
                        "compute_only_ms": compute_ms,
                        "fair_comparison": False,
                        "reason": (
                            "Local GPU vs hosted API differ in RTT, batching, "
                            "cold start, and queueing. Not a fair latency comparison."
                        ),
                    },
                }

                if kind == "noul":
                    p_true = opt_probs.get(self.config.noul_true_label, 0.0)
                    decisions.append(
                        Decision(
                            item_id=request.item_id,
                            question_key=key,
                            kind="noul",
                            value=float(p_true),
                            probabilities=None,
                            confidence=None,
                            latency_ms=wall_ms,
                            input_tokens=completion.input_tokens,
                            output_tokens=completion.output_tokens,
                            resolved_model=completion.resolved_model,
                            serving_path=self.serving_path,
                            attempt=1,
                            error=None,
                            raw={**common_raw, "option_probabilities": opt_probs},
                        )
                    )
                elif kind == "choice":
                    conf = max(opt_probs.values()) if opt_probs else None
                    decisions.append(
                        Decision(
                            item_id=request.item_id,
                            question_key=key,
                            kind="choice",
                            value=chosen_opt,
                            probabilities=opt_probs,
                            confidence=conf,
                            latency_ms=wall_ms,
                            input_tokens=completion.input_tokens,
                            output_tokens=completion.output_tokens,
                            resolved_model=completion.resolved_model,
                            serving_path=self.serving_path,
                            attempt=1,
                            error=None,
                            raw={
                                **common_raw,
                                "confidence_note": (
                                    "max_prob sharpness, not Jev confidence"
                                ),
                            },
                        )
                    )
                else:
                    level_probs = [opt_probs.get(lbl, 0.0) for lbl in options]
                    expectation = sum(i * p for i, p in enumerate(level_probs))
                    legend = {str(i): lbl for i, lbl in enumerate(options)}
                    conf = max(opt_probs.values()) if opt_probs else None
                    decisions.append(
                        Decision(
                            item_id=request.item_id,
                            question_key=key,
                            kind="score",
                            value=float(expectation),
                            probabilities={
                                lbl: opt_probs.get(lbl, 0.0) for lbl in options
                            },
                            confidence=conf,
                            latency_ms=wall_ms,
                            input_tokens=completion.input_tokens,
                            output_tokens=completion.output_tokens,
                            resolved_model=completion.resolved_model,
                            serving_path=self.serving_path,
                            attempt=1,
                            error=None,
                            raw={
                                **common_raw,
                                "legend": legend,
                                "score_note": (
                                    "expectation over level indices; "
                                    "no magnitude interpolation between levels"
                                ),
                            },
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                logger.error("PrefillClient failed key=%s err=%s", key, exc)
                decisions.append(
                    error_decision(
                        item_id=request.item_id,
                        question_key=key,
                        kind=kind,
                        latency_ms=0.0,
                        resolved_model=request.model or self.config.model,
                        serving_path=self.serving_path,
                        attempt=1,
                        error=f"{type(exc).__name__}: {exc}",
                        raw={"exception": repr(exc)},
                    )
                )
        return decisions
