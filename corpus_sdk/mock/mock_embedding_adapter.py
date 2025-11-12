# examples/embedding/mock_embedding_adapter.py
# SPDX-License-Identifier: Apache-2.0
"""
Mock Embedding adapter used in example scripts and conformance tests.

Deterministic behavior suitable for conformance runs. Supports:
- deterministic retryable failures via ctx attrs or text sentinels
- optional native batch or base fallback
- stable health reporting with explicit degraded trigger
- configuration validation (failure_rate, latency range, dimensions)
- optional vector patterns (zeros/ones/etc) for targeted tests
- optional native-batch partial failure collection
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import random
from typing import Any, Dict, List, Optional, Tuple

from corpus_sdk.embedding.embedding_base import (
    BaseEmbeddingAdapter,
    EmbeddingCapabilities,
    EmbedSpec,
    BatchEmbedSpec,
    EmbedResult,
    BatchEmbedResult,
    EmbeddingVector,
    OperationContext as EmbeddingContext,
    # Canonical error types for correct wire codes
    BadRequest,
    NotSupported,
    Unavailable,
    ResourceExhausted,
    ModelNotAvailable,
    TransientNetwork,
)


class MockEmbeddingAdapter(BaseEmbeddingAdapter):
    """
    A mock Embedding adapter for protocol demonstrations & conformance.
    Defaults are deterministic and non-flaky. Dials can be toggled for demos.

    NOTE: This adapter overrides `embed_batch()` to ensure per-item validation
    happens in the **fallback loop** (not preflight), so empty texts become
    item-level failures when batch is NotSupported — matching the conformance
    tests that exercise partial-failure semantics on fallback.
    """

    # ----- Tunables (safe defaults for conformance) -----
    name: str
    supported_models: Tuple[str, ...]
    dimensions_by_model: Dict[str, int]
    max_text_length: int
    max_batch_size: int
    normalizes_at_source: bool
    supports_batch: bool
    token_factor: float
    latency_ms: Tuple[int, int]
    failure_rate: float  # kept for demos; default 0.0 for conformance
    test_vector_pattern: Optional[str]  # None | "zeros" | "ones" | "unit_x" | "gaussian"
    collect_failures_in_native_batch: bool  # opt-in to report per-item failures in native batch

    def __init__(
        self,
        *,
        # Base adapter mode/infra
        mode: str = "thin",
        metrics=None,
        deadline_policy=None,
        truncation=None,
        normalization=None,
        breaker=None,
        cache=None,
        limiter=None,
        tag_model_in_metrics: Optional[bool] = None,
        cache_embed_ttl_s: int = 60,
        cache_caps_ttl_s: int = 30,
        # Mock-specific knobs
        name: str = "mock-embedding",
        supported_models: Tuple[str, ...] = ("mock-embed-512", "mock-embed-1024"),
        dimensions_by_model: Optional[Dict[str, int]] = None,
        max_text_length: int = 4000,
        max_batch_size: int = 128,
        normalizes_at_source: bool = False,
        supports_batch: bool = True,
        token_factor: float = 0.75,
        latency_ms: Tuple[int, int] = (10, 25),
        failure_rate: float = 0.0,  # 0.0 to avoid test flakes
        test_vector_pattern: Optional[str] = None,
        collect_failures_in_native_batch: bool = False,
    ) -> None:
        super().__init__(
            mode=mode,
            metrics=metrics,
            deadline_policy=deadline_policy,
            truncation=truncation,
            normalization=normalization,
            breaker=breaker,
            cache=cache,
            limiter=limiter,
            tag_model_in_metrics=tag_model_in_metrics,
            cache_embed_ttl_s=cache_embed_ttl_s,
            cache_caps_ttl_s=cache_caps_ttl_s,
        )
        self.name = name
        self.supported_models = tuple(supported_models)
        self.dimensions_by_model = dict(
            dimensions_by_model or {"mock-embed-512": 512, "mock-embed-1024": 1024}
        )
        self.max_text_length = int(max_text_length)
        self.max_batch_size = int(max_batch_size)
        self.normalizes_at_source = bool(normalizes_at_source)
        self.supports_batch = bool(supports_batch)
        self.token_factor = float(token_factor)
        self.latency_ms = (int(latency_ms[0]), int(latency_ms[1]))
        self.failure_rate = float(failure_rate)
        self.test_vector_pattern = test_vector_pattern
        self.collect_failures_in_native_batch = bool(collect_failures_in_native_batch)

        # -----------------------------
        # Configuration validation
        # -----------------------------
        if not self.supported_models:
            raise ValueError("supported_models must be a non-empty tuple")
        if self.failure_rate < 0.0 or self.failure_rate > 1.0:
            raise ValueError("failure_rate must be between 0 and 1")
        if self.latency_ms[0] < 0 or self.latency_ms[1] < self.latency_ms[0]:
            raise ValueError("Invalid latency range (min >= 0 and max >= min)")
        if any(dim <= 0 for dim in self.dimensions_by_model.values()):
            raise ValueError("All dimensions must be positive")
        missing = [m for m in self.supported_models if m not in self.dimensions_by_model]
        if missing:
            raise ValueError(f"Missing dimensions for supported models: {missing}")
        if self.test_vector_pattern not in (None, "zeros", "ones", "unit_x", "gaussian"):
            raise ValueError("test_vector_pattern must be one of None|zeros|ones|unit_x|gaussian")

    # ---------------------------------------------------------------------
    # Capabilities & Health
    # ---------------------------------------------------------------------
    async def _do_capabilities(self) -> EmbeddingCapabilities:
        return EmbeddingCapabilities(
            server=self.name,
            version="1.0.0",
            supported_models=self.supported_models,
            max_batch_size=self.max_batch_size,
            max_text_length=self.max_text_length,
            max_dimensions=max(self.dimensions_by_model.values()),
            supports_normalization=True,
            supports_truncation=True,
            supports_token_counting=True,
            idempotent_operations=False,
            supports_multi_tenant=True,
            normalizes_at_source=self.normalizes_at_source,
            truncation_mode="base",
            supports_deadline=True,
        )

    async def _do_health(self, *, ctx: Optional[EmbeddingContext] = None) -> Dict[str, Any]:
        degraded = bool((ctx and ctx.attrs.get("health") == "degraded"))
        if degraded:
            return {
                "ok": False,
                "server": self.name,
                "version": "1.0.0",
                "models": {m: "degraded" for m in self.supported_models},
            }
        return {
            "ok": True,
            "server": self.name,
            "version": "1.0.0",
            "models": {m: "ok" for m in self.supported_models},
        }

    # ---------------------------------------------------------------------
    # Single embed
    # ---------------------------------------------------------------------
    async def _do_embed(
        self,
        spec: EmbedSpec,
        *,
        ctx: Optional[EmbeddingContext] = None,
    ) -> EmbedResult:
        self._maybe_fail(op="embed", ctx=ctx, text=spec.text)

        if spec.model not in self.supported_models:
            raise ModelNotAvailable(f"Model '{spec.model}' is not supported")

        await self._sleep_random()

        dim = self._dimensions_for(spec.model)
        rng = self._rng_for(spec.model, spec.text)
        vec = self._make_vector(dim, rng)

        if self.normalizes_at_source and spec.normalize:
            vec = self._normalize(vec)

        tokens = self._approx_tokens(spec.text)

        return EmbedResult(
            embedding=EmbeddingVector(
                vector=vec,
                text=spec.text,
                model=spec.model,
                dimensions=len(vec),
            ),
            model=spec.model,
            text=spec.text,
            tokens_used=tokens,
            truncated=False,
        )

    # ---------------------------------------------------------------------
    # Batch embed — public override to ensure fallback-per-item validation
    # ---------------------------------------------------------------------
    async def embed_batch(
        self,
        spec: BatchEmbedSpec,
        *,
        ctx: Optional[EmbeddingContext] = None,
    ) -> BatchEmbedResult:
        """
        Override the base to avoid pre-validating per-item non-emptiness before
        attempting native batch, so that when native batch is NotSupported the
        fallback can report **per-item** failures (e.g., empty string) instead
        of failing the entire request up front — matching conformance tests.
        """
        # Reuse base gating/metrics wrapper
        async def _run() -> BatchEmbedResult:
            # Core request validation (model & batch size), but **no per-item empty check here**
            caps = await self._do_capabilities()
            if spec.model not in caps.supported_models:
                raise ModelNotAvailable(f"Model '{spec.model}' is not supported")

            if caps.max_batch_size and len(spec.texts) > caps.max_batch_size:
                raise BadRequest(
                    f"Batch size {len(spec.texts)} exceeds maximum of {caps.max_batch_size}",
                    details={"max_batch_size": caps.max_batch_size},
                )

            # Deterministic truncation per item; allow empty strings to pass through unchanged.
            eff_texts: List[str] = []
            for text in spec.texts:
                if caps.max_text_length:
                    new_text, _ = self._trunc.apply(
                        text,
                        caps.max_text_length,
                        spec.truncate,
                    )
                    eff_texts.append(new_text)
                else:
                    eff_texts.append(text)

            eff_spec = BatchEmbedSpec(
                texts=eff_texts,
                model=spec.model,
                truncate=spec.truncate,
                normalize=spec.normalize,
            )

            try:
                # Primary: native batch path (may collect failures itself if configured)
                if not self.supports_batch:
                    raise NotSupported("native batch not supported in this mode")
                result = await self._do_embed_batch(eff_spec, ctx=ctx)
            except NotSupported:
                # Fallback: per-item path with **per-item validation** & failure capture
                embeddings: List[EmbeddingVector] = []
                failed: List[Dict[str, Any]] = []

                for idx, text in enumerate(eff_spec.texts):
                    try:
                        # Validate per item here so empty strings become item failures.
                        if not isinstance(text, str) or not text.strip():
                            raise BadRequest("text must be a non-empty string")

                        single_spec = EmbedSpec(
                            text=text,
                            model=eff_spec.model,
                            truncate=eff_spec.truncate,
                            normalize=False,  # normalize uniformly below
                        )
                        single = await self._do_embed(single_spec, ctx=ctx)
                        ev = single.embedding

                        if eff_spec.normalize:
                            if not caps.supports_normalization:
                                raise NotSupported("normalization not supported for this adapter")
                            if not caps.normalizes_at_source:
                                vec = self._norm.normalize(ev.vector)
                                ev = EmbeddingVector(
                                    vector=vec,
                                    text=ev.text,
                                    model=ev.model,
                                    dimensions=len(vec),
                                )

                        embeddings.append(ev)
                    except (BadRequest, ModelNotAvailable, NotSupported,
                            Unavailable, ResourceExhausted, TransientNetwork) as item_err:
                        failed.append(
                            {
                                "index": idx,
                                "text": text,
                                "error": type(item_err).__name__,
                                "code": getattr(item_err, "code", None)
                                        or type(item_err).__name__.upper(),
                                "message": getattr(item_err, "message", None) or str(item_err) or "",
                            }
                        )
                    except Exception as item_err:
                        failed.append(
                            {
                                "index": idx,
                                "text": text,
                                "error": type(item_err).__name__,
                                "code": "UNAVAILABLE",
                                "message": str(item_err) or "internal error",
                            }
                        )

                result = BatchEmbedResult(
                    embeddings=embeddings,
                    model=eff_spec.model,
                    total_texts=len(spec.texts),
                    total_tokens=None,
                    failed_texts=failed,
                )

            # Post-processing: normalization for native-batch result if requested
            if eff_spec.normalize and not self.normalizes_at_source:
                if not caps.supports_normalization:
                    raise NotSupported("normalization not supported for this adapter")
                for i, ev in enumerate(result.embeddings):
                    vec = self._norm.normalize(ev.vector)
                    result.embeddings[i] = EmbeddingVector(
                        vector=vec,
                        text=ev.text,
                        model=ev.model,
                        dimensions=len(vec),
                    )

            # Minimal counters (mirror base behavior)
            self._metrics.counter(
                component=self._component,
                name="texts_embedded",
                value=len(result.embeddings),
            )
            if result.total_tokens is not None:
                self._metrics.counter(
                    component=self._component,
                    name="tokens_processed",
                    value=int(result.total_tokens),
                )

            return result

        metric_extra: Dict[str, Any] = {"batch_size": len(spec.texts)}
        error_extra: Dict[str, Any] = {"batch_size": len(spec.texts)}
        if getattr(self, "_tag_model_in_metrics", False):
            model_tag = self._safe_model_tag(spec.model)
            if model_tag:
                metric_extra["model"] = model_tag
                error_extra["model"] = model_tag

        return await self._with_gates_unary(
            op="embed_batch",
            ctx=ctx,
            call_coro=_run(),
            metric_extra=metric_extra,
            error_extra=error_extra,
        )

    # ---------------------------------------------------------------------
    # Native batch hook (used when supports_batch=True)
    # ---------------------------------------------------------------------
    async def _do_embed_batch(
        self,
        spec: BatchEmbedSpec,
        *,
        ctx: Optional[EmbeddingContext] = None,
    ) -> BatchEmbedResult:
        # Allow tests to force base fallback path via `supports_batch=False`
        self._maybe_fail(op="embed_batch", ctx=ctx)

        if spec.model not in self.supported_models:
            raise ModelNotAvailable(f"Model '{spec.model}' is not supported")

        if self.max_batch_size and len(spec.texts) > self.max_batch_size:
            raise BadRequest(
                f"Batch size {len(spec.texts)} exceeds maximum of {self.max_batch_size}"
            )

        await self._sleep_random(bonus_ms=10)

        dim = self._dimensions_for(spec.model)
        embeddings: List[EmbeddingVector] = []
        total_tokens = 0
        failures: List[Dict[str, Any]] = []

        for i, text in enumerate(spec.texts):
            try:
                self._maybe_fail(op="embed_batch:item", ctx=ctx, text=text, per_item=True)

                # If collecting failures natively, treat empty text as per-item BadRequest here too.
                if self.collect_failures_in_native_batch and (not isinstance(text, str) or not text.strip()):
                    raise BadRequest("text must be a non-empty string")

                rng = self._rng_for(spec.model, text)
                vec = self._make_vector(dim, rng)
                if self.normalizes_at_source and spec.normalize:
                    vec = self._normalize(vec)

                embeddings.append(
                    EmbeddingVector(
                        vector=vec,
                        text=text,
                        model=spec.model,
                        dimensions=len(vec),
                    )
                )
                total_tokens += self._approx_tokens(text)
            except (Unavailable, ResourceExhausted, TransientNetwork, BadRequest, ModelNotAvailable) as item_err:
                if self.collect_failures_in_native_batch:
                    failures.append(
                        {
                            "index": i,
                            "error": type(item_err).__name__,
                            "code": getattr(item_err, "code", None) or type(item_err).__name__.upper(),
                            "message": str(item_err) or "",
                        }
                    )
                else:
                    raise
            except Exception as item_err:
                if self.collect_failures_in_native_batch:
                    failures.append(
                        {
                            "index": i,
                            "error": type(item_err).__name__,
                            "code": "UNAVAILABLE",
                            "message": str(item_err) or "internal error",
                        }
                    )
                else:
                    raise

            if (i + 1) % 50 == 0:
                await asyncio.sleep(0)

        return BatchEmbedResult(
            embeddings=embeddings,
            model=spec.model,
            total_texts=len(spec.texts),
            total_tokens=total_tokens,
            failed_texts=failures if self.collect_failures_in_native_batch else [],
        )

    # ---------------------------------------------------------------------
    # Token counting
    # ---------------------------------------------------------------------
    async def _do_count_tokens(
        self,
        text: str,
        model: str,
        *,
        ctx: Optional[EmbeddingContext] = None,
    ) -> int:
        if model not in self.supported_models:
            raise ModelNotAvailable(f"Model '{model}' is not supported")
        await asyncio.sleep(0.005)
        return self._approx_tokens(text)

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------
    def _maybe_fail(
        self,
        *,
        op: str,
        ctx: Optional[EmbeddingContext] = None,
        text: Optional[str] = None,
        per_item: bool = False,
    ) -> None:
        key = (ctx and ctx.attrs.get("simulate_error")) or None
        if key:
            if key == "unavailable":
                raise Unavailable(f"Mocked {op} unavailable", retry_after_ms=500)
            if key == "rate_limited":
                raise ResourceExhausted(f"Mocked {op} rate-limited", retry_after_ms=800)
            if key == "transient":
                raise TransientNetwork(f"Mocked {op} transient network", retry_after_ms=600)

        if text:
            if "[UNAVAILABLE]" in text:
                raise Unavailable(f"Mocked {op} unavailable (text sentinel)", retry_after_ms=500)
            if "[RATE_LIMIT]" in text:
                raise ResourceExhausted(f"Mocked {op} rate-limited (text sentinel)", retry_after_ms=800)
            if "[TRANSIENT]" in text:
                raise TransientNetwork(f"Mocked {op} transient (text sentinel)", retry_after_ms=600)

        if not per_item and self.failure_rate > 0.0 and random.random() < self.failure_rate:
            if random.random() < 0.5:
                raise ResourceExhausted(f"Mocked {op} rate-limited", retry_after_ms=800)
            raise Unavailable(f"Mocked {op} overloaded", retry_after_ms=500)

    async def _sleep_random(self, bonus_ms: int = 0) -> None:
        lo, hi = self.latency_ms
        dur_ms = float(lo if lo == hi else random.uniform(lo, hi + bonus_ms))
        await asyncio.sleep(dur_ms / 1000.0)

    def _dimensions_for(self, model: str) -> int:
        if model not in self.dimensions_by_model:
            raise ModelNotAvailable(f"Model '{model}' not found in dimensions mapping")
        dim = int(self.dimensions_by_model[model])
        if dim <= 0:
            raise BadRequest(f"Invalid dimension ({dim}) for model '{model}'")
        return dim

    def _rng_for(self, model: str, text: str) -> random.Random:
        h = hashlib.sha256(f"{model}|{text}".encode("utf-8")).hexdigest()
        seed = int(h[-16:], 16)
        return random.Random(seed)

    def _make_vector(self, dim: int, rng: random.Random) -> List[float]:
        if self.test_vector_pattern == "zeros":
            return [0.0] * dim
        if self.test_vector_pattern == "ones":
            return [1.0] * dim
        if self.test_vector_pattern == "unit_x":
            v = [0.0] * dim
            v[0] = 1.0
            return v
        if self.test_vector_pattern == "gaussian":
            return [
                math.sqrt(-2.0 * math.log(max(1e-12, rng.random()))) * math.cos(2.0 * math.pi * rng.random())
                for _ in range(dim)
            ]
        return [rng.random() * 2.0 - 1.0 for _ in range(dim)]

    def _normalize(self, vec: List[float]) -> List[float]:
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def _approx_tokens(self, text: str) -> int:
        # Empty returns 0 per conformance; otherwise rough, monotonic heuristic
        if not text:
            return 0
        words = max(1, len(text.split()))
        return int(math.ceil(words / max(0.1, self.token_factor))) + 2
