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

from adapter_sdk.embedding_base import (
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
        # Initialize base instrumentation and policies
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
        # Assign mock config
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
        # ensure dimensions configured for all supported models
        missing = [m for m in self.supported_models if m not in self.dimensions_by_model]
        if missing:
            raise ValueError(f"Missing dimensions for supported models: {missing}")
        # vector pattern sanity
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
            truncation_mode="base",  # base applies truncation before _do_* hooks
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
        # Deterministic failure injection (ctx or text sentinels)
        self._maybe_fail(op="embed", ctx=ctx, text=spec.text)

        if spec.model not in self.supported_models:
            raise ModelNotAvailable(f"Model '{spec.model}' is not supported")

        await self._sleep_random()

        dim = self._dimensions_for(spec.model)
        rng = self._rng_for(spec.model, spec.text)
        vec = self._make_vector(dim, rng)

        # Source-side normalization if requested & advertised
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
            truncated=False,  # base sets this when truncation occurs
        )

    # ---------------------------------------------------------------------
    # Batch embed
    # ---------------------------------------------------------------------
    async def _do_embed_batch(
        self,
        spec: BatchEmbedSpec,
        *,
        ctx: Optional[EmbeddingContext] = None,
    ) -> BatchEmbedResult:
        # Allow tests to force base fallback path
        if not self.supports_batch:
            raise NotSupported("native batch not supported in this mode")

        # Deterministic failure injection at op level
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
                # Per-item deterministic failure (only if triggered by sentinel/ctx).
                # If collect_failures_in_native_batch=True, we capture and continue.
                self._maybe_fail(op="embed_batch:item", ctx=ctx, text=text, per_item=True)

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
                    # skip embedding for this item
                else:
                    # re-raise to let caller handle; base fallback covers partials when NotSupported
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

            # Cooperative yield every N items
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
        # Small, predictable delay
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
        """
        Deterministic failure injector for conformance-friendly testing.

        Priority:
        1) ctx.attrs["simulate_error"] ∈ {"unavailable","rate_limited","transient"}
        2) Text sentinels: "[UNAVAILABLE]", "[RATE_LIMIT]", "[TRANSIENT]"
        3) Optional randomness (failure_rate > 0.0) for demos only

        Per-item failures are only thrown when explicitly triggered; they can be
        collected in native batch if collect_failures_in_native_batch=True.
        """
        # 1) ctx-directed failures
        key = (ctx and ctx.attrs.get("simulate_error")) or None
        if key:
            if key == "unavailable":
                raise Unavailable(f"Mocked {op} unavailable", retry_after_ms=500)
            if key == "rate_limited":
                raise ResourceExhausted(f"Mocked {op} rate-limited", retry_after_ms=800)
            if key == "transient":
                raise TransientNetwork(f"Mocked {op} transient network", retry_after_ms=600)

        # 2) per-text sentinels (only if text provided)
        if text:
            if "[UNAVAILABLE]" in text:
                raise Unavailable(f"Mocked {op} unavailable (text sentinel)", retry_after_ms=500)
            if "[RATE_LIMIT]" in text:
                raise ResourceExhausted(f"Mocked {op} rate-limited (text sentinel)", retry_after_ms=800)
            if "[TRANSIENT]" in text:
                raise TransientNetwork(f"Mocked {op} transient (text sentinel)", retry_after_ms=600)

        # 3) optional RNG for demos (not used in conformance runs)
        if not per_item and self.failure_rate > 0.0 and random.random() < self.failure_rate:
            if random.random() < 0.5:
                raise ResourceExhausted(f"Mocked {op} rate-limited", retry_after_ms=800)
            raise Unavailable(f"Mocked {op} overloaded", retry_after_ms=500)

    async def _sleep_random(self, bonus_ms: int = 0) -> None:
        lo, hi = self.latency_ms
        dur_ms = float(lo if lo == hi else random.uniform(lo, hi + bonus_ms))
        await asyncio.sleep(dur_ms / 1000.0)

    def _dimensions_for(self, model: str) -> int:
        # explicit check to improve error clarity
        if model not in self.dimensions_by_model:
            raise ModelNotAvailable(f"Model '{model}' not found in dimensions mapping")
        dim = int(self.dimensions_by_model[model])
        if dim <= 0:
            raise BadRequest(f"Invalid dimension ({dim}) for model '{model}'")
        return dim

    def _rng_for(self, model: str, text: str) -> random.Random:
        h = hashlib.sha256(f"{model}|{text}".encode("utf-8")).hexdigest()
        # Use a stable integer seed (lower 16 hex chars)
        seed = int(h[-16:], 16)
        return random.Random(seed)

    def _make_vector(self, dim: int, rng: random.Random) -> List[float]:
        # Optional deterministic patterns for specific tests
        if self.test_vector_pattern == "zeros":
            return [0.0] * dim
        if self.test_vector_pattern == "ones":
            return [1.0] * dim
        if self.test_vector_pattern == "unit_x":
            # 1.0 in the first position, zeros elsewhere
            v = [0.0] * dim
            v[0] = 1.0
            return v
        if self.test_vector_pattern == "gaussian":
            # Box-Muller-ish using rng.random(); mean 0, unit-ish variance
            # (still deterministic from rng)
            return [math.sqrt(-2.0 * math.log(max(1e-12, rng.random()))) *
                    math.cos(2.0 * math.pi * rng.random()) for _ in range(dim)]
        # Default: deterministic uniform in [-1, 1)
        return [rng.random() * 2.0 - 1.0 for _ in range(dim)]

    def _normalize(self, vec: List[float]) -> List[float]:
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def _approx_tokens(self, text: str) -> int:
        # Conformance: empty string returns 0; otherwise rough, monotonic heuristic
        if not text:
            return 0
        # Very rough heuristic: words / factor + small overhead
        words = max(1, len(text.split()))
        return int(math.ceil(words / max(0.1, self.token_factor))) + 2


# ---------------------------------------------------------------------------
# Optional: simple self-demo (kept dependency-free)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    async def _demo() -> None:
        random.seed(7)  # deterministic demo for vector generation
        adapter = MockEmbeddingAdapter(
            mode="standalone",
            supports_batch=True,
            normalizes_at_source=False,
            latency_ms=(5, 5),
            failure_rate=0.0,
            test_vector_pattern=None,
            collect_failures_in_native_batch=True,  # show partials in native batch
        )

        # Capabilities
        caps = await adapter.capabilities()
        print("[CAPABILITIES]", caps)

        # Health (ok)
        health_ok = await adapter.health()
        print("[HEALTH OK]", health_ok)

        # Health (degraded)
        degraded_ctx = EmbeddingContext(attrs={"health": "degraded"})
        health_bad = await adapter.health(ctx=degraded_ctx)
        print("[HEALTH DEGRADED]", health_bad)

        # Single embed (normalize at base)
        spec = EmbedSpec(text="hello world", model="mock-embed-512", truncate=True, normalize=True)
        res = await adapter.embed(spec)
        norm = math.sqrt(sum(x * x for x in res.embedding.vector))
        print("[EMBED] dims=", res.embedding.dimensions, "norm≈", round(norm, 6), "truncated=", res.truncated)

        # Batch embed (native) with per-item failures captured
        sent = ["ok item", "will [RATE_LIMIT] here", "also ok", "and [UNAVAILABLE] here"]
        bspec = BatchEmbedSpec(texts=sent, model="mock-embed-1024", normalize=False)
        bres = await adapter.embed_batch(bspec, ctx=EmbeddingContext(attrs={}))
        print("[BATCH] ok_embeddings=", len(bres.embeddings), "failures=", bres.failed_texts)

        # Base fallback demo
        adapter_fallback = MockEmbeddingAdapter(supports_batch=False, latency_ms=(5, 5))
        bres2 = await adapter_fallback.embed_batch(bspec)
        print("[BATCH FALLBACK] ok_embeddings=", len(bres2.embeddings), "failures=", bres2.failed_texts)

        # Count tokens
        tks0 = await adapter.count_tokens("", "mock-embed-512")
        tks1 = await adapter.count_tokens("token counting example text", "mock-embed-512")
        print("[COUNT_TOKENS] empty=", tks0, "example=", tks1)

    asyncio.run(_demo())
