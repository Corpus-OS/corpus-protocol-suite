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
- streaming with configurable chunking patterns
- cache-aware testing with configurable hit/miss behavior
- enhanced metrics and statistics collection

Conformance Guarantees:
- Deterministic by default (seed-based RNG for vectors)
- Cache behavior is configurable but defaults to deterministic
- Streaming counters are accurate and never leak
- All capabilities accurately reflect actual behavior
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple, AsyncIterator

from corpus_sdk.embedding.embedding_base import (
    BaseEmbeddingAdapter,
    EmbeddingCapabilities,
    EmbedSpec,
    BatchEmbedSpec,
    EmbedResult,
    BatchEmbedResult,
    EmbeddingVector,
    EmbedChunk,
    EmbeddingStats,
    OperationContext as EmbeddingContext,
    # Canonical error types for correct wire codes
    BadRequest,
    NotSupported,
    Unavailable,
    ResourceExhausted,
    ModelNotAvailable,
    TransientNetwork,
    DeadlineExceeded,
)


class MockEmbeddingAdapter(BaseEmbeddingAdapter):
    """
    A mock Embedding adapter for protocol demonstrations & conformance.
    Defaults are deterministic and non-flaky. Dials can be toggled for demos.

    Streaming Support:
    - Configurable chunk patterns: "single" (one chunk), "progressive" (vector built piecewise),
      "multi_vector" (multiple complete vectors per chunk)
    - Supports mid-stream abandonment testing
    - Configurable normalization during streaming

    Cache Testing:
    - When caching is enabled (mode=standalone with cache), uses actual cache (BaseEmbeddingAdapter)
    - Optional deterministic simulated cache behavior for thin-mode testing (when enabled via knobs)
    - Deterministic by default (no simulated cache unless configured)

    Statistics:
    - Tracks detailed operation counts, latency, and streaming metrics
    - Configurable to simulate various backend behaviors
    """

    # ----- Tunables (safe defaults for conformance) -----
    name: str
    supported_models: Tuple[str, ...]
    dimensions_by_model: Dict[str, int]
    max_text_length: int
    max_batch_size: int
    normalizes_at_source: bool
    supports_batch: bool
    supports_streaming: bool
    token_factor: float
    latency_ms: Tuple[int, int]
    failure_rate: float  # kept for demos; default 0.0 for conformance
    test_vector_pattern: Optional[str]  # None | "zeros" | "ones" | "unit_x" | "gaussian"
    collect_failures_in_native_batch: bool  # opt-in to report per-item failures in native batch
    yield_every_n: int  # how often to yield in large batches (0 = no yielding)
    simulate_latency: bool  # if False, skips all artificial sleeps

    # ----- Streaming Configuration -----
    stream_chunk_pattern: str  # "single" | "progressive" | "multi_vector"
    stream_min_chunks: int
    stream_max_chunks: int
    stream_yield_interval_ms: Tuple[int, int]  # between chunks
    stream_abandonment_rate: float  # 0.0-1.0, probability of mid-stream cancellation simulation

    # ----- Cache Testing Configuration -----
    cache_behavior: str  # "deterministic" | "force_hit" | "force_miss" | "simulate_error"
    cache_hit_rate: float  # 0.0-1.0, probability of cache hit when behavior="deterministic"
    cache_set_failure_rate: float  # 0.0-1.0, probability of cache.set() failure

    # ----- Internal Statistics -----
    _stats: Dict[str, Any]
    _operation_counts: Dict[str, int]
    _streaming_metrics: Dict[str, Any]
    _stream_active_count: int  # Separate counter for reliable tracking

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
        supports_streaming: bool = True,
        token_factor: float = 0.75,
        latency_ms: Tuple[int, int] = (10, 25),
        failure_rate: float = 0.0,  # 0.0 to avoid test flakes
        test_vector_pattern: Optional[str] = None,
        collect_failures_in_native_batch: bool = False,
        yield_every_n: int = 50,
        simulate_latency: bool = True,
        # Streaming configuration
        stream_chunk_pattern: str = "single",
        stream_min_chunks: int = 1,
        stream_max_chunks: int = 3,
        stream_yield_interval_ms: Tuple[int, int] = (5, 15),
        stream_abandonment_rate: float = 0.0,
        # Cache testing configuration
        cache_behavior: str = "deterministic",
        cache_hit_rate: float = 0.0,  # Default 0.0 for deterministic no-cache behavior
        cache_set_failure_rate: float = 0.0,
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
        self.supports_streaming = bool(supports_streaming)
        self.token_factor = float(token_factor)
        self.latency_ms = (int(latency_ms[0]), int(latency_ms[1]))
        self.failure_rate = float(failure_rate)
        self.test_vector_pattern = test_vector_pattern
        self.collect_failures_in_native_batch = bool(collect_failures_in_native_batch)
        self.yield_every_n = int(yield_every_n)
        self.simulate_latency = bool(simulate_latency)

        # Streaming config
        self.stream_chunk_pattern = stream_chunk_pattern
        self.stream_min_chunks = max(1, int(stream_min_chunks))
        self.stream_max_chunks = max(self.stream_min_chunks, int(stream_max_chunks))
        # FIXED: Use local variables to compute stream_yield_interval_ms
        lo = max(0, int(stream_yield_interval_ms[0]))
        hi = max(lo, int(stream_yield_interval_ms[1]))
        self.stream_yield_interval_ms = (lo, hi)
        self.stream_abandonment_rate = max(0.0, min(1.0, float(stream_abandonment_rate)))

        # Cache testing config - defaults to deterministic no-cache for conformance
        self.cache_behavior = cache_behavior
        self.cache_hit_rate = max(0.0, min(1.0, float(cache_hit_rate)))
        self.cache_set_failure_rate = max(0.0, min(1.0, float(cache_set_failure_rate)))

        # Initialize statistics
        self._stats = {
            "embed_calls": 0,
            "embed_batch_calls": 0,
            "stream_embed_calls": 0,
            "count_tokens_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_set_attempts": 0,
            "cache_set_failures": 0,
            "total_texts_embedded": 0,
            "total_tokens_processed": 0,
            "total_stream_chunks": 0,
            "abandoned_streams": 0,
            "completed_streams": 0,
            "total_processing_time_ms": 0.0,
        }

        self._operation_counts = {}
        self._streaming_metrics = {
            "total_chunks_generated": 0,
            "chunks_by_pattern": {"single": 0, "progressive": 0, "multi_vector": 0},
        }
        self._stream_active_count = 0

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
        if self.yield_every_n < 0:
            raise ValueError("yield_every_n must be >= 0")
        if self.stream_chunk_pattern not in ("single", "progressive", "multi_vector"):
            raise ValueError("stream_chunk_pattern must be one of: single, progressive, multi_vector")
        if self.cache_behavior not in ("deterministic", "force_hit", "force_miss", "simulate_error"):
            raise ValueError("cache_behavior must be one of: deterministic, force_hit, force_miss, simulate_error")

    # ---------------------------------------------------------------------
    # Capability helpers
    # ---------------------------------------------------------------------
    def _real_cache_enabled(self) -> bool:
        """True when BaseEmbeddingAdapter's actual cache path is enabled and meaningful."""
        return bool(
            self._mode == "standalone"
            and getattr(self._cache, "supports_ttl", False)
            and self._cache_embed_ttl_s > 0
        )

    def _simulated_cache_enabled(self) -> bool:
        """
        True when we are intentionally simulating caching in thin mode.
        (Used only for tests/demos; must align with capabilities.)
        """
        if self._mode != "thin":
            return False
        if self.cache_behavior != "deterministic":
            return True
        # Deterministic behavior can still simulate hits/misses if rates are non-zero.
        return (self.cache_hit_rate > 0.0) or (self.cache_set_failure_rate > 0.0)

    def _supports_caching(self) -> bool:
        """Capability truth: real cache enabled OR simulated cache enabled (thin-mode test harness)."""
        return self._real_cache_enabled() or self._simulated_cache_enabled()

    # ---------------------------------------------------------------------
    # Capabilities & Health
    # ---------------------------------------------------------------------
    async def _do_capabilities(self) -> EmbeddingCapabilities:
        """Return accurate capabilities reflecting actual mock behavior."""
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
            supports_streaming=self.supports_streaming,
            supports_batch_embedding=self.supports_batch,
            supports_caching=self._supports_caching(),
            idempotent_writes=False,
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
                "mock_stats": self._get_current_stats(),
            }
        return {
            "ok": True,
            "server": self.name,
            "version": "1.0.0",
            "models": {m: "ok" for m in self.supported_models},
            "mock_stats": self._get_current_stats(),
        }

    # ---------------------------------------------------------------------
    # Single embed
    #   - Real caching (standalone) is handled by BaseEmbeddingAdapter outside _do_embed.
    #   - Simulated caching (thin mode) is only enabled when capabilities say supports_caching=True.
    # ---------------------------------------------------------------------
    async def _do_embed(
        self,
        spec: EmbedSpec,
        *,
        ctx: Optional[EmbeddingContext] = None,
    ) -> EmbedResult:
        """Single embed with OPTIONAL simulated cache behavior for thin-mode tests."""
        self._stats["embed_calls"] += 1
        start_time = time.monotonic()

        self._maybe_fail(op="embed", ctx=ctx, text=spec.text)

        if spec.model not in self.supported_models:
            raise ModelNotAvailable(f"Model '{spec.model}' is not supported")

        # --- Simulated caching only when thin-mode simulation is enabled (and capability says caching supported) ---
        if self._simulated_cache_enabled():
            # Cache key generation sanity (not used further; useful for tests to validate tenant isolation)
            _ = self._embed_cache_key(spec.model, spec.normalize, spec.text, ctx)

            should_hit = self._should_cache_hit("embed", spec.text, ctx)
            if should_hit:
                self._stats["cache_hits"] += 1
                cached_vec = self._make_cached_vector(spec.model, spec.text)
                if self.normalizes_at_source and spec.normalize:
                    cached_vec = self._normalize(cached_vec)

                elapsed_ms = (time.monotonic() - start_time) * 1000
                self._stats["total_processing_time_ms"] += elapsed_ms

                return EmbedResult(
                    embedding=EmbeddingVector(
                        vector=cached_vec,
                        text=spec.text,
                        model=spec.model,
                        dimensions=len(cached_vec),
                    ),
                    model=spec.model,
                    text=spec.text,
                    tokens_used=self._approx_tokens(spec.text),
                    truncated=False,
                )

            # Miss: simulate a cache set attempt (and possibly a failure)
            self._stats["cache_misses"] += 1
            self._stats["cache_set_attempts"] += 1  # FIXED: attempts are counted deterministically
            if self._should_cache_set_fail():
                self._stats["cache_set_failures"] += 1
                if self.cache_behavior == "simulate_error":
                    raise Unavailable("Cache set failed (simulated for testing)")

        # --- Normal computation path (deterministic vectors) ---
        await self._sleep_random()

        dim = self._dimensions_for(spec.model)
        rng = self._rng_for(spec.model, spec.text)
        vec = self._make_vector(dim, rng)

        if self.normalizes_at_source and spec.normalize:
            vec = self._normalize(vec)

        tokens = self._approx_tokens(spec.text)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        self._stats["total_processing_time_ms"] += elapsed_ms
        self._stats["total_texts_embedded"] += 1
        self._stats["total_tokens_processed"] += tokens

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
    # Streaming Implementation with reliable counter tracking
    # ---------------------------------------------------------------------
    async def _do_stream_embed(
        self,
        spec: EmbedSpec,
        *,
        ctx: Optional[EmbeddingContext] = None,
    ) -> AsyncIterator[EmbedChunk]:
        """Implement streaming text embedding with configurable chunk patterns."""
        self._stats["stream_embed_calls"] += 1
        self._stream_active_count += 1

        start_time = time.monotonic()
        stream_id = f"stream_{int(time.time() * 1000)}_{hash(spec.text) % 10000:04d}"

        try:
            self._maybe_fail(op="stream_embed", ctx=ctx, text=spec.text)

            if spec.model not in self.supported_models:
                raise ModelNotAvailable(f"Model '{spec.model}' is not supported")

            dim = self._dimensions_for(spec.model)
            rng = self._rng_for(spec.model, spec.text)

            # Determine chunking pattern
            if self.stream_chunk_pattern == "single":
                chunks = await self._generate_single_chunk(spec, dim, rng, ctx)
            elif self.stream_chunk_pattern == "progressive":
                chunks = await self._generate_progressive_chunks(spec, dim, rng, ctx)
            else:  # multi_vector
                chunks = await self._generate_multi_vector_chunks(spec, dim, rng, ctx)

            total_chunks = len(chunks)

            for chunk_idx, chunk_data in enumerate(chunks):
                # Check for simulated abandonment
                if (
                    self.stream_abandonment_rate > 0
                    and random.random() < self.stream_abandonment_rate
                    and chunk_idx < total_chunks - 1
                ):
                    self._stats["abandoned_streams"] += 1
                    raise asyncio.CancelledError(f"Stream {stream_id} abandoned (simulated)")

                # Simulate inter-chunk latency
                if self.simulate_latency and chunk_idx > 0:
                    await self._sleep_random_range(*self.stream_yield_interval_ms)

                is_final = (chunk_idx == total_chunks - 1)
                chunk = EmbedChunk(
                    embeddings=chunk_data["embeddings"],
                    is_final=is_final,
                    usage=chunk_data.get("usage"),
                    model=spec.model,
                )

                self._stats["total_stream_chunks"] += 1
                self._streaming_metrics["total_chunks_generated"] += 1
                self._streaming_metrics["chunks_by_pattern"][self.stream_chunk_pattern] += 1

                yield chunk

            self._stats["completed_streams"] += 1
            self._stats["total_texts_embedded"] += 1
            self._stats["total_tokens_processed"] += self._approx_tokens(spec.text)

            elapsed_ms = (time.monotonic() - start_time) * 1000
            self._stats["total_processing_time_ms"] += elapsed_ms

        finally:
            # CHANGE #3: Decrement exactly once and never go negative (robust even if closed early)
            self._stream_active_count = max(0, self._stream_active_count - 1)

    async def _generate_single_chunk(
        self,
        spec: EmbedSpec,
        dim: int,
        rng: random.Random,
        ctx: Optional[EmbeddingContext],
    ) -> List[Dict[str, Any]]:
        await self._sleep_random()

        vec = self._make_vector(dim, rng)
        if self.normalizes_at_source and spec.normalize:
            vec = self._normalize(vec)

        embedding = EmbeddingVector(
            vector=vec,
            text=spec.text,
            model=spec.model,
            dimensions=dim,
        )

        return [{
            "embeddings": [embedding],
            "usage": {"tokens": self._approx_tokens(spec.text)}
        }]

    async def _generate_progressive_chunks(
        self,
        spec: EmbedSpec,
        dim: int,
        rng: random.Random,
        ctx: Optional[EmbeddingContext],
    ) -> List[Dict[str, Any]]:
        num_chunks = random.randint(self.stream_min_chunks, self.stream_max_chunks)
        vectors_per_chunk = max(1, dim // num_chunks)

        chunks: List[Dict[str, Any]] = []
        accumulated_vector: List[float] = []

        for chunk_idx in range(num_chunks):
            remaining = dim - len(accumulated_vector)
            if remaining <= 0:
                break

            chunk_size = min(vectors_per_chunk, remaining)
            partial_vec = self._make_vector(chunk_size, rng)
            accumulated_vector.extend(partial_vec)

            current_dim = len(accumulated_vector)
            embedding = EmbeddingVector(
                vector=accumulated_vector.copy(),
                text=spec.text if chunk_idx == 0 else "",
                model=spec.model,
                dimensions=current_dim,
                index=chunk_idx,
                metadata={"partial": True, "total_dimensions": dim} if current_dim < dim else None
            )

            is_final = (chunk_idx == num_chunks - 1 or len(accumulated_vector) >= dim)
            if is_final and self.normalizes_at_source and spec.normalize:
                embedding = EmbeddingVector(
                    vector=self._normalize(accumulated_vector),
                    text=embedding.text,
                    model=embedding.model,
                    dimensions=embedding.dimensions,
                    index=embedding.index,
                    metadata=embedding.metadata,
                )

            chunks.append({
                "embeddings": [embedding],
                "usage": {"tokens": self._approx_tokens(spec.text)} if is_final else None
            })

        return chunks

    async def _generate_multi_vector_chunks(
        self,
        spec: EmbedSpec,
        dim: int,
        rng: random.Random,
        ctx: Optional[EmbeddingContext],
    ) -> List[Dict[str, Any]]:
        num_chunks = random.randint(self.stream_min_chunks, self.stream_max_chunks)
        vectors_per_chunk = random.randint(1, 3)

        chunks: List[Dict[str, Any]] = []
        total_vectors_generated = 0

        for chunk_idx in range(num_chunks):
            chunk_embeddings: List[EmbeddingVector] = []

            for vec_idx in range(vectors_per_chunk):
                vec = self._make_vector(dim, rng)
                if self.normalizes_at_source and spec.normalize:
                    vec = self._normalize(vec)

                embedding = EmbeddingVector(
                    vector=vec,
                    text=spec.text if vec_idx == 0 else f"{spec.text} [variation {vec_idx}]",
                    model=spec.model,
                    dimensions=dim,
                    index=total_vectors_generated,
                )
                chunk_embeddings.append(embedding)
                total_vectors_generated += 1

            is_final = (chunk_idx == num_chunks - 1)
            chunks.append({
                "embeddings": chunk_embeddings,
                "usage": {"tokens": self._approx_tokens(spec.text) * len(chunk_embeddings)} if is_final else None
            })

        return chunks

    # ---------------------------------------------------------------------
    # Batch embed — public override to ensure fallback-per-item validation
    # ---------------------------------------------------------------------
    async def embed_batch(
        self,
        spec: BatchEmbedSpec,
        *,
        ctx: Optional[EmbeddingContext] = None,
    ) -> BatchEmbedResult:
        self._stats["embed_batch_calls"] += 1
        start_time = time.monotonic()

        async def _run() -> BatchEmbedResult:
            caps = await self._do_capabilities()
            if spec.model not in caps.supported_models:
                raise ModelNotAvailable(f"Model '{spec.model}' is not supported")

            if caps.max_batch_size and len(spec.texts) > caps.max_batch_size:
                raise BadRequest(
                    f"Batch size {len(spec.texts)} exceeds maximum of {caps.max_batch_size}",
                    details={"max_batch_size": caps.max_batch_size},
                )

            eff_texts: List[str] = []
            for text in spec.texts:
                if caps.max_text_length:
                    new_text, _ = self._trunc.apply(text, caps.max_text_length, spec.truncate)
                    eff_texts.append(new_text)
                else:
                    eff_texts.append(text)

            eff_spec = BatchEmbedSpec(
                texts=eff_texts,
                model=spec.model,
                truncate=spec.truncate,
                normalize=spec.normalize,
                metadatas=spec.metadatas,
            )

            try:
                if not self.supports_batch:
                    raise NotSupported("native batch not supported in this mode")
                result = await self._do_embed_batch(eff_spec, ctx=ctx)
            except NotSupported:
                result = await self._embed_batch_fallback(eff_spec, spec, caps, ctx=ctx)

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
                        index=ev.index,
                        metadata=ev.metadata,
                    )

            self._stats["total_texts_embedded"] += len(result.embeddings)
            if result.total_tokens is not None:
                self._stats["total_tokens_processed"] += result.total_tokens

            elapsed_ms = (time.monotonic() - start_time) * 1000
            self._stats["total_processing_time_ms"] += elapsed_ms

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
            call=_run,
            metric_extra=metric_extra,
            error_extra=error_extra,
        )

    async def _embed_batch_fallback(
        self,
        eff_spec: BatchEmbedSpec,
        original_spec: BatchEmbedSpec,
        caps: EmbeddingCapabilities,
        *,
        ctx: Optional[EmbeddingContext],
    ) -> BatchEmbedResult:
        embeddings: List[EmbeddingVector] = []
        failed: List[Dict[str, Any]] = []

        for idx, text in enumerate(eff_spec.texts):
            try:
                if not isinstance(text, str) or not text.strip():
                    raise BadRequest("text must be a non-empty string")

                single_spec = EmbedSpec(
                    text=text,
                    model=eff_spec.model,
                    truncate=eff_spec.truncate,
                    normalize=False,
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
                            index=ev.index,
                            metadata=ev.metadata,
                        )

                embeddings.append(ev)

            except (BadRequest, ModelNotAvailable, NotSupported, Unavailable, ResourceExhausted, TransientNetwork) as item_err:
                self._record_failure(
                    failed,
                    index=idx,
                    text=text,
                    err=item_err,
                    metadatas=eff_spec.metadatas,
                )
            except Exception as item_err:
                self._record_failure(
                    failed,
                    index=idx,
                    text=text,
                    err=item_err,
                    metadatas=eff_spec.metadatas,
                )

        return BatchEmbedResult(
            embeddings=embeddings,
            model=eff_spec.model,
            total_texts=len(original_spec.texts),
            total_tokens=None,
            failed_texts=failed,
        )

    async def _do_embed_batch(
        self,
        spec: BatchEmbedSpec,
        *,
        ctx: Optional[EmbeddingContext] = None,
    ) -> BatchEmbedResult:
        self._maybe_fail(op="embed_batch", ctx=ctx)

        if spec.model not in self.supported_models:
            raise ModelNotAvailable(f"Model '{spec.model}' is not supported")

        if self.max_batch_size and len(spec.texts) > self.max_batch_size:
            raise BadRequest(f"Batch size {len(spec.texts)} exceeds maximum of {self.max_batch_size}")

        await self._sleep_random(bonus_ms=10)

        dim = self._dimensions_for(spec.model)
        embeddings: List[EmbeddingVector] = []
        total_tokens = 0
        failures: List[Dict[str, Any]] = []

        for i, text in enumerate(spec.texts):
            try:
                self._maybe_fail(op="embed_batch:item", ctx=ctx, text=text, per_item=True)

                if self.collect_failures_in_native_batch and (not isinstance(text, str) or not text.strip()):
                    raise BadRequest("text must be a non-empty string")

                rng = self._rng_for(spec.model, text)
                vec = self._make_vector(dim, rng)
                if self.normalizes_at_source and spec.normalize:
                    vec = self._normalize(vec)

                metadata = None
                if spec.metadatas and i < len(spec.metadatas):
                    metadata = spec.metadatas[i]

                embeddings.append(
                    EmbeddingVector(
                        vector=vec,
                        text=text,
                        model=spec.model,
                        dimensions=len(vec),
                        index=i,
                        metadata=metadata,
                    )
                )
                total_tokens += self._approx_tokens(text)

            except (Unavailable, ResourceExhausted, TransientNetwork, BadRequest, ModelNotAvailable) as item_err:
                if self.collect_failures_in_native_batch:
                    self._record_failure(
                        failures,
                        index=i,
                        text=None,
                        err=item_err,
                        metadatas=spec.metadatas,
                    )
                else:
                    raise
            except Exception as item_err:
                if self.collect_failures_in_native_batch:
                    self._record_failure(
                        failures,
                        index=i,
                        text=None,
                        err=item_err,
                        metadatas=spec.metadatas,
                    )
                else:
                    raise

            if self.yield_every_n and (i + 1) % self.yield_every_n == 0:
                await asyncio.sleep(0)

        return BatchEmbedResult(
            embeddings=embeddings,
            model=spec.model,
            total_texts=len(spec.texts),
            total_tokens=total_tokens,
            failed_texts=failures if self.collect_failures_in_native_batch else [],
        )

    async def _do_count_tokens(
        self,
        text: str,
        model: str,
        *,
        ctx: Optional[EmbeddingContext] = None,
    ) -> int:
        self._stats["count_tokens_calls"] += 1
        start_time = time.monotonic()

        if model not in self.supported_models:
            raise ModelNotAvailable(f"Model '{model}' is not supported")
        if self.simulate_latency:
            await asyncio.sleep(0.005)

        result = self._approx_tokens(text)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        self._stats["total_processing_time_ms"] += elapsed_ms
        return result

    async def _do_get_stats(self, ctx: Optional[EmbeddingContext] = None) -> EmbeddingStats:
        total_requests = (
            self._stats["embed_calls"]
            + self._stats["embed_batch_calls"]
            + self._stats["stream_embed_calls"]
            + self._stats["count_tokens_calls"]
        )

        avg_time = 0.0
        if total_requests > 0:
            avg_time = self._stats["total_processing_time_ms"] / total_requests

        return EmbeddingStats(
            total_requests=total_requests,
            total_texts=self._stats["total_texts_embedded"],
            total_tokens=self._stats["total_tokens_processed"],
            cache_hits=self._stats["cache_hits"],
            cache_misses=self._stats["cache_misses"],
            avg_processing_time_ms=avg_time,
            error_count=0,
            stream_requests=self._stats["stream_embed_calls"],
            stream_chunks_generated=self._stats["total_stream_chunks"],
            stream_abandoned=self._stats["abandoned_streams"],
        )

    # ---------------------------------------------------------------------
    # Cache Testing Helpers - SIMULATION ONLY for thin mode testing
    # ---------------------------------------------------------------------
    def _should_cache_hit(self, operation: str, text: str, ctx: Optional[EmbeddingContext]) -> bool:
        if self.cache_behavior == "force_hit":
            return True
        if self.cache_behavior == "force_miss":
            return False
        if self.cache_behavior == "simulate_error":
            return False

        if text and "[NO_CACHE]" in text:
            return False

        rng = random.Random(hash(f"{operation}|{text}") % 1_000_000)
        return rng.random() < self.cache_hit_rate

    def _should_cache_set_fail(self) -> bool:
        # Deterministic failure patterns
        if self.cache_behavior == "simulate_error":
            # deterministic: fail every other attempt
            return (self._stats["cache_set_attempts"] % 2) == 1

        rng = random.Random(self._stats["cache_set_attempts"])
        return rng.random() < self.cache_set_failure_rate

    def _make_cached_vector(self, model: str, text: str) -> List[float]:
        dim = self._dimensions_for(model)
        _ = self._rng_for(model, f"CACHED_{text}")  # keep seed-based determinism in case you extend patterns later
        if self.test_vector_pattern == "zeros":
            return [0.0] * dim
        if self.test_vector_pattern == "ones":
            return [1.0] * dim
        return [0.5 + 0.3 * math.sin(i * 0.5) for i in range(dim)]

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
            if key == "deadline":
                raise DeadlineExceeded(f"Mocked {op} deadline exceeded", retry_after_ms=0)

        if text:
            if "[UNAVAILABLE]" in text:
                raise Unavailable(f"Mocked {op} unavailable (text sentinel)", retry_after_ms=500)
            if "[RATE_LIMIT]" in text:
                raise ResourceExhausted(f"Mocked {op} rate-limited (text sentinel)", retry_after_ms=800)
            if "[TRANSIENT]" in text:
                raise TransientNetwork(f"Mocked {op} transient network", retry_after_ms=600)
            if "[DEADLINE]" in text:
                raise DeadlineExceeded(f"Mocked {op} deadline (text sentinel)", retry_after_ms=0)
            if "[NO_CACHE]" in text:
                pass

        if not per_item and self.failure_rate > 0.0 and random.random() < self.failure_rate:
            if random.random() < 0.5:
                raise ResourceExhausted(f"Mocked {op} rate-limited", retry_after_ms=800)
            raise Unavailable(f"Mocked {op} overloaded", retry_after_ms=500)

    async def _sleep_random(self, bonus_ms: int = 0) -> None:
        if not self.simulate_latency:
            return
        lo, hi = self.latency_ms
        dur_ms = float(lo if lo == hi else random.uniform(lo, hi + bonus_ms))
        if dur_ms <= 0:
            return
        await asyncio.sleep(dur_ms / 1000.0)

    async def _sleep_random_range(self, min_ms: int, max_ms: int) -> None:
        if not self.simulate_latency:
            return
        dur_ms = random.uniform(min_ms, max_ms)
        if dur_ms <= 0:
            return
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
                (rng.random() + rng.random() + rng.random()
                 + rng.random() + rng.random() + rng.random()) - 3.0
                for _ in range(dim)
            ]
        return [rng.random() * 2.0 - 1.0 for _ in range(dim)]

    def _normalize(self, vec: List[float]) -> List[float]:
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def _approx_tokens(self, text: str) -> int:
        if not text:
            return 0
        words = max(1, len(text.split()))
        return int(math.ceil(words / max(0.1, self.token_factor))) + 2

    def _record_failure(
        self,
        failures: List[Dict[str, Any]],
        *,
        index: int,
        text: Optional[str],
        err: Exception,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        default_code: str = "UNAVAILABLE",
        default_message: str = "internal error",
    ) -> None:
        code = getattr(err, "code", None) or type(err).__name__.upper() or default_code
        message = getattr(err, "message", None) or str(err) or default_message

        entry: Dict[str, Any] = {
            "index": index,
            "error": type(err).__name__,
            "code": code,
            "message": message,
        }
        if text is not None:
            entry["text"] = text
        if metadatas and index < len(metadatas):
            entry["metadata"] = metadatas[index]
        failures.append(entry)

    def _get_current_stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "streaming_metrics": dict(self._streaming_metrics),
            "active_streams": self._stream_active_count,
            "timestamp": time.time(),
        }

    # ---------------------------------------------------------------------
    # Test Helper Methods
    # ---------------------------------------------------------------------
    def reset_stats(self) -> None:
        self._stats = {
            "embed_calls": 0,
            "embed_batch_calls": 0,
            "stream_embed_calls": 0,
            "count_tokens_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_set_attempts": 0,
            "cache_set_failures": 0,
            "total_texts_embedded": 0,
            "total_tokens_processed": 0,
            "total_stream_chunks": 0,
            "abandoned_streams": 0,
            "completed_streams": 0,
            "total_processing_time_ms": 0.0,
        }
        self._streaming_metrics = {
            "total_chunks_generated": 0,
            "chunks_by_pattern": {"single": 0, "progressive": 0, "multi_vector": 0},
        }
        self._stream_active_count = 0

    def get_detailed_stats(self) -> Dict[str, Any]:
        total_ops = (
            self._stats["embed_calls"]
            + self._stats["embed_batch_calls"]
            + self._stats["stream_embed_calls"]
            + self._stats["count_tokens_calls"]
        )

        avg_time_per_op = 0.0
        if total_ops > 0:
            avg_time_per_op = self._stats["total_processing_time_ms"] / total_ops

        cache_effectiveness = 0.0
        total_cache_accesses = self._stats["cache_hits"] + self._stats["cache_misses"]
        if total_cache_accesses > 0:
            cache_effectiveness = self._stats["cache_hits"] / total_cache_accesses

        computed = {
            "avg_time_per_op_ms": avg_time_per_op,
            "cache_effectiveness": cache_effectiveness,
        }

        if self._stats["cache_set_attempts"] > 0:
            computed["cache_set_failure_rate"] = (
                self._stats["cache_set_failures"] / self._stats["cache_set_attempts"]
            )

        if self._stats["stream_embed_calls"] > 0:
            computed["stream_completion_rate"] = (
                self._stats["completed_streams"] / self._stats["stream_embed_calls"]
            )

        return {
            "operations": dict(self._stats),
            "streaming": dict(self._streaming_metrics),
            "active_streams": self._stream_active_count,
            "computed": computed,
        }
