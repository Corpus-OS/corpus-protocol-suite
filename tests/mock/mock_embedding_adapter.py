# tests/mock/mock_embedding_adapter.py
# SPDX-License-Identifier: Apache-2.0
"""
Mock Embedding adapter used in example scripts and conformance tests.

Goals:
- Deterministic vectors (seed-based RNG) for stable conformance runs
- Clean alignment with BaseEmbeddingAdapter semantics (no overrides of public ops)
- Partial-failure batch semantics are enforced by the BaseEmbeddingAdapter contract
- Streaming support with deterministic chunk patterns
- Stats suitable for tests/monitoring

IMPORTANT:
- This mock does NOT override embed_batch(). The base owns batch validation + fallback semantics.
- This mock does NOT simulate caching. Tests should validate caching via an injected cache
  implementation (e.g., a counting TTL cache) and BaseEmbeddingAdapter’s cache logic.
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
    Mock Embedding adapter.

    Streaming patterns:
      - "single": one final chunk containing one full vector
      - "progressive": multiple chunks containing partial vectors growing over time
      - "multi_vector": multiple chunks, each chunk contains 1-3 complete vectors

    Notes:
      - _do_embed validates non-empty text (item-level validation).
      - _do_embed_batch may optionally collect per-item failures when enabled, but
        BaseEmbeddingAdapter may also choose to pre-split invalid items depending on the contract.
    """

    # ----- Tunables -----
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
    failure_rate: float  # demos only; keep 0.0 for conformance stability
    test_vector_pattern: Optional[str]  # None|"zeros"|"ones"|"unit_x"|"gaussian"
    collect_failures_in_native_batch: bool
    yield_every_n: int
    simulate_latency: bool

    # ----- Streaming config -----
    stream_chunk_pattern: str  # "single" | "progressive" | "multi_vector"
    stream_min_chunks: int
    stream_max_chunks: int
    stream_yield_interval_ms: Tuple[int, int]
    stream_abandonment_rate: float

    # ----- Stats -----
    _stats: Dict[str, Any]
    _stream_active_count: int

    def __init__(
        self,
        *,
        # Base adapter infra
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
        # Mock knobs
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
        failure_rate: float = 0.0,
        test_vector_pattern: Optional[str] = None,
        collect_failures_in_native_batch: bool = False,
        yield_every_n: int = 50,
        simulate_latency: bool = True,
        # Streaming
        stream_chunk_pattern: str = "single",
        stream_min_chunks: int = 1,
        stream_max_chunks: int = 3,
        stream_yield_interval_ms: Tuple[int, int] = (5, 15),
        stream_abandonment_rate: float = 0.0,
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

        self.stream_chunk_pattern = stream_chunk_pattern
        self.stream_min_chunks = max(1, int(stream_min_chunks))
        self.stream_max_chunks = max(self.stream_min_chunks, int(stream_max_chunks))
        lo = max(0, int(stream_yield_interval_ms[0]))
        hi = max(lo, int(stream_yield_interval_ms[1]))
        self.stream_yield_interval_ms = (lo, hi)
        self.stream_abandonment_rate = max(0.0, min(1.0, float(stream_abandonment_rate)))

        self._stats = {
            "embed_calls": 0,
            "embed_batch_calls": 0,
            "stream_embed_calls": 0,
            "count_tokens_calls": 0,
            "total_texts_embedded": 0,
            "total_tokens_processed": 0,
            "total_stream_chunks": 0,
            "abandoned_streams": 0,
            "completed_streams": 0,
            "total_processing_time_ms": 0.0,
        }
        self._stream_active_count = 0

        # Validation
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

    # ---------------------------------------------------------------------
    # Capabilities & Health
    # ---------------------------------------------------------------------
    async def _do_capabilities(self) -> EmbeddingCapabilities:
        # IMPORTANT: must match base cache decision (mode-agnostic)
        supports_caching = bool(
            getattr(self._cache, "supports_ttl", False) and getattr(self, "_cache_embed_ttl_s", 0) > 0
        )
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
            supports_caching=supports_caching,
            idempotent_writes=False,
            supports_multi_tenant=True,
            normalizes_at_source=self.normalizes_at_source,
            truncation_mode="base",
            supports_deadline=True,
        )

    async def _do_health(self, *, ctx: Optional[EmbeddingContext] = None) -> Dict[str, Any]:
        degraded = bool((ctx and ctx.attrs.get("health") == "degraded"))
        return {
            "ok": not degraded,
            "server": self.name,
            "version": "1.0.0",
            "models": {m: ("degraded" if degraded else "ok") for m in self.supported_models},
        }

    # ---------------------------------------------------------------------
    # Single embed (item-level validation lives here)
    # ---------------------------------------------------------------------
    async def _do_embed(self, spec: EmbedSpec, *, ctx: Optional[EmbeddingContext] = None) -> EmbedResult:
        self._stats["embed_calls"] += 1
        t0 = time.monotonic()

        self._maybe_fail(op="embed", ctx=ctx, text=spec.text)

        if spec.model not in self.supported_models:
            raise ModelNotAvailable(f"Model '{spec.model}' is not supported")

        # Item-level validation (contract-friendly for batch partial failures)
        if not isinstance(spec.text, str) or not spec.text.strip():
            raise BadRequest("text must be a non-empty string")

        await self._sleep_random()

        dim = self._dimensions_for(spec.model)
        rng = self._rng_for(spec.model, spec.text)
        vec = self._make_vector(dim, rng)
        if self.normalizes_at_source and spec.normalize:
            vec = self._normalize(vec)

        tokens = self._approx_tokens(spec.text)

        self._stats["total_texts_embedded"] += 1
        self._stats["total_tokens_processed"] += tokens
        self._stats["total_processing_time_ms"] += (time.monotonic() - t0) * 1000.0

        return EmbedResult(
            embedding=EmbeddingVector(
                vector=vec,
                text=spec.text,
                model=spec.model,
                dimensions=len(vec),
                index=None,
                metadata=spec.metadata,
            ),
            model=spec.model,
            text=spec.text,
            tokens_used=tokens,
            truncated=False,
        )

    # ---------------------------------------------------------------------
    # Streaming
    # ---------------------------------------------------------------------
    async def _do_stream_embed(
        self, spec: EmbedSpec, *, ctx: Optional[EmbeddingContext] = None
    ) -> AsyncIterator[EmbedChunk]:
        self._stats["stream_embed_calls"] += 1
        self._stream_active_count += 1
        t0 = time.monotonic()

        stream_id = f"stream_{int(time.time() * 1000)}_{hash(spec.text) % 10000:04d}"

        try:
            self._maybe_fail(op="stream_embed", ctx=ctx, text=spec.text)

            if spec.model not in self.supported_models:
                raise ModelNotAvailable(f"Model '{spec.model}' is not supported")

            if not isinstance(spec.text, str) or not spec.text.strip():
                raise BadRequest("text must be a non-empty string")

            dim = self._dimensions_for(spec.model)
            rng = self._rng_for(spec.model, spec.text)

            if self.stream_chunk_pattern == "single":
                chunks = await self._generate_single_chunk(spec, dim, rng)
            elif self.stream_chunk_pattern == "progressive":
                chunks = await self._generate_progressive_chunks(spec, dim, rng)
            else:
                chunks = await self._generate_multi_vector_chunks(spec, dim, rng)

            total_chunks = len(chunks)

            for i, chunk_data in enumerate(chunks):
                if (
                    self.stream_abandonment_rate > 0
                    and random.random() < self.stream_abandonment_rate
                    and i < total_chunks - 1
                ):
                    self._stats["abandoned_streams"] += 1
                    raise asyncio.CancelledError(f"Stream {stream_id} abandoned (simulated)")

                if self.simulate_latency and i > 0:
                    await self._sleep_random_range(*self.stream_yield_interval_ms)

                is_final = (i == total_chunks - 1)
                self._stats["total_stream_chunks"] += 1

                yield EmbedChunk(
                    embeddings=chunk_data["embeddings"],
                    is_final=is_final,
                    usage=chunk_data.get("usage"),
                    model=spec.model,
                )

            self._stats["completed_streams"] += 1
            self._stats["total_texts_embedded"] += 1
            self._stats["total_tokens_processed"] += self._approx_tokens(spec.text)
            self._stats["total_processing_time_ms"] += (time.monotonic() - t0) * 1000.0

        finally:
            self._stream_active_count = max(0, self._stream_active_count - 1)

    async def _generate_single_chunk(self, spec: EmbedSpec, dim: int, rng: random.Random) -> List[Dict[str, Any]]:
        await self._sleep_random()
        vec = self._make_vector(dim, rng)
        if self.normalizes_at_source and spec.normalize:
            vec = self._normalize(vec)
        ev = EmbeddingVector(vector=vec, text=spec.text, model=spec.model, dimensions=dim)
        return [{"embeddings": [ev], "usage": {"tokens": self._approx_tokens(spec.text)}}]

    async def _generate_progressive_chunks(self, spec: EmbedSpec, dim: int, rng: random.Random) -> List[Dict[str, Any]]:
        num_chunks = random.randint(self.stream_min_chunks, self.stream_max_chunks)
        per = max(1, dim // num_chunks)

        chunks: List[Dict[str, Any]] = []
        acc: List[float] = []

        for idx in range(num_chunks):
            remaining = dim - len(acc)
            if remaining <= 0:
                break
            chunk_size = min(per, remaining)
            acc.extend(self._make_vector(chunk_size, rng))

            cur_dim = len(acc)
            ev = EmbeddingVector(
                vector=acc.copy(),
                text=spec.text if idx == 0 else "",
                model=spec.model,
                dimensions=cur_dim,
                index=idx,
                metadata={"partial": True, "total_dimensions": dim} if cur_dim < dim else None,
            )

            is_final = (idx == num_chunks - 1 or cur_dim >= dim)
            chunks.append(
                {"embeddings": [ev], "usage": {"tokens": self._approx_tokens(spec.text)} if is_final else None}
            )

        return chunks

    async def _generate_multi_vector_chunks(self, spec: EmbedSpec, dim: int, rng: random.Random) -> List[Dict[str, Any]]:
        num_chunks = random.randint(self.stream_min_chunks, self.stream_max_chunks)
        vectors_per = random.randint(1, 3)

        chunks: List[Dict[str, Any]] = []
        global_i = 0

        for chunk_idx in range(num_chunks):
            embs: List[EmbeddingVector] = []
            for j in range(vectors_per):
                vec = self._make_vector(dim, rng)
                if self.normalizes_at_source and spec.normalize:
                    vec = self._normalize(vec)
                embs.append(
                    EmbeddingVector(
                        vector=vec,
                        text=spec.text if j == 0 else f"{spec.text} [variation {j}]",
                        model=spec.model,
                        dimensions=dim,
                        index=global_i,
                    )
                )
                global_i += 1

            is_final = (chunk_idx == num_chunks - 1)
            chunks.append(
                {"embeddings": embs, "usage": {"tokens": self._approx_tokens(spec.text) * len(embs)} if is_final else None}
            )

        return chunks

    # ---------------------------------------------------------------------
    # Native batch hook (BaseEmbeddingAdapter will call this)
    # ---------------------------------------------------------------------
    async def _do_embed_batch(self, spec: BatchEmbedSpec, *, ctx: Optional[EmbeddingContext] = None) -> BatchEmbedResult:
        self._stats["embed_batch_calls"] += 1
        t0 = time.monotonic()

        self._maybe_fail(op="embed_batch", ctx=ctx)

        if spec.model not in self.supported_models:
            raise ModelNotAvailable(f"Model '{spec.model}' is not supported")

        if self.max_batch_size and len(spec.texts) > self.max_batch_size:
            raise BadRequest(f"Batch size {len(spec.texts)} exceeds maximum of {self.max_batch_size}")

        await self._sleep_random(bonus_ms=10)

        dim = self._dimensions_for(spec.model)
        embeddings: List[EmbeddingVector] = []
        failures: List[Dict[str, Any]] = []
        total_tokens = 0

        for i, text in enumerate(spec.texts):
            try:
                self._maybe_fail(op="embed_batch:item", ctx=ctx, text=text, per_item=True)

                # Item-level validation: either collect per-item failures or raise (provider choice)
                if not isinstance(text, str) or not text.strip():
                    raise BadRequest("text must be a non-empty string")

                rng = self._rng_for(spec.model, text)
                vec = self._make_vector(dim, rng)
                if self.normalizes_at_source and spec.normalize:
                    vec = self._normalize(vec)

                md = spec.metadatas[i] if spec.metadatas and i < len(spec.metadatas) else None

                embeddings.append(
                    EmbeddingVector(
                        vector=vec,
                        text=text,
                        model=spec.model,
                        dimensions=len(vec),
                        index=i,
                        metadata=md,
                    )
                )
                total_tokens += self._approx_tokens(text)

            except (BadRequest, Unavailable, ResourceExhausted, TransientNetwork, ModelNotAvailable) as item_err:
                if self.collect_failures_in_native_batch:
                    failures.append(
                        {
                            "index": i,
                            "error": type(item_err).__name__,
                            "code": getattr(item_err, "code", None) or type(item_err).__name__.upper(),
                            "message": getattr(item_err, "message", None) or str(item_err),
                            **({"metadata": spec.metadatas[i]} if spec.metadatas and i < len(spec.metadatas) else {}),
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
                            **({"metadata": spec.metadatas[i]} if spec.metadatas and i < len(spec.metadatas) else {}),
                        }
                    )
                else:
                    raise

            if self.yield_every_n and (i + 1) % self.yield_every_n == 0:
                await asyncio.sleep(0)

        self._stats["total_texts_embedded"] += len(embeddings)
        self._stats["total_tokens_processed"] += total_tokens
        self._stats["total_processing_time_ms"] += (time.monotonic() - t0) * 1000.0

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
    async def _do_count_tokens(self, text: str, model: str, *, ctx: Optional[EmbeddingContext] = None) -> int:
        self._stats["count_tokens_calls"] += 1
        t0 = time.monotonic()

        if model not in self.supported_models:
            raise ModelNotAvailable(f"Model '{model}' is not supported")
        if self.simulate_latency:
            await asyncio.sleep(0.005)

        n = self._approx_tokens(text)
        self._stats["total_processing_time_ms"] += (time.monotonic() - t0) * 1000.0
        return n

    # ---------------------------------------------------------------------
    # Stats
    # ---------------------------------------------------------------------
    async def _do_get_stats(self, ctx: Optional[EmbeddingContext] = None) -> EmbeddingStats:
        total_requests = (
            self._stats["embed_calls"]
            + self._stats["embed_batch_calls"]
            + self._stats["stream_embed_calls"]
            + self._stats["count_tokens_calls"]
        )
        avg = (self._stats["total_processing_time_ms"] / total_requests) if total_requests else 0.0

        # Cache hit/miss counters are owned by BaseEmbeddingAdapter metrics; this mock keeps them 0.
        return EmbeddingStats(
            total_requests=total_requests,
            total_texts=self._stats["total_texts_embedded"],
            total_tokens=self._stats["total_tokens_processed"],
            cache_hits=0,
            cache_misses=0,
            avg_processing_time_ms=avg,
            error_count=0,
            stream_requests=self._stats["stream_embed_calls"],
            stream_chunks_generated=self._stats["total_stream_chunks"],
            stream_abandoned=self._stats["abandoned_streams"],
        )

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
                raise TransientNetwork(f"Mocked {op} transient (text sentinel)", retry_after_ms=600)
            if "[DEADLINE]" in text:
                raise DeadlineExceeded(f"Mocked {op} deadline (text sentinel)", retry_after_ms=0)

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
        if not isinstance(text, str) or not text:
            return 0
        words = max(1, len(text.split()))
        return int(math.ceil(words / max(0.1, self.token_factor))) + 2

    # ---------------------------------------------------------------------
    # Test helpers
    # ---------------------------------------------------------------------
    def reset_stats(self) -> None:
        self._stats = {
            "embed_calls": 0,
            "embed_batch_calls": 0,
            "stream_embed_calls": 0,
            "count_tokens_calls": 0,
            "total_texts_embedded": 0,
            "total_tokens_processed": 0,
            "total_stream_chunks": 0,
            "abandoned_streams": 0,
            "completed_streams": 0,
            "total_processing_time_ms": 0.0,
        }
        self._stream_active_count = 0

    def get_detailed_stats(self) -> Dict[str, Any]:
        total_ops = (
            self._stats["embed_calls"]
            + self._stats["embed_batch_calls"]
            + self._stats["stream_embed_calls"]
            + self._stats["count_tokens_calls"]
        )
        avg = (self._stats["total_processing_time_ms"] / total_ops) if total_ops else 0.0
        return {
            "operations": dict(self._stats),
            "active_streams": self._stream_active_count,
            "computed": {"avg_time_per_op_ms": avg},
        }
