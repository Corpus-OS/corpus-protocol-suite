# examples/embedding/mock_embedding_adapter.py
# SPDX-License-Identifier: Apache-2.0
"""
Mock Embedding adapter used in example scripts.

Implements BaseEmbeddingAdapter hooks for demonstration purposes only.
Simulates latency, deterministic vectors, token counting, batch ops, and transient failures.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import random
from dataclasses import dataclass
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
    # Error types (Option A: import from base)
    BadRequest,
    NotSupported,
    Unavailable,
    ResourceExhausted,
    ModelNotAvailable,
)

@dataclass
class MockEmbeddingAdapter(BaseEmbeddingAdapter):
    """A mock Embedding adapter for protocol demonstrations."""

    # ----- Tunables for tests/demos -----
    name: str = "mock-embedding"
    failure_rate: float = 0.10  # 10% chance of simulated transient failure
    supported_models: Tuple[str, ...] = ("mock-embed-512", "mock-embed-1024")
    dimensions_by_model: Dict[str, int] = None  # set in __post_init__
    max_text_length: int = 4000
    max_batch_size: int = 128
    normalizes_at_source: bool = False  # if True, adapter returns normalized vectors when requested
    token_factor: float = 0.75  # rough token estimate factor
    latency_ms: Tuple[int, int] = (10, 25)  # (min,max) artificial latency per call

    def __post_init__(self) -> None:
        if self.dimensions_by_model is None:
            self.dimensions_by_model = {
                "mock-embed-512": 512,
                "mock-embed-1024": 1024,
            }

    # ---------------------------------------------------------------------
    # Capabilities & Health
    # ---------------------------------------------------------------------
    async def _do_capabilities(self) -> EmbeddingCapabilities:
        return EmbeddingCapabilities(
            server="mock-embedding",
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
            truncation_mode="base",  # base applies truncation policy before _do_*
            supports_deadline=True,
        )

    async def _do_health(self, *, ctx: Optional[EmbeddingContext] = None) -> Dict[str, Any]:
        # Occasionally report degraded to exercise monitoring paths
        if random.random() < 0.20:
            return {
                "ok": False,
                "server": "mock-embedding",
                "version": "1.0.0",
                "models": {m: "degraded" for m in self.supported_models},
            }
        return {
            "ok": True,
            "server": "mock-embedding",
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
        self._maybe_fail("embed")

        if spec.model not in self.supported_models:
            raise ModelNotAvailable(f"Model '{spec.model}' is not supported")

        await self._sleep_random()

        dim = self._dimensions_for(spec.model)
        rng = self._rng_for(spec.model, spec.text)
        vec = self._make_vector(dim, rng)

        # If adapter is configured to normalize at source and the caller requested normalization,
        # return a unit-length vector now (otherwise the Base may normalize later).
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
            truncated=False,  # BaseEmbeddingAdapter sets this flag when truncation occurred
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
        self._maybe_fail("embed_batch")

        if spec.model not in self.supported_models:
            raise ModelNotAvailable(f"Model '{spec.model}' is not supported")

        if self.max_batch_size and len(spec.texts) > self.max_batch_size:
            raise BadRequest(f"Batch size {len(spec.texts)} exceeds maximum of {self.max_batch_size}")

        await self._sleep_random(bonus_ms=10)

        dim = self._dimensions_for(spec.model)
        embeddings: List[EmbeddingVector] = []
        total_tokens = 0

        # Lightweight cooperative scheduling every N items
        for i, text in enumerate(spec.texts):
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

            if (i + 1) % 50 == 0:
                await asyncio.sleep(0)  # yield

        return BatchEmbedResult(
            embeddings=embeddings,
            model=spec.model,
            total_texts=len(spec.texts),
            total_tokens=total_tokens,
            failed_texts=[],  # this mock does not inject per-item failures by default
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
    def _maybe_fail(self, op: str) -> None:
        """Inject transient failures for demonstration purposes."""
        if random.random() < self.failure_rate:
            # Flip between capacity and overload style failures
            if random.random() < 0.5:
                raise ResourceExhausted(f"Mocked {op} rate-limited", retry_after_ms=800)
            raise Unavailable(f"Mocked {op} overloaded", retry_after_ms=500)

    async def _sleep_random(self, bonus_ms: int = 0) -> None:
        lo, hi = self.latency_ms
        dur_ms = random.uniform(lo, hi + bonus_ms)
        await asyncio.sleep(dur_ms / 1000.0)

    def _dimensions_for(self, model: str) -> int:
        try:
            return int(self.dimensions_by_model[model])
        except Exception:
            raise ModelNotAvailable(f"Model '{model}' has no configured dimension")

    def _rng_for(self, model: str, text: str) -> random.Random:
        h = hashlib.sha256(f"{model}|{text}".encode("utf-8")).hexdigest()
        # Use a stable integer seed (take lower 16 hex chars for simplicity)
        seed = int(h[-16:], 16)
        return random.Random(seed)

    def _make_vector(self, dim: int, rng: random.Random) -> List[float]:
        # Deterministic floats in [-1, 1)
        return [(rng.random() * 2.0) - 1.0 for _ in range(dim)]

    def _normalize(self, vec: List[float]) -> List[float]:
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def _approx_tokens(self, text: str) -> int:
        # Very rough heuristic: words / factor + small overhead
        words = max(1, len(text.split()))
        return int(math.ceil(words / max(0.1, self.token_factor))) + 2


# ---------------------------------------------------------------------------
# Optional: simple self-demo (kept dependency-free)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    async def _demo() -> None:
        random.seed(7)  # deterministic demo
        adapter = MockEmbeddingAdapter(failure_rate=0.15, normalizes_at_source=False)

        # Capabilities
        caps = await adapter.capabilities()
        print("[CAPABILITIES]", caps)

        # Health
        health = await adapter.health()
        print("[HEALTH]", health)

        # Single embed
        spec = EmbedSpec(text="hello world", model="mock-embed-512", truncate=True, normalize=True)
        res = await adapter.embed(spec)
        print("[EMBED] dims=", res.embedding.dimensions, "norm≈", round(math.sqrt(sum(x*x for x in res.embedding.vector)), 3))

        # Batch embed
        bspec = BatchEmbedSpec(texts=["a", "quick brown fox", "jumps over the lazy dog"], model="mock-embed-1024", normalize=False)
        bres = await adapter.embed_batch(bspec)
        print("[BATCH] count=", len(bres.embeddings), "total_tokens=", bres.total_tokens)

        # Count tokens
        tks = await adapter.count_tokens("token counting example text", "mock-embed-512")
        print("[COUNT_TOKENS]", tks)

    asyncio.run(_demo())

