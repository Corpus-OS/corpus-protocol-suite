# corpus_sdk/examples/vector/mock_vector_adapter.py
# SPDX-License-Identifier: Apache-2.0
"""
Mock Vector adapter used in Corpus SDK example scripts.

Implements BaseVectorAdapter methods for demonstration purposes only.
Simulates an in-memory vector store with basic filtering and metrics-friendly behavior.
"""

from __future__ import annotations

import asyncio
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from corpus_sdk.vector.vector_base import (
    BaseVectorAdapter,
    Vector,
    VectorMatch,
    QueryResult,
    UpsertResult,
    DeleteResult,
    NamespaceResult,
    QuerySpec,
    UpsertSpec,
    DeleteSpec,
    NamespaceSpec,
    VectorCapabilities,
    OperationContext,
    # Errors
    BadRequest,
    DimensionMismatch,
    IndexNotReady,
    NotSupported,
    ResourceExhausted,
    Unavailable,
)

# ------------------------------- constants --------------------------------- #

METRIC_COSINE = "cosine"
METRIC_EUCLIDEAN = "euclidean"
METRIC_DOTPRODUCT = "dotproduct"

SUPPORTED_DISTANCE_METRICS: Tuple[str, ...] = (
    METRIC_COSINE,
    METRIC_EUCLIDEAN,
    METRIC_DOTPRODUCT,
)

# ------------------------------- utilities --------------------------------- #


def _cosine_sim(a: List[float], b: List[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    den_a = math.sqrt(sum(x * x for x in a)) or 1.0
    den_b = math.sqrt(sum(y * y for y in b)) or 1.0
    return num / (den_a * den_b)


def _euclidean(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _filter_match(meta: Optional[Dict[str, Any]], flt: Optional[Dict[str, Any]]) -> bool:
    """
    Very simple equality-only filter.

    This is intentionally minimal for a mock adapter; it exists to prove:
      - filters can be applied pre-search
      - filters work consistently for query + delete
    Real adapters are expected to support richer filter dialects.
    """
    if not flt:
        return True
    if not meta:
        return False
    for k, v in flt.items():
        if meta.get(k) != v:
            return False
    return True


# ------------------------------- mock state -------------------------------- #


@dataclass
class _NamespaceInfo:
    dimensions: int
    distance_metric: str = METRIC_COSINE  # one of SUPPORTED_DISTANCE_METRICS


@dataclass
class _MemoryStore:
    """
    Very small in-memory store keyed by namespace.
    """

    namespaces: Dict[str, _NamespaceInfo] = field(default_factory=dict)
    # ns -> id -> Vector
    data: Dict[str, Dict[str, Vector]] = field(default_factory=dict)

    def ensure_namespace(self, spec: NamespaceSpec) -> None:
        if spec.namespace not in self.namespaces:
            self.namespaces[spec.namespace] = _NamespaceInfo(
                dimensions=spec.dimensions,
                distance_metric=spec.distance_metric,
            )
            self.data.setdefault(spec.namespace, {})


# ----------------------------- adapter class ------------------------------- #


class MockVectorAdapter(BaseVectorAdapter):
    """
    A mock Vector adapter for protocol demonstrations & conformance tests.
    """

    def __init__(
        self,
        name: str = "mock-vector",
        failure_rate: float = 0.0,
        **kwargs
    ) -> None:
        # Initialize the base adapter first
        super().__init__(**kwargs)
        
        self.name = name
        # Deterministic for conformance runs (no random transient failures by default)
        self.failure_rate = failure_rate
        self._store = _MemoryStore()
        
        # Optional seeding logic
        """
        Optional seeding of a 'default' namespace for demos. Disabled by default to
        avoid masking IndexNotReady or isolation tests. Enable by setting:

            VECTOR_SEED_DEFAULT=1
        """
        if os.getenv("VECTOR_SEED_DEFAULT", "0") == "1":
            if "default" not in self._store.namespaces:
                self._store.ensure_namespace(
                    NamespaceSpec(
                        namespace="default",
                        dimensions=2,
                        distance_metric=METRIC_COSINE,
                    )
                )
            if not self._store.data.get("default"):
                self._store.data["default"] = {
                    "seed1": Vector(
                        id="seed1",
                        vector=[1.0, 0.0],
                        metadata={"label": "alpha"},
                        namespace="default",
                    ),
                    "seed2": Vector(
                        id="seed2",
                        vector=[0.0, 1.0],
                        metadata={"label": "beta"},
                        namespace="default",
                    ),
                }

    # --------------------------- capability probe --------------------------- #

    async def _do_capabilities(self) -> VectorCapabilities:
        # tiny simulated network delay
        await asyncio.sleep(0.01)
        return VectorCapabilities(
            server=self.name,
            version="1.0.0",
            max_dimensions=2048,
            supported_metrics=SUPPORTED_DISTANCE_METRICS,
            supports_namespaces=True,
            supports_metadata_filtering=True,
            supports_batch_operations=True,
            max_batch_size=1000,
            supports_index_management=True,
            idempotent_writes=False,
            supports_multi_tenant=False,
            supports_deadline=True,
            max_top_k=1000,
            max_filter_terms=10,
        )

    # ------------------------------ query ---------------------------------- #

    async def _do_query(
        self,
        spec: QuerySpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> QueryResult:
        if self.failure_rate > 0.0 and random.random() < self.failure_rate:
            raise Unavailable("mock backend overloaded", code="OVERLOADED")

        # Require namespace to exist (no implicit creation)
        if spec.namespace not in self._store.namespaces:
            raise BadRequest(f"unknown namespace '{spec.namespace}'")

        caps = await self.capabilities()

        # Enforce top_k vs capabilities
        if caps.max_top_k is not None and spec.top_k > caps.max_top_k:
            raise BadRequest(
                f"top_k {spec.top_k} exceeds maximum of {caps.max_top_k}",
                details={"max_top_k": caps.max_top_k, "namespace": spec.namespace},
            )

        if spec.filter and caps.max_filter_terms and len(spec.filter) > caps.max_filter_terms:
            raise BadRequest(
                f"filter too complex: {len(spec.filter)} terms exceeds {caps.max_filter_terms}",
                details={
                    "provided_terms": len(spec.filter),
                    "max_terms": caps.max_filter_terms,
                    "namespace": spec.namespace,
                },
            )

        ns_info = self._store.namespaces[spec.namespace]
        # Strict dimension check vs namespace schema
        if len(spec.vector) != ns_info.dimensions:
            raise DimensionMismatch(
                f"query vector dimension {len(spec.vector)} does not match namespace {ns_info.dimensions}",
                details={
                    "expected": ns_info.dimensions,
                    "actual": len(spec.vector),
                    "namespace": spec.namespace,
                },
            )

        # If namespace exists but has no data, surface retryable "index not ready"
        if not self._store.data.get(spec.namespace):
            raise IndexNotReady(
                "index not ready (no data in namespace)",
                retry_after_ms=500,
                details={"namespace": spec.namespace},
            )

        # Gather candidates by filter (pre-search filtering semantics)
        bucket = self._store.data.get(spec.namespace, {})
        candidates = [v for v in bucket.values() if _filter_match(v.metadata, spec.filter)]

        # Score with chosen metric
        scored: List[Tuple[float, float, Vector]] = []
        for v in candidates:
            if ns_info.distance_metric == METRIC_COSINE:
                sim = _cosine_sim(spec.vector, v.vector)
                distance = 1.0 - sim
                score = sim
            elif ns_info.distance_metric == METRIC_EUCLIDEAN:
                distance = _euclidean(spec.vector, v.vector)
                score = 1.0 / (1.0 + distance)
            elif ns_info.distance_metric == METRIC_DOTPRODUCT:
                dp = _dot(spec.vector, v.vector)
                distance = -dp  # lower is better
                score = dp
            else:
                # Should never happen given create_namespace validation, but guard anyway
                raise NotSupported(
                    f"distance_metric '{ns_info.distance_metric}' not supported",
                    details={"namespace": spec.namespace},
                )

            scored.append((score, distance, v))

        # Top-K by score (descending)
        scored.sort(key=lambda t: t[0], reverse=True)
        top = scored[: spec.top_k]
        matches: List[VectorMatch] = []

        for score, distance, v in top:
            # Honor include_vectors/include_metadata flags
            out_vec = v.vector if spec.include_vectors else []
            out_meta = v.metadata if spec.include_metadata else None
            matches.append(
                VectorMatch(
                    vector=Vector(
                        id=v.id,
                        vector=out_vec,
                        metadata=out_meta,
                        namespace=v.namespace,
                    ),
                    score=float(score),
                    distance=float(distance),
                )
            )

        await asyncio.sleep(0.01)

        return QueryResult(
            matches=matches,
            query_vector=list(spec.vector),
            namespace=spec.namespace,
            total_matches=len(scored),
        )

    # ------------------------------ upsert --------------------------------- #

    async def _do_upsert(
        self,
        spec: UpsertSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> UpsertResult:
        if self.failure_rate > 0.0 and random.random() < self.failure_rate:
            raise ResourceExhausted("mock rate limit", retry_after_ms=500)

        ns = spec.namespace

        # Require namespace to exist (no implicit creation)
        if ns not in self._store.namespaces:
            raise BadRequest(f"unknown namespace '{ns}'")

        caps = await self.capabilities()
        if caps.max_batch_size and len(spec.vectors) > caps.max_batch_size:
            # suggested_batch_reduction: how many items to remove to meet the limit
            reduction = len(spec.vectors) - caps.max_batch_size
            reduction = max(1, reduction)
            raise BadRequest(
                f"batch size {len(spec.vectors)} exceeds maximum of {caps.max_batch_size}",
                details={"max_batch_size": caps.max_batch_size, "namespace": ns},
                suggested_batch_reduction=reduction,
            )

        dims = self._store.namespaces[ns].dimensions
        bucket = self._store.data.setdefault(ns, {})

        upserted = 0
        failures: List[Dict[str, Any]] = []

        # Partial-failure semantics: per-item dimension check vs namespace schema
        for i, v in enumerate(spec.vectors):
            if len(v.vector) != dims:
                failures.append(
                    {
                        "id": str(v.id),
                        "index": i,
                        "code": "DIMENSION_MISMATCH",
                        "message": f"expected {dims}, got {len(v.vector)}",
                        "details": {
                            "expected": dims,
                            "actual": len(v.vector),
                            "namespace": ns,
                        },
                    }
                )
                continue
            try:
                bucket[str(v.id)] = Vector(
                    id=str(v.id),
                    vector=list(v.vector),
                    metadata=dict(v.metadata or {}),
                    namespace=ns,
                )
                upserted += 1
            except Exception as e:
                # Defensive: normalize any unexpected failure into a consistent item failure record
                failures.append(
                    {
                        "id": str(v.id),
                        "index": i,
                        "code": "UNAVAILABLE",
                        "message": str(e),
                        "details": {"namespace": ns},
                    }
                )

        await asyncio.sleep(0.005)
        return UpsertResult(
            upserted_count=upserted,
            failed_count=len(failures),
            failures=failures,
        )

    # ------------------------------ delete --------------------------------- #

    async def _do_delete(
        self,
        spec: DeleteSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> DeleteResult:
        ns = spec.namespace
        if ns not in self._store.namespaces:
            # Explicit namespace validation (tests expect BAD_REQUEST here)
            raise BadRequest(f"unknown namespace '{ns}'")

        caps = await self.capabilities()
        if spec.ids and caps.max_batch_size and len(spec.ids) > caps.max_batch_size:
            reduction = len(spec.ids) - caps.max_batch_size
            reduction = max(1, reduction)
            raise BadRequest(
                f"batch size {len(spec.ids)} exceeds maximum of {caps.max_batch_size}",
                details={"max_batch_size": caps.max_batch_size, "namespace": ns},
                suggested_batch_reduction=reduction,
            )

        if spec.filter and caps.max_filter_terms and len(spec.filter) > caps.max_filter_terms:
            raise BadRequest(
                f"filter too complex: {len(spec.filter)} terms exceeds {caps.max_filter_terms}",
                details={
                    "provided_terms": len(spec.filter),
                    "max_terms": caps.max_filter_terms,
                    "namespace": ns,
                },
            )

        bucket = self._store.data.get(ns, {})
        deleted = 0
        failures: List[Dict[str, Any]] = []

        if spec.ids:
            # Idempotent: deleting unknown IDs is a no-op, not a failure
            for vid in spec.ids:
                key = str(vid)
                if key in bucket:
                    del bucket[key]
                    deleted += 1
        elif spec.filter:
            # Delete by simple equality filter
            to_delete = [vid for vid, v in list(bucket.items()) if _filter_match(v.metadata, spec.filter)]
            for vid in to_delete:
                del bucket[vid]
                deleted += 1
        else:
            # Should never reach here because BaseVectorAdapter validates ids|filter
            raise BadRequest("must provide either ids or filter for deletion")

        await asyncio.sleep(0.002)
        return DeleteResult(
            deleted_count=deleted,
            failed_count=len(failures),
            failures=failures,
        )

    # ------------------------- namespace management ------------------------ #

    async def _do_create_namespace(
        self,
        spec: NamespaceSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> NamespaceResult:
        metric = spec.distance_metric
        if metric not in SUPPORTED_DISTANCE_METRICS:
            raise NotSupported(
                "distance_metric must be one of: "
                + ", ".join(SUPPORTED_DISTANCE_METRICS)
            )

        self._store.ensure_namespace(spec)
        await asyncio.sleep(0.001)
        return NamespaceResult(
            success=True,
            namespace=spec.namespace,
            details={"created": True},
        )

    async def _do_delete_namespace(
        self,
        namespace: str,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> NamespaceResult:
        existed = namespace in self._store.namespaces
        self._store.namespaces.pop(namespace, None)
        self._store.data.pop(namespace, None)
        await asyncio.sleep(0.001)
        return NamespaceResult(
            success=True,
            namespace=namespace,
            details={"existed": existed},
        )

    # -------------------------------- health ------------------------------- #

    async def _do_health(
        self,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> Dict[str, Any]:
        """
        Simulate quick health probe with namespace summary.

        ctx.attrs["health"] == "degraded" forces a degraded state while preserving
        the response shape for conformance and observability tests.
        """
        degraded = bool(ctx and ctx.attrs.get("health") == "degraded")
        ns_status = "degraded" if degraded else "ok"

        await asyncio.sleep(0.001)
        return {
            "ok": not degraded,
            "server": self.name,
            "version": "1.0.0",
            "namespaces": {
                ns: {
                    "dimensions": info.dimensions,
                    "metric": info.distance_metric,
                    "count": len(self._store.data.get(ns, {})),
                    "status": ns_status,
                }
                for ns, info in self._store.namespaces.items()
            },
        }


# ------------------------------ demo (optional) ---------------------------- #

if __name__ == "__main__":

    async def _demo() -> None:
        adapter = MockVectorAdapter()
        # Create a namespace
        await adapter.create_namespace(
            NamespaceSpec(namespace="demo", dimensions=3, distance_metric=METRIC_COSINE)
        )

        # Upsert a few vectors
        vecs = [
            Vector(
                id="a",
                vector=[1.0, 0.0, 0.0],
                metadata={"label": "alpha"},
                namespace="demo",
            ),
            Vector(
                id="b",
                vector=[0.0, 1.0, 0.0],
                metadata={"label": "beta"},
                namespace="demo",
            ),
            Vector(
                id="c",
                vector=[0.7, 0.7, 0.0],
                metadata={"label": "gamma"},
                namespace="demo",
            ),
        ]
        await adapter.upsert(UpsertSpec(vectors=vecs, namespace="demo"))

        # Query
        res = await adapter.query(
            QuerySpec(vector=[0.8, 0.6, 0.0], top_k=2, namespace="demo")
        )
        print("Top matches:")
        for m in res.matches:
            print(
                f"- id={m.vector.id} "
                f"score={m.score:.3f} "
                f"distance={m.distance:.3f} "
                f"meta={m.vector.metadata}"
            )

    asyncio.run(_demo())
