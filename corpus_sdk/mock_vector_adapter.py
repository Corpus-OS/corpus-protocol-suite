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
    if not flt:
        return True
    if not meta:
        return False
    # simple equality-only filter for demo purposes
    for k, v in flt.items():
        if meta.get(k) != v:
            return False
    return True


# ------------------------------- mock state -------------------------------- #

@dataclass
class _NamespaceInfo:
    dimensions: int
    distance_metric: str = "cosine"  # "cosine" | "euclidean" | "dotproduct"


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
                dimensions=spec.dimensions, distance_metric=spec.distance_metric
            )
            self.data.setdefault(spec.namespace, {})

    def infer_or_validate_dims(self, ns: str, vec_len: int) -> None:
        """
        If namespace exists, validate length; if not, infer a default namespace with cosine metric.
        """
        if ns in self.namespaces:
            dims = self.namespaces[ns].dimensions
            if vec_len != dims:
                raise DimensionMismatch(f"vector dimension {vec_len} does not match namespace {dims}")
        else:
            # infer namespace with provided vector length
            self.namespaces[ns] = _NamespaceInfo(dimensions=vec_len, distance_metric="cosine")
            self.data.setdefault(ns, {})


# ----------------------------- adapter class ------------------------------- #

@dataclass
class MockVectorAdapter(BaseVectorAdapter):
    """
    A mock Vector adapter for protocol demonstrations.
    """
    name: str = "mock-vector"
    failure_rate: float = 0.05  # chance to simulate transient backend issues
    _store: _MemoryStore = field(default_factory=_MemoryStore)

    # --------------------------- capability probe --------------------------- #

    async def _do_capabilities(self) -> VectorCapabilities:
        # tiny simulated network delay
        await asyncio.sleep(0.01)
        return VectorCapabilities(
            server="mock-vector",
            version="1.0.0",
            max_dimensions=2048,
            supported_metrics=("cosine", "euclidean", "dotproduct"),
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

    async def _do_query(self, spec: QuerySpec, *, ctx: Optional[OperationContext] = None) -> QueryResult:
        # Simulate occasional transient overloads
        if random.random() < self.failure_rate:
            raise Unavailable("mock backend overloaded", code="OVERLOADED")

        # Namespace must exist and be non-empty
        if spec.namespace not in self._store.namespaces or not self._store.data.get(spec.namespace):
            # If namespace exists but has no data, surface INDEX_NOT_READY
            if spec.namespace in self._store.namespaces:
                raise IndexNotReady("index not ready (no data in namespace)")
            raise BadRequest(f"unknown namespace '{spec.namespace}'")

        ns_info = self._store.namespaces[spec.namespace]
        if len(spec.vector) != ns_info.dimensions:
            raise DimensionMismatch(
                f"query vector dimension {len(spec.vector)} does not match namespace {ns_info.dimensions}"
            )

        # Gather candidates by filter
        candidates = [
            v for v in self._store.data[spec.namespace].values()
            if _filter_match(v.metadata, spec.filter)
        ]

        # Score with chosen metric
        scored: List[Tuple[float, float, Vector]] = []
        for v in candidates:
            if ns_info.distance_metric == "cosine":
                sim = _cosine_sim(spec.vector, v.vector)
                distance = 1.0 - sim
                score = sim
            elif ns_info.distance_metric == "euclidean":
                distance = _euclidean(spec.vector, v.vector)
                # Convert to a similarity-ish score (bounded, higher is better)
                score = 1.0 / (1.0 + distance)
            else:  # "dotproduct"
                dp = _dot(spec.vector, v.vector)
                # Distance as negative similarity to keep "lower is better"
                distance = -dp
                score = dp

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
                    vector=Vector(id=v.id, vector=out_vec, metadata=out_meta, namespace=v.namespace),
                    score=float(score),
                    distance=float(distance),
                )
            )

        # tiny simulated latency
        await asyncio.sleep(0.01)

        return QueryResult(
            matches=matches,
            query_vector=list(spec.vector),
            namespace=spec.namespace,
            total_matches=len(scored),
        )

    # ------------------------------ upsert --------------------------------- #

    async def _do_upsert(self, spec: UpsertSpec, *, ctx: Optional[OperationContext] = None) -> UpsertResult:
        if random.random() < self.failure_rate:
            raise ResourceExhausted("mock rate limit", retry_after_ms=500)

        ns = spec.namespace
        if not spec.vectors:
            raise BadRequest("vectors must not be empty")

        # infer/validate namespace dims
        self._store.infer_or_validate_dims(ns, len(spec.vectors[0].vector))

        dims = self._store.namespaces[ns].dimensions
        bucket = self._store.data.setdefault(ns, {})

        upserted = 0
        failures: List[Dict[str, Any]] = []

        for v in spec.vectors:
            if len(v.vector) != dims:
                failures.append(
                    {"id": v.id, "error": "DimensionMismatch", "detail": f"expected {dims}, got {len(v.vector)}"}
                )
                continue
            # naive upsert
            bucket[str(v.id)] = Vector(id=str(v.id), vector=list(v.vector), metadata=dict(v.metadata or {}), namespace=ns)
            upserted += 1

        await asyncio.sleep(0.005)
        return UpsertResult(
            upserted_count=upserted,
            failed_count=len(failures),
            failures=failures,
        )

    # ------------------------------ delete --------------------------------- #

    async def _do_delete(self, spec: DeleteSpec, *, ctx: Optional[OperationContext] = None) -> DeleteResult:
        ns = spec.namespace
        if ns not in self._store.data:
            # Deleting from a non-existent namespace: treat as OK, nothing to delete
            return DeleteResult(deleted_count=0, failed_count=0, failures=[])

        bucket = self._store.data[ns]
        deleted = 0
        failures: List[Dict[str, Any]] = []

        if spec.ids:
            for vid in spec.ids:
                if str(vid) in bucket:
                    del bucket[str(vid)]
                    deleted += 1
                else:
                    failures.append({"id": str(vid), "error": "NotFound"})
        elif spec.filter:
            # Delete by simple equality filter
            to_delete = [vid for vid, v in bucket.items() if _filter_match(v.metadata, spec.filter)]
            for vid in to_delete:
                del bucket[vid]
                deleted += 1
        else:
            # Should never reach here because BaseVectorAdapter validates ids|filter
            raise BadRequest("must provide either ids or filter for deletion")

        await asyncio.sleep(0.002)
        return DeleteResult(deleted_count=deleted, failed_count=len(failures), failures=failures)

    # ------------------------- namespace management ------------------------ #

    async def _do_create_namespace(self, spec: NamespaceSpec, *, ctx: Optional[OperationContext] = None) -> NamespaceResult:
        metric = spec.distance_metric
        if metric not in ("cosine", "euclidean", "dotproduct"):
            raise NotSupported("distance_metric must be one of: cosine, euclidean, dotproduct")

        self._store.ensure_namespace(spec)
        await asyncio.sleep(0.001)
        return NamespaceResult(success=True, namespace=spec.namespace, details={"created": True})

    async def _do_delete_namespace(self, namespace: str, *, ctx: Optional[OperationContext] = None) -> NamespaceResult:
        existed = namespace in self._store.namespaces
        self._store.namespaces.pop(namespace, None)
        self._store.data.pop(namespace, None)
        await asyncio.sleep(0.001)
        return NamespaceResult(success=True, namespace=namespace, details={"existed": existed})

    # -------------------------------- health ------------------------------- #

    async def _do_health(self, *, ctx: Optional[OperationContext] = None) -> Dict[str, Any]:
        # Simulate quick health probe with namespace summary
        await asyncio.sleep(0.001)
        return {
            "ok": True,
            "server": "mock-vector",
            "version": "1.0.0",
            "namespaces": {
                ns: {
                    "dimensions": info.dimensions,
                    "metric": info.distance_metric,
                    "count": len(self._store.data.get(ns, {})),
                }
                for ns, info in self._store.namespaces.items()
            },
        }


# ------------------------------ demo (optional) ---------------------------- #

if __name__ == "__main__":
    async def _demo() -> None:
        adapter = MockVectorAdapter()
        # Create a namespace
        await adapter.create_namespace(NamespaceSpec(namespace="demo", dimensions=3, distance_metric="cosine"))

        # Upsert a few vectors
        vecs = [
            Vector(id="a", vector=[1.0, 0.0, 0.0], metadata={"label": "alpha"}, namespace="demo"),
            Vector(id="b", vector=[0.0, 1.0, 0.0], metadata={"label": "beta"}, namespace="demo"),
            Vector(id="c", vector=[0.7, 0.7, 0.0], metadata={"label": "gamma"}, namespace="demo"),
        ]
        await adapter.upsert(UpsertSpec(vectors=vecs, namespace="demo"))

        # Query
        res = await adapter.query(QuerySpec(vector=[0.8, 0.6, 0.0], top_k=2, namespace="demo"))
        print("Top matches:")
        for m in res.matches:
            print(f"- id={m.vector.id} score={m.score:.3f} distance={m.distance:.3f} meta={m.vector.metadata}")

    asyncio.run(_demo())
