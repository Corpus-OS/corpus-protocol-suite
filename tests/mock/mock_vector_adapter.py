# tests/vector/mock_vector_adapter.py
# SPDX-License-Identifier: Apache-2.0
"""
Mock Vector adapter used in Corpus SDK example scripts.

Implements BaseVectorAdapter methods for demonstration purposes only.
Simulates an in-memory vector store with basic filtering and metrics-friendly behavior.

Alignment goals (conformance + BaseVectorAdapter coverage):
- Implements all required backend hooks, including batch_query.
- Avoids re-entering BaseVectorAdapter public APIs from inside _do_* methods.
- Uses canonical error taxonomy/codes; nuanced conditions go in details/retry_after_ms.
- Uses VectorID consistently to avoid type drift between mock and base.
- Provides deterministic ctx-driven knobs to exercise deadlines, breaker behavior, and error paths.
- Enforces the same namespace semantics as BaseVectorAdapter even when _do_* is called directly.
- Uses percentage-based suggested_batch_reduction to align with BaseVectorAdapter semantics.
- Supports a minimal richer filter dialect: equality + {"k": {"$in": [...]}}.
- Validates filter dialect strictly (unknown operator raises BadRequest upstream).
- Sets text_storage_strategy explicitly to avoid ambiguous capability semantics.
- Contains randomness behind a seeded RNG when failure_rate > 0 for reproducible demos.
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
    VectorID,
    Vector,
    VectorMatch,
    QueryResult,
    UpsertResult,
    DeleteResult,
    NamespaceResult,
    QuerySpec,
    BatchQuerySpec,
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

# Keep capabilities consistent with backend behavior (do not call capabilities() inside _do_*).
_CAP_MAX_DIMENSIONS = 2048
_CAP_MAX_BATCH_SIZE = 1000
_CAP_MAX_TOP_K = 1000
_CAP_MAX_FILTER_TERMS = 10

# Tiny guard to prevent accidental filter explosions via huge "$in" lists.
# This is a mock-specific safety limit used for deterministic tests/examples.
_CAP_MAX_IN_LIST = 256

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
    Minimal filter dialect (mock):

    - Equality: {"k": "v"}
    - IN: {"k": {"$in": [v1, v2, ...]}}

    IMPORTANT:
      Filter shape/operator validation is performed upstream (see _validate_filter_dialect).
      This matcher assumes the dialect is valid. If an unknown operator appears anyway,
      it is treated as "no match" defensively, but tests should never rely on that path.
    """
    if not flt:
        return True
    if not meta:
        return False

    for k, v in flt.items():
        # Operator form: {"k": {"$in": [...]}}
        if isinstance(v, dict):
            if "$in" in v:
                allowed = v.get("$in")
                if not isinstance(allowed, list):
                    return False
                if meta.get(k) not in allowed:
                    return False
                continue
            # Unknown operator (should have been rejected upstream)
            return False

        # Equality form
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
    # ns -> id(str) -> Vector
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

    Deterministic knobs (ctx.attrs) to exercise BaseVectorAdapter behavior:
      - sleep_ms: int  -> adds an await to simulate backend latency (deadlines/breaker tests)
      - fail: str      -> forces a typed error. Supported:
                         "unavailable", "resource_exhausted", "index_not_ready", "not_supported"
      - retry_after_ms: int -> used with forced failures when applicable

    Health knob alignment:
      - _do_health honors only fail=unavailable (to test wire/base normalization for health
        without turning health into a general chaos endpoint).

    Namespace semantics alignment:
      - UpsertSpec.namespace is authoritative; if Vector.namespace is present it MUST match.
        In _do_upsert, any mismatch triggers a request-level BadRequest.

      - BatchQuerySpec.namespace is authoritative; QuerySpec.namespace MUST match.
        In _do_batch_query, any mismatch triggers BadRequest.

    Batch behavior:
      - All-or-nothing: if any query is invalid (dimension mismatch, filter too complex, etc.),
        the entire batch raises an error deterministically.

    Upsert behavior (atomic for validation errors):
      - Any namespace mismatch or dimension mismatch is request-level (atomic) to align with strictness.

    Include flags:
      - When include_vectors=False, Vector.vector is returned as [] (empty list). This is intentional:
        the type is List[float], so [] is the canonical "not included" representation.
    """

    def __init__(
        self,
        name: str = "mock-vector",
        failure_rate: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.name = name
        self.failure_rate = float(failure_rate)
        self._store = _MemoryStore()

        # Seeded RNG for reproducible demo failures
        seed_env = os.getenv("VECTOR_RANDOM_SEED")
        if seed_env is not None:
            try:
                seed = int(seed_env)
            except Exception:
                seed = 0
        else:
            seed = sum((i + 1) * ord(c) for i, c in enumerate(self.name)) & 0xFFFFFFFF
        self._rng = random.Random(seed)

        # Optional seeding logic (demo-only)
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
                        id=VectorID("seed1"),
                        vector=[1.0, 0.0],
                        metadata={"label": "alpha"},
                        namespace="default",
                    ),
                    "seed2": Vector(
                        id=VectorID("seed2"),
                        vector=[0.0, 1.0],
                        metadata={"label": "beta"},
                        namespace="default",
                    ),
                }

    # ------------------------------- helpers -------------------------------- #

    @staticmethod
    def _ctx_attrs(ctx: Optional[OperationContext]) -> Dict[str, Any]:
        if ctx is None or ctx.attrs is None:
            return {}
        try:
            return dict(ctx.attrs)
        except Exception:
            return {}

    async def _maybe_sleep(self, ctx: Optional[OperationContext]) -> None:
        attrs = self._ctx_attrs(ctx)
        sleep_ms = attrs.get("sleep_ms")
        if sleep_ms is None:
            return
        try:
            ms = int(sleep_ms)
            if ms > 0:
                await asyncio.sleep(ms / 1000.0)
        except Exception:
            return

    def _maybe_fail(self, ctx: Optional[OperationContext]) -> None:
        attrs = self._ctx_attrs(ctx)
        mode = attrs.get("fail")
        if not mode:
            return

        retry_after_ms = attrs.get("retry_after_ms")
        ra: Optional[int] = None
        if retry_after_ms is not None:
            try:
                ra = max(0, int(retry_after_ms))
            except Exception:
                ra = None

        m = str(mode).strip().lower()
        if m == "unavailable":
            raise Unavailable(
                "mock backend unavailable",
                retry_after_ms=ra,
                details={"reason": "forced"},
            )
        if m == "resource_exhausted":
            raise ResourceExhausted(
                "mock resource exhausted",
                retry_after_ms=ra,
                details={"reason": "forced"},
            )
        if m == "index_not_ready":
            raise IndexNotReady(
                "mock index not ready",
                retry_after_ms=ra if ra is not None else 500,
                details={"reason": "forced"},
            )
        if m == "not_supported":
            raise NotSupported(
                "mock feature not supported",
                details={"reason": "forced"},
            )

    def _maybe_fail_health(self, ctx: Optional[OperationContext]) -> None:
        """
        Health-specific failure injection.

        High-ROI conformance knob:
          - honors only fail=unavailable (others ignored) to keep health stable and simple,
            while still enabling testing of wire error envelopes and base error normalization.
        """
        attrs = self._ctx_attrs(ctx)
        mode = attrs.get("fail")
        if not mode:
            return
        if str(mode).strip().lower() != "unavailable":
            return

        retry_after_ms = attrs.get("retry_after_ms")
        ra: Optional[int] = None
        if retry_after_ms is not None:
            try:
                ra = max(0, int(retry_after_ms))
            except Exception:
                ra = None

        raise Unavailable(
            "mock backend unavailable",
            retry_after_ms=ra,
            details={"reason": "forced", "op": "health"},
        )

    def _random_failure(self) -> bool:
        """
        Optional non-deterministic failure mode for demos.

        If failure_rate > 0, failures are deterministic with self._rng for reproducibility.
        Conformance runs should use failure_rate=0.0 or ctx.attrs["fail"] for determinism.
        """
        if self.failure_rate <= 0.0:
            return False
        try:
            return self._rng.random() < self.failure_rate
        except Exception:
            return False

    def _require_namespace_exists(self, namespace: str) -> None:
        if namespace not in self._store.namespaces:
            raise BadRequest(f"unknown namespace '{namespace}'")

    def _require_index_ready(self, namespace: str) -> None:
        if not self._store.data.get(namespace):
            raise IndexNotReady(
                "index not ready (no data in namespace)",
                retry_after_ms=500,
                details={"namespace": namespace},
            )

    @staticmethod
    def _suggested_batch_reduction_percent(requested: int, maximum: int) -> Optional[int]:
        """
        Align with BaseVectorAdapter semantics:
            suggested_batch_reduction is a PERCENTAGE hint (0..100).
        """
        try:
            if requested <= 0:
                return None
            if maximum < 0:
                return None
            if requested <= maximum:
                return 0
            return int(100 * (requested - maximum) / requested)
        except Exception:
            return None

    def _validate_filter_dialect(self, flt: Optional[Dict[str, Any]], *, namespace: str) -> None:
        """
        Validate the filter dialect strictly.

        Requirements:
          - flt is a dict (enforced upstream by BaseVectorAdapter, but validated here for direct hook calls)
          - Supported forms:
              * {"k": <scalar>}            equality
              * {"k": {"$in": [..]}}       IN operator
          - Unknown operators raise BadRequest (do not silently filter everything out).
          - $in list size is bounded to prevent accidental explosion in examples/tests.
        """
        if flt is None:
            return
        if not isinstance(flt, dict):
            raise BadRequest(
                "filter must be an object",
                details={"namespace": namespace, "type": type(flt).__name__},
            )

        for k, v in flt.items():
            if isinstance(v, dict):
                # Only supported operator is "$in"
                unknown_ops = [op for op in v.keys() if op != "$in"]
                if unknown_ops:
                    raise BadRequest(
                        "unsupported filter operator",
                        details={
                            "namespace": namespace,
                            "field": k,
                            "operator": unknown_ops[0],
                            "supported": ["$in"],
                        },
                    )
                if "$in" not in v:
                    # dict with no supported operators
                    raise BadRequest(
                        "unsupported filter operator",
                        details={
                            "namespace": namespace,
                            "field": k,
                            "supported": ["$in"],
                        },
                    )

                allowed = v.get("$in")
                if not isinstance(allowed, list):
                    raise BadRequest(
                        "invalid '$in' operand",
                        details={
                            "namespace": namespace,
                            "field": k,
                            "expected": "list",
                            "type": type(allowed).__name__,
                        },
                    )
                if len(allowed) > _CAP_MAX_IN_LIST:
                    raise BadRequest(
                        "filter '$in' list too large",
                        details={
                            "namespace": namespace,
                            "field": k,
                            "max_in": _CAP_MAX_IN_LIST,
                            "provided": len(allowed),
                        },
                    )

    def _enforce_filter_complexity(self, flt: Optional[Dict[str, Any]], *, namespace: str) -> None:
        """
        Backend-specific filter validation + complexity guard.

        Counting rule (simple & deterministic):
          - Each top-level filter key counts as 1 term, regardless of whether it is equality or $in.
        """
        self._validate_filter_dialect(flt, namespace=namespace)

        if flt and _CAP_MAX_FILTER_TERMS and len(flt) > _CAP_MAX_FILTER_TERMS:
            raise BadRequest(
                f"filter too complex: {len(flt)} terms exceeds {_CAP_MAX_FILTER_TERMS}",
                details={
                    "provided_terms": len(flt),
                    "max_terms": _CAP_MAX_FILTER_TERMS,
                    "namespace": namespace,
                },
            )

    def _score_candidates(
        self,
        *,
        query_vector: List[float],
        namespace: str,
        flt: Optional[Dict[str, Any]],
    ) -> Tuple[int, List[Tuple[float, float, Vector]]]:
        ns_info = self._store.namespaces[namespace]
        bucket = self._store.data.get(namespace, {})
        candidates = [v for v in bucket.values() if _filter_match(v.metadata, flt)]

        scored: List[Tuple[float, float, Vector]] = []
        for v in candidates:
            if ns_info.distance_metric == METRIC_COSINE:
                sim = _cosine_sim(query_vector, v.vector)
                distance = 1.0 - sim
                score = sim
            elif ns_info.distance_metric == METRIC_EUCLIDEAN:
                distance = _euclidean(query_vector, v.vector)
                score = 1.0 / (1.0 + distance)
            elif ns_info.distance_metric == METRIC_DOTPRODUCT:
                dp = _dot(query_vector, v.vector)
                distance = -dp
                score = dp
            else:
                raise NotSupported(
                    f"distance_metric '{ns_info.distance_metric}' not supported",
                    details={"namespace": namespace},
                )
            scored.append((float(score), float(distance), v))

        scored.sort(key=lambda t: t[0], reverse=True)
        return (len(scored), scored)

    def _render_matches(
        self,
        *,
        scored: List[Tuple[float, float, Vector]],
        top_k: int,
        include_vectors: bool,
        include_metadata: bool,
    ) -> List[VectorMatch]:
        top = scored[:top_k]
        matches: List[VectorMatch] = []

        for score, distance, v in top:
            # Contract: Vector.vector is List[float], so [] is the canonical "not included" value.
            out_vec = list(v.vector) if include_vectors else []
            out_meta = dict(v.metadata) if (include_metadata and v.metadata is not None) else None
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
        return matches

    # --------------------------- capability probe --------------------------- #

    async def _do_capabilities(self) -> VectorCapabilities:
        await asyncio.sleep(0.01)
        return VectorCapabilities(
            server=self.name,
            version="1.0.0",
            max_dimensions=_CAP_MAX_DIMENSIONS,
            supported_metrics=SUPPORTED_DISTANCE_METRICS,
            supports_namespaces=True,
            supports_metadata_filtering=True,
            supports_batch_operations=True,
            max_batch_size=_CAP_MAX_BATCH_SIZE,
            supports_index_management=True,
            idempotent_writes=False,
            supports_multi_tenant=False,
            supports_deadline=True,
            max_top_k=_CAP_MAX_TOP_K,
            max_filter_terms=_CAP_MAX_FILTER_TERMS,
            supports_batch_queries=True,
            text_storage_strategy="none",
        )

    # ------------------------------ query ---------------------------------- #

    async def _do_query(
        self,
        spec: QuerySpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> QueryResult:
        self._maybe_fail(ctx)
        await self._maybe_sleep(ctx)

        if self._random_failure():
            raise Unavailable(
                "mock backend unavailable",
                retry_after_ms=200,
                details={"reason": "random_failure"},
            )

        self._require_namespace_exists(spec.namespace)
        self._enforce_filter_complexity(spec.filter, namespace=spec.namespace)

        ns_info = self._store.namespaces[spec.namespace]
        if len(spec.vector) != ns_info.dimensions:
            raise DimensionMismatch(
                f"query vector dimension {len(spec.vector)} does not match namespace {ns_info.dimensions}",
                details={
                    "expected": ns_info.dimensions,
                    "actual": len(spec.vector),
                    "namespace": spec.namespace,
                },
            )

        self._require_index_ready(spec.namespace)

        total, scored = self._score_candidates(
            query_vector=list(spec.vector),
            namespace=spec.namespace,
            flt=spec.filter,
        )

        matches = self._render_matches(
            scored=scored,
            top_k=spec.top_k,
            include_vectors=bool(spec.include_vectors),
            include_metadata=bool(spec.include_metadata),
        )

        await asyncio.sleep(0.01)

        return QueryResult(
            matches=matches,
            query_vector=list(spec.vector),
            namespace=spec.namespace,
            total_matches=total,
        )

    # --------------------------- batch query -------------------------------- #

    async def _do_batch_query(
        self,
        spec: BatchQuerySpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> List[QueryResult]:
        """
        Execute multiple similarity queries in a single backend call.

        Behavior (deterministic, all-or-nothing):
          - If any query is invalid (dimension mismatch, filter too complex, namespace mismatch),
            this method raises the corresponding error for the entire batch.
        """
        self._maybe_fail(ctx)
        await self._maybe_sleep(ctx)

        if self._random_failure():
            raise Unavailable(
                "mock backend unavailable",
                retry_after_ms=200,
                details={"reason": "random_failure"},
            )

        self._require_namespace_exists(spec.namespace)
        self._require_index_ready(spec.namespace)

        ns_info = self._store.namespaces[spec.namespace]

        results: List[QueryResult] = []
        for i, q in enumerate(spec.queries):
            if q.namespace != spec.namespace:
                raise BadRequest(
                    f"query[{i}].namespace must match batch namespace",
                    details={
                        "index": i,
                        "batch_namespace": spec.namespace,
                        "query_namespace": q.namespace,
                    },
                )

            self._enforce_filter_complexity(q.filter, namespace=spec.namespace)

            if len(q.vector) != ns_info.dimensions:
                raise DimensionMismatch(
                    f"query[{i}] vector dimension {len(q.vector)} does not match namespace {ns_info.dimensions}",
                    details={
                        "index": i,
                        "expected": ns_info.dimensions,
                        "actual": len(q.vector),
                        "namespace": spec.namespace,
                    },
                )

            total, scored = self._score_candidates(
                query_vector=list(q.vector),
                namespace=spec.namespace,
                flt=q.filter,
            )

            matches = self._render_matches(
                scored=scored,
                top_k=q.top_k,
                include_vectors=bool(q.include_vectors),
                include_metadata=bool(q.include_metadata),
            )

            results.append(
                QueryResult(
                    matches=matches,
                    query_vector=list(q.vector),
                    namespace=spec.namespace,
                    total_matches=total,
                )
            )

        await asyncio.sleep(0.01)
        return results

    # ------------------------------ upsert --------------------------------- #

    async def _do_upsert(
        self,
        spec: UpsertSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> UpsertResult:
        self._maybe_fail(ctx)
        await self._maybe_sleep(ctx)

        if self._random_failure():
            raise ResourceExhausted(
                "mock resource exhausted",
                retry_after_ms=500,
                details={"reason": "random_failure"},
            )

        ns = spec.namespace
        self._require_namespace_exists(ns)

        if _CAP_MAX_BATCH_SIZE and len(spec.vectors) > _CAP_MAX_BATCH_SIZE:
            suggested_pct = self._suggested_batch_reduction_percent(len(spec.vectors), _CAP_MAX_BATCH_SIZE)
            raise BadRequest(
                f"batch size {len(spec.vectors)} exceeds maximum of {_CAP_MAX_BATCH_SIZE}",
                details={"max_batch_size": _CAP_MAX_BATCH_SIZE, "namespace": ns},
                suggested_batch_reduction=suggested_pct,
            )

        # Namespace semantics enforcement (request-level)
        for i, v in enumerate(spec.vectors):
            if v.namespace is not None and v.namespace != ns:
                raise BadRequest(
                    "vector.namespace must match UpsertSpec.namespace",
                    details={
                        "index": i,
                        "spec_namespace": ns,
                        "vector_namespace": v.namespace,
                        "vector_id": str(v.id),
                    },
                )

        dims = self._store.namespaces[ns].dimensions

        # Dimension semantics enforcement (request-level, atomic)
        for i, v in enumerate(spec.vectors):
            if len(v.vector) != dims:
                raise DimensionMismatch(
                    f"vector dimension {len(v.vector)} does not match namespace {dims}",
                    details={
                        "index": i,
                        "expected": dims,
                        "actual": len(v.vector),
                        "namespace": ns,
                        "vector_id": str(v.id),
                    },
                )

        bucket = self._store.data.setdefault(ns, {})

        # Atomic write phase (no partial item failures for validation errors).
        try:
            for v in spec.vectors:
                bucket[str(v.id)] = Vector(
                    id=VectorID(str(v.id)),
                    vector=list(v.vector),
                    metadata=dict(v.metadata or {}),
                    namespace=ns,
                    text=None,
                )
        except Exception as e:
            # Defensive: if something unexpected fails, surface as a retryable backend failure.
            raise Unavailable(
                "mock backend write failed",
                details={"namespace": ns, "type": type(e).__name__},
            ) from e

        await asyncio.sleep(0.005)
        return UpsertResult(
            upserted_count=len(spec.vectors),
            failed_count=0,
            failures=[],
        )

    # ------------------------------ delete --------------------------------- #

    async def _do_delete(
        self,
        spec: DeleteSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> DeleteResult:
        self._maybe_fail(ctx)
        await self._maybe_sleep(ctx)

        ns = spec.namespace
        self._require_namespace_exists(ns)

        if spec.ids and _CAP_MAX_BATCH_SIZE and len(spec.ids) > _CAP_MAX_BATCH_SIZE:
            suggested_pct = self._suggested_batch_reduction_percent(len(spec.ids), _CAP_MAX_BATCH_SIZE)
            raise BadRequest(
                f"batch size {len(spec.ids)} exceeds maximum of {_CAP_MAX_BATCH_SIZE}",
                details={"max_batch_size": _CAP_MAX_BATCH_SIZE, "namespace": ns},
                suggested_batch_reduction=suggested_pct,
            )

        self._enforce_filter_complexity(spec.filter, namespace=ns)

        bucket = self._store.data.get(ns, {})
        deleted = 0

        if spec.ids:
            for vid in spec.ids:
                key = str(vid)
                if key in bucket:
                    del bucket[key]
                    deleted += 1
        elif spec.filter:
            to_delete = [vid for vid, v in list(bucket.items()) if _filter_match(v.metadata, spec.filter)]
            for vid in to_delete:
                del bucket[vid]
                deleted += 1
        else:
            raise BadRequest("must provide either ids or filter for deletion")

        await asyncio.sleep(0.002)
        return DeleteResult(
            deleted_count=deleted,
            failed_count=0,
            failures=[],
        )

    # ------------------------- namespace management ------------------------ #

    async def _do_create_namespace(
        self,
        spec: NamespaceSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> NamespaceResult:
        self._maybe_fail(ctx)
        await self._maybe_sleep(ctx)

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
        self._maybe_fail(ctx)
        await self._maybe_sleep(ctx)

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

        Failure knob alignment:
          - honors only fail=unavailable (allows testing vector.health error normalization).
        """
        self._maybe_fail_health(ctx)
        await self._maybe_sleep(ctx)

        attrs = self._ctx_attrs(ctx)
        degraded = bool(attrs.get("health") == "degraded")
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
        await adapter.create_namespace(
            NamespaceSpec(namespace="demo", dimensions=3, distance_metric=METRIC_COSINE)
        )

        vecs = [
            Vector(
                id=VectorID("a"),
                vector=[1.0, 0.0, 0.0],
                metadata={"label": "alpha"},
                namespace="demo",
            ),
            Vector(
                id=VectorID("b"),
                vector=[0.0, 1.0, 0.0],
                metadata={"label": "beta"},
                namespace="demo",
            ),
            Vector(
                id=VectorID("c"),
                vector=[0.7, 0.7, 0.0],
                metadata={"label": "gamma"},
                namespace="demo",
            ),
        ]
        await adapter.upsert(UpsertSpec(vectors=vecs, namespace="demo"))

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

        # Demonstrate $in filter
        res2 = await adapter.query(
            QuerySpec(
                vector=[0.8, 0.6, 0.0],
                top_k=3,
                namespace="demo",
                filter={"label": {"$in": ["alpha", "gamma"]}},
            )
        )
        print("\nFiltered matches ($in):")
        for m in res2.matches:
            print(f"- id={m.vector.id} label={m.vector.metadata.get('label') if m.vector.metadata else None}")

        # Demonstrate strict operator validation (will raise BadRequest)
        try:
            await adapter.query(
                QuerySpec(
                    vector=[0.8, 0.6, 0.0],
                    top_k=3,
                    namespace="demo",
                    filter={"label": {"$unknown": ["alpha"]}},
                )
            )
        except BadRequest as e:
            print("\nUnknown operator rejected:", e)

        bres = await adapter.batch_query(
            BatchQuerySpec(
                namespace="demo",
                queries=[
                    QuerySpec(vector=[1.0, 0.0, 0.0], top_k=2, namespace="demo"),
                    QuerySpec(vector=[0.0, 1.0, 0.0], top_k=2, namespace="demo"),
                ],
            )
        )
        print("\nBatch matches:")
        for i, qr in enumerate(bres):
            print(f"Query {i}: total={qr.total_matches}")
            for m in qr.matches:
                print(f"  - id={m.vector.id} score={m.score:.3f}")

    asyncio.run(_demo())
