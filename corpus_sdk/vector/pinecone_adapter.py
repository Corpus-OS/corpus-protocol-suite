# corpus_sdk/vector/pinecone_adapter.py
# SPDX-License-Identifier: Apache-2.0
"""
Pinecone Vector adapter for the Vector Protocol V1.1.

This module implements a production-grade adapter on top of the
`BaseVectorAdapter` / `VectorProtocolV1` contract, backed by Pinecone.

Goals
-----
- Map Vector Protocol → Pinecone index operations.
- Preserve async + backpressure semantics via BaseVectorAdapter.
- Normalize Pinecone errors into the vector error taxonomy.
- Support namespaces, metadata filtering, and batch upserts.
- Play nicely in "thin" (externally managed) and "standalone" modes.

Usage
-----
    from pinecone import Pinecone
    from corpus_sdk.vector.vector_base import Vector, UpsertSpec, QuerySpec
    from corpus_sdk.vector.pinecone_adapter import PineconeVectorAdapter

    pc = Pinecone(api_key="...")
    adapter = PineconeVectorAdapter(
        client=pc,
        index_name="my-index",
        metric="cosine",
        dimensions=1536,
        mode="standalone",  # or "thin"
    )

    # Upsert
    await adapter.upsert(
        UpsertSpec(
            namespace="demo",
            vectors=[
                Vector(id="a", vector=[...], metadata={"label": "alpha"}),
            ],
        )
    )

    # Query
    res = await adapter.query(
        QuerySpec(
            vector=[...],
            top_k=5,
            namespace="demo",
            filter={"label": "alpha"},
        )
    )
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Callable, TypeVar

from corpus_sdk.vector.vector_base import (
    BaseVectorAdapter,
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
    AuthError,
    BadRequest,
    DimensionMismatch,
    IndexNotReady,
    ResourceExhausted,
    TransientNetwork,
    Unavailable,
    VectorAdapterError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Try to import the modern Pinecone client.
try:  # pragma: no cover - import surface only
    import pinecone  # type: ignore
    from pinecone import Pinecone  # type: ignore[attr-defined]

    # Best-effort grab of core exception base type.
    try:
        from pinecone.core.client.exceptions import (  # type: ignore[attr-defined]
            PineconeException,
        )
    except Exception:  # pragma: no cover
        PineconeException = Exception  # type: ignore[assignment]
except Exception:  # pragma: no cover
    pinecone = None  # type: ignore[assignment]
    Pinecone = None  # type: ignore[assignment]
    PineconeException = Exception  # type: ignore[assignment]


class PineconeVectorAdapter(BaseVectorAdapter):
    """
    VectorProtocolV1 adapter backed by a Pinecone index.

    Design notes
    ------------
    - Async-first: all Pinecone calls run via `asyncio.to_thread`.
    - Does *not* manage index lifecycle; you create/delete indexes outside.
    - Namespaces are mapped directly to Pinecone namespaces.
    - Batch queries are implemented as parallel fan-out over single-query calls.

    text_storage_strategy
    ---------------------
    - This adapter does not itself manage text persistence.
    - BaseVectorAdapter handles text/docstore integration.
    - Here we only:
        * Validate the strategy value ("metadata", "docstore", "none").
        * Enforce that "docstore" requires a docstore instance.
        * Report the configured strategy via capabilities.
    """

    _component = "vector_pinecone"

    def __init__(
        self,
        *,
        index_name: str,
        client: Optional["Pinecone"] = None,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        project_name: Optional[str] = None,
        metric: str = "cosine",
        dimensions: Optional[int] = None,
        # Soft limits / planning hints
        max_batch_size: int = 100,
        max_top_k: int = 100,
        max_filter_terms: Optional[int] = None,
        text_storage_strategy: str = "metadata",  # "metadata", "docstore", or "none"
        batch_query_max_concurrency: Optional[int] = None,
        # BaseVectorAdapter infra
        metrics=None,
        mode: str = "thin",
        deadline_policy=None,
        breaker=None,
        cache=None,
        limiter=None,
        docstore=None,
        cache_query_ttl_s: Optional[int] = None,
        cache_caps_ttl_s: Optional[int] = None,
        warn_on_standalone_no_metrics: bool = True,
        config=None,
    ) -> None:
        if pinecone is None or Pinecone is None:
            raise RuntimeError(
                "PineconeVectorAdapter requires the `pinecone` Python package. "
                "Install via `pip install pinecone-client` (or the latest `pinecone`)."
            )

        metric = (metric or "cosine").lower().strip()
        if metric not in ("cosine", "euclidean", "dotproduct"):
            raise BadRequest(
                "metric must be one of: cosine, euclidean, dotproduct",
                code="BAD_CONFIG",
            )

        if not index_name or not isinstance(index_name, str):
            raise BadRequest(
                "index_name must be a non-empty string",
                code="BAD_CONFIG",
            )

        text_storage_strategy = (text_storage_strategy or "metadata").strip().lower()
        allowed_text_strategies = {"metadata", "docstore", "none"}
        if text_storage_strategy not in allowed_text_strategies:
            raise BadRequest(
                f"text_storage_strategy must be one of {sorted(allowed_text_strategies)}",
                code="BAD_CONFIG",
            )

        if text_storage_strategy == "docstore" and docstore is None:
            raise BadRequest(
                "text_storage_strategy='docstore' requires a docstore instance",
                code="BAD_CONFIG",
            )

        # Initialize client if not provided
        if client is None:
            api_key = api_key or os.getenv("PINECONE_API_KEY") or ""
            if not api_key:
                raise AuthError(
                    "PineconeVectorAdapter requires an API key "
                    "(pass api_key=... or set PINECONE_API_KEY).",
                    code="PINECONE_AUTH",
                )
            client_kwargs: Dict[str, Any] = {"api_key": api_key}
            if environment:
                # Modern Pinecone client takes project_name/env separately;
                # we pass through when provided.
                client_kwargs["environment"] = environment
            if project_name:
                client_kwargs["project_name"] = project_name

            client = Pinecone(**client_kwargs)  # type: ignore[call-arg]

        # Adapter-local configuration
        self._client: Pinecone = client  # type: ignore[assignment]
        self._index_name = index_name
        self._metric = metric
        self._dimensions = int(dimensions) if dimensions is not None else 0
        self._max_batch_size = int(max_batch_size)
        self._max_top_k = int(max_top_k)
        self._max_filter_terms = max_filter_terms
        self._text_storage_strategy = text_storage_strategy
        self._index: Any = None  # lazily initialized
        self._batch_query_max_concurrency = (
            int(batch_query_max_concurrency)
            if batch_query_max_concurrency is not None and batch_query_max_concurrency > 0
            else None
        )

        # Invoke BaseVectorAdapter init
        super().__init__(
            metrics=metrics,
            mode=mode,
            deadline_policy=deadline_policy,
            breaker=breaker,
            cache=cache,
            limiter=limiter,
            docstore=docstore,
            cache_query_ttl_s=cache_query_ttl_s,
            cache_caps_ttl_s=cache_caps_ttl_s,
            warn_on_standalone_no_metrics=warn_on_standalone_no_metrics,
            config=config,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _get_index(self) -> Any:
        """
        Lazily create and cache the Pinecone index handle.

        This is intentionally not ctx-aware because it is usually called
        during adapter setup; per-request deadlines are enforced in the
        data-path calls via `_call_pinecone`.
        """
        if self._index is not None:
            return self._index

        try:
            # Modern client: client.Index(name)
            index = self._client.Index(self._index_name)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            raise self._translate_error(exc, op="init_index", ctx=None)

        self._index = index
        return index

    @staticmethod
    async def _run_in_thread(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Run blocking client calls on a worker thread.

        This keeps the adapter async-first while dealing with a blocking SDK.
        """
        return await asyncio.to_thread(func, *args, **kwargs)

    @staticmethod
    def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
        """Support both dict-style and attribute-style access."""
        if isinstance(obj, Mapping):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @staticmethod
    def _attach_ctx_details(
        base_details: Dict[str, Any],
        ctx: Optional[OperationContext],
    ) -> Dict[str, Any]:
        """
        Enrich error details with SIEM-safe context fields.

        We deliberately avoid including raw tenant identifiers and only add
        low-cardinality, non-sensitive IDs suitable for logs/metrics.
        """
        details = dict(base_details)
        if ctx is None:
            return details
        if ctx.request_id:
            details.setdefault("request_id", ctx.request_id)
        if ctx.traceparent:
            details.setdefault("traceparent", ctx.traceparent)
        return details

    def _get_ctx_timeout_s(self, ctx: Optional[OperationContext]) -> Optional[float]:
        """
        Derive a per-call timeout in seconds from ctx.deadline_ms.

        Returns:
            - None  → no inner timeout (outer BaseVectorAdapter deadline still applies).
            - 0.0   → deadline already exceeded (callers will typically fail fast).
            - > 0.0 → remaining_ms / 1000.0
        """
        if ctx is None:
            return None
        remaining = ctx.remaining_ms()
        if remaining is None:
            return None
        if remaining <= 0:
            return 0.0
        return remaining / 1000.0

    async def _call_pinecone(
        self,
        *,
        op: str,
        func: Callable[..., T],
        ctx: Optional[OperationContext],
        **kwargs: Any,
    ) -> T:
        """
        Execute a Pinecone SDK call with ctx-aware timeout and error translation.

        Behavior:
            - Uses `_run_in_thread` to keep the adapter async-first.
            - Applies a best-effort per-call timeout derived from ctx.deadline_ms.
            - Maps asyncio.TimeoutError → TransientNetwork with ctx-enriched details.
            - All other errors are translated via `_translate_error`.
        """
        timeout_s = self._get_ctx_timeout_s(ctx)
        try:
            if timeout_s is not None:
                # Inner timeout for the Pinecone RPC, on top of the outer
                # BaseVectorAdapter deadline enforcement.
                return await asyncio.wait_for(
                    self._run_in_thread(func, **kwargs),
                    timeout=timeout_s,
                )
            return await self._run_in_thread(func, **kwargs)
        except asyncio.TimeoutError as exc:
            # Treat RPC-level timeout as transient network failure.
            raise TransientNetwork(
                "Pinecone operation timed out",
                code="TRANSIENT_NETWORK",
                retry_after_ms=500,
                details=self._attach_ctx_details({"op": op}, ctx),
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise self._translate_error(exc, op=op, ctx=ctx) from exc

    def _convert_score(self, raw_score: float) -> Tuple[float, float]:
        """
        Convert Pinecone's `score` into (similarity, distance).

        - cosine: score is similarity → distance = 1 - score
        - euclidean: score is distance → similarity = 1 / (1 + distance)
        - dotproduct: score is similarity → distance = -score
        """
        s = float(raw_score)
        if self._metric == "cosine":
            distance = 1.0 - s
            return s, distance
        if self._metric == "euclidean":
            distance = max(0.0, s)
            similarity = 1.0 / (1.0 + distance)
            return similarity, distance
        if self._metric == "dotproduct":
            distance = -s
            return s, distance
        # Fallback
        return s, s

    def _translate_error(
        self,
        err: Exception,
        *,
        op: str,
        ctx: Optional[OperationContext],
    ) -> VectorAdapterError:
        """
        Map Pinecone exceptions into normalized VectorAdapterError types.

        Note:
            We intentionally treat "no such index"/404 as IndexNotReady to keep
            the error retryable in environments where index lifecycle may lag
            behind adapter startup. Callers that want stricter behavior can
            enforce index existence at provisioning time.

        The returned errors include SIEM-safe ctx metadata (request_id, traceparent).
        """
        msg = str(err) or f"Pinecone error during {op}"
        logger.debug("Pinecone error in %s: %r", op, err)

        # Specific Pinecone exception type (best-effort).
        if isinstance(err, PineconeException):
            # PineconeException often has gRPC-style or HTTP-style status codes.
            status = (
                getattr(err, "status", None)
                or getattr(err, "status_code", None)
                or getattr(getattr(err, "response", None), "status_code", None)
            )
            status_int: Optional[int] = None
            try:
                if status is not None:
                    status_int = int(status)
            except Exception:
                status_int = None

            lowered = msg.lower()
            code_attr = getattr(err, "code", None)
            name_lower = code_attr.lower() if isinstance(code_attr, str) else ""

            # Rate limiting
            if (
                "rate limit" in lowered
                or "too many requests" in lowered
                or "resource exhausted" in lowered
                or "exceeded quota" in lowered
                or status_int == 429
                or "rate_limit" in name_lower
            ):
                return ResourceExhausted(
                    "Pinecone rate limit exceeded",
                    code="RESOURCE_EXHAUSTED",
                    retry_after_ms=500,
                    details=self._attach_ctx_details({"op": op}, ctx),
                )

            # Auth / permission
            if (
                "unauthorized" in lowered
                or "forbidden" in lowered
                or status_int in (401, 403)
                or name_lower in ("unauthorized", "permission_denied", "forbidden")
            ):
                return AuthError(
                    "Pinecone authentication/authorization error",
                    code="AUTH_ERROR",
                    details=self._attach_ctx_details({"op": op}, ctx),
                )

            # Not found / index not ready
            if (
                "not found" in lowered
                or "no such index" in lowered
                or status_int == 404
                or name_lower in ("not_found", "index_not_found")
            ):
                return IndexNotReady(
                    "Pinecone index or data not ready",
                    code="INDEX_NOT_READY",
                    retry_after_ms=1000,
                    details=self._attach_ctx_details({"op": op}, ctx),
                )

            # Timeouts / transient (non-asyncio based, e.g. HTTP gateway timeout)
            if (
                "timeout" in lowered
                or "temporarily unavailable" in lowered
                or "gateway timeout" in lowered
                or name_lower in ("deadline_exceeded",)
            ):
                return TransientNetwork(
                    "Pinecone transient network error",
                    code="TRANSIENT_NETWORK",
                    retry_after_ms=500,
                    details=self._attach_ctx_details({"op": op}, ctx),
                )

            # Validation / bad request
            if (
                "invalid" in lowered
                or "bad request" in lowered
                or status_int in (400, 422)
                or name_lower in ("invalid_argument", "bad_request")
            ):
                return BadRequest(
                    msg,
                    code="BAD_REQUEST",
                    details=self._attach_ctx_details({"op": op}, ctx),
                )

            # Server-side errors
            if status_int is not None and status_int >= 500:
                return Unavailable(
                    "Pinecone service unavailable",
                    code="UNAVAILABLE",
                    retry_after_ms=1000,
                    details=self._attach_ctx_details({"op": op}, ctx),
                )

            # Generic PineconeException
            return Unavailable(
                msg,
                code="UNAVAILABLE",
                details=self._attach_ctx_details({"op": op}, ctx),
            )

        # Generic transport/network issues
        lowered_generic = msg.lower()
        if "timeout" in lowered_generic:
            return TransientNetwork(
                "Pinecone network timeout",
                code="TRANSIENT_NETWORK",
                retry_after_ms=500,
                details=self._attach_ctx_details({"op": op}, ctx),
            )
        if "connection" in lowered_generic:
            return TransientNetwork(
                "Pinecone connection error",
                code="TRANSIENT_NETWORK",
                retry_after_ms=500,
                details=self._attach_ctx_details({"op": op}, ctx),
            )

        # Fallback
        return Unavailable(
            msg,
            code="UNAVAILABLE",
            details=self._attach_ctx_details({"op": op}, ctx),
        )

    # ------------------------------------------------------------------ #
    # BaseVectorAdapter backend hooks
    # ------------------------------------------------------------------ #

    async def _do_capabilities(self) -> VectorCapabilities:
        """
        Report Pinecone-backed capabilities.

        Note:
            - `max_dimensions` is taken from configuration (if provided).
            - `max_batch_size` is a soft planning hint, but we also enforce it
              as a hard limit on upsert/delete batches to protect the backend.
        """
        version = getattr(pinecone, "__version__", "unknown") if pinecone else "unknown"
        return VectorCapabilities(
            server="pinecone",
            version=version,
            max_dimensions=self._dimensions or 0,
            supported_metrics=(self._metric,),
            supports_namespaces=True,
            supports_metadata_filtering=True,
            supports_batch_operations=True,
            max_batch_size=self._max_batch_size or None,
            supports_index_management=False,  # index lifecycle managed externally
            idempotent_writes=True,
            supports_multi_tenant=False,
            supports_deadline=True,
            max_top_k=self._max_top_k or None,
            max_filter_terms=self._max_filter_terms,
            text_storage_strategy=self._text_storage_strategy,
            max_text_length=None,
            supports_batch_queries=True,  # adapter fan-out
        )

    # ------------------------------ query ---------------------------------- #

    async def _do_query(
        self,
        spec: QuerySpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> QueryResult:
        index = self._get_index()

        # Enforce top_k ceiling if configured.
        if self._max_top_k and spec.top_k > self._max_top_k:
            raise BadRequest(
                f"top_k {spec.top_k} exceeds maximum of {self._max_top_k}",
                code="BAD_REQUEST",
                details={
                    "max_top_k": self._max_top_k,
                    "namespace": spec.namespace,
                },
            )

        # Strict dimensionality check if configured.
        if self._dimensions and len(spec.vector) != self._dimensions:
            raise DimensionMismatch(
                f"query vector dimension {len(spec.vector)} does not match configured "
                f"dimensions {self._dimensions}",
                details={
                    "expected": self._dimensions,
                    "actual": len(spec.vector),
                    "namespace": spec.namespace,
                },
            )

        # Optionally guard filter complexity
        if self._max_filter_terms is not None and spec.filter:
            if len(spec.filter) > self._max_filter_terms:
                raise BadRequest(
                    f"filter too complex: {len(spec.filter)} terms exceeds "
                    f"{self._max_filter_terms}",
                    code="BAD_REQUEST",
                    details={
                        "provided_terms": len(spec.filter),
                        "max_terms": self._max_filter_terms,
                        "namespace": spec.namespace,
                    },
                )

        kwargs: Dict[str, Any] = {
            "vector": list(map(float, spec.vector)),
            "top_k": spec.top_k,
            "namespace": spec.namespace,
            "include_values": spec.include_vectors,
            "include_metadata": spec.include_metadata,
        }
        if spec.filter:
            kwargs["filter"] = dict(spec.filter)

        try:
            resp = await self._call_pinecone(
                op="query",
                func=index.query,
                ctx=ctx,
                **kwargs,
            )
        except VectorAdapterError:
            # Already normalized with ctx details.
            raise

        raw_matches: Sequence[Any] = (
            getattr(resp, "matches", None) or resp.get("matches", [])  # type: ignore[call-arg]
        )

        matches: List[VectorMatch] = []
        for m in raw_matches:
            vid = str(self._safe_get(m, "id", ""))
            if not vid:
                continue

            raw_score = float(self._safe_get(m, "score", 0.0) or 0.0)
            sim, dist = self._convert_score(raw_score)

            values = self._safe_get(m, "values", []) if spec.include_vectors else []
            meta = self._safe_get(m, "metadata", None) if spec.include_metadata else None

            # Vector.text is handled by BaseVectorAdapter when docstore is used.
            # For text_storage_strategy == "metadata", upstream callers are
            # expected to store text in metadata if desired.
            vector = Vector(
                id=vid,
                vector=list(values) if values else [],
                metadata=dict(meta) if isinstance(meta, Mapping) else meta,
                namespace=spec.namespace,
                text=None,
            )
            matches.append(
                VectorMatch(
                    vector=vector,
                    score=sim,
                    distance=dist,
                )
            )

        # Empty results are treated as a normal condition; callers decide
        # whether an empty result set is acceptable.
        return QueryResult(
            matches=matches,
            query_vector=list(spec.vector),
            namespace=spec.namespace,
            total_matches=len(raw_matches),
        )

    # ------------------------------ batch_query ---------------------------- #

    async def _do_batch_query(
        self,
        spec: BatchQuerySpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> List[QueryResult]:
        """
        Implement batch query via parallel fan-out of `_do_query`.

        We deliberately bypass BaseVectorAdapter.query() to avoid double
        instrumentation / caching; deadline is enforced at the outer layer.
        """
        # Reuse namespace for each individual QuerySpec if they did not set it.
        queries: List[QuerySpec] = []
        for q in spec.queries:
            if q.namespace != spec.namespace:
                # Normalize namespace to the batch namespace
                queries.append(
                    QuerySpec(
                        vector=q.vector,
                        top_k=q.top_k,
                        namespace=spec.namespace,
                        filter=q.filter,
                        include_metadata=q.include_metadata,
                        include_vectors=q.include_vectors,
                    )
                )
            else:
                queries.append(q)

        semaphore: Optional[asyncio.Semaphore]
        if self._batch_query_max_concurrency:
            semaphore = asyncio.Semaphore(self._batch_query_max_concurrency)
        else:
            semaphore = None

        async def run_one(q: QuerySpec) -> QueryResult:
            if semaphore is None:
                return await self._do_query(q, ctx=ctx)
            async with semaphore:
                return await self._do_query(q, ctx=ctx)

        tasks = [asyncio.create_task(run_one(q)) for q in queries]
        try:
            results = await asyncio.gather(*tasks)
        except Exception as exc:  # noqa: BLE001
            # Any unexpected error here is translated through the outer wrapper.
            raise self._translate_error(exc, op="batch_query", ctx=ctx) from exc

        return results

    # ------------------------------ upsert --------------------------------- #

    async def _do_upsert(
        self,
        spec: UpsertSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> UpsertResult:
        index = self._get_index()

        # Enforce batch size ceiling if configured.
        if self._max_batch_size and len(spec.vectors) > self._max_batch_size:
            raise BadRequest(
                f"upsert batch size {len(spec.vectors)} exceeds maximum of {self._max_batch_size}",
                code="BATCH_TOO_LARGE",
                details={
                    "max_batch_size": self._max_batch_size,
                    "provided": len(spec.vectors),
                    "suggested_batch_reduction": self._max_batch_size,
                    "namespace": spec.namespace,
                },
            )

        if self._dimensions:
            for v in spec.vectors:
                if len(v.vector) != self._dimensions:
                    raise DimensionMismatch(
                        f"vector dimension {len(v.vector)} does not match configured "
                        f"dimensions {self._dimensions}",
                        details={
                            "expected": self._dimensions,
                            "actual": len(v.vector),
                            "namespace": spec.namespace,
                        },
                    )

        payload: List[Dict[str, Any]] = []
        for v in spec.vectors:
            item: Dict[str, Any] = {
                "id": str(v.id),
                "values": [float(x) for x in v.vector],
            }
            if v.metadata is not None:
                meta = dict(v.metadata)
                # If caller wants text in metadata (strategy "metadata"), they can
                # already include it; we don't mutate metadata here.
                item["metadata"] = meta
            payload.append(item)

        try:
            resp = await self._call_pinecone(
                op="upsert",
                func=index.upsert,
                ctx=ctx,
                vectors=payload,
                namespace=spec.namespace,
            )
        except VectorAdapterError:
            # Already normalized with ctx details.
            raise

        # Pinecone usually returns {"upserted_count": N}, but we defensively handle
        # missing fields.
        upserted = 0
        if isinstance(resp, Mapping):
            try:
                upserted = int(resp.get("upserted_count", 0) or 0)
            except Exception:  # pragma: no cover - defensive
                upserted = 0
        if upserted <= 0:
            upserted = len(spec.vectors)

        return UpsertResult(
            upserted_count=upserted,
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
        index = self._get_index()

        kwargs: Dict[str, Any] = {"namespace": spec.namespace}
        if spec.ids:
            # Enforce batch size ceiling if configured.
            if self._max_batch_size and len(spec.ids) > self._max_batch_size:
                raise BadRequest(
                    f"delete batch size {len(spec.ids)} exceeds maximum of {self._max_batch_size}",
                    code="BATCH_TOO_LARGE",
                    details={
                        "max_batch_size": self._max_batch_size,
                        "provided": len(spec.ids),
                        "suggested_batch_reduction": self._max_batch_size,
                        "namespace": spec.namespace,
                    },
                )
            kwargs["ids"] = [str(i) for i in spec.ids]
        elif spec.filter:
            kwargs["filter"] = dict(spec.filter)
        else:
            # BaseVectorAdapter should already enforce ids|filter presence.
            raise BadRequest("must provide either ids or filter for deletion")

        # For namespace-wide delete, callers should use filter or explicitly
        # pass all ids; we intentionally do not expose delete_all here to avoid
        # accidental mass deletes.

        try:
            await self._call_pinecone(
                op="delete",
                func=index.delete,
                ctx=ctx,
                **kwargs,
            )
        except VectorAdapterError:
            # Already normalized with ctx details.
            raise

        # Pinecone delete does not always return counts; for now we report
        # "targeted" as deleted. This is SIEM-safe and stable.
        targeted = len(spec.ids) if spec.ids else 0

        return DeleteResult(
            deleted_count=targeted,
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
        """
        Pinecone namespaces are created implicitly on first write.

        We validate metric/dimensions against adapter configuration, then
        return a successful result without issuing remote calls.
        """
        if spec.distance_metric.lower() != self._metric:
            raise BadRequest(
                f"distance_metric '{spec.distance_metric}' does not match "
                f"configured metric '{self._metric}'",
                code="BAD_CONFIG",
                details={"metric": self._metric},
            )

        if self._dimensions and spec.dimensions != self._dimensions:
            raise BadRequest(
                f"dimensions {spec.dimensions} do not match configured "
                f"dimensions {self._dimensions}",
                code="BAD_CONFIG",
                details={"configured_dimensions": self._dimensions},
            )

        return NamespaceResult(
            success=True,
            namespace=spec.namespace,
            details={"created": False, "note": "Pinecone namespaces are implicit"},
        )

    async def _do_delete_namespace(
        self,
        namespace: str,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> NamespaceResult:
        """
        Delete all vectors in a namespace.

        We *do not* delete the index itself; only data under the namespace.
        """
        index = self._get_index()
        try:
            # Use delete_all=True scoped to the namespace so only that
            # namespace's contents are removed.
            await self._call_pinecone(
                op="delete_namespace",
                func=index.delete,
                ctx=ctx,
                namespace=namespace,
                delete_all=True,
            )
            existed = True
        except VectorAdapterError as exc:
            # If it's a "namespace not found" kind of error, treat as success
            if isinstance(exc, IndexNotReady) or isinstance(exc, BadRequest):
                logger.debug(
                    "delete_namespace(%s) treated as best-effort: %s",
                    namespace,
                    exc,
                )
                existed = False
            else:
                raise

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
        Lightweight health probe using describe_index_stats().

        Behavior:
            - Never raises; returns an `ok` flag and a small namespace summary.
            - Any remote error is captured and reflected as ok=False.
        """
        index = self._get_index()
        version = getattr(pinecone, "__version__", "unknown") if pinecone else "unknown"

        try:
            stats = await self._call_pinecone(
                op="health",
                func=index.describe_index_stats,
                ctx=ctx,
            )
            # Expected shape:
            #   {"namespaces": {"ns": {"vector_count": N}, ...}, ...}
            if isinstance(stats, Mapping):
                ns_stats_raw: Mapping[str, Any] = stats.get("namespaces", {}) or {}
            else:
                ns_stats_raw = getattr(stats, "namespaces", {}) or {}

            namespaces: Dict[str, Any] = {}
            for ns, info in ns_stats_raw.items():
                if not isinstance(info, Mapping):
                    # Best-effort: if info is an object, try attribute access.
                    if hasattr(info, "vector_count"):
                        count = int(getattr(info, "vector_count") or 0)
                    else:
                        continue
                else:
                    count = int(info.get("vector_count", 0) or 0)

                namespaces[str(ns)] = {
                    "dimensions": self._dimensions,
                    "metric": self._metric,
                    "count": count,
                    "status": "ok",
                }

            return {
                "ok": True,
                "server": "pinecone",
                "version": version,
                "namespaces": namespaces,
            }
        except Exception as exc:  # noqa: BLE001
            err = self._translate_error(exc, op="health", ctx=ctx)
            logger.warning("Pinecone health check failed: %s", err)
            return {
                "ok": False,
                "server": "pinecone",
                "version": version,
                "namespaces": {},
                "error_code": err.code,
                "error_message": str(err),
            }


__all__ = [
    "PineconeVectorAdapter",
]
