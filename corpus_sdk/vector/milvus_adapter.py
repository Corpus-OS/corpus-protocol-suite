# corpus_sdk/vector/milvus_adapter.py
# SPDX-License-Identifier: Apache-2.0
"""
Milvus Vector adapter for the Vector Protocol V1.0.

This module implements a production-grade adapter on top of the
`BaseVectorAdapter` / `VectorProtocolV1` contract, backed by Milvus.

Goals
-----
- Map Vector Protocol → Milvus (pymilvus) collection operations.
- Preserve async + backpressure semantics via BaseVectorAdapter.
- Normalize Milvus errors into the vector error taxonomy.
- Support namespaces (mapped to Milvus partitions) and batch upserts.
- Play nicely in "thin" (externally managed) and "standalone" modes.

Usage
-----
    from pymilvus import MilvusClient
    from corpus_sdk.vector.vector_base import Vector, UpsertSpec, QuerySpec
    from corpus_sdk.vector.milvus_adapter import MilvusVectorAdapter

    client = MilvusClient(uri="http://localhost:19530")
    adapter = MilvusVectorAdapter(
        client=client,
        collection_name="my_collection",
        vector_field="vector",
        id_field="id",
        metric="cosine",
        dimensions=1536,
        mode="standalone",  # or "thin"
    )

    # Upsert
    await adapter.upsert(
        UpsertSpec(
            namespace="demo",  # maps to Milvus partition "demo"
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
            # NOTE: metadata filtering is not supported by this adapter.
        )
    )
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .vector_base import (
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
    NotSupported,
    ResourceExhausted,
    TransientNetwork,
    Unavailable,
    VectorAdapterError,
)

logger = logging.getLogger(__name__)

# Try to import the modern Milvus client.
try:  # pragma: no cover - import surface only
    import pymilvus  # type: ignore
    from pymilvus import MilvusClient  # type: ignore[attr-defined]

    try:
        from pymilvus.exceptions import MilvusException  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        MilvusException = Exception  # type: ignore[assignment]
except Exception:  # pragma: no cover
    pymilvus = None  # type: ignore[assignment]
    MilvusClient = None  # type: ignore[assignment]
    MilvusException = Exception  # type: ignore[assignment]


class MilvusVectorAdapter(BaseVectorAdapter):
    """
    VectorProtocolV1 adapter backed by a Milvus collection.

    Design notes
    ------------
    - Async-first: all Milvus calls run via `asyncio.to_thread`.
    - Does *not* manage collection lifecycle; you create/delete collections outside.
    - Namespaces are mapped directly to Milvus partitions.
    - Batch queries are implemented as parallel fan-out over single-query calls.

    text_storage_strategy
    ---------------------
    - This adapter does not itself manage text persistence.
    - BaseVectorAdapter handles text/docstore integration.
    - Here we only:
        * Validate the strategy value ("metadata", "docstore", "none").
        * Enforce that "docstore" requires a docstore instance.
        * Report the configured strategy via capabilities.

    include_vectors
    ---------------
    - Milvus search does not return the full vector by default.
    - This adapter currently does *not* support returning vectors on query.
    - If `include_vectors=True` in QuerySpec, a NotSupported error is raised.
    """

    _component = "vector_milvus"

    def __init__(
        self,
        *,
        collection_name: str,
        vector_field: str,
        id_field: str,
        client: Optional["MilvusClient"] = None,
        uri: Optional[str] = None,
        token: Optional[str] = None,
        db_name: str = "default",
        metric: str = "cosine",
        dimensions: Optional[int] = None,
        metadata_field: Optional[str] = "metadata",
        # Soft limits / planning hints
        max_batch_size: int = 100,
        max_top_k: int = 100,
        max_filter_terms: Optional[int] = None,  # accepted for API symmetry; ignored (no filtering support)
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
        if pymilvus is None or MilvusClient is None:
            raise RuntimeError(
                "MilvusVectorAdapter requires the `pymilvus` Python package. "
                "Install via `pip install pymilvus`."
            )

        if not collection_name or not isinstance(collection_name, str):
            raise BadRequest(
                "collection_name must be a non-empty string",
                code="BAD_CONFIG",
            )
        if not vector_field or not isinstance(vector_field, str):
            raise BadRequest(
                "vector_field must be a non-empty string",
                code="BAD_CONFIG",
            )
        if not id_field or not isinstance(id_field, str):
            raise BadRequest(
                "id_field must be a non-empty string",
                code="BAD_CONFIG",
            )

        metric_norm = (metric or "cosine").lower().strip()
        if metric_norm not in ("cosine", "euclidean", "dotproduct"):
            raise BadRequest(
                "metric must be one of: cosine, euclidean, dotproduct",
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
            uri = uri or os.getenv("MILVUS_URI") or "http://localhost:19530"
            client_kwargs: Dict[str, Any] = {"uri": uri}
            if token:
                client_kwargs["token"] = token
            if db_name:
                client_kwargs["db_name"] = db_name

            client = MilvusClient(**client_kwargs)  # type: ignore[call-arg]

        # Adapter-local configuration
        self._client: MilvusClient = client  # type: ignore[assignment]
        self._collection_name = collection_name
        self._vector_field = vector_field
        self._id_field = id_field
        self._metadata_field = metadata_field
        self._metric = metric_norm
        self._dimensions = int(dimensions) if dimensions is not None else 0
        self._max_batch_size = int(max_batch_size)
        self._max_top_k = int(max_top_k)
        # max_filter_terms is intentionally not stored; filtering is not supported.
        self._text_storage_strategy = text_storage_strategy
        self._batch_query_max_concurrency = (
            int(batch_query_max_concurrency)
            if batch_query_max_concurrency is not None and batch_query_max_concurrency > 0
            else None
        )

        # Metric mapping: vector protocol → Milvus metric_type
        self._milvus_metric_type = {
            "cosine": "COSINE",
            "euclidean": "L2",
            "dotproduct": "IP",
        }[self._metric]

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

    def _get_client(self) -> MilvusClient:
        """Return the underlying Milvus client."""
        return self._client

    @staticmethod
    async def _run_in_thread(func, *args, **kwargs):
        """Run blocking client calls on a worker thread."""
        return await asyncio.to_thread(func, *args, **kwargs)

    @staticmethod
    def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
        """Support both dict-style and attribute-style access."""
        if isinstance(obj, Mapping):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _convert_score(self, raw_score: float) -> Tuple[float, float]:
        """
        Convert Milvus's `distance` into (similarity, distance).

        Milvus returns distance according to metric_type:
        - COSINE: distance in [0, 2]; similarity approximated as 1 - distance.
        - L2: distance is euclidean; similarity = 1 / (1 + distance).
        - IP: distance is inner product similarity; we treat as similarity and
          define distance as -score.
        """
        d = float(raw_score)
        if self._milvus_metric_type == "COSINE":
            distance = max(0.0, min(2.0, d))
            similarity = 1.0 - distance
            return similarity, distance
        if self._milvus_metric_type == "L2":
            distance = max(0.0, d)
            similarity = 1.0 / (1.0 + distance)
            return similarity, distance
        if self._milvus_metric_type == "IP":
            similarity = d
            distance = -similarity
            return similarity, distance
        # Fallback
        return d, d

    def _translate_error(self, err: Exception, *, op: str) -> VectorAdapterError:
        """
        Map Milvus exceptions into normalized VectorAdapterError types.

        Note:
            We intentionally treat "collection not found"/partition-not-found
            as IndexNotReady to keep the error retryable in environments where
            collection lifecycle may lag behind adapter startup. Callers that
            want stricter behavior can enforce collection existence at
            provisioning time.
        """
        msg = str(err) or f"Milvus error during {op}"
        logger.debug("Milvus error in %s: %r", op, err)

        # Specific Milvus exception type (best-effort).
        if isinstance(err, MilvusException):
            status = getattr(err, "code", None)
            status_int: Optional[int] = None
            try:
                if status is not None:
                    status_int = int(status)
            except Exception:
                status_int = None

            lowered = msg.lower()

            # Rate limiting / resource exhausted (heuristic)
            if (
                "rate limit" in lowered
                or "too many requests" in lowered
                or "resource exhausted" in lowered
                or "exceeded quota" in lowered
            ):
                return ResourceExhausted(
                    "Milvus rate limit exceeded",
                    code="RESOURCE_EXHAUSTED",
                    retry_after_ms=500,
                    details={"op": op},
                )

            # Auth / permission
            if "unauthorized" in lowered or "forbidden" in lowered:
                return AuthError(
                    "Milvus authentication/authorization error",
                    code="AUTH_ERROR",
                    details={"op": op},
                )

            # Collection not found / index not ready
            if (
                "collection not found" in lowered
                or "partition not found" in lowered
                or "not found" in lowered
                or (status_int is not None and status_int == 7)  # common NOT_FOUND code
            ):
                return IndexNotReady(
                    "Milvus collection or partition not ready",
                    code="INDEX_NOT_READY",
                    retry_after_ms=1000,
                    details={"op": op},
                )

            # Timeouts / transient
            if "timeout" in lowered or "temporarily unavailable" in lowered:
                return TransientNetwork(
                    "Milvus transient network error",
                    code="TRANSIENT_NETWORK",
                    retry_after_ms=500,
                    details={"op": op},
                )

            # Validation / bad request
            if (
                "invalid" in lowered
                or "illegal" in lowered
                or "bad request" in lowered
            ):
                return BadRequest(
                    msg,
                    code="BAD_REQUEST",
                    details={"op": op},
                )

            # Server-side or unknown errors
            return Unavailable(
                msg,
                code="UNAVAILABLE",
                details={"op": op},
            )

        # Generic transport/network issues
        lowered_generic = msg.lower()
        if "timeout" in lowered_generic:
            return TransientNetwork(
                "Milvus network timeout",
                code="TRANSIENT_NETWORK",
                retry_after_ms=500,
                details={"op": op},
            )
        if "connection" in lowered_generic:
            return TransientNetwork(
                "Milvus connection error",
                code="TRANSIENT_NETWORK",
                retry_after_ms=500,
                details={"op": op},
            )

        # Fallback
        return Unavailable(
            msg,
            code="UNAVAILABLE",
            details={"op": op},
        )

    # ------------------------------------------------------------------ #
    # BaseVectorAdapter backend hooks
    # ------------------------------------------------------------------ #

    async def _do_capabilities(self) -> VectorCapabilities:
        """
        Report Milvus-backed capabilities.

        Note:
            - `max_dimensions` is taken from configuration (if provided).
            - `max_batch_size` is a soft planning hint, but we also enforce it
              as a hard limit on upsert/delete batches to protect the backend.
            - Metadata filtering is currently not supported by this adapter.
        """
        version = getattr(pymilvus, "__version__", "unknown") if pymilvus else "unknown"
        return VectorCapabilities(
            server="milvus",
            version=version,
            max_dimensions=self._dimensions or 0,
            supported_metrics=(self._metric,),
            supports_namespaces=True,  # mapped to partitions
            supports_metadata_filtering=False,
            supports_batch_operations=True,
            max_batch_size=self._max_batch_size or None,
            supports_index_management=False,  # collection lifecycle managed externally
            idempotent_writes=True,
            supports_multi_tenant=False,
            supports_deadline=True,
            max_top_k=self._max_top_k or None,
            max_filter_terms=None,  # filtering not supported
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
        client = self._get_client()

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

        # Metadata filtering is not yet supported.
        if spec.filter:
            raise NotSupported(
                "metadata filtering is not supported by MilvusVectorAdapter",
                code="NOT_SUPPORTED",
                details={"op": "query"},
            )

        # Returning full vectors on query is not supported for now.
        if spec.include_vectors:
            raise NotSupported(
                "include_vectors is not supported by MilvusVectorAdapter "
                "(Milvus search does not return vectors by default)",
                code="NOT_SUPPORTED",
                details={"op": "query"},
            )

        search_params: Dict[str, Any] = {
            "metric_type": self._milvus_metric_type,
            "params": {},
        }

        kwargs: Dict[str, Any] = {
            "collection_name": self._collection_name,
            "data": [list(map(float, spec.vector))],
            "anns_field": self._vector_field,
            "param": search_params,
            "limit": spec.top_k,
        }
        if spec.namespace:
            kwargs["partition_names"] = [spec.namespace]

        # Decide which scalar fields to fetch.
        output_fields = [self._id_field]
        if spec.include_metadata and self._metadata_field:
            output_fields.append(self._metadata_field)
        kwargs["output_fields"] = output_fields

        try:
            # MilvusClient.search returns a list of hits per query vector.
            resp = await self._run_in_thread(client.search, **kwargs)
        except Exception as exc:  # noqa: BLE001
            raise self._translate_error(exc, op="query") from exc

        # resp is typically: [ [hit1, hit2, ...], ... ] per query vector.
        first_hits: Sequence[Any] = resp[0] if resp else []

        matches: List[VectorMatch] = []
        for h in first_hits:
            # For MilvusClient hits, fields are usually available via dict-style.
            fields = getattr(h, "fields", None)
            if isinstance(fields, Mapping):
                row = fields
            elif isinstance(h, Mapping):
                row = h
            else:
                # Fallback: hope the hit behaves like an object with attributes
                row = {
                    self._id_field: getattr(h, self._id_field, None),
                }
                if self._metadata_field:
                    row[self._metadata_field] = getattr(h, self._metadata_field, None)

            vid = row.get(self._id_field)
            if vid is None:
                continue

            # Distance/similarity
            raw_distance = getattr(h, "distance", None)
            if raw_distance is None and isinstance(h, Mapping):
                raw_distance = h.get("distance", 0.0)
            sim, dist = self._convert_score(float(raw_distance or 0.0))

            # We don't return vectors on query for now; see include_vectors notes.
            values: Sequence[float] = []

            if spec.include_metadata and self._metadata_field:
                meta = row.get(self._metadata_field)
            else:
                meta = None

            vector = Vector(
                id=str(vid),
                vector=list(values) if values else [],
                metadata=meta,
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

        return QueryResult(
            matches=matches,
            query_vector=list(spec.vector),
            namespace=spec.namespace,
            total_matches=len(first_hits),
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
            raise self._translate_error(exc, op="batch_query") from exc

        return results

    # ------------------------------ upsert --------------------------------- #

    async def _do_upsert(
        self,
        spec: UpsertSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> UpsertResult:
        client = self._get_client()

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

        rows: List[Dict[str, Any]] = []
        for v in spec.vectors:
            row: Dict[str, Any] = {
                self._id_field: str(v.id),
                self._vector_field: [float(x) for x in v.vector],
            }
            if self._metadata_field and v.metadata is not None:
                # Milvus JSON field can hold arbitrary metadata if schema supports it.
                row[self._metadata_field] = v.metadata
            rows.append(row)

        kwargs: Dict[str, Any] = {
            "collection_name": self._collection_name,
            "data": rows,
        }
        if spec.namespace:
            kwargs["partition_name"] = spec.namespace

        try:
            resp = await self._run_in_thread(client.insert, **kwargs)
        except Exception as exc:  # noqa: BLE001
            raise self._translate_error(exc, op="upsert") from exc

        # Milvus insert returns ids and insert count; we primarily care about count.
        upserted = 0
        if isinstance(resp, Mapping):
            try:
                upserted = int(resp.get("insert_count", 0) or 0)
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
        client = self._get_client()

        # We currently support delete-by-id; delete-by-filter on metadata is not
        # implemented for Milvus due to the complexity of generic expression
        # generation over arbitrary metadata.
        if spec.filter:
            raise NotSupported(
                "delete by metadata filter is not supported by MilvusVectorAdapter",
                code="NOT_SUPPORTED",
                details={"op": "delete"},
            )

        kwargs: Dict[str, Any] = {"collection_name": self._collection_name}
        targeted = 0
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
            # Build a boolean expression on the primary key field.
            # Example: id in ["a","b","c"]
            id_list = ",".join(f'"{str(i)}"' for i in spec.ids)
            expr = f'{self._id_field} in [{id_list}]'
            kwargs["expr"] = expr
            targeted = len(spec.ids)
        else:
            # BaseVectorAdapter should already enforce ids|filter presence.
            raise BadRequest("must provide either ids or filter for deletion", code="BAD_REQUEST")

        if spec.namespace:
            kwargs["partition_name"] = spec.namespace

        try:
            await self._run_in_thread(client.delete, **kwargs)
        except Exception as exc:  # noqa: BLE001
            raise self._translate_error(exc, op="delete") from exc

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
        Milvus namespaces are mapped to partitions within a collection.

        We validate metric/dimensions against adapter configuration, then
        create the partition if it does not exist.
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

        client = self._get_client()

        try:
            partitions = await self._run_in_thread(
                client.list_partitions, self._collection_name
            )
        except Exception as exc:  # noqa: BLE001
            # If we fail to list partitions, we still attempt creation and rely on errors.
            logger.debug("list_partitions failed during create_namespace: %s", exc)
            partitions = []

        existing_names: set[str] = set()
        if isinstance(partitions, list):
            for p in partitions:
                if isinstance(p, Mapping):
                    pname = p.get("partition_name")
                else:
                    pname = getattr(p, "partition_name", None)
                if pname:
                    existing_names.add(str(pname))

        created = False
        if spec.namespace not in existing_names:
            try:
                await self._run_in_thread(
                    client.create_partition,
                    collection_name=self._collection_name,
                    partition_name=spec.namespace,
                )
                created = True
            except Exception as exc:  # noqa: BLE001
                raise self._translate_error(exc, op="create_namespace") from exc

        return NamespaceResult(
            success=True,
            namespace=spec.namespace,
            details={"created": created},
        )

    async def _do_delete_namespace(
        self,
        namespace: str,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> NamespaceResult:
        """
        Delete all vectors in a namespace by dropping the Milvus partition.

        We *do not* delete the collection itself; only the partition's data.
        """
        client = self._get_client()
        existed = False
        try:
            await self._run_in_thread(
                client.drop_partition,
                collection_name=self._collection_name,
                partition_name=namespace,
            )
            existed = True
        except Exception as exc:  # noqa: BLE001
            err = self._translate_error(exc, op="delete_namespace")
            if isinstance(err, IndexNotReady) or isinstance(err, BadRequest):
                # Treat partition-not-found as best-effort success.
                logger.debug(
                    "delete_namespace(%s) treated as best-effort: %s",
                    namespace,
                    exc,
                )
                existed = False
            else:
                raise err

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
        Lightweight health probe using get_collection_stats() and list_partitions().

        Behavior:
            - Never raises; returns an `ok` flag and a small namespace summary.
            - Any remote error is captured and reflected as ok=False.
        """
        client = self._get_client()
        version = getattr(pymilvus, "__version__", "unknown") if pymilvus else "unknown"

        try:
            stats = await self._run_in_thread(
                client.get_collection_stats,
                collection_name=self._collection_name,
            )
            if isinstance(stats, Mapping):
                row_count_raw = stats.get("row_count", 0)
            else:
                row_count_raw = getattr(stats, "row_count", 0)

            try:
                total_count = int(row_count_raw or 0)
            except Exception:
                total_count = 0

            try:
                partitions = await self._run_in_thread(
                    client.list_partitions,
                    self._collection_name,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("list_partitions failed during health: %s", exc)
                partitions = []

            namespaces: Dict[str, Any] = {}
            # Milvus stats do not always expose per-partition counts; we surface
            # per-namespace entries with count=None when not available.
            if isinstance(partitions, list):
                for p in partitions:
                    if isinstance(p, Mapping):
                        pname = p.get("partition_name")
                    else:
                        pname = getattr(p, "partition_name", None)
                    if not pname:
                        continue
                    namespaces[str(pname)] = {
                        "dimensions": self._dimensions,
                        "metric": self._metric,
                        "count": None,  # unknown per-partition count
                        "status": "ok",
                    }

            # Also provide an aggregate "*" namespace summary.
            namespaces.setdefault(
                "*",
                {
                    "dimensions": self._dimensions,
                    "metric": self._metric,
                    "count": total_count,
                    "status": "ok",
                },
            )

            return {
                "ok": True,
                "server": "milvus",
                "version": version,
                "namespaces": namespaces,
            }
        except Exception as exc:  # noqa: BLE001
            err = self._translate_error(exc, op="health")
            logger.warning("Milvus health check failed: %s", err)
            return {
                "ok": False,
                "server": "milvus",
                "version": version,
                "namespaces": {},
                "error_code": err.code,
                "error_message": str(err),
            }


__all__ = [
    "MilvusVectorAdapter",
]
