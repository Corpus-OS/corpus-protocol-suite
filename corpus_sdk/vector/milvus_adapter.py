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
            filter={"category": "shoes"},  # example metadata filter
        )
    )
"""

from __future__ import annotations

import asyncio
import logging
import os
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Set

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
    - Milvus search does not always return the full vector by default.
    - This adapter supports returning vectors on query when requested:
        * If `include_vectors=True` in QuerySpec, the vector field is included
          in Milvus output_fields and populated into results when available.
        * If the schema or server does not return the vector field, an empty
          vector is returned, preserving safety and avoiding extra round trips.

    metadata filtering
    ------------------
    - This adapter supports metadata filtering using a Mongo-style dict:
        * Logical operators: $and, $or, $not
        * Comparison operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin
        * Implicit equality: {"color": "red"}
    - These are translated into Milvus JSON expressions targeting the configured
      JSON metadata field (e.g., metadata["color"] == "red").
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
        self._max_filter_terms = (
            int(max_filter_terms)
            if max_filter_terms is not None and max_filter_terms > 0
            else None
        )
        self._text_storage_strategy = text_storage_strategy
        self._batch_query_max_concurrency = (
            int(batch_query_max_concurrency)
            if batch_query_max_concurrency is not None and batch_query_max_concurrency > 0
            else None
        )

        # Track known namespaces (partitions) and namespaces that have data.
        self._known_partitions: Set[str] = set()
        self._namespaces_with_data: Set[str] = set()

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

    @staticmethod
    def _get_ctx_timeout_s(ctx: Optional[OperationContext]) -> Optional[float]:
        """
        Extract a remaining timeout (in seconds) from OperationContext, if available.
        We only look for a `remaining_timeout_s` attribute to avoid guessing.
        """
        if ctx is None:
            return None
        timeout = getattr(ctx, "remaining_timeout_s", None)
        try:
            return float(timeout) if timeout is not None else None
        except Exception:
            return None

    async def _call_milvus(
        self,
        func,
        op: str,
        ctx: Optional[OperationContext],
        *args,
        **kwargs,
    ):
        """
        Invoke a Milvus client function in a worker thread, honoring ctx deadlines.

        - If ctx.remaining_timeout_s is present and > 0, wrap the call in
          asyncio.wait_for() using that timeout.
        - asyncio.TimeoutError is translated into TransientNetwork.
        - Other exceptions are translated via _translate_error.
        """
        timeout_s = self._get_ctx_timeout_s(ctx)
        try:
            if timeout_s is not None and timeout_s > 0:
                return await asyncio.wait_for(
                    self._run_in_thread(func, *args, **kwargs),
                    timeout=timeout_s,
                )
            return await self._run_in_thread(func, *args, **kwargs)
        except asyncio.TimeoutError as exc:
            raise TransientNetwork(
                "Milvus operation timed out",
                code="TRANSIENT_NETWORK",
                retry_after_ms=500,
                details={"op": op},
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise self._translate_error(exc, op=op) from exc

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

        We intentionally treat "collection not found"/partition-not-found
        as IndexNotReady to keep the error retryable in environments where
        collection lifecycle may lag behind adapter startup.
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

    async def _ensure_namespace_known(
        self,
        namespace: str,
        *,
        op: str,
        ctx: Optional[OperationContext] = None,
    ) -> bool:
        """
        Ensure that a namespace (partition) is known to this adapter.

        Returns:
            True if the namespace exists as a Milvus partition, False if it does not.

        Raises:
            VectorAdapterError via _translate_error on underlying Milvus failures.
        """
        if not namespace:
            return True
        if namespace in self._known_partitions:
            return True

        client = self._get_client()
        partitions = await self._call_milvus(
            client.list_partitions,
            op,
            ctx,
            self._collection_name,
        )

        if isinstance(partitions, list):
            for p in partitions:
                if isinstance(p, Mapping):
                    pname = p.get("partition_name")
                else:
                    pname = getattr(p, "partition_name", None)
                if pname:
                    self._known_partitions.add(str(pname))

        return namespace in self._known_partitions

    def _build_filter_expression(self, filters: Dict[str, Any]) -> str:
        """
        Translate a dictionary filter into a Milvus boolean expression.

        Assumes Milvus 2.3+ JSON field capabilities where self._metadata_field
        is a JSON field.

        Supported operators:
            - Logical: $and, $or, $not
            - Comparison: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin
            - Implicit equality: {"color": "red"}
        """
        if not filters:
            return ""

        if not self._metadata_field:
            raise NotSupported(
                "Metadata filtering requires a configured `metadata_field`",
                code="BAD_CONFIG",
            )

        conditions: List[str] = []

        for key, value in filters.items():
            # 1. Logical operators ($and, $or, $not)
            if key == "$and":
                if not isinstance(value, list):
                    raise BadRequest("$and value must be a list")
                sub_conds = [self._build_filter_expression(f) for f in value]
                sub_conds = [c for c in sub_conds if c]
                if sub_conds:
                    conditions.append(f"({' and '.join(sub_conds)})")
                continue

            if key == "$or":
                if not isinstance(value, list):
                    raise BadRequest("$or value must be a list")
                sub_conds = [self._build_filter_expression(f) for f in value]
                sub_conds = [c for c in sub_conds if c]
                if sub_conds:
                    conditions.append(f"({' or '.join(sub_conds)})")
                continue

            if key == "$not":
                if not isinstance(value, dict):
                    raise BadRequest("$not value must be a dict")
                sub_cond = self._build_filter_expression(value)
                if sub_cond:
                    conditions.append(f"not ({sub_cond})")
                continue

            # 2. Field access: metadata["field_name"]
            milvus_field = f'{self._metadata_field}["{key}"]'

            # 3. Comparison operators or implicit equality
            if isinstance(value, dict):
                for op, op_val in value.items():
                    val_str = json.dumps(op_val)

                    if op == "$eq":
                        conditions.append(f"{milvus_field} == {val_str}")
                    elif op == "$ne":
                        conditions.append(f"{milvus_field} != {val_str}")
                    elif op == "$gt":
                        conditions.append(f"{milvus_field} > {val_str}")
                    elif op == "$gte":
                        conditions.append(f"{milvus_field} >= {val_str}")
                    elif op == "$lt":
                        conditions.append(f"{milvus_field} < {val_str}")
                    elif op == "$lte":
                        conditions.append(f"{milvus_field} <= {val_str}")
                    elif op == "$in":
                        conditions.append(f"{milvus_field} in {val_str}")
                    elif op == "$nin":
                        conditions.append(f"{milvus_field} not in {val_str}")
                    else:
                        raise BadRequest(f"unsupported filter operator: {op}")
            else:
                # Implicit equality: {"color": "red"}
                val_str = json.dumps(value)
                conditions.append(f"{milvus_field} == {val_str}")

        return " and ".join(c for c in conditions if c)

    # ------------------------------------------------------------------ #
    # BaseVectorAdapter backend hooks
    # ------------------------------------------------------------------ #

    async def _do_capabilities(self) -> VectorCapabilities:
        """
        Report Milvus-backed capabilities.

        Notes:
            - `max_dimensions` is set to 0 to avoid double-enforcement in the
              BaseVectorAdapter; dimensionality is enforced explicitly here.
            - `max_batch_size` is advertised here and enforced by
              BaseVectorAdapter for upsert/delete before hitting this layer.
        """
        version = getattr(pymilvus, "__version__", "unknown") if pymilvus else "unknown"
        return VectorCapabilities(
            server="milvus",
            version=version,
            max_dimensions=0,
            supported_metrics=(self._metric,),
            supports_namespaces=True,  # mapped to partitions
            supports_metadata_filtering=True,
            supports_batch_operations=True,
            max_batch_size=self._max_batch_size or None,
            supports_index_management=False,  # collection lifecycle managed externally
            idempotent_writes=True,
            supports_multi_tenant=False,
            supports_deadline=True,
            max_top_k=self._max_top_k or None,
            max_filter_terms=self._max_filter_terms,
            text_storage_strategy=self._text_storage_strategy,
            max_text_length=None,
            supports_batch_queries=True,
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

        # Build metadata filter expression if provided.
        filter_expr = ""
        if spec.filter:
            try:
                filter_expr = self._build_filter_expression(spec.filter)
            except VectorAdapterError:
                raise
            except Exception as e:  # noqa: BLE001
                raise BadRequest(
                    f"Invalid filter expression: {e}",
                    details={"filter": spec.filter},
                )

        # Namespace existence & readiness semantics.
        if spec.namespace:
            namespace_known = await self._ensure_namespace_known(
                spec.namespace, op="query", ctx=ctx
            )
            if not namespace_known:
                raise BadRequest(
                    f"unknown namespace '{spec.namespace}'",
                    details={"namespace": spec.namespace},
                )
            if spec.namespace not in self._namespaces_with_data:
                raise IndexNotReady(
                    "index not ready (no data in namespace)",
                    retry_after_ms=500,
                    details={"namespace": spec.namespace},
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
        if spec.include_vectors:
            output_fields.append(self._vector_field)
        kwargs["output_fields"] = output_fields

        if filter_expr:
            kwargs["filter"] = filter_expr

        resp = await self._call_milvus(
            client.search,
            "query",
            ctx,
            **kwargs,
        )

        # resp is typically: [ [hit1, hit2, ...], ... ] per query vector.
        first_hits: Sequence[Any] = resp[0] if resp else []

        matches: List[VectorMatch] = []
        for h in first_hits:
            fields = getattr(h, "fields", None)
            if isinstance(fields, Mapping):
                row = fields
            elif isinstance(h, Mapping):
                row = h
            else:
                row = {
                    self._id_field: getattr(h, self._id_field, None),
                }
                if self._metadata_field:
                    row[self._metadata_field] = getattr(h, self._metadata_field, None)
                if spec.include_vectors:
                    row[self._vector_field] = getattr(h, self._vector_field, None)

            vid = row.get(self._id_field)
            if vid is None:
                continue

            raw_distance = getattr(h, "distance", None)
            if raw_distance is None and isinstance(h, Mapping):
                raw_distance = h.get("distance", 0.0)
            sim, dist = self._convert_score(float(raw_distance or 0.0))

            values: Sequence[float] = []
            if spec.include_vectors:
                raw_vec = row.get(self._vector_field)
                if isinstance(raw_vec, (list, tuple)):
                    try:
                        values = [float(x) for x in raw_vec]
                    except Exception:
                        values = []

            if spec.include_metadata and self._metadata_field:
                meta = row.get(self._metadata_field)
            else:
                meta = None

            vector = Vector(
                id=str(vid),
                vector=list(values) if spec.include_vectors else [],
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
        queries: List[QuerySpec] = []
        for q in spec.queries:
            if q.namespace != spec.namespace:
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

        failures: List[Dict[str, Any]] = []
        rows: List[Dict[str, Any]] = []

        for index, v in enumerate(spec.vectors):
            if self._dimensions and len(v.vector) != self._dimensions:
                failures.append(
                    {
                        "id": str(v.id),
                        "index": index,
                        "code": "DIMENSION_MISMATCH",
                        "message": f"expected {self._dimensions}, got {len(v.vector)}",
                        "details": {
                            "expected": self._dimensions,
                            "actual": len(v.vector),
                            "namespace": spec.namespace,
                        },
                    }
                )
                continue

            row: Dict[str, Any] = {
                self._id_field: str(v.id),
                self._vector_field: [float(x) for x in v.vector],
            }
            if self._metadata_field and v.metadata is not None:
                row[self._metadata_field] = v.metadata
            rows.append(row)

        kwargs: Dict[str, Any] = {
            "collection_name": self._collection_name,
            "data": rows,
        }
        if spec.namespace:
            kwargs["partition_name"] = spec.namespace

        upserted = 0
        if rows:
            resp = await self._call_milvus(
                client.insert,
                "upsert",
                ctx,
                **kwargs,
            )

            if isinstance(resp, Mapping):
                try:
                    upserted = int(resp.get("insert_count", 0) or 0)
                except Exception:  # pragma: no cover
                    upserted = 0
            if upserted <= 0:
                upserted = len(rows)

            if spec.namespace:
                self._namespaces_with_data.add(spec.namespace)
                self._known_partitions.add(spec.namespace)

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
        client = self._get_client()

        # Namespace existence semantics for delete:
        if spec.namespace:
            namespace_known = await self._ensure_namespace_known(
                spec.namespace, op="delete", ctx=ctx
            )
            if not namespace_known:
                raise BadRequest(
                    f"unknown namespace '{spec.namespace}'",
                    details={"namespace": spec.namespace},
                )

        kwargs: Dict[str, Any] = {"collection_name": self._collection_name}
        deleted_count = 0

        if spec.ids:
            id_strings = [str(i) for i in spec.ids]
            id_list_expr = ",".join(f'"{i}"' for i in id_strings)
            query_expr = f'{self._id_field} in [{id_list_expr}]'

            query_kwargs: Dict[str, Any] = {
                "collection_name": self._collection_name,
                "expr": query_expr,
                "output_fields": [self._id_field],
            }
            if spec.namespace:
                query_kwargs["partition_names"] = [spec.namespace]

            existing_rows = await self._call_milvus(
                client.query,
                "delete",
                ctx,
                **query_kwargs,
            )

            existing_ids: List[str] = []
            if isinstance(existing_rows, list):
                for row in existing_rows:
                    if isinstance(row, Mapping):
                        rid = row.get(self._id_field)
                    else:
                        rid = getattr(row, self._id_field, None)
                    if rid is not None:
                        existing_ids.append(str(rid))

            if not existing_ids:
                return DeleteResult(
                    deleted_count=0,
                    failed_count=0,
                    failures=[],
                )

            existing_expr = ",".join(f'"{i}"' for i in existing_ids)
            expr = f'{self._id_field} in [{existing_expr}]'
            deleted_count = len(existing_ids)
        elif spec.filter:
            try:
                filter_expr = self._build_filter_expression(spec.filter)
            except VectorAdapterError:
                raise
            except Exception as e:  # noqa: BLE001
                raise BadRequest(
                    f"Invalid filter for delete: {e}",
                    details={"filter": spec.filter},
                )

            if not filter_expr:
                return DeleteResult(
                    deleted_count=0,
                    failed_count=0,
                    failures=[],
                )

            query_kwargs: Dict[str, Any] = {
                "collection_name": self._collection_name,
                "expr": filter_expr,
                "output_fields": [self._id_field],
            }
            if spec.namespace:
                query_kwargs["partition_names"] = [spec.namespace]

            existing_rows = await self._call_milvus(
                client.query,
                "delete",
                ctx,
                **query_kwargs,
            )

            existing_ids: List[str] = []
            if isinstance(existing_rows, list):
                for row in existing_rows:
                    if isinstance(row, Mapping):
                        rid = row.get(self._id_field)
                    else:
                        rid = getattr(row, self._id_field, None)
                    if rid is not None:
                        existing_ids.append(str(rid))

            if not existing_ids:
                return DeleteResult(
                    deleted_count=0,
                    failed_count=0,
                    failures=[],
                )

            expr = filter_expr
            deleted_count = len(existing_ids)
        else:
            raise BadRequest(
                "must provide either ids or filter for deletion",
                code="BAD_REQUEST",
            )

        if spec.namespace:
            kwargs["partition_name"] = spec.namespace

        kwargs["filter"] = expr

        await self._call_milvus(
            client.delete,
            "delete",
            ctx,
            **kwargs,
        )

        return DeleteResult(
            deleted_count=deleted_count,
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
            partitions = await self._call_milvus(
                client.list_partitions,
                "create_namespace",
                ctx,
                self._collection_name,
            )
        except VectorAdapterError:
            logger.debug("list_partitions failed during create_namespace", exc_info=True)
            partitions = []

        existing_names: set[str] = set()
        if isinstance(partitions, list):
            for p in partitions:
                if isinstance(p, Mapping):
                    pname = p.get("partition_name")
                else:
                    pname = getattr(p, "partition_name", None)
                if pname:
                    pname_str = str(pname)
                    existing_names.add(pname_str)
                    self._known_partitions.add(pname_str)

        created = False
        if spec.namespace not in existing_names:
            await self._call_milvus(
                client.create_partition,
                "create_namespace",
                ctx,
                collection_name=self._collection_name,
                partition_name=spec.namespace,
            )
            created = True
            self._known_partitions.add(spec.namespace)
        else:
            self._known_partitions.add(spec.namespace)

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
            await self._call_milvus(
                client.drop_partition,
                "delete_namespace",
                ctx,
                collection_name=self._collection_name,
                partition_name=namespace,
            )
            existed = True
        except VectorAdapterError as err:
            if isinstance(err, IndexNotReady) or isinstance(err, BadRequest):
                logger.debug(
                    "delete_namespace(%s) treated as best-effort",
                    namespace,
                    exc_info=True,
                )
                existed = False
            else:
                raise

        self._known_partitions.discard(namespace)
        self._namespaces_with_data.discard(namespace)

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
            stats = await self._call_milvus(
                client.get_collection_stats,
                "health",
                ctx,
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
                partitions = await self._call_milvus(
                    client.list_partitions,
                    "health",
                    ctx,
                    self._collection_name,
                )
            except VectorAdapterError:
                logger.debug("list_partitions failed during health", exc_info=True)
                partitions = []

            namespaces: Dict[str, Any] = {}
            if isinstance(partitions, list):
                for p in partitions:
                    if isinstance(p, Mapping):
                        pname = p.get("partition_name")
                    else:
                        pname = getattr(p, "partition_name", None)
                    if not pname:
                        continue
                    pname_str = str(pname)
                    namespaces[pname_str] = {
                        "dimensions": self._dimensions,
                        "metric": self._metric,
                        "count": None,
                        "status": "ok",
                    }
                    self._known_partitions.add(pname_str)

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
        except VectorAdapterError as err:
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
