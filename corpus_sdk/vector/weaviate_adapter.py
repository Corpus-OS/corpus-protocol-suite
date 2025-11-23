# corpus_sdk/vector/weaviate_adapter.py
# SPDX-License-Identifier: Apache-2.0
"""
Weaviate Vector adapter for the Vector Protocol V1.1.

This module implements a production-grade adapter on top of the
`BaseVectorAdapter` / `VectorProtocolV1` contract, backed by Weaviate.

Goals
-----
- Map Vector Protocol → Weaviate class operations (v3 GraphQL-style client).
- Preserve async + backpressure semantics via BaseVectorAdapter.
- Normalize Weaviate errors into the vector error taxonomy.
- Provide batch upserts and deletes with hard safety limits.
- Play nicely in "thin" (externally managed) and "standalone" modes.

Important design notes
----------------------
- Client API / version:
    * This adapter explicitly targets the v3-style `weaviate.Client` API,
      where the client exposes:
          - `client.query` (GraphQL)
          - `client.data_object` (objects CRUD)
    * If the provided client does not have these attributes, the adapter
      raises a BadRequest config error at initialization.
- Vectors:
    * Assumes *user-provided* vectors:
        - Upserts call Weaviate with explicit vectors for each object.
        - Queries use `nearVector`-style search with the provided query vector.
- Namespaces:
    * Weaviate does not have a generic namespace concept that maps cleanly
      to the protocol's namespace model.
    * This adapter provides LIMITED namespace support by:
        - Encoding namespace into metadata (namespace_field)
        - Filtering queries/deletes by namespace via where filters
        - This is a best-effort implementation suitable for light multi-tenancy
    * For production multi-tenancy, consider:
        - Separate Weaviate classes per tenant
        - Weaviate's native multi-tenancy features (v1.20+)
        - Router-level tenant isolation
- Metadata filtering:
    * Weaviate supports rich GraphQL filters with schema-aware expressions.
    * This adapter translates Mongo-style filters into Weaviate's where syntax.
    * Supported operators: $and, $or, $not, $eq, $ne, $gt, $gte, $lt, $lte
- include_vectors:
    * Weaviate can return vectors via `_additional { vector }`.
    * This adapter supports `include_vectors=True` in QuerySpec and will fetch
      vectors via `_additional.vector` when requested.

text_storage_strategy
---------------------
- This adapter does not itself manage text persistence.
- BaseVectorAdapter handles text/docstore integration.
- Here we only:
    * Validate the strategy value ("metadata", "docstore", "none").
    * Enforce that "docstore" requires a docstore instance.
    * Report the configured strategy via capabilities.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple, TypeVar

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

T = TypeVar("T")

# Try to import the Weaviate client.
try:  # pragma: no cover - import surface only
    import weaviate  # type: ignore

    # v3: weaviate.Client; v4+: still exposes Client but different API.
    WeaviateClient = getattr(weaviate, "Client", None)  # type: ignore[assignment]

    try:
        # v3 base error
        from weaviate.exceptions import WeaviateBaseError  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        WeaviateBaseError = Exception  # type: ignore[assignment]
except Exception:  # pragma: no cover
    weaviate = None  # type: ignore[assignment]
    WeaviateClient = None  # type: ignore[assignment]
    WeaviateBaseError = Exception  # type: ignore[assignment]


class WeaviateVectorAdapter(BaseVectorAdapter):
    """
    VectorProtocolV1 adapter backed by a Weaviate class.

    Design notes
    ------------
    - Async-first: all Weaviate calls run via `asyncio.to_thread`.
    - Does *not* manage Weaviate schema lifecycle; you create/delete classes
      and configure vector indexes outside.
    - Namespaces are mapped via a special metadata field (namespace_field).
      All operations filter by this field for namespace isolation.
    - Batch queries are implemented as parallel fan-out over single-query calls.
    - Metadata filtering is supported via Weaviate's where clause syntax.

    Limitations
    -----------
    - Namespace support is best-effort via metadata filtering.
      For production multi-tenancy, use separate classes or Weaviate's
      native multi-tenancy features.
    """

    _component = "vector_weaviate"

    def __init__(
        self,
        *,
        class_name: str,
        client: Optional["WeaviateClient"] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        # Vector / metric config
        metric: str = "cosine",
        dimensions: Optional[int] = None,
        # Metadata config
        metadata_field: Optional[str] = "metadata",
        namespace_field: str = "_namespace",
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
        if weaviate is None or WeaviateClient is None:
            raise RuntimeError(
                "WeaviateVectorAdapter requires the `weaviate-client` Python package. "
                "Install via `pip install weaviate-client`."
            )

        if not class_name or not isinstance(class_name, str):
            raise BadRequest(
                "class_name must be a non-empty string",
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
            url = url or os.getenv("WEAVIATE_URL") or "http://localhost:8080"
            api_key = api_key or os.getenv("WEAVIATE_API_KEY")

            client_kwargs: Dict[str, Any] = {"url": url}
            # For API-key auth, we use the v3-style `weaviate.AuthApiKey` if available.
            if api_key:
                try:  # pragma: no cover - auth helper import
                    auth_class = getattr(weaviate, "AuthApiKey", None)
                    if auth_class is not None:
                        client_kwargs["auth_client_secret"] = auth_class(api_key=api_key)
                    else:
                        # Fallback: pass api_key through; some deployments wrap this differently.
                        client_kwargs["auth_client_secret"] = api_key
                except Exception:
                    client_kwargs["auth_client_secret"] = api_key

            try:
                client = WeaviateClient(**client_kwargs)  # type: ignore[call-arg]
            except Exception as exc:  # noqa: BLE001
                raise AuthError(
                    f"Failed to create Weaviate client for url={url}: {exc}",
                    code="WEAVIATE_AUTH",
                )

        # Verify that the client looks like a v3-style GraphQL client.
        missing_attrs = [attr for attr in ("query", "data_object") if not hasattr(client, attr)]
        if missing_attrs:
            raise BadRequest(
                "WeaviateVectorAdapter requires a v3-style weaviate.Client "
                "with 'query' and 'data_object' attributes",
                code="BAD_CONFIG",
                details={"missing_attributes": missing_attrs},
            )

        # Adapter-local configuration
        self._client: WeaviateClient = client  # type: ignore[assignment]
        self._class_name = class_name
        self._metadata_field = metadata_field
        self._namespace_field = namespace_field
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

        # Track known namespaces for validation
        self._known_namespaces: Set[str] = set()

        # Metric mapping: vector protocol metric → semantics for distance conversion
        self._metric_mode = self._metric  # "cosine", "euclidean", "dotproduct"

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

    def _get_client(self) -> WeaviateClient:
        """Return the underlying Weaviate client."""
        return self._client

    @staticmethod
    async def _run_in_thread(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Run blocking client calls on a worker thread."""
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
        if getattr(ctx, "request_id", None):
            details.setdefault("request_id", ctx.request_id)
        if getattr(ctx, "traceparent", None):
            details.setdefault("traceparent", ctx.traceparent)
        return details

    def _get_ctx_timeout_s(self, ctx: Optional[OperationContext]) -> Optional[float]:
        """
        Derive a per-call timeout in seconds from ctx.deadline_ms (via remaining_ms).

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

    async def _call_weaviate(
        self,
        *,
        op: str,
        func: Callable[..., T],
        ctx: Optional[OperationContext],
        **kwargs: Any,
    ) -> T:
        """
        Execute a Weaviate SDK call with ctx-aware timeout and error translation.

        Behavior:
            - Uses `_run_in_thread` to keep the adapter async-first.
            - Applies a best-effort per-call timeout derived from ctx.deadline_ms.
            - Maps asyncio.TimeoutError → TransientNetwork with ctx-enriched details.
            - All other errors are translated via `_translate_error`.
        """
        timeout_s = self._get_ctx_timeout_s(ctx)
        try:
            if timeout_s is not None:
                return await asyncio.wait_for(
                    self._run_in_thread(func, **kwargs),
                    timeout=timeout_s,
                )
            return await self._run_in_thread(func, **kwargs)
        except asyncio.TimeoutError as exc:
            raise TransientNetwork(
                "Weaviate operation timed out",
                code="TRANSIENT_NETWORK",
                retry_after_ms=500,
                details=self._attach_ctx_details({"op": op}, ctx),
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise self._translate_error(exc, op=op, ctx=ctx) from exc

    def _convert_distance(self, raw_distance: float) -> Tuple[float, float]:
        """
        Convert Weaviate's distance into (similarity, distance).

        Weaviate returns a *distance*, not a similarity. Semantics:

        - cosine:
            distance ≈ 1 - cosine_similarity
            → similarity ≈ 1 - distance
        - euclidean:
            distance is euclidean; similarity = 1 / (1 + distance)
        - dotproduct:
            We treat distance as derived from similarity; for lack of a
            standardized mapping, we interpret:
                similarity = -distance
        """
        d = float(raw_distance)
        if self._metric_mode == "cosine":
            distance = max(0.0, d)
            similarity = 1.0 - distance
            return similarity, distance
        if self._metric_mode == "euclidean":
            distance = max(0.0, d)
            similarity = 1.0 / (1.0 + distance)
            return similarity, distance
        if self._metric_mode == "dotproduct":
            distance = d
            similarity = -distance
            return similarity, distance
        # Fallback
        return d, d

    def _translate_error(
        self,
        err: Exception,
        *,
        op: str,
        ctx: Optional[OperationContext],
    ) -> VectorAdapterError:
        """
        Map Weaviate exceptions into normalized VectorAdapterError types.

        Note:
            We intentionally treat "class not found"/404 as IndexNotReady to keep
            the error retryable in environments where schema lifecycle may lag
            behind adapter startup. Callers that want stricter behavior can
            enforce class existence at provisioning time.

        The returned errors include SIEM-safe ctx metadata (request_id, traceparent).
        """
        msg = str(err) or f"Weaviate error during {op}"
        logger.debug("Weaviate error in %s: %r", op, err)

        # Specific Weaviate exception type (best-effort).
        if isinstance(err, WeaviateBaseError):
            status = getattr(err, "status_code", None)
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
                or "quota" in lowered
                or status_int == 429
            ):
                return ResourceExhausted(
                    "Weaviate rate limit exceeded",
                    code="RESOURCE_EXHAUSTED",
                    retry_after_ms=500,
                    details=self._attach_ctx_details({"op": op}, ctx),
                )

            # Auth / permission
            if (
                "unauthorized" in lowered
                or "forbidden" in lowered
                or "api key" in lowered
                or status_int in (401, 403)
            ):
                return AuthError(
                    "Weaviate authentication/authorization error",
                    code="AUTH_ERROR",
                    details=self._attach_ctx_details({"op": op}, ctx),
                )

            # Class not found / index not ready
            if (
                "class" in lowered
                and "not found" in lowered
            ) or status_int == 404:
                return IndexNotReady(
                    "Weaviate class/schema not ready",
                    code="INDEX_NOT_READY",
                    retry_after_ms=1000,
                    details=self._attach_ctx_details({"op": op}, ctx),
                )

            # Timeouts / transient
            if (
                "timeout" in lowered
                or "temporarily unavailable" in lowered
                or "gateway timeout" in lowered
            ):
                return TransientNetwork(
                    "Weaviate transient network error",
                    code="TRANSIENT_NETWORK",
                    retry_after_ms=500,
                    details=self._attach_ctx_details({"op": op}, ctx),
                )

            # Validation / bad request
            if (
                "invalid" in lowered
                or "bad request" in lowered
                or "schema validation" in lowered
                or status_int in (400, 422)
            ):
                return BadRequest(
                    msg,
                    code="BAD_REQUEST",
                    details=self._attach_ctx_details({"op": op}, ctx),
                )

            # Server-side or unknown errors
            if status_int is not None and status_int >= 500:
                return Unavailable(
                    "Weaviate service unavailable",
                    code="UNAVAILABLE",
                    retry_after_ms=1000,
                    details=self._attach_ctx_details({"op": op}, ctx),
                )

            # Generic WeaviateBaseError
            return Unavailable(
                msg,
                code="UNAVAILABLE",
                details=self._attach_ctx_details({"op": op}, ctx),
            )

        # Generic transport/network issues
        lowered_generic = msg.lower()
        if "timeout" in lowered_generic:
            return TransientNetwork(
                "Weaviate network timeout",
                code="TRANSIENT_NETWORK",
                retry_after_ms=500,
                details=self._attach_ctx_details({"op": op}, ctx),
            )
        if "connection" in lowered_generic or "connection refused" in lowered_generic:
            return TransientNetwork(
                "Weaviate connection error",
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

    def _build_namespace_filter(self, namespace: str) -> Dict[str, Any]:
        """Build a Weaviate where filter for namespace isolation."""
        return {
            "path": [self._namespace_field],
            "operator": "Equal",
            "valueText": namespace,
        }

    def _build_filter_expression(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate dictionary filter into Weaviate GraphQL where filter.

        Weaviate uses nested dict structure:
        {
            "path": ["metadata", "category"],
            "operator": "Equal",
            "valueText": "shoes"
        }

        Supported operators:
            - Logical: $and, $or, $not
            - Comparison: $eq, $ne, $gt, $gte, $lt, $lte

        Example Input:
            {"category": "shoes", "price": {"$lt": 100}}

        Example Output:
            {
                "operator": "And",
                "operands": [
                    {
                        "path": ["metadata", "category"],
                        "operator": "Equal",
                        "valueText": "shoes"
                    },
                    {
                        "path": ["metadata", "price"],
                        "operator": "LessThan",
                        "valueInt": 100
                    }
                ]
            }
        """
        if not filters:
            return {}

        if not self._metadata_field:
            raise NotSupported(
                "Metadata filtering requires a configured `metadata_field`",
                code="BAD_CONFIG",
            )

        operands: List[Dict[str, Any]] = []

        for key, value in filters.items():
            # Handle logical operators
            if key == "$and":
                if not isinstance(value, list):
                    raise BadRequest("$and value must be a list")
                sub_filters = [self._build_filter_expression(f) for f in value if f]
                if sub_filters:
                    if len(sub_filters) == 1:
                        operands.append(sub_filters[0])
                    else:
                        operands.append(
                            {
                                "operator": "And",
                                "operands": sub_filters,
                            }
                        )
                continue

            if key == "$or":
                if not isinstance(value, list):
                    raise BadRequest("$or value must be a list")
                sub_filters = [self._build_filter_expression(f) for f in value if f]
                if sub_filters:
                    if len(sub_filters) == 1:
                        operands.append(sub_filters[0])
                    else:
                        operands.append(
                            {
                                "operator": "Or",
                                "operands": sub_filters,
                            }
                        )
                continue

            if key == "$not":
                if not isinstance(value, dict):
                    raise BadRequest("$not value must be a dict")
                sub_filter = self._build_filter_expression(value)
                if sub_filter:
                    operands.append(
                        {
                            "operator": "Not",
                            "operands": [sub_filter],
                        }
                    )
                continue

            # Field path for nested metadata
            path = [self._metadata_field, key]

            # Handle comparison operators or implicit equality
            if isinstance(value, dict):
                for op, op_val in value.items():
                    condition: Dict[str, Any] = {"path": path}

                    if op == "$eq":
                        condition["operator"] = "Equal"
                    elif op == "$ne":
                        condition["operator"] = "NotEqual"
                    elif op == "$gt":
                        condition["operator"] = "GreaterThan"
                    elif op == "$gte":
                        condition["operator"] = "GreaterThanEqual"
                    elif op == "$lt":
                        condition["operator"] = "LessThan"
                    elif op == "$lte":
                        condition["operator"] = "LessThanEqual"
                    else:
                        raise BadRequest(f"unsupported filter operator: {op}")

                    # Add value with correct type key
                    if isinstance(op_val, bool):
                        condition["valueBoolean"] = op_val
                    elif isinstance(op_val, int):
                        condition["valueInt"] = op_val
                    elif isinstance(op_val, float):
                        condition["valueNumber"] = op_val
                    elif isinstance(op_val, str):
                        condition["valueText"] = op_val
                    elif op_val is None:
                        # Weaviate doesn't have explicit null handling in where clause
                        # Skip or raise error
                        continue
                    else:
                        condition["valueText"] = str(op_val)

                    operands.append(condition)
            else:
                # Implicit equality
                condition = {
                    "path": path,
                    "operator": "Equal",
                }

                if isinstance(value, bool):
                    condition["valueBoolean"] = value
                elif isinstance(value, int):
                    condition["valueInt"] = value
                elif isinstance(value, float):
                    condition["valueNumber"] = value
                elif isinstance(value, str):
                    condition["valueText"] = value
                elif value is None:
                    continue
                else:
                    condition["valueText"] = str(value)

                operands.append(condition)

        if not operands:
            return {}
        if len(operands) == 1:
            return operands[0]
        return {
            "operator": "And",
            "operands": operands,
        }

    def _combine_filters(self, *filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Combine multiple filters with AND logic."""
        valid_filters = [f for f in filters if f]
        if not valid_filters:
            return None
        if len(valid_filters) == 1:
            return valid_filters[0]
        return {
            "operator": "And",
            "operands": valid_filters,
        }

    # ------------------------------------------------------------------ #
    # BaseVectorAdapter backend hooks
    # ------------------------------------------------------------------ #

    async def _do_capabilities(self) -> VectorCapabilities:
        """
        Report Weaviate-backed capabilities.

        Note:
            - `max_dimensions` is taken from configuration (if provided).
            - `max_batch_size` is a soft planning hint, but we also enforce it
              as a hard limit on upsert/delete batches to protect the backend.
            - Namespaces are supported via metadata field filtering.
            - Metadata filtering is supported via Weaviate's where clause.
        """
        version = getattr(weaviate, "__version__", "unknown") if weaviate else "unknown"
        return VectorCapabilities(
            server="weaviate",
            version=version,
            max_dimensions=self._dimensions or 0,
            supported_metrics=(self._metric,),
            supports_namespaces=True,
            supports_metadata_filtering=True,
            supports_batch_operations=True,
            max_batch_size=self._max_batch_size or None,
            supports_index_management=False,  # schema lifecycle managed externally
            idempotent_writes=True,  # we treat upserts as idempotent on IDs
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

        # Build metadata filter expression if provided.
        user_filter = None
        if spec.filter:
            try:
                user_filter = self._build_filter_expression(spec.filter)
            except VectorAdapterError:
                raise
            except Exception as e:  # noqa: BLE001
                raise BadRequest(
                    f"Invalid filter expression: {e}",
                    details={"filter": spec.filter},
                )

        # Build namespace filter
        namespace_filter = None
        if spec.namespace:
            namespace_filter = self._build_namespace_filter(spec.namespace)

        # Combine filters
        combined_filter = self._combine_filters(namespace_filter, user_filter)

        # Build GraphQL get() fields list.
        props: List[str] = []

        if spec.include_metadata and self._metadata_field:
            props.append(self._metadata_field)

        # Always include namespace field to verify isolation
        props.append(self._namespace_field)

        # Build _additional selection dynamically.
        additional_fields: List[str] = ["id", "distance"]
        if self._metric_mode == "cosine":
            additional_fields.append("certainty")
        if spec.include_vectors:
            additional_fields.append("vector")

        additional_str = " ".join(additional_fields)
        props.append(f"_additional {{ {additional_str} }}")

        vector_arg = list(map(float, spec.vector))

        def do_query() -> Dict[str, Any]:
            # v3-style client.query GraphQL API.
            q = (
                client.query.get(self._class_name, props)
                .with_near_vector({"vector": vector_arg})
                .with_limit(spec.top_k)
            )
            if combined_filter:
                q = q.with_where(combined_filter)
            return q.do()

        try:
            resp = await self._call_weaviate(
                op="query",
                func=do_query,
                ctx=ctx,
            )
        except VectorAdapterError:
            raise

        # Expected response shape:
        # {
        #   "data": {
        #     "Get": {
        #       "<class_name>": [ { <props>, "_additional": {...} }, ... ]
        #     }
        #   }
        # }
        data = self._safe_get(resp, "data", {}) or {}
        get_block = self._safe_get(data, "Get", {}) or {}
        hits: Sequence[Any] = self._safe_get(get_block, self._class_name, []) or []

        matches: List[VectorMatch] = []
        for obj in hits:
            if not isinstance(obj, Mapping):
                continue

            add = obj.get("_additional") or {}
            if not isinstance(add, Mapping):
                add = {}

            vid = add.get("id")
            if vid is None:
                # Fallback: use Weaviate internal UUID on object if present.
                vid = obj.get("id") or obj.get("uuid")
            if vid is None:
                continue

            raw_distance = add.get("distance", 0.0)
            similarity, distance = self._convert_distance(float(raw_distance or 0.0))

            # Vector, if requested.
            if spec.include_vectors:
                values = add.get("vector") or []
                vec_list = list(values) if values else []
            else:
                vec_list = []

            # Metadata, if requested and configured.
            if spec.include_metadata and self._metadata_field:
                meta = obj.get(self._metadata_field)
            else:
                meta = None

            vector = Vector(
                id=str(vid),
                vector=vec_list,
                metadata=meta,
                namespace=spec.namespace,
                text=None,
            )
            matches.append(
                VectorMatch(
                    vector=vector,
                    score=similarity,
                    distance=distance,
                )
            )

        return QueryResult(
            matches=matches,
            query_vector=list(spec.vector),
            namespace=spec.namespace,
            total_matches=len(hits),
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
        # Normalize namespace on each QuerySpec to the batch namespace.
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

        # Per-item dimensionality check with partial failures.
        failures: List[Dict[str, Any]] = []

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

        # Filter out failed vectors
        valid_vectors = [
            v
            for i, v in enumerate(spec.vectors)
            if not any(f.get("index") == i for f in failures)
        ]

        def do_batch_upsert() -> Tuple[int, List[Dict[str, Any]]]:
            """
            Perform upserts sequentially in a worker thread, capturing per-id
            failures instead of failing the whole batch on the first error.
            """
            upserted = 0
            failures_local: List[Dict[str, Any]] = []

            for v in valid_vectors:
                uuid = str(v.id)

                # Build properties payload.
                properties: Dict[str, Any] = {}

                # Add namespace field
                if spec.namespace:
                    properties[self._namespace_field] = spec.namespace

                # Add metadata
                if self._metadata_field and v.metadata is not None:
                    properties[self._metadata_field] = v.metadata

                vec = [float(x) for x in v.vector]

                # First try create.
                try:
                    client.data_object.create(  # type: ignore[attr-defined]
                        data_object=properties,
                        class_name=self._class_name,
                        uuid=uuid,
                        vector=vec,
                    )
                    upserted += 1
                    continue
                except Exception as create_exc:  # noqa: BLE001
                    lowered = str(create_exc).lower()
                    # Only fall back to replace when it looks like an "exists" error.
                    if "already exists" not in lowered and "exists" not in lowered:
                        failures_local.append(
                            {
                                "id": uuid,
                                "stage": "create",
                                "error": str(create_exc),
                            }
                        )
                        continue

                # Replace existing object when create failed due to existence.
                try:
                    client.data_object.replace(  # type: ignore[attr-defined]
                        uuid=uuid,
                        data_object=properties,
                        class_name=self._class_name,
                        vector=vec,
                    )
                    upserted += 1
                except Exception as replace_exc:  # noqa: BLE001
                    failures_local.append(
                        {
                            "id": uuid,
                            "stage": "replace",
                            "error": str(replace_exc),
                        }
                    )

            return upserted, failures_local

        try:
            upserted_count, upsert_failures = await self._call_weaviate(
                op="upsert",
                func=do_batch_upsert,
                ctx=ctx,
            )
        except VectorAdapterError:
            raise

        # Combine dimension failures and upsert failures
        all_failures = failures + upsert_failures
        failed_count = len(all_failures)

        # Track namespace
        if spec.namespace and upserted_count > 0:
            self._known_namespaces.add(spec.namespace)

        return UpsertResult(
            upserted_count=upserted_count,
            failed_count=failed_count,
            failures=all_failures,
        )

    # ------------------------------ delete --------------------------------- #

    async def _do_delete(
        self,
        spec: DeleteSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> DeleteResult:
        client = self._get_client()

        # Enforce batch size ceiling if configured.
        if spec.ids and self._max_batch_size and len(spec.ids) > self._max_batch_size:
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

        if spec.ids:
            # Delete by IDs with namespace filter
            ids = [str(i) for i in spec.ids]

            def do_batch_delete() -> Tuple[int, List[Dict[str, Any]]]:
                """
                Perform deletes sequentially in a worker thread, capturing per-id
                failures while treating "not found" as best-effort success.
                """
                deleted = 0
                failures_local: List[Dict[str, Any]] = []

                for uid in ids:
                    # Verify namespace match before deleting
                    if spec.namespace:
                        try:
                            # Check if object belongs to namespace
                            obj = client.data_object.get_by_id(  # type: ignore[attr-defined]
                                uuid=uid,
                                class_name=self._class_name,
                            )
                            if obj:
                                properties = obj.get("properties", {})
                                obj_namespace = properties.get(self._namespace_field)
                                if obj_namespace != spec.namespace:
                                    # Object exists but in different namespace - skip
                                    continue
                        except Exception:  # noqa: BLE001
                            # If we can't check, proceed with delete attempt
                            pass

                    try:
                        client.data_object.delete(  # type: ignore[attr-defined]
                            uuid=uid,
                            class_name=self._class_name,
                        )
                        deleted += 1
                    except Exception as exc:  # noqa: BLE001
                        lowered = str(exc).lower()
                        if "not found" in lowered:
                            logger.debug(
                                "Weaviate delete(%s) treated as best-effort: %s",
                                uid,
                                exc,
                            )
                            # Not counted as failure, to preserve best-effort semantics.
                            continue
                        failures_local.append(
                            {
                                "id": uid,
                                "stage": "delete",
                                "error": str(exc),
                            }
                        )
                return deleted, failures_local

            try:
                deleted_count, failures_list = await self._call_weaviate(
                    op="delete",
                    func=do_batch_delete,
                    ctx=ctx,
                )
            except VectorAdapterError:
                raise

        elif spec.filter:
            # Delete by filter with namespace
            try:
                user_filter = self._build_filter_expression(spec.filter)
            except VectorAdapterError:
                raise
            except Exception as e:  # noqa: BLE001
                raise BadRequest(
                    f"Invalid filter for delete: {e}",
                    details={"filter": spec.filter},
                )

            # Build namespace filter
            namespace_filter = None
            if spec.namespace:
                namespace_filter = self._build_namespace_filter(spec.namespace)

            # Combine filters
            combined_filter = self._combine_filters(namespace_filter, user_filter)

            if not combined_filter:
                return DeleteResult(
                    deleted_count=0,
                    failed_count=0,
                    failures=[],
                )

            def do_filter_delete() -> Tuple[int, List[Dict[str, Any]]]:
                """Delete by filter using batch delete."""
                result = client.batch.delete_objects(  # type: ignore[attr-defined]
                    class_name=self._class_name,
                    where=combined_filter,
                )
                deleted = 0
                failures_local: List[Dict[str, Any]] = []

                if isinstance(result, dict):
                    deleted = result.get("successful", 0)
                    failed = result.get("failed", 0)
                    if failed > 0:
                        failures_local.append(
                            {
                                "id": "batch",
                                "stage": "filter_delete",
                                "error": f"{failed} objects failed to delete",
                            }
                        )

                return deleted, failures_local

            try:
                deleted_count, failures_list = await self._call_weaviate(
                    op="delete",
                    func=do_filter_delete,
                    ctx=ctx,
                )
            except VectorAdapterError:
                raise
        else:
            # BaseVectorAdapter should already enforce ids|filter presence.
            raise BadRequest(
                "must provide either ids or filter for deletion",
                code="BAD_REQUEST",
            )

        failed_count = len(failures_list)

        return DeleteResult(
            deleted_count=deleted_count,
            failed_count=failed_count,
            failures=failures_list,
        )

    # ------------------------- namespace management ------------------------ #

    async def _do_create_namespace(
        self,
        spec: NamespaceSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> NamespaceResult:
        """
        Create a namespace by tracking it internally.

        Weaviate namespaces are virtual - they're just a metadata field value.
        No actual schema changes are needed.
        """
        # Validate metric matches
        if spec.distance_metric.lower() != self._metric:
            raise BadRequest(
                f"distance_metric '{spec.distance_metric}' does not match "
                f"configured metric '{self._metric}'",
                code="BAD_CONFIG",
                details={"metric": self._metric},
            )

        # Validate dimensions match
        if self._dimensions and spec.dimensions != self._dimensions:
            raise BadRequest(
                f"dimensions {spec.dimensions} do not match configured "
                f"dimensions {self._dimensions}",
                code="BAD_CONFIG",
                details={"configured_dimensions": self._dimensions},
            )

        # Track namespace
        created = spec.namespace not in self._known_namespaces
        self._known_namespaces.add(spec.namespace)

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
        Delete a namespace by removing all objects with that namespace value.
        """
        client = self._get_client()

        # Build namespace filter
        namespace_filter = self._build_namespace_filter(namespace)

        def do_namespace_delete() -> bool:
            """Delete all objects in namespace."""
            try:
                client.batch.delete_objects(  # type: ignore[attr-defined]
                    class_name=self._class_name,
                    where=namespace_filter,
                )
                return True
            except Exception as exc:  # noqa: BLE001
                lowered = str(exc).lower()
                if "not found" in lowered:
                    return False
                raise exc

        try:
            existed = await self._call_weaviate(
                op="delete_namespace",
                func=do_namespace_delete,
                ctx=ctx,
            )
        except Exception as exc:  # noqa: BLE001
            err = self._translate_error(exc, op="delete_namespace", ctx=ctx)
            if isinstance(err, (IndexNotReady, BadRequest)):
                logger.debug(
                    "delete_namespace(%s) treated as best-effort: %s",
                    namespace,
                    exc,
                )
                existed = False
            else:
                raise err

        # Update tracking
        self._known_namespaces.discard(namespace)

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
        Lightweight health probe.

        Behavior:
            - Never raises; returns an `ok` flag and a small namespace summary.
            - Any remote error is captured and reflected as ok=False.
        """
        client = self._get_client()
        version = getattr(weaviate, "__version__", "unknown") if weaviate else "unknown"

        try:
            # v3 client exposes is_ready(); if missing, we treat it as best-effort.
            def probe_ready() -> bool:
                ready_fn = getattr(client, "is_ready", None)
                if callable(ready_fn):
                    return bool(ready_fn())
                # Fallback: if we can fetch schema, we consider it ready.
                schema_get = getattr(client, "schema", None)
                if schema_get is not None and hasattr(schema_get, "get"):
                    try:
                        schema_get.get()  # type: ignore[call-arg,attr-defined]
                        return True
                    except Exception:
                        return False
                return True

            is_ok = await self._call_weaviate(
                op="health",
                func=probe_ready,
                ctx=ctx,
            )

            # Build namespaces dict with known namespaces
            namespaces: Dict[str, Any] = {}

            for ns in self._known_namespaces:
                namespaces[ns] = {
                    "dimensions": self._dimensions,
                    "metric": self._metric,
                    "count": None,  # unknown object count per namespace
                    "status": "ok" if is_ok else "unknown",
                }

            # Always include aggregate namespace
            namespaces.setdefault(
                "*",
                {
                    "dimensions": self._dimensions,
                    "metric": self._metric,
                    "count": None,  # unknown total count
                    "status": "ok" if is_ok else "unknown",
                },
            )

            return {
                "ok": bool(is_ok),
                "server": "weaviate",
                "version": version,
                "namespaces": namespaces,
            }
        except Exception as exc:  # noqa: BLE001
            err = self._translate_error(exc, op="health", ctx=ctx)
            logger.warning("Weaviate health check failed: %s", err)
            return {
                "ok": False,
                "server": "weaviate",
                "version": version,
                "namespaces": {},
                "error_code": err.code,
                "error_message": str(err),
            }


__all__ = [
    "WeaviateVectorAdapter",
]
 
