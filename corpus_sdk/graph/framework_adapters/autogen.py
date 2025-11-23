# corpus_sdk/graph/framework_adapters/autogen.py
# SPDX-License-Identifier: Apache-2.0

"""
AutoGen adapter for Corpus Graph protocol.

This module exposes a Corpus `GraphProtocolV1` implementation as an
AutoGen-friendly client, with:

- Sync + async query APIs
- Sync + async streaming query APIs
- Proper integration with Corpus GraphProtocolV1
- OperationContext propagation derived from AutoGen conversation / metadata
- Error-context enrichment for observability and debugging
- Orchestration, translation, and async→sync bridging via GraphTranslator

Design philosophy
-----------------
- Protocol-first: AutoGen is a thin skin over the Corpus graph adapter.
- All heavy lifting (deadlines, breakers, rate limiting, caching, etc.) lives
  in the underlying `BaseGraphAdapter` / `GraphProtocolV1` implementation.
- This layer focuses on:
    * Translating AutoGen conversation → OperationContext
    * Building raw query / mutation shapes for GraphTranslator
    * Delegating all sync/async and streaming orchestration to GraphTranslator

Responsibilities
----------------
- Provide a convenient, AutoGen-oriented client for graph operations
- Keep all graph operations going through `GraphTranslator` so that
  async→sync bridging, streaming, and error-context logic are centralized
- Preserve protocol-level types (`QueryResult`, `QueryChunk`, etc.) for
  AutoGen callers

Non-responsibilities
--------------------
- Backend-specific graph behavior (lives in graph adapters)
- AutoGen agent orchestration and conversation logic
- MMR and diversification details (handled inside GraphTranslator)
"""

from __future__ import annotations

import logging
from functools import cached_property, wraps
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Callable,
    cast,
)

from corpus_sdk.core.context_translation import (
    from_autogen as core_ctx_from_autogen,
)
from corpus_sdk.core.error_context import attach_context
from corpus_sdk.graph.framework_adapters.common.graph_translation import (
    DefaultGraphFrameworkTranslator,
    GraphTranslator,
    create_graph_translator,
)
from corpus_sdk.graph.framework_adapters.common.framework_utils import (
    CoercionErrorCodes,
)
from corpus_sdk.graph.graph_base import (
    BadRequest,
    BatchOperation,
    BatchResult,
    BulkVerticesResult,
    BulkVerticesSpec,
    DeleteEdgesSpec,
    DeleteNodesSpec,
    DeleteResult,
    GraphCapabilities,
    GraphProtocolV1,
    GraphSchema,
    OperationContext,
    QueryChunk,
    QueryResult,
    UpsertEdgesSpec,
    UpsertNodesSpec,
    UpsertResult,
)

logger = logging.getLogger(__name__)

# Type variables for decorators
T = TypeVar("T")
R = TypeVar("R")


# ---------------------------------------------------------------------------
# Error code constants (standalone, NOT subclassing CoercionErrorCodes)
# ---------------------------------------------------------------------------


class ErrorCodes:
    """
    Error code constants for AutoGen graph adapter.

    This intentionally does not subclass `CoercionErrorCodes`. Shared utilities
    that rely on symbolic names can still refer to this class explicitly while
    the underlying CoercionErrorCodes remain available from the common
    framework utilities for cross-framework alignment.
    """

    BAD_OPERATION_CONTEXT = "BAD_OPERATION_CONTEXT"
    BAD_TRANSLATED_SCHEMA = "BAD_TRANSLATED_SCHEMA"
    BAD_HEALTH_RESULT = "BAD_HEALTH_RESULT"
    BAD_TRANSLATED_RESULT = "BAD_TRANSLATED_RESULT"
    BAD_TRANSLATED_CHUNK = "BAD_TRANSLATED_CHUNK"
    BAD_UPSERT_RESULT = "BAD_UPSERT_RESULT"
    BAD_DELETE_RESULT = "BAD_DELETE_RESULT"
    BAD_BULK_VERTICES_RESULT = "BAD_BULK_VERTICES_RESULT"
    BAD_BATCH_RESULT = "BAD_BATCH_RESULT"
    BAD_ADAPTER_RESULT = "BAD_ADAPTER_RESULT"


# ---------------------------------------------------------------------------
# Error-context decorators with dynamic context extraction
# ---------------------------------------------------------------------------


def _extract_dynamic_context(
    instance: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    operation: str,
) -> Dict[str, Any]:
    """
    Extract rich dynamic context from an AutoGen graph client call.

    Best-effort only (never raises); captures:

    - client_class / adapter_type
    - framework_version (if configured)
    - query_len / query_preview for query / streaming operations
    - batch_ops_count for batch operations
    - basic query options (namespace / dialect / timeout_ms)
    - presence / type hints for conversation / extra_context
    """
    dynamic_ctx: Dict[str, Any] = {
        "client_class": type(instance).__name__,
        "client_operation": operation,
    }

    # Adapter / framework metadata
    try:
        graph_adapter = getattr(instance, "_graph", None)
        if graph_adapter is not None:
            dynamic_ctx["adapter_type"] = type(graph_adapter).__name__
    except Exception:  # noqa: BLE001
        # Best-effort only; never let metadata extraction break user code.
        pass

    framework_version = getattr(instance, "_framework_version", None)
    if framework_version is not None:
        dynamic_ctx["framework_version"] = framework_version

    # Query / streaming metrics
    if operation.startswith(("query_", "stream_query_")):
        query_arg: Optional[str] = None

        # Common shape: first positional arg after self is the query string
        if args and isinstance(args[0], str):
            query_arg = args[0]
        elif isinstance(kwargs.get("query"), str):
            query_arg = kwargs["query"]

        if isinstance(query_arg, str):
            dynamic_ctx["query_len"] = len(query_arg)
            dynamic_ctx["query_preview"] = query_arg[:200]

    # Batch operation metrics
    if operation.startswith("batch_"):
        ops = None
        if "ops" in kwargs and isinstance(kwargs["ops"], list):
            ops = kwargs["ops"]
        elif args and isinstance(args[0], list):
            ops = args[0]
        if ops is not None:
            dynamic_ctx["batch_ops_count"] = len(ops)

    # Common query options
    for key in ("namespace", "dialect", "timeout_ms"):
        if key in kwargs and kwargs[key] is not None:
            dynamic_ctx[key] = kwargs[key]

    # AutoGen conversation / extra_context hints (shape-level only)
    conversation = kwargs.get("conversation")
    if conversation is not None:
        dynamic_ctx["has_conversation"] = True
        dynamic_ctx["conversation_type"] = type(conversation).__name__

    extra_context = kwargs.get("extra_context")
    if isinstance(extra_context, Mapping):
        dynamic_ctx["has_extra_context"] = True
        dynamic_ctx["extra_context_keys"] = list(extra_context.keys())[:10]

    return dynamic_ctx


def _create_error_context_decorator(
    operation: str,
    is_async: bool = False,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Factory for creating error-context decorators with rich per-call metrics.

    This mirrors the pattern used in other framework adapters (LangChain,
    LlamaIndex, Semantic Kernel) for consistent observability while keeping
    the AutoGen layer thin and protocol-first.
    """

    def decorator_factory(
        **static_context: Any,
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            if is_async:

                @wraps(func)
                async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
                    dynamic_context = _extract_dynamic_context(
                        self,
                        args,
                        kwargs,
                        operation,
                    )
                    full_context = {**static_context, **dynamic_context}
                    try:
                        return await func(self, *args, **kwargs)
                    except Exception as exc:  # noqa: BLE001
                        attach_context(
                            exc,
                            framework="autogen",
                            operation=f"graph_{operation}",
                            **full_context,
                        )
                        raise

                return async_wrapper
            else:

                @wraps(func)
                def sync_wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
                    dynamic_context = _extract_dynamic_context(
                        self,
                        args,
                        kwargs,
                        operation,
                    )
                    full_context = {**static_context, **dynamic_context}
                    try:
                        return func(self, *args, **kwargs)
                    except Exception as exc:  # noqa: BLE001
                        attach_context(
                            exc,
                            framework="autogen",
                            operation=f"graph_{operation}",
                            **full_context,
                        )
                        raise

                return sync_wrapper

        return decorator

    return decorator_factory


def with_graph_error_context(
    operation: str,
    **static_context: Any,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for sync methods with rich dynamic context extraction."""
    return _create_error_context_decorator(operation, is_async=False)(**static_context)


def with_async_graph_error_context(
    operation: str,
    **static_context: Any,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for async methods with rich dynamic context extraction."""
    return _create_error_context_decorator(operation, is_async=True)(**static_context)


# Backwards-compatible aliases (if callers were using old names)
with_error_context = with_graph_error_context
with_async_error_context = with_async_graph_error_context


# ---------------------------------------------------------------------------
# Public protocol
# ---------------------------------------------------------------------------


class AutoGenGraphClientProtocol(Protocol):
    """
    Protocol representing the minimal AutoGen-aware graph client interface
    implemented by this module.

    This structural protocol allows callers to type against the graph client
    without depending on the concrete `CorpusAutoGenGraphClient` class.
    """

    # Capabilities / schema / health

    def capabilities(self) -> Dict[str, Any]:
        ...

    async def acapabilities(self) -> Dict[str, Any]:
        ...

    def get_schema(
        self,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> GraphSchema:
        ...

    async def aget_schema(
        self,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> GraphSchema:
        ...

    def health(
        self,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        ...

    async def ahealth(
        self,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        ...

    # Query

    def query(
        self,
        query: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        dialect: Optional[str] = None,
        namespace: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> QueryResult:
        ...

    async def aquery(
        self,
        query: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        dialect: Optional[str] = None,
        namespace: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> QueryResult:
        ...

    def stream_query(
        self,
        query: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        dialect: Optional[str] = None,
        namespace: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[QueryChunk]:
        ...

    async def astream_query(
        self,
        query: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        dialect: Optional[str] = None,
        namespace: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> AsyncIterator[QueryChunk]:
        ...

    # Upsert

    def upsert_nodes(
        self,
        spec: UpsertNodesSpec,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        ...

    async def aupsert_nodes(
        self,
        spec: UpsertNodesSpec,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        ...

    def upsert_edges(
        self,
        spec: UpsertEdgesSpec,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        ...

    async def aupsert_edges(
        self,
        spec: UpsertEdgesSpec,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        ...

    # Delete

    def delete_nodes(
        self,
        spec: DeleteNodesSpec,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        ...

    async def adelete_nodes(
        self,
        spec: DeleteNodesSpec,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        ...

    def delete_edges(
        self,
        spec: DeleteEdgesSpec,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        ...

    async def adelete_edges(
        self,
        spec: DeleteEdgesSpec,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        ...

    # Bulk / batch

    def bulk_vertices(
        self,
        spec: BulkVerticesSpec,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BulkVerticesResult:
        ...

    async def abulk_vertices(
        self,
        spec: BulkVerticesSpec,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BulkVerticesResult:
        ...

    def batch(
        self,
        ops: List[BatchOperation],
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BatchResult:
        ...

    async def abatch(
        self,
        ops: List[BatchOperation],
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BatchResult:
        ...


class CorpusAutoGenGraphClient:
    """
    AutoGen-oriented client wrapper around a Corpus `GraphProtocolV1`.

    This is a thin integration layer that:

    - Translates AutoGen conversation / metadata into a Corpus `OperationContext`
      using `core_ctx_from_autogen`.
    - Uses `GraphTranslator` (with an AutoGen-specific framework translator) to:
        * Build Graph*Spec objects from simple inputs
        * Execute sync + async graph operations
        * Orchestrate streaming with proper cancellation and error handling
    - Delegates all async→sync bridging and streaming glue to GraphTranslator.
    - Attaches rich error context (`attach_context`) on this layer with
      AutoGen-specific hints when failures occur.

    Capabilities / health
    ---------------------
    - The sync `capabilities()` and `health()` methods assume a sync-capable
      adapter implementation for these methods.
    - The async `acapabilities()` / `ahealth()` variants assume the adapter
      exposes proper async methods for these calls.
    """

    class _AutoGenGraphFrameworkTranslator(DefaultGraphFrameworkTranslator):
        """
        AutoGen-specific GraphFrameworkTranslator.

        This translator reuses the common DefaultGraphFrameworkTranslator for
        spec construction and context handling, but deliberately *does not*
        reshape core protocol results:

        - QueryResult is returned as-is
        - QueryChunk is returned as-is
        - BulkVerticesResult is returned as-is
        - BatchResult is returned as-is
        - GraphSchema is returned as-is
        """

        def translate_query_result(
            self,
            result: QueryResult,
            *,
            op_ctx: OperationContext,
            framework_ctx: Optional[Any] = None,
        ) -> QueryResult:
            return result

        def translate_query_chunk(
            self,
            chunk: QueryChunk,
            *,
            op_ctx: OperationContext,
            framework_ctx: Optional[Any] = None,
        ) -> QueryChunk:
            return chunk

        def translate_bulk_vertices_result(
            self,
            result: BulkVerticesResult,
            *,
            op_ctx: OperationContext,
            framework_ctx: Optional[Any] = None,
        ) -> BulkVerticesResult:
            return result

        def translate_batch_result(
            self,
            result: BatchResult,
            *,
            op_ctx: OperationContext,
            framework_ctx: Optional[Any] = None,
        ) -> BatchResult:
            return result

        def translate_schema(
            self,
            schema: GraphSchema,
            *,
            op_ctx: OperationContext,
            framework_ctx: Optional[Any] = None,
        ) -> GraphSchema:
            return schema

    def __init__(
        self,
        *,
        graph_adapter: GraphProtocolV1,
        default_dialect: Optional[str] = None,
        default_namespace: Optional[str] = None,
        default_timeout_ms: Optional[int] = None,
        framework_version: Optional[str] = None,
    ) -> None:
        """
        Initialize an AutoGen-oriented graph client.

        Parameters
        ----------
        graph_adapter:
            Underlying `GraphProtocolV1` implementation.
        default_dialect:
            Optional default query dialect to use when none is provided per call.
        default_namespace:
            Optional default namespace to use when none is provided per call.
        default_timeout_ms:
            Optional default per-query timeout in milliseconds. Used when
            `timeout_ms` is not explicitly passed to query methods.
        framework_version:
            Optional framework version string for observability.
        """
        self._graph: GraphProtocolV1 = graph_adapter
        self._default_dialect: Optional[str] = default_dialect
        self._default_namespace: Optional[str] = default_namespace
        self._default_timeout_ms: Optional[int] = default_timeout_ms
        self._framework_version: Optional[str] = framework_version

    # ------------------------------------------------------------------ #
    # Resource management (context managers)
    # ------------------------------------------------------------------ #

    def __enter__(self) -> CorpusAutoGenGraphClient:
        """Support context manager protocol for resource cleanup."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Clean up resources when exiting context."""
        if hasattr(self._graph, "close"):
            self._graph.close()

    async def __aenter__(self) -> CorpusAutoGenGraphClient:
        """Support async context manager protocol."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Clean up resources when exiting async context."""
        if hasattr(self._graph, "aclose"):
            await self._graph.aclose()

    # ------------------------------------------------------------------ #
    # Translator (lazy, cached) – thin wrapper via create_graph_translator
    # ------------------------------------------------------------------ #

    @cached_property
    def _translator(self) -> GraphTranslator:
        """
        Lazily construct and cache the `GraphTranslator`.

        Uses `create_graph_translator` so registry-based per-framework
        translators remain honored while still allowing our AutoGen-specific
        pass-through translator to be supplied explicitly.
        """
        framework_translator = self._AutoGenGraphFrameworkTranslator()
        return create_graph_translator(
            adapter=self._graph,
            framework="autogen",
            translator=framework_translator,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_ctx(
        self,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Optional[OperationContext]:
        """
        Build an OperationContext from AutoGen-style inputs.

        Expected inputs
        ----------------
        - conversation: AutoGen conversation object (optional)
        - extra_context: Optional mapping merged into attrs (best effort)

        If both are None/empty, returns None and lets downstream helpers
        construct an "empty" OperationContext as needed.
        """
        extra: Dict[str, Any] = dict(extra_context or {})

        if conversation is None and not extra:
            return None

        try:
            ctx = core_ctx_from_autogen(
                conversation,
                framework_version=self._framework_version,
                **extra,
            )
        except Exception as exc:
            attach_context(
                exc,
                framework="autogen",
                operation="context_translation",
            )
            raise

        if not isinstance(ctx, OperationContext):
            raise BadRequest(
                f"from_autogen produced unsupported context type: {type(ctx).__name__}",
                code=ErrorCodes.BAD_OPERATION_CONTEXT,
            )

        return ctx

    @staticmethod
    def _validate_query(query: str) -> None:
        """
        Validate that a query string is non-empty and of the correct type.
        """
        if not isinstance(query, str) or not query.strip():
            raise BadRequest("query must be a non-empty string")

    def _validate_upsert_nodes_spec(self, spec: UpsertNodesSpec) -> None:
        """Validate upsert specification for nodes before processing."""
        if not spec.nodes:
            raise BadRequest("UpsertNodesSpec must contain at least one node")

        for node in spec.nodes:
            if not getattr(node, "id", None):
                raise BadRequest("All nodes must have an ID")

    def _validate_upsert_edges_spec(self, spec: UpsertEdgesSpec) -> None:
        """Validate upsert specification for edges before processing."""
        if not spec.edges:
            raise BadRequest("UpsertEdgesSpec must contain at least one edge")

        for edge in spec.edges:
            if not getattr(edge, "id", None):
                raise BadRequest("All edges must have an ID")

    def _validate_batch_ops(self, ops: List[BatchOperation]) -> None:
        """Validate batch operations before processing."""
        if not ops:
            raise BadRequest("Batch operations list cannot be empty")

        # Check against graph capabilities if available.
        # Only enforce a limit when the backend explicitly sets a positive max.
        caps = self._graph.capabilities()
        max_ops = caps.max_batch_ops
        if max_ops is not None and max_ops > 0 and len(ops) > max_ops:
            raise BadRequest(
                f"Too many batch operations: {len(ops)} (max: {max_ops})",
            )

    def _build_raw_query(
        self,
        query: str,
        *,
        params: Optional[Mapping[str, Any]],
        dialect: Optional[str],
        namespace: Optional[str],
        timeout_ms: Optional[int],
        stream: bool,
    ) -> Mapping[str, Any]:
        """
        Build a raw query mapping suitable for GraphTranslator.

        The common GraphTranslator expects:
            - Either a plain string, or
            - A mapping with:
                * text (str)
                * dialect (optional)
                * params (optional mapping)
                * namespace (optional)
                * timeout_ms (optional)
                * stream (bool)
        """
        effective_dialect = dialect or self._default_dialect
        effective_namespace = namespace or self._default_namespace
        effective_timeout = timeout_ms or self._default_timeout_ms

        raw: Dict[str, Any] = {
            "text": query,
            "params": dict(params or {}),
            "stream": bool(stream),
        }

        if effective_dialect is not None:
            raw["dialect"] = effective_dialect
        if effective_namespace is not None:
            raw["namespace"] = effective_namespace
        if effective_timeout is not None:
            raw["timeout_ms"] = int(effective_timeout)

        return raw

    def _framework_ctx_for_namespace(
        self,
        namespace: Optional[str],
    ) -> Mapping[str, Any]:
        """
        Build a minimal framework_ctx mapping that lets the common translator
        derive a preferred namespace when needed.
        """
        effective_namespace = namespace or self._default_namespace
        return (
            {"namespace": effective_namespace}
            if effective_namespace is not None
            else {}
        )

    def _validate_result_type(
        self,
        result: Any,
        expected_type: type[T],
        operation: str,
        error_code: str,
    ) -> T:
        """
        Validate that a result is of the expected type.

        Args:
            result: The result to validate
            expected_type: The expected type
            operation: Operation name for error message
            error_code: Error code for BadRequest

        Returns:
            The validated result cast to expected type
        """
        if not isinstance(result, expected_type):
            raise BadRequest(
                f"{operation} returned unsupported type: {type(result).__name__}",
                code=error_code,
            )
        return cast(T, result)

    @staticmethod
    def _capabilities_to_dict(caps: GraphCapabilities) -> Dict[str, Any]:
        """
        Convert a GraphCapabilities dataclass into an AutoGen-friendly dict.
        """
        return {
            "server": caps.server,
            "version": caps.version,
            "protocol": caps.protocol,
            "supports_stream_query": caps.supports_stream_query,
            "supported_query_dialects": list(caps.supported_query_dialects or ()),
            "supports_namespaces": caps.supports_namespaces,
            "supports_property_filters": caps.supports_property_filters,
            "supports_bulk_vertices": caps.supports_bulk_vertices,
            "supports_batch": caps.supports_batch,
            "supports_schema": caps.supports_schema,
            "idempotent_writes": caps.idempotent_writes,
            "supports_multi_tenant": caps.supports_multi_tenant,
            "supports_deadline": caps.supports_deadline,
            "max_batch_ops": caps.max_batch_ops,
        }

    # ------------------------------------------------------------------ #
    # Capabilities / schema / health
    # ------------------------------------------------------------------ #

    @with_graph_error_context("capabilities_sync")
    def capabilities(self) -> Dict[str, Any]:
        """
        Sync wrapper around `graph_adapter.capabilities()`.

        Uses GraphTranslator for consistency with other operations.
        """
        caps = self._translator.capabilities()
        return self._capabilities_to_dict(caps)

    @with_async_graph_error_context("capabilities_async")
    async def acapabilities(self) -> Dict[str, Any]:
        """
        Async capabilities accessor with AutoGen-friendly dict output.

        Uses GraphTranslator for consistency with other operations.
        """
        caps = await self._translator.arun_capabilities()
        return self._capabilities_to_dict(caps)

    @with_graph_error_context("get_schema_sync")
    def get_schema(
        self,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> GraphSchema:
        """
        Sync wrapper around `graph_adapter.get_schema(...)`.

        Delegates to GraphTranslator so that async→sync bridging and
        error-context handling are centralized.
        """
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        schema = self._translator.get_schema(
            op_ctx=ctx,
            framework_ctx={},
        )
        return self._validate_result_type(
            schema,
            GraphSchema,
            "GraphTranslator.get_schema",
            ErrorCodes.BAD_TRANSLATED_SCHEMA,
        )

    @with_async_graph_error_context("get_schema_async")
    async def aget_schema(
        self,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> GraphSchema:
        """
        Async wrapper around `graph_adapter.get_schema(...)`.

        Delegates to GraphTranslator.
        """
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        schema = await self._translator.arun_get_schema(
            op_ctx=ctx,
            framework_ctx={},
        )
        return self._validate_result_type(
            schema,
            GraphSchema,
            "GraphTranslator.arun_get_schema",
            ErrorCodes.BAD_TRANSLATED_SCHEMA,
        )

    @with_graph_error_context("health_sync")
    def health(
        self,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Sync health check wrapper.

        Uses GraphTranslator for consistency with other operations.
        """
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        health_result = self._translator.health(
            op_ctx=ctx,
            framework_ctx={},
        )
        return self._validate_result_type(
            health_result,
            Mapping,
            "GraphTranslator.health",
            ErrorCodes.BAD_HEALTH_RESULT,
        )

    @with_async_graph_error_context("health_async")
    async def ahealth(
        self,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Async health check wrapper.

        Uses GraphTranslator for consistency with other operations.
        """
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        health_result = await self._translator.arun_health(
            op_ctx=ctx,
            framework_ctx={},
        )
        return self._validate_result_type(
            health_result,
            Mapping,
            "GraphTranslator.arun_health",
            ErrorCodes.BAD_HEALTH_RESULT,
        )

    # ------------------------------------------------------------------ #
    # Query (sync + async)
    # ------------------------------------------------------------------ #

    @with_graph_error_context("query_sync")
    def query(
        self,
        query: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        dialect: Optional[str] = None,
        namespace: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> QueryResult:
        """
        Execute a non-streaming graph query (sync).

        Returns the underlying `QueryResult` from the GraphProtocol adapter.
        """
        self._validate_query(query)

        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        raw_query = self._build_raw_query(
            query=query,
            params=params,
            dialect=dialect,
            namespace=namespace,
            timeout_ms=timeout_ms,
            stream=False,
        )
        framework_ctx = self._framework_ctx_for_namespace(namespace)

        result = self._translator.query(
            raw_query,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
            mmr_config=None,
        )
        return self._validate_result_type(
            result,
            QueryResult,
            "GraphTranslator.query",
            ErrorCodes.BAD_TRANSLATED_RESULT,
        )

    @with_async_graph_error_context("query_async")
    async def aquery(
        self,
        query: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        dialect: Optional[str] = None,
        namespace: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> QueryResult:
        """
        Execute a non-streaming graph query (async).

        Returns the underlying `QueryResult`.
        """
        self._validate_query(query)

        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        raw_query = self._build_raw_query(
            query=query,
            params=params,
            dialect=dialect,
            namespace=namespace,
            timeout_ms=timeout_ms,
            stream=False,
        )
        framework_ctx = self._framework_ctx_for_namespace(namespace)

        result = await self._translator.arun_query(
            raw_query,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
            mmr_config=None,
        )
        return self._validate_result_type(
            result,
            QueryResult,
            "GraphTranslator.arun_query",
            ErrorCodes.BAD_TRANSLATED_RESULT,
        )

    # ------------------------------------------------------------------ #
    # Streaming query (sync + async)
    # ------------------------------------------------------------------ #

    @with_graph_error_context("stream_query_sync")
    def stream_query(
        self,
        query: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        dialect: Optional[str] = None,
        namespace: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[QueryChunk]:
        """
        Execute a streaming graph query (sync), yielding `QueryChunk` items.

        Delegates streaming orchestration to GraphTranslator, which uses
        SyncStreamBridge under the hood. This method itself does not use
        any async→sync bridges directly.
        """
        self._validate_query(query)

        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        raw_query = self._build_raw_query(
            query=query,
            params=params,
            dialect=dialect,
            namespace=namespace,
            timeout_ms=timeout_ms,
            stream=True,
        )
        framework_ctx = self._framework_ctx_for_namespace(namespace)

        for chunk in self._translator.query_stream(
            raw_query,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        ):
            yield self._validate_result_type(
                chunk,
                QueryChunk,
                "GraphTranslator.query_stream",
                ErrorCodes.BAD_TRANSLATED_CHUNK,
            )

    @with_async_graph_error_context("stream_query_async")
    async def astream_query(
        self,
        query: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        dialect: Optional[str] = None,
        namespace: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> AsyncIterator[QueryChunk]:
        """
        Execute a streaming graph query (async), yielding `QueryChunk` items.
        """
        self._validate_query(query)

        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        raw_query = self._build_raw_query(
            query=query,
            params=params,
            dialect=dialect,
            namespace=namespace,
            timeout_ms=timeout_ms,
            stream=True,
        )
        framework_ctx = self._framework_ctx_for_namespace(namespace)

        async for chunk in self._translator.arun_query_stream(
            raw_query,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        ):
            yield self._validate_result_type(
                chunk,
                QueryChunk,
                "GraphTranslator.arun_query_stream",
                ErrorCodes.BAD_TRANSLATED_CHUNK,
            )

    # ------------------------------------------------------------------ #
    # Upsert nodes / edges (sync + async)
    # ------------------------------------------------------------------ #

    @with_graph_error_context("upsert_nodes_sync")
    def upsert_nodes(
        self,
        spec: UpsertNodesSpec,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        """
        Sync wrapper for upserting nodes.

        Delegates to GraphTranslator with `raw_nodes` taken from `spec.nodes`,
        and passes the desired namespace via framework_ctx so that the
        translator can build the correct UpsertNodesSpec.
        """
        self._validate_upsert_nodes_spec(spec)

        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        framework_ctx = self._framework_ctx_for_namespace(
            getattr(spec, "namespace", None),
        )

        result = self._translator.upsert_nodes(
            spec.nodes,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )
        return self._validate_result_type(
            result,
            UpsertResult,
            "GraphTranslator.upsert_nodes",
            ErrorCodes.BAD_UPSERT_RESULT,
        )

    @with_async_graph_error_context("upsert_nodes_async")
    async def aupsert_nodes(
        self,
        spec: UpsertNodesSpec,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        """
        Async wrapper for upserting nodes.
        """
        self._validate_upsert_nodes_spec(spec)

        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        framework_ctx = self._framework_ctx_for_namespace(
            getattr(spec, "namespace", None),
        )

        result = await self._translator.arun_upsert_nodes(
            spec.nodes,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )
        return self._validate_result_type(
            result,
            UpsertResult,
            "GraphTranslator.arun_upsert_nodes",
            ErrorCodes.BAD_UPSERT_RESULT,
        )

    @with_graph_error_context("upsert_edges_sync")
    def upsert_edges(
        self,
        spec: UpsertEdgesSpec,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        """
        Sync wrapper for upserting edges.
        """
        self._validate_upsert_edges_spec(spec)

        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        framework_ctx = self._framework_ctx_for_namespace(
            getattr(spec, "namespace", None),
        )

        result = self._translator.upsert_edges(
            spec.edges,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )
        return self._validate_result_type(
            result,
            UpsertResult,
            "GraphTranslator.upsert_edges",
            ErrorCodes.BAD_UPSERT_RESULT,
        )

    @with_async_graph_error_context("upsert_edges_async")
    async def aupsert_edges(
        self,
        spec: UpsertEdgesSpec,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        """
        Async wrapper for upserting edges.
        """
        self._validate_upsert_edges_spec(spec)

        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        framework_ctx = self._framework_ctx_for_namespace(
            getattr(spec, "namespace", None),
        )

        result = await self._translator.arun_upsert_edges(
            spec.edges,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )
        return self._validate_result_type(
            result,
            UpsertResult,
            "GraphTranslator.arun_upsert_edges",
            ErrorCodes.BAD_UPSERT_RESULT,
        )

    # ------------------------------------------------------------------ #
    # Delete nodes / edges (sync + async)
    # ------------------------------------------------------------------ #

    @with_graph_error_context("delete_nodes_sync")
    def delete_nodes(
        self,
        spec: DeleteNodesSpec,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        """
        Sync wrapper for deleting nodes.

        Uses DeleteNodesSpec to derive either an ID list or a filter
        expression for the GraphTranslator.
        """
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        framework_ctx = self._framework_ctx_for_namespace(
            getattr(spec, "namespace", None),
        )

        if spec.filter is not None:
            raw_filter_or_ids: Any = spec.filter
        else:
            raw_filter_or_ids = list(spec.ids or [])

        result = self._translator.delete_nodes(
            raw_filter_or_ids,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )
        return self._validate_result_type(
            result,
            DeleteResult,
            "GraphTranslator.delete_nodes",
            ErrorCodes.BAD_DELETE_RESULT,
        )

    @with_async_graph_error_context("delete_nodes_async")
    async def adelete_nodes(
        self,
        spec: DeleteNodesSpec,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        """
        Async wrapper for deleting nodes.
        """
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        framework_ctx = self._framework_ctx_for_namespace(
            getattr(spec, "namespace", None),
        )

        if spec.filter is not None:
            raw_filter_or_ids: Any = spec.filter
        else:
            raw_filter_or_ids = list(spec.ids or [])

        result = await self._translator.arun_delete_nodes(
            raw_filter_or_ids,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )
        return self._validate_result_type(
            result,
            DeleteResult,
            "GraphTranslator.arun_delete_nodes",
            ErrorCodes.BAD_DELETE_RESULT,
        )

    @with_graph_error_context("delete_edges_sync")
    def delete_edges(
        self,
        spec: DeleteEdgesSpec,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        """
        Sync wrapper for deleting edges.
        """
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        framework_ctx = self._framework_ctx_for_namespace(
            getattr(spec, "namespace", None),
        )

        if spec.filter is not None:
            raw_filter_or_ids: Any = spec.filter
        else:
            raw_filter_or_ids = list(spec.ids or [])

        result = self._translator.delete_edges(
            raw_filter_or_ids,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )
        return self._validate_result_type(
            result,
            DeleteResult,
            "GraphTranslator.delete_edges",
            ErrorCodes.BAD_DELETE_RESULT,
        )

    @with_async_graph_error_context("delete_edges_async")
    async def adelete_edges(
        self,
        spec: DeleteEdgesSpec,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        """
        Async wrapper for deleting edges.
        """
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        framework_ctx = self._framework_ctx_for_namespace(
            getattr(spec, "namespace", None),
        )

        if spec.filter is not None:
            raw_filter_or_ids: Any = spec.filter
        else:
            raw_filter_or_ids = list(spec.ids or [])

        result = await self._translator.arun_delete_edges(
            raw_filter_or_ids,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )
        return self._validate_result_type(
            result,
            DeleteResult,
            "GraphTranslator.arun_delete_edges",
            ErrorCodes.BAD_DELETE_RESULT,
        )

    # ------------------------------------------------------------------ #
    # Bulk vertices (sync + async)
    # ------------------------------------------------------------------ #

    @with_graph_error_context("bulk_vertices_sync")
    def bulk_vertices(
        self,
        spec: BulkVerticesSpec,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BulkVerticesResult:
        """
        Sync wrapper for bulk_vertices.

        Converts `BulkVerticesSpec` into the raw request shape expected by
        GraphTranslator and returns the underlying `BulkVerticesResult`.
        """
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)

        raw_request: Mapping[str, Any] = {
            "namespace": spec.namespace,
            "limit": spec.limit,
            "cursor": spec.cursor,
            "filter": spec.filter,
        }

        framework_ctx = self._framework_ctx_for_namespace(spec.namespace)

        result = self._translator.bulk_vertices(
            raw_request,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )
        return self._validate_result_type(
            result,
            BulkVerticesResult,
            "GraphTranslator.bulk_vertices",
            ErrorCodes.BAD_BULK_VERTICES_RESULT,
        )

    @with_async_graph_error_context("bulk_vertices_async")
    async def abulk_vertices(
        self,
        spec: BulkVerticesSpec,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BulkVerticesResult:
        """
        Async wrapper for bulk_vertices.
        """
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)

        raw_request: Mapping[str, Any] = {
            "namespace": spec.namespace,
            "limit": spec.limit,
            "cursor": spec.cursor,
            "filter": spec.filter,
        }

        framework_ctx = self._framework_ctx_for_namespace(spec.namespace)

        result = await self._translator.arun_bulk_vertices(
            raw_request,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )
        return self._validate_result_type(
            result,
            BulkVerticesResult,
            "GraphTranslator.arun_bulk_vertices",
            ErrorCodes.BAD_BULK_VERTICES_RESULT,
        )

    # ------------------------------------------------------------------ #
    # Batch (sync + async)
    # ------------------------------------------------------------------ #

    @with_graph_error_context("batch_sync")
    def batch(
        self,
        ops: List[BatchOperation],
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BatchResult:
        """
        Sync wrapper for batch operations.

        Translates `BatchOperation` dataclasses into the raw mapping shape
        expected by GraphTranslator and returns the underlying `BatchResult`.
        """
        self._validate_batch_ops(ops)

        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)

        raw_batch_ops: List[Mapping[str, Any]] = [
            {"op": op.op, "args": dict(op.args or {})} for op in ops
        ]

        result = self._translator.batch(
            raw_batch_ops,
            op_ctx=ctx,
            framework_ctx={},
        )
        return self._validate_result_type(
            result,
            BatchResult,
            "GraphTranslator.batch",
            ErrorCodes.BAD_BATCH_RESULT,
        )

    @with_async_graph_error_context("batch_async")
    async def abatch(
        self,
        ops: List[BatchOperation],
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BatchResult:
        """
        Async wrapper for batch operations.
        """
        self._validate_batch_ops(ops)

        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)

        raw_batch_ops: List[Mapping[str, Any]] = [
            {"op": op.op, "args": dict(op.args or {})} for op in ops
        ]

        result = await self._translator.arun_batch(
            raw_batch_ops,
            op_ctx=ctx,
            framework_ctx={},
        )
        return self._validate_result_type(
            result,
            BatchResult,
            "GraphTranslator.arun_batch",
            ErrorCodes.BAD_BATCH_RESULT,
        )


__all__ = [
    "AutoGenGraphClientProtocol",
    "CorpusAutoGenGraphClient",
    "ErrorCodes",
    "with_graph_error_context",
    "with_async_graph_error_context",
    "with_error_context",
    "with_async_error_context",
]
