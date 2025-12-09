# corpus_sdk/graph/framework_adapters/mcp.py
# SPDX-License-Identifier: Apache-2.0

"""
MCP adapter for Corpus Graph protocol.

This module exposes a Corpus `GraphProtocolV1` implementation as an
MCP (Model Context Protocol)–friendly graph client, with:

- Async query APIs
- Async streaming query APIs
- Full read/write coverage:
    * Query / streaming query
    * Upsert nodes / edges
    * Delete nodes / edges
    * Bulk vertices
    * Batch operations
- Proper integration with the Corpus GraphProtocolV1 stack via GraphTranslator
- OperationContext propagation derived from MCP context (`from_mcp`)
- Error-context enrichment for observability and debugging
- Centralized orchestration and streaming via GraphTranslator

Design philosophy
-----------------
- Protocol-first: MCP is a thin skin over the Corpus graph adapter.
- All heavy lifting (deadlines, rate limits, caching, retries, etc.) lives in
  the underlying `BaseGraphAdapter` / `GraphProtocolV1` implementation.
- This layer focuses on:
    * Translating MCP context → `OperationContext`
    * Building raw query / mutation shapes for `GraphTranslator`
    * Delegating all async and streaming orchestration to `GraphTranslator`
    * Attaching rich, MCP-flavored error context

Responsibilities
----------------
- Provide an MCP-oriented client for graph operations that:
    * Uses the shared `GraphTranslator` abstraction
    * Preserves protocol-level types (`QueryResult`, `QueryChunk`, etc.)
    * Exposes async-only APIs consistent with MCP’s async execution model
- Keep all graph operations going through `GraphTranslator` so that
  streaming and error-context logic are centralized
- Surface a clean, minimal API (`MCPGraphClientProtocol`) that can be
  wrapped by higher-level MCP “services” or servers

Non-responsibilities
--------------------
- Service-level behaviors (rate limiting, circuit breaking, caching,
  retries, request quotas, etc.) – these belong in a separate service
  layer that wraps this client, not inside the adapter.
- MCP server wiring (tool registration, MCPServer lifecycle) – that
  should live in a separate module (e.g., `graph_server.py`).
- Backend-specific graph behavior – lives in graph adapters that
  implement `GraphProtocolV1`.
- MMR/diversification details – handled inside `GraphTranslator`.
"""

from __future__ import annotations

import logging
from functools import cached_property
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    TypeVar,
)

from corpus_sdk.core.context_translation import from_mcp as core_ctx_from_mcp
from corpus_sdk.core.error_context import attach_context
from corpus_sdk.graph.framework_adapters.common.graph_translation import (
    DefaultGraphFrameworkTranslator,
    GraphFrameworkTranslator,
    GraphTranslator,
    create_graph_translator,
)
from corpus_sdk.graph.framework_adapters.common.framework_utils import (
    create_graph_error_context_decorator,
    graph_capabilities_to_dict,
    validate_batch_operations,
    validate_graph_query,
    validate_graph_result_type,
    validate_upsert_nodes_spec,
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

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# Error codes (aligned with other framework adapters)
# --------------------------------------------------------------------------- #


class ErrorCodes:
    """
    Framework-local error code namespace for the MCP graph adapter.

    These codes are used for adapter/translator-level issues and complement
    any higher-level MCP service error taxonomy.
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


# --------------------------------------------------------------------------- #
# Error-context decorators (async-only, framework-tagged)
# --------------------------------------------------------------------------- #


def with_async_graph_error_context(
    operation: str,
    **static_context: Any,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for async methods with rich dynamic context extraction.

    Thin wrapper over the shared `create_graph_error_context_decorator`
    for the MCP framework.
    """
    return create_graph_error_context_decorator(
        framework="mcp",
        is_async=True,
    )(operation=operation, **static_context)


# Backwards-compatible async alias (matches other adapters’ naming)
with_async_error_context = with_async_graph_error_context


# --------------------------------------------------------------------------- #
# Public protocol (what MCP wrappers should type against)
# --------------------------------------------------------------------------- #


class MCPGraphClientProtocol(Protocol):
    """
    Protocol describing the MCP-aware graph client interface.

    This allows MCP-facing layers to type against the graph client without
    depending on the concrete `CorpusMCPGraphClient` implementation.
    """

    # Capabilities / schema / health -------------------------------------

    async def acapabilities(self) -> Mapping[str, Any]:
        ...

    async def aget_schema(
        self,
        *,
        mcp_context: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> GraphSchema:
        ...

    async def ahealth(
        self,
        *,
        mcp_context: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        ...

    # Query / streaming ---------------------------------------------------

    async def aquery(
        self,
        query: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        dialect: Optional[str] = None,
        namespace: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        mcp_context: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> QueryResult:
        ...

    async def astream_query(
        self,
        query: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        dialect: Optional[str] = None,
        namespace: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        mcp_context: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> AsyncIterator[QueryChunk]:
        ...

    # Upsert --------------------------------------------------------------

    async def aupsert_nodes(
        self,
        spec: UpsertNodesSpec,
        *,
        mcp_context: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        ...

    async def aupsert_edges(
        self,
        spec: UpsertEdgesSpec,
        *,
        mcp_context: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        ...

    # Delete --------------------------------------------------------------

    async def adelete_nodes(
        self,
        spec: DeleteNodesSpec,
        *,
        mcp_context: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        ...

    async def adelete_edges(
        self,
        spec: DeleteEdgesSpec,
        *,
        mcp_context: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        ...

    # Bulk / batch --------------------------------------------------------

    async def abulk_vertices(
        self,
        spec: BulkVerticesSpec,
        *,
        mcp_context: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BulkVerticesResult:
        ...

    async def abatch(
        self,
        ops: List[BatchOperation],
        *,
        mcp_context: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BatchResult:
        ...


# --------------------------------------------------------------------------- #
# MCP-specific GraphFrameworkTranslator (no result reshaping)
# --------------------------------------------------------------------------- #


class MCPGraphFrameworkTranslator(DefaultGraphFrameworkTranslator):
    """
    MCP-specific GraphFrameworkTranslator.

    Reuses the default implementation for spec construction and filters,
    but deliberately returns core protocol types unchanged:

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
        framework_ctx: Optional[Mapping[str, Any]] = None,
    ) -> QueryResult:
        return result

    def translate_query_chunk(
        self,
        chunk: QueryChunk,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Mapping[str, Any]] = None,
    ) -> QueryChunk:
        return chunk

    def translate_bulk_vertices_result(
        self,
        result: BulkVerticesResult,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Mapping[str, Any]] = None,
    ) -> BulkVerticesResult:
        return result

    def translate_batch_result(
        self,
        result: BatchResult,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Mapping[str, Any]] = None,
    ) -> BatchResult:
        return result

    def translate_schema(
        self,
        schema: GraphSchema,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Mapping[str, Any]] = None,
    ) -> GraphSchema:
        return schema


# --------------------------------------------------------------------------- #
# Core MCP-oriented graph client (async-only)
# --------------------------------------------------------------------------- #


class CorpusMCPGraphClient:
    """
    MCP-oriented graph client wrapper around a Corpus `GraphProtocolV1`.

    This is a thin integration layer that:

    - Translates MCP context into a Corpus `OperationContext` using `from_mcp`.
    - Uses `GraphTranslator` (with an MCP-specific framework translator) to:
        * Build Graph*Spec objects from simple inputs
        * Execute async graph operations
        * Orchestrate streaming with proper cancellation and error handling
    - Delegates all async orchestration and streaming glue to GraphTranslator.
    - Attaches rich error context (`attach_context`) on this layer with
      MCP-specific hints when failures occur.

    Higher-level MCP “services” (rate limiting, circuit breaking, caching,
    metrics, etc.) are expected to wrap this client rather than be baked
    into it.
    """

    def __init__(
        self,
        *,
        graph_adapter: GraphProtocolV1,
        default_dialect: Optional[str] = None,
        default_namespace: Optional[str] = None,
        default_timeout_ms: Optional[int] = None,
        framework_version: Optional[str] = None,
        framework_translator: Optional[GraphFrameworkTranslator] = None,
    ) -> None:
        """
        Initialize an MCP-oriented graph client.

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
        framework_translator:
            Optional `GraphFrameworkTranslator` implementation. If not provided,
            `MCPGraphFrameworkTranslator` is used by default.
        """
        self._graph: GraphProtocolV1 = graph_adapter
        self._default_dialect: Optional[str] = default_dialect
        self._default_namespace: Optional[str] = default_namespace
        self._default_timeout_ms: Optional[int] = default_timeout_ms
        self._framework_version: Optional[str] = framework_version
        self._framework_translator: Optional[GraphFrameworkTranslator] = framework_translator

    # ------------------------------------------------------------------ #
    # Resource Management (Async Context Managers)
    # ------------------------------------------------------------------ #

    async def __aenter__(self) -> CorpusMCPGraphClient:
        """Support async context manager protocol."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Clean up resources when exiting async context."""
        if hasattr(self._graph, "aclose"):
            await self._graph.aclose()

    # ------------------------------------------------------------------ #
    # Translator (lazy, cached)
    # ------------------------------------------------------------------ #

    @cached_property
    def _translator(self) -> GraphTranslator:
        """
        Lazily construct and cache the `GraphTranslator`.

        Uses `cached_property` for thread safety and performance.
        """
        framework_translator: GraphFrameworkTranslator = (
            self._framework_translator or MCPGraphFrameworkTranslator()
        )
        return create_graph_translator(
            adapter=self._graph,
            framework="mcp",
            translator=framework_translator,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_ctx(
        self,
        *,
        mcp_context: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Optional[OperationContext]:
        """
        Build an OperationContext from MCP-style inputs.

        Expected inputs
        ----------------
        - mcp_context: MCP context mapping (optional)
        - extra_context: Optional mapping merged into attrs (best effort)

        If both are None/empty, returns None and lets downstream helpers
        construct an "empty" OperationContext as needed.
        """
        extra: Dict[str, Any] = dict(extra_context or {})

        if not mcp_context and not extra:
            return None

        try:
            ctx = core_ctx_from_mcp(
                dict(mcp_context or {}),
                framework_version=self._framework_version,
                **extra,
            )
        except Exception as exc:
            attach_context(
                exc,
                framework="mcp",
                operation="context_translation",
            )
            # Surface a consistent BadRequest with symbolic error code.
            raise BadRequest(
                "Failed to build OperationContext from MCP inputs",
                code=ErrorCodes.BAD_OPERATION_CONTEXT,
            ) from exc

        if not isinstance(ctx, OperationContext):
            raise BadRequest(
                f"from_mcp produced unsupported context type: {type(ctx).__name__}",
                code=ErrorCodes.BAD_OPERATION_CONTEXT,
            )

        return ctx

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

        Expected fields:
            - text (str)
            - dialect (optional)
            - params (optional mapping)
            - namespace (optional)
            - timeout_ms (optional int)
            - stream (bool)
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

    def _framework_ctx(
        self,
        *,
        operation: str,
        namespace: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Mapping[str, Any]:
        """
        Build a framework_ctx mapping that lets the common translator derive
        a preferred namespace and capture framework metadata for observability.
        """
        ctx: Dict[str, Any] = {
            "framework": "mcp",
            "operation": operation,
        }

        if self._framework_version is not None:
            ctx["framework_version"] = self._framework_version

        effective_namespace = namespace or self._default_namespace
        if effective_namespace is not None:
            ctx["namespace"] = effective_namespace

        if request_id is not None:
            ctx["request_id"] = request_id

        return ctx

    def _validate_upsert_edges_spec(self, spec: UpsertEdgesSpec) -> None:
        """
        Basic structural validation for UpsertEdgesSpec.edges to provide
        clearer errors before reaching the adapter / translator.
        """
        if spec.edges is None:
            raise BadRequest("UpsertEdgesSpec.edges must not be None")

        try:
            edges_list = list(spec.edges)
        except TypeError as exc:
            raise BadRequest(
                "UpsertEdgesSpec.edges must be an iterable of edges",
            ) from exc

        if not edges_list:
            raise BadRequest("UpsertEdgesSpec must contain at least one edge")

        # Normalize to list to avoid one-shot iterables.
        spec.edges = edges_list  # type: ignore[assignment]

    # ------------------------------------------------------------------ #
    # Capabilities / schema / health (async-only)
    # ------------------------------------------------------------------ #

    @with_async_graph_error_context("capabilities_async")
    async def acapabilities(self) -> Mapping[str, Any]:
        """
        Async capabilities accessor.

        We delegate to GraphTranslator for consistency, then normalize to a
        simple dict for MCP-facing consumption.
        """
        caps = await self._translator.arun_capabilities()
        return graph_capabilities_to_dict(caps)

    @with_async_graph_error_context("get_schema_async")
    async def aget_schema(
        self,
        *,
        mcp_context: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> GraphSchema:
        """
        Async schema introspection, via GraphTranslator.
        """
        ctx = self._build_ctx(
            mcp_context=mcp_context,
            extra_context=extra_context,
        )
        schema = await self._translator.arun_get_schema(
            op_ctx=ctx,
            framework_ctx=self._framework_ctx(operation="get_schema"),
        )
        return validate_graph_result_type(
            schema,
            expected_type=GraphSchema,
            operation="GraphTranslator.arun_get_schema",
            error_code=ErrorCodes.BAD_TRANSLATED_SCHEMA,
        )

    @with_async_graph_error_context("health_async")
    async def ahealth(
        self,
        *,
        mcp_context: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        """
        Async health check wrapper, delegating orchestration to GraphTranslator.
        """
        ctx = self._build_ctx(
            mcp_context=mcp_context,
            extra_context=extra_context,
        )
        health_result = await self._translator.arun_health(
            op_ctx=ctx,
            framework_ctx=self._framework_ctx(operation="health"),
        )
        return validate_graph_result_type(
            health_result,
            expected_type=Mapping,
            operation="GraphTranslator.arun_health",
            error_code=ErrorCodes.BAD_HEALTH_RESULT,
        )

    # ------------------------------------------------------------------ #
    # Query (async-only)
    # ------------------------------------------------------------------ #

    @with_async_graph_error_context("query_async")
    async def aquery(
        self,
        query: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        dialect: Optional[str] = None,
        namespace: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        mcp_context: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> QueryResult:
        """
        Execute a non-streaming graph query (async).

        Returns the underlying `QueryResult`.
        """
        validate_graph_query(query)

        ctx = self._build_ctx(
            mcp_context=mcp_context,
            extra_context=extra_context,
        )
        raw_query = self._build_raw_query(
            query=query,
            params=params,
            dialect=dialect,
            namespace=namespace,
            timeout_ms=timeout_ms,
            stream=False,
        )
        framework_ctx = self._framework_ctx(
            operation="query",
            namespace=namespace,
        )

        result = await self._translator.arun_query(
            raw_query,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
            mmr_config=None,
        )
        return validate_graph_result_type(
            result,
            expected_type=QueryResult,
            operation="GraphTranslator.arun_query",
            error_code=ErrorCodes.BAD_TRANSLATED_RESULT,
        )

    # ------------------------------------------------------------------ #
    # Streaming query (async-only)
    # ------------------------------------------------------------------ #

    @with_async_graph_error_context("stream_query_async")
    async def astream_query(
        self,
        query: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        dialect: Optional[str] = None,
        namespace: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        mcp_context: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> AsyncIterator[QueryChunk]:
        """
        Execute a streaming graph query (async), yielding `QueryChunk` items.
        """
        validate_graph_query(query)

        ctx = self._build_ctx(
            mcp_context=mcp_context,
            extra_context=extra_context,
        )
        raw_query = self._build_raw_query(
            query=query,
            params=params,
            dialect=dialect,
            namespace=namespace,
            timeout_ms=timeout_ms,
            stream=True,
        )
        framework_ctx = self._framework_ctx(
            operation="stream_query",
            namespace=namespace,
        )

        async for chunk in self._translator.arun_query_stream(
            raw_query,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        ):
            yield validate_graph_result_type(
                chunk,
                expected_type=QueryChunk,
                operation="GraphTranslator.arun_query_stream",
                error_code=ErrorCodes.BAD_TRANSLATED_CHUNK,
            )

    # ------------------------------------------------------------------ #
    # Upsert nodes / edges (async-only)
    # ------------------------------------------------------------------ #

    @with_async_graph_error_context("upsert_nodes_async")
    async def aupsert_nodes(
        self,
        spec: UpsertNodesSpec,
        *,
        mcp_context: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        """
        Async wrapper for upserting nodes via GraphTranslator.
        """
        validate_upsert_nodes_spec(spec)

        ctx = self._build_ctx(
            mcp_context=mcp_context,
            extra_context=extra_context,
        )
        framework_ctx = self._framework_ctx(
            operation="upsert_nodes",
            namespace=getattr(spec, "namespace", None),
        )

        result = await self._translator.arun_upsert_nodes(
            spec.nodes,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )
        return validate_graph_result_type(
            result,
            expected_type=UpsertResult,
            operation="GraphTranslator.arun_upsert_nodes",
            error_code=ErrorCodes.BAD_UPSERT_RESULT,
        )

    @with_async_graph_error_context("upsert_edges_async")
    async def aupsert_edges(
        self,
        spec: UpsertEdgesSpec,
        *,
        mcp_context: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        """
        Async wrapper for upserting edges via GraphTranslator.
        """
        self._validate_upsert_edges_spec(spec)

        ctx = self._build_ctx(
            mcp_context=mcp_context,
            extra_context=extra_context,
        )
        framework_ctx = self._framework_ctx(
            operation="upsert_edges",
            namespace=getattr(spec, "namespace", None),
        )

        result = await self._translator.arun_upsert_edges(
            spec.edges,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )
        return validate_graph_result_type(
            result,
            expected_type=UpsertResult,
            operation="GraphTranslator.arun_upsert_edges",
            error_code=ErrorCodes.BAD_UPSERT_RESULT,
        )

    # ------------------------------------------------------------------ #
    # Delete nodes / edges (async-only)
    # ------------------------------------------------------------------ #

    @with_async_graph_error_context("delete_nodes_async")
    async def adelete_nodes(
        self,
        spec: DeleteNodesSpec,
        *,
        mcp_context: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        """
        Async wrapper for deleting nodes via GraphTranslator.
        """
        ctx = self._build_ctx(
            mcp_context=mcp_context,
            extra_context=extra_context,
        )
        framework_ctx = self._framework_ctx(
            operation="delete_nodes",
            namespace=getattr(spec, "namespace", None),
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
        return validate_graph_result_type(
            result,
            expected_type=DeleteResult,
            operation="GraphTranslator.arun_delete_nodes",
            error_code=ErrorCodes.BAD_DELETE_RESULT,
        )

    @with_async_graph_error_context("delete_edges_async")
    async def adelete_edges(
        self,
        spec: DeleteEdgesSpec,
        *,
        mcp_context: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        """
        Async wrapper for deleting edges via GraphTranslator.
        """
        ctx = self._build_ctx(
            mcp_context=mcp_context,
            extra_context=extra_context,
        )
        framework_ctx = self._framework_ctx(
            operation="delete_edges",
            namespace=getattr(spec, "namespace", None),
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
        return validate_graph_result_type(
            result,
            expected_type=DeleteResult,
            operation="GraphTranslator.arun_delete_edges",
            error_code=ErrorCodes.BAD_DELETE_RESULT,
        )

    # ------------------------------------------------------------------ #
    # Bulk vertices (async-only)
    # ------------------------------------------------------------------ #

    @with_async_graph_error_context("bulk_vertices_async")
    async def abulk_vertices(
        self,
        spec: BulkVerticesSpec,
        *,
        mcp_context: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BulkVerticesResult:
        """
        Async wrapper for bulk_vertices via GraphTranslator.
        """
        ctx = self._build_ctx(
            mcp_context=mcp_context,
            extra_context=extra_context,
        )

        raw_request: Mapping[str, Any] = {
            "namespace": spec.namespace,
            "limit": spec.limit,
            "cursor": spec.cursor,
            "filter": spec.filter,
        }

        framework_ctx = self._framework_ctx(
            operation="bulk_vertices",
            namespace=spec.namespace,
        )

        result = await self._translator.arun_bulk_vertices(
            raw_request,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )
        return validate_graph_result_type(
            result,
            expected_type=BulkVerticesResult,
            operation="GraphTranslator.arun_bulk_vertices",
            error_code=ErrorCodes.BAD_BULK_VERTICES_RESULT,
        )

    # ------------------------------------------------------------------ #
    # Batch (async-only)
    # ------------------------------------------------------------------ #

    @with_async_graph_error_context("batch_async")
    async def abatch(
        self,
        ops: List[BatchOperation],
        *,
        mcp_context: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BatchResult:
        """
        Async wrapper for batch operations via GraphTranslator.

        Translates `BatchOperation` dataclasses into the raw mapping shape
        expected by GraphTranslator and returns the underlying `BatchResult`.
        """
        validate_batch_operations(self._graph, ops)

        ctx = self._build_ctx(
            mcp_context=mcp_context,
            extra_context=extra_context,
        )
        raw_batch_ops: List[Mapping[str, Any]] = [
            {"op": op.op, "args": dict(op.args or {})} for op in ops
        ]
        framework_ctx = self._framework_ctx(operation="batch")

        result = await self._translator.arun_batch(
            raw_batch_ops,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )
        return validate_graph_result_type(
            result,
            expected_type=BatchResult,
            operation="GraphTranslator.arun_batch",
            error_code=ErrorCodes.BAD_BATCH_RESULT,
        )


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def create_mcp_graph_client(
    graph_adapter: GraphProtocolV1,
    **kwargs: Any,
) -> CorpusMCPGraphClient:
    """
    Convenience factory for constructing an MCP-oriented graph client.

    Example
    -------
    ```python
    from corpus_sdk.graph.framework_adapters.mcp import create_mcp_graph_client

    client = create_mcp_graph_client(
        graph_adapter=my_graph_adapter,
        default_namespace="tenant_123",
        framework_version="mcp-graph-0.1.0",
    )
    ```
    """
    return CorpusMCPGraphClient(
        graph_adapter=graph_adapter,
        **kwargs,
    )


__all__ = [
    "MCPGraphClientProtocol",
    "CorpusMCPGraphClient",
    "MCPGraphFrameworkTranslator",
    "create_mcp_graph_client",
    "ErrorCodes",
    "with_async_graph_error_context",
    "with_async_error_context",
]