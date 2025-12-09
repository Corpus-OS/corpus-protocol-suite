# corpus_sdk/mcp/graph_service.py
# SPDX-License-Identifier: Apache-2.0

"""
MCP adapter for Corpus Graph protocol.

This module exposes a Corpus `GraphProtocolV1` implementation as an
MCP-aware graph client, with:

- Async query + streaming query APIs for MCP servers and workflows
- Proper integration with Corpus `GraphProtocolV1` via `GraphTranslator`
- OperationContext propagation derived from MCP context (`from_mcp`)
- Shared error-context decorators for rich observability
- Capabilities / schema / health APIs aligned with other graph adapters

Design philosophy
-----------------
- Protocol-first:
    * All graph operations flow through `GraphTranslator` and the
      underlying `GraphProtocolV1` adapter.
    * This layer is intentionally thin and focuses on:
        - Translating MCP context → `OperationContext`
        - Building raw query shapes for `GraphTranslator`
        - Mapping core protocol results into MCP-friendly response types.

- MCP-specific but minimal:
    * Defines light MCP-facing request/response dataclasses
      (`GraphQueryRequest`, `GraphQueryResult`, etc.).
    * Does **not** implement its own rate limiting, caching, circuit
      breaking, or retry logic. Those belong in:
        - The underlying graph adapter (e.g., `BaseGraphAdapter`), or
        - Higher-level MCP orchestration layers.

Responsibilities
----------------
- Translate MCP context dictionaries into `OperationContext` using
  `from_mcp`, with aligned error codes and `BadRequest` behavior.
- Construct raw query mappings for `GraphTranslator`, including:
    * text, params, dialect, timeout_ms, stream
- Provide MCP-oriented query and streaming APIs that wrap core
  `QueryResult` / `QueryChunk` without reshaping them, beyond
  converting to simple lists of dicts where appropriate.
- Expose:
    * `capabilities` / `acapabilities`
    * `get_schema` / `aget_schema`
    * `health_check` (async)

Non-responsibilities
--------------------
- Backend-specific graph behavior (lives in the `GraphProtocolV1`
  adapter and its implementation).
- Service-level resilience (rate limits, timeouts beyond adapter-
  handled ones, global caching, circuit breaking, retries).
- Transaction orchestration; this module deliberately does **not**
  expose transaction APIs to keep the MCP adapter focused and symmetric
  with the protocol-level graph interface.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from enum import Enum
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

from corpus_sdk.core.context_translation import from_mcp
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
    validate_graph_query,
    validate_graph_result_type,
)
from corpus_sdk.graph.graph_base import (
    BadRequest,
    GraphProtocolV1,
    GraphSchema,
    OperationContext,
    QueryChunk,
    QueryResult,
    BulkVerticesResult,
    BatchResult,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

_FRAMEWORK_NAME = "mcp"


# --------------------------------------------------------------------------- #
# MCP-facing enums and request/response types
# --------------------------------------------------------------------------- #


class GraphStatus(Enum):
    """High-level status for MCP graph responses."""
    SUCCESS = "success"
    FAILURE = "failure"
    DEGRADED = "degraded"
    RATE_LIMITED = "rate_limited"  # reserved for higher layers
    TIMEOUT = "timeout"            # reserved for higher layers
    CACHE_HIT = "cache_hit"        # reserved for higher layers


class QueryType(Enum):
    """Logical query type; typically maps to a dialect."""
    CYPHER = "cypher"
    GREMLIN = "gremlin"
    SPARQL = "sparql"
    TRAVERSAL = "traversal"
    ANALYTICAL = "analytical"
    PATHFINDING = "pathfinding"
    COMMUNITY_DETECTION = "community_detection"


@dataclass
class GraphQueryRequest:
    """MCP-facing graph query request wrapper."""
    query: str
    parameters: Optional[Dict[str, Any]] = None
    query_type: QueryType = QueryType.CYPHER
    timeout: float = 30.0  # seconds
    max_results: Optional[int] = None
    enable_explain: bool = False  # reserved for MCP-specific extensions
    mcp_context: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None


@dataclass
class GraphQueryResult:
    """MCP-facing graph query result wrapper."""
    results: List[Dict[str, Any]]
    execution_time: float
    request_id: str
    query_plan: Optional[Dict[str, Any]] = None
    nodes_processed: Optional[int] = None
    relationships_processed: Optional[int] = None
    status: GraphStatus = GraphStatus.SUCCESS
    error_message: Optional[str] = None


# --------------------------------------------------------------------------- #
# Error codes + decorators (aligned with other adapters)
# --------------------------------------------------------------------------- #


class ErrorCodes:
    """Error code constants for the MCP graph adapter."""

    # Context / translator errors
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


def with_graph_error_context(
    operation: str,
    **static_context: Any,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for sync methods with rich dynamic context extraction.

    Included primarily for parity; the MCP graph client is async-first.
    """
    return create_graph_error_context_decorator(
        framework=_FRAMEWORK_NAME,
        is_async=False,
    )(operation=operation, **static_context)


def with_async_graph_error_context(
    operation: str,
    **static_context: Any,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for async methods with rich dynamic context extraction.
    """
    return create_graph_error_context_decorator(
        framework=_FRAMEWORK_NAME,
        is_async=True,
    )(operation=operation, **static_context)


# Backwards-compatible aliases (mirroring other adapters)
with_error_context = with_graph_error_context
with_async_error_context = with_async_graph_error_context


# --------------------------------------------------------------------------- #
# MCP-specific framework translator (pass-through on results)
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
# Protocol for an MCP-aware graph client
# --------------------------------------------------------------------------- #


class MCPGraphClientProtocol(Protocol):
    """
    Protocol describing the MCP-aware graph client interface.

    Async-first and thin, suitable for wiring into MCP servers.
    """

    async def execute_query(
        self,
        request: GraphQueryRequest,
    ) -> GraphQueryResult:
        ...

    async def stream_query(
        self,
        request: GraphQueryRequest,
    ) -> AsyncIterator[QueryChunk]:
        ...

    def capabilities(self) -> Mapping[str, Any]:
        ...

    async def acapabilities(self) -> Mapping[str, Any]:
        ...

    def get_schema(
        self,
        *,
        mcp_context: Optional[Dict[str, Any]] = None,
    ) -> GraphSchema:
        ...

    async def aget_schema(
        self,
        *,
        mcp_context: Optional[Dict[str, Any]] = None,
    ) -> GraphSchema:
        ...

    async def health_check(
        self,
        *,
        mcp_context: Optional[Dict[str, Any]] = None,
    ) -> Mapping[str, Any]:
        ...


# --------------------------------------------------------------------------- #
# Main MCP graph client
# --------------------------------------------------------------------------- #


class CorpusMCPGraphClient:
    """
    MCP-oriented client wrapper around a Corpus `GraphProtocolV1`.

    This is a thin integration layer that:

    - Translates MCP context dictionaries into a Corpus `OperationContext`
      using `from_mcp`.
    - Uses `GraphTranslator` (with an MCP-specific framework translator) to:
        * Build raw graph query shapes
        * Execute async graph operations
        * Orchestrate streaming with proper error-context handling
    - Attaches rich error context (`attach_context` via decorators) with
      MCP-specific hints when failures occur.
    """

    def __init__(
        self,
        *,
        graph_adapter: GraphProtocolV1,
        framework_version: Optional[str] = None,
        framework_translator: Optional[GraphFrameworkTranslator] = None,
    ) -> None:
        """
        Initialize an MCP-oriented graph client.

        Parameters
        ----------
        graph_adapter:
            Underlying `GraphProtocolV1` implementation.
        framework_version:
            Optional framework version string for observability.
        framework_translator:
            Optional `GraphFrameworkTranslator` implementation. If not
            provided, `MCPGraphFrameworkTranslator` is used by default.
        """
        self._graph: GraphProtocolV1 = graph_adapter
        self._framework_version: Optional[str] = framework_version
        self._framework_translator: Optional[GraphFrameworkTranslator] = (
            framework_translator
        )

    # ------------------------------------------------------------------ #
    # Resource management (context managers)
    # ------------------------------------------------------------------ #

    def __enter__(self) -> CorpusMCPGraphClient:
        """Support sync context manager protocol."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Clean up resources when exiting context."""
        if hasattr(self._graph, "close"):
            self._graph.close()

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
            framework=_FRAMEWORK_NAME,
            translator=framework_translator,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_ctx(
        self,
        *,
        mcp_context: Optional[Dict[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Optional[OperationContext]:
        """
        Build an OperationContext from MCP-style inputs.

        Expected inputs
        ----------------
        - mcp_context: MCP context dict (optional)
        - extra_context: Optional mapping merged into attrs (best effort)

        If all are empty/None, returns None and lets downstream helpers
        construct an "empty" OperationContext as needed.
        """
        extra: Dict[str, Any] = dict(extra_context or {})

        if not mcp_context and not extra:
            return None

        try:
            ctx = from_mcp(
                mcp_context or {},
                framework_version=self._framework_version,
                **extra,
            )
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework=_FRAMEWORK_NAME,
                operation="context_translation",
            )
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
        query_type: QueryType,
        timeout: float,
        stream: bool,
    ) -> Mapping[str, Any]:
        """
        Build a raw query mapping suitable for `GraphTranslator`.

        Expected fields:
            - text (str)
            - dialect (str; derived from `QueryType`)
            - params (optional mapping)
            - timeout_ms (optional int)
            - stream (bool)
        """
        dialect = query_type.value

        raw: Dict[str, Any] = {
            "text": query,
            "params": dict(params or {}),
            "stream": bool(stream),
            "dialect": dialect,
            "timeout_ms": int(timeout * 1000),
        }
        return raw

    def _framework_ctx(
        self,
        *,
        operation: str,
        request_id: Optional[str] = None,
    ) -> Mapping[str, Any]:
        """
        Build a framework_ctx mapping that lets the common translator derive
        framework metadata for observability.
        """
        ctx: Dict[str, Any] = {
            "framework": _FRAMEWORK_NAME,
            "operation": operation,
        }

        if self._framework_version is not None:
            ctx["framework_version"] = self._framework_version

        if request_id is not None:
            ctx["request_id"] = request_id

        return ctx

    def _process_query_results(
        self,
        result: Any,
        max_results: Optional[int],
    ) -> List[Dict[str, Any]]:
        """
        Process and limit query results into a list of dicts for MCP.

        Tries to be forgiving about core result shapes:
        - If `result` has `.rows`, use that.
        - If `result` is iterable, normalize each row.
        """
        # Common pattern: Graph adapters often expose `rows` on QueryResult.
        rows = getattr(result, "rows", result)

        if rows is None:
            return []

        processed: List[Dict[str, Any]] = []

        for row in rows:
            if isinstance(row, dict):
                processed.append(row)
            elif hasattr(row, "_asdict"):  # NamedTuple-style
                processed.append(row._asdict())
            elif hasattr(row, "__dict__"):
                processed.append(dict(row.__dict__))
            else:
                processed.append({"result": row})

        if max_results is not None and len(processed) > max_results:
            processed = processed[:max_results]

        return processed

    # ------------------------------------------------------------------ #
    # Capabilities / schema / health
    # ------------------------------------------------------------------ #

    @with_graph_error_context("capabilities_sync")
    def capabilities(self) -> Mapping[str, Any]:
        """
        Sync wrapper around capabilities, delegating to GraphTranslator.
        """
        caps = self._translator.capabilities()
        return graph_capabilities_to_dict(caps)

    @with_async_graph_error_context("capabilities_async")
    async def acapabilities(self) -> Mapping[str, Any]:
        """
        Async capabilities accessor, via GraphTranslator.
        """
        caps = await self._translator.arun_capabilities()
        return graph_capabilities_to_dict(caps)

    @with_graph_error_context("get_schema_sync")
    def get_schema(
        self,
        *,
        mcp_context: Optional[Dict[str, Any]] = None,
    ) -> GraphSchema:
        """
        Sync schema introspection, via GraphTranslator.
        """
        op_ctx = self._build_ctx(mcp_context=mcp_context)
        schema = self._translator.get_schema(
            op_ctx=op_ctx,
            framework_ctx=self._framework_ctx(operation="get_schema"),
        )
        return validate_graph_result_type(
            schema,
            expected_type=GraphSchema,
            operation="GraphTranslator.get_schema",
            error_code=ErrorCodes.BAD_TRANSLATED_SCHEMA,
        )

    @with_async_graph_error_context("get_schema_async")
    async def aget_schema(
        self,
        *,
        mcp_context: Optional[Dict[str, Any]] = None,
    ) -> GraphSchema:
        """
        Async schema introspection, via GraphTranslator.
        """
        op_ctx = self._build_ctx(mcp_context=mcp_context)
        schema = await self._translator.arun_get_schema(
            op_ctx=op_ctx,
            framework_ctx=self._framework_ctx(operation="get_schema"),
        )
        return validate_graph_result_type(
            schema,
            expected_type=GraphSchema,
            operation="GraphTranslator.arun_get_schema",
            error_code=ErrorCodes.BAD_TRANSLATED_SCHEMA,
        )

    @with_async_graph_error_context("health_async")
    async def health_check(
        self,
        *,
        mcp_context: Optional[Dict[str, Any]] = None,
    ) -> Mapping[str, Any]:
        """
        Async health check wrapper, delegating to GraphTranslator.

        Returns the underlying adapter's health mapping, wrapped with
        MCP framework context via the error decorator.
        """
        op_ctx = self._build_ctx(mcp_context=mcp_context)
        health = await self._translator.arun_health(
            op_ctx=op_ctx,
            framework_ctx=self._framework_ctx(operation="health"),
        )
        return validate_graph_result_type(
            health,
            expected_type=Mapping,
            operation="GraphTranslator.arun_health",
            error_code=ErrorCodes.BAD_HEALTH_RESULT,
        )

    # ------------------------------------------------------------------ #
    # Query (async) – core MCP API
    # ------------------------------------------------------------------ #

    @with_async_graph_error_context("execute_query")
    async def execute_query(
        self,
        request: GraphQueryRequest,
    ) -> GraphQueryResult:
        """
        Execute a non-streaming graph query (async) for MCP.

        Returns an MCP-facing `GraphQueryResult` that wraps the underlying
        `QueryResult` from the GraphProtocol adapter.
        """
        request_id = request.request_id or f"graph_{uuid.uuid4().hex[:8]}"
        start_time = time.time()

        validate_graph_query(request.query)

        # Build contexts for translator
        op_ctx = self._build_ctx(
            mcp_context=request.mcp_context,
            extra_context={"request_id": request_id},
        )
        raw_query = self._build_raw_query(
            query=request.query,
            params=request.parameters,
            query_type=request.query_type,
            timeout=request.timeout,
            stream=False,
        )
        framework_ctx = self._framework_ctx(
            operation="query",
            request_id=request_id,
        )

        result = await self._translator.arun_query(
            raw_query,
            op_ctx=op_ctx,
            framework_ctx=framework_ctx,
            mmr_config=None,
        )
        query_result = validate_graph_result_type(
            result,
            expected_type=QueryResult,
            operation="GraphTranslator.arun_query",
            error_code=ErrorCodes.BAD_TRANSLATED_RESULT,
        )

        processed = self._process_query_results(
            query_result,
            request.max_results,
        )
        execution_time = time.time() - start_time

        # Optional: query plan (MCP nuance) – only if adapter exposes it.
        query_plan: Optional[Dict[str, Any]] = None
        if request.enable_explain and hasattr(self._graph, "explain_query"):
            try:
                # Best-effort, adapter-specific; not part of GraphTranslator.
                query_plan = await self._graph.explain_query(  # type: ignore[attr-defined]
                    raw_query,
                    op_ctx=op_ctx,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to generate query plan for request %s: %s", request_id, exc)

        return GraphQueryResult(
            results=processed,
            execution_time=execution_time,
            request_id=request_id,
            query_plan=query_plan,
            nodes_processed=getattr(query_result, "nodes_processed", None),
            relationships_processed=getattr(
                query_result,
                "relationships_processed",
                None,
            ),
            status=GraphStatus.SUCCESS,
            error_message=None,
        )

    # ------------------------------------------------------------------ #
    # Streaming query (async) – core MCP API
    # ------------------------------------------------------------------ #

    @with_async_graph_error_context("stream_query_async")
    async def stream_query(
        self,
        request: GraphQueryRequest,
    ) -> AsyncIterator[QueryChunk]:
        """
        Execute a streaming graph query (async), yielding `QueryChunk` items.

        Delegates streaming orchestration to `GraphTranslator`.
        """
        validate_graph_query(request.query)

        op_ctx = self._build_ctx(
            mcp_context=request.mcp_context,
            extra_context={"request_id": request.request_id},
        )
        raw_query = self._build_raw_query(
            query=request.query,
            params=request.parameters,
            query_type=request.query_type,
            timeout=request.timeout,
            stream=True,
        )
        framework_ctx = self._framework_ctx(
            operation="stream_query",
            request_id=request.request_id,
        )

        async for chunk in self._translator.arun_query_stream(
            raw_query,
            op_ctx=op_ctx,
            framework_ctx=framework_ctx,
        ):
            yield validate_graph_result_type(
                chunk,
                expected_type=QueryChunk,
                operation="GraphTranslator.arun_query_stream",
                error_code=ErrorCodes.BAD_TRANSLATED_CHUNK,
            )


__all__ = [
    "CorpusMCPGraphClient",
    "MCPGraphClientProtocol",
    "MCPGraphFrameworkTranslator",
    "GraphQueryRequest",
    "GraphQueryResult",
    "QueryType",
    "GraphStatus",
    "ErrorCodes",
    "with_graph_error_context",
    "with_async_graph_error_context",
    "with_error_context",
    "with_async_error_context",
]