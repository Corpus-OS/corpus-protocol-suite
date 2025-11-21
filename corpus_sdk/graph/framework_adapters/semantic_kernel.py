# corpus_sdk/graph/framework_adapters/semantic_kernel.py
# SPDX-License-Identifier: Apache-2.0

"""
Semantic Kernel adapter for Corpus Graph protocol.

This module exposes a Corpus `GraphProtocolV1` implementation as a
Semantic Kernel–friendly client, with:

- Sync + async query APIs
- Sync + async streaming query APIs
- Proper integration with Corpus GraphProtocolV1
- OperationContext propagation derived from Semantic Kernel context + settings
- Error-context enrichment for observability and debugging
- Orchestration, translation, and async→sync bridging via GraphTranslator

Design philosophy
-----------------
- Protocol-first: Semantic Kernel is a thin skin over the Corpus graph adapter.
- All heavy lifting (deadlines, breakers, rate limits, caching, etc.) lives in
  the underlying `BaseGraphAdapter` / `GraphProtocolV1` implementation.
- This layer focuses on:
    * Translating SK context/settings → OperationContext
    * Building raw query / mutation shapes for GraphTranslator
    * Delegating all sync/async and streaming orchestration to GraphTranslator

Responsibilities
----------------
- Provide a Semantic Kernel–oriented client for graph operations
- Keep all graph operations going through `GraphTranslator` so that
  async→sync bridging, streaming, and error-context logic are centralized
- Preserve protocol-level types (`QueryResult`, `QueryChunk`, etc.) for callers

Non-responsibilities
--------------------
- Backend-specific graph behavior (lives in graph adapters)
- Semantic Kernel orchestration / plugin wiring
- MMR and diversification details (handled inside GraphTranslator)
"""

from __future__ import annotations

import logging
from functools import cached_property, wraps
from typing import (
    Any,
    AsyncIterator,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Dict,
    TypeVar,
    Callable,
    cast,
)

from corpus_sdk.core.context_translation import (
    from_semantic_kernel as core_ctx_from_semantic_kernel,
)
from corpus_sdk.core.error_context import attach_context
from corpus_sdk.graph.framework_adapters.common.graph_translation import (
    DefaultGraphFrameworkTranslator,
    GraphTranslator,
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

# Type variables for decorators
T = TypeVar("T")
R = TypeVar("R")

# Error code constants
class ErrorCodes:
    BAD_OPERATION_CONTEXT = "BAD_OPERATION_CONTEXT"
    BAD_TRANSLATED_SCHEMA = "BAD_TRANSLATED_SCHEMA"
    BAD_HEALTH_RESULT = "BAD_HEALTH_RESULT"
    BAD_TRANSLATED_RESULT = "BAD_TRANSLATED_RESULT"
    BAD_TRANSLATED_CHUNK = "BAD_TRANSLATED_CHUNK"
    BAD_UPSERT_RESULT = "BAD_UPSERT_RESULT"
    BAD_DELETE_RESULT = "BAD_DELETE_RESULT"
    BAD_BULK_VERTICES_RESULT = "BAD_BULK_VERTICES_RESULT"
    BAD_BATCH_RESULT = "BAD_BATCH_RESULT"


def with_error_context(
    operation: str,
    **context_kwargs: Any,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to automatically attach error context to exceptions.
    
    Args:
        operation: The operation name for error context
        **context_kwargs: Additional context to attach to errors
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                # Extract additional context from function arguments if needed
                enhanced_context = context_kwargs.copy()
                
                # For query operations, try to extract query info
                if operation in ("query_sync", "query_async", "stream_query_sync", "stream_query_async"):
                    if len(args) > 1 and isinstance(args[1], str):
                        enhanced_context["query"] = args[1]
                
                attach_context(
                    exc,
                    framework="semantic_kernel",
                    operation=operation,
                    **enhanced_context,
                )
                raise
        return wrapper
    return decorator


def with_async_error_context(
    operation: str,
    **context_kwargs: Any,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to automatically attach error context to exceptions in async functions.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return await func(*args, **kwargs)
            except Exception as exc:
                enhanced_context = context_kwargs.copy()
                
                if operation in ("query_sync", "query_async", "stream_query_sync", "stream_query_async"):
                    if len(args) > 1 and isinstance(args[1], str):
                        enhanced_context["query"] = args[1]
                
                attach_context(
                    exc,
                    framework="semantic_kernel",
                    operation=operation,
                    **enhanced_context,
                )
                raise
        return wrapper
    return decorator


class SemanticKernelGraphClientProtocol(Protocol):
    """
    Protocol describing the Semantic Kernel–aware graph client interface.

    This allows callers to type against the client without depending on
    the concrete `CorpusSemanticKernelGraphClient` implementation.
    """

    # Capabilities / schema / health -------------------------------------

    def capabilities(self) -> Mapping[str, Any]:
        ...

    async def acapabilities(self) -> Mapping[str, Any]:
        ...

    def get_schema(
        self,
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> GraphSchema:
        ...

    async def aget_schema(
        self,
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> GraphSchema:
        ...

    def health(
        self,
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        ...

    async def ahealth(
        self,
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        ...

    # Query / streaming ---------------------------------------------------

    def query(
        self,
        query: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        dialect: Optional[str] = None,
        namespace: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
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
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
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
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
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
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> AsyncIterator[QueryChunk]:
        ...

    # Upsert --------------------------------------------------------------

    def upsert_nodes(
        self,
        spec: UpsertNodesSpec,
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        ...

    async def aupsert_nodes(
        self,
        spec: UpsertNodesSpec,
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        ...

    def upsert_edges(
        self,
        spec: UpsertEdgesSpec,
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        ...

    async def aupsert_edges(
        self,
        spec: UpsertEdgesSpec,
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        ...

    # Delete --------------------------------------------------------------

    def delete_nodes(
        self,
        spec: DeleteNodesSpec,
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        ...

    async def adelete_nodes(
        self,
        spec: DeleteNodesSpec,
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        ...

    def delete_edges(
        self,
        spec: DeleteEdgesSpec,
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        ...

    async def adelete_edges(
        self,
        spec: DeleteEdgesSpec,
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        ...

    # Bulk / batch --------------------------------------------------------

    def bulk_vertices(
        self,
        spec: BulkVerticesSpec,
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BulkVerticesResult:
        ...

    async def abulk_vertices(
        self,
        spec: BulkVerticesSpec,
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BulkVerticesResult:
        ...

    def batch(
        self,
        ops: List[BatchOperation],
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BatchResult:
        ...

    async def abatch(
        self,
        ops: List[BatchOperation],
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BatchResult:
        ...


class CorpusSemanticKernelGraphClient:
    """
    Semantic Kernel–oriented client wrapper around a Corpus `GraphProtocolV1`.

    This is a thin integration layer that:

    - Translates Semantic Kernel context + settings into a Corpus
      `OperationContext` using `core_ctx_from_semantic_kernel`.
    - Uses `GraphTranslator` (with a Semantic Kernel–specific framework
      translator) to:
        * Build Graph*Spec objects from simple inputs
        * Execute sync + async graph operations
        * Orchestrate streaming with proper cancellation and error handling
    - Delegates all async→sync bridging and streaming glue to GraphTranslator.
    - Attaches rich error context (`attach_context`) on this layer with
      Semantic Kernel–specific hints when failures occur.
    """

    class _SemanticKernelGraphFrameworkTranslator(DefaultGraphFrameworkTranslator):
        """
        Semantic Kernel–specific GraphFrameworkTranslator.

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
    ) -> None:
        self._graph: GraphProtocolV1 = graph_adapter
        self._default_dialect: Optional[str] = default_dialect
        self._default_namespace: Optional[str] = default_namespace

    # ------------------------------------------------------------------ #
    # Translator (lazy, cached)
    # ------------------------------------------------------------------ #

    @cached_property
    def _translator(self) -> GraphTranslator:
        """
        Lazily construct and cache the `GraphTranslator`.

        Uses `cached_property` for thread safety and performance.
        """
        framework_translator = self._SemanticKernelGraphFrameworkTranslator()
        return GraphTranslator(
            adapter=self._graph,
            framework="semantic_kernel",
            translator=framework_translator,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_ctx(
        self,
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Optional[OperationContext]:
        """
        Build an OperationContext from Semantic Kernel–style inputs.

        Expected inputs
        ----------------
        - context: SK context object (optional)
        - settings: SK PromptExecutionSettings (optional)
        - extra_context: Optional mapping merged into attrs (best effort)
        """
        extra = dict(extra_context or {})

        if context is None and settings is None and not extra:
            return None

        try:
            ctx = core_ctx_from_semantic_kernel(
                context,
                settings=settings,
            )
        except Exception as exc:
            attach_context(
                exc,
                framework="semantic_kernel",
                operation="context_translation",
            )
            raise

        if not isinstance(ctx, OperationContext):
            raise BadRequest(
                f"from_semantic_kernel produced unsupported context type: {type(ctx).__name__}",
                code=ErrorCodes.BAD_OPERATION_CONTEXT,
            )

        if extra:
            # Merge extra_context into attrs in a non-destructive way.
            merged_attrs: Dict[str, Any] = dict(ctx.attrs or {})
            merged_attrs.update(extra)
            ctx = OperationContext(
                request_id=ctx.request_id,
                idempotency_key=ctx.idempotency_key,
                deadline_ms=ctx.deadline_ms,
                traceparent=ctx.traceparent,
                tenant=ctx.tenant,
                attrs=merged_attrs,
            )

        return ctx

    @staticmethod
    def _validate_query(query: str) -> None:
        """
        Validate that a query string is non-empty and of the correct type.
        """
        if not isinstance(query, str) or not query.strip():
            raise BadRequest("query must be a non-empty string")

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

        raw: Dict[str, Any] = {
            "text": query,
            "params": dict(params or {}),
            "stream": bool(stream),
        }
        if effective_dialect is not None:
            raw["dialect"] = effective_dialect
        if effective_namespace is not None:
            raw["namespace"] = effective_namespace
        if timeout_ms is not None:
            raw["timeout_ms"] = int(timeout_ms)
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
        return {"namespace": effective_namespace} if effective_namespace is not None else {}

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

    # ------------------------------------------------------------------ #
    # Capabilities / schema / health
    # ------------------------------------------------------------------ #

    @with_error_context("capabilities_sync")
    def capabilities(self) -> Mapping[str, Any]:
        """
        Sync wrapper around capabilities, delegating async→sync bridging
        to GraphTranslator.
        """
        caps = self._translator.capabilities()
        # We normalize to a simple dict for SK consumption.
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

    @with_async_error_context("capabilities_async")
    async def acapabilities(self) -> Mapping[str, Any]:
        """
        Async capabilities accessor.

        We delegate to GraphTranslator for consistency, then normalize to a
        simple dict for SK consumption.
        """
        caps = await self._translator.arun_capabilities()
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

    @with_error_context("get_schema_sync")
    def get_schema(
        self,
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> GraphSchema:
        """
        Sync schema introspection, via GraphTranslator.
        """
        ctx = self._build_ctx(
            context=context,
            settings=settings,
            extra_context=extra_context,
        )
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

    @with_async_error_context("get_schema_async")
    async def aget_schema(
        self,
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> GraphSchema:
        """
        Async schema introspection, via GraphTranslator.
        """
        ctx = self._build_ctx(
            context=context,
            settings=settings,
            extra_context=extra_context,
        )
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

    @with_error_context("health_sync")
    def health(
        self,
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        """
        Sync health check wrapper.

        Delegates async→sync bridging to GraphTranslator.
        """
        ctx = self._build_ctx(
            context=context,
            settings=settings,
            extra_context=extra_context,
        )
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

    @with_async_error_context("health_async")
    async def ahealth(
        self,
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        """
        Async health check wrapper, delegating orchestration to GraphTranslator.
        """
        ctx = self._build_ctx(
            context=context,
            settings=settings,
            extra_context=extra_context,
        )
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

    @with_error_context("query_sync")
    def query(
        self,
        query: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        dialect: Optional[str] = None,
        namespace: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> QueryResult:
        """
        Execute a non-streaming graph query (sync).

        Returns the underlying `QueryResult`.
        """
        self._validate_query(query)

        ctx = self._build_ctx(
            context=context,
            settings=settings,
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

    @with_async_error_context("query_async")
    async def aquery(
        self,
        query: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        dialect: Optional[str] = None,
        namespace: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> QueryResult:
        """
        Execute a non-streaming graph query (async).

        Returns the underlying `QueryResult`.
        """
        self._validate_query(query)

        ctx = self._build_ctx(
            context=context,
            settings=settings,
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

    @with_error_context("stream_query_sync")
    def stream_query(
        self,
        query: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        dialect: Optional[str] = None,
        namespace: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[QueryChunk]:
        """
        Execute a streaming graph query (sync), yielding `QueryChunk` items.

        Delegates streaming orchestration to GraphTranslator, which uses
        SyncStreamBridge under the hood.
        """
        self._validate_query(query)

        ctx = self._build_ctx(
            context=context,
            settings=settings,
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

    @with_async_error_context("stream_query_async")
    async def astream_query(
        self,
        query: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        dialect: Optional[str] = None,
        namespace: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> AsyncIterator[QueryChunk]:
        """
        Execute a streaming graph query (async), yielding `QueryChunk` items.
        """
        self._validate_query(query)

        ctx = self._build_ctx(
            context=context,
            settings=settings,
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

    @with_error_context("upsert_nodes_sync")
    def upsert_nodes(
        self,
        spec: UpsertNodesSpec,
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        """
        Sync wrapper for upserting nodes via GraphTranslator.
        """
        ctx = self._build_ctx(
            context=context,
            settings=settings,
            extra_context=extra_context,
        )
        framework_ctx = self._framework_ctx_for_namespace(getattr(spec, "namespace", None))

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

    @with_async_error_context("upsert_nodes_async")
    async def aupsert_nodes(
        self,
        spec: UpsertNodesSpec,
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        """
        Async wrapper for upserting nodes via GraphTranslator.
        """
        ctx = self._build_ctx(
            context=context,
            settings=settings,
            extra_context=extra_context,
        )
        framework_ctx = self._framework_ctx_for_namespace(getattr(spec, "namespace", None))

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

    @with_error_context("upsert_edges_sync")
    def upsert_edges(
        self,
        spec: UpsertEdgesSpec,
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        """
        Sync wrapper for upserting edges via GraphTranslator.
        """
        ctx = self._build_ctx(
            context=context,
            settings=settings,
            extra_context=extra_context,
        )
        framework_ctx = self._framework_ctx_for_namespace(getattr(spec, "namespace", None))

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

    @with_async_error_context("upsert_edges_async")
    async def aupsert_edges(
        self,
        spec: UpsertEdgesSpec,
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        """
        Async wrapper for upserting edges via GraphTranslator.
        """
        ctx = self._build_ctx(
            context=context,
            settings=settings,
            extra_context=extra_context,
        )
        framework_ctx = self._framework_ctx_for_namespace(getattr(spec, "namespace", None))

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

    @with_error_context("delete_nodes_sync")
    def delete_nodes(
        self,
        spec: DeleteNodesSpec,
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        """
        Sync wrapper for deleting nodes via GraphTranslator.
        """
        ctx = self._build_ctx(
            context=context,
            settings=settings,
            extra_context=extra_context,
        )
        framework_ctx = self._framework_ctx_for_namespace(getattr(spec, "namespace", None))

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

    @with_async_error_context("delete_nodes_async")
    async def adelete_nodes(
        self,
        spec: DeleteNodesSpec,
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        """
        Async wrapper for deleting nodes via GraphTranslator.
        """
        ctx = self._build_ctx(
            context=context,
            settings=settings,
            extra_context=extra_context,
        )
        framework_ctx = self._framework_ctx_for_namespace(getattr(spec, "namespace", None))

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

    @with_error_context("delete_edges_sync")
    def delete_edges(
        self,
        spec: DeleteEdgesSpec,
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        """
        Sync wrapper for deleting edges via GraphTranslator.
        """
        ctx = self._build_ctx(
            context=context,
            settings=settings,
            extra_context=extra_context,
        )
        framework_ctx = self._framework_ctx_for_namespace(getattr(spec, "namespace", None))

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

    @with_async_error_context("delete_edges_async")
    async def adelete_edges(
        self,
        spec: DeleteEdgesSpec,
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        """
        Async wrapper for deleting edges via GraphTranslator.
        """
        ctx = self._build_ctx(
            context=context,
            settings=settings,
            extra_context=extra_context,
        )
        framework_ctx = self._framework_ctx_for_namespace(getattr(spec, "namespace", None))

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

    @with_error_context("bulk_vertices_sync")
    def bulk_vertices(
        self,
        spec: BulkVerticesSpec,
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BulkVerticesResult:
        """
        Sync wrapper for bulk_vertices via GraphTranslator.
        """
        ctx = self._build_ctx(
            context=context,
            settings=settings,
            extra_context=extra_context,
        )
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

    @with_async_error_context("bulk_vertices_async")
    async def abulk_vertices(
        self,
        spec: BulkVerticesSpec,
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BulkVerticesResult:
        """
        Async wrapper for bulk_vertices via GraphTranslator.
        """
        ctx = self._build_ctx(
            context=context,
            settings=settings,
            extra_context=extra_context,
        )
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

    @with_error_context("batch_sync")
    def batch(
        self,
        ops: List[BatchOperation],
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BatchResult:
        """
        Sync wrapper for batch operations via GraphTranslator.
        """
        ctx = self._build_ctx(
            context=context,
            settings=settings,
            extra_context=extra_context,
        )
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

    @with_async_error_context("batch_async")
    async def abatch(
        self,
        ops: List[BatchOperation],
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BatchResult:
        """
        Async wrapper for batch operations via GraphTranslator.
        """
        ctx = self._build_ctx(
            context=context,
            settings=settings,
            extra_context=extra_context,
        )
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
    "SemanticKernelGraphClientProtocol",
    "CorpusSemanticKernelGraphClient",
    "ErrorCodes",
]
