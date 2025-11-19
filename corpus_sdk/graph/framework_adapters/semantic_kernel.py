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

Design philosophy
-----------------
- Protocol-first: Semantic Kernel is a thin skin over the Corpus graph adapter.
- All heavy lifting (deadlines, breakers, rate limits, caching, etc.) lives in
  the underlying `BaseGraphAdapter` / `GraphProtocolV1` implementation.
- This layer focuses on:
    * Translating SK context/settings → OperationContext
    * Building GraphQuerySpec via GraphTranslator
    * Bridging async Corpus APIs into sync calls via AsyncBridge
    * Providing safe streaming utilities for sync callers via SyncStreamBridge

Usage (example)
---------------

    from corpus_sdk.graph.graph_base import BaseGraphAdapter
    from corpus_sdk.graph.framework_adapters.semantic_kernel import (
        CorpusSemanticKernelGraphClient,
    )

    graph_adapter: BaseGraphAdapter = ...
    client = CorpusSemanticKernelGraphClient(
        graph_adapter=graph_adapter,
        default_dialect="cypher",
        default_namespace="my-graph",
    )

    # Inside an SK function / plugin:

    # Sync query
    result = client.query(
        "MATCH (n) RETURN n LIMIT 5",
        context=sk_context,
        settings=prompt_settings,
    )

    # Async query
    result = await client.aquery(
        "MATCH (n) RETURN n LIMIT 5",
        context=sk_context,
        settings=prompt_settings,
    )

    # Sync streaming query
    for chunk in client.stream_query(
        "MATCH (n) RETURN n LIMIT 100",
        context=sk_context,
        settings=prompt_settings,
    ):
        process(chunk.records)

    # Async streaming query
    async for chunk in client.astream_query(
        "MATCH (n) RETURN n LIMIT 100",
        context=sk_context,
        settings=prompt_settings,
    ):
        process(chunk.records)
"""

from __future__ import annotations

import logging
from typing import (
    Any,
    AsyncIterator,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
)

from corpus_sdk.core.context_translation import (
    from_semantic_kernel as core_ctx_from_semantic_kernel,
)
from corpus_sdk.core.error_context import attach_context
from corpus_sdk.core.sync_stream_bridge import sync_stream
from corpus_sdk.graph.graph_base import (
    BadRequest,
    BatchOperation,
    BatchResult,
    BulkVerticesResult,
    BulkVerticesSpec,
    DeleteEdgesSpec,
    DeleteNodesSpec,
    DeleteResult,
    GraphAdapterError,
    GraphProtocolV1,
    GraphQuerySpec,
    GraphSchema,
    QueryChunk,
    QueryResult,
    UpsertEdgesSpec,
    UpsertNodesSpec,
    UpsertResult,
)
from corpus_sdk.graph.graph_translation import GraphTranslator
from corpus_sdk.llm.framework_adapters.common.async_bridge import AsyncBridge

logger = logging.getLogger(__name__)


class CorpusSemanticKernelGraphClient:
    """
    Semantic Kernel–oriented client wrapper around a Corpus `GraphProtocolV1`.

    This is a thin integration layer that:

    - Translates Semantic Kernel context + settings into a Corpus
      `OperationContext` using `GraphTranslator.from_semantic_kernel` or, as
      a fallback, `core.context_translation.from_semantic_kernel`.
    - Uses `GraphTranslator` to build `GraphQuerySpec` from simple parameters
      (query string, dialect, params, namespace, timeout_ms, stream flag).
    - Provides sync + async APIs for:
        * query / aquery
        * stream_query / astream_query
        * upsert_nodes / aupsert_nodes
        * upsert_edges / aupsert_edges
        * delete_nodes / adelete_nodes
        * delete_edges / adelete_edges
        * bulk_vertices / abulk_vertices
        * batch / abatch
        * get_schema / aget_schema
        * health / ahealth
    - Uses `AsyncBridge` and `sync_stream` to safely bridge async adapter
      methods into synchronous calls.
    - Attaches rich error context (`attach_context`) to every failure path.

    Attributes
    ----------
    graph_adapter:
        Underlying Corpus graph adapter implementing `GraphProtocolV1`.

    default_dialect:
        Default query dialect used when caller does not specify one.

    default_namespace:
        Default logical graph / namespace for operations when not
        explicitly overridden by the caller.
    """

    def __init__(
        self,
        *,
        graph_adapter: GraphProtocolV1,
        default_dialect: Optional[str] = None,
        default_namespace: Optional[str] = None,
    ) -> None:
        self._graph = graph_adapter
        self._default_dialect = default_dialect
        self._default_namespace = default_namespace

        # Centralized query + context translation
        self._translator = GraphTranslator(
            default_dialect=default_dialect,
            default_namespace=default_namespace,
            framework="semantic_kernel",
        )

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _build_ctx(self, **kwargs: Any):
        """
        Build an OperationContext from Semantic Kernel–style inputs.

        Expected kwargs:
            - context: SK context object (optional)
            - settings: SK PromptExecutionSettings (optional)
            - extra_context: Optional mapping merged into attrs (best effort)
        """
        context = kwargs.get("context")
        settings = kwargs.get("settings")
        extra_context = kwargs.get("extra_context") or {}

        if context is None and settings is None and not extra_context:
            return None

        # Preferred path: GraphTranslator's SK-aware helper.
        try:
            return self._translator.from_semantic_kernel(
                context,
                settings=settings,
                **extra_context,
            )
        except Exception as exc:
            # Fallback to core context translation; still attach error context
            logger.debug(
                "GraphTranslator.from_semantic_kernel failed, "
                "falling back to core_ctx_from_semantic_kernel: %s",
                exc,
            )
            try:
                ctx = core_ctx_from_semantic_kernel(
                    context,
                    settings=settings,
                )
            except Exception as core_exc:
                attach_context(
                    core_exc,
                    framework="semantic_kernel",
                    operation="context_translation",
                )
                raise
            return ctx

    def _build_query_spec(
        self,
        query: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        dialect: Optional[str] = None,
        namespace: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        stream: bool = False,
    ) -> GraphQuerySpec:
        """
        Build a `GraphQuerySpec` via GraphTranslator with sane defaults.
        """
        try:
            spec = self._translator.build_query_spec(
                query=query,
                dialect=dialect,
                params=params,
                namespace=namespace,
                timeout_ms=timeout_ms,
                stream=stream,
            )
            return spec
        except Exception as exc:
            attach_context(
                exc,
                framework="semantic_kernel",
                operation="build_query_spec",
                query=query,
                dialect=dialect or self._default_dialect,
                namespace=namespace or self._default_namespace,
                stream=stream,
            )
            raise

    # --------------------------------------------------------------------- #
    # Capabilities / schema / health
    # --------------------------------------------------------------------- #

    def capabilities(self) -> Mapping[str, Any]:
        """
        Sync wrapper around `graph_adapter.capabilities()`.

        Returns the dataclass as a dict for SK-friendly consumption.
        """
        try:
            caps = AsyncBridge.run_async(self._graph.capabilities())
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
        except Exception as exc:
            attach_context(
                exc,
                framework="semantic_kernel",
                operation="capabilities_sync",
            )
            raise

    async def acapabilities(self) -> Mapping[str, Any]:
        """
        Async capabilities accessor with SK-friendly dict output.
        """
        try:
            caps = await self._graph.capabilities()
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
        except Exception as exc:
            attach_context(
                exc,
                framework="semantic_kernel",
                operation="capabilities_async",
            )
            raise

    def get_schema(
        self,
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> GraphSchema:
        """
        Sync wrapper around `graph_adapter.get_schema(...)`.
        """
        ctx = self._build_ctx(context=context, settings=settings, extra_context=extra_context)
        try:
            return AsyncBridge.run_async(self._graph.get_schema(ctx=ctx))
        except Exception as exc:
            attach_context(
                exc,
                framework="semantic_kernel",
                operation="get_schema_sync",
            )
            raise

    async def aget_schema(
        self,
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> GraphSchema:
        """
        Async wrapper around `graph_adapter.get_schema(...)`.
        """
        ctx = self._build_ctx(context=context, settings=settings, extra_context=extra_context)
        try:
            return await self._graph.get_schema(ctx=ctx)
        except Exception as exc:
            attach_context(
                exc,
                framework="semantic_kernel",
                operation="get_schema_async",
            )
            raise

    def health(
        self,
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        """
        Sync health check wrapper.

        Returns the normalized health mapping from the underlying adapter.
        """
        ctx = self._build_ctx(context=context, settings=settings, extra_context=extra_context)
        try:
            return AsyncBridge.run_async(self._graph.health(ctx=ctx))
        except Exception as exc:
            attach_context(
                exc,
                framework="semantic_kernel",
                operation="health_sync",
            )
            raise

    async def ahealth(
        self,
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        """
        Async health check wrapper.
        """
        ctx = self._build_ctx(context=context, settings=settings, extra_context=extra_context)
        try:
            return await self._graph.health(ctx=ctx)
        except Exception as exc:
            attach_context(
                exc,
                framework="semantic_kernel",
                operation="health_async",
            )
            raise

    # --------------------------------------------------------------------- #
    # Query (sync + async)
    # --------------------------------------------------------------------- #

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
        """
        if not isinstance(query, str) or not query.strip():
            raise BadRequest("query must be a non-empty string")

        ctx = self._build_ctx(context=context, settings=settings, extra_context=extra_context)
        spec = self._build_query_spec(
            query=query,
            params=params,
            dialect=dialect,
            namespace=namespace,
            timeout_ms=timeout_ms,
            stream=False,
        )

        try:
            return AsyncBridge.run_async(self._graph.query(spec, ctx=ctx))
        except Exception as exc:
            attach_context(
                exc,
                framework="semantic_kernel",
                operation="query_sync",
                query=query,
                dialect=spec.dialect,
                namespace=spec.namespace,
            )
            raise

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
        """
        if not isinstance(query, str) or not query.strip():
            raise BadRequest("query must be a non-empty string")

        ctx = self._build_ctx(context=context, settings=settings, extra_context=extra_context)
        spec = self._build_query_spec(
            query=query,
            params=params,
            dialect=dialect,
            namespace=namespace,
            timeout_ms=timeout_ms,
            stream=False,
        )

        try:
            return await self._graph.query(spec, ctx=ctx)
        except Exception as exc:
            attach_context(
                exc,
                framework="semantic_kernel",
                operation="query_async",
                query=query,
                dialect=spec.dialect,
                namespace=spec.namespace,
            )
            raise

    # --------------------------------------------------------------------- #
    # Streaming query (sync + async)
    # --------------------------------------------------------------------- #

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

        Uses `sync_stream` (SyncStreamBridge wrapper) under the hood to
        safely bridge the async stream into a synchronous iterator.
        """
        if not isinstance(query, str) or not query.strip():
            raise BadRequest("query must be a non-empty string")

        ctx = self._build_ctx(context=context, settings=settings, extra_context=extra_context)
        spec = self._build_query_spec(
            query=query,
            params=params,
            dialect=dialect,
            namespace=namespace,
            timeout_ms=timeout_ms,
            stream=True,
        )

        async def _agen() -> AsyncIterator[QueryChunk]:
            try:
                async for chunk in self._graph.stream_query(spec, ctx=ctx):
                    yield chunk
            except Exception as exc:
                attach_context(
                    exc,
                    framework="semantic_kernel",
                    operation="stream_query_async",
                    query=query,
                    dialect=spec.dialect,
                    namespace=spec.namespace,
                )
                raise

        for chunk in sync_stream(
            _agen,
            framework="semantic_kernel",
            error_context={
                "operation": "stream_query_sync",
                "query": query,
                "dialect": spec.dialect,
                "namespace": spec.namespace,
            },
        ):
            yield chunk

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
        if not isinstance(query, str) or not query.strip():
            raise BadRequest("query must be a non-empty string")

        ctx = self._build_ctx(context=context, settings=settings, extra_context=extra_context)
        spec = self._build_query_spec(
            query=query,
            params=params,
            dialect=dialect,
            namespace=namespace,
            timeout_ms=timeout_ms,
            stream=True,
        )

        try:
            async for chunk in self._graph.stream_query(spec, ctx=ctx):
                yield chunk
        except Exception as exc:
            attach_context(
                exc,
                framework="semantic_kernel",
                operation="stream_query_async",
                query=query,
                dialect=spec.dialect,
                namespace=spec.namespace,
            )
            raise

    # --------------------------------------------------------------------- #
    # Upsert nodes / edges (sync + async)
    # --------------------------------------------------------------------- #

    def upsert_nodes(
        self,
        spec: UpsertNodesSpec,
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        """
        Sync wrapper for `graph_adapter.upsert_nodes(...)`.
        """
        ctx = self._build_ctx(context=context, settings=settings, extra_context=extra_context)
        try:
            return AsyncBridge.run_async(self._graph.upsert_nodes(spec, ctx=ctx))
        except Exception as exc:
            attach_context(
                exc,
                framework="semantic_kernel",
                operation="upsert_nodes_sync",
                namespace=getattr(spec, "namespace", None),
                count=len(spec.nodes),
            )
            raise

    async def aupsert_nodes(
        self,
        spec: UpsertNodesSpec,
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        """
        Async wrapper for `graph_adapter.upsert_nodes(...)`.
        """
        ctx = self._build_ctx(context=context, settings=settings, extra_context=extra_context)
        try:
            return await self._graph.upsert_nodes(spec, ctx=ctx)
        except Exception as exc:
            attach_context(
                exc,
                framework="semantic_kernel",
                operation="upsert_nodes_async",
                namespace=getattr(spec, "namespace", None),
                count=len(spec.nodes),
            )
            raise

    def upsert_edges(
        self,
        spec: UpsertEdgesSpec,
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        """
        Sync wrapper for `graph_adapter.upsert_edges(...)`.
        """
        ctx = self._build_ctx(context=context, settings=settings, extra_context=extra_context)
        try:
            return AsyncBridge.run_async(self._graph.upsert_edges(spec, ctx=ctx))
        except Exception as exc:
            attach_context(
                exc,
                framework="semantic_kernel",
                operation="upsert_edges_sync",
                namespace=getattr(spec, "namespace", None),
                count=len(spec.edges),
            )
            raise

    async def aupsert_edges(
        self,
        spec: UpsertEdgesSpec,
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        """
        Async wrapper for `graph_adapter.upsert_edges(...)`.
        """
        ctx = self._build_ctx(context=context, settings=settings, extra_context=extra_context)
        try:
            return await self._graph.upsert_edges(spec, ctx=ctx)
        except Exception as exc:
            attach_context(
                exc,
                framework="semantic_kernel",
                operation="upsert_edges_async",
                namespace=getattr(spec, "namespace", None),
                count=len(spec.edges),
            )
            raise

    # --------------------------------------------------------------------- #
    # Delete nodes / edges (sync + async)
    # --------------------------------------------------------------------- #

    def delete_nodes(
        self,
        spec: DeleteNodesSpec,
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        """
        Sync wrapper for `graph_adapter.delete_nodes(...)`.
        """
        ctx = self._build_ctx(context=context, settings=settings, extra_context=extra_context)
        try:
            return AsyncBridge.run_async(self._graph.delete_nodes(spec, ctx=ctx))
        except Exception as exc:
            attach_context(
                exc,
                framework="semantic_kernel",
                operation="delete_nodes_sync",
                namespace=getattr(spec, "namespace", None),
                ids_count=len(spec.ids or []),
            )
            raise

    async def adelete_nodes(
        self,
        spec: DeleteNodesSpec,
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        """
        Async wrapper for `graph_adapter.delete_nodes(...)`.
        """
        ctx = self._build_ctx(context=context, settings=settings, extra_context=extra_context)
        try:
            return await self._graph.delete_nodes(spec, ctx=ctx)
        except Exception as exc:
            attach_context(
                exc,
                framework="semantic_kernel",
                operation="delete_nodes_async",
                namespace=getattr(spec, "namespace", None),
                ids_count=len(spec.ids or []),
            )
            raise

    def delete_edges(
        self,
        spec: DeleteEdgesSpec,
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        """
        Sync wrapper for `graph_adapter.delete_edges(...)`.
        """
        ctx = self._build_ctx(context=context, settings=settings, extra_context=extra_context)
        try:
            return AsyncBridge.run_async(self._graph.delete_edges(spec, ctx=ctx))
        except Exception as exc:
            attach_context(
                exc,
                framework="semantic_kernel",
                operation="delete_edges_sync",
                namespace=getattr(spec, "namespace", None),
                ids_count=len(spec.ids or []),
            )
            raise

    async def adelete_edges(
        self,
        spec: DeleteEdgesSpec,
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        """
        Async wrapper for `graph_adapter.delete_edges(...)`.
        """
        ctx = self._build_ctx(context=context, settings=settings, extra_context=extra_context)
        try:
            return await self._graph.delete_edges(spec, ctx=ctx)
        except Exception as exc:
            attach_context(
                exc,
                framework="semantic_kernel",
                operation="delete_edges_async",
                namespace=getattr(spec, "namespace", None),
                ids_count=len(spec.ids or []),
            )
            raise

    # --------------------------------------------------------------------- #
    # Bulk vertices (sync + async)
    # --------------------------------------------------------------------- #

    def bulk_vertices(
        self,
        spec: BulkVerticesSpec,
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BulkVerticesResult:
        """
        Sync wrapper for `graph_adapter.bulk_vertices(...)`.
        """
        ctx = self._build_ctx(context=context, settings=settings, extra_context=extra_context)
        try:
            return AsyncBridge.run_async(self._graph.bulk_vertices(spec, ctx=ctx))
        except Exception as exc:
            attach_context(
                exc,
                framework="semantic_kernel",
                operation="bulk_vertices_sync",
                namespace=getattr(spec, "namespace", None),
                limit=spec.limit,
            )
            raise

    async def abulk_vertices(
        self,
        spec: BulkVerticesSpec,
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BulkVerticesResult:
        """
        Async wrapper for `graph_adapter.bulk_vertices(...)`.
        """
        ctx = self._build_ctx(context=context, settings=settings, extra_context=extra_context)
        try:
            return await self._graph.bulk_vertices(spec, ctx=ctx)
        except Exception as exc:
            attach_context(
                exc,
                framework="semantic_kernel",
                operation="bulk_vertices_async",
                namespace=getattr(spec, "namespace", None),
                limit=spec.limit,
            )
            raise

    # --------------------------------------------------------------------- #
    # Batch (sync + async)
    # --------------------------------------------------------------------- #

    def batch(
        self,
        ops: List[BatchOperation],
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BatchResult:
        """
        Sync wrapper for `graph_adapter.batch(...)`.
        """
        ctx = self._build_ctx(context=context, settings=settings, extra_context=extra_context)
        try:
            return AsyncBridge.run_async(self._graph.batch(ops, ctx=ctx))
        except Exception as exc:
            attach_context(
                exc,
                framework="semantic_kernel",
                operation="batch_sync",
                ops_count=len(ops),
            )
            raise

    async def abatch(
        self,
        ops: List[BatchOperation],
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BatchResult:
        """
        Async wrapper for `graph_adapter.batch(...)`.
        """
        ctx = self._build_ctx(context=context, settings=settings, extra_context=extra_context)
        try:
            return await self._graph.batch(ops, ctx=ctx)
        except Exception as exc:
            attach_context(
                exc,
                framework="semantic_kernel",
                operation="batch_async",
                ops_count=len(ops),
            )
            raise


__all__ = [
    "CorpusSemanticKernelGraphClient",
]
