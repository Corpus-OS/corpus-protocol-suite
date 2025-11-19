# corpus_sdk/graph/framework_adapters/langchain.py
# SPDX-License-Identifier: Apache-2.0

"""
LangChain adapter for Corpus Graph protocol.

This module exposes a Corpus `GraphProtocolV1` implementation as a
LangChain-friendly client, with:

- Sync + async query APIs (mirroring the vector/LLM patterns)
- Streaming query support (async + sync via SyncStreamBridge)
- Proper integration with Corpus GraphProtocolV1
- Namespace-aware behavior and OperationContext propagation
- Error context attachment for rich observability

Design philosophy
-----------------
- Protocol-first: LangChain is a thin skin over the Corpus graph adapter.
- All heavy lifting (deadlines, breaker, rate-limiter, caching) lives in
  the underlying `BaseGraphAdapter` (or other GraphProtocolV1 impl).
- This layer focuses on:
    * Translating LangChain RunnableConfig → OperationContext
    * Building GraphQuerySpec from simple parameters
    * Bridging async Corpus APIs into a sync interface via AsyncBridge
    * Providing streaming utilities that are safe for sync callers

Usage (example)
---------------

    from corpus_sdk.graph.graph_base import BaseGraphAdapter
    from corpus_sdk.graph.framework_adapters.langchain import (
        CorpusLangChainGraphClient,
    )

    graph_adapter: BaseGraphAdapter = ...
    client = CorpusLangChainGraphClient(
        graph_adapter=graph_adapter,
        default_dialect="cypher",
        default_namespace="my-graph",
    )

    # Sync query
    result = client.query("MATCH (n) RETURN n LIMIT 5")

    # Async query
    result = await client.aquery("MATCH (n) RETURN n LIMIT 5")

    # Sync streaming query
    for chunk in client.stream_query("MATCH (n) RETURN n LIMIT 100", batch_size=50):
        process(chunk.records)

    # Async streaming query
    async for chunk in client.astream_query("MATCH (n) RETURN n LIMIT 100"):
        process(chunk.records)
"""

from __future__ import annotations

import logging
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
)

from corpus_sdk.core.context_translation import (
    from_langchain as context_from_langchain,
)
from corpus_sdk.core.error_context import attach_context
from corpus_sdk.core.sync_stream_bridge import sync_stream
from corpus_sdk.graph.graph_base import (
    BatchOperation,
    BatchResult,
    BadRequest,
    BulkVerticesResult,
    BulkVerticesSpec,
    DeleteEdgesSpec,
    DeleteNodesSpec,
    GraphAdapterError,
    GraphProtocolV1,
    GraphQuerySpec,
    GraphSchema,
    QueryChunk,
    QueryResult,
    UpsertEdgesSpec,
    UpsertNodesSpec,
)
from corpus_sdk.graph.graph_translation import GraphTranslator
from corpus_sdk.llm.framework_adapters.common.async_bridge import AsyncBridge

logger = logging.getLogger(__name__)


class CorpusLangChainGraphClient:
    """
    LangChain-oriented client wrapper around a Corpus `GraphProtocolV1`.

    This is a thin integration layer that:

    - Translates LangChain `RunnableConfig`-like `config` dicts into a
      Corpus `OperationContext` using `context_translation.from_langchain`.
    - Uses `GraphTranslator` to build `GraphQuerySpec` objects from simple
      parameters (query string, dialect, params, namespace, timeout).
    - Provides both sync and async APIs for:
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
      methods into synchronous LangChain-friendly calls.
    - Attaches rich error context to exceptions for easier debugging.

    Attributes
    ----------
    graph_adapter:
        Underlying Corpus graph adapter implementing `GraphProtocolV1`.

    default_dialect:
        Default query dialect to assume when caller does not specify one.

    default_namespace:
        Default logical graph / namespace for queries and mutations
        when not explicitly overridden by the caller.
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

        # GraphTranslator centralizes GraphQuerySpec construction and
        # framework-aware context translation semantics.
        self._translator = GraphTranslator(
            default_dialect=default_dialect,
            default_namespace=default_namespace,
            framework="langchain",
        )

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _build_ctx(self, **kwargs: Any):
        """
        Build an OperationContext from LangChain-style config.

        Expected kwargs:
            - config: RunnableConfig-like mapping (optional)
        """
        config = kwargs.get("config")
        if config is None:
            return None

        # Prefer the GraphTranslator's framework-aware path if available.
        try:
            return self._translator.from_langchain(config)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "GraphTranslator.from_langchain failed; falling back to "
                "core context_from_langchain: %s",
                exc,
            )

        # Fallback: direct use of core context_translation.
        try:
            return context_from_langchain(config)
        except Exception as exc:  # noqa: BLE001
            logger.debug("context_from_langchain failed: %s", exc)
            return None

    def _build_query_spec(
        self,
        query: str,
        *,
        dialect: Optional[str],
        params: Optional[Mapping[str, Any]],
        namespace: Optional[str],
        timeout_ms: Optional[int],
        stream: bool,
    ) -> GraphQuerySpec:
        """
        Build a GraphQuerySpec via GraphTranslator, enforcing defaults.
        """
        eff_dialect = dialect or self._default_dialect
        eff_namespace = namespace or self._default_namespace
        try:
            return self._translator.build_query_spec(
                text=query,
                dialect=eff_dialect,
                params=params,
                namespace=eff_namespace,
                timeout_ms=timeout_ms,
                stream=stream,
            )
        except Exception as exc:  # noqa: BLE001
            # Normalize to BadRequest with attached context for caller visibility.
            attach_context(
                exc,
                framework="langchain",
                component="graph",
                operation="build_query_spec",
                dialect=eff_dialect,
                namespace=eff_namespace,
            )
            if isinstance(exc, GraphAdapterError):
                raise
            raise BadRequest(
                f"failed to build GraphQuerySpec: {exc}",
                details={
                    "dialect": eff_dialect,
                    "namespace": eff_namespace,
                    "stream": stream,
                },
            ) from exc

    # --------------------------------------------------------------------- #
    # Query API (sync + async)
    # --------------------------------------------------------------------- #

    def query(
        self,
        query: str,
        *,
        dialect: Optional[str] = None,
        params: Optional[Mapping[str, Any]] = None,
        namespace: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        **kwargs: Any,
    ) -> QueryResult:
        """
        Execute a non-streaming graph query (sync).

        This uses AsyncBridge under the hood to execute the underlying
        async `graph_adapter.query` method in a safe synchronous manner.
        """
        ctx = self._build_ctx(**kwargs)
        spec = self._build_query_spec(
            query=query,
            dialect=dialect,
            params=params,
            namespace=namespace,
            timeout_ms=timeout_ms,
            stream=False,
        )

        try:
            return AsyncBridge.run_async(self._graph.query(spec, ctx=ctx))
        except Exception as exc:  # noqa: BLE001
            # Attach rich error context before surfacing the exception
            try:
                attach_context(
                    exc,
                    framework="langchain",
                    component="graph",
                    operation="query_sync",
                    dialect=spec.dialect,
                    namespace=spec.namespace,
                )
            except Exception:  # noqa: BLE001
                logger.debug("attach_context failed in query_sync", exc_info=True)
            raise

    async def aquery(
        self,
        query: str,
        *,
        dialect: Optional[str] = None,
        params: Optional[Mapping[str, Any]] = None,
        namespace: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        **kwargs: Any,
    ) -> QueryResult:
        """
        Execute a non-streaming graph query (async).
        """
        ctx = self._build_ctx(**kwargs)
        spec = self._build_query_spec(
            query=query,
            dialect=dialect,
            params=params,
            namespace=namespace,
            timeout_ms=timeout_ms,
            stream=False,
        )
        try:
            return await self._graph.query(spec, ctx=ctx)
        except Exception as exc:  # noqa: BLE001
            try:
                attach_context(
                    exc,
                    framework="langchain",
                    component="graph",
                    operation="query_async",
                    dialect=spec.dialect,
                    namespace=spec.namespace,
                )
            except Exception:  # noqa: BLE001
                logger.debug("attach_context failed in query_async", exc_info=True)
            raise

    # --------------------------------------------------------------------- #
    # Streaming Query API (sync + async)
    # --------------------------------------------------------------------- #

    def stream_query(
        self,
        query: str,
        *,
        dialect: Optional[str] = None,
        params: Optional[Mapping[str, Any]] = None,
        namespace: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[QueryChunk]:
        """
        Execute a streaming graph query (sync), yielding QueryChunk objects.

        This uses SyncStreamBridge (via `sync_stream`) to bridge the async
        `stream_query` call into a safe synchronous iterator, while still
        preserving:
            - deadline propagation
            - backpressure
            - error context
        """
        ctx = self._build_ctx(**kwargs)
        spec = self._build_query_spec(
            query=query,
            dialect=dialect,
            params=params,
            namespace=namespace,
            timeout_ms=timeout_ms,
            stream=True,
        )

        async def _agen() -> AsyncIterator[QueryChunk]:
            try:
                async for chunk in self._graph.stream_query(spec, ctx=ctx):
                    yield chunk
            except Exception as exc:  # noqa: BLE001
                try:
                    attach_context(
                        exc,
                        framework="langchain",
                        component="graph",
                        operation="stream_query_async",
                        dialect=spec.dialect,
                        namespace=spec.namespace,
                    )
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "attach_context failed in stream_query_async",
                        exc_info=True,
                    )
                raise

        for chunk in sync_stream(
            _agen,
            framework="langchain",
            error_context={
                "component": "graph",
                "operation": "stream_query_sync",
                "dialect": spec.dialect,
                "namespace": spec.namespace,
            },
        ):
            yield chunk

    async def astream_query(
        self,
        query: str,
        *,
        dialect: Optional[str] = None,
        params: Optional[Mapping[str, Any]] = None,
        namespace: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[QueryChunk]:
        """
        Execute a streaming graph query (async), yielding QueryChunk objects.
        """
        ctx = self._build_ctx(**kwargs)
        spec = self._build_query_spec(
            query=query,
            dialect=dialect,
            params=params,
            namespace=namespace,
            timeout_ms=timeout_ms,
            stream=True,
        )

        async def _inner() -> AsyncIterator[QueryChunk]:
            try:
                async for chunk in self._graph.stream_query(spec, ctx=ctx):
                    yield chunk
            except Exception as exc:  # noqa: BLE001
                try:
                    attach_context(
                        exc,
                        framework="langchain",
                        component="graph",
                        operation="stream_query_async",
                        dialect=spec.dialect,
                        namespace=spec.namespace,
                    )
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "attach_context failed in stream_query_async (inner)",
                        exc_info=True,
                    )
                raise

        return _inner()

    # --------------------------------------------------------------------- #
    # Write / mutation API (sync + async)
    # These take GraphProtocol spec objects directly; they are thin wrappers
    # that provide error-context + AsyncBridge usage for sync callers.
    # --------------------------------------------------------------------- #

    def upsert_nodes(
        self,
        spec: UpsertNodesSpec,
        **kwargs: Any,
    ) -> UpsertNodesSpec.__annotations__.get("return", Any):  # type: ignore[has-type]
        """
        Batch upsert nodes (sync).

        The caller is expected to construct an `UpsertNodesSpec` in terms
        of core graph types (Node, namespace, etc.). This wrapper only
        handles sync bridging and error context.
        """
        ctx = self._build_ctx(**kwargs)
        try:
            return AsyncBridge.run_async(self._graph.upsert_nodes(spec, ctx=ctx))
        except Exception as exc:  # noqa: BLE001
            try:
                attach_context(
                    exc,
                    framework="langchain",
                    component="graph",
                    operation="upsert_nodes_sync",
                    namespace=spec.namespace,
                    node_count=len(spec.nodes),
                )
            except Exception:  # noqa: BLE001
                logger.debug("attach_context failed in upsert_nodes_sync", exc_info=True)
            raise

    async def aupsert_nodes(
        self,
        spec: UpsertNodesSpec,
        **kwargs: Any,
    ):
        """
        Batch upsert nodes (async).
        """
        ctx = self._build_ctx(**kwargs)
        try:
            return await self._graph.upsert_nodes(spec, ctx=ctx)
        except Exception as exc:  # noqa: BLE001
            try:
                attach_context(
                    exc,
                    framework="langchain",
                    component="graph",
                    operation="upsert_nodes_async",
                    namespace=spec.namespace,
                    node_count=len(spec.nodes),
                )
            except Exception:  # noqa: BLE001
                logger.debug("attach_context failed in upsert_nodes_async", exc_info=True)
            raise

    def upsert_edges(
        self,
        spec: UpsertEdgesSpec,
        **kwargs: Any,
    ):
        """
        Batch upsert edges (sync).
        """
        ctx = self._build_ctx(**kwargs)
        try:
            return AsyncBridge.run_async(self._graph.upsert_edges(spec, ctx=ctx))
        except Exception as exc:  # noqa: BLE001
            try:
                attach_context(
                    exc,
                    framework="langchain",
                    component="graph",
                    operation="upsert_edges_sync",
                    namespace=spec.namespace,
                    edge_count=len(spec.edges),
                )
            except Exception:  # noqa: BLE001
                logger.debug("attach_context failed in upsert_edges_sync", exc_info=True)
            raise

    async def aupsert_edges(
        self,
        spec: UpsertEdgesSpec,
        **kwargs: Any,
    ):
        """
        Batch upsert edges (async).
        """
        ctx = self._build_ctx(**kwargs)
        try:
            return await self._graph.upsert_edges(spec, ctx=ctx)
        except Exception as exc:  # noqa: BLE001
            try:
                attach_context(
                    exc,
                    framework="langchain",
                    component="graph",
                    operation="upsert_edges_async",
                    namespace=spec.namespace,
                    edge_count=len(spec.edges),
                )
            except Exception:  # noqa: BLE001
                logger.debug("attach_context failed in upsert_edges_async", exc_info=True)
            raise

    def delete_nodes(
        self,
        spec: DeleteNodesSpec,
        **kwargs: Any,
    ):
        """
        Batch delete nodes by IDs and/or filter (sync).
        """
        ctx = self._build_ctx(**kwargs)
        try:
            return AsyncBridge.run_async(self._graph.delete_nodes(spec, ctx=ctx))
        except Exception as exc:  # noqa: BLE001
            try:
                attach_context(
                    exc,
                    framework="langchain",
                    component="graph",
                    operation="delete_nodes_sync",
                    namespace=spec.namespace,
                    ids_count=len(spec.ids),
                    has_filter=bool(spec.filter),
                )
            except Exception:  # noqa: BLE001
                logger.debug("attach_context failed in delete_nodes_sync", exc_info=True)
            raise

    async def adelete_nodes(
        self,
        spec: DeleteNodesSpec,
        **kwargs: Any,
    ):
        """
        Batch delete nodes by IDs and/or filter (async).
        """
        ctx = self._build_ctx(**kwargs)
        try:
            return await self._graph.delete_nodes(spec, ctx=ctx)
        except Exception as exc:  # noqa: BLE001
            try:
                attach_context(
                    exc,
                    framework="langchain",
                    component="graph",
                    operation="delete_nodes_async",
                    namespace=spec.namespace,
                    ids_count=len(spec.ids),
                    has_filter=bool(spec.filter),
                )
            except Exception:  # noqa: BLE001
                logger.debug("attach_context failed in delete_nodes_async", exc_info=True)
            raise

    def delete_edges(
        self,
        spec: DeleteEdgesSpec,
        **kwargs: Any,
    ):
        """
        Batch delete edges by IDs and/or filter (sync).
        """
        ctx = self._build_ctx(**kwargs)
        try:
            return AsyncBridge.run_async(self._graph.delete_edges(spec, ctx=ctx))
        except Exception as exc:  # noqa: BLE001
            try:
                attach_context(
                    exc,
                    framework="langchain",
                    component="graph",
                    operation="delete_edges_sync",
                    namespace=spec.namespace,
                    ids_count=len(spec.ids),
                    has_filter=bool(spec.filter),
                )
            except Exception:  # noqa: BLE001
                logger.debug("attach_context failed in delete_edges_sync", exc_info=True)
            raise

    async def adelete_edges(
        self,
        spec: DeleteEdgesSpec,
        **kwargs: Any,
    ):
        """
        Batch delete edges by IDs and/or filter (async).
        """
        ctx = self._build_ctx(**kwargs)
        try:
            return await self._graph.delete_edges(spec, ctx=ctx)
        except Exception as exc:  # noqa: BLE001
            try:
                attach_context(
                    exc,
                    framework="langchain",
                    component="graph",
                    operation="delete_edges_async",
                    namespace=spec.namespace,
                    ids_count=len(spec.ids),
                    has_filter=bool(spec.filter),
                )
            except Exception:  # noqa: BLE001
                logger.debug("attach_context failed in delete_edges_async", exc_info=True)
            raise

    # --------------------------------------------------------------------- #
    # Bulk vertices / batch / schema / health (sync + async)
    # --------------------------------------------------------------------- #

    def bulk_vertices(
        self,
        spec: BulkVerticesSpec,
        **kwargs: Any,
    ) -> BulkVerticesResult:
        """
        Bulk vertex scan (sync).
        """
        ctx = self._build_ctx(**kwargs)
        try:
            return AsyncBridge.run_async(self._graph.bulk_vertices(spec, ctx=ctx))
        except Exception as exc:  # noqa: BLE001
            try:
                attach_context(
                    exc,
                    framework="langchain",
                    component="graph",
                    operation="bulk_vertices_sync",
                    namespace=spec.namespace,
                    limit=spec.limit,
                    has_filter=bool(spec.filter),
                )
            except Exception:  # noqa: BLE001
                logger.debug("attach_context failed in bulk_vertices_sync", exc_info=True)
            raise

    async def abulk_vertices(
        self,
        spec: BulkVerticesSpec,
        **kwargs: Any,
    ) -> BulkVerticesResult:
        """
        Bulk vertex scan (async).
        """
        ctx = self._build_ctx(**kwargs)
        try:
            return await self._graph.bulk_vertices(spec, ctx=ctx)
        except Exception as exc:  # noqa: BLE001
            try:
                attach_context(
                    exc,
                    framework="langchain",
                    component="graph",
                    operation="bulk_vertices_async",
                    namespace=spec.namespace,
                    limit=spec.limit,
                    has_filter=bool(spec.filter),
                )
            except Exception:  # noqa: BLE001
                logger.debug("attach_context failed in bulk_vertices_async", exc_info=True)
            raise

    def batch(
        self,
        ops: List[BatchOperation],
        **kwargs: Any,
    ) -> BatchResult:
        """
        Batch execution of multiple graph operations (sync).
        """
        ctx = self._build_ctx(**kwargs)
        try:
            return AsyncBridge.run_async(self._graph.batch(ops, ctx=ctx))
        except Exception as exc:  # noqa: BLE001
            try:
                attach_context(
                    exc,
                    framework="langchain",
                    component="graph",
                    operation="batch_sync",
                    ops_count=len(ops),
                )
            except Exception:  # noqa: BLE001
                logger.debug("attach_context failed in batch_sync", exc_info=True)
            raise

    async def abatch(
        self,
        ops: List[BatchOperation],
        **kwargs: Any,
    ) -> BatchResult:
        """
        Batch execution of multiple graph operations (async).
        """
        ctx = self._build_ctx(**kwargs)
        try:
            return await self._graph.batch(ops, ctx=ctx)
        except Exception as exc:  # noqa: BLE001
            try:
                attach_context(
                    exc,
                    framework="langchain",
                    component="graph",
                    operation="batch_async",
                    ops_count=len(ops),
                )
            except Exception:  # noqa: BLE001
                logger.debug("attach_context failed in batch_async", exc_info=True)
            raise

    def get_schema(
        self,
        **kwargs: Any,
    ) -> GraphSchema:
        """
        Schema introspection (sync).
        """
        ctx = self._build_ctx(**kwargs)
        try:
            return AsyncBridge.run_async(self._graph.get_schema(ctx=ctx))
        except Exception as exc:  # noqa: BLE001
            try:
                attach_context(
                    exc,
                    framework="langchain",
                    component="graph",
                    operation="get_schema_sync",
                )
            except Exception:  # noqa: BLE001
                logger.debug("attach_context failed in get_schema_sync", exc_info=True)
            raise

    async def aget_schema(
        self,
        **kwargs: Any,
    ) -> GraphSchema:
        """
        Schema introspection (async).
        """
        ctx = self._build_ctx(**kwargs)
        try:
            return await self._graph.get_schema(ctx=ctx)
        except Exception as exc:  # noqa: BLE001
            try:
                attach_context(
                    exc,
                    framework="langchain",
                    component="graph",
                    operation="get_schema_async",
                )
            except Exception:  # noqa: BLE001
                logger.debug("attach_context failed in get_schema_async", exc_info=True)
            raise

    def health(
        self,
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        """
        Health check (sync).
        """
        ctx = self._build_ctx(**kwargs)
        try:
            return AsyncBridge.run_async(self._graph.health(ctx=ctx))
        except Exception as exc:  # noqa: BLE001
            try:
                attach_context(
                    exc,
                    framework="langchain",
                    component="graph",
                    operation="health_sync",
                )
            except Exception:  # noqa: BLE001
                logger.debug("attach_context failed in health_sync", exc_info=True)
            raise

    async def ahealth(
        self,
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        """
        Health check (async).
        """
        ctx = self._build_ctx(**kwargs)
        try:
            return await self._graph.health(ctx=ctx)
        except Exception as exc:  # noqa: BLE001
            try:
                attach_context(
                    exc,
                    framework="langchain",
                    component="graph",
                    operation="health_async",
                )
            except Exception:  # noqa: BLE001
                logger.debug("attach_context failed in health_async", exc_info=True)
            raise


__all__ = [
    "CorpusLangChainGraphClient",
]
