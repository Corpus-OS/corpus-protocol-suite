# corpus_sdk/graph/framework_adapters/crewai.py
# SPDX-License-Identifier: Apache-2.0

"""
CrewAI adapter for Corpus Graph protocol.

This module exposes a Corpus `GraphProtocolV1` implementation as a
CrewAI-friendly client, with:

- Sync + async query APIs
- Streaming query support (async + sync via SyncStreamBridge)
- Proper integration with Corpus GraphProtocolV1
- OperationContext propagation derived from CrewAI tasks
- Error-context enrichment for observability and debugging

Design philosophy
-----------------
- Protocol-first: CrewAI is a thin skin over the Corpus graph adapter.
- All heavy lifting (deadlines, breakers, rate limits, caching) stays in
  the underlying `BaseGraphAdapter` / GraphProtocolV1 implementation.
- This layer focuses on:
    * Translating CrewAI `Task` instances → OperationContext
    * Building GraphQuerySpec via GraphTranslator
    * Bridging async Corpus APIs into sync calls via AsyncBridge
    * Providing safe streaming utilities for sync callers

Usage (example)
---------------

    from corpus_sdk.graph.graph_base import BaseGraphAdapter
    from corpus_sdk.graph.framework_adapters.crewai import (
        CorpusCrewAIGraphClient,
    )

    graph_adapter: BaseGraphAdapter = ...
    client = CorpusCrewAIGraphClient(
        graph_adapter=graph_adapter,
        default_dialect="cypher",
        default_namespace="my-graph",
    )

    # Within a CrewAI Task/tool:

    # Sync query
    result = client.query(
        "MATCH (n) RETURN n LIMIT 5",
        task=task,  # CrewAI Task instance
    )

    # Async query (if your tool is async)
    result = await client.aquery(
        "MATCH (n) RETURN n LIMIT 5",
        task=task,
    )

    # Sync streaming query
    for chunk in client.stream_query(
        "MATCH (n) RETURN n LIMIT 100",
        task=task,
    ):
        process(chunk.records)

    # Async streaming query
    async for chunk in client.astream_query(
        "MATCH (n) RETURN n LIMIT 100",
        task=task,
    ):
        process(chunk.records)
"""

from __future__ import annotations

import logging
from typing import (
    Any,
    AsyncIterator,
    Iterator,
    List,
    Mapping,
    Optional,
)

from corpus_sdk.core.context_translation import (
    from_crewai as context_from_crewai,
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


class CorpusCrewAIGraphClient:
    """
    CrewAI-oriented client wrapper around a Corpus `GraphProtocolV1`.

    This is a thin integration layer that:

    - Translates CrewAI `Task` instances into a Corpus `OperationContext`
      using `GraphTranslator.from_crewai` or, as a fallback,
      `context_translation.from_crewai`.
    - Uses `GraphTranslator` to build `GraphQuerySpec` from simple
      parameters (query string, dialect, params, namespace, timeout).
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
      methods into synchronous calls, mirroring the patterns used for
      LangChain and LlamaIndex integration.
    - Attaches rich error context (framework, component, operation, etc.)
      to every failure path.

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

        # GraphTranslator centralizes GraphQuerySpec construction and
        # framework-aware context translation semantics.
        self._translator = GraphTranslator(
            default_dialect=default_dialect,
            default_namespace=default_namespace,
            framework="crewai",
        )

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _build_ctx(self, **kwargs: Any):
        """
        Build an OperationContext from CrewAI-style inputs.

        Expected kwargs:
            - task: CrewAI Task instance (optional)
            - extra_context: Optional mapping for additional attrs (optional)
        """
        task = kwargs.get("task")
        extra_context = kwargs.get("extra_context") or {}

        if task is None and not extra_context:
            return None

        # Preferred path: use GraphTranslator's framework-specific helper.
        try:
            if extra_context:
                return self._translator.from_crewai(task, **extra_context)
            return self._translator.from_crewai(task)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "GraphTranslator.from_crewai failed; falling back to "
                "core context_from_crewai: %s",
                exc,
            )

        # Fallback: direct use of core context_translation.
        try:
            if extra_context:
                return context_from_crewai(task, **extra_context)
            return context_from_crewai(task)
        except Exception as exc:  # noqa: BLE001
            logger.debug("context_from_crewai failed: %s", exc)
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
            # Normalize to BadRequest with attached context.
            attach_context(
                exc,
                framework="crewai",
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
        Execute a non-streaming graph query (sync) from CrewAI code.

        This bridges the async `graph_adapter.query` method into a
        synchronous call using AsyncBridge.
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
            try:
                attach_context(
                    exc,
                    framework="crewai",
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
        Execute a non-streaming graph query (async) from CrewAI code.
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
                    framework="crewai",
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
        streaming call into a synchronous iterator, preserving:
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
                        framework="crewai",
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
            framework="crewai",
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
                        framework="crewai",
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

        # Return the async generator directly so callers can `async for` on it.
        return _inner()

    # --------------------------------------------------------------------- #
    # Write / mutation API (sync + async)
    # These accept GraphProtocol spec objects directly.
    # --------------------------------------------------------------------- #

    def upsert_nodes(
        self,
        spec: UpsertNodesSpec,
        **kwargs: Any,
    ) -> UpsertResult:
        """
        Batch upsert nodes (sync).
        """
        ctx = self._build_ctx(**kwargs)
        try:
            return AsyncBridge.run_async(self._graph.upsert_nodes(spec, ctx=ctx))
        except Exception as exc:  # noqa: BLE001
            try:
                attach_context(
                    exc,
                    framework="crewai",
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
    ) -> UpsertResult:
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
                    framework="crewai",
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
    ) -> UpsertResult:
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
                    framework="crewai",
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
    ) -> UpsertResult:
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
                    framework="crewai",
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
    ) -> DeleteResult:
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
                    framework="crewai",
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
    ) -> DeleteResult:
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
                    framework="crewai",
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
    ) -> DeleteResult:
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
                    framework="crewai",
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
    ) -> DeleteResult:
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
                    framework="crewai",
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
                    framework="crewai",
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
                    framework="crewai",
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
                    framework="crewai",
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
                    framework="crewai",
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
                    framework="crewai",
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
                    framework="crewai",
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
                    framework="crewai",
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
                    framework="crewai",
                    component="graph",
                    operation="health_async",
                )
            except Exception:  # noqa: BLE001
                logger.debug("attach_context failed in health_async", exc_info=True)
            raise


__all__ = [
    "CorpusCrewAIGraphClient",
]
