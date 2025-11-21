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
from functools import cached_property
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
)

from corpus_sdk.core.context_translation import (
    from_autogen as core_ctx_from_autogen,
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
            op_ctx: OperationContext,  # noqa: ARG002
            framework_ctx: Optional[Any] = None,  # noqa: ARG002
        ) -> QueryResult:
            return result

        def translate_query_chunk(
            self,
            chunk: QueryChunk,
            *,
            op_ctx: OperationContext,  # noqa: ARG002
            framework_ctx: Optional[Any] = None,  # noqa: ARG002
        ) -> QueryChunk:
            return chunk

        def translate_bulk_vertices_result(
            self,
            result: BulkVerticesResult,
            *,
            op_ctx: OperationContext,  # noqa: ARG002
            framework_ctx: Optional[Any] = None,  # noqa: ARG002
        ) -> BulkVerticesResult:
            return result

        def translate_batch_result(
            self,
            result: BatchResult,
            *,
            op_ctx: OperationContext,  # noqa: ARG002
            framework_ctx: Optional[Any] = None,  # noqa: ARG002
        ) -> BatchResult:
            return result

        def translate_schema(
            self,
            schema: GraphSchema,
            *,
            op_ctx: OperationContext,  # noqa: ARG002
            framework_ctx: Optional[Any] = None,  # noqa: ARG002
        ) -> GraphSchema:
            return schema

    def __init__(
        self,
        *,
        graph_adapter: GraphProtocolV1,
        default_dialect: Optional[str] = None,
        default_namespace: Optional[str] = None,
        framework_version: Optional[str] = None,
    ) -> None:
        self._graph: GraphProtocolV1 = graph_adapter
        self._default_dialect: Optional[str] = default_dialect
        self._default_namespace: Optional[str] = default_namespace
        self._framework_version: Optional[str] = framework_version

    # ------------------------------------------------------------------ #
    # Translator (lazy, cached) – mirrors embedding adapter pattern
    # ------------------------------------------------------------------ #

    @cached_property
    def _translator(self) -> GraphTranslator:
        """
        Lazily construct and cache the `GraphTranslator`.

        Uses `cached_property` for thread safety and performance, mirroring
        the embedding adapter pattern.
        """
        framework_translator = self._AutoGenGraphFrameworkTranslator()
        return GraphTranslator(
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
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="autogen",
                operation="context_translation",
            )
            raise

        if not isinstance(ctx, OperationContext):
            raise BadRequest(
                f"from_autogen produced unsupported context type: {type(ctx).__name__}",
                code="BAD_OPERATION_CONTEXT",
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

    def capabilities(self) -> Dict[str, Any]:
        """
        Sync wrapper around `graph_adapter.capabilities()`.

        Assumes the adapter exposes a sync `capabilities()` method.
        Returns the dataclass as a plain dict for AutoGen-friendly consumption.
        """
        try:
            caps = self._graph.capabilities()
            if not isinstance(caps, GraphCapabilities):
                raise BadRequest(
                    f"graph_adapter.capabilities returned unsupported type: {type(caps).__name__}",
                    code="BAD_ADAPTER_RESULT",
                )
            return self._capabilities_to_dict(caps)
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="autogen",
                operation="capabilities_sync",
            )
            raise

    async def acapabilities(self) -> Dict[str, Any]:
        """
        Async capabilities accessor with AutoGen-friendly dict output.

        Assumes the adapter exposes an async `capabilities()` method.
        """
        try:
            caps = await self._graph.capabilities()
            if not isinstance(caps, GraphCapabilities):
                raise BadRequest(
                    f"graph_adapter.capabilities returned unsupported type: {type(caps).__name__}",
                    code="BAD_ADAPTER_RESULT",
                )
            return self._capabilities_to_dict(caps)
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="autogen",
                operation="capabilities_async",
            )
            raise

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
        try:
            schema = self._translator.get_schema(
                op_ctx=ctx,
                framework_ctx={},
            )
            if not isinstance(schema, GraphSchema):
                raise BadRequest(
                    f"GraphTranslator.get_schema returned unsupported type: {type(schema).__name__}",
                    code="BAD_TRANSLATED_SCHEMA",
                )
            return schema
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="autogen",
                operation="get_schema_sync",
            )
            raise

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
        try:
            schema = await self._translator.arun_get_schema(
                op_ctx=ctx,
                framework_ctx={},
            )
            if not isinstance(schema, GraphSchema):
                raise BadRequest(
                    f"GraphTranslator.arun_get_schema returned unsupported type: {type(schema).__name__}",
                    code="BAD_TRANSLATED_SCHEMA",
                )
            return schema
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="autogen",
                operation="get_schema_async",
            )
            raise

    def health(
        self,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Sync health check wrapper.

        Assumes the adapter exposes a sync `health(ctx=...)` method.
        Returns the normalized health mapping from the underlying adapter.
        """
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        try:
            health_result = self._graph.health(ctx=ctx)
            if not isinstance(health_result, Mapping):
                raise BadRequest(
                    f"graph_adapter.health returned unsupported type: {type(health_result).__name__}",
                    code="BAD_HEALTH_RESULT",
                )
            return dict(health_result)
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="autogen",
                operation="health_sync",
            )
            raise

    async def ahealth(
        self,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Async health check wrapper.

        Assumes the adapter exposes an async `health(ctx=...)` method.
        """
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        try:
            health_result = await self._graph.health(ctx=ctx)
            if not isinstance(health_result, Mapping):
                raise BadRequest(
                    f"graph_adapter.health returned unsupported type: {type(health_result).__name__}",
                    code="BAD_HEALTH_RESULT",
                )
            return dict(health_result)
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="autogen",
                operation="health_async",
            )
            raise

    # ------------------------------------------------------------------ #
    # Query (sync + async)
    # ------------------------------------------------------------------ #

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

        try:
            result = self._translator.query(
                raw_query,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
                mmr_config=None,
            )
            if not isinstance(result, QueryResult):
                raise BadRequest(
                    f"GraphTranslator.query returned unsupported type: {type(result).__name__}",
                    code="BAD_TRANSLATED_RESULT",
                )
            return result
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="autogen",
                operation="query_sync",
                query=query,
                dialect=dialect or self._default_dialect,
                namespace=namespace or self._default_namespace,
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

        try:
            result = await self._translator.arun_query(
                raw_query,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
                mmr_config=None,
            )
            if not isinstance(result, QueryResult):
                raise BadRequest(
                    f"GraphTranslator.arun_query returned unsupported type: {type(result).__name__}",
                    code="BAD_TRANSLATED_RESULT",
                )
            return result
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="autogen",
                operation="query_async",
                query=query,
                dialect=dialect or self._default_dialect,
                namespace=namespace or self._default_namespace,
            )
            raise

    # ------------------------------------------------------------------ #
    # Streaming query (sync + async)
    # ------------------------------------------------------------------ #

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

        try:
            for chunk in self._translator.query_stream(
                raw_query,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            ):
                if not isinstance(chunk, QueryChunk):
                    raise BadRequest(
                        f"GraphTranslator.query_stream yielded unsupported type: {type(chunk).__name__}",
                        code="BAD_TRANSLATED_CHUNK",
                    )
                yield chunk
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="autogen",
                operation="stream_query_sync",
                query=query,
                dialect=dialect or self._default_dialect,
                namespace=namespace or self._default_namespace,
            )
            raise

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

        try:
            async for chunk in self._translator.arun_query_stream(
                raw_query,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            ):
                if not isinstance(chunk, QueryChunk):
                    raise BadRequest(
                        f"GraphTranslator.arun_query_stream yielded unsupported type: {type(chunk).__name__}",
                        code="BAD_TRANSLATED_CHUNK",
                    )
                yield chunk
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="autogen",
                operation="stream_query_async",
                query=query,
                dialect=dialect or self._default_dialect,
                namespace=namespace or self._default_namespace,
            )
            raise

    # ------------------------------------------------------------------ #
    # Upsert nodes / edges (sync + async)
    # ------------------------------------------------------------------ #

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
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        framework_ctx = self._framework_ctx_for_namespace(getattr(spec, "namespace", None))

        try:
            result = self._translator.upsert_nodes(
                spec.nodes,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            if not isinstance(result, UpsertResult):
                raise BadRequest(
                    f"GraphTranslator.upsert_nodes returned unsupported type: {type(result).__name__}",
                    code="BAD_UPSERT_RESULT",
                )
            return result
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="autogen",
                operation="upsert_nodes_sync",
                namespace=getattr(spec, "namespace", None),
                count=len(spec.nodes),
            )
            raise

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
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        framework_ctx = self._framework_ctx_for_namespace(getattr(spec, "namespace", None))

        try:
            result = await self._translator.arun_upsert_nodes(
                spec.nodes,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            if not isinstance(result, UpsertResult):
                raise BadRequest(
                    f"GraphTranslator.arun_upsert_nodes returned unsupported type: {type(result).__name__}",
                    code="BAD_UPSERT_RESULT",
                )
            return result
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="autogen",
                operation="upsert_nodes_async",
                namespace=getattr(spec, "namespace", None),
                count=len(spec.nodes),
            )
            raise

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
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        framework_ctx = self._framework_ctx_for_namespace(getattr(spec, "namespace", None))

        try:
            result = self._translator.upsert_edges(
                spec.edges,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            if not isinstance(result, UpsertResult):
                raise BadRequest(
                    f"GraphTranslator.upsert_edges returned unsupported type: {type(result).__name__}",
                    code="BAD_UPSERT_RESULT",
                )
            return result
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="autogen",
                operation="upsert_edges_sync",
                namespace=getattr(spec, "namespace", None),
                count=len(spec.edges),
            )
            raise

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
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        framework_ctx = self._framework_ctx_for_namespace(getattr(spec, "namespace", None))

        try:
            result = await self._translator.arun_upsert_edges(
                spec.edges,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            if not isinstance(result, UpsertResult):
                raise BadRequest(
                    f"GraphTranslator.arun_upsert_edges returned unsupported type: {type(result).__name__}",
                    code="BAD_UPSERT_RESULT",
                )
            return result
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="autogen",
                operation="upsert_edges_async",
                namespace=getattr(spec, "namespace", None),
                count=len(spec.edges),
            )
            raise

    # ------------------------------------------------------------------ #
    # Delete nodes / edges (sync + async)
    # ------------------------------------------------------------------ #

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
        framework_ctx = self._framework_ctx_for_namespace(getattr(spec, "namespace", None))

        if spec.filter is not None:
            raw_filter_or_ids: Any = spec.filter
        else:
            raw_filter_or_ids = list(spec.ids or [])

        try:
            result = self._translator.delete_nodes(
                raw_filter_or_ids,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            if not isinstance(result, DeleteResult):
                raise BadRequest(
                    f"GraphTranslator.delete_nodes returned unsupported type: {type(result).__name__}",
                    code="BAD_DELETE_RESULT",
                )
            return result
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="autogen",
                operation="delete_nodes_sync",
                namespace=getattr(spec, "namespace", None),
                ids_count=len(spec.ids or []),
            )
            raise

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
        framework_ctx = self._framework_ctx_for_namespace(getattr(spec, "namespace", None))

        if spec.filter is not None:
            raw_filter_or_ids: Any = spec.filter
        else:
            raw_filter_or_ids = list(spec.ids or [])

        try:
            result = await self._translator.arun_delete_nodes(
                raw_filter_or_ids,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            if not isinstance(result, DeleteResult):
                raise BadRequest(
                    f"GraphTranslator.arun_delete_nodes returned unsupported type: {type(result).__name__}",
                    code="BAD_DELETE_RESULT",
                )
            return result
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="autogen",
                operation="delete_nodes_async",
                namespace=getattr(spec, "namespace", None),
                ids_count=len(spec.ids or []),
            )
            raise

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
        framework_ctx = self._framework_ctx_for_namespace(getattr(spec, "namespace", None))

        if spec.filter is not None:
            raw_filter_or_ids: Any = spec.filter
        else:
            raw_filter_or_ids = list(spec.ids or [])

        try:
            result = self._translator.delete_edges(
                raw_filter_or_ids,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            if not isinstance(result, DeleteResult):
                raise BadRequest(
                    f"GraphTranslator.delete_edges returned unsupported type: {type(result).__name__}",
                    code="BAD_DELETE_RESULT",
                )
            return result
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="autogen",
                operation="delete_edges_sync",
                namespace=getattr(spec, "namespace", None),
                ids_count=len(spec.ids or []),
            )
            raise

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
        framework_ctx = self._framework_ctx_for_namespace(getattr(spec, "namespace", None))

        if spec.filter is not None:
            raw_filter_or_ids: Any = spec.filter
        else:
            raw_filter_or_ids = list(spec.ids or [])

        try:
            result = await self._translator.arun_delete_edges(
                raw_filter_or_ids,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            if not isinstance(result, DeleteResult):
                raise BadRequest(
                    f"GraphTranslator.arun_delete_edges returned unsupported type: {type(result).__name__}",
                    code="BAD_DELETE_RESULT",
                )
            return result
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="autogen",
                operation="delete_edges_async",
                namespace=getattr(spec, "namespace", None),
                ids_count=len(spec.ids or []),
            )
            raise

    # ------------------------------------------------------------------ #
    # Bulk vertices (sync + async)
    # ------------------------------------------------------------------ #

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

        try:
            result = self._translator.bulk_vertices(
                raw_request,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            if not isinstance(result, BulkVerticesResult):
                raise BadRequest(
                    f"GraphTranslator.bulk_vertices returned unsupported type: {type(result).__name__}",
                    code="BAD_BULK_VERTICES_RESULT",
                )
            return result
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="autogen",
                operation="bulk_vertices_sync",
                namespace=getattr(spec, "namespace", None),
                limit=spec.limit,
            )
            raise

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

        try:
            result = await self._translator.arun_bulk_vertices(
                raw_request,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            if not isinstance(result, BulkVerticesResult):
                raise BadRequest(
                    f"GraphTranslator.arun_bulk_vertices returned unsupported type: {type(result).__name__}",
                    code="BAD_BULK_VERTICES_RESULT",
                )
            return result
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="autogen",
                operation="bulk_vertices_async",
                namespace=getattr(spec, "namespace", None),
                limit=spec.limit,
            )
            raise

    # ------------------------------------------------------------------ #
    # Batch (sync + async)
    # ------------------------------------------------------------------ #

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
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)

        raw_batch_ops: List[Mapping[str, Any]] = [
            {"op": op.op, "args": dict(op.args or {})} for op in ops
        ]

        try:
            result = self._translator.batch(
                raw_batch_ops,
                op_ctx=ctx,
                framework_ctx={},
            )
            if not isinstance(result, BatchResult):
                raise BadRequest(
                    f"GraphTranslator.batch returned unsupported type: {type(result).__name__}",
                    code="BAD_BATCH_RESULT",
                )
            return result
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="autogen",
                operation="batch_sync",
                ops_count=len(ops),
            )
            raise

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
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)

        raw_batch_ops: List[Mapping[str, Any]] = [
            {"op": op.op, "args": dict(op.args or {})} for op in ops
        ]

        try:
            result = await self._translator.arun_batch(
                raw_batch_ops,
                op_ctx=ctx,
                framework_ctx={},
            )
            if not isinstance(result, BatchResult):
                raise BadRequest(
                    f"GraphTranslator.arun_batch returned unsupported type: {type(result).__name__}",
                    code="BAD_BATCH_RESULT",
                )
            return result
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="autogen",
                operation="batch_async",
                ops_count=len(ops),
            )
            raise


__all__ = [
    "CorpusAutoGenGraphClient",
]
