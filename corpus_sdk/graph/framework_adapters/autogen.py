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

import json
import logging
import asyncio
from functools import cached_property
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    TypeVar,
)

from corpus_sdk.core.context_translation import (
    from_autogen as core_ctx_from_autogen,
)
from corpus_sdk.core.error_context import attach_context
from corpus_sdk.graph.framework_adapters.common.graph_translation import (
    DefaultGraphFrameworkTranslator,
    GraphTranslator,
    GraphFrameworkTranslator,
    create_graph_translator,
)
from corpus_sdk.graph.framework_adapters.common.framework_utils import (
    create_graph_error_context_decorator,
    graph_capabilities_to_dict,
    validate_batch_operations,
    validate_graph_query,
    validate_upsert_nodes_spec,
    validate_graph_result_type,
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
    GraphTraversalSpec,
    NotSupported,
    OperationContext,
    QueryChunk,
    QueryResult,
    TraversalResult,
    UpsertEdgesSpec,
    UpsertNodesSpec,
    UpsertResult,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")


# ---------------------------------------------------------------------------
# Error code constants (standalone, NOT subclassing shared coercion enums)
# ---------------------------------------------------------------------------


class ErrorCodes:
    """
    Error code constants for AutoGen graph adapter.

    This intentionally does not subclass shared coercion enums. Shared
    utilities that rely on symbolic names can still refer to this class
    explicitly while keeping cross-framework alignment.
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
    BAD_TRAVERSAL_RESULT = "BAD_TRAVERSAL_RESULT"
    BAD_TRANSACTION_RESULT = "BAD_TRANSACTION_RESULT"


# ---------------------------------------------------------------------------
# Error-context decorators (via common framework utils)
# ---------------------------------------------------------------------------


def with_graph_error_context(
    operation: str,
    **static_context: Any,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for sync methods with graph error-context attachment.

    Thin wrapper over the shared `create_graph_error_context_decorator`
    for the AutoGen framework.
    """
    return create_graph_error_context_decorator(
        framework="autogen",
        is_async=False,
    )(operation=operation, **static_context)


def with_async_graph_error_context(
    operation: str,
    **static_context: Any,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for async methods with graph error-context attachment.

    Thin wrapper over the shared `create_graph_error_context_decorator`
    for the AutoGen framework.
    """
    return create_graph_error_context_decorator(
        framework="autogen",
        is_async=True,
    )(operation=operation, **static_context)


# Backwards-compatible aliases (if callers were using old names)
with_error_context = with_graph_error_context
with_async_error_context = with_async_graph_error_context


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _looks_like_operation_context(obj: Any) -> bool:
    """
    Heuristic check; OperationContext may be a Protocol/alias in some SDK versions.
    """
    if obj is None:
        return False

    # If OperationContext is a real class, this will work; if it's a Protocol,
    # this may raise TypeError in some typing modes.
    try:
        if isinstance(obj, OperationContext):
            return True
    except TypeError:
        pass

    # Fallback to structural check
    attrs = ("request_id", "traceparent", "tenant", "attrs", "to_dict")
    return any(hasattr(obj, attr) for attr in attrs)


def _ensure_not_in_event_loop(sync_api_name: str) -> None:
    """
    Prevent accidentally calling sync APIs from inside an active asyncio event loop.

    This mirrors the safety behavior used in the AutoGen embedding adapter to
    avoid subtle deadlocks and confusing behavior. Call this at the top of
    sync methods that also have async counterparts (query/aquery, etc.).
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop → safe to call sync API.
        return

    raise RuntimeError(
        f"{sync_api_name} was called from inside an active asyncio event loop. "
        f"Use the async variant instead (e.g. 'a{sync_api_name}')."
    )


# ---------------------------------------------------------------------------
# Public AutoGen framework translator
# ---------------------------------------------------------------------------


class AutoGenGraphFrameworkTranslator(DefaultGraphFrameworkTranslator):
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

    Exposed as a top-level class so users can easily subclass and override
    just a subset of behaviors, e.g.:

    ```python
    class MyTranslator(AutoGenGraphFrameworkTranslator):
        def translate_schema(...):
            schema = super().translate_schema(...)
            # tweak schema here
            return schema
    ```
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

    def translate_transaction_result(
        self,
        result: BatchResult,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Mapping[str, Any]] = None,
    ) -> BatchResult:
        return result

    def translate_traversal_result(
        self,
        result: TraversalResult,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Mapping[str, Any]] = None,
    ) -> TraversalResult:
        return result

    def translate_schema(
        self,
        schema: GraphSchema,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Mapping[str, Any]] = None,
    ) -> GraphSchema:
        return schema


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

    def capabilities(self, **kwargs) -> Dict[str, Any]:
        ...

    async def acapabilities(self, **kwargs) -> Dict[str, Any]:
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

    # Bulk / traversal / transaction / batch

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

    def traversal(
        self,
        spec: GraphTraversalSpec,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> TraversalResult:
        ...

    async def atraversal(
        self,
        spec: GraphTraversalSpec,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> TraversalResult:
        ...

    def transaction(
        self,
        ops: List[BatchOperation],
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BatchResult:
        ...

    async def atransaction(
        self,
        ops: List[BatchOperation],
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BatchResult:
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

    def __init__(
        self,
        adapter: Optional[GraphProtocolV1] = None,
        *,
        graph_adapter: Optional[GraphProtocolV1] = None,
        default_dialect: Optional[str] = None,
        default_namespace: Optional[str] = None,
        default_timeout_ms: Optional[int] = None,
        framework_version: Optional[str] = None,
        framework_translator: Optional[GraphFrameworkTranslator] = None,
    ) -> None:
        """
        Initialize an AutoGen-oriented graph client.

        Parameters
        ----------
        adapter:
            Underlying `GraphProtocolV1` implementation. This is the preferred
            parameter name for new callers.

        graph_adapter:
            Backwards-compatible alias for `adapter`. If both `adapter` and
            `graph_adapter` are provided, they must refer to the same object.

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
            Optional custom GraphFrameworkTranslator to use instead of the
            default AutoGen pass-through translator.
        """
        # Resolving adapter / graph_adapter with basic duck-typed validation.
        if graph_adapter is not None and adapter is not None and graph_adapter is not adapter:
            raise TypeError("Provide only one of 'adapter' or 'graph_adapter', not both")

        resolved_adapter: Any = graph_adapter if graph_adapter is not None else adapter
        if resolved_adapter is None:
            raise TypeError("adapter must be a GraphProtocolV1-compatible graph adapter")

        # Minimal duck-type check: we expect a GraphProtocolV1-like surface with
        # capabilities() and query(...) methods. This keeps tests happy even
        # when using simple mock objects instead of real adapters.
        if not hasattr(resolved_adapter, "query") or not hasattr(resolved_adapter, "capabilities"):
            raise TypeError(
                "adapter must implement GraphProtocolV1-like interface with "
                "'query' and 'capabilities' methods"
            )

        self._graph: GraphProtocolV1 = resolved_adapter  # type: ignore[assignment]
        self._default_dialect: Optional[str] = default_dialect
        self._default_namespace: Optional[str] = default_namespace
        self._default_timeout_ms: Optional[int] = default_timeout_ms
        self._framework_version: Optional[str] = framework_version
        self._framework_translator_override: Optional[
            GraphFrameworkTranslator
        ] = framework_translator

        # Resource management flags (idempotent close semantics)
        self._closed: bool = False
        self._aclosed: bool = False

    # ------------------------------------------------------------------ #
    # Resource management (context managers)
    # ------------------------------------------------------------------ #

    def __enter__(self) -> CorpusAutoGenGraphClient:
        """Support context manager protocol for resource cleanup."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Clean up resources when exiting context."""
        # Best-effort sync close; idempotent.
        self.close()

    async def __aenter__(self) -> CorpusAutoGenGraphClient:
        """Support async context manager protocol."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Clean up resources when exiting async context."""
        await self.aclose()

    def close(self) -> None:
        """
        Close the underlying graph adapter if it exposes a `close()` method.

        This is safe to call multiple times; subsequent calls are no-ops.
        """
        if self._closed:
            return
        self._closed = True

        close_fn = getattr(self._graph, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                # Never let cleanup failures propagate to callers.
                logger.debug("Failed to close graph adapter", exc_info=True)

    async def aclose(self) -> None:
        """
        Async close for the underlying graph adapter.

        Prefers an async `aclose()` method when available, otherwise falls back
        to the sync `close()` method.
        """
        if self._aclosed:
            return
        self._aclosed = True

        aclose_fn = getattr(self._graph, "aclose", None)
        if callable(aclose_fn):
            try:
                await aclose_fn()
                # If async close succeeded, we can consider sync-close satisfied.
                self._closed = True
                return
            except Exception:
                logger.debug("Failed to async-close graph adapter", exc_info=True)

        # Fallback to sync close if we haven't already done so.
        if not self._closed:
            self.close()

    # ------------------------------------------------------------------ #
    # Translator (lazy, cached) – thin wrapper via create_graph_translator
    # ------------------------------------------------------------------ #

    @cached_property
    def _translator(self) -> GraphTranslator:
        """
        Lazily construct and cache the `GraphTranslator`.

        Uses `create_graph_translator` so registry-based per-framework
        translators remain honored while still allowing our AutoGen-specific
        pass-through translator (or a caller-provided override) to be supplied
        explicitly.
        """
        framework_translator: GraphFrameworkTranslator = (
            self._framework_translator_override
            or AutoGenGraphFrameworkTranslator()
        )
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

        Context translation is *best-effort*: failures are logged and attached
        for observability, but graph operations must still be able to proceed
        without an OperationContext.

        When a valid OperationContext is produced, its attrs are enriched with
        framework metadata so downstream layers can reliably identify the
        source of the call:
            - framework="autogen"
            - framework_version=<self._framework_version> (if set)
        """
        extra: Dict[str, Any] = dict(extra_context or {})

        if conversation is None and not extra:
            return None

        try:
            ctx_candidate = core_ctx_from_autogen(
                conversation,
                framework_version=self._framework_version,
                **extra,
            )
        except Exception as exc:
            logger.warning(
                "[%s] Failed to build OperationContext from AutoGen inputs; "
                "proceeding without OperationContext. conversation_type=%s extra_keys=%s",
                ErrorCodes.BAD_OPERATION_CONTEXT,
                type(conversation).__name__ if conversation is not None else "None",
                list(extra.keys()),
            )
            attach_context(
                exc,
                framework="autogen",
                operation="context_translation",
                error_code=ErrorCodes.BAD_OPERATION_CONTEXT,
                conversation_type=type(conversation).__name__ if conversation is not None else "None",
                extra_context_keys=list(extra.keys()),
            )
            return None

        if not _looks_like_operation_context(ctx_candidate):
            logger.warning(
                "[%s] from_autogen returned non-OperationContext-like type: %s. "
                "Ignoring OperationContext.",
                ErrorCodes.BAD_OPERATION_CONTEXT,
                type(ctx_candidate).__name__,
            )
            return None

        # Enrich attrs with framework metadata in-place (best-effort)
        try:
            attrs = getattr(ctx_candidate, "attrs", None) or {}
            if not isinstance(attrs, dict):
                attrs = dict(attrs)
            attrs.setdefault("framework", "autogen")
            if self._framework_version is not None:
                attrs.setdefault("framework_version", self._framework_version)
            setattr(ctx_candidate, "attrs", attrs)
        except Exception:
            logger.debug(
                "Failed to enrich OperationContext attrs for AutoGen context",
                exc_info=True,
            )

        return ctx_candidate  # type: ignore[return-value]

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

    def _framework_ctx(
        self,
        *,
        operation: str,
        namespace: Optional[str] = None,
    ) -> Mapping[str, Any]:
        """
        Build a framework_ctx mapping for GraphTranslator with basic
        observability hints and the effective namespace.
        """
        ctx: Dict[str, Any] = {
            "framework": "autogen",
            "operation": operation,
        }

        if self._framework_version is not None:
            ctx["framework_version"] = self._framework_version

        effective_namespace = namespace or self._default_namespace
        if effective_namespace is not None:
            ctx["namespace"] = effective_namespace

        return ctx

    def _validate_upsert_edges_spec(self, spec: UpsertEdgesSpec) -> None:
        """
        AutoGen-local validation for edge upsert specs.

        (We still use shared validation helpers for node specs and batch ops.)
        """
        # Basic structural checks.
        if spec.edges is None:
            raise BadRequest("UpsertEdgesSpec.edges must not be None")

        try:
            edges = list(spec.edges)
        except TypeError as exc:
            raise BadRequest(
                "UpsertEdgesSpec.edges must be an iterable of edges",
            ) from exc

        if not edges:
            raise BadRequest("UpsertEdgesSpec must contain at least one edge")

        for idx, edge in enumerate(edges):
            if not hasattr(edge, "id") or not edge.id:
                raise BadRequest(f"Edge at index {idx} must have an ID")
            if not hasattr(edge, "src") or not edge.src:
                raise BadRequest(f"Edge at index {idx} must have source node ID")
            if not hasattr(edge, "dst") or not edge.dst:
                raise BadRequest(f"Edge at index {idx} must have target node ID")
            if not hasattr(edge, "label") or not edge.label:
                raise BadRequest(f"Edge at index {idx} must have a label")

            # Validate properties are JSON-serializable
            if hasattr(edge, "properties") and edge.properties is not None:
                try:
                    json.dumps(edge.properties)
                except (TypeError, ValueError) as e:
                    raise BadRequest(
                        f"Edge at index {idx} properties must be JSON-serializable: {e}"
                    )

        # Update spec.edges to validated list
        spec.edges = edges  # type: ignore[assignment]

    def _validate_query_params(
        self,
        params: Optional[Mapping[str, Any]],
    ) -> None:
        """
        Lightweight validation for query parameter mappings.

        Keeps the adapter behavior protocol-friendly while catching obvious
        misuse (like passing a bare string instead of a dict).
        """
        if params is not None and not isinstance(params, Mapping):
            raise TypeError(
                f"params must be a mapping (e.g. dict), not {type(params).__name__}"
            )

    # ------------------------------------------------------------------ #
    # Capabilities / schema / health
    # ------------------------------------------------------------------ #

    @with_graph_error_context("capabilities_sync")
    def capabilities(self, **kwargs) -> Dict[str, Any]:
        """
        Sync wrapper around `graph_adapter.capabilities()`.

        Uses GraphTranslator for consistency with other operations and
        normalizes to an AutoGen-friendly dict.
        """
        _ensure_not_in_event_loop("capabilities")

        caps = self._translator.capabilities()
        return graph_capabilities_to_dict(caps)

    @with_async_graph_error_context("capabilities_async")
    async def acapabilities(self, **kwargs) -> Dict[str, Any]:
        """
        Async capabilities accessor with AutoGen-friendly dict output.

        Uses GraphTranslator for consistency with other operations.
        """
        caps = await self._translator.arun_capabilities()
        return graph_capabilities_to_dict(caps)

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
        _ensure_not_in_event_loop("get_schema")

        ctx = self._build_ctx(
            conversation=conversation,
            extra_context=extra_context,
        )
        schema = self._translator.get_schema(
            op_ctx=ctx,
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
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> GraphSchema:
        """
        Async wrapper around `graph_adapter.get_schema(...)`.

        Delegates to GraphTranslator.
        """
        ctx = self._build_ctx(
            conversation=conversation,
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
        _ensure_not_in_event_loop("health")

        ctx = self._build_ctx(
            conversation=conversation,
            extra_context=extra_context,
        )
        health_result = self._translator.health(
            op_ctx=ctx,
            framework_ctx=self._framework_ctx(operation="health"),
        )
        mapping_result = validate_graph_result_type(
            health_result,
            expected_type=Mapping,
            operation="GraphTranslator.health",
            error_code=ErrorCodes.BAD_HEALTH_RESULT,
        )
        # Normalize to a plain dict to honor the return type annotation.
        return dict(mapping_result)

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
        ctx = self._build_ctx(
            conversation=conversation,
            extra_context=extra_context,
        )
        health_result = await self._translator.arun_health(
            op_ctx=ctx,
            framework_ctx=self._framework_ctx(operation="health"),
        )
        mapping_result = validate_graph_result_type(
            health_result,
            expected_type=Mapping,
            operation="GraphTranslator.arun_health",
            error_code=ErrorCodes.BAD_HEALTH_RESULT,
        )
        # Normalize to a plain dict to honor the return type annotation.
        return dict(mapping_result)

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
        _ensure_not_in_event_loop("query")

        validate_graph_query(query, operation="query", error_code="INVALID_QUERY")
        self._validate_query_params(params)

        ctx = self._build_ctx(
            conversation=conversation,
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

        effective_ns = namespace or self._default_namespace
        logger.info(
            "AutoGen graph query: framework=autogen namespace=%s",
            effective_ns,
        )

        # Graceful handling of unsupported dialects:
        # if the adapter raises NotSupported, retry once without an explicit
        # dialect so that the adapter's native default can be used.
        try:
            result = self._translator.query(
                raw_query,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
                mmr_config=None,
            )
        except NotSupported:
            if dialect is not None:
                fallback_raw = dict(raw_query)
                fallback_raw.pop("dialect", None)
                result = self._translator.query(
                    fallback_raw,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                    mmr_config=None,
                )
            else:
                raise

        return validate_graph_result_type(
            result,
            expected_type=QueryResult,
            operation="GraphTranslator.query",
            error_code=ErrorCodes.BAD_TRANSLATED_RESULT,
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
        validate_graph_query(query, operation="aquery", error_code="INVALID_QUERY")
        self._validate_query_params(params)

        ctx = self._build_ctx(
            conversation=conversation,
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

        effective_ns = namespace or self._default_namespace
        logger.info(
            "AutoGen graph async query: framework=autogen namespace=%s",
            effective_ns,
        )

        try:
            result = await self._translator.arun_query(
                raw_query,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
                mmr_config=None,
            )
        except NotSupported:
            if dialect is not None:
                fallback_raw = dict(raw_query)
                fallback_raw.pop("dialect", None)
                result = await self._translator.arun_query(
                    fallback_raw,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                    mmr_config=None,
                )
            else:
                raise

        return validate_graph_result_type(
            result,
            expected_type=QueryResult,
            operation="GraphTranslator.arun_query",
            error_code=ErrorCodes.BAD_TRANSLATED_RESULT,
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
        _ensure_not_in_event_loop("stream_query")

        validate_graph_query(query, operation="stream_query", error_code="INVALID_QUERY")
        self._validate_query_params(params)

        ctx = self._build_ctx(
            conversation=conversation,
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

        effective_ns = namespace or self._default_namespace
        logger.info(
            "AutoGen graph stream_query: framework=autogen namespace=%s",
            effective_ns,
        )

        for chunk in self._translator.query_stream(
            raw_query,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        ):
            yield validate_graph_result_type(
                chunk,
                expected_type=QueryChunk,
                operation="GraphTranslator.query_stream",
                error_code=ErrorCodes.BAD_TRANSLATED_CHUNK,
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
        validate_graph_query(query, operation="astream_query", error_code="INVALID_QUERY")
        self._validate_query_params(params)

        ctx = self._build_ctx(
            conversation=conversation,
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

        effective_ns = namespace or self._default_namespace
        logger.info(
            "AutoGen graph astream_query: framework=autogen namespace=%s",
            effective_ns,
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
        _ensure_not_in_event_loop("upsert_nodes")

        validate_upsert_nodes_spec(spec)

        ctx = self._build_ctx(
            conversation=conversation,
            extra_context=extra_context,
        )
        framework_ctx = self._framework_ctx(
            operation="upsert_nodes",
            namespace=getattr(spec, "namespace", None),
        )

        result = self._translator.upsert_nodes(
            spec.nodes,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )
        return validate_graph_result_type(
            result,
            expected_type=UpsertResult,
            operation="GraphTranslator.upsert_nodes",
            error_code=ErrorCodes.BAD_UPSERT_RESULT,
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
        validate_upsert_nodes_spec(spec)

        ctx = self._build_ctx(
            conversation=conversation,
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
        _ensure_not_in_event_loop("upsert_edges")

        self._validate_upsert_edges_spec(spec)

        ctx = self._build_ctx(
            conversation=conversation,
            extra_context=extra_context,
        )
        framework_ctx = self._framework_ctx(
            operation="upsert_edges",
            namespace=getattr(spec, "namespace", None),
        )

        result = self._translator.upsert_edges(
            spec.edges,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )
        return validate_graph_result_type(
            result,
            expected_type=UpsertResult,
            operation="GraphTranslator.upsert_edges",
            error_code=ErrorCodes.BAD_UPSERT_RESULT,
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

        ctx = self._build_ctx(
            conversation=conversation,
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
        _ensure_not_in_event_loop("delete_nodes")

        ctx = self._build_ctx(
            conversation=conversation,
            extra_context=extra_context,
        )
        namespace = getattr(spec, "namespace", None)
        framework_ctx = self._framework_ctx(
            operation="delete_nodes",
            namespace=namespace,
        )

        if spec.filter is not None:
            raw_filter_or_ids: Any = spec.filter
            logger.debug(
                "delete_nodes using filter in namespace=%s",
                namespace,
            )
        else:
            ids = list(spec.ids or [])
            if not ids:
                raise BadRequest(
                    "DeleteNodesSpec must specify either filter or non-empty ids",
                    code=ErrorCodes.BAD_ADAPTER_RESULT,
                )
            raw_filter_or_ids = ids
            logger.debug(
                "delete_nodes using %d ids in namespace=%s",
                len(ids),
                namespace,
            )

        result = self._translator.delete_nodes(
            raw_filter_or_ids,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )
        return validate_graph_result_type(
            result,
            expected_type=DeleteResult,
            operation="GraphTranslator.delete_nodes",
            error_code=ErrorCodes.BAD_DELETE_RESULT,
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
        ctx = self._build_ctx(
            conversation=conversation,
            extra_context=extra_context,
        )
        namespace = getattr(spec, "namespace", None)
        framework_ctx = self._framework_ctx(
            operation="delete_nodes",
            namespace=namespace,
        )

        if spec.filter is not None:
            raw_filter_or_ids: Any = spec.filter
            logger.debug(
                "adelete_nodes using filter in namespace=%s",
                namespace,
            )
        else:
            ids = list(spec.ids or [])
            if not ids:
                raise BadRequest(
                    "DeleteNodesSpec must specify either filter or non-empty ids",
                    code=ErrorCodes.BAD_ADAPTER_RESULT,
                )
            raw_filter_or_ids = ids
            logger.debug(
                "adelete_nodes using %d ids in namespace=%s",
                len(ids),
                namespace,
            )

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
        _ensure_not_in_event_loop("delete_edges")

        ctx = self._build_ctx(
            conversation=conversation,
            extra_context=extra_context,
        )
        namespace = getattr(spec, "namespace", None)
        framework_ctx = self._framework_ctx(
            operation="delete_edges",
            namespace=namespace,
        )

        if spec.filter is not None:
            raw_filter_or_ids: Any = spec.filter
            logger.debug(
                "delete_edges using filter in namespace=%s",
                namespace,
            )
        else:
            ids = list(spec.ids or [])
            if not ids:
                raise BadRequest(
                    "DeleteEdgesSpec must specify either filter or non-empty ids",
                    code=ErrorCodes.BAD_ADAPTER_RESULT,
                )
            raw_filter_or_ids = ids
            logger.debug(
                "delete_edges using %d ids in namespace=%s",
                len(ids),
                namespace,
            )

        result = self._translator.delete_edges(
            raw_filter_or_ids,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )
        return validate_graph_result_type(
            result,
            expected_type=DeleteResult,
            operation="GraphTranslator.delete_edges",
            error_code=ErrorCodes.BAD_DELETE_RESULT,
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
        ctx = self._build_ctx(
            conversation=conversation,
            extra_context=extra_context,
        )
        namespace = getattr(spec, "namespace", None)
        framework_ctx = self._framework_ctx(
            operation="delete_edges",
            namespace=namespace,
        )

        if spec.filter is not None:
            raw_filter_or_ids: Any = spec.filter
            logger.debug(
                "adelete_edges using filter in namespace=%s",
                namespace,
            )
        else:
            ids = list(spec.ids or [])
            if not ids:
                raise BadRequest(
                    "DeleteEdgesSpec must specify either filter or non-empty ids",
                    code=ErrorCodes.BAD_ADAPTER_RESULT,
                )
            raw_filter_or_ids = ids
            logger.debug(
                "adelete_edges using %d ids in namespace=%s",
                len(ids),
                namespace,
            )

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
        _ensure_not_in_event_loop("bulk_vertices")

        ctx = self._build_ctx(
            conversation=conversation,
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

        result = self._translator.bulk_vertices(
            raw_request,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )
        return validate_graph_result_type(
            result,
            expected_type=BulkVerticesResult,
            operation="GraphTranslator.bulk_vertices",
            error_code=ErrorCodes.BAD_BULK_VERTICES_RESULT,
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
        ctx = self._build_ctx(
            conversation=conversation,
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
    # Traversal (sync + async)
    # ------------------------------------------------------------------ #

    @with_graph_error_context("traversal_sync")
    def traversal(
        self,
        spec: GraphTraversalSpec,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> TraversalResult:
        """
        Sync wrapper for graph traversal.

        Builds a raw traversal request and delegates to GraphTranslator.
        """
        _ensure_not_in_event_loop("traversal")

        ctx = self._build_ctx(
            conversation=conversation,
            extra_context=extra_context,
        )

        raw_request: Mapping[str, Any] = {
            "start_nodes": list(spec.start_nodes),
            "max_depth": spec.max_depth,
            "direction": spec.direction,
            "relationship_types": spec.relationship_types,
            "node_filters": spec.node_filters,
            "relationship_filters": spec.relationship_filters,
            "return_properties": spec.return_properties,
            "namespace": spec.namespace,
        }

        framework_ctx = self._framework_ctx(
            operation="traversal",
            namespace=spec.namespace,
        )

        result = self._translator.traversal(
            raw_request,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )
        return validate_graph_result_type(
            result,
            expected_type=TraversalResult,
            operation="GraphTranslator.traversal",
            error_code=ErrorCodes.BAD_TRAVERSAL_RESULT,
        )

    @with_async_graph_error_context("traversal_async")
    async def atraversal(
        self,
        spec: GraphTraversalSpec,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> TraversalResult:
        """
        Async wrapper for graph traversal.
        """
        ctx = self._build_ctx(
            conversation=conversation,
            extra_context=extra_context,
        )

        raw_request: Mapping[str, Any] = {
            "start_nodes": list(spec.start_nodes),
            "max_depth": spec.max_depth,
            "direction": spec.direction,
            "relationship_types": spec.relationship_types,
            "node_filters": spec.node_filters,
            "relationship_filters": spec.relationship_filters,
            "return_properties": spec.return_properties,
            "namespace": spec.namespace,
        }

        framework_ctx = self._framework_ctx(
            operation="traversal",
            namespace=spec.namespace,
        )

        result = await self._translator.arun_traversal(
            raw_request,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )
        return validate_graph_result_type(
            result,
            expected_type=TraversalResult,
            operation="GraphTranslator.arun_traversal",
            error_code=ErrorCodes.BAD_TRAVERSAL_RESULT,
        )

    # ------------------------------------------------------------------ #
    # Transaction + Batch (sync + async)
    # ------------------------------------------------------------------ #

    @with_graph_error_context("transaction_sync")
    def transaction(
        self,
        ops: List[BatchOperation],
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BatchResult:
        """
        Sync wrapper for transactional batch operations.

        Translates `BatchOperation` dataclasses into the raw mapping shape
        expected by GraphTranslator and returns the underlying `BatchResult`.
        """
        _ensure_not_in_event_loop("transaction")

        # Reuse batch validation; semantics are still a list of BatchOperation.
        validate_batch_operations(ops, operation="transaction", error_code="INVALID_BATCH_OPS")

        ctx = self._build_ctx(
            conversation=conversation,
            extra_context=extra_context,
        )

        raw_ops: List[Mapping[str, Any]] = [
            {"op": op.op, "args": dict(op.args or {})} for op in ops
        ]

        result = self._translator.transaction(
            raw_ops,
            op_ctx=ctx,
            framework_ctx=self._framework_ctx(operation="transaction"),
        )
        return validate_graph_result_type(
            result,
            expected_type=BatchResult,
            operation="GraphTranslator.transaction",
            error_code=ErrorCodes.BAD_TRANSACTION_RESULT,
        )

    @with_async_graph_error_context("transaction_async")
    async def atransaction(
        self,
        ops: List[BatchOperation],
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BatchResult:
        """
        Async wrapper for transactional batch operations.
        """
        validate_batch_operations(ops, operation="atransaction", error_code="INVALID_BATCH_OPS")

        ctx = self._build_ctx(
            conversation=conversation,
            extra_context=extra_context,
        )

        raw_ops: List[Mapping[str, Any]] = [
            {"op": op.op, "args": dict(op.args or {})} for op in ops
        ]

        result = await self._translator.arun_transaction(
            raw_ops,
            op_ctx=ctx,
            framework_ctx=self._framework_ctx(operation="transaction"),
        )
        return validate_graph_result_type(
            result,
            expected_type=BatchResult,
            operation="GraphTranslator.arun_transaction",
            error_code=ErrorCodes.BAD_TRANSACTION_RESULT,
        )

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
        _ensure_not_in_event_loop("batch")

        validate_batch_operations(ops, operation="batch", error_code="INVALID_BATCH_OPS")

        ctx = self._build_ctx(
            conversation=conversation,
            extra_context=extra_context,
        )

        raw_batch_ops: List[Mapping[str, Any]] = [
            {"op": op.op, "args": dict(op.args or {})} for op in ops
        ]

        result = self._translator.batch(
            raw_batch_ops,
            op_ctx=ctx,
            framework_ctx=self._framework_ctx(operation="batch"),
        )
        return validate_graph_result_type(
            result,
            expected_type=BatchResult,
            operation="GraphTranslator.batch",
            error_code=ErrorCodes.BAD_BATCH_RESULT,
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
        validate_batch_operations(ops, operation="abatch", error_code="INVALID_BATCH_OPS")

        ctx = self._build_ctx(
            conversation=conversation,
            extra_context=extra_context,
        )

        raw_batch_ops: List[Mapping[str, Any]] = [
            {"op": op.op, "args": dict(op.args or {})} for op in ops
        ]

        result = await self._translator.arun_batch(
            raw_batch_ops,
            op_ctx=ctx,
            framework_ctx=self._framework_ctx(operation="batch"),
        )
        return validate_graph_result_type(
            result,
            expected_type=BatchResult,
            operation="GraphTranslator.arun_batch",
            error_code=ErrorCodes.BAD_BATCH_RESULT,
        )


__all__ = [
    "AutoGenGraphClientProtocol",
    "AutoGenGraphFrameworkTranslator",
    "CorpusAutoGenGraphClient",
    "ErrorCodes",
    "with_graph_error_context",
    "with_async_graph_error_context",
    "with_error_context",
    "with_async_error_context",
]
