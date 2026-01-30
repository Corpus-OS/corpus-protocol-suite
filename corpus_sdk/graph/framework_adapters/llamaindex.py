# corpus_sdk/graph/framework_adapters/llamaindex.py
# SPDX-License-Identifier: Apache-2.0

"""
LlamaIndex adapter for Corpus Graph protocol.

This module exposes a Corpus `GraphProtocolV1` implementation as a
LlamaIndex-friendly client, with:

- Sync + async query APIs
- Sync + async streaming query APIs
- Proper integration with Corpus GraphProtocolV1
- OperationContext propagation derived from LlamaIndex callback manager
- Error-context enrichment for observability and debugging
- Orchestration, translation, and async→sync bridging via GraphTranslator

Design philosophy
-----------------
- Protocol-first: LlamaIndex is a thin skin over the Corpus graph adapter.
- All heavy lifting (deadlines, rate limits, caching, etc.) stays in
  the underlying `BaseGraphAdapter` / GraphProtocolV1 implementation.
- This layer focuses on:
    * Translating LlamaIndex callback manager → OperationContext
    * Building raw query / mutation shapes for GraphTranslator
    * Delegating all sync/async and streaming orchestration to GraphTranslator

Responsibilities
----------------
- Provide a convenient, LlamaIndex-oriented client for graph operations
- Keep all graph operations going through `GraphTranslator` so that
  async→sync bridging, streaming, and error-context logic are centralized
- Preserve protocol-level types (`QueryResult`, `QueryChunk`, etc.) for
  LlamaIndex callers

Non-responsibilities
--------------------
- Backend-specific graph behavior (lives in graph adapters)
- LlamaIndex index/query engine orchestration logic
- MMR and diversification details (handled inside GraphTranslator)

Namespace precedence (clarified)
--------------------------------
This adapter supports multiple ways to provide a namespace, depending on the
operation shape:

- Query APIs accept an explicit `namespace` argument (highest precedence), then
  fall back to the client defaults.
- Spec-driven APIs (e.g., UpsertNodesSpec, UpsertEdgesSpec, BulkVerticesSpec,
  GraphTraversalSpec, Delete*Spec) use `spec.namespace` when present, then fall
  back to the client defaults.
- The optional `CorpusGraphStore` wrapper can be configured with its own
  namespace; that namespace is passed explicitly to the client query calls.

This is a documentation clarification only: the runtime behavior matches the
existing patterns in this file and across other framework adapters.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
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
    from_llamaindex as core_ctx_from_llamaindex,
)
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


# --------------------------------------------------------------------------- #
# Error code constants (flat, framework-specific)
# --------------------------------------------------------------------------- #


class ErrorCodes:
    BAD_OPERATION_CONTEXT = "BAD_OPERATION_CONTEXT"
    BAD_TRANSLATED_SCHEMA = "BAD_TRANSLATED_SCHEMA"
    BAD_HEALTH_RESULT = "BAD_HEALTH_RESULT"
    BAD_TRANSLATED_RESULT = "BAD_TRANSLATED_RESULT"
    BAD_TRANSLATED_CHUNK = "BAD_TRANSLATED_CHUNK"
    BAD_UPSERT_RESULT = "BAD_UPSERT_RESULT"
    BAD_DELETE_RESULT = "BAD_DELETE_RESULT"
    BAD_BULK_VERTICES_RESULT = "BAD_BULK_VERTICES_RESULT"
    BAD_TRAVERSAL_RESULT = "BAD_TRAVERSAL_RESULT"
    BAD_BATCH_RESULT = "BAD_BATCH_RESULT"
    BAD_TRANSACTION_RESULT = "BAD_TRANSACTION_RESULT"
    BAD_ADAPTER_RESULT = "BAD_ADAPTER_RESULT"
    SYNC_WRAPPER_CALLED_IN_EVENT_LOOP = "SYNC_WRAPPER_CALLED_IN_EVENT_LOOP"

    # More explicit shape/validation codes used within this adapter.
    # NOTE: For backwards compatibility, some call sites still raise with
    # BAD_ADAPTER_RESULT; these constants exist to support clearer future
    # attribution and logging without changing external behavior unexpectedly.
    BAD_ASYNC_ITERATOR_SHAPE = "BAD_ASYNC_ITERATOR_SHAPE"
    INVALID_DELETE_SPEC = "INVALID_DELETE_SPEC"

    # Validation-level constants used by shared validators in this adapter.
    # Keeping these as explicit symbolic strings avoids accidental drift across frameworks.
    INVALID_QUERY = "INVALID_QUERY"
    INVALID_BATCH_OPS = "INVALID_BATCH_OPS"


# --------------------------------------------------------------------------- #
# Event-loop guard to prevent sync-over-async deadlocks
# --------------------------------------------------------------------------- #


def _ensure_not_in_event_loop(api_name: str) -> None:
    """
    Guard against calling sync wrappers from within an active asyncio event loop.

    This mirrors the AutoGen / other framework adapter pattern to avoid
    sync-over-async deadlocks in environments like Jupyter, FastAPI, etc.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop in this thread – safe to use sync wrappers.
        return

    # An event loop is running; raise a clear error directing callers to async APIs.
    raise RuntimeError(
        f"{api_name}() cannot be called from within an active asyncio event loop; "
        f"use a{api_name}() instead. [{ErrorCodes.SYNC_WRAPPER_CALLED_IN_EVENT_LOOP}]",
    )


# --------------------------------------------------------------------------- #
# Error-context decorators (centralized via common framework utils)
# --------------------------------------------------------------------------- #


def with_graph_error_context(
    operation: str,
    **static_context: Any,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for sync methods with rich dynamic context extraction.

    Thin wrapper over the shared `create_graph_error_context_decorator`
    for the LlamaIndex framework.
    """
    return create_graph_error_context_decorator(
        framework="llamaindex",
        is_async=False,
    )(operation=operation, **static_context)


def with_async_graph_error_context(
    operation: str,
    **static_context: Any,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for async methods with rich dynamic context extraction.

    Thin wrapper over the shared `create_graph_error_context_decorator`
    for the LlamaIndex framework.
    """
    return create_graph_error_context_decorator(
        framework="llamaindex",
        is_async=True,
    )(operation=operation, **static_context)


# Backwards-compatible aliases (for older imports)
with_error_context = with_graph_error_context
with_async_error_context = with_async_graph_error_context


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _looks_like_operation_context(obj: Any) -> bool:
    """
    Heuristic check; OperationContext may be a Protocol/alias in some SDK versions.

    This function is intentionally conservative:
    - If OperationContext is a real runtime class, prefer isinstance().
    - If OperationContext is a typing-only Protocol, rely on a structural check.
    - The structural check requires a *minimal coherent set* of attributes,
      rather than any single attribute, to reduce false positives.

    Rationale:
    - This adapter performs best-effort context translation. Accepting arbitrary
      objects as "context" can lead to confusing downstream behavior and harder
      debugging. A stricter check reduces accidental acceptance of unrelated types
      while still allowing forward-compatible OperationContext implementations.
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

    # Fallback to structural check.
    # Require attrs + (to_dict OR request_id OR traceparent) as a minimal set.
    has_attrs = hasattr(obj, "attrs")
    has_to_dict = hasattr(obj, "to_dict")
    has_request_id = hasattr(obj, "request_id")
    has_traceparent = hasattr(obj, "traceparent")

    if not has_attrs:
        return False

    # A context without any identifier/serialization surface is unlikely to be valid.
    if has_to_dict or has_request_id or has_traceparent:
        return True

    return False


def _is_async_iterator(obj: Any) -> bool:
    """
    Return True if the object looks like an AsyncIterator.

    We check for both __aiter__ and __anext__ to avoid treating arbitrary
    awaitables or objects with only partial async iteration methods as
    streaming iterators.
    """
    return hasattr(obj, "__aiter__") and hasattr(obj, "__anext__")


def _normalize_async_iterator(aiter_or_awaitable: Any) -> Any:
    """
    Normalize either:
      - an AsyncIterator, OR
      - an awaitable that resolves to an AsyncIterator,
    into a shape that the caller can safely consume.

    Why:
      GraphTranslator implementations may choose to return async iterators eagerly
      or lazily. Supporting both keeps this adapter robust to translator evolution
      and aligned with other framework adapters that accept both shapes.

    Contract:
      - If input is an awaitable, return it unchanged (caller awaits it).
      - If input is already an AsyncIterator, return it unchanged.
      - Otherwise, raise a TypeError with a clear adapter-specific error code.

    Note:
      We keep this helper small and dependency-free; it is used only in async
      streaming code paths and is not on hot query loops beyond streaming setup.
    """
    if inspect.isawaitable(aiter_or_awaitable):
        return aiter_or_awaitable
    if _is_async_iterator(aiter_or_awaitable):
        return aiter_or_awaitable

    raise TypeError(
        "Expected an AsyncIterator or an awaitable resolving to an AsyncIterator "
        f"from GraphTranslator.arun_query_stream. [{ErrorCodes.BAD_ASYNC_ITERATOR_SHAPE}]",
    )


# --------------------------------------------------------------------------- #
# Public framework translator
# --------------------------------------------------------------------------- #


class LlamaIndexGraphFrameworkTranslator(DefaultGraphFrameworkTranslator):
    """
    LlamaIndex-specific GraphFrameworkTranslator.

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


class LlamaIndexGraphClientProtocol(Protocol):
    """
    Protocol representing the minimal LlamaIndex-aware graph client interface
    implemented by this module.

    This structural protocol allows callers to type against the graph client
    without depending on the concrete `CorpusLlamaIndexGraphClient` class.
    """

    # Capabilities / schema / health -------------------------------------

    def capabilities(self, **kwargs: Any) -> Mapping[str, Any]:
        ...

    async def acapabilities(self, **kwargs: Any) -> Mapping[str, Any]:
        ...

    def get_schema(
        self,
        *,
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> GraphSchema:
        ...

    async def aget_schema(
        self,
        *,
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> GraphSchema:
        ...

    def health(
        self,
        *,
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        ...

    async def ahealth(
        self,
        *,
        callback_manager: Optional[Any] = None,
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
        callback_manager: Optional[Any] = None,
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
        callback_manager: Optional[Any] = None,
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
        callback_manager: Optional[Any] = None,
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
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> AsyncIterator[QueryChunk]:
        ...

    # Upsert --------------------------------------------------------------

    def upsert_nodes(
        self,
        spec: UpsertNodesSpec,
        *,
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        ...

    async def aupsert_nodes(
        self,
        spec: UpsertNodesSpec,
        *,
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        ...

    def upsert_edges(
        self,
        spec: UpsertEdgesSpec,
        *,
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        ...

    async def aupsert_edges(
        self,
        spec: UpsertEdgesSpec,
        *,
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        ...

    # Delete --------------------------------------------------------------

    def delete_nodes(
        self,
        spec: DeleteNodesSpec,
        *,
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        ...

    async def adelete_nodes(
        self,
        spec: DeleteNodesSpec,
        *,
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        ...

    def delete_edges(
        self,
        spec: DeleteEdgesSpec,
        *,
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        ...

    async def adelete_edges(
        self,
        spec: DeleteEdgesSpec,
        *,
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        ...

    # Bulk / batch --------------------------------------------------------

    def bulk_vertices(
        self,
        spec: BulkVerticesSpec,
        *,
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BulkVerticesResult:
        ...

    async def abulk_vertices(
        self,
        spec: BulkVerticesSpec,
        *,
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BulkVerticesResult:
        ...

    def batch(
        self,
        ops: List[BatchOperation],
        *,
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BatchResult:
        ...

    async def abatch(
        self,
        ops: List[BatchOperation],
        *,
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BatchResult:
        ...

    # Transaction ---------------------------------------------------------

    def transaction(
        self,
        ops: List[BatchOperation],
        *,
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BatchResult:
        ...

    async def atransaction(
        self,
        ops: List[BatchOperation],
        *,
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BatchResult:
        ...

    # Traversal -----------------------------------------------------------

    def traversal(
        self,
        spec: GraphTraversalSpec,
        *,
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> TraversalResult:
        ...

    async def atraversal(
        self,
        spec: GraphTraversalSpec,
        *,
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> TraversalResult:
        ...

    # Resource management -------------------------------------------------

    def close(self) -> None:
        ...

    async def aclose(self) -> None:
        ...


class CorpusLlamaIndexGraphClient:
    """
    LlamaIndex-oriented client wrapper around a Corpus `GraphProtocolV1`.

    This is a thin integration layer that:

    - Translates LlamaIndex `CallbackManager` instances into a Corpus
      `OperationContext` using `core_ctx_from_llamaindex`.
    - Uses `GraphTranslator` (with a LlamaIndex-specific framework translator) to:
        * Build Graph*Spec objects from simple inputs
        * Execute sync + async graph operations
        * Orchestrate streaming with proper cancellation and error handling
    - Delegates all async→sync bridging and streaming glue to GraphTranslator.
    - Attaches rich error context (`attach_context`) on this layer with
      LlamaIndex-specific hints when failures occur.
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
        Initialize a LlamaIndex-oriented graph client.

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
            Optional `GraphFrameworkTranslator` implementation. If not provided,
            `LlamaIndexGraphFrameworkTranslator` is used by default.

        Notes
        -----
        Historical documentation note (preserved for stability of wording/search):
        - "graph_adapter: Backwards-compatible alias for `adapter`. If both `adapter` and
           `graph_adapter` are provided, they must refer to the same object."
        - "graph_adapter: Backwards-compatible alias for `adapter`. If both `adapter` and
           `graph_adapter` are provided, they must refer to the same object."
        """
        # Resolving adapter / graph_adapter with basic duck-typed validation.
        if graph_adapter is not None and adapter is not None and graph_adapter is not adapter:
            raise TypeError("Provide only one of 'adapter' or 'graph_adapter', not both")

        resolved_adapter: Any = graph_adapter if graph_adapter is not None else adapter
        if resolved_adapter is None:
            raise TypeError("adapter must be a GraphProtocolV1-compatible graph adapter")

        self._graph: GraphProtocolV1 = resolved_adapter
        self._default_dialect: Optional[str] = default_dialect
        self._default_namespace: Optional[str] = default_namespace
        self._default_timeout_ms: Optional[int] = default_timeout_ms
        self._framework_version: Optional[str] = framework_version
        self._framework_translator: Optional[GraphFrameworkTranslator] = framework_translator
        self._closed: bool = False
        self._aclosed: bool = False

    # ------------------------------------------------------------------ #
    # Resource Management (Context Managers + explicit close)
    # ------------------------------------------------------------------ #

    def __enter__(self) -> CorpusLlamaIndexGraphClient:
        """Support context manager protocol for resource cleanup."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Clean up resources when exiting context."""
        self.close()

    async def __aenter__(self) -> CorpusLlamaIndexGraphClient:
        """Support async context manager protocol."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Clean up resources when exiting async context."""
        await self.aclose()

    def close(self) -> None:
        """
        Explicitly close the underlying graph adapter (sync).

        This is idempotent and logs any close errors instead of raising, to
        avoid surprising exceptions during context teardown.
        """
        if self._closed:
            return
        self._closed = True

        if hasattr(self._graph, "close"):
            try:
                self._graph.close()
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Error while closing graph adapter in close(): %s",
                    e,
                )

    async def aclose(self) -> None:
        """
        Explicitly close the underlying graph adapter (async).

        Prefers an async close method if available, falling back to sync close.
        """
        if self._aclosed:
            return
        self._aclosed = True

        if hasattr(self._graph, "aclose"):
            try:
                await self._graph.aclose()
                # Align sync/async close semantics: a successful async close also
                # satisfies the "closed" state from the perspective of callers.
                self._closed = True
                return
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Error while closing graph adapter in aclose(): %s",
                    e,
                )

        # Fallback to sync close if async close is unavailable or failed.
        self.close()

    # ------------------------------------------------------------------ #
    # Translator (lazy, cached) – DI-aware
    # ------------------------------------------------------------------ #

    @cached_property
    def _translator(self) -> GraphTranslator:
        """
        Lazily construct and cache the `GraphTranslator`.

        Uses `cached_property` for thread safety and performance.
        Honors an injected `framework_translator` if provided; otherwise
        falls back to the default `LlamaIndexGraphFrameworkTranslator`.
        """
        framework_translator: GraphFrameworkTranslator = (
            self._framework_translator or LlamaIndexGraphFrameworkTranslator()
        )
        return create_graph_translator(
            adapter=self._graph,
            framework="llamaindex",
            translator=framework_translator,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_ctx(
        self,
        *,
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Optional[OperationContext]:
        """
        Build an OperationContext from LlamaIndex-style inputs.

        Expected inputs
        ----------------
        - callback_manager: LlamaIndex CallbackManager (optional)
        - extra_context: Optional mapping merged into attrs (best effort)

        If both are None/empty, returns None and lets downstream helpers
        construct an "empty" OperationContext as needed.

        Context translation is best-effort: failures are logged and attached
        for observability, but graph operations may still proceed without an
        OperationContext.
        """
        extra: Dict[str, Any] = dict(extra_context or {})

        if callback_manager is None and not extra:
            return None

        try:
            ctx_candidate = core_ctx_from_llamaindex(
                callback_manager,
                framework_version=self._framework_version,
                **extra,
            )
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="llamaindex",
                operation="context_translation",
                error_code=ErrorCodes.BAD_OPERATION_CONTEXT,
                callback_manager_type=type(callback_manager).__name__
                if callback_manager is not None
                else "None",
                extra_context_keys=list(extra.keys()),
            )
            logger.warning(
                "[%s] Failed to build OperationContext from LlamaIndex inputs; "
                "proceeding without OperationContext: %s",
                ErrorCodes.BAD_OPERATION_CONTEXT,
                exc,
            )
            return None

        if not _looks_like_operation_context(ctx_candidate):
            logger.warning(
                "[%s] from_llamaindex produced non-OperationContext-like type: %s. "
                "Proceeding without OperationContext.",
                ErrorCodes.BAD_OPERATION_CONTEXT,
                type(ctx_candidate).__name__,
            )
            return None

        # Best-effort enrichment of attrs with framework + framework_version.
        try:
            attrs = getattr(ctx_candidate, "attrs", None)
            enriched_attrs: Dict[str, Any]
            if isinstance(attrs, Mapping):
                enriched_attrs = dict(attrs)
            else:
                enriched_attrs = {}

            if "framework" not in enriched_attrs:
                enriched_attrs["framework"] = "llamaindex"
            if (
                self._framework_version is not None
                and "framework_version" not in enriched_attrs
            ):
                enriched_attrs["framework_version"] = self._framework_version

            try:
                setattr(ctx_candidate, "attrs", enriched_attrs)
            except Exception:  # noqa: BLE001
                # Fallback: try to mutate attrs in-place if it's a mutable mapping.
                if isinstance(attrs, dict):
                    attrs.setdefault("framework", "llamaindex")
                    if self._framework_version is not None:
                        attrs.setdefault("framework_version", self._framework_version)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "[%s] Failed to enrich OperationContext.attrs for LlamaIndex: %s",
                ErrorCodes.BAD_OPERATION_CONTEXT,
                exc,
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

        Notes on params validation (best-effort)
        ---------------------------------------
        Some graph adapters serialize params to JSON. This adapter does not
        require params to be JSON-serializable (to preserve flexibility and
        avoid breaking existing integrations), but it does emit a debug log if
        the params are not JSON-serializable to help catch issues earlier.
        """
        effective_dialect = dialect or self._default_dialect
        effective_namespace = namespace or self._default_namespace
        effective_timeout = timeout_ms or self._default_timeout_ms

        # Materialize params to a dict early to ensure stable behavior regardless
        # of the caller's mapping type.
        materialized_params: Dict[str, Any] = dict(params or {})

        # Best-effort JSON-serializability check: logs only (no exception),
        # preserving existing behavior while improving debuggability.
        if materialized_params:
            try:
                json.dumps(materialized_params)
            except (TypeError, ValueError) as exc:
                logger.debug(
                    "Query params are not JSON-serializable (may be OK for some adapters): %s",
                    exc,
                )

        raw: Dict[str, Any] = {
            "text": query,
            "params": materialized_params,
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

        The effective namespace is derived from (in precedence order):
          1) The explicit namespace provided to this helper, if any
          2) The client default namespace, if configured

        This mirrors the existing behavior across this adapter, and is
        documented at the module level under "Namespace precedence".
        """
        ctx: Dict[str, Any] = {
            "framework": "llamaindex",
            "operation": operation,
        }

        if self._framework_version is not None:
            ctx["framework_version"] = self._framework_version

        effective_namespace = namespace or self._default_namespace
        if effective_namespace is not None:
            ctx["namespace"] = effective_namespace

        return ctx

    def _validate_upsert_edges_spec(self, spec: UpsertEdgesSpec) -> List[Any]:
        """
        LlamaIndex-local validation for edge upsert specs.

        Mirrors the stricter validation used in other framework adapters:
        - edges must not be None
        - edges must be iterable and non-empty
        - each edge must have id, src, dst, label
        - properties (if present) must be JSON-serializable
        """
        if spec.edges is None:
            raise BadRequest(
                "UpsertEdgesSpec.edges must not be None",
                code=ErrorCodes.BAD_ADAPTER_RESULT,
            )

        try:
            edges_iter = list(spec.edges)
        except TypeError as exc:
            raise BadRequest(
                "UpsertEdgesSpec.edges must be an iterable of edges",
                code=ErrorCodes.BAD_ADAPTER_RESULT,
            ) from exc

        if not edges_iter:
            raise BadRequest(
                "UpsertEdgesSpec must contain at least one edge",
                code=ErrorCodes.BAD_ADAPTER_RESULT,
            )

        for idx, edge in enumerate(edges_iter):
            if not getattr(edge, "id", None):
                raise BadRequest(
                    f"Edge at index {idx} must have an ID",
                    code=ErrorCodes.BAD_ADAPTER_RESULT,
                )
            if not getattr(edge, "src", None):
                raise BadRequest(
                    f"Edge at index {idx} must have source node ID",
                    code=ErrorCodes.BAD_ADAPTER_RESULT,
                )
            if not getattr(edge, "dst", None):
                raise BadRequest(
                    f"Edge at index {idx} must have target node ID",
                    code=ErrorCodes.BAD_ADAPTER_RESULT,
                )
            if not getattr(edge, "label", None):
                raise BadRequest(
                    f"Edge at index {idx} must have a label",
                    code=ErrorCodes.BAD_ADAPTER_RESULT,
                )
            properties = getattr(edge, "properties", None)
            if properties is not None:
                try:
                    json.dumps(properties)
                except (TypeError, ValueError) as e:
                    raise BadRequest(
                        f"Edge at index {idx} properties must be JSON-serializable: {e}",
                        code=ErrorCodes.BAD_ADAPTER_RESULT,
                    )

        # Previously this function mutated spec.edges to the validated list.
        # For cross-framework alignment and to avoid surprising side effects
        # (specs may be reused or treated as immutable), we return the validated
        # materialized list to the caller instead of mutating the input spec.
        return edges_iter

    def _select_filter_or_ids(
        self,
        *,
        spec_filter: Any,
        spec_ids: Any,
        spec_name: str,
        empty_message: str,
    ) -> Any:
        """
        Shared helper to select either a filter or a non-empty ID list.

        This improves consistency across delete_nodes/delete_edges sync+async
        implementations while preserving the original error handling semantics.

        Parameters
        ----------
        spec_filter:
            The filter expression from a Delete*Spec (may be None).
        spec_ids:
            The ids field from a Delete*Spec (may be None or iterable).
        spec_name:
            Used for logging/debugging and error messages.
        empty_message:
            The human-readable message used if both filter and ids are missing.

        Returns
        -------
        Any:
            The raw filter object or a materialized list of ids.

        Raises
        ------
        BadRequest:
            If neither filter nor a non-empty ids list is provided.
        """
        if spec_filter is not None:
            return spec_filter

        ids = list(spec_ids or [])
        if not ids:
            # Preserve existing behavior: raise BadRequest with BAD_ADAPTER_RESULT.
            # We also include a more specific constant (INVALID_DELETE_SPEC) in
            # the message for easier attribution, without changing external code fields.
            raise BadRequest(
                f"{empty_message} [{ErrorCodes.INVALID_DELETE_SPEC}]",
                code=ErrorCodes.BAD_ADAPTER_RESULT,
            )
        return ids

    def _build_bulk_vertices_request(self, spec: BulkVerticesSpec) -> Mapping[str, Any]:
        """
        Build the raw bulk vertices request mapping for GraphTranslator.

        Single Source of Truth pattern:
        -------------------------------
        Both sync and async bulk_vertices methods call this helper to ensure that
        additions/changes to BulkVerticesSpec fields are reflected uniformly.
        This reduces the risk of "payload materialization drift" between parallel
        code paths without changing semantics or performance characteristics.
        """
        return {
            "namespace": spec.namespace,
            "limit": spec.limit,
            "cursor": spec.cursor,
            "filter": spec.filter,
        }

    def _build_traversal_request(self, spec: GraphTraversalSpec) -> Mapping[str, Any]:
        """
        Build the raw traversal request mapping for GraphTranslator.

        Single Source of Truth pattern:
        -------------------------------
        Both sync and async traversal methods call this helper to ensure that
        additions/changes to GraphTraversalSpec fields are reflected uniformly.
        This reduces the risk of "payload materialization drift" between parallel
        code paths without changing semantics or performance characteristics.
        """
        return {
            "start_nodes": list(spec.start_nodes),
            "max_depth": spec.max_depth,
            "direction": spec.direction,
            "relationship_types": spec.relationship_types,
            "node_filters": spec.node_filters,
            "relationship_filters": spec.relationship_filters,
            "return_properties": spec.return_properties,
            "namespace": spec.namespace,
        }

    # ------------------------------------------------------------------ #
    # Capabilities / schema / health
    # ------------------------------------------------------------------ #

    @with_graph_error_context("capabilities_sync")
    def capabilities(self, **kwargs: Any) -> Mapping[str, Any]:
        """
        Sync wrapper around capabilities, delegating async→sync bridging
        to GraphTranslator.

        kwargs are accepted for forward compatibility but currently not forwarded
        to the translator (consistent with other framework adapters).
        """
        _ensure_not_in_event_loop("capabilities")
        caps = self._translator.capabilities()
        return graph_capabilities_to_dict(caps)

    @with_async_graph_error_context("capabilities_async")
    async def acapabilities(self, **kwargs: Any) -> Mapping[str, Any]:
        """
        Async capabilities accessor.

        We delegate to GraphTranslator for consistency, then normalize to a
        simple dict for LlamaIndex consumption.

        kwargs are accepted for forward compatibility but currently not forwarded
        to the translator (consistent with other framework adapters).
        """
        caps = await self._translator.arun_capabilities()
        return graph_capabilities_to_dict(caps)

    @with_graph_error_context("get_schema_sync")
    def get_schema(
        self,
        *,
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> GraphSchema:
        """
        Sync wrapper around `graph_adapter.get_schema(...)`.

        Delegates to GraphTranslator so that async→sync bridging and
        error-context handling are centralized.
        """
        _ensure_not_in_event_loop("get_schema")
        ctx = self._build_ctx(
            callback_manager=callback_manager,
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
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> GraphSchema:
        """
        Async wrapper around `graph_adapter.get_schema(...)`.

        Delegates to GraphTranslator.
        """
        ctx = self._build_ctx(
            callback_manager=callback_manager,
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
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        """
        Health check (sync).

        Uses GraphTranslator for consistency with other operations.
        """
        _ensure_not_in_event_loop("health")
        ctx = self._build_ctx(
            callback_manager=callback_manager,
            extra_context=extra_context,
        )
        health_result = self._translator.health(
            op_ctx=ctx,
            framework_ctx=self._framework_ctx(operation="health"),
        )
        return validate_graph_result_type(
            health_result,
            expected_type=Mapping,
            operation="GraphTranslator.health",
            error_code=ErrorCodes.BAD_HEALTH_RESULT,
        )

    @with_async_graph_error_context("health_async")
    async def ahealth(
        self,
        *,
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        """
        Health check (async).

        Uses GraphTranslator for consistency with other operations.
        """
        ctx = self._build_ctx(
            callback_manager=callback_manager,
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
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> QueryResult:
        """
        Execute a non-streaming graph query (sync).

        Returns the underlying `QueryResult` from the GraphProtocol adapter.
        """
        _ensure_not_in_event_loop("query")
        validate_graph_query(query, operation="query", error_code=ErrorCodes.INVALID_QUERY)

        ctx = self._build_ctx(
            callback_manager=callback_manager,
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

        try:
            result = self._translator.query(
                raw_query,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
                mmr_config=None,
            )
        except NotSupported:
            # Dialect fallback mirrors other framework adapters:
            # - If the caller explicitly provided a dialect, retry without it.
            # - If the dialect was not explicitly provided, preserve existing behavior.
            if dialect is not None:
                logger.debug(
                    "Dialect not supported; retrying query without dialect. dialect=%s",
                    dialect,
                )
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
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> QueryResult:
        """
        Execute a non-streaming graph query (async).

        Returns the underlying `QueryResult`.
        """
        validate_graph_query(query, operation="aquery", error_code=ErrorCodes.INVALID_QUERY)

        ctx = self._build_ctx(
            callback_manager=callback_manager,
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

        try:
            result = await self._translator.arun_query(
                raw_query,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
                mmr_config=None,
            )
        except NotSupported:
            if dialect is not None:
                logger.debug(
                    "Dialect not supported; retrying async query without dialect. dialect=%s",
                    dialect,
                )
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
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[QueryChunk]:
        """
        Execute a streaming graph query (sync), yielding `QueryChunk` items.

        Delegates streaming orchestration to GraphTranslator, which uses
        SyncStreamBridge under the hood. This method itself does not use
        any async→sync bridges directly.
        """
        _ensure_not_in_event_loop("stream_query")
        validate_graph_query(query, operation="stream_query", error_code=ErrorCodes.INVALID_QUERY)

        ctx = self._build_ctx(
            callback_manager=callback_manager,
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
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> AsyncIterator[QueryChunk]:
        """
        Execute a streaming graph query (async), yielding `QueryChunk` items.
        """
        validate_graph_query(query, operation="astream_query", error_code=ErrorCodes.INVALID_QUERY)

        ctx = self._build_ctx(
            callback_manager=callback_manager,
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

        # Accept either AsyncIterator or awaitable -> AsyncIterator, for cross-framework alignment.
        aiter_or_awaitable = self._translator.arun_query_stream(
            raw_query,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )
        normalized = _normalize_async_iterator(aiter_or_awaitable)
        if inspect.isawaitable(normalized):
            aiter = await normalized  # type: ignore[assignment]
        else:
            aiter = normalized  # type: ignore[assignment]

        # Additional guard: even if the awaitable resolves, ensure we got an AsyncIterator.
        if not _is_async_iterator(aiter):
            raise TypeError(
                "GraphTranslator.arun_query_stream resolved to a non-AsyncIterator "
                f"type: {type(aiter).__name__}. [{ErrorCodes.BAD_ASYNC_ITERATOR_SHAPE}]",
            )

        async for chunk in aiter:
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
        callback_manager: Optional[Any] = None,
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
            callback_manager=callback_manager,
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
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        """
        Async wrapper for upserting nodes.
        """
        validate_upsert_nodes_spec(spec)

        ctx = self._build_ctx(
            callback_manager=callback_manager,
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
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        """
        Sync wrapper for upserting edges.
        """
        _ensure_not_in_event_loop("upsert_edges")
        edges = self._validate_upsert_edges_spec(spec)

        ctx = self._build_ctx(
            callback_manager=callback_manager,
            extra_context=extra_context,
        )
        framework_ctx = self._framework_ctx(
            operation="upsert_edges",
            namespace=getattr(spec, "namespace", None),
        )

        result = self._translator.upsert_edges(
            edges,
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
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        """
        Async wrapper for upserting edges.
        """
        edges = self._validate_upsert_edges_spec(spec)

        ctx = self._build_ctx(
            callback_manager=callback_manager,
            extra_context=extra_context,
        )
        framework_ctx = self._framework_ctx(
            operation="upsert_edges",
            namespace=getattr(spec, "namespace", None),
        )

        result = await self._translator.arun_upsert_edges(
            edges,
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
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        """
        Sync wrapper for deleting nodes.

        Uses DeleteNodesSpec to derive either an ID list or a filter
        expression for the GraphTranslator.
        """
        _ensure_not_in_event_loop("delete_nodes")
        ctx = self._build_ctx(
            callback_manager=callback_manager,
            extra_context=extra_context,
        )
        framework_ctx = self._framework_ctx(
            operation="delete_nodes",
            namespace=getattr(spec, "namespace", None),
        )

        # The selection logic is shared to ensure consistent behavior and
        # reduce drift between sync/async and nodes/edges variants.
        raw_filter_or_ids: Any = self._select_filter_or_ids(
            spec_filter=spec.filter,
            spec_ids=spec.ids,
            spec_name="DeleteNodesSpec",
            empty_message="DeleteNodesSpec must specify either filter or non-empty ids",
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
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        """
        Async wrapper for deleting nodes.
        """
        ctx = self._build_ctx(
            callback_manager=callback_manager,
            extra_context=extra_context,
        )
        framework_ctx = self._framework_ctx(
            operation="delete_nodes",
            namespace=getattr(spec, "namespace", None),
        )

        raw_filter_or_ids: Any = self._select_filter_or_ids(
            spec_filter=spec.filter,
            spec_ids=spec.ids,
            spec_name="DeleteNodesSpec",
            empty_message="DeleteNodesSpec must specify either filter or non-empty ids",
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
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        """
        Sync wrapper for deleting edges.
        """
        _ensure_not_in_event_loop("delete_edges")

        # IMPORTANT BUGFIX:
        # This method must use the same context builder as other operations.
        # Using a misspelled/mismatched builder breaks runtime behavior and tests.
        ctx = self._build_ctx(
            callback_manager=callback_manager,
            extra_context=extra_context,
        )

        framework_ctx = self._framework_ctx(
            operation="delete_edges",
            namespace=getattr(spec, "namespace", None),
        )

        # The selection logic is shared to ensure consistent behavior and
        # reduce drift between sync/async and nodes/edges variants.
        raw_filter_or_ids: Any = self._select_filter_or_ids(
            spec_filter=spec.filter,
            spec_ids=spec.ids,
            spec_name="DeleteEdgesSpec",
            empty_message="DeleteEdgesSpec must specify either filter or non-empty ids",
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
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        """
        Async wrapper for deleting edges.
        """
        ctx = self._build_ctx(
            callback_manager=callback_manager,
            extra_context=extra_context,
        )
        framework_ctx = self._framework_ctx(
            operation="delete_edges",
            namespace=getattr(spec, "namespace", None),
        )

        raw_filter_or_ids: Any = self._select_filter_or_ids(
            spec_filter=spec.filter,
            spec_ids=spec.ids,
            spec_name="DeleteEdgesSpec",
            empty_message="DeleteEdgesSpec must specify either filter or non-empty ids",
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
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BulkVerticesResult:
        """
        Sync wrapper for bulk_vertices.

        Converts `BulkVerticesSpec` into the raw request shape expected by
        GraphTranslator and returns the underlying `BulkVerticesResult`.
        """
        _ensure_not_in_event_loop("bulk_vertices")
        ctx = self._build_ctx(
            callback_manager=callback_manager,
            extra_context=extra_context,
        )

        raw_request = self._build_bulk_vertices_request(spec)

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
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BulkVerticesResult:
        """
        Async wrapper for bulk_vertices.
        """
        ctx = self._build_ctx(
            callback_manager=callback_manager,
            extra_context=extra_context,
        )

        raw_request = self._build_bulk_vertices_request(spec)

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
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> TraversalResult:
        """
        Sync wrapper for graph traversal.

        Builds a raw traversal request and delegates to GraphTranslator.
        """
        _ensure_not_in_event_loop("traversal")

        ctx = self._build_ctx(
            callback_manager=callback_manager,
            extra_context=extra_context,
        )

        raw_request = self._build_traversal_request(spec)

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
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> TraversalResult:
        """
        Async wrapper for graph traversal.
        """
        ctx = self._build_ctx(
            callback_manager=callback_manager,
            extra_context=extra_context,
        )

        raw_request = self._build_traversal_request(spec)

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
    # Batch (sync + async)
    # ------------------------------------------------------------------ #

    @with_graph_error_context("batch_sync")
    def batch(
        self,
        ops: List[BatchOperation],
        *,
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BatchResult:
        """
        Sync wrapper for batch operations.

        Translates `BatchOperation` dataclasses into the raw mapping shape
        expected by GraphTranslator and returns the underlying `BatchResult`.
        """
        _ensure_not_in_event_loop("batch")
        validate_batch_operations(
            ops,
            operation="batch",
            error_code=ErrorCodes.INVALID_BATCH_OPS,
        )

        ctx = self._build_ctx(
            callback_manager=callback_manager,
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
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BatchResult:
        """
        Async wrapper for batch operations.
        """
        validate_batch_operations(
            ops,
            operation="abatch",
            error_code=ErrorCodes.INVALID_BATCH_OPS,
        )

        ctx = self._build_ctx(
            callback_manager=callback_manager,
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

    # ------------------------------------------------------------------ #
    # Transaction (sync + async)
    # ------------------------------------------------------------------ #

    @with_graph_error_context("transaction_sync")
    def transaction(
        self,
        ops: List[BatchOperation],
        *,
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BatchResult:
        """
        Sync wrapper for transactional batch operations.

        Translates `BatchOperation` dataclasses into the raw mapping shape
        expected by GraphTranslator and returns the underlying `BatchResult`.
        """
        _ensure_not_in_event_loop("transaction")
        validate_batch_operations(
            ops,
            operation="transaction",
            error_code=ErrorCodes.INVALID_BATCH_OPS,
        )

        ctx = self._build_ctx(
            callback_manager=callback_manager,
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
        callback_manager: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BatchResult:
        """
        Async wrapper for transactional batch operations.
        """
        validate_batch_operations(
            ops,
            operation="atransaction",
            error_code=ErrorCodes.INVALID_BATCH_OPS,
        )

        ctx = self._build_ctx(
            callback_manager=callback_manager,
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


from typing import TYPE_CHECKING

try:
    # Optional LlamaIndex dependency – only needed if you want CorpusGraphStore
    from llama_index.core.graph_stores.types import GraphStore as _LlamaIndexGraphStore
except ImportError:  # pragma: no cover - optional dependency
    _LlamaIndexGraphStore = None  # type: ignore[assignment]


# ------------------------------------------------------------------ #
# Optional LlamaIndex GraphStore wrapper
# ------------------------------------------------------------------ #

if _LlamaIndexGraphStore is not None:

    class CorpusGraphStore(_LlamaIndexGraphStore):  # type: ignore[misc]
        """
        LlamaIndex `GraphStore` implementation backed by CorpusLlamaIndexGraphClient.

        This is a thin adapter that lets you plug a Corpus graph into
        LlamaIndex's `KnowledgeGraphIndex` (and any other GraphStore consumers).

        NOTE
        ----
        CorpusGraphStore is intentionally *lightly opinionated* and does **not**
        assume a particular triple schema in your Corpus graph. You are expected
        to either:

        - Provide concrete query strings / functions that map:
            (subj, rel, obj) <-> your graph schema
        - Or subclass CorpusGraphStore and override the triplet methods.

        The default implementations of `get`, `get_rel_map`, `upsert_triplet`
        and `delete` raise NotImplementedError to force you to define the
        mapping explicitly for your deployment.
        """

        schema: str = ""

        def __init__(
            self,
            client: "CorpusLlamaIndexGraphClient",
            *,
            namespace: Optional[str] = None,
            # Optional: basic triplet mapping hooks
            get_query: Optional[str] = None,
            get_rel_map_query: Optional[str] = None,
            upsert_triplet_query: Optional[str] = None,
            delete_triplet_query: Optional[str] = None,
        ) -> None:
            """
            Parameters
            ----------
            client:
                Underlying CorpusLlamaIndexGraphClient.
            namespace:
                Optional default namespace to use for all operations.
            get_query / get_rel_map_query / upsert_triplet_query / delete_triplet_query:
                Optional graph-query text for triplet-style operations.

                These are deliberately free-form and depend on your graph
                adapter / dialect. If omitted, the corresponding methods
                will raise NotImplementedError and you should override
                them in a subclass instead.
            """
            self._client = client
            self._namespace = namespace

            self._get_query = get_query
            self._get_rel_map_query = get_rel_map_query
            self._upsert_triplet_query = upsert_triplet_query
            self._delete_triplet_query = delete_triplet_query

        # ----------------------------- #
        # GraphStore protocol methods   #
        # ----------------------------- #

        @property
        def client(self) -> Any:
            """Expose underlying client."""
            return self._client

        def query(
            self,
            query: str,
            param_map: Optional[Dict[str, Any]] = None,
        ) -> Any:
            """
            Generic query interface required by GraphStore.

            This is a simple pass-through to CorpusLlamaIndexGraphClient.query,
            using the configured namespace.
            """
            return self._client.query(
                query,
                params=param_map,
                namespace=self._namespace,
            )

        def get(self, subj: str) -> List[List[str]]:
            """
            Get triplets for a given subject.

            By default this requires `get_query` to be provided, and expects
            the underlying graph adapter to return rows shaped as:
                [[subj, rel, obj], ...].

            Override this method in a subclass if your schema or query
            mechanism differs.
            """
            if self._get_query is None:
                raise NotImplementedError(
                    "CorpusGraphStore.get requires a 'get_query' or an override. "
                    "Provide a graph query that returns [[subj, rel, obj], ...]."
                )

            result = self._client.query(
                self._get_query,
                params={"subj": subj},
                namespace=self._namespace,
            )

            # We deliberately do no strict shape enforcement here – callers
            # can normalize in a subclass if needed.
            return getattr(result, "rows", result)  # type: ignore[return-value]

        def get_rel_map(
            self,
            subjs: Optional[List[str]] = None,
            depth: int = 2,
            limit: int = 30,
        ) -> Dict[str, List[List[str]]]:
            """
            Get a depth-aware relation map.

            Default implementation requires `get_rel_map_query` and expects
            a mapping {subject: [[subj, rel, obj], ...]} from the graph.

            If your adapter exposes something richer, override this method.
            """
            if self._get_rel_map_query is None:
                raise NotImplementedError(
                    "CorpusGraphStore.get_rel_map requires a 'get_rel_map_query' "
                    "or an override."
                )

            params: Dict[str, Any] = {
                "subjs": subjs,
                "depth": depth,
                "limit": limit,
            }
            result = self._client.query(
                self._get_rel_map_query,
                params=params,
                namespace=self._namespace,
            )
            return getattr(result, "rel_map", result)  # type: ignore[return-value]

        def upsert_triplet(self, subj: str, rel: str, obj: str) -> None:
            """
            Upsert a single (subject, relation, object) triplet.

            Default implementation calls a user-supplied `upsert_triplet_query`.
            For tighter integration (e.g., mapping directly to UpsertNodesSpec /
            UpsertEdgesSpec), subclass and implement in terms of
            CorpusLlamaIndexGraphClient.upsert_*.
            """
            if self._upsert_triplet_query is None:
                raise NotImplementedError(
                    "CorpusGraphStore.upsert_triplet requires an "
                    "'upsert_triplet_query' or an override."
                )

            self._client.query(
                self._upsert_triplet_query,
                params={"subj": subj, "rel": rel, "obj": obj},
                namespace=self._namespace,
            )

        def delete(self, subj: str, rel: str, obj: str) -> None:
            """
            Delete a single (subject, relation, object) triplet.

            Default implementation calls a user-supplied `delete_triplet_query`.
            Override to map directly onto graph delete APIs if desired.
            """
            if self._delete_triplet_query is None:
                raise NotImplementedError(
                    "CorpusGraphStore.delete requires a 'delete_triplet_query' "
                    "or an override."
                )

            self._client.query(
                self._delete_triplet_query,
                params={"subj": subj, "rel": rel, "obj": obj},
                namespace=self._namespace,
            )

        def persist(
            self,
            persist_path: str,
            fs: Optional[Any] = None,
        ) -> None:
            """
            Persist the graph store (no-op by default).

            Corpus graphs are typically remote services; persistence is often
            handled by the backend itself. Override this if you maintain a
            local cache or export format.
            """
            return

        def get_schema(self, refresh: bool = False) -> str:
            """
            Get the schema as a string.

            Default implementation simply returns `str(GraphSchema)` from the
            underlying Corpus graph. Override if you want a more structured
            or Cypher-specific schema representation.
            """
            schema = self._client.get_schema()
            return str(schema)

else:
    # LlamaIndex not installed – provide a stub that fails loudly at runtime
    class CorpusGraphStore:  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "llama_index is not installed; CorpusGraphStore is unavailable. "
                "Install llama-index to use this integration."
            )


__all__ = [
    "LlamaIndexGraphClientProtocol",
    "CorpusLlamaIndexGraphClient",
    "LlamaIndexGraphFrameworkTranslator",
    "CorpusGraphStore",
    "ErrorCodes",
    "with_graph_error_context",
    "with_async_graph_error_context",
    "with_error_context",
    "with_async_error_context",
]
