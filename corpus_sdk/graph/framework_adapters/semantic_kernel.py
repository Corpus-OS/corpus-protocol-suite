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

Namespace precedence (clarified)
--------------------------------
This adapter supports multiple ways to provide a namespace, depending on the
operation shape:

- Query APIs accept an explicit `namespace` argument (highest precedence), then
  fall back to the client defaults.
- Spec-driven APIs (e.g., UpsertNodesSpec, UpsertEdgesSpec, BulkVerticesSpec,
  GraphTraversalSpec, Delete*Spec) use `spec.namespace` when present, then fall
  back to the client defaults.
- Semantic Kernel `settings` and `context` are used to derive OperationContext
  for observability and policy propagation, but do not override the explicit
  namespace argument or spec namespace unless your core translation layer
  intentionally injects such values into attrs for downstream consumption.

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
    from_semantic_kernel as core_ctx_from_semantic_kernel,
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


# Error code constants (flat, framework-specific)
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
    BAD_TRANSACTION_RESULT = "BAD_TRANSACTION_RESULT"
    BAD_BATCH_RESULT = "BAD_BATCH_RESULT"
    BAD_ADAPTER_RESULT = "BAD_ADAPTER_RESULT"
    SYNC_WRAPPER_CALLED_IN_EVENT_LOOP = "SYNC_WRAPPER_CALLED_IN_EVENT_LOOP"

    # Validation-level constants used by shared validators in this adapter.
    # Keeping these as explicit symbolic strings avoids accidental drift across frameworks.
    INVALID_QUERY = "INVALID_QUERY"
    INVALID_BATCH_OPS = "INVALID_BATCH_OPS"

    # More explicit constants used by adapter-local guards/helpers.
    BAD_ASYNC_ITERATOR_SHAPE = "BAD_ASYNC_ITERATOR_SHAPE"
    INVALID_DELETE_SPEC = "INVALID_DELETE_SPEC"


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
    for the Semantic Kernel framework.
    """
    return create_graph_error_context_decorator(
        framework="semantic_kernel",
        is_async=False,
    )(operation=operation, **static_context)


def with_async_graph_error_context(
    operation: str,
    **static_context: Any,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for async methods with rich dynamic context extraction.

    Thin wrapper over the shared `create_graph_error_context_decorator`
    for the Semantic Kernel framework.
    """
    return create_graph_error_context_decorator(
        framework="semantic_kernel",
        is_async=True,
    )(operation=operation, **static_context)


# Backwards-compatible aliases (for older imports)
with_error_context = with_graph_error_context
with_async_error_context = with_async_graph_error_context


def _looks_like_operation_context(obj: Any) -> bool:
    """
    Heuristic check; OperationContext may be a Protocol/alias in some SDK versions.

    We first try a nominal isinstance check, then fall back to a structural
    check on common attributes used by OperationContext.

    IMPORTANT (tightened for safety and clarity)
    -------------------------------------------
    The structural check is intentionally conservative: instead of accepting
    any object that has *any* attribute matching OperationContext, it requires
    a minimal coherent set:

    - The object must expose `attrs`, and
    - It must expose at least one of: `to_dict`, `request_id`, `traceparent`

    This reduces false positives (e.g., accidentally accepting unrelated objects)
    while remaining forward-compatible with Protocol-based OperationContext shapes.
    """
    if obj is None:
        return False

    try:
        if isinstance(obj, OperationContext):
            return True
    except TypeError:
        # OperationContext may be a typing.Protocol in some modes.
        pass

    # Require attrs + (to_dict OR request_id OR traceparent) as a minimal set.
    has_attrs = hasattr(obj, "attrs")
    has_to_dict = hasattr(obj, "to_dict")
    has_request_id = hasattr(obj, "request_id")
    has_traceparent = hasattr(obj, "traceparent")

    if not has_attrs:
        return False

    return bool(has_to_dict or has_request_id or has_traceparent)


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
      or lazily (as an awaitable). Supporting both keeps this adapter robust to
      translator evolution and aligned with other framework adapters.

    Contract:
      - If input is an awaitable, return it unchanged (caller awaits it).
      - If input is already an AsyncIterator, return it unchanged.
      - Otherwise, raise a TypeError with a clear adapter-specific error code.
    """
    if inspect.isawaitable(aiter_or_awaitable):
        return aiter_or_awaitable
    if _is_async_iterator(aiter_or_awaitable):
        return aiter_or_awaitable

    raise TypeError(
        "Expected an AsyncIterator or an awaitable resolving to an AsyncIterator "
        f"from GraphTranslator.arun_query_stream. [{ErrorCodes.BAD_ASYNC_ITERATOR_SHAPE}]",
    )


def _ensure_not_in_event_loop(api_name: str) -> None:
    """
    Guard to prevent sync methods from being called inside a running event loop.

    This mirrors the behavior of other framework adapters (e.g., AutoGen /
    LangChain / LlamaIndex) to avoid sync-over-async deadlocks in environments
    like Jupyter, FastAPI, or SK-hosted loops.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running event loop in this thread – safe for sync calls.
        return

    if loop.is_running():
        # We raise a RuntimeError with a stable, symbolic error code to make it
        # easy to detect and handle in higher-level orchestration.
        raise RuntimeError(
            f"{ErrorCodes.SYNC_WRAPPER_CALLED_IN_EVENT_LOOP}: "
            f"Semantic Kernel sync graph API '{api_name}' cannot be called "
            "from an active event loop. Use the corresponding async method instead."
        )


class SemanticKernelGraphClientProtocol(Protocol):
    """
    Protocol describing the Semantic Kernel–aware graph client interface.

    This allows callers to type against the client without depending on
    the concrete `CorpusSemanticKernelGraphClient` implementation.
    """

    # Capabilities / schema / health -------------------------------------

    def capabilities(self, **kwargs: Any) -> Mapping[str, Any]:
        ...

    async def acapabilities(self, **kwargs: Any) -> Mapping[str, Any]:
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

    # Resource management -------------------------------------------------

    def close(self) -> None:
        ...

    async def aclose(self) -> None:
        ...


class SemanticKernelGraphFrameworkTranslator(DefaultGraphFrameworkTranslator):
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

    def translate_transaction_result(
        self,
        result: BatchResult,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> BatchResult:
        return result

    def translate_traversal_result(
        self,
        result: TraversalResult,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> TraversalResult:
        return result

    def translate_schema(
        self,
        schema: GraphSchema,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> GraphSchema:
        return schema


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
        Initialize a Semantic Kernel–oriented graph client.

        Parameters
        ----------
        adapter:
            Underlying `GraphProtocolV1` implementation (preferred parameter name).
        graph_adapter:
            Alternate name for `adapter`. Provide only one of `adapter` or `graph_adapter`.
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
            Optional framework-specific Graph translator. When not provided,
            `SemanticKernelGraphFrameworkTranslator` is used by default.
        """
        if adapter is not None and graph_adapter is not None:
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

        # Idempotent close flags for explicit close() / aclose() and context managers.
        self._closed: bool = False
        self._aclosed: bool = False

    # ------------------------------------------------------------------ #
    # Resource Management (Context Managers)
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "CorpusSemanticKernelGraphClient":
        """Support context manager protocol for resource cleanup."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Clean up resources when exiting context.

        Delegates to the explicit close() method so that cleanup behavior
        is centralized and idempotent.
        """
        try:
            self.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Error while closing graph client in __exit__: %s",
                exc,
            )

    async def __aenter__(self) -> "CorpusSemanticKernelGraphClient":
        """Support async context manager protocol."""
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> None:
        """
        Clean up resources when exiting async context.

        Delegates to the explicit aclose() method so that cleanup behavior
        is centralized and idempotent.
        """
        try:
            await self.aclose()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Error while closing graph client in __aexit__: %s",
                exc,
            )

    def close(self) -> None:
        """
        Explicitly close underlying resources (sync).

        This method is idempotent and safe to call multiple times. It will
        call `close()` on the underlying graph adapter when available.
        """
        if self._closed:
            return
        self._closed = True

        if hasattr(self._graph, "close"):
            try:
                self._graph.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Error while closing graph adapter in close(): %s",
                    exc,
                )

    async def aclose(self) -> None:
        """
        Explicitly close underlying resources (async).

        This method is idempotent and safe to call multiple times. It will
        call `aclose()` on the underlying graph adapter when available.

        Close semantics alignment (clarified)
        ------------------------------------
        - If async close succeeds, we also mark the sync-close flag as closed,
          because from a caller perspective the underlying adapter is closed.
        - If async close is unavailable or fails, we fall back to sync close
          to avoid leaking resources in mixed adapter implementations.
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
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Error while closing graph adapter in aclose(): %s",
                    exc,
                )

        # Fallback to sync close if async close is unavailable or failed.
        self.close()

    # ------------------------------------------------------------------ #
    # Translator (lazy, cached)
    # ------------------------------------------------------------------ #

    @cached_property
    def _translator(self) -> GraphTranslator:
        """
        Lazily construct and cache the `GraphTranslator`.

        Uses `cached_property` for thread safety and performance.
        """
        framework_translator = self._framework_translator or SemanticKernelGraphFrameworkTranslator()
        return create_graph_translator(
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

        If all are empty/None, returns None and lets downstream helpers
        construct an "empty" OperationContext as needed.

        Context translation is best-effort: failures are logged and attached
        for observability, but graph operations proceed without an
        OperationContext when necessary.
        """
        extra: Dict[str, Any] = dict(extra_context or {})

        if context is None and settings is None and not extra:
            return None

        try:
            ctx_candidate = core_ctx_from_semantic_kernel(
                context,
                settings=settings,
                framework_version=self._framework_version,
                **extra,
            )
        except Exception as exc:  # noqa: BLE001
            # Attach rich error context but do not fail the graph call purely
            # because context translation misbehaved.
            attach_context(
                exc,
                framework="semantic_kernel",
                operation="context_translation",
                error_code=ErrorCodes.BAD_OPERATION_CONTEXT,
                context_type=type(context).__name__ if context is not None else "None",
                settings_type=type(settings).__name__ if settings is not None else "None",
                extra_context_keys=list(extra.keys()),
            )
            logger.warning(
                "[%s] Failed to build OperationContext from Semantic Kernel inputs; "
                "proceeding without OperationContext. context_type=%s settings_type=%s extra_keys=%s",
                ErrorCodes.BAD_OPERATION_CONTEXT,
                type(context).__name__ if context is not None else "None",
                type(settings).__name__ if settings is not None else "None",
                list(extra.keys()),
            )
            return None

        if not _looks_like_operation_context(ctx_candidate):
            logger.warning(
                "[%s] from_semantic_kernel produced unsupported context type %s; "
                "proceeding without OperationContext.",
                ErrorCodes.BAD_OPERATION_CONTEXT,
                type(ctx_candidate).__name__,
            )
            return None

        ctx: OperationContext = ctx_candidate  # type: ignore[assignment]

        # Best-effort enrichment of OperationContext.attrs with framework metadata.
        attrs = getattr(ctx, "attrs", None)
        try:
            if isinstance(attrs, dict):
                if "framework" not in attrs:
                    attrs["framework"] = "semantic_kernel"
                if self._framework_version is not None and "framework_version" not in attrs:
                    attrs["framework_version"] = self._framework_version
        except Exception:  # noqa: BLE001
            # Attrs may be immutable or otherwise protected; this is non-fatal.
            logger.debug(
                "Failed to enrich OperationContext.attrs for Semantic Kernel framework.",
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
        Build a framework_ctx mapping that lets the common translator derive
        a preferred namespace and capture framework metadata for observability.
        """
        ctx: Dict[str, Any] = {
            "framework": "semantic_kernel",
            "operation": operation,
        }

        if self._framework_version is not None:
            ctx["framework_version"] = self._framework_version

        effective_namespace = namespace or self._default_namespace
        if effective_namespace is not None:
            ctx["namespace"] = effective_namespace

        return ctx

    def _select_filter_or_ids(
        self,
        *,
        spec_filter: Any,
        spec_ids: Any,
        empty_message: str,
    ) -> Any:
        """
        Shared helper to select either a filter or a non-empty ID list.

        This improves consistency across delete_nodes/delete_edges sync+async
        implementations while preserving the original error handling semantics.
        """
        if spec_filter is not None:
            return spec_filter

        ids = list(spec_ids or [])
        if not ids:
            # Preserve existing behavior: raise BadRequest with BAD_ADAPTER_RESULT.
            # Include an additional stable identifier in the message for easier
            # attribution without changing external code fields.
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

    def _validate_upsert_edges_spec(self, spec: UpsertEdgesSpec) -> List[Any]:
        """
        Basic structural validation for UpsertEdgesSpec.edges to provide
        clearer errors before reaching the adapter / translator.

        This mirrors the stricter checks in other framework adapters:
        - edges must not be None
        - edges must be iterable and non-empty
        - each edge must have id, src, dst, label
        - properties (if present) must be JSON-serializable

        IMPORTANT (side-effect avoidance)
        --------------------------------
        This helper returns a validated, materialized list of edges instead of
        mutating spec.edges. Avoiding mutation prevents surprising behavior if
        the caller reuses a spec object or passes an iterator intended to be
        consumed once.
        """
        if spec.edges is None:
            raise BadRequest(
                "UpsertEdgesSpec.edges must not be None",
                code=ErrorCodes.BAD_ADAPTER_RESULT,
            )

        try:
            edges_list = list(spec.edges)
        except TypeError as exc:
            raise BadRequest(
                "UpsertEdgesSpec.edges must be an iterable of edges",
                code=ErrorCodes.BAD_ADAPTER_RESULT,
            ) from exc

        if not edges_list:
            raise BadRequest(
                "UpsertEdgesSpec must contain at least one edge",
                code=ErrorCodes.BAD_ADAPTER_RESULT,
            )

        for idx, edge in enumerate(edges_list):
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

            props = getattr(edge, "properties", None)
            if props is not None:
                try:
                    json.dumps(props)
                except (TypeError, ValueError) as exc:
                    raise BadRequest(
                        f"Edge at index {idx} properties must be JSON-serializable: {exc}",
                        code=ErrorCodes.BAD_ADAPTER_RESULT,
                    ) from exc

        return edges_list

    # ------------------------------------------------------------------ #
    # Capabilities / schema / health
    # ------------------------------------------------------------------ #

    @with_graph_error_context("capabilities_sync")
    def capabilities(self, **kwargs: Any) -> Mapping[str, Any]:
        """
        Sync wrapper around capabilities, delegating async→sync bridging
        to GraphTranslator.

        kwargs are accepted for forward compatibility:
        - Future GraphTranslator/adapter implementations may accept capability
          filters or options.
        - This adapter forwards kwargs when supported, and falls back to the
          current signature without raising if not supported.
        """
        _ensure_not_in_event_loop("capabilities")

        # Forward kwargs when supported; otherwise fall back without error.
        # We avoid signature inspection (which can be expensive and brittle)
        # by using a simple TypeError fallback.
        try:
            caps = self._translator.capabilities(**kwargs)  # type: ignore[misc]
        except TypeError:
            if kwargs:
                logger.debug(
                    "GraphTranslator.capabilities does not accept kwargs; ignoring: %s",
                    sorted(kwargs.keys()),
                )
            caps = self._translator.capabilities()

        # Normalize to a simple dict for SK consumption via shared helper.
        return graph_capabilities_to_dict(caps)

    @with_async_graph_error_context("capabilities_async")
    async def acapabilities(self, **kwargs: Any) -> Mapping[str, Any]:
        """
        Async capabilities accessor.

        We delegate to GraphTranslator for consistency, then normalize to a
        simple dict for SK consumption.

        kwargs are accepted for forward compatibility and forwarded when supported.
        """
        try:
            caps = await self._translator.arun_capabilities(**kwargs)  # type: ignore[misc]
        except TypeError:
            if kwargs:
                logger.debug(
                    "GraphTranslator.arun_capabilities does not accept kwargs; ignoring: %s",
                    sorted(kwargs.keys()),
                )
            caps = await self._translator.arun_capabilities()

        return graph_capabilities_to_dict(caps)

    @with_graph_error_context("get_schema_sync")
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
        _ensure_not_in_event_loop("get_schema")
        ctx = self._build_ctx(
            context=context,
            settings=settings,
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
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        """
        Sync health check wrapper.

        Delegates async→sync bridging to GraphTranslator.
        """
        _ensure_not_in_event_loop("health")
        ctx = self._build_ctx(
            context=context,
            settings=settings,
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
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> QueryResult:
        """
        Execute a non-streaming graph query (sync).

        Returns the underlying `QueryResult`.
        """
        _ensure_not_in_event_loop("query")
        validate_graph_query(query, operation="query", error_code=ErrorCodes.INVALID_QUERY)

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
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> QueryResult:
        """
        Execute a non-streaming graph query (async).

        Returns the underlying `QueryResult`.
        """
        validate_graph_query(query, operation="aquery", error_code=ErrorCodes.INVALID_QUERY)

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
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[QueryChunk]:
        """
        Execute a streaming graph query (sync), yielding `QueryChunk` items.

        Delegates streaming orchestration to GraphTranslator, which uses
        SyncStreamBridge under the hood.
        """
        _ensure_not_in_event_loop("stream_query")
        validate_graph_query(
            query,
            operation="stream_query",
            error_code=ErrorCodes.INVALID_QUERY,
        )

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
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> AsyncIterator[QueryChunk]:
        """
        Execute a streaming graph query (async), yielding `QueryChunk` items.

        Note:
        - Some GraphTranslator implementations return an AsyncIterator directly.
        - Others return an awaitable that resolves to an AsyncIterator.
        This adapter supports both shapes to prevent brittle coupling.
        """
        validate_graph_query(
            query,
            operation="astream_query",
            error_code=ErrorCodes.INVALID_QUERY,
        )

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
        framework_ctx = self._framework_ctx(
            operation="stream_query",
            namespace=namespace,
        )

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
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        """
        Sync wrapper for upserting nodes via GraphTranslator.
        """
        _ensure_not_in_event_loop("upsert_nodes")
        validate_upsert_nodes_spec(spec)

        ctx = self._build_ctx(
            context=context,
            settings=settings,
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
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        """
        Async wrapper for upserting nodes via GraphTranslator.
        """
        validate_upsert_nodes_spec(spec)

        ctx = self._build_ctx(
            context=context,
            settings=settings,
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
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        """
        Sync wrapper for upserting edges via GraphTranslator.
        """
        _ensure_not_in_event_loop("upsert_edges")
        edges = self._validate_upsert_edges_spec(spec)

        ctx = self._build_ctx(
            context=context,
            settings=settings,
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
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        """
        Async wrapper for upserting edges via GraphTranslator.
        """
        edges = self._validate_upsert_edges_spec(spec)

        ctx = self._build_ctx(
            context=context,
            settings=settings,
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
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        """
        Sync wrapper for deleting nodes via GraphTranslator.
        """
        _ensure_not_in_event_loop("delete_nodes")
        ctx = self._build_ctx(
            context=context,
            settings=settings,
            extra_context=extra_context,
        )
        framework_ctx = self._framework_ctx(
            operation="delete_nodes",
            namespace=getattr(spec, "namespace", None),
        )

        raw_filter_or_ids: Any = self._select_filter_or_ids(
            spec_filter=spec.filter,
            spec_ids=spec.ids,
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
        framework_ctx = self._framework_ctx(
            operation="delete_nodes",
            namespace=getattr(spec, "namespace", None),
        )

        raw_filter_or_ids: Any = self._select_filter_or_ids(
            spec_filter=spec.filter,
            spec_ids=spec.ids,
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
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        """
        Sync wrapper for deleting edges via GraphTranslator.
        """
        _ensure_not_in_event_loop("delete_edges")
        ctx = self._build_ctx(
            context=context,
            settings=settings,
            extra_context=extra_context,
        )
        framework_ctx = self._framework_ctx(
            operation="delete_edges",
            namespace=getattr(spec, "namespace", None),
        )

        raw_filter_or_ids: Any = self._select_filter_or_ids(
            spec_filter=spec.filter,
            spec_ids=spec.ids,
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
        framework_ctx = self._framework_ctx(
            operation="delete_edges",
            namespace=getattr(spec, "namespace", None),
        )

        raw_filter_or_ids: Any = self._select_filter_or_ids(
            spec_filter=spec.filter,
            spec_ids=spec.ids,
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
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BulkVerticesResult:
        """
        Sync wrapper for bulk_vertices via GraphTranslator.
        """
        _ensure_not_in_event_loop("bulk_vertices")
        ctx = self._build_ctx(
            context=context,
            settings=settings,
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
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> TraversalResult:
        """
        Sync wrapper for graph traversal.

        Builds a raw traversal request and delegates to GraphTranslator.
        """
        _ensure_not_in_event_loop("traversal")

        ctx = self._build_ctx(
            context=context,
            settings=settings,
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
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> TraversalResult:
        """
        Async wrapper for graph traversal.
        """
        ctx = self._build_ctx(
            context=context,
            settings=settings,
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
    # Transaction + Batch (sync + async)
    # ------------------------------------------------------------------ #

    @with_graph_error_context("transaction_sync")
    def transaction(
        self,
        ops: List[BatchOperation],
        *,
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BatchResult:
        """
        Sync wrapper for transactional batch operations.

        Translates `BatchOperation` dataclasses into the raw mapping shape
        expected by GraphTranslator and returns the underlying `BatchResult`.
        """
        _ensure_not_in_event_loop("transaction")

        # Reuse batch validation; semantics are still a list of BatchOperation.
        validate_batch_operations(
            ops,
            operation="transaction",
            error_code=ErrorCodes.INVALID_BATCH_OPS,
        )

        ctx = self._build_ctx(
            context=context,
            settings=settings,
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
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
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
            context=context,
            settings=settings,
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

    # ------------------------------------------------------------------ #
    # Batch (sync + async)
    # ------------------------------------------------------------------ #

    @with_graph_error_context("batch_sync")
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
        _ensure_not_in_event_loop("batch")
        validate_batch_operations(
            ops,
            operation="batch",
            error_code=ErrorCodes.INVALID_BATCH_OPS,
        )

        ctx = self._build_ctx(
            context=context,
            settings=settings,
            extra_context=extra_context,
        )
        raw_batch_ops: List[Mapping[str, Any]] = [
            {"op": op.op, "args": dict(op.args or {})} for op in ops
        ]
        framework_ctx = self._framework_ctx(operation="batch")

        result = self._translator.batch(
            raw_batch_ops,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
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
        context: Optional[Any] = None,
        settings: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BatchResult:
        """
        Async wrapper for batch operations via GraphTranslator.
        """
        validate_batch_operations(
            ops,
            operation="abatch",
            error_code=ErrorCodes.INVALID_BATCH_OPS,
        )

        ctx = self._build_ctx(
            context=context,
            settings=settings,
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


# ------------------------------------------------------------------ #
# Optional Semantic Kernel dependency – only needed for SK plugin integration
# ------------------------------------------------------------------ #

try:
    # The package name for Semantic Kernel in Python is typically "semantic_kernel".
    # We keep the import optional so this adapter can be imported without SK installed.
    import semantic_kernel as _semantic_kernel  # type: ignore  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    _semantic_kernel = None  # type: ignore[assignment]


# ------------------------------------------------------------------ #
# Optional Semantic Kernel Plugin wrapper
# ------------------------------------------------------------------ #

if _semantic_kernel is not None:

    class CorpusSemanticKernelPlugin:
        """
        Semantic Kernel–friendly plugin wrapper backed by CorpusSemanticKernelGraphClient.

        Purpose
        -------
        This wrapper mirrors the "soft integration" pattern used by other framework
        adapters (e.g., LlamaIndex GraphStore):

        - The core adapter remains usable without Semantic Kernel installed.
        - If Semantic Kernel is installed, this plugin can be registered into a
          Kernel/plugin system for integration and integration testing.

        Design notes
        ------------
        - This class is deliberately lightly opinionated: it exposes graph operations
          in a SK-friendly shape, but does not assume a specific SK plugin decorator
          or registration API, which can vary across SK versions.
        - Methods accept `context` and `settings` as optional `Any` and forward them
          through to the underlying client, preserving OperationContext propagation.
        """

        def __init__(
            self,
            client: "CorpusSemanticKernelGraphClient",
            *,
            namespace: Optional[str] = None,
        ) -> None:
            """
            Parameters
            ----------
            client:
                Underlying CorpusSemanticKernelGraphClient instance.
            namespace:
                Optional default namespace to pass explicitly for plugin operations.
                This mirrors the behavior of other framework wrapper classes: a plugin
                may be configured with a stable namespace rather than relying on caller
                defaults.
            """
            self._client = client
            self._namespace = namespace

        @property
        def client(self) -> Any:
            """Expose underlying client for advanced callers and integration tests."""
            return self._client

        # ----------------------------- #
        # Capabilities / schema / health
        # ----------------------------- #

        def capabilities(self, **kwargs: Any) -> Mapping[str, Any]:
            """
            Capabilities passthrough.

            kwargs are accepted and forwarded for forward compatibility, consistent
            with the underlying client API.
            """
            return self._client.capabilities(**kwargs)

        async def acapabilities(self, **kwargs: Any) -> Mapping[str, Any]:
            """Async capabilities passthrough."""
            return await self._client.acapabilities(**kwargs)

        def get_schema(
            self,
            *,
            context: Optional[Any] = None,
            settings: Optional[Any] = None,
            extra_context: Optional[Mapping[str, Any]] = None,
        ) -> GraphSchema:
            """Schema passthrough."""
            return self._client.get_schema(
                context=context,
                settings=settings,
                extra_context=extra_context,
            )

        async def aget_schema(
            self,
            *,
            context: Optional[Any] = None,
            settings: Optional[Any] = None,
            extra_context: Optional[Mapping[str, Any]] = None,
        ) -> GraphSchema:
            """Async schema passthrough."""
            return await self._client.aget_schema(
                context=context,
                settings=settings,
                extra_context=extra_context,
            )

        def health(
            self,
            *,
            context: Optional[Any] = None,
            settings: Optional[Any] = None,
            extra_context: Optional[Mapping[str, Any]] = None,
        ) -> Mapping[str, Any]:
            """Health passthrough."""
            return self._client.health(
                context=context,
                settings=settings,
                extra_context=extra_context,
            )

        async def ahealth(
            self,
            *,
            context: Optional[Any] = None,
            settings: Optional[Any] = None,
            extra_context: Optional[Mapping[str, Any]] = None,
        ) -> Mapping[str, Any]:
            """Async health passthrough."""
            return await self._client.ahealth(
                context=context,
                settings=settings,
                extra_context=extra_context,
            )

        # ----------------------------- #
        # Query / streaming
        # ----------------------------- #

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
            Query passthrough.

            Namespace resolution:
            - If `namespace` is passed explicitly, it is used.
            - Otherwise, the plugin-level configured namespace is used (if set).
            - Otherwise, the client default namespace behavior applies.
            """
            effective_namespace = namespace if namespace is not None else self._namespace
            return self._client.query(
                query,
                params=params,
                dialect=dialect,
                namespace=effective_namespace,
                timeout_ms=timeout_ms,
                context=context,
                settings=settings,
                extra_context=extra_context,
            )

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
            """Async query passthrough (same namespace resolution rules as query)."""
            effective_namespace = namespace if namespace is not None else self._namespace
            return await self._client.aquery(
                query,
                params=params,
                dialect=dialect,
                namespace=effective_namespace,
                timeout_ms=timeout_ms,
                context=context,
                settings=settings,
                extra_context=extra_context,
            )

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
            """Streaming query passthrough (sync)."""
            effective_namespace = namespace if namespace is not None else self._namespace
            return self._client.stream_query(
                query,
                params=params,
                dialect=dialect,
                namespace=effective_namespace,
                timeout_ms=timeout_ms,
                context=context,
                settings=settings,
                extra_context=extra_context,
            )

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
            """Streaming query passthrough (async)."""
            effective_namespace = namespace if namespace is not None else self._namespace
            async for chunk in self._client.astream_query(
                query,
                params=params,
                dialect=dialect,
                namespace=effective_namespace,
                timeout_ms=timeout_ms,
                context=context,
                settings=settings,
                extra_context=extra_context,
            ):
                yield chunk

        # ----------------------------- #
        # Resource management
        # ----------------------------- #

        def close(self) -> None:
            """Close passthrough for plugin users."""
            self._client.close()

        async def aclose(self) -> None:
            """Async close passthrough for plugin users."""
            await self._client.aclose()

else:
    # Semantic Kernel not installed – provide a stub that fails loudly at runtime
    class CorpusSemanticKernelPlugin:  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "semantic_kernel is not installed; CorpusSemanticKernelPlugin is unavailable. "
                "Install semantic-kernel to use this integration."
            )


__all__ = [
    "SemanticKernelGraphClientProtocol",
    "CorpusSemanticKernelGraphClient",
    "SemanticKernelGraphFrameworkTranslator",
    "CorpusSemanticKernelPlugin",
    "ErrorCodes",
    "with_graph_error_context",
    "with_async_graph_error_context",
    "with_error_context",
    "with_async_error_context",
]
