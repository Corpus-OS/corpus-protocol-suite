# corpus_sdk/graph/framework_adapters/crewai.py
# SPDX-License-Identifier: Apache-2.0

"""
CrewAI adapter for Corpus Graph protocol.

This module exposes a Corpus `GraphProtocolV1` implementation as a
CrewAI-friendly client, with:

- Sync + async query APIs
- Sync + async streaming query APIs
- Proper integration with Corpus GraphProtocolV1
- OperationContext propagation derived from CrewAI tasks / metadata
- Error-context enrichment for observability and debugging
- Orchestration, translation, and async→sync bridging via GraphTranslator

Design philosophy
-----------------
- Protocol-first: CrewAI is a thin skin over the Corpus graph adapter.
- All heavy lifting (deadlines, breakers, rate limiting, caching, etc.) lives
  in the underlying `BaseGraphAdapter` / `GraphProtocolV1` implementation.
- This layer focuses on:
    * Translating CrewAI Task → OperationContext
    * Building raw query / mutation shapes for GraphTranslator
    * Delegating all sync/async and streaming orchestration to GraphTranslator

Responsibilities
----------------
- Provide a convenient, CrewAI-oriented client for graph operations
- Keep all graph operations going through `GraphTranslator` so that
  async→sync bridging, streaming, and error-context logic are centralized
- Preserve protocol-level types (`QueryResult`, `QueryChunk`, etc.) for
  CrewAI callers

Non-responsibilities
--------------------
- Backend-specific graph behavior (lives in graph adapters)
- CrewAI agent orchestration and task logic
- MMR and diversification details (handled inside GraphTranslator)

Compatibility notes
-------------------
- CrewAI is an **optional dependency**. This module intentionally does not
  hard-import CrewAI packages at import time.
- Real CrewAI integration is provided through **soft-imported tool helpers**
  at the bottom of this file (e.g., `create_crewai_graph_tools()`).
  Importing this module does not require CrewAI to be installed.
- When CrewAI is installed, you can build CrewAI-native `BaseTool` wrappers
  (from `crewai.tools.base_tool`) around this client to run true end-to-end
  integration tests in CrewAI agent environments.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import inspect
import json
import logging
import threading
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
    Sequence,
    TypeVar,
)

from corpus_sdk.core.context_translation import (
    from_crewai as core_ctx_from_crewai,
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

# Type variables for decorators
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
    # Keeping these as explicit symbolic strings avoids accidental drift.
    INVALID_QUERY = "INVALID_QUERY"
    INVALID_BATCH_OPS = "INVALID_BATCH_OPS"

    # Optional-tool-specific validation codes. These are not exposed as protocol errors;
    # they exist to keep tool behavior clear and debuggable when LLMs pass bad inputs.
    INVALID_TOOL_PARAM = "INVALID_TOOL_PARAM"


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
    for the CrewAI framework.
    """
    return create_graph_error_context_decorator(
        framework="crewai",
        is_async=False,
    )(operation=operation, **static_context)


def with_async_graph_error_context(
    operation: str,
    **static_context: Any,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for async methods with rich dynamic context extraction.

    Thin wrapper over the shared `create_graph_error_context_decorator`
    for the CrewAI framework.
    """
    return create_graph_error_context_decorator(
        framework="crewai",
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


def _is_running_event_loop() -> bool:
    """
    Return True if called while an asyncio event loop is running in this thread.

    This is used by optional CrewAI tool helpers to provide safe sync execution
    in environments that execute tools from within async agent runtimes.
    """
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def _ensure_not_in_event_loop(sync_api_name: str) -> None:
    """
    Prevent deadlocks from calling sync graph APIs in async contexts.

    This mirrors the embedding adapters' safety guard and gives a clear,
    actionable error if a sync method is used from inside a running event
    loop (e.g., Jupyter, FastAPI, or async CrewAI agents).
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop: safe to call sync API.
        return
    raise RuntimeError(
        f"{sync_api_name} was called from inside an active asyncio event loop. "
        f"Use the async variant instead (e.g. 'await a{sync_api_name}(...)'). "
        f"[{ErrorCodes.SYNC_WRAPPER_CALLED_IN_EVENT_LOOP}]"
    )


def _json_safe_snapshot(value: Any, *, max_items: int = 200, max_str: int = 10_000) -> Any:
    """
    Best-effort conversion into a JSON-ish snapshot for tool-return values and logs.

    Security / correctness:
    - Limits container sizes to avoid memory bloat.
    - Truncates long strings.
    - Falls back to repr() for unknown objects.

    NOTE:
    - This helper is used only in optional CrewAI tool helpers below.
    - Core protocol logic continues to return protocol-level types (QueryResult, etc.).
    """
    try:
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return value if len(value) <= max_str else value[:max_str] + "…"
        if isinstance(value, Mapping):
            out: Dict[str, Any] = {}
            for i, (k, v) in enumerate(value.items()):
                if i >= max_items:
                    out["…"] = f"truncated after {max_items} items"
                    break
                out[str(k)] = _json_safe_snapshot(v, max_items=max_items, max_str=max_str)
            return out
        if isinstance(value, (list, tuple)):
            out_list: List[Any] = []
            for i, v in enumerate(value):
                if i >= max_items:
                    out_list.append(f"… truncated after {max_items} items")
                    break
                out_list.append(_json_safe_snapshot(v, max_items=max_items, max_str=max_str))
            return out_list
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            return _json_safe_snapshot(to_dict(), max_items=max_items, max_str=max_str)
        # Fallback: ensure representable
        return repr(value)
    except Exception:  # noqa: BLE001
        return {"repr": repr(value)}


async def _normalize_async_iterator(aiter_or_awaitable: Any) -> AsyncIterator[Any]:
    """
    Normalize streaming return shapes into a concrete AsyncIterator.

    Some GraphTranslator implementations return:
      - an AsyncIterator directly, OR
      - an awaitable that resolves to an AsyncIterator.

    This helper makes the adapter resilient to those implementation choices while:
      - keeping runtime behavior unchanged, and
      - providing correct typing (always returns an AsyncIterator when awaited).

    IMPORTANT:
      This function intentionally does not enforce strict type checks on the resolved
      value (best-effort). If a backend/translator returns an invalid shape, the
      subsequent `async for` will raise, and error-context decorators will attach
      observability context as designed.
    """
    if inspect.isawaitable(aiter_or_awaitable):
        resolved = await aiter_or_awaitable
        return resolved  # type: ignore[return-value]
    return aiter_or_awaitable  # type: ignore[return-value]


# Dedicated, bounded executor for tool-in-event-loop compatibility.
# Used only by create_crewai_graph_tools(...) to keep tool runs safe in async CrewAI runtimes.
_CREWAI_TOOL_BRIDGE_EXECUTOR: Optional[concurrent.futures.ThreadPoolExecutor] = None
_CREWAI_TOOL_BRIDGE_EXECUTOR_LOCK = threading.Lock()


def _shutdown_crewai_tool_bridge_executor() -> None:
    """
    Best-effort shutdown of the optional tool bridge executor.

    Rationale:
      The executor is only used for optional CrewAI tool helpers to run sync
      graph calls from within running asyncio event loops.

    Why shutdown matters:
      In test suites and short-lived runtimes, a non-daemon thread pool can
      outlive intended lifetimes and appear as a resource leak.

    Safety:
      - This function is idempotent.
      - Shutdown failures are swallowed to avoid impacting application teardown.
      - If tools are used again after shutdown, the executor is recreated.
    """
    global _CREWAI_TOOL_BRIDGE_EXECUTOR  # noqa: PLW0603
    with _CREWAI_TOOL_BRIDGE_EXECUTOR_LOCK:
        ex = _CREWAI_TOOL_BRIDGE_EXECUTOR
        _CREWAI_TOOL_BRIDGE_EXECUTOR = None

    if ex is None:
        return

    try:
        # cancel_futures=True is supported on modern Python versions. If unsupported,
        # we swallow exceptions (best-effort cleanup) to avoid impacting teardown.
        ex.shutdown(wait=False, cancel_futures=True)  # type: ignore[call-arg]
    except Exception:
        logger.debug("Failed to shutdown CrewAI tool bridge executor", exc_info=True)


# Ensure we do not leak threads across interpreter shutdown in test harnesses.
atexit.register(_shutdown_crewai_tool_bridge_executor)


def _run_blocking_in_crewai_tool_thread(fn: Callable[[], T]) -> T:
    """
    Run a blocking graph call in a bounded thread pool.

    Why:
      CrewAI tool execution may occur inside an async runtime (event loop running),
      but graph client sync APIs deliberately refuse to run inside event loops.

    Safety/performance:
      - bounded pool (max_workers=4) to prevent unbounded thread creation
      - used only in the optional tool integration path
      - atexit hook and client.close() invoke a best-effort shutdown to avoid leaks
    """
    global _CREWAI_TOOL_BRIDGE_EXECUTOR  # noqa: PLW0603
    with _CREWAI_TOOL_BRIDGE_EXECUTOR_LOCK:
        if _CREWAI_TOOL_BRIDGE_EXECUTOR is None:
            _CREWAI_TOOL_BRIDGE_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
                max_workers=4,
                thread_name_prefix="corpus-crewai-tool",
            )
        executor = _CREWAI_TOOL_BRIDGE_EXECUTOR

    # NOTE: .result() is intentional: tools are expected to block (sync interface).
    return executor.submit(fn).result()


def _coerce_bounded_positive_int(
    value: Any,
    *,
    name: str,
    default: int,
    min_value: int = 1,
    max_value: int = 100,
) -> int:
    """
    Convert a possibly-LLM-provided value into a safe bounded positive int.

    Why:
      Tool parameters are frequently produced by LLMs and can be strings ("25"),
      floats ("25.0"), or invalid values (None, negative numbers).

    Behavior:
      - If conversion fails, returns the provided default.
      - If converted value is out of bounds, clamps to [min_value, max_value].
    """
    try:
        # Allow strings and floats that represent integers ("25", 25.0).
        ivalue = int(value)
    except Exception:  # noqa: BLE001
        logger.debug(
            "[%s] Invalid tool param %s=%r; defaulting to %d",
            ErrorCodes.INVALID_TOOL_PARAM,
            name,
            value,
            default,
        )
        return default

    if ivalue < min_value:
        return min_value
    if ivalue > max_value:
        return max_value
    return ivalue


# --------------------------------------------------------------------------- #
# Framework translator (public, reusable, overridable)
# --------------------------------------------------------------------------- #


class CrewAIGraphFrameworkTranslator(DefaultGraphFrameworkTranslator):
    """
    CrewAI-specific GraphFrameworkTranslator.

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


class CrewAIGraphClientProtocol(Protocol):
    """
    Protocol representing the minimal CrewAI-aware graph client interface
    implemented by this module.

    This structural protocol allows callers to type against the graph client
    without depending on the concrete `CorpusCrewAIGraphClient` class.
    """

    # Capabilities / schema / health -------------------------------------

    def capabilities(self, **kwargs) -> Dict[str, Any]:
        ...

    async def acapabilities(self, **kwargs) -> Dict[str, Any]:
        ...

    def get_schema(
        self,
        *,
        task: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> GraphSchema:
        ...

    async def aget_schema(
        self,
        *,
        task: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> GraphSchema:
        ...

    def health(
        self,
        *,
        task: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        ...

    async def ahealth(
        self,
        *,
        task: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
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
        task: Optional[Any] = None,
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
        task: Optional[Any] = None,
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
        task: Optional[Any] = None,
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
        task: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> AsyncIterator[QueryChunk]:
        ...

    # Upsert --------------------------------------------------------------

    def upsert_nodes(
        self,
        spec: UpsertNodesSpec,
        *,
        task: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        ...

    async def aupsert_nodes(
        self,
        spec: UpsertNodesSpec,
        *,
        task: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        ...

    def upsert_edges(
        self,
        spec: UpsertEdgesSpec,
        *,
        task: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        ...

    async def aupsert_edges(
        self,
        spec: UpsertEdgesSpec,
        *,
        task: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        ...

    # Delete --------------------------------------------------------------

    def delete_nodes(
        self,
        spec: DeleteNodesSpec,
        *,
        task: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        ...

    async def adelete_nodes(
        self,
        spec: DeleteNodesSpec,
        *,
        task: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        ...

    def delete_edges(
        self,
        spec: DeleteEdgesSpec,
        *,
        task: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        ...

    async def adelete_edges(
        self,
        spec: DeleteEdgesSpec,
        *,
        task: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        ...

    # Bulk / batch --------------------------------------------------------

    def bulk_vertices(
        self,
        spec: BulkVerticesSpec,
        *,
        task: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BulkVerticesResult:
        ...

    async def abulk_vertices(
        self,
        spec: BulkVerticesSpec,
        *,
        task: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BulkVerticesResult:
        ...

    def batch(
        self,
        ops: List[BatchOperation],
        *,
        task: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BatchResult:
        ...

    async def abatch(
        self,
        ops: List[BatchOperation],
        *,
        task: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BatchResult:
        ...

    # Resource lifecycle --------------------------------------------------

    def close(self) -> None:
        ...

    async def aclose(self) -> None:
        ...


class CorpusCrewAIGraphClient:
    """
    CrewAI-oriented client wrapper around a Corpus `GraphProtocolV1`.

    This is a thin integration layer that:

    - Translates CrewAI Task / metadata into a Corpus `OperationContext`
      using `core_ctx_from_crewai`.
    - Uses `GraphTranslator` (with a CrewAI-specific framework translator) to:
        * Build Graph*Spec objects from simple inputs
        * Execute sync + async graph operations
        * Orchestrate streaming with proper cancellation and error handling
    - Delegates all async→sync bridging and streaming glue to GraphTranslator.
    - Attaches rich error context (`attach_context`) on this layer with
      CrewAI-specific hints when failures occur.
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
        Initialize a CrewAI-oriented graph client.

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
            Optional GraphFrameworkTranslator override. When not provided,
            `CrewAIGraphFrameworkTranslator` is used.
        """
        if adapter is not None and graph_adapter is not None:
            raise TypeError("Provide only one of 'adapter' or 'graph_adapter', not both")

        resolved_adapter: Any = graph_adapter if graph_adapter is not None else adapter
        if resolved_adapter is None:
            raise TypeError("adapter must be a GraphProtocolV1-compatible graph adapter")

        # Minimal duck-type check consistent with other framework adapters:
        # we require the adapter to present a GraphProtocol-like surface.
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
        self._framework_translator_override: Optional[GraphFrameworkTranslator] = framework_translator

        # Resource management flags (idempotent close semantics)
        self._closed: bool = False
        self._aclosed: bool = False

        logger.info(
            "CorpusCrewAIGraphClient initialized (default_dialect=%r, default_namespace=%r, framework_version=%r)",
            self._default_dialect,
            self._default_namespace,
            self._framework_version,
        )

    # ------------------------------------------------------------------ #
    # Resource Management (Context Managers)
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "CorpusCrewAIGraphClient":
        """Support context manager protocol for resource cleanup."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Clean up resources when exiting context."""
        self.close()

    async def __aenter__(self) -> "CorpusCrewAIGraphClient":
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

        # Also tear down the optional tool bridge executor. This is safe and idempotent:
        # - It prevents thread leaks in short-lived runtimes (tests).
        # - If tools are invoked again later, the executor will be recreated.
        _shutdown_crewai_tool_bridge_executor()

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
                # As with close(), shut down optional tool bridge resources.
                _shutdown_crewai_tool_bridge_executor()
                return
            except Exception:
                logger.debug("Failed to async-close graph adapter", exc_info=True)

        # Fallback to sync close if we haven't already done so.
        if not self._closed:
            self.close()

    # ------------------------------------------------------------------ #
    # Translator (lazy, cached) – mirrors AutoGen adapter pattern
    # ------------------------------------------------------------------ #

    @cached_property
    def _translator(self) -> GraphTranslator:
        """
        Lazily construct and cache the `GraphTranslator`.

        Uses `cached_property` for thread safety and performance, mirroring
        the embedding / AutoGen adapter patterns.
        """
        framework_translator: GraphFrameworkTranslator = (
            self._framework_translator_override or CrewAIGraphFrameworkTranslator()
        )
        return create_graph_translator(
            adapter=self._graph,
            framework="crewai",
            translator=framework_translator,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_ctx(
        self,
        *,
        task: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Optional[OperationContext]:
        """
        Build an OperationContext from CrewAI-style inputs.

        Expected inputs
        ----------------
        - task: CrewAI Task instance (optional)
        - extra_context: Optional mapping merged into attrs (best effort)

        If both are None/empty, returns None and lets downstream helpers
        construct an "empty" OperationContext as needed.

        Context translation is best-effort: failures are logged and
        attached for observability, but graph operations may still
        proceed without an OperationContext.
        """
        extra = dict(extra_context or {})

        if task is None and not extra:
            return None

        try:
            ctx_candidate = core_ctx_from_crewai(
                task,
                framework_version=self._framework_version,
                **extra,
            )
        except Exception as exc:
            logger.warning(
                "[%s] Failed to build OperationContext from CrewAI inputs; "
                "proceeding without OperationContext. task_type=%s extra_keys=%s",
                ErrorCodes.BAD_OPERATION_CONTEXT,
                type(task).__name__ if task is not None else "None",
                list(extra.keys()),
            )
            attach_context(
                exc,
                framework="crewai",
                operation="context_translation",
                error_code=ErrorCodes.BAD_OPERATION_CONTEXT,
                task_type=type(task).__name__ if task is not None else "None",
                extra_context_keys=list(extra.keys()),
            )
            return None

        if _looks_like_operation_context(ctx_candidate):
            # Enrich OperationContext.attrs with framework metadata so that
            # downstream observability and GraphTranslator always see the
            # framework identity and version without depending on upstream
            # context translation details.
            #
            # Implementation note:
            # - We avoid mutating the original ctx object to reduce surprising
            #   side effects if callers reuse/freeze OperationContext instances.
            attrs_obj = getattr(ctx_candidate, "attrs", {}) or {}
            attrs: Dict[str, Any] = dict(attrs_obj) if not isinstance(attrs_obj, dict) else dict(attrs_obj)
            attrs.setdefault("framework", "crewai")
            if self._framework_version is not None:
                attrs.setdefault("framework_version", self._framework_version)

            return OperationContext(
                request_id=getattr(ctx_candidate, "request_id", None),
                idempotency_key=getattr(ctx_candidate, "idempotency_key", None),
                deadline_ms=getattr(ctx_candidate, "deadline_ms", None),
                traceparent=getattr(ctx_candidate, "traceparent", None),
                tenant=getattr(ctx_candidate, "tenant", None),
                attrs=attrs,
            )

        logger.warning(
            "[%s] from_crewai returned non-OperationContext-like type: %s. "
            "Ignoring OperationContext.",
            ErrorCodes.BAD_OPERATION_CONTEXT,
            type(ctx_candidate).__name__,
        )
        return None

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
            "framework": "crewai",
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
        CrewAI-local validation for edge upsert specs.

        Similar to the AutoGen adapter:
        - edges must not be None
        - edges must be iterable and non-empty
        - each edge must have required structural fields
        - properties (if present) must be JSON-serializable

        IMPORTANT:
        - This method intentionally avoids mutating `spec.edges`.
          Mutating input specs is an avoidable footgun if specs become frozen,
          reused across calls, or treated as immutable by upstream callers.
        - The validated, materialized edge list is returned to the caller so
          downstream execution can safely consume it without re-iterating.
        """
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

            if hasattr(edge, "properties") and edge.properties is not None:
                try:
                    json.dumps(edge.properties)
                except (TypeError, ValueError) as e:
                    raise BadRequest(
                        f"Edge at index {idx} properties must be JSON-serializable: {e}"
                    )

        # NOTE: We intentionally do NOT assign `spec.edges = edges`.
        # Returning the validated list avoids side effects while preserving behavior.
        return edges

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
        Sync wrapper around capabilities, delegating async→sync bridging
        to GraphTranslator.

        Compatibility note:
        - Accepts arbitrary **kwargs and intentionally ignores unknown keys to
          support "rich context" calling styles in conformance environments.
        """
        _ensure_not_in_event_loop("capabilities")

        caps = self._translator.capabilities()
        return graph_capabilities_to_dict(caps)

    @with_async_graph_error_context("capabilities_async")
    async def acapabilities(self, **kwargs) -> Dict[str, Any]:
        """
        Async capabilities accessor.

        We delegate to GraphTranslator for consistency, then normalize to a
        simple dict for CrewAI consumption.

        Compatibility note:
        - Accepts arbitrary **kwargs and intentionally ignores unknown keys to
          support "rich context" calling styles in conformance environments.
        """
        caps = await self._translator.arun_capabilities()
        return graph_capabilities_to_dict(caps)

    @with_graph_error_context("get_schema_sync")
    def get_schema(
        self,
        *,
        task: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> GraphSchema:
        """
        Sync wrapper around `graph_adapter.get_schema(...)`.

        Delegates to GraphTranslator so that async→sync bridging and
        error-context handling are centralized.
        """
        _ensure_not_in_event_loop("get_schema")

        ctx = self._build_ctx(task=task, extra_context=extra_context)
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
        task: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> GraphSchema:
        """
        Async wrapper around `graph_adapter.get_schema(...)`.

        Delegates to GraphTranslator.
        """
        ctx = self._build_ctx(task=task, extra_context=extra_context)
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
        task: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Health check (sync).

        Uses GraphTranslator for consistency with other operations.
        """
        _ensure_not_in_event_loop("health")

        ctx = self._build_ctx(task=task, extra_context=extra_context)
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
        return dict(mapping_result)

    @with_async_graph_error_context("health_async")
    async def ahealth(
        self,
        *,
        task: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Health check (async).

        Uses GraphTranslator for consistency with other operations.
        """
        ctx = self._build_ctx(task=task, extra_context=extra_context)
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
        task: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> QueryResult:
        """
        Execute a non-streaming graph query (sync).

        Returns the underlying `QueryResult` from the GraphProtocol adapter.
        """
        _ensure_not_in_event_loop("query")

        validate_graph_query(query, operation="query", error_code=ErrorCodes.INVALID_QUERY)
        self._validate_query_params(params)

        ctx = self._build_ctx(task=task, extra_context=extra_context)
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
            # Dialect fallback mirrors the AutoGen adapter behavior:
            # some backends reject unknown dialects but can execute without it.
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
        task: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> QueryResult:
        """
        Execute a non-streaming graph query (async).

        Returns the underlying `QueryResult`.
        """
        validate_graph_query(query, operation="aquery", error_code=ErrorCodes.INVALID_QUERY)
        self._validate_query_params(params)

        ctx = self._build_ctx(task=task, extra_context=extra_context)
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
        task: Optional[Any] = None,
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
        self._validate_query_params(params)

        ctx = self._build_ctx(task=task, extra_context=extra_context)
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
        task: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> AsyncIterator[QueryChunk]:
        """
        Execute a streaming graph query (async), yielding `QueryChunk` items.

        Compatibility note:
        Some GraphTranslator implementations return:
          - an AsyncIterator directly, OR
          - an awaitable that resolves to an AsyncIterator.
        This method supports both forms to avoid brittle tests and allow
        translator evolution without breaking framework adapters.
        """
        validate_graph_query(query, operation="astream_query", error_code=ErrorCodes.INVALID_QUERY)
        self._validate_query_params(params)

        ctx = self._build_ctx(task=task, extra_context=extra_context)
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

        # Normalize translator return shapes into a concrete AsyncIterator with correct typing.
        aiter = await _normalize_async_iterator(aiter_or_awaitable)

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
        task: Optional[Any] = None,
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

        ctx = self._build_ctx(task=task, extra_context=extra_context)
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
        task: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        """
        Async wrapper for upserting nodes.
        """
        validate_upsert_nodes_spec(spec)

        ctx = self._build_ctx(task=task, extra_context=extra_context)
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
        task: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        """
        Sync wrapper for upserting edges.
        """
        _ensure_not_in_event_loop("upsert_edges")

        validated_edges = self._validate_upsert_edges_spec(spec)

        ctx = self._build_ctx(task=task, extra_context=extra_context)
        framework_ctx = self._framework_ctx(
            operation="upsert_edges",
            namespace=getattr(spec, "namespace", None),
        )

        result = self._translator.upsert_edges(
            validated_edges,
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
        task: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        """
        Async wrapper for upserting edges.
        """
        validated_edges = self._validate_upsert_edges_spec(spec)

        ctx = self._build_ctx(task=task, extra_context=extra_context)
        framework_ctx = self._framework_ctx(
            operation="upsert_edges",
            namespace=getattr(spec, "namespace", None),
        )

        result = await self._translator.arun_upsert_edges(
            validated_edges,
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
        task: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        """
        Sync wrapper for deleting nodes.

        Uses DeleteNodesSpec to derive either an ID list or a filter
        expression for the GraphTranslator.
        """
        _ensure_not_in_event_loop("delete_nodes")

        ctx = self._build_ctx(task=task, extra_context=extra_context)
        framework_ctx = self._framework_ctx(
            operation="delete_nodes",
            namespace=getattr(spec, "namespace", None),
        )

        if spec.filter is not None:
            raw_filter_or_ids: Any = spec.filter
        else:
            ids = list(spec.ids or [])
            if not ids:
                raise BadRequest(
                    "DeleteNodesSpec must specify either filter or non-empty ids",
                    code=ErrorCodes.BAD_ADAPTER_RESULT,
                )
            raw_filter_or_ids = ids

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
        task: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        """
        Async wrapper for deleting nodes.
        """
        ctx = self._build_ctx(task=task, extra_context=extra_context)
        framework_ctx = self._framework_ctx(
            operation="delete_nodes",
            namespace=getattr(spec, "namespace", None),
        )

        if spec.filter is not None:
            raw_filter_or_ids: Any = spec.filter
        else:
            ids = list(spec.ids or [])
            if not ids:
                raise BadRequest(
                    "DeleteNodesSpec must specify either filter or non-empty ids",
                    code=ErrorCodes.BAD_ADAPTER_RESULT,
                )
            raw_filter_or_ids = ids

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
        task: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        """
        Sync wrapper for deleting edges.
        """
        _ensure_not_in_event_loop("delete_edges")

        ctx = self._build_ctx(task=task, extra_context=extra_context)
        framework_ctx = self._framework_ctx(
            operation="delete_edges",
            namespace=getattr(spec, "namespace", None),
        )

        if spec.filter is not None:
            raw_filter_or_ids: Any = spec.filter
        else:
            ids = list(spec.ids or [])
            if not ids:
                raise BadRequest(
                    "DeleteEdgesSpec must specify either filter or non-empty ids",
                    code=ErrorCodes.BAD_ADAPTER_RESULT,
                )
            raw_filter_or_ids = ids

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
        task: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        """
        Async wrapper for deleting edges.
        """
        ctx = self._build_ctx(task=task, extra_context=extra_context)
        framework_ctx = self._framework_ctx(
            operation="delete_edges",
            namespace=getattr(spec, "namespace", None),
        )

        if spec.filter is not None:
            raw_filter_or_ids: Any = spec.filter
        else:
            ids = list(spec.ids or [])
            if not ids:
                raise BadRequest(
                    "DeleteEdgesSpec must specify either filter or non-empty ids",
                    code=ErrorCodes.BAD_ADAPTER_RESULT,
                )
            raw_filter_or_ids = ids

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
        task: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BulkVerticesResult:
        """
        Sync wrapper for bulk_vertices.

        Converts `BulkVerticesSpec` into the raw request shape expected by
        GraphTranslator and returns the underlying `BulkVerticesResult`.
        """
        _ensure_not_in_event_loop("bulk_vertices")

        ctx = self._build_ctx(task=task, extra_context=extra_context)

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
        task: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BulkVerticesResult:
        """
        Async wrapper for bulk_vertices.
        """
        ctx = self._build_ctx(task=task, extra_context=extra_context)

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
        task: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> TraversalResult:
        """
        Sync wrapper for graph traversal.

        Builds a raw traversal request and delegates to GraphTranslator.
        """
        _ensure_not_in_event_loop("traversal")

        ctx = self._build_ctx(
            task=task,
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
        task: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> TraversalResult:
        """
        Async wrapper for graph traversal.
        """
        ctx = self._build_ctx(
            task=task,
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
        task: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BatchResult:
        """
        Sync wrapper for transactional batch operations.

        Translates `BatchOperation` dataclasses into the raw mapping shape
        expected by GraphTranslator and returns the underlying `BatchResult`.
        """
        _ensure_not_in_event_loop("transaction")

        # Reuse batch validation; semantics are still a list of BatchOperation.
        validate_batch_operations(ops, operation="transaction", error_code=ErrorCodes.INVALID_BATCH_OPS)

        ctx = self._build_ctx(
            task=task,
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
        task: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BatchResult:
        """
        Async wrapper for transactional batch operations.
        """
        validate_batch_operations(ops, operation="atransaction", error_code=ErrorCodes.INVALID_BATCH_OPS)

        ctx = self._build_ctx(
            task=task,
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
        task: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BatchResult:
        """
        Sync wrapper for batch operations.

        Translates `BatchOperation` dataclasses into the raw mapping shape
        expected by GraphTranslator and returns the underlying `BatchResult`.
        """
        _ensure_not_in_event_loop("batch")

        validate_batch_operations(ops, operation="batch", error_code=ErrorCodes.INVALID_BATCH_OPS)

        ctx = self._build_ctx(task=task, extra_context=extra_context)

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
        task: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BatchResult:
        """
        Async wrapper for batch operations.
        """
        validate_batch_operations(ops, operation="abatch", error_code=ErrorCodes.INVALID_BATCH_OPS)

        ctx = self._build_ctx(task=task, extra_context=extra_context)

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


# ---------------------------------------------------------------------------
# Optional CrewAI integration helpers (soft import)
# ---------------------------------------------------------------------------

def create_crewai_graph_tools(
    client: "CorpusCrewAIGraphClient",
    *,
    name_prefix: str = "graph",
    description_prefix: str = "Corpus graph tool",
) -> List[Any]:
    """
    Create CrewAI-native BaseTool wrappers for common graph operations.

    Why this exists:
    - The graph adapter itself is intentionally dependency-free and framework-light.
    - When CrewAI is installed, callers often want real CrewAI `BaseTool` objects
      to register on agents (e.g., in a Crew task/agent tool list).
    - The AutoGen adapter provides a real tool-wiring helper; this mirrors that
      approach for CrewAI.

    Soft dependency:
    - CrewAI is imported lazily. Importing this module does not require CrewAI.
    - If CrewAI is not installed, this function raises a clear RuntimeError with install instructions.

    Notes:
    - Tools provide both _run (sync) and _arun (async) implementations.
    - In _run, if called inside an event loop, execution is bridged to a bounded
      thread pool to preserve the adapter’s sync-in-event-loop safety guarantees.
    - Return values are JSON strings containing JSON-safe snapshots for tool-calling
      compatibility. (Protocol-level methods still return QueryResult/QueryChunk/etc.)
    """
    try:
        # CrewAI tool system – BaseTool is the canonical integration surface.
        # This import path matches current CrewAI source layout.
        from crewai.tools.base_tool import BaseTool  # type: ignore[import-not-found]
    except ImportError:
        # Some CrewAI versions have moved/aliased BaseTool. Try a compatible fallback
        # before failing with a clear error.
        try:
            from crewai.tools import BaseTool  # type: ignore[import-not-found]
        except ImportError as exc:  # noqa: BLE001
            raise RuntimeError(
                "CrewAI dependencies are not installed or BaseTool import path is unavailable. Install with:\n"
                '  pip install -U "crewai"\n'
                "Then retry create_crewai_graph_tools(...)."
            ) from exc

    # Hard cap for tool outputs to avoid accidental context-window explosions.
    # This is a best-effort guard; the primary size protection remains _json_safe_snapshot().
    _TOOL_JSON_MAX_CHARS = 120_000

    def _json_result(payload: Mapping[str, Any]) -> str:
        """
        Serialize a tool result to JSON.

        We keep tool outputs strictly serializable and size-bounded via _json_safe_snapshot().
        Additionally, we compact JSON output to reduce token footprint for agents.
        """
        snap = _json_safe_snapshot(payload)
        out = json.dumps(snap, ensure_ascii=False, separators=(",", ":"))
        if len(out) <= _TOOL_JSON_MAX_CHARS:
            return out

        # If we still exceeded the hard cap, re-snapshot more aggressively and add a warning.
        # This preserves tool compatibility (still valid JSON) while protecting the agent.
        aggressive = dict(payload)
        aggressive["_warning"] = f"tool output exceeded {_TOOL_JSON_MAX_CHARS} chars; aggressively truncated"
        snap2 = _json_safe_snapshot(aggressive, max_items=50, max_str=2_000)
        out2 = json.dumps(snap2, ensure_ascii=False, separators=(",", ":"))
        if len(out2) <= _TOOL_JSON_MAX_CHARS:
            return out2

        # Final fallback: guarantee bounded output without returning invalid JSON.
        # We keep a minimal JSON object rather than returning a raw truncated string.
        minimal = {
            "_warning": f"tool output exceeded {_TOOL_JSON_MAX_CHARS} chars even after truncation",
            "truncated": True,
        }
        return json.dumps(minimal, ensure_ascii=False, separators=(",", ":"))

    def _bridge_sync_call(fn: Callable[[], Any]) -> Any:
        """
        Execute a sync tool operation safely.

        - If no event loop is running, call directly.
        - If an event loop is running, execute in a bounded worker thread.
        """
        if not _is_running_event_loop():
            return fn()
        return _run_blocking_in_crewai_tool_thread(fn)

    class _GraphQueryTool(BaseTool):
        name: str
        description: str

        def _run(
            self,
            query: str,
            params: Optional[Mapping[str, Any]] = None,
            dialect: Optional[str] = None,
            namespace: Optional[str] = None,
            timeout_ms: Optional[int] = None,
            task: Optional[Any] = None,
            extra_context: Optional[Mapping[str, Any]] = None,
        ) -> str:
            def _work() -> str:
                res = client.query(
                    query,
                    params=params,
                    dialect=dialect,
                    namespace=namespace,
                    timeout_ms=timeout_ms,
                    task=task,
                    extra_context=extra_context,
                )
                return _json_result({"result": res})

            return _bridge_sync_call(_work)

        async def _arun(
            self,
            query: str,
            params: Optional[Mapping[str, Any]] = None,
            dialect: Optional[str] = None,
            namespace: Optional[str] = None,
            timeout_ms: Optional[int] = None,
            task: Optional[Any] = None,
            extra_context: Optional[Mapping[str, Any]] = None,
        ) -> str:
            res = await client.aquery(
                query,
                params=params,
                dialect=dialect,
                namespace=namespace,
                timeout_ms=timeout_ms,
                task=task,
                extra_context=extra_context,
            )
            return _json_result({"result": res})

    class _GraphStreamQueryTool(BaseTool):
        name: str
        description: str

        def _run(
            self,
            query: str,
            params: Optional[Mapping[str, Any]] = None,
            dialect: Optional[str] = None,
            namespace: Optional[str] = None,
            timeout_ms: Optional[int] = None,
            task: Optional[Any] = None,
            extra_context: Optional[Mapping[str, Any]] = None,
            max_chunks: int = 25,
        ) -> str:
            def _work() -> str:
                # Minor Note on max_chunks:
                # - LLMs may pass strings ("25") or invalid values (negative numbers).
                # - We coerce and clamp to protect callers and avoid edge-case errors.
                max_chunks_i = _coerce_bounded_positive_int(
                    max_chunks,
                    name="max_chunks",
                    default=25,
                    min_value=1,
                    max_value=100,
                )

                chunks: List[Any] = []
                count = 0
                for ch in client.stream_query(
                    query,
                    params=params,
                    dialect=dialect,
                    namespace=namespace,
                    timeout_ms=timeout_ms,
                    task=task,
                    extra_context=extra_context,
                ):
                    chunks.append(ch)
                    count += 1
                    if count >= max_chunks_i:
                        break
                return _json_result({"chunks": chunks, "truncated": count >= max_chunks_i})

            return _bridge_sync_call(_work)

        async def _arun(
            self,
            query: str,
            params: Optional[Mapping[str, Any]] = None,
            dialect: Optional[str] = None,
            namespace: Optional[str] = None,
            timeout_ms: Optional[int] = None,
            task: Optional[Any] = None,
            extra_context: Optional[Mapping[str, Any]] = None,
            max_chunks: int = 25,
        ) -> str:
            max_chunks_i = _coerce_bounded_positive_int(
                max_chunks,
                name="max_chunks",
                default=25,
                min_value=1,
                max_value=100,
            )

            chunks: List[Any] = []
            count = 0
            async for ch in client.astream_query(
                query,
                params=params,
                dialect=dialect,
                namespace=namespace,
                timeout_ms=timeout_ms,
                task=task,
                extra_context=extra_context,
            ):
                chunks.append(ch)
                count += 1
                if count >= max_chunks_i:
                    break
            return _json_result({"chunks": chunks, "truncated": count >= max_chunks_i})

    class _GraphBulkVerticesTool(BaseTool):
        name: str
        description: str

        def _run(
            self,
            namespace: Optional[str] = None,
            limit: int = 50,
            cursor: Optional[str] = None,
            filter: Optional[Mapping[str, Any]] = None,
            task: Optional[Any] = None,
            extra_context: Optional[Mapping[str, Any]] = None,
        ) -> str:
            def _work() -> str:
                spec = BulkVerticesSpec(namespace=namespace, limit=limit, cursor=cursor, filter=filter)
                res = client.bulk_vertices(spec, task=task, extra_context=extra_context)
                return _json_result({"result": res})

            return _bridge_sync_call(_work)

        async def _arun(
            self,
            namespace: Optional[str] = None,
            limit: int = 50,
            cursor: Optional[str] = None,
            filter: Optional[Mapping[str, Any]] = None,
            task: Optional[Any] = None,
            extra_context: Optional[Mapping[str, Any]] = None,
        ) -> str:
            spec = BulkVerticesSpec(namespace=namespace, limit=limit, cursor=cursor, filter=filter)
            res = await client.abulk_vertices(spec, task=task, extra_context=extra_context)
            return _json_result({"result": res})

    class _GraphBatchTool(BaseTool):
        name: str
        description: str

        def _run(
            self,
            ops: Sequence[Mapping[str, Any]],
            task: Optional[Any] = None,
            extra_context: Optional[Mapping[str, Any]] = None,
        ) -> str:
            def _work() -> str:
                batch_ops: List[BatchOperation] = []
                for idx, item in enumerate(list(ops)):
                    if not isinstance(item, Mapping):
                        raise TypeError(f"batch ops[{idx}] must be a mapping with keys 'op' and 'args'")
                    op = item.get("op")
                    args = item.get("args") or {}
                    if not isinstance(op, str) or not op:
                        raise TypeError(f"batch ops[{idx}]['op'] must be a non-empty string")
                    if not isinstance(args, Mapping):
                        raise TypeError(f"batch ops[{idx}]['args'] must be a mapping")
                    batch_ops.append(BatchOperation(op=op, args=dict(args)))

                # Use shared validation so tool behavior matches adapter expectations.
                validate_batch_operations(batch_ops, operation="batch", error_code=ErrorCodes.INVALID_BATCH_OPS)

                res = client.batch(batch_ops, task=task, extra_context=extra_context)
                return _json_result({"result": res})

            return _bridge_sync_call(_work)

        async def _arun(
            self,
            ops: Sequence[Mapping[str, Any]],
            task: Optional[Any] = None,
            extra_context: Optional[Mapping[str, Any]] = None,
        ) -> str:
            batch_ops: List[BatchOperation] = []
            for idx, item in enumerate(list(ops)):
                if not isinstance(item, Mapping):
                    raise TypeError(f"batch ops[{idx}] must be a mapping with keys 'op' and 'args'")
                op = item.get("op")
                args = item.get("args") or {}
                if not isinstance(op, str) or not op:
                    raise TypeError(f"batch ops[{idx}]['op'] must be a non-empty string")
                if not isinstance(args, Mapping):
                    raise TypeError(f"batch ops[{idx}]['args'] must be a mapping")
                batch_ops.append(BatchOperation(op=op, args=dict(args)))

            validate_batch_operations(batch_ops, operation="abatch", error_code=ErrorCodes.INVALID_BATCH_OPS)

            res = await client.abatch(batch_ops, task=task, extra_context=extra_context)
            return _json_result({"result": res})

    tools: List[Any] = [
        _GraphQueryTool(
            name=f"{name_prefix}_query",
            description=f"{description_prefix}: execute a graph query (non-streaming).",
        ),
        _GraphStreamQueryTool(
            name=f"{name_prefix}_stream_query",
            description=f"{description_prefix}: execute a graph query with streaming chunks (bounded).",
        ),
        _GraphBulkVerticesTool(
            name=f"{name_prefix}_bulk_vertices",
            description=f"{description_prefix}: bulk-scan vertices with pagination inputs.",
        ),
        _GraphBatchTool(
            name=f"{name_prefix}_batch",
            description=f"{description_prefix}: execute a batch of graph operations.",
        ),
    ]

    return tools


__all__ = [
    "CrewAIGraphClientProtocol",
    "CrewAIGraphFrameworkTranslator",
    "CorpusCrewAIGraphClient",
    "ErrorCodes",
    "with_graph_error_context",
    "with_async_graph_error_context",
    "with_error_context",
    "with_async_error_context",
    # Optional CrewAI integration helper (soft import)
    "create_crewai_graph_tools",
]
