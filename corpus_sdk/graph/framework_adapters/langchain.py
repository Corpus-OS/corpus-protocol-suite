# corpus_sdk/graph/framework_adapters/langchain.py
# SPDX-License-Identifier: Apache-2.0

"""
LangChain adapter for Corpus Graph protocol.

This module exposes a Corpus `GraphProtocolV1` implementation as a
LangChain-friendly client, with:

- Sync + async query APIs
- Sync + async streaming query APIs
- Proper integration with Corpus GraphProtocolV1
- OperationContext propagation derived from LangChain config / metadata
- Error-context enrichment for observability and debugging
- Orchestration, translation, and async→sync bridging via GraphTranslator

Design philosophy
-----------------
- Protocol-first: LangChain is a thin skin over the Corpus graph adapter.
- All heavy lifting (deadlines, breakers, rate limiting, caching, etc.) lives
  in the underlying `BaseGraphAdapter` / `GraphProtocolV1` implementation.
- This layer focuses on:
    * Translating LangChain RunnableConfig / config → OperationContext
    * Building raw query / mutation shapes for GraphTranslator
    * Delegating all sync/async and streaming orchestration to GraphTranslator

Responsibilities
----------------
- Provide a convenient, LangChain-oriented client for graph operations
- Keep all graph operations going through `GraphTranslator` so that
  async→sync bridging, streaming, and error-context logic are centralized
- Preserve protocol-level types (`QueryResult`, `QueryChunk`, etc.) for
  LangChain callers

Non-responsibilities
--------------------
- Backend-specific graph behavior (lives in graph adapters)
- LangChain chain/agent orchestration and config logic
- MMR and diversification details (handled inside GraphTranslator)

Compatibility notes
-------------------
- LangChain is an **optional dependency**. This module intentionally does not
  hard-import LangChain packages at import time.
- Real LangChain tool integration is provided through **soft-imported tool helpers**
  at the bottom of this file (e.g., `create_langchain_graph_tools()` and
  `create_corpus_graph_tool()`).
  Importing this module does not require LangChain to be installed.
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
    from_langchain as core_ctx_from_langchain,
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

# Type variables for decorators
T = TypeVar("T")

# ---------------------------------------------------------------------------
# Optional LangChain Tool soft-import + bounded async→sync bridge
# ---------------------------------------------------------------------------

# Dedicated, bounded executor for tool-in-event-loop compatibility.
# Used only by create_langchain_graph_tools(...) / create_corpus_graph_tool(...)
# to keep tool runs safe in async LangChain runtimes.
_LANGCHAIN_TOOL_BRIDGE_EXECUTOR: Optional[concurrent.futures.ThreadPoolExecutor] = None
_LANGCHAIN_TOOL_BRIDGE_EXECUTOR_LOCK = threading.Lock()


def _shutdown_langchain_tool_bridge_executor() -> None:
    """
    Best-effort shutdown for the bounded tool bridge executor.

    Why this exists:
    - Tool wrappers may run in short-lived processes (tests, CLIs, workers).
    - If the executor is never shut down, threads can linger and create noise.

    Safety:
    - Idempotent: safe to call multiple times.
    - Best-effort: never raises to callers.
    """
    global _LANGCHAIN_TOOL_BRIDGE_EXECUTOR  # noqa: PLW0603
    with _LANGCHAIN_TOOL_BRIDGE_EXECUTOR_LOCK:
        executor = _LANGCHAIN_TOOL_BRIDGE_EXECUTOR
        _LANGCHAIN_TOOL_BRIDGE_EXECUTOR = None

    if executor is None:
        return

    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except Exception:  # noqa: BLE001
        # Never allow shutdown failures to propagate.
        logger.debug("Failed to shutdown LangChain tool bridge executor", exc_info=True)


# Ensure the executor does not leak threads across interpreter shutdown.
atexit.register(_shutdown_langchain_tool_bridge_executor)


def _is_running_event_loop() -> bool:
    """
    Return True if called while an asyncio event loop is running in this thread.

    This is used by optional LangChain tool helpers to provide safe sync execution
    in environments that execute tools from within async runtimes.
    """
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def _run_blocking_in_langchain_tool_thread(fn: Callable[[], T]) -> T:
    """
    Run a blocking graph call in a bounded thread pool.

    Why:
      LangChain tool execution can occur inside an async runtime (event loop running),
      but the graph client sync APIs deliberately refuse to run inside event loops.

    Safety/performance:
      - bounded pool (max_workers=4) to prevent unbounded thread creation
      - used only in the optional tool integration path
      - executor is shut down on client close and at interpreter exit (atexit)
    """
    global _LANGCHAIN_TOOL_BRIDGE_EXECUTOR  # noqa: PLW0603
    with _LANGCHAIN_TOOL_BRIDGE_EXECUTOR_LOCK:
        if _LANGCHAIN_TOOL_BRIDGE_EXECUTOR is None:
            _LANGCHAIN_TOOL_BRIDGE_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
                max_workers=4,
                thread_name_prefix="corpus-langchain-tool",
            )
        executor = _LANGCHAIN_TOOL_BRIDGE_EXECUTOR

    # Block until the work completes; tool wrappers are sync by design.
    return executor.submit(fn).result()


def _json_safe_snapshot(value: Any, *, max_items: int = 200, max_str: int = 10_000) -> Any:
    """
    Best-effort conversion into a JSON-ish snapshot for tool-return values and logs.

    Security / correctness:
    - Limits container sizes to avoid memory bloat.
    - Truncates long strings to prevent context-window blowups in agent runtimes.
    - Falls back to repr() for unknown objects.

    NOTE:
    - This helper is used only in optional LangChain tool helpers below.
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


def _normalize_async_iterator(aiter_or_awaitable: Any) -> "AsyncIterator[Any]":
    """
    Normalize either:
      - an AsyncIterator, OR
      - an awaitable that resolves to an AsyncIterator,
    into a concrete AsyncIterator.

    Why:
      GraphTranslator implementations (or versions) may return either a ready
      AsyncIterator or an awaitable that yields one. Centralizing this logic
      keeps adapter streaming semantics stable while allowing translator evolution
      without brittle tests or duplicated adapter code.

    Compatibility:
      - This helper mirrors the robust normalization behavior used in other
        framework adapters and is intentionally dependency-free.
    """

    async def _await_and_return() -> AsyncIterator[Any]:
        resolved = await aiter_or_awaitable
        return resolved

    if inspect.isawaitable(aiter_or_awaitable):
        return _await_and_return()  # type: ignore[return-value]
    return aiter_or_awaitable  # type: ignore[return-value]


def _require_langchain_base_tool() -> Any:
    """
    Soft-import LangChain's BaseTool.

    This intentionally avoids importing LangChain at module import time so that
    this adapter remains dependency-light and import-safe in minimal environments.
    """
    try:
        # Try langchain_core.tools first (langchain 0.1.x and newer)
        from langchain_core.tools import BaseTool  # type: ignore[import-not-found]
    except ImportError:
        try:
            # Fallback to langchain.tools for older versions
            from langchain.tools import BaseTool  # type: ignore[import-not-found]
        except ImportError as exc:  # noqa: BLE001
            raise ImportError(
                "LangChain tools are not installed. Install with:\n"
                '  pip install -U "langchain" "langchain-core"\n'
                "Then retry using create_langchain_graph_tools(...) / create_corpus_graph_tool(...)."
            ) from exc
    return BaseTool


# ---------------------------------------------------------------------------
# Error code constants (flat, framework-specific)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Error-context decorators (centralized via common framework utils)
# ---------------------------------------------------------------------------


def with_graph_error_context(
    operation: str,
    **static_context: Any,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for sync methods with rich dynamic context extraction.

    Thin wrapper over the shared `create_graph_error_context_decorator`
    for the LangChain framework.
    """
    return create_graph_error_context_decorator(
        framework="langchain",
        is_async=False,
    )(operation=operation, **static_context)


def with_async_graph_error_context(
    operation: str,
    **static_context: Any,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for async methods with rich dynamic context extraction.

    Thin wrapper over the shared `create_graph_error_context_decorator`
    for the LangChain framework.
    """
    return create_graph_error_context_decorator(
        framework="langchain",
        is_async=True,
    )(operation=operation, **static_context)


# Backwards-compatible aliases (for older imports)
with_error_context = with_graph_error_context
with_async_error_context = with_async_graph_error_context


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _looks_like_operation_context(obj: Any) -> bool:
    """
    Heuristic check; OperationContext may be a Protocol/alias in some SDK versions.

    We try a direct isinstance check when possible, but fall back to a
    structural check on common OperationContext attributes so that this
    adapter is resilient to typing/aliasing changes.
    """
    if obj is None:
        return False

    try:
        if isinstance(obj, OperationContext):
            return True
    except TypeError:
        # OperationContext may be a Protocol in some typing modes
        pass

    attrs = ("request_id", "traceparent", "tenant", "attrs", "to_dict")
    return any(hasattr(obj, attr) for attr in attrs)


def _ensure_not_in_event_loop(sync_api_name: str) -> None:
    """
    Guard against calling sync graph APIs from within an active asyncio loop.

    This prevents subtle sync-over-async deadlocks in environments like
    Jupyter, FastAPI, or async LangChain chains. Callers are directed to
    use the corresponding async variant instead.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running event loop -> safe for sync calls.
        return

    raise RuntimeError(
        f"{sync_api_name} was called from inside an active asyncio event loop. "
        f"Use the async variant instead (e.g. 'a{sync_api_name}'). "
        f"[{ErrorCodes.SYNC_WRAPPER_CALLED_IN_EVENT_LOOP}]"
    )


# ---------------------------------------------------------------------------
# Public framework translator
# ---------------------------------------------------------------------------


class LangChainGraphFrameworkTranslator(DefaultGraphFrameworkTranslator):
    """
    LangChain-specific GraphFrameworkTranslator.

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


# ---------------------------------------------------------------------------
# Public protocol
# ---------------------------------------------------------------------------


class LangChainGraphClientProtocol(Protocol):
    """
    Protocol representing the minimal LangChain-aware graph client interface
    implemented by this module.

    This structural protocol allows callers to type against the graph client
    without depending on the concrete `CorpusLangChainGraphClient` class.
    """

    # Capabilities / schema / health -------------------------------------

    def capabilities(self, **kwargs) -> Mapping[str, Any]:
        ...

    async def acapabilities(self, **kwargs) -> Mapping[str, Any]:
        ...

    def get_schema(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> GraphSchema:
        ...

    async def aget_schema(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> GraphSchema:
        ...

    def health(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        ...

    async def ahealth(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
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
        config: Optional[Mapping[str, Any]] = None,
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
        config: Optional[Mapping[str, Any]] = None,
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
        config: Optional[Mapping[str, Any]] = None,
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
        config: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> AsyncIterator[QueryChunk]:
        ...

    # Upsert --------------------------------------------------------------

    def upsert_nodes(
        self,
        spec: UpsertNodesSpec,
        *,
        config: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        ...

    async def aupsert_nodes(
        self,
        spec: UpsertNodesSpec,
        *,
        config: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        ...

    def upsert_edges(
        self,
        spec: UpsertEdgesSpec,
        *,
        config: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        ...

    async def aupsert_edges(
        self,
        spec: UpsertEdgesSpec,
        *,
        config: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        ...

    # Delete --------------------------------------------------------------

    def delete_nodes(
        self,
        spec: DeleteNodesSpec,
        *,
        config: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        ...

    async def adelete_nodes(
        self,
        spec: DeleteNodesSpec,
        *,
        config: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        ...

    def delete_edges(
        self,
        spec: DeleteEdgesSpec,
        *,
        config: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        ...

    async def adelete_edges(
        self,
        spec: DeleteEdgesSpec,
        *,
        config: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        ...

    # Bulk / batch --------------------------------------------------------

    def bulk_vertices(
        self,
        spec: BulkVerticesSpec,
        *,
        config: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BulkVerticesResult:
        ...

    async def abulk_vertices(
        self,
        spec: BulkVerticesSpec,
        *,
        config: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BulkVerticesResult:
        ...

    def batch(
        self,
        ops: List[BatchOperation],
        *,
        config: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BatchResult:
        ...

    async def abatch(
        self,
        ops: List[BatchOperation],
        *,
        config: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BatchResult:
        ...

    # Resource management -------------------------------------------------

    def close(self) -> None:
        ...

    async def aclose(self) -> None:
        ...


# ---------------------------------------------------------------------------
# Main LangChain client
# ---------------------------------------------------------------------------


class CorpusLangChainGraphClient:
    """
    LangChain-oriented client wrapper around a Corpus `GraphProtocolV1`.

    This is a thin integration layer that:

    - Translates LangChain config / metadata into a Corpus `OperationContext`
      using `core_ctx_from_langchain`.
    - Uses `GraphTranslator` (with a LangChain-specific framework translator) to:
        * Build Graph*Spec objects from simple inputs
        * Execute sync + async graph operations
        * Orchestrate streaming with proper cancellation and error handling
    - Delegates all async→sync bridging and streaming glue to GraphTranslator.
    - Attaches rich error context (`attach_context`) on this layer with
      LangChain-specific hints when failures occur.
    """

    def __init__(
        self,
        *,
        graph_adapter: Optional[GraphProtocolV1] = None,
        adapter: Optional[GraphProtocolV1] = None,
        default_dialect: Optional[str] = None,
        default_namespace: Optional[str] = None,
        default_timeout_ms: Optional[int] = None,
        framework_version: Optional[str] = None,
        framework_translator: Optional[GraphFrameworkTranslator] = None,
    ) -> None:
        """
        Initialize a LangChain-oriented graph client.

        Parameters
        ----------
        graph_adapter:
            Underlying `GraphProtocolV1` implementation (legacy parameter name).
        adapter:
            Underlying `GraphProtocolV1` implementation (standard parameter name).
            Either adapter or graph_adapter must be provided, but not both.
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
            Optional custom framework translator. If not provided, the default
            `LangChainGraphFrameworkTranslator` is used.

        Typing note:
            `framework_translator` is typed as `GraphFrameworkTranslator` so callers
            can pass either the default translator or any compatible override
            without narrowing to a concrete base class.
        """
        if adapter is not None and graph_adapter is not None:
            raise TypeError(
                "Cannot specify both 'adapter' and 'graph_adapter' parameters. "
                "Please use only one."
            )
        if adapter is None and graph_adapter is None:
            raise TypeError(
                "Must specify either 'adapter' or 'graph_adapter' parameter."
            )

        resolved_adapter = graph_adapter if graph_adapter is not None else adapter
        self._graph: GraphProtocolV1 = resolved_adapter
        self._default_dialect: Optional[str] = default_dialect
        self._default_namespace: Optional[str] = default_namespace
        self._default_timeout_ms: Optional[int] = default_timeout_ms
        self._framework_version: Optional[str] = framework_version
        self._framework_translator: Optional[GraphFrameworkTranslator] = framework_translator

        # Resource management flags (idempotent close semantics)
        self._closed: bool = False
        self._aclosed: bool = False

    # ------------------------------------------------------------------ #
    # Resource Management (Context Managers)
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "CorpusLangChainGraphClient":
        """Support context manager protocol for resource cleanup."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Clean up resources when exiting context.

        Uses the explicit `close()` method so that resource-management
        semantics remain centralized and idempotent.
        """
        self.close()

    async def __aenter__(self) -> "CorpusLangChainGraphClient":
        """Support async context manager protocol."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Clean up resources when exiting async context.

        Uses the explicit `aclose()` method to honor async cleanup paths.
        """
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
                result = close_fn()
                # If close() returns a coroutine, it means the adapter only has async close
                if inspect.iscoroutine(result):
                    result.close()  # Suppress the "never awaited" warning
                    logger.warning(
                        "Adapter has async-only close() - call aclose() instead or use async context manager"
                    )
            except Exception as e:  # noqa: BLE001
                # Preserve the existing warning-level behavior, but include
                # stack context for easier debugging without changing semantics.
                logger.warning(
                    "Error while closing graph adapter in close(): %s",
                    e,
                    exc_info=True,
                )

        # Also close the optional bounded tool executor so tests and short-lived
        # processes do not leak threads. This is safe and idempotent.
        _shutdown_langchain_tool_bridge_executor()

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
            except Exception as e:  # noqa: BLE001
                # Preserve warning-level behavior, but include stack context.
                logger.warning(
                    "Error while async-closing graph adapter in aclose(): %s",
                    e,
                    exc_info=True,
                )

        # Fallback to sync close if we haven't already done so.
        if not self._closed:
            self.close()

        # As above, ensure the optional tool executor is stopped.
        _shutdown_langchain_tool_bridge_executor()

    # ------------------------------------------------------------------ #
    # Translator (lazy, cached) – mirrors CrewAI adapter pattern
    # ------------------------------------------------------------------ #

    @cached_property
    def _translator(self) -> GraphTranslator:
        """
        Lazily construct and cache the `GraphTranslator`.

        Uses `cached_property` for thread safety and performance, mirroring
        the CrewAI adapter patterns. Allows callers to inject a custom
        framework_translator via __init__.
        """
        framework_translator: GraphFrameworkTranslator = (
            self._framework_translator or LangChainGraphFrameworkTranslator()
        )
        return create_graph_translator(
            adapter=self._graph,
            framework="langchain",
            translator=framework_translator,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_ctx(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Optional[OperationContext]:
        """
        Build an OperationContext from LangChain-style inputs.

        Expected inputs
        ----------------
        - config: RunnableConfig-like mapping (optional)
        - extra_context: Optional mapping merged into attrs (best effort)

        If both are None/empty, returns None and lets downstream helpers
        construct an "empty" OperationContext as needed.

        On translation failure, we log + attach error context and fall back
        to `None` instead of raising, to avoid breaking graph calls purely
        due to context issues.
        """
        extra = dict(extra_context or {})

        if config is None and not extra:
            return None

        try:
            ctx_candidate = core_ctx_from_langchain(
                config,
                framework_version=self._framework_version,
                **extra,
            )
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="langchain",
                operation="context_translation",
                error_code=ErrorCodes.BAD_OPERATION_CONTEXT,
                config_snapshot=str(config)[:1024] if config is not None else None,
                config_type=type(config).__name__ if config is not None else "None",
                extra_context_keys=list(extra.keys()),
            )
            logger.warning(
                "Failed to build OperationContext from LangChain inputs; "
                "proceeding without OperationContext: %s",
                exc,
            )
            return None

        if not _looks_like_operation_context(ctx_candidate):
            logger.warning(
                "from_langchain produced unsupported context type %s; "
                "proceeding without OperationContext.",
                type(ctx_candidate).__name__,
            )
            return None

        # Best-effort enrichment of attrs with framework metadata while
        # preserving the original OperationContext instance.
        try:
            attrs = getattr(ctx_candidate, "attrs", None)
            if isinstance(attrs, Mapping):
                # Copy to avoid mutating shared mappings, if any.
                enriched_attrs: Dict[str, Any] = dict(attrs)
            else:
                enriched_attrs = {}

            enriched_attrs.setdefault("framework", "langchain")
            if self._framework_version is not None:
                enriched_attrs.setdefault("framework_version", self._framework_version)

            try:
                setattr(ctx_candidate, "attrs", enriched_attrs)
            except Exception:
                # If attrs is not assignable, fall back to in-place update where possible.
                try:
                    if hasattr(attrs, "setdefault"):
                        attrs.setdefault("framework", "langchain")
                        if self._framework_version is not None:
                            attrs.setdefault("framework_version", self._framework_version)
                except Exception:
                    logger.debug(
                        "Failed to update OperationContext attrs in-place for LangChain",
                        exc_info=True,
                    )
        except Exception:
            logger.debug(
                "Failed to enrich OperationContext attrs for LangChain",
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
            "framework": "langchain",
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
        LangChain-local validation for edge upsert specs.

        Mirrors the AutoGen / CrewAI adapters:
        - edges must not be None
        - edges must be iterable and non-empty
        - each edge must have an ID

        Returns a validated list of edges without mutating the original spec.
        """
        if spec.edges is None:
            raise BadRequest("UpsertEdgesSpec.edges must not be None")

        try:
            edges_iter = list(spec.edges)
        except TypeError as exc:
            raise BadRequest(
                "UpsertEdgesSpec.edges must be an iterable of edges",
            ) from exc

        if not edges_iter:
            raise BadRequest("UpsertEdgesSpec must contain at least one edge")

        for edge in edges_iter:
            if not getattr(edge, "id", None):
                raise BadRequest("All edges must have an ID")

        return edges_iter

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
    def capabilities(self, **kwargs) -> Mapping[str, Any]:
        """
        Sync wrapper around capabilities, delegating async→sync bridging
        to GraphTranslator.
        """
        _ensure_not_in_event_loop("capabilities")
        caps = self._translator.capabilities()
        return graph_capabilities_to_dict(caps)

    @with_async_graph_error_context("capabilities_async")
    async def acapabilities(self, **kwargs) -> Mapping[str, Any]:
        """
        Async capabilities accessor.

        We delegate to GraphTranslator for consistency, then normalize to a
        simple dict for LangChain consumption.
        """
        caps = await self._translator.arun_capabilities()
        return graph_capabilities_to_dict(caps)

    @with_graph_error_context("get_schema_sync")
    def get_schema(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> GraphSchema:
        """
        Sync wrapper around `graph_adapter.get_schema(...)`.

        Delegates to GraphTranslator so that async→sync bridging and
        error-context handling are centralized.
        """
        _ensure_not_in_event_loop("get_schema")
        ctx = self._build_ctx(config=config, extra_context=extra_context)
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
        config: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> GraphSchema:
        """
        Async wrapper around `graph_adapter.get_schema(...)`.

        Delegates to GraphTranslator.
        """
        ctx = self._build_ctx(config=config, extra_context=extra_context)
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
        config: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        """
        Health check (sync).

        Uses GraphTranslator for consistency with other operations.
        """
        _ensure_not_in_event_loop("health")
        ctx = self._build_ctx(config=config, extra_context=extra_context)
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
        config: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        """
        Health check (async).

        Uses GraphTranslator for consistency with other operations.
        """
        ctx = self._build_ctx(config=config, extra_context=extra_context)
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
        config: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> QueryResult:
        """
        Execute a non-streaming graph query (sync).

        Returns the underlying `QueryResult` from the GraphProtocol adapter.
        """
        _ensure_not_in_event_loop("query")
        validate_graph_query(query, operation="query", error_code=ErrorCodes.INVALID_QUERY)
        self._validate_query_params(params)

        ctx = self._build_ctx(config=config, extra_context=extra_context)
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
            # Dialect fallback mirrors CrewAI/AutoGen adapter behavior:
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
        config: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> QueryResult:
        """
        Execute a non-streaming graph query (async).

        Returns the underlying `QueryResult`.
        """
        validate_graph_query(query, operation="aquery", error_code=ErrorCodes.INVALID_QUERY)
        self._validate_query_params(params)

        ctx = self._build_ctx(config=config, extra_context=extra_context)
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
        config: Optional[Mapping[str, Any]] = None,
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

        ctx = self._build_ctx(config=config, extra_context=extra_context)
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
        config: Optional[Mapping[str, Any]] = None,
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

        ctx = self._build_ctx(config=config, extra_context=extra_context)
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

        # Normalize the possible return forms (AsyncIterator vs awaitable->AsyncIterator)
        # to avoid coupling adapter behavior to a specific GraphTranslator style.
        aiter_normalized = _normalize_async_iterator(aiter_or_awaitable)
        if inspect.isawaitable(aiter_normalized):
            aiter = await aiter_normalized  # type: ignore[assignment]
        else:
            aiter = aiter_normalized  # type: ignore[assignment]

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
        config: Optional[Mapping[str, Any]] = None,
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

        ctx = self._build_ctx(config=config, extra_context=extra_context)
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
        config: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        """
        Async wrapper for upserting nodes.
        """
        validate_upsert_nodes_spec(spec)

        ctx = self._build_ctx(config=config, extra_context=extra_context)
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
        config: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        """
        Sync wrapper for upserting edges.
        """
        _ensure_not_in_event_loop("upsert_edges")
        edges = self._validate_upsert_edges_spec(spec)

        ctx = self._build_ctx(config=config, extra_context=extra_context)
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
        config: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        """
        Async wrapper for upserting edges.
        """
        edges = self._validate_upsert_edges_spec(spec)

        ctx = self._build_ctx(config=config, extra_context=extra_context)
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
        config: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        """
        Sync wrapper for deleting nodes.

        Uses DeleteNodesSpec to derive either an ID list or a filter
        expression for the GraphTranslator.
        """
        _ensure_not_in_event_loop("delete_nodes")
        ctx = self._build_ctx(config=config, extra_context=extra_context)
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
        config: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        """
        Async wrapper for deleting nodes.
        """
        ctx = self._build_ctx(config=config, extra_context=extra_context)
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
        config: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        """
        Sync wrapper for deleting edges.
        """
        _ensure_not_in_event_loop("delete_edges")
        ctx = self._build_ctx(config=config, extra_context=extra_context)
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
        config: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> DeleteResult:
        """
        Async wrapper for deleting edges.
        """
        ctx = self._build_ctx(config=config, extra_context=extra_context)
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
        config: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BulkVerticesResult:
        """
        Sync wrapper for bulk_vertices.

        Converts `BulkVerticesSpec` into the raw request shape expected by
        GraphTranslator and returns the underlying `BulkVerticesResult`.
        """
        _ensure_not_in_event_loop("bulk_vertices")
        ctx = self._build_ctx(config=config, extra_context=extra_context)

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
        config: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BulkVerticesResult:
        """
        Async wrapper for bulk_vertices.
        """
        ctx = self._build_ctx(config=config, extra_context=extra_context)

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
        config: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> TraversalResult:
        """
        Sync wrapper for graph traversal.

        Builds a raw traversal request and delegates to GraphTranslator.
        """
        _ensure_not_in_event_loop("traversal")

        ctx = self._build_ctx(
            config=config,
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
        config: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> TraversalResult:
        """
        Async wrapper for graph traversal.
        """
        ctx = self._build_ctx(
            config=config,
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
        config: Optional[Mapping[str, Any]] = None,
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
            config=config,
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
        config: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BatchResult:
        """
        Async wrapper for transactional batch operations.
        """
        validate_batch_operations(ops, operation="atransaction", error_code=ErrorCodes.INVALID_BATCH_OPS)

        ctx = self._build_ctx(
            config=config,
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
        config: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BatchResult:
        """
        Sync wrapper for batch operations.

        Translates `BatchOperation` dataclasses into the raw mapping shape
        expected by GraphTranslator and returns the underlying `BatchResult`.
        """
        _ensure_not_in_event_loop("batch")
        validate_batch_operations(ops, operation="batch", error_code=ErrorCodes.INVALID_BATCH_OPS)

        ctx = self._build_ctx(config=config, extra_context=extra_context)

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
        config: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> BatchResult:
        """
        Async wrapper for batch operations.
        """
        validate_batch_operations(ops, operation="abatch", error_code=ErrorCodes.INVALID_BATCH_OPS)

        ctx = self._build_ctx(config=config, extra_context=extra_context)

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
# Optional LangChain Tool wrapper (soft import)
# ---------------------------------------------------------------------------

class CorpusGraphTool:
    """
    LangChain Tool wrapper around `CorpusLangChainGraphClient`.

    This allows the Corpus graph client to be used directly in LangChain
    agents / tools-based workflows.

    By default it exposes a very simple interface:

        - `query`: required graph query string
        - `params`: optional mapping of query parameters
        - `dialect`: optional query dialect override
        - `namespace`: optional namespace override
        - `timeout_ms`: optional per-call timeout

    You can subclass this Tool to tighten or reshape the input schema as
    needed for a particular agent.

    IMPORTANT:
    This symbol is preserved for backwards compatibility, but the real LangChain
    BaseTool-backed implementations are created via soft-import helpers:
      - create_langchain_graph_tools(...)
      - create_corpus_graph_tool(...)

    If LangChain is not installed, attempting to instantiate this class will
    raise an ImportError with installation instructions.
    """

    # This class is intentionally *not* a real LangChain BaseTool at import time.
    # LangChain is an optional dependency; tool integration is provided via the
    # helper factories below which do a soft import and build real BaseTool types.
    def __init__(self, *_: Any, **__: Any) -> None:
        raise ImportError(
            "CorpusGraphTool requires LangChain to be installed. Install with:\n"
            '  pip install -U "langchain"\n'
            "Then create tools via create_langchain_graph_tools(...) or create_corpus_graph_tool(...)."
        )


def create_langchain_graph_tools(
    client: CorpusLangChainGraphClient,
    *,
    name_prefix: str = "graph",
    description_prefix: str = "Corpus graph tool",
) -> List[Any]:
    """
    Create LangChain-native BaseTool wrappers for common graph operations.

    Why this exists:
    - The graph adapter itself is intentionally dependency-free and framework-light.
    - When LangChain is installed, callers often want real LangChain `BaseTool` objects
      to register on agents (e.g., in a LangChain tool list).
    - The AutoGen and CrewAI adapters provide real tool-wiring helpers; this mirrors that
      approach for LangChain while keeping LangChain as a soft dependency.

    Soft dependency:
    - LangChain is imported lazily. Importing this module does not require LangChain.
    - If LangChain is not installed, this function raises a clear ImportError with install instructions.

    Notes:
    - Tools provide both _run (sync) and _arun (async) implementations.
    - In _run, if called inside an event loop, execution is bridged to a bounded
      thread pool to preserve the adapter’s sync-in-event-loop safety guarantees.
    - Return values are JSON strings containing JSON-safe snapshots for tool-calling
      compatibility. (Protocol-level methods still return QueryResult/QueryChunk/etc.)
    """
    BaseTool = _require_langchain_base_tool()

    def _json_result(payload: Mapping[str, Any]) -> str:
        """
        Serialize a tool result to JSON.

        We keep tool outputs strictly serializable and size-bounded via _json_safe_snapshot().
        """
        return json.dumps(_json_safe_snapshot(payload), ensure_ascii=False)

    def _bridge_sync_call(fn: Callable[[], Any]) -> Any:
        """
        Execute a sync tool operation safely.

        - If no event loop is running, call directly.
        - If an event loop is running, execute in a bounded worker thread.
        """
        if not _is_running_event_loop():
            return fn()
        return _run_blocking_in_langchain_tool_thread(fn)

    def _validated_max_chunks(value: Any) -> int:
        """
        Validate and normalize max_chunks inputs from LLMs and tool callers.

        This mirrors the defensive approach used in other framework adapters:
        - accept ints or int-like strings (e.g., "25")
        - reject non-positive values to avoid edge-case behavior
        - clamp overly large values to a safe upper bound to prevent runaway
          memory/time usage when a tool caller supplies an extremely large limit
          (intentionally or accidentally)

        The upper bound is intentionally conservative: tools should provide a
        representative sample of chunks, not an unbounded stream dump.
        """
        _MAX_TOOL_CHUNKS = 100
        try:
            as_int = int(value)
        except Exception as exc:  # noqa: BLE001
            raise BadRequest(
                f"max_chunks must be an integer, got {type(value).__name__}",
                code=ErrorCodes.BAD_ADAPTER_RESULT,
            ) from exc
        if as_int <= 0:
            raise BadRequest(
                f"max_chunks must be > 0, got {as_int}",
                code=ErrorCodes.BAD_ADAPTER_RESULT,
            )
        if as_int > _MAX_TOOL_CHUNKS:
            # Clamp rather than raise: this keeps tools robust to LLM overreach
            # while still preventing excessive iteration and output growth.
            return _MAX_TOOL_CHUNKS
        return as_int

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
            config: Optional[Mapping[str, Any]] = None,
            extra_context: Optional[Mapping[str, Any]] = None,
        ) -> str:
            def _work() -> str:
                res = client.query(
                    query,
                    params=params,
                    dialect=dialect,
                    namespace=namespace,
                    timeout_ms=timeout_ms,
                    config=config,
                    extra_context=extra_context,
                )
                return _json_result({"result": _json_safe_snapshot(res)})

            return _bridge_sync_call(_work)

        async def _arun(
            self,
            query: str,
            params: Optional[Mapping[str, Any]] = None,
            dialect: Optional[str] = None,
            namespace: Optional[str] = None,
            timeout_ms: Optional[int] = None,
            config: Optional[Mapping[str, Any]] = None,
            extra_context: Optional[Mapping[str, Any]] = None,
        ) -> str:
            res = await client.aquery(
                query,
                params=params,
                dialect=dialect,
                namespace=namespace,
                timeout_ms=timeout_ms,
                config=config,
                extra_context=extra_context,
            )
            return _json_result({"result": _json_safe_snapshot(res)})

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
            config: Optional[Mapping[str, Any]] = None,
            extra_context: Optional[Mapping[str, Any]] = None,
            max_chunks: Any = 25,
        ) -> str:
            def _work() -> str:
                limit = _validated_max_chunks(max_chunks)
                chunks: List[Any] = []
                count = 0
                for ch in client.stream_query(
                    query,
                    params=params,
                    dialect=dialect,
                    namespace=namespace,
                    timeout_ms=timeout_ms,
                    config=config,
                    extra_context=extra_context,
                ):
                    chunks.append(_json_safe_snapshot(ch))
                    count += 1
                    if count >= limit:
                        break
                return _json_result({"chunks": chunks, "truncated": count >= limit})

            return _bridge_sync_call(_work)

        async def _arun(
            self,
            query: str,
            params: Optional[Mapping[str, Any]] = None,
            dialect: Optional[str] = None,
            namespace: Optional[str] = None,
            timeout_ms: Optional[int] = None,
            config: Optional[Mapping[str, Any]] = None,
            extra_context: Optional[Mapping[str, Any]] = None,
            max_chunks: Any = 25,
        ) -> str:
            limit = _validated_max_chunks(max_chunks)
            chunks: List[Any] = []
            count = 0
            async for ch in client.astream_query(
                query,
                params=params,
                dialect=dialect,
                namespace=namespace,
                timeout_ms=timeout_ms,
                config=config,
                extra_context=extra_context,
            ):
                chunks.append(_json_safe_snapshot(ch))
                count += 1
                if count >= limit:
                    break
            return _json_result({"chunks": chunks, "truncated": count >= limit})

    class _GraphBulkVerticesTool(BaseTool):
        name: str
        description: str

        def _run(
            self,
            namespace: Optional[str] = None,
            limit: int = 50,
            cursor: Optional[str] = None,
            filter: Optional[Mapping[str, Any]] = None,
            config: Optional[Mapping[str, Any]] = None,
            extra_context: Optional[Mapping[str, Any]] = None,
        ) -> str:
            def _work() -> str:
                spec = BulkVerticesSpec(namespace=namespace, limit=limit, cursor=cursor, filter=filter)
                res = client.bulk_vertices(spec, config=config, extra_context=extra_context)
                return _json_result({"result": _json_safe_snapshot(res)})

            return _bridge_sync_call(_work)

        async def _arun(
            self,
            namespace: Optional[str] = None,
            limit: int = 50,
            cursor: Optional[str] = None,
            filter: Optional[Mapping[str, Any]] = None,
            config: Optional[Mapping[str, Any]] = None,
            extra_context: Optional[Mapping[str, Any]] = None,
        ) -> str:
            spec = BulkVerticesSpec(namespace=namespace, limit=limit, cursor=cursor, filter=filter)
            res = await client.abulk_vertices(spec, config=config, extra_context=extra_context)
            return _json_result({"result": _json_safe_snapshot(res)})

    class _GraphBatchTool(BaseTool):
        name: str
        description: str

        def _run(
            self,
            ops: Sequence[Mapping[str, Any]],
            config: Optional[Mapping[str, Any]] = None,
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

                res = client.batch(batch_ops, config=config, extra_context=extra_context)
                return _json_result({"result": _json_safe_snapshot(res)})

            return _bridge_sync_call(_work)

        async def _arun(
            self,
            ops: Sequence[Mapping[str, Any]],
            config: Optional[Mapping[str, Any]] = None,
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

            res = await client.abatch(batch_ops, config=config, extra_context=extra_context)
            return _json_result({"result": _json_safe_snapshot(res)})

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


def create_corpus_graph_tool(
    *,
    graph_adapter: GraphProtocolV1,
    default_dialect: Optional[str] = None,
    default_namespace: Optional[str] = None,
    default_timeout_ms: Optional[int] = None,
    framework_version: Optional[str] = None,
    name: str = "corpus_graph",
    description: Optional[str] = None,
    framework_translator: Optional[GraphFrameworkTranslator] = None,
) -> Any:
    """
    Convenience factory to create a `CorpusLangChainGraphClient` and wrap it
    in a LangChain-native tool in one go.

    If LangChain tools are not installed, this will raise an ImportError
    so that misuse is surfaced clearly.

    Example
    -------
        graph_adapter = MyGraphAdapter(...)
        graph_tool = create_corpus_graph_tool(
            graph_adapter=graph_adapter,
            default_namespace="prod",
        )
        agent = initialize_agent(
            tools=[graph_tool],
            llm=...,
            agent_type=AgentType.OPENAI_FUNCTIONS,
        )
    """
    BaseTool = _require_langchain_base_tool()

    # Build the client first; it remains dependency-free and protocol-first.
    client = CorpusLangChainGraphClient(
        graph_adapter=graph_adapter,
        default_dialect=default_dialect,
        default_namespace=default_namespace,
        default_timeout_ms=default_timeout_ms,
        framework_version=framework_version,
        framework_translator=framework_translator,
    )

    # Create a single-tool wrapper for backwards compatibility with the existing
    # factory shape. This uses the same hardened serialization/bridge patterns
    # as create_langchain_graph_tools(...).
    tools = create_langchain_graph_tools(
        client,
        name_prefix="corpus",
        description_prefix="Corpus graph tool",
    )

    # The first tool is the query tool; rename it to match requested defaults.
    # This avoids duplicating tool code while keeping a stable factory API.
    query_tool = tools[0]
    try:
        query_tool.name = name
        if description is not None:
            query_tool.description = description
    except Exception:
        # If tool attributes are not assignable for some reason, fall back to
        # returning the tool as-is; this should still be functional.
        logger.debug("Failed to override name/description on LangChain tool", exc_info=True)

    # Ensure the returned object is a LangChain BaseTool instance when LangChain is installed.
    if not isinstance(query_tool, BaseTool):
        # This should not happen in normal environments; keep a defensive guard.
        raise RuntimeError("Unexpected tool type returned by create_langchain_graph_tools")

    return query_tool


__all__ = [
    "LangChainGraphClientProtocol",
    "CorpusLangChainGraphClient",
    "LangChainGraphFrameworkTranslator",
    "ErrorCodes",
    "with_graph_error_context",
    "with_async_graph_error_context",
    "with_error_context",
    "with_async_error_context",
    # Tool integration (soft import)
    "CorpusGraphTool",
    "create_langchain_graph_tools",
    "create_corpus_graph_tool",
]
