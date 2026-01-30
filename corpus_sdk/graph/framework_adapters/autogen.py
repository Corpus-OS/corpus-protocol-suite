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

Compatibility notes
-------------------
- AutoGen is an **optional dependency**. This module intentionally does not
  hard-import AutoGen packages at import time.
- Real AutoGen integration is provided through **soft-imported tool helpers**
  at the bottom of this file (e.g., `create_autogen_graph_tools()`).
  Importing this module does not require AutoGen to be installed.
- When AutoGen is installed, you can build AutoGen-native `FunctionTool`
  wrappers (from `autogen_core.tools`) around this client to run true end-to-end
  integration tests in AutoGen AgentChat/Core environments.

  Reference (AutoGen docs):
  - Tools overview: https://microsoft.github.io/autogen/stable//user-guide/core-user-guide/components/tools.html
  - FunctionTool API: https://microsoft.github.io/autogen/stable//reference/python/autogen_core.tools.html
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, is_dataclass
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

    # Fallback to a lightweight structural check (safe across typing modes).
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


def _json_safe_snapshot(value: Any, *, max_items: int = 200, max_str: int = 10_000) -> Any:
    """
    Best-effort conversion into a JSON-ish snapshot for tool-return values and logs.

    Security / correctness:
    - Limits container sizes to avoid memory bloat.
    - Truncates long strings.
    - Falls back to repr() for unknown objects.

    NOTE:
    - This is used only in optional AutoGen tool helpers below.
    - Core protocol logic continues to return protocol-level types (QueryResult, etc.).
    """
    try:
        # Fast paths for primitives (low overhead, safe to serialize).
        if value is None or isinstance(value, (bool, int, float)):
            return value

        # Bound string size to avoid log/tool payload bloat.
        if isinstance(value, str):
            return value if len(value) <= max_str else value[:max_str] + "…"

        # Map-like structures: bound number of items.
        if isinstance(value, Mapping):
            out: Dict[str, Any] = {}
            for i, (k, v) in enumerate(value.items()):
                if i >= max_items:
                    out["…"] = f"truncated after {max_items} items"
                    break
                out[str(k)] = _json_safe_snapshot(v, max_items=max_items, max_str=max_str)
            return out

        # Sequence-like structures: bound number of items.
        if isinstance(value, (list, tuple)):
            out_list: List[Any] = []
            for i, v in enumerate(value):
                if i >= max_items:
                    out_list.append(f"… truncated after {max_items} items")
                    break
                out_list.append(_json_safe_snapshot(v, max_items=max_items, max_str=max_str))
            return out_list

        # Dataclass objects: serialize via asdict (stable, deterministic).
        if is_dataclass(value):
            return _json_safe_snapshot(asdict(value), max_items=max_items, max_str=max_str)

        # Common protocol objects often expose to_dict(); use it if present.
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            return _json_safe_snapshot(to_dict(), max_items=max_items, max_str=max_str)

        # Final fallback: ensure representable without raising.
        return repr(value)
    except Exception:  # noqa: BLE001
        # Defensive: snapshotting must never throw in tool paths.
        return {"repr": repr(value)}


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
    just a subset of behaviors.
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
    """

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
        # Resolving adapter / graph_adapter with basic duck-typed validation.
        if graph_adapter is not None and adapter is not None and graph_adapter is not adapter:
            raise TypeError("Provide only one of 'adapter' or 'graph_adapter', not both")

        resolved_adapter: Any = graph_adapter if graph_adapter is not None else adapter
        if resolved_adapter is None:
            raise TypeError("adapter must be a GraphProtocolV1-compatible graph adapter")

        # Minimal duck-type check: we expect a GraphProtocolV1-like surface with
        # capabilities() and query(...) methods.
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

    # ------------------------------------------------------------------ #
    # Resource management (context managers)
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "CorpusAutoGenGraphClient":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    async def __aenter__(self) -> "CorpusAutoGenGraphClient":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.aclose()

    def close(self) -> None:
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
        if self._aclosed:
            return
        self._aclosed = True

        aclose_fn = getattr(self._graph, "aclose", None)
        if callable(aclose_fn):
            try:
                await aclose_fn()
                self._closed = True
                return
            except Exception:
                # Never let cleanup failures propagate to callers.
                logger.debug("Failed to async-close graph adapter", exc_info=True)

        if not self._closed:
            self.close()

    # ------------------------------------------------------------------ #
    # Translator (lazy, cached)
    # ------------------------------------------------------------------ #

    @cached_property
    def _translator(self) -> GraphTranslator:
        framework_translator: GraphFrameworkTranslator = (
            self._framework_translator_override or AutoGenGraphFrameworkTranslator()
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
            # Context translation must never break graph operations; proceed without ctx.
            logger.warning(
                "[%s] Failed to build OperationContext from AutoGen inputs; proceeding without OperationContext.",
                ErrorCodes.BAD_OPERATION_CONTEXT,
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
                "[%s] from_autogen returned non-OperationContext-like type: %s. Ignoring OperationContext.",
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
            logger.debug("Failed to enrich OperationContext attrs for AutoGen context", exc_info=True)

        return ctx_candidate  # type: ignore[return-value]

    def _framework_ctx(self, *, operation: str, namespace: Optional[str] = None) -> Mapping[str, Any]:
        ctx: Dict[str, Any] = {"framework": "autogen", "operation": operation}
        if self._framework_version is not None:
            ctx["framework_version"] = self._framework_version
        effective_namespace = namespace or self._default_namespace
        if effective_namespace is not None:
            ctx["namespace"] = effective_namespace
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
        effective_dialect = dialect or self._default_dialect
        effective_namespace = namespace or self._default_namespace
        effective_timeout = timeout_ms or self._default_timeout_ms

        raw: Dict[str, Any] = {"text": query, "params": dict(params or {}), "stream": bool(stream)}
        if effective_dialect is not None:
            raw["dialect"] = effective_dialect
        if effective_namespace is not None:
            raw["namespace"] = effective_namespace
        if effective_timeout is not None:
            raw["timeout_ms"] = int(effective_timeout)
        return raw

    def _validate_query_params(self, params: Optional[Mapping[str, Any]]) -> None:
        # Lightweight hardening: catch obvious misuse (e.g., passing a bare string).
        if params is not None and not isinstance(params, Mapping):
            raise TypeError(f"params must be a mapping (e.g. dict), not {type(params).__name__}")

    # ------------------------------------------------------------------ #
    # Capabilities / schema / health
    # ------------------------------------------------------------------ #

    @with_graph_error_context("capabilities_sync")
    def capabilities(self, **kwargs: Any) -> Dict[str, Any]:
        _ensure_not_in_event_loop("capabilities")
        caps = self._translator.capabilities()
        return graph_capabilities_to_dict(caps)

    @with_async_graph_error_context("capabilities_async")
    async def acapabilities(self, **kwargs: Any) -> Dict[str, Any]:
        caps = await self._translator.arun_capabilities()
        return graph_capabilities_to_dict(caps)

    @with_graph_error_context("get_schema_sync")
    def get_schema(
        self,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> GraphSchema:
        _ensure_not_in_event_loop("get_schema")
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        schema = self._translator.get_schema(op_ctx=ctx, framework_ctx=self._framework_ctx(operation="get_schema"))
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
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        schema = await self._translator.arun_get_schema(op_ctx=ctx, framework_ctx=self._framework_ctx(operation="get_schema"))
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
        _ensure_not_in_event_loop("health")
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        health_result = self._translator.health(op_ctx=ctx, framework_ctx=self._framework_ctx(operation="health"))
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
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        health_result = await self._translator.arun_health(op_ctx=ctx, framework_ctx=self._framework_ctx(operation="health"))
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
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> QueryResult:
        _ensure_not_in_event_loop("query")
        validate_graph_query(query, operation="query", error_code="INVALID_QUERY")
        self._validate_query_params(params)

        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        raw_query = self._build_raw_query(
            query=query,
            params=params,
            dialect=dialect,
            namespace=namespace,
            timeout_ms=timeout_ms,
            stream=False,
        )
        framework_ctx = self._framework_ctx(operation="query", namespace=namespace)

        try:
            result = self._translator.query(raw_query, op_ctx=ctx, framework_ctx=framework_ctx, mmr_config=None)
        except NotSupported:
            # Graceful handling of unsupported dialects: retry once without an explicit dialect.
            if dialect is not None:
                fallback_raw = dict(raw_query)
                fallback_raw.pop("dialect", None)
                result = self._translator.query(fallback_raw, op_ctx=ctx, framework_ctx=framework_ctx, mmr_config=None)
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
        validate_graph_query(query, operation="aquery", error_code="INVALID_QUERY")
        self._validate_query_params(params)

        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        raw_query = self._build_raw_query(
            query=query,
            params=params,
            dialect=dialect,
            namespace=namespace,
            timeout_ms=timeout_ms,
            stream=False,
        )
        framework_ctx = self._framework_ctx(operation="query", namespace=namespace)

        try:
            result = await self._translator.arun_query(raw_query, op_ctx=ctx, framework_ctx=framework_ctx, mmr_config=None)
        except NotSupported:
            # Graceful handling of unsupported dialects: retry once without an explicit dialect.
            if dialect is not None:
                fallback_raw = dict(raw_query)
                fallback_raw.pop("dialect", None)
                result = await self._translator.arun_query(fallback_raw, op_ctx=ctx, framework_ctx=framework_ctx, mmr_config=None)
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
        _ensure_not_in_event_loop("stream_query")
        validate_graph_query(query, operation="stream_query", error_code="INVALID_QUERY")
        self._validate_query_params(params)

        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        raw_query = self._build_raw_query(
            query=query,
            params=params,
            dialect=dialect,
            namespace=namespace,
            timeout_ms=timeout_ms,
            stream=True,
        )
        framework_ctx = self._framework_ctx(operation="stream_query", namespace=namespace)

        for chunk in self._translator.query_stream(raw_query, op_ctx=ctx, framework_ctx=framework_ctx):
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
        validate_graph_query(query, operation="astream_query", error_code="INVALID_QUERY")
        self._validate_query_params(params)

        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        raw_query = self._build_raw_query(
            query=query,
            params=params,
            dialect=dialect,
            namespace=namespace,
            timeout_ms=timeout_ms,
            stream=True,
        )
        framework_ctx = self._framework_ctx(operation="stream_query", namespace=namespace)

        async for chunk in self._translator.arun_query_stream(raw_query, op_ctx=ctx, framework_ctx=framework_ctx):
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
        _ensure_not_in_event_loop("upsert_nodes")
        validate_upsert_nodes_spec(spec)

        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        framework_ctx = self._framework_ctx(operation="upsert_nodes", namespace=getattr(spec, "namespace", None))

        result = self._translator.upsert_nodes(spec.nodes, op_ctx=ctx, framework_ctx=framework_ctx)
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
        validate_upsert_nodes_spec(spec)

        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        framework_ctx = self._framework_ctx(operation="upsert_nodes", namespace=getattr(spec, "namespace", None))

        result = await self._translator.arun_upsert_nodes(spec.nodes, op_ctx=ctx, framework_ctx=framework_ctx)
        return validate_graph_result_type(
            result,
            expected_type=UpsertResult,
            operation="GraphTranslator.arun_upsert_nodes",
            error_code=ErrorCodes.BAD_UPSERT_RESULT,
        )

    # NOTE: Upsert edges validation remains in shared layers or adapter-level validation.
    # If you need strict local validation like embeddings, you can add it here, but we
    # deliberately avoid non-essential changes to prevent test regressions.

    @with_graph_error_context("upsert_edges_sync")
    def upsert_edges(
        self,
        spec: UpsertEdgesSpec,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> UpsertResult:
        _ensure_not_in_event_loop("upsert_edges")

        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        framework_ctx = self._framework_ctx(operation="upsert_edges", namespace=getattr(spec, "namespace", None))

        result = self._translator.upsert_edges(spec.edges, op_ctx=ctx, framework_ctx=framework_ctx)
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
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        framework_ctx = self._framework_ctx(operation="upsert_edges", namespace=getattr(spec, "namespace", None))

        result = await self._translator.arun_upsert_edges(spec.edges, op_ctx=ctx, framework_ctx=framework_ctx)
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
        _ensure_not_in_event_loop("delete_nodes")

        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        framework_ctx = self._framework_ctx(operation="delete_nodes", namespace=getattr(spec, "namespace", None))

        raw_filter_or_ids: Any
        if spec.filter is not None:
            raw_filter_or_ids = spec.filter
        else:
            ids = list(spec.ids or [])
            if not ids:
                raise BadRequest(
                    "DeleteNodesSpec must specify either filter or non-empty ids",
                    code=ErrorCodes.BAD_ADAPTER_RESULT,
                )
            raw_filter_or_ids = ids

        result = self._translator.delete_nodes(raw_filter_or_ids, op_ctx=ctx, framework_ctx=framework_ctx)
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
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        framework_ctx = self._framework_ctx(operation="delete_nodes", namespace=getattr(spec, "namespace", None))

        raw_filter_or_ids: Any
        if spec.filter is not None:
            raw_filter_or_ids = spec.filter
        else:
            ids = list(spec.ids or [])
            if not ids:
                raise BadRequest(
                    "DeleteNodesSpec must specify either filter or non-empty ids",
                    code=ErrorCodes.BAD_ADAPTER_RESULT,
                )
            raw_filter_or_ids = ids

        result = await self._translator.arun_delete_nodes(raw_filter_or_ids, op_ctx=ctx, framework_ctx=framework_ctx)
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
        _ensure_not_in_event_loop("delete_edges")

        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        framework_ctx = self._framework_ctx(operation="delete_edges", namespace=getattr(spec, "namespace", None))

        raw_filter_or_ids: Any
        if spec.filter is not None:
            raw_filter_or_ids = spec.filter
        else:
            ids = list(spec.ids or [])
            if not ids:
                raise BadRequest(
                    "DeleteEdgesSpec must specify either filter or non-empty ids",
                    code=ErrorCodes.BAD_ADAPTER_RESULT,
                )
            raw_filter_or_ids = ids

        result = self._translator.delete_edges(raw_filter_or_ids, op_ctx=ctx, framework_ctx=framework_ctx)
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
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        framework_ctx = self._framework_ctx(operation="delete_edges", namespace=getattr(spec, "namespace", None))

        raw_filter_or_ids: Any
        if spec.filter is not None:
            raw_filter_or_ids = spec.filter
        else:
            ids = list(spec.ids or [])
            if not ids:
                raise BadRequest(
                    "DeleteEdgesSpec must specify either filter or non-empty ids",
                    code=ErrorCodes.BAD_ADAPTER_RESULT,
                )
            raw_filter_or_ids = ids

        result = await self._translator.arun_delete_edges(raw_filter_or_ids, op_ctx=ctx, framework_ctx=framework_ctx)
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
        _ensure_not_in_event_loop("bulk_vertices")

        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        raw_request: Mapping[str, Any] = {
            "namespace": spec.namespace,
            "limit": spec.limit,
            "cursor": spec.cursor,
            "filter": spec.filter,
        }
        framework_ctx = self._framework_ctx(operation="bulk_vertices", namespace=spec.namespace)

        result = self._translator.bulk_vertices(raw_request, op_ctx=ctx, framework_ctx=framework_ctx)
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
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        raw_request: Mapping[str, Any] = {
            "namespace": spec.namespace,
            "limit": spec.limit,
            "cursor": spec.cursor,
            "filter": spec.filter,
        }
        framework_ctx = self._framework_ctx(operation="bulk_vertices", namespace=spec.namespace)

        result = await self._translator.arun_bulk_vertices(raw_request, op_ctx=ctx, framework_ctx=framework_ctx)
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
        _ensure_not_in_event_loop("traversal")

        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
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
        framework_ctx = self._framework_ctx(operation="traversal", namespace=spec.namespace)

        result = self._translator.traversal(raw_request, op_ctx=ctx, framework_ctx=framework_ctx)
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
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
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
        framework_ctx = self._framework_ctx(operation="traversal", namespace=spec.namespace)

        result = await self._translator.arun_traversal(raw_request, op_ctx=ctx, framework_ctx=framework_ctx)
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
        _ensure_not_in_event_loop("transaction")
        validate_batch_operations(ops, operation="transaction", error_code="INVALID_BATCH_OPS")

        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        raw_ops: List[Mapping[str, Any]] = [{"op": op.op, "args": dict(op.args or {})} for op in ops]

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
        validate_batch_operations(ops, operation="atransaction", error_code="INVALID_BATCH_OPS")

        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        raw_ops: List[Mapping[str, Any]] = [{"op": op.op, "args": dict(op.args or {})} for op in ops]

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
        _ensure_not_in_event_loop("batch")
        validate_batch_operations(ops, operation="batch", error_code="INVALID_BATCH_OPS")

        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        raw_batch_ops: List[Mapping[str, Any]] = [{"op": op.op, "args": dict(op.args or {})} for op in ops]

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
        validate_batch_operations(ops, operation="abatch", error_code="INVALID_BATCH_OPS")

        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        raw_batch_ops: List[Mapping[str, Any]] = [{"op": op.op, "args": dict(op.args or {})} for op in ops]

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
# Optional AutoGen integration helpers (soft import)
# ---------------------------------------------------------------------------

def create_autogen_graph_tools(
    client: "CorpusAutoGenGraphClient",
    *,
    name_prefix: str = "graph",
    description_prefix: str = "Corpus graph tool",
) -> List[Any]:
    """
    Create AutoGen-native FunctionTool wrappers for common graph operations.

    Why this exists:
    - The graph adapter itself is intentionally dependency-free and framework-light.
    - When AutoGen is installed, callers often want real AutoGen `Tool` objects
      to register on agents (e.g., AgentChat assistants).
    - The embedding adapter provides an AutoGen wiring helper via `create_vector_memory()`;
      this mirrors that approach for graph operations.

    Soft dependency:
    - AutoGen is imported lazily. Importing this module does not require AutoGen.
    - If AutoGen is not installed, this function raises a clear RuntimeError with install instructions.

    Notes:
    - Tool functions are async to avoid calling sync APIs inside event loops.
    - Return values are JSON-safe snapshots for tool-calling compatibility.
      (Protocol-level methods still return QueryResult/QueryChunk/etc. normally.)

    Reference (AutoGen docs):
    - Tools overview: https://microsoft.github.io/autogen/stable//user-guide/core-user-guide/components/tools.html
    - FunctionTool API: https://microsoft.github.io/autogen/stable//reference/python/autogen_core.tools.html
    """
    try:
        # AutoGen tool system (Core) – FunctionTool wraps a Python function as a tool.
        # Most recent modular AutoGen releases provide this at autogen_core.tools.
        from autogen_core.tools import FunctionTool  # type: ignore[import-not-found]
    except ImportError as exc:  # noqa: BLE001
        # Some environments may have legacy "pyautogen" installed instead of modular autogen-core.
        # We intentionally do not attempt to hard-depend on legacy packages here.
        raise RuntimeError(
            "AutoGen tool dependencies are not installed. Install with:\n"
            '  pip install -U "autogen-core" "autogen-agentchat"\n'
            "Then retry create_autogen_graph_tools(...)."
        ) from exc

    # ----------------------------
    # Tool implementations (async)
    # ----------------------------

    async def graph_query(
        query: str,
        params: Optional[Mapping[str, Any]] = None,
        dialect: Optional[str] = None,
        namespace: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        """
        Execute a graph query and return a JSON-safe summary.
        """
        res = await client.aquery(
            query,
            params=params,
            dialect=dialect,
            namespace=namespace,
            timeout_ms=timeout_ms,
            conversation=conversation,
            extra_context=extra_context,
        )
        return {"result": _json_safe_snapshot(res)}

    async def graph_stream_query(
        query: str,
        params: Optional[Mapping[str, Any]] = None,
        dialect: Optional[str] = None,
        namespace: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
        max_chunks: int = 25,
    ) -> Mapping[str, Any]:
        """
        Execute a streaming graph query and return up to `max_chunks` chunks.

        This is intentionally bounded to avoid runaway streams in tool calls.
        """
        chunks: List[Any] = []

        # Defensive normalization: enforce a non-negative bound.
        # (This is tool-only behavior; it does not affect the protocol API.)
        limit = int(max_chunks)
        if limit < 0:
            raise ValueError("max_chunks must be >= 0")

        aiter = client.astream_query(
            query,
            params=params,
            dialect=dialect,
            namespace=namespace,
            timeout_ms=timeout_ms,
            conversation=conversation,
            extra_context=extra_context,
        )

        count = 0
        async for ch in aiter:
            if count >= limit:
                break
            chunks.append(_json_safe_snapshot(ch))
            count += 1

        return {"chunks": chunks, "truncated": count >= limit and limit != 0}

    async def graph_bulk_vertices(
        namespace: Optional[str] = None,
        limit: int = 50,
        cursor: Optional[str] = None,
        filter: Optional[Mapping[str, Any]] = None,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        """
        Bulk-scan vertices (paged) and return a JSON-safe result snapshot.
        """
        spec = BulkVerticesSpec(namespace=namespace, limit=limit, cursor=cursor, filter=filter)
        res = await client.abulk_vertices(spec, conversation=conversation, extra_context=extra_context)
        return {"result": _json_safe_snapshot(res)}

    async def graph_batch(
        ops: Sequence[Mapping[str, Any]],
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        """
        Execute a batch of graph operations.

        Input ops format:
            [{"op": "<name>", "args": {...}}, ...]
        """
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

        # NOTE: We intentionally do not call validate_batch_operations() here because:
        # - This tool surface accepts mapping-based ops that may be "looser" than internal callers.
        # - Core protocol methods already validate via validate_batch_operations() before execution.
        res = await client.abatch(batch_ops, conversation=conversation, extra_context=extra_context)
        return {"result": _json_safe_snapshot(res)}

    # ----------------------------
    # Tool registration
    # ----------------------------

    tools: List[Any] = []

    tools.append(
        FunctionTool(
            graph_query,
            description=f"{description_prefix}: execute a graph query (non-streaming).",
            name=f"{name_prefix}_query",
        )
    )
    tools.append(
        FunctionTool(
            graph_stream_query,
            description=f"{description_prefix}: execute a graph query with streaming chunks (bounded).",
            name=f"{name_prefix}_stream_query",
        )
    )
    tools.append(
        FunctionTool(
            graph_bulk_vertices,
            description=f"{description_prefix}: bulk-scan vertices with pagination inputs.",
            name=f"{name_prefix}_bulk_vertices",
        )
    )
    tools.append(
        FunctionTool(
            graph_batch,
            description=f"{description_prefix}: execute a batch of graph operations.",
            name=f"{name_prefix}_batch",
        )
    )

    return tools


__all__ = [
    "AutoGenGraphClientProtocol",
    "AutoGenGraphFrameworkTranslator",
    "CorpusAutoGenGraphClient",
    "ErrorCodes",
    "with_graph_error_context",
    "with_async_graph_error_context",
    "with_error_context",
    "with_async_error_context",
    # Optional AutoGen integration helper (soft import)
    "create_autogen_graph_tools",
]
