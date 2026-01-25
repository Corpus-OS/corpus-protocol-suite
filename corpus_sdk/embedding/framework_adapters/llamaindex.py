# corpus_sdk/embedding/framework_adapters/llamaindex.py
# SPDX-License-Identifier: Apache-2.0

"""
LlamaIndex adapter for Corpus Embedding protocol.

This module exposes Corpus `EmbeddingProtocolV1` implementations as
`llama_index.core.embeddings.BaseEmbedding`, with:

- Full compatibility with LlamaIndex's Settings configuration
- Support for LlamaIndex node-based document processing and chunking strategies
- Context normalization using existing `context_translation.from_llamaindex`
- Framework-agnostic orchestration via `EmbeddingTranslator`
- Async → sync bridging handled in the common embedding layer
- Rich error context attachment for observability

Design notes / philosophy
-------------------------
- **Protocol-first**: we require only an `embed` method (duck-typed) instead of
  strict inheritance from a specific adapter base class.
- **Resilient to framework evolution**: LlamaIndex’s internals and signatures
  change; we filter/normalize context defensively and keep our adapter surface stable.
- **Observability-first**: all embedding operations attach rich error context:
  framework identity, model info, batch sizes, node IDs, trace/workflow IDs, etc.
- **Fail-safe context translation**: context translation must never break embeddings.
  If translation fails, we proceed without `OperationContext` and attach diagnostic context.
- **Strict by default** (configurable): non-string inputs in batch operations are rejected
  unless `strict_text_types=False`, in which case they are treated as empty and receive
  zero-vector embeddings to preserve row alignment.

Resilience (retries, caching, rate limiting, etc.) is expected to be provided
by the underlying adapter, typically a BaseEmbeddingAdapter subclass.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from functools import cached_property, wraps
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    TypedDict,
)

from corpus_sdk.core.context_translation import (
    from_llamaindex as context_from_llamaindex,
)
from corpus_sdk.core.error_context import attach_context
from corpus_sdk.embedding.embedding_base import (
    EmbeddingProtocolV1,
    OperationContext,
)
from corpus_sdk.embedding.framework_adapters.common.embedding_translation import (
    BatchConfig,
    EmbeddingTranslator,
    TextNormalizationConfig,
    create_embedding_translator,
)
from corpus_sdk.embedding.framework_adapters.common.framework_utils import (
    CoercionErrorCodes,
    coerce_embedding_matrix,
    coerce_embedding_vector,
    warn_if_extreme_batch,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")

# ---------------------------------------------------------------------------
# Framework identity & version
# ---------------------------------------------------------------------------

_FRAMEWORK_NAME = "llamaindex"

# Best-effort LlamaIndex version detection (never hard-fail on import errors)
try:
    import llama_index  # type: ignore

    _FRAMEWORK_VERSION: Optional[str] = getattr(llama_index, "__version__", None)
except Exception:  # noqa: BLE001
    _FRAMEWORK_VERSION = None

# ---------------------------------------------------------------------------
# Safe conditional imports for LlamaIndex
# ---------------------------------------------------------------------------
#
# Rationale:
# - This module should be importable even if LlamaIndex is not installed,
#   since Corpus SDK may be used in environments that support multiple frameworks.
# - We provide minimal stubs so type checking and import-time evaluation work.
# - The tests are written to skip interface checks when LLAMAINDEX_AVAILABLE=False.
#

try:
    from llama_index.core.embeddings import BaseEmbedding  # type: ignore
    from llama_index.core.callbacks import CallbackManager  # type: ignore

    # DEFAULT_EMBED_BATCH_SIZE has moved across LlamaIndex versions; never hard-fail.
    try:
        from llama_index.core.embeddings import DEFAULT_EMBED_BATCH_SIZE  # type: ignore
    except Exception:  # noqa: BLE001
        DEFAULT_EMBED_BATCH_SIZE = 512  # type: ignore[assignment]

    LLAMAINDEX_AVAILABLE = True
except ImportError:  # pragma: no cover

    class BaseEmbedding:  # type: ignore[no-redef]
        """
        Minimal fallback BaseEmbedding when LlamaIndex is not installed.

        Note: This is only to prevent import-time failures. Real usage in a
        non-LlamaIndex environment is a misconfiguration (tests skip).
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class CallbackManager:  # type: ignore[no-redef]
        """Fallback CallbackManager stub when LlamaIndex is not installed."""
        pass

    DEFAULT_EMBED_BATCH_SIZE = 512
    LLAMAINDEX_AVAILABLE = False

# ---------------------------------------------------------------------------
# Error codes (aligned with other embedding adapters)
# ---------------------------------------------------------------------------


class ErrorCodes:
    """
    Error code constants for the LlamaIndex embedding adapter.

    Notes:
    - The shared coercion helpers use `EMBEDDING_COERCION_ERROR_CODES`, a
      `CoercionErrorCodes` instance derived from these values.
    - Framework-specific codes are used in logs and attached context.
    """

    # Coercion-level (used by common framework_utils)
    INVALID_EMBEDDING_RESULT = "INVALID_EMBEDDING_RESULT"
    EMPTY_EMBEDDING_RESULT = "EMPTY_EMBEDDING_RESULT"
    EMBEDDING_CONVERSION_ERROR = "EMBEDDING_CONVERSION_ERROR"

    # LlamaIndex-specific context/config errors
    LLAMAINDEX_CONTEXT_INVALID = "LLAMAINDEX_CONTEXT_INVALID"
    LLAMAINDEX_CONFIG_INVALID = "LLAMAINDEX_CONFIG_INVALID"

    # Sync wrapper misuse errors (parity with other adapters)
    SYNC_WRAPPER_CALLED_IN_EVENT_LOOP = "SYNC_WRAPPER_CALLED_IN_EVENT_LOOP"


# Shared coercion configuration used across all embedding operations.
EMBEDDING_COERCION_ERROR_CODES: CoercionErrorCodes = CoercionErrorCodes(
    invalid_result=ErrorCodes.INVALID_EMBEDDING_RESULT,
    empty_result=ErrorCodes.EMPTY_EMBEDDING_RESULT,
    conversion_error=ErrorCodes.EMBEDDING_CONVERSION_ERROR,
    framework_label=_FRAMEWORK_NAME,
)

# ---------------------------------------------------------------------------
# Typed contexts / config (intentionally small + stable)
# ---------------------------------------------------------------------------


class LlamaIndexContext(TypedDict, total=False):
    """
    Structured type for LlamaIndex execution context.

    LlamaIndex frequently passes kwargs through multiple layers; we only
    accept/forward the fields that are useful and stable:
    - node_ids: ids of nodes/chunks being embedded
    - index_id: identifier for the index
    - callback_manager: LlamaIndex callback manager
    - trace_id: tracing identifier
    - workflow: user/framework workflow hint
    """

    node_ids: Optional[List[str]]
    index_id: Optional[str]
    callback_manager: Optional[Any]
    trace_id: Optional[str]
    workflow: Optional[str]


class LlamaIndexAdapterConfig(TypedDict, total=False):
    """
    Adapter-level configuration for the LlamaIndex embedding adapter.

    This is *not* per-call LlamaIndex BaseEmbedding kwargs; it configures adapter behavior.

    Fields
    ------
    enable_operation_context_propagation:
        If True, include the OperationContext instance in framework_ctx as
        `_operation_context` for downstream inspection. Defaults to True.

    strict_text_types:
        If True, reject non-string items in batch embedding calls rather than
        silently treating them as empty. Defaults to True.

    max_node_ids_in_context:
        Maximum number of node_ids to include in framework_ctx and error context
        to prevent log bloat. Defaults to 100.
    """

    enable_operation_context_propagation: bool
    strict_text_types: bool
    max_node_ids_in_context: int


# ---------------------------------------------------------------------------
# Safety / robustness utilities
# ---------------------------------------------------------------------------


def _validate_texts_are_strings(texts: Sequence[Any], *, op_name: str) -> None:
    """
    Fail fast if a caller provides non-string items.

    We intentionally do not coerce arbitrary objects to str here, because that can
    silently embed repr() outputs and lead to confusing retrieval behavior.
    """
    for i, t in enumerate(texts):
        if not isinstance(t, str):
            raise TypeError(
                f"{op_name} expects Sequence[str]; item {i} is {type(t).__name__}",
            )


def _safe_snapshot(value: Any, *, max_items: int = 200, max_str: int = 5_000) -> Any:
    """
    Best-effort conversion into a JSON-ish, safe-to-log snapshot.

    - Limits container size to reduce log bloat
    - Truncates long strings
    - Falls back to repr() for unknown objects
    """
    try:
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return value if len(value) <= max_str else value[:max_str] + "…"
        if isinstance(value, Mapping):
            out: Dict[str, Any] = {}
            for idx, (k, v) in enumerate(value.items()):
                if idx >= max_items:
                    out["…"] = f"truncated after {max_items} items"
                    break
                out[str(k)] = _safe_snapshot(v, max_items=max_items, max_str=max_str)
            return out
        # Include set for improved observability robustness (e.g., tags)
        if isinstance(value, (list, tuple, set)):
            out_list: List[Any] = []
            for idx, v in enumerate(list(value)):
                if idx >= max_items:
                    out_list.append(f"… truncated after {max_items} items")
                    break
                out_list.append(_safe_snapshot(v, max_items=max_items, max_str=max_str))
            return out_list
        return repr(value)
    except Exception:  # noqa: BLE001
        return {"repr": repr(value)}


def _looks_like_operation_context(obj: Any) -> bool:
    """
    OperationContext may be a concrete class or a Protocol/alias depending on runtime.

    Prefer isinstance when it works; fall back to a lightweight structural heuristic
    aligned with corpus_sdk.embedding.embedding_base.OperationContext.
    """
    if obj is None:
        return False
    try:
        if isinstance(obj, OperationContext):
            return True
    except TypeError:
        # OperationContext may be a Protocol/typing alias at runtime
        pass

    # Structural heuristic aligned with OperationContext fields/behaviors.
    return any(
        hasattr(obj, attr)
        for attr in (
            "request_id",
            "idempotency_key",
            "deadline_ms",
            "traceparent",
            "tenant",
            "attrs",
            "remaining_ms",
            "to_dict",  # tolerated if some impls provide it
        )
    )


def _infer_dim_from_matrix(mat: List[List[float]]) -> Optional[int]:
    """Best-effort embedding dimension inference from a 2D embedding matrix."""
    if not mat:
        return None
    first = mat[0]
    if not isinstance(first, list):
        return None
    return len(first)


def _filter_llamaindex_context_from_kwargs(kwargs: Mapping[str, Any]) -> LlamaIndexContext:
    """
    Extract only known LlamaIndex context keys from BaseEmbedding kwargs.

    LlamaIndex can pass a wide variety of kwargs depending on execution path; we keep
    this adapter robust by filtering to known keys for framework_ctx and context translation.
    """
    ctx: LlamaIndexContext = {}
    for key in ("node_ids", "index_id", "callback_manager", "trace_id", "workflow"):
        if key in kwargs:
            ctx[key] = kwargs[key]  # type: ignore[literal-required]

    # Normalize node_ids to a List[str] when feasible (some flows pass tuples/iterables).
    node_ids = ctx.get("node_ids")
    if node_ids is not None and not isinstance(node_ids, list):
        try:
            ctx["node_ids"] = list(node_ids)  # type: ignore[assignment,arg-type]
        except Exception:
            pass

    return ctx


def _suggest_async_name(sync_api_name: str) -> str:
    """
    Provide an accurate async-method hint for error messages.

    LlamaIndex naming conventions:
    - _get_*  -> _aget_*
    - health/capabilities -> ahealth/acapabilities
    """
    if sync_api_name.startswith("_get_"):
        return sync_api_name.replace("_get_", "_aget_", 1)
    if sync_api_name.startswith("_get"):
        return "_a" + sync_api_name[1:]
    return "a" + sync_api_name


def _ensure_not_in_event_loop(sync_api_name: str) -> None:
    """
    Prevent deadlocks from calling sync APIs in async contexts.

    This guard enforces a clear contract:
    - In async code, use async variants (`_aget_*`, `ahealth`, `acapabilities`).
    - In sync code, use the sync methods directly.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running event loop: safe to call sync method.
        return

    async_hint = _suggest_async_name(sync_api_name)
    raise RuntimeError(
        f"{sync_api_name} was called from inside an active asyncio event loop. "
        f"Use the async variant instead (e.g. 'await {async_hint}()'). "
        f"[{ErrorCodes.SYNC_WRAPPER_CALLED_IN_EVENT_LOOP}]"
    )


def _maybe_close_sync(obj: Any) -> None:
    """
    Best-effort *sync* resource cleanup.

    Preference:
      1) aclose() if present (awaited via asyncio.run if coroutine)
      2) close() if present:
           - awaited via asyncio.run if coroutinefunction
           - called directly if sync
           - awaited via asyncio.run if sync returns a coroutine

    IMPORTANT:
      Callers must ensure they are NOT in a running event loop (use _ensure_not_in_event_loop).
    """
    if obj is None:
        return

    aclose = getattr(obj, "aclose", None)
    if callable(aclose):
        res = aclose()
        if asyncio.iscoroutine(res):
            asyncio.run(res)
        return

    close = getattr(obj, "close", None)
    if not callable(close):
        return

    if asyncio.iscoroutinefunction(close):
        asyncio.run(close())
        return

    res = close()
    if asyncio.iscoroutine(res):
        asyncio.run(res)


async def _maybe_close_async(obj: Any) -> None:
    """
    Best-effort async resource cleanup with prioritization.

    - Prefer an async `aclose()` method if present.
    - Fall back to a coroutine `close()` if the object defines one.
    - Fall back to a sync `close()` executed in a worker thread.
    """
    if obj is None:
        return

    aclose = getattr(obj, "aclose", None)
    if callable(aclose):
        res = aclose()
        if asyncio.iscoroutine(res):
            await res
        return

    close = getattr(obj, "close", None)
    if not callable(close):
        return

    if asyncio.iscoroutinefunction(close):
        await close()
    else:
        await asyncio.to_thread(close)


# ---------------------------------------------------------------------------
# Error-context decorators with dynamic context extraction
# ---------------------------------------------------------------------------


def _extract_dynamic_context(
    instance: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    operation: str,
) -> Dict[str, Any]:
    """
    Extract rich dynamic context from method call for enhanced observability.

    Provides per-call metrics:
    - text_len for single-text operations
    - texts_count / empty_texts_count for batch operations
    - node_ids (bounded), node_count, index_id, callback_manager presence
    - trace_id, workflow
    - model/model_name from the embedding instance
    - embedding_dim_hint when available
    """
    dynamic_ctx: Dict[str, Any] = {
        "model": getattr(instance, "model_name", "unknown"),
        "model_name": getattr(instance, "model_name", "unknown"),
        "framework_version": _FRAMEWORK_VERSION,
    }

    dim_hint = getattr(instance, "_embedding_dim_hint", None)
    if isinstance(dim_hint, int):
        dynamic_ctx["embedding_dim"] = dim_hint

    # Text-based metrics:
    # - For "query"/"text": args[0] is typically the string
    # - For "texts": args[0] is typically a sequence
    if operation in ("query", "text") and args and isinstance(args[0], str):
        dynamic_ctx["text_len"] = len(args[0])
    elif operation == "texts" and args:
        maybe_texts = args[0]
        # Strings are Sequences; guard to avoid miscounting characters as batch size.
        if isinstance(maybe_texts, Sequence) and not isinstance(maybe_texts, (str, bytes)):
            # More defensive extraction; never let extraction break embeddings.
            try:
                dynamic_ctx["texts_count"] = len(maybe_texts)  # type: ignore[arg-type]
            except Exception:
                pass
            try:
                empty_count = sum(
                    1
                    for text in maybe_texts  # type: ignore[assignment]
                    if not isinstance(text, str) or not text.strip()
                )
                if empty_count:
                    dynamic_ctx["empty_texts_count"] = empty_count
            except Exception:
                pass

    ctx = _filter_llamaindex_context_from_kwargs(kwargs)

    # Node IDs: include a bounded subset, plus counts/truncation info.
    if ctx.get("node_ids") is not None:
        node_ids = ctx.get("node_ids") or []
        max_nodes = getattr(instance, "_max_node_ids_in_context", 100)
        bounded = node_ids[:max_nodes]
        dynamic_ctx["node_ids"] = bounded
        dynamic_ctx["node_count"] = len(node_ids)
        if len(node_ids) > len(bounded):
            dynamic_ctx["node_ids_truncated"] = True

    # Common tracing / workflow fields
    for key in ("index_id", "trace_id", "workflow"):
        if key in ctx:
            dynamic_ctx[key] = ctx[key]  # type: ignore[literal-required]

    if "callback_manager" in ctx:
        dynamic_ctx["has_callback_manager"] = bool(ctx.get("callback_manager"))

    return dynamic_ctx


def _create_error_context_decorator(
    operation: str,
    is_async: bool = False,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Factory for creating error context decorators with rich per-call metrics.

    Notes:
    - This mirrors the decorator-based pattern in other framework adapters for consistency.
    - We attach both static and dynamic context for debuggability and observability.
    - We never swallow the exception; we enrich it via attach_context and re-raise.
    """

    def decorator_factory(**static_context: Any) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            if is_async:

                @wraps(func)
                async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
                    dynamic_context = _extract_dynamic_context(self, args, kwargs, operation)
                    full_context = {
                        **static_context,
                        **dynamic_context,
                        "framework_version": _FRAMEWORK_VERSION,
                    }
                    try:
                        return await func(self, *args, **kwargs)
                    except Exception as exc:  # noqa: BLE001
                        attach_context(
                            exc,
                            framework=_FRAMEWORK_NAME,
                            operation=f"embedding_{operation}",
                            **full_context,
                        )
                        raise

                return async_wrapper

            @wraps(func)
            def sync_wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
                dynamic_context = _extract_dynamic_context(self, args, kwargs, operation)
                full_context = {
                    **static_context,
                    **dynamic_context,
                    "framework_version": _FRAMEWORK_VERSION,
                }
                try:
                    return func(self, *args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    attach_context(
                        exc,
                        framework=_FRAMEWORK_NAME,
                        operation=f"embedding_{operation}",
                        **full_context,
                    )
                    raise

            return sync_wrapper

        return decorator

    return decorator_factory


# Convenience decorators with rich context extraction
def with_embedding_error_context(operation: str, **static_context: Any) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for sync methods with rich dynamic context extraction."""
    static_context.setdefault("error_codes", EMBEDDING_COERCION_ERROR_CODES)
    static_context.setdefault("framework_version", _FRAMEWORK_VERSION)
    return _create_error_context_decorator(operation, is_async=False)(**static_context)


def with_async_embedding_error_context(operation: str, **static_context: Any) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for async methods with rich dynamic context extraction."""
    static_context.setdefault("error_codes", EMBEDDING_COERCION_ERROR_CODES)
    static_context.setdefault("framework_version", _FRAMEWORK_VERSION)
    return _create_error_context_decorator(operation, is_async=True)(**static_context)


# ---------------------------------------------------------------------------
# Main adapter class
# ---------------------------------------------------------------------------


class CorpusLlamaIndexEmbeddings(BaseEmbedding):
    """
    LlamaIndex `BaseEmbedding` backed by a Corpus `EmbeddingProtocolV1` adapter.

    LlamaIndex-Specific Responsibilities
    ------------------------------------
    - Integrate with LlamaIndex's global `Settings` for embedding configuration
    - Support LlamaIndex node-based document processing and chunking strategies
    - Provide embeddings for both document nodes and query text
    - Work with LlamaIndex's callback system for observability
    - Support LlamaIndex's async patterns for high-performance retrieval
    - Handle LlamaIndex service context and configuration patterns

    All embedding logic lives in:
    - `corpus_sdk.embedding.framework_adapters.common.embedding_translation`
    - Concrete `EmbeddingProtocolV1` adapter implementations.
    """

    def __init__(
        self,
        corpus_adapter: EmbeddingProtocolV1,
        model_name: str = "corpus-embedding-protocol",
        batch_config: Optional[BatchConfig] = None,
        text_normalization_config: Optional[TextNormalizationConfig] = None,
        callback_manager: Optional[CallbackManager] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        *,
        llamaindex_config: Optional[LlamaIndexAdapterConfig] = None,
        embedding_dimension: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Corpus LlamaIndex Embeddings.

        Parameters
        ----------
        corpus_adapter:
            Underlying Corpus embedding adapter implementing EmbeddingProtocolV1 (duck-typed).
        model_name:
            LlamaIndex-facing model identifier (used by Settings and observability).
        batch_config / text_normalization_config:
            Shared common-layer knobs for batching and normalization.
        callback_manager:
            LlamaIndex callback manager (optional).
        embed_batch_size:
            LlamaIndex batch-size hint. Must be positive.
        llamaindex_config:
            Adapter-level configuration (separate from per-call kwargs).
        embedding_dimension:
            Optional explicit embedding dimension override. Required if the adapter
            does not implement `get_embedding_dimension()`.
        """
        # Behavioral validation (duck-typed) instead of strict isinstance
        if not hasattr(corpus_adapter, "embed") or not callable(getattr(corpus_adapter, "embed", None)):
            raise TypeError(
                "corpus_adapter must implement an EmbeddingProtocolV1-compatible "
                "interface with an 'embed' method"
            )

        if embed_batch_size < 1:
            raise ValueError("embed_batch_size must be positive")

        # Validate + normalize llamaindex_config
        if llamaindex_config is None:
            llamaindex_config = {}
        if not isinstance(llamaindex_config, Mapping):
            raise ValueError(
                f"[{ErrorCodes.LLAMAINDEX_CONFIG_INVALID}] "
                f"llamaindex_config must be a Mapping, got {type(llamaindex_config).__name__}",
            )

        # Strict config validation: reject unknown keys to prevent silent misconfiguration.
        allowed_keys = {"enable_operation_context_propagation", "strict_text_types", "max_node_ids_in_context"}
        unknown = set(dict(llamaindex_config).keys()) - allowed_keys
        if unknown:
            raise ValueError(
                f"[{ErrorCodes.LLAMAINDEX_CONFIG_INVALID}] "
                f"llamaindex_config contains unknown keys: {sorted(unknown)}"
            )

        # Normalize + default-fill config for predictable behavior
        normalized_config: LlamaIndexAdapterConfig = dict(llamaindex_config)  # type: ignore[assignment]
        normalized_config.setdefault("enable_operation_context_propagation", True)
        normalized_config.setdefault("strict_text_types", True)
        normalized_config.setdefault("max_node_ids_in_context", 100)

        # Coerce types defensively (matches tests: bool coercion + int coercion)
        normalized_config["enable_operation_context_propagation"] = bool(
            normalized_config["enable_operation_context_propagation"]
        )
        normalized_config["strict_text_types"] = bool(normalized_config["strict_text_types"])
        normalized_config["max_node_ids_in_context"] = int(normalized_config["max_node_ids_in_context"])

        # Store core config + knobs
        self.corpus_adapter = corpus_adapter
        self._model_name = model_name
        self.batch_config = batch_config
        self.text_normalization_config = text_normalization_config
        self._embed_batch_size = embed_batch_size
        self._embedding_dimension_override = embedding_dimension
        self.llamaindex_config: LlamaIndexAdapterConfig = normalized_config

        # Thread-safety for translator lazy init (multi-threaded ingestion / concurrent calls)
        self._translator_lock = threading.Lock()

        # Fast-access fields for extractors and batching
        self._max_node_ids_in_context: int = normalized_config["max_node_ids_in_context"]
        self._embedding_dim_hint: Optional[int] = None

        # Enforce known embedding dimension to avoid incorrect fallbacks
        if (not hasattr(self.corpus_adapter, "get_embedding_dimension")) and (self._embedding_dimension_override is None):
            raise ValueError(
                "Embedding dimension is unknown. Either implement "
                "`get_embedding_dimension()` on the corpus_adapter or pass "
                "`embedding_dimension=...` to CorpusLlamaIndexEmbeddings.",
            )

        # Initialize BaseEmbedding with LlamaIndex expected parameters
        super().__init__(
            model_name=self._model_name,
            embed_batch_size=self._embed_batch_size,
            callback_manager=callback_manager,
            **kwargs,
        )

        logger.info(
            "CorpusLlamaIndexEmbeddings initialized with model_name=%s, embed_batch_size=%d, "
            "framework_version=%s, embedding_dimension_override=%s, llamaindex_config=%s",
            self._model_name,
            self._embed_batch_size,
            _FRAMEWORK_VERSION or "unknown",
            self._embedding_dimension_override,
            self.llamaindex_config,
        )

    # ------------------------------------------------------------------ #
    # LlamaIndex-facing identity
    # ------------------------------------------------------------------ #

    @property
    def model_name(self) -> str:
        """Return model name for LlamaIndex settings integration."""
        return self._model_name

    # ------------------------------------------------------------------ #
    # Translator: lazy + thread-safe init
    # ------------------------------------------------------------------ #

    @cached_property
    def _translator(self) -> EmbeddingTranslator:
        """
        Lazily construct and cache the `EmbeddingTranslator`.

        Thread-safe lazy initialization via a lock + double-check pattern to avoid
        duplicate initialization under concurrent access.
        """
        with self._translator_lock:
            existing = self.__dict__.get("_translator")
            if isinstance(existing, EmbeddingTranslator):
                return existing

            translator = create_embedding_translator(
                adapter=self.corpus_adapter,
                framework=_FRAMEWORK_NAME,
                translator=None,  # use registry/default generic translator
                batch_config=self.batch_config,
                text_normalization_config=self.text_normalization_config,
            )
            logger.debug(
                "EmbeddingTranslator initialized for LlamaIndex with model_name=%s",
                self.model_name,
            )
            return translator

    # ------------------------------------------------------------------ #
    # Context building: LlamaIndexContext -> OperationContext + framework_ctx
    # ------------------------------------------------------------------ #

    def _build_core_context(
        self,
        *,
        llamaindex_context: Optional[LlamaIndexContext] = None,
    ) -> Optional[OperationContext]:
        """
        Build a core OperationContext from LlamaIndex context with robust error handling.

        This method focuses purely on translating framework-specific context into the
        core OperationContext structure used by the embedding layer.
        """
        core_ctx: Optional[OperationContext] = None

        if llamaindex_context is None:
            return None

        if not isinstance(llamaindex_context, Mapping):
            logger.warning(
                "[%s] llamaindex_context should be a Mapping, got %s; ignoring context",
                ErrorCodes.LLAMAINDEX_CONTEXT_INVALID,
                type(llamaindex_context).__name__,
            )
            return None

        self._validate_llamaindex_context_structure(llamaindex_context)

        try:
            # Some SDKs accept framework_version, older ones may not.
            try:
                core_ctx_candidate = context_from_llamaindex(
                    llamaindex_context,
                    framework_version=_FRAMEWORK_VERSION,
                )
            except TypeError:
                core_ctx_candidate = context_from_llamaindex(llamaindex_context)

            # IMPORTANT:
            # - OperationContext is frozen in the base SDK; do NOT rebuild or mutate it here.
            # - Framework metadata belongs in framework_ctx (already handled separately).
            if _looks_like_operation_context(core_ctx_candidate):
                core_ctx = core_ctx_candidate  # type: ignore[assignment]
                logger.debug(
                    "Successfully created OperationContext from LlamaIndex context with index_id=%s (framework_version=%s)",
                    llamaindex_context.get("index_id", "unknown"),
                    _FRAMEWORK_VERSION or "unknown",
                )
            else:
                logger.warning(
                    "context_from_llamaindex returned non-OperationContext type: %s. Proceeding without OperationContext.",
                    type(core_ctx_candidate).__name__,
                )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Failed to create OperationContext from LlamaIndex context: %s. Proceeding with degraded context.",
                e,
            )
            # Never allow context attachment to break embeddings.
            try:
                attach_context(
                    e,
                    framework=_FRAMEWORK_NAME,
                    operation="context_build",
                    context_snapshot=_safe_snapshot(llamaindex_context),
                    framework_version=_FRAMEWORK_VERSION,
                    error_codes=EMBEDDING_COERCION_ERROR_CODES,
                    llamaindex_config=_safe_snapshot(self.llamaindex_config),
                )
            except Exception:
                pass

        return core_ctx

    def _build_framework_context(
        self,
        *,
        core_ctx: Optional[OperationContext],
        llamaindex_context: Optional[LlamaIndexContext] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Build framework-specific context for the LlamaIndex execution environment.

        Includes:
        - Framework identity and version
        - Model information
        - Adapter configuration
        - Best-effort embedding dimension hint
        - LlamaIndex routing fields (node_ids, index_id, trace_id, workflow)
        - Callback manager presence flag
        - OperationContext propagation when enabled
        """
        framework_ctx: Dict[str, Any] = {
            "framework": _FRAMEWORK_NAME,
            "model": self.model_name,
            "model_name": self.model_name,
            "error_codes": EMBEDDING_COERCION_ERROR_CODES,
            "llamaindex_config": dict(self.llamaindex_config),
        }
        if _FRAMEWORK_VERSION is not None:
            framework_ctx["framework_version"] = _FRAMEWORK_VERSION

        dim_hint = self._embedding_dim_hint or self.embedding_dimension
        framework_ctx["embedding_dim_hint"] = dim_hint

        if llamaindex_context:
            node_ids = llamaindex_context.get("node_ids")
            if node_ids is not None:
                bounded = (node_ids or [])[: self._max_node_ids_in_context]
                framework_ctx["node_ids"] = bounded
                framework_ctx["node_count"] = len(node_ids or [])
                if len(node_ids or []) > len(bounded):
                    framework_ctx["node_ids_truncated"] = True

            for key in ("index_id", "trace_id", "workflow"):
                if key in llamaindex_context:
                    framework_ctx[key] = llamaindex_context.get(key)

            if "callback_manager" in llamaindex_context:
                framework_ctx["has_callback_manager"] = bool(
                    llamaindex_context.get("callback_manager"),
                )

        # Include any extra call-specific hints while preserving structure.
        framework_ctx.update({k: v for k, v in kwargs.items() if not k.startswith("_")})

        # Stash OperationContext for downstream inspection (configurable).
        if core_ctx is not None and self.llamaindex_config.get(
            "enable_operation_context_propagation",
            True,
        ):
            framework_ctx["_operation_context"] = core_ctx

        return framework_ctx

    def _build_contexts(
        self,
        *,
        llamaindex_context: Optional[LlamaIndexContext] = None,
        **kwargs: Any,
    ) -> Tuple[Optional[OperationContext], Dict[str, Any]]:
        """
        Build contexts for LlamaIndex execution environment with comprehensive validation.

        Returns
        -------
        Tuple of:
        - core_ctx: core OperationContext or None if no/invalid context
        - framework_ctx: LlamaIndex-specific context for translator
        """
        core_ctx = self._build_core_context(llamaindex_context=llamaindex_context)
        framework_ctx = self._build_framework_context(
            core_ctx=core_ctx,
            llamaindex_context=llamaindex_context,
            **kwargs,
        )
        return core_ctx, framework_ctx

    def _validate_llamaindex_context_structure(self, context: Mapping[str, Any]) -> None:
        """Validate LlamaIndex context structure and log warnings for anomalies."""
        if not any(
            key in context
            for key in ("node_ids", "index_id", "callback_manager", "trace_id", "workflow")
        ):
            logger.debug(
                "LlamaIndex context missing common fields (node_ids, index_id, etc.) - reduced context for embeddings",
            )

    # ------------------------------------------------------------------ #
    # Coercion helpers (delegate to shared framework_utils)
    # ------------------------------------------------------------------ #

    def _coerce_embedding_matrix(self, result: Any) -> List[List[float]]:
        """
        Coerce translator result into a List[List[float]] embedding matrix.

        Delegates to the shared framework_utils implementation so behavior is consistent
        across all framework adapters.
        """
        return coerce_embedding_matrix(
            result=result,
            framework=_FRAMEWORK_NAME,
            error_codes=EMBEDDING_COERCION_ERROR_CODES,
            logger=logger,
        )

    def _coerce_embedding_vector(self, result: Any) -> List[float]:
        """
        Coerce translator result for a single-text embed into List[float].

        Delegates to the shared framework_utils implementation and preserves semantics
        (first row when multiple are returned).
        """
        return coerce_embedding_vector(
            result=result,
            framework=_FRAMEWORK_NAME,
            error_codes=EMBEDDING_COERCION_ERROR_CODES,
            logger=logger,
        )

    # ------------------------------------------------------------------ #
    # Empty text + batch warnings + dim hint updates
    # ------------------------------------------------------------------ #

    @property
    def embedding_dimension(self) -> int:
        """
        Get embedding dimension for proper zero vector fallback.

        Never guesses; requires either:
        - adapter.get_embedding_dimension(), or
        - embedding_dimension override passed at init.
        """
        if hasattr(self.corpus_adapter, "get_embedding_dimension"):
            try:
                return int(self.corpus_adapter.get_embedding_dimension())
            except Exception as e:  # noqa: BLE001
                logger.debug("Failed to get embedding dimension from adapter: %s", e)
                if self._embedding_dimension_override is not None:
                    return int(self._embedding_dimension_override)
                raise

        if self._embedding_dimension_override is not None:
            return int(self._embedding_dimension_override)

        # Should be unreachable due to __init__ check
        raise RuntimeError(
            "Embedding dimension is unknown. Adapter does not expose `get_embedding_dimension()` "
            "and no `embedding_dimension` override was provided.",
        )

    def _handle_empty_text(self, text: str) -> List[float]:
        """Handle empty/whitespace-only text by returning an all-zero vector."""
        dim = self.embedding_dimension
        logger.warning(
            "Empty text provided for embedding, returning zero vector (dimension=%d)",
            dim,
        )
        return [0.0] * dim

    def _warn_if_extreme_batch(self, texts: Sequence[Any], *, op_name: str) -> None:
        """
        Emit a soft warning if an extremely large batch is requested without an explicit
        BatchConfig.max_batch_size. This is informational only.
        """
        # warn_if_extreme_batch expects strings; filter defensively so strict_text_types=False
        # cannot break warnings (warnings should never be a failure mode).
        safe_texts: List[str] = [t for t in texts if isinstance(t, str)]
        warn_if_extreme_batch(
            framework=_FRAMEWORK_NAME,
            texts=safe_texts,
            op_name=op_name,
            batch_config=self.batch_config,
            logger=logger,
        )

    def _update_dim_hint(self, dim: Optional[int]) -> None:
        """
        Thread-safe, best-effort dimension hint update.

        Uses first-write-wins semantics under the existing translator lock to
        avoid races in concurrent first-embed scenarios.
        """
        if dim is None:
            return
        if self._embedding_dim_hint is not None:
            return

        with self._translator_lock:
            if self._embedding_dim_hint is None:
                self._embedding_dim_hint = dim

    # ------------------------------------------------------------------ #
    # Shared embedding implementations to avoid duplication
    # ------------------------------------------------------------------ #

    def _embed_single_text(self, text: str, llamaindex_context: LlamaIndexContext) -> List[float]:
        """Unified single text embedding implementation."""
        core_ctx, framework_ctx = self._build_contexts(llamaindex_context=llamaindex_context)

        start = time.perf_counter()
        translated = self._translator.embed(
            raw_texts=text,
            op_ctx=core_ctx,
            framework_ctx=framework_ctx,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        vec = self._coerce_embedding_vector(translated)
        self._update_dim_hint(len(vec))

        logger.debug(
            "LlamaIndex embed_single_text completed: dim=%d latency_ms=%.2f",
            len(vec),
            elapsed_ms,
        )
        return vec

    async def _aembed_single_text(self, text: str, llamaindex_context: LlamaIndexContext) -> List[float]:
        """Unified async single text embedding implementation."""
        core_ctx, framework_ctx = self._build_contexts(llamaindex_context=llamaindex_context)

        start = time.perf_counter()
        translated = await self._translator.arun_embed(
            raw_texts=text,
            op_ctx=core_ctx,
            framework_ctx=framework_ctx,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        vec = self._coerce_embedding_vector(translated)
        self._update_dim_hint(len(vec))

        logger.debug(
            "LlamaIndex aembed_single_text completed: dim=%d latency_ms=%.2f",
            len(vec),
            elapsed_ms,
        )
        return vec

    def _embed_text_batch(self, texts: Sequence[Any], llamaindex_context: LlamaIndexContext) -> List[List[float]]:
        """
        Unified batch text embedding implementation (preserves row alignment).

        If strict_text_types=False, non-string items are treated as empty and receive
        all-zero vectors to preserve output row alignment with the input order.
        """
        self._warn_if_extreme_batch(texts, op_name="_get_text_embeddings")

        texts_list = list(texts)
        if self.llamaindex_config.get("strict_text_types", True):
            _validate_texts_are_strings(texts_list, op_name="_get_text_embeddings")

        non_empty_texts: List[str] = [t for t in texts_list if isinstance(t, str) and t.strip()]
        empty_indices_list = [i for i, t in enumerate(texts_list) if not isinstance(t, str) or not t.strip()]
        empty_indices = set(empty_indices_list)

        # If everything is empty/non-string, return all-zero rows.
        if not non_empty_texts:
            dim = self.embedding_dimension
            return [[0.0] * dim for _ in texts_list]

        core_ctx, framework_ctx = self._build_contexts(llamaindex_context=llamaindex_context)

        start = time.perf_counter()
        translated = self._translator.embed(
            raw_texts=non_empty_texts,
            op_ctx=core_ctx,
            framework_ctx=framework_ctx,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        embeddings = self._coerce_embedding_matrix(translated)
        dim = _infer_dim_from_matrix(embeddings)
        self._update_dim_hint(dim)

        # Re-insert empty rows to preserve input alignment.
        if empty_indices:
            inferred_dim = dim if dim is not None else self.embedding_dimension
            result_embeddings: List[List[float]] = []
            non_empty_idx = 0
            for i in range(len(texts_list)):
                if i in empty_indices:
                    result_embeddings.append([0.0] * inferred_dim)
                else:
                    result_embeddings.append(embeddings[non_empty_idx])
                    non_empty_idx += 1

            logger.debug(
                "LlamaIndex embed_text_batch completed: total=%d non_empty=%d dim=%s latency_ms=%.2f",
                len(texts_list),
                len(non_empty_texts),
                inferred_dim,
                elapsed_ms,
            )
            return result_embeddings

        logger.debug(
            "LlamaIndex embed_text_batch completed: total=%d dim=%s latency_ms=%.2f",
            len(texts_list),
            dim,
            elapsed_ms,
        )
        return embeddings

    async def _aembed_text_batch(self, texts: Sequence[Any], llamaindex_context: LlamaIndexContext) -> List[List[float]]:
        """
        Unified async batch text embedding implementation (preserves row alignment).

        If strict_text_types=False, non-string items are treated as empty and receive
        all-zero vectors to preserve output row alignment with the input order.
        """
        self._warn_if_extreme_batch(texts, op_name="_aget_text_embeddings")

        texts_list = list(texts)
        if self.llamaindex_config.get("strict_text_types", True):
            _validate_texts_are_strings(texts_list, op_name="_aget_text_embeddings")

        non_empty_texts: List[str] = [t for t in texts_list if isinstance(t, str) and t.strip()]
        empty_indices_list = [i for i, t in enumerate(texts_list) if not isinstance(t, str) or not t.strip()]
        empty_indices = set(empty_indices_list)

        if not non_empty_texts:
            dim = self.embedding_dimension
            return [[0.0] * dim for _ in texts_list]

        core_ctx, framework_ctx = self._build_contexts(llamaindex_context=llamaindex_context)

        start = time.perf_counter()
        translated = await self._translator.arun_embed(
            raw_texts=non_empty_texts,
            op_ctx=core_ctx,
            framework_ctx=framework_ctx,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        embeddings = self._coerce_embedding_matrix(translated)
        dim = _infer_dim_from_matrix(embeddings)
        self._update_dim_hint(dim)

        if empty_indices:
            inferred_dim = dim if dim is not None else self.embedding_dimension
            result_embeddings: List[List[float]] = []
            non_empty_idx = 0
            for i in range(len(texts_list)):
                if i in empty_indices:
                    result_embeddings.append([0.0] * inferred_dim)
                else:
                    result_embeddings.append(embeddings[non_empty_idx])
                    non_empty_idx += 1

            logger.debug(
                "LlamaIndex aembed_text_batch completed: total=%d non_empty=%d dim=%s latency_ms=%.2f",
                len(texts_list),
                len(non_empty_texts),
                inferred_dim,
                elapsed_ms,
            )
            return result_embeddings

        logger.debug(
            "LlamaIndex aembed_text_batch completed: total=%d dim=%s latency_ms=%.2f",
            len(texts_list),
            dim,
            elapsed_ms,
        )
        return embeddings

    # ------------------------------------------------------------------ #
    # Capabilities / health passthrough (via EmbeddingTranslator)
    # ------------------------------------------------------------------ #

    @with_embedding_error_context("capabilities")
    def capabilities(self) -> Mapping[str, Any]:
        """
        Sync capabilities passthrough.

        Delegates to EmbeddingTranslator.capabilities(), which centralizes
        async/sync adapter methods and error context behavior across frameworks.
        """
        _ensure_not_in_event_loop("capabilities")
        return self._translator.capabilities()

    @with_async_embedding_error_context("capabilities_async")
    async def acapabilities(self) -> Mapping[str, Any]:
        """
        Async capabilities passthrough.

        Delegates to EmbeddingTranslator.arun_capabilities().
        """
        return await self._translator.arun_capabilities()

    @with_embedding_error_context("health")
    def health(self) -> Mapping[str, Any]:
        """
        Sync health passthrough.

        Delegates to EmbeddingTranslator.health().
        """
        _ensure_not_in_event_loop("health")
        return self._translator.health()

    @with_async_embedding_error_context("health_async")
    async def ahealth(self) -> Mapping[str, Any]:
        """
        Async health passthrough.

        Delegates to EmbeddingTranslator.arun_health().
        """
        return await self._translator.arun_health()

    # ------------------------------------------------------------------ #
    # Resource management (context managers)
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """
        Close underlying resources if they expose a close() method.

        This includes:
        - The underlying corpus_adapter
        - The EmbeddingTranslator if it was constructed and exposes close()
        """
        # Deadlock prevention: never run sync close inside a running event loop.
        _ensure_not_in_event_loop("close")

        # Close translator if already initialized (avoid forcing construction).
        translator = self.__dict__.get("_translator")
        if isinstance(translator, EmbeddingTranslator):
            try:
                _maybe_close_sync(translator)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Error while closing embedding translator in close(): %s",
                    e,
                )

        try:
            _maybe_close_sync(self.corpus_adapter)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Error while closing embedding adapter in close(): %s",
                e,
            )

    async def aclose(self) -> None:
        """
        Async-close underlying resources if they expose async closers.

        Prefers:
        - translator.aclose() / translator.close()
        - corpus_adapter.aclose() / corpus_adapter.close()
        """
        translator = self.__dict__.get("_translator")
        if isinstance(translator, EmbeddingTranslator):
            try:
                await _maybe_close_async(translator)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Error while closing embedding translator in aclose(): %s",
                    e,
                )

        try:
            await _maybe_close_async(self.corpus_adapter)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Error while closing embedding adapter in aclose(): %s",
                e,
            )

    def __enter__(self) -> "CorpusLlamaIndexEmbeddings":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    async def __aenter__(self) -> "CorpusLlamaIndexEmbeddings":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.aclose()

    # ------------------------------------------------------------------ #
    # LlamaIndex BaseEmbedding abstract methods (sync + async)
    # ------------------------------------------------------------------ #

    @with_embedding_error_context("query")
    def _get_query_embedding(self, query: str, **kwargs: Any) -> List[float]:
        """
        Sync query embedding implementation for LlamaIndex.

        Notes on validation:
        - We raise TypeError for non-string inputs with an actionable message.
        - Empty/whitespace strings return a zero vector of known dimension.
        """
        _ensure_not_in_event_loop("_get_query_embedding")

        if not isinstance(query, str):
            raise TypeError(f"embedding_query expects str; got {type(query).__name__}")
        if not query.strip():
            return self._handle_empty_text(query)

        context = _filter_llamaindex_context_from_kwargs(kwargs)
        return self._embed_single_text(query, context)

    @with_async_embedding_error_context("query")
    async def _aget_query_embedding(self, query: str, **kwargs: Any) -> List[float]:
        """
        Async query embedding implementation for LlamaIndex.

        Same validation semantics as sync.
        """
        if not isinstance(query, str):
            raise TypeError(f"embedding_query expects str; got {type(query).__name__}")
        if not query.strip():
            return self._handle_empty_text(query)

        context = _filter_llamaindex_context_from_kwargs(kwargs)
        return await self._aembed_single_text(query, context)

    @with_embedding_error_context("text")
    def _get_text_embedding(self, text: str, **kwargs: Any) -> List[float]:
        """
        Sync text embedding implementation for LlamaIndex nodes.

        Same validation semantics as query embedding.
        """
        _ensure_not_in_event_loop("_get_text_embedding")

        if not isinstance(text, str):
            raise TypeError(f"embedding_text expects str; got {type(text).__name__}")
        if not text.strip():
            return self._handle_empty_text(text)

        context = _filter_llamaindex_context_from_kwargs(kwargs)
        return self._embed_single_text(text, context)

    @with_async_embedding_error_context("text")
    async def _aget_text_embedding(self, text: str, **kwargs: Any) -> List[float]:
        """
        Async text embedding implementation for LlamaIndex nodes.

        Same validation semantics as sync.
        """
        if not isinstance(text, str):
            raise TypeError(f"embedding_text expects str; got {type(text).__name__}")
        if not text.strip():
            return self._handle_empty_text(text)

        context = _filter_llamaindex_context_from_kwargs(kwargs)
        return await self._aembed_single_text(text, context)

    @with_embedding_error_context("texts")
    def _get_text_embeddings(self, texts: Sequence[Any], **kwargs: Any) -> List[List[float]]:
        """
        Batch text embedding implementation for LlamaIndex nodes.

        Notes:
        - When strict_text_types=True (default), non-strings raise TypeError.
        - When strict_text_types=False, non-strings are treated as empty and get zero vectors.
        - Output row alignment always matches input order/length.
        """
        _ensure_not_in_event_loop("_get_text_embeddings")

        context = _filter_llamaindex_context_from_kwargs(kwargs)
        return self._embed_text_batch(texts, context)

    @with_async_embedding_error_context("texts")
    async def _aget_text_embeddings(self, texts: Sequence[Any], **kwargs: Any) -> List[List[float]]:
        """
        Async batch text embedding implementation for LlamaIndex nodes.

        Same semantics as sync batch embedding.
        """
        context = _filter_llamaindex_context_from_kwargs(kwargs)
        return await self._aembed_text_batch(texts, context)


# ---------------------------------------------------------------------------
# LlamaIndex Settings / Service Registration Helpers
# ---------------------------------------------------------------------------


def configure_llamaindex_embeddings(
    corpus_adapter: EmbeddingProtocolV1,
    model_name: str = "corpus-embedding-protocol",
    llamaindex_config: Optional[LlamaIndexAdapterConfig] = None,
    **kwargs: Any,
) -> CorpusLlamaIndexEmbeddings:
    """
    Configure and optionally register Corpus embeddings with LlamaIndex.

    Mirrors patterns across framework adapters:
    - Always constructs and returns a `CorpusLlamaIndexEmbeddings` instance.
    - When LlamaIndex is installed and `llama_index.core.Settings` is available,
      it attempts to register embeddings as the global `Settings.embed_model`.
      If that fails (version differences, custom configs, etc.), it logs a warning
      and still returns the embeddings instance.

    Note:
    - `embedding_dimension=...` can be provided via **kwargs when the adapter does not
      implement get_embedding_dimension().
    """
    embeddings = CorpusLlamaIndexEmbeddings(
        corpus_adapter=corpus_adapter,
        model_name=model_name,
        llamaindex_config=llamaindex_config,
        **kwargs,
    )

    if not LLAMAINDEX_AVAILABLE:
        logger.debug(
            "LlamaIndex is not installed; returning embeddings without global Settings registration.",
        )
        return embeddings

    # Best-effort registration with LlamaIndex global Settings (not required to use embeddings)
    try:
        from llama_index.core import Settings  # type: ignore

        try:
            Settings.embed_model = embeddings
            logger.info(
                "Corpus LlamaIndex embeddings configured and registered as Settings.embed_model: %s",
                model_name,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Failed to configure LlamaIndex Settings.embed_model: %s. You may need to register the embeddings manually.",
                e,
            )
    except ImportError:  # pragma: no cover
        logger.debug(
            "LlamaIndex Settings API not available; unable to auto-register embed_model. Returning embeddings instance only.",
        )

    return embeddings


def register_with_llamaindex(
    corpus_adapter: EmbeddingProtocolV1,
    model_name: str = "corpus-embedding-protocol",
    llamaindex_config: Optional[LlamaIndexAdapterConfig] = None,
    **kwargs: Any,
) -> CorpusLlamaIndexEmbeddings:
    """
    Alias for `configure_llamaindex_embeddings`.

    This mirrors the `register_with_*` naming convention across framework adapters.
    """
    return configure_llamaindex_embeddings(
        corpus_adapter=corpus_adapter,
        model_name=model_name,
        llamaindex_config=llamaindex_config,
        **kwargs,
    )


__all__ = [
    "CorpusLlamaIndexEmbeddings",
    "LlamaIndexContext",
    "LlamaIndexAdapterConfig",
    "configure_llamaindex_embeddings",
    "register_with_llamaindex",
    "ErrorCodes",
    "with_embedding_error_context",
    "with_async_embedding_error_context",
    "LLAMAINDEX_AVAILABLE",
]
