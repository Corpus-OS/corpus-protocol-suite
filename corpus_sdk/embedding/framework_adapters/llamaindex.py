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

The design leverages LlamaIndex's focus on efficient indexing and retrieval
while maintaining the protocol-first Corpus embedding stack.

Resilience (retries, caching, rate limiting, etc.) is expected to be provided
by the underlying adapter, typically a BaseEmbeddingAdapter subclass.
"""

from __future__ import annotations

import logging
import threading
import time
from functools import cached_property, wraps
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Callable,
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
    EmbeddingTranslator,
    BatchConfig,
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

try:  # Best-effort LlamaIndex version detection
    import llama_index  # type: ignore

    _FRAMEWORK_VERSION: Optional[str] = getattr(llama_index, "__version__", None)
except Exception:  # noqa: BLE001
    _FRAMEWORK_VERSION = None

# ---------------------------------------------------------------------------
# Safe conditional imports for LlamaIndex
# ---------------------------------------------------------------------------

try:
    from llama_index.core.embeddings import BaseEmbedding, DEFAULT_EMBED_BATCH_SIZE
    from llama_index.core.callbacks import CallbackManager

    LLAMAINDEX_AVAILABLE = True
except ImportError:  # pragma: no cover - only used when LlamaIndex isn't installed

    class BaseEmbedding:  # type: ignore[no-redef]
        """
        Minimal fallback BaseEmbedding when LlamaIndex is not installed.

        This is only here so importing this module doesn't explode in environments
        without LlamaIndex. Any attempt to actually *use* this class without
        LlamaIndex should be considered a misconfiguration.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            # Deliberately do nothing; real usage will fail earlier in practice.
            pass

    class CallbackManager:  # type: ignore[no-redef]
        """Fallback CallbackManager stub when LlamaIndex is not installed."""
        pass

    # Reasonable default used when the real constant is not available
    DEFAULT_EMBED_BATCH_SIZE = 512
    LLAMAINDEX_AVAILABLE = False


# ---------------------------------------------------------------------------
# Error codes (aligned with other embedding adapters)
# ---------------------------------------------------------------------------


class ErrorCodes:
    """
    Error code constants for the LlamaIndex embedding adapter.

    This is a simple namespace for framework-specific codes. The shared
    coercion helpers use `EMBEDDING_COERCION_ERROR_CODES`, which is a
    `CoercionErrorCodes` instance derived from these values.
    """

    # Coercion-level (used by common framework_utils)
    INVALID_EMBEDDING_RESULT = "INVALID_EMBEDDING_RESULT"
    EMPTY_EMBEDDING_RESULT = "EMPTY_EMBEDDING_RESULT"
    EMBEDDING_CONVERSION_ERROR = "EMBEDDING_CONVERSION_ERROR"

    # LlamaIndex-specific context errors
    LLAMAINDEX_CONTEXT_INVALID = "LLAMAINDEX_CONTEXT_INVALID"
    LLAMAINDEX_CONFIG_INVALID = "LLAMAINDEX_CONFIG_INVALID"


# Coercion configuration for the common embedding utils
EMBEDDING_COERCION_ERROR_CODES: CoercionErrorCodes = CoercionErrorCodes(
    invalid_result=ErrorCodes.INVALID_EMBEDDING_RESULT,
    empty_result=ErrorCodes.EMPTY_EMBEDDING_RESULT,
    conversion_error=ErrorCodes.EMBEDDING_CONVERSION_ERROR,
    framework_label=_FRAMEWORK_NAME,
)


class LlamaIndexContext(TypedDict, total=False):
    """Structured type for LlamaIndex execution context."""
    node_ids: Optional[List[str]]
    index_id: Optional[str]
    callback_manager: Optional[Any]
    trace_id: Optional[str]
    workflow: Optional[str]


class LlamaIndexAdapterConfig(TypedDict, total=False):
    """
    Adapter-level configuration for the LlamaIndex embedding adapter.

    This is *not* the per-call LlamaIndex BaseEmbedding kwargs; it configures
    adapter behavior for context propagation and strictness.

    Fields
    ------
    enable_operation_context_propagation:
        If True, include the OperationContext instance in framework_ctx as
        `_operation_context` for downstream inspection. Defaults to True.

    strict_text_types:
        If True, reject non-string items in batch embedding calls rather than
        silently treating them as empty. Defaults to True for hardening parity
        with the other framework adapters.

    max_node_ids_in_context:
        Maximum number of node_ids to include in framework_ctx and error context
        to prevent log bloat. Defaults to 100.
    """

    enable_operation_context_propagation: bool
    strict_text_types: bool
    max_node_ids_in_context: int


# ---------------------------------------------------------------------------
# Safety / robustness utilities (input validation + safe snapshots)
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
        if isinstance(value, (list, tuple)):
            out_list: List[Any] = []
            for idx, v in enumerate(value):
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
    OperationContext may be a concrete type or a Protocol/alias depending on the SDK.

    Prefer isinstance when it works; fall back to a lightweight structural
    heuristic to avoid false negatives.
    """
    if obj is None:
        return False
    try:
        if isinstance(obj, OperationContext):
            return True
    except TypeError:
        # OperationContext may be a Protocol/typing alias at runtime
        pass

    return any(
        hasattr(obj, attr)
        for attr in (
            "trace_id",
            "request_id",
            "user_id",
            "tags",
            "metadata",
            "to_dict",
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

    LlamaIndex passes a variety of kwargs depending on execution path; we keep
    this adapter resilient by filtering to known keys for framework_ctx and for
    context translation.
    """
    ctx: LlamaIndexContext = {}
    for key in ("node_ids", "index_id", "callback_manager", "trace_id", "workflow"):
        if key in kwargs:
            ctx[key] = kwargs[key]  # type: ignore[literal-required]
    return ctx


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

    # Text-based metrics
    if operation in ("query", "text") and args and isinstance(args[0], str):
        dynamic_ctx["text_len"] = len(args[0])
    elif operation == "texts" and args and isinstance(args[0], Sequence):
        texts_seq = args[0]
        dynamic_ctx["texts_count"] = len(texts_seq)
        empty_count = sum(
            1 for text in texts_seq if not isinstance(text, str) or not text.strip()
        )
        if empty_count:
            dynamic_ctx["empty_texts_count"] = empty_count

    ctx = _filter_llamaindex_context_from_kwargs(kwargs)
    if ctx.get("node_ids") is not None:
        node_ids = ctx.get("node_ids") or []
        max_nodes = getattr(instance, "_max_node_ids_in_context", 100)
        bounded = node_ids[:max_nodes]
        dynamic_ctx["node_ids"] = bounded
        dynamic_ctx["node_count"] = len(node_ids)
        if len(node_ids) > len(bounded):
            dynamic_ctx["node_ids_truncated"] = True

    # Loop extraction for parity with other adapters
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

    This provides the same rich observability as more complex context
    managers while maintaining decorator consistency with other adapters.
    """

    def decorator_factory(
        **static_context: Any,
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            if is_async:

                @wraps(func)
                async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
                    dynamic_context = _extract_dynamic_context(
                        self,
                        args,
                        kwargs,
                        operation,
                    )
                    # Explicit framework_version inclusion for consistency + future-proofing.
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
                dynamic_context = _extract_dynamic_context(
                    self,
                    args,
                    kwargs,
                    operation,
                )
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
def with_embedding_error_context(
    operation: str,
    **static_context: Any,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for sync methods with rich dynamic context extraction."""
    static_context.setdefault("error_codes", EMBEDDING_COERCION_ERROR_CODES)
    static_context.setdefault("framework_version", _FRAMEWORK_VERSION)
    return _create_error_context_decorator(operation, is_async=False)(**static_context)


def with_async_embedding_error_context(
    operation: str,
    **static_context: Any,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for async methods with rich dynamic context extraction."""
    static_context.setdefault("error_codes", EMBEDDING_COERCION_ERROR_CODES)
    static_context.setdefault("framework_version", _FRAMEWORK_VERSION)
    return _create_error_context_decorator(operation, is_async=True)(**static_context)


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
    ):
        """
        Initialize Corpus LlamaIndex Embeddings.

        Parameters
        ----------
        llamaindex_config:
            Adapter-level configuration (separate from per-call kwargs). Mirrors the
            `*_config` pattern used across the framework adapters.
        embedding_dimension:
            Optional explicit embedding dimension override. Required if the adapter
            does not implement `get_embedding_dimension()`.
        """
        # Behavioral validation (duck-typed) instead of strict isinstance
        if not hasattr(corpus_adapter, "embed") or not callable(
            getattr(corpus_adapter, "embed", None)
        ):
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
        normalized_config: LlamaIndexAdapterConfig = dict(llamaindex_config)  # type: ignore[assignment]
        normalized_config.setdefault("enable_operation_context_propagation", True)
        normalized_config.setdefault("strict_text_types", True)
        normalized_config.setdefault("max_node_ids_in_context", 100)

        normalized_config["enable_operation_context_propagation"] = bool(
            normalized_config["enable_operation_context_propagation"]
        )
        normalized_config["strict_text_types"] = bool(normalized_config["strict_text_types"])
        normalized_config["max_node_ids_in_context"] = int(normalized_config["max_node_ids_in_context"])

        self.corpus_adapter = corpus_adapter
        self._model_name = model_name
        self.batch_config = batch_config
        self.text_normalization_config = text_normalization_config
        self._embed_batch_size = embed_batch_size
        self._embedding_dimension_override = embedding_dimension
        self.llamaindex_config: LlamaIndexAdapterConfig = normalized_config

        # Thread-safety for translator lazy init (multi-threaded ingestion / concurrent calls).
        self._translator_lock = threading.Lock()

        # Keep these private for fast access in extractors
        self._max_node_ids_in_context: int = normalized_config["max_node_ids_in_context"]
        self._embedding_dim_hint: Optional[int] = None

        # Enforce known embedding dimension to avoid incorrect fallbacks
        if (
            not hasattr(self.corpus_adapter, "get_embedding_dimension")
            and self._embedding_dimension_override is None
        ):
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
            "CorpusLlamaIndexEmbeddings initialized with model_name=%s, "
            "embed_batch_size=%d, framework_version=%s, embedding_dimension=%s, "
            "llamaindex_config=%s",
            self._model_name,
            self._embed_batch_size,
            _FRAMEWORK_VERSION or "unknown",
            self._embedding_dimension_override,
            self.llamaindex_config,
        )

    # ------------------------------------------------------------------ #
    # Core LlamaIndex Property Implementation
    # ------------------------------------------------------------------ #

    @property
    def model_name(self) -> str:
        """Return model name for LlamaIndex settings integration."""
        return self._model_name

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
        core_ctx: Optional[OperationContext] = None
        framework_ctx: Dict[str, Any] = {
            "framework": _FRAMEWORK_NAME,
            # Canonical model + LlamaIndex-native model_name for observability parity
            "model": self.model_name,
            "model_name": self.model_name,
            # Include error_codes in framework_ctx for downstream consistency
            "error_codes": EMBEDDING_COERCION_ERROR_CODES,
            # Canonical adapter config key for parity with other adapters
            "llamaindex_config": dict(self.llamaindex_config),
        }
        if _FRAMEWORK_VERSION is not None:
            framework_ctx["framework_version"] = _FRAMEWORK_VERSION

        # Surface best-effort dim hint for parity with other adapters
        dim_hint = self._embedding_dim_hint or self.embedding_dimension
        framework_ctx["embedding_dim_hint"] = dim_hint

        # Validate input type for llamaindex_context
        if llamaindex_context is not None:
            if not isinstance(llamaindex_context, Mapping):
                logger.warning(
                    "[%s] llamaindex_context should be a Mapping, got %s; ignoring context",
                    ErrorCodes.LLAMAINDEX_CONTEXT_INVALID,
                    type(llamaindex_context).__name__,
                )
                llamaindex_context = None
            else:
                self._validate_llamaindex_context_structure(llamaindex_context)

        # Convert LlamaIndex context to core OperationContext with defensive handling
        if llamaindex_context is not None:
            try:
                # Best-effort propagation of framework version into translation (if supported)
                try:
                    core_ctx_candidate = context_from_llamaindex(
                        llamaindex_context,
                        framework_version=_FRAMEWORK_VERSION,
                    )
                except TypeError:
                    core_ctx_candidate = context_from_llamaindex(llamaindex_context)

                if _looks_like_operation_context(core_ctx_candidate):
                    core_ctx = core_ctx_candidate  # type: ignore[assignment]

                    # Enrich OperationContext attrs with framework metadata (best-effort).
                    attrs = dict(getattr(core_ctx, "attrs", {}) or {})
                    attrs.setdefault("framework", _FRAMEWORK_NAME)
                    if _FRAMEWORK_VERSION is not None:
                        attrs.setdefault("framework_version", _FRAMEWORK_VERSION)

                    # Preserve existing field values if OperationContext is immutable.
                    try:
                        core_ctx = OperationContext(
                            request_id=getattr(core_ctx, "request_id", None),
                            idempotency_key=getattr(core_ctx, "idempotency_key", None),
                            deadline_ms=getattr(core_ctx, "deadline_ms", None),
                            traceparent=getattr(core_ctx, "traceparent", None),
                            tenant=getattr(core_ctx, "tenant", None),
                            attrs=attrs,
                        )
                    except Exception:
                        # If OperationContext can't be reconstructed, keep the candidate.
                        try:
                            setattr(core_ctx, "attrs", attrs)  # type: ignore[attr-defined]
                        except Exception:
                            pass

                    logger.debug(
                        "Successfully created OperationContext from LlamaIndex context "
                        "with index_id=%s (framework_version=%s)",
                        llamaindex_context.get("index_id", "unknown"),
                        _FRAMEWORK_VERSION or "unknown",
                    )
                else:
                    logger.warning(
                        "context_from_llamaindex returned non-OperationContext type: %s. "
                        "Proceeding without OperationContext.",
                        type(core_ctx_candidate).__name__,
                    )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Failed to create OperationContext from LlamaIndex context: %s. "
                    "Proceeding with degraded context.",
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

        # Add LlamaIndex-specific context for nodes and retrieval (bounded)
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
                    llamaindex_context.get("callback_manager")
                )

        # Include any extra call-specific hints while preserving structure
        framework_ctx.update({k: v for k, v in kwargs.items() if not k.startswith("_")})

        # Stash OperationContext for downstream inspection (configurable)
        if core_ctx is not None and self.llamaindex_config.get(
            "enable_operation_context_propagation", True
        ):
            framework_ctx["_operation_context"] = core_ctx

        return core_ctx, framework_ctx

    def _validate_llamaindex_context_structure(
        self,
        context: Mapping[str, Any],
    ) -> None:
        """Validate LlamaIndex context structure and log warnings for anomalies."""
        if not any(
            key in context
            for key in (
                "node_ids",
                "index_id",
                "callback_manager",
                "trace_id",
                "workflow",
            )
        ):
            logger.debug(
                "LlamaIndex context missing common fields (node_ids, index_id, etc.) - "
                "reduced context for embeddings",
            )

    def _coerce_embedding_matrix(self, result: Any) -> List[List[float]]:
        """
        Coerce translator result into a List[List[float]] embedding matrix.

        Delegates to the shared framework_utils implementation so behavior
        is consistent across all framework adapters.
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

        Delegates to the shared framework_utils implementation and preserves
        the existing semantics (first row when multiple are returned).
        """
        return coerce_embedding_vector(
            result=result,
            framework=_FRAMEWORK_NAME,
            error_codes=EMBEDDING_COERCION_ERROR_CODES,
            logger=logger,
        )

    # ------------------------------------------------------------------ #
    # Helpers for empty text + batch warnings
    # ------------------------------------------------------------------ #

    @property
    def embedding_dimension(self) -> int:
        """
        Get embedding dimension for proper zero vector fallback.

        Returns
        -------
        int
            Embedding dimension. Never guesses; requires either:
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

        # Should be unreachable due to __init__ check, but keep a hard failure
        raise RuntimeError(
            "Embedding dimension is unknown. Adapter does not expose "
            "`get_embedding_dimension()` and no `embedding_dimension` "
            "override was provided.",
        )

    def _handle_empty_text(self, text: str) -> List[float]:
        """
        Handle empty text by returning appropriate zero vector.
        """
        dim = self.embedding_dimension
        logger.warning(
            "Empty text provided for embedding, returning zero vector (dimension=%d)",
            dim,
        )
        return [0.0] * dim

    def _warn_if_extreme_batch(self, texts: Sequence[str], *, op_name: str) -> None:
        """
        Emit a soft warning if an extremely large batch is requested
        without an explicit BatchConfig.max_batch_size.
        """
        warn_if_extreme_batch(
            framework=_FRAMEWORK_NAME,
            texts=texts,
            op_name=op_name,
            batch_config=self.batch_config,
            logger=logger,
        )

    def _embed_single_text(
        self,
        text: str,
        llamaindex_context: LlamaIndexContext,
    ) -> List[float]:
        """Unified single text embedding implementation to eliminate duplication."""
        core_ctx, framework_ctx = self._build_contexts(
            llamaindex_context=llamaindex_context,
        )

        logger.debug(
            "Embedding single text for LlamaIndex index: %s, node count: %d",
            llamaindex_context.get("index_id", "unknown"),
            len(llamaindex_context.get("node_ids", []) or []),
        )

        start = time.perf_counter()
        translated = self._translator.embed(
            raw_texts=text,
            op_ctx=core_ctx,
            framework_ctx=framework_ctx,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        vec = self._coerce_embedding_vector(translated)
        self._embedding_dim_hint = len(vec)

        logger.debug(
            "LlamaIndex embed_single_text completed: dim=%d latency_ms=%.2f",
            len(vec),
            elapsed_ms,
        )
        return vec

    async def _aembed_single_text(
        self,
        text: str,
        llamaindex_context: LlamaIndexContext,
    ) -> List[float]:
        """Unified async single text embedding implementation to eliminate duplication."""
        core_ctx, framework_ctx = self._build_contexts(
            llamaindex_context=llamaindex_context,
        )

        logger.debug(
            "Async embedding single text for LlamaIndex index: %s, node count: %d",
            llamaindex_context.get("index_id", "unknown"),
            len(llamaindex_context.get("node_ids", []) or []),
        )

        start = time.perf_counter()
        translated = await self._translator.arun_embed(
            raw_texts=text,
            op_ctx=core_ctx,
            framework_ctx=framework_ctx,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        vec = self._coerce_embedding_vector(translated)
        self._embedding_dim_hint = len(vec)

        logger.debug(
            "LlamaIndex aembed_single_text completed: dim=%d latency_ms=%.2f",
            len(vec),
            elapsed_ms,
        )
        return vec

    def _embed_text_batch(
        self,
        texts: Sequence[str],
        llamaindex_context: LlamaIndexContext,
    ) -> List[List[float]]:
        """Unified batch text embedding implementation to eliminate duplication."""
        self._warn_if_extreme_batch(texts, op_name="_get_text_embeddings")

        texts_list = list(texts)
        if self.llamaindex_config.get("strict_text_types", True):
            _validate_texts_are_strings(texts_list, op_name="_get_text_embeddings")

        non_empty_texts = [t for t in texts_list if isinstance(t, str) and t.strip()]
        empty_indices_list = [
            i for i, t in enumerate(texts_list) if not isinstance(t, str) or not t.strip()
        ]
        empty_indices = set(empty_indices_list)

        if not non_empty_texts:
            dim = self.embedding_dimension
            return [[0.0] * dim for _ in texts_list]

        core_ctx, framework_ctx = self._build_contexts(
            llamaindex_context=llamaindex_context,
        )

        logger.debug(
            "Embedding %d texts for LlamaIndex index: %s, node count: %d",
            len(texts_list),
            llamaindex_context.get("index_id", "unknown"),
            len(llamaindex_context.get("node_ids", []) or []),
        )

        start = time.perf_counter()
        translated = self._translator.embed(
            raw_texts=non_empty_texts,
            op_ctx=core_ctx,
            framework_ctx=framework_ctx,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        embeddings = self._coerce_embedding_matrix(translated)

        dim = _infer_dim_from_matrix(embeddings)
        if dim is not None:
            self._embedding_dim_hint = dim

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

    async def _aembed_text_batch(
        self,
        texts: Sequence[str],
        llamaindex_context: LlamaIndexContext,
    ) -> List[List[float]]:
        """Unified async batch text embedding implementation to eliminate duplication."""
        self._warn_if_extreme_batch(texts, op_name="_aget_text_embeddings")

        texts_list = list(texts)
        if self.llamaindex_config.get("strict_text_types", True):
            _validate_texts_are_strings(texts_list, op_name="_aget_text_embeddings")

        non_empty_texts = [t for t in texts_list if isinstance(t, str) and t.strip()]
        empty_indices_list = [
            i for i, t in enumerate(texts_list) if not isinstance(t, str) or not t.strip()
        ]
        empty_indices = set(empty_indices_list)

        if not non_empty_texts:
            dim = self.embedding_dimension
            return [[0.0] * dim for _ in texts_list]

        core_ctx, framework_ctx = self._build_contexts(
            llamaindex_context=llamaindex_context,
        )

        logger.debug(
            "Async embedding %d texts for LlamaIndex index: %s, node count: %d",
            len(texts_list),
            llamaindex_context.get("index_id", "unknown"),
            len(llamaindex_context.get("node_ids", []) or []),
        )

        start = time.perf_counter()
        translated = await self._translator.arun_embed(
            raw_texts=non_empty_texts,
            op_ctx=core_ctx,
            framework_ctx=framework_ctx,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        embeddings = self._coerce_embedding_matrix(translated)

        dim = _infer_dim_from_matrix(embeddings)
        if dim is not None:
            self._embedding_dim_hint = dim

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
    # Core LlamaIndex Abstract Method Implementation
    # ------------------------------------------------------------------ #

    @with_embedding_error_context("query")
    def _get_query_embedding(self, query: str, **kwargs: Any) -> List[float]:
        """
        Sync query embedding implementation for LlamaIndex.
        """
        if not query or not query.strip():
            return self._handle_empty_text(query)

        context = _filter_llamaindex_context_from_kwargs(kwargs)
        return self._embed_single_text(query, context)

    @with_async_embedding_error_context("query")
    async def _aget_query_embedding(self, query: str, **kwargs: Any) -> List[float]:
        """
        Async query embedding implementation for LlamaIndex.
        """
        if not query or not query.strip():
            return self._handle_empty_text(query)

        context = _filter_llamaindex_context_from_kwargs(kwargs)
        return await self._aembed_single_text(query, context)

    @with_embedding_error_context("text")
    def _get_text_embedding(self, text: str, **kwargs: Any) -> List[float]:
        """
        Sync text embedding implementation for LlamaIndex nodes.
        """
        if not text or not text.strip():
            return self._handle_empty_text(text)

        context = _filter_llamaindex_context_from_kwargs(kwargs)
        return self._embed_single_text(text, context)

    @with_async_embedding_error_context("text")
    async def _aget_text_embedding(self, text: str, **kwargs: Any) -> List[float]:
        """
        Async text embedding implementation for LlamaIndex nodes.
        """
        if not text or not text.strip():
            return self._handle_empty_text(text)

        context = _filter_llamaindex_context_from_kwargs(kwargs)
        return await self._aembed_single_text(text, context)

    @with_embedding_error_context("texts")
    def _get_text_embeddings(
        self,
        texts: Sequence[str],
        **kwargs: Any,
    ) -> List[List[float]]:
        """
        Batch text embedding implementation for LlamaIndex nodes.
        """
        context = _filter_llamaindex_context_from_kwargs(kwargs)
        return self._embed_text_batch(texts, context)

    @with_async_embedding_error_context("texts")
    async def _aget_text_embeddings(
        self,
        texts: Sequence[str],
        **kwargs: Any,
    ) -> List[List[float]]:
        """
        Async batch text embedding implementation for LlamaIndex nodes.
        """
        context = _filter_llamaindex_context_from_kwargs(kwargs)
        return await self._aembed_text_batch(texts, context)


# ------------------------------------------------------------------ #
# LlamaIndex Settings / Service Registration Helpers
# ------------------------------------------------------------------ #


def configure_llamaindex_embeddings(
    corpus_adapter: EmbeddingProtocolV1,
    model_name: str = "corpus-embedding-protocol",
    llamaindex_config: Optional[LlamaIndexAdapterConfig] = None,
    **kwargs: Any,
) -> CorpusLlamaIndexEmbeddings:
    """
    Configure and optionally register Corpus embeddings with LlamaIndex.

    This mirrors the behavior of `register_with_semantic_kernel`:

    - Always constructs and returns a `CorpusLlamaIndexEmbeddings` instance.
    - When LlamaIndex is installed and `llama_index.core.Settings` is available,
      it will *attempt* to register the embeddings as the global `embed_model`.
      If that fails (version differences, custom configs, etc.), it logs a warning
      and still returns the embeddings instance.
    """
    embeddings = CorpusLlamaIndexEmbeddings(
        corpus_adapter=corpus_adapter,
        model_name=model_name,
        llamaindex_config=llamaindex_config,
        **kwargs,
    )

    if not LLAMAINDEX_AVAILABLE:
        logger.debug(
            "LlamaIndex is not installed; returning embeddings without global "
            "Settings registration.",
        )
        return embeddings

    # Best-effort registration with LlamaIndex global Settings
    try:
        from llama_index.core import Settings  # type: ignore

        try:
            Settings.embed_model = embeddings
            logger.info(
                "Corpus LlamaIndex embeddings configured and registered as "
                "Settings.embed_model: %s",
                model_name,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Failed to configure LlamaIndex Settings.embed_model: %s. "
                "You may need to register the embeddings manually.",
                e,
            )
    except ImportError:  # pragma: no cover - older / different LlamaIndex layouts
        logger.debug(
            "LlamaIndex Settings API not available; unable to auto-register "
            "embed_model. Returning embeddings instance only.",
        )

    return embeddings


def register_with_llamaindex(
    corpus_adapter: EmbeddingProtocolV1,
    model_name: str = "corpus-embedding-protocol",
    llamaindex_config: Optional[LlamaIndexAdapterConfig] = None,
    **kwargs: Any,
) -> CorpusLlamaIndexEmbeddings:
    """
    Alias for `configure_llamaindex_embeddings` to mirror the
    `register_with_semantic_kernel` naming convention.

    This helper is convenient when you want symmetrical API shapes across
    framework adapters.
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
