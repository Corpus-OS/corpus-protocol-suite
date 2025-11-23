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

Resilience (retries, caching, rate limiting, etc.) is expected to be provided by the underlying adapter, typically a BaseEmbeddingAdapter subclass.
"""

from __future__ import annotations

import logging
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

# Type variables for decorators
T = TypeVar("T")

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

        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
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


class ErrorCodes(CoercionErrorCodes):
    """
    Error code constants for the LlamaIndex embedding adapter.

    Inherits from CoercionErrorCodes so shared coercion utilities can
    reference the same symbolic names while remaining framework-specific.
    """

    INVALID_EMBEDDING_RESULT = "INVALID_EMBEDDING_RESULT"
    EMPTY_EMBEDDING_RESULT = "EMPTY_EMBEDDING_RESULT"
    EMBEDDING_CONVERSION_ERROR = "EMBEDDING_CONVERSION_ERROR"
    LLAMAINDEX_CONTEXT_INVALID = "LLAMAINDEX_CONTEXT_INVALID"


class LlamaIndexContext(TypedDict, total=False):
    """Structured type for LlamaIndex execution context."""
    node_ids: Optional[List[str]]
    index_id: Optional[str]
    callback_manager: Optional[Any]
    trace_id: Optional[str]
    workflow: Optional[str]


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
                    full_context = {**static_context, **dynamic_context}
                    try:
                        return await func(self, *args, **kwargs)
                    except Exception as exc:  # noqa: BLE001
                        attach_context(
                            exc,
                            framework="llamaindex",
                            operation=f"embedding_{operation}",
                            **full_context,
                        )
                        raise

                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
                    dynamic_context = _extract_dynamic_context(
                        self,
                        args,
                        kwargs,
                        operation,
                    )
                    full_context = {**static_context, **dynamic_context}
                    try:
                        return func(self, *args, **kwargs)
                    except Exception as exc:  # noqa: BLE001
                        attach_context(
                            exc,
                            framework="llamaindex",
                            operation=f"embedding_{operation}",
                            **full_context,
                        )
                        raise

                return sync_wrapper

        return decorator
    return decorator_factory


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
    - node_ids, node_count, index_id, callback_manager presence
    - model_name from the embedding instance
    """
    dynamic_ctx: Dict[str, Any] = {
        "model_name": getattr(instance, "model_name", "unknown"),
    }

    # Extract text-based metrics
    if operation in ["query", "text"] and args and isinstance(args[0], str):
        dynamic_ctx["text_len"] = len(args[0])
    elif operation == "texts" and args and isinstance(args[0], Sequence):
        texts_seq = args[0]
        dynamic_ctx["texts_count"] = len(texts_seq)
        empty_count = sum(
            1 for text in texts_seq if not isinstance(text, str) or not text.strip()
        )
        if empty_count > 0:
            dynamic_ctx["empty_texts_count"] = empty_count

    # Extract LlamaIndex-specific context from kwargs
    if "node_ids" in kwargs:
        dynamic_ctx["node_ids"] = kwargs["node_ids"]
        dynamic_ctx["node_count"] = len(kwargs["node_ids"]) if kwargs["node_ids"] else 0
    if "index_id" in kwargs:
        dynamic_ctx["index_id"] = kwargs["index_id"]
    if "callback_manager" in kwargs:
        dynamic_ctx["has_callback_manager"] = bool(kwargs["callback_manager"])

    return dynamic_ctx


# Convenience decorators with rich context extraction
def with_embedding_error_context(
    operation: str,
    **static_context: Any,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for sync methods with rich dynamic context extraction."""
    return _create_error_context_decorator(operation, is_async=False)(**static_context)


def with_async_embedding_error_context(
    operation: str,
    **static_context: Any,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for async methods with rich dynamic context extraction."""
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
        llama_index_config: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        **kwargs: Any,
    ):
        """
        Initialize Corpus LlamaIndex Embeddings.
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

        self.corpus_adapter = corpus_adapter
        self._model_name = model_name
        self.batch_config = batch_config
        self.text_normalization_config = text_normalization_config
        self.llama_index_config = llama_index_config or {}
        self._embed_batch_size = embed_batch_size

        # Initialize BaseEmbedding with LlamaIndex expected parameters
        super().__init__(
            model_name=self._model_name,
            embed_batch_size=self._embed_batch_size,
            callback_manager=callback_manager,
            **kwargs,
        )

        logger.info(
            "CorpusLlamaIndexEmbeddings initialized with model_name=%s, "
            "embed_batch_size=%d",
            self._model_name,
            self._embed_batch_size,
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
        """
        translator = create_embedding_translator(
            adapter=self.corpus_adapter,
            framework="llamaindex",
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
        llamaindex_context: Optional[Mapping[str, Any]] = None,
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
            "framework": "llamaindex",
            "model_name": self.model_name,
        }

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
                core_ctx_candidate = context_from_llamaindex(llamaindex_context)
                if isinstance(core_ctx_candidate, OperationContext):
                    core_ctx = core_ctx_candidate
                    logger.debug(
                        "Successfully created OperationContext from LlamaIndex context "
                        "with index_id=%s",
                        llamaindex_context.get("index_id", "unknown"),
                    )
                else:
                    logger.warning(
                        "context_from_llamaindex returned non-OperationContext type: %s. "
                        "Proceeding with empty OperationContext.",
                        type(core_ctx_candidate).__name__,
                    )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Failed to create OperationContext from LlamaIndex context: %s. "
                    "Proceeding with degraded context.",
                    e,
                )
                try:
                    snapshot = dict(llamaindex_context)
                except Exception:  # noqa: BLE001
                    snapshot = {"repr": repr(llamaindex_context)}
                try:
                    attach_context(
                        e,
                        framework="llamaindex",
                        operation="context_build",
                        context_snapshot=snapshot,
                    )
                except Exception:
                    # Never mask upstream exceptions when attaching context
                    pass

        # Add LlamaIndex-specific context for nodes and retrieval
        if llamaindex_context:
            if "node_ids" in llamaindex_context:
                framework_ctx["node_ids"] = llamaindex_context["node_ids"]
            if "index_id" in llamaindex_context:
                framework_ctx["index_id"] = llamaindex_context["index_id"]
            if "callback_manager" in llamaindex_context:
                framework_ctx["callback_manager"] = llamaindex_context[
                    "callback_manager"
                ]
            if "trace_id" in llamaindex_context:
                framework_ctx["trace_id"] = llamaindex_context["trace_id"]

        # Include any extra call-specific hints while preserving structure
        framework_ctx.update(kwargs)

        # Stash OperationContext for downstream inspection
        if core_ctx is not None:
            framework_ctx["_operation_context"] = core_ctx

        return core_ctx, framework_ctx

    def _validate_llamaindex_context_structure(self, context: Mapping[str, Any]) -> None:
        """Validate LlamaIndex context structure and log warnings for anomalies."""
        if not any(
            key in context
            for key in ("node_ids", "index_id", "callback_manager", "trace_id")
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
            error_codes=ErrorCodes,
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
            error_codes=ErrorCodes,
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
            Embedding dimension, with fallback to common default (768)
        """
        if hasattr(self.corpus_adapter, "get_embedding_dimension"):
            try:
                return int(self.corpus_adapter.get_embedding_dimension())
            except Exception as e:  # noqa: BLE001
                logger.debug(
                    "Failed to get embedding dimension from adapter: %s", e
                )

        # Common fallback dimension
        return 768

    def _handle_empty_text(self, text: str) -> List[float]:
        """
        Handle empty text by returning appropriate zero vector.
        """
        logger.warning("Empty text provided for embedding, returning zero vector")
        dimension = self.embedding_dimension
        return [0.0] * dimension

    def _warn_if_extreme_batch(self, texts: Sequence[str], *, op_name: str) -> None:
        """
        Emit a soft warning if an extremely large batch is requested
        without an explicit BatchConfig.max_batch_size.
        """
        warn_if_extreme_batch(
            framework="llamaindex",
            texts=texts,
            op_name=op_name,
            batch_config=self.batch_config,
            logger=logger,
        )

    def _embed_single_text(
        self,
        text: str,
        llamaindex_context: Dict[str, Any],
    ) -> List[float]:
        """Unified single text embedding implementation to eliminate duplication."""
        core_ctx, framework_ctx = self._build_contexts(
            llamaindex_context=llamaindex_context,
            **llamaindex_context,
        )

        logger.debug(
            "Embedding single text for LlamaIndex index: %s, node count: %d",
            llamaindex_context.get("index_id", "unknown"),
            len(llamaindex_context.get("node_ids", [])),
        )

        translated = self._translator.embed(
            raw_texts=text,
            op_ctx=core_ctx,
            framework_ctx=framework_ctx,
        )
        return self._coerce_embedding_vector(translated)

    async def _aembed_single_text(
        self,
        text: str,
        llamaindex_context: Dict[str, Any],
    ) -> List[float]:
        """Unified async single text embedding implementation to eliminate duplication."""
        core_ctx, framework_ctx = self._build_contexts(
            llamaindex_context=llamaindex_context,
            **llamaindex_context,
        )

        logger.debug(
            "Async embedding single text for LlamaIndex index: %s, node count: %d",
            llamaindex_context.get("index_id", "unknown"),
            len(llamaindex_context.get("node_ids", [])),
        )

        translated = await self._translator.arun_embed(
            raw_texts=text,
            op_ctx=core_ctx,
            framework_ctx=framework_ctx,
        )
        return self._coerce_embedding_vector(translated)

    def _embed_text_batch(
        self,
        texts: Sequence[str],
        llamaindex_context: Dict[str, Any],
    ) -> List[List[float]]:
        """Unified batch text embedding implementation to eliminate duplication."""
        self._warn_if_extreme_batch(texts, op_name="_get_text_embeddings")

        texts_list = list(texts)
        non_empty_texts = [
            t for t in texts_list if isinstance(t, str) and t.strip()
        ]
        empty_indices = [
            i for i, t in enumerate(texts_list)
            if not isinstance(t, str) or not t.strip()
        ]

        if not non_empty_texts:
            dimension = self.embedding_dimension
            return [[0.0] * dimension for _ in texts_list]

        core_ctx, framework_ctx = self._build_contexts(
            llamaindex_context=llamaindex_context,
            **llamaindex_context,
        )

        logger.debug(
            "Embedding %d texts for LlamaIndex index: %s, node count: %d",
            len(texts_list),
            llamaindex_context.get("index_id", "unknown"),
            len(llamaindex_context.get("node_ids", [])),
        )

        translated = self._translator.embed(
            raw_texts=non_empty_texts,
            op_ctx=core_ctx,
            framework_ctx=framework_ctx,
        )
        embeddings = self._coerce_embedding_matrix(translated)

        if empty_indices:
            dimension = (
                len(embeddings[0]) if embeddings else self.embedding_dimension
            )
            result_embeddings: List[List[float]] = []
            non_empty_idx = 0
            for i in range(len(texts_list)):
                if i in empty_indices:
                    result_embeddings.append([0.0] * dimension)
                else:
                    result_embeddings.append(embeddings[non_empty_idx])
                    non_empty_idx += 1
            return result_embeddings

        return embeddings

    async def _aembed_text_batch(
        self,
        texts: Sequence[str],
        llamaindex_context: Dict[str, Any],
    ) -> List[List[float]]:
        """Unified async batch text embedding implementation to eliminate duplication."""
        self._warn_if_extreme_batch(texts, op_name="_aget_text_embeddings")

        texts_list = list(texts)
        non_empty_texts = [
            t for t in texts_list if isinstance(t, str) and t.strip()
        ]
        empty_indices = [
            i for i, t in enumerate(texts_list)
            if not isinstance(t, str) or not t.strip()
        ]

        if not non_empty_texts:
            dimension = self.embedding_dimension
            return [[0.0] * dimension for _ in texts_list]

        core_ctx, framework_ctx = self._build_contexts(
            llamaindex_context=llamaindex_context,
            **llamaindex_context,
        )

        logger.debug(
            "Async embedding %d texts for LlamaIndex index: %s, node count: %d",
            len(texts_list),
            llamaindex_context.get("index_id", "unknown"),
            len(llamaindex_context.get("node_ids", [])),
        )

        translated = await self._translator.arun_embed(
            raw_texts=non_empty_texts,
            op_ctx=core_ctx,
            framework_ctx=framework_ctx,
        )
        embeddings = self._coerce_embedding_matrix(translated)

        if empty_indices:
            dimension = (
                len(embeddings[0]) if embeddings else self.embedding_dimension
            )
            result_embeddings: List[List[float]] = []
            non_empty_idx = 0
            for i in range(len(texts_list)):
                if i in empty_indices:
                    result_embeddings.append([0.0] * dimension)
                else:
                    result_embeddings.append(embeddings[non_empty_idx])
                    non_empty_idx += 1
            return result_embeddings

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
        return self._embed_single_text(query, kwargs)

    @with_async_embedding_error_context("query")
    async def _aget_query_embedding(self, query: str, **kwargs: Any) -> List[float]:
        """
        Async query embedding implementation for LlamaIndex.
        """
        if not query or not query.strip():
            return self._handle_empty_text(query)
        return await self._aembed_single_text(query, kwargs)

    @with_embedding_error_context("text")
    def _get_text_embedding(self, text: str, **kwargs: Any) -> List[float]:
        """
        Sync text embedding implementation for LlamaIndex nodes.
        """
        if not text or not text.strip():
            return self._handle_empty_text(text)
        return self._embed_single_text(text, kwargs)

    @with_async_embedding_error_context("text")
    async def _aget_text_embedding(self, text: str, **kwargs: Any) -> List[float]:
        """
        Async text embedding implementation for LlamaIndex nodes.
        """
        if not text or not text.strip():
            return self._handle_empty_text(text)
        return await self._aembed_single_text(text, kwargs)

    @with_embedding_error_context("texts")
    def _get_text_embeddings(
        self,
        texts: Sequence[str],
        **kwargs: Any,
    ) -> List[List[float]]:
        """
        Batch text embedding implementation for LlamaIndex nodes.
        """
        return self._embed_text_batch(texts, kwargs)

    @with_async_embedding_error_context("texts")
    async def _aget_text_embeddings(
        self,
        texts: Sequence[str],
        **kwargs: Any,
    ) -> List[List[float]]:
        """
        Async batch text embedding implementation for LlamaIndex nodes.
        """
        return await self._aembed_text_batch(texts, kwargs)


# ------------------------------------------------------------------ #
# LlamaIndex Settings / Service Registration Helpers
# ------------------------------------------------------------------ #


def configure_llamaindex_embeddings(
    corpus_adapter: EmbeddingProtocolV1,
    model_name: str = "corpus-embedding-protocol",
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

    Parameters
    ----------
    corpus_adapter:
        Corpus embedding protocol adapter implementing `EmbeddingProtocolV1`.
    model_name:
        Model identifier for embedding operations as seen by LlamaIndex.
    **kwargs:
        Additional arguments for `CorpusLlamaIndexEmbeddings` (e.g. batch_config,
        text_normalization_config, callback_manager, embed_batch_size).

    Returns
    -------
    CorpusLlamaIndexEmbeddings
        Configured embedding service instance.
    """
    embeddings = CorpusLlamaIndexEmbeddings(
        corpus_adapter=corpus_adapter,
        model_name=model_name,
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
    **kwargs: Any,
) -> CorpusLlamaIndexEmbeddings:
    """
    Alias for `configure_llamaindex_embeddings` to mirror the
    `register_with_semantic_kernel` naming convention.

    This helper is convenient when you want symmetrical API shapes across
    framework adapters:

    - `register_with_semantic_kernel(kernel, ...)`
    - `register_with_llamaindex(corpus_adapter, ...)`
    """
    return configure_llamaindex_embeddings(
        corpus_adapter=corpus_adapter,
        model_name=model_name,
        **kwargs,
    )


__all__ = [
    "CorpusLlamaIndexEmbeddings",
    "LlamaIndexContext",
    "configure_llamaindex_embeddings",
    "register_with_llamaindex",
    "ErrorCodes",
    "with_embedding_error_context",
    "with_async_embedding_error_context",
    "LLAMAINDEX_AVAILABLE",
]
