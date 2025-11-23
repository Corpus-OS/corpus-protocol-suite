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

from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.callbacks import CallbackManager

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

logger = logging.getLogger(__name__)

# Type variables for decorators
T = TypeVar("T")

# Use LlamaIndex's default batch size constant
try:
    from llama_index.core.embeddings import DEFAULT_EMBED_BATCH_SIZE
except ImportError:  # pragma: no cover - fallback for older LlamaIndex
    DEFAULT_EMBED_BATCH_SIZE = 512


# ---------------------------------------------------------------------------
# Error codes (aligned with other embedding adapters)
# ---------------------------------------------------------------------------


class ErrorCodes:
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
    
    This provides the same rich observability as the context manager approach
    while maintaining decorator consistency with other adapters.
    """
    def decorator_factory(
        **static_context: Any,
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            if is_async:
                @wraps(func)
                async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
                    # Extract dynamic context from call
                    dynamic_context = _extract_dynamic_context(self, args, kwargs, operation)
                    full_context = {**static_context, **dynamic_context}
                    
                    try:
                        return await func(self, *args, **kwargs)
                    except Exception as exc:
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
                    # Extract dynamic context from call
                    dynamic_context = _extract_dynamic_context(self, args, kwargs, operation)
                    full_context = {**static_context, **dynamic_context}
                    
                    try:
                        return func(self, *args, **kwargs)
                    except Exception as exc:
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
    
    Provides the same per-call metrics as the context manager approach:
    - text_len for single text operations
    - texts_count for batch operations  
    - node_ids, index_id, callback_manager presence
    - model_name from instance
    """
    dynamic_ctx: Dict[str, Any] = {
        "model_name": getattr(instance, "model_name", "unknown"),
    }
    
    # Extract text-based metrics
    if operation in ["query", "text"] and args and isinstance(args[0], str):
        dynamic_ctx["text_len"] = len(args[0])
    elif operation in ["texts"] and args and isinstance(args[0], list):
        dynamic_ctx["texts_count"] = len(args[0])
        # Also include empty text count for better diagnostics
        empty_count = sum(1 for text in args[0] if not text or not text.strip())
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

    Example:
    ```python
    from llama_index.core import VectorStoreIndex, ServiceContext
    from llama_index.core.node_parser import SentenceSplitter
    from corpus_sdk.embedding.framework_adapters.llamaindex import (
        CorpusLlamaIndexEmbeddings,
        configure_llamaindex_embeddings
    )

    # Initialize with any Corpus EmbeddingProtocolV1 adapter
    embeddings = configure_llamaindex_embeddings(
        corpus_adapter=my_adapter,
        model_name="text-embedding-3-large",
        embed_batch_size=256
    )

    # Use with LlamaIndex service context
    service_context = ServiceContext.from_defaults(
        embed_model=embeddings,
        node_parser=SentenceSplitter(chunk_size=512)
    )

    # Create index with Corpus embeddings
    index = VectorStoreIndex.from_documents(
        documents=documents,
        service_context=service_context
    )

    # Embeddings automatically receive LlamaIndex context during queries
    query_engine = index.as_query_engine()
    response = query_engine.query("What are the key findings?")
    ```

    Error Handling Example:
    ```python
    try:
        results = embeddings._get_text_embeddings(
            texts=research_nodes,
            node_ids=[node.node_id for node in nodes],
            index_id="research_papers"
        )
    except Exception as e:
        # Rich error context automatically attached with text counts, node info, etc.
        logger.error("Embedding failed with context", exc_info=e)
    ```

    Non-responsibilities
    --------------------
    - Document chunking and node creation (handled by LlamaIndex)
    - Index management and storage (handled by LlamaIndex vector stores)
    - Retrieval strategies and query planning (handled by LlamaIndex query engines)

    All embedding logic lives in:
    - `corpus_sdk.embedding.framework_adapters.common.embedding_translation`
    - Concrete `EmbeddingProtocolV1` adapter implementations.

    Attributes
    ----------
    corpus_adapter:
        Underlying Corpus embedding adapter implementing `EmbeddingProtocolV1`.

    model_name:
        Optional model identifier used in LlamaIndex settings. Defaults to
        "corpus-embedding-protocol". Can be overridden via LlamaIndex Settings.

    batch_config:
        Optional `BatchConfig` to control batching behavior.

    text_normalization_config:
        Optional `TextNormalizationConfig` to control whitespace cleanup,
        truncation, casing, encoding, etc.

    llama_index_config:
        Optional LlamaIndex-specific configuration for service context
        integration and callback management.
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

        Parameters
        ----------
        corpus_adapter:
            Corpus embedding protocol adapter
        model_name:
            Model identifier for LlamaIndex settings integration
        batch_config:
            Batching configuration for embedding requests
        text_normalization_config:
            Text normalization settings
        llama_index_config:
            LlamaIndex-specific configuration
        callback_manager:
            LlamaIndex callback manager for observability
        embed_batch_size:
            Batch size for embedding operations, defaults to LlamaIndex's standard
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
    ) -> Tuple[Optional[OperationContext], Dict[str, Any], Dict[str, Any]]:
        """
        Build contexts for LlamaIndex execution environment with comprehensive validation.

        Returns
        -------
        Tuple of:
        - core_ctx: core OperationContext or None if no/invalid context
        - op_ctx_dict: normalized dict for embedding layer (preserving rich context)
        - framework_ctx: LlamaIndex-specific context for translator
        """
        core_ctx: Optional[OperationContext] = None
        op_ctx_dict: Dict[str, Any] = {}
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
                # Validate structure for better observability
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
                        "Proceeding with empty OperationContext dictionary.",
                        type(core_ctx_candidate).__name__,
                    )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Failed to create OperationContext from LlamaIndex context: %s. "
                    "Proceeding with degraded context.",
                    e,
                )
                try:
                    attach_context(
                        e,
                        framework="llamaindex",
                        operation="context_build",
                        context_snapshot=dict(llamaindex_context),
                    )
                except Exception:
                    # Do not mask original issues in context translation
                    pass

        # Preserve rich OperationContext and also provide a dict representation
        if core_ctx is not None:
            op_ctx_dict = {"_operation_context": core_ctx}
            if hasattr(core_ctx, "to_dict"):
                op_ctx_dict.update(core_ctx.to_dict())
            elif hasattr(core_ctx, "__dict__"):
                op_ctx_dict.update(core_ctx.__dict__)
        else:
            op_ctx_dict = {}

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

        # Also expose the OperationContext itself to the translator for maximum fidelity
        if core_ctx is not None:
            framework_ctx["_operation_context"] = core_ctx

        return core_ctx, op_ctx_dict, framework_ctx

    def _validate_llamaindex_context_structure(self, context: Mapping[str, Any]) -> None:
        """Validate LlamaIndex context structure and log warnings for anomalies."""
        # Check for common LlamaIndex context fields
        if not any(key in context for key in ['node_ids', 'index_id', 'callback_manager', 'trace_id']):
            logger.debug(
                "LlamaIndex context missing common fields (node_ids, index_id, etc.) - "
                "reduced context for embeddings"
            )

    def _coerce_embedding_matrix(self, result: Any) -> List[List[float]]:
        """
        Coerce translator result into a List[List[float]] embedding matrix with comprehensive validation.

        Supported shapes:
        - {"embeddings": [[...], [...]], "model": "...", "usage": {...}}
        - Direct matrix: [[...], [...]]
        - EmbedResult-like with `.embeddings` attribute

        Python 3.9–compatible (avoids `match` / `case`).
        """
        if isinstance(result, Mapping) and "embeddings" in result:
            embeddings_obj: Any = result["embeddings"]
        elif hasattr(result, "embeddings"):
            embeddings_obj = getattr(result, "embeddings")
        else:
            embeddings_obj = result

        if not isinstance(embeddings_obj, Sequence):
            raise TypeError(
                f"[{ErrorCodes.INVALID_EMBEDDING_RESULT}] "
                f"Translator result does not contain a valid embeddings sequence: "
                f"type={type(embeddings_obj).__name__}"
            )

        matrix: List[List[float]] = []
        for i, row in enumerate(embeddings_obj):
            if not isinstance(row, Sequence):
                raise TypeError(
                    f"[{ErrorCodes.INVALID_EMBEDDING_RESULT}] "
                    f"Expected each embedding row to be a sequence, "
                    f"got {type(row).__name__} at index {i}"
                )
            
            # Validate row is not empty
            if len(row) == 0:
                logger.warning("Empty embedding row at index %d, skipping", i)
                continue

            try:
                embedding_vector = [float(x) for x in row]
                matrix.append(embedding_vector)
            except (TypeError, ValueError) as e:
                raise TypeError(
                    f"[{ErrorCodes.EMBEDDING_CONVERSION_ERROR}] "
                    f"Failed to convert embedding values to float at row {i}: {e}"
                ) from e

        if not matrix:
            raise ValueError(
                f"[{ErrorCodes.EMPTY_EMBEDDING_RESULT}] "
                "Translator returned no valid embedding rows"
            )

        logger.debug(
            "Successfully coerced embedding matrix with %d rows",
            len(matrix),
        )
        return matrix

    def _coerce_embedding_vector(self, result: Any) -> List[float]:
        """
        Coerce translator result for a single-text embed into List[float] with validation.

        Strategy:
        - If the matrix is empty → _coerce_embedding_matrix raises
        - If it has exactly one row → return that row
        - If it has multiple rows → return the first row and log a warning
        """
        matrix = self._coerce_embedding_matrix(result)

        if len(matrix) > 1:
            logger.warning(
                "Expected a single embedding for query, but got %d rows; "
                "using the first row.",
                len(matrix),
            )

        return matrix[0]

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
        if isinstance(texts, (str, bytes)):
            return

        batch_size = len(texts)
        if batch_size <= 10_000:
            return

        max_batch_size = (
            None
            if self.batch_config is None
            else getattr(self.batch_config, "max_batch_size", None)
        )
        if max_batch_size is None:
            logger.warning(
                "%s called with batch_size=%d and no explicit BatchConfig.max_batch_size; "
                "ensure your adapter/translator can handle very large batches.",
                op_name,
                batch_size,
            )

    def _embed_single_text(self, text: str, llamaindex_context: Dict[str, Any]) -> List[float]:
        """Unified single text embedding implementation to eliminate duplication."""
        core_ctx, op_ctx_dict, framework_ctx = self._build_contexts(
            llamaindex_context=llamaindex_context,
            **llamaindex_context,
        )

        # Rich logging with dynamic context
        logger.debug(
            "Embedding single text for LlamaIndex index: %s, node count: %d",
            llamaindex_context.get("index_id", "unknown"),
            len(llamaindex_context.get("node_ids", [])),
        )

        translated = self._translator.embed(
            raw_texts=text,
            op_ctx=op_ctx_dict,
            framework_ctx=framework_ctx,
        )
        return self._coerce_embedding_vector(translated)

    async def _aembed_single_text(self, text: str, llamaindex_context: Dict[str, Any]) -> List[float]:
        """Unified async single text embedding implementation to eliminate duplication."""
        core_ctx, op_ctx_dict, framework_ctx = self._build_contexts(
            llamaindex_context=llamaindex_context,
            **llamaindex_context,
        )

        # Rich logging with dynamic context
        logger.debug(
            "Async embedding single text for LlamaIndex index: %s, node count: %d",
            llamaindex_context.get("index_id", "unknown"),
            len(llamaindex_context.get("node_ids", [])),
        )

        translated = await self._translator.arun_embed(
            raw_texts=text,
            op_ctx=op_ctx_dict,
            framework_ctx=framework_ctx,
        )
        return self._coerce_embedding_vector(translated)

    def _embed_text_batch(self, texts: List[str], llamaindex_context: Dict[str, Any]) -> List[List[float]]:
        """Unified batch text embedding implementation to eliminate duplication."""
        self._warn_if_extreme_batch(texts, op_name="_get_text_embeddings")

        non_empty_texts = [t for t in texts if t and t.strip()]
        empty_indices = [i for i, t in enumerate(texts) if not t or not t.strip()]

        if not non_empty_texts:
            dimension = self.embedding_dimension
            return [[0.0] * dimension for _ in texts]

        core_ctx, op_ctx_dict, framework_ctx = self._build_contexts(
            llamaindex_context=llamaindex_context,
            **llamaindex_context,
        )

        # Rich logging with dynamic context
        logger.debug(
            "Embedding %d texts for LlamaIndex index: %s, node count: %d",
            len(texts),
            llamaindex_context.get("index_id", "unknown"),
            len(llamaindex_context.get("node_ids", [])),
        )

        translated = self._translator.embed(
            raw_texts=non_empty_texts,
            op_ctx=op_ctx_dict,
            framework_ctx=framework_ctx,
        )
        embeddings = self._coerce_embedding_matrix(translated)

        if empty_indices:
            dimension = (
                len(embeddings[0]) if embeddings else self.embedding_dimension
            )
            result_embeddings: List[List[float]] = []
            non_empty_idx = 0
            for i in range(len(texts)):
                if i in empty_indices:
                    result_embeddings.append([0.0] * dimension)
                else:
                    result_embeddings.append(embeddings[non_empty_idx])
                    non_empty_idx += 1
            return result_embeddings

        return embeddings

    async def _aembed_text_batch(self, texts: List[str], llamaindex_context: Dict[str, Any]) -> List[List[float]]:
        """Unified async batch text embedding implementation to eliminate duplication."""
        self._warn_if_extreme_batch(texts, op_name="_aget_text_embeddings")

        non_empty_texts = [t for t in texts if t and t.strip()]
        empty_indices = [i for i, t in enumerate(texts) if not t or not t.strip()]

        if not non_empty_texts:
            dimension = self.embedding_dimension
            return [[0.0] * dimension for _ in texts]

        core_ctx, op_ctx_dict, framework_ctx = self._build_contexts(
            llamaindex_context=llamaindex_context,
            **llamaindex_context,
        )

        # Rich logging with dynamic context
        logger.debug(
            "Async embedding %d texts for LlamaIndex index: %s, node count: %d",
            len(texts),
            llamaindex_context.get("index_id", "unknown"),
            len(llamaindex_context.get("node_ids", [])),
        )

        translated = await self._translator.arun_embed(
            raw_texts=non_empty_texts,
            op_ctx=op_ctx_dict,
            framework_ctx=framework_ctx,
        )
        embeddings = self._coerce_embedding_matrix(translated)

        if empty_indices:
            dimension = (
                len(embeddings[0]) if embeddings else self.embedding_dimension
            )
            result_embeddings: List[List[float]] = []
            non_empty_idx = 0
            for i in range(len(texts)):
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
    def _get_text_embeddings(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        """
        Batch text embedding implementation for LlamaIndex nodes.
        """
        return self._embed_text_batch(texts, kwargs)

    @with_async_embedding_error_context("texts")
    async def _aget_text_embeddings(
        self,
        texts: List[str],
        **kwargs: Any,
    ) -> List[List[float]]:
        """
        Async batch text embedding implementation for LlamaIndex nodes.
        """
        return await self._aembed_text_batch(texts, kwargs)


# ------------------------------------------------------------------ #
# LlamaIndex Settings Integration
# ------------------------------------------------------------------ #


def configure_llamaindex_embeddings(
    corpus_adapter: EmbeddingProtocolV1,
    model_name: str = "corpus-embedding-protocol",
    **kwargs: Any,
) -> CorpusLlamaIndexEmbeddings:
    """
    Configure and return Corpus embeddings for LlamaIndex global settings.

    Example:
    ```python
    from llama_index.core import Settings
    from corpus_sdk.embedding.framework_adapters.llamaindex import configure_llamaindex_embeddings

    # Configure for global LlamaIndex settings
    embeddings = configure_llamaindex_embeddings(
        corpus_adapter=my_adapter,
        model_name="text-embedding-3-large",
        embed_batch_size=512
    )

    # Set as global embed model
    Settings.embed_model = embeddings
    ```

    Parameters
    ----------
    corpus_adapter:
        Corpus embedding protocol adapter
    model_name:
        Model identifier for LlamaIndex settings
    **kwargs:
        Additional arguments for CorpusLlamaIndexEmbeddings

    Returns
    -------
    CorpusLlamaIndexEmbeddings
        Configured embedding model for LlamaIndex
    """
    embeddings = CorpusLlamaIndexEmbeddings(
        corpus_adapter=corpus_adapter,
        model_name=model_name,
        **kwargs,
    )

    logger.info("Corpus LlamaIndex embeddings configured: %s", model_name)
    return embeddings


__all__ = [
    "CorpusLlamaIndexEmbeddings",
    "LlamaIndexContext",
    "configure_llamaindex_embeddings",
    "ErrorCodes",
    "with_embedding_error_context",
    "with_async_embedding_error_context",
]
