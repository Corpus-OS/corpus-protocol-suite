# corpus_sdk/embedding/framework_adapters/llamaindex.py
# SPDX-License-Identifier: Apache-2.0

"""
LlamaIndex adapter for Corpus Embedding protocol.

This module exposes Corpus `EmbeddingProtocolV1` implementations as
`llama_index.core.embeddings.BaseEmbedding`, with:

- Full compatibility with LlamaIndex's Settings configuration
- Support for LlamaIndex node-based document processing
- Context normalization using existing `context_translation.from_llamaindex`
- Framework-agnostic orchestration via `EmbeddingTranslator`
- Async → sync bridging handled in the common embedding layer
- Rich error context attachment for observability

The design leverages LlamaIndex's focus on efficient indexing and retrieval
while maintaining the protocol-first Corpus embedding stack.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import cached_property
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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

# Use LlamaIndex's default batch size constant
try:
    from llama_index.core.embeddings import DEFAULT_EMBED_BATCH_SIZE
except ImportError:
    DEFAULT_EMBED_BATCH_SIZE = 512


# ---------------------------------------------------------------------------
# Error codes (aligned with other embedding adapters)
# ---------------------------------------------------------------------------


class ErrorCodes:
    INVALID_EMBEDDING_RESULT = "INVALID_EMBEDDING_RESULT"
    EMPTY_EMBEDDING_RESULT = "EMPTY_EMBEDDING_RESULT"
    EMBEDDING_CONVERSION_ERROR = "EMBEDDING_CONVERSION_ERROR"


# ---------------------------------------------------------------------------
# Error Context Decorator (shared schema)
# ---------------------------------------------------------------------------


@contextmanager
def _embedding_error_context(
    operation: str,
    *,
    model_name: str,
    text_len: Optional[int] = None,
    texts_count: Optional[int] = None,
    framework_ctx: Optional[Dict[str, Any]] = None,
):
    """
    Context manager for consistent error context attachment in embedding operations.

    Attaches a Corpus-wide error context payload with:
      - framework="llamaindex"
      - operation=<operation>
      - model_name=<embedding model>
      - text_len / texts_count
      - node_ids / index_id / has_callback_manager (when available)
    """
    try:
        yield
    except Exception as exc:  # noqa: BLE001
        ctx: Dict[str, Any] = {
            "framework": "llamaindex",
            "operation": operation,
            "model_name": model_name,
        }
        if text_len is not None:
            ctx["text_len"] = text_len
        if texts_count is not None:
            ctx["texts_count"] = texts_count

        if framework_ctx:
            if "node_ids" in framework_ctx:
                ctx["node_ids"] = framework_ctx.get("node_ids")
            if "index_id" in framework_ctx:
                ctx["index_id"] = framework_ctx.get("index_id")
            if "callback_manager" in framework_ctx:
                ctx["has_callback_manager"] = bool(
                    framework_ctx.get("callback_manager")
                )

        try:
            attach_context(exc, **ctx)
        except Exception:
            # Never mask the original error if context attachment fails
            pass
        raise


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
        """
        # Validate critical parameters
        if not isinstance(corpus_adapter, EmbeddingProtocolV1):
            raise TypeError("corpus_adapter must implement EmbeddingProtocolV1")

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
        return create_embedding_translator(
            adapter=self.corpus_adapter,
            framework="llamaindex",
            translator=None,  # use registry/default generic translator
            batch_config=self.batch_config,
            text_normalization_config=self.text_normalization_config,
        )

    def _build_contexts(
        self,
        *,
        llamaindex_context: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[OperationContext, Dict[str, Any], Dict[str, Any]]:
        """
        Build contexts for LlamaIndex execution environment.

        Returns
        -------
        Tuple of:
        - `core_ctx`: core OperationContext (from existing context translation)
        - `op_ctx_dict`: normalized dict for embedding layer
        - `framework_ctx`: LlamaIndex-specific context for translator
        """
        # Validate input
        if llamaindex_context is not None and not isinstance(
            llamaindex_context, Mapping
        ):
            logger.warning(
                "llamaindex_context should be a Mapping, got %s",
                type(llamaindex_context),
            )
            llamaindex_context = None

        try:
            core_ctx: OperationContext = context_from_llamaindex(llamaindex_context)

            # Normalized dict for embedding OperationContext reconstruction
            op_ctx_dict: Dict[str, Any] = {}
            if hasattr(core_ctx, "to_dict"):
                op_ctx_dict = core_ctx.to_dict()
            elif hasattr(core_ctx, "__dict__"):
                op_ctx_dict = core_ctx.__dict__

            # Framework-level context for LlamaIndex-specific hints
            framework_ctx: Dict[str, Any] = {
                "framework": "llamaindex",
                "model_name": self.model_name,
            }

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

            # Add any additional kwargs to framework context
            framework_ctx.update(kwargs)

            return core_ctx, op_ctx_dict, framework_ctx
        except Exception as e:
            logger.error("Failed to build LlamaIndex contexts: %s", e)
            raise ValueError(f"Context building failed: {e}") from e

    # ------------------------------------------------------------------ #
    # Result coercion (Python 3.9 compatible, with ErrorCodes)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _coerce_embedding_matrix(result: Any) -> List[List[float]]:
        """
        Coerce translator result into a List[List[float]] embedding matrix.

        Supported shapes:
        - {"embeddings": [[...], [...]], "model": "...", "usage": {...}}
        - Direct matrix: [[...], [...]]
        - EmbedResult-like with `.embeddings` attribute
        """
        if isinstance(result, Mapping) and "embeddings" in result:
            embeddings_obj: Any = result["embeddings"]
        elif hasattr(result, "embeddings"):
            embeddings_obj = getattr(result, "embeddings")
        else:
            embeddings_obj = result

        if not isinstance(embeddings_obj, Sequence):
            raise TypeError(
                "Translator result does not contain a valid embeddings sequence: "
                f"type={type(embeddings_obj).__name__}",
                code=ErrorCodes.INVALID_EMBEDDING_RESULT,
            )

        matrix: List[List[float]] = []
        for i, row in enumerate(embeddings_obj):
            if not isinstance(row, Sequence):
                raise TypeError(
                    "Expected each embedding row to be a sequence, "
                    f"got {type(row).__name__} at index {i}",
                    code=ErrorCodes.INVALID_EMBEDDING_RESULT,
                )
            try:
                matrix.append([float(x) for x in row])
            except (TypeError, ValueError) as e:
                raise TypeError(
                    f"Failed to convert embedding values to float at row {i}: {e}",
                    code=ErrorCodes.EMBEDDING_CONVERSION_ERROR,
                ) from e

        return matrix

    @staticmethod
    def _coerce_embedding_vector(result: Any) -> List[float]:
        """
        Coerce translator result for a single-text embed into List[float].
        """
        matrix = CorpusLlamaIndexEmbeddings._coerce_embedding_matrix(result)

        if not matrix:
            raise ValueError(
                "Translator returned no embeddings for single-text input",
                code=ErrorCodes.EMPTY_EMBEDDING_RESULT,
            )

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
    def _get_embedding_dimension(self) -> int:
        """
        Get embedding dimension for proper zero vector fallback.

        Returns
        -------
        int
            Embedding dimension, with fallback to common default (768)
        """
        if hasattr(self.corpus_adapter, "get_embedding_dimension"):
            try:
                return self.corpus_adapter.get_embedding_dimension()
            except Exception as e:
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
        dimension = self._get_embedding_dimension
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

    # ------------------------------------------------------------------ #
    # Core LlamaIndex Abstract Method Implementation
    # ------------------------------------------------------------------ #

    def _get_query_embedding(self, query: str, **kwargs: Any) -> List[float]:
        """
        Sync query embedding implementation for LlamaIndex.
        """
        if not query or not query.strip():
            return self._handle_empty_text(query)

        _core_ctx, op_ctx_dict, framework_ctx = self._build_contexts(
            llamaindex_context=kwargs,
            **kwargs,
        )

        with _embedding_error_context(
            operation="query",
            text_len=len(query),
            framework_ctx=framework_ctx,
            model_name=self.model_name,
        ):
            translated = self._translator.embed(
                raw_texts=query,
                op_ctx=op_ctx_dict,
                framework_ctx=framework_ctx,
            )
            return self._coerce_embedding_vector(translated)

    async def _aget_query_embedding(self, query: str, **kwargs: Any) -> List[float]:
        """
        Async query embedding implementation for LlamaIndex.
        """
        if not query or not query.strip():
            return self._handle_empty_text(query)

        _core_ctx, op_ctx_dict, framework_ctx = self._build_contexts(
            llamaindex_context=kwargs,
            **kwargs,
        )

        with _embedding_error_context(
            operation="query_async",
            text_len=len(query),
            framework_ctx=framework_ctx,
            model_name=self.model_name,
        ):
            translated = await self._translator.arun_embed(
                raw_texts=query,
                op_ctx=op_ctx_dict,
                framework_ctx=framework_ctx,
            )
            return self._coerce_embedding_vector(translated)

    def _get_text_embedding(self, text: str, **kwargs: Any) -> List[float]:
        """
        Sync text embedding implementation for LlamaIndex nodes.
        """
        if not text or not text.strip():
            return self._handle_empty_text(text)

        _core_ctx, op_ctx_dict, framework_ctx = self._build_contexts(
            llamaindex_context=kwargs,
            **kwargs,
        )

        with _embedding_error_context(
            operation="text",
            text_len=len(text),
            framework_ctx=framework_ctx,
            model_name=self.model_name,
        ):
            translated = self._translator.embed(
                raw_texts=text,
                op_ctx=op_ctx_dict,
                framework_ctx=framework_ctx,
            )
            return self._coerce_embedding_vector(translated)

    async def _aget_text_embedding(self, text: str, **kwargs: Any) -> List[float]:
        """
        Async text embedding implementation for LlamaIndex nodes.
        """
        if not text or not text.strip():
            return self._handle_empty_text(text)

        _core_ctx, op_ctx_dict, framework_ctx = self._build_contexts(
            llamaindex_context=kwargs,
            **kwargs,
        )

        with _embedding_error_context(
            operation="text_async",
            text_len=len(text),
            framework_ctx=framework_ctx,
            model_name=self.model_name,
        ):
            translated = await self._translator.arun_embed(
                raw_texts=text,
                op_ctx=op_ctx_dict,
                framework_ctx=framework_ctx,
            )
            return self._coerce_embedding_vector(translated)

    def _get_text_embeddings(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        """
        Batch text embedding implementation for LlamaIndex nodes.
        """
        self._warn_if_extreme_batch(texts, op_name="_get_text_embeddings")

        non_empty_texts = [t for t in texts if t and t.strip()]
        empty_indices = [i for i, t in enumerate(texts) if not t or not t.strip()]

        if not non_empty_texts:
            dimension = self._get_embedding_dimension
            return [[0.0] * dimension for _ in texts]

        _core_ctx, op_ctx_dict, framework_ctx = self._build_contexts(
            llamaindex_context=kwargs,
            **kwargs,
        )

        with _embedding_error_context(
            operation="texts",
            texts_count=len(texts),
            framework_ctx=framework_ctx,
            model_name=self.model_name,
        ):
            translated = self._translator.embed(
                raw_texts=non_empty_texts,
                op_ctx=op_ctx_dict,
                framework_ctx=framework_ctx,
            )
            embeddings = self._coerce_embedding_matrix(translated)

            if empty_indices:
                dimension = (
                    len(embeddings[0]) if embeddings else self._get_embedding_dimension
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

    async def _aget_text_embeddings(
        self,
        texts: List[str],
        **kwargs: Any,
    ) -> List[List[float]]:
        """
        Async batch text embedding implementation for LlamaIndex nodes.
        """
        self._warn_if_extreme_batch(texts, op_name="_aget_text_embeddings")

        non_empty_texts = [t for t in texts if t and t.strip()]
        empty_indices = [i for i, t in enumerate(texts) if not t or not t.strip()]

        if not non_empty_texts:
            dimension = self._get_embedding_dimension
            return [[0.0] * dimension for _ in texts]

        _core_ctx, op_ctx_dict, framework_ctx = self._build_contexts(
            llamaindex_context=kwargs,
            **kwargs,
        )

        with _embedding_error_context(
            operation="texts_async",
            texts_count=len(texts),
            framework_ctx=framework_ctx,
            model_name=self.model_name,
        ):
            translated = await self._translator.arun_embed(
                raw_texts=non_empty_texts,
                op_ctx=op_ctx_dict,
                framework_ctx=framework_ctx,
            )
            embeddings = self._coerce_embedding_matrix(translated)

            if empty_indices:
                dimension = (
                    len(embeddings[0]) if embeddings else self._get_embedding_dimension
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
# LlamaIndex Settings Integration
# ------------------------------------------------------------------ #


def configure_llamaindex_embeddings(
    corpus_adapter: EmbeddingProtocolV1,
    model_name: str = "corpus-embedding-protocol",
    **kwargs: Any,
) -> CorpusLlamaIndexEmbeddings:
    """
    Configure and return Corpus embeddings for LlamaIndex global settings.
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
    "configure_llamaindex_embeddings",
    "ErrorCodes",
]
