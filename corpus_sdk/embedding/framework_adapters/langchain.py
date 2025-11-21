# corpus_sdk/embedding/framework_adapters/langchain.py
# SPDX-License-Identifier: Apache-2.0

"""
LangChain adapter for Corpus Embedding protocol.

This module exposes Corpus `EmbeddingProtocolV1` implementations as
`langchain_core.embeddings.Embeddings`, with:

- Sync + async embedding for documents and queries
- Context normalization via `context_translation.from_langchain`
- Framework-agnostic orchestration via `EmbeddingTranslator`
- Async → sync bridging handled in the common embedding layer
- Rich error context attachment for observability
- Model selection via framework_ctx / OperationContext attrs

The design mirrors the Corpus LangChain LLM adapter: this is a *thin*,
framework-specific skin over the protocol-first Corpus embedding stack.
"""

from __future__ import annotations

import logging
from functools import wraps
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
)

from pydantic import BaseModel, ConfigDict, PrivateAttr
from langchain_core.embeddings import Embeddings

from corpus_sdk.core.context_translation import (
    from_langchain as context_from_langchain,
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

# Error code constants
class ErrorCodes:
    INVALID_EMBEDDING_RESULT = "INVALID_EMBEDDING_RESULT"
    EMPTY_EMBEDDING_RESULT = "EMPTY_EMBEDDING_RESULT"
    EMBEDDING_CONVERSION_ERROR = "EMBEDDING_CONVERSION_ERROR"


def with_embedding_error_context(
    operation: str,
    **context_kwargs: Any,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to automatically attach error context to embedding exceptions.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                enhanced_context = context_kwargs.copy()
                attach_context(
                    exc,
                    framework="langchain",
                    operation=f"embedding_{operation}",
                    **enhanced_context,
                )
                raise
        return wrapper
    return decorator


def with_async_embedding_error_context(
    operation: str,
    **context_kwargs: Any,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to automatically attach error context to async embedding exceptions.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return await func(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                enhanced_context = context_kwargs.copy()
                attach_context(
                    exc,
                    framework="langchain",
                    operation=f"embedding_{operation}",
                    **enhanced_context,
                )
                raise
        return wrapper
    return decorator


class CorpusLangChainEmbeddings(BaseModel, Embeddings):
    """
    LangChain `Embeddings` backed by a Corpus `EmbeddingProtocolV1` adapter.

    Inherits from `BaseModel` to support Pydantic-style initialization (standard
    in LangChain) and `Embeddings` to satisfy the interface contract.

    Attributes
    ----------
    corpus_adapter:
        Underlying Corpus embedding adapter implementing `EmbeddingProtocolV1`.

    model:
        Optional default model identifier. Can be overridden per call by
        passing `model=...` to `embed_documents` / `embed_query` or their
        async variants.

    batch_config:
        Optional `BatchConfig` to control batching behavior. If None, the
        defaults in the common embedding layer are used.

    text_normalization_config:
        Optional `TextNormalizationConfig` to control whitespace cleanup,
        truncation, casing, encoding, etc.
    """

    corpus_adapter: EmbeddingProtocolV1
    model: Optional[str] = None
    batch_config: Optional[BatchConfig] = None
    text_normalization_config: Optional[TextNormalizationConfig] = None

    # Pydantic v2 configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Private attribute for caching the translator instance
    _translator_cache: Optional[EmbeddingTranslator] = PrivateAttr(default=None)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @property
    def _translator(self) -> EmbeddingTranslator:
        """
        Lazily construct and cache the `EmbeddingTranslator`.

        Uses a PrivateAttr cache so this remains compatible with Pydantic v2.
        """
        if self._translator_cache is None:
            self._translator_cache = create_embedding_translator(
                adapter=self.corpus_adapter,
                framework="langchain",
                translator=None,  # use registry/default generic translator
                batch_config=self.batch_config,
                text_normalization_config=self.text_normalization_config,
            )
        return self._translator_cache

    def _build_contexts(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
        model: Optional[str] = None,
    ) -> Tuple[OperationContext, Dict[str, Any], Dict[str, Any]]:
        """
        Build:
        - `core_ctx`: core OperationContext (from context_translation)
        - `op_ctx_dict`: normalized dict version of core_ctx (for embedding layer)
        - `framework_ctx`: dict passed to the translator for LangChain-specific hints
        """
        core_ctx: OperationContext = context_from_langchain(config)

        # Normalized dict for embedding OperationContext reconstruction.
        op_ctx_dict: Dict[str, Any] = {}
        if hasattr(core_ctx, "to_dict"):
            op_ctx_dict = core_ctx.to_dict()
        elif hasattr(core_ctx, "__dict__"):
            op_ctx_dict = core_ctx.__dict__

        # Framework-level context: currently we mainly care about a model hint.
        framework_ctx: Dict[str, Any] = {
            "framework": "langchain",
        }
        effective_model = model or self.model
        if effective_model:
            framework_ctx["model"] = effective_model

        return core_ctx, op_ctx_dict, framework_ctx

    def _coerce_embedding_matrix(self, result: Any) -> List[List[float]]:
        """
        Coerce translator result into a List[List[float]] embedding matrix.

        Expected shapes supported:
        - Default translator:
            {"embeddings": [[...], [...]], "model": "...", "usage": {...}}
        - Direct matrix:
            [[...], [...]]
        - EmbedResult-like with `.embeddings` attribute:
            result.embeddings -> [[...], [...]]

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

    def _coerce_embedding_vector(self, result: Any) -> List[float]:
        """
        Coerce translator result for a single-text embed into List[float].

        Strategy:
        - If the matrix is empty → raise
        - If it has exactly one row → return that row
        - If it has multiple rows → return the first row and log a warning
        """
        matrix = self._coerce_embedding_matrix(result)

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

    def _warn_if_extreme_batch(
        self,
        texts: Sequence[str],
        *,
        op_name: str,
    ) -> None:
        """
        Soft warning for extremely large batches when no batch_config limit
        is configured. Actual batching / chunking is handled by the translator.
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
    # Async API
    # ------------------------------------------------------------------ #

    @with_async_embedding_error_context("documents")
    async def aembed_documents(
        self,
        texts: List[str],
        *,
        config: Optional[Mapping[str, Any]] = None,
        model: Optional[str] = None,
        **_: Any,
    ) -> List[List[float]]:
        """
        Async embedding for multiple documents.

        Parameters
        ----------
        texts:
            List of documents to embed.
        config:
            Optional LangChain RunnableConfig-like dict. Used only for
            context translation (request_id, tenant, deadline, tags, etc.).
        model:
            Optional per-call model override.
        """
        self._warn_if_extreme_batch(texts, op_name="aembed_documents")

        _core_ctx, op_ctx_dict, framework_ctx = self._build_contexts(
            config=config,
            model=model,
        )

        translated = await self._translator.arun_embed(
            raw_texts=texts,
            op_ctx=op_ctx_dict,
            framework_ctx=framework_ctx,
        )
        return self._coerce_embedding_matrix(translated)

    @with_async_embedding_error_context("query")
    async def aembed_query(
        self,
        text: str,
        *,
        config: Optional[Mapping[str, Any]] = None,
        model: Optional[str] = None,
        **_: Any,
    ) -> List[float]:
        """
        Async embedding for a single query.

        Parameters
        ----------
        text:
            Query text to embed.
        config:
            Optional LangChain RunnableConfig-like dict.
        model:
            Optional per-call model override.
        """
        _core_ctx, op_ctx_dict, framework_ctx = self._build_contexts(
            config=config,
            model=model,
        )

        translated = await self._translator.arun_embed(
            raw_texts=text,
            op_ctx=op_ctx_dict,
            framework_ctx=framework_ctx,
        )
        return self._coerce_embedding_vector(translated)

    # ------------------------------------------------------------------ #
    # Sync API
    # ------------------------------------------------------------------ #

    @with_embedding_error_context("documents")
    def embed_documents(
        self,
        texts: List[str],
        *,
        config: Optional[Mapping[str, Any]] = None,
        model: Optional[str] = None,
        **_: Any,
    ) -> List[List[float]]:
        """
        Sync embedding for multiple documents.

        Uses the synchronous `EmbeddingTranslator.embed` API, which internally
        bridges async protocol calls and respects any `deadline_ms` timeout
        encoded in the OperationContext.
        """
        self._warn_if_extreme_batch(texts, op_name="embed_documents")

        _core_ctx, op_ctx_dict, framework_ctx = self._build_contexts(
            config=config,
            model=model,
        )

        translated = self._translator.embed(
            raw_texts=texts,
            op_ctx=op_ctx_dict,
            framework_ctx=framework_ctx,
        )
        return self._coerce_embedding_matrix(translated)

    @with_embedding_error_context("query")
    def embed_query(
        self,
        text: str,
        *,
        config: Optional[Mapping[str, Any]] = None,
        model: Optional[str] = None,
        **_: Any,
    ) -> List[float]:
        """
        Sync embedding for a single query.

        Uses the synchronous `EmbeddingTranslator.embed` API, which internally
        bridges async protocol calls and respects any `deadline_ms` timeout
        encoded in the OperationContext.
        """
        _core_ctx, op_ctx_dict, framework_ctx = self._build_contexts(
            config=config,
            model=model,
        )

        translated = self._translator.embed(
            raw_texts=text,
            op_ctx=op_ctx_dict,
            framework_ctx=framework_ctx,
        )
        return self._coerce_embedding_vector(translated)


__all__ = [
    "CorpusLangChainEmbeddings",
    "ErrorCodes",
]
