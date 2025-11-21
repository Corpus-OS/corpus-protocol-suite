# corpus_sdk/embedding/framework_adapters/crewai.py
# SPDX-License-Identifier: Apache-2.0

"""
CrewAI adapter for Corpus Embedding protocol.

This module exposes Corpus `EmbeddingProtocolV1` implementations as
embedding services within CrewAI agents and workflows, with:

- Seamless integration with CrewAI agent `embedder` attribute
- Support for CrewAI knowledge sources and RAG workflows
- Context normalization for CrewAI-specific execution context
- Framework-agnostic orchestration via `EmbeddingTranslator`
- Async → sync bridging using `AsyncBridge`
- Rich error context attachment for observability

The design follows CrewAI's adapter patterns while maintaining the
protocol-first Corpus embedding stack.
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
    Protocol,
    TypeVar,
    Callable,
)

from corpus_sdk.core.context_translation import (
    from_crewai as context_from_crewai,
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
            except Exception as exc:
                enhanced_context = context_kwargs.copy()
                attach_context(
                    exc,
                    framework="crewai",
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
            except Exception as exc:
                enhanced_context = context_kwargs.copy()
                attach_context(
                    exc,
                    framework="crewai",
                    operation=f"embedding_{operation}",
                    **enhanced_context,
                )
                raise
        return wrapper
    return decorator


class CrewAIEmbedder(Protocol):
    """
    Protocol representing the embedder interface expected by CrewAI agents.

    This allows type-safe integration with CrewAI's agent embedder system
    without requiring a hard dependency on CrewAI at type-check time.
    """

    def embed_documents(
        self,
        texts: List[str],
        *,
        crewai_context: Optional[Mapping[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """Embed multiple documents for CrewAI RAG workflows."""
        ...

    def embed_query(
        self,
        text: str,
        *,
        crewai_context: Optional[Mapping[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[float]:
        """Embed a single query for CrewAI retrieval."""
        ...


class CorpusCrewAIEmbeddings:
    """
    CrewAI embedding service backed by a Corpus `EmbeddingProtocolV1` adapter.

    This class implements the CrewAI embedder interface and can be directly
    assigned to CrewAI agents via the `embedder` attribute.

    Example:
    ```python
    from crewai import Agent
    from corpus_sdk.embedding.framework_adapters.crewai import create_embedder

    embedder = create_embedder(corpus_adapter=my_adapter)
    agent = Agent(
        role="Researcher",
        goal="Research AI developments",
        backstory="Expert analyst",
        embedder=embedder  # Direct assignment
    )
    ```

    Attributes
    ----------
    corpus_adapter: Underlying Corpus embedding protocol adapter
    model: Optional default model identifier
    batch_config: Optional batching configuration
    text_normalization_config: Optional text normalization settings
    crewai_config: Optional CrewAI-specific configuration
    """

    def __init__(
        self,
        corpus_adapter: EmbeddingProtocolV1,
        model: Optional[str] = None,
        batch_config: Optional[BatchConfig] = None,
        text_normalization_config: Optional[TextNormalizationConfig] = None,
        crewai_config: Optional[Dict[str, Any]] = None,
    ):
        self.corpus_adapter = corpus_adapter
        self.model = model
        self.batch_config = batch_config
        self.text_normalization_config = text_normalization_config
        self.crewai_config = crewai_config or {}

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @cached_property
    def _translator(self) -> EmbeddingTranslator:
        """
        Lazily construct and cache the `EmbeddingTranslator`.
        """
        return create_embedding_translator(
            adapter=self.corpus_adapter,
            framework="crewai",
            translator=None,
            batch_config=self.batch_config,
            text_normalization_config=self.text_normalization_config,
        )

    def _build_contexts(
        self,
        *,
        crewai_context: Optional[Mapping[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[OperationContext, Dict[str, Any], Dict[str, Any]]:
        """
        Build contexts for CrewAI execution environment.
        """
        # Convert CrewAI context to core OperationContext
        core_ctx: OperationContext = context_from_crewai(c

rewai_context)

        # Normalized dict for embedding OperationContext
        op_ctx_dict: Dict[str, Any] = {}
        if hasattr(core_ctx, "to_dict"):
            op_ctx_dict = core_ctx.to_dict()
        elif hasattr(core_ctx, "__dict__"):
            op_ctx_dict = core_ctx.__dict__

        # Framework-level context for CrewAI-specific hints
        framework_ctx: Dict[str, Any] = {
            "framework": "crewai",
        }

        effective_model = model or self.model
        if effective_model:
            framework_ctx["model"] = effective_model

        # Add CrewAI-specific context
        if crewai_context:
            framework_ctx["agent_role"] = crewai_context.get("agent_role")
            framework_ctx["task_id"] = crewai_context.get("task_id")
            framework_ctx["workflow"] = crewai_context.get("workflow")

        framework_ctx.update(kwargs)
        return core_ctx, op_ctx_dict, framework_ctx

    def _coerce_embedding_matrix(self, result: Any) -> List[List[float]]:
        """
        Coerce translator result into embedding matrix.

        Supports:
        - {"embeddings": [[...], [...]], "model": "...", "usage": {...}}
        - Direct matrix: [[...], [...]]
        - EmbedResult-like with `.embeddings` attribute

        Implemented without `match`/`case` so this module works on Python 3.9+.
        """
        embeddings_obj: Any

        if isinstance(result, Mapping) and "embeddings" in result:
            embeddings_obj = result["embeddings"]
        elif hasattr(result, "embeddings"):
            embeddings_obj = getattr(result, "embeddings")
        else:
            embeddings_obj = result

        if not isinstance(embeddings_obj, Sequence):
            raise TypeError(
                f"Translator result does not contain valid embeddings sequence: "
                f"type={type(embeddings_obj).__name__}",
                code=ErrorCodes.INVALID_EMBEDDING_RESULT,
            )

        matrix: List[List[float]] = []
        for i, row in enumerate(embeddings_obj):
            if not isinstance(row, Sequence):
                raise TypeError(
                    f"Expected embedding row to be sequence, got {type(row).__name__} at index {i}",
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
        Coerce translator result for single-text embed.
        """
        matrix = self._coerce_embedding_matrix(result)

        if not matrix:
            raise ValueError(
                "Translator returned no embeddings for single-text input",
                code=ErrorCodes.EMPTY_EMBEDDING_RESULT,
            )

        if len(matrix) > 1:
            logger.warning(
                "Expected single embedding for query, got %d rows; using first row.",
                len(matrix),
            )

        return matrix[0]

    # ------------------------------------------------------------------ #
    # Core Embedding API (CrewAI Compatible)
    # ------------------------------------------------------------------ #

    @with_embedding_error_context("documents")
    def embed_documents(
        self,
        texts: List[str],
        *,
        crewai_context: Optional[Mapping[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """
        Sync embedding for multiple documents.

        Used by CrewAI agents during RAG operations and knowledge processing.
        """
        # Soft warning for very large batches when no batch_config is provided.
        if isinstance(texts, Sequence) and not isinstance(texts, (str, bytes)):
            batch_size = len(texts)
            if (
                batch_size > 10_000
                and (self.batch_config is None or getattr(self.batch_config, "max_batch_size", None) is None)
            ):
                logger.warning(
                    "embed_documents called with batch_size=%d and no batch_config.max_batch_size; "
                    "ensure your adapter/translator can safely handle large batches.",
                    batch_size,
                )

        _, op_ctx_dict, framework_ctx = self._build_contexts(
            crewai_context=crewai_context,
            model=model,
            **kwargs,
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
        crewai_context: Optional[Mapping[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[float]:
        """
        Sync embedding for a single query.

        Used by CrewAI for query understanding and retrieval.
        """
        _, op_ctx_dict, framework_ctx = self._build_contexts(
            crewai_context=crewai_context,
            model=model,
            **kwargs,
        )

        translated = self._translator.embed(
            raw_texts=text,
            op_ctx=op_ctx_dict,
            framework_ctx=framework_ctx,
        )
        return self._coerce_embedding_vector(translated)

    # ------------------------------------------------------------------ #
    # Async API for CrewAI Flows
    # ------------------------------------------------------------------ #

    @with_async_embedding_error_context("documents")
    async def aembed_documents(
        self,
        texts: List[str],
        *,
        crewai_context: Optional[Mapping[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """
        Async embedding for multiple documents.
        """
        # Soft warning for very large batches when no batch_config is provided.
        if isinstance(texts, Sequence) and not isinstance(texts, (str, bytes)):
            batch_size = len(texts)
            if (
                batch_size > 10_000
                and (self.batch_config is None or getattr(self.batch_config, "max_batch_size", None) is None)
            ):
                logger.warning(
                    "aembed_documents called with batch_size=%d and no batch_config.max_batch_size; "
                    "ensure your adapter/translator can safely handle large batches.",
                    batch_size,
                )

        _, op_ctx_dict, framework_ctx = self._build_contexts(
            crewai_context=crewai_context,
            model=model,
            **kwargs,
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
        crewai_context: Optional[Mapping[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[float]:
        """
        Async embedding for a single query.
        """
        _, op_ctx_dict, framework_ctx = self._build_contexts(
            crewai_context=crewai_context,
            model=model,
            **kwargs,
        )

        translated = await self._translator.arun_embed(
            raw_texts=text,
            op_ctx=op_ctx_dict,
            framework_ctx=framework_ctx,
        )
        return self._coerce_embedding_vector(translated)


def create_embedder(
    corpus_adapter: EmbeddingProtocolV1,
    model: Optional[str] = None,
    **kwargs: Any,
) -> CrewAIEmbedder:
    """
    Create a CrewAI-compatible embedder for agent integration.

    Example:
    ```python
    from crewai import Agent
    from corpus_sdk.embedding.framework_adapters.crewai import create_embedder

    embedder = create_embedder(
        corpus_adapter=my_adapter,
        model="text-embedding-3-large"
    )

    agent = Agent(
        role="Researcher",
        goal="Research AI developments", 
        backstory="Expert analyst",
        embedder=embedder,  # Direct assignment
        tools=[...]
    )
    ```

    Parameters
    ----------
    corpus_adapter: Corpus embedding protocol adapter
    model: Model identifier for embedding operations
    **kwargs: Additional arguments for CorpusCrewAIEmbeddings

    Returns
    -------
    CrewAIEmbedder compatible embedder instance
    """
    return CorpusCrewAIEmbeddings(
        corpus_adapter=corpus_adapter,
        model=model,
        **kwargs,
    )


__all__ = [
    "CorpusCrewAIEmbeddings",
    "CrewAIEmbedder",
    "create_embedder",
    "ErrorCodes",
]
