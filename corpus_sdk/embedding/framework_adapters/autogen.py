# corpus_sdk/embedding/framework_adapters/autogen.py
# SPDX-License-Identifier: Apache-2.0

"""
AutoGen adapter for Corpus Embedding protocol.

This module exposes Corpus `EmbeddingProtocolV1` implementations for
use with Microsoft AutoGen multi-agent conversations, with:

- Full compatibility with AutoGen's `EmbeddingFunction` protocol
- Support for AutoGen's group chat and agent memory systems
- Context normalization using existing `context_translation.from_autogen`
- Framework-agnostic orchestration via `EmbeddingTranslator`
- Async → sync bridging using `AsyncBridge`
- Rich error context attachment for observability

The design integrates seamlessly with AutoGen's agent workflows while
maintaining the protocol-first Corpus embedding stack.
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
    Union,
    Protocol,
    TypeVar,
    Callable,
    cast,
)

from corpus_sdk.core.context_translation import (
    from_autogen as context_from_autogen,
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
                    framework="autogen",
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
                    framework="autogen",
                    operation=f"embedding_{operation}",
                    **enhanced_context,
                )
                raise

        return wrapper

    return decorator


class AutoGenRetriever(Protocol):
    """
    Protocol representing AutoGen VectorStoreRetriever interface.
    
    Shorter name for cleaner usage throughout the module.
    """

    @property
    def vectorstore(self) -> Any:
        """Underlying vector store used for retrieval."""
        ...

    def retrieve(self, query: str, **kwargs: Any) -> Any:
        """Retrieve documents for the given query."""
        ...


class CorpusAutoGenEmbeddings:
    """
    AutoGen embedding function backed by a Corpus `EmbeddingProtocolV1` adapter.

    AutoGen-Specific Responsibilities
    ---------------------------------
    - Implement AutoGen's `EmbeddingFunction` protocol for vector stores
    - Support AutoGen agent memory and retrieval-augmented generation
    - Integrate with AutoGen's group chat and multi-agent workflows
    - Provide embeddings for agent context and document retrieval
    - Work with AutoGen's `VectorStoreRetriever` and custom retrievers

    Non-responsibilities
    --------------------
    - Agent orchestration and conversation management (handled by AutoGen)
    - Retrieval logic and similarity search (handled by AutoGen retrievers)
    - Multi-agent communication patterns (handled by AutoGen group chats)

    All embedding logic lives in:
    - `corpus_sdk.embedding.framework_adapters.common.embedding_translation`
    - Concrete `EmbeddingProtocolV1` adapter implementations.

    Attributes
    ----------
    corpus_adapter:
        Underlying Corpus embedding adapter implementing `EmbeddingProtocolV1`.

    model:
        Optional model identifier used in AutoGen contexts.

    batch_config:
        Optional `BatchConfig` to control batching behavior.

    text_normalization_config:
        Optional `TextNormalization_config` to control whitespace cleanup,
        truncation, casing, encoding, etc.

    autogen_config:
        Optional AutoGen-specific configuration for agent context
        and workflow integration.
    """

    def __init__(
        self,
        corpus_adapter: EmbeddingProtocolV1,
        model: Optional[str] = None,
        batch_config: Optional[BatchConfig] = None,
        text_normalization_config: Optional[TextNormalizationConfig] = None,
        autogen_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Corpus AutoGen Embeddings.

        Parameters
        ----------
        corpus_adapter:
            Corpus embedding protocol adapter
        model:
            Model identifier for embedding operations
        batch_config:
            Batching configuration for embedding requests
        text_normalization_config:
            Text normalization settings
        autogen_config:
            AutoGen-specific configuration for agent workflows
        """
        self.corpus_adapter = corpus_adapter
        self.model = model
        self.batch_config = batch_config
        self.text_normalization_config = text_normalization_config
        self.autogen_config = autogen_config or {}

    # ------------------------------------------------------------------ #
    # Core AutoGen EmbeddingFunction Protocol Implementation
    # ------------------------------------------------------------------ #

    @cached_property
    def _translator(self) -> EmbeddingTranslator:
        """
        Lazily construct and cache the `EmbeddingTranslator`.

        Uses `cached_property` for thread safety and performance.
        """
        return create_embedding_translator(
            adapter=self.corpus_adapter,
            framework="autogen",
            translator=None,  # use registry/default generic translator
            batch_config=self.batch_config,
            text_normalization_config=self.text_normalization_config,
        )

    def _build_contexts(
        self,
        *,
        autogen_context: Optional[Mapping[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[OperationContext, Dict[str, Any], Dict[str, Any]]:
        """
        Build contexts for AutoGen execution environment.

        Uses the existing `context_from_autogen` implementation.

        Parameters
        ----------
        autogen_context:
            Optional AutoGen context containing agent information,
            conversation state, and workflow metadata.

        Returns
        -------
        Tuple of:
        - `core_ctx`: core OperationContext (from existing context translation)
        - `op_ctx_dict`: normalized dict for embedding layer
        - `framework_ctx`: AutoGen-specific context for translator
        """
        # Use existing context translation implementation
        core_ctx: OperationContext = context_from_autogen(autogen_context)

        # Normalized dict for embedding OperationContext reconstruction
        op_ctx_dict: Dict[str, Any] = {}
        if hasattr(core_ctx, "to_dict"):
            op_ctx_dict = core_ctx.to_dict()
        elif hasattr(core_ctx, "__dict__"):
            op_ctx_dict = core_ctx.__dict__

        # Framework-level context for AutoGen-specific hints
        framework_ctx: Dict[str, Any] = {
            "framework": "autogen",
        }

        # Add model information if available
        effective_model = model or self.model
        if effective_model:
            framework_ctx["model"] = effective_model

        # Add AutoGen-specific context for agent workflows
        if autogen_context:
            if "agent_name" in autogen_context:
                framework_ctx["agent_name"] = autogen_context["agent_name"]
            if "conversation_id" in autogen_context:
                framework_ctx["conversation_id"] = autogen_context["conversation_id"]
            if "workflow_type" in autogen_context:
                framework_ctx["workflow_type"] = autogen_context["workflow_type"]
            if "retriever_name" in autogen_context:
                framework_ctx["retriever_name"] = autogen_context["retriever_name"]

        # Add any additional kwargs to framework context
        framework_ctx.update(kwargs)

        return core_ctx, op_ctx_dict, framework_ctx

    def _coerce_embedding_matrix(self, result: Any) -> List[List[float]]:
        """
        Coerce translator result into a List[List[float]] embedding matrix.

        Supports the same result formats as other adapters:
        - {"embeddings": [[...], [...]], "model": "...", "usage": {...}}
        - Direct matrix: [[...], [...]]
        - EmbedResult-like with `.embeddings` attribute

        This ensures consistency across all framework adapters.

        Note: Implemented without match/case so this module works on Python 3.9+.
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
                f"Translator result does not contain a valid embeddings sequence: "
                f"type={type(embeddings_obj).__name__}",
                code=ErrorCodes.INVALID_EMBEDDING_RESULT,
            )

        matrix: List[List[float]] = []
        for i, row in enumerate(embeddings_obj):
            if not isinstance(row, Sequence):
                raise TypeError(
                    f"Expected each embedding row to be a sequence, "
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

        Normalizes via `_coerce_embedding_matrix` and handles single/multiple rows.
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

    # ------------------------------------------------------------------ #
    # Core AutoGen EmbeddingFunction Interface
    # ------------------------------------------------------------------ #

    @with_embedding_error_context("function_call")
    def __call__(self, texts: List[str]) -> List[List[float]]:
        """
        Make the instance callable for AutoGen's EmbeddingFunction protocol.

        This enables direct usage with AutoGen's VectorStoreRetriever:
        ```python
        retriever = VectorStoreRetriever(
            vectorstore=Chroma(embedding_function=CorpusAutoGenEmbeddings(...)),
            ...
        )
        ```

        Parameters
        ----------
        texts:
            List of texts to embed

        Returns
        -------
        List[List[float]]
            Batch of text embedding vectors
        """
        return self.embed_documents(texts)

    @with_embedding_error_context("documents")
    def embed_documents(
        self,
        texts: List[str],
        *,
        autogen_context: Optional[Mapping[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """
        Sync embedding for multiple documents.

        This is the primary method used by AutoGen's retrieval systems
        for document embedding and agent memory.

        Parameters
        ----------
        texts:
            List of documents to embed
        autogen_context:
            Optional AutoGen context containing agent and conversation info
        model:
            Optional per-call model override
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

        core_ctx, op_ctx_dict, framework_ctx = self._build_contexts(
            autogen_context=autogen_context,
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
        autogen_context: Optional[Mapping[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[float]:
        """
        Sync embedding for a single query.

        Used by AutoGen for query understanding and retrieval in
        multi-agent conversations.

        Parameters
        ----------
        text:
            Query text to embed
        autogen_context:
            Optional AutoGen context
        model:
            Optional per-call model override
        """
        core_ctx, op_ctx_dict, framework_ctx = self._build_contexts(
            autogen_context=autogen_context,
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
    # Async API for AutoGen Async Workflows
    # ------------------------------------------------------------------ #

    @with_async_embedding_error_context("documents")
    async def aembed_documents(
        self,
        texts: List[str],
        *,
        autogen_context: Optional[Mapping[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """
        Async embedding for multiple documents.

        Designed for use with AutoGen's async workflows and
        event-driven agent systems.

        Parameters
        ----------
        texts:
            List of documents to embed
        autogen_context:
            Optional AutoGen context
        model:
            Optional per-call model override
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

        core_ctx, op_ctx_dict, framework_ctx = self._build_contexts(
            autogen_context=autogen_context,
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
        autogen_context: Optional[Mapping[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[float]:
        """
        Async embedding for a single query.

        Used in AutoGen's asynchronous agent workflows and
        flow-based conversation systems.

        Parameters
        ----------
        text:
            Query text to embed
        autogen_context:
            Optional AutoGen context
        model:
            Optional per-call model override
        """
        core_ctx, op_ctx_dict, framework_ctx = self._build_contexts(
            autogen_context=autogen_context,
            model=model,
            **kwargs,
        )

        translated = await self._translator.arun_embed(
            raw_texts=text,
            op_ctx=op_ctx_dict,
            framework_ctx=framework_ctx,
        )
        return self._coerce_embedding_vector(translated)


# ------------------------------------------------------------------ #
# AutoGen-Specific Helper Functions
# ------------------------------------------------------------------ #

def _validate_vector_store_for_autogen(vector_store: Any) -> None:
    """
    Best-effort validation that the provided vector_store looks like a
    typical vector store object AutoGen expects.

    We keep this intentionally loose to avoid over-constraining users:
      - Accepts stores that expose common vector-store methods like
        `similarity_search` or `query`.
    """
    if vector_store is None:
        raise TypeError("vector_store must not be None")

    has_similarity = hasattr(vector_store, "similarity_search")
    has_query = hasattr(vector_store, "query")

    if not (has_similarity or has_query):
        logger.warning(
            "Vector store %r does not expose common methods like 'similarity_search' "
            "or 'query'. It may not be compatible with AutoGen's VectorStoreRetriever.",
            type(vector_store).__name__,
        )


def create_retriever(
    corpus_adapter: EmbeddingProtocolV1,
    vector_store: Any,
    **kwargs: Any,
) -> AutoGenRetriever:
    """
    Create an AutoGen VectorStoreRetriever with Corpus embeddings.

    This provides a convenient way to create AutoGen retrievers
    with Corpus embeddings in a single function call.

    Example usage:
    ```python
    from corpus_sdk.embedding.framework_adapters.autogen import create_retriever
    from chromadb import Chroma

    # Create vector store
    vectorstore = Chroma(collection_name="autogen_docs")

    # Create retriever with Corpus embeddings
    retriever = create_retriever(
        corpus_adapter=my_adapter,
        vector_store=vectorstore,
        model="text-embedding-3-large"
    )

    # Use with AutoGen agent
    agent = AssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant.",
        retrieve_config={
            "retriever": retriever,
            "config_list": config_list,
        },
    )
    ```

    Parameters
    ----------
    corpus_adapter:
        Corpus embedding protocol adapter
    vector_store:
        Vector store instance (Chroma, FAISS, etc.)
    **kwargs:
        Additional arguments for CorpusAutoGenEmbeddings

    Returns
    -------
    AutoGenRetriever
        AutoGen VectorStoreRetriever instance using Corpus embeddings.
    """
    try:
        from autogen.retrieve_utils import VectorStoreRetriever
    except ImportError:
        logger.error(
            "AutoGen not installed. Please install with: pip install pyautogen"
        )
        raise

    # Best-effort validation before mutating the vector store.
    _validate_vector_store_for_autogen(vector_store)

    embedding_function = CorpusAutoGenEmbeddings(
        corpus_adapter=corpus_adapter,
        **kwargs,
    )

    # Configure the vector store with our embedding function.
    # Prefer public attribute; fall back to private `_embedding_function`
    # for libraries that don't expose a clean setter.
    if hasattr(vector_store, "embedding_function"):
        setattr(vector_store, "embedding_function", embedding_function)
    elif hasattr(vector_store, "_embedding_function"):
        logger.debug(
            "Setting private attribute '_embedding_function' on vector_store %r. "
            "If the vector store library changes its internals, this may break.",
            type(vector_store).__name__,
        )
        setattr(vector_store, "_embedding_function", embedding_function)
    else:
        logger.warning(
            "Vector store %r does not expose an 'embedding_function' or "
            "'_embedding_function' attribute. You may need to configure the "
            "embedding function manually.",
            type(vector_store).__name__,
        )

    retriever = VectorStoreRetriever(vectorstore=vector_store)

    logger.info(
        "AutoGen retriever created with Corpus embeddings"
    )

    return retriever


def register_embeddings(
    corpus_adapter: EmbeddingProtocolV1,
    model: Optional[str] = None,
    **kwargs: Any,
) -> CorpusAutoGenEmbeddings:
    """
    Register Corpus embeddings for global use in AutoGen workflows.

    This function provides a centralized way to configure Corpus embeddings
    for multiple AutoGen agents and retrievers.

    Example usage:
    ```python
    from corpus_sdk.embedding.framework_adapters.autogen import register_embeddings

    # Register globally
    embedder = register_embeddings(
        corpus_adapter=my_adapter,
        model="text-embedding-3-large"
    )

    # Use across multiple agents and retrievers
    ```

    Parameters
    ----------
    corpus_adapter:
        Corpus embedding protocol adapter
    model:
        Model identifier for embedding operations
    **kwargs:
        Additional arguments for CorpusAutoGenEmbeddings

    Returns
    -------
    CorpusAutoGenEmbeddings
        Configured embedding function for AutoGen
    """
    embeddings = CorpusAutoGenEmbeddings(
        corpus_adapter=corpus_adapter,
        model=model,
        **kwargs,
    )

    logger.info(
        f"Corpus AutoGen embeddings registered: {model or 'default model'}"
    )

    return embeddings


__all__ = [
    "CorpusAutoGenEmbeddings",
    "AutoGenRetriever",
    "create_retriever",
    "register_embeddings",
    "ErrorCodes",
]
