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
from functools import cached_property
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
)

from corpus_sdk.core.context_translation import (
    from_autogen as context_from_autogen,  # Using existing implementation
)
from corpus_sdk.embedding.embedding_base import (
    EmbeddingProtocolV1,
)
from corpus_sdk.embedding.framework_adapters.common.embedding_translation import (
    EmbeddingTranslator,
    BatchConfig,
    TextNormalizationConfig,
    create_embedding_translator,
)
from corpus_sdk.llm.framework_adapters.common.error_context import attach_context

logger = logging.getLogger(__name__)


class AutoGenVectorStoreRetrieverProtocol(Protocol):
    """
    Protocol representing the minimal AutoGen VectorStoreRetriever interface
    used by this module.

    The concrete implementation is typically `autogen.retrieve_utils.VectorStoreRetriever`,
    but this structural protocol avoids a hard dependency at type-check time while
    still providing strong typing for the helper function return type.
    """

    @property
    def vectorstore(self) -> Any:
        """
        Underlying vector store used for retrieval.

        The concrete type is AutoGen- and application-specific, so it is left
        as `Any` here to keep the protocol flexible.
        """
        ...

    def retrieve(self, query: str, **kwargs: Any) -> Any:
        """
        Retrieve documents for the given query.

        Implementations may return framework-specific document or node types;
        callers in this module do not rely on the concrete return type.
        """
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
    ) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
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
        core_ctx = context_from_autogen(autogen_context)

        # Normalized dict for embedding OperationContext reconstruction
        op_ctx_dict: Dict[str, Any] = core_ctx.to_dict()

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

    @staticmethod
    def _coerce_embedding_matrix(result: Any) -> List[List[float]]:
        """
        Coerce translator result into a List[List[float]] embedding matrix.

        Supports the same result formats as other adapters:
        - {"embeddings": [[...], [...]], "model": "...", "usage": {...}}
        - Direct matrix: [[...], [...]]
        - EmbedResult-like with `.embeddings` attribute

        This ensures consistency across all framework adapters.
        """
        embeddings_obj: Any

        match result:
            case {"embeddings": emb}:
                embeddings_obj = emb
            case _ if hasattr(result, "embeddings"):
                embeddings_obj = getattr(result, "embeddings")
            case _:
                embeddings_obj = result

        if not isinstance(embeddings_obj, Sequence):
            raise TypeError(
                f"Translator result does not contain a valid embeddings sequence: "
                f"type={type(embeddings_obj).__name__}"
            )

        matrix: List[List[float]] = []
        for i, row in enumerate(embeddings_obj):
            if not isinstance(row, Sequence):
                raise TypeError(
                    f"Expected each embedding row to be a sequence, "
                    f"got {type(row).__name__} at index {i}"
                )
            try:
                matrix.append([float(x) for x in row])
            except (TypeError, ValueError) as e:
                raise TypeError(
                    f"Failed to convert embedding values to float at row {i}: {e}"
                ) from e

        return matrix

    @staticmethod
    def _coerce_embedding_vector(result: Any) -> List[float]:
        """
        Coerce translator result for a single-text embed into List[float].

        Normalizes via `_coerce_embedding_matrix` and handles single/multiple rows.
        """
        matrix = CorpusAutoGenEmbeddings._coerce_embedding_matrix(result)

        if not matrix:
            raise ValueError("Translator returned no embeddings for single-text input")

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
        core_ctx, op_ctx_dict, framework_ctx = self._build_contexts(
            autogen_context=autogen_context,
            model=model,
            **kwargs,
        )

        try:
            translated = self._translator.embed(
                raw_texts=texts,
                op_ctx=op_ctx_dict,
                framework_ctx=framework_ctx,
            )
            return self._coerce_embedding_matrix(translated)
        except Exception as exc:  # noqa: BLE001
            # Enrich with AutoGen-specific context; core protocol context is
            # already attached inside EmbeddingTranslator.
            try:
                attach_context(
                    exc,
                    embedding_operation="embed_documents",
                    texts_count=len(texts),
                    agent_name=framework_ctx.get("agent_name"),
                    conversation_id=framework_ctx.get("conversation_id"),
                    workflow_type=framework_ctx.get("workflow_type"),
                    retriever_name=framework_ctx.get("retriever_name"),
                )
            except Exception:
                # Never mask the original error
                pass
            raise

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

        try:
            translated = self._translator.embed(
                raw_texts=text,
                op_ctx=op_ctx_dict,
                framework_ctx=framework_ctx,
            )
            return self._coerce_embedding_vector(translated)
        except Exception as exc:  # noqa: BLE001
            # Enrich with AutoGen-specific context; core protocol context is
            # already attached inside EmbeddingTranslator.
            try:
                attach_context(
                    exc,
                    embedding_operation="embed_query",
                    text_len=len(text or ""),
                    agent_name=framework_ctx.get("agent_name"),
                    conversation_id=framework_ctx.get("conversation_id"),
                    workflow_type=framework_ctx.get("workflow_type"),
                    retriever_name=framework_ctx.get("retriever_name"),
                )
            except Exception:
                # Never mask the original error
                pass
            raise

    # ------------------------------------------------------------------ #
    # Async API for AutoGen Async Workflows
    # ------------------------------------------------------------------ #

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
        core_ctx, op_ctx_dict, framework_ctx = self._build_contexts(
            autogen_context=autogen_context,
            model=model,
            **kwargs,
        )

        try:
            translated = await self._translator.arun_embed(
                raw_texts=texts,
                op_ctx=op_ctx_dict,
                framework_ctx=framework_ctx,
            )
            return self._coerce_embedding_matrix(translated)
        except Exception as exc:  # noqa: BLE001
            # Enrich with AutoGen-specific context for async workflows.
            try:
                attach_context(
                    exc,
                    embedding_operation="aembed_documents",
                    texts_count=len(texts),
                    agent_name=framework_ctx.get("agent_name"),
                    conversation_id=framework_ctx.get("conversation_id"),
                    workflow_type=framework_ctx.get("workflow_type"),
                    retriever_name=framework_ctx.get("retriever_name"),
                )
            except Exception:
                # Never mask the original error
                pass
            raise

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

        try:
            translated = await self._translator.arun_embed(
                raw_texts=text,
                op_ctx=op_ctx_dict,
                framework_ctx=framework_ctx,
            )
            return self._coerce_embedding_vector(translated)
        except Exception as exc:  # noqa: BLE001
            # Enrich with AutoGen-specific context for async query embeddings.
            try:
                attach_context(
                    exc,
                    embedding_operation="aembed_query",
                    text_len=len(text or ""),
                    agent_name=framework_ctx.get("agent_name"),
                    conversation_id=framework_ctx.get("conversation_id"),
                    workflow_type=framework_ctx.get("workflow_type"),
                    retriever_name=framework_ctx.get("retriever_name"),
                )
            except Exception:
                # Never mask the original error
                pass
            raise


# ------------------------------------------------------------------ #
# AutoGen-Specific Helper Functions
# ------------------------------------------------------------------ #

def create_autogen_retriever(
    corpus_adapter: EmbeddingProtocolV1,
    vector_store: Any,
    **kwargs: Any,
) -> AutoGenVectorStoreRetrieverProtocol:
    """
    Create an AutoGen VectorStoreRetriever with Corpus embeddings.

    This provides a convenient way to create AutoGen retrievers
    with Corpus embeddings in a single function call.

    Example usage:
    ```python
    from autogen.retrieve_utils import create_autogen_retriever
    from chromadb import Chroma

    # Create vector store
    vectorstore = Chroma(collection_name="autogen_docs")

    # Create retriever with Corpus embeddings
    retriever = create_autogen_retriever(
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
    AutoGenVectorStoreRetrieverProtocol
        AutoGen VectorStoreRetriever instance using Corpus embeddings.
    """
    try:
        from autogen.retrieve_utils import VectorStoreRetriever
    except ImportError:
        logger.error(
            "AutoGen not installed. Please install with: pip install pyautogen"
        )
        raise

    embedding_function = CorpusAutoGenEmbeddings(
        corpus_adapter=corpus_adapter,
        **kwargs,
    )

    # Configure the vector store with our embedding function
    if hasattr(vector_store, '_embedding_function'):
        vector_store._embedding_function = embedding_function
    elif hasattr(vector_store, 'embedding_function'):
        vector_store.embedding_function = embedding_function

    retriever = VectorStoreRetriever(vectorstore=vector_store)

    logger.info(
        f"AutoGen retriever created with Corpus embeddings"
    )

    return retriever


def register_autogen_embedding(
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
    from corpus_sdk.embedding.framework_adapters.autogen import register_autogen_embedding

    # Register globally
    embedder = register_autogen_embedding(
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
    "create_autogen_retriever",
    "register_autogen_embedding",
]
