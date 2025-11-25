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
- Async → sync bridging handled in the common embedding layer
- Rich error context attachment for observability

The design integrates seamlessly with AutoGen's agent workflows while
maintaining the protocol-first Corpus embedding stack.

Resilience (retries, caching, rate limiting, etc.) is expected to be provided by the underlying adapter, typically a BaseEmbeddingAdapter subclass.
"""

from __future__ import annotations

import asyncio
import logging
from functools import cached_property, wraps
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    List,
    TypedDict,
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
from corpus_sdk.embedding.framework_adapters.common.framework_utils import (
    CoercionErrorCodes,
    coerce_embedding_matrix,
    coerce_embedding_vector,
    warn_if_extreme_batch,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# Error codes + context types
# --------------------------------------------------------------------------- #


class ErrorCodes:
    """
    Error code constants for AutoGen embedding adapter.

    This is a simple namespace for framework-specific codes. The shared
    coercion helpers use `EMBEDDING_COERCION_ERROR_CODES`, which is a
    `CoercionErrorCodes` instance derived from these values.
    """

    # Coercion-level (used by framework_utils)
    INVALID_EMBEDDING_RESULT = "INVALID_EMBEDDING_RESULT"
    EMPTY_EMBEDDING_RESULT = "EMPTY_EMBEDDING_RESULT"
    EMBEDDING_CONVERSION_ERROR = "EMBEDDING_CONVERSION_ERROR"

    # AutoGen-specific context errors
    AUTOGEN_CONTEXT_INVALID = "AUTOGEN_CONTEXT_INVALID"


# Coercion configuration for the common embedding utils
EMBEDDING_COERCION_ERROR_CODES: CoercionErrorCodes = CoercionErrorCodes(
    invalid_result=ErrorCodes.INVALID_EMBEDDING_RESULT,
    empty_result=ErrorCodes.EMPTY_EMBEDDING_RESULT,
    conversion_error=ErrorCodes.EMBEDDING_CONVERSION_ERROR,
    framework_label="autogen",
)


class AutoGenContext(TypedDict, total=False):
    """Structured type for AutoGen execution context."""
    agent_name: Optional[str]
    conversation_id: Optional[str]
    workflow_type: Optional[str]
    retriever_name: Optional[str]
    request_id: Optional[str]
    user_id: Optional[str]


# --------------------------------------------------------------------------- #
# AutoGen retriever protocol
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Error-context decorators with dynamic context extraction
# --------------------------------------------------------------------------- #


def _extract_dynamic_context(
    instance: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    operation: str,
) -> Dict[str, Any]:
    """
    Extract dynamic context from an embedding call for enhanced observability.

    Captures:
    - model identifier from the embedding instance
    - text_len for single-text operations
    - texts_count / empty_texts_count for batch operations
    - key AutoGen routing fields (conversation_id, agent_name, workflow_type, retriever_name)
    """
    dynamic_ctx: Dict[str, Any] = {
        "model": getattr(instance, "model", "unknown"),
        "framework_version": getattr(instance, "_framework_version", None),
    }

    # Text / batch metrics
    if operation in ("query",) and args and isinstance(args[0], str):
        dynamic_ctx["text_len"] = len(args[0])
    elif operation in ("documents", "function_call") and args and isinstance(args[0], Sequence):
        texts_seq = args[0]
        dynamic_ctx["texts_count"] = len(texts_seq)
        empty_count = sum(
            1 for text in texts_seq
            if not isinstance(text, str) or not text.strip()
        )
        if empty_count:
            dynamic_ctx["empty_texts_count"] = empty_count

    # AutoGen-specific context (if passed through kwargs)
    autogen_context = kwargs.get("autogen_context") or {}
    if isinstance(autogen_context, Mapping):
        if "conversation_id" in autogen_context:
            dynamic_ctx["conversation_id"] = autogen_context["conversation_id"]
        if "agent_name" in autogen_context:
            dynamic_ctx["agent_name"] = autogen_context["agent_name"]
        if "workflow_type" in autogen_context:
            dynamic_ctx["workflow_type"] = autogen_context["workflow_type"]
        if "retriever_name" in autogen_context:
            dynamic_ctx["retriever_name"] = autogen_context["retriever_name"]

    return dynamic_ctx


def _create_error_context_decorator(
    operation: str,
    is_async: bool = False,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Factory for creating error context decorators with rich per-call metrics.

    Mirrors the pattern used in other framework adapters (e.g., LLM, vector)
    to keep behavior consistent.
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
                    full_context: Dict[str, Any] = {
                        "error_codes": EMBEDDING_COERCION_ERROR_CODES,
                        **static_context,
                        **dynamic_context,
                    }
                    try:
                        return await func(self, *args, **kwargs)
                    except Exception as exc:  # noqa: BLE001
                        attach_context(
                            exc,
                            framework="autogen",
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
                    full_context: Dict[str, Any] = {
                        "error_codes": EMBEDDING_COERCION_ERROR_CODES,
                        **static_context,
                        **dynamic_context,
                    }
                    try:
                        return func(self, *args, **kwargs)
                    except Exception as exc:  # noqa: BLE001
                        attach_context(
                            exc,
                            framework="autogen",
                            operation=f"embedding_{operation}",
                            **full_context,
                        )
                        raise

                return sync_wrapper

        return decorator

    return decorator_factory


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


# --------------------------------------------------------------------------- #
# Core AutoGen EmbeddingFunction implementation
# --------------------------------------------------------------------------- #


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
        Optional `TextNormalizationConfig` to control whitespace cleanup,
        truncation, casing, encoding, etc.

    autogen_config:
        Optional AutoGen-specific configuration for agent context
        and workflow integration.

    framework_version:
        Optional framework version string for observability alignment.
    """

    def __init__(
        self,
        corpus_adapter: EmbeddingProtocolV1,
        model: Optional[str] = None,
        batch_config: Optional[BatchConfig] = None,
        text_normalization_config: Optional[TextNormalizationConfig] = None,
        autogen_config: Optional[Dict[str, Any]] = None,
        framework_version: Optional[str] = None,
    ) -> None:
        # Behavioral validation (duck-typed) instead of strict isinstance
        if not hasattr(corpus_adapter, "embed") or not callable(
            getattr(corpus_adapter, "embed", None),
        ):
            raise TypeError(
                "corpus_adapter must implement an EmbeddingProtocolV1-compatible "
                "interface with an 'embed' method",
            )

        # Light config validation: fail fast on clearly wrong types.
        if batch_config is not None and not isinstance(batch_config, BatchConfig):
            raise TypeError(
                f"batch_config must be a BatchConfig instance, "
                f"got {type(batch_config).__name__}",
            )
        if (
            text_normalization_config is not None
            and not isinstance(text_normalization_config, TextNormalizationConfig)
        ):
            raise TypeError(
                "text_normalization_config must be a TextNormalizationConfig instance, "
                f"got {type(text_normalization_config).__name__}",
            )

        self.corpus_adapter = corpus_adapter
        self.model = model
        self.batch_config = batch_config
        self.text_normalization_config = text_normalization_config
        self.autogen_config: Dict[str, Any] = autogen_config or {}
        self._framework_version: Optional[str] = framework_version

        logger.info(
            "CorpusAutoGenEmbeddings initialized with model=%s, autogen_config=%r, framework_version=%r",
            self.model or "default",
            self.autogen_config,
            self._framework_version,
        )

    # ------------------------------------------------------------------ #
    # Resource management (context managers)
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "CorpusAutoGenEmbeddings":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if hasattr(self.corpus_adapter, "close"):
            try:
                self.corpus_adapter.close()  # type: ignore[call-arg]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error while closing embedding adapter in __exit__: %s", exc)

    async def __aenter__(self) -> "CorpusAutoGenEmbeddings":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if hasattr(self.corpus_adapter, "aclose"):
            try:
                await self.corpus_adapter.aclose()  # type: ignore[call-arg]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error while closing embedding adapter in __aexit__: %s", exc)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @cached_property
    def _translator(self) -> EmbeddingTranslator:
        """
        Lazily construct and cache the `EmbeddingTranslator`.

        Uses `cached_property` for thread safety and performance.
        """
        translator = create_embedding_translator(
            adapter=self.corpus_adapter,
            framework="autogen",
            translator=None,  # use registry/default generic translator
            batch_config=self.batch_config,
            text_normalization_config=self.text_normalization_config,
        )
        logger.debug(
            "EmbeddingTranslator initialized for AutoGen with model=%s",
            self.model or "default",
        )
        return translator

    # ---- context helpers ------------------------------------------------- #

    def _build_core_context(
        self,
        autogen_context: Optional[Mapping[str, Any]],
    ) -> Optional[OperationContext]:
        """
        Build an OperationContext from an AutoGen-style context mapping.

        Context translation is *best-effort*: failures are logged and
        attached to the exception, but embedding operations must still
        succeed without an OperationContext.
        """
        if autogen_context is None:
            return None

        if not isinstance(autogen_context, Mapping):
            logger.warning(
                "[%s] autogen_context should be a Mapping, got %s; ignoring context",
                ErrorCodes.AUTOGEN_CONTEXT_INVALID,
                type(autogen_context).__name__,
            )
            return None

        try:
            core_candidate = context_from_autogen(
                autogen_context,
                framework_version=self._framework_version,
            )
        except Exception as exc:  # noqa: BLE001
            # Context translation is best-effort and must never break embeddings.
            logger.warning(
                "Failed to create OperationContext from autogen_context: %s. "
                "Proceeding without OperationContext.",
                exc,
            )
            try:
                snapshot = dict(autogen_context)
            except TypeError:
                snapshot = {"repr": str(autogen_context)}
            attach_context(
                exc,
                framework="autogen",
                operation="context_build",
                autogen_context_snapshot=snapshot,
                autogen_config=self.autogen_config,
                framework_version=self._framework_version,
                error_codes=EMBEDDING_COERCION_ERROR_CODES,
            )
            return None

        if isinstance(core_candidate, OperationContext):
            logger.debug(
                "Successfully created OperationContext from AutoGen context "
                "with conversation_id=%s",
                autogen_context.get("conversation_id", "unknown"),
            )
            return core_candidate

        logger.warning(
            "context_from_autogen returned non-OperationContext type: %s. "
            "Ignoring OperationContext.",
            type(core_candidate).__name__,
        )
        return None

    def _build_framework_context(
        self,
        *,
        autogen_context: Optional[Mapping[str, Any]],
        model: Optional[str],
        core_ctx: Optional[OperationContext],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Build the framework-specific context mapping for the translator.

        This carries observability hints and AutoGen-specific routing fields,
        separate from the protocol-level OperationContext.
        """
        effective_model = model or self.model

        base: Dict[str, Any] = {
            "framework": "autogen",
            "autogen_config": dict(self.autogen_config),
        }
        if self._framework_version is not None:
            base["framework_version"] = self._framework_version
        if effective_model:
            base["model"] = effective_model

        if autogen_context:
            if "agent_name" in autogen_context:
                base["agent_name"] = autogen_context["agent_name"]
            if "conversation_id" in autogen_context:
                base["conversation_id"] = autogen_context["conversation_id"]
            if "workflow_type" in autogen_context:
                base["workflow_type"] = autogen_context["workflow_type"]
            if "retriever_name" in autogen_context:
                base["retriever_name"] = autogen_context["retriever_name"]

        # Include any extra call-specific hints.
        base.update(kwargs)

        # Also expose the OperationContext itself for downstream inspection.
        if core_ctx is not None:
            base["_operation_context"] = core_ctx

        return base

    def _build_contexts(
        self,
        *,
        autogen_context: Optional[Mapping[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[Optional[OperationContext], Dict[str, Any]]:
        """
        Build both OperationContext and framework_ctx for an embedding call.

        This is a thin wrapper around `_build_core_context` and
        `_build_framework_context` to keep call sites terse.
        """
        core_ctx = self._build_core_context(autogen_context)
        framework_ctx = self._build_framework_context(
            autogen_context=autogen_context,
            model=model,
            core_ctx=core_ctx,
            **kwargs,
        )
        return core_ctx, framework_ctx

    def _coerce_embedding_matrix(self, result: Any) -> List[List[float]]:
        """
        Thin wrapper around shared coercion utility for matrix outputs.
        """
        return coerce_embedding_matrix(
            result=result,
            framework="autogen",
            error_codes=EMBEDDING_COERCION_ERROR_CODES,
            logger=logger,
        )

    def _coerce_embedding_vector(self, result: Any) -> List[float]:
        """
        Thin wrapper around shared coercion utility for single-vector outputs.
        """
        return coerce_embedding_vector(
            result=result,
            framework="autogen",
            error_codes=EMBEDDING_COERCION_ERROR_CODES,
            logger=logger,
        )

    # ------------------------------------------------------------------ #
    # Health / capabilities passthrough
    # ------------------------------------------------------------------ #

    @with_embedding_error_context("capabilities")
    def capabilities(self) -> Mapping[str, Any]:
        """
        Best-effort capabilities passthrough to the underlying adapter (sync).
        """
        if hasattr(self.corpus_adapter, "capabilities"):
            return self.corpus_adapter.capabilities()  # type: ignore[no-any-return]
        return {}

    @with_async_embedding_error_context("capabilities_async")
    async def acapabilities(self) -> Mapping[str, Any]:
        """
        Best-effort capabilities passthrough to the underlying adapter (async).
        """
        if hasattr(self.corpus_adapter, "acapabilities"):
            return await self.corpus_adapter.acapabilities()  # type: ignore[no-any-return]
        if hasattr(self.corpus_adapter, "capabilities"):
            return await asyncio.to_thread(self.corpus_adapter.capabilities)  # type: ignore[arg-type]
        return {}

    @with_embedding_error_context("health")
    def health(self) -> Mapping[str, Any]:
        """
        Best-effort health passthrough to the underlying adapter (sync).
        """
        if hasattr(self.corpus_adapter, "health"):
            return self.corpus_adapter.health()  # type: ignore[no-any-return]
        return {}

    @with_async_embedding_error_context("health_async")
    async def ahealth(self) -> Mapping[str, Any]:
        """
        Best-effort health passthrough to the underlying adapter (async).
        """
        if hasattr(self.corpus_adapter, "ahealth"):
            return await self.corpus_adapter.ahealth()  # type: ignore[no-any-return]
        if hasattr(self.corpus_adapter, "health"):
            return await asyncio.to_thread(self.corpus_adapter.health)  # type: ignore[arg-type]
        return {}

    # ------------------------------------------------------------------ #
    # Core AutoGen EmbeddingFunction Interface
    # ------------------------------------------------------------------ #

    @with_embedding_error_context("function_call")
    def __call__(
        self,
        texts: Sequence[str],
        *,
        autogen_context: Optional[Mapping[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """
        Make the instance callable for AutoGen's EmbeddingFunction protocol.

        This enables direct usage with AutoGen's VectorStoreRetriever:
        ```python
        retriever = VectorStoreRetriever(
            vectorstore=Chroma(embedding_function=CorpusAutoGenEmbeddings(...)),
            ...
        )
        ```
        """
        # AutoGen generally passes a list, but Sequence[str] keeps us flexible.
        return self.embed_documents(
            list(texts),
            autogen_context=autogen_context,
            model=model,
            **kwargs,
        )

    @with_embedding_error_context("documents")
    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        autogen_context: Optional[Mapping[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """
        Sync embedding for multiple documents.

        This is the primary method used by AutoGen's retrieval systems
        for document embedding and agent memory.
        """
        texts_list = list(texts)
        warn_if_extreme_batch(
            framework="autogen",
            texts=texts_list,
            op_name="embed_documents",
            batch_config=self.batch_config,
            logger=logger,
        )

        core_ctx, framework_ctx = self._build_contexts(
            autogen_context=autogen_context,
            model=model,
            **kwargs,
        )

        logger.debug(
            "Sync embedding %d documents for AutoGen conversation: %s",
            len(texts_list),
            framework_ctx.get("conversation_id", "unknown"),
        )

        translated = self._translator.embed(
            raw_texts=texts_list,
            op_ctx=core_ctx,
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
        """
        core_ctx, framework_ctx = self._build_contexts(
            autogen_context=autogen_context,
            model=model,
            **kwargs,
        )

        logger.debug(
            "Sync embedding query for AutoGen conversation: %s",
            framework_ctx.get("conversation_id", "unknown"),
        )

        translated = self._translator.embed(
            raw_texts=text,
            op_ctx=core_ctx,
            framework_ctx=framework_ctx,
        )
        return self._coerce_embedding_vector(translated)

    # ------------------------------------------------------------------ #
    # Async API for AutoGen Async Workflows
    # ------------------------------------------------------------------ #

    @with_async_embedding_error_context("documents")
    async def aembed_documents(
        self,
        texts: Sequence[str],
        *,
        autogen_context: Optional[Mapping[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """
        Async embedding for multiple documents.

        Designed for use with AutoGen's async workflows and
        event-driven agent systems.
        """
        texts_list = list(texts)
        warn_if_extreme_batch(
            framework="autogen",
            texts=texts_list,
            op_name="aembed_documents",
            batch_config=self.batch_config,
            logger=logger,
        )

        core_ctx, framework_ctx = self._build_contexts(
            autogen_context=autogen_context,
            model=model,
            **kwargs,
        )

        logger.debug(
            "Async embedding %d documents for AutoGen conversation: %s",
            len(texts_list),
            framework_ctx.get("conversation_id", "unknown"),
        )

        translated = await self._translator.arun_embed(
            raw_texts=texts_list,
            op_ctx=core_ctx,
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
        """
        core_ctx, framework_ctx = self._build_contexts(
            autogen_context=autogen_context,
            model=model,
            **kwargs,
        )

        logger.debug(
            "Async embedding query for AutoGen conversation: %s",
            framework_ctx.get("conversation_id", "unknown"),
        )

        translated = await self._translator.arun_embed(
            raw_texts=text,
            op_ctx=core_ctx,
            framework_ctx=framework_ctx,
        )
        return self._coerce_embedding_vector(translated)


# --------------------------------------------------------------------------- #
# AutoGen-Specific Helper Functions
# --------------------------------------------------------------------------- #


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
    *,
    model: Optional[str] = None,
    batch_config: Optional[BatchConfig] = None,
    text_normalization_config: Optional[TextNormalizationConfig] = None,
    autogen_config: Optional[Dict[str, Any]] = None,
    framework_version: Optional[str] = None,
) -> AutoGenRetriever:
    """
    Create an AutoGen VectorStoreRetriever with Corpus embeddings.

    This provides a convenient way to create AutoGen retrievers
    with Corpus embeddings in a single function call.

    Parameters
    ----------
    corpus_adapter:
        Underlying embedding adapter implementing `EmbeddingProtocolV1`.
    vector_store:
        Vector store instance compatible with AutoGen's VectorStoreRetriever.
    model:
        Optional model identifier to use for embeddings.
    batch_config:
        Optional batching configuration forwarded to `CorpusAutoGenEmbeddings`.
    text_normalization_config:
        Optional text normalization configuration forwarded to
        `CorpusAutoGenEmbeddings`.
    autogen_config:
        Optional AutoGen-specific configuration for agent/workflow integration.
    framework_version:
        Optional framework version string for observability alignment.

    Example usage:
    ```python
    from corpus_sdk.embedding.framework_adapters.autogen import create_retriever
    from chromadb import Chroma

    vectorstore = Chroma(collection_name="autogen_docs")

    retriever = create_retriever(
        corpus_adapter=my_adapter,
        vector_store=vectorstore,
        model="text-embedding-3-large",
    )
    ```
    """
    try:
        from autogen.retrieve_utils import VectorStoreRetriever
    except ImportError as exc:  # noqa: BLE001
        message = (
            "AutoGen is not installed. To use create_retriever, install the "
            "AutoGen package, for example: 'pip install pyautogen'."
        )
        logger.error(message)
        raise RuntimeError(message) from exc

    # Best-effort validation before mutating the vector store.
    _validate_vector_store_for_autogen(vector_store)

    embedding_function = CorpusAutoGenEmbeddings(
        corpus_adapter=corpus_adapter,
        model=model,
        batch_config=batch_config,
        text_normalization_config=text_normalization_config,
        autogen_config=autogen_config,
        framework_version=framework_version,
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

    logger.info("AutoGen retriever created with Corpus embeddings")

    return retriever


def register_embeddings(
    corpus_adapter: EmbeddingProtocolV1,
    model: Optional[str] = None,
    batch_config: Optional[BatchConfig] = None,
    text_normalization_config: Optional[TextNormalizationConfig] = None,
    autogen_config: Optional[Dict[str, Any]] = None,
    framework_version: Optional[str] = None,
) -> CorpusAutoGenEmbeddings:
    """
    Register Corpus embeddings for global use in AutoGen workflows.

    This function provides a centralized way to configure Corpus embeddings
    for multiple AutoGen agents and retrievers.
    """
    embeddings = CorpusAutoGenEmbeddings(
        corpus_adapter=corpus_adapter,
        model=model,
        batch_config=batch_config,
        text_normalization_config=text_normalization_config,
        autogen_config=autogen_config,
        framework_version=framework_version,
    )

    logger.info(
        "Corpus AutoGen embeddings registered: %s (framework_version=%r)",
        model or "default model",
        framework_version,
    )

    return embeddings


__all__ = [
    "CorpusAutoGenEmbeddings",
    "AutoGenContext",
    "AutoGenRetriever",
    "create_retriever",
    "register_embeddings",
    "ErrorCodes",
    "with_embedding_error_context",
    "with_async_embedding_error_context",
]
