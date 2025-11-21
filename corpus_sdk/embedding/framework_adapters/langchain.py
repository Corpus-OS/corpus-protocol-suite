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
    TypedDict,
)

from pydantic import BaseModel, ConfigDict, PrivateAttr, field_validator
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
    # Correct spelling (primary)
    LANGCHAIN_CONFIG_INVALID = "LANGCHAIN_CONFIG_INVALID"

class LangChainConfig(TypedDict, total=False):
    """Structured type for LangChain RunnableConfig context."""
    configurable: Optional[Dict[str, Any]]
    tags: Optional[List[str]]
    metadata: Optional[Dict[str, Any]]
    callbacks: Optional[Any]
    run_name: Optional[str]
    run_id: Optional[str]


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

    Example
    -------
    ```python
    from langchain.vectorstores import Chroma
    from corpus_sdk.embedding.framework_adapters.langchain import CorpusLangChainEmbeddings

    embeddings = CorpusLangChainEmbeddings(
        corpus_adapter=my_adapter,
        model="text-embedding-3-large",
        batch_config=BatchConfig(max_batch_size=1000)
    )

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="research_papers"
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )
    ```

    Error Handling Example
    ----------------------
    ```python
    try:
        results = embeddings.embed_documents(
            texts=research_docs,
            config={
                "tags": ["research", "batch-processing"],
                "metadata": {"pipeline": "document-indexing"}
            }
        )
    except Exception as e:
        # Rich error context automatically attached
        logger.error("Embedding failed with context", exc_info=e)
    ```

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

    @field_validator("corpus_adapter")
    @classmethod
    def validate_corpus_adapter(cls, v: EmbeddingProtocolV1) -> EmbeddingProtocolV1:
        """
        Validate that corpus_adapter implements the required embedding protocol.

        We do a behavioral check (presence of `embed`) instead of strict type
        checking to remain flexible with Protocol-based adapters.
        """
        if not hasattr(v, "embed") or not callable(getattr(v, "embed")):
            raise ValueError(
                "corpus_adapter must implement EmbeddingProtocolV1 with an 'embed' method"
            )
        return v

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
            logger.debug(
                "EmbeddingTranslator initialized for LangChain with model: %s",
                self.model or "default",
            )
        return self._translator_cache

    def _build_contexts(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[Optional[OperationContext], Dict[str, Any], Dict[str, Any]]:
        """
        Build contexts for LangChain execution environment with comprehensive validation.

        Returns
        -------
        Tuple of:
        - core_ctx: core OperationContext or None if no/invalid context
        - op_ctx_dict: normalized dict for embedding layer with rich context preservation
        - framework_ctx: LangChain-specific context for translator
        """
        core_ctx: Optional[OperationContext] = None
        op_ctx_dict: Dict[str, Any] = {}
        framework_ctx: Dict[str, Any] = {
            "framework": "langchain",
        }

        # Convert LangChain config to core OperationContext with defensive handling
        if config is not None:
            try:
                self._validate_langchain_config_structure(config)

                core_ctx_candidate = context_from_langchain(config)
                if isinstance(core_ctx_candidate, OperationContext):
                    core_ctx = core_ctx_candidate
                    logger.debug(
                        "Successfully created OperationContext from LangChain config "
                        "with run_id: %s",
                        config.get("run_id", "unknown"),
                    )
                else:
                    logger.warning(
                        "context_from_langchain returned non-OperationContext type: %s. "
                        "Proceeding with empty OperationContext.",
                        type(core_ctx_candidate).__name__,
                    )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Failed to create OperationContext from LangChain config: %s. "
                    "Proceeding with empty OperationContext.",
                    e,
                )
                # Attach a snapshot for observability while preserving behavior
                try:
                    snapshot = dict(config)
                except Exception:  # noqa: BLE001
                    snapshot = {"repr": repr(config)}
                attach_context(
                    e,
                    framework="langchain",
                    operation="context_build",
                    config_snapshot=snapshot,
                )

        # Build comprehensive context dictionary preserving rich OperationContext
        if core_ctx is not None:
            # Preserve the rich OperationContext object for maximum fidelity
            op_ctx_dict = {"_operation_context": core_ctx}

            # Include structured dict representation for compatibility
            if hasattr(core_ctx, "to_dict"):
                op_ctx_dict.update(core_ctx.to_dict())
            elif hasattr(core_ctx, "__dict__"):
                op_ctx_dict.update(core_ctx.__dict__)
        else:
            op_ctx_dict = {}

        # Framework-level context for LangChain-specific optimizations
        effective_model = model or self.model
        if effective_model:
            framework_ctx["model"] = effective_model

        # Add LangChain-specific context for observability
        if config:
            framework_ctx.update(
                {
                    "tags": config.get("tags"),
                    "run_name": config.get("run_name"),
                    "run_id": config.get("run_id"),
                    "metadata": config.get("metadata"),
                }
            )

            # If `configurable` sub-context exists, surface it for downstream logic
            configurable = config.get("configurable")
            if isinstance(configurable, Mapping):
                framework_ctx["configurable"] = dict(configurable)

        # Include any extra call-specific hints
        framework_ctx.update(kwargs)

        return core_ctx, op_ctx_dict, framework_ctx

    def _validate_langchain_config_structure(
        self,
        config: Mapping[str, Any],
    ) -> None:
        """
        Validate LangChain config structure and log warnings for anomalies.

        This is intentionally non-fatal for maximal compatibility: we only
        log and enrich context instead of raising hard errors.
        """
        if not isinstance(config, Mapping):
            logger.warning(
                "[%s] LangChain config is not a Mapping (got %s); "
                "context translation may be degraded.",
                ErrorCodes.LANGCHAIN_CONFIG_INVALID,
                type(config).__name__,
            )
            return

        # Check for common LangChain config fields to improve diagnostics
        if not any(
            key in config
            for key in ("tags", "metadata", "run_name", "run_id", "callbacks")
        ):
            logger.debug(
                "LangChain config missing common fields (tags, metadata, run_name, "
                "run_id, callbacks) – reduced context for embeddings.",
            )

    def _coerce_embedding_matrix(self, result: Any) -> List[List[float]]:
        """
        Coerce translator result into a List[List[float]] embedding matrix with comprehensive validation.

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
        **kwargs: Any,
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
        **kwargs:
            Additional framework-specific parameters.
        """
        self._warn_if_extreme_batch(texts, op_name="aembed_documents")

        core_ctx, op_ctx_dict, framework_ctx = self._build_contexts(
            config=config,
            model=model,
            **kwargs,
        )

        # Pass the rich OperationContext through for maximum fidelity
        if core_ctx is not None:
            framework_ctx["_operation_context"] = core_ctx

        logger.debug(
            "Async embedding %d documents for LangChain run: %s",
            len(texts),
            config.get("run_id", "unknown") if config else "unknown",
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
        **kwargs: Any,
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
        **kwargs:
            Additional framework-specific parameters.
        """
        core_ctx, op_ctx_dict, framework_ctx = self._build_contexts(
            config=config,
            model=model,
            **kwargs,
        )

        # Pass the rich OperationContext through for maximum fidelity
        if core_ctx is not None:
            framework_ctx["_operation_context"] = core_ctx

        logger.debug(
            "Async embedding query for LangChain run: %s",
            config.get("run_id", "unknown") if config else "unknown",
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
        **kwargs: Any,
    ) -> List[List[float]]:
        """
        Sync embedding for multiple documents.

        Uses the synchronous `EmbeddingTranslator.embed` API, which internally
        bridges async protocol calls and respects any `deadline_ms` timeout
        encoded in the OperationContext.
        """
        self._warn_if_extreme_batch(texts, op_name="embed_documents")

        core_ctx, op_ctx_dict, framework_ctx = self._build_contexts(
            config=config,
            model=model,
            **kwargs,
        )

        # Pass the rich OperationContext through for maximum fidelity
        if core_ctx is not None:
            framework_ctx["_operation_context"] = core_ctx

        logger.debug(
            "Sync embedding %d documents for LangChain run: %s",
            len(texts),
            config.get("run_name", "unknown") if config else "unknown",
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
        **kwargs: Any,
    ) -> List[float]:
        """
        Sync embedding for a single query.

        Uses the synchronous `EmbeddingTranslator.embed` API, which internally
        bridges async protocol calls and respects any `deadline_ms` timeout
        encoded in the OperationContext.
        """
        core_ctx, op_ctx_dict, framework_ctx = self._build_contexts(
            config=config,
            model=model,
            **kwargs,
        )

        # Pass the rich OperationContext through for maximum fidelity
        if core_ctx is not None:
            framework_ctx["_operation_context"] = core_ctx

        logger.debug(
            "Sync embedding query for LangChain run: %s",
            config.get("run_name", "unknown") if config else "unknown",
        )

        translated = self._translator.embed(
            raw_texts=text,
            op_ctx=op_ctx_dict,
            framework_ctx=framework_ctx,
        )
        return self._coerce_embedding_vector(translated)


__all__ = [
    "CorpusLangChainEmbeddings",
    "LangChainConfig",
    "ErrorCodes",
]