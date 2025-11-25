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

Resilience (retries, caching, rate limiting, etc.) is expected to be provided
by the underlying adapter, typically a BaseEmbeddingAdapter subclass.
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
from corpus_sdk.embedding.framework_adapters.common.framework_utils import (
    CoercionErrorCodes,
    coerce_embedding_matrix,
    coerce_embedding_vector,
    warn_if_extreme_batch,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

_FRAMEWORK_NAME = "langchain"

# ---------------------------------------------------------------------------
# Safe conditional import for LangChain Embeddings
# ---------------------------------------------------------------------------

try:
    from langchain_core.embeddings import Embeddings  # type: ignore[no-redef]

    LANGCHAIN_AVAILABLE = True
except ImportError:  # pragma: no cover - only used when LangChain isn't installed

    class Embeddings:  # type: ignore[no-redef]
        """
        Minimal fallback base class when LangChain is not installed.

        This is only to keep imports from failing. Using the adapter without
        LangChain installed is effectively a misconfiguration.
        """

        pass

    LANGCHAIN_AVAILABLE = False


class ErrorCodes:
    """
    Error code constants for LangChain embedding adapter.

    This is a simple namespace for framework-specific codes. The shared
    coercion helpers use `EMBEDDING_COERCION_ERROR_CODES`, which is a
    `CoercionErrorCodes` instance derived from these values.
    """

    # Coercion-level (used by framework_utils)
    INVALID_EMBEDDING_RESULT = "INVALID_EMBEDDING_RESULT"
    EMPTY_EMBEDDING_RESULT = "EMPTY_EMBEDDING_RESULT"
    EMBEDDING_CONVERSION_ERROR = "EMBEDDING_CONVERSION_ERROR"

    # LangChain-specific config errors
    LANGCHAIN_CONFIG_INVALID = "LANGCHAIN_CONFIG_INVALID"


# Coercion configuration for the common embedding utils
EMBEDDING_COERCION_ERROR_CODES: CoercionErrorCodes = CoercionErrorCodes(
    invalid_result=ErrorCodes.INVALID_EMBEDDING_RESULT,
    empty_result=ErrorCodes.EMPTY_EMBEDDING_RESULT,
    conversion_error=ErrorCodes.EMBEDDING_CONVERSION_ERROR,
    framework_label=_FRAMEWORK_NAME,
)


class LangChainConfig(TypedDict, total=False):
    """
    Structured type for LangChain RunnableConfig-like context.

    This mirrors the common fields exposed by LangChain's RunnableConfig /
    invocation layer and is used both for type safety and for observability
    context extraction.
    """

    configurable: Optional[Dict[str, Any]]
    tags: Optional[List[str]]
    metadata: Optional[Dict[str, Any]]
    callbacks: Optional[Any]
    run_name: Optional[str]
    run_id: Optional[str]


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
    Extract rich dynamic context from a LangChain embedding call.

    Captures:
    - model identifier from the embedding instance
    - framework_version if present
    - text_len for single-text operations
    - texts_count / empty_texts_count for batch operations
    - LangChain routing fields (run_id, run_name, tags)
    """
    dynamic_ctx: Dict[str, Any] = {
        "model": getattr(instance, "model", "unknown"),
        "framework_version": getattr(instance, "framework_version", None),
        "framework_name": _FRAMEWORK_NAME,
    }

    # Text / batch metrics
    if operation == "query" and args and isinstance(args[0], str):
        dynamic_ctx["text_len"] = len(args[0])
    elif operation == "documents" and args and isinstance(args[0], Sequence):
        texts_seq = args[0]
        dynamic_ctx["texts_count"] = len(texts_seq)
        empty_count = sum(
            1 for text in texts_seq
            if not isinstance(text, str) or not text.strip()
        )
        if empty_count:
            dynamic_ctx["empty_texts_count"] = empty_count

    # LangChain-specific config (if passed via keyword)
    config = kwargs.get("config") or {}
    if isinstance(config, Mapping):
        if "run_id" in config:
            dynamic_ctx["run_id"] = config["run_id"]
        if "run_name" in config:
            dynamic_ctx["run_name"] = config["run_name"]
        if "tags" in config:
            dynamic_ctx["tags"] = config["tags"]

    return dynamic_ctx


def _create_error_context_decorator(
    operation: str,
    is_async: bool = False,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Factory for creating error-context decorators with rich per-call metrics.

    Mirrors the pattern used in other framework adapters (LlamaIndex,
    Semantic Kernel, AutoGen, CrewAI) for consistent observability.
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
                            framework=_FRAMEWORK_NAME,
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
                            framework=_FRAMEWORK_NAME,
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
    # Always include coercion error codes in error context.
    static_context.setdefault("error_codes", EMBEDDING_COERCION_ERROR_CODES)
    return _create_error_context_decorator(operation, is_async=False)(**static_context)


def with_async_embedding_error_context(
    operation: str,
    **static_context: Any,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for async methods with rich dynamic context extraction."""
    static_context.setdefault("error_codes", EMBEDDING_COERCION_ERROR_CODES)
    return _create_error_context_decorator(operation, is_async=True)(**static_context)


class CorpusLangChainEmbeddings(BaseModel, Embeddings):
    """
    LangChain `Embeddings` backed by a Corpus `EmbeddingProtocolV1` adapter.

    Inherits from `BaseModel` to support Pydantic-style initialization (standard
    in LangChain) and `Embeddings` to satisfy the interface contract.

    Example
    -------
    ```python
    from langchain.vectorstores import Chroma
    from corpus_sdk.embedding.framework_adapters.langchain import (
        configure_langchain_embeddings,
    )

    embeddings = configure_langchain_embeddings(
        corpus_adapter=my_adapter,
        model="text-embedding-3-large",
        batch_config=BatchConfig(max_batch_size=1000),
    )

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="research_papers",
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10},
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
                "metadata": {"pipeline": "document-indexing"},
            },
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

    framework_version:
        Optional framework version string for observability and context
        translation (e.g., LangChain version).

    batch_config:
        Optional `BatchConfig` to control batching behavior. If None, the
        defaults in the common embedding layer are used.

    text_normalization_config:
        Optional `TextNormalizationConfig` to control whitespace cleanup,
        truncation, casing, encoding, etc.
    """

    corpus_adapter: EmbeddingProtocolV1
    model: Optional[str] = None
    framework_version: Optional[str] = None
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
    # Resource management / lifecycle helpers
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "CorpusLangChainEmbeddings":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        adapter = self.corpus_adapter
        close = getattr(adapter, "close", None)
        if callable(close):
            try:
                close()
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Error while closing embedding adapter in __exit__: %s",
                    e,
                )

    async def __aenter__(self) -> "CorpusLangChainEmbeddings":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        adapter = self.corpus_adapter
        aclose = getattr(adapter, "aclose", None)
        if callable(aclose):
            try:
                await aclose()
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Error while closing embedding adapter in __aexit__: %s",
                    e,
                )

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
                framework=_FRAMEWORK_NAME,
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
        config: Optional[LangChainConfig] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[Optional[OperationContext], Dict[str, Any]]:
        """
        Build contexts for LangChain execution environment with comprehensive validation.

        Returns
        -------
        Tuple of:
        - core_ctx: OperationContext instance (or None if unavailable)
        - framework_ctx: LangChain-specific context for translator
        """
        core_ctx: Optional[OperationContext] = None
        framework_ctx: Dict[str, Any] = {
            "framework": _FRAMEWORK_NAME,
        }
        if self.framework_version is not None:
            framework_ctx["framework_version"] = self.framework_version

        # Convert LangChain config to core OperationContext with defensive handling
        if config is not None:
            try:
                self._validate_langchain_config_structure(config)

                core_ctx_candidate = context_from_langchain(
                    config,
                    framework_version=self.framework_version,
                )
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
                        "Proceeding without OperationContext.",
                        type(core_ctx_candidate).__name__,
                    )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Failed to create OperationContext from LangChain config: %s. "
                    "Proceeding without OperationContext.",
                    e,
                )
                # Attach a snapshot for observability while preserving behavior
                try:
                    snapshot = dict(config)
                except Exception:  # noqa: BLE001
                    snapshot = {"repr": repr(config)}
                attach_context(
                    e,
                    framework=_FRAMEWORK_NAME,
                    operation="context_build",
                    config_snapshot=snapshot,
                    framework_version=self.framework_version,
                )

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

        # Also surface the OperationContext itself for downstream inspection, if present
        if core_ctx is not None:
            framework_ctx["_operation_context"] = core_ctx

        return core_ctx, framework_ctx

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
        warn_if_extreme_batch(
            framework=_FRAMEWORK_NAME,
            texts=texts,
            op_name=op_name,
            batch_config=self.batch_config,
            logger=logger,
        )

    # ------------------------------------------------------------------ #
    # Async API
    # ------------------------------------------------------------------ #

    @with_async_embedding_error_context("documents")
    async def aembed_documents(
        self,
        texts: Sequence[str],
        *,
        config: Optional[LangChainConfig] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """
        Async embedding for multiple documents.

        Parameters
        ----------
        texts:
            Sequence of documents to embed.
        config:
            Optional LangChain RunnableConfig-like dict. Used only for
            context translation (request_id, tenant, deadline, tags, etc.).
        model:
            Optional per-call model override.
        **kwargs:
            Additional framework-specific parameters.
        """
        self._warn_if_extreme_batch(texts, op_name="aembed_documents")

        core_ctx, framework_ctx = self._build_contexts(
            config=config,
            model=model,
            **kwargs,
        )

        logger.debug(
            "Async embedding %d documents for LangChain run: %s",
            len(texts),
            config.get("run_id", "unknown") if config else "unknown",
        )

        translated = await self._translator.arun_embed(
            raw_texts=list(texts),
            op_ctx=core_ctx,
            framework_ctx=framework_ctx,
        )
        return self._coerce_embedding_matrix(translated)

    @with_async_embedding_error_context("query")
    async def aembed_query(
        self,
        text: str,
        *,
        config: Optional[LangChainConfig] = None,
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
        core_ctx, framework_ctx = self._build_contexts(
            config=config,
            model=model,
            **kwargs,
        )

        logger.debug(
            "Async embedding query for LangChain run: %s",
            config.get("run_id", "unknown") if config else "unknown",
        )

        translated = await self._translator.arun_embed(
            raw_texts=text,
            op_ctx=core_ctx,
            framework_ctx=framework_ctx,
        )
        return self._coerce_embedding_vector(translated)

    # ------------------------------------------------------------------ #
    # Sync API
    # ------------------------------------------------------------------ #

    @with_embedding_error_context("documents")
    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        config: Optional[LangChainConfig] = None,
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

        core_ctx, framework_ctx = self._build_contexts(
            config=config,
            model=model,
            **kwargs,
        )

        logger.debug(
            "Sync embedding %d documents for LangChain run: %s",
            len(texts),
            config.get("run_name", "unknown") if config else "unknown",
        )

        translated = self._translator.embed(
            raw_texts=list(texts),
            op_ctx=core_ctx,
            framework_ctx=framework_ctx,
        )
        return self._coerce_embedding_matrix(translated)

    @with_embedding_error_context("query")
    def embed_query(
        self,
        text: str,
        *,
        config: Optional[LangChainConfig] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[float]:
        """
        Sync embedding for a single query.

        Uses the synchronous `EmbeddingTranslator.embed` API, which internally
        bridges async protocol calls and respects any `deadline_ms` timeout
        encoded in the OperationContext.
        """
        core_ctx, framework_ctx = self._build_contexts(
            config=config,
            model=model,
            **kwargs,
        )

        logger.debug(
            "Sync embedding query for LangChain run: %s",
            config.get("run_name", "unknown") if config else "unknown",
        )

        translated = self._translator.embed(
            raw_texts=text,
            op_ctx=core_ctx,
            framework_ctx=framework_ctx,
        )
        return self._coerce_embedding_vector(translated)


# ------------------------------------------------------------------ #
# LangChain "configuration / registration" helpers
# ------------------------------------------------------------------ #


def configure_langchain_embeddings(
    corpus_adapter: EmbeddingProtocolV1,
    model: Optional[str] = None,
    framework_version: Optional[str] = None,
    **kwargs: Any,
) -> CorpusLangChainEmbeddings:
    """
    Configure and return Corpus embeddings for LangChain usage.

    This mirrors the *shape* of the Semantic Kernel / LlamaIndex helpers:

    - Always constructs and returns a `CorpusLangChainEmbeddings` instance.
    - If LangChain is not installed, the adapter still constructs, but you
      obviously won't be able to plug it into real LangChain pipelines.

    Unlike Semantic Kernel or LlamaIndex, LangChain does not expose a single
    global "Settings" object for embeddings, so this helper does *not* attempt
    any global registration; you pass the returned instance into vectorstores,
    retrievers, chains, etc.

    Example
    -------
    ```python
    from corpus_sdk.embedding.framework_adapters.langchain import (
        configure_langchain_embeddings,
    )

    embeddings = configure_langchain_embeddings(
        corpus_adapter=my_adapter,
        model="text-embedding-3-large",
    )
    ```

    Parameters
    ----------
    corpus_adapter:
        Corpus embedding protocol adapter implementing `EmbeddingProtocolV1`.
    model:
        Optional default model identifier.
    framework_version:
        Optional framework version string (e.g. LangChain version).
    **kwargs:
        Additional arguments for `CorpusLangChainEmbeddings`
        (e.g. batch_config, text_normalization_config).

    Returns
    -------
    CorpusLangChainEmbeddings
        Configured embeddings instance ready for LangChain integration.
    """
    embeddings = CorpusLangChainEmbeddings(
        corpus_adapter=corpus_adapter,
        model=model,
        framework_version=framework_version,
        **kwargs,
    )

    if not LANGCHAIN_AVAILABLE:
        logger.debug(
            "LangChain is not installed; returning embeddings without any "
            "framework-level integration.",
        )
    else:
        logger.info(
            "Corpus LangChain embeddings configured with model=%s, framework_version=%s",
            model or "default",
            framework_version or "unknown",
        )

    return embeddings


def register_with_langchain(
    corpus_adapter: EmbeddingProtocolV1,
    model: Optional[str] = None,
    framework_version: Optional[str] = None,
    **kwargs: Any,
) -> CorpusLangChainEmbeddings:
    """
    Alias for `configure_langchain_embeddings` to mirror the
    `register_with_semantic_kernel` / `register_with_llamaindex`
    naming convention.

    This helper is primarily for API symmetry across framework adapters,
    rather than actual global registration.
    """
    return configure_langchain_embeddings(
        corpus_adapter=corpus_adapter,
        model=model,
        framework_version=framework_version,
        **kwargs,
    )


__all__ = [
    "CorpusLangChainEmbeddings",
    "LangChainConfig",
    "ErrorCodes",
    "configure_langchain_embeddings",
    "register_with_langchain",
    "with_embedding_error_context",
    "with_async_embedding_error_context",
    "LANGCHAIN_AVAILABLE",
]
