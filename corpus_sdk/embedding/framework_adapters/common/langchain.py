# corpus_sdk/embedding/framework_adapters/langchain.py
# SPDX-License-Identifier: Apache-2.0

"""
LangChain adapter for Corpus Embedding protocol.

This module exposes Corpus `EmbeddingProtocolV1` implementations as
`langchain_core.embeddings.Embeddings`, with:

- Sync + async embedding for documents and queries
- Context normalization via `context_translation.from_langchain`
- Framework-agnostic orchestration via `EmbeddingTranslator`
- Async → sync bridging using `AsyncBridge`
- Rich error context attachment for observability
- Model selection via framework_ctx / OperationContext attrs

The design mirrors the Corpus LangChain LLM adapter: this is a *thin*,
framework-specific skin over the protocol-first Corpus embedding stack.
"""

from __future__ import annotations

import logging
from functools import cached_property
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from langchain_core.embeddings import Embeddings

from corpus_sdk.core.async_bridge import AsyncBridge
from corpus_sdk.core.context_translation import (
    from_langchain as context_from_langchain,
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


class EmbeddingOperationError(Exception):
    """Base exception for embedding operations with context."""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}


def _with_error_context(
    framework: str,
    operation: str,
    **default_context: Any,
) -> Any:
    """
    Decorator to attach error context to exceptions from embedding operations.
    
    Args:
        framework: The framework name ("langchain")
        operation: The operation name ("embed_documents", "embed_query")
        default_context: Default context values to attach
    """
    def decorator(func):
        if isinstance(func, type) or not callable(func):
            raise TypeError("_with_error_context can only decorate callables")
            
        async def async_wrapper(self, *args, **kwargs):
            core_ctx = None
            try:
                # Extract core_ctx from method arguments or build it
                if hasattr(self, '_build_contexts'):
                    # For class methods that use _build_contexts
                    config = kwargs.get('config')
                    model = kwargs.get('model')
                    core_ctx, _, _ = self._build_contexts(config=config, model=model)
                return await func(self, *args, **kwargs)
            except Exception as exc:
                context = default_context.copy()
                context.update({
                    "framework": framework,
                    "embedding_operation": operation,
                    "request_id": getattr(core_ctx, "request_id", None) if core_ctx else None,
                    "tenant": getattr(core_ctx, "tenant", None) if core_ctx else None,
                })
                # Add operation-specific context
                if operation == "embed_documents":
                    context["texts_count"] = len(kwargs.get('texts', []))
                elif operation == "embed_query":
                    context["text_len"] = len(kwargs.get('text', ''))
                
                try:
                    attach_context(exc, **context)
                except Exception:
                    pass  # Never mask original error
                raise

        def sync_wrapper(self, *args, **kwargs):
            core_ctx = None
            try:
                if hasattr(self, '_build_contexts'):
                    config = kwargs.get('config')
                    model = kwargs.get('model')
                    core_ctx, _, _ = self._build_contexts(config=config, model=model)
                return func(self, *args, **kwargs)
            except Exception as exc:
                context = default_context.copy()
                context.update({
                    "framework": framework,
                    "embedding_operation": operation,
                    "request_id": getattr(core_ctx, "request_id", None) if core_ctx else None,
                    "tenant": getattr(core_ctx, "tenant", None) if core_ctx else None,
                })
                if operation == "embed_documents":
                    context["texts_count"] = len(kwargs.get('texts', []))
                elif operation == "embed_query":
                    context["text_len"] = len(kwargs.get('text', ''))
                
                try:
                    attach_context(exc, **context)
                except Exception:
                    pass
                raise

        return async_wrapper if func.__name__.startswith('a') else sync_wrapper
    return decorator


class CorpusLangChainEmbeddings(Embeddings):
    """
    LangChain `Embeddings` backed by a Corpus `EmbeddingProtocolV1` adapter.

    Responsibilities (this layer)
    -----------------------------
    - Accept LangChain-style calls (`embed_documents`, `embed_query`,
      and their async variants).
    - Derive an `OperationContext` from optional LangChain `config`
      via `context_translation.from_langchain`.
    - Build a small `framework_ctx` (currently just `model`) that is
      passed to the common embedding translator.
    - Use `EmbeddingTranslator` to:
        * Build EmbedSpecs
        * Call the underlying adapter
        * Translate the result to a framework-facing shape
    - Use `AsyncBridge` to bridge async protocol calls for sync APIs.
    - Attach structured error context via `attach_context`.

    Non-responsibilities
    --------------------
    - Text normalization (whitespace, truncation, casing).
    - Batching logic, token-aware batching, cost tracking.
    - Provider-specific behavior (rate limits, retries, etc.).

    All of those live in:
    - `corpus_sdk.embedding.framework_adapters.common.embedding_translation`
    - Concrete `EmbeddingProtocolV1` adapter implementations.

    Attributes
    ----------
    corpus_adapter:
        Underlying Corpus embedding adapter implementing `EmbeddingProtocolV1`.

    model:
        Optional default model identifier. Can be overridden per call by
        passing `model=...` to `embed_documents` / `embed_query` /
        their async variants. If unset, the underlying adapter / translator
        is responsible for choosing a default.

    batch_config:
        Optional `BatchConfig` to control batching behavior. If None, the
        default config in the common embedding layer is used.

    text_normalization_config:
        Optional `TextNormalizationConfig` to control whitespace cleanup,
        truncation, casing, encoding, etc. If None, the default is used.
    """

    corpus_adapter: EmbeddingProtocolV1
    model: Optional[str] = None
    batch_config: Optional[BatchConfig] = None
    text_normalization_config: Optional[TextNormalizationConfig] = None

    # Pydantic v2-style config: allow arbitrary types like EmbeddingProtocolV1.
    model_config = {"arbitrary_types_allowed": True}

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @cached_property
    def _translator(self) -> EmbeddingTranslator:
        """
        Lazily construct and cache the `EmbeddingTranslator`.

        Uses `cached_property` for cleaner implementation and thread safety.
        """
        return create_embedding_translator(
            adapter=self.corpus_adapter,
            framework="langchain",
            translator=None,  # use registry/default generic translator
            batch_config=self.batch_config,
            text_normalization_config=self.text_normalization_config,
        )

    def _build_contexts(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
        model: Optional[str] = None,
    ) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        """
        Build:
        - `core_ctx`: core OperationContext (from context_translation)
        - `op_ctx_dict`: normalized dict version of core_ctx (for embedding layer)
        - `framework_ctx`: dict passed to the translator for LangChain-specific hints

        We deliberately pass `op_ctx_dict` into the embedding translation layer
        so that it can reconstruct its own embedding OperationContext type
        independently.
        """
        core_ctx = context_from_langchain(config)

        # Normalized dict for embedding OperationContext reconstruction.
        op_ctx_dict: Dict[str, Any] = core_ctx.to_dict()

        # Framework-level context: currently we mainly care about model hint.
        framework_ctx: Dict[str, Any] = {}
        effective_model = model or self.model
        if effective_model:
            framework_ctx["model"] = effective_model

        return core_ctx, op_ctx_dict, framework_ctx

    @staticmethod
    def _coerce_embedding_matrix(result: Any) -> List[List[float]]:
        """
        Coerce translator result into a List[List[float]] embedding matrix.

        Expected shapes supported:
        - Default translator:
            {"embeddings": [[...], [...]], "model": "...", "usage": {...}}
        - Direct matrix:
            [[...], [...]]
        - EmbedResult-like with `.embeddings` attribute:
            result.embeddings -> [[...], [...]]

        This makes the adapter resilient to future translator customizations,
        as long as they expose an `embeddings` vector-of-vectors somewhere.
        """
        # Use structural pattern matching for cleaner type handling (Python 3.10+)
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

        We always normalize via `_coerce_embedding_matrix` and then:
        - If the matrix is empty → raise
        - If it has exactly one row → return that row
        - If it has multiple rows → return the first row but log a warning
        """
        matrix = CorpusLangChainEmbeddings._coerce_embedding_matrix(result)
        
        if not matrix:
            raise ValueError("Translator returned no embeddings for single-text input")

        if len(matrix) > 1:
            logger.warning(
                "Expected a single embedding for query, but got %d rows; "
                "using the first row.",
                len(matrix),
            )

        return matrix[0]

    def _get_timeout_from_context(self, core_ctx: Any) -> Optional[float]:
        """Extract timeout from core context, converting ms to seconds."""
        if hasattr(core_ctx, "deadline_ms") and core_ctx.deadline_ms is not None:
            return core_ctx.deadline_ms / 1000.0
        return None

    # ------------------------------------------------------------------ #
    # Async API
    # ------------------------------------------------------------------ #

    @_with_error_context("langchain", "embed_documents")
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
        _, op_ctx_dict, framework_ctx = self._build_contexts(
            config=config,
            model=model,
        )

        translated = await self._translator.arun_embed(
            raw_texts=texts,
            op_ctx=op_ctx_dict,
            framework_ctx=framework_ctx,
        )
        return self._coerce_embedding_matrix(translated)

    @_with_error_context("langchain", "embed_query")  
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
        _, op_ctx_dict, framework_ctx = self._build_contexts(
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
    # Sync API (via AsyncBridge)
    # ------------------------------------------------------------------ #

    @_with_error_context("langchain", "embed_documents")
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

        Uses `AsyncBridge` on top of the async translator path so:
        - We honor `deadline_ms` from the OperationContext as a timeout.
        - We avoid nested event loops in Jupyter / async environments.
        """
        core_ctx, op_ctx_dict, framework_ctx = self._build_contexts(
            config=config,
            model=model,
        )
        timeout = self._get_timeout_from_context(core_ctx)

        async def _coro() -> List[List[float]]:
            translated = await self._translator.arun_embed(
                raw_texts=texts,
                op_ctx=op_ctx_dict,
                framework_ctx=framework_ctx,
            )
            return self._coerce_embedding_matrix(translated)

        return AsyncBridge.run_async(_coro(), timeout=timeout)

    @_with_error_context("langchain", "embed_query")
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

        Uses `AsyncBridge` on top of the async translator path with respect
        to any `deadline_ms` set in the OperationContext.
        """
        core_ctx, op_ctx_dict, framework_ctx = self._build_contexts(
            config=config,
            model=model,
        )
        timeout = self._get_timeout_from_context(core_ctx)

        async def _coro() -> List[float]:
            translated = await self._translator.arun_embed(
                raw_texts=text,
                op_ctx=op_ctx_dict,
                framework_ctx=framework_ctx,
            )
            return self._coerce_embedding_vector(translated)

        return AsyncBridge.run_async(_coro(), timeout=timeout)


__all__ = [
    "CorpusLangChainEmbeddings",
]
