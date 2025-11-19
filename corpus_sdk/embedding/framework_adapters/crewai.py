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
from functools import cached_property
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from corpus_sdk.core.async_bridge import AsyncBridge
from corpus_sdk.core.context_translation import (
    from_crewai as context_from_crewai,
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


class CorpusCrewAIEmbeddings:
    """
    CrewAI embedding service backed by a Corpus `EmbeddingProtocolV1` adapter.

    Responsibilities (this layer)
    -----------------------------
    - Provide embeddings for CrewAI agents via `embed_documents` and `embed_query`
    - Integrate with CrewAI agent `embedder` attribute and knowledge sources
    - Derive `OperationContext` from CrewAI execution context
    - Build framework_ctx for model selection and CrewAI-specific hints
    - Use `EmbeddingTranslator` for core embedding logic
    - Handle sync/async execution patterns compatible with CrewAI flows
    - Attach structured error context for CrewAI workflows

    Non-responsibilities
    --------------------
    - Text normalization, batching logic, token-aware batching
    - Provider-specific behavior (rate limits, retries, etc.)

    All of those live in:
    - `corpus_sdk.embedding.framework_adapters.common.embedding_translation`
    - Concrete `EmbeddingProtocolV1` adapter implementations.

    Attributes
    ----------
    corpus_adapter:
        Underlying Corpus embedding adapter implementing `EmbeddingProtocolV1`.

    model:
        Optional default model identifier. Can be overridden via CrewAI
        agent configuration or execution context.

    batch_config:
        Optional `BatchConfig` to control batching behavior.

    text_normalization_config:
        Optional `TextNormalizationConfig` to control whitespace cleanup,
        truncation, casing, encoding, etc.

    crewai_config:
        Optional CrewAI-specific configuration for agent context and
        knowledge source integration.
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

        Uses `cached_property` for thread safety and performance.
        """
        return create_embedding_translator(
            adapter=self.corpus_adapter,
            framework="crewai",
            translator=None,  # use registry/default generic translator
            batch_config=self.batch_config,
            text_normalization_config=self.text_normalization_config,
        )

    def _build_contexts(
        self,
        *,
        crewai_context: Optional[Mapping[str, Any]] = None,
        model: Optional[str] = None,
    ) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        """
        Build contexts for CrewAI execution environment.

        Parameters
        ----------
        crewai_context:
            Optional CrewAI execution context containing agent info,
            task details, and workflow metadata.
        model:
            Optional per-call model override.

        Returns
        -------
        Tuple of:
        - `core_ctx`: core OperationContext (from context_translation)
        - `op_ctx_dict`: normalized dict for embedding layer
        - `framework_ctx`: CrewAI-specific context for translator
        """
        # Convert CrewAI context to core OperationContext
        core_ctx = context_from_crewai(crewai_context)

        # Normalized dict for embedding OperationContext reconstruction
        op_ctx_dict: Dict[str, Any] = core_ctx.to_dict()

        # Framework-level context for CrewAI-specific hints
        framework_ctx: Dict[str, Any] = {}
        effective_model = model or self.model
        if effective_model:
            framework_ctx["model"] = effective_model

        # Add CrewAI-specific context for knowledge sources and agent roles
        if crewai_context:
            framework_ctx["crewai_agent_role"] = crewai_context.get("agent_role")
            framework_ctx["crewai_task_id"] = crewai_context.get("task_id")
            framework_ctx["crewai_workflow"] = crewai_context.get("workflow")

        return core_ctx, op_ctx_dict, framework_ctx

    @staticmethod
    def _coerce_embedding_matrix(result: Any) -> List[List[float]]:
        """
        Coerce translator result into a List[List[float]] embedding matrix.

        Supports the same result formats as the LangChain adapter:
        - {"embeddings": [[...], [...]], "model": "...", "usage": {...}}
        - Direct matrix: [[...], [...]]
        - EmbedResult-like with `.embeddings` attribute
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
        matrix = CorpusCrewAIEmbeddings._coerce_embedding_matrix(result)

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
    # Core Embedding API (CrewAI Compatible)
    # ------------------------------------------------------------------ #

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

        This method is designed to be called by CrewAI agents during
        RAG operations and knowledge source processing.

        Parameters
        ----------
        texts:
            List of documents to embed.
        crewai_context:
            Optional CrewAI execution context containing agent role,
            task information, and workflow metadata.
        model:
            Optional per-call model override.
        """
        core_ctx, op_ctx_dict, framework_ctx = self._build_contexts(
            crewai_context=crewai_context,
            model=model,
        )
        timeout = self._get_timeout_from_context(core_ctx)

        async def _coro() -> List[List[float]]:
            try:
                translated = await self._translator.arun_embed(
                    raw_texts=texts,
                    op_ctx=op_ctx_dict,
                    framework_ctx=framework_ctx,
                )
                return self._coerce_embedding_matrix(translated)
            except Exception as exc:  # noqa: BLE001
                try:
                    attach_context(
                        exc,
                        framework="crewai",
                        embedding_operation="embed_documents",
                        texts_count=len(texts),
                        agent_role=framework_ctx.get("crewai_agent_role"),
                        task_id=framework_ctx.get("crewai_task_id"),
                        request_id=getattr(core_ctx, "request_id", None),
                        tenant=getattr(core_ctx, "tenant", None),
                    )
                except Exception:
                    pass  # Never mask original error
                raise

        return AsyncBridge.run_async(_coro(), timeout=timeout)

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

        Used by CrewAI for query understanding, retrieval, and
        agent decision-making processes.

        Parameters
        ----------
        text:
            Query text to embed.
        crewai_context:
            Optional CrewAI execution context.
        model:
            Optional per-call model override.
        """
        core_ctx, op_ctx_dict, framework_ctx = self._build_contexts(
            crewai_context=crewai_context,
            model=model,
        )
        timeout = self._get_timeout_from_context(core_ctx)

        async def _coro() -> List[float]:
            try:
                translated = await self._translator.arun_embed(
                    raw_texts=text,
                    op_ctx=op_ctx_dict,
                    framework_ctx=framework_ctx,
                )
                return self._coerce_embedding_vector(translated)
            except Exception as exc:  # noqa: BLE001
                try:
                    attach_context(
                        exc,
                        framework="crewai",
                        embedding_operation="embed_query",
                        text_len=len(text or ""),
                        agent_role=framework_ctx.get("crewai_agent_role"),
                        task_id=framework_ctx.get("crewai_task_id"),
                        request_id=getattr(core_ctx, "request_id", None),
                        tenant=getattr(core_ctx, "tenant", None),
                    )
                except Exception:
                    pass
                raise

        return AsyncBridge.run_async(_coro(), timeout=timeout)

    # ------------------------------------------------------------------ #
    # Async API for CrewAI Flows
    # ------------------------------------------------------------------ #

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

        Designed for use with CrewAI's async flows and event-driven workflows.

        Parameters
        ----------
        texts:
            List of documents to embed.
        crewai_context:
            Optional CrewAI execution context.
        model:
            Optional per-call model override.
        """
        core_ctx, op_ctx_dict, framework_ctx = self._build_contexts(
            crewai_context=crewai_context,
            model=model,
        )

        try:
            translated = await self._translator.arun_embed(
                raw_texts=texts,
                op_ctx=op_ctx_dict,
                framework_ctx=framework_ctx,
            )
            return self._coerce_embedding_matrix(translated)
        except Exception as exc:  # noqa: BLE001
            try:
                attach_context(
                    exc,
                    framework="crewai",
                    embedding_operation="embed_documents",
                    texts_count=len(texts),
                    agent_role=framework_ctx.get("crewai_agent_role"),
                    task_id=framework_ctx.get("crewai_task_id"),
                    request_id=getattr(core_ctx, "request_id", None),
                    tenant=getattr(core_ctx, "tenant", None),
                )
            except Exception:
                pass
            raise

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

        Used in CrewAI's asynchronous workflows and flow-based executions.

        Parameters
        ----------
        text:
            Query text to embed.
        crewai_context:
            Optional CrewAI execution context.
        model:
            Optional per-call model override.
        """
        core_ctx, op_ctx_dict, framework_ctx = self._build_contexts(
            crewai_context=crewai_context,
            model=model,
        )

        try:
            translated = await self._translator.arun_embed(
                raw_texts=text,
                op_ctx=op_ctx_dict,
                framework_ctx=framework_ctx,
            )
            return self._coerce_embedding_vector(translated)
        except Exception as exc:  # noqa: BLE001
            try:
                attach_context(
                    exc,
                    framework="crewai",
                    embedding_operation="embed_query",
                    text_len=len(text or ""),
                    agent_role=framework_ctx.get("crewai_agent_role"),
                    task_id=framework_ctx.get("crewai_task_id"),
                    request_id=getattr(core_ctx, "request_id", None),
                    tenant=getattr(core_ctx, "tenant", None),
                )
            except Exception:
                pass
            raise


# Convenience function for CrewAI integration
def create_crewai_embedder(
    corpus_adapter: EmbeddingProtocolV1,
    model: Optional[str] = None,
    **kwargs: Any,
) -> CorpusCrewAIEmbeddings:
    """
    Create a CrewAI-compatible embedder for use with Agent embedder attribute.

    Example usage:
    ```python
    from crewai import Agent
    from corpus_sdk.embedding.framework_adapters.crewai import create_crewai_embedder

    # Create embedder
    embedder = create_crewai_embedder(
        corpus_adapter=my_adapter,
        model="text-embedding-3-large"
    )

    # Use with CrewAI agent
    agent = Agent(
        role="Researcher",
        goal="Research latest AI developments",
        backstory="Expert research analyst",
        embedder=embedder,
        tools=[...]
    )
    ```
    """
    return CorpusCrewAIEmbeddings(
        corpus_adapter=corpus_adapter,
        model=model,
        **kwargs
    )


__all__ = [
    "CorpusCrewAIEmbeddings",
    "create_crewai_embedder",
]
