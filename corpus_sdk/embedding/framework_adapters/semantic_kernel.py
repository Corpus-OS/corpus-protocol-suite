# corpus_sdk/embedding/framework_adapters/semantic_kernel.py
# SPDX-License-Identifier: Apache-2.0

"""
Semantic Kernel adapter for Corpus Embedding protocol.

This module exposes Corpus `EmbeddingProtocolV1` implementations as
`semantic_kernel.embeddings.EmbeddingGenerationBase` services, with:

- Full compatibility with Semantic Kernel's AI service registration
- Support for Semantic Kernel's plugin system and function chaining
- Context normalization using existing `context_translation.from_semantic_kernel`
- Framework-agnostic orchestration via `EmbeddingTranslator`
- Async → sync bridging using `AsyncBridge`
- Rich error context attachment for observability

The design integrates seamlessly with Semantic Kernel's planner and
plugin architecture while maintaining the protocol-first Corpus embedding stack.
"""

from __future__ import annotations

import logging
from functools import cached_property
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from corpus_sdk.core.async_bridge import AsyncBridge
from corpus_sdk.core.context_translation import (
    from_semantic_kernel as context_from_semantic_kernel,  # Using existing implementation
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


class CorpusSemanticKernelEmbeddings:
    """
    Semantic Kernel embedding service backed by a Corpus `EmbeddingProtocolV1` adapter.

    Semantic Kernel-Specific Responsibilities
    -----------------------------------------
    - Implement Semantic Kernel's `EmbeddingGenerationBase` interface
    - Support Semantic Kernel's plugin system and function chaining
    - Integrate with Semantic Kernel's planner and memory systems
    - Provide embeddings for semantic memory and text similarity
    - Work with Semantic Kernel's AI service registration pattern

    Non-responsibilities
    --------------------
    - Plugin management and function registration (handled by Semantic Kernel)
    - Planner orchestration and goal decomposition (handled by Semantic Kernel)
    - Memory storage and retrieval (handled by Semantic Kernel memory stores)

    All embedding logic lives in:
    - `corpus_sdk.embedding.framework_adapters.common.embedding_translation`
    - Concrete `EmbeddingProtocolV1` adapter implementations.

    Attributes
    ----------
    corpus_adapter:
        Underlying Corpus embedding adapter implementing `EmbeddingProtocolV1`.

    model_id:
        Optional model identifier used in Semantic Kernel contexts.

    batch_config:
        Optional `BatchConfig` to control batching behavior.

    text_normalization_config:
        Optional `TextNormalization_config` to control whitespace cleanup,
        truncation, casing, encoding, etc.

    sk_config:
        Optional Semantic Kernel-specific configuration for plugin
        context and service registration.
    """

    def __init__(
        self,
        corpus_adapter: EmbeddingProtocolV1,
        model_id: Optional[str] = None,
        batch_config: Optional[BatchConfig] = None,
        text_normalization_config: Optional[TextNormalizationConfig] = None,
        sk_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Corpus Semantic Kernel Embeddings.

        Parameters
        ----------
        corpus_adapter:
            Corpus embedding protocol adapter
        model_id:
            Model identifier for embedding operations
        batch_config:
            Batching configuration for embedding requests
        text_normalization_config:
            Text normalization settings
        sk_config:
            Semantic Kernel-specific configuration
        """
        self.corpus_adapter = corpus_adapter
        self.model_id = model_id
        self.batch_config = batch_config
        self.text_normalization_config = text_normalization_config
        self.sk_config = sk_config or {}

    # ------------------------------------------------------------------ #
    # Core Semantic Kernel EmbeddingGeneration Interface
    # ------------------------------------------------------------------ #

    @cached_property
    def _translator(self) -> EmbeddingTranslator:
        """
        Lazily construct and cache the `EmbeddingTranslator`.

        Uses `cached_property` for thread safety and performance.
        """
        return create_embedding_translator(
            adapter=self.corpus_adapter,
            framework="semantic_kernel",
            translator=None,  # use registry/default generic translator
            batch_config=self.batch_config,
            text_normalization_config=self.text_normalization_config,
        )

    def _build_contexts(
        self,
        *,
        sk_context: Optional[Mapping[str, Any]] = None,
        model_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        """
        Build contexts for Semantic Kernel execution environment.

        Uses the existing `context_from_semantic_kernel` implementation.

        Parameters
        ----------
        sk_context:
            Optional Semantic Kernel context containing kernel,
            function, and plugin information.

        Returns
        -------
        Tuple of:
        - `core_ctx`: core OperationContext (from existing context translation)
        - `op_ctx_dict`: normalized dict for embedding layer
        - `framework_ctx`: Semantic Kernel-specific context for translator
        """
        # Use existing context translation implementation
        core_ctx = context_from_semantic_kernel(sk_context)

        # Normalized dict for embedding OperationContext reconstruction
        op_ctx_dict: Dict[str, Any] = core_ctx.to_dict()

        # Framework-level context for Semantic Kernel-specific hints
        framework_ctx: Dict[str, Any] = {
            "framework": "semantic_kernel",
        }

        # Add model information if available
        effective_model_id = model_id or self.model_id
        if effective_model_id:
            framework_ctx["model_id"] = effective_model_id

        # Add Semantic Kernel-specific context for plugins and functions
        if sk_context:
            if "plugin_name" in sk_context:
                framework_ctx["plugin_name"] = sk_context["plugin_name"]
            if "function_name" in sk_context:
                framework_ctx["function_name"] = sk_context["function_name"]
            if "kernel_id" in sk_context:
                framework_ctx["kernel_id"] = sk_context["kernel_id"]
            if "memory_type" in sk_context:
                framework_ctx["memory_type"] = sk_context["memory_type"]

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
        matrix = CorpusSemanticKernelEmbeddings._coerce_embedding_matrix(result)

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
    # Core Semantic Kernel EmbeddingGeneration Methods
    # ------------------------------------------------------------------ #

    def generate_embeddings(
        self,
        texts: List[str],
        *,
        sk_context: Optional[Mapping[str, Any]] = None,
        model_id: Optional[str] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """
        Sync embedding generation for multiple texts.

        This is the primary method used by Semantic Kernel's memory
        systems and plugin functions for batch embedding generation.

        Parameters
        ----------
        texts:
            List of texts to embed
        sk_context:
            Optional Semantic Kernel context containing kernel and plugin info
        model_id:
            Optional per-call model override
        """
        core_ctx, op_ctx_dict, framework_ctx = self._build_contexts(
            sk_context=sk_context,
            model_id=model_id,
            **kwargs,
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
                        framework="semantic_kernel",
                        embedding_operation="generate_embeddings",
                        texts_count=len(texts),
                        plugin_name=framework_ctx.get("plugin_name"),
                        function_name=framework_ctx.get("function_name"),
                        kernel_id=framework_ctx.get("kernel_id"),
                        request_id=getattr(core_ctx, "request_id", None),
                        tenant=getattr(core_ctx, "tenant", None),
                    )
                except Exception:
                    pass  # Never mask original error
                raise

        return AsyncBridge.run_async(_coro(), timeout=timeout)

    async def generate_embeddings_async(
        self,
        texts: List[str],
        *,
        sk_context: Optional[Mapping[str, Any]] = None,
        model_id: Optional[str] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """
        Async embedding generation for multiple texts.

        This is used by Semantic Kernel's async plugin functions
        and memory operations.

        Parameters
        ----------
        texts:
            List of texts to embed
        sk_context:
            Optional Semantic Kernel context
        model_id:
            Optional per-call model override
        """
        core_ctx, op_ctx_dict, framework_ctx = self._build_contexts(
            sk_context=sk_context,
            model_id=model_id,
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
            try:
                attach_context(
                    exc,
                    framework="semantic_kernel",
                    embedding_operation="generate_embeddings_async",
                    texts_count=len(texts),
                    plugin_name=framework_ctx.get("plugin_name"),
                    function_name=framework_ctx.get("function_name"),
                    kernel_id=framework_ctx.get("kernel_id"),
                    request_id=getattr(core_ctx, "request_id", None),
                    tenant=getattr(core_ctx, "tenant", None),
                )
            except Exception:
                pass
            raise

    # ------------------------------------------------------------------ #
    # Semantic Kernel Memory Compatibility Methods
    # ------------------------------------------------------------------ #

    def generate_embedding(
        self,
        text: str,
        *,
        sk_context: Optional[Mapping[str, Any]] = None,
        model_id: Optional[str] = None,
        **kwargs: Any,
    ) -> List[float]:
        """
        Sync embedding generation for a single text.

        Used by Semantic Kernel for single text embedding in
        memory operations and similarity calculations.

        Parameters
        ----------
        text:
            Text to embed
        sk_context:
            Optional Semantic Kernel context
        model_id:
            Optional per-call model override
        """
        core_ctx, op_ctx_dict, framework_ctx = self._build_contexts(
            sk_context=sk_context,
            model_id=model_id,
            **kwargs,
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
                        framework="semantic_kernel",
                        embedding_operation="generate_embedding",
                        text_len=len(text or ""),
                        plugin_name=framework_ctx.get("plugin_name"),
                        function_name=framework_ctx.get("function_name"),
                        kernel_id=framework_ctx.get("kernel_id"),
                        request_id=getattr(core_ctx, "request_id", None),
                        tenant=getattr(core_ctx, "tenant", None),
                    )
                except Exception:
                    pass
                raise

        return AsyncBridge.run_async(_coro(), timeout=timeout)

    async def generate_embedding_async(
        self,
        text: str,
        *,
        sk_context: Optional[Mapping[str, Any]] = None,
        model_id: Optional[str] = None,
        **kwargs: Any,
    ) -> List[float]:
        """
        Async embedding generation for a single text.

        Used by Semantic Kernel's async memory operations and
        plugin functions.

        Parameters
        ----------
        text:
            Text to embed
        sk_context:
            Optional Semantic Kernel context
        model_id:
            Optional per-call model override
        """
        core_ctx, op_ctx_dict, framework_ctx = self._build_contexts(
            sk_context=sk_context,
            model_id=model_id,
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
            try:
                attach_context(
                    exc,
                    framework="semantic_kernel",
                    embedding_operation="generate_embedding_async",
                    text_len=len(text or ""),
                    plugin_name=framework_ctx.get("plugin_name"),
                    function_name=framework_ctx.get("function_name"),
                    kernel_id=framework_ctx.get("kernel_id"),
                    request_id=getattr(core_ctx, "request_id", None),
                    tenant=getattr(core_ctx, "tenant", None),
                )
            except Exception:
                pass
            raise

    # ------------------------------------------------------------------ #
    # Alternative Method Names for Broader Compatibility
    # ------------------------------------------------------------------ #

    def embed_documents(
        self,
        texts: List[str],
        **kwargs: Any,
    ) -> List[List[float]]:
        """
        Alternative method name for document embedding.

        Provides compatibility with various Semantic Kernel patterns
        and third-party integrations.

        Parameters
        ----------
        texts:
            List of documents to embed
        """
        return self.generate_embeddings(texts, **kwargs)

    def embed_query(
        self,
        text: str,
        **kwargs: Any,
    ) -> List[float]:
        """
        Alternative method name for query embedding.

        Provides compatibility with various Semantic Kernel patterns
        and third-party integrations.

        Parameters
        ----------
        text:
            Query text to embed
        """
        return self.generate_embedding(text, **kwargs)

    async def aembed_documents(
        self,
        texts: List[str],
        **kwargs: Any,
    ) -> List[List[float]]:
        """
        Async alternative for document embedding.

        Parameters
        ----------
        texts:
            List of documents to embed
        """
        return await self.generate_embeddings_async(texts, **kwargs)

    async def aembed_query(
        self,
        text: str,
        **kwargs: Any,
    ) -> List[float]:
        """
        Async alternative for query embedding.

        Parameters
        ----------
        text:
            Query text to embed
        """
        return await self.generate_embedding_async(text, **kwargs)


# ------------------------------------------------------------------ #
# Semantic Kernel Service Registration Helpers
# ------------------------------------------------------------------ #

def register_with_semantic_kernel(
    kernel: Any,
    corpus_adapter: EmbeddingProtocolV1,
    service_id: Optional[str] = None,
    model_id: Optional[str] = None,
    **kwargs: Any,
) -> CorpusSemanticKernelEmbeddings:
    """
    Register Corpus embeddings as a service with Semantic Kernel.

    This provides seamless integration with Semantic Kernel's
    service registration system.

    Example usage:
    ```python
    import semantic_kernel as sk
    from corpus_sdk.embedding.framework_adapters.semantic_kernel import register_with_semantic_kernel

    # Create kernel
    kernel = sk.Kernel()

    # Register embedding service
    embedder = register_with_semantic_kernel(
        kernel=kernel,
        corpus_adapter=my_adapter,
        service_id="corpus_embeddings",
        model_id="text-embedding-3-large"
    )

    # Now use with Semantic Kernel memory and plugins
    memory_builder = sk.MemoryBuilder(kernel=kernel)
    ```

    Parameters
    ----------
    kernel:
        Semantic Kernel instance
    corpus_adapter:
        Corpus embedding protocol adapter
    service_id:
        Optional service identifier for Semantic Kernel
    model_id:
        Optional model identifier for embedding operations
    **kwargs:
        Additional arguments for CorpusSemanticKernelEmbeddings

    Returns
    -------
    CorpusSemanticKernelEmbeddings
        Registered embedding service
    """
    embeddings = CorpusSemanticKernelEmbeddings(
        corpus_adapter=corpus_adapter,
        model_id=model_id,
        **kwargs,
    )

    # Register with Semantic Kernel's service collection
    try:
        # For newer Semantic Kernel versions with service collection
        if hasattr(kernel, 'add_service'):
            kernel.add_service(embeddings, service_id=service_id)
        # For older versions or alternative registration patterns
        elif hasattr(kernel, 'register_embedding_generation'):
            kernel.register_embedding_generation(embeddings, service_id=service_id)
        else:
            logger.warning(
                "Semantic Kernel service registration not available. "
                "Manual service configuration required."
            )
    except Exception as e:
        logger.warning(
            f"Failed to auto-register with Semantic Kernel: {e}. "
            f"Manual service configuration required."
        )

    logger.info(
        f"Corpus Semantic Kernel embeddings registered: {service_id or 'default service'}"
    )

    return embeddings


def create_memory_builder(
    corpus_adapter: EmbeddingProtocolV1,
    model_id: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[CorpusSemanticKernelEmbeddings, Any]:
    """
    Create a Semantic Kernel memory builder with Corpus embeddings.

    This provides a convenient way to set up Semantic Kernel memory
    with Corpus embeddings in a single function call.

    Example usage:
    ```python
    from corpus_sdk.embedding.framework_adapters.semantic_kernel import create_memory_builder

    # Create memory builder with Corpus embeddings
    embedder, memory_builder = create_memory_builder(
        corpus_adapter=my_adapter,
        model_id="text-embedding-3-large"
    )

    # Build memory with your preferred storage
    memory = memory_builder.build()
    ```

    Parameters
    ----------
    corpus_adapter:
        Corpus embedding protocol adapter
    model_id:
        Optional model identifier for embedding operations
    **kwargs:
        Additional arguments for CorpusSemanticKernelEmbeddings

    Returns
    -------
    Tuple[CorpusSemanticKernelEmbeddings, Any]
        Embedding service and memory builder instance
    """
    try:
        import semantic_kernel as sk
        from semantic_kernel.memory import MemoryBuilder
    except ImportError:
        logger.error(
            "Semantic Kernel not installed. Please install with: pip install semantic-kernel"
        )
        raise

    # Create kernel
    kernel = sk.Kernel()

    # Register embedding service
    embeddings = register_with_semantic_kernel(
        kernel=kernel,
        corpus_adapter=corpus_adapter,
        model_id=model_id,
        **kwargs,
    )

    # Create memory builder
    memory_builder = MemoryBuilder(kernel=kernel)

    logger.info(
        "Semantic Kernel memory builder created with Corpus embeddings"
    )

    return embeddings, memory_builder


__all__ = [
    "CorpusSemanticKernelEmbeddings",
    "register_with_semantic_kernel",
    "create_memory_builder",
]
