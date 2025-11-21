# corpus_sdk/embedding/framework_adapters/semantic_kernel.py
# SPDX-License-Identifier: Apache-2.0

"""
Semantic Kernel adapter for Corpus Embedding protocol.

This module exposes Corpus `EmbeddingProtocolV1` implementations as
Semantic Kernel embedding services, with:

- Full compatibility with Semantic Kernel's embedding service patterns
- Support for Semantic Kernel's plugin system and function chaining
- Context normalization using `context_translation.from_semantic_kernel`
- Framework-agnostic orchestration via `EmbeddingTranslator`
- Async → sync bridging handled in the common embedding layer
- Rich error context attachment for observability

The design integrates with Semantic Kernel's planner and plugin architecture
while maintaining the protocol-first Corpus embedding stack.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import cached_property
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Protocol,
)

from corpus_sdk.core.context_translation import (
    from_semantic_kernel as context_from_semantic_kernel,
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

# ---------------------------------------------------------------------------
# Safe conditional import for Semantic Kernel base class
# ---------------------------------------------------------------------------

try:
    from semantic_kernel.connectors.ai.embeddings.embedding_generator_base import (
        EmbeddingGeneratorBase,
    )
except ImportError:  # pragma: no cover - only used when SK isn't installed
    class EmbeddingGeneratorBase:  # type: ignore[no-redef]
        """Fallback base class when Semantic Kernel is not installed."""
        pass


# ---------------------------------------------------------------------------
# Error codes (kept in sync with other framework adapters)
# ---------------------------------------------------------------------------

class ErrorCodes:
    INVALID_EMBEDDING_RESULT = "INVALID_EMBEDDING_RESULT"
    EMPTY_EMBEDDING_RESULT = "EMPTY_EMBEDDING_RESULT"
    EMBEDDING_CONVERSION_ERROR = "EMBEDDING_CONVERSION_ERROR"


# ---------------------------------------------------------------------------
# Error Context Decorator (Eliminates Boilerplate)
# ---------------------------------------------------------------------------

@contextmanager
def _embedding_error_context(
    operation: str,
    *,
    text_len: Optional[int] = None,
    texts_count: Optional[int] = None,
    framework_ctx: Optional[Dict[str, Any]] = None,
    model_id: Optional[str] = None,
) -> None:
    """
    Context manager for consistent error context attachment in embedding operations.

    Eliminates repetitive try/except boilerplate across all embedding methods.
    """
    try:
        yield
    except Exception as exc:  # noqa: BLE001
        # Build error context from parameters
        error_context: Dict[str, Any] = {
            "semantic_kernel_model_id": model_id,
        }

        if text_len is not None:
            error_context["text_len"] = text_len
        if texts_count is not None:
            error_context["texts_count"] = texts_count
        if framework_ctx:
            for key in ("plugin_name", "function_name", "kernel_id", "memory_type"):
                if key in framework_ctx:
                    error_context[key] = framework_ctx[key]

        # Attach structured context without masking the original error
        try:
            attach_context(
                exc,
                framework="semantic_kernel",
                operation=f"embedding_{operation}",
                **error_context,
            )
        except Exception:
            # Never mask the original error if context attachment fails
            pass
        raise


class SemanticKernelMemoryBuilderProtocol(Protocol):
    """
    Structural protocol describing the minimal MemoryBuilder interface
    used by this module.

    This avoids a hard dependency on Semantic Kernel's concrete types
    while still giving helper functions a precise return type.
    """

    def build(self) -> Any:
        """
        Build and return a memory instance.

        The concrete memory type is application-specific and therefore
        left as `Any` here.
        """
        ...


class CorpusSemanticKernelEmbeddings(EmbeddingGeneratorBase):
    """
    Semantic Kernel embedding service backed by a Corpus `EmbeddingProtocolV1` adapter.

    Semantic Kernel-Specific Responsibilities
    -----------------------------------------
    - Implement Semantic Kernel's embedding generator interface
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
    """

    def __init__(
        self,
        corpus_adapter: EmbeddingProtocolV1,
        model_id: Optional[str] = None,
        batch_config: Optional[BatchConfig] = None,
        text_normalization_config: Optional[TextNormalizationConfig] = None,
        sk_config: Optional[Dict[str, Any]] = None,
    ) -> None:
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
        if not isinstance(corpus_adapter, EmbeddingProtocolV1):
            # Runtime compatibility guard; keeps misconfigured services from registering.
            raise TypeError("corpus_adapter must implement EmbeddingProtocolV1")

        if sk_config is not None and not isinstance(sk_config, dict):
            raise TypeError("sk_config must be a dictionary")

        super().__init__()  # EmbeddingGeneratorBase init (no-op for dummy fallback)

        self.corpus_adapter = corpus_adapter
        self.model_id = model_id
        self.batch_config = batch_config
        self.text_normalization_config = text_normalization_config
        self.sk_config = sk_config or {}

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
    ) -> Tuple[OperationContext, Dict[str, Any], Dict[str, Any]]:
        """
        Build contexts for Semantic Kernel execution environment.

        Uses the existing `context_from_semantic_kernel` implementation.

        Parameters
        ----------
        sk_context:
            Optional Semantic Kernel context containing kernel,
            function, and plugin information.
        model_id:
            Optional per-call model override.
        **kwargs:
            Additional framework-level hints to be passed through to the
            translator as part of `framework_ctx`.

        Returns
        -------
        Tuple of:
        - `core_ctx`: core OperationContext (from existing context translation)
        - `op_ctx_dict`: normalized dict for embedding layer
        - `framework_ctx`: Semantic Kernel-specific context for translator
        """
        if sk_context is not None and not isinstance(sk_context, Mapping):
            logger.warning("sk_context should be a Mapping, got %s", type(sk_context))
            sk_context = None

        core_ctx: OperationContext = context_from_semantic_kernel(sk_context)

        # Normalized dict for embedding OperationContext reconstruction
        op_ctx_dict: Dict[str, Any] = {}
        if hasattr(core_ctx, "to_dict"):
            op_ctx_dict = core_ctx.to_dict()
        elif hasattr(core_ctx, "__dict__"):
            op_ctx_dict = core_ctx.__dict__

        # Framework-level context for Semantic Kernel-specific hints
        framework_ctx: Dict[str, Any] = {
            "framework": "semantic_kernel",
        }

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

        Supported result shapes:
        - {"embeddings": [[...], [...]], "model": "...", "usage": {...}}
        - Direct matrix: [[...], [...]]
        - EmbedResult-like with `.embeddings` attribute
        """
        # Python 3.9-compatible variant (no match/case)
        if isinstance(result, Mapping) and "embeddings" in result:
            embeddings_obj: Any = result["embeddings"]
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

    @staticmethod
    def _coerce_embedding_vector(result: Any) -> List[float]:
        """
        Coerce translator result for a single-text embed into List[float].
        """
        matrix = CorpusSemanticKernelEmbeddings._coerce_embedding_matrix(result)

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

    @property
    def _get_embedding_dimension(self) -> int:
        """
        Get embedding dimension for proper zero vector fallback.

        Returns
        -------
        int
            Embedding dimension, with fallback to common default (768)
        """
        if hasattr(self.corpus_adapter, "get_embedding_dimension"):
            try:
                return self.corpus_adapter.get_embedding_dimension()  # type: ignore[no-any-return]
            except Exception as e:  # noqa: BLE001
                logger.debug("Failed to get embedding dimension from adapter: %s", e)

        # Common fallback dimension
        return 768

    def _handle_empty_text(self, text: str) -> List[float]:
        """
        Handle empty text by returning an appropriate zero vector.
        """
        logger.warning("Empty text provided for embedding, returning zero vector")
        dimension = self._get_embedding_dimension
        return [0.0] * dimension

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

        Primary method used by Semantic Kernel's memory systems and plugin
        functions for batch embedding generation.
        """
        if not texts:
            logger.warning("generate_embeddings called with empty texts list")
            return []

        # Filter out empty texts and handle them separately
        non_empty_texts = [t for t in texts if t and t.strip()]
        empty_indices = [i for i, t in enumerate(texts) if not t or not t.strip()]

        if not non_empty_texts:
            dimension = self._get_embedding_dimension
            return [[0.0] * dimension for _ in texts]

        _, op_ctx_dict, framework_ctx = self._build_contexts(
            sk_context=sk_context,
            model_id=model_id,
            **kwargs,
        )

        with _embedding_error_context(
            "generate_embeddings",
            texts_count=len(texts),
            framework_ctx=framework_ctx,
            model_id=model_id or self.model_id,
        ):
            translated = self._translator.embed(
                raw_texts=non_empty_texts,
                op_ctx=op_ctx_dict,
                framework_ctx=framework_ctx,
            )
            embeddings = self._coerce_embedding_matrix(translated)

            if empty_indices:
                dimension = len(embeddings[0]) if embeddings else self._get_embedding_dimension
                result_embeddings: List[List[float]] = []
                non_empty_idx = 0
                for i in range(len(texts)):
                    if i in empty_indices:
                        result_embeddings.append([0.0] * dimension)
                    else:
                        result_embeddings.append(embeddings[non_empty_idx])
                        non_empty_idx += 1
                return result_embeddings

            return embeddings

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

        Used by Semantic Kernel's async plugin functions and memory operations.
        """
        if not texts:
            logger.warning("generate_embeddings_async called with empty texts list")
            return []

        non_empty_texts = [t for t in texts if t and t.strip()]
        empty_indices = [i for i, t in enumerate(texts) if not t or not t.strip()]

        if not non_empty_texts:
            dimension = self._get_embedding_dimension
            return [[0.0] * dimension for _ in texts]

        _, op_ctx_dict, framework_ctx = self._build_contexts(
            sk_context=sk_context,
            model_id=model_id,
            **kwargs,
        )

        with _embedding_error_context(
            "generate_embeddings_async",
            texts_count=len(texts),
            framework_ctx=framework_ctx,
            model_id=model_id or self.model_id,
        ):
            translated = await self._translator.arun_embed(
                raw_texts=non_empty_texts,
                op_ctx=op_ctx_dict,
                framework_ctx=framework_ctx,
            )
            embeddings = self._coerce_embedding_matrix(translated)

            if empty_indices:
                dimension = len(embeddings[0]) if embeddings else self._get_embedding_dimension
                result_embeddings: List[List[float]] = []
                non_empty_idx = 0
                for i in range(len(texts)):
                    if i in empty_indices:
                        result_embeddings.append([0.0] * dimension)
                    else:
                        result_embeddings.append(embeddings[non_empty_idx])
                        non_empty_idx += 1
                return result_embeddings

            return embeddings

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

        Used by Semantic Kernel for single-text embedding in memory operations
        and similarity calculations.
        """
        if not text or not text.strip():
            return self._handle_empty_text(text)

        _, op_ctx_dict, framework_ctx = self._build_contexts(
            sk_context=sk_context,
            model_id=model_id,
            **kwargs,
        )

        with _embedding_error_context(
            "generate_embedding",
            text_len=len(text),
            framework_ctx=framework_ctx,
            model_id=model_id or self.model_id,
        ):
            translated = self._translator.embed(
                raw_texts=text,
                op_ctx=op_ctx_dict,
                framework_ctx=framework_ctx,
            )
            return self._coerce_embedding_vector(translated)

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

        Used by Semantic Kernel's async memory operations and plugin functions.
        """
        if not text or not text.strip():
            return self._handle_empty_text(text)

        _, op_ctx_dict, framework_ctx = self._build_contexts(
            sk_context=sk_context,
            model_id=model_id,
            **kwargs,
        )

        with _embedding_error_context(
            "generate_embedding_async",
            text_len=len(text),
            framework_ctx=framework_ctx,
            model_id=model_id or self.model_id,
        ):
            translated = await self._translator.arun_embed(
                raw_texts=text,
                op_ctx=op_ctx_dict,
                framework_ctx=framework_ctx,
            )
            return self._coerce_embedding_vector(translated)

    # ------------------------------------------------------------------ #
    # Convenience aliases for broader compatibility
    # ------------------------------------------------------------------ #

    def embed_documents(
        self,
        texts: List[str],
        **kwargs: Any,
    ) -> List[List[float]]:
        """Alias for document embedding."""
        return self.generate_embeddings(texts, **kwargs)

    def embed_query(
        self,
        text: str,
        **kwargs: Any,
    ) -> List[float]:
        """Alias for query embedding."""
        return self.generate_embedding(text, **kwargs)

    async def aembed_documents(
        self,
        texts: List[str],
        **kwargs: Any,
    ) -> List[List[float]]:
        """Async alias for document embedding."""
        return await self.generate_embeddings_async(texts, **kwargs)

    async def aembed_query(
        self,
        text: str,
        **kwargs: Any,
    ) -> List[float]:
        """Async alias for query embedding."""
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

    This provides integration with Semantic Kernel's service registration system.
    """
    if kernel is None:
        raise ValueError("kernel cannot be None")

    embeddings = CorpusSemanticKernelEmbeddings(
        corpus_adapter=corpus_adapter,
        model_id=model_id,
        **kwargs,
    )

    registration_successful = False
    try:
        # Try multiple registration patterns for different SK versions
        registration_methods = [
            getattr(kernel, "add_service", None),
            getattr(kernel, "register_embedding_generation", None),
            getattr(kernel, "register_service", None),
        ]

        for method in registration_methods:
            if method is None:
                continue
            try:
                method(embeddings, service_id=service_id)
                registration_successful = True
                logger.debug("Registered with Semantic Kernel using %s", method.__name__)
                break
            except (TypeError, AttributeError) as e:  # noqa: BLE001
                logger.debug("Registration method %s failed: %s", method.__name__, e)
                continue

        if not registration_successful:
            logger.warning(
                "No compatible Semantic Kernel service registration method found. "
                "Manual service configuration may be required."
            )

    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Failed to auto-register with Semantic Kernel: %s. "
            "Manual service configuration may be required.",
            e,
        )

    logger.info(
        "Corpus Semantic Kernel embeddings registered: %s",
        service_id or "default service",
    )

    return embeddings


def create_memory_builder(
    corpus_adapter: EmbeddingProtocolV1,
    model_id: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[CorpusSemanticKernelEmbeddings, SemanticKernelMemoryBuilderProtocol]:
    """
    Create a Semantic Kernel memory builder with Corpus embeddings.

    Returns the embedding service and a MemoryBuilder-like instance.
    """
    try:
        import semantic_kernel as sk
        from semantic_kernel.memory import MemoryBuilder
    except ImportError as e:  # noqa: BLE001
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
    memory_builder: SemanticKernelMemoryBuilderProtocol = MemoryBuilder(kernel=kernel)

    logger.info(
        "Semantic Kernel memory builder created with Corpus embeddings"
    )

    return embeddings, memory_builder


__all__ = [
    "CorpusSemanticKernelEmbeddings",
    "register_with_semantic_kernel",
    "create_memory_builder",
    "SemanticKernelMemoryBuilderProtocol",
    "ErrorCodes",
]
