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
from functools import cached_property, wraps
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

# Type variables for decorators
T = TypeVar("T")

# ---------------------------------------------------------------------------
# Safe conditional import for Semantic Kernel base class
# ---------------------------------------------------------------------------

try:
    from semantic_kernel.connectors.ai.embeddings.embedding_generator_base import (
        EmbeddingGeneratorBase,
    )
    SEMANTIC_KERNEL_AVAILABLE = True
except ImportError:  # pragma: no cover - only used when SK isn't installed
    class EmbeddingGeneratorBase:  # type: ignore[no-redef]
        """Fallback base class when Semantic Kernel is not installed."""
        pass

    SEMANTIC_KERNEL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Error codes (kept in sync with other framework adapters)
# ---------------------------------------------------------------------------


class ErrorCodes:
    INVALID_EMBEDDING_RESULT = "INVALID_EMBEDDING_RESULT"
    EMPTY_EMBEDDING_RESULT = "EMPTY_EMBEDDING_RESULT"
    EMBEDDING_CONVERSION_ERROR = "EMBEDDING_CONVERSION_ERROR"
    SEMANTIC_KERNEL_CONTEXT_INVALID = "SEMANTIC_KERNEL_CONTEXT_INVALID"


class SemanticKernelContext(TypedDict, total=False):
    """Structured type for Semantic Kernel execution context."""
    plugin_name: Optional[str]
    function_name: Optional[str]
    kernel_id: Optional[str]
    memory_type: Optional[str]
    request_id: Optional[str]
    user_id: Optional[str]


def _create_error_context_decorator(
    operation: str,
    is_async: bool = False,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Factory for creating error context decorators with rich per-call metrics.
    """
    def decorator_factory(
        **static_context: Any,
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            if is_async:
                @wraps(func)
                async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
                    # Extract dynamic context from call
                    dynamic_context = _extract_dynamic_context(self, args, kwargs, operation)
                    full_context = {**static_context, **dynamic_context}

                    try:
                        return await func(self, *args, **kwargs)
                    except Exception as exc:  # noqa: BLE001
                        attach_context(
                            exc,
                            framework="semantic_kernel",
                            operation=f"embedding_{operation}",
                            **full_context,
                        )
                        raise

                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
                    # Extract dynamic context from call
                    dynamic_context = _extract_dynamic_context(self, args, kwargs, operation)
                    full_context = {**static_context, **dynamic_context}

                    try:
                        return func(self, *args, **kwargs)
                    except Exception as exc:  # noqa: BLE001
                        attach_context(
                            exc,
                            framework="semantic_kernel",
                            operation=f"embedding_{operation}",
                            **full_context,
                        )
                        raise

                return sync_wrapper

        return decorator
    return decorator_factory


def _extract_dynamic_context(
    instance: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    operation: str,
) -> Dict[str, Any]:
    """
    Extract rich dynamic context from method call for enhanced observability.
    """
    dynamic_ctx: Dict[str, Any] = {
        "model_id": getattr(instance, "model_id", "unknown"),
    }

    # Extract text-based metrics
    if operation in ["generate_embedding"] and args and isinstance(args[0], str):
        dynamic_ctx["text_len"] = len(args[0])
    elif operation in ["generate_embeddings"] and args and isinstance(args[0], list):
        dynamic_ctx["texts_count"] = len(args[0])
        # Also include empty text count for better diagnostics
        empty_count = sum(1 for text in args[0] if not text or not text.strip())
        if empty_count > 0:
            dynamic_ctx["empty_texts_count"] = empty_count

    # Extract Semantic Kernel-specific context from kwargs
    sk_context = kwargs.get("sk_context", {})
    if sk_context:
        if "plugin_name" in sk_context:
            dynamic_ctx["plugin_name"] = sk_context["plugin_name"]
        if "function_name" in sk_context:
            dynamic_ctx["function_name"] = sk_context["function_name"]
        if "kernel_id" in sk_context:
            dynamic_ctx["kernel_id"] = sk_context["kernel_id"]
        if "memory_type" in sk_context:
            dynamic_ctx["memory_type"] = sk_context["memory_type"]

    return dynamic_ctx


# Convenience decorators with rich context extraction
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

    Example:
    ```python
    import semantic_kernel as sk
    from semantic_kernel.planners import SequentialPlanner
    from corpus_sdk.embedding.framework_adapters.semantic_kernel import (
        CorpusSemanticKernelEmbeddings,
        register_with_semantic_kernel
    )

    # Create kernel and register Corpus embeddings
    kernel = sk.Kernel()
    embeddings = register_with_semantic_kernel(
        kernel=kernel,
        corpus_adapter=my_adapter,
        model_id="text-embedding-3-large",
        service_id="corpus_embeddings"
    )

    # Use with Semantic Kernel plugins and memory
    kernel.import_plugin_from_object(MyPlugin(), "my_plugin")

    # Embeddings automatically receive Semantic Kernel context during execution
    planner = SequentialPlanner(kernel)
    plan = await planner.create_plan_async("Summarize the document")
    result = await plan.invoke_async()

    # Direct embedding usage
    embedding = await embeddings.generate_embedding_async(
        text="Hello world",
        sk_context={
            "plugin_name": "text_processor",
            "function_name": "embed_text",
            "kernel_id": kernel.kernel_id
        }
    )
    ```

    Error Handling Example:
    ```python
    try:
        embeddings = embeddings.generate_embeddings(
            texts=documents,
            sk_context={
                "plugin_name": "document_processor",
                "function_name": "batch_embed"
            }
        )
    except Exception as e:
        # Rich error context automatically attached with plugin info, text counts, etc.
        logger.error("Embedding failed with context", exc_info=e)
    ```

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
        # Behavioral validation (duck-typed) instead of strict isinstance
        if not hasattr(corpus_adapter, "embed") or not callable(
            getattr(corpus_adapter, "embed", None)
        ):
            raise TypeError(
                "corpus_adapter must implement an EmbeddingProtocolV1-compatible "
                "interface with an 'embed' method"
            )

        if sk_config is not None and not isinstance(sk_config, dict):
            raise TypeError("sk_config must be a dictionary")

        super().__init__()  # EmbeddingGeneratorBase init (no-op for dummy fallback)

        self.corpus_adapter = corpus_adapter
        self.model_id = model_id
        self.batch_config = batch_config
        self.text_normalization_config = text_normalization_config
        self.sk_config = sk_config or {}

        logger.info(
            "CorpusSemanticKernelEmbeddings initialized with model_id=%s",
            model_id or "default",
        )

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
            framework="semantic_kernel",
            translator=None,  # use registry/default generic translator
            batch_config=self.batch_config,
            text_normalization_config=self.text_normalization_config,
        )
        logger.debug(
            "EmbeddingTranslator initialized for Semantic Kernel with model_id=%s",
            self.model_id or "default",
        )
        return translator

    def _build_contexts(
        self,
        *,
        sk_context: Optional[Mapping[str, Any]] = None,
        model_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[Optional[OperationContext], Dict[str, Any]]:
        """
        Build contexts for Semantic Kernel execution environment with comprehensive validation.

        Returns
        -------
        Tuple of:
        - core_ctx: core OperationContext or None if no/invalid context
        - framework_ctx: Semantic Kernel-specific context for translator
        """
        core_ctx: Optional[OperationContext] = None
        framework_ctx: Dict[str, Any] = {
            "framework": "semantic_kernel",
        }

        # Validate input type for sk_context
        if sk_context is not None:
            if not isinstance(sk_context, Mapping):
                logger.warning(
                    "[%s] sk_context should be a Mapping, got %s; ignoring context",
                    ErrorCodes.SEMANTIC_KERNEL_CONTEXT_INVALID,
                    type(sk_context).__name__,
                )
                sk_context = None
            else:
                # Validate structure for better observability
                self._validate_semantic_kernel_context_structure(sk_context)

        # Convert Semantic Kernel context to core OperationContext with defensive handling
        if sk_context is not None:
            try:
                core_ctx_candidate = context_from_semantic_kernel(sk_context)
                if isinstance(core_ctx_candidate, OperationContext):
                    core_ctx = core_ctx_candidate
                    logger.debug(
                        "Successfully created OperationContext from Semantic Kernel context "
                        "with plugin: %s, function: %s",
                        sk_context.get("plugin_name", "unknown"),
                        sk_context.get("function_name", "unknown"),
                    )
                else:
                    logger.warning(
                        "context_from_semantic_kernel returned non-OperationContext type: %s. "
                        "Proceeding with empty OperationContext.",
                        type(core_ctx_candidate).__name__,
                    )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Failed to create OperationContext from Semantic Kernel context: %s. "
                    "Proceeding with degraded context.",
                    e,
                )
                try:
                    attach_context(
                        e,
                        framework="semantic_kernel",
                        operation="context_build",
                        context_snapshot=dict(sk_context),
                    )
                except Exception:  # noqa: BLE001
                    # Do not mask original issues in context translation
                    pass

        # Framework-level context for Semantic Kernel-specific hints
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
            if "request_id" in sk_context:
                framework_ctx["request_id"] = sk_context["request_id"]
            if "user_id" in sk_context:
                framework_ctx["user_id"] = sk_context["user_id"]

        # Include any extra call-specific hints while preserving structure
        framework_ctx.update(kwargs)

        # Also expose the OperationContext itself to the translator for maximum fidelity
        if core_ctx is not None:
            framework_ctx["_operation_context"] = core_ctx

        return core_ctx, framework_ctx

    def _validate_semantic_kernel_context_structure(self, context: Mapping[str, Any]) -> None:
        """Validate Semantic Kernel context structure and log warnings for anomalies."""
        # Check for common Semantic Kernel context fields
        if not any(
            key in context
            for key in ["plugin_name", "function_name", "kernel_id", "memory_type"]
        ):
            logger.debug(
                "Semantic Kernel context missing common fields (plugin_name, function_name, etc.) - "
                "reduced context for embeddings"
            )

    def _coerce_embedding_matrix(self, result: Any) -> List[List[float]]:
        """
        Coerce translator result into a List[List[float]] embedding matrix with comprehensive validation.

        Supported result shapes:
        - {"embeddings": [[...], [...]], "model": "...", "usage": {...}}
        - {"embedding": [...]}  (single vector; wrapped into a single-row matrix)
        - Direct matrix: [[...], [...]]
        - EmbedResult-like with `.embeddings` or `.embedding` attribute
        """
        # Python 3.9-compatible variant (no match/case)
        if isinstance(result, Mapping):
            if "embeddings" in result:
                embeddings_obj: Any = result["embeddings"]
            elif "embedding" in result:
                embeddings_obj = [result["embedding"]]
            else:
                embeddings_obj = result
        elif hasattr(result, "embeddings"):
            embeddings_obj = getattr(result, "embeddings")
        elif hasattr(result, "embedding"):
            embeddings_obj = [getattr(result, "embedding")]
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
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
                raise TypeError(
                    f"[{ErrorCodes.INVALID_EMBEDDING_RESULT}] "
                    f"Expected each embedding row to be a sequence of numbers, "
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
        Coerce translator result for a single-text embed into List[float].
        """
        matrix = self._coerce_embedding_matrix(result)

        if len(matrix) > 1:
            logger.warning(
                "Expected a single embedding for query, but got %d rows; "
                "using the first row.",
                len(matrix),
            )

        return matrix[0]

    @property
    def embedding_dimension(self) -> int:
        """
        Get embedding dimension for proper zero vector fallback.

        Returns
        -------
        int
            Embedding dimension, with fallback to common default (768)
        """
        if hasattr(self.corpus_adapter, "get_embedding_dimension"):
            try:
                return int(self.corpus_adapter.get_embedding_dimension())
            except Exception as e:  # noqa: BLE001
                logger.debug("Failed to get embedding dimension from adapter: %s", e)

        # Common fallback dimension
        return 768

    def _handle_empty_text(self, text: str) -> List[float]:
        """
        Handle empty text by returning an appropriate zero vector.
        """
        logger.warning("Empty text provided for embedding, returning zero vector")
        dimension = self.embedding_dimension
        return [0.0] * dimension

    def _warn_if_extreme_batch(self, texts: Sequence[str], *, op_name: str) -> None:
        """
        Emit a soft warning if an extremely large batch is requested
        without an explicit BatchConfig.max_batch_size configured.

        This mirrors the behavior of other framework adapters (e.g., LangChain, LlamaIndex).
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

    def _embed_single_text(self, text: str, sk_context: Dict[str, Any]) -> List[float]:
        """Unified single text embedding implementation."""
        core_ctx, framework_ctx = self._build_contexts(
            sk_context=sk_context,
            **sk_context,
        )

        # Rich logging with dynamic context
        logger.debug(
            "Embedding single text for Semantic Kernel plugin: %s, function: %s",
            sk_context.get("plugin_name", "unknown"),
            sk_context.get("function_name", "unknown"),
        )

        translated = self._translator.embed(
            raw_texts=text,
            op_ctx=core_ctx,
            framework_ctx=framework_ctx,
        )
        return self._coerce_embedding_vector(translated)

    async def _aembed_single_text(self, text: str, sk_context: Dict[str, Any]) -> List[float]:
        """Unified async single text embedding implementation."""
        core_ctx, framework_ctx = self._build_contexts(
            sk_context=sk_context,
            **sk_context,
        )

        # Rich logging with dynamic context
        logger.debug(
            "Async embedding single text for Semantic Kernel plugin: %s, function: %s",
            sk_context.get("plugin_name", "unknown"),
            sk_context.get("function_name", "unknown"),
        )

        translated = await self._translator.arun_embed(
            raw_texts=text,
            op_ctx=core_ctx,
            framework_ctx=framework_ctx,
        )
        return self._coerce_embedding_vector(translated)

    def _embed_text_batch(
        self,
        texts: Sequence[str],
        sk_context: Dict[str, Any],
    ) -> List[List[float]]:
        """Unified batch text embedding implementation."""
        texts_list = list(texts)

        if not texts_list:
            logger.warning("Empty texts list provided for embedding")
            return []

        self._warn_if_extreme_batch(texts_list, op_name="generate_embeddings")

        # Filter out empty texts and handle them separately
        non_empty_texts = [t for t in texts_list if t and t.strip()]
        empty_indices = [i for i, t in enumerate(texts_list) if not t or not t.strip()]

        if not non_empty_texts:
            dimension = self.embedding_dimension
            return [[0.0] * dimension for _ in texts_list]

        core_ctx, framework_ctx = self._build_contexts(
            sk_context=sk_context,
            **sk_context,
        )

        # Rich logging with dynamic context
        logger.debug(
            "Embedding %d texts for Semantic Kernel plugin: %s, function: %s",
            len(texts_list),
            sk_context.get("plugin_name", "unknown"),
            sk_context.get("function_name", "unknown"),
        )

        translated = self._translator.embed(
            raw_texts=non_empty_texts,
            op_ctx=core_ctx,
            framework_ctx=framework_ctx,
        )
        embeddings = self._coerce_embedding_matrix(translated)

        if empty_indices:
            dimension = len(embeddings[0]) if embeddings else self.embedding_dimension
            result_embeddings: List[List[float]] = []
            non_empty_idx = 0
            for i in range(len(texts_list)):
                if i in empty_indices:
                    result_embeddings.append([0.0] * dimension)
                else:
                    result_embeddings.append(embeddings[non_empty_idx])
                    non_empty_idx += 1
            return result_embeddings

        return embeddings

    async def _aembed_text_batch(
        self,
        texts: Sequence[str],
        sk_context: Dict[str, Any],
    ) -> List[List[float]]:
        """Unified async batch text embedding implementation."""
        texts_list = list(texts)

        if not texts_list:
            logger.warning("Empty texts list provided for async embedding")
            return []

        self._warn_if_extreme_batch(texts_list, op_name="generate_embeddings_async")

        non_empty_texts = [t for t in texts_list if t and t.strip()]
        empty_indices = [i for i, t in enumerate(texts_list) if not t or not t.strip()]

        if not non_empty_texts:
            dimension = self.embedding_dimension
            return [[0.0] * dimension for _ in texts_list]

        core_ctx, framework_ctx = self._build_contexts(
            sk_context=sk_context,
            **sk_context,
        )

        # Rich logging with dynamic context
        logger.debug(
            "Async embedding %d texts for Semantic Kernel plugin: %s, function: %s",
            len(texts_list),
            sk_context.get("plugin_name", "unknown"),
            sk_context.get("function_name", "unknown"),
        )

        translated = await self._translator.arun_embed(
            raw_texts=non_empty_texts,
            op_ctx=core_ctx,
            framework_ctx=framework_ctx,
        )
        embeddings = self._coerce_embedding_matrix(translated)

        if empty_indices:
            dimension = len(embeddings[0]) if embeddings else self.embedding_dimension
            result_embeddings: List[List[float]] = []
            non_empty_idx = 0
            for i in range(len(texts_list)):
                if i in empty_indices:
                    result_embeddings.append([0.0] * dimension)
                else:
                    result_embeddings.append(embeddings[non_empty_idx])
                    non_empty_idx += 1
            return result_embeddings

        return embeddings

    # ------------------------------------------------------------------ #
    # Core Semantic Kernel EmbeddingGeneration Methods
    # ------------------------------------------------------------------ #

    @with_embedding_error_context("generate_embeddings")
    def generate_embeddings(
        self,
        texts: Sequence[str],
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
        context_dict: Dict[str, Any] = dict(sk_context or {})
        if model_id:
            context_dict["model_id"] = model_id
        context_dict.update(kwargs)

        return self._embed_text_batch(texts, context_dict)

    @with_async_embedding_error_context("generate_embeddings")
    async def generate_embeddings_async(
        self,
        texts: Sequence[str],
        *,
        sk_context: Optional[Mapping[str, Any]] = None,
        model_id: Optional[str] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """
        Async embedding generation for multiple texts.

        Used by Semantic Kernel's async plugin functions and memory operations.
        """
        context_dict: Dict[str, Any] = dict(sk_context or {})
        if model_id:
            context_dict["model_id"] = model_id
        context_dict.update(kwargs)

        return await self._aembed_text_batch(texts, context_dict)

    @with_embedding_error_context("generate_embedding")
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

        context_dict: Dict[str, Any] = dict(sk_context or {})
        if model_id:
            context_dict["model_id"] = model_id
        context_dict.update(kwargs)

        return self._embed_single_text(text, context_dict)

    @with_async_embedding_error_context("generate_embedding")
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

        context_dict: Dict[str, Any] = dict(sk_context or {})
        if model_id:
            context_dict["model_id"] = model_id
        context_dict.update(kwargs)

        return await self._aembed_single_text(text, context_dict)

    # ------------------------------------------------------------------ #
    # Convenience aliases for broader compatibility
    # ------------------------------------------------------------------ #

    @with_embedding_error_context("embed_documents")
    def embed_documents(
        self,
        texts: Sequence[str],
        **kwargs: Any,
    ) -> List[List[float]]:
        """Alias for document embedding."""
        return self.generate_embeddings(texts, **kwargs)

    @with_embedding_error_context("embed_query")
    def embed_query(
        self,
        text: str,
        **kwargs: Any,
    ) -> List[float]:
        """Alias for query embedding."""
        return self.generate_embedding(text, **kwargs)

    @with_async_embedding_error_context("embed_documents")
    async def aembed_documents(
        self,
        texts: Sequence[str],
        **kwargs: Any,
    ) -> List[List[float]]:
        """Async alias for document embedding."""
        return await self.generate_embeddings_async(texts, **kwargs)

    @with_async_embedding_error_context("embed_query")
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

    Example:
    ```python
    import semantic_kernel as sk
    from corpus_sdk.embedding.framework_adapters.semantic_kernel import register_with_semantic_kernel

    kernel = sk.Kernel()
    embeddings = register_with_semantic_kernel(
        kernel=kernel,
        corpus_adapter=my_adapter,
        model_id="text-embedding-3-large",
        service_id="corpus_embeddings"
    )

    # Now available as kernel service for plugins and planners
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
        Model identifier for embedding operations
    **kwargs:
        Additional arguments for CorpusSemanticKernelEmbeddings

    Returns
    -------
    CorpusSemanticKernelEmbeddings
        Registered embedding service instance
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


__all__ = [
    "CorpusSemanticKernelEmbeddings",
    "SemanticKernelContext",
    "register_with_semantic_kernel",
    "ErrorCodes",
    "with_embedding_error_context",
    "with_async_embedding_error_context",
]
