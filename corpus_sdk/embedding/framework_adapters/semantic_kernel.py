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

Resilience (retries, caching, rate limiting, etc.) is expected to be provided
by the underlying adapter, typically a BaseEmbeddingAdapter subclass.
"""

from __future__ import annotations

import asyncio
import logging
import threading
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
from corpus_sdk.embedding.framework_adapters.common.framework_utils import (
    CoercionErrorCodes,
    coerce_embedding_matrix,
    coerce_embedding_vector,
    warn_if_extreme_batch,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Framework identity & version
# ---------------------------------------------------------------------------

_FRAMEWORK_NAME = "semantic_kernel"

try:  # Best-effort Semantic Kernel version detection
    import semantic_kernel as _semantic_kernel  # type: ignore

    _FRAMEWORK_VERSION: Optional[str] = getattr(
        _semantic_kernel,
        "__version__",
        None,
    )
except Exception:  # noqa: BLE001
    _FRAMEWORK_VERSION = None

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
    """
    Error code constants for the Semantic Kernel embedding adapter.

    This is a simple namespace for framework-specific codes. The shared
    coercion helpers use `EMBEDDING_COERCION_ERROR_CODES`, which is a
    `CoercionErrorCodes` instance derived from these values.
    """

    INVALID_EMBEDDING_RESULT = "INVALID_EMBEDDING_RESULT"
    EMPTY_EMBEDDING_RESULT = "EMPTY_EMBEDDING_RESULT"
    EMBEDDING_CONVERSION_ERROR = "EMBEDDING_CONVERSION_ERROR"
    SEMANTIC_KERNEL_CONTEXT_INVALID = "SEMANTIC_KERNEL_CONTEXT_INVALID"
    SEMANTIC_KERNEL_CONFIG_INVALID = "SEMANTIC_KERNEL_CONFIG_INVALID"
    INVALID_TEXT_TYPE = "INVALID_TEXT_TYPE"


# Coercion configuration for the common embedding utils
EMBEDDING_COERCION_ERROR_CODES: CoercionErrorCodes = CoercionErrorCodes(
    invalid_result=ErrorCodes.INVALID_EMBEDDING_RESULT,
    empty_result=ErrorCodes.EMPTY_EMBEDDING_RESULT,
    conversion_error=ErrorCodes.EMBEDDING_CONVERSION_ERROR,
    framework_label=_FRAMEWORK_NAME,
)


class SemanticKernelContext(TypedDict, total=False):
    """Structured type for Semantic Kernel execution context."""
    plugin_name: Optional[str]
    function_name: Optional[str]
    kernel_id: Optional[str]
    memory_type: Optional[str]
    request_id: Optional[str]
    user_id: Optional[str]


class SemanticKernelConfig(TypedDict, total=False):
    """
    Structured configuration for Semantic Kernel adapter toggles.

    Notes
    -----
    This stays intentionally small and “adapter-only”:
    - It should not leak Semantic Kernel internals, only behavior toggles.
    """
    strict_text_types: bool
    fallback_to_simple_context: bool
    enable_context_propagation: bool


# ---------------------------------------------------------------------------
# Small shared helpers (alignment + safety)
# ---------------------------------------------------------------------------


def _safe_snapshot(obj: Any) -> Dict[str, Any]:
    """Best-effort snapshot for error context attachment (never throws)."""
    if obj is None:
        return {}
    try:
        return dict(obj)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001
        return {"repr": repr(obj)}


def _validate_text_is_string(text: Any, *, op_name: str) -> None:
    """Fail fast with a clear error when strict text typing is enabled."""
    if not isinstance(text, str):
        raise TypeError(
            f"[{ErrorCodes.INVALID_TEXT_TYPE}] {op_name} expects a string, got {type(text).__name__}",
        )


def _validate_texts_are_strings(texts: Sequence[Any], *, op_name: str) -> None:
    """Fail fast with a clear error when strict text typing is enabled."""
    for i, t in enumerate(texts):
        if not isinstance(t, str):
            raise TypeError(
                f"[{ErrorCodes.INVALID_TEXT_TYPE}] {op_name} expects Sequence[str]; "
                f"element at index {i} is {type(t).__name__}",
            )


# ---------------------------------------------------------------------------
# Error-context decorators with dynamic context extraction
# ---------------------------------------------------------------------------


def _create_error_context_decorator(
    operation: str,
    is_async: bool = False,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Factory for creating error context decorators with rich per-call metrics.

    Parity note:
    - These decorators always attach `error_codes` so coercion failures
      and adapter failures share the same structured code map.
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


def _extract_dynamic_context(
    instance: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    operation: str,
) -> Dict[str, Any]:
    """
    Extract rich dynamic context from method call for enhanced observability.

    Captures:
    - model_id from the embedding instance
    - framework_version
    - text_len / texts_count / empty_texts_count
    - Semantic Kernel routing fields (plugin_name, function_name, kernel_id, etc.)
    """
    dynamic_ctx: Dict[str, Any] = {
        "model_id": getattr(instance, "model_id", "unknown"),
        "framework_version": _FRAMEWORK_VERSION,
    }

    # Text / batch metrics
    if operation == "generate_embedding" and args:
        if isinstance(args[0], str):
            dynamic_ctx["text_len"] = len(args[0])
    elif operation == "generate_embeddings" and args and isinstance(args[0], Sequence):
        texts_seq = args[0]
        dynamic_ctx["texts_count"] = len(texts_seq)
        empty_count = sum(
            1 for text in texts_seq if not isinstance(text, str) or not text.strip(),
        )
        if empty_count > 0:
            dynamic_ctx["empty_texts_count"] = empty_count

    # Semantic Kernel-specific context (loop style for parity + maintainability)
    sk_context = kwargs.get("sk_context") or {}
    if isinstance(sk_context, Mapping) and sk_context:
        for key in (
            "plugin_name",
            "function_name",
            "kernel_id",
            "memory_type",
            "request_id",
            "user_id",
        ):
            if key in sk_context:
                dynamic_ctx[key] = sk_context[key]

    return dynamic_ctx


def with_embedding_error_context(
    operation: str,
    **static_context: Any,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for sync methods with rich dynamic context extraction."""
    static_context.setdefault("error_codes", EMBEDDING_COERCION_ERROR_CODES)
    static_context.setdefault("framework_version", _FRAMEWORK_VERSION)
    return _create_error_context_decorator(operation, is_async=False)(**static_context)


def with_async_embedding_error_context(
    operation: str,
    **static_context: Any,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for async methods with rich dynamic context extraction."""
    static_context.setdefault("error_codes", EMBEDDING_COERCION_ERROR_CODES)
    static_context.setdefault("framework_version", _FRAMEWORK_VERSION)
    return _create_error_context_decorator(operation, is_async=True)(**static_context)


# ---------------------------------------------------------------------------
# Main Semantic Kernel adapter
# ---------------------------------------------------------------------------


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
        semantic_kernel_config: Optional[SemanticKernelConfig] = None,
        *,
        embedding_dimension: Optional[int] = None,
    ) -> None:
        """
        Initialize Corpus Semantic Kernel Embeddings.
        """
        # Behavioral validation (duck-typed) instead of strict isinstance
        if not hasattr(corpus_adapter, "embed") or not callable(
            getattr(corpus_adapter, "embed", None),
        ):
            raise TypeError(
                "corpus_adapter must implement an EmbeddingProtocolV1-compatible "
                "interface with an 'embed' method",
            )

        super().__init__()  # EmbeddingGeneratorBase init (no-op for dummy fallback)

        self.corpus_adapter = corpus_adapter
        self.model_id = model_id
        self.batch_config = batch_config
        self.text_normalization_config = text_normalization_config

        self.semantic_kernel_config: SemanticKernelConfig = self._validate_semantic_kernel_config(
            semantic_kernel_config or {},
        )

        self._embedding_dimension_override = embedding_dimension

        # Translator lifecycle (thread-safe lazy init)
        self._translator_lock = threading.Lock()
        self._translator_instance: Optional[EmbeddingTranslator] = None

        # Enforce known embedding dimension to avoid incorrect fallbacks
        if (
            not hasattr(self.corpus_adapter, "get_embedding_dimension")
            and self._embedding_dimension_override is None
        ):
            raise ValueError(
                "Embedding dimension is unknown. Either implement "
                "`get_embedding_dimension()` on the corpus_adapter or pass "
                "`embedding_dimension=...` to CorpusSemanticKernelEmbeddings.",
            )

        # Cache a stable dimension hint for observability (never guesses)
        self._embedding_dim_hint: Optional[int] = None
        try:
            self._embedding_dim_hint = int(self.embedding_dimension)
        except Exception:  # noqa: BLE001
            # Dimension enforcement happens above; this is purely a best-effort hint.
            self._embedding_dim_hint = None

        logger.info(
            "CorpusSemanticKernelEmbeddings initialized with model_id=%s, "
            "framework_version=%s, embedding_dimension=%s",
            model_id or "default",
            _FRAMEWORK_VERSION or "unknown",
            self._embedding_dimension_override,
        )

    # ------------------------------------------------------------------ #
    # Configuration validation
    # ------------------------------------------------------------------ #

    def _validate_semantic_kernel_config(
        self,
        config: SemanticKernelConfig,
    ) -> SemanticKernelConfig:
        """Validate and normalize Semantic Kernel adapter config with sensible defaults."""
        validated: SemanticKernelConfig = dict(config)

        validated.setdefault("strict_text_types", True)
        validated.setdefault("fallback_to_simple_context", True)
        validated.setdefault("enable_context_propagation", True)

        # Bool coercion for robustness
        for key in (
            "strict_text_types",
            "fallback_to_simple_context",
            "enable_context_propagation",
        ):
            try:
                validated[key] = bool(validated[key])
            except Exception:  # noqa: BLE001
                raise TypeError(
                    f"[{ErrorCodes.SEMANTIC_KERNEL_CONFIG_INVALID}] "
                    f"{key} must be coercible to bool",
                ) from None

        return validated

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @property
    def _translator(self) -> EmbeddingTranslator:
        """
        Lazily construct and cache the `EmbeddingTranslator` (thread-safe).

        Note:
        - `cached_property` is not guaranteed thread-safe under concurrent access.
          This explicit lock avoids double initialization in multi-threaded hosts.
        """
        existing = self._translator_instance
        if isinstance(existing, EmbeddingTranslator):
            return existing

        with self._translator_lock:
            existing = self._translator_instance
            if isinstance(existing, EmbeddingTranslator):
                return existing

            translator = create_embedding_translator(
                adapter=self.corpus_adapter,
                framework=_FRAMEWORK_NAME,
                translator=None,  # use registry/default generic translator
                batch_config=self.batch_config,
                text_normalization_config=self.text_normalization_config,
            )
            self._translator_instance = translator

            logger.debug(
                "EmbeddingTranslator initialized for Semantic Kernel with model_id=%s",
                self.model_id or "default",
            )
            return translator

    def _build_contexts(
        self,
        *,
        sk_context: Optional[SemanticKernelContext] = None,
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
            "framework": _FRAMEWORK_NAME,
            "semantic_kernel_config": dict(self.semantic_kernel_config),
            "error_codes": EMBEDDING_COERCION_ERROR_CODES,
        }
        if _FRAMEWORK_VERSION is not None:
            framework_ctx["framework_version"] = _FRAMEWORK_VERSION
        if self._embedding_dim_hint is not None:
            framework_ctx["embedding_dim_hint"] = self._embedding_dim_hint

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
                self._validate_semantic_kernel_context_structure(sk_context)

        # Convert SK context to OperationContext with defensive handling
        if sk_context is not None:
            try:
                core_ctx_candidate = context_from_semantic_kernel(sk_context)
                if isinstance(core_ctx_candidate, OperationContext):
                    # Enrich OperationContext with framework metadata
                    attrs = dict(getattr(core_ctx_candidate, "attrs", {}) or {})
                    attrs.setdefault("framework", _FRAMEWORK_NAME)
                    if _FRAMEWORK_VERSION is not None:
                        attrs.setdefault("framework_version", _FRAMEWORK_VERSION)

                    core_ctx = OperationContext(
                        request_id=core_ctx_candidate.request_id,
                        idempotency_key=core_ctx_candidate.idempotency_key,
                        deadline_ms=core_ctx_candidate.deadline_ms,
                        traceparent=core_ctx_candidate.traceparent,
                        tenant=core_ctx_candidate.tenant,
                        attrs=attrs,
                    )

                    logger.debug(
                        "Successfully created OperationContext from Semantic Kernel context "
                        "with plugin: %s, function: %s (framework_version=%s)",
                        sk_context.get("plugin_name", "unknown"),
                        sk_context.get("function_name", "unknown"),
                        _FRAMEWORK_VERSION or "unknown",
                    )
                else:
                    logger.warning(
                        "context_from_semantic_kernel returned non-OperationContext type: %s. "
                        "Proceeding without OperationContext.",
                        type(core_ctx_candidate).__name__,
                    )
                    if self.semantic_kernel_config["fallback_to_simple_context"]:
                        core_ctx = OperationContext()
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Failed to create OperationContext from Semantic Kernel context: %s. "
                    "Proceeding with degraded context.",
                    e,
                )
                try:
                    attach_context(
                        e,
                        framework=_FRAMEWORK_NAME,
                        operation="context_build",
                        context_snapshot=_safe_snapshot(sk_context),
                        framework_version=_FRAMEWORK_VERSION,
                        error_codes=EMBEDDING_COERCION_ERROR_CODES,
                        semantic_kernel_config=dict(self.semantic_kernel_config),
                    )
                except Exception:  # noqa: BLE001
                    # Do not mask original issues in context translation
                    pass
                if self.semantic_kernel_config["fallback_to_simple_context"]:
                    core_ctx = OperationContext()

        # Framework-level context for Semantic Kernel-specific hints
        effective_model_id = model_id or self.model_id
        if effective_model_id:
            framework_ctx["model_id"] = effective_model_id

        # Add Semantic Kernel-specific context (loop style)
        if sk_context:
            for key in (
                "plugin_name",
                "function_name",
                "kernel_id",
                "memory_type",
                "request_id",
                "user_id",
            ):
                if key in sk_context:
                    framework_ctx[key] = sk_context[key]

        # Include any extra call-specific hints (exclude private keys)
        framework_ctx.update({k: v for k, v in kwargs.items() if not k.startswith("_")})

        # Also expose the OperationContext itself (when enabled)
        if core_ctx is not None and self.semantic_kernel_config["enable_context_propagation"]:
            framework_ctx["_operation_context"] = core_ctx

        return core_ctx, framework_ctx

    def _validate_semantic_kernel_context_structure(
        self,
        context: Mapping[str, Any],
    ) -> None:
        """Validate Semantic Kernel context structure and log warnings for anomalies."""
        if not any(
            key in context
            for key in (
                "plugin_name",
                "function_name",
                "kernel_id",
                "memory_type",
                "request_id",
            )
        ):
            logger.debug(
                "Semantic Kernel context missing common fields "
                "(plugin_name, function_name, etc.) - reduced context for embeddings",
            )

    def _coerce_embedding_matrix(self, result: Any) -> List[List[float]]:
        """
        Coerce translator result into a List[List[float]] embedding matrix.
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
        """
        return coerce_embedding_vector(
            result=result,
            framework=_FRAMEWORK_NAME,
            error_codes=EMBEDDING_COERCION_ERROR_CODES,
            logger=logger,
        )

    @property
    def embedding_dimension(self) -> int:
        """
        Get embedding dimension for proper zero vector fallback.

        Never guesses – requires either:
        - adapter.get_embedding_dimension(), or
        - explicit embedding_dimension override passed at init.
        """
        if hasattr(self.corpus_adapter, "get_embedding_dimension"):
            try:
                return int(self.corpus_adapter.get_embedding_dimension())
            except Exception as e:  # noqa: BLE001
                logger.debug(
                    "Failed to get embedding dimension from adapter: %s", e,
                )
                if self._embedding_dimension_override is not None:
                    return int(self._embedding_dimension_override)
                raise

        if self._embedding_dimension_override is not None:
            return int(self._embedding_dimension_override)

        # Should be unreachable due to __init__ guard, but keep a hard failure
        raise RuntimeError(
            "Embedding dimension is unknown. Adapter does not expose "
            "`get_embedding_dimension()` and no `embedding_dimension` "
            "override was provided.",
        )

    def _handle_empty_text(self, text: str) -> List[float]:
        """
        Handle empty text by returning an appropriate zero vector.
        """
        dim = self.embedding_dimension
        logger.warning(
            "Empty text provided for embedding, returning zero vector (dimension=%d)",
            dim,
        )
        return [0.0] * dim

    def _warn_if_extreme_batch(self, texts: Sequence[str], *, op_name: str) -> None:
        """
        Soft warning for extremely large batches.
        """
        warn_if_extreme_batch(
            framework=_FRAMEWORK_NAME,
            texts=texts,
            op_name=op_name,
            batch_config=self.batch_config,
            logger=logger,
        )

    # ------------------------------------------------------------------ #
    # Health / capabilities passthrough (parity with other adapters)
    # ------------------------------------------------------------------ #

    @with_embedding_error_context("capabilities")
    def capabilities(self) -> Mapping[str, Any]:
        """Best-effort capabilities passthrough to the underlying adapter (sync)."""
        if hasattr(self.corpus_adapter, "capabilities"):
            return self.corpus_adapter.capabilities()  # type: ignore[no-any-return]
        return {}

    @with_async_embedding_error_context("capabilities_async")
    async def acapabilities(self) -> Mapping[str, Any]:
        """Best-effort capabilities passthrough to the underlying adapter (async)."""
        if hasattr(self.corpus_adapter, "acapabilities"):
            return await self.corpus_adapter.acapabilities()  # type: ignore[no-any-return]
        if hasattr(self.corpus_adapter, "capabilities"):
            return await asyncio.to_thread(self.corpus_adapter.capabilities)  # type: ignore[arg-type]
        return {}

    @with_embedding_error_context("health")
    def health(self) -> Mapping[str, Any]:
        """Best-effort health passthrough to the underlying adapter (sync)."""
        if hasattr(self.corpus_adapter, "health"):
            return self.corpus_adapter.health()  # type: ignore[no-any-return]
        return {}

    @with_async_embedding_error_context("health_async")
    async def ahealth(self) -> Mapping[str, Any]:
        """Best-effort health passthrough to the underlying adapter (async)."""
        if hasattr(self.corpus_adapter, "ahealth"):
            return await self.corpus_adapter.ahealth()  # type: ignore[no-any-return]
        if hasattr(self.corpus_adapter, "health"):
            return await asyncio.to_thread(self.corpus_adapter.health)  # type: ignore[arg-type]
        return {}

    # ------------------------------------------------------------------ #
    # Unified internal embedding helpers
    # ------------------------------------------------------------------ #

    def _embed_single_text(
        self,
        text: str,
        sk_context: SemanticKernelContext,
        *,
        model_id: Optional[str] = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[float]:
        """Unified single text embedding implementation."""
        core_ctx, framework_ctx = self._build_contexts(
            sk_context=sk_context,
            model_id=model_id,
            **(extra_kwargs or {}),
        )

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

    async def _aembed_single_text(
        self,
        text: str,
        sk_context: SemanticKernelContext,
        *,
        model_id: Optional[str] = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[float]:
        """Unified async single text embedding implementation."""
        core_ctx, framework_ctx = self._build_contexts(
            sk_context=sk_context,
            model_id=model_id,
            **(extra_kwargs or {}),
        )

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
        sk_context: SemanticKernelContext,
        *,
        model_id: Optional[str] = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[List[float]]:
        """Unified batch text embedding implementation."""
        texts_list = list(texts)

        if not texts_list:
            logger.warning("Empty texts list provided for embedding")
            return []

        if self.semantic_kernel_config["strict_text_types"]:
            _validate_texts_are_strings(texts_list, op_name="generate_embeddings")

        self._warn_if_extreme_batch(texts_list, op_name="generate_embeddings")

        non_empty_texts = [t for t in texts_list if t.strip()]
        empty_indices = [i for i, t in enumerate(texts_list) if not t.strip()]

        if not non_empty_texts:
            dim = self.embedding_dimension
            return [[0.0] * dim for _ in texts_list]

        core_ctx, framework_ctx = self._build_contexts(
            sk_context=sk_context,
            model_id=model_id,
            **(extra_kwargs or {}),
        )

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
            dim = len(embeddings[0]) if embeddings else self.embedding_dimension
            result_embeddings: List[List[float]] = []
            non_empty_idx = 0
            for i in range(len(texts_list)):
                if i in empty_indices:
                    result_embeddings.append([0.0] * dim)
                else:
                    result_embeddings.append(embeddings[non_empty_idx])
                    non_empty_idx += 1
            return result_embeddings

        return embeddings

    async def _aembed_text_batch(
        self,
        texts: Sequence[str],
        sk_context: SemanticKernelContext,
        *,
        model_id: Optional[str] = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[List[float]]:
        """Unified async batch text embedding implementation."""
        texts_list = list(texts)

        if not texts_list:
            logger.warning("Empty texts list provided for async embedding")
            return []

        if self.semantic_kernel_config["strict_text_types"]:
            _validate_texts_are_strings(texts_list, op_name="generate_embeddings_async")

        self._warn_if_extreme_batch(texts_list, op_name="generate_embeddings_async")

        non_empty_texts = [t for t in texts_list if t.strip()]
        empty_indices = [i for i, t in enumerate(texts_list) if not t.strip()]

        if not non_empty_texts:
            dim = self.embedding_dimension
            return [[0.0] * dim for _ in texts_list]

        core_ctx, framework_ctx = self._build_contexts(
            sk_context=sk_context,
            model_id=model_id,
            **(extra_kwargs or {}),
        )

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
            dim = len(embeddings[0]) if embeddings else self.embedding_dimension
            result_embeddings: List[List[float]] = []
            non_empty_idx = 0
            for i in range(len(texts_list)):
                if i in empty_indices:
                    result_embeddings.append([0.0] * dim)
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
        sk_context: Optional[SemanticKernelContext] = None,
        model_id: Optional[str] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """
        Sync embedding generation for multiple texts.
        """
        context_dict: SemanticKernelContext = dict(sk_context or {})  # type: ignore[assignment]
        return self._embed_text_batch(
            texts,
            context_dict,
            model_id=model_id,
            extra_kwargs=dict(kwargs),
        )

    @with_async_embedding_error_context("generate_embeddings")
    async def generate_embeddings_async(
        self,
        texts: Sequence[str],
        *,
        sk_context: Optional[SemanticKernelContext] = None,
        model_id: Optional[str] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """
        Async embedding generation for multiple texts.
        """
        context_dict: SemanticKernelContext = dict(sk_context or {})  # type: ignore[assignment]
        return await self._aembed_text_batch(
            texts,
            context_dict,
            model_id=model_id,
            extra_kwargs=dict(kwargs),
        )

    @with_embedding_error_context("generate_embedding")
    def generate_embedding(
        self,
        text: str,
        *,
        sk_context: Optional[SemanticKernelContext] = None,
        model_id: Optional[str] = None,
        **kwargs: Any,
    ) -> List[float]:
        """
        Sync embedding generation for a single text.
        """
        if self.semantic_kernel_config["strict_text_types"]:
            _validate_text_is_string(text, op_name="generate_embedding")

        if not text or not text.strip():
            return self._handle_empty_text(text)

        context_dict: SemanticKernelContext = dict(sk_context or {})  # type: ignore[assignment]
        return self._embed_single_text(
            text,
            context_dict,
            model_id=model_id,
            extra_kwargs=dict(kwargs),
        )

    @with_async_embedding_error_context("generate_embedding")
    async def generate_embedding_async(
        self,
        text: str,
        *,
        sk_context: Optional[SemanticKernelContext] = None,
        model_id: Optional[str] = None,
        **kwargs: Any,
    ) -> List[float]:
        """
        Async embedding generation for a single text.
        """
        if self.semantic_kernel_config["strict_text_types"]:
            _validate_text_is_string(text, op_name="generate_embedding_async")

        if not text or not text.strip():
            return self._handle_empty_text(text)

        context_dict: SemanticKernelContext = dict(sk_context or {})  # type: ignore[assignment]
        return await self._aembed_single_text(
            text,
            context_dict,
            model_id=model_id,
            extra_kwargs=dict(kwargs),
        )

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
                "Manual service configuration may be required.",
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
    "SemanticKernelConfig",
    "register_with_semantic_kernel",
    "ErrorCodes",
    "with_embedding_error_context",
    "with_async_embedding_error_context",
]
