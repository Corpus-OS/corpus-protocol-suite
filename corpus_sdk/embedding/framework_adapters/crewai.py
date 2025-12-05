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
- Async → sync bridging handled in the common embedding layer
- Rich error context attachment for observability

The design follows CrewAI's adapter patterns while maintaining the
protocol-first Corpus embedding stack.

Resilience (retries, caching, rate limiting, etc.) is expected to be provided
by the underlying adapter, typically a BaseEmbeddingAdapter subclass.
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
    Protocol,
    TypeVar,
    Callable,
    TypedDict,
)

from corpus_sdk.core.context_translation import (
    from_crewai as context_from_crewai,
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

_FRAMEWORK_NAME = "crewai"


class ErrorCodes:
    """
    Error code constants for CrewAI embedding adapter.

    This is a simple namespace for framework-specific codes. The shared
    coercion helpers use `EMBEDDING_COERCION_ERROR_CODES`, which is a
    `CoercionErrorCodes` instance derived from these values.
    """

    # Coercion-level (used by framework_utils)
    INVALID_EMBEDDING_RESULT = "INVALID_EMBEDDING_RESULT"
    EMPTY_EMBEDDING_RESULT = "EMPTY_EMBEDDING_RESULT"
    EMBEDDING_CONVERSION_ERROR = "EMBEDDING_CONVERSION_ERROR"

    # CrewAI-specific context errors
    CREWAI_CONTEXT_INVALID = "CREWAI_CONTEXT_INVALID"


# Coercion configuration for the common embedding utils
EMBEDDING_COERCION_ERROR_CODES: CoercionErrorCodes = CoercionErrorCodes(
    invalid_result=ErrorCodes.INVALID_EMBEDDING_RESULT,
    empty_result=ErrorCodes.EMPTY_EMBEDDING_RESULT,
    conversion_error=ErrorCodes.EMBEDDING_CONVERSION_ERROR,
)


class CrewAIContext(TypedDict, total=False):
    """Structured type for CrewAI execution context."""
    agent_role: Optional[str]
    task_id: Optional[str]
    workflow: Optional[str]
    agent_id: Optional[str]
    crew_id: Optional[str]
    process_id: Optional[str]


class CrewAIEmbedder(Protocol):
    """
    Protocol representing the embedder interface expected by CrewAI agents.

    This allows type-safe integration with CrewAI's agent embedder system
    without requiring a hard dependency on CrewAI at type-check time.
    """

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        crewai_context: Optional[CrewAIContext] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """Embed multiple documents for CrewAI RAG workflows."""
        ...

    def embed_query(
        self,
        text: str,
        *,
        crewai_context: Optional[CrewAIContext] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[float]:
        """Embed a single query for CrewAI retrieval."""
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
    Extract rich dynamic context from a CrewAI embedding call.

    Captures:
    - model identifier from the embedding instance
    - text_len for single-text operations
    - texts_count / empty_texts_count for batch operations
    - CrewAI routing fields (agent_role, task_id, crew_id, workflow, process_id)
    """
    dynamic_ctx: Dict[str, Any] = {
        "model": getattr(instance, "model", "unknown"),
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

    # CrewAI-specific context (if passed via keyword)
    crewai_context = kwargs.get("crewai_context") or {}
    if isinstance(crewai_context, Mapping):
        if "agent_role" in crewai_context:
            dynamic_ctx["agent_role"] = crewai_context["agent_role"]
        if "task_id" in crewai_context:
            dynamic_ctx["task_id"] = crewai_context["task_id"]
        if "workflow" in crewai_context:
            dynamic_ctx["workflow"] = crewai_context["workflow"]
        if "crew_id" in crewai_context:
            dynamic_ctx["crew_id"] = crewai_context["crew_id"]
        if "agent_id" in crewai_context:
            dynamic_ctx["agent_id"] = crewai_context["agent_id"]
        if "process_id" in crewai_context:
            dynamic_ctx["process_id"] = crewai_context["process_id"]

    return dynamic_ctx


def _create_error_context_decorator(
    operation: str,
    is_async: bool = False,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Factory for creating error context decorators with rich per-call metrics.

    Mirrors the pattern used in other framework adapters for consistent
    observability.
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
                    full_context = {
                        **static_context,
                        **dynamic_context,
                        "error_codes": EMBEDDING_COERCION_ERROR_CODES,
                        "framework_version": getattr(self, "_framework_version", None),
                    }
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
                    full_context = {
                        **static_context,
                        **dynamic_context,
                        "error_codes": EMBEDDING_COERCION_ERROR_CODES,
                        "framework_version": getattr(self, "_framework_version", None),
                    }
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
    return _create_error_context_decorator(operation, is_async=False)(**static_context)


def with_async_embedding_error_context(
    operation: str,
    **static_context: Any,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for async methods with rich dynamic context extraction."""
    return _create_error_context_decorator(operation, is_async=True)(**static_context)


class CrewAIConfig(TypedDict, total=False):
    """Structured configuration for CrewAI-specific settings."""
    fallback_to_simple_context: bool
    enable_agent_context_propagation: bool
    task_aware_batching: bool


class CorpusCrewAIEmbeddings:
    """
    CrewAI embedding service backed by a Corpus `EmbeddingProtocolV1` adapter.

    This class implements the CrewAI embedder interface and can be directly
    assigned to CrewAI agents via the `embedder` attribute.
    """

    def __init__(
        self,
        corpus_adapter: EmbeddingProtocolV1,
        model: Optional[str] = None,
        batch_config: Optional[BatchConfig] = None,
        text_normalization_config: Optional[TextNormalizationConfig] = None,
        crewai_config: Optional[CrewAIConfig] = None,
        framework_version: Optional[str] = None,
    ):
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
        self.crewai_config = self._validate_crewai_config(crewai_config or {})
        self._framework_version: Optional[str] = framework_version

        logger.info(
            "CorpusCrewAIEmbeddings initialized with model: %s, config: %s, framework_version=%s",
            model or "default",
            self.crewai_config,
            self._framework_version,
        )

    # ------------------------------------------------------------------ #
    # Resource management (context managers)
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "CorpusCrewAIEmbeddings":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if hasattr(self.corpus_adapter, "close"):
            try:
                self.corpus_adapter.close()  # type: ignore[call-arg]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error while closing embedding adapter in __exit__: %s", exc)

    async def __aenter__(self) -> "CorpusCrewAIEmbeddings":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if hasattr(self.corpus_adapter, "aclose"):
            try:
                await self.corpus_adapter.aclose()  # type: ignore[call-arg]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error while closing embedding adapter in __aexit__: %s", exc)

    # ------------------------------------------------------------------ #
    # Health / capabilities passthrough
    # ------------------------------------------------------------------ #

    @with_embedding_error_context("capabilities")
    def capabilities(self) -> Mapping[str, Any]:
        """
        Sync wrapper around underlying adapter capabilities, if available.
        """
        if hasattr(self.corpus_adapter, "capabilities"):
            caps_method = self.corpus_adapter.capabilities
            # Check if it's a coroutine function (async method)
            import inspect
            if inspect.iscoroutinefunction(caps_method):
                # Run the async method in a new event loop
                import asyncio
                import dataclasses
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        return {}
                except RuntimeError:
                    pass
                result = asyncio.run(caps_method())
                # Convert dataclass to dict if needed
                if dataclasses.is_dataclass(result):
                    return dataclasses.asdict(result)  # type: ignore[return-value]
                return result  # type: ignore[no-any-return]
            else:
                result = caps_method()
                # Convert dataclass to dict if needed
                import dataclasses
                if dataclasses.is_dataclass(result):
                    return dataclasses.asdict(result)  # type: ignore[return-value]
                return result  # type: ignore[no-any-return]
        raise NotImplementedError(
            "Underlying embedding adapter does not implement capabilities()",
        )

    @with_async_embedding_error_context("capabilities")
    async def acapabilities(self) -> Mapping[str, Any]:
        """
        Async wrapper around underlying adapter capabilities, if available.
        """
        import dataclasses
        import inspect
        if hasattr(self.corpus_adapter, "acapabilities"):
            result = await self.corpus_adapter.acapabilities()
            if dataclasses.is_dataclass(result):
                return dataclasses.asdict(result)  # type: ignore[return-value]
            return result  # type: ignore[no-any-return]
        if hasattr(self.corpus_adapter, "capabilities"):
            caps_method = self.corpus_adapter.capabilities
            # Check if it's async (coroutine function)
            if inspect.iscoroutinefunction(caps_method):
                result = await caps_method()
            else:
                import asyncio
                result = await asyncio.to_thread(caps_method)
            if dataclasses.is_dataclass(result):
                return dataclasses.asdict(result)  # type: ignore[return-value]
            return result  # type: ignore[no-any-return]
        raise NotImplementedError(
            "Underlying embedding adapter does not implement capabilities()/acapabilities()",
        )

    @with_embedding_error_context("health")
    def health(self) -> Mapping[str, Any]:
        """
        Sync wrapper around underlying adapter health, if available.
        """
        if hasattr(self.corpus_adapter, "health"):
            health_method = self.corpus_adapter.health
            # Check if it's a coroutine function (async method)
            import inspect
            if inspect.iscoroutinefunction(health_method):
                # Run the async method in a new event loop
                import asyncio
                import dataclasses
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        return {}
                except RuntimeError:
                    pass
                result = asyncio.run(health_method())
                # Convert dataclass to dict if needed
                if dataclasses.is_dataclass(result):
                    return dataclasses.asdict(result)  # type: ignore[return-value]
                return result  # type: ignore[no-any-return]
            else:
                result = health_method()
                # Convert dataclass to dict if needed
                import dataclasses
                if dataclasses.is_dataclass(result):
                    return dataclasses.asdict(result)  # type: ignore[return-value]
                return result  # type: ignore[no-any-return]
        raise NotImplementedError(
            "Underlying embedding adapter does not implement health()",
        )

    @with_async_embedding_error_context("health")
    async def ahealth(self) -> Mapping[str, Any]:
        """
        Async wrapper around underlying adapter health, if available.
        """
        import dataclasses
        import inspect
        if hasattr(self.corpus_adapter, "ahealth"):
            result = await self.corpus_adapter.ahealth()
            if dataclasses.is_dataclass(result):
                return dataclasses.asdict(result)  # type: ignore[return-value]
            return result  # type: ignore[no-any-return]
        if hasattr(self.corpus_adapter, "health"):
            health_method = self.corpus_adapter.health
            # Check if it's async (coroutine function)
            if inspect.iscoroutinefunction(health_method):
                result = await health_method()
            else:
                import asyncio
                result = await asyncio.to_thread(health_method)
            if dataclasses.is_dataclass(result):
                return dataclasses.asdict(result)  # type: ignore[return-value]
            return result  # type: ignore[no-any-return]
        raise NotImplementedError(
            "Underlying embedding adapter does not implement health()/ahealth()",
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _validate_crewai_config(self, config: CrewAIConfig) -> CrewAIConfig:
        """Validate and normalize CrewAI configuration with sensible defaults."""
        validated: CrewAIConfig = config.copy()

        validated.setdefault("fallback_to_simple_context", True)
        validated.setdefault("enable_agent_context_propagation", True)
        validated.setdefault("task_aware_batching", False)

        # Bool coercion for robustness
        for key in (
            "fallback_to_simple_context",
            "enable_agent_context_propagation",
            "task_aware_batching",
        ):
            validated[key] = bool(validated[key])

        return validated

    @cached_property
    def _translator(self) -> EmbeddingTranslator:
        """
        Lazily construct and cache the `EmbeddingTranslator`.

        Uses `cached_property` for thread safety and optimal performance
        in multi-agent CrewAI environments.
        """
        translator = create_embedding_translator(
            adapter=self.corpus_adapter,
            framework=_FRAMEWORK_NAME,
            translator=None,
            batch_config=self.batch_config,
            text_normalization_config=self.text_normalization_config,
            framework_version=self._framework_version,
        )
        logger.debug(
            "EmbeddingTranslator initialized for CrewAI with model: %s",
            self.model or "default",
        )
        return translator

    def _build_contexts(
        self,
        *,
        crewai_context: Optional[CrewAIContext] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[Optional[OperationContext], Dict[str, Any]]:
        """
        Build contexts for CrewAI execution environment with comprehensive validation.

        Returns
        -------
        Tuple of:
        - core_ctx: core OperationContext or None if no/invalid context
        - framework_ctx: CrewAI-specific context for translator
        """
        core_ctx: Optional[OperationContext] = None
        framework_ctx: Dict[str, Any] = {
            "framework": _FRAMEWORK_NAME,
            "config": self.crewai_config,
        }
        if self._framework_version is not None:
            framework_ctx["framework_version"] = self._framework_version

        # Convert CrewAI context to core OperationContext with comprehensive error handling
        if crewai_context is not None:
            try:
                self._validate_crewai_context_structure(crewai_context)

                core_ctx_candidate = context_from_crewai(
                    crewai_context,
                    framework_version=self._framework_version,
                )
                if isinstance(core_ctx_candidate, OperationContext):
                    core_ctx = core_ctx_candidate
                    logger.debug(
                        "Successfully created OperationContext from CrewAI context "
                        "for agent: %s, task: %s",
                        crewai_context.get("agent_role", "unknown"),
                        crewai_context.get("task_id", "unknown"),
                    )
                else:
                    logger.warning(
                        "context_from_crewai returned non-OperationContext type: %s. "
                        "Using empty context.",
                        type(core_ctx_candidate).__name__,
                    )
                    if self.crewai_config["fallback_to_simple_context"]:
                        core_ctx = OperationContext()
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Failed to create OperationContext from crewai_context: %s. "
                    "Using empty context.",
                    e,
                )
                try:
                    snapshot = dict(crewai_context)
                except Exception:  # noqa: BLE001
                    snapshot = {"repr": repr(crewai_context)}
                attach_context(
                    e,
                    framework=_FRAMEWORK_NAME,
                    operation="context_build",
                    crewai_context_snapshot=snapshot,
                    config=self.crewai_config,
                    framework_version=self._framework_version,
                )

        # Framework-level context for CrewAI-specific optimizations
        effective_model = model or self.model
        if effective_model:
            framework_ctx["model"] = effective_model

        # Add rich CrewAI-specific context for observability and optimization
        if crewai_context and isinstance(crewai_context, Mapping):
            framework_ctx.update(
                {
                    "agent_role": crewai_context.get("agent_role"),
                    "task_id": crewai_context.get("task_id"),
                    "workflow": crewai_context.get("workflow"),
                    "agent_id": crewai_context.get("agent_id"),
                    "crew_id": crewai_context.get("crew_id"),
                    "process_id": crewai_context.get("process_id"),
                },
            )

            if (
                self.crewai_config["task_aware_batching"]
                and crewai_context.get("task_id")
            ):
                framework_ctx["batch_strategy"] = (
                    f"task_aware_{crewai_context['task_id']}"
                )

        # Include any extra call-specific hints while preserving structure
        framework_ctx.update({k: v for k, v in kwargs.items() if not k.startswith("_")})

        # Stash OperationContext for downstream inspection when enabled
        if (
            core_ctx is not None
            and self.crewai_config["enable_agent_context_propagation"]
        ):
            framework_ctx["_operation_context"] = core_ctx

        return core_ctx, framework_ctx

    def _validate_crewai_context_structure(self, context: Mapping[str, Any]) -> None:
        """Validate CrewAI context structure and log warnings for anomalies."""
        if not isinstance(context, Mapping):
            raise ValueError(
                f"[{ErrorCodes.CREWAI_CONTEXT_INVALID}] "
                f"CrewAI context must be a mapping, got {type(context).__name__}",
            )

        if not context.get("agent_role") and not context.get("task_id"):
            logger.debug(
                "CrewAI context missing both agent_role and task_id - "
                "reduced observability for embeddings",
            )

    def _coerce_embedding_matrix(self, result: Any) -> List[List[float]]:
        """
        Coerce translator result into embedding matrix with comprehensive validation.

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
        Coerce translator result for single-text embed with validation.

        Delegates to the shared framework_utils implementation and preserves
        the existing semantics (first row when multiple are returned).
        """
        return coerce_embedding_vector(
            result=result,
            framework=_FRAMEWORK_NAME,
            error_codes=EMBEDDING_COERCION_ERROR_CODES,
            logger=logger,
        )

    # ------------------------------------------------------------------ #
    # Core Embedding API (CrewAI Compatible)
    # ------------------------------------------------------------------ #

    @with_embedding_error_context("documents")
    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        crewai_context: Optional[CrewAIContext] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """
        Sync embedding for multiple documents.

        Used by CrewAI agents during RAG operations and knowledge processing.
        """
        warn_if_extreme_batch(
            framework=_FRAMEWORK_NAME,
            texts=texts,
            op_name="embed_documents",
            batch_config=self.batch_config,
            logger=logger,
        )

        core_ctx, framework_ctx = self._build_contexts(
            crewai_context=crewai_context,
            model=model,
            **kwargs,
        )

        texts_list = list(texts)
        logger.debug(
            "Embedding %d documents for CrewAI agent: %s, task: %s",
            len(texts_list),
            crewai_context.get("agent_role", "unknown") if (crewai_context and isinstance(crewai_context, Mapping)) else "unknown",
            crewai_context.get("task_id", "unknown") if (crewai_context and isinstance(crewai_context, Mapping)) else "unknown",
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
        crewai_context: Optional[CrewAIContext] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[float]:
        """
        Sync embedding for a single query.

        Used by CrewAI for query understanding and retrieval in agent workflows.
        """
        core_ctx, framework_ctx = self._build_contexts(
            crewai_context=crewai_context,
            model=model,
            **kwargs,
        )

        logger.debug(
            "Embedding query for CrewAI agent: %s",
            crewai_context.get("agent_role", "unknown") if (crewai_context and isinstance(crewai_context, Mapping)) else "unknown",
        )

        translated = self._translator.embed(
            raw_texts=text,
            op_ctx=core_ctx,
            framework_ctx=framework_ctx,
        )
        return self._coerce_embedding_vector(translated)

    # ------------------------------------------------------------------ #
    # Async API for CrewAI Flows
    # ------------------------------------------------------------------ #

    @with_async_embedding_error_context("documents")
    async def aembed_documents(
        self,
        texts: Sequence[str],
        *,
        crewai_context: Optional[CrewAIContext] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """
        Async embedding for multiple documents.

        Designed for async CrewAI workflows and parallel agent execution.
        """
        warn_if_extreme_batch(
            framework=_FRAMEWORK_NAME,
            texts=texts,
            op_name="aembed_documents",
            batch_config=self.batch_config,
            logger=logger,
        )

        core_ctx, framework_ctx = self._build_contexts(
            crewai_context=crewai_context,
            model=model,
            **kwargs,
        )

        texts_list = list(texts)
        logger.debug(
            "Async embedding %d documents for CrewAI task: %s",
            len(texts_list),
            crewai_context.get("task_id", "unknown") if (crewai_context and isinstance(crewai_context, Mapping)) else "unknown",
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
        crewai_context: Optional[CrewAIContext] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[float]:
        """
        Async embedding for a single query.

        Optimized for async CrewAI agent interactions and real-time workflows.
        """
        core_ctx, framework_ctx = self._build_contexts(
            crewai_context=crewai_context,
            model=model,
            **kwargs,
        )

        translated = await self._translator.arun_embed(
            raw_texts=text,
            op_ctx=core_ctx,
            framework_ctx=framework_ctx,
        )
        return self._coerce_embedding_vector(translated)


# ------------------------------------------------------------------ #
# CrewAI Registration Helpers
# ------------------------------------------------------------------ #


def create_embedder(
    corpus_adapter: EmbeddingProtocolV1,
    model: Optional[str] = None,
    *,
    framework_version: Optional[str] = None,
    **kwargs: Any,
) -> CrewAIEmbedder:
    """
    Create a CrewAI-compatible embedder for seamless agent integration.

    This is the simplest entry-point when you want to manually assign
    `embedder=...` on individual agents.
    """
    embedder = CorpusCrewAIEmbeddings(
        corpus_adapter=corpus_adapter,
        model=model,
        framework_version=framework_version,
        **kwargs,
    )

    logger.info(
        "CrewAI embedder created successfully with model: %s, framework_version=%s",
        model or "default",
        framework_version,
    )

    return embedder


def register_with_crewai(
    crew: Any,
    corpus_adapter: EmbeddingProtocolV1,
    model: Optional[str] = None,
    *,
    framework_version: Optional[str] = None,
    **kwargs: Any,
) -> CorpusCrewAIEmbeddings:
    """
    Register Corpus embeddings with a CrewAI `Crew` instance.

    This helper:
    - Creates a `CorpusCrewAIEmbeddings` instance
    - Attempts to attach it as `embedder` on each agent in `crew.agents`
    - Logs warnings instead of failing hard if the shape is unexpected
    """
    if crew is None:
        raise ValueError("crew cannot be None")

    embedder = CorpusCrewAIEmbeddings(
        corpus_adapter=corpus_adapter,
        model=model,
        framework_version=framework_version,
        **kwargs,
    )

    agents_attr = getattr(crew, "agents", None)
    if agents_attr is None:
        logger.warning(
            "Crew object %r has no 'agents' attribute; cannot auto-attach embedder. "
            "Assign it manually on each agent (agent.embedder = embedder).",
            type(crew).__name__,
        )
    else:
        try:
            agents = agents_attr() if callable(agents_attr) else agents_attr
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to introspect crew.agents on %r: %s. "
                "You may need to attach the embedder manually.",
                type(crew).__name__,
                exc,
            )
            agents = []

        attached = 0
        for agent in agents or []:
            if hasattr(agent, "embedder"):
                setattr(agent, "embedder", embedder)
                attached += 1
            else:
                logger.debug(
                    "CrewAI agent %r has no 'embedder' attribute; skipping.",
                    type(agent).__name__,
                )

        logger.info(
            "Corpus CrewAI embedder registered for crew %r; attached to %d agents",
            getattr(crew, "name", None) or type(crew).__name__,
            attached,
        )

    return embedder


__all__ = [
    "CorpusCrewAIEmbeddings",
    "CrewAIEmbedder",
    "CrewAIContext",
    "CrewAIConfig",
    "create_embedder",
    "register_with_crewai",
    "ErrorCodes",
    "with_embedding_error_context",
    "with_async_embedding_error_context",
]
