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

Resilience (retries, caching, rate limiting, etc.) is expected to be provided
by the underlying adapter, typically a BaseEmbeddingAdapter subclass.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from functools import cached_property, wraps
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
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
    BatchConfig,
    EmbeddingTranslator,
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
    CREWAI_CONTEXT_FALLBACK_FAILED = "CREWAI_CONTEXT_FALLBACK_FAILED"


# Coercion configuration for the common embedding utils
EMBEDDING_COERCION_ERROR_CODES: CoercionErrorCodes = CoercionErrorCodes(
    invalid_result=ErrorCodes.INVALID_EMBEDDING_RESULT,
    empty_result=ErrorCodes.EMPTY_EMBEDDING_RESULT,
    conversion_error=ErrorCodes.EMBEDDING_CONVERSION_ERROR,
    framework_label=_FRAMEWORK_NAME,
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
# Safety / robustness utilities (input validation + safe snapshots)
# --------------------------------------------------------------------------- #


def _validate_texts_are_strings(texts: Sequence[Any], *, op_name: str) -> None:
    """
    Fail fast if a caller provides non-string items.

    We intentionally do not coerce arbitrary objects to str here, because that can
    silently embed repr() outputs and lead to confusing retrieval behavior.
    """
    for i, t in enumerate(texts):
        if not isinstance(t, str):
            raise TypeError(
                f"{op_name} expects Sequence[str]; item {i} is {type(t).__name__}",
            )


def _safe_snapshot(value: Any, *, max_items: int = 200, max_str: int = 5_000) -> Any:
    """
    Best-effort conversion into a JSON-ish, safe-to-log snapshot.

    - Limits container size to reduce log bloat
    - Truncates long strings
    - Falls back to repr() for unknown objects
    """
    try:
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return value if len(value) <= max_str else value[:max_str] + "…"
        if isinstance(value, Mapping):
            out: Dict[str, Any] = {}
            for idx, (k, v) in enumerate(value.items()):
                if idx >= max_items:
                    out["…"] = f"truncated after {max_items} items"
                    break
                out[str(k)] = _safe_snapshot(v, max_items=max_items, max_str=max_str)
            return out
        if isinstance(value, (list, tuple)):
            out_list: List[Any] = []
            for idx, v in enumerate(value):
                if idx >= max_items:
                    out_list.append(f"… truncated after {max_items} items")
                    break
                out_list.append(_safe_snapshot(v, max_items=max_items, max_str=max_str))
            return out_list
        return repr(value)
    except Exception:  # noqa: BLE001
        return {"repr": repr(value)}


def _looks_like_operation_context(obj: Any) -> bool:
    """
    OperationContext may be a concrete type or a Protocol/alias depending on the SDK.

    Prefer isinstance when it works; fall back to a lightweight structural
    heuristic to avoid false negatives.
    """
    if obj is None:
        return False
    try:
        if isinstance(obj, OperationContext):
            return True
    except TypeError:
        # OperationContext may be a Protocol/typing alias at runtime
        pass

    return any(
        hasattr(obj, attr)
        for attr in (
            "trace_id",
            "request_id",
            "user_id",
            "tags",
            "metadata",
            "to_dict",
        )
    )


def _try_make_empty_operation_context() -> Optional[OperationContext]:
    """
    Best-effort creation of an empty OperationContext.

    Some SDKs may expose OperationContext as a Protocol/alias, or require args.
    This helper prevents runtime crashes by returning None if construction fails.
    """
    try:
        # Only attempt if it's actually callable at runtime (e.g. a class).
        if callable(OperationContext):
            ctx = OperationContext()  # type: ignore[call-arg]
            return ctx if _looks_like_operation_context(ctx) else None
    except Exception:  # noqa: BLE001
        return None
    return None


def _infer_dim_from_matrix(mat: List[List[float]]) -> Optional[int]:
    """Best-effort embedding dimension inference from a 2D embedding matrix."""
    if not mat:
        return None
    first = mat[0]
    if not isinstance(first, list):
        return None
    return len(first)


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
        "framework_version": getattr(instance, "_framework_version", None),
    }

    # Optional best-effort dimension hint (populated after first successful embed)
    dim_hint = getattr(instance, "_embedding_dim_hint", None)
    if isinstance(dim_hint, int):
        dynamic_ctx["embedding_dim"] = dim_hint

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
        for key in ("agent_role", "task_id", "workflow", "crew_id", "agent_id", "process_id"):
            if key in crewai_context:
                dynamic_ctx[key] = crewai_context[key]

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
                    full_context: Dict[str, Any] = {
                        **static_context,
                        **dynamic_context,
                        "error_codes": EMBEDDING_COERCION_ERROR_CODES,
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

            @wraps(func)
            def sync_wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
                dynamic_context = _extract_dynamic_context(
                    self,
                    args,
                    kwargs,
                    operation,
                )
                full_context: Dict[str, Any] = {
                    **static_context,
                    **dynamic_context,
                    "error_codes": EMBEDDING_COERCION_ERROR_CODES,
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
        self.crewai_config = self._validate_crewai_config(crewai_config or {})
        self._framework_version: Optional[str] = framework_version

        # Best-effort embedding dimension hint for observability and downstream context.
        self._embedding_dim_hint: Optional[int] = None

        # Thread-safe initialization of the cached translator under concurrent access.
        self._translator_lock = threading.Lock()

        logger.info(
            "CorpusCrewAIEmbeddings initialized with model=%s, config=%s, framework_version=%r",
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
        """Sync wrapper around underlying adapter capabilities, if available."""
        if hasattr(self.corpus_adapter, "capabilities"):
            return self.corpus_adapter.capabilities()  # type: ignore[no-any-return]
        return {}

    @with_async_embedding_error_context("capabilities")
    async def acapabilities(self) -> Mapping[str, Any]:
        """Async wrapper around underlying adapter capabilities, if available."""
        if hasattr(self.corpus_adapter, "acapabilities"):
            return await self.corpus_adapter.acapabilities()  # type: ignore[no-any-return]
        if hasattr(self.corpus_adapter, "capabilities"):
            return await asyncio.to_thread(self.corpus_adapter.capabilities)  # type: ignore[arg-type]
        return {}

    @with_embedding_error_context("health")
    def health(self) -> Mapping[str, Any]:
        """Sync wrapper around underlying adapter health, if available."""
        if hasattr(self.corpus_adapter, "health"):
            return self.corpus_adapter.health()  # type: ignore[no-any-return]
        return {}

    @with_async_embedding_error_context("health")
    async def ahealth(self) -> Mapping[str, Any]:
        """Async wrapper around underlying adapter health, if available."""
        if hasattr(self.corpus_adapter, "ahealth"):
            return await self.corpus_adapter.ahealth()  # type: ignore[no-any-return]
        if hasattr(self.corpus_adapter, "health"):
            return await asyncio.to_thread(self.corpus_adapter.health)  # type: ignore[arg-type]
        return {}

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

        Uses `cached_property` for caching and an explicit lock to avoid
        duplicate initialization under concurrent first access.
        """
        with self._translator_lock:
            # If another thread finished initialization while we were waiting,
            # return the cached value immediately.
            existing = self.__dict__.get("_translator")
            if isinstance(existing, EmbeddingTranslator):
                return existing

            translator = create_embedding_translator(
                adapter=self.corpus_adapter,
                framework=_FRAMEWORK_NAME,
                translator=None,
                batch_config=self.batch_config,
                text_normalization_config=self.text_normalization_config,
            )
            logger.debug(
                "EmbeddingTranslator initialized for CrewAI with model=%s",
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
            "crewai_config": dict(self.crewai_config),  # single canonical key
        }
        if self._framework_version is not None:
            framework_ctx["framework_version"] = self._framework_version

        # Expose best-effort dim hint for observability parity with other adapters.
        if isinstance(self._embedding_dim_hint, int):
            framework_ctx["embedding_dim_hint"] = self._embedding_dim_hint

        # Convert CrewAI context to core OperationContext (best-effort; never fatal)
        if crewai_context is not None:
            try:
                self._validate_crewai_context_structure(crewai_context)

                core_ctx_candidate = context_from_crewai(
                    crewai_context,
                    framework_version=self._framework_version,
                )
                if _looks_like_operation_context(core_ctx_candidate):
                    core_ctx = core_ctx_candidate  # type: ignore[assignment]
                    logger.debug(
                        "Successfully created OperationContext from CrewAI context "
                        "for agent=%s task=%s",
                        crewai_context.get("agent_role", "unknown"),
                        crewai_context.get("task_id", "unknown"),
                    )
                else:
                    logger.warning(
                        "context_from_crewai returned non-OperationContext-like type: %s.",
                        type(core_ctx_candidate).__name__,
                    )
                    if self.crewai_config["fallback_to_simple_context"]:
                        fallback = _try_make_empty_operation_context()
                        if fallback is None:
                            logger.warning(
                                "[%s] fallback_to_simple_context enabled, but could not "
                                "construct OperationContext safely; proceeding with None.",
                                ErrorCodes.CREWAI_CONTEXT_FALLBACK_FAILED,
                            )
                        core_ctx = fallback
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to create OperationContext from crewai_context: %s. "
                    "Proceeding without OperationContext.",
                    exc,
                )
                attach_context(
                    exc,
                    framework=_FRAMEWORK_NAME,
                    operation="context_build",
                    crewai_context_snapshot=_safe_snapshot(crewai_context),
                    crewai_config=dict(self.crewai_config),
                    framework_version=self._framework_version,
                    error_codes=EMBEDDING_COERCION_ERROR_CODES,
                )
                if self.crewai_config["fallback_to_simple_context"]:
                    core_ctx = _try_make_empty_operation_context()

        # Framework-level model selection
        effective_model = model or self.model
        if effective_model:
            framework_ctx["model"] = effective_model

        # Add CrewAI-specific context for observability
        if crewai_context:
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

            if self.crewai_config["task_aware_batching"] and crewai_context.get("task_id"):
                framework_ctx["batch_strategy"] = f"task_aware_{crewai_context['task_id']}"

        # Include extra call-specific hints while preserving structure
        framework_ctx.update({k: v for k, v in kwargs.items() if not k.startswith("_")})

        # Stash OperationContext for downstream inspection when enabled
        if core_ctx is not None and self.crewai_config["enable_agent_context_propagation"]:
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
                "CrewAI context missing both agent_role and task_id - reduced observability",
            )

    def _coerce_embedding_matrix(self, result: Any) -> List[List[float]]:
        """Coerce translator result into embedding matrix with shared validation."""
        return coerce_embedding_matrix(
            result=result,
            framework=_FRAMEWORK_NAME,
            error_codes=EMBEDDING_COERCION_ERROR_CODES,
            logger=logger,
        )

    def _coerce_embedding_vector(self, result: Any) -> List[float]:
        """Coerce translator result into a single embedding vector with shared validation."""
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
        """Sync embedding for multiple documents (CrewAI RAG + knowledge)."""
        texts_list = list(texts)
        _validate_texts_are_strings(texts_list, op_name="embed_documents")

        warn_if_extreme_batch(
            framework=_FRAMEWORK_NAME,
            texts=texts_list,
            op_name="embed_documents",
            batch_config=self.batch_config,
            logger=logger,
        )

        core_ctx, framework_ctx = self._build_contexts(
            crewai_context=crewai_context,
            model=model,
            **kwargs,
        )

        logger.debug(
            "Embedding %d documents for CrewAI agent=%s task=%s",
            len(texts_list),
            crewai_context.get("agent_role", "unknown") if crewai_context else "unknown",
            crewai_context.get("task_id", "unknown") if crewai_context else "unknown",
        )

        start = time.perf_counter()
        translated = self._translator.embed(
            raw_texts=texts_list,
            op_ctx=core_ctx,
            framework_ctx=framework_ctx,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        mat = self._coerce_embedding_matrix(translated)

        dim = _infer_dim_from_matrix(mat)
        if dim is not None:
            self._embedding_dim_hint = dim

        logger.debug(
            "CrewAI embed_documents completed: docs=%d dim=%s latency_ms=%.2f",
            len(mat),
            dim,
            elapsed_ms,
        )
        return mat

    @with_embedding_error_context("query")
    def embed_query(
        self,
        text: str,
        *,
        crewai_context: Optional[CrewAIContext] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[float]:
        """Sync embedding for a single query (CrewAI retrieval)."""
        if not isinstance(text, str):
            raise TypeError(f"embed_query expects str; got {type(text).__name__}")

        core_ctx, framework_ctx = self._build_contexts(
            crewai_context=crewai_context,
            model=model,
            **kwargs,
        )

        logger.debug(
            "Embedding query for CrewAI agent=%s",
            crewai_context.get("agent_role", "unknown") if crewai_context else "unknown",
        )

        start = time.perf_counter()
        translated = self._translator.embed(
            raw_texts=text,
            op_ctx=core_ctx,
            framework_ctx=framework_ctx,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        vec = self._coerce_embedding_vector(translated)

        self._embedding_dim_hint = len(vec)

        logger.debug(
            "CrewAI embed_query completed: dim=%d latency_ms=%.2f",
            len(vec),
            elapsed_ms,
        )
        return vec

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
        """Async embedding for multiple documents (async CrewAI workflows)."""
        texts_list = list(texts)
        _validate_texts_are_strings(texts_list, op_name="aembed_documents")

        warn_if_extreme_batch(
            framework=_FRAMEWORK_NAME,
            texts=texts_list,
            op_name="aembed_documents",
            batch_config=self.batch_config,
            logger=logger,
        )

        core_ctx, framework_ctx = self._build_contexts(
            crewai_context=crewai_context,
            model=model,
            **kwargs,
        )

        logger.debug(
            "Async embedding %d documents for CrewAI task=%s",
            len(texts_list),
            crewai_context.get("task_id", "unknown") if crewai_context else "unknown",
        )

        start = time.perf_counter()
        translated = await self._translator.arun_embed(
            raw_texts=texts_list,
            op_ctx=core_ctx,
            framework_ctx=framework_ctx,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        mat = self._coerce_embedding_matrix(translated)

        dim = _infer_dim_from_matrix(mat)
        if dim is not None:
            self._embedding_dim_hint = dim

        logger.debug(
            "CrewAI aembed_documents completed: docs=%d dim=%s latency_ms=%.2f",
            len(mat),
            dim,
            elapsed_ms,
        )
        return mat

    @with_async_embedding_error_context("query")
    async def aembed_query(
        self,
        text: str,
        *,
        crewai_context: Optional[CrewAIContext] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[float]:
        """Async embedding for a single query (real-time async workflows)."""
        if not isinstance(text, str):
            raise TypeError(f"aembed_query expects str; got {type(text).__name__}")

        core_ctx, framework_ctx = self._build_contexts(
            crewai_context=crewai_context,
            model=model,
            **kwargs,
        )

        start = time.perf_counter()
        translated = await self._translator.arun_embed(
            raw_texts=text,
            op_ctx=core_ctx,
            framework_ctx=framework_ctx,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        vec = self._coerce_embedding_vector(translated)

        self._embedding_dim_hint = len(vec)

        logger.debug(
            "CrewAI aembed_query completed: dim=%d latency_ms=%.2f",
            len(vec),
            elapsed_ms,
        )
        return vec


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
        "CrewAI embedder created successfully with model=%s framework_version=%r",
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
        return embedder

    try:
        # Some frameworks expose agents as a property/list; some as a method.
        agents = agents_attr() if callable(agents_attr) else agents_attr
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to introspect crew.agents on %r: %s. You may need to attach manually.",
            type(crew).__name__,
            exc,
        )
        attach_context(
            exc,
            framework=_FRAMEWORK_NAME,
            operation="register_with_crewai",
            crew_snapshot=_safe_snapshot(
                {"type": type(crew).__name__, "name": getattr(crew, "name", None)},
            ),
            error_codes=EMBEDDING_COERCION_ERROR_CODES,
            framework_version=framework_version,
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
