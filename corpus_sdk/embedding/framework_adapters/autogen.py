# corpus_sdk/embedding/framework_adapters/autogen.py
# SPDX-License-Identifier: Apache-2.0

"""
AutoGen adapter for Corpus Embedding protocol.

This module exposes Corpus `EmbeddingProtocolV1` implementations for
use with Microsoft AutoGen multi-agent conversations, with:

- Full compatibility with AutoGen's `EmbeddingFunction` protocol
- Support for AutoGen's group chat and agent memory systems
- Context normalization using existing `context_translation.from_autogen`
- Framework-agnostic orchestration via `EmbeddingTranslator`
- Async → sync bridging handled in the common embedding layer
- Rich error context attachment for observability

Design notes / philosophy
-------------------------
- **Protocol-first**: we require only an `embed` method (duck-typed) instead of
  strict inheritance from a specific adapter base class.
- **Resilient to framework evolution**: AutoGen’s internals and signatures
  change; we filter/normalize context defensively and keep our adapter surface stable.
- **Observability-first**: all embedding operations attach rich error context:
  framework identity, model info, batch sizes, node IDs, trace/workflow IDs, etc.
- **Fail-safe context translation**: context translation must never break embeddings.
  If translation fails, we proceed without `OperationContext` and attach diagnostic context.
- **Strict by default**: non-string inputs in batch operations are rejected to avoid
  embedding `repr()` output by accident. If you need softer behavior, wrap this
  class and preprocess inputs before calling it.

Resilience (retries, caching, rate limiting, etc.) is expected to be provided
by the underlying adapter, typically a `BaseEmbeddingAdapter` subclass.
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
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    List,
    TypedDict,
)

from corpus_sdk.core.context_translation import (
    from_autogen as context_from_autogen,
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

# Warn-once guard for private vector-store mutations.
_WARNED_PRIVATE_EMBEDDING_ATTR = False


# --------------------------------------------------------------------------- #
# Error codes + context types
# --------------------------------------------------------------------------- #


class ErrorCodes:
    """
    Error code constants for AutoGen embedding adapter.

    This is a simple namespace for framework-specific codes. The shared
    coercion helpers use `EMBEDDING_COERCION_ERROR_CODES`, which is a
    `CoercionErrorCodes` instance derived from these values.
    """

    # Coercion-level (used by framework_utils)
    INVALID_EMBEDDING_RESULT = "INVALID_EMBEDDING_RESULT"
    EMPTY_EMBEDDING_RESULT = "EMPTY_EMBEDDING_RESULT"
    EMBEDDING_CONVERSION_ERROR = "EMBEDDING_CONVERSION_ERROR"

    # AutoGen-specific context errors
    AUTOGEN_CONTEXT_INVALID = "AUTOGEN_CONTEXT_INVALID"


# Coercion configuration for the common embedding utils
EMBEDDING_COERCION_ERROR_CODES: CoercionErrorCodes = CoercionErrorCodes(
    invalid_result=ErrorCodes.INVALID_EMBEDDING_RESULT,
    empty_result=ErrorCodes.EMPTY_EMBEDDING_RESULT,
    conversion_error=ErrorCodes.EMBEDDING_CONVERSION_ERROR,
    framework_label="autogen",
)


class AutoGenContext(TypedDict, total=False):
    """Structured type for AutoGen execution context."""
    agent_name: Optional[str]
    conversation_id: Optional[str]
    workflow_type: Optional[str]
    retriever_name: Optional[str]
    request_id: Optional[str]
    user_id: Optional[str]


# --------------------------------------------------------------------------- #
# AutoGen retriever protocol
# --------------------------------------------------------------------------- #


class AutoGenRetriever(Protocol):
    """
    Protocol representing AutoGen VectorStoreRetriever interface.

    Shorter name for cleaner usage throughout the module.
    """

    @property
    def vectorstore(self) -> Any:
        """Underlying vector store used for retrieval."""
        ...

    def retrieve(self, query: str, **kwargs: Any) -> Any:
        """Retrieve documents for the given query."""
        ...


# --------------------------------------------------------------------------- #
# Small utilities (validation / safe snapshots)
# --------------------------------------------------------------------------- #


def _validate_texts_are_strings(texts: Sequence[Any], *, op_name: str) -> None:
    """
    Fail fast if a caller provides non-string items.

    We intentionally do not coerce arbitrary objects to str here, because
    that can silently embed repr() outputs and lead to confusing retrieval.
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
    OperationContext can be a concrete type OR a Protocol/alias depending on the SDK.

    We prefer an isinstance check when it works, but fall back to a lightweight
    structural heuristic to avoid false negatives.
    """
    if obj is None:
        return False
    try:
        if isinstance(obj, OperationContext):
            return True
    except TypeError:
        # OperationContext might be a Protocol / typing alias.
        pass

    # Heuristic: common context-ish fields/methods found in typical contexts.
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
    Extract dynamic context from an embedding call for enhanced observability.

    Captures:
    - model identifier from the embedding instance
    - text_len for single-text operations
    - texts_count / empty_texts_count for batch operations
    - key AutoGen routing fields (conversation_id, agent_name, workflow_type, retriever_name)

    Also includes best-effort dimension hints once known.
    """
    dynamic_ctx: Dict[str, Any] = {
        "model": getattr(instance, "model", "unknown"),
        "framework_version": getattr(instance, "_framework_version", None),
    }

    # Optional hint: populated after first successful embed
    dim_hint = getattr(instance, "_embedding_dim_hint", None)
    if isinstance(dim_hint, int):
        dynamic_ctx["embedding_dim"] = dim_hint

    # Text / batch metrics
    if operation in ("query",) and args and isinstance(args[0], str):
        dynamic_ctx["text_len"] = len(args[0])
    elif operation in ("documents", "function_call") and args and isinstance(args[0], Sequence):
        texts_seq = args[0]
        dynamic_ctx["texts_count"] = len(texts_seq)
        empty_count = sum(
            1 for text in texts_seq
            if not isinstance(text, str) or not text.strip()
        )
        if empty_count:
            dynamic_ctx["empty_texts_count"] = empty_count

    # AutoGen-specific context (if passed through kwargs)
    autogen_context = kwargs.get("autogen_context") or {}
    if isinstance(autogen_context, Mapping):
        if "conversation_id" in autogen_context:
            dynamic_ctx["conversation_id"] = autogen_context["conversation_id"]
        if "agent_name" in autogen_context:
            dynamic_ctx["agent_name"] = autogen_context["agent_name"]
        if "workflow_type" in autogen_context:
            dynamic_ctx["workflow_type"] = autogen_context["workflow_type"]
        if "retriever_name" in autogen_context:
            dynamic_ctx["retriever_name"] = autogen_context["retriever_name"]

    return dynamic_ctx


def _create_error_context_decorator(
    operation: str,
    is_async: bool = False,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Factory for creating error context decorators with rich per-call metrics.

    Mirrors the pattern used in other framework adapters (e.g., LLM, vector)
    to keep behavior consistent.
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
                        "error_codes": EMBEDDING_COERCION_ERROR_CODES,
                        **static_context,
                        **dynamic_context,
                    }
                    try:
                        return await func(self, *args, **kwargs)
                    except Exception as exc:  # noqa: BLE001
                        attach_context(
                            exc,
                            framework="autogen",
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
                    full_context: Dict[str, Any] = {
                        "error_codes": EMBEDDING_COERCION_ERROR_CODES,
                        **static_context,
                        **dynamic_context,
                    }
                    try:
                        return func(self, *args, **kwargs)
                    except Exception as exc:  # noqa: BLE001
                        attach_context(
                            exc,
                            framework="autogen",
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


# --------------------------------------------------------------------------- #
# Core AutoGen EmbeddingFunction implementation
# --------------------------------------------------------------------------- #


class CorpusAutoGenEmbeddings:
    """
    AutoGen embedding function backed by a Corpus `EmbeddingProtocolV1` adapter.

    AutoGen-Specific Responsibilities
    ---------------------------------
    - Implement AutoGen's `EmbeddingFunction` protocol for vector stores
    - Support AutoGen agent memory and retrieval-augmented generation
    - Integrate with AutoGen's group chat and multi-agent workflows
    - Provide embeddings for agent context and document retrieval
    - Work with AutoGen's `VectorStoreRetriever` and custom retrievers

    Non-responsibilities
    --------------------
    - Agent orchestration and conversation management (handled by AutoGen)
    - Retrieval logic and similarity search (handled by AutoGen retrievers)
    - Multi-agent communication patterns (handled by AutoGen group chats)

    All embedding logic lives in:
    - `corpus_sdk.embedding.framework_adapters.common.embedding_translation`
    - Concrete `EmbeddingProtocolV1` adapter implementations.

    Attributes
    ----------
    corpus_adapter:
        Underlying Corpus embedding adapter implementing `EmbeddingProtocolV1`.

    model:
        Optional model identifier used in AutoGen contexts.

    batch_config:
        Optional `BatchConfig` to control batching behavior.

    text_normalization_config:
        Optional `TextNormalizationConfig` to control whitespace cleanup,
        truncation, casing, encoding, etc.

    autogen_config:
        Optional AutoGen-specific configuration for agent context
        and workflow integration.

    framework_version:
        Optional framework version string for observability alignment.
    """

    def __init__(
        self,
        corpus_adapter: EmbeddingProtocolV1,
        model: Optional[str] = None,
        batch_config: Optional[BatchConfig] = None,
        text_normalization_config: Optional[TextNormalizationConfig] = None,
        autogen_config: Optional[Dict[str, Any]] = None,
        framework_version: Optional[str] = None,
    ) -> None:
        # Behavioral validation (duck-typed) instead of strict isinstance
        if not hasattr(corpus_adapter, "embed") or not callable(
            getattr(corpus_adapter, "embed", None),
        ):
            adapter_type = type(corpus_adapter).__name__ if corpus_adapter is not None else "None"
            raise TypeError(
                "corpus_adapter must implement an EmbeddingProtocolV1-compatible "
                f"interface with an 'embed' method, got {adapter_type}",
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
        self.autogen_config: Dict[str, Any] = autogen_config or {}
        self._framework_version: Optional[str] = framework_version

        # Guard lazy translator initialization under concurrency.
        self._translator_lock = threading.Lock()

        # Observability: best-effort dim hint set after first embed.
        self._embedding_dim_hint: Optional[int] = None

        logger.info(
            "CorpusAutoGenEmbeddings initialized with model=%s, autogen_config=%r, framework_version=%r",
            self.model or "default",
            self.autogen_config,
            self._framework_version,
        )

    # ------------------------------------------------------------------ #
    # Resource management (context managers)
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "CorpusAutoGenEmbeddings":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if hasattr(self.corpus_adapter, "close"):
            try:
                self.corpus_adapter.close()  # type: ignore[call-arg]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error while closing embedding adapter in __exit__: %s", exc)

    async def __aenter__(self) -> "CorpusAutoGenEmbeddings":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if hasattr(self.corpus_adapter, "aclose"):
            try:
                await self.corpus_adapter.aclose()  # type: ignore[call-arg]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error while closing embedding adapter in __aexit__: %s", exc)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @cached_property
    def _translator(self) -> EmbeddingTranslator:
        """
        Lazily construct and cache the `EmbeddingTranslator`.

        We use `cached_property` for ergonomic caching and add an explicit lock
        to avoid duplicate initialization under concurrent first access.
        """
        with self._translator_lock:
            # If another thread finished initialization while we were waiting,
            # return the cached value immediately.
            existing = self.__dict__.get("_translator")
            if isinstance(existing, EmbeddingTranslator):
                return existing

            translator = create_embedding_translator(
                adapter=self.corpus_adapter,
                framework="autogen",
                translator=None,  # use registry/default generic translator
                batch_config=self.batch_config,
                text_normalization_config=self.text_normalization_config,
            )
            logger.debug(
                "EmbeddingTranslator initialized for AutoGen with model=%s",
                self.model or "default",
            )
            return translator

    # ---- context helpers ------------------------------------------------- #

    def _build_core_context(
        self,
        autogen_context: Optional[Mapping[str, Any]],
    ) -> Optional[OperationContext]:
        """
        Build an OperationContext from an AutoGen-style context mapping.

        Context translation is *best-effort*: failures are logged and
        attached to the exception, but embedding operations must still
        succeed without an OperationContext.
        """
        if autogen_context is None:
            return None

        if not isinstance(autogen_context, Mapping):
            logger.warning(
                "[%s] autogen_context should be a Mapping, got %s; ignoring context",
                ErrorCodes.AUTOGEN_CONTEXT_INVALID,
                type(autogen_context).__name__,
            )
            return None

        try:
            core_candidate = context_from_autogen(
                autogen_context,
                framework_version=self._framework_version,
            )
        except Exception as exc:  # noqa: BLE001
            # Context translation is best-effort and must never break embeddings.
            logger.warning(
                "Failed to create OperationContext from autogen_context: %s. "
                "Proceeding without OperationContext.",
                exc,
            )
            attach_context(
                exc,
                framework="autogen",
                operation="context_build",
                autogen_context_snapshot=_safe_snapshot(autogen_context),
                autogen_config=self.autogen_config,
                framework_version=self._framework_version,
                error_codes=EMBEDDING_COERCION_ERROR_CODES,
            )
            return None

        # Avoid brittle isinstance-only checks in case OperationContext
        # is a Protocol/typing alias; accept "OperationContext-like" values.
        if _looks_like_operation_context(core_candidate):
            logger.debug(
                "Successfully created OperationContext from AutoGen context "
                "with conversation_id=%s",
                autogen_context.get("conversation_id", "unknown"),
            )
            return core_candidate  # type: ignore[return-value]

        logger.warning(
            "context_from_autogen returned non-OperationContext-like type: %s. "
            "Ignoring OperationContext.",
            type(core_candidate).__name__,
        )
        return None

    def _build_framework_context(
        self,
        *,
        autogen_context: Optional[Mapping[str, Any]],
        model: Optional[str],
        core_ctx: Optional[OperationContext],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Build the framework-specific context mapping for the translator.

        This carries observability hints and AutoGen-specific routing fields,
        separate from the protocol-level OperationContext.
        """
        effective_model = model or self.model

        base: Dict[str, Any] = {
            "framework": "autogen",
            "autogen_config": dict(self.autogen_config),
        }
        if self._framework_version is not None:
            base["framework_version"] = self._framework_version
        if effective_model:
            base["model"] = effective_model

        if autogen_context:
            if "agent_name" in autogen_context:
                base["agent_name"] = autogen_context["agent_name"]
            if "conversation_id" in autogen_context:
                base["conversation_id"] = autogen_context["conversation_id"]
            if "workflow_type" in autogen_context:
                base["workflow_type"] = autogen_context["workflow_type"]
            if "retriever_name" in autogen_context:
                base["retriever_name"] = autogen_context["retriever_name"]

        # Include any extra call-specific hints.
        base.update(kwargs)

        # Also expose the OperationContext itself for downstream inspection.
        if core_ctx is not None:
            base["_operation_context"] = core_ctx

        # Observability: include best-effort embedding dimension hint.
        if isinstance(self._embedding_dim_hint, int):
            base["embedding_dim_hint"] = self._embedding_dim_hint

        return base

    def _build_contexts(
        self,
        *,
        autogen_context: Optional[Mapping[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[Optional[OperationContext], Dict[str, Any]]:
        """
        Build both OperationContext and framework_ctx for an embedding call.

        This is a thin wrapper around `_build_core_context` and
        `_build_framework_context` to keep call sites terse.
        """
        core_ctx = self._build_core_context(autogen_context)
        framework_ctx = self._build_framework_context(
            autogen_context=autogen_context,
            model=model,
            core_ctx=core_ctx,
            **kwargs,
        )
        return core_ctx, framework_ctx

    def _coerce_embedding_matrix(self, result: Any) -> List[List[float]]:
        """
        Thin wrapper around shared coercion utility for matrix outputs.
        """
        return coerce_embedding_matrix(
            result=result,
            framework="autogen",
            error_codes=EMBEDDING_COERCION_ERROR_CODES,
            logger=logger,
        )

    def _coerce_embedding_vector(self, result: Any) -> List[float]:
        """
        Thin wrapper around shared coercion utility for single-vector outputs.
        """
        return coerce_embedding_vector(
            result=result,
            framework="autogen",
            error_codes=EMBEDDING_COERCION_ERROR_CODES,
            logger=logger,
        )

    @staticmethod
    def _infer_dim_from_matrix(mat: List[List[float]]) -> Optional[int]:
        if not mat:
            return None
        first = mat[0]
        if not isinstance(first, list):
            return None
        return len(first)

    # ------------------------------------------------------------------ #
    # Health / capabilities passthrough
    # ------------------------------------------------------------------ #

    @with_embedding_error_context("capabilities")
    def capabilities(self) -> Mapping[str, Any]:
        """
        Best-effort capabilities passthrough to the underlying adapter (sync).

        If the underlying adapter only exposes async capabilities, callers
        should prefer `acapabilities()`; this method will simply return `{}`.
        """
        if hasattr(self.corpus_adapter, "capabilities"):
            caps_method = self.corpus_adapter.capabilities
            # Only support sync here; async forms should go through acapabilities.
            if asyncio.iscoroutinefunction(caps_method):
                logger.warning(
                    "Underlying embedding adapter exposes async 'capabilities'; "
                    "use 'acapabilities()' instead of sync 'capabilities()'.",
                )
                return {}
            return caps_method()  # type: ignore[no-any-return]
        return {}

    @with_async_embedding_error_context("capabilities_async")
    async def acapabilities(self) -> Mapping[str, Any]:
        """
        Best-effort capabilities passthrough to the underlying adapter (async).

        Preference order:
        1) `acapabilities` on the adapter
        2) `capabilities` on the adapter (awaited if async, or run in a thread if sync)
        3) `{}` if neither is present
        """
        if hasattr(self.corpus_adapter, "acapabilities"):
            return await self.corpus_adapter.acapabilities()  # type: ignore[no-any-return]
        if hasattr(self.corpus_adapter, "capabilities"):
            caps_method = self.corpus_adapter.capabilities
            if asyncio.iscoroutinefunction(caps_method):
                return await caps_method()  # type: ignore[no-any-return]
            # Sync fallback: run in thread pool
            return await asyncio.to_thread(caps_method)  # type: ignore[arg-type]
        return {}

    @with_embedding_error_context("health")
    def health(self) -> Mapping[str, Any]:
        """
        Best-effort health passthrough to the underlying adapter (sync).

        If the underlying adapter only exposes async health, callers
        should prefer `ahealth()`; this method will simply return `{}`.
        """
        if hasattr(self.corpus_adapter, "health"):
            health_method = self.corpus_adapter.health
            # Only support sync here; async forms should go through ahealth.
            if asyncio.iscoroutinefunction(health_method):
                logger.warning(
                    "Underlying embedding adapter exposes async 'health'; "
                    "use 'ahealth()' instead of sync 'health()'.",
                )
                return {}
            return health_method()  # type: ignore[no-any-return]
        return {}

    @with_async_embedding_error_context("health_async")
    async def ahealth(self) -> Mapping[str, Any]:
        """
        Best-effort health passthrough to the underlying adapter (async).

        Preference order:
        1) `ahealth` on the adapter
        2) `health` on the adapter (awaited if async, or run in a thread if sync)
        3) `{}` if neither is present
        """
        if hasattr(self.corpus_adapter, "ahealth"):
            return await self.corpus_adapter.ahealth()  # type: ignore[no-any-return]
        if hasattr(self.corpus_adapter, "health"):
            health_method = self.corpus_adapter.health
            if asyncio.iscoroutinefunction(health_method):
                return await health_method()  # type: ignore[no-any-return]
            return await asyncio.to_thread(health_method)  # type: ignore[arg-type]
        return {}

    # ------------------------------------------------------------------ #
    # Core AutoGen EmbeddingFunction Interface
    # ------------------------------------------------------------------ #

    @with_embedding_error_context("function_call")
    def __call__(
        self,
        texts: Sequence[str],
        *,
        autogen_context: Optional[Mapping[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """
        Make the instance callable for AutoGen's EmbeddingFunction protocol.

        This enables direct usage with AutoGen's VectorStoreRetriever:
        ```python
        retriever = VectorStoreRetriever(
            vectorstore=Chroma(embedding_function=CorpusAutoGenEmbeddings(...)),
            ...
        )
        ```
        """
        # AutoGen generally passes a list, but Sequence[str] keeps us flexible.
        return self.embed_documents(
            list(texts),
            autogen_context=autogen_context,
            model=model,
            **kwargs,
        )

    @with_embedding_error_context("documents")
    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        autogen_context: Optional[Mapping[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """
        Sync embedding for multiple documents.

        This is the primary method used by AutoGen's retrieval systems
        for document embedding and agent memory.
        """
        texts_list = list(texts)
        _validate_texts_are_strings(texts_list, op_name="embed_documents")

        warn_if_extreme_batch(
            framework="autogen",
            texts=texts_list,
            op_name="embed_documents",
            batch_config=self.batch_config,
            logger=logger,
        )

        core_ctx, framework_ctx = self._build_contexts(
            autogen_context=autogen_context,
            model=model,
            **kwargs,
        )

        logger.debug(
            "Sync embedding %d documents for AutoGen conversation: %s",
            len(texts_list),
            framework_ctx.get("conversation_id", "unknown"),
        )

        start = time.perf_counter()
        translated = self._translator.embed(
            raw_texts=texts_list,
            op_ctx=core_ctx,
            framework_ctx=framework_ctx,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        mat = self._coerce_embedding_matrix(translated)

        # Observability: remember dim once known.
        dim = self._infer_dim_from_matrix(mat)
        if dim is not None:
            self._embedding_dim_hint = dim

        logger.debug(
            "Sync embedding completed: docs=%d dim=%s latency_ms=%.2f conversation=%s",
            len(mat),
            dim,
            elapsed_ms,
            framework_ctx.get("conversation_id", "unknown"),
        )
        return mat

    @with_embedding_error_context("query")
    def embed_query(
        self,
        text: str,
        *,
        autogen_context: Optional[Mapping[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[float]:
        """
        Sync embedding for a single query.

        Used by AutoGen for query understanding and retrieval in
        multi-agent conversations.
        """
        if not isinstance(text, str):
            raise TypeError(f"embed_query expects str; got {type(text).__name__}")

        core_ctx, framework_ctx = self._build_contexts(
            autogen_context=autogen_context,
            model=model,
            **kwargs,
        )

        logger.debug(
            "Sync embedding query for AutoGen conversation: %s",
            framework_ctx.get("conversation_id", "unknown"),
        )

        start = time.perf_counter()
        translated = self._translator.embed(
            raw_texts=text,
            op_ctx=core_ctx,
            framework_ctx=framework_ctx,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        vec = self._coerce_embedding_vector(translated)

        if self._embedding_dim_hint is None:
            self._embedding_dim_hint = len(vec)

        logger.debug(
            "Sync embedding query completed: dim=%d latency_ms=%.2f conversation=%s",
            len(vec),
            elapsed_ms,
            framework_ctx.get("conversation_id", "unknown"),
        )
        return vec

    # ------------------------------------------------------------------ #
    # Async API for AutoGen Async Workflows
    # ------------------------------------------------------------------ #

    @with_async_embedding_error_context("documents")
    async def aembed_documents(
        self,
        texts: Sequence[str],
        *,
        autogen_context: Optional[Mapping[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """
        Async embedding for multiple documents.

        Designed for use with AutoGen's async workflows and
        event-driven agent systems.
        """
        texts_list = list(texts)
        _validate_texts_are_strings(texts_list, op_name="aembed_documents")

        warn_if_extreme_batch(
            framework="autogen",
            texts=texts_list,
            op_name="aembed_documents",
            batch_config=self.batch_config,
            logger=logger,
        )

        core_ctx, framework_ctx = self._build_contexts(
            autogen_context=autogen_context,
            model=model,
            **kwargs,
        )

        logger.debug(
            "Async embedding %d documents for AutoGen conversation: %s",
            len(texts_list),
            framework_ctx.get("conversation_id", "unknown"),
        )

        start = time.perf_counter()
        translated = await self._translator.arun_embed(
            raw_texts=texts_list,
            op_ctx=core_ctx,
            framework_ctx=framework_ctx,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        mat = self._coerce_embedding_matrix(translated)

        dim = self._infer_dim_from_matrix(mat)
        if dim is not None:
            self._embedding_dim_hint = dim

        logger.debug(
            "Async embedding completed: docs=%d dim=%s latency_ms=%.2f conversation=%s",
            len(mat),
            dim,
            elapsed_ms,
            framework_ctx.get("conversation_id", "unknown"),
        )
        return mat

    @with_async_embedding_error_context("query")
    async def aembed_query(
        self,
        text: str,
        *,
        autogen_context: Optional[Mapping[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[float]:
        """
        Async embedding for a single query.

        Used in AutoGen's asynchronous agent workflows and
        flow-based conversation systems.
        """
        if not isinstance(text, str):
            raise TypeError(f"aembed_query expects str; got {type(text).__name__}")

        core_ctx, framework_ctx = self._build_contexts(
            autogen_context=autogen_context,
            model=model,
            **kwargs,
        )

        logger.debug(
            "Async embedding query for AutoGen conversation: %s",
            framework_ctx.get("conversation_id", "unknown"),
        )

        start = time.perf_counter()
        translated = await self._translator.arun_embed(
            raw_texts=text,
            op_ctx=core_ctx,
            framework_ctx=framework_ctx,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        vec = self._coerce_embedding_vector(translated)

        if self._embedding_dim_hint is None:
            self._embedding_dim_hint = len(vec)

        logger.debug(
            "Async embedding query completed: dim=%d latency_ms=%.2f conversation=%s",
            len(vec),
            elapsed_ms,
            framework_ctx.get("conversation_id", "unknown"),
        )
        return vec


# --------------------------------------------------------------------------- #
# AutoGen-Specific Helper Functions
# --------------------------------------------------------------------------- #


def _validate_vector_store_for_autogen(vector_store: Any) -> None:
    """
    Best-effort validation that the provided vector_store looks like a
    typical vector store object AutoGen expects.

    We keep this intentionally loose to avoid over-constraining users:
      - Accepts stores that expose common vector-store methods like
        `similarity_search` or `query`.
    """
    if vector_store is None:
        raise TypeError("vector_store must not be None")

    has_similarity = hasattr(vector_store, "similarity_search")
    has_query = hasattr(vector_store, "query")

    if not (has_similarity or has_query):
        logger.warning(
            "Vector store %r does not expose common methods like 'similarity_search' "
            "or 'query'. It may not be compatible with AutoGen's VectorStoreRetriever.",
            type(vector_store).__name__,
        )


def _set_vector_store_embedding_function(vector_store: Any, embedding_function: Any) -> None:
    """
    Configure the vector store with our embedding function.

    Improvement: prefer the best available public API, then fall back.

    Order:
      1) public setter method if present
      2) public attribute `embedding_function`
      3) private attribute `_embedding_function` (warn once)
    """
    global _WARNED_PRIVATE_EMBEDDING_ATTR  # noqa: PLW0603

    # 1) Public setter patterns (varies by vector store implementation)
    for setter_name in ("set_embedding_function", "set_embedding_fn"):
        setter = getattr(vector_store, setter_name, None)
        if callable(setter):
            setter(embedding_function)
            return

    # 2) Public attribute
    if hasattr(vector_store, "embedding_function"):
        setattr(vector_store, "embedding_function", embedding_function)
        return

    # 3) Private attribute (pragmatic fallback)
    if hasattr(vector_store, "_embedding_function"):
        if not _WARNED_PRIVATE_EMBEDDING_ATTR:
            _WARNED_PRIVATE_EMBEDDING_ATTR = True
            logger.warning(
                "Setting private attribute '_embedding_function' on vector_store %r. "
                "If the vector store library changes its internals, this may break.",
                type(vector_store).__name__,
            )
        setattr(vector_store, "_embedding_function", embedding_function)
        return

    logger.warning(
        "Vector store %r does not expose an embedding setter or known embedding attribute. "
        "You may need to configure the embedding function manually.",
        type(vector_store).__name__,
    )


def create_retriever(
    corpus_adapter: EmbeddingProtocolV1,
    vector_store: Any,
    *,
    model: Optional[str] = None,
    batch_config: Optional[BatchConfig] = None,
    text_normalization_config: Optional[TextNormalizationConfig] = None,
    autogen_config: Optional[Dict[str, Any]] = None,
    framework_version: Optional[str] = None,
    **retriever_kwargs: Any,
) -> AutoGenRetriever:
    """
    Create an AutoGen VectorStoreRetriever with Corpus embeddings.

    This provides a convenient way to create AutoGen retrievers
    with Corpus embeddings in a single function call.

    Parameters
    ----------
    corpus_adapter:
        Underlying embedding adapter implementing `EmbeddingProtocolV1`.
    vector_store:
        Vector store instance compatible with AutoGen's VectorStoreRetriever.
    model:
        Optional model identifier to use for embeddings.
    batch_config:
        Optional batching configuration forwarded to `CorpusAutoGenEmbeddings`.
    text_normalization_config:
        Optional text normalization configuration forwarded to
        `CorpusAutoGenEmbeddings`.
    autogen_config:
        Optional AutoGen-specific configuration for agent/workflow integration.
    framework_version:
        Optional framework version string for observability alignment.
    retriever_kwargs:
        Additional keyword arguments forwarded to AutoGen's VectorStoreRetriever.

    Example usage:
    ```python
    from corpus_sdk.embedding.framework_adapters.autogen import create_retriever
    from chromadb import Chroma

    vectorstore = Chroma(collection_name="autogen_docs")

    retriever = create_retriever(
        corpus_adapter=my_adapter,
        vector_store=vectorstore,
        model="text-embedding-3-large",
    )
    ```
    """
    try:
        from autogen.retrieve_utils import VectorStoreRetriever
    except ImportError as exc:  # noqa: BLE001
        message = (
            "AutoGen is not installed. To use create_retriever, install the "
            "AutoGen package, for example: 'pip install pyautogen'."
        )
        logger.error(message)
        raise RuntimeError(message) from exc

    # Best-effort validation before mutating the vector store.
    _validate_vector_store_for_autogen(vector_store)

    embedding_function = CorpusAutoGenEmbeddings(
        corpus_adapter=corpus_adapter,
        model=model,
        batch_config=batch_config,
        text_normalization_config=text_normalization_config,
        autogen_config=autogen_config,
        framework_version=framework_version,
    )

    _set_vector_store_embedding_function(vector_store, embedding_function)

    retriever = VectorStoreRetriever(vectorstore=vector_store, **retriever_kwargs)

    logger.info("AutoGen retriever created with Corpus embeddings")

    return retriever


def register_embeddings(
    corpus_adapter: EmbeddingProtocolV1,
    model: Optional[str] = None,
    batch_config: Optional[BatchConfig] = None,
    text_normalization_config: Optional[TextNormalizationConfig] = None,
    autogen_config: Optional[Dict[str, Any]] = None,
    framework_version: Optional[str] = None,
) -> CorpusAutoGenEmbeddings:
    """
    Register Corpus embeddings for global use in AutoGen workflows.

    This function provides a centralized way to configure Corpus embeddings
    for multiple AutoGen agents and retrievers.
    """
    embeddings = CorpusAutoGenEmbeddings(
        corpus_adapter=corpus_adapter,
        model=model,
        batch_config=batch_config,
        text_normalization_config=text_normalization_config,
        autogen_config=autogen_config,
        framework_version=framework_version,
    )

    logger.info(
        "Corpus AutoGen embeddings registered: %s (framework_version=%r)",
        model or "default model",
        framework_version,
    )

    return embeddings


__all__ = [
    "CorpusAutoGenEmbeddings",
    "AutoGenContext",
    "AutoGenRetriever",
    "create_retriever",
    "register_embeddings",
    "ErrorCodes",
    "with_embedding_error_context",
    "with_async_embedding_error_context",
]
