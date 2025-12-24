# corpus_sdk/embedding/framework_adapters/autogen.py
# SPDX-License-Identifier: Apache-2.0

"""
AutoGen adapter for Corpus Embedding protocol.

This module exposes Corpus `EmbeddingProtocolV1` implementations for use with the
modern Microsoft AutoGen ecosystem (AgentChat/Core/Ext), with:

- A callable embedding function compatible with Chroma-style `embedding_function`
- Integration helper for AutoGen's Chroma-backed memory (`autogen_ext.memory.chromadb`)
- Support for async + sync embedding APIs
- Context normalization using `context_translation.from_autogen`
- Framework-agnostic orchestration via `EmbeddingTranslator`
- Rich error context attachment for observability

Design notes / philosophy
-------------------------
- **Protocol-first**: we require only an `embed` method (duck-typed) instead of
  strict inheritance from a specific adapter base class.
- **Resilient to framework evolution**: framework internals and signatures change;
  we filter/normalize context defensively and keep our adapter surface stable.
- **Observability-first**: embedding operations attach rich error context:
  framework identity, model info, batch sizes, node IDs, trace/workflow IDs, etc.
- **Fail-safe context translation**: context translation must never break embeddings.
  If translation fails, we proceed without `OperationContext` and attach diagnostic context.
- **Strict by default**: non-string inputs in batch operations are rejected to avoid
  embedding `repr()` output by accident. If you need softer behavior, wrap this
  class and preprocess inputs before calling it.
- **Async/sync bridge discipline**: sync APIs refuse to run inside an active event
  loop and instruct callers to use async variants instead, avoiding deadlocks and
  subtle hangs.

Compatibility notes
-------------------
- This module keeps AutoGen as an **optional dependency** by performing AutoGen imports
  lazily inside integration helpers (e.g., `create_vector_memory`). Importing this module
  does not require AutoGen to be installed.
- Modern Microsoft AutoGen packages typically require **Python 3.10+**. If you run tests
  or integration against AutoGen, ensure your environment matches that requirement.
- This adapter is intentionally framework-light: it provides a stable embedding callable
  and a modern AutoGen memory wiring helper, without binding to unstable retriever internals.

Where the embedding logic actually lives
----------------------------------------
All embedding orchestration, batching, normalization, retries/caching expectations, etc.
are handled by the shared embedding layer and the concrete Corpus adapter:

- `corpus_sdk.embedding.framework_adapters.common.embedding_translation`
- Concrete `EmbeddingProtocolV1` adapter implementations

This module mainly:
- Translates AutoGen-style context into `OperationContext` (best effort)
- Packages rich observability context around calls
- Provides a callable embedding function and optional AutoGen memory wiring
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

from corpus_sdk.core.async_bridge import AsyncBridge
from corpus_sdk.core.context_translation import from_autogen as context_from_autogen
from corpus_sdk.core.error_context import attach_context
from corpus_sdk.embedding.embedding_base import EmbeddingProtocolV1, OperationContext
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

# --------------------------------------------------------------------------- #
# Error codes + context types
# --------------------------------------------------------------------------- #


class ErrorCodes:
    """
    Error code constants for AutoGen embedding adapter.

    Shared coercion helpers consume `EMBEDDING_COERCION_ERROR_CODES`, which is a
    `CoercionErrorCodes` instance derived from these values.
    """

    # Coercion-level (used by framework_utils)
    INVALID_EMBEDDING_RESULT = "INVALID_EMBEDDING_RESULT"
    EMPTY_EMBEDDING_RESULT = "EMPTY_EMBEDDING_RESULT"
    EMBEDDING_CONVERSION_ERROR = "EMBEDDING_CONVERSION_ERROR"

    # AutoGen-specific context errors
    AUTOGEN_CONTEXT_INVALID = "AUTOGEN_CONTEXT_INVALID"

    # Sync/async bridge errors
    SYNC_WRAPPER_CALLED_IN_EVENT_LOOP = "SYNC_WRAPPER_CALLED_IN_EVENT_LOOP"


EMBEDDING_COERCION_ERROR_CODES: CoercionErrorCodes = CoercionErrorCodes(
    invalid_result=ErrorCodes.INVALID_EMBEDDING_RESULT,
    empty_result=ErrorCodes.EMPTY_EMBEDDING_RESULT,
    conversion_error=ErrorCodes.EMBEDDING_CONVERSION_ERROR,
    framework_label="autogen",
)


class AutoGenContext(TypedDict, total=False):
    """
    Structured type for AutoGen execution context.

    This is intentionally small and permissive; callers may include additional keys.
    """
    agent_name: Optional[str]
    conversation_id: Optional[str]
    workflow_type: Optional[str]
    retriever_name: Optional[str]
    request_id: Optional[str]
    user_id: Optional[str]


class AutoGenMemory(Protocol):
    """
    Loose protocol representing a modern AutoGen memory component.

    This is intentionally minimal; it exists to provide a usable return type for
    `create_vector_memory()` without binding tightly to AutoGen implementation details.
    """

    async def add(self, *args: Any, **kwargs: Any) -> Any: ...
    async def query(self, *args: Any, **kwargs: Any) -> Any: ...
    async def close(self, *args: Any, **kwargs: Any) -> Any: ...


# --------------------------------------------------------------------------- #
# Small utilities (validation / safe snapshots / loop guards)
# --------------------------------------------------------------------------- #


def _validate_texts_are_strings(texts: Sequence[Any], *, op_name: str) -> None:
    """
    Fail fast if a caller provides non-string items.

    We intentionally do not coerce arbitrary objects to str here, because that can silently
    embed repr() outputs and lead to confusing retrieval behavior.
    """
    for i, t in enumerate(texts):
        if not isinstance(t, str):
            raise TypeError(f"{op_name} expects Sequence[str]; item {i} is {type(t).__name__}")


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


def _ensure_not_in_event_loop(sync_api_name: str) -> None:
    """
    Guard: sync wrappers must not be called from within a running event loop.

    This prevents deadlocks / hangs from calling blocking sync bridges in async contexts.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError(
        f"{sync_api_name} was called from inside an active asyncio event loop. "
        f"Use the async variant instead (e.g. 'await a{sync_api_name}()'). "
        f"[{ErrorCodes.SYNC_WRAPPER_CALLED_IN_EVENT_LOOP}]"
    )


def _run_coro_sync(coro: Any, *, api_name: str, timeout_s: Optional[float] = None) -> Any:
    """
    Run a coroutine from sync code using the SDK's AsyncBridge.

    NOTE: We intentionally *do not* attempt to nest into an already-running event loop
    from here; callers must use async variants in those contexts.
    """
    _ensure_not_in_event_loop(api_name)
    return AsyncBridge.run_async(coro, timeout=timeout_s)


async def _maybe_close_async(adapter: Any) -> None:
    """
    Best-effort async resource cleanup.

    Preference:
      1) aclose() if present
      2) close() if present (await if coroutinefunction, else run in thread)
    """
    aclose = getattr(adapter, "aclose", None)
    if callable(aclose):
        await aclose()
        return

    close = getattr(adapter, "close", None)
    if not callable(close):
        return

    if asyncio.iscoroutinefunction(close):
        await close()
    else:
        await asyncio.to_thread(close)


def _maybe_close_sync(adapter: Any) -> None:
    """
    Best-effort sync resource cleanup.

    Preference:
      1) aclose() if present (run via sync bridge)
      2) close() if present (run coroutine via bridge if async)
    """
    aclose = getattr(adapter, "aclose", None)
    if callable(aclose):
        try:
            _run_coro_sync(aclose(), api_name="close")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error while closing adapter via aclose(): %s", exc)
        return

    close = getattr(adapter, "close", None)
    if not callable(close):
        return

    try:
        if asyncio.iscoroutinefunction(close):
            _run_coro_sync(close(), api_name="close")
        else:
            close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Error while closing adapter via close(): %s", exc)


def _looks_like_operation_context(obj: Any) -> bool:
    """
    OperationContext can be a concrete type OR a Protocol/alias depending on the SDK.

    Prefer an isinstance check when it works; otherwise use a lightweight structural heuristic.
    """
    if obj is None:
        return False
    try:
        if isinstance(obj, OperationContext):
            return True
    except TypeError:
        # OperationContext might be a Protocol / typing alias.
        pass

    return any(
        hasattr(obj, attr)
        for attr in ("request_id", "traceparent", "tenant", "attrs", "to_dict")
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
    Extract per-call dynamic context for observability.

    Captures:
    - model identifier
    - text_len for single-text ops
    - texts_count / empty_texts_count for batch ops
    - key routing fields from autogen_context (conversation_id, agent_name, workflow_type, retriever_name)
    - embedding_dim hint once known
    """
    dynamic_ctx: Dict[str, Any] = {
        "model": getattr(instance, "model", "unknown"),
        "framework_version": getattr(instance, "_framework_version", None),
    }

    dim_hint = getattr(instance, "_embedding_dim_hint", None)
    if isinstance(dim_hint, int):
        dynamic_ctx["embedding_dim"] = dim_hint

    if operation in ("query",) and args and isinstance(args[0], str):
        dynamic_ctx["text_len"] = len(args[0])
    elif operation in ("documents", "function_call") and args:
        maybe_texts = args[0]
        # IMPORTANT: strings are Sequences, but not "batch texts"
        if isinstance(maybe_texts, Sequence) and not isinstance(maybe_texts, (str, bytes)):
            dynamic_ctx["texts_count"] = len(maybe_texts)
            empty_count = sum(
                1
                for text in maybe_texts
                if not isinstance(text, str) or not text.strip()
            )
            if empty_count:
                dynamic_ctx["empty_texts_count"] = empty_count

    autogen_context = kwargs.get("autogen_context") or {}
    if isinstance(autogen_context, Mapping):
        for key in ("conversation_id", "agent_name", "workflow_type", "retriever_name"):
            if key in autogen_context:
                dynamic_ctx[key] = autogen_context[key]

    return dynamic_ctx


def _create_error_context_decorator(
    operation: str,
    is_async: bool = False,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Factory for creating error context decorators with rich per-call metrics.

    Mirrors the pattern used in other framework adapters to keep behavior consistent.
    """

    def decorator_factory(**static_context: Any) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            if is_async:

                @wraps(func)
                async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
                    dynamic_context = _extract_dynamic_context(self, args, kwargs, operation)
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

                return async_wrapper  # type: ignore[return-value]

            @wraps(func)
            def sync_wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
                dynamic_context = _extract_dynamic_context(self, args, kwargs, operation)
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

            return sync_wrapper  # type: ignore[return-value]

        return decorator

    return decorator_factory


def with_embedding_error_context(operation: str, **static_context: Any) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for sync methods with rich dynamic context extraction."""
    return _create_error_context_decorator(operation, is_async=False)(**static_context)


def with_async_embedding_error_context(operation: str, **static_context: Any) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for async methods with rich dynamic context extraction."""
    return _create_error_context_decorator(operation, is_async=True)(**static_context)


# --------------------------------------------------------------------------- #
# Core embedding function implementation
# --------------------------------------------------------------------------- #


class CorpusAutoGenEmbeddings:
    """
    AutoGen-compatible embedding function backed by a Corpus `EmbeddingProtocolV1` adapter.

    This class is designed to be:
    - Callable for vector-store embedding_function protocols (`__call__`)
    - Explicit for query/document embedding (`embed_query`, `embed_documents`)
    - Async-capable (`aembed_query`, `aembed_documents`)

    It is intentionally **framework-light**: it does not depend on AutoGen types at import time.
    AutoGen integration is achieved via `create_vector_memory()` which wires this callable into
    AutoGen's ChromaDB-backed memory.
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
        if not hasattr(corpus_adapter, "embed") or not callable(getattr(corpus_adapter, "embed", None)):
            adapter_type = type(corpus_adapter).__name__ if corpus_adapter is not None else "None"
            raise TypeError(
                "corpus_adapter must implement an EmbeddingProtocolV1-compatible interface "
                f"with an 'embed' method, got {adapter_type}",
            )

        # Light config validation: fail fast on clearly wrong types.
        if batch_config is not None and not isinstance(batch_config, BatchConfig):
            raise TypeError(f"batch_config must be a BatchConfig instance, got {type(batch_config).__name__}")
        if text_normalization_config is not None and not isinstance(text_normalization_config, TextNormalizationConfig):
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

        # Guard lazy translator initialization + dim hint update under concurrency.
        self._lock = threading.Lock()

        # Observability: best-effort dim hint set after first embed (best-effort, first-write-wins).
        # This is for observability only and never used for correctness.
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
        # Cleanup is delegated to _maybe_close_sync, which already does best-effort
        # handling and logging. No extra try/except needed here.
        _maybe_close_sync(self.corpus_adapter)

    async def __aenter__(self) -> "CorpusAutoGenEmbeddings":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        try:
            await _maybe_close_async(self.corpus_adapter)
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
        with self._lock:
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
            logger.debug("EmbeddingTranslator initialized for AutoGen with model=%s", self.model or "default")
            return translator

    def _update_dim_hint(self, dim: Optional[int]) -> None:
        """
        Thread-safe, best-effort dim hint update.

        This is used only for observability/metrics; it is never used to drive
        correctness or adapter behavior. First non-None write wins.
        """
        if dim is None:
            return
        if self._embedding_dim_hint is not None:
            return
        with self._lock:
            if self._embedding_dim_hint is None:
                self._embedding_dim_hint = dim

    def _build_core_context(self, autogen_context: Optional[Mapping[str, Any]]) -> Optional[OperationContext]:
        """
        Build an OperationContext from an AutoGen-style context mapping.

        Context translation is best-effort: failures are logged and attached to the exception,
        but embedding operations must still succeed without an OperationContext.
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
            logger.warning(
                "Failed to create OperationContext from autogen_context: %s. Proceeding without OperationContext.",
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

        if _looks_like_operation_context(core_candidate):
            return core_candidate  # type: ignore[return-value]

        logger.warning(
            "context_from_autogen returned non-OperationContext-like type: %s. Ignoring OperationContext.",
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

        This carries observability hints and AutoGen routing fields, separate from
        the protocol-level OperationContext.
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
            for key in ("agent_name", "conversation_id", "workflow_type", "retriever_name"):
                if key in autogen_context:
                    base[key] = autogen_context[key]

        base.update(kwargs)

        if core_ctx is not None:
            base["_operation_context"] = core_ctx
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
        core_ctx = self._build_core_context(autogen_context)
        framework_ctx = self._build_framework_context(
            autogen_context=autogen_context,
            model=model,
            core_ctx=core_ctx,
            **kwargs,
        )
        return core_ctx, framework_ctx

    def _coerce_embedding_matrix(self, result: Any) -> List[List[float]]:
        """Thin wrapper around shared coercion utility for matrix outputs."""
        return coerce_embedding_matrix(
            result=result,
            framework="autogen",
            error_codes=EMBEDDING_COERCION_ERROR_CODES,
            logger=logger,
        )

    def _coerce_embedding_vector(self, result: Any) -> List[float]:
        """Thin wrapper around shared coercion utility for single-vector outputs."""
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
    # Capabilities / health passthrough (NO BLANK RETURNS)
    # ------------------------------------------------------------------ #

    @with_embedding_error_context("capabilities")
    def capabilities(self) -> Mapping[str, Any]:
        """
        Sync capabilities passthrough.

        - If adapter `capabilities` is async, run it via AsyncBridge (and guard loops).
        - If adapter method doesn't exist, return {} to indicate "not supported".
        """
        caps = getattr(self.corpus_adapter, "capabilities", None)
        if not callable(caps):
            return {}

        if asyncio.iscoroutinefunction(caps):
            return _run_coro_sync(caps(), api_name="capabilities")  # type: ignore[no-any-return]
        return caps()  # type: ignore[no-any-return]

    @with_async_embedding_error_context("capabilities_async")
    async def acapabilities(self) -> Mapping[str, Any]:
        """
        Async capabilities passthrough.

        Preference order:
        1) `acapabilities` on the adapter
        2) `capabilities` on the adapter (await if async, or run in a thread if sync)
        3) `{}` if neither is present
        """
        caps = getattr(self.corpus_adapter, "capabilities", None)
        acaps = getattr(self.corpus_adapter, "acapabilities", None)

        if callable(acaps):
            return await acaps()  # type: ignore[no-any-return]

        if not callable(caps):
            return {}

        if asyncio.iscoroutinefunction(caps):
            return await caps()  # type: ignore[no-any-return]
        return await asyncio.to_thread(caps)  # type: ignore[arg-type]

    @with_embedding_error_context("health")
    def health(self) -> Mapping[str, Any]:
        """
        Sync health passthrough.

        - If adapter `health` is async, run it via AsyncBridge (and guard loops).
        - If adapter method doesn't exist, return {} to indicate "not supported".
        """
        health = getattr(self.corpus_adapter, "health", None)
        if not callable(health):
            return {}

        if asyncio.iscoroutinefunction(health):
            return _run_coro_sync(health(), api_name="health")  # type: ignore[no-any-return]
        return health()  # type: ignore[no-any-return]

    @with_async_embedding_error_context("health_async")
    async def ahealth(self) -> Mapping[str, Any]:
        """
        Async health passthrough.

        Preference order:
        1) `ahealth` on the adapter
        2) `health` on the adapter (await if async, or run in a thread if sync)
        3) `{}` if neither is present
        """
        health = getattr(self.corpus_adapter, "health", None)
        ahealth = getattr(self.corpus_adapter, "ahealth", None)

        if callable(ahealth):
            return await ahealth()  # type: ignore[no-any-return]

        if not callable(health):
            return {}

        if asyncio.iscoroutinefunction(health):
            return await health()  # type: ignore[no-any-return]
        return await asyncio.to_thread(health)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ #
    # EmbeddingFunction interface (sync, guarded)
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
        Callable interface expected by many vector stores:
          embedding_function(texts: Sequence[str]) -> List[List[float]]

        This is a thin wrapper over `embed_documents`, with an event-loop guard
        to prevent misuse from async contexts.
        """
        _ensure_not_in_event_loop("__call__")
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

        Used for indexing/memory writes in typical RAG pipelines.
        """
        _ensure_not_in_event_loop("embed_documents")

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

        start = time.perf_counter()
        translated = self._translator.embed(
            raw_texts=texts_list,
            op_ctx=core_ctx,
            framework_ctx=framework_ctx,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        mat = self._coerce_embedding_matrix(translated)
        self._update_dim_hint(self._infer_dim_from_matrix(mat))

        logger.debug(
            "Sync embedding completed: docs=%d dim=%s latency_ms=%.2f conversation=%s",
            len(mat),
            self._embedding_dim_hint,
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

        Used for retrieval queries / similarity searches.
        """
        _ensure_not_in_event_loop("embed_query")

        if not isinstance(text, str):
            raise TypeError(f"embed_query expects str; got {type(text).__name__}")

        core_ctx, framework_ctx = self._build_contexts(
            autogen_context=autogen_context,
            model=model,
            **kwargs,
        )

        start = time.perf_counter()
        translated = self._translator.embed(
            raw_texts=text,
            op_ctx=core_ctx,
            framework_ctx=framework_ctx,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        vec = self._coerce_embedding_vector(translated)
        self._update_dim_hint(len(vec))

        logger.debug(
            "Sync embedding query completed: dim=%d latency_ms=%.2f conversation=%s",
            len(vec),
            elapsed_ms,
            framework_ctx.get("conversation_id", "unknown"),
        )
        return vec

    # ------------------------------------------------------------------ #
    # Async API
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

        Suitable for async AutoGen flows and event-driven pipelines.
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

        start = time.perf_counter()
        translated = await self._translator.arun_embed(
            raw_texts=texts_list,
            op_ctx=core_ctx,
            framework_ctx=framework_ctx,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        mat = self._coerce_embedding_matrix(translated)
        self._update_dim_hint(self._infer_dim_from_matrix(mat))

        logger.debug(
            "Async embedding completed: docs=%d dim=%s latency_ms=%.2f conversation=%s",
            len(mat),
            self._embedding_dim_hint,
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
        """
        if not isinstance(text, str):
            raise TypeError(f"aembed_query expects str; got {type(text).__name__}")

        core_ctx, framework_ctx = self._build_contexts(
            autogen_context=autogen_context,
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
        self._update_dim_hint(len(vec))

        logger.debug(
            "Async embedding query completed: dim=%d latency_ms=%.2f conversation=%s",
            len(vec),
            elapsed_ms,
            framework_ctx.get("conversation_id", "unknown"),
        )
        return vec


# --------------------------------------------------------------------------- #
# Modern AutoGen integration helper: ChromaDBVectorMemory
# --------------------------------------------------------------------------- #


def create_vector_memory(
    corpus_adapter: EmbeddingProtocolV1,
    *,
    collection_name: str = "corpus_autogen_memory",
    persistence_path: Optional[str] = None,
    model: Optional[str] = None,
    batch_config: Optional[BatchConfig] = None,
    text_normalization_config: Optional[TextNormalizationConfig] = None,
    autogen_config: Optional[Dict[str, Any]] = None,
    framework_version: Optional[str] = None,
    k: int = 3,
    score_threshold: Optional[float] = None,
) -> AutoGenMemory:
    """
    Create a modern AutoGen ChromaDB vector memory configured to use Corpus embeddings.

    AutoGen is an optional dependency: imports are performed lazily.

    Parameters
    ----------
    corpus_adapter:
        Underlying embedding adapter implementing `EmbeddingProtocolV1`.
    collection_name:
        Chroma collection name for the memory store.
    persistence_path:
        Optional path for persistent Chroma storage. If None, uses default behavior.
    model, batch_config, text_normalization_config, autogen_config, framework_version:
        Forwarded to `CorpusAutoGenEmbeddings` construction.
    k:
        Default number of nearest neighbors to retrieve.
    score_threshold:
        Optional similarity score threshold; depends on AutoGen/Chroma semantics.

    Returns
    -------
    AutoGenMemory
        A ChromaDBVectorMemory instance (typed loosely via Protocol).
    """
    try:
        from autogen_ext.memory.chromadb import (  # type: ignore[import-not-found]
            ChromaDBVectorMemory,
            PersistentChromaDBVectorMemoryConfig,
            CustomEmbeddingFunctionConfig,
        )
    except ImportError as exc:  # noqa: BLE001
        raise RuntimeError(
            "AutoGen Chroma memory dependencies are not installed. Install with:\n"
            '  pip install -U "autogen-agentchat" "autogen-core" "autogen-ext[chromadb]"'
        ) from exc

    # AutoGen uses a function+params config to build the embedding function.
    def _embedding_fn_factory(**params: Any) -> Any:
        return CorpusAutoGenEmbeddings(
            corpus_adapter=params["corpus_adapter"],
            model=params.get("model"),
            batch_config=params.get("batch_config"),
            text_normalization_config=params.get("text_normalization_config"),
            autogen_config=params.get("autogen_config"),
            framework_version=params.get("framework_version"),
        )

    embedding_function_config = CustomEmbeddingFunctionConfig(
        function=_embedding_fn_factory,
        params={
            "corpus_adapter": corpus_adapter,
            "model": model,
            "batch_config": batch_config,
            "text_normalization_config": text_normalization_config,
            "autogen_config": autogen_config or {},
            "framework_version": framework_version,
        },
    )

    cfg = PersistentChromaDBVectorMemoryConfig(
        collection_name=collection_name,
        persistence_path=persistence_path,
        embedding_function_config=embedding_function_config,
        k=k,
        score_threshold=score_threshold,
    )

    return ChromaDBVectorMemory(config=cfg)


def register_embeddings(
    corpus_adapter: EmbeddingProtocolV1,
    model: Optional[str] = None,
    batch_config: Optional[BatchConfig] = None,
    text_normalization_config: Optional[TextNormalizationConfig] = None,
    autogen_config: Optional[Dict[str, Any]] = None,
    framework_version: Optional[str] = None,
) -> CorpusAutoGenEmbeddings:
    """
    Convenience constructor for a reusable embedding function instance.

    Useful when you want to pass the same embedding function to multiple components.
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
    "AutoGenMemory",
    "create_vector_memory",
    "register_embeddings",
    "ErrorCodes",
    "with_embedding_error_context",
    "with_async_embedding_error_context",
]
