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

Design notes / philosophy
-------------------------
- **Protocol-first**: only a duck-typed `embed` method is required on the
  `corpus_adapter`, rather than strict subclassing of a specific base class.
- **Resilient to framework evolution**: Semantic Kernel APIs and signatures
  evolve; this adapter defensively normalizes context and avoids depending on
  unstable internals.
- **Observability-first**: every embedding operation attaches rich error context
  (framework identity, model info, batch metrics, SK routing fields, context
  snapshots) via `attach_context`.
- **Fail-safe context translation**: context translation via
  `context_from_semantic_kernel` must never break embeddings; failures are
  logged with snapshots and embeddings proceed without a core context.
- **Strict by default** (configurable): non-string inputs in batch operations
  are rejected unless `strict_text_types=False`, in which case they are treated
  as empty and receive zero-vector embeddings while preserving row alignment.
- **Async-safe sync usage**: sync APIs enforce guard rails to prevent calling
  them from inside an active asyncio event loop; callers are guided to their
  async counterparts with explicit error codes.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from functools import wraps
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, TypeVar, Callable, TypedDict

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

    _FRAMEWORK_VERSION: Optional[str] = getattr(_semantic_kernel, "__version__", None)
except Exception:  # noqa: BLE001
    _FRAMEWORK_VERSION = None

# ---------------------------------------------------------------------------
# Safe conditional import for Semantic Kernel base class
# ---------------------------------------------------------------------------

try:
    from semantic_kernel.connectors.ai.embeddings.embedding_generator_base import (  # type: ignore
        EmbeddingGeneratorBase,
    )

    SEMANTIC_KERNEL_AVAILABLE = True
except ImportError:  # pragma: no cover - only used when SK isn't installed

    class EmbeddingGeneratorBase:  # type: ignore[no-redef]
        """Fallback base class when Semantic Kernel is not installed."""

        def __init__(self, *_: Any, **__: Any) -> None:
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

    # Sync wrapper misuse (parity with other adapters)
    SYNC_WRAPPER_CALLED_IN_EVENT_LOOP = "SYNC_WRAPPER_CALLED_IN_EVENT_LOOP"


# Coercion configuration for the common embedding utils
EMBEDDING_COERCION_ERROR_CODES: CoercionErrorCodes = CoercionErrorCodes(
    invalid_result=ErrorCodes.INVALID_EMBEDDING_RESULT,
    empty_result=ErrorCodes.EMPTY_EMBEDDING_RESULT,
    conversion_error=ErrorCodes.EMBEDDING_CONVERSION_ERROR,
    framework_label=_FRAMEWORK_NAME,
)


# ---------------------------------------------------------------------------
# Context + config types
# ---------------------------------------------------------------------------


class SemanticKernelContext(TypedDict, total=False):
    """Structured type for Semantic Kernel execution context."""
    plugin_name: Optional[str]
    function_name: Optional[str]
    kernel_id: Optional[str]
    memory_type: Optional[str]
    request_id: Optional[str]
    user_id: Optional[str]
    # Allow arbitrary extra fields (captured via snapshot in error context)
    execution_settings: Any  # noqa: ANN401


class SemanticKernelAdapterConfig(TypedDict, total=False):
    """
    Adapter-level configuration for the Semantic Kernel embedding adapter.

    Fields
    ------
    enable_operation_context_propagation:
        If True, include the OperationContext instance in framework_ctx as
        `_operation_context` for downstream inspection. Defaults to True.

    strict_text_types:
        If True, reject non-string items in batch embedding calls rather than
        silently treating them as empty. Defaults to True.

    max_items_in_context:
        Maximum number of context items to include/snapshot for logs. Defaults to 100.
    """
    enable_operation_context_propagation: bool
    strict_text_types: bool
    max_items_in_context: int


# ---------------------------------------------------------------------------
# Helpers (validation, snapshots, event-loop guards, async closing)
# ---------------------------------------------------------------------------


def _safe_snapshot(value: Any, *, max_items: int = 100, max_str: int = 5_000) -> Any:
    """
    Best-effort conversion into a safe-to-log snapshot.

    - Limits container size to reduce log bloat
    - Truncates long strings
    - Falls back to repr() for unknown objects

    NOTE:
    - This is intended for observability payloads; it does not guarantee
      redaction of secrets, but it significantly reduces accidental large dumps.
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
        # Include set support for parity with other framework adapters (e.g., tags)
        if isinstance(value, (list, tuple, set)):
            out_list: List[Any] = []
            for idx, v in enumerate(list(value)):
                if idx >= max_items:
                    out_list.append(f"… truncated after {max_items} items")
                    break
                out_list.append(_safe_snapshot(v, max_items=max_items, max_str=max_str))
            return out_list
        return repr(value)
    except Exception:  # noqa: BLE001
        return {"repr": repr(value)}


def _validate_text_is_string(text: Any, *, op_name: str) -> None:
    if not isinstance(text, str):
        raise TypeError(f"{op_name} expects str; got {type(text).__name__}")


def _validate_texts_are_strings(texts: Sequence[Any], *, op_name: str) -> None:
    for i, t in enumerate(texts):
        # Tests look for "item 1 is int"/"item 1 is object" style wording.
        if not isinstance(t, str):
            raise TypeError(f"{op_name} expects Sequence[str]; item {i} is {type(t).__name__}")


def _normalize_sk_config(sk_config: Mapping[str, Any]) -> SemanticKernelAdapterConfig:
    """
    Validate and normalize adapter-level Semantic Kernel configuration.

    IMPORTANT:
    - Rejects unknown keys to prevent silent misconfiguration.
    - Coerces bool/int values defensively for robustness.
    """
    allowed_keys = {"enable_operation_context_propagation", "strict_text_types", "max_items_in_context"}
    unknown = set(dict(sk_config).keys()) - allowed_keys
    if unknown:
        raise ValueError(
            f"[{ErrorCodes.SEMANTIC_KERNEL_CONFIG_INVALID}] "
            f"sk_config contains unknown keys: {sorted(unknown)}"
        )

    cfg: SemanticKernelAdapterConfig = dict(sk_config)  # type: ignore[assignment]
    cfg.setdefault("enable_operation_context_propagation", True)
    cfg.setdefault("strict_text_types", True)
    cfg.setdefault("max_items_in_context", 100)

    cfg["enable_operation_context_propagation"] = bool(cfg["enable_operation_context_propagation"])
    cfg["strict_text_types"] = bool(cfg["strict_text_types"])
    cfg["max_items_in_context"] = int(cfg["max_items_in_context"])
    return cfg


def _ensure_not_in_event_loop(
    sync_api_name: str,
    *,
    async_alternative: Optional[str] = None,
) -> None:
    """
    Prevent deadlocks from calling sync APIs in async contexts.

    In async code, callers must use the async variants (e.g. generate_embedding_async).
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running event loop: safe to call sync method.
        return

    if async_alternative:
        suggestion = f"Use the async variant instead (e.g. 'await {async_alternative}(...)'). "
    else:
        suggestion = "Use the corresponding async variant instead. "
    raise RuntimeError(
        f"{sync_api_name} was called from inside an active asyncio event loop. "
        f"{suggestion}"
        f"[{ErrorCodes.SYNC_WRAPPER_CALLED_IN_EVENT_LOOP}]"
    )


def _maybe_close_sync(obj: Any) -> None:
    """
    Best-effort *sync* resource cleanup.

    Preference:
      1) aclose() if present:
           - if it returns a coroutine, run it via asyncio.run
      2) close() if present:
           - if coroutinefunction: asyncio.run(close())
           - else call directly; if return is coroutine: asyncio.run(return)

    IMPORTANT:
      Callers must ensure they are NOT in a running event loop (use _ensure_not_in_event_loop).
    """
    if obj is None:
        return

    aclose = getattr(obj, "aclose", None)
    if callable(aclose):
        res = aclose()
        if asyncio.iscoroutine(res):
            asyncio.run(res)
        return

    close = getattr(obj, "close", None)
    if not callable(close):
        return

    if asyncio.iscoroutinefunction(close):
        asyncio.run(close())
        return

    res = close()
    if asyncio.iscoroutine(res):
        asyncio.run(res)


async def _maybe_close_async(obj: Any) -> None:
    """
    Best-effort async resource cleanup with prioritization.

    - Prefer an async `aclose()` method if present (supports both async def and sync-returning-coroutine).
    - Fall back to a coroutine `close()` if defined.
    - Fall back to a sync `close()` executed in a worker thread.
    """
    if obj is None:
        return

    aclose = getattr(obj, "aclose", None)
    if callable(aclose):
        res = aclose()
        if asyncio.iscoroutine(res):
            await res
        return

    close = getattr(obj, "close", None)
    if not callable(close):
        return

    if asyncio.iscoroutinefunction(close):
        await close()
    else:
        await asyncio.to_thread(close)


# ---------------------------------------------------------------------------
# Error-context decorators with dynamic context extraction
# ---------------------------------------------------------------------------


def _extract_dynamic_context(
    instance: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    *,
    operation: str,
) -> Dict[str, Any]:
    """
    Extract dynamic context for richer observability.

    Captures:
    - model_id / model
    - framework_version
    - embedding_dim hint (when available)
    - text_len / texts_count / empty_texts_count
    - Semantic Kernel routing fields from sk_context
    - snapshot of sk_context (for nested/complex cases)
    """
    ctx: Dict[str, Any] = {
        "model": getattr(instance, "model_id", None) or "unknown",
        "model_id": getattr(instance, "model_id", None) or "unknown",
        "framework_version": _FRAMEWORK_VERSION,
    }

    dim_hint = getattr(instance, "_embedding_dim_hint", None)
    if isinstance(dim_hint, int):
        ctx["embedding_dim"] = dim_hint

    # Metrics (defensive: extraction must never break attach_context)
    try:
        if operation == "embedding_query":
            if args and isinstance(args[0], str):
                ctx["text_len"] = len(args[0])
        elif operation == "embedding_documents":
            if args:
                maybe_texts = args[0]
                # Strings are Sequences; avoid counting characters as documents.
                if isinstance(maybe_texts, Sequence) and not isinstance(maybe_texts, (str, bytes)):
                    try:
                        ctx["texts_count"] = len(maybe_texts)  # type: ignore[arg-type]
                    except Exception:
                        pass
                    try:
                        empty_count = 0
                        for t in maybe_texts:  # type: ignore[assignment]
                            if not isinstance(t, str) or not t.strip():
                                empty_count += 1
                        if empty_count:
                            ctx["empty_texts_count"] = empty_count
                    except Exception:
                        pass
    except Exception:
        pass

    sk_context = kwargs.get("sk_context")
    if isinstance(sk_context, Mapping) and sk_context:
        # Extract known routing fields
        try:
            for key in (
                "plugin_name",
                "function_name",
                "kernel_id",
                "memory_type",
                "request_id",
                "user_id",
            ):
                if key in sk_context:
                    ctx[key] = sk_context[key]
        except Exception:
            pass

        # Snapshot for nested/complex cases (execution_settings, etc.)
        try:
            cfg = getattr(instance, "sk_config", {}) or {}
            max_items = 100
            if isinstance(cfg, Mapping):
                try:
                    max_items = int(cfg.get("max_items_in_context", 100))
                except Exception:
                    max_items = 100
            ctx["sk_context_snapshot"] = _safe_snapshot(sk_context, max_items=max_items)
        except Exception:
            pass

    elif sk_context is not None and not isinstance(sk_context, Mapping):
        # Non-mapping contexts are tolerated; include a snapshot for debugging.
        try:
            ctx["sk_context_snapshot"] = _safe_snapshot({"repr": repr(sk_context)})
        except Exception:
            pass

    return ctx


def _create_error_context_decorator(
    operation: str,
    *,
    is_async: bool,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator factory that attaches structured error context.

    IMPORTANT:
    Tests expect ctx["operation"] to be exactly:
    - "embedding_documents" for document/batch calls
    - "embedding_query" for single/query calls
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if is_async:

            @wraps(func)
            async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
                dyn = _extract_dynamic_context(self, args, kwargs, operation=operation)
                try:
                    return await func(self, *args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    attach_context(
                        exc,
                        framework=_FRAMEWORK_NAME,
                        operation=operation,
                        error_codes=EMBEDDING_COERCION_ERROR_CODES,
                        **dyn,
                    )
                    raise

            return async_wrapper  # type: ignore[return-value]

        @wraps(func)
        def sync_wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
            dyn = _extract_dynamic_context(self, args, kwargs, operation=operation)
            try:
                return func(self, *args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                attach_context(
                    exc,
                    framework=_FRAMEWORK_NAME,
                    operation=operation,
                    error_codes=EMBEDDING_COERCION_ERROR_CODES,
                    **dyn,
                )
                raise

        return sync_wrapper  # type: ignore[return-value]

    return decorator


def with_embedding_error_context(operation: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    return _create_error_context_decorator(operation, is_async=False)


def with_async_embedding_error_context(operation: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    return _create_error_context_decorator(operation, is_async=True)


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
        *,
        embedding_dimension: Optional[int] = None,
        sk_config: Optional[Mapping[str, Any]] = None,
        **_: Any,
    ) -> None:
        """
        Initialize Corpus Semantic Kernel Embeddings.

        Notes
        -----
        - `sk_config` is adapter-level behavior config (not per-call context).
        - `embedding_dimension` is required if adapter lacks get_embedding_dimension().
        """
        # Common user mistakes: None or wrong type.
        if corpus_adapter is None or not hasattr(corpus_adapter, "embed") or not callable(
            getattr(corpus_adapter, "embed", None)
        ):
            raise TypeError(
                "corpus_adapter must implement an EmbeddingProtocolV1-compatible interface with an 'embed' method"
            )

        if sk_config is None:
            sk_config = {}
        if not isinstance(sk_config, Mapping):
            raise TypeError(
                f"sk_config must be a Mapping (dict-like), got {type(sk_config).__name__}"
            )

        super().__init__()  # no-op in fallback, required when SK installed

        self.corpus_adapter = corpus_adapter
        self.model_id = model_id
        self.batch_config = batch_config
        self.text_normalization_config = text_normalization_config

        self._embedding_dimension_override = embedding_dimension
        # Strict config validation + normalization
        self.sk_config: SemanticKernelAdapterConfig = _normalize_sk_config(sk_config)

        # Thread-safe translator lazy init
        self._translator_lock = threading.Lock()
        self._translator_instance: Optional[EmbeddingTranslator] = None

        # Best-effort embedding dimension hint (populated after first successful embed)
        self._embedding_dim_hint: Optional[int] = None

        # Enforce known embedding dimension to avoid incorrect fallbacks
        if (
            not hasattr(self.corpus_adapter, "get_embedding_dimension")
            and self._embedding_dimension_override is None
        ):
            raise ValueError(
                "Embedding dimension is unknown. Either implement "
                "`get_embedding_dimension()` on the corpus_adapter or pass "
                "`embedding_dimension=...` to CorpusSemanticKernelEmbeddings."
            )

    # ------------------------------------------------------------------ #
    # Translator + dimension
    # ------------------------------------------------------------------ #

    @property
    def _translator(self) -> EmbeddingTranslator:
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
                translator=None,
                batch_config=self.batch_config,
                text_normalization_config=self.text_normalization_config,
            )
            self._translator_instance = translator
            logger.debug(
                "EmbeddingTranslator initialized for Semantic Kernel with model_id=%s",
                self.model_id or "default",
            )
            return translator

    @property
    def embedding_dimension(self) -> int:
        """
        Get embedding dimension for proper zero vector fallback.

        Precedence:
        - Use adapter.get_embedding_dimension() when available and successful
        - Fall back to explicit embedding_dimension override if adapter fails
        """
        if hasattr(self.corpus_adapter, "get_embedding_dimension"):
            try:
                return int(self.corpus_adapter.get_embedding_dimension())
            except Exception as e:  # noqa: BLE001
                logger.debug("Failed to get embedding dimension from adapter: %s", e)
                if self._embedding_dimension_override is not None:
                    return int(self._embedding_dimension_override)
                raise

        if self._embedding_dimension_override is not None:
            return int(self._embedding_dimension_override)

        raise RuntimeError(
            "Embedding dimension is unknown. Adapter does not expose "
            "`get_embedding_dimension()` and no `embedding_dimension` override was provided."
        )

    def _update_dim_hint(self, dim: Optional[int]) -> None:
        """
        Thread-safe, best-effort dimension hint update.

        Uses first-write-wins semantics under the existing translator lock to
        avoid races in concurrent first-embed scenarios.
        """
        if dim is None:
            return
        if self._embedding_dim_hint is not None:
            return

        with self._translator_lock:
            if self._embedding_dim_hint is None:
                self._embedding_dim_hint = dim

    # ------------------------------------------------------------------ #
    # Context building (SK context → OperationContext + framework_ctx)
    # ------------------------------------------------------------------ #

    def _build_contexts(
        self,
        *,
        sk_context: Optional[Any] = None,
    ) -> Tuple[Optional[OperationContext], Dict[str, Any]]:
        """
        Build contexts for Semantic Kernel execution environment.

        - Tolerates non-mapping sk_context (logs + ignores)
        - Uses context_translation.from_semantic_kernel when possible
        - Never allows context translation to break embedding calls
        """
        core_ctx: Optional[OperationContext] = None
        framework_ctx: Dict[str, Any] = {
            "framework": _FRAMEWORK_NAME,
            "model": self.model_id or "unknown",
            "model_id": self.model_id or "unknown",
            "error_codes": EMBEDDING_COERCION_ERROR_CODES,
            # mirror adapter config into framework_ctx for downstream inspection
            "enable_operation_context_propagation": self.sk_config["enable_operation_context_propagation"],
            "strict_text_types": self.sk_config["strict_text_types"],
            "max_items_in_context": self.sk_config["max_items_in_context"],
        }
        if _FRAMEWORK_VERSION is not None:
            framework_ctx["framework_version"] = _FRAMEWORK_VERSION

        # Provide a best-effort dimension hint for downstream logging and tooling.
        try:
            dim_hint = self._embedding_dim_hint or self.embedding_dimension
        except Exception:
            dim_hint = self._embedding_dim_hint
        framework_ctx["embedding_dim_hint"] = dim_hint

        ctx_map: Optional[Mapping[str, Any]] = None
        if sk_context is None:
            ctx_map = None
        elif isinstance(sk_context, Mapping):
            ctx_map = sk_context
        else:
            # Must tolerate invalid types (tests expect no crash)
            logger.warning(
                "[%s] sk_context should be a Mapping, got %s; ignoring context",
                ErrorCodes.SEMANTIC_KERNEL_CONTEXT_INVALID,
                type(sk_context).__name__,
            )
            ctx_map = None

        if ctx_map is not None:
            # surface SK-specific fields into framework_ctx
            for key in (
                "plugin_name",
                "function_name",
                "kernel_id",
                "memory_type",
                "request_id",
                "user_id",
            ):
                if key in ctx_map:
                    framework_ctx[key] = ctx_map[key]

            try:
                candidate = context_from_semantic_kernel(dict(ctx_map))
                if isinstance(candidate, OperationContext):
                    core_ctx = candidate
            except Exception as e:  # noqa: BLE001
                # Context translation failures must never break embeddings.
                try:
                    attach_context(
                        e,
                        framework=_FRAMEWORK_NAME,
                        operation="context_build",
                        error_codes=EMBEDDING_COERCION_ERROR_CODES,
                        sk_context_snapshot=_safe_snapshot(
                            ctx_map,
                            max_items=self.sk_config["max_items_in_context"],
                        ),
                        framework_version=_FRAMEWORK_VERSION,
                    )
                except Exception:
                    pass

        # Optionally expose OperationContext for downstream inspection
        if core_ctx is not None and framework_ctx.get("enable_operation_context_propagation", True):
            framework_ctx["_operation_context"] = core_ctx

        return core_ctx, framework_ctx

    # ------------------------------------------------------------------ #
    # Coercion + empty handling
    # ------------------------------------------------------------------ #

    def _coerce_embedding_matrix(self, result: Any) -> List[List[float]]:
        return coerce_embedding_matrix(
            result=result,
            framework=_FRAMEWORK_NAME,
            error_codes=EMBEDDING_COERCION_ERROR_CODES,
            logger=logger,
        )

    def _coerce_embedding_vector(self, result: Any) -> List[float]:
        return coerce_embedding_vector(
            result=result,
            framework=_FRAMEWORK_NAME,
            error_codes=EMBEDDING_COERCION_ERROR_CODES,
            logger=logger,
        )

    def _handle_empty_text(self, _: str) -> List[float]:
        dim = self.embedding_dimension
        return [0.0] * dim

    def _warn_if_extreme_batch(self, texts: Sequence[Any], *, op_name: str) -> None:
        """
        Soft warning for extremely large batches.

        IMPORTANT:
        - Warnings must never fail. Filter to strings so lenient mode cannot break this.
        """
        safe_texts: List[str] = [t for t in texts if isinstance(t, str)]
        warn_if_extreme_batch(
            framework=_FRAMEWORK_NAME,
            texts=safe_texts,
            op_name=op_name,
            batch_config=self.batch_config,
            logger=logger,
        )

    # ------------------------------------------------------------------ #
    # Capabilities / health passthrough via EmbeddingTranslator
    # ------------------------------------------------------------------ #

    @with_embedding_error_context("capabilities")
    def capabilities(self) -> Mapping[str, Any]:
        """
        Sync capabilities passthrough.

        Delegates to EmbeddingTranslator.capabilities(), which centralizes
        async/sync adapter behavior and error context.
        """
        _ensure_not_in_event_loop("capabilities", async_alternative="acapabilities")
        return self._translator.capabilities()

    @with_async_embedding_error_context("capabilities_async")
    async def acapabilities(self) -> Mapping[str, Any]:
        """
        Async capabilities passthrough.

        Delegates to EmbeddingTranslator.arun_capabilities().
        """
        return await self._translator.arun_capabilities()

    @with_embedding_error_context("health")
    def health(self) -> Mapping[str, Any]:
        """
        Sync health passthrough.

        Delegates to EmbeddingTranslator.health().
        """
        _ensure_not_in_event_loop("health", async_alternative="ahealth")
        return self._translator.health()

    @with_async_embedding_error_context("health_async")
    async def ahealth(self) -> Mapping[str, Any]:
        """
        Async health passthrough.

        Delegates to EmbeddingTranslator.arun_health().
        """
        return await self._translator.arun_health()

    # ------------------------------------------------------------------ #
    # Resource management (context managers + explicit close)
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """
        Close underlying resources if they expose a close() method.

        This includes:
        - The underlying corpus_adapter
        - The EmbeddingTranslator if it was constructed and exposes close()

        Deadlock prevention:
        - Sync close must not be called from an active event loop.
          Use aclose() instead.
        """
        _ensure_not_in_event_loop("close", async_alternative="aclose")

        translator = self._translator_instance
        if isinstance(translator, EmbeddingTranslator):
            try:
                _maybe_close_sync(translator)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Error while closing embedding translator in close(): %s",
                    e,
                )

        try:
            _maybe_close_sync(self.corpus_adapter)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Error while closing embedding adapter in close(): %s",
                e,
            )

    async def aclose(self) -> None:
        """
        Async-close underlying resources if they expose async closers.

        Prefers:
        - translator.aclose() / translator.close()
        - corpus_adapter.aclose() / corpus_adapter.close()
        """
        translator = self._translator_instance
        if isinstance(translator, EmbeddingTranslator):
            try:
                await _maybe_close_async(translator)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Error while closing embedding translator in aclose(): %s",
                    e,
                )

        try:
            await _maybe_close_async(self.corpus_adapter)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Error while closing embedding adapter in aclose(): %s",
                e,
            )

    def __enter__(self) -> "CorpusSemanticKernelEmbeddings":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    async def __aenter__(self) -> "CorpusSemanticKernelEmbeddings":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.aclose()

    # ------------------------------------------------------------------ #
    # Unified embedding helpers (single + batch)
    # ------------------------------------------------------------------ #

    def _embed_single_text(self, text: str, *, sk_context: Any = None) -> List[float]:
        core_ctx, framework_ctx = self._build_contexts(sk_context=sk_context)

        # Ensure propagation behavior even if _build_contexts is monkeypatched in tests.
        if core_ctx is not None and framework_ctx.get("enable_operation_context_propagation", True):
            framework_ctx["_operation_context"] = core_ctx
        else:
            framework_ctx.pop("_operation_context", None)

        translated = self._translator.embed(
            raw_texts=text,
            op_ctx=core_ctx,
            framework_ctx=framework_ctx,
        )
        vec = self._coerce_embedding_vector(translated)
        self._update_dim_hint(len(vec))
        return vec

    async def _aembed_single_text(self, text: str, *, sk_context: Any = None) -> List[float]:
        core_ctx, framework_ctx = self._build_contexts(sk_context=sk_context)

        if core_ctx is not None and framework_ctx.get("enable_operation_context_propagation", True):
            framework_ctx["_operation_context"] = core_ctx
        else:
            framework_ctx.pop("_operation_context", None)

        translated = await self._translator.arun_embed(
            raw_texts=text,
            op_ctx=core_ctx,
            framework_ctx=framework_ctx,
        )
        vec = self._coerce_embedding_vector(translated)
        self._update_dim_hint(len(vec))
        return vec

    def _embed_text_batch(self, texts: Sequence[Any], *, sk_context: Any = None, op_name: str) -> List[List[float]]:
        texts_list = list(texts)

        if not texts_list:
            return []

        if self.sk_config["strict_text_types"]:
            _validate_texts_are_strings(texts_list, op_name=op_name)

        # When lenient, treat non-strings as empty.
        normalized: List[str] = []
        empty_indices: List[int] = []
        for i, t in enumerate(texts_list):
            if isinstance(t, str) and t.strip():
                normalized.append(t)
            else:
                empty_indices.append(i)

        self._warn_if_extreme_batch(texts_list, op_name=op_name)

        if not normalized:
            dim = self.embedding_dimension
            return [[0.0] * dim for _ in texts_list]

        core_ctx, framework_ctx = self._build_contexts(sk_context=sk_context)

        if core_ctx is not None and framework_ctx.get("enable_operation_context_propagation", True):
            framework_ctx["_operation_context"] = core_ctx
        else:
            framework_ctx.pop("_operation_context", None)

        translated = self._translator.embed(
            raw_texts=normalized,
            op_ctx=core_ctx,
            framework_ctx=framework_ctx,
        )
        mat = self._coerce_embedding_matrix(translated)

        dim = len(mat[0]) if mat else None
        self._update_dim_hint(dim)

        # Re-insert zero rows for empty inputs
        if empty_indices:
            inferred_dim = dim if dim is not None else self.embedding_dimension
            out: List[List[float]] = []
            j = 0
            empty_set = set(empty_indices)
            for i in range(len(texts_list)):
                if i in empty_set:
                    out.append([0.0] * inferred_dim)
                else:
                    out.append(mat[j])
                    j += 1
            return out

        return mat

    async def _aembed_text_batch(self, texts: Sequence[Any], *, sk_context: Any = None, op_name: str) -> List[List[float]]:
        texts_list = list(texts)

        if not texts_list:
            return []

        if self.sk_config["strict_text_types"]:
            _validate_texts_are_strings(texts_list, op_name=op_name)

        normalized: List[str] = []
        empty_indices: List[int] = []
        for i, t in enumerate(texts_list):
            if isinstance(t, str) and t.strip():
                normalized.append(t)
            else:
                empty_indices.append(i)

        self._warn_if_extreme_batch(texts_list, op_name=op_name)

        if not normalized:
            dim = self.embedding_dimension
            return [[0.0] * dim for _ in texts_list]

        core_ctx, framework_ctx = self._build_contexts(sk_context=sk_context)

        if core_ctx is not None and framework_ctx.get("enable_operation_context_propagation", True):
            framework_ctx["_operation_context"] = core_ctx
        else:
            framework_ctx.pop("_operation_context", None)

        translated = await self._translator.arun_embed(
            raw_texts=normalized,
            op_ctx=core_ctx,
            framework_ctx=framework_ctx,
        )
        mat = self._coerce_embedding_matrix(translated)

        dim = len(mat[0]) if mat else None
        self._update_dim_hint(dim)

        if empty_indices:
            inferred_dim = dim if dim is not None else self.embedding_dimension
            out: List[List[float]] = []
            j = 0
            empty_set = set(empty_indices)
            for i in range(len(texts_list)):
                if i in empty_set:
                    out.append([0.0] * inferred_dim)
                else:
                    out.append(mat[j])
                    j += 1
            return out

        return mat

    # ------------------------------------------------------------------ #
    # Core Semantic Kernel interface methods
    # ------------------------------------------------------------------ #

    @with_embedding_error_context("embedding_documents")
    def generate_embeddings(
        self,
        texts: Sequence[Any],
        *,
        sk_context: Any = None,
        **__: Any,
    ) -> List[List[float]]:
        _ensure_not_in_event_loop(
            "generate_embeddings",
            async_alternative="generate_embeddings_async",
        )
        # Tests expect type errors mentioning generate_embeddings specifically.
        return self._embed_text_batch(
            texts,
            sk_context=sk_context,
            op_name="generate_embeddings",
        )

    @with_embedding_error_context("embedding_query")
    def generate_embedding(
        self,
        text: Any,
        *,
        sk_context: Any = None,
        **__: Any,
    ) -> List[float]:
        _ensure_not_in_event_loop(
            "generate_embedding",
            async_alternative="generate_embedding_async",
        )

        if self.sk_config["strict_text_types"]:
            _validate_text_is_string(text, op_name="generate_embedding")

        if not isinstance(text, str):
            # lenient path: treat as empty
            return self._handle_empty_text("")

        if not text.strip():
            return self._handle_empty_text(text)

        return self._embed_single_text(text, sk_context=sk_context)

    @with_async_embedding_error_context("embedding_documents")
    async def generate_embeddings_async(
        self,
        texts: Sequence[Any],
        *,
        sk_context: Any = None,
        **__: Any,
    ) -> List[List[float]]:
        return await self._aembed_text_batch(
            texts,
            sk_context=sk_context,
            op_name="generate_embeddings_async",
        )

    @with_async_embedding_error_context("embedding_query")
    async def generate_embedding_async(
        self,
        text: Any,
        *,
        sk_context: Any = None,
        **__: Any,
    ) -> List[float]:
        if self.sk_config["strict_text_types"]:
            _validate_text_is_string(text, op_name="generate_embedding_async")

        if not isinstance(text, str):
            return self._handle_empty_text("")

        if not text.strip():
            return self._handle_empty_text(text)

        return await self._aembed_single_text(text, sk_context=sk_context)

    # ------------------------------------------------------------------ #
    # Convenience aliases (tests rely on these)
    # ------------------------------------------------------------------ #

    @with_embedding_error_context("embedding_documents")
    def embed_documents(self, texts: Sequence[Any], *, sk_context: Any = None, **kwargs: Any) -> List[List[float]]:
        _ensure_not_in_event_loop(
            "embed_documents",
            async_alternative="aembed_documents",
        )
        # Tests look for "embed_documents expects Sequence[str]" errors.
        if self.sk_config["strict_text_types"]:
            _validate_texts_are_strings(list(texts), op_name="embed_documents")
        return self.generate_embeddings(texts, sk_context=sk_context, **kwargs)

    @with_embedding_error_context("embedding_query")
    def embed_query(self, text: Any, *, sk_context: Any = None, **kwargs: Any) -> List[float]:
        _ensure_not_in_event_loop(
            "embed_query",
            async_alternative="aembed_query",
        )
        if self.sk_config["strict_text_types"]:
            _validate_text_is_string(text, op_name="embed_query")
        return self.generate_embedding(text, sk_context=sk_context, **kwargs)

    @with_async_embedding_error_context("embedding_documents")
    async def aembed_documents(self, texts: Sequence[Any], *, sk_context: Any = None, **kwargs: Any) -> List[List[float]]:
        if self.sk_config["strict_text_types"]:
            _validate_texts_are_strings(list(texts), op_name="aembed_documents")
        return await self.generate_embeddings_async(texts, sk_context=sk_context, **kwargs)

    @with_async_embedding_error_context("embedding_query")
    async def aembed_query(self, text: Any, *, sk_context: Any = None, **kwargs: Any) -> List[float]:
        if self.sk_config["strict_text_types"]:
            _validate_text_is_string(text, op_name="aembed_query")
        return await self.generate_embedding_async(text, sk_context=sk_context, **kwargs)


# ------------------------------------------------------------------ #
# Semantic Kernel service registration helpers
# ------------------------------------------------------------------ #


def configure_semantic_kernel_embeddings(
    corpus_adapter: EmbeddingProtocolV1,
    model_id: Optional[str] = None,
    sk_config: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> CorpusSemanticKernelEmbeddings:
    """
    Construct and return a `CorpusSemanticKernelEmbeddings` instance.

    This mirrors the shape of other framework helpers: always returns the
    embeddings instance, and callers can optionally register it elsewhere.
    """
    return CorpusSemanticKernelEmbeddings(
        corpus_adapter=corpus_adapter,
        model_id=model_id,
        sk_config=sk_config,
        **kwargs,
    )


def register_with_semantic_kernel(
    kernel: Any,
    corpus_adapter: EmbeddingProtocolV1,
    service_id: Optional[str] = None,
    model_id: Optional[str] = None,
    **kwargs: Any,
) -> CorpusSemanticKernelEmbeddings:
    """
    Register Corpus embeddings as a service with Semantic Kernel.

    Registration strategy:
    1) kernel.add_service(service, service_id=...)
    2) kernel.register_embedding_generation(service, service_id=...)
    3) otherwise: do nothing, still return embeddings (tests require no raise)
    """
    if kernel is None:
        raise ValueError("kernel cannot be None")

    embeddings = CorpusSemanticKernelEmbeddings(
        corpus_adapter=corpus_adapter,
        model_id=model_id,
        **kwargs,
    )

    # Prefer add_service when available
    add_service = getattr(kernel, "add_service", None)
    if callable(add_service):
        try:
            add_service(embeddings, service_id=service_id)
            return embeddings
        except TypeError:
            pass

    # Fallback: older/alt API
    reg = getattr(kernel, "register_embedding_generation", None)
    if callable(reg):
        try:
            reg(embeddings, service_id=service_id)
            return embeddings
        except TypeError:
            pass

    # No known methods; return embeddings without raising
    return embeddings


__all__ = [
    "CorpusSemanticKernelEmbeddings",
    "SEMANTIC_KERNEL_AVAILABLE",
    "SemanticKernelContext",
    "ErrorCodes",
    "register_with_semantic_kernel",
    "configure_semantic_kernel_embeddings",
]
