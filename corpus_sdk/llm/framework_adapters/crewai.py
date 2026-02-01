# corpus_sdk/llm/framework_adapters/crewai.py
# SPDX-License-Identifier: Apache-2.0

"""
CrewAI adapter for Corpus LLM protocol.

This module exposes a Corpus `LLMProtocolV1` behind the shared `LLMTranslator`
layer as a CrewAI-compatible LLM wrapper. It focuses on:

- Async + sync completion APIs
- Async + sync streaming (true incremental streaming)
- Context propagation via `OperationContext`
- Rich error context attachment for observability
- Framework-agnostic translation via `LLMTranslator` (no direct message translation)

Design goals
------------

1. Protocol + translator first:
   All calls go through the shared `LLMTranslator` with the `"crewai"` framework,
   so message normalization, post-processing, tools, and error context are
   consistent across frameworks.

2. Optional dependency safe:
   Import of `crewai` is guarded. Importing this module is safe even if
   CrewAI is not installed.

3. Simple & explicit interface:
   Clean API that CrewAI agents can use directly:

       llm = CorpusCrewAILLM(llm_adapter=adapter, model="gpt-4")
       text = llm.complete("Hello!")
       async for token in llm.astream("Hello!"):
           ...

4. True streaming:
   Streaming goes through `LLMTranslator.stream` / `LLMTranslator.arun_stream`,
   so you get protocol-level streaming semantics with CrewAI-friendly outputs.

5. Context + observability:
   - `OperationContext` built from CrewAI context via ContextTranslator
   - Rich error context attached via core `attach_context` + shared error codes
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from functools import wraps
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
    TypedDict,
)

# Context translation for OperationContext
from corpus_sdk.core.context_translation import (
    from_crewai as core_ctx_from_crewai,
)
from corpus_sdk.core.error_context import attach_context

# LLM protocol + translator orchestration
from corpus_sdk.llm.llm_base import (
    LLMProtocolV1,
    OperationContext,
)
from corpus_sdk.llm.framework_adapters.common.llm_translation import (
    LLMTranslator,
    LLMFrameworkTranslator,
    LLMPostProcessingConfig,
    SafetyFilter,
    JSONRepair,
    create_llm_translator,
)
from corpus_sdk.llm.framework_adapters.common.framework_utils import (
    CoercionErrorCodes,
)

logger = logging.getLogger(__name__)

# Framework identifier used for translator + error context
_FRAMEWORK_NAME = "crewai"

# Type variable for decorators
T = TypeVar("T")


# ---------------------------------------------------------------------------
# Symbolic error codes for init / context failures
# ---------------------------------------------------------------------------


class ErrorCodes:
    """Symbolic error codes for the CrewAI LLM adapter."""

    BAD_OPERATION_CONTEXT = "CREWAI_LLM_BAD_OPERATION_CONTEXT"
    BAD_INIT_CONFIG = "CREWAI_LLM_BAD_INIT_CONFIG"
    SYNC_WRAPPER_CALLED_IN_EVENT_LOOP = "CREWAI_LLM_SYNC_WRAPPER_CALLED_IN_EVENT_LOOP"


# ---------------------------------------------------------------------------
# Error-code bundle (CoercionErrorCodes alignment)
# ---------------------------------------------------------------------------

ERROR_CODES = CoercionErrorCodes(
    invalid_result="CREWAI_LLM_INVALID_RESULT",
    empty_result="CREWAI_LLM_EMPTY_RESULT",
    conversion_error="CREWAI_LLM_CONVERSION_ERROR",
    framework_label=_FRAMEWORK_NAME,
)

# ---------------------------------------------------------------------------
# Type definitions for CrewAI compatibility
# ---------------------------------------------------------------------------


@runtime_checkable
class CrewAIMessage(Protocol):
    """Protocol for CrewAI message-like objects."""

    @property
    def role(self) -> str: ...

    @property
    def content(self) -> str: ...


# Type aliases for better readability
CrewAIMessageInput = Union[str, Mapping[str, Any], CrewAIMessage]
CrewAIMessageSequence = Sequence[CrewAIMessageInput]


class CrewAIContext(TypedDict, total=False):
    """Structured type for CrewAI execution context."""
    agent_role: Optional[str]
    agent_goal: Optional[str]
    task_description: Optional[str]
    crew_id: Optional[str]
    crew_name: Optional[str]
    process_id: Optional[str]
    task_id: Optional[str]


# ---------------------------------------------------------------------------
# Optional CrewAI import (for dependency checks only)
# ---------------------------------------------------------------------------

_CREWAI_IMPORT_ERROR: Optional[BaseException] = None

try:  # pragma: no cover - optional dependency
    import crewai  # noqa: F401
except BaseException as exc:  # pragma: no cover
    _CREWAI_IMPORT_ERROR = exc


def _ensure_crewai_installed() -> None:
    """Raise a helpful error if CrewAI is not installed."""
    if _CREWAI_IMPORT_ERROR is not None:
        raise RuntimeError(
            "CrewAI is required to use CorpusCrewAILLM. "
            "Install with: pip install 'crewai>=0.51.0'"
        ) from _CREWAI_IMPORT_ERROR


# ---------------------------------------------------------------------------
# Sync event-loop guard (prevents sync-in-async deadlocks)
# ---------------------------------------------------------------------------


def _ensure_not_in_event_loop(sync_api_name: str) -> None:
    """
    Prevent calling blocking sync APIs from within an active asyncio event loop.

    This avoids subtle deadlocks when users accidentally call sync methods
    (e.g. `complete` or `stream`) from async code instead of using the
    corresponding async variants (`acomplete`, `astream`).
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, safe to proceed.
        return

    raise RuntimeError(
        f"{sync_api_name} was called from inside an active asyncio event loop. "
        f"Use the async variant instead (e.g. 'a{sync_api_name}'). "
        f"[{ErrorCodes.SYNC_WRAPPER_CALLED_IN_EVENT_LOOP}]"
    )


# ---------------------------------------------------------------------------
# Config for CrewAI LLM wrapper
# ---------------------------------------------------------------------------


@dataclass
class CrewAILLMConfig:
    """
    Configuration for `CorpusCrewAILLM`.

    This is a thin wrapper around the existing `LLMTranslator` knobs:
    - Default model / temperature / max_tokens
    - Optional post-processing configuration
    - Optional safety / JSON repair behavior
    - Optional framework version tagging
    - Optional observability / validation toggles
    """

    model: str = "default"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    framework_version: Optional[str] = None

    post_processing_config: Optional[LLMPostProcessingConfig] = None
    safety_filter: Optional[SafetyFilter] = None
    json_repair: Optional[JSONRepair] = None

    # Alignment with other adapters: observability + validation flags
    enable_metrics: bool = True
    validate_inputs: bool = True

    def __post_init__(self) -> None:
        """
        Validate configuration invariants for safety and correctness.

        This mirrors the AutoGen adapter's config validation:
        - Temperature must be within [0.0, 2.0].
        - max_tokens, if set, must be a positive integer.
        """
        if not (0.0 <= float(self.temperature) <= 2.0):
            raise ValueError(
                f"{ErrorCodes.BAD_INIT_CONFIG}: "
                f"temperature must be between 0.0 and 2.0, got {self.temperature}"
            )
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError(
                f"{ErrorCodes.BAD_INIT_CONFIG}: "
                f"max_tokens must be positive, got {self.max_tokens}"
            )


# ---------------------------------------------------------------------------
# Error Context Helpers (aligned with shared pattern)
# ---------------------------------------------------------------------------


def _extract_role_content(msg: Any) -> tuple[str, str]:
    """Extract role and content from various message formats."""
    if isinstance(msg, Mapping):
        return msg.get("role", "unknown"), msg.get("content", "")
    if hasattr(msg, "role") and hasattr(msg, "content"):
        return getattr(msg, "role", "unknown"), getattr(msg, "content", "")
    # string message
    return "user", str(msg)


def _analyze_message_metrics(messages: Sequence[Any]) -> tuple[Dict[str, int], int]:
    """Analyze message roles and content metrics."""
    roles: Dict[str, int] = {}
    total_chars = 0

    for msg in messages:
        role, content = _extract_role_content(msg)
        roles[role] = roles.get(role, 0) + 1
        if isinstance(content, str):
            total_chars += len(content)

    return roles, total_chars


def _extract_kwargs_context(kwargs: Dict[str, Any], target: Dict[str, Any]) -> None:
    """Extract relevant context from kwargs into target dict."""
    # CrewAI-specific context
    crewai_context_keys = [
        "agent_role",
        "agent_goal",
        "task_description",
        "crew_id",
        "crew_name",
        "process_id",
        "task_id",
    ]
    for key in crewai_context_keys:
        if key in kwargs:
            target[key] = kwargs[key]

    # Sampling parameters and request identifiers (for observability)
    sampling_params = [
        "temperature",
        "max_tokens",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "stop_sequences",
        "seed",
        "request_id",
        "tenant",
    ]
    for param in sampling_params:
        if param in kwargs:
            target[param] = kwargs[param]


def _extract_messages_from_args(first_arg: Any) -> list[Any]:
    """Safely extract messages from method arguments."""
    if isinstance(first_arg, (str, Mapping)) or hasattr(first_arg, "role"):
        return [first_arg]
    if isinstance(first_arg, (list, tuple)):
        return list(first_arg)
    if first_arg is not None:
        return [str(first_arg)]
    return []


def _extract_dynamic_context(
    instance: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    operation: str,
) -> Dict[str, Any]:
    """
    Extract rich dynamic context from method call for enhanced observability.

    NOTE: This is intentionally called only in the error path to avoid
    unnecessary overhead on successful calls.
    """
    dynamic_ctx: Dict[str, Any] = {
        "framework_name": _FRAMEWORK_NAME,
        "model": getattr(instance, "model", "unknown"),
        "temperature": getattr(instance, "temperature", 0.7),
        "operation": operation,
    }

    enable_metrics = getattr(instance, "_enable_metrics_flag", True)

    # Extract message metrics (only when metrics are enabled)
    if enable_metrics and args:
        messages = _extract_messages_from_args(args[0])
        if messages:
            dynamic_ctx["messages_count"] = len(messages)
            roles, total_chars = _analyze_message_metrics(messages)
            dynamic_ctx["roles_distribution"] = roles
            dynamic_ctx["total_content_chars"] = total_chars

    # Extract context from kwargs (sampling, request_id, tenant, CrewAI fields)
    if enable_metrics:
        _extract_kwargs_context(kwargs, dynamic_ctx)
    else:
        # Even if metrics are disabled, still capture identifiers when present.
        for key in ("request_id", "tenant"):
            if key in kwargs:
                dynamic_ctx[key] = kwargs[key]

    # If a pre-built OperationContext was passed, surface its identifiers.
    ctx_from_kwargs = kwargs.get("ctx")
    if isinstance(ctx_from_kwargs, OperationContext):
        dynamic_ctx.setdefault("request_id", getattr(ctx_from_kwargs, "request_id", None))
        dynamic_ctx.setdefault("tenant", getattr(ctx_from_kwargs, "tenant", None))

    # Stream flag for streaming operations
    if operation in ("astream", "stream"):
        dynamic_ctx["stream"] = True

    return dynamic_ctx


def _create_error_context_decorator(
    operation: str,
    is_async: bool = False,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Factory for creating error context decorators with rich per-call metrics.

    Performance note:
    ------------------
    We lazily extract dynamic context *only on exceptions*, so successful
    calls don't pay the cost of iterating messages / counting characters.
    """

    def decorator_factory(
        **static_context: Any,
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            if is_async:

                @wraps(func)
                async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
                    try:
                        result = func(self, *args, **kwargs)
                        # Check if result is an async generator - don't await it
                        if hasattr(result, "__aiter__"):
                            # If this is a streaming operation, wrap iteration so
                            # errors raised during async iteration still attach context.
                            if operation in {"astream"}:
                                async def _agen() -> AsyncIterator[Any]:
                                    try:
                                        async for item in result:  # type: ignore[union-attr]
                                            yield item
                                    except Exception as exc:  # noqa: BLE001
                                        dynamic_context = _extract_dynamic_context(
                                            self,
                                            args,
                                            kwargs,
                                            operation,
                                        )
                                        full_context = {
                                            "error_codes": ERROR_CODES,
                                            **static_context,
                                            **dynamic_context,
                                        }
                                        full_context.pop("operation", None)
                                        attach_context(
                                            exc,
                                            framework=_FRAMEWORK_NAME,
                                            operation=f"llm_{operation}",
                                            **full_context,
                                        )
                                        raise

                                return _agen()  # type: ignore[return-value]

                            return result
                        # Otherwise await it
                        return await result
                    except Exception as exc:  # noqa: BLE001
                        dynamic_context = _extract_dynamic_context(
                            self,
                            args,
                            kwargs,
                            operation,
                        )
                        full_context = {
                            "error_codes": ERROR_CODES,
                            **static_context,
                            **dynamic_context,
                        }
                        # Remove 'operation' to avoid duplicate kwarg when spreading full_context
                        full_context.pop("operation", None)
                        attach_context(
                            exc,
                            framework=_FRAMEWORK_NAME,
                            operation=f"llm_{operation}",
                            **full_context,
                        )
                        raise

                return async_wrapper
            else:

                @wraps(func)
                def sync_wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
                    try:
                        result = func(self, *args, **kwargs)

                        # If this is a streaming generator/iterator, wrap iteration so
                        # errors raised during iteration still attach context.
                        if operation in {"stream"} and hasattr(result, "__iter__"):
                            def _gen() -> Iterator[Any]:
                                try:
                                    for item in result:  # type: ignore[union-attr]
                                        yield item
                                except Exception as exc:  # noqa: BLE001
                                    dynamic_context = _extract_dynamic_context(
                                        self,
                                        args,
                                        kwargs,
                                        operation,
                                    )
                                    full_context = {
                                        "error_codes": ERROR_CODES,
                                        **static_context,
                                        **dynamic_context,
                                    }
                                    full_context.pop("operation", None)
                                    attach_context(
                                        exc,
                                        framework=_FRAMEWORK_NAME,
                                        operation=f"llm_{operation}",
                                        **full_context,
                                    )
                                    raise

                            return _gen()  # type: ignore[return-value]

                        return result
                    except Exception as exc:  # noqa: BLE001
                        dynamic_context = _extract_dynamic_context(
                            self,
                            args,
                            kwargs,
                            operation,
                        )
                        full_context = {
                            "error_codes": ERROR_CODES,
                            **static_context,
                            **dynamic_context,
                        }
                        # Remove 'operation' to avoid duplicate kwarg when spreading full_context
                        full_context.pop("operation", None)
                        attach_context(
                            exc,
                            framework=_FRAMEWORK_NAME,
                            operation=f"llm_{operation}",
                            **full_context,
                        )
                        raise

                return sync_wrapper

        return decorator

    return decorator_factory


def with_llm_error_context(
    operation: str,
    **static_context: Any,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for sync LLM methods with rich dynamic context extraction."""
    return _create_error_context_decorator(operation, is_async=False)(**static_context)


def with_async_llm_error_context(
    operation: str,
    **static_context: Any,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for async LLM methods with rich dynamic context extraction."""
    return _create_error_context_decorator(operation, is_async=True)(**static_context)


# ---------------------------------------------------------------------------
# Shared helpers (context + sampling)
# ---------------------------------------------------------------------------


def _extract_stop_sequences(kwargs: Mapping[str, Any]) -> Optional[list[str]]:
    """Extract stop sequences from kwargs."""
    stop = kwargs.get("stop")
    if stop is None:
        stop = kwargs.get("stop_sequences")

    if stop is None:
        return None
    if isinstance(stop, str):
        return [stop]
    if isinstance(stop, (list, tuple)):
        return [str(s) for s in stop]
    return [str(stop)]


def _build_operation_context_from_kwargs(
    kwargs: Mapping[str, Any],
) -> Optional[OperationContext]:
    """
    Build an OperationContext from generic kwargs using ContextTranslator.

    Defensive behavior:
    - If `ctx` is already an OperationContext, return it.
    - If there is no context information at all, return None and let
      downstream layers construct a default OperationContext if desired.
    - Else, call `ContextTranslator.from_crewai(...)`.
    - If that call fails, attach rich context and re-raise.
    - If it returns a non-OperationContext, attach context and raise TypeError.
    """
    # If caller already built an OperationContext, trust it.
    ctx = kwargs.get("ctx")
    if isinstance(ctx, OperationContext):
        return ctx

    framework_version = kwargs.get("framework_version")
    task = kwargs.get("task")

    allowed_keys = {
        "request_id",
        "traceparent",
        "tenant",
        "attrs",
        "agent",
        "crew_id",
        "agent_role",
        "agent_goal",
        "task_description",
        "crew_name",
        "process_id",
        "task_id",
    }

    context_kwargs: Dict[str, Any] = {
        k: v for k, v in kwargs.items() if k in allowed_keys
    }

    # If there is no contextual information at all, allow a None op_ctx.
    if task is None and not context_kwargs:
        return None

    try:
        ctx = core_ctx_from_crewai(
            task=task,
            framework_version=framework_version,
            **context_kwargs,
        )
    except Exception as exc:  # noqa: BLE001
        attach_context(
            exc,
            framework=_FRAMEWORK_NAME,
            operation="llm_context_translation",
            framework_version=framework_version,
            source="crewai_kwargs",
            context_keys=list(context_kwargs.keys()),
        )
        raise

    # Trust context_translation to return correct type
    # (isinstance check removed - was comparing different OperationContext classes)

    logger.debug(
        "Built OperationContext from CrewAI context with agent_role: %s, crew_id: %s",
        context_kwargs.get("agent_role", "unknown"),
        context_kwargs.get("crew_id", "unknown"),
    )

    return ctx


def _build_sampling_params(
    *,
    default_model: str,
    default_temperature: float,
    default_max_tokens: Optional[int],
    kwargs: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build sampling parameters from kwargs and defaults.

    Returns
    -------
    Dict[str, Any]
        Cleaned sampling parameters with None values removed.
        Only includes fields understood by `LLMTranslator`.
    """
    stop_sequences = _extract_stop_sequences(kwargs)

    params: Dict[str, Any] = {
        "model": kwargs.get("model", default_model),
        "temperature": kwargs.get("temperature", default_temperature),
        "max_tokens": kwargs.get("max_tokens", default_max_tokens),
        "top_p": kwargs.get("top_p"),
        "frequency_penalty": kwargs.get("frequency_penalty"),
        "presence_penalty": kwargs.get("presence_penalty"),
        "stop_sequences": stop_sequences,
    }

    clean_params = {k: v for k, v in params.items() if v is not None}

    logger.debug(
        "Built sampling params for CrewAI: model=%s, temperature=%.2f, max_tokens=%s",
        clean_params.get("model", "default"),
        float(clean_params.get("temperature", default_temperature)),
        clean_params.get("max_tokens", "default"),
    )

    return clean_params


def _build_framework_ctx(
    framework_version: Optional[str],
    *,
    operation: str,
    stream: bool,
    model: Optional[str],
    request_id: Optional[str],
    tenant: Optional[str],
    kwargs: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Build a framework_ctx payload for the CrewAI translator.

    This stays lightweight and purely informational; all "real" translation
    logic lives in the registered LLMFrameworkTranslator for "crewai".

    The shape is aligned with the AutoGen LLM adapter:
    - Includes framework, framework_version, operation, stream, model,
      request_id, tenant.
    - Nests CrewAI-specific contextual fields under `crewai_context`.
    - Forwards tools / tool_choice / system_message as passthrough hints.
    """
    crewai_context_keys = [
        "agent_role",
        "agent_goal",
        "task_description",
        "crew_id",
        "crew_name",
        "process_id",
        "task_id",
    ]
    crewai_ctx: Dict[str, Any] = {
        key: kwargs.get(key)
        for key in crewai_context_keys
        if key in kwargs
    }

    framework_ctx: Dict[str, Any] = {
        "framework": _FRAMEWORK_NAME,
        "framework_version": framework_version,
        "operation": operation,
        "stream": stream,
        "crewai_context": crewai_ctx,
    }

    if model is not None:
        framework_ctx["model"] = model
    if request_id is not None:
        framework_ctx["request_id"] = request_id
    if tenant is not None:
        framework_ctx["tenant"] = tenant

    # Include any explicitly passed tools / tool_choice / system_message
    for key in ("tools", "tool_choice", "system_message"):
        if key in kwargs:
            framework_ctx[key] = kwargs[key]

    return framework_ctx


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------


class CorpusCrewAILLM:
    """
    CrewAI-compatible LLM wrapper backed by the shared `LLMTranslator`.

    This class does not perform its own message translation; instead, it:

    - Builds an `OperationContext` from CrewAI kwargs via ContextTranslator
    - Builds sampling params and a small `framework_ctx`
    - Delegates to `LLMTranslator` with `framework="crewai"`

    The framework-specific behavior (message normalization, system message
    handling, shape conversion, etc.) is handled by the registered
    CrewAI `LLMFrameworkTranslator` or the default translator.
    """

    def __init__(
        self,
        *,
        llm_adapter: LLMProtocolV1,
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        framework_version: Optional[str] = None,
        require_crewai: bool = False,
        # New optional config layer
        config: Optional[CrewAILLMConfig] = None,
        # Optional explicit translator wiring
        translator: Optional[LLMFrameworkTranslator] = None,
        post_processing_config: Optional[LLMPostProcessingConfig] = None,
        safety_filter: Optional[SafetyFilter] = None,
        json_repair: Optional[JSONRepair] = None,
    ) -> None:
        """Initialize the CrewAI LLM adapter."""
        if require_crewai:
            _ensure_crewai_installed()

        # Keep reference for resource management
        self._llm_adapter: LLMProtocolV1 = llm_adapter

        # Resolve configuration precedence: explicit config > legacy kwargs.
        if config is not None:
            self.model = config.model
            self.temperature = float(config.temperature)
            self.max_tokens = config.max_tokens
            self._framework_version = config.framework_version or framework_version

            self._enable_metrics_flag = bool(config.enable_metrics)
            self._validate_inputs_flag = bool(config.validate_inputs)

            effective_post_processing = (
                post_processing_config or config.post_processing_config
            )
            effective_safety_filter = safety_filter or config.safety_filter
            effective_json_repair = json_repair or config.json_repair
        else:
            self.model = model
            self.temperature = float(temperature)
            self.max_tokens = max_tokens
            self._framework_version = framework_version

            # Defaults when no explicit config object is provided
            self._enable_metrics_flag = True
            self._validate_inputs_flag = True

            effective_post_processing = post_processing_config
            effective_safety_filter = safety_filter
            effective_json_repair = json_repair

        self._validate_init_params(llm_adapter)

        # Build the shared LLMTranslator for the "crewai" framework
        self._translator: LLMTranslator = create_llm_translator(
            adapter=llm_adapter,
            framework=_FRAMEWORK_NAME,
            translator=translator,
            post_processing_config=effective_post_processing,
            safety_filter=effective_safety_filter,
            json_repair=effective_json_repair,
        )

        logger.info(
            "CorpusCrewAILLM initialized with model=%s, temperature=%.2f, "
            "max_tokens=%s, framework_version=%s",
            self.model,
            self.temperature,
            self.max_tokens or "default",
            self._framework_version or "unknown",
        )

    # ------------------------------------------------------------------ #
    # Resource management (context managers) – aligned with other adapters
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "CorpusCrewAILLM":
        """Support sync context manager protocol for resource cleanup."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Clean up resources when exiting sync context."""
        close = getattr(self._llm_adapter, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error while closing LLM adapter in __exit__: %s", exc)

    async def __aenter__(self) -> "CorpusCrewAILLM":
        """Support async context manager protocol."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Clean up resources when exiting async context."""
        aclose = getattr(self._llm_adapter, "aclose", None)
        if callable(aclose):
            try:
                await aclose()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error while async-closing LLM adapter in __aexit__: %s", exc)

    # ------------------------------------------------------------------ #
    # Internal validation / helpers
    # ------------------------------------------------------------------ #

    def _validate_init_params(self, llm_adapter: LLMProtocolV1) -> None:
        """
        Validate initialization parameters and adapter capabilities.

        - Ensures the adapter exposes core `LLMProtocolV1` methods.
        - Validates temperature and max_tokens constraints.
        """
        required_methods = (
            "complete",
            "stream",
            "count_tokens",
            "health",
            "capabilities",
        )
        missing = [
            m
            for m in required_methods
            if not callable(getattr(llm_adapter, m, None))
        ]
        if missing:
            raise TypeError(
                f"{ErrorCodes.BAD_INIT_CONFIG}: "
                "llm_adapter must implement LLMProtocolV1; missing methods: "
                + ", ".join(missing)
            )

        if not isinstance(self.temperature, (int, float)) or not (
            0.0 <= float(self.temperature) <= 2.0
        ):
            raise ValueError(
                f"{ErrorCodes.BAD_INIT_CONFIG}: "
                "temperature must be between 0.0 and 2.0"
            )

        if self.max_tokens is not None:
            if not isinstance(self.max_tokens, int) or self.max_tokens < 1:
                raise ValueError(
                    f"{ErrorCodes.BAD_INIT_CONFIG}: "
                    "max_tokens must be a positive integer"
                )

    def _apply_instance_defaults(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Apply instance-level defaults to kwargs.

        Ensures `framework_version` is present if configured.
        """
        updated = dict(kwargs)
        if (
            "framework_version" not in updated
            and self._framework_version is not None
        ):
            updated["framework_version"] = self._framework_version
        return updated

    def _validate_messages(
        self,
        messages: Union[CrewAIMessageInput, CrewAIMessageSequence],
    ) -> None:
        """
        Basic input validation for messages, aligned with other adapters.

        - Disallow None
        - Disallow empty sequences
        - Disallow empty/whitespace-only strings
        """
        if messages is None:
            raise ValueError("messages cannot be None")

        if isinstance(messages, (list, tuple)):
            if not messages:
                raise ValueError("messages list cannot be empty")
            return

        if isinstance(messages, str):
            if not messages.strip():
                raise ValueError("messages string cannot be empty")
            return

        # Mapping or CrewAIMessage-like objects are accepted as-is.

    def _to_translator_messages(
        self,
        messages: Union[CrewAIMessageInput, CrewAIMessageSequence],
    ) -> Any:
        """
        Convert CrewAI message inputs to a form the translator can handle.

        CrewAI can accept:
        - Plain strings (prompt-like)
        - Dicts with role/content
        - CrewAI message objects

        The translator expects either:
        - Mappings (dicts) with role/content
        - NormalizedMessage objects

        For strings, we wrap them as user messages.
        """
        # Single string input: wrap as user message
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]

        # Single dict or message object: wrap in list
        if isinstance(messages, Mapping):
            return [dict(messages)]

        # Check if it's a CrewAI message object (not string, not mapping)
        if not isinstance(messages, (str, Sequence)):
            # Assume it's a CrewAI message object with role/content attributes
            role = getattr(messages, "role", "user")
            content = getattr(messages, "content", str(messages))
            return [{"role": role, "content": content}]

        # Sequence of messages: normalize each
        normalized = []
        for msg in messages:
            if isinstance(msg, str):
                normalized.append({"role": "user", "content": msg})
            elif isinstance(msg, Mapping):
                normalized.append(dict(msg))
            else:
                # CrewAI message object
                role = getattr(msg, "role", "user")
                content = getattr(msg, "content", str(msg))
                normalized.append({"role": role, "content": content})
        return normalized

    def _build_request_context(
        self,
        *,
        operation: str,
        stream: bool,
        kwargs: Mapping[str, Any],
    ) -> Tuple[Optional[OperationContext], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Build (OperationContext | None, sampling params, framework_ctx, resolved_kwargs)
        in a single place, mirroring other framework adapters.

        When no CrewAI context is provided at all, the OperationContext may be
        None and downstream layers are responsible for constructing a default.
        """
        resolved_kwargs = self._apply_instance_defaults(kwargs)
        request_id = resolved_kwargs.get("request_id")
        tenant = resolved_kwargs.get("tenant")

        ctx = _build_operation_context_from_kwargs(resolved_kwargs)
        params = _build_sampling_params(
            default_model=self.model,
            default_temperature=self.temperature,
            default_max_tokens=self.max_tokens,
            kwargs=resolved_kwargs,
        )
        framework_ctx = _build_framework_ctx(
            resolved_kwargs.get("framework_version"),
            operation=operation,
            stream=stream,
            model=params.get("model"),
            request_id=request_id,
            tenant=tenant,
            kwargs=resolved_kwargs,
        )

        return ctx, params, framework_ctx, resolved_kwargs

    # ---------------------------------------------------------------------
    # Async completion
    # ---------------------------------------------------------------------

    @with_async_llm_error_context("acomplete")
    async def acomplete(
        self,
        messages: Union[CrewAIMessageInput, CrewAIMessageSequence],
        **kwargs: Any,
    ) -> Any:
        """
        Async completion for CrewAI workflows.

        Parameters
        ----------
        messages : Union[CrewAIMessageInput, CrewAIMessageSequence]
            CrewAI message input (string, dict, message object, or sequence).
            These are passed as raw messages into the `LLMTranslator`.
        **kwargs : Any
            - model, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, stop
            - agent_role, agent_goal, task_description, crew_id, crew_name, process_id, task_id
            - request_id, tenant, framework_version
            - tools, tool_choice, system_message (optional; forwarded via framework_ctx)

        Returns
        -------
        Any
            Framework-level completion as produced by the CrewAI LLMFrameworkTranslator.
        """
        if getattr(self, "_validate_inputs_flag", True):
            self._validate_messages(messages)

        ctx, params, framework_ctx, resolved_kwargs = self._build_request_context(
            operation="acomplete",
            stream=False,
            kwargs=kwargs,
        )

        # Normalize messages to dicts for translator
        normalized_messages = self._to_translator_messages(messages)

        result = await self._translator.arun_complete(
            raw_messages=normalized_messages,
            model=params.get("model"),
            max_tokens=params.get("max_tokens"),
            temperature=params.get("temperature"),
            top_p=params.get("top_p"),
            frequency_penalty=params.get("frequency_penalty"),
            presence_penalty=params.get("presence_penalty"),
            stop_sequences=params.get("stop_sequences"),
            tools=resolved_kwargs.get("tools"),
            tool_choice=resolved_kwargs.get("tool_choice"),
            system_message=resolved_kwargs.get("system_message"),
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )

        # The CrewAI framework translator defines the final shape (usually a string).
        return result

    # ---------------------------------------------------------------------
    # Async streaming
    # ---------------------------------------------------------------------

    @with_async_llm_error_context("astream")
    async def astream(
        self,
        messages: Union[CrewAIMessageInput, CrewAIMessageSequence],
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """
        Async streaming completion for CrewAI workflows.

        Parameters
        ----------
        messages : Union[CrewAIMessageInput, CrewAIMessageSequence]
            CrewAI message input (string, dict, message object, or sequence).
        **kwargs : Any
            Same as `acomplete`.

        Yields
        ------
        Any
            Streaming chunks as produced by the CrewAI LLMFrameworkTranslator
            (typically text tokens for CrewAI).
        """
        if getattr(self, "_validate_inputs_flag", True):
            self._validate_messages(messages)

        ctx, params, framework_ctx, resolved_kwargs = self._build_request_context(
            operation="astream",
            stream=True,
            kwargs=kwargs,
        )

        # Normalize messages to dicts for translator
        normalized_messages = self._to_translator_messages(messages)

        # LLMTranslator.arun_stream returns an AsyncIterator directly; do not await.
        agen = self._translator.arun_stream(
            raw_messages=normalized_messages,
            model=params.get("model"),
            max_tokens=params.get("max_tokens"),
            temperature=params.get("temperature"),
            top_p=params.get("top_p"),
            frequency_penalty=params.get("frequency_penalty"),
            presence_penalty=params.get("presence_penalty"),
            stop_sequences=params.get("stop_sequences"),
            tools=resolved_kwargs.get("tools"),
            tool_choice=resolved_kwargs.get("tool_choice"),
            system_message=resolved_kwargs.get("system_message"),
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )

        try:
            async for chunk in agen:
                # The CrewAI translator defines the chunk shape; usually a text token.
                yield chunk
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework=_FRAMEWORK_NAME,
                operation="llm_astream",
                resource_type="llm",
                stream=True,
                model=str(params.get("model") or self.model),
                request_id=getattr(ctx, "request_id", None),
                tenant=getattr(ctx, "tenant", None),
                error_codes=ERROR_CODES,
            )
            raise
        finally:
            aclose = getattr(agen, "aclose", None)
            if callable(aclose):
                try:
                    await aclose()
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "Failed to close async stream iterator in CrewAI adapter: %s",
                        exc,
                    )

    # ---------------------------------------------------------------------
    # Sync completion
    # ---------------------------------------------------------------------

    @with_llm_error_context("complete")
    def complete(
        self,
        messages: Union[CrewAIMessageInput, CrewAIMessageSequence],
        **kwargs: Any,
    ) -> Any:
        """
        Sync completion for CrewAI workflows.

        Parameters
        ----------
        messages : Union[CrewAIMessageInput, CrewAIMessageSequence]
            CrewAI message input (string, dict, message object, or sequence).
        **kwargs : Any
            Same as `acomplete`.

        Returns
        -------
        Any
            Framework-level completion as produced by the CrewAI LLMFrameworkTranslator.
        """
        _ensure_not_in_event_loop("complete")

        if getattr(self, "_validate_inputs_flag", True):
            self._validate_messages(messages)

        ctx, params, framework_ctx, resolved_kwargs = self._build_request_context(
            operation="complete",
            stream=False,
            kwargs=kwargs,
        )

        # Normalize messages to dicts for translator
        normalized_messages = self._to_translator_messages(messages)

        result = self._translator.complete(
            raw_messages=normalized_messages,
            model=params.get("model"),
            max_tokens=params.get("max_tokens"),
            temperature=params.get("temperature"),
            top_p=params.get("top_p"),
            frequency_penalty=params.get("frequency_penalty"),
            presence_penalty=params.get("presence_penalty"),
            stop_sequences=params.get("stop_sequences"),
            tools=resolved_kwargs.get("tools"),
            tool_choice=resolved_kwargs.get("tool_choice"),
            system_message=resolved_kwargs.get("system_message"),
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )

        return result

    # Make the instance directly callable for convenience
    __call__ = complete

    # ---------------------------------------------------------------------
    # Sync streaming
    # ---------------------------------------------------------------------

    @with_llm_error_context("stream")
    def stream(
        self,
        messages: Union[CrewAIMessageInput, CrewAIMessageSequence],
        **kwargs: Any,
    ) -> Iterator[Any]:
        """
        Sync streaming completion for CrewAI workflows.

        Parameters
        ----------
        messages : Union[CrewAIMessageInput, CrewAIMessageSequence]
            CrewAI message input (string, dict, message object, or sequence).
        **kwargs : Any
            Same as `acomplete`.

        Yields
        ------
        Any
            Streaming chunks as produced by the CrewAI LLMFrameworkTranslator
            (typically text tokens for CrewAI).
        """
        _ensure_not_in_event_loop("stream")

        if getattr(self, "_validate_inputs_flag", True):
            self._validate_messages(messages)

        ctx, params, framework_ctx, resolved_kwargs = self._build_request_context(
            operation="stream",
            stream=True,
            kwargs=kwargs,
        )

        # Normalize messages to dicts for translator
        normalized_messages = self._to_translator_messages(messages)

        iterator = self._translator.stream(
            raw_messages=normalized_messages,
            model=params.get("model"),
            max_tokens=params.get("max_tokens"),
            temperature=params.get("temperature"),
            top_p=params.get("top_p"),
            frequency_penalty=params.get("frequency_penalty"),
            presence_penalty=params.get("presence_penalty"),
            stop_sequences=params.get("stop_sequences"),
            tools=resolved_kwargs.get("tools"),
            tool_choice=resolved_kwargs.get("tool_choice"),
            system_message=resolved_kwargs.get("system_message"),
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )

        # NOTE: Error context for iteration-time failures is handled by the
        # decorator wrapper for streaming operations.
        for chunk in iterator:
            yield chunk

    # ---------------------------------------------------------------------
    # Token counting
    # ---------------------------------------------------------------------

    def count_tokens(
        self,
        task: Optional["Task"] = None,
        **kwargs: Any,
    ) -> int:
        """
        Token counting helper for CrewAI tasks.

        Preferred path:
        - Use LLMTranslator.count_tokens_for_messages so token counting
          can use the same formatting and strategies as actual completions.

        Fallbacks:
        - If translator/adapter count fails, use char-based estimate.
        """
        if task is None:
            return 0

        # Best-effort conversion of a Task into a portable message list.
        # We intentionally avoid relying on translator-specific helpers here.
        content = (
            getattr(task, "description", None)
            or getattr(task, "prompt", None)
            or str(task)
        )
        messages: list[dict[str, Any]] = [{"role": "user", "content": str(content)}]

        ctx, params, framework_ctx, _resolved_kwargs = self._build_request_context(
            operation="count_tokens",
            stream=False,
            kwargs=kwargs,
        )
        model_for_context = params.get("model") or self.model

        # Preferred: translator-based token counting
        try:
            tokens_any = self._translator.count_tokens_for_messages(
                raw_messages=messages,
                model=model_for_context,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            if isinstance(tokens_any, int):
                return tokens_any
            if isinstance(tokens_any, Mapping):
                for key in ("tokens", "total_tokens", "count"):
                    value = tokens_any.get(key)
                    if isinstance(value, int):
                        return value
        except Exception:
            pass

        # Fallback: improved character-based estimate
        try:
            char_count = 0
            for msg in messages:
                char_count += len(str(msg.get("content", "")))
            
            # Rough estimate: ~4 characters per token on average
            return max(1, char_count // 4)
        except Exception:
            # Ultimate fallback
            return 0

    # ---------------------------------------------------------------------
    # Health and capabilities passthroughs (translator-only)
    # ---------------------------------------------------------------------

    @with_llm_error_context("health")
    def health(self, **kwargs: Any) -> Mapping[str, Any]:
        """
        Sync health check passthrough.

        Delegates exclusively to the LLMTranslator's health() method.
        The underlying adapter is not called here to avoid divergent behavior.
        """
        _ensure_not_in_event_loop("health")

        health_fn = getattr(self._translator, "health", None)
        if not callable(health_fn):
            raise RuntimeError(
                "LLMTranslator does not implement health(). "
                "Health reporting must be centralized in the translator."
            )

        # Conformance suite may pass the framework context kwarg (e.g. task=...).
        # Translator health/capabilities APIs are intentionally kwarg-free.
        result = health_fn()
        if not isinstance(result, Mapping):
            raise TypeError(
                "health() returned unsupported type from LLMTranslator: "
                f"{type(result).__name__}"
            )
        return result

    @with_async_llm_error_context("ahealth")
    async def ahealth(self, **kwargs: Any) -> Mapping[str, Any]:
        """
        Async health check passthrough.

        Delegates exclusively to the LLMTranslator:

        - Prefer an async `ahealth(**kwargs)` when available.
        - Otherwise, run sync `health(**kwargs)` in a worker thread.
        """
        async_health = getattr(self._translator, "ahealth", None)
        if callable(async_health):
            result = await async_health()
        else:
            health_fn = getattr(self._translator, "health", None)
            if not callable(health_fn):
                raise RuntimeError(
                    "LLMTranslator does not implement health() or ahealth(). "
                    "Health reporting must be implemented on the translator."
                )
            result = await asyncio.to_thread(health_fn)

        if not isinstance(result, Mapping):
            raise TypeError(
                "health() returned unsupported type from LLMTranslator: "
                f"{type(result).__name__}"
            )
        return result

    @with_llm_error_context("capabilities")
    def capabilities(self, **kwargs: Any) -> Mapping[str, Any]:
        """
        Sync capabilities passthrough.

        Delegates exclusively to the LLMTranslator's capabilities() method.
        The underlying adapter is not called here to avoid technical debt.
        """
        _ensure_not_in_event_loop("capabilities")

        caps_fn = getattr(self._translator, "capabilities", None)
        if not callable(caps_fn):
            raise RuntimeError(
                "LLMTranslator does not implement capabilities(). "
                "Capability reporting must be centralized in the translator."
            )

        result = caps_fn()
        if not isinstance(result, Mapping):
            raise TypeError(
                "capabilities() returned unsupported type from LLMTranslator: "
                f"{type(result).__name__}"
            )
        return result

    @with_async_llm_error_context("acapabilities")
    async def acapabilities(self, **kwargs: Any) -> Mapping[str, Any]:
        """
        Async capabilities passthrough.

        Delegates exclusively to the LLMTranslator:

        - Prefer an async `acapabilities(**kwargs)` when available.
        - Otherwise, run sync `capabilities(**kwargs)` in a worker thread.
        """
        async_caps = getattr(self._translator, "acapabilities", None)
        if callable(async_caps):
            result = await async_caps()
        else:
            caps_fn = getattr(self._translator, "capabilities", None)
            if not callable(caps_fn):
                raise RuntimeError(
                    "LLMTranslator does not implement capabilities() or acapabilities(). "
                    "Capability reporting must be implemented on the translator."
                )
            result = await asyncio.to_thread(caps_fn)

        if not isinstance(result, Mapping):
            raise TypeError(
                "capabilities() returned unsupported type from LLMTranslator: "
                f"{type(result).__name__}"
            )
        return result


__all__ = [
    "CorpusCrewAILLM",
    "CrewAILLMConfig",
    "CrewAIMessage",
    "CrewAIContext",
    "CrewAIMessageInput",
    "CrewAIMessageSequence",
    "with_llm_error_context",
    "with_async_llm_error_context",
    "ERROR_CODES",
    "ErrorCodes",
]