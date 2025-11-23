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
   - `OperationContext` built from CrewAI context via `ContextTranslator`
   - Rich error context attached via core `attach_context` with `operation="llm_*"`
"""

from __future__ import annotations

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

from corpus_sdk.core.context_translation import ContextTranslator
from corpus_sdk.core.error_context import attach_context
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

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Error-code bundle (shared-style CoercionErrorCodes for this framework)
# ---------------------------------------------------------------------------

ERROR_CODES = CoercionErrorCodes(
    invalid_result="CREWAI_LLM_INVALID_RESULT",
    empty_result="CREWAI_LLM_EMPTY_RESULT",
    conversion_error="CREWAI_LLM_CONVERSION_ERROR",
    framework_label="crewai",
)

# Symbolic code for init/config errors (for log/search friendliness)
INIT_CONFIG_ERROR = "CREWAI_LLM_BAD_INIT_CONFIG"

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
    """

    model: str = "default"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    framework_version: Optional[str] = None

    post_processing_config: Optional[LLMPostProcessingConfig] = None
    safety_filter: Optional[SafetyFilter] = None
    json_repair: Optional[JSONRepair] = None


# ---------------------------------------------------------------------------
# Error context helpers (aligned with other LLM adapters)
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

    # Sampling parameters
    sampling_params = [
        "temperature",
        "max_tokens",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "stop_sequences",
        "seed",
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

    This is called only in the error path to avoid hot-path overhead.
    """
    dynamic_ctx: Dict[str, Any] = {
        "model": getattr(instance, "model", "unknown"),
        "temperature": getattr(instance, "temperature", 0.7),
        "operation": operation,
        # NOTE: ERROR_CODES will be added once in the decorator; no need here.
    }

    # Message metrics
    if args:
        messages = _extract_messages_from_args(args[0])
        if messages:
            dynamic_ctx["messages_count"] = len(messages)
            roles, total_chars = _analyze_message_metrics(messages)
            dynamic_ctx["roles_distribution"] = roles
            dynamic_ctx["total_content_chars"] = total_chars

    # Context + sampling flags
    _extract_kwargs_context(kwargs, dynamic_ctx)

    # Streaming hint
    if operation in ("astream", "stream"):
        dynamic_ctx["stream"] = True

    return dynamic_ctx


def _create_error_context_decorator(
    operation: str,
    is_async: bool = False,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Factory for creating error-context decorators with rich per-call metrics.

    - Attaches `framework="crewai"` and `operation=f"llm_{operation}"`
    - Injects `ERROR_CODES` into the attached context
    """

    def decorator_factory(
        **static_context: Any,
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            if is_async:

                @wraps(func)
                async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
                    try:
                        return await func(self, *args, **kwargs)
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
                        attach_context(
                            exc,
                            framework="crewai",
                            operation=f"llm_{operation}",
                            **full_context,
                        )
                        raise

                return async_wrapper
            else:

                @wraps(func)
                def sync_wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
                    try:
                        return func(self, *args, **kwargs)
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
                        attach_context(
                            exc,
                            framework="crewai",
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
    return _create_error_context_decorator(operation, is_async=False)(
        **static_context
    )


def with_async_llm_error_context(
    operation: str,
    **static_context: Any,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for async LLM methods with rich dynamic context extraction."""
    return _create_error_context_decorator(operation, is_async=True)(
        **static_context
    )


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


def _build_operation_context_from_kwargs(kwargs: Mapping[str, Any]) -> OperationContext:
    """
    Build OperationContext from generic kwargs using core ContextTranslator.

    Defensive behavior:
    - If `ctx` is already an OperationContext, return it.
    - Else, call `ContextTranslator.from_crewai_context(...)`.
    - If that call fails, attach rich context and re-raise.
    - If it returns a non-OperationContext, attach context and raise TypeError.
    """
    # Caller-provided context wins
    existing_ctx = kwargs.get("ctx")
    if isinstance(existing_ctx, OperationContext):
        return existing_ctx

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

    try:
        ctx = ContextTranslator.from_crewai_context(
            task=task,
            framework_version=framework_version,
            **context_kwargs,
        )
    except Exception as exc:  # noqa: BLE001
        # Mirror other adapters: attach context on translation failure
        attach_context(
            exc,
            framework="crewai",
            operation="llm_context_translation",
            framework_version=framework_version,
            source="crewai_kwargs",
            **context_kwargs,
        )
        raise

    if not isinstance(ctx, OperationContext):
        exc = TypeError(
            "ContextTranslator.from_crewai_context produced unsupported "
            f"context type: {type(ctx).__name__}"
        )
        attach_context(
            exc,
            framework="crewai",
            operation="llm_context_translation",
            framework_version=framework_version,
            returned_type=type(ctx).__name__,
        )
        raise exc

    logger.debug(
        "Built OperationContext from CrewAI context with agent_role=%s, crew_id=%s",
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
    """
    Build sampling parameters from kwargs and defaults.

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
        float(clean_params.get("temperature", 0.7)),
        clean_params.get("max_tokens", "default"),
    )

    return clean_params


def _build_framework_ctx(
    framework_version: Optional[str],
    kwargs: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Build a framework_ctx payload for the CrewAI translator.

    This stays lightweight and purely informational; all translation logic
    lives in the registered LLMFrameworkTranslator for "crewai".
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
        "framework": "crewai",
        "framework_version": framework_version,
        "crewai_context": crewai_ctx,
    }

    # Optional knobs that the CrewAI framework translator may care about
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

    This class does not talk directly to provider APIs; instead, it:

    - Builds an `OperationContext` from CrewAI kwargs
    - Builds sampling params and a small `framework_ctx`
    - Delegates to `LLMTranslator` with `framework="crewai"`

    Framework-specific behavior (message normalization, system messages,
    OpenAI-shape handling, etc.) is handled by the registered
    `LLMFrameworkTranslator` for "crewai".
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
        config: Optional[CrewAILLMConfig] = None,
        translator: Optional[LLMFrameworkTranslator] = None,
        post_processing_config: Optional[LLMPostProcessingConfig] = None,
        safety_filter: Optional[SafetyFilter] = None,
        json_repair: Optional[JSONRepair] = None,
    ) -> None:
        """
        Initialize the CrewAI LLM adapter.
        """
        if require_crewai:
            _ensure_crewai_installed()

        # Resolve configuration precedence: explicit config > legacy kwargs.
        if config is not None:
            self.model = config.model
            self.temperature = float(config.temperature)
            self.max_tokens = config.max_tokens
            self._framework_version = config.framework_version or framework_version

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

            effective_post_processing = post_processing_config
            effective_safety_filter = safety_filter
            effective_json_repair = json_repair

        self._validate_init_params(llm_adapter)

        self._translator: LLMTranslator = create_llm_translator(
            adapter=llm_adapter,
            framework="crewai",
            translator=translator,
            post_processing_config=effective_post_processing,
            safety_filter=effective_safety_filter,
            json_repair=effective_json_repair,
        )

        self._llm_adapter = llm_adapter

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
                logger.warning("Error while closing LLM adapter: %s", exc)

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
                logger.warning("Error while async-closing LLM adapter: %s", exc)

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
                f"{INIT_CONFIG_ERROR}: "
                "llm_adapter must implement LLMProtocolV1; missing methods: "
                + ", ".join(missing)
            )

        if not isinstance(self.temperature, (int, float)) or not (
            0.0 <= float(self.temperature) <= 2.0
        ):
            raise ValueError(
                f"{INIT_CONFIG_ERROR}: temperature must be between 0.0 and 2.0"
            )

        if self.max_tokens is not None:
            if not isinstance(self.max_tokens, int) or self.max_tokens < 1:
                raise ValueError(
                    f"{INIT_CONFIG_ERROR}: max_tokens must be a positive integer"
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

    # ------------------------------------------------------------------ #
    # Async completion
    # ------------------------------------------------------------------ #

    @with_async_llm_error_context("acomplete")
    async def acomplete(
        self,
        messages: Union[CrewAIMessageInput, CrewAIMessageSequence],
        **kwargs: Any,
    ) -> Any:
        """
        Async completion for CrewAI workflows.
        """
        kwargs = self._apply_instance_defaults(kwargs)
        ctx = _build_operation_context_from_kwargs(kwargs)
        params = _build_sampling_params(
            default_model=self.model,
            default_temperature=self.temperature,
            default_max_tokens=self.max_tokens,
            kwargs=kwargs,
        )
        framework_ctx = _build_framework_ctx(
            kwargs.get("framework_version"),
            kwargs,
        )

        result = await self._translator.arun_complete(
            raw_messages=messages,
            model=params.get("model"),
            max_tokens=params.get("max_tokens"),
            temperature=params.get("temperature"),
            top_p=params.get("top_p"),
            frequency_penalty=params.get("frequency_penalty"),
            presence_penalty=params.get("presence_penalty"),
            stop_sequences=params.get("stop_sequences"),
            tools=kwargs.get("tools"),
            tool_choice=kwargs.get("tool_choice"),
            system_message=kwargs.get("system_message"),
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )

        return result

    # ------------------------------------------------------------------ #
    # Async streaming
    # ------------------------------------------------------------------ #

    @with_async_llm_error_context("astream")
    async def astream(
        self,
        messages: Union[CrewAIMessageInput, CrewAIMessageSequence],
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """
        Async streaming completion for CrewAI workflows.
        """
        kwargs = self._apply_instance_defaults(kwargs)
        ctx = _build_operation_context_from_kwargs(kwargs)
        params = _build_sampling_params(
            default_model=self.model,
            default_temperature=self.temperature,
            default_max_tokens=self.max_tokens,
            kwargs=kwargs,
        )
        framework_ctx = _build_framework_ctx(
            kwargs.get("framework_version"),
            kwargs,
        )

        agen = await self._translator.arun_stream(
            raw_messages=messages,
            model=params.get("model"),
            max_tokens=params.get("max_tokens"),
            temperature=params.get("temperature"),
            top_p=params.get("top_p"),
            frequency_penalty=params.get("frequency_penalty"),
            presence_penalty=params.get("presence_penalty"),
            stop_sequences=params.get("stop_sequences"),
            tools=kwargs.get("tools"),
            tool_choice=kwargs.get("tool_choice"),
            system_message=kwargs.get("system_message"),
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )

        async for chunk in agen:
            yield chunk

    # ------------------------------------------------------------------ #
    # Sync completion
    # ------------------------------------------------------------------ #

    @with_llm_error_context("complete")
    def complete(
        self,
        messages: Union[CrewAIMessageInput, CrewAIMessageSequence],
        **kwargs: Any,
    ) -> Any:
        """
        Sync completion for CrewAI workflows.
        """
        kwargs = self._apply_instance_defaults(kwargs)
        ctx = _build_operation_context_from_kwargs(kwargs)
        params = _build_sampling_params(
            default_model=self.model,
            default_temperature=self.temperature,
            default_max_tokens=self.max_tokens,
            kwargs=kwargs,
        )
        framework_ctx = _build_framework_ctx(
            kwargs.get("framework_version"),
            kwargs,
        )

        result = self._translator.complete(
            raw_messages=messages,
            model=params.get("model"),
            max_tokens=params.get("max_tokens"),
            temperature=params.get("temperature"),
            top_p=params.get("top_p"),
            frequency_penalty=params.get("frequency_penalty"),
            presence_penalty=params.get("presence_penalty"),
            stop_sequences=params.get("stop_sequences"),
            tools=kwargs.get("tools"),
            tool_choice=kwargs.get("tool_choice"),
            system_message=kwargs.get("system_message"),
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )

        return result

    # Direct call convenience: llm(...) behaves like llm.complete(...)
    __call__ = complete

    # ------------------------------------------------------------------ #
    # Sync streaming
    # ------------------------------------------------------------------ #

    @with_llm_error_context("stream")
    def stream(
        self,
        messages: Union[CrewAIMessageInput, CrewAIMessageSequence],
        **kwargs: Any,
    ) -> Iterator[Any]:
        """
        Sync streaming completion for CrewAI workflows.
        """
        kwargs = self._apply_instance_defaults(kwargs)
        ctx = _build_operation_context_from_kwargs(kwargs)
        params = _build_sampling_params(
            default_model=self.model,
            default_temperature=self.temperature,
            default_max_tokens=self.max_tokens,
            kwargs=kwargs,
        )
        framework_ctx = _build_framework_ctx(
            kwargs.get("framework_version"),
            kwargs,
        )

        iterator = self._translator.stream(
            raw_messages=messages,
            model=params.get("model"),
            max_tokens=params.get("max_tokens"),
            temperature=params.get("temperature"),
            top_p=params.get("top_p"),
            frequency_penalty=params.get("frequency_penalty"),
            presence_penalty=params.get("presence_penalty"),
            stop_sequences=params.get("stop_sequences"),
            tools=kwargs.get("tools"),
            tool_choice=kwargs.get("tool_choice"),
            system_message=kwargs.get("system_message"),
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )

        for chunk in iterator:
            yield chunk


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
]

