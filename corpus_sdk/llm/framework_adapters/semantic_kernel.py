# corpus_sdk/llm/framework_adapters/semantic_kernel.py
# SPDX-License-Identifier: Apache-2.0

"""
Semantic Kernel adapter for Corpus LLM protocol.

This module exposes a Corpus `LLMProtocolV1` behind the shared `LLMTranslator`
layer as a Semantic Kernel `ChatCompletionClientBase` implementation with:

- Async + streaming support
- Context propagation via `OperationContext`
- Rich error context for observability
- Framework-agnostic translation via `LLMTranslator` (no direct message translation)

Design goals
------------

1. Protocol + translator first:
   All calls go through the shared `LLMTranslator` with the `"semantic_kernel"`
   framework, so message normalization, post-processing, tools, and error
   context are consistent across frameworks.

2. Optional dependency safe:
   Import of `semantic_kernel` is guarded. Importing this module is safe even if
   Semantic Kernel is not installed.

3. Simple & explicit interface:
   Clean API that Semantic Kernel can use directly:

       sk_chat = CorpusSemanticKernelChatCompletion(
           llm_adapter=adapter,
           model="gpt-4",
           temperature=0.7,
       )
       response = await sk_chat.get_chat_message_content(chat_history, settings)

4. True streaming:
   Streaming goes through `LLMTranslator.arun_stream`, so you get
   protocol-level streaming semantics with Semantic Kernel friendly outputs.

5. Context + observability:
   - `OperationContext` built from Semantic Kernel settings
   - Rich error context attached via lazy, decorator-based helpers
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import wraps
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    TYPE_CHECKING,
    cast,
    Callable,
    Tuple,
    TypeVar,
)

from corpus_sdk.core.context_translation import (
    from_semantic_kernel as context_from_semantic_kernel,
)
from corpus_sdk.llm.framework_adapters.common.error_context import attach_context
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

# Framework identifier used consistently for translator + error context
_FRAMEWORK_NAME = "semantic_kernel"

# Symbolic init error prefix for easier log/search correlation
_INIT_ERROR_CODE = "SEMANTIC_KERNEL_LLM_BAD_INIT"

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Optional Semantic Kernel imports
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    from semantic_kernel.connectors.ai.chat_completion_client_base import (
        ChatCompletionClientBase,
        ChatHistory,
        PromptExecutionSettings,
    )
    from semantic_kernel.contents import (
        AuthorRole,
        ChatMessageContent,
        StreamingChatMessageContent,
    )
    _SEMANTIC_KERNEL_IMPORT_ERROR: Optional[Exception] = None
else:  # pragma: no cover - optional dependency path
    try:
        from semantic_kernel.connectors.ai.chat_completion_client_base import (
            ChatCompletionClientBase,
            ChatHistory,
            PromptExecutionSettings,
        )
        from semantic_kernel.contents import (
            AuthorRole,
            ChatMessageContent,
            StreamingChatMessageContent,
        )
    except Exception as _IMPORT_EXC:  # pragma: no cover - only hit when SK missing/mismatched
        ChatCompletionClientBase = object  # type: ignore[assignment]
        ChatHistory = Any  # type: ignore[assignment]
        PromptExecutionSettings = Any  # type: ignore[assignment]
        AuthorRole = Any  # type: ignore[assignment]
        ChatMessageContent = Any  # type: ignore[assignment]
        StreamingChatMessageContent = Any  # type: ignore[assignment]
        _SEMANTIC_KERNEL_IMPORT_ERROR = _IMPORT_EXC
    else:
        _SEMANTIC_KERNEL_IMPORT_ERROR = None


def _ensure_semantic_kernel_installed() -> None:
    """Raise a helpful error if Semantic Kernel is not installed."""
    if _SEMANTIC_KERNEL_IMPORT_ERROR is not None:
        raise RuntimeError(
            "semantic-kernel is not installed or failed to import. "
            "Install it via: pip install semantic-kernel"
        ) from _SEMANTIC_KERNEL_IMPORT_ERROR


class SemanticKernelLLMProtocol(Protocol):
    """
    Structural protocol for Semantic Kernel-compatible Corpus LLM.

    This protocol is kept for backward-compatibility and documentation
    purposes. The concrete implementation in this module delegates to
    `LLMTranslator` rather than implementing these methods directly on
    the underlying `LLMProtocolV1` adapter.
    """

    async def get_chat_message_content(
        self,
        chat_history: "ChatHistory",
        settings: "PromptExecutionSettings",
        **kwargs: Any,
    ) -> "ChatMessageContent":
        ...

    async def get_chat_message_contents(
        self,
        chat_history: "ChatHistory",
        settings: "PromptExecutionSettings",
        **kwargs: Any,
    ) -> List["ChatMessageContent"]:
        ...

    async def get_streaming_chat_message_content(
        self,
        chat_history: "ChatHistory",
        settings: "PromptExecutionSettings",
        **kwargs: Any,
    ) -> AsyncIterator["StreamingChatMessageContent"]:
        ...

    async def get_streaming_chat_message_contents(
        self,
        chat_history: "ChatHistory",
        settings: "PromptExecutionSettings",
        **kwargs: Any,
    ) -> AsyncIterator["StreamingChatMessageContent"]:
        ...


# ---------------------------------------------------------------------------
# Error-code bundle (CoercionErrorCodes alignment)
# ---------------------------------------------------------------------------

ERROR_CODES = CoercionErrorCodes(
    invalid_result="SEMANTIC_KERNEL_LLM_INVALID_RESULT",
    empty_result="SEMANTIC_KERNEL_LLM_EMPTY_RESULT",
    conversion_error="SEMANTIC_KERNEL_LLM_CONVERSION_ERROR",
    framework_label=_FRAMEWORK_NAME,
)


# ---------------------------------------------------------------------------
# Lightweight metrics for observability
# ---------------------------------------------------------------------------


def _analyze_chat_history(chat_history: Any) -> Dict[str, Any]:
    """
    Derive lightweight observability metrics from a Semantic Kernel ChatHistory.

    Returns a dict with:
    - roles_distribution: {role: count}
    - total_content_chars: int
    """
    roles_distribution: Dict[str, int] = {}
    total_chars = 0

    try:
        for msg in chat_history or []:
            role = getattr(msg, "role", None)
            if role is None and hasattr(msg, "author_role"):
                role = getattr(msg, "author_role", None)
            if role is None:
                role = "unknown"

            # SK message content can be plain text or richer content; we fall back to str().
            content = getattr(msg, "content", None)
            if content is None and hasattr(msg, "items"):
                # Some SK versions have multi-part content; collapse to string.
                try:
                    content = "".join(str(item) for item in msg.items)  # type: ignore[attr-defined]
                except Exception:
                    content = str(getattr(msg, "items", ""))

            if not isinstance(content, str):
                content = str(content) if content is not None else ""

            roles_distribution[role] = roles_distribution.get(role, 0) + 1
            total_chars += len(content)
    except Exception as metrics_exc:  # pragma: no cover - metrics must never be fatal
        logger.debug("Failed to analyze Semantic Kernel chat history: %s", metrics_exc)

    return {
        "roles_distribution": roles_distribution,
        "total_content_chars": total_chars,
    }


# ---------------------------------------------------------------------------
# Error context helpers (decorator-based, lazy)
# ---------------------------------------------------------------------------


def _extract_dynamic_context(
    instance: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    operation: str,
) -> Dict[str, Any]:
    """
    Extract rich dynamic context for error enrichment.

    This is called *only* on the error path to avoid overhead for successful calls.
    """
    dynamic_ctx: Dict[str, Any] = {
        "framework_name": _FRAMEWORK_NAME,
        "model": getattr(instance, "model", "unknown"),
        "temperature": getattr(instance, "temperature", 0.7),
        "operation": operation,
    }

    enable_metrics = getattr(instance, "_enable_metrics_flag", True)

    if enable_metrics:
        # Chat history metrics
        try:
            chat_history = None
            if args:
                chat_history = args[0]
            elif "chat_history" in kwargs:
                chat_history = kwargs["chat_history"]

            if chat_history is not None:
                try:
                    dynamic_ctx["messages_count"] = len(chat_history)  # type: ignore[arg-type]
                except Exception:
                    pass

                metrics = _analyze_chat_history(chat_history)
                dynamic_ctx.update(metrics)
        except Exception as metrics_exc:  # pragma: no cover
            logger.debug(
                "Failed to compute Semantic Kernel message metrics for error context: %s",
                metrics_exc,
            )

        # Try to include resolved sampling params + context identifiers
        # by reusing the instance helpers on the error path.
        try:
            settings: Optional[Any] = None
            if len(args) >= 2:
                settings = args[1]
            elif "settings" in kwargs:
                settings = kwargs["settings"]

            if settings is not None:
                stream = operation in (
                    "get_streaming_chat_message_content",
                    "get_streaming_chat_message_contents",
                )
                ctx, params, model_for_context, _ = instance._build_request_context(  # type: ignore[attr-defined]
                    settings,
                    kwargs,
                    operation=operation,
                    stream=stream,
                )

                dynamic_ctx["resolved_model"] = model_for_context
                dynamic_ctx["request_id"] = getattr(ctx, "request_id", None)
                dynamic_ctx["tenant"] = getattr(ctx, "tenant", None)

                for key in (
                    "model",
                    "temperature",
                    "max_tokens",
                    "top_p",
                    "frequency_penalty",
                    "presence_penalty",
                    "stop_sequences",
                ):
                    if key in params and params[key] is not None:
                        dynamic_ctx[f"resolved_{key}"] = params[key]
        except Exception as ctx_exc:  # pragma: no cover
            logger.debug(
                "Failed to compute Semantic Kernel context params for error context: %s",
                ctx_exc,
            )

    return dynamic_ctx


def _create_error_context_decorator(
    operation: str,
    is_async: bool = False,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Factory for creating error-context decorators with lazy dynamic context.

    Successful calls are unaffected; on exception we compute metrics and
    attach them via `attach_context` with a consistent LLM-oriented operation.
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
                        dynamic_ctx = _extract_dynamic_context(
                            self,
                            args,
                            kwargs,
                            operation,
                        )
                        full_ctx = {
                            "error_codes": ERROR_CODES,
                            **static_context,
                            **dynamic_ctx,
                        }
                        attach_context(
                            exc,
                            framework=_FRAMEWORK_NAME,
                            operation=f"llm_{operation}",
                            **full_ctx,
                        )
                        raise

                return async_wrapper
            else:

                @wraps(func)
                def sync_wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
                    try:
                        return func(self, *args, **kwargs)
                    except Exception as exc:  # noqa: BLE001
                        dynamic_ctx = _extract_dynamic_context(
                            self,
                            args,
                            kwargs,
                            operation,
                        )
                        full_ctx = {
                            "error_codes": ERROR_CODES,
                            **static_context,
                            **dynamic_ctx,
                        }
                        attach_context(
                            exc,
                            framework=_FRAMEWORK_NAME,
                            operation=f"llm_{operation}",
                            **full_ctx,
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
# Config for Semantic Kernel wrapper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SemanticKernelChatConfig:
    """
    Configuration for `CorpusSemanticKernelChatCompletion`.

    This is a thin wrapper around the existing `LLMTranslator` knobs:
    - Default model / temperature / max_tokens
    - Optional post-processing configuration
    - Optional safety / JSON repair behavior
    - Optional framework version tagging
    - Optional metrics + input validation toggles
    """

    model: str = "default"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    framework_version: Optional[str] = None

    post_processing_config: Optional[LLMPostProcessingConfig] = None
    safety_filter: Optional[SafetyFilter] = None
    json_repair: Optional[JSONRepair] = None

    # Behavior toggles (aligned with other adapters)
    enable_metrics: bool = True
    validate_inputs: bool = True

    def __post_init__(self) -> None:
        """
        Validate configuration values eagerly so misconfigurations fail fast.
        """
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                f"temperature must be between 0.0 and 2.0, got {self.temperature}"
            )
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError(
                f"max_tokens must be positive, got {self.max_tokens}"
            )


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------


class CorpusSemanticKernelChatCompletion(ChatCompletionClientBase):
    """
    Semantic Kernel `ChatCompletionClientBase` backed by a Corpus LLM via
    the shared `LLMTranslator` with `framework="semantic_kernel"`.

    This class does not perform its own message translation or talk directly
    to provider APIs; instead, it:

    - Builds an `OperationContext` from Semantic Kernel settings
    - Builds sampling params and a small `framework_ctx`
    - Delegates to `LLMTranslator` with `framework="semantic_kernel"`

    Framework-specific behavior (message normalization, OpenAI shape handling,
    etc.) is handled by the registered Semantic Kernel `LLMFrameworkTranslator`.
    """

    llm_adapter: LLMProtocolV1
    model: str = "default"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    framework_version: Optional[str] = None

    def __init__(
        self,
        llm_adapter: LLMProtocolV1,
        *,
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        framework_version: Optional[str] = None,
        service_id: Optional[str] = None,
        # Optional configuration wrapper
        config: Optional[SemanticKernelChatConfig] = None,
        # Optional explicit framework translator wiring
        translator: Optional[LLMFrameworkTranslator] = None,
        post_processing_config: Optional[LLMPostProcessingConfig] = None,
        safety_filter: Optional[SafetyFilter] = None,
        json_repair: Optional[JSONRepair] = None,
    ) -> None:
        _ensure_semantic_kernel_installed()

        # Resolve configuration precedence: config object > legacy kwargs.
        if config is not None:
            self.model = config.model
            self.temperature = float(config.temperature)
            self.max_tokens = config.max_tokens
            self.framework_version = config.framework_version or framework_version

            effective_post_processing = (
                post_processing_config or config.post_processing_config
            )
            effective_safety_filter = safety_filter or config.safety_filter
            effective_json_repair = json_repair or config.json_repair

            self._enable_metrics_flag = config.enable_metrics
            self._validate_inputs_flag = config.validate_inputs
        else:
            self.model = model
            self.temperature = float(temperature)
            self.max_tokens = max_tokens
            self.framework_version = framework_version

            effective_post_processing = post_processing_config
            effective_safety_filter = safety_filter
            effective_json_repair = json_repair

            # Defaults aligned with other adapters
            self._enable_metrics_flag = True
            self._validate_inputs_flag = True

        # Validate adapter + sampling defaults in a shared, symbolic way.
        self._validate_init_params(llm_adapter)

        super().__init__(service_id=service_id)

        # Build the shared LLMTranslator for the "semantic_kernel" framework.
        self._translator: LLMTranslator = create_llm_translator(
            adapter=llm_adapter,
            framework=_FRAMEWORK_NAME,
            translator=translator,
            post_processing_config=effective_post_processing,
            safety_filter=effective_safety_filter,
            json_repair=effective_json_repair,
        )

        self.llm_adapter = llm_adapter  # kept for introspection/metrics if needed

        logger.info(
            "CorpusSemanticKernelChatCompletion initialized: "
            "model=%s, temperature=%.2f, max_tokens=%s, service_id=%s, framework_version=%s",
            self.model,
            self.temperature,
            self.max_tokens if self.max_tokens is not None else "default",
            service_id or "default",
            self.framework_version or "unknown",
        )

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
                f"{_INIT_ERROR_CODE}: "
                "llm_adapter must implement LLMProtocolV1; missing methods: "
                + ", ".join(missing)
            )

        if not isinstance(self.temperature, (int, float)) or not (
            0.0 <= float(self.temperature) <= 2.0
        ):
            raise ValueError(
                f"{_INIT_ERROR_CODE}: temperature must be between 0.0 and 2.0"
            )

        if self.max_tokens is not None:
            if not isinstance(self.max_tokens, int) or self.max_tokens < 1:
                raise ValueError(
                    f"{_INIT_ERROR_CODE}: max_tokens must be a positive integer"
                )

    def _validate_chat_history(self, chat_history: "ChatHistory") -> None:
        """
        Validate Semantic Kernel chat history structure before handing it
        to the translator.

        Ensures:
        - chat_history is non-empty
        - each message has a role/author_role
        - each message exposes content or items
        """
        if not chat_history:
            raise ValueError("Chat history cannot be empty")

        for idx, msg in enumerate(chat_history):
            if not (hasattr(msg, "role") or hasattr(msg, "author_role")):
                raise TypeError(
                    f"chat_history[{idx}] is missing 'role' or 'author_role'"
                )
            if not (hasattr(msg, "content") or hasattr(msg, "items")):
                raise TypeError(
                    f"chat_history[{idx}] is missing 'content' or 'items'"
                )

    # ------------------------------------------------------------------ #
    # Internal helpers: context, sampling, framework_ctx, request_context
    # ------------------------------------------------------------------ #

    def _build_operation_context(
        self,
        settings: "PromptExecutionSettings",
    ) -> OperationContext:
        """
        Build OperationContext from Semantic Kernel prompt settings.

        Delegates to `context_from_semantic_kernel` with defensive error
        context attachment.
        """
        try:
            return context_from_semantic_kernel(
                settings=settings,
                framework_version=self.framework_version,
            )
        except Exception as ctx_exc:
            attach_context(
                ctx_exc,
                framework=_FRAMEWORK_NAME,
                operation="llm_context_translation",
                error_codes=ERROR_CODES,
                framework_version=self.framework_version,
                settings_type=type(settings).__name__,
            )
            raise

    def _build_sampling_params(
        self,
        settings: "PromptExecutionSettings",
        kwargs: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Build sampling parameters from Semantic Kernel PromptExecutionSettings
        and explicit kwargs, with instance defaults as final fallback.

        Returns a dict suitable for passing into `LLMTranslator`.
        """
        # Model resolution precedence:
        # 1. Explicit model kwarg
        # 2. settings.model_id / settings.model / settings.deployment_name
        # 3. Instance default
        model = (
            kwargs.get("model")
            or getattr(settings, "model_id", None)
            or getattr(settings, "model", None)
            or getattr(settings, "deployment_name", None)
            or self.model
        )

        temperature = getattr(settings, "temperature", None)
        if temperature is None:
            temperature = self.temperature

        max_tokens = getattr(settings, "max_tokens", None)
        if max_tokens is None:
            max_tokens = self.max_tokens

        raw_stop = (
            getattr(settings, "stop_sequences", None)
            or getattr(settings, "stop", None)
        )

        stop_sequences: Optional[List[str]]
        if raw_stop is None:
            stop_sequences = None
        elif isinstance(raw_stop, str):
            stop_sequences = [raw_stop]
        elif isinstance(raw_stop, (list, tuple)):
            stop_sequences = [str(s) for s in raw_stop]
        else:
            stop_sequences = [str(raw_stop)]

        params: Dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": getattr(settings, "top_p", None),
            "frequency_penalty": getattr(settings, "frequency_penalty", None),
            "presence_penalty": getattr(settings, "presence_penalty", None),
            "stop_sequences": stop_sequences,
        }

        # Strip None values so the translator sees a clean param set.
        clean_params = {k: v for k, v in params.items() if v is not None}

        logger.debug(
            "Semantic Kernel sampling params resolved: model=%s, temperature=%.3f, "
            "max_tokens=%s, top_p=%s, stop_sequences=%s",
            clean_params.get("model", self.model),
            clean_params.get("temperature", self.temperature),
            clean_params.get("max_tokens", "default"),
            clean_params.get("top_p", "default"),
            clean_params.get("stop_sequences", "default"),
        )

        return clean_params

    def _build_framework_ctx(
        self,
        settings: "PromptExecutionSettings",
        kwargs: Mapping[str, Any],
        *,
        operation: str,
        stream: bool,
    ) -> Dict[str, Any]:
        """
        Build a lightweight framework_ctx payload for the Semantic Kernel translator.

        This stays informational; all translation logic lives in the registered
        `LLMFrameworkTranslator` for `"semantic_kernel"`.
        """
        framework_ctx: Dict[str, Any] = {
            "framework": _FRAMEWORK_NAME,
            "framework_version": self.framework_version,
            "service_id": getattr(self, "service_id", None),
            "settings_type": type(settings).__name__,
            "operation": operation,
            "stream": stream,
        }

        # Include any explicitly passed tools / tool_choice / system_message
        # so the Semantic Kernel translator can see them if it wants.
        for key in ("tools", "tool_choice", "system_message"):
            if key in kwargs:
                framework_ctx[key] = kwargs[key]

        return framework_ctx

    def _build_request_context(
        self,
        settings: "PromptExecutionSettings",
        kwargs: Mapping[str, Any],
        *,
        operation: str,
        stream: bool,
    ) -> Tuple[OperationContext, Dict[str, Any], str, Dict[str, Any]]:
        """
        Build and bundle OperationContext, sampling params, model_for_context,
        and framework_ctx for a single SK request.
        """
        ctx = self._build_operation_context(settings)
        params = self._build_sampling_params(settings, kwargs)
        model_for_context = params.get("model", self.model)
        framework_ctx = self._build_framework_ctx(
            settings,
            kwargs,
            operation=operation,
            stream=stream,
        )
        return ctx, params, model_for_context, framework_ctx

    # ------------------------------------------------------------------ #
    # Semantic Kernel async API (via LLMTranslator)
    # ------------------------------------------------------------------ #

    @with_async_llm_error_context("get_chat_message_content")
    async def get_chat_message_content(
        self,
        chat_history: "ChatHistory",
        settings: "PromptExecutionSettings",
        **kwargs: Any,
    ) -> "ChatMessageContent":
        """
        Execute a single chat completion call via LLMTranslator.

        The Semantic Kernel framework translator is responsible for turning
        the raw messages + completion into `ChatMessageContent`.
        """
        if getattr(self, "_validate_inputs_flag", True):
            self._validate_chat_history(chat_history)

        ctx, params, model_for_context, framework_ctx = self._build_request_context(
            settings,
            kwargs,
            operation="get_chat_message_content",
            stream=False,
        )

        result = await self._translator.arun_complete(
            raw_messages=chat_history,
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
        return cast("ChatMessageContent", result)

    @with_async_llm_error_context("get_chat_message_contents")
    async def get_chat_message_contents(
        self,
        chat_history: "ChatHistory",
        settings: "PromptExecutionSettings",
        **kwargs: Any,
    ) -> List["ChatMessageContent"]:
        """
        Convenience wrapper returning a single-element list for SK APIs
        that expect multiple choices.
        """
        message = await self.get_chat_message_content(chat_history, settings, **kwargs)
        return [message]

    @with_async_llm_error_context("get_streaming_chat_message_content")
    async def get_streaming_chat_message_content(
        self,
        chat_history: "ChatHistory",
        settings: "PromptExecutionSettings",
        **kwargs: Any,
    ) -> AsyncIterator["StreamingChatMessageContent"]:
        """
        Streaming chat completion via LLMTranslator.

        Yields incremental StreamingChatMessageContent chunks compatible
        with Semantic Kernel's streaming APIs. The SK framework translator
        defines the chunk shape.
        """
        if getattr(self, "_validate_inputs_flag", True):
            self._validate_chat_history(chat_history)

        ctx, params, model_for_context, framework_ctx = self._build_request_context(
            settings,
            kwargs,
            operation="get_streaming_chat_message_content",
            stream=True,
        )

        agen: Optional[AsyncIterator[Any]] = None
        try:
            agen = await self._translator.arun_stream(
                raw_messages=chat_history,
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
                # The Semantic Kernel translator defines the chunk shape;
                # expected: StreamingChatMessageContent.
                yield cast("StreamingChatMessageContent", chunk)
        finally:
            # Best-effort cleanup of async generator resources.
            if agen is not None:
                close_method = getattr(agen, "aclose", None)
                if close_method and callable(close_method):
                    try:
                        await close_method()
                    except Exception:
                        logger.debug(
                            "Failed to close Semantic Kernel streaming generator",
                            exc_info=True,
                        )

    @with_async_llm_error_context("get_streaming_chat_message_contents")
    async def get_streaming_chat_message_contents(
        self,
        chat_history: "ChatHistory",
        settings: "PromptExecutionSettings",
        **kwargs: Any,
    ) -> AsyncIterator["StreamingChatMessageContent"]:
        """
        Alias for get_streaming_chat_message_content that satisfies the
        Semantic Kernel multi-message streaming interface.
        """
        async for msg in self.get_streaming_chat_message_content(
            chat_history,
            settings,
            **kwargs,
        ):
            yield msg

    # ------------------------------------------------------------------ #
    # Token counting (translator + heuristic fallback, for parity)
    # ------------------------------------------------------------------ #

    def _combine_chat_history_for_counting(
        self,
        chat_history: "ChatHistory",
    ) -> str:
        """
        Combine Semantic Kernel chat history into a single string for
        heuristic token counting.
        """
        parts: List[str] = []
        for msg in chat_history or []:
            role = getattr(msg, "role", None)
            if role is None and hasattr(msg, "author_role"):
                role = getattr(msg, "author_role", None)
            if role is None:
                role = "user"

            content = getattr(msg, "content", None)
            if content is None and hasattr(msg, "items"):
                try:
                    content = "".join(str(item) for item in msg.items)  # type: ignore[attr-defined]
                except Exception:
                    content = str(getattr(msg, "items", ""))

            if not isinstance(content, str):
                content = str(content) if content is not None else ""

            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def count_tokens(
        self,
        chat_history: "ChatHistory",
        settings: "PromptExecutionSettings",
        **kwargs: Any,
    ) -> int:
        """
        Token counting helper for Semantic Kernel chat history.

        Preferred path:
        - Use LLMTranslator.count_tokens_for_messages so token counting
          can use the same formatting and strategies as actual completions.

        Fallbacks:
        - If translator/adapter count fails, use char-based estimate.
        """
        if not chat_history:
            return 0

        ctx, params, model_for_context, framework_ctx = self._build_request_context(
            settings,
            kwargs,
            operation="count_tokens",
            stream=False,
        )

        # Preferred: translator-based token counting
        try:
            tokens_any = self._translator.count_tokens_for_messages(
                raw_messages=chat_history,
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
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Semantic Kernel translator-based token counting failed: %s",
                exc,
            )

        # Fallback: simple character-based heuristic
        combined = self._combine_chat_history_for_counting(chat_history)
        if not combined:
            return 0
        return max(1, len(combined) // 4)


__all__ = [
    "SemanticKernelLLMProtocol",
    "SemanticKernelChatConfig",
    "CorpusSemanticKernelChatCompletion",
    "with_llm_error_context",
    "with_async_llm_error_context",
    "ERROR_CODES",
]
