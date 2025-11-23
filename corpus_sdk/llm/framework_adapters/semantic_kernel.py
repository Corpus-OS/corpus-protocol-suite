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
   - Rich error context attached via the framework-level error_context helpers
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
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

logger = logging.getLogger(__name__)

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
# Config for Semantic Kernel wrapper
# ---------------------------------------------------------------------------


@dataclass
class SemanticKernelChatConfig:
    """
    Configuration for `CorpusSemanticKernelChatCompletion`.

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

    This class no longer performs its own message translation or talks
    directly to `LLMProtocolV1`; instead, it:

    - Builds an `OperationContext` from Semantic Kernel settings
    - Builds sampling params and a small `framework_ctx`
    - Delegates to `LLMTranslator` with `framework="semantic_kernel"`

    The framework-specific behavior (message normalization, OpenAI shape
    conversion, etc.) is handled by the registered Semantic Kernel
    `LLMFrameworkTranslator` or the default translator.
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
        if _SEMANTIC_KERNEL_IMPORT_ERROR is not None:
            raise RuntimeError(
                "semantic-kernel is not installed or failed to import. "
                "Install it via: pip install semantic-kernel"
            ) from _SEMANTIC_KERNEL_IMPORT_ERROR

        # Harden adapter validation: we rely on async protocol surface.
        if not hasattr(llm_adapter, "acomplete") or not callable(
            getattr(llm_adapter, "acomplete", None)
        ):
            raise TypeError(
                "llm_adapter must implement LLMProtocolV1 with an async 'acomplete' method"
            )
        if not hasattr(llm_adapter, "astream") or not callable(
            getattr(llm_adapter, "astream", None)
        ):
            raise TypeError(
                "llm_adapter must implement LLMProtocolV1 with an async 'astream' method"
            )

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
        else:
            self.model = model
            self.temperature = float(temperature)
            self.max_tokens = max_tokens
            self.framework_version = framework_version

            effective_post_processing = post_processing_config
            effective_safety_filter = safety_filter
            effective_json_repair = json_repair

        # Single validation point for sampling defaults.
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                f"temperature must be between 0.0 and 2.0, got {self.temperature}"
            )

        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError(
                f"max_tokens must be positive, got {self.max_tokens}"
            )

        super().__init__(service_id=service_id)

        # Build the shared LLMTranslator for the "semantic_kernel" framework.
        self._translator: LLMTranslator = create_llm_translator(
            adapter=llm_adapter,
            framework="semantic_kernel",
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
    # Error context management
    # ------------------------------------------------------------------ #

    def _build_error_context(
        self,
        operation: str,
        stream: bool,
        messages_count: int,
        model: str,
        ctx: OperationContext,
        params: Dict[str, Any],
        extra_metrics: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build consistent error context payload.

        Includes:
        - resource_type, operation, stream
        - routing + sampling parameters
        - request identifiers
        - lightweight message metrics (roles, content size) when available
        """
        error_ctx: Dict[str, Any] = {
            "resource_type": "llm",
            "operation": operation,
            "messages_count": messages_count,
            "model": model,
            "temperature": params.get("temperature"),
            "max_tokens": params.get("max_tokens"),
            "top_p": params.get("top_p"),
            "frequency_penalty": params.get("frequency_penalty"),
            "presence_penalty": params.get("presence_penalty"),
            "stop_sequences": params.get("stop_sequences"),
            "request_id": ctx.request_id,
            "tenant": ctx.tenant,
            "stream": stream,
        }

        if extra_metrics:
            error_ctx.update(extra_metrics)

        return error_ctx

    @asynccontextmanager
    async def _error_context_async(
        self,
        operation: str,
        *,
        stream: bool,
        messages_count: int,
        model: str,
        ctx: OperationContext,
        params: Dict[str, Any],
        extra_metrics: Optional[Mapping[str, Any]] = None,
    ):
        """
        Async error-context wrapper.

        Ensures any exception raised within is enriched with structured,
        framework-specific context, without mutating the original type.
        """
        error_ctx = self._build_error_context(
            operation=operation,
            stream=stream,
            messages_count=messages_count,
            model=model,
            ctx=ctx,
            params=params,
            extra_metrics=extra_metrics,
        )
        try:
            yield
        except BaseException as exc:  # noqa: BLE001
            try:
                # `framework` is passed as the required origin identifier,
                # and error_ctx carries the rest of the metadata.
                attach_context(exc, framework="semantic_kernel", **error_ctx)
            except Exception:  # pragma: no cover - never mask original error
                logger.debug(
                    "Failed to attach Semantic Kernel error context",
                    exc_info=True,
                )
            raise

    # ------------------------------------------------------------------ #
    # Internal helpers: context, sampling, framework_ctx
    # ------------------------------------------------------------------ #

    def _build_operation_context(
        self,
        settings: "PromptExecutionSettings",
    ) -> OperationContext:
        """
        Build OperationContext from Semantic Kernel prompt settings.

        Delegates to `context_from_semantic_kernel` with defensive logging.
        """
        try:
            return context_from_semantic_kernel(
                settings=settings,
                framework_version=self.framework_version,
            )
        except Exception as ctx_exc:
            logger.warning(
                "Failed to build OperationContext from Semantic Kernel settings: %s",
                ctx_exc,
            )
            # Let the context translator control default semantics; if it
            # fails, we re-raise, as a partially-formed context is worse.
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

        stop_sequences = (
            getattr(settings, "stop_sequences", None)
            or getattr(settings, "stop", None)
        )

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
    ) -> Dict[str, Any]:
        """
        Build a lightweight framework_ctx payload for the Semantic Kernel translator.

        This stays informational; all translation logic lives in the registered
        `LLMFrameworkTranslator` for `"semantic_kernel"`.
        """
        framework_ctx: Dict[str, Any] = {
            "framework": "semantic_kernel",
            "framework_version": self.framework_version,
            "service_id": getattr(self, "service_id", None),
            "settings_type": type(settings).__name__,
        }

        # Include any explicitly passed tools / tool_choice / system_message
        # so the Semantic Kernel translator can see them if it wants.
        for key in ("tools", "tool_choice", "system_message"):
            if key in kwargs:
                framework_ctx[key] = kwargs[key]

        return framework_ctx

    # ------------------------------------------------------------------ #
    # Semantic Kernel async API (via LLMTranslator)
    # ------------------------------------------------------------------ #

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
        if not chat_history:
            raise ValueError("Chat history cannot be empty")

        ctx = self._build_operation_context(settings)
        params = self._build_sampling_params(settings, kwargs)
        framework_ctx = self._build_framework_ctx(settings, kwargs)

        model_for_context = params.get("model", self.model)
        messages_count = len(chat_history)
        metrics = _analyze_chat_history(chat_history)

        async with self._error_context_async(
            "get_chat_message_content",
            stream=False,
            messages_count=messages_count,
            model=model_for_context,
            ctx=ctx,
            params=params,
            extra_metrics=metrics,
        ):
            # Delegate to LLMTranslator; the SK framework translator defines
            # the exact return shape (expected: ChatMessageContent).
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
        if not chat_history:
            raise ValueError("Chat history cannot be empty")

        ctx = self._build_operation_context(settings)
        params = self._build_sampling_params(settings, kwargs)
        framework_ctx = self._build_framework_ctx(settings, kwargs)

        model_for_context = params.get("model", self.model)
        messages_count = len(chat_history)
        metrics = _analyze_chat_history(chat_history)

        async with self._error_context_async(
            "get_streaming_chat_message_content",
            stream=True,
            messages_count=messages_count,
            model=model_for_context,
            ctx=ctx,
            params=params,
            extra_metrics=metrics,
        ):
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


__all__ = [
    "SemanticKernelLLMProtocol",
    "SemanticKernelChatConfig",
    "CorpusSemanticKernelChatCompletion",
]
