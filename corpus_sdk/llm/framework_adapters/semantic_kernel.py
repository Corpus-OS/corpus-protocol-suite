# corpus_sdk/llm/framework_adapters/semantic_kernel.py
# SPDX-License-Identifier: Apache-2.0

"""
Semantic Kernel adapter for Corpus LLM protocol.

This module exposes a Corpus `LLMProtocolV1` as a Semantic Kernel
`ChatCompletionClientBase` implementation with:

- Direct protocol access (no unnecessary abstraction layers)
- Async + streaming support
- Context propagation via `OperationContext`
- Rich error context for observability
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    TYPE_CHECKING,
)

from corpus_sdk.core.context_translation import (
    from_semantic_kernel as context_from_semantic_kernel,
)
from corpus_sdk.core.error_context import attach_context
from corpus_sdk.llm.framework_adapters.common.message_translation import (
    from_semantic_kernel,
    to_corpus,
    NormalizedMessage,
)
from corpus_sdk.llm.llm_base import (
    LLMChunk,
    LLMCompletion,
    LLMProtocolV1,
    OperationContext,
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
    """Structural protocol for Semantic Kernel-compatible Corpus LLM."""

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


class CorpusSemanticKernelChatCompletion(ChatCompletionClientBase):
    """
    Semantic Kernel `ChatCompletionClientBase` backed by a Corpus LLM protocol.

    Usage:
        sk_chat = CorpusSemanticKernelChatCompletion(
            llm_adapter=adapter,
            model="gpt-4",
            temperature=0.7,
        )
        response = await sk_chat.get_chat_message_content(chat_history, settings)
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
    ) -> None:
        if _SEMANTIC_KERNEL_IMPORT_ERROR is not None:
            raise RuntimeError(
                "semantic-kernel is not installed or failed to import. "
                "Install it via: pip install semantic-kernel"
            ) from _SEMANTIC_KERNEL_IMPORT_ERROR

        # Harden adapter validation: require the async protocol surface we use.
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

        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")

        if max_tokens is not None and max_tokens < 1:
            raise ValueError("max_tokens must be positive")

        super().__init__(service_id=service_id)

        self.llm_adapter = llm_adapter
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = max_tokens
        self.framework_version = framework_version

        logger.info(
            "CorpusSemanticKernelChatCompletion initialized: "
            "model=%s, temperature=%.2f, max_tokens=%s, service_id=%s",
            self.model,
            self.temperature,
            self.max_tokens if self.max_tokens is not None else "default",
            service_id or "default",
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
        Build consistent error context for attach_context.

        Includes:
        - framework, operation, stream
        - routing + sampling parameters
        - request identifiers
        - lightweight message metrics (roles, content size) when available
        """
        error_ctx: Dict[str, Any] = {
            "framework": "semantic_kernel",
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
                attach_context(exc, **error_ctx)
            except Exception:  # pragma: no cover - never mask original error
                logger.debug("Failed to attach Semantic Kernel error context", exc_info=True)
            raise

    # ------------------------------------------------------------------ #
    # Internal translation + parameter helpers
    # ------------------------------------------------------------------ #

    def _translate_messages(self, chat_history: "ChatHistory") -> List[Dict[str, Any]]:
        """
        Semantic Kernel ChatHistory → Corpus wire format ({role, content} dicts).
        """
        if not chat_history:
            logger.warning("Empty chat history provided to Semantic Kernel adapter")
            return []

        try:
            messages: List[Dict[str, Any]] = []
            for msg in chat_history:
                role = getattr(msg, "role", None)
                if role is None and hasattr(msg, "author_role"):
                    role = getattr(msg, "author_role", None)

                content = getattr(msg, "content", None)
                metadata = getattr(msg, "metadata", {}) or {}

                sk_msg = {
                    "role": role,
                    "content": content,
                    "metadata": metadata,
                }
                messages.append(sk_msg)

            normalized: List[NormalizedMessage] = [from_semantic_kernel(m) for m in messages]
            return [dict(m) for m in to_corpus(normalized)]
        except Exception as e:
            logger.error("Failed to translate Semantic Kernel messages: %s", e)
            raise ValueError(f"Semantic Kernel message translation failed: {e}") from e

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

        # Strip None values so the protocol sees a clean param set.
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

    def _completion_to_chat_message(
        self,
        completion: LLMCompletion,
    ) -> "ChatMessageContent":
        """
        LLMCompletion → Semantic Kernel ChatMessageContent.
        """
        metadata: Dict[str, Any] = {
            "model": completion.model,
            "finish_reason": completion.finish_reason,
        }

        if completion.usage is not None:
            metadata["usage"] = {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
            }

        return ChatMessageContent(
            role=AuthorRole.ASSISTANT,
            content=completion.text,
            metadata=metadata,
        )

    def _chunk_to_streaming_chat_message(
        self,
        chunk: LLMChunk,
    ) -> "StreamingChatMessageContent":
        """
        LLMChunk → Semantic Kernel StreamingChatMessageContent.
        """
        metadata: Dict[str, Any] = {
            "is_final": chunk.is_final,
        }

        if chunk.model is not None:
            metadata["model"] = chunk.model

        if chunk.usage_so_far is not None:
            metadata["usage_so_far"] = {
                "prompt_tokens": chunk.usage_so_far.prompt_tokens,
                "completion_tokens": chunk.usage_so_far.completion_tokens,
                "total_tokens": chunk.usage_so_far.total_tokens,
            }

        return StreamingChatMessageContent(
            role=AuthorRole.ASSISTANT,
            content=chunk.text,
            metadata=metadata,
        )

    # ------------------------------------------------------------------ #
    # Semantic Kernel async API
    # ------------------------------------------------------------------ #

    async def get_chat_message_content(
        self,
        chat_history: "ChatHistory",
        settings: "PromptExecutionSettings",
        **kwargs: Any,
    ) -> "ChatMessageContent":
        """
        Execute a single chat completion call via direct protocol access.
        """
        if not chat_history:
            raise ValueError("Chat history cannot be empty")

        corpus_messages = self._translate_messages(chat_history)
        ctx = self._build_operation_context(settings)
        params = self._build_sampling_params(settings, kwargs)

        model_for_context = params.get("model", self.model)
        messages_count = len(corpus_messages)
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
            completion: LLMCompletion = await self.llm_adapter.acomplete(
                messages=corpus_messages,
                ctx=ctx,
                **params,
            )
            return self._completion_to_chat_message(completion)

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
        Streaming chat completion via direct protocol access.

        Yields incremental StreamingChatMessageContent chunks compatible
        with Semantic Kernel's streaming APIs.
        """
        if not chat_history:
            raise ValueError("Chat history cannot be empty")

        corpus_messages = self._translate_messages(chat_history)
        ctx = self._build_operation_context(settings)
        params = self._build_sampling_params(settings, kwargs)

        model_for_context = params.get("model", self.model)
        messages_count = len(corpus_messages)
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
            async for chunk in self.llm_adapter.astream(
                messages=corpus_messages,
                ctx=ctx,
                **params,
            ):
                yield self._chunk_to_streaming_chat_message(chunk)

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
    "CorpusSemanticKernelChatCompletion",
]