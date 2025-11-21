# corpus_sdk/llm/framework_adapters/semantic_kernel.py
# SPDX-License-Identifier: Apache-2.0

"""
Semantic Kernel adapter for Corpus LLM protocol.

This module exposes a Corpus `LLMProtocolV1` as a Semantic Kernel
`ChatCompletionClientBase` implementation so that:

- SK agents can use any Corpus-backed LLM implementation as a chat completion service.
- Async + streaming flows remain async-first (no background threads).
- Context / deadlines / tenant hints are propagated via `OperationContext`.
- Sampling parameters are bridged from `PromptExecutionSettings` to Corpus.
- Rich error context is attached for enhanced observability and debugging.

Design goals
------------

* Protocol-first, translator-centric:
    Semantic Kernel is an integration surface. All real behavior goes
    through the Corpus `LLMProtocolV1`, with a lightweight local translator
    that centralizes protocol calls for this adapter.

* Optional dependency safe:
    This module can be imported without Semantic Kernel installed.
    Attempting to *instantiate* the adapter without SK will raise a
    clear RuntimeError.

* Non-lossy metadata where feasible:
    - We preserve SK-side model / finish_reason / usage in message metadata.
    - Deadlines / timeouts flowing through `PromptExecutionSettings` are
      mapped into `OperationContext.deadline_ms` via
      `ContextTranslator.from_semantic_kernel_context`.

* Async-first:
    - No sync wrappers or background threads.
    - Cancellation works via normal asyncio task cancellation semantics
      plus ctx.deadline_ms at the Corpus protocol layer.

* Centralized resilience:
    - Any retries / backoff / circuit breaking live in the
      `LLMProtocolV1` implementation, not here.

Usage:
    sk_chat = CorpusSemanticKernelChatCompletion(
        llm_adapter=adapter,
        model="gpt-4",
        temperature=0.7
    )
    response = await sk_chat.get_chat_message_content(chat_history, settings)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from functools import cached_property
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
)

from corpus_sdk.core.context_translation import ContextTranslator
from corpus_sdk.core.error_context import attach_context
from corpus_sdk.llm.framework_adapters.common.message_translation import (
    MessageTranslator,
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

try:  # pragma: no cover - import surface only; behavior is type-driven
    from semantic_kernel.connectors.ai.chat_completion_client_base import (  # type: ignore[import]
        ChatCompletionClientBase,
        ChatHistory,
        PromptExecutionSettings,
    )
    from semantic_kernel.contents import (  # type: ignore[import]
        AuthorRole,
        ChatMessageContent,
        StreamingChatMessageContent,
    )
except Exception as _IMPORT_EXC:  # pragma: no cover - runtime guard
    # We still allow importing this module so that type checking works.
    # Instantiating the adapter without SK present will raise a RuntimeError.
    ChatCompletionClientBase = object  # type: ignore[misc,assignment]
    ChatHistory = Any  # type: ignore[misc,assignment]
    PromptExecutionSettings = Any  # type: ignore[misc,assignment]
    AuthorRole = Any  # type: ignore[misc,assignment]
    ChatMessageContent = Any  # type: ignore[misc,assignment]
    StreamingChatMessageContent = Any  # type: ignore[misc,assignment]
    _SEMANTIC_KERNEL_IMPORT_ERROR: Optional[Exception] = _IMPORT_EXC
else:  # pragma: no cover - trivial branch
    _SEMANTIC_KERNEL_IMPORT_ERROR = None


# ---------------------------------------------------------------------------
# Protocol (structural interface)
# ---------------------------------------------------------------------------


class SemanticKernelLLMProtocol(Protocol):
    """
    Structural protocol for a Semantic Kernel–compatible Corpus LLM adapter.

    This matches the key async interface SK uses, without depending on the
    concrete `CorpusSemanticKernelChatCompletion` class.
    """

    async def get_chat_message_content(
        self,
        chat_history: ChatHistory,
        settings: PromptExecutionSettings,
        **kwargs: Any,
    ) -> ChatMessageContent:
        ...

    async def get_chat_message_contents(
        self,
        chat_history: ChatHistory,
        settings: PromptExecutionSettings,
        **kwargs: Any,
    ) -> List[ChatMessageContent]:
        ...

    async def get_streaming_chat_message_content(
        self,
        chat_history: ChatHistory,
        settings: PromptExecutionSettings,
        **kwargs: Any,
    ) -> AsyncIterator[StreamingChatMessageContent]:
        ...

    async def get_streaming_chat_message_contents(
        self,
        chat_history: ChatHistory,
        settings: PromptExecutionSettings,
        **kwargs: Any,
    ) -> AsyncIterator[StreamingChatMessageContent]:
        ...


# ---------------------------------------------------------------------------
# Internal helpers (robust with validation)
# ---------------------------------------------------------------------------


def _author_role_to_protocol_role(role: Any) -> str:
    """
    Map SK AuthorRole → Corpus protocol role string.

    We deliberately avoid relying on exact enum values; instead we use a
    string-based mapping to be resilient to minor API changes.
    """
    if role is None:
        return "user"

    # AuthorRole may be an enum with `.name` or `.value`.
    name = getattr(role, "name", None) or getattr(role, "value", None) or str(role)
    key = str(name).strip().lower()

    if "system" in key:
        return "system"
    if "assistant" in key or key == "ai":
        return "assistant"
    if "tool" in key or "function" in key:
        return "tool"
    return "user"


def _extract_text_from_chat_message(msg: Any) -> str:
    """
    Best-effort extraction of human-readable text from a ChatMessageContent.

    Strategy:
    - Prefer `msg.content` when present.
    - If empty, fall back to concatenation of text-bearing `items` when available.
    - As a last resort, use `str(msg)`.
    """
    # Preferred: direct content string.
    content = getattr(msg, "content", None)
    if isinstance(content, str) and content:
        return content

    # Fallback: iterate items with `.text` fields (TextContent, etc.).
    items = getattr(msg, "items", None)
    parts: List[str] = []
    if items:
        for item in items:
            text = getattr(item, "text", None)
            if isinstance(text, str) and text:
                parts.append(text)
    if parts:
        return "\n".join(parts)

    # Final fallback.
    try:
        return str(msg)
    except Exception:  # pragma: no cover - ultra defensive
        return ""


def _history_to_normalized_messages(history: ChatHistory) -> List[NormalizedMessage]:
    """
    Convert SK ChatHistory → list[NormalizedMessage] via the common translator.

    We shape each SK chat message into the minimal SK dict format expected by
    `MessageTranslator.from_semantic_kernel`, and then later map that to
    Corpus wire format via `MessageTranslator.to_corpus`.
    """
    if not history:
        logger.warning("Empty chat history provided to Semantic Kernel adapter")
        return []

    try:
        normalized: List[NormalizedMessage] = []

        # `ChatHistory` is iterable in recent SK versions; if not, we fall back
        # to `history.messages` when present.
        try:
            iterable = list(history)  # type: ignore[arg-type]
        except TypeError:
            iterable = getattr(history, "messages", []) or []

        for msg in iterable:
            role = _author_role_to_protocol_role(getattr(msg, "role", None))
            text = _extract_text_from_chat_message(msg)
            metadata: Mapping[str, Any] = getattr(msg, "metadata", {}) or {}

            minimal: Dict[str, Any] = {
                "role": role,
                "content": text,
                "metadata": dict(metadata),
            }

            nm = MessageTranslator.from_semantic_kernel(minimal)
            normalized.append(nm)

        return normalized
    except Exception as e:
        logger.error("Failed to translate Semantic Kernel chat history: %s", e)
        raise ValueError(f"Chat history translation failed: {e}") from e


def _normalized_to_corpus_messages(
    messages: List[NormalizedMessage],
) -> List[Dict[str, Any]]:
    """
    NormalizedMessage list → Corpus protocol wire messages.
    """
    return MessageTranslator.to_corpus(messages)


def _log_sampling_param_warnings(params: Mapping[str, Any]) -> None:
    """
    Best-effort validation of SK sampling params.

    We do not enforce ranges here (that's the protocol/adapter's job),
    but we log when values are clearly outside typical production ranges.
    """
    temp = params.get("temperature")
    if temp is not None and isinstance(temp, (int, float)):
        if not (0.0 <= float(temp) <= 2.0):
            logger.warning(
                "Semantic Kernel settings.temperature=%r outside typical [0.0, 2.0] range",
                temp,
            )

    top_p = params.get("top_p")
    if top_p is not None and isinstance(top_p, (int, float)):
        if not (0.0 < float(top_p) <= 1.0):
            logger.warning(
                "Semantic Kernel settings.top_p=%r outside typical (0.0, 1.0] range",
                top_p,
            )


def _build_ctx_and_sampling_params(
    *,
    settings: PromptExecutionSettings,
    framework_version: Optional[str] = None,
    model_override: Optional[str] = None,
    default_model: str,
    default_temperature: float,
    default_max_tokens: Optional[int],
) -> tuple[OperationContext, Dict[str, Any]]:
    """
    Build OperationContext and Corpus sampling params from SK prompt settings.
    """
    # Use ContextTranslator to preserve tenant / trace / deadline when possible.
    ctx: Optional[OperationContext]
    try:
        ctx = ContextTranslator.from_semantic_kernel_context(
            sk_context=None,
            settings=settings,
            framework_version=framework_version,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "ContextTranslator.from_semantic_kernel_context failed: %s",
            exc,
            extra={"framework": "semantic_kernel"},
        )
        ctx = None

    if not isinstance(ctx, OperationContext):
        # Fallback: empty but valid context
        ctx = OperationContext(
            request_id=None,
            idempotency_key=None,
            deadline_ms=None,
            traceparent=None,
            tenant=None,
            attrs={},
        )

    # Settings introspection is intentionally conservative and string-based to
    # tolerate minor SK version changes.
    def _get(name: str) -> Optional[Any]:
        return getattr(settings, name, None)

    # Model resolution:
    # 1) explicit override
    # 2) settings.model_id / settings.model / settings.deployment_name
    # 3) adapter default.
    model = (
        model_override
        or _get("model_id")
        or _get("model")
        or _get("deployment_name")
        or default_model
    )

    # Sampling params from settings, falling back to adapter defaults.
    temperature = _get("temperature")
    if temperature is None:
        temperature = default_temperature

    max_tokens = _get("max_tokens")
    if max_tokens is None:
        max_tokens = default_max_tokens

    params: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": _get("top_p"),
        "frequency_penalty": _get("frequency_penalty"),
        "presence_penalty": _get("presence_penalty"),
        # SK often uses `stop_sequences` or `stop` depending on provider.
        "stop_sequences": _get("stop_sequences") or _get("stop"),
    }

    # Light validation / warnings; real enforcement happens in protocol/adapter.
    _log_sampling_param_warnings(params)

    # Strip out Nones so the protocol sees a clean param dict.
    return ctx, {k: v for k, v in params.items() if v is not None}


def _completion_to_chat_message(
    completion: LLMCompletion,
) -> ChatMessageContent:
    """
    LLMCompletion → SK ChatMessageContent.

    We attach model / finish_reason / usage into metadata for SK-side
    consumers that care about these details.
    """
    metadata: Dict[str, Any] = {
        "model": completion.model,
        "model_family": completion.model_family,
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
    chunk: LLMChunk,
) -> StreamingChatMessageContent:
    """
    LLMChunk → SK StreamingChatMessageContent.

    We treat `chunk.text` as the delta for streaming. Model and usage_so_far
    are exposed via metadata.
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


# ---------------------------------------------------------------------------
# Local LLM translator (mini abstraction over LLMProtocolV1)
# --------------------------------------------------------------------------- #


class _LocalLLMTranslator:
    """
    Minimal translator for the Semantic Kernel adapter.

    This local translator centralizes calls into `LLMProtocolV1` and exposes
    a stable async interface:

        - arun_complete
        - arun_stream

    so the adapter logic can remain focused on SK-specific shaping
    (ChatHistory in, ChatMessageContent/StreamingChatMessageContent out).
    """

    def __init__(self, adapter: LLMProtocolV1, framework: str = "semantic_kernel") -> None:
        self._adapter = adapter
        self._framework = framework  # reserved for future use (metrics, logging, etc.)

    async def arun_complete(
        self,
        *,
        messages: List[Dict[str, Any]],
        op_ctx: OperationContext,
        params: Dict[str, Any],
    ) -> LLMCompletion:
        """Async completion routed directly to the underlying LLMProtocolV1."""
        return await self._adapter.acomplete(
            messages=messages,
            ctx=op_ctx,
            **params,
        )

    async def arun_stream(
        self,
        *,
        messages: List[Dict[str, Any]],
        op_ctx: OperationContext,
        params: Dict[str, Any],
    ) -> AsyncIterator[LLMChunk]:
        """Async streaming routed directly to the underlying LLMProtocolV1."""
        async for chunk in self._adapter.astream(
            messages=messages,
            ctx=op_ctx,
            **params,
        ):
            yield chunk


# ---------------------------------------------------------------------------
# Public adapter
# ---------------------------------------------------------------------------


class CorpusSemanticKernelChatCompletion(ChatCompletionClientBase):
    """
    Semantic Kernel `ChatCompletionClientBase` backed by a Corpus LLM protocol.

    This class allows SK agents to use any implementation of `LLMProtocolV1`
    as their chat completion service, via a small local translator.

    Attributes
    ----------
    llm_adapter:
        Underlying Corpus adapter implementing `LLMProtocolV1`.

    default model + sampling defaults:
        Used when SK settings do not override them.

    framework_version:
        Optional Semantic Kernel version string for context attribution.

    service_id:
        Optional SK service identifier used by the Kernel's service registry.
    """

    # Underlying protocol implementation
    llm_adapter: LLMProtocolV1

    # Default sampling configuration
    _default_model: str
    _default_temperature: float
    _default_max_tokens: Optional[int]

    # Optional framework metadata
    _framework_version: Optional[str]

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
            # Give a very direct, one-hop error for misconfiguration.
            raise RuntimeError(
                "semantic-kernel is not installed. "
                "Install it via `pip install semantic-kernel` to use "
                "CorpusSemanticKernelChatCompletion."
            ) from _SEMANTIC_KERNEL_IMPORT_ERROR

        # Validate critical parameters
        if not isinstance(llm_adapter, LLMProtocolV1):
            raise TypeError("llm_adapter must implement LLMProtocolV1")

        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")

        if max_tokens is not None and max_tokens < 1:
            raise ValueError("max_tokens must be positive")

        super().__init__(service_id=service_id)

        self.llm_adapter = llm_adapter
        self._default_model = model
        self._default_temperature = float(temperature)
        self._default_max_tokens = max_tokens
        self._framework_version = framework_version

    # ------------------------------------------------------------------ #
    # Translator (cached for optimal performance)
    # ------------------------------------------------------------------ #

    @cached_property
    def _translator(self) -> _LocalLLMTranslator:
        """
        Lazily construct and cache the local translator.

        All orchestration specific to Semantic Kernel is centralized here,
        while protocol behavior lives in the underlying `LLMProtocolV1`.
        """
        return _LocalLLMTranslator(
            adapter=self.llm_adapter,
            framework="semantic_kernel",
        )

    # ------------------------------------------------------------------ #
    # Error-context helper (robust and consistent)
    # ------------------------------------------------------------------ #

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
    ):
        """
        Async error-context wrapper to centralize attach_context usage.
        """
        try:
            yield
        except BaseException as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="semantic_kernel",
                operation=operation,
                messages_count=messages_count,
                model=model,
                temperature=params.get("temperature"),
                max_tokens=params.get("max_tokens"),
                top_p=params.get("top_p"),
                frequency_penalty=params.get("frequency_penalty"),
                presence_penalty=params.get("presence_penalty"),
                request_id=ctx.request_id,
                tenant=ctx.tenant,
                stream=stream,
            )
            raise

    # ------------------------------------------------------------------ #
    # Core async API expected by ChatCompletionClientBase
    # ------------------------------------------------------------------ #

    async def get_chat_message_content(
        self,
        chat_history: ChatHistory,
        settings: PromptExecutionSettings,
        **kwargs: Any,
    ) -> ChatMessageContent:
        """
        Execute a single chat completion call via the Corpus LLM protocol.

        Parameters
        ----------
        chat_history:
            Semantic Kernel ChatHistory containing the conversation.

        settings:
            PromptExecutionSettings with model and sampling parameters.

        kwargs:
            Additional parameters including:
            - model: Optional model override
            - Other framework-specific context

        Returns
        -------
        ChatMessageContent
            Semantic Kernel chat message with assistant response.

        Raises
        ------
        ValueError
            If chat_history is empty or message translation fails.
        RuntimeError
            If the underlying LLM protocol call fails.
        """
        if not chat_history:
            raise ValueError("Chat history cannot be empty")

        normalized = _history_to_normalized_messages(chat_history)
        corpus_messages = _normalized_to_corpus_messages(normalized)

        ctx, params = _build_ctx_and_sampling_params(
            settings=settings,
            framework_version=self._framework_version,
            model_override=kwargs.get("model"),
            default_model=self._default_model,
            default_temperature=self._default_temperature,
            default_max_tokens=self._default_max_tokens,
        )

        model_for_context = params.get("model", self._default_model)
        messages_count = len(corpus_messages)

        async with self._error_context_async(
            "get_chat_message_content",
            stream=False,
            messages_count=messages_count,
            model=model_for_context,
            ctx=ctx,
            params=params,
        ):
            completion: LLMCompletion = await self._translator.arun_complete(
                messages=corpus_messages,
                op_ctx=ctx,
                params=params,
            )
            return _completion_to_chat_message(completion)

    async def get_chat_message_contents(
        self,
        chat_history: ChatHistory,
        settings: PromptExecutionSettings,
        **kwargs: Any,
    ) -> List[ChatMessageContent]:
        """
        Convenience wrapper returning a single-element list for SK APIs that
        expect a collection of messages.

        Parameters
        ----------
        chat_history:
            Semantic Kernel ChatHistory containing the conversation.

        settings:
            PromptExecutionSettings with model and sampling parameters.

        kwargs:
            Additional parameters including model override and context.

        Returns
        -------
        List[ChatMessageContent]
            Single-element list containing the assistant response.

        Note
        ----
        This method exists for compatibility with SK APIs that expect
        multiple messages, though typically only one response is generated.
        """
        msg = await self.get_chat_message_content(chat_history, settings, **kwargs)
        return [msg]

    async def get_streaming_chat_message_content(
        self,
        chat_history: ChatHistory,
        settings: PromptExecutionSettings,
        **kwargs: Any,
    ) -> AsyncIterator[StreamingChatMessageContent]:
        """
        Streaming chat completion via the local translator's `arun_stream`.

        Yields Semantic Kernel `StreamingChatMessageContent` objects for each
        chunk produced by the Corpus protocol.

        Parameters
        ----------
        chat_history:
            Semantic Kernel ChatHistory containing the conversation.

        settings:
            PromptExecutionSettings with model and sampling parameters.

        kwargs:
            Additional parameters including model override and context.

        Yields
        ------
        StreamingChatMessageContent
            Incremental streaming chat messages with metadata.

        Raises
        ------
        ValueError
            If chat_history is empty or message translation fails.
        RuntimeError
            If the underlying LLM protocol streaming fails.
        """
        if not chat_history:
            raise ValueError("Chat history cannot be empty")

        normalized = _history_to_normalized_messages(chat_history)
        corpus_messages = _normalized_to_corpus_messages(normalized)

        ctx, params = _build_ctx_and_sampling_params(
            settings=settings,
            framework_version=self._framework_version,
            model_override=kwargs.get("model"),
            default_model=self._default_model,
            default_temperature=self._default_temperature,
            default_max_tokens=self._default_max_tokens,
        )

        model_for_context = params.get("model", self._default_model)
        messages_count = len(corpus_messages)

        async with self._error_context_async(
            "get_streaming_chat_message_content",
            stream=True,
            messages_count=messages_count,
            model=model_for_context,
            ctx=ctx,
            params=params,
        ):
            async for chunk in self._translator.arun_stream(
                messages=corpus_messages,
                op_ctx=ctx,
                params=params,
            ):
                yield _chunk_to_streaming_chat_message(chunk)

    async def get_streaming_chat_message_contents(
        self,
        chat_history: ChatHistory,
        settings: PromptExecutionSettings,
        **kwargs: Any,
    ) -> AsyncIterator[StreamingChatMessageContent]:
        """
        Convenience wrapper providing an alias returning an async iterator of
        streaming messages. Recent SK versions favor the singular name, but
        some APIs may expect the plural form.

        Parameters
        ----------
        chat_history:
            Semantic Kernel ChatHistory containing the conversation.

        settings:
            PromptExecutionSettings with model and sampling parameters.

        kwargs:
            Additional parameters including model override and context.

        Yields
        ------
        StreamingChatMessageContent
            Incremental streaming chat messages with metadata.

        Note
        ----
        This method delegates to `get_streaming_chat_message_content` and
        exists for API compatibility with different SK versions.
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
