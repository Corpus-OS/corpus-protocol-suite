# corpus_sdk/llm/framework_adapters/llamaindex.py
# SPDX-License-Identifier: Apache-2.0

"""
LlamaIndex adapter for Corpus LLM protocol.

This module exposes a Corpus `LLMProtocolV1` as a LlamaIndex `LLM`
implementation, with:

- Async + sync chat generation
- Async + sync streaming chat (true incremental streaming)
- Context propagation via `OperationContext`
- Protocol-first design: Direct LLMProtocolV1 access with message translation
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager, contextmanager
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    Mapping,
)

from corpus_sdk.core.context_translation import (
    from_llamaindex as context_from_llamaindex,
)
from corpus_sdk.core.error_context import attach_context
from corpus_sdk.llm.framework_adapters.common.message_translation import (
    NormalizedMessage,
    from_llamaindex,
    to_corpus,
)
from corpus_sdk.llm.llm_base import (
    LLMChunk,
    LLMCompletion,
    LLMProtocolV1,
    OperationContext,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional LlamaIndex imports
# ---------------------------------------------------------------------------

_LLAMAINDEX_IMPORT_ERROR: Optional[BaseException] = None

try:
    from llama_index.core.llms import (
        LLM,
        ChatMessage,
        ChatResponse,
        ChatResponseAsyncGen,
        ChatResponseGen,
        MessageRole,
    )
    from llama_index.core.callbacks import CallbackManager
except BaseException as exc:  # pragma: no cover - optional dependency path
    _LLAMAINDEX_IMPORT_ERROR = exc
    # Fallbacks for type checking and import-time safety
    LLM = object  # type: ignore[assignment]
    ChatMessage = object  # type: ignore[assignment]
    ChatResponse = object  # type: ignore[assignment]
    ChatResponseAsyncGen = AsyncIterator[Any]  # type: ignore[assignment]
    ChatResponseGen = Iterator[Any]  # type: ignore[assignment]
    MessageRole = object  # type: ignore[assignment]
    CallbackManager = object  # type: ignore[assignment]


def _ensure_llamaindex_installed() -> None:
    """Raise helpful error if LlamaIndex is not installed."""
    if _LLAMAINDEX_IMPORT_ERROR is not None:
        raise RuntimeError(
            "LlamaIndex is required to use CorpusLlamaIndexLLM. "
            "Install with: pip install 'llama-index-core>=0.10.0'"
        ) from _LLAMAINDEX_IMPORT_ERROR


class LlamaIndexLLMProtocol(Protocol):
    """Structural protocol for LlamaIndex-compatible Corpus LLM."""

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        ...

    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseAsyncGen:
        ...

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        ...

    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        ...

    def count_tokens(self, messages: Sequence[ChatMessage], **kwargs: Any) -> int:
        ...


# ---------------------------------------------------------------------------
# ChatResponse builders
# ---------------------------------------------------------------------------


def _build_chat_response(
    text: str,
    *,
    model: Optional[str] = None,
    finish_reason: Optional[str] = None,
    usage: Optional[Any] = None,
) -> ChatResponse:
    """
    Construct a minimal LlamaIndex ChatResponse from a Corpus LLMCompletion.

    Returns
    -------
    ChatResponse
        LlamaIndex chat response with proper metadata.
    """
    try:
        role_assistant = getattr(MessageRole, "ASSISTANT", "assistant")
    except Exception:
        role_assistant = "assistant"

    msg = ChatMessage(
        role=role_assistant,
        content=text,
    )

    additional_kwargs: Dict[str, Any] = {}
    if model is not None:
        additional_kwargs["model"] = model
    if finish_reason is not None:
        additional_kwargs["finish_reason"] = finish_reason
    if usage is not None:
        additional_kwargs["usage"] = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }

    try:
        return ChatResponse(message=msg, raw=None, additional_kwargs=additional_kwargs)
    except TypeError:
        # Older versions may not support additional_kwargs
        return ChatResponse(message=msg)


def _build_chat_response_from_chunk(chunk: LLMChunk) -> ChatResponse:
    """
    Build a ChatResponse for streaming from a single LLMChunk.

    Returns
    -------
    ChatResponse
        LlamaIndex chat response with incremental content.
    """
    try:
        role_assistant = getattr(MessageRole, "ASSISTANT", "assistant")
    except Exception:
        role_assistant = "assistant"

    msg = ChatMessage(
        role=role_assistant,
        content=chunk.text,
    )

    delta = chunk.text or ""
    usage = getattr(chunk, "usage_so_far", None)

    additional_kwargs: Dict[str, Any] = {
        "is_final": getattr(chunk, "is_final", False),
        "model": getattr(chunk, "model", None),
    }
    if usage is not None:
        additional_kwargs["usage_so_far"] = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }

    try:
        return ChatResponse(
            message=msg,
            delta=delta,
            raw=None,
            additional_kwargs=additional_kwargs,
        )
    except TypeError:
        # Older versions without delta/additional_kwargs
        return ChatResponse(message=msg)


# ---------------------------------------------------------------------------
# Internal metrics helpers for observability
# ---------------------------------------------------------------------------


def _analyze_messages_for_context(messages: Sequence[ChatMessage]) -> Dict[str, Any]:
    """
    Derive role distribution and content metrics for error context.

    Parameters
    ----------
    messages : Sequence[ChatMessage]
        LlamaIndex chat messages.

    Returns
    -------
    Dict[str, Any]
        Metrics including messages_count, roles_distribution and total_content_chars.
    """
    roles: Dict[str, int] = {}
    total_chars = 0

    for msg in messages:
        # LlamaIndex messages may expose `role` or `message.role`
        role = getattr(msg, "role", getattr(msg, "type", "unknown"))
        content = getattr(msg, "content", "")
        roles[role] = roles.get(role, 0) + 1
        if isinstance(content, str):
            total_chars += len(content)

    return {
        "messages_count": len(messages),
        "roles_distribution": roles,
        "total_content_chars": total_chars,
    }


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------


class CorpusLlamaIndexLLM(LLM):
    """
    LlamaIndex `LLM` implementation backed by a Corpus `LLMProtocolV1`.

    This is a thin integration layer with direct protocol access.

    Usage:
        llm = CorpusLlamaIndexLLM(
            llm_adapter=adapter,
            model="gpt-4",
            temperature=0.7
        )
        response = llm.chat([ChatMessage(role="user", content="Hello!")])
    """

    llm_adapter: LLMProtocolV1
    model: str = "default"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    framework_version: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data: Any) -> None:
        _ensure_llamaindex_installed()

        if "llm_adapter" in data and not hasattr(data["llm_adapter"], "complete"):
            raise TypeError("llm_adapter must implement LLMProtocolV1")

        if "temperature" in data and not 0 <= data["temperature"] <= 2:
            raise ValueError("temperature must be between 0 and 2")

        if "max_tokens" in data and data.get("max_tokens") is not None and data["max_tokens"] < 1:
            raise ValueError("max_tokens must be positive")

        super().__init__(**data)

        logger.info(
            "CorpusLlamaIndexLLM initialized with model=%s, temperature=%.2f",
            self.model,
            self.temperature,
        )

    # ------------------------------------------------------------------ #
    # Error context management
    # ------------------------------------------------------------------ #

    def _build_error_context(
        self,
        operation: str,
        stream: bool,
        messages: Sequence[ChatMessage],
        model: str,
        ctx: OperationContext,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build consistent, rich error context for observability.

        Includes both sampling parameters and message-level metrics.
        """
        msg_metrics = _analyze_messages_for_context(messages)
        return {
            "framework": "llamaindex",
            "operation": operation,
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
            **msg_metrics,
        }

    @contextmanager
    def _error_context(
        self,
        operation: str,
        stream: bool,
        messages: Sequence[ChatMessage],
        model: str,
        ctx: OperationContext,
        params: Dict[str, Any],
    ):
        """Sync error-context wrapper."""
        error_ctx = self._build_error_context(operation, stream, messages, model, ctx, params)
        try:
            yield
        except BaseException as exc:  # noqa: BLE001
            attach_context(exc, **error_ctx)
            raise

    @asynccontextmanager
    async def _error_context_async(
        self,
        operation: str,
        stream: bool,
        messages: Sequence[ChatMessage],
        model: str,
        ctx: OperationContext,
        params: Dict[str, Any],
    ):
        """Async error-context wrapper."""
        error_ctx = self._build_error_context(operation, stream, messages, model, ctx, params)
        try:
            yield
        except BaseException as exc:  # noqa: BLE001
            attach_context(exc, **error_ctx)
            raise

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _translate_messages(self, messages: Sequence[ChatMessage]) -> List[Dict[str, Any]]:
        """LlamaIndex messages → Corpus wire format."""
        if not messages:
            logger.warning("Empty messages list provided to LlamaIndex adapter")
            return []

        try:
            normalized: List[NormalizedMessage] = [from_llamaindex(m) for m in messages]
            return [dict(m) for m in to_corpus(normalized)]
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to translate LlamaIndex messages: %s", e)
            raise ValueError(f"Message translation failed: {e}") from e

    def _build_operation_context(self, kwargs: Mapping[str, Any]) -> OperationContext:
        """Build OperationContext from LlamaIndex kwargs."""
        callback_manager = kwargs.get("callback_manager")
        return context_from_llamaindex(
            callback_manager,
            framework_version=self.framework_version,
        )

    def _extract_stop_sequences(self, kwargs: Mapping[str, Any]) -> Optional[List[str]]:
        """Extract stop sequences from kwargs."""
        stop = kwargs.get("stop") or kwargs.get("stop_sequences")
        if stop is None:
            return None
        if isinstance(stop, str):
            return [stop]
        if isinstance(stop, (list, tuple)):
            return [str(s) for s in stop]
        return [str(stop)]

    def _build_sampling_params(
        self,
        stop: Optional[List[str]],
        kwargs: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Build sampling parameters from kwargs, with validation."""
        temperature = kwargs.get("temperature", self.temperature)
        if not 0 <= temperature <= 2:
            logger.warning("Temperature %s out of reasonable range 0-2", temperature)

        params: Dict[str, Any] = {
            "model": kwargs.get("model", self.model),
            "temperature": temperature,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "top_p": kwargs.get("top_p"),
            "frequency_penalty": kwargs.get("frequency_penalty"),
            "presence_penalty": kwargs.get("presence_penalty"),
            "stop_sequences": stop,
        }
        # Strip None for a clean param set
        return {k: v for k, v in params.items() if v is not None}

    # ------------------------------------------------------------------ #
    # Async API
    # ------------------------------------------------------------------ #

    async def achat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        """
        Async chat entrypoint required by LlamaIndex.

        Parameters
        ----------
        messages : Sequence[ChatMessage]
            Sequence of LlamaIndex ChatMessage objects.
        **kwargs : Any
            LlamaIndex parameters including callback_manager and sampling params.

        Returns
        -------
        ChatResponse
            LlamaIndex chat response with message content and metadata.
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")

        corpus_messages = self._translate_messages(messages)
        stop_sequences = self._extract_stop_sequences(kwargs)
        ctx = self._build_operation_context(kwargs)
        params = self._build_sampling_params(stop_sequences, kwargs)

        model_for_context = params.get("model", self.model)

        async with self._error_context_async(
            "achat",
            stream=False,
            messages=messages,
            model=model_for_context,
            ctx=ctx,
            params=params,
        ):
            result: LLMCompletion = await self.llm_adapter.acomplete(
                messages=corpus_messages,
                ctx=ctx,
                **params,
            )

            return _build_chat_response(
                text=result.text,
                model=getattr(result, "model", None),
                finish_reason=getattr(result, "finish_reason", None),
                usage=getattr(result, "usage", None),
            )

    async def astream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        """
        Async streaming chat entrypoint required by LlamaIndex.

        Parameters
        ----------
        messages : Sequence[ChatMessage]
            Sequence of LlamaIndex ChatMessage objects.
        **kwargs : Any
            LlamaIndex parameters including callback_manager and sampling params.

        Returns
        -------
        ChatResponseAsyncGen
            Async generator of ChatResponse objects with incremental content.
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")

        corpus_messages = self._translate_messages(messages)
        stop_sequences = self._extract_stop_sequences(kwargs)
        ctx = self._build_operation_context(kwargs)
        params = self._build_sampling_params(stop_sequences, kwargs)

        model_for_context = params.get("model", self.model)

        async with self._error_context_async(
            "astream_chat",
            stream=True,
            messages=messages,
            model=model_for_context,
            ctx=ctx,
            params=params,
        ):
            async def _gen() -> AsyncIterator[ChatResponse]:
                # Direct protocol access for streaming
                async for chunk in self.llm_adapter.astream(
                    messages=corpus_messages,
                    ctx=ctx,
                    **params,
                ):
                    yield _build_chat_response_from_chunk(chunk)

            return _gen()

    # ------------------------------------------------------------------ #
    # Sync API
    # ------------------------------------------------------------------ #

    def chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        """
        Sync chat entrypoint required by LlamaIndex.

        Parameters
        ----------
        messages : Sequence[ChatMessage]
            Sequence of LlamaIndex ChatMessage objects.
        **kwargs : Any
            LlamaIndex parameters including callback_manager and sampling params.

        Returns
        -------
        ChatResponse
            LlamaIndex chat response with message content and metadata.
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")

        corpus_messages = self._translate_messages(messages)
        stop_sequences = self._extract_stop_sequences(kwargs)
        ctx = self._build_operation_context(kwargs)
        params = self._build_sampling_params(stop_sequences, kwargs)

        model_for_context = params.get("model", self.model)

        with self._error_context(
            "chat",
            stream=False,
            messages=messages,
            model=model_for_context,
            ctx=ctx,
            params=params,
        ):
            result: LLMCompletion = self.llm_adapter.complete(
                messages=corpus_messages,
                ctx=ctx,
                **params,
            )

            return _build_chat_response(
                text=result.text,
                model=getattr(result, "model", None),
                finish_reason=getattr(result, "finish_reason", None),
                usage=getattr(result, "usage", None),
            )

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseGen:
        """
        Sync streaming chat entrypoint required by LlamaIndex.

        Parameters
        ----------
        messages : Sequence[ChatMessage]
            Sequence of LlamaIndex ChatMessage objects.
        **kwargs : Any
            LlamaIndex parameters including callback_manager and sampling params.

        Returns
        -------
        ChatResponseGen
            Generator of ChatResponse objects with incremental content.
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")

        corpus_messages = self._translate_messages(messages)
        stop_sequences = self._extract_stop_sequences(kwargs)
        ctx = self._build_operation_context(kwargs)
        params = self._build_sampling_params(stop_sequences, kwargs)

        model_for_context = params.get("model", self.model)

        with self._error_context(
            "stream_chat",
            stream=True,
            messages=messages,
            model=model_for_context,
            ctx=ctx,
            params=params,
        ):
            def _gen() -> Iterator[ChatResponse]:
                # Direct protocol access for streaming
                for chunk in self.llm_adapter.stream(
                    messages=corpus_messages,
                    ctx=ctx,
                    **params,
                ):
                    yield _build_chat_response_from_chunk(chunk)

            return _gen()

    # ------------------------------------------------------------------ #
    # Token counting
    # ------------------------------------------------------------------ #

    def count_tokens(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> int:
        """
        Token counting helper.

        Parameters
        ----------
        messages : Sequence[ChatMessage]
            Sequence of LlamaIndex ChatMessage objects.
        **kwargs : Any
            Additional parameters.

        Returns
        -------
        int
            Estimated token count for the messages.
        """
        if not messages:
            return 0

        ctx = self._build_operation_context(kwargs)
        model = kwargs.get("model", self.model)

        # Try protocol adapter counting first
        if hasattr(self.llm_adapter, "count_tokens"):
            try:
                combined_text = self._combine_messages_for_counting(messages)
                tokens = self.llm_adapter.count_tokens(
                    text=combined_text,
                    model=model,
                    ctx=ctx,
                )
                return int(tokens)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Protocol count_tokens failed: %s", exc)

        # Fall back to improved character-based estimate
        try:
            return self._estimate_tokens_from_messages(messages)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Token estimation failed: %s", exc)

        # Ultimate fallback to simple character count
        return self._simple_token_estimate(messages)

    def _combine_messages_for_counting(self, messages: Sequence[ChatMessage]) -> str:
        """Combine messages into a single string for token counting."""
        parts: List[str] = []
        for msg in messages:
            role = getattr(msg, "type", getattr(msg, "role", "user"))
            content = str(getattr(msg, "content", ""))
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def _estimate_tokens_from_messages(self, messages: Sequence[ChatMessage]) -> int:
        """Improved token estimation with better heuristics."""
        combined_text = self._combine_messages_for_counting(messages)

        if not combined_text:
            return 0

        char_count = len(combined_text)
        message_count = len(messages)

        char_based = max(1, char_count // 4)
        message_based = max(1, message_count)

        return max(char_based, message_based)

    def _simple_token_estimate(self, messages: Sequence[ChatMessage]) -> int:
        """Simple fallback token estimation."""
        combined_text = self._combine_messages_for_counting(messages)
        return max(1, len(combined_text) // 4)


__all__ = [
    "LlamaIndexLLMProtocol",
    "CorpusLlamaIndexLLM",
]