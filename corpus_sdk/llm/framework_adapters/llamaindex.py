# corpus_sdk/llm/framework_adapters/llamaindex.py
# SPDX-License-Identifier: Apache-2.0

"""
LlamaIndex adapter for Corpus LLM protocol.

This module exposes a Corpus `LLMProtocolV1` as a LlamaIndex `LLM`
implementation, with:

- Async + sync chat generation
- Async + sync streaming chat (true incremental streaming)
- Context propagation via `OperationContext`
- Protocol-first, translator-centric design: LlamaIndex is a thin skin over Corpus
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager, contextmanager
from functools import cached_property
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

from corpus_sdk.core.context_translation import ContextTranslator
from corpus_sdk.core.error_context import attach_context
from corpus_sdk.llm.framework_adapters.common.llm_translation import (
    DefaultLLMFrameworkTranslator,
    LLMTranslator,
)
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

try:  # pragma: no cover - import shape differs across versions
    # Newer LlamaIndex (0.10+) layout
    from llama_index.core.llms import (  # type: ignore[import]
        LLM,
        ChatMessage,
        ChatResponse,
        ChatResponseAsyncGen,
        ChatResponseGen,
        MessageRole,
    )
    from llama_index.core.callbacks import CallbackManager  # type: ignore[import]
except BaseException as exc:  # pragma: no cover - optional dependency
    _LLAMAINDEX_IMPORT_ERROR = exc
    # Minimal stand-ins for type checkers; never used at runtime if LI is missing.
    LLM = object  # type: ignore[assignment]
    ChatMessage = object  # type: ignore[assignment]
    ChatResponse = object  # type: ignore[assignment]
    ChatResponseAsyncGen = AsyncIterator[Any]  # type: ignore[assignment]
    ChatResponseGen = Iterator[Any]  # type: ignore[assignment]
    MessageRole = object  # type: ignore[assignment]
    CallbackManager = object  # type: ignore[assignment]


def _ensure_llamaindex_installed() -> None:
    """
    Raise a helpful error if LlamaIndex is not installed.

    Importing this module is always safe; instantiating the adapter
    checks the optional dependency.
    """
    if _LLAMAINDEX_IMPORT_ERROR is not None:
        raise RuntimeError(
            "LlamaIndex is required to use CorpusLlamaIndexLLM. "
            "Install with: pip install 'llama-index-core>=0.10.0'"
        ) from _LLAMAINDEX_IMPORT_ERROR


# ---------------------------------------------------------------------------
# Protocol (structural interface)
# ---------------------------------------------------------------------------


class LlamaIndexLLMProtocol(Protocol):
    """
    Structural protocol for a LlamaIndex-compatible Corpus LLM.

    This lets callers type against the adapter interface without
    depending on the concrete `CorpusLlamaIndexLLM` class.
    """

    async def achat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        ...

    async def astream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        ...

    def chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        ...

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseGen:
        ...

    def count_tokens(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> int:
        ...


# ---------------------------------------------------------------------------
# Internal helpers (robust with validation)
# ---------------------------------------------------------------------------


def _translate_messages_to_corpus(
    messages: Sequence[Any],
) -> List[Dict[str, str]]:
    """
    LlamaIndex ChatMessage -> Corpus wire messages with error handling.
    """
    if not messages:
        logger.warning("Empty messages list provided to LlamaIndex adapter")
        return []

    try:
        normalized: List[NormalizedMessage] = [from_llamaindex(m) for m in messages]
        return [dict(m) for m in to_corpus(normalized)]
    except Exception as e:
        logger.error("Failed to translate LlamaIndex messages: %s", e)
        raise ValueError(f"Message translation failed: {e}") from e


def _extract_stop_sequences(kwargs: Mapping[str, Any]) -> Optional[List[str]]:
    """
    Heuristically extract stop sequences from LlamaIndex kwargs.

    Supports:
        - stop="foo"
        - stop=["foo", "bar"]
        - stop_sequences=[...]
    """
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


def _build_operation_context_from_callbacks(
    *,
    adapter: "CorpusLlamaIndexLLM",
    kwargs: Mapping[str, Any],
) -> Optional[OperationContext]:
    """
    Build an OperationContext using the LlamaIndex callback manager.

    Prefer the `callback_manager` passed in kwargs; otherwise fall back
    to the adapter's own callback manager attribute, if present.
    """
    cbm: Optional[Any] = kwargs.get("callback_manager", None)
    if cbm is None:
        cbm = getattr(adapter, "callback_manager", None)

    try:
        return ContextTranslator.from_llamaindex_callback_manager(cbm)
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "ContextTranslator.from_llamaindex_callback_manager failed: %s",
            exc,
            extra={"framework": "llamaindex"},
        )
        return None


def _sampling_params_from_kwargs(
    *,
    adapter: "CorpusLlamaIndexLLM",
    stop_sequences: Optional[List[str]],
    kwargs: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Map LlamaIndex sampling kwargs into Corpus/translator params.
    """
    # Validate temperature if provided
    temperature = kwargs.get("temperature", adapter.temperature)
    if not 0 <= temperature <= 2:
        logger.warning("Temperature %s out of reasonable range 0-2", temperature)

    params: Dict[str, Any] = {
        "model": kwargs.get("model", adapter.model),
        "max_tokens": kwargs.get("max_tokens", adapter.max_tokens),
        "temperature": temperature,
        "top_p": kwargs.get("top_p"),
        "frequency_penalty": kwargs.get("frequency_penalty"),
        "presence_penalty": kwargs.get("presence_penalty"),
        "stop_sequences": stop_sequences,
    }
    # Strip None values so the translator sees a clean param set.
    return {k: v for k, v in params.items() if v is not None}


def _build_chat_response(
    text: str,
    *,
    model: Optional[str] = None,
    finish_reason: Optional[str] = None,
    usage: Optional[Any] = None,
) -> ChatResponse:
    """
    Construct a minimal LlamaIndex ChatResponse from a Corpus LLMCompletion.
    """
    try:
        role_assistant = getattr(MessageRole, "ASSISTANT", "assistant")
    except Exception:  # pragma: no cover
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
        return ChatResponse(message=msg)  # type: ignore[call-arg]


def _build_chat_response_from_chunk(chunk: LLMChunk) -> ChatResponse:
    """
    Build a ChatResponse for streaming from a single LLMChunk.

    The `delta` is set to the chunk text so consumers can reconstruct
    the full message incrementally if desired.
    """
    try:
        role_assistant = getattr(MessageRole, "ASSISTANT", "assistant")
    except Exception:  # pragma: no cover
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
        return ChatResponse(message=msg)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------


class CorpusLlamaIndexLLM(LLM):  # type: ignore[misc]
    """
    LlamaIndex `LLM` implementation backed by a Corpus `LLMProtocolV1`.

    This class is a thin integration layer:

    - Messages are normalized via `message_translation.from_llamaindex`.
    - Context is derived from LlamaIndex's callback manager via
      `ContextTranslator.from_llamaindex_callback_manager`.
    - All policy / resilience and async→sync bridging lives in
      `LLMTranslator` and the underlying `LLMProtocolV1`, not here.

    Usage:
        llm = CorpusLlamaIndexLLM(
            llm_adapter=adapter,
            model="gpt-4",
            temperature=0.7
        )
        response = llm.chat([ChatMessage(role="user", content="Hello!")])
    """

    # Underlying protocol implementation
    llm_adapter: LLMProtocolV1

    # Defaults for sampling with validation
    model: str = "default"
    temperature: float = 0.7
    max_tokens: Optional[int] = None

    # Pydantic / LlamaIndex config
    model_config = {"arbitrary_types_allowed": True}

    class _LlamaIndexLLMFrameworkTranslator(DefaultLLMFrameworkTranslator):
        """
        LlamaIndex-specific framework translator.

        Currently just inherits the default behavior (pass-through of
        protocol-level results), but exists as a dedicated hook for
        LlamaIndex-specific customizations in the future.
        """

        pass

    def __init__(self, **data: Any) -> None:
        _ensure_llamaindex_installed()
        
        # Validate critical parameters if provided
        if 'llm_adapter' in data and not isinstance(data['llm_adapter'], LLMProtocolV1):
            raise TypeError("llm_adapter must implement LLMProtocolV1")
        
        if 'temperature' in data and not 0 <= data['temperature'] <= 2:
            raise ValueError("temperature must be between 0 and 2")
            
        if 'max_tokens' in data and data['max_tokens'] is not None and data['max_tokens'] < 1:
            raise ValueError("max_tokens must be positive")
        
        super().__init__(**data)

    # ------------------------------------------------------------------ #
    # Translator (cached for performance)
    # ------------------------------------------------------------------ #

    @cached_property
    def _translator(self) -> LLMTranslator:
        """
        Lazily construct and cache the `LLMTranslator`.

        All orchestration, including any async→sync bridging needed by the
        underlying protocol implementation, is centralized in `LLMTranslator`.
        """
        framework_translator = self._LlamaIndexLLMFrameworkTranslator()
        return LLMTranslator(
            adapter=self.llm_adapter,
            framework="llamaindex",
            translator=framework_translator,
        )

    # ------------------------------------------------------------------ #
    # Error-context helpers (robust and consistent)
    # ------------------------------------------------------------------ #

    @contextmanager
    def _error_context(
        self,
        operation: str,
        *,
        stream: bool,
        messages_count: int,
        model: str,
        ctx: Optional[OperationContext],
        params: Dict[str, Any],
    ):
        """
        Sync error-context wrapper to centralize attach_context usage.
        """
        try:
            yield
        except BaseException as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="llamaindex",
                operation=operation,
                messages_count=messages_count,
                model=model,
                temperature=params.get("temperature"),
                max_tokens=params.get("max_tokens"),
                top_p=params.get("top_p"),
                frequency_penalty=params.get("frequency_penalty"),
                presence_penalty=params.get("presence_penalty"),
                request_id=getattr(ctx, "request_id", None) if ctx is not None else None,
                tenant=getattr(ctx, "tenant", None) if ctx is not None else None,
                stream=stream,
            )
            raise

    @asynccontextmanager
    async def _error_context_async(
        self,
        operation: str,
        *,
        stream: bool,
        messages_count: int,
        model: str,
        ctx: Optional[OperationContext],
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
                framework="llamaindex",
                operation=operation,
                messages_count=messages_count,
                model=model,
                temperature=params.get("temperature"),
                max_tokens=params.get("max_tokens"),
                top_p=params.get("top_p"),
                frequency_penalty=params.get("frequency_penalty"),
                presence_penalty=params.get("presence_penalty"),
                request_id=getattr(ctx, "request_id", None) if ctx is not None else None,
                tenant=getattr(ctx, "tenant", None) if ctx is not None else None,
                stream=stream,
            )
            raise

    # ------------------------------------------------------------------ #
    # Core async chat API (production-ready with validation)
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
        messages:
            Sequence of LlamaIndex ChatMessage objects.

        kwargs:
            - callback_manager: CallbackManager for context propagation
            - model, temperature, max_tokens, top_p, frequency_penalty,
              presence_penalty, stop/stop_sequences

        Returns
        -------
        ChatResponse
            LlamaIndex chat response with message content and metadata.
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")

        corpus_messages = _translate_messages_to_corpus(messages)
        stop_sequences = _extract_stop_sequences(kwargs)
        ctx = _build_operation_context_from_callbacks(adapter=self, kwargs=kwargs)
        params = _sampling_params_from_kwargs(
            adapter=self,
            stop_sequences=stop_sequences,
            kwargs=kwargs,
        )

        model_for_context = params.get("model", self.model)
        messages_count = len(corpus_messages)

        async with self._error_context_async(
            "achat",
            stream=False,
            messages_count=messages_count,
            model=model_for_context,
            ctx=ctx,
            params=params,
        ):
            result: LLMCompletion = await self._translator.arun_complete(
                messages=corpus_messages,
                op_ctx=ctx,
                params=params,
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

        Uses `LLMTranslator.arun_stream` and yields ChatResponse objects
        incrementally as chunks arrive.

        Parameters
        ----------
        messages:
            Sequence of LlamaIndex ChatMessage objects.

        kwargs:
            - callback_manager: CallbackManager for context propagation
            - model, temperature, max_tokens, top_p, frequency_penalty,
              presence_penalty, stop/stop_sequences

        Returns
        -------
        ChatResponseAsyncGen
            Async generator of ChatResponse objects with incremental content.
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")

        corpus_messages = _translate_messages_to_corpus(messages)
        stop_sequences = _extract_stop_sequences(kwargs)
        ctx = _build_operation_context_from_callbacks(adapter=self, kwargs=kwargs)
        params = _sampling_params_from_kwargs(
            adapter=self,
            stop_sequences=stop_sequences,
            kwargs=kwargs,
        )

        model_for_context = params.get("model", self.model)
        messages_count = len(corpus_messages)

        async with self._error_context_async(
            "astream_chat",
            stream=True,
            messages_count=messages_count,
            model=model_for_context,
            ctx=ctx,
            params=params,
        ):
            async def _gen() -> AsyncIterator[ChatResponse]:
                async for chunk in self._translator.arun_stream(
                    messages=corpus_messages,
                    op_ctx=ctx,
                    params=params,
                ):
                    yield _build_chat_response_from_chunk(chunk)

            return _gen()

    # ------------------------------------------------------------------ #
    # Sync chat API (translator does async→sync bridging)
    # ------------------------------------------------------------------ #

    def chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        """
        Sync chat.

        Uses the synchronous `LLMTranslator.complete` path.

        Parameters
        ----------
        messages:
            Sequence of LlamaIndex ChatMessage objects.

        kwargs:
            - callback_manager: CallbackManager for context propagation
            - model, temperature, max_tokens, top_p, frequency_penalty,
              presence_penalty, stop/stop_sequences

        Returns
        -------
        ChatResponse
            LlamaIndex chat response with message content and metadata.
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")

        corpus_messages = _translate_messages_to_corpus(messages)
        stop_sequences = _extract_stop_sequences(kwargs)
        ctx = _build_operation_context_from_callbacks(adapter=self, kwargs=kwargs)
        params = _sampling_params_from_kwargs(
            adapter=self,
            stop_sequences=stop_sequences,
            kwargs=kwargs,
        )

        model_for_context = params.get("model", self.model)
        messages_count = len(corpus_messages)

        with self._error_context(
            "chat",
            stream=False,
            messages_count=messages_count,
            model=model_for_context,
            ctx=ctx,
            params=params,
        ):
            result: LLMCompletion = self._translator.complete(
                messages=corpus_messages,
                op_ctx=ctx,
                params=params,
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
        Sync streaming chat.

        Uses the synchronous `LLMTranslator.stream` path, which is
        responsible for any async→sync bridging required by the
        underlying protocol implementation.

        Parameters
        ----------
        messages:
            Sequence of LlamaIndex ChatMessage objects.

        kwargs:
            - callback_manager: CallbackManager for context propagation
            - model, temperature, max_tokens, top_p, frequency_penalty,
              presence_penalty, stop/stop_sequences

        Returns
        -------
        ChatResponseGen
            Generator of ChatResponse objects with incremental content.
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")

        corpus_messages = _translate_messages_to_corpus(messages)
        stop_sequences = _extract_stop_sequences(kwargs)
        ctx = _build_operation_context_from_callbacks(adapter=self, kwargs=kwargs)
        params = _sampling_params_from_kwargs(
            adapter=self,
            stop_sequences=stop_sequences,
            kwargs=kwargs,
        )

        model_for_context = params.get("model", self.model)
        messages_count = len(corpus_messages)

        with self._error_context(
            "stream_chat",
            stream=True,
            messages_count=messages_count,
            model=model_for_context,
            ctx=ctx,
            params=params,
        ):
            def _gen() -> Iterator[ChatResponse]:
                for chunk in self._translator.stream(
                    messages=corpus_messages,
                    op_ctx=ctx,
                    params=params,
                ):
                    yield _build_chat_response_from_chunk(chunk)

            return _gen()

    # ------------------------------------------------------------------ #
    # Token counting (robust with multiple fallbacks)
    # ------------------------------------------------------------------ #

    def count_tokens(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> int:
        """
        Token counting helper.

        Strategy:
        1. Try translator's count_tokens (handles async bridging)
        2. Try protocol adapter's count_tokens
        3. Fall back to improved character-based estimate

        Parameters
        ----------
        messages:
            Sequence of LlamaIndex ChatMessage objects.

        Returns
        -------
        int
            Estimated token count for the messages.

        Note
        ----
        The fallback heuristic is intentionally simple: callers who need
        precise accounting should rely on the adapter's or translator's
        own `count_tokens` implementation.
        """
        if not messages:
            return 0

        corpus_messages = _translate_messages_to_corpus(messages)
        ctx = _build_operation_context_from_callbacks(adapter=self, kwargs=kwargs)
        params: Dict[str, Any] = {"model": kwargs.get("model", self.model)}

        translator = self._translator

        # 1. Try translator-level counting (handles async bridging)
        if hasattr(translator, "count_tokens"):
            try:
                tokens = translator.count_tokens(
                    messages=corpus_messages,
                    op_ctx=ctx,
                    params=params,
                )
                return int(tokens)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "translator.count_tokens failed in CorpusLlamaIndexLLM, "
                    "falling back to heuristic: %s",
                    exc,
                    extra={"framework": "llamaindex"},
                )

        # 2. Try protocol adapter counting
        if hasattr(self.llm_adapter, "count_tokens"):
            try:
                # Build combined text for counting
                combined_text = self._combine_messages_for_counting(messages)
                tokens = self.llm_adapter.count_tokens(
                    text=combined_text,
                    model=params["model"],
                    ctx=ctx,
                )
                return int(tokens)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Protocol count_tokens failed: %s", exc)

        # 3. Improved character-based estimate
        return self._estimate_tokens_from_messages(messages)

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
            
        # More sophisticated estimation: 
        # - 4 chars per token for English text
        # - Minimum 1 token per message
        char_count = len(combined_text)
        message_count = len(messages)
        
        # Balance between character-based and message-based estimation
        char_based = max(1, char_count // 4)
        message_based = max(1, message_count)
        
        return max(char_based, message_based)


__all__ = [
    "LlamaIndexLLMProtocol",
    "CorpusLlamaIndexLLM",
]
