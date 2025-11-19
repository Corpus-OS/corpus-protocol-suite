# corpus_sdk/llm/framework_adapters/llamaindex.py
# SPDX-License-Identifier: Apache-2.0

"""
LlamaIndex adapter for Corpus LLM protocol.

This module exposes a Corpus `BaseLLMAdapter` as a LlamaIndex `LLM`
implementation, with:

- Async + sync chat generation
- Async + sync streaming chat (true incremental streaming)
- Context propagation via `OperationContext`
- Protocol-first design: LlamaIndex is a thin skin over Corpus
- Optional transient error retry with exponential backoff
- Configurable streaming bridge for testing and customization

Design goals
------------

1. Protocol-first:
   LlamaIndex is just an integration surface. All real behavior flows
   through `BaseLLMAdapter` and the LLM protocol in `llm_base.py`.

2. Optional dependency safe:
   Import of LlamaIndex is guarded. If it is not installed, importing
   this module is still safe, but instantiating the adapter will raise
   a clear `RuntimeError`.

3. True streaming:
   - Async streaming uses `BaseLLMAdapter.stream()` directly.
   - Sync streaming uses `SyncStreamBridge` for production-grade
     sync streaming with backpressure, cancellation, and retry.

4. Context + observability:
   - Request/trace IDs, deadlines, tenants, and tags are mapped into
     `OperationContext` using `ContextTranslator.from_llamaindex_callback_manager`.
   - Errors are enriched with framework-specific context via `attach_context`.

5. Cancellation-friendly:
   - Sync streaming optionally accepts a `cancel_event` (threading.Event)
     via kwargs to allow callers to stop consumption early.

This is SDK infrastructure, not business logic.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from corpus_sdk.llm.llm_base import (
    BaseLLMAdapter,
    LLMChunk,
    LLMCompletion,
    OperationContext,
    TransientNetwork,
    Unavailable,
)
from corpus_sdk.llm.framework_adapters.common.async_bridge import AsyncBridge
from corpus_sdk.llm.framework_adapters.common.context_translation import (
    ContextTranslator,
)
from corpus_sdk.llm.framework_adapters.common.error_context import attach_context
from corpus_sdk.llm.framework_adapters.common.message_translation import (
    NormalizedMessage,
    from_llamaindex,
    to_corpus,
)
from corpus_sdk.llm.framework_adapters.common.sync_stream_bridge import SyncStreamBridge

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
# Internal helpers
# ---------------------------------------------------------------------------


def _translate_messages_to_corpus(
    messages: Sequence[Any],
) -> List[Mapping[str, str]]:
    """
    LlamaIndex ChatMessage -> Corpus wire messages.
    """
    normalized: List[NormalizedMessage] = [from_llamaindex(m) for m in messages]
    return to_corpus(normalized)


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
) -> OperationContext:
    """
    Build an OperationContext using the LlamaIndex callback manager.

    Prefer the `callback_manager` passed in kwargs; otherwise fall back
    to the adapter's own callback manager attribute, if present.
    """
    cbm: Optional[Any] = kwargs.get("callback_manager", None)
    if cbm is None:
        cbm = getattr(adapter, "callback_manager", None)

    return ContextTranslator.from_llamaindex_callback_manager(cbm)


def _sampling_params_from_kwargs(
    *,
    adapter: "CorpusLlamaIndexLLM",
    stop_sequences: Optional[List[str]],
    kwargs: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Map LlamaIndex sampling kwargs into Corpus adapter params.
    """
    return {
        "model": kwargs.get("model", adapter.model),
        "max_tokens": kwargs.get("max_tokens", adapter.max_tokens),
        "temperature": kwargs.get("temperature", adapter.temperature),
        "top_p": kwargs.get("top_p"),
        "frequency_penalty": kwargs.get("frequency_penalty"),
        "presence_penalty": kwargs.get("presence_penalty"),
        "stop_sequences": stop_sequences,
    }


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
    LlamaIndex `LLM` implementation backed by a Corpus LLM adapter.

    This class implements the full LlamaIndex LLM interface with
    production-grade streaming, error handling, and configurability.

    Attributes
    ----------
    corpus_adapter:
        Underlying Corpus `BaseLLMAdapter` instance.
    model:
        Default model identifier to send to Corpus.
    temperature:
        Default sampling temperature.
    max_tokens:
        Default max_tokens limit (adapter may have its own default).

    stream_queue_maxsize:
        Max number of pending ChatResponse chunks buffered between the
        background thread and the caller in sync streaming.
    stream_poll_timeout_s:
        Timeout (seconds) when polling the queue in sync streaming;
        allows tuning for high-latency or low-latency workloads.
    stream_join_timeout_s:
        Timeout (seconds) when joining the background streaming thread
        at the end of sync streaming.

    max_transient_retries:
        Number of retry attempts for transient errors during sync streaming
        before the first chunk is emitted. Default: 0 (no retry).
    transient_backoff_s:
        Initial backoff delay (seconds) for streaming retry. Uses exponential
        backoff: attempt N sleeps for `backoff * (2 ** (N - 1))` seconds.
        Default: 0.25.
    stream_transient_error_types:
        Tuple of exception types to consider transient and eligible for retry
        during sync streaming. Default: (TransientNetwork, Unavailable).

    stream_bridge_factory:
        Optional factory function for creating SyncStreamBridge instances.
        Primarily used for testing/mocking. If None, uses default factory.

    Examples
    --------
    Basic usage:

        llm = CorpusLlamaIndexLLM(
            corpus_adapter=OpenAIAdapter(api_key="..."),
            model="gpt-4",
        )

    With retry enabled:

        llm = CorpusLlamaIndexLLM(
            corpus_adapter=OpenAIAdapter(api_key="..."),
            model="gpt-4",
            max_transient_retries=2,
            transient_backoff_s=0.5,
        )

    With custom bridge factory (testing):

        def mock_bridge_factory(**kwargs):
            return MockSyncStreamBridge(**kwargs)

        llm = CorpusLlamaIndexLLM(
            corpus_adapter=mock_adapter,
            stream_bridge_factory=mock_bridge_factory,
        )
    """

    corpus_adapter: BaseLLMAdapter
    model: str = "default"
    temperature: float = 0.7
    max_tokens: Optional[int] = None

    # Sync streaming configuration
    stream_queue_maxsize: int = 16
    stream_poll_timeout_s: float = 0.1
    stream_join_timeout_s: float = 2.0

    # Transient retry configuration for sync streaming
    max_transient_retries: int = 0
    transient_backoff_s: float = 0.25
    stream_transient_error_types: Tuple[Type[BaseException], ...] = (
        TransientNetwork,
        Unavailable,
    )

    # Dependency injection for testing
    stream_bridge_factory: Optional[Callable[..., SyncStreamBridge]] = None

    # Pydantic v2 config
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data: Any) -> None:
        _ensure_llamaindex_installed()
        super().__init__(**data)

    def _create_stream_bridge(
        self,
        *,
        coro_factory: Callable[[], Awaitable[AsyncIterator[ChatResponse]]],
        error_context: Dict[str, Any],
        **stream_overrides: Any,
    ) -> SyncStreamBridge:
        """
        Create a SyncStreamBridge instance with current configuration.

        This method centralizes bridge creation and allows for dependency
        injection via stream_bridge_factory.

        Parameters
        ----------
        coro_factory:
            Factory function that produces an awaitable that returns an
            async iterator of ChatResponse.

        error_context:
            Additional context to attach to any errors raised during streaming.

        stream_overrides:
            Per-call overrides for streaming parameters.

        Returns
        -------
        SyncStreamBridge
            Configured bridge instance ready to run.
        """
        # Determine configuration with overrides
        queue_maxsize = stream_overrides.get("stream_queue_maxsize", self.stream_queue_maxsize)
        poll_timeout_s = stream_overrides.get("stream_poll_timeout_s", self.stream_poll_timeout_s)
        join_timeout_s = stream_overrides.get("stream_join_timeout_s", self.stream_join_timeout_s)
        max_retries = stream_overrides.get("stream_max_transient_retries", self.max_transient_retries)
        backoff_s = stream_overrides.get("stream_transient_backoff_s", self.transient_backoff_s)
        error_types = stream_overrides.get("stream_transient_error_types", self.stream_transient_error_types)
        cancel_event = stream_overrides.get("cancel_event")

        # Use custom factory if provided (for testing), otherwise use default
        if self.stream_bridge_factory is not None:
            return self.stream_bridge_factory(
                coro_factory=coro_factory,
                queue_maxsize=queue_maxsize,
                poll_timeout_s=poll_timeout_s,
                join_timeout_s=join_timeout_s,
                cancel_event=cancel_event,
                framework="llamaindex",
                error_context=error_context,
                max_transient_retries=max_retries,
                transient_backoff_s=backoff_s,
                transient_error_types=error_types,
            )

        return SyncStreamBridge(
            coro_factory=coro_factory,
            queue_maxsize=queue_maxsize,
            poll_timeout_s=poll_timeout_s,
            join_timeout_s=join_timeout_s,
            cancel_event=cancel_event,
            framework="llamaindex",
            error_context=error_context,
            max_transient_retries=max_retries,
            transient_backoff_s=backoff_s,
            transient_error_types=error_types,
        )

    # ------------------------------------------------------------------ #
    # Core async chat API
    # ------------------------------------------------------------------ #

    async def achat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        """
        Async chat entrypoint required by LlamaIndex.
        """
        corpus_messages = _translate_messages_to_corpus(messages)
        stop_sequences = _extract_stop_sequences(kwargs)
        ctx = _build_operation_context_from_callbacks(adapter=self, kwargs=kwargs)
        params = _sampling_params_from_kwargs(
            adapter=self,
            stop_sequences=stop_sequences,
            kwargs=kwargs,
        )

        try:
            result: LLMCompletion = await self.corpus_adapter.complete(
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
        except BaseException as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="llamaindex",
                operation="achat",
                messages_count=len(messages),
                model=params.get("model", self.model),
                temperature=params.get("temperature"),
                max_tokens=params.get("max_tokens"),
                top_p=params.get("top_p"),
                frequency_penalty=params.get("frequency_penalty"),
                presence_penalty=params.get("presence_penalty"),
                request_id=getattr(ctx, "request_id", None),
                tenant=getattr(ctx, "tenant", None),
                stream=False,
            )
            raise

    async def astream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        """
        Async streaming chat entrypoint required by LlamaIndex.

        Uses `BaseLLMAdapter.stream` directly and yields ChatResponse
        objects incrementally as chunks arrive.
        """
        corpus_messages = _translate_messages_to_corpus(messages)
        stop_sequences = _extract_stop_sequences(kwargs)
        ctx = _build_operation_context_from_callbacks(adapter=self, kwargs=kwargs)
        params = _sampling_params_from_kwargs(
            adapter=self,
            stop_sequences=stop_sequences,
            kwargs=kwargs,
        )

        model_for_context = params.get("model", self.model)

        try:
            stream: AsyncIterator[LLMChunk] = await self.corpus_adapter.stream(
                messages=corpus_messages,
                ctx=ctx,
                **params,
            )

            async def _gen() -> AsyncIterator[ChatResponse]:
                try:
                    async for chunk in stream:
                        yield _build_chat_response_from_chunk(chunk)
                finally:
                    aclose = getattr(stream, "aclose", None)
                    if callable(aclose):
                        try:
                            await aclose()
                        except Exception as cleanup_error:  # noqa: BLE001
                            logger.debug(
                                "LlamaIndex stream cleanup failed: %s",
                                cleanup_error,
                                extra={"framework": "llamaindex"},
                            )

            return _gen()

        except BaseException as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="llamaindex",
                operation="astream_chat",
                messages_count=len(messages),
                model=model_for_context,
                temperature=params.get("temperature"),
                max_tokens=params.get("max_tokens"),
                top_p=params.get("top_p"),
                frequency_penalty=params.get("frequency_penalty"),
                presence_penalty=params.get("presence_penalty"),
                request_id=getattr(ctx, "request_id", None),
                tenant=getattr(ctx, "tenant", None),
                stream=True,
            )
            raise

    # ------------------------------------------------------------------ #
    # Sync chat API (bridged via AsyncBridge + SyncStreamBridge)
    # ------------------------------------------------------------------ #

    def chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        """
        Sync chat.

        Uses `AsyncBridge.run_async` to avoid nested event loop issues.
        """
        try:
            return AsyncBridge.run_async(self.achat(messages, **kwargs))
        except BaseException as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="llamaindex",
                operation="chat",
                messages_count=len(messages),
                model=kwargs.get("model", self.model),
                stream=False,
            )
            raise

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseGen:
        """
        Sync streaming chat.

        Bridges `astream_chat` to a blocking generator using SyncStreamBridge
        for production-grade sync streaming with:

        - Background thread management
        - Bounded queue for backpressure
        - Proper cleanup and error propagation
        - Optional transient retry (controlled by max_transient_retries)

        Parameters
        ----------
        messages:
            Sequence of LlamaIndex ChatMessage objects.

        kwargs:
            - cancel_event: Optional[threading.Event] for early cancellation
            - Per-call streaming overrides:
                * stream_queue_maxsize
                * stream_poll_timeout_s
                * stream_join_timeout_s
                * stream_max_transient_retries
                * stream_transient_backoff_s
                * stream_transient_error_types
            - All other parameters as in `achat`.

        Returns
        -------
        ChatResponseGen
            Iterator of ChatResponse chunks.
        """
        cancel_event = kwargs.pop("cancel_event", None)
        if cancel_event is not None and not isinstance(cancel_event, threading.Event):
            logger.warning(
                "stream_chat cancel_event is not a threading.Event; ignoring",
                extra={"framework": "llamaindex"},
            )
            cancel_event = None

        # Extract per-call streaming overrides
        stream_overrides = {
            "cancel_event": cancel_event,
            "stream_queue_maxsize": kwargs.pop("stream_queue_maxsize", self.stream_queue_maxsize),
            "stream_poll_timeout_s": kwargs.pop("stream_poll_timeout_s", self.stream_poll_timeout_s),
            "stream_join_timeout_s": kwargs.pop("stream_join_timeout_s", self.stream_join_timeout_s),
            "stream_max_transient_retries": kwargs.pop(
                "stream_max_transient_retries", self.max_transient_retries
            ),
            "stream_transient_backoff_s": kwargs.pop(
                "stream_transient_backoff_s", self.transient_backoff_s
            ),
            "stream_transient_error_types": kwargs.pop(
                "stream_transient_error_types", self.stream_transient_error_types
            ),
        }

        # Validate error types if overridden
        error_types = stream_overrides["stream_transient_error_types"]
        if error_types is not None and not isinstance(error_types, tuple):
            raise TypeError("stream_transient_error_types must be a tuple of exception types")

        model_for_context = kwargs.get("model", self.model)

        async def _coro_factory() -> AsyncIterator[ChatResponse]:
            # Note: we return the async iterator, not iterate here
            return await self.astream_chat(messages, **kwargs)

        error_context: Dict[str, Any] = {
            "operation": "stream_chat",
            "messages_count": len(messages),
            "model": model_for_context,
            **{k: v for k, v in stream_overrides.items() if k != "cancel_event"},
        }

        bridge = self._create_stream_bridge(
            coro_factory=_coro_factory,
            error_context=error_context,
            **stream_overrides,
        )

        return bridge.run()

    # ------------------------------------------------------------------ #
    # Optional: token counting bridge
    # ------------------------------------------------------------------ #

    async def acount_tokens(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> int:
        """
        Optional async helper that uses Corpus `count_tokens` when available.
        """
        if not hasattr(self.corpus_adapter, "count_tokens"):
            raise NotImplementedError(
                "Underlying Corpus adapter does not support count_tokens()"
            )

        corpus_messages = _translate_messages_to_corpus(messages)
        combined = "\n".join(f"{m['role']}:{m['content']}" for m in corpus_messages)

        ctx = _build_operation_context_from_callbacks(adapter=self, kwargs=kwargs)

        try:
            return await self.corpus_adapter.count_tokens(
                text=combined,
                model=kwargs.get("model", self.model),
                ctx=ctx,
            )
        except BaseException as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="llamaindex",
                operation="acount_tokens",
                messages_count=len(messages),
                model=kwargs.get("model", self.model),
            )
            raise

    def count_tokens(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> int:
        """
        Sync token counting helper.

        Uses async `acount_tokens` when supported. If the Corpus adapter
        does not implement `count_tokens`, falls back to a naive heuristic
        based on character length (~4 chars per token).
        """
        try:
            return AsyncBridge.run_async(self.acount_tokens(messages, **kwargs))
        except (NotImplementedError, AttributeError):
            corpus_messages = _translate_messages_to_corpus(messages)
            total_chars = sum(len(m.get("content", "")) for m in corpus_messages)
            approx_tokens = max(1, total_chars // 4)
            logger.debug(
                "count_tokens not supported by Corpus adapter; "
                "using naive fallback (%s chars -> ~%s tokens)",
                total_chars,
                approx_tokens,
                extra={"framework": "llamaindex"},
            )
            return approx_tokens
        except BaseException as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="llamaindex",
                operation="count_tokens",
                messages_count=len(messages),
                model=kwargs.get("model", self.model),
            )
            raise


__all__ = [
    "CorpusLlamaIndexLLM",
]
