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
   - Sync streaming uses a background thread + queue to bridge the
     async iterator to a blocking generator without buffering the
     entire response.

4. Context + observability:
   - Request/trace IDs, deadlines, tenants, and tags are mapped into
     `OperationContext` using `ContextTranslator.from_llamaindex_callback_manager`.
   - Errors are propagated as-is; we do not swallow provider errors.

5. Cancellation-friendly:
   - Sync streaming optionally accepts a `cancel_event` (threading.Event)
     via kwargs to allow callers to stop consumption early without
     hanging on teardown.

This is SDK infrastructure, not business logic.
"""

from __future__ import annotations

import logging
import threading
from queue import Queue, Empty
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
)

from corpus_sdk.llm.llm_base import (
    BaseLLMAdapter,
    LLMChunk,
    LLMCompletion,
    OperationContext,
)
from corpus_sdk.llm.framework_adapters.common.message_translation import (
    NormalizedMessage,
    from_llamaindex,
    to_corpus,
)
from corpus_sdk.llm.framework_adapters.common.context_translation import (
    ContextTranslator,
)
from corpus_sdk.llm.framework_adapters.common.async_bridge import AsyncBridge

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
    if isinstance(stop, Iterable):
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
    stream_thread_join_timeout_s:
        Timeout (seconds) when joining the background streaming thread
        at the end of sync streaming.
    stream_poll_timeout_s:
        Timeout (seconds) when polling the queue in sync streaming;
        allows tuning for high-latency or low-latency workloads.
    """

    corpus_adapter: BaseLLMAdapter
    model: str = "default"
    temperature: float = 0.7
    max_tokens: Optional[int] = None

    # Sync streaming tuning knobs
    stream_queue_maxsize: int = 16
    stream_thread_join_timeout_s: float = 2.0
    stream_poll_timeout_s: float = 0.1

    # Pydantic v2 config
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data: Any) -> None:  # pragma: no cover - thin wrapper
        _ensure_llamaindex_installed()
        super().__init__(**data)

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

    # ------------------------------------------------------------------ #
    # Sync chat API (bridged via AsyncBridge)
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
            logger.exception(
                "CorpusLlamaIndexLLM.chat failed",
                exc_info=exc,
                extra={"framework": "llamaindex"},
            )
            raise

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseGen:
        """
        Sync streaming chat.

        Bridges `astream_chat` to a blocking generator using a background
        thread + queue:

        - Background thread runs the async streaming coroutine.
        - Chunks are pushed into a bounded queue.
        - The main thread yields chunks as they arrive.
        - A sentinel (None) signals completion.
        - Any exception in the worker is re-raised in the main thread.

        Cancellation
        ------------
        Accepts an optional `cancel_event` (threading.Event) via kwargs:

            cancel = threading.Event()
            for chunk in llm.stream_chat(messages, cancel_event=cancel):
                ...
                if should_stop:
                    cancel.set()

        Cancellation is best-effort:
        - The main thread stops yielding further chunks as soon as
          `cancel_event.is_set()` is observed.
        - The worker thread is allowed to finish naturally; it is joined
          with a bounded timeout to avoid hanging the caller.
        """
        cancel_event = kwargs.pop("cancel_event", None)
        if cancel_event is not None and not isinstance(cancel_event, threading.Event):
            logger.warning(
                "stream_chat cancel_event is not a threading.Event; ignoring",
                extra={"framework": "llamaindex"},
            )
            cancel_event = None

        q: "Queue[Optional[ChatResponse]]" = Queue(maxsize=self.stream_queue_maxsize)
        error_holder: List[BaseException] = []
        done_event = threading.Event()

        def _worker() -> None:
            """
            Background thread that runs `astream_chat` and enqueues
            ChatResponse chunks.
            """

            async def _run_stream() -> None:
                async for resp in await self.astream_chat(messages, **kwargs):
                    q.put(resp)

            try:
                AsyncBridge.run_async(_run_stream())
            except BaseException as exc:  # noqa: BLE001
                # Capture the error for the main thread; avoid double-logging.
                error_holder.append(exc)
            finally:
                q.put(None)
                done_event.set()

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        try:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    logger.debug(
                        "stream_chat cancellation requested; stopping consumption",
                        extra={"framework": "llamaindex"},
                    )
                    break

                try:
                    item = q.get(timeout=self.stream_poll_timeout_s)
                except Empty:
                    if done_event.is_set() and q.empty():
                        break
                    continue

                if item is None:
                    break

                yield item

            thread.join(timeout=self.stream_thread_join_timeout_s)
            if thread.is_alive():  # pragma: no cover - defensive
                logger.debug(
                    "LlamaIndex streaming worker thread did not terminate within %.2fs",
                    self.stream_thread_join_timeout_s,
                    extra={"framework": "llamaindex"},
                )

            if error_holder:
                raise error_holder[0]

        except BaseException as exc:  # noqa: BLE001
            logger.exception(
                "CorpusLlamaIndexLLM.stream_chat failed",
                exc_info=exc,
                extra={"framework": "llamaindex"},
            )
            raise

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

        return await self.corpus_adapter.count_tokens(
            text=combined,
            model=kwargs.get("model", self.model),
            ctx=ctx,
        )

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
            total_chars = sum(len(m["content"]) for m in corpus_messages)
            approx_tokens = total_chars // 4
            logger.debug(
                "count_tokens not supported by Corpus adapter; "
                "using naive fallback (%s chars -> ~%s tokens)",
                total_chars,
                approx_tokens,
                extra={"framework": "llamaindex"},
            )
            return approx_tokens


__all__ = [
    "CorpusLlamaIndexLLM",
]
