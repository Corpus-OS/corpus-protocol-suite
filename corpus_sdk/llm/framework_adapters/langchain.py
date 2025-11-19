# corpus_sdk/llm/framework_adapters/langchain.py
# SPDX-License-Identifier: Apache-2.0

"""
LangChain adapter for Corpus LLM protocol.

This module exposes Corpus `BaseLLMAdapter` implementations as
`langchain_core` chat models, with:

- Async + sync generation
- Async + sync streaming (true incremental streaming)
- Proper callback integration (on_llm_end, on_llm_new_token)
- Protocol-first design: LangChain is a thin skin over Corpus

Example
-------

    from corpus_sdk.llm.openai_adapter import OpenAIAdapter
    from corpus_sdk.llm.framework_adapters.langchain import CorpusLangChainLLM

    corpus_llm = OpenAIAdapter(api_key="...")
    lc_llm = CorpusLangChainLLM(
        corpus_adapter=corpus_llm,
        model="gpt-4.1",
    )

    # Sync call
    result = lc_llm.invoke("Hello!")
    print(result.content)

    # Async call
    result = await lc_llm.ainvoke("Hello!")

    # Streaming (sync)
    for chunk in lc_llm.stream("Hello!"):
        print(chunk.content, end="", flush=True)

    # Streaming (async)
    async for chunk in lc_llm.astream("Hello!"):
        print(chunk.content, end="", flush=True)
"""

from __future__ import annotations

import logging
import threading
from queue import Queue
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from corpus_sdk.llm.llm_base import (
    BaseLLMAdapter,
    LLMChunk,
    LLMCompletion,
    OperationContext,
)
from corpus_sdk.llm.framework_adapters.common.message_translation import (
    from_langchain,
    to_corpus,
    NormalizedMessage,  # kept for type hints / future use
)
from corpus_sdk.llm.framework_adapters.common.context_translation import (
    from_langchain as context_from_langchain,
)
from corpus_sdk.llm.framework_adapters.common.async_bridge import AsyncBridge

logger = logging.getLogger(__name__)


class CorpusLangChainLLM(BaseChatModel):
    """
    LangChain `BaseChatModel` implementation backed by a Corpus `BaseLLMAdapter`.

    This is a *thin* integration layer:
    - Messages are normalized via `message_translation.from_langchain`
    - Context is derived from LangChain's `RunnableConfig` via
      `context_translation.from_langchain`
    - All policy / resilience (deadlines, breaker, limiter, cache) lives in the
      underlying `BaseLLMAdapter`, not here.

    Attributes
    ----------
    corpus_adapter:
        Underlying Corpus LLM adapter implementing `LLMProtocolV1`.

    model:
        Default model identifier to pass to the adapter (optional override per call).

    temperature:
        Default sampling temperature (can be overridden per invocation).

    max_tokens:
        Default max tokens for generation (can be overridden per invocation).

    stream_queue_maxsize:
        Max queue size for sync streaming bridge. `<= 0` means unbounded.
        This is a backpressure knob between the background async worker and
        the sync consumer.

    stream_thread_join_timeout:
        Timeout (seconds) for waiting on the background streaming thread
        during cleanup. Prevents hangs in pathological cases.
    """

    corpus_adapter: BaseLLMAdapter
    model: str = "default"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream_queue_maxsize: int = 16
    stream_thread_join_timeout: float = 5.0

    # Pydantic v2-style config: allow arbitrary types like BaseLLMAdapter.
    model_config = {"arbitrary_types_allowed": True}

    # ------------------------------------------------------------------ #
    # LangChain-required properties
    # ------------------------------------------------------------------ #

    @property
    def _llm_type(self) -> str:
        """Identifier used by LangChain in serialization / introspection."""
        return "corpus"

    # ------------------------------------------------------------------ #
    # Capabilities / feature helpers
    # ------------------------------------------------------------------ #

    def supports_streaming(self) -> bool:
        """
        Return True if the underlying adapter advertises streaming support.

        If capabilities cannot be fetched for any reason, we default to True,
        since most modern adapters support streaming and the adapter itself
        will raise `NotSupported` if it does not.
        """
        try:
            caps = AsyncBridge.run_async(self.corpus_adapter.capabilities())
            return bool(getattr(caps, "supports_streaming", True))
        except Exception as exc:  # noqa: BLE001
            logger.debug("capabilities() failed when checking supports_streaming: %s", exc)
            return True

    # ------------------------------------------------------------------ #
    # Internal translation helpers
    # ------------------------------------------------------------------ #

    def _translate_messages(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """
        LangChain messages → Corpus wire format ({role, content} dicts).
        """
        normalized: List[NormalizedMessage] = [from_langchain(m) for m in messages]
        # `to_corpus` returns list[Mapping[str, Any]]; cast to Dict for callers.
        return [dict(m) for m in to_corpus(normalized)]

    def _build_context_and_params(
        self,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> tuple[OperationContext, Dict[str, Any]]:
        """
        Extract `OperationContext` and Corpus sampling params from LangChain kwargs.

        Expected kwargs (optional):
            - config: RunnableConfig-like dict (for context translation)
            - model, temperature, max_tokens, top_p, frequency_penalty, presence_penalty

        The returned `params` dict is shaped to match `LLMProtocolV1.complete/stream`.
        """
        config = kwargs.get("config")
        ctx = context_from_langchain(config)

        params: Dict[str, Any] = {
            "model": kwargs.get("model", self.model),
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "top_p": kwargs.get("top_p"),
            "frequency_penalty": kwargs.get("frequency_penalty"),
            "presence_penalty": kwargs.get("presence_penalty"),
            "stop_sequences": stop,
        }

        # Strip None values so the adapter sees a clean param set that matches
        # its signature (no unexpected kwargs).
        params = {k: v for k, v in params.items() if v is not None}
        return ctx, params

    @staticmethod
    def _build_chat_result(completion: LLMCompletion) -> ChatResult:
        """
        Map a Corpus `LLMCompletion` into a LangChain `ChatResult`.
        """
        usage = completion.usage
        usage_dict: Dict[str, int] = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }

        message = AIMessage(
            content=completion.text,
            additional_kwargs={
                "model": completion.model,
                "model_family": completion.model_family,
                "finish_reason": completion.finish_reason,
            },
        )

        generation_info: Dict[str, Any] = {
            "model": completion.model,
            "model_family": completion.model_family,
            "finish_reason": completion.finish_reason,
            "usage": usage_dict,
        }

        generation = ChatGeneration(
            message=message,
            generation_info=generation_info,
        )
        return ChatResult(generations=[generation])

    @staticmethod
    def _chunk_to_generation_chunk(chunk: LLMChunk) -> ChatGenerationChunk:
        """
        Map a Corpus `LLMChunk` into a LangChain `ChatGenerationChunk`.
        """
        text = chunk.text or ""
        ai_chunk = AIMessageChunk(content=text)

        gen_info: Dict[str, Any] = {
            "model": chunk.model,
            "is_final": chunk.is_final,
        }
        if chunk.usage_so_far is not None:
            gen_info["usage_so_far"] = {
                "prompt_tokens": chunk.usage_so_far.prompt_tokens,
                "completion_tokens": chunk.usage_so_far.completion_tokens,
                "total_tokens": chunk.usage_so_far.total_tokens,
            }

        return ChatGenerationChunk(
            message=ai_chunk,
            generation_info=gen_info,
        )

    # ------------------------------------------------------------------ #
    # Core async → Corpus calls
    # ------------------------------------------------------------------ #

    async def _acall_corpus_complete(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Core async path: LC messages → Corpus `complete()` → `ChatResult`.
        """
        corpus_messages = self._translate_messages(messages)
        ctx, params = self._build_context_and_params(stop=stop, **kwargs)

        completion = await self.corpus_adapter.complete(
            messages=corpus_messages,
            ctx=ctx,
            **params,
        )
        return self._build_chat_result(completion)

    async def _acall_corpus_stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """
        Core async streaming path: LC messages → Corpus `stream()` → `ChatGenerationChunk`.
        """
        corpus_messages = self._translate_messages(messages)
        ctx, params = self._build_context_and_params(stop=stop, **kwargs)

        # `BaseLLMAdapter.stream` is async and returns an AsyncIterator[LLMChunk].
        stream = await self.corpus_adapter.stream(
            messages=corpus_messages,
            ctx=ctx,
            **params,
        )

        async for chunk in stream:
            yield self._chunk_to_generation_chunk(chunk)

    # ------------------------------------------------------------------ #
    # LangChain async API
    # ------------------------------------------------------------------ #

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Async chat generation entrypoint used by LangChain.
        """
        try:
            result = await self._acall_corpus_complete(messages, stop=stop, **kwargs)
            if run_manager is not None:
                await run_manager.on_llm_end(result)
            return result
        except Exception as exc:  # noqa: BLE001
            if run_manager is not None:
                await run_manager.on_llm_error(exc)
            raise

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """
        Async streaming entrypoint used by LangChain.

        Emits `ChatGenerationChunk` instances incrementally and forwards
        tokens to LangChain callbacks via `on_llm_new_token`.
        """
        try:
            async for gen_chunk in self._acall_corpus_stream(messages, stop=stop, **kwargs):
                text = gen_chunk.message.content or ""
                if run_manager is not None and text:
                    await run_manager.on_llm_new_token(text)
                yield gen_chunk

            # LangChain doesn't strictly require `on_llm_end` here, but it is
            # useful for tools that rely on a well-formed lifecycle.
            if run_manager is not None:
                empty_result = ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(content=""),
                            generation_info={"streaming": True},
                        )
                    ]
                )
                await run_manager.on_llm_end(empty_result)

        except Exception as exc:  # noqa: BLE001
            if run_manager is not None:
                await run_manager.on_llm_error(exc)
            raise

    # ------------------------------------------------------------------ #
    # LangChain sync API
    # ------------------------------------------------------------------ #

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Sync chat generation entrypoint used by LangChain.

        Uses `AsyncBridge` to run the async Corpus call under the hood.
        """
        try:
            result = AsyncBridge.run_async(
                self._acall_corpus_complete(messages, stop=stop, **kwargs)
            )
            if run_manager is not None:
                run_manager.on_llm_end(result)
            return result
        except Exception as exc:  # noqa: BLE001
            if run_manager is not None:
                run_manager.on_llm_error(exc)
            raise

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """
        Sync streaming entrypoint used by LangChain.

        Bridges the async streaming iterator via a background thread and a Queue,
        yielding `ChatGenerationChunk` objects as they arrive without buffering the
        entire response.

        Best-effort cancellation:
            - If the caller stops consuming early (breaks the iterator),
              we signal the worker thread via a `cancel_event`.
            - The worker checks this flag and attempts to close the underlying
              async generator (if it exposes `aclose()`).
        """
        # Queue between worker and main thread. maxsize <= 0 ⇒ unbounded.
        maxsize = self.stream_queue_maxsize if self.stream_queue_maxsize > 0 else 0
        queue: "Queue[Optional[ChatGenerationChunk]]" = Queue(maxsize=maxsize)

        error_holder: List[BaseException] = []
        done_event = threading.Event()
        cancel_event = threading.Event()
        finished_normally = False

        def worker() -> None:
            """
            Background thread that runs the async Corpus stream and enqueues chunks.
            """

            async def run_stream() -> None:
                corpus_messages = self._translate_messages(messages)
                ctx, params = self._build_context_and_params(stop=stop, **kwargs)

                agen = await self.corpus_adapter.stream(
                    messages=corpus_messages,
                    ctx=ctx,
                    **params,
                )

                try:
                    async for chunk in agen:
                        if cancel_event.is_set():
                            # Best-effort shutdown: ask the async generator to close
                            # itself so the adapter can tear down any upstream stream.
                            close = getattr(agen, "aclose", None)
                            if callable(close):
                                try:
                                    await close()
                                except Exception as close_exc:  # noqa: BLE001
                                    logger.debug(
                                        "Error closing Corpus stream after cancellation: %s",
                                        close_exc,
                                    )
                            break

                        queue.put(CorpusLangChainLLM._chunk_to_generation_chunk(chunk))
                finally:
                    # Always signal completion to the consumer.
                    queue.put(None)

            try:
                AsyncBridge.run_async(run_stream())
            except BaseException as exc:  # noqa: BLE001
                error_holder.append(exc)
                # Ensure sentinel is in the queue even on error.
                queue.put(None)
            finally:
                done_event.set()

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        try:
            while True:
                chunk = queue.get()
                if chunk is None:
                    finished_normally = True
                    break

                if run_manager is not None and chunk.message.content:
                    run_manager.on_llm_new_token(chunk.message.content)

                yield chunk

            # Wait a bit for the worker thread to finish cleanup.
            thread.join(timeout=self.stream_thread_join_timeout)
            if thread.is_alive():
                logger.warning(
                    "CorpusLangChainLLM streaming worker thread did not exit "
                    "within %.2fs; continuing anyway",
                    self.stream_thread_join_timeout,
                )

            if error_holder:
                exc = error_holder[0]
                if run_manager is not None:
                    run_manager.on_llm_error(exc)
                raise exc

            if run_manager is not None:
                empty_result = ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(content=""),
                            generation_info={"streaming": True},
                        )
                    ]
                )
                run_manager.on_llm_end(empty_result)

        except Exception as exc:  # noqa: BLE001
            # Caller stopped early or error occurred on the main thread.
            if not finished_normally:
                cancel_event.set()
                # Give worker a short window to react to cancellation.
                done_event.wait(timeout=self.stream_thread_join_timeout)

            if run_manager is not None:
                run_manager.on_llm_error(exc)
            raise

    # ------------------------------------------------------------------ #
    # Optional: token counting / estimation
    # ------------------------------------------------------------------ #

    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        """
        Estimate token count for a list of LangChain messages.

        Behavior:
        - If the underlying adapter's `count_tokens` succeeds, use that.
        - On any error, fall back to a naive character-based estimate.

        NOTE:
            The fallback heuristic (chars ÷ 4) is intentionally simple: callers
            who need precise accounting should rely on the adapter's
            `count_tokens` implementation.
        """
        # Build a combined text similar to BaseLLMAdapter's preflight behavior.
        parts: List[str] = []
        for m in messages:
            role = getattr(m, "type", None) or getattr(m, "role", "user")
            parts.append(f"{role}:{str(m.content)}")
        text = "\n".join(parts)

        try:
            tokens = AsyncBridge.run_async(
                self.corpus_adapter.count_tokens(
                    text=text,
                    model=self.model,
                    ctx=None,  # no explicit OperationContext from LC here
                )
            )
            return int(tokens)
        except Exception as exc:  # noqa: BLE001
            logger.debug("count_tokens failed in CorpusLangChainLLM, using fallback: %s", exc)
            # Very rough heuristic: ~4 characters per token.
            if not text:
                return 0
            return max(1, len(text) // 4)


__all__ = ["CorpusLangChainLLM"]
