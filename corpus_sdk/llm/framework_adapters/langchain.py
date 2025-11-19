# corpus_sdk/llm/framework_adapters/langchain.py
# SPDX-License-Identifier: Apache-2.0

"""
LangChain adapter for Corpus LLM protocol.

This module exposes Corpus `BaseLLMAdapter` implementations as
`langchain_core` chat models, with:

- Async + sync generation
- Async + sync streaming (true incremental streaming)
- Proper callback integration (on_llm_end, on_llm_new_token for streaming)
- Protocol-first design: LangChain is a thin skin over Corpus

Design goals
------------

1. Protocol-first:
   LangChain is an integration surface, not the source of truth.
   All real behavior flows through `BaseLLMAdapter`.

2. Non-invasive:
   No hacks into LangChain internals. We implement the official
   `BaseChatModel` contract.

3. True streaming:
   Async streaming uses Corpus' `stream()` directly.
   Sync streaming uses a background thread to bridge the async iterator
   without buffering the whole response.

4. Safe defaults with escape hatches:
   - `strict_roles` lets you opt into stricter role validation.
   - Timeouts can be applied via the underlying Corpus adapter or via
     the `AsyncBridge` layer if needed.
"""

from __future__ import annotations

import threading
from queue import Queue
from typing import Any, AsyncIterator, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, TypeVar, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
    Callbacks,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
    LLMResult,
)

from corpus_sdk.llm.llm_base import BaseLLMAdapter, OperationContext
from corpus_sdk.llm.framework_adapters.common.message_translation import (
    MessageTranslator,
    NormalizedMessage,
)
from corpus_sdk.llm.framework_adapters.common.context_translation import ContextTranslator
from corpus_sdk.llm.framework_adapters.common.async_bridge import AsyncBridge

T = TypeVar("T")


class CorpusLangChainLLM(BaseChatModel):
    """
    LangChain `BaseChatModel` implementation backed by a Corpus LLM adapter.

    Typical usage
    -------------

        from corpus_sdk.llm.openai_adapter import OpenAIAdapter
        from corpus_sdk.llm.framework_adapters.langchain import CorpusLangChainLLM

        corpus_llm = OpenAIAdapter(api_key="...")
        lc_llm = CorpusLangChainLLM.from_corpus_adapter(
            corpus_adapter=corpus_llm,
            model="gpt-4o",
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

    Configuration knobs
    -------------------
    - `model`: default model name for Corpus.
    - `temperature`, `max_tokens`: default sampling params.
    - `strict_roles`: if True, LangChain → Corpus role mapping is stricter
      (MessageTranslator enforces supported roles and may raise).
    """

    corpus_adapter: BaseLLMAdapter
    model: str = "default"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    strict_roles: Optional[bool] = None  # None → MessageTranslator decides (e.g., via env)

    class Config:
        arbitrary_types_allowed = True

    # ------------------------------------------------------------------ #
    # LangChain-required properties
    # ------------------------------------------------------------------ #

    @property
    def _llm_type(self) -> str:
        return "corpus"

    @property
    def lc_serializable(self) -> bool:
        # We don’t guarantee that `corpus_adapter` is serializable (often it isn’t).
        return False

    # ------------------------------------------------------------------ #
    # Convenience constructor / capabilities helpers
    # ------------------------------------------------------------------ #

    @classmethod
    def from_corpus_adapter(
        cls,
        corpus_adapter: BaseLLMAdapter,
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        strict_roles: Optional[bool] = None,
    ) -> "CorpusLangChainLLM":
        """
        Convenience constructor to build an LC model from a Corpus adapter.
        """
        return cls(
            corpus_adapter=corpus_adapter,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            strict_roles=strict_roles,
        )

    async def _adapter_capabilities(self) -> Any:
        """
        Fetch capabilities from the underlying Corpus adapter.

        Adapters are free to cache this internally; this method is only used
        for introspection helpers.
        """
        return await self.corpus_adapter.capabilities()

    def supports_streaming(self) -> bool:
        """
        Returns True if the underlying Corpus adapter supports streaming.
        """
        caps = AsyncBridge.run_async(self._adapter_capabilities())
        return bool(getattr(caps, "supports_streaming", True))

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _normalize_messages(self, messages: List[BaseMessage]) -> List[NormalizedMessage]:
        """
        LangChain BaseMessage → NormalizedMessage (protocol intermediate).
        """
        return [
            MessageTranslator.from_langchain(m, strict_roles=self.strict_roles)
            for m in messages
        ]

    def _build_corpus_messages(self, normalized: List[NormalizedMessage]) -> List[Dict[str, Any]]:
        """
        NormalizedMessage → Corpus protocol wire format.
        """
        return MessageTranslator.to_corpus(normalized)

    def _extract_ctx_and_params(
        self,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Tuple[OperationContext, Dict[str, Any]]:
        """
        Extract Corpus OperationContext and sampling params from LC kwargs.
        """
        # LangChain passes RunnableConfig in `config` kwarg (if present).
        config = kwargs.get("config")
        ctx = ContextTranslator.from_langchain_config(config)

        # Allow overrides via kwargs; otherwise use adapter defaults / class defaults.
        params: Dict[str, Any] = {
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p"),
            "frequency_penalty": kwargs.get("frequency_penalty"),
            "presence_penalty": kwargs.get("presence_penalty"),
            "stop_sequences": stop,
            "model": kwargs.get("model", self.model),
        }

        # Remove None values so Corpus adapter sees clean param set.
        params = {k: v for k, v in params.items() if v is not None}
        return ctx, params

    def _build_chat_result(
        self,
        text: str,
        model: Optional[str] = None,
        finish_reason: Optional[str] = None,
        usage: Optional[Any] = None,
    ) -> ChatResult:
        """
        Build LangChain ChatResult from Corpus completion response.
        """
        message = AIMessage(
            content=text,
            additional_kwargs={
                "model": model,
                "finish_reason": finish_reason,
            },
        )

        generation_info: Dict[str, Any] = {
            "finish_reason": finish_reason,
        }
        if usage is not None:
            generation_info["usage"] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }

        generation = ChatGeneration(
            message=message,
            generation_info=generation_info,
        )
        return ChatResult(generations=[generation])

    # ------------------------------------------------------------------ #
    # Core async generation
    # ------------------------------------------------------------------ #

    async def _acall_corpus_complete(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Core async path: LangChain messages → Corpus.complete() → ChatResult.
        """
        normalized = self._normalize_messages(messages)
        corpus_messages = self._build_corpus_messages(normalized)
        ctx, params = self._extract_ctx_and_params(stop=stop, **kwargs)

        result = await self.corpus_adapter.complete(
            messages=corpus_messages,
            ctx=ctx,
            **params,
        )

        return self._build_chat_result(
            text=result.text,
            model=getattr(result, "model", None),
            finish_reason=getattr(result, "finish_reason", None),
            usage=getattr(result, "usage", None),
        )

    async def _acall_corpus_stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """
        Core async streaming path: LangChain messages → Corpus.stream() → ChatGenerationChunk.
        """
        normalized = self._normalize_messages(messages)
        corpus_messages = self._build_corpus_messages(normalized)
        ctx, params = self._extract_ctx_and_params(stop=stop, **kwargs)

        stream = await self.corpus_adapter.stream(
            messages=corpus_messages,
            ctx=ctx,
            **params,
        )

        # Protocol: async generator yielding Corpus chunks with `.text`, `.model`, etc.
        try:
            async for chunk in stream:
                text = getattr(chunk, "text", "") or ""
                model = getattr(chunk, "model", None)

                ai_chunk = AIMessageChunk(content=text)
                gen_chunk = ChatGenerationChunk(message=ai_chunk, generation_info={"model": model})
                yield gen_chunk
        finally:
            # Best-effort cleanup for adapters that expose aclose().
            agen = getattr(stream, "aclose", None)
            if callable(agen):
                try:
                    await agen()
                except Exception:
                    # Don’t mask upstream errors with cleanup failures.
                    pass

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
        result = await self._acall_corpus_complete(messages, stop=stop, **kwargs)

        # LC already called on_llm_start; we just need to signal end.
        if run_manager is not None:
            await run_manager.on_llm_end(result)

        return result

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
        async for gen_chunk in self._acall_corpus_stream(messages, stop=stop, **kwargs):
            if run_manager is not None:
                # Minimal callback: new token.
                await run_manager.on_llm_new_token(gen_chunk.message.content or "")
            yield gen_chunk

        # Build a synthetic ChatResult for on_llm_end (optional but useful).
        if run_manager is not None:
            # We don't have the full text here without buffering.
            # LangChain doesn't strictly require a full ChatResult in streaming,
            # but we can emit an empty result as a termination signal.
            empty_result = ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(content=""),
                        generation_info={"streaming": True},
                    )
                ]
            )
            await run_manager.on_llm_end(empty_result)

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

        Bridges to `_acall_corpus_complete` using AsyncBridge.
        """
        result = AsyncBridge.run_async(
            self._acall_corpus_complete(messages, stop=stop, **kwargs)
        )

        if run_manager is not None:
            run_manager.on_llm_end(result)

        return result

    def stream(
        self,
        input: Union[str, BaseMessage, List[BaseMessage]],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """
        Sync streaming interface; returns an iterator of ChatGenerationChunk.

        This is a true streaming bridge: it uses a background thread and
        an async iterator under the hood, but yields chunks to the caller
        as they arrive, without buffering the full response.
        """
        # Normalize input to a list of messages using BaseChatModel helper.
        messages: List[BaseMessage] = self._convert_input_to_messages(input)

        # Configure callback manager for this run.
        callback_manager = self.callback_manager.configure(callbacks, self)
        run_manager = callback_manager.on_llm_start(
            {"name": self.__class__.__name__},
            [messages],
            invocation_params=kwargs,
        )

        # Queue and sentinel for cross-thread streaming.
        q: "Queue[Optional[ChatGenerationChunk]]" = Queue()
        error_holder: List[BaseException] = []

        def _worker() -> None:
            """
            Background thread: runs the async streaming coroutine and
            pushes chunks into the queue.
            """

            async def _run_stream() -> None:
                async for gen_chunk in self._astream(
                    messages, stop=stop, run_manager=None, **kwargs
                ):
                    # Forward tokens via sync run_manager.
                    if run_manager is not None:
                        run_manager.on_llm_new_token(gen_chunk.message.content or "")
                    q.put(gen_chunk)

            try:
                AsyncBridge.run_async(_run_stream())
            except BaseException as exc:  # noqa: BLE001
                error_holder.append(exc)
            finally:
                # Sentinel to tell the main thread we're done.
                q.put(None)

        # Start background streaming thread.
        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        # Consume chunks as they arrive.
        while True:
            item = q.get()
            if item is None:
                break
            yield item

        # Propagate any error from the background thread.
        if error_holder:
            exc = error_holder[0]
            # Make sure callbacks see the error as an end-of-run event.
            run_manager.on_llm_error(exc) if run_manager is not None else None
            raise exc

        # Final callback signaling end-of-run.
        if run_manager is not None:
            # As in async streaming, we don't rebuild full text here.
            empty_result = ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(content=""),
                        generation_info={"streaming": True},
                    )
                ]
            )
            run_manager.on_llm_end(empty_result)

    # ------------------------------------------------------------------ #
    # Optional: token counting / LLMResult compatibility
    # ------------------------------------------------------------------ #

    async def _acount_tokens(self, messages: List[BaseMessage], **kwargs: Any) -> int:
        """
        Optional: use Corpus `count_tokens` if available.
        """
        if not hasattr(self.corpus_adapter, "count_tokens"):
            raise NotImplementedError("Underlying Corpus adapter does not support count_tokens()")

        normalized = self._normalize_messages(messages)
        corpus_messages = self._build_corpus_messages(normalized)
        ctx, params = self._extract_ctx_and_params(**kwargs)

        result = await self.corpus_adapter.count_tokens(
            messages=corpus_messages,
            ctx=ctx,
            model=params.get("model", self.model),
        )
        return int(getattr(result, "total_tokens", 0))

    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        """
        Optional LangChain helper for token estimation.

        This uses the Corpus `count_tokens` API when available; otherwise,
        it falls back to naive estimation (length of concatenated content).
        """
        try:
            return AsyncBridge.run_async(self._acount_tokens(messages))
        except (NotImplementedError, AttributeError):
            # Naive fallback: total character length as a rough proxy.
            return sum(len(str(m.content)) for m in messages)
