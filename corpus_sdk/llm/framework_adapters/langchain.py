# corpus_sdk/llm/framework_adapters/langchain.py
# SPDX-License-Identifier: Apache-2.0

"""
LangChain adapter for Corpus LLM protocol.

This module exposes Corpus `LLMProtocolV1` implementations as
`langchain_core` chat models, with:

- Async + sync generation
- Async + sync streaming (true incremental streaming)
- Proper callback integration (on_llm_end, on_llm_new_token, on_llm_error)
- Protocol-first design: Direct LLMProtocolV1 access with message translation
- Production-grade error handling and observability

Design principles
-----------------
- Direct protocol access: No unnecessary abstraction layers
- Framework-native: Full LangChain callback and streaming support  
- Observable: Rich error context and comprehensive logging
- Robust: Production-ready with proper resource management
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Protocol

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from corpus_sdk.core.context_translation import (
    from_langchain as context_from_langchain,
)
from corpus_sdk.core.error_context import attach_context
from corpus_sdk.llm.framework_adapters.common.message_translation import (
    NormalizedMessage,
    from_langchain,
    to_corpus,
)
from corpus_sdk.llm.llm_base import (
    LLMChunk,
    LLMCompletion,
    LLMProtocolV1,
    OperationContext,
)

logger = logging.getLogger(__name__)

# Framework constants for consistency
_FRAMEWORK_NAME = "langchain"


class LangChainLLMProtocol(Protocol):
    """
    Structural protocol for LangChain-compatible Corpus chat models.
    """

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        ...

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        ...

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        ...

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        ...


class CorpusLangChainLLM(BaseChatModel):
    """
    LangChain `BaseChatModel` implementation backed by a Corpus `LLMProtocolV1`.

    This is a *thin* integration layer:

    - Messages are normalized via `message_translation.from_langchain`.
    - Context is derived from LangChain's `RunnableConfig` via
      `context_translation.from_langchain`.
    - All LLM calls go directly to the underlying protocol implementation.

    Usage:
        llm = CorpusLangChainLLM(
            llm_adapter=adapter,
            model="gpt-4",
            temperature=0.7
        )
        result = llm.invoke([HumanMessage(content="Hello!")])
    """

    # Underlying protocol implementation
    llm_adapter: LLMProtocolV1

    # Defaults for sampling with validation
    model: str = "default"
    temperature: float = 0.7
    max_tokens: Optional[int] = None

    # Pydantic v2-style config
    model_config = {
        "arbitrary_types_allowed": True,
        "protected_namespaces": (),
    }

    def __init__(
        self,
        *,
        llm_adapter: LLMProtocolV1,
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        framework_version: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        # Validate critical parameters
        if not hasattr(llm_adapter, "complete") or not callable(getattr(llm_adapter, "complete")):
            raise TypeError("llm_adapter must implement LLMProtocolV1 with 'complete' method")

        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")

        if max_tokens is not None and max_tokens < 1:
            raise ValueError("max_tokens must be positive")

        super().__init__(**kwargs)
        self._llm: LLMProtocolV1 = llm_adapter
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = max_tokens
        self._framework_version = framework_version

        logger.info(
            "CorpusLangChainLLM initialized with model=%s, temperature=%.2f",
            self.model,
            self.temperature,
        )

    # ------------------------------------------------------------------ #
    # LangChain-required properties
    # ------------------------------------------------------------------ #

    @property
    def _llm_type(self) -> str:
        """Identifier used by LangChain in serialization / introspection."""
        return "corpus"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return identifying parameters for LangChain serialization."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "framework_version": self._framework_version,
        }

    # ------------------------------------------------------------------ #
    # Error context management (consistent with other adapters)
    # ------------------------------------------------------------------ #

    def _build_error_context(
        self,
        operation: str,
        stream: bool,
        messages_count: int,
        model: str,
        ctx: OperationContext,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build consistent error context across all operations."""
        return {
            "framework": _FRAMEWORK_NAME,
            "operation": operation,
            "messages_count": messages_count,
            "model": model,
            "temperature": params.get("temperature"),
            "max_tokens": params.get("max_tokens"),
            "top_p": params.get("top_p"),
            "frequency_penalty": params.get("frequency_penalty"),
            "presence_penalty": params.get("presence_penalty"),
            "request_id": ctx.request_id,
            "tenant": ctx.tenant,
            "stream": stream,
        }

    @contextmanager
    def _error_context(
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
        Sync error-context wrapper to centralize attach_context usage.
        """
        error_ctx = self._build_error_context(
            operation, stream, messages_count, model, ctx, params
        )
        try:
            yield
        except BaseException as exc:
            attach_context(exc, **error_ctx)
            raise

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
        error_ctx = self._build_error_context(
            operation, stream, messages_count, model, ctx, params
        )
        try:
            yield
        except BaseException as exc:
            attach_context(exc, **error_ctx)
            raise

    # ------------------------------------------------------------------ #
    # Internal translation helpers (robust and typed)
    # ------------------------------------------------------------------ #

    def _translate_messages(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """
        LangChain messages → Corpus wire format ({role, content} dicts).
        
        Returns
        -------
        List[Dict[str, Any]]
            Normalized messages in Corpus wire format
        """
        if not messages:
            logger.warning("Empty messages list provided to LangChain adapter")
            return []

        try:
            normalized: List[NormalizedMessage] = [from_langchain(m) for m in messages]
            return [dict(m) for m in to_corpus(normalized)]
        except Exception as e:
            logger.error("Failed to translate LangChain messages: %s", e)
            raise ValueError(f"Message translation failed: {e}") from e

    def _build_context_and_params(
        self,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> tuple[OperationContext, Dict[str, Any]]:
        """
        Extract `OperationContext` and Corpus sampling params from LangChain kwargs.

        Returns
        -------
        tuple[OperationContext, Dict[str, Any]]
            Operation context and cleaned sampling parameters
        """
        config = kwargs.get("config")
        ctx = context_from_langchain(
            config,
            framework_version=self._framework_version,
        )

        # Build parameters with validation
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

        # Strip None values so the protocol sees a clean param set
        clean_params = {k: v for k, v in params.items() if v is not None}
        return ctx, clean_params

    @staticmethod
    def _build_chat_result(completion: LLMCompletion) -> ChatResult:
        """
        Map a Corpus `LLMCompletion` into a LangChain `ChatResult`.
        
        Returns
        -------
        ChatResult
            LangChain chat result with proper metadata
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
            response_metadata={
                "model": completion.model,
                "finish_reason": completion.finish_reason,
                "usage": usage_dict,
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
        return ChatResult(generations=[generation], llm_output=generation_info)

    @staticmethod
    def _chunk_to_generation_chunk(chunk: LLMChunk) -> ChatGenerationChunk:
        """
        Map a Corpus `LLMChunk` into a LangChain `ChatGenerationChunk`.
        
        Returns
        -------
        ChatGenerationChunk
            LangChain generation chunk with proper metadata
        """
        text = chunk.text or ""
        ai_chunk = AIMessageChunk(content=text)

        gen_info: Dict[str, Any] = {
            "model": chunk.model,
            "is_final": chunk.is_final,
            "finish_reason": getattr(chunk, "finish_reason", None),
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
    # LangChain async API (direct protocol access)
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

        Parameters
        ----------
        messages : List[BaseMessage]
            List of LangChain messages
        stop : Optional[List[str]], optional
            Stop sequences, by default None
        run_manager : Optional[AsyncCallbackManagerForLLMRun], optional
            Async callback manager, by default None
        **kwargs : Any
            Additional LangChain parameters

        Returns
        -------
        ChatResult
            LangChain chat result

        Raises
        ------
        Exception
            Any exception from the underlying LLM with rich context attached
        """
        corpus_messages = self._translate_messages(messages)
        ctx, params = self._build_context_and_params(stop=stop, **kwargs)
        model_for_context = params.get("model", self.model)
        messages_count = len(corpus_messages)

        # Notify start of LLM call
        if run_manager is not None:
            await run_manager.on_llm_start(
                self,
                messages,
                invocation_params=params,
                run_id=ctx.request_id,
            )

        async with self._error_context_async(
            "complete_async",
            stream=False,
            messages_count=messages_count,
            model=model_for_context,
            ctx=ctx,
            params=params,
        ):
            # Direct protocol access - no unnecessary abstraction
            result: LLMCompletion = await self._llm.acomplete(
                messages=corpus_messages,
                ctx=ctx,
                **params,
            )
            chat_result = self._build_chat_result(result)

            if run_manager is not None:
                await run_manager.on_llm_end(chat_result)

            return chat_result

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """
        Async streaming entrypoint used by LangChain.

        Parameters
        ----------
        messages : List[BaseMessage]
            List of LangChain messages
        stop : Optional[List[str]], optional
            Stop sequences, by default None
        run_manager : Optional[AsyncCallbackManagerForLLMRun], optional
            Async callback manager, by default None
        **kwargs : Any
            Additional LangChain parameters

        Yields
        ------
        ChatGenerationChunk
            Streaming generation chunks

        Raises
        ------
        Exception
            Any exception from the underlying LLM with rich context attached
        """
        corpus_messages = self._translate_messages(messages)
        ctx, params = self._build_context_and_params(stop=stop, **kwargs)
        model_for_context = params.get("model", self.model)
        messages_count = len(corpus_messages)

        # Notify start of LLM call
        if run_manager is not None:
            await run_manager.on_llm_start(
                self,
                messages,
                invocation_params=params,
                run_id=ctx.request_id,
            )

        async with self._error_context_async(
            "stream_async",
            stream=True,
            messages_count=messages_count,
            model=model_for_context,
            ctx=ctx,
            params=params,
        ):
            stream_canceled = False
            try:
                # Direct protocol access for streaming
                async for chunk in self._llm.astream(
                    messages=corpus_messages,
                    ctx=ctx,
                    **params,
                ):
                    gen_chunk = self._chunk_to_generation_chunk(chunk)
                    text = gen_chunk.message.content or ""

                    if run_manager is not None and text:
                        try:
                            await run_manager.on_llm_new_token(text, chunk=gen_chunk)
                        except Exception as callback_error:
                            logger.warning(
                                "LLM new token callback failed: %s",
                                callback_error,
                            )
                            stream_canceled = True
                            break

                    yield gen_chunk

            except Exception as stream_error:
                if run_manager is not None:
                    await run_manager.on_llm_error(stream_error)
                raise

            finally:
                if run_manager is not None and not stream_canceled:
                    completion_result = ChatResult(
                        generations=[
                            ChatGeneration(
                                message=AIMessage(content=""),
                                generation_info={
                                    "streaming": True,
                                    "completed": True,
                                    "model": model_for_context,
                                },
                            )
                        ]
                    )
                    await run_manager.on_llm_end(completion_result)

    # ------------------------------------------------------------------ #
    # LangChain sync API (consistent with async)
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

        Parameters
        ----------
        messages : List[BaseMessage]
            List of LangChain messages
        stop : Optional[List[str]], optional
            Stop sequences, by default None
        run_manager : Optional[CallbackManagerForLLMRun], optional
            Sync callback manager, by default None
        **kwargs : Any
            Additional LangChain parameters

        Returns
        -------
        ChatResult
            LangChain chat result

        Raises
        ------
        Exception
            Any exception from the underlying LLM with rich context attached
        """
        corpus_messages = self._translate_messages(messages)
        ctx, params = self._build_context_and_params(stop=stop, **kwargs)
        model_for_context = params.get("model", self.model)
        messages_count = len(corpus_messages)

        # Notify start of LLM call
        if run_manager is not None:
            run_manager.on_llm_start(
                self,
                messages,
                invocation_params=params,
                run_id=ctx.request_id,
            )

        with self._error_context(
            "complete_sync",
            stream=False,
            messages_count=messages_count,
            model=model_for_context,
            ctx=ctx,
            params=params,
        ):
            # Direct protocol access
            result: LLMCompletion = self._llm.complete(
                messages=corpus_messages,
                ctx=ctx,
                **params,
            )
            chat_result = self._build_chat_result(result)

            if run_manager is not None:
                run_manager.on_llm_end(chat_result)

            return chat_result

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """
        Sync streaming entrypoint used by LangChain.

        Parameters
        ----------
        messages : List[BaseMessage]
            List of LangChain messages
        stop : Optional[List[str]], optional
            Stop sequences, by default None
        run_manager : Optional[CallbackManagerForLLMRun], optional
            Sync callback manager, by default None
        **kwargs : Any
            Additional LangChain parameters

        Yields
        ------
        ChatGenerationChunk
            Streaming generation chunks

        Raises
        ------
        Exception
            Any exception from the underlying LLM with rich context attached
        """
        corpus_messages = self._translate_messages(messages)
        ctx, params = self._build_context_and_params(stop=stop, **kwargs)
        model_for_context = params.get("model", self.model)
        messages_count = len(corpus_messages)

        # Notify start of LLM call
        if run_manager is not None:
            run_manager.on_llm_start(
                self,
                messages,
                invocation_params=params,
                run_id=ctx.request_id,
            )

        with self._error_context(
            "stream_sync",
            stream=True,
            messages_count=messages_count,
            model=model_for_context,
            ctx=ctx,
            params=params,
        ):
            stream_canceled = False
            try:
                # Direct protocol access for streaming
                for chunk in self._llm.stream(
                    messages=corpus_messages,
                    ctx=ctx,
                    **params,
                ):
                    gen_chunk = self._chunk_to_generation_chunk(chunk)
                    text = gen_chunk.message.content or ""

                    if run_manager is not None and text:
                        try:
                            run_manager.on_llm_new_token(text, chunk=gen_chunk)
                        except Exception as callback_error:
                            logger.warning(
                                "LLM new token callback failed: %s",
                                callback_error,
                            )
                            stream_canceled = True
                            break

                    yield gen_chunk

            except Exception as stream_error:
                if run_manager is not None:
                    run_manager.on_llm_error(stream_error)
                raise

            finally:
                if run_manager is not None and not stream_canceled:
                    completion_result = ChatResult(
                        generations=[
                            ChatGeneration(
                                message=AIMessage(content=""),
                                generation_info={
                                    "streaming": True,
                                    "completed": True,
                                    "model": model_for_context,
                                },
                            )
                        ]
                    )
                    run_manager.on_llm_end(completion_result)

    # ------------------------------------------------------------------ #
    # Token counting (robust with multiple fallbacks)
    # ------------------------------------------------------------------ #

    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        """
        Estimate token count for a list of LangChain messages.

        Parameters
        ----------
        messages : List[BaseMessage]
            List of LangChain messages

        Returns
        -------
        int
            Estimated token count
        """
        if not messages:
            return 0

        # Build Corpus messages and context for counting
        corpus_messages = self._translate_messages(messages)
        ctx = context_from_langchain(
            None,
            framework_version=self._framework_version,
        )

        # Try protocol adapter counting first
        if hasattr(self._llm, "count_tokens"):
            try:
                combined_text = self._combine_messages_for_counting(messages)
                tokens = self._llm.count_tokens(
                    text=combined_text,
                    model=self.model,
                    ctx=ctx,
                )
                return int(tokens)
            except Exception as exc:
                logger.debug("Protocol count_tokens failed: %s", exc)

        # Fall back to improved character-based estimate
        try:
            return self._estimate_tokens_from_messages(messages)
        except Exception as exc:
            logger.debug("Token estimation failed: %s", exc)

        # Ultimate fallback to superclass
        return super().get_num_tokens_from_messages(messages)

    def _combine_messages_for_counting(self, messages: List[BaseMessage]) -> str:
        """Combine messages into a single string for token counting."""
        parts: List[str] = []
        for msg in messages:
            role = getattr(msg, "type", "user")
            content = str(getattr(msg, "content", ""))
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def _estimate_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        """Improved token estimation with better heuristics."""
        combined_text = self._combine_messages_for_counting(messages)

        if not combined_text:
            return 0

        # More sophisticated estimation
        char_count = len(combined_text)
        message_count = len(messages)

        char_based = max(1, char_count // 4)
        message_based = max(1, message_count)

        return max(char_based, message_based)

    def get_num_tokens(self, text: str) -> int:
        """
        Token counting for single text string.

        Parameters
        ----------
        text : str
            Input text to count tokens for

        Returns
        -------
        int
            Estimated token count
        """
        from langchain_core.messages import HumanMessage

        return self.get_num_tokens_from_messages([HumanMessage(content=text)])


__all__ = [
    "LangChainLLMProtocol",
    "CorpusLangChainLLM",
]