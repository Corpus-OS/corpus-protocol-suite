# corpus_sdk/llm/framework_adapters/autogen.py
# SPDX-License-Identifier: Apache-2.0

"""
AutoGen adapter for Corpus LLM protocol.

This module exposes a Corpus `LLMProtocolV1` implementation as an OpenAI-style
chat client suitable for use with AutoGen's configuration system.

Key responsibilities
--------------------
- Accept OpenAI-style message dicts from AutoGen
- Convert them to Corpus wire messages (via message normalization helpers)
- Build OperationContext from AutoGen conversation / metadata
- Construct sampling / routing parameters (model, temperature, stop, etc.)
- Delegate sync/async + streaming orchestration to `LLMTranslator`
- Convert protocol-level `LLMCompletion` / `LLMChunk` into OpenAI-style
  ChatCompletion / ChatCompletionChunk payloads
- Enrich exceptions with AutoGen-specific debug metadata via `attach_context`

Design principles
-----------------
- Protocol-first:
    The Corpus `LLMProtocolV1` is the source of truth; this module is a thin
    compatibility layer for AutoGen.

- Translator-centric:
    All async→sync bridging, streaming glue, and common error-handling are
    handled by `LLMTranslator`, mirroring the graph adapter pattern.

- Non-invasive:
    No retries, circuit breaking, or deadlines are implemented here beyond
    what the LLM protocol / translator already provide.

- Rich error context:
    Exceptions are annotated via `attach_context` with framework-specific
    metadata, without mutating their messages or types.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager, contextmanager
from functools import cached_property
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)
from uuid import uuid4

from corpus_sdk.core.context_translation import ContextTranslator
from corpus_sdk.core.error_context import attach_context
from corpus_sdk.llm.framework_adapters.common.message_translation import (
    NormalizedMessage,
    from_autogen,
    to_corpus,
)
from corpus_sdk.llm.framework_adapters.common.llm_translation import (
    DefaultLLMFrameworkTranslator,
    LLMTranslator,
)
from corpus_sdk.llm.llm_base import (
    LLMChunk,
    LLMCompletion,
    LLMProtocolV1,
    OperationContext,
)

logger = logging.getLogger(__name__)


class AutoGenLLMClientProtocol(Protocol):
    """
    Protocol representing the minimal AutoGen-aware LLM client interface
    implemented by this module.

    This structural protocol allows callers to type against the chat client
    without depending on the concrete `CorpusAutoGenChatClient` class.
    """

    async def acreate(
        self,
        messages: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
        ...

    def create(
        self,
        messages: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        ...


def _now_epoch_s() -> int:
    return int(time.time())


def _new_id(prefix: str = "chatcmpl") -> str:
    return f"{prefix}-{uuid4().hex}"


class CorpusAutoGenChatClient:
    """
    OpenAI-style chat client backed by a Corpus `LLMProtocolV1`.

    This class is intended to be dropped into AutoGen configs wherever an
    OpenAI-compatible chat client is expected. It exposes `create` and
    `acreate` methods with familiar semantics:

        - `create(messages=[...], model="...", stream=False)`
        - `acreate(messages=[...], model="...", stream=True)`

    Messages are expected to be OpenAI-style dicts:

        {"role": "user", "content": "Hello"}

    They are normalized into Corpus wire messages and then passed through
    `LLMTranslator` to the underlying `LLMProtocolV1` implementation.

    Streaming
    ---------
    - Async streaming: `acreate(..., stream=True)` returns an async iterator
      of OpenAI ChatCompletion chunk payloads.

    - Sync streaming: `create(..., stream=True)` uses the sync streaming
      path of `LLMTranslator` to yield chunks synchronously.

    Error context
    -------------
    All exceptions arising inside this adapter are enriched via
    `attach_context(exc, framework="autogen", ...)`, making it easy for
    higher-level handlers to inspect request-specific metadata without
    modifying exception messages or types.
    """

    class _AutoGenLLMFrameworkTranslator(DefaultLLMFrameworkTranslator):
        """
        AutoGen-specific LLM framework translator.

        Currently this subclass does not override any behavior from
        `DefaultLLMFrameworkTranslator`, but it exists to mirror the graph
        adapter pattern and to provide a dedicated hook for AutoGen-specific
        customizations in the future.
        """

        pass

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        *,
        llm_adapter: LLMProtocolV1,
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        framework_version: Optional[str] = None,
    ) -> None:
        self._llm: LLMProtocolV1 = llm_adapter
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = max_tokens
        self._framework_version = framework_version

    # ------------------------------------------------------------------ #
    # Translator (lazy, cached)
    # ------------------------------------------------------------------ #

    @cached_property
    def _translator(self) -> LLMTranslator:
        """
        Lazily construct and cache the `LLMTranslator`.

        This mirrors the graph adapter pattern: all async→sync bridging,
        streaming orchestration, and protocol-level error handling are
        centralized in `LLMTranslator`.
        """
        framework_translator = self._AutoGenLLMFrameworkTranslator()
        return LLMTranslator(
            adapter=self._llm,
            framework="autogen",
            translator=framework_translator,
        )

    # ------------------------------------------------------------------ #
    # Error-context helpers (to avoid boilerplate)
    # ------------------------------------------------------------------ #

    @contextmanager
    def _error_context(
        self,
        operation: str,
        *,
        stream: bool,
        messages_count: int,
        model: str,
        ctx: OperationContext,
        params: Mapping[str, Any],
    ):
        """
        Sync error-context wrapper to centralize attach_context usage.
        """
        try:
            yield
        except BaseException as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="autogen",
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

    @asynccontextmanager
    async def _error_context_async(
        self,
        operation: str,
        *,
        stream: bool,
        messages_count: int,
        model: str,
        ctx: OperationContext,
        params: Mapping[str, Any],
    ):
        """
        Async error-context wrapper to centralize attach_context usage.
        """
        try:
            yield
        except BaseException as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="autogen",
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
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _translate_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> Sequence[Mapping[str, Any]]:
        """
        OpenAI/AutoGen-style messages → Corpus wire messages.

        Uses `from_autogen` → `NormalizedMessage` → `to_corpus`.
        """
        normalized: Sequence[NormalizedMessage] = [from_autogen(m) for m in messages]
        return to_corpus(normalized)

    def _build_ctx_and_params(
        self,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[OperationContext, Dict[str, Any]]:
        """
        Construct OperationContext and sampling params from kwargs.

        Recognized kwargs (removed from kwargs when processed):
            - model
            - temperature
            - max_tokens
            - top_p
            - frequency_penalty
            - presence_penalty
            - stop  (str | list[str])
            - system_message
            - request_id
            - tenant

        Everything else is ignored at this layer. If additional data needs
        to reach the protocol layer, it should be carried via the context
        translator (conversation / extra_context) and OperationContext.attrs.
        """
        ctx = ContextTranslator.from_autogen_context(
            conversation=conversation,
            extra=extra_context,
            framework_version=self._framework_version,
        )

        # Optional overrides for request_id / tenant.
        request_id = kwargs.pop("request_id", None)
        tenant = kwargs.pop("tenant", None)

        if request_id is not None or tenant is not None:
            ctx = OperationContext(
                request_id=request_id or ctx.request_id,
                idempotency_key=ctx.idempotency_key,
                deadline_ms=ctx.deadline_ms,
                traceparent=ctx.traceparent,
                tenant=tenant or ctx.tenant,
                attrs=ctx.attrs,
            )

        # Stop sequences: OpenAI uses "stop" (str or list[str]).
        stop_arg = kwargs.pop("stop", None)
        stop_sequences: Optional[list[str]] = None
        if isinstance(stop_arg, str):
            stop_sequences = [stop_arg]
        elif isinstance(stop_arg, (list, tuple)):
            stop_sequences = [str(s) for s in stop_arg]

        # Sampling and routing params.
        params: Dict[str, Any] = {
            "model": kwargs.pop("model", self.model),
            "temperature": kwargs.pop("temperature", self.temperature),
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
            "top_p": kwargs.pop("top_p", None),
            "frequency_penalty": kwargs.pop("frequency_penalty", None),
            "presence_penalty": kwargs.pop("presence_penalty", None),
            "stop_sequences": stop_sequences,
            "system_message": kwargs.pop("system_message", None),
        }

        # Drop None values so the protocol sees a clean param set.
        params = {k: v for k, v in params.items() if v is not None}
        return ctx, params

    @staticmethod
    def _completion_to_openai(
        result: LLMCompletion,
        *,
        completion_id: Optional[str] = None,
        created: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Convert a Corpus `LLMCompletion` into an OpenAI ChatCompletion payload.
        """
        completion_id = completion_id or _new_id()
        created = created or _now_epoch_s()

        usage = result.usage
        usage_dict = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": result.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result.text,
                    },
                    "finish_reason": result.finish_reason,
                }
            ],
            "usage": usage_dict,
        }

    @staticmethod
    def _chunk_to_openai(
        chunk: LLMChunk,
        *,
        stream_id: str,
        created: int,
        model_fallback: str,
        is_first: bool,
    ) -> Dict[str, Any]:
        """
        Convert a Corpus `LLMChunk` into an OpenAI ChatCompletion chunk payload.
        """
        delta: Dict[str, Any] = {}

        # OpenAI typically sends role only on the first delta.
        if is_first:
            delta["role"] = "assistant"

        if chunk.text:
            delta["content"] = chunk.text

        choice: Dict[str, Any] = {
            "index": 0,
            "delta": delta,
            "finish_reason": "stop" if chunk.is_final else None,
        }

        payload: Dict[str, Any] = {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": chunk.model or model_fallback,
            "choices": [choice],
        }

        if chunk.usage_so_far is not None:
            usage = chunk.usage_so_far
            payload["usage"] = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }

        return payload

    # ------------------------------------------------------------------ #
    # Public async API (AutoGen-facing)
    # ------------------------------------------------------------------ #

    async def acreate(
        self,
        messages: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
        """
        AutoGen-style async entrypoint.

        Usage:
            await client.acreate(messages=[...], model="...", stream=False)
            async for chunk in client.acreate(messages=[...], stream=True): ...

        Parameters
        ----------
        messages:
            OpenAI-style messages list.

        kwargs:
            - stream: bool (default False)
            - model, temperature, max_tokens, top_p, frequency_penalty,
              presence_penalty, stop, system_message, conversation, context,
              request_id, tenant, etc.

        Returns
        -------
        Dict[str, Any] | AsyncIterator[Dict[str, Any]]
        """
        stream = bool(kwargs.pop("stream", False))

        # Extract AutoGen/extra context.
        conversation = kwargs.pop("conversation", None)
        extra_context = kwargs.pop("context", None)
        if extra_context is not None and not isinstance(extra_context, Mapping):
            extra_context = None

        ctx, params = self._build_ctx_and_params(
            conversation=conversation,
            extra_context=extra_context,
            **kwargs,
        )
        corpus_messages = self._translate_messages(messages)
        model_for_context = params.get("model", self.model)
        messages_count = len(messages)

        if not stream:
            async with self._error_context_async(
                "complete_async",
                stream=False,
                messages_count=messages_count,
                model=model_for_context,
                ctx=ctx,
                params=params,
            ):
                result = await self._translator.arun_complete(
                    messages=corpus_messages,
                    op_ctx=ctx,
                    params=params,
                )
                return self._completion_to_openai(result)

        # Streaming: return async iterator of OpenAI-style chunks.
        async def _gen() -> AsyncIterator[Dict[str, Any]]:
            stream_id = _new_id()
            created = _now_epoch_s()
            is_first = True

            async with self._error_context_async(
                "stream_async",
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
                    yield self._chunk_to_openai(
                        chunk,
                        stream_id=stream_id,
                        created=created,
                        model_fallback=model_for_context,
                        is_first=is_first,
                    )
                    is_first = False

        return _gen()

    # ------------------------------------------------------------------ #
    # Public sync API (AutoGen-facing)
    # ------------------------------------------------------------------ #

    def create(
        self,
        messages: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        AutoGen-style sync entrypoint.

        Usage:
            client.create(messages=[...], model="...", stream=False)
            for chunk in client.create(messages=[...], stream=True): ...

        Parameters
        ----------
        messages:
            OpenAI-style messages list.

        kwargs:
            - stream: bool (default False)
            - model, temperature, max_tokens, top_p, frequency_penalty,
              presence_penalty, stop, system_message, conversation, context,
              request_id, tenant, etc.

        Returns
        -------
        Dict[str, Any] | Iterator[Dict[str, Any]]
        """
        stream = bool(kwargs.pop("stream", False))

        # Extract AutoGen/extra context.
        conversation = kwargs.pop("conversation", None)
        extra_context = kwargs.pop("context", None)
        if extra_context is not None and not isinstance(extra_context, Mapping):
            extra_context = None

        ctx, params = self._build_ctx_and_params(
            conversation=conversation,
            extra_context=extra_context,
            **kwargs,
        )
        corpus_messages = self._translate_messages(messages)
        model_for_context = params.get("model", self.model)
        messages_count = len(messages)

        if not stream:
            with self._error_context(
                "complete_sync",
                stream=False,
                messages_count=messages_count,
                model=model_for_context,
                ctx=ctx,
                params=params,
            ):
                result = self._translator.complete(
                    messages=corpus_messages,
                    op_ctx=ctx,
                    params=params,
                )
                return self._completion_to_openai(result)

        # Streaming: sync path via LLMTranslator.stream.
        def _iter() -> Iterator[Dict[str, Any]]:
            stream_id = _new_id()
            created = _now_epoch_s()
            is_first = True

            with self._error_context(
                "stream_sync",
                stream=True,
                messages_count=messages_count,
                model=model_for_context,
                ctx=ctx,
                params=params,
            ):
                for chunk in self._translator.stream(
                    messages=corpus_messages,
                    op_ctx=ctx,
                    params=params,
                ):
                    yield self._chunk_to_openai(
                        chunk,
                        stream_id=stream_id,
                        created=created,
                        model_fallback=model_for_context,
                        is_first=is_first,
                    )
                    is_first = False

        return _iter()

    # Allow direct call usage: client(...) behaves like client.create(...)
    def __call__(
        self,
        messages: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        return self.create(messages, **kwargs)


__all__ = [
    "AutoGenLLMClientProtocol",
    "CorpusAutoGenChatClient",
]
