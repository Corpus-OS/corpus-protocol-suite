# corpus_sdk/llm/framework_adapters/autogen.py
# SPDX-License-Identifier: Apache-2.0

"""
AutoGen adapter for Corpus LLM protocol.

This module exposes a Corpus `BaseLLMAdapter` as an OpenAI-style chat client
suitable for use with AutoGen's configuration system.

Key responsibilities
--------------------
- Convert OpenAI-style message dicts → Corpus wire messages
- Bridge async-first Corpus APIs to AutoGen's sync + async expectations
- Expose non-streaming and streaming `create` / `acreate` methods
- Preserve framework context and enrich exceptions with debug metadata
- Use a shared SyncStreamBridge for production-grade sync streaming

Design principles
-----------------
- Protocol-first:
    Corpus `BaseLLMAdapter` remains the source of truth; this module is a
    thin compatibility layer for AutoGen.

- Async-first:
    All core work is done in async helpers (`_acreate_openai`, `_astream_openai`),
    with sync `create` implemented via `AsyncBridge` + `SyncStreamBridge`.

- Non-invasive:
    No retries, circuit breaking, or deadlines are implemented here beyond
    optional transient retry in the sync streaming bridge. Those concerns
    belong in `BaseLLMAdapter` and infra layers.

- Rich error context:
    Exceptions are annotated via `attach_context` with framework-specific
    metadata, without mutating their messages or types.
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
from uuid import uuid4

from corpus_sdk.llm.llm_base import (
    BaseLLMAdapter,
    LLMChunk,
    LLMCompletion,
    OperationContext,
    TransientNetwork,
    Unavailable,
)
from corpus_sdk.llm.framework_adapters.common.async_bridge import AsyncBridge
from corpus_sdk.llm.framework_adapters.common.context_translation import ContextTranslator
from corpus_sdk.llm.framework_adapters.common.error_context import attach_context
from corpus_sdk.llm.framework_adapters.common.message_translation import (
    NormalizedMessage,
    from_autogen,
    to_corpus,
)
from corpus_sdk.llm.framework_adapters.common.sync_stream_bridge import SyncStreamBridge

logger = logging.getLogger(__name__)


class CorpusAutoGenChatClient:
    """
    OpenAI-style chat client backed by a Corpus `BaseLLMAdapter`.

    This class is intended to be dropped into AutoGen configs wherever an
    OpenAI-compatible chat client is expected. It exposes `create` and
    `acreate` methods with familiar semantics:

        - `create(messages=[...], model="...", stream=False)`
        - `acreate(messages=[...], model="...", stream=True)`

    Messages are expected to be OpenAI-style dicts:

        {"role": "user", "content": "Hello"}

    and are internally normalized via `from_autogen` and `to_corpus`.

    Streaming
    ---------
    - Async streaming: `acreate(..., stream=True)` returns an async iterator
      of OpenAI ChatCompletion chunk payloads.

    - Sync streaming: `create(..., stream=True)` uses `SyncStreamBridge`
      to run the async stream in a background thread and yield chunks
      synchronously, with backpressure, optional cancellation, and optional
      transient retry.

    Error context
    -------------
    All exceptions arising inside this adapter are enriched via
    `attach_context(exc, framework="autogen", ...)`, making it easy for
    higher-level handlers to inspect request-specific metadata without
    modifying exception messages or types.
    """

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        *,
        corpus_adapter: BaseLLMAdapter,
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        # Streaming tuning knobs for sync streaming (defaults)
        stream_queue_maxsize: int = 16,
        stream_poll_timeout_s: float = 0.1,
        stream_join_timeout_s: float = 2.0,
        # Transient retry knobs for sync streaming
        max_transient_retries: int = 0,
        transient_backoff_s: float = 0.25,
        stream_transient_error_types: Optional[Tuple[Type[BaseException], ...]] = None,
    ) -> None:
        self._adapter = corpus_adapter
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = max_tokens

        # Default sync streaming configuration
        self.stream_queue_maxsize = int(stream_queue_maxsize)
        self.stream_poll_timeout_s = float(stream_poll_timeout_s)
        self.stream_join_timeout_s = float(stream_join_timeout_s)

        # Default transient retry for sync streaming (before first item)
        self.max_transient_retries = int(max_transient_retries)
        self.transient_backoff_s = float(transient_backoff_s)

        if stream_transient_error_types is None:
            self.stream_transient_error_types: Tuple[Type[BaseException], ...] = (
                TransientNetwork,
                Unavailable,
            )
        else:
            self.stream_transient_error_types = stream_transient_error_types

    # ------------------------------------------------------------------ #
    # Core translation helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _now_epoch_s() -> int:
        return int(time.time())

    @staticmethod
    def _new_id(prefix: str = "chatcmpl") -> str:
        return f"{prefix}-{uuid4().hex}"

    def _translate_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> List[Mapping[str, str]]:
        """
        OpenAI/AutoGen-style messages → Corpus wire messages.

        Uses `from_autogen` → `NormalizedMessage` → `to_corpus`.
        """
        normalized: List[NormalizedMessage] = [from_autogen(m) for m in messages]
        return to_corpus(normalized)

    def _build_ctx_and_params(
        self,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[OperationContext, Dict[str, Any]]:
        """
        Construct OperationContext and Corpus sampling params from kwargs.

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

        Everything else is ignored at this layer (and should be handled by
        BaseLLMAdapter via ctx.attrs if needed).
        """
        # Build OperationContext from AutoGen-specific context if provided.
        ctx = ContextTranslator.from_autogen_context(
            conversation=conversation,
            extra=extra_context,
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
        stop_sequences: Optional[List[str]] = None
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

        # Drop None values so BaseLLMAdapter sees a clean param set.
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
        completion_id = completion_id or CorpusAutoGenChatClient._new_id()
        created = created or CorpusAutoGenChatClient._now_epoch_s()

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
    # Async core implementations
    # ------------------------------------------------------------------ #

    async def _acreate_openai(
        self,
        messages: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Core async non-streaming path:
            OpenAI messages → Corpus.complete → OpenAI ChatCompletion.
        """
        # Extract AutoGen/extra context first.
        conversation = kwargs.pop("conversation", None)
        extra_context = kwargs.pop("context", None)
        if extra_context is not None and not isinstance(extra_context, Mapping):
            extra_context = None

        corpus_messages = self._translate_messages(messages)
        ctx, params = self._build_ctx_and_params(
            conversation=conversation,
            extra_context=extra_context,
            **kwargs,
        )

        try:
            result = await self._adapter.complete(
                messages=corpus_messages,
                ctx=ctx,
                **params,
            )
            return self._completion_to_openai(result)
        except BaseException as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="autogen",
                operation="complete",
                messages_count=len(messages),
                model=params.get("model", self.model),
                temperature=params.get("temperature"),
                max_tokens=params.get("max_tokens"),
                top_p=params.get("top_p"),
                frequency_penalty=params.get("frequency_penalty"),
                presence_penalty=params.get("presence_penalty"),
                request_id=ctx.request_id,
                tenant=ctx.tenant,
                stream=False,
            )
            raise

    async def _astream_openai(
        self,
        messages: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Core async streaming path:
            OpenAI messages → Corpus.stream → OpenAI ChatCompletion chunks.
        """
        conversation = kwargs.pop("conversation", None)
        extra_context = kwargs.pop("context", None)
        if extra_context is not None and not isinstance(extra_context, Mapping):
            extra_context = None

        corpus_messages = self._translate_messages(messages)
        ctx, params = self._build_ctx_and_params(
            conversation=conversation,
            extra_context=extra_context,
            **kwargs,
        )

        stream_id = self._new_id()
        created = self._now_epoch_s()
        model_for_context = params.get("model", self.model)
        is_first = True

        try:
            agen = await self._adapter.stream(
                messages=corpus_messages,
                ctx=ctx,
                **params,
            )

            try:
                async for chunk in agen:
                    yield self._chunk_to_openai(
                        chunk,
                        stream_id=stream_id,
                        created=created,
                        model_fallback=model_for_context,
                        is_first=is_first,
                    )
                    is_first = False
            finally:
                aclose = getattr(agen, "aclose", None)
                if callable(aclose):
                    try:
                        await aclose()
                    except Exception as cleanup_error:  # noqa: BLE001
                        logger.debug(
                            "AutoGen adapter: stream cleanup failed: %s",
                            cleanup_error,
                            extra={"framework": "autogen"},
                        )

        except BaseException as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="autogen",
                operation="stream",
                messages_count=len(messages),
                model=model_for_context,
                temperature=params.get("temperature"),
                max_tokens=params.get("max_tokens"),
                top_p=params.get("top_p"),
                frequency_penalty=params.get("frequency_penalty"),
                presence_penalty=params.get("presence_penalty"),
                request_id=ctx.request_id,
                tenant=ctx.tenant,
                stream=True,
            )
            raise

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

        if not stream:
            return await self._acreate_openai(messages, **kwargs)

        # Streaming: return async iterator of OpenAI-style chunks.
        return self._astream_openai(messages, **kwargs)

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
            - cancel_event: Optional[threading.Event] for early cancellation
            - Per-call streaming overrides:
                * stream_queue_maxsize
                * stream_poll_timeout_s
                * stream_join_timeout_s
                * stream_max_transient_retries
                * stream_transient_backoff_s
                * stream_transient_error_types
            - All other parameters as in `acreate`.

        Returns
        -------
        Dict[str, Any] | Iterator[Dict[str, Any]]
        """
        stream = bool(kwargs.pop("stream", False))
        cancel_event = kwargs.pop("cancel_event", None)
        if cancel_event is not None and not isinstance(cancel_event, threading.Event):
            raise TypeError("cancel_event must be a threading.Event if provided")

        # Per-call streaming overrides (fall back to instance defaults)
        stream_queue_maxsize = int(kwargs.pop("stream_queue_maxsize", self.stream_queue_maxsize))
        stream_poll_timeout_s = float(kwargs.pop("stream_poll_timeout_s", self.stream_poll_timeout_s))
        stream_join_timeout_s = float(kwargs.pop("stream_join_timeout_s", self.stream_join_timeout_s))
        stream_max_transient_retries = int(
            kwargs.pop("stream_max_transient_retries", self.max_transient_retries)
        )
        stream_transient_backoff_s = float(
            kwargs.pop("stream_transient_backoff_s", self.transient_backoff_s)
        )
        stream_transient_error_types = kwargs.pop("stream_transient_error_types", None)

        if stream_transient_error_types is not None and not isinstance(
            stream_transient_error_types, tuple
        ):
            raise TypeError("stream_transient_error_types must be a tuple of exception types")

        if not stream:
            # Non-streaming: simple AsyncBridge wrapper.
            try:
                return AsyncBridge.run_async(self._acreate_openai(messages, **kwargs))
            except BaseException as exc:  # noqa: BLE001
                # We may not know ctx/params here, but we can still enrich with
                # basic context; _acreate_openai also attaches its own context.
                attach_context(
                    exc,
                    framework="autogen",
                    operation="complete_sync_wrapper",
                    messages_count=len(messages),
                    model=kwargs.get("model", self.model),
                    stream=False,
                )
                raise

        # Streaming: use SyncStreamBridge for robust sync streaming behavior.
        model_for_context = kwargs.get("model", self.model)

        async def _factory() -> AsyncIterator[Dict[str, Any]]:
            # Note: we return the async iterator, not iterate here.
            return self._astream_openai(messages, **kwargs)

        # Decide which transient error types to use for this call.
        effective_transient_types: Tuple[Type[BaseException], ...]
        if stream_transient_error_types is not None:
            effective_transient_types = stream_transient_error_types
        else:
            effective_transient_types = self.stream_transient_error_types

        error_context: Dict[str, Any] = {
            "operation": "stream",
            "messages_count": len(messages),
            "model": model_for_context,
            "stream_queue_maxsize": stream_queue_maxsize,
            "stream_poll_timeout_s": stream_poll_timeout_s,
            "stream_join_timeout_s": stream_join_timeout_s,
            "stream_max_transient_retries": stream_max_transient_retries,
            "stream_transient_backoff_s": stream_transient_backoff_s,
            "stream": True,
        }
        if effective_transient_types:
            error_context["stream_transient_error_types"] = [
                t.__name__ for t in effective_transient_types
            ]

        bridge = SyncStreamBridge(
            coro_factory=_factory,
            queue_maxsize=stream_queue_maxsize,
            poll_timeout_s=stream_poll_timeout_s,
            join_timeout_s=stream_join_timeout_s,
            cancel_event=cancel_event,
            framework="autogen",
            error_context=error_context,
            max_transient_retries=stream_max_transient_retries,
            transient_backoff_s=stream_transient_backoff_s,
            transient_error_types=effective_transient_types,
        )

        return bridge.run()

    # Allow direct call usage: client(...) behaves like client.create(...)
    def __call__(
        self,
        messages: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        return self.create(messages, **kwargs)


__all__ = [
    "CorpusAutoGenChatClient",
]
