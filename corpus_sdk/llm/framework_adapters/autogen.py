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
- Delegate sync/async + streaming orchestration directly to LLM protocol
- Convert protocol-level `LLMCompletion` / `LLMChunk` into OpenAI-style
  ChatCompletion / ChatCompletionChunk payloads
- Enrich exceptions with AutoGen-specific debug metadata via `attach_context`

Design principles
-----------------
- Protocol-first:
    The Corpus `LLMProtocolV1` is the source of truth; this module is a thin
    compatibility layer for AutoGen.

- Direct protocol calls:
    No unnecessary abstraction layers - call LLM protocol methods directly
    after handling AutoGen-specific translation.

- Non-invasive:
    No retries, circuit breaking, or deadlines are implemented here beyond
    what the LLM protocol already provides.

- Rich error context:
    Exceptions are annotated via `attach_context` with framework-specific
    metadata, without mutating their messages or types.
"""

from __future__ import annotations

import logging
import time
from functools import wraps
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Dict,
    Iterator,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
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
from corpus_sdk.llm.llm_base import (
    LLMChunk,
    LLMCompletion,
    LLMProtocolV1,
    OperationContext,
)

logger = logging.getLogger(__name__)

# Type variables for decorators
T = TypeVar("T")


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


# ---------------------------------------------------------------------------
# Error Context Decorators (Rich Dynamic Context)
# ---------------------------------------------------------------------------


def _create_error_context_decorator(
    operation: str,
    is_async: bool = False,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Factory for creating error context decorators with rich per-call metrics.
    """
    def decorator_factory(
        **static_context: Any,
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            if is_async:

                @wraps(func)
                async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
                    # Extract dynamic context from call
                    dynamic_context = _extract_dynamic_context(self, args, kwargs, operation)
                    full_context = {**static_context, **dynamic_context}

                    try:
                        return await func(self, *args, **kwargs)
                    except Exception as exc:  # noqa: BLE001
                        attach_context(
                            exc,
                            framework="autogen",
                            operation=f"llm_{operation}",
                            **full_context,
                        )
                        raise

                return async_wrapper
            else:

                @wraps(func)
                def sync_wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
                    # Extract dynamic context from call
                    dynamic_context = _extract_dynamic_context(self, args, kwargs, operation)
                    full_context = {**static_context, **dynamic_context}

                    try:
                        return func(self, *args, **kwargs)
                    except Exception as exc:  # noqa: BLE001
                        attach_context(
                            exc,
                            framework="autogen",
                            operation=f"llm_{operation}",
                            **full_context,
                        )
                        raise

                return sync_wrapper

        return decorator
    return decorator_factory


def _extract_dynamic_context(
    instance: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    operation: str,
) -> Dict[str, Any]:
    """
    Extract rich dynamic context from method call for enhanced observability.
    """
    dynamic_ctx: Dict[str, Any] = {
        "model": getattr(instance, "model", "unknown"),
        "operation": operation,
    }

    # Extract message-based metrics
    if args and isinstance(args[0], (list, tuple)):
        messages = args[0]
        dynamic_ctx["messages_count"] = len(messages)

        # Calculate rough content metrics and roles distribution
        roles: Dict[str, int] = {}
        total_chars = 0
        for msg in messages:
            if not isinstance(msg, Mapping):
                continue
            role = msg.get("role", "unknown")
            roles[role] = roles.get(role, 0) + 1
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)

        dynamic_ctx["roles_distribution"] = roles
        dynamic_ctx["total_content_chars"] = total_chars

    # Extract AutoGen-specific context presence flags
    if kwargs.get("conversation") is not None:
        dynamic_ctx["has_conversation"] = True
    if kwargs.get("context") is not None:
        dynamic_ctx["has_extra_context"] = True

    # Extract sampling parameters (if provided)
    sampling_params = [
        "temperature",
        "max_tokens",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
    ]
    for param in sampling_params:
        if param in kwargs:
            dynamic_ctx[param] = kwargs[param]

    # Stream flag
    if "stream" in kwargs:
        dynamic_ctx["stream"] = bool(kwargs["stream"])

    # Request-scoped identifiers (if provided)
    if "request_id" in kwargs:
        dynamic_ctx["request_id"] = kwargs["request_id"]
    if "tenant" in kwargs:
        dynamic_ctx["tenant"] = kwargs["tenant"]

    return dynamic_ctx


# Convenience decorators with rich context extraction
def with_llm_error_context(
    operation: str,
    **static_context: Any,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for sync LLM methods with rich dynamic context extraction."""
    return _create_error_context_decorator(operation, is_async=False)(**static_context)


def with_async_llm_error_context(
    operation: str,
    **static_context: Any,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for async LLM methods with rich dynamic context extraction."""
    return _create_error_context_decorator(operation, is_async=True)(**static_context)


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

    They are normalized into Corpus wire messages and then passed directly
    to the underlying `LLMProtocolV1` implementation.

    Example:
    ```python
    from corpus_sdk.llm.framework_adapters.autogen import CorpusAutoGenChatClient
    import autogen

    # Initialize with any Corpus LLMProtocolV1 adapter
    llm_client = CorpusAutoGenChatClient(
        llm_adapter=my_adapter,
        model="gpt-4",
        temperature=0.7
    )

    # Use with AutoGen agent configuration
    agent = autogen.AssistantAgent(
        name="assistant",
        llm_config={
            "config_list": [{
                "model": "gpt-4",
                "api_key": "sk-...",  # Not used by Corpus client
                "client": llm_client  # Direct client assignment
            }]
        }
    )
    ```

    Error Handling Example:
    ```python
    try:
        response = llm_client.create(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4",
            temperature=0.7,
            conversation=agent_conversation
        )
    except Exception as e:
        # Rich error context automatically attached with message counts, roles, etc.
        logger.error("LLM call failed with context", exc_info=e)
    ```

    Streaming
    ---------
    - Async streaming: `acreate(..., stream=True)` returns an async iterator
      of OpenAI ChatCompletion chunk payloads.

    - Sync streaming: `create(..., stream=True)` returns an iterator of
      ChatCompletion chunk payloads.

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
        llm_adapter: LLMProtocolV1,
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        framework_version: Optional[str] = None,
    ) -> None:
        if not hasattr(llm_adapter, "complete") or not callable(
            getattr(llm_adapter, "complete", None)
        ):
            raise TypeError("llm_adapter must implement LLMProtocolV1 with 'complete' method")

        self._llm: LLMProtocolV1 = llm_adapter
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = max_tokens
        self._framework_version = framework_version

        logger.info(
            "CorpusAutoGenChatClient initialized with model=%s, temperature=%.2f",
            self.model,
            self.temperature,
        )

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

    def _execute_completion(
        self,
        messages: Sequence[Mapping[str, Any]],
        ctx: OperationContext,
        params: Dict[str, Any],
        is_async: bool = False,
    ) -> Union[LLMCompletion, Awaitable[LLMCompletion]]:
        """
        Unified completion execution using direct protocol calls.

        For `is_async=False` returns `LLMCompletion`.
        For `is_async=True` returns `Awaitable[LLMCompletion]`.
        """
        corpus_messages = self._translate_messages(messages)

        logger.debug(
            "Executing %s completion for %d messages with model: %s",
            "async" if is_async else "sync",
            len(messages),
            params.get("model", self.model),
        )

        if is_async:
            return self._llm.acomplete(
                messages=corpus_messages,
                ctx=ctx,
                **params,
            )
        else:
            return self._llm.complete(
                messages=corpus_messages,
                ctx=ctx,
                **params,
            )

    def _execute_streaming(
        self,
        messages: Sequence[Mapping[str, Any]],
        ctx: OperationContext,
        params: Dict[str, Any],
        is_async: bool = False,
    ) -> Union[Iterator[LLMChunk], AsyncIterator[LLMChunk]]:
        """
        Unified streaming execution using direct protocol calls.

        For `is_async=False` returns `Iterator[LLMChunk]`.
        For `is_async=True` returns `AsyncIterator[LLMChunk]`.
        """
        corpus_messages = self._translate_messages(messages)

        logger.debug(
            "Executing %s streaming for %d messages with model: %s",
            "async" if is_async else "sync",
            len(messages),
            params.get("model", self.model),
        )

        if is_async:
            return self._llm.astream(
                messages=corpus_messages,
                ctx=ctx,
                **params,
            )
        else:
            return self._llm.stream(
                messages=corpus_messages,
                ctx=ctx,
                **params,
            )

    # ------------------------------------------------------------------ #
    # Public async API (AutoGen-facing)
    # ------------------------------------------------------------------ #

    @with_async_llm_error_context("acreate")
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

        if not stream:
            result_awaitable = self._execute_completion(
                messages=messages,
                ctx=ctx,
                params=params,
                is_async=True,
            )
            result = await result_awaitable
            return self._completion_to_openai(result)

        # Streaming: return async iterator of OpenAI-style chunks.
        async def _gen() -> AsyncIterator[Dict[str, Any]]:
            stream_id = _new_id()
            created = _now_epoch_s()
            is_first = True

            chunk_iter = self._execute_streaming(
                messages=messages,
                ctx=ctx,
                params=params,
                is_async=True,
            )

            async for chunk in chunk_iter:
                yield self._chunk_to_openai(
                    chunk,
                    stream_id=stream_id,
                    created=created,
                    model_fallback=params.get("model", self.model),
                    is_first=is_first,
                )
                is_first = False

        return _gen()

    # ------------------------------------------------------------------ #
    # Public sync API (AutoGen-facing)
    # ------------------------------------------------------------------ #

    @with_llm_error_context("create")
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

        if not stream:
            result = self._execute_completion(
                messages=messages,
                ctx=ctx,
                params=params,
                is_async=False,
            )
            return self._completion_to_openai(result)

        # Streaming: sync path via direct protocol streaming.
        def _iter() -> Iterator[Dict[str, Any]]:
            stream_id = _new_id()
            created = _now_epoch_s()
            is_first = True

            chunk_iter = self._execute_streaming(
                messages=messages,
                ctx=ctx,
                params=params,
                is_async=False,
            )

            for chunk in chunk_iter:
                yield self._chunk_to_openai(
                    chunk,
                    stream_id=stream_id,
                    created=created,
                    model_fallback=params.get("model", self.model),
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
    "with_llm_error_context",
    "with_async_llm_error_context",
]