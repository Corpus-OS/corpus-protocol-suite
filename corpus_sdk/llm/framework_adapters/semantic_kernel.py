# corpus_sdk/llm/framework_adapters/semantic_kernel.py
# SPDX-License-Identifier: Apache-2.0

"""
Semantic Kernel adapter for Corpus LLM protocol.

This module exposes a Corpus `BaseLLMAdapter` as a Semantic Kernel
`ChatCompletionClientBase` implementation so that:

- SK agents can use any Corpus-backed LLM adapter as a chat completion service.
- Async + streaming flows remain async-first (no background threads).
- Context / deadlines / tenant hints are propagated via `OperationContext`.
- Sampling parameters are bridged from `PromptExecutionSettings` to Corpus.
- Optional transient error retry with exponential backoff.
- Rich error context for enhanced observability and debugging.

Design goals
------------

* Protocol-first:
    Semantic Kernel is an integration surface. All real behavior goes
    through the Corpus `BaseLLMAdapter` and the LLM protocol in
    `corpus_sdk.llm.llm_base`.

* Optional dependency safe:
    This module can be imported without Semantic Kernel installed.
    Attempting to *instantiate* the adapter without SK will raise a
    clear RuntimeError.

* Non-lossy metadata where feasible:
    - We preserve SK-side model / finish_reason / usage in message metadata.
    - Deadlines / timeouts flowing through `PromptExecutionSettings` are
      mapped into `OperationContext.deadline_ms` via
      `ContextTranslator.from_semantic_kernel_context`.

* Async-first:
    - No sync wrappers or background threads.
    - Cancellation works via normal asyncio task cancellation semantics
      plus ctx.deadline_ms at the Corpus adapter layer.

* Production resilience:
    - Configurable retry for transient errors
    - Rich error context attachment for debugging
    - Comprehensive logging and observability

Typical usage
-------------

    from semantic_kernel.agents import ChatCompletionAgent
    from corpus_sdk.llm.openai_adapter import OpenAIAdapter
    from corpus_sdk.llm.framework_adapters.semantic_kernel import (
        CorpusSemanticKernelChatCompletion,
    )

    corpus_adapter = OpenAIAdapter(api_key="...")

    sk_llm = CorpusSemanticKernelChatCompletion(
        corpus_adapter=corpus_adapter,
        model="gpt-4o",
        # Optional: enable retry for transient errors
        max_transient_retries=2,
        transient_backoff_s=0.5,
    )

    agent = ChatCompletionAgent(
        service=sk_llm,
        name="MyAgent",
        instructions="You are a helpful assistant.",
    )
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Mapping,
    Optional,
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
from corpus_sdk.core.context_translation import ContextTranslator
from corpus_sdk.core.error_context import attach_context
from corpus_sdk.llm.framework_adapters.common.message_translation import (
    MessageTranslator,
    NormalizedMessage,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Semantic Kernel imports
# ---------------------------------------------------------------------------

try:  # pragma: no cover - import surface only; behavior is type-driven
    from semantic_kernel.connectors.ai.chat_completion_client_base import (  # type: ignore[import]
        ChatCompletionClientBase,
        ChatHistory,
        PromptExecutionSettings,
    )
    from semantic_kernel.contents import (  # type: ignore[import]
        AuthorRole,
        ChatMessageContent,
        StreamingChatMessageContent,
    )
except Exception as _IMPORT_EXC:  # pragma: no cover - runtime guard
    # We still allow importing this module so that type checking works.
    # Instantiating the adapter without SK present will raise a RuntimeError.
    ChatCompletionClientBase = object  # type: ignore[misc,assignment]
    ChatHistory = Any  # type: ignore[misc,assignment]
    PromptExecutionSettings = Any  # type: ignore[misc,assignment]
    AuthorRole = Any  # type: ignore[misc,assignment]
    ChatMessageContent = Any  # type: ignore[misc,assignment]
    StreamingChatMessageContent = Any  # type: ignore[misc,assignment]
    _SEMANTIC_KERNEL_IMPORT_ERROR: Optional[Exception] = _IMPORT_EXC
else:  # pragma: no cover - trivial branch
    _SEMANTIC_KERNEL_IMPORT_ERROR = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _author_role_to_protocol_role(role: Any) -> str:
    """
    Map SK AuthorRole → Corpus protocol role string.

    We deliberately avoid relying on exact enum values; instead we use a
    string-based mapping to be resilient to minor API changes.
    """
    if role is None:
        return "user"

    # AuthorRole may be an enum with `.name` or `.value`.
    name = getattr(role, "name", None) or getattr(role, "value", None) or str(role)
    key = str(name).strip().lower()

    if "system" in key:
        return "system"
    if "assistant" in key or key == "ai":
        return "assistant"
    if "tool" in key or "function" in key:
        return "tool"
    return "user"


def _extract_text_from_chat_message(msg: Any) -> str:
    """
    Best-effort extraction of human-readable text from a ChatMessageContent.

    Strategy:
    - Prefer `msg.content` when present.
    - If empty, fall back to concatenation of text-bearing `items` when available.
    - As a last resort, use `str(msg)`.
    """
    # Preferred: direct content string.
    content = getattr(msg, "content", None)
    if isinstance(content, str) and content:
        return content

    # Fallback: iterate items with `.text` fields (TextContent, etc.).
    items = getattr(msg, "items", None)
    parts: List[str] = []
    if items:
        for item in items:
            text = getattr(item, "text", None)
            if isinstance(text, str) and text:
                parts.append(text)
    if parts:
        return "\n".join(parts)

    # Final fallback.
    try:
        return str(msg)
    except Exception:  # pragma: no cover - ultra defensive
        return ""


def _history_to_normalized_messages(history: ChatHistory) -> List[NormalizedMessage]:
    """
    Convert SK ChatHistory → list[NormalizedMessage] via the common translator.

    We shape each SK chat message into the minimal SK dict format expected by
    `MessageTranslator.from_semantic_kernel`, and then later map that to
    Corpus wire format via `MessageTranslator.to_corpus`.
    """
    normalized: List[NormalizedMessage] = []

    # `ChatHistory` is iterable in recent SK versions; if not, we fall back
    # to `history.messages` when present.
    try:
        iterable = list(history)  # type: ignore[arg-type]
    except TypeError:
        iterable = getattr(history, "messages", []) or []

    for msg in iterable:
        role = _author_role_to_protocol_role(getattr(msg, "role", None))
        text = _extract_text_from_chat_message(msg)
        metadata: Mapping[str, Any] = getattr(msg, "metadata", {}) or {}

        minimal: Dict[str, Any] = {
            "role": role,
            "content": text,
            "metadata": dict(metadata),
        }

        nm = MessageTranslator.from_semantic_kernel(minimal)
        normalized.append(nm)

    return normalized


def _normalized_to_corpus_messages(
    messages: List[NormalizedMessage],
) -> List[Dict[str, Any]]:
    """
    NormalizedMessage list → Corpus protocol wire messages.
    """
    return MessageTranslator.to_corpus(messages)


def _log_sampling_param_warnings(params: Mapping[str, Any]) -> None:
    """
    Best-effort validation of SK sampling params.

    We do not enforce ranges here (that's the adapter's job via BaseLLMAdapter),
    but we log when values are clearly outside typical production ranges.
    """
    temp = params.get("temperature")
    if temp is not None and isinstance(temp, (int, float)):
        if not (0.0 <= float(temp) <= 2.0):
            logger.warning(
                "Semantic Kernel settings.temperature=%r outside typical [0.0, 2.0] range",
                temp,
            )

    top_p = params.get("top_p")
    if top_p is not None and isinstance(top_p, (int, float)):
        if not (0.0 < float(top_p) <= 1.0):
            logger.warning(
                "Semantic Kernel settings.top_p=%r outside typical (0.0, 1.0] range",
                top_p,
            )


def _build_ctx_and_sampling_params(
    *,
    settings: PromptExecutionSettings,
    framework_version: Optional[str] = None,
    model_override: Optional[str] = None,
    default_model: str,
    default_temperature: float,
    default_max_tokens: Optional[int],
) -> Tuple[OperationContext, Dict[str, Any]]:
    """
    Build OperationContext and Corpus sampling params from SK prompt settings.
    """
    # Use ContextTranslator to preserve tenant / trace / deadline when possible.
    ctx = ContextTranslator.from_semantic_kernel_context(
        sk_context=None,
        settings=settings,
        framework_version=framework_version,
    )

    # Settings introspection is intentionally conservative and string-based to
    # tolerate minor SK version changes.
    def _get(name: str) -> Optional[Any]:
        return getattr(settings, name, None)

    # Model resolution:
    # 1) explicit override
    # 2) settings.model_id / settings.model / settings.deployment_name
    # 3) adapter default.
    model = (
        model_override
        or _get("model_id")
        or _get("model")
        or _get("deployment_name")
        or default_model
    )

    # Sampling params from settings, falling back to adapter defaults.
    temperature = _get("temperature")
    if temperature is None:
        temperature = default_temperature

    max_tokens = _get("max_tokens")
    if max_tokens is None:
        max_tokens = default_max_tokens

    params: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": _get("top_p"),
        "frequency_penalty": _get("frequency_penalty"),
        "presence_penalty": _get("presence_penalty"),
        # SK often uses `stop_sequences` or `stop` depending on provider.
        "stop_sequences": _get("stop_sequences") or _get("stop"),
    }

    # Light validation / warnings; real enforcement happens in BaseLLMAdapter.
    _log_sampling_param_warnings(params)

    # Strip out Nones so the Corpus adapter sees a clean param dict.
    return ctx, {k: v for k, v in params.items() if v is not None}


def _completion_to_chat_message(
    completion: LLMCompletion,
) -> ChatMessageContent:
    """
    LLMCompletion → SK ChatMessageContent.

    We attach model / finish_reason / usage into metadata for SK-side
    consumers that care about these details.
    """
    metadata: Dict[str, Any] = {
        "model": completion.model,
        "model_family": completion.model_family,
        "finish_reason": completion.finish_reason,
    }
    if completion.usage is not None:
        metadata["usage"] = {
            "prompt_tokens": completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens,
            "total_tokens": completion.usage.total_tokens,
        }

    return ChatMessageContent(
        role=AuthorRole.ASSISTANT,
        content=completion.text,
        metadata=metadata,
    )


def _chunk_to_streaming_chat_message(
    chunk: LLMChunk,
) -> StreamingChatMessageContent:
    """
    LLMChunk → SK StreamingChatMessageContent.

    We treat `chunk.text` as the delta for streaming. Model and usage_so_far
    are exposed via metadata.
    """
    metadata: Dict[str, Any] = {
        "is_final": chunk.is_final,
    }
    if chunk.model is not None:
        metadata["model"] = chunk.model
    if chunk.usage_so_far is not None:
        metadata["usage_so_far"] = {
            "prompt_tokens": chunk.usage_so_far.prompt_tokens,
            "completion_tokens": chunk.usage_so_far.completion_tokens,
            "total_tokens": chunk.usage_so_far.total_tokens,
        }

    return StreamingChatMessageContent(
        role=AuthorRole.ASSISTANT,
        content=chunk.text,
        metadata=metadata,
    )


async def _with_transient_retry(
    coro_func: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    backoff_s: float,
    error_types: Tuple[Type[BaseException], ...],
    error_context: Dict[str, Any],
) -> T:
    """
    Execute an async function with transient error retry.

    Only retries for specified error types and only up to max_retries.
    Uses exponential backoff between attempts.
    """
    attempt = 0
    last_exc: Optional[BaseException] = None

    while True:
        try:
            return await coro_func()
        except error_types as exc:
            last_exc = exc
            if attempt >= max_retries:
                break

            attempt += 1
            delay = backoff_s * (2.0 ** (attempt - 1))
            
            attach_context(
                exc,
                framework="semantic_kernel",
                operation="transient_retry",
                attempt=attempt,
                max_retries=max_retries,
                delay_s=delay,
                **error_context,
            )

            logger.warning(
                "Transient error in Semantic Kernel adapter (attempt %d/%d), "
                "retrying after %.2fs: %s",
                attempt,
                max_retries,
                delay,
                exc,
            )
            
            await asyncio.sleep(delay)
        except BaseException as exc:
            last_exc = exc
            break

    if last_exc is not None:
        attach_context(
            last_exc,
            framework="semantic_kernel",
            operation="final_attempt",
            attempt=attempt,
            max_retries=max_retries,
            **error_context,
        )
        raise last_exc
    
    # This should never happen, but for type safety
    raise RuntimeError("Unexpected error in _with_transient_retry")


# ---------------------------------------------------------------------------
# Public adapter
# ---------------------------------------------------------------------------


class CorpusSemanticKernelChatCompletion(ChatCompletionClientBase):
    """
    Semantic Kernel `ChatCompletionClientBase` backed by a Corpus LLM adapter.

    This class allows SK agents to use any Corpus-backed LLM implementation
    (implementing `BaseLLMAdapter`) as their chat completion service.

    Attributes
    ----------
    corpus_adapter:
        Underlying Corpus adapter implementing `BaseLLMAdapter`.
    model:
        Default model identifier to use when SK settings do not override it.
    temperature:
        Default sampling temperature.
    max_tokens:
        Default maximum tokens for completions (if SK settings do not override).
    framework_version:
        Optional Semantic Kernel version string for context attribution.
    service_id:
        Optional SK service identifier used by the Kernel's service registry.

    max_transient_retries:
        Number of retry attempts for transient errors. Default: 0 (no retry).
    transient_backoff_s:
        Initial backoff delay (seconds) for retry. Uses exponential backoff.
        Default: 0.25.
    transient_error_types:
        Tuple of exception types to consider transient and eligible for retry.
        Default: (TransientNetwork, Unavailable).

    Examples
    --------
    Basic usage:

        sk_llm = CorpusSemanticKernelChatCompletion(
            corpus_adapter=OpenAIAdapter(api_key="..."),
            model="gpt-4",
        )

    With retry enabled:

        sk_llm = CorpusSemanticKernelChatCompletion(
            corpus_adapter=OpenAIAdapter(api_key="..."),
            model="gpt-4",
            max_transient_retries=3,
            transient_backoff_s=0.5,
        )
    """

    def __init__(
        self,
        corpus_adapter: BaseLLMAdapter,
        *,
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        framework_version: Optional[str] = None,
        service_id: Optional[str] = None,
        # Retry configuration
        max_transient_retries: int = 0,
        transient_backoff_s: float = 0.25,
        transient_error_types: Tuple[Type[BaseException], ...] = (
            TransientNetwork,
            Unavailable,
        ),
    ) -> None:
        if _SEMANTIC_KERNEL_IMPORT_ERROR is not None:
            # Give a very direct, one-hop error for misconfiguration.
            raise RuntimeError(
                "semantic-kernel is not installed. "
                "Install it via `pip install semantic-kernel` to use "
                "CorpusSemanticKernelChatCompletion."
            ) from _SEMANTIC_KERNEL_IMPORT_ERROR

        super().__init__(service_id=service_id)

        self._corpus_adapter = corpus_adapter
        self._default_model = model
        self._default_temperature = float(temperature)
        self._default_max_tokens = max_tokens
        self._framework_version = framework_version

        # Retry configuration
        self._max_transient_retries = max(0, int(max_transient_retries))
        self._transient_backoff_s = float(transient_backoff_s)
        self._transient_error_types = transient_error_types

    async def _execute_with_retry(
        self,
        operation: str,
        coro_func: Callable[[], Awaitable[T]],
        context: Dict[str, Any],
    ) -> T:
        """
        Execute an async operation with optional transient error retry.

        Parameters
        ----------
        operation:
            Name of the operation for error context.
        coro_func:
            Async function to execute.
        context:
            Additional context for error reporting.

        Returns
        -------
        T
            Result of the async operation.
        """
        error_context = {
            "operation": operation,
            **context,
        }

        if self._max_transient_retries > 0:
            return await _with_transient_retry(
                coro_func,
                max_retries=self._max_transient_retries,
                backoff_s=self._transient_backoff_s,
                error_types=self._transient_error_types,
                error_context=error_context,
            )
        else:
            try:
                return await coro_func()
            except BaseException as exc:
                attach_context(exc, framework="semantic_kernel", **error_context)
                raise

    # ------------------------------------------------------------------ #
    # Core async API expected by ChatCompletionClientBase
    # ------------------------------------------------------------------ #

    async def get_chat_message_content(
        self,
        chat_history: ChatHistory,
        settings: PromptExecutionSettings,
        **kwargs: Any,
    ) -> ChatMessageContent:
        """
        Execute a single chat completion call via the Corpus adapter.

        Parameters
        ----------
        chat_history:
            Semantic Kernel chat history (messages so far).
        settings:
            Provider-specific or generic prompt execution settings.
        **kwargs:
            Additional provider-specific settings; we look at `model` and
            retry overrides: `max_transient_retries`, `transient_backoff_s`,
            `transient_error_types`.

        Returns
        -------
        ChatMessageContent
            SK chat message content with assistant response.
        """
        # Extract per-call retry overrides
        max_retries_override = kwargs.pop("max_transient_retries", None)
        backoff_override = kwargs.pop("transient_backoff_s", None)
        error_types_override = kwargs.pop("transient_error_types", None)

        normalized = _history_to_normalized_messages(chat_history)
        corpus_messages = _normalized_to_corpus_messages(normalized)

        ctx, params = _build_ctx_and_sampling_params(
            settings=settings,
            framework_version=self._framework_version,
            model_override=kwargs.get("model"),
            default_model=self._default_model,
            default_temperature=self._default_temperature,
            default_max_tokens=self._default_max_tokens,
        )

        async def _complete() -> ChatMessageContent:
            completion = await self._corpus_adapter.complete(
                messages=corpus_messages,
                ctx=ctx,
                **params,
            )
            return _completion_to_chat_message(completion)

        # Use overrides if provided, otherwise instance defaults
        max_retries = (
            max_retries_override if max_retries_override is not None 
            else self._max_transient_retries
        )
        backoff_s = (
            backoff_override if backoff_override is not None 
            else self._transient_backoff_s
        )
        error_types = (
            error_types_override if error_types_override is not None 
            else self._transient_error_types
        )

        context = {
            "messages_count": len(corpus_messages),
            "model": params.get("model", self._default_model),
            "temperature": params.get("temperature"),
            "max_tokens": params.get("max_tokens"),
            "top_p": params.get("top_p"),
            "frequency_penalty": params.get("frequency_penalty"),
            "presence_penalty": params.get("presence_penalty"),
            "request_id": getattr(ctx, "request_id", None),
            "tenant": getattr(ctx, "tenant", None),
            "stream": False,
            "max_transient_retries": max_retries,
            "transient_backoff_s": backoff_s,
        }

        if error_types:
            context["transient_error_types"] = [t.__name__ for t in error_types]

        if max_retries > 0:
            return await _with_transient_retry(
                _complete,
                max_retries=max_retries,
                backoff_s=backoff_s,
                error_types=error_types,
                error_context=context,
            )
        else:
            try:
                return await _complete()
            except BaseException as exc:
                attach_context(exc, framework="semantic_kernel", **context)
                raise

    async def get_chat_message_contents(
        self,
        chat_history: ChatHistory,
        settings: PromptExecutionSettings,
        **kwargs: Any,
    ) -> List[ChatMessageContent]:
        """
        Convenience wrapper returning a single-element list for SK APIs that
        expect a collection of messages.
        """
        msg = await self.get_chat_message_content(chat_history, settings, **kwargs)
        return [msg]

    async def get_streaming_chat_message_content(
        self,
        chat_history: ChatHistory,
        settings: PromptExecutionSettings,
        **kwargs: Any,
    ) -> AsyncIterator[StreamingChatMessageContent]:
        """
        Streaming chat completion via Corpus `stream()`.

        Yields Semantic Kernel `StreamingChatMessageContent` objects for each
        chunk produced by the Corpus adapter.

        Parameters
        ----------
        chat_history:
            Semantic Kernel chat history.
        settings:
            Prompt execution settings.
        **kwargs:
            Additional settings including retry overrides.

        Yields
        ------
        StreamingChatMessageContent
            Streaming chat message content chunks.
        """
        # Extract per-call retry overrides (for the initial stream creation)
        max_retries_override = kwargs.pop("max_transient_retries", None)
        backoff_override = kwargs.pop("transient_backoff_s", None)
        error_types_override = kwargs.pop("transient_error_types", None)

        normalized = _history_to_normalized_messages(chat_history)
        corpus_messages = _normalized_to_corpus_messages(normalized)

        ctx, params = _build_ctx_and_sampling_params(
            settings=settings,
            framework_version=self._framework_version,
            model_override=kwargs.get("model"),
            default_model=self._default_model,
            default_temperature=self._default_temperature,
            default_max_tokens=self._default_max_tokens,
        )

        context = {
            "operation": "get_streaming_chat_message_content",
            "messages_count": len(corpus_messages),
            "model": params.get("model", self._default_model),
            "temperature": params.get("temperature"),
            "max_tokens": params.get("max_tokens"),
            "top_p": params.get("top_p"),
            "frequency_penalty": params.get("frequency_penalty"),
            "presence_penalty": params.get("presence_penalty"),
            "request_id": getattr(ctx, "request_id", None),
            "tenant": getattr(ctx, "tenant", None),
            "stream": True,
        }

        # Use overrides if provided, otherwise instance defaults
        max_retries = (
            max_retries_override if max_retries_override is not None 
            else self._max_transient_retries
        )
        backoff_s = (
            backoff_override if backoff_override is not None 
            else self._transient_backoff_s
        )
        error_types = (
            error_types_override if error_types_override is not None 
            else self._transient_error_types
        )

        async def _create_stream() -> AsyncIterator[LLMChunk]:
            return await self._corpus_adapter.stream(
                messages=corpus_messages,
                ctx=ctx,
                **params,
            )

        # Retry only the stream creation, not individual chunks
        if max_retries > 0:
            stream_context = context.copy()
            stream_context.update({
                "max_transient_retries": max_retries,
                "transient_backoff_s": backoff_s,
            })
            if error_types:
                stream_context["transient_error_types"] = [t.__name__ for t in error_types]

            stream = await _with_transient_retry(
                _create_stream,
                max_retries=max_retries,
                backoff_s=backoff_s,
                error_types=error_types,
                error_context=stream_context,
            )
        else:
            try:
                stream = await _create_stream()
            except BaseException as exc:
                attach_context(exc, framework="semantic_kernel", **context)
                raise

        try:
            async for chunk in stream:
                yield _chunk_to_streaming_chat_message(chunk)
        except BaseException as exc:
            attach_context(exc, framework="semantic_kernel", **context)
            raise
        finally:
            # Best-effort cleanup for adapters that implement aclose() on streams.
            aclose = getattr(stream, "aclose", None)
            if callable(aclose):
                try:
                    await aclose()
                except Exception as cleanup_exc:  # pragma: no cover - extremely defensive
                    logger.debug(
                        "CorpusSemanticKernelChatCompletion: stream cleanup failed: %s",
                        cleanup_exc,
                    )

    async def get_streaming_chat_message_contents(
        self,
        chat_history: ChatHistory,
        settings: PromptExecutionSettings,
        **kwargs: Any,
    ) -> AsyncIterator[StreamingChatMessageContent]:
        """
        Convenience wrapper providing an alias returning an async iterator of
        streaming messages. Recent SK versions favor the singular name, but
        some APIs may expect the plural form.
        """
        async for msg in self.get_streaming_chat_message_content(
            chat_history,
            settings,
            **kwargs,
        ):
            yield msg


__all__ = [
    "CorpusSemanticKernelChatCompletion",
]
