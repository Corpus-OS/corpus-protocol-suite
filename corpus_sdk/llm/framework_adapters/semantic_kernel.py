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
    )

    agent = ChatCompletionAgent(
        service=sk_llm,
        name="MyAgent",
        instructions="You are a helpful assistant.",
    )
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict, List, Mapping, Optional, Tuple

from corpus_sdk.llm.llm_base import (
    BaseLLMAdapter,
    LLMChunk,
    LLMCompletion,
    OperationContext,
)
from corpus_sdk.llm.framework_adapters.common.context_translation import ContextTranslator
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
            Additional provider-specific settings; we only look at `model`.
        """
        try:
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

            completion = await self._corpus_adapter.complete(
                messages=corpus_messages,
                ctx=ctx,
                **params,
            )

            return _completion_to_chat_message(completion)

        except (TimeoutError, OSError) as exc:
            # Network-ish or timeout-ish failures get logged with a clearer tag.
            logger.warning(
                "CorpusSemanticKernelChatCompletion.get_chat_message_content network/timeout "
                "failure: %s",
                exc,
            )
            raise
        except Exception as exc:
            logger.exception(
                "CorpusSemanticKernelChatCompletion.get_chat_message_content failed: %s",
                exc,
            )
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
        """
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

        stream = await self._corpus_adapter.stream(
            messages=corpus_messages,
            ctx=ctx,
            **params,
        )

        try:
            async for chunk in stream:
                yield _chunk_to_streaming_chat_message(chunk)
        except (TimeoutError, OSError) as exc:
            logger.warning(
                "CorpusSemanticKernelChatCompletion.get_streaming_chat_message_content "
                "network/timeout failure: %s",
                exc,
            )
            raise
        except Exception as exc:
            logger.exception(
                "CorpusSemanticKernelChatCompletion.get_streaming_chat_message_content "
                "failed: %s",
                exc,
            )
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
