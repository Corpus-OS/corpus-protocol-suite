# corpus_sdk/llm/framework_adapters/llamaindex.py
# SPDX-License-Identifier: Apache-2.0

"""
LlamaIndex adapter for Corpus LLM protocol.

This module exposes a Corpus `LLMProtocolV1` as a LlamaIndex `LLM`
implementation, with:

- Async + sync chat generation
- Async + sync streaming chat (true incremental streaming)
- Context propagation via `OperationContext`
- Protocol-first design: all calls go through the LLMTranslator layer
- Rich, configurable error context for observability
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    Mapping,
)

from corpus_sdk.core.context_translation import (
    from_llamaindex as context_from_llamaindex,
)
from corpus_sdk.llm.llm_base import (
    BadRequest,
    LLMProtocolV1,
    OperationContext,
)
from corpus_sdk.llm.framework_adapters.common.error_context import attach_context
from corpus_sdk.llm.framework_adapters.common.llm_translation import (
    LLMTranslator,
    create_llm_translator,
    get_llm_translator_factory,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional LlamaIndex imports
# ---------------------------------------------------------------------------

_LLAMAINDEX_IMPORT_ERROR: Optional[BaseException] = None

try:  # pragma: no cover - optional dependency path
    from llama_index.core.llms import (
        LLM,
        ChatMessage,
        ChatResponse,
        ChatResponseAsyncGen,
        ChatResponseGen,
        MessageRole,
        LLMMetadata,
    )
    from llama_index.core.callbacks import CallbackManager
except BaseException as exc:  # pragma: no cover
    _LLAMAINDEX_IMPORT_ERROR = exc
    # Fallbacks for type checking and import-time safety
    LLM = object  # type: ignore[assignment]
    ChatMessage = object  # type: ignore[assignment]
    ChatResponse = object  # type: ignore[assignment]
    ChatResponseAsyncGen = AsyncIterator[Any]  # type: ignore[assignment]
    ChatResponseGen = Iterator[Any]  # type: ignore[assignment]
    MessageRole = object  # type: ignore[assignment]
    LLMMetadata = object  # type: ignore[assignment]
    CallbackManager = object  # type: ignore[assignment]


def _ensure_llamaindex_installed() -> None:
    """Raise helpful error if LlamaIndex is not installed."""
    if _LLAMAINDEX_IMPORT_ERROR is not None:
        raise RuntimeError(
            "LlamaIndex is required to use CorpusLlamaIndexLLM. "
            "Install with: pip install 'llama-index-core>=0.10.0'"
        ) from _LLAMAINDEX_IMPORT_ERROR


class LlamaIndexLLMProtocol(Protocol):
    """Structural protocol for LlamaIndex-compatible Corpus LLM."""

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        ...

    async def astream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        ...

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        ...

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseGen:
        ...

    def count_tokens(self, messages: Sequence[ChatMessage], **kwargs: Any) -> int:
        ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LlamaIndexLLMConfig:
    """
    Configuration for the LlamaIndex adapter.

    Attributes
    ----------
    enable_error_context:
        Whether to attach rich error context with LlamaIndex-specific metadata.

    include_message_metrics_on_error:
        Whether to include per-message metrics (role distribution, char counts)
        when attaching error context. This is evaluated lazily *only on errors*.

    context_window:
        Optional override for the model context window, in tokens. When set,
        it is exposed via the LlamaIndex `metadata` property so that RAG
        components can reason about available context size.
    """

    enable_error_context: bool = True
    include_message_metrics_on_error: bool = True
    context_window: Optional[int] = None

    def __post_init__(self) -> None:
        if self.context_window is not None and self.context_window < 1:
            raise ValueError("context_window must be a positive integer when provided")


# ---------------------------------------------------------------------------
# ChatResponse builders
# ---------------------------------------------------------------------------


def _usage_to_dict(usage: Any) -> Dict[str, Optional[int]]:
    """
    Normalize usage objects into a simple dict.

    Supports either:
    - objects with .prompt_tokens / .completion_tokens / .total_tokens
    - mappings with those keys
    """
    if usage is None:
        return {}

    if isinstance(usage, Mapping):
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
    else:
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)

    result: Dict[str, Optional[int]] = {
        "prompt_tokens": int(prompt_tokens) if isinstance(prompt_tokens, int) else None,
        "completion_tokens": int(completion_tokens) if isinstance(completion_tokens, int) else None,
        "total_tokens": int(total_tokens) if isinstance(total_tokens, int) else None,
    }
    # Strip all-None payloads
    return {k: v for k, v in result.items() if v is not None}


def _build_chat_response(
    text: str,
    *,
    model: Optional[str] = None,
    finish_reason: Optional[str] = None,
    usage: Optional[Any] = None,
) -> ChatResponse:
    """
    Construct a minimal LlamaIndex ChatResponse from completion-like data.

    The caller is responsible for extracting the text/model/usage/finish_reason
    from whatever the LLMTranslator returned.
    """
    try:
        role_assistant = getattr(MessageRole, "ASSISTANT", "assistant")
    except Exception:
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

    usage_dict = _usage_to_dict(usage)
    if usage_dict:
        additional_kwargs["usage"] = usage_dict

    try:
        return ChatResponse(message=msg, raw=None, additional_kwargs=additional_kwargs)
    except TypeError:
        # Older versions may not support additional_kwargs
        return ChatResponse(message=msg)


def _extract_completion_fields(
    result: Any,
) -> tuple[str, Optional[str], Optional[str], Optional[Any]]:
    """
    Extract (text, model, finish_reason, usage) from a completion-like object
    returned by LLMTranslator.from_completion.

    Supports:
    - Mapping results: {"text", "model", "finish_reason", "usage"}
    - Objects with .text / .model / .finish_reason / .usage
    - Fallback: treat as plain text.
    """
    # Mapping-style (default LLMTranslator behavior)
    if isinstance(result, Mapping):
        text = str(
            result.get("text")
            or result.get("content")
            or result.get("message", {}).get("content", "")
            or ""
        )
        model = result.get("model")
        finish_reason = result.get("finish_reason")
        usage = result.get("usage")
        return text, model, finish_reason, usage

    # Attribute-style objects (custom translators)
    if hasattr(result, "text"):
        text = str(getattr(result, "text", "") or "")
        model = getattr(result, "model", None)
        finish_reason = getattr(result, "finish_reason", None)
        usage = getattr(result, "usage", None)
        return text, model, finish_reason, usage

    # Fallback: treat as plain text
    return str(result), None, None, None


def _build_chat_response_from_translator_result(result: Any) -> ChatResponse:
    """
    Build ChatResponse from whatever LLMTranslator returned for a completion.
    """
    text, model, finish_reason, usage = _extract_completion_fields(result)
    return _build_chat_response(
        text=text,
        model=model,
        finish_reason=finish_reason,
        usage=usage,
    )


def _extract_chunk_fields(
    chunk_obj: Any,
) -> tuple[str, Optional[str], Optional[Any], bool]:
    """
    Extract (text, model, usage_so_far, is_final) from streaming chunk-like objects
    returned by LLMTranslator.from_chunk.

    Supports:
    - Mapping results: {"text", "model", "usage_so_far", "is_final"}
    - Objects with .text / .model / .usage_so_far / .is_final
    - Fallbacks using .delta or treating as plain text.
    """
    # Mapping-style (default LLMTranslator.from_chunk)
    if isinstance(chunk_obj, Mapping):
        text = str(chunk_obj.get("text") or chunk_obj.get("delta") or "")
        model = chunk_obj.get("model")
        usage = chunk_obj.get("usage_so_far") or chunk_obj.get("usage")
        is_final = bool(chunk_obj.get("is_final", False))
        return text, model, usage, is_final

    # Attribute-style
    text = str(
        getattr(chunk_obj, "text", "")
        or getattr(chunk_obj, "delta", "")
        or ""
    )
    model = getattr(chunk_obj, "model", None)
    usage = getattr(chunk_obj, "usage_so_far", None)
    is_final = bool(getattr(chunk_obj, "is_final", False))
    return text, model, usage, is_final


def _build_chat_response_from_chunk_like(chunk_obj: Any) -> ChatResponse:
    """
    Build a ChatResponse for streaming from a single chunk-like object.
    """
    try:
        role_assistant = getattr(MessageRole, "ASSISTANT", "assistant")
    except Exception:
        role_assistant = "assistant"

    text, model, usage, is_final = _extract_chunk_fields(chunk_obj)

    msg = ChatMessage(
        role=role_assistant,
        content=text,
    )

    delta = text or ""

    additional_kwargs: Dict[str, Any] = {
        "is_final": is_final,
        "model": model,
    }

    usage_dict = _usage_to_dict(usage)
    if usage_dict:
        additional_kwargs["usage_so_far"] = usage_dict

    try:
        return ChatResponse(
            message=msg,
            delta=delta,
            raw=None,
            additional_kwargs=additional_kwargs,
        )
    except TypeError:
        # Older versions without delta/additional_kwargs
        return ChatResponse(message=msg)


# ---------------------------------------------------------------------------
# Internal metrics helpers for observability
# ---------------------------------------------------------------------------


def _analyze_messages_for_context(messages: Sequence[ChatMessage]) -> Dict[str, Any]:
    """
    Derive role distribution and content metrics for error context.

    Only used lazily when an error actually occurs to avoid overhead on
    successful calls.
    """
    roles: Dict[str, int] = {}
    total_chars = 0

    for msg in messages:
        # LlamaIndex messages may expose `role` or `type`
        role = getattr(msg, "role", getattr(msg, "type", "unknown"))
        content = getattr(msg, "content", "")
        roles[role] = roles.get(role, 0) + 1
        if isinstance(content, str):
            total_chars += len(content)

    return {
        "messages_count": len(messages),
        "roles_distribution": roles,
        "total_content_chars": total_chars,
    }


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------


class CorpusLlamaIndexLLM(LLM):
    """
    LlamaIndex `LLM` implementation backed by a Corpus `LLMProtocolV1`.

    This integration is fully aligned with the LLMTranslator layer:

    - All LLM calls go through `LLMTranslator` (no direct message_translation
      or protocol calls in this file).
    - Message normalization, system message handling, tools, safety, and
      post-processing are handled centrally by the translator.
    - This file focuses on:
        * LlamaIndex-specific context building
        * Error-context attachment
        * Mapping translator results to `ChatResponse` / streaming responses.
    """

    llm_adapter: LLMProtocolV1
    model: str = "default"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    framework_version: Optional[str] = None
    config: Optional[LlamaIndexLLMConfig] = None
    # Optional explicit context window override; surfaced via metadata.
    context_window: Optional[int] = None

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data: Any) -> None:
        _ensure_llamaindex_installed()

        if "llm_adapter" in data and not hasattr(data["llm_adapter"], "complete"):
            raise TypeError("llm_adapter must implement LLMProtocolV1")

        if "temperature" in data and not 0 <= data["temperature"] <= 2:
            raise ValueError("temperature must be between 0 and 2")

        if (
            "max_tokens" in data
            and data.get("max_tokens") is not None
            and data["max_tokens"] < 1
        ):
            raise ValueError("max_tokens must be positive")

        super().__init__(**data)

        # Concrete config instance (immutable dataclass).
        # Precedence is simple: if the user supplied a LlamaIndexLLMConfig,
        # use it as-is; otherwise use defaults.
        self._config: LlamaIndexLLMConfig = (
            self.config
            if isinstance(self.config, LlamaIndexLLMConfig)
            else LlamaIndexLLMConfig()
        )

        # If config specifies a context_window, make sure the instance exposes it.
        if self._config.context_window is not None:
            self.context_window = self._config.context_window

        # Fail fast if there is no registered translator factory for LlamaIndex.
        # This prevents silently falling back to the default translator, which
        # may not know how to handle ChatMessage objects.
        factory = get_llm_translator_factory("llamaindex")
        if factory is None:
            raise RuntimeError(
                "No LLMFrameworkTranslator registered for framework='llamaindex'. "
                "Call register_llm_translator('llamaindex', factory) before "
                "constructing CorpusLlamaIndexLLM."
            )

        # LLMTranslator orchestrator for the LlamaIndex framework.
        self._translator: LLMTranslator = create_llm_translator(
            adapter=self.llm_adapter,
            framework="llamaindex",
        )

        logger.info(
            "CorpusLlamaIndexLLM initialized with model=%s, temperature=%.2f, max_tokens=%s",
            self.model,
            self.temperature,
            self.max_tokens or "default",
        )

    # ------------------------------------------------------------------ #
    # LlamaIndex-required metadata
    # ------------------------------------------------------------------ #

    @property
    def metadata(self) -> LLMMetadata:
        """
        LlamaIndex metadata describing this LLM's capabilities.

        This is critical for RAG components (chunking, query planning, etc.)
        to understand the effective context window and output length.
        """
        # Prefer an explicit context_window if configured; otherwise fall back
        # to a conservative default (LlamaIndex will clamp as needed).
        ctx_window = self.context_window
        if not isinstance(ctx_window, int) or ctx_window <= 0:
            ctx_window = 2048

        # num_output should be an upper bound on expected completion length.
        # Use max_tokens when provided, otherwise a safe default.
        if isinstance(self.max_tokens, int) and self.max_tokens > 0:
            num_output = self.max_tokens
        else:
            num_output = 256

        model_name = self.model or "unknown"

        system_role = getattr(MessageRole, "SYSTEM", None)
        if system_role is None:
            system_role = "system"

        return LLMMetadata(
            context_window=ctx_window,
            num_output=num_output,
            is_chat_model=True,
            is_function_calling_model=False,
            model_name=model_name,
            system_role=system_role,
        )

    # ------------------------------------------------------------------ #
    # Error context management
    # ------------------------------------------------------------------ #

    def _build_error_context(
        self,
        operation: str,
        stream: bool,
        messages: Sequence[ChatMessage],
        model: str,
        ctx: OperationContext,
        params: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Build consistent, rich error context for observability.

        Metrics on messages are computed lazily and only when enabled
        in config and when an error actually occurs.
        """
        base_ctx: Dict[str, Any] = {
            "framework": "llamaindex",
            "resource_type": "llm",
            "operation": operation,
            "model": model,
            "temperature": params.get("temperature"),
            "max_tokens": params.get("max_tokens"),
            "top_p": params.get("top_p"),
            "frequency_penalty": params.get("frequency_penalty"),
            "presence_penalty": params.get("presence_penalty"),
            "stop_sequences": params.get("stop_sequences"),
            "request_id": ctx.request_id,
            "tenant": ctx.tenant,
            "stream": stream,
        }

        if self._config.include_message_metrics_on_error:
            msg_metrics = _analyze_messages_for_context(messages)
            base_ctx.update(msg_metrics)

        return base_ctx

    @contextmanager
    def _error_context(
        self,
        operation: str,
        stream: bool,
        messages: Sequence[ChatMessage],
        model: str,
        ctx: OperationContext,
        params: Mapping[str, Any],
    ):
        """Sync error-context wrapper with lazy metrics."""
        try:
            yield
        except BaseException as exc:  # noqa: BLE001
            if self._config.enable_error_context:
                error_ctx = self._build_error_context(
                    operation,
                    stream,
                    messages,
                    model,
                    ctx,
                    params,
                )
                attach_context(exc, **error_ctx)
            raise

    @asynccontextmanager
    async def _error_context_async(
        self,
        operation: str,
        stream: bool,
        messages: Sequence[ChatMessage],
        model: str,
        ctx: OperationContext,
        params: Mapping[str, Any],
    ):
        """Async error-context wrapper with lazy metrics."""
        try:
            yield
        except BaseException as exc:  # noqa: BLE001
            if self._config.enable_error_context:
                error_ctx = self._build_error_context(
                    operation,
                    stream,
                    messages,
                    model,
                    ctx,
                    params,
                )
                attach_context(exc, **error_ctx)
            raise

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_operation_context(self, kwargs: Mapping[str, Any]) -> OperationContext:
        """
        Build OperationContext from LlamaIndex kwargs via context_translation.

        This keeps all LLM-level context building centralized in the core
        context translators instead of baking assumptions into this adapter.
        """
        callback_manager = kwargs.get("callback_manager")
        return context_from_llamaindex(
            callback_manager,
            framework_version=self.framework_version,
        )

    def _extract_stop_sequences(self, kwargs: Mapping[str, Any]) -> Optional[List[str]]:
        """
        Extract stop sequences from kwargs (handles multiple LlamaIndex shapes).

        Checks:
        - kwargs["stop"]
        - kwargs["stop_sequences"]
        - kwargs["additional_kwargs"]["stop" or "stop_sequences"]
        """
        stop: Any = kwargs.get("stop") or kwargs.get("stop_sequences")

        additional = kwargs.get("additional_kwargs")
        if stop is None and isinstance(additional, Mapping):
            stop = additional.get("stop") or additional.get("stop_sequences")

        if stop is None:
            return None
        if isinstance(stop, str):
            stop = stop.strip()
            return [stop] if stop else None
        if isinstance(stop, (list, tuple)):
            values = [str(s).strip() for s in stop if str(s).strip()]
            return values or None
        return [str(stop)]

    def _build_sampling_params(
        self,
        kwargs: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Build sampling parameters from kwargs, with validation.

        This is the single source of truth for how we derive sampling params
        (including stop sequences) from LlamaIndex's kwargs.
        """
        stop_sequences = self._extract_stop_sequences(kwargs)

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
            "stop_sequences": stop_sequences,
        }
        # Strip None for a clean param set
        clean = {k: v for k, v in params.items() if v is not None}

        logger.debug(
            "Built sampling params for LlamaIndex: model=%s, temperature=%.2f, max_tokens=%s",
            clean.get("model", self.model),
            clean.get("temperature", self.temperature),
            clean.get("max_tokens", self.max_tokens),
        )

        return clean

    def _build_framework_ctx(
        self,
        kwargs: Mapping[str, Any],
        *,
        operation: str,
        stream: bool,
    ) -> Dict[str, Any]:
        """
        Build a small framework_ctx payload for the LLMTranslator.

        This allows the framework-specific LLMFrameworkTranslator to see
        LlamaIndex-specific metadata and operation context if needed.
        """
        return {
            "framework": "llamaindex",
            "framework_version": self.framework_version,
            "operation": operation,
            "stream": stream,
            "callback_manager": kwargs.get("callback_manager"),
            "llamaindex_additional_kwargs": kwargs.get("additional_kwargs"),
        }

    def _build_request_context(
        self,
        kwargs: Mapping[str, Any],
        *,
        operation: str,
        stream: bool,
    ) -> tuple[OperationContext, Dict[str, Any], str, Dict[str, Any]]:
        """
        Unified helper to construct:
        - OperationContext
        - sampling params
        - model_for_context
        - framework_ctx
        """
        ctx = self._build_operation_context(kwargs)
        params = self._build_sampling_params(kwargs)
        model_for_context = params.get("model", self.model)
        framework_ctx = self._build_framework_ctx(
            kwargs,
            operation=operation,
            stream=stream,
        )
        return ctx, params, model_for_context, framework_ctx

    # ------------------------------------------------------------------ #
    # Async API
    # ------------------------------------------------------------------ #

    async def achat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        """
        Async chat entrypoint required by LlamaIndex.

        The flow is:
        - Build OperationContext from LlamaIndex callback/context info.
        - Derive sampling params from kwargs.
        - Call LLMTranslator.arun_complete with raw LlamaIndex messages.
        - Convert translator result into ChatResponse.
        """
        if not messages:
            raise BadRequest(
                "Messages list cannot be empty",
                code="BAD_MESSAGES",
            )

        ctx, params, model_for_context, framework_ctx = self._build_request_context(
            kwargs,
            operation="achat",
            stream=False,
        )

        async with self._error_context_async(
            "achat",
            stream=False,
            messages=messages,
            model=model_for_context,
            ctx=ctx,
            params=params,
        ):
            result = await self._translator.arun_complete(
                raw_messages=messages,
                model=params.get("model"),
                max_tokens=params.get("max_tokens"),
                temperature=params.get("temperature"),
                top_p=params.get("top_p"),
                frequency_penalty=params.get("frequency_penalty"),
                presence_penalty=params.get("presence_penalty"),
                stop_sequences=params.get("stop_sequences"),
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

            return _build_chat_response_from_translator_result(result)

    async def astream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        """
        Async streaming chat entrypoint required by LlamaIndex.

        All streaming goes through `LLMTranslator.arun_stream`, and each
        streaming chunk is converted into a `ChatResponse`.
        """
        if not messages:
            raise BadRequest(
                "Messages list cannot be empty",
                code="BAD_MESSAGES",
            )

        ctx, params, model_for_context, framework_ctx = self._build_request_context(
            kwargs,
            operation="astream_chat",
            stream=True,
        )

        async with self._error_context_async(
            "astream_chat",
            stream=True,
            messages=messages,
            model=model_for_context,
            ctx=ctx,
            params=params,
        ):
            async def _gen() -> AsyncIterator[ChatResponse]:
                agen = await self._translator.arun_stream(
                    raw_messages=messages,
                    model=params.get("model"),
                    max_tokens=params.get("max_tokens"),
                    temperature=params.get("temperature"),
                    top_p=params.get("top_p"),
                    frequency_penalty=params.get("frequency_penalty"),
                    presence_penalty=params.get("presence_penalty"),
                    stop_sequences=params.get("stop_sequences"),
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )

                try:
                    async for chunk_obj in agen:
                        yield _build_chat_response_from_chunk_like(chunk_obj)
                finally:
                    # Explicit cleanup of async generator if supported
                    aclose = getattr(agen, "aclose", None)
                    if callable(aclose):
                        try:
                            await aclose()
                        except Exception as close_exc:  # noqa: BLE001
                            logger.debug(
                                "Failed to close async stream iterator: %s",
                                close_exc,
                            )

            return _gen()

    # ------------------------------------------------------------------ #
    # Sync API
    # ------------------------------------------------------------------ #

    def chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        """
        Sync chat entrypoint required by LlamaIndex.

        Uses LLMTranslator.complete under the hood (sync bridge).
        """
        if not messages:
            raise BadRequest(
                "Messages list cannot be empty",
                code="BAD_MESSAGES",
            )

        ctx, params, model_for_context, framework_ctx = self._build_request_context(
            kwargs,
            operation="chat",
            stream=False,
        )

        with self._error_context(
            "chat",
            stream=False,
            messages=messages,
            model=model_for_context,
            ctx=ctx,
            params=params,
        ):
            result = self._translator.complete(
                raw_messages=messages,
                model=params.get("model"),
                max_tokens=params.get("max_tokens"),
                temperature=params.get("temperature"),
                top_p=params.get("top_p"),
                frequency_penalty=params.get("frequency_penalty"),
                presence_penalty=params.get("presence_penalty"),
                stop_sequences=params.get("stop_sequences"),
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

            return _build_chat_response_from_translator_result(result)

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseGen:
        """
        Sync streaming chat entrypoint required by LlamaIndex.

        Uses LLMTranslator.stream, which bridges async streaming into sync.
        """
        if not messages:
            raise BadRequest(
                "Messages list cannot be empty",
                code="BAD_MESSAGES",
            )

        ctx, params, model_for_context, framework_ctx = self._build_request_context(
            kwargs,
            operation="stream_chat",
            stream=True,
        )

        with self._error_context(
            "stream_chat",
            stream=True,
            messages=messages,
            model=model_for_context,
            ctx=ctx,
            params=params,
        ):
            def _gen() -> Iterator[ChatResponse]:
                stream_iter = self._translator.stream(
                    raw_messages=messages,
                    model=params.get("model"),
                    max_tokens=params.get("max_tokens"),
                    temperature=params.get("temperature"),
                    top_p=params.get("top_p"),
                    frequency_penalty=params.get("frequency_penalty"),
                    presence_penalty=params.get("presence_penalty"),
                    stop_sequences=params.get("stop_sequences"),
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )

                try:
                    for chunk_obj in stream_iter:
                        yield _build_chat_response_from_chunk_like(chunk_obj)
                finally:
                    # Explicit cleanup of sync iterator if supported
                    close = getattr(stream_iter, "close", None)
                    if callable(close):
                        try:
                            close()
                        except Exception as close_exc:  # noqa: BLE001
                            logger.debug(
                                "Failed to close sync stream iterator: %s",
                                close_exc,
                            )

            return _gen()

    # ------------------------------------------------------------------ #
    # Token counting
    # ------------------------------------------------------------------ #

    def count_tokens(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> int:
        """
        Token counting helper.

        Preferred path:
        - Use LLMTranslator.count_tokens_for_messages so token counting
          can use the same formatting and strategies as actual completions.

        Fallbacks:
        - If translator/adapter count fails, use improved char-based estimate.
        - If that fails, use the simple character-based heuristic.
        """
        if not messages:
            return 0

        ctx, params, model_for_context, framework_ctx = self._build_request_context(
            kwargs,
            operation="count_tokens",
            stream=False,
        )

        # Preferred: translator-based token counting
        try:
            tokens_any = self._translator.count_tokens_for_messages(
                raw_messages=messages,
                model=model_for_context,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            return int(tokens_any)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Translator-based token counting failed: %s", exc)

        # Fall back to improved character-based estimate
        try:
            return self._estimate_tokens_from_messages(messages)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Token estimation failed: %s", exc)

        # Ultimate fallback to simple character count
        return self._simple_token_estimate(messages)

    def _combine_messages_for_counting(
        self,
        messages: Sequence[ChatMessage],
    ) -> str:
        """Combine messages into a single string for token counting."""
        parts: List[str] = []
        for msg in messages:
            role = getattr(msg, "type", getattr(msg, "role", "user"))
            content = str(getattr(msg, "content", ""))
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def _estimate_tokens_from_messages(
        self,
        messages: Sequence[ChatMessage],
    ) -> int:
        """Improved token estimation with better heuristics."""
        combined_text = self._combine_messages_for_counting(messages)

        if not combined_text:
            return 0

        char_count = len(combined_text)
        message_count = len(messages)

        char_based = max(1, char_count // 4)
        message_based = max(1, message_count)

        return max(char_based, message_based)

    def _simple_token_estimate(
        self,
        messages: Sequence[ChatMessage],
    ) -> int:
        """Simple fallback token estimation."""
        combined_text = self._combine_messages_for_counting(messages)
        return max(1, len(combined_text) // 4)


__all__ = [
    "LlamaIndexLLMProtocol",
    "CorpusLlamaIndexLLM",
    "LlamaIndexLLMConfig",
]
