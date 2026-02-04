# corpus_sdk/llm/framework_adapters/llamaindex.py
# SPDX-License-Identifier: Apache-2.0

"""
LlamaIndex adapter for Corpus LLM protocol.

This module exposes a Corpus `LLMProtocolV1` implementation as a LlamaIndex
`LLM`, fully wired through the shared `LLMTranslator` layer with:

- Async + sync chat APIs (`achat` / `chat`)
- Async + sync streaming chat (`astream_chat` / `stream_chat`)
- Centralized token counting via `LLMTranslator.count_tokens_for_messages`
- OperationContext propagation from LlamaIndex callbacks / config
- Rich, configurable error context aligned with other LLM framework adapters

Key responsibilities
--------------------
- Accept LlamaIndex `ChatMessage` sequences and pass them through the
  framework-agnostic `LLMTranslator` (no direct protocol calls in this file).
- Build `OperationContext` from LlamaIndex callback state via
  `context_from_llamaindex(...)`.
- Derive sampling parameters (model, temperature, max_tokens, stop, etc.)
  from LlamaIndex kwargs in a single, well-defined place.
- Delegate all completion / streaming orchestration to `LLMTranslator`
  for both sync and async flows.
- Convert translator-level results and stream chunks into LlamaIndex
  `ChatResponse` / streaming `ChatResponse` generators.
- Attach structured, LlamaIndex-flavored error context using the shared
  error-context utilities, with lazy message metrics computed only on error.

Design principles
-----------------
- Protocol-first:
    `LLMProtocolV1` is the source of truth; this adapter is a thin,
    LlamaIndex-oriented skin over the shared `LLMTranslator` stack.

- Translator-centric:
    Message normalization, provider-specific quirks, safety, JSON repair,
    and token usage shaping all live in `LLMTranslator` and its registered
    `LLMFrameworkTranslator` for `framework="llamaindex"`.

- Low-friction integration:
    The class behaves like a normal LlamaIndex `LLM`, exposing metadata
    (`LLMMetadata`) with context window and num_output hints so RAG components
    can reason about chunk sizes and planning.

- Observability without hot-path cost:
    Error context includes model, sampling params, request identifiers,
    and (optionally) message metrics, but those metrics are only computed
    when an exception actually occurs.

Non-responsibilities
--------------------
- Provider-level retries, circuit breaking, or caching – these should live
  in the underlying `LLMProtocolV1` adapter or a higher-level service layer.
- Business logic, routing, or tool wiring – those belong in application code
  or separate orchestration modules, not in this adapter.
- Direct manipulation of LlamaIndex graphs, indexes, or query engines – this
  file is purely about LLM behavior behind the `LLM` interface.
"""

from __future__ import annotations

import asyncio
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
from corpus_sdk.core.error_context import attach_context
from corpus_sdk.llm.llm_base import (
    BadRequest,
    LLMProtocolV1,
    OperationContext,
)
from corpus_sdk.llm.framework_adapters.common.llm_translation import (
    LLMTranslator,
    create_llm_translator,
    get_llm_translator_factory,
)
from corpus_sdk.llm.framework_adapters.common.framework_utils import (
    CoercionErrorCodes,
    coerce_token_usage,
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
    from llama_index.core.base.llms.types import (
        CompletionResponse,
        CompletionResponseAsyncGen,
        CompletionResponseGen,
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
    CompletionResponse = object  # type: ignore[assignment]
    CompletionResponseAsyncGen = AsyncIterator[Any]  # type: ignore[assignment]
    CompletionResponseGen = Iterator[Any]  # type: ignore[assignment]
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
# Error-code bundle (aligned with other LLM adapters)
# ---------------------------------------------------------------------------

ERROR_CODES = CoercionErrorCodes(
    invalid_result="LLAMAINDEX_LLM_INVALID_RESULT",
    empty_result="LLAMAINDEX_LLM_EMPTY_RESULT",
    conversion_error="LLAMAINDEX_LLM_CONVERSION_ERROR",
    framework_label="llamaindex",
)


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

    Uses shared `coerce_token_usage` from framework_utils for consistency
    across framework adapters, with a simple fallback for direct attribute access.
    """
    if usage is None:
        return {}

    # Primary path: shared coercion utility
    try:
        # Wrap usage in expected format for coerce_token_usage
        payload = {"usage": usage} if isinstance(usage, Mapping) else usage
        
        token_usage = coerce_token_usage(
            payload,
            framework="llamaindex",
            error_codes=ERROR_CODES,
            logger=logger,
        )
        return {
            k: v for k, v in {
                "prompt_tokens": token_usage.prompt_tokens,
                "completion_tokens": token_usage.completion_tokens,
                "total_tokens": token_usage.total_tokens,
            }.items() if v is not None
        }
    except Exception:  # noqa: BLE001
        # Fallback: direct attribute/mapping access
        if isinstance(usage, Mapping):
            return {
                k: int(v) for k, v in {
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                }.items() if isinstance(v, int)
            }
        
        # Object with attributes
        return {
            k: int(v) for k, v in {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }.items() if isinstance(v, int)
        }


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


def _to_translator_messages(messages: Sequence[ChatMessage]) -> List[Dict[str, Any]]:
    """
    Convert LlamaIndex ChatMessage objects to generic dicts for translator.

    The translator expects mappings (dicts) with role/content, not
    framework-specific message objects.
    """
    result = []
    for msg in messages:
        # Extract role (may be enum or string)
        role_raw = getattr(msg, "role", "user")
        if hasattr(role_raw, "value"):
            role = str(role_raw.value)
        else:
            role = str(role_raw)

        # Extract content from blocks or direct attribute
        content = ""
        if hasattr(msg, "blocks"):
            blocks = getattr(msg, "blocks", [])
            for block in blocks:
                if hasattr(block, "text"):
                    content += str(block.text)
                else:
                    content += str(block)
        else:
            content = str(getattr(msg, "content", ""))

        result.append({"role": role, "content": content})
    return result


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
# Event loop guards for sync APIs
# ---------------------------------------------------------------------------


def _ensure_not_in_event_loop(
    sync_api_name: str,
    async_api_name: Optional[str] = None,
) -> None:
    """
    Prevent deadlocks from calling sync APIs inside an active asyncio event loop.

    This is a lightweight guard used only on sync entrypoints. If a running
    event loop is detected, we raise a clear RuntimeError with guidance to
    use the async variant instead.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop: safe to call sync API.
        return

    hint = ""
    if async_api_name:
        hint = f" Use the async variant instead (e.g. '{async_api_name}')."
    raise RuntimeError(
        f"{sync_api_name} was called from inside an active asyncio event loop."
        f"{hint} [LLAMAINDEX_LLM_SYNC_WRAPPER_CALLED_IN_EVENT_LOOP]"
    )


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

        # LLMTranslator orchestrator for the LlamaIndex framework.
        # If no factory is registered, the translator layer will use
        # DefaultLLMFrameworkTranslator.
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
    # Resource management (context managers) – aligned with other adapters
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "CorpusLlamaIndexLLM":
        """Support sync context manager protocol for resource cleanup."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Clean up resources when exiting sync context."""
        close = getattr(self.llm_adapter, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error while closing LLM adapter in __exit__: %s", exc)

    async def __aenter__(self) -> "CorpusLlamaIndexLLM":
        """Support async context manager protocol."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Clean up resources when exiting async context."""
        aclose = getattr(self.llm_adapter, "aclose", None)
        if callable(aclose):
            try:
                await aclose()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error while async-closing LLM adapter in __aexit__: %s", exc)

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
            "operation": f"llm_{operation}",
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
            "error_codes": ERROR_CODES,
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

        The translation is hardened with error-context attachment so that
        failures in context construction are observable and diagnosable.
        """
        callback_manager = kwargs.get("callback_manager")

        try:
            ctx = context_from_llamaindex(
                callback_manager,
                framework_version=self.framework_version,
            )
        except Exception as exc:  # noqa: BLE001
            # Attach context-translation specific metadata so failures here are
            # distinguishable from protocol or provider failures.
            attach_context(
                exc,
                framework="llamaindex",
                resource_type="llm",
                operation="llm_context_translation",
                framework_version=self.framework_version,
                error_codes=ERROR_CODES,
                callback_manager_type=type(callback_manager).__name__,
            )
            raise

        # Duck-type check for OperationContext (handle cross-module instances)
        if not (isinstance(ctx, OperationContext) or 
                (hasattr(ctx, "request_id") and hasattr(ctx, "attrs"))):
            type_name = type(ctx).__name__
            exc = TypeError(
                "context_from_llamaindex produced unsupported context type "
                f"{type_name}"
            )
            attach_context(
                exc,
                framework="llamaindex",
                resource_type="llm",
                operation="llm_context_translation",
                framework_version=self.framework_version,
                error_codes=ERROR_CODES,
                returned_type=type_name,
            )
            raise exc

        return ctx

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
            float(clean.get("temperature", self.temperature)),
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
            # Convert LlamaIndex ChatMessages to dicts for translator
            normalized_messages = _to_translator_messages(messages)

            result = await self._translator.arun_complete(
                raw_messages=normalized_messages,
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

        async def _gen() -> AsyncIterator[ChatResponse]:
            # Keep error-context active for the full streaming iteration.
            async with self._error_context_async(
                "astream_chat",
                stream=True,
                messages=messages,
                model=model_for_context,
                ctx=ctx,
                params=params,
            ):
                # Convert LlamaIndex ChatMessages to dicts for translator
                normalized_messages = _to_translator_messages(messages)

                # LLMTranslator.arun_stream returns an AsyncIterator directly; do not await.
                agen = self._translator.arun_stream(
                    raw_messages=normalized_messages,
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
        _ensure_not_in_event_loop("chat", "achat")

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
            # Convert LlamaIndex ChatMessages to dicts for translator
            normalized_messages = _to_translator_messages(messages)

            result = self._translator.complete(
                raw_messages=normalized_messages,
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
        _ensure_not_in_event_loop("stream_chat", "astream_chat")

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

        def _gen() -> Iterator[ChatResponse]:
            # Keep error-context active for the full streaming iteration.
            with self._error_context(
                "stream_chat",
                stream=True,
                messages=messages,
                model=model_for_context,
                ctx=ctx,
                params=params,
            ):
                # Convert LlamaIndex ChatMessages to dicts for translator
                normalized_messages = _to_translator_messages(messages)

                stream_iter = self._translator.stream(
                    raw_messages=normalized_messages,
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

                for chunk_obj in stream_iter:
                    yield _build_chat_response_from_chunk_like(chunk_obj)

                # NOTE: Do not forcibly close the iterator.
                # The SyncStreamBridge-based iterators may defer raising worker
                # exceptions until the iterator naturally unwinds; calling
                # close() can suppress those errors.

        return _gen()

    # ------------------------------------------------------------------ #
    # Completion methods (required by LlamaIndex LLM base class)
    # ------------------------------------------------------------------ #

    def complete(
        self,
        prompt: str,
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse:
        """
        Sync completion entrypoint required by LlamaIndex.

        Converts the prompt to a user message and delegates to chat().
        """
        _ensure_not_in_event_loop("complete", "acomplete")

        # Convert prompt to a ChatMessage
        user_message = ChatMessage(role=MessageRole.USER, content=prompt)
        
        # Delegate to chat()
        chat_response = self.chat(messages=[user_message], **kwargs)
        
        # Convert ChatResponse to CompletionResponse
        return CompletionResponse(
            text=chat_response.message.content or "",
            additional_kwargs=chat_response.additional_kwargs,
            raw=chat_response.raw,
        )

    async def acomplete(
        self,
        prompt: str,
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse:
        """
        Async completion entrypoint required by LlamaIndex.

        Converts the prompt to a user message and delegates to achat().
        """
        # Convert prompt to a ChatMessage
        user_message = ChatMessage(role=MessageRole.USER, content=prompt)
        
        # Delegate to achat()
        chat_response = await self.achat(messages=[user_message], **kwargs)
        
        # Convert ChatResponse to CompletionResponse
        return CompletionResponse(
            text=chat_response.message.content or "",
            additional_kwargs=chat_response.additional_kwargs,
            raw=chat_response.raw,
        )

    def stream_complete(
        self,
        prompt: str,
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponseGen:
        """
        Sync streaming completion entrypoint required by LlamaIndex.

        Converts the prompt to a user message and delegates to stream_chat().
        """
        _ensure_not_in_event_loop("stream_complete", "astream_complete")

        # Convert prompt to a ChatMessage
        user_message = ChatMessage(role=MessageRole.USER, content=prompt)
        
        # Delegate to stream_chat()
        chat_stream = self.stream_chat(messages=[user_message], **kwargs)
        
        def _gen() -> Iterator[CompletionResponse]:
            for chat_response in chat_stream:
                yield CompletionResponse(
                    text=chat_response.delta or "",
                    additional_kwargs=chat_response.additional_kwargs,
                    raw=chat_response.raw,
                    delta=chat_response.delta,
                )
        
        return _gen()

    async def astream_complete(
        self,
        prompt: str,
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponseAsyncGen:
        """
        Async streaming completion entrypoint required by LlamaIndex.

        Converts the prompt to a user message and delegates to astream_chat().
        """
        # Convert prompt to a ChatMessage
        user_message = ChatMessage(role=MessageRole.USER, content=prompt)
        
        # Delegate to astream_chat()
        chat_stream = await self.astream_chat(messages=[user_message], **kwargs)
        
        async def _gen() -> AsyncIterator[CompletionResponse]:
            async for chat_response in chat_stream:
                yield CompletionResponse(
                    text=chat_response.delta or "",
                    additional_kwargs=chat_response.additional_kwargs,
                    raw=chat_response.raw,
                    delta=chat_response.delta,
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
        """Token counting helper (no silent fallback)."""

        if not messages:
            return 0

        ctx, params, model_for_context, framework_ctx = self._build_request_context(
            kwargs,
            operation="count_tokens",
            stream=False,
        )

        # Convert LlamaIndex ChatMessages to dicts for translator
        normalized_messages = _to_translator_messages(messages)

        tokens_any = self._translator.count_tokens_for_messages(
            raw_messages=normalized_messages,
            model=model_for_context,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )
        if isinstance(tokens_any, int):
            return tokens_any
        if isinstance(tokens_any, Mapping):
            for key in ("tokens", "total_tokens", "count"):
                value = tokens_any.get(key)
                if isinstance(value, int):
                    return value
        raise TypeError(
            f"{ERROR_CODES.BAD_USAGE_RESULT}: count_tokens returned unsupported type "
            f"{type(tokens_any).__name__}"
        )

    async def acount_tokens(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> int:
        """Async token counting wrapper for conformance parity."""
        return self.count_tokens(messages, **kwargs)

    # ------------------------------------------------------------------ #
    # Health / capabilities via translator only (no adapter fallback)
    # ------------------------------------------------------------------ #

    def health(self, **kwargs: Any) -> Mapping[str, Any]:
        """
        Synchronous health check.

        Resolution:
        1. self._translator.health(**kwargs)

        The translator is the single source of truth. If it does not implement
        health(), this is treated as a configuration error.
        """
        translator_health = getattr(self._translator, "health", None)
        if not callable(translator_health):
            raise AttributeError(
                "LLMTranslator for framework='llamaindex' does not implement health(). "
                "Health semantics must be provided by the translator."
            )

        try:
            return translator_health()
        except Exception as exc:  # noqa: BLE001
            if self._config.enable_error_context:
                attach_context(
                    exc,
                    framework="llamaindex",
                    resource_type="llm",
                    operation="health",
                    model=self.model,
                    error_codes=ERROR_CODES,
                    source="translator",
                )
            raise

    async def ahealth(self, **kwargs: Any) -> Mapping[str, Any]:
        """
        Async health check.

        Resolution:
        1. self._translator.ahealth(**kwargs) if available
        2. self._translator.health(**kwargs) via worker thread

        No fallback to the underlying llm_adapter – the translator is the
        only supported surface.
        """
        loop = asyncio.get_running_loop()

        translator_ahealth = getattr(self._translator, "ahealth", None)
        if callable(translator_ahealth):
            try:
                return await translator_ahealth()  # type: ignore[misc]
            except Exception as exc:  # noqa: BLE001
                if self._config.enable_error_context:
                    attach_context(
                        exc,
                        framework="llamaindex",
                        resource_type="llm",
                        operation="health",
                        model=self.model,
                        error_codes=ERROR_CODES,
                        source="translator_async",
                    )
                raise

        translator_health = getattr(self._translator, "health", None)
        if callable(translator_health):
            try:
                return await loop.run_in_executor(
                    None,
                    lambda: translator_health(),
                )
            except Exception as exc:  # noqa: BLE001
                if self._config.enable_error_context:
                    attach_context(
                        exc,
                        framework="llamaindex",
                        resource_type="llm",
                        operation="health",
                        model=self.model,
                        error_codes=ERROR_CODES,
                        source="translator_sync_thread",
                    )
                raise

        raise AttributeError(
            "LLMTranslator for framework='llamaindex' does not implement "
            "ahealth() or health(). Health semantics must be provided by the translator."
        )

    def capabilities(self, **kwargs: Any) -> Mapping[str, Any]:
        """
        Synchronous capabilities query.

        Resolution:
        1. self._translator.capabilities(**kwargs)

        The translator is the single source of truth. If it does not implement
        capabilities(), this is treated as a configuration error.
        """
        translator_capabilities = getattr(self._translator, "capabilities", None)
        if not callable(translator_capabilities):
            raise AttributeError(
                "LLMTranslator for framework='llamaindex' does not implement "
                "capabilities(). Capabilities semantics must be provided by the translator."
            )

        try:
            return translator_capabilities()
        except Exception as exc:  # noqa: BLE001
            if self._config.enable_error_context:
                attach_context(
                    exc,
                    framework="llamaindex",
                    resource_type="llm",
                    operation="capabilities",
                    model=self.model,
                    error_codes=ERROR_CODES,
                    source="translator",
                )
            raise

    async def acapabilities(self, **kwargs: Any) -> Mapping[str, Any]:
        """
        Async capabilities query.

        Resolution:
        1. self._translator.acapabilities(**kwargs) if available
        2. self._translator.capabilities(**kwargs) via worker thread

        No fallback to the underlying llm_adapter – the translator is the
        only supported surface.
        """
        loop = asyncio.get_running_loop()

        translator_acapabilities = getattr(self._translator, "acapabilities", None)
        if callable(translator_acapabilities):
            try:
                return await translator_acapabilities()  # type: ignore[misc]
            except Exception as exc:  # noqa: BLE001
                if self._config.enable_error_context:
                    attach_context(
                        exc,
                        framework="llamaindex",
                        resource_type="llm",
                        operation="capabilities",
                        model=self.model,
                        error_codes=ERROR_CODES,
                        source="translator_async",
                    )
                raise

        translator_capabilities = getattr(self._translator, "capabilities", None)
        if callable(translator_capabilities):
            try:
                return await loop.run_in_executor(
                    None,
                    lambda: translator_capabilities(),
                )
            except Exception as exc:  # noqa: BLE001
                if self._config.enable_error_context:
                    attach_context(
                        exc,
                        framework="llamaindex",
                        resource_type="llm",
                        operation="capabilities",
                        model=self.model,
                        error_codes=ERROR_CODES,
                        source="translator_sync_thread",
                )
                raise

        raise AttributeError(
            "LLMTranslator for framework='llamaindex' does not implement "
            "acapabilities() or capabilities(). Capabilities semantics must be "
            "provided by the translator."
        )


__all__ = [
    "LlamaIndexLLMProtocol",
    "CorpusLlamaIndexLLM",
    "LlamaIndexLLMConfig",
]
