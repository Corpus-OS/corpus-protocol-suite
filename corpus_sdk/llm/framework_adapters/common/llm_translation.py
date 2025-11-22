# corpus_sdk/llm/framework_adapters/common/llm_translation.py
# SPDX-License-Identifier: Apache-2.0
"""
Framework-agnostic LLM → Framework translation layer.

Purpose
-------
Provide a high-level orchestration and translation layer between:

- The Corpus LLM Protocol V1 (`LLMProtocolV1` / `BaseLLMAdapter`), and
- Framework-specific chat integrations (LangChain, LlamaIndex, SK, AutoGen, CrewAI, custom).

This module is intentionally *framework-neutral* and focuses on:

- Translating framework-level message objects into Corpus wire messages
  ({role, content}) via NormalizedMessage
- Translating `LLMCompletion` / `LLMChunk` / capabilities / health responses
  back into framework-facing shapes
- Providing sync + async APIs, including streaming via a sync bridge
- Attaching rich error context for observability while delegating all hard
  policies (deadlines, breakers, caching, rate limiting) to the adapter

Context translation
-------------------
This module does **not** parse framework configs directly. Instead:

- `corpus_sdk.core.context_translation` is responsible for taking
  framework-native contexts (LangChain RunnableConfig, LlamaIndex CallbackManager,
  etc.) and producing a core `OperationContext` type.
- Callers pass either an LLM `OperationContext` or a simple dict-like context
  into the methods here; we normalize that via `ctx_from_dict` into the core
  context, and then adapt into the LLM protocol's `OperationContext`.

Streaming
---------
For streaming completions, this module exposes:

- An async API that yields translated framework chunks, and
- A sync API that wraps the async iterator via `SyncStreamBridge`, preserving
  proper cancellation and error propagation.

Registry
--------
A small registry lets you register per-framework LLM translators:

- `register_llm_translator("my_framework", factory)`
- `create_llm_translator("my_framework", adapter, ...)`

This makes it straightforward to plug in framework-specific behaviors while
reusing the common orchestration logic here.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from corpus_sdk.llm.llm_base import (
    LLMProtocolV1,
    OperationContext,
    LLMCompletion,
    LLMChunk,
    LLMCapabilities,
    BadRequest,
)
from corpus_sdk.llm.framework_adapters.common.message_translation import (
    NormalizedMessage,
    to_corpus,
    from_generic_dict,
)
from corpus_sdk.core.context_translation import from_dict as ctx_from_dict
from corpus_sdk.core.sync_bridge import SyncStreamBridge
from corpus_sdk.core.async_bridge import AsyncBridge
from corpus_sdk.core.error_context import attach_context

LOG = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Helpers: OperationContext normalization
# =============================================================================


def _ensure_llm_operation_context(
    ctx: Optional[Union[OperationContext, Mapping[str, Any]]],
) -> OperationContext:
    """
    Normalize various context shapes into an LLM OperationContext.

    Accepts:
        - None:
            Uses context_translation.from_dict({}) to construct an "empty"
            core OperationContext, then adapts it into the LLM OperationContext.
        - OperationContext:
            Returned as-is.
        - Mapping[str, Any]:
            Interpreted via context_translation.from_dict, then adapted into
            an LLM OperationContext.

    This mirrors the graph translation layer and keeps responsibilities clean:
        - Framework-native → normalized core context happens in
          corpus_sdk.core.context_translation.
        - This helper simply ensures the LLM adapter receives the right type
          and shape for its OperationContext.
    """
    if ctx is None:
        core_ctx = ctx_from_dict({})
    elif isinstance(ctx, OperationContext):
        return ctx
    elif isinstance(ctx, Mapping):
        core_ctx = ctx_from_dict(ctx)
    else:
        raise BadRequest(
            f"Unsupported context type: {type(ctx).__name__}",
            code="BAD_OPERATION_CONTEXT",
        )

    # Reconstruct as LLM OperationContext with validation and without
    # leaking any framework-native details.
    return OperationContext(
        request_id=getattr(core_ctx, "request_id", None),
        idempotency_key=getattr(core_ctx, "idempotency_key", None),
        deadline_ms=getattr(core_ctx, "deadline_ms", None),
        traceparent=getattr(core_ctx, "traceparent", None),
        tenant=getattr(core_ctx, "tenant", None),
        attrs=getattr(core_ctx, "attrs", None) or {},
    )


# =============================================================================
# Framework-agnostic translator protocol
# =============================================================================


class LLMFrameworkTranslator(Protocol):
    """
    Per-framework translator contract for LLM operations.

    Implementations are responsible for:
        - Converting framework-level message inputs into NormalizedMessage[]
        - Deciding how system messages are handled
        - Converting framework-native tool definitions / tool_choice into
          Corpus wire-compatible shapes
        - Translating LLMCompletion / LLMChunk / capabilities / health /
          token counts back into framework-level outputs
        - Optionally decorating messages before send (guardrails, tags, etc.)
        - Optionally suggesting a preferred model when none is specified
    """

    # ---- input translation ----

    def to_normalized_messages(
        self,
        raw_messages: Any,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> List[NormalizedMessage]:
        """
        Translate framework-native messages into a list of NormalizedMessage.

        raw_messages may be:
            - List[NormalizedMessage]
            - List[Mapping[str, Any]]
            - Single Mapping[str, Any] (treated as length-1 list)
            - Framework-specific message objects (LangChain, LlamaIndex, etc.)
        """
        ...

    def build_system_message(
        self,
        normalized_messages: List[NormalizedMessage],
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Tuple[Optional[str], List[NormalizedMessage]]:
        """
        Decide how to handle system messages.

        Typical behavior:
            - Extract the first message with role "system" and return:
                (system_message_text, remaining_messages_without_that_system)
            - Or leave system messages in-place and return (None, messages)

        Returns:
            (system_message, remaining_messages)
        """
        ...

    def build_tools(
        self,
        raw_tools: Optional[Any],
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Translate framework-native tool definitions into Corpus tool schema.

        Returned structure should match the LLMProtocolV1 expectations:
            [ {"type": "function", "function": { ... }}, ... ] or None
        """
        ...

    def build_tool_choice(
        self,
        raw_tool_choice: Optional[Any],
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Optional[Union[str, Dict[str, Any]]]:
        """
        Translate framework-native tool_choice into the Corpus wire format.

        Typical values:
            - "auto", "none", "required"
            - A specific tool descriptor as a dict
            - None (adapter chooses)
        """
        ...

    # ---- output translation ----

    def from_completion(
        self,
        completion: LLMCompletion,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """
        Convert an LLMCompletion into a framework-level result object.
        """
        ...

    def from_chunk(
        self,
        chunk: LLMChunk,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """
        Convert a streaming LLMChunk into a framework-level chunk representation.
        """
        ...

    def from_count_tokens(
        self,
        token_count: int,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """
        Convert raw token count into a framework-level count response.
        """
        ...

    def from_health(
        self,
        health: Mapping[str, Any],
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """
        Convert adapter health mapping into a framework-facing health result.
        """
        ...

    def from_capabilities(
        self,
        caps: LLMCapabilities,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """
        Convert LLMCapabilities into a framework-facing capabilities structure.
        """
        ...

    # ---- optional hooks ----

    def preferred_model(
        self,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Optional[str]:
        """
        Optional hook for translators to derive a default model identifier.

        This can come from:
            - framework_ctx (e.g., configured model for a given index/router)
            - op_ctx.attrs (e.g., "llm_model" key)
        """
        ...

    def decorate_messages_before_send(
        self,
        messages: List[NormalizedMessage],
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> List[NormalizedMessage]:
        """
        Optional hook that can inject guardrails, additional context, or
        other framework-specific message transformations before calling
        the adapter.

        Default behavior is to return messages unchanged.
        """
        ...


# =============================================================================
# Default generic translator implementation
# =============================================================================


class DefaultLLMFrameworkTranslator:
    """
    Generic, framework-neutral translator implementation.

    Behaviors:
        - Treats raw_messages as:
            * list of NormalizedMessage → pass-through
            * mapping or list of mappings → via from_generic_dict
        - Extracts the first "system" message (if any) as system_message and
          drops it from the remaining messages.
        - Assumes tools/tool_choice are already in Corpus-compatible shape when
          provided as JSON-like dicts/values; shallow validation only.
        - For results:
            * LLMCompletion → neutral dict with text/model/usage/tool_calls
            * LLMChunk → neutral dict for streaming
            * LLMCapabilities → asdict() representation
            * health → pass-through mapping
            * count_tokens → raw integer
    """

    def to_normalized_messages(
        self,
        raw_messages: Any,
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> List[NormalizedMessage]:
        # Accept a single mapping as a single message
        if isinstance(raw_messages, Mapping):
            raw_seq: Iterable[Any] = [raw_messages]
        else:
            raw_seq = raw_messages

        if not isinstance(raw_seq, Iterable) or isinstance(raw_seq, (str, bytes)):
            raise BadRequest(
                "raw_messages must be a mapping or iterable of messages",
                code="BAD_MESSAGES",
            )

        messages: List[NormalizedMessage] = []

        for idx, m in enumerate(raw_seq):
            if isinstance(m, NormalizedMessage):
                messages.append(m)
                continue

            if isinstance(m, Mapping):
                messages.append(from_generic_dict(m))
                continue

            raise BadRequest(
                f"raw_messages[{idx}] must be a NormalizedMessage or mapping",
                code="BAD_MESSAGES",
                details={"index": idx, "type": type(m).__name__},
            )

        if not messages:
            raise BadRequest(
                "raw_messages must contain at least one message",
                code="BAD_MESSAGES",
            )

        return messages

    def build_system_message(
        self,
        normalized_messages: List[NormalizedMessage],
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Tuple[Optional[str], List[NormalizedMessage]]:
        """
        Default policy:
            - Extract the first message with role == "system" as the dedicated
              system_message string, drop it from the remaining messages.
            - If none exists, return (None, original_messages).
        """
        system_message: Optional[str] = None
        remaining: List[NormalizedMessage] = []

        for msg in normalized_messages:
            if system_message is None and msg.role == "system":
                system_message = msg.content
                continue
            remaining.append(msg)

        return system_message, remaining

    def build_tools(
        self,
        raw_tools: Optional[Any],
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Optional[List[Dict[str, Any]]]:
        if raw_tools is None:
            return None

        if isinstance(raw_tools, Mapping):
            raw_tools = [raw_tools]

        if not isinstance(raw_tools, Sequence):
            raise BadRequest(
                "tools must be a mapping or a sequence of mappings",
                code="BAD_TOOLS",
            )

        tools: List[Dict[str, Any]] = []
        for idx, t in enumerate(raw_tools):
            if not isinstance(t, Mapping):
                raise BadRequest(
                    f"tools[{idx}] must be a mapping",
                    code="BAD_TOOLS",
                    details={"index": idx, "type": type(t).__name__},
                )
            tools.append(dict(t))

        return tools

    def build_tool_choice(
        self,
        raw_tool_choice: Optional[Any],
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Optional[Union[str, Dict[str, Any]]]:
        if raw_tool_choice is None:
            return None

        if isinstance(raw_tool_choice, str):
            return raw_tool_choice

        if isinstance(raw_tool_choice, Mapping):
            return dict(raw_tool_choice)

        raise BadRequest(
            "tool_choice must be a string or mapping",
            code="BAD_TOOL_CHOICE",
            details={"type": type(raw_tool_choice).__name__},
        )

    def from_completion(
        self,
        completion: LLMCompletion,
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Any:
        """
        Default: return a neutral dict compatible with JSON.
        """
        return {
            "text": completion.text,
            "model": completion.model,
            "model_family": completion.model_family,
            "usage": {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
            },
            "finish_reason": completion.finish_reason,
            "tool_calls": [asdict(tc) for tc in completion.tool_calls],
        }

    def from_chunk(
        self,
        chunk: LLMChunk,
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Any:
        """
        Default: return a neutral dict per streaming chunk.
        """
        usage = chunk.usage_so_far
        return {
            "text": chunk.text,
            "is_final": chunk.is_final,
            "model": chunk.model,
            "usage_so_far": (
                {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                }
                if usage is not None
                else None
            ),
            "tool_calls": [asdict(tc) for tc in chunk.tool_calls],
        }

    def from_count_tokens(
        self,
        token_count: int,
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Any:
        """
        Default: return the integer count directly.
        """
        return int(token_count)

    def from_health(
        self,
        health: Mapping[str, Any],
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Any:
        """
        Default: shallow copy of health mapping.
        """
        return dict(health)

    def from_capabilities(
        self,
        caps: LLMCapabilities,
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Any:
        """
        Default: capabilities as a plain dict via asdict().
        """
        return asdict(caps)

    def preferred_model(
        self,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Optional[str]:
        """
        Default: derive model from context attrs if present, else None.
        """
        attrs = op_ctx.attrs or {}
        candidate = attrs.get("llm_model") or attrs.get("model")
        if candidate is None:
            return None
        value = str(candidate).strip()
        return value or None

    def decorate_messages_before_send(
        self,
        messages: List[NormalizedMessage],
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> List[NormalizedMessage]:
        """
        Default: no-op, return messages unchanged.
        """
        return list(messages)


# =============================================================================
# LLM Translator Orchestrator
# =============================================================================


class LLMTranslator:
    """
    Framework-agnostic orchestrator for LLM operations.

    This class:
        - Accepts framework-level inputs and a normalized OperationContext
        - Delegates to an LLMFrameworkTranslator to translate messages,
          tools, and tool_choice
        - Calls into an LLMProtocolV1 adapter to execute operations
        - Provides sync + async variants for core operations:
            * complete
            * stream
            * count_tokens
            * count_tokens_for_messages (helper)
            * health
            * capabilities
        - Handles streaming via SyncStreamBridge for sync callers
        - Attaches rich error context for diagnostics

    It does *not*:
        - Implement any backend-specific logic (that lives in BaseLLMAdapter subclasses)
        - Apply additional policies; all hardening & limits are delegated to the adapter
    """

    def __init__(
        self,
        *,
        adapter: LLMProtocolV1,
        framework: str = "generic",
        translator: Optional[LLMFrameworkTranslator] = None,
    ) -> None:
        self._adapter = adapter
        self._framework = framework
        self._translator: LLMFrameworkTranslator = translator or DefaultLLMFrameworkTranslator()

    # --------------------------------------------------------------------- #
    # Sync Complete (uses AsyncBridge)
    # --------------------------------------------------------------------- #

    def complete(
        self,
        raw_messages: Any,
        *,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[Any] = None,
        tool_choice: Optional[Any] = None,
        system_message: Optional[str] = None,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """
        Synchronous complete API.

        Uses AsyncBridge to call the async adapter from a sync context.
        """
        ctx = _ensure_llm_operation_context(op_ctx)
        timeout = ctx.deadline_ms / 1000.0 if ctx.deadline_ms else None

        async def _complete_coro() -> Any:
            try:
                normalized = self._translator.to_normalized_messages(
                    raw_messages,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )
                normalized = self._translator.decorate_messages_before_send(
                    normalized,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )

                auto_system, remaining = self._translator.build_system_message(
                    normalized,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )

                effective_system = system_message if system_message is not None else auto_system
                tools_corpus = self._translator.build_tools(
                    tools,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )
                tool_choice_corpus = self._translator.build_tool_choice(
                    tool_choice,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )

                wire_messages = to_corpus(remaining)
                effective_model = model or self._translator.preferred_model(
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )

                result = await self._adapter.complete(
                    messages=wire_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop_sequences=stop_sequences,
                    model=effective_model,
                    system_message=effective_system,
                    tools=tools_corpus,
                    tool_choice=tool_choice_corpus,
                    ctx=ctx,
                )

                if not isinstance(result, LLMCompletion):
                    raise BadRequest(
                        f"adapter.complete returned unsupported type: {type(result).__name__}",
                        code="BAD_ADAPTER_RESULT",
                    )

                return self._translator.from_completion(
                    result,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )
            except Exception as exc:
                attach_context(
                    exc,
                    framework=self._framework,
                    llm_operation="complete",
                    request_id=ctx.request_id,
                    tenant=ctx.tenant,
                )
                raise

        return AsyncBridge.run_async(_complete_coro(), timeout=timeout)

    # --------------------------------------------------------------------- #
    # Async Complete
    # --------------------------------------------------------------------- #

    async def arun_complete(
        self,
        raw_messages: Any,
        *,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[Any] = None,
        tool_choice: Optional[Any] = None,
        system_message: Optional[str] = None,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """
        Async complete API.

        Preferred for async applications and services.
        """
        ctx = _ensure_llm_operation_context(op_ctx)

        try:
            normalized = self._translator.to_normalized_messages(
                raw_messages,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            normalized = self._translator.decorate_messages_before_send(
                normalized,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

            auto_system, remaining = self._translator.build_system_message(
                normalized,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

            effective_system = system_message if system_message is not None else auto_system
            tools_corpus = self._translator.build_tools(
                tools,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            tool_choice_corpus = self._translator.build_tool_choice(
                tool_choice,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

            wire_messages = to_corpus(remaining)
            effective_model = model or self._translator.preferred_model(
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

            result = await self._adapter.complete(
                messages=wire_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop_sequences=stop_sequences,
                model=effective_model,
                system_message=effective_system,
                tools=tools_corpus,
                tool_choice=tool_choice_corpus,
                ctx=ctx,
            )

            if not isinstance(result, LLMCompletion):
                raise BadRequest(
                    f"adapter.complete returned unsupported type: {type(result).__name__}",
                    code="BAD_ADAPTER_RESULT",
                )

            return self._translator.from_completion(
                result,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
        except Exception as exc:
            attach_context(
                exc,
                framework=self._framework,
                llm_operation="complete",
                request_id=ctx.request_id,
                tenant=ctx.tenant,
            )
            raise

    # --------------------------------------------------------------------- #
    # Sync Stream (uses SyncStreamBridge)
    # --------------------------------------------------------------------- #

    def stream(
        self,
        raw_messages: Any,
        *,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[Any] = None,
        tool_choice: Optional[Any] = None,
        system_message: Optional[str] = None,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Iterator[Any]:
        """
        Synchronous streaming API.

        Returns a sync iterator that yields framework-level streaming chunks
        by bridging the async adapter.stream(...) via SyncStreamBridge.
        """
        ctx = _ensure_llm_operation_context(op_ctx)

        async def _stream_factory() -> AsyncIterator[Any]:
            try:
                normalized = self._translator.to_normalized_messages(
                    raw_messages,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )
                normalized = self._translator.decorate_messages_before_send(
                    normalized,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )

                auto_system, remaining = self._translator.build_system_message(
                    normalized,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )

                effective_system = system_message if system_message is not None else auto_system
                tools_corpus = self._translator.build_tools(
                    tools,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )
                tool_choice_corpus = self._translator.build_tool_choice(
                    tool_choice,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )

                wire_messages = to_corpus(remaining)
                effective_model = model or self._translator.preferred_model(
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )

                agen = self._adapter.stream(
                    messages=wire_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop_sequences=stop_sequences,
                    model=effective_model,
                    system_message=effective_system,
                    tools=tools_corpus,
                    tool_choice=tool_choice_corpus,
                    ctx=ctx,
                )

                async for chunk in agen:
                    if not isinstance(chunk, LLMChunk):
                        raise BadRequest(
                            f"adapter.stream yielded unsupported type: {type(chunk).__name__}",
                            code="BAD_ADAPTER_RESULT",
                        )
                    yield self._translator.from_chunk(
                        chunk,
                        op_ctx=ctx,
                        framework_ctx=framework_ctx,
                    )
            except Exception as exc:
                attach_context(
                    exc,
                    framework=self._framework,
                    llm_operation="stream",
                    request_id=ctx.request_id,
                    tenant=ctx.tenant,
                )
                raise

        bridge = SyncStreamBridge(
            coro_factory=_stream_factory,
            framework=self._framework,
            error_context={
                "operation": "llm.stream",
                "request_id": ctx.request_id,
                "tenant": ctx.tenant,
            },
        )
        return bridge.run()

    # --------------------------------------------------------------------- #
    # Async Stream
    # --------------------------------------------------------------------- #

    async def arun_stream(
        self,
        raw_messages: Any,
        *,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[Any] = None,
        tool_choice: Optional[Any] = None,
        system_message: Optional[str] = None,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> AsyncIterator[Any]:
        """
        Async streaming API.

        Returns an async iterator yielding framework-level streaming chunks.
        """
        ctx = _ensure_llm_operation_context(op_ctx)

        try:
            normalized = self._translator.to_normalized_messages(
                raw_messages,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            normalized = self._translator.decorate_messages_before_send(
                normalized,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

            auto_system, remaining = self._translator.build_system_message(
                normalized,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

            effective_system = system_message if system_message is not None else auto_system
            tools_corpus = self._translator.build_tools(
                tools,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            tool_choice_corpus = self._translator.build_tool_choice(
                tool_choice,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

            wire_messages = to_corpus(remaining)
            effective_model = model or self._translator.preferred_model(
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

            agen = self._adapter.stream(
                messages=wire_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop_sequences=stop_sequences,
                model=effective_model,
                system_message=effective_system,
                tools=tools_corpus,
                tool_choice=tool_choice_corpus,
                ctx=ctx,
            )

            async for chunk in agen:
                if not isinstance(chunk, LLMChunk):
                    raise BadRequest(
                        f"adapter.stream yielded unsupported type: {type(chunk).__name__}",
                        code="BAD_ADAPTER_RESULT",
                    )
                yield self._translator.from_chunk(
                    chunk,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )
        except Exception as exc:
            attach_context(
                exc,
                framework=self._framework,
                llm_operation="stream",
                request_id=ctx.request_id,
                tenant=ctx.tenant,
            )
            raise

    # --------------------------------------------------------------------- #
    # Sync count_tokens (text) (uses AsyncBridge)
    # --------------------------------------------------------------------- #

    def count_tokens(
        self,
        text: str,
        *,
        model: Optional[str] = None,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """
        Synchronous count_tokens wrapper around adapter.count_tokens().
        """
        ctx = _ensure_llm_operation_context(op_ctx)
        timeout = ctx.deadline_ms / 1000.0 if ctx.deadline_ms else None

        async def _count_coro() -> Any:
            try:
                result = await self._adapter.count_tokens(
                    text=text,
                    model=model,
                    ctx=ctx,
                )
                return self._translator.from_count_tokens(
                    int(result),
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )
            except Exception as exc:
                attach_context(
                    exc,
                    framework=self._framework,
                    llm_operation="count_tokens",
                    request_id=ctx.request_id,
                    tenant=ctx.tenant,
                )
                raise

        return AsyncBridge.run_async(_count_coro(), timeout=timeout)

    # --------------------------------------------------------------------- #
    # Async count_tokens (text)
    # --------------------------------------------------------------------- #

    async def arun_count_tokens(
        self,
        text: str,
        *,
        model: Optional[str] = None,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """
        Async count_tokens wrapper.
        """
        ctx = _ensure_llm_operation_context(op_ctx)

        try:
            result = await self._adapter.count_tokens(
                text=text,
                model=model,
                ctx=ctx,
            )
            return self._translator.from_count_tokens(
                int(result),
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
        except Exception as exc:
            attach_context(
                exc,
                framework=self._framework,
                llm_operation="count_tokens",
                request_id=ctx.request_id,
                tenant=ctx.tenant,
            )
            raise

    # --------------------------------------------------------------------- #
    # Helper: count_tokens for messages (sync + async)
    # --------------------------------------------------------------------- #

    def count_tokens_for_messages(
        self,
        raw_messages: Any,
        *,
        model: Optional[str] = None,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """
        Synchronous helper to count tokens for a set of chat messages.

        This flattens messages and optional system message into a single text
        string using a simple "role: content" format, then delegates to
        adapter.count_tokens().
        """
        ctx = _ensure_llm_operation_context(op_ctx)
        timeout = ctx.deadline_ms / 1000.0 if ctx.deadline_ms else None

        async def _count_msgs_coro() -> Any:
            try:
                normalized = self._translator.to_normalized_messages(
                    raw_messages,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )
                auto_system, remaining = self._translator.build_system_message(
                    normalized,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )

                parts: List[str] = []
                if auto_system:
                    parts.append(f"system:{auto_system}")
                for msg in remaining:
                    parts.append(f"{msg.role}:{msg.content}")
                combined = "\n".join(parts)

                result = await self._adapter.count_tokens(
                    text=combined,
                    model=model,
                    ctx=ctx,
                )
                return self._translator.from_count_tokens(
                    int(result),
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )
            except Exception as exc:
                attach_context(
                    exc,
                    framework=self._framework,
                    llm_operation="count_tokens_for_messages",
                    request_id=ctx.request_id,
                    tenant=ctx.tenant,
                )
                raise

        return AsyncBridge.run_async(_count_msgs_coro(), timeout=timeout)

    async def arun_count_tokens_for_messages(
        self,
        raw_messages: Any,
        *,
        model: Optional[str] = None,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """
        Async helper to count tokens for chat messages.
        """
        ctx = _ensure_llm_operation_context(op_ctx)

        try:
            normalized = self._translator.to_normalized_messages(
                raw_messages,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            auto_system, remaining = self._translator.build_system_message(
                normalized,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

            parts: List[str] = []
            if auto_system:
                parts.append(f"system:{auto_system}")
            for msg in remaining:
                parts.append(f"{msg.role}:{msg.content}")
            combined = "\n".join(parts)

            result = await self._adapter.count_tokens(
                text=combined,
                model=model,
                ctx=ctx,
            )
            return self._translator.from_count_tokens(
                int(result),
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
        except Exception as exc:
            attach_context(
                exc,
                framework=self._framework,
                llm_operation="count_tokens_for_messages",
                request_id=ctx.request_id,
                tenant=ctx.tenant,
            )
            raise

    # --------------------------------------------------------------------- #
    # Health (sync + async)
    # --------------------------------------------------------------------- #

    def health(
        self,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """
        Synchronous health check wrapper.
        """
        ctx = _ensure_llm_operation_context(op_ctx)
        timeout = ctx.deadline_ms / 1000.0 if ctx.deadline_ms else None

        async def _health_coro() -> Any:
            try:
                h = await self._adapter.health(ctx=ctx)
                return self._translator.from_health(
                    h,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )
            except Exception as exc:
                attach_context(
                    exc,
                    framework=self._framework,
                    llm_operation="health",
                    request_id=ctx.request_id,
                    tenant=ctx.tenant,
                )
                raise

        return AsyncBridge.run_async(_health_coro(), timeout=timeout)

    async def arun_health(
        self,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """
        Async health check wrapper.
        """
        ctx = _ensure_llm_operation_context(op_ctx)

        try:
            h = await self._adapter.health(ctx=ctx)
            return self._translator.from_health(
                h,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
        except Exception as exc:
            attach_context(
                exc,
                framework=self._framework,
                llm_operation="health",
                request_id=ctx.request_id,
                tenant=ctx.tenant,
            )
            raise

    # --------------------------------------------------------------------- #
    # Capabilities (sync + async)
    # --------------------------------------------------------------------- #

    def capabilities(
        self,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """
        Synchronous capabilities wrapper.

        Returns framework-level capabilities derived from LLMCapabilities.
        """
        ctx = _ensure_llm_operation_context(op_ctx)
        timeout = ctx.deadline_ms / 1000.0 if ctx.deadline_ms else None

        async def _caps_coro() -> Any:
            try:
                caps = await self._adapter.capabilities()
                return self._translator.from_capabilities(
                    caps,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )
            except Exception as exc:
                attach_context(
                    exc,
                    framework=self._framework,
                    llm_operation="capabilities",
                    request_id=ctx.request_id,
                    tenant=ctx.tenant,
                )
                raise

        return AsyncBridge.run_async(_caps_coro(), timeout=timeout)

    async def arun_capabilities(
        self,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """
        Async capabilities wrapper.
        """
        ctx = _ensure_llm_operation_context(op_ctx)

        try:
            caps = await self._adapter.capabilities()
            return self._translator.from_capabilities(
                caps,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
        except Exception as exc:
            attach_context(
                exc,
                framework=self._framework,
                llm_operation="capabilities",
                request_id=ctx.request_id,
                tenant=ctx.tenant,
            )
            raise


# =============================================================================
# Registry for per-framework translators
# =============================================================================


_TranslatorFactory = Callable[[LLMProtocolV1], LLMFrameworkTranslator]
_LLM_TRANSLATOR_FACTORIES: Dict[str, _TranslatorFactory] = {}


def register_llm_translator(
    framework: str,
    factory: _TranslatorFactory,
) -> None:
    """
    Register or override an LLMFrameworkTranslator factory for a given framework.

    Example
    -------
        def make_langchain_llm_translator(adapter: LLMProtocolV1) -> LLMFrameworkTranslator:
            return LangChainLLMTranslator(adapter=adapter)

        register_llm_translator("langchain", make_langchain_llm_translator)
    """
    if not framework or not isinstance(framework, str):
        raise BadRequest(
            "framework name must be a non-empty string",
            code="BAD_TRANSLATOR_REGISTRATION",
        )
    if not callable(factory):
        raise BadRequest(
            "translator factory must be callable",
            code="BAD_TRANSLATOR_REGISTRATION",
        )
    _LLM_TRANSLATOR_FACTORIES[framework] = factory
    LOG.debug("Registered LLM translator factory for framework=%s", framework)


def get_llm_translator_factory(framework: str) -> Optional[_TranslatorFactory]:
    """
    Return a previously registered LLMFrameworkTranslator factory for a framework, if any.
    """
    return _LLM_TRANSLATOR_FACTORIES.get(framework)


def create_llm_translator(
    *,
    adapter: LLMProtocolV1,
    framework: str = "generic",
    translator: Optional[LLMFrameworkTranslator] = None,
) -> LLMTranslator:
    """
    Convenience helper to construct an LLMTranslator for a given framework.

    Behavior:
        - If `translator` is provided explicitly, it is used as-is.
        - Else, if a factory is registered for `framework`, it is used.
        - Else, DefaultLLMFrameworkTranslator is used.
    """
    if translator is None:
        factory = get_llm_translator_factory(framework)
        if factory is not None:
            translator = factory(adapter)
        else:
            translator = DefaultLLMFrameworkTranslator()
    return LLMTranslator(adapter=adapter, framework=framework, translator=translator)


__all__ = [
    "LLMFrameworkTranslator",
    "DefaultLLMFrameworkTranslator",
    "LLMTranslator",
    "register_llm_translator",
    "get_llm_translator_factory",
    "create_llm_translator",
]