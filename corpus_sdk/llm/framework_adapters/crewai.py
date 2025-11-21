# corpus_sdk/llm/framework_adapters/crewai.py
# SPDX-License-Identifier: Apache-2.0

"""
CrewAI adapter for Corpus LLM protocol.

This module exposes a Corpus `LLMProtocolV1` as a CrewAI-compatible LLM
wrapper. It provides direct LLM protocol access with message translation,
focusing on:

- Async + sync completion APIs
- Async + sync streaming (true incremental streaming)
- Context propagation via `OperationContext`
- Error context attachment for observability
- Direct protocol access with CrewAI message translation

Design goals
------------

1. Protocol-first:
   Direct access to `LLMProtocolV1` with CrewAI message translation.

2. Optional dependency safe:
   Import of `crewai` is guarded. Importing this module is safe even if
   CrewAI is not installed.

3. Simple & explicit interface:
   Clean API that CrewAI agents can use directly:

       llm = CorpusCrewAILLM(llm_adapter=adapter, model="gpt-4")
       text = llm.complete("Hello!")
       async for token in llm.astream("Hello!"):
           ...

4. True streaming:
   Direct async/sync streaming via the underlying protocol implementation.

5. Context + observability:
   - `OperationContext` built from CrewAI context
   - Rich error context for debugging
"""

from __future__ import annotations

from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)

# Context translation for OperationContext
from corpus_sdk.core.context_translation import from_crewai as from_crewai_context

# Message translation for CrewAI-like messages
from corpus_sdk.llm.framework_adapters.common.message_translation import (
    NormalizedMessage,
    from_crewai as from_crewai_message,
    to_corpus,
)

from corpus_sdk.core.error_context import attach_context
from corpus_sdk.llm.llm_base import (
    LLMCompletion,
    LLMProtocolV1,
    OperationContext,
)

# ---------------------------------------------------------------------------
# Type definitions for CrewAI compatibility
# ---------------------------------------------------------------------------


@runtime_checkable
class CrewAIMessage(Protocol):
    """Protocol for CrewAI message-like objects."""
    role: str
    content: str


# Type aliases for better readability
CrewAIMessageInput = Union[str, Mapping[str, Any], CrewAIMessage]
CrewAIMessageSequence = Sequence[CrewAIMessageInput]

# ---------------------------------------------------------------------------
# Optional CrewAI import (for dependency checks only)
# ---------------------------------------------------------------------------

_CREWAI_IMPORT_ERROR: Optional[BaseException] = None

try:  # pragma: no cover - optional dependency
    import crewai  # noqa: F401
except BaseException as exc:  # pragma: no cover
    _CREWAI_IMPORT_ERROR = exc


def _ensure_crewai_installed() -> None:
    """Raise a helpful error if CrewAI is not installed."""
    if _CREWAI_IMPORT_ERROR is not None:
        raise RuntimeError(
            "CrewAI is required to use CorpusCrewAILLM. "
            "Install with: pip install 'crewai>=0.51.0'"
        ) from _CREWAI_IMPORT_ERROR


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _normalize_messages(
    input_messages: Union[CrewAIMessageInput, CrewAIMessageSequence],
) -> Sequence[Mapping[str, Any]]:
    """
    Normalize various CrewAI-friendly inputs into Corpus wire messages.
    """
    if isinstance(input_messages, (str, Mapping)) or hasattr(input_messages, "role"):
        messages_list: Sequence[CrewAIMessageInput] = [input_messages]
    elif isinstance(input_messages, Sequence):
        messages_list = input_messages
    else:
        messages_list = [str(input_messages)]

    normalized: Sequence[NormalizedMessage] = [
        from_crewai_message(m) for m in messages_list
    ]
    return to_corpus(normalized)


def _extract_stop_sequences(kwargs: Mapping[str, Any]) -> Optional[list[str]]:
    """Extract stop sequences from kwargs."""
    stop = kwargs.get("stop")
    if stop is None:
        stop = kwargs.get("stop_sequences")

    if stop is None:
        return None
    if isinstance(stop, str):
        return [stop]
    if isinstance(stop, (list, tuple)):
        return [str(s) for s in stop]
    return [str(stop)]


def _build_operation_context_from_kwargs(kwargs: Mapping[str, Any]) -> OperationContext:
    """
    Build OperationContext from generic kwargs using context translation.

    - Respects an explicitly provided OperationContext (`ctx`) if present.
    - Passes only known/intentional context fields into `from_crewai_context`
      instead of forwarding the entire kwargs dict.
    """
    # If caller already built an OperationContext, trust it.
    ctx = kwargs.get("ctx")
    if isinstance(ctx, OperationContext):
        return ctx

    framework_version = kwargs.get("framework_version")
    task = kwargs.get("task")

    # Only include fields that are meaningful for context translation.
    allowed_keys = {
        "request_id",
        "traceparent",
        "tenant",
        "attrs",
        "agent",
        "crew_id",
        "agent_role",
        "agent_goal",
        "task_description",
        "crew_name",
        "process_id",
    }

    context_kwargs: Dict[str, Any] = {
        k: v for k, v in kwargs.items() if k in allowed_keys
    }

    return from_crewai_context(
        task=task,
        framework_version=framework_version,
        **context_kwargs,
    )


def _build_sampling_params(
    *,
    default_model: str,
    default_temperature: float,
    default_max_tokens: Optional[int],
    kwargs: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build complete sampling parameters from kwargs and defaults."""
    stop_sequences = _extract_stop_sequences(kwargs)

    params: Dict[str, Any] = {
        "model": kwargs.get("model", default_model),
        "temperature": kwargs.get("temperature", default_temperature),
        "max_tokens": kwargs.get("max_tokens", default_max_tokens),
        "top_p": kwargs.get("top_p"),
        "frequency_penalty": kwargs.get("frequency_penalty"),
        "presence_penalty": kwargs.get("presence_penalty"),
        "stop_sequences": stop_sequences,
        "seed": kwargs.get("seed"),
        "top_k": kwargs.get("top_k"),
    }

    # Drop None values so adapters see only explicit parameters
    return {k: v for k, v in params.items() if v is not None}


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------


class CorpusCrewAILLM:
    """
    CrewAI-compatible LLM wrapper backed by a Corpus `LLMProtocolV1`.

    Provides direct access to the LLM protocol with a CrewAI-friendly interface.
    """

    def __init__(
        self,
        *,
        llm_adapter: LLMProtocolV1,
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        framework_version: Optional[str] = None,
        require_crewai: bool = False,
    ) -> None:
        if require_crewai:
            _ensure_crewai_installed()

        self._llm: LLMProtocolV1 = llm_adapter
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = max_tokens
        self._framework_version = framework_version

    # ------------------------------------------------------------------ #
    # Error-context helpers
    # ------------------------------------------------------------------ #

    def _attach_error_context(
        self,
        exc: BaseException,
        operation: str,
        *,
        stream: bool,
        messages_count: int,
        model: str,
        ctx: OperationContext,
        params: Mapping[str, Any],
    ) -> None:
        """Attach error context to exceptions."""
        extra: Dict[str, Any] = {}
        attrs = getattr(ctx, "attrs", {})
        if isinstance(attrs, Mapping):
            crewai_info = attrs.get("crewai")
            if isinstance(crewai_info, Mapping):
                for key in (
                    "agent_role",
                    "agent_goal",
                    "task_description",
                    "crew_id",
                    "crew_name",
                ):
                    if key in crewai_info:
                        extra[key] = crewai_info[key]

        attach_context(
            exc,
            framework="crewai",
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
            **extra,
        )

    # ------------------------------------------------------------------ #
    # Internal helper: apply instance-level defaults to kwargs
    # ------------------------------------------------------------------ #

    def _apply_instance_defaults(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Return a shallow copy of kwargs, ensuring instance-level defaults
        (like framework_version) are applied if not explicitly provided.
        """
        if "framework_version" in kwargs or self._framework_version is None:
            return dict(kwargs)

        updated = dict(kwargs)
        updated["framework_version"] = self._framework_version
        return updated

    # ------------------------------------------------------------------ #
    # Core async API
    # ------------------------------------------------------------------ #

    async def acomplete(
        self,
        messages: Union[CrewAIMessageInput, CrewAIMessageSequence],
        **kwargs: Any,
    ) -> str:
        """Async completion."""
        kwargs = self._apply_instance_defaults(kwargs)
        corpus_messages = _normalize_messages(messages)
        ctx = _build_operation_context_from_kwargs(kwargs)
        params = _build_sampling_params(
            default_model=self.model,
            default_temperature=self.temperature,
            default_max_tokens=self.max_tokens,
            kwargs=kwargs,
        )

        model_for_context = params.get("model", self.model)
        messages_count = len(corpus_messages)

        try:
            result: LLMCompletion = await self._llm.acomplete(
                messages=corpus_messages,
                ctx=ctx,
                **params,
            )
            return result.text
        except BaseException as exc:  # noqa: BLE001
            self._attach_error_context(
                exc,
                "acomplete",
                stream=False,
                messages_count=messages_count,
                model=model_for_context,
                ctx=ctx,
                params=params,
            )
            raise

    async def astream(
        self,
        messages: Union[CrewAIMessageInput, CrewAIMessageSequence],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Async streaming completion."""
        kwargs = self._apply_instance_defaults(kwargs)
        corpus_messages = _normalize_messages(messages)
        ctx = _build_operation_context_from_kwargs(kwargs)
        params = _build_sampling_params(
            default_model=self.model,
            default_temperature=self.temperature,
            default_max_tokens=self.max_tokens,
            kwargs=kwargs,
        )

        model_for_context = params.get("model", self.model)
        messages_count = len(corpus_messages)

        try:
            async for chunk in self._llm.astream(
                messages=corpus_messages,
                ctx=ctx,
                **params,
            ):
                # Be defensive: chunk.text may be None depending on adapter
                yield getattr(chunk, "text", "") or ""
        except BaseException as exc:  # noqa: BLE001
            self._attach_error_context(
                exc,
                "astream",
                stream=True,
                messages_count=messages_count,
                model=model_for_context,
                ctx=ctx,
                params=params,
            )
            raise

    # ------------------------------------------------------------------ #
    # Sync API (for typical CrewAI usage)
    # ------------------------------------------------------------------ #

    def complete(
        self,
        messages: Union[CrewAIMessageInput, CrewAIMessageSequence],
        **kwargs: Any,
    ) -> str:
        """Sync completion."""
        kwargs = self._apply_instance_defaults(kwargs)
        corpus_messages = _normalize_messages(messages)
        ctx = _build_operation_context_from_kwargs(kwargs)
        params = _build_sampling_params(
            default_model=self.model,
            default_temperature=self.temperature,
            default_max_tokens=self.max_tokens,
            kwargs=kwargs,
        )

        model_for_context = params.get("model", self.model)
        messages_count = len(corpus_messages)

        try:
            result: LLMCompletion = self._llm.complete(
                messages=corpus_messages,
                ctx=ctx,
                **params,
            )
            return result.text
        except BaseException as exc:  # noqa: BLE001
            self._attach_error_context(
                exc,
                "complete",
                stream=False,
                messages_count=messages_count,
                model=model_for_context,
                ctx=ctx,
                params=params,
            )
            raise

    # Make the instance directly callable for convenience
    __call__ = complete

    def stream(
        self,
        messages: Union[CrewAIMessageInput, CrewAIMessageSequence],
        **kwargs: Any,
    ) -> Iterator[str]:
        """Sync streaming completion."""
        kwargs = self._apply_instance_defaults(kwargs)
        corpus_messages = _normalize_messages(messages)
        ctx = _build_operation_context_from_kwargs(kwargs)
        params = _build_sampling_params(
            default_model=self.model,
            default_temperature=self.temperature,
            default_max_tokens=self.max_tokens,
            kwargs=kwargs,
        )

        model_for_context = params.get("model", self.model)
        messages_count = len(corpus_messages)

        try:
            for chunk in self._llm.stream(
                messages=corpus_messages,
                ctx=ctx,
                **params,
            ):
                yield getattr(chunk, "text", "") or ""
        except BaseException as exc:  # noqa: BLE001
            self._attach_error_context(
                exc,
                "stream",
                stream=True,
                messages_count=messages_count,
                model=model_for_context,
                ctx=ctx,
                params=params,
            )
            raise


__all__ = [
    "CorpusCrewAILLM",
    "CrewAIMessage",
    "CrewAIMessageInput",
    "CrewAIMessageSequence",
]
