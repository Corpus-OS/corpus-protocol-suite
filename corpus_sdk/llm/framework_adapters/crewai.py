# corpus_sdk/llm/framework_adapters/crewai.py
# SPDX-License-Identifier: Apache-2.0

"""
CrewAI adapter for Corpus LLM protocol.

This module exposes a Corpus `LLMProtocolV1` as a CrewAI-compatible LLM
wrapper. It is intentionally framework-agnostic on the CrewAI side and
focuses on:

- Async + sync completion APIs
- Async + sync streaming (true incremental streaming)
- Context propagation via `OperationContext` where possible
- Error context attachment for better observability
- Protocol-first, translator-centric architecture via `LLMTranslator`

Design goals
------------

1. Protocol-first:
   CrewAI is treated as an integration surface. All real behavior flows
   through `LLMProtocolV1` and the LLM translator in
   `llm_translation.py`.

2. Optional dependency safe:
   Import of `crewai` is guarded. Importing this module is safe even if
   CrewAI is not installed. Instantiating `CorpusCrewAILLM` can enforce
   the dependency via `require_crewai=True`.

3. Simple & explicit interface:
   This adapter exposes a small, explicit API that is easy to use from
   CrewAI agents and tasks:

       llm = CorpusCrewAILLM(llm_adapter=adapter, model="gpt-4")
       text = llm.complete("Hello!")
       async for token in llm.astream("Hello!"):
           ...

   You can typically pass this `llm` instance as the LLM for a CrewAI
   Agent, as long as the Agent expects a callable that takes a prompt
   (or messages) and returns text.

4. True streaming:
   - Async streaming uses `LLMTranslator.arun_stream` (which delegates to
     the underlying protocol implementation).
   - Sync streaming uses `LLMTranslator.stream`, which encapsulates any
     async→sync bridging inside the translator.

5. Context + observability:
   - `OperationContext` is built from generic kwargs (request_id, tenant,
     attrs, CrewAI hints) when provided.
   - Errors are enriched with framework-specific context via
     `attach_context(framework="crewai", ...)`.

This is SDK infrastructure, not business logic. It does not attempt to
mirror CrewAI's internal LLM configuration model; instead, it focuses on
being a robust, callable LLM wrapper that CrewAI can invoke.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager, contextmanager
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

from corpus_sdk.core.error_context import attach_context
from corpus_sdk.llm.framework_adapters.common.message_translation import (
    NormalizedMessage,
    from_crewai,
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
    import crewai  # type: ignore[unused-ignore]  # noqa: F401
except BaseException as exc:  # pragma: no cover
    _CREWAI_IMPORT_ERROR = exc


def _ensure_crewai_installed() -> None:
    """
    Raise a helpful error if CrewAI is not installed.

    Importing this module is always safe; instantiating the adapter checks
    the optional dependency when requested.
    """
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

    Uses the shared message translation infrastructure.
    """
    if isinstance(input_messages, (str, Mapping)) or hasattr(input_messages, "role"):
        messages_list: Sequence[CrewAIMessageInput] = [input_messages]
    elif isinstance(input_messages, Sequence):
        messages_list = input_messages
    else:
        # Extreme fallback: stringify
        messages_list = [str(input_messages)]

    normalized: Sequence[NormalizedMessage] = [from_crewai(m) for m in messages_list]
    return to_corpus(normalized)


def _extract_stop_sequences(kwargs: Mapping[str, Any]) -> Optional[list[str]]:
    """
    Heuristically extract stop sequences from kwargs.

    Supports:
        - stop="foo"
        - stop=["foo", "bar"]
        - stop_sequences=[...]
    """
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
    Build an OperationContext from generic kwargs.

    This intentionally does *not* depend on CrewAI internals; it simply
    looks for common tracing / tenancy keys that callers may pass through
    (e.g., from a higher-level router).

    Recognized keys (all optional):
        - ctx: OperationContext (if already provided)
        - request_id, traceparent, tenant
        - attrs (mapping)
        - CrewAI context: agent, task, crew_id, etc. (stored in attrs["crewai"])
    """
    ctx = kwargs.get("ctx")
    if isinstance(ctx, OperationContext):
        return ctx

    request_id = kwargs.get("request_id")
    traceparent = kwargs.get("traceparent")
    tenant = kwargs.get("tenant")
    attrs = kwargs.get("attrs") or {}

    # Extract CrewAI-specific context if available
    crewai_context: Dict[str, Any] = {}
    for key in [
        "agent",
        "task",
        "crew_id",
        "agent_role",
        "agent_goal",
        "task_description",
        "crew_name",
        "process_id",
    ]:
        if key in kwargs:
            crewai_context[key] = kwargs[key]

    if not isinstance(attrs, Mapping):
        attrs = {"raw_attrs": attrs}

    attrs = dict(attrs)
    if crewai_context:
        attrs.setdefault("crewai", {}).update(crewai_context)

    return OperationContext(
        request_id=str(request_id) if request_id is not None else None,
        traceparent=str(traceparent) if traceparent is not None else None,
        tenant=str(tenant) if tenant is not None else None,
        attrs=attrs,
    )


def _build_sampling_params(
    *,
    default_model: str,
    default_temperature: float,
    default_max_tokens: Optional[int],
    kwargs: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Build complete sampling parameters from kwargs and defaults.

    Single unified function that handles all parameter extraction and precedence.
    """
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

    return {k: v for k, v in params.items() if v is not None}


# ---------------------------------------------------------------------------
# Configuration Builder (thin wrapper)
# ---------------------------------------------------------------------------


class CorpusCrewAILLMConfig:
    """
    Configuration builder for CorpusCrewAILLM to simplify setup.

    Usage:
        config = (CorpusCrewAILLMConfig(llm_adapter)
                  .with_model("gpt-4")
                  .with_temperature(0.7)
                  .with_max_tokens(2048))
        llm = config.build()
    """

    def __init__(self, llm_adapter: LLMProtocolV1) -> None:
        self.llm_adapter = llm_adapter
        self.model = "default"
        self.temperature = 0.7
        self.max_tokens: Optional[int] = None
        self.framework_version: Optional[str] = None
        self.require_crewai: bool = False

    def with_model(self, model: str) -> CorpusCrewAILLMConfig:
        self.model = model
        return self

    def with_temperature(self, temperature: float) -> CorpusCrewAILLMConfig:
        self.temperature = float(temperature)
        return self

    def with_max_tokens(self, max_tokens: Optional[int]) -> CorpusCrewAILLMConfig:
        self.max_tokens = max_tokens
        return self

    def with_framework_version(self, version: str) -> CorpusCrewAILLMConfig:
        self.framework_version = version
        return self

    def with_crewai_required(self, required: bool = True) -> CorpusCrewAILLMConfig:
        self.require_crewai = required
        return self

    def build(self) -> "CorpusCrewAILLM":
        return CorpusCrewAILLM(
            llm_adapter=self.llm_adapter,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            framework_version=self.framework_version,
            require_crewai=self.require_crewai,
        )


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------


class CorpusCrewAILLM:
    """
    CrewAI-compatible LLM wrapper backed by a Corpus `LLMProtocolV1`.

    CrewAI code can treat this as a callable LLM:

        llm = CorpusCrewAILLM(llm_adapter=adapter, model="gpt-4o")
        text = llm.complete("Hello from CrewAI!")

    Or, if your CrewAI version accepts async callables:

        text = await llm.acomplete("Hello async!")

    Streaming is also supported:

        # Sync streaming
        for token in llm.stream("Hello!"):
            print(token, end="", flush=True)

        # Async streaming
        async for token in llm.astream("Hello!"):
            print(token, end="", flush=True)

    Attributes
    ----------
    llm_adapter:
        Underlying Corpus `LLMProtocolV1` implementation.
    model:
        Default model identifier to send to Corpus.
    temperature:
        Default sampling temperature.
    max_tokens:
        Default max_tokens limit (adapter may have its own default).
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
    # Translator (lazy, cached)
    # ------------------------------------------------------------------ #

    class _CrewAILLMFrameworkTranslator(DefaultLLMFrameworkTranslator):
        """
        CrewAI-specific LLM framework translator.

        Currently this subclass does not override any behavior from
        `DefaultLLMFrameworkTranslator`, but it exists to mirror the other
        framework adapters and to provide a dedicated hook for CrewAI-specific
        customizations in the future.
        """

        pass

    @property
    def _translator(self) -> LLMTranslator:
        """
        Lazily construct the `LLMTranslator`.

        We use a simple property here instead of cached_property to avoid
        importing functools for environments that prefer minimal imports, but
        the semantics are effectively "construct once, reuse".
        """
        if not hasattr(self, "_translator_instance"):
            framework_translator = self._CrewAILLMFrameworkTranslator()
            self._translator_instance = LLMTranslator(
                adapter=self._llm,
                framework="crewai",
                translator=framework_translator,
            )
        return self._translator_instance  # type: ignore[attr-defined]

    # ------------------------------------------------------------------ #
    # Error-context helpers
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
            extra: Dict[str, Any] = {}
            attrs = getattr(ctx, "attrs", {})
            if isinstance(attrs, Mapping):
                crewai_info = attrs.get("crewai", {})
                if isinstance(crewai_info, Mapping):
                    for key in [
                        "agent_role",
                        "agent_goal",
                        "task_description",
                        "crew_id",
                        "crew_name",
                    ]:
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
            extra: Dict[str, Any] = {}
            attrs = getattr(ctx, "attrs", {})
            if isinstance(attrs, Mapping):
                crewai_info = attrs.get("crewai", {})
                if isinstance(crewai_info, Mapping):
                    for key in [
                        "agent_role",
                        "agent_goal",
                        "task_description",
                        "crew_id",
                        "crew_name",
                    ]:
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
            raise

    # ------------------------------------------------------------------ #
    # Core async API
    # ------------------------------------------------------------------ #

    async def acomplete(
        self,
        messages: Union[CrewAIMessageInput, CrewAIMessageSequence],
        **kwargs: Any,
    ) -> str:
        """
        Async completion.

        Parameters
        ----------
        messages:
            Prompt or chat messages:
                - str
                - {"role": "...", "content": "..."}
                - Sequence[str | mapping]

        kwargs:
            Sampling parameters and optional context hints:
                - model, temperature, max_tokens, top_p, frequency_penalty,
                  presence_penalty, stop/stop_sequences, seed, top_k
                - ctx: Optional OperationContext
                - request_id, traceparent, tenant, attrs
                - CrewAI context: agent, task, crew_id, etc.

        Returns
        -------
        str
            Final response text from the underlying Corpus LLM.
        """
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

        async with self._error_context_async(
            "acomplete",
            stream=False,
            messages_count=messages_count,
            model=model_for_context,
            ctx=ctx,
            params=params,
        ):
            result: LLMCompletion = await self._translator.arun_complete(
                messages=corpus_messages,
                op_ctx=ctx,
                params=params,
            )
            return result.text

    async def astream(
        self,
        messages: Union[CrewAIMessageInput, CrewAIMessageSequence],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Async streaming completion.

        Yields partial text chunks as they arrive from the Corpus LLM.
        """
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

        async def _gen() -> AsyncIterator[str]:
            async with self._error_context_async(
                "astream",
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
                    yield getattr(chunk, "text", "") or ""

        return _gen()

    # ------------------------------------------------------------------ #
    # Sync API (for typical CrewAI usage)
    # ------------------------------------------------------------------ #

    def complete(
        self,
        messages: Union[CrewAIMessageInput, CrewAIMessageSequence],
        **kwargs: Any,
    ) -> str:
        """
        Sync completion.

        Thin sync wrapper around the LLMTranslator's `complete`.
        """
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

        with self._error_context(
            "complete_sync",
            stream=False,
            messages_count=messages_count,
            model=model_for_context,
            ctx=ctx,
            params=params,
        ):
            result: LLMCompletion = self._translator.complete(
                messages=corpus_messages,
                op_ctx=ctx,
                params=params,
            )
            return result.text

    # Make the instance directly callable for convenience:
    __call__ = complete

    def stream(
        self,
        messages: Union[CrewAIMessageInput, CrewAIMessageSequence],
        **kwargs: Any,
    ) -> Iterator[str]:
        """
        Sync streaming completion.

        Uses the LLMTranslator's `stream` method, which internally handles
        any async→sync bridging needed by the underlying protocol adapter.

        Parameters
        ----------
        messages:
            Prompt or chat messages.

        kwargs:
            Same sampling and context parameters as in `complete`/`acomplete`.

        Returns
        -------
        Iterator[str]
            Iterator of partial text chunks.
        """
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

        def _iter() -> Iterator[str]:
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
                    yield getattr(chunk, "text", "") or ""

        return _iter()


__all__ = [
    "CorpusCrewAILLM",
    "CorpusCrewAILLMConfig",
    "CrewAIMessageInput",
    "CrewAIMessageSequence",
]
