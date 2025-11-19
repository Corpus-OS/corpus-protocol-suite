# corpus_sdk/llm/framework_adapters/crewai.py
# SPDX-License-Identifier: Apache-2.0

"""
CrewAI adapter for Corpus LLM protocol.

This module exposes a Corpus `BaseLLMAdapter` as a CrewAI-compatible LLM
wrapper. It is intentionally framework-agnostic on the CrewAI side and
focuses on:

- Async + sync completion APIs
- Async + sync streaming (true incremental streaming)
- Context propagation via `OperationContext` where possible
- Error context attachment for better observability
- Production-grade sync streaming using `SyncStreamBridge`

Design goals
------------

1. Protocol-first:
   CrewAI is treated as an integration surface. All real behavior flows
   through `BaseLLMAdapter` and the LLM protocol in `llm_base.py`.

2. Optional dependency safe:
   Import of `crewai` is guarded. Importing this module is safe even if
   CrewAI is not installed. Instantiating `CorpusCrewAILLM` will raise
   a clear `RuntimeError` if CrewAI is missing (to avoid confusing
   runtime failures in agent code).

3. Simple & explicit interface:
   This adapter exposes a small, explicit API that is easy to use from
   CrewAI agents and tasks:

       llm = CorpusCrewAILLM(corpus_adapter=adapter, model="gpt-4")
       text = llm.complete("Hello!")
       async for token in llm.astream("Hello!"):
           ...

   You can typically pass this `llm` instance as the LLM for a CrewAI
   Agent, as long as the Agent expects a callable that takes a prompt
   (or messages) and returns text.

4. True streaming:
   - Async streaming uses `BaseLLMAdapter.stream()` directly.
   - Sync streaming uses `SyncStreamBridge` to bridge the async iterator
     to a blocking generator without buffering the entire response.

5. Context + observability:
   - Basic `OperationContext` is built from kwargs (request_id, tenant,
     attrs) when provided.
   - Errors are enriched with framework-specific context via
     `attach_context(framework="crewai", ...)`.

6. Cancellation-friendly:
   - Sync streaming optionally accepts a `cancel_event` (threading.Event)
     via kwargs to allow callers to stop consumption early.

This is SDK infrastructure, not business logic. It does not attempt to
mirror CrewAI's internal LLM configuration model; instead, it focuses on
being a robust, callable LLM wrapper that CrewAI can invoke.
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
    Protocol,
    Sequence,
    Tuple,
    Type,
    Union,
    runtime_checkable,
)

from corpus_sdk.llm.llm_base import (
    BaseLLMAdapter,
    LLMChunk,
    LLMCompletion,
    OperationContext,
    TransientNetwork,
    Unavailable,
)
from corpus_sdk.llm.framework_adapters.common.async_bridge import AsyncBridge
from corpus_sdk.llm.framework_adapters.common.error_context import attach_context
from corpus_sdk.llm.framework_adapters.common.sync_stream_bridge import SyncStreamBridge
from corpus_sdk.llm.framework_adapters.common.message_translation import (
    from_crewai,
    to_corpus,
    NormalizedMessage,
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
    the optional dependency.
    """
    if _CREWAI_IMPORT_ERROR is not None:
        raise RuntimeError(
            "CrewAI is required to use CorpusCrewAILLM. "
            "Install with: pip install 'crewai>=0.51.0'"
        ) from _CREWAI_IMPORT_ERROR


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_messages(
    input_messages: Union[CrewAIMessageInput, CrewAIMessageSequence],
) -> List[Mapping[str, str]]:
    """
    Normalize various CrewAI-friendly inputs into Corpus wire messages.

    Uses the shared message translation infrastructure.
    """
    if isinstance(input_messages, (str, Mapping)) or hasattr(input_messages, 'role'):
        messages_list: Sequence[CrewAIMessageInput] = [input_messages]
    elif isinstance(input_messages, Sequence):
        messages_list = input_messages
    else:
        # Extreme fallback: stringify
        messages_list = [str(input_messages)]

    # Use shared message translation
    normalized: List[NormalizedMessage] = [from_crewai(m) for m in messages_list]
    return [dict(m) for m in to_corpus(normalized)]


def _extract_stop_sequences(kwargs: Mapping[str, Any]) -> Optional[List[str]]:
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
    Build a comprehensive OperationContext from generic kwargs.

    This intentionally does *not* depend on CrewAI internals; it simply
    looks for common tracing / tenancy keys that callers may pass through
    (e.g., from a higher-level router).

    Recognized keys (all optional):
        - ctx: OperationContext (if already provided)
        - request_id, traceparent, tenant
        - attrs (mapping)
        - CrewAI context: agent, task, crew_id, etc.
    """
    ctx = kwargs.get("ctx")
    if isinstance(ctx, OperationContext):
        return ctx

    request_id = kwargs.get("request_id")
    traceparent = kwargs.get("traceparent")
    tenant = kwargs.get("tenant")
    attrs = kwargs.get("attrs") or {}

    # Extract CrewAI-specific context if available
    crewai_context = {}
    for key in ['agent', 'task', 'crew_id', 'agent_role', 'agent_goal', 
                'task_description', 'crew_name', 'process_id']:
        if key in kwargs:
            crewai_context[key] = kwargs[key]

    try:
        if not isinstance(attrs, Mapping):
            attrs = {"raw_attrs": attrs}
        
        # Merge CrewAI context into attrs
        if crewai_context:
            attrs = {**attrs, "crewai": crewai_context}
    except Exception:  # noqa: BLE001
        attrs = {}

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
    
    # Build complete parameter set with proper precedence
    params = {
        # Sampling parameters
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
    
    # Remove None values to keep the parameter set clean
    return {k: v for k, v in params.items() if v is not None}


def _build_error_context(
    operation: str,
    messages_count: int,
    model: str,
    ctx: OperationContext,
    stream: bool = False,
    **additional: Any,
) -> Dict[str, Any]:
    """
    Build rich error context for observability.
    
    Mirrors the context richness of LangChain/LlamaIndex adapters.
    """
    error_ctx = {
        "framework": "crewai",
        "operation": operation,
        "messages_count": messages_count,
        "model": model,
        "stream": stream,
        "request_id": getattr(ctx, "request_id", None),
        "tenant": getattr(ctx, "tenant", None),
        "timestamp": time.time(),
    }
    
    # Add any CrewAI-specific context from operation context
    attrs = getattr(ctx, "attrs", {})
    if isinstance(attrs, Mapping):
        crewai_info = attrs.get("crewai", {})
        for key in ['agent_role', 'agent_goal', 'task_description', 'crew_id', 'crew_name']:
            if key in crewai_info:
                error_ctx[key] = crewai_info[key]
    
    error_ctx.update(additional)
    return {k: v for k, v in error_ctx.items() if v is not None}


# ---------------------------------------------------------------------------
# Configuration Builder
# ---------------------------------------------------------------------------

class CorpusCrewAILLMConfig:
    """
    Configuration builder for CorpusCrewAILLM to simplify setup.
    
    Usage:
        config = (CorpusCrewAILLMConfig(adapter)
                 .with_model("gpt-4")
                 .with_temperature(0.7)
                 .with_streaming(queue_maxsize=32))
        llm = config.build()
    """
    
    def __init__(self, corpus_adapter: BaseLLMAdapter):
        self.corpus_adapter = corpus_adapter
        self.model = "default"
        self.temperature = 0.7
        self.max_tokens: Optional[int] = None
        
        # Streaming defaults
        self.stream_queue_maxsize = 16
        self.stream_poll_timeout_s = 0.1
        self.stream_join_timeout_s = 2.0
        
        # Transient retry defaults (now opt-in)
        self.max_transient_retries = 0  # Default: no retries
        self.transient_backoff_s = 0.25
        self.stream_transient_error_types: Tuple[Type[BaseException], ...] = (
            TransientNetwork,
            Unavailable,
        )
        
        # Testing
        self.stream_bridge_factory: Optional[Callable[..., SyncStreamBridge]] = None
        self.require_crewai = False
    
    def with_model(self, model: str) -> CorpusCrewAILLMConfig:
        self.model = model
        return self
    
    def with_temperature(self, temperature: float) -> CorpusCrewAILLMConfig:
        self.temperature = temperature
        return self
    
    def with_max_tokens(self, max_tokens: Optional[int]) -> CorpusCrewAILLMConfig:
        self.max_tokens = max_tokens
        return self
    
    def with_streaming(
        self,
        queue_maxsize: int = 16,
        poll_timeout_s: float = 0.1,
        join_timeout_s: float = 2.0,
    ) -> CorpusCrewAILLMConfig:
        self.stream_queue_maxsize = queue_maxsize
        self.stream_poll_timeout_s = poll_timeout_s
        self.stream_join_timeout_s = join_timeout_s
        return self
    
    def with_retry_policy(
        self,
        max_retries: int = 3,
        backoff_s: float = 0.25,
        error_types: Optional[Tuple[Type[BaseException], ...]] = None,
    ) -> CorpusCrewAILLMConfig:
        """Opt-in retry policy for transient errors."""
        self.max_transient_retries = max_retries
        self.transient_backoff_s = backoff_s
        if error_types is not None:
            self.stream_transient_error_types = error_types
        return self
    
    def with_crewai_required(self, required: bool = True) -> CorpusCrewAILLMConfig:
        self.require_crewai = required
        return self
    
    def build(self) -> CorpusCrewAILLM:
        return CorpusCrewAILLM(
            corpus_adapter=self.corpus_adapter,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream_queue_maxsize=self.stream_queue_maxsize,
            stream_poll_timeout_s=self.stream_poll_timeout_s,
            stream_join_timeout_s=self.stream_join_timeout_s,
            max_transient_retries=self.max_transient_retries,
            transient_backoff_s=self.transient_backoff_s,
            stream_transient_error_types=self.stream_transient_error_types,
            stream_bridge_factory=self.stream_bridge_factory,
            require_crewai=self.require_crewai,
        )


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------

class CorpusCrewAILLM:
    """
    CrewAI-compatible LLM wrapper backed by a Corpus `BaseLLMAdapter`.

    This class is deliberately small and explicit. CrewAI code can treat
    it as a callable LLM:

        llm = CorpusCrewAILLM(corpus_adapter=adapter, model="gpt-4o")
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
    corpus_adapter:
        Underlying Corpus `BaseLLMAdapter` instance.
    model:
        Default model identifier to send to Corpus.
    temperature:
        Default sampling temperature.
    max_tokens:
        Default max_tokens limit (adapter may have its own default).

    stream_queue_maxsize:
        Max number of pending text chunks buffered between the background
        thread and the caller in sync streaming.
    stream_poll_timeout_s:
        Timeout (seconds) when polling the queue in sync streaming.
    stream_join_timeout_s:
        Timeout (seconds) when joining the background streaming thread.

    max_transient_retries:
        Number of retry attempts for transient errors during sync streaming
        before the first chunk is emitted. Default: 0 (no retry).
    transient_backoff_s:
        Initial backoff delay (seconds) for streaming retry.
    stream_transient_error_types:
        Tuple of exception types to consider transient and eligible for retry
        during sync streaming. Default: (TransientNetwork, Unavailable).

    stream_bridge_factory:
        Optional factory function for creating SyncStreamBridge instances.
        Primarily used for testing/mocking. If None, uses default factory.
    """

    corpus_adapter: BaseLLMAdapter

    # Defaults for sampling
    model: str
    temperature: float
    max_tokens: Optional[int]

    # Sync streaming configuration
    stream_queue_maxsize: int
    stream_poll_timeout_s: float
    stream_join_timeout_s: float

    # Transient retry configuration for sync streaming (now opt-in)
    max_transient_retries: int
    transient_backoff_s: float
    stream_transient_error_types: Tuple[Type[BaseException], ...]

    # Dependency injection for testing
    stream_bridge_factory: Optional[Callable[..., SyncStreamBridge]]

    def __init__(
        self,
        *,
        corpus_adapter: BaseLLMAdapter,
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        # Streaming config
        stream_queue_maxsize: int = 16,
        stream_poll_timeout_s: float = 0.1,
        stream_join_timeout_s: float = 2.0,
        # Transient retry for sync streaming (opt-in, default: no retries)
        max_transient_retries: int = 0,
        transient_backoff_s: float = 0.25,
        stream_transient_error_types: Tuple[Type[BaseException], ...] = (
            TransientNetwork,
            Unavailable,
        ),
        # Optional hook for tests
        stream_bridge_factory: Optional[Callable[..., SyncStreamBridge]] = None,
        # Optional: ensure CrewAI is installed if the user wants hard safety
        require_crewai: bool = False,
    ) -> None:
        if require_crewai:
            _ensure_crewai_installed()

        self.corpus_adapter = corpus_adapter
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = max_tokens

        self.stream_queue_maxsize = int(stream_queue_maxsize)
        self.stream_poll_timeout_s = float(stream_poll_timeout_s)
        self.stream_join_timeout_s = float(stream_join_timeout_s)

        self.max_transient_retries = int(max_transient_retries)
        self.transient_backoff_s = float(transient_backoff_s)
        self.stream_transient_error_types = stream_transient_error_types

        self.stream_bridge_factory = stream_bridge_factory

    # ------------------------------------------------------------------ #
    # Internal: stream bridge creation
    # ------------------------------------------------------------------ #

    def _create_stream_bridge(
        self,
        *,
        coro_factory: Callable[[], Awaitable[AsyncIterator[str]]],
        error_context: Dict[str, Any],
        cancel_event: Optional[threading.Event],
        stream_overrides: Optional[Mapping[str, Any]] = None,
    ) -> SyncStreamBridge:
        """
        Create a SyncStreamBridge instance with current configuration.
        """
        overrides = dict(stream_overrides or {})

        queue_maxsize = overrides.get("stream_queue_maxsize", self.stream_queue_maxsize)
        poll_timeout_s = overrides.get("stream_poll_timeout_s", self.stream_poll_timeout_s)
        join_timeout_s = overrides.get("stream_join_timeout_s", self.stream_join_timeout_s)
        max_retries = overrides.get("stream_max_transient_retries", self.max_transient_retries)
        backoff_s = overrides.get("stream_transient_backoff_s", self.transient_backoff_s)
        error_types = overrides.get(
            "stream_transient_error_types", self.stream_transient_error_types
        )

        if error_types is not None and not isinstance(error_types, tuple):
            raise TypeError("stream_transient_error_types must be a tuple of exception types")

        if self.stream_bridge_factory is not None:
            return self.stream_bridge_factory(
                coro_factory=coro_factory,
                queue_maxsize=queue_maxsize,
                poll_timeout_s=poll_timeout_s,
                join_timeout_s=join_timeout_s,
                cancel_event=cancel_event,
                framework="crewai",
                error_context=error_context,
                max_transient_retries=max_retries,
                transient_backoff_s=backoff_s,
                transient_error_types=error_types,
            )

        return SyncStreamBridge(
            coro_factory=coro_factory,
            queue_maxsize=queue_maxsize,
            poll_timeout_s=poll_timeout_s,
            join_timeout_s=join_timeout_s,
            cancel_event=cancel_event,
            framework="crewai",
            error_context=error_context,
            max_transient_retries=max_retries,
            transient_backoff_s=backoff_s,
            transient_error_types=error_types,
        )

    # ------------------------------------------------------------------ #
    # Core async API
    # ------------------------------------------------------------------ #

    async def _astream_from_corpus_messages(
        self,
        corpus_messages: List[Mapping[str, str]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Internal async streaming from pre-normalized Corpus messages.
        
        This exists so stream() can avoid double normalization and share
        error-context logic with astream.
        
        Yields partial text chunks as they arrive from the Corpus adapter.
        """
        ctx = _build_operation_context_from_kwargs(kwargs)
        params = _build_sampling_params(
            default_model=self.model,
            default_temperature=self.temperature,
            default_max_tokens=self.max_tokens,
            kwargs=kwargs,
        )

        model_for_context = params.get("model", self.model)

        try:
            stream: AsyncIterator[LLMChunk] = await self.corpus_adapter.stream(
                messages=corpus_messages,
                ctx=ctx,
                **params,
            )

            try:
                async for chunk in stream:
                    yield getattr(chunk, "text", "") or ""
            finally:
                aclose = getattr(stream, "aclose", None)
                if callable(aclose):
                    try:
                        await aclose()
                    except Exception as cleanup_error:  # noqa: BLE001
                        logger.debug(
                            "CrewAI stream cleanup failed: %s",
                            cleanup_error,
                            extra={"framework": "crewai"},
                        )

        except BaseException as exc:  # noqa: BLE001
            error_ctx = _build_error_context(
                operation="astream",
                messages_count=len(corpus_messages),
                model=model_for_context,
                ctx=ctx,
                stream=True,
                temperature=params.get("temperature"),
                max_tokens=params.get("max_tokens"),
                top_p=params.get("top_p"),
                frequency_penalty=params.get("frequency_penalty"),
                presence_penalty=params.get("presence_penalty"),
            )
            attach_context(exc, **error_ctx)
            raise

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
            Final response text from the underlying Corpus adapter.
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

        try:
            result: LLMCompletion = await self.corpus_adapter.complete(
                messages=corpus_messages,
                ctx=ctx,
                **params,
            )

            return result.text
        except BaseException as exc:  # noqa: BLE001
            error_ctx = _build_error_context(
                operation="acomplete",
                messages_count=len(corpus_messages),
                model=model_for_context,
                ctx=ctx,
                stream=False,
                temperature=params.get("temperature"),
                max_tokens=params.get("max_tokens"),
                top_p=params.get("top_p"),
                frequency_penalty=params.get("frequency_penalty"),
                presence_penalty=params.get("presence_penalty"),
            )
            attach_context(exc, **error_ctx)
            raise

    async def astream(
        self,
        messages: Union[CrewAIMessageInput, CrewAIMessageSequence],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Async streaming completion.

        Yields partial text chunks as they arrive from the Corpus adapter.
        """
        corpus_messages = _normalize_messages(messages)
        async for chunk in self._astream_from_corpus_messages(corpus_messages, **kwargs):
            yield chunk

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

        Thin sync wrapper around `acomplete` using `AsyncBridge`.
        """
        try:
            return AsyncBridge.run_async(self.acomplete(messages, **kwargs))
        except BaseException as exc:  # noqa: BLE001
            corpus_messages = _normalize_messages(messages)
            ctx = _build_operation_context_from_kwargs(kwargs)
            error_ctx = _build_error_context(
                operation="complete",
                messages_count=len(corpus_messages),
                model=kwargs.get("model", self.model),
                ctx=ctx,
                stream=False,
            )
            attach_context(exc, **error_ctx)
            raise

    # Make the instance directly callable for convenience:
    __call__ = complete

    def stream(
        self,
        messages: Union[CrewAIMessageInput, CrewAIMessageSequence],
        **kwargs: Any,
    ) -> Iterator[str]:
        """
        Sync streaming completion.

        Bridges `astream` to a blocking generator using `SyncStreamBridge`.

        Parameters
        ----------
        messages:
            Prompt or chat messages.

        kwargs:
            - cancel_event: Optional[threading.Event] for early cancellation
            - Per-call streaming overrides:
                * stream_queue_maxsize
                * stream_poll_timeout_s
                * stream_join_timeout_s
                * stream_max_transient_retries (opt-in retry policy)
                * stream_transient_backoff_s
                * stream_transient_error_types
            - All other parameters as in `acomplete`/`astream`.

        Returns
        -------
        Iterator[str]
            Iterator of partial text chunks.
        """
        cancel_event = kwargs.pop("cancel_event", None)
        if cancel_event is not None and not isinstance(cancel_event, threading.Event):
            logger.warning(
                "stream cancel_event is not a threading.Event; ignoring",
                extra={"framework": "crewai"},
            )
            cancel_event = None

        # Extract per-call streaming overrides (mirroring LlamaIndex adapter)
        stream_overrides: Dict[str, Any] = {
            "stream_queue_maxsize": kwargs.pop(
                "stream_queue_maxsize", self.stream_queue_maxsize
            ),
            "stream_poll_timeout_s": kwargs.pop(
                "stream_poll_timeout_s", self.stream_poll_timeout_s
            ),
            "stream_join_timeout_s": kwargs.pop(
                "stream_join_timeout_s", self.stream_join_timeout_s
            ),
            "stream_max_transient_retries": kwargs.pop(
                "stream_max_transient_retries", self.max_transient_retries
            ),
            "stream_transient_backoff_s": kwargs.pop(
                "stream_transient_backoff_s", self.transient_backoff_s
            ),
            "stream_transient_error_types": kwargs.pop(
                "stream_transient_error_types", self.stream_transient_error_types
            ),
        }

        # Validate override types where necessary
        error_types = stream_overrides["stream_transient_error_types"]
        if error_types is not None and not isinstance(error_types, tuple):
            raise TypeError("stream_transient_error_types must be a tuple of exception types")

        corpus_messages = _normalize_messages(messages)
        model_for_context = kwargs.get("model", self.model)
        ctx = _build_operation_context_from_kwargs(kwargs)

        async def _coro_factory() -> AsyncIterator[str]:
            # Use internal async generator directly (no await needed)
            return self._astream_from_corpus_messages(corpus_messages, **kwargs)

        error_context = _build_error_context(
            operation="stream",
            messages_count=len(corpus_messages),
            model=model_for_context,
            ctx=ctx,
            stream=True,
            **stream_overrides,
        )

        bridge = self._create_stream_bridge(
            coro_factory=_coro_factory,
            error_context=error_context,
            cancel_event=cancel_event,
            stream_overrides=stream_overrides,
        )

        return bridge.run()

    # ------------------------------------------------------------------ #
    # Optional: token counting helper
    # ------------------------------------------------------------------ #

    async def acount_tokens(
        self,
        messages: Union[CrewAIMessageInput, CrewAIMessageSequence],
        **kwargs: Any,
    ) -> int:
        """
        Optional async helper that uses Corpus `count_tokens` when available.
        """
        if not hasattr(self.corpus_adapter, "count_tokens"):
            raise NotImplementedError(
                "Underlying Corpus adapter does not support count_tokens()"
            )

        corpus_messages = _normalize_messages(messages)
        combined = "\n".join(f"{m['role']}:{m['content']}" for m in corpus_messages)

        ctx = _build_operation_context_from_kwargs(kwargs)
        model_for_context = kwargs.get("model", self.model)

        try:
            return await self.corpus_adapter.count_tokens(
                text=combined,
                model=model_for_context,
                ctx=ctx,
            )
        except BaseException as exc:  # noqa: BLE001
            error_ctx = _build_error_context(
                operation="acount_tokens",
                messages_count=len(corpus_messages),
                model=model_for_context,
                ctx=ctx,
                stream=False,
            )
            attach_context(exc, **error_ctx)
            raise

    def count_tokens(
        self,
        messages: Union[CrewAIMessageInput, CrewAIMessageSequence],
        **kwargs: Any,
    ) -> int:
        """
        Sync token counting helper.

        Uses async `acount_tokens` when supported. If the Corpus adapter
        does not implement `count_tokens`, raises NotImplementedError
        rather than using naive fallback to avoid misleading results.
        """
        try:
            return AsyncBridge.run_async(self.acount_tokens(messages, **kwargs))
        except (NotImplementedError, AttributeError):
            logger.warning(
                "count_tokens not supported by Corpus adapter; "
                "consider implementing a proper token counter",
                extra={"framework": "crewai"},
            )
            raise NotImplementedError(
                "Token counting is not supported by the underlying Corpus adapter. "
                "Please implement a proper token counter for accurate results."
            )
        except BaseException as exc:  # noqa: BLE001
            corpus_messages = _normalize_messages(messages)
            ctx = _build_operation_context_from_kwargs(kwargs)
            error_ctx = _build_error_context(
                operation="count_tokens",
                messages_count=len(corpus_messages),
                model=kwargs.get("model", self.model),
                ctx=ctx,
                stream=False,
            )
            attach_context(exc, **error_ctx)
            raise


__all__ = [
    "CorpusCrewAILLM",
    "CorpusCrewAILLMConfig",
    "CrewAIMessageInput",
    "CrewAIMessageSequence",
]
