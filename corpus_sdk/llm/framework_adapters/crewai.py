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
- Rich error context attachment for observability
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

import logging
from functools import wraps
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
    TypedDict,
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
    LLMChunk,
    LLMCompletion,
    LLMProtocolV1,
    OperationContext,
)

logger = logging.getLogger(__name__)

# Type variables for decorators
T = TypeVar("T")

# ---------------------------------------------------------------------------
# Type definitions for CrewAI compatibility
# ---------------------------------------------------------------------------


@runtime_checkable
class CrewAIMessage(Protocol):
    """Protocol for CrewAI message-like objects."""
    
    @property
    def role(self) -> str: ...
    
    @property 
    def content(self) -> str: ...


# Type aliases for better readability
CrewAIMessageInput = Union[str, Mapping[str, Any], CrewAIMessage]
CrewAIMessageSequence = Sequence[CrewAIMessageInput]


class CrewAIContext(TypedDict, total=False):
    """Structured type for CrewAI execution context."""
    agent_role: Optional[str]
    agent_goal: Optional[str]
    task_description: Optional[str]
    crew_id: Optional[str]
    crew_name: Optional[str]
    process_id: Optional[str]
    task_id: Optional[str]


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
# Error Context Helpers (Extracted for Clarity)
# ---------------------------------------------------------------------------

def _extract_role_content(msg: Any) -> tuple[str, str]:
    """Extract role and content from various message formats."""
    if isinstance(msg, Mapping):
        return msg.get("role", "unknown"), msg.get("content", "")
    elif hasattr(msg, "role") and hasattr(msg, "content"):
        return getattr(msg, "role", "unknown"), getattr(msg, "content", "")
    else:  # string message
        return "user", str(msg)


def _analyze_message_metrics(messages: Sequence[Any]) -> tuple[Dict[str, int], int]:
    """Analyze message roles and content metrics."""
    roles: Dict[str, int] = {}
    total_chars = 0
    
    for msg in messages:
        role, content = _extract_role_content(msg)
        roles[role] = roles.get(role, 0) + 1
        if isinstance(content, str):
            total_chars += len(content)
    
    return roles, total_chars


def _extract_kwargs_context(kwargs: Dict[str, Any], target: Dict[str, Any]) -> None:
    """Extract relevant context from kwargs into target dict."""
    # CrewAI-specific context
    crewai_context_keys = [
        "agent_role", "agent_goal", "task_description", 
        "crew_id", "crew_name", "process_id", "task_id"
    ]
    for key in crewai_context_keys:
        if key in kwargs:
            target[key] = kwargs[key]
    
    # Sampling parameters
    sampling_params = [
        "temperature", "max_tokens", "top_p", "frequency_penalty", 
        "presence_penalty", "stop", "seed"
    ]
    for param in sampling_params:
        if param in kwargs:
            target[param] = kwargs[param]


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
        "temperature": getattr(instance, "temperature", 0.7),
        "operation": operation,
    }
    
    # Extract message metrics
    if args:
        messages = _extract_messages_from_args(args[0])
        if messages:
            dynamic_ctx["messages_count"] = len(messages)
            roles, total_chars = _analyze_message_metrics(messages)
            dynamic_ctx["roles_distribution"] = roles
            dynamic_ctx["total_content_chars"] = total_chars
    
    # Extract context from kwargs
    _extract_kwargs_context(kwargs, dynamic_ctx)
    
    # Stream flag for streaming operations
    if operation in ["astream", "stream"]:
        dynamic_ctx["stream"] = True
    
    return dynamic_ctx


def _extract_messages_from_args(first_arg: Any) -> list[Any]:
    """Safely extract messages from method arguments."""
    if isinstance(first_arg, (str, Mapping)) or hasattr(first_arg, "role"):
        return [first_arg]
    elif isinstance(first_arg, (list, tuple)):
        return list(first_arg)
    elif first_arg is not None:
        return [str(first_arg)]
    return []


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
                    dynamic_context = _extract_dynamic_context(self, args, kwargs, operation)
                    full_context = {**static_context, **dynamic_context}
                    
                    try:
                        return await func(self, *args, **kwargs)
                    except Exception as exc:
                        attach_context(
                            exc,
                            framework="crewai",
                            operation=f"llm_{operation}",
                            **full_context,
                        )
                        raise
                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
                    dynamic_context = _extract_dynamic_context(self, args, kwargs, operation)
                    full_context = {**static_context, **dynamic_context}
                    
                    try:
                        return func(self, *args, **kwargs)
                    except Exception as exc:
                        attach_context(
                            exc,
                            framework="crewai",
                            operation=f"llm_{operation}",
                            **full_context,
                        )
                        raise
                return sync_wrapper
        return decorator
    return decorator_factory


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


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _normalize_messages(
    input_messages: Union[CrewAIMessageInput, CrewAIMessageSequence],
) -> Sequence[Mapping[str, Any]]:
    """
    Normalize various CrewAI-friendly inputs into Corpus wire messages.
    
    Returns
    -------
    Sequence[Mapping[str, Any]]
        Normalized Corpus wire format messages
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

    Returns
    -------
    OperationContext
        Context object with CrewAI execution context
    """
    # If caller already built an OperationContext, trust it.
    ctx = kwargs.get("ctx")
    if isinstance(ctx, OperationContext):
        return ctx

    framework_version = kwargs.get("framework_version")
    task = kwargs.get("task")

    # Only include fields that are meaningful for context translation.
    allowed_keys = {
        "request_id", "traceparent", "tenant", "attrs", "agent",
        "crew_id", "agent_role", "agent_goal", "task_description",
        "crew_name", "process_id", "task_id",
    }

    context_kwargs: Dict[str, Any] = {
        k: v for k, v in kwargs.items() if k in allowed_keys
    }

    ctx = from_crewai_context(
        task=task,
        framework_version=framework_version,
        **context_kwargs,
    )

    logger.debug(
        "Built OperationContext from CrewAI context with agent_role: %s, crew_id: %s",
        context_kwargs.get("agent_role", "unknown"),
        context_kwargs.get("crew_id", "unknown"),
    )

    return ctx


def _build_sampling_params(
    *,
    default_model: str,
    default_temperature: float,
    default_max_tokens: Optional[int],
    kwargs: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build complete sampling parameters from kwargs and defaults.
    
    Returns
    -------
    Dict[str, Any]
        Cleaned sampling parameters with None values removed
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

    # Drop None values so adapters see only explicit parameters
    clean_params = {k: v for k, v in params.items() if v is not None}

    logger.debug(
        "Built sampling params for CrewAI: model=%s, temperature=%.2f, max_tokens=%s",
        clean_params.get("model", "default"),
        clean_params.get("temperature", 0.7),
        clean_params.get("max_tokens", "default"),
    )

    return clean_params


def _extract_completion_text(result: Any) -> str:
    """Extract text content from various completion result types.
    
    Returns
    -------
    str
        Extracted text content
    """
    if isinstance(result, LLMCompletion):
        return result.text
    if hasattr(result, "text"):
        return getattr(result, "text")
    return str(result)


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------


class CorpusCrewAILLM:
    """
    CrewAI-compatible LLM wrapper backed by a Corpus `LLMProtocolV1`.

    Provides direct access to the LLM protocol with a CrewAI-friendly interface.

    Example:
    ```python
    from corpus_sdk.llm.framework_adapters.crewai import CorpusCrewAILLM
    from crewai import Agent, Task, Crew

    # Initialize with any Corpus LLMProtocolV1 adapter
    llm = CorpusCrewAILLM(
        llm_adapter=my_adapter,
        model="gpt-4",
        temperature=0.7,
        max_tokens=1000
    )

    # Use with CrewAI agent
    researcher = Agent(
        role="Senior Research Analyst",
        goal="Uncover breakthrough AI research insights",
        backstory="Expert analyst with deep technical understanding",
        llm=llm,  # Direct LLM assignment
        tools=[web_search_tool, document_processor]
    )

    # The LLM automatically receives CrewAI context during task execution
    task = Task(
        description="Research latest LLM architectures",
        agent=researcher,
        expected_output="Comprehensive analysis report"
    )

    crew = Crew(agents=[researcher], tasks=[task])
    result = crew.kickoff()
    ```

    Error Handling Example:
    ```python
    try:
        response = llm.complete(
            "Analyze the latest AI developments",
            agent_role="Research Analyst",
            crew_id="ai_research_team",
            temperature=0.8
        )
    except Exception as e:
        # Rich error context automatically attached with agent info, message counts, etc.
        logger.error("LLM call failed with context", exc_info=e)
    ```
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
        """Initialize the CrewAI LLM adapter.
        
        Parameters
        ----------
        llm_adapter : LLMProtocolV1
            Corpus LLM protocol adapter instance
        model : str, optional
            Default model name, by default "default"
        temperature : float, optional
            Default sampling temperature, by default 0.7
        max_tokens : Optional[int], optional
            Default maximum tokens, by default None
        framework_version : Optional[str], optional
            CrewAI framework version for context, by default None
        require_crewai : bool, optional
            Whether to validate CrewAI installation, by default False
            
        Raises
        ------
        RuntimeError
            If CrewAI is required but not installed
        TypeError
            If llm_adapter doesn't implement required protocol
        """
        if require_crewai:
            _ensure_crewai_installed()

        if not hasattr(llm_adapter, "complete") or not callable(getattr(llm_adapter, "complete")):
            raise TypeError("llm_adapter must implement LLMProtocolV1 with 'complete' method")

        self._llm: LLMProtocolV1 = llm_adapter
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = max_tokens
        self._framework_version = framework_version

        logger.info(
            "CorpusCrewAILLM initialized with model=%s, temperature=%.2f, max_tokens=%s",
            self.model,
            self.temperature,
            self.max_tokens or "default",
        )

    def _apply_instance_defaults(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Apply instance-level defaults to kwargs.
        
        Returns
        -------
        Dict[str, Any]
            Updated kwargs with instance defaults applied
        """
        if "framework_version" in kwargs or self._framework_version is None:
            return dict(kwargs)

        updated = dict(kwargs)
        updated["framework_version"] = self._framework_version
        return updated

    def _execute_completion_sync(
        self,
        messages: Sequence[Mapping[str, Any]],
        ctx: OperationContext,
        params: Dict[str, Any],
    ) -> Any:
        """
        Execute a synchronous completion against the underlying protocol.

        Returns
        -------
        Any
            Raw result from the adapter
        """
        logger.debug(
            "Executing sync completion for %d messages with model: %s, agent_role: %s",
            len(messages),
            params.get("model", self.model),
            getattr(ctx, "attrs", {}).get("agent_role", "unknown"),
        )
        return self._llm.complete(
            messages=messages,
            ctx=ctx,
            **params,
        )

    async def _execute_completion_async(
        self,
        messages: Sequence[Mapping[str, Any]],
        ctx: OperationContext,
        params: Dict[str, Any],
    ) -> Any:
        """
        Execute an asynchronous completion against the underlying protocol.

        Returns
        -------
        Any
            Raw result from the adapter
        """
        logger.debug(
            "Executing async completion for %d messages with model: %s, agent_role: %s",
            len(messages),
            params.get("model", self.model),
            getattr(ctx, "attrs", {}).get("agent_role", "unknown"),
        )
        return await self._llm.acomplete(
            messages=messages,
            ctx=ctx,
            **params,
        )

    def _execute_streaming_sync(
        self,
        messages: Sequence[Mapping[str, Any]],
        ctx: OperationContext,
        params: Dict[str, Any],
    ) -> Iterator[str]:
        """
        Execute synchronous streaming against the underlying protocol.

        Yields
        ------
        Iterator[str]
            Plain text tokens derived from LLMChunk objects
        """
        logger.debug(
            "Executing sync streaming for %d messages with model: %s, agent_role: %s",
            len(messages),
            params.get("model", self.model),
            getattr(ctx, "attrs", {}).get("agent_role", "unknown"),
        )

        for chunk in self._llm.stream(
            messages=messages,
            ctx=ctx,
            **params,
        ):
            yield getattr(chunk, "text", "") or ""

    async def _execute_streaming_async(
        self,
        messages: Sequence[Mapping[str, Any]],
        ctx: OperationContext,
        params: Dict[str, Any],
    ) -> AsyncIterator[str]:
        """
        Execute asynchronous streaming against the underlying protocol.

        Yields
        ------
        AsyncIterator[str]
            Plain text tokens derived from LLMChunk objects
        """
        logger.debug(
            "Executing async streaming for %d messages with model: %s, agent_role: %s",
            len(messages),
            params.get("model", self.model),
            getattr(ctx, "attrs", {}).get("agent_role", "unknown"),
        )

        async for chunk in self._llm.astream(
            messages=messages,
            ctx=ctx,
            **params,
        ):
            yield getattr(chunk, "text", "") or ""

    @with_async_llm_error_context("acomplete")
    async def acomplete(
        self,
        messages: Union[CrewAIMessageInput, CrewAIMessageSequence],
        **kwargs: Any,
    ) -> str:
        """
        Async completion for CrewAI workflows.

        Parameters
        ----------
        messages : Union[CrewAIMessageInput, CrewAIMessageSequence]
            CrewAI message input (string, dict, message object, or sequence)
        **kwargs : Any
            - model, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, stop
            - agent_role, agent_goal, task_description, crew_id, crew_name, process_id
            - request_id, tenant, framework_version

        Returns
        -------
        str
            Completion text result

        Raises
        ------
        Exception
            Any exception from the underlying LLM with rich context attached
        """
        kwargs = self._apply_instance_defaults(kwargs)
        corpus_messages = _normalize_messages(messages)
        ctx = _build_operation_context_from_kwargs(kwargs)
        params = _build_sampling_params(
            default_model=self.model,
            default_temperature=self.temperature,
            default_max_tokens=self.max_tokens,
            kwargs=kwargs,
        )

        raw_result = await self._execute_completion_async(
            messages=corpus_messages,
            ctx=ctx,
            params=params,
        )

        return _extract_completion_text(raw_result)

    @with_async_llm_error_context("astream")
    async def astream(
        self,
        messages: Union[CrewAIMessageInput, CrewAIMessageSequence],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Async streaming completion for CrewAI workflows.

        Parameters
        ----------
        messages : Union[CrewAIMessageInput, CrewAIMessageSequence]
            CrewAI message input (string, dict, message object, or sequence)
        **kwargs : Any
            - model, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, stop
            - agent_role, agent_goal, task_description, crew_id, crew_name, process_id
            - request_id, tenant, framework_version

        Yields
        ------
        str
            Streaming text tokens

        Raises
        ------
        Exception
            Any exception from the underlying LLM with rich context attached
        """
        kwargs = self._apply_instance_defaults(kwargs)
        corpus_messages = _normalize_messages(messages)
        ctx = _build_operation_context_from_kwargs(kwargs)
        params = _build_sampling_params(
            default_model=self.model,
            default_temperature=self.temperature,
            default_max_tokens=self.max_tokens,
            kwargs=kwargs,
        )

        async for token in self._execute_streaming_async(
            messages=corpus_messages,
            ctx=ctx,
            params=params,
        ):
            yield token

    @with_llm_error_context("complete")
    def complete(
        self,
        messages: Union[CrewAIMessageInput, CrewAIMessageSequence],
        **kwargs: Any,
    ) -> str:
        """
        Sync completion for CrewAI workflows.

        Parameters
        ----------
        messages : Union[CrewAIMessageInput, CrewAIMessageSequence]
            CrewAI message input (string, dict, message object, or sequence)
        **kwargs : Any
            - model, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, stop
            - agent_role, agent_goal, task_description, crew_id, crew_name, process_id
            - request_id, tenant, framework_version

        Returns
        -------
        str
            Completion text result

        Raises
        ------
        Exception
            Any exception from the underlying LLM with rich context attached
        """
        kwargs = self._apply_instance_defaults(kwargs)
        corpus_messages = _normalize_messages(messages)
        ctx = _build_operation_context_from_kwargs(kwargs)
        params = _build_sampling_params(
            default_model=self.model,
            default_temperature=self.temperature,
            default_max_tokens=self.max_tokens,
            kwargs=kwargs,
        )

        raw_result = self._execute_completion_sync(
            messages=corpus_messages,
            ctx=ctx,
            params=params,
        )

        return _extract_completion_text(raw_result)

    # Make the instance directly callable for convenience
    __call__ = complete

    @with_llm_error_context("stream")
    def stream(
        self,
        messages: Union[CrewAIMessageInput, CrewAIMessageSequence],
        **kwargs: Any,
    ) -> Iterator[str]:
        """
        Sync streaming completion for CrewAI workflows.

        Parameters
        ----------
        messages : Union[CrewAIMessageInput, CrewAIMessageSequence]
            CrewAI message input (string, dict, message object, or sequence)
        **kwargs : Any
            - model, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, stop
            - agent_role, agent_goal, task_description, crew_id, crew_name, process_id
            - request_id, tenant, framework_version

        Yields
        ------
        str
            Streaming text tokens

        Raises
        ------
        Exception
            Any exception from the underlying LLM with rich context attached
        """
        kwargs = self._apply_instance_defaults(kwargs)
        corpus_messages = _normalize_messages(messages)
        ctx = _build_operation_context_from_kwargs(kwargs)
        params = _build_sampling_params(
            default_model=self.model,
            default_temperature=self.temperature,
            default_max_tokens=self.max_tokens,
            kwargs=kwargs,
        )

        for token in self._execute_streaming_sync(
            messages=corpus_messages,
            ctx=ctx,
            params=params,
        ):
            yield token


__all__ = [
    "CorpusCrewAILLM",
    "CrewAIMessage",
    "CrewAIContext",
    "CrewAIMessageInput",
    "CrewAIMessageSequence",
    "with_llm_error_context",
    "with_async_llm_error_context",
]