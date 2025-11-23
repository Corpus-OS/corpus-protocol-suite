# corpus_sdk/llm/framework_adapters/autogen.py
# SPDX-License-Identifier: Apache-2.0

"""
AutoGen adapter for Corpus LLM protocol.

This module exposes a Corpus `LLMProtocolV1` implementation (via the
framework-agnostic `LLMTranslator`) as an OpenAI-style chat client
suitable for use with AutoGen's configuration system.

Key responsibilities
--------------------
- Accept OpenAI-style message dicts from AutoGen
- Convert them to Corpus wire messages via the LLM translation layer
- Build OperationContext from AutoGen conversation / metadata
- Construct sampling / routing parameters (model, temperature, stop, etc.)
- Delegate sync/async + streaming orchestration to `LLMTranslator`
- Convert translator-level completions/streams into OpenAI-style
  ChatCompletion / ChatCompletionChunk payloads
- Enrich exceptions with AutoGen-specific debug metadata via `attach_context`

Design principles
-----------------
- Protocol-first:
    The Corpus `LLMProtocolV1` is the source of truth; this module is a thin
    compatibility layer for AutoGen on top of the shared LLM translation layer.

- LLMTranslator orchestration:
    All message normalization and protocol calls go through `LLMTranslator`,
    keeping framework-specific concerns localized and consistent.

- Non-invasive:
    No retries, circuit breaking, or deadlines are implemented beyond what
    this client explicitly configures; the underlying adapter can still apply
    its own policies.

- Rich error context:
    Exceptions are annotated via `attach_context` with framework-specific and
    per-call metadata, without mutating their messages or types.

Config & behavior overrides
---------------------------
AutoGen callers can customize behavior through `AutoGenClientConfig`:

- model: default model identifier
- temperature: default temperature (validated 0–2)
- max_tokens: default max_tokens (must be positive if set)
- framework_version: optional framework version tag
- enable_metrics: reserved for future in-client metrics (currently unused)
- validate_inputs: enable strict message validation before translation
- timeout: reserved; prefer request_timeout for async calls
- request_timeout: per-call timeout for async completions
- max_retries: simple async retry count for transient failures

These can be passed either as an `AutoGenClientConfig` instance or as
keyword arguments to `CorpusAutoGenChatClient(...)`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from functools import wraps
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
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
)
from uuid import uuid4

from corpus_sdk.core.context_translation import ContextTranslator
from corpus_sdk.core.error_context import attach_context
from corpus_sdk.llm.framework_adapters.common.llm_translation import (
    LLMTranslator,
    LLMPostProcessingConfig,
    create_llm_translator,
)
from corpus_sdk.llm.llm_base import (
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


@dataclass
class AutoGenClientConfig:
    """
    Configuration for AutoGen client behavior.

    This configuration centralizes all tunable parameters for the
    `CorpusAutoGenChatClient`. It can be constructed directly or via
    `from_dict` for convenient integration with config files.

    Fields
    ------
    model:
        Default model identifier used when AutoGen does not provide one.

    temperature:
        Default sampling temperature. Must be in the range [0, 2].

    max_tokens:
        Default max_tokens for completions (if provided, must be positive).

    framework_version:
        Optional framework version string propagated into context metadata.

    enable_metrics:
        Reserved for in-client metrics; currently not used directly.

    validate_inputs:
        If True, incoming message lists are strictly validated for
        structure (`role` and `content` fields) before translation.

    timeout:
        Reserved for future use; prefer `request_timeout` for async calls.

    max_retries:
        Number of retry attempts for async non-streaming calls on failures.

    request_timeout:
        Per-call timeout (in seconds) for async non-streaming calls.
    """

    model: str = "default"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    framework_version: Optional[str] = None
    enable_metrics: bool = True
    validate_inputs: bool = True
    timeout: Optional[float] = None
    max_retries: int = 0
    request_timeout: float = 60.0

    def __post_init__(self) -> None:
        """
        Auto-validate configuration when the dataclass is instantiated directly.

        This ensures that even when callers construct the config themselves,
        the key invariants are enforced consistently.
        """
        if not (0.0 <= float(self.temperature) <= 2.0):
            raise ValueError(
                f"temperature must be between 0.0 and 2.0, got {self.temperature}"
            )
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        if self.request_timeout is None or self.request_timeout <= 0:
            raise ValueError(
                f"request_timeout must be positive, got {self.request_timeout}"
            )

    @classmethod
    def from_dict(cls, config_dict: Mapping[str, Any]) -> "AutoGenClientConfig":
        """
        Create a configuration from a mapping, ignoring unknown keys.
        """
        filtered: Dict[str, Any] = {
            key: value for key, value in config_dict.items() if key in cls.__annotations__
        }
        return cls(**filtered)


def _now_epoch_s() -> int:
    return int(time.time())


def _new_id(prefix: str = "chatcmpl") -> str:
    return f"{prefix}-{uuid4().hex}"


# ---------------------------------------------------------------------------
# Error Context Decorators (with lazy, low-cost context extraction)
# ---------------------------------------------------------------------------


def _extract_basic_context(
    instance: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    operation: str,
) -> Dict[str, Any]:
    """
    Extract fast, low-cost context for error enrichment.

    Only captures cheap fields that are safe to compute on every call:
    - model name
    - operation
    - messages_count (if first arg is a sequence)
    - stream flag, request_id, tenant (when present)
    """
    dynamic_ctx: Dict[str, Any] = {
        "model": getattr(instance, "model", "unknown"),
        "operation": operation,
    }

    # Messages count (cheap length check)
    try:
        if args and isinstance(args[0], (list, tuple)):
            dynamic_ctx["messages_count"] = len(args[0])
    except Exception:
        # Best-effort only; never fail here
        pass

    if "stream" in kwargs:
        dynamic_ctx["stream"] = bool(kwargs["stream"])
    if "request_id" in kwargs:
        dynamic_ctx["request_id"] = kwargs["request_id"]
    if "tenant" in kwargs:
        dynamic_ctx["tenant"] = kwargs["tenant"]

    return dynamic_ctx


def _compute_detailed_context(
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute richer, more expensive context details for error enrichment.

    This is called only when an exception occurs to avoid hot-path overhead.
    """
    detailed: Dict[str, Any] = {}
    try:
        # Message-based metrics
        if args and isinstance(args[0], (list, tuple)):
            messages = args[0]
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

            detailed["roles_distribution"] = roles
            detailed["total_content_chars"] = total_chars

        # AutoGen-specific context presence flags
        if kwargs.get("conversation") is not None:
            detailed["has_conversation"] = True
        if kwargs.get("context") is not None:
            detailed["has_extra_context"] = True

        # Sampling parameters (if provided)
        sampling_params = [
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
        ]
        for param in sampling_params:
            if param in kwargs:
                detailed[param] = kwargs[param]

    except Exception as ctx_error:  # noqa: BLE001
        # Never let context computation impact control flow
        logger.debug("Failed to compute detailed error context: %s", ctx_error)

    return detailed


def _create_error_context_decorator(
    operation: str,
    is_async: bool = False,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Factory for creating error context decorators with lazy context extraction.
    """

    def decorator_factory(
        **static_context: Any,
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            if is_async:

                @wraps(func)
                async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
                    basic_context = _extract_basic_context(self, args, kwargs, operation)
                    base_context = {**static_context, **basic_context}

                    try:
                        return await func(self, *args, **kwargs)
                    except Exception as exc:  # noqa: BLE001
                        detailed = _compute_detailed_context(args, kwargs)
                        full_context = {**base_context, **detailed}
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
                    basic_context = _extract_basic_context(self, args, kwargs, operation)
                    base_context = {**static_context, **basic_context}

                    try:
                        return func(self, *args, **kwargs)
                    except Exception as exc:  # noqa: BLE001
                        detailed = _compute_detailed_context(args, kwargs)
                        full_context = {**base_context, **detailed}
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


# Convenience decorators with rich context extraction
def with_llm_error_context(
    operation: str,
    **static_context: Any,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for sync LLM methods with lazy dynamic context extraction."""
    return _create_error_context_decorator(operation, is_async=False)(**static_context)


def with_async_llm_error_context(
    operation: str,
    **static_context: Any,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for async LLM methods with lazy dynamic context extraction."""
    return _create_error_context_decorator(operation, is_async=True)(**static_context)


class CorpusAutoGenChatClient:
    """
    OpenAI-style chat client backed by a Corpus `LLMProtocolV1` via `LLMTranslator`.

    This class is intended to be dropped into AutoGen configs wherever an
    OpenAI-compatible chat client is expected. It exposes `create` and
    `acreate` methods with familiar semantics:

        - `create(messages=[...], model="...", stream=False)`
        - `acreate(messages=[...], model="...", stream=True)`

    Messages are expected to be OpenAI-style dicts:

        {"role": "user", "content": "Hello"}

    They are handed directly to the shared LLM translation layer, which
    normalizes them into Corpus protocol messages before calling the
    underlying `LLMProtocolV1` adapter.

    Example
    -------
    ```python
    from corpus_sdk.llm.framework_adapters.autogen import CorpusAutoGenChatClient
    import autogen

    # Initialize with any Corpus LLMProtocolV1 adapter
    llm_client = CorpusAutoGenChatClient(
        llm_adapter=my_adapter,
        model="gpt-4",
        temperature=0.7,
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

    Error Handling Example
    ----------------------
    ```python
    try:
        response = llm_client.create(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4",
            temperature=0.7,
            conversation=agent_conversation,
        )
    except Exception as e:
        # Rich error context automatically attached with message counts,
        # roles, and configuration details.
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
        config: Optional[AutoGenClientConfig] = None,
        framework: str = "autogen",
        translator: Optional[LLMTranslator] = None,
        post_processing_config: Optional[LLMPostProcessingConfig] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the AutoGen chat client.

        Parameters
        ----------
        llm_adapter:
            An implementation of `LLMProtocolV1`, used by the shared
            `LLMTranslator` orchestrator.

        config:
            Optional `AutoGenClientConfig` instance. If omitted, a config is
            built from remaining keyword arguments.

        framework:
            Framework name forwarded to `create_llm_translator` (defaults
            to "autogen").

        translator:
            Optional pre-constructed `LLMTranslator`. If provided, it is
            used as-is instead of calling `create_llm_translator`.

        post_processing_config:
            Optional `LLMPostProcessingConfig` passed to `create_llm_translator`
            when a translator is not supplied.

        **kwargs:
            Additional configuration keys used to construct
            `AutoGenClientConfig` when `config` is None.
        """
        self._config: AutoGenClientConfig = config or AutoGenClientConfig.from_dict(kwargs)

        # Validate adapter and key configuration values
        self._validate_init_params(llm_adapter)

        # Create or adopt LLMTranslator
        if translator is not None:
            self._translator: LLMTranslator = translator
        else:
            self._translator = create_llm_translator(
                adapter=llm_adapter,
                framework=framework,
                translator=None,
                post_processing_config=post_processing_config,
            )

        # Convenience attributes preserved from prior interface
        self.model: str = self._config.model
        self.temperature: float = float(self._config.temperature)
        self.max_tokens: Optional[int] = self._config.max_tokens
        self._framework_version: Optional[str] = self._config.framework_version
        self._validate_inputs_flag: bool = self._config.validate_inputs
        self._enable_metrics_flag: bool = self._config.enable_metrics
        self._request_timeout: float = self._config.request_timeout
        self._max_retries: int = self._config.max_retries

        logger.info(
            "CorpusAutoGenChatClient initialized with model=%s, temperature=%.2f",
            self.model,
            self.temperature,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers: validation & context
    # ------------------------------------------------------------------ #

    def _validate_init_params(self, llm_adapter: LLMProtocolV1) -> None:
        """
        Validate initialization parameters and adapter capabilities.

        - Ensures the adapter exposes core `LLMProtocolV1` methods.
        - Validates temperature and max_tokens constraints from config.
        """
        # Duck-typed adapter validation instead of strict isinstance on Protocol
        required_methods = ("complete", "stream", "count_tokens", "health", "capabilities")
        missing = [m for m in required_methods if not callable(getattr(llm_adapter, m, None))]
        if missing:
            raise TypeError(
                f"llm_adapter must implement LLMProtocolV1; missing methods: {', '.join(missing)}"
            )

        temperature = self._config.temperature
        if not isinstance(temperature, (int, float)) or not (0.0 <= float(temperature) <= 2.0):
            raise ValueError("temperature must be between 0 and 2")

        if self._config.max_tokens is not None:
            if not isinstance(self._config.max_tokens, int) or self._config.max_tokens < 1:
                raise ValueError("max_tokens must be positive")

    def _validate_messages(self, messages: Sequence[Mapping[str, Any]]) -> None:
        """
        Validate message structure before processing.

        Ensures:
        - messages is non-empty
        - each item is a Mapping
        - each message has 'role' and 'content' keys
        """
        if not messages:
            raise ValueError("messages list cannot be empty")

        for index, msg in enumerate(messages):
            if not isinstance(msg, Mapping):
                raise TypeError(
                    f"messages[{index}] must be a mapping, got {type(msg).__name__}"
                )
            if "role" not in msg:
                raise ValueError(f"messages[{index}] is missing required 'role' field")
            if "content" not in msg:
                raise ValueError(f"messages[{index}] is missing required 'content' field")

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

        # Drop None values so the translator sees a clean param set.
        params = {k: v for k, v in params.items() if v is not None}
        return ctx, params

    @staticmethod
    def _completion_to_openai(
        result: Any,
        *,
        completion_id: Optional[str] = None,
        created: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Convert a translator-level completion into an OpenAI ChatCompletion payload.

        The LLMTranslator's default `from_completion` returns a dict:

            {
                "text": ...,
                "model": ...,
                "model_family": ...,
                "usage": {
                    "prompt_tokens": ...,
                    "completion_tokens": ...,
                    "total_tokens": ...,
                },
                "finish_reason": ...,
                "tool_calls": [...],
            }

        But we also support attribute-style objects for flexibility.
        """
        completion_id = completion_id or _new_id()
        created = created or _now_epoch_s()

        text: str
        model: str
        finish_reason: Optional[str]
        usage_dict: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        if isinstance(result, Mapping):
            text = str(
                result.get("text")
                or result.get("content")
                or (result.get("message") or {}).get("content", "")
                or ""
            )
            model = str(result.get("model") or "unknown")
            finish_reason = result.get("finish_reason")

            usage = result.get("usage") or {}
            if isinstance(usage, Mapping):
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    value = usage.get(key)
                    if isinstance(value, int):
                        usage_dict[key] = value
        else:
            # Attribute-based object
            text = str(getattr(result, "text", "") or "")
            model = str(getattr(result, "model", "unknown"))
            finish_reason = getattr(result, "finish_reason", None)
            usage = getattr(result, "usage", None)
            if usage is not None:
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    value = getattr(usage, key, None)
                    if isinstance(value, int):
                        usage_dict[key] = value

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text,
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage_dict,
        }

    @staticmethod
    def _chunk_to_openai(
        chunk: Any,
        *,
        stream_id: str,
        created: int,
        model_fallback: str,
        is_first: bool,
    ) -> Dict[str, Any]:
        """
        Convert a translator-level streaming chunk into an OpenAI ChatCompletion chunk.

        The default LLMTranslator `from_chunk` yields dicts of the form:

            {
                "text": ...,
                "is_final": bool,
                "model": ...,
                "usage_so_far": {
                    "prompt_tokens": ...,
                    "completion_tokens": ...,
                    "total_tokens": ...,
                } | None,
                "tool_calls": [...],
            }

        Attribute-style chunk objects are also supported.
        """
        # Extract basic fields from dict or object
        if isinstance(chunk, Mapping):
            text = str(chunk.get("text") or chunk.get("delta") or "")
            is_final = bool(chunk.get("is_final", False))
            model = str(chunk.get("model") or model_fallback)
            usage = chunk.get("usage_so_far") or chunk.get("usage")
        else:
            text = str(getattr(chunk, "text", "") or getattr(chunk, "delta", "") or "")
            is_final = bool(getattr(chunk, "is_final", False))
            model = str(getattr(chunk, "model", model_fallback))
            usage = getattr(chunk, "usage_so_far", None)

        delta: Dict[str, Any] = {}

        # OpenAI typically sends role only on the first delta.
        if is_first:
            delta["role"] = "assistant"

        if text:
            delta["content"] = text

        choice: Dict[str, Any] = {
            "index": 0,
            "delta": delta,
            "finish_reason": "stop" if is_final else None,
        }

        payload: Dict[str, Any] = {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [choice],
        }

        if isinstance(usage, Mapping):
            usage_payload: Dict[str, int] = {}
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = usage.get(key)
                if isinstance(value, int):
                    usage_payload[key] = value
            if usage_payload:
                payload["usage"] = usage_payload

        return payload

    async def _run_with_retries_async(
        self,
        func: Callable[[], Awaitable[Any]],
    ) -> Any:
        """
        Execute an async operation with simple retry and timeout semantics.

        - Uses `self._config.request_timeout` for per-attempt timeout.
        - Retries up to `self._config.max_retries` times.
        """
        attempts = 0
        last_error: Optional[BaseException] = None

        while attempts <= self._max_retries:
            try:
                if self._request_timeout and self._request_timeout > 0:
                    return await asyncio.wait_for(func(), timeout=self._request_timeout)
                return await func()
            except asyncio.CancelledError:
                # Cancellation should always propagate immediately.
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                attempts += 1
                if attempts > self._max_retries:
                    raise
                logger.warning(
                    "AutoGen async LLM call failed (attempt %d/%d): %s",
                    attempts,
                    self._max_retries,
                    exc,
                )
                # Simple backoff capped at 5s
                await asyncio.sleep(min(2.0 ** attempts, 5.0))

        # Should not be reachable, but keeps type checkers satisfied.
        if last_error is not None:
            raise last_error

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
        if self._validate_inputs_flag:
            self._validate_messages(messages)

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
            async def _invoke() -> Any:
                return await self._translator.arun_complete(
                    raw_messages=messages,
                    op_ctx=ctx,
                    framework_ctx=None,
                    **params,
                )

            result = await self._run_with_retries_async(_invoke)
            return self._completion_to_openai(result)

        # Streaming: return async iterator of OpenAI-style chunks.
        async def _gen() -> AsyncIterator[Dict[str, Any]]:
            stream_id = _new_id()
            created = _now_epoch_s()
            is_first = True
            chunk_iter: Optional[AsyncIterator[Any]] = None

            try:
                chunk_iter = await self._translator.arun_stream(
                    raw_messages=messages,
                    op_ctx=ctx,
                    framework_ctx=None,
                    **params,
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
            except Exception:  # noqa: BLE001
                logger.error("Streaming iteration failed in acreate", exc_info=True)
                raise
            finally:
                if chunk_iter is not None:
                    aclose = getattr(chunk_iter, "aclose", None)
                    if callable(aclose):
                        try:
                            await aclose()
                        except Exception as cleanup_error:  # noqa: BLE001
                            logger.warning(
                                "Stream cleanup failed in acreate: %s",
                                cleanup_error,
                            )

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
        if self._validate_inputs_flag:
            self._validate_messages(messages)

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
            result = self._translator.complete(
                raw_messages=messages,
                op_ctx=ctx,
                framework_ctx=None,
                **params,
            )
            return self._completion_to_openai(result)

        # Streaming: sync path via translator streaming with proper cleanup.
        def _iter() -> Iterator[Dict[str, Any]]:
            stream_id = _new_id()
            created = _now_epoch_s()
            is_first = True
            chunk_iter: Optional[Iterator[Any]] = None

            try:
                chunk_iter = self._translator.stream(
                    raw_messages=messages,
                    op_ctx=ctx,
                    framework_ctx=None,
                    **params,
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
            except Exception:  # noqa: BLE001
                logger.error("Streaming iteration failed in create", exc_info=True)
                raise
            finally:
                if chunk_iter is not None:
                    close = getattr(chunk_iter, "close", None)
                    if callable(close):
                        try:
                            close()
                        except Exception as cleanup_error:  # noqa: BLE001
                            logger.warning(
                                "Stream cleanup failed in create: %s",
                                cleanup_error,
                            )

        return _iter()

    # Allow direct call usage: client(...) behaves like client.create(...)
    def __call__(
        self,
        messages: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        return self.create(messages, **kwargs)


__all__ = [
    "AutoGenClientConfig",
    "AutoGenLLMClientProtocol",
    "CorpusAutoGenChatClient",
    "with_llm_error_context",
    "with_async_llm_error_context",
]
