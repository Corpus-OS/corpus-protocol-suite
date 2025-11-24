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

from corpus_sdk.core.context_translation import (
    from_autogen as core_ctx_from_autogen,
)
from corpus_sdk.core.error_context import attach_context
from corpus_sdk.llm.framework_adapters.common.framework_utils import (
    CoercionErrorCodes,
    coerce_token_usage,
)
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

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Error codes / coercion config
# ---------------------------------------------------------------------------


class ErrorCodes:
    """
    Symbolic error codes for the AutoGen LLM adapter.

    These are primarily for observability / correlation and are not tied to a
    specific exception type.
    """

    BAD_OPERATION_CONTEXT = "BAD_OPERATION_CONTEXT"
    BAD_INIT_CONFIG = "BAD_INIT_CONFIG"
    BAD_COMPLETION_RESULT = "BAD_COMPLETION_RESULT"
    BAD_STREAM_CHUNK = "BAD_STREAM_CHUNK"
    BAD_USAGE_RESULT = "BAD_USAGE_RESULT"


_USAGE_ERROR_CODES = CoercionErrorCodes(
    invalid_result=ErrorCodes.BAD_USAGE_RESULT,
    empty_result=ErrorCodes.BAD_USAGE_RESULT,
    conversion_error=ErrorCodes.BAD_USAGE_RESULT,
    framework_label="autogen",
)


# ---------------------------------------------------------------------------
# Public protocol
# ---------------------------------------------------------------------------


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
    """

    model: str = "default"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    framework_version: Optional[str] = None
    enable_metrics: bool = True
    validate_inputs: bool = True
    timeout: Optional[float] = None  # Reserved
    max_retries: int = 0
    request_timeout: float = 60.0

    def __post_init__(self) -> None:
        if not (0.0 <= float(self.temperature) <= 2.0):
            raise ValueError(
                f"{ErrorCodes.BAD_INIT_CONFIG}: "
                f"temperature must be between 0.0 and 2.0, got {self.temperature}"
            )
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError(
                f"{ErrorCodes.BAD_INIT_CONFIG}: "
                f"max_tokens must be positive, got {self.max_tokens}"
            )
        if self.request_timeout is None or self.request_timeout <= 0:
            raise ValueError(
                f"{ErrorCodes.BAD_INIT_CONFIG}: "
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
# Error-context decorators (lazy, low-cost extraction)
# ---------------------------------------------------------------------------


def _extract_basic_context(
    instance: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    operation: str,
) -> Dict[str, Any]:
    """
    Extract fast, low-cost context for error enrichment.
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

    This is invoked only when an exception occurs to stay off the hot path.
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

        # AutoGen-specific knobs
        if kwargs.get("conversation") is not None:
            detailed["has_conversation"] = True
        if kwargs.get("context") is not None:
            detailed["has_extra_context"] = True

        # Sampling parameters
        for param in (
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
        ):
            if param in kwargs:
                detailed[param] = kwargs[param]

    except Exception as ctx_error:  # noqa: BLE001
        logger.debug("Failed to compute detailed error context: %s", ctx_error)

    return detailed


def _create_error_context_decorator(
    operation: str,
    is_async: bool = False,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Factory for creating error-context decorators with lazy context extraction.
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


# ---------------------------------------------------------------------------
# Concrete AutoGen client
# ---------------------------------------------------------------------------


class CorpusAutoGenChatClient:
    """
    OpenAI-style chat client backed by a Corpus `LLMProtocolV1` via `LLMTranslator`.

    This class is intended to be dropped into AutoGen configs wherever an
    OpenAI-compatible chat client is expected. It exposes `create` and
    `acreate` methods with familiar semantics:

        - `create(messages=[...], model="...", stream=False)`
        - `acreate(messages=[...], model="...", stream=True)`

    All protocol calls are delegated to `LLMTranslator`, keeping this layer
    thin and focused on:
    - AutoGen message / parameter handling
    - Context construction
    - OpenAI-style response shaping
    - Error-context enrichment
    """

    # ------------------------------------------------------------------ #
    # Construction / resource management
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

        # Keep a direct reference to the underlying adapter for lifecycle.
        self._adapter: LLMProtocolV1 = llm_adapter

        # Validate adapter + key config invariants.
        self._validate_init_params(llm_adapter)

        # Create or adopt LLMTranslator (all protocol logic lives there).
        if translator is not None:
            self._translator: LLMTranslator = translator
        else:
            self._translator = create_llm_translator(
                adapter=llm_adapter,
                framework=framework,
                translator=None,
                post_processing_config=post_processing_config,
            )

        # Convenience attributes preserved for compatibility / logging.
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

    # Context manager support to mirror graph / embedding adapters.
    def __enter__(self) -> "CorpusAutoGenChatClient":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if hasattr(self._adapter, "close"):
            try:
                self._adapter.close()  # type: ignore[call-arg]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error while closing LLM adapter in __exit__: %s", exc)

    async def __aenter__(self) -> "CorpusAutoGenChatClient":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if hasattr(self._adapter, "aclose"):
            try:
                await self._adapter.aclose()  # type: ignore[call-arg]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error while closing LLM adapter in __aexit__: %s", exc)

    # ------------------------------------------------------------------ #
    # Internal helpers: validation & context
    # ------------------------------------------------------------------ #

    def _validate_init_params(self, llm_adapter: LLMProtocolV1) -> None:
        """
        Validate initialization parameters and adapter capabilities.

        - Ensures the adapter exposes core `LLMProtocolV1` methods.
        - Validates temperature and max_tokens constraints from config.
        """
        required_methods = (
            "complete",
            "stream",
            "count_tokens",
            "health",
            "capabilities",
        )
        missing = [m for m in required_methods if not callable(getattr(llm_adapter, m, None))]
        if missing:
            raise TypeError(
                f"{ErrorCodes.BAD_INIT_CONFIG}: "
                f"llm_adapter must implement LLMProtocolV1; missing methods: "
                f"{', '.join(missing)}"
            )

        temperature = self._config.temperature
        if not isinstance(temperature, (int, float)) or not (0.0 <= float(temperature) <= 2.0):
            raise ValueError(
                f"{ErrorCodes.BAD_INIT_CONFIG}: temperature must be between 0 and 2"
            )

        if self._config.max_tokens is not None:
            if not isinstance(self._config.max_tokens, int) or self._config.max_tokens < 1:
                raise ValueError(
                    f"{ErrorCodes.BAD_INIT_CONFIG}: max_tokens must be positive"
                )

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

    # ---- context & params helpers ------------------------------------- #

    def _build_core_context(
        self,
        *,
        conversation: Optional[Any],
        extra_context: Optional[Mapping[str, Any]],
        request_id: Optional[str],
        tenant: Optional[str],
    ) -> Optional[OperationContext]:
        """
        Construct an OperationContext from AutoGen-style inputs.

        Context translation is best-effort: failures are logged and
        annotated via attach_context, but we fall back to None so
        protocol calls can still succeed without an OperationContext.
        """
        extra: Dict[str, Any] = dict(extra_context or {})

        if conversation is None and not extra:
            return None

        try:
            ctx = core_ctx_from_autogen(
                conversation,
                framework_version=self._framework_version,
                **extra,
            )
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="autogen",
                operation="context_translation",
            )
            logger.warning(
                "Failed to create OperationContext from AutoGen conversation/context: %s. "
                "Proceeding without OperationContext.",
                exc,
            )
            return None

        if not isinstance(ctx, OperationContext):
            logger.warning(
                "%s: from_autogen produced unsupported context type: %s; "
                "ignoring OperationContext.",
                ErrorCodes.BAD_OPERATION_CONTEXT,
                type(ctx).__name__,
            )
            return None

        # Optional overrides for request_id / tenant.
        if request_id is not None or tenant is not None:
            ctx = OperationContext(
                request_id=request_id or ctx.request_id,
                idempotency_key=ctx.idempotency_key,
                deadline_ms=ctx.deadline_ms,
                traceparent=ctx.traceparent,
                tenant=tenant or ctx.tenant,
                attrs=ctx.attrs,
            )

        return ctx

    def _build_sampling_params(
        self,
        *,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Extract sampling / routing parameters from kwargs.

        Recognized keys are removed from kwargs; unknown keys are ignored.
        """
        # Stop sequences: OpenAI uses "stop" (str or list[str]).
        stop_arg = kwargs.pop("stop", None)
        stop_sequences: Optional[list[str]] = None
        if isinstance(stop_arg, str):
            stop_sequences = [stop_arg]
        elif isinstance(stop_arg, (list, tuple)):
            stop_sequences = [str(s) for s in stop_arg]

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
        return {k: v for k, v in params.items() if v is not None}

    def _build_framework_ctx(
        self,
        *,
        operation: str,
        stream: bool,
        model: Optional[str],
        request_id: Optional[str],
        tenant: Optional[str],
    ) -> Dict[str, Any]:
        """
        Build a framework_ctx mapping for LLMTranslator calls.

        Carries observability hints and AutoGen-specific routing fields,
        separate from the protocol-level OperationContext.
        """
        ctx: Dict[str, Any] = {
            "framework": "autogen",
            "operation": operation,
            "stream": stream,
        }
        if self._framework_version is not None:
            ctx["framework_version"] = self._framework_version
        if model is not None:
            ctx["model"] = model
        if request_id is not None:
            ctx["request_id"] = request_id
        if tenant is not None:
            ctx["tenant"] = tenant
        return ctx

    def _build_ctx_and_params(
        self,
        *,
        conversation: Optional[Any],
        extra_context: Optional[Mapping[str, Any]],
        stream: bool,
        operation: str,
        **kwargs: Any,
    ) -> Tuple[Optional[OperationContext], Dict[str, Any], Dict[str, Any]]:
        """
        High-level helper to build:
        - OperationContext
        - sampling params
        - framework_ctx for LLMTranslator

        This keeps the public create/acreate methods thin and symmetric.
        """
        # Extract request-scoped identifiers early so both context and
        # framework_ctx can see them.
        request_id = kwargs.pop("request_id", None)
        tenant = kwargs.pop("tenant", None)

        ctx = self._build_core_context(
            conversation=conversation,
            extra_context=extra_context,
            request_id=request_id,
            tenant=tenant,
        )

        params = self._build_sampling_params(kwargs=kwargs)
        model_for_ctx = str(params.get("model")) if "model" in params else None

        framework_ctx = self._build_framework_ctx(
            operation=operation,
            stream=stream,
            model=model_for_ctx,
            request_id=request_id,
            tenant=tenant,
        )

        return ctx, params, framework_ctx

    # ------------------------------------------------------------------ #
    # OpenAI-style shaping helpers (thin, usage via framework_utils)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _completion_to_openai(
        result: Any,
        *,
        completion_id: Optional[str] = None,
        created: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Convert a translator-level completion into an OpenAI ChatCompletion payload.

        Translator default `from_completion` returns a dict like:

            {
                "text": ...,
                "model": ...,
                "usage": {...},
                "finish_reason": ...,
                ...
            }

        This function stays thin: it does not touch protocol logic, only
        shapes the already-normalized result.
        """
        completion_id = completion_id or _new_id()
        created = created or _now_epoch_s()

        # Text / model / finish_reason best-effort extraction.
        if isinstance(result, Mapping):
            text = str(
                result.get("text")
                or result.get("content")
                or (result.get("message") or {}).get("content", "")
                or ""
            )
            model = str(result.get("model") or "unknown")
            finish_reason = result.get("finish_reason")
        else:
            text = str(getattr(result, "text", "") or "")
            model = str(getattr(result, "model", "unknown"))
            finish_reason = getattr(result, "finish_reason", None)

        # Usage: prefer shared coercion utility for consistency / bounds.
        usage_dict: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        try:
            token_usage = coerce_token_usage(
                result,
                framework="autogen",
                error_codes=_USAGE_ERROR_CODES,
                logger=logger,
            )
            usage_dict = {
                "prompt_tokens": token_usage.prompt_tokens,
                "completion_tokens": token_usage.completion_tokens,
                "total_tokens": token_usage.total_tokens,
            }
        except Exception as exc:  # noqa: BLE001
            # Fall back to simple mapping-based extraction if available.
            logger.debug(
                "AutoGen: failed to coerce token usage from completion: %s",
                exc,
            )
            if isinstance(result, Mapping):
                usage = result.get("usage") or {}
                if isinstance(usage, Mapping):
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                        value = usage.get(key)
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

        Translator `from_chunk` yields dicts of the form:

            {
                "text": ...,
                "is_final": bool,
                "model": ...,
                "usage_so_far": {...} | None,
                ...
            }
        """
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
            # Best-effort reuse of shared coercion, but keep it soft-failing.
            try:
                token_usage = coerce_token_usage(
                    {"usage": usage},
                    framework="autogen",
                    error_codes=_USAGE_ERROR_CODES,
                    logger=logger,
                )
                usage_payload = {
                    "prompt_tokens": token_usage.prompt_tokens,
                    "completion_tokens": token_usage.completion_tokens,
                    "total_tokens": token_usage.total_tokens,
                }
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "AutoGen: failed to coerce token usage from stream chunk: %s",
                    exc,
                )
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
                await asyncio.sleep(min(2.0**attempts, 5.0))

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
        """
        if self._validate_inputs_flag:
            self._validate_messages(messages)

        stream = bool(kwargs.pop("stream", False))

        # Extract AutoGen / extra context (kept thin, passed through to context helpers).
        conversation = kwargs.pop("conversation", None)
        extra_context = kwargs.pop("context", None)
        if extra_context is not None and not isinstance(extra_context, Mapping):
            extra_context = None

        ctx, params, framework_ctx = self._build_ctx_and_params(
            conversation=conversation,
            extra_context=extra_context,
            stream=stream,
            operation="acreate_stream" if stream else "acreate",
            **kwargs,
        )

        if not stream:
            async def _invoke() -> Any:
                return await self._translator.arun_complete(
                    raw_messages=messages,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
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
                    framework_ctx=framework_ctx,
                    **params,
                )

                async for chunk in chunk_iter:
                    yield self._chunk_to_openai(
                        chunk,
                        stream_id=stream_id,
                        created=created,
                        model_fallback=str(params.get("model", self.model)),
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
        """
        if self._validate_inputs_flag:
            self._validate_messages(messages)

        stream = bool(kwargs.pop("stream", False))

        # Extract AutoGen / extra context.
        conversation = kwargs.pop("conversation", None)
        extra_context = kwargs.pop("context", None)
        if extra_context is not None and not isinstance(extra_context, Mapping):
            extra_context = None

        ctx, params, framework_ctx = self._build_ctx_and_params(
            conversation=conversation,
            extra_context=extra_context,
            stream=stream,
            operation="create_stream" if stream else "create",
            **kwargs,
        )

        if not stream:
            result = self._translator.complete(
                raw_messages=messages,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
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
                    framework_ctx=framework_ctx,
                    **params,
                )

                for chunk in chunk_iter:
                    yield self._chunk_to_openai(
                        chunk,
                        stream_id=stream_id,
                        created=created,
                        model_fallback=str(params.get("model", self.model)),
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
    "ErrorCodes",
    "with_llm_error_context",
    "with_async_llm_error_context",
]
