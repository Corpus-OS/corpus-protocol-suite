# corpus_sdk/llm/framework_adapters/langchain.py
# SPDX-License-Identifier: Apache-2.0

"""
LangChain adapter for Corpus LLM protocol.

This module exposes Corpus `LLMProtocolV1` implementations via the shared
`LLMTranslator` layer as `langchain_core` chat models, with:

- Async + sync generation
- Async + sync streaming (true incremental streaming)
- Proper callback integration (on_llm_end, on_llm_new_token, on_llm_error)
- Protocol-first, translator-based design (no direct message translation)
- Production-grade error handling and observability
- Centralized token counting via LLMTranslator with robust fallbacks

Design goals
------------

1. Protocol + translator first:
   All calls go through the shared `LLMTranslator` with the `"langchain"`
   framework, so message normalization, post-processing, tools, and error context
   are consistent across frameworks.

2. Optional dependency safe:
   Import of LangChain is guarded. Importing this module is safe even if
   LangChain is not installed.

3. Simple & explicit interface:
   Clean API that LangChain can use directly by plugging in this client
   as a `BaseChatModel` implementation.

4. True streaming:
   Streaming goes through `LLMTranslator.stream` / `LLMTranslator.arun_stream`,
   so you get protocol-level streaming semantics with LangChain–friendly outputs.

5. Context + observability:
   - `OperationContext` built from LangChain config via `ContextTranslator`
   - Rich error context attached via core `attach_context`
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import wraps
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from corpus_sdk.core.context_translation import ContextTranslator
from corpus_sdk.core.error_context import attach_context
from corpus_sdk.llm.llm_base import (
    LLMProtocolV1,
    OperationContext,
)
from corpus_sdk.llm.framework_adapters.common.llm_translation import (
    JSONRepair,
    LLMFrameworkTranslator,
    LLMPostProcessingConfig,
    LLMTranslator,
    SafetyFilter,
    create_llm_translator,
)
from corpus_sdk.llm.framework_adapters.common.framework_utils import (
    CoercionErrorCodes,
)

logger = logging.getLogger(__name__)

# Framework identifier used consistently for translator + error context
_FRAMEWORK_NAME = "langchain"

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Error-code bundle (CoercionErrorCodes alignment)
# ---------------------------------------------------------------------------

ERROR_CODES = CoercionErrorCodes(
    invalid_result="LANGCHAIN_LLM_INVALID_RESULT",
    empty_result="LANGCHAIN_LLM_EMPTY_RESULT",
    conversion_error="LANGCHAIN_LLM_CONVERSION_ERROR",
    framework_label=_FRAMEWORK_NAME,
)

# Symbolic code for init/config errors (for log/search friendliness)
INIT_CONFIG_ERROR = "LANGCHAIN_LLM_BAD_INIT_CONFIG"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LangChainLLMConfig:
    """
    Configuration for `CorpusLangChainLLM`.

    Mirrors shared `LLMTranslator` knobs while remaining LangChain-specific,
    and aligns validation with other framework adapters (e.g. AutoGen):

    - temperature: must be between 0.0 and 2.0 (inclusive)
    - max_tokens: if provided, must be a positive integer
    """

    model: str = "default"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    framework_version: Optional[str] = None

    post_processing_config: Optional[LLMPostProcessingConfig] = None
    safety_filter: Optional[SafetyFilter] = None
    json_repair: Optional[JSONRepair] = None

    # Behavior toggles (kept simple and conservative)
    enable_metrics: bool = True
    validate_inputs: bool = True

    def __post_init__(self) -> None:
        """Validate configuration immediately when constructed."""
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                f"{INIT_CONFIG_ERROR}: "
                f"temperature must be between 0.0 and 2.0, got {self.temperature}"
            )
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError(
                f"{INIT_CONFIG_ERROR}: "
                f"max_tokens must be positive, got {self.max_tokens}"
            )


# ---------------------------------------------------------------------------
# Error context helpers (lazy, decorator-based, aligned)
# ---------------------------------------------------------------------------


def _extract_dynamic_context(
    instance: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    operation: str,
) -> Dict[str, Any]:
    """
    Extract dynamic context for error attachment.

    Called *only* on the error path by the decorators below to avoid overhead
    on successful calls. Aligned with the richer context used in the AutoGen
    and CrewAI LLM adapters (operation name, streaming flag, request/tenant).
    """
    dynamic_ctx: Dict[str, Any] = {
        "framework_name": _FRAMEWORK_NAME,
        "model": getattr(instance, "model", "unknown"),
        "temperature": getattr(instance, "temperature", 0.7),
        "operation": operation,
    }

    # Messages metrics
    if args:
        first_arg = args[0]
        if isinstance(first_arg, Sequence):
            messages = [m for m in first_arg if isinstance(m, BaseMessage)]
        elif isinstance(first_arg, BaseMessage):
            messages = [first_arg]
        else:
            messages = []

        if messages:
            dynamic_ctx["messages_count"] = len(messages)
            roles: Dict[str, int] = {}
            total_chars = 0
            for msg in messages:
                role = getattr(msg, "type", "unknown")
                roles[role] = roles.get(role, 0) + 1
                content = getattr(msg, "content", "")
                if isinstance(content, str):
                    total_chars += len(content)
            dynamic_ctx["roles_distribution"] = roles
            dynamic_ctx["total_content_chars"] = total_chars

    # Sampling params if present
    for param in ("max_tokens", "top_p", "frequency_penalty", "presence_penalty"):
        if param in kwargs:
            dynamic_ctx[param] = kwargs[param]

    if "stop" in kwargs:
        dynamic_ctx["stop"] = kwargs["stop"]

    # LangChain config flag
    if "config" in kwargs:
        dynamic_ctx["has_config"] = True

    # Request-scoped identifiers (for cross-framework observability alignment)
    for key in ("request_id", "tenant"):
        if key in kwargs:
            dynamic_ctx[key] = kwargs[key]

    # Streaming flag aligned to operation name
    if operation in ("astream", "stream"):
        dynamic_ctx["stream"] = True

    return dynamic_ctx


def _create_error_context_decorator(
    operation: str,
    is_async: bool = False,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Factory for creating error-context decorators with lazy dynamic context.

    Successful calls are unaffected; on exception we compute metrics and
    attach them via `attach_context` with a consistent LLM-oriented operation.
    """

    def decorator_factory(
        **static_context: Any,
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            if is_async:

                @wraps(func)
                async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
                    try:
                        return await func(self, *args, **kwargs)
                    except Exception as exc:  # noqa: BLE001
                        dynamic_ctx = _extract_dynamic_context(
                            self,
                            args,
                            kwargs,
                            operation,
                        )
                        full_ctx = {
                            "error_codes": ERROR_CODES,
                            **static_context,
                            **dynamic_ctx,
                        }
                        attach_context(
                            exc,
                            framework=_FRAMEWORK_NAME,
                            operation=f"llm_{operation}",
                            **full_ctx,
                        )
                        raise

                return async_wrapper
            else:

                @wraps(func)
                def sync_wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
                    try:
                        return func(self, *args, **kwargs)
                    except Exception as exc:  # noqa: BLE001
                        dynamic_ctx = _extract_dynamic_context(
                            self,
                            args,
                            kwargs,
                            operation,
                        )
                        full_ctx = {
                            "error_codes": ERROR_CODES,
                            **static_context,
                            **dynamic_ctx,
                        }
                        attach_context(
                            exc,
                            framework=_FRAMEWORK_NAME,
                            operation=f"llm_{operation}",
                            **full_ctx,
                        )
                        raise

                return sync_wrapper

        return decorator

    return decorator_factory


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
# Context construction helper (defensive, ContextTranslator-based)
# ---------------------------------------------------------------------------


def _build_operation_context_from_config(
    config: Any,
    framework_version: Optional[str],
) -> OperationContext:
    """
    Build an OperationContext from a LangChain config via ContextTranslator.

    Defensive behavior:
    - If `config` is already an OperationContext, return it.
    - Else, call `ContextTranslator.from_langchain(config, ...)`.
    - If that call fails, attach rich context and re-raise.
    - If it returns a non-OperationContext, attach context and raise TypeError.
    """
    if isinstance(config, OperationContext):
        return config

    try:
        ctx = ContextTranslator.from_langchain(
            config=config,
            framework_version=framework_version,
        )
    except Exception as exc:  # noqa: BLE001
        attach_context(
            exc,
            framework=_FRAMEWORK_NAME,
            operation="llm_context_translation",
            framework_version=framework_version,
            source="langchain_config",
            config_type=type(config).__name__,
        )
        raise

    if not isinstance(ctx, OperationContext):
        exc = TypeError(
            "ContextTranslator.from_langchain produced unsupported context type "
            f"{type(ctx).__name__}"
        )
        attach_context(
            exc,
            framework=_FRAMEWORK_NAME,
            operation="llm_context_translation",
            framework_version=framework_version,
            returned_type=type(ctx).__name__,
        )
        raise exc

    return ctx


# ---------------------------------------------------------------------------
# Structural protocol for type-checking (unchanged externally)
# ---------------------------------------------------------------------------


class LangChainLLMProtocol(Protocol):
    """
    Structural protocol for LangChain-compatible Corpus chat models.
    """

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        ...

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        ...

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        ...

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        ...


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------


class CorpusLangChainLLM(BaseChatModel):
    """
    LangChain `BaseChatModel` implementation backed by a Corpus `LLMTranslator`.

    Key points:
    - No direct message translation; all message/format handling
      is delegated to `LLMTranslator` with framework="langchain".
    - This class:
        * Validates inputs
        * Builds `OperationContext` from LangChain config
        * Builds sampling params + lightweight `framework_ctx`
        * Wires LangChain callbacks
        * Uses translator for generation, streaming, and token counting
    """

    # Pydantic v2-style config
    model_config = {
        "arbitrary_types_allowed": True,
        "protected_namespaces": (),
    }

    _translator: LLMTranslator

    model: str = "default"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    _framework_version: Optional[str] = None
    _config: LangChainLLMConfig

    def __init__(
        self,
        *,
        llm_adapter: LLMProtocolV1,
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        framework_version: Optional[str] = None,
        config: Optional[LangChainLLMConfig] = None,
        translator: Optional[LLMFrameworkTranslator] = None,
        post_processing_config: Optional[LLMPostProcessingConfig] = None,
        safety_filter: Optional[SafetyFilter] = None,
        json_repair: Optional[JSONRepair] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the LangChain adapter.
        """
        # Resolve configuration precedence: explicit config > individual params.
        if config is not None:
            self._config = config
            self.model = config.model
            self.temperature = float(config.temperature)
            self.max_tokens = config.max_tokens
            self._framework_version = config.framework_version or framework_version

            effective_post_processing = (
                post_processing_config or config.post_processing_config
            )
            effective_safety_filter = safety_filter or config.safety_filter
            effective_json_repair = json_repair or config.json_repair
        else:
            # Let LangChainLLMConfig handle validation + error messages.
            self._config = LangChainLLMConfig(
                model=model,
                temperature=float(temperature),
                max_tokens=max_tokens,
                framework_version=framework_version,
            )
            self.model = self._config.model
            self.temperature = self._config.temperature
            self.max_tokens = self._config.max_tokens
            self._framework_version = self._config.framework_version

            effective_post_processing = post_processing_config
            effective_safety_filter = safety_filter
            effective_json_repair = json_repair

        # Validate adapter + config invariants in a single place.
        self._validate_init_params(llm_adapter)

        super().__init__(**kwargs)

        # Build the shared LLMTranslator for the "langchain" framework.
        self._translator = create_llm_translator(
            adapter=llm_adapter,
            framework=_FRAMEWORK_NAME,
            translator=translator,
            post_processing_config=effective_post_processing,
            safety_filter=effective_safety_filter,
            json_repair=effective_json_repair,
        )

        logger.info(
            "CorpusLangChainLLM initialized with model=%s, temperature=%.2f, "
            "max_tokens=%s, framework_version=%s",
            self.model,
            self.temperature,
            self.max_tokens or "default",
            self._framework_version or "unknown",
        )

    # ------------------------------------------------------------------ #
    # Init validation helper (symbolic error codes for logs/search)
    # ------------------------------------------------------------------ #

    def _validate_init_params(self, llm_adapter: LLMProtocolV1) -> None:
        """
        Validate initialization parameters and adapter capabilities.

        - Ensures the adapter exposes core `LLMProtocolV1` methods.
        - Validates temperature and max_tokens constraints.
        """
        required_methods = (
            "complete",
            "stream",
            "count_tokens",
            "health",
            "capabilities",
        )
        missing = [
            m
            for m in required_methods
            if not callable(getattr(llm_adapter, m, None))
        ]
        if missing:
            raise TypeError(
                f"{INIT_CONFIG_ERROR}: "
                "llm_adapter must implement LLMProtocolV1; missing methods: "
                + ", ".join(missing)
            )

        if not isinstance(self.temperature, (int, float)) or not (
            0.0 <= float(self.temperature) <= 2.0
        ):
            raise ValueError(
                f"{INIT_CONFIG_ERROR}: temperature must be between 0.0 and 2.0"
            )

        if self.max_tokens is not None:
            if not isinstance(self.max_tokens, int) or self.max_tokens < 1:
                raise ValueError(
                    f"{INIT_CONFIG_ERROR}: max_tokens must be a positive integer"
                )

    # ------------------------------------------------------------------ #
    # LangChain-required properties
    # ------------------------------------------------------------------ #

    @property
    def _llm_type(self) -> str:
        """Identifier used by LangChain in serialization / introspection."""
        return "corpus"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return identifying parameters for LangChain serialization."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "framework_version": self._framework_version,
        }

    # ------------------------------------------------------------------ #
    # Input validation helpers
    # ------------------------------------------------------------------ #

    def _validate_messages(self, messages: List[BaseMessage]) -> None:
        """
        Validate message structure before handing them to the translator.
        """
        if not self._config.validate_inputs:
            return

        if not messages:
            raise ValueError("messages list cannot be empty")

        for idx, msg in enumerate(messages):
            if not isinstance(msg, BaseMessage):
                raise TypeError(
                    f"messages[{idx}] must be a LangChain BaseMessage, got {type(msg)}"
                )
            if getattr(msg, "content", None) in ("", None):
                raise ValueError(f"messages[{idx}] has empty content")

    # ------------------------------------------------------------------ #
    # Context + sampling helpers
    # ------------------------------------------------------------------ #

    def _build_context_and_params(
        self,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> tuple[OperationContext, Dict[str, Any]]:
        """
        Extract OperationContext and sampling params from LangChain kwargs.

        Returns
        -------
        (op_ctx, sampling_params)
        """
        config = kwargs.get("config")
        ctx = _build_operation_context_from_config(
            config=config,
            framework_version=self._framework_version,
        )

        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        params: Dict[str, Any] = {
            "model": kwargs.get("model", self.model),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": kwargs.get("top_p"),
            "frequency_penalty": kwargs.get("frequency_penalty"),
            "presence_penalty": kwargs.get("presence_penalty"),
            "stop_sequences": stop,
        }

        clean_params = {k: v for k, v in params.items() if v is not None}
        return ctx, clean_params

    def _build_framework_ctx(
        self,
        *,
        operation: str,
        stream: bool,
        model: Optional[str],
        config: Any,
        tools: Any = None,
        tool_choice: Any = None,
        system_message: Any = None,
        request_id: Optional[str] = None,
        tenant: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build a framework_ctx payload for the LangChain translator.

        This mirrors the richer framework_ctx used in the AutoGen and CrewAI
        adapters, carrying:

        - Framework identity + version
        - Operation name and streaming flag
        - Effective model used for routing
        - Optional request/tenant identifiers
        - LangChain config blob for observability
        - Tooling + system message hints
        """
        framework_ctx: Dict[str, Any] = {
            "framework": _FRAMEWORK_NAME,
            "framework_version": self._framework_version,
            "operation": operation,
            "stream": bool(stream),
        }

        if model is not None:
            framework_ctx["model"] = model

        if request_id is not None:
            framework_ctx["request_id"] = request_id
        if tenant is not None:
            framework_ctx["tenant"] = tenant

        if config is not None:
            framework_ctx["langchain_config"] = config

        if tools is not None:
            framework_ctx["tools"] = tools
        if tool_choice is not None:
            framework_ctx["tool_choice"] = tool_choice
        if system_message is not None:
            framework_ctx["system_message"] = system_message

        # Metrics toggle can be useful for downstream observers.
        framework_ctx["enable_metrics"] = bool(self._config.enable_metrics)

        return framework_ctx

    # ------------------------------------------------------------------ #
    # Result normalization helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _ensure_chat_result(result: Any) -> ChatResult:
        """
        Normalize translator output into a LangChain ChatResult.

        The `"langchain"` framework translator is expected to return ChatResult
        directly; this helper provides a defensive fallback for dict/string.
        """
        if isinstance(result, ChatResult):
            return result

        if isinstance(result, dict):
            text = str(
                result.get("text")
                or result.get("content")
                or result.get("message", {}).get("content", "")
            )
        else:
            text = str(result)

        message = AIMessage(content=text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @staticmethod
    def _ensure_generation_chunk(chunk: Any) -> ChatGenerationChunk:
        """
        Normalize translator streaming output into ChatGenerationChunk.
        """
        if isinstance(chunk, ChatGenerationChunk):
            return chunk

        if isinstance(chunk, dict):
            text = str(chunk.get("text") or chunk.get("delta") or "")
        else:
            text = str(chunk)

        ai_chunk = AIMessageChunk(content=text)
        return ChatGenerationChunk(message=ai_chunk)

    # ------------------------------------------------------------------ #
    # LangChain async API (via LLMTranslator)
    # ------------------------------------------------------------------ #

    @with_async_llm_error_context("agenerate")
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Async chat generation entrypoint used by LangChain.
        """
        self._validate_messages(messages)

        ctx, params = self._build_context_and_params(
            stop=stop,
            **kwargs,
        )
        model_for_context = params.get("model", self.model)

        framework_ctx = self._build_framework_ctx(
            operation="agenerate",
            stream=False,
            model=model_for_context,
            config=kwargs.get("config"),
            tools=kwargs.get("tools"),
            tool_choice=kwargs.get("tool_choice"),
            system_message=kwargs.get("system_message"),
            request_id=kwargs.get("request_id"),
            tenant=kwargs.get("tenant"),
        )

        if run_manager is not None:
            await run_manager.on_llm_start(
                self,
                messages,
                invocation_params=params,
                run_id=ctx.request_id,
            )

        try:
            result = await self._translator.arun_complete(
                raw_messages=messages,
                model=model_for_context,
                max_tokens=params.get("max_tokens"),
                temperature=params.get("temperature"),
                top_p=params.get("top_p"),
                frequency_penalty=params.get("frequency_penalty"),
                presence_penalty=params.get("presence_penalty"),
                stop_sequences=params.get("stop_sequences"),
                tools=kwargs.get("tools"),
                tool_choice=kwargs.get("tool_choice"),
                system_message=kwargs.get("system_message"),
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

            chat_result = self._ensure_chat_result(result)

            if run_manager is not None:
                await run_manager.on_llm_end(chat_result)

            return chat_result
        except Exception as exc:  # noqa: BLE001
            if run_manager is not None:
                await run_manager.on_llm_error(exc)
            raise

    @with_async_llm_error_context("astream")
    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """
        Async streaming entrypoint used by LangChain.
        """
        self._validate_messages(messages)

        ctx, params = self._build_context_and_params(
            stop=stop,
            **kwargs,
        )
        model_for_context = params.get("model", self.model)

        framework_ctx = self._build_framework_ctx(
            operation="astream",
            stream=True,
            model=model_for_context,
            config=kwargs.get("config"),
            tools=kwargs.get("tools"),
            tool_choice=kwargs.get("tool_choice"),
            system_message=kwargs.get("system_message"),
            request_id=kwargs.get("request_id"),
            tenant=kwargs.get("tenant"),
        )

        if run_manager is not None:
            await run_manager.on_llm_start(
                self,
                messages,
                invocation_params=params,
                run_id=ctx.request_id,
            )

        stream_canceled = False
        agen: Optional[AsyncIterator[Any]] = None

        try:
            agen = await self._translator.arun_stream(
                raw_messages=messages,
                model=model_for_context,
                max_tokens=params.get("max_tokens"),
                temperature=params.get("temperature"),
                top_p=params.get("top_p"),
                frequency_penalty=params.get("frequency_penalty"),
                presence_penalty=params.get("presence_penalty"),
                stop_sequences=params.get("stop_sequences"),
                tools=kwargs.get("tools"),
                tool_choice=kwargs.get("tool_choice"),
                system_message=kwargs.get("system_message"),
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

            async for chunk in agen:
                gen_chunk = self._ensure_generation_chunk(chunk)
                text = gen_chunk.message.content or ""

                if run_manager is not None and text:
                    try:
                        await run_manager.on_llm_new_token(text, chunk=gen_chunk)
                    except Exception as callback_error:  # noqa: BLE001
                        logger.warning(
                            "LLM new token callback failed: %s",
                            callback_error,
                        )
                        stream_canceled = True
                        break

                yield gen_chunk
        except Exception as exc:  # noqa: BLE001
            if run_manager is not None:
                await run_manager.on_llm_error(exc)
            raise
        finally:
            if agen is not None and hasattr(agen, "aclose"):
                try:
                    await agen.aclose()  # type: ignore[func-returns-value]
                except Exception as cleanup_error:  # noqa: BLE001
                    logger.warning(
                        "Async stream cleanup failed in LangChain adapter: %s",
                        cleanup_error,
                    )

            if run_manager is not None and not stream_canceled:
                completion_result = ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(content=""),
                            generation_info={
                                "streaming": True,
                                "completed": True,
                                "model": model_for_context,
                            },
                        )
                    ]
                )
                await run_manager.on_llm_end(completion_result)

    # ------------------------------------------------------------------ #
    # LangChain sync API (via LLMTranslator)
    # ------------------------------------------------------------------ #

    @with_llm_error_context("generate")
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Sync chat generation entrypoint used by LangChain.
        """
        self._validate_messages(messages)

        ctx, params = self._build_context_and_params(
            stop=stop,
            **kwargs,
        )
        model_for_context = params.get("model", self.model)

        framework_ctx = self._build_framework_ctx(
            operation="generate",
            stream=False,
            model=model_for_context,
            config=kwargs.get("config"),
            tools=kwargs.get("tools"),
            tool_choice=kwargs.get("tool_choice"),
            system_message=kwargs.get("system_message"),
            request_id=kwargs.get("request_id"),
            tenant=kwargs.get("tenant"),
        )

        if run_manager is not None:
            run_manager.on_llm_start(
                self,
                messages,
                invocation_params=params,
                run_id=ctx.request_id,
            )

        try:
            result = self._translator.complete(
                raw_messages=messages,
                model=model_for_context,
                max_tokens=params.get("max_tokens"),
                temperature=params.get("temperature"),
                top_p=params.get("top_p"),
                frequency_penalty=params.get("frequency_penalty"),
                presence_penalty=params.get("presence_penalty"),
                stop_sequences=params.get("stop_sequences"),
                tools=kwargs.get("tools"),
                tool_choice=kwargs.get("tool_choice"),
                system_message=kwargs.get("system_message"),
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

            chat_result = self._ensure_chat_result(result)

            if run_manager is not None:
                run_manager.on_llm_end(chat_result)

            return chat_result
        except Exception as exc:  # noqa: BLE001
            if run_manager is not None:
                run_manager.on_llm_error(exc)
            raise

    @with_llm_error_context("stream")
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """
        Sync streaming entrypoint used by LangChain.
        """
        self._validate_messages(messages)

        ctx, params = self._build_context_and_params(
            stop=stop,
            **kwargs,
        )
        model_for_context = params.get("model", self.model)

        framework_ctx = self._build_framework_ctx(
            operation="stream",
            stream=True,
            model=model_for_context,
            config=kwargs.get("config"),
            tools=kwargs.get("tools"),
            tool_choice=kwargs.get("tool_choice"),
            system_message=kwargs.get("system_message"),
            request_id=kwargs.get("request_id"),
            tenant=kwargs.get("tenant"),
        )

        if run_manager is not None:
            run_manager.on_llm_start(
                self,
                messages,
                invocation_params=params,
                run_id=ctx.request_id,
            )

        stream_canceled = False
        iterator: Optional[Iterator[Any]] = None

        try:
            iterator = self._translator.stream(
                raw_messages=messages,
                model=model_for_context,
                max_tokens=params.get("max_tokens"),
                temperature=params.get("temperature"),
                top_p=params.get("top_p"),
                frequency_penalty=params.get("frequency_penalty"),
                presence_penalty=params.get("presence_penalty"),
                stop_sequences=params.get("stop_sequences"),
                tools=kwargs.get("tools"),
                tool_choice=kwargs.get("tool_choice"),
                system_message=kwargs.get("system_message"),
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

            for chunk in iterator:
                gen_chunk = self._ensure_generation_chunk(chunk)
                text = gen_chunk.message.content or ""

                if run_manager is not None and text:
                    try:
                        run_manager.on_llm_new_token(text, chunk=gen_chunk)
                    except Exception as callback_error:  # noqa: BLE001
                        logger.warning(
                            "LLM new token callback failed: %s",
                            callback_error,
                        )
                        stream_canceled = True
                        break

                yield gen_chunk
        except Exception as exc:  # noqa: BLE001
            if run_manager is not None:
                run_manager.on_llm_error(exc)
            raise
        finally:
            if iterator is not None and hasattr(iterator, "close"):
                try:
                    iterator.close()  # type: ignore[func-returns-value]
                except Exception as cleanup_error:  # noqa: BLE001
                    logger.warning(
                        "Sync stream cleanup failed in LangChain adapter: %s",
                        cleanup_error,
                    )

            if run_manager is not None and not stream_canceled:
                completion_result = ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(content=""),
                            generation_info={
                                "streaming": True,
                                "completed": True,
                                "model": model_for_context,
                            },
                        )
                    ]
                )
                run_manager.on_llm_end(completion_result)

    # ------------------------------------------------------------------ #
    # Token counting (via LLMTranslator with robust fallbacks)
    # ------------------------------------------------------------------ #

    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        """
        Estimate token count for a list of LangChain messages.

        Primary path:
            Use LLMTranslator.count_tokens_for_messages so that token counting
            respects provider-specific formatting and configuration.

        Fallback:
            Character-based heuristic estimation.
        """
        if not messages:
            return 0

        ctx = _build_operation_context_from_config(
            config=None,
            framework_version=self._framework_version,
        )
        framework_ctx: Dict[str, Any] = {
            "framework": _FRAMEWORK_NAME,
            "framework_version": self._framework_version,
        }

        try:
            result = self._translator.count_tokens_for_messages(
                raw_messages=messages,
                model=self.model,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

            if isinstance(result, int):
                return result
            if isinstance(result, Mapping):
                for key in ("tokens", "count", "total_tokens"):
                    value = result.get(key)
                    if isinstance(value, int):
                        return value
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "LLMTranslator.count_tokens_for_messages failed, "
                "falling back to heuristic: %s",
                exc,
            )

        combined_text = self._combine_messages_for_counting(messages)
        if not combined_text:
            return 0

        char_count = len(combined_text)
        message_count = len(messages)
        char_based = max(1, char_count // 4)
        message_based = max(1, message_count)
        return max(char_based, message_based)

    def _combine_messages_for_counting(self, messages: List[BaseMessage]) -> str:
        """Combine messages into a single string for heuristic token counting."""
        parts: List[str] = []
        for msg in messages:
            role = getattr(msg, "type", "user")
            content = str(getattr(msg, "content", ""))
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def get_num_tokens(self, text: str) -> int:
        """
        Token counting for a single text string.

        Implemented via get_num_tokens_from_messages for consistency.
        """
        return self.get_num_tokens_from_messages([HumanMessage(content=text)])


__all__ = [
    "LangChainLLMProtocol",
    "LangChainLLMConfig",
    "CorpusLangChainLLM",
    "with_llm_error_context",
    "with_async_llm_error_context",
    "ERROR_CODES",
]
