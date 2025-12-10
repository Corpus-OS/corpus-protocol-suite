# corpus_sdk/mcp/llm_service.py
# SPDX-License-Identifier: Apache-2.0

"""
MCP adapter for Corpus LLM protocol.

This module exposes a Corpus `LLMProtocolV1` behind the shared `LLMTranslator`
layer as an MCP-oriented LLM service with:

- Context-aware MCP → OperationContext translation via `from_mcp`
- Async non-streaming + streaming completion APIs
- Framework-agnostic translation via `LLMTranslator` (no direct message translation)
- Token counting via `LLMTranslator` with a robust heuristic fallback
- Lightweight token estimation helper for quick local heuristics
- Rich error context attachment for observability via `CoercionErrorCodes`

Design goals
------------

1. Protocol + translator first:
   All calls go through the shared `LLMTranslator` with the `"mcp"` framework,
   so message normalization, post-processing, tools, and error context are
   consistent with other framework adapters.

2. Thin MCP-facing surface:
   Keep this module focused on:
   - Building `OperationContext` from raw MCP metadata via `from_mcp`
   - Assembling sampling parameters + a small `framework_ctx`
   - Delegating to `LLMTranslator` for completion + streaming + token counting

3. Alignment with other adapters:
   Mirror the structure of the LangChain / LlamaIndex / Semantic Kernel adapters:
   - Validated config dataclass
   - `LLMProtocolV1` adapter validation
   - Shared error-code bundle via `CoercionErrorCodes`
   - Token counting via `LLMTranslator` with robust fallback
   - Simple token estimation helper for callers that need it
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Mapping,
    Optional,
    Sequence,
)

from corpus_sdk.core.context_translation import from_mcp
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

_FRAMEWORK_NAME = "mcp"


# ---------------------------------------------------------------------------
# Error-code bundle (CoercionErrorCodes alignment)
# ---------------------------------------------------------------------------

ERROR_CODES = CoercionErrorCodes(
    invalid_result="MCP_LLM_INVALID_RESULT",
    empty_result="MCP_LLM_EMPTY_RESULT",
    conversion_error="MCP_LLM_CONVERSION_ERROR",
    framework_label=_FRAMEWORK_NAME,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MCPLLMServiceConfig:
    """
    Configuration for `MCPLLMTranslationService`.

    This mirrors the other framework adapters and wires through to the
    shared `LLMTranslator` knobs:

    - Default model / temperature / max_tokens
    - Optional post-processing configuration
    - Optional safety / JSON repair behavior
    - Optional framework version tagging
    - Optional metrics + input validation toggles (for callers)
    """

    model: str = "default"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    framework_version: Optional[str] = None

    post_processing_config: Optional[LLMPostProcessingConfig] = None
    safety_filter: Optional[SafetyFilter] = None
    json_repair: Optional[JSONRepair] = None

    # Behavior toggles (kept simple + aligned with other adapters)
    enable_metrics: bool = True
    validate_inputs: bool = True

    def __post_init__(self) -> None:
        """Validate configuration immediately when constructed."""
        if not 0.0 <= float(self.temperature) <= 2.0:
            raise ValueError(
                "MCPLLMServiceConfig: temperature must be between 0.0 and 2.0, "
                f"got {self.temperature}"
            )
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError(
                "MCPLLMServiceConfig: max_tokens must be positive, "
                f"got {self.max_tokens}"
            )


# ---------------------------------------------------------------------------
# MCP-facing LLM service
# ---------------------------------------------------------------------------


class MCPLLMTranslationService:
    """
    MCP-facing LLM service backed by a Corpus `LLMTranslator`.

    This is intentionally thin and aligned with the other framework adapters:

    - Builds an `OperationContext` from MCP metadata via `from_mcp(...)`
    - Builds sampling params and a small `framework_ctx`
    - Delegates to `LLMTranslator` with `framework="mcp"`
    - Provides:
        * `acomplete`   – async, non-streaming completion
        * `astream`     – async streaming completion
        * `count_tokens_for_prompt` – token counting via `LLMTranslator`
          with robust fallback
        * `estimate_tokens_for_prompt` – simple char-based heuristic

    The MCP-specific behavior (how MCP messages are shaped, how tools are
    wired, etc.) is handled in the registered `"mcp"` `LLMFrameworkTranslator`,
    not in this file.
    """

    _translator: LLMTranslator

    model: str = "default"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    _framework_version: Optional[str] = None
    _config: MCPLLMServiceConfig

    def __init__(
        self,
        *,
        llm_adapter: LLMProtocolV1,
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        framework_version: Optional[str] = None,
        config: Optional[MCPLLMServiceConfig] = None,
        translator: Optional[LLMFrameworkTranslator] = None,
        post_processing_config: Optional[LLMPostProcessingConfig] = None,
        safety_filter: Optional[SafetyFilter] = None,
        json_repair: Optional[JSONRepair] = None,
    ) -> None:
        """
        Initialize the MCP LLM service.

        Parameters
        ----------
        llm_adapter :
            Implementation of `LLMProtocolV1` (Corpus core LLM protocol).
        model :
            Default model name to use when callers do not override it.
        temperature :
            Default sampling temperature.
        max_tokens :
            Default max_tokens for completions (optional).
        framework_version :
            Optional `"mcp"` framework version tag for observability.
        config :
            Optional `MCPLLMServiceConfig`. If provided, overrides individual
            `model` / `temperature` / `max_tokens` / `framework_version` kwargs.
        translator :
            Optional explicit `LLMFrameworkTranslator` for `"mcp"`.
        post_processing_config, safety_filter, json_repair :
            Optional knobs forwarded into `create_llm_translator`.
        """
        # Resolve configuration precedence: explicit config > individual kwargs.
        if config is not None:
            self._config = config
            self.model = config.model
            self.temperature = float(config.temperature)
            self.max_tokens = config.max_tokens
            self._framework_version = config.framework_version or framework_version

            effective_pp = post_processing_config or config.post_processing_config
            effective_safety = safety_filter or config.safety_filter
            effective_json_repair = json_repair or config.json_repair
        else:
            self._config = MCPLLMServiceConfig(
                model=model,
                temperature=float(temperature),
                max_tokens=max_tokens,
                framework_version=framework_version,
            )
            self.model = self._config.model
            self.temperature = self._config.temperature
            self.max_tokens = self._config.max_tokens
            self._framework_version = self._config.framework_version

            effective_pp = post_processing_config
            effective_safety = safety_filter
            effective_json_repair = json_repair

        # Validate the adapter + basic sampling invariants.
        self._validate_init_params(llm_adapter)

        # Build the shared LLMTranslator for the "mcp" framework.
        self._translator = create_llm_translator(
            adapter=llm_adapter,
            framework=_FRAMEWORK_NAME,
            translator=translator,
            post_processing_config=effective_pp,
            safety_filter=effective_safety,
            json_repair=effective_json_repair,
        )

        logger.info(
            "MCPLLMTranslationService initialized with model=%s, temperature=%.2f, "
            "max_tokens=%s, framework_version=%s",
            self.model,
            self.temperature,
            self.max_tokens or "default",
            self._framework_version or "unknown",
        )

    # ------------------------------------------------------------------ #
    # Init validation helper
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
                "MCPLLMTranslationService: llm_adapter must implement LLMProtocolV1; "
                "missing methods: " + ", ".join(missing)
            )

        if not isinstance(self.temperature, (int, float)) or not (
            0.0 <= float(self.temperature) <= 2.0
        ):
            raise ValueError(
                "MCPLLMTranslationService: temperature must be between 0.0 and 2.0"
            )

        if self.max_tokens is not None:
            if not isinstance(self.max_tokens, int) or self.max_tokens < 1:
                raise ValueError(
                    "MCPLLMTranslationService: max_tokens must be a positive integer"
                )

    # ------------------------------------------------------------------ #
    # Internal helpers: validation, context, sampling, framework_ctx
    # ------------------------------------------------------------------ #

    def _validate_prompt(self, prompt: str) -> None:
        """Very lightweight prompt validation hook."""
        if not prompt or not str(prompt).strip():
            raise ValueError("prompt cannot be empty")

    def _build_operation_context(
        self,
        mcp_context: Mapping[str, Any],
    ) -> OperationContext:
        """
        Build `OperationContext` from MCP context via `from_mcp`.

        Any errors are wrapped with additional context but re-raised.
        """
        try:
            return from_mcp(mcp_context)
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework=_FRAMEWORK_NAME,
                operation="llm_context_translation",
                error_codes=ERROR_CODES,
                source="mcp_context",
                context_type=type(mcp_context).__name__,
            )
            raise

    def _build_sampling_params(
        self,
        *,
        model: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> Dict[str, Any]:
        """
        Build sampling parameters with instance defaults as fallback.

        This is kept deliberately small: callers can pass additional kwargs
        via higher-level MCP plumbing if needed (e.g., top_p).
        """
        params: Dict[str, Any] = {
            "model": model or self.model,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }
        # Strip None for a clean param set
        return {k: v for k, v in params.items() if v is not None}

    def _build_framework_ctx(
        self,
        *,
        operation: str,
        stream: bool,
        model: str,
        request_id: Optional[str],
        mcp_context: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Build a small `framework_ctx` payload for the MCP translator.

        This is informational; all real translation logic lives in the
        registered `"mcp"` `LLMFrameworkTranslator`.
        """
        framework_ctx: Dict[str, Any] = {
            "framework": _FRAMEWORK_NAME,
            "framework_version": self._framework_version,
            "operation": operation,
            "stream": stream,
            "model": model,
            "request_id": request_id,
        }

        # Forward a small, sanitized MCP context slice if present.
        if mcp_context:
            framework_ctx["mcp_context_keys"] = list(mcp_context.keys())

        return framework_ctx

    # ------------------------------------------------------------------ #
    # Token estimation + counting
    # ------------------------------------------------------------------ #

    def _estimate_tokens(self, text: str) -> int:
        """
        Simple, conservative token estimator.

        Uses the same style as other adapters: ~4 characters per token as
        a rough heuristic. This is only used for callers that want quick,
        local estimates; actual provider counting should be done via
        `count_tokens_for_messages` when available.
        """
        if not text:
            return 0
        return max(1, len(text) // 4)

    # Public helper for callers that want a quick heuristic estimate.
    def estimate_tokens_for_prompt(self, prompt: str) -> int:
        return self._estimate_tokens(prompt)

    def count_tokens_for_prompt(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        mcp_context: Optional[Mapping[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> int:
        """
        Count tokens for a single prompt using LLMTranslator when possible.

        Preferred path:
        - Use `LLMTranslator.count_tokens_for_messages` so token counting
          matches provider-specific formatting and behavior.

        Fallback:
        - If translator/adapter counting fails or returns an unexpected shape,
          fall back to the local `_estimate_tokens` heuristic.
        """
        if not prompt or not prompt.strip():
            return 0

        # Build a minimal MCP OperationContext for counting.
        mcp_ctx: Mapping[str, Any] = mcp_context or {}
        try:
            ctx = self._build_operation_context(mcp_ctx)
        except Exception:
            # If context building itself fails, just use the heuristic.
            logger.debug("MCP token count: failed to build OperationContext; using heuristic")
            return self._estimate_tokens(prompt)

        params = self._build_sampling_params(
            model=model,
            temperature=None,
            max_tokens=None,
        )
        model_for_ctx = str(params.get("model", self.model))
        framework_ctx = self._build_framework_ctx(
            operation="count_tokens",
            stream=False,
            model=model_for_ctx,
            request_id=request_id or ctx.request_id,
            mcp_context=mcp_ctx,
        )

        # Single user message for counting.
        messages = [{"role": "user", "content": prompt}]

        try:
            tokens_any = self._translator.count_tokens_for_messages(
                raw_messages=messages,
                model=model_for_ctx,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

            # Direct int result
            if isinstance(tokens_any, int):
                return tokens_any

            # Mapping-style result from translator
            if isinstance(tokens_any, Mapping):
                for key in ("tokens", "total_tokens", "count"):
                    value = tokens_any.get(key)
                    if isinstance(value, int):
                        return value

            logger.debug(
                "MCP token count: unexpected result shape %r; falling back to heuristic",
                type(tokens_any),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "MCP token count via translator failed; falling back to heuristic: %s",
                exc,
            )

        return self._estimate_tokens(prompt)

    # ------------------------------------------------------------------ #
    # MCP async completion API (non-streaming)
    # ------------------------------------------------------------------ #

    async def acomplete(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_message: Optional[str] = None,
        tools: Optional[Sequence[Any]] = None,
        tool_choice: Optional[Any] = None,
        mcp_context: Optional[Mapping[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> Any:
        """
        Async non-streaming completion for MCP callers.

        Parameters
        ----------
        prompt :
            User prompt (single-turn). The MCP translator is responsible for
            mapping this into whatever raw message structure is needed.
        model, max_tokens, temperature :
            Optional sampling overrides; fall back to instance defaults.
        system_message :
            Optional explicit system message; passed through to the translator.
        tools, tool_choice :
            Optional tool definitions / selection, forwarded into translator.
        mcp_context :
            Arbitrary MCP metadata that will be converted into an
            `OperationContext` via `from_mcp`.
        request_id :
            Optional request identifier for observability.

        Returns
        -------
        Any
            Whatever shape the registered `"mcp"` `LLMFrameworkTranslator`
            produces for a completion (often a dict with text/model/usage).
        """
        if self._config.validate_inputs:
            self._validate_prompt(prompt)

        mcp_context = mcp_context or {}
        ctx = self._build_operation_context(mcp_context)
        params = self._build_sampling_params(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        model_for_ctx = str(params.get("model", self.model))
        framework_ctx = self._build_framework_ctx(
            operation="acomplete",
            stream=False,
            model=model_for_ctx,
            request_id=request_id or ctx.request_id,
            mcp_context=mcp_context,
        )

        # Build a simple chat-style message list: single user message.
        messages = [{"role": "user", "content": prompt}]

        try:
            result = await self._translator.arun_complete(
                raw_messages=messages,
                model=params.get("model"),
                max_tokens=params.get("max_tokens"),
                temperature=params.get("temperature"),
                top_p=None,
                frequency_penalty=None,
                presence_penalty=None,
                stop_sequences=None,
                tools=tools,
                tool_choice=tool_choice,
                system_message=system_message,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            return result
        except Exception as exc:  # noqa: BLE001
            # Attach rich error context but never mask the original exception.
            try:
                attach_context(
                    exc,
                    framework=_FRAMEWORK_NAME,
                    operation="llm_acomplete",
                    error_codes=ERROR_CODES,
                    model=model_for_ctx,
                    request_id=request_id or ctx.request_id,
                    prompt_chars=len(prompt),
                    stream=False,
                )
            except Exception:  # noqa: BLE001
                pass
            raise

    # ------------------------------------------------------------------ #
    # MCP async streaming API
    # ------------------------------------------------------------------ #

    async def astream(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_message: Optional[str] = None,
        tools: Optional[Sequence[Any]] = None,
        tool_choice: Optional[Any] = None,
        mcp_context: Optional[Mapping[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> AsyncIterator[Any]:
        """
        Async streaming completion for MCP callers.

        Yields whatever streaming chunk shape the `"mcp"` framework translator
        defines (commonly `{"text": ..., "is_final": ...}`-style dicts).
        """
        if self._config.validate_inputs:
            self._validate_prompt(prompt)

        mcp_context = mcp_context or {}
        ctx = self._build_operation_context(mcp_context)
        params = self._build_sampling_params(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        model_for_ctx = str(params.get("model", self.model))
        framework_ctx = self._build_framework_ctx(
            operation="astream",
            stream=True,
            model=model_for_ctx,
            request_id=request_id or ctx.request_id,
            mcp_context=mcp_context,
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            agen = await self._translator.arun_stream(
                raw_messages=messages,
                model=params.get("model"),
                max_tokens=params.get("max_tokens"),
                temperature=params.get("temperature"),
                top_p=None,
                frequency_penalty=None,
                presence_penalty=None,
                stop_sequences=None,
                tools=tools,
                tool_choice=tool_choice,
                system_message=system_message,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

            async for chunk in agen:
                yield chunk
        except Exception as exc:  # noqa: BLE001
            try:
                attach_context(
                    exc,
                    framework=_FRAMEWORK_NAME,
                    operation="llm_astream",
                    error_codes=ERROR_CODES,
                    model=model_for_ctx,
                    request_id=request_id or ctx.request_id,
                    prompt_chars=len(prompt),
                    stream=True,
                )
            except Exception:  # noqa: BLE001
                pass
            raise


__all__ = [
    "MCPLLMServiceConfig",
    "MCPLLMTranslationService",
    "ERROR_CODES",
]

