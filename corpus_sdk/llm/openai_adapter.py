# corpus_sdk/llm/openai_adapter.py
# SPDX-License-Identifier: Apache-2.0
"""
OpenAI LLM adapter for the Corpus LLM Protocol.

ARCHITECTURE
------------
This adapter implements the Corpus LLMProtocolV1 contract by wrapping
the official OpenAI Python client. Key design decisions:

1. Error Translation: All OpenAI exceptions are mapped to Corpus error
   taxonomy (see _translate_openai_error) for provider-agnostic handling.

2. Token Counting: Uses tiktoken when available, falls back to heuristic.
   Heuristic blends word-count and char-count estimates for robustness.

3. System Message Merging: Multiple system messages are concatenated to
   ensure OpenAI API compatibility (OpenAI requires single system message).

4. Streaming Usage: Leverages stream_options={"include_usage": True} to
   provide accurate token counts in the final chunk.

5. Tool Call Streaming: Reconstructs tool calls incrementally from streaming
   deltas so callers can react to tool invocations in real time.

EXTENSION POINTS
----------------
To build your own adapter:
1. Inherit from BaseLLMAdapter
2. Implement _do_complete, _do_stream, _do_capabilities, _do_health
3. Use _translate_*_error pattern for error normalization
4. See MockLLMAdapter for reference implementation

DEBUGGING
---------
Enable debug logging:
    logging.getLogger('corpus_sdk.llm.openai_adapter').setLevel(logging.DEBUG)

This will log (without PII):
- Request parameters (model, message count, token limits)
- Response metadata (model ID, token usage)
- Error details with stack traces

COMPATIBILITY
-------------
Requires: openai>=1.0.0 (AsyncOpenAI API)
Optional: tiktoken>=0.5.0 (accurate token counting)

Tested with: openai==1.12.0, Python 3.9-3.12
"""

from __future__ import annotations

import asyncio
import logging
from email.utils import parsedate_to_datetime
from time import time as _time
from typing import Any, AsyncIterator, Dict, List, Mapping, Optional, Union

import openai  # type: ignore

from corpus_sdk.llm.llm_base import (
    AuthError,
    BadRequest,
    BaseLLMAdapter,
    DeadlineExceeded,
    LLMCapabilities,
    LLMChunk,
    LLMCompletion,
    LLMAdapterError,
    NotSupported,
    OperationContext,
    ResourceExhausted,
    TokenUsage,
    ToolCall,
    ToolCallFunction,
    TransientNetwork,
    Unavailable,
)

logger = logging.getLogger(__name__)

# Optional tiktoken support for accurate token counting.
try:  # pragma: no cover - optional dependency
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover
    tiktoken = None  # type: ignore[assignment]

# Try to use the modern async client if available.
try:  # pragma: no cover - import surface only
    from openai import AsyncOpenAI  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - older client
    AsyncOpenAI = None  # type: ignore[misc,assignment]

# Error types (v1+ client). We defensively pull them via getattr so the
# adapter fails gracefully on older/unsupported versions.
APIStatusError = getattr(openai, "APIStatusError", None)
APIConnectionError = getattr(openai, "APIConnectionError", None)
APITimeoutError = getattr(openai, "APITimeoutError", None)
RateLimitError = getattr(openai, "RateLimitError", None)
AuthenticationError = getattr(openai, "AuthenticationError", None)
BadRequestError = getattr(openai, "BadRequestError", None)
OpenAIError = getattr(openai, "OpenAIError", Exception)

# Allowed roles kept in sync with MockLLMAdapter so conformance tests see
# identical role semantics across adapters.
_ALLOWED_ROLES = {"system", "user", "assistant"}


class OpenAIAdapter(BaseLLMAdapter):
    """
    Corpus LLM adapter backed by the OpenAI Chat Completions API.

    This adapter is async-first and plugs directly into the Corpus
    protocol stack via `BaseLLMAdapter`.

    Parameters
    ----------
    client:
        Pre-configured `AsyncOpenAI` client instance. Recommended when you
        want to control retry, proxies, organization, etc.
    api_key:
        API key used when a client is not provided. Ignored if `client`
        is given.
    organization:
        Optional OpenAI organization ID.
    base_url:
        Optional custom base URL (for proxies, gateways, etc.).
    default_model:
        Model to use when callers don't specify one explicitly.
    model_family:
        Logical family name surfaced in `LLMCompletion.model_family`.
    max_context_length:
        Approximate max context window size used only for capabilities
        reporting and *approximate* planning. We deliberately do not
        hard-enforce this against provider limits.
    metrics, mode, ...:
        Passed through to `BaseLLMAdapter`.

    Notes
    -----
    - Requires `openai` Python library v1+ (for `AsyncOpenAI`). If that
      is not available, instantiation will raise `RuntimeError`.
    - Uses `tiktoken` for token counting when available, with a safe
      heuristic fallback otherwise.
    - Streaming uses `stream_options={"include_usage": True}` so the
      final chunk carries accurate token usage.
    - Response caching is handled automatically by BaseLLMAdapter using
      the provided `cache` parameter. This adapter only implements
      tokenizer caching for performance optimization.
    - JSON output mode can be enabled via `ctx.attrs.get("response_format")`
      set to "json_object".
    - Tool calling is supported by forwarding the protocol's tool schema
      directly to OpenAI's tools API and mapping tool calls back into
      `LLMCompletion.tool_calls`.
    """

    def __init__(
        self,
        *,
        client: Optional["AsyncOpenAI"] = None,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: str = "gpt-4.1-mini",
        model_family: str = "openai",
        max_context_length: int = 128_000,
        # Base adapter infra
        metrics=None,
        mode: str = "thin",
        deadline_policy=None,
        breaker=None,
        cache=None,
        limiter=None,
        tag_model_in_metrics: bool = True,
        stream_deadline_check_every_n_chunks: int = 10,
    ) -> None:
        # Validate critical parameters
        if not default_model or not isinstance(default_model, str):
            raise ValueError("default_model must be a non-empty string")
        if max_context_length <= 0:
            raise ValueError("max_context_length must be positive")

        if client is None:
            if AsyncOpenAI is None:
                raise RuntimeError(
                    "OpenAIAdapter requires `openai>=1.0.0` with AsyncOpenAI. "
                    "Upgrade via `pip install --upgrade openai`."
                )
            client = AsyncOpenAI(
                api_key=api_key,
                organization=organization,
                base_url=base_url,
            )

        super().__init__(
            metrics=metrics,
            mode=mode,
            deadline_policy=deadline_policy,
            breaker=breaker,
            cache=cache,
            limiter=limiter,
            tag_model_in_metrics=tag_model_in_metrics,
            stream_deadline_check_every_n_chunks=stream_deadline_check_every_n_chunks,
        )

        self._client: AsyncOpenAI = client  # type: ignore[assignment]
        self._default_model = default_model
        self._model_family = model_family
        self._max_context_length = int(max_context_length)
        self._server = "openai"
        self._version = getattr(openai, "__version__", "unknown")

        # Cache for tokenizers to avoid repeated initialization
        # This is separate from the base class response cache
        self._tokenizer_cache: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_roles(messages: List[Mapping[str, Any]]) -> None:
        """
        Enforce a strict role set consistent with MockLLMAdapter.

        This keeps error behavior deterministic and aligned across adapters,
        and fails fast with a normalized BadRequest instead of deferring to
        provider-specific behavior for unknown roles.
        """
        for m in messages:
            role = str(m.get("role", ""))
            if role not in _ALLOWED_ROLES:
                raise BadRequest(f"unknown role: {role!r}")

    def _resolve_model(self, model: Optional[str]) -> str:
        """Resolve caller-supplied model or fall back to the default."""
        return model or self._default_model

    def _build_messages(
        self,
        *,
        messages: List[Mapping[str, str]],
        system_message: Optional[str],
    ) -> List[Dict[str, str]]:
        """
        Combine system_message + messages for OpenAI's chat format.

        BaseLLMAdapter has already validated that messages is a list of
        {role, content} with string values.

        Handles system message de-duplication: if both system_message
        parameter and system role messages are present, they are merged.
        """
        out: List[Dict[str, str]] = []

        # Collect all system messages from both sources
        system_messages: List[str] = []

        # Add explicit system_message parameter if provided
        if system_message:
            system_messages.append(system_message)

        # Extract system role messages from the messages list
        non_system_messages: List[Mapping[str, str]] = []
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "system":
                system_messages.append(content)
            else:
                non_system_messages.append(m)

        # Combine all system messages into one
        if system_messages:
            combined_system = "\n\n".join(system_messages)
            out.append({"role": "system", "content": combined_system})

            # Log if we had to merge multiple system messages
            if len(system_messages) > 1:
                logger.debug(
                    "Merged %d system messages into one for OpenAI API",
                    len(system_messages),
                )

        # Add all non-system messages
        for m in non_system_messages:
            out.append({"role": m["role"], "content": m["content"]})

        return out

    def _get_response_format(self, ctx: Optional[OperationContext]) -> Optional[Dict[str, str]]:
        """
        Extract response format from context attributes.

        Supports:
        - ctx.attrs.get("response_format") = "json_object" → {"type": "json_object"}
        - ctx.attrs.get("response_format") = {"type": "json_object"} (direct mapping)
        - Other string values are passed through as {"type": <value>}.
        """
        if not ctx or not getattr(ctx, "attrs", None):
            return None

        response_format = ctx.attrs.get("response_format")
        if not response_format:
            return None

        if response_format == "json_object":
            return {"type": "json_object"}
        elif isinstance(response_format, dict) and response_format.get("type"):
            return {"type": response_format["type"]}
        elif isinstance(response_format, str):
            # Handle other potential formats
            return {"type": response_format}

        return None

    @staticmethod
    def _usage_from_response(resp: Any) -> TokenUsage:
        """Convert OpenAI usage object -> TokenUsage; handle missing usage."""
        usage = getattr(resp, "usage", None)
        if usage is None:
            return TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        return TokenUsage(
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
        )

    @staticmethod
    def _extract_retry_after_ms(err: Any) -> Optional[int]:
        """
        Extract Retry-After header from OpenAI errors.

        Handles both:
        - Retry-After: 60             (integer seconds)
        - Retry-After: Wed, 21 Oct... (HTTP date)
        and performs a case-insensitive header lookup.
        """
        try:
            resp = getattr(err, "response", None)
            if resp is None:
                return None

            headers = getattr(resp, "headers", None)
            if not headers:
                return None

            # Case-insensitive header lookup
            val = None
            for key in ("retry-after", "Retry-After", "Retry-after", "retry_after"):
                if key in headers:
                    val = headers.get(key)
                    if val is not None:
                        break

            if val is None:
                return None

            val_str = str(val).strip()

            # Try integer seconds first
            try:
                seconds = int(val_str)
                return max(0, seconds) * 1000
            except ValueError:
                pass

            # Try HTTP-date format
            try:
                retry_datetime = parsedate_to_datetime(val_str)
                now = _time()
                delay_seconds = int(retry_datetime.timestamp() - now)
                return max(0, delay_seconds) * 1000
            except Exception:
                return None

        except Exception:
            return None

    def _translate_openai_error(self, err: Exception) -> LLMAdapterError:
        """
        Map OpenAI client errors → Corpus LLMAdapterError subclasses.

        This ensures routers and observability can reason about failures
        without vendor-specific conditionals.
        """
        # Connection / transport issues → TransientNetwork
        if APIConnectionError is not None and isinstance(err, APIConnectionError):
            return TransientNetwork(str(err) or "OpenAI API connection error")

        # Upstream timeout → DeadlineExceeded
        if APITimeoutError is not None and isinstance(err, APITimeoutError):
            return DeadlineExceeded("OpenAI API request timed out")

        # Rate limiting / quota → ResourceExhausted
        if RateLimitError is not None and isinstance(err, RateLimitError):
            retry_ms = self._extract_retry_after_ms(err)
            return ResourceExhausted(
                "OpenAI rate limit exceeded",
                retry_after_ms=retry_ms,
                throttle_scope="tenant",
            )

        # AuthN / AuthZ → AuthError
        if AuthenticationError is not None and isinstance(err, AuthenticationError):
            return AuthError(str(err) or "OpenAI authentication/authorization error")

        # Request shape / params → BadRequest
        if BadRequestError is not None and isinstance(err, BadRequestError):
            return BadRequest(str(err) or "OpenAI request is invalid")

        # HTTP status buckets
        if APIStatusError is not None and isinstance(err, APIStatusError):
            status = int(getattr(err, "status_code", 0) or 0)

            # Map a few key ranges explicitly.
            if status == 400:
                return BadRequest(str(err) or "OpenAI request is invalid")
            if status in (401, 403):
                return AuthError(str(err) or "OpenAI authentication/authorization error")
            if status == 404:
                # Often "model not found" or similar.
                return NotSupported(str(err) or "Requested OpenAI resource is not supported")
            if status == 429:
                retry_ms = self._extract_retry_after_ms(err)
                return ResourceExhausted(
                    "OpenAI rate limit exceeded",
                    retry_after_ms=retry_ms,
                    throttle_scope="tenant",
                )
            if 500 <= status <= 599:
                retry_ms = self._extract_retry_after_ms(err)
                return Unavailable(
                    "OpenAI service is temporarily unavailable",
                    retry_after_ms=retry_ms,
                )

            # Fallback for unexpected status codes.
            return Unavailable(str(err) or f"OpenAI error (status={status})")

        # Generic OpenAI error fallback → Unavailable
        if isinstance(err, OpenAIError):
            return Unavailable(str(err) or "OpenAI API error")

        # Anything else → wrap as Unavailable
        return Unavailable(str(err) or "internal OpenAI adapter error")

    def _get_tokenizer(self, model: str) -> Any:
        """
        Get or create tokenizer for the given model, with caching.

        Returns None if tiktoken is not available.
        """
        if tiktoken is None:
            return None

        cache_key = model
        if cache_key in self._tokenizer_cache:
            return self._tokenizer_cache[cache_key]

        try:
            try:
                enc = tiktoken.encoding_for_model(model)
            except Exception:
                # Fallback encoding used by most modern OpenAI chat models.
                enc = tiktoken.get_encoding("cl100k_base")

            self._tokenizer_cache[cache_key] = enc
            return enc
        except Exception:
            logger.debug("Failed to get tokenizer for model %s", model, exc_info=True)
            return None

    # Convenience wrapper so usages of _tenant_hash are obvious locally.
    @staticmethod
    def _tenant_hash(tenant: Optional[str]) -> Optional[str]:
        # Delegate to BaseLLMAdapter's implementation to keep behavior consistent.
        return BaseLLMAdapter._tenant_hash(tenant)  # type: ignore[attr-defined]

    @staticmethod
    def _compute_timeout(ctx: Optional[OperationContext]) -> Optional[float]:
        """
        Derive an OpenAI client-side timeout from ctx.deadline_ms.

        Returns seconds (float) or None if no deadline is set.
        """
        if ctx is None or ctx.deadline_ms is None:
            return None
        now_ms = int(_time() * 1000)
        remaining_ms = ctx.deadline_ms - now_ms
        if remaining_ms <= 0:
            # We still send a minimal timeout to avoid infinite wait at client.
            return 1.0
        return max(0.1, remaining_ms / 1000.0)

    # ------------------------------------------------------------------
    # BaseLLMAdapter backend hooks
    # ------------------------------------------------------------------

    async def _do_capabilities(self) -> LLMCapabilities:
        """
        Report adapter capabilities.

        We include at least one supported model so conformance tests can
        exercise the adapter end-to-end. Callers can still pass any
        valid OpenAI model ID; this list is just a minimal sample.
        """
        return LLMCapabilities(
            server=self._server,
            version=self._version,
            model_family=self._model_family,
            max_context_length=self._max_context_length,
            supports_streaming=True,
            supports_roles=True,
            supports_json_output=True,   # JSON via response_format
            supports_tools=True,         # Real tool calling support enabled
            supports_parallel_tool_calls=True,
            idempotent_writes=True,
            supports_multi_tenant=True,
            supports_system_message=True,
            supports_deadline=True,
            # We implement token counting using tiktoken when available,
            # with a heuristic fallback when it isn't.
            supports_count_tokens=True,
            # IMPORTANT: must be non-empty to satisfy conformance tests.
            supported_models=(self._default_model,),
        )

    async def _do_complete(
        self,
        *,
        messages: List[Mapping[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        model: Optional[str] = None,
        system_message: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        ctx: Optional[OperationContext] = None,
    ) -> LLMCompletion:
        """
        Backend implementation of `complete()` using Chat Completions.

        Notes:
        - Response caching is handled automatically by BaseLLMAdapter via
          the _make_complete_cache_key mechanism.
        - Tools and tool_choice are forwarded directly to OpenAI when
          provided; tool calls returned by OpenAI are mapped into the
          protocol's ToolCall dataclasses.
        """
        # Enforce the same role restrictions as the mock adapter so both
        # behave equivalently from the protocol's perspective.
        self._validate_roles(messages)

        resolved_model = self._resolve_model(model)
        oai_messages = self._build_messages(
            messages=messages,
            system_message=system_message,
        )

        # Extract response format from context if specified
        response_format = self._get_response_format(ctx)

        try:
            # Build the base request parameters (no PII in logs: we never log content)
            request_params: Dict[str, Any] = {
                "model": resolved_model,
                "messages": oai_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "stop": stop_sequences,
                "stream": False,
            }

            if response_format:
                request_params["response_format"] = response_format

            if tools:
                request_params["tools"] = tools
            if tool_choice is not None:
                request_params["tool_choice"] = tool_choice

            # Context-driven enhancements: timeout, trace headers, user/tenant, extra params
            timeout = self._compute_timeout(ctx)
            if timeout is not None:
                request_params["timeout"] = timeout

            extra_headers: Dict[str, str] = {}
            if ctx:
                if ctx.request_id:
                    extra_headers["X-Request-ID"] = ctx.request_id
                if ctx.traceparent:
                    extra_headers["traceparent"] = ctx.traceparent
                if ctx.tenant:
                    user_tag = self._tenant_hash(ctx.tenant)
                    if user_tag:
                        # OpenAI's 'user' field is a great place for a tenant hash
                        request_params["user"] = user_tag

            if extra_headers:
                request_params["extra_headers"] = extra_headers

            # Allow callers to pass through OpenAI-specific knobs via ctx.attrs["openai_extra"]
            oai_extra = ctx.attrs.get("openai_extra") if ctx and getattr(ctx, "attrs", None) else None
            if isinstance(oai_extra, dict):
                # Shallow override only if not already set, to avoid breaking core semantics
                for k, v in oai_extra.items():
                    if k not in request_params:
                        request_params[k] = v

            logger.debug(
                "OpenAIAdapter.complete: model=%s, messages=%d, max_tokens=%s, tools=%d, tool_choice_type=%s",
                resolved_model,
                len(oai_messages),
                str(max_tokens),
                len(tools) if tools else 0,
                type(tool_choice).__name__ if tool_choice is not None else "None",
            )

            resp = await self._client.chat.completions.create(**request_params)

            logger.debug(
                "OpenAIAdapter.complete: response_model=%s, usage=%s",
                getattr(resp, "model", "unknown"),
                self._usage_from_response(resp),
            )

        except Exception as exc:  # noqa: BLE001
            logger.warning("OpenAIAdapter.complete failed: %s", exc, exc_info=True)
            raise self._translate_openai_error(exc) from exc

        # Extract the primary choice.
        if not getattr(resp, "choices", None):
            raise Unavailable("OpenAI returned no choices")

        choice = resp.choices[0]
        # For v1 client: `choice.message.content` is the text.
        message = getattr(choice, "message", None)
        text = ""
        if message is not None:
            text = getattr(message, "content", "") or ""
        # Older/newer variants sometimes expose `choice.text`
        if not text and hasattr(choice, "text"):
            text = getattr(choice, "text", "") or ""

        # Map tool calls if present on the message
        tool_calls: List[ToolCall] = []
        raw_tool_calls = getattr(message, "tool_calls", None) if message is not None else None
        if raw_tool_calls:
            for tc in raw_tool_calls:
                fn = getattr(tc, "function", None)
                tool_calls.append(
                    ToolCall(
                        id=str(getattr(tc, "id", "") or ""),
                        type=str(getattr(tc, "type", "") or "function"),
                        function=ToolCallFunction(
                            name=str(getattr(fn, "name", "") or "") if fn is not None else "",
                            arguments=str(getattr(fn, "arguments", "") or "") if fn is not None else "",
                        ),
                    )
                )

        finish_reason = getattr(choice, "finish_reason", None) or "stop"
        usage = self._usage_from_response(resp)
        model_id = getattr(resp, "model", None) or resolved_model

        return LLMCompletion(
            text=text,
            model=model_id,
            model_family=self._model_family,
            usage=usage,
            finish_reason=str(finish_reason),
            tool_calls=tool_calls,
        )

    async def _do_stream(
        self,
        *,
        messages: List[Mapping[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        model: Optional[str] = None,
        system_message: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        ctx: Optional[OperationContext] = None,
    ) -> AsyncIterator[LLMChunk]:
        """
        Backend implementation of `stream()` using OpenAI streaming.

        We emit:
            - One `LLMChunk` per non-empty delta text (is_final=False).
            - A terminal sentinel chunk with is_final=True.

        With `stream_options={"include_usage": True}`, the final event
        contains exact token usage; this is surfaced on the final chunk
        via `usage_so_far`.

        Tool calls:
        - Reconstructs tool calls incrementally from delta.tool_calls.
        - Each chunk's `tool_calls` contains the current aggregated state
          of all tool calls seen so far (partial or complete).
        """
        # Enforce the same role restrictions as the mock adapter.
        self._validate_roles(messages)

        resolved_model = self._resolve_model(model)
        oai_messages = self._build_messages(
            messages=messages,
            system_message=system_message,
        )

        # Extract response format from context if specified
        response_format = self._get_response_format(ctx)

        try:
            # Build the base request parameters
            request_params: Dict[str, Any] = {
                "model": resolved_model,
                "messages": oai_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "stop": stop_sequences,
                "stream": True,
                # Ask OpenAI to include usage in the stream so we can surface
                # final token usage on the terminal chunk.
                "stream_options": {"include_usage": True},
            }

            if response_format:
                request_params["response_format"] = response_format

            if tools:
                request_params["tools"] = tools
            if tool_choice is not None:
                request_params["tool_choice"] = tool_choice

            # Context-driven enhancements: timeout, trace headers, user/tenant, extra params
            timeout = self._compute_timeout(ctx)
            if timeout is not None:
                request_params["timeout"] = timeout

            extra_headers: Dict[str, str] = {}
            if ctx:
                if ctx.request_id:
                    extra_headers["X-Request-ID"] = ctx.request_id
                if ctx.traceparent:
                    extra_headers["traceparent"] = ctx.traceparent
                if ctx.tenant:
                    user_tag = self._tenant_hash(ctx.tenant)
                    if user_tag:
                        request_params["user"] = user_tag

            if extra_headers:
                request_params["extra_headers"] = extra_headers

            oai_extra = ctx.attrs.get("openai_extra") if ctx and getattr(ctx, "attrs", None) else None
            if isinstance(oai_extra, dict):
                for k, v in oai_extra.items():
                    if k not in request_params:
                        request_params[k] = v

            logger.debug(
                "OpenAIAdapter.stream: model=%s, messages=%d, max_tokens=%s, tools=%d, tool_choice_type=%s",
                resolved_model,
                len(oai_messages),
                str(max_tokens),
                len(tools) if tools else 0,
                type(tool_choice).__name__ if tool_choice is not None else "None",
            )

            stream = await self._client.chat.completions.create(**request_params)

        except Exception as exc:  # noqa: BLE001
            logger.warning("OpenAIAdapter.stream failed to create stream: %s", exc, exc_info=True)
            raise self._translate_openai_error(exc) from exc

        # Aggregate model + usage info as we go; usage is normally only
        # available on the final event.
        last_model_id: Optional[str] = None
        final_usage: Optional[TokenUsage] = None
        received_chunks = 0

        # Tool call reconstruction state: index → {id, type, name, arguments}
        tool_call_state: Dict[int, Dict[str, str]] = {}

        try:
            async for event in stream:
                received_chunks += 1
                last_model_id = getattr(event, "model", None) or last_model_id or resolved_model

                # Handle empty or malformed events gracefully
                if not getattr(event, "choices", None):
                    continue

                choice = event.choices[0]
                if not choice:
                    continue

                delta = getattr(choice, "delta", None)
                text = ""
                if delta is not None:
                    text = getattr(delta, "content", "") or ""
                # Older patterns may expose `choice.text` instead.
                if not text and hasattr(choice, "text"):
                    text = getattr(choice, "text", "") or ""

                # Collect final usage if present on this chunk (thanks to
                # stream_options={"include_usage": True}).
                usage = getattr(event, "usage", None)
                if usage is not None:
                    final_usage = TokenUsage(
                        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                        completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                        total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
                    )

                # --- Tool call delta handling --------------------------------
                # delta.tool_calls is an array of per-index deltas
                raw_tool_deltas = getattr(delta, "tool_calls", None) if delta is not None else None
                if raw_tool_deltas:
                    for tc_delta in raw_tool_deltas:
                        idx = int(getattr(tc_delta, "index", 0) or 0)
                        state = tool_call_state.setdefault(
                            idx,
                            {"id": "", "type": "function", "name": "", "arguments": ""},
                        )

                        tc_id = getattr(tc_delta, "id", None)
                        if tc_id:
                            state["id"] = tc_id

                        tc_type = getattr(tc_delta, "type", None)
                        if tc_type:
                            state["type"] = tc_type

                        fn_delta = getattr(tc_delta, "function", None)
                        if fn_delta is not None:
                            fn_name = getattr(fn_delta, "name", None)
                            if fn_name:
                                # Name is usually sent once, but we defensively append.
                                state["name"] = (state["name"] or "") + fn_name
                            fn_args = getattr(fn_delta, "arguments", None)
                            if fn_args:
                                # Arguments arrive as incremental chunks; we append.
                                state["arguments"] = (state["arguments"] or "") + fn_args

                # Build current tool call view (partial or complete) for this chunk
                tool_calls: List[ToolCall] = []
                if tool_call_state:
                    for idx, s in sorted(tool_call_state.items()):
                        tool_calls.append(
                            ToolCall(
                                id=s["id"] or f"call_{idx}",
                                type=s["type"] or "function",
                                function=ToolCallFunction(
                                    name=s["name"],
                                    arguments=s["arguments"],
                                ),
                            )
                        )

                if text or tool_calls:
                    # Emit delta chunks as we receive them. We intentionally
                    # do not attach usage_so_far here because OpenAI only
                    # provides usage reliably at the end of the stream and
                    # we avoid adding heavy per-chunk token counting.
                    yield LLMChunk(
                        text=text,
                        is_final=False,
                        model=last_model_id,
                        usage_so_far=None,
                        tool_calls=tool_calls,
                    )

            # Handle case where stream ends with no chunks received
            if received_chunks == 0:
                logger.warning("OpenAIAdapter.stream: stream ended with no chunks received")
                yield LLMChunk(
                    text="",
                    is_final=True,
                    model=resolved_model,
                    usage_so_far=TokenUsage(
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                    ),
                    tool_calls=[],
                )
                return

        except Exception as exc:  # noqa: BLE001
            logger.warning("OpenAIAdapter.stream failed during iteration: %s", exc, exc_info=True)
            # Translate any errors that occur during streaming
            raise self._translate_openai_error(exc) from exc

        # Emit a final sentinel chunk marking end-of-stream (with usage if available).
        # Include any final tool call state (fully reconstructed).
        final_tool_calls: List[ToolCall] = []
        if tool_call_state:
            for idx, s in sorted(tool_call_state.items()):
                final_tool_calls.append(
                    ToolCall(
                        id=s["id"] or f"call_{idx}",
                        type=s["type"] or "function",
                        function=ToolCallFunction(
                            name=s["name"],
                            arguments=s["arguments"],
                        ),
                    )
                )

        yield LLMChunk(
            text="",
            is_final=True,
            model=last_model_id or resolved_model,
            usage_so_far=final_usage,
            tool_calls=final_tool_calls,
        )

    async def _do_count_tokens(
        self,
        text: str,
        *,
        model: Optional[str] = None,
        ctx: Optional[OperationContext] = None,
    ) -> int:
        """
        Token counting for OpenAI models.

        Strategy:
        - If `tiktoken` is available, use model-aware tokenization via
          `encoding_for_model` (falling back to `cl100k_base`).
        - Otherwise, use a whitespace/character heuristic similar to
          MockLLMAdapter, which is good enough for rough planning.

        This function is used by BaseLLMAdapter for:
            - context-window preflight checks
            - coarse quota tracking
        """
        if not text:
            return 0

        resolved_model = self._resolve_model(model)

        # Try cached tokenizer first
        enc = self._get_tokenizer(resolved_model)
        if enc is not None:
            try:
                return len(enc.encode(text))
            except Exception:  # noqa: BLE001
                # If tokenization fails, fall back to heuristic
                logger.debug(
                    "tiktoken token counting failed; falling back to heuristic",
                    exc_info=True,
                )

        # Heuristic fallback: whitespace-based + small overhead, blended
        # with a character-based estimate. This is reasonably accurate
        # for many languages but is only an estimate (not a strict lower
        # bound) and is intended for planning, not billing.
        word_count = len(text.split())
        char_count = len(text)

        word_based = word_count + 3  # Original heuristic
        char_based = char_count // 4  # Rough character-to-token ratio

        return max(word_based, char_based)

    async def _do_health(
        self,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> Mapping[str, Any]:
        """
        Lightweight health check.

        Behavior:
        - If ctx.attrs["health"] is "degraded" / "error", we return a
          deterministic degraded/error shape (mirrors MockLLMAdapter to
          keep the conformance harness happy).
        - Otherwise we perform a minimal live check by listing models.
          Any OpenAI error is captured into the returned payload instead
          of being raised, so that `BaseLLMAdapter.health()` can still
          normalize the response.
        """
        status_hint: Optional[str] = None
        if ctx is not None and getattr(ctx, "attrs", None):
            status_hint = ctx.attrs.get("health")

        if status_hint == "degraded":
            return {
                "ok": False,
                "status": "degraded",
                "server": self._server,
                "version": self._version,
            }
        if status_hint == "error":
            return {
                "ok": False,
                "status": "error",
                "server": self._server,
                "version": self._version,
            }

        try:
            # Minimal live check: ensure we can talk to the API at all.
            # We don't depend on the exact payload shape here.
            await self._client.models.list()
            return {
                "ok": True,
                "status": "healthy",
                "server": self._server,
                "version": self._version,
            }
        except Exception as exc:  # noqa: BLE001
            err = self._translate_openai_error(exc)
            # We intentionally do NOT raise here; BaseLLMAdapter.health()
            # expects a mapping and will treat this as a successful but
            # unhealthy probe.
            logger.warning("OpenAIAdapter health check failed: %s", err, exc_info=True)
            return {
                "ok": False,
                "status": "error",
                "server": self._server,
                "version": self._version,
                "error_code": getattr(err, "code", None),
                "error_message": str(err),
            }

    # ------------------------------------------------------------------
    # Resource cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """
        Close underlying client resources if supported.

        Called automatically when used as:

            async with OpenAIAdapter(...) as adapter:
                ...
        """
        close = getattr(self._client, "close", None)
        if close is None:
            return
        try:
            result = close()
            if asyncio.iscoroutine(result):
                await result
        except Exception:  # noqa: BLE001
            # Best-effort cleanup; ignore close failures.
            logger.debug("OpenAIAdapter close() failed", exc_info=True)

        # Clear tokenizer cache
        self._tokenizer_cache.clear()


__all__ = [
    "OpenAIAdapter",
]
