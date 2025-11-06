# adapter_sdk/llm_base.py
# SPDX-License-Identifier: Apache-2.0
"""
Adapter SDK — LLM Protocol V1 (public contract + production-grade base)

Purpose
-------
A stable, vendor-neutral API for calling Large Language Models — with structured errors,
streaming support, token usage accounting, deadline propagation, and SIEM-safe metrics.

This protocol enables seamless integration with any LLM provider while maintaining
production-grade observability, security, and operational rigor.

Design Philosophy
-----------------
- Minimal surface area: Core operations only, no vendor-specific extensions
- Async-first: All operations are non-blocking for high-concurrency environments
- Production hardened: Built-in metrics, error taxonomy, and context propagation
- Extensible: Capability discovery allows for provider-specific features

Deliberate Non-Goals
--------------------
- No retries, hedging, model selection, routing, fallback, or policy enforcement.
- No tokenizer transforms or tool-calling orchestration.
- No vendor-specific helpers or SDK wrappers.
- No client-side auto-stream reassembly.

Those behaviors live in the **Corpus Router** and upper control-plane layers.

Mode Strategy (Composition vs. Standalone)
------------------------------------------
mode: "thin" (default) - For composition with external providers/managers. All policies
      are no-op. Use this when you have your own scheduling/caching/rate limiting/etc.

mode: "standalone" - For direct use. Enables basic deadline enforcement, circuit breaking,
      and in-memory caching + token-bucket rate limiting. Suitable for development and
      light production use. Not a replacement for a full external provider.

Versioning
----------
Follow SemVer against LLM_PROTOCOL_VERSION. Minor versions are strictly additive.
- Patch (x.y.Z): Editorial clarifications, non-breaking fixes
- Minor (x.Y.z): New optional parameters, capabilities, or methods
- Major (X.y.z): Breaking changes to signatures or behavior

Wire Contract (Canonical Interface)
-----------------------------------
The canonical interoperability surface for this protocol is the JSON wire envelope.
This module defines a code-level interface (LLMProtocolV1 / BaseLLMAdapter) plus a
thin wire adapter (WireLLMHandler) that maps envelopes ⇄ typed methods.

All requests MUST use the following envelope shape:

    {
        "op": "llm.<operation>",
        "ctx": {
            "request_id": "...",
            "idempotency_key": "...",
            "deadline_ms": 1234567890,
            "traceparent": "...",
            "tenant": "...",
            "attrs": { ... }
        },
        "args": { ... }  # operation-specific
    }

Unary Responses (success):

    {
        "ok": true,
        "code": "OK",
        "ms": <float>,          # elapsed milliseconds (best-effort)
        "result": { ... }       # operation-specific payload
    }

Unary Responses (error):

    {
        "ok": false,
        "code": "<UPPER_SNAKE_CASE>",   # e.g. BAD_REQUEST, AUTH_ERROR, UNAVAILABLE
        "error": "<ErrorClassName>",    # e.g. BadRequest
        "message": "<human readable>",
        "retry_after_ms": <int|null>,
        "details": { ... } | null,
        "ms": <float>
    }

Streaming (llm.stream):

Request:

    {
        "op": "llm.stream",
        "ctx": { ... },
        "args": {
            "messages": [ ... ],
            "max_tokens": <int|null>,
            "temperature": <float|null>,
            "model": "<model-id>|null",
            "system_message": "<str>|null"
        }
    }

Stream Responses:
    - Zero or more chunk envelopes:

        {
            "ok": true,
            "code": "OK",
            "ms": <float>,
            "chunk": {
                "text": "<partial>",
                "is_final": false,
                "model": "<model-id>|null",
                "usage_so_far": {
                    "prompt_tokens": <int>,
                    "completion_tokens": <int>,
                    "total_tokens": <int>
                } | null
            }
        }

    - On terminal success, last chunk SHOULD have "is_final": true.
    - On error, a single error envelope (same shape as unary error) terminates the stream.

The WireLLMHandler in this file is the reference adapter for this contract and is
intentionally transport-agnostic (HTTP, gRPC, WebSocket, etc.).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import time
from dataclasses import dataclass, asdict
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

LOG = logging.getLogger(__name__)

# Minor bump: additive fields in LLMCapabilities and new error type.
LLM_PROTOCOL_VERSION = "1.0.0"
LLM_PROTOCOL_ID = "llm/v1.0"

# =============================================================================
# Normalized Errors (with retry hints and operational guidance)
# =============================================================================

class LLMAdapterError(Exception):
    """
    Base exception for all LLM adapter errors.

    Provides structured error information including retry guidance, throttling context,
    and operational suggestions for callers to handle failures gracefully.

    Attributes:
        message: Human-readable error description
        code: Machine-readable error code for programmatic handling
        retry_after_ms: Suggested delay before retry (None if not retryable)
        throttle_scope: Scope of throttling ("tenant", "model", "global", etc.)
        suggested_token_reduction: Percentage reduction suggestion for quota errors
        details: Additional context-specific error details (JSON-serializable)
    """
    def __init__(
        self,
        message: str = "",
        *,
        code: Optional[str] = None,
        retry_after_ms: Optional[int] = None,
        throttle_scope: Optional[str] = None,   # "tenant", "cluster", "model", etc.
        suggested_token_reduction: Optional[int] = None,  # percent (0–100)
        details: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.retry_after_ms = retry_after_ms
        self.throttle_scope = throttle_scope
        self.suggested_token_reduction = suggested_token_reduction
        self.details = dict(details or {})

    def __str__(self) -> str:
        base = self.message or self.__class__.__name__
        if self.code:
            base += f" [code={self.code}]"
        if self.retry_after_ms is not None:
            base += f" retry_after_ms={self.retry_after_ms}"
        if self.throttle_scope:
            base += f" throttle_scope={self.throttle_scope}"
        if self.suggested_token_reduction is not None:
            base += f" suggested_token_reduction={self.suggested_token_reduction}%"
        if self.details:
            base += f" details={self.details}"
        return base


class BadRequest(LLMAdapterError):
    """Client sent an invalid request (malformed messages, invalid parameters)."""
    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("code", "BAD_REQUEST")
        super().__init__(message, **kwargs)


class AuthError(LLMAdapterError):
    """Authentication or authorization failed (invalid credentials, permissions)."""
    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("code", "AUTH_ERROR")
        super().__init__(message, **kwargs)


class ResourceExhausted(LLMAdapterError):
    """Quota, rate limit, or resource constraints exceeded."""
    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("code", "RESOURCE_EXHAUSTED")
        super().__init__(message, **kwargs)


class TransientNetwork(LLMAdapterError):
    """Transient network failure that may succeed on retry."""
    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("code", "TRANSIENT_NETWORK")
        super().__init__(message, **kwargs)


class Unavailable(LLMAdapterError):
    """Service is temporarily unavailable or overloaded."""
    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("code", "UNAVAILABLE")
        super().__init__(message, **kwargs)


class NotSupported(LLMAdapterError):
    """Requested operation or parameter is not supported by this adapter."""
    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("code", "NOT_SUPPORTED")
        super().__init__(message, **kwargs)


class ModelOverloaded(LLMAdapterError):
    """Specific model is currently overloaded and cannot handle requests."""
    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("code", "MODEL_OVERLOADED")
        super().__init__(message, **kwargs)


class DeadlineExceeded(LLMAdapterError):
    """Operation exceeded ctx.deadline_ms budget."""
    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("code", "DEADLINE_EXCEEDED")
        super().__init__(message, **kwargs)


# =============================================================================
# Context (used for deadlines, identity, SIEM-safe metrics)
# =============================================================================

@dataclass(frozen=True)
class OperationContext:
    """
    Context for LLM operations providing tracing, deadlines, and multi-tenant isolation.

    All context information is propagated through the call chain and used for
    observability, security, and operational control without exposing sensitive data.

    Attributes:
        request_id: Unique identifier for the request chain (correlation ID)
        idempotency_key: Key for ensuring idempotent operations (when supported)
        deadline_ms: Absolute epoch milliseconds when operation should timeout
        traceparent: W3C Trace Context header for distributed tracing
        tenant: Multi-tenant isolation scope (NEVER logged or exposed in metrics)
        attrs: Additional operation attributes for extensibility and middleware
    """
    request_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    deadline_ms: Optional[int] = None  # absolute epoch ms
    traceparent: Optional[str] = None
    tenant: Optional[str] = None       # NEVER log raw - hash only in metrics
    attrs: Mapping[str, Any] = None

    def __post_init__(self) -> None:
        """Ensure attrs is always a valid dictionary."""
        if self.attrs is None:
            object.__setattr__(self, "attrs", {})


# =============================================================================
# Metrics Interface (SIEM-safe, low-cardinality)
# =============================================================================

class MetricsSink(Protocol):
    """
    Protocol for metrics collection implementations.

    Used for operational monitoring without exposing sensitive information.
    All metrics must be low-cardinality and never include PII or tenant identifiers.
    """
    def observe(
        self,
        *,
        component: str,
        op: str,
        ms: float,
        ok: bool,
        code: str = "OK",
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Record operation timing and status.

        Args:
            component: Component name (e.g., "llm")
            op: Operation name (e.g., "complete", "stream")
            ms: Operation duration in milliseconds
            ok: Whether operation succeeded
            code: Status code (error class name or "OK")
            extra: Additional low-cardinality dimensions
        """
        ...

    def counter(
        self,
        *,
        component: str,
        name: str,
        value: int = 1,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Increment a counter metric.

        Args:
            component: Component name (e.g., "llm")
            name: Counter name (e.g., "requests", "tokens")
            value: Increment value
            extra: Additional low-cardinality dimensions
        """
        ...


class NoopMetrics:
    """No-operation metrics sink for testing or when metrics are disabled."""
    def observe(self, **_: Any) -> None: ...
    def counter(self, **_: Any) -> None: ...


# =============================================================================
# Result Models (structured, typed responses)
# =============================================================================

@dataclass
class TokenUsage:
    """
    Token usage accounting for cost tracking and quota management.

    Attributes:
        prompt_tokens: Number of tokens in the input prompt
        completion_tokens: Number of tokens in the generated output
        total_tokens: Sum of prompt and completion tokens
    """
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class LLMCompletion:
    """
    Complete LLM response with metadata and token accounting.

    Attributes:
        text: Generated text content
        model: Specific model identifier used for generation
        model_family: Model family for routing and analytics ("gpt-4", "claude-3", etc.)
        usage: Token usage breakdown for cost tracking
        finish_reason: Reason for generation stopping ("stop", "length", "error", "tool_call")
    """
    text: str
    model: str
    model_family: str  # "gpt-4", "claude-3", "gemini-pro"
    usage: TokenUsage
    finish_reason: str  # "stop", "length", "error", "tool_call"


@dataclass
class LLMChunk:
    """
    Streaming response chunk for real-time output.

    Attributes:
        text: Partial generated text content
        is_final: Whether this is the final chunk in the stream
        model: Model identifier (may be None until final chunk)
        usage_so_far: Progressive token usage (may be None until final chunk)
    """
    text: str
    is_final: bool = False
    model: Optional[str] = None
    usage_so_far: Optional[TokenUsage] = None  # Progressive token counts


# =============================================================================
# Pluggable policy & infra interfaces (deadline, breaker, cache, limiter)
# =============================================================================

class DeadlinePolicy(Protocol):
    """Strategy to apply time budgets (ctx.deadline_ms) to awaits."""
    async def wrap(self, awaitable, ctx: Optional[OperationContext]) -> Any: ...


class CircuitBreaker(Protocol):
    """Minimal circuit breaker interface."""
    def allow(self) -> bool: ...
    def on_success(self) -> None: ...
    def on_error(self, err: Exception) -> None: ...


class Cache(Protocol):
    """Minimal async cache interface (used for non-streaming .complete())."""
    async def get(self, key: str) -> Optional[Any]: ...
    async def set(self, key: str, value: Any, ttl_s: int) -> None: ...


class RateLimiter(Protocol):
    """Minimal rate limiter interface."""
    async def acquire(self) -> None: ...
    def release(self) -> None: ...


class NoopDeadline:
    """No-op deadline policy (no timing/timeout behavior)."""
    async def wrap(self, awaitable, ctx: Optional[OperationContext]) -> Any:
        return await awaitable


class SimpleDeadline:
    """
    Deadline policy that enforces ctx.deadline_ms using asyncio.wait_for.
    If ctx.deadline_ms is None, it passes through without timeout.
    """
    async def wrap(self, awaitable, ctx: Optional[OperationContext]) -> Any:
        if ctx is None or ctx.deadline_ms is None:
            return await awaitable
        now_ms = int(time.time() * 1000)
        remaining_ms = max(0, ctx.deadline_ms - now_ms)
        if remaining_ms <= 0:
            raise DeadlineExceeded(
                "deadline already exceeded",
                details={"remaining_ms": 0},
            )
        try:
            return await asyncio.wait_for(awaitable, timeout=remaining_ms / 1000.0)
        except asyncio.TimeoutError as e:
            raise DeadlineExceeded(
                "operation timed out",
                details={"remaining_ms": 0},
            ) from e


class NoopBreaker:
    def allow(self) -> bool: return True
    def on_success(self) -> None: ...
    def on_error(self, err: Exception) -> None: ...


class SimpleCircuitBreaker:
    """
    Extremely small circuit breaker (counts consecutive failures).
    Not distributed; intended for standalone/dev use only.
    """
    def __init__(self, *, failure_threshold: int = 5, recovery_after_s: float = 10.0) -> None:
        self._failure_threshold = max(1, int(failure_threshold))
        self._recovery_after_s = max(0.1, float(recovery_after_s))
        self._failures = 0
        self._opened_at: Optional[float] = None

    def allow(self) -> bool:
        if self._opened_at is None:
            return True
        # Half-open if recovery window elapsed: allow a trial request.
        if (time.monotonic() - self._opened_at) >= self._recovery_after_s:
            return True
        return False

    def on_success(self) -> None:
        # Reset on success (closes breaker if it was open/half-open)
        self._failures = 0
        self._opened_at = None

    def on_error(self, _err: Exception) -> None:
        # Count consecutive failures and open when threshold exceeded.
        self._failures += 1
        if self._failures >= self._failure_threshold:
            self._opened_at = time.monotonic()


class NoopCache:
    """No-op cache used in thin/composed mode."""
    async def get(self, key: str) -> Optional[Any]:
        return None
    async def set(self, key: str, value: Any, ttl_s: int) -> None:
        return None


class InMemoryTTLCache:
    """
    Simple in-memory TTL cache for .complete() responses.

    Not multi-process safe; not distributed; suitable for standalone/dev only.
    Eviction is opportunistic; callers MUST NOT rely on strong cache semantics.
    """
    def __init__(self) -> None:
        self._store: Dict[str, Tuple[float, Any]] = {}

    async def get(self, key: str) -> Optional[Any]:
        now = time.monotonic()
        item = self._store.get(key)
        if not item:
            return None
        exp, val = item
        if now >= exp:
            # expire lazily
            try:
                del self._store[key]
            except Exception:
                pass
            return None
        return val

    async def set(self, key: str, value: Any, ttl_s: int) -> None:
        exp = time.monotonic() + max(1, int(ttl_s))
        self._store[key] = (exp, value)
        # Opportunistic pruning to avoid unbounded growth.
        if len(self._store) > 4096:
            try:
                for k, (e, _) in list(self._store.items())[:2048]:
                    if time.monotonic() >= e:
                        del self._store[k]
            except Exception:
                pass


class NoopLimiter:
    """No-op limiter used in thin/composed mode."""
    async def acquire(self) -> None:
        return None
    def release(self) -> None:
        return None


class TokenBucketLimiter:
    """
    Simple token bucket limiter (per-process).

    NOTE: This is intentionally simple and only suitable for dev/small scale.
    """
    def __init__(self, *, rate: float = 50.0, capacity: int = 100) -> None:
        self._rate = float(rate)
        self._capacity = max(1, int(capacity))
        self._tokens = float(capacity)
        self._last = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        delta = now - self._last
        self._last = now
        self._tokens = min(self._capacity, self._tokens + delta * self._rate)

    async def acquire(self) -> None:
        # Busy-wait with small sleeps; OK for dev/standalone.
        while True:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            await asyncio.sleep(0.01)

    def release(self) -> None:
        # Classic token bucket: charge on acquire only.
        return None


# =============================================================================
# Capabilities (dynamic discovery for routing and planning)
# =============================================================================

@dataclass(frozen=True)
class LLMCapabilities:
    """
    Describes the capabilities and limitations of an LLM adapter implementation.

    Used by routing layers for intelligent model selection, request planning,
    and feature compatibility checking.

    Attributes:
        server: Backend server identifier (e.g., "openai", "anthropic", "local-llm")
        version: Adapter or backend version string
        model_family: Primary model family supported ("gpt-4", "claude-3", etc.)
        max_context_length: Maximum context window in tokens
        supports_streaming: Whether streaming responses are supported
        supports_roles: Whether role-based message formatting is supported
        supports_json_output: Whether JSON-structured output is supported
        supports_parallel_tool_calls: Whether parallel function/tool calls are supported
        idempotent_writes: Whether operations are idempotent with idempotency_key
        supports_multi_tenant: Whether multi-tenant isolation is supported
        supports_system_message: Whether explicit system messages are supported

        # Additive, optional fields (alignment with Embedding base parity)
        supports_deadline: Whether adapter cooperates with deadline cancellation
        supports_count_tokens: Whether count_tokens is supported/accurate
        supported_models: Optional list of specific models; empty means "adapter-defined"
    """
    server: str
    version: str
    model_family: str
    max_context_length: int
    supports_streaming: bool = True
    supports_roles: bool = True
    supports_json_output: bool = False
    supports_parallel_tool_calls: bool = False
    idempotent_writes: bool = False
    supports_multi_tenant: bool = False
    supports_system_message: bool = True
    supports_deadline: bool = True
    supports_count_tokens: bool = True
    supported_models: Tuple[str, ...] = ()  # empty ⇒ not enumerated / open set


# =============================================================================
# Stable Protocol Interface (async, versioned contract)
# =============================================================================

@runtime_checkable
class LLMProtocolV1(Protocol):
    """
    Protocol defining the LLM Protocol V1 interface.

    Implement this protocol to create compatible LLM adapters. All methods are async
    and designed for high-concurrency environments. The protocol is runtime-checkable
    for dynamic adapter validation.
    """

    async def capabilities(self) -> LLMCapabilities:
        """
        Get the capabilities of this LLM adapter.

        Returns:
            LLMCapabilities: Description of supported features and limitations

        Note:
            This method is async to support dynamic capability discovery in
            distributed systems where capabilities may change or require
            network calls to determine.
        """
        ...

    async def complete(
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
        ctx: Optional[OperationContext] = None,
    ) -> LLMCompletion:
        """
        Execute a complete LLM conversation and return the full response.

        Args:
            messages: Conversation history as list of role-content mappings.
                     Each message must have "role" and "content" keys.
            max_tokens: Maximum tokens to generate (None for model default)
            temperature: Sampling temperature (0.0 to 2.0, None for default)
            top_p: Nucleus sampling parameter (None for default)
            frequency_penalty: Frequency penalty (-2.0 to 2.0, None for default)
            presence_penalty: Presence penalty (-2.0 to 2.0, None for default)
            stop_sequences: Sequences that will stop generation (None for default)
            model: Specific model to use (None for default or adapter-chosen)
            system_message: Optional system message for conversation context
            ctx: Operation context for tracing, deadlines, and multi-tenancy

        Returns:
            LLMCompletion: Complete response with text, metadata, and token usage

        Raises:
            BadRequest, AuthError, ResourceExhausted, ModelOverloaded,
            TransientNetwork, Unavailable, NotSupported
        """
        ...

    async def stream(
        self,
        *,
        messages: List[Mapping[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        system_message: Optional[str] = None,
        ctx: Optional[OperationContext] = None,
    ) -> AsyncIterator[LLMChunk]:
        """
        Execute an LLM conversation and stream results as they are generated.

        Args:
            messages: Conversation history as list of role-content mappings
            max_tokens: Maximum tokens to generate (None for model default)
            temperature: Sampling temperature (0.0 to 2.0, None for default)
            model: Specific model to use (None for default or adapter-chosen)
            system_message: Optional system message for conversation context
            ctx: Operation context for tracing, deadlines, and multi-tenancy

        Yields:
            LLMChunk: Stream chunks with partial text and optional metadata

        Raises:
            BadRequest, AuthError, ResourceExhausted, ModelOverloaded,
            TransientNetwork, Unavailable, NotSupported
        """
        ...

    async def count_tokens(
        self,
        text: str,
        *,
        model: Optional[str] = None,
        ctx: Optional[OperationContext] = None,
    ) -> int:
        """
        Count the number of tokens in the given text for the specified model.

        Args:
            text: Input text to count tokens for
            model: Specific model to use for tokenization (None for default)
            ctx: Operation context for tracing and multi-tenancy

        Returns:
            int: Number of tokens in the text according to model's tokenizer

        Raises:
            BadRequest: For invalid text or model
            NotSupported: If token counting is not supported
            AuthError: For authentication failures
        """
        ...

    async def health(self, *, ctx: Optional[OperationContext] = None) -> Mapping[str, Any]:
        """
        Check the health status of the LLM backend.

        Args:
            ctx: Operation context for tracing and multi-tenancy

        Returns:
            Mapping with health information including:
            - ok: Boolean overall health status
            - server: Backend server identifier
            - version: Backend version information

        Raises:
            Unavailable: If the health check fails
        """
        ...


# =============================================================================
# Base Instrumented Adapter (validation, metrics, error handling)
# =============================================================================

class BaseLLMAdapter(LLMProtocolV1):
    """
    Base class for implementing LLM Protocol V1 adapters.

    Provides common validation, metrics instrumentation, error handling, and
    SIEM-safe observability. Implementers should override the `_do_*` methods
    to provide backend-specific functionality.

    This base:
      - Normalizes errors into LLMAdapterError subclasses
      - Enforces simple sampling and message validation
      - Applies deadline policy if configured
      - Integrates with pluggable breaker/cache/limiter
      - Exposes a canonical, testable surface for router/control-plane use
    """

    _component = "llm"

    def __init__(
        self,
        *,
        metrics: Optional[MetricsSink] = None,
        mode: str = "thin",
        # optional pluggable policies/infra
        deadline_policy: Optional[DeadlinePolicy] = None,
        breaker: Optional[CircuitBreaker] = None,
        cache: Optional[Cache] = None,
        limiter: Optional[RateLimiter] = None,
        tag_model_in_metrics: bool = True,
        cache_ttl_s: int = 60,
    ) -> None:
        """
        Initialize the LLM adapter with metrics instrumentation and optional policies.

        Args:
            metrics: Metrics sink for operational monitoring. Uses NoopMetrics if None.
            mode: "thin" (default) for composition (all no-op hooks); "standalone" enables
                  basic deadline enforcement, a small circuit breaker, in-memory caching,
                  and a token-bucket rate limiter.
            deadline_policy: Optional deadline policy to enforce ctx.deadline_ms.
            breaker: Optional circuit breaker; defaults based on mode.
            cache: Optional async cache (used for complete only); defaults based on mode.
            limiter: Optional rate limiter; defaults based on mode.
            tag_model_in_metrics: Whether to include 'model' as a metric tag when available.
            cache_ttl_s: Default TTL for in-memory cache entries when enabled.
        """
        self._metrics: MetricsSink = metrics or NoopMetrics()
        self._tag_model_in_metrics: bool = bool(tag_model_in_metrics)
        self._cache_ttl_s: int = max(1, int(cache_ttl_s))

        m = (mode or "thin").strip().lower()
        if m not in {"thin", "standalone"}:
            m = "thin"
        self._mode = m

        # Instantiate default policies/infra based on mode (explicit args win)
        if self._mode == "standalone":
            # Advisory warning if metrics missing
            if metrics is None:
                LOG.warning(
                    "Using standalone mode without metrics - "
                    "consider providing a metrics sink for production use"
                )

            self._deadline: DeadlinePolicy = deadline_policy or SimpleDeadline()
            self._breaker: CircuitBreaker = breaker or SimpleCircuitBreaker()
            self._cache: Cache = cache or InMemoryTTLCache()
            self._limiter: RateLimiter = limiter or TokenBucketLimiter()
        else:
            # thin/composed: all infra is effectively no-op by default
            self._deadline = deadline_policy or NoopDeadline()
            self._breaker = breaker or NoopBreaker()
            self._cache = cache or NoopCache()
            self._limiter = limiter or NoopLimiter()

    # --- internal helpers (validation and instrumentation) ---

    @staticmethod
    def _validate_messages(messages: List[Mapping[str, str]]) -> None:
        """
        Validate that messages list conforms to required format.

        Requirements:
            - Non-empty list
            - Each item has "role" and "content" keys
        """
        if (
            not messages
            or not all(isinstance(m, Mapping) and "role" in m and "content" in m for m in messages)
        ):
            raise BadRequest("messages must be a non-empty list of {role, content} mappings")

    @staticmethod
    def _validate_sampling_params(
        *,
        temperature: Optional[float],
        top_p: Optional[float],
        frequency_penalty: Optional[float],
        presence_penalty: Optional[float],
    ) -> None:
        """
        Validate common sampling parameters within safe ranges.

        This is deliberately conservative to prevent accidental production misconfig.
        """
        if temperature is not None and not (0.0 <= temperature <= 2.0):
            raise BadRequest("temperature must be within [0.0, 2.0]")
        if top_p is not None and not (0.0 < top_p <= 1.0):
            raise BadRequest("top_p must be within (0.0, 1.0]")
        if frequency_penalty is not None and not (-2.0 <= frequency_penalty <= 2.0):
            raise BadRequest("frequency_penalty must be within [-2.0, 2.0]")
        if presence_penalty is not None and not (-2.0 <= presence_penalty <= 2.0):
            raise BadRequest("presence_penalty must be within [-2.0, 2.0]")

    @staticmethod
    def _tenant_hash(t: Optional[str]) -> Optional[str]:
        """
        Create privacy-preserving hash of tenant identifier for metrics.

        Raw tenant identifiers MUST NEVER be emitted to logs/metrics.
        """
        if not t:
            return None
        return hashlib.sha256(t.encode()).hexdigest()[:12]

    def _record(
        self,
        op: str,
        t0: float,
        ok: bool,
        *,
        code: str = "OK",
        ctx: Optional[OperationContext] = None,
        **extra: Any,
    ) -> None:
        """
        Record operation metrics with context and tenant hashing.

        Never exposes raw tenant identifiers in metrics. Safe for SIEM systems.
        """
        try:
            ms = (time.monotonic() - t0) * 1000.0
            x = dict(extra or {})
            if ctx:
                x["tenant"] = self._tenant_hash(ctx.tenant)
            self._metrics.observe(
                component=self._component,
                op=op,
                ms=ms,
                ok=ok,
                code=code,
                extra=x or None,
            )
        except Exception:
            # Metrics failures MUST NOT impact the main control path.
            pass

    def _preflight_deadline(self, ctx: Optional[OperationContext]) -> None:
        """
        Fail fast if ctx.deadline_ms already expired.

        This avoids burning capacity on obviously doomed work.
        """
        if ctx and ctx.deadline_ms is not None:
            now_ms = int(time.time() * 1000)
            if now_ms >= ctx.deadline_ms:
                raise DeadlineExceeded(
                    "deadline already exceeded",
                    details={"remaining_ms": 0},
                )

    @staticmethod
    def _hash_str(s: Optional[str]) -> str:
        """Stable hash helper for cache keys."""
        if s is None:
            return "none"
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    @staticmethod
    def _messages_fingerprint(messages: List[Mapping[str, str]]) -> str:
        """
        Compute a stable digest over role/content pairs (order matters).

        Used for cache key construction; MUST NOT leak raw content.
        """
        h = hashlib.sha256()
        for m in messages:
            role = str(m.get("role", ""))
            content = str(m.get("content", ""))
            h.update(role.encode("utf-8"))
            h.update(b"\x1f")
            h.update(content.encode("utf-8"))
            h.update(b"\x1e")
        return h.hexdigest()

    def _make_complete_cache_key(
        self,
        *,
        model: Optional[str],
        system_message: Optional[str],
        messages: List[Mapping[str, str]],
        max_tokens: Optional[int],
        temperature: Optional[float],
        top_p: Optional[float],
        frequency_penalty: Optional[float],
        presence_penalty: Optional[float],
        stop_sequences: Optional[List[str]],
        caps: LLMCapabilities,
    ) -> str:
        """
        Construct a cache key for complete().

        Includes model, system message hash, message fingerprint, and sampling params
        to avoid cross-polluting responses across materially different requests.
        """
        parts = {
            "model": str(model or "default"),
            "system_hash": self._hash_str(system_message),
            "msgs": self._messages_fingerprint(messages),
            "max_toks": str(max_tokens) if max_tokens is not None else "none",
            "temp": f"{temperature:.6f}" if temperature is not None else "none",
            "top_p": f"{top_p:.6f}" if top_p is not None else "none",
            "freq_pen": f"{frequency_penalty:.6f}" if frequency_penalty is not None else "none",
            "pres_pen": f"{presence_penalty:.6f}" if presence_penalty is not None else "none",
            "stops": ",".join(stop_sequences) if stop_sequences else "none",
            "json": "1" if caps.supports_json_output else "0",
            "family": caps.model_family,
            "ver": caps.version,
        }
        raw = "|".join(f"{k}={v}" for k, v in sorted(parts.items()))
        return f"llm.complete:{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"

    async def _apply_deadline(self, awaitable, ctx: Optional[OperationContext]) -> Any:
        """
        Apply the configured deadline policy to an awaitable.

        Any asyncio.TimeoutError is normalized into DeadlineExceeded.
        """
        try:
            return await self._deadline.wrap(awaitable, ctx)
        except asyncio.TimeoutError as e:
            raise DeadlineExceeded(
                "operation timed out",
                details={"remaining_ms": 0},
            ) from e

    async def _preflight_context_window_if_supported(
        self,
        *,
        messages: List[Mapping[str, str]],
        system_message: Optional[str],
        max_tokens: Optional[int],
        model: Optional[str],
        ctx: Optional[OperationContext],
        caps: LLMCapabilities,
    ) -> None:
        """
        Optional context-window preflight using count_tokens if supported.

        This is soft validation:
          - Uses count_tokens if available.
          - Prevents obviously invalid (prompt + max_tokens) > max_context_length.
          - Never blocks the call on transient count_tokens errors.
        """
        if not caps.supports_count_tokens or caps.max_context_length <= 0:
            return
        if max_tokens is not None and max_tokens < 0:
            raise BadRequest("max_tokens must be >= 0")

        # Construct a simple concatenated text for counting; adapters may override
        # behavior in _do_count_tokens for more accurate per-message encoding.
        parts: List[str] = []
        if system_message:
            parts.append(f"system:{system_message}")
        for m in messages:
            parts.append(f"{m.get('role','')}:{m.get('content','')}")
        combined = "\n".join(parts)

        try:
            prompt_tokens = await self._apply_deadline(
                self._do_count_tokens(text=combined, model=model, ctx=ctx),
                ctx,
            )
        except NotSupported:
            return  # adapter does not support counting precisely
        except LLMAdapterError:
            return  # do not block on preflight failures

        if max_tokens is not None:
            total_possible = prompt_tokens + max(0, int(max_tokens))
            if total_possible > caps.max_context_length:
                raise BadRequest(
                    f"prompt tokens ({prompt_tokens}) + max_tokens ({max_tokens}) "
                    f"exceed max_context_length ({caps.max_context_length})"
                )

    def _gate_model_if_listed(self, *, model: Optional[str], caps: LLMCapabilities) -> None:
        """
        If adapter advertises an explicit supported_models list, ensure the requested
        model is in that list. Otherwise, the adapter is treated as open-set.
        """
        if model and caps.supported_models and model not in caps.supported_models:
            raise BadRequest(f"model '{model}' is not supported by this adapter")

    # --- final public APIs (validation + instrumentation) ---

    async def capabilities(self) -> LLMCapabilities:
        """Get the capabilities of this LLM adapter."""
        return await self._do_capabilities()

    async def complete(
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
        ctx: Optional[OperationContext] = None,
    ) -> LLMCompletion:
        """
        Execute a complete LLM conversation with validation, policy hooks, and metrics.

        See LLMProtocolV1.complete for full contract details.
        """
        self._validate_messages(messages)
        self._validate_sampling_params(
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
        self._preflight_deadline(ctx)

        # Circuit breaker gate
        if not self._breaker.allow():
            raise Unavailable("circuit open")

        # Rate limit acquire (may block)
        await self._limiter.acquire()

        t0 = time.monotonic()
        try:
            caps = await self._do_capabilities()
            self._gate_model_if_listed(model=model, caps=caps)

            # Optional soft preflight for context length if supported
            await self._preflight_context_window_if_supported(
                messages=messages,
                system_message=system_message,
                max_tokens=max_tokens,
                model=model,
                ctx=ctx,
                caps=caps,
            )

            # Optional cache (standalone/dev only via InMemoryTTLCache)
            cache_key = None
            cached: Optional[LLMCompletion] = None
            if isinstance(self._cache, InMemoryTTLCache):
                cache_key = self._make_complete_cache_key(
                    model=model,
                    system_message=system_message,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop_sequences=stop_sequences,
                    caps=caps,
                )
                cached = await self._cache.get(cache_key)

            if cached:
                self._metrics.counter(
                    component=self._component,
                    name="cache_hits",
                    value=1,
                )
                result: LLMCompletion = cached
            else:
                # Execute under deadline policy
                result = await self._apply_deadline(
                    self._do_complete(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                        stop_sequences=stop_sequences,
                        model=model,
                        system_message=system_message,
                        ctx=ctx,
                    ),
                    ctx,
                )
                # Best-effort set cache (do not fail request if cache write fails)
                if cache_key is not None:
                    try:
                        await self._cache.set(cache_key, result, ttl_s=self._cache_ttl_s)
                    except Exception:
                        pass

            # Metrics
            extra: Dict[str, Any] = {}
            if self._tag_model_in_metrics and result.model:
                extra["model"] = result.model
            self._record("complete", t0, True, ctx=ctx, **extra)

            # Counters
            self._metrics.counter(
                component=self._component,
                name="requests_total",
                value=1,
            )
            if result.usage and isinstance(result.usage.total_tokens, int):
                self._metrics.counter(
                    component=self._component,
                    name="tokens_processed",
                    value=int(result.usage.total_tokens),
                )

            self._breaker.on_success()
            return result

        except LLMAdapterError as e:
            # Normalized/known failures
            extra = {"code": type(e).__name__}
            if self._tag_model_in_metrics and model:
                extra["model"] = model
            self._record("complete", t0, False, code=type(e).__name__, ctx=ctx, **extra)
            self._breaker.on_error(e)
            raise
        except Exception as e:
            # Unexpected failures are surfaced and recorded as UnhandledException
            extra = {}
            if self._tag_model_in_metrics and model:
                extra["model"] = model
            self._record("complete", t0, False, code="UnhandledException", ctx=ctx, **extra)
            self._breaker.on_error(e)
            raise
        finally:
            self._limiter.release()

    async def stream(
        self,
        *,
        messages: List[Mapping[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        system_message: Optional[str] = None,
        ctx: Optional[OperationContext] = None,
    ) -> AsyncIterator[LLMChunk]:
        """
        Execute an LLM conversation with streaming, validation, and metrics.

        See LLMProtocolV1.stream for full contract details.

        Notes:
            - Context-window preflight uses max_tokens if available.
            - Deadline is applied per-chunk via the configured DeadlinePolicy.
        """
        self._validate_messages(messages)
        self._validate_sampling_params(
            temperature=temperature,
            top_p=None,  # streaming signature has fewer params; adapters may still use defaults
            frequency_penalty=None,
            presence_penalty=None,
        )
        self._preflight_deadline(ctx)

        # Circuit breaker gate
        if not self._breaker.allow():
            raise Unavailable("circuit open")

        # Rate limit acquire
        await self._limiter.acquire()

        t0 = time.monotonic()
        finished_ok = False
        try:
            caps = await self._do_capabilities()
            self._gate_model_if_listed(model=model, caps=caps)

            # Optional soft preflight for context length if supported
            await self._preflight_context_window_if_supported(
                messages=messages,
                system_message=system_message,
                max_tokens=max_tokens,
                model=model,
                ctx=ctx,
                caps=caps,
            )

            # Wrap the underlying async generator so we can enforce deadline on each __anext__
            agen = self._do_stream(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                model=model,
                system_message=system_message,
                ctx=ctx,
            )

            while True:
                try:
                    # Enforce deadline per chunk fetch
                    chunk = await self._apply_deadline(agen.__anext__(), ctx)
                except StopAsyncIteration:
                    finished_ok = True
                    break

                yield chunk

            # Metrics (best-effort)
            extra: Dict[str, Any] = {}
            if self._tag_model_in_metrics and model:
                extra["model"] = model
            self._record("stream", t0, True, ctx=ctx, **extra)
            self._metrics.counter(
                component=self._component,
                name="stream_requests_total",
                value=1,
            )
        except LLMAdapterError as e:
            extra = {"code": type(e).__name__}
            if self._tag_model_in_metrics and model:
                extra["model"] = model
            self._record("stream", t0, False, code=type(e).__name__, ctx=ctx, **extra)
            self._breaker.on_error(e)
            raise
        except Exception as e:
            extra = {}
            if self._tag_model_in_metrics and model:
                extra["model"] = model
            self._record("stream", t0, False, code="UnhandledException", ctx=ctx, **extra)
            self._breaker.on_error(e)
            raise
        finally:
            self._limiter.release()
            if finished_ok:
                # Only mark success if we drained the stream cleanly.
                self._breaker.on_success()

    async def count_tokens(
        self,
        text: str,
        *,
        model: Optional[str] = None,
        ctx: Optional[OperationContext] = None,
    ) -> int:
        """
        Count tokens in text with metrics instrumentation.

        Delegates to `_do_count_tokens` and wraps it with deadline + metrics.
        """
        if not isinstance(text, str) or not text:
            raise BadRequest("text must be a non-empty string")

        t0 = time.monotonic()
        try:
            caps = await self._do_capabilities()
            if model and caps.supported_models and model not in caps.supported_models:
                raise BadRequest(f"model '{model}' is not supported by this adapter")
            if not caps.supports_count_tokens:
                raise NotSupported("count_tokens is not supported by this adapter")

            self._preflight_deadline(ctx)
            result = await self._apply_deadline(
                self._do_count_tokens(text=text, model=model, ctx=ctx),
                ctx,
            )

            extra: Dict[str, Any] = {}
            if self._tag_model_in_metrics and model:
                extra["model"] = model
            self._record(
                "count_tokens",
                t0,
                True,
                ctx=ctx,
                text_length=len(text),
                **extra,
            )
            self._metrics.counter(
                component=self._component,
                name="count_tokens_calls",
                value=1,
            )
            return int(result)
        except LLMAdapterError as e:
            self._record(
                "count_tokens",
                t0,
                False,
                code=type(e).__name__,
                ctx=ctx,
                model=str(model or ""),
            )
            raise
        except Exception as e:
            self._record(
                "count_tokens",
                t0,
                False,
                code="UnhandledException",
                ctx=ctx,
                model=str(model or ""),
            )
            raise

    async def health(self, *, ctx: Optional[OperationContext] = None) -> Mapping[str, Any]:
        """
        Check health status with metrics instrumentation.

        This is intentionally small: adapters should return a minimal mapping.
        """
        t0 = time.monotonic()
        try:
            self._preflight_deadline(ctx)
            h = await self._apply_deadline(self._do_health(ctx=ctx), ctx)
            self._record("health", t0, True, ctx=ctx)
            return {
                "ok": bool(h.get("ok", True)),
                "server": str(h.get("server", "")),
                "version": str(h.get("version", "")),
            }
        except LLMAdapterError as e:
            self._record("health", t0, False, code=type(e).__name__, ctx=ctx)
            raise
        except Exception as e:
            self._record("health", t0, False, code="UnhandledException", ctx=ctx)
            # Normalize unexpected errors as Unavailable for callers
            raise Unavailable("health check failed") from e

    # --- hooks to implement per backend (override these) ---

    async def _do_capabilities(self) -> LLMCapabilities:
        """
        Implement to return adapter-specific capabilities.

        Must be cheap and side-effect free; may call remote endpoints if required.
        """
        raise NotImplementedError

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
        ctx: Optional[OperationContext] = None,
    ) -> LLMCompletion:
        """
        Implement complete LLM conversation with validated inputs.

        Called only after base has enforced protocol-level validation.
        """
        raise NotImplementedError

    async def _do_stream(
        self,
        *,
        messages: List[Mapping[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        system_message: Optional[str] = None,
        ctx: Optional[OperationContext] = None,
    ) -> AsyncIterator[LLMChunk]:
        """
        Implement streaming LLM conversation with validated inputs.

        Implementers should yield LLMChunk instances.
        """
        raise NotImplementedError

    async def _do_count_tokens(
        self,
        text: str,
        *,
        model: Optional[str] = None,
        ctx: Optional[OperationContext] = None,
    ) -> int:
        """
        Implement token counting for the specified model.

        May use provider tokenizer or an approximate local tokenizer.
        """
        raise NotImplementedError

    async def _do_health(self, *, ctx: Optional[OperationContext] = None) -> Mapping[str, Any]:
        """
        Implement health check for the LLM backend.

        Should not raise on minor/transient issues; callers rely on this
        for coarse readiness.
        """
        raise NotImplementedError


# =============================================================================
# Wire-Level Helpers (canonical envelopes)
# =============================================================================

def _ctx_from_wire(ctx_dict: Mapping[str, Any]) -> OperationContext:
    """
    Convert a wire-level ctx dict into an OperationContext.

    Unknown keys are ignored per protocol rules to allow forward compatibility.
    """
    if ctx_dict is None:
        return OperationContext()
    return OperationContext(
        request_id=ctx_dict.get("request_id"),
        idempotency_key=ctx_dict.get("idempotency_key"),
        deadline_ms=ctx_dict.get("deadline_ms"),
        traceparent=ctx_dict.get("traceparent"),
        tenant=ctx_dict.get("tenant"),
        attrs=ctx_dict.get("attrs") or {},
    )


def _error_to_wire(e: Exception, ms: float) -> Dict[str, Any]:
    """
    Map LLMAdapterError (or unexpected Exception) to canonical error envelope.

    This is the single source of truth for wire-level error normalization.
    """
    if isinstance(e, LLMAdapterError):
        return {
            "ok": False,
            "code": (e.code or type(e).__name__.upper()),
            "error": type(e).__name__,
            "message": e.message,
            "retry_after_ms": e.retry_after_ms,
            "details": e.details or None,
            "ms": ms,
        }
    # Fallback: treat as UNAVAILABLE/INTERNAL
    return {
        "ok": False,
        "code": "UNAVAILABLE",
        "error": type(e).__name__,
        "message": str(e) or "internal error",
        "retry_after_ms": None,
        "details": None,
        "ms": ms,
    }


def _success_to_wire(result: Any, ms: float) -> Dict[str, Any]:
    """
    Map typed result objects to canonical success envelope.

    Uses dataclasses.asdict() where applicable, else passes through.
    """
    if hasattr(result, "__dataclass_fields__"):
        payload = asdict(result)
    else:
        payload = result
    return {
        "ok": True,
        "code": "OK",
        "ms": ms,
        "result": payload,
    }


def _chunk_to_wire(chunk: LLMChunk, ms: float) -> Dict[str, Any]:
    """
    Map an LLMChunk to a canonical streaming envelope.

    This is used only by WireLLMHandler.handle_stream.
    """
    if hasattr(chunk, "__dataclass_fields__"):
        payload = asdict(chunk)
    else:
        payload = {
            "text": chunk.text,
            "is_final": getattr(chunk, "is_final", False),
            "model": getattr(chunk, "model", None),
            "usage_so_far": asdict(chunk.usage_so_far) if getattr(chunk, "usage_so_far", None) else None,
        }
    return {
        "ok": True,
        "code": "OK",
        "ms": ms,
        "chunk": payload,
    }


class WireLLMHandler:
    """
    Thin wire-level adapter that exposes an LLMProtocolV1 implementation using
    the canonical JSON envelope contract.

    This handler is transport-agnostic and can be used with HTTP, gRPC, WebSockets, etc.
    """

    def __init__(self, adapter: LLMProtocolV1):
        self._adapter = adapter

    async def handle(self, envelope: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Handle a single unary request envelope and return a response envelope.

        Supports:
            - llm.capabilities
            - llm.complete
            - llm.count_tokens
            - llm.health

        Streaming (llm.stream) MUST use handle_stream.
        """
        t0 = time.monotonic()
        try:
            op = envelope.get("op")
            if not isinstance(op, str):
                raise BadRequest("missing or invalid 'op'")

            ctx = _ctx_from_wire(envelope.get("ctx") or {})
            args = envelope.get("args") or {}

            if op == "llm.capabilities":
                res = await self._adapter.capabilities()
                return _success_to_wire(asdict(res), (time.monotonic() - t0) * 1000.0)

            if op == "llm.complete":
                res = await self._adapter.complete(
                    messages=args.get("messages") or [],
                    max_tokens=args.get("max_tokens"),
                    temperature=args.get("temperature"),
                    top_p=args.get("top_p"),
                    frequency_penalty=args.get("frequency_penalty"),
                    presence_penalty=args.get("presence_penalty"),
                    stop_sequences=args.get("stop_sequences"),
                    model=args.get("model"),
                    system_message=args.get("system_message"),
                    ctx=ctx,
                )
                return _success_to_wire(asdict(res), (time.monotonic() - t0) * 1000.0)

            if op == "llm.count_tokens":
                text = args.get("text")
                if not isinstance(text, str):
                    raise BadRequest("text must be a string")
                res = await self._adapter.count_tokens(
                    text=text,
                    model=args.get("model"),
                    ctx=ctx,
                )
                return _success_to_wire(res, (time.monotonic() - t0) * 1000.0)

            if op == "llm.health":
                res = await self._adapter.health(ctx=ctx)
                return _success_to_wire(res, (time.monotonic() - t0) * 1000.0)

            # llm.stream is handled via handle_stream (streaming), not here.
            raise NotSupported(f"unknown or non-unary operation '{op}'")

        except Exception as e:
            ms = (time.monotonic() - t0) * 1000.0
            return _error_to_wire(e, ms)

    async def handle_stream(self, envelope: Mapping[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """
        Handle a streaming request envelope.

        Expects:
            op: "llm.stream"
            ctx: { ... }
            args: { messages, max_tokens?, temperature?, model?, system_message? }

        Yields:
            Streaming envelopes containing "chunk" or a terminal error envelope.
        """
        t0 = time.monotonic()
        op = envelope.get("op")
        if op != "llm.stream":
            yield _error_to_wire(BadRequest("op must be 'llm.stream' for streaming"), 0.0)
            return

        ctx = _ctx_from_wire(envelope.get("ctx") or {})
        args = envelope.get("args") or {}

        try:
            agen = self._adapter.stream(
                messages=args.get("messages") or [],
                max_tokens=args.get("max_tokens"),
                temperature=args.get("temperature"),
                model=args.get("model"),
                system_message=args.get("system_message"),
                ctx=ctx,
            )

            async for chunk in agen:
                ms = (time.monotonic() - t0) * 1000.0
                yield _chunk_to_wire(chunk, ms)
        except Exception as e:
            ms = (time.monotonic() - t0) * 1000.0
            yield _error_to_wire(e, ms)


__all__ = [
    "LLM_PROTOCOL_VERSION",
    "LLM_PROTOCOL_ID",
    "LLMAdapterError",
    "BadRequest",
    "AuthError",
    "ResourceExhausted",
    "TransientNetwork",
    "Unavailable",
    "NotSupported",
    "ModelOverloaded",
    "DeadlineExceeded",
    "OperationContext",
    "MetricsSink",
    "NoopMetrics",
    "TokenUsage",
    "LLMCompletion",
    "LLMChunk",
    "DeadlinePolicy",
    "CircuitBreaker",
    "Cache",
    "RateLimiter",
    "NoopDeadline",
    "SimpleDeadline",
    "NoopBreaker",
    "SimpleCircuitBreaker",
    "NoopCache",
    "InMemoryTTLCache",
    "NoopLimiter",
    "TokenBucketLimiter",
    "LLMCapabilities",
    "LLMProtocolV1",
    "BaseLLMAdapter",
    # wire helpers
    "WireLLMHandler",
    "_ctx_from_wire",
    "_error_to_wire",
    "_success_to_wire",
    "_chunk_to_wire",
]
