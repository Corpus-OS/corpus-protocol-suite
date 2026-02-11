# corpus_sdk/llm/llm_base.py
# SPDX-License-Identifier: Apache-2.0

"""
Adapter SDK — LLM Protocol V1 (public contract + production-grade base)

Purpose
-------
A stable, vendor-neutral API for calling Large Language Models — with:

- Structured, normalized error taxonomy (SIEM-safe, machine-actionable)
- Streaming support for low-latency partial responses
- Token usage accounting for cost/quota management
- Deadline propagation and cooperative cancellation
- Built-in hooks for circuit breaking, rate limiting, and caching
- Canonical JSON wire envelopes via WireLLMHandler (transport-agnostic)
- First-class support for Tool Calling (Function Calling)

This protocol enables seamless integration with any LLM provider while maintaining
production-grade observability, security, and operational rigor.

Design Philosophy
-----------------
- Minimal core surface:
    * capabilities()
    * complete()
    * stream()
    * count_tokens()
    * health()
- Async-first: all operations are non-blocking and awaitable.
- Provider-neutral: no hard-coded vendor specifics; adapters map into this contract.
- DRY infra:
    * Shared gate wrappers for breaker / limiter / deadlines / metrics.
    * Single source of truth for error normalization and wire envelopes.
- Extensible:
    * Capabilities advertise feature flags.
    * Callers can branch on server/version/model_family/supported_models.

Deliberate Non-Goals
--------------------
- No retries, hedging, routing, model selection, or fallback.
- No tool-calling orchestration or agent frameworks (execution loops).
- No tokenizer transforms or prompt rewriting.
- No client-side stream reassembly helpers.

Those behaviors live in your router / control-plane / orchestration layers.

Mode Strategy (Composition vs. Standalone)
------------------------------------------
mode: "thin" (default)
    - For composition under an external manager/router.
    - All policies default to no-op:
        * No caching
        * No breaker
        * No rate limiter
    - Use when your infra already handles concurrency and resilience.

mode: "standalone"
    - For direct use in services.
    - Enables:
        * SimpleDeadline (ctx.deadline_ms)
        * SimpleCircuitBreaker
        * InMemoryTTLCache for complete()
        * TokenBucketLimiter
    - Intended for development / light production; NOT a distributed control plane.

Versioning
----------
Follow SemVer against LLM_PROTOCOL_VERSION (wire & type contract).

- Patch (x.y.Z):
    Editorial/documentation changes and strictly non-breaking code changes.
- Minor (x.Y.z):
    Additive fields, capabilities, or methods (must be backward compatible).
- Major (X.y.z):
    Breaking changes to signatures or semantics (avoid in base; prefer additive).

Wire Contract (Canonical Interface)
-----------------------------------
This module defines the in-process protocol and a reference wire adapter.

Canonical JSON envelope:

    Request:
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

    Unary Success:
        {
            "ok": true,
            "code": "OK",
            "ms": <float>,          # elapsed milliseconds (best-effort)
            "result": { ... }
        }

    Unary Error:
        {
            "ok": false,
            "code": "<UPPER_SNAKE_CASE>",   # e.g. BAD_REQUEST, AUTH_ERROR
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
                "top_p": <float|null>,
                "frequency_penalty": <float|null>,
                "presence_penalty": <float|null>,
                "stop_sequences": [<str>, ...] | null,
                "model": "<model-id>|null",
                "system_message": "<str>|null",
                "tools": [ ... ] | null,
                "tool_choice": "..." | { ... } | null
            }
        }

    Stream Responses:
        - Zero or more chunk envelopes:
            {
                "ok": true,
                "code": "STREAMING",
                "ms": <float>,
                "chunk": {
                    "text": "<partial>",
                    "is_final": false,
                    "model": "<model-id>|null",
                    "tool_calls": [ ... ] | null,
                    "usage_so_far": {
                        "prompt_tokens": <int>,
                        "completion_tokens": <int>,
                        "total_tokens": <int>
                    } | null
                }
            }

        - On terminal success, last chunk SHOULD set "is_final": true.
        - On error, a single error envelope is sent and terminates the stream.

LLM_PROTOCOL_ID is advertised in capabilities; it is not required per request.

IMPORTANT WIRE STRICTNESS NOTE (Alignment)
------------------------------------------
For interoperability and forward/backward compatibility, this SDK treats the
WIRE boundary as strict:

- Envelopes MUST include top-level keys: op, ctx, args
- ctx MUST be an object (mapping)
- args MUST be an object (mapping)

This is intentionally stricter than in-process calls (ctx is optional there),
and is aligned with the canonical envelope contract.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import secrets
import time
from dataclasses import dataclass, asdict, field
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

LOG = logging.getLogger(__name__)

LLM_PROTOCOL_VERSION = "1.0.0"
LLM_PROTOCOL_ID = "llm/v1.0"

# =============================================================================
# Normalized Errors (with retry hints and structured details)
# =============================================================================


class LLMAdapterError(Exception):
    """
    Base exception for all LLM adapter errors.

    All adapter implementations SHOULD raise subclasses of this error so that callers
    and the wire handler can make consistent, machine-actionable decisions.

    Attributes:
        message:
            Human-readable description (safe for logs and clients).
        code:
            Upper-snake-case machine code; when omitted, wire layer derives from class.
        retry_after_ms:
            Optional client backoff hint (for 429 / overload / maintenance).
        throttle_scope:
            Scope of throttling ("tenant", "model", "cluster", etc.) when applicable.
        suggested_token_reduction:
            Optional hint (0-100) to suggest prompt size reduction on quota/limits.
        details:
            Additional JSON-safe context (never include secrets/PII).
    """

    def __init__(
        self,
        message: str = "",
        *,
        code: Optional[str] = None,
        retry_after_ms: Optional[int] = None,
        throttle_scope: Optional[str] = None,
        suggested_token_reduction: Optional[int] = None,
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
    """
    Client error: malformed messages, invalid parameters, or unsupported options.

    Examples:
        - Empty messages
        - Invalid temperature/top_p ranges
        - Unknown model when capabilities.supported_models is authoritative
    """

    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("code", "BAD_REQUEST")
        super().__init__(message, **kwargs)


class AuthError(LLMAdapterError):
    """
    Authentication / authorization failure.

    Examples:
        - Invalid API key
        - Missing/invalid credentials
        - Tenant not allowed to access a given model
    """

    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("code", "AUTH_ERROR")
        super().__init__(message, **kwargs)


class ResourceExhausted(LLMAdapterError):
    """
    Quota, rate limit, or resource exhaustion.

    Callers should use retry_after_ms and/or throttle_scope when present.
    """

    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("code", "RESOURCE_EXHAUSTED")
        super().__init__(message, **kwargs)


class TransientNetwork(LLMAdapterError):
    """
    Retryable network failure.

    Indicates transport issues between adapter and upstream provider.
    """

    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("code", "TRANSIENT_NETWORK")
        super().__init__(message, **kwargs)


class Unavailable(LLMAdapterError):
    """
    Backend unavailable / overloaded / maintenance.

    Used when the adapter cannot reach the provider or service is degraded.
    """

    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("code", "UNAVAILABLE")
        super().__init__(message, **kwargs)


class NotSupported(LLMAdapterError):
    """
    Unsupported operation or parameter.

    Examples:
        - count_tokens not implemented
        - streaming not supported by provider
        - parallel tool calls not supported
    """

    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("code", "NOT_SUPPORTED")
        super().__init__(message, **kwargs)


class ModelOverloaded(LLMAdapterError):
    """
    Specific model is overloaded or hot-partitioned.

    Allows routers to distinguish model-level overload from global failures.
    """

    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("code", "MODEL_OVERLOADED")
        super().__init__(message, **kwargs)


class DeadlineExceeded(LLMAdapterError):
    """
    Operation exceeded the caller's deadline budget (ctx.deadline_ms).

    Emitted when:
        - Preflight sees an already-expired deadline.
        - DeadlinePolicy/asyncio.wait_for triggers a timeout.
    """

    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("code", "DEADLINE_EXCEEDED")
        super().__init__(message, **kwargs)


# =============================================================================
# Operation Context (tracing, deadlines, multi-tenant isolation)
# =============================================================================


@dataclass(frozen=True)
class OperationContext:
    """
    Context for LLM operations.

    All fields are optional and advisory but SHOULD be propagated by callers.

    Attributes:
        request_id:
            Correlation ID for tracing across systems.
        idempotency_key:
            For idempotent operations (e.g., retried completions).
        deadline_ms:
            Absolute epoch ms; used by SimpleDeadline and preflight checks.
        traceparent:
            W3C traceparent header for distributed tracing.
        tenant:
            Tenant/project identifier; never logged directly (only hashed).
        attrs:
            Additional JSON-serializable attributes for routing/middleware.
    """

    request_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    deadline_ms: Optional[int] = None
    traceparent: Optional[str] = None
    tenant: Optional[str] = None
    attrs: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        if self.attrs is None:
            object.__setattr__(self, "attrs", {})


# =============================================================================
# Metrics Interface (SIEM-safe, low-cardinality)
# =============================================================================


class MetricsSink(Protocol):
    """
    Metrics collection protocol.

    Implementations MUST:
        - Avoid PII.
        - Avoid high-cardinality labels.
        - Hash tenant identifiers when needed.
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
        ...

    def counter(
        self,
        *,
        component: str,
        name: str,
        value: int = 1,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        ...


class NoopMetrics:
    """No-op metrics sink for tests or minimal deployments."""

    def observe(self, **_: Any) -> None:
        ...

    def counter(self, **_: Any) -> None:
        ...


# =============================================================================
# Result Models (structured, JSON-safe responses)
# =============================================================================


@dataclass(frozen=True)
class ToolCallFunction:
    """
    Representation of a function call invocation within a tool call.
    """
    name: str
    arguments: str  # JSON string of arguments


@dataclass(frozen=True)
class ToolCall:
    """
    Structured tool call (function invocation) request.
    """
    id: str
    type: str  # e.g., "function"
    function: ToolCallFunction


@dataclass
class TokenUsage:
    """
    Token usage accounting for cost tracking and quota management.

    All fields are integers and JSON-serializable.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class LLMCompletion:
    """
    Full LLM completion result.

    Attributes:
        text:
            Final response text (router/orchestrator may post-process).
        model:
            Concrete model identifier (e.g. "gpt-4.1", "claude-3-opus").
        model_family:
            Logical family ("gpt-4", "claude-3", "gemini-pro", etc.).
        usage:
            TokenUsage with prompt/completion/total.
        finish_reason:
            "stop", "length", "error", "tool_calls", etc.
        tool_calls:
            List of tool/function calls generated by the model.
    """

    text: str
    model: str
    model_family: str
    usage: TokenUsage
    finish_reason: str
    tool_calls: List[ToolCall] = field(default_factory=list)


@dataclass
class LLMChunk:
    """
    Streaming response chunk.

    Attributes:
        text:
            Partial text delta (MAY be empty for non-text events if extended).
        is_final:
            Whether this chunk is the final one in the stream.
        model:
            Model identifier; MAY be None until final chunk.
        usage_so_far:
            Optional TokenUsage snapshot; typically only on final chunk.
        tool_calls:
            Partial or complete tool call deltas (adapter-dependent).
    """

    text: str
    is_final: bool = False
    model: Optional[str] = None
    usage_so_far: Optional[TokenUsage] = None
    tool_calls: List[ToolCall] = field(default_factory=list)


# =============================================================================
# Policy / Infra Extension Points
# =============================================================================


class DeadlinePolicy(Protocol):
    """Strategy interface for applying ctx.deadline_ms to awaitables."""

    async def wrap(self, awaitable: Awaitable[Any], ctx: Optional[OperationContext]) -> Any:
        ...


class CircuitBreaker(Protocol):
    """Minimal circuit breaker interface for gating upstream calls."""

    def allow(self) -> bool:
        ...

    def on_success(self) -> None:
        ...

    def on_error(self, err: Exception) -> None:
        ...


class Cache(Protocol):
    """Async cache interface used for complete() results."""

    async def get(self, key: str) -> Optional[Any]:
        ...

    async def set(self, key: str, value: Any, ttl_s: int) -> None:
        ...


class TTLAwareCache(Protocol):
    """
    Optional TTL capability surface for cache implementations.

    NOTE:
        This is deliberately separate from Cache to avoid forcing every cache
        implementation (and type-checkers) to define supports_ttl.

    BaseLLMAdapter uses getattr(cache, "supports_ttl", None) to detect this
    capability at runtime; caches that do not expose supports_ttl are treated
    as TTL-capable by default for backward compatibility.
    """

    @property
    def supports_ttl(self) -> bool:
        ...


class RateLimiter(Protocol):
    """Simple rate limiter interface."""

    async def acquire(self) -> None:
        ...

    def release(self) -> None:
        ...


class NoopDeadline:
    """No-op deadline policy (used in thin/composed mode by default)."""

    async def wrap(self, awaitable: Awaitable[Any], ctx: Optional[OperationContext]) -> Any:
        return await awaitable


class SimpleDeadline:
    """
    Deadline policy that enforces ctx.deadline_ms via asyncio.wait_for.

    Behavior:
        - If no deadline: pass-through.
        - If already expired: raises DeadlineExceeded immediately.
        - If timed out: raises DeadlineExceeded.
    """

    async def wrap(self, awaitable: Awaitable[Any], ctx: Optional[OperationContext]) -> Any:
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
    """Breaker that never trips; safe default for thin mode."""

    def allow(self) -> bool:
        return True

    def on_success(self) -> None:
        ...

    def on_error(self, err: Exception) -> None:
        ...


class SimpleCircuitBreaker:
    """
    Tiny per-process circuit breaker.

    Not distributed. Intended for standalone/dev usage.

    Semantics:
        - Opens after N consecutive failures.
        - Once the recovery window has elapsed, allow() simply returns True
          again on subsequent calls; there is no distinct half-open state
          that restricts to a single probe call. This is intentionally
          simpler than a full half-open implementation.
    """

    def __init__(self, *, failure_threshold: int = 5, recovery_after_s: float = 10.0) -> None:
        self._failure_threshold = max(1, int(failure_threshold))
        self._recovery_after_s = max(0.1, float(recovery_after_s))
        self._failures = 0
        self._opened_at: Optional[float] = None

    def allow(self) -> bool:
        if self._opened_at is None:
            return True
        # Allow calls again once the recovery window has passed. There is no
        # explicit half-open state; the next successful call will fully reset.
        if (time.monotonic() - self._opened_at) >= self._recovery_after_s:
            return True
        return False

    def on_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def on_error(self, _err: Exception) -> None:
        self._failures += 1
        if self._failures >= self._failure_threshold:
            self._opened_at = time.monotonic()


class NoopCache:
    """
    No-op cache implementation.

    This implementation never stores values. The ttl_s parameter on set() is
    accepted for interface compatibility but ignored.
    """

    async def get(self, key: str) -> Optional[Any]:
        return None

    async def set(self, key: str, value: Any, ttl_s: int) -> None:
        """Accepts ttl_s for interface compatibility but ignores it."""
        return None

    # Optional capability surface (not required by Cache Protocol)
    @property
    def supports_ttl(self) -> bool:
        return False


class InMemoryTTLCache:
    """
    In-memory TTL cache for complete() results.

    Characteristics:
        - Per-process only; NOT shared/distributed across processes.
        - Not thread-safe; intended for single-threaded event loop usage.
        - Opportunistic pruning; callers must not rely on strong guarantees.
        - Safe default for standalone/dev mode.

    NOTE:
        ttl_s <= 0 means "do not cache" and is treated as a hard opt-out.
        This allows BaseLLMAdapter.cache_ttl_s == 0 to disable caching end-to-end.
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
            try:
                del self._store[key]
            except Exception:
                pass
            return None
        return val

    async def set(self, key: str, value: Any, ttl_s: int) -> None:
        # Treat ttl_s <= 0 as "do not cache" (hard opt-out).
        if int(ttl_s) <= 0:
            return None

        exp = time.monotonic() + max(1, int(ttl_s))
        self._store[key] = (exp, value)
        # Basic pruning to avoid unbounded growth.
        if len(self._store) > 4096:
            try:
                for k, (e, _) in list(self._store.items())[:2048]:
                    if time.monotonic() >= e:
                        del self._store[k]
            except Exception:
                pass

    # Optional capability surface (not required by Cache Protocol)
    @property
    def supports_ttl(self) -> bool:
        return True


class NoopLimiter:
    """No-op rate limiter."""

    async def acquire(self) -> None:
        return None

    def release(self) -> None:
        return None


class TokenBucketLimiter:
    """
    Simple token-bucket limiter; per-process only.

    Notes:
        - Not thread-safe; intended for single-threaded async event loops.
        - Charges only on acquire(); release() is a no-op.
        - Intended for standalone/dev; use a real distributed limiter in production.
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
        while True:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            await asyncio.sleep(0.01)

    def release(self) -> None:
        return None


# =============================================================================
# Capabilities (dynamic discovery for routing and planning)
# =============================================================================


@dataclass(frozen=True)
class LLMCapabilities:
    """
    Describes the capabilities and limits of an LLM adapter.

    Used by routers/control-plane for:
        - Model selection and feature routing
        - Input validation (e.g., context window)
        - Detecting support for streaming / JSON modes / tools / deadlines
    """

    server: str
    version: str
    model_family: str
    max_context_length: int
    protocol: str = LLM_PROTOCOL_ID
    supports_streaming: bool = True
    supports_roles: bool = True
    supports_json_output: bool = False
    supports_tools: bool = False
    supports_parallel_tool_calls: bool = False
    supports_tool_choice: bool = False
    max_tool_calls_per_turn: Optional[int] = None
    idempotent_writes: bool = False
    supports_multi_tenant: bool = False
    supports_system_message: bool = True
    supports_deadline: bool = True
    supports_count_tokens: bool = True
    supported_models: Tuple[str, ...] = ()  # empty ⇒ open/adapter-defined set


# =============================================================================
# Stable Protocol Interface (async, versioned contract)
# =============================================================================


@runtime_checkable
class LLMProtocolV1(Protocol):
    """
    Language-level contract for LLM adapters.

    Implementations MUST:
        - Be async-only.
        - Raise LLMAdapterError subclasses on failure.
        - Follow semantics documented here.
    """

    async def capabilities(self) -> LLMCapabilities:
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
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        ctx: Optional[OperationContext] = None,
    ) -> LLMCompletion:
        ...

    async def stream(
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
        ...

    async def count_tokens(
        self,
        text: str,
        *,
        model: Optional[str] = None,
        ctx: Optional[OperationContext] = None,
    ) -> int:
        ...

    async def health(self, *, ctx: Optional[OperationContext] = None) -> Mapping[str, Any]:
        ...


# =============================================================================
# Base Instrumented Adapter (validation, metrics, DRY gates)
# =============================================================================


class BaseLLMAdapter(LLMProtocolV1):
    """
    Base implementation of LLMProtocolV1.

    This class:
        - Validates requests (messages, sampling params, tools).
        - Ensures JSON-serializability of message payloads (fast-fail).
        - Applies deadline policies and preflight checks.
        - Wraps calls with circuit breaker + rate limiter.
        - Optionally caches complete() results (standalone mode).
        - Emits SIEM-safe metrics (hashed tenant IDs).
        - Provides DRY gate helpers for unary and streaming operations.

    Backend implementers override only the `_do_*` hooks.
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
        stream_deadline_check_every_n_chunks: int = 10,
    ) -> None:
        """
        Initialize the LLM adapter with common infra.

        Args:
            metrics:
                Metrics sink implementation; defaults to NoopMetrics.
            mode:
                "thin":
                    - All hooks are effectively no-op unless explicitly provided.
                "standalone":
                    - Enables SimpleDeadline, SimpleCircuitBreaker,
                      InMemoryTTLCache, TokenBucketLimiter.
            deadline_policy:
                Custom DeadlinePolicy; overrides mode defaults when provided.
            breaker:
                Custom CircuitBreaker; overrides mode defaults when provided.
            cache:
                Custom Cache; used only for complete() responses.
            limiter:
                Custom RateLimiter; used for all operations.
            tag_model_in_metrics:
                If True, attaches model to metrics where known.
            cache_ttl_s:
                TTL for entries in the standalone in-memory cache.
                NOTE: cache_ttl_s == 0 disables caching end-to-end.
            stream_deadline_check_every_n_chunks:
                For streaming, perform deadline checks every N chunks instead
                of for every single chunk. Reduces overhead for high-volume
                streams while preserving deadline semantics.
        """
        self._metrics: MetricsSink = metrics or NoopMetrics()
        self._tag_model_in_metrics: bool = bool(tag_model_in_metrics)

        # Configuration validation
        if int(cache_ttl_s) < 0:
            raise ValueError("cache_ttl_s must be non-negative")
        # Allow 0 to mean "no caching" (end-to-end).
        self._cache_ttl_s: int = max(0, int(cache_ttl_s))

        if int(stream_deadline_check_every_n_chunks) < 1:
            raise ValueError("stream_deadline_check_every_n_chunks must be >= 1")
        self._stream_deadline_check_every_n_chunks: int = max(
            1, int(stream_deadline_check_every_n_chunks)
        )

        m = (mode or "thin").strip().lower()
        if m not in {"thin", "standalone"}:
            m = "thin"
        self._mode = m

        if self._mode == "standalone":
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
            self._deadline = deadline_policy or NoopDeadline()
            self._breaker = breaker or NoopBreaker()
            self._cache = cache or NoopCache()
            self._limiter = limiter or NoopLimiter()

        # Capabilities cache key is namespaced per adapter instance to avoid
        # accidental collisions when sharing a cache across multiple adapters.
        self._caps_cache_key = (
            f"llm:capabilities:"
            f"{self.__class__.__module__}.{self.__class__.__qualname__}:{id(self)}"
        )

    # --- async context management (resource cleanup hint) --------------------

    async def __aenter__(self) -> "BaseLLMAdapter":
        """
        Support async context manager usage:

            async with MyAdapter(...) as adapter:
                ...

        Backend implementations may override close() for resource cleanup.
        """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def close(self) -> None:
        """
        Clean up resources (e.g., HTTP sessions, connection pools).

        Override in concrete adapters as needed. Default is a no-op.
        """
        return None

    # --- internal helpers (validation, metrics, cache safety) ----------------

    @staticmethod
    def _safe_deepcopy(obj: Any) -> Any:
        """
        Best-effort deepcopy helper for cache safety.

        Cache implementations frequently store objects by reference. Deep-copying
        cached results on both read and write prevents cross-request mutation
        and accidental state bleed across callers.
        """
        try:
            return copy.deepcopy(obj)
        except Exception:
            # Deepcopy failures must never break callers; return best-effort.
            return obj

    @staticmethod
    def _bucket_wait_ms(ms: float) -> str:
        """
        Low-cardinality wait-time buckets for limiter observability.

        This keeps metrics cardinality bounded while still providing signal
        when the limiter is actively applying backpressure.
        """
        if ms < 1.0:
            return "<1ms"
        if ms < 10.0:
            return "1-10ms"
        if ms < 100.0:
            return "10-100ms"
        return ">=100ms"

    @staticmethod
    def _validate_messages(messages: List[Mapping[str, str]]) -> None:
        """
        Validate that messages is a non-empty list of {role, content} mappings
        with string values.
        """
        if not messages:
            raise BadRequest(
                "messages must be a non-empty list of mappings with string 'role' and 'content'"
            )
        for m in messages:
            if not isinstance(m, Mapping):
                raise BadRequest(
                    "each message must be a mapping with string 'role' and 'content'"
                )
            if "role" not in m or "content" not in m:
                raise BadRequest(
                    "each message must include 'role' and 'content' keys"
                )
            if not isinstance(m["role"], str) or not isinstance(m["content"], str):
                raise BadRequest(
                    "each message must have string 'role' and 'content' fields"
                )

    @staticmethod
    def _validate_tools(tools: Optional[List[Dict[str, Any]]]) -> None:
        """
        Validate tool definitions structure if provided.
        """
        if tools is None:
            return
        if not isinstance(tools, list):
            raise BadRequest("tools must be a list")
        for t in tools:
            if not isinstance(t, dict):
                raise BadRequest("each tool must be a dictionary")
            if "type" not in t:
                raise BadRequest("each tool must specify 'type'")
            if t["type"] == "function" and "function" not in t:
                raise BadRequest("function tools must include 'function' definition")

    @staticmethod
    def _validate_tool_choice(tool_choice: Optional[Union[str, Dict[str, Any]]]) -> None:
        """
        Validate tool_choice parameter if provided.
        """
        if tool_choice is None:
            return
        if isinstance(tool_choice, str):
            valid_choices = {"auto", "none", "required"}
            if tool_choice not in valid_choices:
                raise BadRequest(
                    f"tool_choice must be one of {valid_choices} or a tool specification"
                )
        elif not isinstance(tool_choice, dict):
            raise BadRequest("tool_choice must be a string or dictionary")

    @staticmethod
    def _validate_stop_sequences(stop_sequences: Optional[List[str]]) -> None:
        """
        Validate stop_sequences parameter if provided.
        """
        if stop_sequences is None:
            return
        if not isinstance(stop_sequences, list) or any(
            not isinstance(s, str) for s in stop_sequences
        ):
            raise BadRequest("stop_sequences must be a list of strings")

    @staticmethod
    def _validate_message_content_serializable(messages: List[Mapping[str, str]]) -> None:
        """
        Ensure messages are JSON-serializable.

        Aligns with graph adapter behavior: fail fast before invoking backends
        when payload contains unserializable types.
        """
        try:
            json.dumps(messages)
        except (TypeError, ValueError) as e:
            raise BadRequest(f"messages must be JSON-serializable: {e}")

    @staticmethod
    def _validate_sampling_params(
        *,
        temperature: Optional[float],
        top_p: Optional[float],
        frequency_penalty: Optional[float],
        presence_penalty: Optional[float],
    ) -> None:
        """
        Validate core sampling parameters in conservative, production-safe ranges.
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
        Hash tenant for metrics/logging.

        Raw tenant identifiers MUST NEVER be emitted.
        """
        if not t:
            return None
        return hashlib.sha256(t.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _hash_str(s: Optional[str]) -> str:
        """
        Hash a string into a stable, non-reversible identifier.

        Intended for metrics/logging and cache keys to reduce PII exposure.
        Hash collisions are possible in theory but extremely unlikely for our
        usage and are acceptable for this purpose.
        """
        if s is None:
            return "none"
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    @staticmethod
    def _messages_fingerprint(messages: List[Mapping[str, str]]) -> str:
        """
        Stable fingerprint for a messages list without leaking raw content.

        Uses a SHA-256 hash over role/content pairs. This deliberately trades
        away reversibility to minimize PII exposure. Callers MUST NOT rely on
        this as a cryptographic signature or for strict uniqueness guarantees.
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

    @staticmethod
    def _generate_tool_call_id() -> str:
        """
        Generate a unique tool call ID.
        """
        # Use a randomness-based ID rather than time-based hashing for better
        # collision resistance under high concurrency.
        return f"call_{secrets.token_hex(8)}"

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
        Emit a timing metric for an operation.

        Any failures in metrics emission are swallowed.
        """
        try:
            ms = (time.monotonic() - t0) * 1000.0
            x = dict(extra or {})
            if ctx:
                tenant_h = self._tenant_hash(ctx.tenant)
                if tenant_h:
                    x["tenant_hash"] = tenant_h
            self._metrics.observe(
                component=self._component,
                op=op,
                ms=ms,
                ok=ok,
                code=code,
                extra=x or None,
            )
            if not ok:
                # Standard error counter (SIEM-safe, low-cardinality).
                self._metrics.counter(
                    component=self._component,
                    name="errors_total",
                    value=1,
                    extra={"op": op, "code": code},
                )
        except Exception:
            pass

    def _preflight_deadline(self, ctx: Optional[OperationContext]) -> None:
        """
        Fast-fail if ctx.deadline_ms is already elapsed.
        """
        if ctx and ctx.deadline_ms is not None:
            now_ms = int(time.time() * 1000)
            if now_ms >= ctx.deadline_ms:
                raise DeadlineExceeded(
                    "deadline already exceeded",
                    details={"remaining_ms": 0},
                )

    def _cache_supports_ttl(self) -> bool:
        """
        Determine whether the configured cache supports TTL semantics.

        - If cache exposes supports_ttl, respect it.
        - If not exposed, assume TTL-capable for backward compatibility.
        - Never let capability probing break the call path.
        """
        try:
            v = getattr(self._cache, "supports_ttl", None)
            if v is None:
                return True
            return bool(v)
        except Exception:
            return True

    async def _maybe_apply_deadline(
        self,
        awaitable: Awaitable[Any],
        ctx: Optional[OperationContext],
        *,
        enabled: bool,
    ) -> Any:
        """
        Apply DeadlinePolicy only when enabled.

        This preserves capability↔behavior alignment when a provider/adapter
        explicitly reports supports_deadline == False.
        """
        if not enabled:
            return await awaitable
        return await self._apply_deadline(awaitable, ctx)

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
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[Union[str, Dict[str, Any]]],
        caps: LLMCapabilities,
        ctx: Optional[OperationContext],
    ) -> str:
        """
        Construct a cache key for complete().

        Includes:
            - Model & system message hash
            - Messages fingerprint
            - Sampling params
            - Tool definitions and choice (stable JSON hash)
            - Stop sequences hash (to avoid embedding plaintext stops into cache keys)
            - Capabilities fingerprint
            - Tenant hash (when present)

        SECURITY NOTE:
            This method intentionally avoids embedding raw prompt content,
            system message text, tool definitions, or stop sequence text in
            cache keys. Instead it uses SHA-256 digests to reduce the risk of
            sensitive data exposure in cache backends.
        """
        caps_fingerprint_payload = {
            "server": caps.server,
            "version": caps.version,
            "model_family": caps.model_family,
            "max_context_length": caps.max_context_length,
            "protocol": caps.protocol,
            "supports_streaming": caps.supports_streaming,
            "supports_roles": caps.supports_roles,
            "supports_json_output": caps.supports_json_output,
            "supports_tools": caps.supports_tools,
            "supports_parallel_tool_calls": caps.supports_parallel_tool_calls,
            "supports_tool_choice": caps.supports_tool_choice,
            "max_tool_calls_per_turn": caps.max_tool_calls_per_turn,
            "idempotent_writes": caps.idempotent_writes,
            "supports_multi_tenant": caps.supports_multi_tenant,
            "supports_system_message": caps.supports_system_message,
            "supports_deadline": caps.supports_deadline,
            "supports_count_tokens": caps.supports_count_tokens,
            "supported_models": caps.supported_models,
        }
        caps_hash = hashlib.sha256(
            json.dumps(caps_fingerprint_payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

        # Hash tools/choice stably (do not embed plaintext in cache key)
        tools_json = json.dumps(tools, sort_keys=True, default=str) if tools else "none"
        choice_json = json.dumps(tool_choice, sort_keys=True, default=str) if tool_choice else "none"

        # Hash stop sequences (do not embed plaintext in cache key)
        stops_json = json.dumps(stop_sequences, sort_keys=True, default=str) if stop_sequences else "none"

        parts: Dict[str, str] = {
            "model": str(model or "default"),
            "system_hash": self._hash_str(system_message),
            "msgs": self._messages_fingerprint(messages),
            "max_toks": str(max_tokens) if max_tokens is not None else "none",
            "temp": repr(temperature) if temperature is not None else "none",
            "top_p": repr(top_p) if top_p is not None else "none",
            "freq_pen": repr(frequency_penalty) if frequency_penalty is not None else "none",
            "pres_pen": repr(presence_penalty) if presence_penalty is not None else "none",
            "stops": self._hash_str(stops_json),
            "tools": self._hash_str(tools_json),
            "tool_choice": self._hash_str(choice_json),
            "caps": caps_hash,
        }
        if ctx and ctx.tenant:
            th = self._tenant_hash(ctx.tenant)
            if th:
                parts["tenant"] = th
        raw = "|".join(f"{k}={v}" for k, v in sorted(parts.items()))
        return f"llm.complete:{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"

    async def _apply_deadline(
        self,
        awaitable: Awaitable[Any],
        ctx: Optional[OperationContext],
    ) -> Any:
        """
        Apply DeadlinePolicy and normalize asyncio.TimeoutError.

        All timeouts become DeadlineExceeded with DEADLINE_EXCEEDED semantics.
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
        enforce_deadline: bool,
    ) -> None:
        """
        Optional context-window preflight using count_tokens() if supported.

        Behavior:
            - Uses backend count_tokens implementation to estimate prompt size.
            - If max_tokens is provided and sum exceeds max_context_length:
                  raises BadRequest.
            - Never blocks the call on count_tokens errors.
        """
        if not caps.supports_count_tokens or caps.max_context_length <= 0:
            return
        if max_tokens is not None and max_tokens < 0:
            raise BadRequest("max_tokens must be >= 0")

        parts: List[str] = []
        if system_message:
            parts.append(f"system:{system_message}")
        for m in messages:
            parts.append(f"{m.get('role','')}:{m.get('content','')}")
        combined = "\n".join(parts)

        try:
            prompt_tokens = await self._maybe_apply_deadline(
                self._do_count_tokens(text=combined, model=model, ctx=ctx),
                ctx,
                enabled=enforce_deadline,
            )
        except NotSupported:
            return
        except LLMAdapterError:
            return

        if max_tokens is not None:
            total_possible = prompt_tokens + max(0, int(max_tokens))
            if total_possible > caps.max_context_length:
                raise BadRequest(
                    f"prompt tokens ({prompt_tokens}) + max_tokens ({max_tokens}) "
                    f"exceed max_context_length ({caps.max_context_length})"
                )

    def _gate_model_if_listed(self, *, model: Optional[str], caps: LLMCapabilities) -> None:
        """
        If capabilities enumerate supported_models, enforce membership.
        """
        if model and caps.supported_models and model not in caps.supported_models:
            raise BadRequest(f"model '{model}' is not supported by this adapter")

    # --- DRY gate wrappers (unary + streaming) -------------------------------

    async def _with_gates_unary(
        self,
        *,
        op: str,
        ctx: Optional[OperationContext],
        call: Callable[[], Awaitable[Any]],
        metric_extra: Mapping[str, Any] = None,
        after_success: Optional[Callable[[Any, Dict[str, Any]], None]] = None,
    ) -> Any:
        """
        Shared wrapper for unary operations.

        Applies:
            - circuit breaker allow/on_{success,error}
            - rate limiter acquire/release
            - metrics recording (timing + status)
            - optional after_success hook for extra metrics

        NOTE (capability↔behavior alignment):
            Deadline semantics are enforced by the operation itself after it
            evaluates capabilities.supports_deadline. This avoids applying
            deadlines for adapters that explicitly declare they do not support
            deadline enforcement.
        """
        metric_extra = dict(metric_extra or {})

        if not self._breaker.allow():
            # Match cross-SDK observability patterns: breaker open counter.
            try:
                self._metrics.counter(
                    component=self._component,
                    name="breaker_open_total",
                    value=1,
                    extra={"op": op},
                )
            except Exception:
                pass

            e = Unavailable("circuit open")
            code = e.code or type(e).__name__
            t0 = time.monotonic()
            self._record(op, t0, False, code=code, ctx=ctx, **metric_extra)
            raise e

        # Observe limiter backpressure in low-cardinality buckets.
        wait_t0 = time.monotonic()
        await self._limiter.acquire()
        waited_ms = (time.monotonic() - wait_t0) * 1000.0
        if waited_ms > 0:
            try:
                self._metrics.counter(
                    component=self._component,
                    name="limiter_wait_buckets_total",
                    value=1,
                    extra={"op": op, "bucket": self._bucket_wait_ms(waited_ms)},
                )
            except Exception:
                pass

        t0 = time.monotonic()
        try:
            result = await call()

            if after_success is not None:
                try:
                    after_success(result, metric_extra)
                except Exception:
                    # Metrics-only; ignore failures.
                    pass

            self._record(op, t0, True, ctx=ctx, **metric_extra)
            self._breaker.on_success()
            return result

        except LLMAdapterError as e:
            code = e.code or type(e).__name__
            self._record(op, t0, False, code=code, ctx=ctx, **metric_extra)
            self._breaker.on_error(e)
            raise

        except Exception as e:
            self._record(op, t0, False, code="UnhandledException", ctx=ctx, **metric_extra)
            self._breaker.on_error(e)
            raise

        finally:
            self._limiter.release()

    async def _with_gates_stream(
        self,
        *,
        op: str,
        ctx: Optional[OperationContext],
        # IMPORTANT: to reduce cognitive load and remove nonlocal toggles,
        # the factory returns (agen, deadline_on) explicitly.
        agen_factory: Callable[[], Awaitable[Tuple[AsyncIterator[LLMChunk], bool]]],
        metric_extra: Mapping[str, Any] = None,
    ) -> AsyncIterator[LLMChunk]:
        """
        Shared wrapper for streaming operations.

        Applies:
            - circuit breaker allow/on_{success,error}
            - rate limiter acquire/release
            - metrics for overall stream duration and outcome

        Optimization:
            - Deadline is checked every `stream_deadline_check_every_n_chunks`
              instead of on every single chunk to reduce overhead on hot paths.

        Backpressure:
            - Async iteration itself provides natural backpressure: chunks
              are only produced as fast as the consumer awaits them.
            - For more aggressive backpressure, supply a custom RateLimiter
              that enforces per-chunk limits in the adapter's _do_stream.

        Capability↔behavior alignment:
            - Deadline checks are enabled/disabled using the deadline_on boolean
              returned by agen_factory (derived from capabilities.supports_deadline).
        """
        metric_extra = dict(metric_extra or {})

        if not self._breaker.allow():
            # Match cross-SDK observability patterns: breaker open counter.
            try:
                self._metrics.counter(
                    component=self._component,
                    name="breaker_open_total",
                    value=1,
                    extra={"op": op},
                )
            except Exception:
                pass

            e = Unavailable("circuit open")
            code = e.code or type(e).__name__
            t0 = time.monotonic()
            self._record(op, t0, False, code=code, ctx=ctx, **metric_extra)
            raise e

        # Observe limiter backpressure in low-cardinality buckets.
        wait_t0 = time.monotonic()
        await self._limiter.acquire()
        waited_ms = (time.monotonic() - wait_t0) * 1000.0
        if waited_ms > 0:
            try:
                self._metrics.counter(
                    component=self._component,
                    name="limiter_wait_buckets_total",
                    value=1,
                    extra={"op": op, "bucket": self._bucket_wait_ms(waited_ms)},
                )
            except Exception:
                pass

        t0 = time.monotonic()
        check_n = self._stream_deadline_check_every_n_chunks

        async def _gen() -> AsyncIterator[LLMChunk]:
            chunk_count = 0
            tokens_total = 0
            saw_final = False
            agen: Optional[AsyncIterator[LLMChunk]] = None
            deadline_on: bool = True

            try:
                agen, deadline_on = await agen_factory()

                async for chunk in agen:
                    chunk_count += 1

                    # Enforce protocol surface: adapters must yield LLMChunk instances.
                    if not isinstance(chunk, LLMChunk):
                        raise Unavailable("stream yielded non-LLMChunk item")

                    if bool(getattr(chunk, "is_final", False)):
                        saw_final = True

                    # Robust usage tracking: update accumulated usage if present in this chunk
                    # This handles cases where usage is only in the final chunk or sent progressively.
                    usage = getattr(chunk, "usage_so_far", None)
                    if usage and isinstance(getattr(usage, "total_tokens", None), int):
                        # Assume usage is cumulative (so far); latest non-zero value wins
                        tokens_total = int(usage.total_tokens)

                    if deadline_on and check_n > 0 and (chunk_count % check_n) == 0:
                        self._preflight_deadline(ctx)

                    yield chunk

                metric_extra["chunks"] = chunk_count
                self._record(op, t0, True, ctx=ctx, **metric_extra)
                self._metrics.counter(
                    component=self._component,
                    name="stream_requests_total",
                    value=1,
                )
                self._metrics.counter(
                    component=self._component,
                    name="stream_chunks_total",
                    value=chunk_count,
                )
                if chunk_count > 0 and not saw_final:
                    # Observability-only: stream ended without any is_final=True chunk.
                    self._metrics.counter(
                        component=self._component,
                        name="stream_missing_final_chunk_total",
                        value=1,
                    )
                if tokens_total > 0:
                    # Total tokens for this stream (captured from usage_so_far).
                    self._metrics.counter(
                        component=self._component,
                        name="stream_tokens_total",
                        value=tokens_total,
                    )
                    self._metrics.counter(
                        component=self._component,
                        name="tokens_processed",
                        value=tokens_total,
                    )
                self._breaker.on_success()

            except asyncio.CancelledError as e:
                # Ensure cancellation does not leak underlying provider streams.
                metric_extra["chunks"] = chunk_count
                self._record(op, t0, False, code="CancelledError", ctx=ctx, **metric_extra)
                self._breaker.on_error(e)
                raise

            except LLMAdapterError as e:
                metric_extra["chunks"] = chunk_count
                code = e.code or type(e).__name__
                self._record(op, t0, False, code=code, ctx=ctx, **metric_extra)
                self._breaker.on_error(e)
                raise

            except Exception as e:
                metric_extra["chunks"] = chunk_count
                self._record(op, t0, False, code="UnhandledException", ctx=ctx, **metric_extra)
                self._breaker.on_error(e)
                raise

            finally:
                self._limiter.release()
                # Streaming cleanup: best-effort close of underlying async generator.
                if agen is not None:
                    close_fn = getattr(agen, "aclose", None)
                    if callable(close_fn):
                        try:
                            # Shield to avoid being interrupted during cleanup.
                            await asyncio.shield(close_fn())
                        except Exception:
                            pass

        return _gen()

    # --- Public API: capabilities / complete / stream / count_tokens / health -

    async def capabilities(self) -> LLMCapabilities:
        """
        Return adapter capabilities.

        SHOULD be fast and side-effect free; MAY call upstream discovery APIs.
        """
        t0 = time.monotonic()
        try:
            # Respect cache_ttl_s == 0 as a hard disable.
            use_cache = (self._cache_ttl_s > 0) and self._cache_supports_ttl()
            if use_cache:
                cached = await self._cache.get(self._caps_cache_key)
                if cached:
                    self._metrics.counter(
                        component=self._component,
                        name="cache_hits",
                        value=1,
                        extra={"op": "capabilities"},
                    )
                    self._record("capabilities", t0, True)
                    # Cache safety: return a defensive copy to prevent mutation bleed.
                    return self._safe_deepcopy(cached)
            caps = await self._do_capabilities()
            if use_cache:
                try:
                    # Cache safety: store a defensive copy to prevent mutation bleed.
                    await self._cache.set(self._caps_cache_key, self._safe_deepcopy(caps), ttl_s=self._cache_ttl_s)
                except Exception:
                    pass
            self._record("capabilities", t0, True)
            return caps
        except LLMAdapterError as e:
            code = e.code or type(e).__name__
            self._record("capabilities", t0, False, code=code)
            raise
        except Exception as e:
            self._record("capabilities", t0, False, code="UNAVAILABLE")
            raise Unavailable("capabilities fetch failed") from e

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
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        ctx: Optional[OperationContext] = None,
    ) -> LLMCompletion:
        """
        Execute a full LLM completion with validation, policy hooks, and metrics.

        See LLMProtocolV1.complete for the detailed contract.
        """
        self._validate_messages(messages)
        self._validate_message_content_serializable(messages)
        self._validate_sampling_params(
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
        self._validate_tools(tools)
        self._validate_tool_choice(tool_choice)
        self._validate_stop_sequences(stop_sequences)

        if max_tokens is not None and int(max_tokens) < 0:
            raise BadRequest("max_tokens must be >= 0")

        async def _call() -> LLMCompletion:
            caps = await self.capabilities()
            self._gate_model_if_listed(model=model, caps=caps)

            # Capability↔behavior alignment: system_message gating.
            if system_message is not None and system_message != "" and not caps.supports_system_message:
                raise NotSupported("system_message is not supported by this adapter")

            # Capability↔behavior alignment: deadline enforcement.
            enforce_deadline = bool(caps.supports_deadline)
            if enforce_deadline:
                self._preflight_deadline(ctx)

            if (tools or tool_choice) and not caps.supports_tools:
                raise NotSupported("tools are not supported by this adapter")

            await self._preflight_context_window_if_supported(
                messages=messages,
                system_message=system_message,
                max_tokens=max_tokens,
                model=model,
                ctx=ctx,
                caps=caps,
                enforce_deadline=enforce_deadline,
            )

            # Respect cache_ttl_s == 0 as a hard disable.
            use_cache = (self._cache_ttl_s > 0) and self._cache_supports_ttl()

            cache_key: Optional[str] = None
            if use_cache:
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
                    tools=tools,
                    tool_choice=tool_choice,
                    caps=caps,
                    ctx=ctx,
                )
                cached = await self._cache.get(cache_key)
                if cached is not None:
                    self._metrics.counter(
                        component=self._component,
                        name="cache_hits",
                        value=1,
                    )
                    # Cache safety: return a defensive copy to prevent mutation bleed.
                    return self._safe_deepcopy(cached)  # type: ignore[return-value]

            result = await self._maybe_apply_deadline(
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
                    tools=tools,
                    tool_choice=tool_choice,
                    ctx=ctx,
                ),
                ctx,
                enabled=enforce_deadline,
            )

            if use_cache and cache_key is not None:
                try:
                    # Cache safety: store a defensive copy to prevent mutation bleed.
                    await self._cache.set(cache_key, self._safe_deepcopy(result), ttl_s=self._cache_ttl_s)
                except Exception:
                    pass

            return result

        def _after_success(result: LLMCompletion, metric_extra: Dict[str, Any]) -> None:
            if self._tag_model_in_metrics and getattr(result, "model", None):
                metric_extra.setdefault("model", result.model)
            self._metrics.counter(
                component=self._component,
                name="requests_total",
                value=1,
            )
            usage = getattr(result, "usage", None)
            if usage and isinstance(getattr(usage, "total_tokens", None), int):
                self._metrics.counter(
                    component=self._component,
                    name="tokens_processed",
                    value=int(usage.total_tokens),
                )

        return await self._with_gates_unary(
            op="complete",
            ctx=ctx,
            call=_call,
            metric_extra={},
            after_success=_after_success,
        )

    async def stream(
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
        Streaming completion with full sampling parity.

        Accepts:
            - temperature, top_p
            - frequency_penalty, presence_penalty
            - stop_sequences
            - tools, tool_choice
        and forwards them to _do_stream. This aligns streaming with complete().
        """
        self._validate_messages(messages)
        self._validate_message_content_serializable(messages)
        self._validate_sampling_params(
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
        self._validate_tools(tools)
        self._validate_tool_choice(tool_choice)
        self._validate_stop_sequences(stop_sequences)

        if max_tokens is not None and int(max_tokens) < 0:
            raise BadRequest("max_tokens must be >= 0")

        metric_extra: Dict[str, Any] = {}
        if self._tag_model_in_metrics and model:
            metric_extra["model"] = model

        async def agen_factory() -> Tuple[AsyncIterator[LLMChunk], bool]:
            """
            Create the underlying adapter stream generator and return a boolean
            indicating whether deadline semantics are enabled for this stream.

            This removes the need for nonlocal toggles and keeps capability↔behavior
            alignment explicit and easy to reason about.
            """
            caps = await self.capabilities()
            self._gate_model_if_listed(model=model, caps=caps)

            deadline_on = bool(caps.supports_deadline)

            if not caps.supports_streaming:
                raise NotSupported("stream is not supported by this adapter")

            if system_message is not None and system_message != "" and not caps.supports_system_message:
                raise NotSupported("system_message is not supported by this adapter")

            if deadline_on:
                self._preflight_deadline(ctx)

            if (tools or tool_choice) and not caps.supports_tools:
                raise NotSupported("tools are not supported by this adapter")

            await self._preflight_context_window_if_supported(
                messages=messages,
                system_message=system_message,
                max_tokens=max_tokens,
                model=model,
                ctx=ctx,
                caps=caps,
                enforce_deadline=deadline_on,
            )

            agen = self._do_stream(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop_sequences=stop_sequences,
                model=model,
                system_message=system_message,
                tools=tools,
                tool_choice=tool_choice,
                ctx=ctx,
            )
            return agen, deadline_on

        generator = await self._with_gates_stream(
            op="stream",
            ctx=ctx,
            agen_factory=agen_factory,
            metric_extra=metric_extra,
        )
        async for chunk in generator:
            yield chunk

    async def count_tokens(
        self,
        text: str,
        *,
        model: Optional[str] = None,
        ctx: Optional[OperationContext] = None,
    ) -> int:
        """
        Count tokens for the given text/model pair with full instrumentation.

        Honors:
            - capabilities.supported_models (if enumerated)
            - capabilities.supports_count_tokens
            - ctx.deadline_ms via DeadlinePolicy when capabilities.supports_deadline is True
        """
        if not isinstance(text, str):
            raise BadRequest("text must be a string")

        t0 = time.monotonic()
        extra: Dict[str, Any] = {"text_length": len(text)}
        if self._tag_model_in_metrics and model:
            extra["model"] = model
        try:
            caps = await self.capabilities()
            if model and caps.supported_models and model not in caps.supported_models:
                raise BadRequest(f"model '{model}' is not supported by this adapter")
            if not caps.supports_count_tokens:
                raise NotSupported("count_tokens is not supported by this adapter")

            enforce_deadline = bool(caps.supports_deadline)
            if enforce_deadline:
                self._preflight_deadline(ctx)

            result = await self._maybe_apply_deadline(
                self._do_count_tokens(text=text, model=model, ctx=ctx),
                ctx,
                enabled=enforce_deadline,
            )

            self._record(
                "count_tokens",
                t0,
                True,
                ctx=ctx,
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
                code=(e.code or type(e).__name__),
                ctx=ctx,
                **extra,
            )
            raise

        except Exception as e:
            self._record(
                "count_tokens",
                t0,
                False,
                code="UnhandledException",
                ctx=ctx,
                **extra,
            )
            raise

    async def health(self, *, ctx: Optional[OperationContext] = None) -> Mapping[str, Any]:
        """
        Health check endpoint with normalized response shape.

        Returns:
            {
                "ok": bool,
                "server": str,
                "version": str,
            }

        Implementations may include additional keys in _do_health; they are
        normalized down by this wrapper for callers.
        """
        t0 = time.monotonic()
        try:
            caps = await self.capabilities()
            enforce_deadline = bool(caps.supports_deadline)
            if enforce_deadline:
                self._preflight_deadline(ctx)

            h = await self._maybe_apply_deadline(
                self._do_health(ctx=ctx),
                ctx,
                enabled=enforce_deadline,
            )
            self._record("health", t0, True, ctx=ctx)
            return {
                "ok": bool(h.get("ok", True)),
                "server": str(h.get("server", "")),
                "version": str(h.get("version", "")),
            }
        except LLMAdapterError as e:
            self._record("health", t0, False, code=(e.code or type(e).__name__), ctx=ctx)
            raise
        except Exception as e:
            self._record("health", t0, False, code="UnhandledException", ctx=ctx)
            raise Unavailable("health check failed") from e

    # --- backend hooks -------------------------------------------------------

    async def _do_capabilities(self) -> LLMCapabilities:
        """
        Return adapter-specific capabilities.

        Must be implemented by concrete adapters.
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
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        ctx: Optional[OperationContext] = None,
    ) -> LLMCompletion:
        """
        Backend implementation of complete().

        Base has already:
            - validated messages and params
            - applied preflight checks
        """
        raise NotImplementedError

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
        Backend implementation of stream().

        Must yield LLMChunk instances.
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
        Backend implementation of count_tokens().
        """
        raise NotImplementedError

    async def _do_health(self, *, ctx: Optional[OperationContext] = None) -> Mapping[str, Any]:
        """
        Backend implementation of health().

        Should be lightweight and resilient; callers rely on this for readiness.
        """
        raise NotImplementedError


# =============================================================================
# Wire-Level Helpers (canonical envelopes)
# =============================================================================


def _ctx_from_wire(ctx_dict: Mapping[str, Any]) -> OperationContext:
    """
    Convert wire-level ctx mapping into OperationContext.

    Unknown keys are ignored for forward compatibility.
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


def _ensure_json_serializable(payload: Any, *, field: str) -> None:
    """
    Ensure a payload is JSON-serializable.

    Wire contract requires canonical JSON; if an adapter returns objects that
    cannot be serialized, hard-fail at the boundary with a stable error.

    SECURITY NOTE:
        This intentionally does NOT include the payload in error messages.
        Payload values may contain secrets/PII.
    """
    try:
        json.dumps(payload)
    except (TypeError, ValueError) as e:
        raise Unavailable(f"{field} is not JSON-serializable") from e


def _success_to_wire(result: Any, ms: float) -> Dict[str, Any]:
    """
    Wrap successful (unary) results into canonical success envelope.

    Cross-SDK alignment:
      - Unary success uses code="OK"
      - Streaming chunks use code="STREAMING" (see _chunk_to_wire)
    """
    if hasattr(result, "__dataclass_fields__"):
        payload = asdict(result)
    else:
        payload = result

    _ensure_json_serializable(payload, field="result")

    return {
        "ok": True,
        "code": "OK",
        "ms": ms,
        "result": payload,
    }


def _chunk_to_wire(chunk: LLMChunk, ms: float) -> Dict[str, Any]:
    """
    Wrap an LLMChunk into the canonical streaming envelope shape.

    Cross-SDK alignment:
      - Streaming frames use code="STREAMING"
      - Payload lives under "chunk"
    """
    if hasattr(chunk, "__dataclass_fields__"):
        payload = asdict(chunk)
    else:
        # Fallback for non-dataclass implementations (kept defensive)
        payload = {
            "text": getattr(chunk, "text", ""),
            "is_final": bool(getattr(chunk, "is_final", False)),
            "model": getattr(chunk, "model", None),
            "usage_so_far": (
                asdict(chunk.usage_so_far)
                if getattr(chunk, "usage_so_far", None) is not None
                and hasattr(chunk.usage_so_far, "__dataclass_fields__")
                else getattr(chunk, "usage_so_far", None)
            ),
            "tool_calls": [
                asdict(tc) if hasattr(tc, "__dataclass_fields__") else tc
                for tc in getattr(chunk, "tool_calls", []) or []
            ],
        }

    _ensure_json_serializable(payload, field="chunk")

    return {
        "ok": True,
        "code": "STREAMING",
        "ms": ms,
        "chunk": payload,
    }


def _error_to_wire(e: Exception, ms: float) -> Dict[str, Any]:
    """
    Normalize exceptions into canonical error envelopes.

    Cross-SDK alignment:
      - Always includes retry_after_ms and details (nullable)
      - Always includes ms

    Security hardening:
      - For unknown/unhandled exceptions, return a stable, non-leaky message.

    Operational triage:
      - For unknown/unhandled exceptions, include a SIEM-safe details payload
        with error_type to support debugging without leaking internals.
    """
    if isinstance(e, LLMAdapterError):
        details = dict(e.details or {})
        if e.throttle_scope is not None:
            details.setdefault("throttle_scope", e.throttle_scope)
        if e.suggested_token_reduction is not None:
            details.setdefault("suggested_token_reduction", e.suggested_token_reduction)
        return {
            "ok": False,
            "code": (e.code or type(e).__name__.upper()),
            "error": type(e).__name__,
            "message": e.message,
            "retry_after_ms": e.retry_after_ms,
            "details": details or None,
            "ms": ms,
        }

    return {
        "ok": False,
        "code": "UNAVAILABLE",
        "error": type(e).__name__,
        # Unknown exception: stable message; do not echo raw exception text (may leak internals).
        "message": "internal error",
        "retry_after_ms": None,
        "details": None,
        "ms": ms,
    }


class WireLLMHandler:
    """
    Reference wire adapter for LLMProtocolV1.

    Transport-agnostic: plug into HTTP, gRPC, WebSocket, etc.

    Supported unary ops:
        - llm.capabilities
        - llm.complete
        - llm.count_tokens
        - llm.health

    Streaming:
        - llm.stream (via handle_stream)

    Error propagation:
        For streaming, this handler represents adapter exceptions as a final
        JSON envelope on the stream. For transports with a native error channel
        (e.g., gRPC status codes), callers may wish to translate that final
        envelope into transport-specific errors instead of forwarding it as
        data.
    """

    def __init__(self, adapter: LLMProtocolV1):
        self._adapter = adapter

    @staticmethod
    def _require_mapping(obj: Any, *, field: str) -> Mapping[str, Any]:
        if not isinstance(obj, Mapping):
            raise BadRequest(f"{field} must be an object")
        return obj

    @staticmethod
    def _require_key(envelope: Mapping[str, Any], key: str) -> Any:
        if key not in envelope:
            raise BadRequest(f"missing required '{key}'")
        return envelope.get(key)

    async def handle(self, envelope: Any) -> Dict[str, Any]:
        """
        Handle unary LLM operations via JSON envelope.

        Wire strictness:
            - envelope MUST be an object
            - MUST include: op, ctx, args
            - ctx MUST be an object
            - args MUST be an object
        """
        t0 = time.monotonic()
        try:
            env = self._require_mapping(envelope, field="envelope")

            op = env.get("op")
            if not isinstance(op, str):
                raise BadRequest("missing or invalid 'op'")

            ctx_raw = self._require_key(env, "ctx")
            args_raw = self._require_key(env, "args")
            ctx_map = self._require_mapping(ctx_raw, field="ctx")
            args = self._require_mapping(args_raw, field="args")

            ctx = _ctx_from_wire(ctx_map)

            if op == "llm.capabilities":
                res = await self._adapter.capabilities()
                return _success_to_wire(res, (time.monotonic() - t0) * 1000.0)

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
                    tools=args.get("tools"),
                    tool_choice=args.get("tool_choice"),
                    ctx=ctx,
                )
                return _success_to_wire(res, (time.monotonic() - t0) * 1000.0)

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

            # llm.stream is handled exclusively via handle_stream.
            raise NotSupported(f"unknown or non-unary operation '{op}'")

        except Exception as e:
            ms = (time.monotonic() - t0) * 1000.0
            return _error_to_wire(e, ms)

    async def handle_stream(self, envelope: Any) -> AsyncIterator[Dict[str, Any]]:
        """
        Handle streaming llm.stream requests.

        Expects:
            op: "llm.stream"
            ctx: { ... }   (REQUIRED)
            args: { ... }  (REQUIRED)

        Note:
            This method forwards chunks as they are produced by the adapter.
            For JSON-envelope style transports, an error is emitted as a final
            envelope on the stream. Other transports (e.g. raw gRPC streams)
            may choose to map this to a terminal stream error instead.
        """
        t0 = time.monotonic()
        try:
            env = self._require_mapping(envelope, field="envelope")
        except Exception as e:
            yield _error_to_wire(e, 0.0)
            return

        op = env.get("op")
        if op != "llm.stream":
            yield _error_to_wire(BadRequest("op must be 'llm.stream' for streaming"), 0.0)
            return

        try:
            ctx_raw = self._require_key(env, "ctx")
            args_raw = self._require_key(env, "args")
            ctx_map = self._require_mapping(ctx_raw, field="ctx")
            args = self._require_mapping(args_raw, field="args")
        except Exception as e:
            yield _error_to_wire(e, 0.0)
            return

        ctx = _ctx_from_wire(ctx_map)

        try:
            agen = self._adapter.stream(
                messages=args.get("messages") or [],
                max_tokens=args.get("max_tokens"),
                temperature=args.get("temperature"),
                top_p=args.get("top_p"),
                frequency_penalty=args.get("frequency_penalty"),
                presence_penalty=args.get("presence_penalty"),
                stop_sequences=args.get("stop_sequences"),
                model=args.get("model"),
                system_message=args.get("system_message"),
                tools=args.get("tools"),
                tool_choice=args.get("tool_choice"),
                ctx=ctx,
            )
            async for chunk in agen:
                ms = (time.monotonic() - t0) * 1000.0
                yield _chunk_to_wire(chunk, ms)
        except Exception as e:
            ms = (time.monotonic() - t0) * 1000.0
            # Emit a final error envelope on the stream channel.
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
    "ToolCallFunction",
    "ToolCall",
    "LLMCompletion",
    "LLMChunk",
    "DeadlinePolicy",
    "CircuitBreaker",
    "Cache",
    "TTLAwareCache",
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
    "WireLLMHandler",
    "_ctx_from_wire",
    "_error_to_wire",
    "_success_to_wire",
    "_chunk_to_wire",
]
