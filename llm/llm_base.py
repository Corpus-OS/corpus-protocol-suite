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

Versioning
----------
Follow SemVer against LLM_PROTOCOL_VERSION. Minor versions are strictly additive.
- Patch (x.y.Z): Editorial clarifications, non-breaking fixes
- Minor (x.Y.z): New optional parameters, capabilities, or methods
- Major (X.y.z): Breaking changes to signatures or behavior
"""

from __future__ import annotations
import time
import hashlib
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    AsyncIterator,
    runtime_checkable,
)

LLM_PROTOCOL_VERSION = "1.1.0"

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
        throttle_scope: Scope of throttling ("tenant", "model", "global")
        suggested_token_reduction: Percentage reduction suggestion for quota errors
        details: Additional context-specific error details
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

class BadRequest(LLMAdapterError):
    """Client sent an invalid request (malformed messages, invalid parameters)."""
    pass

class AuthError(LLMAdapterError):
    """Authentication or authorization failed (invalid credentials, permissions)."""
    pass

class ResourceExhausted(LLMAdapterError):
    """Quota, rate limit, or resource constraints exceeded."""
    pass

class TransientNetwork(LLMAdapterError):
    """Transient network failure that may succeed on retry."""
    pass

class Unavailable(LLMAdapterError):
    """Service is temporarily unavailable or overloaded."""
    pass

class NotSupported(LLMAdapterError):
    """Requested operation or parameter is not supported by this adapter."""
    pass

class ModelOverloaded(LLMAdapterError):
    """Specific model is currently overloaded and cannot handle requests."""
    pass

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
    supports_system_message: bool = True  # Explicit system message support

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
            BadRequest: For invalid parameters or malformed messages
            AuthError: For authentication or authorization failures
            ResourceExhausted: For quota or rate limit exceeded
            ModelOverloaded: For model-specific capacity issues
            TransientNetwork: For retryable network failures
            Unavailable: For service unavailable errors
            NotSupported: For unsupported parameters or operations
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
            BadRequest: For invalid parameters or malformed messages
            AuthError: For authentication or authorization failures  
            ResourceExhausted: For quota or rate limit exceeded
            ModelOverloaded: For model-specific capacity issues
            TransientNetwork: For retryable network failures
            Unavailable: For service unavailable errors
            NotSupported: For unsupported parameters or operations
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
    
    Example:
        class OpenAIAdapter(BaseLLMAdapter):
            async def _do_complete(self, *, messages, max_tokens, temperature, ctx):
                # OpenAI-specific implementation
                response = await openai_client.chat.completions.create(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return LLMCompletion(
                    text=response.choices[0].message.content,
                    model=response.model,
                    model_family="gpt-4",
                    usage=TokenUsage(...),
                    finish_reason=response.choices[0].finish_reason
                )
    """

    _component = "llm"

    def __init__(self, *, metrics: Optional[MetricsSink] = None) -> None:
        """
        Initialize the LLM adapter with metrics instrumentation.
        
        Args:
            metrics: Metrics sink for operational monitoring. Uses NoopMetrics if None.
        """
        self._metrics: MetricsSink = metrics or NoopMetrics()

    # --- internal helpers (validation and instrumentation) ---

    @staticmethod
    def _validate_messages(messages: List[Mapping[str,str]]) -> None:
        """Validate that messages list conforms to required format."""
        if not messages or not all(isinstance(m, Mapping) and "role" in m and "content" in m for m in messages):
            raise BadRequest("messages must be a non-empty list of {role, content} mappings")

    @staticmethod
    def _tenant_hash(t: Optional[str]) -> Optional[str]:
        """Create privacy-preserving hash of tenant identifier for metrics."""
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
        **extra: Any
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
                extra=x or None
            )
        except Exception:
            # Never let metrics recording break the operation
            pass

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
        Execute a complete LLM conversation with validation and metrics.
        
        See LLMProtocolV1.complete for full documentation.
        """
        self._validate_messages(messages)
        t0 = time.monotonic()
        try:
            result = await self._do_complete(
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
            )
            self._record("complete", t0, True, ctx=ctx)
            return result
        except LLMAdapterError as e:
            self._record("complete", t0, False, code=type(e).__name__, ctx=ctx)
            raise

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
        Execute an LLM conversation with streaming and metrics.
        
        See LLMProtocolV1.stream for full documentation.
        """
        self._validate_messages(messages)
        t0 = time.monotonic()
        try:
            async for chunk in self._do_stream(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                model=model,
                system_message=system_message,
                ctx=ctx,
            ):
                yield chunk
            self._record("stream", t0, True, ctx=ctx)
        except LLMAdapterError as e:
            self._record("stream", t0, False, code=type(e).__name__, ctx=ctx)
            raise

    async def count_tokens(
        self,
        text: str,
        *,
        model: Optional[str] = None,
        ctx: Optional[OperationContext] = None,
    ) -> int:
        """Count tokens in text with metrics instrumentation."""
        return await self._do_count_tokens(text=text, model=model, ctx=ctx)

    async def health(self, *, ctx: Optional[OperationContext] = None) -> Mapping[str, Any]:
        """Check health status with metrics instrumentation."""
        h = await self._do_health(ctx=ctx)
        return {
            "ok": bool(h.get("ok", True)),
            "server": str(h.get("server", "")),
            "version": str(h.get("version", "")),
        }

    # --- hooks to implement per backend (override these) ---

    async def _do_capabilities(self) -> LLMCapabilities:
        """Implement to return adapter-specific capabilities."""
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
        """Implement complete LLM conversation with validated inputs."""
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
        """Implement streaming LLM conversation with validated inputs."""
        raise NotImplementedError

    async def _do_count_tokens(
        self,
        text: str,
        *,
        model: Optional[str] = None,
        ctx: Optional[OperationContext] = None,
    ) -> int:
        """Implement token counting for the specified model."""
        raise NotImplementedError

    async def _do_health(self, *, ctx: Optional[OperationContext] = None) -> Mapping[str, Any]:
        """Implement health check for the LLM backend."""
        raise NotImplementedError


__all__ = [
    "LLM_PROTOCOL_VERSION",
    "LLMAdapterError",
    "BadRequest",
    "AuthError",
    "ResourceExhausted",
    "TransientNetwork",
    "Unavailable",
    "NotSupported",
    "ModelOverloaded",
    "OperationContext",
    "TokenUsage",
    "LLMCompletion",
    "LLMChunk",
    "LLMCapabilities",
    "LLMProtocolV1",
    "BaseLLMAdapter",
]
