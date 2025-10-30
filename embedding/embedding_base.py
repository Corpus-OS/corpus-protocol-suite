# adapter_sdk/embedding_base.py
# SPDX-License-Identifier: Apache-2.0
"""
Adapter SDK — Embedding Protocol V1 (public contract + production-grade base)

Purpose
-------
A stable, vendor-neutral API for text embedding generation — with structured errors,
caching strategies, rate limiting, and production observability.

This protocol enables seamless integration with any embedding model provider while
maintaining production-grade security, performance monitoring, and operational rigor
for high-scale text embedding workloads.

Design Philosophy
-----------------
- Minimal surface area: Core embedding operations only, no vendor-specific extensions
- Async-first: All operations are non-blocking for high-concurrency environments
- Production hardened: Built-in caching, circuit breaking, backpressure, and metrics
- Extensible: Capability discovery allows for model-specific features
- Performance optimized: Built-in caching strategies for embedding generation

Deliberate Non-Goals
--------------------
- No text preprocessing, tokenization, or chunking strategies
- No model training, fine-tuning, or version management
- No vendor-specific model architectures or optimizations
- No client-side embedding post-processing or normalization

Those behaviors live in the text processing and model management layers.

Versioning
----------
Follow SemVer against EMBEDDING_PROTOCOL_VERSION. Minor versions are strictly additive.
- Patch (x.y.Z): Editorial clarifications, non-breaking fixes
- Minor (x.Y.z): New optional parameters, capabilities, or methods  
- Major (X.y.z): Breaking changes to signatures or behavior
"""

from __future__ import annotations
import time
import hashlib
from dataclasses import dataclass
from typing import (
    Any, Dict, List, Mapping, Optional, Protocol, Tuple, Iterable,
    runtime_checkable, AsyncIterator, Union
)

EMBEDDING_PROTOCOL_VERSION = "1.0.0"

# =============================================================================
# Core Type Definitions
# =============================================================================

@dataclass(frozen=True)
class EmbeddingVector:
    """
    A single embedding vector with metadata.
    
    Attributes:
        vector: The embedding vector as a list of floats
        text: The source text that was embedded
        model: Model used to generate the embedding
        dimensions: Vector dimensions
    """
    vector: List[float]
    text: str
    model: str
    dimensions: int

@dataclass(frozen=True)
class EmbeddingResult:
    """
    Result from embedding generation operations.
    
    Attributes:
        embeddings: List of generated embedding vectors
        model: Model used for generation
        total_tokens: Total tokens processed (if available)
        processing_time_ms: Time taken to generate embeddings
    """
    embeddings: List[EmbeddingVector]
    model: str
    total_tokens: Optional[int] = None
    processing_time_ms: Optional[float] = None

@dataclass(frozen=True)
class EmbeddingBatch:
    """
    Batch of texts for embedding generation.
    
    Attributes:
        texts: List of texts to embed
        model: Target model for embedding generation
        truncate: Whether to truncate long texts (True) or error (False)
        normalize: Whether to normalize output vectors to unit length
    """
    texts: List[str]
    model: str
    truncate: bool = True
    normalize: bool = False

# =============================================================================
# Normalized Errors (with retry hints and operational guidance)
# =============================================================================

class EmbeddingAdapterError(Exception):
    """
    Base exception for all embedding adapter errors.
    
    Provides structured error information including retry guidance, resource limits,
    and operational suggestions for callers to handle failures gracefully.
    
    Attributes:
        message: Human-readable error description
        code: Machine-readable error code for programmatic handling
        retry_after_ms: Suggested delay before retry (None if not retryable)
        resource_scope: Scope of resource limitation ("model", "token_limit", "rate_limit")
        suggested_batch_reduction: Percentage reduction suggestion for batch size
        details: Additional context-specific error details
    """
    def __init__(
        self,
        message: str = "",
        *,
        code: Optional[str] = None,
        retry_after_ms: Optional[int] = None,
        resource_scope: Optional[str] = None,
        suggested_batch_reduction: Optional[int] = None,
        details: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.retry_after_ms = retry_after_ms
        self.resource_scope = resource_scope
        self.suggested_batch_reduction = suggested_batch_reduction
        self.details = dict(details or {})

    def asdict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization and logging."""
        return {
            "message": self.message,
            "code": self.code,
            "retry_after_ms": self.retry_after_ms,
            "resource_scope": self.resource_scope,
            "suggested_batch_reduction": self.suggested_batch_reduction,
            "details": {k: self.details[k] for k in sorted(self.details)},
        }

class BadRequest(EmbeddingAdapterError):
    """Client sent an invalid request (malformed texts, invalid parameters)."""
    pass

class AuthError(EmbeddingAdapterError):
    """Authentication or authorization failed (invalid credentials, permissions)."""
    pass

class ResourceExhausted(EmbeddingAdapterError):
    """Quota, rate limit, or resource constraints exceeded."""
    pass

class TextTooLong(EmbeddingAdapterError):
    """Input text exceeds model's maximum context length."""
    pass

class ModelNotAvailable(EmbeddingAdapterError):
    """Requested embedding model is not available."""
    pass

class TransientNetwork(EmbeddingAdapterError):
    """Transient network failure that may succeed on retry."""
    pass

class Unavailable(EmbeddingAdapterError):
    """Service is temporarily unavailable or overloaded."""
    pass

class NotSupported(EmbeddingAdapterError):
    """Requested operation or parameter is not supported."""
    pass

# =============================================================================
# Context (used for deadlines, identity, SIEM-safe metrics)
# =============================================================================

@dataclass(frozen=True)
class OperationContext:
    """
    Context for embedding operations providing tracing, deadlines, and multi-tenant isolation.
    
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
    deadline_ms: Optional[int] = None
    traceparent: Optional[str] = None
    tenant: Optional[str] = None
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
            component: Component name (e.g., "embedding")
            op: Operation name (e.g., "embed", "embed_batch")
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
            component: Component name (e.g., "embedding") 
            name: Counter name (e.g., "texts_embedded", "tokens_processed")
            value: Increment value
            extra: Additional low-cardinality dimensions
        """
        ...

class NoopMetrics:
    """No-operation metrics sink for testing or when metrics are disabled."""
    def observe(self, **_: Any) -> None: ...
    def counter(self, **_: Any) -> None: ...

# =============================================================================
# Capabilities (dynamic discovery for routing and planning)
# =============================================================================

@dataclass(frozen=True)
class EmbeddingCapabilities:
    """
    Describes the capabilities and limitations of an embedding adapter implementation.
    
    Used by routing layers for intelligent model selection, request planning,
    and feature compatibility checking across different embedding providers.
    
    Attributes:
        server: Backend server identifier (e.g., "openai", "cohere", "huggingface")
        version: Backend server version string
        supported_models: Supported embedding model names
        max_batch_size: Maximum texts per batch operation
        max_text_length: Maximum characters per text input
        max_dimensions: Maximum vector dimensions supported
        supports_normalization: Whether vector normalization is supported
        supports_truncation: Whether text truncation is supported
        supports_token_counting: Whether token counting is available
        idempotent_operations: Whether operations are idempotent with idempotency_key
        supports_multi_tenant: Whether multi-tenant isolation is supported
    """
    server: str
    version: str
    supported_models: Tuple[str, ...]
    max_batch_size: Optional[int] = None
    max_text_length: Optional[int] = None
    max_dimensions: Optional[int] = None
    supports_normalization: bool = False
    supports_truncation: bool = True
    supports_token_counting: bool = False
    idempotent_operations: bool = False
    supports_multi_tenant: bool = False

# =============================================================================
# Operation Specifications
# =============================================================================

@dataclass(frozen=True)
class EmbedSpec:
    """
    Specification for single text embedding generation.
    
    Attributes:
        text: Text to convert to embedding
        model: Target model for embedding generation
        truncate: Whether to truncate long texts (True) or error (False)
        normalize: Whether to normalize output vector to unit length
    """
    text: str
    model: str
    truncate: bool = True
    normalize: bool = False

@dataclass(frozen=True)
class BatchEmbedSpec:
    """
    Specification for batch text embedding generation.
    
    Attributes:
        texts: List of texts to embed
        model: Target model for embedding generation
        truncate: Whether to truncate long texts (True) or error (False)
        normalize: Whether to normalize output vectors to unit length
    """
    texts: List[str]
    model: str
    truncate: bool = True
    normalize: bool = False

# =============================================================================
# Operation Results
# =============================================================================

@dataclass
class EmbedResult:
    """
    Result from single text embedding generation.
    
    Attributes:
        embedding: Generated embedding vector
        model: Model used for generation
        text: Original input text
        tokens_used: Number of tokens processed (if available)
        truncated: Whether input text was truncated
    """
    embedding: EmbeddingVector
    model: str
    text: str
    tokens_used: Optional[int] = None
    truncated: bool = False

@dataclass
class BatchEmbedResult:
    """
    Result from batch text embedding generation.
    
    Attributes:
        embeddings: List of generated embedding vectors
        model: Model used for generation
        total_texts: Total number of texts processed
        total_tokens: Total tokens processed (if available)
        failed_texts: List of texts that failed to process with error details
    """
    embeddings: List[EmbeddingVector]
    model: str
    total_texts: int
    total_tokens: Optional[int] = None
    failed_texts: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.failed_texts is None:
            self.failed_texts = []

# =============================================================================
# Stable Protocol Interface (async, versioned contract)
# =============================================================================

@runtime_checkable
class EmbeddingProtocolV1(Protocol):
    """
    Protocol defining the Embedding Protocol V1 interface.
    
    Implement this protocol to create compatible embedding adapters. All methods are async
    and designed for high-concurrency environments. The protocol is runtime-checkable
    for dynamic adapter validation.
    """

    async def capabilities(self) -> EmbeddingCapabilities:
        """
        Get the capabilities of this embedding adapter.
        
        Returns:
            EmbeddingCapabilities: Description of supported features and limitations
            
        Note:
            This method is async to support dynamic capability discovery in
            distributed systems where capabilities may change or require
            network calls to determine.
        """
        ...

    async def embed(
        self,
        spec: EmbedSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> EmbedResult:
        """
        Generate embedding for a single text.
        
        Args:
            spec: Embedding specification including text and model parameters
            ctx: Operation context for tracing, deadlines, and multi-tenancy
            
        Returns:
            EmbedResult: Single embedding result with vector and metadata
            
        Raises:
            BadRequest: For invalid text or malformed parameters
            AuthError: For authentication or authorization failures
            ResourceExhausted: For quota or rate limit exceeded
            TextTooLong: If text exceeds model's maximum length without truncation
            ModelNotAvailable: If requested model is not available
            NotSupported: If normalization or other features are not supported
            TransientNetwork: For retryable network failures
            Unavailable: For service unavailable errors
        """
        ...

    async def embed_batch(
        self,
        spec: BatchEmbedSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> BatchEmbedResult:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            spec: Batch embedding specification including texts and parameters
            ctx: Operation context for tracing, deadlines, and multi-tenancy
            
        Returns:
            BatchEmbedResult: Batch embedding results with success/failure details
            
        Raises:
            BadRequest: For invalid texts or malformed parameters
            AuthError: For authentication or authorization failures
            ResourceExhausted: For quota or rate limit exceeded
            TextTooLong: If any text exceeds model's maximum length without truncation
            ModelNotAvailable: If requested model is not available
            NotSupported: If batch operations are not supported
            TransientNetwork: For retryable network failures
            Unavailable: For service unavailable errors
        """
        ...

    async def count_tokens(
        self,
        text: str,
        model: str,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> int:
        """
        Count tokens in text for a specific model.
        
        Args:
            text: Text to count tokens for
            model: Model to use for tokenization
            ctx: Operation context for tracing and multi-tenancy
            
        Returns:
            int: Number of tokens in the text according to model's tokenizer
            
        Raises:
            BadRequest: For invalid text or model
            ModelNotAvailable: If requested model is not available
            NotSupported: If token counting is not supported
            AuthError: For authentication failures
        """
        ...

    async def health(self, *, ctx: Optional[OperationContext] = None) -> Dict[str, Any]:
        """
        Check the health status of the embedding backend.
        
        Args:
            ctx: Operation context for tracing and multi-tenancy
            
        Returns:
            Dictionary with health information including:
            - ok: Boolean overall health status
            - server: Backend server identifier
            - version: Backend version information
            - models: Available models and their status
            
        Raises:
            Unavailable: If the health check fails or backend is unreachable
        """
        ...

# =============================================================================
# Base Instrumented Adapter (validation, metrics, error handling)
# =============================================================================

class BaseEmbeddingAdapter(EmbeddingProtocolV1):
    """
    Base class for implementing Embedding Protocol V1 adapters.
    
    Provides common validation, metrics instrumentation, error handling, and
    SIEM-safe observability. Implementers should override the `_do_*` methods
    to provide backend-specific functionality while getting production-ready
    infrastructure for free.
    
    Example:
        class OpenAIEmbeddingAdapter(BaseEmbeddingAdapter):
            async def _do_embed(self, spec: EmbedSpec, *, ctx: Optional[OperationContext]) -> EmbedResult:
                # OpenAI-specific implementation
                response = await self._client.embeddings.create(
                    input=spec.text,
                    model=spec.model
                )
                return EmbedResult(
                    embedding=EmbeddingVector(
                        vector=response.data[0].embedding,
                        text=spec.text,
                        model=spec.model,
                        dimensions=len(response.data[0].embedding)
                    ),
                    model=spec.model,
                    text=spec.text
                )
    """

    _component = "embedding"

    def __init__(self, *, metrics: Optional[MetricsSink] = None) -> None:
        """
        Initialize the embedding adapter with metrics instrumentation.
        
        Args:
            metrics: Metrics sink for operational monitoring. Uses NoopMetrics if None.
        """
        self._metrics: MetricsSink = metrics or NoopMetrics()

    # --- internal helpers (validation and instrumentation) ---

    @staticmethod
    def _require_non_empty(name: str, value: str) -> None:
        """
        Validate that a string value is non-empty.
        
        Args:
            name: Parameter name for error messages
            value: Value to validate
            
        Raises:
            BadRequest: If value is empty or not a string
        """
        if not isinstance(value, str) or not value.strip():
            raise BadRequest(f"{name} must be a non-empty string")

    @staticmethod
    def _validate_text(text: str, max_length: Optional[int] = None) -> None:
        """
        Validate that text is properly formed and within length limits.
        
        Args:
            text: Text to validate
            max_length: Optional maximum length constraint
            
        Raises:
            BadRequest: If text is empty or exceeds maximum length
        """
        if not text or not isinstance(text, str):
            raise BadRequest("text must be a non-empty string")
        if max_length and len(text) > max_length:
            raise BadRequest(f"text exceeds maximum length of {max_length} characters")

    @staticmethod
    def _tenant_hash(tenant: Optional[str]) -> Optional[str]:
        """
        Create privacy-preserving hash of tenant identifier for metrics.
        
        Args:
            tenant: Raw tenant identifier
            
        Returns:
            Hashed tenant identifier (first 12 chars of SHA256) or None
        """
        if not tenant: 
            return None
        return hashlib.sha256(tenant.encode()).hexdigest()[:12]

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
        
        Args:
            op: Operation name
            t0: Start time from time.monotonic()
            ok: Whether operation succeeded
            code: Status code for metrics
            ctx: Operation context for tenant information
            **extra: Additional metric dimensions
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

    async def capabilities(self) -> EmbeddingCapabilities:
        """Get the capabilities of this embedding adapter."""
        return await self._do_capabilities()

    async def embed(
        self,
        spec: EmbedSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> EmbedResult:
        """
        Generate embedding for a single text with validation and metrics.
        
        See EmbeddingProtocolV1.embed for full documentation.
        """
        self._require_non_empty("text", spec.text)
        self._require_non_empty("model", spec.model)
        
        # Get capabilities to validate against limits
        capabilities = await self._do_capabilities()
        if spec.model not in capabilities.supported_models:
            raise ModelNotAvailable(f"Model '{spec.model}' is not supported")
            
        if capabilities.max_text_length:
            self._validate_text(spec.text, capabilities.max_text_length)

        t0 = time.monotonic()
        try:
            result = await self._do_embed(spec, ctx=ctx)
            self._record("embed", t0, True, ctx=ctx, model=spec.model, text_length=len(spec.text))
            return result
        except EmbeddingAdapterError as e:
            self._record("embed", t0, False, code=type(e).__name__, ctx=ctx, model=spec.model)
            raise

    async def embed_batch(
        self,
        spec: BatchEmbedSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> BatchEmbedResult:
        """
        Generate embeddings for multiple texts with validation and metrics.
        
        See EmbeddingProtocolV1.embed_batch for full documentation.
        """
        self._require_non_empty("model", spec.model)
        if not spec.texts:
            raise BadRequest("texts must not be empty")
            
        # Get capabilities to validate against limits
        capabilities = await self._do_capabilities()
        if spec.model not in capabilities.supported_models:
            raise ModelNotAvailable(f"Model '{spec.model}' is not supported")
            
        if capabilities.max_batch_size and len(spec.texts) > capabilities.max_batch_size:
            raise BadRequest(f"Batch size {len(spec.texts)} exceeds maximum of {capabilities.max_batch_size}")

        # Validate individual texts
        for text in spec.texts:
            self._require_non_empty("text", text)
            if capabilities.max_text_length:
                self._validate_text(text, capabilities.max_text_length)

        t0 = time.monotonic()
        try:
            result = await self._do_embed_batch(spec, ctx=ctx)
            self._record("embed_batch", t0, True, ctx=ctx, model=spec.model, 
                        batch_size=len(spec.texts), successful_embeddings=len(result.embeddings))
            return result
        except EmbeddingAdapterError as e:
            self._record("embed_batch", t0, False, code=type(e).__name__, ctx=ctx, model=spec.model)
            raise

    async def count_tokens(
        self,
        text: str,
        model: str,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> int:
        """Count tokens in text with validation and metrics."""
        self._require_non_empty("text", text)
        self._require_non_empty("model", model)
        
        t0 = time.monotonic()
        try:
            result = await self._do_count_tokens(text, model, ctx=ctx)
            self._record("count_tokens", t0, True, ctx=ctx, model=model, text_length=len(text))
            return result
        except EmbeddingAdapterError as e:
            self._record("count_tokens", t0, False, code=type(e).__name__, ctx=ctx, model=model)
            raise

    async def health(self, *, ctx: Optional[OperationContext] = None) -> Dict[str, Any]:
        """Check health status with metrics instrumentation."""
        t0 = time.monotonic()
        try:
            h = await self._do_health(ctx=ctx)
            self._record("health", t0, True, ctx=ctx)
            return {
                "ok": bool(h.get("ok", True)),
                "server": str(h.get("server", "")),
                "version": str(h.get("version", "")),
                "models": h.get("models", {}),
            }
        except EmbeddingAdapterError as e:
            self._record("health", t0, False, code=type(e).__name__, ctx=ctx)
            raise

    # --- hooks to implement per backend (override these) ---

    async def _do_capabilities(self) -> EmbeddingCapabilities:
        """Implement to return adapter-specific capabilities."""
        raise NotImplementedError

    async def _do_embed(
        self,
        spec: EmbedSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> EmbedResult:
        """Implement single text embedding with validated inputs."""
        raise NotImplementedError

    async def _do_embed_batch(
        self,
        spec: BatchEmbedSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> BatchEmbedResult:
        """Implement batch text embedding with validated inputs."""
        raise NotImplementedError

    async def _do_count_tokens(
        self,
        text: str,
        model: str,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> int:
        """Implement token counting for the specified model."""
        raise NotImplementedError

    async def _do_health(self, *, ctx: Optional[OperationContext] = None) -> Dict[str, Any]:
        """Implement health check for the embedding backend."""
        raise NotImplementedError


__all__ = [
    "EMBEDDING_PROTOCOL_VERSION",
    "EmbeddingVector",
    "EmbeddingResult",
    "EmbeddingBatch",
    "EmbeddingAdapterError",
    "BadRequest",
    "AuthError",
    "ResourceExhausted",
    "TextTooLong",
    "ModelNotAvailable",
    "TransientNetwork",
    "Unavailable",
    "NotSupported",
    "OperationContext",
    "EmbedSpec",
    "BatchEmbedSpec",
    "EmbedResult",
    "BatchEmbedResult",
    "EmbeddingCapabilities",
    "EmbeddingProtocolV1",
    "BaseEmbeddingAdapter",
]
