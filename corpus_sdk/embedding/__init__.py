# corpus_sdk/embedding/__init__.py
# SPDX-License-Identifier: Apache-2.0

"""
Embedding Protocol V1 - Public API

This module provides the public interface for the Embedding Protocol.
All public types and handlers are re-exported here for clean imports.
"""

from corpus_sdk.embedding.embedding_base import (
    # Protocol version
    EMBEDDING_PROTOCOL_VERSION,
    EMBEDDING_PROTOCOL_ID,
    
    # Core types
    EmbeddingVector,
    EmbeddingResult,
    EmbeddingBatch,
    EmbedChunk,
    EmbeddingStats,
    
    # Error types
    EmbeddingAdapterError,
    BadRequest,
    AuthError,
    ResourceExhausted,
    TextTooLong,
    ModelNotAvailable,
    TransientNetwork,
    Unavailable,
    NotSupported,
    DeadlineExceeded,
    
    # Context and metrics
    OperationContext,
    MetricsSink,
    NoopMetrics,
    
    # Specifications
    EmbedSpec,
    BatchEmbedSpec,
    
    # Results
    EmbedResult,
    BatchEmbedResult,
    
    # Capabilities
    EmbeddingCapabilities,
    
    # Protocol interface
    EmbeddingProtocolV1,
    BaseEmbeddingAdapter,
    
    # Policy interfaces
    DeadlinePolicy,
    TruncationPolicy,
    NormalizationPolicy,
    CircuitBreaker,
    Cache,
    RateLimiter,
    
    # Policy implementations
    NoopDeadline,
    EnforcingDeadline,
    SimpleCharTruncation,
    L2Normalization,
    NoopBreaker,
    SimpleCircuitBreaker,
    NoopCache,
    InMemoryTTLCache,
    NoopLimiter,
    TokenBucketLimiter,
    
    # Wire handler
    WireEmbeddingHandler,
    
    # Wire helpers
    _ctx_from_wire,
    _error_to_wire,
    _success_to_wire,
    _stream_chunk_to_wire,
)

__all__ = [
    "EMBEDDING_PROTOCOL_VERSION",
    "EMBEDDING_PROTOCOL_ID",
    "EmbeddingVector",
    "EmbeddingResult",
    "EmbeddingBatch",
    "EmbedChunk",
    "EmbeddingStats",
    "EmbeddingAdapterError",
    "BadRequest",
    "AuthError",
    "ResourceExhausted",
    "TextTooLong",
    "ModelNotAvailable",
    "TransientNetwork",
    "Unavailable",
    "NotSupported",
    "DeadlineExceeded",
    "OperationContext",
    "EmbedSpec",
    "BatchEmbedSpec",
    "EmbedResult",
    "BatchEmbedResult",
    "EmbeddingCapabilities",
    "EmbeddingProtocolV1",
    "BaseEmbeddingAdapter",
    "DeadlinePolicy",
    "TruncationPolicy",
    "NormalizationPolicy",
    "CircuitBreaker",
    "Cache",
    "RateLimiter",
    "NoopDeadline",
    "EnforcingDeadline",
    "SimpleCharTruncation",
    "L2Normalization",
    "NoopBreaker",
    "SimpleCircuitBreaker",
    "NoopCache",
    "InMemoryTTLCache",
    "NoopLimiter",
    "TokenBucketLimiter",
    "MetricsSink",
    "NoopMetrics",
    "WireEmbeddingHandler",
    "_ctx_from_wire",
    "_error_to_wire",
    "_success_to_wire",
    "_stream_chunk_to_wire",
]

__version__ = EMBEDDING_PROTOCOL_VERSION
