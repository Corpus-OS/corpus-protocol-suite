# corpus_sdk/vector/__init__.py
# SPDX-License-Identifier: Apache-2.0

"""
Vector Protocol V1 - Public API

This module provides the public interface for the Vector Protocol.
All public types and handlers are re-exported here for clean imports.
"""

from corpus_sdk.vector.vector_base import (
    # Protocol version
    VECTOR_PROTOCOL_VERSION,
    VECTOR_PROTOCOL_ID,
    
    # Core types
    VectorID,
    Vector,
    VectorMatch,
    QueryResult,
    
    # Document storage
    Document,
    DocStore,
    InMemoryDocStore,
    RedisDocStore,
    
    # Error types
    VectorAdapterError,
    BadRequest,
    AuthError,
    ResourceExhausted,
    DimensionMismatch,
    IndexNotReady,
    TransientNetwork,
    Unavailable,
    NotSupported,
    DeadlineExceeded,
    
    # Context and metrics
    OperationContext,
    MetricsSink,
    NoopMetrics,
    
    # Specifications
    QuerySpec,
    BatchQuerySpec,
    UpsertSpec,
    DeleteSpec,
    NamespaceSpec,
    
    # Results
    UpsertResult,
    DeleteResult,
    NamespaceResult,
    
    # Capabilities
    VectorCapabilities,
    
    # Protocol interface
    VectorProtocolV1,
    VectorAdapterConfig,
    BaseVectorAdapter,
    
    # Policy interfaces
    DeadlinePolicy,
    CircuitBreaker,
    Cache,
    RateLimiter,
    
    # Policy implementations
    NoopDeadline,
    SimpleDeadline,
    NoopBreaker,
    SimpleCircuitBreaker,
    NoopCache,
    InMemoryTTLCache,
    NoopLimiter,
    SimpleTokenBucketLimiter,
    
    # Wire handler
    WireVectorHandler,
    
    # Wire helpers
    _ctx_from_wire,
    _error_to_wire,
    _success_to_wire,
)

__all__ = [
    "VECTOR_PROTOCOL_VERSION",
    "VECTOR_PROTOCOL_ID",
    "VectorID",
    "Vector",
    "VectorMatch",
    "QueryResult",
    "Document",
    "DocStore",
    "InMemoryDocStore",
    "RedisDocStore",
    "VectorAdapterError",
    "BadRequest",
    "AuthError",
    "ResourceExhausted",
    "DimensionMismatch",
    "IndexNotReady",
    "TransientNetwork",
    "Unavailable",
    "NotSupported",
    "DeadlineExceeded",
    "OperationContext",
    "QuerySpec",
    "BatchQuerySpec",
    "UpsertSpec",
    "DeleteSpec",
    "NamespaceSpec",
    "UpsertResult",
    "DeleteResult",
    "NamespaceResult",
    "VectorCapabilities",
    "VectorProtocolV1",
    "VectorAdapterConfig",
    "BaseVectorAdapter",
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
    "SimpleTokenBucketLimiter",
    "MetricsSink",
    "NoopMetrics",
    "WireVectorHandler",
    "_ctx_from_wire",
    "_error_to_wire",
    "_success_to_wire",
]

__version__ = VECTOR_PROTOCOL_VERSION
