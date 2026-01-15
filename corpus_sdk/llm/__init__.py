# corpus_sdk/llm/__init__.py
# SPDX-License-Identifier: Apache-2.0

"""
LLM Protocol V1 - Public API

This module provides the public interface for the LLM Protocol.
All public types and handlers are re-exported here for clean imports.
"""

from corpus_sdk.llm.llm_base import (
    # Protocol version
    LLM_PROTOCOL_VERSION,
    LLM_PROTOCOL_ID,
    
    # Error types
    LLMAdapterError,
    BadRequest,
    AuthError,
    ResourceExhausted,
    TransientNetwork,
    Unavailable,
    NotSupported,
    ModelOverloaded,
    DeadlineExceeded,
    
    # Context and metrics
    OperationContext,
    MetricsSink,
    NoopMetrics,
    
    # Result models
    TokenUsage,
    ToolCallFunction,
    ToolCall,
    LLMCompletion,
    LLMChunk,
    
    # Policy interfaces
    DeadlinePolicy,
    CircuitBreaker,
    Cache,
    TTLAwareCache,
    RateLimiter,
    
    # Policy implementations
    NoopDeadline,
    SimpleDeadline,
    NoopBreaker,
    SimpleCircuitBreaker,
    NoopCache,
    InMemoryTTLCache,
    NoopLimiter,
    TokenBucketLimiter,
    
    # Capabilities
    LLMCapabilities,
    
    # Protocol interface
    LLMProtocolV1,
    BaseLLMAdapter,
    
    # Wire handler
    WireLLMHandler,
    
    # Wire helpers (advanced users only)
    _ctx_from_wire,
    _error_to_wire,
    _success_to_wire,
    _chunk_to_wire,
)

__all__ = [
    # Protocol version
    "LLM_PROTOCOL_VERSION",
    "LLM_PROTOCOL_ID",
    
    # Error types
    "LLMAdapterError",
    "BadRequest",
    "AuthError",
    "ResourceExhausted",
    "TransientNetwork",
    "Unavailable",
    "NotSupported",
    "ModelOverloaded",
    "DeadlineExceeded",
    
    # Context and metrics
    "OperationContext",
    "MetricsSink",
    "NoopMetrics",
    
    # Result models
    "TokenUsage",
    "ToolCallFunction",
    "ToolCall",
    "LLMCompletion",
    "LLMChunk",
    
    # Policy interfaces
    "DeadlinePolicy",
    "CircuitBreaker",
    "Cache",
    "TTLAwareCache",
    "RateLimiter",
    
    # Policy implementations
    "NoopDeadline",
    "SimpleDeadline",
    "NoopBreaker",
    "SimpleCircuitBreaker",
    "NoopCache",
    "InMemoryTTLCache",
    "NoopLimiter",
    "TokenBucketLimiter",
    
    # Capabilities
    "LLMCapabilities",
    
    # Protocol interface
    "LLMProtocolV1",
    "BaseLLMAdapter",
    
    # Wire handler
    "WireLLMHandler",
    
    # Wire helpers
    "_ctx_from_wire",
    "_error_to_wire",
    "_success_to_wire",
    "_chunk_to_wire",
]

__version__ = LLM_PROTOCOL_VERSION
