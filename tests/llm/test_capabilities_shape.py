# SPDX-License-Identifier: Apache-2.0
"""
LLM Conformance — Capabilities shape and validation.
Covers:
  • Returns LLMCapabilities instance
  • All identity fields (server, version, model_family) are non-empty
  • Resource limits are positive and reasonable
  • All feature flags are boolean
  • Model enumeration is valid (non-empty tuple of strings)
  • Logical consistency between capabilities and adapter behavior
"""
import pytest
from corpus_sdk.examples.llm.mock_llm_adapter import MockLLMAdapter
from corpus_sdk.llm.llm_base import LLMCapabilities, OperationContext
from corpus_sdk.examples.common.ctx import make_ctx
from corpus_sdk.examples.common.errors import NotSupported

pytestmark = pytest.mark.asyncio


async def test_capabilities_shape_and_required_fields():
    """Quick smoke test of essential capabilities fields."""
    adapter = MockLLMAdapter(failure_rate=0.0)
    caps = await adapter.capabilities()
    assert isinstance(caps, LLMCapabilities)
    assert caps.server == "mock"
    assert caps.model_family == "mock"
    assert caps.max_context_length > 0
    assert caps.supports_streaming is True
    assert isinstance(caps.supported_models, tuple)
    assert len(caps.supported_models) >= 1

async def test_capabilities_returns_correct_type():
    """
    SPECIFICATION.md §8.4 — Capabilities Discovery
    
    Capabilities MUST return an LLMCapabilities dataclass instance.
    """
    adapter = MockLLMAdapter(failure_rate=0.0)
    caps = await adapter.capabilities()
    assert isinstance(caps, LLMCapabilities), \
        "capabilities() must return LLMCapabilities instance"


async def test_capabilities_identity_fields():
    """
    Identity fields (server, version, model_family) MUST be non-empty strings.
    """
    adapter = MockLLMAdapter(failure_rate=0.0)
    caps = await adapter.capabilities()
    
    assert isinstance(caps.server, str) and len(caps.server) > 0, \
        "server must be a non-empty string"
    assert isinstance(caps.version, str) and len(caps.version) > 0, \
        "version must be a non-empty string"
    assert isinstance(caps.model_family, str) and len(caps.model_family) > 0, \
        "model_family must be a non-empty string"


async def test_capabilities_resource_limits():
    """
    Resource limits MUST be positive integers within reasonable bounds.
    """
    adapter = MockLLMAdapter(failure_rate=0.0)
    caps = await adapter.capabilities()
    
    assert isinstance(caps.max_context_length, int), \
        "max_context_length must be an integer"
    assert caps.max_context_length > 0, \
        "max_context_length must be positive"
    assert caps.max_context_length <= 10_000_000, \
        "max_context_length should be reasonable (≤10M tokens)"


async def test_capabilities_feature_flags_are_boolean():
    """
    All feature flags MUST be boolean values (not truthy/falsy objects).
    """
    adapter = MockLLMAdapter(failure_rate=0.0)
    caps = await adapter.capabilities()
    
    # Collect all feature flags
    flags = {
        "supports_streaming": caps.supports_streaming,
        "supports_roles": caps.supports_roles,
        "supports_json_output": caps.supports_json_output,
        "supports_parallel_tool_calls": caps.supports_parallel_tool_calls,
        "idempotent_writes": caps.idempotent_writes,
        "supports_multi_tenant": caps.supports_multi_tenant,
        "supports_system_message": caps.supports_system_message,
        "supports_deadline": caps.supports_deadline,
        "supports_count_tokens": caps.supports_count_tokens,
    }
    
    for name, value in flags.items():
        assert isinstance(value, bool), \
            f"{name} must be a boolean, got {type(value).__name__}"


async def test_capabilities_supported_models_structure():
    """
    supported_models MUST be a non-empty tuple of non-empty strings.
    """
    adapter = MockLLMAdapter(failure_rate=0.0)
    caps = await adapter.capabilities()
    
    assert isinstance(caps.supported_models, tuple), \
        "supported_models must be a tuple"
    assert len(caps.supported_models) >= 1, \
        "Must support at least one model"
    
    for i, model in enumerate(caps.supported_models):
        assert isinstance(model, str), \
            f"Model at index {i} must be a string, got {type(model).__name__}"
        assert len(model) > 0, \
            f"Model at index {i} must be non-empty"


async def test_capabilities_consistency_with_count_tokens():
    """
    If supports_count_tokens is False, count_tokens() SHOULD raise NotSupported.
    If True, count_tokens() MUST work.
    """
    adapter = MockLLMAdapter(failure_rate=0.0)
    caps = await adapter.capabilities()
    ctx = make_ctx(OperationContext, tenant="test", request_id="test-caps-001")
    
    if caps.supports_count_tokens:
        # If supported, count_tokens should work
        count = await adapter.count_tokens("test text", ctx=ctx)
        assert isinstance(count, int) and count >= 0, \
            "count_tokens should return non-negative integer"
    else:
        # If not supported, should raise NotSupported
        # (Note: MockLLMAdapter always supports it, but this documents the contract)
        pass  # Skip for mock, but real adapters should be tested


async def test_capabilities_consistency_with_streaming():
    """
    If supports_streaming is False, stream() SHOULD raise NotSupported.
    If True, stream() MUST work.
    """
    adapter = MockLLMAdapter(failure_rate=0.0)
    caps = await adapter.capabilities()
    ctx = make_ctx(OperationContext, tenant="test", request_id="test-caps-002")
    
    if caps.supports_streaming:
        # If supported, streaming should work
        chunks = []
        async for chunk in adapter.stream(
            messages=[{"role": "user", "content": "test"}],
            ctx=ctx
        ):
            chunks.append(chunk)
            if len(chunks) >= 3:  # Don't consume entire stream
                break
        assert len(chunks) > 0, "Stream should yield chunks"
    else:
        # If not supported, should raise NotSupported
        # (Note: MockLLMAdapter always supports it)
        pass  # Skip for mock


async def test_capabilities_all_fields_present():
    """
    Comprehensive check that all expected fields are present and valid.
    This is a catch-all test for completeness.
    """
    adapter = MockLLMAdapter(failure_rate=0.0)
    caps = await adapter.capabilities()
    
    # Identity (3 fields)
    assert hasattr(caps, 'server') and caps.server
    assert hasattr(caps, 'version') and caps.version
    assert hasattr(caps, 'model_family') and caps.model_family
    
    # Limits (1 field)
    assert hasattr(caps, 'max_context_length') and caps.max_context_length > 0
    
    # Features (9 boolean fields)
    feature_flags = [
        'supports_streaming',
        'supports_roles',
        'supports_json_output',
        'supports_parallel_tool_calls',
        'idempotent_writes',
        'supports_multi_tenant',
        'supports_system_message',
        'supports_deadline',
        'supports_count_tokens',
    ]
    
    for flag in feature_flags:
        assert hasattr(caps, flag), f"Missing feature flag: {flag}"
        assert isinstance(getattr(caps, flag), bool), \
            f"{flag} must be boolean"
    
    # Models (1 field)
    assert hasattr(caps, 'supported_models')
    assert isinstance(caps.supported_models, tuple)
    assert len(caps.supported_models) > 0


async def test_capabilities_idempotency():
    """
    Multiple calls to capabilities() SHOULD return consistent results.
    """
    adapter = MockLLMAdapter(failure_rate=0.0)
    
    caps1 = await adapter.capabilities()
    caps2 = await adapter.capabilities()
    
    # Should return same structure
    assert caps1.server == caps2.server
    assert caps1.version == caps2.version
    assert caps1.model_family == caps2.model_family
    assert caps1.max_context_length == caps2.max_context_length
    assert caps1.supports_streaming == caps2.supports_streaming
    assert caps1.supported_models == caps2.supported_models
