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
  • Idempotency across multiple calls
"""
import pytest
from corpus_sdk.llm.llm_base import (
    LLMCapabilities,
    OperationContext,
    NotSupported,
)

pytestmark = pytest.mark.asyncio

# Constants for capabilities validation
MAX_REASONABLE_CONTEXT_LENGTH = 10_000_000  # 10M tokens upper bound
MIN_CONTEXT_LENGTH = 1                      # Minimum context length
MIN_SUPPORTED_MODELS = 1                    # Must support at least 1 model


async def test_capabilities_capabilities_shape_and_required_fields(adapter):
    """Quick smoke test of essential capabilities fields."""
    caps = await adapter.capabilities()
    
    assert isinstance(caps, LLMCapabilities), "Should return LLMCapabilities instance"
    assert caps.server and isinstance(caps.server, str), "Server should be non-empty string"
    assert caps.model_family and isinstance(caps.model_family, str), "Model family should be non-empty string"
    assert caps.max_context_length > 0, "Max context length should be positive"
    assert isinstance(caps.supports_streaming, bool), "supports_streaming should be boolean"
    assert isinstance(caps.supported_models, tuple), "supported_models should be tuple"
    assert len(caps.supported_models) >= MIN_SUPPORTED_MODELS, f"Should support at least {MIN_SUPPORTED_MODELS} model"


async def test_capabilities_returns_correct_type(adapter):
    """
    SPECIFICATION.md §8.4 — Capabilities Discovery

    Capabilities MUST return an LLMCapabilities dataclass instance.
    """
    caps = await adapter.capabilities()
    assert isinstance(caps, LLMCapabilities), \
        "capabilities() must return LLMCapabilities instance"


async def test_capabilities_identity_fields(adapter):
    """
    Identity fields (server, version, model_family) MUST be non-empty strings.
    """
    caps = await adapter.capabilities()

    # Server validation
    assert isinstance(caps.server, str), "server must be a string"
    assert len(caps.server.strip()) > 0, "server must be non-empty string"
    assert not caps.server.isspace(), "server must not be only whitespace"

    # Version validation  
    assert isinstance(caps.version, str), "version must be a string"
    assert len(caps.version.strip()) > 0, "version must be non-empty string"
    assert not caps.version.isspace(), "version must not be only whitespace"

    # Model family validation
    assert isinstance(caps.model_family, str), "model_family must be a string"
    assert len(caps.model_family.strip()) > 0, "model_family must be non-empty string"
    assert not caps.model_family.isspace(), "model_family must not be only whitespace"


async def test_capabilities_resource_limits(adapter):
    """
    Resource limits MUST be positive integers within reasonable bounds.
    """
    caps = await adapter.capabilities()

    assert isinstance(caps.max_context_length, int), \
        "max_context_length must be an integer"
    assert caps.max_context_length >= MIN_CONTEXT_LENGTH, \
        f"max_context_length must be at least {MIN_CONTEXT_LENGTH}"
    assert caps.max_context_length <= MAX_REASONABLE_CONTEXT_LENGTH, \
        f"max_context_length should be reasonable (≤{MAX_REASONABLE_CONTEXT_LENGTH:,} tokens)"


async def test_capabilities_feature_flags_are_boolean(adapter):
    """
    All feature flags MUST be boolean values (not truthy/falsy objects).
    """
    caps = await adapter.capabilities()

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
            f"{name} must be a boolean, got {type(value).__name__} ({value})"


async def test_capabilities_supported_models_structure(adapter):
    """
    supported_models MUST be a non-empty tuple of non-empty strings.
    """
    caps = await adapter.capabilities()

    assert isinstance(caps.supported_models, tuple), \
        "supported_models must be a tuple, not list or other sequence"
    assert len(caps.supported_models) >= MIN_SUPPORTED_MODELS, \
        f"Must support at least {MIN_SUPPORTED_MODELS} model"

    for i, model in enumerate(caps.supported_models):
        assert isinstance(model, str), \
            f"Model at index {i} must be a string, got {type(model).__name__}"
        assert len(model.strip()) > 0, \
            f"Model at index {i} must be non-empty string"
        assert not model.isspace(), \
            f"Model at index {i} must not be only whitespace"


async def test_capabilities_consistency_with_count_tokens(adapter):
    """
    If supports_count_tokens is False, count_tokens() SHOULD raise NotSupported.
    If True, count_tokens() MUST work.
    """
    caps = await adapter.capabilities()
    ctx = OperationContext(tenant="test", request_id="test-caps-001")

    if caps.supports_count_tokens:
        # If supported, must work correctly
        count = await adapter.count_tokens("test text for counting", ctx=ctx)
        assert isinstance(count, int), \
            "count_tokens should return integer when supported"
        assert count >= 0, \
            "count_tokens should return non-negative integer"
    else:
        # If not supported, should raise NotSupported (but we don't enforce in tests)
        pass


async def test_capabilities_consistency_with_streaming(adapter):
    """
    If supports_streaming is False, stream() SHOULD raise NotSupported.
    If True, stream() MUST work.
    """
    caps = await adapter.capabilities()
    ctx = OperationContext(tenant="test", request_id="test-caps-002")

    if caps.supports_streaming:
        # If supported, must work correctly
        chunks = []
        async for chunk in adapter.stream(
            messages=[{"role": "user", "content": "test streaming consistency"}],
            model=caps.supported_models[0],
            ctx=ctx,
        ):
            chunks.append(chunk)
            if len(chunks) >= 3:  # Get a few chunks to verify it works
                break
        assert len(chunks) > 0, "stream() should yield chunks when supported"
        # Verify chunk structure
        for chunk in chunks:
            assert hasattr(chunk, 'text'), "Stream chunks should have text attribute"
    else:
        # If not supported, should raise NotSupported (but we don't enforce in tests)
        pass


async def test_capabilities_all_fields_present(adapter):
    """
    Comprehensive check that all expected fields are present and valid.
    """
    caps = await adapter.capabilities()

    # Identity fields (required)
    required_fields = ["server", "version", "model_family"]
    for field in required_fields:
        assert hasattr(caps, field), f"Missing required field: {field}"
        value = getattr(caps, field)
        assert isinstance(value, str) and value.strip(), f"{field} must be non-empty string"

    # Resource limits (required)
    assert hasattr(caps, "max_context_length"), "Missing max_context_length"
    assert isinstance(caps.max_context_length, int) and caps.max_context_length > 0

    # Feature flags (all should be present and boolean)
    feature_flags = [
        "supports_streaming",
        "supports_roles", 
        "supports_json_output",
        "supports_parallel_tool_calls",
        "idempotent_writes",
        "supports_multi_tenant",
        "supports_system_message",
        "supports_deadline",
        "supports_count_tokens",
    ]
    for flag in feature_flags:
        assert hasattr(caps, flag), f"Missing feature flag: {flag}"
        assert isinstance(getattr(caps, flag), bool), \
            f"{flag} must be boolean, got {type(getattr(caps, flag)).__name__}"

    # Model enumeration (required)
    assert hasattr(caps, "supported_models"), "Missing supported_models"
    assert isinstance(caps.supported_models, tuple), "supported_models must be tuple"
    assert len(caps.supported_models) >= 1, "Must support at least one model"


async def test_capabilities_idempotency(adapter):
    """
    Multiple calls to capabilities() SHOULD return consistent results.
    """
    # Call capabilities multiple times
    caps1 = await adapter.capabilities()
    caps2 = await adapter.capabilities()
    caps3 = await adapter.capabilities()

    # Identity fields should be identical
    assert caps1.server == caps2.server == caps3.server, "Server should be consistent"
    assert caps1.version == caps2.version == caps3.version, "Version should be consistent"
    assert caps1.model_family == caps2.model_family == caps3.model_family, "Model family should be consistent"

    # Resource limits should be identical
    assert caps1.max_context_length == caps2.max_context_length == caps3.max_context_length, \
        "Max context length should be consistent"

    # Feature flags should be identical
    feature_flags = [
        "supports_streaming", "supports_roles", "supports_json_output",
        "supports_parallel_tool_calls", "idempotent_writes", "supports_multi_tenant",
        "supports_system_message", "supports_deadline", "supports_count_tokens"
    ]
    for flag in feature_flags:
        val1 = getattr(caps1, flag)
        val2 = getattr(caps2, flag) 
        val3 = getattr(caps3, flag)
        assert val1 == val2 == val3, f"Feature flag {flag} should be consistent: {val1}, {val2}, {val3}"

    # Model list should be identical
    assert caps1.supported_models == caps2.supported_models == caps3.supported_models, \
        "Supported models should be consistent"


async def test_capabilities_reasonable_model_names(adapter):
    """
    Model names should be reasonable identifiers (not garbage).
    """
    caps = await adapter.capabilities()

    for model in caps.supported_models:
        # Basic sanity checks for model names
        assert len(model) <= 100, f"Model name too long: {model}"
        assert not model.startswith(" "), f"Model name should not start with space: '{model}'"
        assert not model.endswith(" "), f"Model name should not end with space: '{model}'"
        
        # Should contain some alphanumeric characters
        assert any(c.isalnum() for c in model), f"Model name should contain alphanumeric chars: {model}"


async def test_capabilities_no_duplicate_models(adapter):
    """
    Supported models list should not contain duplicates.
    """
    caps = await adapter.capabilities()
    
    # Check for duplicates (case-sensitive)
    unique_models = set(caps.supported_models)
    if len(unique_models) != len(caps.supported_models):
        # Find duplicates
        from collections import Counter
        duplicates = [model for model, count in Counter(caps.supported_models).items() if count > 1]
        pytest.fail(f"Supported models contain duplicates: {duplicates}")