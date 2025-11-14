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
from corpus_sdk.llm.llm_base import (
    LLMCapabilities,
    OperationContext,
    NotSupported,
)
from examples.common.ctx import make_ctx

pytestmark = pytest.mark.asyncio


async def test_capabilities_capabilities_shape_and_required_fields(adapter):
    """Quick smoke test of essential capabilities fields."""
    caps = await adapter.capabilities()
    assert isinstance(caps, LLMCapabilities)
    assert caps.server
    assert caps.model_family
    assert caps.max_context_length > 0
    assert isinstance(caps.supports_streaming, bool)
    assert isinstance(caps.supported_models, tuple)
    assert len(caps.supported_models) >= 1


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

    assert isinstance(caps.server, str) and len(caps.server) > 0, \
        "server must be a non-empty string"
    assert isinstance(caps.version, str) and len(caps.version) > 0, \
        "version must be a non-empty string"
    assert isinstance(caps.model_family, str) and len(caps.model_family) > 0, \
        "model_family must be a non-empty string"


async def test_capabilities_resource_limits(adapter):
    """
    Resource limits MUST be positive integers within reasonable bounds.
    """
    caps = await adapter.capabilities()

    assert isinstance(caps.max_context_length, int), \
        "max_context_length must be an integer"
    assert caps.max_context_length > 0, \
        "max_context_length must be positive"
    assert caps.max_context_length <= 10_000_000, \
        "max_context_length should be reasonable (≤10M tokens)"


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
            f"{name} must be a boolean, got {type(value).__name__}"


async def test_capabilities_supported_models_structure(adapter):
    """
    supported_models MUST be a non-empty tuple of non-empty strings.
    """
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


async def test_capabilities_consistency_with_count_tokens(adapter):
    """
    If supports_count_tokens is False, count_tokens() SHOULD raise NotSupported.
    If True, count_tokens() MUST work.
    """
    caps = await adapter.capabilities()
    ctx = make_ctx(OperationContext, tenant="test", request_id="test-caps-001")

    if caps.supports_count_tokens:
        count = await adapter.count_tokens("test text", ctx=ctx)
        assert isinstance(count, int) and count >= 0, \
            "count_tokens should return non-negative integer"
    else:
        # If not supported, real adapters SHOULD raise NotSupported.
        # We don't enforce that here; conformance harness may test separately.
        pass


async def test_capabilities_consistency_with_streaming(adapter):
    """
    If supports_streaming is False, stream() SHOULD raise NotSupported.
    If True, stream() MUST work.
    """
    caps = await adapter.capabilities()
    ctx = make_ctx(OperationContext, tenant="test", request_id="test-caps-002")

    if caps.supports_streaming:
        chunks = []
        async for chunk in adapter.stream(
            messages=[{"role": "user", "content": "test"}],
            model=caps.supported_models[0],
            ctx=ctx,
        ):
            chunks.append(chunk)
            if len(chunks) >= 3:
                break
        assert len(chunks) > 0, "stream() should yield at least one chunk"
    else:
        # Real adapters SHOULD raise NotSupported in this branch.
        pass


async def test_capabilities_all_fields_present(adapter):
    """
    Comprehensive check that all expected fields are present and valid.
    """
    caps = await adapter.capabilities()

    # Identity
    assert hasattr(caps, "server") and caps.server
    assert hasattr(caps, "version") and caps.version
    assert hasattr(caps, "model_family") and caps.model_family

    # Limits
    assert hasattr(caps, "max_context_length")
    assert caps.max_context_length > 0

    # Feature flags
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
            f"{flag} must be boolean"

    # Models
    assert hasattr(caps, "supported_models")
    assert isinstance(caps.supported_models, tuple)
    assert len(caps.supported_models) > 0


async def test_capabilities_idempotency(adapter):
    """
    Multiple calls to capabilities() SHOULD return consistent results.
    """
    caps1 = await adapter.capabilities()
    caps2 = await adapter.capabilities()

    assert caps1.server == caps2.server
    assert caps1.version == caps2.version
    assert caps1.model_family == caps2.model_family
    assert caps1.max_context_length == caps2.max_context_length
    assert caps1.supports_streaming == caps2.supports_streaming
    assert caps1.supported_models == caps2.supported_models