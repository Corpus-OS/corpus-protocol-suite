# SPDX-License-Identifier: Apache-2.0
"""
Vector Conformance — Capabilities shape & identity.

Spec refs:
  • SPECIFICATION.md §9.2 (Capabilities)
  • SPECIFICATION.md §6.2 (Capability Discovery)
"""

import pytest
from corpus_sdk.vector.vector_base import VectorCapabilities, VECTOR_PROTOCOL_ID

pytestmark = pytest.mark.asyncio


async def test_capabilities_capabilities_returns_correct_type(adapter):
    """Verify capabilities() returns VectorCapabilities instance."""
    caps = await adapter.capabilities()
    assert isinstance(caps, VectorCapabilities)


async def test_capabilities_identity_fields(adapter):
    """Verify identity fields are non-empty strings and protocol is correct."""
    caps = await adapter.capabilities()
    assert isinstance(caps.server, str) and caps.server
    assert isinstance(caps.version, str) and caps.version
    assert caps.protocol == VECTOR_PROTOCOL_ID


async def test_capabilities_supported_metrics(adapter):
    """Verify supported_metrics is non-empty tuple of valid metric names."""
    caps = await adapter.capabilities()
    assert isinstance(caps.supported_metrics, tuple)
    assert caps.supported_metrics  # non-empty
    valid_metrics = {"cosine", "euclidean", "dotproduct"}
    for metric in caps.supported_metrics:
        assert metric in valid_metrics, f"Invalid metric: {metric}"


async def test_capabilities_resource_limits_positive(adapter):
    """Verify resource limits are positive integers when defined."""
    caps = await adapter.capabilities()
    if caps.max_dimensions is not None:
        assert caps.max_dimensions > 0
    if caps.max_top_k is not None:
        assert caps.max_top_k > 0
    if caps.max_batch_size is not None:
        assert caps.max_batch_size > 0


async def test_capabilities_feature_flags_boolean(adapter):
    """Verify all feature flags are boolean values."""
    caps = await adapter.capabilities()

    bool_fields = [
        "supports_namespaces",
        "supports_metadata_filtering",
        "supports_batch_operations",
        "supports_index_management",
        "idempotent_writes",
        "supports_multi_tenant",
        "supports_deadline",
    ]
    for field_name in bool_fields:
        value = getattr(caps, field_name)
        assert isinstance(value, bool), f"{field_name} must be boolean, got {type(value).__name__}"


async def test_capabilities_idempotent_calls(adapter):
    """Verify multiple capabilities() calls return consistent results."""
    caps1 = await adapter.capabilities()
    caps2 = await adapter.capabilities()
    assert caps1 == caps2


async def test_capabilities_all_required_fields_present(adapter):
    """Verify all required capability fields are present and valid."""
    caps = await adapter.capabilities()

    assert caps.server and isinstance(caps.server, str)
    assert caps.version and isinstance(caps.version, str)
    assert caps.protocol == VECTOR_PROTOCOL_ID
    assert isinstance(caps.supported_metrics, tuple) and caps.supported_metrics