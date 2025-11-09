# SPDX-License-Identifier: Apache-2.0
"""
Vector Conformance — Capabilities shape & identity.

Spec refs:
  • SPECIFICATION.md §9.2 (Capabilities)
  • SPECIFICATION.md §6.2 (Capability Discovery)
"""

import pytest

from corpus_sdk.examples.vector.mock_vector_adapter import MockVectorAdapter
from adapter_sdk.vector_base import VectorCapabilities, VECTOR_PROTOCOL_ID

pytestmark = pytest.mark.asyncio


async def test_capabilities_returns_correct_type():
    a = MockVectorAdapter()
    caps = await a.capabilities()
    assert isinstance(caps, VectorCapabilities)


async def test_capabilities_identity_fields():
    a = MockVectorAdapter()
    caps = await a.capabilities()
    assert isinstance(caps.server, str) and caps.server
    assert isinstance(caps.version, str) and caps.version
    assert caps.protocol == VECTOR_PROTOCOL_ID


async def test_capabilities_supported_metrics():
    a = MockVectorAdapter()
    caps = await a.capabilities()
    assert isinstance(caps.supported_metrics, tuple)
    assert caps.supported_metrics  # non-empty
    for m in caps.supported_metrics:
        assert m in ("cosine", "euclidean", "dotproduct")


async def test_capabilities_resource_limits_positive():
    a = MockVectorAdapter()
    caps = await a.capabilities()
    if caps.max_dimensions is not None:
        assert caps.max_dimensions > 0
    if caps.max_top_k is not None:
        assert caps.max_top_k > 0
    if caps.max_batch_size is not None:
        assert caps.max_batch_size > 0


async def test_capabilities_feature_flags_boolean():
    a = MockVectorAdapter()
    caps = await a.capabilities()

    bool_fields = [
        "supports_namespaces",
        "supports_metadata_filtering",
        "supports_batch_operations",
        "supports_index_management",
        "idempotent_writes",
        "supports_multi_tenant",
        "supports_deadline",
    ]
    for name in bool_fields:
        assert isinstance(getattr(caps, name), bool)


async def test_capabilities_idempotent_calls():
    a = MockVectorAdapter()
    c1 = await a.capabilities()
    c2 = await a.capabilities()
    assert c1 == c2


async def test_capabilities_all_required_fields_present():
    a = MockVectorAdapter()
    caps = await a.capabilities()

    assert caps.server
    assert caps.version
    assert caps.protocol == VECTOR_PROTOCOL_ID
    assert isinstance(caps.supported_metrics, tuple)
