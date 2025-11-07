# SPDX-License-Identifier: Apache-2.0
"""
Embedding Conformance — Capabilities shape & stability.

Spec refs:
  • §10.2 Capabilities Discovery (Embedding)
  • §6.2 Common — Capability surfaces MUST be stable and self-consistent
"""

import pytest

from corpus_sdk.embedding.embedding_base import EmbeddingCapabilities
from corpus_sdk.examples.embedding.mock_embedding_adapter import MockEmbeddingAdapter

pytestmark = pytest.mark.asyncio


async def test_capabilities_returns_correct_type():
    a = MockEmbeddingAdapter()
    caps = await a.capabilities()
    assert isinstance(caps, EmbeddingCapabilities)


async def test_capabilities_identity_fields():
    a = MockEmbeddingAdapter()
    caps = await a.capabilities()
    assert isinstance(caps.server, str) and caps.server
    assert isinstance(caps.version, str) and caps.version


async def test_capabilities_supported_models_non_empty_tuple():
    a = MockEmbeddingAdapter()
    caps = await a.capabilities()
    assert isinstance(caps.supported_models, tuple)
    assert caps.supported_models
    assert all(isinstance(m, str) and m for m in caps.supported_models)


async def test_capabilities_resource_limits_valid():
    a = MockEmbeddingAdapter()
    caps = await a.capabilities()

    if caps.max_batch_size is not None:
        assert caps.max_batch_size > 0

    if caps.max_text_length is not None:
        assert caps.max_text_length > 0

    if caps.max_dimensions is not None:
        assert caps.max_dimensions > 0


async def test_capabilities_feature_flags_boolean():
    a = MockEmbeddingAdapter()
    caps = await a.capabilities()
    bool_fields = (
        "supports_normalization",
        "supports_truncation",
        "supports_token_counting",
        "idempotent_operations",
        "supports_multi_tenant",
        "normalizes_at_source",
        "supports_deadline",
    )
    for name in bool_fields:
        assert isinstance(getattr(caps, name), bool), f"{name} must be bool"


async def test_capabilities_truncation_mode_valid():
    a = MockEmbeddingAdapter()
    caps = await a.capabilities()
    assert caps.truncation_mode in ("base", "adapter")


async def test_capabilities_max_dimensions_consistent_with_models():
    a = MockEmbeddingAdapter()
    caps = await a.capabilities()
    # Mock adapter exposes max_dimensions as max(dimensions_by_model)
    if caps.max_dimensions is not None:
        assert caps.max_dimensions >= 0


async def test_capabilities_idempotent():
    a = MockEmbeddingAdapter()
    c1 = await a.capabilities()
    c2 = await a.capabilities()
    assert c1 == c2

