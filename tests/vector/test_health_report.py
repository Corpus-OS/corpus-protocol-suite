# SPDX-License-Identifier: Apache-2.0
"""
Vector Conformance — Health endpoint.

Spec refs:
  • SPECIFICATION.md §9.3 (health)
  • SPECIFICATION.md §6.4 (Observability Interfaces)
"""

import pytest

from corpus_sdk.examples.vector.mock_vector_adapter import MockVectorAdapter

pytestmark = pytest.mark.asyncio


async def test_health_returns_required_fields():
    a = MockVectorAdapter()
    h = await a.health()
    assert isinstance(h, dict)
    assert "ok" in h
    assert "server" in h
    assert "version" in h


async def test_health_includes_namespaces():
    a = MockVectorAdapter()
    h = await a.health()
    assert "namespaces" in h
    assert isinstance(h["namespaces"], dict)


async def test_health_status_ok_bool():
    a = MockVectorAdapter()
    h = await a.health()
    assert isinstance(h["ok"], bool)


async def test_health_shape_consistent_on_error():
    """
    If adapter surfaces health errors, BaseVectorAdapter should normalize to
    Unavailable with consistent envelope shape. Here we just ensure no crash
    and that required keys exist.
    """
    a = MockVectorAdapter()
    h = await a.health()
    assert all(k in h for k in ("ok", "server", "version"))
