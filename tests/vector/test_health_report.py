# SPDX-License-Identifier: Apache-2.0
"""
Vector Conformance — Health endpoint.

Spec refs:
  • SPECIFICATION.md §9.3 (health)
  • SPECIFICATION.md §6.4 (Observability Interfaces)
"""

import pytest

pytestmark = pytest.mark.asyncio


async def test_health_health_returns_required_fields(adapter):
    """Verify health endpoint returns all required fields."""
    health = await adapter.health()

    assert isinstance(health, dict)
    assert "ok" in health
    assert "server" in health
    assert "version" in health


async def test_health_health_includes_namespaces(adapter):
    """Verify health response includes namespace information."""
    health = await adapter.health()

    assert "namespaces" in health
    assert isinstance(health["namespaces"], dict)


async def test_health_status_ok_bool(adapter):
    """Verify health status is a boolean value."""
    health = await adapter.health()

    assert isinstance(health["ok"], bool)


async def test_health_shape_consistent_on_error(adapter):
    """
    If adapter surfaces health errors, BaseVectorAdapter should normalize to
    Unavailable with consistent envelope shape. Here we just ensure no crash
    and that required keys exist.
    """
    health = await adapter.health()

    required_fields = {"ok", "server", "version"}
    assert all(field in health for field in required_fields)

    assert isinstance(health["ok"], bool)
    assert isinstance(health["server"], str)
    assert isinstance(health["version"], str)


async def test_health_identity_fields_stable(adapter):
    """Verify health identity fields remain stable across calls."""
    health1 = await adapter.health()
    health2 = await adapter.health()

    assert health1["server"] == health2["server"]
    assert health1["version"] == health2["version"]


async def test_health_identity_fields_nonempty_strings(adapter):
    """Verify health identity fields are non-empty strings."""
    health = await adapter.health()
    assert isinstance(health.get("server"), str) and health["server"]
    assert isinstance(health.get("version"), str) and health["version"]
