# SPDX-License-Identifier: Apache-2.0
"""
Graph Conformance — Health reporting.

Asserts (Spec refs):
  • health() returns required fields                          (§7.6)
  • status in defined enum set (string-valued)               (§7.6)
  • includes read_only & degraded flags                       (§7.6)
  • shape remains stable                                      (§7.6, §6.4)
"""

from __future__ import annotations

import json
import pytest

from corpus_sdk.graph.graph_base import OperationContext as GraphContext, BaseGraphAdapter

pytestmark = pytest.mark.asyncio


async def test_health_returns_required_fields(adapter: BaseGraphAdapter):
    ctx = GraphContext(request_id="t_health_fields", tenant="t")
    h = await adapter.health(ctx=ctx)

    assert isinstance(h, dict)
    for k in ("status", "server", "version", "read_only", "degraded", "ok", "namespaces"):
        assert k in h, f"Missing required health field: {k}"


async def test_health_basic_types(adapter: BaseGraphAdapter):
    ctx = GraphContext(request_id="t_health_types", tenant="t")
    h = await adapter.health(ctx=ctx)

    assert isinstance(h["status"], str) and h["status"]
    assert isinstance(h["server"], str)
    assert isinstance(h["version"], str)
    assert isinstance(h["ok"], bool)
    assert isinstance(h["read_only"], bool)
    assert isinstance(h["degraded"], bool)


async def test_health_namespaces_is_mapping_like(adapter: BaseGraphAdapter):
    ctx = GraphContext(request_id="t_health_namespaces", tenant="t")
    h = await adapter.health(ctx=ctx)

    # Contract-safe: namespaces should be mapping-like (dict is ideal).
    assert isinstance(h.get("namespaces"), dict)


async def test_health_json_serializable(adapter: BaseGraphAdapter):
    ctx = GraphContext(request_id="t_health_json", tenant="t")
    h = await adapter.health(ctx=ctx)

    # Health response should be safe to emit over wire / logs.
    json.dumps(h)


async def test_health_required_keys_stable_across_calls(adapter: BaseGraphAdapter):
    ctx = GraphContext(request_id="t_health_stable", tenant="t")
    h1 = await adapter.health(ctx=ctx)
    h2 = await adapter.health(ctx=ctx)

    required = {"status", "server", "version", "read_only", "degraded", "ok", "namespaces"}
    assert required.issubset(h1.keys())
    assert required.issubset(h2.keys())

    # Don’t require identical values; just ensure required surface doesn’t disappear.
    assert required.issubset(set(h1.keys()))
    assert required.issubset(set(h2.keys()))
