# SPDX-License-Identifier: Apache-2.0
"""
Graph Conformance — Schema operations.

Asserts (Spec refs):
  • get_schema returns dict with expected keys                (§7.5)
  • standalone mode caches schema (cache_hits counter)       (§5.3, §13.1)
"""
import pytest
from typing import Optional, Mapping, Any, List

from corpus_sdk.examples.graph.mock_graph_adapter import MockGraphAdapter
from corpus_sdk.graph.graph_base import OperationContext as GraphContext, MetricsSink
from corpus_sdk.examples.common.ctx import make_ctx

pytestmark = pytest.mark.asyncio


class CaptureMetrics(MetricsSink):
    def __init__(self) -> None:
        self.observations: List[dict] = []
        self.counters: List[dict] = []
    def observe(self, *, component: str, op: str, ms: float, ok: bool, code: str = "OK", extra: Optional[Mapping[str, Any]] = None) -> None:
        self.observations.append({"component": component, "op": op, "ok": ok, "code": code, "extra": dict(extra or {})})
    def counter(self, *, component: str, name: str, value: int = 1, extra: Optional[Mapping[str, Any]] = None) -> None:
        self.counters.append({"component": component, "name": name, "value": value, "extra": dict(extra or {})})


async def test_get_schema_returns_dict():
    a = MockGraphAdapter()
    ctx = make_ctx(GraphContext, request_id="t_schema_dict", tenant="t")
    s = await a.get_schema(ctx=ctx)
    assert isinstance(s, dict)


async def test_get_schema_structure_valid():
    a = MockGraphAdapter()
    ctx = make_ctx(GraphContext, request_id="t_schema_keys", tenant="t")
    s = await a.get_schema(ctx=ctx)
    assert "nodes" in s and "edges" in s


async def test_schema_cached_in_standalone_mode():
    metrics = CaptureMetrics()
    a = MockGraphAdapter(mode="standalone", metrics=metrics)
    ctx = make_ctx(GraphContext, request_id="t_schema_cache", tenant="t-cache")
    await a.get_schema(ctx=ctx)  # warm
    await a.get_schema(ctx=ctx)  # should hit cache
    assert any(c["name"] == "cache_hits" for c in metrics.counters)
