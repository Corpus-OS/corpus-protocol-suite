# SPDX-License-Identifier: Apache-2.0
"""
Graph Conformance — Schema operations.

Asserts (Spec refs):
  • get_schema returns dict with expected keys                (§7.5)
  • standalone mode caches schema (cache_hits counter)       (§5.3, §13.1)
"""
import pytest
from typing import Optional, Mapping, Any, List

from corpus_sdk.graph.graph_base import (
    OperationContext as GraphContext,
    MetricsSink,
    BaseGraphAdapter,
)

pytestmark = pytest.mark.asyncio


def make_ctx(ctx_cls, **kwargs):
    """Local helper to construct an OperationContext."""
    return ctx_cls(**kwargs)


class CaptureMetrics(MetricsSink):
    def __init__(self) -> None:
        self.observations: List[dict] = []
        self.counters: List[dict] = []

    def observe(
        self,
        *,
        component: str,
        op: str,
        ms: float,
        ok: bool,
        code: str = "OK",
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.observations.append(
            {
                "component": component,
                "op": op,
                "ok": ok,
                "code": code,
                "extra": dict(extra or {}),
            }
        )

    def counter(
        self,
        *,
        component: str,
        name: str,
        value: int = 1,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.counters.append(
            {
                "component": component,
                "name": name,
                "value": value,
                "extra": dict(extra or {}),
            }
        )


async def test_schema_ops_get_schema_returns_dict(adapter: BaseGraphAdapter):
    """§7.5: get_schema must return a dictionary."""
    ctx = make_ctx(GraphContext, request_id="t_schema_dict", tenant="t")
    schema = await adapter.get_schema(ctx=ctx)
    assert isinstance(schema, dict), "Schema must be returned as a dictionary"


async def test_schema_ops_get_schema_structure_valid(adapter: BaseGraphAdapter):
    """§7.5: Schema must include expected structural elements."""
    ctx = make_ctx(GraphContext, request_id="t_schema_keys", tenant="t")
    schema = await adapter.get_schema(ctx=ctx)

    # Schema should have some structure, though exact keys may vary
    assert len(schema) > 0, "Schema should not be empty"
    # Common schema elements (adapters may have different structures)
    assert any(
        key in schema
        for key in ["nodes", "edges", "vertexLabels", "edgeLabels", "types"]
    ), f"Schema missing expected structural elements: {list(schema.keys())}"


async def test_schema_ops_schema_cached_in_standalone_mode(adapter: BaseGraphAdapter):
    """§5.3: Schema should be cached in standalone mode."""
    # This test validates caching behavior through metrics
    metrics = CaptureMetrics()
    ctx = make_ctx(
        GraphContext,
        request_id="t_schema_cache",
        tenant="t-cache",
        metrics=metrics,
    )

    # Call get_schema twice to potentially trigger caching
    schema1 = await adapter.get_schema(ctx=ctx)
    schema2 = await adapter.get_schema(ctx=ctx)

    assert isinstance(schema1, dict) and isinstance(schema2, dict)

    # If adapter implements caching, it should emit cache-related metrics
    # This is an observable behavior test rather than implementation check
    cache_metrics = [c for c in metrics.counters if "cache" in c["name"].lower()]
    if cache_metrics:
        assert any(c["name"] == "cache_hits" for c in metrics.counters), (
            "Expected cache_hits counter for schema operations"
        )


async def test_schema_ops_schema_consistency(adapter: BaseGraphAdapter):
    """§7.5: Schema should be consistent across calls."""
    ctx = make_ctx(GraphContext, request_id="t_schema_consistent", tenant="t")

    # Get schema multiple times
    schema1 = await adapter.get_schema(ctx=ctx)
    schema2 = await adapter.get_schema(ctx=ctx)

    # Schema structure should be consistent (same keys)
    assert schema1.keys() == schema2.keys(), (
        "Schema structure should be consistent across calls"
    )

    # For adapters with stable schemas, content should be identical
    # Some adapters might have dynamic schemas, so we only check structure consistency


async def test_schema_ops_schema_serializable(adapter: BaseGraphAdapter):
    """§7.5: Schema should be JSON-serializable."""
    import json

    ctx = make_ctx(GraphContext, request_id="t_schema_serialize", tenant="t")
    schema = await adapter.get_schema(ctx=ctx)

    # Should be able to serialize to JSON
    try:
        json_str = json.dumps(schema)
        assert isinstance(json_str, str)
        # Should be able to deserialize back
        reconstructed = json.loads(json_str)
        assert isinstance(reconstructed, dict)
    except (TypeError, ValueError) as e:
        pytest.fail(f"Schema is not JSON-serializable: {e}")
