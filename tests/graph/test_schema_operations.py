# SPDX-License-Identifier: Apache-2.0
"""
Graph Conformance — Schema operations.

Asserts (Spec refs):
  • get_schema returns GraphSchema with expected structure     (§7.5)
  • schema is stable and JSON-serializable                     (§7.5)
"""
import json
from typing import Optional, Mapping, Any, List

import pytest

from corpus_sdk.graph.graph_base import (
    OperationContext as GraphContext,
    MetricsSink,
    BaseGraphAdapter,
    GraphSchema,
)

pytestmark = pytest.mark.asyncio


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


async def test_schema_ops_get_schema_returns_graphschema(adapter: BaseGraphAdapter):
    """§7.5: get_schema must return a GraphSchema instance."""
    ctx = GraphContext(request_id="t_schema_dict", tenant="t")
    schema = await adapter.get_schema(ctx=ctx)
    assert isinstance(schema, GraphSchema)
    assert isinstance(schema.nodes, Mapping)
    assert isinstance(schema.edges, Mapping)


async def test_schema_ops_get_schema_structure_valid(adapter: BaseGraphAdapter):
    """§7.5: Schema must include expected structural elements."""
    ctx = GraphContext(request_id="t_schema_keys", tenant="t")
    schema = await adapter.get_schema(ctx=ctx)

    assert isinstance(schema.nodes, Mapping)
    assert isinstance(schema.edges, Mapping)

    # Schema should have some structure
    assert schema.nodes or schema.edges, "Schema should not be empty"


async def test_schema_ops_schema_consistency(adapter: BaseGraphAdapter):
    """§7.5: Schema should be consistent across calls."""
    ctx = GraphContext(request_id="t_schema_consistent", tenant="t")

    schema1 = await adapter.get_schema(ctx=ctx)
    schema2 = await adapter.get_schema(ctx=ctx)

    assert isinstance(schema1, GraphSchema)
    assert isinstance(schema2, GraphSchema)
    assert schema1.nodes.keys() == schema2.nodes.keys()
    assert schema1.edges.keys() == schema2.edges.keys()


async def test_schema_ops_schema_serializable(adapter: BaseGraphAdapter):
    """§7.5: Schema should be JSON-serializable."""
    from dataclasses import asdict

    ctx = GraphContext(request_id="t_schema_serialize", tenant="t")
    schema = await adapter.get_schema(ctx=ctx)

    payload = asdict(schema)
    json_str = json.dumps(payload)
    assert isinstance(json_str, str)

    reconstructed = json.loads(json_str)
    assert isinstance(reconstructed, dict)
    assert "nodes" in reconstructed
    assert "edges" in reconstructed
