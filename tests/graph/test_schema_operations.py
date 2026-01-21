# SPDX-License-Identifier: Apache-2.0
"""
Graph Conformance — Schema operations.

Asserts (Spec refs):
  • get_schema returns GraphSchema with expected structure     (§7.5)
  • schema is stable and JSON-serializable                     (§7.5)
"""
# SPDX-License-Identifier: Apache-2.0
"""
Graph Conformance — Schema operations.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Mapping

import pytest

from corpus_sdk.graph.graph_base import OperationContext as GraphContext, BaseGraphAdapter, GraphSchema, NotSupported

pytestmark = pytest.mark.asyncio


async def test_get_schema_capability_alignment(adapter: BaseGraphAdapter):
    caps = await adapter.capabilities()
    ctx = GraphContext(request_id="t_schema_cap", tenant="t")

    if not getattr(caps, "supports_schema", False):
        with pytest.raises(NotSupported):
            await adapter.get_schema(ctx=ctx)
        return

    schema = await adapter.get_schema(ctx=ctx)
    assert isinstance(schema, GraphSchema)
    assert isinstance(schema.nodes, Mapping)
    assert isinstance(schema.edges, Mapping)


async def test_schema_consistency_and_serializable_when_supported(adapter: BaseGraphAdapter):
    caps = await adapter.capabilities()
    if not getattr(caps, "supports_schema", False):
        with pytest.raises(NotSupported):
            await adapter.get_schema(ctx=GraphContext(request_id="t_schema_nsup", tenant="t"))
        return

    ctx = GraphContext(request_id="t_schema_consistent", tenant="t")
    s1 = await adapter.get_schema(ctx=ctx)
    s2 = await adapter.get_schema(ctx=ctx)

    assert isinstance(s1, GraphSchema)
    assert isinstance(s2, GraphSchema)

    assert set(s1.nodes.keys()) == set(s2.nodes.keys())
    assert set(s1.edges.keys()) == set(s2.edges.keys())

    assert isinstance(json.dumps(asdict(s1)), str)
