# SPDX-License-Identifier: Apache-2.0
"""
Graph Conformance — Dialect validation.

Asserts (Spec refs):
  • Unknown dialects rejected with NotSupported               (§7.4)
  • Known dialects accepted                                   (§7.4)
  • Error messages include the offending dialect              (§7.4, §6.3)
"""
import pytest

from corpus_sdk.examples.graph.mock_graph_adapter import MockGraphAdapter
from corpus_sdk.graph.graph_base import (
    OperationContext as GraphContext,
    NotSupported,
)
from corpus_sdk.examples.common.ctx import make_ctx

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("dialect", ["unknown", "sql", "sparql"])
async def test_unknown_dialect_rejected(dialect):
    a = MockGraphAdapter()
    ctx = make_ctx(GraphContext, request_id=f"t_dialect_bad_{dialect}", tenant="t")
    with pytest.raises(NotSupported):
        await a.query(dialect=dialect, text="X", ctx=ctx)


@pytest.mark.parametrize("dialect", ["cypher", "opencypher"])
async def test_known_dialect_accepted(dialect):
    a = MockGraphAdapter()
    ctx = make_ctx(GraphContext, request_id=f"t_dialect_ok_{dialect}", tenant="t")
    rows = await a.query(dialect=dialect, text="RETURN 1", ctx=ctx)
    assert isinstance(rows, list)


async def test_dialect_not_in_capabilities_raises_not_supported():
    a = MockGraphAdapter()
    ctx = make_ctx(GraphContext, request_id="t_dialect_gremlin", tenant="t")
    with pytest.raises(NotSupported) as ei:
        await a.query(dialect="gremlin", text="g.V().limit(1)", ctx=ctx)
    assert "gremlin" in str(ei.value)


async def test_error_message_includes_dialect_name():
    a = MockGraphAdapter()
    ctx = make_ctx(GraphContext, request_id="t_dialect_gql", tenant="t")
    with pytest.raises(NotSupported) as ei:
        await a.query(dialect="gql", text="{}", ctx=ctx)
    assert "gql" in str(ei.value)
