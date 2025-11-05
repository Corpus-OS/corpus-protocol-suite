# SPDX-License-Identifier: Apache-2.0
"""
Graph Conformance — Basic query behavior.

Asserts (Spec refs):
  • list-of-mapping results                                   (§7.3.2)
  • dialect + text validation                                  (§7.4, §17.2)
  • params binding accepts odd strings safely                  (§7.3.2)
  • empty/None params accepted                                 (§7.3.2)
"""
import pytest

from corpus_sdk.examples.graph.mock_graph_adapter import MockGraphAdapter
from corpus_sdk.graph.graph_base import (
    OperationContext as GraphContext,
    BadRequest,
)
from corpus_sdk.examples.common.ctx import make_ctx, clear_time_cache

pytestmark = pytest.mark.asyncio


async def test_query_returns_list_of_mappings():
    clear_time_cache()
    a = MockGraphAdapter()
    ctx = make_ctx(GraphContext, request_id="t_query_rows", tenant="test")
    rows = await a.query(dialect="cypher", text="MATCH (n) RETURN n LIMIT 2", ctx=ctx)
    assert isinstance(rows, list)
    assert rows and isinstance(rows[0], dict)


async def test_query_requires_dialect_and_text():
    a = MockGraphAdapter()
    ctx = make_ctx(GraphContext, request_id="t_query_req", tenant="test")
    with pytest.raises(BadRequest):
        await a.query(dialect="cypher", text="", ctx=ctx)


async def test_query_params_are_bound_safely():
    a = MockGraphAdapter()
    ctx = make_ctx(GraphContext, request_id="t_query_bind", tenant="test")
    rows = await a.query(
        dialect="cypher",
        text="MATCH (u:User {email: $email}) RETURN u",
        params={"email": "'; DROP ALL; --"},
        ctx=ctx,
    )
    assert isinstance(rows, list)


async def test_query_empty_params_allowed():
    a = MockGraphAdapter()
    ctx = make_ctx(GraphContext, request_id="t_query_empty_params", tenant="test")
    rows = await a.query(dialect="cypher", text="RETURN 1", params=None, ctx=ctx)
    assert isinstance(rows, list)
