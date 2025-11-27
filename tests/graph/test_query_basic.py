# SPDX-License-Identifier: Apache-2.0
"""
Graph Conformance — Basic query behavior.

Asserts (Spec refs):
  • list-of-mapping results                                   (§7.3.2)
  • dialect + text validation                                 (§7.4, §17.2)
  • params binding accepts odd strings safely                 (§7.3.2)
  • empty/None params accepted                                (§7.3.2)
"""
import pytest

from corpus_sdk.graph.graph_base import (
    OperationContext as GraphContext,
    BadRequest,
    BaseGraphAdapter,
    GraphQuerySpec,
    QueryResult,
)

pytestmark = pytest.mark.asyncio


def clear_time_cache():
    """Placeholder to mirror previous API; no caching in this version."""
    pass


async def test_query_ops_returns_list_of_mappings(adapter: BaseGraphAdapter):
    """§7.3.2: Query must return list-of-mapping records."""
    clear_time_cache()
    ctx = GraphContext(request_id="t_query_rows", tenant="test")
    spec = GraphQuerySpec(text="RETURN 1 as value", dialect="cypher")
    result = await adapter.query(spec, ctx=ctx)

    assert isinstance(result, QueryResult)
    assert isinstance(result.records, list), "Query records must be a list"
    if result.records:
        assert isinstance(
            result.records[0], dict
        ), "Each query record must be a mapping/dict"


async def test_query_ops_requires_text(adapter: BaseGraphAdapter):
    """§7.4: Query must validate text parameter."""
    ctx = GraphContext(request_id="t_query_req", tenant="test")
    spec = GraphQuerySpec(text="", dialect="cypher")

    with pytest.raises(BadRequest) as exc_info:
        await adapter.query(spec, ctx=ctx)

    error_msg = str(exc_info.value).lower()
    assert any(term in error_msg for term in ["text", "empty", "required"]), (
        f"Error should mention text requirement: {error_msg}"
    )


async def test_query_ops_params_are_bound_safely(adapter: BaseGraphAdapter):
    """§7.3.2: Query parameters must be safely bound."""
    ctx = GraphContext(request_id="t_query_bind", tenant="test")
    spec = GraphQuerySpec(
        text="RETURN $param as value",
        dialect="cypher",
        params={"param": "'; DROP ALL; --"},
    )

    result = await adapter.query(spec, ctx=ctx)
    assert isinstance(result, QueryResult)
    assert isinstance(result.records, list)


async def test_query_ops_empty_params_allowed(adapter: BaseGraphAdapter):
    """§7.3.2: Empty or None parameters must be accepted."""
    ctx = GraphContext(request_id="t_query_empty_params", tenant="test")

    # None params
    spec_none = GraphQuerySpec(text="RETURN 1 as value", dialect="cypher", params=None)
    res_none = await adapter.query(spec_none, ctx=ctx)
    assert isinstance(res_none.records, list), "Query with None params should work"

    # Empty dict params
    spec_empty = GraphQuerySpec(text="RETURN 1 as value", dialect="cypher", params={})
    res_empty = await adapter.query(spec_empty, ctx=ctx)
    assert isinstance(res_empty.records, list), "Query with empty params should work"


async def test_query_ops_valid_dialect_required(adapter: BaseGraphAdapter):
    """§7.4: Query must use valid dialect from capabilities."""
    caps = await adapter.capabilities()
    if not caps.supported_query_dialects:
        pytest.skip("Adapter declares no dialects")

    ctx = GraphContext(request_id="t_query_valid_dialect", tenant="test")
    valid_dialect = caps.supported_query_dialects[0]

    spec = GraphQuerySpec(text="RETURN 1 as value", dialect=valid_dialect)
    result = await adapter.query(spec, ctx=ctx)
    assert isinstance(
        result.records, list
    ), f"Query with valid dialect '{valid_dialect}' should work"
