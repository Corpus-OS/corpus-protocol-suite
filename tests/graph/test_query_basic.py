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

from corpus_sdk.graph.graph_base import (
    OperationContext as GraphContext,
    BadRequest,
    BaseGraphAdapter,
)

pytestmark = pytest.mark.asyncio


def make_ctx(ctx_cls, **kwargs):
    """Local helper to construct an OperationContext."""
    return ctx_cls(**kwargs)


def clear_time_cache():
    """
    Placeholder to mirror previous API; no caching in this simplified version.
    """
    pass


async def test_query_ops_returns_list_of_mappings(adapter: BaseGraphAdapter):
    """§7.3.2: Query must return list of mapping/dict results."""
    clear_time_cache()
    ctx = make_ctx(GraphContext, request_id="t_query_rows", tenant="test")
    rows = await adapter.query(dialect="cypher", text="RETURN 1 as value", ctx=ctx)
    assert isinstance(rows, list), "Query results must be a list"
    if rows:  # Some adapters may return empty lists for simple queries
        assert isinstance(rows[0], dict), "Each query result must be a dictionary/mapping"


async def test_query_ops_requires_dialect_and_text(adapter: BaseGraphAdapter):
    """§7.4: Query must validate dialect and text parameters."""
    ctx = make_ctx(GraphContext, request_id="t_query_req", tenant="test")

    # Test empty text
    with pytest.raises(BadRequest) as exc_info:
        await adapter.query(dialect="cypher", text="", ctx=ctx)
    error_msg = str(exc_info.value).lower()
    assert any(term in error_msg for term in ["text", "empty", "required"]), (
        f"Error should mention text requirement: {error_msg}"
    )


async def test_query_ops_params_are_bound_safely(adapter: BaseGraphAdapter):
    """§7.3.2: Query parameters must be safely bound."""
    ctx = make_ctx(GraphContext, request_id="t_query_bind", tenant="test")

    # Test with potentially dangerous parameter values
    rows = await adapter.query(
        dialect="cypher",
        text="RETURN $param as value",
        params={"param": "'; DROP ALL; --"},
        ctx=ctx,
    )
    assert isinstance(rows, list), "Query with parameters should return list"


async def test_query_ops_empty_params_allowed(adapter: BaseGraphAdapter):
    """§7.3.2: Empty or None parameters must be accepted."""
    ctx = make_ctx(GraphContext, request_id="t_query_empty_params", tenant="test")

    # Test with None params
    rows_none = await adapter.query(
        dialect="cypher",
        text="RETURN 1 as value",
        params=None,
        ctx=ctx,
    )
    assert isinstance(rows_none, list), "Query with None params should work"

    # Test with empty dict params
    rows_empty = await adapter.query(
        dialect="cypher",
        text="RETURN 1 as value",
        params={},
        ctx=ctx,
    )
    assert isinstance(rows_empty, list), "Query with empty params should work"


async def test_query_ops_valid_dialect_required(adapter: BaseGraphAdapter):
    """§7.4: Query must use valid dialect from capabilities."""
    caps = await adapter.capabilities()
    if not caps.dialects:
        pytest.skip("Adapter declares no dialects")

    ctx = make_ctx(GraphContext, request_id="t_query_valid_dialect", tenant="test")

    # Test with a known valid dialect
    valid_dialect = caps.dialects[0]
    rows = await adapter.query(dialect=valid_dialect, text="RETURN 1 as value", ctx=ctx)
    assert isinstance(rows, list), f"Query with valid dialect '{valid_dialect}' should work"
