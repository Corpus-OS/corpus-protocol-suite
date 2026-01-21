# SPDX-License-Identifier: Apache-2.0
"""
Graph Conformance — Dialect validation.

Asserts (Spec refs):
  • Unknown dialects rejected with NotSupported               (§7.4)
  • Known dialects accepted                                   (§7.4)
  • Error messages include the offending dialect              (§7.4, §6.3)
"""
from __future__ import annotations

import pytest

from corpus_sdk.graph.graph_base import (
    OperationContext as GraphContext,
    NotSupported,
    BaseGraphAdapter,
    GraphQuerySpec,
    QueryResult,
)

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("dialect", ["unknown", "sql", "sparql"])
async def test_unknown_dialect_behavior_is_capability_consistent(adapter: BaseGraphAdapter, dialect: str):
    caps = await adapter.capabilities()
    ctx = GraphContext(request_id=f"t_dialect_unknown_{dialect}", tenant="t")
    declared = tuple(getattr(caps, "supported_query_dialects", ()) or ())
    spec = GraphQuerySpec(text="RETURN 1", dialect=dialect)

    if declared:
        if dialect in declared:
            res = await adapter.query(spec, ctx=ctx)
            assert isinstance(res, QueryResult)
            return
        with pytest.raises(NotSupported):
            await adapter.query(spec, ctx=ctx)
        return

    # Opaque dialect set: adapter may accept or reject; if it rejects, it should be NotSupported.
    try:
        res = await adapter.query(spec, ctx=ctx)
        assert isinstance(res, QueryResult)
    except NotSupported:
        pass


async def test_known_dialect_accepted_when_declared(adapter: BaseGraphAdapter):
    caps = await adapter.capabilities()
    declared = tuple(getattr(caps, "supported_query_dialects", ()) or ())
    ctx = GraphContext(request_id="t_dialect_known", tenant="t")

    if declared:
        res = await adapter.query(GraphQuerySpec(text="RETURN 1", dialect=declared[0]), ctx=ctx)
        assert isinstance(res.records, list)
        return

    res = await adapter.query(GraphQuerySpec(text="RETURN 1", dialect=None), ctx=ctx)
    assert isinstance(res.records, list)


async def test_error_message_includes_dialect_when_rejected_due_to_declared_list(adapter: BaseGraphAdapter):
    caps = await adapter.capabilities()
    declared = tuple(getattr(caps, "supported_query_dialects", ()) or ())
    if not declared:
        return

    dialect = "__definitely_not_supported__"
    if dialect in declared:
        return

    ctx = GraphContext(request_id="t_dialect_msg", tenant="t")
    with pytest.raises(NotSupported) as ei:
        await adapter.query(GraphQuerySpec(text="RETURN 1", dialect=dialect), ctx=ctx)
    assert dialect in str(ei.value)
