# SPDX-License-Identifier: Apache-2.0
"""
Graph Conformance — Dialect validation.

Asserts (Spec refs):
  • Unknown dialects rejected with NotSupported               (§7.4)
  • Known dialects accepted                                   (§7.4)
  • Error messages include the offending dialect              (§7.4, §6.3)
"""
import pytest

from corpus_sdk.graph.graph_base import (
    OperationContext as GraphContext,
    NotSupported,
    BaseGraphAdapter,
    GraphQuerySpec,
)

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("dialect", ["unknown", "sql", "sparql"])
async def test_unknown_dialect_rejected(adapter: BaseGraphAdapter, dialect: str):
    caps = await adapter.capabilities()
    if dialect in caps.supported_query_dialects:
        pytest.skip(f"Adapter declares dialect '{dialect}'; cannot treat it as unknown")

    ctx = GraphContext(request_id=f"t_dialect_bad_{dialect}", tenant="t")
    spec = GraphQuerySpec(text="X", dialect=dialect)

    with pytest.raises(NotSupported):
        await adapter.query(spec, ctx=ctx)


async def test_known_dialect_accepted(adapter: BaseGraphAdapter):
    caps = await adapter.capabilities()
    if not caps.supported_query_dialects:
        pytest.skip("Adapter declares no dialects")

    dialect = caps.supported_query_dialects[0]
    ctx = GraphContext(request_id=f"t_dialect_ok_{dialect}", tenant="t")
    spec = GraphQuerySpec(text="RETURN 1", dialect=dialect)

    res = await adapter.query(spec, ctx=ctx)
    assert isinstance(res.records, list)


async def test_dialect_not_in_capabilities_raises_not_supported(
    adapter: BaseGraphAdapter,
):
    caps = await adapter.capabilities()
    unknown = "gremlin"
    if unknown in caps.supported_query_dialects:
        pytest.skip(f"Adapter unexpectedly supports '{unknown}' dialect")

    ctx = GraphContext(request_id="t_dialect_gremlin", tenant="t")
    spec = GraphQuerySpec(text="g.V().limit(1)", dialect=unknown)

    with pytest.raises(NotSupported) as ei:
        await adapter.query(spec, ctx=ctx)
    assert "gremlin" in str(ei.value)


async def test_error_message_includes_dialect_name(adapter: BaseGraphAdapter):
    caps = await adapter.capabilities()
    dialect = "gql"
    if dialect in caps.supported_query_dialects:
        pytest.skip(f"Adapter unexpectedly supports '{dialect}' dialect")

    ctx = GraphContext(request_id="t_dialect_gql", tenant="t")
    spec = GraphQuerySpec(text="{}", dialect=dialect)

    with pytest.raises(NotSupported) as ei:
        await adapter.query(spec, ctx=ctx)
    assert "gql" in str(ei.value)
