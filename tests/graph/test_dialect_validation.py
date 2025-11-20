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
)

pytestmark = pytest.mark.asyncio


def make_ctx(ctx_cls, **kwargs):
    """Local helper to construct an OperationContext."""
    return ctx_cls(**kwargs)


@pytest.mark.parametrize("dialect", ["unknown", "sql", "sparql"])
async def test_unknown_dialect_rejected(adapter, dialect):
    caps = await adapter.capabilities()
    # If the adapter *does* support this dialect, we can't treat it as "unknown".
    if dialect in getattr(caps, "dialects", []):
        pytest.skip(f"Adapter declares dialect '{dialect}'; cannot treat it as unknown")

    ctx = make_ctx(GraphContext, request_id=f"t_dialect_bad_{dialect}", tenant="t")
    with pytest.raises(NotSupported):
        await adapter.query(dialect=dialect, text="X", ctx=ctx)


async def test_known_dialect_accepted(adapter):
    caps = await adapter.capabilities()
    if not getattr(caps, "dialects", []):
        pytest.skip("Adapter declares no dialects")

    # Use the first declared dialect as a known-good dialect.
    dialect = caps.dialects[0]
    ctx = make_ctx(GraphContext, request_id=f"t_dialect_ok_{dialect}", tenant="t")
    rows = await adapter.query(dialect=dialect, text="RETURN 1", ctx=ctx)
    assert isinstance(rows, list)


async def test_dialect_not_in_capabilities_raises_not_supported(adapter):
    caps = await adapter.capabilities()
    unknown = "gremlin"
    if unknown in getattr(caps, "dialects", []):
        pytest.skip(f"Adapter unexpectedly supports '{unknown}' dialect")

    ctx = make_ctx(GraphContext, request_id="t_dialect_gremlin", tenant="t")
    with pytest.raises(NotSupported) as ei:
        await adapter.query(dialect=unknown, text="g.V().limit(1)", ctx=ctx)
    assert "gremlin" in str(ei.value)


async def test_error_message_includes_dialect_name(adapter):
    caps = await adapter.capabilities()
    dialect = "gql"
    if dialect in getattr(caps, "dialects", []):
        pytest.skip(f"Adapter unexpectedly supports '{dialect}' dialect")

    ctx = make_ctx(GraphContext, request_id="t_dialect_gql", tenant="t")
    with pytest.raises(NotSupported) as ei:
        await adapter.query(dialect=dialect, text="{}", ctx=ctx)
    assert "gql" in str(ei.value)
