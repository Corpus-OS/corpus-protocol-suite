# SPDX-License-Identifier: Apache-2.0
"""
Graph Conformance — Capabilities shape and invariants.

Asserts:
  • Returns GraphCapabilities instance with identity fields
  • Dialects is a non-empty tuple[str, ...]
  • Feature flags are booleans and sane
  • Batch/Rate-limit fields valid
  • Repeated calls are idempotent (consistent)
"""
import pytest

from corpus_sdk.examples.graph.mock_graph_adapter import MockGraphAdapter
from corpus_sdk.graph.graph_base import GraphCapabilities

pytestmark = pytest.mark.asyncio


async def test_capabilities_returns_correct_type():
    """
    SPECIFICATION.md §5.1 — Capabilities discovery
    """
    caps = await MockGraphAdapter().capabilities()
    assert isinstance(caps, GraphCapabilities)


async def test_capabilities_identity_fields():
    caps = await MockGraphAdapter().capabilities()
    assert isinstance(caps.server, str) and caps.server
    assert isinstance(caps.version, str) and caps.version


async def test_capabilities_dialects_tuple():
    caps = await MockGraphAdapter().capabilities()
    assert isinstance(caps.dialects, tuple) and len(caps.dialects) > 0
    assert all(isinstance(d, str) for d in caps.dialects)


async def test_capabilities_feature_flags_are_boolean():
    caps = await MockGraphAdapter().capabilities()
    flags = [
        caps.supports_txn,
        caps.supports_schema_ops,
        caps.idempotent_writes,
        caps.supports_multi_tenant,
        caps.supports_streaming,
        caps.supports_bulk_ops,
        caps.supports_deadline,
    ]
    assert all(isinstance(f, bool) for f in flags)


async def test_capabilities_max_batch_ops_valid():
    caps = await MockGraphAdapter().capabilities()
    assert caps.max_batch_ops is None or (isinstance(caps.max_batch_ops, int) and caps.max_batch_ops > 0)


async def test_capabilities_rate_limit_unit():
    caps = await MockGraphAdapter().capabilities()
    assert caps.rate_limit_unit in ("requests_per_second", "tokens_per_minute")


async def test_capabilities_idempotency():
    a = MockGraphAdapter()
    c1 = await a.capabilities()
    c2 = await a.capabilities()
    assert (c1.server, c1.version, c1.dialects) == (c2.server, c2.version, c2.dialects)
