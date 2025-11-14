# SPDX-License-Identifier: Apache-2.0
"""
Graph Conformance — Capabilities shape and invariants.

Asserts (Spec refs):
  • Returns GraphCapabilities instance with identity fields  (§7.2, §6.2)
  • Dialects is a non-empty tuple[str, ...]                 (§7.2, §7.4)
  • Feature flags are booleans and sane                     (§7.2)
  • Batch/Rate-limit fields valid                           (§7.2, §6.2)
  • Repeated calls are idempotent (consistent)              (§6.2)
"""
import pytest

from corpus_sdk.graph.graph_base import GraphCapabilities

pytestmark = pytest.mark.asyncio


async def test_capabilities_returns_correct_type(adapter):
    caps = await adapter.capabilities()
    assert isinstance(caps, GraphCapabilities)


async def test_capabilities_identity_fields(adapter):
    caps = await adapter.capabilities()
    assert isinstance(caps.server, str) and caps.server
    assert isinstance(caps.version, str) and caps.version


async def test_capabilities_dialects_tuple(adapter):
    caps = await adapter.capabilities()
    assert isinstance(caps.dialects, tuple) and len(caps.dialects) > 0
    assert all(isinstance(d, str) for d in caps.dialects)


async def test_capabilities_feature_flags_are_boolean(adapter):
    caps = await adapter.capabilities()
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


async def test_capabilities_max_batch_ops_valid(adapter):
    caps = await adapter.capabilities()
    assert caps.max_batch_ops is None or (
        isinstance(caps.max_batch_ops, int) and caps.max_batch_ops > 0
    )


async def test_capabilities_rate_limit_unit(adapter):
    caps = await adapter.capabilities()
    assert caps.rate_limit_unit in ("requests_per_second", "tokens_per_minute")


async def test_capabilities_idempotency(adapter):
    c1 = await adapter.capabilities()
    c2 = await adapter.capabilities()
    assert (c1.server, c1.version, c1.dialects) == (c2.server, c2.version, c2.dialects)