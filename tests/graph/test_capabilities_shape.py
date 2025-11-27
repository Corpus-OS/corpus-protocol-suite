# SPDX-License-Identifier: Apache-2.0
"""
Graph Conformance — Capabilities shape and invariants.

Asserts (Spec refs):
  • Returns GraphCapabilities instance with identity fields  (§7.2, §6.2)
  • Dialects is a non-empty tuple[str, ...]                 (§7.2, §7.4)
  • Feature flags are booleans and sane                     (§7.2)
  • Batch fields valid                                      (§7.2, §6.2)
  • Repeated calls are idempotent (consistent)              (§6.2)
"""
import pytest

from corpus_sdk.graph.graph_base import GraphCapabilities, GRAPH_PROTOCOL_ID

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
    assert isinstance(caps.supported_query_dialects, tuple)
    assert len(caps.supported_query_dialects) > 0
    assert all(isinstance(d, str) for d in caps.supported_query_dialects)


async def test_capabilities_feature_flags_are_boolean(adapter):
    caps = await adapter.capabilities()
    flags = [
        caps.supports_stream_query,
        caps.supports_namespaces,
        caps.supports_property_filters,
        caps.supports_bulk_vertices,
        caps.supports_batch,
        caps.supports_schema,
        caps.idempotent_writes,
        caps.supports_multi_tenant,
        caps.supports_deadline,
        caps.supports_transaction,
        caps.supports_traversal,
        caps.supports_path_queries,
    ]
    assert all(isinstance(f, bool) for f in flags)


async def test_capabilities_max_batch_ops_valid(adapter):
    caps = await adapter.capabilities()
    assert caps.max_batch_ops is None or (
        isinstance(caps.max_batch_ops, int) and caps.max_batch_ops > 0
    )


async def test_capabilities_protocol(adapter):
    caps = await adapter.capabilities()
    assert isinstance(caps.protocol, str) and caps.protocol
    assert caps.protocol == GRAPH_PROTOCOL_ID


async def test_capabilities_idempotency(adapter):
    c1 = await adapter.capabilities()
    c2 = await adapter.capabilities()
    assert (
        c1.server,
        c1.version,
        c1.supported_query_dialects,
    ) == (
        c2.server,
        c2.version,
        c2.supported_query_dialects,
    )
