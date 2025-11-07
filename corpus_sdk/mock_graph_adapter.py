# SPDX-License-Identifier: Apache-2.0
"""
Mock Graph adapter used in Corpus SDK example scripts.

Implements BaseGraphAdapter methods for demonstration purposes only.
Simulates latency, simple CRUD, Cypher-like queries, streaming rows, and batch ops.
"""

from __future__ import annotations
import asyncio
import random
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Iterable, List, Mapping, Optional, Tuple

from corpus_sdk.graph.graph_base import (
    BaseGraphAdapter,
    GraphCapabilities,
    OperationContext as GraphContext,
    GraphID,
    BadRequest,
    NotSupported,
    Unavailable,
    ResourceExhausted,
    HealthStatus,
)
from corpus_sdk.examples.common.ctx import make_ctx
from corpus_sdk.examples.common.printing import box, print_kv, print_json


@dataclass
class MockGraphAdapter(BaseGraphAdapter):
    """A mock Graph adapter for protocol demonstrations."""
    name: str = "mock-graph"
    failure_rate: float = 0.10  # 10% chance of simulated transient failure
    supported_dialects: Tuple[str, ...] = ("cypher", "opencypher")

    # -----------------------------
    # Capabilities & health
    # -----------------------------
    async def _do_capabilities(self) -> GraphCapabilities:
        # Only use fields defined in GraphCapabilities
        return GraphCapabilities(
            server="mock-graph",
            version="1.0.0",
            dialects=self.supported_dialects,
            supports_txn=True,
            supports_schema_ops=True,
            max_batch_ops=1_000,
            retryable_codes=(),            # none for the mock
            rate_limit_unit="requests_per_second",
            max_qps=500,
            idempotent_writes=False,
            supports_multi_tenant=True,
            supports_streaming=True,
            supports_bulk_ops=True,
            supports_deadline=True,
        )

    async def _do_health(self, *, ctx: Optional[GraphContext] = None) -> Dict[str, Any]:
        # Occasionally report degraded; use HealthStatus constants to align with base
        if random.random() < 0.20:
            return {"status": HealthStatus.DEGRADED, "server": "mock-graph", "version": "1.0.0"}
        return {"status": HealthStatus.OK, "server": "mock-graph", "version": "1.0.0"}

    # -----------------------------
    # CRUD
    # -----------------------------
    async def _do_create_vertex(
        self, label: str, props: Mapping[str, Any], *, ctx: Optional[GraphContext] = None
    ) -> GraphID:
        self._maybe_fail("create_vertex")
        if not label:
            raise BadRequest("label must be non-empty")
        if not isinstance(props, Mapping):
            raise BadRequest("props must be a mapping")
        await asyncio.sleep(0.01)
        return GraphID(f"v:{label}:{abs(hash(tuple(sorted(props.items())))) % 10_000}")

    async def _do_delete_vertex(self, vertex_id: str, *, ctx: Optional[GraphContext] = None) -> None:
        self._maybe_fail("delete_vertex")
        if not vertex_id:
            raise BadRequest("vertex_id must be provided")
        await asyncio.sleep(0.005)
        return None

    async def _do_create_edge(
        self,
        label: str,
        from_id: str,
        to_id: str,
        props: Mapping[str, Any],
        *,
        ctx: Optional[GraphContext] = None,
    ) -> GraphID:
        self._maybe_fail("create_edge")
        if not all([label, from_id, to_id]):
            raise BadRequest("edge requires label, from_id, to_id")
        if not isinstance(props, Mapping):
            raise BadRequest("props must be a mapping")
        await asyncio.sleep(0.01)
        return GraphID(f"e:{label}:{abs(hash((from_id, to_id))) % 10_000}")

    async def _do_delete_edge(self, edge_id: str, *, ctx: Optional[GraphContext] = None) -> None:
        self._maybe_fail("delete_edge")
        if not edge_id:
            raise BadRequest("edge_id must be provided")
        await asyncio.sleep(0.005)
        return None

    # -----------------------------
    # Query & streaming
    # -----------------------------
    async def _do_query(
        self,
        *,
        dialect: str,
        text: str,
        params: Mapping[str, Any],
        ctx: Optional[GraphContext] = None,
    ) -> List[Mapping[str, Any]]:
        self._maybe_fail("query")
        self._guard_dialect(dialect)
        if not text:
            raise BadRequest("query text must be non-empty")
        await asyncio.sleep(0.02)
        # Return deterministic mock rows based on params
        seed = abs(hash((text, tuple(sorted((params or {}).items()))))) % 3 + 1
        return [{"row": i + 1, "ok": True, "dialect": dialect} for i in range(seed)]

    async def _do_stream_query(
        self,
        *,
        dialect: str,
        text: str,
        params: Mapping[str, Any],
        ctx: Optional[GraphContext] = None,
    ) -> AsyncIterator[Mapping[str, Any]]:
        self._maybe_fail("stream_query")
        self._guard_dialect(dialect)
        if not text:
            raise BadRequest("query text must be non-empty")
        # Stream a few rows with tiny delays
        count = abs(hash(text)) % 4 + 2
        for i in range(count):
            await asyncio.sleep(0.01)
            yield {"row": i + 1, "dialect": dialect, "ok": True}
        # no explicit final sentinel; iterator close is the end

    # -----------------------------
    # Bulk & batch
    # -----------------------------
    async def _do_bulk_vertices(
        self,
        vertices: List[Tuple[str, Mapping[str, Any]]],
        *,
        ctx: Optional[GraphContext] = None,
    ) -> List[GraphID]:
        self._maybe_fail("bulk_vertices")
        out: List[GraphID] = []
        i = 0
        async_sleep_every = 50
        for label, props in vertices:
            if not label or not isinstance(props, Mapping):
                raise BadRequest("each vertex must be (label, props_mapping)")
            out.append(GraphID(f"v:{label}:{i}"))
            i += 1
            if i % async_sleep_every == 0:
                await asyncio.sleep(0.001)
        return out

    async def _do_batch(
        self,
        ops: List[Mapping[str, Any]],
        *,
        ctx: Optional[GraphContext] = None,
    ) -> List[Mapping[str, Any]]:
        self._maybe_fail("batch")
        results: List[Mapping[str, Any]] = []
        for op in ops:
            kind = op.get("type")
            if kind == "create_vertex":
                vid = await self._do_create_vertex(op.get("label", ""), op.get("props", {}), ctx=ctx)
                results.append({"ok": True, "type": kind, "id": str(vid)})
            elif kind == "create_edge":
                eid = await self._do_create_edge(
                    op.get("label", ""),
                    str(op.get("from_id", "")),
                    str(op.get("to_id", "")),
                    op.get("props", {}),
                    ctx=ctx,
                )
                results.append({"ok": True, "type": kind, "id": str(eid)})
            else:
                results.append({"ok": False, "type": kind or "<unknown>", "error": "NOT_SUPPORTED"})
        return results

    # -----------------------------
    # Schema (optional in V1)
    # -----------------------------
    async def _do_get_schema(self, *, ctx: Optional[GraphContext] = None) -> Dict[str, Any]:
        self._maybe_fail("get_schema")
        await asyncio.sleep(0.005)
        return {"nodes": ["User", "Doc"], "edges": ["READ", "LINKS"]}

    # -----------------------------
    # Internals
    # -----------------------------
    def _guard_dialect(self, dialect: str) -> None:
        # Base validates known dialects; this additionally checks our declared support set.
        if dialect not in self.supported_dialects:
            raise NotSupported(f"dialect '{dialect}' not supported; supported={self.supported_dialects}")

    def _maybe_fail(self, op: str) -> None:
        """Inject transient failures for demonstration purposes."""
        if random.random() < self.failure_rate:
            # Flip between capacity and rate-limit style failures
            if random.random() < 0.5:
                raise Unavailable(f"Mocked {op} unavailable", retry_after_ms=500)
            raise ResourceExhausted(f"Mocked {op} rate-limited", retry_after_ms=800)


# ---------------------------------------------------------------------------
# Demo usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """Run this module directly to see mock graph behavior in action."""

    async def _demo() -> None:
        random.seed(7)  # deterministic demo
        box("MockGraphAdapter Demo")
        adapter = MockGraphAdapter(failure_rate=0.25)
        ctx = make_ctx(GraphContext, tenant="demo-tenant")

        # --- Capabilities ---
        print("\n=== CAPABILITIES ===")
        caps = await adapter.capabilities()
        print_json(caps.__dict__)

        # --- Health ---
        print("\n=== HEALTH ===")
        health = await adapter.health(ctx=ctx)
        print_kv(health)

        # --- CRUD ---
        print("\n=== CRUD ===")
        v1 = await adapter.create_vertex("User", {"name": "Ada"}, ctx=ctx)
        v2 = await adapter.create_vertex("User", {"name": "Grace"}, ctx=ctx)
        e = await adapter.create_edge("FOLLOWS", str(v1), str(v2), {"since": 2021}, ctx=ctx)
        print_kv({"v1": v1, "v2": v2, "edge": e})
        await adapter.delete_edge(str(e), ctx=ctx)
        await adapter.delete_vertex(str(v2), ctx=ctx)

        # --- Query ---
        print("\n=== QUERY (cypher) ===")
        rows = await adapter.query(
            dialect="cypher",
            text="MATCH (u:User) RETURN u LIMIT 3",
            params=None,
            ctx=ctx,
        )
        print_json(rows)

        # --- Stream Query ---
        print("\n=== STREAM QUERY (cypher) ===")
        async for row in adapter.stream_query(
            dialect="cypher",
            text="MATCH (u:User)-[:READ]->(d:Doc) RETURN u,d LIMIT 5",
            params=None,
            ctx=ctx,
        ):
            print(row)

        # --- Batch ---
        print("\n=== BATCH OPS ===")
        results = await adapter.batch(
            [
                {"type": "create_vertex", "label": "Doc", "props": {"id": "d1"}},
                {
                    "type": "create_edge",
                    "label": "READ",
                    "from_id": str(v1),
                    "to_id": "v:Doc:d1",
                    "props": {},
                },
                {"type": "unknown_op"},
            ],
            ctx=ctx,
        )
        print_json(results)

        # --- Schema ---
        print("\n=== SCHEMA ===")
        schema = await adapter.get_schema(ctx=ctx)
        print_json(schema)

        print("\n[done]")

    asyncio.run(_demo())
