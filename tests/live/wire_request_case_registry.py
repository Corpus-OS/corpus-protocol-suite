# SPDX-License-Identifier: Apache-2.0
"""
End-to-end wire-level request validation for the CORPUS Protocol.

This module exercises the *real adapter* and checks that for every
protocol operation (`op`) defined in the wire request case registry:

  1. The adapter builds a canonical wire envelope:
        { "op": <str>, "ctx": <obj>, "args": <obj> }
  2. The envelope validates against the corresponding
        *.envelope.request JSON Schema via the shared SchemaRegistry.
  3. Operation-specific invariants on `args` are satisfied where
     JSON Schema alone is not expressive enough.

It ties together:

  * tests/live/wire_request_case_registry.py
      → canonical list of wire request cases
  * tests/utils/schema_registry.py
      → SchemaRegistry + assert_valid(schema_id, envelope, ...)
  * the adapter fixture in tests/conftest.py
      → real adapter under test

This file focuses on *live wire requests*: “if a user called this op
through our adapter right now, would the raw JSON we send over the wire
actually match the protocol?”
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import pytest

from tests.utils.schema_registry import assert_valid
from tests.live.wire_request_case_registry import WireRequestCase, get_pytest_params


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


def _get_builder(adapter: Any, case: WireRequestCase) -> Callable[[], Dict[str, Any]]:
    """
    Resolve the adapter builder method for this case, or skip if not implemented.

    This lets adapters bring ops online incrementally while keeping the
    wire conformance tests in place.
    """
    builder = getattr(adapter, case.build_method, None)
    if builder is None or not callable(builder):
        pytest.skip(
            f"Adapter does not implement '{case.build_method}' "
            f"for wire case '{case.id}'"
        )
    return builder


def _assert_envelope_common(envelope: Dict[str, Any], case: WireRequestCase) -> None:
    """Basic envelope shape + ctx invariants shared by all components."""
    assert isinstance(envelope, dict), f"{case.id}: envelope must be dict, got {type(envelope)}"
    missing = {"op", "ctx", "args"} - set(envelope.keys())
    assert not missing, f"{case.id}: envelope missing required top-level keys: {missing}"

    # op
    op = envelope["op"]
    assert op == case.op, f"{case.id}: op mismatch; expected {case.op!r}, got {op!r}"

    # ctx
    ctx = envelope["ctx"]
    assert isinstance(ctx, dict), f"{case.id}: ctx must be object, got {type(ctx)}"
    assert "request_id" in ctx, f"{case.id}: ctx.request_id required"
    assert isinstance(ctx["request_id"], str) and ctx["request_id"], (
        f"{case.id}: ctx.request_id must be non-empty string"
    )

    if "deadline_ms" in ctx:
        dm = ctx["deadline_ms"]
        assert isinstance(dm, int) and dm > 0, (
            f"{case.id}: ctx.deadline_ms must be positive int, got {dm!r}"
        )

    if "tenant" in ctx and ctx["tenant"] is not None:
        assert isinstance(ctx["tenant"], str), f"{case.id}: ctx.tenant must be string if present"

    # args
    args = envelope["args"]
    assert isinstance(args, dict), f"{case.id}: args must be object, got {type(args)}"


# ---------------------------------------------------------------------------
# Embedding-specific helpers
# ---------------------------------------------------------------------------


def _extract_embedding_text_inputs(args: Dict[str, Any], case_id: str) -> List[str]:
    """
    Normalize embedding inputs to a list of strings.

    Rules:
      * Prefer 'texts' if present – must be list[str]
      * Else use 'text' – must be string, normalized to [text]
      * No silent normalization of incorrect types:
          - texts=str → FAIL (type error)
          - text=list → FAIL (type error)
    """
    if "texts" in args and args["texts"] is not None:
        texts = args["texts"]
        if isinstance(texts, list) and all(isinstance(t, str) for t in texts):
            return texts
        raise AssertionError(
            f"{case_id}: args.texts must be list[str]; "
            f"got {type(texts).__name__} ({texts!r})"
        )

    if "text" in args and args["text"] is not None:
        text = args["text"]
        if isinstance(text, str):
            return [text]
        raise AssertionError(
            f"{case_id}: args.text must be str when present; "
            f"got {type(text).__name__} ({text!r})"
        )

    raise AssertionError(
        f"{case_id}: embedding args must provide 'text' (str) or 'texts' (list[str])"
    )


def _assert_embedding_embed_args(args: Dict[str, Any], case: WireRequestCase) -> None:
    """Embedding.embed + embed_batch arg invariants."""
    texts = _extract_embedding_text_inputs(args, case.id)

    if case.op == "embedding.embed":
        assert len(texts) == 1, f"{case.id}: embed expects a single input; got {len(texts)}"
    elif case.op == "embedding.embed_batch":
        assert len(texts) >= 1, f"{case.id}: embed_batch expects at least one input"
    else:
        # safeguard; shouldn't happen
        raise AssertionError(f"{case.id}: unexpected op for embed assertion: {case.op!r}")

    model = args.get("model")
    assert isinstance(model, str) and model, f"{case.id}: args.model must be non-empty string"

    if "truncate" in args and args["truncate"] is not None:
        assert isinstance(args["truncate"], bool), f"{case.id}: args.truncate must be bool"

    if "normalize" in args and args["normalize"] is not None:
        assert isinstance(args["normalize"], bool), f"{case.id}: args.normalize must be bool"


def _assert_embedding_count_tokens_args(args: Dict[str, Any], case: WireRequestCase) -> None:
    """
    Embedding.count_tokens invariants for both single + batch shapes.

    We exercise single/batch via different cases, but only enforce
    semantic rules here, not the exact field name.
    """
    texts = _extract_embedding_text_inputs(args, case.id)
    assert len(texts) >= 1, f"{case.id}: count_tokens requires at least one input"

    model = args.get("model")
    assert isinstance(model, str) and model, f"{case.id}: args.model must be non-empty string"


# ---------------------------------------------------------------------------
# Vector-specific helpers
# ---------------------------------------------------------------------------


def _assert_vector_query_args(args: Dict[str, Any], case: WireRequestCase) -> None:
    """
    Minimal invariants for vector.query-style ops.

    We assume the adapter already built the concrete shape for the backend,
    but we still sanity-check the key semantic fields.
    """
    query = args.get("query")
    assert isinstance(query, dict), f"{case.id}: args.query must be object"

    namespace = query.get("namespace")
    assert isinstance(namespace, str) and namespace, (
        f"{case.id}: query.namespace must be non-empty string"
    )

    # For raw-vector queries we expect a numeric 'vector' list.
    # Text-based queries (query_by_text) can still carry 'vector', or they may
    # have a 'text' field instead; we only assert vector when present.
    vector = query.get("vector")
    if vector is not None:
        assert isinstance(vector, list) and vector, (
            f"{case.id}: query.vector must be non-empty list when present"
        )
        assert all(isinstance(x, (int, float)) for x in vector), (
            f"{case.id}: query.vector must be numeric list"
        )


def _assert_vector_upsert_args(args: Dict[str, Any], case: WireRequestCase) -> None:
    vectors = args.get("vectors")
    assert isinstance(vectors, list) and vectors, (
        f"{case.id}: args.vectors must be non-empty list"
    )
    for i, v in enumerate(vectors):
        assert isinstance(v, dict), f"{case.id}: vectors[{i}] must be object"
        assert "id" in v, f"{case.id}: vectors[{i}] missing 'id'"
        assert "vector" in v, f"{case.id}: vectors[{i}] missing 'vector'"


def _assert_vector_delete_args(args: Dict[str, Any], case: WireRequestCase) -> None:
    """
    Lightweight invariants for vector.delete-style ops.

    Typically allow:
      - delete by IDs
      - delete by filter
      - delete all in namespace
    We simply assert that at least *one* selector is present.
    """
    namespace = args.get("namespace")
    assert isinstance(namespace, str) and namespace, (
        f"{case.id}: args.namespace must be non-empty string"
    )

    ids = args.get("ids")
    flt = args.get("filter")
    delete_all = args.get("delete_all", False)

    assert ids or flt or delete_all, (
        f"{case.id}: vector.delete must specify ids, filter, or delete_all"
    )


def _assert_vector_namespace_args(args: Dict[str, Any], case: WireRequestCase) -> None:
    """
    Minimal check for namespace create/delete.
    """
    namespace = args.get("namespace")
    assert isinstance(namespace, str) and namespace, (
        f"{case.id}: args.namespace must be non-empty string"
    )


# ---------------------------------------------------------------------------
# Graph-specific helpers
# ---------------------------------------------------------------------------


def _assert_graph_query_args(args: Dict[str, Any], case: WireRequestCase) -> None:
    """
    Invariants for graph.query and graph.stream_query wire args.

    WireGraphHandler expects args to match GraphQuerySpec(**args):

        {
          "text": "...",
          "dialect": "...",        # optional
          "params": {...},         # optional
          "namespace": "...",      # optional
          "timeout_ms": 123,       # optional
          "stream": false          # used by routers/adapters
        }
    """
    text = args.get("text")
    assert isinstance(text, str) and text.strip(), (
        f"{case.id}: args.text must be non-empty string"
    )

    dialect = args.get("dialect")
    if dialect is not None:
        assert isinstance(dialect, str) and dialect, (
            f"{case.id}: args.dialect must be non-empty string when present"
        )

    params = args.get("params")
    if params is not None:
        assert isinstance(params, dict), f"{case.id}: args.params must be object when present"

    timeout_ms = args.get("timeout_ms")
    if timeout_ms is not None:
        assert isinstance(timeout_ms, int) and timeout_ms > 0, (
            f"{case.id}: args.timeout_ms must be positive int when present"
        )

    stream = args.get("stream")
    if stream is not None:
        assert isinstance(stream, bool), f"{case.id}: args.stream must be bool when present"


def _assert_graph_upsert_nodes_args(args: Dict[str, Any], case: WireRequestCase) -> None:
    """
    Invariants for graph.upsert_nodes wire args.

    WireGraphHandler builds UpsertNodesSpec from:

        {
          "nodes": [ { "id": "...", ... }, ... ],
          "namespace": "..."
        }
    """
    nodes = args.get("nodes")
    assert isinstance(nodes, list) and nodes, (
        f"{case.id}: args.nodes must be non-empty list"
    )
    for i, node in enumerate(nodes):
        assert isinstance(node, dict), f"{case.id}: nodes[{i}] must be object"
        nid = node.get("id")
        assert isinstance(nid, str) and nid, f"{case.id}: nodes[{i}].id must be non-empty string"

    if "namespace" in args and args["namespace"] is not None:
        assert isinstance(args["namespace"], str), (
            f"{case.id}: args.namespace must be string when present"
        )


def _assert_graph_upsert_edges_args(args: Dict[str, Any], case: WireRequestCase) -> None:
    """
    Invariants for graph.upsert_edges wire args.

    WireGraphHandler builds UpsertEdgesSpec from:

        {
          "edges": [
            { "id": "...", "src": "...", "dst": "...", "label": "...", ... },
            ...
          ],
          "namespace": "..."
        }
    """
    edges = args.get("edges")
    assert isinstance(edges, list) and edges, (
        f"{case.id}: args.edges must be non-empty list"
    )
    for i, edge in enumerate(edges):
        assert isinstance(edge, dict), f"{case.id}: edges[{i}] must be object"
        for key in ("id", "src", "dst", "label"):
            val = edge.get(key)
            assert isinstance(val, str) and val, (
                f"{case.id}: edges[{i}].{key} must be non-empty string"
            )

    if "namespace" in args and args["namespace"] is not None:
        assert isinstance(args["namespace"], str), (
            f"{case.id}: args.namespace must be string when present"
        )


def _assert_graph_delete_nodes_args(args: Dict[str, Any], case: WireRequestCase) -> None:
    """
    Invariants for graph.delete_nodes wire args.

    WireGraphHandler builds DeleteNodesSpec from:

        {
          "ids": [ "...", ... ],           # may be empty if filter given
          "namespace": "...",
          "filter": { ... }                # optional
        }
    """
    ids = args.get("ids") or []
    flt = args.get("filter")

    assert isinstance(ids, list), f"{case.id}: args.ids must be list"
    if ids:
        for i, nid in enumerate(ids):
            assert isinstance(nid, str) and nid, (
                f"{case.id}: ids[{i}] must be non-empty string"
            )

    if flt is not None:
        assert isinstance(flt, dict), f"{case.id}: args.filter must be object when present"

    assert ids or flt, f"{case.id}: delete_nodes requires ids or filter"

    if "namespace" in args and args["namespace"] is not None:
        assert isinstance(args["namespace"], str), (
            f"{case.id}: args.namespace must be string when present"
        )


def _assert_graph_delete_edges_args(args: Dict[str, Any], case: WireRequestCase) -> None:
    """
    Invariants for graph.delete_edges wire args.

    Same structure as delete_nodes but for edge IDs.
    """
    ids = args.get("ids") or []
    flt = args.get("filter")

    assert isinstance(ids, list), f"{case.id}: args.ids must be list"
    if ids:
        for i, eid in enumerate(ids):
            assert isinstance(eid, str) and eid, (
                f"{case.id}: ids[{i}] must be non-empty string"
            )

    if flt is not None:
        assert isinstance(flt, dict), f"{case.id}: args.filter must be object when present"

    assert ids or flt, f"{case.id}: delete_edges requires ids or filter"

    if "namespace" in args and args["namespace"] is not None:
        assert isinstance(args["namespace"], str), (
            f"{case.id}: args.namespace must be string when present"
        )


def _assert_graph_bulk_vertices_args(args: Dict[str, Any], case: WireRequestCase) -> None:
    """
    Invariants for graph.bulk_vertices wire args.

    WireGraphHandler builds BulkVerticesSpec from:

        {
          "namespace": "...",        # optional
          "limit": 100,
          "cursor": "...",           # optional
          "filter": { ... }          # optional
        }
    """
    limit = args.get("limit")
    assert isinstance(limit, int) and limit > 0, (
        f"{case.id}: args.limit must be positive int"
    )

    if "namespace" in args and args["namespace"] is not None:
        assert isinstance(args["namespace"], str), (
            f"{case.id}: args.namespace must be string when present"
        )

    if "cursor" in args and args["cursor"] is not None:
        assert isinstance(args["cursor"], str), (
            f"{case.id}: args.cursor must be string when present"
        )

    if "filter" in args and args["filter"] is not None:
        assert isinstance(args["filter"], dict), (
            f"{case.id}: args.filter must be object when present"
        )


# ---------------------------------------------------------------------------
# Op-specific dispatch
# ---------------------------------------------------------------------------


def _assert_args_for_case(args: Dict[str, Any], case: WireRequestCase) -> None:
    """
    Operation-specific arg validation on top of schema validation.

    This is where we encode real-world expectations that are hard to
    capture in pure JSON Schema (e.g. "embed expects exactly one input").
    """
    if case.component == "embedding":
        if case.op in ("embedding.embed", "embedding.embed_batch"):
            _assert_embedding_embed_args(args, case)
        elif case.op == "embedding.count_tokens":
            _assert_embedding_count_tokens_args(args, case)

    elif case.component == "vector":
        if case.op == "vector.query":
            _assert_vector_query_args(args, case)
        elif case.op == "vector.upsert":
            _assert_vector_upsert_args(args, case)
        elif case.op == "vector.delete":
            _assert_vector_delete_args(args, case)
        elif case.op in ("vector.create_namespace", "vector.delete_namespace"):
            _assert_vector_namespace_args(args, case)

    elif case.component == "graph":
        if case.op in ("graph.query", "graph.stream_query"):
            _assert_graph_query_args(args, case)
        elif case.op == "graph.upsert_nodes":
            _assert_graph_upsert_nodes_args(args, case)
        elif case.op == "graph.upsert_edges":
            _assert_graph_upsert_edges_args(args, case)
        elif case.op == "graph.delete_nodes":
            _assert_graph_delete_nodes_args(args, case)
        elif case.op == "graph.delete_edges":
            _assert_graph_delete_edges_args(args, case)
        elif case.op == "graph.bulk_vertices":
            _assert_graph_bulk_vertices_args(args, case)

    # LLM / capabilities / health / schema / batch / transaction / traversal
    # are mostly covered by schema + envelope checks; we currently rely on
    # schema + the common envelope checks for those.


# ---------------------------------------------------------------------------
# Main parametrized test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", get_pytest_params(), ids=lambda c: c.id)
def test_wire_request_envelopes_protocol_and_schema(case: WireRequestCase, adapter: Any) -> None:
    """
    For each wire operation:

      * Call the adapter's builder (e.g. build_llm_complete_envelope)
      * Assert protocol-level envelope shape (op/ctx/args)
      * Validate against the *.envelope.request JSON Schema
      * Run operation-specific arg validation (where defined)

    This is the bridge between "what a user actually sends" and the
    schema/golden tests.
    """
    builder = _get_builder(adapter, case)
    envelope = builder()

    # 1) Envelope shape + ctx invariants
    _assert_envelope_common(envelope, case)

    # 2) Schema validation against canonical *.envelope.request
    assert_valid(case.schema_id, envelope, context=f"wire:{case.id}")

    # 3) Operation-specific args invariants
    _assert_args_for_case(envelope["args"], case)
