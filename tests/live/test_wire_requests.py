# SPDX-License-Identifier: Apache-2.0
"""
End-to-end wire-level request validation for the CORPUS Protocol.

This module exercises the *real adapter* and checks that for every
protocol operation (`op`):

  1. The adapter builds a canonical wire envelope:
     { "op", "ctx", "args" } for requests
  2. The envelope validates against the corresponding *.envelope.request
     JSON Schema via the shared SchemaRegistry
  3. Operation-specific invariants on `args` are satisfied
     (e.g. shape of texts/text, vector dimensions, etc.)
  4. The envelope survives a standard JSON serialization round-trip
     unchanged, approximating the actual HTTP wire JSON.

It complements:

  * schemas/**                 → JSON Schema contracts
  * tests/schema/test_schema_lint.py
      → meta-lint of the schemas themselves
  * tests/golden/**           → golden request/response fixtures
  * tests/golden/test_golden.py
      → golden <→ schema validation
  * tests/stream/test_stream_validator.py
      → streaming envelope validation
  * tests/conftest.py         → adapter fixture + protocol summary

This file focuses on *live wire requests*: “if a user called this op
through our adapter right now, would the raw JSON we send over the wire
actually match the protocol?”
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import pytest

from tests.utils.schema_registry import assert_valid


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WireRequestCase:
    """
    Single wire-level request scenario.

    Attributes:
        id:          Stable ID used in pytest parameterization.
        component:   "llm", "vector", "embedding", or "graph".
        op:         Canonical protocol operation name (e.g. "llm.complete").
        build_method: Name of zero-arg adapter method that returns the
                      *full* envelope {op, ctx, args}.
        schema_id:   $id of the envelope.request JSON Schema for this op.
    """
    id: str
    component: str
    op: str
    build_method: str
    schema_id: str


# ---------------------------------------------------------------------------
# Exhaustive request surface (per op, plus multi-shape ops)
# ---------------------------------------------------------------------------

WIRE_REQUEST_CASES: List[WireRequestCase] = [
    # --------------------------- LLM --------------------------- #
    WireRequestCase(
        id="llm_complete",
        component="llm",
        op="llm.complete",
        build_method="build_llm_complete_envelope",
        schema_id="https://corpusos.com/schemas/llm/llm.envelope.request.json",
    ),
    WireRequestCase(
        id="llm_count_tokens",
        component="llm",
        op="llm.count_tokens",
        build_method="build_llm_count_tokens_envelope",
        schema_id="https://corpusos.com/schemas/llm/llm.envelope.request.json",
    ),
    WireRequestCase(
        id="llm_capabilities",
        component="llm",
        op="llm.capabilities",
        build_method="build_llm_capabilities_envelope",
        schema_id="https://corpusos.com/schemas/llm/llm.envelope.request.json",
    ),
    WireRequestCase(
        id="llm_health",
        component="llm",
        op="llm.health",
        build_method="build_llm_health_envelope",
        schema_id="https://corpusos.com/schemas/llm/llm.envelope.request.json",
    ),

    # ------------------------- VECTOR -------------------------- #
    WireRequestCase(
        id="vector_query",
        component="vector",
        op="vector.query",
        build_method="build_vector_query_envelope",
        schema_id="https://corpusos.com/schemas/vector/vector.envelope.request.json",
    ),
    WireRequestCase(
        id="vector_upsert",
        component="vector",
        op="vector.upsert",
        build_method="build_vector_upsert_envelope",
        schema_id="https://corpusos.com/schemas/vector/vector.envelope.request.json",
    ),
    WireRequestCase(
        id="vector_delete",
        component="vector",
        op="vector.delete",
        build_method="build_vector_delete_envelope",
        schema_id="https://corpusos.com/schemas/vector/vector.envelope.request.json",
    ),
    WireRequestCase(
        id="vector_namespace_create",
        component="vector",
        op="vector.namespace.create",
        build_method="build_vector_namespace_create_envelope",
        schema_id="https://corpusos.com/schemas/vector/vector.envelope.request.json",
    ),
    WireRequestCase(
        id="vector_namespace_delete",
        component="vector",
        op="vector.namespace.delete",
        build_method="build_vector_namespace_delete_envelope",
        schema_id="https://corpusos.com/schemas/vector/vector.envelope.request.json",
    ),
    WireRequestCase(
        id="vector_capabilities",
        component="vector",
        op="vector.capabilities",
        build_method="build_vector_capabilities_envelope",
        schema_id="https://corpusos.com/schemas/vector/vector.envelope.request.json",
    ),
    WireRequestCase(
        id="vector_health",
        component="vector",
        op="vector.health",
        build_method="build_vector_health_envelope",
        schema_id="https://corpusos.com/schemas/vector/vector.envelope.request.json",
    ),

    # ------------------------ EMBEDDING ------------------------ #
    WireRequestCase(
        id="embedding_embed",
        component="embedding",
        op="embedding.embed",
        build_method="build_embedding_embed_envelope",
        schema_id="https://corpusos.com/schemas/embedding/embedding.envelope.request.json",
    ),
    WireRequestCase(
        id="embedding_embed_batch",
        component="embedding",
        op="embedding.embed_batch",
        build_method="build_embedding_embed_batch_envelope",
        schema_id="https://corpusos.com/schemas/embedding/embedding.envelope.request.json",
    ),

    # **Exhaustive shapes for embedding.count_tokens**
    WireRequestCase(
        id="embedding_count_tokens_single",
        component="embedding",
        op="embedding.count_tokens",
        build_method="build_embedding_count_tokens_single_envelope",
        schema_id="https://corpusos.com/schemas/embedding/embedding.envelope.request.json",
    ),
    WireRequestCase(
        id="embedding_count_tokens_batch",
        component="embedding",
        op="embedding.count_tokens",
        build_method="build_embedding_count_tokens_batch_envelope",
        schema_id="https://corpusos.com/schemas/embedding/embedding.envelope.request.json",
    ),

    WireRequestCase(
        id="embedding_capabilities",
        component="embedding",
        op="embedding.capabilities",
        build_method="build_embedding_capabilities_envelope",
        schema_id="https://corpusos.com/schemas/embedding/embedding.envelope.request.json",
    ),
    WireRequestCase(
        id="embedding_health",
        component="embedding",
        op="embedding.health",
        build_method="build_embedding_health_envelope",
        schema_id="https://corpusos.com/schemas/embedding/embedding.envelope.request.json",
    ),

    # ------------------------- GRAPH --------------------------- #
    WireRequestCase(
        id="graph_query",
        component="graph",
        op="graph.query",
        build_method="build_graph_query_envelope",
        schema_id="https://corpusos.com/schemas/graph/graph.envelope.request.json",
    ),
    WireRequestCase(
        id="graph_stream_query",
        component="graph",
        op="graph.stream_query",
        build_method="build_graph_stream_query_envelope",
        schema_id="https://corpusos.com/schemas/graph/graph.envelope.request.json",
    ),
    WireRequestCase(
        id="graph_vertex_create",
        component="graph",
        op="graph.vertex.create",
        build_method="build_graph_vertex_create_envelope",
        schema_id="https://corpusos.com/schemas/graph/graph.envelope.request.json",
    ),
    WireRequestCase(
        id="graph_vertex_delete",
        component="graph",
        op="graph.vertex.delete",
        build_method="build_graph_vertex_delete_envelope",
        schema_id="https://corpusos.com/schemas/graph/graph.envelope.request.json",
    ),
    WireRequestCase(
        id="graph_edge_create",
        component="graph",
        op="graph.edge.create",
        build_method="build_graph_edge_create_envelope",
        schema_id="https://corpusos.com/schemas/graph/graph.envelope.request.json",
    ),
    WireRequestCase(
        id="graph_batch",
        component="graph",
        op="graph.batch",
        build_method="build_graph_batch_envelope",
        schema_id="https://corpusos.com/schemas/graph/graph.envelope.request.json",
    ),
    WireRequestCase(
        id="graph_capabilities",
        component="graph",
        op="graph.capabilities",
        build_method="build_graph_capabilities_envelope",
        schema_id="https://corpusos.com/schemas/graph/graph.envelope.request.json",
    ),
    WireRequestCase(
        id="graph_health",
        component="graph",
        op="graph.health",
        build_method="build_graph_health_envelope",
        schema_id="https://corpusos.com/schemas/graph/graph.envelope.request.json",
    ),
]


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
        assert isinstance(dm, int) and dm > 0, f"{case.id}: ctx.deadline_ms must be positive int, got {dm!r}"

    if "tenant" in ctx and ctx["tenant"] is not None:
        assert isinstance(ctx["tenant"], str), f"{case.id}: ctx.tenant must be string if present"

    # args
    args = envelope["args"]
    assert isinstance(args, dict), f"{case.id}: args must be object, got {type(args)}"


def _json_roundtrip(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Approximate the actual HTTP wire JSON by forcing a standard-library
    JSON serialization + parse.

    This:
      * Ensures everything is JSON-serializable
      * Catches hidden types (e.g. datetimes, Decimals, custom classes)
      * Mirrors what a typical `json=` HTTP client path would do
    """
    payload = json.dumps(
        obj,
        ensure_ascii=False,
        separators=(",", ":"),  # compact, deterministic wrt content
    )
    return json.loads(payload)


# ---------------------------------------------------------------------------
# Embedding-specific helpers (fixed text/texts handling)
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

    We exercise *two* wire scenarios:
      - embedding_count_tokens_single → likely uses 'text'
      - embedding_count_tokens_batch  → likely uses 'texts'
    But we only enforce semantic rules, not exactly which field is used.
    """
    texts = _extract_embedding_text_inputs(args, case.id)
    assert len(texts) >= 1, f"{case.id}: count_tokens requires at least one input"

    model = args.get("model")
    assert isinstance(model, str) and model, f"{case.id}: args.model must be non-empty string"


# ---------------------------------------------------------------------------
# Vector-specific helpers (lightweight)
# ---------------------------------------------------------------------------

def _assert_vector_query_args(args: Dict[str, Any], case: WireRequestCase) -> None:
    query = args.get("query")
    assert isinstance(query, dict), f"{case.id}: args.query must be object"
    namespace = query.get("namespace")
    assert isinstance(namespace, str) and namespace, f"{case.id}: query.namespace must be non-empty string"
    vector = query.get("vector")
    assert isinstance(vector, list) and vector, f"{case.id}: query.vector must be non-empty list"
    assert all(isinstance(x, (int, float)) for x in vector), (
        f"{case.id}: query.vector must be numeric list"
    )


def _assert_vector_upsert_args(args: Dict[str, Any], case: WireRequestCase) -> None:
    vectors = args.get("vectors")
    assert isinstance(vectors, list) and vectors, f"{case.id}: args.vectors must be non-empty list"
    for i, v in enumerate(vectors):
        assert isinstance(v, dict), f"{case.id}: vectors[{i}] must be object"
        assert "id" in v, f"{case.id}: vectors[{i}] missing 'id'"
        assert "vector" in v, f"{case.id}: vectors[{i}] missing 'vector'"


# ---------------------------------------------------------------------------
# Graph-specific helpers (lightweight)
# ---------------------------------------------------------------------------

def _assert_graph_query_args(args: Dict[str, Any], case: WireRequestCase) -> None:
    query = args.get("query")
    assert isinstance(query, dict), f"{case.id}: args.query must be object"
    dialect = query.get("dialect")
    assert isinstance(dialect, str) and dialect, f"{case.id}: query.dialect must be non-empty string"
    text = query.get("text")
    assert isinstance(text, str) and text, f"{case.id}: query.text must be non-empty string"


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

    elif case.component == "graph":
        if case.op in ("graph.query", "graph.stream_query"):
            _assert_graph_query_args(args, case)

    # llm / capabilities / health requests are mostly covered by schema;
    # we currently rely on schema + envelope checks for those.


# ---------------------------------------------------------------------------
# Main parametrized test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", WIRE_REQUEST_CASES, ids=lambda c: c.id)
def test_wire_request_envelopes_protocol_and_schema(case: WireRequestCase, adapter: Any) -> None:
    """
    For each wire operation:

      * Call the adapter's builder (e.g. build_llm_complete_envelope)
      * Assert protocol-level envelope shape (op/ctx/args)
      * Serialize + deserialize via standard JSON to approximate HTTP wire JSON
      * Validate the wire-level envelope against the *.envelope.request JSON Schema
      * Run operation-specific arg validation on the wire-level args

    This is the bridge between "what a user actually sends" and the
    schema/golden tests.
    """
    builder = _get_builder(adapter, case)
    envelope = builder()

    # 1) Envelope shape + ctx invariants (pre-serialization)
    _assert_envelope_common(envelope, case)

    # 2) Force a JSON round-trip to approximate actual HTTP wire JSON
    wire_envelope = _json_roundtrip(envelope)

    # Strict: serialization must not mutate the envelope shape/content
    # (catches e.g. non-JSON types, custom encoders, etc.)
    assert wire_envelope == envelope, (
        f"{case.id}: envelope mutated by JSON serialization round-trip"
    )

    # 3) Schema validation against canonical *.envelope.request (wire-level)
    assert_valid(case.schema_id, wire_envelope, context=f"wire:{case.id}")

    # 4) Operation-specific args invariants on the wire-level payload
    _assert_args_for_case(wire_envelope["args"], case)