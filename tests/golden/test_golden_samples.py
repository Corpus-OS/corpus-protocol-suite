# SPDX-License-Identifier: Apache-2.0
"""
Golden sample validation against Corpus Protocol schemas (Draft 2020-12).

- Validates each SIEM-safe golden fixture under tests/golden/ against the correct
  component envelope/schema $id (not the generic common ones).
- Verifies invariants (schema_version present, result_hash correctness,
  partial-success math, vector dimension consistency).
- Validates NDJSON stream rules (exactly one terminal frame; no data after terminal)
  for LLM and Graph streams.

Helpers expected:
  tests/utils/schema_registry.py :: assert_valid(schema_id, obj, context="")
  tests/utils/stream_validator.py :: validate_ndjson_stream(ndjson, union_schema_id=None, component=None)
"""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
import pytest

from tests.utils.schema_registry import assert_valid
from tests.utils.stream_validator import validate_ndjson_stream

# point directly at tests/golden/
GOLDEN = Path(__file__).resolve().parent


def _canon_json(obj) -> bytes:
    """Canonical JSON for hashing: sorted keys, tight separators, UTF-8."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


# ------------------------------------------------------------------------------
# Golden → schema mapping
# Keep in sync with files actually present in tests/golden/.
# Each case SKIPS if the golden file is not present locally.
# ------------------------------------------------------------------------------

CASES = [
    # ====================== LLM ======================
    ("llm_complete_request.json",
     "https://adaptersdk.org/schemas/llm/llm.envelope.request.json"),
    ("llm_complete_success.json",
     "https://adaptersdk.org/schemas/llm/llm.envelope.success.json"),
    ("llm_count_tokens_request.json",
     "https://adaptersdk.org/schemas/llm/llm.envelope.request.json"),
    ("llm_count_tokens_success.json",
     "https://adaptersdk.org/schemas/llm/llm.envelope.success.json"),
    ("llm_capabilities_success.json",
     "https://adaptersdk.org/schemas/llm/llm.envelope.success.json"),
    ("llm_health_success.json",
     "https://adaptersdk.org/schemas/llm/llm.envelope.success.json"),
    # Per-frame samples (optional)
    ("llm_stream_frame_data.json",
     "https://adaptersdk.org/schemas/llm/llm.stream.frame.data.json"),
    ("llm_stream_frame_end.json",
     "https://adaptersdk.org/schemas/llm/llm.stream.frame.end.json"),
    ("llm_stream_frame_error.json",
     "https://adaptersdk.org/schemas/llm/llm.stream.frame.error.json"),

    # Optional LLM request variants
    ("llm_complete_request_with_tools.json",
     "https://adaptersdk.org/schemas/llm/llm.envelope.request.json"),
    ("llm_complete_request_with_response_format.json",
     "https://adaptersdk.org/schemas/llm/llm.envelope.request.json"),

    # ===================== VECTOR ====================
    ("vector_query_request.json",
     "https://adaptersdk.org/schemas/vector/vector.envelope.request.json"),
    ("vector_query_success.json",
     "https://adaptersdk.org/schemas/vector/vector.envelope.success.json"),
    ("vector_upsert_request.json",
     "https://adaptersdk.org/schemas/vector/vector.envelope.request.json"),
    ("vector_upsert_success.json",
     "https://adaptersdk.org/schemas/vector/vector.envelope.success.json"),
    ("vector_delete_request.json",
     "https://adaptersdk.org/schemas/vector/vector.envelope.request.json"),
    ("vector_delete_success.json",
     "https://adaptersdk.org/schemas/vector/vector.envelope.success.json"),
    ("vector_namespace_create_request.json",
     "https://adaptersdk.org/schemas/vector/vector.envelope.request.json"),
    ("vector_namespace_create_success.json",
     "https://adaptersdk.org/schemas/vector/vector.envelope.success.json"),
    ("vector_namespace_delete_request.json",
     "https://adaptersdk.org/schemas/vector/vector.envelope.request.json"),
    ("vector_namespace_delete_success.json",
     "https://adaptersdk.org/schemas/vector/vector.envelope.success.json"),
    ("vector_capabilities_success.json",
     "https://adaptersdk.org/schemas/vector/vector.envelope.success.json"),
    ("vector_health_success.json",
     "https://adaptersdk.org/schemas/vector/vector.envelope.success.json"),
    ("vector_error_dimension_mismatch.json",
     "https://adaptersdk.org/schemas/vector/vector.envelope.error.json"),
    # Optional partial success fixture (if you keep one for vector)
    ("vector_partial_success_result.json",
     "https://adaptersdk.org/schemas/vector/vector.envelope.success.json"),

    # ==================== EMBEDDING ==================
    ("embedding_embed_request.json",
     "https://adaptersdk.org/schemas/embedding/embedding.envelope.request.json"),
    ("embedding_embed_success.json",
     "https://adaptersdk.org/schemas/embedding/embedding.envelope.success.json"),
    ("embedding_embed_batch_request.json",
     "https://adaptersdk.org/schemas/embedding/embedding.envelope.request.json"),
    ("embedding_embed_batch_success.json",
     "https://adaptersdk.org/schemas/embedding/embedding.envelope.success.json"),
    ("embedding_count_tokens_request.json",
     "https://adaptersdk.org/schemas/embedding/embedding.envelope.request.json"),
    ("embedding_count_tokens_success.json",
     "https://adaptersdk.org/schemas/embedding/embedding.envelope.success.json"),
    ("embedding_capabilities_request.json",
     "https://adaptersdk.org/schemas/embedding/embedding.envelope.request.json"),
    ("embedding_capabilities_success.json",
     "https://adaptersdk.org/schemas/embedding/embedding.envelope.success.json"),
    ("embedding_health_request.json",
     "https://adaptersdk.org/schemas/embedding/embedding.envelope.request.json"),
    ("embedding_health_success.json",
     "https://adaptersdk.org/schemas/embedding/embedding.envelope.success.json"),
    ("embedding_partial_success_result.json",
     "https://adaptersdk.org/schemas/embedding/embedding.envelope.success.json"),
    # Optional explicit error sample
    ("embedding_error_text_too_long.json",
     "https://adaptersdk.org/schemas/embedding/embedding.envelope.error.json"),

    # ====================== GRAPH ====================
    ("graph_query_request.json",
     "https://adaptersdk.org/schemas/graph/graph.envelope.request.json"),
    ("graph_query_success.json",
     "https://adaptersdk.org/schemas/graph/graph.envelope.success.json"),
    ("graph_stream_query_request.json",
     "https://adaptersdk.org/schemas/graph/graph.envelope.request.json"),
    ("graph_capabilities_success.json",
     "https://adaptersdk.org/schemas/graph/graph.envelope.success.json"),
    ("graph_health_success.json",
     "https://adaptersdk.org/schemas/graph/graph.envelope.success.json"),
    ("graph_vertex_create_request.json",
     "https://adaptersdk.org/schemas/graph/graph.envelope.request.json"),
    ("graph_vertex_delete_request.json",
     "https://adaptersdk.org/schemas/graph/graph.envelope.request.json"),
    ("graph_edge_create_request.json",
     "https://adaptersdk.org/schemas/graph/graph.envelope.request.json"),
    ("graph_batch_request.json",
     "https://adaptersdk.org/schemas/graph/graph.envelope.request.json"),
    ("graph_batch_partial_success.json",
     "https://adaptersdk.org/schemas/graph/graph.envelope.success.json"),
    ("graph_id_success.json",
     "https://adaptersdk.org/schemas/graph/graph.envelope.success.json"),
    ("graph_ack_success.json",
     "https://adaptersdk.org/schemas/graph/graph.envelope.success.json"),
    # Per-frame samples
    ("graph_stream_frame_data.json",
     "https://adaptersdk.org/schemas/graph/graph.stream.frame.data.json"),
    ("graph_stream_frame_end.json",
     "https://adaptersdk.org/schemas/graph/graph.stream.frame.end.json"),
    ("graph_stream_frame_error.json",
     "https://adaptersdk.org/schemas/graph/graph.stream.frame.error.json"),

    # ===================== LLM/Generic Error =====================
    ("error_envelope_example.json",
     "https://adaptersdk.org/schemas/llm/llm.envelope.error.json"),
]


@pytest.mark.parametrize("fname,schema_id", CASES)
def test_golden_validates(fname: str, schema_id: str):
    """
    Validate each golden document against its authoritative component schema.
    Skips gracefully if a given golden fixture isn’t present locally.
    """
    path = GOLDEN / fname
    if not path.exists():
        pytest.skip(f"{fname} fixture not present")
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert_valid(schema_id, doc, context=fname)


# ------------------------------------------------------------------------------
# NDJSON streaming validations (exactly one terminal; no data after terminal)
# For LLM we validate with the union schema.
# For Graph, if you do not have a union schema, the validator can enforce
# per-line frame validation based on component="graph".
# ------------------------------------------------------------------------------

def _read_text_if_exists(rel: str) -> str | None:
    p = GOLDEN / rel
    return p.read_text(encoding="utf-8") if p.exists() else None


def test_llm_stream_ndjson_union_validates():
    ndj = _read_text_if_exists("llm_stream.ndjson")
    if ndj is None:
        pytest.skip("llm_stream.ndjson fixture not present")
    validate_ndjson_stream(
        ndj,
        union_schema_id="https://adaptersdk.org/schemas/llm/llm.stream.frames.ndjson.schema.json",
        component="llm",
    )


def test_graph_stream_ndjson_validates_frames_and_terminal_rules():
    ndj = _read_text_if_exists("graph_stream.ndjson")
    if ndj is None:
        pytest.skip("graph_stream.ndjson fixture not present")
    # If you later add a graph union schema, pass it here; otherwise let the
    # validator enforce per-line frame schema and terminal rules with component="graph".
    validate_ndjson_stream(
        ndj,
        union_schema_id=None,
        component="graph",
    )


# ------------------------------------------------------------------------------
# Invariants
# ------------------------------------------------------------------------------

def test_llm_success_result_hash_matches():
    """
    If result_hash is present, ensure it equals SHA-256 of canonical result.
    """
    p = GOLDEN / "llm_complete_success.json"
    if not p.exists():
        pytest.skip("llm_complete_success.json fixture not present")
    doc = json.loads(p.read_text(encoding="utf-8"))
    result = doc.get("result")
    rh = doc.get("result_hash")
    if rh and result is not None:
        digest = hashlib.sha256(_canon_json(result)).hexdigest()
        assert rh.lower() == digest, "result_hash must be SHA-256 of canonical JSON of result"


def test_embedding_partial_success_invariants():
    """
    For PARTIAL_SUCCESS, enforce:
      - successes + failures == len(items)
      - successes >= 1 and failures >= 1
    """
    p = GOLDEN / "embedding_partial_success_result.json"
    if not p.exists():
        pytest.skip("embedding_partial_success_result.json fixture not present")
    doc = json.loads(p.read_text(encoding="utf-8"))
    assert doc["code"] == "PARTIAL_SUCCESS"
    res = doc["result"]
    assert res["successes"] + res["failures"] == len(res["items"])
    assert res["successes"] >= 1 and res["failures"] >= 1


def test_graph_batch_partial_success_invariants():
    """
    For graph.batch PARTIAL_SUCCESS, enforce:
      - successes + failures == len(items)
      - successes >= 1 and failures >= 1
    """
    p = GOLDEN / "graph_batch_partial_success.json"
    if not p.exists():
        pytest.skip("graph_batch_partial_success.json fixture not present")
    doc = json.loads(p.read_text(encoding="utf-8"))
    assert doc["code"] == "PARTIAL_SUCCESS"
    res = doc["result"]
    assert res["successes"] + res["failures"] == len(res["items"])
    assert res["successes"] >= 1 and res["failures"] >= 1


def test_vector_dimension_invariants():
    """
    Vector dimensions should be consistent within response matches.
    Only checks if matches exist and include raw vectors.
    """
    p = GOLDEN / "vector_query_success.json"
    if not p.exists():
        pytest.skip("vector_query_success.json fixture not present")
    doc = json.loads(p.read_text(encoding="utf-8"))
    result = doc.get("result", {})
    matches = result.get("matches", [])
    if not matches:
        return
    first_vec = matches[0].get("vector", {}).get("vector")
    if not isinstance(first_vec, list):
        return
    dim = len(first_vec)
    for idx, m in enumerate(matches[1:], start=1):
        mv = m.get("vector", {}).get("vector")
        if isinstance(mv, list):
            assert len(mv) == dim, f"match[{idx}] vector dimension differs (expected {dim})"


def test_schema_version_present_on_success_envelopes():
    """
    All success envelopes should include schema_version in golden samples.
    Applied to any case that maps to an envelope.success schema id.
    """
    missing = []
    for fname, schema_id in CASES:
        if "envelope.success.json" in schema_id:
            path = GOLDEN / fname
            if not path.exists():
                continue
            doc = json.loads(path.read_text(encoding="utf-8"))
            if "schema_version" not in doc or not re.match(r"^\d+\.\d+\.\d+$", str(doc["schema_version"])):
                missing.append(fname)
    if missing:
        pytest.fail("Missing or invalid schema_version in success envelopes: " + ", ".join(missing))
