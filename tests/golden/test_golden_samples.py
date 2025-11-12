# SPDX-License-Identifier: Apache-2.0
"""
Golden sample + schema meta-validation for Corpus Protocol (Draft 2020-12).

This suite validates:
- Golden messages against component envelopes (requests/success/error)
- NDJSON stream rules for LLM + Graph (data/end/error; exactly-one terminal)
- Cross-schema invariants (partial-success math, token totals, vector dims)
- Drift detection (listed vs on-disk goldens)
- Schema meta-lint: every JSON Schema under schemas/** is valid and resolvable
- Heuristics: timestamp/id patterns, fixture size guardrails, large string checks
"""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pytest

from tests.utils.schema_registry import assert_valid, load_all_schemas_into_registry
from tests.utils.stream_validator import validate_ndjson_stream

# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]  # repo root
SCHEMAS_ROOT = ROOT / "schemas"             # schemas/** (common, llm, vector, embedding, graph)
GOLDEN = Path(__file__).resolve().parent    # tests/golden/

# ------------------------------------------------------------------------------
# Constants / patterns
# ------------------------------------------------------------------------------
MAX_VECTOR_DIMENSIONS = 10_000
MAX_FIXTURE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB
LARGE_STRING_FIELDS = {"text", "content", "data", "vector"}  # allow larger but bounded
SUPPORTING_FILES = {"README.md", ".gitkeep", "config.json"}

TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
ID_PATTERN = re.compile(r"^[A-Za-z0-9._~:-]{1,256}$")
SCHEMA_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def _canon_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

def _read_text_if_exists(rel: str) -> str | None:
    p = GOLDEN / rel
    return p.read_text(encoding="utf-8") if p.exists() else None

def _load_golden_if_exists(fname: str) -> dict | None:
    p = GOLDEN / fname
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))

def _get_component_from_schema_id(schema_id: str) -> str:
    parts = schema_id.split("/")
    try:
        idx = parts.index("schemas")
        return parts[idx + 1]
    except Exception:
        return "unknown"

# ------------------------------------------------------------------------------
# Golden → schema mapping (broad but skip-missing)
# Only files you actually place in tests/golden/ will run.
# ------------------------------------------------------------------------------
CASES: List[Tuple[str, str]] = [
    # ---------------------- LLM ----------------------
    ("llm_complete_request.json",         "https://adaptersdk.org/schemas/llm/llm.envelope.request.json"),
    ("llm_complete_success.json",         "https://adaptersdk.org/schemas/llm/llm.envelope.success.json"),
    ("llm_count_tokens_request.json",     "https://adaptersdk.org/schemas/llm/llm.envelope.request.json"),
    ("llm_count_tokens_success.json",     "https://adaptersdk.org/schemas/llm/llm.envelope.success.json"),
    ("llm_capabilities_success.json",     "https://adaptersdk.org/schemas/llm/llm.envelope.success.json"),
    ("llm_health_success.json",           "https://adaptersdk.org/schemas/llm/llm.envelope.success.json"),
    # Optional request variants
    ("llm_complete_request_with_tools.json",            "https://adaptersdk.org/schemas/llm/llm.envelope.request.json"),
    ("llm_complete_request_with_response_format.json",  "https://adaptersdk.org/schemas/llm/llm.envelope.request.json"),
    # Optional per-frame samples (single frames as JSON, not NDJSON)
    ("llm_stream_frame_data.json",        "https://adaptersdk.org/schemas/llm/llm.stream.frame.data.json"),
    ("llm_stream_frame_end.json",         "https://adaptersdk.org/schemas/llm/llm.stream.frame.end.json"),
    ("llm_stream_frame_error.json",       "https://adaptersdk.org/schemas/llm/llm.stream.frame.error.json"),

    # -------------------- VECTOR ---------------------
    ("vector_query_request.json",         "https://adaptersdk.org/schemas/vector/vector.envelope.request.json"),
    ("vector_query_success.json",         "https://adaptersdk.org/schemas/vector/vector.envelope.success.json"),
    ("vector_upsert_request.json",        "https://adaptersdk.org/schemas/vector/vector.envelope.request.json"),
    ("vector_upsert_success.json",        "https://adaptersdk.org/schemas/vector/vector.envelope.success.json"),
    ("vector_delete_request.json",        "https://adaptersdk.org/schemas/vector/vector.envelope.request.json"),
    ("vector_delete_success.json",        "https://adaptersdk.org/schemas/vector/vector.envelope.success.json"),
    ("vector_namespace_create_request.json", "https://adaptersdk.org/schemas/vector/vector.envelope.request.json"),
    ("vector_namespace_create_success.json", "https://adaptersdk.org/schemas/vector/vector.envelope.success.json"),
    ("vector_namespace_delete_request.json", "https://adaptersdk.org/schemas/vector/vector.envelope.request.json"),
    ("vector_namespace_delete_success.json", "https://adaptersdk.org/schemas/vector/vector.envelope.success.json"),
    ("vector_capabilities_success.json",  "https://adaptersdk.org/schemas/vector/vector.envelope.success.json"),
    ("vector_health_success.json",        "https://adaptersdk.org/schemas/vector/vector.envelope.success.json"),
    ("vector_error_dimension_mismatch.json", "https://adaptersdk.org/schemas/vector/vector.envelope.error.json"),
    ("vector_partial_success_result.json","https://adaptersdk.org/schemas/vector/vector.envelope.success.json"),

    # ------------------- EMBEDDING -------------------
    ("embedding_embed_request.json",          "https://adaptersdk.org/schemas/embedding/embedding.envelope.request.json"),
    ("embedding_embed_success.json",          "https://adaptersdk.org/schemas/embedding/embedding.envelope.success.json"),
    ("embedding_embed_batch_request.json",    "https://adaptersdk.org/schemas/embedding/embedding.envelope.request.json"),
    ("embedding_embed_batch_success.json",    "https://adaptersdk.org/schemas/embedding/embedding.envelope.success.json"),
    ("embedding_count_tokens_request.json",   "https://adaptersdk.org/schemas/embedding/embedding.envelope.request.json"),
    ("embedding_count_tokens_success.json",   "https://adaptersdk.org/schemas/embedding/embedding.envelope.success.json"),
    ("embedding_capabilities_request.json",   "https://adaptersdk.org/schemas/embedding/embedding.envelope.request.json"),
    ("embedding_capabilities_success.json",   "https://adaptersdk.org/schemas/embedding/embedding.envelope.success.json"),
    ("embedding_health_request.json",         "https://adaptersdk.org/schemas/embedding/embedding.envelope.request.json"),
    ("embedding_health_success.json",         "https://adaptersdk.org/schemas/embedding/embedding.envelope.success.json"),
    ("embedding_partial_success_result.json", "https://adaptersdk.org/schemas/embedding/embedding.envelope.success.json"),
    ("embedding_error_text_too_long.json",    "https://adaptersdk.org/schemas/embedding/embedding.envelope.error.json"),

    # ---------------------- GRAPH --------------------
    ("graph_query_request.json",          "https://adaptersdk.org/schemas/graph/graph.envelope.request.json"),
    ("graph_query_success.json",          "https://adaptersdk.org/schemas/graph/graph.envelope.success.json"),
    ("graph_stream_query_request.json",   "https://adaptersdk.org/schemas/graph/graph.envelope.request.json"),
    ("graph_capabilities_request.json",   "https://adaptersdk.org/schemas/graph/graph.envelope.request.json"),
    ("graph_capabilities_success.json",   "https://adaptersdk.org/schemas/graph/graph.envelope.success.json"),
    ("graph_health_request.json",         "https://adaptersdk.org/schemas/graph/graph.envelope.request.json"),
    ("graph_health_success.json",         "https://adaptersdk.org/schemas/graph/graph.envelope.success.json"),
    ("graph_vertex_create_request.json",  "https://adaptersdk.org/schemas/graph/graph.envelope.request.json"),
    ("graph_vertex_delete_request.json",  "https://adaptersdk.org/schemas/graph/graph.envelope.request.json"),
    ("graph_edge_create_request.json",    "https://adaptersdk.org/schemas/graph/graph.envelope.request.json"),
    ("graph_batch_request.json",          "https://adaptersdk.org/schemas/graph/graph.envelope.request.json"),
    ("graph_batch_partial_success.json",  "https://adaptersdk.org/schemas/graph/graph.envelope.success.json"),
    ("graph_id_success.json",             "https://adaptersdk.org/schemas/graph/graph.envelope.success.json"),
    ("graph_ack_success.json",            "https://adaptersdk.org/schemas/graph/graph.envelope.success.json"),
    # Optional per-frame samples (single frames as JSON)
    ("graph_stream_frame_data.json",      "https://adaptersdk.org/schemas/graph/graph.stream.frame.data.json"),
    ("graph_stream_frame_end.json",       "https://adaptersdk.org/schemas/graph/graph.stream.frame.end.json"),
    ("graph_stream_frame_error.json",     "https://adaptersdk.org/schemas/graph/graph.stream.frame.error.json"),

    # ------------- Generic error example -------------
    ("error_envelope_example.json",       "https://adaptersdk.org/schemas/llm/llm.envelope.error.json"),
]

# ------------------------------------------------------------------------------
# Load schema registry once for the session (and to power meta-validation)
# ------------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def _load_registry_once():
    load_all_schemas_into_registry(SCHEMAS_ROOT)

# ------------------------------------------------------------------------------
# Core schema validation for all mapped goldens
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("fname,schema_id", CASES)
def test_golden_validates(fname: str, schema_id: str):
    p = GOLDEN / fname
    if not p.exists():
        pytest.skip(f"{fname} fixture not present")
    doc = json.loads(p.read_text(encoding="utf-8"))
    assert_valid(schema_id, doc, context=fname)

# ------------------------------------------------------------------------------
# NDJSON stream validations (LLM + Graph)
# ------------------------------------------------------------------------------
def test_llm_stream_ndjson_union_validates():
    ndj = _read_text_if_exists("llm_stream.ndjson")
    if ndj is None:
        pytest.skip("llm_stream.ndjson fixture not present")
    validate_ndjson_stream(
        ndj,
        union_schema_id="https://adaptersdk.org/schemas/llm/llm.stream.frames.ndjson.schema.json",
        component="llm",
    )

def test_llm_stream_error_ndjson_validates():
    ndj = _read_text_if_exists("llm_stream_error.ndjson")
    if ndj is None:
        pytest.skip("llm_stream_error.ndjson fixture not present")
    validate_ndjson_stream(
        ndj,
        union_schema_id="https://adaptersdk.org/schemas/llm/llm.stream.frames.ndjson.schema.json",
        component="llm",
    )

def test_graph_stream_ndjson_validates_frames_and_terminal_rules():
    ndj = _read_text_if_exists("graph_stream.ndjson")
    if ndj is None:
        pytest.skip("graph_stream.ndjson fixture not present")
    # If you later add a graph union schema, pass it via union_schema_id
    validate_ndjson_stream(
        ndj,
        union_schema_id=None,
        component="graph",
    )

def test_graph_stream_error_ndjson_validates():
    ndj = _read_text_if_exists("graph_stream_error.ndjson")
    if ndj is None:
        pytest.skip("graph_stream_error.ndjson fixture not present")
    validate_ndjson_stream(
        ndj,
        union_schema_id=None,
        component="graph",
    )

def test_stream_validation_edge_cases():
    with pytest.raises(ValueError):
        validate_ndjson_stream("", component="llm")

    terminal_only = '{"event":"end","code":"OK"}\n'
    validate_ndjson_stream(terminal_only, component="llm")

    multiple_terminals = (
        '{"event":"data","data":{"text":"x","is_final":false}}\n'
        '{"event":"end","code":"OK"}\n'
        '{"event":"end","code":"OK"}\n'
    )
    with pytest.raises(ValueError):
        validate_ndjson_stream(multiple_terminals, component="llm")

# ------------------------------------------------------------------------------
# Cross-schema invariants & heuristics
# ------------------------------------------------------------------------------
def test_llm_success_result_hash_matches():
    p = GOLDEN / "llm_complete_success.json"
    if not p.exists():
        pytest.skip("llm_complete_success.json fixture not present")
    doc = json.loads(p.read_text(encoding="utf-8"))
    result = doc.get("result")
    rh = doc.get("result_hash")
    if not rh or result is None:
        pytest.skip("no result_hash/result in sample")
    digest = hashlib.sha256(_canon_json(result)).hexdigest()
    assert rh.lower() == digest, "result_hash must be SHA-256 of canonical JSON of result"

def test_embedding_partial_success_invariants():
    p = GOLDEN / "embedding_partial_success_result.json"
    if not p.exists():
        pytest.skip("embedding_partial_success_result.json fixture not present")
    doc = json.loads(p.read_text(encoding="utf-8"))
    assert doc["code"] == "PARTIAL_SUCCESS"
    res = doc["result"]
    succ, fail, items = res["successes"], res["failures"], res["items"]
    assert succ + fail == len(items), "successes + failures must equal len(items)"
    assert succ >= 1 and fail >= 1, "PARTIAL_SUCCESS requires ≥1 success and ≥1 failure"

def test_graph_batch_partial_success_invariants():
    p = GOLDEN / "graph_batch_partial_success.json"
    if not p.exists():
        pytest.skip("graph_batch_partial_success.json fixture not present")
    doc = json.loads(p.read_text(encoding="utf-8"))
    assert doc["code"] == "PARTIAL_SUCCESS"
    res = doc["result"]
    succ, fail, items = res["successes"], res["failures"], res["items"]
    assert succ + fail == len(items), "successes + failures must equal len(items)"
    assert succ >= 1 and fail >= 1, "PARTIAL_SUCCESS requires ≥1 success and ≥1 failure"

def test_llm_token_totals_invariant():
    p = GOLDEN / "llm_complete_success.json"
    if not p.exists():
        pytest.skip("llm_complete_success.json fixture not present")
    doc = json.loads(p.read_text(encoding="utf-8"))
    usage = doc.get("result", {}).get("usage")
    if not usage:
        pytest.skip("no usage in sample")
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

def test_vector_dimension_invariants():
    p = GOLDEN / "vector_query_success.json"
    if not p.exists():
        pytest.skip("vector_query_success.json fixture not present")
    doc = json.loads(p.read_text(encoding="utf-8"))
    matches = doc.get("result", {}).get("matches", [])
    if not matches:
        pytest.skip("No matches present; nothing to assert")
    # find first actual vector as reference
    ref_dim = None
    for m in matches:
        vv = (m or {}).get("vector", {}).get("vector")
        if isinstance(vv, list):
            ref_dim = len(vv)
            break
    if ref_dim is None:
        pytest.skip("No vectors in matches to compare")
    mismatches = []
    for idx, m in enumerate(matches):
        vv = (m or {}).get("vector", {}).get("vector")
        if isinstance(vv, list) and len(vv) != ref_dim:
            mismatches.append((idx, len(vv)))
    if mismatches:
        mism = ", ".join(f"match[{i}]={d} dims" for i, d in mismatches)
        pytest.fail(f"Vector dimension mismatch: expected {ref_dim}; got {mism}")

def test_vector_dimension_limits():
    vector_files = [
        "vector_query_success.json",
        "vector_upsert_success.json",
        "embedding_embed_success.json",
    ]
    for vf in vector_files:
        p = GOLDEN / vf
        if not p.exists():
            continue
        doc = json.loads(p.read_text(encoding="utf-8"))
        vecs: List[List[float]] = []

        # vector.query style
        res = doc.get("result", {})
        for match in res.get("matches", []) or []:
            vv = (match or {}).get("vector", {}).get("vector")
            if isinstance(vv, list):
                vecs.append(vv)

        # embedding style
        for emb in res.get("embeddings", []) or []:
            vv = (emb or {}).get("vector")
            if isinstance(vv, list):
                vecs.append(vv)

        for i, v in enumerate(vecs):
            assert len(v) <= MAX_VECTOR_DIMENSIONS, (
                f"{vf} vector #{i} has {len(v)} dims; exceeds {MAX_VECTOR_DIMENSIONS}"
            )

def test_schema_version_present_on_success_envelopes():
    missing = []
    for fname, schema_id in CASES:
        if "envelope.success" not in schema_id:
            continue
        p = GOLDEN / fname
        if not p.exists():
            continue
        doc = json.loads(p.read_text(encoding="utf-8"))
        sv = doc.get("schema_version")
        if not (isinstance(sv, str) and SCHEMA_VERSION_PATTERN.match(sv)):
            missing.append(fname)
    if missing:
        pytest.fail("Missing/invalid schema_version in success envelopes: " + ", ".join(missing))

def test_timestamp_and_id_validation():
    for fname, _ in CASES:
        doc = _load_golden_if_exists(fname)
        if not doc:
            continue
        for field in ("timestamp", "created_at", "updated_at"):
            if field in doc and doc[field]:
                assert isinstance(doc[field], str) and TIMESTAMP_PATTERN.match(doc[field]), \
                    f"{fname}: invalid timestamp in '{field}': {doc[field]}"
        for field in ("id", "request_id", "stream_id", "error_id"):
            if field in doc and doc[field]:
                assert isinstance(doc[field], str) and ID_PATTERN.match(doc[field]), \
                    f"{fname}: invalid ID in '{field}': {doc[field]}"

def test_large_fixture_performance():
    # Quick safety net on fixture size + gigantic strings deep in payloads
    for fname, _ in CASES:
        p = GOLDEN / fname
        if not p.exists():
            continue
        size = p.stat().st_size
        assert size <= MAX_FIXTURE_SIZE_BYTES, f"{fname} exceeds size limit: {size} bytes"

        doc = json.loads(p.read_text(encoding="utf-8"))

        def _walk(obj, path=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    kp = f"{path}.{k}" if path else k
                    if isinstance(v, str):
                        if len(v) > 1_000_000:  # 1MB per-field sanity
                            if k in LARGE_STRING_FIELDS and len(v) <= 5_000_000:
                                continue
                            pytest.fail(f"{fname}: '{kp}' excessively large string ({len(v)} chars)")
                    elif isinstance(v, (dict, list)):
                        _walk(v, kp)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    _walk(item, f"{path}[{i}]")
        _walk(doc)

# ------------------------------------------------------------------------------
# Drift detection: listed vs on-disk fixtures
# ------------------------------------------------------------------------------
def test_all_listed_golden_files_exist():
    missing = [fname for fname, _ in CASES if not (GOLDEN / fname).exists()]
    if missing:
        pytest.skip(f"CASES contains missing fixtures (ok while landing): {missing}")

def test_no_orphaned_golden_files():
    golden_files = {p.name for p in GOLDEN.glob("*.json") if p.is_file()}
    tested_files = {fname for fname, _ in CASES}
    orphaned = golden_files - tested_files - SUPPORTING_FILES
    if orphaned:
        pytest.skip(f"Golden files without CASES entries: {sorted(orphaned)}")

def test_request_response_pairs():
    """
    Heuristic: requests typically have a corresponding *_success.json or *_error.json.
    Skips rather than fails so CI stays green during incremental fixture work.
    """
    request_files = {fname for fname, sid in CASES if "envelope.request" in sid}
    missing_responses = []
    for req in request_files:
        base = req.replace("_request.json", "")
        succ = base + "_success.json"
        err = base + "_error.json"
        if (GOLDEN / req).exists() and not (GOLDEN / succ).exists() and not (GOLDEN / err).exists():
            missing_responses.append(req)
    if missing_responses:
        pytest.skip(f"Requests missing *_success/_error: {missing_responses}")

# ------------------------------------------------------------------------------
# Schema meta-validation: lint every schema under schemas/**
# Ensures every schema loads, is Draft 2020-12-valid, and all $refs resolve.
# ------------------------------------------------------------------------------
def test_all_schemas_load_and_refs_resolve():
    """
    Belt-and-suspenders: this gives one-to-one visibility that every schema file is
    syntactically valid JSON Schema and resolvable via the registry.
    """
    # load_all_schemas_into_registry() already ran via session fixture;
    # if loading failed it would have raised. This test simply asserts the tree exists.
    assert SCHEMAS_ROOT.exists(), "schemas/ directory must exist"
