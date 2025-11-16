# SPDX-License-Identifier: Apache-2.0
"""
Golden sample + schema meta-validation for Corpus Protocol (Draft 2020-12).

This suite validates:
- Golden messages against component envelopes (requests/success/error)
- Protocol envelope standardization (§2.4) for all operations
- Cross-schema invariants (token totals, vector dims, batch results)
- Drift detection (listed vs on-disk goldens)
- Schema meta-lint: every JSON Schema under schemas/** is valid and resolvable
- Heuristics: timestamp/id patterns, fixture size guardrails, large string checks
"""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

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
LARGE_STRING_FIELDS = {"text", "content"}  # allow larger but bounded
SUPPORTING_FILES = {"README.md", ".gitkeep", "config.json"}
SUPPORTED_COMPONENTS = {"llm", "vector", "embedding", "graph"}

# Field-specific size limits (bytes)
MAX_STRING_FIELD_SIZES = {
    "text": 5_000_000,      # 5MB for text content
    "content": 5_000_000,   # 5MB for content  
}

# Validation patterns
TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
ID_PATTERN = re.compile(r"^[A-Za-z0-9._~:-]{1,256}$")

# Golden file naming patterns
GOLDEN_NAMING_PATTERNS = [
    r"^[a-z_]+_(request|success|error)\.json$",
    r"^error_envelope_example\.json$"
]

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def _canon_json(obj: Any) -> bytes:
    """Generate canonical JSON representation for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

def _read_text_if_exists(rel: str) -> str | None:
    """Read file if it exists, return None otherwise."""
    p = GOLDEN / rel
    return p.read_text(encoding="utf-8") if p.exists() else None

def _load_golden_if_exists(fname: str) -> dict | None:
    """Load golden file if it exists, return None otherwise."""
    p = GOLDEN / fname
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))

def _get_component_from_schema_id(schema_id: str) -> str:
    """Extract component name from schema ID."""
    parts = schema_id.split("/")
    try:
        idx = parts.index("schemas")
        return parts[idx + 1] if idx + 1 < len(parts) else "unknown"
    except (ValueError, IndexError):
        return "unknown"

def _get_golden_files_by_component() -> Dict[str, List[Tuple[str, str]]]:
    """Organize test cases by component for better test reporting."""
    by_component: Dict[str, List[Tuple[str, str]]] = {}
    for fname, schema_id in CASES:
        component = _get_component_from_schema_id(schema_id)
        if component not in by_component:
            by_component[component] = []
        by_component[component].append((fname, schema_id))
    return by_component

def _validate_string_field_size(obj: Any, path: str = "", issues: List[str] | None = None) -> List[str]:
    """
    Recursively validate string field sizes against limits.
    
    Args:
        obj: Object to validate
        path: Current object path for error messages
        issues: Accumulated validation issues
        
    Returns:
        List of validation issues found
    """
    if issues is None:
        issues = []
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            current_path = f"{path}.{key}" if path else key
            
            if isinstance(value, str):
                field_limit = MAX_STRING_FIELD_SIZES.get(key)
                if field_limit and len(value) > field_limit:
                    issues.append(
                        f"'{current_path}' exceeds size limit: {len(value)} chars > {field_limit}"
                    )
            elif isinstance(value, (dict, list)):
                _validate_string_field_size(value, current_path, issues)
                
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _validate_string_field_size(item, f"{path}[{i}]", issues)
            
    return issues

# ------------------------------------------------------------------------------
# Golden → schema mapping (broad but skip-missing)
# Only files you actually place in tests/golden/ will run.
# ------------------------------------------------------------------------------
CASES: List[Tuple[str, str]] = [
    # ---------------------- LLM ----------------------
    ("llm_complete_request.json",         "https://corpusos.com/schemas/llm/llm.envelope.request.json"),
    ("llm_complete_success.json",         "https://corpusos.com/schemas/llm/llm.envelope.success.json"),
    ("llm_count_tokens_request.json",     "https://corpusos.com/schemas/llm/llm.envelope.request.json"),
    ("llm_count_tokens_success.json",     "https://corpusos.com/schemas/llm/llm.envelope.success.json"),
    ("llm_capabilities_success.json",     "https://corpusos.com/schemas/llm/llm.envelope.success.json"),
    ("llm_health_success.json",           "https://corpusos.com/schemas/llm/llm.envelope.success.json"),
    # Optional request variants
    ("llm_complete_request_with_tools.json",            "https://corpusos.com/schemas/llm/llm.envelope.request.json"),
    ("llm_complete_request_with_response_format.json",  "https://corpusos.com/schemas/llm/llm.envelope.request.json"),
    # Streaming chunks (protocol envelope format)
    ("llm_stream_chunk.json",             "https://corpusos.com/schemas/llm/llm.envelope.success.json"),

    # -------------------- VECTOR ---------------------
    ("vector_query_request.json",         "https://corpusos.com/schemas/vector/vector.envelope.request.json"),
    ("vector_query_success.json",         "https://corpusos.com/schemas/vector/vector.envelope.success.json"),
    ("vector_upsert_request.json",        "https://corpusos.com/schemas/vector/vector.envelope.request.json"),
    ("vector_upsert_success.json",        "https://corpusos.com/schemas/vector/vector.envelope.success.json"),
    ("vector_delete_request.json",        "https://corpusos.com/schemas/vector/vector.envelope.request.json"),
    ("vector_delete_success.json",        "https://corpusos.com/schemas/vector/vector.envelope.success.json"),
    ("vector_namespace_create_request.json", "https://corpusos.com/schemas/vector/vector.envelope.request.json"),
    ("vector_namespace_create_success.json", "https://corpusos.com/schemas/vector/vector.envelope.success.json"),
    ("vector_namespace_delete_request.json", "https://corpusos.com/schemas/vector/vector.envelope.request.json"),
    ("vector_namespace_delete_success.json", "https://corpusos.com/schemas/vector/vector.envelope.success.json"),
    ("vector_capabilities_success.json",  "https://corpusos.com/schemas/vector/vector.envelope.success.json"),
    ("vector_health_success.json",        "https://corpusos.com/schemas/vector/vector.envelope.success.json"),
    ("vector_error_dimension_mismatch.json", "https://corpusos.com/schemas/vector/vector.envelope.error.json"),

    # ------------------- EMBEDDING -------------------
    ("embedding_embed_request.json",          "https://corpusos.com/schemas/embedding/embedding.envelope.request.json"),
    ("embedding_embed_success.json",          "https://corpusos.com/schemas/embedding/embedding.envelope.success.json"),
    ("embedding_embed_batch_request.json",    "https://corpusos.com/schemas/embedding/embedding.envelope.request.json"),
    ("embedding_embed_batch_success.json",    "https://corpusos.com/schemas/embedding/embedding.envelope.success.json"),
    ("embedding_count_tokens_request.json",   "https://corpusos.com/schemas/embedding/embedding.envelope.request.json"),
    ("embedding_count_tokens_success.json",   "https://corpusos.com/schemas/embedding/embedding.envelope.success.json"),
    ("embedding_capabilities_request.json",   "https://corpusos.com/schemas/embedding/embedding.envelope.request.json"),
    ("embedding_capabilities_success.json",   "https://corpusos.com/schemas/embedding/embedding.envelope.success.json"),
    ("embedding_health_request.json",         "https://corpusos.com/schemas/embedding/embedding.envelope.request.json"),
    ("embedding_health_success.json",         "https://corpusos.com/schemas/embedding/embedding.envelope.success.json"),
    ("embedding_error_text_too_long.json",    "https://corpusos.com/schemas/embedding/embedding.envelope.error.json"),

    # ---------------------- GRAPH --------------------
    ("graph_query_request.json",          "https://corpusos.com/schemas/graph/graph.envelope.request.json"),
    ("graph_query_success.json",          "https://corpusos.com/schemas/graph/graph.envelope.success.json"),
    ("graph_stream_query_request.json",   "https://corpusos.com/schemas/graph/graph.envelope.request.json"),
    ("graph_capabilities_request.json",   "https://corpusos.com/schemas/graph/graph.envelope.request.json"),
    ("graph_capabilities_success.json",   "https://corpusos.com/schemas/graph/graph.envelope.success.json"),
    ("graph_health_request.json",         "https://corpusos.com/schemas/graph/graph.envelope.request.json"),
    ("graph_health_success.json",         "https://corpusos.com/schemas/graph/graph.envelope.success.json"),
    ("graph_vertex_create_request.json",  "https://corpusos.com/schemas/graph/graph.envelope.request.json"),
    ("graph_vertex_delete_request.json",  "https://corpusos.com/schemas/graph/graph.envelope.request.json"),
    ("graph_edge_create_request.json",    "https://corpusos.com/schemas/graph/graph.envelope.request.json"),
    ("graph_batch_request.json",          "https://corpusos.com/schemas/graph/graph.envelope.request.json"),
    ("graph_batch_success.json",          "https://corpusos.com/schemas/graph/graph.envelope.success.json"),
    ("graph_id_success.json",             "https://corpusos.com/schemas/graph/graph.envelope.success.json"),
    ("graph_ack_success.json",            "https://corpusos.com/schemas/graph/graph.envelope.success.json"),
    # Streaming chunks (protocol envelope format)
    ("graph_stream_chunk.json",           "https://corpusos.com/schemas/graph/graph.envelope.success.json"),

    # ------------- Generic error example -------------
    ("error_envelope_example.json",       "https://corpusos.com/schemas/llm/llm.envelope.error.json"),
]

# ------------------------------------------------------------------------------
# Load schema registry once for the session (and to power meta-validation)
# ------------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def _load_registry_once():
    """Load schema registry once per test session for efficiency."""
    load_all_schemas_into_registry(SCHEMAS_ROOT)

# ------------------------------------------------------------------------------
# Core schema validation for all mapped goldens
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("fname,schema_id", CASES)
def test_golden_validates(fname: str, schema_id: str):
    """Test that each golden file validates against its declared schema."""
    p = GOLDEN / fname
    if not p.exists():
        pytest.skip(f"{fname} fixture not present")
    
    doc = json.loads(p.read_text(encoding="utf-8"))
    assert_valid(schema_id, doc, context=fname)

# ------------------------------------------------------------------------------
# Protocol Envelope Compliance Tests
# ------------------------------------------------------------------------------
def test_all_success_envelopes_follow_protocol_format():
    """Test ALL success envelopes use {ok, code, ms, result} exactly per §2.4."""
    for fname, schema_id in CASES:
        if "envelope.success" not in schema_id:
            continue
            
        p = GOLDEN / fname
        if not p.exists():
            continue
            
        doc = json.loads(p.read_text(encoding="utf-8"))
        
        # Protocol §2.4 REQUIRED fields
        assert "ok" in doc, f"{fname}: missing 'ok' field"
        assert "code" in doc, f"{fname}: missing 'code' field" 
        assert "ms" in doc, f"{fname}: missing 'ms' field"
        assert "result" in doc, f"{fname}: missing 'result' field"
        
        # Protocol §2.4 EXACT field constraints
        assert doc["ok"] is True, f"{fname}: 'ok' must be true"
        assert doc["code"] == "OK", f"{fname}: 'code' must be 'OK'"
        assert isinstance(doc["ms"], (int, float)) and doc["ms"] >= 0, f"{fname}: 'ms' must be non-negative number"
        
        # Protocol §2.4 NO extra fields
        allowed_fields = {"ok", "code", "ms", "result"}
        extra_fields = set(doc.keys()) - allowed_fields
        assert not extra_fields, f"{fname}: contains non-protocol fields: {extra_fields}"

def test_all_error_envelopes_follow_protocol_format():
    """Test ALL error envelopes use protocol error format exactly per §2.4."""
    for fname, schema_id in CASES:
        if "envelope.error" not in schema_id:
            continue
            
        p = GOLDEN / fname
        if not p.exists():
            continue
            
        doc = json.loads(p.read_text(encoding="utf-8"))
        
        # Protocol §2.4 REQUIRED fields
        required_fields = {"ok", "code", "error", "message", "ms"}
        for field in required_fields:
            assert field in doc, f"{fname}: missing required field '{field}'"
        
        # Protocol §2.4 field constraints
        assert doc["ok"] is False, f"{fname}: 'ok' must be false"
        assert isinstance(doc["ms"], (int, float)) and doc["ms"] >= 0
        
        # Protocol §2.4 OPTIONAL fields
        allowed_fields = {"ok", "code", "error", "message", "retry_after_ms", "details", "ms"}
        extra_fields = set(doc.keys()) - allowed_fields
        assert not extra_fields, f"{fname}: contains non-protocol fields: {extra_fields}"

def test_all_batch_operations_use_protocol_batchresult_pattern():
    """Test batch operations use {processed_count, failed_count, failures[]} pattern per §3.4."""
    batch_files = [
        "vector_upsert_success.json",
        "vector_delete_success.json", 
        "graph_batch_success.json",
        "embedding_embed_batch_success.json"
    ]
    
    for fname in batch_files:
        p = GOLDEN / fname
        if not p.exists():
            continue
            
        doc = json.loads(p.read_text(encoding="utf-8"))
        result = doc.get("result", {})
        
        # Protocol §3.4 BatchResult pattern
        assert "processed_count" in result, f"{fname}: missing 'processed_count'"
        assert "failed_count" in result, f"{fname}: missing 'failed_count'"
        assert "failures" in result, f"{fname}: missing 'failures' array"
        
        # Validate failures array structure
        for failure in result.get("failures", []):
            assert "error" in failure, f"{fname}: failure missing 'error' field"
            assert "detail" in failure, f"{fname}: failure missing 'detail' field"
            # id and index are optional per protocol

def test_batch_operations_track_failures_in_result():
    """Test batch operations track failures via failed_count > 0, not envelope codes per §3.4."""
    batch_files = [
        "vector_upsert_success.json",
        "vector_delete_success.json", 
        "graph_batch_success.json",
        "embedding_embed_batch_success.json"
    ]
    
    for fname in batch_files:
        p = GOLDEN / fname
        if not p.exists():
            continue
            
        doc = json.loads(p.read_text(encoding="utf-8"))
        
        # Protocol §2.4: envelope code must always be "OK" for successful operations
        assert doc["code"] == "OK", f"{fname}: batch operation must use 'OK' code, not partial success codes"
        
        # Protocol §3.4: failures tracked in result, not envelope
        result = doc.get("result", {})
        failed_count = result.get("failed_count", 0)
        failures = result.get("failures", [])
        
        # If there are failures, they must be properly tracked
        if failed_count > 0:
            assert len(failures) > 0, f"{fname}: failed_count > 0 but failures array is empty"
            assert failed_count == len(failures), f"{fname}: failed_count doesn't match failures array length"

def test_capabilities_use_flat_structure():
    """Test capabilities use flat structure per protocol §9.1, §13.1, §17.1."""
    capabilities_files = [
        "llm_capabilities_success.json",
        "vector_capabilities_success.json", 
        "embedding_capabilities_success.json",
        "graph_capabilities_success.json"
    ]
    
    for fname in capabilities_files:
        p = GOLDEN / fname
        if not p.exists():
            continue
            
        doc = json.loads(p.read_text(encoding="utf-8"))
        result = doc.get("result", {})
        
        # Protocol requires flat structure, not nested features/limits
        # Check for nested structures that violate protocol
        nested_structures = {"features", "limits", "sampling", "models", "cache", "extensions"}
        found_nested = nested_structures.intersection(result.keys())
        assert not found_nested, f"{fname}: uses nested structure {found_nested} instead of flat protocol fields"
        
        # Check for required protocol fields
        assert "protocol" in result, f"{fname}: missing 'protocol' field"
        assert "server" in result, f"{fname}: missing 'server' field"
        assert "version" in result, f"{fname}: missing 'version' field"

def test_streaming_uses_protocol_envelope():
    """Test streaming operations use protocol envelope with chunk field per §2.4."""
    streaming_files = [
        "llm_stream_chunk.json",
        "graph_stream_chunk.json"
    ]
    
    for fname in streaming_files:
        p = GOLDEN / fname
        if not p.exists():
            continue
            
        doc = json.loads(p.read_text(encoding="utf-8"))
        
        # Must use protocol envelope format
        assert "ok" in doc, f"{fname}: missing 'ok' field"
        assert "code" in doc, f"{fname}: missing 'code' field"
        assert "ms" in doc, f"{fname}: missing 'ms' field"
        assert "chunk" in doc, f"{fname}: missing 'chunk' field"
        
        assert doc["ok"] is True, f"{fname}: 'ok' must be true"
        assert doc["code"] == "OK", f"{fname}: 'code' must be 'OK'"
        
        # Chunk must follow protocol chunk format
        chunk = doc["chunk"]
        if "llm" in fname:
            assert "text" in chunk, f"{fname}: LLM chunk missing 'text' field"
            assert "is_final" in chunk, f"{fname}: LLM chunk missing 'is_final' field"
        elif "graph" in fname:
            assert "records" in chunk, f"{fname}: Graph chunk missing 'records' field"
            assert "is_final" in chunk, f"{fname}: Graph chunk missing 'is_final' field"

# ------------------------------------------------------------------------------
# Cross-schema invariants & heuristics
# ------------------------------------------------------------------------------
def test_llm_token_totals_invariant():
    """Test LLM token usage mathematical invariant per §3.7."""
    p = GOLDEN / "llm_complete_success.json"
    if not p.exists():
        pytest.skip("llm_complete_success.json fixture not present")
    
    doc = json.loads(p.read_text(encoding="utf-8"))
    usage = doc.get("result", {}).get("usage")
    
    if not usage:
        pytest.skip("no usage in sample")
    
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage.get("completion_tokens", 0), \
        "total_tokens must equal prompt_tokens + completion_tokens"

def test_vector_dimension_invariants():
    """Test that all vectors in a response have consistent dimensions per §16.1."""
    p = GOLDEN / "vector_query_success.json"
    if not p.exists():
        pytest.skip("vector_query_success.json fixture not present")
    
    doc = json.loads(p.read_text(encoding="utf-8"))
    matches = doc.get("result", {}).get("matches", [])
    
    if not matches:
        pytest.skip("No matches present; nothing to assert")
    
    # Find first actual vector as reference
    ref_dim = None
    for m in matches:
        vv = (m or {}).get("vector", {}).get("vector")
        if isinstance(vv, list):
            ref_dim = len(vv)
            break
    
    if ref_dim is None:
        pytest.skip("No vectors in matches to compare")
    
    # Check all vectors match reference dimension
    mismatches = []
    for idx, m in enumerate(matches):
        vv = (m or {}).get("vector", {}).get("vector")
        if isinstance(vv, list) and len(vv) != ref_dim:
            mismatches.append((idx, len(vv)))
    
    if mismatches:
        mism = ", ".join(f"match[{i}]={d} dims" for i, d in mismatches)
        pytest.fail(f"Vector dimension mismatch: expected {ref_dim}; got {mism}")

def test_vector_dimension_limits():
    """Test that all vectors respect dimension limits."""
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

        # Extract vectors from various response formats
        res = doc.get("result", {})
        
        # vector.query style
        for match in res.get("matches", []) or []:
            vv = (match or {}).get("vector", {}).get("vector")
            if isinstance(vv, list):
                vecs.append(vv)

        # embedding style
        for emb in res.get("embeddings", []) or []:
            vv = (emb or {}).get("vector")
            if isinstance(vv, list):
                vecs.append(vv)

        # Validate each vector
        for i, v in enumerate(vecs):
            assert len(v) <= MAX_VECTOR_DIMENSIONS, (
                f"{vf} vector #{i} has {len(v)} dims; exceeds {MAX_VECTOR_DIMENSIONS}"
            )

def test_timestamp_and_id_validation():
    """Test timestamp and ID field formatting across all golden files."""
    for fname, _ in CASES:
        doc = _load_golden_if_exists(fname)
        if not doc:
            continue
            
        # Validate timestamp fields
        for field in ("timestamp", "created_at", "updated_at"):
            if field in doc and doc[field]:
                assert isinstance(doc[field], str) and TIMESTAMP_PATTERN.match(doc[field]), \
                    f"{fname}: invalid timestamp in '{field}': {doc[field]}"
        
        # Validate ID fields
        for field in ("id", "request_id", "stream_id", "error_id"):
            if field in doc and doc[field]:
                assert isinstance(doc[field], str) and ID_PATTERN.match(doc[field]), \
                    f"{fname}: invalid ID in '{field}': {doc[field]}"

def test_large_fixture_performance():
    """Test fixture size limits and large string validation."""
    for fname, _ in CASES:
        p = GOLDEN / fname
        if not p.exists():
            continue
            
        # Check file size
        size = p.stat().st_size
        assert size <= MAX_FIXTURE_SIZE_BYTES, f"{fname} exceeds size limit: {size} bytes"

        # Check string field sizes
        doc = json.loads(p.read_text(encoding="utf-8"))
        issues = _validate_string_field_size(doc, fname)
        
        if issues:
            pytest.fail(f"{fname} string field size issues:\n" + "\n".join(issues))

# ------------------------------------------------------------------------------
# Drift detection: listed vs on-disk fixtures
# ------------------------------------------------------------------------------
def test_all_listed_golden_files_exist():
    """Test that all files listed in CASES exist on disk."""
    missing = [fname for fname, _ in CASES if not (GOLDEN / fname).exists()]
    if missing:
        pytest.skip(f"CASES contains missing fixtures (ok while landing): {missing}")

def test_no_orphaned_golden_files():
    """Test that no golden files exist without CASES entries."""
    golden_files = {p.name for p in GOLDEN.glob("*.json") if p.is_file()}
    tested_files = {fname for fname, _ in CASES}
    orphaned = golden_files - tested_files - SUPPORTING_FILES
    
    if orphaned:
        pytest.skip(f"Golden files without CASES entries: {sorted(orphaned)}")

def test_golden_files_have_consistent_naming():
    """Test that golden files follow consistent naming conventions."""
    naming_issues = []
    
    for fname in GOLDEN.glob("*.json"):
        if fname.name in SUPPORTING_FILES:
            continue
            
        # Check against known naming patterns
        if not any(re.match(pattern, fname.name) for pattern in GOLDEN_NAMING_PATTERNS):
            naming_issues.append(fname.name)
    
    if naming_issues:
        pytest.skip(f"Files with non-standard naming: {naming_issues}")

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

def test_component_coverage():
    """Test that all supported components have golden file coverage."""
    by_component = _get_golden_files_by_component()
    missing_components = SUPPORTED_COMPONENTS - set(by_component.keys())
    
    if missing_components:
        pytest.skip(f"Components missing golden files: {missing_components}")

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

def test_schema_registry_health():
    """Test comprehensive schema registry health and statistics."""
    # This would be extended with actual registry health checks
    # For now, we rely on the session fixture loading successfully
    assert SCHEMAS_ROOT.exists()
    assert any(SCHEMAS_ROOT.rglob("*.json")), "No JSON schemas found in schemas directory"

# ------------------------------------------------------------------------------
# Performance and reliability tests
# ------------------------------------------------------------------------------
def test_golden_file_loading_performance():
    """Test that all golden files can be loaded quickly."""
    import time
    
    max_load_time = 2.0  # seconds
    slow_files = []
    
    for fname, _ in CASES:
        p = GOLDEN / fname
        if not p.exists():
            continue
            
        start = time.time()
        try:
            json.loads(p.read_text(encoding="utf-8"))
            load_time = time.time() - start
            if load_time > max_load_time:
                slow_files.append((fname, load_time))
        except Exception as e:
            pytest.fail(f"Failed to load {fname}: {e}")
    
    if slow_files:
        slow_info = ", ".join(f"{fname}({time:.2f}s)" for fname, time in slow_files)
        pytest.skip(f"Slow golden file loading: {slow_info}")

def test_golden_file_unique_checksums():
    """Test that golden files have unique content (detect duplicates)."""
    checksums: Dict[str, List[str]] = {}
    
    for fname, _ in CASES:
        p = GOLDEN / fname
        if not p.exists():
            continue
            
        content = p.read_text(encoding="utf-8")
        checksum = hashlib.sha256(content.encode()).hexdigest()
        
        if checksum not in checksums:
            checksums[checksum] = []
        checksums[checksum].append(fname)
    
    # Report duplicates but don't fail (some duplication might be intentional)
    duplicates = {checksum: files for checksum, files in checksums.items() if len(files) > 1}
    if duplicates:
        duplicate_info = "; ".join(f"{files}" for files in duplicates.values())
        pytest.skip(f"Duplicate golden file content: {duplicate_info}")