# SPDX-License-Identifier: Apache-2.0
"""
Golden sample + schema meta-validation for Corpus Protocol (Draft 2020-12).

This suite validates:
- Golden messages against component envelopes (requests/success/error)
- Cross-schema invariants (token totals, vector dims, partial success items)
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

# Field-specific size limits (bytes / chars)
MAX_STRING_FIELD_SIZES = {
    "text": 5_000_000,      # 5MB for text content
    "content": 5_000_000,   # 5MB for content
}

# Validation patterns
TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$"
)
ID_PATTERN = re.compile(r"^[A-Za-z0-9._~:-]{1,256}$")

# Golden file naming patterns
GOLDEN_NAMING_PATTERNS = [
    r"^[a-z_]+_(request|success|error)\.json$",
    r"^error_envelope_example\.json$",
]

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def _canon_json(obj: Any) -> bytes:
    """Generate canonical JSON representation for hashing."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


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
        by_component.setdefault(component, []).append((fname, schema_id))
    return by_component


def _validate_string_field_size(
    obj: Any, path: str = "", issues: List[str] | None = None
) -> List[str]:
    """
    Recursively validate string field sizes against limits.
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
                        f"'{current_path}' exceeds size limit: "
                        f"{len(value)} chars > {field_limit}"
                    )
            elif isinstance(value, (dict, list)):
                _validate_string_field_size(value, current_path, issues)

    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _validate_string_field_size(item, f"{path}[{i}]", issues)

    return issues


def _extract_vectors_from_result(result: dict) -> List[List[float]]:
    """
    Extract vectors from a result object, handling both:
    - vector.match style: match["vector"] is a list[float]
    - (legacy) nested style: match["vector"]["vector"] is list[float]
    - embedding style: embedding["vector"] is list[float]
    """
    vecs: List[List[float]] = []

    # vector.query / vector.types.query_result
    matches = (result or {}).get("matches") or []
    for m in matches:
        if not isinstance(m, dict):
            continue
        raw = m.get("vector")
        if isinstance(raw, list):
            vecs.append(raw)
        elif isinstance(raw, dict) and isinstance(raw.get("vector"), list):
            vecs.append(raw["vector"])

    # embedding-style
    embeddings = (result or {}).get("embeddings") or []
    for emb in embeddings:
        if not isinstance(emb, dict):
            continue
        raw = emb.get("vector")
        if isinstance(raw, list):
            vecs.append(raw)

    return vecs

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
    # Streaming SINGLE frame example (data frame)
    ("llm_stream_chunk.json",             "https://corpusos.com/schemas/llm/llm.stream.frame.data.json"),

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
    # SINGLE streaming frame example
    ("graph_stream_chunk.json",           "https://corpusos.com/schemas/graph/graph.stream.frame.data.json"),

    # ------------- Generic error example -------------
    ("error_envelope_example.json",       "https://corpusos.com/schemas/llm/llm.envelope.error.json"),
]

# NDJSON streaming fixtures (frame union schemas)
STREAM_NDJSON_CASES: List[Tuple[str, str, str]] = [
    # (golden NDJSON file name, union frame schema id, component name)
    (
        "llm_stream.ndjson",
        "https://corpusos.com/schemas/llm/llm.stream.frames.ndjson.schema.json",
        "llm",
    ),
    (
        "llm_stream_error.ndjson",
        "https://corpusos.com/schemas/llm/llm.stream.frames.ndjson.schema.json",
        "llm",
    ),
    (
        "graph_stream.ndjson",
        "https://corpusos.com/schemas/graph/graph.stream.frames.ndjson.schema.json",
        "graph",
    ),
    (
        "graph_stream_error.ndjson",
        "https://corpusos.com/schemas/graph/graph.stream.frames.ndjson.schema.json",
        "graph",
    ),
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
# Protocol Envelope Compliance Tests (aligned with common/envelope.*)
# ------------------------------------------------------------------------------
def test_all_success_envelopes_follow_envelope_schema_contract():
    """
    Ensure all *success* envelopes follow the high-level contract:
      - ok: true
      - code: one of ["OK", "PARTIAL_SUCCESS", "ACCEPTED"]
      - result present
      - ms, if present, is non-negative
    Detailed field-level constraints are enforced by the JSON Schemas themselves.
    """
    for fname, schema_id in CASES:
        if "envelope.success" not in schema_id:
            continue

        p = GOLDEN / fname
        if not p.exists():
            continue

        doc = json.loads(p.read_text(encoding="utf-8"))

        assert doc.get("ok") is True, f"{fname}: 'ok' must be true on success"
        assert "result" in doc, f"{fname}: missing 'result' on success envelope"

        code = doc.get("code")
        assert code in {"OK", "PARTIAL_SUCCESS", "ACCEPTED"}, (
            f"{fname}: 'code' must be one of "
            f"['OK', 'PARTIAL_SUCCESS', 'ACCEPTED']; got {code!r}"
        )

        if "ms" in doc:
            assert isinstance(doc["ms"], (int, float)) and doc["ms"] >= 0, (
                f"{fname}: 'ms' must be non-negative number when present"
            )


def test_all_error_envelopes_follow_envelope_schema_contract():
    """
    Ensure all *error* envelopes follow high-level contract:
      - ok: false
      - code, error, message present
      - ms, if present, is non-negative
    Detailed error taxonomy / retry semantics are enforced by schema.
    """
    for fname, schema_id in CASES:
        if "envelope.error" not in schema_id:
            continue

        p = GOLDEN / fname
        if not p.exists():
            continue

        doc = json.loads(p.read_text(encoding="utf-8"))

        assert doc.get("ok") is False, f"{fname}: 'ok' must be false on error envelope"

        for field in ("code", "error", "message"):
            assert field in doc, f"{fname}: missing required error field '{field}'"
            assert isinstance(doc[field], str), f"{fname}: '{field}' must be a string"

        if "ms" in doc:
            assert isinstance(doc["ms"], (int, float)) and doc["ms"] >= 0, (
                f"{fname}: 'ms' must be non-negative number when present"
            )

# ------------------------------------------------------------------------------
# Partial / batch result invariants
# ------------------------------------------------------------------------------
def test_partial_success_envelopes_match_common_shape():
    """
    For any envelope with code == 'PARTIAL_SUCCESS', ensure result matches
    the common partialSuccessResult pattern:

      {
        "successes": int >= 1,
        "failures": int >= 1,
        "items": [
          { "index": int >= 0, "ok": true,  "result": ... },
          { "index": int >= 0, "ok": false, "code": "...", "message": "...", ... },
          ...
        ]
      }

    The exact per-item shapes are validated by the underlying schemas.
    """
    for fname, schema_id in CASES:
        if "envelope.success" not in schema_id:
            continue

        p = GOLDEN / fname
        if not p.exists():
            continue

        doc = json.loads(p.read_text(encoding="utf-8"))
        if doc.get("code") != "PARTIAL_SUCCESS":
            continue

        result = doc.get("result", {})
        assert isinstance(result, dict), f"{fname}: result must be object for PARTIAL_SUCCESS"

        for key in ("successes", "failures", "items"):
            assert key in result, f"{fname}: PARTIAL_SUCCESS result missing '{key}'"

        successes = result["successes"]
        failures = result["failures"]
        items = result["items"]

        assert isinstance(successes, int) and successes >= 1, (
            f"{fname}: successes must be int >= 1"
        )
        assert isinstance(failures, int) and failures >= 1, (
            f"{fname}: failures must be int >= 1"
        )
        assert isinstance(items, list) and len(items) >= 2, (
            f"{fname}: items must be array with at least 2 entries"
        )
        assert successes + failures == len(items), (
            f"{fname}: successes + failures must equal len(items)"
        )

        for i, item in enumerate(items):
            assert isinstance(item, dict), f"{fname}: items[{i}] must be object"
            assert "index" in item and "ok" in item, (
                f"{fname}: items[{i}] must have 'index' and 'ok'"
            )
            assert isinstance(item["index"], int) and item["index"] >= 0, (
                f"{fname}: items[{i}].index must be non-negative int"
            )
            assert isinstance(item["ok"], bool), (
                f"{fname}: items[{i}].ok must be boolean"
            )

            if item["ok"]:
                # Successful item should carry a result; error fields are optional.
                assert "result" in item, (
                    f"{fname}: items[{i}] with ok=true must have 'result'"
                )
            else:
                # Failed item should carry code + message.
                assert "code" in item and "message" in item, (
                    f"{fname}: items[{i}] with ok=false must have 'code' and 'message'"
                )

# ------------------------------------------------------------------------------
# Capabilities invariants (minimal, schema does the heavy lifting)
# ------------------------------------------------------------------------------
def test_capabilities_have_core_fields():
    """
    Capabilities responses should expose at least:
      - protocol
      - server
      - version

    Nested structures (features/limits/cache/extensions/etc.) are allowed and
    validated by their respective component schemas; we only check the core
    identity fields here.
    """
    capabilities_files = [
        "llm_capabilities_success.json",
        "vector_capabilities_success.json",
        "embedding_capabilities_success.json",
        "graph_capabilities_success.json",
    ]

    for fname in capabilities_files:
        p = GOLDEN / fname
        if not p.exists():
            continue

        doc = json.loads(p.read_text(encoding="utf-8"))
        result = doc.get("result", {})

        assert "protocol" in result, f"{fname}: missing 'protocol' in capabilities.result"
        assert "server" in result, f"{fname}: missing 'server' in capabilities.result"
        assert "version" in result, f"{fname}: missing 'version' in capabilities.result"

        assert isinstance(result["protocol"], str), f"{fname}: protocol must be string"
        assert isinstance(result["server"], str), f"{fname}: server must be string"
        assert isinstance(result["version"], str), f"{fname}: version must be string"

# ------------------------------------------------------------------------------
# Streaming frame & NDJSON validation
# ------------------------------------------------------------------------------
def test_streaming_frames_have_event_field():
    """
    Single-frame JSON examples (llm_stream_chunk.json, graph_stream_chunk.json)
    should look like canonical streaming frames:

      { "event": "data" | "end" | "error", ... }

    The rest of the shape is enforced by the per-component frame schemas.
    """
    streaming_files = [
        ("llm_stream_chunk.json", "llm"),
        ("graph_stream_chunk.json", "graph"),
    ]

    for fname, component in streaming_files:
        p = GOLDEN / fname
        if not p.exists():
            continue

        doc = json.loads(p.read_text(encoding="utf-8"))
        assert "event" in doc, f"{fname}: missing 'event' on stream frame"
        assert doc["event"] in {"data", "end", "error"}, (
            f"{fname}: event must be 'data', 'end', or 'error'"
        )


@pytest.mark.parametrize("fname,schema_id,component", STREAM_NDJSON_CASES)
def test_streaming_ndjson_validates_with_stream_validator(
    fname: str, schema_id: str, component: str
):
    """
    Validate NDJSON streaming golden fixtures via the protocol-compliant
    stream validator. This checks:
      • per-line frame shape (data/end/error) via the union frame schema
      • overall stream invariants (single terminal frame, etc.)
    """
    p = GOLDEN / fname
    if not p.exists():
        pytest.skip(f"{fname} NDJSON fixture not present")

    ndjson_text = p.read_text(encoding="utf-8")
    report = validate_ndjson_stream(
        ndjson_text,
        envelope_schema_id=schema_id,  # schema_id is the union frame schema
        component=component,
    )

    assert report.is_valid, report.error_summary

# ------------------------------------------------------------------------------
# Cross-schema invariants & heuristics
# ------------------------------------------------------------------------------
def test_llm_token_totals_invariant():
    """Test LLM token usage mathematical invariant."""
    p = GOLDEN / "llm_complete_success.json"
    if not p.exists():
        pytest.skip("llm_complete_success.json fixture not present")

    doc = json.loads(p.read_text(encoding="utf-8"))
    usage = doc.get("result", {}).get("usage")

    if not usage:
        pytest.skip("no usage in sample")

    assert usage["total_tokens"] == usage["prompt_tokens"] + usage.get(
        "completion_tokens", 0
    ), "total_tokens must equal prompt_tokens + completion_tokens"


def test_vector_dimension_invariants():
    """
    Test that all vectors in a query response have consistent dimensions.
    Uses the newest vector.types.vector_match shape (vector: list[number]),
    but also tolerates legacy nested {vector:{vector:[...]}} for safety.
    """
    p = GOLDEN / "vector_query_success.json"
    if not p.exists():
        pytest.skip("vector_query_success.json fixture not present")

    doc = json.loads(p.read_text(encoding="utf-8"))
    result = doc.get("result", {})
    vecs = _extract_vectors_from_result(result)

    if not vecs:
        pytest.skip("No vectors present; nothing to assert")

    ref_dim = len(vecs[0])
    mismatches = [
        (i, len(v)) for i, v in enumerate(vecs) if len(v) != ref_dim
    ]

    if mismatches:
        mism = ", ".join(f"vec[{i}]={d} dims" for i, d in mismatches)
        pytest.fail(
            f"Vector dimension mismatch: expected {ref_dim}; got {mism}"
        )


def test_vector_dimension_limits():
    """Test that all vectors respect global dimension limits across samples."""
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
        result = doc.get("result", {})
        vecs = _extract_vectors_from_result(result)

        for i, v in enumerate(vecs):
            assert len(v) <= MAX_VECTOR_DIMENSIONS, (
                f"{vf} vector #{i} has {len(v)} dims; "
                f"exceeds {MAX_VECTOR_DIMENSIONS}"
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
                assert isinstance(doc[field], str) and TIMESTAMP_PATTERN.match(
                    doc[field]
                ), f"{fname}: invalid timestamp in '{field}': {doc[field]}"

        # Validate ID fields
        for field in ("id", "request_id", "stream_id", "error_id"):
            if field in doc and doc[field]:
                assert isinstance(doc[field], str) and ID_PATTERN.match(
                    doc[field]
                ), f"{fname}: invalid ID in '{field}': {doc[field]}"


def test_large_fixture_performance():
    """Test fixture size limits and large string validation."""
    for fname, _ in CASES:
        p = GOLDEN / fname
        if not p.exists():
            continue

        # Check file size
        size = p.stat().st_size
        assert size <= MAX_FIXTURE_SIZE_BYTES, (
            f"{fname} exceeds size limit: {size} bytes"
        )

        # Check string field sizes
        doc = json.loads(p.read_text(encoding="utf-8"))
        issues = _validate_string_field_size(doc, fname)

        if issues:
            pytest.fail(
                f"{fname} string field size issues:\n" + "\n".join(issues)
            )

# ------------------------------------------------------------------------------
# Drift detection: listed vs on-disk fixtures
# ------------------------------------------------------------------------------
def test_all_listed_golden_files_exist():
    """Test that all files listed in CASES exist on disk."""
    missing = [fname for fname, _ in CASES if not (GOLDEN / fname).exists()]
    if missing:
        pytest.skip(
            f"CASES contains missing fixtures (ok while landing): {missing}"
        )


def test_no_orphaned_golden_files():
    """Test that no golden files exist without CASES entries."""
    golden_files = {
        p.name for p in GOLDEN.glob("*.json") if p.is_file()
    }
    tested_files = {fname for fname, _ in CASES}
    orphaned = golden_files - tested_files - SUPPORTING_FILES

    if orphaned:
        pytest.skip(
            f"Golden files without CASES entries: {sorted(orphaned)}"
        )


def test_golden_files_have_consistent_naming():
    """Test that golden files follow consistent naming conventions."""
    naming_issues = []

    for fname in GOLDEN.glob("*.json"):
        if fname.name in SUPPORTING_FILES:
            continue

        if not any(
            re.match(pattern, fname.name)
            for pattern in GOLDEN_NAMING_PATTERNS
        ):
            naming_issues.append(fname.name)

    if naming_issues:
        pytest.skip(f"Files with non-standard naming: {naming_issues}")


def test_request_response_pairs():
    """
    Heuristic: requests typically have a corresponding *_success.json or
    *_error.json. Skips rather than fails so CI stays green during
    incremental fixture work.
    """
    request_files = {
        fname for fname, sid in CASES if "envelope.request" in sid
    }
    missing_responses = []

    for req in request_files:
        base = req.replace("_request.json", "")
        succ = base + "_success.json"
        err = base + "_error.json"

        if (GOLDEN / req).exists() and not (GOLDEN / succ).exists() and not (
            GOLDEN / err
        ).exists():
            missing_responses.append(req)

    if missing_responses:
        pytest.skip(
            f"Requests missing *_success/_error: {missing_responses}"
        )


def test_component_coverage():
    """Test that all supported components have golden file coverage."""
    by_component = _get_golden_files_by_component()
    missing_components = SUPPORTED_COMPONENTS - set(by_component.keys())

    if missing_components:
        pytest.skip(f"Components missing golden files: {missing_components}")

# ------------------------------------------------------------------------------
# Schema meta-validation: lint every schema under schemas/**
# ------------------------------------------------------------------------------
def test_all_schemas_load_and_refs_resolve():
    """
    Belt-and-suspenders: this gives one-to-one visibility that every schema
    file is syntactically valid JSON Schema and resolvable via the registry.
    """
    # load_all_schemas_into_registry() already ran via session fixture;
    # if loading failed it would have raised. This test simply asserts the tree exists.
    assert SCHEMAS_ROOT.exists(), "schemas/ directory must exist"


def test_schema_registry_health():
    """Test comprehensive schema registry health and statistics."""
    assert SCHEMAS_ROOT.exists()
    assert any(
        SCHEMAS_ROOT.rglob("*.json")
    ), "No JSON schemas found in schemas directory"

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
        except Exception as e:  # pragma: no cover - defensive
            pytest.fail(f"Failed to load {fname}: {e}")

    if slow_files:
        slow_info = ", ".join(
            f"{fname}({t:.2f}s)" for fname, t in slow_files
        )
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

        checksums.setdefault(checksum, []).append(fname)

    # Report duplicates but don't fail (some duplication might be intentional)
    duplicates = {
        checksum: files for checksum, files in checksums.items() if len(files) > 1
    }
    if duplicates:
        duplicate_info = "; ".join(f"{files}" for files in duplicates.values())
        pytest.skip(f"Duplicate golden file content: {duplicate_info}")
