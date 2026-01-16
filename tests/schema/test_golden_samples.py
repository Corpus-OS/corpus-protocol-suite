# SPDX-License-Identifier: Apache-2.0
"""
Golden sample + schema meta-validation for Corpus Protocol (Draft 2020-12).

Validates:
- Golden fixtures against inferred operation-level schemas (request/success/error) and selected type schemas
- Envelope invariants for success/error/streaming envelopes (aligned to SCHEMA.md)
- NDJSON stream validation (stream success frames + error termination)
- Cross-schema invariants (token totals, vector dims)
- BatchResult invariants (only when result is batch-shaped)
- Capabilities core fields (protocol/server/version)
- Performance/guardrails: parse-time, fixture size, large string checks
- Duplicate fixture content advisory
- Schema registry health (schemas load, metaschema-valid, refs resolve)

Auto-discovery:
- Scans tests/golden/** for *.json and *.ndjson
- Infers schema IDs from filenames (preferred)
- Uses small OVERRIDES maps for legacy/special names

Closed-loop checks:
- Verifies every inferred schema_id exists in the loaded schema registry BEFORE validation.
- Produces clear failure messages when a golden filename implies a missing schema.
- Performs the same closed-loop check for NDJSON fixtures.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import pytest

from tests.utils.schema_registry import (
    assert_valid,
    load_all_schemas_into_registry,
    list_schemas,
)
from tests.utils.stream_validator import validate_ndjson_stream

# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]  # repo root
SCHEMAS_ROOT = (ROOT / "schemas") if (ROOT / "schemas").exists() else (ROOT / "schema")
GOLDEN = ROOT / "tests" / "golden"

SCHEMA_BASE = "https://corpusos.com/schemas"

# ------------------------------------------------------------------------------
# Constants / patterns
# ------------------------------------------------------------------------------
MAX_VECTOR_DIMENSIONS = 10_000
MAX_FIXTURE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB
SUPPORTING_FILES = {"README.md", ".gitkeep", "config.json"}

# Optional milliseconds (exactly 3 digits if present)
RFC3339_ZULU_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{3})?Z$"
)
ID_PATTERN = re.compile(r"^[A-Za-z0-9._~:-]{1,256}$")

MAX_STRING_FIELD_SIZES = {"text": 5_000_000, "content": 5_000_000}

SUPPORTED_COMPONENTS = {"llm", "vector", "embedding", "graph"}

# Guardrail: if legacy "stream_frame_*" goldens appear, fail.
# (If you truly keep frame schemas, add explicit mapping instead of these leftovers.)
STREAM_FRAME_LEFTOVER_RE = re.compile(
    r"^(llm|graph|embedding)_stream_frame_(data|end|error)\.json$"
)

# ------------------------------------------------------------------------------
# Overrides (keep tiny; prefer renaming files to match convention)
# ------------------------------------------------------------------------------
# JSON: relpath -> schema_id
OVERRIDES_JSON: Dict[str, str] = {
    # Legacy error envelope names
    "llm/llm_error_envelope.json": f"{SCHEMA_BASE}/llm/llm.envelope.error.json",
    "vector/vector_error_dimension_mismatch.json": f"{SCHEMA_BASE}/vector/vector.envelope.error.json",
    # Stream chunk examples (single-frame JSON)
    "llm/llm_stream_chunk.json": f"{SCHEMA_BASE}/llm/llm.stream.success.json",
    "graph/graph_stream_chunk.json": f"{SCHEMA_BASE}/graph/graph.stream_query.success.json",
    "embedding/embedding_stream_chunk.json": f"{SCHEMA_BASE}/embedding/embedding.stream_embed.success.json",
    # Dotted schema filenames as goldens (type-ish)
    "llm/llm_sampling_params.json": f"{SCHEMA_BASE}/llm/llm.sampling.params.json",
    "llm/llm_tools_schema.json": f"{SCHEMA_BASE}/llm/llm.tools.schema.json",
}

# NDJSON: relpath -> (schema_id, component)
OVERRIDES_NDJSON: Dict[str, Tuple[str, str]] = {
    "llm/llm_stream.ndjson": (f"{SCHEMA_BASE}/llm/llm.stream.success.json", "llm"),
    "llm/llm_stream_error.ndjson": (f"{SCHEMA_BASE}/llm/llm.stream.success.json", "llm"),
    "graph/graph_stream.ndjson": (f"{SCHEMA_BASE}/graph/graph.stream_query.success.json", "graph"),
    "graph/graph_stream_error.ndjson": (f"{SCHEMA_BASE}/graph/graph.stream_query.success.json", "graph"),
    "embedding/embedding_stream.ndjson": (f"{SCHEMA_BASE}/embedding/embedding.stream_embed.success.json", "embedding"),
}

# Aliases to match SCHEMA.md op naming if goldens use alternate tokens
OP_ALIASES: Dict[str, Dict[str, str]] = {
    "vector": {"namespace_create": "create_namespace", "namespace_delete": "delete_namespace"},
    "graph": {"edge_create": "upsert_edges"},
}

# ------------------------------------------------------------------------------
# Discovery + inference
# ------------------------------------------------------------------------------
def _infer_graph_dot_variant_schema_id(component: str, filename: str) -> str:
    """
    Handles:
      graph.delete_nodes.by_id.request.json
      graph.upsert_nodes.single.success.json

    Canonicalizes to:
      graph.delete_nodes.request.json
      graph.upsert_nodes.success.json
    """
    stem = filename[:-5]  # strip .json
    parts = stem.split(".")
    if len(parts) < 3 or parts[0] != "graph":
        raise ValueError(f"Not a graph dot-variant: {filename}")

    kind = parts[-1]
    if kind not in {"request", "success"}:
        raise ValueError(f"Graph dot-variant must end with request/success: {filename}")

    variants = {"by_id", "single"}
    if parts[-2] in variants:
        op_parts = parts[1:-2]
    else:
        op_parts = parts[1:-1]

    op = ".".join(op_parts)
    if not op:
        raise ValueError(f"Graph dot-variant missing op: {filename}")

    return f"{SCHEMA_BASE}/{component}/{component}.{op}.{kind}.json"


def _infer_schema_id_from_json_relpath(relpath: str) -> str:
    """
    Supported golden JSON naming patterns:

    A) Operation fixtures:
       <component>_<op>_<request|success|error>.json
       -> <component>.<op>.<kind>.json (with optional aliases)

    B) Type fixtures:
       <component>_types_<name>.json -> <component>.types.<name>.json

    C) Embedding count_tokens split by single/batch OR no suffix:
       embedding_count_tokens_request.json
       embedding_count_tokens_request_single.json
       embedding_count_tokens_request_batch.json
         -> embedding.count_tokens.request.json

       embedding_count_tokens_success.json
       embedding_count_tokens_success_single.json
       embedding_count_tokens_success_batch.json
         -> embedding.count_tokens.success.json

    D) Graph dot variants:
       graph.delete_nodes.by_id.request.json -> graph.delete_nodes.request.json
       graph.upsert_nodes.single.success.json -> graph.upsert_nodes.success.json
    """
    if relpath in OVERRIDES_JSON:
        return OVERRIDES_JSON[relpath]

    p = Path(relpath)
    if len(p.parts) < 2:
        raise ValueError(f"Golden JSON must be under tests/golden/<component>/: {relpath}")

    component = p.parts[0]
    if component not in SUPPORTED_COMPONENTS:
        raise ValueError(f"Unknown component folder '{component}' in {relpath}")

    filename = p.name
    stem = p.stem

    if STREAM_FRAME_LEFTOVER_RE.match(filename):
        raise ValueError(
            f"Legacy stream-frame golden exists but SCHEMA.md does not define it: {relpath}"
        )

    # Graph dot-variant request/success
    if (
        component == "graph"
        and (filename.endswith(".request.json") or filename.endswith(".success.json"))
        and "." in filename
    ):
        return _infer_graph_dot_variant_schema_id(component, filename)

    # Embedding count_tokens request/success (suffix optional)
    if component == "embedding":
        if stem in {
            "embedding_count_tokens_request",
            "embedding_count_tokens_request_single",
            "embedding_count_tokens_request_batch",
        }:
            return f"{SCHEMA_BASE}/embedding/embedding.count_tokens.request.json"

        if stem in {
            "embedding_count_tokens_success",
            "embedding_count_tokens_success_single",
            "embedding_count_tokens_success_batch",
        }:
            return f"{SCHEMA_BASE}/embedding/embedding.count_tokens.success.json"

    # Type fixtures: <component>_types_<name>.json
    if stem.startswith(f"{component}_types_"):
        tname = stem[len(f"{component}_types_") :]
        if not tname:
            raise ValueError(f"Missing type name in {relpath}")
        return f"{SCHEMA_BASE}/{component}/{component}.types.{tname}.json"

    # Standard operation fixtures: <component>_<op>_<kind>.json
    parts = stem.split("_")
    if len(parts) >= 3 and parts[0] == component and parts[-1] in {"request", "success", "error"}:
        kind = parts[-1]
        op = "_".join(parts[1:-1]).strip()
        if not op:
            raise ValueError(f"Missing op in golden filename: {relpath}")
        op = OP_ALIASES.get(component, {}).get(op, op)
        return f"{SCHEMA_BASE}/{component}/{component}.{op}.{kind}.json"

    raise ValueError(f"Cannot infer schema id for golden: {relpath}")


def _infer_ndjson_case(relpath: str) -> Tuple[str, str]:
    """
    NDJSON fixtures are short-named in your repo; rely on OVERRIDES_NDJSON.
    """
    if relpath in OVERRIDES_NDJSON:
        return OVERRIDES_NDJSON[relpath]
    raise ValueError(f"NDJSON schema inference not configured for {relpath}. Add override.")


def _discover_cases_and_errors() -> Tuple[List[Tuple[str, str]], Optional[str]]:
    cases: List[Tuple[str, str]] = []
    try:
        if not GOLDEN.exists():
            return [], f"Golden directory not found: {GOLDEN}"

        for fp in sorted(GOLDEN.rglob("*.json")):
            if fp.name in SUPPORTING_FILES:
                continue
            rel = fp.relative_to(GOLDEN).as_posix()
            schema_id = _infer_schema_id_from_json_relpath(rel)
            cases.append((rel, schema_id))

        return cases, None
    except Exception as e:
        return [], f"Golden JSON discovery failed: {e}"


def _discover_ndjson_cases_and_errors() -> Tuple[List[Tuple[str, str, str]], Optional[str]]:
    cases: List[Tuple[str, str, str]] = []
    try:
        if not GOLDEN.exists():
            return [], f"Golden directory not found: {GOLDEN}"

        for fp in sorted(GOLDEN.rglob("*.ndjson")):
            if fp.name in SUPPORTING_FILES:
                continue
            rel = fp.relative_to(GOLDEN).as_posix()
            schema_id, component = _infer_ndjson_case(rel)
            cases.append((rel, schema_id, component))

        return cases, None
    except Exception as e:
        return [], f"Golden NDJSON discovery failed: {e}"


CASES, CASES_DISCOVERY_ERROR = _discover_cases_and_errors()
STREAM_NDJSON_CASES, NDJSON_DISCOVERY_ERROR = _discover_ndjson_cases_and_errors()

# ------------------------------------------------------------------------------
# Session setup
# ------------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def _load_registry_once():
    load_all_schemas_into_registry(SCHEMAS_ROOT)

# ------------------------------------------------------------------------------
# Discovery guardrails
# ------------------------------------------------------------------------------
def test_golden_discovery_succeeds():
    if CASES_DISCOVERY_ERROR:
        pytest.fail(CASES_DISCOVERY_ERROR)
    if NDJSON_DISCOVERY_ERROR:
        pytest.fail(NDJSON_DISCOVERY_ERROR)
    assert GOLDEN.exists(), f"{GOLDEN} must exist"


def test_inferred_schema_ids_follow_convention():
    for fname, schema_id in CASES:
        assert schema_id.startswith(f"{SCHEMA_BASE}/"), (
            f"{fname}: schema_id not under {SCHEMA_BASE}: {schema_id}"
        )

# ------------------------------------------------------------------------------
# Closed-loop registry checks
# ------------------------------------------------------------------------------
def _closest_schema_id_matches(schema_id: str, known_ids: List[str], limit: int = 8) -> List[str]:
    hits: List[str] = []
    needle = schema_id.lower()
    for sid in known_ids:
        s = sid.lower()
        if s.endswith(needle) or needle.endswith(s) or needle in s:
            hits.append(sid)

    if not hits:
        fname = schema_id.rsplit("/", 1)[-1]
        for sid in known_ids:
            if sid.rsplit("/", 1)[-1] == fname:
                hits.append(sid)

    return sorted(hits)[:limit]


def test_inferred_schema_ids_exist_in_registry():
    registry = list_schemas()  # {schema_id: file_path}
    known_ids = sorted(registry.keys())

    missing: List[str] = []
    for golden_rel, schema_id in CASES:
        if schema_id not in registry:
            suggestions = _closest_schema_id_matches(schema_id, known_ids, limit=8)
            suggestion_block = ""
            if suggestions:
                suggestion_block = "\n    Closest matches:\n      - " + "\n      - ".join(suggestions)
            missing.append(
                f"  - Golden: {golden_rel}\n"
                f"    Inferred schema_id: {schema_id}\n"
                f"    Registry has {len(known_ids)} schemas loaded."
                f"{suggestion_block}"
            )

    if missing:
        pytest.fail(
            "Some golden files imply schema IDs that do not exist in the schema registry.\n"
            "This usually means: (a) the golden filename is wrong, (b) the schema file/$id is missing, "
            "or (c) SCHEMA.md and /schema/** drifted.\n\n"
            + "\n\n".join(missing)
        )


def test_inferred_ndjson_schema_ids_exist_in_registry():
    registry = list_schemas()
    known_ids = sorted(registry.keys())

    missing: List[str] = []
    for golden_rel, schema_id, component in STREAM_NDJSON_CASES:
        if schema_id not in registry:
            suggestions = _closest_schema_id_matches(schema_id, known_ids, limit=8)
            suggestion_block = ""
            if suggestions:
                suggestion_block = "\n    Closest matches:\n      - " + "\n      - ".join(suggestions)
            missing.append(
                f"  - NDJSON: {golden_rel}\n"
                f"    Component: {component}\n"
                f"    Schema id: {schema_id}\n"
                f"    Registry has {len(known_ids)} schemas loaded."
                f"{suggestion_block}"
            )

    if missing:
        pytest.fail(
            "Some NDJSON fixtures reference schema IDs that do not exist in the schema registry.\n\n"
            + "\n\n".join(missing)
        )

# ------------------------------------------------------------------------------
# Core: golden validates against inferred schema
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("fname,schema_id", CASES)
def test_golden_validates(fname: str, schema_id: str):
    p = GOLDEN / fname
    if not p.exists():
        pytest.skip(f"{fname} fixture not present")

    doc = json.loads(p.read_text(encoding="utf-8"))
    assert_valid(schema_id, doc, context=fname)

# ------------------------------------------------------------------------------
# Envelope compliance heuristics (aligned to SCHEMA.md)
# ------------------------------------------------------------------------------
def test_success_envelopes_follow_common_success_contract():
    """
    For any golden validated against a *non-streaming* success schema:
      - ok == True
      - code == "OK"
      - ms is non-negative number
      - result exists
    Only enforced for objects that look like envelopes (have ok+code).
    """
    for fname, schema_id in CASES:
        if not schema_id.endswith(".success.json"):
            continue
        # Exclude streaming success schemas (handled in test_stream_envelope_contract)
        if ".stream" in schema_id:
            continue

        p = GOLDEN / fname
        if not p.exists():
            continue

        doc = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(doc, dict) or "ok" not in doc or "code" not in doc:
            continue

        assert doc.get("ok") is True, f"{fname}: ok must be true"
        assert doc.get("code") == "OK", f"{fname}: code must be 'OK' (got {doc.get('code')!r})"
        assert "ms" in doc and isinstance(doc["ms"], (int, float)) and doc["ms"] >= 0, f"{fname}: ms must be >= 0"
        assert "result" in doc, f"{fname}: missing result"


def test_error_envelopes_follow_common_error_contract():
    """
    For any golden validated against an error envelope schema:
      - ok == False
      - required fields exist
    """
    for fname, schema_id in CASES:
        if not schema_id.endswith("envelope.error.json"):
            continue

        p = GOLDEN / fname
        if not p.exists():
            continue

        doc = json.loads(p.read_text(encoding="utf-8"))
        assert doc.get("ok") is False, f"{fname}: ok must be false"
        for field in ("code", "error", "message", "retry_after_ms", "details", "ms"):
            assert field in doc, f"{fname}: missing required field {field}"
        assert isinstance(doc["message"], str) and doc["message"], f"{fname}: message must be non-empty"
        assert isinstance(doc["ms"], (int, float)) and doc["ms"] >= 0, f"{fname}: ms must be >= 0"


def test_stream_envelope_contract():
    """
    Streaming single-frame JSON goldens must use protocol STREAMING envelope with chunk.
    Detected by schema_id (future-proof; no hardcoded filenames).
    """
    for fname, schema_id in CASES:
        if ".stream" not in schema_id or not schema_id.endswith(".success.json"):
            continue

        p = GOLDEN / fname
        if not p.exists():
            continue

        doc = json.loads(p.read_text(encoding="utf-8"))
        assert isinstance(doc, dict), f"{fname}: streaming golden must be a JSON object"
        assert doc.get("ok") is True, f"{fname}: ok must be true"
        assert doc.get("code") == "STREAMING", f"{fname}: code must be 'STREAMING' (got {doc.get('code')!r})"
        assert "ms" in doc and isinstance(doc["ms"], (int, float)) and doc["ms"] >= 0, f"{fname}: ms must be >= 0"
        assert "chunk" in doc, f"{fname}: missing chunk"

# ------------------------------------------------------------------------------
# Request envelope context sanity (aligned to SCHEMA.md)
# ------------------------------------------------------------------------------
def test_request_envelopes_have_valid_context_and_args():
    """
    For request envelopes:
      - ctx exists and is an object
      - args exists and is an object
      - if ctx.deadline_ms present => int >= 0
      - if ctx.attrs present => object
    """
    for fname, schema_id in CASES:
        if not schema_id.endswith(".request.json"):
            continue

        p = GOLDEN / fname
        if not p.exists():
            continue

        doc = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            continue

        assert "ctx" in doc, f"{fname}: missing ctx"
        assert isinstance(doc["ctx"], dict), f"{fname}: ctx must be object"

        assert "args" in doc, f"{fname}: missing args"
        assert isinstance(doc["args"], dict), f"{fname}: args must be object"

        ctx = doc["ctx"]
        if "deadline_ms" in ctx and ctx["deadline_ms"] is not None:
            assert isinstance(ctx["deadline_ms"], int) and ctx["deadline_ms"] >= 0, (
                f"{fname}: deadline_ms must be int >= 0"
            )
        if "attrs" in ctx and ctx["attrs"] is not None:
            assert isinstance(ctx["attrs"], dict), f"{fname}: ctx.attrs must be object"

# ------------------------------------------------------------------------------
# Embedding unary stream flag (aligned to SCHEMA.md)
# ------------------------------------------------------------------------------
def test_embedding_embed_request_stream_flag_is_absent_or_false():
    """
    embedding.embed is unary; streaming uses embedding.stream_embed.
    Ensure args.stream is absent or False for embedding.embed.request goldens.
    """
    target_schema = f"{SCHEMA_BASE}/embedding/embedding.embed.request.json"
    for fname, schema_id in CASES:
        if schema_id != target_schema:
            continue

        p = GOLDEN / fname
        if not p.exists():
            continue

        doc = json.loads(p.read_text(encoding="utf-8"))
        args = doc.get("args", {})
        if isinstance(args, dict) and "stream" in args:
            assert args["stream"] is False, (
                f"{fname}: embedding.embed must have stream=false if present"
            )

# ------------------------------------------------------------------------------
# NDJSON stream validation
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("fname,schema_id,component", STREAM_NDJSON_CASES)
def test_streaming_ndjson_validates_with_stream_validator(fname: str, schema_id: str, component: str):
    p = GOLDEN / fname
    if not p.exists():
        pytest.skip(f"{fname} NDJSON fixture not present")

    ndjson_text = p.read_text(encoding="utf-8")
    report = validate_ndjson_stream(
        ndjson_text,
        envelope_schema_id=schema_id,
        component=component,
    )
    assert report.is_valid, report.error_summary

# ------------------------------------------------------------------------------
# Capabilities semantic check
# ------------------------------------------------------------------------------
def test_capabilities_have_core_fields():
    """Capabilities success results should include protocol/server/version."""
    for fname, schema_id in CASES:
        if not schema_id.endswith(".capabilities.success.json"):
            continue

        p = GOLDEN / fname
        if not p.exists():
            continue

        doc = json.loads(p.read_text(encoding="utf-8"))
        result = doc.get("result", {})
        assert isinstance(result, dict), f"{fname}: result must be object"
        assert "protocol" in result, f"{fname}: missing result.protocol"
        assert "server" in result, f"{fname}: missing result.server"
        assert "version" in result, f"{fname}: missing result.version"

# ------------------------------------------------------------------------------
# Batch invariants (conditional; only if result looks batch-shaped)
# ------------------------------------------------------------------------------
def test_all_batch_operations_use_protocol_batchresult_pattern():
    """
    If a success result contains any BatchResult keys, require the full pattern:
      processed_count, failed_count, failures[]
    """
    for fname, schema_id in CASES:
        if not schema_id.endswith(".success.json"):
            continue

        p = GOLDEN / fname
        if not p.exists():
            continue

        doc = json.loads(p.read_text(encoding="utf-8"))
        result = doc.get("result", {})
        if not isinstance(result, dict):
            continue

        batch_keys = {"processed_count", "failed_count", "failures"}
        if not (batch_keys & set(result.keys())):
            continue  # not a batch-shaped result

        assert "processed_count" in result, f"{fname}: missing result.processed_count"
        assert "failed_count" in result, f"{fname}: missing result.failed_count"
        assert "failures" in result, f"{fname}: missing result.failures"
        assert isinstance(result["failures"], list), f"{fname}: result.failures must be a list"


def test_batch_operations_track_failures_in_result():
    """If failed_count is present, it must match len(failures) (when failures present)."""
    for fname, schema_id in CASES:
        if not schema_id.endswith(".success.json"):
            continue

        p = GOLDEN / fname
        if not p.exists():
            continue

        doc = json.loads(p.read_text(encoding="utf-8"))
        result = doc.get("result", {})
        if not isinstance(result, dict):
            continue

        if "failed_count" not in result:
            continue

        failed_count = result.get("failed_count")
        failures = result.get("failures")

        if isinstance(failed_count, int) and isinstance(failures, list):
            assert failed_count == len(failures), (
                f"{fname}: failed_count ({failed_count}) != len(failures) ({len(failures)})"
            )

# ------------------------------------------------------------------------------
# Cross-schema invariants (lightweight)
# ------------------------------------------------------------------------------
def test_llm_token_totals_invariant():
    p = GOLDEN / "llm/llm_complete_success.json"
    if not p.exists():
        pytest.skip("llm/llm_complete_success.json fixture not present")

    doc = json.loads(p.read_text(encoding="utf-8"))
    usage = doc.get("result", {}).get("usage")
    if not usage:
        pytest.skip("no usage in sample")

    assert usage["total_tokens"] == usage["prompt_tokens"] + usage.get("completion_tokens", 0), (
        "total_tokens must equal prompt_tokens + completion_tokens"
    )


def _extract_vectors_from_result(result: dict) -> List[List[float]]:
    vectors: List[List[float]] = []

    matches = result.get("matches") or []
    for m in matches:
        if not isinstance(m, dict):
            continue
        v = m.get("vector")
        if isinstance(v, list):
            vectors.append(v)
        elif isinstance(v, dict) and isinstance(v.get("vector"), list):
            vectors.append(v["vector"])

    emb = result.get("embedding")
    if isinstance(emb, dict) and isinstance(emb.get("vector"), list):
        vectors.append(emb["vector"])

    embeddings = result.get("embeddings") or []
    for e in embeddings:
        if isinstance(e, dict) and isinstance(e.get("vector"), list):
            vectors.append(e["vector"])

    return vectors


def test_vector_dimension_invariants_and_limits():
    vector_files = [
        "vector/vector_query_success.json",
        "embedding/embedding_embed_success.json",
        "embedding/embedding_embed_batch_success.json",
    ]
    for vf in vector_files:
        p = GOLDEN / vf
        if not p.exists():
            continue

        doc = json.loads(p.read_text(encoding="utf-8"))
        vecs = _extract_vectors_from_result(doc.get("result", {}))
        if not vecs:
            continue

        ref_dim = len(vecs[0])
        for i, v in enumerate(vecs):
            assert len(v) == ref_dim, f"{vf}: vector[{i}] dim mismatch: {len(v)} != {ref_dim}"
            assert len(v) <= MAX_VECTOR_DIMENSIONS, (
                f"{vf}: vector[{i}] too large: {len(v)} > {MAX_VECTOR_DIMENSIONS}"
            )

# ------------------------------------------------------------------------------
# Heuristics: timestamps, ids, size guardrails
# ------------------------------------------------------------------------------
def test_timestamp_and_id_validation():
    for fname, _sid in CASES:
        p = GOLDEN / fname
        if not p.exists():
            continue

        doc = json.loads(p.read_text(encoding="utf-8"))

        if "timestamp" in doc and doc["timestamp"]:
            assert isinstance(doc["timestamp"], str) and RFC3339_ZULU_PATTERN.match(doc["timestamp"]), (
                f"{fname}: invalid timestamp: {doc['timestamp']}"
            )

        for field in ("id", "request_id", "stream_id", "error_id"):
            if field in doc and doc[field]:
                assert isinstance(doc[field], str) and ID_PATTERN.match(doc[field]), (
                    f"{fname}: invalid ID in '{field}': {doc[field]}"
                )


def _validate_string_field_size(
    obj: Any, path: str = "", issues: Optional[List[str]] = None
) -> List[str]:
    if issues is None:
        issues = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            current_path = f"{path}.{key}" if path else key
            if isinstance(value, str):
                lim = MAX_STRING_FIELD_SIZES.get(key)
                if lim and len(value) > lim:
                    issues.append(f"{current_path} exceeds limit: {len(value)} > {lim}")
            elif isinstance(value, (dict, list)):
                _validate_string_field_size(value, current_path, issues)

    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _validate_string_field_size(item, f"{path}[{i}]", issues)

    return issues


def test_large_fixture_performance():
    for fname, _sid in CASES:
        p = GOLDEN / fname
        if not p.exists():
            continue

        size = p.stat().st_size
        assert size <= MAX_FIXTURE_SIZE_BYTES, f"{fname} exceeds size limit: {size} bytes"

        doc = json.loads(p.read_text(encoding="utf-8"))
        issues = _validate_string_field_size(doc, fname)
        if issues:
            pytest.fail(f"{fname} string field size issues:\n" + "\n".join(issues))

# ------------------------------------------------------------------------------
# Parse-time performance & reliability
# ------------------------------------------------------------------------------
def test_golden_file_loading_performance():
    """Fail if any golden JSON file takes too long to parse (guard against accidental huge/slow fixtures)."""
    import time

    max_load_time = 2.0  # seconds
    slow_files: List[Tuple[str, float]] = []

    for fname, _schema_id in CASES:
        p = GOLDEN / fname
        if not p.exists():
            continue

        start = time.time()
        json.loads(p.read_text(encoding="utf-8"))
        load_time = time.time() - start

        if load_time > max_load_time:
            slow_files.append((fname, load_time))

    if slow_files:
        slow_info = "\n".join(
            f"  - {fname}: {t:.2f}s"
            for fname, t in sorted(slow_files, key=lambda x: x[1], reverse=True)
        )
        pytest.fail(
            "Some golden JSON fixtures are too slow to parse.\n"
            f"Max allowed: {max_load_time:.2f}s\n"
            "Slow files:\n"
            f"{slow_info}\n"
            "Fix: shrink the fixture(s), reduce large strings, or split into smaller samples."
        )

# ------------------------------------------------------------------------------
# Duplicate content check (advisory; skip rather than fail)
# ------------------------------------------------------------------------------
def test_golden_file_unique_checksums():
    checksums: Dict[str, List[str]] = {}
    for fname, _sid in CASES:
        p = GOLDEN / fname
        if not p.exists():
            continue
        checksum = hashlib.sha256(p.read_bytes()).hexdigest()
        checksums.setdefault(checksum, []).append(fname)

    duplicates = {ck: files for ck, files in checksums.items() if len(files) > 1}
    if duplicates:
        dup_info = "; ".join(str(v) for v in duplicates.values())
        pytest.skip(f"Duplicate golden file content: {dup_info}")

# ------------------------------------------------------------------------------
# Schema registry health (belt + suspenders)
# ------------------------------------------------------------------------------
def test_all_schemas_load_and_refs_resolve():
    """
    load_all_schemas_into_registry() runs in the session fixture; if it returns,
    schemas loaded and $refs should already be resolvable. This is a simple assertion
    to keep a readable failure location if the schema root is missing.
    """
    assert SCHEMAS_ROOT.exists(), f"{SCHEMAS_ROOT} must exist"


def test_schema_registry_health():
    assert SCHEMAS_ROOT.exists(), f"{SCHEMAS_ROOT} must exist"
    assert any(SCHEMAS_ROOT.rglob("*.json")), "No JSON schemas found under schemas root"
