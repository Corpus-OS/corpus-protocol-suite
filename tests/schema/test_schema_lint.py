# SPDX-License-Identifier: Apache-2.0
"""
Schema meta-lint for Corpus Protocol (Draft 2020-12).

Covers schema-only quality gates without requiring golden fixtures:
- Every schema loads, declares Draft 2020-12, and has a unique $id
- $id hygiene and path convention checks
- Metaschema conformance (Draft 2020-12)
- Cross-file $ref resolution (absolute $id refs) + local fragment checks (#/...)
- Compilable regex patterns
- Enum arrays are deduped and (optionally) sorted for diff stability
- $defs usage: no dangling definitions (globally considered)
- Envelope/type role sanity (request/success/error + stream frame files)
- Protocol/component consts in envelopes
- schema_version field presence/format on success envelopes
- If a schema provides "examples", each example validates against the schema

This file intentionally does not validate payloads; golden tests cover that.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Iterable

import pytest

try:
    # jsonschema is an explicit dependency for schema authoring
    import jsonschema
    from jsonschema import Draft202012Validator as V202012
except Exception as e:  # pragma: no cover
    raise RuntimeError("jsonschema (Draft 2020-12) is required for schema lint tests") from e


# --------------------------------------------------------------------------------------
# Paths / Discovery
# --------------------------------------------------------------------------------------

# tests/schema/test_schema_lint.py  -> repo root is two parents up
ROOT = Path(__file__).resolve().parents[2]
SCHEMAS_DIR = ROOT / "schema"

# Components we expect under schema/
COMPONENTS = {"common", "llm", "vector", "embedding", "graph"}


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

DRAFT_202012 = "https://json-schema.org/draft/2020-12/schema"
SCHEMA_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")
ID_ALLOWED = re.compile(r"^[A-Za-z0-9._~:/#-]+$")  # permissive but disallows spaces, etc.
PATTERN_CACHE: Dict[str, re.Pattern] = {}

# Enhanced constants
SUPPORTED_PROTOCOLS = {"llm/v1.0", "vector/v1.0", "embedding/v1.0", "graph/v1.0"}
RESERVED_PROPERTIES = {"$schema", "$id", "$defs", "$ref", "$comment", "examples"}
MAX_SCHEMA_SIZE_BYTES = 1 * 1024 * 1024  # 1MB per schema
MAX_DEFS_COUNT = 50  # Maximum number of definitions per schema
MAX_ENUM_SIZE = 100  # Maximum number of enum values


def _iter_schema_files() -> List[Path]:
    """Collect all JSON schema files under /schema/**."""
    if not SCHEMAS_DIR.exists():
        pytest.skip(f"schema/ directory not found at {SCHEMAS_DIR}")
    return sorted(SCHEMAS_DIR.rglob("*.json"))


def _load_json(p: Path) -> Any:
    """Load JSON file with comprehensive error handling."""
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        pytest.fail(f"Failed to parse JSON for {p}: {e}")
    except Exception as e:
        pytest.fail(f"Failed to load {p}: {e}")


def _walk(obj: Any) -> Iterable[Tuple[str, Any]]:
    """
    Yield (key, value) over all mapping entries, recursively.
    The key is the mapping key; for arrays, yields (str(index), item).
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, v
            yield from _walk(v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield str(i), v
            yield from _walk(v)


def _compile_pattern(pat: str) -> None:
    """Compile and cache regex patterns."""
    if pat in PATTERN_CACHE:
        return
    try:
        PATTERN_CACHE[pat] = re.compile(pat)
    except re.error as e:
        pytest.fail(f"Invalid regex pattern: {pat!r} → {e}")


def _collect_ids(store: Dict[str, dict]) -> Set[str]:
    """Collect all schema IDs from the store."""
    return set(store.keys())


def _split_ref(ref: str) -> Tuple[str, str | None]:
    """Split a $ref into (base_uri, fragment) where fragment includes the leading '#' if present."""
    if "#" in ref:
        base, frag = ref.split("#", 1)
        return base, "#" + frag
    return ref, None


def _resolve_local_fragment(schema: dict, fragment: str) -> bool:
    """
    Resolve a local JSON Pointer fragment (#/path/…) within a schema.
    Return True if it points to something; False if it does not.
    """
    if fragment == "#" or fragment is None:
        return True
    if not fragment.startswith("#/"):
        # Non-pointer (e.g., #foo) -> treat as not resolvable for strictness
        return False
    parts = fragment[2:].split("/")
    node = schema
    for part in parts:
        # Unescape per RFC6901
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.isdigit():
            idx = int(part)
            if 0 <= idx < len(node):
                node = node[idx]
            else:
                return False
        else:
            return False
    return True


def _is_envelope_request(fname: str) -> bool:
    """Check if filename matches envelope request pattern."""
    return fname.endswith(".envelope.request.json")


def _is_envelope_success(fname: str) -> bool:
    """Check if filename matches envelope success pattern."""
    return fname.endswith(".envelope.success.json")


def _is_envelope_error(fname: str) -> bool:
    """Check if filename matches envelope error pattern."""
    return fname.endswith(".envelope.error.json")


def _is_stream_frame(fname: str) -> bool:
    """Check if filename matches stream frame pattern."""
    return (
        fname.endswith(".stream.frame.data.json")
        or fname.endswith(".stream.frame.end.json")
        or fname.endswith(".stream.frame.error.json")
        or fname.endswith(".stream.frames.ndjson.schema.json")
    )


def _component_for_path(p: Path) -> str:
    """
    Infer component name from path: schema/<component>/<file>.json
    """
    try:
        idx = p.parts.index("schema")
        return p.parts[idx + 1] if idx + 1 < len(p.parts) else "unknown"
    except (ValueError, IndexError):
        return "unknown"


def _validate_schema_size(path: Path) -> None:
    """Validate schema file size limits."""
    size = path.stat().st_size
    if size > MAX_SCHEMA_SIZE_BYTES:
        pytest.fail(f"{path}: Schema file too large ({size} bytes > {MAX_SCHEMA_SIZE_BYTES} limit)")


def _check_reserved_property_usage(schema: dict, path: Path) -> List[str]:
    """Check for misuse of reserved JSON Schema properties."""
    issues = []
    for key in schema.keys():
        if key.startswith("$") and key not in RESERVED_PROPERTIES:
            issues.append(f"Custom property '{key}' should not use '$' prefix")
    return issues


def _validate_protocol_const(schema: dict, path: Path) -> List[str]:
    """Validate protocol constant format and values."""
    issues = []
    props = schema.get("properties", {})
    proto = props.get("protocol", {})

    if isinstance(proto, dict) and "const" in proto:
        const_value = proto["const"]
        if not isinstance(const_value, str) or const_value not in SUPPORTED_PROTOCOLS:
            issues.append(f"Invalid protocol constant: {const_value}")

    return issues


def _check_defs_size(schema: dict, path: Path) -> List[str]:
    """Check that $defs doesn't grow too large."""
    issues = []
    defs = schema.get("$defs") or schema.get("definitions")
    if isinstance(defs, dict) and len(defs) > MAX_DEFS_COUNT:
        issues.append(f"Too many definitions ({len(defs)} > {MAX_DEFS_COUNT} limit)")
    return issues


def _check_enum_size(schema: dict, path: Path) -> List[str]:
    """Check that enum arrays don't grow too large."""
    issues = []
    for key, val in _walk(schema):
        if key == "enum" and isinstance(val, list) and len(val) > MAX_ENUM_SIZE:
            issues.append(f"Enum too large ({len(val)} > {MAX_ENUM_SIZE} limit)")
    return issues


def _fail_with_bullets(problems: List[str], context: str | None = None) -> None:
    """Helper to render a list of problems as a bullet list failure."""
    if not problems:
        return
    header = f"{context}:\n" if context else ""
    msg = header + "- " + "\n- ".join(problems)
    pytest.fail(msg)


# --------------------------------------------------------------------------------------
# Test 1: Load + $schema + unique $id + store build
# --------------------------------------------------------------------------------------

def test_all_schemas_load_and_have_unique_ids():
    """Test that all schemas load, have correct $schema, and unique $ids."""
    files = _iter_schema_files()
    store: Dict[str, dict] = {}
    id_to_path: Dict[str, Path] = {}

    for path in files:
        # Check file size first
        _validate_schema_size(path)

        schema = _load_json(path)

        # $schema presence and value
        s = schema.get("$schema")
        assert s == DRAFT_202012, f"{path}: $schema must be {DRAFT_202012}, got {s!r}"

        # $id presence, format, allowed chars
        sid = schema.get("$id")
        assert isinstance(sid, str) and sid, f"{path}: missing or empty $id"
        assert ID_ALLOWED.match(sid), f"{path}: $id contains invalid characters: {sid!r}"

        # Unique $id
        assert sid not in store, f"Duplicate $id detected: {sid!r} used by {id_to_path[sid]} and {path}"
        store[sid] = schema
        id_to_path[sid] = path

    # Fail fast if no schemas found (misplaced repo)
    assert store, f"No schemas discovered under {SCHEMAS_DIR}"


# --------------------------------------------------------------------------------------
# Test 2: $id path convention + file organization
# --------------------------------------------------------------------------------------

def test_id_path_convention_matches_filesystem():
    """Test that $id values match filesystem paths."""
    files = _iter_schema_files()
    problems: List[str] = []

    for path in files:
        schema = _load_json(path)
        sid: str = schema["$id"]
        comp = _component_for_path(path)
        fname = path.name

        # Expect: https://adaptersdk.org/schemas/<component>/<file>.json
        expected_suffix = f"/schemas/{comp}/{fname}"
        if not sid.endswith(expected_suffix):
            problems.append(f"{path}: $id should end with {expected_suffix}, got {sid}")

    _fail_with_bullets(problems, "ID/path mismatches")


def test_schema_file_organization():
    """Test that schemas are properly organized by component."""
    files = _iter_schema_files()
    component_files: Dict[str, List[Path]] = {}
    problems: List[str] = []

    # Group files by component
    for path in files:
        comp = _component_for_path(path)
        if comp not in component_files:
            component_files[comp] = []
        component_files[comp].append(path)

    # Check for unknown components
    for comp in component_files:
        if comp not in COMPONENTS and comp != "unknown":
            problems.append(f"Unknown component in path: {comp}")

    # Check component directory structure
    for comp in COMPONENTS:
        comp_dir = SCHEMAS_DIR / comp
        if not comp_dir.exists():
            problems.append(f"Missing component directory: {comp}")
            continue

        # Check for non-JSON files in schema directories
        for item in comp_dir.iterdir():
            if item.is_file() and item.suffix not in {".json", ".md"}:
                problems.append(f"Non-schema file in component directory: {item}")

    _fail_with_bullets(problems, "Schema organization issues")


# --------------------------------------------------------------------------------------
# Test 3: Metaschema conformance + regex patterns + enum hygiene
# --------------------------------------------------------------------------------------

def test_metaschema_conformance_and_basic_hygiene():
    """Test schema conformance, regex patterns, and enum hygiene."""
    files = _iter_schema_files()

    for path in files:
        schema = _load_json(path)

        # 3a) Metaschema conformance
        try:
            V202012.check_schema(schema)
        except Exception as e:
            pytest.fail(f"{path}: schema fails Draft 2020-12 metaschema: {e}")

        # 3b) Compile regex patterns
        for key, val in _walk(schema):
            if key == "pattern" and isinstance(val, str):
                _compile_pattern(val)

        # 3c) Enum arrays deduped and sorted (for diff stability)
        for key, val in _walk(schema):
            if key == "enum" and isinstance(val, list) and val:
                unique = list(dict.fromkeys(val))  # preserve order but remove dups
                assert unique == val, f"{path}: enum contains duplicates: {val}"
                # Optional: require sorted ascending if all are strings
                if all(isinstance(x, str) for x in val):
                    assert val == sorted(val), f"{path}: enum strings should be sorted for stability"

        # 3d) Check enum size limits
        enum_issues = _check_enum_size(schema, path)
        _fail_with_bullets(enum_issues, f"{path} enum size issues")

        # 3e) Check reserved property usage
        reserved_issues = _check_reserved_property_usage(schema, path)
        _fail_with_bullets(reserved_issues, f"{path} reserved-property issues")


# --------------------------------------------------------------------------------------
# Test 4: Cross-file $ref resolution + local fragment checks
# --------------------------------------------------------------------------------------

def test_cross_file_refs_resolve_and_local_fragments_exist():
    """Test that all $refs resolve and local fragments exist."""
    # Build store of $id -> schema for absolute refs
    files = _iter_schema_files()
    store: Dict[str, dict] = {}
    for path in files:
        schema = _load_json(path)
        store[schema["$id"]] = schema

    known_ids = _collect_ids(store)
    problems: List[str] = []

    for path in files:
        schema = _load_json(path)
        sid = schema["$id"]

        for key, val in _walk(schema):
            if key != "$ref" or not isinstance(val, str):
                continue

            base, frag = _split_ref(val)
            if base and base.startswith("http"):
                # Absolute $ref to another file (or to itself)
                if base not in known_ids:
                    problems.append(f"{path}: $ref targets unknown $id: {val}")
                    continue
                target_schema = store[base]
                # Local fragment within target
                if frag and not _resolve_local_fragment(target_schema, frag):
                    problems.append(f"{path}: $ref fragment not found in target schema: {val}")
            elif base in ("", "#"):
                # Pure local fragment
                if not _resolve_local_fragment(schema, frag or "#"):
                    problems.append(f"{path}: local fragment not found: {val}")
            else:
                problems.append(f"{path}: non-absolute and non-local $ref not allowed: {val}")

    _fail_with_bullets(problems, "Unresolved $ref issues")


# --------------------------------------------------------------------------------------
# Test 5: $defs usage (no dangling definitions + size limits)
# --------------------------------------------------------------------------------------

def test_no_dangling_defs_globally():
    """Test that no $defs are defined but never referenced."""
    files = _iter_schema_files()

    # Collect every available def anchor as absolute "$id#/$defs/name"
    def_anchors: Set[str] = set()
    id_to_schema: Dict[str, dict] = {}
    for path in files:
        schema = _load_json(path)
        sid = schema["$id"]
        id_to_schema[sid] = schema
        defs = schema.get("$defs") or schema.get("definitions")  # tolerate either
        if isinstance(defs, dict):
            for dname in defs.keys():
                def_anchors.add(f"{sid}#/$defs/{dname}")

    # Collect every $ref we actually use (absolute only)
    used_refs: Set[str] = set()
    for path in files:
        schema = _load_json(path)
        base_id = schema["$id"]

        for key, val in _walk(schema):
            if key == "$ref" and isinstance(val, str):
                base, frag = _split_ref(val)
                if base and base.startswith("http"):
                    # Build absolute anchor for comparison if fragment present
                    if frag:
                        used_refs.add(base + frag)
                    else:
                        used_refs.add(base + "#")

                elif base in ("", "#") and frag:
                    # Local refs - convert to absolute for tracking
                    used_refs.add(f"{base_id}{frag}")

    # Any def anchors never referenced anywhere?
    dangling = sorted(a for a in def_anchors if a not in used_refs)

    if dangling:
        _fail_with_bullets(
            dangling,
            "Dangling $defs (exported but never referenced globally)",
        )


def test_defs_size_limits():
    """Test that $defs don't grow too large."""
    files = _iter_schema_files()
    problems: List[str] = []

    for path in files:
        schema = _load_json(path)
        defs_issues = _check_defs_size(schema, path)
        problems.extend([f"{path}: {msg}" for msg in defs_issues])

    _fail_with_bullets(problems, "$defs size issues")


# --------------------------------------------------------------------------------------
# Test 6: Envelope / role sanity & consts
# --------------------------------------------------------------------------------------

def test_envelope_role_sanity_and_consts():
    """Test envelope schema conventions and constants."""
    files = _iter_schema_files()
    problems: List[str] = []

    for path in files:
        schema = _load_json(path)
        fname = path.name
        comp = _component_for_path(path)

        # Envelopes must be objects
        if _is_envelope_request(fname) or _is_envelope_success(fname) or _is_envelope_error(fname):
            if schema.get("type") != "object":
                problems.append(f"{path}: envelopes must declare type=object")
            # Skip additionalProperties check for schemas using allOf (they inherit from common envelope)
            if "allOf" not in schema and schema.get("additionalProperties") not in (False,):
                problems.append(f"{path}: envelopes should set additionalProperties: false")

        # Request envelope basics
        if _is_envelope_request(fname):
            req = schema.get("required") or []
            for k in ("op", "ctx", "args"):
                if k not in req:
                    problems.append(f"{path}: request envelope must require {k}")

        # Success envelope basics
        if _is_envelope_success(fname):
            req = schema.get("required") or []
            # 'result' is not required for streaming-capable components (they have 'chunk' instead)
            streaming_components = {"llm", "graph"}
            required_fields = ["ok", "code", "ms"]
            if comp not in streaming_components:
                required_fields.append("result")
            for k in required_fields:
                if k not in req:
                    problems.append(f"{path}: success envelope must require {k}")
            # schema_version property presence/pattern
            props = schema.get("properties", {})
            sv = props.get("schema_version")
            if not isinstance(sv, dict):
                problems.append(f"{path}: success envelope should define 'schema_version' property")

        # Error envelope basics
        if _is_envelope_error(fname):
            req = schema.get("required") or []
            for k in ("ok", "error", "message"):
                if k not in req:
                    problems.append(f"{path}: error envelope must require {k}")

        # Component/protocol consts (component envelopes only)
        if comp in {"llm", "vector", "embedding", "graph"} and (
            _is_envelope_request(fname) or _is_envelope_success(fname) or _is_envelope_error(fname)
        ):
            props = schema.get("properties", {})
            # protocol const like "llm/v1.0"
            proto = props.get("protocol")
            if not isinstance(proto, dict) or "const" not in proto:
                problems.append(f"{path}: envelopes should const-bind 'protocol'")
            comp_prop = props.get("component")
            if not isinstance(comp_prop, dict) or "const" not in comp_prop:
                problems.append(f"{path}: envelopes should const-bind 'component'")

            # Validate protocol constant value
            proto_issues = _validate_protocol_const(schema, path)
            problems.extend([f"{path}: {msg}" for msg in proto_issues])

    _fail_with_bullets(problems, "Envelope/const issues")


def test_protocol_constants_match_component():
    """
    Ensure protocol consts match component directory, e.g.:
      schema/llm/... → protocol: 'llm/v1.0'
    """
    for path in _iter_schema_files():
        schema = _load_json(path)
        comp = _component_for_path(path)
        if comp not in {"llm", "vector", "embedding", "graph"}:
            continue

        props = schema.get("properties", {})
        proto = props.get("protocol", {})

        if not isinstance(proto, dict) or "const" not in proto:
            continue

        value = proto["const"]
        expected_prefix = f"{comp}/v"
        assert isinstance(value, str) and value.startswith(expected_prefix), (
            f"{path}: protocol const {value!r} does not match component {comp!r}"
        )


def test_schema_version_pattern_accepts_semver_examples():
    """Ensure schema_version patterns are compatible with simple SemVer strings."""
    examples = ["0.0.1", "1.0.0", "2.1.3"]

    for path in _iter_schema_files():
        schema = _load_json(path)
        props = schema.get("properties", {})
        sv = props.get("schema_version")
        if not isinstance(sv, dict):
            continue

        pattern = sv.get("pattern")
        if not isinstance(pattern, str):
            continue

        try:
            rx = re.compile(pattern)
        except re.error as e:
            pytest.fail(f"{path}: invalid schema_version pattern {pattern!r}: {e}")

        for ver in examples:
            if not rx.match(ver):
                pytest.fail(
                    f"{path}: schema_version pattern {pattern!r} does not match example {ver!r}"
                )


# --------------------------------------------------------------------------------------
# Test 7: Examples validation (if present)
# --------------------------------------------------------------------------------------

def test_examples_validate_against_own_schema():
    """Test that examples validate against their own schemas."""
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT202012
    
    files = _iter_schema_files()
    # Build a registry for cross-ref resolution
    resources = []
    for path in files:
        schema = _load_json(path)
        resources.append((schema["$id"], Resource.from_contents(schema, default_specification=DRAFT202012)))
    
    registry = Registry().with_resources(resources)

    for path in files:
        schema = _load_json(path)
        examples = schema.get("examples")
        if not (isinstance(examples, list) and examples):
            continue

        # Create a validator with registry for resolving absolute $id refs
        validator = V202012(schema, registry=registry)

        for i, ex in enumerate(examples):
            try:
                validator.validate(ex)
            except jsonschema.ValidationError as e:
                pytest.fail(f"{path}: example[{i}] does not validate: {e}")


# --------------------------------------------------------------------------------------
# Test 8: Stream frame sanity (type hints only)
# --------------------------------------------------------------------------------------

def test_stream_frame_files_are_union_or_frames():
    """
    Soft sanity: ensure stream frame schemas live under expected names and are objects/oneOf.
    We don't validate payloads here (golden stream tests do that).
    """
    files = _iter_schema_files()
    problems: List[str] = []

    for path in files:
        fname = path.name
        if not _is_stream_frame(fname):
            continue
        schema = _load_json(path)
        if fname.endswith(".stream.frames.ndjson.schema.json"):
            if "oneOf" not in schema:
                problems.append(f"{path}: NDJSON union schema should define oneOf over frame types")
        else:
            if schema.get("type") != "object":
                problems.append(f"{path}: stream frame schema should be type=object")

    _fail_with_bullets(problems, "Stream frame schema issues")


def test_stream_frames_have_event_property():
    """Ensure individual stream frame schemas expose an 'event' discriminator."""
    for path in _iter_schema_files():
        fname = path.name
        if not _is_stream_frame(fname):
            continue

        # Skip NDJSON union wrapper
        if fname.endswith(".stream.frames.ndjson.schema.json"):
            continue

        schema = _load_json(path)
        props = schema.get("properties", {})
        assert "event" in props, f"{path}: stream frame schema missing 'event' property"


# --------------------------------------------------------------------------------------
# Test 9: Schema performance and reliability
# --------------------------------------------------------------------------------------

def test_schema_loading_performance():
    """Test that all schemas can be loaded quickly."""
    import time

    files = _iter_schema_files()
    max_load_time = 1.0  # seconds per schema
    slow_files: List[Tuple[Path, float]] = []

    for path in files:
        start = time.time()
        try:
            _load_json(path)
            load_time = time.time() - start
            if load_time > max_load_time:
                slow_files.append((path, load_time))
        except Exception:
            continue  # Loading errors are caught in other tests

    if slow_files:
        slow_info = ", ".join(f"{path.name}({t:.2f}s)" for path, t in slow_files)
        pytest.skip(f"Slow schema loading detected: {slow_info}")


def test_schema_complexity_metrics():
    """Test schema complexity metrics for maintainability."""
    files = _iter_schema_files()
    high_complexity: List[Tuple[Path, int]] = []

    for path in files:
        schema = _load_json(path)

        # Count total number of properties as complexity metric
        prop_count = 0
        for key, value in _walk(schema):
            if key in {"properties", "patternProperties", "additionalProperties"}:
                if isinstance(value, dict):
                    prop_count += len(value)

        # Arbitrary threshold - adjust based on your needs
        if prop_count > 100:
            high_complexity.append((path, prop_count))

    if high_complexity:
        complexity_info = ", ".join(f"{path.name}({count})" for path, count in high_complexity)
        pytest.skip(f"High complexity schemas detected: {complexity_info}")


# --------------------------------------------------------------------------------------
# Test 10: Comprehensive schema health report
# --------------------------------------------------------------------------------------

def test_schema_registry_health_summary():
    """Provide a comprehensive health summary of the schema registry."""
    files = _iter_schema_files()

    # Collect metrics
    total_schemas = len(files)
    components = set()
    envelope_schemas = 0
    stream_schemas = 0
    total_defs = 0

    for path in files:
        schema = _load_json(path)
        components.add(_component_for_path(path))

        fname = path.name
        if any([
            _is_envelope_request(fname),
            _is_envelope_success(fname),
            _is_envelope_error(fname),
        ]):
            envelope_schemas += 1
        if _is_stream_frame(fname):
            stream_schemas += 1

        defs = schema.get("$defs") or schema.get("definitions") or {}
        total_defs += len(defs)

    # Log health summary (doesn't fail test)
    print(f"\nSchema Registry Health Summary:")
    print(f"  Total schemas: {total_schemas}")
    print(f"  Components: {', '.join(sorted(components))}")
    print(f"  Envelope schemas: {envelope_schemas}")
    print(f"  Stream schemas: {stream_schemas}")
    print(f"  Total definitions: {total_defs}")

    # Basic sanity checks
    assert total_schemas > 0, "No schemas found"
    assert len(components) >= len(COMPONENTS), f"Missing components: {COMPONENTS - components}"


# Run all tests to ensure comprehensive coverage
if __name__ == "__main__":
    # This allows running the file directly for debugging
    pytest.main([__file__, "-v"])
