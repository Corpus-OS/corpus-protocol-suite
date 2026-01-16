# SPDX-License-Identifier: Apache-2.0
"""
Schema meta-lint for Corpus Protocol (Draft 2020-12).

Covers schema-only quality gates without requiring golden fixtures, and MUST CONFORM to SCHEMA.md:
- Every schema loads, declares Draft 2020-12, and has a unique $id
- $id hygiene and path convention checks
- Metaschema conformance (Draft 2020-12)
- Cross-file $ref resolution (absolute $id refs) + local fragment checks (#/...)
- Compilable regex patterns
- Enum arrays are deduped (ordering is NOT enforced; SCHEMA.md does not require sorted enums)
- $defs / definitions usage: no dangling definitions (globally considered)
- Envelope role sanity consistent with SCHEMA.md:
  - Common envelopes (common/envelope.*.json) are normative for required/additionalProperties.
  - Protocol envelopes should allOf/$ref the corresponding common envelope.
  - Do NOT require envelopes to declare protocol/component/schema_version fields (SCHEMA.md does not define them on envelopes).
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

# Prefer ./schemas if present, else ./schema
SCHEMAS_DIR = (ROOT / "schemas") if (ROOT / "schemas").exists() else (ROOT / "schema")

# Components we expect under schema(s)/ (SCHEMA.md also defines ndjson union schema)
COMPONENTS = {"common", "llm", "vector", "embedding", "graph", "ndjson"}


# --------------------------------------------------------------------------------------
# Utilities / constants
# --------------------------------------------------------------------------------------

DRAFT_202012 = "https://json-schema.org/draft/2020-12/schema"
SCHEMA_ID_BASE = "https://corpusos.com/schemas"

ID_ALLOWED = re.compile(r"^[A-Za-z0-9._~:/#-]+$")  # permissive but disallows spaces, etc.
PATTERN_CACHE: Dict[str, re.Pattern] = {}

# Enhanced constants (guardrails; SCHEMA.md does not define these, but they are safe lint-only checks)
SUPPORTED_PROTOCOLS = {"llm/v1.0", "vector/v1.0", "embedding/v1.0", "graph/v1.0"}
MAX_SCHEMA_SIZE_BYTES = 1 * 1024 * 1024  # 1MB per schema
MAX_DEFS_COUNT = 50  # Maximum number of definitions per schema
MAX_ENUM_SIZE = 100  # Maximum number of enum values

# Draft 2020-12 reserved "$" keywords (non-exhaustive but covers normal usage).
# If you legitimately need a new "$keyword", add it here deliberately.
ALLOWED_DOLLAR_KEYWORDS = {
    "$schema",
    "$id",
    "$ref",
    "$defs",
    "$comment",
    "$anchor",
    "$dynamicRef",
    "$dynamicAnchor",
    "$vocabulary",
    # tolerate older/different drafts if they appear in shared tooling
    "$recursiveRef",
    "$recursiveAnchor",
}


def _iter_schema_files() -> List[Path]:
    """Collect all JSON schema files under /schema(s)/**."""
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
    node: Any = schema
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
    # common: envelope.request.json ; protocol: llm.envelope.request.json, etc.
    return fname == "envelope.request.json" or fname.endswith(".envelope.request.json")


def _is_envelope_success(fname: str) -> bool:
    # common: envelope.success.json ; protocol: llm.envelope.success.json, etc.
    return fname == "envelope.success.json" or fname.endswith(".envelope.success.json")


def _is_envelope_error(fname: str) -> bool:
    # common: envelope.error.json ; protocol: llm.envelope.error.json, etc.
    return fname == "envelope.error.json" or fname.endswith(".envelope.error.json")


def _is_common_stream_envelope(fname: str) -> bool:
    # SCHEMA.md: common/envelope.stream.success.json
    return fname == "envelope.stream.success.json"


def _component_for_path(p: Path) -> str:
    """
    Infer component name from path: schema(s)/<component>/<file>.json
    """
    try:
        rel = p.relative_to(SCHEMAS_DIR)
        return rel.parts[0] if rel.parts else "unknown"
    except Exception:
        return "unknown"


def _validate_schema_size(path: Path) -> None:
    """Validate schema file size limits."""
    size = path.stat().st_size
    if size > MAX_SCHEMA_SIZE_BYTES:
        pytest.fail(f"{path}: Schema file too large ({size} bytes > {MAX_SCHEMA_SIZE_BYTES} limit)")


def _check_reserved_dollar_keyword_usage(schema: dict, path: Path) -> List[str]:
    """
    Flag custom "$*" keys. JSON Schema reserves "$" for vocabulary keywords.
    """
    issues: List[str] = []
    if not isinstance(schema, dict):
        return issues
    for key in schema.keys():
        if isinstance(key, str) and key.startswith("$") and key not in ALLOWED_DOLLAR_KEYWORDS:
            issues.append(f"Custom property '{key}' should not use '$' prefix (reserved keyword space)")
    return issues


def _validate_protocol_const(schema: dict) -> List[str]:
    """
    Validate protocol constant format and values, but ONLY if present.
    SCHEMA.md uses protocol consts on some capability schemas; envelopes do not require them.
    """
    issues: List[str] = []
    props = schema.get("properties", {})
    proto = props.get("protocol", {})

    if isinstance(proto, dict) and "const" in proto:
        const_value = proto["const"]
        if not isinstance(const_value, str) or const_value not in SUPPORTED_PROTOCOLS:
            issues.append(f"Invalid protocol constant: {const_value!r}")

    return issues


def _check_defs_size(schema: dict) -> List[str]:
    """Check that $defs/definitions doesn't grow too large."""
    issues: List[str] = []
    defs = schema.get("$defs") or schema.get("definitions")
    if isinstance(defs, dict) and len(defs) > MAX_DEFS_COUNT:
        issues.append(f"Too many definitions ({len(defs)} > {MAX_DEFS_COUNT} limit)")
    return issues


def _check_enum_size(schema: dict) -> List[str]:
    """Check that enum arrays don't grow too large."""
    issues: List[str] = []
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


def _allof_refs(schema: dict) -> List[str]:
    """Return any $ref values found at the top-level allOf entries."""
    refs: List[str] = []
    if not isinstance(schema, dict):
        return refs
    allof = schema.get("allOf")
    if not isinstance(allof, list):
        return refs
    for entry in allof:
        if isinstance(entry, dict) and isinstance(entry.get("$ref"), str):
            refs.append(entry["$ref"])
    return refs


# --------------------------------------------------------------------------------------
# Test 1: Load + $schema + unique $id + store build
# --------------------------------------------------------------------------------------

def test_all_schemas_load_and_have_unique_ids():
    """Test that all schemas load, have correct $schema, and unique $ids."""
    files = _iter_schema_files()
    store: Dict[str, dict] = {}
    id_to_path: Dict[str, Path] = {}

    for path in files:
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

        expected = f"{SCHEMA_ID_BASE}/{comp}/{fname}"
        if sid != expected:
            problems.append(f"{path}: $id should be {expected}, got {sid}")

    _fail_with_bullets(problems, "ID/path mismatches")


def test_schema_file_organization():
    """Test that schemas are properly organized by component."""
    files = _iter_schema_files()
    component_files: Dict[str, List[Path]] = {}
    problems: List[str] = []

    # Group files by component
    for path in files:
        comp = _component_for_path(path)
        component_files.setdefault(comp, []).append(path)

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
    """Test schema conformance, regex patterns, enum hygiene, and '$' keyword usage."""
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

        # 3c) Enum arrays deduped (SCHEMA.md does not require ordering)
        for key, val in _walk(schema):
            if key == "enum" and isinstance(val, list) and val:
                unique = list(dict.fromkeys(val))  # preserve order but remove dups
                assert unique == val, f"{path}: enum contains duplicates: {val}"

        # 3d) Check enum size limits
        enum_issues = _check_enum_size(schema)
        _fail_with_bullets([f"{path}: {msg}" for msg in enum_issues], f"{path} enum size issues")

        # 3e) Reserved "$" keyword usage
        dollar_issues = _check_reserved_dollar_keyword_usage(schema, path)
        _fail_with_bullets(dollar_issues, f"{path} reserved-$keyword issues")


# --------------------------------------------------------------------------------------
# Test 4: Cross-file $ref resolution + local fragment checks
# --------------------------------------------------------------------------------------

def test_cross_file_refs_resolve_and_local_fragments_exist():
    """Test that all $refs resolve and local fragments exist."""
    files = _iter_schema_files()
    store: Dict[str, dict] = {}
    for path in files:
        schema = _load_json(path)
        store[schema["$id"]] = schema

    known_ids = _collect_ids(store)
    problems: List[str] = []

    for path in files:
        schema = _load_json(path)

        for key, val in _walk(schema):
            if key != "$ref" or not isinstance(val, str):
                continue

            base, frag = _split_ref(val)
            if base and base.startswith("http"):
                if base not in known_ids:
                    problems.append(f"{path}: $ref targets unknown $id: {val}")
                    continue
                target_schema = store[base]
                if frag and not _resolve_local_fragment(target_schema, frag):
                    problems.append(f"{path}: $ref fragment not found in target schema: {val}")
            elif base in ("", "#"):
                if not _resolve_local_fragment(schema, frag or "#"):
                    problems.append(f"{path}: local fragment not found: {val}")
            else:
                problems.append(f"{path}: non-absolute and non-local $ref not allowed: {val}")

    _fail_with_bullets(problems, "Unresolved $ref issues")


# --------------------------------------------------------------------------------------
# Test 5: $defs / definitions usage (no dangling definitions + size limits)
# --------------------------------------------------------------------------------------

def test_no_dangling_defs_globally():
    """Test that no $defs/definitions are defined but never referenced."""
    files = _iter_schema_files()

    # Collect every available def anchor as absolute "$id#/<container>/<name>"
    def_anchors: Set[str] = set()
    for path in files:
        schema = _load_json(path)
        sid = schema["$id"]

        defs = schema.get("$defs")
        if isinstance(defs, dict):
            for dname in defs.keys():
                def_anchors.add(f"{sid}#/$defs/{dname}")

        legacy_defs = schema.get("definitions")
        if isinstance(legacy_defs, dict):
            for dname in legacy_defs.keys():
                def_anchors.add(f"{sid}#/definitions/{dname}")

    # Collect every $ref we actually use (absolute + local normalized to absolute)
    used_refs: Set[str] = set()
    for path in files:
        schema = _load_json(path)
        base_id = schema["$id"]

        for key, val in _walk(schema):
            if key == "$ref" and isinstance(val, str):
                base, frag = _split_ref(val)
                if base and base.startswith("http"):
                    used_refs.add(base + (frag or "#"))
                elif base in ("", "#") and frag:
                    used_refs.add(f"{base_id}{frag}")

    dangling = sorted(a for a in def_anchors if a not in used_refs)
    if dangling:
        _fail_with_bullets(
            dangling,
            "Dangling $defs/definitions (exported but never referenced globally)",
        )


def test_defs_size_limits():
    """Test that $defs/definitions don't grow too large."""
    problems: List[str] = []

    for path in _iter_schema_files():
        schema = _load_json(path)
        defs_issues = _check_defs_size(schema)
        problems.extend([f"{path}: {msg}" for msg in defs_issues])

    _fail_with_bullets(problems, "$defs size issues")


# --------------------------------------------------------------------------------------
# Test 6: Envelope role sanity (SCHEMA.md-conformant) + optional protocol const hygiene
# --------------------------------------------------------------------------------------

def test_envelope_role_sanity():
    """
    Test envelope schema conventions per SCHEMA.md.

    Key rule: common/* envelopes are normative for required/additionalProperties.
    Protocol envelopes typically inherit via allOf + $ref to common envelopes.
    This lint MUST NOT require duplicated 'required' on inheriting envelopes.
    """
    problems: List[str] = []

    common_request_id = f"{SCHEMA_ID_BASE}/common/envelope.request.json"
    common_success_id = f"{SCHEMA_ID_BASE}/common/envelope.success.json"
    common_error_id = f"{SCHEMA_ID_BASE}/common/envelope.error.json"
    common_stream_id = f"{SCHEMA_ID_BASE}/common/envelope.stream.success.json"

    for path in _iter_schema_files():
        schema = _load_json(path)
        fname = path.name
        comp = _component_for_path(path)

        is_env = (
            _is_envelope_request(fname)
            or _is_envelope_success(fname)
            or _is_envelope_error(fname)
            or _is_common_stream_envelope(fname)
        )

        if not is_env:
            continue

        # Envelopes should be objects
        if schema.get("type") != "object":
            problems.append(f"{path}: envelopes must declare type=object")

        # ---- Common envelopes are normative ----
        if comp == "common" and fname == "envelope.request.json":
            req = schema.get("required") or []
            for k in ("op", "ctx", "args"):
                if k not in req:
                    problems.append(f"{path}: common request envelope must require {k}")
            # SCHEMA.md: request envelope allows additional properties (extensibility at boundary)
            if schema.get("additionalProperties") is not True:
                problems.append(f"{path}: common request envelope additionalProperties must be true")

        if comp == "common" and fname == "envelope.success.json":
            req = schema.get("required") or []
            for k in ("ok", "code", "ms", "result"):
                if k not in req:
                    problems.append(f"{path}: common success envelope must require {k}")
            if schema.get("additionalProperties") is not False:
                problems.append(f"{path}: common success envelope additionalProperties must be false")

        if comp == "common" and fname == "envelope.error.json":
            req = schema.get("required") or []
            for k in ("ok", "code", "error", "message", "retry_after_ms", "details", "ms"):
                if k not in req:
                    problems.append(f"{path}: common error envelope must require {k}")
            if schema.get("additionalProperties") is not False:
                problems.append(f"{path}: common error envelope additionalProperties must be false")

        if comp == "common" and fname == "envelope.stream.success.json":
            req = schema.get("required") or []
            for k in ("ok", "code", "ms", "chunk"):
                if k not in req:
                    problems.append(f"{path}: common stream success envelope must require {k}")
            if schema.get("additionalProperties") is not False:
                problems.append(f"{path}: common stream success envelope additionalProperties must be false")

        # ---- Protocol envelopes should allOf/$ref corresponding common envelope ----
        if comp in {"llm", "vector", "embedding", "graph"}:
            refs = _allof_refs(schema)

            if fname.endswith(".envelope.request.json"):
                if common_request_id not in refs:
                    problems.append(f"{path}: protocol request envelope should allOf/$ref {common_request_id}")

            if fname.endswith(".envelope.success.json"):
                if common_success_id not in refs:
                    problems.append(f"{path}: protocol success envelope should allOf/$ref {common_success_id}")

            if fname.endswith(".envelope.error.json"):
                if common_error_id not in refs:
                    problems.append(f"{path}: protocol error envelope should allOf/$ref {common_error_id}")

            # Streaming operation success schemas: allOf/$ref common stream envelope
            # (They may not be named *.envelope.*, so we detect by reference)
            if common_stream_id in refs and schema.get("type") != "object":
                problems.append(f"{path}: streaming success schema should declare type=object")

    _fail_with_bullets(problems, "Envelope issues")


def test_protocol_constants_match_component_when_present():
    """
    SCHEMA.md uses protocol consts on *some* schemas (e.g., capabilities types).
    If a schema defines properties.protocol.const, ensure it matches <component>/v*.
    (Do NOT require that every schema has protocol const.)
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

        # Optional: validate against known supported protocol strings
        proto_issues = _validate_protocol_const(schema)
        if proto_issues:
            _fail_with_bullets([f"{path}: {msg}" for msg in proto_issues], "Protocol const issues")


# --------------------------------------------------------------------------------------
# Test 7: Examples validation (if present)
# --------------------------------------------------------------------------------------

def test_examples_validate_against_own_schema():
    """Test that examples validate against their own schemas."""
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT202012

    files = _iter_schema_files()

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

        validator = V202012(schema, registry=registry)

        for i, ex in enumerate(examples):
            try:
                validator.validate(ex)
            except jsonschema.ValidationError as e:
                pytest.fail(f"{path}: example[{i}] does not validate: {e}")


# --------------------------------------------------------------------------------------
# Test 8: Schema performance and reliability
# --------------------------------------------------------------------------------------

def test_schema_loading_performance():
    """Fail if schema JSON loading is unexpectedly slow (signals huge/complex schemas)."""
    import time

    files = _iter_schema_files()
    max_load_time = 1.0  # seconds per schema
    slow_files: List[Tuple[Path, float]] = []

    for path in files:
        start = time.time()
        _load_json(path)
        load_time = time.time() - start
        if load_time > max_load_time:
            slow_files.append((path, load_time))

    if slow_files:
        slow_info = ", ".join(f"{path.name}({t:.2f}s)" for path, t in slow_files)
        pytest.fail(f"Slow schema loading detected: {slow_info}")


def test_schema_complexity_metrics():
    """Advisory: flag very high schema complexity (skip rather than fail)."""
    files = _iter_schema_files()
    high_complexity: List[Tuple[Path, int]] = []

    for path in files:
        schema = _load_json(path)

        prop_count = 0
        for key, value in _walk(schema):
            if key in {"properties", "patternProperties"} and isinstance(value, dict):
                prop_count += len(value)

        if prop_count > 100:
            high_complexity.append((path, prop_count))

    if high_complexity:
        complexity_info = ", ".join(f"{path.name}({count})" for path, count in high_complexity)
        pytest.skip(f"High complexity schemas detected: {complexity_info}")


# --------------------------------------------------------------------------------------
# Test 9: Comprehensive schema health report
# --------------------------------------------------------------------------------------

def test_schema_registry_health_summary():
    """Provide a comprehensive health summary of the schema registry."""
    files = _iter_schema_files()

    total_schemas = len(files)
    components = set()
    envelope_schemas = 0
    stream_envelopes = 0
    total_defs = 0

    for path in files:
        schema = _load_json(path)
        components.add(_component_for_path(path))

        fname = path.name
        if _is_envelope_request(fname) or _is_envelope_success(fname) or _is_envelope_error(fname):
            envelope_schemas += 1
        if _is_common_stream_envelope(fname):
            stream_envelopes += 1

        defs = schema.get("$defs") or schema.get("definitions") or {}
        if isinstance(defs, dict):
            total_defs += len(defs)

    print(f"\nSchema Registry Health Summary:")
    print(f"  Schema root: {SCHEMAS_DIR}")
    print(f"  Total schemas: {total_schemas}")
    print(f"  Components: {', '.join(sorted(components))}")
    print(f"  Envelope schemas: {envelope_schemas}")
    print(f"  Common stream envelope schemas: {stream_envelopes}")
    print(f"  Total definitions: {total_defs}")

    assert total_schemas > 0, "No schemas found"
    missing_components = COMPONENTS - components
    assert not missing_components, f"Missing components: {missing_components}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
