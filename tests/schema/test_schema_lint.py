# SPDX-License-Identifier: Apache-2.0
"""
Schema meta-lint for Corpus Protocol (Draft 2020-12).

Covers schema-only quality gates without requiring golden fixtures, and MUST CONFORM to SCHEMA.md:
- Every schema loads, declares Draft 2020-12, and has a unique $id
- $id hygiene and path convention checks (supports subdirectories; $id must match full relative path)
- Metaschema conformance (Draft 2020-12)
- Cross-file $ref resolution (absolute $id refs) + local fragment checks (#/...)
- Compilable regex patterns
- Enum arrays are deduped (ordering is NOT enforced; SCHEMA.md does not require sorted enums)
- $defs / definitions usage: no dangling definitions (globally considered)
- Envelope role sanity consistent with SCHEMA.md:
  - Common envelopes (common/envelope.*.json) are normative for required/additionalProperties.
  - Protocol envelopes should allOf/$ref the corresponding common envelope.
  - Streaming operation success schemas MUST allOf/$ref common/envelope.stream.success.json.
  - Do NOT require envelopes to declare protocol/component/schema_version fields (SCHEMA.md does not define them on envelopes).
- Additional SCHEMA.md-aligned invariants for *common* envelopes:
  - common/envelope.success.json: ok.const == true, code.const == "OK"
  - common/envelope.error.json: ok.const == false, code.pattern == "^[A-Z_]+$"
  - common/envelope.stream.success.json: ok.const == true, code.const == "STREAMING"
  - common/envelope.request.json: properties.ctx is a $ref to common/operation_context.json
- If a schema provides "examples", each example validates against the schema

This file intentionally does not validate payloads; golden tests cover that.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

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
    Infer component name from path: schema(s)/<component>/.../<file>.json
    """
    try:
        rel = p.relative_to(SCHEMAS_DIR)
        return rel.parts[0] if rel.parts else "unknown"
    except Exception:
        return "unknown"


def _check_reserved_dollar_keyword_usage(schema: Any, path: Path) -> List[str]:
    """
    Flag custom "$*" keys anywhere in the schema tree. JSON Schema reserves "$" for vocabulary keywords.
    """
    issues: List[str] = []

    def _recurse(node: Any, jpath: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(k, str) and k.startswith("$") and k not in ALLOWED_DOLLAR_KEYWORDS:
                    issues.append(f"{path}: {jpath}.{k}: custom '$' key is not allowed (reserved keyword space)")
                _recurse(v, f"{jpath}.{k}" if jpath else k)
        elif isinstance(node, list):
            for i, item in enumerate(node):
                _recurse(item, f"{jpath}[{i}]")

    _recurse(schema, "")
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


def _prop(schema: dict, name: str) -> dict | None:
    """Convenience: return schema['properties'][name] if it's a dict."""
    props = schema.get("properties")
    if not isinstance(props, dict):
        return None
    v = props.get(name)
    return v if isinstance(v, dict) else None


def _is_streaming_success_schema(path: Path) -> bool:
    """
    Heuristic for streaming success schemas:
    - Component is one of llm/graph/embedding
    - Filename includes 'stream'
    - Filename ends with '.success.json'
    This matches SCHEMA.md naming for:
      llm.stream.success.json
      graph.stream_query.success.json
      embedding.stream_embed.success.json
    """
    comp = _component_for_path(path)
    if comp not in {"llm", "graph", "embedding"}:
        return False
    fname = path.name
    return fname.endswith(".success.json") and ("stream" in fname)


# --------------------------------------------------------------------------------------
# Test 1: Load + $schema + unique $id + store build
# --------------------------------------------------------------------------------------

def test_all_schemas_load_and_have_unique_ids():
    """Test that all schemas load, have correct $schema, and unique $ids."""
    files = _iter_schema_files()
    store: Dict[str, dict] = {}
    id_to_path: Dict[str, Path] = {}

    for path in files:
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
    """
    Test that $id values match filesystem paths, including subdirectories.

    Required convention:
      $id == https://corpusos.com/schemas/<relative_path_from_schema_root>

    Example:
      schemas/llm/ops/llm.complete.request.json
        -> https://corpusos.com/schemas/llm/ops/llm.complete.request.json
    """
    files = _iter_schema_files()
    problems: List[str] = []

    for path in files:
        schema = _load_json(path)
        sid: str = schema["$id"]

        rel = path.relative_to(SCHEMAS_DIR).as_posix()
        expected = f"{SCHEMA_ID_BASE}/{rel}"
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

        # Check for non-JSON files in schema directories (allow .md for docs)
        for item in comp_dir.rglob("*"):
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

        # 3d) Reserved "$" keyword usage (anywhere in tree)
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
                # Policy: SCHEMA.md expects $id-based absolute refs or local fragments
                problems.append(f"{path}: non-absolute and non-local $ref not allowed: {val}")

    _fail_with_bullets(problems, "Unresolved $ref issues")


# --------------------------------------------------------------------------------------
# Test 5: $defs / definitions usage (no dangling definitions)
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


# --------------------------------------------------------------------------------------
# Test 6: Envelope role sanity (SCHEMA.md-conformant) + common-envelope invariants
# --------------------------------------------------------------------------------------

def test_envelope_role_sanity_and_common_invariants():
    """
    Test envelope schema conventions per SCHEMA.md.

    Key rule: common/* envelopes are normative for required/additionalProperties.
    Protocol envelopes typically inherit via allOf + $ref to common envelopes.
    Streaming operation success schemas MUST inherit via allOf + $ref to common stream envelope.
    This lint MUST NOT require duplicated 'required' on inheriting envelopes.

    Also enforces additional SCHEMA.md-aligned invariants on common envelopes:
      - success: ok const true; code const OK
      - error: ok const false; code pattern canonical
      - stream success: ok const true; code const STREAMING
      - request: ctx should $ref operation_context.json
    """
    problems: List[str] = []

    common_request_id = f"{SCHEMA_ID_BASE}/common/envelope.request.json"
    common_success_id = f"{SCHEMA_ID_BASE}/common/envelope.success.json"
    common_error_id = f"{SCHEMA_ID_BASE}/common/envelope.error.json"
    common_stream_id = f"{SCHEMA_ID_BASE}/common/envelope.stream.success.json"
    operation_ctx_id = f"{SCHEMA_ID_BASE}/common/operation_context.json"

    for path in _iter_schema_files():
        schema = _load_json(path)
        fname = path.name
        comp = _component_for_path(path)

        is_env = (
            _is_envelope_request(fname)
            or _is_envelope_success(fname)
            or _is_envelope_error(fname)
            or _is_common_stream_envelope(fname)
            or _is_streaming_success_schema(path)
        )
        if not is_env:
            continue

        # Envelopes and streaming success schemas should be objects
        if schema.get("type") != "object":
            problems.append(f"{path}: envelope-like schemas must declare type=object")

        # ---- Common envelopes are normative ----
        if comp == "common" and fname == "envelope.request.json":
            req = schema.get("required") or []
            for k in ("op", "ctx", "args"):
                if k not in req:
                    problems.append(f"{path}: common request envelope must require {k}")

            # SCHEMA.md: request envelope allows additional properties (extensibility at boundary)
            if schema.get("additionalProperties") is not True:
                problems.append(f"{path}: common request envelope additionalProperties must be true")

            # SCHEMA.md-aligned invariant: ctx should $ref operation_context.json
            ctx = _prop(schema, "ctx")
            if not ctx or ctx.get("$ref") != operation_ctx_id:
                problems.append(f"{path}: properties.ctx must be $ref {operation_ctx_id}")

        if comp == "common" and fname == "envelope.success.json":
            req = schema.get("required") or []
            for k in ("ok", "code", "ms", "result"):
                if k not in req:
                    problems.append(f"{path}: common success envelope must require {k}")
            if schema.get("additionalProperties") is not False:
                problems.append(f"{path}: common success envelope additionalProperties must be false")

            # SCHEMA.md-aligned invariants: ok const true; code const OK
            ok = _prop(schema, "ok")
            if not ok or ok.get("const") is not True:
                problems.append(f"{path}: properties.ok must have const true")
            code = _prop(schema, "code")
            if not code or code.get("const") != "OK":
                problems.append(f"{path}: properties.code must have const 'OK'")

        if comp == "common" and fname == "envelope.error.json":
            req = schema.get("required") or []
            for k in ("ok", "code", "error", "message", "retry_after_ms", "details", "ms"):
                if k not in req:
                    problems.append(f"{path}: common error envelope must require {k}")
            if schema.get("additionalProperties") is not False:
                problems.append(f"{path}: common error envelope additionalProperties must be false")

            # SCHEMA.md-aligned invariants: ok const false; code pattern canonical
            ok = _prop(schema, "ok")
            if not ok or ok.get("const") is not False:
                problems.append(f"{path}: properties.ok must have const false")
            code = _prop(schema, "code")
            if not code or code.get("pattern") != "^[A-Z_]+$":
                problems.append(f"{path}: properties.code must have pattern '^[A-Z_]+$'")

        if comp == "common" and fname == "envelope.stream.success.json":
            req = schema.get("required") or []
            for k in ("ok", "code", "ms", "chunk"):
                if k not in req:
                    problems.append(f"{path}: common stream success envelope must require {k}")
            if schema.get("additionalProperties") is not False:
                problems.append(f"{path}: common stream success envelope additionalProperties must be false")

            # SCHEMA.md-aligned invariants: ok const true; code const STREAMING
            ok = _prop(schema, "ok")
            if not ok or ok.get("const") is not True:
                problems.append(f"{path}: properties.ok must have const true")
            code = _prop(schema, "code")
            if not code or code.get("const") != "STREAMING":
                problems.append(f"{path}: properties.code must have const 'STREAMING'")

        # ---- Protocol envelopes should allOf/$ref corresponding common envelope ----
        if comp in {"llm", "vector", "embedding", "graph"}:
            refs = _allof_refs(schema)

            if fname.endswith(".envelope.request.json") and common_request_id not in refs:
                problems.append(f"{path}: protocol request envelope should allOf/$ref {common_request_id}")

            if fname.endswith(".envelope.success.json") and common_success_id not in refs:
                problems.append(f"{path}: protocol success envelope should allOf/$ref {common_success_id}")

            if fname.endswith(".envelope.error.json") and common_error_id not in refs:
                problems.append(f"{path}: protocol error envelope should allOf/$ref {common_error_id}")

            # Streaming operation success schemas MUST allOf/$ref the common stream envelope
            if _is_streaming_success_schema(path) and common_stream_id not in refs:
                problems.append(f"{path}: streaming success schema must allOf/$ref {common_stream_id}")

    _fail_with_bullets(problems, "Envelope issues")


# --------------------------------------------------------------------------------------
# Test 7: Examples validation (if present)
# --------------------------------------------------------------------------------------

def test_examples_validate_against_own_schema():
    """
    Test that examples validate against their own schemas.

    Uses 'referencing' registry if available (preferred), otherwise falls back to RefResolver.
    """
    files = _iter_schema_files()

    # Build an id->schema store for ref resolution
    store: Dict[str, dict] = {}
    for path in files:
        schema = _load_json(path)
        store[schema["$id"]] = schema

    # Preferred path: referencing-based registry (jsonschema v4+)
    try:
        from referencing import Registry, Resource  # type: ignore
        from referencing.jsonschema import DRAFT202012  # type: ignore

        resources = []
        for sid, schema in store.items():
            resources.append((sid, Resource.from_contents(schema, default_specification=DRAFT202012)))
        registry = Registry().with_resources(resources)

        for path in files:
            schema = store[_load_json(path)["$id"]]
            examples = schema.get("examples")
            if not (isinstance(examples, list) and examples):
                continue

            validator = V202012(schema, registry=registry)
            for i, ex in enumerate(examples):
                try:
                    validator.validate(ex)
                except jsonschema.ValidationError as e:
                    pytest.fail(f"{path}: example[{i}] does not validate: {e}")
        return

    except Exception:
        # Fallback: deprecated RefResolver (works without 'referencing' import)
        try:
            from jsonschema import RefResolver  # type: ignore
        except Exception as e:
            pytest.fail(
                "Examples validation requires either 'referencing' (preferred) or jsonschema.RefResolver. "
                f"Neither is available: {e}"
            )
            return

        for path in files:
            schema = _load_json(path)
            examples = schema.get("examples")
            if not (isinstance(examples, list) and examples):
                continue

            resolver = RefResolver.from_schema(schema, store=store)  # type: ignore[arg-type]
            validator = V202012(schema, resolver=resolver)
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
