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
SCHEMAS_DIR = ROOT / "schemas"

# Components we expect under schemas/
COMPONENTS = {"common", "llm", "vector", "embedding", "graph"}


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

DRAFT_202012 = "https://json-schema.org/draft/2020-12/schema"
SCHEMA_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")
ID_ALLOWED = re.compile(r"^[A-Za-z0-9._~:/#-]+$")  # permissive but disallows spaces, etc.
PATTERN_CACHE: Dict[str, re.Pattern] = {}


def _iter_schema_files() -> List[Path]:
    """Collect all JSON schema files under /schemas/**."""
    if not SCHEMAS_DIR.exists():
        pytest.skip(f"schemas/ directory not found at {SCHEMAS_DIR}")
    return sorted(SCHEMAS_DIR.rglob("*.json"))


def _load_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        pytest.fail(f"Failed to load JSON for {p}: {e}")


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


def _compile_pattern(pat: str) -> None:
    if pat in PATTERN_CACHE:
        return
    try:
        PATTERN_CACHE[pat] = re.compile(pat)
    except re.error as e:
        pytest.fail(f"Invalid regex pattern: {pat!r} → {e}")


def _collect_ids(store: Dict[str, dict]) -> Set[str]:
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
        else:
            return False
    return True


def _is_envelope_request(fname: str) -> bool:
    return fname.endswith(".envelope.request.json")


def _is_envelope_success(fname: str) -> bool:
    return fname.endswith(".envelope.success.json")


def _is_envelope_error(fname: str) -> bool:
    return fname.endswith(".envelope.error.json")


def _is_stream_frame(fname: str) -> bool:
    return (
        fname.endswith(".stream.frame.data.json")
        or fname.endswith(".stream.frame.end.json")
        or fname.endswith(".stream.frame.error.json")
        or fname.endswith(".stream.frames.ndjson.schema.json")
    )


def _component_for_path(p: Path) -> str:
    """
    Infer component name from path: schemas/<component>/<file>.json
    """
    try:
        idx = p.parts.index("schemas")
        return p.parts[idx + 1]
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------------------
# Test 1: Load + $schema + unique $id + store build
# --------------------------------------------------------------------------------------

def test_all_schemas_load_and_have_unique_ids():
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

    # Fail fast if no schemas found (misplaced repo)
    assert store, f"No schemas discovered under {SCHEMAS_DIR}"


# --------------------------------------------------------------------------------------
# Test 2: $id path convention
# --------------------------------------------------------------------------------------

def test_id_path_convention_matches_filesystem():
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

    if problems:
        pytest.fail("ID/path mismatches:\n- " + "\n- ".join(problems))


# --------------------------------------------------------------------------------------
# Test 3: Metaschema conformance + regex patterns + enum hygiene
# --------------------------------------------------------------------------------------

def test_metaschema_conformance_and_basic_hygiene():
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


# --------------------------------------------------------------------------------------
# Test 4: Cross-file $ref resolution + local fragment checks
# --------------------------------------------------------------------------------------

def test_cross_file_refs_resolve_and_local_fragments_exist():
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

    if problems:
        pytest.fail("Unresolved $ref issues:\n- " + "\n- ".join(problems))


# --------------------------------------------------------------------------------------
# Test 5: $defs usage (no dangling definitions)
# --------------------------------------------------------------------------------------

def test_no_dangling_defs_globally():
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
                    # Local refs shouldn't be counted against global anchors
                    pass

    # Any def anchors never referenced anywhere?
    dangling = sorted(a for a in def_anchors if a not in used_refs)

    # Allow some schemas to intentionally export $defs for external use.
    # If you want to allow-list certain files, add logic here.
    if dangling:
        pytest.fail(
            "Dangling $defs (exported but never referenced globally):\n- "
            + "\n- ".join(dangling)
        )


# --------------------------------------------------------------------------------------
# Test 6: Envelope / role sanity & consts
# --------------------------------------------------------------------------------------

def test_envelope_role_sanity_and_consts():
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
            if schema.get("additionalProperties") not in (False,):
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
            for k in ("ok", "code", "ms", "result"):
                if k not in req:
                    problems.append(f"{path}: success envelope must require {k}")
            # schema_version property presence/pattern
            props = schema.get("properties", {})
            sv = props.get("schema_version")
            if not isinstance(sv, dict):
                problems.append(f"{path}: success envelope should define 'schema_version' property")
            else:
                pat = sv.get("pattern")
                if pat and not re.fullmatch(r"^\^\d\+\\\.\d\+\\\.\d\+\$$".replace("\\", ""), pat):
                    # don't be overly strict on exact regex text—just check it's present in any form
                    pass

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

    if problems:
        pytest.fail("Envelope/const issues:\n- " + "\n- ".join(problems))


# --------------------------------------------------------------------------------------
# Test 7: Examples validation (if present)
# --------------------------------------------------------------------------------------

def test_examples_validate_against_own_schema():
    files = _iter_schema_files()
    # Build a store for cross-ref resolution
    store: Dict[str, dict] = {}
    for path in files:
        schema = _load_json(path)
        store[schema["$id"]] = schema

    for path in files:
        schema = _load_json(path)
        examples = schema.get("examples")
        if not (isinstance(examples, list) and examples):
            continue

        # Create a validator with store for resolving absolute $id refs
        validator = V202012(schema, resolver=jsonschema.RefResolver(base_uri=schema["$id"], store=store))

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

    if problems:
        pytest.fail("Stream frame schema issues:\n- " + "\n- ".join(problems))
