# SPDX-License-Identifier: Apache-2.0
"""
Schema Registry (Draft 2020-12)

- Recursively loads all JSON Schemas under ./schemas
- Indexes by $id and by relative file path
- Exposes:
    get_validator(schema_id: str) -> jsonschema.Validator
    validate_json(schema_id: str, obj: Any) -> None (raises ValidationError)
    assert_valid(schema_id: str, obj: Any) -> None (pytest-friendly)
- Strict Draft 2020-12 with $id-based resolution
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import jsonschema
from jsonschema import Draft202012Validator
from jsonschema.exceptions import RefResolutionError, ValidationError


_SCHEMAS_ROOT_ENV = "CORPUS_SCHEMAS_ROOT"
_DEFAULT_SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "schemas"

# Global, thread-safe singletons
_STORE_LOCK = threading.RLock()
_SCHEMA_STORE: Dict[str, dict] = {}
_VALIDATOR_CACHE: Dict[str, Draft202012Validator] = {}


def _schemas_root() -> Path:
    # Allow override for monorepo / workspace setups
    root = os.environ.get(_SCHEMAS_ROOT_ENV)
    return Path(root).resolve() if root else _DEFAULT_SCHEMAS_DIR


def _iter_schema_files(root: Path):
    for p in root.rglob("*.json"):
        # Skip obvious non-schema files if any
        yield p


def _load_all_schemas() -> None:
    with _STORE_LOCK:
        if _SCHEMA_STORE:
            return

        root = _schemas_root()
        if not root.exists():
            raise RuntimeError(f"Schemas root not found: {root}")

        for file_path in _iter_schema_files(root):
            try:
                with file_path.open("r", encoding="utf-8") as f:
                    schema = json.load(f)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON in schema: {file_path}: {e}") from e

            schema_id = schema.get("$id")
            if not schema_id:
                # Allow local refs by relative path too (useful for debugging)
                schema_id = file_path.as_uri()

            if schema_id in _SCHEMA_STORE:
                raise RuntimeError(
                    f"Duplicate schema $id detected:\n"
                    f"  {schema_id}\nFirst: {_SCHEMA_STORE[schema_id].get('__file__')}\n"
                    f"Second: {file_path}"
                )

            # Attach virtual metadata for debugging
            schema["__file__"] = str(file_path)
            _SCHEMA_STORE[schema_id] = schema


def _build_resolver() -> jsonschema.RefResolver:
    # jsonschema 4.x: in-Draft resolver creation with store allows $id lookup
    return jsonschema.RefResolver.from_schema(
        {"$id": "urn:__root__", "$schema": "https://json-schema.org/draft/2020-12/schema"},
        store=_SCHEMA_STORE,
    )


def _make_validator(schema_id: str) -> Draft202012Validator:
    with _STORE_LOCK:
        if not _SCHEMA_STORE:
            _load_all_schemas()

        if schema_id not in _SCHEMA_STORE:
            # Try to be helpful: list close matches
            keys = list(_SCHEMA_STORE.keys())
            suggestion = next((k for k in keys if schema_id in k or k.endswith(schema_id)), None)
            hint = f"\nDid you mean: {suggestion}" if suggestion else ""
            raise KeyError(f"Schema not found by $id: {schema_id}{hint}")

        schema = _SCHEMA_STORE[schema_id]
        resolver = _build_resolver()
        try:
            validator = Draft202012Validator(schema, resolver=resolver, format_checker=jsonschema.draft202012_format_checker)
        except RefResolutionError as e:
            raise RuntimeError(f"Failed to resolve $refs for schema '{schema_id}': {e}") from e

        return validator


def get_validator(schema_id: str) -> Draft202012Validator:
    """
    Return a cached Draft202012 validator for the given $id.
    """
    with _STORE_LOCK:
        v = _VALIDATOR_CACHE.get(schema_id)
        if v is not None:
            return v
        v = _make_validator(schema_id)
        _VALIDATOR_CACHE[schema_id] = v
        return v


def validate_json(schema_id: str, obj: Any) -> None:
    """
    Validate an object against the schema identified by $id.
    Raises jsonschema.ValidationError on failure.
    """
    validator = get_validator(schema_id)
    validator.validate(obj)


# PyTest-friendly assert helper (richer message)
def assert_valid(schema_id: str, obj: Any, *, context: Optional[str] = None) -> None:
    try:
        validate_json(schema_id, obj)
    except ValidationError as e:
        location = f"\ncontext: {context}" if context else ""
        source = ""
        try:
            source = f"\nschema_file: {_SCHEMA_STORE[schema_id].get('__file__')}"
        except Exception:
            pass
        raise AssertionError(f"JSON did not validate against {schema_id}:{location}{source}\n{e.message}\n\nPath: {list(e.path)}\nSchemaPath: {list(e.schema_path)}") from e


# Optional CLI for local debugging:
if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Validate JSON against a registered schema $id")
    parser.add_argument("schema_id", help="Schema $id (exact) or suffix")
    parser.add_argument("json_file", help="Path to a JSON document to validate")
    args = parser.parse_args()

    _load_all_schemas()

    # Allow suffix match for convenience
    key = args.schema_id
    if key not in _SCHEMA_STORE:
        for k in _SCHEMA_STORE:
            if k.endswith(args.schema_id):
                key = k
                break

    with open(args.json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        validate_json(key, data)
        print(f"OK: {args.json_file} validates against {key}")
        sys.exit(0)
    except ValidationError as e:
        print(f"ERROR: validation failed against {key}\n{e}", file=sys.stderr)
        sys.exit(1)
