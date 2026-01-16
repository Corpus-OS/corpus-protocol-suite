# SPDX-License-Identifier: Apache-2.0
"""
Schema Registry (Draft 2020-12)

- Recursively loads all JSON Schemas under ./schemas (preferred) or ./schema (fallback)
- Indexes by $id and by relative file path
- Exposes:
    get_validator(schema_id: str) -> jsonschema.Validator
    validate_json(schema_id: str, obj: Any) -> None (raises ValidationError)
    assert_valid(schema_id: str, obj: Any) -> None (pytest-friendly)
- Strict Draft 2020-12 with $id-based resolution

Resolution model:
- Uses `referencing.Registry` (Draft 2020-12) for $id-based resolution (no deprecated RefResolver).

Notes on configuration:
- CORPUS_SCHEMAS_ROOT (env var) is supported as an override.
- For tests/DI, you can also pass an explicit schemas_root Path WITHOUT mutating os.environ.
  (This avoids process-wide side effects.)

Also exposes a SchemaRegistry class suitable for dependency injection
(e.g. stream validators) while still sharing the underlying store by default.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import jsonschema
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012


_SCHEMAS_ROOT_ENV = "CORPUS_SCHEMAS_ROOT"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SCHEMAS_DIR = (_REPO_ROOT / "schemas") if (_REPO_ROOT / "schemas").exists() else (_REPO_ROOT / "schema")

# Global, thread-safe singletons with loaded state tracking
_STORE_LOCK = threading.RLock()
_SCHEMA_STORE: Dict[str, dict] = {}
_VALIDATOR_CACHE: Dict[str, Draft202012Validator] = {}
_SCHEMA_PATHS: Dict[str, str] = {}
_REGISTRY: Optional[Registry] = None
_LOADED = False  # Track loading state explicitly

# Explicit override root (preferred over env var). Does NOT mutate os.environ.
_SCHEMAS_ROOT_OVERRIDE: Optional[Path] = None


def _set_schemas_root_override(root: Optional[Path]) -> None:
    """
    Set an explicit schemas root override for this process without mutating os.environ.

    If schemas were already loaded and the override changes, this will clear caches so
    subsequent loads come from the new root.
    """
    global _SCHEMAS_ROOT_OVERRIDE, _LOADED
    with _STORE_LOCK:
        new_root = root.resolve() if root is not None else None

        if new_root is not None and not new_root.exists():
            raise RuntimeError(f"Schema root override not found: {new_root}")

        if _SCHEMAS_ROOT_OVERRIDE == new_root:
            return

        # If we've already loaded schemas and the root changes, reset caches/state.
        if _LOADED:
            clear_cache()

        _SCHEMAS_ROOT_OVERRIDE = new_root


def _schemas_root() -> Path:
    """Get schemas root directory with override and environment variable support."""
    if _SCHEMAS_ROOT_OVERRIDE is not None:
        return _SCHEMAS_ROOT_OVERRIDE

    root = os.environ.get(_SCHEMAS_ROOT_ENV)
    if root:
        path = Path(root).resolve()
        if not path.exists():
            raise RuntimeError(f"Schema root from {_SCHEMAS_ROOT_ENV} not found: {path}")
        return path

    return _DEFAULT_SCHEMAS_DIR


def _iter_schema_files(root: Path) -> List[Path]:
    """Find all JSON schema files, skipping obvious non-schema files."""
    schema_files: List[Path] = []
    for p in root.rglob("*.json"):
        # Skip common non-schema files and build directories
        if p.name in {"package.json", "tsconfig.json", "package-lock.json", "composer.json"}:
            continue
        # Skip common exclusion directories
        if any(part in p.parts for part in {"node_modules", ".git", "dist", "build", "__pycache__"}):
            continue
        schema_files.append(p)
    return sorted(schema_files)  # Deterministic order for testing


def _validate_schema_metadata(schema: dict, file_path: Path) -> None:
    """Validate basic schema structure and metadata."""
    if not isinstance(schema, dict):
        raise ValueError(f"Schema must be a JSON object: {file_path}")

    # Skip if not a JSON Schema (no $schema)
    if "$schema" not in schema:
        return

    # Check for Draft 2020-12 compliance
    schema_version = schema.get("$schema")
    if schema_version != "https://json-schema.org/draft/2020-12/schema":
        raise ValueError(
            f"Schema must declare $schema as Draft 2020-12: {file_path} (got {schema_version})"
        )

    if "$id" not in schema:
        raise ValueError(f"Schema must declare $id: {file_path}")

    schema_id = schema["$id"]
    if not isinstance(schema_id, str):
        raise ValueError(f"Schema $id must be a string: {file_path}")

    # Validate URI format more thoroughly (SCHEMA.md uses https://...; allow urn/tag for tooling)
    if not (schema_id.startswith(("http://", "https://", "urn:", "tag:"))):
        raise ValueError(
            f"Schema $id must be a valid URI (http/https/urn/tag): {file_path} -> {schema_id}"
        )


def _build_registry_from_store(store: Dict[str, dict]) -> Registry:
    """
    Build a Draft 2020-12 referencing.Registry from the schema store.
    This is the correct resolution mechanism for jsonschema Draft 2020-12.
    """
    resources = []
    for sid, schema in store.items():
        # Keep schemas pristine (no injected metadata keys).
        resources.append((sid, Resource.from_contents(schema, default_specification=DRAFT202012)))
    return Registry().with_resources(resources)


def _load_all_schemas() -> None:
    """Load all schemas into the global store with comprehensive validation."""
    global _LOADED, _REGISTRY

    with _STORE_LOCK:
        if _LOADED:
            return

        root = _schemas_root()
        if not root.exists():
            raise RuntimeError(f"Schemas root not found: {root}")

        loaded_files: List[str] = []
        duplicate_ids: List[str] = []
        schema_count = 0

        # Track duplicates with file context
        seen_ids: Dict[str, Path] = {}

        for file_path in _iter_schema_files(root):
            try:
                with file_path.open("r", encoding="utf-8") as f:
                    schema = json.load(f)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON in schema {file_path}: {e}") from e

            # Skip if not a JSON Schema
            if not isinstance(schema, dict) or "$schema" not in schema:
                continue

            # Validate schema structure
            try:
                _validate_schema_metadata(schema, file_path)
            except ValueError as e:
                raise RuntimeError(str(e)) from e

            schema_id = schema["$id"]

            # Check for duplicates with helpful error
            if schema_id in seen_ids:
                duplicate_ids.append(
                    f"  {schema_id}\n    First:  {seen_ids[schema_id]}\n    Second: {file_path}"
                )
                continue

            seen_ids[schema_id] = file_path

            # Store schema (pristine) and separate metadata mappings
            _SCHEMA_STORE[schema_id] = schema
            _SCHEMA_PATHS[schema_id] = str(file_path)
            loaded_files.append(file_path.name)
            schema_count += 1

        if duplicate_ids:
            raise RuntimeError("Duplicate schema $id detected:\n" + "\n".join(duplicate_ids))

        # Build the referencing registry once, after store is complete
        _REGISTRY = _build_registry_from_store(_SCHEMA_STORE)

        if schema_count > 0:
            print(f"✅ Loaded {schema_count} schemas from {root}: {', '.join(sorted(loaded_files))}")
        else:
            print("⚠️  No JSON schemas found - check schema root configuration")

        _LOADED = True


def _make_validator(schema_id: str) -> Draft202012Validator:
    """Create a validator for the given schema ID with comprehensive error handling."""
    with _STORE_LOCK:
        if not _LOADED:
            _load_all_schemas()

        if schema_id not in _SCHEMA_STORE:
            available_ids = list(_SCHEMA_STORE.keys())

            if not available_ids:
                raise KeyError(
                    f"Schema not found: {schema_id}\nNo schemas loaded. "
                    f"Check {_SCHEMAS_ROOT_ENV} or pass schemas_root explicitly."
                )

            suggestions: List[str] = []

            # Exact suffix or substring matching
            for available_id in available_ids:
                if available_id.endswith(schema_id) or schema_id in available_id:
                    suggestions.append(available_id)

            # Also try filename matching
            schema_filename = Path(schema_id).name
            for available_id in available_ids:
                if Path(_SCHEMA_PATHS[available_id]).name == schema_filename:
                    if available_id not in suggestions:
                        suggestions.append(available_id)

            error_msg = f"Schema not found: {schema_id}"
            if suggestions:
                error_msg += "\nDid you mean one of:\n  " + "\n  ".join(sorted(suggestions)[:5])
            else:
                error_msg += (
                    f"\nAvailable schemas ({len(available_ids)}):\n  "
                    + "\n  ".join(sorted(available_ids)[:8])
                )
                if len(available_ids) > 8:
                    error_msg += f"\n  ... and {len(available_ids) - 8} more"

            raise KeyError(error_msg)

        schema = _SCHEMA_STORE[schema_id]
        schema_path = _SCHEMA_PATHS.get(schema_id, "unknown")

        if _REGISTRY is None:
            # Should not happen if _load_all_schemas ran, but be defensive.
            _REGISTRY = _build_registry_from_store(_SCHEMA_STORE)

        try:
            # Validate the schema itself first
            Draft202012Validator.check_schema(schema)

            # Create validator with Draft 2020-12 registry-based resolution + format checking
            validator = Draft202012Validator(
                schema,
                registry=_REGISTRY,
                format_checker=jsonschema.draft202012_format_checker,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to create validator for schema '{schema_id}' ({schema_path}): {e}"
            ) from e

        return validator


def get_validator(schema_id: str) -> Draft202012Validator:
    """Return a cached Draft202012 validator for the given $id."""
    with _STORE_LOCK:
        if schema_id in _VALIDATOR_CACHE:
            return _VALIDATOR_CACHE[schema_id]

        validator = _make_validator(schema_id)
        _VALIDATOR_CACHE[schema_id] = validator
        return validator


def validate_json(schema_id: str, obj: Any) -> None:
    """Validate an object against the schema identified by $id."""
    validator = get_validator(schema_id)
    validator.validate(obj)


def _format_value(value: Any, max_length: int = 200) -> str:
    """Format value for error messages with length limiting."""
    formatted = repr(value)
    if len(formatted) > max_length:
        return formatted[:max_length] + "..."
    return formatted


class SchemaRegistry:
    """
    Lightweight registry wrapper for dependency injection.

    By default it delegates to the module-level global registry (same store/caches).
    You can provide schemas_root to load schemas from a specific directory WITHOUT
    mutating environment variables.
    """

    def __init__(self, *, schemas_root: Optional[Path] = None) -> None:
        if schemas_root is not None:
            _set_schemas_root_override(schemas_root)
        _load_all_schemas()

    def get_validator(self, schema_id: str) -> Draft202012Validator:
        return get_validator(schema_id)

    def validate_json(self, schema_id: str, obj: Any) -> None:
        validate_json(schema_id, obj)

    def assert_valid(
        self,
        schema_id: str,
        obj: Any,
        *,
        context: Optional[str] = None,
    ) -> None:
        assert_valid(schema_id, obj, context=context, registry=self)

    @property
    def schema_paths(self) -> Dict[str, str]:
        _load_all_schemas()
        with _STORE_LOCK:
            return _SCHEMA_PATHS.copy()

    def list_schemas(self) -> Dict[str, str]:
        return list_schemas()

    def get_schema(self, schema_id: str) -> dict:
        return get_schema(schema_id)


def assert_valid(
    schema_id: str,
    obj: Any,
    *,
    context: Optional[str] = None,
    registry: Optional[SchemaRegistry] = None,
) -> None:
    """
    Pytest-friendly validation with rich error messages.
    Raises AssertionError on validation failure.
    """
    try:
        if registry is None:
            validate_json(schema_id, obj)
        else:
            registry.validate_json(schema_id, obj)
    except ValidationError as e:
        # Prefer registry's paths if present, else fall back to global mapping
        if registry is not None:
            try:
                paths = registry.schema_paths
            except Exception:
                paths = _SCHEMA_PATHS
        else:
            paths = _SCHEMA_PATHS

        error_parts = [
            f"JSON validation failed against {schema_id}",
            f"Schema file: {paths.get(schema_id, 'unknown')}",
            f"Error: {e.message}",
            (
                f"Failing value: {_format_value(e.instance)}"
                if e.instance is not None
                else "Failing value: <none>"
            ),
            f"JSON path: {'.'.join(str(p) for p in e.path) if e.path else '<root>'}",
            f"Schema path: {'.'.join(str(p) for p in e.schema_path) if e.schema_path else '<root>'}",
        ]

        if context:
            error_parts.insert(1, f"Context: {context}")

        raise AssertionError("\n".join(error_parts)) from e


def preload_all_schemas(*, schemas_root: Optional[Path] = None) -> None:
    """
    Force preload all schemas and validate the entire registry.

    Optionally provide schemas_root WITHOUT mutating os.environ.
    """
    if schemas_root is not None:
        _set_schemas_root_override(schemas_root)
    _load_all_schemas()

    with _STORE_LOCK:
        for schema_id in list(_SCHEMA_STORE.keys()):
            try:
                get_validator(schema_id)
            except Exception as e:
                file_path = _SCHEMA_PATHS.get(schema_id, "unknown")
                raise RuntimeError(
                    f"Failed to create validator for {schema_id} ({file_path}): {e}"
                ) from e


def load_all_schemas_into_registry(schemas_root: Optional[Path] = None) -> None:
    """
    Test helper used by golden/schema tests.

    If schemas_root is provided, it will:
      • load from that directory (without setting env vars)
      • preload and validate all schemas
    """
    preload_all_schemas(schemas_root=schemas_root)


def list_schemas() -> Dict[str, str]:
    """Get mapping of all schema IDs to their file paths."""
    _load_all_schemas()
    with _STORE_LOCK:
        return _SCHEMA_PATHS.copy()


def get_schema(schema_id: str) -> dict:
    """Get the raw schema document by $id (copy to prevent mutation)."""
    _load_all_schemas()
    with _STORE_LOCK:
        if schema_id not in _SCHEMA_STORE:
            raise KeyError(f"Schema not found: {schema_id}")
        return _SCHEMA_STORE[schema_id].copy()


def clear_cache() -> None:
    """Clear all caches/state (primarily for testing)."""
    global _LOADED, _REGISTRY, _SCHEMAS_ROOT_OVERRIDE
    with _STORE_LOCK:
        _SCHEMA_STORE.clear()
        _VALIDATOR_CACHE.clear()
        _SCHEMA_PATHS.clear()
        _REGISTRY = None
        _LOADED = False
        _SCHEMAS_ROOT_OVERRIDE = None


# Command-line interface for validation and debugging
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Validate JSON against a registered schema $id",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  %(prog)s llm.envelope.request.json sample_request.json
  %(prog)s https://corpusos.com/schemas/llm/llm.envelope.request.json sample_request.json

Environment:
  {_SCHEMAS_ROOT_ENV}    Override the default schemas directory
        """,
    )
    parser.add_argument("schema_id", nargs="?", help="Schema $id (exact) or file name suffix")
    parser.add_argument("json_file", nargs="?", help="Path to JSON document to validate")
    parser.add_argument("--list", action="store_true", help="List all available schemas")
    parser.add_argument("--preload", action="store_true", help="Preload and validate all schemas")
    parser.add_argument("--stats", action="store_true", help="Show schema registry statistics")

    args = parser.parse_args()

    try:
        if args.list:
            preload_all_schemas()
            schemas = list_schemas()
            print(f"Available schemas ({len(schemas)}):")
            root = _schemas_root()
            for schema_id, file_path in sorted(schemas.items()):
                p = Path(file_path)
                try:
                    rel = p.relative_to(root)
                except ValueError:
                    rel = p
                print(f"  {schema_id} -> {rel}")
            sys.exit(0)

        if args.preload:
            preload_all_schemas()
            print("✅ All schemas loaded and validated successfully")
            sys.exit(0)

        if args.stats:
            preload_all_schemas()
            print("Schema Registry Statistics:")
            print(f"  Schemas loaded: {len(_SCHEMA_STORE)}")
            print(f"  Validators cached: {len(_VALIDATOR_CACHE)}")
            print(f"  Schema root: {_schemas_root()}")
            sys.exit(0)

        if not args.schema_id or not args.json_file:
            parser.print_help()
            sys.exit(1)

        preload_all_schemas()

        target_id = args.schema_id
        if target_id not in _SCHEMA_STORE:
            # Try suffix matching
            matches = [sid for sid in _SCHEMA_STORE.keys() if sid.endswith(target_id)]
            if not matches:
                # Try filename matching
                schema_filename = Path(target_id).name
                matches = [sid for sid, path in _SCHEMA_PATHS.items() if Path(path).name == schema_filename]

            if len(matches) == 1:
                target_id = matches[0]
                print(f"🔍 Using schema: {target_id}")
            elif len(matches) > 1:
                print(f"❌ Multiple schemas match '{args.schema_id}':")
                for match in sorted(matches):
                    print(f"   {match}")
                sys.exit(1)
            else:
                print(f"❌ Schema not found: {args.schema_id}")
                print("   Use --list to see available schemas")
                sys.exit(1)

        with open(args.json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        validate_json(target_id, data)
        print(f"✅ {args.json_file} validates against {target_id}")
        sys.exit(0)

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
