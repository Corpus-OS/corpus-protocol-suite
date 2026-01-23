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
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jsonschema
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

logger = logging.getLogger(__name__)

_SCHEMAS_ROOT_ENV = "CORPUS_SCHEMAS_ROOT"
_ALLOW_NON_SCHEMA_JSON_ENV = "CORPUS_ALLOW_NON_SCHEMA_JSON"

# SCHEMA.md version tolerance fragment prefix (convention):
# schema_id#version/<semver>
_VERSION_FRAGMENT_PREFIX = "#version/"

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


def _allow_non_schema_json() -> bool:
    """
    Strict posture per SCHEMA.md: schema root should contain schemas, not arbitrary JSON.

    If you must allow non-schema JSON under schema root (legacy tooling, transitional states),
    set CORPUS_ALLOW_NON_SCHEMA_JSON=true to skip those files rather than failing.
    """
    v = os.environ.get(_ALLOW_NON_SCHEMA_JSON_ENV, "false").strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def _split_schema_id(schema_id: str) -> Tuple[str, Optional[str]]:
    """
    Split schema_id into (base_id, fragment).

    Returns:
      - base_id: everything before '#'
      - fragment: '#...' including leading '#', or None
    """
    s = (schema_id or "").strip()
    if "#" not in s:
        return s, None
    base, frag = s.split("#", 1)
    return base, "#" + frag


def _base_schema_id(schema_id: str) -> str:
    """
    Normalize a schema_id for store lookup and caching.

    SCHEMA.md version tolerance uses schema_id#version/<semver>. The on-disk schemas'
    $id values do NOT include these fragments (per SCHEMA.md $id convention), so we
    always look up by the base $id (fragment stripped).

    We still keep the original schema_id for error/context reporting.
    """
    base, _frag = _split_schema_id(schema_id)
    return base


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
    """Find all JSON files under schema root, skipping obvious non-schema files."""
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
    """Validate basic schema structure and metadata per SCHEMA.md."""
    if not isinstance(schema, dict):
        raise ValueError(f"Schema must be a JSON object: {file_path}")

    # $schema is required for schemas and must be Draft 2020-12
    schema_version = schema.get("$schema")
    if schema_version != "https://json-schema.org/draft/2020-12/schema":
        raise ValueError(
            f"Schema must declare $schema as Draft 2020-12: {file_path} (got {schema_version!r})"
        )

    if "$id" not in schema:
        raise ValueError(f"Schema must declare $id: {file_path}")

    schema_id = schema["$id"]
    if not isinstance(schema_id, str) or not schema_id.strip():
        raise ValueError(f"Schema $id must be a non-empty string: {file_path}")

    # SCHEMA.md $id convention is https://corpusos.com/schemas/{component}/{filename}
    # Keep it strict for conformance tooling: require http(s) + "/schemas/" namespace.
    if not schema_id.startswith(("http://", "https://")):
        raise ValueError(f"Schema $id must be an http(s) URI: {file_path} -> {schema_id!r}")
    if "/schemas/" not in schema_id:
        raise ValueError(f"Schema $id must include '/schemas/' namespace: {file_path} -> {schema_id!r}")


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

        loaded_relpaths: List[str] = []
        duplicate_ids: List[str] = []
        schema_count = 0

        # Track duplicates with file context
        seen_ids: Dict[str, Path] = {}

        allow_non_schema = _allow_non_schema_json()

        for file_path in _iter_schema_files(root):
            try:
                with file_path.open("r", encoding="utf-8") as f:
                    doc = json.load(f)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON in schema file {file_path}: {e}") from e

            # In strict mode, anything under schema root should be a Draft 2020-12 schema object.
            if not isinstance(doc, dict) or "$schema" not in doc:
                if allow_non_schema:
                    logger.debug("Skipping non-schema JSON under schema root: %s", file_path)
                    continue
                raise RuntimeError(
                    f"Non-schema JSON found under schema root (missing Draft 2020-12 $schema): {file_path}"
                )

            schema = doc

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

            try:
                rel = file_path.relative_to(root)
                loaded_relpaths.append(rel.as_posix())
            except ValueError:
                loaded_relpaths.append(file_path.as_posix())

            schema_count += 1

        if duplicate_ids:
            raise RuntimeError("Duplicate schema $id detected:\n" + "\n".join(duplicate_ids))

        if schema_count == 0:
            # Strict posture: schema infra must load schemas.
            raise RuntimeError(
                f"No JSON Schemas found under {root}. "
                f"Check {_SCHEMAS_ROOT_ENV} or schema root layout."
            )

        # Build the referencing registry once, after store is complete
        _REGISTRY = _build_registry_from_store(_SCHEMA_STORE)

        logger.info("Loaded %d schemas from %s", schema_count, root)
        logger.debug("Loaded schema files: %s", ", ".join(sorted(loaded_relpaths)))

        _LOADED = True


def _make_validator(schema_id: str) -> Draft202012Validator:
    """Create a validator for the given schema ID with comprehensive error handling."""
    global _REGISTRY  # FIX: Declare global to avoid UnboundLocalError
    
    with _STORE_LOCK:
        if not _LOADED:
            _load_all_schemas()

        requested_id = (schema_id or "").strip()
        base_id = _base_schema_id(requested_id)

        if base_id not in _SCHEMA_STORE:
            available_ids = list(_SCHEMA_STORE.keys())

            if not available_ids:
                raise KeyError(
                    f"Schema not found: {requested_id}\nNo schemas loaded. "
                    f"Check {_SCHEMAS_ROOT_ENV} or pass schemas_root explicitly."
                )

            suggestions: List[str] = []

            # Exact suffix or substring matching (against base IDs)
            for available_id in available_ids:
                if available_id.endswith(base_id) or base_id in available_id:
                    suggestions.append(available_id)

            # Also try filename matching
            schema_filename = Path(base_id).name
            for available_id in available_ids:
                ap = _SCHEMA_PATHS.get(available_id)
                if ap and Path(ap).name == schema_filename:
                    if available_id not in suggestions:
                        suggestions.append(available_id)

            error_msg = f"Schema not found: {requested_id}"
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

        schema = _SCHEMA_STORE[base_id]
        schema_path = _SCHEMA_PATHS.get(base_id, "unknown")

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
                f"Failed to create validator for schema '{requested_id}' ({schema_path}): {e}"
            ) from e

        return validator


def get_validator(schema_id: str) -> Draft202012Validator:
    """
    Return a cached Draft202012 validator for the given $id.

    Caching is keyed by the base schema $id (fragments stripped), so callers may pass
    SCHEMA.md version-tolerant IDs like:
      https://.../llm/llm.complete.request.json#version/1.0.0
    """
    base_id = _base_schema_id(schema_id)
    with _STORE_LOCK:
        if base_id in _VALIDATOR_CACHE:
            return _VALIDATOR_CACHE[base_id]

        validator = _make_validator(schema_id)
        _VALIDATOR_CACHE[base_id] = validator
        return validator


def validate_json(schema_id: str, obj: Any) -> None:
    """Validate an object against the schema identified by $id (version fragments tolerated)."""
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
    requested_id = (schema_id or "").strip()
    base_id = _base_schema_id(requested_id)

    try:
        if registry is None:
            validate_json(requested_id, obj)
        else:
            registry.validate_json(requested_id, obj)
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
            f"JSON validation failed against {requested_id}",
            f"Schema file: {paths.get(base_id, 'unknown')}",
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
    """Get mapping of all base schema IDs to their file paths."""
    _load_all_schemas()
    with _STORE_LOCK:
        return _SCHEMA_PATHS.copy()


def get_schema(schema_id: str) -> dict:
    """
    Get the raw schema document by $id (copy to prevent mutation).

    Version fragments (e.g., #version/1.0.0) are tolerated and resolved to the base schema $id.
    """
    _load_all_schemas()
    requested_id = (schema_id or "").strip()
    base_id = _base_schema_id(requested_id)
    with _STORE_LOCK:
        if base_id not in _SCHEMA_STORE:
            raise KeyError(f"Schema not found: {requested_id}")
        return _SCHEMA_STORE[base_id].copy()


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
  %(prog)s https://corpusos.com/schemas/llm/llm.complete.request.json#version/1.0.0 sample_request.json

Environment:
  {_SCHEMAS_ROOT_ENV}                 Override the default schemas directory
  {_ALLOW_NON_SCHEMA_JSON_ENV}        If true, skip non-schema JSON under schema root (default false)
        """,
    )
    parser.add_argument("schema_id", nargs="?", help="Schema $id (exact), versioned id, or file name suffix")
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
            print("✅ All schemas loaded and validators created successfully")
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

        requested = args.schema_id.strip()
        base_requested = _base_schema_id(requested)

        target_id = requested

        # If user passed a filename suffix (or base id not found), try matching.
        if base_requested not in _SCHEMA_STORE:
            # Try suffix matching against base IDs
            matches = [sid for sid in _SCHEMA_STORE.keys() if sid.endswith(base_requested)]
            if not matches:
                # Try filename matching
                schema_filename = Path(base_requested).name
                matches = [
                    sid
                    for sid, path in _SCHEMA_PATHS.items()
                    if Path(path).name == schema_filename
                ]

            if len(matches) == 1:
                # Preserve the fragment the user supplied (if any) by re-attaching it.
                _base, frag = _split_schema_id(requested)
                target_id = matches[0] + (frag or "")
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
