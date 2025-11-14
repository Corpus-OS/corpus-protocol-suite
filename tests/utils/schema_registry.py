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
from typing import Any, Dict, List, Optional, Tuple

import jsonschema
from jsonschema import Draft202012Validator
from jsonschema.exceptions import RefResolutionError, ValidationError


_SCHEMAS_ROOT_ENV = "CORPUS_SCHEMAS_ROOT"
_DEFAULT_SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "schemas"

# Global, thread-safe singletons
_STORE_LOCK = threading.RLock()
_SCHEMA_STORE: Dict[str, dict] = {}
_VALIDATOR_CACHE: Dict[str, Draft202012Validator] = {}
_SCHEMA_PATHS: Dict[str, str] = {}


def _schemas_root() -> Path:
    """Get schemas root directory with environment variable override support."""
    root = os.environ.get(_SCHEMAS_ROOT_ENV)
    return Path(root).resolve() if root else _DEFAULT_SCHEMAS_DIR


def _iter_schema_files(root: Path) -> List[Path]:
    """Find all JSON schema files, skipping obvious non-schema files."""
    schema_files = []
    for p in root.rglob("*.json"):
        # Skip package.json, tsconfig.json, etc. if they exist
        if p.name in {"package.json", "tsconfig.json", "package-lock.json"}:
            continue
        schema_files.append(p)
    return sorted(schema_files)  # Deterministic order for testing


def _validate_schema_metadata(schema: dict, file_path: Path) -> None:
    """Validate basic schema structure and metadata."""
    if not isinstance(schema, dict):
        raise ValueError(f"Schema must be a JSON object: {file_path}")
    
    # Check for Draft 2020-12 compliance
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise ValueError(f"Schema must declare $schema as Draft 2020-12: {file_path}")
    
    if "$id" not in schema:
        raise ValueError(f"Schema must declare $id: {file_path}")
    
    schema_id = schema["$id"]
    if not isinstance(schema_id, str) or not schema_id.startswith(("http://", "https://")):
        raise ValueError(f"Schema $id must be a valid URI: {file_path} -> {schema_id}")


def _load_all_schemas() -> None:
    """Load all schemas into the global store with comprehensive validation."""
    with _STORE_LOCK:
        if _SCHEMA_STORE:
            return

        root = _schemas_root()
        if not root.exists():
            raise RuntimeError(f"Schemas root not found: {root}")

        loaded_files = []
        duplicate_ids = []
        
        for file_path in _iter_schema_files(root):
            try:
                with file_path.open("r", encoding="utf-8") as f:
                    schema = json.load(f)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON in schema: {file_path}: {e}") from e

            # Validate schema structure
            try:
                _validate_schema_metadata(schema, file_path)
            except ValueError as e:
                raise RuntimeError(str(e)) from e

            schema_id = schema["$id"]
            
            # Check for duplicates with helpful error
            if schema_id in _SCHEMA_STORE:
                existing_file = _SCHEMA_STORE[schema_id].get("__file__", "unknown")
                duplicate_ids.append(f"  {schema_id}\n    First:  {existing_file}\n    Second: {file_path}")
                continue

            # Store schema with metadata
            schema["__file__"] = str(file_path)
            _SCHEMA_STORE[schema_id] = schema
            _SCHEMA_PATHS[schema_id] = str(file_path)
            loaded_files.append(file_path.name)

        if duplicate_ids:
            raise RuntimeError("Duplicate schema $id detected:\n" + "\n".join(duplicate_ids))
        
        print(f"✅ Loaded {len(loaded_files)} schemas: {', '.join(sorted(loaded_files))}")


def _build_resolver() -> jsonschema.RefResolver:
    """Build a resolver with the global schema store for $id-based resolution."""
    return jsonschema.RefResolver.from_schema(
        {
            "$id": "urn:corpus:root",
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Corpus Schema Registry Root"
        },
        store=_SCHEMA_STORE,
    )


def _make_validator(schema_id: str) -> Draft202012Validator:
    """Create a validator for the given schema ID with comprehensive error handling."""
    with _STORE_LOCK:
        if not _SCHEMA_STORE:
            _load_all_schemas()

        if schema_id not in _SCHEMA_STORE:
            # Provide helpful suggestions for unknown schema IDs
            available_ids = list(_SCHEMA_STORE.keys())
            suggestions = []
            
            # Try suffix matching
            for available_id in available_ids:
                if available_id.endswith(schema_id):
                    suggestions.append(available_id)
                elif schema_id in available_id:
                    suggestions.append(available_id)
            
            error_msg = f"Schema not found: {schema_id}"
            if suggestions:
                error_msg += f"\nDid you mean one of:\n  " + "\n  ".join(suggestions[:5])  # Limit to 5 suggestions
            else:
                error_msg += f"\nAvailable schemas ({len(available_ids)}):\n  " + "\n  ".join(
                    sorted(available_ids)[:10]  # Show first 10
                )
            
            raise KeyError(error_msg)

        schema = _SCHEMA_STORE[schema_id]
        resolver = _build_resolver()
        
        try:
            # Validate the schema itself first
            Draft202012Validator.check_schema(schema)
            
            # Create validator with format checking
            validator = Draft202012Validator(
                schema, 
                resolver=resolver, 
                format_checker=jsonschema.draft202012_format_checker
            )
        except RefResolutionError as e:
            raise RuntimeError(
                f"Failed to resolve $refs for schema '{schema_id}' ({schema.get('__file__')}): {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to create validator for schema '{schema_id}' ({schema.get('__file__')}): {e}"
            ) from e

        return validator


def get_validator(schema_id: str) -> Draft202012Validator:
    """
    Return a cached Draft202012 validator for the given $id.
    
    Args:
        schema_id: The $id of the schema to validate against
        
    Returns:
        Draft202012Validator: Cached validator instance
        
    Raises:
        KeyError: If schema_id is not found
        RuntimeError: If schema validation fails
    """
    with _STORE_LOCK:
        if schema_id in _VALIDATOR_CACHE:
            return _VALIDATOR_CACHE[schema_id]
        
        validator = _make_validator(schema_id)
        _VALIDATOR_CACHE[schema_id] = validator
        return validator


def validate_json(schema_id: str, obj: Any) -> None:
    """
    Validate an object against the schema identified by $id.
    
    Args:
        schema_id: The $id of the schema to validate against
        obj: The JSON-serializable object to validate
        
    Raises:
        ValidationError: If validation fails
        KeyError: If schema_id is not found
        RuntimeError: If schema validation fails
    """
    validator = get_validator(schema_id)
    validator.validate(obj)


def assert_valid(schema_id: str, obj: Any, *, context: Optional[str] = None) -> None:
    """
    Pytest-friendly validation with rich error messages.
    
    Args:
        schema_id: The $id of the schema to validate against
        obj: The JSON-serializable object to validate
        context: Optional context string for error messages
        
    Raises:
        AssertionError: If validation fails, with detailed error information
    """
    try:
        validate_json(schema_id, obj)
    except ValidationError as e:
        # Build comprehensive error message
        error_parts = [
            f"JSON validation failed against {schema_id}",
            f"Schema file: {_SCHEMA_PATHS.get(schema_id, 'unknown')}",
            f"Error: {e.message}",
            f"Failing value: {repr(e.instance)}" if e.instance is not None else "Failing value: <none>",
            f"JSON path: {'.'.join(str(p) for p in e.path) if e.path else '<root>'}", 
            f"Schema path: {'.'.join(str(p) for p in e.schema_path) if e.schema_path else '<root>'}"
        ]
        
        if context:
            error_parts.insert(1, f"Context: {context}")
            
        raise AssertionError("\n".join(error_parts)) from e


def preload_all_schemas() -> None:
    """Force preload all schemas and validate the entire registry."""
    _load_all_schemas()
    
    # Validate all schemas can create validators
    with _STORE_LOCK:
        for schema_id in list(_SCHEMA_STORE.keys()):
            try:
                get_validator(schema_id)
            except Exception as e:
                raise RuntimeError(f"Failed to create validator for {schema_id}: {e}") from e


def list_schemas() -> Dict[str, str]:
    """
    Get mapping of all schema IDs to their file paths.
    
    Returns:
        Dict[str, str]: Mapping of schema $id to file path
    """
    _load_all_schemas()
    return _SCHEMA_PATHS.copy()


def get_schema(schema_id: str) -> dict:
    """
    Get the raw schema document by $id.
    
    Args:
        schema_id: The $id of the schema to retrieve
        
    Returns:
        dict: The schema document
        
    Raises:
        KeyError: If schema_id is not found
    """
    _load_all_schemas()
    with _STORE_LOCK:
        if schema_id not in _SCHEMA_STORE:
            raise KeyError(f"Schema not found: {schema_id}")
        return _SCHEMA_STORE[schema_id].copy()  # Return copy to prevent mutation


def clear_cache() -> None:
    """Clear all caches (primarily for testing)."""
    with _STORE_LOCK:
        _SCHEMA_STORE.clear()
        _VALIDATOR_CACHE.clear()
        _SCHEMA_PATHS.clear()


# Command-line interface for validation and debugging
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Validate JSON against a registered schema $id",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s llm.envelope.request.json sample_request.json
  %(prog)s https://adaptersdk.org/schemas/llm/llm.envelope.request.json sample_request.json
  
Environment:
  CORPUS_SCHEMAS_ROOT    Override the default schemas directory
        """
    )
    parser.add_argument("schema_id", help="Schema $id (exact) or file name suffix")
    parser.add_argument("json_file", help="Path to JSON document to validate")
    parser.add_argument("--list", action="store_true", help="List all available schemas")
    parser.add_argument("--preload", action="store_true", help="Preload and validate all schemas")
    
    args = parser.parse_args()

    try:
        if args.list:
            preload_all_schemas()
            schemas = list_schemas()
            print(f"Available schemas ({len(schemas)}):")
            for schema_id, file_path in sorted(schemas.items()):
                print(f"  {schema_id} -> {file_path}")
            sys.exit(0)
            
        if args.preload:
            preload_all_schemas()
            print("✅ All schemas loaded and validated successfully")
            sys.exit(0)

        # Normal validation mode
        preload_all_schemas()
        
        # Find matching schema ID
        target_id = args.schema_id
        if target_id not in _SCHEMA_STORE:
            # Try suffix matching for convenience
            matches = [sid for sid in _SCHEMA_STORE.keys() if sid.endswith(target_id)]
            if len(matches) == 1:
                target_id = matches[0]
                print(f"🔍 Using schema: {target_id}")
            elif len(matches) > 1:
                print(f"❌ Multiple schemas match '{args.schema_id}':")
                for match in matches:
                    print(f"   {match}")
                sys.exit(1)
            else:
                print(f"❌ Schema not found: {args.schema_id}")
                sys.exit(1)

        with open(args.json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        validate_json(target_id, data)
        print(f"✅ {args.json_file} validates against {target_id}")
        sys.exit(0)
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)