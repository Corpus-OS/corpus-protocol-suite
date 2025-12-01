# SPDX-License-Identifier: Apache-2.0
"""
Wire-level validators for CORPUS Protocol conformance testing.

This module provides validation logic for wire-level request envelopes:

  - Envelope structure validation (op, ctx, args)
  - Context field validation (request_id, deadline_ms, tenant, etc.)
  - JSON serialization round-trip validation
  - Schema validation with version tolerance
  - Operation-specific argument validators

Separated from test execution to allow:
  - Reuse in production code for request validation
  - Unit testing of validators in isolation
  - Clear separation between "what to validate" and "how to validate"

IMPORTANT: This module is the source of truth for validation semantics.
When the protocol spec changes (e.g., new ctx fields, updated args constraints,
new valid roles/priorities), this module must be updated to stay in sync.
The constants and validators defined here should match the JSON Schema
definitions in schemas/*.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type Aliases
# ---------------------------------------------------------------------------

EnvelopeDict = Dict[str, Any]
ArgsDict = Dict[str, Any]
CtxDict = Dict[str, Any]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Envelope structure
REQUIRED_ENVELOPE_KEYS: FrozenSet[str] = frozenset({"op", "ctx", "args"})
REQUIRED_CTX_KEYS: FrozenSet[str] = frozenset({"request_id"})
OPTIONAL_CTX_KEYS: FrozenSet[str] = frozenset({
    "deadline_ms",
    "tenant", 
    "trace_id",
    "span_id",
    "priority",
    "idempotency_key",
})

# Validation limits
MAX_REQUEST_ID_LENGTH = 128
MIN_REQUEST_ID_LENGTH = 1
MAX_DEADLINE_MS = 3_600_000  # 1 hour
MIN_DEADLINE_MS = 1
MAX_TENANT_LENGTH = 256

# Vector limits
MAX_VECTOR_DIMENSIONS = 65536
MIN_VECTOR_DIMENSIONS = 1
MAX_TOP_K = 10_000
MIN_TOP_K = 1

# Batch limits
MAX_BATCH_SIZE = 1000
MAX_TEXT_LENGTH = 1_000_000  # ~1MB


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValidatorConfig:
    """Configuration for validation behavior."""
    
    enable_json_roundtrip: bool = True
    schema_version_tolerance: str = "strict"  # strict | tolerant | warn
    schema_base_url: str = "https://corpusos.com/schemas"
    
    @classmethod
    def from_env(cls) -> "ValidatorConfig":
        """Create configuration from environment variables."""
        return cls(
            enable_json_roundtrip=os.environ.get(
                "CORPUS_VALIDATION_FULL", "true"
            ).lower() == "true",
            schema_version_tolerance=os.environ.get(
                "CORPUS_SCHEMA_VERSION_TOLERANCE", "strict"
            ),
            schema_base_url=os.environ.get(
                "CORPUS_SCHEMA_BASE_URL", "https://corpusos.com/schemas"
            ),
        )


# Global config instance
CONFIG = ValidatorConfig.from_env()


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ValidationError(Exception):
    """Base exception for validation failures."""
    
    def __init__(
        self,
        message: str,
        case_id: Optional[str] = None,
        field: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.case_id = case_id
        self.field = field
        self.details = details or {}
        
        prefix = f"{case_id}: " if case_id else ""
        field_info = f" (field: {field})" if field else ""
        super().__init__(f"{prefix}{message}{field_info}")


class EnvelopeShapeError(ValidationError):
    """Envelope missing required structure."""
    pass


class EnvelopeTypeError(ValidationError):
    """Envelope field has wrong type."""
    pass


class CtxValidationError(ValidationError):
    """Context field validation failed."""
    pass


class ArgsValidationError(ValidationError):
    """Operation-specific args validation failed."""
    pass


class SchemaValidationError(ValidationError):
    """JSON Schema validation failed."""
    pass


class SerializationError(ValidationError):
    """JSON serialization/deserialization failed."""
    pass


# ---------------------------------------------------------------------------
# Schema Validation Cache
# ---------------------------------------------------------------------------

class SchemaValidationCache:
    """
    Thread-safe LRU cache for schema validation results.
    
    Caches validation results keyed by (schema_id, envelope_hash) to avoid
    redundant validation of identical requests.
    """
    
    def __init__(self, max_size: int = 256) -> None:
        self._cache: Dict[str, bool] = {}
        self._access_order: List[str] = []
        self._max_size = max_size
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    @staticmethod
    def _envelope_to_key(schema_id: str, envelope: EnvelopeDict) -> str:
        """Convert envelope to stable cache key."""
        canonical = json.dumps(envelope, sort_keys=True, separators=(",", ":"))
        # Use 32 hex chars (128 bits) for negligible collision risk at scale
        envelope_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]
        return f"{schema_id}:{envelope_hash}"
    
    def get(self, schema_id: str, envelope: EnvelopeDict) -> Optional[bool]:
        """Get cached validation result."""
        key = self._envelope_to_key(schema_id, envelope)
        with self._lock:
            if key in self._cache:
                self._access_order.remove(key)
                self._access_order.append(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None
    
    def set(self, schema_id: str, envelope: EnvelopeDict, valid: bool) -> None:
        """Cache validation result."""
        key = self._envelope_to_key(schema_id, envelope)
        with self._lock:
            if key in self._cache:
                self._access_order.remove(key)
            elif len(self._cache) >= self._max_size:
                oldest = self._access_order.pop(0)
                del self._cache[oldest]
            
            self._cache[key] = valid
            self._access_order.append(key)
    
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._hits = 0
            self._misses = 0
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
            }


# Global cache instance
_schema_cache = SchemaValidationCache()


def get_schema_cache() -> SchemaValidationCache:
    """Get the global schema validation cache."""
    return _schema_cache


# ---------------------------------------------------------------------------
# Envelope Structure Validation
# ---------------------------------------------------------------------------

def validate_envelope_shape(
    envelope: Any,
    case_id: Optional[str] = None,
) -> None:
    """
    Validate envelope has correct top-level structure.
    
    Args:
        envelope: The envelope to validate.
        case_id: Optional case ID for error messages.
    
    Raises:
        EnvelopeShapeError: If envelope structure is invalid.
        EnvelopeTypeError: If envelope is wrong type.
    """
    if not isinstance(envelope, dict):
        raise EnvelopeTypeError(
            f"Envelope must be dict, got {type(envelope).__name__}",
            case_id=case_id,
            details={"actual_type": type(envelope).__name__},
        )
    
    missing = REQUIRED_ENVELOPE_KEYS - set(envelope.keys())
    if missing:
        raise EnvelopeShapeError(
            f"Envelope missing required keys: {sorted(missing)}",
            case_id=case_id,
            details={"missing": sorted(missing), "present": sorted(envelope.keys())},
        )
    
    extra = set(envelope.keys()) - REQUIRED_ENVELOPE_KEYS
    if extra:
        logger.debug(f"Envelope has extra top-level keys: {sorted(extra)}")


def validate_op_field(
    envelope: EnvelopeDict,
    expected_op: str,
    case_id: Optional[str] = None,
) -> None:
    """
    Validate 'op' field matches expected operation.
    
    Args:
        envelope: The envelope to validate.
        expected_op: Expected operation name.
        case_id: Optional case ID for error messages.
    
    Raises:
        EnvelopeTypeError: If op is not a string.
        ValidationError: If op doesn't match expected.
    """
    op = envelope["op"]
    
    if not isinstance(op, str):
        raise EnvelopeTypeError(
            f"'op' must be string, got {type(op).__name__}",
            case_id=case_id,
            field="op",
        )
    
    if op != expected_op:
        raise ValidationError(
            f"Operation mismatch: expected '{expected_op}', got '{op}'",
            case_id=case_id,
            field="op",
            details={"expected": expected_op, "actual": op},
        )


def validate_ctx_field(
    envelope: EnvelopeDict,
    case_id: Optional[str] = None,
) -> None:
    """
    Validate 'ctx' field structure and contents.
    
    Args:
        envelope: The envelope to validate.
        case_id: Optional case ID for error messages.
    
    Raises:
        EnvelopeTypeError: If ctx or fields have wrong type.
        CtxValidationError: If ctx field values are invalid.
    """
    ctx = envelope["ctx"]
    
    if not isinstance(ctx, dict):
        raise EnvelopeTypeError(
            f"'ctx' must be object, got {type(ctx).__name__}",
            case_id=case_id,
            field="ctx",
        )
    
    # Required: request_id
    _validate_request_id(ctx, case_id)
    
    # Optional fields
    if "deadline_ms" in ctx:
        _validate_deadline_ms(ctx["deadline_ms"], case_id)
    
    if "tenant" in ctx and ctx["tenant"] is not None:
        _validate_tenant(ctx["tenant"], case_id)
    
    if "trace_id" in ctx and ctx["trace_id"] is not None:
        _validate_string_field(ctx["trace_id"], "ctx.trace_id", case_id)
    
    if "span_id" in ctx and ctx["span_id"] is not None:
        _validate_string_field(ctx["span_id"], "ctx.span_id", case_id)
    
    if "priority" in ctx and ctx["priority"] is not None:
        _validate_priority(ctx["priority"], case_id)
    
    if "idempotency_key" in ctx and ctx["idempotency_key"] is not None:
        _validate_string_field(ctx["idempotency_key"], "ctx.idempotency_key", case_id)
    
    # Warn on unknown fields
    known = REQUIRED_CTX_KEYS | OPTIONAL_CTX_KEYS
    unknown = set(ctx.keys()) - known
    if unknown:
        logger.debug(f"ctx has extension fields: {sorted(unknown)}")


def _validate_request_id(ctx: CtxDict, case_id: Optional[str]) -> None:
    """Validate request_id field."""
    if "request_id" not in ctx:
        raise CtxValidationError(
            "'ctx.request_id' is required",
            case_id=case_id,
            field="ctx.request_id",
        )
    
    request_id = ctx["request_id"]
    
    if not isinstance(request_id, str):
        raise EnvelopeTypeError(
            f"'ctx.request_id' must be string, got {type(request_id).__name__}",
            case_id=case_id,
            field="ctx.request_id",
        )
    
    if not (MIN_REQUEST_ID_LENGTH <= len(request_id) <= MAX_REQUEST_ID_LENGTH):
        raise CtxValidationError(
            f"'ctx.request_id' length must be {MIN_REQUEST_ID_LENGTH}-{MAX_REQUEST_ID_LENGTH}, "
            f"got {len(request_id)}",
            case_id=case_id,
            field="ctx.request_id",
            details={"length": len(request_id)},
        )


def _validate_deadline_ms(deadline: Any, case_id: Optional[str]) -> None:
    """Validate deadline_ms field."""
    if not isinstance(deadline, int):
        raise EnvelopeTypeError(
            f"'ctx.deadline_ms' must be integer, got {type(deadline).__name__}",
            case_id=case_id,
            field="ctx.deadline_ms",
        )
    
    if not (MIN_DEADLINE_MS <= deadline <= MAX_DEADLINE_MS):
        raise CtxValidationError(
            f"'ctx.deadline_ms' must be {MIN_DEADLINE_MS}-{MAX_DEADLINE_MS}, got {deadline}",
            case_id=case_id,
            field="ctx.deadline_ms",
            details={"value": deadline},
        )


def _validate_tenant(tenant: Any, case_id: Optional[str]) -> None:
    """Validate tenant field."""
    if not isinstance(tenant, str):
        raise EnvelopeTypeError(
            f"'ctx.tenant' must be string, got {type(tenant).__name__}",
            case_id=case_id,
            field="ctx.tenant",
        )
    
    if len(tenant) > MAX_TENANT_LENGTH:
        raise CtxValidationError(
            f"'ctx.tenant' length must be <= {MAX_TENANT_LENGTH}, got {len(tenant)}",
            case_id=case_id,
            field="ctx.tenant",
        )


def _validate_priority(priority: Any, case_id: Optional[str]) -> None:
    """Validate priority field."""
    valid_priorities = {"low", "normal", "high", "critical"}
    
    if not isinstance(priority, str):
        raise EnvelopeTypeError(
            f"'ctx.priority' must be string, got {type(priority).__name__}",
            case_id=case_id,
            field="ctx.priority",
        )
    
    if priority not in valid_priorities:
        raise CtxValidationError(
            f"'ctx.priority' must be one of {sorted(valid_priorities)}, got '{priority}'",
            case_id=case_id,
            field="ctx.priority",
        )


def _validate_string_field(value: Any, field_name: str, case_id: Optional[str]) -> None:
    """Validate a generic string field."""
    if not isinstance(value, str):
        raise EnvelopeTypeError(
            f"'{field_name}' must be string, got {type(value).__name__}",
            case_id=case_id,
            field=field_name,
        )


def validate_args_field(
    envelope: EnvelopeDict,
    case_id: Optional[str] = None,
) -> None:
    """
    Validate 'args' field is an object.
    
    Args:
        envelope: The envelope to validate.
        case_id: Optional case ID for error messages.
    
    Raises:
        EnvelopeTypeError: If args is not a dict.
    """
    args = envelope["args"]
    
    if not isinstance(args, dict):
        raise EnvelopeTypeError(
            f"'args' must be object, got {type(args).__name__}",
            case_id=case_id,
            field="args",
        )


def validate_envelope_common(
    envelope: EnvelopeDict,
    expected_op: str,
    case_id: Optional[str] = None,
) -> None:
    """
    Run all common envelope validations.
    
    Args:
        envelope: The envelope to validate.
        expected_op: Expected operation name.
        case_id: Optional case ID for error messages.
    
    Raises:
        ValidationError: If any validation fails.
    """
    validate_envelope_shape(envelope, case_id)
    validate_op_field(envelope, expected_op, case_id)
    validate_ctx_field(envelope, case_id)
    validate_args_field(envelope, case_id)


# ---------------------------------------------------------------------------
# JSON Round-Trip Validation
# ---------------------------------------------------------------------------

def json_roundtrip(
    envelope: EnvelopeDict,
    case_id: Optional[str] = None,
    skip_if_disabled: bool = True,
) -> EnvelopeDict:
    """
    Force JSON serialization round-trip to validate wire format.
    
    This catches:
      - Non-serializable types (datetime, Decimal, custom classes)
      - Hidden encoding issues
      - Objects that serialize differently than expected
    
    Args:
        envelope: The envelope to round-trip.
        case_id: Optional case ID for error messages.
        skip_if_disabled: If True, returns envelope unchanged when disabled.
    
    Returns:
        The deserialized envelope.
    
    Raises:
        SerializationError: If serialization or deserialization fails.
    """
    if skip_if_disabled and not CONFIG.enable_json_roundtrip:
        # Quick serializability check without full round-trip
        try:
            json.dumps(envelope, ensure_ascii=False)
            return envelope
        except (TypeError, ValueError) as e:
            raise SerializationError(
                f"Envelope not JSON-serializable: {e}",
                case_id=case_id,
                details={"error": str(e)},
            )
    
    try:
        payload = json.dumps(
            envelope,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as e:
        raise SerializationError(
            f"JSON serialization failed: {e}",
            case_id=case_id,
            details={"error": str(e)},
        )
    
    try:
        return json.loads(payload)
    except json.JSONDecodeError as e:
        raise SerializationError(
            f"JSON deserialization failed: {e}",
            case_id=case_id,
            details={"error": str(e), "payload_preview": payload[:200]},
        )


def assert_roundtrip_equality(
    original: EnvelopeDict,
    roundtripped: EnvelopeDict,
    case_id: Optional[str] = None,
) -> None:
    """
    Assert envelope wasn't mutated by serialization.
    
    Args:
        original: Original envelope before round-trip.
        roundtripped: Envelope after JSON round-trip.
        case_id: Optional case ID for error messages.
    
    Raises:
        SerializationError: If envelopes don't match.
    """
    if roundtripped != original:
        diff = _find_dict_diff(original, roundtripped)
        raise SerializationError(
            "Envelope mutated by JSON round-trip",
            case_id=case_id,
            details={"diff": diff},
        )


def _find_dict_diff(
    original: Dict[str, Any],
    modified: Dict[str, Any],
    path: str = "",
) -> List[str]:
    """Find differences between two dicts for debugging."""
    diffs = []
    all_keys = set(original.keys()) | set(modified.keys())
    
    for key in sorted(all_keys):
        key_path = f"{path}.{key}" if path else key
        
        if key not in original:
            diffs.append(f"Added: {key_path}")
        elif key not in modified:
            diffs.append(f"Removed: {key_path}")
        elif original[key] != modified[key]:
            if isinstance(original[key], dict) and isinstance(modified[key], dict):
                diffs.extend(_find_dict_diff(original[key], modified[key], key_path))
            else:
                orig_type = type(original[key]).__name__
                mod_type = type(modified[key]).__name__
                diffs.append(f"Changed: {key_path} ({orig_type} -> {mod_type})")
    
    return diffs


# ---------------------------------------------------------------------------
# Schema Validation
# ---------------------------------------------------------------------------

def validate_against_schema(
    schema_id: str,
    envelope: EnvelopeDict,
    case_id: Optional[str] = None,
    use_cache: bool = True,
) -> None:
    """
    Validate envelope against JSON Schema.
    
    Args:
        schema_id: Schema $id URL.
        envelope: Envelope to validate.
        case_id: Optional case ID for error messages.
        use_cache: Whether to use validation cache.
    
    Raises:
        SchemaValidationError: If validation fails.
    """
    # Check cache first
    if use_cache:
        cached = _schema_cache.get(schema_id, envelope)
        if cached is True:
            return
        elif cached is False:
            raise SchemaValidationError(
                "Schema validation failed (cached)",
                case_id=case_id,
                details={"schema_id": schema_id},
            )
    
    # Perform validation
    try:
        # Import here to avoid circular dependency and allow this module
        # to be used without the schema registry
        from tests.utils.schema_registry import assert_valid
        assert_valid(schema_id, envelope, context=f"wire:{case_id or 'unknown'}")
        
        if use_cache:
            _schema_cache.set(schema_id, envelope, True)
            
    except ImportError:
        logger.warning("Schema registry not available, skipping schema validation")
    except Exception as e:
        if use_cache:
            _schema_cache.set(schema_id, envelope, False)
        raise SchemaValidationError(
            f"Schema validation failed: {e}",
            case_id=case_id,
            details={"schema_id": schema_id, "error": str(e)},
        )


def validate_with_version_tolerance(
    envelope: EnvelopeDict,
    primary_schema_id: str,
    schema_versions: Tuple[str, ...],
    component: str,
    case_id: Optional[str] = None,
) -> None:
    """
    Validate against schema with version fallback support.
    
    In strict mode, only validates against primary_schema_id.
    In tolerant/warn mode, tries primary_schema_id first, then falls back
    to version-derived URLs.
    
    Args:
        envelope: Envelope to validate.
        primary_schema_id: Primary schema URL (tried first in all modes).
        schema_versions: Tuple of version strings for fallback URLs.
        component: Component name for building versioned URLs.
        case_id: Optional case ID for error messages.
    
    Raises:
        SchemaValidationError: If all versions fail validation.
    """
    if CONFIG.schema_version_tolerance == "strict":
        validate_against_schema(primary_schema_id, envelope, case_id)
        return
    
    # Try primary schema first
    errors = []
    try:
        validate_against_schema(primary_schema_id, envelope, case_id)
        return
    except SchemaValidationError as e:
        errors.append(("primary", str(e)))
    
    # Fall back to version-derived URLs
    for version in schema_versions:
        schema_id = f"{CONFIG.schema_base_url}/{component}/{version}/{component}.envelope.request.json"
        
        # Skip if this would be the same as primary (avoid duplicate attempt)
        if schema_id == primary_schema_id:
            continue
        
        try:
            validate_against_schema(schema_id, envelope, case_id)
            if CONFIG.schema_version_tolerance == "warn":
                logger.warning(f"{case_id}: Validated with fallback schema version {version}")
            return
        except SchemaValidationError as e:
            errors.append((version, str(e)))
    
    raise SchemaValidationError(
        f"Validation failed against primary schema and all fallback versions: {schema_versions}",
        case_id=case_id,
        details={"version_errors": errors},
    )


# ---------------------------------------------------------------------------
# Operation-Specific Args Validators
# ---------------------------------------------------------------------------

# Type alias for validator functions
ArgsValidator = Callable[[ArgsDict, Optional[str]], None]


def validate_llm_complete_args(args: ArgsDict, case_id: Optional[str] = None) -> None:
    """Validate llm.complete operation arguments."""
    if "prompt" not in args and "messages" not in args:
        raise ArgsValidationError(
            "llm.complete requires 'prompt' or 'messages'",
            case_id=case_id,
            field="args",
        )
    
    if "prompt" in args:
        prompt = args["prompt"]
        if not isinstance(prompt, str):
            raise ArgsValidationError(
                f"'args.prompt' must be string, got {type(prompt).__name__}",
                case_id=case_id,
                field="args.prompt",
            )
        if len(prompt) > MAX_TEXT_LENGTH:
            raise ArgsValidationError(
                f"'args.prompt' exceeds max length {MAX_TEXT_LENGTH}",
                case_id=case_id,
                field="args.prompt",
            )
    
    if "max_tokens" in args:
        _validate_positive_int(args["max_tokens"], "args.max_tokens", case_id)
    
    if "temperature" in args:
        _validate_temperature(args["temperature"], case_id)
    
    if "stream" in args:
        _validate_bool(args["stream"], "args.stream", case_id)
    
    if "stop" in args and args["stop"] is not None:
        _validate_stop_sequences(args["stop"], case_id)


def validate_llm_chat_args(args: ArgsDict, case_id: Optional[str] = None) -> None:
    """Validate llm.chat operation arguments."""
    if "messages" not in args:
        raise ArgsValidationError(
            "llm.chat requires 'messages'",
            case_id=case_id,
            field="args.messages",
        )
    
    messages = args["messages"]
    if not isinstance(messages, list):
        raise ArgsValidationError(
            f"'args.messages' must be array, got {type(messages).__name__}",
            case_id=case_id,
            field="args.messages",
        )
    
    if len(messages) == 0:
        raise ArgsValidationError(
            "'args.messages' must not be empty",
            case_id=case_id,
            field="args.messages",
        )
    
    for i, msg in enumerate(messages):
        _validate_chat_message(msg, i, case_id)
    
    if "tools" in args and args["tools"] is not None:
        _validate_tools(args["tools"], case_id)
    
    if "tool_choice" in args and args["tool_choice"] is not None:
        _validate_tool_choice(args["tool_choice"], case_id)
    
    # Shared with llm.complete
    if "max_tokens" in args:
        _validate_positive_int(args["max_tokens"], "args.max_tokens", case_id)
    
    if "temperature" in args:
        _validate_temperature(args["temperature"], case_id)


def validate_llm_count_tokens_args(args: ArgsDict, case_id: Optional[str] = None) -> None:
    """Validate llm.count_tokens operation arguments."""
    if "text" not in args and "messages" not in args:
        raise ArgsValidationError(
            "llm.count_tokens requires 'text' or 'messages'",
            case_id=case_id,
            field="args",
        )
    
    if "text" in args:
        if not isinstance(args["text"], str):
            raise ArgsValidationError(
                f"'args.text' must be string, got {type(args['text']).__name__}",
                case_id=case_id,
                field="args.text",
            )


def validate_vector_query_args(args: ArgsDict, case_id: Optional[str] = None) -> None:
    """Validate vector.query operation arguments."""
    if "vector" not in args and "text" not in args:
        raise ArgsValidationError(
            "vector.query requires 'vector' or 'text'",
            case_id=case_id,
            field="args",
        )
    
    if "vector" in args:
        _validate_vector(args["vector"], "args.vector", case_id)
    
    if "text" in args:
        if not isinstance(args["text"], str):
            raise ArgsValidationError(
                f"'args.text' must be string, got {type(args['text']).__name__}",
                case_id=case_id,
                field="args.text",
            )
    
    if "top_k" in args:
        top_k = args["top_k"]
        if not isinstance(top_k, int) or not (MIN_TOP_K <= top_k <= MAX_TOP_K):
            raise ArgsValidationError(
                f"'args.top_k' must be integer in [{MIN_TOP_K}, {MAX_TOP_K}]",
                case_id=case_id,
                field="args.top_k",
            )
    
    if "namespace" in args and args["namespace"] is not None:
        if not isinstance(args["namespace"], str):
            raise ArgsValidationError(
                "'args.namespace' must be string",
                case_id=case_id,
                field="args.namespace",
            )
    
    if "filter" in args and args["filter"] is not None:
        if not isinstance(args["filter"], dict):
            raise ArgsValidationError(
                "'args.filter' must be object",
                case_id=case_id,
                field="args.filter",
            )
    
    if "include_metadata" in args:
        _validate_bool(args["include_metadata"], "args.include_metadata", case_id)
    
    if "include_values" in args:
        _validate_bool(args["include_values"], "args.include_values", case_id)


def validate_vector_upsert_args(args: ArgsDict, case_id: Optional[str] = None) -> None:
    """Validate vector.upsert operation arguments."""
    if "vectors" not in args:
        raise ArgsValidationError(
            "vector.upsert requires 'vectors'",
            case_id=case_id,
            field="args.vectors",
        )
    
    vectors = args["vectors"]
    if not isinstance(vectors, list):
        raise ArgsValidationError(
            f"'args.vectors' must be array, got {type(vectors).__name__}",
            case_id=case_id,
            field="args.vectors",
        )
    
    if len(vectors) == 0:
        raise ArgsValidationError(
            "'args.vectors' must not be empty",
            case_id=case_id,
            field="args.vectors",
        )
    
    if len(vectors) > MAX_BATCH_SIZE:
        raise ArgsValidationError(
            f"'args.vectors' exceeds max batch size {MAX_BATCH_SIZE}",
            case_id=case_id,
            field="args.vectors",
        )
    
    expected_dims = None
    for i, vec in enumerate(vectors):
        dims = _validate_vector_record(vec, i, case_id)
        if expected_dims is None:
            expected_dims = dims
        elif dims != expected_dims:
            raise ArgsValidationError(
                f"Dimension mismatch: vectors[0] has {expected_dims} dims, "
                f"vectors[{i}] has {dims}",
                case_id=case_id,
                field=f"args.vectors[{i}].values",
            )


def validate_vector_delete_args(args: ArgsDict, case_id: Optional[str] = None) -> None:
    """Validate vector.delete operation arguments."""
    if "ids" not in args and "filter" not in args and "delete_all" not in args:
        raise ArgsValidationError(
            "vector.delete requires 'ids', 'filter', or 'delete_all'",
            case_id=case_id,
            field="args",
        )
    
    if "ids" in args:
        ids = args["ids"]
        if not isinstance(ids, list):
            raise ArgsValidationError(
                "'args.ids' must be array",
                case_id=case_id,
                field="args.ids",
            )
        if not all(isinstance(id_, str) for id_ in ids):
            raise ArgsValidationError(
                "'args.ids' must contain only strings",
                case_id=case_id,
                field="args.ids",
            )
    
    if "delete_all" in args:
        _validate_bool(args["delete_all"], "args.delete_all", case_id)


def validate_vector_fetch_args(args: ArgsDict, case_id: Optional[str] = None) -> None:
    """Validate vector.fetch operation arguments."""
    if "ids" not in args:
        raise ArgsValidationError(
            "vector.fetch requires 'ids'",
            case_id=case_id,
            field="args.ids",
        )
    
    ids = args["ids"]
    if not isinstance(ids, list):
        raise ArgsValidationError(
            "'args.ids' must be array",
            case_id=case_id,
            field="args.ids",
        )
    
    if not all(isinstance(id_, str) for id_ in ids):
        raise ArgsValidationError(
            "'args.ids' must contain only strings",
            case_id=case_id,
            field="args.ids",
        )


def validate_embedding_embed_args(args: ArgsDict, case_id: Optional[str] = None) -> None:
    """Validate embedding.embed operation arguments."""
    if "text" not in args and "texts" not in args:
        raise ArgsValidationError(
            "embedding.embed requires 'text' or 'texts'",
            case_id=case_id,
            field="args",
        )
    
    if "text" in args:
        text = args["text"]
        if not isinstance(text, str):
            raise ArgsValidationError(
                f"'args.text' must be string, got {type(text).__name__}",
                case_id=case_id,
                field="args.text",
            )
        if len(text) > MAX_TEXT_LENGTH:
            raise ArgsValidationError(
                f"'args.text' exceeds max length {MAX_TEXT_LENGTH}",
                case_id=case_id,
                field="args.text",
            )
    
    if "texts" in args:
        texts = args["texts"]
        if not isinstance(texts, list):
            raise ArgsValidationError(
                f"'args.texts' must be array, got {type(texts).__name__}",
                case_id=case_id,
                field="args.texts",
            )
        if len(texts) > MAX_BATCH_SIZE:
            raise ArgsValidationError(
                f"'args.texts' exceeds max batch size {MAX_BATCH_SIZE}",
                case_id=case_id,
                field="args.texts",
            )
        for i, t in enumerate(texts):
            if not isinstance(t, str):
                raise ArgsValidationError(
                    f"'args.texts[{i}]' must be string",
                    case_id=case_id,
                    field=f"args.texts[{i}]",
                )
    
    if "dimensions" in args:
        _validate_positive_int(args["dimensions"], "args.dimensions", case_id)
    
    if "model" in args and args["model"] is not None:
        if not isinstance(args["model"], str):
            raise ArgsValidationError(
                "'args.model' must be string",
                case_id=case_id,
                field="args.model",
            )


def validate_graph_query_args(args: ArgsDict, case_id: Optional[str] = None) -> None:
    """Validate graph.query operation arguments."""
    if "query" not in args:
        raise ArgsValidationError(
            "graph.query requires 'query'",
            case_id=case_id,
            field="args.query",
        )
    
    query = args["query"]
    if not isinstance(query, str):
        raise ArgsValidationError(
            f"'args.query' must be string, got {type(query).__name__}",
            case_id=case_id,
            field="args.query",
        )
    
    if "language" in args:
        valid_languages = {"cypher", "gremlin", "sparql", "graphql"}
        if args["language"] not in valid_languages:
            raise ArgsValidationError(
                f"'args.language' must be one of {sorted(valid_languages)}",
                case_id=case_id,
                field="args.language",
            )
    
    if "parameters" in args and args["parameters"] is not None:
        if not isinstance(args["parameters"], dict):
            raise ArgsValidationError(
                "'args.parameters' must be object",
                case_id=case_id,
                field="args.parameters",
            )
    
    if "timeout_ms" in args:
        _validate_positive_int(args["timeout_ms"], "args.timeout_ms", case_id)


def validate_graph_mutate_args(args: ArgsDict, case_id: Optional[str] = None) -> None:
    """Validate graph.mutate operation arguments."""
    if "mutations" not in args:
        raise ArgsValidationError(
            "graph.mutate requires 'mutations'",
            case_id=case_id,
            field="args.mutations",
        )
    
    mutations = args["mutations"]
    if not isinstance(mutations, list):
        raise ArgsValidationError(
            f"'args.mutations' must be array, got {type(mutations).__name__}",
            case_id=case_id,
            field="args.mutations",
        )
    
    valid_types = {
        "create_node", "create_edge", 
        "update_node", "update_edge",
        "delete_node", "delete_edge",
    }
    
    for i, mut in enumerate(mutations):
        if not isinstance(mut, dict):
            raise ArgsValidationError(
                f"'args.mutations[{i}]' must be object",
                case_id=case_id,
                field=f"args.mutations[{i}]",
            )
        if "type" not in mut:
            raise ArgsValidationError(
                f"'args.mutations[{i}].type' is required",
                case_id=case_id,
                field=f"args.mutations[{i}].type",
            )
        if mut["type"] not in valid_types:
            raise ArgsValidationError(
                f"'args.mutations[{i}].type' must be one of {sorted(valid_types)}",
                case_id=case_id,
                field=f"args.mutations[{i}].type",
            )


def validate_graph_traverse_args(args: ArgsDict, case_id: Optional[str] = None) -> None:
    """Validate graph.traverse operation arguments."""
    if "start_node" not in args:
        raise ArgsValidationError(
            "graph.traverse requires 'start_node'",
            case_id=case_id,
            field="args.start_node",
        )
    
    if "depth" in args:
        depth = args["depth"]
        if not isinstance(depth, int) or depth < 0:
            raise ArgsValidationError(
                "'args.depth' must be non-negative integer",
                case_id=case_id,
                field="args.depth",
            )
    
    if "direction" in args:
        valid_directions = {"outbound", "inbound", "both"}
        if args["direction"] not in valid_directions:
            raise ArgsValidationError(
                f"'args.direction' must be one of {sorted(valid_directions)}",
                case_id=case_id,
                field="args.direction",
            )
    
    if "edge_types" in args and args["edge_types"] is not None:
        if not isinstance(args["edge_types"], list):
            raise ArgsValidationError(
                "'args.edge_types' must be array",
                case_id=case_id,
                field="args.edge_types",
            )


# ---------------------------------------------------------------------------
# Helper Validators
# ---------------------------------------------------------------------------

def _validate_vector(vector: Any, field_name: str, case_id: Optional[str]) -> None:
    """Validate a vector array."""
    if not isinstance(vector, list):
        raise ArgsValidationError(
            f"'{field_name}' must be array, got {type(vector).__name__}",
            case_id=case_id,
            field=field_name,
        )
    
    if not (MIN_VECTOR_DIMENSIONS <= len(vector) <= MAX_VECTOR_DIMENSIONS):
        raise ArgsValidationError(
            f"'{field_name}' dimensions must be {MIN_VECTOR_DIMENSIONS}-"
            f"{MAX_VECTOR_DIMENSIONS}, got {len(vector)}",
            case_id=case_id,
            field=field_name,
        )
    
    if not all(isinstance(v, (int, float)) for v in vector):
        raise ArgsValidationError(
            f"'{field_name}' must contain only numbers",
            case_id=case_id,
            field=field_name,
        )


def _validate_vector_record(record: Any, index: int, case_id: Optional[str]) -> int:
    """Validate a vector record in upsert. Returns dimension count."""
    field_prefix = f"args.vectors[{index}]"
    
    if not isinstance(record, dict):
        raise ArgsValidationError(
            f"'{field_prefix}' must be object",
            case_id=case_id,
            field=field_prefix,
        )
    
    if "id" not in record:
        raise ArgsValidationError(
            f"'{field_prefix}.id' is required",
            case_id=case_id,
            field=f"{field_prefix}.id",
        )
    
    if not isinstance(record["id"], str):
        raise ArgsValidationError(
            f"'{field_prefix}.id' must be string",
            case_id=case_id,
            field=f"{field_prefix}.id",
        )
    
    if "values" not in record:
        raise ArgsValidationError(
            f"'{field_prefix}.values' is required",
            case_id=case_id,
            field=f"{field_prefix}.values",
        )
    
    values = record["values"]
    _validate_vector(values, f"{field_prefix}.values", case_id)
    
    if "metadata" in record and record["metadata"] is not None:
        if not isinstance(record["metadata"], dict):
            raise ArgsValidationError(
                f"'{field_prefix}.metadata' must be object",
                case_id=case_id,
                field=f"{field_prefix}.metadata",
            )
    
    return len(values)


def _validate_chat_message(msg: Any, index: int, case_id: Optional[str]) -> None:
    """Validate a chat message."""
    field_prefix = f"args.messages[{index}]"
    
    if not isinstance(msg, dict):
        raise ArgsValidationError(
            f"'{field_prefix}' must be object",
            case_id=case_id,
            field=field_prefix,
        )
    
    if "role" not in msg:
        raise ArgsValidationError(
            f"'{field_prefix}.role' is required",
            case_id=case_id,
            field=f"{field_prefix}.role",
        )
    
    valid_roles = {"system", "user", "assistant", "tool"}
    if msg["role"] not in valid_roles:
        raise ArgsValidationError(
            f"'{field_prefix}.role' must be one of {sorted(valid_roles)}",
            case_id=case_id,
            field=f"{field_prefix}.role",
        )
    
    if "content" not in msg:
        raise ArgsValidationError(
            f"'{field_prefix}.content' is required",
            case_id=case_id,
            field=f"{field_prefix}.content",
        )


def _validate_tools(tools: Any, case_id: Optional[str]) -> None:
    """Validate tools array."""
    if not isinstance(tools, list):
        raise ArgsValidationError(
            "'args.tools' must be array",
            case_id=case_id,
            field="args.tools",
        )
    
    for i, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise ArgsValidationError(
                f"'args.tools[{i}]' must be object",
                case_id=case_id,
                field=f"args.tools[{i}]",
            )
        if "name" not in tool:
            raise ArgsValidationError(
                f"'args.tools[{i}].name' is required",
                case_id=case_id,
                field=f"args.tools[{i}].name",
            )


def _validate_tool_choice(choice: Any, case_id: Optional[str]) -> None:
    """Validate tool_choice field."""
    if isinstance(choice, str):
        valid_choices = {"auto", "none", "required"}
        if choice not in valid_choices:
            raise ArgsValidationError(
                f"'args.tool_choice' string must be one of {sorted(valid_choices)}",
                case_id=case_id,
                field="args.tool_choice",
            )
    elif isinstance(choice, dict):
        if "type" not in choice:
            raise ArgsValidationError(
                "'args.tool_choice.type' is required when tool_choice is object",
                case_id=case_id,
                field="args.tool_choice.type",
            )
    else:
        raise ArgsValidationError(
            f"'args.tool_choice' must be string or object, got {type(choice).__name__}",
            case_id=case_id,
            field="args.tool_choice",
        )


def _validate_temperature(temp: Any, case_id: Optional[str]) -> None:
    """Validate temperature field."""
    if not isinstance(temp, (int, float)):
        raise ArgsValidationError(
            f"'args.temperature' must be number, got {type(temp).__name__}",
            case_id=case_id,
            field="args.temperature",
        )
    if not (0 <= temp <= 2):
        raise ArgsValidationError(
            f"'args.temperature' must be in [0, 2], got {temp}",
            case_id=case_id,
            field="args.temperature",
        )


def _validate_stop_sequences(stop: Any, case_id: Optional[str]) -> None:
    """Validate stop sequences."""
    if isinstance(stop, str):
        return
    if isinstance(stop, list):
        if not all(isinstance(s, str) for s in stop):
            raise ArgsValidationError(
                "'args.stop' array must contain only strings",
                case_id=case_id,
                field="args.stop",
            )
        return
    raise ArgsValidationError(
        f"'args.stop' must be string or array, got {type(stop).__name__}",
        case_id=case_id,
        field="args.stop",
    )


def _validate_positive_int(value: Any, field_name: str, case_id: Optional[str]) -> None:
    """Validate a positive integer field."""
    if not isinstance(value, int) or value <= 0:
        raise ArgsValidationError(
            f"'{field_name}' must be positive integer",
            case_id=case_id,
            field=field_name,
        )


def _validate_bool(value: Any, field_name: str, case_id: Optional[str]) -> None:
    """Validate a boolean field."""
    if not isinstance(value, bool):
        raise ArgsValidationError(
            f"'{field_name}' must be boolean, got {type(value).__name__}",
            case_id=case_id,
            field=field_name,
        )


# ---------------------------------------------------------------------------
# Validator Registry
# ---------------------------------------------------------------------------

ARGS_VALIDATORS: Dict[str, ArgsValidator] = {
    "validate_llm_complete_args": validate_llm_complete_args,
    "validate_llm_chat_args": validate_llm_chat_args,
    "validate_llm_count_tokens_args": validate_llm_count_tokens_args,
    "validate_vector_query_args": validate_vector_query_args,
    "validate_vector_upsert_args": validate_vector_upsert_args,
    "validate_vector_delete_args": validate_vector_delete_args,
    "validate_vector_fetch_args": validate_vector_fetch_args,
    "validate_embedding_embed_args": validate_embedding_embed_args,
    "validate_graph_query_args": validate_graph_query_args,
    "validate_graph_mutate_args": validate_graph_mutate_args,
    "validate_graph_traverse_args": validate_graph_traverse_args,
}


def get_args_validator(name: str) -> Optional[ArgsValidator]:
    """Get an args validator by name."""
    return ARGS_VALIDATORS.get(name)


def validate_args_for_operation(
    args: ArgsDict,
    validator_name: Optional[str],
    case_id: Optional[str] = None,
) -> None:
    """
    Run operation-specific args validation if validator is defined.
    
    Args:
        args: The args dict to validate.
        validator_name: Name of the validator function, or None.
        case_id: Optional case ID for error messages.
    
    Raises:
        ArgsValidationError: If validation fails.
    """
    if validator_name is None:
        return
    
    validator = get_args_validator(validator_name)
    if validator is None:
        logger.warning(f"Unknown args validator: {validator_name}")
        return
    
    validator(args, case_id)


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def validate_wire_envelope(
    envelope: EnvelopeDict,
    expected_op: str,
    schema_id: str,
    schema_versions: Tuple[str, ...] = ("v1",),
    component: str = "",
    args_validator: Optional[str] = None,
    case_id: Optional[str] = None,
) -> EnvelopeDict:
    """
    Complete wire envelope validation pipeline.
    
    Runs all validation steps:
      1. Envelope structure
      2. JSON round-trip
      3. Schema validation (with version tolerance)
      4. Operation-specific args validation
    
    Args:
        envelope: The envelope to validate.
        expected_op: Expected operation name.
        schema_id: Primary JSON Schema URL.
        schema_versions: Tuple of supported versions.
        component: Component name for schema URL construction.
        args_validator: Name of args validator function.
        case_id: Optional case ID for error messages.
    
    Returns:
        The validated (possibly round-tripped) envelope.
    
    Raises:
        ValidationError: If any validation step fails.
    """
    # 1. Structure validation
    validate_envelope_common(envelope, expected_op, case_id)
    
    # 2. JSON round-trip
    wire_envelope = json_roundtrip(envelope, case_id)
    
    if CONFIG.enable_json_roundtrip:
        assert_roundtrip_equality(envelope, wire_envelope, case_id)
    
    # 3. Schema validation
    if component:
        validate_with_version_tolerance(
            wire_envelope, schema_id, schema_versions, component, case_id
        )
    else:
        validate_against_schema(schema_id, wire_envelope, case_id)
    
    # 4. Args validation
    validate_args_for_operation(wire_envelope["args"], args_validator, case_id)
    
    return wire_envelope
