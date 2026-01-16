# SPDX-License-Identifier: Apache-2.0
"""
Wire-level validators for CORPUS Protocol conformance testing.

This module provides validation logic for wire-level request envelopes:

  - Envelope structure validation (op, ctx, args)
  - Context field validation (OperationContext, aligned to SCHEMA.md)
  - JSON serialization round-trip validation
  - Strict JSON Schema validation (Draft 2020-12) with SCHEMA.md version tolerance
  - Operation-specific argument validators (lightweight, schema-aligned checks)
  - Coverage gates: ensure schema request-ops are covered by tests/live/wire_cases.py

Separated from test execution to allow:
  - Reuse in production code for request validation (best-effort mode)
  - Unit testing of validators in isolation
  - Clear separation between "what to validate" and "how to validate"

ALIGNMENT NOTE (SCHEMA.md)
--------------------------
SCHEMA.md is normative for field names, types, required/optional status, enums, and constraints.

Key points implemented here:
- common/envelope.request.json: requires op, ctx, args; additionalProperties: true
- common/operation_context.json: all fields optional and nullable; additionalProperties: true
- Type schemas often use additionalProperties: false; therefore STRICT conformance must run schema validation.
- Version tolerance is normative: schema_id#version/<semver> fallback is implemented when enabled.

Policy constraints (max sizes, “nice-to-have” semantics) are WARN-only by default so schema-valid payloads
do not fail conformance unless explicitly enforced.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
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
# Constants (SCHEMA.md-aligned)
# ---------------------------------------------------------------------------

# Envelope structure: common/envelope.request.json
REQUIRED_ENVELOPE_KEYS: FrozenSet[str] = frozenset({"op", "ctx", "args"})

# OperationContext keys: common/operation_context.json (all optional, nullable)
KNOWN_CTX_KEYS: FrozenSet[str] = frozenset(
    {
        "request_id",
        "idempotency_key",
        "deadline_ms",
        "traceparent",
        "tenant",
        "attrs",
    }
)

# Policy limits (NOT schema-required). Must not reject schema-valid payloads unless enforced.
POLICY_MAX_REQUEST_ID_LENGTH = 128
POLICY_MIN_REQUEST_ID_LENGTH = 1

POLICY_MAX_VECTOR_DIMENSIONS = 10_000
POLICY_MAX_TOP_K = 10_000
POLICY_MAX_BATCH_SIZE = 1000
POLICY_MAX_TEXT_LENGTH = 1_000_000  # ~1MB

# SCHEMA.md version tolerance fragment format: "#version/<semver>"
_VERSION_FRAGMENT_PREFIX = "version/"
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValidatorConfig:
    """
    Configuration for validation behavior.

    enable_json_roundtrip:
      - True: full JSON round-trip check (wire safety)
      - False: quick JSON-serializability check only

    schema_validation_mode:
      - "strict": schema registry missing => FAIL (conformance posture)
      - "best_effort": schema registry missing => WARN + continue (production reuse posture)

    schema_version_tolerance:
      - "strict": validate only primary schema_id
      - "tolerant": try primary, then #version/<semver> variants (no extra warnings)
      - "warn": same as tolerant but logs when fallback is used

    policy_enforcement:
      - "off": no policy checks
      - "warn": warn-only (default) so schema-valid payloads pass
      - "enforce": raise on policy violations
    """

    enable_json_roundtrip: bool = True
    schema_validation_mode: str = "strict"          # strict | best_effort
    schema_version_tolerance: str = "strict"        # strict | tolerant | warn
    policy_enforcement: str = "warn"                # off | warn | enforce

    @classmethod
    def from_env(cls) -> "ValidatorConfig":
        return cls(
            enable_json_roundtrip=os.environ.get("CORPUS_VALIDATION_FULL", "true").lower() == "true",
            schema_validation_mode=os.environ.get("CORPUS_SCHEMA_VALIDATION_MODE", "strict").lower(),
            schema_version_tolerance=os.environ.get("CORPUS_SCHEMA_VERSION_TOLERANCE", "strict").lower(),
            policy_enforcement=os.environ.get("CORPUS_POLICY_ENFORCEMENT", "warn").lower(),
        )


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
    pass


class EnvelopeTypeError(ValidationError):
    pass


class CtxValidationError(ValidationError):
    pass


class ArgsValidationError(ValidationError):
    pass


class SchemaValidationError(ValidationError):
    pass


class SerializationError(ValidationError):
    pass


# ---------------------------------------------------------------------------
# Helpers: Policy checks (warn-only by default)
# ---------------------------------------------------------------------------

def _policy_violation(
    message: str,
    *,
    case_id: Optional[str],
    field: Optional[str],
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Handle a policy violation according to CONFIG.policy_enforcement.

    - off: do nothing
    - warn: log warning
    - enforce: raise ValidationError
    """
    mode = CONFIG.policy_enforcement
    if mode == "off":
        return
    if mode == "warn":
        logger.warning(f"{case_id + ': ' if case_id else ''}{message}" + (f" (field: {field})" if field else ""))
        return
    if mode == "enforce":
        raise ValidationError(message, case_id=case_id, field=field, details=details or {})
    # Unknown mode -> warn (fail-safe)
    logger.warning(f"Unknown CORPUS_POLICY_ENFORCEMENT={mode!r}; treating as warn. {message}")


# ---------------------------------------------------------------------------
# Schema Validation Cache
# ---------------------------------------------------------------------------

class SchemaValidationCache:
    """
    Thread-safe LRU cache for schema validation results.

    Caches results keyed by (schema_id, envelope_hash) to avoid redundant validation.
    Note: schema_id may include fragments (e.g., #version/1.2.3).
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
        canonical = json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        envelope_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]
        return f"{schema_id}:{envelope_hash}"

    def get(self, schema_id: str, envelope: EnvelopeDict) -> Optional[bool]:
        key = self._envelope_to_key(schema_id, envelope)
        with self._lock:
            if key in self._cache:
                # Refresh LRU
                try:
                    self._access_order.remove(key)
                except ValueError:
                    # Defensive: reconstruct if order drifted
                    self._access_order = [k for k in self._access_order if k in self._cache]
                self._access_order.append(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def set(self, schema_id: str, envelope: EnvelopeDict, valid: bool) -> None:
        key = self._envelope_to_key(schema_id, envelope)
        with self._lock:
            if key in self._cache:
                try:
                    self._access_order.remove(key)
                except ValueError:
                    pass
            elif len(self._cache) >= self._max_size:
                # Evict oldest existing key
                while self._access_order:
                    oldest = self._access_order.pop(0)
                    if oldest in self._cache:
                        del self._cache[oldest]
                        break
            self._cache[key] = valid
            self._access_order.append(key)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._hits = 0
            self._misses = 0

    @property
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
            }


_schema_cache = SchemaValidationCache()


def get_schema_cache() -> SchemaValidationCache:
    return _schema_cache


# ---------------------------------------------------------------------------
# Envelope Structure Validation (SCHEMA.md-aligned)
# ---------------------------------------------------------------------------

def validate_envelope_shape(envelope: Any, case_id: Optional[str] = None) -> None:
    """
    Validate envelope has correct top-level structure per common/envelope.request.json.

    Requires: op, ctx, args
    Types: envelope dict; ctx dict; args dict
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

    # ctx + args must be objects on the wire
    if not isinstance(envelope.get("ctx"), dict):
        raise EnvelopeTypeError(
            f"'ctx' must be object, got {type(envelope.get('ctx')).__name__}",
            case_id=case_id,
            field="ctx",
        )
    if not isinstance(envelope.get("args"), dict):
        raise EnvelopeTypeError(
            f"'args' must be object, got {type(envelope.get('args')).__name__}",
            case_id=case_id,
            field="args",
        )

    # Request envelope allows additionalProperties:true (extra keys are permitted)
    extra = set(envelope.keys()) - REQUIRED_ENVELOPE_KEYS
    if extra:
        logger.debug(f"Envelope has extra top-level keys (allowed): {sorted(extra)}")


def validate_op_field(envelope: EnvelopeDict, expected_op: str, case_id: Optional[str] = None) -> None:
    """
    Validate 'op' field matches expected operation.

    Note: common/envelope.request.json only enforces string; op-specific request schemas enforce const.
    This equality check enforces operation conformance.
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


def validate_ctx_field(envelope: EnvelopeDict, case_id: Optional[str] = None) -> None:
    """
    Validate 'ctx' per SCHEMA.md common/operation_context.json.

    All fields are optional and nullable:
      - request_id: string|null
      - idempotency_key: string|null
      - deadline_ms: integer|null, minimum 0
      - traceparent: string|null
      - tenant: string|null
      - attrs: object|null (additionalProperties: true)

    Unknown ctx keys are permitted (additionalProperties: true).
    """
    ctx = envelope["ctx"]
    if not isinstance(ctx, dict):
        raise EnvelopeTypeError(
            f"'ctx' must be object, got {type(ctx).__name__}",
            case_id=case_id,
            field="ctx",
        )

    # request_id: string|null (optional)
    if "request_id" in ctx:
        _validate_optional_string(ctx.get("request_id"), "ctx.request_id", case_id)
        rid = ctx.get("request_id")
        if isinstance(rid, str):
            if len(rid) < POLICY_MIN_REQUEST_ID_LENGTH:
                _policy_violation(
                    f"'ctx.request_id' length below policy minimum {POLICY_MIN_REQUEST_ID_LENGTH} (got {len(rid)})",
                    case_id=case_id,
                    field="ctx.request_id",
                    details={"length": len(rid)},
                )
            if len(rid) > POLICY_MAX_REQUEST_ID_LENGTH:
                _policy_violation(
                    f"'ctx.request_id' length exceeds policy maximum {POLICY_MAX_REQUEST_ID_LENGTH} (got {len(rid)})",
                    case_id=case_id,
                    field="ctx.request_id",
                    details={"length": len(rid)},
                )

    # idempotency_key: string|null (optional)
    if "idempotency_key" in ctx:
        _validate_optional_string(ctx.get("idempotency_key"), "ctx.idempotency_key", case_id)

    # deadline_ms: integer|null, minimum 0 (optional)
    if "deadline_ms" in ctx:
        deadline = ctx.get("deadline_ms")
        if deadline is not None and not isinstance(deadline, int):
            raise EnvelopeTypeError(
                f"'ctx.deadline_ms' must be integer|null, got {type(deadline).__name__}",
                case_id=case_id,
                field="ctx.deadline_ms",
            )
        if isinstance(deadline, int) and deadline < 0:
            raise CtxValidationError(
                "'ctx.deadline_ms' must be >= 0",
                case_id=case_id,
                field="ctx.deadline_ms",
                details={"value": deadline},
            )

    # traceparent: string|null (optional)
    if "traceparent" in ctx:
        _validate_optional_string(ctx.get("traceparent"), "ctx.traceparent", case_id)

    # tenant: string|null (optional)
    if "tenant" in ctx:
        _validate_optional_string(ctx.get("tenant"), "ctx.tenant", case_id)

    # attrs: object|null (optional)
    if "attrs" in ctx:
        attrs = ctx.get("attrs")
        if attrs is not None and not isinstance(attrs, dict):
            raise EnvelopeTypeError(
                f"'ctx.attrs' must be object|null, got {type(attrs).__name__}",
                case_id=case_id,
                field="ctx.attrs",
            )

    unknown = set(ctx.keys()) - KNOWN_CTX_KEYS
    if unknown:
        logger.debug(f"ctx has extension fields (allowed): {sorted(unknown)}")


def _validate_optional_string(value: Any, field_name: str, case_id: Optional[str]) -> None:
    """Validate string|null."""
    if value is None:
        return
    if not isinstance(value, str):
        raise EnvelopeTypeError(
            f"'{field_name}' must be string|null, got {type(value).__name__}",
            case_id=case_id,
            field=field_name,
        )


def validate_args_field(envelope: EnvelopeDict, case_id: Optional[str] = None) -> None:
    """Validate args is an object."""
    args = envelope["args"]
    if not isinstance(args, dict):
        raise EnvelopeTypeError(
            f"'args' must be object, got {type(args).__name__}",
            case_id=case_id,
            field="args",
        )


def validate_envelope_common(envelope: EnvelopeDict, expected_op: str, case_id: Optional[str] = None) -> None:
    """Run common envelope validations (shape + op conformance + ctx types + args type)."""
    validate_envelope_shape(envelope, case_id)
    validate_op_field(envelope, expected_op, case_id)
    validate_ctx_field(envelope, case_id)
    validate_args_field(envelope, case_id)


# ---------------------------------------------------------------------------
# JSON Round-Trip Validation
# ---------------------------------------------------------------------------

def json_roundtrip(envelope: EnvelopeDict, case_id: Optional[str] = None, skip_if_disabled: bool = True) -> EnvelopeDict:
    """
    Force JSON serialization round-trip to validate wire format.

    When disabled, still ensures serializability without a full decode/encode cycle.
    """
    if skip_if_disabled and not CONFIG.enable_json_roundtrip:
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
        payload = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as e:
        raise SerializationError(
            f"JSON serialization failed: {e}",
            case_id=case_id,
            details={"error": str(e)},
        )

    try:
        roundtripped = json.loads(payload)
        if not isinstance(roundtripped, dict):
            raise SerializationError(
                "JSON round-trip did not produce an object envelope",
                case_id=case_id,
                details={"actual_type": type(roundtripped).__name__},
            )
        return roundtripped
    except json.JSONDecodeError as e:
        raise SerializationError(
            f"JSON deserialization failed: {e}",
            case_id=case_id,
            details={"error": str(e), "payload_preview": payload[:200]},
        )


def assert_roundtrip_equality(original: EnvelopeDict, roundtripped: EnvelopeDict, case_id: Optional[str] = None) -> None:
    """Assert envelope wasn't mutated by serialization."""
    if roundtripped != original:
        diff = _find_dict_diff(original, roundtripped)
        raise SerializationError(
            "Envelope mutated by JSON round-trip",
            case_id=case_id,
            details={"diff": diff},
        )


def _find_dict_diff(original: Dict[str, Any], modified: Dict[str, Any], path: str = "") -> List[str]:
    diffs: List[str] = []
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
# Schema Validation (STRICT conformance + SCHEMA.md version tolerance)
# ---------------------------------------------------------------------------

def _schema_registry_available() -> bool:
    """Check if schema registry module can be imported."""
    try:
        import tests.utils.schema_registry  # noqa: F401
        return True
    except Exception:
        return False


def validate_against_schema(
    schema_id: str,
    envelope: EnvelopeDict,
    case_id: Optional[str] = None,
    use_cache: bool = True,
) -> None:
    """
    Validate envelope against JSON Schema using the schema registry.

    Conformance posture:
      - STRICT: missing registry => FAIL
      - BEST_EFFORT: missing registry => WARN + continue
    """
    if use_cache:
        cached = _schema_cache.get(schema_id, envelope)
        if cached is True:
            return
        if cached is False:
            raise SchemaValidationError(
                "Schema validation failed (cached)",
                case_id=case_id,
                details={"schema_id": schema_id},
            )

    try:
        from tests.utils.schema_registry import assert_valid  # type: ignore

        assert_valid(schema_id, envelope, context=f"wire:{case_id or 'unknown'}")
        if use_cache:
            _schema_cache.set(schema_id, envelope, True)

    except ImportError as e:
        # Registry missing: strict vs best_effort behavior
        if CONFIG.schema_validation_mode == "strict":
            raise SchemaValidationError(
                "Schema registry not available in STRICT mode",
                case_id=case_id,
                details={"schema_id": schema_id, "import_error": str(e)},
            ) from e
        logger.warning("Schema registry not available; skipping schema validation (best_effort mode)")
        return

    except Exception as e:
        if use_cache:
            _schema_cache.set(schema_id, envelope, False)
        raise SchemaValidationError(
            f"Schema validation failed: {e}",
            case_id=case_id,
            details={"schema_id": schema_id, "error": str(e)},
        )


def _build_versioned_schema_id(base_schema_id: str, version: str) -> str:
    """
    Build SCHEMA.md version-tolerant schema id: <base>#version/<semver>.

    Accepts:
      - semver: "1.2.3" -> "#version/1.2.3"
      - already-prefixed: "version/1.2.3" -> "#version/1.2.3"
      - already-fragmented: "<base>#version/1.2.3" -> returned as-is
    """
    if "#" in base_schema_id:
        # If caller already passed a fragment, do not rewrite the base.
        # This keeps behavior explicit.
        return base_schema_id

    v = version.strip()
    if v.startswith(_VERSION_FRAGMENT_PREFIX):
        frag = v
    else:
        frag = f"{_VERSION_FRAGMENT_PREFIX}{v}"
    return f"{base_schema_id}#{frag}"


def validate_with_version_tolerance(
    envelope: EnvelopeDict,
    primary_schema_id: str,
    accepted_versions: Tuple[str, ...],
    case_id: Optional[str] = None,
) -> None:
    """
    Validate against schema with SCHEMA.md version tolerance.

    Modes:
      - strict: validate only primary_schema_id
      - tolerant: validate primary, then try primary#version/<semver> in order
      - warn: same as tolerant + logs when fallback version succeeds

    Notes:
      - This relies on the schema registry/resolution layer supporting the SCHEMA.md
        fragment convention (#version/<semver>).
      - accepted_versions should be semver strings (e.g., "1.2.3") or "version/1.2.3".
    """
    mode = CONFIG.schema_version_tolerance
    if mode == "strict":
        validate_against_schema(primary_schema_id, envelope, case_id)
        return

    # Always try primary first
    primary_error: Optional[SchemaValidationError] = None
    try:
        validate_against_schema(primary_schema_id, envelope, case_id)
        return
    except SchemaValidationError as e:
        primary_error = e

    # If no accepted versions provided, fail with primary error
    if not accepted_versions:
        raise primary_error

    # Try versioned schema ids in order
    errors: List[Tuple[str, str]] = [("primary", str(primary_error))]
    for ver in accepted_versions:
        v = (ver or "").strip()
        if not v:
            continue

        # Permit passing a fully-qualified schema_id directly (rare but explicit)
        if v.startswith(("http://", "https://")):
            candidate = v
        else:
            # Validate semver format when possible; if it doesn't match, still attempt (spec may allow)
            if not v.startswith(_VERSION_FRAGMENT_PREFIX) and _SEMVER_RE.match(v) is None:
                logger.debug(f"{case_id}: accepted_versions entry {v!r} is not strict semver; attempting anyway")
            candidate = _build_versioned_schema_id(primary_schema_id, v)

        try:
            validate_against_schema(candidate, envelope, case_id)
            if mode == "warn":
                logger.warning(f"{case_id}: validated using SCHEMA.md version fallback {candidate}")
            return
        except SchemaValidationError as e:
            errors.append((candidate, str(e)))

    raise SchemaValidationError(
        "Validation failed against primary schema and all SCHEMA.md version fallbacks",
        case_id=case_id,
        details={"primary_schema_id": primary_schema_id, "accepted_versions": list(accepted_versions), "errors": errors},
    )


# ---------------------------------------------------------------------------
# Operation-Specific Args Validators (schema-aligned, lightweight)
# ---------------------------------------------------------------------------

ArgsValidator = Callable[[ArgsDict, Optional[str]], None]


# -------------------------- LLM -------------------------- #

def validate_llm_complete_args(args: ArgsDict, case_id: Optional[str] = None) -> None:
    """
    Lightweight checks aligned to SCHEMA.md llm.complete args shape.

    Note: strictness (additionalProperties, nested type correctness, etc.) is enforced by schema validation.
    """
    _require_key(args, "messages", "args.messages", case_id)
    messages = args["messages"]
    if not isinstance(messages, list) or len(messages) < 1:
        raise ArgsValidationError("'args.messages' must be a non-empty array", case_id=case_id, field="args.messages")

    for i, msg in enumerate(messages):
        _validate_llm_message(msg, i, case_id)

    if "max_tokens" in args and args["max_tokens"] is not None:
        _validate_int_min(args["max_tokens"], 0, "args.max_tokens", case_id)

    if "temperature" in args and args["temperature"] is not None:
        _validate_number_range(args["temperature"], 0.0, 2.0, "args.temperature", case_id)

    if "stop_sequences" in args and args["stop_sequences"] is not None:
        _validate_string_array(args["stop_sequences"], "args.stop_sequences", case_id)

    _policy_check_large_strings(args, case_id)


def validate_llm_stream_args(args: ArgsDict, case_id: Optional[str] = None) -> None:
    """llm.stream args match llm.complete shape; schema enforces op-specifics."""
    validate_llm_complete_args(args, case_id)


def validate_llm_count_tokens_args(args: ArgsDict, case_id: Optional[str] = None) -> None:
    _require_key(args, "text", "args.text", case_id)
    _validate_string(args["text"], "args.text", case_id)
    _policy_check_large_strings(args, case_id)


# -------------------------- Vector -------------------------- #

def validate_vector_query_args(args: ArgsDict, case_id: Optional[str] = None) -> None:
    _require_key(args, "vector", "args.vector", case_id)
    _require_key(args, "top_k", "args.top_k", case_id)

    _validate_number_array(args["vector"], "args.vector", case_id, min_items=1)
    _validate_int_min(args["top_k"], 1, "args.top_k", case_id)

    if isinstance(args.get("top_k"), int) and args["top_k"] > POLICY_MAX_TOP_K:
        _policy_violation(
            f"'args.top_k' exceeds policy maximum {POLICY_MAX_TOP_K} (got {args['top_k']})",
            case_id=case_id,
            field="args.top_k",
            details={"value": args["top_k"]},
        )


def validate_vector_batch_query_args(args: ArgsDict, case_id: Optional[str] = None) -> None:
    _require_key(args, "queries", "args.queries", case_id)
    queries = args["queries"]
    if not isinstance(queries, list) or len(queries) < 1:
        raise ArgsValidationError("'args.queries' must be non-empty array", case_id=case_id, field="args.queries")
    for i, q in enumerate(queries):
        if not isinstance(q, dict):
            raise ArgsValidationError(f"'args.queries[{i}]' must be object", case_id=case_id, field=f"args.queries[{i}]")
        validate_vector_query_args(q, case_id=case_id)


def validate_vector_upsert_args(args: ArgsDict, case_id: Optional[str] = None) -> None:
    _require_key(args, "vectors", "args.vectors", case_id)
    vectors = args["vectors"]
    if not isinstance(vectors, list) or len(vectors) < 1:
        raise ArgsValidationError("'args.vectors' must be non-empty array", case_id=case_id, field="args.vectors")

    if len(vectors) > POLICY_MAX_BATCH_SIZE:
        _policy_violation(
            f"'args.vectors' exceeds policy max batch size {POLICY_MAX_BATCH_SIZE} (got {len(vectors)})",
            case_id=case_id,
            field="args.vectors",
            details={"count": len(vectors)},
        )

    # Dimension consistency is NOT schema-required (per SCHEMA.md). Keep as policy-gated.
    ref_dim: Optional[int] = None

    for i, vrec in enumerate(vectors):
        if not isinstance(vrec, dict):
            raise ArgsValidationError(f"'args.vectors[{i}]' must be object", case_id=case_id, field=f"args.vectors[{i}]")
        _require_key(vrec, "id", f"args.vectors[{i}].id", case_id)
        _require_key(vrec, "vector", f"args.vectors[{i}].vector", case_id)

        _validate_nonempty_string(vrec["id"], f"args.vectors[{i}].id", case_id)
        _validate_number_array(vrec["vector"], f"args.vectors[{i}].vector", case_id, min_items=1)

        dim = len(vrec["vector"]) if isinstance(vrec["vector"], list) else None
        if isinstance(dim, int):
            if ref_dim is None:
                ref_dim = dim
            elif dim != ref_dim:
                _policy_violation(
                    f"Vector dimension mismatch in batch: expected {ref_dim}, got {dim}",
                    case_id=case_id,
                    field=f"args.vectors[{i}].vector",
                    details={"expected": ref_dim, "actual": dim},
                )

    _policy_check_large_strings(args, case_id)


def validate_vector_delete_args(args: ArgsDict, case_id: Optional[str] = None) -> None:
    has_ids = "ids" in args
    has_filter = "filter" in args
    if not has_ids and not has_filter:
        raise ArgsValidationError("vector.delete requires 'ids' or 'filter'", case_id=case_id, field="args")

    if has_ids:
        ids = args.get("ids")
        if not isinstance(ids, list) or len(ids) < 1 or not all(isinstance(x, str) and x.strip() for x in ids):
            raise ArgsValidationError("'args.ids' must be non-empty array of non-empty strings", case_id=case_id, field="args.ids")

    if has_filter:
        flt = args.get("filter")
        if not isinstance(flt, dict) or len(flt) < 1:
            raise ArgsValidationError("'args.filter' must be object with at least 1 key", case_id=case_id, field="args.filter")


def validate_vector_namespace_args(args: ArgsDict, case_id: Optional[str] = None) -> None:
    _require_key(args, "namespace", "args.namespace", case_id)
    _validate_nonempty_string(args["namespace"], "args.namespace", case_id)
    if "dimensions" in args and args["dimensions"] is not None:
        _validate_int_min(args["dimensions"], 1, "args.dimensions", case_id)


# -------------------------- Embedding -------------------------- #

def validate_embedding_embed_args(args: ArgsDict, case_id: Optional[str] = None) -> None:
    _require_key(args, "text", "args.text", case_id)
    _require_key(args, "model", "args.model", case_id)
    _validate_nonempty_string(args["text"], "args.text", case_id)
    _validate_nonempty_string(args["model"], "args.model", case_id)

    # Unary embed: stream must be absent or false
    if "stream" in args and args["stream"] is not False:
        raise ArgsValidationError(
            "embedding.embed must have stream absent or false (streaming uses embedding.stream_embed)",
            case_id=case_id,
            field="args.stream",
        )
    _policy_check_large_strings(args, case_id)


def validate_embedding_embed_batch_args(args: ArgsDict, case_id: Optional[str] = None) -> None:
    _require_key(args, "texts", "args.texts", case_id)
    _require_key(args, "model", "args.model", case_id)

    texts = args["texts"]
    if not isinstance(texts, list) or len(texts) < 1 or not all(isinstance(t, str) and t.strip() for t in texts):
        raise ArgsValidationError("'args.texts' must be non-empty array of non-empty strings", case_id=case_id, field="args.texts")

    if len(texts) > POLICY_MAX_BATCH_SIZE:
        _policy_violation(
            f"'args.texts' exceeds policy max batch size {POLICY_MAX_BATCH_SIZE} (got {len(texts)})",
            case_id=case_id,
            field="args.texts",
            details={"count": len(texts)},
        )

    _validate_nonempty_string(args["model"], "args.model", case_id)
    _policy_check_large_strings(args, case_id)


def validate_embedding_stream_embed_args(args: ArgsDict, case_id: Optional[str] = None) -> None:
    _require_key(args, "text", "args.text", case_id)
    _require_key(args, "model", "args.model", case_id)
    _validate_nonempty_string(args["text"], "args.text", case_id)
    _validate_nonempty_string(args["model"], "args.model", case_id)
    _policy_check_large_strings(args, case_id)


def validate_embedding_count_tokens_args(args: ArgsDict, case_id: Optional[str] = None) -> None:
    _require_key(args, "text", "args.text", case_id)
    _require_key(args, "model", "args.model", case_id)
    _validate_string(args["text"], "args.text", case_id)
    _validate_nonempty_string(args["model"], "args.model", case_id)
    _policy_check_large_strings(args, case_id)


def validate_embedding_get_stats_args(args: ArgsDict, case_id: Optional[str] = None) -> None:
    _policy_check_large_strings(args, case_id)


# -------------------------- Graph -------------------------- #

def validate_graph_query_args(args: ArgsDict, case_id: Optional[str] = None) -> None:
    _require_key(args, "text", "args.text", case_id)
    _validate_nonempty_string(args["text"], "args.text", case_id)
    _policy_check_large_strings(args, case_id)


def validate_graph_upsert_nodes_args(args: ArgsDict, case_id: Optional[str] = None) -> None:
    _require_key(args, "nodes", "args.nodes", case_id)
    nodes = args["nodes"]
    if not isinstance(nodes, list) or len(nodes) < 1:
        raise ArgsValidationError("'args.nodes' must be non-empty array", case_id=case_id, field="args.nodes")


def validate_graph_upsert_edges_args(args: ArgsDict, case_id: Optional[str] = None) -> None:
    _require_key(args, "edges", "args.edges", case_id)
    edges = args["edges"]
    if not isinstance(edges, list) or len(edges) < 1:
        raise ArgsValidationError("'args.edges' must be non-empty array", case_id=case_id, field="args.edges")


def validate_graph_delete_nodes_args(args: ArgsDict, case_id: Optional[str] = None) -> None:
    _validate_ids_or_filter(args, case_id, ids_field="ids", filter_field="filter")


def validate_graph_delete_edges_args(args: ArgsDict, case_id: Optional[str] = None) -> None:
    _validate_ids_or_filter(args, case_id, ids_field="ids", filter_field="filter")


def validate_graph_bulk_vertices_args(args: ArgsDict, case_id: Optional[str] = None) -> None:
    # Keep lightweight; schema enforces strictness.
    if "limit" in args and args["limit"] is not None:
        _validate_int_min(args["limit"], 1, "args.limit", case_id)


def validate_graph_batch_args(args: ArgsDict, case_id: Optional[str] = None) -> None:
    _require_key(args, "ops", "args.ops", case_id)
    ops = args["ops"]
    if not isinstance(ops, list) or len(ops) < 1:
        raise ArgsValidationError("'args.ops' must be non-empty array", case_id=case_id, field="args.ops")
    for i, op in enumerate(ops):
        if not isinstance(op, dict):
            raise ArgsValidationError(f"'args.ops[{i}]' must be object", case_id=case_id, field=f"args.ops[{i}]")
        _require_key(op, "op", f"args.ops[{i}].op", case_id)
        _require_key(op, "args", f"args.ops[{i}].args", case_id)
        _validate_nonempty_string(op["op"], f"args.ops[{i}].op", case_id)
        if not isinstance(op["args"], dict):
            raise ArgsValidationError(f"'args.ops[{i}].args' must be object", case_id=case_id, field=f"args.ops[{i}].args")


# ---------------------------------------------------------------------------
# Helper Validators (shared)
# ---------------------------------------------------------------------------

def _require_key(obj: Dict[str, Any], key: str, field_name: str, case_id: Optional[str]) -> None:
    if key not in obj:
        raise ArgsValidationError(f"Missing required field '{field_name}'", case_id=case_id, field=field_name)


def _validate_int_min(value: Any, minimum: int, field_name: str, case_id: Optional[str]) -> None:
    if not isinstance(value, int):
        raise ArgsValidationError(f"'{field_name}' must be integer", case_id=case_id, field=field_name)
    if value < minimum:
        raise ArgsValidationError(f"'{field_name}' must be >= {minimum}", case_id=case_id, field=field_name)


def _validate_number_range(value: Any, lo: float, hi: float, field_name: str, case_id: Optional[str]) -> None:
    if not isinstance(value, (int, float)):
        raise ArgsValidationError(f"'{field_name}' must be number", case_id=case_id, field=field_name)
    v = float(value)
    if v < lo or v > hi:
        raise ArgsValidationError(f"'{field_name}' must be in [{lo}, {hi}]", case_id=case_id, field=field_name)


def _validate_string(value: Any, field_name: str, case_id: Optional[str]) -> None:
    if not isinstance(value, str):
        raise ArgsValidationError(f"'{field_name}' must be string", case_id=case_id, field=field_name)


def _validate_nonempty_string(value: Any, field_name: str, case_id: Optional[str]) -> None:
    _validate_string(value, field_name, case_id)
    if not value.strip():
        raise ArgsValidationError(f"'{field_name}' must be non-empty", case_id=case_id, field=field_name)


def _validate_string_array(value: Any, field_name: str, case_id: Optional[str]) -> None:
    if not isinstance(value, list):
        raise ArgsValidationError(f"'{field_name}' must be array[string]", case_id=case_id, field=field_name)
    if not all(isinstance(x, str) for x in value):
        raise ArgsValidationError(f"'{field_name}' must contain only strings", case_id=case_id, field=field_name)


def _validate_number_array(value: Any, field_name: str, case_id: Optional[str], *, min_items: int = 0) -> None:
    if not isinstance(value, list):
        raise ArgsValidationError(f"'{field_name}' must be array[number]", case_id=case_id, field=field_name)
    if len(value) < min_items:
        raise ArgsValidationError(f"'{field_name}' must have at least {min_items} item(s)", case_id=case_id, field=field_name)
    if not all(isinstance(x, (int, float)) for x in value):
        raise ArgsValidationError(f"'{field_name}' must contain only numbers", case_id=case_id, field=field_name)
    if len(value) > POLICY_MAX_VECTOR_DIMENSIONS:
        _policy_violation(
            f"'{field_name}' exceeds policy maximum dimensions {POLICY_MAX_VECTOR_DIMENSIONS} (got {len(value)})",
            case_id=case_id,
            field=field_name,
            details={"dims": len(value)},
        )


def _validate_ids_or_filter(args: ArgsDict, case_id: Optional[str], *, ids_field: str, filter_field: str) -> None:
    has_ids = ids_field in args
    has_filter = filter_field in args
    if not has_ids and not has_filter:
        raise ArgsValidationError(f"Requires '{ids_field}' or '{filter_field}'", case_id=case_id, field="args")
    if has_ids:
        ids = args.get(ids_field)
        if not isinstance(ids, list) or len(ids) < 1 or not all(isinstance(x, str) and x.strip() for x in ids):
            raise ArgsValidationError(f"'args.{ids_field}' must be non-empty array of non-empty strings", case_id=case_id, field=f"args.{ids_field}")
    if has_filter:
        flt = args.get(filter_field)
        if not isinstance(flt, dict) or len(flt) < 1:
            raise ArgsValidationError(f"'args.{filter_field}' must be object with at least 1 key", case_id=case_id, field=f"args.{filter_field}")


def _validate_llm_message(msg: Any, index: int, case_id: Optional[str]) -> None:
    field_prefix = f"args.messages[{index}]"
    if not isinstance(msg, dict):
        raise ArgsValidationError(f"'{field_prefix}' must be object", case_id=case_id, field=field_prefix)
    if "role" not in msg:
        raise ArgsValidationError(f"'{field_prefix}.role' is required", case_id=case_id, field=f"{field_prefix}.role")
    if "content" not in msg:
        raise ArgsValidationError(f"'{field_prefix}.content' is required", case_id=case_id, field=f"{field_prefix}.content")

    if not isinstance(msg["role"], str):
        raise ArgsValidationError(f"'{field_prefix}.role' must be string", case_id=case_id, field=f"{field_prefix}.role")
    if not isinstance(msg["content"], str):
        raise ArgsValidationError(f"'{field_prefix}.content' must be string", case_id=case_id, field=f"{field_prefix}.content")

    # Role enums may be schema-defined or left as string in SCHEMA.md; keep non-standard roles policy-only.
    if msg["role"] not in {"system", "user", "assistant", "tool"}:
        _policy_violation(
            f"'{field_prefix}.role' is non-standard role {msg['role']!r}",
            case_id=case_id,
            field=f"{field_prefix}.role",
            details={"role": msg["role"]},
        )


def _policy_check_large_strings(args: ArgsDict, case_id: Optional[str]) -> None:
    """Warn/enforce on unexpectedly large strings without breaking schema conformance by default."""
    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, str) and len(v) > POLICY_MAX_TEXT_LENGTH:
                    _policy_violation(
                        f"String field '{k}' exceeds policy max length {POLICY_MAX_TEXT_LENGTH} (got {len(v)})",
                        case_id=case_id,
                        field=k,
                        details={"length": len(v)},
                    )
                else:
                    walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)
    walk(args)


# ---------------------------------------------------------------------------
# Validator Registry
# ---------------------------------------------------------------------------

ARGS_VALIDATORS: Dict[str, ArgsValidator] = {
    # LLM
    "validate_llm_complete_args": validate_llm_complete_args,
    "validate_llm_stream_args": validate_llm_stream_args,
    "validate_llm_count_tokens_args": validate_llm_count_tokens_args,
    # Vector
    "validate_vector_query_args": validate_vector_query_args,
    "validate_vector_batch_query_args": validate_vector_batch_query_args,
    "validate_vector_upsert_args": validate_vector_upsert_args,
    "validate_vector_delete_args": validate_vector_delete_args,
    "validate_vector_namespace_args": validate_vector_namespace_args,
    # Embedding
    "validate_embedding_embed_args": validate_embedding_embed_args,
    "validate_embedding_embed_batch_args": validate_embedding_embed_batch_args,
    "validate_embedding_stream_embed_args": validate_embedding_stream_embed_args,
    "validate_embedding_count_tokens_args": validate_embedding_count_tokens_args,
    "validate_embedding_get_stats_args": validate_embedding_get_stats_args,
    # Graph
    "validate_graph_query_args": validate_graph_query_args,
    "validate_graph_upsert_nodes_args": validate_graph_upsert_nodes_args,
    "validate_graph_upsert_edges_args": validate_graph_upsert_edges_args,
    "validate_graph_delete_nodes_args": validate_graph_delete_nodes_args,
    "validate_graph_delete_edges_args": validate_graph_delete_edges_args,
    "validate_graph_bulk_vertices_args": validate_graph_bulk_vertices_args,
    "validate_graph_batch_args": validate_graph_batch_args,
}


def get_args_validator(name: str) -> Optional[ArgsValidator]:
    return ARGS_VALIDATORS.get(name)


def validate_args_for_operation(args: ArgsDict, validator_name: Optional[str], case_id: Optional[str] = None) -> None:
    """
    Run operation-specific args validation if validator is defined.

    Unknown validator_name => warn + continue (schema is authoritative in STRICT mode).
    """
    if validator_name is None:
        return
    validator = get_args_validator(validator_name)
    if validator is None:
        logger.warning(f"Unknown args validator: {validator_name}")
        return
    validator(args, case_id)


# ---------------------------------------------------------------------------
# Coverage gates (schema request ops <-> tests/live/wire_cases.py)
# ---------------------------------------------------------------------------

def list_request_operation_schema_ids() -> List[str]:
    """
    Enumerate request operation schema IDs from the loaded schema registry.

    Filters out:
      - common/*
      - type schemas (*.types.*)
      - envelope request schemas (*.envelope.request.json)

    This is intentionally conservative and relies on SCHEMA.md naming conventions:
      <component>/<component>.<op>.request.json
    """
    try:
        from tests.utils.schema_registry import list_schemas  # type: ignore
    except ImportError as e:
        if CONFIG.schema_validation_mode == "strict":
            raise SchemaValidationError(
                "Schema registry not available in STRICT mode (cannot enumerate schemas)",
                details={"import_error": str(e)},
            ) from e
        logger.warning("Schema registry not available; cannot enumerate request op schemas (best_effort mode)")
        return []

    registry = list_schemas()  # {schema_id: file_path}
    out: List[str] = []
    for sid in registry.keys():
        if "/common/" in sid:
            continue
        if ".types." in sid:
            continue
        if sid.endswith(".envelope.request.json"):
            continue
        if sid.endswith(".request.json"):
            out.append(sid)
    return sorted(out)


def _schema_id_to_request_op(schema_id: str) -> Optional[str]:
    """
    Convert an operation request schema_id into an op string.

    Expected filename: <component>.<op>.request.json
      e.g. https://.../llm/llm.complete.request.json -> "llm.complete"

    Returns None for non-op schemas (envelopes, types, common, non-request, etc.).
    """
    if not schema_id.endswith(".request.json"):
        return None
    if schema_id.endswith(".envelope.request.json"):
        return None
    if "/common/" in schema_id:
        return None
    if ".types." in schema_id:
        return None

    fname = schema_id.rsplit("/", 1)[-1]
    if not fname.endswith(".request.json"):
        return None
    op = fname[: -len(".request.json")]
    return op or None


def assert_all_wire_case_args_validators_exist() -> None:
    """
    Sanity gate: every args_validator referenced by tests/live/wire_cases.py must exist in ARGS_VALIDATORS.
    """
    try:
        from tests.live.wire_cases import WIRE_REQUEST_CASES  # type: ignore
    except Exception as e:
        raise ValidationError(
            "Failed to import wire cases from tests/live/wire_cases.py",
            details={"error": str(e)},
        ) from e

    missing: List[str] = []
    for c in WIRE_REQUEST_CASES:
        if c.args_validator and c.args_validator not in ARGS_VALIDATORS:
            missing.append(f"{c.id}: {c.args_validator}")

    if missing:
        raise ValidationError(
            "Some wire request cases reference unknown args validators",
            details={"missing": missing, "missing_count": len(missing)},
        )


def assert_all_schema_request_ops_have_cases() -> None:
    """
    Conformance gate: ensure every request op schema has at least one wire case.

    Wire cases are defined in tests/live/wire_cases.py (canonical path).
    """
    # Import wire cases from the canonical wire path.
    try:
        from tests.live.wire_cases import WIRE_REQUEST_CASES  # type: ignore
    except Exception as e:
        raise ValidationError(
            "Failed to import wire cases from tests/live/wire_cases.py",
            details={"error": str(e)},
        ) from e

    covered_ops = {c.op for c in WIRE_REQUEST_CASES}
    request_op_schema_ids = list_request_operation_schema_ids()

    missing_ops: List[str] = []
    for sid in request_op_schema_ids:
        op = _schema_id_to_request_op(sid)
        if op is None:
            continue
        if op not in covered_ops:
            missing_ops.append(f"{op}  (schema_id={sid})")

    if missing_ops:
        raise ValidationError(
            "Some request operation schemas have no wire test cases (coverage drift)",
            details={"missing_ops": missing_ops, "missing_count": len(missing_ops)},
        )


# ---------------------------------------------------------------------------
# Convenience pipeline
# ---------------------------------------------------------------------------

def validate_wire_envelope(
    envelope: EnvelopeDict,
    expected_op: str,
    schema_id: str,
    accepted_versions: Tuple[str, ...] = (),
    args_validator: Optional[str] = None,
    case_id: Optional[str] = None,
) -> EnvelopeDict:
    """
    Complete wire envelope validation pipeline.

    Steps:
      1) Envelope structure validation (SCHEMA.md request envelope + op conformance)
      2) JSON round-trip validation (wire safety)
      3) Schema validation:
           - strict: schema_id only
           - tolerant/warn: schema_id, then schema_id#version/<semver> for accepted_versions
      4) Args validation (lightweight, schema-aligned; schema remains authoritative)

    accepted_versions:
      - SCHEMA.md version tolerance list (semver strings like "1.2.3")
      - used only when CONFIG.schema_version_tolerance != "strict"
    """
    # 1) Envelope validation
    validate_envelope_common(envelope, expected_op, case_id)

    # 2) JSON round-trip
    wire_envelope = json_roundtrip(envelope, case_id)
    if CONFIG.enable_json_roundtrip:
        assert_roundtrip_equality(envelope, wire_envelope, case_id)

    # 3) Schema validation (+ SCHEMA.md version tolerance)
    if CONFIG.schema_version_tolerance == "strict":
        validate_against_schema(schema_id, wire_envelope, case_id)
    else:
        validate_with_version_tolerance(wire_envelope, schema_id, accepted_versions, case_id)

    # 4) Args validation
    validate_args_for_operation(wire_envelope["args"], args_validator, case_id)

    return wire_envelope
