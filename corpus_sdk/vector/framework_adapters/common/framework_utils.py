# corpus_sdk/vector/framework_adapters/common/framework_utils.py
# SPDX-License-Identifier: Apache-2.0
"""
Shared utilities for framework-specific *vector* adapters.

This module centralizes common logic used across all vector-store adapters:

- Coercing provider / runtime results into canonical shapes:
  * Vectors → List[float] / List[List[float]]
  * Hits    → List[Mapping[str, Any]] with normalized fields
- Enforcing vector-level resource limits (top-k, dimensions, total values)
- Emitting consistent, framework-aware warnings
- Validating numeric safety (NaN/Inf, overflow-ish sizes)
- Normalizing vector search context and attaching it to framework_ctx
- Optional streaming helpers for vector event streams

It intentionally stays *framework-neutral* and uses only:

- Standard library types
- Simple, caller-provided error codes and limits
- No direct dependencies on specific vector-store runtimes

Adapters remain responsible for:

- Choosing framework names and passing them in
- Supplying appropriate error code bundles
- Deciding which limits/flags to use per environment
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error codes, limits, validation flags
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VectorCoercionErrorCodes:
    """
    Structured bundle of error codes used during vector coercion / validation.

    These codes are surfaced in exception messages so individual frameworks
    can attach or filter on them in higher-level error handlers.

    Attributes
    ----------
    invalid_vector_result:
        Code used when the vector container or rows are invalid.

    invalid_hit_result:
        Code used when the hit container or hit rows are invalid.

    empty_result:
        Code used when no valid vectors/hits remain after processing.

    conversion_error:
        Code used when value conversion to float fails.

    score_out_of_range:
        Code used when scores/similarities fall outside expected ranges
        during normalization (e.g., distance → similarity).

    vector_dimension_exceeded:
        Code used when vector dimensionality exceeds configured limits.

    vector_norm_invalid:
        Code used when vector values are NaN/inf or non-finite.

    framework_label:
        Default framework label used when no explicit framework name is passed.
    """

    invalid_vector_result: str = "INVALID_VECTOR_RESULT"
    invalid_hit_result: str = "INVALID_VECTOR_HIT_RESULT"
    empty_result: str = "EMPTY_VECTOR_RESULT"
    conversion_error: str = "VECTOR_CONVERSION_ERROR"
    score_out_of_range: str = "VECTOR_SCORE_OUT_OF_RANGE"
    vector_dimension_exceeded: str = "VECTOR_DIMENSION_EXCEEDED"
    vector_norm_invalid: str = "VECTOR_NORM_INVALID"
    framework_label: str = "vector"


@dataclass(frozen=True)
class VectorResourceLimits:
    """
    Resource limits for vector search operations.

    All fields are optional; if None, the limit is effectively disabled.

    Attributes
    ----------
    max_hits_per_query:
        Maximum number of hits returned for a single query.

    max_total_hits:
        Maximum total number of hits across a multi-step operation.

    max_vector_dim:
        Maximum allowed vector dimensionality.

    max_total_vector_values:
        Maximum total number of vector values (rows * dim) across an operation.

    max_total_stream_events:
        Maximum number of events processed by streaming helpers.
    """

    max_hits_per_query: Optional[int] = None
    max_total_hits: Optional[int] = None
    max_vector_dim: Optional[int] = None
    max_total_vector_values: Optional[int] = None
    max_total_stream_events: Optional[int] = 10_000


@dataclass(frozen=True)
class VectorValidationFlags:
    """
    Flags controlling how strict vector validation should be.

    These can be tuned per-adapter or per-environment (dev vs prod).
    """

    validate_ids: bool = True
    validate_scores: bool = True
    validate_vector_dims: bool = True
    enforce_finite_values: bool = True
    normalize_distance_to_similarity: bool = False
    fail_on_limit_exceeded: bool = False
    validate_error_codes: bool = True
    validate_framework_name: bool = True
    strict_stream_limits: bool = False


@dataclass(frozen=True)
class TopKWarningConfig:
    """
    Configuration for top-k warnings.

    Attributes
    ----------
    warn_threshold:
        Soft threshold above which a warning is logged when a very large
        top-k or hit count is requested.

    framework_label:
        Name of the framework used only for logging.
    """

    warn_threshold: int = 10_000
    framework_label: str = "vector"


# ---------------------------------------------------------------------------
# Internal helpers: framework + error code validation
# ---------------------------------------------------------------------------


def _infer_framework_name(
    error_codes: Optional[VectorCoercionErrorCodes],
    framework: Optional[str],
    *,
    source: Optional[Any] = None,
    flags: Optional[VectorValidationFlags] = None,
) -> str:
    """
    Normalize or infer a framework name for logging / diagnostics.

    Priority chain:
    1. Explicit `framework` argument (if valid).
    2. `error_codes.framework_label` (if present and non-empty).
    3. Class-name inference from `source` (e.g. adapter instance).
    4. Fallback to "vector".

    If validation is enabled and the explicit framework is invalid, raises
    ValueError. Everything else is best-effort.
    """
    flags = flags or VectorValidationFlags()

    # 1) Explicit framework
    if framework is not None:
        value = str(framework).strip()
        if not value:
            if flags.validate_framework_name:
                raise ValueError("framework must be a non-empty string when provided")
            return "vector"
        return value.lower()

    # 2) error_codes.framework_label
    if error_codes is not None:
        label = getattr(error_codes, "framework_label", None)
        if isinstance(label, str) and label.strip():
            return label.strip().lower()

    # 3) Class-name inference from source
    if source is not None:
        try:
            cls = getattr(source, "__class__", type(source))
            name = getattr(cls, "__name__", "") or ""
            if name:
                return name.lower()
        except Exception:  # noqa: BLE001
            # Best-effort only; ignore failures.
            pass

    # 4) Fallback
    return "vector"


def _validate_error_codes(
    error_codes: VectorCoercionErrorCodes,
    *,
    logger: Optional[logging.Logger] = None,
    flags: Optional[VectorValidationFlags] = None,
) -> None:
    """
    Ensure the provided error code bundle looks structurally valid.

    This is intentionally lightweight and only checks for non-empty strings.
    """
    flags = flags or VectorValidationFlags()
    log = logger or LOG

    if not flags.validate_error_codes:
        return

    required_fields = (
        "invalid_vector_result",
        "invalid_hit_result",
        "empty_result",
        "conversion_error",
        "score_out_of_range",
        "vector_dimension_exceeded",
        "vector_norm_invalid",
    )
    missing: List[str] = []
    empty: List[str] = []

    for field in required_fields:
        if not hasattr(error_codes, field):
            missing.append(field)
            continue
        value = getattr(error_codes, field)
        if not isinstance(value, str) or not value.strip():
            empty.append(field)

    if missing or empty:
        message = (
            f"VectorCoercionErrorCodes is missing fields={missing} "
            f"or has empty fields={empty}"
        )
        if flags.fail_on_limit_exceeded:
            raise TypeError(message)
        log.warning("Invalid error_codes configuration: %s", message)


# ---------------------------------------------------------------------------
# Extraction helpers for vectors / hits
# ---------------------------------------------------------------------------


def _extract_vectors_object(result: Any) -> Any:
    """
    Extract the underlying vectors object from a variety of result shapes.

    Supported shapes:
    - Mapping with "vectors" or "embeddings": {"vectors": [[...], ...], ...}
    - Object with `.vectors` or `.embeddings` attribute
    - Raw list / sequence of vectors: [[...], [...]]
    - Raw single vector: [...]
    """
    if isinstance(result, Mapping):
        if "vectors" in result:
            return result["vectors"]
        if "embeddings" in result:
            return result["embeddings"]

    if hasattr(result, "vectors"):
        return getattr(result, "vectors")
    if hasattr(result, "embeddings"):
        return getattr(result, "embeddings")

    return result


def _extract_vector_object(result: Any) -> Any:
    """
    Extract a single vector object from a result shape.

    Supported shapes:
    - Mapping with "vector" or "embedding": {"vector": [...], ...}
    - Object with `.vector` or `.embedding` attribute
    - Raw vector: [...]
    """
    if isinstance(result, Mapping):
        if "vector" in result:
            return result["vector"]
        if "embedding" in result:
            return result["embedding"]

    if hasattr(result, "vector"):
        return getattr(result, "vector")
    if hasattr(result, "embedding"):
        return getattr(result, "embedding")

    return result


def _extract_hits_object(result: Any) -> Any:
    """
    Extract the underlying hits / matches object from various shapes.

    Supported shapes:
    - Mapping with "hits", "matches", or "results"
    - Object with `.hits` / `.matches` attribute
    - Raw list / sequence of hit-like objects
    """
    if isinstance(result, Mapping):
        if "hits" in result:
            return result["hits"]
        if "matches" in result:
            return result["matches"]
        if "results" in result:
            return result["results"]

    if hasattr(result, "hits"):
        return getattr(result, "hits")
    if hasattr(result, "matches"):
        return getattr(result, "matches")

    return result


def _as_mapping(obj: Any) -> Optional[Mapping[str, Any]]:
    """
    Best-effort conversion of an arbitrary object to a Mapping[str, Any].

    - If already a Mapping → returned as-is
    - If has __dict__ → return that dict
    - Otherwise, returns None and the caller decides how to react
    """
    if isinstance(obj, Mapping):
        return obj
    if hasattr(obj, "__dict__"):
        try:
            return dict(vars(obj))
        except Exception:  # noqa: BLE001
            return None
    return None


# ---------------------------------------------------------------------------
# Vector coercion helpers
# ---------------------------------------------------------------------------


def coerce_vector_matrix(
    result: Any,
    *,
    framework: Optional[str] = None,
    error_codes: VectorCoercionErrorCodes,
    limits: Optional[VectorResourceLimits] = None,
    flags: Optional[VectorValidationFlags] = None,
    logger: Optional[logging.Logger] = None,
) -> List[List[float]]:
    """
    Coerce a generic vector result into a `List[List[float]]` matrix.

    This is the single source of truth for vector coercion logic across
    vector-store adapters.

    Parameters
    ----------
    result:
        Arbitrary result returned by a vector store / translator.
    framework:
        Optional framework label for logging / diagnostics. If None, falls
        back to `error_codes.framework_label` and then "vector".
    error_codes:
        Bundle of error codes to embed in exception messages.
    limits:
        Optional resource limits for dimension / total value caps.
    flags:
        Optional validation flags controlling strictness.
    logger:
        Optional logger; if omitted, the module-level logger is used.

    Returns
    -------
    List[List[float]]
        Canonical vector matrix.

    Raises
    ------
    TypeError, ValueError
        If the result is not structurally valid or exceeds strict limits.
    """
    flags = flags or VectorValidationFlags()
    log = logger or LOG
    framework_name = _infer_framework_name(error_codes, framework, flags=flags)
    _validate_error_codes(error_codes, logger=log, flags=flags)

    vectors_obj = _extract_vectors_object(result)

    # Normalize to matrix-like structure
    if isinstance(vectors_obj, Sequence) and not isinstance(vectors_obj, (str, bytes)):
        if vectors_obj and isinstance(vectors_obj[0], Sequence) and not isinstance(
            vectors_obj[0], (str, bytes),
        ):
            raw_rows = vectors_obj  # matrix
        else:
            # Single vector
            raw_rows = [vectors_obj]
    else:
        raise TypeError(
            f"[{error_codes.invalid_vector_result}] "
            f"{framework_name}: result does not contain a valid vector sequence "
            f"(type={type(vectors_obj).__name__})"
        )

    matrix: List[List[float]] = []
    limits = limits or VectorResourceLimits()
    total_values = 0

    for idx, row in enumerate(raw_rows):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise TypeError(
                f"[{error_codes.invalid_vector_result}] "
                f"{framework_name}: expected each vector row to be a non-string "
                f"sequence, got {type(row).__name__} at index {idx}"
            )

        if len(row) == 0:
            log.warning(
                "%s: empty vector row at index %d, skipping",
                framework_name,
                idx,
            )
            continue

        vector: List[float] = []

        for j, x in enumerate(row):
            try:
                v = float(x)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"[{error_codes.conversion_error}] "
                    f"{framework_name}: failed to convert vector value to float "
                    f"at row={idx}, col={j}: {exc}"
                ) from exc

            if flags.enforce_finite_values and not math.isfinite(v):
                raise TypeError(
                    f"[{error_codes.vector_norm_invalid}] "
                    f"{framework_name}: non-finite vector value at "
                    f"row={idx}, col={j}: {v!r}"
                )

            vector.append(v)

        # Per-vector dimension limits
        if limits.max_vector_dim is not None and flags.validate_vector_dims:
            if len(vector) > limits.max_vector_dim:
                message = (
                    f"[{error_codes.vector_dimension_exceeded}] "
                    f"{framework_name}: vector dimension {len(vector)} at row={idx} "
                    f"exceeds max_vector_dim={limits.max_vector_dim}"
                )
                if flags.fail_on_limit_exceeded:
                    raise ValueError(message)
                log.warning(message)
                vector = vector[: limits.max_vector_dim]

        matrix.append(vector)
        total_values += len(vector)

    if not matrix:
        raise ValueError(
            f"[{error_codes.empty_result}] "
            f"{framework_name}: no valid vectors found in result"
        )

    # Total value limits
    if limits.max_total_vector_values is not None:
        if total_values > limits.max_total_vector_values:
            message = (
                f"[{error_codes.vector_dimension_exceeded}] "
                f"{framework_name}: total vector values={total_values} exceed "
                f"max_total_vector_values={limits.max_total_vector_values}"
            )
            if flags.fail_on_limit_exceeded:
                raise ValueError(message)
            log.warning(message)

            # Truncate matrix to respect total_values limit
            truncated: List[List[float]] = []
            running = 0
            for row in matrix:
                remaining = limits.max_total_vector_values - running
                if remaining <= 0:
                    break
                if len(row) <= remaining:
                    truncated.append(row)
                    running += len(row)
                else:
                    truncated.append(row[:remaining])
                    running += remaining
                    break
            matrix = truncated

    log.debug(
        "%s: successfully coerced vector matrix with %d rows (original_type=%s)",
        framework_name,
        len(matrix),
        type(vectors_obj).__name__,
    )
    return matrix


def coerce_vector(
    result: Any,
    *,
    framework: Optional[str] = None,
    error_codes: VectorCoercionErrorCodes,
    limits: Optional[VectorResourceLimits] = None,
    flags: Optional[VectorValidationFlags] = None,
    logger: Optional[logging.Logger] = None,
) -> List[float]:
    """
    Coerce a generic vector result into a single `List[float]` vector.

    Strategy:
    - Use `coerce_vector_matrix` to normalize the result
    - If the matrix has exactly one row → return that row
    - If it has multiple rows → return the first row and log a warning

    Parameters
    ----------
    result:
        Arbitrary result returned by a vector store / translator.
    framework:
        Optional framework label for logging / diagnostics. If None, falls
        back to `error_codes.framework_label` and then "vector".
    error_codes:
        Bundle of error codes to embed in exception messages.
    limits:
        Optional resource limits for dimension / total value caps.
    flags:
        Optional validation flags controlling strictness.
    logger:
        Optional logger; if omitted, the module-level logger is used.
    """
    flags = flags or VectorValidationFlags()
    log = logger or LOG
    framework_name = _infer_framework_name(error_codes, framework, flags=flags)

    # For single-vector cases, allow direct vector extraction
    vector_obj = _extract_vector_object(result)
    matrix = coerce_vector_matrix(
        vector_obj,
        framework=framework_name,
        error_codes=error_codes,
        limits=limits,
        flags=flags,
        logger=log,
    )

    if len(matrix) > 1:
        log.warning(
            "%s: expected a single vector but received %d rows; using the first row",
            framework_name,
            len(matrix),
        )

    return matrix[0]


# ---------------------------------------------------------------------------
# Hit coercion helpers
# ---------------------------------------------------------------------------


def warn_if_extreme_k(
    k: int,
    *,
    framework: Optional[str] = None,
    op_name: str,
    warning_config: Optional[TopKWarningConfig] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Emit a soft warning if an extremely large top-k / hit count is requested.

    This is used by adapters to surface potentially dangerous hit sizes while
    preserving existing behavior (no hard failures).

    Parameters
    ----------
    k:
        Requested top-k / hit count.
    framework:
        Optional framework name for logging (e.g. "langchain", "llamaindex").
    op_name:
        Name of the operation emitting the warning (e.g. "similarity_search").
    warning_config:
        Optional TopKWarningConfig; if omitted, a default is used.
    logger:
        Optional logger; if omitted, the module-level logger is used.
    """
    log = logger or LOG
    cfg = warning_config or TopKWarningConfig(
        framework_label=(framework or "vector").lower()
    )

    if k <= cfg.warn_threshold:
        return

    log.warning(
        "%s (%s): requested top_k / hit count of %d, which exceeds "
        "warn_threshold=%d; ensure your vector store can handle this safely.",
        cfg.framework_label,
        framework or "vector",
        k,
        cfg.warn_threshold,
    )


def _normalize_hit_id(hit: Mapping[str, Any]) -> Optional[str]:
    """
    Best-effort extraction of a hit's identifier.

    Tries `id`, then `document_id`, then `doc_id`.
    """
    for key in ("id", "document_id", "doc_id"):
        value = hit.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _normalize_hit_score(
    hit: Mapping[str, Any],
    *,
    framework_name: str,
    error_codes: VectorCoercionErrorCodes,
    flags: VectorValidationFlags,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract and normalize score and distance from a hit.

    Returns (score, distance), where:

    - score is a similarity-like scalar (higher is better)
    - distance is the raw distance value (if provided)
    """
    raw_score = hit.get("score")
    similarity = hit.get("similarity")
    distance = hit.get("distance") if "distance" in hit else hit.get("dist")

    def _to_float(value: Any, label: str) -> Optional[float]:
        if value is None:
            return None
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise TypeError(
                f"[{error_codes.conversion_error}] "
                f"{framework_name}: {label} value {value!r} is not numeric"
            )
        if flags.enforce_finite_values and not math.isfinite(v):
            raise TypeError(
                f"[{error_codes.vector_norm_invalid}] "
                f"{framework_name}: {label} value {value!r} is not finite"
            )
        return v

    score_val: Optional[float] = None
    dist_val: Optional[float] = None

    if raw_score is not None:
        score_val = _to_float(raw_score, "score")
    elif similarity is not None:
        score_val = _to_float(similarity, "similarity")

    if distance is not None:
        dist_val = _to_float(distance, "distance")

    # If we have distance but no score, optionally normalize
    if dist_val is not None and score_val is None:
        if flags.normalize_distance_to_similarity:
            # Simple, bounded [0,1] similarity transform
            d = max(dist_val, 0.0)
            sim = 1.0 / (1.0 + d)
            if not 0.0 <= sim <= 1.0:
                raise ValueError(
                    f"[{error_codes.score_out_of_range}] "
                    f"{framework_name}: normalized similarity {sim} out of [0,1]"
                )
            score_val = sim
        else:
            # Fallback: treat negative distance as higher similarity
            score_val = -dist_val

    return score_val, dist_val


def coerce_hits(
    result: Any,
    *,
    framework: Optional[str] = None,
    error_codes: VectorCoercionErrorCodes,
    limits: Optional[VectorResourceLimits] = None,
    flags: Optional[VectorValidationFlags] = None,
    logger: Optional[logging.Logger] = None,
) -> List[Mapping[str, Any]]:
    """
    Coerce a generic vector-search result into a canonical list of hit mappings.

    Canonical hit shape (best-effort):
    - "id": str
    - "score": float (similarity, higher is better)
    - "distance": float (if provided)
    - "vector": List[float] (if provided)
    - "metadata": Mapping[str, Any] (if provided)
    - Other provider-specific fields are preserved as-is.

    Parameters
    ----------
    result:
        Arbitrary result returned by a vector store / translator.
    framework:
        Optional framework label for logging / diagnostics. If None, falls
        back to `error_codes.framework_label` and then "vector".
    error_codes:
        Bundle of error codes to embed in exception messages.
    limits:
        Optional resource limits for hit counts and vector dims.
    flags:
        Optional validation flags controlling strictness.
    logger:
        Optional logger; if omitted, the module-level logger is used.
    """
    flags = flags or VectorValidationFlags()
    log = logger or LOG
    framework_name = _infer_framework_name(error_codes, framework, flags=flags)
    _validate_error_codes(error_codes, logger=log, flags=flags)

    limits = limits or VectorResourceLimits()
    hits_obj = _extract_hits_object(result)

    if hits_obj is None:
        raise TypeError(
            f"[{error_codes.invalid_hit_result}] "
            f"{framework_name}: vector result does not contain hits/matches"
        )

    if isinstance(hits_obj, (str, bytes)) or not isinstance(hits_obj, Sequence):
        raise TypeError(
            f"[{error_codes.invalid_hit_result}] "
            f"{framework_name}: hits container must be a sequence, "
            f"got {type(hits_obj).__name__}"
        )

    hits: List[Mapping[str, Any]] = []

    for idx, raw_hit in enumerate(hits_obj):
        mapping = _as_mapping(raw_hit)
        if mapping is None:
            message = (
                f"[{error_codes.invalid_hit_result}] "
                f"{framework_name}: hit at index {idx} is not mapping-like "
                f"(type={type(raw_hit).__name__})"
            )
            if flags.fail_on_limit_exceeded:
                raise TypeError(message)
            log.warning(message)
            continue

        canonical: Dict[str, Any] = dict(mapping)

        # Normalize id
        if flags.validate_ids:
            hit_id = _normalize_hit_id(mapping)
            if not hit_id:
                message = (
                    f"[{error_codes.invalid_hit_result}] "
                    f"{framework_name}: hit at index {idx} missing valid id"
                )
                if flags.fail_on_limit_exceeded:
                    raise ValueError(message)
                log.warning(message)
            else:
                canonical["id"] = hit_id

        # Normalize score / distance
        if flags.validate_scores:
            score_val, dist_val = _normalize_hit_score(
                mapping,
                framework_name=framework_name,
                error_codes=error_codes,
                flags=flags,
            )
            if score_val is not None:
                canonical["score"] = score_val
            if dist_val is not None:
                canonical["distance"] = dist_val

        # Normalize metadata
        metadata = mapping.get("metadata") or mapping.get("meta", None)
        if metadata is not None and isinstance(metadata, Mapping):
            canonical["metadata"] = dict(metadata)

        # Normalize vector if available
        if "vector" in mapping or "embedding" in mapping:
            try:
                vector = coerce_vector(
                    mapping,
                    framework=framework_name,
                    error_codes=error_codes,
                    limits=limits,
                    flags=flags,
                    logger=log,
                )
                canonical["vector"] = vector
            except Exception as exc:  # noqa: BLE001
                message = (
                    f"[{error_codes.invalid_vector_result}] "
                    f"{framework_name}: failed to coerce vector for hit index {idx}: {exc}"
                )
                if flags.fail_on_limit_exceeded:
                    raise
                log.warning(message)

        hits.append(canonical)

    if not hits:
        raise ValueError(
            f"[{error_codes.empty_result}] "
            f"{framework_name}: no valid hits found in vector result"
        )

    # Per-query hit limits
    if limits.max_hits_per_query is not None and len(hits) > limits.max_hits_per_query:
        message = (
            f"[{error_codes.invalid_hit_result}] "
            f"{framework_name}: hits_per_query={len(hits)} exceeds "
            f"max_hits_per_query={limits.max_hits_per_query}"
        )
        if flags.fail_on_limit_exceeded:
            raise ValueError(message)
        log.warning(message)
        hits = hits[: limits.max_hits_per_query]

    warn_if_extreme_k(
        len(hits),
        framework=framework_name,
        op_name="coerce_hits",
        logger=log,
    )

    return hits


def enforce_hit_limits(
    hits: Sequence[Mapping[str, Any]],
    *,
    framework: Optional[str] = None,
    op_name: str,
    error_codes: VectorCoercionErrorCodes,
    limits: Optional[VectorResourceLimits],
    flags: Optional[VectorValidationFlags] = None,
    logger: Optional[logging.Logger] = None,
) -> List[Mapping[str, Any]]:
    """
    Enforce global hit limits (max_total_hits) on a sequence of hits.

    Returns a truncated list when limits are exceeded (unless configured to fail).

    Parameters
    ----------
    hits:
        Sequence of hit mappings.
    framework:
        Optional framework label for logging / diagnostics. If None, falls
        back to `error_codes.framework_label` and then "vector".
    op_name:
        Name of the operation enforcing the limits.
    error_codes:
        Error code bundle.
    limits:
        Optional VectorResourceLimits; if None, no limits are enforced.
    flags:
        Optional VectorValidationFlags controlling strictness.
    logger:
        Optional logger.
    """
    flags = flags or VectorValidationFlags()
    log = logger or LOG
    framework_name = _infer_framework_name(error_codes, framework, flags=flags)
    _validate_error_codes(error_codes, logger=log, flags=flags)

    limits = limits or VectorResourceLimits()
    hits_list = list(hits)

    if limits.max_total_hits is not None and len(hits_list) > limits.max_total_hits:
        message = (
            f"[{error_codes.invalid_hit_result}] "
            f"{framework_name}: {op_name} total_hits={len(hits_list)} exceeds "
            f"max_total_hits={limits.max_total_hits}"
        )
        if flags.fail_on_limit_exceeded:
            raise ValueError(message)
        log.warning(message)
        hits_list = hits_list[: limits.max_total_hits]

    return hits_list


# ---------------------------------------------------------------------------
# Vector search context helpers
# ---------------------------------------------------------------------------


def normalize_vector_context(
    search_context: Optional[Mapping[str, Any]],
    *,
    framework: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Normalize vector search context into a consistently-shaped dict.

    This helper is tolerant of:
    - Missing or None context
    - Alternate key casing (indexName → index_name, etc.)

    Parameters
    ----------
    search_context:
        Optional mapping with vector-search context.
    framework:
        Optional framework label for logging / diagnostics.
    logger:
        Optional logger.
    """
    log = logger or LOG
    framework_name = (framework or "vector").lower()

    if search_context is None:
        return {}

    if not isinstance(search_context, Mapping):
        log.warning(
            "%s: search_context must be a Mapping, got %s; ignoring context",
            framework_name,
            type(search_context).__name__,
        )
        return {}

    ctx: Dict[str, Any] = {}
    for key, value in search_context.items():
        ctx[key] = value

    def _maybe_move(src: str, dst: str) -> None:
        if src in ctx and dst not in ctx:
            ctx[dst] = ctx[src]

    # Normalize variants
    _maybe_move("indexName", "index_name")
    _maybe_move("collectionName", "collection_name")
    _maybe_move("namespaceId", "namespace")
    _maybe_move("tenantId", "tenant_id")
    _maybe_move("userId", "user_id")
    _maybe_move("requestId", "request_id")
    _maybe_move("traceId", "trace_id")

    return ctx


def attach_vector_context_to_framework_ctx(
    framework_ctx: MutableMapping[str, Any],
    *,
    vector_context: Mapping[str, Any],
    limits: Optional[VectorResourceLimits] = None,
    flags: Optional[VectorValidationFlags] = None,
) -> None:
    """
    Attach normalized vector-search context to a framework-level context dict.

    This helper mutates `framework_ctx` in-place.
    """
    flags = flags or VectorValidationFlags()

    for key in (
        "index_name",
        "collection_name",
        "namespace",
        "tenant_id",
        "user_id",
        "request_id",
        "trace_id",
        "source",
        "pipeline",
        "stage",
    ):
        if key in vector_context:
            framework_ctx[key] = vector_context[key]

    # Optional "batch strategy" hint for downstream batching logic
    index_name = vector_context.get("index_name")
    namespace = vector_context.get("namespace")
    stage = vector_context.get("stage")

    if limits and flags.validate_vector_dims:
        if stage:
            framework_ctx.setdefault("batch_strategy", f"stage_{stage}")
        elif index_name:
            framework_ctx.setdefault("batch_strategy", f"index_{index_name}")
        elif namespace:
            framework_ctx.setdefault("batch_strategy", f"namespace_{namespace}")


# ---------------------------------------------------------------------------
# Streaming / event helpers
# ---------------------------------------------------------------------------


def iter_vector_events(
    events: Iterable[Any],
    *,
    framework: Optional[str] = None,
    op_name: str,
    limits: Optional[VectorResourceLimits] = None,
    flags: Optional[VectorValidationFlags] = None,
    logger: Optional[logging.Logger] = None,
) -> Iterator[Any]:
    """
    Wrap an event stream with basic safety limits.

    - Counts events and enforces `max_total_stream_events` if configured.
    - Logs at most one warning when the limit is exceeded.
    - When `strict_stream_limits` is True, stops iteration after limit.

    Parameters
    ----------
    events:
        Iterable of vector-store events.
    framework:
        Optional framework label for logging / diagnostics.
    op_name:
        Name of the operation consuming the events.
    limits:
        Optional VectorResourceLimits; if None, a default is used.
    flags:
        Optional VectorValidationFlags controlling strictness.
    logger:
        Optional logger.
    """
    flags = flags or VectorValidationFlags()
    log = logger or LOG
    framework_name = (framework or "vector").lower()
    limits = limits or VectorResourceLimits()

    max_events = limits.max_total_stream_events
    count = 0
    warned = False

    for event in events:
        count += 1
        if max_events is not None and count > max_events:
            if not warned:
                log.warning(
                    "%s: %s exceeded max_total_stream_events=%d; "
                    "subsequent events will %s.",
                    framework_name,
                    op_name,
                    max_events,
                    "be dropped" if flags.strict_stream_limits else "still be yielded",
                )
                warned = True
            if flags.strict_stream_limits:
                break
        yield event


__all__ = [
    "VectorCoercionErrorCodes",
    "VectorResourceLimits",
    "VectorValidationFlags",
    "TopKWarningConfig",
    "coerce_vector_matrix",
    "coerce_vector",
    "coerce_hits",
    "enforce_hit_limits",
    "warn_if_extreme_k",
    "normalize_vector_context",
    "attach_vector_context_to_framework_ctx",
    "iter_vector_events",
]
