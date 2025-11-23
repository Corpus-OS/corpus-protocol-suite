# corpus_sdk/embedding/framework_adapters/common/framework_utils.py
# SPDX-License-Identifier: Apache-2.0
"""
Shared utilities for framework-specific embedding adapters.

This module centralizes common logic used across all framework adapters:

- Coercing adapter / translator results into a canonical
  ``List[List[float]]`` or ``List[float]`` shape
- Emitting consistent, framework-aware batch-size warnings
- Validating embedding dimension consistency

It intentionally stays *framework-neutral* and uses only:

- Standard library types
- Simple, caller-provided error codes
- Optional, duck-typed batch configuration

Adapters remain responsible for:

- Choosing the appropriate error codes
- Providing the correct framework name and operation name
- Applying framework-specific context / logging
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Protocol, Sequence


LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Maximum dimension to prevent memory exhaustion attacks
MAX_EMBEDDING_DIMENSION = 50_000

# Maximum number of rows to validate dimensions on (for performance)
MAX_DIMENSION_VALIDATION_ROWS = 10_000


# ---------------------------------------------------------------------------
# Framework inference helper
# ---------------------------------------------------------------------------


def _infer_framework_name(
    error_codes: Any,
    explicit_framework: Optional[str] = None,
) -> str:
    """
    Infer framework name from error codes class with proper fallback chain.

    Priority:
    1. Explicit framework parameter
    2. error_codes.framework_name attribute (if present)
    3. Inferred from error_codes class name
    4. Fallback to "adapter"

    Parameters
    ----------
    error_codes:
        Error codes object (typically CoercionErrorCodes subclass).
    explicit_framework:
        Explicit framework override.

    Returns
    -------
    str
        Framework name for logging and error messages.
    """
    # Priority 1: Explicit override
    if explicit_framework:
        return explicit_framework

    # Priority 2: Explicit attribute
    if hasattr(error_codes, "framework_name"):
        framework_name = getattr(error_codes, "framework_name")
        if isinstance(framework_name, str) and framework_name:
            return framework_name

    # Priority 3: Infer from class name
    class_name = type(error_codes).__name__
    if class_name and class_name != "CoercionErrorCodes":
        # Remove common suffixes
        name = re.sub(r"(ErrorCodes|Error|Codes)$", "", class_name)
        
        # Convert PascalCase to snake_case
        # E.g., "LangChain" -> "lang_chain"
        name = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
        
        if name:
            return name

    # Priority 4: Fallback
    return "adapter"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_error_codes(error_codes: Any) -> None:
    """
    Validate that error_codes has all required attributes.

    Parameters
    ----------
    error_codes:
        Error codes object to validate.

    Raises
    ------
    TypeError
        If error_codes is missing required attributes.
    """
    required_attrs = [
        "INVALID_EMBEDDING_RESULT",
        "EMPTY_EMBEDDING_RESULT",
        "EMBEDDING_CONVERSION_ERROR",
        "DIMENSION_MISMATCH",
    ]
    
    missing_attrs = [
        attr for attr in required_attrs if not hasattr(error_codes, attr)
    ]
    
    if missing_attrs:
        raise TypeError(
            f"error_codes missing required attributes: {', '.join(missing_attrs)}. "
            f"Must provide a CoercionErrorCodes instance or compatible object."
        )


def _validate_float_value(value: float, *, framework: str) -> None:
    """
    Validate that a float value is safe (not NaN or infinity).

    Parameters
    ----------
    value:
        Float value to validate.
    framework:
        Framework name for error messages.

    Raises
    ------
    ValueError
        If value is NaN or infinity.
    """
    import math
    
    if math.isnan(value):
        raise ValueError(
            f"{framework}: embedding contains NaN value - this typically indicates "
            f"an upstream computation error"
        )
    
    if math.isinf(value):
        raise ValueError(
            f"{framework}: embedding contains infinity value - this may indicate "
            f"numerical instability or overflow"
        )


# ---------------------------------------------------------------------------
# Typing helpers
# ---------------------------------------------------------------------------


class HasMaxBatchSize(Protocol):
    """
    Minimal protocol for batch configuration objects.

    This is intentionally duck-typed so it can be satisfied by:

    - `corpus_sdk.embedding.framework_adapters.common.BatchConfig`
    - Any other object exposing a compatible `max_batch_size` attribute
    """

    max_batch_size: Optional[int]


@dataclass(frozen=True)
class CoercionErrorCodes:
    """
    Structured bundle of error codes used during embedding coercion.

    These codes are surfaced in exception messages so individual frameworks
    can attach or filter on them in higher-level error handlers.

    Attributes
    ----------
    INVALID_EMBEDDING_RESULT:
        Code used when the result structure is not a valid embedding container.

    EMPTY_EMBEDDING_RESULT:
        Code used when no valid embedding rows remain after processing.

    EMBEDDING_CONVERSION_ERROR:
        Code used when numeric conversion to float fails.

    DIMENSION_MISMATCH:
        Code used when embedding rows have inconsistent dimensions.
    
    framework_name:
        Optional framework identifier for improved error messages and logging.
        If not provided, will be inferred from the class name.
    """

    INVALID_EMBEDDING_RESULT: str = "INVALID_EMBEDDING_RESULT"
    EMPTY_EMBEDDING_RESULT: str = "EMPTY_EMBEDDING_RESULT"
    EMBEDDING_CONVERSION_ERROR: str = "EMBEDDING_CONVERSION_ERROR"
    DIMENSION_MISMATCH: str = "DIMENSION_MISMATCH"
    framework_name: Optional[str] = None


@dataclass(frozen=True)
class BatchWarningConfig:
    """
    Configuration for batch-size warnings.

    Attributes
    ----------
    warn_threshold:
        Soft threshold above which a warning is logged when no explicit
        batch size is configured. Default: 10_000 items.

    framework_label:
        Name of the framework (e.g., "langchain", "llamaindex") used
        only for logging / observability.
    """

    warn_threshold: int = 10_000
    framework_label: str = "unknown"


# ---------------------------------------------------------------------------
# Embedding coercion helpers
# ---------------------------------------------------------------------------


def _extract_embeddings_object(result: Any) -> Any:
    """
    Extract the underlying embeddings object from a variety of result shapes.

    Supported shapes:
    - Mapping with "embeddings": {"embeddings": [[...], [...]], ...}
    - Mapping with "embedding": {"embedding": [...], ...} (single vector)
    - Object with `.embeddings` attribute
    - Object with `.embedding` attribute (single vector)
    - Raw list / sequence of vectors: [[...], [...]]
    - Raw single vector: [...]
    """
    # Mapping with explicit key
    if isinstance(result, Mapping):
        if "embeddings" in result:
            return result["embeddings"]
        if "embedding" in result:
            # Single vector; caller will normalize to matrix if needed
            return result["embedding"]

    # Attribute-based (EmbedResult-like objects)
    if hasattr(result, "embeddings"):
        return getattr(result, "embeddings")
    if hasattr(result, "embedding"):
        return getattr(result, "embedding")

    # Fallback: treat result as the embeddings object itself
    return result


def _validate_dimension_consistency(
    matrix: List[List[float]],
    *,
    framework: str,
    error_codes: CoercionErrorCodes,
    logger: Optional[logging.Logger] = None,
    validate_values: bool = True,
) -> None:
    """
    Validate that all embedding rows have the same dimension.

    Includes security checks for:
    - Maximum dimension size (prevents memory exhaustion)
    - NaN/infinity values (indicates computation errors)
    - Performance limit on number of rows validated

    Parameters
    ----------
    matrix:
        List of embedding vectors to validate.
    framework:
        Framework label for error messages.
    error_codes:
        Bundle of error codes to embed in exception messages.
    logger:
        Optional logger for warnings.
    validate_values:
        If True, also validate that values are not NaN or infinity.

    Raises
    ------
    ValueError
        If embedding rows have inconsistent dimensions, exceed max dimension,
        or contain invalid values.
    """
    if not matrix:
        return

    log = logger or LOG
    expected_dim = len(matrix[0])

    # Security check: prevent memory exhaustion
    if expected_dim > MAX_EMBEDDING_DIMENSION:
        raise ValueError(
            f"{framework}: embedding dimension {expected_dim} exceeds maximum "
            f"allowed dimension {MAX_EMBEDDING_DIMENSION} - this may indicate "
            f"a malformed result or potential memory exhaustion attack"
        )

    # Performance optimization: limit validation on very large matrices
    rows_to_validate = min(len(matrix), MAX_DIMENSION_VALIDATION_ROWS)
    if len(matrix) > MAX_DIMENSION_VALIDATION_ROWS:
        log.debug(
            "%s: matrix has %d rows, validating dimensions on first %d rows only",
            framework,
            len(matrix),
            MAX_DIMENSION_VALIDATION_ROWS,
        )

    # Validate dimension consistency
    for idx in range(1, rows_to_validate):
        row = matrix[idx]
        if len(row) != expected_dim:
            # Collect all mismatches for diagnostics
            mismatches = [
                (i, len(r))
                for i, r in enumerate(matrix[:rows_to_validate])
                if len(r) != expected_dim
            ]
            log.error(
                "%s: dimension mismatch detected - expected %d, found rows with "
                "dimensions: %s",
                framework,
                expected_dim,
                mismatches[:10],  # Limit to first 10 for readability
            )
            raise ValueError(
                f"[{error_codes.DIMENSION_MISMATCH}] "
                f"{framework}: embedding dimension mismatch at row {idx}: "
                f"expected {expected_dim}, got {len(row)}"
            )

    # Optional value validation (NaN/infinity check)
    if validate_values and matrix:
        # Sample check on first row only for performance
        for value in matrix[0]:
            try:
                _validate_float_value(value, framework=framework)
            except ValueError:
                # If first row has issues, do full scan to report all problems
                log.warning(
                    "%s: detected invalid values in embeddings, performing full scan",
                    framework,
                )
                for row_idx, row in enumerate(matrix[:100]):  # Limit scan
                    for col_idx, val in enumerate(row):
                        try:
                            _validate_float_value(val, framework=framework)
                        except ValueError as e:
                            raise ValueError(
                                f"{framework}: invalid value at row {row_idx}, "
                                f"column {col_idx}: {e}"
                            ) from e
                raise  # Re-raise original error if we didn't find more details


def coerce_embedding_matrix(
    result: Any,
    *,
    error_codes: CoercionErrorCodes,
    logger: Optional[logging.Logger] = None,
    validate_dimensions: bool = True,
    validate_values: bool = False,
    framework: Optional[str] = None,
) -> List[List[float]]:
    """
    Coerce a generic embedding result into a `List[List[float]]` matrix.

    This is the *single* source of truth for coercion logic across all
    framework adapters.

    Parameters
    ----------
    result:
        Arbitrary result returned by the translator / adapter.
    error_codes:
        Bundle of error codes to embed in exception messages. Must have
        attributes: INVALID_EMBEDDING_RESULT, EMPTY_EMBEDDING_RESULT,
        EMBEDDING_CONVERSION_ERROR, DIMENSION_MISMATCH.
    logger:
        Optional logger; if omitted, the module-level logger is used.
    validate_dimensions:
        If True (default), validate that all rows have consistent dimensions.
        Set to False for performance-critical paths where dimension consistency
        is guaranteed upstream.
    validate_values:
        If True, validate that embedding values are not NaN or infinity.
        Default False for performance; enable for debugging or untrusted sources.
    framework:
        Optional framework label for logging. If not provided, will be inferred
        from error_codes class name or framework_name attribute.

    Returns
    -------
    List[List[float]]
        Canonical embedding matrix with validated dimensions.

    Raises
    ------
    TypeError
        If error_codes is invalid or result does not contain a valid
        embeddings sequence.
    ValueError
        If no non-empty embedding rows remain after processing, or if
        dimensions are inconsistent when validation is enabled, or if
        values are invalid when value validation is enabled.

    Examples
    --------
    >>> # Standard usage with automatic framework inference
    >>> from corpus_sdk.embedding.framework_adapters.langchain import ErrorCodes
    >>> result = {"embeddings": [[1.0, 2.0], [3.0, 4.0]]}
    >>> matrix = coerce_embedding_matrix(result, error_codes=ErrorCodes())
    >>> # Infers framework="langchain" from ErrorCodes class name
    
    >>> # Performance-critical path with validation disabled
    >>> matrix = coerce_embedding_matrix(
    ...     trusted_result,
    ...     error_codes=error_codes,
    ...     validate_dimensions=False,
    ... )
    
    >>> # Debugging with full validation enabled
    >>> matrix = coerce_embedding_matrix(
    ...     untrusted_result,
    ...     error_codes=error_codes,
    ...     validate_dimensions=True,
    ...     validate_values=True,
    ... )
    """
    log = logger or LOG

    # Input validation
    _validate_error_codes(error_codes)

    # Infer framework name with proper fallback chain
    framework_label = _infer_framework_name(error_codes, framework)

    embeddings_obj = _extract_embeddings_object(result)

    # Normalize to matrix shape
    raw_rows: Sequence[Sequence[Any]]

    if isinstance(embeddings_obj, Sequence) and not isinstance(
        embeddings_obj, (str, bytes)
    ):
        if embeddings_obj and isinstance(embeddings_obj[0], Sequence) and not isinstance(
            embeddings_obj[0], (str, bytes)
        ):
            # Already a matrix: [[1,2], [3,4]]
            raw_rows = embeddings_obj
        elif embeddings_obj:
            # Single vector: [1,2,3] → [[1,2,3]]
            raw_rows = [embeddings_obj]
        else:
            # Empty sequence
            raw_rows = []
    else:
        raise TypeError(
            f"[{error_codes.INVALID_EMBEDDING_RESULT}] "
            f"{framework_label}: translator result does not contain a valid embeddings "
            f"sequence (type={type(embeddings_obj).__name__})"
        )

    matrix: List[List[float]] = []

    for idx, row in enumerate(raw_rows):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise TypeError(
                f"[{error_codes.INVALID_EMBEDDING_RESULT}] "
                f"{framework_label}: expected each embedding row to be a non-string "
                f"sequence, got {type(row).__name__} at index {idx}"
            )

        if len(row) == 0:
            log.warning(
                "%s: empty embedding row at index %d, skipping",
                framework_label,
                idx,
            )
            continue

        try:
            vector = [float(x) for x in row]
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"[{error_codes.EMBEDDING_CONVERSION_ERROR}] "
                f"{framework_label}: failed to convert embedding values to float at "
                f"row {idx}: {exc}"
            ) from exc

        matrix.append(vector)

    if not matrix:
        raise ValueError(
            f"[{error_codes.EMPTY_EMBEDDING_RESULT}] "
            f"{framework_label}: translator returned no valid embedding rows"
        )

    # Validate dimension consistency and values
    if validate_dimensions:
        _validate_dimension_consistency(
            matrix,
            framework=framework_label,
            error_codes=error_codes,
            logger=log,
            validate_values=validate_values,
        )

    log.debug(
        "%s: successfully coerced embedding matrix with %d rows, dimension %d "
        "(original_type=%s)",
        framework_label,
        len(matrix),
        len(matrix[0]) if matrix else 0,
        type(embeddings_obj).__name__,
    )
    return matrix


def coerce_embedding_vector(
    result: Any,
    *,
    error_codes: CoercionErrorCodes,
    logger: Optional[logging.Logger] = None,
    framework: Optional[str] = None,
) -> List[float]:
    """
    Coerce a generic embedding result into a single `List[float]` vector.

    Strategy:
    - Use `coerce_embedding_matrix` to normalize the result
    - If the matrix has exactly one row → return that row
    - If it has multiple rows → return the first row and log a warning

    Parameters
    ----------
    result:
        Arbitrary result returned by the translator / adapter.
    error_codes:
        Bundle of error codes to embed in exception messages.
    logger:
        Optional logger; if omitted, the module-level logger is used.
    framework:
        Optional framework label for logging. If not provided, will be inferred
        from error_codes class name or framework_name attribute.

    Returns
    -------
    List[float]
        Single embedding vector.

    Raises
    ------
    TypeError, ValueError
        Propagated from `coerce_embedding_matrix`.
    """
    log = logger or LOG

    matrix = coerce_embedding_matrix(
        result,
        error_codes=error_codes,
        logger=log,
        framework=framework,
    )

    if len(matrix) > 1:
        # Use same framework inference for consistent messaging
        framework_label = _infer_framework_name(error_codes, framework)
        log.warning(
            "%s: expected a single embedding vector but received %d rows; "
            "using the first row",
            framework_label,
            len(matrix),
        )

    return matrix[0]


# ---------------------------------------------------------------------------
# Batch-size warning helper
# ---------------------------------------------------------------------------


def warn_if_extreme_batch(
    texts: Sequence[str],
    *,
    framework: str,
    op_name: str,
    batch_config: Optional[HasMaxBatchSize],
    warning_config: Optional[BatchWarningConfig] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Emit a soft warning if an extremely large batch is requested without an
    explicit `max_batch_size` in the batch configuration.

    This is used by framework adapters to surface potentially dangerous
    batch sizes while preserving existing behavior (no hard failures).

    Parameters
    ----------
    texts:
        Sequence of texts passed to the framework embedding API. A single
        string is treated as *not* a batch and ignored.
    framework:
        Name of the framework issuing the call (e.g., "langchain").
    op_name:
        Name of the high-level operation (e.g., "embed_documents").
    batch_config:
        Optional batch configuration object exposing a `max_batch_size` attribute.
    warning_config:
        Optional `BatchWarningConfig`. If omitted, defaults are used.
    logger:
        Optional logger; if omitted, the module-level logger is used.
    """
    log = logger or LOG
    cfg = warning_config or BatchWarningConfig(framework_label=framework)

    # Treat a single string as non-batch input; nothing to warn about.
    if isinstance(texts, (str, bytes)):
        return

    batch_size = len(texts)
    if batch_size <= cfg.warn_threshold:
        return

    if batch_config is None:
        max_batch_size: Optional[int] = None
    else:
        # Duck-typed: any object with `max_batch_size` attribute is accepted
        max_batch_size = getattr(batch_config, "max_batch_size", None)

    if max_batch_size is not None:
        # Caller has explicitly configured a max batch size; assume they know
        # what they're doing and avoid extra noise.
        return

    log.warning(
        "%s (%s): %s called with batch_size=%d and no explicit max_batch_size; "
        "ensure your adapter/translator can safely handle very large batches.",
        cfg.framework_label,
        framework,
        op_name,
        batch_size,
    )


__all__ = [
    "HasMaxBatchSize",
    "CoercionErrorCodes",
    "BatchWarningConfig",
    "coerce_embedding_matrix",
    "coerce_embedding_vector",
    "warn_if_extreme_batch",
]
