# corpus_sdk/embedding/framework_adapters/common/framework_utils.py
# SPDX-License-Identifier: Apache-2.0
"""
Shared utilities for framework-specific embedding adapters.

This module centralizes common logic used across all framework adapters:

- Coercing adapter / translator results into a canonical
  ``List[List[float]]`` or ``List[float]`` shape
- Emitting consistent, framework-aware batch-size warnings

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
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Protocol, Sequence


LOG = logging.getLogger(__name__)


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
    invalid_result:
        Code used when the result structure is not a valid embedding container.

    empty_result:
        Code used when no valid embedding rows remain after processing.

    conversion_error:
        Code used when numeric conversion to float fails.
    """

    invalid_result: str
    empty_result: str
    conversion_error: str


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


def coerce_embedding_matrix(
    result: Any,
    *,
    framework: str,
    error_codes: CoercionErrorCodes,
    logger: Optional[logging.Logger] = None,
) -> List[List[float]]:
    """
    Coerce a generic embedding result into a `List[List[float]]` matrix.

    This is the *single* source of truth for coercion logic across all
    framework adapters.

    Parameters
    ----------
    result:
        Arbitrary result returned by the translator / adapter.
    framework:
        Framework label for logging / diagnostics (e.g., "langchain").
    error_codes:
        Bundle of error codes to embed in exception messages.
    logger:
        Optional logger; if omitted, the module-level logger is used.

    Returns
    -------
    List[List[float]]
        Canonical embedding matrix.

    Raises
    ------
    TypeError
        If the result does not contain a valid embeddings sequence.
    ValueError
        If no non-empty embedding rows remain after processing.
    """
    log = logger or LOG

    embeddings_obj = _extract_embeddings_object(result)

    # Fast path: already a sequence of sequences of numbers
    if isinstance(embeddings_obj, Sequence) and embeddings_obj and isinstance(
        embeddings_obj[0], Sequence
    ):
        # We still validate / convert to float below, but we know this is the
        # common "matrix" shape and can avoid extra branching.
        raw_rows = embeddings_obj
    else:
        # Normalize single vectors into a one-row matrix
        if isinstance(embeddings_obj, Sequence):
            # Distinguish between vector and matrix by looking for nested sequences
            if embeddings_obj and not isinstance(embeddings_obj[0], Sequence):
                raw_rows = [embeddings_obj]
            else:
                raw_rows = embeddings_obj
        else:
            raise TypeError(
                f"[{error_codes.invalid_result}] "
                f"{framework}: translator result does not contain a valid embeddings "
                f"sequence (type={type(embeddings_obj).__name__})"
            )

    matrix: List[List[float]] = []

    for idx, row in enumerate(raw_rows):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise TypeError(
                f"[{error_codes.invalid_result}] "
                f"{framework}: expected each embedding row to be a non-string sequence, "
                f"got {type(row).__name__} at index {idx}"
            )

        if len(row) == 0:
            log.warning(
                "%s: empty embedding row at index %d, skipping",
                framework,
                idx,
            )
            continue

        try:
            vector = [float(x) for x in row]
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"[{error_codes.conversion_error}] "
                f"{framework}: failed to convert embedding values to float at "
                f"row {idx}: {exc}"
            ) from exc

        matrix.append(vector)

    if not matrix:
        raise ValueError(
            f"[{error_codes.empty_result}] "
            f"{framework}: translator returned no valid embedding rows"
        )

    log.debug(
        "%s: successfully coerced embedding matrix with %d rows "
        "(original_type=%s)",
        framework,
        len(matrix),
        type(embeddings_obj).__name__,
    )
    return matrix


def coerce_embedding_vector(
    result: Any,
    *,
    framework: str,
    error_codes: CoercionErrorCodes,
    logger: Optional[logging.Logger] = None,
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
    framework:
        Framework label for logging / diagnostics (e.g., "llamaindex").
    error_codes:
        Bundle of error codes to embed in exception messages.
    logger:
        Optional logger; if omitted, the module-level logger is used.

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
        framework=framework,
        error_codes=error_codes,
        logger=log,
    )

    if len(matrix) > 1:
        log.warning(
            "%s: expected a single embedding vector but received %d rows; "
            "using the first row",
            framework,
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
