# corpus_sdk/examples/common/errors.py
# SPDX-License-Identifier: Apache-2.0
"""
Examples — minimal cross-domain error utilities.

We intentionally DO NOT define our own error classes here.
Instead, tests and mocks should raise the domain-specific errors from:
  - corpus_sdk.llm.llm_base
  - corpus_sdk.graph.graph_base
  - corpus_sdk.vector.vector_base

This module only provides small, dependency-free helpers that work with those
domain error types without importing any proprietary/router code.
"""

from __future__ import annotations

import contextlib
from typing import Any, Optional, Tuple, Type

# Import domain error types (best-effort; keep imports local & light)
# LLM
from corpus_sdk.llm.llm_base import (  # type: ignore
    LLMAdapterError as _LLMBase,
    ResourceExhausted as _LLMResourceExhausted,
    Unavailable as _LLMUnavailable,
    TransientNetwork as _LLMTransientNetwork,
    ModelOverloaded as _LLMModelOverloaded,
)
# Graph
from corpus_sdk.graph.graph_base import (  # type: ignore
    AdapterError as _GraphBase,
    ResourceExhausted as _GraphResourceExhausted,
    Unavailable as _GraphUnavailable,
    TransientNetwork as _GraphTransientNetwork,
)
# Vector
from corpus_sdk.vector.vector_base import (  # type: ignore
    VectorAdapterError as _VectorBase,
    ResourceExhausted as _VectorResourceExhausted,
    Unavailable as _VectorUnavailable,
    TransientNetwork as _VectorTransientNetwork,
    IndexNotReady as _VectorIndexNotReady,
)

# A union of the three domain base classes, for isinstance checks
_DOMAIN_BASES: Tuple[Type[BaseException], ...] = (_LLMBase, _GraphBase, _VectorBase)

# Retryable “network/availability/limit” classes across domains
_RETRYABLE_TYPES: Tuple[Type[BaseException], ...] = (
    _LLMResourceExhausted, _LLMUnavailable, _LLMTransientNetwork, _LLMModelOverloaded,
    _GraphResourceExhausted, _GraphUnavailable, _GraphTransientNetwork,
    _VectorResourceExhausted, _VectorUnavailable, _VectorTransientNetwork, _VectorIndexNotReady,
)


@contextlib.contextmanager
def adapt_errors(component: str, operation: str, *, wrap_as: Type[BaseException] | None = None):
    """
    Wrap any non-domain exceptions into a provided error type, preserving the message.
    Use this only in examples/demos; production adapters should map errors explicitly.

    Args:
        component: human-readable source (e.g., "openai", "neo4j")
        operation: operation name (e.g., "complete", "query")
        wrap_as: if set, wrap unknown exceptions as this type
    """
    try:
        yield
    except _DOMAIN_BASES:
        # Already a domain error; let it pass through unchanged
        raise
    except Exception as e:
        if wrap_as is None:
            raise
        # Construct a simple message; avoid leaking complex objects
        msg = f"{component} {operation} failed: {e}"
        # Try to include retry hints if the original exception had them
        retry_after_ms = getattr(e, "retry_after_ms", None)
        try:
            if retry_after_ms is not None:
                raise wrap_as(msg, retry_after_ms=retry_after_ms) from e  # type: ignore[misc]
            raise wrap_as(msg) from e
        except TypeError:
            # Fallback if the target type doesn't accept retry_after_ms kwarg
            raise wrap_as(msg) from e


def is_retryable_error(err: BaseException) -> bool:
    """
    Heuristic: consider common availability/network/limit errors retryable.
    Works across LLM/Graph/Vector domain error types.
    """
    return isinstance(err, _RETRYABLE_TYPES)


def get_retry_after_ms(err: BaseException) -> Optional[int]:
    """
    Extract retry-after hint from domain errors, if present.
    """
    val = getattr(err, "retry_after_ms", None)
    try:
        return int(val) if val is not None else None
    except Exception:
        return None


__all__ = [
    "adapt_errors",
    "is_retryable_error",
    "get_retry_after_ms",
]
