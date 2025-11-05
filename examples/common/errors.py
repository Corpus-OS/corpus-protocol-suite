# SPDX-License-Identifier: Apache-2.0
"""
Lightweight, open-source error taxonomy used in Corpus SDK examples.

These mirror the normalized error types from corpus_sdk.common.errors,
but without any proprietary router integration or retry heuristics.
They are simple, educational, and production-safe for open-source use.
"""

import contextlib
from typing import Any, Dict, Optional


class AdapterError(Exception):
    """Base class for all adapter-level exceptions."""
    code = "ADAPTER_ERROR"
    retryable = False
    status_code = 500

    def __init__(self, message: str = "", **meta: Any):
        super().__init__(message)
        self.message = str(message) if message else self.__class__.__name__
        self.meta = self._validate_meta(meta)

    def __str__(self) -> str:
        extra = f" ({self.meta})" if self.meta else ""
        return f"{self.__class__.__name__}: {self.message}{extra}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self.message)}, **{self.meta})"

    def to_dict(self) -> Dict[str, Any]:
        """Return a structured, serializable representation for logging or metrics."""
        return {
            "type": "adapter_error",
            "error": self.__class__.__name__,
            "code": self.code,
            "retryable": self.retryable,
            "status_code": self.status_code,
            "message": self.message,
            "meta": self.meta,
            "module": self.__class__.__module__,
        }

    def _validate_meta(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure meta values are JSON-serializable and safe."""
        validated = {}
        for k, v in meta.items():
            if not isinstance(k, str):
                k = str(k)
            if isinstance(v, (str, int, float, bool, type(None))):
                validated[k] = v
            else:
                # Convert complex objects to strings for safety
                validated[k] = str(v)
        return validated

    @property
    def retry_after_ms(self) -> Optional[int]:
        """Optional retry-after hint in milliseconds."""
        return self.meta.get("retry_after_ms")


# --- Cross-domain ------------------------------------------------------------

class BadRequest(AdapterError):
    code = "BAD_REQUEST"
    status_code = 400


class AuthError(AdapterError):
    code = "AUTH_ERROR"
    status_code = 401


class ResourceExhausted(AdapterError):
    code = "RESOURCE_EXHAUSTED"
    retryable = True
    status_code = 429

    def __init__(self, message: str = "", retry_after_ms: Optional[int] = None, **meta: Any):
        if retry_after_ms is not None:
            meta["retry_after_ms"] = retry_after_ms
        super().__init__(message, **meta)


class TransientNetwork(AdapterError):
    code = "TRANSIENT_NETWORK"
    retryable = True
    status_code = 503


class Unavailable(AdapterError):
    code = "UNAVAILABLE"
    retryable = True
    status_code = 503


class NotSupported(AdapterError):
    code = "NOT_SUPPORTED"
    status_code = 501


# --- LLM-specific ------------------------------------------------------------

class ModelOverloaded(Unavailable):
    code = "MODEL_OVERLOADED"
    status_code = 503

    def __init__(self, message: str = "", retry_after_ms: Optional[int] = None, **meta: Any):
        if retry_after_ms is not None:
            meta["retry_after_ms"] = retry_after_ms
        super().__init__(message, **meta)


class ContentFiltered(BadRequest):
    code = "CONTENT_FILTERED"
    status_code = 400


# --- Vector/Embedding-specific -----------------------------------------------

class DimensionMismatch(BadRequest):
    code = "DIMENSION_MISMATCH"
    status_code = 400


class IndexNotReady(Unavailable):
    code = "INDEX_NOT_READY"
    retryable = True
    status_code = 503


class TextTooLong(BadRequest):
    code = "TEXT_TOO_LONG"
    status_code = 400

    def __init__(self, message: str = "", actual_length: Optional[int] = None, max_length: Optional[int] = None, **meta: Any):
        if actual_length is not None:
            meta["actual_length"] = actual_length
        if max_length is not None:
            meta["max_length"] = max_length
        super().__init__(message, **meta)


# --- Utility functions -------------------------------------------------------

@contextlib.contextmanager
def adapt_errors(component: str, operation: str, raise_on: type[AdapterError] = AdapterError):
    """
    Context manager to wrap external exceptions with adapter errors.
    
    Args:
        component: Name of the component (e.g., "openai", "pinecone")
        operation: Operation being performed (e.g., "complete", "query")
        raise_on: Specific error type to raise for unhandled exceptions
    """
    try:
        yield
    except AdapterError:
        raise  # Don't re-wrap our own errors
    except Exception as e:
        raise raise_on(f"{component} {operation} failed: {e}") from e


def is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable based on the error taxonomy."""
    if isinstance(error, AdapterError):
        return error.retryable
    return False


def get_retry_after_ms(error: Exception) -> Optional[int]:
    """Extract retry-after hint from error if available."""
    if isinstance(error, AdapterError):
        return error.retry_after_ms
    return None


__all__ = [
    "AdapterError",
    "BadRequest",
    "AuthError",
    "ResourceExhausted", 
    "TransientNetwork",
    "Unavailable",
    "NotSupported",
    "ModelOverloaded",
    "ContentFiltered",
    "DimensionMismatch",
    "IndexNotReady",
    "TextTooLong",
    "adapt_errors",
    "is_retryable_error",
    "get_retry_after_ms",
]
