# SPDX-License-Identifier: Apache-2.0
"""
Lightweight, open-source error taxonomy used in Corpus SDK examples.

These mirror the normalized error types from corpus_sdk.common.errors,
but without any proprietary router integration or retry heuristics.
They are simple, educational, and production-safe for open-source use.
"""

class AdapterError(Exception):
    """Base class for all adapter-level exceptions."""
    code = "ADAPTER_ERROR"
    retryable = False
    status_code = 500

    def __init__(self, message: str = "", **meta):
        super().__init__(message)
        self.message = message
        self.meta = meta or {}

    def __str__(self) -> str:
        extra = f" ({self.meta})" if self.meta else ""
        return f"{self.__class__.__name__}: {self.message}{extra}"

    def to_dict(self) -> dict:
        """Return a structured, serializable representation for logging or metrics."""
        return {
            "error": self.__class__.__name__,
            "code": self.code,
            "retryable": self.retryable,
            "status_code": self.status_code,
            "message": self.message,
            "meta": self.meta,
        }

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
]
