"""
Lightweight, open-source error taxonomy used in Corpus SDK examples.
These mirror the normalized error types from corpus_sdk.common.errors,
but without any proprietary router integration or retry heuristics.
"""

class AdapterError(Exception):
    """Base class for all adapter-level exceptions."""
    code = "ADAPTER_ERROR"

    def __init__(self, message: str = "", **meta):
        super().__init__(message)
        self.message = message
        self.meta = meta

    def __str__(self) -> str:
        extra = f" ({self.meta})" if self.meta else ""
        return f"{self.__class__.__name__}: {self.message}{extra}"


class BadRequest(AdapterError):
    code = "BAD_REQUEST"


class AuthError(AdapterError):
    code = "AUTH_ERROR"


class ResourceExhausted(AdapterError):
    code = "RESOURCE_EXHAUSTED"


class TransientNetwork(AdapterError):
    code = "TRANSIENT_NETWORK"


class Unavailable(AdapterError):
    code = "UNAVAILABLE"


class NotSupported(AdapterError):
    code = "NOT_SUPPORTED"


# --- LLM-specific ------------------------------------------------------------

class ModelOverloaded(Unavailable):
    code = "MODEL_OVERLOADED"


class ContentFiltered(BadRequest):
    code = "CONTENT_FILTERED"


# --- Vector/Embedding-specific -----------------------------------------------

class DimensionMismatch(BadRequest):
    code = "DIMENSION_MISMATCH"


class IndexNotReady(Unavailable):
    code = "INDEX_NOT_READY"


class TextTooLong(BadRequest):
    code = "TEXT_TOO_LONG"


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
