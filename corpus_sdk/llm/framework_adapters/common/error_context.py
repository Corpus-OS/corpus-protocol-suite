# corpus_sdk/llm/framework_adapters/common/error_context.py
# SPDX-License-Identifier: Apache-2.0

"""
Error context utilities for framework adapters.

This module provides helpers for attaching rich debugging context to exceptions
as they propagate through framework adapters. This context enrichment enables:

- Post-mortem debugging with framework-specific metadata
- Error aggregation and analysis across multiple frameworks
- Contextual logging without modifying exception messages
- Discovery of error origins in multi-layer architectures

The attached context is deliberately stored as exception attributes rather than
in log messages to preserve the original exception type and message while still
providing debugging information to downstream handlers.

Typical usage
-------------

    from corpus_sdk.llm.framework_adapters.common.error_context import attach_context

    try:
        result = await adapter.complete(messages=messages)
    except Exception as exc:
        attach_context(
            exc,
            framework="langchain",
            operation="complete",
            messages_count=len(messages),
            model="gpt-4",
        )
        raise

Later, in error handlers or observability systems:

    except Exception as exc:
        if hasattr(exc, "__corpus_context__"):
            context = exc.__corpus_context__
            logger.error(
                "Adapter error",
                extra={
                    "framework": context.get("framework"),
                    "operation": context.get("operation"),
                    "messages_count": context.get("messages_count"),
                }
            )

Design philosophy
-----------------

* Lightweight:
    Context attachment is a simple attribute assignment with no serialization
    or external dependencies.

* Non-invasive:
    Does not modify the exception message, type, or traceback. The original
    exception propagates unchanged except for the added attributes.

* Safe:
    All operations are wrapped in try/except to ensure that context attachment
    failures never mask the original exception.

* Discoverable:
    Uses both `__corpus_context__` (canonical) and `__<framework>_context__`
    (framework-specific) attributes for maximum discoverability in debuggers
    and error aggregation systems.

* Composable:
    Multiple calls to attach_context merge contexts rather than overwriting,
    allowing different layers to contribute their own context.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, MutableMapping, Optional

logger = logging.getLogger(__name__)


def attach_context(
    exc: BaseException,
    framework: str,
    **context: Any,
) -> None:
    """
    Attach framework-specific debugging context to an exception.

    This function enriches exceptions with metadata that aids in debugging,
    error aggregation, and observability. The context is stored as exception
    attributes and does not modify the exception's message or type.

    Two attributes are set on the exception:

    1. `__corpus_context__` (canonical):
        Contains all context from Corpus SDK perspective.

    2. `__<framework>_context__` (framework-specific):
        Contains the same context but with a framework-specific attribute
        name (e.g., `__langchain_context__`, `__autogen_context__`).
        This aids discoverability when debugging framework-specific issues.

    If context already exists on the exception (from a previous call),
    the new context is merged with the existing context. The `framework`
    key is always set and will not be overwritten if already present.

    Parameters
    ----------
    exc:
        The exception to enrich with context. Can be any BaseException,
        including built-in exceptions and custom error types.

    framework:
        Framework identifier (e.g., "langchain", "llamaindex", "autogen",
        "semantic_kernel"). This is used both as a context key and to
        generate the framework-specific attribute name.

    **context:
        Arbitrary keyword arguments representing the context to attach.
        Common keys include:
            - operation: str (e.g., "complete", "stream", "count_tokens")
            - messages_count: int
            - model: str
            - temperature: float
            - max_tokens: int
            - request_id: str
            - tenant: str (should be hashed if present)
            - error_stage: str (e.g., "translation", "api_call", "response_parsing")

        Avoid including PII or sensitive data in context.

    Examples
    --------
    Basic usage in an adapter:

        try:
            result = await corpus_adapter.complete(messages=messages)
        except Exception as exc:
            attach_context(
                exc,
                framework="langchain",
                operation="complete",
                messages_count=len(messages),
                model="gpt-4",
            )
            raise

    Multiple layers adding context:

        # Layer 1: Framework adapter
        try:
            result = await corpus_adapter.complete(...)
        except Exception as exc:
            attach_context(exc, framework="langchain", operation="complete")
            raise

        # Layer 2: Router
        try:
            result = await langchain_adapter.invoke(...)
        except Exception as exc:
            attach_context(exc, framework="router", strategy="latency_based")
            raise

        # Later inspection:
        except Exception as exc:
            if hasattr(exc, "__corpus_context__"):
                # Context contains keys from both layers:
                # {"framework": "langchain", "operation": "complete",
                #  "strategy": "latency_based"}
                context = exc.__corpus_context__

    Notes
    -----
    - Context attachment is best-effort; any failures during attachment
      are logged but do not prevent the original exception from propagating.

    - The `framework` parameter is deliberately required (not **context)
      to ensure every context attachment explicitly declares its origin.

    - Context is stored as a regular dict, not a frozen/immutable structure,
      to allow multiple layers to contribute context. However, callers
      should treat the context as append-only (no key deletion).

    - For performance-critical paths, consider conditionally attaching
      context only when detailed debugging is enabled, though the overhead
      is typically negligible (<1μs per call).
    """
    try:
        # Start with empty context dict
        merged_context: MutableMapping[str, Any] = {}

        # Attempt to merge with any existing corpus context
        if hasattr(exc, "__corpus_context__"):
            try:
                existing = getattr(exc, "__corpus_context__")
                if isinstance(existing, Mapping):
                    merged_context.update(existing)
            except Exception as merge_error:  # noqa: BLE001
                # If we can't read the existing context, log but continue.
                # This should never happen in practice but we're being defensive.
                logger.debug(
                    "Failed to merge existing __corpus_context__: %s",
                    merge_error,
                    extra={"framework": framework},
                )

        # Set framework if not already present (preserve existing if set)
        merged_context.setdefault("framework", framework)

        # Merge the new context
        merged_context.update(context)

        # Set canonical corpus context
        setattr(exc, "__corpus_context__", merged_context)

        # Set framework-specific context for discoverability
        # e.g., __langchain_context__, __autogen_context__
        framework_attr = f"__{framework}_context__"
        setattr(exc, framework_attr, merged_context)

    except Exception as attachment_error:  # noqa: BLE001
        # Context attachment should never interfere with exception propagation.
        # If anything goes wrong, log it and move on.
        logger.debug(
            "Failed to attach error context to %s: %s",
            type(exc).__name__,
            attachment_error,
            extra={"framework": framework},
        )


def get_context(
    exc: BaseException,
    *,
    framework: Optional[str] = None,
) -> Mapping[str, Any]:
    """
    Retrieve attached context from an exception.

    Parameters
    ----------
    exc:
        The exception to retrieve context from.

    framework:
        Optional framework identifier. If provided, attempts to retrieve
        the framework-specific context attribute first (e.g.,
        `__langchain_context__`) before falling back to `__corpus_context__`.

    Returns
    -------
    Mapping[str, Any]
        The attached context, or an empty dict if no context is present.

    Examples
    --------
        try:
            result = await adapter.complete(...)
        except Exception as exc:
            context = get_context(exc)
            logger.error(
                "Adapter failed",
                extra={
                    "operation": context.get("operation"),
                    "messages_count": context.get("messages_count"),
                }
            )

        # Framework-specific retrieval:
        context = get_context(exc, framework="langchain")
    """
    try:
        # Try framework-specific context first if requested
        if framework:
            framework_attr = f"__{framework}_context__"
            if hasattr(exc, framework_attr):
                ctx = getattr(exc, framework_attr)
                if isinstance(ctx, Mapping):
                    return ctx

        # Fall back to canonical corpus context
        if hasattr(exc, "__corpus_context__"):
            ctx = getattr(exc, "__corpus_context__")
            if isinstance(ctx, Mapping):
                return ctx

    except Exception as retrieval_error:  # noqa: BLE001
        logger.debug(
            "Failed to retrieve error context from %s: %s",
            type(exc).__name__,
            retrieval_error,
        )

    return {}


def has_context(
    exc: BaseException,
    *,
    framework: Optional[str] = None,
) -> bool:
    """
    Check if an exception has attached context.

    Parameters
    ----------
    exc:
        The exception to check.

    framework:
        Optional framework identifier. If provided, checks for framework-
        specific context instead of canonical corpus context.

    Returns
    -------
    bool
        True if the exception has context attached, False otherwise.

    Examples
    --------
        except Exception as exc:
            if has_context(exc):
                context = get_context(exc)
                logger.error("Error with context", extra=context)
            else:
                logger.error("Error without context")
    """
    try:
        if framework:
            framework_attr = f"__{framework}_context__"
            if hasattr(exc, framework_attr):
                ctx = getattr(exc, framework_attr)
                return isinstance(ctx, Mapping) and len(ctx) > 0

        if hasattr(exc, "__corpus_context__"):
            ctx = getattr(exc, "__corpus_context__")
            return isinstance(ctx, Mapping) and len(ctx) > 0

    except Exception:  # noqa: BLE001
        pass

    return False


def clear_context(exc: BaseException) -> None:
    """
    Remove all attached context from an exception.

    This is primarily useful in testing or in scenarios where you need to
    sanitize exceptions before serialization.

    Parameters
    ----------
    exc:
        The exception to clear context from.

    Examples
    --------
        # Sanitize exception before serialization
        try:
            result = await adapter.complete(...)
        except Exception as exc:
            clear_context(exc)
            # Serialize exc without context metadata
            serialized = json.dumps({"error": str(exc)})

    Notes
    -----
    This function removes both `__corpus_context__` and any framework-specific
    context attributes it can find. However, it does not attempt to discover
    all possible framework-specific attributes; it only removes ones that
    follow the `__<framework>_context__` pattern for known frameworks.

    In practice, you rarely need this function; context attachment is designed
    to be transparent and does not interfere with normal exception handling.
    """
    try:
        # Remove canonical corpus context
        if hasattr(exc, "__corpus_context__"):
            delattr(exc, "__corpus_context__")

        # Remove known framework-specific contexts
        known_frameworks = [
            "langchain",
            "llamaindex",
            "autogen",
            "semantic_kernel",
            "haystack",
        ]
        for framework in known_frameworks:
            framework_attr = f"__{framework}_context__"
            if hasattr(exc, framework_attr):
                delattr(exc, framework_attr)

    except Exception as clear_error:  # noqa: BLE001
        logger.debug(
            "Failed to clear error context from %s: %s",
            type(exc).__name__,
            clear_error,
        )


__all__ = [
    "attach_context",
    "get_context",
    "has_context",
    "clear_context",
]
