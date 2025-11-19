# corpus_sdk/llm/framework_adapters/common/error_context.py
# SPDX-License-Identifier: Apache-2.0

"""
Error context utilities for framework / component adapters.

This module provides helpers for attaching rich debugging context to exceptions
as they propagate through adapters (LLM, vector, graph, embedding, etc.).
This context enrichment enables:

- Post-mortem debugging with framework- or component-specific metadata
- Error aggregation and analysis across multiple frameworks and subsystems
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
            framework="langchain",  # or "vector_pinecone", "graph_router", etc.
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
                    "resource_type": context.get("resource_type"),  # e.g. "llm", "vector"
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
    (framework-/component-specific) attributes for maximum discoverability in
    debuggers and error aggregation systems.

* Composable:
    Multiple calls to attach_context merge contexts rather than overwriting,
    allowing different layers (LLM, vector, graph, embedding, router, etc.)
    to contribute their own context.
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
    Attach debugging context to an exception.

    This function enriches exceptions with metadata that aids in debugging,
    error aggregation, and observability. The context is stored as exception
    attributes and does not modify the exception's message or type.

    Two attributes are set on the exception:

    1. `__corpus_context__` (canonical):
        Contains all context from Corpus SDK perspective.

    2. `__<framework>_context__` (framework/component-specific):
        Contains the same context but with a more specific attribute name
        (e.g., `__langchain_context__`, `__vector_pinecone_context__`).
        This aids discoverability when debugging specific integrations.

    If context already exists on the exception (from a previous call),
    the new context is merged with the existing context. The `framework`
    key is always set and will not be overwritten if already present.

    Parameters
    ----------
    exc:
        The exception to enrich with context. Can be any BaseException,
        including built-in exceptions and custom error types.

    framework:
        Identifier for the origin of this context. This can be:
            - A framework name (e.g., "langchain", "llamaindex", "autogen")
            - A component name (e.g., "vector_pinecone", "graph_router",
              "embedding_openai")
        It is used both as a context key and to generate the
        framework/component-specific attribute name.

    **context:
        Arbitrary keyword arguments representing the context to attach.

        Common keys for LLMs might include:
            - operation: str (e.g., "complete", "stream", "count_tokens")
            - messages_count: int
            - model: str
            - temperature: float
            - max_tokens: int

        Common keys for vector / graph / embedding might include:
            - resource_type: str (e.g., "vector", "graph", "embedding", "llm")
            - index_name / collection / namespace / graph_id
            - top_k, vector_count
            - operation: str (e.g., "query", "upsert", "delete")

        Cross-cutting fields:
            - request_id: str
            - tenant: str (should be hashed if present)
            - error_stage: str (e.g., "translation", "api_call", "response_parsing")

        Avoid including PII or sensitive data in context.

    Examples
    --------
    Basic usage in an LLM adapter:

        try:
            result = await corpus_adapter.complete(messages=messages)
        except Exception as exc:
            attach_context(
                exc,
                framework="langchain",
                resource_type="llm",
                operation="complete",
                messages_count=len(messages),
                model="gpt-4",
            )
            raise

    In a vector adapter:

        try:
            result = await vector_adapter.query(...)
        except Exception as exc:
            attach_context(
                exc,
                framework="vector_pinecone",
                resource_type="vector",
                operation="query",
                index_name="my-index",
                top_k=10,
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

        # Set framework/component-specific context for discoverability
        # e.g., __langchain_context__, __vector_pinecone_context__
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
        Optional origin identifier. If provided, attempts to retrieve
        the framework/component-specific context attribute first (e.g.,
        `__langchain_context__`, `__vector_pinecone_context__`) before
        falling back to `__corpus_context__`.

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
                    "resource_type": context.get("resource_type"),
                }
            )

        # Framework/component-specific retrieval:
        context = get_context(exc, framework="vector_pinecone")
    """
    try:
        # Try framework-/component-specific context first if requested
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
        Optional origin identifier. If provided, checks for framework/
        component-specific context instead of canonical corpus context.

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


def clear_context(
    exc: BaseException,
    *,
    framework: Optional[str] = None,
) -> None:
    """
    Remove attached context from an exception.

    This is primarily useful in testing or in scenarios where you need to
    sanitize exceptions before serialization.

    Parameters
    ----------
    exc:
        The exception to clear context from.

    framework:
        Optional origin identifier. If provided, only the corresponding
        framework/component-specific attribute (e.g. `__langchain_context__`)
        is removed, along with `__corpus_context__` if present.

        If None (default), all framework/component-specific context attributes
        that follow the `__<name>_context__` pattern are removed, in addition
        to `__corpus_context__`.

    Examples
    --------
        # Sanitize exception before serialization
        try:
            result = await adapter.complete(...)
        except Exception as exc:
            clear_context(exc)
            # Serialize exc without context metadata
            serialized = json.dumps({"error": str(exc)})

        # Only clear context for a specific origin:
        clear_context(exc, framework="vector_pinecone")

    Notes
    -----
    This function removes both `__corpus_context__` and framework/component-
    specific context attributes that follow the `__<name>_context__` pattern.
    It does not attempt to discover arbitrary custom attributes; if callers
    choose a different naming scheme, they must clear those manually.

    In practice, you rarely need this function; context attachment is designed
    to be transparent and does not interfere with normal exception handling.
    """
    try:
        # Remove canonical corpus context
        if hasattr(exc, "__corpus_context__"):
            try:
                delattr(exc, "__corpus_context__")
            except Exception as clear_error:  # noqa: BLE001
                logger.debug(
                    "Failed to delete __corpus_context__ from %s: %s",
                    type(exc).__name__,
                    clear_error,
                )

        # If a specific origin is requested, just remove that one.
        if framework:
            framework_attr = f"__{framework}_context__"
            if hasattr(exc, framework_attr):
                try:
                    delattr(exc, framework_attr)
                except Exception as clear_error:  # noqa: BLE001
                    logger.debug(
                        "Failed to delete %s from %s: %s",
                        framework_attr,
                        type(exc).__name__,
                        clear_error,
                    )
            return

        # Otherwise, remove all attributes that match the convention
        # __<name>_context__ regardless of the name.
        for attr in list(dir(exc)):
            # We only care about attributes that:
            #   - start with "__"
            #   - end with "_context__"
            # This matches names like "__langchain_context__",
            # "__vector_pinecone_context__", etc.
            if not (attr.startswith("__") and attr.endswith("_context__")):
                continue

            # Skip the canonical __corpus_context__ which was already handled.
            if attr == "__corpus_context__":
                continue

            if hasattr(exc, attr):
                try:
                    delattr(exc, attr)
                except Exception as clear_error:  # noqa: BLE001
                    logger.debug(
                        "Failed to delete %s from %s: %s",
                        attr,
                        type(exc).__name__,
                        clear_error,
                    )

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
