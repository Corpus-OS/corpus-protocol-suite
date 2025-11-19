# corpus_sdk/core/context_translation.py
# SPDX-License-Identifier: Apache-2.0

"""
Context translation utilities for Corpus framework adapters.

This module normalizes framework-specific "context" objects into a
canonical `OperationContext` that can be used by any Corpus protocol
layer (LLM, vector, embedding, graph), so that:

- Request IDs / trace IDs are preserved
- Deadlines / timeouts propagate correctly
- Tenant / auth / tags are carried across protocol boundaries
- Framework-specific metadata is captured without polluting core types

Design goals
------------
- Protocol-first: this is SDK infrastructure, not business logic.
- Framework-agnostic: no hard runtime dependency on any single protocol.
- Non-destructive: never drop context fields unless clearly unsafe.
- Debuggable: provide helpers to snapshot/serialize context for logs/metrics.
- Extensible: support custom frameworks via a pluggable registry.

Primary entry points
--------------------
Each `from_*` function returns a core `OperationContext` instance with:

    OperationContext(
        request_id: str | None,
        tenant: str | None,
        deadline_ms: int | None,
        traceparent: str | None,
        attrs: dict,
    )

The `OperationContext` type itself is protocol-agnostic and can be used
by LLM, vector, embedding, graph, or any other Corpus adapter.

- from_langchain
- from_llamaindex
- from_semantic_kernel
- from_autogen
- from_crewai
- from_mcp
- from_dict

Registry-based entry points
---------------------------
For custom frameworks or overrides, you can register a translator that
produces a normalized dict, which is then wrapped into an OperationContext:

    def my_framework_translator(raw_ctx) -> NormalizedContext:
        ...

    register_framework_translator("my_framework", my_framework_translator)

    ctx = translate_framework("my_framework", raw_ctx)

Round-trip helpers
------------------
These helpers take an OperationContext (core type) and convert it back
into framework-specific metadata:

- to_langchain
- to_llamaindex
- to_semantic_kernel
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

from corpus_sdk.core.operation_context import OperationContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Reserved attribute namespace for Corpus-specific fields
CORPUS_ATTR_PREFIX = "corpus_"

#: Reserved keys for framework identification
ATTR_FRAMEWORK = f"{CORPUS_ATTR_PREFIX}framework"
ATTR_FRAMEWORK_VERSION = f"{CORPUS_ATTR_PREFIX}framework_version"

#: Common metadata keys we extract
KEY_TENANT = "tenant"
KEY_DEADLINE_MS = "deadline_ms"
KEY_TIMEOUT = "timeout"
KEY_TIMEOUT_MS = "timeout_ms"
KEY_TRACEPARENT = "traceparent"
KEY_REQUEST_ID = "request_id"
KEY_RUN_ID = "run_id"
KEY_TRACE_ID = "trace_id"
KEY_CORRELATION_ID = "correlation_id"


# Internal normalized context shape used before constructing OperationContext:
# {
#   "request_id": str | None,
#   "tenant": str | None,
#   "deadline_ms": int | None,
#   "traceparent": str | None,
#   "attrs": Dict[str, Any],
# }
NormalizedContext = Dict[str, Any]


class ContextTranslationError(ValueError):
    """Raised when context translation or validation fails."""


# ---------------------------------------------------------------------------
# Registry for custom / override translators
# ---------------------------------------------------------------------------

FrameworkTranslator = Callable[..., NormalizedContext]

_framework_translators: Dict[str, FrameworkTranslator] = {}


def register_framework_translator(name: str, translator: FrameworkTranslator) -> None:
    """
    Register or override a framework translator.

    The translator must return a *normalized* context dict; this module will
    validate it and wrap it into an OperationContext. This is useful for:
    - Custom frameworks not covered by built-in `from_*` helpers.
    - Overriding default behavior for existing frameworks.

    Example
    -------
        def my_translator(raw_ctx) -> NormalizedContext:
            # ...extract fields...
            return {
                "request_id": "...",
                "tenant": "...",
                "deadline_ms": 1234,
                "traceparent": "...",
                "attrs": {...},
            }

        register_framework_translator("my_framework", my_translator)

        ctx = translate_framework("my_framework", raw_ctx)
    """
    if not name or not isinstance(name, str):
        raise ContextTranslationError("Translator name must be a non-empty string")
    if not callable(translator):
        raise ContextTranslationError("Translator must be callable")
    _framework_translators[name] = translator
    logger.debug("Registered framework translator: %s", name)


def get_framework_translator(name: str) -> Optional[FrameworkTranslator]:
    """Return a previously registered framework translator, if any."""
    return _framework_translators.get(name)


def translate_framework(name: str, *args: Any, **kwargs: Any) -> OperationContext:
    """
    Dispatch translation to a registered framework translator and wrap
    the result into an OperationContext.

    Raises:
        ContextTranslationError if no translator is registered or if the
        returned context is invalid.
    """
    translator = get_framework_translator(name)
    if translator is None:
        raise ContextTranslationError(f"No framework translator registered for: {name!r}")

    ctx_dict = translator(*args, **kwargs)
    _validate_normalized_context(ctx_dict, source=f"registry:{name}")
    return OperationContext(**ctx_dict)


# ---------------------------------------------------------------------------
# Core translation functions for built-in frameworks
# ---------------------------------------------------------------------------


def from_langchain(
    config: Optional[Mapping[str, Any]],
    *,
    framework_version: Optional[str] = None,
) -> OperationContext:
    """
    LangChain RunnableConfig → OperationContext.

    Expected shape:
        {
            "run_id": ...,
            "tags": [...],
            "metadata": {
                "tenant": ...,
                "deadline_ms": ...,
                "traceparent": ...,
            },
            "recursion_limit": ...,
        }

    Args:
        config: LangChain RunnableConfig (or None)
        framework_version: Optional version string

    Returns:
        OperationContext
    """
    if not config:
        ctx_dict = _empty_context()
        _validate_normalized_context(ctx_dict, source="langchain")
        return OperationContext(**ctx_dict)

    metadata = _get_dict(config, "metadata")

    attrs: Dict[str, Any] = {ATTR_FRAMEWORK: "langchain"}
    if framework_version:
        attrs[ATTR_FRAMEWORK_VERSION] = framework_version

    # Extract tags
    tags = config.get("tags")
    if tags:
        attrs["tags"] = list(tags)

    # Extract recursion_limit
    recursion_limit = config.get("recursion_limit")
    if recursion_limit is not None:
        attrs["langchain_recursion_limit"] = recursion_limit

    # Extract run_name
    run_name = config.get("run_name")
    if run_name:
        attrs["run_name"] = run_name

    # Merge user attrs from metadata
    user_attrs = _get_dict(metadata, "attrs")
    attrs.update(user_attrs)

    common = _extract_common_fields(config, metadata)

    ctx_dict: NormalizedContext = {
        **common,
        "attrs": attrs,
    }
    _validate_normalized_context(ctx_dict, source="langchain")
    return OperationContext(**ctx_dict)


def from_llamaindex(
    callback_manager: Optional[Any],
    *,
    framework_version: Optional[str] = None,
) -> OperationContext:
    """
    LlamaIndex CallbackManager → OperationContext.

    Extracts:
    - trace_id / span_id
    - request_id / run_id
    - metadata from callback manager

    Args:
        callback_manager: LlamaIndex CallbackManager (or None)
        framework_version: Optional version string

    Returns:
        OperationContext
    """
    if callback_manager is None:
        ctx_dict = _empty_context()
        _validate_normalized_context(ctx_dict, source="llamaindex")
        return OperationContext(**ctx_dict)

    # Try common attributes
    trace_id = getattr(callback_manager, "trace_id", None)
    span_id = getattr(callback_manager, "span_id", None)  # noqa: F841  # reserved for future use
    request_id = (
        getattr(callback_manager, "request_id", None)
        or getattr(callback_manager, "run_id", None)
    )

    # Try to get metadata dict
    metadata = (
        getattr(callback_manager, "context", None)
        or getattr(callback_manager, "metadata", None)
        or {}
    )
    if not isinstance(metadata, Mapping):
        logger.debug(
            "LlamaIndex callback_manager.metadata/context is not a Mapping; got %r",
            type(metadata),
        )
        metadata = {}

    attrs: Dict[str, Any] = {ATTR_FRAMEWORK: "llamaindex"}
    if framework_version:
        attrs[ATTR_FRAMEWORK_VERSION] = framework_version

    # Extract tags
    tags = getattr(callback_manager, "tags", None)
    if tags:
        attrs["tags"] = list(tags)

    # Merge user attrs from metadata
    user_attrs = _get_dict(metadata, "attrs")
    attrs.update(user_attrs)

    common = _extract_common_fields({}, metadata)
    # Prefer explicit IDs from callback manager when available
    common["request_id"] = request_id or trace_id or common["request_id"]

    ctx_dict: NormalizedContext = {
        **common,
        "attrs": attrs,
    }
    _validate_normalized_context(ctx_dict, source="llamaindex")
    return OperationContext(**ctx_dict)


def from_semantic_kernel(
    context: Optional[Any],
    *,
    settings: Optional[Any] = None,
    framework_version: Optional[str] = None,
) -> OperationContext:
    """
    Semantic Kernel context → OperationContext.

    Extracts from context.variables and optional settings.

    Args:
        context: SK context object
        settings: Optional PromptExecutionSettings
        framework_version: Optional version string

    Returns:
        OperationContext
    """
    if context is None and settings is None:
        ctx_dict = _empty_context()
        _validate_normalized_context(ctx_dict, source="semantic_kernel")
        return OperationContext(**ctx_dict)

    # SK often has "variables" or "context_variables"
    metadata = (
        _safe_get(context, ["variables"])
        or _safe_get(context, ["context_variables"])
        or {}
    )
    if not isinstance(metadata, Mapping):
        logger.debug(
            "Semantic Kernel context.variables/context_variables is not a Mapping; got %r",
            type(metadata),
        )
        metadata = {}

    attrs: Dict[str, Any] = {ATTR_FRAMEWORK: "semantic_kernel"}
    if framework_version:
        attrs[ATTR_FRAMEWORK_VERSION] = framework_version

    # Extract function/plugin info
    function_name = _safe_get(context, ["function", "name"])
    plugin_name = _safe_get(context, ["plugin", "name"])
    if function_name:
        attrs["function_name"] = function_name
    if plugin_name:
        attrs["plugin_name"] = plugin_name

    # Merge user attrs from metadata
    user_attrs = _get_dict(metadata, "attrs")
    attrs.update(user_attrs)

    common = _extract_common_fields({}, metadata)

    # Extract deadline from settings if present, overriding metadata-based value
    if settings is not None:
        settings_timeout = (
            getattr(settings, "timeout_ms", None)
            or getattr(settings, "timeout", None)
        )
        if settings_timeout is not None:
            common["deadline_ms"] = _coerce_deadline_ms(settings_timeout)

    ctx_dict: NormalizedContext = {
        **common,
        "attrs": attrs,
    }
    _validate_normalized_context(ctx_dict, source="semantic_kernel")
    return OperationContext(**ctx_dict)


def from_autogen(
    conversation: Optional[Any] = None,
    *,
    framework_version: Optional[str] = None,
    **extra: Any,
) -> OperationContext:
    """
    AutoGen conversation → OperationContext.

    Extracts:
    - conversation_id / id
    - run_id / request_id
    - metadata / config

    Args:
        conversation: AutoGen conversation object
        framework_version: Optional version string
        **extra: Additional context fields

    Returns:
        OperationContext
    """
    if conversation is None and not extra:
        ctx_dict = _empty_context()
        _validate_normalized_context(ctx_dict, source="autogen")
        return OperationContext(**ctx_dict)

    # Extract IDs
    conversation_id = (
        getattr(conversation, "conversation_id", None)
        or getattr(conversation, "id", None)
    )
    run_id = getattr(conversation, "run_id", None)
    request_id = getattr(conversation, "request_id", None)

    # Extract metadata
    metadata = (
        getattr(conversation, "metadata", None)
        or getattr(conversation, "config", None)
        or getattr(conversation, "context", None)
        or {}
    )
    if not isinstance(metadata, Mapping):
        logger.debug(
            "AutoGen conversation.metadata/config/context is not a Mapping; got %r",
            type(metadata),
        )
        metadata = {}

    # Merge extra kwargs
    merged = dict(metadata)
    merged.update(extra)

    attrs: Dict[str, Any] = {ATTR_FRAMEWORK: "autogen"}
    if framework_version:
        attrs[ATTR_FRAMEWORK_VERSION] = framework_version
    if conversation_id is not None:
        attrs["conversation_id"] = str(conversation_id)

    # Merge user attrs
    user_attrs = _get_dict(merged, "attrs")
    attrs.update(user_attrs)

    common = _extract_common_fields({}, merged)
    common["request_id"] = request_id or run_id or common["request_id"]

    ctx_dict: NormalizedContext = {
        **common,
        "attrs": attrs,
    }
    _validate_normalized_context(ctx_dict, source="autogen")
    return OperationContext(**ctx_dict)


def from_crewai(
    task: Optional[Any] = None,
    *,
    framework_version: Optional[str] = None,
    **extra: Any,
) -> OperationContext:
    """
    CrewAI task → OperationContext.

    Extracts:
    - task.id / task.task_id
    - crew.name / agent.name
    - task.metadata

    Args:
        task: CrewAI task object
        framework_version: Optional version string
        **extra: Additional context fields

    Returns:
        OperationContext
    """
    if task is None and not extra:
        ctx_dict = _empty_context()
        _validate_normalized_context(ctx_dict, source="crewai")
        return OperationContext(**ctx_dict)

    # Extract IDs
    task_id = getattr(task, "id", None) or getattr(task, "task_id", None)
    run_id = getattr(task, "run_id", None)
    request_id = getattr(task, "request_id", None)

    # Extract crew/agent info
    crew_name = _safe_get(task, ["crew", "name"])
    agent_name = _safe_get(task, ["agent", "name"])

    # Extract metadata
    metadata = getattr(task, "metadata", None) or {}
    if not isinstance(metadata, Mapping):
        logger.debug(
            "CrewAI task.metadata is not a Mapping; got %r",
            type(metadata),
        )
        metadata = {}

    # Merge extra kwargs
    merged = dict(metadata)
    merged.update(extra)

    attrs: Dict[str, Any] = {ATTR_FRAMEWORK: "crewai"}
    if framework_version:
        attrs[ATTR_FRAMEWORK_VERSION] = framework_version
    if task_id is not None:
        attrs["task_id"] = str(task_id)
    if crew_name:
        attrs["crew_name"] = crew_name
    if agent_name:
        attrs["agent_name"] = agent_name

    # Merge user attrs
    user_attrs = _get_dict(merged, "attrs")
    attrs.update(user_attrs)

    common = _extract_common_fields({}, merged)
    common["request_id"] = request_id or run_id or common["request_id"]

    ctx_dict: NormalizedContext = {
        **common,
        "attrs": attrs,
    }
    _validate_normalized_context(ctx_dict, source="crewai")
    return OperationContext(**ctx_dict)


def from_mcp(
    request: Mapping[str, Any],
    *,
    connection_metadata: Optional[Mapping[str, Any]] = None,
) -> OperationContext:
    """
    MCP JSON-RPC request → OperationContext.

    Expected shape:
        {
            "id": "123",
            "method": "tools.call",
            "params": {
                "toolName": "...",
                "context": {
                    "tenant": "...",
                    "traceparent": "...",
                }
            }
        }

    Args:
        request: MCP JSON-RPC request
        connection_metadata: Optional connection-level metadata

    Returns:
        OperationContext
    """
    params = request.get("params") or {}
    context = params.get("context") or {}
    if not isinstance(context, Mapping):
        logger.debug(
            "MCP request.params.context is not a Mapping; got %r",
            type(context),
        )
        context = {}

    # Merge connection metadata
    conn_meta = connection_metadata or {}
    merged = dict(conn_meta)
    merged.update(context)

    method = request.get("method")
    tool_name = params.get("toolName") or params.get("tool_name")

    attrs: Dict[str, Any] = {ATTR_FRAMEWORK: "mcp"}
    if method:
        attrs["mcp_method"] = str(method)
    if tool_name:
        attrs["mcp_tool_name"] = str(tool_name)

    # Merge user attrs
    user_attrs = _get_dict(merged, "attrs")
    attrs.update(user_attrs)

    common = _extract_common_fields({}, merged)
    request_id = request.get("id")
    if request_id is not None:
        common["request_id"] = str(request_id)

    ctx_dict: NormalizedContext = {
        **common,
        "attrs": attrs,
    }
    _validate_normalized_context(ctx_dict, source="mcp")
    return OperationContext(**ctx_dict)


def from_dict(ctx: Optional[Mapping[str, Any]]) -> OperationContext:
    """
    Generic dict → OperationContext.

    Expected keys (all optional):
        {
            "request_id": ...,
            "tenant": ...,
            "deadline_ms": ...,
            "traceparent": ...,
            "attrs": {...}
        }

    Args:
        ctx: Dictionary with context fields

    Returns:
        OperationContext
    """
    if not ctx:
        ctx_dict = _empty_context()
        _validate_normalized_context(ctx_dict, source="dict")
        return OperationContext(**ctx_dict)

    attrs = dict(ctx.get("attrs") or {})
    common = _extract_common_fields(ctx, {})

    ctx_dict: NormalizedContext = {
        **common,
        "attrs": attrs,
    }
    _validate_normalized_context(ctx_dict, source="dict")
    return OperationContext(**ctx_dict)


# ---------------------------------------------------------------------------
# Round-trip helpers (for framework adapters that need to call back)
# ---------------------------------------------------------------------------


def to_langchain(ctx: OperationContext) -> Dict[str, Any]:
    """
    OperationContext → LangChain RunnableConfig-like dict.

    Preserves key Corpus context fields in metadata + tags.

    Args:
        ctx: OperationContext to convert

    Returns:
        Dict suitable for LangChain RunnableConfig
    """
    config: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}

    tenant = ctx.tenant
    deadline_ms = ctx.deadline_ms
    traceparent = ctx.traceparent
    request_id = ctx.request_id
    attrs = ctx.attrs or {}

    if tenant is not None:
        metadata[KEY_TENANT] = tenant
    if deadline_ms is not None:
        metadata[KEY_DEADLINE_MS] = deadline_ms
    if traceparent is not None:
        metadata[KEY_TRACEPARENT] = traceparent
    if request_id is not None:
        metadata[KEY_REQUEST_ID] = request_id
        config["run_id"] = request_id

    # Preserve attrs in metadata["attrs"]
    if attrs:
        attrs_copy = dict(attrs)
        tags = attrs_copy.pop("tags", None)
        if tags and isinstance(tags, (list, tuple)):
            config["tags"] = list(tags)
        metadata["attrs"] = attrs_copy

    if metadata:
        config["metadata"] = metadata

    return config


def to_llamaindex(ctx: OperationContext) -> Dict[str, Any]:
    """
    OperationContext → metadata dict for LlamaIndex.

    Args:
        ctx: OperationContext to convert

    Returns:
        Dict suitable for LlamaIndex metadata
    """
    metadata: Dict[str, Any] = {}

    request_id = ctx.request_id
    tenant = ctx.tenant
    deadline_ms = ctx.deadline_ms
    traceparent = ctx.traceparent
    attrs = ctx.attrs or {}

    if request_id is not None:
        metadata[KEY_REQUEST_ID] = request_id
    if tenant is not None:
        metadata[KEY_TENANT] = tenant
    if deadline_ms is not None:
        metadata[KEY_DEADLINE_MS] = deadline_ms
    if traceparent is not None:
        metadata[KEY_TRACEPARENT] = traceparent

    if attrs:
        metadata["attrs"] = dict(attrs)

    return metadata


def to_semantic_kernel(ctx: OperationContext) -> Dict[str, Any]:
    """
    OperationContext → SK-style variables dict.

    Args:
        ctx: OperationContext to convert

    Returns:
        Dict suitable for SK context variables
    """
    variables: Dict[str, Any] = {}

    request_id = ctx.request_id
    tenant = ctx.tenant
    deadline_ms = ctx.deadline_ms
    traceparent = ctx.traceparent
    attrs = ctx.attrs or {}

    if request_id is not None:
        variables[KEY_REQUEST_ID] = request_id
    if tenant is not None:
        variables[KEY_TENANT] = tenant
    if deadline_ms is not None:
        variables[KEY_DEADLINE_MS] = deadline_ms
    if traceparent is not None:
        variables[KEY_TRACEPARENT] = traceparent

    if attrs:
        for key, value in attrs.items():
            variables.setdefault(key, value)

    return variables


# ---------------------------------------------------------------------------
# Internal extraction / validation helpers
# ---------------------------------------------------------------------------


def _extract_common_fields(
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Extract all common fields in one pass.

    Returns:
        {
            "request_id": str | None,
            "tenant": str | None,
            "deadline_ms": int | None,
            "traceparent": str | None,
        }
    """
    return {
        "request_id": _extract_request_id(config, metadata),
        "tenant": _extract_tenant(metadata),
        "deadline_ms": _extract_deadline_ms(metadata),
        "traceparent": _extract_traceparent(metadata),
    }


def _validate_normalized_context(
    ctx: NormalizedContext,
    *,
    source: str = "unknown",
) -> None:
    """
    Validate normalized context structure before wrapping into OperationContext.

    Ensures the required keys are present. Does not validate types strictly,
    since different protocol layers may coerce or extend values differently.
    """
    required_keys = {"request_id", "tenant", "deadline_ms", "traceparent", "attrs"}
    missing = [key for key in required_keys if key not in ctx]
    if missing:
        logger.error(
            "Normalized context validation failed for source=%s; "
            "missing keys=%s; ctx=%r",
            source,
            missing,
            ctx,
        )
        raise ContextTranslationError(
            f"Missing required keys in normalized context: {missing}"
        )


def _empty_context() -> NormalizedContext:
    """Create a well-formed empty normalized context dict."""
    return {
        "request_id": None,
        "tenant": None,
        "deadline_ms": None,
        "traceparent": None,
        "attrs": {},
    }


def _extract_request_id(
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Optional[str]:
    """Extract request ID from config or metadata."""
    request_id = (
        config.get(KEY_REQUEST_ID)
        or config.get(KEY_RUN_ID)
        or config.get("id")
        or metadata.get(KEY_REQUEST_ID)
        or metadata.get(KEY_RUN_ID)
        or metadata.get("id")
    )
    return str(request_id) if request_id is not None else None


def _extract_tenant(metadata: Mapping[str, Any]) -> Optional[str]:
    """Extract tenant from metadata."""
    tenant = metadata.get(KEY_TENANT)
    return str(tenant) if tenant is not None else None


def _extract_deadline_ms(metadata: Mapping[str, Any]) -> Optional[int]:
    """Extract deadline in milliseconds from metadata."""
    deadline = (
        metadata.get(KEY_DEADLINE_MS)
        or metadata.get(KEY_TIMEOUT_MS)
        or metadata.get(KEY_TIMEOUT)
    )
    return _coerce_deadline_ms(deadline)


def _extract_traceparent(metadata: Mapping[str, Any]) -> Optional[str]:
    """Extract traceparent from metadata."""
    traceparent = (
        metadata.get(KEY_TRACEPARENT)
        or metadata.get("trace_parent")
    )
    return str(traceparent) if traceparent is not None else None


def _coerce_deadline_ms(value: Any) -> Optional[int]:
    """
    Coerce various deadline/timeout representations into milliseconds.

    Convention:
    - None → None
    - int/float < ~1e8 are treated as seconds and converted to ms
    - int/float >= ~1e8 are assumed to already be milliseconds
    - timedelta-like objects use total_seconds()
    """
    if value is None:
        return None

    # timedelta-like
    if hasattr(value, "total_seconds"):
        try:
            seconds = float(value.total_seconds())
            return int(seconds * 1000)
        except Exception:
            logger.debug("Failed to coerce deadline from timedelta-like: %r", value)
            return None

    # numeric
    if isinstance(value, (int, float)):
        # Heuristic: if > 100M, assume ms (≈ 3 years)
        if float(value) > 1e8:
            return int(value)
        return int(float(value) * 1000.0)

    logger.debug("Unsupported deadline/timeout type for coercion: %r", type(value))
    return None


def _safe_get(obj: Any, path: Sequence[Union[str, int]]) -> Any:
    """Safely traverse nested mapping/attribute paths."""
    current = obj
    for key in path:
        if current is None:
            return None
        try:
            if isinstance(key, int):
                current = current[key]
            elif isinstance(current, Mapping):
                current = current.get(key)
            else:
                current = getattr(current, key)
        except Exception:
            return None
    return current


def _get_dict(obj: Any, key: str) -> Dict[str, Any]:
    """Get a dict from obj[key] or obj.key, defaulting to empty dict."""
    if isinstance(obj, Mapping):
        value = obj.get(key, {})
    else:
        value = getattr(obj, key, {})
    if not isinstance(value, Mapping):
        if value:
            logger.debug(
                "_get_dict expected Mapping for key=%r, got %r; value ignored",
                key,
                type(value),
            )
        return {}
    return dict(value)


__all__ = [
    "ContextTranslationError",
    "register_framework_translator",
    "get_framework_translator",
    "translate_framework",
    "from_langchain",
    "from_llamaindex",
    "from_semantic_kernel",
    "from_autogen",
    "from_crewai",
    "from_mcp",
    "from_dict",
    "to_langchain",
    "to_llamaindex",
    "to_semantic_kernel",
]
