# corpus_sdk/llm/framework_adapters/common/context_translation.py
# SPDX-License-Identifier: Apache-2.0

"""
Context translation utilities for Corpus framework adapters.

This module normalizes framework-specific "context" objects into Corpus
`OperationContext` instances so that:

- Request IDs / trace IDs are preserved
- Deadlines / timeouts propagate correctly
- Tenant / auth / tags are carried across protocol boundaries
- Framework-specific metadata is captured without polluting core types

Design goals
------------
- Protocol-first: this is SDK infrastructure, not business logic.
- Framework-agnostic: no hard runtime dependency on any single framework.
- Non-destructive: never drop context fields unless clearly unsafe.
- Debuggable: provide helpers to snapshot/serialize context for logs/metrics.

Primary entry points
--------------------
- ContextTranslator.from_langchain_config
- ContextTranslator.from_llamaindex_callback_manager
- ContextTranslator.from_semantic_kernel_context
- ContextTranslator.from_autogen_context
- ContextTranslator.from_crewai_context
- ContextTranslator.from_mcp_request
- ContextTranslator.from_generic_dict

Optional round-trip helpers
---------------------------
- ContextTranslator.to_langchain_config
- ContextTranslator.to_llamaindex_metadata
- ContextTranslator.to_semantic_kernel_context

Debug helpers
-------------
- snapshot_raw_context
- serialize_operation_context
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Union,
    TYPE_CHECKING,
)

from corpus_sdk.llm.llm_base import OperationContext

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    # Optional imports for type hints; we explicitly avoid hard runtime deps.
    try:
        from langchain_core.runnables import RunnableConfig  # type: ignore[import]
    except Exception:  # pragma: no cover
        RunnableConfig = Mapping[str, Any]  # type: ignore[misc]

    try:
        from llama_index.core.callbacks import CallbackManager  # type: ignore[import]
    except Exception:  # pragma: no cover
        CallbackManager = Any  # type: ignore[misc]

    try:
        from semantic_kernel.kernel_pydantic import KernelContext  # type: ignore[import]
        from semantic_kernel.connectors.ai.prompt_execution_settings import (
            PromptExecutionSettings,
        )  # type: ignore[import]
    except Exception:  # pragma: no cover
        KernelContext = Any  # type: ignore[misc]
        PromptExecutionSettings = Any  # type: ignore[misc]


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reserved keys / constants
# ---------------------------------------------------------------------------

#: Reserved attribute namespace for Corpus-specific fields injected into attrs
CORPUS_ATTR_PREFIX = "corpus_"

#: Reserved keys commonly used inside OperationContext.attrs
ATTR_KEY_FRAMEWORK = f"{CORPUS_ATTR_PREFIX}framework"
ATTR_KEY_FRAMEWORK_VERSION = f"{CORPUS_ATTR_PREFIX}framework_version"
ATTR_KEY_FRAMEWORK_RUN_NAME = f"{CORPUS_ATTR_PREFIX}framework_run_name"
ATTR_KEY_LANGCHAIN_RECURSION_LIMIT = f"{CORPUS_ATTR_PREFIX}langchain_recursion_limit"
ATTR_KEY_AUTOGEN_CONVERSATION_ID = f"{CORPUS_ATTR_PREFIX}autogen_conversation_id"
ATTR_KEY_CREWAI_TASK_ID = f"{CORPUS_ATTR_PREFIX}crewai_task_id"
ATTR_KEY_MCP_METHOD = f"{CORPUS_ATTR_PREFIX}mcp_method"
ATTR_KEY_MCP_TOOL_NAME = f"{CORPUS_ATTR_PREFIX}mcp_tool_name"

#: Common metadata keys we try to honor when present
META_KEY_TENANT = "tenant"
META_KEY_DEADLINE_MS = "deadline_ms"
META_KEY_TRACEPARENT = "traceparent"
META_KEY_REQUEST_ID = "request_id"
META_KEY_CORRELATION_ID = "correlation_id"

#: Maximum length for raw context repr in debug snapshots
MAX_RAW_CONTEXT_REPR_LEN = 4096


# ---------------------------------------------------------------------------
# Helper data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeadlineInfo:
    """Normalized view of a deadline/timeout value in milliseconds."""

    deadline_ms: Optional[int]
    source: Optional[str] = None


@dataclass(frozen=True)
class TracingInfo:
    """
    Tracing identifiers extracted from framework context.

    Not all fields will be populated for every framework.
    """

    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    traceparent: Optional[str] = None
    span_id: Optional[str] = None
    correlation_id: Optional[str] = None
    source: Optional[str] = None


@dataclass(frozen=True)
class RawContextSnapshot:
    """
    Serializable snapshot of a raw framework context object.

    Used for diagnostics / observability – not in hot paths.
    """

    framework: str
    raw_type: str
    raw_repr: str
    normalized: Dict[str, Any]


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _coerce_deadline_ms(value: Any, *, source: Optional[str] = None) -> Optional[int]:
    """
    Coerce various deadline/timeout representations into milliseconds.

    Accepts:
    - None → None
    - int/float (assumed seconds if < 10^8, otherwise ms)
    - dicts like {"seconds": 10} or {"ms": 500}
    - datetime.timedelta-like objects (if they have total_seconds())

    When heuristics are applied or values are discarded, a debug log
    with the given `source` (framework / adapter name) is emitted.
    """
    if value is None:
        return None

    src = f" [{source}]" if source else ""

    # timedelta-like
    if hasattr(value, "total_seconds"):
        try:
            seconds = float(value.total_seconds())
            return int(seconds * 1000)
        except Exception:
            logger.debug("Failed to coerce timedelta-like deadline%s: %r", src, value)
            return None

    # numeric
    if isinstance(value, (int, float)):
        # Heuristic: if > ~3 years worth of seconds, assume ms
        if float(value) > 1e8:
            logger.debug(
                "Interpreting large numeric deadline as milliseconds%s: %r",
                src,
                value,
            )
            return int(value)
        return int(float(value) * 1000.0)

    # dict-like
    if isinstance(value, Mapping):
        # Prefer explicit ms
        if "ms" in value:
            try:
                return int(value["ms"])
            except Exception:
                logger.debug("Failed to coerce dict-based ms deadline%s: %r", src, value)
                return None
        # Fallback to seconds
        if "seconds" in value:
            try:
                return int(float(value["seconds"]) * 1000.0)
            except Exception:
                logger.debug(
                    "Failed to coerce dict-based seconds deadline%s: %r",
                    src,
                    value,
                )
                return None

    # Unknown format – don't guess.
    logger.debug(
        "Ignoring unsupported deadline format%s; value=%r", src, value
    )
    return None


def _safe_get(obj: Any, path: Sequence[Union[str, int]]) -> Any:
    """
    Safely traverse nested mapping/attribute paths.

    Example:
        _safe_get(cfg, ["metadata", "trace_id"])
        _safe_get(cb, ["context", "trace_id"])
    """
    current: Any = obj
    for key in path:
        if current is None:
            return None
        try:
            if isinstance(key, int):
                current = current[key]  # type: ignore[index]
            else:
                # Try mapping access first
                if isinstance(current, Mapping):
                    current = current.get(key)
                else:
                    current = getattr(current, key)
        except Exception:
            return None
    return current


def _extract_trace_from_metadata(metadata: Mapping[str, Any], source: str) -> TracingInfo:
    """
    Extract tracing identifiers from a generic metadata mapping.
    """
    request_id = (
        metadata.get(META_KEY_REQUEST_ID)
        or metadata.get("run_id")
        or metadata.get("id")
        or metadata.get("requestId")
    )
    traceparent = metadata.get(META_KEY_TRACEPARENT) or metadata.get("trace_parent")
    trace_id = metadata.get("trace_id") or metadata.get("traceId")
    span_id = metadata.get("span_id") or metadata.get("spanId")
    correlation_id = (
        metadata.get(META_KEY_CORRELATION_ID)
        or metadata.get("operation_id")
        or metadata.get("operationId")
    )

    return TracingInfo(
        request_id=str(request_id) if request_id is not None else None,
        trace_id=str(trace_id) if trace_id is not None else None,
        traceparent=str(traceparent) if traceparent is not None else None,
        span_id=str(span_id) if span_id is not None else None,
        correlation_id=str(correlation_id) if correlation_id is not None else None,
        source=source,
    )


def _merge_attrs(*sources: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple attribute dicts, later sources taking precedence,
    with a special rule for Corpus-reserved keys.

    - For normal keys: later sources overwrite earlier sources.
    - For Corpus-reserved keys (prefix `CORPUS_ATTR_PREFIX`):
        *once set they are never overwritten* by later sources.

    Convention:
        Pass Corpus/internal attrs *first* and external/user attrs *later*
        so that user-provided values cannot clobber reserved Corpus keys.
    """
    merged: Dict[str, Any] = {}
    for src in sources:
        for key, value in (src or {}).items():
            if key.startswith(CORPUS_ATTR_PREFIX) and key in merged:
                # Don't overwrite existing Corpus-reserved attributes
                continue
            merged[key] = value
    return merged


def _normalize_meta(
    metadata: Mapping[str, Any],
    *,
    source: str,
    settings: Optional[Any] = None,
) -> tuple[TracingInfo, DeadlineInfo, Optional[str]]:
    """
    Normalize common context metadata into tracing, deadline, and tenant.

    Applies the standard Corpus conventions for:
    - request/trace IDs via _extract_trace_from_metadata
    - deadline_ms / timeout_ms / timeout via _coerce_deadline_ms
      (optionally augmented by a `settings` object with timeout fields)
    - tenant via META_KEY_TENANT

    If both metadata and settings provide a deadline/timeout, the
    `settings` value wins and a debug message is emitted for clarity.
    """
    trace_info = _extract_trace_from_metadata(metadata, source=source)

    # Metadata-based deadline
    raw_deadline = (
        metadata.get(META_KEY_DEADLINE_MS)
        or metadata.get("timeout_ms")
        or metadata.get("timeout")
    )
    deadline_ms = _coerce_deadline_ms(raw_deadline, source=source)

    # Settings-based deadline may override metadata if present
    if settings is not None:
        settings_deadline = getattr(settings, "timeout_ms", None) or getattr(
            settings, "timeout", None
        )
        if settings_deadline is not None:
            logger.debug(
                "Using settings-based timeout over metadata-based deadline "
                "for source '%s'", source
            )
            deadline_ms = _coerce_deadline_ms(settings_deadline, source=f"{source}.settings")

    deadline_info = DeadlineInfo(deadline_ms=deadline_ms, source=source)
    tenant = metadata.get(META_KEY_TENANT)

    return trace_info, deadline_info, tenant


# ---------------------------------------------------------------------------
# Main context translator
# ---------------------------------------------------------------------------


class ContextTranslator:
    """
    Translate framework-specific context/config into Corpus OperationContext.

    All methods are static to keep this class stateless and easy to use
    from any adapter module.

    Note:
        We deliberately do *not* apply caching here. Context objects are
        often short-lived and sometimes mutable; this module is also not
        expected to be a hot path compared to model calls, so caching
        adds complexity without clear benefit.
    """

    # ------------------------------------------------------------------ #
    # LangChain
    # ------------------------------------------------------------------ #

    @staticmethod
    def from_langchain_config(
        config: Optional[Mapping[str, Any]],
        *,
        framework_version: Optional[str] = None,
    ) -> OperationContext:
        """
        LangChain RunnableConfig → OperationContext.

        Expected shape (informal):
            {
                "run_id": ...,
                "tags": [...],
                "metadata": {
                    "tenant": ...,
                    "deadline_ms": ...,
                    "traceparent": ...,
                    ...
                },
                "recursion_limit": ...,
                ...
            }
        """
        if not config:
            return OperationContext()

        raw_metadata = config.get("metadata") or {}
        metadata: Mapping[str, Any] = raw_metadata if isinstance(raw_metadata, Mapping) else {}

        trace_info, deadline_info, tenant = _normalize_meta(
            metadata, source="langchain.config"
        )

        # Tags and recursion
        tags = list(config.get("tags") or [])  # type: ignore[arg-type]
        recursion_limit = config.get("recursion_limit")

        attrs: Dict[str, Any] = {
            "tags": tags,
            ATTR_KEY_FRAMEWORK: "langchain",
        }
        if framework_version:
            attrs[ATTR_KEY_FRAMEWORK_VERSION] = framework_version
        if recursion_limit is not None:
            attrs[ATTR_KEY_LANGCHAIN_RECURSION_LIMIT] = recursion_limit
        if "run_name" in config:
            attrs[ATTR_KEY_FRAMEWORK_RUN_NAME] = config.get("run_name")

        meta_attrs_obj = metadata.get("attrs")
        meta_attrs = meta_attrs_obj if isinstance(meta_attrs_obj, Mapping) else {}

        # Corpus attrs go first so they win for corpus_* keys.
        attrs = _merge_attrs(attrs, meta_attrs)

        return OperationContext(
            request_id=trace_info.request_id
            or (str(config.get("run_id")) if config.get("run_id") else None),
            tenant=str(tenant) if tenant is not None else None,
            deadline_ms=deadline_info.deadline_ms,
            traceparent=trace_info.traceparent,
            attrs=attrs,
        )

    # ------------------------------------------------------------------ #
    # LlamaIndex
    # ------------------------------------------------------------------ #

    @staticmethod
    def from_llamaindex_callback_manager(
        cbm: Optional[Any],
        *,
        framework_version: Optional[str] = None,
    ) -> OperationContext:
        """
        LlamaIndex CallbackManager → OperationContext.

        We try to extract:
        - trace_id / span_id
        - run_id / request_id
        - any extra metadata exposed on the callback manager.
        """
        if cbm is None:
            return OperationContext()

        # Try a few common attributes
        trace_id = getattr(cbm, "trace_id", None)
        span_id = getattr(cbm, "span_id", None)
        request_id = getattr(cbm, "request_id", None) or getattr(cbm, "run_id", None)

        # Some versions expose a dict-like metadata/context
        meta_candidate = getattr(cbm, "context", None) or getattr(cbm, "metadata", None)
        metadata: Mapping[str, Any] = meta_candidate if isinstance(meta_candidate, Mapping) else {}

        trace_info, deadline_info, tenant = _normalize_meta(
            metadata, source="llamaindex.callback_manager"
        )

        attrs: Dict[str, Any] = {
            ATTR_KEY_FRAMEWORK: "llamaindex",
        }
        if framework_version:
            attrs[ATTR_KEY_FRAMEWORK_VERSION] = framework_version

        # Attach any "tags" like thing if present
        tags = getattr(cbm, "tags", None)
        if tags:
            attrs["tags"] = list(tags)

        meta_attrs_obj = metadata.get("attrs")
        meta_attrs = meta_attrs_obj if isinstance(meta_attrs_obj, Mapping) else {}

        attrs = _merge_attrs(attrs, meta_attrs)

        return OperationContext(
            request_id=(
                trace_info.request_id
                or (str(request_id) if request_id is not None else None)
                or (str(trace_id) if trace_id is not None else None)
            ),
            tenant=str(tenant) if tenant is not None else None,
            # Prefer explicit deadlines if LlamaIndex starts surfacing them.
            deadline_ms=deadline_info.deadline_ms,
            traceparent=trace_info.traceparent,
            attrs=attrs,
        )

    # ------------------------------------------------------------------ #
    # Semantic Kernel
    # ------------------------------------------------------------------ #

    @staticmethod
    def from_semantic_kernel_context(
        sk_context: Optional[Any],
        *,
        settings: Optional[Any] = None,
        framework_version: Optional[str] = None,
    ) -> OperationContext:
        """
        Semantic Kernel context → OperationContext.

        We treat `sk_context` as a generic object that *may* expose:
        - variables / metadata dict
        - correlation / operation IDs
        - tenant hints

        `settings` (PromptExecutionSettings) may carry timeout-type info.
        """
        if sk_context is None and settings is None:
            return OperationContext()

        # SK often has a "variables" OR "context_variables" mapping
        raw_meta = (
            _safe_get(sk_context, ["variables"])  # type: ignore[arg-type]
            or _safe_get(sk_context, ["context_variables"])  # type: ignore[arg-type]
            or {}
        )
        metadata: Mapping[str, Any] = raw_meta if isinstance(raw_meta, Mapping) else {}

        trace_info, deadline_info, tenant = _normalize_meta(
            metadata,
            source="semantic_kernel.context",
            settings=settings,
        )

        attrs: Dict[str, Any] = {
            ATTR_KEY_FRAMEWORK: "semantic_kernel",
        }
        if framework_version:
            attrs[ATTR_KEY_FRAMEWORK_VERSION] = framework_version

        # If we know function name / plugin, try to include
        function_name = _safe_get(sk_context, ["function", "name"])
        plugin_name = _safe_get(sk_context, ["plugin", "name"])
        if function_name:
            attrs.setdefault("sk_function_name", function_name)
        if plugin_name:
            attrs.setdefault("sk_plugin_name", plugin_name)

        meta_attrs_obj = metadata.get("attrs")
        meta_attrs = meta_attrs_obj if isinstance(meta_attrs_obj, Mapping) else {}

        attrs = _merge_attrs(attrs, meta_attrs)

        return OperationContext(
            request_id=trace_info.request_id,
            tenant=str(tenant) if tenant is not None else None,
            deadline_ms=deadline_info.deadline_ms,
            traceparent=trace_info.traceparent,
            attrs=attrs,
        )

    # ------------------------------------------------------------------ #
    # AutoGen
    # ------------------------------------------------------------------ #

    @staticmethod
    def from_autogen_context(
        conversation: Optional[Any] = None,
        *,
        framework_version: Optional[str] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> OperationContext:
        """
        AutoGen conversation / context → OperationContext.

        We do not depend on concrete AutoGen types, but heuristically look for:
        - conversation_id / id
        - summary_id / run_id / request_id
        - metadata / config dicts with tenant, deadline, trace info
        """
        if conversation is None and not extra:
            return OperationContext()

        # Try to detect IDs on the conversation object
        conversation_id = getattr(conversation, "conversation_id", None) or getattr(
            conversation, "id", None
        )
        run_id = getattr(conversation, "run_id", None)
        request_id = getattr(conversation, "request_id", None)

        meta_candidate = (
            getattr(conversation, "metadata", None)
            or getattr(conversation, "config", None)
            or getattr(conversation, "context", None)
            or {}
        )
        metadata: Mapping[str, Any] = meta_candidate if isinstance(meta_candidate, Mapping) else {}

        # Merge in extra context from kwargs
        extra_metadata: Mapping[str, Any] = extra or {}

        merged_meta: Dict[str, Any] = dict(metadata)
        merged_meta.update(extra_metadata)

        trace_info, deadline_info, tenant = _normalize_meta(
            merged_meta, source="autogen.context"
        )

        attrs: Dict[str, Any] = {
            ATTR_KEY_FRAMEWORK: "autogen",
        }
        if framework_version:
            attrs[ATTR_KEY_FRAMEWORK_VERSION] = framework_version
        if conversation_id is not None:
            attrs[ATTR_KEY_AUTOGEN_CONVERSATION_ID] = str(conversation_id)

        meta_attrs_obj = merged_meta.get("attrs")
        meta_attrs = meta_attrs_obj if isinstance(meta_attrs_obj, Mapping) else {}

        attrs = _merge_attrs(attrs, meta_attrs)

        return OperationContext(
            request_id=(
                trace_info.request_id
                or (str(request_id) if request_id is not None else None)
                or (str(run_id) if run_id is not None else None)
            ),
            tenant=str(tenant) if tenant is not None else None,
            deadline_ms=deadline_info.deadline_ms,
            traceparent=trace_info.traceparent,
            attrs=attrs,
        )

    # ------------------------------------------------------------------ #
    # CrewAI
    # ------------------------------------------------------------------ #

    @staticmethod
    def from_crewai_context(
        task: Optional[Any] = None,
        *,
        framework_version: Optional[str] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> OperationContext:
        """
        CrewAI task / run context → OperationContext.

        Heuristics:
        - task.id / task.task_id
        - task.crew.name / task.agent.name
        - task.metadata / extra dicts
        """
        if task is None and not extra:
            return OperationContext()

        task_id = getattr(task, "id", None) or getattr(task, "task_id", None)
        run_id = getattr(task, "run_id", None)
        request_id = getattr(task, "request_id", None)

        crew_name = _safe_get(task, ["crew", "name"])
        agent_name = _safe_get(task, ["agent", "name"])

        meta_candidate = getattr(task, "metadata", None) or {}
        metadata: Mapping[str, Any] = meta_candidate if isinstance(meta_candidate, Mapping) else {}

        extra_metadata: Mapping[str, Any] = extra or {}
        merged_meta: Dict[str, Any] = dict(metadata)
        merged_meta.update(extra_metadata)

        trace_info, deadline_info, tenant = _normalize_meta(
            merged_meta, source="crewai.context"
        )

        attrs: Dict[str, Any] = {
            ATTR_KEY_FRAMEWORK: "crewai",
        }
        if framework_version:
            attrs[ATTR_KEY_FRAMEWORK_VERSION] = framework_version
        if task_id is not None:
            attrs[ATTR_KEY_CREWAI_TASK_ID] = str(task_id)
        if crew_name:
            attrs["crew_name"] = crew_name
        if agent_name:
            attrs["agent_name"] = agent_name

        meta_attrs_obj = merged_meta.get("attrs")
        meta_attrs = meta_attrs_obj if isinstance(meta_attrs_obj, Mapping) else {}

        attrs = _merge_attrs(attrs, meta_attrs)

        return OperationContext(
            request_id=(
                trace_info.request_id
                or (str(request_id) if request_id is not None else None)
                or (str(run_id) if run_id is not None else None)
            ),
            tenant=str(tenant) if tenant is not None else None,
            deadline_ms=deadline_info.deadline_ms,
            traceparent=trace_info.traceparent,
            attrs=attrs,
        )

    # ------------------------------------------------------------------ #
    # MCP (Model Context Protocol)
    # ------------------------------------------------------------------ #

    @staticmethod
    def from_mcp_request(
        request: Mapping[str, Any],
        *,
        connection_metadata: Optional[Mapping[str, Any]] = None,
    ) -> OperationContext:
        """
        MCP JSON-RPC request → OperationContext.

        Expected shape (informal):
            {
                "id": "123",
                "method": "tools.call",
                "params": {
                    "toolName": "...",
                    "arguments": {...},
                    "context": {
                        "tenant": "...",
                        "traceparent": "...",
                        "deadline_ms": ...
                    }
                }
            }
        """
        params = request.get("params") or {}
        context = params.get("context") or {}
        if not isinstance(context, Mapping):
            context = {}

        # Combine context with connection-level metadata (if any)
        conn_meta = connection_metadata or {}
        merged_meta: Dict[str, Any] = dict(conn_meta)
        merged_meta.update(context)

        trace_info, deadline_info, tenant = _normalize_meta(
            merged_meta, source="mcp.request"
        )

        method = request.get("method")
        tool_name = params.get("toolName") or params.get("tool_name")

        attrs: Dict[str, Any] = {
            ATTR_KEY_FRAMEWORK: "mcp",
        }
        if method:
            attrs[ATTR_KEY_MCP_METHOD] = str(method)
        if tool_name:
            attrs[ATTR_KEY_MCP_TOOL_NAME] = str(tool_name)

        meta_attrs_obj = merged_meta.get("attrs")
        meta_attrs = meta_attrs_obj if isinstance(meta_attrs_obj, Mapping) else {}

        attrs = _merge_attrs(attrs, meta_attrs)

        return OperationContext(
            request_id=str(request.get("id")) if request.get("id") is not None else trace_info.request_id,
            tenant=str(tenant) if tenant is not None else None,
            deadline_ms=deadline_info.deadline_ms,
            traceparent=trace_info.traceparent,
            attrs=attrs,
        )

    # ------------------------------------------------------------------ #
    # Generic dict
    # ------------------------------------------------------------------ #

    @staticmethod
    def from_generic_dict(ctx: Optional[Mapping[str, Any]]) -> OperationContext:
        """
        Generic dict → OperationContext.

        Expected keys (all optional):
            {
                "request_id": ...,
                "tenant": ...,
                "deadline_ms": ... (ms),
                "traceparent": ...,
                "attrs": {...}
            }
        """
        if not ctx:
            return OperationContext()

        attrs = dict(ctx.get("attrs") or {})

        trace_info, deadline_info, tenant = _normalize_meta(
            ctx, source="generic.dict"
        )

        return OperationContext(
            request_id=trace_info.request_id,
            tenant=str(tenant) if tenant is not None else None,
            deadline_ms=deadline_info.deadline_ms,
            traceparent=trace_info.traceparent,
            attrs=attrs,
        )

    # ------------------------------------------------------------------ #
    # Optional "to_*" helpers for round-trip propagation
    # ------------------------------------------------------------------ #

    @staticmethod
    def to_langchain_config(
        ctx: OperationContext,
        *,
        base: Optional[MutableMapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        OperationContext → LangChain RunnableConfig-like dict.

        This does *not* guarantee a 1:1 mapping with all LangChain features,
        but it preserves key Corpus context fields in `metadata` + `tags`.
        """
        config: Dict[str, Any] = dict(base or {})

        metadata = dict(config.get("metadata") or {})
        attrs = dict(ctx.attrs or {})

        if ctx.tenant is not None:
            metadata[META_KEY_TENANT] = ctx.tenant
        if ctx.deadline_ms is not None:
            metadata[META_KEY_DEADLINE_MS] = ctx.deadline_ms
        if ctx.traceparent is not None:
            metadata[META_KEY_TRACEPARENT] = ctx.traceparent
        if ctx.request_id is not None:
            metadata[META_KEY_REQUEST_ID] = ctx.request_id

        # Preserve attrs in metadata["attrs"]
        meta_attrs = dict(metadata.get("attrs") or {})
        meta_attrs.update(attrs)
        metadata["attrs"] = meta_attrs

        config["metadata"] = metadata

        # Use tags from attrs if present
        tags = attrs.get("tags")
        if isinstance(tags, (list, tuple)):
            config.setdefault("tags", list(tags))

        # Run ID as top-level if present
        if ctx.request_id is not None:
            config.setdefault("run_id", ctx.request_id)

        return config

    @staticmethod
    def to_llamaindex_metadata(ctx: OperationContext) -> Dict[str, Any]:
        """
        OperationContext → metadata dict suitable for LlamaIndex global config.

        This does *not* construct full LlamaIndex objects; it's simply a
        payload that adapters can attach to their configuration / callback
        manager.
        """
        metadata: Dict[str, Any] = {}

        if ctx.request_id is not None:
            metadata[META_KEY_REQUEST_ID] = ctx.request_id
        if ctx.tenant is not None:
            metadata[META_KEY_TENANT] = ctx.tenant
        if ctx.deadline_ms is not None:
            metadata[META_KEY_DEADLINE_MS] = ctx.deadline_ms
        if ctx.traceparent is not None:
            metadata[META_KEY_TRACEPARENT] = ctx.traceparent

        if ctx.attrs:
            attrs = dict(ctx.attrs)
            metadata["attrs"] = attrs

        return metadata

    @staticmethod
    def to_semantic_kernel_context(
        ctx: OperationContext,
        *,
        base_variables: Optional[MutableMapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        OperationContext → SK-style variables dict.

        We do not instantiate full SK KernelContext objects here to avoid a
        hard dependency. Instead we return a dict that can be fed into SK
        context constructors.
        """
        variables: Dict[str, Any] = dict(base_variables or {})

        if ctx.request_id is not None:
            variables[META_KEY_REQUEST_ID] = ctx.request_id
        if ctx.tenant is not None:
            variables[META_KEY_TENANT] = ctx.tenant
        if ctx.deadline_ms is not None:
            variables[META_KEY_DEADLINE_MS] = ctx.deadline_ms
        if ctx.traceparent is not None:
            variables[META_KEY_TRACEPARENT] = ctx.traceparent

        for key, value in (ctx.attrs or {}).items():
            # Do not clobber explicit variables
            variables.setdefault(key, value)

        return variables


# ---------------------------------------------------------------------------
# Debug / introspection helpers
# ---------------------------------------------------------------------------


def snapshot_raw_context(framework: str, raw_ctx: Any) -> RawContextSnapshot:
    """
    Create a debug snapshot of a raw framework context object.

    Intended for logging / metrics / diagnostics. We try to create a
    reasonably small, JSON-serializable normalized view without
    traversing large graphs or dumping unbounded reprs.
    """
    raw_type = type(raw_ctx).__name__
    try:
        raw_repr = repr(raw_ctx)
    except Exception:
        raw_repr = f"<unreprable {raw_type}>"

    if len(raw_repr) > MAX_RAW_CONTEXT_REPR_LEN:
        raw_repr = raw_repr[:MAX_RAW_CONTEXT_REPR_LEN] + "... (truncated)"

    normalized: Dict[str, Any] = {}

    # Try to extract a few common fields
    if isinstance(raw_ctx, Mapping):
        for key in ("run_id", "request_id", "tenant", "trace_id", "deadline_ms", "timeout"):
            if key in raw_ctx:
                normalized[key] = raw_ctx[key]
    else:
        for attr in ("run_id", "request_id", "tenant", "trace_id", "deadline_ms", "timeout"):
            if hasattr(raw_ctx, attr):
                try:
                    normalized[attr] = getattr(raw_ctx, attr)
                except Exception:
                    continue

    return RawContextSnapshot(
        framework=framework,
        raw_type=raw_type,
        raw_repr=raw_repr,
        normalized=normalized,
    )


def serialize_operation_context(ctx: OperationContext) -> Dict[str, Any]:
    """
    Serialize an OperationContext into a JSON-friendly dict.

    NOTE:
        This is meant for logs/metrics. If you have PII/PHI in attrs,
        you should filter/mask it *before* calling this helper.
    """
    return {
        "request_id": getattr(ctx, "request_id", None),
        "tenant": getattr(ctx, "tenant", None),
        "deadline_ms": getattr(ctx, "deadline_ms", None),
        "traceparent": getattr(ctx, "traceparent", None),
        "attrs": dict(getattr(ctx, "attrs", {}) or {}),
    }


__all__ = [
    "ContextTranslator",
    "DeadlineInfo",
    "TracingInfo",
    "RawContextSnapshot",
    "snapshot_raw_context",
    "serialize_operation_context",
]
