from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    TYPE_CHECKING,
)

# ---------------------------------------------------------------------------
# Optional typing imports for external frameworks (for type checkers only)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage  # type: ignore
    from llama_index.llms.base import ChatMessage  # type: ignore

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STRICT_ROLE_VALIDATION: bool = os.getenv("CORPUS_STRICT_ROLES", "0") in {
    "1",
    "true",
    "TRUE",
}

# Fallback role to use when an unknown role is encountered and strict
# validation is disabled. Must be one of the canonical roles or "assistant".
_UNKNOWN_ROLE_FALLBACK: str = os.getenv(
    "CORPUS_UNKNOWN_ROLE", "assistant"
).strip().lower()


def set_strict_role_validation(enabled: bool) -> None:
    """Enable or disable strict role validation at runtime."""
    global STRICT_ROLE_VALIDATION
    STRICT_ROLE_VALIDATION = bool(enabled)


def set_unknown_role_fallback(fallback: str) -> None:
    """
    Configure the fallback canonical role used for unknown roles
    when STRICT_ROLE_VALIDATION is disabled.

    Allowed values: "system", "user", "assistant", "tool", "function".
    """
    global _UNKNOWN_ROLE_FALLBACK
    value = str(fallback).strip().lower()
    if value not in {"system", "user", "assistant", "tool", "function"}:
        raise ValueError(
            "Unknown role fallback must be one of "
            '{"system", "user", "assistant", "tool", "function"}, '
            f"got {fallback!r}"
        )
    _UNKNOWN_ROLE_FALLBACK = value


@dataclass
class NormalizedMessage:
    """
    Protocol-centric representation of a single chat message.

    Attributes
    ----------
    role:
        Canonical role string. Typically one of:
        "system", "user", "assistant", "tool", "function".
        Custom roles are allowed but highly discouraged unless you know
        every consumer understands them.

        If a framework provides a non-standard role (e.g. "critic",
        "planner"), the original value is preserved in
        `metadata["corpus_original_role"]` and `role` is mapped to the
        closest protocol role.

    content:
        String form of the message. If the source framework uses rich or
        structured content (e.g. OpenAI-style content parts, tool calls,
        images), we:

        * Convert a human-readable representation into this string, AND
        * Store the full structured value in
          `metadata["corpus_raw_content"]` so adapters can rehydrate it.

    metadata:
        Arbitrary metadata carried through the pipeline. We reserve a
        few namespaced keys:

        * "corpus_raw"           – the original framework message object
        * "corpus_raw_content"   – original structured content (if any)
        * "corpus_original_role" – non-canonical role when remapped
    """

    role: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Convenience helpers for framework adapters / tests ------------------

    def copy_with_content(self, content: str) -> "NormalizedMessage":
        """Cheap helper for adapters that only mutate content."""
        return NormalizedMessage(
            role=self.role,
            content=content,
            metadata=dict(self.metadata),
        )

    @property
    def raw_content(self) -> Optional[Any]:
        """
        Structured content, if any, preserved from the framework.

        This is whatever was originally placed in `corpus_raw_content`
        by the `from_*` functions (e.g. list[parts], dict payload).
        """
        return self.metadata.get("corpus_raw_content")

    @property
    def original_role(self) -> Optional[str]:
        """
        Original framework-specific role before normalization.

        For example, a LangChain `HumanMessage` might yield:
            role == "user"
            original_role == "human"
        """
        val = self.metadata.get("corpus_original_role")
        return str(val) if val is not None else None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_CANONICAL_ROLE_MAP: Dict[str, str] = {
    # System
    "system": "system",
    # User / human
    "user": "user",
    "human": "user",
    "humanmessage": "user",
    # Assistant / AI
    "assistant": "assistant",
    "ai": "assistant",
    "aimessage": "assistant",
    "model": "assistant",
    # Tools / functions
    "function": "function",
    "tool": "tool",
    "tool_message": "tool",
    "toolmessage": "tool",
}


def _normalize_role(
    raw_role: Optional[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Map a framework-specific role into a protocol role.

    * Uses a conservative mapping table.
    * Preserves the original role in metadata["corpus_original_role"] if changed.
    * If STRICT_ROLE_VALIDATION is enabled and the role is unknown, raises ValueError.
    """
    if metadata is None:
        metadata = {}

    if raw_role is None:
        return "user", metadata  # safe default

    original = str(raw_role)
    key = original.strip().lower()

    if key in _CANONICAL_ROLE_MAP:
        canonical = _CANONICAL_ROLE_MAP[key]
        if canonical != original:
            metadata.setdefault("corpus_original_role", original)
        return canonical, metadata

    # Exotic or framework-specific role.
    if STRICT_ROLE_VALIDATION:
        raise ValueError(f"Unknown message role: {original!r}")

    # Default: apply fallback canonical role but record the original.
    metadata.setdefault("corpus_original_role", original)
    fallback = _UNKNOWN_ROLE_FALLBACK
    if fallback not in {"system", "user", "assistant", "tool", "function"}:
        fallback = "assistant"
    return fallback, metadata


def _stringify_content(content: Any) -> Tuple[str, Optional[Any]]:
    """
    Convert arbitrary content into a string while preserving structure.

    Returns (string_form, raw_structured_or_None).

    Strategy:
    * If already a string -> pass through.
    * If dict or list[dict] -> JSON-encode (stable, explicit, sorted keys) and
      return the original structure so we can stash it in metadata.
    * Otherwise -> str() and don't treat as structured.

    We also guard against deep / recursive structures by catching
    RecursionError when materializing lists or dumping JSON.
    """
    if isinstance(content, str):
        return content, None

    # Mapping -> JSON if possible
    if isinstance(content, Mapping):
        try:
            return (
                json.dumps(content, ensure_ascii=False, sort_keys=True),
                content,
            )
        except (TypeError, RecursionError):
            return repr(content), content

    # Try to treat as generic iterable (but not string)
    try:
        as_list = list(content)  # type: ignore[arg-type]
    except (TypeError, RecursionError):
        # Not iterable or too recursive; fall back to simple string.
        return str(content), None

    # Common OpenAI-style content parts: [{"type": "...", ...}, ...]
    if as_list and all(isinstance(p, Mapping) for p in as_list):
        try:
            return (
                json.dumps(as_list, ensure_ascii=False, sort_keys=True),
                as_list,
            )
        except (TypeError, RecursionError):
            return repr(as_list), as_list

    # Fallback for "other" iterables
    return str(as_list), as_list


def _with_raw_metadata(
    msg: NormalizedMessage,
    raw: Any,
    raw_content: Optional[Any],
) -> NormalizedMessage:
    """
    Attach raw framework objects into metadata in a reserved namespace.
    """
    meta = dict(msg.metadata)
    meta.setdefault("corpus_raw", raw)
    if raw_content is not None and "corpus_raw_content" not in meta:
        meta["corpus_raw_content"] = raw_content
    return NormalizedMessage(role=msg.role, content=msg.content, metadata=meta)


# ---------------------------------------------------------------------------
# Corpus wire helpers
# ---------------------------------------------------------------------------


def to_corpus(messages: Iterable[NormalizedMessage]) -> List[Dict[str, Any]]:
    """
    Convert normalized messages to Corpus protocol wire format.

    Today, this is a simple `[{"role": str, "content": str}, ...]` list,
    but we keep the helper so that if/when Corpus adds richer content, the
    adaptation surface is centralized here.
    """
    return [{"role": m.role, "content": m.content} for m in messages]


def to_corpus_single(message: NormalizedMessage) -> Dict[str, Any]:
    """
    Public single-message variant of `to_corpus`, useful for validators
    and simple adapters.
    """
    return {"role": message.role, "content": message.content}


def from_corpus(payload: Mapping[str, Any]) -> NormalizedMessage:
    """
    Convert a single Corpus wire message into a `NormalizedMessage`.

    Useful for tests and for MCP / transport layers that already speak
    the wire format.
    """
    role, meta = _normalize_role(str(payload.get("role", "user")), {})
    content, raw_struct = _stringify_content(payload.get("content", ""))

    nm = NormalizedMessage(role=role, content=content, metadata=meta)
    return _with_raw_metadata(nm, payload, raw_struct)


def from_corpus_many(payloads: Iterable[Mapping[str, Any]]) -> List[NormalizedMessage]:
    """
    Convert an iterable of Corpus wire messages into a list of
    `NormalizedMessage` objects.
    """
    return [from_corpus(p) for p in payloads]


# ---------------------------------------------------------------------------
# LangChain
# ---------------------------------------------------------------------------


def from_langchain(msg: "BaseMessage") -> NormalizedMessage:
    """
    LangChain BaseMessage -> NormalizedMessage.

    Supports:
    * SystemMessage
    * HumanMessage
    * AIMessage
    * FunctionMessage / ToolMessage (newer LangChain versions)

    We avoid importing message classes at module import time so that this
    module doesn't hard-require LangChain.
    """
    raw_role = getattr(msg, "type", None) or getattr(msg, "role", None)
    raw_content = getattr(msg, "content", "")

    content, structured = _stringify_content(raw_content)

    metadata: Dict[str, Any] = {}
    additional = getattr(msg, "additional_kwargs", None)
    if isinstance(additional, Mapping):
        metadata.update(dict(additional))

    role, metadata = _normalize_role(raw_role, metadata)
    nm = NormalizedMessage(role=role, content=content, metadata=metadata)
    return _with_raw_metadata(
        nm,
        msg,
        structured if structured is not None else raw_content,
    )


def to_langchain(msg: NormalizedMessage) -> Any:
    """
    NormalizedMessage -> LangChain BaseMessage.

    We import lazily so LangChain remains an optional dependency.
    """
    try:
        from langchain_core.messages import (  # type: ignore
            SystemMessage,
            HumanMessage,
            AIMessage,
            FunctionMessage,
            ToolMessage,
        )
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "LangChain is not installed, cannot create LangChain messages"
        ) from exc

    role = msg.role
    meta = dict(msg.metadata)

    # If we preserved structured content, use it; otherwise fall back to str.
    raw_content = meta.pop("corpus_raw_content", None)
    meta.pop("corpus_raw", None)
    meta.pop("corpus_original_role", None)

    content = raw_content if raw_content is not None else msg.content

    # All remaining meta fields become additional_kwargs.
    additional_kwargs: Dict[str, Any] = meta

    if role == "system":
        cls = SystemMessage
    elif role == "user":
        cls = HumanMessage
    elif role == "assistant":
        cls = AIMessage
    elif role in {"function", "tool"}:
        # Prefer ToolMessage if available; otherwise FunctionMessage.
        cls = ToolMessage
    else:
        # Exotic roles fall back to HumanMessage, but original is in metadata.
        cls = HumanMessage

    return cls(content=content, additional_kwargs=additional_kwargs)


# ---------------------------------------------------------------------------
# LlamaIndex
# ---------------------------------------------------------------------------


def from_llamaindex(msg: "ChatMessage") -> NormalizedMessage:
    """
    LlamaIndex ChatMessage -> NormalizedMessage.

    Expected shape:
      - attribute `role` (enum or str)
      - attribute `content`
      - attribute `additional_kwargs` (optional)
    """
    raw_role = getattr(msg, "role", None)
    if hasattr(raw_role, "value"):
        raw_role = getattr(raw_role, "value")

    raw_content = getattr(msg, "content", "")
    content, structured = _stringify_content(raw_content)

    metadata: Dict[str, Any] = {}
    additional = getattr(msg, "additional_kwargs", None)
    if isinstance(additional, Mapping):
        metadata.update(dict(additional))

    role, metadata = _normalize_role(raw_role, metadata)
    nm = NormalizedMessage(role=role, content=content, metadata=metadata)
    return _with_raw_metadata(
        nm,
        msg,
        structured if structured is not None else raw_content,
    )


def to_llamaindex(msg: NormalizedMessage) -> Any:
    """
    NormalizedMessage -> LlamaIndex ChatMessage.
    """
    try:
        from llama_index.llms.base import ChatMessage, MessageRole  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "llama-index-core is not installed, cannot create ChatMessage"
        ) from exc

    role_map = {
        "system": MessageRole.SYSTEM,
        "user": MessageRole.USER,
        "assistant": MessageRole.ASSISTANT,
        "function": MessageRole.FUNCTION,
        "tool": getattr(MessageRole, "TOOL", MessageRole.FUNCTION),
    }
    role_enum = role_map.get(msg.role, MessageRole.USER)

    meta = dict(msg.metadata)
    raw_content = meta.pop("corpus_raw_content", None)
    meta.pop("corpus_raw", None)
    meta.pop("corpus_original_role", None)

    content = raw_content if raw_content is not None else msg.content

    return ChatMessage(role=role_enum, content=content, additional_kwargs=meta)


# ---------------------------------------------------------------------------
# Semantic Kernel
# ---------------------------------------------------------------------------


def from_semantic_kernel(msg: Mapping[str, Any]) -> NormalizedMessage:
    """
    Semantic Kernel minimal dict -> NormalizedMessage.

    We deliberately operate on dicts here since SK has multiple internal
    message representations; SK-side adapters are responsible for shaping
    to this minimal format first.
    """
    raw_role = msg.get("role", "user")
    raw_content = msg.get("content", "")
    content, structured = _stringify_content(raw_content)

    # Optional "metadata" bucket; everything else is left on corpus_raw.
    metadata = dict(msg.get("metadata", {}) or {})

    role, metadata = _normalize_role(raw_role, metadata)
    nm = NormalizedMessage(role=role, content=content, metadata=metadata)
    return _with_raw_metadata(
        nm,
        msg,
        structured if structured is not None else raw_content,
    )


def to_semantic_kernel(msg: NormalizedMessage) -> Dict[str, Any]:
    """
    NormalizedMessage -> Semantic Kernel dict representation.

    SK typically wraps this again into `ChatMessageContent` or similar types.
    """
    meta = dict(msg.metadata)
    raw_content = meta.pop("corpus_raw_content", None)

    # Preserve internal metadata under a namespaced key inside SK metadata.
    corpus_meta: Dict[str, Any] = {
        k: meta.pop(k) for k in tuple(meta) if k.startswith("corpus_")
    }

    if corpus_meta:
        meta.setdefault("corpus_metadata", corpus_meta)

    content = raw_content if raw_content is not None else msg.content

    return {
        "role": msg.role,
        "content": content,
        "metadata": meta,
    }


# ---------------------------------------------------------------------------
# AutoGen
# ---------------------------------------------------------------------------


def from_autogen(msg: Any) -> NormalizedMessage:
    """
    AutoGen message -> NormalizedMessage.

    AutoGen uses OpenAI-style message dicts in many places; we support a
    conservative subset:

    * Mapping with keys: role, content, name, tool_calls, function_call.
    * Objects with attributes: role, content, tool_calls, function_call.

    All non-standard fields are preserved in metadata.
    """
    if isinstance(msg, Mapping):
        raw_role = msg.get("role", "user")
        raw_content = msg.get("content", "")
        content, structured = _stringify_content(raw_content)

        metadata: Dict[str, Any] = {}
        for k, v in msg.items():
            if k in {"role", "content"}:
                continue
            metadata[k] = v
    else:
        raw_role = getattr(msg, "role", "user")
        raw_content = getattr(msg, "content", "")
        content, structured = _stringify_content(raw_content)

        metadata = {}
        for attr in ("name", "tool_calls", "function_call", "kwargs", "extras"):
            if hasattr(msg, attr):
                metadata[attr] = getattr(msg, attr)

    role, metadata = _normalize_role(raw_role, metadata)
    nm = NormalizedMessage(role=role, content=content, metadata=metadata)
    return _with_raw_metadata(
        nm,
        msg,
        structured if structured is not None else raw_content,
    )


def to_autogen(msg: NormalizedMessage) -> Dict[str, Any]:
    """
    NormalizedMessage -> AutoGen-style OpenAI dict.

    We intentionally return a plain dict so AutoGen can feed it directly
    into its own OpenAI client wrappers.
    """
    meta = dict(msg.metadata)
    raw_content = meta.pop("corpus_raw_content", None)

    corpus_meta: Dict[str, Any] = {
        k: meta.pop(k) for k in tuple(meta) if k.startswith("corpus_")
    }

    content = raw_content if raw_content is not None else msg.content

    base: Dict[str, Any] = {
        "role": msg.role,
        "content": content,
    }

    if meta or corpus_meta:
        # Reuse meta dict; only attach corpus_meta under a namespaced key.
        if corpus_meta:
            meta.setdefault("corpus_metadata", corpus_meta)
        base["metadata"] = meta

    return base


# ---------------------------------------------------------------------------
# CrewAI
# ---------------------------------------------------------------------------


def from_crewai(msg: Any) -> NormalizedMessage:
    """
    CrewAI message -> NormalizedMessage.

    CrewAI primarily passes around dicts / pydantic models that are
    structurally similar to OpenAI messages. We support:

    * Mapping with role/content
    * Objects with .role / .content attributes

    All additional keys/attributes are preserved in metadata.
    """
    if isinstance(msg, Mapping):
        raw_role = msg.get("role", "user")
        raw_content = msg.get("content", "")
        content, structured = _stringify_content(raw_content)
        metadata = {k: v for k, v in msg.items() if k not in {"role", "content"}}
    else:
        raw_role = getattr(msg, "role", "user")
        raw_content = getattr(msg, "content", "")
        content, structured = _stringify_content(raw_content)

        metadata = {}
        for attr in ("name", "agent", "tool_calls", "function_call", "kwargs"):
            if hasattr(msg, attr):
                metadata[attr] = getattr(msg, attr)

    role, metadata = _normalize_role(raw_role, metadata)
    nm = NormalizedMessage(role=role, content=content, metadata=metadata)
    return _with_raw_metadata(
        nm,
        msg,
        structured if structured is not None else raw_content,
    )


def to_crewai(msg: NormalizedMessage) -> Dict[str, Any]:
    """
    NormalizedMessage -> CrewAI-compatible message dict.

    We emit a conservative OpenAI-like shape. CrewAI adapters can wrap
    this into richer internal types if needed.
    """
    meta = dict(msg.metadata)
    raw_content = meta.pop("corpus_raw_content", None)

    corpus_meta: Dict[str, Any] = {
        k: meta.pop(k) for k in tuple(meta) if k.startswith("corpus_")
    }

    content = raw_content if raw_content is not None else msg.content

    result: Dict[str, Any] = {
        "role": msg.role,
        "content": content,
    }

    if meta or corpus_meta:
        if corpus_meta:
            meta.setdefault("corpus_metadata", corpus_meta)
        result["metadata"] = meta

    return result


# ---------------------------------------------------------------------------
# Test / Validation helpers
# ---------------------------------------------------------------------------

_FRAMEWORK_TRANSLATORS: Dict[
    str,
    Tuple[Callable[[Any], NormalizedMessage], Callable[[NormalizedMessage], Any]],
] = {
    "corpus": (from_corpus, to_corpus_single),
    "langchain": (from_langchain, to_langchain),
    "llamaindex": (from_llamaindex, to_llamaindex),
    "semantic_kernel": (from_semantic_kernel, to_semantic_kernel),
    "autogen": (from_autogen, to_autogen),
    "crewai": (from_crewai, to_crewai),
}


def _messages_equivalent(original: Any, roundtripped: Any, framework: str) -> bool:
    """
    Lightweight equality check for round-trip validation.

    This is intentionally conservative: it checks only the core fields
    that matter for protocol correctness (role, content). Frameworks
    can add stricter tests on top if needed.
    """
    # Mapping-based messages (Corpus, AutoGen, CrewAI, SK)
    if isinstance(original, Mapping) and isinstance(roundtripped, Mapping):
        orig_role = original.get("role")
        rt_role = roundtripped.get("role")
        orig_content = original.get("content")
        rt_content = roundtripped.get("content")
        return orig_role == rt_role and orig_content == rt_content

    # Generic objects with role/type and content attributes (LangChain, LlamaIndex)
    def _extract(obj: Any) -> Tuple[Optional[str], Optional[Any]]:
        if obj is None:
            return None, None

        # Some frameworks use `type` instead of `role`
        role = getattr(obj, "role", None)
        if hasattr(role, "value"):
            role = getattr(role, "value")
        if role is None:
            role = getattr(obj, "type", None)

        content = getattr(obj, "content", None)
        return role, content

    orig_role, orig_content = _extract(original)
    rt_role, rt_content = _extract(roundtripped)

    # For frameworks that remap roles (e.g., Human -> user), we only
    # require equality in the framework-facing role surface.
    return orig_role == rt_role and orig_content == rt_content


def validate_message_roundtrip(framework: str, test_cases: List[Any]) -> bool:
    """
    Validate that a framework's messages survive a normalize+denormalize
    round-trip without losing core semantics.

    Parameters
    ----------
    framework:
        One of:
            "corpus", "langchain", "llamaindex",
            "semantic_kernel", "autogen", "crewai"

    test_cases:
        List of framework-native messages (BaseMessage, dict, etc.)

    Returns
    -------
    bool
        True if all messages pass role+content equivalence checks.

    Notes
    -----
    * This is primarily intended for tests (e.g., in `tests/utils`).
    * It is deliberately conservative and focuses on:
        - role / type
        - content
      Framework-specific metadata is expected to change (we add
      `corpus_metadata`, `corpus_raw`, etc.).
    """
    if framework not in _FRAMEWORK_TRANSLATORS:
        raise ValueError(
            f"Unknown framework '{framework}'. "
            f"Expected one of: {sorted(_FRAMEWORK_TRANSLATORS.keys())}"
        )

    from_fn, to_fn = _FRAMEWORK_TRANSLATORS[framework]

    for original in test_cases:
        normalized = from_fn(original)
        roundtripped = to_fn(normalized)
        if not _messages_equivalent(original, roundtripped, framework):
            return False

    return True
