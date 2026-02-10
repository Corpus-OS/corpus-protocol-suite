# corpus_sdk/llm/framework_adapters/common/message_translation.py
# SPDX-License-Identifier: Apache-2.0

"""
Message translation for Corpus chat framework adapters.

This module normalizes *chat-style* message objects from various upstream
frameworks into Corpus protocol messages so that:

- Role names are canonicalized into the Corpus chat role set
  ("system", "user", "assistant", "tool", "function")
- Content is stringified while preserving structured data
- Framework-specific metadata is captured without loss
- Round-trip translation is possible when going back to the source framework

Scope
-----
- This module is **LLM/chat-specific**: it is meant for chat messages only.
- It is **framework-agnostic**: implementations for LangChain, LlamaIndex,
  Semantic Kernel, AutoGen, CrewAI, MCP, etc. are optional adapters on top of a
  protocol-first core.
- Non-chat products (vector, graph, embedding) should use their own protocol
  types rather than `NormalizedMessage`.

Design goals
------------
- Protocol-first:
    The Corpus chat wire format ({role, content}) is the source of truth.
- Non-destructive:
    Preserve original framework objects and structured content in metadata.
- Framework-agnostic:
    No hard runtime dependency on any single framework; imports live inside
    adapter functions and are optional.
- Configurable:
    Strict vs. lenient role validation, configurable fallback role.
    Configuration is stored in context-local variables to remain safe in
    multi-threaded or async applications.

Primary entry points
--------------------
Core / protocol:
- NormalizedMessage
- to_corpus: [NormalizedMessage] → [{"role": str, "content": str}]
- from_corpus: {"role": str, "content": Any} → NormalizedMessage
- from_corpus_many
- from_generic_dict / to_generic_dict

Framework adapters:
- from_langchain / to_langchain
- from_llamaindex / to_llamaindex
- from_semantic_kernel / to_semantic_kernel
- from_autogen / to_autogen
- from_crewai / to_crewai
- from_mcp / to_mcp

Notes
-----
- MCP JSON-RPC envelopes (e.g. {"id", "method", "params", ...}) are handled at
  higher layers (e.g. context translation). This module only deals with
  *chat-style* message objects that have a "role" and "content".
"""

from __future__ import annotations

import json
import os
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from langchain_core.messages import BaseMessage  # type: ignore
    from llama_index.core.llms import ChatMessage  # type: ignore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env_flag(name: str, default: str = "0") -> bool:
    """
    Parse a boolean-ish environment variable in a consistent, case-insensitive way.

    Truthy values: "1", "true", "yes", "on" (any case, with surrounding whitespace allowed).
    Everything else is treated as False.
    """
    val = os.getenv(name, default)
    if not isinstance(val, str):
        return False
    return val.strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Configuration (context-local for thread/async safety)
# ---------------------------------------------------------------------------

_STRICT_ROLE_VALIDATION_DEFAULT: bool = _env_flag("CORPUS_STRICT_ROLES", "0")

_UNKNOWN_ROLE_FALLBACK_DEFAULT: str = os.getenv(
    "CORPUS_UNKNOWN_ROLE",
    "assistant",
).strip().lower()
if _UNKNOWN_ROLE_FALLBACK_DEFAULT not in {"system", "user", "assistant", "tool", "function"}:
    _UNKNOWN_ROLE_FALLBACK_DEFAULT = "assistant"

#: Context-local strict role validation flag.
STRICT_ROLE_VALIDATION_VAR: ContextVar[bool] = ContextVar(
    "corpus_strict_role_validation",
    default=_STRICT_ROLE_VALIDATION_DEFAULT,
)

#: Context-local unknown role fallback.
_UNKNOWN_ROLE_FALLBACK_VAR: ContextVar[str] = ContextVar(
    "corpus_unknown_role_fallback",
    default=_UNKNOWN_ROLE_FALLBACK_DEFAULT,
)


def set_strict_role_validation(enabled: bool) -> None:
    """
    Enable or disable strict role validation for the current context.

    This uses a ContextVar, so changes are local to the current task / thread
    (and any children that inherit the context), rather than a process-wide
    global. This avoids cross-request interference in multi-tenant servers.
    """
    STRICT_ROLE_VALIDATION_VAR.set(bool(enabled))


def set_unknown_role_fallback(fallback: str) -> None:
    """
    Configure the fallback canonical role for unknown roles for the current context.

    Allowed values: "system", "user", "assistant", "tool", "function".

    This uses a ContextVar, so changes are local to the current task / thread.
    """
    value = str(fallback).strip().lower()
    if value not in {"system", "user", "assistant", "tool", "function"}:
        raise ValueError(
            "Unknown role fallback must be one of "
            '{"system", "user", "assistant", "tool", "function"}, '
            f"got {fallback!r}"
        )
    _UNKNOWN_ROLE_FALLBACK_VAR.set(value)


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------


@dataclass
class NormalizedMessage:
    """
    Protocol-centric representation of a single chat message.

    This is the "narrow waist" across frameworks: all upstream chat messages
    (LangChain, LlamaIndex, SK, AutoGen, CrewAI, MCP, or custom) should be
    mapped into this structure before being sent over the Corpus chat protocol.

    Attributes
    ----------
    role:
        Canonical Corpus chat role string:
            "system", "user", "assistant", "tool", "function".
        Non-standard roles from upstream frameworks are remapped into this
        set, with the original value preserved in metadata["corpus_original_role"].

    content:
        String form of the message body, suitable for wire transport.
        Structured or rich content (dict, list-of-parts, etc.) is stringified
        and the original preserved in metadata["corpus_raw_content"].

    metadata:
        Arbitrary metadata from upstream frameworks or callers.

        Reserved Corpus keys:
            - "corpus_raw": the original framework message object or payload
            - "corpus_raw_content": original structured content before stringifying
            - "corpus_original_role": non-canonical role when remapped

        All other keys are considered integration-specific and are preserved
        for round-tripping or debugging.
    """

    role: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def copy_with_content(self, content: str) -> "NormalizedMessage":
        """Create a shallow copy with new content."""
        return NormalizedMessage(
            role=self.role,
            content=content,
            metadata=dict(self.metadata),
        )

    @property
    def raw_content(self) -> Optional[Any]:
        """Original structured content, if preserved."""
        return self.metadata.get("corpus_raw_content")

    @property
    def original_role(self) -> Optional[str]:
        """Original framework role before normalization, if remapped."""
        val = self.metadata.get("corpus_original_role")
        return str(val) if val is not None else None


# ---------------------------------------------------------------------------
# Role normalization
# ---------------------------------------------------------------------------

# Corpus chat protocol canonical roles:
#   "system", "user", "assistant", "tool", "function"
#
# All framework-specific / upstream role labels are mapped into this set.
_ROLE_MAP: Dict[str, str] = {
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


def _normalize_role(raw_role: Optional[str]) -> Tuple[str, Optional[str]]:
    """
    Map upstream/framework role to canonical Corpus chat role.

    Returns:
        (canonical_role, original_role_if_changed)

    May raise:
        ValueError: if strict role validation is enabled for the current context
        and the role is not recognized.
    """
    if raw_role is None:
        return "user", None

    original = str(raw_role)
    key = original.strip().lower()

    if key in _ROLE_MAP:
        canonical = _ROLE_MAP[key]
        return canonical, (original if canonical != original else None)

    # Unknown role
    if STRICT_ROLE_VALIDATION_VAR.get():
        raise ValueError(f"Unknown message role: {original!r}")

    # Lenient fallback (context-local)
    fallback = _UNKNOWN_ROLE_FALLBACK_VAR.get()
    if fallback not in {"system", "user", "assistant", "tool", "function"}:
        fallback = "assistant"
    return fallback, original


# ---------------------------------------------------------------------------
# Content stringification
# ---------------------------------------------------------------------------


def _stringify_content(content: Any) -> Tuple[str, Optional[Any]]:
    """
    Convert arbitrary content into a string while preserving structure.

    Returns:
        (string_form, raw_structured_or_None)

    Strategy:
    - str → pass through
    - Mapping → JSON + preserve original
    - list[Mapping] → JSON + preserve original (OpenAI-style parts)
    - other iterables → str(list(...)) + preserve original list
    - fallthrough → str() conversion

    Note:
    - json.dumps uses default=str so that datetime and other non-serializable
      objects are safely represented without raising TypeError.
    """
    if isinstance(content, str):
        return content, None

    # Mapping → JSON
    if isinstance(content, Mapping):
        try:
            return (
                json.dumps(
                    content,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                ),
                content,
            )
        except (TypeError, RecursionError):
            return repr(content), content

    # Try iterable (but not string)
    try:
        as_list = list(content)  # type: ignore[arg-type]
    except (TypeError, RecursionError):
        return str(content), None

    # OpenAI-style content parts: [{"type": "...", ...}, ...]
    if as_list and all(isinstance(p, Mapping) for p in as_list):
        try:
            return (
                json.dumps(
                    as_list,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                ),
                as_list,
            )
        except (TypeError, RecursionError):
            return repr(as_list), as_list

    # Other iterables
    return str(as_list), as_list


# ---------------------------------------------------------------------------
# NormalizedMessage construction helper
# ---------------------------------------------------------------------------


def _build_normalized_message(
    raw_role: Any,
    raw_content: Any,
    base_metadata: Optional[Mapping[str, Any]] = None,
) -> NormalizedMessage:
    """
    Shared helper to construct a NormalizedMessage from raw role/content.

    - Normalizes the role using `_normalize_role`
    - Stringifies content using `_stringify_content`
    - Adds `corpus_original_role` and `corpus_raw_content` when applicable
    - Starts from an optional base metadata mapping
    """
    role, original_role = _normalize_role(raw_role)
    content, structured = _stringify_content(raw_content)

    metadata: Dict[str, Any] = dict(base_metadata or {})

    if original_role is not None:
        metadata["corpus_original_role"] = original_role
    if structured is not None:
        metadata["corpus_raw_content"] = structured

    return NormalizedMessage(role=role, content=content, metadata=metadata)


# ---------------------------------------------------------------------------
# Corpus wire format
# ---------------------------------------------------------------------------


def to_corpus(messages: Iterable[NormalizedMessage]) -> List[Dict[str, Any]]:
    """
    Convert normalized messages to Corpus chat protocol wire format.

    Returns:
        [{"role": str, "content": str}, ...]
    """
    return [{"role": m.role, "content": m.content} for m in messages]


def from_corpus(payload: Mapping[str, Any]) -> NormalizedMessage:
    """
    Convert a Corpus wire message to NormalizedMessage.

    Args:
        payload: {"role": str, "content": Any, ...}

    Returns:
        NormalizedMessage

    May raise:
        ValueError: if strict role validation is enabled and the role is unknown.
    """
    base_metadata: Dict[str, Any] = {"corpus_raw": payload}

    return _build_normalized_message(
        raw_role=payload.get("role", "user"),
        raw_content=payload.get("content", ""),
        base_metadata=base_metadata,
    )


def from_corpus_many(payloads: Iterable[Mapping[str, Any]]) -> List[NormalizedMessage]:
    """Convert multiple Corpus wire messages to a list of NormalizedMessage."""
    return [from_corpus(p) for p in payloads]


# ---------------------------------------------------------------------------
# Generic dict helpers (framework-agnostic)
# ---------------------------------------------------------------------------


def from_generic_dict(msg: Mapping[str, Any]) -> NormalizedMessage:
    """
    Generic dict → NormalizedMessage.

    This is a framework-agnostic adapter for callers that already operate on
    simple dicts and don't want to depend on a specific framework's types.

    Expected (but not strictly required) shape:
        {
            "role": str,
            "content": Any,
            "metadata": { ... }  # optional
            ... other keys ...
        }

    Behavior:
    - Uses `msg["role"]` and `msg["content"]` if present, with defaults.
    - Copies any `msg["metadata"]` (if a mapping) into `metadata`.
    - Preserves the original dict as metadata["corpus_raw"].

    May raise:
        ValueError: if strict role validation is enabled and the role is unknown.
    """
    raw_role = msg.get("role", "user")
    raw_content = msg.get("content", "")

    metadata: Dict[str, Any] = {"corpus_raw": msg}

    # Merge user metadata if present
    user_meta = msg.get("metadata")
    if isinstance(user_meta, Mapping):
        metadata.update(dict(user_meta))

    return _build_normalized_message(
        raw_role=raw_role,
        raw_content=raw_content,
        base_metadata=metadata,
    )


def to_generic_dict(msg: NormalizedMessage) -> Dict[str, Any]:
    """
    NormalizedMessage → generic dict.

    Returns:
        {
            "role": str,
            "content": Any,
            "metadata": { ... non-corpus metadata ... }
        }

    This is useful for custom or internal frameworks that want a neutral,
    non-framework-branded representation of a chat message.
    """
    # Reconstruct content from structured form if available
    content = msg.raw_content if msg.raw_content is not None else msg.content

    # Build metadata (exclude reserved corpus_* keys)
    metadata: Dict[str, Any] = {
        k: v
        for k, v in msg.metadata.items()
        if not k.startswith("corpus_")
    }

    return {
        "role": msg.role,
        "content": content,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# LangChain
# ---------------------------------------------------------------------------


def from_langchain(msg: "BaseMessage") -> NormalizedMessage:
    """
    LangChain BaseMessage → NormalizedMessage.

    Supports typical LangChain message types:
    - SystemMessage, HumanMessage, AIMessage
    - FunctionMessage, ToolMessage

    Args:
        msg: LangChain BaseMessage

    Returns:
        NormalizedMessage

    May raise:
        ValueError: if strict role validation is enabled and the role is unknown.
    """
    raw_role = getattr(msg, "type", None) or getattr(msg, "role", None)
    raw_content = getattr(msg, "content", "")

    metadata: Dict[str, Any] = {"corpus_raw": msg}

    # Preserve additional_kwargs
    additional = getattr(msg, "additional_kwargs", None)
    if isinstance(additional, Mapping):
        metadata.update(dict(additional))

    return _build_normalized_message(
        raw_role=raw_role,
        raw_content=raw_content,
        base_metadata=metadata,
    )


def to_langchain(msg: NormalizedMessage) -> Any:
    """
    NormalizedMessage → LangChain BaseMessage.

    Raises:
        RuntimeError: If LangChain is not installed.
    """
    try:
        from langchain_core.messages import (  # type: ignore
            SystemMessage,
            HumanMessage,
            AIMessage,
            ToolMessage,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "LangChain is not installed, cannot create LangChain messages"
        ) from exc

    # Reconstruct content
    content = msg.raw_content if msg.raw_content is not None else msg.content

    # Build additional_kwargs from metadata (excluding reserved corpus_* keys)
    additional_kwargs: Dict[str, Any] = {
        k: v
        for k, v in msg.metadata.items()
        if not k.startswith("corpus_")
    }

    # Select message class based on canonical role
    if msg.role == "system":
        return SystemMessage(content=content, additional_kwargs=additional_kwargs)
    if msg.role == "user":
        return HumanMessage(content=content, additional_kwargs=additional_kwargs)
    if msg.role == "assistant":
        return AIMessage(content=content, additional_kwargs=additional_kwargs)
    if msg.role in {"function", "tool"}:
        # LangChain's ToolMessage in many versions expects a `tool_call_id`.
        # We try to recover it from metadata if present; callers that need
        # strict round-tripping for tool messages should ensure they store
        # `tool_call_id` (or compatible) in message.metadata.
        tool_call_id = (
            additional_kwargs.get("tool_call_id")
            or additional_kwargs.get("id")
            or additional_kwargs.get("call_id")
        )

        return ToolMessage(
            content=content,
            tool_call_id=tool_call_id,
            additional_kwargs=additional_kwargs,
        )

    # Fallback for unknown roles: treat as user
    return HumanMessage(content=content, additional_kwargs=additional_kwargs)


# ---------------------------------------------------------------------------
# LlamaIndex
# ---------------------------------------------------------------------------


def from_llamaindex(msg: "ChatMessage") -> NormalizedMessage:
    """
    LlamaIndex ChatMessage → NormalizedMessage.

    Args:
        msg: LlamaIndex ChatMessage

    Returns:
        NormalizedMessage

    May raise:
        ValueError: if strict role validation is enabled and the role is unknown.
    """
    raw_role = getattr(msg, "role", None)
    if hasattr(raw_role, "value"):
        raw_role = getattr(raw_role, "value")

    raw_content = getattr(msg, "content", "")

    metadata: Dict[str, Any] = {"corpus_raw": msg}

    # Preserve additional_kwargs
    additional = getattr(msg, "additional_kwargs", None)
    if isinstance(additional, Mapping):
        metadata.update(dict(additional))

    return _build_normalized_message(
        raw_role=raw_role,
        raw_content=raw_content,
        base_metadata=metadata,
    )


def to_llamaindex(msg: NormalizedMessage) -> Any:
    """
    NormalizedMessage → LlamaIndex ChatMessage.

    Raises:
        RuntimeError: If llama-index is not installed.
    """
    try:
        from llama_index.core.llms import ChatMessage, MessageRole  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "llama-index is not installed, cannot create ChatMessage"
        ) from exc

    # Map canonical role to MessageRole enum
    role_map = {
        "system": MessageRole.SYSTEM,
        "user": MessageRole.USER,
        "assistant": MessageRole.ASSISTANT,
        "function": MessageRole.FUNCTION,
        "tool": getattr(MessageRole, "TOOL", MessageRole.FUNCTION),
    }
    role_enum = role_map.get(msg.role, MessageRole.USER)

    # Reconstruct content
    content = msg.raw_content if msg.raw_content is not None else msg.content

    # Build additional_kwargs (excluding reserved corpus_* keys)
    additional_kwargs = {
        k: v
        for k, v in msg.metadata.items()
        if not k.startswith("corpus_")
    }

    return ChatMessage(
        role=role_enum,
        content=content,
        additional_kwargs=additional_kwargs,
    )


# ---------------------------------------------------------------------------
# Semantic Kernel
# ---------------------------------------------------------------------------


def from_semantic_kernel(msg: Mapping[str, Any]) -> NormalizedMessage:
    """
    Semantic Kernel message dict → NormalizedMessage.

    Expected shape:
        {"role": str, "content": Any, "metadata": {...}}

    Args:
        msg: SK message dict

    Returns:
        NormalizedMessage

    May raise:
        ValueError: if strict role validation is enabled and the role is unknown.
    """
    raw_role = msg.get("role", "user")
    raw_content = msg.get("content", "")

    metadata: Dict[str, Any] = {"corpus_raw": msg}

    # Preserve SK metadata
    sk_metadata = msg.get("metadata", {})
    if isinstance(sk_metadata, Mapping):
        metadata.update(dict(sk_metadata))

    return _build_normalized_message(
        raw_role=raw_role,
        raw_content=raw_content,
        base_metadata=metadata,
    )


def to_semantic_kernel(msg: NormalizedMessage) -> Dict[str, Any]:
    """
    NormalizedMessage → Semantic Kernel message dict.

    Returns:
        {"role": str, "content": Any, "metadata": {...}}
    """
    # Reconstruct content
    content = msg.raw_content if msg.raw_content is not None else msg.content

    # Build metadata (exclude corpus_* keys)
    sk_metadata = {
        k: v
        for k, v in msg.metadata.items()
        if not k.startswith("corpus_")
    }

    return {
        "role": msg.role,
        "content": content,
        "metadata": sk_metadata,
    }


# ---------------------------------------------------------------------------
# AutoGen
# ---------------------------------------------------------------------------


def from_autogen(msg: Mapping[str, Any] | object) -> NormalizedMessage:
    """
    AutoGen message → NormalizedMessage.

    Supports:
    - Mapping with keys: role, content, name, tool_calls, function_call
    - Objects with attributes: role, content, ...

    Args:
        msg: AutoGen message (dict or object)

    Returns:
        NormalizedMessage

    May raise:
        ValueError: if strict role validation is enabled and the role is unknown.
    """
    if isinstance(msg, Mapping):
        raw_role = msg.get("role", "user")
        raw_content = msg.get("content", "")

        metadata: Dict[str, Any] = {"corpus_raw": msg}
        for k, v in msg.items():
            if k not in {"role", "content"}:
                metadata[k] = v

    else:
        raw_role = getattr(msg, "role", "user")
        raw_content = getattr(msg, "content", "")

        metadata = {"corpus_raw": msg}
        for attr in ("name", "tool_calls", "function_call"):
            if hasattr(msg, attr):
                metadata[attr] = getattr(msg, attr)

    return _build_normalized_message(
        raw_role=raw_role,
        raw_content=raw_content,
        base_metadata=metadata,
    )


def to_autogen(msg: NormalizedMessage) -> Dict[str, Any]:
    """
    NormalizedMessage → AutoGen-style dict.

    Returns:
        {"role": str, "content": Any, ...}
    """
    # Reconstruct content
    content = msg.raw_content if msg.raw_content is not None else msg.content

    # Build result dict
    result: Dict[str, Any] = {
        "role": msg.role,
        "content": content,
    }

    # Add non-corpus metadata
    for k, v in msg.metadata.items():
        if not k.startswith("corpus_"):
            result[k] = v

    return result


# ---------------------------------------------------------------------------
# CrewAI
# ---------------------------------------------------------------------------


def from_crewai(msg: Mapping[str, Any] | object) -> NormalizedMessage:
    """
    CrewAI message → NormalizedMessage.

    Supports:
    - Mapping with role/content
    - Objects with .role / .content attributes

    Args:
        msg: CrewAI message (dict or object)

    Returns:
        NormalizedMessage

    May raise:
        ValueError: if strict role validation is enabled and the role is unknown.
    """
    if isinstance(msg, Mapping):
        raw_role = msg.get("role", "user")
        raw_content = msg.get("content", "")

        metadata: Dict[str, Any] = {"corpus_raw": msg}
        for k, v in msg.items():
            if k not in {"role", "content"}:
                metadata[k] = v
    else:
        raw_role = getattr(msg, "role", "user")
        raw_content = getattr(msg, "content", "")

        metadata = {"corpus_raw": msg}
        for attr in ("name", "agent", "tool_calls", "function_call"):
            if hasattr(msg, attr):
                metadata[attr] = getattr(msg, attr)

    return _build_normalized_message(
        raw_role=raw_role,
        raw_content=raw_content,
        base_metadata=metadata,
    )


def to_crewai(msg: NormalizedMessage) -> Dict[str, Any]:
    """
    NormalizedMessage → CrewAI-compatible dict.

    Returns:
        {"role": str, "content": Any, ...}
    """
    # Reconstruct content
    content = msg.raw_content if msg.raw_content is not None else msg.content

    # Build result dict
    result: Dict[str, Any] = {
        "role": msg.role,
        "content": content,
    }

    # Add non-corpus metadata
    for k, v in msg.metadata.items():
        if not k.startswith("corpus_"):
            result[k] = v

    return result


# ---------------------------------------------------------------------------
# MCP
# ---------------------------------------------------------------------------


def from_mcp(msg: Mapping[str, Any] | object) -> NormalizedMessage:
    """
    MCP message → NormalizedMessage.

    Supported shapes
    ----------------
    - Mapping-based messages (recommended):
        {
            "role": str,          # required for correct role mapping
            "content": Any,       # required; may be string or structured
            "metadata": Mapping,  # optional, merged into metadata
            ... other keys ...    # preserved as top-level metadata fields
        }

    - Object-based messages:
        Objects exposing `.role` and `.content` attributes, plus optional
        attributes such as `metadata`, `tool_calls`, `function_call`,
        `id`, `conversation_id`, `created_at`, etc.

    Behavior
    --------
    - Uses `_normalize_role` for role handling (strict vs lenient).
    - Uses `_stringify_content` for content, preserving structured values in
      metadata["corpus_raw_content"].
    - Stores the original message in metadata["corpus_raw"].
    - Preserves non-reserved keys as metadata entries so they can round-trip.
    - If a `metadata` field is present but is **not** a Mapping, it is kept
      as-is and additionally stored as `metadata["mcp_metadata_raw"]` to avoid
      losing shape information.

    May raise:
        ValueError: if strict role validation is enabled and the role is unknown.
    """
    if isinstance(msg, Mapping):
        raw_role = msg.get("role", "user")
        raw_content = msg.get("content", "")

        metadata: Dict[str, Any] = {"corpus_raw": msg}
        # Preserve all other top-level keys besides role/content
        for k, v in msg.items():
            if k in {"role", "content"}:
                continue
            if k == "metadata":
                # Merge mapping-style metadata, preserve non-mapping as raw
                if isinstance(v, Mapping):
                    metadata.update(dict(v))
                else:
                    metadata["metadata"] = v
                    metadata["mcp_metadata_raw"] = v
            else:
                metadata[k] = v
    else:
        raw_role = getattr(msg, "role", "user")
        raw_content = getattr(msg, "content", "")

        metadata = {"corpus_raw": msg}
        # Preserve common MCP-style attributes when present
        for attr in (
            "name",
            "metadata",
            "tool_calls",
            "function_call",
            "id",
            "conversation_id",
            "created_at",
        ):
            if not hasattr(msg, attr):
                continue
            value = getattr(msg, attr)
            if attr == "metadata":
                # Merge mapping-style metadata, preserve non-mapping as raw
                if isinstance(value, Mapping):
                    metadata.update(dict(value))
                else:
                    metadata["metadata"] = value
                    metadata["mcp_metadata_raw"] = value
            else:
                metadata[attr] = value

    return _build_normalized_message(
        raw_role=raw_role,
        raw_content=raw_content,
        base_metadata=metadata,
    )


def to_mcp(msg: NormalizedMessage) -> Dict[str, Any]:
    """
    NormalizedMessage → MCP-style dict.

    Returns:
        {
            "role": str,
            "content": Any,
            ... non-corpus metadata flattened at top level ...
        }

    Notes
    -----
    - Content reconstruction prefers `msg.raw_content` (i.e. the preserved
      structured form from `corpus_raw_content`) when available.
    - All non-`corpus_*` metadata keys are emitted as top-level fields so that
      keys like `id`, `conversation_id`, `tool_calls`, etc. round-trip cleanly.
    - This function does not attempt to reconstruct a full JSON-RPC envelope;
      it only returns a chat-style message object suitable for MCP tooling.
    """
    # Reconstruct content
    content = msg.raw_content if msg.raw_content is not None else msg.content

    result: Dict[str, Any] = {
        "role": msg.role,
        "content": content,
    }

    # Add non-corpus metadata as top-level fields for maximum compatibility
    for k, v in msg.metadata.items():
        if not k.startswith("corpus_"):
            result[k] = v

    return result


__all__ = [
    # Core type
    "NormalizedMessage",
    # Configuration
    "set_strict_role_validation",
    "set_unknown_role_fallback",
    # Corpus wire format
    "to_corpus",
    "from_corpus",
    "from_corpus_many",
    # Generic dict helpers
    "from_generic_dict",
    "to_generic_dict",
    # Framework adapters
    "from_langchain",
    "to_langchain",
    "from_llamaindex",
    "to_llamaindex",
    "from_semantic_kernel",
    "to_semantic_kernel",
    "from_autogen",
    "to_autogen",
    "from_crewai",
    "to_crewai",
    "from_mcp",
    "to_mcp",
]
