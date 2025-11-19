# corpus_sdk/llm/framework_adapters/common/message_translation.py
# SPDX-License-Identifier: Apache-2.0

"""
Message translation for Corpus framework adapters.

This module normalizes framework-specific message objects into Corpus
protocol messages so that:

- Role names are canonicalized (e.g., "human" → "user")
- Content is stringified while preserving structured data
- Metadata is captured without loss
- Round-trip translation is possible

Design goals
------------
- Protocol-first: wire format is the source of truth
- Non-destructive: preserve original objects in metadata
- Framework-agnostic: no hard runtime dependencies
- Configurable: strict vs. lenient role validation

Primary entry points
--------------------
For each framework:
- from_<framework>: framework message → NormalizedMessage
- to_<framework>: NormalizedMessage → framework message

Wire format:
- to_corpus: [NormalizedMessage] → [{"role": str, "content": str}]
- from_corpus: {"role": str, "content": str} → NormalizedMessage
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from langchain_core.messages import BaseMessage  # type: ignore
    from llama_index.core.llms import ChatMessage  # type: ignore

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STRICT_ROLE_VALIDATION: bool = os.getenv("CORPUS_STRICT_ROLES", "0") in {
    "1", "true", "TRUE"
}

_UNKNOWN_ROLE_FALLBACK: str = os.getenv(
    "CORPUS_UNKNOWN_ROLE", "assistant"
).strip().lower()


def set_strict_role_validation(enabled: bool) -> None:
    """Enable or disable strict role validation at runtime."""
    global STRICT_ROLE_VALIDATION
    STRICT_ROLE_VALIDATION = bool(enabled)


def set_unknown_role_fallback(fallback: str) -> None:
    """
    Configure the fallback canonical role for unknown roles.

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


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------


@dataclass
class NormalizedMessage:
    """
    Protocol-centric representation of a single chat message.

    Attributes
    ----------
    role:
        Canonical role string: "system", "user", "assistant", "tool", "function".
        Non-standard roles are remapped and original preserved in metadata.

    content:
        String form of the message. Structured content is stringified and
        preserved in metadata["corpus_raw_content"].

    metadata:
        Arbitrary metadata. Reserved keys:
        - "corpus_raw": original framework message object
        - "corpus_raw_content": original structured content
        - "corpus_original_role": non-canonical role when remapped
    """

    role: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def copy_with_content(self, content: str) -> "NormalizedMessage":
        """Create a copy with new content."""
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
        """Original framework role before normalization."""
        val = self.metadata.get("corpus_original_role")
        return str(val) if val is not None else None


# ---------------------------------------------------------------------------
# Role normalization
# ---------------------------------------------------------------------------

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
    Map framework role to canonical protocol role.

    Returns:
        (canonical_role, original_role_if_changed)
    """
    if raw_role is None:
        return "user", None

    original = str(raw_role)
    key = original.strip().lower()

    if key in _ROLE_MAP:
        canonical = _ROLE_MAP[key]
        return canonical, (original if canonical != original else None)

    # Unknown role
    if STRICT_ROLE_VALIDATION:
        raise ValueError(f"Unknown message role: {original!r}")

    # Fallback
    fallback = _UNKNOWN_ROLE_FALLBACK
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
    - other → str() conversion
    """
    if isinstance(content, str):
        return content, None

    # Mapping → JSON
    if isinstance(content, Mapping):
        try:
            return json.dumps(content, ensure_ascii=False, sort_keys=True), content
        except (TypeError, RecursionError):
            return repr(content), content

    # Try iterable (but not string)
    try:
        as_list = list(content)  # type: ignore
    except (TypeError, RecursionError):
        return str(content), None

    # OpenAI-style content parts: [{"type": "...", ...}, ...]
    if as_list and all(isinstance(p, Mapping) for p in as_list):
        try:
            return json.dumps(as_list, ensure_ascii=False, sort_keys=True), as_list
        except (TypeError, RecursionError):
            return repr(as_list), as_list

    # Other iterables
    return str(as_list), as_list


# ---------------------------------------------------------------------------
# Corpus wire format
# ---------------------------------------------------------------------------


def to_corpus(messages: Iterable[NormalizedMessage]) -> List[Dict[str, Any]]:
    """
    Convert normalized messages to Corpus protocol wire format.

    Returns:
        [{"role": str, "content": str}, ...]
    """
    return [{"role": m.role, "content": m.content} for m in messages]


def from_corpus(payload: Mapping[str, Any]) -> NormalizedMessage:
    """
    Convert a Corpus wire message to NormalizedMessage.

    Args:
        payload: {"role": str, "content": str, ...}

    Returns:
        NormalizedMessage
    """
    role, original_role = _normalize_role(str(payload.get("role", "user")))
    content, structured = _stringify_content(payload.get("content", ""))

    metadata: Dict[str, Any] = {"corpus_raw": payload}
    if original_role is not None:
        metadata["corpus_original_role"] = original_role
    if structured is not None:
        metadata["corpus_raw_content"] = structured

    return NormalizedMessage(role=role, content=content, metadata=metadata)


def from_corpus_many(payloads: Iterable[Mapping[str, Any]]) -> List[NormalizedMessage]:
    """Convert multiple Corpus wire messages to NormalizedMessage list."""
    return [from_corpus(p) for p in payloads]


# ---------------------------------------------------------------------------
# LangChain
# ---------------------------------------------------------------------------


def from_langchain(msg: "BaseMessage") -> NormalizedMessage:
    """
    LangChain BaseMessage → NormalizedMessage.

    Supports:
    - SystemMessage, HumanMessage, AIMessage
    - FunctionMessage, ToolMessage

    Args:
        msg: LangChain BaseMessage

    Returns:
        NormalizedMessage
    """
    raw_role = getattr(msg, "type", None) or getattr(msg, "role", None)
    raw_content = getattr(msg, "content", "")

    role, original_role = _normalize_role(raw_role)
    content, structured = _stringify_content(raw_content)

    metadata: Dict[str, Any] = {"corpus_raw": msg}
    if original_role is not None:
        metadata["corpus_original_role"] = original_role
    if structured is not None:
        metadata["corpus_raw_content"] = structured

    # Preserve additional_kwargs
    additional = getattr(msg, "additional_kwargs", None)
    if isinstance(additional, Mapping):
        metadata.update(dict(additional))

    return NormalizedMessage(role=role, content=content, metadata=metadata)


def to_langchain(msg: NormalizedMessage) -> Any:
    """
    NormalizedMessage → LangChain BaseMessage.

    Raises:
        RuntimeError: If LangChain is not installed
    """
    try:
        from langchain_core.messages import (  # type: ignore
            SystemMessage,
            HumanMessage,
            AIMessage,
            ToolMessage,
        )
    except Exception as exc:
        raise RuntimeError(
            "LangChain is not installed, cannot create LangChain messages"
        ) from exc

    # Reconstruct content
    content = msg.raw_content if msg.raw_content is not None else msg.content

    # Build additional_kwargs from metadata
    additional_kwargs = {
        k: v
        for k, v in msg.metadata.items()
        if not k.startswith("corpus_")
    }

    # Select message class
    if msg.role == "system":
        return SystemMessage(content=content, additional_kwargs=additional_kwargs)
    elif msg.role == "user":
        return HumanMessage(content=content, additional_kwargs=additional_kwargs)
    elif msg.role == "assistant":
        return AIMessage(content=content, additional_kwargs=additional_kwargs)
    elif msg.role in {"function", "tool"}:
        return ToolMessage(content=content, additional_kwargs=additional_kwargs)
    else:
        # Fallback for unknown roles
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
    """
    raw_role = getattr(msg, "role", None)
    if hasattr(raw_role, "value"):
        raw_role = getattr(raw_role, "value")

    raw_content = getattr(msg, "content", "")

    role, original_role = _normalize_role(raw_role)
    content, structured = _stringify_content(raw_content)

    metadata: Dict[str, Any] = {"corpus_raw": msg}
    if original_role is not None:
        metadata["corpus_original_role"] = original_role
    if structured is not None:
        metadata["corpus_raw_content"] = structured

    # Preserve additional_kwargs
    additional = getattr(msg, "additional_kwargs", None)
    if isinstance(additional, Mapping):
        metadata.update(dict(additional))

    return NormalizedMessage(role=role, content=content, metadata=metadata)


def to_llamaindex(msg: NormalizedMessage) -> Any:
    """
    NormalizedMessage → LlamaIndex ChatMessage.

    Raises:
        RuntimeError: If llama-index is not installed
    """
    try:
        from llama_index.core.llms import ChatMessage, MessageRole  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "llama-index is not installed, cannot create ChatMessage"
        ) from exc

    # Map role to MessageRole enum
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

    # Build additional_kwargs
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
    Semantic Kernel dict → NormalizedMessage.

    Expected shape:
        {"role": str, "content": str, "metadata": {...}}

    Args:
        msg: SK message dict

    Returns:
        NormalizedMessage
    """
    raw_role = msg.get("role", "user")
    raw_content = msg.get("content", "")

    role, original_role = _normalize_role(raw_role)
    content, structured = _stringify_content(raw_content)

    metadata: Dict[str, Any] = {"corpus_raw": msg}
    if original_role is not None:
        metadata["corpus_original_role"] = original_role
    if structured is not None:
        metadata["corpus_raw_content"] = structured

    # Preserve SK metadata
    sk_metadata = msg.get("metadata", {})
    if isinstance(sk_metadata, Mapping):
        metadata.update(dict(sk_metadata))

    return NormalizedMessage(role=role, content=content, metadata=metadata)


def to_semantic_kernel(msg: NormalizedMessage) -> Dict[str, Any]:
    """
    NormalizedMessage → Semantic Kernel dict.

    Returns:
        {"role": str, "content": str, "metadata": {...}}
    """
    # Reconstruct content
    content = msg.raw_content if msg.raw_content is not None else msg.content

    # Build metadata (exclude corpus_ keys)
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


def from_autogen(msg: Any) -> NormalizedMessage:
    """
    AutoGen message → NormalizedMessage.

    Supports:
    - Mapping with keys: role, content, name, tool_calls, function_call
    - Objects with attributes: role, content, ...

    Args:
        msg: AutoGen message (dict or object)

    Returns:
        NormalizedMessage
    """
    if isinstance(msg, Mapping):
        raw_role = msg.get("role", "user")
        raw_content = msg.get("content", "")
        content, structured = _stringify_content(raw_content)

        metadata: Dict[str, Any] = {"corpus_raw": msg}
        for k, v in msg.items():
            if k not in {"role", "content"}:
                metadata[k] = v
    else:
        raw_role = getattr(msg, "role", "user")
        raw_content = getattr(msg, "content", "")
        content, structured = _stringify_content(raw_content)

        metadata = {"corpus_raw": msg}
        for attr in ("name", "tool_calls", "function_call"):
            if hasattr(msg, attr):
                metadata[attr] = getattr(msg, attr)

    role, original_role = _normalize_role(raw_role)
    if original_role is not None:
        metadata["corpus_original_role"] = original_role
    if structured is not None:
        metadata["corpus_raw_content"] = structured

    return NormalizedMessage(role=role, content=content, metadata=metadata)


def to_autogen(msg: NormalizedMessage) -> Dict[str, Any]:
    """
    NormalizedMessage → AutoGen-style dict.

    Returns:
        {"role": str, "content": str, ...}
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


def from_crewai(msg: Any) -> NormalizedMessage:
    """
    CrewAI message → NormalizedMessage.

    Supports:
    - Mapping with role/content
    - Objects with .role / .content attributes

    Args:
        msg: CrewAI message (dict or object)

    Returns:
        NormalizedMessage
    """
    if isinstance(msg, Mapping):
        raw_role = msg.get("role", "user")
        raw_content = msg.get("content", "")
        content, structured = _stringify_content(raw_content)

        metadata: Dict[str, Any] = {"corpus_raw": msg}
        for k, v in msg.items():
            if k not in {"role", "content"}:
                metadata[k] = v
    else:
        raw_role = getattr(msg, "role", "user")
        raw_content = getattr(msg, "content", "")
        content, structured = _stringify_content(raw_content)

        metadata = {"corpus_raw": msg}
        for attr in ("name", "agent", "tool_calls", "function_call"):
            if hasattr(msg, attr):
                metadata[attr] = getattr(msg, attr)

    role, original_role = _normalize_role(raw_role)
    if original_role is not None:
        metadata["corpus_original_role"] = original_role
    if structured is not None:
        metadata["corpus_raw_content"] = structured

    return NormalizedMessage(role=role, content=content, metadata=metadata)


def to_crewai(msg: NormalizedMessage) -> Dict[str, Any]:
    """
    NormalizedMessage → CrewAI-compatible dict.

    Returns:
        {"role": str, "content": str, ...}
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
]
