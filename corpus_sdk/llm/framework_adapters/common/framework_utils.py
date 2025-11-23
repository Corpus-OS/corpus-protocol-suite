# corpus_sdk/llm/framework_adapters/common/framework_utils.py
# SPDX-License-Identifier: Apache-2.0
"""
Shared utilities for framework-specific LLM adapters.

This module centralizes common logic used across all LLM framework adapters:

- Coercing provider / translator results into canonical **text** or **chat message**
  shapes
- Coercing token-usage structures safely
- Emitting consistent, framework-aware warnings
- Providing light security / resource limits (depth, size, token caps, stream caps)
- Inferring / normalizing framework names for observability
- Best-effort streaming extraction for text and chat deltas

It intentionally stays *framework-neutral* and uses only:

- Standard library types
- Simple, caller-provided error codes + limits
- No direct dependencies on any specific LLM provider
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
)

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core typing + config helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoercionErrorCodes:
    """
    Structured bundle of error codes used during LLM coercion.

    Attributes
    ----------
    invalid_result:
        Used when the result structure is not a valid container (e.g. no
        usable text / usage / message fields found).

    empty_result:
        Used when no meaningful content is present after processing (e.g.
        empty string, no messages, etc.).

    conversion_error:
        Used when numeric conversion fails (e.g. token counts) or when
        otherwise-valid content cannot be converted into the expected type.

    framework_label:
        Optional default framework label ("langchain", "llamaindex", etc.).
        Used as a fallback when the caller does not supply one explicitly.
    """

    invalid_result: str
    empty_result: str
    conversion_error: str
    framework_label: str = "unknown"


@dataclass(frozen=True)
class TokenUsage:
    """Canonical representation of token usage for an LLM operation."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class PromptWarningConfig:
    """
    Configuration for soft warnings on very large prompts.

    Attributes
    ----------
    warn_char_threshold:
        Total character count above which a warning may be emitted.

    warn_message_threshold:
        Number of messages above which a warning may be emitted.

    framework_label:
        Label used only for logging / observability.
    """

    warn_char_threshold: int = 200_000
    warn_message_threshold: int = 10_000
    framework_label: str = "unknown"


@dataclass(frozen=True)
class ResourceLimits:
    """
    Resource limits for content coercion and streaming.

    Attributes
    ----------
    max_content_depth:
        Maximum nesting depth when flattening content structures.

    max_single_content_size:
        Maximum number of characters to retain for a single content block.

    max_stream_events:
        Hard cap on the number of stream events to consume; None disables
        this limit.
    """

    max_content_depth: int = 16
    max_single_content_size: int = 10_000_000
    max_stream_events: Optional[int] = 10_000


@dataclass(frozen=True)
class TokenLimits:
    """
    Bounds for token usage coercion.

    Attributes
    ----------
    max_tokens_per_field:
        Maximum allowed value for individual fields such as prompt_tokens
        or completion_tokens.

    max_total_tokens:
        Maximum allowed value for total_tokens.
    """

    max_tokens_per_field: int = 100_000_000
    max_total_tokens: int = 200_000_000


_ALLOWED_ROLES = {"system", "user", "assistant", "tool", "function"}


# ---------------------------------------------------------------------------
# Framework + error-code helpers
# ---------------------------------------------------------------------------


def validate_role(role: Any) -> bool:
    """
    Validate a chat role against a conservative allow-list.

    This is intentionally strict: anything outside the known set is treated
    as invalid (caller can decide whether to reject or just log).
    """
    return isinstance(role, str) and role in _ALLOWED_ROLES


def infer_framework_name(source: Any, default: str = "unknown") -> str:
    """
    Best-effort framework inference based on an object or module path.

    Examples
    --------
    - "langchain"                   → "langchain"
    - object from langchain_core    → "langchain"
    - object from llama_index.core  → "llamaindex"
    - adapter in corpus_sdk         → "corpus"

    This is optional; callers can still pass an explicit framework label.
    """
    if isinstance(source, str) and source.strip():
        return source.strip()

    if source is None:
        return default

    mod = getattr(source, "__module__", "") or ""

    if mod.startswith("langchain"):
        return "langchain"
    if mod.startswith("llama_index") or mod.startswith("llamaindex"):
        return "llamaindex"
    if mod.startswith("semantic_kernel"):
        return "semantic_kernel"
    if mod.startswith("corpus_sdk"):
        return "corpus"

    return default


def _normalize_framework(
    framework: Optional[str],
    error_codes: CoercionErrorCodes,
) -> str:
    """
    Normalize the framework label using either the explicit argument or
    the error_codes.framework_label as a fallback.
    """
    if isinstance(framework, str) and framework.strip():
        return framework.strip()

    label = (error_codes.framework_label or "").strip()
    return label or "unknown"


def _validate_error_codes(error_codes: CoercionErrorCodes) -> None:
    """
    Validate that a CoercionErrorCodes bundle is structurally sound.

    This catches configuration mistakes early, rather than silently allowing
    empty strings or wrong types.
    """
    if not isinstance(error_codes, CoercionErrorCodes):
        raise TypeError("error_codes must be an instance of CoercionErrorCodes")

    for field_name in ("invalid_result", "empty_result", "conversion_error"):
        value = getattr(error_codes, field_name, None)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"CoercionErrorCodes.{field_name} must be a non-empty string"
            )


# ---------------------------------------------------------------------------
# Content flattening + text coercion
# ---------------------------------------------------------------------------


def _flatten_content_blocks(
    content: Any,
    *,
    limits: Optional[ResourceLimits] = None,
    _depth: int = 0,
) -> str:
    """
    Flatten modern LLM content structures into a single string.

    Supported content patterns (best-effort):
    - "simple text"
    - {"text": "content"}
    - {"content": "text"}
    - [{"type": "text", "text": "part1"}, {"type": "text", "text": "part2"}]
    - Mixed tool/text blocks, e.g.:
        [
            {"type": "text", "text": "Before tool"},
            {"type": "tool_use", ...},
            {"type": "text", "text": "After tool"},
        ]
    - Arbitrary nested combinations of the above

    Resource limits (depth + max size) are enforced defensively.
    """
    limits = limits or ResourceLimits()

    if _depth > limits.max_content_depth:
        LOG.debug("max_content_depth exceeded while flattening (depth=%d)", _depth)
        return ""

    if content is None:
        return ""

    # Fast path: simple string
    if isinstance(content, str):
        text = content
    elif isinstance(content, Mapping):
        # Common patterns: {"type": "text", "text": "..."} or {"content": "..."}
        if "text" in content and isinstance(content["text"], str):
            text = content["text"]
        elif "content" in content:
            inner = content["content"]
            if isinstance(inner, str):
                text = inner
            else:
                text = _flatten_content_blocks(
                    inner,
                    limits=limits,
                    _depth=_depth + 1,
                )
        else:
            # Last-resort stringification for unknown shapes
            text = str(content)
    elif isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        parts: List[str] = []
        for block in content:
            try:
                part = _flatten_content_blocks(
                    block,
                    limits=limits,
                    _depth=_depth + 1,
                )
            except Exception:
                # Never let a single bad block take down the whole response
                continue
            if part:
                parts.append(part)
        text = "".join(parts)
    else:
        text = str(content)

    # Enforce max-single-content size
    if len(text) > limits.max_single_content_size:
        LOG.warning(
            "content size %d exceeds max_single_content_size=%d; truncating",
            len(text),
            limits.max_single_content_size,
        )
        return text[: limits.max_single_content_size]

    return text


def _extract_text_object(
    result: Any,
    *,
    limits: Optional[ResourceLimits],
) -> Any:
    """
    Extract the primary "text-like" object from an arbitrary LLM result.

    This does *not* guarantee a string; it returns the most promising object
    for text coercion (which may be a string, list of spans, or content blocks).
    """
    # Direct string
    if isinstance(result, str):
        return result

    # Common mapping-based responses
    if isinstance(result, Mapping):
        if "text" in result:
            return result["text"]
        if "completion" in result:
            return result["completion"]
        if "content" in result:
            content = result["content"]
            if isinstance(content, str):
                return content
            return _flatten_content_blocks(content, limits=limits)

        # OpenAI-style: {"choices": [...]}
        choices = result.get("choices")
        if isinstance(choices, Sequence) and choices:
            first = choices[0]
            if isinstance(first, Mapping):
                if "text" in first:
                    return first["text"]

                # Chat-style: {"message": {"content": ...}}
                message = first.get("message")
                if isinstance(message, Mapping) and "content" in message:
                    return _flatten_content_blocks(
                        message["content"],
                        limits=limits,
                    )

                # Delta-style streaming snapshots
                delta = first.get("delta")
                if isinstance(delta, Mapping) and "content" in delta:
                    return _flatten_content_blocks(
                        delta["content"],
                        limits=limits,
                    )

    # Attribute-based access for SDK-specific objects
    for attr in ("text", "completion", "content"):
        if hasattr(result, attr):
            value = getattr(result, attr)
            if attr == "content" and not isinstance(value, str):
                return _flatten_content_blocks(value, limits=limits)
            return value

    # Last resort: return as-is and let caller decide
    return result


def coerce_text_completion(
    result: Any,
    *,
    framework: Optional[str],
    error_codes: CoercionErrorCodes,
    logger: Optional[logging.Logger] = None,
    limits: Optional[ResourceLimits] = None,
) -> str:
    """
    Coerce an arbitrary LLM completion result into a plain text string.

    Strategy:
    - Extract a "text-like" object via `_extract_text_object`
    - If string → validate & return
    - If sequence → join as string
    - Otherwise → error with framework-specific error codes
    """
    _validate_error_codes(error_codes)
    framework_name = _normalize_framework(framework, error_codes)
    log = logger or LOG

    obj = _extract_text_object(result, limits=limits)

    # Fast paths
    if isinstance(obj, str):
        text = obj
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        try:
            text = "".join(str(part) for part in obj)
        except Exception as exc:  # noqa: BLE001
            raise TypeError(
                f"[{error_codes.conversion_error}] "
                f"{framework_name}: failed to join text sequence: {exc}"
            ) from exc
    else:
        raise TypeError(
            f"[{error_codes.invalid_result}] "
            f"{framework_name}: result does not contain a valid text payload "
            f"(type={type(obj).__name__})"
        )

    if text == "":
        raise ValueError(
            f"[{error_codes.empty_result}] "
            f"{framework_name}: completion text is empty"
        )

    log.debug("%s: coerced text completion (length=%d)", framework_name, len(text))
    return text


# ---------------------------------------------------------------------------
# Chat message coercion
# ---------------------------------------------------------------------------


def _coerce_single_chat_message(
    raw: Any,
    *,
    framework_name: str,
    error_codes: CoercionErrorCodes,
    limits: ResourceLimits,
    skip_invalid: bool,
    logger: logging.Logger,
) -> Optional[Dict[str, str]]:
    """
    Internal helper to coerce a single chat message-like object.

    Returns a normalized {"role": ..., "content": ...} dict or None if
    skip_invalid=True and the message is discarded.
    """
    role: str
    content: Any

    if isinstance(raw, Mapping):
        role = raw.get("role", "assistant")
        content = raw.get("content")
    elif isinstance(raw, str):
        # Treat a bare string as an assistant message
        role = "assistant"
        content = raw
    else:
        msg = (
            f"[{error_codes.invalid_result}] "
            f"{framework_name}: chat message is not a mapping or string "
            f"(type={type(raw).__name__})"
        )
        if skip_invalid:
            logger.warning("%s; skipping invalid message", msg)
            return None
        raise TypeError(msg)

    if not validate_role(role):
        msg = (
            f"[{error_codes.invalid_result}] "
            f"{framework_name}: invalid chat role {role!r}"
        )
        if skip_invalid:
            logger.warning("%s; skipping invalid role", msg)
            return None
        raise ValueError(msg)

    if content is None:
        content_str = ""
    elif isinstance(content, str):
        content_str = content
    else:
        content_str = _flatten_content_blocks(content, limits=limits)

    if content_str == "":
        msg = (
            f"[{error_codes.empty_result}] "
            f"{framework_name}: empty content for role {role!r}"
        )
        if skip_invalid:
            logger.debug("%s; skipping empty message", msg)
            return None
        raise ValueError(msg)

    return {"role": role, "content": content_str}


def coerce_chat_messages(
    result: Any,
    *,
    framework: Optional[str],
    error_codes: CoercionErrorCodes,
    skip_invalid: bool = False,
    logger: Optional[logging.Logger] = None,
    limits: Optional[ResourceLimits] = None,
) -> List[Dict[str, str]]:
    """
    Coerce an arbitrary chat result into a list of normalized messages.

    Normalized shape
    ----------------
    Each message is a dict:
        {"role": <str>, "content": <str>}

    Supported input patterns (best-effort)
    --------------------------------------
    - List[{"role": ..., "content": ...}]
    - {"messages": [...]}              # e.g. framework-specific histories
    - {"role": ..., "content": ...}    # single message
    - {"choices": [...]}               # OpenAI-style responses
        * uses the first choice's message/delta/content by default
    - "plain text"                     # treated as a single assistant message

    Parameters
    ----------
    result:
        Arbitrary provider / framework result object.
    framework:
        Framework label for logging / diagnostics.
    error_codes:
        Error code bundle (must be valid).
    skip_invalid:
        If True, invalid messages are logged-and-skipped instead of raising.
    logger:
        Optional logger.
    limits:
        Optional ResourceLimits for content flattening.

    Returns
    -------
    List[Dict[str, str]]
        Non-empty list of normalized chat messages.
    """
    _validate_error_codes(error_codes)
    framework_name = _normalize_framework(framework, error_codes)
    log = logger or LOG
    limits = limits or ResourceLimits()

    # Decide what we treat as the "message container"
    messages_raw: List[Any]

    if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        # Already a sequence of message-like objects
        messages_raw = list(result)
    elif isinstance(result, Mapping):
        if "messages" in result and isinstance(result["messages"], Sequence):
            messages_raw = list(result["messages"])
        elif "choices" in result and isinstance(result["choices"], Sequence):
            choices = result["choices"]
            if not choices:
                messages_raw = []
            else:
                first = choices[0]
                if isinstance(first, Mapping):
                    msg = (
                        first.get("message")
                        or first.get("delta")
                        or first.get("content")
                        or first
                    )
                    messages_raw = [msg]
                else:
                    messages_raw = [first]
        elif "role" in result or "content" in result:
            messages_raw = [result]
        else:
            raise TypeError(
                f"[{error_codes.invalid_result}] "
                f"{framework_name}: result does not contain recognizable "
                f"chat message structures"
            )
    elif isinstance(result, str):
        messages_raw = [result]
    else:
        raise TypeError(
            f"[{error_codes.invalid_result}] "
            f"{framework_name}: unsupported chat result type "
            f"(type={type(result).__name__})"
        )

    normalized: List[Dict[str, str]] = []

    for idx, raw in enumerate(messages_raw):
        msg = _coerce_single_chat_message(
            raw,
            framework_name=framework_name,
            error_codes=error_codes,
            limits=limits,
            skip_invalid=skip_invalid,
            logger=log,
        )
        if msg is not None:
            normalized.append(msg)

    if not normalized:
        raise ValueError(
            f"[{error_codes.empty_result}] "
            f"{framework_name}: no valid chat messages found"
        )

    log.debug(
        "%s: coerced %d chat messages (origin_type=%s)",
        framework_name,
        len(normalized),
        type(result).__name__,
    )
    return normalized


# ---------------------------------------------------------------------------
# Token usage coercion
# ---------------------------------------------------------------------------


def coerce_token_usage(
    result: Any,
    *,
    framework: Optional[str],
    error_codes: CoercionErrorCodes,
    logger: Optional[logging.Logger] = None,
    limits: Optional[TokenLimits] = None,
) -> TokenUsage:
    """
    Coerce a generic LLM response into a canonical TokenUsage structure.

    Accepts a variety of shapes:
    - {"usage": {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}}
    - {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}
    - Objects with `.usage` mapping attribute

    Enforces:
    - Non-negative, finite integers
    - Upper bounds for individual fields and total_tokens
    """
    _validate_error_codes(error_codes)
    framework_name = _normalize_framework(framework, error_codes)
    log = logger or LOG
    limits = limits or TokenLimits()

    # Locate the "usage" mapping
    if isinstance(result, Mapping) and "usage" in result and isinstance(
        result["usage"], Mapping
    ):
        usage = result["usage"]
    elif isinstance(result, Mapping):
        usage = result
    elif hasattr(result, "usage") and isinstance(getattr(result, "usage"), Mapping):
        usage = getattr(result, "usage")
    else:
        raise TypeError(
            f"[{error_codes.invalid_result}] "
            f"{framework_name}: result does not expose a usable 'usage' mapping"
        )

    def _to_int(key: str) -> int:
        raw = usage.get(key, 0)
        try:
            if isinstance(raw, float) and not math.isfinite(raw):
                raise ValueError(f"{key} is non-finite: {raw!r}")
            value = int(raw)
        except (TypeError, ValueError, OverflowError) as exc:  # noqa: BLE001
            raise TypeError(
                f"[{error_codes.conversion_error}] "
                f"{framework_name}: invalid {key} value {raw!r}: {exc}"
            ) from exc

        if value < 0 or value > limits.max_tokens_per_field:
            raise ValueError(
                f"[{error_codes.conversion_error}] "
                f"{framework_name}: {key} out of allowed range "
                f"(value={value}, max={limits.max_tokens_per_field})"
            )
        return value

    prompt_tokens = _to_int("prompt_tokens")
    completion_tokens = _to_int("completion_tokens")

    if "total_tokens" in usage:
        total_tokens = _to_int("total_tokens")
    else:
        total_tokens = prompt_tokens + completion_tokens

    if total_tokens > limits.max_total_tokens:
        raise ValueError(
            f"[{error_codes.conversion_error}] "
            f"{framework_name}: total_tokens out of allowed range "
            f"(value={total_tokens}, max={limits.max_total_tokens})"
        )

    log.debug(
        "%s: coerced token usage prompt=%d completion=%d total=%d",
        framework_name,
        prompt_tokens,
        completion_tokens,
        total_tokens,
    )

    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------


def iter_text_from_stream_events(
    events: Iterable[Any],
    *,
    framework: Optional[str],
    error_codes: CoercionErrorCodes,
    logger: Optional[logging.Logger] = None,
    limits: Optional[ResourceLimits] = None,
    skip_errors: bool = True,
) -> Iterator[str]:
    """
    Iterate over a stream of provider events and yield text deltas.

    This is a *best-effort* abstraction over common streaming patterns:
    - OpenAI ChatCompletionChunk / CompletionChunk
    - Anthropic-style content deltas
    - Framework-specific wrappers that expose "text", "content", or "choices"

    Behavior
    --------
    - Respects ResourceLimits.max_stream_events (if set)
    - Uses `_extract_text_object` per event, then emits non-empty strings
    - When `skip_errors=True`, logs-and-skips malformed events instead of raising

    Parameters
    ----------
    events:
        Iterable of provider-specific stream events.
    framework:
        Framework label for logging / diagnostics.
    error_codes:
        Error codes bundle for structured failures when skip_errors=False.
    logger:
        Optional logger.
    limits:
        Optional ResourceLimits for content flattening and stream caps.
    skip_errors:
        If True, individual event errors do not terminate the iterator.
    """
    _validate_error_codes(error_codes)
    framework_name = _normalize_framework(framework, error_codes)
    log = logger or LOG
    limits = limits or ResourceLimits()

    max_events = limits.max_stream_events
    count = 0

    for event in events:
        count += 1

        if max_events is not None and count > max_events:
            log.warning(
                "%s: max_stream_events=%d exceeded; stopping stream consumption",
                framework_name,
                max_events,
            )
            break

        try:
            obj = _extract_text_object(event, limits=limits)
            if isinstance(obj, str):
                if obj:
                    yield obj
            elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
                chunk = "".join(str(part) for part in obj)
                if chunk:
                    yield chunk
            # else: ignore events with no textual content
        except Exception as exc:  # noqa: BLE001
            if skip_errors:
                log.warning(
                    "%s: skipping stream event %d due to extraction error: %s",
                    framework_name,
                    count,
                    exc,
                )
                continue

            raise TypeError(
                f"[{error_codes.invalid_result}] "
                f"{framework_name}: failed to extract text from stream event: {exc}"
            ) from exc


def _extract_message_like_from_event(event: Any) -> Any:
    """
    Best-effort extraction of a message-like object from a stream event.

    This does *not* normalize the message – it just pulls out the most
    likely message/delta payload for further coercion.
    """
    if isinstance(event, Mapping):
        # OpenAI-like patterns
        choices = event.get("choices")
        if isinstance(choices, Sequence) and choices:
            out: List[Any] = []
            for ch in choices:
                if not isinstance(ch, Mapping):
                    out.append(ch)
                    continue
                msg = ch.get("delta") or ch.get("message") or ch.get("content") or ch
                out.append(msg)
            return out

        if "message" in event:
            return event["message"]
        if "delta" in event:
            return event["delta"]
        if "role" in event or "content" in event:
            return event

    # Fall through: let caller decide
    return event


def iter_chat_messages_from_stream_events(
    events: Iterable[Any],
    *,
    framework: Optional[str],
    error_codes: CoercionErrorCodes,
    logger: Optional[logging.Logger] = None,
    limits: Optional[ResourceLimits] = None,
    skip_errors: bool = True,
) -> Iterator[Dict[str, str]]:
    """
    Iterate over a stream of provider events and yield **normalized chat messages**.

    This is built on top of `coerce_chat_messages` and uses the same normalized
    shape:

        {"role": <str>, "content": <str>}

    Behavior
    --------
    - Respects ResourceLimits.max_stream_events (if set)
    - For each event, extracts message-like payload(s) and normalizes them
    - When `skip_errors=True`, logs-and-skips invalid events / messages

    Parameters
    ----------
    events:
        Iterable of provider-specific stream events.
    framework:
        Framework label for logging / diagnostics.
    error_codes:
        Error codes bundle for structured failures when skip_errors=False.
    logger:
        Optional logger.
    limits:
        Optional ResourceLimits (used for content flattening).
    skip_errors:
        If True, invalid messages are logged-and-skipped.
    """
    _validate_error_codes(error_codes)
    framework_name = _normalize_framework(framework, error_codes)
    log = logger or LOG
    limits = limits or ResourceLimits()

    max_events = limits.max_stream_events
    count = 0

    for event in events:
        count += 1

        if max_events is not None and count > max_events:
            log.warning(
                "%s: max_stream_events=%d exceeded; stopping chat stream consumption",
                framework_name,
                max_events,
            )
            break

        try:
            msg_like = _extract_message_like_from_event(event)

            # Normalize to a small container and reuse chat coercion
            if isinstance(msg_like, Sequence) and not isinstance(
                msg_like,
                (str, bytes),
            ):
                container = msg_like
            else:
                container = [msg_like]

            messages = coerce_chat_messages(
                container,
                framework=framework_name,
                error_codes=error_codes,
                skip_invalid=True,  # per-event invalids are fine to drop
                logger=log,
                limits=limits,
            )

            for msg in messages:
                yield msg

        except Exception as exc:  # noqa: BLE001
            if skip_errors:
                log.warning(
                    "%s: skipping chat stream event %d due to error: %s",
                    framework_name,
                    count,
                    exc,
                )
                continue

            raise


__all__ = [
    "CoercionErrorCodes",
    "TokenUsage",
    "PromptWarningConfig",
    "ResourceLimits",
    "TokenLimits",
    "validate_role",
    "infer_framework_name",
    "coerce_text_completion",
    "coerce_chat_messages",
    "coerce_token_usage",
    "iter_text_from_stream_events",
    "iter_chat_messages_from_stream_events",
]
