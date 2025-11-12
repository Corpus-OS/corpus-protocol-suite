# SPDX-License-Identifier: Apache-2.0
"""
Streaming Frames Validator

Validates NDJSON / SSE / WebSocket-like JSON frames against a union frame schema ($id),
and enforces protocol-level invariants:

- Exactly one terminal frame (event=end OR event=error)
- No data frames after terminal
- Optional max frame size enforcement
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from .schema_registry import assert_valid


@dataclass(frozen=True)
class StreamValidationReport:
    total_frames: int
    data_frames: int
    ended_ok: bool
    errored: bool
    terminal_seen: bool


class StreamProtocolError(AssertionError):
    pass


def _parse_ndjson_lines(ndjson_text: str) -> List[dict]:
    frames: List[dict] = []
    for i, line in enumerate(ndjson_text.splitlines(), 1):
        line = line.strip()
        if not line:
            # skip empty line(s)
            continue
        try:
            frames.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise StreamProtocolError(f"Invalid NDJSON at line {i}: {e}") from e
    return frames


def _parse_sse_lines(sse_text: str) -> List[dict]:
    """
    Expects classic SSE format:
        event: <event>
        data: {json}
    (blank line between events tolerated but not required)
    """
    frames: List[dict] = []
    current_event: Optional[str] = None
    current_data_lines: List[str] = []

    def flush():
        nonlocal current_event, current_data_lines
        if current_event is None:
            return
        try:
            payload = json.loads("\n".join(current_data_lines)) if current_data_lines else {}
        except json.JSONDecodeError as e:
            raise StreamProtocolError(f"Invalid SSE data JSON for event '{current_event}': {e}") from e
        frames.append({"event": current_event, "data": payload} if current_event == "data" else {"event": current_event, **payload})
        current_event, current_data_lines = None, []

    for raw in sse_text.splitlines():
        line = raw.strip("\n")
        if not line:
            flush()
            continue

        if line.startswith("event:"):
            flush()
            current_event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            current_data_lines.append(line[len("data:"):].lstrip())
        else:
            # Ignore other SSE fields for now
            pass

    flush()
    return frames


def validate_stream_frames(
    frames: Iterable[dict],
    *,
    union_schema_id: str,
    component: str,
    max_frame_bytes: Optional[int] = 1_048_576,  # 1 MiB guidance
) -> StreamValidationReport:
    """
    Validate a pre-parsed iterable of frames against the union schema (oneOf of data/end/error)
    and enforce terminal rules.
    """
    total = 0
    data_count = 0
    terminal_seen = False
    ended_ok = False
    errored = False

    for f in frames:
        total += 1

        if max_frame_bytes is not None:
            # Quick size check; if it fails, surface an assertion with context
            approx_size = len(json.dumps(f, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
            if approx_size > max_frame_bytes:
                raise StreamProtocolError(f"Frame #{total} exceeds max_frame_bytes={max_frame_bytes} (got ~{approx_size} bytes)")

        # Validate shape
        assert_valid(union_schema_id, f, context=f"{component}.stream frame #{total}")

        event = f.get("event")
        if terminal_seen:
            raise StreamProtocolError(f"Data after terminal frame: saw '{event}' at frame #{total}")

        if event == "data":
            data_count += 1
        elif event == "end":
            terminal_seen = True
            ended_ok = True
        elif event == "error":
            terminal_seen = True
            errored = True
        else:
            # Union schema should have caught unknown events, but keep a guard:
            raise StreamProtocolError(f"Unknown event type '{event}' at frame #{total}")

    if not terminal_seen:
        raise StreamProtocolError("Stream completed without a terminal frame (end or error)")

    # Cannot be both if schema is correct, but assert anyway:
    if ended_ok and errored:
        raise StreamProtocolError("Stream contained both 'end' and 'error' terminal frames")

    return StreamValidationReport(
        total_frames=total,
        data_frames=data_count,
        ended_ok=ended_ok,
        errored=errored,
        terminal_seen=terminal_seen,
    )


# Convenience adapters

def validate_ndjson_stream(
    ndjson_text: str,
    *,
    union_schema_id: str,
    component: str,
    max_frame_bytes: Optional[int] = 1_048_576,
) -> StreamValidationReport:
    frames = _parse_ndjson_lines(ndjson_text)
    return validate_stream_frames(
        frames,
        union_schema_id=union_schema_id,
        component=component,
        max_frame_bytes=max_frame_bytes,
    )


def validate_sse_stream(
    sse_text: str,
    *,
    union_schema_id: str,
    component: str,
    max_frame_bytes: Optional[int] = 1_048_576,
) -> StreamValidationReport:
    frames = _parse_sse_lines(sse_text)
    return validate_stream_frames(
        frames,
        union_schema_id=union_schema_id,
        component=component,
        max_frame_bytes=max_frame_bytes,
    )
