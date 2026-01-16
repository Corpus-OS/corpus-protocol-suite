# SPDX-License-Identifier: Apache-2.0
"""
Protocol-Compliant Stream Validator for Corpus Protocol Suite.

SCHEMA.md-aligned streaming validator:

- Streaming success frames MUST be the canonical streaming envelope:
    { ok: true, code: "STREAMING", ms: number>=0, chunk: <payload> }
- Stream termination MUST be exactly one terminal condition:
    - a success chunk with chunk.is_final == true, OR
    - a standard error envelope (not a streaming error)
- No content after terminal frame.
- Error envelopes MUST include:
    ok=false, code, error, message, retry_after_ms, details, ms

Validation modes:
- STRICT / SAMPLED: perform JSON Schema validation via schema_registry.assert_valid
- LAZY / COLLECT_ERRORS: still enforce SCHEMA.md protocol invariants (no permissive drift)

Transport:
- NDJSON / SSE / RAW_JSON supported.
- Transport parsers DO NOT strip unknown keys (to avoid masking schema violations).
"""

from __future__ import annotations

import asyncio
import json
import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from .schema_registry import assert_valid, SchemaRegistry


class StreamFormat(Enum):
    """Supported stream formats."""
    NDJSON = "ndjson"
    SSE = "sse"
    RAW_JSON = "raw_json"


class ValidationMode(Enum):
    """Validation strictness modes."""
    STRICT = "strict"                 # Validate every frame
    SAMPLED = "sampled"               # Validate sample of frames
    LAZY = "lazy"                     # Validate only protocol rules, skip schema validation
    COLLECT_ERRORS = "collect_errors" # Collect all errors instead of failing fast


@dataclass(frozen=True)
class ValidationError:
    """A single validation error."""
    frame_number: int
    error_type: str
    message: str
    exception: Optional[Exception] = None


@dataclass(frozen=True)
class StreamValidationReport:
    """Comprehensive report of stream validation results."""
    total_frames: int
    data_frames: int
    ended_ok: bool
    errored: bool
    terminal_seen: bool
    validation_time_ms: float
    total_bytes: int
    max_frame_bytes: int
    terminal_frame_position: int
    frames_validated: int
    frames_skipped: int
    format: StreamFormat
    mode: ValidationMode
    validation_errors: list[ValidationError] = field(default_factory=list)

    # Performance metrics
    bytes_per_second: float = field(init=False)
    frames_per_second: float = field(init=False)
    validation_coverage: float = field(init=False)

    def __post_init__(self) -> None:
        seconds = self.validation_time_ms / 1000.0
        object.__setattr__(self, "bytes_per_second", self.total_bytes / seconds if seconds > 0 else float("inf"))
        object.__setattr__(self, "frames_per_second", self.total_frames / seconds if seconds > 0 else float("inf"))
        object.__setattr__(self, "validation_coverage", self.frames_validated / self.total_frames if self.total_frames > 0 else 1.0)

    @property
    def is_valid(self) -> bool:
        return len(self.validation_errors) == 0

    @property
    def error_summary(self) -> str:
        if not self.validation_errors:
            return "No validation errors"

        error_counts: dict[str, int] = {}
        for err in self.validation_errors:
            error_counts[err.error_type] = error_counts.get(err.error_type, 0) + 1

        lines = [f"Found {len(self.validation_errors)} validation errors:"]
        for etype, count in sorted(error_counts.items()):
            lines.append(f"  - {etype}: {count} error(s)")
        return "\n".join(lines)


@dataclass(frozen=True)
class ValidationConfig:
    """
    Configuration for stream validation.

    IMPORTANT (SCHEMA.md):
    - For streaming frames, pass a *streaming* success schema ID such as:
        https://corpusos.com/schemas/common/envelope.stream.success.json
      OR a protocol operation streaming success schema such as:
        https://corpusos.com/schemas/llm/llm.stream.success.json
        https://corpusos.com/schemas/graph/graph.stream_query.success.json
        https://corpusos.com/schemas/embedding/embedding.stream_embed.success.json
    """
    stream_frame_schema_id: str
    component: str
    max_frame_bytes: Optional[int] = 1_048_576  # 1 MiB
    mode: ValidationMode = ValidationMode.STRICT
    sample_rate: float = 0.1  # For SAMPLED mode
    performance_warning_threshold_ms: int = 1000
    large_stream_threshold: int = 1000
    enable_content_warnings: bool = True
    schema_registry: Optional[SchemaRegistry] = None

    # Telemetry hooks
    on_frame_validated: Optional[Callable[[dict[str, Any], int], None]] = None
    on_performance_warning: Optional[Callable[[str], None]] = None
    on_validation_error: Optional[Callable[[ValidationError], None]] = None


class StreamProtocolError(AssertionError):
    """Raised when stream protocol invariants are violated."""
    pass


class FrameSizeExceededError(StreamProtocolError):
    """Raised when frame exceeds size limits."""
    pass


class StreamParser(ABC):
    """Abstract base class for stream parsers."""

    @abstractmethod
    def parse(self, content: str) -> list[dict[str, Any]]:
        """Parse content into protocol envelope frames."""
        raise NotImplementedError

    @abstractmethod
    async def parse_async(self, content: str) -> list[dict[str, Any]]:
        """Parse content asynchronously into protocol envelope frames."""
        raise NotImplementedError

    @abstractmethod
    async def parse_streaming(self, lines: AsyncIterable[str]) -> AsyncIterator[dict[str, Any]]:
        """Parse content in true streaming fashion for unbounded streams."""
        raise NotImplementedError


class NDJSONParser(StreamParser):
    """NDJSON stream parser that extracts protocol envelopes."""

    def parse(self, content: str) -> list[dict[str, Any]]:
        frames: list[dict[str, Any]] = []
        for i, line in enumerate(content.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                frame = json.loads(line)
            except json.JSONDecodeError as e:
                raise StreamProtocolError(f"Invalid NDJSON at line #{i}: {e}") from e

            if not isinstance(frame, dict):
                raise StreamProtocolError(f"Frame #{i} must be a JSON object, got {type(frame).__name__}")

            # SCHEMA.md alignment: do NOT strip keys; let schema/protocol checks catch extras.
            frames.append(frame)

        return frames

    async def parse_async(self, content: str) -> list[dict[str, Any]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.parse, content)

    async def parse_streaming(self, lines: AsyncIterable[str]) -> AsyncIterator[dict[str, Any]]:
        line_num = 0
        async for line in lines:
            line_num += 1
            line = line.strip()
            if not line:
                continue
            try:
                frame = json.loads(line)
            except json.JSONDecodeError as e:
                raise StreamProtocolError(f"Invalid NDJSON at line #{line_num}: {e}") from e

            if not isinstance(frame, dict):
                raise StreamProtocolError(f"Frame #{line_num} must be a JSON object, got {type(frame).__name__}")

            # SCHEMA.md alignment: do NOT strip keys.
            yield frame


class SSEParser(StreamParser):
    """Server-Sent Events (SSE) parser that extracts protocol envelopes from 'data:' blocks."""

    def parse(self, sse_text: str) -> list[dict[str, Any]]:
        frames: list[dict[str, Any]] = []
        current_data_lines: list[str] = []

        def flush() -> None:
            nonlocal current_data_lines
            if not current_data_lines:
                return
            data_content = "\n".join(current_data_lines).strip()
            current_data_lines = []
            if not data_content:
                return
            try:
                envelope = json.loads(data_content)
            except json.JSONDecodeError as e:
                raise StreamProtocolError(f"Invalid JSON in SSE data: {e}") from e
            if not isinstance(envelope, dict):
                raise StreamProtocolError(f"SSE data must contain JSON object, got {type(envelope).__name__}")
            # SCHEMA.md alignment: do NOT strip keys.
            frames.append(envelope)

        for _line_num, raw_line in enumerate(sse_text.splitlines(), 1):
            line = raw_line.rstrip("\n\r")
            if not line:
                flush()
                continue

            if line.startswith("data:"):
                current_data_lines.append(line[5:].lstrip())
            elif line.startswith(("event:", "id:", "retry:", ":")):
                # ignore metadata/comment lines
                continue
            else:
                # treat unknown lines as data (tolerant)
                current_data_lines.append(line)

        flush()
        return frames

    async def parse_async(self, content: str) -> list[dict[str, Any]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.parse, content)

    async def parse_streaming(self, lines: AsyncIterable[str]) -> AsyncIterator[dict[str, Any]]:
        current_data_lines: list[str] = []

        async def flush() -> AsyncIterator[dict[str, Any]]:
            nonlocal current_data_lines
            if not current_data_lines:
                return
            data_content = "\n".join(current_data_lines).strip()
            current_data_lines = []
            if not data_content:
                return
            try:
                envelope = json.loads(data_content)
            except json.JSONDecodeError as e:
                raise StreamProtocolError(f"Invalid JSON in SSE data: {e}") from e
            if not isinstance(envelope, dict):
                raise StreamProtocolError(f"SSE data must contain JSON object, got {type(envelope).__name__}")
            # SCHEMA.md alignment: do NOT strip keys.
            yield envelope

        async for raw_line in lines:
            line = raw_line.rstrip("\n\r")
            if not line:
                async for frame in flush():
                    yield frame
                continue

            if line.startswith("data:"):
                current_data_lines.append(line[5:].lstrip())
            elif line.startswith(("event:", "id:", "retry:", ":")):
                continue
            else:
                current_data_lines.append(line)

        async for frame in flush():
            yield frame


class FrameValidator:
    """Protocol envelope validation logic (SCHEMA.md-aligned)."""

    STREAMING_CODE = "STREAMING"
    OK_CODE = "OK"

    @staticmethod
    def estimate_frame_size(frame: dict[str, Any]) -> int:
        return len(json.dumps(frame, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))

    @staticmethod
    def check_frame_size(frame: dict[str, Any], frame_num: int, max_frame_bytes: Optional[int]) -> None:
        if max_frame_bytes is None:
            return
        frame_size = FrameValidator.estimate_frame_size(frame)
        if frame_size > max_frame_bytes:
            raise FrameSizeExceededError(
                f"Frame #{frame_num} exceeds max_frame_bytes={max_frame_bytes} "
                f"(got {frame_size} bytes, ~{frame_size/1024/1024:.1f}MB)"
            )

    @staticmethod
    def validate_protocol_envelope(frame: dict[str, Any], frame_num: int) -> None:
        """
        Validate SCHEMA.md envelope invariants:

        - Streaming success: ok=true, code="STREAMING", ms>=0, chunk present
        - Error envelope: ok=false, required fields include retry_after_ms + details, ms>=0
        """
        if "ok" not in frame:
            raise StreamProtocolError(f"Frame #{frame_num}: missing 'ok' field")
        if not isinstance(frame["ok"], bool):
            raise StreamProtocolError(
                f"Frame #{frame_num}: 'ok' must be boolean, got {type(frame['ok']).__name__}"
            )

        # Success envelope (streaming frames only in this validator)
        if frame["ok"] is True:
            # SCHEMA.md streaming frames use code=STREAMING and chunk (no 'result')
            if frame.get("code") != FrameValidator.STREAMING_CODE:
                raise StreamProtocolError(
                    f"Frame #{frame_num}: streaming success code must be {FrameValidator.STREAMING_CODE!r}, "
                    f"got {frame.get('code')!r}"
                )

            if "ms" not in frame or not isinstance(frame.get("ms"), (int, float)) or frame["ms"] < 0:
                raise StreamProtocolError(f"Frame #{frame_num}: 'ms' must be non-negative number")

            if "chunk" not in frame:
                raise StreamProtocolError(f"Frame #{frame_num}: streaming frame missing 'chunk' field")

            if "result" in frame:
                raise StreamProtocolError(f"Frame #{frame_num}: streaming frame must NOT contain 'result'")

        # Error envelope
        else:
            required_fields = {"code", "error", "message", "retry_after_ms", "details", "ms"}
            missing = [f for f in sorted(required_fields) if f not in frame]
            if missing:
                raise StreamProtocolError(
                    f"Frame #{frame_num}: error envelope missing field(s): {', '.join(missing)}"
                )

            if not isinstance(frame.get("ms"), (int, float)) or frame["ms"] < 0:
                raise StreamProtocolError(f"Frame #{frame_num}: 'ms' must be non-negative number")

            # SCHEMA.md: retry_after_ms is integer or null; details is object or null (schema enforces)
            # Here we just ensure keys exist; stricter typing is deferred to schema validation.

    @staticmethod
    def should_validate_frame(frame_num: int, mode: ValidationMode, sample_rate: float) -> bool:
        if mode in (ValidationMode.STRICT, ValidationMode.COLLECT_ERRORS):
            return True
        if mode == ValidationMode.SAMPLED:
            if frame_num == 1:
                return True
            # Deterministic sampling (stable across runs)
            return (hash(str(frame_num)) % 100) < int(sample_rate * 100)
        return False  # LAZY


class StreamValidationEngine:
    """SCHEMA.md-aligned stream validation engine."""

    def __init__(self, config: ValidationConfig):
        self.config = config
        self._parsers: dict[StreamFormat, StreamParser] = {
            StreamFormat.NDJSON: NDJSONParser(),
            StreamFormat.SSE: SSEParser(),
        }

    def _emit_performance_warning(self, message: str) -> None:
        if self.config.on_performance_warning:
            self.config.on_performance_warning(message)
        else:
            warnings.warn(message, UserWarning)

    def _emit_validation_error(self, error: ValidationError) -> None:
        if self.config.on_validation_error:
            self.config.on_validation_error(error)

    def _emit_frame_validated(self, frame: dict[str, Any], frame_num: int) -> None:
        if self.config.on_frame_validated:
            self.config.on_frame_validated(frame, frame_num)

    def _handle_validation_error(
        self,
        error: Exception,
        frame_num: int,
        error_type: str,
        collect_errors: bool,
        validation_errors: list[ValidationError],
    ) -> None:
        v = ValidationError(frame_num, error_type, str(error), error)
        if collect_errors:
            validation_errors.append(v)
            self._emit_validation_error(v)
        else:
            raise v.exception if v.exception else error

    def validate_frames(
        self,
        frames: Iterable[dict[str, Any]],
        format: StreamFormat = StreamFormat.RAW_JSON,
    ) -> StreamValidationReport:
        start_time = time.time()

        total = 0
        data_count = 0
        terminal_seen = False
        terminal_frame_position = 0
        ended_ok = False
        errored = False
        total_bytes = 0
        max_frame_size = 0
        frames_validated = 0
        frames_skipped = 0
        validation_errors: list[ValidationError] = []
        collect_errors = self.config.mode == ValidationMode.COLLECT_ERRORS

        for frame_num, frame in enumerate(frames, 1):
            total = frame_num

            # Always size-check
            try:
                frame_size = FrameValidator.estimate_frame_size(frame)
                total_bytes += frame_size
                max_frame_size = max(max_frame_size, frame_size)
                FrameValidator.check_frame_size(frame, frame_num, self.config.max_frame_bytes)
            except Exception as e:
                self._handle_validation_error(e, frame_num, "size_exceeded", collect_errors, validation_errors)
                if collect_errors:
                    continue
                raise

            # Always protocol envelope check (SCHEMA.md-aligned)
            try:
                FrameValidator.validate_protocol_envelope(frame, frame_num)
            except Exception as e:
                self._handle_validation_error(e, frame_num, "protocol_envelope", collect_errors, validation_errors)
                if collect_errors:
                    continue
                raise

            # Conditional schema validation
            should_validate = FrameValidator.should_validate_frame(frame_num, self.config.mode, self.config.sample_rate)
            if should_validate:
                frames_validated += 1
                try:
                    assert_valid(
                        self.config.stream_frame_schema_id,
                        frame,
                        context=f"{self.config.component}.stream frame #{frame_num}",
                        registry=self.config.schema_registry,
                    )
                    self._emit_frame_validated(frame, frame_num)
                except AssertionError as e:
                    self._handle_validation_error(e, frame_num, "schema_validation", collect_errors, validation_errors)
                    if collect_errors:
                        continue
                    raise
            else:
                frames_skipped += 1

            # Streaming semantics (SCHEMA.md §5.3 style):
            # - terminal is either chunk.is_final==true (success) OR error envelope
            try:
                if terminal_seen:
                    raise StreamProtocolError(
                        f"Data after terminal frame at frame #{frame_num} "
                        f"(terminal was at frame #{terminal_frame_position})"
                    )

                if frame.get("ok") is True:
                    # streaming data frame
                    data_count += 1
                    chunk = frame.get("chunk")
                    if isinstance(chunk, dict) and chunk.get("is_final") is True:
                        terminal_seen = True
                        ended_ok = True
                        terminal_frame_position = frame_num
                else:
                    # error terminates the stream
                    terminal_seen = True
                    errored = True
                    terminal_frame_position = frame_num

            except Exception as e:
                self._handle_validation_error(e, frame_num, "protocol_violation", collect_errors, validation_errors)
                if collect_errors:
                    continue
                raise

        validation_time_ms = (time.time() - start_time) * 1000

        # Final integrity check: must see terminal
        try:
            if not terminal_seen:
                raise StreamProtocolError(f"Stream completed without terminal frame. Processed {total} frames.")
        except Exception as e:
            self._handle_validation_error(e, total, "stream_integrity", collect_errors, validation_errors)

        # Performance warning
        if total > self.config.large_stream_threshold and validation_time_ms > self.config.performance_warning_threshold_ms:
            coverage = frames_validated / total if total > 0 else 0.0
            self._emit_performance_warning(
                f"Large stream validation took {validation_time_ms:.0f}ms for {total} frames "
                f"(mode={self.config.mode.value}, coverage={coverage:.1%})"
            )

        return StreamValidationReport(
            total_frames=total,
            data_frames=data_count,
            ended_ok=ended_ok,
            errored=errored,
            terminal_seen=terminal_seen,
            validation_time_ms=validation_time_ms,
            total_bytes=total_bytes,
            max_frame_bytes=max_frame_size,
            terminal_frame_position=terminal_frame_position,
            frames_validated=frames_validated,
            frames_skipped=frames_skipped,
            format=format,
            mode=self.config.mode,
            validation_errors=validation_errors,
        )

    async def validate_frames_async(
        self,
        frames: AsyncIterable[dict[str, Any]],
        format: StreamFormat = StreamFormat.RAW_JSON,
    ) -> StreamValidationReport:
        start_time = time.time()

        total = 0
        data_count = 0
        terminal_seen = False
        terminal_frame_position = 0
        ended_ok = False
        errored = False
        total_bytes = 0
        max_frame_size = 0
        frames_validated = 0
        frames_skipped = 0
        validation_errors: list[ValidationError] = []
        collect_errors = self.config.mode == ValidationMode.COLLECT_ERRORS

        async for frame in frames:
            total += 1
            frame_num = total

            try:
                frame_size = FrameValidator.estimate_frame_size(frame)
                total_bytes += frame_size
                max_frame_size = max(max_frame_size, frame_size)
                FrameValidator.check_frame_size(frame, frame_num, self.config.max_frame_bytes)
            except Exception as e:
                self._handle_validation_error(e, frame_num, "size_exceeded", collect_errors, validation_errors)
                if collect_errors:
                    continue
                raise

            try:
                FrameValidator.validate_protocol_envelope(frame, frame_num)
            except Exception as e:
                self._handle_validation_error(e, frame_num, "protocol_envelope", collect_errors, validation_errors)
                if collect_errors:
                    continue
                raise

            should_validate = FrameValidator.should_validate_frame(frame_num, self.config.mode, self.config.sample_rate)
            if should_validate:
                frames_validated += 1
                try:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None,
                        lambda f=frame, n=frame_num: assert_valid(
                            self.config.stream_frame_schema_id,
                            f,
                            context=f"{self.config.component}.stream frame #{n}",
                            registry=self.config.schema_registry,
                        ),
                    )
                    self._emit_frame_validated(frame, frame_num)
                except AssertionError as e:
                    self._handle_validation_error(e, frame_num, "schema_validation", collect_errors, validation_errors)
                    if collect_errors:
                        continue
                    raise
            else:
                frames_skipped += 1

            try:
                if terminal_seen:
                    raise StreamProtocolError(
                        f"Data after terminal frame at frame #{frame_num} "
                        f"(terminal was at frame #{terminal_frame_position})"
                    )

                if frame.get("ok") is True:
                    data_count += 1
                    chunk = frame.get("chunk")
                    if isinstance(chunk, dict) and chunk.get("is_final") is True:
                        terminal_seen = True
                        ended_ok = True
                        terminal_frame_position = frame_num
                else:
                    terminal_seen = True
                    errored = True
                    terminal_frame_position = frame_num

            except Exception as e:
                self._handle_validation_error(e, frame_num, "protocol_violation", collect_errors, validation_errors)
                if collect_errors:
                    continue
                raise

        validation_time_ms = (time.time() - start_time) * 1000

        try:
            if not terminal_seen:
                raise StreamProtocolError(f"Stream completed without terminal frame. Processed {total} frames.")
        except Exception as e:
            self._handle_validation_error(e, total, "stream_integrity", collect_errors, validation_errors)

        if total > self.config.large_stream_threshold and validation_time_ms > self.config.performance_warning_threshold_ms:
            coverage = frames_validated / total if total > 0 else 0.0
            self._emit_performance_warning(
                f"Large async stream validation took {validation_time_ms:.0f}ms for {total} frames "
                f"(mode={self.config.mode.value}, coverage={coverage:.1%})"
            )

        return StreamValidationReport(
            total_frames=total,
            data_frames=data_count,
            ended_ok=ended_ok,
            errored=errored,
            terminal_seen=terminal_seen,
            validation_time_ms=validation_time_ms,
            total_bytes=total_bytes,
            max_frame_bytes=max_frame_size,
            terminal_frame_position=terminal_frame_position,
            frames_validated=frames_validated,
            frames_skipped=frames_skipped,
            format=format,
            mode=self.config.mode,
            validation_errors=validation_errors,
        )

    def validate_ndjson(self, ndjson_text: str) -> StreamValidationReport:
        parser = self._parsers[StreamFormat.NDJSON]
        frames = parser.parse(ndjson_text)
        return self.validate_frames(frames, StreamFormat.NDJSON)

    def validate_sse(self, sse_text: str) -> StreamValidationReport:
        parser = self._parsers[StreamFormat.SSE]
        frames = parser.parse(sse_text)
        return self.validate_frames(frames, StreamFormat.SSE)

    async def validate_ndjson_async(self, ndjson_text: str) -> StreamValidationReport:
        parser = self._parsers[StreamFormat.NDJSON]
        frames = await parser.parse_async(ndjson_text)
        return await self.validate_frames_async(self._to_async_iterable(frames), StreamFormat.NDJSON)

    async def validate_sse_async(self, sse_text: str) -> StreamValidationReport:
        parser = self._parsers[StreamFormat.SSE]
        frames = await parser.parse_async(sse_text)
        return await self.validate_frames_async(self._to_async_iterable(frames), StreamFormat.SSE)

    async def validate_ndjson_streaming(self, lines: AsyncIterable[str]) -> StreamValidationReport:
        parser = self._parsers[StreamFormat.NDJSON]
        frames = parser.parse_streaming(lines)
        return await self.validate_frames_async(frames, StreamFormat.NDJSON)

    async def validate_sse_streaming(self, lines: AsyncIterable[str]) -> StreamValidationReport:
        parser = self._parsers[StreamFormat.SSE]
        frames = parser.parse_streaming(lines)
        return await self.validate_frames_async(frames, StreamFormat.SSE)

    @staticmethod
    async def _to_async_iterable(frames: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
        for frame in frames:
            yield frame


# Convenience functions

def validate_ndjson_stream(
    ndjson_text: str,
    stream_frame_schema_id: str,
    component: str,
    **kwargs: Any,
) -> StreamValidationReport:
    config = ValidationConfig(
        stream_frame_schema_id=stream_frame_schema_id,
        component=component,
        **kwargs,
    )
    return StreamValidationEngine(config).validate_ndjson(ndjson_text)


def validate_sse_stream(
    sse_text: str,
    stream_frame_schema_id: str,
    component: str,
    **kwargs: Any,
) -> StreamValidationReport:
    config = ValidationConfig(
        stream_frame_schema_id=stream_frame_schema_id,
        component=component,
        **kwargs,
    )
    return StreamValidationEngine(config).validate_sse(sse_text)


async def validate_ndjson_stream_async(
    ndjson_text: str,
    stream_frame_schema_id: str,
    component: str,
    **kwargs: Any,
) -> StreamValidationReport:
    config = ValidationConfig(
        stream_frame_schema_id=stream_frame_schema_id,
        component=component,
        **kwargs,
    )
    return await StreamValidationEngine(config).validate_ndjson_async(ndjson_text)


async def validate_sse_stream_async(
    sse_text: str,
    stream_frame_schema_id: str,
    component: str,
    **kwargs: Any,
) -> StreamValidationReport:
    config = ValidationConfig(
        stream_frame_schema_id=stream_frame_schema_id,
        component=component,
        **kwargs,
    )
    return await StreamValidationEngine(config).validate_sse_async(sse_text)
