# SPDX-License-Identifier: Apache-2.0
"""
Protocol-Compliant Stream Validator for Corpus Protocol Suite.

Validates that streaming operations follow protocol §2.4 envelope format and §2.7 streaming semantics.
This validator ensures ALL stream frames use the canonical {ok, code, ms, result/chunk} envelope structure.

Features:
- Validates protocol envelope structure on every frame
- Enforces streaming termination semantics
- Performance-optimized for production use
- Transport-agnostic (works with NDJSON, SSE, WebSocket)
- Comprehensive validation reporting
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
from typing import Any, Callable, Optional, Union

from .schema_registry import assert_valid, SchemaRegistry


class StreamFormat(Enum):
    """Supported stream formats."""
    NDJSON = "ndjson"
    SSE = "sse"
    RAW_JSON = "raw_json"


class ValidationMode(Enum):
    """Validation strictness modes."""
    STRICT = "strict"  # Validate every frame
    SAMPLED = "sampled"  # Validate sample of frames
    LAZY = "lazy"  # Validate only protocol rules, skip schema validation
    COLLECT_ERRORS = "collect_errors"  # Collect all errors instead of failing fast


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
        """Calculate derived metrics."""
        seconds = self.validation_time_ms / 1000.0
        
        # Bytes per second
        object.__setattr__(self, 'bytes_per_second', 
                          self.total_bytes / seconds if seconds > 0 else float('inf'))
        
        # Frames per second  
        object.__setattr__(self, 'frames_per_second',
                          self.total_frames / seconds if seconds > 0 else float('inf'))
        
        # Validation coverage
        object.__setattr__(self, 'validation_coverage',
                          self.frames_validated / self.total_frames if self.total_frames > 0 else 1.0)
    
    @property
    def is_valid(self) -> bool:
        """Check if stream is valid (no errors collected)."""
        return len(self.validation_errors) == 0
    
    @property
    def error_summary(self) -> str:
        """Get a summary of validation errors."""
        if not self.validation_errors:
            return "No validation errors"
        
        error_counts: dict[str, int] = {}
        for error in self.validation_errors:
            error_counts[error.error_type] = error_counts.get(error.error_type, 0) + 1
        
        summary_parts = [f"Found {len(self.validation_errors)} validation errors:"]
        for error_type, count in sorted(error_counts.items()):
            summary_parts.append(f"  - {error_type}: {count} error(s)")
        
        return "\n".join(summary_parts)


@dataclass(frozen=True)
class ValidationConfig:
    """Configuration for stream validation."""
    envelope_schema_id: str  # Schema for protocol envelope (e.g., llm.envelope.success.json)
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
        pass
    
    @abstractmethod
    async def parse_async(self, content: str) -> list[dict[str, Any]]:
        """Parse content asynchronously into protocol envelope frames."""
        pass
    
    @abstractmethod
    async def parse_streaming(self, lines: AsyncIterable[str]) -> AsyncIterator[dict[str, Any]]:
        """Parse content in true streaming fashion for unbounded streams."""
        pass


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
                if not isinstance(frame, dict):
                    raise StreamProtocolError(f"Frame #{i} must be a JSON object, got {type(frame).__name__}")
                
                # Extract protocol envelope - ignore transport-specific fields
                envelope = self._extract_protocol_envelope(frame, i)
                frames.append(envelope)
            except json.JSONDecodeError as e:
                raise StreamProtocolError(f"Invalid NDJSON at line #{i}: {e}") from e
        return frames
    
    def _extract_protocol_envelope(self, frame: dict[str, Any], frame_num: int) -> dict[str, Any]:
        """Extract protocol envelope from transport frame."""
        # For NDJSON, the frame should already be the protocol envelope
        # But handle cases where transport adds wrapper fields
        protocol_fields = {"ok", "code", "ms", "result", "chunk", "error", "message", "retry_after_ms", "details"}
        
        # If frame contains only protocol fields, return as-is
        if set(frame.keys()).issubset(protocol_fields):
            return frame
        
        # If frame has extra fields, extract protocol envelope
        envelope = {k: v for k, v in frame.items() if k in protocol_fields}
        if not envelope:
            raise StreamProtocolError(f"Frame #{frame_num} contains no protocol envelope fields")
        
        return envelope
    
    async def parse_async(self, content: str) -> list[dict[str, Any]]:
        """Async implementation that uses thread pool for CPU-bound work."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.parse, content)
    
    async def parse_streaming(self, lines: AsyncIterable[str]) -> AsyncIterator[dict[str, Any]]:
        """True streaming parse for unbounded NDJSON streams."""
        line_num = 0
        async for line in lines:
            line_num += 1
            line = line.strip()
            if not line:
                continue
            try:
                frame = json.loads(line)
                if not isinstance(frame, dict):
                    raise StreamProtocolError(f"Frame #{line_num} must be a JSON object, got {type(frame).__name__}")
                
                envelope = self._extract_protocol_envelope(frame, line_num)
                yield envelope
            except json.JSONDecodeError as e:
                raise StreamProtocolError(f"Invalid NDJSON at line #{line_num}: {e}") from e


class SSEParser(StreamParser):
    """Server-Sent Events (SSE) parser that extracts protocol envelopes."""
    
    def parse(self, sse_text: str) -> list[dict[str, Any]]:
        frames: list[dict[str, Any]] = []
        current_data_lines: list[str] = []

        def flush() -> None:
            nonlocal current_data_lines
            if not current_data_lines:
                return
                
            try:
                data_content = "\n".join(current_data_lines).strip()
                if data_content:
                    # Parse the protocol envelope from SSE data
                    envelope = json.loads(data_content)
                    if not isinstance(envelope, dict):
                        raise StreamProtocolError(f"SSE data must contain JSON object, got {type(envelope).__name__}")
                    frames.append(envelope)
            except json.JSONDecodeError as e:
                raise StreamProtocolError(f"Invalid JSON in SSE data: {e}") from e
            finally:
                current_data_lines = []

        for line_num, raw_line in enumerate(sse_text.splitlines(), 1):
            line = raw_line.rstrip('\n\r')
            
            if not line:
                flush()
                continue

            if line.startswith("data:"):
                current_data_lines.append(line[5:].lstrip())
            elif line.startswith("event:") or line.startswith("id:") or line.startswith("retry:") or line.startswith(":"):
                # Ignore event, id, retry, and comment lines - we only care about data
                continue
            else:
                # Treat unknown lines as data (per SSE spec)
                current_data_lines.append(line)

        flush()
        return frames
    
    async def parse_async(self, content: str) -> list[dict[str, Any]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.parse, content)
    
    async def parse_streaming(self, lines: AsyncIterable[str]) -> AsyncIterator[dict[str, Any]]:
        """True streaming parse for unbounded SSE streams."""
        current_data_lines: list[str] = []

        async def flush() -> AsyncIterator[dict[str, Any]]:
            nonlocal current_data_lines
            if not current_data_lines:
                return
                
            try:
                data_content = "\n".join(current_data_lines).strip()
                if data_content:
                    envelope = json.loads(data_content)
                    if not isinstance(envelope, dict):
                        raise StreamProtocolError(f"SSE data must contain JSON object, got {type(envelope).__name__}")
                    yield envelope
            except json.JSONDecodeError as e:
                raise StreamProtocolError(f"Invalid JSON in SSE data: {e}") from e
            finally:
                current_data_lines = []

        async for raw_line in lines:
            line = raw_line.rstrip('\n\r')
            
            if not line:
                async for frame in flush():
                    yield frame
                continue

            if line.startswith("data:"):
                current_data_lines.append(line[5:].lstrip())
            elif line.startswith("event:") or line.startswith("id:") or line.startswith("retry:") or line.startswith(":"):
                continue
            else:
                current_data_lines.append(line)

        async for frame in flush():
            yield frame


class FrameValidator:
    """Protocol envelope validation logic."""
    
    @staticmethod
    def estimate_frame_size(frame: dict[str, Any]) -> int:
        """Efficient frame size estimation."""
        return len(json.dumps(frame, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    
    @staticmethod
    def check_frame_size(frame: dict[str, Any], frame_num: int, max_frame_bytes: Optional[int]) -> None:
        """Check frame size against limits."""
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
        """Validate protocol §2.4 envelope structure."""
        # Check required success envelope fields
        if frame.get("ok") is True:
            if "code" not in frame:
                raise StreamProtocolError(f"Frame #{frame_num}: missing 'code' field")
            if frame.get("code") != "OK":
                raise StreamProtocolError(f"Frame #{frame_num}: code must be 'OK', got '{frame.get('code')}'")
            if "ms" not in frame:
                raise StreamProtocolError(f"Frame #{frame_num}: missing 'ms' field")
            if not (isinstance(frame.get("ms"), (int, float)) and frame["ms"] >= 0):
                raise StreamProtocolError(f"Frame #{frame_num}: 'ms' must be non-negative number")
            
            # Must have either result or chunk, not both
            has_result = "result" in frame
            has_chunk = "chunk" in frame
            if not (has_result or has_chunk):
                raise StreamProtocolError(f"Frame #{frame_num}: must have either 'result' or 'chunk' field")
            if has_result and has_chunk:
                raise StreamProtocolError(f"Frame #{frame_num}: cannot have both 'result' and 'chunk' fields")
        
        # Check error envelope fields
        elif frame.get("ok") is False:
            required_fields = {"code", "error", "message", "ms"}
            for field in required_fields:
                if field not in frame:
                    raise StreamProtocolError(f"Frame #{frame_num}: error envelope missing '{field}' field")
    
    @staticmethod
    def should_validate_frame(frame_num: int, mode: ValidationMode, sample_rate: float) -> bool:
        """Determine if frame should be validated based on mode and sampling."""
        if mode in (ValidationMode.STRICT, ValidationMode.COLLECT_ERRORS):
            return True
        elif mode == ValidationMode.SAMPLED:
            # Always validate first frame
            if frame_num == 1:
                return True
            # Sample based on rate (pseudo-random but deterministic)
            return (hash(str(frame_num)) % 100) < (sample_rate * 100)
        else:  # LAZY mode
            return False


class StreamValidationEngine:
    """Protocol-compliant stream validation engine."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self._parsers: dict[StreamFormat, StreamParser] = {
            StreamFormat.NDJSON: NDJSONParser(),
            StreamFormat.SSE: SSEParser(),
        }
    
    def _emit_performance_warning(self, message: str) -> None:
        """Emit performance warning through hook or warnings module."""
        if self.config.on_performance_warning:
            self.config.on_performance_warning(message)
        else:
            warnings.warn(message, UserWarning)
    
    def _emit_validation_error(self, error: ValidationError) -> None:
        """Emit validation error through hook."""
        if self.config.on_validation_error:
            self.config.on_validation_error(error)
    
    def _emit_frame_validated(self, frame: dict[str, Any], frame_num: int) -> None:
        """Emit frame validated event through hook."""
        if self.config.on_frame_validated:
            self.config.on_frame_validated(frame, frame_num)
    
    def _handle_validation_error(
        self, 
        error: Exception, 
        frame_num: int, 
        error_type: str,
        collect_errors: bool,
        validation_errors: list[ValidationError]
    ) -> None:
        """Handle validation error based on collection mode."""
        validation_error = ValidationError(frame_num, error_type, str(error), error)
        
        if collect_errors:
            validation_errors.append(validation_error)
            self._emit_validation_error(validation_error)
        else:
            raise validation_error.exception if validation_error.exception else error

    def validate_frames(
        self, 
        frames: Iterable[dict[str, Any]],
        format: StreamFormat = StreamFormat.RAW_JSON
    ) -> StreamValidationReport:
        """Validate protocol envelope frames synchronously."""
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

            # Size validation (always performed)
            try:
                frame_size = FrameValidator.estimate_frame_size(frame)
                total_bytes += frame_size
                max_frame_size = max(max_frame_size, frame_size)
                FrameValidator.check_frame_size(frame, frame_num, self.config.max_frame_bytes)
            except (FrameSizeExceededError, StreamProtocolError) as e:
                self._handle_validation_error(e, frame_num, "size_exceeded", collect_errors, validation_errors)
                if collect_errors:
                    continue
                else:
                    raise

            # Protocol envelope validation (always performed)
            try:
                FrameValidator.validate_protocol_envelope(frame, frame_num)
            except StreamProtocolError as e:
                self._handle_validation_error(e, frame_num, "protocol_envelope", collect_errors, validation_errors)
                if collect_errors:
                    continue
                else:
                    raise

            # Schema validation (conditional)
            should_validate = FrameValidator.should_validate_frame(
                frame_num, self.config.mode, self.config.sample_rate
            )
            
            if should_validate:
                frames_validated += 1
                try:
                    assert_valid(
                        self.config.envelope_schema_id, 
                        frame, 
                        context=f"{self.config.component}.stream frame #{frame_num}",
                        registry=self.config.schema_registry
                    )
                    self._emit_frame_validated(frame, frame_num)
                except AssertionError as e:
                    self._handle_validation_error(e, frame_num, "schema_validation", collect_errors, validation_errors)
                    if collect_errors:
                        continue
                    else:
                        raise
            else:
                frames_skipped += 1

            # Protocol streaming semantics validation (always performed)
            try:
                if terminal_seen:
                    raise StreamProtocolError(
                        f"Data after terminal frame at frame #{frame_num} "
                        f"(terminal was at frame #{terminal_frame_position})"
                    )

                # Check if this is a data frame
                if frame.get("ok") is True and "chunk" in frame:
                    data_count += 1
                    
                    # Check if this is a terminal success frame
                    chunk = frame["chunk"]
                    if chunk.get("is_final") is True:
                        terminal_seen = True
                        ended_ok = True
                        terminal_frame_position = frame_num
                
                # Check if this is a terminal error frame
                elif frame.get("ok") is False:
                    terminal_seen = True
                    errored = True
                    terminal_frame_position = frame_num
                    
            except StreamProtocolError as e:
                self._handle_validation_error(e, frame_num, "protocol_violation", collect_errors, validation_errors)
                if collect_errors:
                    continue
                else:
                    raise

        validation_time_ms = (time.time() - start_time) * 1000

        # Final stream integrity checks
        try:
            if not terminal_seen:
                raise StreamProtocolError(
                    f"Stream completed without terminal frame. Processed {total} frames."
                )
        except StreamProtocolError as e:
            self._handle_validation_error(e, total, "stream_integrity", collect_errors, validation_errors)

        # Performance warnings
        if (total > self.config.large_stream_threshold and 
            validation_time_ms > self.config.performance_warning_threshold_ms):
            self._emit_performance_warning(
                f"Large stream validation took {validation_time_ms:.0f}ms for {total} frames "
                f"(mode={self.config.mode.value}, coverage={frames_validated/total:.1%})"
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
        format: StreamFormat = StreamFormat.RAW_JSON
    ) -> StreamValidationReport:
        """Validate protocol envelope frames asynchronously."""
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

            # Size validation
            try:
                frame_size = FrameValidator.estimate_frame_size(frame)
                total_bytes += frame_size
                max_frame_size = max(max_frame_size, frame_size)
                FrameValidator.check_frame_size(frame, frame_num, self.config.max_frame_bytes)
            except (FrameSizeExceededError, StreamProtocolError) as e:
                self._handle_validation_error(e, frame_num, "size_exceeded", collect_errors, validation_errors)
                if collect_errors:
                    continue
                else:
                    raise

            # Protocol envelope validation
            try:
                FrameValidator.validate_protocol_envelope(frame, frame_num)
            except StreamProtocolError as e:
                self._handle_validation_error(e, frame_num, "protocol_envelope", collect_errors, validation_errors)
                if collect_errors:
                    continue
                else:
                    raise

            # Schema validation
            should_validate = FrameValidator.should_validate_frame(
                frame_num, self.config.mode, self.config.sample_rate
            )
            
            if should_validate:
                frames_validated += 1
                try:
                    # Run schema validation in thread pool since it's likely CPU-bound
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None, 
                        lambda f=frame, n=frame_num: assert_valid(
                            self.config.envelope_schema_id,
                            f,
                            context=f"{self.config.component}.stream frame #{n}",
                            registry=self.config.schema_registry
                        )
                    )
                    self._emit_frame_validated(frame, frame_num)
                except AssertionError as e:
                    self._handle_validation_error(e, frame_num, "schema_validation", collect_errors, validation_errors)
                    if collect_errors:
                        continue
                    else:
                        raise
            else:
                frames_skipped += 1

            # Protocol streaming semantics validation
            try:
                if terminal_seen:
                    raise StreamProtocolError(
                        f"Data after terminal frame at frame #{frame_num} "
                        f"(terminal was at frame #{terminal_frame_position})"
                    )

                if frame.get("ok") is True and "chunk" in frame:
                    data_count += 1
                    
                    chunk = frame["chunk"]
                    if chunk.get("is_final") is True:
                        terminal_seen = True
                        ended_ok = True
                        terminal_frame_position = frame_num
                
                elif frame.get("ok") is False:
                    terminal_seen = True
                    errored = True
                    terminal_frame_position = frame_num
                    
            except StreamProtocolError as e:
                self._handle_validation_error(e, frame_num, "protocol_violation", collect_errors, validation_errors)
                if collect_errors:
                    continue
                else:
                    raise

        validation_time_ms = (time.time() - start_time) * 1000

        # Final stream integrity checks
        try:
            if not terminal_seen:
                raise StreamProtocolError(f"Stream completed without terminal frame. Processed {total} frames.")
        except StreamProtocolError as e:
            self._handle_validation_error(e, total, "stream_integrity", collect_errors, validation_errors)

        # Performance warnings
        if (total > self.config.large_stream_threshold and 
            validation_time_ms > self.config.performance_warning_threshold_ms):
            self._emit_performance_warning(
                f"Large async stream validation took {validation_time_ms:.0f}ms for {total} frames "
                f"(mode={self.config.mode.value}, coverage={frames_validated/total:.1%})"
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
        """Validate NDJSON text containing protocol envelopes."""
        parser = self._parsers[StreamFormat.NDJSON]
        frames = parser.parse(ndjson_text)
        return self.validate_frames(frames, StreamFormat.NDJSON)
    
    def validate_sse(self, sse_text: str) -> StreamValidationReport:
        """Validate SSE text containing protocol envelopes."""
        parser = self._parsers[StreamFormat.SSE]
        frames = parser.parse(sse_text)
        return self.validate_frames(frames, StreamFormat.SSE)
    
    async def validate_ndjson_async(self, ndjson_text: str) -> StreamValidationReport:
        """Validate NDJSON text asynchronously."""
        parser = self._parsers[StreamFormat.NDJSON]
        frames = await parser.parse_async(ndjson_text)
        return await self.validate_frames_async(self._to_async_iterable(frames), StreamFormat.NDJSON)
    
    async def validate_sse_async(self, sse_text: str) -> StreamValidationReport:
        """Validate SSE text asynchronously."""
        parser = self._parsers[StreamFormat.SSE]
        frames = await parser.parse_async(sse_text)
        return await self.validate_frames_async(self._to_async_iterable(frames), StreamFormat.SSE)
    
    async def validate_ndjson_streaming(
        self, 
        lines: AsyncIterable[str]
    ) -> StreamValidationReport:
        """Validate NDJSON in true streaming fashion for unbounded streams."""
        parser = self._parsers[StreamFormat.NDJSON]
        frames = parser.parse_streaming(lines)
        return await self.validate_frames_async(frames, StreamFormat.NDJSON)
    
    async def validate_sse_streaming(
        self, 
        lines: AsyncIterable[str]
    ) -> StreamValidationReport:
        """Validate SSE in true streaming fashion for unbounded streams."""
        parser = self._parsers[StreamFormat.SSE]
        frames = parser.parse_streaming(lines)
        return await self.validate_frames_async(frames, StreamFormat.SSE)
    
    @staticmethod
    async def _to_async_iterable(frames: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
        """Convert list to async iterable."""
        for frame in frames:
            yield frame


# Convenience functions for simple use cases

def validate_ndjson_stream(
    ndjson_text: str,
    envelope_schema_id: str,
    component: str,
    **kwargs: Any
) -> StreamValidationReport:
    """Convenience function for simple NDJSON validation."""
    config = ValidationConfig(envelope_schema_id=envelope_schema_id, component=component, **kwargs)
    engine = StreamValidationEngine(config)
    return engine.validate_ndjson(ndjson_text)


def validate_sse_stream(
    sse_text: str,
    envelope_schema_id: str,
    component: str,
    **kwargs: Any
) -> StreamValidationReport:
    """Convenience function for simple SSE validation."""
    config = ValidationConfig(envelope_schema_id=envelope_schema_id, component=component, **kwargs)
    engine = StreamValidationEngine(config)
    return engine.validate_sse(sse_text)


async def validate_ndjson_stream_async(
    ndjson_text: str,
    envelope_schema_id: str,
    component: str,
    **kwargs: Any
) -> StreamValidationReport:
    """Convenience function for simple async NDJSON validation."""
    config = ValidationConfig(envelope_schema_id=envelope_schema_id, component=component, **kwargs)
    engine = StreamValidationEngine(config)
    return await engine.validate_ndjson_async(ndjson_text)


async def validate_sse_stream_async(
    sse_text: str,
    envelope_schema_id: str,
    component: str,
    **kwargs: Any
) -> StreamValidationReport:
    """Convenience function for simple async SSE validation."""
    config = ValidationConfig(envelope_schema_id=envelope_schema_id, component=component, **kwargs)
    engine = StreamValidationEngine(config)
    return await engine.validate_sse_async(sse_text)