# SPDX-License-Identifier: Apache-2.0
"""
Advanced Streaming Frames Validator

Validates NDJSON / SSE / WebSocket-like JSON frames against a union frame schema ($id),
and enforces protocol-level invariants with production-grade performance and reliability.

Features:
- Exactly one terminal frame (event=end OR event=error)
- No data frames after terminal
- Configurable max frame size enforcement
- Performance-optimized sampling for large streams
- Comprehensive validation reporting with metrics
- Streaming-friendly memory efficiency
- Async support for high-throughput applications
- True streaming parse for unbounded streams
- Telemetry hooks for observability integration
- Error collection mode for comprehensive validation
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


class TerminalType(Enum):
    """Terminal frame types."""
    END = "end"
    ERROR = "error"
    NONE = "none"


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
    union_schema_id: str
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


class StreamPerformanceWarning(UserWarning):
    """Warning for performance-related stream issues."""
    pass


class FrameSizeExceededError(StreamProtocolError):
    """Raised when frame exceeds size limits."""
    pass


class StreamParser(ABC):
    """Abstract base class for stream parsers."""
    
    @abstractmethod
    def parse(self, content: str) -> list[dict[str, Any]]:
        """Parse content into frames."""
        pass
    
    @abstractmethod
    async def parse_async(self, content: str) -> list[dict[str, Any]]:
        """Parse content asynchronously into frames."""
        pass
    
    @abstractmethod
    async def parse_streaming(self, lines: AsyncIterable[str]) -> AsyncIterator[dict[str, Any]]:
        """Parse content in true streaming fashion for unbounded streams."""
        pass


class NDJSONParser(StreamParser):
    """NDJSON stream parser."""
    
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
                frames.append(frame)
            except json.JSONDecodeError as e:
                raise StreamProtocolError(f"Invalid NDJSON at line #{i}: {e}") from e
        return frames
    
    async def parse_async(self, content: str) -> list[dict[str, Any]]:
        """Async implementation that uses thread pool for CPU-bound work."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.parse, content)
    
    async def parse_streaming(self, lines: AsyncIterable[str]) -> AsyncIterator[dict[str, Any]]:
        """
        True streaming parse for unbounded NDJSON streams.
        
        Example:
            async for line in websocket_stream:
                # Lines are processed as they arrive
                pass
        """
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
                yield frame
            except json.JSONDecodeError as e:
                raise StreamProtocolError(f"Invalid NDJSON at line #{line_num}: {e}") from e


class SSEParser(StreamParser):
    """Server-Sent Events (SSE) stream parser."""
    
    def parse(self, sse_text: str) -> list[dict[str, Any]]:
        frames: list[dict[str, Any]] = []
        current_event: Optional[str] = None
        current_data_lines: list[str] = []
        current_id: Optional[str] = None

        def flush() -> None:
            nonlocal current_event, current_data_lines, current_id
            if current_event is None:
                return
                
            try:
                data_content = "\n".join(current_data_lines).strip()
                payload: dict[str, Any] = {}
                
                # Parse data as JSON if present
                if data_content:
                    try:
                        payload = json.loads(data_content)
                    except json.JSONDecodeError:
                        # If not valid JSON, treat as plain text
                        payload = {"message": data_content}
                
                # Construct frame according to SSE semantics
                frame: dict[str, Any] = {"event": current_event}
                if current_event == "data":
                    frame["data"] = payload
                else:
                    frame.update(payload)
                    
                if current_id:
                    frame["id"] = current_id
                    
                frames.append(frame)
            except Exception as e:
                raise StreamProtocolError(f"Invalid SSE data for event '{current_event}': {e}") from e
            finally:
                current_event, current_data_lines, current_id = None, [], None

        for line_num, raw_line in enumerate(sse_text.splitlines(), 1):
            line = raw_line.rstrip('\n\r')
            
            if not line:
                flush()
                continue

            if line.startswith("event:"):
                flush()
                event_value = line[6:].strip()
                # SSE spec: empty event defaults to "message"
                current_event = event_value if event_value else "message"
            elif line.startswith("data:"):
                current_data_lines.append(line[5:].lstrip())
            elif line.startswith("id:"):
                current_id = line[3:].strip()
            elif line.startswith("retry:") or line.startswith(":"):
                # Ignore retry and comment lines
                continue
            else:
                # Unknown line format - append to data
                current_data_lines.append(line)

        flush()
        return frames
    
    async def parse_async(self, content: str) -> list[dict[str, Any]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.parse, content)
    
    async def parse_streaming(self, lines: AsyncIterable[str]) -> AsyncIterator[dict[str, Any]]:
        """
        True streaming parse for unbounded SSE streams.
        
        Example:
            async for line in sse_stream:
                # SSE events are parsed and yielded as they complete
                pass
        """
        current_event: Optional[str] = None
        current_data_lines: list[str] = []
        current_id: Optional[str] = None

        async def flush() -> AsyncIterator[dict[str, Any]]:
            nonlocal current_event, current_data_lines, current_id
            if current_event is None:
                return
                
            try:
                data_content = "\n".join(current_data_lines).strip()
                payload: dict[str, Any] = {}
                
                if data_content:
                    try:
                        payload = json.loads(data_content)
                    except json.JSONDecodeError:
                        payload = {"message": data_content}
                
                frame: dict[str, Any] = {"event": current_event}
                if current_event == "data":
                    frame["data"] = payload
                else:
                    frame.update(payload)
                    
                if current_id:
                    frame["id"] = current_id
                    
                yield frame
            except Exception as e:
                raise StreamProtocolError(f"Invalid SSE data for event '{current_event}': {e}") from e
            finally:
                current_event, current_data_lines, current_id = None, [], None

        async for raw_line in lines:
            line = raw_line.rstrip('\n\r')
            
            if not line:
                async for frame in flush():
                    yield frame
                continue

            if line.startswith("event:"):
                async for frame in flush():
                    yield frame
                event_value = line[6:].strip()
                current_event = event_value if event_value else "message"
            elif line.startswith("data:"):
                current_data_lines.append(line[5:].lstrip())
            elif line.startswith("id:"):
                current_id = line[3:].strip()
            elif line.startswith("retry:") or line.startswith(":"):
                continue
            else:
                current_data_lines.append(line)

        async for frame in flush():
            yield frame


class FrameValidator:
    """Core frame validation logic."""
    
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
    def check_content_warnings(frame: dict[str, Any], frame_num: int) -> None:
        """Check for potentially problematic frame content."""
        if frame.get("event") == "data" and "data" in frame:
            data = frame["data"]
            if isinstance(data, str) and len(data) > 10_000:
                warnings.warn(
                    f"Large data frame #{frame_num}: {len(data)} characters",
                    StreamPerformanceWarning
                )
            elif isinstance(data, (dict, list)) and len(str(data)) > 50_000:
                warnings.warn(
                    f"Large structured data frame #{frame_num}: {len(str(data))} characters",
                    StreamPerformanceWarning
                )
    
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
    """Main stream validation engine with sync and async support."""
    
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
            warnings.warn(message, StreamPerformanceWarning)
    
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
        """Validate frames synchronously."""
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
            except FrameSizeExceededError as e:
                self._handle_validation_error(e, frame_num, "size_exceeded", collect_errors, validation_errors)
                if collect_errors:
                    continue
                else:
                    raise

            # Content warnings
            if self.config.enable_content_warnings:
                FrameValidator.check_content_warnings(frame, frame_num)

            # Schema validation (conditional)
            should_validate = FrameValidator.should_validate_frame(
                frame_num, self.config.mode, self.config.sample_rate
            )
            
            if should_validate:
                frames_validated += 1
                try:
                    assert_valid(
                        self.config.union_schema_id, 
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

            # Protocol validation (always performed)
            event = frame.get("event")
            
            try:
                if terminal_seen:
                    raise StreamProtocolError(
                        f"Data after terminal frame: saw '{event}' at frame #{frame_num} "
                        f"(terminal was at frame #{terminal_frame_position})"
                    )

                if event == "data":
                    data_count += 1
                elif event == "end":
                    terminal_seen = True
                    ended_ok = True
                    terminal_frame_position = frame_num
                elif event == "error":
                    terminal_seen = True
                    errored = True
                    terminal_frame_position = frame_num
                else:
                    raise StreamProtocolError(f"Unknown event type '{event}' at frame #{frame_num}")
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

            if ended_ok and errored:
                raise StreamProtocolError("Stream contained both 'end' and 'error' terminal frames")
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
        """Validate frames asynchronously."""
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
            except FrameSizeExceededError as e:
                self._handle_validation_error(e, frame_num, "size_exceeded", collect_errors, validation_errors)
                if collect_errors:
                    continue
                else:
                    raise

            # Content warnings
            if self.config.enable_content_warnings:
                FrameValidator.check_content_warnings(frame, frame_num)

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
                            self.config.union_schema_id,
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

            # Protocol validation
            event = frame.get("event")
            
            try:
                if terminal_seen:
                    raise StreamProtocolError(
                        f"Data after terminal frame: saw '{event}' at frame #{frame_num} "
                        f"(terminal was at frame #{terminal_frame_position})"
                    )

                if event == "data":
                    data_count += 1
                elif event == "end":
                    terminal_seen = True
                    ended_ok = True
                    terminal_frame_position = frame_num
                elif event == "error":
                    terminal_seen = True
                    errored = True
                    terminal_frame_position = frame_num
                else:
                    raise StreamProtocolError(f"Unknown event type '{event}' at frame #{frame_num}")
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

            if ended_ok and errored:
                raise StreamProtocolError("Stream contained both 'end' and 'error' terminal frames")
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
        """Validate NDJSON text."""
        parser = self._parsers[StreamFormat.NDJSON]
        frames = parser.parse(ndjson_text)
        return self.validate_frames(frames, StreamFormat.NDJSON)
    
    def validate_sse(self, sse_text: str) -> StreamValidationReport:
        """Validate SSE text."""
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
        """
        Validate NDJSON in true streaming fashion for unbounded streams.
        
        Args:
            lines: Async iterable of NDJSON lines
            
        Returns:
            StreamValidationReport: Validation results
            
        Example:
            async def read_from_websocket():
                async with websockets.connect(...) as websocket:
                    async for message in websocket:
                        yield message
            
            report = await engine.validate_ndjson_streaming(read_from_websocket())
        """
        parser = self._parsers[StreamFormat.NDJSON]
        frames = parser.parse_streaming(lines)
        return await self.validate_frames_async(frames, StreamFormat.NDJSON)
    
    async def validate_sse_streaming(
        self, 
        lines: AsyncIterable[str]
    ) -> StreamValidationReport:
        """
        Validate SSE in true streaming fashion for unbounded streams.
        
        Args:
            lines: Async iterable of SSE lines
            
        Returns:
            StreamValidationReport: Validation results
            
        Example:
            async def read_sse_stream():
                async with aiohttp.ClientSession() as session:
                    async with session.get(...) as response:
                        async for line in response.content:
                            yield line.decode()
            
            report = await engine.validate_sse_streaming(read_sse_stream())
        """
        parser = self._parsers[StreamFormat.SSE]
        frames = parser.parse_streaming(lines)
        return await self.validate_frames_async(frames, StreamFormat.SSE)
    
    @staticmethod
    async def _to_async_iterable(frames: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
        """Convert list to async iterable."""
        for frame in frames:
            yield frame


# Factory functions for common use cases

def create_strict_validator(
    union_schema_id: str,
    component: str,
    max_frame_bytes: Optional[int] = None,
    **kwargs: Any
) -> StreamValidationEngine:
    """Create a strict validator for maximum reliability."""
    config = ValidationConfig(
        union_schema_id=union_schema_id,
        component=component,
        max_frame_bytes=max_frame_bytes,
        mode=ValidationMode.STRICT,
        enable_content_warnings=True,
        **kwargs
    )
    return StreamValidationEngine(config)


def create_performance_validator(
    union_schema_id: str,
    component: str,
    sample_rate: float = 0.1,
    **kwargs: Any
) -> StreamValidationEngine:
    """Create a performance-optimized validator for high-throughput scenarios."""
    config = ValidationConfig(
        union_schema_id=union_schema_id,
        component=component,
        max_frame_bytes=10_485_760,  # 10MB for large streams
        mode=ValidationMode.SAMPLED,
        sample_rate=sample_rate,
        enable_content_warnings=False,
        performance_warning_threshold_ms=5000,
        **kwargs
    )
    return StreamValidationEngine(config)


def create_lazy_validator(
    union_schema_id: str,
    component: str,
    **kwargs: Any
) -> StreamValidationEngine:
    """Create a lazy validator for protocol-only validation."""
    config = ValidationConfig(
        union_schema_id=union_schema_id,
        component=component,
        mode=ValidationMode.LAZY,
        enable_content_warnings=False,
        **kwargs
    )
    return StreamValidationEngine(config)


def create_error_collecting_validator(
    union_schema_id: str,
    component: str,
    **kwargs: Any
) -> StreamValidationEngine:
    """Create a validator that collects all errors instead of failing fast."""
    config = ValidationConfig(
        union_schema_id=union_schema_id,
        component=component,
        mode=ValidationMode.COLLECT_ERRORS,
        enable_content_warnings=True,
        **kwargs
    )
    return StreamValidationEngine(config)


# Convenience functions for simple use cases

def validate_ndjson_stream(
    ndjson_text: str,
    union_schema_id: str,
    component: str,
    **kwargs: Any
) -> StreamValidationReport:
    """Convenience function for simple NDJSON validation."""
    config = ValidationConfig(union_schema_id=union_schema_id, component=component, **kwargs)
    engine = StreamValidationEngine(config)
    return engine.validate_ndjson(ndjson_text)


def validate_sse_stream(
    sse_text: str,
    union_schema_id: str,
    component: str,
    **kwargs: Any
) -> StreamValidationReport:
    """Convenience function for simple SSE validation."""
    config = ValidationConfig(union_schema_id=union_schema_id, component=component, **kwargs)
    engine = StreamValidationEngine(config)
    return engine.validate_sse(sse_text)


async def validate_ndjson_stream_async(
    ndjson_text: str,
    union_schema_id: str,
    component: str,
    **kwargs: Any
) -> StreamValidationReport:
    """Convenience function for simple async NDJSON validation."""
    config = ValidationConfig(union_schema_id=union_schema_id, component=component, **kwargs)
    engine = StreamValidationEngine(config)
    return await engine.validate_ndjson_async(ndjson_text)


async def validate_sse_stream_async(
    sse_text: str,
    union_schema_id: str,
    component: str,
    **kwargs: Any
) -> StreamValidationReport:
    """Convenience function for simple async SSE validation."""
    config = ValidationConfig(union_schema_id=union_schema_id, component=component, **kwargs)
    engine = StreamValidationEngine(config)
    return await engine.validate_sse_async(sse_text)


async def validate_ndjson_stream_streaming(
    lines: AsyncIterable[str],
    union_schema_id: str,
    component: str,
    **kwargs: Any
) -> StreamValidationReport:
    """Convenience function for true streaming NDJSON validation."""
    config = ValidationConfig(union_schema_id=union_schema_id, component=component, **kwargs)
    engine = StreamValidationEngine(config)
    return await engine.validate_ndjson_streaming(lines)


async def validate_sse_stream_streaming(
    lines: AsyncIterable[str],
    union_schema_id: str,
    component: str,
    **kwargs: Any
) -> StreamValidationReport:
    """Convenience function for true streaming SSE validation."""
    config = ValidationConfig(union_schema_id=union_schema_id, component=component, **kwargs)
    engine = StreamValidationEngine(config)
    return await engine.validate_sse_streaming(lines)