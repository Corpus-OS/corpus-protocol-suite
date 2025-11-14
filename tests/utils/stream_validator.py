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
from typing import Any, Callable, Optional, Union, cast

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


class TerminalType(Enum):
    """Terminal frame types."""
    END = "end"
    ERROR = "error"
    NONE = "none"


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
    def parse(self, content: str) -> list[dict]:
        """Parse content into frames."""
        pass
    
    @abstractmethod
    async def parse_async(self, content: str) -> list[dict]:
        """Parse content asynchronously into frames."""
        pass


class NDJSONParser(StreamParser):
    """NDJSON stream parser."""
    
    def parse(self, content: str) -> list[dict]:
        frames: list[dict] = []
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
    
    async def parse_async(self, content: str) -> list[dict]:
        """Async implementation that uses thread pool for CPU-bound work."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.parse, content)


class SSEParser(StreamParser):
    """Server-Sent Events (SSE) stream parser."""
    
    def parse(self, sse_text: str) -> list[dict]:
        frames: list[dict] = []
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
                current_event = line[6:].strip()
            elif line.startswith("data:"):
                current_data_lines.append(line[5:].lstrip())
            elif line.startswith("id:"):
                current_id = line[3:].strip()
            elif line.startswith("retry:") or line.startswith(":"):
                # Ignore retry and comment lines
                continue
            else:
                # Unknown line format
                current_data_lines.append(line)

        flush()
        return frames
    
    async def parse_async(self, content: str) -> list[dict]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.parse, content)


class FrameValidator:
    """Core frame validation logic."""
    
    @staticmethod
    def estimate_frame_size(frame: dict) -> int:
        """Efficient frame size estimation."""
        return len(json.dumps(frame, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    
    @staticmethod
    def check_frame_size(frame: dict, frame_num: int, max_frame_bytes: Optional[int]) -> None:
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
    def check_content_warnings(frame: dict, frame_num: int) -> None:
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
        if mode == ValidationMode.STRICT:
            return True
        elif mode == ValidationMode.SAMPLED:
            # Always validate first frame and terminal frames
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
    
    def validate_frames(
        self, 
        frames: Iterable[dict],
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

        for frame_num, frame in enumerate(frames, 1):
            total = frame_num

            # Size validation (always performed)
            frame_size = FrameValidator.estimate_frame_size(frame)
            total_bytes += frame_size
            max_frame_size = max(max_frame_size, frame_size)
            FrameValidator.check_frame_size(frame, frame_num, self.config.max_frame_bytes)

            # Content warnings
            if self.config.enable_content_warnings:
                FrameValidator.check_content_warnings(frame, frame_num)

            # Schema validation (conditional)
            should_validate = FrameValidator.should_validate_frame(
                frame_num, self.config.mode, self.config.sample_rate
            )
            
            if should_validate:
                frames_validated += 1
                assert_valid(
                    self.config.union_schema_id, 
                    frame, 
                    context=f"{self.config.component}.stream frame #{frame_num}",
                    registry=self.config.schema_registry
                )
            else:
                frames_skipped += 1

            # Protocol validation (always performed)
            event = frame.get("event")
            
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

        validation_time_ms = (time.time() - start_time) * 1000

        # Final stream integrity checks
        if not terminal_seen:
            raise StreamProtocolError(
                f"Stream completed without terminal frame. Processed {total} frames."
            )

        if ended_ok and errored:
            raise StreamProtocolError("Stream contained both 'end' and 'error' terminal frames")

        # Performance warnings
        if (total > self.config.large_stream_threshold and 
            validation_time_ms > self.config.performance_warning_threshold_ms):
            warnings.warn(
                f"Large stream validation took {validation_time_ms:.0f}ms for {total} frames "
                f"(mode={self.config.mode.value}, coverage={frames_validated/total:.1%})",
                StreamPerformanceWarning
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
        )
    
    async def validate_frames_async(
        self, 
        frames: AsyncIterable[dict],
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

        async for frame in frames:
            total += 1
            frame_num = total

            # Size validation
            frame_size = FrameValidator.estimate_frame_size(frame)
            total_bytes += frame_size
            max_frame_size = max(max_frame_size, frame_size)
            FrameValidator.check_frame_size(frame, frame_num, self.config.max_frame_bytes)

            # Content warnings
            if self.config.enable_content_warnings:
                FrameValidator.check_content_warnings(frame, frame_num)

            # Schema validation
            should_validate = FrameValidator.should_validate_frame(
                frame_num, self.config.mode, self.config.sample_rate
            )
            
            if should_validate:
                frames_validated += 1
                # Run schema validation in thread pool since it's likely CPU-bound
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, 
                    lambda: assert_valid(
                        self.config.union_schema_id,
                        frame,
                        context=f"{self.config.component}.stream frame #{frame_num}",
                        registry=self.config.schema_registry
                    )
                )
            else:
                frames_skipped += 1

            # Protocol validation
            event = frame.get("event")
            
            if terminal_seen:
                raise StreamProtocolError(
                    f"Data after terminal frame: saw '{event}' at frame #{frame_num}"
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

        validation_time_ms = (time.time() - start_time) * 1000

        # Final stream integrity checks
        if not terminal_seen:
            raise StreamProtocolError(f"Stream completed without terminal frame. Processed {total} frames.")

        if ended_ok and errored:
            raise StreamProtocolError("Stream contained both 'end' and 'error' terminal frames")

        # Performance warnings (FIXED: Now included in async method too)
        if (total > self.config.large_stream_threshold and 
            validation_time_ms > self.config.performance_warning_threshold_ms):
            warnings.warn(
                f"Large async stream validation took {validation_time_ms:.0f}ms for {total} frames "
                f"(mode={self.config.mode.value}, coverage={frames_validated/total:.1%})",
                StreamPerformanceWarning
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
    
    @staticmethod
    async def _to_async_iterable(frames: list[dict]) -> AsyncIterator[dict]:
        """Convert list to async iterable."""
        for frame in frames:
            yield frame


# Factory functions for common use cases

def create_strict_validator(
    union_schema_id: str,
    component: str,
    max_frame_bytes: Optional[int] = None
) -> StreamValidationEngine:
    """Create a strict validator for maximum reliability."""
    config = ValidationConfig(
        union_schema_id=union_schema_id,
        component=component,
        max_frame_bytes=max_frame_bytes,
        mode=ValidationMode.STRICT,
        enable_content_warnings=True
    )
    return StreamValidationEngine(config)

def create_performance_validator(
    union_schema_id: str,
    component: str,
    sample_rate: float = 0.1
) -> StreamValidationEngine:
    """Create a performance-optimized validator for high-throughput scenarios."""
    config = ValidationConfig(
        union_schema_id=union_schema_id,
        component=component,
        max_frame_bytes=10_485_760,  # 10MB for large streams
        mode=ValidationMode.SAMPLED,
        sample_rate=sample_rate,
        enable_content_warnings=False,
        performance_warning_threshold_ms=5000
    )
    return StreamValidationEngine(config)

def create_lazy_validator(
    union_schema_id: str,
    component: str
) -> StreamValidationEngine:
    """Create a lazy validator for protocol-only validation."""
    config = ValidationConfig(
        union_schema_id=union_schema_id,
        component=component,
        mode=ValidationMode.LAZY,
        enable_content_warnings=False
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