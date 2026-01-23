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
- STRICT / SAMPLED / COLLECT_ERRORS: perform JSON Schema validation via schema_registry.assert_valid
- LAZY: enforce protocol invariants only (still SCHEMA.md-correct)

Transport:
- NDJSON / SSE supported.
- RAW_JSON is supported via validate_frames / validate_frames_async (already-parsed frames).
- Parsers preserve full frames (do NOT strip unknown keys).
- Parsers attach raw byte-lengths when available to avoid double-serialization for sizing.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import warnings
import zlib
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, AsyncIterator, Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from tests.utils.schema_registry import SchemaRegistry, assert_valid


class StreamFormat(Enum):
    """Supported stream formats."""

    NDJSON = "ndjson"
    SSE = "sse"
    RAW_JSON = "raw_json"


class ValidationMode(Enum):
    """Validation strictness modes."""

    STRICT = "strict"  # Validate every frame
    SAMPLED = "sampled"  # Validate deterministic sample of frames
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
        seconds = self.validation_time_ms / 1000.0
        object.__setattr__(
            self, "bytes_per_second", self.total_bytes / seconds if seconds > 0 else float("inf")
        )
        object.__setattr__(
            self, "frames_per_second", self.total_frames / seconds if seconds > 0 else float("inf")
        )
        object.__setattr__(
            self,
            "validation_coverage",
            self.frames_validated / self.total_frames if self.total_frames > 0 else 1.0,
        )

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
      OR an operation streaming success schema such as:
        https://corpusos.com/schemas/llm/llm.stream.success.json
        https://corpusos.com/schemas/graph/graph.stream_query.success.json
        https://corpusos.com/schemas/embedding/embedding.stream_embed.success.json

    NOTE:
    - If the stream terminates with an error envelope (ok=false), schema validation MUST
      validate that terminal frame against an error-envelope schema (component-specific
      if available, else common/envelope.error.json).
    """

    stream_frame_schema_id: str
    component: str
    max_frame_bytes: Optional[int] = 1_048_576  # 1 MiB
    mode: ValidationMode = ValidationMode.STRICT
    sample_rate: float = 0.1  # For SAMPLED mode (0.0..1.0)
    sampling_seed: str = "corpus"  # Deterministic across processes/hosts
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


@dataclass(frozen=True)
class FrameItem:
    """
    Frame plus optional raw byte-length metadata.

    For NDJSON/SSE, parsers can provide raw byte sizes without re-serializing dicts.
    """

    frame: dict[str, Any]
    raw_bytes: Optional[int] = None


class StreamParser(ABC):
    """Abstract base class for stream parsers."""

    @abstractmethod
    def parse(self, content: str) -> list[FrameItem]:
        raise NotImplementedError

    @abstractmethod
    async def parse_async(self, content: str) -> list[FrameItem]:
        raise NotImplementedError

    @abstractmethod
    async def parse_streaming(self, lines: AsyncIterable[str]) -> AsyncIterator[FrameItem]:
        raise NotImplementedError


class NDJSONParser(StreamParser):
    """NDJSON stream parser that yields protocol envelopes and raw byte sizes."""

    def parse(self, content: str) -> list[FrameItem]:
        items: list[FrameItem] = []
        for i, line in enumerate(content.splitlines(), 1):
            raw = line.strip()
            if not raw:
                continue

            raw_bytes = len(raw.encode("utf-8"))
            try:
                frame = json.loads(raw)
            except json.JSONDecodeError as e:
                raise StreamProtocolError(f"Invalid NDJSON at line #{i}: {e}") from e

            if not isinstance(frame, dict):
                raise StreamProtocolError(f"Frame #{i} must be a JSON object, got {type(frame).__name__}")

            items.append(FrameItem(frame=frame, raw_bytes=raw_bytes))

        return items

    async def parse_async(self, content: str) -> list[FrameItem]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.parse, content)

    async def parse_streaming(self, lines: AsyncIterable[str]) -> AsyncIterator[FrameItem]:
        line_num = 0
        async for line in lines:
            line_num += 1
            raw = line.strip()
            if not raw:
                continue

            raw_bytes = len(raw.encode("utf-8"))
            try:
                frame = json.loads(raw)
            except json.JSONDecodeError as e:
                raise StreamProtocolError(f"Invalid NDJSON at line #{line_num}: {e}") from e

            if not isinstance(frame, dict):
                raise StreamProtocolError(
                    f"Frame #{line_num} must be a JSON object, got {type(frame).__name__}"
                )

            yield FrameItem(frame=frame, raw_bytes=raw_bytes)


class SSEParser(StreamParser):
    """Server-Sent Events (SSE) parser that yields envelopes and raw byte sizes from 'data:' blocks."""

    def parse(self, sse_text: str) -> list[FrameItem]:
        items: list[FrameItem] = []
        current_data_lines: list[str] = []

        def flush() -> None:
            nonlocal current_data_lines
            if not current_data_lines:
                return
            data_content = "\n".join(current_data_lines).strip()
            current_data_lines = []
            if not data_content:
                return

            raw_bytes = len(data_content.encode("utf-8"))
            try:
                envelope = json.loads(data_content)
            except json.JSONDecodeError as e:
                raise StreamProtocolError(f"Invalid JSON in SSE data: {e}") from e

            if not isinstance(envelope, dict):
                raise StreamProtocolError(f"SSE data must contain JSON object, got {type(envelope).__name__}")

            items.append(FrameItem(frame=envelope, raw_bytes=raw_bytes))

        for _line_num, raw_line in enumerate(sse_text.splitlines(), 1):
            line = raw_line.rstrip("\n\r")
            if not line:
                flush()
                continue

            if line.startswith("data:"):
                current_data_lines.append(line[5:].lstrip())
            elif line.startswith(("event:", "id:", "retry:", ":")):
                continue
            else:
                current_data_lines.append(line)

        flush()
        return items

    async def parse_async(self, content: str) -> list[FrameItem]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.parse, content)

    async def parse_streaming(self, lines: AsyncIterable[str]) -> AsyncIterator[FrameItem]:
        current_data_lines: list[str] = []

        async def flush() -> AsyncIterator[FrameItem]:
            nonlocal current_data_lines
            if not current_data_lines:
                return
            data_content = "\n".join(current_data_lines).strip()
            current_data_lines = []
            if not data_content:
                return

            raw_bytes = len(data_content.encode("utf-8"))
            try:
                envelope = json.loads(data_content)
            except json.JSONDecodeError as e:
                raise StreamProtocolError(f"Invalid JSON in SSE data: {e}") from e

            if not isinstance(envelope, dict):
                raise StreamProtocolError(f"SSE data must contain JSON object, got {type(envelope).__name__}")

            yield FrameItem(frame=envelope, raw_bytes=raw_bytes)

        async for raw_line in lines:
            line = raw_line.rstrip("\n\r")

            if not line:
                async for item in flush():
                    yield item
                continue

            if line.startswith("data:"):
                current_data_lines.append(line[5:].lstrip())
            elif line.startswith(("event:", "id:", "retry:", ":")):
                continue
            else:
                current_data_lines.append(line)

        async for item in flush():
            yield item


class FrameValidator:
    """Protocol envelope validation logic (SCHEMA.md-aligned)."""

    STREAMING_CODE = "STREAMING"
    _ERROR_CODE_RE = re.compile(r"^[A-Z_]+$")

    # For conformance, treat envelopes as closed (additionalProperties: false).
    # In warnings-enabled mode, unexpected keys are warnings; otherwise they are errors.
    _STREAM_SUCCESS_KEYS = frozenset({"ok", "code", "ms", "chunk"})
    _ERROR_KEYS = frozenset({"ok", "code", "error", "message", "retry_after_ms", "details", "ms"})

    @staticmethod
    def estimate_frame_size(frame: dict[str, Any]) -> int:
        """
        Fallback size estimate when raw byte length is unavailable.
        This uses compact JSON serialization; for NDJSON/SSE we prefer parser-provided raw_bytes.
        """
        return len(json.dumps(frame, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))

    @staticmethod
    def check_frame_size(frame_bytes: int, frame_num: int, max_frame_bytes: Optional[int]) -> None:
        if max_frame_bytes is None:
            return
        if frame_bytes > max_frame_bytes:
            raise FrameSizeExceededError(
                f"Frame #{frame_num} exceeds max_frame_bytes={max_frame_bytes} "
                f"(got {frame_bytes} bytes, ~{frame_bytes/1024/1024:.1f}MB)"
            )

    @staticmethod
    def _warn_or_raise(extra_keys: set[str], frame_num: int, kind: str, *, enable_warnings: bool) -> None:
        """
        Outputs/contracts are closed by schema (additionalProperties: false).
        For conformance, extra keys are a violation. In production-style usage, warnings may be enabled.
        """
        if not extra_keys:
            return
        msg = f"Frame #{frame_num}: {kind} envelope contains unexpected key(s): {', '.join(sorted(extra_keys))}"
        if enable_warnings:
            warnings.warn(msg, UserWarning)
        else:
            raise StreamProtocolError(msg)

    @staticmethod
    def validate_protocol_envelope(
        frame: dict[str, Any], frame_num: int, *, enable_content_warnings: bool = True
    ) -> None:
        """
        Validate SCHEMA.md envelope invariants:

        - Streaming success: ok=true, code="STREAMING", ms>=0, chunk present, no extra keys (closed contract)
        - Error envelope: ok=false, required fields present, required types, ms>=0, no extra keys (closed contract)

        Note:
        - This function is used in all validation modes, including LAZY, to keep protocol-only validation
          SCHEMA.md-correct without requiring JSON Schema evaluation.
        """
        if "ok" not in frame:
            raise StreamProtocolError(f"Frame #{frame_num}: missing 'ok' field")
        if not isinstance(frame["ok"], bool):
            raise StreamProtocolError(
                f"Frame #{frame_num}: 'ok' must be boolean, got {type(frame['ok']).__name__}"
            )

        if frame["ok"] is True:
            extra = set(frame.keys()) - set(FrameValidator._STREAM_SUCCESS_KEYS)
            if extra:
                FrameValidator._warn_or_raise(
                    extra, frame_num, "streaming success", enable_warnings=enable_content_warnings
                )

            if frame.get("code") != FrameValidator.STREAMING_CODE:
                raise StreamProtocolError(
                    f"Frame #{frame_num}: streaming success code must be {FrameValidator.STREAMING_CODE!r}, "
                    f"got {frame.get('code')!r}"
                )

            ms = frame.get("ms")
            if not isinstance(ms, (int, float)) or float(ms) < 0.0:
                raise StreamProtocolError(f"Frame #{frame_num}: 'ms' must be non-negative number")

            if "chunk" not in frame:
                raise StreamProtocolError(f"Frame #{frame_num}: streaming frame missing 'chunk' field")

            chunk = frame.get("chunk")
            if not isinstance(chunk, dict):
                raise StreamProtocolError(
                    f"Frame #{frame_num}: 'chunk' must be an object, got {type(chunk).__name__}"
                )
            if "is_final" not in chunk:
                raise StreamProtocolError(f"Frame #{frame_num}: chunk missing required 'is_final' field")
            if not isinstance(chunk.get("is_final"), bool):
                raise StreamProtocolError(
                    f"Frame #{frame_num}: chunk.is_final must be boolean, got {type(chunk.get('is_final')).__name__}"
                )

        else:
            extra = set(frame.keys()) - set(FrameValidator._ERROR_KEYS)
            if extra:
                FrameValidator._warn_or_raise(extra, frame_num, "error", enable_warnings=enable_content_warnings)

            required_fields = FrameValidator._ERROR_KEYS - {"ok"}
            missing = [f for f in sorted(required_fields) if f not in frame]
            if missing:
                raise StreamProtocolError(f"Frame #{frame_num}: error envelope missing field(s): {', '.join(missing)}")

            code = frame.get("code")
            if not isinstance(code, str) or not code:
                raise StreamProtocolError(f"Frame #{frame_num}: 'code' must be non-empty string")
            if FrameValidator._ERROR_CODE_RE.match(code) is None:
                raise StreamProtocolError(f"Frame #{frame_num}: 'code' must match ^[A-Z_]+$, got {code!r}")

            err = frame.get("error")
            if not isinstance(err, str) or not err:
                raise StreamProtocolError(f"Frame #{frame_num}: 'error' must be non-empty string")

            msg = frame.get("message")
            if not isinstance(msg, str):
                raise StreamProtocolError(f"Frame #{frame_num}: 'message' must be string")

            retry_after_ms = frame.get("retry_after_ms")
            if retry_after_ms is not None:
                if not isinstance(retry_after_ms, int):
                    raise StreamProtocolError(
                        f"Frame #{frame_num}: 'retry_after_ms' must be integer|null, got {type(retry_after_ms).__name__}"
                    )
                if retry_after_ms < 0:
                    raise StreamProtocolError(f"Frame #{frame_num}: 'retry_after_ms' must be >= 0")

            details = frame.get("details")
            if details is not None and not isinstance(details, dict):
                raise StreamProtocolError(
                    f"Frame #{frame_num}: 'details' must be object|null, got {type(details).__name__}"
                )

            ms = frame.get("ms")
            if not isinstance(ms, (int, float)) or float(ms) < 0.0:
                raise StreamProtocolError(f"Frame #{frame_num}: 'ms' must be non-negative number")


def _deterministic_sample_hit(frame_num: int, sample_rate: float, seed: str) -> bool:
    """
    Deterministic sampling across processes/hosts.

    Uses crc32 over "<seed>:<frame_num>" for stable selection.
    """
    if sample_rate >= 1.0:
        return True
    if sample_rate <= 0.0:
        return False
    threshold = int(sample_rate * 10_000)  # 0..10000
    key = f"{seed}:{frame_num}".encode("utf-8")
    bucket = zlib.crc32(key) % 10_000
    return bucket < threshold


class _StreamState:
    """Shared state machine for streaming semantics."""

    def __init__(self) -> None:
        self.data_count = 0
        self.terminal_seen = False
        self.terminal_frame_position = 0
        self.ended_ok = False
        self.errored = False

    def apply_stream_semantics(self, frame: dict[str, Any], frame_num: int) -> None:
        if self.terminal_seen:
            raise StreamProtocolError(
                f"Data after terminal frame at frame #{frame_num} "
                f"(terminal was at frame #{self.terminal_frame_position})"
            )

        if frame.get("ok") is True:
            self.data_count += 1
            chunk = frame["chunk"]
            if chunk.get("is_final") is True:
                self.terminal_seen = True
                self.ended_ok = True
                self.terminal_frame_position = frame_num
        else:
            self.terminal_seen = True
            self.errored = True
            self.terminal_frame_position = frame_num


class StreamValidationEngine:
    """SCHEMA.md-aligned stream validation engine."""

    def __init__(self, config: ValidationConfig):
        self.config = config
        self._parsers: dict[StreamFormat, StreamParser] = {
            StreamFormat.NDJSON: NDJSONParser(),
            StreamFormat.SSE: SSEParser(),
        }
        self._cached_error_schema_id: Optional[str] = None

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

    def _should_schema_validate(self, frame_num: int) -> bool:
        if self.config.mode in (ValidationMode.STRICT, ValidationMode.COLLECT_ERRORS):
            return True
        if self.config.mode == ValidationMode.SAMPLED:
            if frame_num == 1:
                return True
            return _deterministic_sample_hit(frame_num, self.config.sample_rate, self.config.sampling_seed)
        return False  # LAZY

    def _resolve_error_schema_id(self) -> str:
        """
        Prefer component-specific envelope.error schema when available, else fall back to common.

        Tries (in order):
          1) https://corpusos.com/schemas/<component>/<component>.envelope.error.json   (if component provided)
          2) https://corpusos.com/schemas/common/envelope.error.json
        """
        if self._cached_error_schema_id is not None:
            return self._cached_error_schema_id

        candidates: list[str] = []
        comp = (self.config.component or "").strip()
        if comp:
            candidates.append(f"https://corpusos.com/schemas/{comp}/{comp}.envelope.error.json")
        candidates.append("https://corpusos.com/schemas/common/envelope.error.json")

        # Verify existence by attempting to validate an empty-ish frame would be noisy;
        # instead, we optimistically cache the first candidate and fall back on KeyError
        # at validation time.
        self._cached_error_schema_id = candidates[0]
        self._error_schema_candidates = candidates  # type: ignore[attr-defined]
        return self._cached_error_schema_id

    def _schema_id_for_frame(self, frame: dict[str, Any]) -> str:
        if frame.get("ok") is True:
            return self.config.stream_frame_schema_id
        return self._resolve_error_schema_id()

    def _schema_validate_sync(self, frame: dict[str, Any], frame_num: int) -> None:
        """
        Schema-validate a frame.

        - ok=true frames validate against config.stream_frame_schema_id
        - ok=false frames validate against a standard error-envelope schema
        """
        schema_id = self._schema_id_for_frame(frame)

        # If we picked a component-specific error schema that doesn't exist, fall back to common.
        if frame.get("ok") is False:
            candidates: list[str] = getattr(self, "_error_schema_candidates", [schema_id])  # type: ignore[attr-defined]
            last_keyerr: Optional[KeyError] = None
            for cand in candidates:
                try:
                    assert_valid(
                        cand,
                        frame,
                        context=f"{self.config.component}.stream frame #{frame_num}",
                        registry=self.config.schema_registry,
                    )
                    return
                except KeyError as e:
                    last_keyerr = e
                    continue
            if last_keyerr is not None:
                raise last_keyerr
            return

        assert_valid(
            schema_id,
            frame,
            context=f"{self.config.component}.stream frame #{frame_num}",
            registry=self.config.schema_registry,
        )

    async def _schema_validate_async(self, frame: dict[str, Any], frame_num: int) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self._schema_validate_sync(frame, frame_num))

    def _frame_bytes(self, item: FrameItem) -> int:
        return item.raw_bytes if item.raw_bytes is not None else FrameValidator.estimate_frame_size(item.frame)

    def validate_items(self, items: Iterable[FrameItem], format: StreamFormat) -> StreamValidationReport:
        start_time = time.time()

        total = 0
        total_bytes = 0
        max_frame_size = 0
        frames_validated = 0
        frames_skipped = 0
        validation_errors: list[ValidationError] = []
        collect_errors = self.config.mode == ValidationMode.COLLECT_ERRORS

        state = _StreamState()

        for frame_num, item in enumerate(items, 1):
            total = frame_num
            frame = item.frame

            # size accounting
            try:
                b = self._frame_bytes(item)
                total_bytes += b
                max_frame_size = max(max_frame_size, b)
                FrameValidator.check_frame_size(b, frame_num, self.config.max_frame_bytes)
            except Exception as e:
                self._handle_validation_error(e, frame_num, "size_exceeded", collect_errors, validation_errors)
                if collect_errors:
                    continue
                raise

            # protocol envelope check (always)
            try:
                FrameValidator.validate_protocol_envelope(
                    frame,
                    frame_num,
                    enable_content_warnings=self.config.enable_content_warnings,
                )
            except Exception as e:
                self._handle_validation_error(e, frame_num, "protocol_envelope", collect_errors, validation_errors)
                if collect_errors:
                    continue
                raise

            # schema validation (conditional)
            if self._should_schema_validate(frame_num):
                frames_validated += 1
                try:
                    self._schema_validate_sync(frame, frame_num)
                    self._emit_frame_validated(frame, frame_num)
                except AssertionError as e:
                    self._handle_validation_error(e, frame_num, "schema_validation", collect_errors, validation_errors)
                    if collect_errors:
                        continue
                    raise
            else:
                frames_skipped += 1

            # streaming semantics
            try:
                state.apply_stream_semantics(frame, frame_num)
            except Exception as e:
                self._handle_validation_error(e, frame_num, "protocol_violation", collect_errors, validation_errors)
                if collect_errors:
                    continue
                raise

        validation_time_ms = (time.time() - start_time) * 1000

        # terminal must be seen
        try:
            if not state.terminal_seen:
                raise StreamProtocolError(f"Stream completed without terminal frame. Processed {total} frames.")
        except Exception as e:
            self._handle_validation_error(e, total, "stream_integrity", collect_errors, validation_errors)

        # performance warning
        if (
            total > self.config.large_stream_threshold
            and validation_time_ms > self.config.performance_warning_threshold_ms
        ):
            coverage = frames_validated / total if total > 0 else 0.0
            self._emit_performance_warning(
                f"Large stream validation took {validation_time_ms:.0f}ms for {total} frames "
                f"(mode={self.config.mode.value}, coverage={coverage:.1%})"
            )

        return StreamValidationReport(
            total_frames=total,
            data_frames=state.data_count,
            ended_ok=state.ended_ok,
            errored=state.errored,
            terminal_seen=state.terminal_seen,
            validation_time_ms=validation_time_ms,
            total_bytes=total_bytes,
            max_frame_bytes=max_frame_size,
            terminal_frame_position=state.terminal_frame_position,
            frames_validated=frames_validated,
            frames_skipped=frames_skipped,
            format=format,
            mode=self.config.mode,
            validation_errors=validation_errors,
        )

    async def validate_items_async(self, items: AsyncIterable[FrameItem], format: StreamFormat) -> StreamValidationReport:
        start_time = time.time()

        total = 0
        total_bytes = 0
        max_frame_size = 0
        frames_validated = 0
        frames_skipped = 0
        validation_errors: list[ValidationError] = []
        collect_errors = self.config.mode == ValidationMode.COLLECT_ERRORS

        state = _StreamState()

        async for item in items:
            total += 1
            frame_num = total
            frame = item.frame

            try:
                b = self._frame_bytes(item)
                total_bytes += b
                max_frame_size = max(max_frame_size, b)
                FrameValidator.check_frame_size(b, frame_num, self.config.max_frame_bytes)
            except Exception as e:
                self._handle_validation_error(e, frame_num, "size_exceeded", collect_errors, validation_errors)
                if collect_errors:
                    continue
                raise

            try:
                FrameValidator.validate_protocol_envelope(
                    frame,
                    frame_num,
                    enable_content_warnings=self.config.enable_content_warnings,
                )
            except Exception as e:
                self._handle_validation_error(e, frame_num, "protocol_envelope", collect_errors, validation_errors)
                if collect_errors:
                    continue
                raise

            if self._should_schema_validate(frame_num):
                frames_validated += 1
                try:
                    await self._schema_validate_async(frame, frame_num)
                    self._emit_frame_validated(frame, frame_num)
                except AssertionError as e:
                    self._handle_validation_error(e, frame_num, "schema_validation", collect_errors, validation_errors)
                    if collect_errors:
                        continue
                    raise
            else:
                frames_skipped += 1

            try:
                state.apply_stream_semantics(frame, frame_num)
            except Exception as e:
                self._handle_validation_error(e, frame_num, "protocol_violation", collect_errors, validation_errors)
                if collect_errors:
                    continue
                raise

        validation_time_ms = (time.time() - start_time) * 1000

        try:
            if not state.terminal_seen:
                raise StreamProtocolError(f"Stream completed without terminal frame. Processed {total} frames.")
        except Exception as e:
            self._handle_validation_error(e, total, "stream_integrity", collect_errors, validation_errors)

        if (
            total > self.config.large_stream_threshold
            and validation_time_ms > self.config.performance_warning_threshold_ms
        ):
            coverage = frames_validated / total if total > 0 else 0.0
            self._emit_performance_warning(
                f"Large async stream validation took {validation_time_ms:.0f}ms for {total} frames "
                f"(mode={self.config.mode.value}, coverage={coverage:.1%})"
            )

        return StreamValidationReport(
            total_frames=total,
            data_frames=state.data_count,
            ended_ok=state.ended_ok,
            errored=state.errored,
            terminal_seen=state.terminal_seen,
            validation_time_ms=validation_time_ms,
            total_bytes=total_bytes,
            max_frame_bytes=max_frame_size,
            terminal_frame_position=state.terminal_frame_position,
            frames_validated=frames_validated,
            frames_skipped=frames_skipped,
            format=format,
            mode=self.config.mode,
            validation_errors=validation_errors,
        )

    # Public helpers (string inputs)

    def validate_ndjson(self, ndjson_text: str) -> StreamValidationReport:
        parser = self._parsers[StreamFormat.NDJSON]
        items = parser.parse(ndjson_text)
        return self.validate_items(items, StreamFormat.NDJSON)

    def validate_sse(self, sse_text: str) -> StreamValidationReport:
        parser = self._parsers[StreamFormat.SSE]
        items = parser.parse(sse_text)
        return self.validate_items(items, StreamFormat.SSE)

    async def validate_ndjson_async(self, ndjson_text: str) -> StreamValidationReport:
        parser = self._parsers[StreamFormat.NDJSON]
        items = await parser.parse_async(ndjson_text)

        async def gen() -> AsyncIterator[FrameItem]:
            for it in items:
                yield it

        return await self.validate_items_async(gen(), StreamFormat.NDJSON)

    async def validate_sse_async(self, sse_text: str) -> StreamValidationReport:
        parser = self._parsers[StreamFormat.SSE]
        items = await parser.parse_async(sse_text)

        async def gen() -> AsyncIterator[FrameItem]:
            for it in items:
                yield it

        return await self.validate_items_async(gen(), StreamFormat.SSE)

    async def validate_ndjson_streaming(self, lines: AsyncIterable[str]) -> StreamValidationReport:
        parser = self._parsers[StreamFormat.NDJSON]
        items = parser.parse_streaming(lines)
        return await self.validate_items_async(items, StreamFormat.NDJSON)

    async def validate_sse_streaming(self, lines: AsyncIterable[str]) -> StreamValidationReport:
        parser = self._parsers[StreamFormat.SSE]
        items = parser.parse_streaming(lines)
        return await self.validate_items_async(items, StreamFormat.SSE)

    # Public helpers (already-parsed frames)

    def validate_frames(self, frames: Iterable[dict[str, Any]]) -> StreamValidationReport:
        """Validate an iterable of already-parsed frames (no raw byte metadata available)."""
        items = (FrameItem(frame=f, raw_bytes=None) for f in frames)
        return self.validate_items(items, StreamFormat.RAW_JSON)

    async def validate_frames_async(self, frames: AsyncIterable[dict[str, Any]]) -> StreamValidationReport:
        """Validate an async iterable of already-parsed frames (no raw byte metadata available)."""

        async def items() -> AsyncIterator[FrameItem]:
            async for f in frames:
                yield FrameItem(frame=f, raw_bytes=None)

        return await self.validate_items_async(items(), StreamFormat.RAW_JSON)


# ------------------------------------------------------------------------------
# Convenience functions
# ------------------------------------------------------------------------------


def _coalesce_stream_schema_id(
    *,
    stream_frame_schema_id: Optional[str],
    envelope_schema_id: Optional[str],
) -> str:
    """
    Backward-compatible schema ID selection.

    Historical callers used envelope_schema_id=...
    Canonical callers use stream_frame_schema_id=...

    Exactly one must be provided.
    """
    schema_id = stream_frame_schema_id or envelope_schema_id
    if not schema_id:
        raise TypeError("Missing required argument: stream_frame_schema_id (or legacy envelope_schema_id)")
    return schema_id


def validate_ndjson_stream(
    ndjson_text: str,
    stream_frame_schema_id: Optional[str] = None,
    component: str = "",
    *,
    envelope_schema_id: Optional[str] = None,
    **kwargs: Any,
) -> StreamValidationReport:
    schema_id = _coalesce_stream_schema_id(
        stream_frame_schema_id=stream_frame_schema_id,
        envelope_schema_id=envelope_schema_id,
    )
    config = ValidationConfig(
        stream_frame_schema_id=schema_id,
        component=component,
        **kwargs,
    )
    return StreamValidationEngine(config).validate_ndjson(ndjson_text)


def validate_sse_stream(
    sse_text: str,
    stream_frame_schema_id: Optional[str] = None,
    component: str = "",
    *,
    envelope_schema_id: Optional[str] = None,
    **kwargs: Any,
) -> StreamValidationReport:
    schema_id = _coalesce_stream_schema_id(
        stream_frame_schema_id=stream_frame_schema_id,
        envelope_schema_id=envelope_schema_id,
    )
    config = ValidationConfig(
        stream_frame_schema_id=schema_id,
        component=component,
        **kwargs,
    )
    return StreamValidationEngine(config).validate_sse(sse_text)


async def validate_ndjson_stream_async(
    ndjson_text: str,
    stream_frame_schema_id: Optional[str] = None,
    component: str = "",
    *,
    envelope_schema_id: Optional[str] = None,
    **kwargs: Any,
) -> StreamValidationReport:
    schema_id = _coalesce_stream_schema_id(
        stream_frame_schema_id=stream_frame_schema_id,
        envelope_schema_id=envelope_schema_id,
    )
    config = ValidationConfig(
        stream_frame_schema_id=schema_id,
        component=component,
        **kwargs,
    )
    return await StreamValidationEngine(config).validate_ndjson_async(ndjson_text)


async def validate_sse_stream_async(
    sse_text: str,
    stream_frame_schema_id: Optional[str] = None,
    component: str = "",
    *,
    envelope_schema_id: Optional[str] = None,
    **kwargs: Any,
) -> StreamValidationReport:
    schema_id = _coalesce_stream_schema_id(
        stream_frame_schema_id=stream_frame_schema_id,
        envelope_schema_id=envelope_schema_id,
    )
    config = ValidationConfig(
        stream_frame_schema_id=schema_id,
        component=component,
        **kwargs,
    )
    return await StreamValidationEngine(config).validate_sse_async(sse_text)
