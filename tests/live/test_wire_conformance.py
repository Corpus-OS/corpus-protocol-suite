# SPDX-License-Identifier: Apache-2.0
"""
Production adapter conformance tests for CORPUS Protocol wire format.

This module contains the actual pytest tests that validate adapters produce
correct wire-level request envelopes. It ties together:

  - wire_cases: Test case registry and definitions
  - wire_validators: Validation logic and helpers

This suite is designed to run as a protocol conformance gate in CI/CD,
independent of golden fixture regression tests.

Usage:
    # Run all conformance tests
    pytest test_wire_conformance.py -v

    # Run only core operations (via marker)
    pytest test_wire_conformance.py -v -m "core"

    # Run only LLM tests (via marker)
    pytest test_wire_conformance.py -v -m "llm"

    # Run vector but not batch operations
    pytest test_wire_conformance.py -v -m "vector and not batch"

    # Skip schema validation (faster iteration)
    pytest test_wire_conformance.py -v --skip-schema

    # With specific adapter
    pytest test_wire_conformance.py -v --adapter=openai

Available markers (automatically applied based on case metadata):
    llm, vector, embedding, graph   - Component markers
    core, health, batch, streaming  - Tag markers
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest

from wire_cases import (
    WireRequestCase,
    get_pytest_params,
    get_registry,
)
from wire_validators import (
    ValidationError,
    EnvelopeShapeError,
    EnvelopeTypeError,
    CtxValidationError,
    ArgsValidationError,
    SchemaValidationError,      # kept for completeness / external use
    SerializationError,
    validate_envelope_common,
    json_roundtrip,
    assert_roundtrip_equality,
    validate_with_version_tolerance,  # kept for completeness / external use
    validate_args_for_operation,
    validate_wire_envelope,
    get_schema_cache,
    CONFIG as VALIDATOR_CONFIG,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ConformanceTestConfig:
    """Configuration for conformance test execution."""

    enable_metrics: bool = True
    skip_schema_validation: bool = False
    verbose_failures: bool = True

    @classmethod
    def from_env(cls) -> "ConformanceTestConfig":
        """Load configuration from environment."""
        return cls(
            enable_metrics=os.environ.get(
                "CORPUS_ENABLE_METRICS", "true"
            ).lower() == "true",
            skip_schema_validation=os.environ.get(
                "CORPUS_SKIP_SCHEMA_VALIDATION", "false"
            ).lower() == "true",
            verbose_failures=os.environ.get(
                "CORPUS_VERBOSE_FAILURES", "true"
            ).lower() == "true",
        )


CONFIG = ConformanceTestConfig.from_env()


# ---------------------------------------------------------------------------
# Metrics Collection (session-scoped)
# ---------------------------------------------------------------------------

@dataclass
class ValidationMetrics:
    """Thread-safe metrics collection for test runs."""

    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    validation_times: List[Tuple[str, float]] = field(default_factory=list)
    successes: Dict[str, int] = field(default_factory=dict)
    failures: Dict[str, Dict[str, int]] = field(default_factory=dict)
    skipped: Dict[str, str] = field(default_factory=dict)

    def record_success(self, case_id: str, duration: float) -> None:
        """Record successful validation."""
        with self._lock:
            self.validation_times.append((case_id, duration))
            self.successes[case_id] = self.successes.get(case_id, 0) + 1

    def record_failure(self, case_id: str, error_type: str, duration: float) -> None:
        """Record validation failure."""
        with self._lock:
            self.validation_times.append((case_id, duration))
            if case_id not in self.failures:
                self.failures[case_id] = {}
            self.failures[case_id][error_type] = (
                self.failures[case_id].get(error_type, 0) + 1
            )

    def record_skip(self, case_id: str, reason: str) -> None:
        """Record skipped test."""
        with self._lock:
            self.skipped[case_id] = reason

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        with self._lock:
            times = [t[1] for t in self.validation_times]
            return {
                "total_runs": len(self.validation_times),
                "total_successes": sum(self.successes.values()),
                "total_failures": sum(
                    sum(errs.values()) for errs in self.failures.values()
                ),
                "total_skipped": len(self.skipped),
                "avg_duration_ms": (sum(times) / len(times) * 1000) if times else 0,
                "min_duration_ms": min(times) * 1000 if times else 0,
                "max_duration_ms": max(times) * 1000 if times else 0,
                "successes_by_case": dict(self.successes),
                "failures_by_case": dict(self.failures),
                "skipped_cases": dict(self.skipped),
            }

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self.validation_times.clear()
            self.successes.clear()
            self.failures.clear()
            self.skipped.clear()


# Session-scoped metrics singleton for aggregate reporting
_session_metrics: Optional[ValidationMetrics] = None


def get_session_metrics() -> ValidationMetrics:
    """Get session-scoped metrics for aggregate reporting across all tests."""
    global _session_metrics
    if _session_metrics is None:
        _session_metrics = ValidationMetrics()
    return _session_metrics


def reset_session_metrics() -> None:
    """Reset session metrics (called at session start)."""
    global _session_metrics
    _session_metrics = ValidationMetrics()


# ---------------------------------------------------------------------------
# Adapter Resolution
# ---------------------------------------------------------------------------

def get_adapter_builder(
    adapter: Any,
    case: WireRequestCase,
) -> Optional[Callable[[], Dict[str, Any]]]:
    """
    Get the builder method from adapter for a test case.

    Returns None if adapter doesn't implement the method.
    """
    builder = getattr(adapter, case.build_method, None)
    if builder is None or not callable(builder):
        return None
    return builder


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_config() -> ConformanceTestConfig:
    """Provide test configuration."""
    return CONFIG


@pytest.fixture(scope="session")
def session_metrics() -> ValidationMetrics:
    """Provide session-scoped metrics for aggregate reporting."""
    reset_session_metrics()
    return get_session_metrics()


@pytest.fixture(scope="session")
def case_registry():
    """Provide the wire request case registry."""
    return get_registry()


# Note: The 'adapter' fixture should be provided by conftest.py
# Example conftest.py:
#
#   @pytest.fixture(scope="session")
#   def adapter(request):
#       adapter_name = request.config.getoption("--adapter", default="default")
#       return load_adapter(adapter_name)


# ---------------------------------------------------------------------------
# Main Parametrized Test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case",
    get_pytest_params(),
    ids=lambda c: c.id,
)
def test_wire_request_envelope(
    case: WireRequestCase,
    adapter: Any,
) -> None:
    """
    Validate wire-level request envelope for a protocol operation.

    Steps:
      1. Get builder method from adapter
      2. Build the envelope
      3. Validate envelope structure
      4. JSON round-trip validation
      5. Schema validation (with version tolerance)
      6. Operation-specific args validation

    Args:
        case: The test case definition.
        adapter: The protocol adapter fixture (from conftest.py).
    """
    start_time = time.perf_counter()
    metrics = get_session_metrics()

    # Get builder
    builder = get_adapter_builder(adapter, case)
    if builder is None:
        metrics.record_skip(case.id, f"Adapter missing: {case.build_method}")
        pytest.skip(f"Adapter does not implement '{case.build_method}'")

    try:
        # Build envelope
        envelope = builder()

        # Run validation pipeline
        if CONFIG.skip_schema_validation:
            # Partial validation without schema
            validate_envelope_common(envelope, case.op, case.id)
            wire_envelope = json_roundtrip(envelope, case.id)
            if VALIDATOR_CONFIG.enable_json_roundtrip:
                assert_roundtrip_equality(envelope, wire_envelope, case.id)
            validate_args_for_operation(
                wire_envelope["args"], case.args_validator, case.id
            )
        else:
            # Full validation including schema
            validate_wire_envelope(
                envelope=envelope,
                expected_op=case.op,
                schema_id=case.schema_id,
                schema_versions=case.schema_versions,
                component=case.component,
                args_validator=case.args_validator,
                case_id=case.id,
            )

        # Record success
        duration = time.perf_counter() - start_time
        if CONFIG.enable_metrics:
            metrics.record_success(case.id, duration)

        logger.info(f"✅ {case.id}: Passed ({duration*1000:.1f}ms)")

    except ValidationError as e:
        duration = time.perf_counter() - start_time
        if CONFIG.enable_metrics:
            metrics.record_failure(case.id, type(e).__name__, duration)

        if CONFIG.verbose_failures:
            logger.error(f"❌ {case.id}: {type(e).__name__}")
            logger.error(f"   Message: {e}")
            if e.field:
                logger.error(f"   Field: {e.field}")
            if e.details:
                logger.error(f"   Details: {e.details}")

        raise


# ---------------------------------------------------------------------------
# Pytest Markers for Filtered Runs
# ---------------------------------------------------------------------------

# Register custom markers in conftest.py or pytest.ini:
#   markers =
#       llm: LLM protocol operations
#       vector: Vector protocol operations
#       embedding: Embedding protocol operations
#       graph: Graph protocol operations
#       core: Core operations every adapter must support
#       health: Health check endpoints
#       batch: Batch operations
#       streaming: Streaming operations
#
# Usage:
#   pytest test_wire_conformance.py -m "llm"
#   pytest test_wire_conformance.py -m "core"
#   pytest test_wire_conformance.py -m "not streaming"


def _get_markers_for_case(case: WireRequestCase) -> List[str]:
    """Get pytest markers for a case based on component and tags."""
    markers = [case.component]
    markers.extend(case.tags)
    return markers


def pytest_collection_modifyitems(config, items) -> None:
    """Add markers to test items based on case metadata."""
    for item in items:
        callspec = getattr(item, "callspec", None)
        if callspec is None:
            continue

        case = callspec.params.get("case")
        if not isinstance(case, WireRequestCase):
            continue

        # Add component marker
        item.add_marker(getattr(pytest.mark, case.component))

        # Add tag markers
        for tag in case.tags:
            item.add_marker(getattr(pytest.mark, tag))


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------

class TestEnvelopeEdgeCases:
    """Test edge cases for envelope validation."""

    def test_missing_op_rejected(self, adapter: Any) -> None:
        """Envelope without 'op' should be rejected."""
        from wire_validators import validate_envelope_shape

        envelope = {"ctx": {"request_id": "test"}, "args": {}}

        with pytest.raises(EnvelopeShapeError, match="missing required keys"):
            validate_envelope_shape(envelope, case_id="test_missing_op")

    def test_missing_ctx_rejected(self, adapter: Any) -> None:
        """Envelope without 'ctx' should be rejected."""
        from wire_validators import validate_envelope_shape

        envelope = {"op": "llm.complete", "args": {}}

        with pytest.raises(EnvelopeShapeError, match="missing required keys"):
            validate_envelope_shape(envelope, case_id="test_missing_ctx")

    def test_missing_args_rejected(self, adapter: Any) -> None:
        """Envelope without 'args' should be rejected."""
        from wire_validators import validate_envelope_shape

        envelope = {"op": "llm.complete", "ctx": {"request_id": "test"}}

        with pytest.raises(EnvelopeShapeError, match="missing required keys"):
            validate_envelope_shape(envelope, case_id="test_missing_args")

    def test_non_dict_envelope_rejected(self, adapter: Any) -> None:
        """Non-dict envelope should be rejected."""
        from wire_validators import validate_envelope_shape

        with pytest.raises(EnvelopeTypeError, match="must be dict"):
            validate_envelope_shape(["not", "a", "dict"], case_id="test_non_dict")

    def test_empty_request_id_rejected(self, adapter: Any) -> None:
        """Empty request_id should be rejected."""
        from wire_validators import validate_ctx_field

        envelope = {"ctx": {"request_id": ""}}

        with pytest.raises(CtxValidationError, match="length must be"):
            validate_ctx_field(envelope, case_id="test_empty_request_id")

    def test_max_request_id_accepted(self, adapter: Any) -> None:
        """Maximum length request_id should be accepted."""
        from wire_validators import validate_ctx_field, MAX_REQUEST_ID_LENGTH

        envelope = {"ctx": {"request_id": "x" * MAX_REQUEST_ID_LENGTH}}
        validate_ctx_field(envelope, case_id="test_max_request_id")

    def test_oversized_request_id_rejected(self, adapter: Any) -> None:
        """Request_id exceeding max length should be rejected."""
        from wire_validators import validate_ctx_field, MAX_REQUEST_ID_LENGTH

        envelope = {"ctx": {"request_id": "x" * (MAX_REQUEST_ID_LENGTH + 1)}}

        with pytest.raises(CtxValidationError, match="length must be"):
            validate_ctx_field(envelope, case_id="test_oversized_request_id")

    def test_deadline_ms_zero_rejected(self, adapter: Any) -> None:
        """Zero deadline_ms should be rejected."""
        from wire_validators import validate_ctx_field

        envelope = {"ctx": {"request_id": "test", "deadline_ms": 0}}

        with pytest.raises(CtxValidationError, match="deadline_ms"):
            validate_ctx_field(envelope, case_id="test_deadline_zero")

    def test_negative_deadline_rejected(self, adapter: Any) -> None:
        """Negative deadline_ms should be rejected."""
        from wire_validators import validate_ctx_field

        envelope = {"ctx": {"request_id": "test", "deadline_ms": -100}}

        with pytest.raises(CtxValidationError, match="deadline_ms"):
            validate_ctx_field(envelope, case_id="test_deadline_negative")

    def test_invalid_priority_rejected(self, adapter: Any) -> None:
        """Invalid priority value should be rejected."""
        from wire_validators import validate_ctx_field

        envelope = {"ctx": {"request_id": "test", "priority": "super_urgent"}}

        with pytest.raises(CtxValidationError, match="priority"):
            validate_ctx_field(envelope, case_id="test_invalid_priority")

    def test_valid_priorities_accepted(self, adapter: Any) -> None:
        """All valid priority values should be accepted."""
        from wire_validators import validate_ctx_field

        for priority in ["low", "normal", "high", "critical"]:
            envelope = {"ctx": {"request_id": "test", "priority": priority}}
            validate_ctx_field(envelope, case_id=f"test_priority_{priority}")


class TestSerializationEdgeCases:
    """Test JSON serialization edge cases."""

    def test_non_serializable_rejected(self, adapter: Any) -> None:
        """Non-JSON-serializable values should be rejected."""
        from wire_validators import json_roundtrip

        envelope = {
            "op": "llm.complete",
            "ctx": {"request_id": "test"},
            "args": {"callback": lambda x: x},
        }

        with pytest.raises(SerializationError):
            json_roundtrip(envelope, case_id="test_non_serializable")

    def test_unicode_preserved(self, adapter: Any) -> None:
        """Unicode should be preserved through round-trip."""
        from wire_validators import json_roundtrip

        unicode_text = "Hello 世界 🌍 مرحبا שלום"
        envelope = {
            "op": "llm.complete",
            "ctx": {"request_id": "test"},
            "args": {"prompt": unicode_text},
        }

        result = json_roundtrip(envelope, case_id="test_unicode")
        assert result["args"]["prompt"] == unicode_text

    def test_float_precision_preserved(self, adapter: Any) -> None:
        """Float precision should be preserved through round-trip."""
        from wire_validators import json_roundtrip

        value = 0.123456789012345
        envelope = {
            "op": "vector.query",
            "ctx": {"request_id": "test"},
            "args": {"vector": [value, value, value]},
        }

        result = json_roundtrip(envelope, case_id="test_float_precision")
        assert result["args"]["vector"][0] == value

    def test_deeply_nested_structure(self, adapter: Any) -> None:
        """Deeply nested structures should serialize correctly."""
        from wire_validators import json_roundtrip

        nested = {"level": 0}
        current = nested
        for i in range(1, 50):
            current["nested"] = {"level": i}
            current = current["nested"]

        envelope = {
            "op": "graph.mutate",
            "ctx": {"request_id": "test"},
            "args": {"data": nested},
        }

        result = json_roundtrip(envelope, case_id="test_deep_nesting")
        assert result == envelope


class TestArgsValidationEdgeCases:
    """Test operation-specific args validation edge cases."""

    def test_llm_complete_missing_prompt_and_messages(self, adapter: Any) -> None:
        """llm.complete without prompt or messages should be rejected."""
        from wire_validators import validate_llm_complete_args

        with pytest.raises(ArgsValidationError, match="requires 'prompt' or 'messages'"):
            validate_llm_complete_args({}, case_id="test")

    def test_llm_chat_empty_messages_rejected(self, adapter: Any) -> None:
        """llm.chat with empty messages should be rejected."""
        from wire_validators import validate_llm_chat_args

        with pytest.raises(ArgsValidationError, match="must not be empty"):
            validate_llm_chat_args({"messages": []}, case_id="test")

    def test_llm_chat_invalid_role_rejected(self, adapter: Any) -> None:
        """llm.chat with invalid role should be rejected."""
        from wire_validators import validate_llm_chat_args

        args = {"messages": [{"role": "invalid", "content": "test"}]}

        with pytest.raises(ArgsValidationError, match="role"):
            validate_llm_chat_args(args, case_id="test")

    def test_vector_query_missing_vector_and_text(self, adapter: Any) -> None:
        """vector.query without vector or text should be rejected."""
        from wire_validators import validate_vector_query_args

        with pytest.raises(ArgsValidationError, match="requires 'vector' or 'text'"):
            validate_vector_query_args({}, case_id="test")

    def test_vector_empty_dimensions_rejected(self, adapter: Any) -> None:
        """Empty vector should be rejected."""
        from wire_validators import validate_vector_query_args

        with pytest.raises(ArgsValidationError, match="dimensions"):
            validate_vector_query_args({"vector": []}, case_id="test")

    def test_vector_non_numeric_rejected(self, adapter: Any) -> None:
        """Non-numeric values in vector should be rejected."""
        from wire_validators import validate_vector_query_args

        with pytest.raises(ArgsValidationError, match="numbers"):
            validate_vector_query_args({"vector": [1.0, "bad", 3.0]}, case_id="test")

    def test_vector_upsert_dimension_mismatch(self, adapter: Any) -> None:
        """Vectors with inconsistent dimensions should be rejected."""
        from wire_validators import validate_vector_upsert_args

        args = {
            "vectors": [
                {"id": "v1", "values": [1.0, 2.0, 3.0]},
                {"id": "v2", "values": [1.0, 2.0]},
            ]
        }

        with pytest.raises(ArgsValidationError, match="mismatch"):
            validate_vector_upsert_args(args, case_id="test")

    def test_embedding_missing_text_and_texts(self, adapter: Any) -> None:
        """embedding.embed without text or texts should be rejected."""
        from wire_validators import validate_embedding_embed_args

        with pytest.raises(ArgsValidationError, match="requires 'text' or 'texts'"):
            validate_embedding_embed_args({}, case_id="test")

    def test_graph_query_invalid_language(self, adapter: Any) -> None:
        """graph.query with invalid language should be rejected."""
        from wire_validators import validate_graph_query_args

        args = {"query": "MATCH (n) RETURN n", "language": "sql"}

        with pytest.raises(ArgsValidationError, match="language"):
            validate_graph_query_args(args, case_id="test")

    def test_graph_mutate_invalid_type(self, adapter: Any) -> None:
        """graph.mutate with invalid mutation type should be rejected."""
        from wire_validators import validate_graph_mutate_args

        args = {"mutations": [{"type": "invalid_mutation"}]}

        with pytest.raises(ArgsValidationError, match="type"):
            validate_graph_mutate_args(args, case_id="test")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    """Add conformance test summary to pytest output."""
    if not CONFIG.enable_metrics:
        return

    metrics = get_session_metrics()
    summary = metrics.get_summary()

    if summary["total_runs"] == 0:
        return

    terminalreporter.write_sep("=", "Wire Conformance Summary")
    terminalreporter.write_line(f"Total runs:     {summary['total_runs']}")
    terminalreporter.write_line(f"Successes:      {summary['total_successes']}")
    terminalreporter.write_line(f"Failures:       {summary['total_failures']}")
    terminalreporter.write_line(f"Skipped:        {summary['total_skipped']}")
    terminalreporter.write_line(f"Avg duration:   {summary['avg_duration_ms']:.1f}ms")

    if summary["failures_by_case"]:
        terminalreporter.write_line("")
        terminalreporter.write_line("Failures by case:")
        for case_id, errors in summary["failures_by_case"].items():
            for error_type, count in errors.items():
                terminalreporter.write_line(f"  {case_id}: {error_type} ({count})")

    if summary["skipped_cases"]:
        terminalreporter.write_line("")
        terminalreporter.write_line("Skipped cases:")
        for case_id, reason in summary["skipped_cases"].items():
            terminalreporter.write_line(f"  {case_id}: {reason}")


# ---------------------------------------------------------------------------
# CLI Hooks
# ---------------------------------------------------------------------------

def pytest_addoption(parser) -> None:
    """Add CLI options for conformance tests."""
    parser.addoption(
        "--adapter",
        action="store",
        default="default",
        help="Adapter to test (default: 'default')",
    )
    parser.addoption(
        "--skip-schema",
        action="store_true",
        default=False,
        help="Skip JSON Schema validation",
    )
    parser.addoption(
        "--conformance-verbose",
        action="store_true",
        default=False,
        help="Verbose conformance test output",
    )


def pytest_configure(config) -> None:
    """Configure test run based on CLI options."""
    global CONFIG

    if config.getoption("--skip-schema", default=False):
        CONFIG = ConformanceTestConfig(
            enable_metrics=CONFIG.enable_metrics,
            skip_schema_validation=True,
            verbose_failures=CONFIG.verbose_failures,
        )

    if config.getoption("--conformance-verbose", default=False):
        CONFIG = ConformanceTestConfig(
            enable_metrics=CONFIG.enable_metrics,
            skip_schema_validation=CONFIG.skip_schema_validation,
            verbose_failures=True,
        )

    # Reset session metrics at start of test session
    reset_session_metrics()

    # Register markers
    config.addinivalue_line("markers", "llm: LLM protocol operations")
    config.addinivalue_line("markers", "vector: Vector protocol operations")
    config.addinivalue_line("markers", "embedding: Embedding protocol operations")
    config.addinivalue_line("markers", "graph: Graph protocol operations")
    config.addinivalue_line("markers", "core: Core operations every adapter must support")
    config.addinivalue_line("markers", "health: Health check endpoints")
    config.addinivalue_line("markers", "batch: Batch operations")
    config.addinivalue_line("markers", "streaming: Streaming operations")


# ---------------------------------------------------------------------------
# Coverage Report Generation
# ---------------------------------------------------------------------------

def generate_coverage_report() -> Dict[str, Any]:
    """Generate test coverage report."""
    registry = get_registry()
    metrics = get_session_metrics()
    summary = metrics.get_summary()

    return {
        "registry": registry.get_coverage_summary(),
        "execution": summary,
        "schema_cache": get_schema_cache().stats,
    }


# ---------------------------------------------------------------------------
# Standalone Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Allow running directly for quick checks
    pytest.main([__file__, "-v", "--tb=short"])
