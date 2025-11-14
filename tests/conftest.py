# SPDX-License-Identifier: Apache-2.0
"""
Pytest plugin: comprehensive terminal summary for Corpus Protocol conformance.

Drop this into `tests/conftest.py` (or any file auto-discovered by pytest)
to get detailed per-protocol conformance reporting with certification levels,
failure analysis, and actionable guidance.
"""

from __future__ import annotations

import os
import time
import importlib
from typing import Dict, List, Optional, Tuple, Any

import pytest


# ---------------------------------------------------------------------------
# Pluggable adapter fixture (SDK-agnostic, no hard dependencies)
# ---------------------------------------------------------------------------

# Environment variable for fully-qualified adapter class:
#   CORPUS_ADAPTER="package.module:ClassName"
#
# This allows vendors (or you) to plug in ANY adapter implementation that
# satisfies the expected protocol interface, without this file knowing
# anything about corpus_sdk or concrete adapter types.
ADAPTER_ENV = "CORPUS_ADAPTER"

# Default adapter used when CORPUS_ADAPTER is not set.
# This assumes a local mock implementation lives in the tests repo at:
#   tests/mocks/mock_llm_adapter.py
# with a class named:
#   MockLLMAdapter
DEFAULT_ADAPTER = "tests.mocks.mock_llm_adapter:MockLLMAdapter"

# Optional endpoint for network-backed adapters. If set, we will attempt to
# pass it to the adapter's constructor as one of:
#   endpoint=, base_url=, or url=
# The adapter is free to ignore it or not accept these parameters.
ENDPOINT_ENV = "CORPUS_ENDPOINT"

# Cached adapter class to avoid repeated import and attribute lookups.
_ADAPTER_CLASS: Optional[type] = None
_ADAPTER_SPEC_USED: Optional[str] = None


def _load_class_from_spec(spec: str) -> type:
    """
    Load a class from a 'package.module:ClassName' string.

    This function performs no assumptions about the underlying SDK or
    type hierarchy; it simply loads a Python object by module and name.
    """
    module_name, _, class_name = spec.partition(":")
    if not module_name or not class_name:
        raise RuntimeError(
            f"Invalid adapter spec '{spec}'. Expected 'package.module:ClassName'."
        )

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"Failed to import adapter module '{module_name}' for spec '{spec}'."
        ) from exc

    try:
        cls = getattr(module, class_name)
    except AttributeError as exc:
        raise RuntimeError(
            f"Adapter class '{class_name}' not found in module '{module_name}' "
            f"for spec '{spec}'."
        ) from exc

    if not callable(cls):
        raise RuntimeError(
            f"Resolved adapter '{spec}' is not callable. "
            f"Expected a class or factory function."
        )

    return cls  # type: ignore[return-value]


def _get_adapter_class() -> type:
    """
    Resolve and cache the adapter class based on environment configuration.

    Resolution order:
      1. If CORPUS_ADAPTER is set, use that spec.
      2. Otherwise, fall back to DEFAULT_ADAPTER.

    The resolved class is cached for the duration of the test session to
    avoid repeated dynamic imports and attribute lookups.
    """
    global _ADAPTER_CLASS, _ADAPTER_SPEC_USED

    if _ADAPTER_CLASS is not None:
        return _ADAPTER_CLASS

    spec = os.getenv(ADAPTER_ENV, DEFAULT_ADAPTER)
    _ADAPTER_SPEC_USED = spec
    _ADAPTER_CLASS = _load_class_from_spec(spec)
    return _ADAPTER_CLASS


@pytest.fixture
def adapter():
    """
    Generic, pluggable adapter fixture.

    This fixture is intentionally SDK-agnostic: it does not import or
    reference corpus_sdk, LLMAdapter, or any concrete adapter types.

    Behavior:
      - Reads CORPUS_ADAPTER (if set) as 'package.module:ClassName'.
      - Otherwise uses DEFAULT_ADAPTER (tests.mocks.mock_llm_adapter:MockLLMAdapter).
      - If CORPUS_ENDPOINT is set, it will *attempt* to pass it to the
        adapter's constructor using common parameter names: endpoint=,
        base_url=, or url=. If none of these match the adapter's signature,
        it falls back to calling the adapter with no arguments.

    This keeps the tests flexible: any implementation can be swapped in
    via environment configuration, as long as it matches the expected
    protocol interface at runtime.
    """
    Adapter = _get_adapter_class()
    endpoint = os.getenv(ENDPOINT_ENV)

    if endpoint:
        # Best-effort injection of endpoint-style parameter. We deliberately
        # do not inspect signatures to avoid extra overhead and complexity;
        # instead we try a few common names and gracefully fall back.
        for kw in ("endpoint", "base_url", "url"):
            try:
                return Adapter(**{kw: endpoint})  # type: ignore[arg-type]
            except TypeError:
                # The adapter's __init__ does not accept this keyword;
                # try the next one.
                continue

        # As a final fallback, instantiate without arguments. This allows
        # adapters that obtain configuration from other sources (e.g. env).
        try:
            return Adapter()  # type: ignore[call-arg]
        except TypeError as exc:
            raise RuntimeError(
                f"Failed to instantiate adapter '{_ADAPTER_SPEC_USED or Adapter}' "
                f"with or without endpoint from {ENDPOINT_ENV}."
            ) from exc

    # Local/mock mode (no endpoint configured)
    try:
        return Adapter()  # type: ignore[call-arg]
    except TypeError as exc:
        raise RuntimeError(
            f"Failed to instantiate adapter '{_ADAPTER_SPEC_USED or Adapter}' "
            f"without arguments. Ensure your mock/adapter has a no-arg "
            f"constructor or configure {ENDPOINT_ENV} appropriately."
        ) from exc


# ---------------------------------------------------------------------------
# Existing Corpus Protocol conformance reporting plugin
# ---------------------------------------------------------------------------

# Protocol configuration with certification levels - CORRECTED FOR 78 GOLDEN TESTS
PROTOCOLS = ["llm", "vector", "graph", "embedding", "schema", "golden"]

PROTOCOL_DISPLAY_NAMES = {
    "llm": "LLM Protocol V1.0",
    "vector": "Vector Protocol V1.0",
    "graph": "Graph Protocol V1.0",
    "embedding": "Embedding Protocol V1.0",
    "schema": "Schema Conformance",
    "golden": "Golden Wire Validation"
}

# CORRECTED: Updated golden tests from 73 to 78 to match actual test_golden_samples.py
CONFORMANCE_LEVELS = {
    "llm": {"gold": 61, "silver": 49, "development": 31},
    "vector": {"gold": 72, "silver": 58, "development": 36},
    "graph": {"gold": 68, "silver": 54, "development": 34},
    "embedding": {"gold": 75, "silver": 60, "development": 38},
    "schema": {"gold": 13, "silver": 10, "development": 7},    # 13 schema meta-lint tests
    "golden": {"gold": 78, "silver": 62, "development": 39},   # CORRECTED: 78 golden tests (55 parametrized + 23 standalone)
}

TEST_CATEGORIES = {
    "llm": {
        "core_ops": "Core Operations",
        "message_validation": "Message Validation",
        "sampling_params": "Sampling Parameters",
        "streaming": "Streaming Semantics",
        "error_handling": "Error Handling",
        "capabilities": "Capabilities Discovery",
        "observability": "Observability & Privacy",
        "deadline": "Deadline Semantics",
        "token_counting": "Token Counting",
        "health": "Health Endpoint",
        "wire_envelopes": "Wire Envelopes & Routing"
    },
    "vector": {
        "core_ops": "Core Operations",
        "capabilities": "Capabilities Discovery",
        "namespace": "Namespace Management",
        "upsert": "Upsert Operations",
        "query": "Query Operations",
        "delete": "Delete Operations",
        "filtering": "Filtering Semantics",
        "dimension_validation": "Dimension Validation",
        "error_handling": "Error Handling",
        "deadline": "Deadline Semantics",
        "health": "Health Endpoint",
        "observability": "Observability & Privacy",
        "batch_limits": "Batch Size Limits",
        "wire_envelopes": "Wire Envelopes & Routing"
    },
    "graph": {
        "core_ops": "Core Operations",
        "crud_validation": "CRUD Validation",
        "query_ops": "Query Operations",
        "dialect_validation": "Dialect Validation",
        "streaming": "Streaming Semantics",
        "batch_ops": "Batch Operations",
        "schema_ops": "Schema Operations",
        "error_handling": "Error Handling",
        "capabilities": "Capabilities Discovery",
        "observability": "Observability & Privacy",
        "deadline": "Deadline Semantics",
        "health": "Health Endpoint",
        "wire_envelopes": "Wire Envelopes & Routing"
    },
    "embedding": {
        "core_ops": "Core Operations",
        "capabilities": "Capabilities Discovery",
        "batch_partial": "Batch & Partial Failures",
        "truncation": "Truncation & Length",
        "normalization": "Normalization Semantics",
        "token_counting": "Token Counting",
        "error_handling": "Error Handling",
        "deadline": "Deadline Semantics",
        "health": "Health Endpoint",
        "observability": "Observability & Privacy",
        "caching": "Caching & Idempotency",
        "wire_contract": "Wire Contract"
    },
    # ALIGNED WITH SCHEMA_CONFORMANCE.md TEST CATEGORIES
    "schema": {
        "schema_loading": "Schema Loading & IDs",
        "file_organization": "File Organization",
        "metaschema_hygiene": "Metaschema & Hygiene",
        "cross_references": "Cross-References",
        "definitions": "Definitions",
        "envelopes_constants": "Envelopes & Constants",
        "examples_validation": "Examples Validation",
        "stream_frames": "Stream Frames",
        "performance_metrics": "Performance & Metrics"
    },
    "golden": {
        "core_validation": "Core Schema Validation",
        "ndjson_stream": "NDJSON Stream Validation",
        "cross_invariants": "Cross-Schema Invariants",
        "version_format": "Schema Version & Format",
        "drift_detection": "Drift Detection",
        "performance_reliability": "Performance & Reliability",
        "component_coverage": "Component Coverage"
    }
}

SPEC_SECTION_MAPPING = {
    "llm": {
        "core_ops": "§8.3 Operations",
        "message_validation": "§8.3 Operations",
        "sampling_params": "§8.3 Operations",
        "streaming": "§8.3 Operations & §4.1.3 Streaming Frames",
        "error_handling": "§8.5 LLM-Specific Errors",
        "capabilities": "§8.4 Model Discovery",
        "observability": "§6.4 Observability Interfaces & §13 Observability and Monitoring",
        "deadline": "§4.3 Deadline Propagation & §6.1 Operation Context",
        "token_counting": "§8.3 Operations",
        "health": "§8.3 Operations",
        "wire_envelopes": "§4.1 Wire-First Canonical Form"
    },
    "vector": {
        "core_ops": "§9.3 Operations",
        "capabilities": "§9.3 Operations",
        "namespace": "§9.3 Operations",
        "upsert": "§9.3 Operations",
        "query": "§9.3 Operations",
        "delete": "§9.3 Operations",
        "filtering": "§9.3 Operations",
        "dimension_validation": "§9.5 Vector-Specific Errors",
        "error_handling": "§9.5 Vector-Specific Errors & §12.4 Error Mapping Table",
        "deadline": "§4.3 Deadline Propagation & §6.1 Operation Context",
        "health": "§9.3 Operations",
        "observability": "§6.4 Observability Interfaces & §13 Observability and Monitoring",
        "batch_limits": "§9.3 Operations & §12.5 Partial Failure Contracts",
        "wire_envelopes": "§4.1 Wire-First Canonical Form"
    },
    "graph": {
        "core_ops": "§7.3 Operations",
        "crud_validation": "§7.3.1 Vertex/Edge CRUD",
        "query_ops": "§7.3.2 Queries",
        "dialect_validation": "§7.4 Dialects",
        "streaming": "§7.3.2 Streaming Finalization & §4.1.3 Streaming Frames",
        "batch_ops": "§7.3.3 Batch Operations",
        "schema_ops": "§7.5 Schema Operations (Optional)",
        "error_handling": "§7.3 Operations & §12.4 Error Mapping Table",
        "capabilities": "§7.3 Operations & §6.2 Capability Discovery",
        "observability": "§6.4 Observability Interfaces & §13 Observability and Monitoring",
        "deadline": "§4.3 Deadline Propagation & §6.1 Operation Context",
        "health": "§7.6 Health",
        "wire_envelopes": "§4.1 Wire-First Canonical Form"
    },
    "embedding": {
        "core_ops": "§10.3 Operations (Normative Signatures)",
        "capabilities": "§10.5 Capabilities",
        "batch_partial": "§10.3 Operations & §12.5 Partial Failure Contracts",
        "truncation": "§10.6 Semantics",
        "normalization": "§10.6 Semantics",
        "token_counting": "§10.3 Operations",
        "error_handling": "§10.4 Errors (Embedding-Specific) & §12.4 Error Mapping Table",
        "deadline": "§4.3 Deadline Propagation & §6.1 Operation Context",
        "health": "§10.3 Operations",
        "observability": "§6.4 Observability Interfaces & §13 Observability and Monitoring",
        "caching": "§11.6 Caching (Implementation Guidance)",
        "wire_contract": "§4.1 Wire-First Canonical Form"
    },
    # ALIGNED WITH SCHEMA_CONFORMANCE.md TEST STRUCTURE
    "schema": {
        "schema_loading": "Schema Meta-Lint Suite - Schema Loading & IDs",
        "file_organization": "Schema Meta-Lint Suite - File Organization",
        "metaschema_hygiene": "Schema Meta-Lint Suite - Metaschema & Hygiene",
        "cross_references": "Schema Meta-Lint Suite - Cross-References",
        "definitions": "Schema Meta-Lint Suite - Definitions",
        "envelopes_constants": "Schema Meta-Lint Suite - Envelopes & Constants",
        "examples_validation": "Schema Meta-Lint Suite - Examples Validation",
        "stream_frames": "Schema Meta-Lint Suite - Stream Frames",
        "performance_metrics": "Schema Meta-Lint Suite - Performance & Metrics"
    },
    "golden": {
        "core_validation": "Golden Samples Suite - Core Schema Validation",
        "ndjson_stream": "Golden Samples Suite - NDJSON Stream Validation",
        "cross_invariants": "Golden Samples Suite - Cross-Schema Invariants",
        "version_format": "Schema Version & Format",
        "drift_detection": "Golden Samples Suite - Drift Detection",
        "performance_reliability": "Golden Samples Suite - Performance & Reliability",
        "component_coverage": "Golden Samples Suite - Component Coverage"
    }
}

# ALIGNED WITH ACTUAL TEST NAMES FROM SCHEMA_CONFORMANCE.md
ERROR_GUIDANCE_MAPPING = {
    "schema": {
        "schema_loading": {
            "test_schema_loading": {
                "error_patterns": {
                    "invalid_schema": "Schema file failed to load or parse",
                    "missing_schema": "Required $schema field missing or invalid"
                },
                "quick_fix": "Ensure all schema files are valid JSON and include $schema: 'https://json-schema.org/draft/2020-12/schema'",
                "examples": "See SCHEMA_CONFORMANCE.md - Schema Loading & IDs section"
            },
            "test_unique_ids": {
                "error_patterns": {
                    "duplicate_id": "Duplicate $id found across schema files",
                    "invalid_id_format": "$id does not follow https://adaptersdk.org/schemas/ format"
                },
                "quick_fix": "Ensure each schema has unique $id following convention: https://adaptersdk.org/schemas/<component>/<file>.json",
                "examples": "See SCHEMA_CONFORMANCE.md - $id hygiene requirements"
            }
        },
        "metaschema_hygiene": {
            "test_metaschema_compliance": {
                "error_patterns": {
                    "draft_2020_12_violation": "Schema violates JSON Schema Draft 2020-12",
                    "invalid_keywords": "Unknown or invalid JSON Schema keywords used"
                },
                "quick_fix": "Validate schema against Draft 2020-12 metaschema and remove unsupported keywords",
                "examples": "See SCHEMA_CONFORMANCE.md - Metaschema & Hygiene section"
            },
            "test_regex_patterns": {
                "error_patterns": {
                    "invalid_regex": "Regular expression pattern does not compile",
                    "unsupported_regex_flags": "Regex uses unsupported flags"
                },
                "quick_fix": "Fix regex patterns to use supported ECMA 262 syntax without flags",
                "examples": "See SCHEMA_CONFORMANCE.md - Pattern hygiene requirements"
            }
        },
        "cross_references": {
            "test_ref_resolution": {
                "error_patterns": {
                    "unresolved_ref": "$ref cannot be resolved to known schema $id",
                    "invalid_fragment": "Fragment (#/definitions/...) points to non-existent definition"
                },
                "quick_fix": "Ensure all $ref values point to valid $ids or internal fragments",
                "examples": "See SCHEMA_CONFORMANCE.md - Cross-References section"
            }
        }
    },
    "golden": {
        "core_validation": {
            "test_golden_validates": {
                "error_patterns": {
                    "schema_validation_failed": "Golden sample does not validate against its declared schema",
                    "missing_schema_reference": "Golden file missing $schema reference"
                },
                "quick_fix": "Update golden sample to match schema or fix schema definition",
                "examples": "See SCHEMA_CONFORMANCE.md - Golden Samples Suite section"
            }
        },
        "ndjson_stream": {
            "test_llm_stream_ndjson_union_validates": {
                "error_patterns": {
                    "invalid_frame_sequence": "Stream frames violate terminal frame rules",
                    "missing_terminal_frame": "Stream missing required end or error frame"
                },
                "quick_fix": "Ensure streams have exactly one terminal frame (end/error) after data frames",
                "examples": "See SCHEMA_CONFORMANCE.md - NDJSON Stream Validation"
            }
        },
        "cross_invariants": {
            "test_partial_success_math": {
                "error_patterns": {
                    "count_mismatch": "successes + failures ≠ total items in partial success",
                    "invalid_indexing": "Failure indices out of bounds"
                },
                "quick_fix": "Ensure partial success counts are mathematically consistent",
                "examples": "See SCHEMA_CONFORMANCE.md - Cross-Schema Invariants"
            }
        }
    },
    "llm": {
        "streaming": {
            "test_stream_finalization": {
                "error_patterns": {
                    "missing_final_chunk": "Ensure stream ends with terminal frame per §4.1.3",
                    "premature_close": "Connection must remain open until terminal frame per §7.3.2 Streaming Finalization",
                    "chunk_format": "Each frame must follow §4.1.3 Streaming Frames format"
                },
                "quick_fix": "Add terminal frame (event: 'end' or event: 'error') after all data frames",
                "examples": "See §4.1.3 for frame format and §7.3.2 for streaming finalization rules"
            }
        },
        "sampling_params": {
            "test_temperature_validation": {
                "error_patterns": {
                    "invalid_range": "Temperature must be between 0.0 and 2.0 per §8.3",
                    "type_error": "Temperature must be float, not string per §4.1 Numeric Types"
                },
                "quick_fix": "Clamp temperature values to valid range [0.0, 2.0] and ensure numeric types",
                "examples": "See §8.3 for parameter validation and §4.1 for numeric type rules"
            }
        }
    },
    "vector": {
        "namespace": {
            "test_namespace_isolation": {
                "error_patterns": {
                    "cross_namespace_leak": "Data must be strictly isolated per §14.1 Tenant Isolation",
                    "invalid_namespace": "Namespace must follow §9.3 Operations requirements"
                },
                "quick_fix": "Validate namespace format and enforce isolation at storage layer",
                "examples": "See §9.3 for namespace operations and §14.1 for tenant isolation requirements"
            }
        },
        "dimension_validation": {
            "test_dimension_mismatch": {
                "error_patterns": {
                    "dimension_mismatch": "Vector dimensions must match index dimensions per §9.5",
                    "invalid_dimension": "Dimensions must be positive integers per §4.1 Numeric Types"
                },
                "quick_fix": "Validate vector dimensions before upsert operations",
                "examples": "See §9.5 for dimension handling and §4.1 for numeric validation"
            }
        }
    }
}


class CorpusProtocolPlugin:
    """Pytest plugin for comprehensive Corpus Protocol conformance reporting."""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.test_reports: Dict[str, List[Any]] = {}
        self.protocol_counts: Dict[str, Dict[str, int]] = {}

    def pytest_sessionstart(self, session):
        """Record session start time and initialize tracking."""
        self.start_time = time.time()
        self.test_reports = {proto: [] for proto in PROTOCOLS}
        self.test_reports["other"] = []
        self.protocol_counts = {proto: {} for proto in PROTOCOLS}

    def _get_test_counts(self, terminalreporter) -> Dict[str, int]:
        """Get counts of passed, failed, skipped tests."""
        return {
            "passed": len(terminalreporter.stats.get("passed", [])),
            "failed": len(terminalreporter.stats.get("failed", [])),
            "skipped": len(terminalreporter.stats.get("skipped", [])),
            "error": len(terminalreporter.stats.get("error", [])),
            "xfailed": len(terminalreporter.stats.get("xfailed", [])),
            "xpassed": len(terminalreporter.stats.get("xpassed", [])),
        }

    def _get_total_tests(self, counts: Dict[str, int]) -> int:
        """Calculate total number of tests run."""
        return sum(counts.values())

    def _get_duration(self) -> float:
        """Calculate test session duration."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    def _categorize_test_by_protocol(self, nodeid: str) -> Tuple[str, str]:
        """Categorize test by protocol and test category."""
        # Handle both POSIX and Windows paths
        nodeid_lower = nodeid.lower()

        for proto in PROTOCOLS:
            if f"tests/{proto}/" in nodeid_lower or f"tests\\{proto}\\" in nodeid_lower:
                # Further categorize by test type
                test_name = nodeid_lower
                category = "unknown"

                for cat_key, cat_name in TEST_CATEGORIES.get(proto, {}).items():
                    if cat_key in test_name or cat_name.lower() in test_name:
                        category = cat_key
                        break

                return proto, category

        return "other", "unknown"

    def _categorize_failures(self, failed_reports: List) -> Dict[str, Dict[str, List[Any]]]:
        """Categorize failed tests by protocol and category - FIXED ARCHITECTURE."""
        by_protocol = {proto: {} for proto in PROTOCOLS}
        by_protocol["other"] = {}

        for rep in failed_reports:
            nodeid = getattr(rep, "nodeid", "") or ""
            proto, category = self._categorize_test_by_protocol(nodeid)

            if proto not in by_protocol:
                by_protocol[proto] = {}
            if category not in by_protocol[proto]:
                by_protocol[proto][category] = []  # Store actual reports, not counts

            by_protocol[proto][category].append(rep)  # Add the full report

        return by_protocol

    def _calculate_conformance_level(self, protocol: str, passed_count: int) -> Tuple[str, int]:
        """Calculate conformance level and progress to next level."""
        levels = CONFORMANCE_LEVELS.get(protocol, {})

        if passed_count >= levels.get("gold", 0):
            return "🥇 Gold", 0
        elif passed_count >= levels.get("silver", 0):
            needed = levels.get("gold", 0) - passed_count
            return "🥈 Silver", needed
        elif passed_count >= levels.get("development", 0):
            needed = levels.get("silver", 0) - passed_count
            return "🔬 Development", needed
        else:
            needed = levels.get("development", 0) - passed_count
            return "❌ Below Development", needed

    def _get_spec_section(self, protocol: str, category: str) -> str:
        """Get specification section for a test category."""
        protocol_map = SPEC_SECTION_MAPPING.get(protocol, {})
        return protocol_map.get(category, "See protocol specification")

    def _get_error_guidance(self, protocol: str, category: str, test_name: str) -> Dict[str, str]:
        """Get specific error guidance for a test failure."""
        protocol_guidance = ERROR_GUIDANCE_MAPPING.get(protocol, {})
        category_guidance = protocol_guidance.get(category, {})
        test_guidance = category_guidance.get(test_name, {})

        return {
            "error_patterns": test_guidance.get("error_patterns", {}),
            "quick_fix": test_guidance.get("quick_fix", "Review specification section above"),
            "examples": test_guidance.get("examples", "See specification for implementation details")
        }

    def _extract_test_name(self, nodeid: str) -> str:
        """Extract the test function name from nodeid."""
        # nodeid format: "tests/llm/test_streaming.py::test_stream_finalization"
        parts = nodeid.split("::")
        return parts[-1] if len(parts) > 1 else "unknown_test"

    def _print_platinum_certification(self, terminalreporter, counts: Dict[str, int], duration: float):
        """Print Platinum certification summary."""
        total_tests = self._get_total_tests(counts)

        terminalreporter.write_sep("=", "🏆 CORPUS PROTOCOL SUITE - PLATINUM CERTIFIED")
        terminalreporter.write_line(
            f"All {total_tests} conformance tests passed across 6 test suites"
        )
        terminalreporter.write_line("")

        # Show protocol breakdown
        terminalreporter.write_line("Protocol Conformance Status:")
        for proto in PROTOCOLS:
            display_name = PROTOCOL_DISPLAY_NAMES.get(proto, proto.upper())
            level, _ = self._calculate_conformance_level(proto, CONFORMANCE_LEVELS[proto]["gold"])
            terminalreporter.write_line(f"  ✅ {display_name}: {level}")

        terminalreporter.write_line("")
        terminalreporter.write_line(f"⏱️  Completed in {duration:.2f}s")
        terminalreporter.write_line("🎯 Status: Ready for production deployment")
        terminalreporter.write_line("📚 Specification: All requirements met per Corpus Protocol Suite V1.0")

    def _print_gold_certification(self, terminalreporter, protocol_results: Dict[str, int], duration: float):
        """Print Gold certification summary with progress to Platinum."""
        terminalreporter.write_sep("=", "🥇 CORPUS PROTOCOL SUITE - GOLD CERTIFIED")

        terminalreporter.write_line("Protocol Conformance Status:")
        platinum_ready = True

        for proto in PROTOCOLS:
            display_name = PROTOCOL_DISPLAY_NAMES.get(proto, proto.upper())
            passed = protocol_results.get(proto, 0)
            level, needed = self._calculate_conformance_level(proto, passed)

            if "Gold" in level:
                terminalreporter.write_line(f"  ✅ {display_name}: {level}")
            else:
                platinum_ready = False
                terminalreporter.write_line(f"  ⚠️  {display_name}: {level} ({needed} tests to Gold)")

        terminalreporter.write_line("")
        if platinum_ready:
            terminalreporter.write_line("🎯 All protocols at Gold level - Platinum certification available!")
        else:
            terminalreporter.write_line("🎯 Focus on protocols below Gold level for Platinum certification")

        terminalreporter.write_line(f"⏱️  Completed in {duration:.2f}s")
        terminalreporter.write_line("📚 Review CONFORMANCE.md for detailed test-to-spec mapping")

    def _print_failure_analysis(self, terminalreporter, by_protocol: Dict[str, Dict[str, List[Any]]], duration: float):
        """Print detailed failure analysis with actionable guidance - FIXED ARCHITECTURE."""
        terminalreporter.write_sep("=", "❌ CORPUS PROTOCOL CONFORMANCE ANALYSIS")

        total_failures = 0
        for proto_failures in by_protocol.values():
            for reports_list in proto_failures.values():
                total_failures += len(reports_list)  # Count from actual reports

        terminalreporter.write_line(f"Found {total_failures} conformance issue(s) across protocols:")
        terminalreporter.write_line("")

        # Show failures by protocol and category with specific guidance
        for proto, categories in by_protocol.items():
            if not categories:
                continue

            display_name = PROTOCOL_DISPLAY_NAMES.get(proto, proto.upper())
            terminalreporter.write_line(f"{display_name}:")

            for category, reports_list in categories.items():
                count = len(reports_list)
                category_name = TEST_CATEGORIES.get(proto, {}).get(category, category.replace('_', ' ').title())
                spec_section = self._get_spec_section(proto, category)

                terminalreporter.write_line(f"  ❌ {category_name}: {count} failure(s)")
                terminalreporter.write_line(f"      Specification: {spec_section}")

                # Show specific guidance for each failed test - FIXED LOGIC
                for rep in reports_list:
                    test_name = self._extract_test_name(rep.nodeid)
                    guidance = self._get_error_guidance(proto, category, test_name)

                    # Only show guidance if we have specific advice
                    if guidance["quick_fix"] != "Review specification section above":
                        terminalreporter.write_line(f"      Test: {test_name}")
                        terminalreporter.write_line(f"      Quick fix: {guidance['quick_fix']}")

                        # Show error patterns if available
                        error_patterns = guidance.get("error_patterns", {})
                        if error_patterns:
                            # Try to extract actual error message for pattern matching
                            error_msg = getattr(rep, 'longrepr', str(rep))
                            for pattern_key, pattern_desc in error_patterns.items():
                                if pattern_key.lower() in str(error_msg).lower():
                                    terminalreporter.write_line(f"      Detected: {pattern_desc}")
                                    break

            terminalreporter.write_line("")

        # Certification impact
        terminalreporter.write_line("Certification Impact:")
        failing_protocols = [p for p, cats in by_protocol.items() if cats and p != "other"]

        if failing_protocols:
            terminalreporter.write_line("  ⚠️  Platinum certification blocked by failures in:")
            for proto in failing_protocols:
                display_name = PROTOCOL_DISPLAY_NAMES.get(proto, proto.upper())
                terminalreporter.write_line(f"      - {display_name}")
        else:
            terminalreporter.write_line("  ✅ No protocol conformance failures - review 'other' category tests")

        terminalreporter.write_line("")
        terminalreporter.write_line("Next Steps:")
        terminalreporter.write_line("  1. Review failing tests above with spec section references")
        terminalreporter.write_line("  2. Check SPECIFICATION.md for detailed requirements")
        terminalreporter.write_line("  3. Run individual protocol tests: make test-{protocol}-conformance")
        terminalreporter.write_line("  4. Review error guidance in test output for specific fixes")
        terminalreporter.write_line(f"⏱️  Completed in {duration:.2f}s")

    def _collect_protocol_results(self, terminalreporter) -> Dict[str, int]:
        """Collect passed test counts per protocol."""
        protocol_results = {proto: 0 for proto in PROTOCOLS}

        # Count passed tests per protocol
        for test_report in terminalreporter.stats.get("passed", []):
            nodeid = getattr(test_report, "nodeid", "") or ""
            proto, _ = self._categorize_test_by_protocol(nodeid)
            if proto in protocol_results:
                protocol_results[proto] += 1

        return protocol_results

    def _is_platinum_certified(self, protocol_results: Dict[str, int]) -> bool:
        """Check if all protocols meet Platinum certification requirements."""
        for proto in PROTOCOLS:
            passed = protocol_results.get(proto, 0)
            if passed < CONFORMANCE_LEVELS[proto]["gold"]:
                return False
        return True

    def pytest_terminal_summary(self, terminalreporter, exitstatus, config):
        """Generate comprehensive Corpus Protocol conformance summary."""
        # Collect both test failures and internal errors
        failed_reports = []
        for key in ("failed", "error"):
            failed_reports.extend(terminalreporter.stats.get(key, []))

        # Get test counts and duration
        counts = self._get_test_counts(terminalreporter)
        duration = self._get_duration()

        # Collect protocol-specific results
        protocol_results = self._collect_protocol_results(terminalreporter)

        # Check if any tests actually failed
        if not failed_reports:
            # All tests that ran have passed.

            if self._is_platinum_certified(protocol_results):
                # Best case: All protocols ran and hit Gold.
                self._print_platinum_certification(terminalreporter, counts, duration)
            else:
                # Partial pass: (e.g., only 'make test-llm-conformance' ran)
                # Show the mixed-level "Gold" summary.
                self._print_gold_certification(terminalreporter, protocol_results, duration)
            return

        # We have actual failures. Show the analysis.
        by_protocol = self._categorize_failures(failed_reports)
        self._print_failure_analysis(terminalreporter, by_protocol, duration)

    def pytest_runtest_logstart(self, nodeid, location):
        """Show protocol being tested for better progress visibility."""
        proto, category = self._categorize_test_by_protocol(nodeid)
        if proto != "other":
            # Could enhance with progress reporting
            pass


# Instantiate the plugin
corpus_protocol_plugin = CorpusProtocolPlugin()


# Pytest hook functions (delegate to plugin instance)
def pytest_sessionstart(session):
    corpus_protocol_plugin.pytest_sessionstart(session)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    corpus_protocol_plugin.pytest_terminal_summary(terminalreporter, exitstatus, config)


def pytest_runtest_logstart(nodeid, location):
    corpus_protocol_plugin.pytest_runtest_logstart(nodeid, location)


# Optional: Add custom markers for better test organization
def pytest_configure(config):
    """Register custom markers for protocol tests."""
    markers = [
        "llm: LLM Protocol V1.0 conformance tests",
        "vector: Vector Protocol V1.0 conformance tests",
        "graph: Graph Protocol V1.0 conformance tests",
        "embedding: Embedding Protocol V1.0 conformance tests",
        "schema: Schema conformance validation tests",
        "golden: Golden wire message validation tests",
        "slow: Tests that take longer to run (skip with -m 'not slow')",
        "conformance: All protocol conformance tests",
    ]

    for marker in markers:
        config.addinivalue_line("markers", marker)