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
import re
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from functools import lru_cache

import pytest


# ---------------------------------------------------------------------------
# Configuration Validation & Performance Optimizations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProtocolConfig:
    """Immutable protocol configuration with validation."""
    name: str
    display_name: str
    conformance_levels: Dict[str, int]
    test_categories: Dict[str, str]
    spec_sections: Dict[str, str]
    error_guidance: Dict[str, Dict[str, Any]]
    
    def validate(self) -> None:
        """Validate protocol configuration for consistency."""
        if not self.display_name:
            raise ValueError(f"Protocol {self.name}: display_name cannot be empty")
        
        required_levels = {"gold", "silver", "development"}
        if not required_levels.issubset(self.conformance_levels.keys()):
            raise ValueError(f"Protocol {self.name}: missing required conformance levels: {required_levels}")
        
        for level_name, threshold in self.conformance_levels.items():
            if not isinstance(threshold, int) or threshold < 0:
                raise ValueError(f"Protocol {self.name}: conformance level {level_name} must be positive integer")
        
        # Validate that all categories have spec sections
        for category in self.test_categories:
            if category not in self.spec_sections:
                raise ValueError(f"Protocol {self.name}: category '{category}' missing spec section mapping")
        
        # Validate error guidance structure
        for category, tests in self.error_guidance.items():
            if category not in self.test_categories:
                raise ValueError(f"Protocol {self.name}: error guidance for unknown category '{category}'")
            for test_name, guidance in tests.items():
                if "quick_fix" not in guidance:
                    raise ValueError(f"Protocol {self.name}: test {test_name} missing quick_fix in error guidance")


class ProtocolRegistry:
    """Central registry for protocol configurations with validation."""
    
    def __init__(self):
        self._protocols: Dict[str, ProtocolConfig] = {}
        self._category_cache: Dict[str, Set[str]] = {}
    
    def register(self, config: ProtocolConfig) -> None:
        """Register and validate a protocol configuration."""
        config.validate()
        self._protocols[config.name] = config
        # Precompute category sets for fast lookups
        self._category_cache[config.name] = set(config.test_categories.keys())
    
    def get(self, protocol: str) -> Optional[ProtocolConfig]:
        """Get protocol configuration."""
        return self._protocols.get(protocol)
    
    def get_category_names(self, protocol: str) -> Set[str]:
        """Get category names for fast membership testing."""
        return self._category_cache.get(protocol, set())
    
    def validate_all(self) -> None:
        """Validate all registered protocols."""
        for protocol in self._protocols.values():
            protocol.validate()


# Initialize global registry
protocol_registry = ProtocolRegistry()


# ---------------------------------------------------------------------------
# Performance-optimized protocol configurations
# ---------------------------------------------------------------------------

# CORRECTED: Updated golden tests from 73 to 78 to match actual test_golden_samples.py
PROTOCOLS_CONFIG = {
    "llm": ProtocolConfig(
        name="llm",
        display_name="LLM Protocol V1.0",
        conformance_levels={"gold": 61, "silver": 49, "development": 31},
        test_categories={
            "wire_contract": "Wire Contract & Routing",
            "core_ops": "Core Operations",
            "message_validation": "Message Validation",
            "sampling_params": "Sampling Parameters",
            "streaming": "Streaming Semantics",
            "error_handling": "Error Handling",
            "capabilities": "Capabilities Discovery",
            "observability": "Observability & Privacy",
            "deadline": "Deadline Semantics",
            "token_counting": "Token Counting",
            "health": "Health Endpoint"
        },
        spec_sections={
            "wire_contract": "§4.1 Wire-First Canonical Form",
            "core_ops": "§8.3 Operations",
            "message_validation": "§8.3 Operations",
            "sampling_params": "§8.3 Operations",
            "streaming": "§8.3 Operations & §4.1.3 Streaming Frames",
            "error_handling": "§8.5 LLM-Specific Errors",
            "capabilities": "§8.4 Model Discovery",
            "observability": "§6.4 Observability Interfaces & §13 Observability and Monitoring",
            "deadline": "§4.3 Deadline Propagation & §6.1 Operation Context",
            "token_counting": "§8.3 Operations",
            "health": "§8.3 Operations"
        },
        error_guidance={
            "wire_contract": {
                "test_wire_envelope_validation": {
                    "error_patterns": {
                        "missing_required_fields": "Wire envelope missing required fields per §4.1",
                        "invalid_field_types": "Field types don't match canonical form requirements"
                    },
                    "quick_fix": "Ensure all wire envelopes include required fields with correct types",
                    "examples": "See §4.1 for wire envelope format and field requirements"
                }
            },
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
                },
                "test_top_p_validation": {
                    "error_patterns": {
                        "invalid_range": "top_p must be between 0.0 and 1.0 per §8.3",
                        "exclusive_range": "top_p=0.0 and top_p=1.0 have special semantics per §8.3.2"
                    },
                    "quick_fix": "Validate top_p range and handle edge cases appropriately",
                    "examples": "See §8.3.2 for top_p semantics and validation rules"
                }
            },
            "core_ops": {
                "test_chat_completion": {
                    "error_patterns": {
                        "invalid_message_roles": "Message roles must conform to §8.3.1 allowed values",
                        "missing_messages": "Request must contain non-empty messages array per §8.3"
                    },
                    "quick_fix": "Validate message structure and role enumeration before processing",
                    "examples": "See §8.3.1 for message format and role requirements"
                }
            }
        }
    ),
    
    "vector": ProtocolConfig(
        name="vector",
        display_name="Vector Protocol V1.0",
        conformance_levels={"gold": 72, "silver": 58, "development": 36},
        test_categories={
            "wire_contract": "Wire Contract & Routing",
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
            "batch_limits": "Batch Size Limits"
        },
        spec_sections={
            "wire_contract": "§4.1 Wire-First Canonical Form",
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
            "batch_limits": "§9.3 Operations & §12.5 Partial Failure Contracts"
        },
        error_guidance={
            "wire_contract": {
                "test_wire_envelope_validation": {
                    "error_patterns": {
                        "missing_required_fields": "Wire envelope missing required fields per §4.1",
                        "invalid_field_types": "Field types don't match canonical form requirements"
                    },
                    "quick_fix": "Ensure all wire envelopes include required fields with correct types",
                    "examples": "See §4.1 for wire envelope format and field requirements"
                }
            },
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
            },
            "batch_limits": {
                "test_batch_size_limits": {
                    "error_patterns": {
                        "batch_too_large": "Batch size exceeds maximum allowed per §9.3",
                        "partial_failure_handling": "Partial failures not properly reported per §12.5"
                    },
                    "quick_fix": "Implement batch size validation and partial success reporting",
                    "examples": "See §9.3 for batch limits and §12.5 for partial failure contracts"
                }
            }
        }
    ),
    
    "graph": ProtocolConfig(
        name="graph",
        display_name="Graph Protocol V1.0",
        conformance_levels={"gold": 68, "silver": 54, "development": 34},
        test_categories={
            "wire_contract": "Wire Contract & Routing",
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
            "health": "Health Endpoint"
        },
        spec_sections={
            "wire_contract": "§4.1 Wire-First Canonical Form",
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
            "health": "§7.6 Health"
        },
        error_guidance={
            "wire_contract": {
                "test_wire_envelope_validation": {
                    "error_patterns": {
                        "missing_required_fields": "Wire envelope missing required fields per §4.1",
                        "invalid_field_types": "Field types don't match canonical form requirements"
                    },
                    "quick_fix": "Ensure all wire envelopes include required fields with correct types",
                    "examples": "See §4.1 for wire envelope format and field requirements"
                }
            },
            "query_ops": {
                "test_cypher_query_validation": {
                    "error_patterns": {
                        "invalid_cypher": "Cypher query syntax validation failed per §7.4.1",
                        "unsupported_clause": "Query uses unsupported Cypher clauses per dialect"
                    },
                    "quick_fix": "Validate Cypher syntax and check supported features in capabilities",
                    "examples": "See §7.4.1 for Cypher dialect requirements and validation"
                }
            },
            "crud_validation": {
                "test_vertex_lifecycle": {
                    "error_patterns": {
                        "duplicate_vertex": "Vertex creation with duplicate ID violated constraints",
                        "invalid_properties": "Vertex properties violate schema constraints"
                    },
                    "quick_fix": "Check vertex ID uniqueness and property schema compliance",
                    "examples": "See §7.3.1 for vertex CRUD operations and constraints"
                }
            }
        }
    ),
    
    "embedding": ProtocolConfig(
        name="embedding",
        display_name="Embedding Protocol V1.0",
        conformance_levels={"gold": 75, "silver": 60, "development": 38},
        test_categories={
            "wire_contract": "Wire Contract & Routing",
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
            "caching": "Caching & Idempotency"
        },
        spec_sections={
            "wire_contract": "§4.1 Wire-First Canonical Form",
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
            "caching": "§11.6 Caching (Implementation Guidance)"
        },
        error_guidance={
            "wire_contract": {
                "test_wire_envelope_validation": {
                    "error_patterns": {
                        "missing_required_fields": "Wire envelope missing required fields per §4.1",
                        "invalid_field_types": "Field types don't match canonical form requirements"
                    },
                    "quick_fix": "Ensure all wire envelopes include required fields with correct types",
                    "examples": "See §4.1 for wire envelope format and field requirements"
                }
            },
            "batch_partial": {
                "test_partial_batch_failures": {
                    "error_patterns": {
                        "inconsistent_reporting": "Partial success counts don't match input batch size",
                        "missing_failure_details": "Failure objects missing required error details per §12.5"
                    },
                    "quick_fix": "Ensure partial success response matches input batch size with proper error indexing",
                    "examples": "See §12.5 for partial failure contract requirements"
                }
            },
            "truncation": {
                "test_auto_truncation": {
                    "error_patterns": {
                        "truncation_not_supported": "Model doesn't support truncation but input exceeds limits",
                        "invalid_truncation_parameter": "truncation parameter not in allowed values"
                    },
                    "quick_fix": "Implement truncation strategy or validate input length against model limits",
                    "examples": "See §10.6 for truncation semantics and parameter validation"
                }
            }
        }
    ),
    
    "schema": ProtocolConfig(
        name="schema",
        display_name="Schema Conformance",
        conformance_levels={"gold": 13, "silver": 10, "development": 7},
        test_categories={
            "schema_loading": "Schema Loading & IDs",
            "file_organization": "File Organization",
            "metaschema_hygiene": "Metaschema & Hygiene",
            "cross_references": "Cross-References",
            "definitions": "Definitions",
            "contract_constants": "Contract & Constants",
            "examples_validation": "Examples Validation",
            "stream_frames": "Stream Frames",
            "performance_metrics": "Performance & Metrics"
        },
        spec_sections={
            "schema_loading": "Schema Meta-Lint Suite - Schema Loading & IDs",
            "file_organization": "Schema Meta-Lint Suite - File Organization",
            "metaschema_hygiene": "Schema Meta-Lint Suite - Metaschema & Hygiene",
            "cross_references": "Schema Meta-Lint Suite - Cross-References",
            "definitions": "Schema Meta-Lint Suite - Definitions",
            "contract_constants": "Schema Meta-Lint Suite - Contract & Constants",
            "examples_validation": "Schema Meta-Lint Suite - Examples Validation",
            "stream_frames": "Schema Meta-Lint Suite - Stream Frames",
            "performance_metrics": "Schema Meta-Lint Suite - Performance & Metrics"
        },
        error_guidance={
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
        }
    ),
    
    "golden": ProtocolConfig(
        name="golden",
        display_name="Golden Wire Validation", 
        conformance_levels={"gold": 78, "silver": 62, "development": 39},
        test_categories={
            "core_validation": "Core Schema Validation",
            "ndjson_stream": "NDJSON Stream Validation",
            "cross_invariants": "Cross-Schema Invariants",
            "version_format": "Schema Version & Format",
            "drift_detection": "Drift Detection",
            "performance_reliability": "Performance & Reliability",
            "component_coverage": "Component Coverage"
        },
        spec_sections={
            "core_validation": "Golden Samples Suite - Core Schema Validation",
            "ndjson_stream": "Golden Samples Suite - NDJSON Stream Validation",
            "cross_invariants": "Golden Samples Suite - Cross-Schema Invariants",
            "version_format": "Schema Version & Format",
            "drift_detection": "Golden Samples Suite - Drift Detection",
            "performance_reliability": "Golden Samples Suite - Performance & Reliability",
            "component_coverage": "Golden Samples Suite - Component Coverage"
        },
        error_guidance={
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
        }
    )
}

# Register all protocols
for protocol_config in PROTOCOLS_CONFIG.values():
    protocol_registry.register(protocol_config)

# Validate all configurations on module load
protocol_registry.validate_all()

# Protocol display names for quick access
PROTOCOL_DISPLAY_NAMES = {proto: config.display_name for proto, config in PROTOCOLS_CONFIG.items()}
PROTOCOLS = list(PROTOCOLS_CONFIG.keys())


# ---------------------------------------------------------------------------
# Performance-optimized adapter system
# ---------------------------------------------------------------------------

# Environment variable for fully-qualified adapter class:
#   CORPUS_ADAPTER="package.module:ClassName"
ADAPTER_ENV = "CORPUS_ADAPTER"
DEFAULT_ADAPTER = "tests.mock.mock_llm_adapter:MockLLMAdapter"
ENDPOINT_ENV = "CORPUS_ENDPOINT"

# Thread-safe caching with validation
_ADAPTER_CLASS: Optional[type] = None
_ADAPTER_SPEC_USED: Optional[str] = None
_ADAPTER_VALIDATED: bool = False


class AdapterValidationError(RuntimeError):
    """Custom exception for adapter validation failures."""
    pass


def _validate_adapter_class(cls: type) -> None:
    """Validate that adapter class meets minimum interface requirements."""
    # REMOVED the restrictive method validation that required all adapters
    # to have generate, embed, and query methods
    
    if not callable(cls):
        raise AdapterValidationError(
            f"Adapter class {cls.__name__} is not callable (missing __init__ or __call__)."
        )
    
    # Optional: check if it's a known protocol adapter type
    # LLM adapters should have complete/stream methods
    # Vector adapters should have embed/query methods
    # Graph adapters should have query methods
    # This is informational only, not enforced


def _load_class_from_spec(spec: str) -> type:
    """
    Load and validate a class from a 'package.module:ClassName' string.
    """
    module_name, _, class_name = spec.partition(':')
    if not module_name or not class_name:
        raise AdapterValidationError(
            f"Invalid adapter spec '{spec}'. Expected 'package.module:ClassName'."
        )

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise AdapterValidationError(
            f"Failed to import adapter module '{module_name}' for spec '{spec}'."
        ) from exc

    try:
        cls = getattr(module, class_name)
    except AttributeError as exc:
        raise AdapterValidationError(
            f"Adapter class '{class_name}' not found in module '{module_name}' "
            f"for spec '{spec}'."
        ) from exc

    # Validate the loaded class
    _validate_adapter_class(cls)
    
    return cls


def _get_adapter_class() -> type:
    """
    Resolve, validate, and cache the adapter class.
    """
    global _ADAPTER_CLASS, _ADAPTER_SPEC_USED, _ADAPTER_VALIDATED

    if _ADAPTER_CLASS is not None and _ADAPTER_VALIDATED:
        return _ADAPTER_CLASS

    spec = os.getenv(ADAPTER_ENV, DEFAULT_ADAPTER)
    _ADAPTER_SPEC_USED = spec
    
    try:
        _ADAPTER_CLASS = _load_class_from_spec(spec)
        _ADAPTER_VALIDATED = True
    except AdapterValidationError:
        # Reset state on validation failure
        _ADAPTER_CLASS = None
        _ADAPTER_SPEC_USED = None
        _ADAPTER_VALIDATED = False
        raise

    return _ADAPTER_CLASS


@pytest.fixture
def adapter():
    """
    Generic, pluggable adapter fixture with enhanced validation.
    """
    Adapter = _get_adapter_class()
    endpoint = os.getenv(ENDPOINT_ENV)

    if endpoint:
        # Enhanced parameter injection with better error reporting
        param_names = ["endpoint", "base_url", "url"]
        attempted_params = []
        
        for kw in param_names:
            attempted_params.append(kw)
            try:
                instance = Adapter(**{kw: endpoint})
                return instance
            except TypeError:
                continue

        # Final fallback with detailed error message
        try:
            return Adapter()
        except TypeError as exc:
            raise AdapterValidationError(
                f"Failed to instantiate adapter '{_ADAPTER_SPEC_USED}' with endpoint. "
                f"Tried parameters: {param_names}. "
                f"Adapter constructor signature may be incompatible."
            ) from exc

    # Local/mock mode
    try:
        return Adapter()
    except TypeError as exc:
        raise AdapterValidationError(
            f"Failed to instantiate adapter '{_ADAPTER_SPEC_USED}' without arguments. "
            f"Ensure your adapter has a no-arg constructor or configure {ENDPOINT_ENV}."
        ) from exc


# ---------------------------------------------------------------------------
# Performance-optimized test categorization
# ---------------------------------------------------------------------------

class TestCategorizer:
    """
    High-performance test categorization with caching and pattern matching.
    """
    
    def __init__(self):
        self._protocol_patterns = self._build_protocol_patterns()
        self._category_patterns = self._build_category_patterns()
        self._cache: Dict[str, Tuple[str, str]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def _build_protocol_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for protocol detection."""
        patterns = {}
        for proto in PROTOCOLS:
            # Match both POSIX and Windows paths
            pattern = re.compile(rf'tests[\\/]{re.escape(proto)}[\\/]', re.IGNORECASE)
            patterns[proto] = pattern
        return patterns
    
    def _build_category_patterns(self) -> Dict[str, Dict[re.Pattern, str]]:
        """Compile regex patterns for category detection."""
        category_patterns = {}
        for proto, config in PROTOCOLS_CONFIG.items():
            proto_patterns = {}
            for category_key, category_name in config.test_categories.items():
                # Create patterns for both category key and display name
                patterns = [
                    re.compile(rf'\b{re.escape(category_key)}\b', re.IGNORECASE),
                    re.compile(rf'\b{re.escape(category_name.lower())}\b', re.IGNORECASE)
                ]
                for pattern in patterns:
                    proto_patterns[pattern] = category_key
            category_patterns[proto] = proto_patterns
        return category_patterns
    
    @lru_cache(maxsize=1000)
    def categorize_test(self, nodeid: str) -> Tuple[str, str]:
        """
        Categorize test by protocol and category with caching.
        """
        nodeid_lower = nodeid.lower()
        
        # Protocol detection
        protocol = "other"
        for proto, pattern in self._protocol_patterns.items():
            if pattern.search(nodeid_lower):
                protocol = proto
                break
        
        if protocol == "other":
            return "other", "unknown"
        
        # Category detection with cached patterns
        category = "unknown"
        proto_patterns = self._category_patterns.get(protocol, {})
        
        for pattern, category_key in proto_patterns.items():
            if pattern.search(nodeid_lower):
                category = category_key
                break
        
        return protocol, category
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics for performance monitoring."""
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_size": len(self._cache)
        }


# Global categorizer instance
test_categorizer = TestCategorizer()


# ---------------------------------------------------------------------------
# Optimized Corpus Protocol Plugin
# ---------------------------------------------------------------------------

class CorpusProtocolPlugin:
    """High-performance pytest plugin for Corpus Protocol conformance reporting."""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.test_reports: Dict[str, List[Any]] = {}
        self.protocol_counts: Dict[str, Dict[str, int]] = {}
        self._protocol_results_cache: Optional[Dict[str, int]] = None
        
    def pytest_sessionstart(self, session):
        """Record session start time and initialize tracking."""
        self.start_time = time.time()
        self.test_reports = {proto: [] for proto in PROTOCOLS}
        self.test_reports["other"] = []
        self.protocol_counts = {proto: {} for proto in PROTOCOLS}
        self._protocol_results_cache = None
        
        # Log performance optimization status
        session.config.option.verbose and print(
            "🔧 Corpus Protocol Plugin: Performance optimizations enabled - "
            "cached test categorization, pre-validated configurations"
        )

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

    def _categorize_failures(self, failed_reports: List) -> Dict[str, Dict[str, List[Any]]]:
        """
        High-performance failure categorization.
        
        Optimized for large test suites by using cached categorization
        and minimizing redundant computations.
        """
        by_protocol: Dict[str, Dict[str, List[Any]]] = {}
        
        # Pre-initialize structure
        for proto in PROTOCOLS:
            by_protocol[proto] = {}
        by_protocol["other"] = {}
        
        # Single-pass categorization with caching
        for rep in failed_reports:
            nodeid = getattr(rep, "nodeid", "") or ""
            proto, category = test_categorizer.categorize_test(nodeid)
            
            if category not in by_protocol[proto]:
                by_protocol[proto][category] = []
            
            by_protocol[proto][category].append(rep)
        
        # Remove empty categories for cleaner output
        return {
            proto: {cat: reports for cat, reports in categories.items() if reports}
            for proto, categories in by_protocol.items()
            if any(categories.values())
        }

    def _calculate_conformance_level(self, protocol: str, passed_count: int) -> Tuple[str, int]:
        """Calculate conformance level and progress to next level."""
        config = protocol_registry.get(protocol)
        if not config:
            return "❌ Unknown Protocol", 0
            
        levels = config.conformance_levels

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
        config = protocol_registry.get(protocol)
        if not config:
            return "See protocol specification"
        return config.spec_sections.get(category, "See protocol specification")

    def _get_error_guidance(self, protocol: str, category: str, test_name: str) -> Dict[str, str]:
        """Get specific error guidance for a test failure."""
        config = protocol_registry.get(protocol)
        if not config:
            return {
                "error_patterns": {},
                "quick_fix": "Review specification section above",
                "examples": "See specification for implementation details"
            }
            
        category_guidance = config.error_guidance.get(category, {})
        test_guidance = category_guidance.get(test_name, {})
        
        # Fallback guidance for uncovered tests
        if not test_guidance:
            test_guidance = {
                "error_patterns": {},
                "quick_fix": f"Review {config.display_name} specification section: {self._get_spec_section(protocol, category)}",
                "examples": f"See {config.display_name} documentation for implementation examples"
            }

        return {
            "error_patterns": test_guidance.get("error_patterns", {}),
            "quick_fix": test_guidance.get("quick_fix", "Review specification section above"),
            "examples": test_guidance.get("examples", "See specification for implementation details")
        }

    def _extract_test_name(self, nodeid: str) -> str:
        """Extract the test function name from nodeid."""
        parts = nodeid.split("::")
        return parts[-1] if len(parts) > 1 else "unknown_test"

    def _print_platinum_certification(self, terminalreporter, counts: Dict[str, int], duration: float):
        """Print Platinum certification summary."""
        total_tests = self._get_total_tests(counts)

        terminalreporter.write_sep("=", "🏆 CORPUS PROTOCOL SUITE - PLATINUM CERTIFIED")
        terminalreporter.write_line(
            f"All {total_tests} conformance tests passed across {len(PROTOCOLS)} test suites"
        )
        terminalreporter.write_line("")

        # Show protocol breakdown
        terminalreporter.write_line("Protocol Conformance Status:")
        for proto in PROTOCOLS:
            config = protocol_registry.get(proto)
            if config:
                level, _ = self._calculate_conformance_level(proto, config.conformance_levels["gold"])
                terminalreporter.write_line(f"  ✅ {config.display_name}: {level}")

        terminalreporter.write_line("")
        terminalreporter.write_line(f"⏱️  Completed in {duration:.2f}s")
        terminalreporter.write_line("🎯 Status: Ready for production deployment")
        terminalreporter.write_line("📚 Specification: All requirements met per Corpus Protocol Suite V1.0")
        
        # Performance stats
        cache_stats = test_categorizer.get_cache_stats()
        terminalreporter.write_line(f"🔧 Performance: {cache_stats['cache_hits']} cache hits, {cache_stats['cache_misses']} misses")

    def _print_gold_certification(self, terminalreporter, protocol_results: Dict[str, int], duration: float):
        """Print Gold certification summary with progress to Platinum."""
        terminalreporter.write_sep("=", "🥇 CORPUS PROTOCOL SUITE - GOLD CERTIFIED")

        terminalreporter.write_line("Protocol Conformance Status:")
        platinum_ready = True

        for proto in PROTOCOLS:
            config = protocol_registry.get(proto)
            if not config:
                continue
                
            passed = protocol_results.get(proto, 0)
            level, needed = self._calculate_conformance_level(proto, passed)

            if "Gold" in level:
                terminalreporter.write_line(f"  ✅ {config.display_name}: {level}")
            else:
                platinum_ready = False
                terminalreporter.write_line(f"  ⚠️  {config.display_name}: {level} ({needed} tests to Gold)")

        terminalreporter.write_line("")
        if platinum_ready:
            terminalreporter.write_line("🎯 All protocols at Gold level - Platinum certification available!")
        else:
            terminalreporter.write_line("🎯 Focus on protocols below Gold level for Platinum certification")

        terminalreporter.write_line(f"⏱️  Completed in {duration:.2f}s")
        terminalreporter.write_line("📚 Review CONFORMANCE.md for detailed test-to-spec mapping")

    def _print_failure_analysis(self, terminalreporter, by_protocol: Dict[str, Dict[str, List[Any]]], duration: float):
        """Print detailed failure analysis with actionable guidance."""
        terminalreporter.write_sep("=", "❌ CORPUS PROTOCOL CONFORMANCE ANALYSIS")

        total_failures = 0
        for proto_failures in by_protocol.values():
            for reports_list in proto_failures.values():
                total_failures += len(reports_list)

        terminalreporter.write_line(f"Found {total_failures} conformance issue(s) across protocols:")
        terminalreporter.write_line("")

        # Show failures by protocol and category with specific guidance
        for proto, categories in by_protocol.items():
            if not categories:
                continue

            config = protocol_registry.get(proto)
            display_name = config.display_name if config else proto.upper()
            terminalreporter.write_line(f"{display_name}:")

            for category, reports_list in categories.items():
                count = len(reports_list)
                category_name = "Unknown Category"
                if config:
                    category_name = config.test_categories.get(category, category.replace('_', ' ').title())
                spec_section = self._get_spec_section(proto, category)

                terminalreporter.write_line(f"  ❌ {category_name}: {count} failure(s)")
                terminalreporter.write_line(f"      Specification: {spec_section}")

                # Show specific guidance for each failed test
                for rep in reports_list:
                    test_name = self._extract_test_name(rep.nodeid)
                    guidance = self._get_error_guidance(proto, category, test_name)

                    terminalreporter.write_line(f"      Test: {test_name}")
                    terminalreporter.write_line(f"      Quick fix: {guidance['quick_fix']}")

                    # Enhanced error pattern matching
                    error_patterns = guidance.get("error_patterns", {})
                    if error_patterns:
                        error_msg = str(getattr(rep, 'longrepr', '')).lower()
                        matched_patterns = []
                        for pattern_key, pattern_desc in error_patterns.items():
                            if pattern_key.lower() in error_msg:
                                matched_patterns.append(pattern_desc)
                        
                        if matched_patterns:
                            terminalreporter.write_line(f"      Detected: {', '.join(matched_patterns)}")

            terminalreporter.write_line("")

        # Certification impact
        terminalreporter.write_line("Certification Impact:")
        failing_protocols = [p for p, cats in by_protocol.items() if cats and p != "other"]

        if failing_protocols:
            terminalreporter.write_line("  ⚠️  Platinum certification blocked by failures in:")
            for proto in failing_protocols:
                config = protocol_registry.get(proto)
                display_name = config.display_name if config else proto.upper()
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
        """Collect passed test counts per protocol with caching."""
        if self._protocol_results_cache is not None:
            return self._protocol_results_cache

        protocol_results = {proto: 0 for proto in PROTOCOLS}

        # Count passed tests per protocol using cached categorization
        for test_report in terminalreporter.stats.get("passed", []):
            nodeid = getattr(test_report, "nodeid", "") or ""
            proto, _ = test_categorizer.categorize_test(nodeid)
            if proto in protocol_results:
                protocol_results[proto] += 1

        self._protocol_results_cache = protocol_results
        return protocol_results

    def _is_platinum_certified(self, protocol_results: Dict[str, int]) -> bool:
        """Check if all protocols meet Platinum certification requirements."""
        for proto in PROTOCOLS:
            config = protocol_registry.get(proto)
            if not config:
                return False
            passed = protocol_results.get(proto, 0)
            if passed < config.conformance_levels["gold"]:
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
            if self._is_platinum_certified(protocol_results):
                self._print_platinum_certification(terminalreporter, counts, duration)
            else:
                self._print_gold_certification(terminalreporter, protocol_results, duration)
            return

        # We have actual failures. Show the analysis.
        by_protocol = self._categorize_failures(failed_reports)
        self._print_failure_analysis(terminalreporter, by_protocol, duration)

    def pytest_runtest_logstart(self, nodeid, location):
        """Show protocol being tested for better progress visibility."""
        proto, category = test_categorizer.categorize_test(nodeid)
        if proto != "other":
            # Enhanced progress reporting could be added here
            pass


# Instantiate the plugin
corpus_protocol_plugin = CorpusProtocolPlugin()


# Pytest hook functions
def pytest_sessionstart(session):
    corpus_protocol_plugin.pytest_sessionstart(session)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    corpus_protocol_plugin.pytest_terminal_summary(terminalreporter, exitstatus, config)


def pytest_runtest_logstart(nodeid, location):
    corpus_protocol_plugin.pytest_runtest_logstart(nodeid, location)


# Enhanced configuration with validation
def pytest_configure(config):
    """Register custom markers and validate plugin configuration."""
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
    
    # Validate adapter configuration early
    if config.option.verbose:
        try:
            _get_adapter_class()
            print("✅ Adapter configuration validated successfully")
        except AdapterValidationError as e:
            print(f"❌ Adapter configuration error: {e}")
