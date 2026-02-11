# SPDX-License-Identifier: Apache-2.0
"""
CORPUS Protocol Conformance Plugin for pytest.

This module implements the official certification framework for CORPUS Protocol
implementations. It is both test infrastructure and a product component - the
certification tiers, scoring policies, and conformance criteria defined here
are normative for CORPUS ecosystem compliance.

Capabilities:
    - Dynamic certification scoring (Platinum/Gold/Silver/Development)
    - Per-protocol conformance tracking across 11 protocol suites
    - Actionable error guidance with spec section references
    - Machine-readable JSON export for CI/CD integration
    - Pluggable adapter system for testing any CORPUS implementation

Certification Tiers:
    - Platinum: 100% passing across all protocols (production-ready)
    - Gold: 100% passing within a single protocol
    - Silver: ≥80% passing (integration-ready)
    - Development: ≥50% passing (early development)

Scoring Policies:
    - Default: skipped/xfailed tests excluded from denominator
    - Strict (CORPUS_STRICT=1): all collected tests count toward score

Environment Variables:
    CORPUS_ADAPTER       Adapter class spec (package.module:ClassName)
    CORPUS_ENDPOINT      Optional endpoint URL for adapter instantiation
    CORPUS_STRICT        Strict scoring mode (skips/xfails count against you)
    CORPUS_MAX_FAILURES  Limit failure output per category (prevents log spam)
    CORPUS_REPORT_JSON   Write JSON summary to this path
    CORPUS_REPORT_DIR    Write summary.json to this directory
    CORPUS_PLAIN_OUTPUT  Disable emoji output for CI environments
"""

# SQLite workaround for ChromaDB/CrewAI compatibility
# ChromaDB requires SQLite >= 3.35.0, but system may have older version
# Use pysqlite3-binary which includes a newer SQLite version
import sys
try:
    import pysqlite3
    # Override sqlite3 module with pysqlite3 before any imports
    sys.modules['sqlite3'] = sys.modules['pysqlite3']
except ImportError:
    # If pysqlite3 not available, use system sqlite3
    # (some tests may skip due to version requirements)
    pass

import os
import time
import math
import importlib
import re
import inspect
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set, Iterable
from dataclasses import dataclass

import pytest


# ---------------------------------------------------------------------------
# Configuration Validation & Performance Optimizations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProtocolConfig:
    """Immutable protocol configuration with validation."""
    name: str
    display_name: str
    # Optional reference/baseline values only (not used for scoring).
    # Useful for docs, historical tracking, or sanity checks.
    conformance_levels: Optional[Dict[str, int]]
    test_categories: Dict[str, str]
    spec_sections: Dict[str, str]
    error_guidance: Dict[str, Dict[str, Any]]

    def validate(self) -> None:
        """Validate protocol configuration for consistency."""
        if not self.display_name:
            raise ValueError(f"Protocol {self.name}: display_name cannot be empty")

        # conformance_levels are optional and not authoritative for scoring.
        # If present, validate shape (non-negative ints), but do not require keys.
        if self.conformance_levels is not None:
            for level_name, threshold in self.conformance_levels.items():
                if not isinstance(threshold, int) or threshold < 0:
                    raise ValueError(
                        f"Protocol {self.name}: conformance level {level_name} must be non-negative integer"
                    )

        # Validate that all categories have spec sections
        for category in self.test_categories:
            if category not in self.spec_sections:
                raise ValueError(
                    f"Protocol {self.name}: category '{category}' missing spec section mapping"
                )

        # Validate error guidance structure
        for category, tests in self.error_guidance.items():
            if category not in self.test_categories:
                raise ValueError(
                    f"Protocol {self.name}: error guidance for unknown category '{category}'"
                )
            for test_name, guidance in tests.items():
                if "quick_fix" not in guidance:
                    raise ValueError(
                        f"Protocol {self.name}: test {test_name} missing quick_fix in error guidance"
                    )


class ProtocolRegistry:
    """Central registry for protocol configurations with validation."""

    def __init__(self):
        self._protocols: Dict[str, ProtocolConfig] = {}
        self._category_cache: Dict[str, Set[str]] = {}

    def register(self, config: ProtocolConfig) -> None:
        """Register and validate a protocol configuration."""
        config.validate()
        self._protocols[config.name] = config
        self._category_cache[config.name] = set(config.test_categories.keys())

    def get(self, protocol: str) -> Optional[ProtocolConfig]:
        return self._protocols.get(protocol)

    def get_category_names(self, protocol: str) -> Set[str]:
        return self._category_cache.get(protocol, set())

    def validate_all(self) -> None:
        for protocol in self._protocols.values():
            protocol.validate()


protocol_registry = ProtocolRegistry()


# ---------------------------------------------------------------------------
# Performance-optimized protocol configurations
# ---------------------------------------------------------------------------

PROTOCOLS_CONFIG: Dict[str, ProtocolConfig] = {
    "llm": ProtocolConfig(
        name="llm",
        display_name="LLM Protocol V1.0",
        conformance_levels={"gold": 132, "silver": 106, "development": 66},
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
            "health": "Health Endpoint",
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
            "health": "§8.3 Operations",
        },
        error_guidance={
            "wire_contract": {
                "test_wire_envelope_validation": {
                    "error_patterns": {
                        "missing_required_fields": "Wire envelope missing required fields per §4.1",
                        "invalid_field_types": (
                            "Field types don't match canonical form requirements "
                            "(e.g., ctx.deadline_ms is integer|null with minimum 0)"
                        ),
                    },
                    "quick_fix": "Ensure all wire envelopes include required fields with correct types",
                    "examples": "See §4.1 for wire envelope format and field requirements",
                }
            },
            "streaming": {
                "test_stream_finalization": {
                    "error_patterns": {
                        "missing_final_chunk": (
                            "Ensure stream terminates with a terminal condition per §4.1.3 "
                            "(either a final streaming chunk with chunk.is_final=true or an error envelope)"
                        ),
                        "premature_close": (
                            "Connection must remain open until the terminal condition per §7.3.2 "
                            "(final chunk is_final=true or an error envelope)"
                        ),
                        "chunk_format": (
                            "Each streaming success frame must use the streaming success envelope "
                            "and code='STREAMING' per §4.1.3"
                        ),
                    },
                    "quick_fix": (
                        "Ensure streams terminate with exactly one terminal condition: "
                        "either a streaming success frame whose chunk.is_final is true, "
                        "or a standard error envelope. No frames after the terminal condition."
                    ),
                    "examples": (
                        "See §4.1.3 for streaming success envelope format (code='STREAMING' + chunk) "
                        "and §7.3.2 for terminal condition rules (chunk.is_final=true OR error envelope)."
                    ),
                }
            },
            "sampling_params": {
                "test_temperature_validation": {
                    "error_patterns": {
                        "invalid_range": "Temperature must be between 0.0 and 2.0 per §8.3",
                        "type_error": "Temperature must be float, not string per §4.1 Numeric Types",
                    },
                    "quick_fix": "Clamp temperature values to valid range [0.0, 2.0] and ensure numeric types",
                    "examples": "See §8.3 for parameter validation and §4.1 for numeric type rules",
                },
                "test_top_p_validation": {
                    "error_patterns": {
                        "invalid_range": "top_p must be between 0.0 and 1.0 per §8.3",
                        "exclusive_range": "top_p=0.0 and top_p=1.0 have special semantics per §8.3.2",
                    },
                    "quick_fix": "Validate top_p range and handle edge cases appropriately",
                    "examples": "See §8.3.2 for top_p semantics and validation rules",
                },
            },
            "core_ops": {
                "test_chat_completion": {
                    "error_patterns": {
                        "invalid_message_roles": (
                            "Message roles should follow recommended values per §8.3.1 "
                            "(schema allows any string, but interoperability expects common roles)."
                        ),
                        "missing_messages": "Request must contain non-empty messages array per §8.3",
                    },
                    "quick_fix": "Validate message structure and use interoperable role values before processing",
                    "examples": "See §8.3.1 for message format and role guidance",
                }
            },
        },
    ),
    "vector": ProtocolConfig(
        name="vector",
        display_name="Vector Protocol V1.0",
        conformance_levels={"gold": 108, "silver": 87, "development": 54},
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
            "batch_limits": "Batch Size Limits",
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
            "batch_limits": "§9.3 Operations & §12.5 Partial Failure Contracts",
        },
        error_guidance={
            "wire_contract": {
                "test_wire_envelope_validation": {
                    "error_patterns": {
                        "missing_required_fields": "Wire envelope missing required fields per §4.1",
                        "invalid_field_types": (
                            "Field types don't match canonical form requirements "
                            "(e.g., ctx.deadline_ms is integer|null with minimum 0)"
                        ),
                    },
                    "quick_fix": "Ensure all wire envelopes include required fields with correct types",
                    "examples": "See §4.1 for wire envelope format and field requirements",
                }
            },
            "namespace": {
                "test_namespace_isolation": {
                    "error_patterns": {
                        "cross_namespace_leak": "Data must be strictly isolated per §14.1 Tenant Isolation",
                        "invalid_namespace": "Namespace must follow §9.3 Operations requirements",
                    },
                    "quick_fix": "Validate namespace format and enforce isolation at storage layer",
                    "examples": "See §9.3 for namespace operations and §14.1 for tenant isolation requirements",
                }
            },
            "dimension_validation": {
                "test_dimension_mismatch": {
                    "error_patterns": {
                        "dimension_mismatch": (
                            "Adapters should detect and report vector dimension mismatches per §9.5 "
                            "(schema alone may not encode all backend/index dimension constraints)."
                        ),
                        "invalid_dimension": "Dimensions must be positive integers per §4.1 Numeric Types",
                    },
                    "quick_fix": "Validate vector dimensions before upsert operations",
                    "examples": "See §9.5 for dimension handling and §4.1 for numeric validation",
                }
            },
            "batch_limits": {
                "test_batch_size_limits": {
                    "error_patterns": {
                        "batch_too_large": "Batch size exceeds maximum allowed per §9.3",
                        "partial_failure_handling": "Partial failures not properly reported per §12.5",
                    },
                    "quick_fix": "Implement batch size validation and partial success reporting",
                    "examples": "See §9.3 for batch limits and §12.5 for partial failure contracts",
                }
            },
        },
    ),
    "graph": ProtocolConfig(
        name="graph",
        display_name="Graph Protocol V1.0",
        conformance_levels={"gold": 99, "silver": 80, "development": 50},
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
            "health": "Health Endpoint",
        },
        spec_sections={
            "wire_contract": "§4.1 Wire-First Canonical Form",
            "core_ops": "§7.3 Operations",
            "crud_validation": "§7.3.1 Node/Edge CRUD",
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
        },
        error_guidance={
            "wire_contract": {
                "test_wire_envelope_validation": {
                    "error_patterns": {
                        "missing_required_fields": "Wire envelope missing required fields per §4.1",
                        "invalid_field_types": (
                            "Field types don't match canonical form requirements "
                            "(e.g., ctx.deadline_ms is integer|null with minimum 0)"
                        ),
                    },
                    "quick_fix": "Ensure all wire envelopes include required fields with correct types",
                    "examples": "See §4.1 for wire envelope format and field requirements",
                }
            },
            "query_ops": {
                "test_cypher_query_validation": {
                    "error_patterns": {
                        "invalid_cypher": "Cypher query syntax validation failed per §7.4.1",
                        "unsupported_clause": "Query uses unsupported Cypher clauses per dialect",
                    },
                    "quick_fix": "Validate Cypher syntax and check supported features in capabilities",
                    "examples": "See §7.4.1 for Cypher dialect requirements and validation",
                }
            },
            "crud_validation": {
                "test_node_lifecycle": {
                    "error_patterns": {
                        "duplicate_node": "Node creation with duplicate ID violated constraints",
                        "invalid_properties": "Node properties violate schema constraints",
                    },
                    "quick_fix": "Check node ID uniqueness and property schema compliance",
                    "examples": "See §7.3.1 for node CRUD operations and constraints",
                }
            },
        },
    ),
    "embedding": ProtocolConfig(
        name="embedding",
        display_name="Embedding Protocol V1.0",
        conformance_levels={"gold": 135, "silver": 108, "development": 68},
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
            "caching": "Caching & Idempotency",
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
            "caching": "§11.6 Caching (Implementation Guidance)",
        },
        error_guidance={
            "wire_contract": {
                "test_wire_envelope_validation": {
                    "error_patterns": {
                        "missing_required_fields": "Wire envelope missing required fields per §4.1",
                        "invalid_field_types": (
                            "Field types don't match canonical form requirements "
                            "(e.g., ctx.deadline_ms is integer|null with minimum 0)"
                        ),
                    },
                    "quick_fix": "Ensure all wire envelopes include required fields with correct types",
                    "examples": "See §4.1 for wire envelope format and field requirements",
                }
            },
            "batch_partial": {
                "test_partial_batch_failures": {
                    "error_patterns": {
                        "inconsistent_reporting": "Partial success counts don't match input batch size",
                        "missing_failure_details": "Failure objects missing required error details per §12.5",
                    },
                    "quick_fix": "Ensure partial success response matches input batch size with proper error indexing",
                    "examples": "See §12.5 for partial failure contract requirements",
                }
            },
            "truncation": {
                "test_auto_truncation": {
                    "error_patterns": {
                        "truncation_not_supported": "Model doesn't support truncation but input exceeds limits",
                        "invalid_truncation_parameter": "truncation parameter not in allowed values",
                    },
                    "quick_fix": "Implement truncation strategy or validate input length against model limits",
                    "examples": "See §10.6 for truncation semantics and parameter validation",
                }
            },
        },
    ),
    "wire": ProtocolConfig(
        name="wire",
        display_name="Wire Request Conformance Suite",
        conformance_levels={"gold": 76, "silver": 61, "development": 38},
        test_categories={"wire": "Wire Request Envelope Conformance"},
        spec_sections={"wire": "Wire Request Conformance Suite (tests/live/test_wire_conformance.py)"},
        error_guidance={
            "wire": {
                "test_wire_request_envelope": {
                    "error_patterns": {
                        "schema validation failed": (
                            "Envelope does not validate against the JSON Schemas resolved by the schema registry"
                        ),
                        "validationerror": (
                            "Envelope fails structural or args validation in tests.live.wire_validators"
                        ),
                    },
                    "quick_fix": (
                        "Ensure your adapter's build_*_envelope methods produce envelopes that match the "
                        "CORPUS Protocol wire envelope schemas and the constraints enforced in "
                        "tests/live/wire_validators.py."
                    ),
                    "examples": (
                        "See tests/live/wire_cases.py for the canonical WireRequestCase definitions and "
                        "tests/live/wire_validators.py for the validation pipeline."
                    ),
                },
            }
        },
    ),
    "schema": ProtocolConfig(
        name="schema",
        display_name="CORPUS Schema Conformance Suite",
        conformance_levels={"gold": 199, "silver": 160, "development": 100},
        test_categories={
            "schema_loading": "Schema Loading & IDs",
            "file_organization": "File Organization",
            "metaschema_hygiene": "Metaschema & Hygiene",
            "cross_references": "Cross-References",
            "definitions": "Definitions",
            "contract_constants": "Contract & Constants",
            "examples_validation": "Examples Validation",
            "stream_frames": "Stream Frames",
            "performance_metrics": "Performance & Metrics",
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
            "performance_metrics": "Schema Meta-Lint Suite - Performance & Metrics",
        },
        error_guidance={
            "schema_loading": {
                "test_schema_loading": {
                    "error_patterns": {
                        "invalid_schema": "Schema file failed to load or parse",
                        "missing_schema": "Required $schema field missing or invalid",
                    },
                    "quick_fix": (
                        "Ensure all schema files are valid JSON and include "
                        "$schema: 'https://json-schema.org/draft/2020-12/schema'"
                    ),
                    "examples": "See SCHEMA_CONFORMANCE.md - Schema Loading & IDs section",
                },
                "test_unique_ids": {
                    "error_patterns": {
                        "duplicate_id": "Duplicate $id found across schema files",
                        "invalid_id_format": "$id does not follow https://corpusos.com/schemas/ format",
                    },
                    "quick_fix": (
                        "Ensure each schema has unique $id following convention: "
                        "https://corpusos.com/schemas/<component>/<file>.json"
                    ),
                    "examples": "See SCHEMA_CONFORMANCE.md - $id hygiene requirements",
                },
            },
            "metaschema_hygiene": {
                "test_metaschema_compliance": {
                    "error_patterns": {
                        "draft_2020_12_violation": "Schema violates JSON Schema Draft 2020-12",
                        "invalid_keywords": "Unknown or invalid JSON Schema keywords used",
                    },
                    "quick_fix": (
                        "Validate schema against Draft 2020-12 metaschema and remove unsupported keywords"
                    ),
                    "examples": "See SCHEMA_CONFORMANCE.md - Metaschema & Hygiene section",
                },
                "test_regex_patterns": {
                    "error_patterns": {
                        "invalid_regex": "Regular expression pattern does not compile",
                        "unsupported_regex_flags": "Regex uses unsupported flags",
                    },
                    "quick_fix": "Fix regex patterns to use supported ECMA 262 syntax without flags",
                    "examples": "See SCHEMA_CONFORMANCE.md - Pattern hygiene requirements",
                },
            },
            "cross_references": {
                "test_ref_resolution": {
                    "error_patterns": {
                        "unresolved_ref": "$ref cannot be resolved to known schema $id",
                        "invalid_fragment": "Fragment (#/definitions/...) points to non-existent definition",
                    },
                    "quick_fix": "Ensure all $ref values point to valid $ids or internal fragments",
                    "examples": "See SCHEMA_CONFORMANCE.md - Cross-References section",
                }
            },
        },
    ),
    "embedding_frameworks": ProtocolConfig(
        name="embedding_frameworks",
        display_name="Embedding Framework Adapters V1.0",
        conformance_levels={"gold": 418, "silver": 335, "development": 209},
        test_categories={
            "framework_specific": "Framework-Specific Adapters",
            "contract_interface": "Cross-Framework Interface Conformance",
            "contract_shapes": "Cross-Framework Shape & Batching",
            "contract_context": "Cross-Framework Context & Error Handling",
            "registry_infra": "Registry Infrastructure",
            "robustness": "Robustness & Evil Backend Tests",
        },
        spec_sections={
            "framework_specific": "§10.3 Operations + Framework Integration",
            "contract_interface": "§10.3, §10.6, §7.2",
            "contract_shapes": "§10.6, §12.5",
            "contract_context": "§6.3, §13, §10.4",
            "registry_infra": "§6.1 Framework Registration",
            "robustness": "§6.3, §12.1, §12.5",
        },
        error_guidance={
            "framework_specific": {
                "test_constructor_rejects_adapter_without_embed": {
                    "error_patterns": {
                        "missing_embed_method": "Corpus adapter must implement embed() method per §10.3",
                        "invalid_adapter_type": "Adapter doesn't conform to EmbeddingProtocolV1 interface",
                    },
                    "quick_fix": "Ensure corpus adapter implements the required EmbeddingProtocolV1 interface",
                    "examples": "See §10.3 for required embedding protocol methods",
                }
            },
            "contract_interface": {
                "test_sync_embedding_interface_conformance": {
                    "error_patterns": {
                        "missing_method": "Framework adapter missing required embedding methods",
                        "signature_mismatch": "Method signatures don't match framework expectations",
                    },
                    "quick_fix": "Ensure all required sync/async embedding methods are implemented with correct signatures",
                    "examples": "See test_contract_interface_conformance.py for interface requirements",
                }
            },
            "contract_shapes": {
                "test_batch_output_row_count_matches_input_length": {
                    "error_patterns": {
                        "row_count_mismatch": "Batch embedding returned wrong number of rows",
                        "shape_validation_failed": "Embedding matrix shape doesn't match input length",
                    },
                    "quick_fix": "Ensure embed() returns exactly N vectors for N input texts",
                    "examples": "See test_contract_shapes_and_batching.py for shape validation details",
                }
            },
            "contract_context": {
                "test_error_context_is_attached_on_sync_batch_failure": {
                    "error_patterns": {
                        "missing_error_context": "Errors not decorated with framework/operation metadata",
                        "context_attachment_failed": "attach_context() not called on error path",
                    },
                    "quick_fix": "Add @error_context decorator to all public embedding methods",
                    "examples": "See §6.3 for error context requirements",
                }
            },
        },
    ),
    "graph_frameworks": ProtocolConfig(
        name="graph_frameworks",
        display_name="Graph Framework Adapters V1.0",
        conformance_levels={"gold": 574, "silver": 460, "development": 287},
        test_categories={
            "framework_specific": "Framework-Specific Graph Adapters",
            "contract_interface": "Cross-Framework Interface Conformance",
            "contract_shapes": "Cross-Framework Shape & Batch Semantics",
            "contract_context": "Cross-Framework Context & Error Handling",
            "registry_infra": "Registry Infrastructure",
            "robustness": "Robustness & Evil Backend Tests",
        },
        spec_sections={
            "framework_specific": "§7.3 Operations + Framework Integration",
            "contract_interface": "§7.3, §7.2",
            "contract_shapes": "§7.3.3, §12.5",
            "contract_context": "§6.3, §13",
            "registry_infra": "§6.1 Framework Registration",
            "robustness": "§6.3, §12.1, §12.5",
        },
        error_guidance={
            "framework_specific": {
                "test_context_manager_closes_underlying_graph_adapter": {
                    "error_patterns": {
                        "missing_close": "Underlying graph adapter.close()/aclose() is never called",
                        "resource_leak": "Connections or sessions remain open after context manager exits",
                    },
                    "quick_fix": (
                        "Ensure your framework adapter implements __enter__/__exit__ (and async variants) "
                        "and delegates cleanup to the underlying Corpus graph adapter."
                    ),
                    "examples": (
                        "See tests/frameworks/graph/test_*_graph_adapter.py "
                        "for context manager expectations."
                    ),
                }
            },
            "contract_shapes": {
                "test_batch_result_length_matches_ops_when_supported": {
                    "error_patterns": {
                        "length_mismatch": "Batch() returned a different number of results than operations submitted",
                        "silent_drop": "Some operations appear to be silently dropped from the response",
                    },
                    "quick_fix": (
                        "Ensure that for N batch operations your adapter returns exactly N results, "
                        "in the same order, even when some operations fail."
                    ),
                    "examples": (
                        "See tests/frameworks/graph/test_contract_shapes_and_batching.py "
                        "and §7.3.3 + §12.5 for batch semantics."
                    ),
                },
                "test_bulk_vertices_result_type_stable_when_supported": {
                    "error_patterns": {
                        "type_instability": "bulk_vertices() result type changes across calls",
                        "wrong_shape": "Result does not match the expected vertex collection shape",
                    },
                    "quick_fix": (
                        "Normalize bulk_vertices() return type to a consistent collection "
                        "(e.g. list of vertex objects) and keep it stable across calls."
                    ),
                    "examples": (
                        "See tests/frameworks/graph/test_contract_shapes_and_batching.py "
                        "for expected bulk_vertices() behavior."
                    ),
                },
            },
            "contract_context": {
                "test_error_context_is_attached_on_sync_query_failure": {
                    "error_patterns": {
                        "missing_error_context": "Exceptions raised from query() lack framework/operation metadata",
                        "lost_context": "Framework-specific context (config, conversation, task) is not attached on error",
                    },
                    "quick_fix": (
                        "Wrap public methods with the @error_context decorator and ensure that your "
                        "_build_ctx() logic attaches framework and operation metadata to the OperationContext."
                    ),
                    "examples": (
                        "See tests/frameworks/graph/test_contract_context_and_error_context.py "
                        "and §6.3 + §13 for error context requirements."
                    ),
                }
            },
            "robustness": {
                "test_wrong_batch_length_from_backend_causes_error_or_obvious_mismatch": {
                    "error_patterns": {
                        "batch_mismatch_ignored": "Adapter passes through backend batch length mismatches silently",
                        "no_validation": "No validation of translator/backend batch result length vs input ops",
                    },
                    "quick_fix": (
                        "Validate that the backend/translator returns the same number of batch results as "
                        "operations submitted; raise a clear error (with context) if they differ."
                    ),
                    "examples": (
                        "See tests/frameworks/graph/test_with_mock_backends.py and §12.5 "
                        "for partial failure and batch validation guidance."
                    ),
                },
                "test_invalid_backend_result_causes_errors_for_sync_query": {
                    "error_patterns": {
                        "invalid_result_shape": "Adapter accepts backend results of the wrong type/shape",
                        "silent_accept": "Adapter does not surface schema/shape issues from backends",
                    },
                    "quick_fix": (
                        "Add strict validation layers around backend/translator output, ensuring it matches "
                        "the expected Graph Protocol result schema and raising a descriptive error otherwise."
                    ),
                    "examples": (
                        "See tests/frameworks/graph/test_with_mock_backends.py and §6.3 + §12.1 "
                        "for robustness expectations."
                    ),
                },
            },
        },
    ),
    "llm_frameworks": ProtocolConfig(
        name="llm_frameworks",
        display_name="LLM Framework Adapters V1.0",
        conformance_levels={"gold": 624, "silver": 500, "development": 312},
        test_categories={
            "framework_specific": "Framework-Specific LLM Adapters",
            "contract_interface": "Cross-Framework Interface Conformance",
            "contract_shapes": "Cross-Framework Shape & Batching",
            "contract_context": "Cross-Framework Context & Error Handling",
            "registry_infra": "Registry Infrastructure",
            "robustness": "Robustness & Evil Backend Tests",
        },
        spec_sections={
            "framework_specific": "§8.3 Operations + Framework Integration",
            "contract_interface": "§8.3, §8.6, §7.2",
            "contract_shapes": "§8.3, §12.5",
            "contract_context": "§6.3, §13, §8.5",
            "registry_infra": "§6.1 Framework Registration",
            "robustness": "§6.3, §12.1, §12.5",
        },
        error_guidance={
            "contract_interface": {
                "test_sync_llm_interface_conformance": {
                    "error_patterns": {
                        "missing_method": "Framework adapter missing required LLM methods (e.g. chat, completion).",
                        "signature_mismatch": "LLM method signatures don't match framework expectations.",
                    },
                    "quick_fix": "Ensure all required sync/async LLM methods are implemented with correct signatures.",
                    "examples": "See tests/frameworks/llm/test_contract_interface_conformance.py for requirements.",
                }
            },
            "contract_shapes": {
                "test_batch_output_length_matches_input_length": {
                    "error_patterns": {
                        "length_mismatch": "Batch LLM call returned a different number of results than requests submitted.",
                    },
                    "quick_fix": "Ensure batch APIs return exactly N results for N inputs, preserving order.",
                    "examples": "See tests/frameworks/llm/test_contract_shapes_and_batching.py for details.",
                }
            },
            "contract_context": {
                "test_error_context_is_attached_on_sync_batch_failure": {
                    "error_patterns": {
                        "missing_error_context": "Exceptions from LLM methods lack framework/operation metadata.",
                    },
                    "quick_fix": "Wrap public LLM entrypoints with @error_context and attach operation metadata.",
                    "examples": "See §6.3 for error context and tests/frameworks/llm/test_contract_context_and_error_context.py.",
                }
            },
        },
    ),
    "vector_frameworks": ProtocolConfig(
        name="vector_frameworks",
        display_name="Vector Framework Adapters V1.0",
        conformance_levels={"gold": 958, "silver": 767, "development": 479},
        test_categories={
            "framework_specific": "Framework-Specific Vector Adapters",
            "contract_interface": "Cross-Framework Interface Conformance",
            "contract_shapes": "Cross-Framework Shape & Batch Semantics",
            "contract_context": "Cross-Framework Context & Error Handling",
            "registry_infra": "Registry Infrastructure",
            "robustness": "Robustness & Evil Backend Tests",
        },
        spec_sections={
            "framework_specific": "§9.3 Operations + Framework Integration",
            "contract_interface": "§9.3, §7.2",
            "contract_shapes": "§9.3, §12.5",
            "contract_context": "§6.3, §13, §9.5",
            "registry_infra": "§6.1 Framework Registration",
            "robustness": "§6.3, §12.1, §12.5",
        },
        error_guidance={
            "contract_shapes": {
                "test_batch_result_length_matches_input_length": {
                    "error_patterns": {
                        "length_mismatch": "Batch() returned a different number of results than vectors submitted.",
                    },
                    "quick_fix": "Ensure that for N input vectors you return exactly N results, in order.",
                    "examples": "See tests/frameworks/vector/test_contract_shapes_and_batching.py.",
                }
            },
            "contract_interface": {
                "test_sync_vector_interface_conformance": {
                    "error_patterns": {
                        "missing_method": "Framework adapter missing required vector index methods.",
                    },
                    "quick_fix": "Implement all required sync/async vector operations with correct signatures.",
                    "examples": "See tests/frameworks/vector/test_contract_interface_conformance.py.",
                }
            },
        },
    ),
}

for protocol_config in PROTOCOLS_CONFIG.values():
    protocol_registry.register(protocol_config)
protocol_registry.validate_all()

PROTOCOL_DISPLAY_NAMES = {proto: config.display_name for proto, config in PROTOCOLS_CONFIG.items()}

PROTOCOLS: List[str] = [
    "llm",
    "vector",
    "graph",
    "embedding",
    "llm_frameworks",
    "vector_frameworks",
    "embedding_frameworks",
    "graph_frameworks",
    "wire",
    "schema",
    "golden",
]
CONFIG_PROTOCOLS = set(PROTOCOLS_CONFIG.keys())
if CONFIG_PROTOCOLS != set(PROTOCOLS):
    raise ValueError(
        f"PROTOCOLS list {PROTOCOLS} must match PROTOCOLS_CONFIG keys {sorted(CONFIG_PROTOCOLS)}"
    )


# ---------------------------------------------------------------------------
# Summary Context & Output Formatting
# ---------------------------------------------------------------------------

@dataclass
class SummaryContext:
    protocol_outcomes: Dict[str, Dict[str, int]]
    duration: float
    strict: bool
    max_failures: Optional[int]
    report_json_path: Optional[str]
    report_dir: Optional[str]


PLAIN_OUTPUT_ENV = "CORPUS_PLAIN_OUTPUT"
STRICT_ENV = "CORPUS_STRICT"
MAX_FAILURES_ENV = "CORPUS_MAX_FAILURES"
REPORT_JSON_ENV = "CORPUS_REPORT_JSON"
REPORT_DIR_ENV = "CORPUS_REPORT_DIR"


# ---------------------------------------------------------------------------
# Adapter system
# ---------------------------------------------------------------------------

ADAPTER_ENV = "CORPUS_ADAPTER"
DEFAULT_ADAPTER = "tests.mock.mock_llm_adapter:MockLLMAdapter"
ENDPOINT_ENV = "CORPUS_ENDPOINT"

_ADAPTER_CLASS: Optional[type] = None
_ADAPTER_SPEC_USED: Optional[str] = None
_ADAPTER_VALIDATED: bool = False


class AdapterValidationError(RuntimeError):
    pass


def _validate_adapter_class(cls: type) -> None:
    if not inspect.isclass(cls):
        raise AdapterValidationError(
            f"Adapter spec must resolve to a class; got {type(cls)!r} from {cls!r}."
        )


def _load_class_from_spec(spec: str) -> type:
    module_name, _, class_name = spec.partition(":")
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

    _validate_adapter_class(cls)
    return cls


def _get_adapter_class() -> type:
    global _ADAPTER_CLASS, _ADAPTER_SPEC_USED, _ADAPTER_VALIDATED

    if _ADAPTER_CLASS is not None and _ADAPTER_VALIDATED:
        return _ADAPTER_CLASS

    spec = os.getenv(ADAPTER_ENV, DEFAULT_ADAPTER)
    _ADAPTER_SPEC_USED = spec

    try:
        _ADAPTER_CLASS = _load_class_from_spec(spec)
        _ADAPTER_VALIDATED = True
    except AdapterValidationError:
        _ADAPTER_CLASS = None
        _ADAPTER_SPEC_USED = None
        _ADAPTER_VALIDATED = False
        raise

    return _ADAPTER_CLASS


def _apply_pytest_adapter_option(config: pytest.Config) -> None:
    adapter_opt = getattr(getattr(config, "option", None), "adapter", None)
    if not adapter_opt or adapter_opt == "default":
        return
    os.environ[ADAPTER_ENV] = adapter_opt


@pytest.fixture(scope="session")
def adapter():
    Adapter = _get_adapter_class()
    endpoint = os.getenv(ENDPOINT_ENV)

    if endpoint:
        param_names = ["endpoint", "base_url", "url"]
        for kw in param_names:
            try:
                return Adapter(**{kw: endpoint})
            except TypeError:
                continue

        try:
            return Adapter()
        except TypeError as exc:
            raise AdapterValidationError(
                f"Failed to instantiate adapter '{_ADAPTER_SPEC_USED}' with endpoint. "
                f"Tried parameters: {param_names}. "
                f"Adapter constructor signature may be incompatible."
            ) from exc

    try:
        return Adapter()
    except TypeError as exc:
        raise AdapterValidationError(
            f"Failed to instantiate adapter '{_ADAPTER_SPEC_USED}' without arguments. "
            f"Ensure your adapter has a no-arg constructor or configure {ENDPOINT_ENV}."
        ) from exc


# ---------------------------------------------------------------------------
# Test categorization
# ---------------------------------------------------------------------------

class TestCategorizer:
    def __init__(self):
        self._protocol_patterns = self._build_protocol_patterns()
        self._category_patterns = self._build_category_patterns()
        self._cache: Dict[str, Tuple[str, str]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def _build_protocol_patterns(self) -> Dict[str, re.Pattern]:
        patterns: Dict[str, re.Pattern] = {}
        base_patterns = {
            "llm": r"tests[\\/]llm[\\/]",
            "vector": r"tests[\\/]vector[\\/]",
            "graph": r"tests[\\/]graph[\\/]",
            "embedding": r"tests[\\/]embedding[\\/]",
            "schema": r"tests[\\/]schema[\\/]",
            "llm_frameworks": r"tests[\\/]frameworks[\\/]llm[\\/]",
            "vector_frameworks": r"tests[\\/]frameworks[\\/]vector[\\/]",
            "embedding_frameworks": r"tests[\\/]frameworks[\\/]embedding[\\/]",
            "graph_frameworks": r"tests[\\/]frameworks[\\/]graph[\\/]",
            "wire": r"tests[\\/]live[\\/]",
        }
        for proto, pattern_str in base_patterns.items():
            patterns[proto] = re.compile(pattern_str, re.IGNORECASE)
        return patterns

    def _build_category_patterns(self) -> Dict[str, Dict[re.Pattern, str]]:
        category_patterns: Dict[str, Dict[re.Pattern, str]] = {}
        for proto, config in PROTOCOLS_CONFIG.items():
            proto_patterns: Dict[re.Pattern, str] = {}
            for category_key, category_name in config.test_categories.items():
                pats = [
                    re.compile(rf"\b{re.escape(category_key)}\b", re.IGNORECASE),
                    re.compile(rf"\b{re.escape(category_name.lower())}\b", re.IGNORECASE),
                ]
                for p in pats:
                    proto_patterns[p] = category_key
            category_patterns[proto] = proto_patterns
        return category_patterns

    def categorize_test(self, nodeid: str) -> Tuple[str, str]:
        nodeid_lower = (nodeid or "").lower()
        cached = self._cache.get(nodeid_lower)
        if cached is not None:
            self._cache_hits += 1
            return cached
        self._cache_misses += 1

        if "tests/frameworks/llm/" in nodeid_lower or "tests\\frameworks\\llm\\" in nodeid_lower:
            protocol = "llm_frameworks"
        elif "tests/frameworks/vector/" in nodeid_lower or "tests\\frameworks\\vector\\" in nodeid_lower:
            protocol = "vector_frameworks"
        elif "tests/frameworks/embedding/" in nodeid_lower or "tests\\frameworks\\embedding\\" in nodeid_lower:
            protocol = "embedding_frameworks"
        elif "tests/frameworks/graph/" in nodeid_lower or "tests\\frameworks\\graph\\" in nodeid_lower:
            protocol = "graph_frameworks"
        elif "tests/live/" in nodeid_lower or "tests\\live\\" in nodeid_lower:
            protocol = "wire"
        else:
            protocol = "other"
            for proto, pattern in self._protocol_patterns.items():
                if pattern.search(nodeid_lower):
                    protocol = proto
                    break

        if protocol == "wire":
            result = ("wire", "wire")
            self._cache[nodeid_lower] = result
            return result

        if protocol == "other":
            result = ("other", "unknown")
            self._cache[nodeid_lower] = result
            return result

        category = "unknown"
        proto_patterns = self._category_patterns.get(protocol, {})
        for pattern, category_key in proto_patterns.items():
            if pattern.search(nodeid_lower):
                category = category_key
                break

        result = (protocol, category)
        self._cache[nodeid_lower] = result
        return result

    def get_cache_stats(self) -> Dict[str, int]:
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_size": len(self._cache),
        }


test_categorizer = TestCategorizer()


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

class CorpusProtocolPlugin:
    def __init__(self):
        self.start_time: Optional[float] = None
        self._protocol_outcomes_cache: Optional[Dict[str, Dict[str, int]]] = None
        self.plain_output: bool = False
        self.verbose: bool = False
        self.unmapped_categories: Set[Tuple[str, str]] = set()
        self.strict: bool = False
        self.max_failures: Optional[int] = None
        self.report_json_path: Optional[str] = None
        self.report_dir: Optional[str] = None

    # -------- Formatting --------

    def _use_plain_output(self) -> bool:
        val = os.getenv(PLAIN_OUTPUT_ENV, "").strip().lower()
        return val in {"1", "true", "yes", "plain"}

    def _fmt(self, emoji: str, text: str) -> str:
        if self.plain_output or not emoji:
            return text
        return f"{emoji} {text}"

    def _level_display(self, raw_level: str) -> str:
        emoji = {
            "Gold": "🥇",
            "Silver": "🥈",
            "Development": "🔬",
            "Below Development": "❌",
            "No Tests Scored": "⚠️",
        }.get(raw_level, "")
        return self._fmt(emoji, raw_level)

    # -------- Policy --------

    def _read_bool_env(self, name: str, default: bool = False) -> bool:
        val = os.getenv(name)
        if val is None:
            return default
        return val.strip().lower() in {"1", "true", "yes", "on"}

    def _read_int_env(self, name: str) -> Optional[int]:
        val = os.getenv(name)
        if not val:
            return None
        try:
            n = int(val.strip())
        except ValueError:
            return None
        return n if n >= 0 else None

    def _init_policy(self) -> None:
        self.strict = self._read_bool_env(STRICT_ENV, default=False)
        self.max_failures = self._read_int_env(MAX_FAILURES_ENV)
        self.report_json_path = os.getenv(REPORT_JSON_ENV) or None
        self.report_dir = os.getenv(REPORT_DIR_ENV) or None

    # -------- Session --------

    def pytest_sessionstart(self, session: pytest.Session) -> None:
        self.start_time = time.time()
        self._protocol_outcomes_cache = None
        self.plain_output = self._use_plain_output()
        self.verbose = bool(getattr(session.config, "option", None) and session.config.option.verbose)
        self.unmapped_categories.clear()
        self._init_policy()

        if self.verbose:
            print(self._fmt("🔧", "CORPUS Protocol Plugin: ") + "cached categorization, validated config")

    def _get_duration(self) -> float:
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    # -------- Aggregation --------

    def _collect_protocol_outcomes(self, terminalreporter) -> Dict[str, Dict[str, int]]:
        if self._protocol_outcomes_cache is not None:
            return self._protocol_outcomes_cache

        outcomes = ("passed", "failed", "error", "skipped", "xfailed", "xpassed")
        proto_map: Dict[str, Dict[str, int]] = {p: {o: 0 for o in outcomes} for p in PROTOCOLS}
        proto_map["other"] = {o: 0 for o in outcomes}

        for outcome in outcomes:
            for rep in terminalreporter.stats.get(outcome, []):
                nodeid = getattr(rep, "nodeid", "") or ""
                proto, _cat = test_categorizer.categorize_test(nodeid)
                if proto not in proto_map:
                    proto = "other"
                proto_map[proto][outcome] += 1

        self._protocol_outcomes_cache = proto_map
        return proto_map

    def _scored_totals_for_protocol(self, proto_out: Dict[str, int]) -> Tuple[int, int, int]:
        """
        Return (scored_total, scored_passed, collected_total).

        Policy:
          - Default (CORPUS_STRICT=0): exclude skipped + xfailed from scoring; include xpassed as attention
          - Strict  (CORPUS_STRICT=1): include everything in scoring
        """
        collected_total = sum(proto_out.values())
        passed = proto_out.get("passed", 0)

        default_scored_total = (
            proto_out.get("passed", 0)
            + proto_out.get("failed", 0)
            + proto_out.get("error", 0)
            + proto_out.get("xpassed", 0)
        )

        scored_total = collected_total if self.strict else default_scored_total
        scored_passed = passed
        return scored_total, scored_passed, collected_total

    # -------- Categorization --------

    def _categorize_reports(self, reports: List[Any]) -> Dict[str, Dict[str, List[Any]]]:
        by_protocol: Dict[str, Dict[str, List[Any]]] = {p: {} for p in PROTOCOLS}
        by_protocol["other"] = {}

        for rep in reports:
            nodeid = getattr(rep, "nodeid", "") or ""
            proto, category = test_categorizer.categorize_test(nodeid)
            if proto not in by_protocol:
                proto = "other"
            by_protocol[proto].setdefault(category, []).append(rep)

        return {
            proto: {cat: lst for cat, lst in cats.items() if lst}
            for proto, cats in by_protocol.items()
            if any(cats.values())
        }

    # -------- Levels (raw, then format for terminal) --------

    def _calculate_conformance_level_raw(self, passed_count: int, total_scored: int) -> Tuple[str, int]:
        if total_scored == 0:
            return "No Tests Scored", 0
        if passed_count == total_scored:
            return "Gold", 0

        silver_threshold = math.ceil(total_scored * 0.80)
        if passed_count >= silver_threshold:
            needed = total_scored - passed_count
            return "Silver", max(0, needed)

        dev_threshold = math.ceil(total_scored * 0.50)
        if passed_count >= dev_threshold:
            needed = silver_threshold - passed_count
            return "Development", max(0, needed)

        needed = dev_threshold - passed_count
        return "Below Development", max(0, needed)

    # -------- Spec/error helpers --------

    def _get_spec_section(self, protocol: str, category: str) -> str:
        config = protocol_registry.get(protocol)
        if not config:
            return "See protocol specification"
        spec = config.spec_sections.get(category)
        if spec:
            return spec
        self.unmapped_categories.add((protocol, category))
        base = "See protocol specification (category unmapped; check for spec drift)"
        if self.verbose:
            base += f" [UNMAPPED {protocol}:{category}]"
        return base

    def _get_error_guidance(self, protocol: str, category: str, test_name: str) -> Dict[str, Any]:
        config = protocol_registry.get(protocol)
        if not config:
            return {
                "error_patterns": {},
                "quick_fix": "Review specification section above",
                "examples": "See specification for implementation details",
            }

        category_guidance = config.error_guidance.get(category, {})
        test_guidance = category_guidance.get(test_name, {})

        if not test_guidance:
            test_guidance = {
                "error_patterns": {},
                "quick_fix": f"Review {config.display_name}: {self._get_spec_section(protocol, category)}",
                "examples": f"See {config.display_name} documentation for implementation examples",
            }

        return {
            "error_patterns": test_guidance.get("error_patterns", {}),
            "quick_fix": test_guidance.get("quick_fix", "Review specification section above"),
            "examples": test_guidance.get("examples", "See specification for implementation details"),
        }

    def _extract_test_name(self, nodeid: str) -> str:
        parts = nodeid.split("::")
        return parts[-1] if len(parts) > 1 else "unknown_test"

    # -------- Adapter diagnostics --------

    def _adapter_diag_line(self) -> str:
        spec = os.getenv(ADAPTER_ENV, DEFAULT_ADAPTER)
        endpoint = os.getenv(ENDPOINT_ENV)
        endpoint_state = "set" if endpoint else "not set"
        strict_state = "on" if self.strict else "off"
        return (
            f"{self._fmt('🔌', 'Adapter:')} {spec} | "
            f"{ENDPOINT_ENV}: {endpoint_state} | "
            f"{self._fmt('⚖️', 'Strict:')} {strict_state}"
        )

    # -------- Platinum “why not” (data-only) --------

    def _why_not_platinum_data(self, protocol_outcomes: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
        totals = {"failed": 0, "error": 0, "skipped": 0, "xfailed": 0, "xpassed": 0}
        for proto in PROTOCOLS:
            out = protocol_outcomes.get(proto, {})
            for k in totals:
                totals[k] += out.get(k, 0)

        reasons: List[str] = []

        # Always blockers:
        if totals["failed"] or totals["error"]:
            reasons.append(f"{totals['failed']} failed, {totals['error']} error")
        if totals["xpassed"]:
            reasons.append(f"{totals['xpassed']} xpassed")

        # Strict-mode additional blockers:
        if self.strict:
            if totals["skipped"]:
                reasons.append(f"{totals['skipped']} skipped")
            if totals["xfailed"]:
                reasons.append(f"{totals['xfailed']} xfailed")
        else:
            # Non-blocking (but informative) in default mode:
            if not reasons:
                if totals["skipped"]:
                    reasons.append(f"{totals['skipped']} skipped")
                if totals["xfailed"]:
                    reasons.append(f"{totals['xfailed']} xfailed")

        blocked = bool(
            (totals["failed"] or totals["error"] or totals["xpassed"])
            or (self.strict and (totals["skipped"] or totals["xfailed"]))
        )

        return {
            "blocked": blocked,
            "totals": totals,
            "strict": self.strict,
            "reasons": reasons,
        }

    def _why_not_platinum_line(self, protocol_outcomes: Dict[str, Dict[str, int]]) -> str:
        data = self._why_not_platinum_data(protocol_outcomes)
        if not data["reasons"]:
            return self._fmt("ℹ️", "Not Platinum because:") + " certification policy conditions not met"
        return self._fmt("ℹ️", "Not Platinum because:") + " " + ", ".join(data["reasons"])

    def _is_platinum_certified(self, protocol_outcomes: Dict[str, Dict[str, int]]) -> bool:
        for proto in PROTOCOLS:
            out = protocol_outcomes.get(proto, {})
            scored_total, scored_passed, collected_total = self._scored_totals_for_protocol(out)
            if collected_total == 0:
                continue
            if scored_total == 0:
                continue
            if scored_passed != scored_total:
                return False
            if out.get("xpassed", 0) > 0:
                return False
            if out.get("failed", 0) > 0 or out.get("error", 0) > 0:
                return False
            if self.strict and (out.get("skipped", 0) > 0 or out.get("xfailed", 0) > 0):
                return False
        return True

    # -------- Status iterator --------

    def _iter_protocol_status(self, ctx: SummaryContext):
        for proto in PROTOCOLS:
            config = protocol_registry.get(proto)
            if not config:
                continue
            out = ctx.protocol_outcomes.get(proto, {})
            scored_total, scored_passed, collected_total = self._scored_totals_for_protocol(out)
            if collected_total == 0:
                continue
            raw_level, needed = self._calculate_conformance_level_raw(scored_passed, scored_total)
            yield proto, config, out, scored_total, scored_passed, collected_total, raw_level, needed

    # -------- Output helpers --------

    def _print_protocol_failure_summary(
        self,
        terminalreporter,
        proto: str,
        categories: Dict[str, List[Any]],
    ) -> None:
        if proto == "other":
            display_name = "Other (non-CORPUS conformance tests)"
        else:
            config = protocol_registry.get(proto)
            display_name = config.display_name if config else proto.upper()
        terminalreporter.write_line(f"{display_name}:")

        max_n = self.max_failures
        for category, reports_list in categories.items():
            original_count = len(reports_list)
            shown_list = reports_list
            truncated = False
            if max_n is not None and max_n > 0 and original_count > max_n and not self.verbose:
                shown_list = reports_list[:max_n]
                truncated = True

            if proto == "other":
                category_name = category.replace("_", " ").title() if category != "unknown" else "Unknown Category"
                spec_section = "Not part of CORPUS certification suite"
            else:
                config = protocol_registry.get(proto)
                category_name = (
                    config.test_categories.get(category, category.replace("_", " ").title())
                    if config
                    else "Unknown Category"
                )
                spec_section = self._get_spec_section(proto, category)

            terminalreporter.write_line(
                f"  {self._fmt('❌', 'Failure')} {category_name}: {original_count} issue(s)"
            )
            terminalreporter.write_line(f"      Specification: {spec_section}")

            for rep in shown_list:
                test_name = self._extract_test_name(rep.nodeid)
                guidance = (
                    self._get_error_guidance(proto, category, test_name)
                    if proto != "other"
                    else {"error_patterns": {}, "quick_fix": "Review this test's implementation.", "examples": ""}
                )

                terminalreporter.write_line(f"      Test: {test_name}")
                terminalreporter.write_line(f"      Quick fix: {guidance['quick_fix']}")

                error_patterns = guidance.get("error_patterns", {})
                if error_patterns:
                    error_msg = str(getattr(rep, "longrepr", "")).lower()
                    matched_patterns = [
                        desc for key, desc in error_patterns.items() if key.lower() in error_msg
                    ]
                    if matched_patterns:
                        terminalreporter.write_line(f"      Detected: {', '.join(matched_patterns)}")

            if truncated:
                terminalreporter.write_line(
                    f"      {self._fmt('…', 'Truncated:')} showing {len(shown_list)} of {original_count}. "
                    f"Set {MAX_FAILURES_ENV}=0 or run with -v to show all."
                )

        terminalreporter.write_line("")

    def _print_xpassed_summary(self, terminalreporter, xpassed_reports: List[Any]) -> None:
        if not xpassed_reports:
            return
        terminalreporter.write_sep("=", self._fmt("🟦", "UNEXPECTED PASSES (XPASS)"))
        terminalreporter.write_line(
            "These tests were marked xfail but passed. Usually the xfail is stale or behavior changed."
        )
        terminalreporter.write_line("")

        by_protocol = self._categorize_reports(xpassed_reports)
        for proto, categories in by_protocol.items():
            if not categories:
                continue
            display_name = (
                "Other (non-CORPUS conformance tests)"
                if proto == "other"
                else (protocol_registry.get(proto).display_name if protocol_registry.get(proto) else proto.upper())
            )
            terminalreporter.write_line(f"{display_name}:")
            for category, reps in categories.items():
                cat_name = category
                if proto != "other":
                    cfg = protocol_registry.get(proto)
                    if cfg:
                        cat_name = cfg.test_categories.get(category, category)
                terminalreporter.write_line(f"  {self._fmt('⚠️', 'XPASS')} {cat_name}: {len(reps)}")

                shown = reps
                if (
                    self.max_failures is not None
                    and self.max_failures > 0
                    and len(reps) > self.max_failures
                    and not self.verbose
                ):
                    shown = reps[: self.max_failures]
                    terminalreporter.write_line(
                        f"      {self._fmt('…', 'Truncated:')} showing {len(shown)} of {len(reps)}. "
                        f"Set {MAX_FAILURES_ENV}=0 or run -v."
                    )
                for rep in shown:
                    terminalreporter.write_line(f"      Test: {self._extract_test_name(rep.nodeid)}")
            terminalreporter.write_line("")

    # -------- Certification printers --------

    def _print_platinum_certification(self, terminalreporter, ctx: SummaryContext) -> None:
        terminalreporter.write_sep("=", self._fmt("🏆", "CORPUS PROTOCOL SUITE - PLATINUM CERTIFIED"))
        terminalreporter.write_line(self._adapter_diag_line())
        terminalreporter.write_line("")

        terminalreporter.write_line("Protocol & Framework Conformance Status (scored / collected):")
        for _proto, config, _out, scored_total, scored_passed, collected_total, raw_level, _needed in self._iter_protocol_status(ctx):
            terminalreporter.write_line(
                f"  {self._fmt('✅', 'PASS')} {config.display_name}: {self._level_display(raw_level)} "
                f"({scored_passed}/{scored_total} scored; {collected_total} collected)"
            )

        terminalreporter.write_line("")
        terminalreporter.write_line(self._fmt("🎯", "Status: Ready for production deployment"))
        terminalreporter.write_line(f"{self._fmt('⏱️', 'Completed in')} {ctx.duration:.2f}s")

        cache_stats = test_categorizer.get_cache_stats()
        terminalreporter.write_line(
            f"{self._fmt('🔧', 'Performance:')} "
            f"{cache_stats['cache_hits']} cache hits, {cache_stats['cache_misses']} misses "
            f"(cache size: {cache_stats['cache_size']})"
        )

    def _print_gold_certification(self, terminalreporter, ctx: SummaryContext) -> None:
        terminalreporter.write_sep("=", self._fmt("🥇", "CORPUS PROTOCOL SUITE - GOLD CERTIFIED"))
        terminalreporter.write_line(self._adapter_diag_line())
        terminalreporter.write_line("")

        terminalreporter.write_line("Protocol & Framework Conformance Status (scored / collected):")
        platinum_ready = True

        for _proto, config, _out, scored_total, scored_passed, collected_total, raw_level, needed in self._iter_protocol_status(ctx):
            level_disp = self._level_display(raw_level)
            if raw_level == "Gold":
                terminalreporter.write_line(
                    f"  {self._fmt('✅', 'PASS')} {config.display_name}: {level_disp} "
                    f"({scored_passed}/{scored_total} scored; {collected_total} collected)"
                )
            else:
                platinum_ready = False
                next_label = "Gold" if raw_level == "Silver" else ("Silver" if raw_level == "Development" else "Development")
                terminalreporter.write_line(
                    f"  {self._fmt('⚠️', 'WARN')} {config.display_name}: {level_disp} "
                    f"({scored_passed}/{scored_total} scored; {collected_total} collected; {needed} to {next_label})"
                )

        terminalreporter.write_line("")
        if platinum_ready:
            terminalreporter.write_line(self._fmt("🎯", "All protocols at Gold level - Platinum certification available!"))
        else:
            terminalreporter.write_line(self._fmt("🎯", "Focus on protocols below Gold to reach Platinum (100% scored pass)."))

        terminalreporter.write_line(self._why_not_platinum_line(ctx.protocol_outcomes))
        terminalreporter.write_line(f"{self._fmt('⏱️', 'Completed in')} {ctx.duration:.2f}s")

        cache_stats = test_categorizer.get_cache_stats()
        terminalreporter.write_line(
            f"{self._fmt('🔧', 'Performance:')} "
            f"{cache_stats['cache_hits']} cache hits, {cache_stats['cache_misses']} misses "
            f"(cache size: {cache_stats['cache_size']})"
        )

    def _print_failure_analysis(
        self,
        terminalreporter,
        ctx: SummaryContext,
        failed_error_reports: List[Any],
        xpassed_reports: List[Any],
    ) -> None:
        terminalreporter.write_sep("=", self._fmt("❌", "CORPUS PROTOCOL CONFORMANCE ANALYSIS"))
        terminalreporter.write_line(self._adapter_diag_line())
        terminalreporter.write_line(self._why_not_platinum_line(ctx.protocol_outcomes))
        terminalreporter.write_line("")

        total_issues = len(failed_error_reports) + len(xpassed_reports)
        terminalreporter.write_line(f"Found {total_issues} issue(s) across protocols.")
        terminalreporter.write_line("")

        if failed_error_reports:
            terminalreporter.write_sep("-", self._fmt("🟥", "FAILURES & ERRORS"))
            by_protocol = self._categorize_reports(failed_error_reports)
            for proto, categories in by_protocol.items():
                if categories:
                    self._print_protocol_failure_summary(terminalreporter, proto, categories)

        self._print_xpassed_summary(terminalreporter, xpassed_reports)

        terminalreporter.write_line("Next Steps:")
        terminalreporter.write_line("  1. Fix failures/errors first (they always block certification).")
        terminalreporter.write_line("  2. For XPASS: remove/adjust stale xfail markers or confirm behavior changes.")
        if self.strict:
            terminalreporter.write_line("  3. Strict mode is enabled: skips/xfails also block Platinum.")
        terminalreporter.write_line("")

        cache_stats = test_categorizer.get_cache_stats()
        terminalreporter.write_line(
            f"{self._fmt('🔧', 'Performance:')} "
            f"{cache_stats['cache_hits']} cache hits, {cache_stats['cache_misses']} misses "
            f"(cache size: {cache_stats['cache_size']})"
        )
        terminalreporter.write_line(f"{self._fmt('⏱️', 'Completed in')} {ctx.duration:.2f}s")

    # -------- JSON export --------

    def _write_json_report(self, ctx: SummaryContext) -> None:
        path: Optional[Path] = None
        if ctx.report_json_path:
            path = Path(ctx.report_json_path)
        elif ctx.report_dir:
            path = Path(ctx.report_dir) / "summary.json"
        if path is None:
            return

        path.parent.mkdir(parents=True, exist_ok=True)

        def per_proto_obj(proto: str) -> Dict[str, Any]:
            out = ctx.protocol_outcomes.get(proto, {})
            scored_total, scored_passed, collected_total = self._scored_totals_for_protocol(out)
            raw_level, needed = self._calculate_conformance_level_raw(scored_passed, scored_total)
            cfg = protocol_registry.get(proto)
            return {
                "display_name": (cfg.display_name if cfg else proto),
                "reference_levels": (cfg.conformance_levels if cfg else None),
                "outcomes": dict(out),
                "collected_total": collected_total,
                "scored_total": scored_total,
                "scored_passed": scored_passed,
                "level": raw_level,
                "tests_needed_to_next_level": needed,
            }

        payload = {
            "version": 2,
            "generated_at_epoch_s": int(time.time()),
            "policy": {"strict": ctx.strict, "max_failures": ctx.max_failures},
            "adapter": {"spec": os.getenv(ADAPTER_ENV, DEFAULT_ADAPTER), "endpoint_set": bool(os.getenv(ENDPOINT_ENV))},
            "duration_s": ctx.duration,
            "protocols": {p: per_proto_obj(p) for p in PROTOCOLS},
            "platinum_certified": self._is_platinum_certified(ctx.protocol_outcomes),
            "why_not_platinum": self._why_not_platinum_data(ctx.protocol_outcomes),
        }

        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    # -------- Main hook --------

    def pytest_terminal_summary(self, terminalreporter, exitstatus, config) -> None:
        duration = self._get_duration()
        protocol_outcomes = self._collect_protocol_outcomes(terminalreporter)

        ctx = SummaryContext(
            protocol_outcomes=protocol_outcomes,
            duration=duration,
            strict=self.strict,
            max_failures=self.max_failures,
            report_json_path=self.report_json_path,
            report_dir=self.report_dir,
        )

        failed_error_reports: List[Any] = []
        failed_error_reports.extend(terminalreporter.stats.get("failed", []))
        failed_error_reports.extend(terminalreporter.stats.get("error", []))
        xpassed_reports: List[Any] = list(terminalreporter.stats.get("xpassed", []))

        if not failed_error_reports and not xpassed_reports:
            if self._is_platinum_certified(protocol_outcomes):
                self._print_platinum_certification(terminalreporter, ctx)
            else:
                self._print_gold_certification(terminalreporter, ctx)
        else:
            self._print_failure_analysis(
                terminalreporter,
                ctx,
                failed_error_reports=failed_error_reports,
                xpassed_reports=xpassed_reports,
            )

        self._write_json_report(ctx)


corpus_protocol_plugin = CorpusProtocolPlugin()


def pytest_sessionstart(session: pytest.Session) -> None:
    corpus_protocol_plugin.pytest_sessionstart(session)


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    corpus_protocol_plugin.pytest_terminal_summary(terminalreporter, exitstatus, config)


def pytest_configure(config: pytest.Config) -> None:
    markers = [
        "llm: LLM Protocol V1.0 conformance tests",
        "vector: Vector Protocol V1.0 conformance tests",
        "graph: Graph Protocol V1.0 conformance tests",
        "embedding: Embedding Protocol V1.0 conformance tests",
        "llm_frameworks: LLM framework adapter conformance tests",
        "vector_frameworks: Vector framework adapter conformance tests",
        "embedding_frameworks: Embedding framework adapter conformance tests",
        "graph_frameworks: Graph framework adapter conformance tests",
        "wire: Wire Request Conformance tests (tests/live)",
        "schema: Schema conformance validation tests",
        "slow: Tests that take longer to run (skip with -m 'not slow')",
        "conformance: All protocol conformance tests",
    ]
    for marker in markers:
        config.addinivalue_line("markers", marker)

    _apply_pytest_adapter_option(config)

    if getattr(config, "option", None) and getattr(config.option, "verbose", False):
        try:
            _get_adapter_class()
            print("✅ Adapter configuration validated successfully")
        except AdapterValidationError as e:
            print(f"❌ Adapter configuration error: {e}")
            spec = os.getenv(ADAPTER_ENV, DEFAULT_ADAPTER)
            module_name, _, class_name = spec.partition(":")
            try:
                module = importlib.import_module(module_name)
                cls = getattr(module, class_name, None)
                if cls and inspect.isclass(cls):
                    try:
                        sig = str(inspect.signature(cls))
                        print(f"   Constructor signature: {class_name}{sig}")
                    except (ValueError, TypeError):
                        pass
            except Exception:
                pass


__all__ = [
    # pytest integration surface
    "pytest_sessionstart",
    "pytest_terminal_summary",
    "pytest_configure",
    # public fixtures
    "adapter",
    # configuration/registry (useful for tooling / introspection)
    "protocol_registry",
    "PROTOCOLS_CONFIG",
    "PROTOCOLS",
    "PROTOCOL_DISPLAY_NAMES",
    # plugin instance (advanced users)
    "corpus_protocol_plugin",
]
