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
import inspect
from typing import Dict, List, Optional, Tuple, Any, Set
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
    # NOTE: conformance_levels are reference values only. Certification scoring
    # is computed dynamically from collected tests in this session.
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
            raise ValueError(
                f"Protocol {self.name}: missing required conformance levels: {required_levels}"
            )
        
        for level_name, threshold in self.conformance_levels.items():
            if not isinstance(threshold, int) or threshold < 0:
                raise ValueError(
                    f"Protocol {self.name}: conformance level {level_name} must be positive integer"
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

# NOTE: conformance_levels remain as reference values, but certification scoring
# is computed dynamically based on collected tests per protocol.
PROTOCOLS_CONFIG = {
    "llm": ProtocolConfig(
        name="llm",
        display_name="LLM Protocol V1.0",
        # Reference: currently ~111 llm tests in the suite
        conformance_levels={"gold": 111, "silver": 89, "development": 56},
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
        # Reference: ~73 vector tests (adapter + protocol-level)
        conformance_levels={"gold": 73, "silver": 58, "development": 36},
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
                "test_node_lifecycle": {
                    "error_patterns": {
                        "duplicate_node": "Node creation with duplicate ID violated constraints",
                        "invalid_properties": "Node properties violate schema constraints"
                    },
                    "quick_fix": "Check node ID uniqueness and property schema compliance",
                    "examples": "See §7.3.1 for node CRUD operations and constraints"
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

    # NEW: Dedicated protocol entry for the wire conformance suite (tests/live)
    "wire": ProtocolConfig(
        name="wire",
        display_name="Wire Request Conformance Suite",
        # Authoritative numbers from wire_cases.py + edge-case tests:
        # 48 parametrized request cases + 25 edge/validator tests = 73
        conformance_levels={"gold": 73, "silver": 58, "development": 37},
        test_categories={
            # We treat all tests in tests/live as belonging to this single category.
            "wire": "Wire Request Envelope Conformance",
        },
        spec_sections={
            "wire": "Wire Request Conformance Suite (tests/live/test_wire_conformance.py)",
        },
        error_guidance={
            "wire": {
                # Main parametrized wire test
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
        }
    ),
    
    "schema": ProtocolConfig(
        name="schema",
        display_name="CORPUS Schema Conformance Suite",
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
                    "quick_fix": (
                        "Ensure all schema files are valid JSON and include "
                        "$schema: 'https://json-schema.org/draft/2020-12/schema'"
                    ),
                    "examples": "See SCHEMA_CONFORMANCE.md - Schema Loading & IDs section"
                },
                "test_unique_ids": {
                    "error_patterns": {
                        "duplicate_id": "Duplicate $id found across schema files",
                        "invalid_id_format": "$id does not follow https://adaptersdk.org/schemas/ format"
                    },
                    "quick_fix": (
                        "Ensure each schema has unique $id following convention: "
                        "https://adaptersdk.org/schemas/<component>/<file>.json"
                    ),
                    "examples": "See SCHEMA_CONFORMANCE.md - $id hygiene requirements"
                }
            },
            "metaschema_hygiene": {
                "test_metaschema_compliance": {
                    "error_patterns": {
                        "draft_2020_12_violation": "Schema violates JSON Schema Draft 2020-12",
                        "invalid_keywords": "Unknown or invalid JSON Schema keywords used"
                    },
                    "quick_fix": (
                        "Validate schema against Draft 2020-12 metaschema and remove unsupported keywords"
                    ),
                    "examples": "See SCHEMA_CONFORMANCE.md - Metaschema & Hygiene section"
                },
                "test_regex_patterns": {
                    "error_patterns": {
                        "invalid_regex": "Regular expression pattern does not compile",
                        "unsupported_regex_flags": "Regex uses unsupported flags"
                    },
                    "quick_fix": (
                        "Fix regex patterns to use supported ECMA 262 syntax without flags"
                    ),
                    "examples": "See SCHEMA_CONFORMANCE.md - Pattern hygiene requirements"
                }
            },
            "cross_references": {
                "test_ref_resolution": {
                    "error_patterns": {
                        "unresolved_ref": "$ref cannot be resolved to known schema $id",
                        "invalid_fragment": "Fragment (#/definitions/...) points to non-existent definition"
                    },
                    "quick_fix": (
                        "Ensure all $ref values point to valid $ids or internal fragments"
                    ),
                    "examples": "See SCHEMA_CONFORMANCE.md - Cross-References section"
                }
            }
        }
    ),
    
    "golden": ProtocolConfig(
        name="golden",
        display_name="CORPUS Golden Wire Suite", 
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
                        "schema_validation_failed": (
                            "Golden sample does not validate against its declared schema"
                        ),
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
                    "quick_fix": (
                        "Ensure streams have exactly one terminal frame (end/error) after data frames"
                    ),
                    "examples": "See SCHEMA_CONFORMANCE.md - NDJSON Stream Validation"
                }
            },
            "cross_invariants": {
                "test_partial_success_math": {
                    "error_patterns": {
                        "count_mismatch": (
                            "successes + failures ≠ total items in partial success"
                        ),
                        "invalid_indexing": "Failure indices out of bounds"
                    },
                    "quick_fix": (
                        "Ensure partial success counts are mathematically consistent"
                    ),
                    "examples": "See SCHEMA_CONFORMANCE.md - Cross-Schema Invariants"
                }
            }
        }
    ),

    # NEW: Embedding framework adapter suite
    "embedding_frameworks": ProtocolConfig(
        name="embedding_frameworks",
        display_name="Embedding Framework Adapters V1.0",
        # Actual test count from analysis: 121 tests (reference)
        conformance_levels={"gold": 121, "silver": 97, "development": 60},
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
                    "quick_fix": "Ensure corpus adapter implements the required EmbeddingProtocolV1 interface and exposes embed()/aembed() as expected.",
                    "examples": "See §10.3 for required embedding protocol methods and framework adapter examples in tests/frameworks/embedding/.",
                },
            },
            "contract_interface": {
                "test_sync_embedding_interface_conformance": {
                    "error_patterns": {
                        "missing_method": "Framework adapter missing required embedding methods",
                        "signature_mismatch": "Method signatures don't match framework expectations",
                    },
                    "quick_fix": "Ensure all required sync/async embedding methods are implemented with correct signatures for the framework (e.g., embed_documents, embed_query, async variants).",
                    "examples": "See test_contract_interface_conformance.py for interface requirements.",
                },
            },
            "contract_shapes": {
                "test_batch_output_row_count_matches_input_length": {
                    "error_patterns": {
                        "row_count_mismatch": "Batch embedding returned wrong number of rows",
                        "shape_validation_failed": "Embedding matrix shape doesn't match input length",
                    },
                    "quick_fix": "Ensure embed() returns exactly N vectors for N input texts, and that each vector has consistent dimension.",
                    "examples": "See test_contract_shapes_and_batching.py for shape validation details.",
                },
            },
            "contract_context": {
                "test_error_context_is_attached_on_sync_batch_failure": {
                    "error_patterns": {
                        "missing_error_context": "Errors not decorated with framework/operation metadata",
                        "context_attachment_failed": "attach_context() not called on error path",
                    },
                    "quick_fix": "Decorate public embedding methods with @error_context(framework='<name>') and ensure failures attach OperationContext.",
                    "examples": "See §6.3 and test_contract_context_and_error_context.py for error context requirements.",
                },
            },
            # registry_infra and robustness fall back to generic guidance
        }
    ),

    # NEW: Graph framework adapter suite
    "graph_frameworks": ProtocolConfig(
        name="graph_frameworks",
        display_name="Graph Framework Adapters V1.0",
        # From graph framework conformance doc: 184 tests (reference)
        conformance_levels={"gold": 184, "silver": 147, "development": 92},
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
            "contract_shapes": {
                "test_batch_result_length_matches_ops_when_supported": {
                    "error_patterns": {
                        "length_mismatch": "Batch result length does not match number of operations",
                        "batch_shape_invalid": "Batch result structure is inconsistent across calls",
                    },
                    "quick_fix": "Ensure batch() / abatch() return one result per operation in order, with a stable result type/shape.",
                    "examples": "See test_contract_shapes_and_batching.py for batch semantics and shape validation.",
                },
            },
            "contract_context": {
                "test_error_context_is_attached_on_sync_query_failure": {
                    "error_patterns": {
                        "missing_error_context": "Errors not decorated with framework/operation metadata",
                        "context_attachment_failed": "Context not attached when backend query fails",
                    },
                    "quick_fix": "Decorate public graph methods with @error_context(framework='<name>') and attach OperationContext on failures.",
                    "examples": "See §6.3 and test_contract_context_and_error_context.py for error context requirements.",
                },
            },
            # Other categories fall back to generic guidance in plugin
        }
    ),
}

# Register all protocols
for protocol_config in PROTOCOLS_CONFIG.values():
    protocol_registry.register(protocol_config)

# Validate all configurations on module load
protocol_registry.validate_all()

# Protocol display names for quick access
PROTOCOL_DISPLAY_NAMES = {
    proto: config.display_name for proto, config in PROTOCOLS_CONFIG.items()
}

# Explicit, stable protocol order for summaries / UX
PROTOCOLS = [
    "llm",
    "vector",
    "graph",
    "embedding",
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
    """Aggregated context for final terminal summary."""
    protocol_passed: Dict[str, int]
    protocol_total: Dict[str, int]
    outcome_counts: Dict[str, int]
    duration: float


PLAIN_OUTPUT_ENV = "CORPUS_PLAIN_OUTPUT"


# ---------------------------------------------------------------------------
# Performance-optimized adapter system
# ---------------------------------------------------------------------------

# Environment variable for fully-qualified adapter class:
#   CORPUS_ADAPTER="package.module:ClassName"
ADAPTER_ENV = "CORPUS_ADAPTER"
DEFAULT_ADAPTER = "tests.mock.mock_llm_adapter:MockLLMAdapter"
ENDPOINT_ENV = "CORPUS_ENDPOINT"

# NOTE: adapter class caching uses module-level globals for performance.
# Pytest executes collection and fixture construction on a single main thread,
# so this is safe in typical usage. If this code is adapted for multi-threaded
# plugin execution, wrap _get_adapter_class() in a threading.Lock.
_ADAPTER_CLASS: Optional[type] = None
_ADAPTER_SPEC_USED: Optional[str] = None
_ADAPTER_VALIDATED: bool = False


class AdapterValidationError(RuntimeError):
    """Custom exception for adapter validation failures."""
    pass


def _validate_adapter_class(cls: type) -> None:
    """Validate that adapter class meets minimum interface requirements."""
    if not inspect.isclass(cls):
        raise AdapterValidationError(
            f"Adapter spec must resolve to a class; got {type(cls)!r} from {cls!r}."
        )
    # Adapter interface requirements are intentionally minimal; protocol-specific
    # tests assert capabilities and method shapes.


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


def _apply_pytest_adapter_option(config: pytest.Config) -> None:
    """
    Bridge pytest's --adapter option (used by wire tests in tests/live)
    into the CORPUS_ADAPTER environment that this plugin uses.

    This keeps a single source of truth for adapter selection without
    duplicating command-line flags.
    """
    adapter_opt = getattr(getattr(config, "option", None), "adapter", None)
    if not adapter_opt or adapter_opt == "default":
        # Either the option wasn't provided, or the caller wants the default.
        return

    # Only set env var if user explicitly requested an adapter on the CLI.
    # This ensures existing CORPUS_ADAPTER env config is preserved unless
    # the --adapter flag is used.
    os.environ[ADAPTER_ENV] = adapter_opt


@pytest.fixture(scope="session")
def adapter():
    """
    Generic, pluggable adapter fixture with enhanced validation.

    The adapter class is resolved from CORPUS_ADAPTER (or the default mock
    adapter), and an optional CORPUS_ENDPOINT is passed into the constructor
    using common parameter names.
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
        """Compile regex patterns for protocol detection based on directory."""
        patterns: Dict[str, re.Pattern] = {}

        for proto in PROTOCOLS:
            # Framework adapter suites live under tests/frameworks/<kind>/
            if proto == "embedding_frameworks":
                patterns[proto] = re.compile(
                    r'tests[\\/]frameworks[\\/]embedding[\\/]', re.IGNORECASE
                )
            elif proto == "graph_frameworks":
                patterns[proto] = re.compile(
                    r'tests[\\/]frameworks[\\/]graph[\\/]', re.IGNORECASE
                )
            else:
                # Match both POSIX and Windows paths, e.g. "tests/llm/", "tests\\llm\\"
                patterns[proto] = re.compile(
                    rf'tests[\\/]{re.escape(proto)}[\\/]', re.IGNORECASE
                )

        return patterns
    
    def _build_category_patterns(self) -> Dict[str, Dict[re.Pattern, str]]:
        """Compile regex patterns for category detection."""
        category_patterns: Dict[str, Dict[re.Pattern, str]] = {}
        for proto, config in PROTOCOLS_CONFIG.items():
            proto_patterns: Dict[re.Pattern, str] = {}
            for category_key, category_name in config.test_categories.items():
                # Create patterns for both category key and display name
                patterns = [
                    re.compile(rf'\b{re.escape(category_key)}\b', re.IGNORECASE),
                    re.compile(rf'\b{re.escape(category_name.lower())}\b', re.IGNORECASE),
                ]
                for pattern in patterns:
                    proto_patterns[pattern] = category_key
            category_patterns[proto] = proto_patterns
        return category_patterns
    
    def categorize_test(self, nodeid: str) -> Tuple[str, str]:
        """
        Categorize test by protocol and category with caching.

        This works for the per-protocol suites in tests/<proto>/ as well as the
        shared wire conformance suite in tests/live. Wire tests are *not* counted
        towards the llm/vector/graph/embedding adapter suites; they are grouped
        under the dedicated 'wire' protocol.

        Framework adapter suites under tests/frameworks/<kind>/ are mapped to
        embedding_frameworks / graph_frameworks and then categorized by filename
        (framework-specific vs contract vs registry vs robustness).
        """
        nodeid_lower = (nodeid or "").lower()
        
        # Cache lookup
        cached = self._cache.get(nodeid_lower)
        if cached is not None:
            self._cache_hits += 1
            return cached
        
        self._cache_misses += 1

        # --- 1. Detect protocol via directory patterns ---
        protocol = "other"
        for proto, pattern in self._protocol_patterns.items():
            if pattern.search(nodeid_lower):
                protocol = proto
                break

        # Special handling for shared wire tests under tests/live.
        if protocol == "other" and (
            "tests/live" in nodeid_lower or "tests\\live" in nodeid_lower
        ):
            protocol = "wire"

        # For the dedicated wire suite, we keep categorization simple: everything
        # is considered part of the 'wire' category.
        if protocol == "wire":
            result = ("wire", "wire")
            self._cache[nodeid_lower] = result
            return result

        # --- 2. Framework adapter suites: derive category from filename ---
        if protocol in {"embedding_frameworks", "graph_frameworks"}:
            # Note: we map based on file naming conventions from the coverage docs.
            if "test_contract_interface" in nodeid_lower:
                category = "contract_interface"
            elif "test_contract_shapes_and_batching" in nodeid_lower or "test_contract_shapes" in nodeid_lower:
                category = "contract_shapes"
            elif "test_contract_context_and_error_context" in nodeid_lower or "test_contract_context" in nodeid_lower:
                category = "contract_context"
            elif "registry_self_check" in nodeid_lower or "test_graph_registry_self_check" in nodeid_lower or "test_embedding_registry_self_check" in nodeid_lower or "test_registry" in nodeid_lower:
                category = "registry_infra"
            elif "test_with_mock_backends" in nodeid_lower or "test_with_mock_" in nodeid_lower:
                category = "robustness"
            else:
                category = "framework_specific"

            result = (protocol, category)
            self._cache[nodeid_lower] = result
            return result

        if protocol == "other":
            result = ("other", "unknown")
            self._cache[nodeid_lower] = result
            return result
        
        # --- 3. Category detection with cached patterns for core protocol suites ---
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
        """Get cache statistics for performance monitoring."""
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_size": len(self._cache),
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
        self._protocol_totals_cache: Optional[Dict[str, int]] = None
        self.plain_output: bool = False
        self.verbose: bool = False
        self.unmapped_categories: Set[Tuple[str, str]] = set()
        
    # -------- Emoji / plain output helpers --------

    def _use_plain_output(self) -> bool:
        """Determine whether to suppress emojis in output."""
        val = os.getenv(PLAIN_OUTPUT_ENV, "").strip().lower()
        return val in {"1", "true", "yes", "plain"}

    def _fmt(self, emoji: str, text: str) -> str:
        """Format a label with optional emoji, respecting CORPUS_PLAIN_OUTPUT."""
        if self.plain_output or not emoji:
            return text
        return f"{emoji} {text}"

    # -------- Session bookkeeping --------

    def pytest_sessionstart(self, session):
        """Record session start time and initialize tracking."""
        self.start_time = time.time()
        self.test_reports = {proto: [] for proto in PROTOCOLS}
        self.test_reports["other"] = []
        self.protocol_counts = {proto: {} for proto in PROTOCOLS}
        self._protocol_results_cache = None
        self._protocol_totals_cache = None
        self.plain_output = self._use_plain_output()
        self.verbose = bool(getattr(session.config, "option", None) and session.config.option.verbose)
        self.unmapped_categories.clear()
        
        if session.config.option.verbose:
            print(
                self._fmt("🔧", "CORPUS Protocol Plugin: Performance optimizations enabled - ")
                + "cached test categorization, pre-validated configurations"
            )

    # -------- Simple metrics helpers --------

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

    # -------- Failure categorization --------

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

    # -------- Dynamic conformance levels --------

    def _calculate_conformance_level(
        self,
        protocol: str,
        passed_count: int,
        total_collected: int,
    ) -> Tuple[str, int]:
        """
        Dynamically calculate level based on percentage of collected tests.

        - Gold: 100% passing
        - Silver: >= 80% passing
        - Development: >= 50% passing
        - Below Development: < 50% passing

        The integer in the result is "tests needed to reach the next level".
        """
        if total_collected == 0:
            return self._fmt("⚠️", "No Tests Found"), 0

        # strict 100% for Gold
        if passed_count == total_collected:
            return self._fmt("🥇", "Gold"), 0
        
        # 80% for Silver
        silver_threshold = int(total_collected * 0.80)
        if passed_count >= silver_threshold:
            needed = total_collected - passed_count  # to Gold
            return self._fmt("🥈", "Silver"), max(0, needed)

        # 50% for Development
        dev_threshold = int(total_collected * 0.50)
        if passed_count >= dev_threshold:
            needed = silver_threshold - passed_count  # to Silver
            return self._fmt("🔬", "Development"), max(0, needed)

        # Below Development
        needed = dev_threshold - passed_count  # to Development
        return self._fmt("❌", "Below Development"), max(0, needed)

    # -------- Spec and error guidance helpers --------

    def _get_spec_section(self, protocol: str, category: str) -> str:
        """
        Get specification section for a test category.
        
        Unmapped categories are explicitly called out to help catch drift
        between config and the actual test surface.
        """
        config = protocol_registry.get(protocol)
        if not config:
            return "See protocol specification"

        spec = config.spec_sections.get(category)
        if spec:
            return spec

        # Mark unmapped / unknown categories so drift is visible.
        self.unmapped_categories.add((protocol, category))
        base = "See protocol specification (category unmapped; check for spec drift)"
        if self.verbose:
            base += f" [UNMAPPED {protocol}:{category}]"
        return base

    def _get_error_guidance(self, protocol: str, category: str, test_name: str) -> Dict[str, str]:
        """Get specific error guidance for a test failure."""
        config = protocol_registry.get(protocol)
        if not config:
            return {
                "error_patterns": {},
                "quick_fix": "Review specification section above",
                "examples": "See specification for implementation details",
            }
            
        category_guidance = config.error_guidance.get(category, {})
        test_guidance = category_guidance.get(test_name, {})
        
        # Fallback guidance for uncovered tests
        if not test_guidance:
            test_guidance = {
                "error_patterns": {},
                "quick_fix": (
                    f"Review {config.display_name} specification section: "
                    f"{self._get_spec_section(protocol, category)}"
                ),
                "examples": (
                    f"See {config.display_name} documentation for implementation examples"
                ),
            }

        return {
            "error_patterns": test_guidance.get("error_patterns", {}),
            "quick_fix": test_guidance.get("quick_fix", "Review specification section above"),
            "examples": test_guidance.get("examples", "See specification for implementation details"),
        }

    def _extract_test_name(self, nodeid: str) -> str:
        """Extract the test function name from nodeid."""
        parts = nodeid.split("::")
        return parts[-1] if len(parts) > 1 else "unknown_test"

    # -------- Protocol result aggregation --------

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

    def _collect_protocol_totals(self, terminalreporter) -> Dict[str, int]:
        """
        Collect total collected tests per protocol (all outcomes).

        This is used for dynamic scoring so you never have to hand-tune
        thresholds when adding/removing tests.
        """
        if self._protocol_totals_cache is not None:
            return self._protocol_totals_cache

        totals = {proto: 0 for proto in PROTOCOLS}
        for outcome in ("passed", "failed", "error", "skipped", "xfailed", "xpassed"):
            for test_report in terminalreporter.stats.get(outcome, []):
                nodeid = getattr(test_report, "nodeid", "") or ""
                proto, _ = test_categorizer.categorize_test(nodeid)
                if proto in totals:
                    totals[proto] += 1

        self._protocol_totals_cache = totals
        return totals

    def _is_platinum_certified(self, ctx: SummaryContext) -> bool:
        """
        Check if all protocols meet Platinum certification requirements.

        Platinum: every protocol that has any collected tests must be Gold
        (i.e. 100% passing in this session).
        """
        for proto in PROTOCOLS:
            total = ctx.protocol_total.get(proto, 0)
            if total == 0:
                continue
            passed = ctx.protocol_passed.get(proto, 0)
            if passed != total:
                return False
        return True

    # -------- Output helpers for failure analysis --------

    def _print_protocol_failure_summary(
        self,
        terminalreporter,
        proto: str,
        categories: Dict[str, List[Any]],
    ) -> None:
        """Print failures for a single protocol with actionable guidance."""
        if proto == "other":
            display_name = "Other (non-CORPUS conformance tests)"
        else:
            config = protocol_registry.get(proto)
            display_name = config.display_name if config else proto.upper()
        terminalreporter.write_line(f"{display_name}:")

        for category, reports_list in categories.items():
            count = len(reports_list)

            if proto == "other":
                category_name = (
                    category.replace("_", " ").title() if category != "unknown" else "Unknown Category"
                )
                spec_section = "Not part of CORPUS certification suite"
            else:
                config = protocol_registry.get(proto)
                category_name = "Unknown Category"
                if config:
                    category_name = config.test_categories.get(
                        category, category.replace("_", " ").title()
                    )
                spec_section = self._get_spec_section(proto, category)

            terminalreporter.write_line(f"  {self._fmt('❌', 'Failure')} {category_name}: {count} failure(s)")
            terminalreporter.write_line(f"      Specification: {spec_section}")

            # Show specific guidance for each failed test
            for rep in reports_list:
                test_name = self._extract_test_name(rep.nodeid)
                guidance = (
                    self._get_error_guidance(proto, category, test_name)
                    if proto != "other"
                    else {
                        "error_patterns": {},
                        "quick_fix": "Review this test's implementation and related documentation.",
                        "examples": "",
                    }
                )

                terminalreporter.write_line(f"      Test: {test_name}")
                terminalreporter.write_line(f"      Quick fix: {guidance['quick_fix']}")

                # Enhanced error pattern matching
                error_patterns = guidance.get("error_patterns", {})
                if error_patterns:
                    error_msg = str(getattr(rep, "longrepr", "")).lower()
                    matched_patterns = []
                    for pattern_key, pattern_desc in error_patterns.items():
                        if pattern_key.lower() in error_msg:
                            matched_patterns.append(pattern_desc)
                    
                    if matched_patterns:
                        terminalreporter.write_line(f"      Detected: {', '.join(matched_patterns)}")

        terminalreporter.write_line("")

    # -------- Top-level summary printers --------

    def _print_platinum_certification(self, terminalreporter, ctx: SummaryContext):
        """Print Platinum certification summary."""
        total_tests = self._get_total_tests(ctx.outcome_counts)

        terminalreporter.write_sep("=", self._fmt("🏆", "CORPUS PROTOCOL SUITE - PLATINUM CERTIFIED"))
        terminalreporter.write_line(
            f"All {total_tests} conformance tests passed across {len(PROTOCOLS)} protocol and adapter suites"
        )
        terminalreporter.write_line("")

        # Show protocol breakdown
        terminalreporter.write_line("Protocol Conformance Status:")
        for proto in PROTOCOLS:
            # Skip framework adapter suites here; they get a dedicated section below
            if proto in {"embedding_frameworks", "graph_frameworks"}:
                continue

            config = protocol_registry.get(proto)
            if not config:
                continue
            total = ctx.protocol_total.get(proto, 0)
            if total == 0:
                continue
            passed = ctx.protocol_passed.get(proto, 0)
            level, _ = self._calculate_conformance_level(proto, passed, total)
            terminalreporter.write_line(
                f"  {self._fmt('✅', 'PASS')} {config.display_name}: {level} "
                f"({passed}/{total} tests passing)"
            )

        # Framework adapter suites
        terminalreporter.write_line("")
        terminalreporter.write_line("Framework Adapter Status:")
        for proto in ("embedding_frameworks", "graph_frameworks"):
            config = protocol_registry.get(proto)
            if not config:
                continue
            total = ctx.protocol_total.get(proto, 0)
            if total == 0:
                continue
            passed = ctx.protocol_passed.get(proto, 0)
            level, _ = self._calculate_conformance_level(proto, passed, total)
            terminalreporter.write_line(
                f"  {self._fmt('✅', 'PASS')} {config.display_name}: {level} "
                f"({passed}/{total} framework adapter tests passing)"
            )

        terminalreporter.write_line("")
        terminalreporter.write_line(
            self._fmt("🎯", "Status: Ready for production deployment")
        )
        terminalreporter.write_line(
            self._fmt("📚", "Specification: All requirements met per CORPUS Protocol Suite V1.0")
        )
        terminalreporter.write_line(
            self._fmt(
                "ℹ️",
                "Tests not under a known protocol directory are reported as 'other' and do not affect certification."
            )
        )
        
        # Performance stats
        cache_stats = test_categorizer.get_cache_stats()
        terminalreporter.write_line(
            f"{self._fmt('🔧', 'Performance:')} "
            f"{cache_stats['cache_hits']} cache hits, {cache_stats['cache_misses']} misses "
            f"(cache size: {cache_stats['cache_size']})"
        )
        terminalreporter.write_line(
            f"{self._fmt('⏱️', 'Completed in')} {ctx.duration:.2f}s"
        )

        if self.verbose and self.unmapped_categories:
            terminalreporter.write_line(
                "Unmapped categories detected (possible config drift):"
            )
            for proto, cat in sorted(self.unmapped_categories):
                terminalreporter.write_line(f"  - {proto}:{cat}")

    def _print_gold_certification(self, terminalreporter, ctx: SummaryContext):
        """Print Gold certification summary with progress to Platinum."""
        terminalreporter.write_sep("=", self._fmt("🥇", "CORPUS PROTOCOL SUITE - GOLD CERTIFIED"))

        terminalreporter.write_line("Protocol Conformance Status:")
        platinum_ready = True

        for proto in PROTOCOLS:
            config = protocol_registry.get(proto)
            if not config:
                continue

            total = ctx.protocol_total.get(proto, 0)
            if total == 0:
                continue
                
            passed = ctx.protocol_passed.get(proto, 0)
            level, needed = self._calculate_conformance_level(proto, passed, total)

            label_prefix = "PASS"
            emoji = "✅"
            if "Gold" not in level:
                platinum_ready = False
                label_prefix = "WARN"
                emoji = "⚠️"

            # For both protocol and adapter suites, show the same style line here;
            # we differentiate in the text below.
            if "Gold" in level:
                terminalreporter.write_line(
                    f"  {self._fmt('✅', 'PASS')} {config.display_name}: {level} "
                    f"({passed}/{total} tests passing)"
                )
            else:
                if "Silver" in level:
                    next_label = "Gold"
                elif "Development" in level:
                    next_label = "Silver"
                else:
                    next_label = "Development"
                terminalreporter.write_line(
                    f"  {self._fmt('⚠️', 'WARN')} {config.display_name}: {level} "
                    f"({passed}/{total} tests passing; {needed} test(s) to {next_label})"
                )

        terminalreporter.write_line("")
        if platinum_ready:
            terminalreporter.write_line(
                self._fmt("🎯", "All protocols and adapter suites at Gold level - Platinum certification available!")
            )
        else:
            terminalreporter.write_line(
                self._fmt(
                    "🎯",
                    "Focus on suites below Gold level to reach Platinum certification (100% passing across all suites)."
                )
            )

        terminalreporter.write_line(
            self._fmt(
                "ℹ️",
                "Tests not under a known protocol directory are reported as 'other' and do not affect certification."
            )
        )
        terminalreporter.write_line(
            f"{self._fmt('⏱️', 'Completed in')} {ctx.duration:.2f}s"
        )
        terminalreporter.write_line(
            self._fmt("📚", "Review CONFORMANCE.md for detailed test-to-spec mapping")
        )

        # Framework adapter suites explicit recap
        terminalreporter.write_line("")
        terminalreporter.write_line("Framework Adapter Status:")
        for proto in ("embedding_frameworks", "graph_frameworks"):
            config = protocol_registry.get(proto)
            if not config:
                continue
            total = ctx.protocol_total.get(proto, 0)
            if total == 0:
                continue
            passed = ctx.protocol_passed.get(proto, 0)
            level, needed = self._calculate_conformance_level(proto, passed, total)
            if "Gold" in level:
                terminalreporter.write_line(
                    f"  {self._fmt('✅', 'PASS')} {config.display_name}: {level} "
                    f"({passed}/{total} framework adapter tests passing)"
                )
            else:
                if "Silver" in level:
                    next_label = "Gold"
                elif "Development" in level:
                    next_label = "Silver"
                else:
                    next_label = "Development"
                terminalreporter.write_line(
                    f"  {self._fmt('⚠️', 'WARN')} {config.display_name}: {level} "
                    f"({passed}/{total} framework adapter tests passing; {needed} test(s) to {next_label})"
                )

        cache_stats = test_categorizer.get_cache_stats()
        terminalreporter.write_line(
            f"{self._fmt('🔧', 'Performance:')} "
            f"{cache_stats['cache_hits']} cache hits, {cache_stats['cache_misses']} misses "
            f"(cache size: {cache_stats['cache_size']})"
        )

        if self.verbose and self.unmapped_categories:
            terminalreporter.write_line(
                "Unmapped categories detected (possible config drift):"
            )
            for proto, cat in sorted(self.unmapped_categories):
                terminalreporter.write_line(f"  - {proto}:{cat}")

    def _print_failure_analysis(
        self,
        terminalreporter,
        ctx: SummaryContext,
        by_protocol: Dict[str, Dict[str, List[Any]]],
    ):
        """Print detailed failure analysis with actionable guidance."""
        terminalreporter.write_sep("=", self._fmt("❌", "CORPUS PROTOCOL CONFORMANCE ANALYSIS"))

        total_failures = 0
        for proto_failures in by_protocol.values():
            for reports_list in proto_failures.values():
                total_failures += len(reports_list)

        terminalreporter.write_line(
            f"Found {total_failures} conformance issue(s) across protocols and adapter suites:"
        )
        terminalreporter.write_line("")

        # Show failures by protocol and category with specific guidance
        for proto, categories in by_protocol.items():
            if not categories:
                continue
            self._print_protocol_failure_summary(terminalreporter, proto, categories)

        # Certification impact
        terminalreporter.write_line("Certification Impact:")
        failing_protocols = [p for p, cats in by_protocol.items() if cats and p != "other"]

        if failing_protocols:
            terminalreporter.write_line(
                f"  {self._fmt('⚠️', 'Platinum certification blocked by failures in:')}"
            )
            for proto in failing_protocols:
                config = protocol_registry.get(proto)
                display_name = config.display_name if config else proto.upper()
                terminalreporter.write_line(f"      - {display_name}")
        else:
            terminalreporter.write_line(
                f"  {self._fmt('✅', 'No protocol conformance failures - review ''other'' category tests')}"
            )
            terminalreporter.write_line(
                "     (Tests under 'other' are not part of CORPUS protocol certification but may still be important.)"
            )

        terminalreporter.write_line("")
        terminalreporter.write_line("Next Steps:")
        terminalreporter.write_line("  1. Review failing tests above with spec section references")
        terminalreporter.write_line("  2. Check SPECIFICATION.md for detailed requirements") 
        terminalreporter.write_line("  3. Run individual protocol tests: make test-{protocol}-conformance")
        terminalreporter.write_line("  4. Review error guidance in test output for specific fixes")

        cache_stats = test_categorizer.get_cache_stats()
        terminalreporter.write_line(
            f"{self._fmt('🔧', 'Performance:')} "
            f"{cache_stats['cache_hits']} cache hits, {cache_stats['cache_misses']} misses "
            f"(cache size: {cache_stats['cache_size']})"
        )
        terminalreporter.write_line(
            f"{self._fmt('⏱️', 'Completed in')} {ctx.duration:.2f}s"
        )

        if self.verbose and self.unmapped_categories:
            terminalreporter.write_line(
                "Unmapped categories detected (possible config drift):"
            )
            for proto, cat in sorted(self.unmapped_categories):
                terminalreporter.write_line(f"  - {proto}:{cat}")

    # -------- Main terminal summary hook --------

    def pytest_terminal_summary(self, terminalreporter, exitstatus, config):
        """Generate comprehensive CORPUS Protocol conformance summary."""
        # Collect both test failures and internal errors
        failed_reports = []
        for key in ("failed", "error"):
            failed_reports.extend(terminalreporter.stats.get(key, []))

        # Build summary context
        counts = self._get_test_counts(terminalreporter)
        duration = self._get_duration()
        protocol_results = self._collect_protocol_results(terminalreporter)
        protocol_totals = self._collect_protocol_totals(terminalreporter)
        ctx = SummaryContext(
            protocol_passed=protocol_results,
            protocol_total=protocol_totals,
            outcome_counts=counts,
            duration=duration,
        )

        # Check if any tests actually failed
        if not failed_reports:
            if self._is_platinum_certified(ctx):
                self._print_platinum_certification(terminalreporter, ctx)
            else:
                self._print_gold_certification(terminalreporter, ctx)
            return

        # We have actual failures. Show the analysis.
        by_protocol = self._categorize_failures(failed_reports)
        self._print_failure_analysis(terminalreporter, ctx, by_protocol)

    def pytest_runtest_logstart(self, nodeid, location):
        """Show protocol being tested for better progress visibility (optional)."""
        proto, category = test_categorizer.categorize_test(nodeid)
        # Hook point for future progress reporting; intentionally quiet for now.
        _ = (proto, category)


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
        "embedding_frameworks: Embedding framework adapter tests",
        "graph_frameworks: Graph framework adapter tests",
        "wire: Wire Request Conformance tests (tests/live)",
        "schema: Schema conformance validation tests",
        "golden: Golden wire message validation tests",
        "slow: Tests that take longer to run (skip with -m 'not slow')",
        "conformance: All protocol conformance tests",
    ]

    for marker in markers:
        config.addinivalue_line("markers", marker)
    
    # If the wire suite added a --adapter option, mirror that into CORPUS_ADAPTER
    # so the shared adapter() fixture and wire tests stay in sync.
    _apply_pytest_adapter_option(config)
    
    # Validate adapter configuration early (best-effort)
    if getattr(config, "option", None) and getattr(config.option, "verbose", False):
        try:
            _get_adapter_class()
            print("✅ Adapter configuration validated successfully")
        except AdapterValidationError as e:
            print(f"❌ Adapter configuration error: {e}")
