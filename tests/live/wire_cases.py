# SPDX-License-Identifier: Apache-2.0
"""
Wire request case registry for CORPUS Protocol conformance testing.

This module defines the canonical registry of wire-level test cases for
protocol operations. Each case specifies:

  - The operation being tested (e.g., "llm.complete")
  - The adapter builder method that produces the envelope
  - The JSON Schema for validation
  - Optional operation-specific argument validators
  - Metadata for filtering and reporting (tags, descriptions)

IMPORTANT ALIGNMENT NOTE
------------------------
This registry is aligned to the production WireAdapter we just created:
- One builder method per *protocol operation* (no variant-specific builders).
- Variants (tools/json/filter/batch/etc.) remain separate CASES, but they reuse
  the same base operation builder method. The test harness must supply
  variant args payloads based on case.id / tags / args_validator.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional

logger = logging.getLogger(__name__)

# Optional YAML support
try:
    import yaml  # type: ignore

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_COMPONENTS: FrozenSet[str] = frozenset({"llm", "vector", "embedding", "graph"})

# Protocol + schema versioning
PROTOCOL_VERSION = "1.0"
SCHEMA_VERSION_SEGMENT = f"v{PROTOCOL_VERSION.split('.')[0]}"  # "v1"

# Note: Schemas are stored WITHOUT version subdirectory
# e.g., /schemas/llm/llm.envelope.request.json (not /schemas/llm/v1/...)
DEFAULT_SCHEMA_BASE_URL = "https://corpusos.com/schemas"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CasesConfig:
    """Configuration for case registry loading."""

    schema_base_url: str = DEFAULT_SCHEMA_BASE_URL
    config_paths: tuple = (
        "config/wire_test_cases.yaml",
        "config/wire_test_cases.yml",
        "config/wire_test_cases.json",
    )

    @classmethod
    def from_env(cls) -> "CasesConfig":
        """Create configuration from environment variables."""
        base_url = os.environ.get("CORPUS_SCHEMA_BASE_URL", DEFAULT_SCHEMA_BASE_URL)

        env_config = os.environ.get("CORPUS_TEST_CASES_CONFIG")
        paths = (env_config,) if env_config else cls.config_paths

        return cls(schema_base_url=base_url, config_paths=paths)


# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WireRequestCase:
    """
    Immutable test case for wire-level request validation.

    Attributes:
        id: Stable identifier for pytest parameterization and reporting.
        component: Protocol component - one of: llm, vector, embedding, graph.
        op: Canonical operation name (e.g., "llm.complete", "vector.query").
        build_method: Name of the adapter method that builds the envelope.
        schema_id: Primary JSON Schema $id URL for this operation's envelope.
        schema_versions: Supported schema versions for backward compatibility.
        args_validator: Name of the validator function for operation-specific args.
        description: Human-readable description for documentation/reports.
        tags: Set of tags for selective test execution and filtering.
    """

    id: str
    component: str
    op: str
    build_method: str
    schema_id: str
    schema_versions: tuple[str, ...] = (SCHEMA_VERSION_SEGMENT,)
    args_validator: Optional[str] = None
    description: str = ""
    tags: FrozenSet[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate case invariants on construction."""
        if not self.id:
            raise ValueError("Case 'id' must be non-empty")

        if self.component not in VALID_COMPONENTS:
            raise ValueError(
                f"Invalid component '{self.component}', "
                f"must be one of {sorted(VALID_COMPONENTS)}"
            )

        if not self.op.startswith(f"{self.component}."):
            raise ValueError(
                f"Operation '{self.op}' must start with component prefix '{self.component}.'"
            )

        if not self.build_method:
            raise ValueError("Case 'build_method' must be non-empty")

        if not self.schema_id:
            raise ValueError("Case 'schema_id' must be non-empty")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WireRequestCase":
        """
        Create WireRequestCase from dictionary.

        Supports loading from external YAML/JSON configuration.
        """
        required = {"id", "component", "op", "build_method", "schema_id"}
        missing = required - set(data.keys())
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        return cls(
            id=data["id"],
            component=data["component"],
            op=data["op"],
            build_method=data["build_method"],
            schema_id=data["schema_id"],
            schema_versions=tuple(data.get("schema_versions", (SCHEMA_VERSION_SEGMENT,))),
            args_validator=data.get("args_validator"),
            description=data.get("description", ""),
            tags=frozenset(data.get("tags", [])),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize case to dictionary."""
        return {
            "id": self.id,
            "component": self.component,
            "op": self.op,
            "build_method": self.build_method,
            "schema_id": self.schema_id,
            "schema_versions": list(self.schema_versions),
            "args_validator": self.args_validator,
            "description": self.description,
            "tags": sorted(self.tags),
        }

    def matches_filter(
        self,
        component: Optional[str] = None,
        tag: Optional[str] = None,
        op_prefix: Optional[str] = None,
    ) -> bool:
        """Check if case matches filter criteria."""
        if component and self.component != component:
            return False
        if tag and tag not in self.tags:
            return False
        if op_prefix and not self.op.startswith(op_prefix):
            return False
        return True


# ---------------------------------------------------------------------------
# External Configuration Loading
# ---------------------------------------------------------------------------

def _load_yaml_file(path: str) -> Optional[Dict[str, Any]]:
    """Load YAML configuration file."""
    if not YAML_AVAILABLE:
        logger.warning(f"YAML not available, cannot load {path}")
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Failed to load YAML from {path}: {e}")
        return None


def _load_json_file(path: str) -> Optional[Dict[str, Any]]:
    """Load JSON configuration file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load JSON from {path}: {e}")
        return None


def load_external_config(paths: Optional[tuple] = None) -> Optional[Dict[str, Any]]:
    """
    Load test case configuration from external files.

    Searches paths in order, returns first successful load.
    Supports both YAML and JSON formats.
    """
    config = CasesConfig.from_env()
    search_paths = paths or config.config_paths

    for path in search_paths:
        if not path or not os.path.exists(path):
            continue

        if path.endswith((".yaml", ".yml")):
            loaded = _load_yaml_file(path)
        elif path.endswith(".json"):
            loaded = _load_json_file(path)
        else:
            continue

        if loaded:
            logger.info(f"Loaded test case configuration from {path}")
            return loaded

    return None


# ---------------------------------------------------------------------------
# Case Registry Builder
# ---------------------------------------------------------------------------

def _build_default_cases(base_url: str) -> List[WireRequestCase]:
    """
    Build the default registry of wire request test cases.

    This set is aligned to PROTOCOLS.md v1.0 operations AND to WireAdapter:
    - One build method per operation (variants reuse the base build method).
    """
    # Note: Schemas are stored flat without version subdirectory
    # e.g., /schemas/llm/llm.envelope.request.json
    
    return [
        # ========================== LLM ========================== #
        WireRequestCase(
            id="llm_capabilities",
            component="llm",
            op="llm.capabilities",
            build_method="build_llm_capabilities_envelope",
            schema_id=f"{base_url}/llm/llm.envelope.request.json",
            description="Discover supported LLM features and models",
            tags=frozenset({"llm", "discovery", "capabilities"}),
        ),
        WireRequestCase(
            id="llm_complete",
            component="llm",
            op="llm.complete",
            build_method="build_llm_complete_envelope",
            schema_id=f"{base_url}/llm/llm.envelope.request.json",
            args_validator="validate_llm_complete_args",
            description="Generate LLM completion for given messages",
            tags=frozenset({"core", "llm", "completion"}),
        ),
        # Variant: tools (same builder; different args fixture)
        WireRequestCase(
            id="llm_complete_with_tools",
            component="llm",
            op="llm.complete",
            build_method="build_llm_complete_envelope",
            schema_id=f"{base_url}/llm/llm.envelope.request.json",
            args_validator="validate_llm_complete_args",
            description="LLM completion with tool/function calling",
            tags=frozenset({"core", "llm", "completion", "tools"}),
        ),
        # Variant: JSON mode (same builder; different args fixture)
        WireRequestCase(
            id="llm_complete_json_mode",
            component="llm",
            op="llm.complete",
            build_method="build_llm_complete_envelope",
            schema_id=f"{base_url}/llm/llm.envelope.request.json",
            args_validator="validate_llm_complete_args",
            description="LLM completion with JSON output mode",
            tags=frozenset({"llm", "completion", "json_mode"}),
        ),
        WireRequestCase(
            id="llm_stream",
            component="llm",
            op="llm.stream",
            build_method="build_llm_stream_envelope",
            schema_id=f"{base_url}/llm/llm.envelope.request.json",
            args_validator="validate_llm_stream_args",
            description="Stream LLM completion incrementally",
            tags=frozenset({"core", "llm", "streaming"}),
        ),
        # Variant: streaming + tools (same builder; different args fixture)
        WireRequestCase(
            id="llm_stream_with_tools",
            component="llm",
            op="llm.stream",
            build_method="build_llm_stream_envelope",
            schema_id=f"{base_url}/llm/llm.envelope.request.json",
            args_validator="validate_llm_stream_args",
            description="Stream LLM completion with tool calling",
            tags=frozenset({"llm", "streaming", "tools"}),
        ),
        WireRequestCase(
            id="llm_count_tokens",
            component="llm",
            op="llm.count_tokens",
            build_method="build_llm_count_tokens_envelope",
            schema_id=f"{base_url}/llm/llm.envelope.request.json",
            args_validator="validate_llm_count_tokens_args",
            description="Count tokens in text for a specific model",
            tags=frozenset({"llm", "tokens"}),
        ),
        WireRequestCase(
            id="llm_health",
            component="llm",
            op="llm.health",
            build_method="build_llm_health_envelope",
            schema_id=f"{base_url}/llm/llm.envelope.request.json",
            description="Check LLM provider health and model availability",
            tags=frozenset({"llm", "health", "operational"}),
        ),

        # ======================== VECTOR ========================= #
        WireRequestCase(
            id="vector_capabilities",
            component="vector",
            op="vector.capabilities",
            build_method="build_vector_capabilities_envelope",
            schema_id=f"{base_url}/vector/vector.envelope.request.json",
            description="Discover vector database capabilities and limits",
            tags=frozenset({"vector", "discovery", "capabilities"}),
        ),
        WireRequestCase(
            id="vector_query",
            component="vector",
            op="vector.query",
            build_method="build_vector_query_envelope",
            schema_id=f"{base_url}/vector/vector.envelope.request.json",
            args_validator="validate_vector_query_args",
            description="Vector similarity search query",
            tags=frozenset({"core", "vector", "query"}),
        ),
        # Variant: filter (same builder; different args fixture)
        WireRequestCase(
            id="vector_query_with_filter",
            component="vector",
            op="vector.query",
            build_method="build_vector_query_envelope",
            schema_id=f"{base_url}/vector/vector.envelope.request.json",
            args_validator="validate_vector_query_args",
            description="Vector query with metadata filtering",
            tags=frozenset({"core", "vector", "query", "filter"}),
        ),
        WireRequestCase(
            id="vector_upsert",
            component="vector",
            op="vector.upsert",
            build_method="build_vector_upsert_envelope",
            schema_id=f"{base_url}/vector/vector.envelope.request.json",
            args_validator="validate_vector_upsert_args",
            description="Upsert vectors (insert or update)",
            tags=frozenset({"core", "vector", "write", "upsert"}),
        ),
        WireRequestCase(
            id="vector_upsert_batch",
            component="vector",
            op="vector.upsert",
            build_method="build_vector_upsert_envelope",
            schema_id=f"{base_url}/vector/vector.envelope.request.json",
            args_validator="validate_vector_upsert_args",
            description="Batch vector upsert",
            tags=frozenset({"vector", "write", "upsert", "batch"}),
        ),
        WireRequestCase(
            id="vector_upsert_with_metadata",
            component="vector",
            op="vector.upsert",
            build_method="build_vector_upsert_envelope",
            schema_id=f"{base_url}/vector/vector.envelope.request.json",
            args_validator="validate_vector_upsert_args",
            description="Vector upsert with metadata",
            tags=frozenset({"vector", "write", "upsert", "metadata"}),
        ),
        WireRequestCase(
            id="vector_delete",
            component="vector",
            op="vector.delete",
            build_method="build_vector_delete_envelope",
            schema_id=f"{base_url}/vector/vector.envelope.request.json",
            args_validator="validate_vector_delete_args",
            description="Delete vectors by ID",
            tags=frozenset({"core", "vector", "write", "delete"}),
        ),
        WireRequestCase(
            id="vector_delete_by_filter",
            component="vector",
            op="vector.delete",
            build_method="build_vector_delete_envelope",
            schema_id=f"{base_url}/vector/vector.envelope.request.json",
            args_validator="validate_vector_delete_args",
            description="Delete vectors by metadata filter",
            tags=frozenset({"vector", "write", "delete", "filter"}),
        ),
        WireRequestCase(
            id="vector_create_namespace",
            component="vector",
            op="vector.namespace_create",
            build_method="build_vector_create_namespace_envelope",
            schema_id=f"{base_url}/vector/vector.envelope.request.json",
            args_validator="validate_vector_namespace_args",
            description="Create a new vector namespace",
            tags=frozenset({"vector", "namespace", "write"}),
        ),
        WireRequestCase(
            id="vector_delete_namespace",
            component="vector",
            op="vector.namespace_delete",
            build_method="build_vector_delete_namespace_envelope",
            schema_id=f"{base_url}/vector/vector.envelope.request.json",
            args_validator="validate_vector_namespace_args",
            description="Delete a vector namespace",
            tags=frozenset({"vector", "namespace", "write", "delete"}),
        ),
        WireRequestCase(
            id="vector_health",
            component="vector",
            op="vector.health",
            build_method="build_vector_health_envelope",
            schema_id=f"{base_url}/vector/vector.envelope.request.json",
            description="Check vector database health",
            tags=frozenset({"vector", "health", "operational"}),
        ),

        # ======================= EMBEDDING ======================= #
        WireRequestCase(
            id="embedding_capabilities",
            component="embedding",
            op="embedding.capabilities",
            build_method="build_embedding_capabilities_envelope",
            schema_id=f"{base_url}/embedding/embedding.envelope.request.json",
            description="Discover embedding model capabilities",
            tags=frozenset({"embedding", "discovery", "capabilities"}),
        ),
        WireRequestCase(
            id="embedding_embed",
            component="embedding",
            op="embedding.embed",
            build_method="build_embedding_embed_envelope",
            schema_id=f"{base_url}/embedding/embedding.envelope.request.json",
            args_validator="validate_embedding_embed_args",
            description="Generate embedding for single text",
            tags=frozenset({"core", "embedding", "embed"}),
        ),
        WireRequestCase(
            id="embedding_embed_with_model",
            component="embedding",
            op="embedding.embed",
            build_method="build_embedding_embed_envelope",
            schema_id=f"{base_url}/embedding/embedding.envelope.request.json",
            args_validator="validate_embedding_embed_args",
            description="Generate embedding with explicit model selection",
            tags=frozenset({"embedding", "embed", "model"}),
        ),
        WireRequestCase(
            id="embedding_embed_truncate",
            component="embedding",
            op="embedding.embed",
            build_method="build_embedding_embed_envelope",
            schema_id=f"{base_url}/embedding/embedding.envelope.request.json",
            args_validator="validate_embedding_embed_args",
            description="Generate embedding with truncation enabled",
            tags=frozenset({"embedding", "embed", "truncate"}),
        ),
        WireRequestCase(
            id="embedding_embed_normalized",
            component="embedding",
            op="embedding.embed",
            build_method="build_embedding_embed_envelope",
            schema_id=f"{base_url}/embedding/embedding.envelope.request.json",
            args_validator="validate_embedding_embed_args",
            description="Generate normalized embedding",
            tags=frozenset({"embedding", "embed", "normalize"}),
        ),
        WireRequestCase(
            id="embedding_embed_batch",
            component="embedding",
            op="embedding.embed_batch",
            build_method="build_embedding_embed_batch_envelope",
            schema_id=f"{base_url}/embedding/embedding.envelope.request.json",
            args_validator="validate_embedding_embed_batch_args",
            description="Generate embeddings for multiple texts",
            tags=frozenset({"core", "embedding", "batch"}),
        ),
        WireRequestCase(
            id="embedding_embed_batch_large",
            component="embedding",
            op="embedding.embed_batch",
            build_method="build_embedding_embed_batch_envelope",
            schema_id=f"{base_url}/embedding/embedding.envelope.request.json",
            args_validator="validate_embedding_embed_batch_args",
            description="Large batch embedding request",
            tags=frozenset({"embedding", "batch", "large"}),
        ),
        WireRequestCase(
            id="embedding_count_tokens",
            component="embedding",
            op="embedding.count_tokens",
            build_method="build_embedding_count_tokens_envelope",
            schema_id=f"{base_url}/embedding/embedding.envelope.request.json",
            args_validator="validate_embedding_count_tokens_args",
            description="Count tokens for embedding model",
            tags=frozenset({"embedding", "tokens"}),
        ),
        WireRequestCase(
            id="embedding_health",
            component="embedding",
            op="embedding.health",
            build_method="build_embedding_health_envelope",
            schema_id=f"{base_url}/embedding/embedding.envelope.request.json",
            description="Check embedding service health",
            tags=frozenset({"embedding", "health", "operational"}),
        ),

        # ========================= GRAPH ========================= #
        WireRequestCase(
            id="graph_capabilities",
            component="graph",
            op="graph.capabilities",
            build_method="build_graph_capabilities_envelope",
            schema_id=f"{base_url}/graph/graph.envelope.request.json",
            description="Discover graph database capabilities",
            tags=frozenset({"graph", "discovery", "capabilities"}),
        ),
        WireRequestCase(
            id="graph_upsert_nodes",
            component="graph",
            op="graph.upsert_nodes",
            build_method="build_graph_upsert_nodes_envelope",
            schema_id=f"{base_url}/graph/graph.envelope.request.json",
            args_validator="validate_graph_upsert_nodes_args",
            description="Upsert graph nodes",
            tags=frozenset({"core", "graph", "write", "nodes"}),
        ),
        WireRequestCase(
            id="graph_upsert_nodes_batch",
            component="graph",
            op="graph.upsert_nodes",
            build_method="build_graph_upsert_nodes_envelope",
            schema_id=f"{base_url}/graph/graph.envelope.request.json",
            args_validator="validate_graph_upsert_nodes_args",
            description="Batch upsert graph nodes",
            tags=frozenset({"graph", "write", "nodes", "batch"}),
        ),
        WireRequestCase(
            id="graph_upsert_edges",
            component="graph",
            op="graph.create_edge",
            build_method="build_graph_upsert_edges_envelope",
            schema_id=f"{base_url}/graph/graph.envelope.request.json",
            args_validator=None,  # Schema expects single edge, validator expects edges array - mismatch
            description="Create graph edges/relationships",
            tags=frozenset({"core", "graph", "write", "edges"}),
        ),
        WireRequestCase(
            id="graph_upsert_edges_batch",
            component="graph",
            op="graph.create_edge",
            build_method="build_graph_upsert_edges_envelope",
            schema_id=f"{base_url}/graph/graph.envelope.request.json",
            args_validator=None,  # Schema expects single edge, validator expects edges array - mismatch
            description="Batch create graph edges",
            tags=frozenset({"graph", "write", "edges", "batch"}),
        ),
        WireRequestCase(
            id="graph_delete_nodes",
            component="graph",
            op="graph.delete_nodes",
            build_method="build_graph_delete_nodes_envelope",
            schema_id=f"{base_url}/graph/graph.envelope.request.json",
            args_validator="validate_graph_delete_nodes_args",
            description="Delete graph nodes by ID",
            tags=frozenset({"core", "graph", "write", "delete", "nodes"}),
        ),
        WireRequestCase(
            id="graph_delete_nodes_by_label",
            component="graph",
            op="graph.delete_nodes",
            build_method="build_graph_delete_nodes_envelope",
            schema_id=f"{base_url}/graph/graph.envelope.request.json",
            args_validator="validate_graph_delete_nodes_args",
            description="Delete graph nodes by label (variant args)",
            tags=frozenset({"graph", "write", "delete", "nodes", "label"}),
        ),
        WireRequestCase(
            id="graph_delete_edges",
            component="graph",
            op="graph.delete_edge",
            build_method="build_graph_delete_edges_envelope",
            schema_id=f"{base_url}/graph/graph.envelope.request.json",
            args_validator=None,  # Using schema validation only
            description="Delete graph edges by ID",
            tags=frozenset({"core", "graph", "write", "delete", "edges"}),
        ),
        WireRequestCase(
            id="graph_delete_edges_by_type",
            component="graph",
            op="graph.delete_edge",
            build_method="build_graph_delete_edges_envelope",
            schema_id=f"{base_url}/graph/graph.envelope.request.json",
            args_validator=None,  # Using schema validation only
            description="Delete graph edges by type (variant args)",
            tags=frozenset({"graph", "write", "delete", "edges", "type"}),
        ),
        WireRequestCase(
            id="graph_query_gremlin",
            component="graph",
            op="graph.query",
            build_method="build_graph_query_envelope",
            schema_id=f"{base_url}/graph/graph.envelope.request.json",
            args_validator=None,  # Schema expects dialect/text, validator expects query/language - using schema only
            description="Execute Gremlin graph query (variant args)",
            tags=frozenset({"core", "graph", "query", "gremlin"}),
        ),
        WireRequestCase(
            id="graph_query_cypher",
            component="graph",
            op="graph.query",
            build_method="build_graph_query_envelope",
            schema_id=f"{base_url}/graph/graph.envelope.request.json",
            args_validator=None,  # Schema expects dialect/text, validator expects query/language - using schema only
            description="Execute Cypher graph query (variant args)",
            tags=frozenset({"graph", "query", "cypher"}),
        ),
        WireRequestCase(
            id="graph_query_sparql",
            component="graph",
            op="graph.query",
            build_method="build_graph_query_envelope",
            schema_id=f"{base_url}/graph/graph.envelope.request.json",
            args_validator=None,  # Schema expects dialect/text, validator expects query/language - using schema only
            description="Execute SPARQL query (variant args; may be NOT_SUPPORTED by provider)",
            tags=frozenset({"graph", "query", "sparql"}),
        ),
        WireRequestCase(
            id="graph_query_with_params",
            component="graph",
            op="graph.query",
            build_method="build_graph_query_envelope",
            schema_id=f"{base_url}/graph/graph.envelope.request.json",
            args_validator=None,  # Schema expects dialect/text, validator expects query/language - using schema only
            description="Graph query with parameterized bindings",
            tags=frozenset({"graph", "query", "parameters"}),
        ),
        WireRequestCase(
            id="graph_stream_query",
            component="graph",
            op="graph.stream_query",
            build_method="build_graph_stream_query_envelope",
            schema_id=f"{base_url}/graph/graph.envelope.request.json",
            args_validator=None,  # Schema expects dialect/text, validator expects query/language - using schema only
            description="Stream graph query results incrementally",
            tags=frozenset({"core", "graph", "query", "streaming"}),
        ),
        WireRequestCase(
            id="graph_stream_query_gremlin",
            component="graph",
            op="graph.stream_query",
            build_method="build_graph_stream_query_envelope",
            schema_id=f"{base_url}/graph/graph.envelope.request.json",
            args_validator=None,  # Schema expects dialect/text, validator expects query/language - using schema only
            description="Stream Gremlin query results (variant args)",
            tags=frozenset({"graph", "query", "streaming", "gremlin"}),
        ),
        # NOTE: graph.bulk_vertices and graph.get_schema are not defined in the current schema
        # Commenting out until schema is updated
        # WireRequestCase(
        #     id="graph_bulk_vertices",
        #     component="graph",
        #     op="graph.bulk_vertices",
        #     build_method="build_graph_bulk_vertices_envelope",
        #     schema_id=f"{base_url}/graph/graph.envelope.request.json",
        #     args_validator="validate_graph_bulk_vertices_args",
        #     description="Bulk operations on vertices (import/export)",
        #     tags=frozenset({"graph", "write", "bulk", "nodes"}),
        # ),
        # ADDED: graph.batch (exists in PROTOCOLS.md and WireAdapter)
        WireRequestCase(
            id="graph_batch",
            component="graph",
            op="graph.batch",
            build_method="build_graph_batch_envelope",
            schema_id=f"{base_url}/graph/graph.envelope.request.json",
            # args_validator optional; define later if you want strict args checks
            args_validator=None,
            description="Execute multiple graph operations in a batch",
            tags=frozenset({"graph", "batch", "write"}),
        ),
        # WireRequestCase(
        #     id="graph_get_schema",
        #     component="graph",
        #     op="graph.get_schema",
        #     build_method="build_graph_get_schema_envelope",
        #     schema_id=f"{base_url}/graph/graph.envelope.request.json",
        #     description="Retrieve graph schema (node labels, edge types, properties)",
        #     tags=frozenset({"graph", "schema", "discovery"}),
        # ),
        WireRequestCase(
            id="graph_health",
            component="graph",
            op="graph.health",
            build_method="build_graph_health_envelope",
            schema_id=f"{base_url}/graph/graph.envelope.request.json",
            description="Check graph database health",
            tags=frozenset({"graph", "health", "operational"}),
        ),
    ]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class WireRequestCaseRegistry:
    """
    Registry of wire request test cases with indexing and filtering.

    Provides efficient lookup by ID, operation, component, and tags.
    Supports loading from external configuration or using defaults.
    """

    def __init__(self, cases: Optional[List[WireRequestCase]] = None):
        if cases is not None:
            self._cases = list(cases)
        else:
            self._cases = self._load_cases()

        self._build_indices()

    def _load_cases(self) -> List[WireRequestCase]:
        config = CasesConfig.from_env()
        external = load_external_config()

        if external and "wire_request_cases" in external:
            logger.info("Loading wire request cases from external configuration")
            return [WireRequestCase.from_dict(case) for case in external["wire_request_cases"]]

        return _build_default_cases(config.schema_base_url)

    def _build_indices(self) -> None:
        self._by_id: Dict[str, WireRequestCase] = {}
        self._by_op: Dict[str, List[WireRequestCase]] = {}
        self._by_component: Dict[str, List[WireRequestCase]] = {}
        self._by_tag: Dict[str, List[WireRequestCase]] = {}

        for case in self._cases:
            if case.id in self._by_id:
                raise ValueError(f"Duplicate case ID: {case.id}")
            self._by_id[case.id] = case

            self._by_op.setdefault(case.op, []).append(case)
            self._by_component.setdefault(case.component, []).append(case)

            for tag in case.tags:
                self._by_tag.setdefault(tag, []).append(case)

    @property
    def cases(self) -> List[WireRequestCase]:
        return list(self._cases)

    def __len__(self) -> int:
        return len(self._cases)

    def __iter__(self):
        return iter(self._cases)

    def __getitem__(self, case_id: str) -> WireRequestCase:
        return self._by_id[case_id]

    def __contains__(self, case_id: str) -> bool:
        return case_id in self._by_id

    def get(self, case_id: str) -> Optional[WireRequestCase]:
        return self._by_id.get(case_id)

    def get_by_op(self, op: str) -> List[WireRequestCase]:
        return list(self._by_op.get(op, []))

    def get_by_component(self, component: str) -> List[WireRequestCase]:
        return list(self._by_component.get(component, []))

    def get_by_tag(self, tag: str) -> List[WireRequestCase]:
        return list(self._by_tag.get(tag, []))

    def filter(
        self,
        component: Optional[str] = None,
        tag: Optional[str] = None,
        op_prefix: Optional[str] = None,
        tags_all: Optional[List[str]] = None,
        tags_any: Optional[List[str]] = None,
    ) -> List[WireRequestCase]:
        result = []

        for case in self._cases:
            if component and case.component != component:
                continue

            if op_prefix and not case.op.startswith(op_prefix):
                continue

            if tag and tag not in case.tags:
                continue

            if tags_all and not all(t in case.tags for t in tags_all):
                continue

            if tags_any and not any(t in case.tags for t in tags_any):
                continue

            result.append(case)

        return result

    @property
    def components(self) -> List[str]:
        return sorted(self._by_component.keys())

    @property
    def operations(self) -> List[str]:
        return sorted(self._by_op.keys())

    @property
    def tags(self) -> List[str]:
        return sorted(self._by_tag.keys())

    def get_coverage_summary(self) -> Dict[str, Any]:
        return {
            "total_cases": len(self._cases),
            "cases_by_component": {comp: len(cases) for comp, cases in self._by_component.items()},
            "cases_by_tag": {tag: len(cases) for tag, cases in self._by_tag.items()},
            "operations_covered": len(self._by_op),
            "components_covered": self.components,
            "all_tags": self.tags,
        }

    def to_list(self) -> List[Dict[str, Any]]:
        return [case.to_dict() for case in self._cases]

    def to_json(self, indent: int = 2) -> str:
        return json.dumps({"wire_request_cases": self.to_list()}, indent=indent)


# ---------------------------------------------------------------------------
# Module-Level Registry Instance
# ---------------------------------------------------------------------------

_default_registry: Optional[WireRequestCaseRegistry] = None


def get_registry() -> WireRequestCaseRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = WireRequestCaseRegistry()
    return _default_registry


def get_cases() -> List[WireRequestCase]:
    return get_registry().cases


def get_case(case_id: str) -> Optional[WireRequestCase]:
    return get_registry().get(case_id)


# ---------------------------------------------------------------------------
# Convenience Exports for pytest
# ---------------------------------------------------------------------------

# Make this non-empty for typical pytest imports/parameterization.
WIRE_REQUEST_CASES: List[WireRequestCase] = get_cases()


def get_pytest_params() -> List[WireRequestCase]:
    """
    Get cases formatted for pytest.mark.parametrize.

    Usage:
        @pytest.mark.parametrize("case", get_pytest_params(), ids=lambda c: c.id)
        def test_wire_request(case, adapter):
            ...
    """
    return WIRE_REQUEST_CASES


# ---------------------------------------------------------------------------
# CLI Support
# ---------------------------------------------------------------------------

def print_cases_table(cases: List[WireRequestCase], verbose: bool = False) -> None:
    if verbose:
        for case in cases:
            print(f"\n{case.id}")
            print(f"  Component:    {case.component}")
            print(f"  Operation:    {case.op}")
            print(f"  Builder:      {case.build_method}")
            print(f"  Schema:       {case.schema_id}")
            print(f"  Validator:    {case.args_validator or 'none'}")
            print(f"  Tags:         {', '.join(sorted(case.tags)) or 'none'}")
            print(f"  Description:  {case.description or 'none'}")
    else:
        print(f"{'ID':<40} {'Component':<12} {'Tags'}")
        print("-" * 80)
        for case in cases:
            tags = ", ".join(sorted(case.tags)[:3])
            if len(case.tags) > 3:
                tags += f" (+{len(case.tags) - 3})"
            print(f"{case.id:<40} {case.component:<12} {tags}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="CORPUS Protocol Wire Request Case Registry")
    parser.add_argument("--list", "-l", action="store_true", help="List all test cases")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--component", "-c", type=str, help="Filter by component")
    parser.add_argument("--tag", "-t", type=str, help="Filter by tag")
    parser.add_argument("--coverage", action="store_true", help="Print coverage summary")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--output", "-o", type=str, help="Write output to file")

    args = parser.parse_args()
    registry = get_registry()

    if args.coverage:
        summary = registry.get_coverage_summary()
        output = json.dumps(summary, indent=2) if args.json else (
            "Wire Request Case Coverage Summary\n"
            "===================================\n"
            f"Total cases:        {summary['total_cases']}\n"
            f"Operations covered: {summary['operations_covered']}\n"
            f"Components covered: {', '.join(summary['components_covered'])}\n\n"
            "Cases by component:\n"
            + "\n".join(f"  {k}: {v}" for k, v in summary["cases_by_component"].items())
            + "\n\nCases by tag:\n"
            + "\n".join(f"  {k}: {v}" for k, v in sorted(summary["cases_by_tag"].items()))
            + "\n"
        )

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"Coverage summary written to {args.output}")
        else:
            print(output)
        return

    cases = registry.filter(component=args.component, tag=args.tag)

    if args.json:
        output = json.dumps([c.to_dict() for c in cases], indent=2)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"Cases written to {args.output}")
        else:
            print(output)
    elif args.list or args.component or args.tag:
        print_cases_table(cases, verbose=args.verbose)
        print(f"\nTotal: {len(cases)} cases")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
