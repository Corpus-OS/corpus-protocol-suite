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

Separated from test execution to allow:
  - Reuse across different test harnesses
  - External configuration via YAML/JSON
  - Tooling (coverage reports, OpenAPI generation, etc.)
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
    import yaml
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
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WireRequestCase":
        """
        Create WireRequestCase from dictionary.
        
        Supports loading from external YAML/JSON configuration.
        """
        required = {"id", "component", "op", "build_method"}
        missing = required - set(data.keys())
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        
        return cls(
            id=data["id"],
            component=data["component"],
            op=data["op"],
            build_method=data["build_method"],
            schema_id=data.get("schema_id", ""),
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
    
    Args:
        paths: Tuple of file paths to search. If None, uses default paths.
    
    Returns:
        Loaded configuration dict, or None if no config found.
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
    
    This is the canonical set of protocol operations that should be tested
    for wire-level conformance. Matches the Operation Registry in the
    Corpus Protocol Specification v1.0.
    """
    v = SCHEMA_VERSION_SEGMENT

    return [
        # ========================== LLM ========================== #
        # §11.1 capabilities
        WireRequestCase(
            id="llm_capabilities",
            component="llm",
            op="llm.capabilities",
            build_method="build_llm_capabilities_envelope",
            schema_id=f"{base_url}/llm/{v}/llm.envelope.request.json",
            description="Discover supported LLM features and models",
            tags=frozenset({"llm", "discovery", "capabilities"}),
        ),
        # §11.2 complete
        WireRequestCase(
            id="llm_complete",
            component="llm",
            op="llm.complete",
            build_method="build_llm_complete_envelope",
            schema_id=f"{base_url}/llm/{v}/llm.envelope.request.json",
            args_validator="validate_llm_complete_args",
            description="Generate LLM completion for given messages",
            tags=frozenset({"core", "llm", "completion"}),
        ),
        WireRequestCase(
            id="llm_complete_with_tools",
            component="llm",
            op="llm.complete",
            build_method="build_llm_complete_with_tools_envelope",
            schema_id=f"{base_url}/llm/{v}/llm.envelope.request.json",
            args_validator="validate_llm_complete_args",
            description="LLM completion with tool/function calling",
            tags=frozenset({"core", "llm", "completion", "tools"}),
        ),
        WireRequestCase(
            id="llm_complete_json_mode",
            component="llm",
            op="llm.complete",
            build_method="build_llm_complete_json_mode_envelope",
            schema_id=f"{base_url}/llm/{v}/llm.envelope.request.json",
            args_validator="validate_llm_complete_args",
            description="LLM completion with JSON output mode",
            tags=frozenset({"llm", "completion", "json_mode"}),
        ),
        # §11.3 stream
        WireRequestCase(
            id="llm_stream",
            component="llm",
            op="llm.stream",
            build_method="build_llm_stream_envelope",
            schema_id=f"{base_url}/llm/{v}/llm.envelope.request.json",
            args_validator="validate_llm_stream_args",
            description="Stream LLM completion incrementally",
            tags=frozenset({"core", "llm", "streaming"}),
        ),
        WireRequestCase(
            id="llm_stream_with_tools",
            component="llm",
            op="llm.stream",
            build_method="build_llm_stream_with_tools_envelope",
            schema_id=f"{base_url}/llm/{v}/llm.envelope.request.json",
            args_validator="validate_llm_stream_args",
            description="Stream LLM completion with tool calling",
            tags=frozenset({"llm", "streaming", "tools"}),
        ),
        # §11.4 count_tokens
        WireRequestCase(
            id="llm_count_tokens",
            component="llm",
            op="llm.count_tokens",
            build_method="build_llm_count_tokens_envelope",
            schema_id=f"{base_url}/llm/{v}/llm.envelope.request.json",
            args_validator="validate_llm_count_tokens_args",
            description="Count tokens in text for a specific model",
            tags=frozenset({"llm", "tokens"}),
        ),
        # §11.5 health
        WireRequestCase(
            id="llm_health",
            component="llm",
            op="llm.health",
            build_method="build_llm_health_envelope",
            schema_id=f"{base_url}/llm/{v}/llm.envelope.request.json",
            description="Check LLM provider health and model availability",
            tags=frozenset({"llm", "health", "operational"}),
        ),

        # ======================== VECTOR ========================= #
        # §15.1 capabilities
        WireRequestCase(
            id="vector_capabilities",
            component="vector",
            op="vector.capabilities",
            build_method="build_vector_capabilities_envelope",
            schema_id=f"{base_url}/vector/{v}/vector.envelope.request.json",
            description="Discover vector database capabilities and limits",
            tags=frozenset({"vector", "discovery", "capabilities"}),
        ),
        # §15.2 query
        WireRequestCase(
            id="vector_query",
            component="vector",
            op="vector.query",
            build_method="build_vector_query_envelope",
            schema_id=f"{base_url}/vector/{v}/vector.envelope.request.json",
            args_validator="validate_vector_query_args",
            description="Vector similarity search query",
            tags=frozenset({"core", "vector", "query"}),
        ),
        WireRequestCase(
            id="vector_query_with_filter",
            component="vector",
            op="vector.query",
            build_method="build_vector_query_with_filter_envelope",
            schema_id=f"{base_url}/vector/{v}/vector.envelope.request.json",
            args_validator="validate_vector_query_args",
            description="Vector query with metadata filtering",
            tags=frozenset({"core", "vector", "query", "filter"}),
        ),
        WireRequestCase(
            id="vector_query_by_text",
            component="vector",
            op="vector.query",
            build_method="build_vector_query_by_text_envelope",
            schema_id=f"{base_url}/vector/{v}/vector.envelope.request.json",
            args_validator="validate_vector_query_args",
            description="Vector query using text (auto-embedded)",
            tags=frozenset({"vector", "query", "text"}),
        ),
        # §15.3 upsert
        WireRequestCase(
            id="vector_upsert",
            component="vector",
            op="vector.upsert",
            build_method="build_vector_upsert_envelope",
            schema_id=f"{base_url}/vector/{v}/vector.envelope.request.json",
            args_validator="validate_vector_upsert_args",
            description="Upsert vectors (insert or update)",
            tags=frozenset({"core", "vector", "write", "upsert"}),
        ),
        WireRequestCase(
            id="vector_upsert_batch",
            component="vector",
            op="vector.upsert",
            build_method="build_vector_upsert_batch_envelope",
            schema_id=f"{base_url}/vector/{v}/vector.envelope.request.json",
            args_validator="validate_vector_upsert_args",
            description="Batch vector upsert",
            tags=frozenset({"vector", "write", "upsert", "batch"}),
        ),
        WireRequestCase(
            id="vector_upsert_with_metadata",
            component="vector",
            op="vector.upsert",
            build_method="build_vector_upsert_with_metadata_envelope",
            schema_id=f"{base_url}/vector/{v}/vector.envelope.request.json",
            args_validator="validate_vector_upsert_args",
            description="Vector upsert with metadata",
            tags=frozenset({"vector", "write", "upsert", "metadata"}),
        ),
        # §15.4 delete
        WireRequestCase(
            id="vector_delete",
            component="vector",
            op="vector.delete",
            build_method="build_vector_delete_envelope",
            schema_id=f"{base_url}/vector/{v}/vector.envelope.request.json",
            args_validator="validate_vector_delete_args",
            description="Delete vectors by ID",
            tags=frozenset({"core", "vector", "write", "delete"}),
        ),
        WireRequestCase(
            id="vector_delete_by_filter",
            component="vector",
            op="vector.delete",
            build_method="build_vector_delete_by_filter_envelope",
            schema_id=f"{base_url}/vector/{v}/vector.envelope.request.json",
            args_validator="validate_vector_delete_args",
            description="Delete vectors by metadata filter",
            tags=frozenset({"vector", "write", "delete", "filter"}),
        ),
        WireRequestCase(
            id="vector_delete_all",
            component="vector",
            op="vector.delete",
            build_method="build_vector_delete_all_envelope",
            schema_id=f"{base_url}/vector/{v}/vector.envelope.request.json",
            args_validator="validate_vector_delete_args",
            description="Delete all vectors in namespace",
            tags=frozenset({"vector", "write", "delete"}),
        ),
        # §15.5 create_namespace
        WireRequestCase(
            id="vector_create_namespace",
            component="vector",
            op="vector.create_namespace",
            build_method="build_vector_create_namespace_envelope",
            schema_id=f"{base_url}/vector/{v}/vector.envelope.request.json",
            args_validator="validate_vector_namespace_args",
            description="Create a new vector namespace",
            tags=frozenset({"vector", "namespace", "write"}),
        ),
        # §15.6 delete_namespace
        WireRequestCase(
            id="vector_delete_namespace",
            component="vector",
            op="vector.delete_namespace",
            build_method="build_vector_delete_namespace_envelope",
            schema_id=f"{base_url}/vector/{v}/vector.envelope.request.json",
            args_validator="validate_vector_namespace_args",
            description="Delete a vector namespace",
            tags=frozenset({"vector", "namespace", "write", "delete"}),
        ),
        # §15.7 health
        WireRequestCase(
            id="vector_health",
            component="vector",
            op="vector.health",
            build_method="build_vector_health_envelope",
            schema_id=f"{base_url}/vector/{v}/vector.envelope.request.json",
            description="Check vector database health",
            tags=frozenset({"vector", "health", "operational"}),
        ),

        # ======================= EMBEDDING ======================= #
        # §19.1 capabilities
        WireRequestCase(
            id="embedding_capabilities",
            component="embedding",
            op="embedding.capabilities",
            build_method="build_embedding_capabilities_envelope",
            schema_id=f"{base_url}/embedding/{v}/embedding.envelope.request.json",
            description="Discover embedding model capabilities",
            tags=frozenset({"embedding", "discovery", "capabilities"}),
        ),
        # §19.2 embed
        WireRequestCase(
            id="embedding_embed",
            component="embedding",
            op="embedding.embed",
            build_method="build_embedding_embed_envelope",
            schema_id=f"{base_url}/embedding/{v}/embedding.envelope.request.json",
            args_validator="validate_embedding_embed_args",
            description="Generate embedding for single text",
            tags=frozenset({"core", "embedding", "embed"}),
        ),
        WireRequestCase(
            id="embedding_embed_with_model",
            component="embedding",
            op="embedding.embed",
            build_method="build_embedding_embed_with_model_envelope",
            schema_id=f"{base_url}/embedding/{v}/embedding.envelope.request.json",
            args_validator="validate_embedding_embed_args",
            description="Generate embedding with explicit model selection",
            tags=frozenset({"embedding", "embed", "model"}),
        ),
        WireRequestCase(
            id="embedding_embed_truncate",
            component="embedding",
            op="embedding.embed",
            build_method="build_embedding_embed_truncate_envelope",
            schema_id=f"{base_url}/embedding/{v}/embedding.envelope.request.json",
            args_validator="validate_embedding_embed_args",
            description="Generate embedding with truncation enabled",
            tags=frozenset({"embedding", "embed", "truncate"}),
        ),
        WireRequestCase(
            id="embedding_embed_normalized",
            component="embedding",
            op="embedding.embed",
            build_method="build_embedding_embed_normalized_envelope",
            schema_id=f"{base_url}/embedding/{v}/embedding.envelope.request.json",
            args_validator="validate_embedding_embed_args",
            description="Generate normalized embedding",
            tags=frozenset({"embedding", "embed", "normalize"}),
        ),
        # §19.3 embed_batch
        WireRequestCase(
            id="embedding_embed_batch",
            component="embedding",
            op="embedding.embed_batch",
            build_method="build_embedding_embed_batch_envelope",
            schema_id=f"{base_url}/embedding/{v}/embedding.envelope.request.json",
            args_validator="validate_embedding_embed_batch_args",
            description="Generate embeddings for multiple texts",
            tags=frozenset({"core", "embedding", "batch"}),
        ),
        WireRequestCase(
            id="embedding_embed_batch_large",
            component="embedding",
            op="embedding.embed_batch",
            build_method="build_embedding_embed_batch_large_envelope",
            schema_id=f"{base_url}/embedding/{v}/embedding.envelope.request.json",
            args_validator="validate_embedding_embed_batch_args",
            description="Large batch embedding request",
            tags=frozenset({"embedding", "batch", "large"}),
        ),
        # §19.4 count_tokens
        WireRequestCase(
            id="embedding_count_tokens",
            component="embedding",
            op="embedding.count_tokens",
            build_method="build_embedding_count_tokens_envelope",
            schema_id=f"{base_url}/embedding/{v}/embedding.envelope.request.json",
            args_validator="validate_embedding_count_tokens_args",
            description="Count tokens for embedding model",
            tags=frozenset({"embedding", "tokens"}),
        ),
        # §19.5 health
        WireRequestCase(
            id="embedding_health",
            component="embedding",
            op="embedding.health",
            build_method="build_embedding_health_envelope",
            schema_id=f"{base_url}/embedding/{v}/embedding.envelope.request.json",
            description="Check embedding service health",
            tags=frozenset({"embedding", "health", "operational"}),
        ),

        # ========================= GRAPH ========================= #
        # §7.1 capabilities
        WireRequestCase(
            id="graph_capabilities",
            component="graph",
            op="graph.capabilities",
            build_method="build_graph_capabilities_envelope",
            schema_id=f"{base_url}/graph/{v}/graph.envelope.request.json",
            description="Discover graph database capabilities",
            tags=frozenset({"graph", "discovery", "capabilities"}),
        ),
        # §7.2 upsert_nodes
        WireRequestCase(
            id="graph_upsert_nodes",
            component="graph",
            op="graph.upsert_nodes",
            build_method="build_graph_upsert_nodes_envelope",
            schema_id=f"{base_url}/graph/{v}/graph.envelope.request.json",
            args_validator="validate_graph_upsert_nodes_args",
            description="Upsert graph nodes/vertices",
            tags=frozenset({"core", "graph", "write", "nodes"}),
        ),
        WireRequestCase(
            id="graph_upsert_nodes_batch",
            component="graph",
            op="graph.upsert_nodes",
            build_method="build_graph_upsert_nodes_batch_envelope",
            schema_id=f"{base_url}/graph/{v}/graph.envelope.request.json",
            args_validator="validate_graph_upsert_nodes_args",
            description="Batch upsert graph nodes",
            tags=frozenset({"graph", "write", "nodes", "batch"}),
        ),
        # §7.3 upsert_edges
        WireRequestCase(
            id="graph_upsert_edges",
            component="graph",
            op="graph.upsert_edges",
            build_method="build_graph_upsert_edges_envelope",
            schema_id=f"{base_url}/graph/{v}/graph.envelope.request.json",
            args_validator="validate_graph_upsert_edges_args",
            description="Upsert graph edges/relationships",
            tags=frozenset({"core", "graph", "write", "edges"}),
        ),
        WireRequestCase(
            id="graph_upsert_edges_batch",
            component="graph",
            op="graph.upsert_edges",
            build_method="build_graph_upsert_edges_batch_envelope",
            schema_id=f"{base_url}/graph/{v}/graph.envelope.request.json",
            args_validator="validate_graph_upsert_edges_args",
            description="Batch upsert graph edges",
            tags=frozenset({"graph", "write", "edges", "batch"}),
        ),
        # §7.4 delete_nodes
        WireRequestCase(
            id="graph_delete_nodes",
            component="graph",
            op="graph.delete_nodes",
            build_method="build_graph_delete_nodes_envelope",
            schema_id=f"{base_url}/graph/{v}/graph.envelope.request.json",
            args_validator="validate_graph_delete_nodes_args",
            description="Delete graph nodes by ID",
            tags=frozenset({"core", "graph", "write", "delete", "nodes"}),
        ),
        WireRequestCase(
            id="graph_delete_nodes_by_label",
            component="graph",
            op="graph.delete_nodes",
            build_method="build_graph_delete_nodes_by_label_envelope",
            schema_id=f"{base_url}/graph/{v}/graph.envelope.request.json",
            args_validator="validate_graph_delete_nodes_args",
            description="Delete graph nodes by label",
            tags=frozenset({"graph", "write", "delete", "nodes", "label"}),
        ),
        # §7.5 delete_edges
        WireRequestCase(
            id="graph_delete_edges",
            component="graph",
            op="graph.delete_edges",
            build_method="build_graph_delete_edges_envelope",
            schema_id=f"{base_url}/graph/{v}/graph.envelope.request.json",
            args_validator="validate_graph_delete_edges_args",
            description="Delete graph edges by ID",
            tags=frozenset({"core", "graph", "write", "delete", "edges"}),
        ),
        WireRequestCase(
            id="graph_delete_edges_by_type",
            component="graph",
            op="graph.delete_edges",
            build_method="build_graph_delete_edges_by_type_envelope",
            schema_id=f"{base_url}/graph/{v}/graph.envelope.request.json",
            args_validator="validate_graph_delete_edges_args",
            description="Delete graph edges by type",
            tags=frozenset({"graph", "write", "delete", "edges", "type"}),
        ),
        # §7.6 query
        WireRequestCase(
            id="graph_query_gremlin",
            component="graph",
            op="graph.query",
            build_method="build_graph_query_gremlin_envelope",
            schema_id=f"{base_url}/graph/{v}/graph.envelope.request.json",
            args_validator="validate_graph_query_args",
            description="Execute Gremlin graph query",
            tags=frozenset({"core", "graph", "query", "gremlin"}),
        ),
        WireRequestCase(
            id="graph_query_cypher",
            component="graph",
            op="graph.query",
            build_method="build_graph_query_cypher_envelope",
            schema_id=f"{base_url}/graph/{v}/graph.envelope.request.json",
            args_validator="validate_graph_query_args",
            description="Execute Cypher graph query",
            tags=frozenset({"graph", "query", "cypher"}),
        ),
        WireRequestCase(
            id="graph_query_sparql",
            component="graph",
            op="graph.query",
            build_method="build_graph_query_sparql_envelope",
            schema_id=f"{base_url}/graph/{v}/graph.envelope.request.json",
            args_validator="validate_graph_query_args",
            description="Execute SPARQL RDF query",
            tags=frozenset({"graph", "query", "sparql"}),
        ),
        WireRequestCase(
            id="graph_query_with_params",
            component="graph",
            op="graph.query",
            build_method="build_graph_query_with_params_envelope",
            schema_id=f"{base_url}/graph/{v}/graph.envelope.request.json",
            args_validator="validate_graph_query_args",
            description="Graph query with parameterized bindings",
            tags=frozenset({"graph", "query", "parameters"}),
        ),
        # §7.7 stream_query
        WireRequestCase(
            id="graph_stream_query",
            component="graph",
            op="graph.stream_query",
            build_method="build_graph_stream_query_envelope",
            schema_id=f"{base_url}/graph/{v}/graph.envelope.request.json",
            args_validator="validate_graph_query_args",
            description="Stream graph query results incrementally",
            tags=frozenset({"core", "graph", "query", "streaming"}),
        ),
        WireRequestCase(
            id="graph_stream_query_gremlin",
            component="graph",
            op="graph.stream_query",
            build_method="build_graph_stream_query_gremlin_envelope",
            schema_id=f"{base_url}/graph/{v}/graph.envelope.request.json",
            args_validator="validate_graph_query_args",
            description="Stream Gremlin query results",
            tags=frozenset({"graph", "query", "streaming", "gremlin"}),
        ),
        # §7.8 bulk_vertices
        WireRequestCase(
            id="graph_bulk_vertices",
            component="graph",
            op="graph.bulk_vertices",
            build_method="build_graph_bulk_vertices_envelope",
            schema_id=f"{base_url}/graph/{v}/graph.envelope.request.json",
            args_validator="validate_graph_bulk_vertices_args",
            description="Bulk load vertices from external source",
            tags=frozenset({"graph", "write", "bulk", "nodes"}),
        ),
        # §7.9 get_schema
        WireRequestCase(
            id="graph_get_schema",
            component="graph",
            op="graph.get_schema",
            build_method="build_graph_get_schema_envelope",
            schema_id=f"{base_url}/graph/{v}/graph.envelope.request.json",
            description="Retrieve graph schema (node labels, edge types, properties)",
            tags=frozenset({"graph", "schema", "discovery"}),
        ),
        # §7.10 health
        WireRequestCase(
            id="graph_health",
            component="graph",
            op="graph.health",
            build_method="build_graph_health_envelope",
            schema_id=f"{base_url}/graph/{v}/graph.envelope.request.json",
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
        """
        Initialize registry with cases.
        
        Args:
            cases: List of cases. If None, loads from config or uses defaults.
        """
        if cases is not None:
            self._cases = list(cases)
        else:
            self._cases = self._load_cases()
        
        self._build_indices()
    
    def _load_cases(self) -> List[WireRequestCase]:
        """Load cases from external config or use defaults."""
        config = CasesConfig.from_env()
        external = load_external_config()
        
        if external and "wire_request_cases" in external:
            logger.info("Loading wire request cases from external configuration")
            return [
                WireRequestCase.from_dict(case)
                for case in external["wire_request_cases"]
            ]
        
        return _build_default_cases(config.schema_base_url)
    
    def _build_indices(self) -> None:
        """Build lookup indices for efficient access."""
        self._by_id: Dict[str, WireRequestCase] = {}
        self._by_op: Dict[str, List[WireRequestCase]] = {}
        self._by_component: Dict[str, List[WireRequestCase]] = {}
        self._by_tag: Dict[str, List[WireRequestCase]] = {}
        
        for case in self._cases:
            # Index by ID (unique)
            if case.id in self._by_id:
                raise ValueError(f"Duplicate case ID: {case.id}")
            self._by_id[case.id] = case
            
            # Index by operation
            self._by_op.setdefault(case.op, []).append(case)
            
            # Index by component
            self._by_component.setdefault(case.component, []).append(case)
            
            # Index by tags
            for tag in case.tags:
                self._by_tag.setdefault(tag, []).append(case)
    
    @property
    def cases(self) -> List[WireRequestCase]:
        """Get all cases (immutable copy)."""
        return list(self._cases)
    
    def __len__(self) -> int:
        return len(self._cases)
    
    def __iter__(self):
        return iter(self._cases)
    
    def __getitem__(self, case_id: str) -> WireRequestCase:
        """Get case by ID."""
        return self._by_id[case_id]
    
    def __contains__(self, case_id: str) -> bool:
        """Check if case ID exists."""
        return case_id in self._by_id
    
    def get(self, case_id: str) -> Optional[WireRequestCase]:
        """Get case by ID, or None if not found."""
        return self._by_id.get(case_id)
    
    def get_by_op(self, op: str) -> List[WireRequestCase]:
        """Get all cases for an operation."""
        return list(self._by_op.get(op, []))
    
    def get_by_component(self, component: str) -> List[WireRequestCase]:
        """Get all cases for a component."""
        return list(self._by_component.get(component, []))
    
    def get_by_tag(self, tag: str) -> List[WireRequestCase]:
        """Get all cases with a specific tag."""
        return list(self._by_tag.get(tag, []))
    
    def filter(
        self,
        component: Optional[str] = None,
        tag: Optional[str] = None,
        op_prefix: Optional[str] = None,
        tags_all: Optional[List[str]] = None,
        tags_any: Optional[List[str]] = None,
    ) -> List[WireRequestCase]:
        """
        Filter cases by multiple criteria.
        
        Args:
            component: Filter by component name.
            tag: Filter by single tag (shorthand for tags_any with one tag).
            op_prefix: Filter by operation prefix.
            tags_all: Case must have ALL of these tags.
            tags_any: Case must have at least ONE of these tags.
        
        Returns:
            List of matching cases.
        """
        result = []
        
        for case in self._cases:
            # Component filter
            if component and case.component != component:
                continue
            
            # Operation prefix filter
            if op_prefix and not case.op.startswith(op_prefix):
                continue
            
            # Single tag filter (legacy/shorthand)
            if tag and tag not in case.tags:
                continue
            
            # All tags filter
            if tags_all and not all(t in case.tags for t in tags_all):
                continue
            
            # Any tags filter
            if tags_any and not any(t in case.tags for t in tags_any):
                continue
            
            result.append(case)
        
        return result
    
    @property
    def components(self) -> List[str]:
        """Get list of all components with cases."""
        return sorted(self._by_component.keys())
    
    @property
    def operations(self) -> List[str]:
        """Get list of all operations with cases."""
        return sorted(self._by_op.keys())
    
    @property
    def tags(self) -> List[str]:
        """Get list of all tags used in cases."""
        return sorted(self._by_tag.keys())
    
    def get_coverage_summary(self) -> Dict[str, Any]:
        """
        Generate coverage summary for reporting.
        
        Returns:
            Dictionary with coverage statistics.
        """
        return {
            "total_cases": len(self._cases),
            "cases_by_component": {
                comp: len(cases) for comp, cases in self._by_component.items()
            },
            "cases_by_tag": {
                tag: len(cases) for tag, cases in self._by_tag.items()
            },
            "operations_covered": len(self._by_op),
            "components_covered": self.components,
            "all_tags": self.tags,
        }
    
    def to_list(self) -> List[Dict[str, Any]]:
        """Serialize all cases to list of dicts."""
        return [case.to_dict() for case in self._cases]
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize registry to JSON string."""
        return json.dumps({"wire_request_cases": self.to_list()}, indent=indent)


# ---------------------------------------------------------------------------
# Module-Level Registry Instance
# ---------------------------------------------------------------------------

# Default registry instance, loaded at import time
_default_registry: Optional[WireRequestCaseRegistry] = None


def get_registry() -> WireRequestCaseRegistry:
    """
    Get the default wire request case registry.
    
    Lazy-loads on first access to avoid import-time side effects
    in environments that don't need the full registry.
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = WireRequestCaseRegistry()
    return _default_registry


def get_cases() -> List[WireRequestCase]:
    """Get all wire request cases from the default registry."""
    return get_registry().cases


def get_case(case_id: str) -> Optional[WireRequestCase]:
    """Get a specific case by ID from the default registry."""
    return get_registry().get(case_id)


# ---------------------------------------------------------------------------
# Convenience Exports for pytest
# ---------------------------------------------------------------------------

# These are commonly used in test parameterization
WIRE_REQUEST_CASES: List[WireRequestCase] = []  # Populated lazily


def _ensure_cases_loaded() -> None:
    """Ensure WIRE_REQUEST_CASES is populated."""
    global WIRE_REQUEST_CASES
    if not WIRE_REQUEST_CASES:
        WIRE_REQUEST_CASES.extend(get_cases())


def get_pytest_params() -> List[WireRequestCase]:
    """
    Get cases formatted for pytest.mark.parametrize.
    
    Usage:
        @pytest.mark.parametrize("case", get_pytest_params(), ids=lambda c: c.id)
        def test_wire_request(case, adapter):
            ...
    """
    _ensure_cases_loaded()
    return WIRE_REQUEST_CASES


# ---------------------------------------------------------------------------
# CLI Support
# ---------------------------------------------------------------------------

def print_cases_table(
    cases: List[WireRequestCase],
    verbose: bool = False,
) -> None:
    """Print cases in a formatted table."""
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
        # Compact table format
        print(f"{'ID':<40} {'Component':<12} {'Tags'}")
        print("-" * 80)
        for case in cases:
            tags = ", ".join(sorted(case.tags)[:3])
            if len(case.tags) > 3:
                tags += f" (+{len(case.tags) - 3})"
            print(f"{case.id:<40} {case.component:<12} {tags}")


def main() -> None:
    """CLI entry point for case registry inspection."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="CORPUS Protocol Wire Request Case Registry",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all test cases",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--component", "-c",
        type=str,
        help="Filter by component",
    )
    parser.add_argument(
        "--tag", "-t",
        type=str,
        help="Filter by tag",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Print coverage summary",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Write output to file",
    )
    
    args = parser.parse_args()
    registry = get_registry()
    
    if args.coverage:
        summary = registry.get_coverage_summary()
        if args.json:
            output = json.dumps(summary, indent=2)
        else:
            output = f"""Wire Request Case Coverage Summary
===================================
Total cases:        {summary['total_cases']}
Operations covered: {summary['operations_covered']}
Components covered: {', '.join(summary['components_covered'])}

Cases by component:
{chr(10).join(f"  {k}: {v}" for k, v in summary['cases_by_component'].items())}

Cases by tag:
{chr(10).join(f"  {k}: {v}" for k, v in sorted(summary['cases_by_tag'].items()))}
"""
        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"Coverage summary written to {args.output}")
        else:
            print(output)
        return
    
    # Filter cases
    cases = registry.filter(
        component=args.component,
        tag=args.tag,
    )
    
    if args.json:
        output = json.dumps([c.to_dict() for c in cases], indent=2)
        if args.output:
            with open(args.output, "w") as f:
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
