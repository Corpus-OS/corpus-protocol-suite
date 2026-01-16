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

SCHEMA.md ALIGNMENT NOTE
------------------------
This registry uses operation-level request schemas (e.g., llm.complete.request.json),
not just envelope.request schemas, to ensure:
- op is const-bound to the correct operation name
- args are validated against the operation-specific args schema

COVERAGE NOTE (BOTH SCENARIOS)
------------------------------
Some operations may be present in SCHEMA.md but not yet implemented by a given adapter,
or not yet shipped in a given repo's schema bundle. To "cover both scenarios" without
commenting cases out, each case includes explicit gates:

  - requires_schema: if True, the harness should confirm schema_id exists in the loaded schema registry;
    if missing, the harness should skip (not fail) the case.
  - requires_builder: if True, the harness should confirm the adapter implements build_method;
    if missing, the harness should skip (not fail) the case.

These gates are pure metadata: they do not change runtime performance of this registry.
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

# Schema versioning (SCHEMA.md uses SemVer, and $id paths are not versioned by directory)
SCHEMA_VERSION = "1.0.0"
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
        schema_id: Primary JSON Schema $id URL for this operation's request envelope.
        schema_versions: Supported schema versions for backward compatibility (SemVer).
        args_validator: Name of the validator function for operation-specific args.
        description: Human-readable description for documentation/reports.
        tags: Set of tags for selective test execution and filtering.

        requires_schema: If True, harness should skip if schema_id not present in registry.
        requires_builder: If True, harness should skip if adapter lacks build_method.
    """

    id: str
    component: str
    op: str
    build_method: str
    schema_id: str
    schema_versions: tuple[str, ...] = (SCHEMA_VERSION,)
    args_validator: Optional[str] = None
    description: str = ""
    tags: FrozenSet[str] = field(default_factory=frozenset)

    # Coverage gates (metadata only; harness uses these to decide skip vs fail)
    requires_schema: bool = True
    requires_builder: bool = True

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
            schema_versions=tuple(data.get("schema_versions", (SCHEMA_VERSION,))),
            args_validator=data.get("args_validator"),
            description=data.get("description", ""),
            tags=frozenset(data.get("tags", [])),
            requires_schema=bool(data.get("requires_schema", True)),
            requires_builder=bool(data.get("requires_builder", True)),
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
            "requires_schema": self.requires_schema,
            "requires_builder": self.requires_builder,
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

    This set is aligned to SCHEMA.md operation naming and operation-level request schemas:
      <base_url>/<component>/<component>.<operation>.request.json

    Variants reuse the same build method and operation schema; the harness supplies variant args.

    NOTE: For operations that may not exist in all adapters or schema bundles, we still include
    cases but rely on requires_schema/requires_builder gates for skip behavior in the harness.
    """
    return [
        # ========================== LLM ========================== #
        WireRequestCase(
            id="llm_capabilities",
            component="llm",
            op="llm.capabilities",
            build_method="build_llm_capabilities_envelope",
            schema_id=f"{base_url}/llm/llm.capabilities.request.json",
            description="Discover supported LLM features and models",
            tags=frozenset({"llm", "discovery", "capabilities"}),
        ),
        WireRequestCase(
            id="llm_complete",
            component="llm",
            op="llm.complete",
            build_method="build_llm_complete_envelope",
            schema_id=f"{base_url}/llm/llm.complete.request.json",
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
            schema_id=f"{base_url}/llm/llm.complete.request.json",
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
            schema_id=f"{base_url}/llm/llm.complete.request.json",
            args_validator="validate_llm_complete_args",
            description="LLM completion with JSON output mode",
            tags=frozenset({"llm", "completion", "json_mode"}),
        ),
        WireRequestCase(
            id="llm_stream",
            component="llm",
            op="llm.stream",
            build_method="build_llm_stream_envelope",
            schema_id=f"{base_url}/llm/llm.stream.request.json",
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
            schema_id=f"{base_url}/llm/llm.stream.request.json",
            args_validator="validate_llm_stream_args",
            description="Stream LLM completion with tool calling",
            tags=frozenset({"llm", "streaming", "tools"}),
        ),
        WireRequestCase(
            id="llm_count_tokens",
            component="llm",
            op="llm.count_tokens",
            build_method="build_llm_count_tokens_envelope",
            schema_id=f"{base_url}/llm/llm.count_tokens.request.json",
            args_validator="validate_llm_count_tokens_args",
            description="Count tokens in text for a specific model",
            tags=frozenset({"llm", "tokens"}),
        ),
        WireRequestCase(
            id="llm_health",
            component="llm",
            op="llm.health",
            build_method="build_llm_health_envelope",
            schema_id=f"{base_url}/llm/llm.health.request.json",
            description="Check LLM provider health and model availability",
            tags=frozenset({"llm", "health", "operational"}),
        ),

        # ======================== VECTOR ========================= #
        WireRequestCase(
            id="vector_capabilities",
            component="vector",
            op="vector.capabilities",
            build_method="build_vector_capabilities_envelope",
            schema_id=f"{base_url}/vector/vector.capabilities.request.json",
            description="Discover vector database capabilities and limits",
            tags=frozenset({"vector", "discovery", "capabilities"}),
        ),
        WireRequestCase(
            id="vector_query",
            component="vector",
            op="vector.query",
            build_method="build_vector_query_envelope",
            schema_id=f"{base_url}/vector/vector.query.request.json",
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
            schema_id=f"{base_url}/vector/vector.query.request.json",
            args_validator="validate_vector_query_args",
            description="Vector query with metadata filtering",
            tags=frozenset({"core", "vector", "query", "filter"}),
        ),
        WireRequestCase(
            id="vector_batch_query",
            component="vector",
            op="vector.batch_query",
            build_method="build_vector_batch_query_envelope",
            schema_id=f"{base_url}/vector/vector.batch_query.request.json",
            args_validator="validate_vector_batch_query_args",
            description="Batch vector similarity search queries",
            tags=frozenset({"vector", "query", "batch"}),
            # Cover both scenarios: some adapters may not implement yet, some schema bundles may not include yet.
            requires_schema=True,
            requires_builder=True,
        ),
        WireRequestCase(
            id="vector_upsert",
            component="vector",
            op="vector.upsert",
            build_method="build_vector_upsert_envelope",
            schema_id=f"{base_url}/vector/vector.upsert.request.json",
            args_validator="validate_vector_upsert_args",
            description="Upsert vectors (insert or update)",
            tags=frozenset({"core", "vector", "write", "upsert"}),
        ),
        WireRequestCase(
            id="vector_upsert_batch",
            component="vector",
            op="vector.upsert",
            build_method="build_vector_upsert_envelope",
            schema_id=f"{base_url}/vector/vector.upsert.request.json",
            args_validator="validate_vector_upsert_args",
            description="Batch vector upsert",
            tags=frozenset({"vector", "write", "upsert", "batch"}),
        ),
        WireRequestCase(
            id="vector_upsert_with_metadata",
            component="vector",
            op="vector.upsert",
            build_method="build_vector_upsert_envelope",
            schema_id=f"{base_url}/vector/vector.upsert.request.json",
            args_validator="validate_vector_upsert_args",
            description="Vector upsert with metadata",
            tags=frozenset({"vector", "write", "upsert", "metadata"}),
        ),
        WireRequestCase(
            id="vector_delete",
            component="vector",
            op="vector.delete",
            build_method="build_vector_delete_envelope",
            schema_id=f"{base_url}/vector/vector.delete.request.json",
            args_validator="validate_vector_delete_args",
            description="Delete vectors by ID",
            tags=frozenset({"core", "vector", "write", "delete"}),
        ),
        WireRequestCase(
            id="vector_delete_by_filter",
            component="vector",
            op="vector.delete",
            build_method="build_vector_delete_envelope",
            schema_id=f"{base_url}/vector/vector.delete.request.json",
            args_validator="validate_vector_delete_args",
            description="Delete vectors by metadata filter",
            tags=frozenset({"vector", "write", "delete", "filter"}),
        ),
        WireRequestCase(
            id="vector_create_namespace",
            component="vector",
            op="vector.create_namespace",
            build_method="build_vector_create_namespace_envelope",
            schema_id=f"{base_url}/vector/vector.create_namespace.request.json",
            args_validator="validate_vector_namespace_args",
            description="Create a new vector namespace",
            tags=frozenset({"vector", "namespace", "write"}),
        ),
        WireRequestCase(
            id="vector_delete_namespace",
            component="vector",
            op="vector.delete_namespace",
            build_method="build_vector_delete_namespace_envelope",
            schema_id=f"{base_url}/vector/vector.delete_namespace.request.json",
            args_validator="validate_vector_namespace_args",
            description="Delete a vector namespace",
            tags=frozenset({"vector", "namespace", "write", "delete"}),
        ),
        WireRequestCase(
            id="vector_health",
            component="vector",
            op="vector.health",
            build_method="build_vector_health_envelope",
            schema_id=f"{base_url}/vector/vector.health.request.json",
            description="Check vector database health",
            tags=frozenset({"vector", "health", "operational"}),
        ),

        # ======================= EMBEDDING ======================= #
        WireRequestCase(
            id="embedding_capabilities",
            component="embedding",
            op="embedding.capabilities",
            build_method="build_embedding_capabilities_envelope",
            schema_id=f"{base_url}/embedding/embedding.capabilities.request.json",
            description="Discover embedding model capabilities",
            tags=frozenset({"embedding", "discovery", "capabilities"}),
        ),
        WireRequestCase(
            id="embedding_embed",
            component="embedding",
            op="embedding.embed",
            build_method="build_embedding_embed_envelope",
            schema_id=f"{base_url}/embedding/embedding.embed.request.json",
            args_validator="validate_embedding_embed_args",
            description="Generate embedding for single text",
            tags=frozenset({"core", "embedding", "embed"}),
        ),
        WireRequestCase(
            id="embedding_embed_with_model",
            component="embedding",
            op="embedding.embed",
            build_method="build_embedding_embed_envelope",
            schema_id=f"{base_url}/embedding/embedding.embed.request.json",
            args_validator="validate_embedding_embed_args",
            description="Generate embedding with explicit model selection",
            tags=frozenset({"embedding", "embed", "model"}),
        ),
        WireRequestCase(
            id="embedding_embed_truncate",
            component="embedding",
            op="embedding.embed",
            build_method="build_embedding_embed_envelope",
            schema_id=f"{base_url}/embedding/embedding.embed.request.json",
            args_validator="validate_embedding_embed_args",
            description="Generate embedding with truncation enabled",
            tags=frozenset({"embedding", "embed", "truncate"}),
        ),
        WireRequestCase(
            id="embedding_embed_normalized",
            component="embedding",
            op="embedding.embed",
            build_method="build_embedding_embed_envelope",
            schema_id=f"{base_url}/embedding/embedding.embed.request.json",
            args_validator="validate_embedding_embed_args",
            description="Generate normalized embedding",
            tags=frozenset({"embedding", "embed", "normalize"}),
        ),
        WireRequestCase(
            id="embedding_embed_batch",
            component="embedding",
            op="embedding.embed_batch",
            build_method="build_embedding_embed_batch_envelope",
            schema_id=f"{base_url}/embedding/embedding.embed_batch.request.json",
            args_validator="validate_embedding_embed_batch_args",
            description="Generate embeddings for multiple texts",
            tags=frozenset({"core", "embedding", "batch"}),
        ),
        WireRequestCase(
            id="embedding_embed_batch_large",
            component="embedding",
            op="embedding.embed_batch",
            build_method="build_embedding_embed_batch_envelope",
            schema_id=f"{base_url}/embedding/embedding.embed_batch.request.json",
            args_validator="validate_embedding_embed_batch_args",
            description="Large batch embedding request",
            tags=frozenset({"embedding", "batch", "large"}),
        ),
        WireRequestCase(
            id="embedding_stream_embed",
            component="embedding",
            op="embedding.stream_embed",
            build_method="build_embedding_stream_embed_envelope",
            schema_id=f"{base_url}/embedding/embedding.stream_embed.request.json",
            args_validator="validate_embedding_stream_embed_args",
            description="Stream embedding generation for a single text",
            tags=frozenset({"embedding", "streaming", "embed"}),
            requires_schema=True,
            requires_builder=True,
        ),
        WireRequestCase(
            id="embedding_count_tokens",
            component="embedding",
            op="embedding.count_tokens",
            build_method="build_embedding_count_tokens_envelope",
            schema_id=f"{base_url}/embedding/embedding.count_tokens.request.json",
            args_validator="validate_embedding_count_tokens_args",
            description="Count tokens for embedding model",
            tags=frozenset({"embedding", "tokens"}),
        ),
        WireRequestCase(
            id="embedding_get_stats",
            component="embedding",
            op="embedding.get_stats",
            build_method="build_embedding_get_stats_envelope",
            schema_id=f"{base_url}/embedding/embedding.get_stats.request.json",
            args_validator="validate_embedding_get_stats_args",
            description="Retrieve embedding service statistics (counters, cache, timings)",
            tags=frozenset({"embedding", "stats", "operational"}),
            requires_schema=True,
            requires_builder=True,
        ),
        WireRequestCase(
            id="embedding_health",
            component="embedding",
            op="embedding.health",
            build_method="build_embedding_health_envelope",
            schema_id=f"{base_url}/embedding/embedding.health.request.json",
            description="Check embedding service health",
            tags=frozenset({"embedding", "health", "operational"}),
        ),

        # ========================= GRAPH ========================= #
        WireRequestCase(
            id="graph_capabilities",
            component="graph",
            op="graph.capabilities",
            build_method="build_graph_capabilities_envelope",
            schema_id=f"{base_url}/graph/graph.capabilities.request.json",
            description="Discover graph database capabilities",
            tags=frozenset({"graph", "discovery", "capabilities"}),
        ),
        WireRequestCase(
            id="graph_upsert_nodes",
            component="graph",
            op="graph.upsert_nodes",
            build_method="build_graph_upsert_nodes_envelope",
            schema_id=f"{base_url}/graph/graph.upsert_nodes.request.json",
            args_validator="validate_graph_upsert_nodes_args",
            description="Upsert graph nodes",
            tags=frozenset({"core", "graph", "write", "nodes"}),
        ),
        WireRequestCase(
            id="graph_upsert_nodes_batch",
            component="graph",
            op="graph.upsert_nodes",
            build_method="build_graph_upsert_nodes_envelope",
            schema_id=f"{base_url}/graph/graph.upsert_nodes.request.json",
            args_validator="validate_graph_upsert_nodes_args",
            description="Batch upsert graph nodes",
            tags=frozenset({"graph", "write", "nodes", "batch"}),
        ),
        WireRequestCase(
            id="graph_upsert_edges",
            component="graph",
            op="graph.upsert_edges",
            build_method="build_graph_upsert_edges_envelope",
            schema_id=f"{base_url}/graph/graph.upsert_edges.request.json",
            args_validator=None,  # Using schema validation only (args shape enforced by schema_id above)
            description="Create/upsert graph edges/relationships",
            tags=frozenset({"core", "graph", "write", "edges"}),
        ),
        WireRequestCase(
            id="graph_upsert_edges_batch",
            component="graph",
            op="graph.upsert_edges",
            build_method="build_graph_upsert_edges_envelope",
            schema_id=f"{base_url}/graph/graph.upsert_edges.request.json",
            args_validator=None,  # Using schema validation only (args shape enforced by schema_id above)
            description="Batch create/upsert graph edges",
            tags=frozenset({"graph", "write", "edges", "batch"}),
        ),
        WireRequestCase(
            id="graph_delete_nodes",
            component="graph",
            op="graph.delete_nodes",
            build_method="build_graph_delete_nodes_envelope",
            schema_id=f"{base_url}/graph/graph.delete_nodes.request.json",
            args_validator="validate_graph_delete_nodes_args",
            description="Delete graph nodes by ID",
            tags=frozenset({"core", "graph", "write", "delete", "nodes"}),
        ),
        WireRequestCase(
            id="graph_delete_nodes_by_label",
            component="graph",
            op="graph.delete_nodes",
            build_method="build_graph_delete_nodes_envelope",
            schema_id=f"{base_url}/graph/graph.delete_nodes.request.json",
            args_validator="validate_graph_delete_nodes_args",
            description="Delete graph nodes by label (variant args)",
            tags=frozenset({"graph", "write", "delete", "nodes", "label"}),
        ),
        WireRequestCase(
            id="graph_delete_edges",
            component="graph",
            op="graph.delete_edges",
            build_method="build_graph_delete_edges_envelope",
            schema_id=f"{base_url}/graph/graph.delete_edges.request.json",
            args_validator=None,  # Using schema validation only (args shape enforced by schema_id above)
            description="Delete graph edges by ID",
            tags=frozenset({"core", "graph", "write", "delete", "edges"}),
        ),
        WireRequestCase(
            id="graph_delete_edges_by_type",
            component="graph",
            op="graph.delete_edges",
            build_method="build_graph_delete_edges_envelope",
            schema_id=f"{base_url}/graph/graph.delete_edges.request.json",
            args_validator=None,  # Using schema validation only (args shape enforced by schema_id above)
            description="Delete graph edges by type (variant args)",
            tags=frozenset({"graph", "write", "delete", "edges", "type"}),
        ),
        WireRequestCase(
            id="graph_query_gremlin",
            component="graph",
            op="graph.query",
            build_method="build_graph_query_envelope",
            schema_id=f"{base_url}/graph/graph.query.request.json",
            args_validator=None,  # Using schema validation only (dialect/text enforced by schema_id above)
            description="Execute Gremlin graph query (variant args)",
            tags=frozenset({"core", "graph", "query", "gremlin"}),
        ),
        WireRequestCase(
            id="graph_query_cypher",
            component="graph",
            op="graph.query",
            build_method="build_graph_query_envelope",
            schema_id=f"{base_url}/graph/graph.query.request.json",
            args_validator=None,  # Using schema validation only (dialect/text enforced by schema_id above)
            description="Execute Cypher graph query (variant args)",
            tags=frozenset({"graph", "query", "cypher"}),
        ),
        WireRequestCase(
            id="graph_query_sparql",
            component="graph",
            op="graph.query",
            build_method="build_graph_query_envelope",
            schema_id=f"{base_url}/graph/graph.query.request.json",
            args_validator=None,  # Using schema validation only (dialect/text enforced by schema_id above)
            description="Execute SPARQL query (variant args; may be NOT_SUPPORTED by provider)",
            tags=frozenset({"graph", "query", "sparql"}),
        ),
        WireRequestCase(
            id="graph_query_with_params",
            component="graph",
            op="graph.query",
            build_method="build_graph_query_envelope",
            schema_id=f"{base_url}/graph/graph.query.request.json",
            args_validator=None,  # Using schema validation only (params enforced by schema_id above)
            description="Graph query with parameterized bindings",
            tags=frozenset({"graph", "query", "parameters"}),
        ),
        WireRequestCase(
            id="graph_stream_query",
            component="graph",
            op="graph.stream_query",
            build_method="build_graph_stream_query_envelope",
            schema_id=f"{base_url}/graph/graph.stream_query.request.json",
            args_validator=None,  # Using schema validation only (args enforced by schema_id above)
            description="Stream graph query results incrementally",
            tags=frozenset({"core", "graph", "query", "streaming"}),
        ),
        WireRequestCase(
            id="graph_stream_query_gremlin",
            component="graph",
            op="graph.stream_query",
            build_method="build_graph_stream_query_envelope",
            schema_id=f"{base_url}/graph/graph.stream_query.request.json",
            args_validator=None,  # Using schema validation only (args enforced by schema_id above)
            description="Stream Gremlin query results (variant args)",
            tags=frozenset({"graph", "query", "streaming", "gremlin"}),
        ),
        WireRequestCase(
            id="graph_bulk_vertices",
            component="graph",
            op="graph.bulk_vertices",
            build_method="build_graph_bulk_vertices_envelope",
            schema_id=f"{base_url}/graph/graph.bulk_vertices.request.json",
            args_validator="validate_graph_bulk_vertices_args",
            description="Bulk vertices export/import via cursor pagination",
            tags=frozenset({"graph", "bulk", "nodes", "read"}),
            requires_schema=True,
            requires_builder=True,
        ),
        WireRequestCase(
            id="graph_batch",
            component="graph",
            op="graph.batch",
            build_method="build_graph_batch_envelope",
            schema_id=f"{base_url}/graph/graph.batch.request.json",
            args_validator=None,
            description="Execute multiple graph operations in a batch",
            tags=frozenset({"graph", "batch", "write"}),
        ),
        WireRequestCase(
            id="graph_get_schema",
            component="graph",
            op="graph.get_schema",
            build_method="build_graph_get_schema_envelope",
            schema_id=f"{base_url}/graph/graph.get_schema.request.json",
            args_validator=None,
            description="Retrieve graph schema (node labels, edge types, properties)",
            tags=frozenset({"graph", "schema", "discovery"}),
            requires_schema=True,
            requires_builder=True,
        ),
        WireRequestCase(
            id="graph_transaction",
            component="graph",
            op="graph.transaction",
            build_method="build_graph_transaction_envelope",
            schema_id=f"{base_url}/graph/graph.transaction.request.json",
            args_validator=None,
            description="Execute graph operations as a single transaction",
            tags=frozenset({"graph", "transaction", "write"}),
            requires_schema=True,
            requires_builder=True,
        ),
        WireRequestCase(
            id="graph_traversal",
            component="graph",
            op="graph.traversal",
            build_method="build_graph_traversal_envelope",
            schema_id=f"{base_url}/graph/graph.traversal.request.json",
            args_validator=None,
            description="Traverse graph from start nodes with filters and depth",
            tags=frozenset({"graph", "traversal", "query"}),
            requires_schema=True,
            requires_builder=True,
        ),
        WireRequestCase(
            id="graph_health",
            component="graph",
            op="graph.health",
            build_method="build_graph_health_envelope",
            schema_id=f"{base_url}/graph/graph.health.request.json",
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
            print(f"  Gates:        requires_schema={case.requires_schema}, requires_builder={case.requires_builder}")
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
