# corpus_sdk/vector/framework_adapters/llamaindex.py
# SPDX-License-Identifier: Apache-2.0

"""
LlamaIndex adapter for Corpus Vector protocol.

This module exposes Corpus `BaseVectorAdapter` implementations as
`llama_index.core.vector_stores.types.BasePydanticVectorStore` instances, with:

- Sync + async add/query/delete APIs (matching and extending LlamaIndex expectations)
- Proper integration with Corpus VectorProtocolV1 via VectorTranslator
- Namespace + metadata filter handling (capability-aware)
- Batch upserts and deletes that respect backend limits (enforced in translator)
- Optional client-side score thresholding
- Full OperationContext propagation via `corpus_sdk.core.context_translation.from_llamaindex`
- Rich error context via `corpus_sdk.core.error_context.attach_context`
- Optional streaming query support via VectorTranslator.query_stream
- Optional Maximal Marginal Relevance (MMR) query variants
- Comprehensive configuration validation with Pydantic
- Graceful error recovery for partial batch failures

Design philosophy
-----------------
- Protocol-first: LlamaIndex is a thin skin over Corpus vector adapters.
- All heavy lifting (backpressure, deadlines, breakers, etc.) lives in
  the underlying `BaseVectorAdapter` / `VectorTranslator`, not here.
- This layer focuses on:
    * Translating LlamaIndex Nodes ↔ Corpus Vector objects
    * Respecting VectorCapabilities (namespaces, filters, batch sizes)
    * Using VectorTranslator for all sync/async/stream orchestration
    * Exposing advanced retrieval patterns (streaming, MMR) without leaking
      protocol details.

Typical usage
-------------

    from llama_index.core import VectorStoreIndex
    from corpus_sdk.vector.pinecone_adapter import PineconeVectorAdapter
    from corpus_sdk.vector.framework_adapters.llamaindex import (
        CorpusLlamaIndexVectorStore,
    )

    corpus_adapter = PineconeVectorAdapter(
        index_name="my-index",
        api_key="...",
        dimensions=1536,
    )

    vector_store = CorpusLlamaIndexVectorStore(
        corpus_adapter=corpus_adapter,
        namespace="docs",
        batch_size=100,
        score_threshold=0.1,
    )

    index = VectorStoreIndex.from_vector_store(vector_store)

    # Insert documents via LlamaIndex (which will handle embeddings)
    index.insert("hello world")

    # Standard query (LlamaIndex will provide query_embedding)
    query_engine = index.as_query_engine()
    response = query_engine.query("hello")

    # Direct vector-store query
    from llama_index.core.vector_stores.types import VectorStoreQuery

    query = VectorStoreQuery(query_embedding=[...], similarity_top_k=4)
    result = vector_store.query(query)

    # Streaming query: yield NodeWithScore one by one
    for node_with_score in vector_store.query_stream(query):
        handle(node_with_score)

    # MMR query (sync extension)
    mmr_result = vector_store.query_mmr(query, lambda_mult=0.5, fetch_k=16)

    # Async power-users can call:
    mmr_result_async = await vector_store.aquery_mmr(query, lambda_mult=0.5, fetch_k=16)

"""

from __future__ import annotations

import logging
import math
from functools import cached_property
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from pydantic import Field, field_validator, model_validator

from llama_index.core.schema import (
    BaseNode,
    MetadataMode,
    NodeWithScore,
    TextNode,
)
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
    VectorStoreInfo,
    MetadataInfo,
    MetadataFilters,
)
from llama_index.core.vector_stores.utils import (
    node_to_metadata_dict,
    metadata_dict_to_node,
    legacy_metadata_dict_to_node,
)

from corpus_sdk.vector.vector_base import (
    BaseVectorAdapter,
    Vector,
    VectorMatch,
    QueryResult,
    DeleteResult,
    UpsertResult,
    OperationContext,
    VectorCapabilities,
    # Errors
    BadRequest,
    NotSupported,
    VectorAdapterError,
)
from corpus_sdk.vector.framework_adapters.common.vector_translation import (
    DefaultVectorFrameworkTranslator,
    VectorTranslator,
)

from corpus_sdk.core.context_translation import from_llamaindex as context_from_llamaindex
from corpus_sdk.core.error_context import attach_context

logger = logging.getLogger(__name__)

Metadata = Dict[str, Any]


class CorpusLlamaIndexVectorStore(BasePydanticVectorStore):
    """
    LlamaIndex `BasePydanticVectorStore` implementation backed by a Corpus
    `BaseVectorAdapter` (VectorProtocolV1).

    This class is a thin integration layer:
    - Nodes are mapped to Corpus VectorProtocol `Vector` objects.
    - VectorStoreQuery calls map to VectorTranslator `query()` calls.
    - Namespaces + metadata filters are honored based on VectorCapabilities.
    - Sync APIs use VectorTranslator's sync methods; async APIs use its async methods.
    - Streaming uses VectorTranslator.query_stream.
    - Optional MMR variants leverage database scores for relevance + cosine diversity.

    Key LlamaIndex-specific behaviors:
    - Nodes must have embeddings pre-computed before calling `add()`/`aadd()`
    - Metadata is flattened using LlamaIndex's `node_to_metadata_dict`
    - `ref_doc_id` and `node_id` are preserved in metadata for proper document lifecycle
    - Streaming queries yield `NodeWithScore` objects one by one for responsive UIs
    - MMR queries respect the original database similarity scores while adding diversity

    Configuration validation ensures parameters are within reasonable bounds while
    allowing backend capabilities to further constrain actual operations.
    """

    # LlamaIndex VectorStore flags
    stores_text: bool = True
    flat_metadata: bool = True

    # Corpus integration fields with validation
    corpus_adapter: BaseVectorAdapter = Field(
        ...,
        description="Underlying Corpus vector adapter implementing VectorProtocolV1"
    )
    
    namespace: Optional[str] = Field(
        "default",
        description="Default namespace for all operations. Ignored if backend doesn't support namespaces."
    )
    
    batch_size: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Planning hint for upsert/delete batches. Actual batch size is min(batch_size, backend_max_batch_size)"
    )
    
    default_top_k: int = Field(
        default=4,
        ge=1,
        le=1000,
        description="Default number of results returned when query doesn't specify similarity_top_k"
    )
    
    score_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional minimum similarity score (0.0-1.0) for client-side filtering of results"
    )

    # Reserved metadata keys for internal mapping
    id_field: str = Field(
        default="id",
        description="Metadata key for storing the vector ID (maps to node_id)"
    )
    text_field: str = Field(
        default="text", 
        description="Metadata key for storing node text content"
    )
    node_id_field: str = Field(
        default="node_id",
        description="Metadata key for storing LlamaIndex node ID"
    )
    ref_doc_id_field: str = Field(
        default="ref_doc_id",
        description="Metadata key for storing reference document ID"
    )

    # Cached capabilities (lazy-loaded)
    _caps: Optional[VectorCapabilities] = None

    # Pydantic v2-style config
    model_config = {"arbitrary_types_allowed": True}

    # ------------------------------------------------------------------ #
    # Configuration validation
    # ------------------------------------------------------------------ #

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """Ensure batch_size is reasonable for vector operations."""
        if v < 1:
            raise ValueError("batch_size must be at least 1")
        if v > 10000:
            logger.warning(
                "batch_size %d is unusually large; consider reducing for better performance",
                v
            )
        return v

    @field_validator("default_top_k")
    @classmethod
    def validate_default_top_k(cls, v: int) -> int:
        """Ensure default_top_k is reasonable for similarity search."""
        if v < 1:
            raise ValueError("default_top_k must be at least 1")
        if v > 1000:
            logger.warning(
                "default_top_k %d is unusually large; most applications use 4-100",
                v
            )
        return v

    @field_validator("score_threshold")
    @classmethod
    def validate_score_threshold(cls, v: Optional[float]) -> Optional[float]:
        """Ensure score_threshold is a valid similarity score."""
        if v is not None:
            if v < 0.0 or v > 1.0:
                raise ValueError("score_threshold must be between 0.0 and 1.0")
            if v > 0.9:
                logger.warning(
                    "score_threshold %.2f is very high; may filter out relevant results",
                    v
                )
        return v

    @model_validator(mode="after")
    def validate_reserved_fields(self) -> CorpusLlamaIndexVectorStore:
        """Ensure reserved metadata field names don't conflict."""
        reserved_fields = {
            self.id_field, 
            self.text_field, 
            self.node_id_field, 
            self.ref_doc_id_field
        }
        if len(reserved_fields) != 4:
            raise ValueError(
                f"Reserved metadata fields must be unique: {reserved_fields}"
            )
        return self

    # ------------------------------------------------------------------ #
    # Translator wiring
    # ------------------------------------------------------------------ #

    @cached_property
    def _translator(self) -> VectorTranslator:
        """
        Lazily construct and cache the VectorTranslator.

        The translator owns:
        - Sync/async orchestration
        - Batch splitting according to backend capabilities
        - Raw→spec translation (dicts → QuerySpec/UpsertSpec/DeleteSpec)
        - Error recovery and partial failure handling for batch operations
        """
        framework_translator = DefaultVectorFrameworkTranslator()
        return VectorTranslator(
            adapter=self.corpus_adapter,
            framework="llamaindex",
            translator=framework_translator,
        )

    # ------------------------------------------------------------------ #
    # VectorStore metadata / identification
    # ------------------------------------------------------------------ #

    @classmethod
    def class_name(cls) -> str:
        """LlamaIndex class identifier."""
        return "CorpusLlamaIndexVectorStore"

    @property
    def client(self) -> BaseVectorAdapter:
        """Expose the underlying Corpus adapter as the 'client'."""
        return self.corpus_adapter

    @property
    def vector_store_info(self) -> VectorStoreInfo:
        """
        Basic vector store metadata used by some LlamaIndex tools / UIs.

        This is intentionally minimal and protocol-agnostic. LlamaIndex uses
        this information for query planning and tool descriptions in agent workflows.
        """
        return VectorStoreInfo(
            name="corpus",
            description="Corpus VectorProtocol-backed vector store for LlamaIndex.",
            metadata_info=[
                MetadataInfo(
                    name=self.node_id_field,
                    description="LlamaIndex node ID.",
                    type="str",
                ),
                MetadataInfo(
                    name=self.ref_doc_id_field,
                    description="Reference document ID associated with the node.",
                    type="str",
                ),
            ],
        )

    # ------------------------------------------------------------------ #
    # Capabilities / context helpers
    # ------------------------------------------------------------------ #

    def _get_caps_sync(self) -> VectorCapabilities:
        """
        Synchronously fetch and cache VectorCapabilities via VectorTranslator.
        
        This method is called automatically during operations that need capability
        checks. The result is cached for the lifetime of the vector store instance.
        """
        if self._caps is not None:
            return self._caps
        try:
            caps = self._translator.capabilities()
            self._caps = caps
            return caps
        except Exception as exc:  # noqa: BLE001
            attach_context(exc, framework="llamaindex", operation="capabilities_sync")
            raise

    async def _get_caps_async(self) -> VectorCapabilities:
        """
        Async capability fetch with caching via VectorTranslator.
        
        Used by async operations to ensure capability checks don't block the event loop.
        The result is shared with sync operations via the cached _caps attribute.
        """
        if self._caps is not None:
            return self._caps
        try:
            caps = await self._translator.arun_capabilities()
            self._caps = caps
            return caps
        except Exception as exc:  # noqa: BLE001
            attach_context(exc, framework="llamaindex", operation="capabilities_async")
            raise

    def _build_ctx(self, **kwargs: Any) -> Optional[OperationContext]:
        """
        Build an OperationContext from LlamaIndex-style kwargs.

        Priority:
        1. Explicit OperationContext via `ctx` or `operation_context`.
        2. LlamaIndex CallbackManager via `callback_manager` using
           `context_from_llamaindex`.
        3. None (no context).

        LlamaIndex typically provides callback managers for tracing and monitoring,
        which we translate into Corpus OperationContext for distributed tracing.
        """
        ctx = kwargs.get("ctx") or kwargs.get("operation_context")
        if isinstance(ctx, OperationContext):
            return ctx

        callback_manager = kwargs.get("callback_manager")
        if callback_manager is None:
            return None

        try:
            return context_from_llamaindex(callback_manager)
        except Exception as exc:  # noqa: BLE001
            logger.debug("context_from_llamaindex failed: %s", exc)
            return None

    def _effective_namespace(self, namespace: Optional[str]) -> Optional[str]:
        """
        Resolve namespace using explicit override or store default.

        If the underlying adapter does not support namespaces, this value is
        still passed down, but the adapter may ignore it. This allows the same
        code to work with both namespace-aware and namespace-agnostic backends.
        """
        return namespace if namespace is not None else self.namespace

    # ------------------------------------------------------------------ #
    # Translation helpers: LlamaIndex Nodes ↔ Corpus Vector
    # ------------------------------------------------------------------ #

    def _nodes_to_corpus_vectors(
        self,
        nodes: Sequence[BaseNode],
        namespace: Optional[str],
    ) -> List[Vector]:
        """
        Convert LlamaIndex nodes (with embeddings) into Corpus `Vector` objects.

        Assumes:
        - Each node has a non-empty embedding accessible via `get_embedding()`
          or `.embedding` (LlamaIndex typically handles embedding computation)
        - Metadata is flattened via `node_to_metadata_dict` for backend storage
        - Node IDs are preserved and used as vector IDs for consistency

        Raises BadRequest if any node lacks an embedding, as this indicates
        a configuration issue in the LlamaIndex pipeline.
        """
        vectors: List[Vector] = []
        ns = self._effective_namespace(namespace)

        for node in nodes:
            # Extract embedding - LlamaIndex should have computed this already
            embedding = node.get_embedding() if hasattr(node, "get_embedding") else None
            if embedding is None:
                embedding = getattr(node, "embedding", None)
            if embedding is None:
                raise BadRequest(
                    f"Node {getattr(node, 'node_id', None) or node} has no embedding; "
                    "ensure embeddings are set before calling add(). LlamaIndex typically "
                    "handles embedding computation via ServiceContext or embedding models.",
                    code="NO_EMBEDDING",
                )

            # Flatten metadata using LlamaIndex's standard approach
            metadata = node_to_metadata_dict(
                node,
                remove_text=False,
                mode=MetadataMode.ALL,
            )

            # Ensure reserved keys are populated for proper node reconstruction
            node_id = getattr(node, "node_id", None) or getattr(node, "id_", None)
            ref_doc_id = getattr(node, "ref_doc_id", None)

            metadata = dict(metadata or {})
            metadata[self.node_id_field] = node_id
            if ref_doc_id is not None:
                metadata[self.ref_doc_id_field] = ref_doc_id

            # Text is stored in metadata; docstore is handled by LlamaIndex upstream
            text = node.get_content(metadata_mode=MetadataMode.NONE) or ""

            metadata[self.text_field] = text
            metadata[self.id_field] = node_id

            vectors.append(
                Vector(
                    id=str(node_id),
                    vector=[float(x) for x in embedding],
                    metadata=metadata,
                    namespace=ns,
                    text=None,  # Text stored in metadata, not separate field
                )
            )

        return vectors

    def _matches_to_nodes(
        self,
        matches: Sequence[VectorMatch],
    ) -> List[NodeWithScore]:
        """
        Convert Corpus `VectorMatch` objects into LlamaIndex NodeWithScore.

        The similarity score returned is `VectorMatch.score`, which is assumed
        to be higher-is-better (normalized similarity as defined by the adapter).

        Node reconstruction uses LlamaIndex's standard utilities with fallbacks:
        1. Try `metadata_dict_to_node` (modern approach)
        2. Try `legacy_metadata_dict_to_node` (backward compatibility)
        3. Fall back to basic TextNode with preserved metadata

        This ensures maximum compatibility with different LlamaIndex versions and
        node types while preserving all original node attributes when possible.
        """
        results: List[NodeWithScore] = []

        for m in matches:
            v = m.vector
            meta = dict(v.metadata or {})

            # Extract core fields from metadata
            text = meta.pop(self.text_field, None)
            node_id = meta.pop(self.node_id_field, v.id)  # Fallback to vector ID
            ref_doc_id = meta.get(self.ref_doc_id_field)

            # Reconstruct node from metadata using LlamaIndex's utilities
            node: BaseNode
            try:
                # First try modern node reconstruction
                node = metadata_dict_to_node(meta, text=text, node_id=node_id)
            except Exception as exc:
                logger.debug(
                    "metadata_dict_to_node failed for node %s: %s; trying legacy method",
                    node_id, exc
                )
                try:
                    # Fall back to legacy reconstruction for older LlamaIndex versions
                    node = legacy_metadata_dict_to_node(meta, text=text, node_id=node_id)
                except Exception as exc2:
                    logger.debug(
                        "legacy_metadata_dict_to_node also failed for node %s: %s; "
                        "falling back to TextNode", node_id, exc2
                    )
                    # Final fallback: basic TextNode with preserved metadata
                    node = TextNode(
                        text=text or "",
                        id_=str(node_id),
                        metadata=meta,
                    )

            # Restore ref_doc_id if present and supported by node type
            if ref_doc_id is not None:
                try:
                    node.ref_doc_id = ref_doc_id  # type: ignore[attr-defined]
                except Exception:
                    # If the node type does not support ref_doc_id, ignore silently
                    # as this is not a critical failure for most operations
                    logger.debug("Node type %s does not support ref_doc_id", type(node).__name__)

            results.append(
                NodeWithScore(
                    node=node,
                    score=float(m.score),
                )
            )

        return results

    def _apply_score_threshold(
        self,
        matches: List[VectorMatch],
    ) -> List[VectorMatch]:
        """
        Optionally filter matches by a minimum score threshold.

        This client-side filtering provides consistent behavior across different
        backends that may have varying support for server-side score filtering.
        Applied after query execution to ensure we respect backend result limits.
        """
        if self.score_threshold is None:
            return matches
        threshold = float(self.score_threshold)
        return [m for m in matches if float(m.score) >= threshold]

    # ------------------------------------------------------------------ #
    # Metadata filter translation (richer operators)
    # ------------------------------------------------------------------ #

    def _metadata_filters_to_corpus_filter(
        self,
        filters: Optional[MetadataFilters],
        *,
        doc_ids: Optional[Sequence[str]] = None,
        node_ids: Optional[Sequence[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Translate LlamaIndex MetadataFilters + doc_ids/node_ids into a Corpus
        metadata filter dict.

        Operator mapping (best-effort):
        - EQ / ==        → {key: value}
        - NE / !=        → {key: {"$ne": value}}
        - IN / ANY       → {key: {"$in": [values...]}}
        - NIN / NOT_IN   → {key: {"$nin": [values...]}}
        - GT             → {key: {"$gt": value}}
        - GTE            → {key: {"$gte": value}}
        - LT             → {key: {"$lt": value}}
        - LTE            → {key: {"$lte": value}}

        doc_ids and node_ids are translated to `$in` constraints on
        `self.ref_doc_id_field` and `self.node_id_field`, respectively.

        Unknown operators are logged and skipped rather than causing errors,
        ensuring forward compatibility with new LlamaIndex filter types.

        Returns a filter dict suitable for backends that support metadata filtering,
        or None if no filters are specified.
        """
        clauses: List[Dict[str, Any]] = []

        # User-specified metadata filters from LlamaIndex query
        if filters is not None and getattr(filters, "filters", None):
            for f in filters.filters:
                key = getattr(f, "key", None)
                value = getattr(f, "value", None)
                operator = getattr(f, "operator", None)

                if key is None:
                    continue

                op_name = str(operator).upper() if operator is not None else "EQ"

                if op_name in ("EQ", "=="):
                    clauses.append({key: value})
                elif op_name in ("NE", "!="):
                    clauses.append({key: {"$ne": value}})
                elif op_name in ("IN", "ANY"):
                    if isinstance(value, (list, tuple, set)):
                        clauses.append({key: {"$in": list(value)}})
                    else:
                        clauses.append({key: {"$in": [value]}})
                elif op_name in ("NIN", "NOT_IN"):
                    if isinstance(value, (list, tuple, set)):
                        clauses.append({key: {"$nin": list(value)}})
                    else:
                        clauses.append({key: {"$nin": [value]}})
                elif op_name == "GT":
                    clauses.append({key: {"$gt": value}})
                elif op_name == "GTE":
                    clauses.append({key: {"$gte": value}})
                elif op_name == "LT":
                    clauses.append({key: {"$lt": value}})
                elif op_name == "LTE":
                    clauses.append({key: {"$lte": value}})
                else:
                    logger.debug(
                        "Unsupported metadata filter operator %r for key=%r; skipping",
                        op_name,
                        key,
                    )

        # Restrict by ref_doc_id if doc_ids are provided (common LlamaIndex pattern)
        if doc_ids:
            clauses.append(
                {
                    self.ref_doc_id_field: {
                        "$in": [str(d) for d in doc_ids],
                    }
                }
            )

        # Restrict by node_id if node_ids are provided
        if node_ids:
            clauses.append(
                {
                    self.node_id_field: {
                        "$in": [str(n) for n in node_ids],
                    }
                }
            )

        if not clauses:
            return None

        # Determine boolean condition if MetadataFilters exposes it.
        # LlamaIndex typically uses AND semantics by default.
        condition = getattr(filters, "condition", None) if filters is not None else None
        cond_name = str(condition).upper() if condition is not None else "AND"

        if len(clauses) == 1:
            # Single predicate; no need for $and/$or wrapping.
            return clauses[0]

        if cond_name in ("OR", "ANY"):
            return {"$or": clauses}

        # Default: AND semantics (most common in LlamaIndex queries)
        return {"$and": clauses}

    # ------------------------------------------------------------------ #
    # Raw request builders for VectorTranslator
    # ------------------------------------------------------------------ #

    def _build_upsert_request(
        self,
        vectors: List[Vector],
        *,
        namespace: Optional[str],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Build the raw upsert request + framework_ctx for VectorTranslator.

        The VectorTranslator handles batching according to backend capabilities
        and provides graceful error recovery for partial batch failures.
        """
        ns = self._effective_namespace(namespace)
        raw: Dict[str, Any] = {
            "namespace": ns,
            "vectors": vectors,
        }
        framework_ctx: Dict[str, Any] = {"namespace": ns} if ns is not None else {}
        return raw, framework_ctx

    def _build_query_request(
        self,
        embedding: Sequence[float],
        *,
        top_k: int,
        namespace: Optional[str],
        filter: Optional[Mapping[str, Any]],
        include_vectors: bool,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Build the raw query request + framework_ctx for VectorTranslator.

        Used by both standard queries and MMR queries (when include_vectors=True).
        The translator handles capability validation and query optimization.
        """
        ns = self._effective_namespace(namespace)
        raw: Dict[str, Any] = {
            "vector": [float(x) for x in embedding],
            "top_k": int(top_k),
            "namespace": ns,
            "filter": dict(filter) if filter else None,
            "include_metadata": True,
            "include_vectors": bool(include_vectors),
        }
        framework_ctx: Dict[str, Any] = {"namespace": ns} if ns is not None else {}
        return raw, framework_ctx

    def _build_delete_request(
        self,
        *,
        ids: Optional[List[str]],
        namespace: Optional[str],
        filter: Optional[Mapping[str, Any]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Build the raw delete request + framework_ctx for VectorTranslator.

        The translator handles capability validation and ensures delete operations
        respect backend limits and authentication requirements.
        """
        ns = self._effective_namespace(namespace)
        raw: Dict[str, Any] = {
            "namespace": ns,
            "ids": [str(i) for i in ids] if ids else None,
            "filter": dict(filter) if filter else None,
        }
        framework_ctx: Dict[str, Any] = {"namespace": ns} if ns is not None else {}
        return raw, framework_ctx

    # ------------------------------------------------------------------ #
    # Validation helpers aligned with VectorCapabilities
    # ------------------------------------------------------------------ #

    def _validate_query_params_sync(
        self,
        top_k: int,
        namespace: Optional[str],
        filter: Optional[Mapping[str, Any]],
    ) -> int:
        """
        Validate query parameters in a sync context against capabilities.

        Ensures the requested top_k doesn't exceed backend limits and that
        metadata filtering is supported if filters are provided. Returns the
        validated top_k value for use in the actual query.

        Raises NotSupported if the backend cannot handle the requested operation.
        """
        caps = self._get_caps_sync()
        ns = self._effective_namespace(namespace)
        effective_top_k = int(top_k)

        if caps.max_top_k is not None and effective_top_k > caps.max_top_k:
            err = BadRequest(
                f"top_k {effective_top_k} exceeds maximum of {caps.max_top_k}",
                code="BAD_TOP_K",
                details={"max_top_k": caps.max_top_k, "namespace": ns},
            )
            attach_context(
                err,
                framework="llamaindex",
                operation="query_sync",
                namespace=ns,
                top_k=effective_top_k,
            )
            raise err

        if filter and not caps.supports_metadata_filtering:
            err = NotSupported(
                "metadata filtering is not supported by the underlying vector adapter",
                code="FILTER_NOT_SUPPORTED",
                details={"namespace": ns},
            )
            attach_context(
                err,
                framework="llamaindex",
                operation="query_sync",
                namespace=ns,
                top_k=effective_top_k,
            )
            raise err

        return effective_top_k

    async def _validate_query_params_async(
        self,
        top_k: int,
        namespace: Optional[str],
        filter: Optional[Mapping[str, Any]],
    ) -> int:
        """
        Validate query parameters in an async context against capabilities.

        Async version of _validate_query_params_sync for use in async operations.
        Ensures non-blocking capability checks during async query execution.
        """
        caps = await self._get_caps_async()
        ns = self._effective_namespace(namespace)
        effective_top_k = int(top_k)

        if caps.max_top_k is not None and effective_top_k > caps.max_top_k:
            err = BadRequest(
                f"top_k {effective_top_k} exceeds maximum of {caps.max_top_k}",
                code="BAD_TOP_K",
                details={"max_top_k": caps.max_top_k, "namespace": ns},
            )
            attach_context(
                err,
                framework="llamaindex",
                operation="query_async",
                namespace=ns,
                top_k=effective_top_k,
            )
            raise err

        if filter and not caps.supports_metadata_filtering:
            err = NotSupported(
                "metadata filtering is not supported by the underlying vector adapter",
                code="FILTER_NOT_SUPPORTED",
                details={"namespace": ns},
            )
            attach_context(
                err,
                framework="llamaindex",
                operation="query_async",
                namespace=ns,
                top_k=effective_top_k,
            )
            raise err

        return effective_top_k

    def _validate_delete_params_sync(
        self,
        *,
        ids: Optional[List[str]],
        namespace: Optional[str],
        filter: Optional[Mapping[str, Any]],
    ) -> None:
        """
        Validate delete parameters in a sync context against capabilities.

        Ensures either ids or filter are provided and that metadata filter-based
        deletes are supported by the backend. Prevents invalid delete operations
        that would fail at the backend level.
        """
        caps = self._get_caps_sync()
        ns = self._effective_namespace(namespace)

        if filter and not caps.supports_metadata_filtering:
            err = NotSupported(
                "delete by metadata filter is not supported by the underlying vector adapter",
                code="FILTER_NOT_SUPPORTED",
                details={"namespace": ns},
            )
            attach_context(
                err,
                framework="llamaindex",
                operation="delete_sync",
                namespace=ns,
                ids_count=len(ids or []),
            )
            raise err

        if not ids and not filter:
            err = BadRequest(
                "must provide ids or filter for delete",
                code="BAD_DELETE",
            )
            attach_context(
                err,
                framework="llamaindex",
                operation="delete_sync",
                namespace=ns,
                ids_count=0,
            )
            raise err

    async def _validate_delete_params_async(
        self,
        *,
        ids: Optional[List[str]],
        namespace: Optional[str],
        filter: Optional[Mapping[str, Any]],
    ) -> None:
        """
        Validate delete parameters in an async context against capabilities.

        Async version of _validate_delete_params_sync for use in async delete operations.
        """
        caps = await self._get_caps_async()
        ns = self._effective_namespace(namespace)

        if filter and not caps.supports_metadata_filtering:
            err = NotSupported(
                "delete by metadata filter is not supported by the underlying vector adapter",
                code="FILTER_NOT_SUPPORTED",
                details={"namespace": ns},
            )
            attach_context(
                err,
                framework="llamaindex",
                operation="delete_async",
                namespace=ns,
                ids_count=len(ids or []),
            )
            raise err

        if not ids and not filter:
            err = BadRequest(
                "must provide ids or filter for delete",
                code="BAD_DELETE",
            )
            attach_context(
                err,
                framework="llamaindex",
                operation="delete_async",
                namespace=ns,
                ids_count=0,
            )
            raise err

    def _validate_query_result_type(
        self,
        result: Any,
        *,
        operation: str,
    ) -> QueryResult:
        """
        Ensure the translator returned a QueryResult.

        Provides a safety check against translator misconfiguration or version
        mismatches. Raises BadRequest with detailed context if the result type
        is unexpected.
        """
        if isinstance(result, QueryResult):
            return result

        err = BadRequest(
            f"{operation} returned unsupported type: {type(result).__name__}",
            code="BAD_TRANSLATED_RESULT",
        )
        attach_context(
            err,
            framework="llamaindex",
            operation=operation,
        )
        raise err

    # ------------------------------------------------------------------ #
    # Graceful error recovery for batch operations
    # ------------------------------------------------------------------ #

    def _handle_partial_upsert_failure(
        self,
        result: UpsertResult,
        total_nodes: int,
        namespace: Optional[str],
    ) -> None:
        """
        Handle partial failures in batch upsert operations gracefully.

        Logs warnings for partial failures but doesn't raise exceptions for
        non-critical failures, allowing successful operations to proceed.
        Only raises exceptions for complete failures or critical errors.
        """
        if result.failed_count and result.failed_count > 0:
            successful = result.upserted_count or 0
            failed = result.failed_count
            
            logger.warning(
                "Partial upsert failure: %d/%d nodes succeeded, %d failed in namespace %s",
                successful, total_nodes, failed, namespace or "default"
            )
            
            # Log details of individual failures for debugging
            if result.failures:
                for failure in result.failures[:5]:  # Log first 5 failures
                    logger.debug("Upsert failure: %s", failure)
                if len(result.failures) > 5:
                    logger.debug("... and %d more failures", len(result.failures) - 5)

        # Only raise exception if no nodes were upserted at all
        if (result.upserted_count or 0) == 0 and total_nodes > 0:
            raise VectorAdapterError(
                f"All {total_nodes} nodes failed to upsert",
                code="BATCH_UPSERT_FAILED",
                details={
                    "total_nodes": total_nodes,
                    "namespace": namespace,
                    "failures": result.failures or [],
                }
            )

    # ------------------------------------------------------------------ #
    # MMR helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cosine_sim(a: Sequence[float], b: Sequence[float]) -> float:
        """Simple cosine similarity helper."""
        dot = 0.0
        na = 0.0
        nb = 0.0
        for x, y in zip(a, b):
            dot += x * y
            na += x * x
            nb += y * y
        if na <= 0.0 or nb <= 0.0:
            return 0.0
        return float(dot / (math.sqrt(na) * math.sqrt(nb)))

    def _mmr_select_indices(
        self,
        query_vec: Sequence[float],
        candidate_matches: List[VectorMatch],
        k: int,
        lambda_mult: float,
    ) -> List[int]:
        """
        Improved MMR selector that respects original database scores and caches similarities.

        Args:
            query_vec: The query embedding vector (used for consistency, though we use DB scores)
            candidate_matches: Candidate matches with original scores and vectors
            k: Number of results to select
            lambda_mult: MMR lambda parameter (0-1), higher values favor relevance

        Returns:
            Indices into candidate_matches for selected results

        Note: We use the original database similarity scores for relevance rather than
        recomputing similarities, as these scores are typically more accurate and
        consistent with the backend's similarity metric.
        """
        if not candidate_matches or k <= 0:
            return []

        k = min(k, len(candidate_matches))
        if k == 0:
            return []

        # Use original scores from database as relevance measure
        original_scores = [float(match.score) for match in candidate_matches]
        candidate_vecs = [match.vector.vector or [] for match in candidate_matches]

        # Normalize original scores to 0-1 range for consistency
        max_orig_score = max(original_scores) if original_scores else 1.0
        if max_orig_score <= 0.0:
            normalized_scores = [0.0] * len(original_scores)
        else:
            normalized_scores = [score / max_orig_score for score in original_scores]

        # Precompute all pairwise similarities with caching
        similarity_cache: Dict[Tuple[int, int], float] = {}

        def get_similarity(i: int, j: int) -> float:
            if (i, j) in similarity_cache:
                return similarity_cache[(i, j)]
            if (j, i) in similarity_cache:
                return similarity_cache[(j, i)]

            vec_i = candidate_vecs[i]
            vec_j = candidate_vecs[j]
            if not vec_i or not vec_j or len(vec_i) != len(vec_j):
                sim = 0.0
            else:
                sim = self._cosine_sim(vec_i, vec_j)

            similarity_cache[(i, j)] = sim
            return sim

        selected: List[int] = []
        candidates = list(range(len(candidate_matches)))

        # Start with the most relevant document based on original score
        if candidates:
            first_idx = max(candidates, key=lambda i: normalized_scores[i])
            selected.append(first_idx)
            candidates.remove(first_idx)

        while candidates and len(selected) < k:
            best_idx = None
            best_score = -float("inf")

            for idx in candidates:
                relevance = normalized_scores[idx]

                max_similarity = 0.0
                for sel_idx in selected:
                    similarity = get_similarity(idx, sel_idx)
                    max_similarity = max(max_similarity, similarity)

                mmr_score = lambda_mult * relevance - (1.0 - lambda_mult) * max_similarity

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx is None:
                break

            selected.append(best_idx)
            candidates.remove(best_idx)

        return selected

    # ------------------------------------------------------------------ #
    # LlamaIndex VectorStore sync API
    # ------------------------------------------------------------------ #

    def add(
        self,
        nodes: Sequence[BaseNode],
        **kwargs: Any,
    ) -> List[str]:
        """
        Add LlamaIndex nodes to the vector store (sync).

        Behavior:
        - Expects nodes to already have embeddings (computed by LlamaIndex)
        - Flattens node metadata and stores text alongside vectors
        - Batch behavior is enforced by VectorTranslator according to capabilities
        - Handles partial failures gracefully, logging warnings but not failing
          entirely unless all operations fail

        Args:
            nodes: Sequence of BaseNode objects with pre-computed embeddings
            **kwargs: Additional arguments including:
                - namespace: Optional namespace override
                - callback_manager: For context propagation
                - ctx: Explicit OperationContext

        Returns:
            List of node IDs that were successfully added

        Raises:
            BadRequest: If nodes lack embeddings or other validation fails
            VectorAdapterError: If all nodes fail to upsert
        """
        if not nodes:
            return []

        namespace: Optional[str] = kwargs.get("namespace")
        ctx = self._build_ctx(**kwargs)

        vectors = self._nodes_to_corpus_vectors(nodes, namespace=namespace)
        ids = [str(getattr(n, "node_id", None) or getattr(n, "id_", None)) for n in nodes]

        raw_request, framework_ctx = self._build_upsert_request(
            vectors,
            namespace=namespace,
        )

        try:
            result = self._translator.upsert(
                raw_request,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            
            # Handle partial failures gracefully
            if isinstance(result, UpsertResult):
                self._handle_partial_upsert_failure(result, len(nodes), namespace)
                
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="llamaindex",
                operation="upsert_sync",
                namespace=self._effective_namespace(namespace),
            )
            raise

        return ids

    async def aadd(
        self,
        nodes: Sequence[BaseNode],
        **kwargs: Any,
    ) -> List[str]:
        """
        Add LlamaIndex nodes to the vector store (async).

        Async version of add() with the same behavior and error handling.
        Uses async VectorTranslator methods for non-blocking operation.

        Args:
            nodes: Sequence of BaseNode objects with pre-computed embeddings
            **kwargs: Additional arguments including namespace and context

        Returns:
            List of node IDs that were successfully added

        Raises:
            BadRequest: If nodes lack embeddings or other validation fails
            VectorAdapterError: If all nodes fail to upsert
        """
        if not nodes:
            return []

        namespace: Optional[str] = kwargs.get("namespace")
        ctx = self._build_ctx(**kwargs)

        vectors = self._nodes_to_corpus_vectors(nodes, namespace=namespace)
        ids = [str(getattr(n, "node_id", None) or getattr(n, "id_", None)) for n in nodes]

        raw_request, framework_ctx = self._build_upsert_request(
            vectors,
            namespace=namespace,
        )

        try:
            result = await self._translator.arun_upsert(
                raw_request,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            
            # Handle partial failures gracefully
            if isinstance(result, UpsertResult):
                self._handle_partial_upsert_failure(result, len(nodes), namespace)
                
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="llamaindex",
                operation="upsert_async",
                namespace=self._effective_namespace(namespace),
            )
            raise

        return ids

    def delete(
        self,
        ref_doc_id: str,
        **kwargs: Any,
    ) -> None:
        """
        Delete vectors associated with a given ref_doc_id (sync).

        This implements LlamaIndex's standard VectorStore contract for document
        deletion. All nodes with the given ref_doc_id will be removed.

        Args:
            ref_doc_id: Reference document ID to delete
            **kwargs: Additional arguments including namespace and context

        Note: This operation is atomic at the backend level - either all matching
        vectors are deleted or none are, ensuring consistency.
        """
        namespace: Optional[str] = kwargs.get("namespace")
        ctx = self._build_ctx(**kwargs)

        filter_dict: Dict[str, Any] = {self.ref_doc_id_field: ref_doc_id}

        self._validate_delete_params_sync(
            ids=None,
            namespace=namespace,
            filter=filter_dict,
        )

        raw_request, framework_ctx = self._build_delete_request(
            ids=None,
            namespace=namespace,
            filter=filter_dict,
        )

        try:
            self._translator.delete(
                raw_request,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="llamaindex",
                operation="delete_sync",
                namespace=self._effective_namespace(namespace),
            )
            raise

    async def adelete(
        self,
        ref_doc_id: str,
        **kwargs: Any,
    ) -> None:
        """
        Delete vectors associated with a given ref_doc_id (async).

        Async version of delete() with the same behavior and error handling.

        Args:
            ref_doc_id: Reference document ID to delete
            **kwargs: Additional arguments including namespace and context
        """
        namespace: Optional[str] = kwargs.get("namespace")
        ctx = self._build_ctx(**kwargs)

        filter_dict: Dict[str, Any] = {self.ref_doc_id_field: ref_doc_id}

        await self._validate_delete_params_async(
            ids=None,
            namespace=namespace,
            filter=filter_dict,
        )

        raw_request, framework_ctx = self._build_delete_request(
            ids=None,
            namespace=namespace,
            filter=filter_dict,
        )

        try:
            await self._translator.arun_delete(
                raw_request,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="llamaindex",
                operation="delete_async",
                namespace=self._effective_namespace(namespace),
            )
            raise

    def delete_nodes(
        self,
        node_ids: Sequence[str],
        **kwargs: Any,
    ) -> None:
        """
        Delete vectors by node IDs (sync).

        This is a convenience method some LlamaIndex components use. It maps
        node IDs directly to vector IDs for precise deletion.

        Args:
            node_ids: Sequence of node IDs to delete
            **kwargs: Additional arguments including namespace and context

        Note: Unlike ref_doc_id deletion, this operates on specific node IDs
        and is useful for targeted cleanup operations.
        """
        if not node_ids:
            return

        namespace: Optional[str] = kwargs.get("namespace")
        ctx = self._build_ctx(**kwargs)

        ids = [str(i) for i in node_ids]

        self._validate_delete_params_sync(
            ids=ids,
            namespace=namespace,
            filter=None,
        )

        raw_request, framework_ctx = self._build_delete_request(
            ids=ids,
            namespace=namespace,
            filter=None,
        )

        try:
            self._translator.delete(
                raw_request,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="llamaindex",
                operation="delete_nodes_sync",
                namespace=self._effective_namespace(namespace),
            )
            raise

    async def adelete_nodes(
        self,
        node_ids: Sequence[str],
        **kwargs: Any,
    ) -> None:
        """
        Delete vectors by node IDs (async).

        Async version of delete_nodes() with the same behavior and error handling.

        Args:
            node_ids: Sequence of node IDs to delete
            **kwargs: Additional arguments including namespace and context
        """
        if not node_ids:
            return

        namespace: Optional[str] = kwargs.get("namespace")
        ctx = self._build_ctx(**kwargs)

        ids = [str(i) for i in node_ids]

        await self._validate_delete_params_async(
            ids=ids,
            namespace=namespace,
            filter=None,
        )

        raw_request, framework_ctx = self._build_delete_request(
            ids=ids,
            namespace=namespace,
            filter=None,
        )

        try:
            await self._translator.arun_delete(
                raw_request,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="llamaindex",
                operation="delete_nodes_async",
                namespace=self._effective_namespace(namespace),
            )
            raise

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """
        Perform similarity query and return a VectorStoreQueryResult (sync).

        Assumes `query.query_embedding` has already been computed by LlamaIndex.
        This is the standard query method called by LlamaIndex during retrieval.

        Args:
            query: VectorStoreQuery with query_embedding, filters, and other parameters
            **kwargs: Additional arguments including namespace and context

        Returns:
            VectorStoreQueryResult with nodes, similarities, and IDs

        Raises:
            NotSupported: If query_embedding is None
            BadRequest: If query parameters exceed backend capabilities
        """
        if query.query_embedding is None:
            raise NotSupported(
                "VectorStoreQuery.query_embedding is None; LlamaIndex must "
                "provide a query embedding before calling CorpusLlamaIndexVectorStore.query.",
                code="NO_QUERY_EMBEDDING",
            )

        namespace: Optional[str] = kwargs.get("namespace")
        ctx = self._build_ctx(**kwargs)

        top_k_raw = query.similarity_top_k or self.default_top_k
        corpus_filter = self._metadata_filters_to_corpus_filter(
            query.filters,
            doc_ids=query.doc_ids,
            node_ids=query.node_ids,
        )

        top_k = self._validate_query_params_sync(
            top_k=top_k_raw,
            namespace=namespace,
            filter=corpus_filter,
        )

        embedding = [float(x) for x in query.query_embedding]

        raw_query, framework_ctx = self._build_query_request(
            embedding,
            top_k=top_k,
            namespace=namespace,
            filter=corpus_filter,
            include_vectors=False,
        )

        try:
            result_any = self._translator.query(
                raw_query,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="llamaindex",
                operation="query_sync",
                namespace=self._effective_namespace(namespace),
                top_k=top_k,
            )
            raise

        result = self._validate_query_result_type(
            result_any,
            operation="translator.query_sync",
        )

        matches_list: List[VectorMatch] = list(result.matches or [])
        matches_list = self._apply_score_threshold(matches_list)

        nodes_with_scores = self._matches_to_nodes(matches_list)
        similarities = [nws.score for nws in nodes_with_scores]
        ids = [
            getattr(nws.node, "node_id", None) or getattr(nws.node, "id_", None)
            for nws in nodes_with_scores
        ]

        return VectorStoreQueryResult(
            nodes=nodes_with_scores,
            similarities=similarities,
            ids=[str(i) if i is not None else "" for i in ids],
        )

    async def aquery(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """
        Perform similarity query and return a VectorStoreQueryResult (async).

        Async version of query() with the same behavior and error handling.
        Used by LlamaIndex's async query engines for non-blocking retrieval.

        Args:
            query: VectorStoreQuery with query_embedding, filters, and other parameters
            **kwargs: Additional arguments including namespace and context

        Returns:
            VectorStoreQueryResult with nodes, similarities, and IDs

        Raises:
            NotSupported: If query_embedding is None
            BadRequest: If query parameters exceed backend capabilities
        """
        if query.query_embedding is None:
            raise NotSupported(
                "VectorStoreQuery.query_embedding is None; LlamaIndex must "
                "provide a query embedding before calling aquery.",
                code="NO_QUERY_EMBEDDING",
            )

        namespace: Optional[str] = kwargs.get("namespace")
        ctx = self._build_ctx(**kwargs)

        top_k_raw = query.similarity_top_k or self.default_top_k
        corpus_filter = self._metadata_filters_to_corpus_filter(
            query.filters,
            doc_ids=query.doc_ids,
            node_ids=query.node_ids,
        )

        top_k = await self._validate_query_params_async(
            top_k=top_k_raw,
            namespace=namespace,
            filter=corpus_filter,
        )

        embedding = [float(x) for x in query.query_embedding]

        raw_query, framework_ctx = self._build_query_request(
            embedding,
            top_k=top_k,
            namespace=namespace,
            filter=corpus_filter,
            include_vectors=False,
        )

        try:
            result_any = await self._translator.arun_query(
                raw_query,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="llamaindex",
                operation="query_async",
                namespace=self._effective_namespace(namespace),
                top_k=top_k,
            )
            raise

        result = self._validate_query_result_type(
            result_any,
            operation="translator.query_async",
        )

        matches_list: List[VectorMatch] = list(result.matches or [])
        matches_list = self._apply_score_threshold(matches_list)

        nodes_with_scores = self._matches_to_nodes(matches_list)
        similarities = [nws.score for nws in nodes_with_scores]
        ids = [
            getattr(nws.node, "node_id", None) or getattr(nws.node, "id_", None)
            for nws in nodes_with_scores
        ]

        return VectorStoreQueryResult(
            nodes=nodes_with_scores,
            similarities=similarities,
            ids=[str(i) if i is not None else "" for i in ids],
        )

    def query_stream(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> Iterator[NodeWithScore]:
        """
        Streaming similarity query (sync), yielding NodeWithScore one by one.

        This uses VectorTranslator.query_stream to bridge async query into a
        streaming sync iterator while keeping backend semantics identical.
        Useful for responsive UIs and progressive result display.

        Args:
            query: VectorStoreQuery with query_embedding, filters, and other parameters
            **kwargs: Additional arguments including namespace and context

        Yields:
            NodeWithScore objects one by one as they become available

        Raises:
            NotSupported: If query_embedding is None
            BadRequest: If query parameters exceed backend capabilities
        """
        if query.query_embedding is None:
            raise NotSupported(
                "VectorStoreQuery.query_embedding is None; LlamaIndex must "
                "provide a query embedding before calling query_stream.",
                code="NO_QUERY_EMBEDDING",
            )

        namespace: Optional[str] = kwargs.get("namespace")
        ctx = self._build_ctx(**kwargs)

        top_k_raw = query.similarity_top_k or self.default_top_k
        corpus_filter = self._metadata_filters_to_corpus_filter(
            query.filters,
            doc_ids=query.doc_ids,
            node_ids=query.node_ids,
        )

        top_k = self._validate_query_params_sync(
            top_k=top_k_raw,
            namespace=namespace,
            filter=corpus_filter,
        )

        embedding = [float(x) for x in query.query_embedding]

        raw_query, framework_ctx = self._build_query_request(
            embedding,
            top_k=top_k,
            namespace=namespace,
            filter=corpus_filter,
            include_vectors=False,
        )

        try:
            for item in self._translator.query_stream(
                raw_query,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            ):
                if not isinstance(item, VectorMatch):
                    err = VectorAdapterError(
                        f"translator.query_stream returned unexpected type: {type(item).__name__}"
                    )
                    attach_context(
                        err,
                        framework="llamaindex",
                        operation="query_stream",
                        namespace=self._effective_namespace(namespace),
                        top_k=top_k,
                    )
                    raise err

                match = item
                if self.score_threshold is not None and float(match.score) < float(
                    self.score_threshold
                ):
                    continue

                nodes_with_scores = self._matches_to_nodes([match])
                if nodes_with_scores:
                    yield nodes_with_scores[0]
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="llamaindex",
                operation="query_stream",
                namespace=self._effective_namespace(namespace),
                top_k=top_k,
            )
            raise

    # ------------------------------------------------------------------ #
    # MMR query APIs (sync + async)
    # ------------------------------------------------------------------ #

    def query_mmr(
        self,
        query: VectorStoreQuery,
        *,
        lambda_mult: float = 0.5,
        fetch_k: Optional[int] = None,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """
        Perform Maximal Marginal Relevance (MMR) query (sync).

        This runs a similarity query with a larger `fetch_k` and then
        selects a subset of results via MMR based on vector geometry and
        original database scores. Provides a good balance between relevance
        and diversity in retrieval results.

        Args:
            query: VectorStoreQuery with query_embedding and other parameters
            lambda_mult: MMR lambda parameter (0-1), higher values favor relevance
            fetch_k: Number of candidates to fetch for MMR selection (defaults to 4*k)
            **kwargs: Additional arguments including namespace and context

        Returns:
            VectorStoreQueryResult with MMR-selected nodes and scores

        Raises:
            NotSupported: If query_embedding is None
            BadRequest: If lambda_mult is invalid or parameters exceed capabilities
        """
        if query.query_embedding is None:
            raise NotSupported(
                "VectorStoreQuery.query_embedding is None; LlamaIndex must "
                "provide a query embedding before calling query_mmr.",
                code="NO_QUERY_EMBEDDING",
            )

        if not (0.0 <= lambda_mult <= 1.0):
            err = BadRequest(
                f"lambda_mult must be in [0, 1], got {lambda_mult}",
                code="BAD_MMR_LAMBDA",
            )
            attach_context(
                err,
                framework="llamaindex",
                operation="query_mmr_sync",
                lambda_mult=lambda_mult,
            )
            raise err

        namespace: Optional[str] = kwargs.get("namespace")
        ctx = self._build_ctx(**kwargs)

        k_raw = query.similarity_top_k or self.default_top_k
        k = int(k_raw)
        if k <= 0:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        effective_fetch_k = int(fetch_k or max(k * 4, k + 5))

        corpus_filter = self._metadata_filters_to_corpus_filter(
            query.filters,
            doc_ids=query.doc_ids,
            node_ids=query.node_ids,
        )

        # Validate against capabilities using fetch_k
        top_k_fetch = self._validate_query_params_sync(
            top_k=effective_fetch_k,
            namespace=namespace,
            filter=corpus_filter,
        )

        embedding = [float(x) for x in query.query_embedding]

        raw_query, framework_ctx = self._build_query_request(
            embedding,
            top_k=top_k_fetch,
            namespace=namespace,
            filter=corpus_filter,
            include_vectors=True,
        )

        try:
            result_any = self._translator.query(
                raw_query,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="llamaindex",
                operation="query_mmr_sync",
                namespace=self._effective_namespace(namespace),
                top_k=top_k_fetch,
            )
            raise

        result = self._validate_query_result_type(
            result_any,
            operation="translator.query_mmr_sync",
        )

        matches_list: List[VectorMatch] = list(result.matches or [])
        matches_list = self._apply_score_threshold(matches_list)

        if not matches_list:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        indices = self._mmr_select_indices(
            query_vec=embedding,
            candidate_matches=matches_list,
            k=k,
            lambda_mult=lambda_mult,
        )

        selected_matches = [matches_list[i] for i in indices]
        nodes_with_scores = self._matches_to_nodes(selected_matches)
        similarities = [nws.score for nws in nodes_with_scores]
        ids = [
            getattr(nws.node, "node_id", None) or getattr(nws.node, "id_", None)
            for nws in nodes_with_scores
        ]

        return VectorStoreQueryResult(
            nodes=nodes_with_scores,
            similarities=similarities,
            ids=[str(i) if i is not None else "" for i in ids],
        )

    async def aquery_mmr(
        self,
        query: VectorStoreQuery,
        *,
        lambda_mult: float = 0.5,
        fetch_k: Optional[int] = None,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """
        Perform Maximal Marginal Relevance (MMR) query (async).

        Async version of query_mmr() with the same behavior and error handling.
        Useful for non-blocking MMR retrieval in async LlamaIndex applications.

        Args:
            query: VectorStoreQuery with query_embedding and other parameters
            lambda_mult: MMR lambda parameter (0-1), higher values favor relevance
            fetch_k: Number of candidates to fetch for MMR selection (defaults to 4*k)
            **kwargs: Additional arguments including namespace and context

        Returns:
            VectorStoreQueryResult with MMR-selected nodes and scores

        Raises:
            NotSupported: If query_embedding is None
            BadRequest: If lambda_mult is invalid or parameters exceed capabilities
        """
        if query.query_embedding is None:
            raise NotSupported(
                "VectorStoreQuery.query_embedding is None; LlamaIndex must "
                "provide a query embedding before calling aquery_mmr.",
                code="NO_QUERY_EMBEDDING",
            )

        if not (0.0 <= lambda_mult <= 1.0):
            err = BadRequest(
                f"lambda_mult must be in [0, 1], got {lambda_mult}",
                code="BAD_MMR_LAMBDA",
            )
            attach_context(
                err,
                framework="llamaindex",
                operation="query_mmr_async",
                lambda_mult=lambda_mult,
            )
            raise err

        namespace: Optional[str] = kwargs.get("namespace")
        ctx = self._build_ctx(**kwargs)

        k_raw = query.similarity_top_k or self.default_top_k
        k = int(k_raw)
        if k <= 0:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        effective_fetch_k = int(fetch_k or max(k * 4, k + 5))

        corpus_filter = self._metadata_filters_to_corpus_filter(
            query.filters,
            doc_ids=query.doc_ids,
            node_ids=query.node_ids,
        )

        top_k_fetch = await self._validate_query_params_async(
            top_k=effective_fetch_k,
            namespace=namespace,
            filter=corpus_filter,
        )

        embedding = [float(x) for x in query.query_embedding]

        raw_query, framework_ctx = self._build_query_request(
            embedding,
            top_k=top_k_fetch,
            namespace=namespace,
            filter=corpus_filter,
            include_vectors=True,
        )

        try:
            result_any = await self._translator.arun_query(
                raw_query,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="llamaindex",
                operation="query_mmr_async",
                namespace=self._effective_namespace(namespace),
                top_k=top_k_fetch,
            )
            raise

        result = self._validate_query_result_type(
            result_any,
            operation="translator.query_mmr_async",
        )

        matches_list: List[VectorMatch] = list(result.matches or [])
        matches_list = self._apply_score_threshold(matches_list)

        if not matches_list:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        indices = self._mmr_select_indices(
            query_vec=embedding,
            candidate_matches=matches_list,
            k=k,
            lambda_mult=lambda_mult,
        )

        selected_matches = [matches_list[i] for i in indices]
        nodes_with_scores = self._matches_to_nodes(selected_matches)
        similarities = [nws.score for nws in nodes_with_scores]
        ids = [
            getattr(nws.node, "node_id", None) or getattr(nws.node, "id_", None)
            for nws in nodes_with_scores
        ]

        return VectorStoreQueryResult(
            nodes=nodes_with_scores,
            similarities=similarities,
            ids=[str(i) if i is not None else "" for i in ids],
        )


__all__ = [
    "CorpusLlamaIndexVectorStore",
]