# corpus_sdk/vector/framework_adapters/llamaindex.py
# SPDX-License-Identifier: Apache-2.0

"""
LlamaIndex adapter for Corpus Vector protocol.

This module exposes Corpus `VectorProtocolV1` implementations as
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
  the underlying `VectorProtocolV1` / `VectorTranslator`, not here.
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
from functools import cached_property, wraps
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
    VectorProtocolV1,
    Vector,
    VectorMatch,
    QueryChunk,
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
from corpus_sdk.vector.framework_adapters.common.framework_utils import (
    VectorCoercionErrorCodes,
    VectorResourceLimits,
    VectorValidationFlags,
    TopKWarningConfig,
    warn_if_extreme_k,
    normalize_vector_context,
    attach_vector_context_to_framework_ctx,
)

from corpus_sdk.core.context_translation import (
    from_llamaindex as ctx_from_llamaindex,
    from_dict as ctx_from_dict,
)
from corpus_sdk.core.error_context import attach_context

logger = logging.getLogger(__name__)

Metadata = Dict[str, Any]

# --------------------------------------------------------------------------- #
# Shared vector framework configuration / limits
# --------------------------------------------------------------------------- #

VECTOR_ERROR_CODES = VectorCoercionErrorCodes(framework_label="llamaindex")
VECTOR_LIMITS = VectorResourceLimits()
VECTOR_FLAGS = VectorValidationFlags()
TOPK_WARNING_CONFIG = TopKWarningConfig(framework_label="llamaindex")


# --------------------------------------------------------------------------- #
# Error-context decorators (sync + async)
# --------------------------------------------------------------------------- #


def with_error_context(
    operation: str,
    **context_kwargs: Any,
):
    def decorator(func):
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                attach_context(
                    exc,
                    framework="llamaindex",
                    operation=operation,
                    **context_kwargs,
                )
                raise

        return wrapper

    return decorator


def with_async_error_context(
    operation: str,
    **context_kwargs: Any,
):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                attach_context(
                    exc,
                    framework="llamaindex",
                    operation=operation,
                    **context_kwargs,
                )
                raise

        return wrapper

    return decorator


class CorpusLlamaIndexVectorStore(BasePydanticVectorStore):
    """
    LlamaIndex `BasePydanticVectorStore` implementation backed by a Corpus
    `VectorProtocolV1`.

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
    corpus_adapter: VectorProtocolV1 = Field(
        ...,
        description="Underlying Corpus vector adapter implementing VectorProtocolV1",
    )

    namespace: Optional[str] = Field(
        "default",
        description="Default namespace for all operations. Ignored if backend doesn't support namespaces.",
    )

    batch_size: int = Field(
        default=100,
        ge=1,
        le=10000,
        description=(
            "Planning hint for upsert/delete batches. Actual batch size is "
            "min(batch_size, backend_max_batch_size)"
        ),
    )

    default_top_k: int = Field(
        default=4,
        ge=1,
        le=1000,
        description=(
            "Default number of results returned when query doesn't "
            "specify similarity_top_k"
        ),
    )

    score_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Optional minimum similarity score (0.0-1.0) for client-side "
            "filtering of results"
        ),
    )

    # Reserved metadata keys for internal mapping
    id_field: str = Field(
        default="id",
        description="Metadata key for storing the vector ID (maps to node_id)",
    )
    text_field: str = Field(
        default="text",
        description="Metadata key for storing node text content",
    )
    node_id_field: str = Field(
        default="node_id",
        description="Metadata key for storing LlamaIndex node ID",
    )
    ref_doc_id_field: str = Field(
        default="ref_doc_id",
        description="Metadata key for storing reference document ID",
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
                v,
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
                v,
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
                    v,
                )
        return v

    @model_validator(mode="after")
    def validate_reserved_fields(self) -> CorpusLlamaIndexVectorStore:
        """Ensure reserved metadata field names don't conflict."""
        reserved_fields = {
            self.id_field,
            self.text_field,
            self.node_id_field,
            self.ref_doc_id_field,
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
    def client(self) -> VectorProtocolV1:
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
        2. Dict-like context via ctx_from_dict.
        3. LlamaIndex CallbackManager via ctx_from_llamaindex.
        4. None (no context).
        """
        ctx = kwargs.get("ctx") or kwargs.get("operation_context")
        if isinstance(ctx, OperationContext):
            return ctx

        if isinstance(ctx, Mapping):
            try:
                maybe = ctx_from_dict(ctx)
                if isinstance(maybe, OperationContext):
                    return maybe
            except Exception as exc:  # noqa: BLE001
                logger.debug("ctx_from_dict failed in _build_ctx: %s", exc)

        callback_manager = kwargs.get("callback_manager")
        if callback_manager is None:
            return None

        try:
            maybe = ctx_from_llamaindex(callback_manager)
            if isinstance(maybe, OperationContext):
                return maybe
        except Exception as exc:  # noqa: BLE001
            logger.debug("ctx_from_llamaindex failed: %s", exc)

        return None

    def _effective_namespace(self, namespace: Optional[str]) -> Optional[str]:
        """
        Resolve namespace using explicit override or store default.
        """
        return namespace if namespace is not None else self.namespace

    def _framework_ctx_for_namespace(
        self,
        namespace: Optional[str],
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build a framework_ctx enriched with normalized vector context.
        """
        ns = self._effective_namespace(namespace)
        raw_ctx: Dict[str, Any] = {}
        if ns is not None:
            raw_ctx["namespace"] = ns
        if extra_context:
            raw_ctx.update(extra_context)

        vector_ctx = normalize_vector_context(
            raw_ctx,
            framework="llamaindex",
            logger=logger,
        )

        framework_ctx: Dict[str, Any] = {}
        attach_vector_context_to_framework_ctx(
            framework_ctx,
            vector_context=vector_ctx,
            limits=VECTOR_LIMITS,
            flags=VECTOR_FLAGS,
        )
        return framework_ctx

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
        """
        vectors: List[Vector] = []
        ns = self._effective_namespace(namespace)

        for node in nodes:
            # Extract embedding - LlamaIndex should have computed this already
            embedding = (
                node.get_embedding() if hasattr(node, "get_embedding") else None
            )
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
                node = metadata_dict_to_node(meta, text=text, node_id=node_id)
            except Exception as exc:
                logger.debug(
                    "metadata_dict_to_node failed for node %s: %s; trying legacy method",
                    node_id,
                    exc,
                )
                try:
                    node = legacy_metadata_dict_to_node(
                        meta, text=text, node_id=node_id
                    )
                except Exception as exc2:
                    logger.debug(
                        "legacy_metadata_dict_to_node also failed for node %s: %s; "
                        "falling back to TextNode",
                        node_id,
                        exc2,
                    )
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
                    logger.debug(
                        "Node type %s does not support ref_doc_id",
                        type(node).__name__,
                    )

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

        # Restrict by ref_doc_id if doc_ids are provided
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

        # Determine boolean condition
        condition = getattr(filters, "condition", None) if filters is not None else None
        cond_name = str(condition).upper() if condition is not None else "AND"

        if len(clauses) == 1:
            return clauses[0]

        if cond_name in ("OR", "ANY"):
            return {"$or": clauses}

        # Default: AND semantics
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
        """
        ns = self._effective_namespace(namespace)
        raw: Dict[str, Any] = {
            "namespace": ns,
            "vectors": vectors,
        }
        framework_ctx = self._framework_ctx_for_namespace(ns)
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
        """
        ns = self._effective_namespace(namespace)
        raw: Dict[str, Any] = {
            "vector": [float(x) for x in embedding],
            "top_k": int(top_k),
            "namespace": ns,
            "filters": dict(filter) if filter else None,
            "include_metadata": True,
            "include_vectors": bool(include_vectors),
        }
        framework_ctx = self._framework_ctx_for_namespace(ns)
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
        """
        ns = self._effective_namespace(namespace)
        raw: Dict[str, Any] = {
            "namespace": ns,
            "ids": [str(i) for i in ids] if ids else None,
            "filter": dict(filter) if filter else None,
        }
        framework_ctx = self._framework_ctx_for_namespace(ns)
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
        """
        if result.failed_count and result.failed_count > 0:
            successful = result.upserted_count or 0
            failed = result.failed_count

            logger.warning(
                "Partial upsert failure: %d/%d nodes succeeded, %d failed in namespace %s",
                successful,
                total_nodes,
                failed,
                namespace or "default",
            )

            if result.failures:
                for failure in result.failures[:5]:
                    logger.debug("Upsert failure: %s", failure)
                if len(result.failures) > 5:
                    logger.debug(
                        "... and %d more failures", len(result.failures) - 5
                    )

        if (result.upserted_count or 0) == 0 and total_nodes > 0:
            raise VectorAdapterError(
                f"All {total_nodes} nodes failed to upsert",
                code="BATCH_UPSERT_FAILED",
                details={
                    "total_nodes": total_nodes,
                    "namespace": namespace,
                    "failures": result.failures or [],
                },
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
        """
        if not candidate_matches or k <= 0:
            return []

        k = min(k, len(candidate_matches))
        if k == 0:
            return []

        # If lambda_mult == 1.0, pure relevance ranking
        if lambda_mult >= 1.0:
            scores = [float(m.score) for m in candidate_matches]
            sorted_indices = sorted(
                range(len(candidate_matches)),
                key=lambda i: scores[i],
                reverse=True,
            )
            return sorted_indices[:k]

        # Use original scores from database as relevance measure
        original_scores = [float(match.score) for match in candidate_matches]
        dim = len(query_vec)

        # Candidate vectors, tolerating missing/mismatched dims
        candidate_vecs: List[List[float]] = []
        for match in candidate_matches:
            vec = match.vector.vector or []
            if not vec or (dim > 0 and len(vec) != dim):
                candidate_vecs.append([])
            else:
                candidate_vecs.append([float(x) for x in vec])

        max_orig_score = max(original_scores) if original_scores else 1.0
        if max_orig_score <= 0.0:
            normalized_scores = [0.0] * len(original_scores)
        else:
            normalized_scores = [score / max_orig_score for score in original_scores]

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

    @with_error_context("add_sync")
    def add(
        self,
        nodes: Sequence[BaseNode],
        **kwargs: Any,
    ) -> List[str]:
        """
        Add LlamaIndex nodes to the vector store (sync).
        """
        if not nodes:
            return []

        namespace: Optional[str] = kwargs.get("namespace")
        ctx = self._build_ctx(**kwargs)

        vectors = self._nodes_to_corpus_vectors(nodes, namespace=namespace)
        ids = [
            str(getattr(n, "node_id", None) or getattr(n, "id_", None)) for n in nodes
        ]

        raw_request, framework_ctx = self._build_upsert_request(
            vectors,
            namespace=namespace,
        )

        result = self._translator.upsert(
            raw_request,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )

        if isinstance(result, UpsertResult):
            self._handle_partial_upsert_failure(result, len(nodes), namespace)

        return ids

    @with_async_error_context("add_async")
    async def aadd(
        self,
        nodes: Sequence[BaseNode],
        **kwargs: Any,
    ) -> List[str]:
        """
        Add LlamaIndex nodes to the vector store (async).
        """
        if not nodes:
            return []

        namespace: Optional[str] = kwargs.get("namespace")
        ctx = self._build_ctx(**kwargs)

        vectors = self._nodes_to_corpus_vectors(nodes, namespace=namespace)
        ids = [
            str(getattr(n, "node_id", None) or getattr(n, "id_", None)) for n in nodes
        ]

        result = await self._translator.arun_upsert(
            raw_request := self._build_upsert_request(
                vectors,
                namespace=namespace,
            )[0],
            op_ctx=ctx,
            framework_ctx=self._build_upsert_request(
                vectors,
                namespace=namespace,
            )[1],
        )

        if isinstance(result, UpsertResult):
            self._handle_partial_upsert_failure(result, len(nodes), namespace)

        return ids

    @with_error_context("delete_sync")
    def delete(
        self,
        ref_doc_id: str,
        **kwargs: Any,
    ) -> None:
        """
        Delete vectors associated with a given ref_doc_id (sync).
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

        self._translator.delete(
            raw_request,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )

    @with_async_error_context("delete_async")
    async def adelete(
        self,
        ref_doc_id: str,
        **kwargs: Any,
    ) -> None:
        """
        Delete vectors associated with a given ref_doc_id (async).
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

        await self._translator.arun_delete(
            raw_request,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )

    @with_error_context("delete_nodes_sync")
    def delete_nodes(
        self,
        node_ids: Sequence[str],
        **kwargs: Any,
    ) -> None:
        """
        Delete vectors by node IDs (sync).
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

        self._translator.delete(
            raw_request,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )

    @with_async_error_context("delete_nodes_async")
    async def adelete_nodes(
        self,
        node_ids: Sequence[str],
        **kwargs: Any,
    ) -> None:
        """
        Delete vectors by node IDs (async).
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

        await self._translator.arun_delete(
            raw_request,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )

    @with_error_context("query_sync")
    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """
        Perform similarity query and return a VectorStoreQueryResult (sync).
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

        warn_if_extreme_k(
            top_k,
            framework="llamaindex",
            op_name="query_sync",
            warning_config=TOPK_WARNING_CONFIG,
            logger=logger,
        )

        embedding = [float(x) for x in query.query_embedding]

        raw_query, framework_ctx = self._build_query_request(
            embedding,
            top_k=top_k,
            namespace=namespace,
            filter=corpus_filter,
            include_vectors=False,
        )

        result_any = self._translator.query(
            raw_query,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )

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

    @with_async_error_context("query_async")
    async def aquery(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """
        Perform similarity query and return a VectorStoreQueryResult (async).
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

        warn_if_extreme_k(
            top_k,
            framework="llamaindex",
            op_name="query_async",
            warning_config=TOPK_WARNING_CONFIG,
            logger=logger,
        )

        embedding = [float(x) for x in query.query_embedding]

        raw_query, framework_ctx = self._build_query_request(
            embedding,
            top_k=top_k,
            namespace=namespace,
            filter=corpus_filter,
            include_vectors=False,
        )

        result_any = await self._translator.arun_query(
            raw_query,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )

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

    @with_error_context("query_stream")
    def query_stream(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> Iterator[NodeWithScore]:
        """
        Streaming similarity query (sync), yielding NodeWithScore one by one.
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

        warn_if_extreme_k(
            top_k,
            framework="llamaindex",
            op_name="query_stream",
            warning_config=TOPK_WARNING_CONFIG,
            logger=logger,
        )

        embedding = [float(x) for x in query.query_embedding]

        raw_query, framework_ctx = self._build_query_request(
            embedding,
            top_k=top_k,
            namespace=namespace,
            filter=corpus_filter,
            include_vectors=False,
        )

        yielded = 0

        for chunk in self._translator.query_stream(
            raw_query,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        ):
            if not isinstance(chunk, QueryChunk):
                err = VectorAdapterError(
                    f"translator.query_stream returned unexpected type: {type(chunk).__name__}",
                    code="BAD_STREAM_CHUNK",
                )
                attach_context(
                    err,
                    framework="llamaindex",
                    operation="query_stream",
                    namespace=self._effective_namespace(namespace),
                    top_k=top_k,
                )
                raise err

            raw_matches = list(chunk.matches or [])
            filtered_matches = self._apply_score_threshold(raw_matches)

            for match in filtered_matches:
                if yielded >= top_k:
                    return
                nodes_with_scores = self._matches_to_nodes([match])
                if nodes_with_scores:
                    yield nodes_with_scores[0]
                    yielded += 1

    # ------------------------------------------------------------------ #
    # MMR query APIs (sync + async)
    # ------------------------------------------------------------------ #

    @with_error_context("query_mmr_sync")
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

        warn_if_extreme_k(
            effective_fetch_k,
            framework="llamaindex",
            op_name="query_mmr_sync_fetch_k",
            warning_config=TOPK_WARNING_CONFIG,
            logger=logger,
        )

        corpus_filter = self._metadata_filters_to_corpus_filter(
            query.filters,
            doc_ids=query.doc_ids,
            node_ids=query.node_ids,
        )

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

        result_any = self._translator.query(
            raw_query,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )

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

    @with_async_error_context("query_mmr_async")
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

        warn_if_extreme_k(
            effective_fetch_k,
            framework="llamaindex",
            op_name="query_mmr_async_fetch_k",
            warning_config=TOPK_WARNING_CONFIG,
            logger=logger,
        )

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

        result_any = await self._translator.arun_query(
            raw_query,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )

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
    "with_error_context",
    "with_async_error_context",
]
