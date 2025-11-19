# corpus_sdk/vector/framework_adapters/common/vector_translation.py
# SPDX-License-Identifier: Apache-2.0
"""
Framework-agnostic Vector → Framework translation layer.

Purpose
-------
Provide a high-level orchestration and translation layer between:

- The Corpus Vector Protocol V1 (`VectorProtocolV1` / `BaseVectorAdapter`), and
- Framework-specific vector integrations (LangChain, LlamaIndex, SK, AutoGen, CrewAI, custom).

This module is intentionally *framework-neutral* and focuses on:

- Building `VectorQuerySpec` / upsert specs from framework-level inputs
- Translating `QueryResult` / `QueryChunk` / upsert results back to framework-facing shapes
- Applying optional MMR / diversification on query results (CRITICAL for vector search)
- Handling rich metadata filters (including $and / $or / range operators)
- Providing sync + async APIs, including streaming via a sync bridge
- Attaching rich error context for observability

Context translation
-------------------
This module does **not** parse framework configs directly. Instead:

- `corpus_sdk.core.context_translation` is responsible for taking framework-native
  contexts (LangChain RunnableConfig, LlamaIndex CallbackManager, etc.) and producing
  a core `OperationContext`.
- Callers pass either an `OperationContext` or a simple dict-like context into
  the methods here; we normalize that via `from_dict` into the vector adapter's
  `OperationContext`.

MMR / diversification
---------------------
MMR (Maximal Marginal Relevance) is CRITICAL for vector search quality:

- Reduces redundancy in retrieved documents
- Balances relevance vs diversity
- Configurable lambda parameter: relevance weight vs diversity weight
- Uses cosine similarity by default, with pluggable similarity functions

This layer supports MMR re-ranking on vector query results that expose:
- A relevance score field (e.g. "score", "distance", "similarity")
- An embedding field (e.g. "embedding", "vector")

Filter handling
---------------
Metadata filters can use a rich, adapter-agnostic DSL:

- Logical combinators:    {"$and": [...]} / {"$or": [...]}
- Range operators:        {"field": {"$gt": v, "$lt": v2}} or tuple ["field", ">", v]
- Simple equality:        {"field": value}

The normalized filter is passed through to the vector adapter, which may further
interpret or constrain semantics based on its own capabilities.

Streaming
---------
For streaming queries, this module exposes:

- An async API that yields translated framework chunks, and
- A sync API that wraps the async generator via `SyncStreamBridge`, preserving
  proper cancellation and error propagation.

Registry
--------
A small registry lets you register per-framework vector translators:

- `register_vector_translator("my_framework", factory)`
- `create_vector_translator("my_framework", adapter, ...)`

This makes it straightforward to plug in framework-specific behaviors while
reusing the common orchestration logic here.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from corpus_sdk.vector.vector_base import (
    VectorProtocolV1,
    OperationContext,
    VectorQuerySpec,
    VectorUpsertSpec,
    VectorDeleteSpec,
    VectorUpdateSpec,
    QueryResult,
    QueryChunk,
    UpsertResult,
    DeleteResult,
    UpdateResult,
    VectorStats,
    VectorAdapterError,
    BadRequest,
    NotSupported,
)

from corpus_sdk.core.context_translation import from_dict as ctx_from_dict
from corpus_sdk.core.sync_stream_bridge import SyncStreamBridge
from corpus_sdk.core.async_bridge import AsyncBridge
from corpus_sdk.llm.framework_adapters.common.error_context import attach_context

LOG = logging.getLogger(__name__)

T = TypeVar("T")
Record = Mapping[str, Any]


# =============================================================================
# Helpers: OperationContext normalization
# =============================================================================


def _ensure_operation_context(
    ctx: Optional[Union[OperationContext, Mapping[str, Any]]],
) -> OperationContext:
    """
    Normalize various context shapes into a vector OperationContext.

    Accepts:
        - None: returns an "empty" context
        - OperationContext: returned as-is
        - Mapping[str, Any]: interpreted via context_translation.from_dict,
          then adapted into a vector OperationContext.

    This keeps responsibilities clean:
        - Framework-native → Normalized dict/OperationContext happens in
          corpus_sdk.core.context_translation (from_langchain, from_llamaindex, etc.)
        - This helper simply ensures the vector adapter receives the right type.
    """
    if ctx is None:
        core_ctx = ctx_from_dict({})
    elif isinstance(ctx, OperationContext):
        return ctx
    elif isinstance(ctx, Mapping):
        core_ctx = ctx_from_dict(ctx)
    else:
        raise BadRequest(
            f"Unsupported context type: {type(ctx).__name__}",
            code="BAD_OPERATION_CONTEXT",
        )

    # Reconstruct as vector OperationContext with validation
    return OperationContext(
        request_id=getattr(core_ctx, "request_id", None),
        idempotency_key=getattr(core_ctx, "idempotency_key", None),
        deadline_ms=getattr(core_ctx, "deadline_ms", None),
        traceparent=getattr(core_ctx, "traceparent", None),
        tenant=getattr(core_ctx, "tenant", None),
        attrs=getattr(core_ctx, "attrs", None) or {},
    )


# =============================================================================
# MMR configuration
# =============================================================================


SimilarityFn = Callable[[Sequence[float], Sequence[float]], float]


@dataclass(frozen=True)
class MMRConfig:
    """
    Configuration for Maximal Marginal Relevance re-ranking.

    MMR is CRITICAL for vector search quality:
    - Reduces redundancy in retrieved documents
    - Balances relevance (from vector similarity) vs diversity
    - Especially important for RAG applications where diverse context is valuable

    Attributes:
        enabled:
            Whether to apply MMR. If False, results are returned as-is.

        k:
            Number of results to keep after diversification. If None, the full
            set of records is re-ordered but not truncated.

        lambda_mult:
            Tradeoff between relevance and diversity, in [0, 1].
            Higher (e.g. 0.9) → more weight on original relevance scores
            Lower (e.g. 0.3)  → more weight on diversity between results
            Default 0.5 balances both equally.

        score_key:
            Record field name containing the original relevance score.
            Common values: "score", "distance", "similarity", "relevance"
            Must be convertible to float.

        vector_key:
            Record field name containing the embedding vector for diversity.
            Common values: "embedding", "vector", "dense_vector"
            Must be a sequence of floats.

        similarity_fn:
            Optional custom similarity metric between two embedding vectors.
            If None, cosine similarity is used (standard for vector search).
            
        invert_score:
            If True, treat lower scores as better (e.g., L2 distance).
            If False, treat higher scores as better (e.g., cosine similarity).
            Default False assumes higher = more similar.
    """

    enabled: bool = False
    k: Optional[int] = None
    lambda_mult: float = 0.5
    score_key: str = "score"
    vector_key: str = "embedding"
    similarity_fn: Optional[SimilarityFn] = None
    invert_score: bool = False

    def __post_init__(self):
        """Validate MMR configuration parameters."""
        if self.enabled:
            if not (0.0 <= self.lambda_mult <= 1.0):
                raise ValueError(f"lambda_mult must be in [0, 1], got {self.lambda_mult}")
            if self.k is not None and self.k < 0:
                raise ValueError(f"k must be non-negative, got {self.k}")

    def effective_k(self, n: int) -> int:
        """Resolve final k against the number of available records."""
        if not self.enabled:
            return n
        if self.k is None or self.k <= 0:
            return n
        return min(self.k, n)


# =============================================================================
# Filter translation helpers
# =============================================================================


class FilterTranslator:
    """
    Helper for normalizing framework-level filter DSLs into a mapping suitable
    for VectorProtocol filters.

    Supported input shapes (examples):
        - {"field": "value"}
        - {"$and": [ {...}, {...} ]}
        - {"$or":  [ {...}, {...} ]}
        - ["field", ">", 10]          → {"field": {"$gt": 10}}
        - [ ["age", ">", 18], ["age", "<", 65] ]
        - [{"field": "v1"}, {"field": "v2"}]  → {"$and": [...]}

    This module intentionally does not enforce the semantics of these operators;
    adapters can interpret or restrict them as needed.
    """

    _RANGE_OP_MAP: Mapping[str, str] = {
        ">": "$gt",
        "<": "$lt",
        ">=": "$gte",
        "<=": "$lte",
        "==": "$eq",
        "=": "$eq",
        "!=": "$ne",
    }

    def normalize(self, raw: Any) -> Optional[Mapping[str, Any]]:
        """Normalize arbitrary filter DSL into a mapping, or None."""
        if raw is None:
            return None

        # Already a mapping: shallow-copy to avoid mutating caller state.
        if isinstance(raw, Mapping):
            return self._normalize_mapping(raw)

        # Sequence of filters or single tuple condition.
        if isinstance(raw, (list, tuple)):
            return self._normalize_sequence(raw)

        # Anything else is unsupported.
        raise BadRequest(
            f"Unsupported filter type: {type(raw).__name__}",
            code="BAD_FILTER",
        )

    def _normalize_mapping(self, raw: Mapping[str, Any]) -> Mapping[str, Any]:
        # If it already looks like a logical combinator or range structure,
        # we keep it as-is but ensure nested filters are normalized.
        if "$and" in raw or "$or" in raw:
            out: Dict[str, Any] = {}
            for key, value in raw.items():
                if key in ("$and", "$or") and isinstance(value, (list, tuple)):
                    out[key] = [self.normalize(v) for v in value]
                else:
                    out[key] = value
            return out

        # Otherwise, treat as a simple equality / basic range map.
        out: Dict[str, Any] = {}
        for key, value in raw.items():
            if isinstance(value, (list, tuple)) and len(value) == 2:
                # Example: {"age": (">", 18)} -> {"age": {"$gt": 18}}
                op, v = value
                mapped = self._RANGE_OP_MAP.get(str(op))
                if mapped:
                    out[key] = {mapped: v}
                else:
                    # Unrecognized operator - log warning and keep as-is
                    LOG.warning(
                        "FilterTranslator: unrecognized range operator %r for field %r; "
                        "keeping original value",
                        op,
                        key,
                    )
                    out[key] = value
            else:
                out[key] = value
        return out

    def _normalize_sequence(self, raw: Sequence[Any]) -> Mapping[str, Any]:
        # Tuple condition: ["age", ">", 18]
        if len(raw) == 3 and not isinstance(raw[0], (list, tuple, Mapping)):
            field, op, value = raw
            mapped = self._RANGE_OP_MAP.get(str(op))
            if not mapped:
                raise BadRequest(
                    f"Unsupported filter operator: {op!r}",
                    code="BAD_FILTER_OPERATOR",
                    details={"operator": op},
                )
            return {str(field): {mapped: value}}

        # Otherwise, assume a list of filters combined via AND.
        parts = [self.normalize(part) for part in raw]
        return {"$and": parts}


# =============================================================================
# Framework-agnostic translator protocol
# =============================================================================


class VectorFrameworkTranslator(Protocol):
    """
    Per-framework translator contract.

    Implementations are responsible for:
        - Converting framework-level query/mutation inputs into Vector*Spec types
        - Converting Vector results into framework-level outputs
        - (Optionally) applying MMR or other post-processing steps

    The default implementation provided here is generic and treats inputs as
    dicts/arrays that already closely match VectorSpec shapes. Frameworks with
    richer abstractions (LangChain VectorStore, LlamaIndex VectorIndex, etc.) 
    can provide their own implementations and register them via the registry.
    """

    # ---- query translation ----

    def build_query_spec(
        self,
        raw_query: Any,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
        stream: bool = False,
    ) -> VectorQuerySpec:
        ...

    def translate_query_result(
        self,
        result: QueryResult,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        ...

    def translate_query_chunk(
        self,
        chunk: QueryChunk,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        ...

    # ---- mutation translation ----

    def build_upsert_spec(
        self,
        raw_documents: Any,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> VectorUpsertSpec:
        ...

    def translate_upsert_result(
        self,
        result: UpsertResult,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        ...

    def build_delete_spec(
        self,
        raw_filter_or_ids: Any,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> VectorDeleteSpec:
        ...

    def translate_delete_result(
        self,
        result: DeleteResult,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        ...

    def build_update_spec(
        self,
        raw_updates: Any,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> VectorUpdateSpec:
        ...

    def translate_update_result(
        self,
        result: UpdateResult,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        ...

    # ---- stats / inspection ----

    def translate_stats(
        self,
        stats: VectorStats,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        ...

    # ---- optional hooks ----

    def preferred_namespace(
        self,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Optional[str]:
        """
        Optional hook for translators to derive a default namespace/collection.

        This can come from:
            - framework_ctx (e.g., which collection/index is active)
            - op_ctx.attrs (e.g., "vector_namespace" or "collection" key)
        """
        ...


# =============================================================================
# Default generic translator implementation
# =============================================================================


class DefaultVectorFrameworkTranslator:
    """
    Generic, framework-neutral translator implementation.

    Behaviors:
        - Treats raw_query as either:
            * a list/array (query vector), or
            * a mapping with VectorQuerySpec-like keys
        - Treats mutation inputs as:
            * sequences of dicts with Document fields, or
            * sequences of already-built Document-compatible mappings

        - For results:
            * QueryResult → list of matches with scores
            * QueryChunk  → list of matches per chunk
            * UpsertResult → list of IDs
            * DeleteResult → count
            * UpdateResult → count
            * VectorStats → underlying stats object

    Frameworks with richer abstractions (LangChain Document, LlamaIndex Node, etc.)
    are expected to provide their own VectorFrameworkTranslator implementation.
    """

    def __init__(self) -> None:
        self._filter_translator = FilterTranslator()

    # ---- namespace helper ----

    def preferred_namespace(
        self,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Optional[str]:
        # Priority: explicit framework_ctx > vector_namespace attr > collection attr > namespace attr
        if isinstance(framework_ctx, Mapping):
            # Try collection first (vector-specific)
            ns = framework_ctx.get("collection") or framework_ctx.get("namespace")
            if ns is not None and str(ns).strip():
                return str(ns)
        
        attrs = op_ctx.attrs or {}
        # Try vector_namespace first (most specific)
        ns = attrs.get("vector_namespace")
        if ns is not None and str(ns).strip():
            return str(ns)
        
        # Try collection (vector-specific)
        ns = attrs.get("collection")
        if ns is not None and str(ns).strip():
            return str(ns)
        
        # Fall back to generic namespace
        ns = attrs.get("namespace")
        if ns is not None and str(ns).strip():
            return str(ns)
        
        return None

    # ---- query translation ----

    def build_query_spec(
        self,
        raw_query: Any,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
        stream: bool = False,
    ) -> VectorQuerySpec:
        namespace = self.preferred_namespace(op_ctx=op_ctx, framework_ctx=framework_ctx)

        # Case 1: raw_query is a vector (list/tuple of floats)
        if isinstance(raw_query, (list, tuple)):
            try:
                vector = [float(v) for v in raw_query]
                return VectorQuerySpec(
                    vector=vector,
                    top_k=10,  # Default
                    filters=None,
                    namespace=namespace,
                    include_metadata=True,
                    include_vectors=False,
                    stream=stream,
                )
            except (TypeError, ValueError):
                raise BadRequest(
                    "raw_query as list must contain numeric values for vector",
                    code="BAD_QUERY_VECTOR",
                )

        # Case 2: raw_query is a mapping with VectorQuerySpec fields
        if isinstance(raw_query, Mapping):
            vector = raw_query.get("vector")
            if not isinstance(vector, (list, tuple)):
                raise BadRequest(
                    "raw_query.vector must be a list/tuple of floats",
                    code="BAD_QUERY",
                )
            
            try:
                vector = [float(v) for v in vector]
            except (TypeError, ValueError):
                raise BadRequest(
                    "raw_query.vector must contain numeric values",
                    code="BAD_QUERY_VECTOR",
                )

            top_k = raw_query.get("top_k", 10)
            filters = raw_query.get("filters")
            rq_namespace = raw_query.get("namespace") or raw_query.get("collection")
            include_metadata = raw_query.get("include_metadata", True)
            include_vectors = raw_query.get("include_vectors", False)

            # Normalize filters
            if filters is not None:
                filters = self._filter_translator.normalize(filters)

            return VectorQuerySpec(
                vector=vector,
                top_k=int(top_k),
                filters=filters,
                namespace=str(rq_namespace) if rq_namespace is not None else namespace,
                include_metadata=bool(include_metadata),
                include_vectors=bool(include_vectors),
                stream=bool(raw_query.get("stream", stream)),
            )

        raise BadRequest(
            f"Unsupported raw_query type: {type(raw_query).__name__}",
            code="BAD_QUERY",
        )

    def translate_query_result(
        self,
        result: QueryResult,
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Any:
        # Generic behavior: return the matches list with metadata
        return {
            "matches": list(result.matches or []),
            "namespace": result.namespace,
            "usage": dict(result.usage or {}) if result.usage else None,
        }

    def translate_query_chunk(
        self,
        chunk: QueryChunk,
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Any:
        # Generic behavior: return the chunk as a simple dict
        return {
            "matches": list(chunk.matches or []),
            "is_final": bool(chunk.is_final),
            "usage": dict(chunk.usage or {}) if chunk.usage is not None else None,
        }

    # ---- mutation translation ----

    def build_upsert_spec(
        self,
        raw_documents: Any,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> VectorUpsertSpec:
        from corpus_sdk.vector.vector_base import Document  # local import to avoid cycles

        namespace = self.preferred_namespace(op_ctx=op_ctx, framework_ctx=framework_ctx)

        if isinstance(raw_documents, Mapping):
            raw_documents = [raw_documents]
        if not isinstance(raw_documents, Iterable):
            raise BadRequest(
                "raw_documents must be a mapping or iterable",
                code="BAD_DOCUMENTS",
            )

        documents: List[Document] = []
        for idx, item in enumerate(raw_documents):
            if isinstance(item, Document):
                documents.append(item)
                continue
            if not isinstance(item, Mapping):
                raise BadRequest(
                    f"raw_documents[{idx}] must be a mapping or Document",
                    code="BAD_DOCUMENTS",
                    details={"index": idx, "type": type(item).__name__},
                )
            
            # Extract vector
            vector = item.get("vector") or item.get("embedding")
            if not isinstance(vector, (list, tuple)):
                raise BadRequest(
                    f"raw_documents[{idx}] must have 'vector' or 'embedding' field",
                    code="BAD_DOCUMENT_VECTOR",
                    details={"index": idx},
                )
            
            try:
                vector = [float(v) for v in vector]
            except (TypeError, ValueError):
                raise BadRequest(
                    f"raw_documents[{idx}].vector must contain numeric values",
                    code="BAD_DOCUMENT_VECTOR",
                    details={"index": idx},
                )

            doc_ns = item.get("namespace") or item.get("collection") or namespace
            doc = Document(
                id=item.get("id"),
                vector=vector,
                metadata=item.get("metadata") or {},
                namespace=doc_ns,
            )
            documents.append(doc)

        return VectorUpsertSpec(documents=documents, namespace=namespace)

    def translate_upsert_result(
        self,
        result: UpsertResult,
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Any:
        return {
            "ids": list(result.ids or []),
            "count": len(result.ids or []),
        }

    def build_delete_spec(
        self,
        raw_filter_or_ids: Any,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> VectorDeleteSpec:
        namespace = self.preferred_namespace(op_ctx=op_ctx, framework_ctx=framework_ctx)

        ids: List[str] = []
        filter_expr: Optional[Mapping[str, Any]] = None

        # Mapping → filter-based delete
        if isinstance(raw_filter_or_ids, Mapping):
            filter_expr = self._filter_translator.normalize(raw_filter_or_ids)
        # Iterable of IDs or filters
        elif isinstance(raw_filter_or_ids, (list, tuple)):
            if not raw_filter_or_ids:
                ids = []
                filter_expr = None
            else:
                first = raw_filter_or_ids[0]
                # If first element is mapping, treat as list of filters AND'ed together
                if isinstance(first, Mapping):
                    filter_expr = self._filter_translator.normalize(raw_filter_or_ids)
                else:
                    ids = [str(x) for x in raw_filter_or_ids]
        else:
            # Single scalar ID
            ids = [str(raw_filter_or_ids)]

        return VectorDeleteSpec(ids=ids, namespace=namespace, filters=filter_expr)

    def translate_delete_result(
        self,
        result: DeleteResult,
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Any:
        return {
            "count": result.count or 0,
        }

    def build_update_spec(
        self,
        raw_updates: Any,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> VectorUpdateSpec:
        namespace = self.preferred_namespace(op_ctx=op_ctx, framework_ctx=framework_ctx)

        if isinstance(raw_updates, Mapping):
            raw_updates = [raw_updates]
        if not isinstance(raw_updates, Iterable):
            raise BadRequest(
                "raw_updates must be a mapping or iterable",
                code="BAD_UPDATES",
            )

        updates: List[Dict[str, Any]] = []
        for idx, item in enumerate(raw_updates):
            if not isinstance(item, Mapping):
                raise BadRequest(
                    f"raw_updates[{idx}] must be a mapping",
                    code="BAD_UPDATES",
                    details={"index": idx, "type": type(item).__name__},
                )
            
            update_dict: Dict[str, Any] = {
                "id": str(item.get("id")),
            }
            
            # Optional vector update
            if "vector" in item or "embedding" in item:
                vector = item.get("vector") or item.get("embedding")
                try:
                    update_dict["vector"] = [float(v) for v in vector]
                except (TypeError, ValueError):
                    raise BadRequest(
                        f"raw_updates[{idx}].vector must contain numeric values",
                        code="BAD_UPDATE_VECTOR",
                        details={"index": idx},
                    )
            
            # Optional metadata update
            if "metadata" in item:
                update_dict["metadata"] = dict(item["metadata"])
            
            updates.append(update_dict)

        return VectorUpdateSpec(updates=updates, namespace=namespace)

    def translate_update_result(
        self,
        result: UpdateResult,
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Any:
        return {
            "count": result.count or 0,
        }

    def translate_stats(
        self,
        stats: VectorStats,
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Any:
        return stats


# =============================================================================
# Vector Translator Orchestrator
# =============================================================================


class VectorTranslator:
    """
    Framework-agnostic orchestrator for vector operations.

    This class:
        - Accepts framework-level inputs and a normalized OperationContext
        - Delegates to a VectorFrameworkTranslator to build specs and translate results
        - Calls into a VectorProtocolV1 adapter to execute operations
        - Provides sync + async variants for all core operations
        - Handles streaming via SyncStreamBridge for sync callers
        - Applies MMR re-ranking before returning results (CRITICAL for vector quality)
        - Attaches rich error context for diagnostics

    Sync methods use AsyncBridge to call async adapters from sync contexts.

    It does *not*:
        - Know anything about framework-native context objects (RunnableConfig, etc.)
        - Implement any backend-specific logic (that lives in BaseVectorAdapter subclasses)
    """

    def __init__(
        self,
        *,
        adapter: VectorProtocolV1,
        framework: str = "generic",
        translator: Optional[VectorFrameworkTranslator] = None,
    ) -> None:
        self._adapter = adapter
        self._framework = framework
        self._translator: VectorFrameworkTranslator = translator or DefaultVectorFrameworkTranslator()

    # --------------------------------------------------------------------- #
    # Internal MMR helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        
        if na <= 0.0 or nb <= 0.0:
            return 0.0
        return float(dot / (na * nb))

    def _apply_mmr_to_query_result(
        self,
        result: QueryResult,
        mmr_config: Optional[MMRConfig],
    ) -> QueryResult:
        """
        Apply Maximal Marginal Relevance re-ranking to a QueryResult.

        MMR is CRITICAL for vector search quality - it reduces redundancy
        and balances relevance vs diversity in retrieved documents.

        MMR is only applied if:
            - mmr_config is not None and enabled is True
            - There are at least 2 matches
            - Each match contains:
                * a numeric score field (mmr_config.score_key), and
                * a vector field (mmr_config.vector_key) with equal dimensions
        """
        if mmr_config is None or not mmr_config.enabled:
            return result

        matches = list(result.matches or [])
        n = len(matches)
        if n <= 1:
            return result

        # Extract scores and vectors
        scores: List[float] = []
        vectors: List[List[float]] = []

        for idx, match in enumerate(matches):
            if not isinstance(match, Mapping):
                LOG.debug("MMR skipped: match[%d] is not a mapping", idx)
                return result
            if mmr_config.score_key not in match or mmr_config.vector_key not in match:
                LOG.debug(
                    "MMR skipped: match[%d] missing score or vector key (%s/%s)",
                    idx,
                    mmr_config.score_key,
                    mmr_config.vector_key,
                )
                return result
            try:
                scores.append(float(match[mmr_config.score_key]))
            except (TypeError, ValueError):
                LOG.debug("MMR skipped: match[%d] score is not numeric", idx)
                return result

            vec_raw = match[mmr_config.vector_key]
            if not isinstance(vec_raw, (list, tuple)):
                LOG.debug("MMR skipped: match[%d] vector is not a sequence", idx)
                return result
            try:
                vec = [float(v) for v in vec_raw]
            except (TypeError, ValueError):
                LOG.debug("MMR skipped: match[%d] vector elements not numeric", idx)
                return result
            vectors.append(vec)

        # Check dimension consistency
        dim = len(vectors[0])
        if dim == 0 or any(len(v) != dim for v in vectors):
            LOG.debug("MMR skipped: inconsistent or zero-length vector dimensions")
            return result

        # Normalize scores - handle both similarity and distance metrics
        if mmr_config.invert_score:
            # Lower is better (e.g., L2 distance) - invert for normalization
            max_score = max(scores)
            if max_score <= 0.0:
                normalized_scores = [1.0 for _ in scores]
            else:
                # Invert: high distance → low score
                normalized_scores = [(max_score - s) / max_score for s in scores]
        else:
            # Higher is better (e.g., cosine similarity) - normalize to [0, 1]
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score
            
            if score_range <= 0.0:
                # All scores identical
                normalized_scores = [1.0 for _ in scores]
            else:
                # Normalize to [0, 1] preserving relative differences
                normalized_scores = [(s - min_score) / score_range for s in scores]

        sim_fn: SimilarityFn = mmr_config.similarity_fn or self._cosine_similarity
        k = mmr_config.effective_k(n)
        lambda_mult = float(mmr_config.lambda_mult)

        selected_indices: List[int] = []
        candidates: List[int] = list(range(n))
        
        # Optimized similarity cache with canonical key ordering
        similarity_cache: Dict[Tuple[int, int], float] = {}

        def get_similarity(i: int, j: int) -> float:
            """Get similarity with efficient caching using canonical keys."""
            key = (min(i, j), max(i, j))
            if key not in similarity_cache:
                similarity_cache[key] = sim_fn(vectors[i], vectors[j])
            return similarity_cache[key]

        # Seed with most relevant by original score
        if candidates:
            best_first = max(candidates, key=lambda idx: normalized_scores[idx])
            selected_indices.append(best_first)
            candidates.remove(best_first)

        # Greedy MMR selection
        while candidates and len(selected_indices) < k:
            best_idx: Optional[int] = None
            best_mmr_score = -float("inf")

            for idx in candidates:
                relevance = normalized_scores[idx]
                diversity_penalty = 0.0
                if selected_indices:
                    diversity_penalty = max(
                        get_similarity(idx, j) for j in selected_indices
                    )
                mmr_score = lambda_mult * relevance - (1.0 - lambda_mult) * diversity_penalty
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = idx

            if best_idx is None:
                break

            selected_indices.append(best_idx)
            candidates.remove(best_idx)

        # Reorder: selected items first, then remaining in original order
        reordered: List[Record] = [matches[i] for i in selected_indices]
        if k < n:
            remaining = [matches[i] for i in range(n) if i not in selected_indices]
            reordered.extend(remaining)

        LOG.info(
            "MMR applied: %d matches → %d selected with lambda=%.2f (invert_score=%s)",
            n,
            len(selected_indices),
            lambda_mult,
            mmr_config.invert_score,
        )

        return QueryResult(
            matches=reordered,
            namespace=result.namespace,
            usage=result.usage,
        )

    # --------------------------------------------------------------------- #
    # Sync Query APIs (use AsyncBridge)
    # --------------------------------------------------------------------- #

    def query(
        self,
        raw_query: Any,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
        mmr_config: Optional[MMRConfig] = None,
    ) -> Any:
        """
        Synchronous query API.

        Uses AsyncBridge to call the async adapter from a sync context.

        Steps:
            - Normalize OperationContext
            - Build VectorQuerySpec via translator
            - Call adapter.query(...) via AsyncBridge
            - Apply MMR (CRITICAL for vector quality)
            - Translate result back to framework-level shape
        """
        async def _query_coro():
            ctx = _ensure_operation_context(op_ctx)
            try:
                spec = self._translator.build_query_spec(
                    raw_query,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                    stream=False,
                )
                result = await self._adapter.query(spec, ctx=ctx)
                
                if not isinstance(result, QueryResult):
                    raise BadRequest(
                        f"adapter.query returned unsupported type: {type(result).__name__}",
                        code="BAD_ADAPTER_RESULT",
                    )

                # Apply MMR - CRITICAL for vector search quality
                result_mmr = self._apply_mmr_to_query_result(result, mmr_config=mmr_config)
                return self._translator.translate_query_result(
                    result_mmr,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )
            except Exception as exc:
                # Attach error context to all exceptions
                attach_context(
                    exc,
                    framework=self._framework,
                    vector_operation="query",
                    request_id=ctx.request_id,
                    tenant=ctx.tenant,
                )
                raise

        # Use AsyncBridge with deadline from context
        ctx = _ensure_operation_context(op_ctx)
        timeout = ctx.deadline_ms / 1000.0 if ctx.deadline_ms else None
        return AsyncBridge.run_async(_query_coro(), timeout=timeout)

    # --------------------------------------------------------------------- #
    # Async Query APIs
    # --------------------------------------------------------------------- #

    async def arun_query(
        self,
        raw_query: Any,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
        mmr_config: Optional[MMRConfig] = None,
    ) -> Any:
        """
        Async query API (preferred for async applications).

        Fully async:
            - await adapter.query(...)
            - MMR re-ranking
            - result translation
        """
        ctx = _ensure_operation_context(op_ctx)
        try:
            spec = self._translator.build_query_spec(
                raw_query,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
                stream=False,
            )
            result = await self._adapter.query(spec, ctx=ctx)
            
            if not isinstance(result, QueryResult):
                raise BadRequest(
                    f"adapter.query returned unsupported type: {type(result).__name__}",
                    code="BAD_ADAPTER_RESULT",
                )

            result_mmr = self._apply_mmr_to_query_result(result, mmr_config=mmr_config)
            return self._translator.translate_query_result(
                result_mmr,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
        except Exception as exc:
            attach_context(
                exc,
                framework=self._framework,
                vector_operation="query",
                request_id=ctx.request_id,
                tenant=ctx.tenant,
            )
            raise

    # --------------------------------------------------------------------- #
    # Streaming Query APIs
    # --------------------------------------------------------------------- #

    def query_stream(
        self,
        raw_query: Any,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Iterator[Any]:
        """
        Synchronous streaming query API.

        Exposes a sync iterator that yields framework-level chunks by
        bridging the async adapter.stream_query(...) via SyncStreamBridge.

        Note: MMR cannot be applied to streaming results since we need
        the full result set for diversity calculation. Use non-streaming
        query() with MMR for best quality.
        """
        ctx = _ensure_operation_context(op_ctx)

        async def _stream_factory() -> AsyncIterator[Any]:
            """Factory that creates the async stream."""
            spec = self._translator.build_query_spec(
                raw_query,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
                stream=True,
            )
            try:
                async for chunk in self._adapter.stream_query(spec, ctx=ctx):
                    yield self._translator.translate_query_chunk(
                        chunk,
                        op_ctx=ctx,
                        framework_ctx=framework_ctx,
                    )
            except Exception as exc:
                attach_context(
                    exc,
                    framework=self._framework,
                    vector_operation="stream_query",
                    request_id=ctx.request_id,
                    tenant=ctx.tenant,
                )
                raise

        # Use SyncStreamBridge for sync streaming
        bridge = SyncStreamBridge(
            coro_factory=_stream_factory,
            framework=self._framework,
            error_context={
                "operation": "vector.query_stream",
                "request_id": ctx.request_id,
                "tenant": ctx.tenant,
            },
        )
        return bridge.run()

    async def arun_query_stream(
        self,
        raw_query: Any,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> AsyncIterator[Any]:
        """
        Async streaming query API.

        Returns an async iterator yielding framework-level chunks.

        Note: MMR cannot be applied to streaming results.
        """
        ctx = _ensure_operation_context(op_ctx)
        spec = self._translator.build_query_spec(
            raw_query,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
            stream=True,
        )

        try:
            async for chunk in self._adapter.stream_query(spec, ctx=ctx):
                yield self._translator.translate_query_chunk(
                    chunk,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )
        except Exception as exc:
            attach_context(
                exc,
                framework=self._framework,
                vector_operation="stream_query",
                request_id=ctx.request_id,
                tenant=ctx.tenant,
            )
            raise

    # --------------------------------------------------------------------- #
    # Sync mutation APIs (use AsyncBridge)
    # --------------------------------------------------------------------- #

    def upsert(
        self,
        raw_documents: Any,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """Synchronous upsert (uses AsyncBridge)."""
        async def _upsert_coro():
            ctx = _ensure_operation_context(op_ctx)
            try:
                spec = self._translator.build_upsert_spec(
                    raw_documents,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )
                result = await self._adapter.upsert(spec, ctx=ctx)
                
                if not isinstance(result, UpsertResult):
                    raise BadRequest(
                        f"adapter.upsert returned unsupported type: {type(result).__name__}",
                        code="BAD_ADAPTER_RESULT",
                    )
                
                return self._translator.translate_upsert_result(
                    result,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )
            except Exception as exc:
                attach_context(
                    exc,
                    framework=self._framework,
                    vector_operation="upsert",
                    request_id=ctx.request_id,
                    tenant=ctx.tenant,
                )
                raise

        ctx = _ensure_operation_context(op_ctx)
        timeout = ctx.deadline_ms / 1000.0 if ctx.deadline_ms else None
        return AsyncBridge.run_async(_upsert_coro(), timeout=timeout)

    async def arun_upsert(
        self,
        raw_documents: Any,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """Async upsert."""
        ctx = _ensure_operation_context(op_ctx)
        try:
            spec = self._translator.build_upsert_spec(
                raw_documents,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            result = await self._adapter.upsert(spec, ctx=ctx)
            
            if not isinstance(result, UpsertResult):
                raise BadRequest(
                    f"adapter.upsert returned unsupported type: {type(result).__name__}",
                    code="BAD_ADAPTER_RESULT",
                )
            
            return self._translator.translate_upsert_result(
                result,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
        except Exception as exc:
            attach_context(
                exc,
                framework=self._framework,
                vector_operation="upsert",
                request_id=ctx.request_id,
                tenant=ctx.tenant,
            )
            raise

    def delete(
        self,
        raw_filter_or_ids: Any,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """Synchronous delete (uses AsyncBridge)."""
        async def _delete_coro():
            ctx = _ensure_operation_context(op_ctx)
            try:
                spec = self._translator.build_delete_spec(
                    raw_filter_or_ids,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )
                result = await self._adapter.delete(spec, ctx=ctx)
                
                if not isinstance(result, DeleteResult):
                    raise BadRequest(
                        f"adapter.delete returned unsupported type: {type(result).__name__}",
                        code="BAD_ADAPTER_RESULT",
                    )
                
                return self._translator.translate_delete_result(
                    result,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )
            except Exception as exc:
                attach_context(
                    exc,
                    framework=self._framework,
                    vector_operation="delete",
                    request_id=ctx.request_id,
                    tenant=ctx.tenant,
                )
                raise

        ctx = _ensure_operation_context(op_ctx)
        timeout = ctx.deadline_ms / 1000.0 if ctx.deadline_ms else None
        return AsyncBridge.run_async(_delete_coro(), timeout=timeout)

    async def arun_delete(
        self,
        raw_filter_or_ids: Any,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """Async delete."""
        ctx = _ensure_operation_context(op_ctx)
        try:
            spec = self._translator.build_delete_spec(
                raw_filter_or_ids,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            result = await self._adapter.delete(spec, ctx=ctx)
            
            if not isinstance(result, DeleteResult):
                raise BadRequest(
                    f"adapter.delete returned unsupported type: {type(result).__name__}",
                    code="BAD_ADAPTER_RESULT",
                )
            
            return self._translator.translate_delete_result(
                result,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
        except Exception as exc:
            attach_context(
                exc,
                framework=self._framework,
                vector_operation="delete",
                request_id=ctx.request_id,
                tenant=ctx.tenant,
            )
            raise

    def update(
        self,
        raw_updates: Any,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """Synchronous update (uses AsyncBridge)."""
        async def _update_coro():
            ctx = _ensure_operation_context(op_ctx)
            try:
                spec = self._translator.build_update_spec(
                    raw_updates,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )
                result = await self._adapter.update(spec, ctx=ctx)
                
                if not isinstance(result, UpdateResult):
                    raise BadRequest(
                        f"adapter.update returned unsupported type: {type(result).__name__}",
                        code="BAD_ADAPTER_RESULT",
                    )
                
                return self._translator.translate_update_result(
                    result,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )
            except Exception as exc:
                attach_context(
                    exc,
                    framework=self._framework,
                    vector_operation="update",
                    request_id=ctx.request_id,
                    tenant=ctx.tenant,
                )
                raise

        ctx = _ensure_operation_context(op_ctx)
        timeout = ctx.deadline_ms / 1000.0 if ctx.deadline_ms else None
        return AsyncBridge.run_async(_update_coro(), timeout=timeout)

    async def arun_update(
        self,
        raw_updates: Any,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """Async update."""
        ctx = _ensure_operation_context(op_ctx)
        try:
            spec = self._translator.build_update_spec(
                raw_updates,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            result = await self._adapter.update(spec, ctx=ctx)
            
            if not isinstance(result, UpdateResult):
                raise BadRequest(
                    f"adapter.update returned unsupported type: {type(result).__name__}",
                    code="BAD_ADAPTER_RESULT",
                )
            
            return self._translator.translate_update_result(
                result,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
        except Exception as exc:
            attach_context(
                exc,
                framework=self._framework,
                vector_operation="update",
                request_id=ctx.request_id,
                tenant=ctx.tenant,
            )
            raise

    # --------------------------------------------------------------------- #
    # Stats / Inspection
    # --------------------------------------------------------------------- #

    def get_stats(
        self,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """Synchronous get_stats (uses AsyncBridge)."""
        async def _stats_coro():
            ctx = _ensure_operation_context(op_ctx)
            try:
                stats = await self._adapter.get_stats(ctx=ctx)
                
                if not isinstance(stats, VectorStats):
                    raise BadRequest(
                        f"adapter.get_stats returned unsupported type: {type(stats).__name__}",
                        code="BAD_ADAPTER_RESULT",
                    )

                return self._translator.translate_stats(
                    stats,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )
            except Exception as exc:
                attach_context(
                    exc,
                    framework=self._framework,
                    vector_operation="get_stats",
                    request_id=ctx.request_id,
                    tenant=ctx.tenant,
                )
                raise

        ctx = _ensure_operation_context(op_ctx)
        timeout = ctx.deadline_ms / 1000.0 if ctx.deadline_ms else None
        return AsyncBridge.run_async(_stats_coro(), timeout=timeout)

    async def arun_get_stats(
        self,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """Async get_stats."""
        ctx = _ensure_operation_context(op_ctx)
        try:
            stats = await self._adapter.get_stats(ctx=ctx)
            
            if not isinstance(stats, VectorStats):
                raise BadRequest(
                    f"adapter.get_stats returned unsupported type: {type(stats).__name__}",
                    code="BAD_ADAPTER_RESULT",
                )

            return self._translator.translate_stats(
                stats,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
        except Exception as exc:
            attach_context(
                exc,
                framework=self._framework,
                vector_operation="get_stats",
                request_id=ctx.request_id,
                tenant=ctx.tenant,
            )
            raise


# =============================================================================
# Registry for per-framework translators
# =============================================================================


_TranslatorFactory = Callable[[VectorProtocolV1], VectorFrameworkTranslator]
_VECTOR_TRANSLATOR_FACTORIES: Dict[str, _TranslatorFactory] = {}


def register_vector_translator(
    framework: str,
    factory: _TranslatorFactory,
) -> None:
    """
    Register or override a VectorFrameworkTranslator factory for a given framework.

    Example
    -------
        def make_langchain_translator(adapter: VectorProtocolV1) -> VectorFrameworkTranslator:
            return LangChainVectorTranslator(adapter=adapter)

        register_vector_translator("langchain", make_langchain_translator)
    """
    if not framework or not isinstance(framework, str):
        raise BadRequest(
            "framework name must be a non-empty string",
            code="BAD_TRANSLATOR_REGISTRATION",
        )
    if not callable(factory):
        raise BadRequest(
            "translator factory must be callable",
            code="BAD_TRANSLATOR_REGISTRATION",
        )
    _VECTOR_TRANSLATOR_FACTORIES[framework] = factory
    LOG.debug("Registered vector translator factory for framework=%s", framework)


def get_vector_translator_factory(framework: str) -> Optional[_TranslatorFactory]:
    """Return a previously registered translator factory for a framework, if any."""
    return _VECTOR_TRANSLATOR_FACTORIES.get(framework)


def create_vector_translator(
    *,
    adapter: VectorProtocolV1,
    framework: str = "generic",
    translator: Optional[VectorFrameworkTranslator] = None,
) -> VectorTranslator:
    """
    Convenience helper to construct a VectorTranslator for a given framework.

    Behavior:
        - If `translator` is provided explicitly, it is used as-is.
        - Else, if a factory is registered for `framework`, it is used.
        - Else, DefaultVectorFrameworkTranslator is used.
    """
    if translator is None:
        factory = get_vector_translator_factory(framework)
        if factory is not None:
            translator = factory(adapter)
        else:
            translator = DefaultVectorFrameworkTranslator()
    return VectorTranslator(adapter=adapter, framework=framework, translator=translator)


__all__ = [
    "MMRConfig",
    "SimilarityFn",
    "FilterTranslator",
    "VectorFrameworkTranslator",
    "DefaultVectorFrameworkTranslator",
    "VectorTranslator",
    "register_vector_translator",
    "get_vector_translator_factory",
    "create_vector_translator",
]
