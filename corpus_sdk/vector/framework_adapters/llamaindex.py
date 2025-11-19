# corpus_sdk/vector/framework_adapters/llamaindex.py
# SPDX-License-Identifier: Apache-2.0

"""
LlamaIndex adapter for Corpus Vector protocol.

This module exposes Corpus `BaseVectorAdapter` implementations as
`llama_index.core.vector_stores.types.BasePydanticVectorStore` instances, with:

- Sync + async add/query/delete APIs (matching and extending LlamaIndex expectations)
- Proper integration with Corpus VectorProtocolV1
- Namespace + metadata filter handling (capability-aware)
- Batch upserts and deletes that respect backend limits
- Optional client-side score thresholding
- Full OperationContext propagation via `corpus_sdk.core.context_translation.from_llamaindex`
- Rich error context via `corpus_sdk.core.error_context.attach_context`
- Optional streaming query support via SyncStreamBridge (`sync_stream`)
- Optional Maximal Marginal Relevance (MMR) query variants

Design philosophy
-----------------
- Protocol-first: LlamaIndex is a thin skin over Corpus vector adapters.
- All heavy lifting (backpressure, deadlines, breakers, etc.) lives in
  the underlying `BaseVectorAdapter`, not here.
- This layer focuses on:
    * Translating LlamaIndex Nodes ↔ Corpus Vector objects
    * Respecting VectorCapabilities (namespaces, filters, batch sizes)
    * Bridging async Corpus APIs into LlamaIndex's sync interface via AsyncBridge
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
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

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
    UpsertResult,
    DeleteResult,
    QuerySpec,
    UpsertSpec,
    DeleteSpec,
    OperationContext,
    VectorCapabilities,
    # Errors
    BadRequest,
    NotSupported,
    VectorAdapterError,
)

from corpus_sdk.llm.framework_adapters.common.async_bridge import AsyncBridge
from corpus_sdk.core.context_translation import from_llamaindex as context_from_llamaindex
from corpus_sdk.core.error_context import attach_context
from corpus_sdk.core.sync_stream_bridge import sync_stream

logger = logging.getLogger(__name__)

Metadata = Dict[str, Any]


class CorpusLlamaIndexVectorStore(BasePydanticVectorStore):
    """
    LlamaIndex `BasePydanticVectorStore` implementation backed by a Corpus
    `BaseVectorAdapter` (VectorProtocolV1).

    This class is a thin integration layer:
    - Nodes are mapped to Corpus VectorProtocol `Vector` objects.
    - VectorStoreQuery calls map to adapter `query()` calls.
    - Namespaces + metadata filters are honored based on VectorCapabilities.
    - Sync APIs are implemented via `AsyncBridge` on top of async Corpus methods.
    - Async APIs are exposed for advanced callers that want to stay async end-to-end.
    - Optional MMR variants leverage database scores for relevance + cosine diversity.
    """

    # LlamaIndex VectorStore flags
    stores_text: bool = True
    flat_metadata: bool = True

    # Corpus integration fields
    corpus_adapter: BaseVectorAdapter
    namespace: Optional[str] = "default"
    batch_size: int = 100
    default_top_k: int = 4
    score_threshold: Optional[float] = None

    # Reserved metadata keys for internal mapping
    id_field: str = "id"
    text_field: str = "text"
    node_id_field: str = "node_id"
    ref_doc_id_field: str = "ref_doc_id"

    # Cached capabilities (lazy-loaded)
    _caps: Optional[VectorCapabilities] = None

    # Pydantic v2-style config
    model_config = {"arbitrary_types_allowed": True}

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

        This is intentionally minimal and protocol-agnostic.
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
        Synchronously fetch and cache VectorCapabilities.

        Uses AsyncBridge to call the async adapter.
        """
        if self._caps is not None:
            return self._caps
        try:
            caps = AsyncBridge.run_async(self.corpus_adapter.capabilities())
            self._caps = caps
            return caps
        except Exception as exc:  # noqa: BLE001
            attach_context(exc, framework="llamaindex", operation="capabilities")
            raise

    async def _get_caps_async(self) -> VectorCapabilities:
        """Async capability fetch with caching."""
        if self._caps is not None:
            return self._caps
        try:
            caps = await self.corpus_adapter.capabilities()
            self._caps = caps
            return caps
        except Exception as exc:  # noqa: BLE001
            attach_context(exc, framework="llamaindex", operation="capabilities")
            raise

    def _build_ctx(self, **kwargs: Any) -> Optional[OperationContext]:
        """
        Build an OperationContext from LlamaIndex-style kwargs.

        Priority:
        1. Explicit OperationContext via `ctx` or `operation_context`.
        2. LlamaIndex CallbackManager via `callback_manager` using
           `context_from_llamaindex`.
        3. None (no context).
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
        still passed down, but the adapter may ignore it.
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
          or `.embedding`.
        - Metadata is flattened via `node_to_metadata_dict`.
        """
        vectors: List[Vector] = []
        ns = self._effective_namespace(namespace)

        for node in nodes:
            # Extract embedding
            embedding = node.get_embedding() if hasattr(node, "get_embedding") else None
            if embedding is None:
                embedding = getattr(node, "embedding", None)
            if embedding is None:
                raise BadRequest(
                    f"Node {getattr(node, 'node_id', None) or node} has no embedding; "
                    "ensure embeddings are set before calling add().",
                    code="NO_EMBEDDING",
                )

            # Flatten metadata
            metadata = node_to_metadata_dict(
                node,
                remove_text=False,
                mode=MetadataMode.ALL,
            )

            # Ensure reserved keys are populated
            node_id = getattr(node, "node_id", None) or getattr(node, "id_", None)
            ref_doc_id = getattr(node, "ref_doc_id", None)

            metadata = dict(metadata or {})
            metadata[self.node_id_field] = node_id
            if ref_doc_id is not None:
                metadata[self.ref_doc_id_field] = ref_doc_id

            # Text is stored in metadata; docstore handled by LlamaIndex
            text = node.get_content(metadata_mode=MetadataMode.NONE) or ""

            metadata[self.text_field] = text
            metadata[self.id_field] = node_id

            vectors.append(
                Vector(
                    id=str(node_id),
                    vector=[float(x) for x in embedding],
                    metadata=metadata,
                    namespace=ns,
                    text=None,
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
        """
        results: List[NodeWithScore] = []

        for m in matches:
            v = m.vector
            meta = dict(v.metadata or {})

            text = meta.pop(self.text_field, None)
            node_id = meta.pop(self.node_id_field, v.id)
            ref_doc_id = meta.get(self.ref_doc_id_field)

            # Reconstruct node from metadata; fall back to TextNode if needed.
            node: BaseNode
            try:
                node = metadata_dict_to_node(meta, text=text, node_id=node_id)
            except Exception:
                try:
                    node = legacy_metadata_dict_to_node(meta, text=text, node_id=node_id)
                except Exception:
                    node = TextNode(
                        text=text or "",
                        id_=str(node_id),
                        metadata=meta,
                    )

            if ref_doc_id is not None:
                try:
                    node.ref_doc_id = ref_doc_id  # type: ignore[attr-defined]
                except Exception:
                    # If the node type does not support ref_doc_id, ignore silently.
                    pass

            results.append(
                NodeWithScore(
                    node=node,
                    score=float(m.score),
                )
            )

        return results

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

        Improvements over a naive implementation:
        - Supports boolean condition on filters (AND vs OR) via `filters.condition`.
        - Supports EQ / IN and range-style operators (GT / GTE / LT / LTE / NE).
        - Emits an AST-style filter with `$and` / `$or` when multiple predicates
          are present, which most Corpus adapters can interpret.

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

        Unknown operators are logged and skipped rather than causing errors.
        """
        clauses: List[Dict[str, Any]] = []

        # User-specified metadata filters
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

        # Determine boolean condition if MetadataFilters exposes it.
        condition = getattr(filters, "condition", None) if filters is not None else None
        cond_name = str(condition).upper() if condition is not None else "AND"

        if len(clauses) == 1:
            # Single predicate; no need for $and/$or wrapping.
            return clauses[0]

        if cond_name in ("OR", "ANY"):
            return {"$or": clauses}

        # Default: AND semantics
        return {"$and": clauses}

    # ------------------------------------------------------------------ #
    # Core async Corpus operations
    # ------------------------------------------------------------------ #

    async def _aupsert_vectors(
        self,
        vectors: List[Vector],
        *,
        namespace: Optional[str],
        ctx: Optional[OperationContext],
    ) -> UpsertResult:
        """
        Async upsert of a batch of Corpus Vector objects.

        Respects backend max_batch_size if reported in capabilities.
        Aggregates per-batch UpsertResult into a single result.
        """
        caps = await self._get_caps_async()
        max_batch = caps.max_batch_size or self.batch_size or len(vectors)
        effective_batch_size = max(1, min(max_batch, self.batch_size or max_batch))

        upserted_total = 0
        failures_total: List[Dict[str, Any]] = []

        ns = self._effective_namespace(namespace)

        for i in range(0, len(vectors), effective_batch_size):
            batch = vectors[i : i + effective_batch_size]
            try:
                spec = UpsertSpec(namespace=ns, vectors=batch)
                result = await self.corpus_adapter.upsert(spec, ctx=ctx)
                upserted_total += int(result.upserted_count or 0)
                if result.failures:
                    failures_total.extend(list(result.failures))
            except Exception as exc:  # noqa: BLE001
                attach_context(
                    exc,
                    framework="llamaindex",
                    operation="upsert",
                    namespace=ns,
                    batch_index=i // effective_batch_size,
                    batch_size=len(batch),
                )
                raise

        return UpsertResult(
            upserted_count=upserted_total,
            failed_count=len(failures_total),
            failures=failures_total,
        )

    async def _aquery_embedding(
        self,
        embedding: Sequence[float],
        *,
        top_k: int,
        namespace: Optional[str],
        filter: Optional[Mapping[str, Any]],
        include_vectors: bool,
        ctx: Optional[OperationContext],
    ) -> List[VectorMatch]:
        """
        Async similarity query for a single embedding.
        """
        caps = await self._get_caps_async()
        ns = self._effective_namespace(namespace)

        if caps.max_top_k is not None and top_k > caps.max_top_k:
            raise BadRequest(
                f"top_k {top_k} exceeds maximum of {caps.max_top_k}",
                code="BAD_TOP_K",
                details={"max_top_k": caps.max_top_k, "namespace": ns},
            )

        if filter and not caps.supports_metadata_filtering:
            raise NotSupported(
                "metadata filtering is not supported by the underlying vector adapter",
                code="FILTER_NOT_SUPPORTED",
                details={"namespace": ns},
            )

        spec = QuerySpec(
            vector=[float(x) for x in embedding],
            top_k=top_k,
            namespace=ns,
            filter=dict(filter) if filter else None,
            include_metadata=True,
            include_vectors=include_vectors,
        )

        try:
            result: QueryResult = await self.corpus_adapter.query(spec, ctx=ctx)
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="llamaindex",
                operation="query",
                namespace=ns,
                top_k=top_k,
            )
            raise

        matches = list(result.matches or [])

        # Apply optional client-side score threshold
        if self.score_threshold is not None:
            matches = [m for m in matches if float(m.score) >= float(self.score_threshold)]

        return matches

    async def _adelete_vectors(
        self,
        *,
        ids: Optional[List[str]],
        namespace: Optional[str],
        filter: Optional[Mapping[str, Any]],
        ctx: Optional[OperationContext],
    ) -> DeleteResult:
        """
        Async delete by IDs or filter.
        """
        caps = await self._get_caps_async()
        ns = self._effective_namespace(namespace)

        if filter and not caps.supports_metadata_filtering:
            raise NotSupported(
                "delete by metadata filter is not supported by the underlying vector adapter",
                code="FILTER_NOT_SUPPORTED",
                details={"namespace": ns},
            )

        if not ids and not filter:
            raise BadRequest(
                "must provide ids or filter for delete",
                code="BAD_DELETE",
            )

        spec = DeleteSpec(
            namespace=ns,
            ids=[str(i) for i in ids] if ids else None,
            filter=dict(filter) if filter else None,
        )

        try:
            result: DeleteResult = await self.corpus_adapter.delete(spec, ctx=ctx)
            return result
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="llamaindex",
                operation="delete",
                namespace=ns,
                ids_count=len(ids or []),
            )
            raise

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
            query_vec: The query embedding vector
            candidate_matches: Candidate matches with original scores and vectors
            k: Number of results to select
            lambda_mult: MMR lambda parameter (0-1), higher values favor relevance

        Returns:
            Indices into candidate_matches for selected results
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
        similarity_cache: Dict[tuple[int, int], float] = {}

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
        - Expects nodes to already have embeddings.
        - Flattens node metadata and stores text alongside vectors.
        - Respects backend batch size limits when performing upserts.
        """
        if not nodes:
            return []

        namespace: Optional[str] = kwargs.get("namespace")
        ctx = self._build_ctx(**kwargs)

        vectors = self._nodes_to_corpus_vectors(nodes, namespace=namespace)
        ids = [str(getattr(n, "node_id", None) or getattr(n, "id_", None)) for n in nodes]

        try:
            AsyncBridge.run_async(
                self._aupsert_vectors(
                    vectors,
                    namespace=namespace,
                    ctx=ctx,
                )
            )
        except Exception:
            raise

        return ids

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
        ids = [str(getattr(n, "node_id", None) or getattr(n, "id_", None)) for n in nodes]

        await self._aupsert_vectors(
            vectors,
            namespace=namespace,
            ctx=ctx,
        )
        return ids

    def delete(
        self,
        ref_doc_id: str,
        **kwargs: Any,
    ) -> None:
        """
        Delete vectors associated with a given ref_doc_id (sync).

        LlamaIndex's VectorStore contract passes `ref_doc_id` here. We translate
        that into a metadata filter on `self.ref_doc_id_field`.
        """
        namespace: Optional[str] = kwargs.get("namespace")
        ctx = self._build_ctx(**kwargs)

        filter_dict: Dict[str, Any] = {self.ref_doc_id_field: ref_doc_id}
        AsyncBridge.run_async(
            self._adelete_vectors(
                ids=None,
                namespace=namespace,
                filter=filter_dict,
                ctx=ctx,
            )
        )

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
        await self._adelete_vectors(
            ids=None,
            namespace=namespace,
            filter=filter_dict,
            ctx=ctx,
        )

    def delete_nodes(
        self,
        node_ids: Sequence[str],
        **kwargs: Any,
    ) -> None:
        """
        Delete vectors by node IDs (sync).

        This is a convenience method some LlamaIndex components use. It maps
        node IDs directly to vector IDs.
        """
        if not node_ids:
            return

        namespace: Optional[str] = kwargs.get("namespace")
        ctx = self._build_ctx(**kwargs)

        AsyncBridge.run_async(
            self._adelete_vectors(
                ids=[str(i) for i in node_ids],
                namespace=namespace,
                filter=None,
                ctx=ctx,
            )
        )

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

        await self._adelete_vectors(
            ids=[str(i) for i in node_ids],
            namespace=namespace,
            filter=None,
            ctx=ctx,
        )

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """
        Perform similarity query and return a VectorStoreQueryResult (sync).

        Assumes `query.query_embedding` has already been computed by LlamaIndex.
        """
        if query.query_embedding is None:
            raise NotSupported(
                "VectorStoreQuery.query_embedding is None; LlamaIndex must "
                "provide a query embedding before calling CorpusLlamaIndexVectorStore.query.",
                code="NO_QUERY_EMBEDDING",
            )

        namespace: Optional[str] = kwargs.get("namespace")
        ctx = self._build_ctx(**kwargs)

        top_k = query.similarity_top_k or self.default_top_k
        corpus_filter = self._metadata_filters_to_corpus_filter(
            query.filters,
            doc_ids=query.doc_ids,
            node_ids=query.node_ids,
        )

        embedding = [float(x) for x in query.query_embedding]

        try:
            matches = AsyncBridge.run_async(
                self._aquery_embedding(
                    embedding,
                    top_k=top_k,
                    namespace=namespace,
                    filter=corpus_filter,
                    include_vectors=False,
                    ctx=ctx,
                )
            )
        except Exception:
            raise

        nodes_with_scores = self._matches_to_nodes(matches)
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
        """
        if query.query_embedding is None:
            raise NotSupported(
                "VectorStoreQuery.query_embedding is None; LlamaIndex must "
                "provide a query embedding before calling aquery.",
                code="NO_QUERY_EMBEDDING",
            )

        namespace: Optional[str] = kwargs.get("namespace")
        ctx = self._build_ctx(**kwargs)

        top_k = query.similarity_top_k or self.default_top_k
        corpus_filter = self._metadata_filters_to_corpus_filter(
            query.filters,
            doc_ids=query.doc_ids,
            node_ids=query.node_ids,
        )

        embedding = [float(x) for x in query.query_embedding]

        matches = await self._aquery_embedding(
            embedding,
            top_k=top_k,
            namespace=namespace,
            filter=corpus_filter,
            include_vectors=False,
            ctx=ctx,
        )

        nodes_with_scores = self._matches_to_nodes(matches)
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

        This uses SyncStreamBridge under the hood via `sync_stream` to bridge
        the async query into a synchronous iterator. The backend query itself
        is still a single async call; this just exposes results incrementally.
        """
        if query.query_embedding is None:
            raise NotSupported(
                "VectorStoreQuery.query_embedding is None; LlamaIndex must "
                "provide a query embedding before calling query_stream.",
                code="NO_QUERY_EMBEDDING",
            )

        namespace: Optional[str] = kwargs.get("namespace")
        ctx = self._build_ctx(**kwargs)

        top_k = query.similarity_top_k or self.default_top_k
        corpus_filter = self._metadata_filters_to_corpus_filter(
            query.filters,
            doc_ids=query.doc_ids,
            node_ids=query.node_ids,
        )

        embedding = [float(x) for x in query.query_embedding]

        async def _stream_coro():
            matches = await self._aquery_embedding(
                embedding,
                top_k=top_k,
                namespace=namespace,
                filter=corpus_filter,
                include_vectors=False,
                ctx=ctx,
            )
            for match in matches:
                yield match

        for match in sync_stream(
            _stream_coro,
            framework="llamaindex",
            error_context={
                "operation": "vector_query_stream",
                "namespace": namespace,
                "top_k": top_k,
            },
        ):
            nodes_with_scores = self._matches_to_nodes([match])
            if nodes_with_scores:
                yield nodes_with_scores[0]

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
        original database scores.

        Args:
            query: VectorStoreQuery with a precomputed query_embedding
            lambda_mult: MMR lambda parameter (0-1), higher values favor relevance
            fetch_k: Number of candidates to fetch for MMR selection; if None,
                     defaults to `max(k * 4, k + 5)` where k is similarity_top_k
            **kwargs: Additional arguments, e.g. namespace, ctx, callback_manager

        Returns:
            VectorStoreQueryResult with MMR-selected nodes.
        """
        if query.query_embedding is None:
            raise NotSupported(
                "VectorStoreQuery.query_embedding is None; LlamaIndex must "
                "provide a query embedding before calling query_mmr.",
                code="NO_QUERY_EMBEDDING",
            )

        namespace: Optional[str] = kwargs.get("namespace")
        ctx = self._build_ctx(**kwargs)

        k = query.similarity_top_k or self.default_top_k
        effective_fetch_k = fetch_k or max(k * 4, k + 5)

        corpus_filter = self._metadata_filters_to_corpus_filter(
            query.filters,
            doc_ids=query.doc_ids,
            node_ids=query.node_ids,
        )

        embedding = [float(x) for x in query.query_embedding]

        try:
            matches = AsyncBridge.run_async(
                self._aquery_embedding(
                    embedding,
                    top_k=effective_fetch_k,
                    namespace=namespace,
                    filter=corpus_filter,
                    include_vectors=True,
                    ctx=ctx,
                )
            )
        except Exception:
            raise

        if not matches:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        indices = self._mmr_select_indices(
            query_vec=embedding,
            candidate_matches=matches,
            k=k,
            lambda_mult=lambda_mult,
        )

        selected_matches = [matches[i] for i in indices]
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
        """
        if query.query_embedding is None:
            raise NotSupported(
                "VectorStoreQuery.query_embedding is None; LlamaIndex must "
                "provide a query embedding before calling aquery_mmr.",
                code="NO_QUERY_EMBEDDING",
            )

        namespace: Optional[str] = kwargs.get("namespace")
        ctx = self._build_ctx(**kwargs)

        k = query.similarity_top_k or self.default_top_k
        effective_fetch_k = fetch_k or max(k * 4, k + 5)

        corpus_filter = self._metadata_filters_to_corpus_filter(
            query.filters,
            doc_ids=query.doc_ids,
            node_ids=query.node_ids,
        )

        embedding = [float(x) for x in query.query_embedding]

        matches = await self._aquery_embedding(
            embedding,
            top_k=effective_fetch_k,
            namespace=namespace,
            filter=corpus_filter,
            include_vectors=True,
            ctx=ctx,
        )

        if not matches:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        indices = self._mmr_select_indices(
            query_vec=embedding,
            candidate_matches=matches,
            k=k,
            lambda_mult=lambda_mult,
        )

        selected_matches = [matches[i] for i in indices]
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
