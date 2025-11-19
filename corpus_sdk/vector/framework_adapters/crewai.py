# corpus_sdk/vector/framework_adapters/crewai.py
# SPDX-License-Identifier: Apache-2.0

"""
CrewAI tool adapter for Corpus Vector protocol.

This module exposes a Corpus `BaseVectorAdapter` as a CrewAI `BaseTool`
that performs semantic vector search with:

- Sync + async run APIs (`_run` / `_arun`)
- Optional streaming search via `stream_search(...)` using SyncStreamBridge
- Proper integration with Corpus VectorProtocolV1
- Namespace + metadata filter handling (capability-aware)
- Optional client-side score thresholding
- Optional embedding function integration
- Optional Max Marginal Relevance (MMR) diversification
- Optional OperationContext propagation via context_translation

Design philosophy
-----------------
- Protocol-first: CrewAI is a thin skin over Corpus vector adapters.
- All heavy lifting (backpressure, deadlines, breakers, etc.) lives in
  the underlying `BaseVectorAdapter`, not here.
- This layer focuses on:
    * Translating Corpus VectorMatch → JSON-serializable payloads
    * Respecting VectorCapabilities (namespaces, filters, top_k limits)
    * Propagating OperationContext when provided (crewai task or dict)
    * Providing an ergonomic tool interface for CrewAI agents

Typical usage
-------------

    from crewai import Agent
    from corpus_sdk.vector.pinecone_adapter import PineconeVectorAdapter
    from corpus_sdk.vector.framework_adapters.crewai import (
        CorpusCrewAIVectorSearchTool,
    )

    adapter = PineconeVectorAdapter(
        index_name="my-index",
        api_key="...",
        dimensions=1536,
    )

    # Provide an embedding function that maps List[str] -> List[List[float]]
    def embed_texts(texts: list[str]) -> list[list[float]]:
        ...

    search_tool = CorpusCrewAIVectorSearchTool(
        corpus_adapter=adapter,
        embedding_function=embed_texts,
        namespace="docs",
        default_top_k=4,
        score_threshold=0.2,
        use_mmr_by_default=True,
        mmr_lambda=0.5,
    )

    researcher = Agent(
        role="RAG Researcher",
        goal="Find the most relevant documents for a query",
        tools=[search_tool],
    )

    # Normal usage (sync via CrewAI)
    results = search_tool.run(
        query="what is corpus?",
        k=5,
        return_scores=True,
    )

    # Advanced streaming usage (manual, outside CrewAI's tool planner):
    for item in search_tool.stream_search(
        query="what is corpus?",
        k=5,
        return_scores=True,
    ):
        process(item)
"""

from __future__ import annotations

import logging
import math
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from pydantic import BaseModel, Field
from crewai.tools import BaseTool

from corpus_sdk.vector.vector_base import (
    BaseVectorAdapter,
    VectorMatch,
    QueryResult,
    QuerySpec,
    OperationContext,
    VectorCapabilities,
    # Errors
    BadRequest,
    NotSupported,
)
from corpus_sdk.core.async_bridge import AsyncBridge
from corpus_sdk.core.error_context import attach_context
from corpus_sdk.core.context_translation import (
    from_crewai as ctx_from_crewai,
    from_dict as ctx_from_dict,
)
from corpus_sdk.core.sync_bridge import sync_stream

logger = logging.getLogger(__name__)

Embeddings = Sequence[Sequence[float]]
Metadata = Dict[str, Any]


class CorpusVectorSearchInput(BaseModel):
    """
    Input schema for CorpusCrewAIVectorSearchTool.

    Fields are designed so that CrewAI can auto-generate tool calls,
    while still allowing advanced users to override behavior.
    """

    query: str = Field(
        ...,
        description="Natural language search query.",
    )

    k: Optional[int] = Field(
        default=None,
        description="Number of results to return. Defaults to the tool's default_top_k.",
    )

    namespace: Optional[str] = Field(
        default=None,
        description=(
            "Optional namespace override for the vector index. "
            "If omitted, the tool's configured namespace is used."
        ),
    )

    filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional metadata filter to restrict matches. "
            "Filter semantics are defined by the underlying Corpus adapter."
        ),
    )

    use_mmr: Optional[bool] = Field(
        default=None,
        description=(
            "Whether to apply Maximal Marginal Relevance (MMR) re-ranking "
            "for diversity. If omitted, the tool's use_mmr_by_default is used."
        ),
    )

    mmr_lambda: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "MMR lambda parameter in [0, 1]. Higher values favor relevance, "
            "lower values favor diversity. Defaults to the tool's mmr_lambda."
        ),
    )

    fetch_k: Optional[int] = Field(
        default=None,
        gt=0,
        description=(
            "Number of initial candidates to fetch before MMR re-ranking. "
            "Defaults to max(k * 4, k + 5) when MMR is enabled."
        ),
    )

    return_scores: Optional[bool] = Field(
        default=False,
        description="If true, include similarity scores in the results.",
    )

    embedding: Optional[List[float]] = Field(
        default=None,
        description=(
            "Optional precomputed query embedding. If provided, "
            "embedding_function is not used."
        ),
    )

    context: Optional[Any] = Field(
        default=None,
        description=(
            "Optional context used to build a Corpus OperationContext. "
            "For advanced use, this can be either:\n"
            "- A CrewAI Task-like object (translated via context_translation.from_crewai), or\n"
            "- A plain dict (translated via context_translation.from_dict)."
        ),
    )


class CorpusCrewAIVectorSearchTool(BaseTool):
    """
    CrewAI `BaseTool` implementation backed by a Corpus `BaseVectorAdapter`.

    This tool performs semantic vector search against a Corpus-backed index
    and returns a list of JSON objects:

        [
            {"text": "...", "metadata": {...}, "score": 0.87, "id": "..."},
            ...
        ]

    Key behaviors
    -------------
    - Uses an embedding function (if provided) to embed query text.
    - Respects VectorCapabilities (namespaces, filters, top_k limits).
    - Supports optional MMR re-ranking for diversity.
    - Optionally thresholds matches by score_threshold.
    - Optionally propagates OperationContext using context_translation.
    - Offers an advanced `stream_search(...)` API using SyncStreamBridge.
    """

    # CrewAI metadata
    name: str = "corpus_vector_search"
    description: str = (
        "Semantic vector search over a Corpus-backed index. "
        "Returns a JSON list of the most relevant text snippets and metadata "
        "for the given query, with optional MMR diversification."
    )

    # Tool input schema
    args_schema: Type[BaseModel] = CorpusVectorSearchInput

    # Core configuration
    corpus_adapter: BaseVectorAdapter
    namespace: Optional[str] = "default"

    id_field: str = "id"
    text_field: str = "page_content"
    metadata_field: Optional[str] = None

    score_threshold: Optional[float] = None
    default_top_k: int = 4

    # Optional embedding integration
    embedding_function: Optional[Any] = None  # Callable[[List[str]], Embeddings]

    # Optional MMR configuration
    use_mmr_by_default: bool = False
    mmr_lambda: float = 0.5

    # Cached capabilities
    _caps: Optional[VectorCapabilities] = None

    # Optional static OperationContext for advanced scenarios
    static_operation_context: Optional[OperationContext] = None

    # ------------------------------------------------------------------ #
    # Internal helpers: capabilities / context
    # ------------------------------------------------------------------ #

    async def _get_caps_async(self) -> VectorCapabilities:
        """
        Async capability fetch with caching.

        Uses the underlying adapter's capabilities() API and attaches
        rich error context on failure.
        """
        if self._caps is not None:
            return self._caps
        try:
            caps = await self.corpus_adapter.capabilities()
            self._caps = caps
            return caps
        except Exception as exc:  # noqa: BLE001
            attach_context(exc, framework="crewai", operation="capabilities")
            raise

    def _effective_namespace(self, namespace: Optional[str]) -> Optional[str]:
        """
        Resolve namespace using explicit override or tool default.
        """
        return namespace if namespace is not None else self.namespace

    def _resolve_operation_context(
        self,
        call_context: Optional[Any],
    ) -> Optional[OperationContext]:
        """
        Build or reuse an OperationContext for this call.

        Resolution order:
        1. If `call_context` is already an OperationContext, use it.
        2. If `call_context` is a Mapping, use context_translation.from_dict.
        3. Otherwise, attempt context_translation.from_crewai (for Task-like objects).
        4. Fallback to `static_operation_context` if present.
        5. Else, return None.
        """
        if isinstance(call_context, OperationContext):
            return call_context

        if call_context is not None:
            # Try dict-based translation first
            try:
                if isinstance(call_context, Mapping):
                    return ctx_from_dict(call_context)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "CorpusCrewAIVectorSearchTool: from_dict context translation failed: %s",
                    exc,
                )

            # Fall back to CrewAI-specific translation
            try:
                return ctx_from_crewai(call_context)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "CorpusCrewAIVectorSearchTool: from_crewai context translation failed: %s",
                    exc,
                )

        # Static context configured on the tool instance
        if isinstance(self.static_operation_context, OperationContext):
            return self.static_operation_context

        return None

    # ------------------------------------------------------------------ #
    # Embedding + match translation helpers
    # ------------------------------------------------------------------ #

    def _embed_query(
        self,
        query: str,
        *,
        embedding: Optional[Sequence[float]] = None,
    ) -> List[float]:
        """
        Ensure a single query embedding is available.

        Behavior:
        - If `embedding` is provided, coerce to float list.
        - Else, if `embedding_function` is set, compute embedding for [query].
        - Else, raise NotSupported.
        """
        if embedding is not None:
            return [float(x) for x in embedding]

        if self.embedding_function is None:
            raise NotSupported(
                "No embedding_function configured; caller must supply query embedding",
                code="NO_EMBEDDING_FUNCTION",
                details={"framework": "crewai", "query_preview": query[:64]},
            )

        try:
            embs = self.embedding_function([query])
        except Exception as exc:  # noqa: BLE001
            # Attach context to the underlying embedding error for observability
            attach_context(
                exc,
                framework="crewai",
                operation="embedding_function",
                query_preview=query[:128],
            )
            raise BadRequest(
                f"embedding_function failed for query: {exc}",
                code="EMBEDDING_ERROR",
            )

        if not embs or len(embs) != 1:
            raise BadRequest(
                "embedding_function must return exactly one embedding for a single query",
                code="BAD_EMBEDDINGS",
            )

        return [float(x) for x in embs[0]]

    def _match_to_payload(
        self,
        match: VectorMatch,
        *,
        return_scores: bool,
    ) -> Dict[str, Any]:
        """
        Convert a single VectorMatch into a JSON-serializable payload.
        """
        v = match.vector
        meta = dict(v.metadata or {})

        # Handle metadata envelope if configured
        if self.metadata_field and self.metadata_field in meta:
            nested = meta.get(self.metadata_field) or {}
            if isinstance(nested, Mapping):
                user_meta: Dict[str, Any] = dict(nested)
            else:
                user_meta = {}
        else:
            user_meta = meta

        # Extract text and id from metadata
        text = meta.get(self.text_field) or ""
        doc_id = meta.get(self.id_field) or getattr(v, "id", None)

        # Remove internal keys
        user_meta.pop(self.text_field, None)
        user_meta.pop(self.id_field, None)

        payload: Dict[str, Any] = {
            "text": text,
            "metadata": user_meta,
        }
        if doc_id is not None:
            payload["id"] = str(doc_id)
        if return_scores:
            payload["score"] = float(match.score)

        return payload

    # ------------------------------------------------------------------ #
    # Core async Corpus query operation
    # ------------------------------------------------------------------ #

    async def _aquery_embedding(
        self,
        embedding: Sequence[float],
        *,
        k: int,
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

        if caps.max_top_k is not None and k > caps.max_top_k:
            raise BadRequest(
                f"top_k {k} exceeds maximum of {caps.max_top_k}",
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
            top_k=k,
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
                framework="crewai",
                operation="query",
                namespace=ns,
                top_k=k,
            )
            raise

        matches = list(result.matches or [])

        # Apply optional client-side score threshold
        if self.score_threshold is not None:
            threshold = float(self.score_threshold)
            matches = [m for m in matches if float(m.score) >= threshold]

        return matches

    # ------------------------------------------------------------------ #
    # MMR utilities
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
        candidate_vecs: List[List[float]] = [
            list(match.vector.vector or []) for match in candidate_matches
        ]

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
            first_idx = max(candidates, key=lambda idx: normalized_scores[idx])
            selected.append(first_idx)
            candidates.remove(first_idx)

        while candidates and len(selected) < k:
            best_idx = None
            best_score = -float("inf")

            for idx in candidates:
                # Relevance term using original database score
                relevance = normalized_scores[idx]

                # Diversity term: max similarity to already selected items
                max_similarity = 0.0
                for sel_idx in selected:
                    similarity = get_similarity(idx, sel_idx)
                    max_similarity = max(max_similarity, similarity)

                # MMR score balancing relevance and diversity
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
    # High-level async search flows (simple vs MMR)
    # ------------------------------------------------------------------ #

    async def _asearch_simple(
        self,
        args: CorpusVectorSearchInput,
    ) -> List[Dict[str, Any]]:
        """
        Simple top-k search without MMR.
        """
        top_k = int(args.k or self.default_top_k)
        ns = args.namespace
        ctx = self._resolve_operation_context(args.context)

        query_emb = self._embed_query(args.query, embedding=args.embedding)
        matches = await self._aquery_embedding(
            query_emb,
            k=top_k,
            namespace=ns,
            filter=args.filter,
            include_vectors=False,
            ctx=ctx,
        )

        return_scores = bool(args.return_scores)
        return [self._match_to_payload(m, return_scores=return_scores) for m in matches]

    async def _asearch_with_mmr(
        self,
        args: CorpusVectorSearchInput,
    ) -> List[Dict[str, Any]]:
        """
        MMR-based search that first fetches candidates and then re-ranks them.
        """
        top_k = int(args.k or self.default_top_k)
        if top_k <= 0:
            return []

        ns = args.namespace
        ctx = self._resolve_operation_context(args.context)

        lambda_mult = (
            float(args.mmr_lambda)
            if args.mmr_lambda is not None
            else float(self.mmr_lambda)
        )
        # Clamp to [0, 1] defensively
        lambda_mult = max(0.0, min(1.0, lambda_mult))

        # Fetch more candidates than we will return
        fetch_k = args.fetch_k or max(top_k * 4, top_k + 5)

        query_emb = self._embed_query(args.query, embedding=args.embedding)
        matches = await self._aquery_embedding(
            query_emb,
            k=int(fetch_k),
            namespace=ns,
            filter=args.filter,
            include_vectors=True,
            ctx=ctx,
        )

        if not matches:
            return []

        indices = self._mmr_select_indices(
            query_vec=query_emb,
            candidate_matches=matches,
            k=top_k,
            lambda_mult=lambda_mult,
        )

        return_scores = bool(args.return_scores)
        return [
            self._match_to_payload(matches[i], return_scores=return_scores)
            for i in indices
        ]

    async def _asearch(
        self,
        args: CorpusVectorSearchInput,
    ) -> List[Dict[str, Any]]:
        """
        Unified async search entry point, dispatching to simple or MMR-based search.
        """
        use_mmr = (
            bool(args.use_mmr)
            if args.use_mmr is not None
            else bool(self.use_mmr_by_default)
        )

        if use_mmr:
            return await self._asearch_with_mmr(args)
        return await self._asearch_simple(args)

    # ------------------------------------------------------------------ #
    # CrewAI Tool API: sync + async
    # ------------------------------------------------------------------ #

    def _run(self, **kwargs: Any) -> List[Dict[str, Any]]:
        """
        Synchronous tool entrypoint used by CrewAI.

        Internally bridges to the async implementation via AsyncBridge.
        """
        args = self.args_schema(**kwargs)  # type: ignore[arg-type]
        try:
            return AsyncBridge.run_async(self._asearch(args))
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="crewai",
                operation="tool_run",
                query=getattr(args, "query", None),
            )
            raise

    async def _arun(self, **kwargs: Any) -> List[Dict[str, Any]]:
        """
        Async tool entrypoint for advanced users.

        CrewAI *may* call this in some configurations, but typical usage
        goes through `_run`.
        """
        args = self.args_schema(**kwargs)  # type: ignore[arg-type]
        try:
            return await self._asearch(args)
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="crewai",
                operation="tool_arun",
                query=getattr(args, "query", None),
            )
            raise

    # ------------------------------------------------------------------ #
    # Advanced: streaming search using SyncStreamBridge
    # ------------------------------------------------------------------ #

    def stream_search(self, **kwargs: Any) -> Iterator[Dict[str, Any]]:
        """
        Streaming search API for advanced callers (outside CrewAI's planner).

        This method:
        - Uses the same input schema as `_run` / `_arun`.
        - Bridges an async generator into a synchronous iterator using
          `SyncStreamBridge` via the `sync_stream` helper.
        - Applies MMR when configured, but still streams the final ordering
          incrementally to the caller.

        Example
        -------
            for result in tool.stream_search(
                query="hello world",
                k=5,
                return_scores=True,
            ):
                handle(result)
        """
        args = self.args_schema(**kwargs)  # type: ignore[arg-type]

        # Decide whether to use MMR
        use_mmr = (
            bool(args.use_mmr)
            if args.use_mmr is not None
            else bool(self.use_mmr_by_default)
        )

        top_k = int(args.k or self.default_top_k)
        if top_k <= 0:
            return iter(())  # Empty iterator

        ns = args.namespace
        ctx = self._resolve_operation_context(args.context)
        return_scores = bool(args.return_scores)

        query_emb = self._embed_query(args.query, embedding=args.embedding)

        async def _stream_coro():
            """
            Async generator that fetches matches once, applies optional MMR,
            and then yields VectorMatch objects one by one.
            """
            # When streaming, we still want to have enough candidates for MMR.
            fetch_k = args.fetch_k or max(top_k * 4, top_k + 5) if use_mmr else top_k

            matches = await self._aquery_embedding(
                query_emb,
                k=int(fetch_k),
                namespace=ns,
                filter=args.filter,
                include_vectors=use_mmr,
                ctx=ctx,
            )

            if not matches:
                return

            if use_mmr:
                lambda_mult = (
                    float(args.mmr_lambda)
                    if args.mmr_lambda is not None
                    else float(self.mmr_lambda)
                )
                lambda_mult = max(0.0, min(1.0, lambda_mult))
                indices = self._mmr_select_indices(
                    query_vec=query_emb,
                    candidate_matches=matches,
                    k=top_k,
                    lambda_mult=lambda_mult,
                )
                for idx in indices:
                    yield matches[idx]
            else:
                # No MMR, just yield top_k directly
                for match in matches[:top_k]:
                    yield match

        # Bridge the async generator into a synchronous iterator
        for match in sync_stream(
            _stream_coro,
            framework="crewai",
            error_context={
                "operation": "stream_search",
                "namespace": ns,
                "top_k": top_k,
            },
        ):
            # Convert each VectorMatch to a JSON-serializable payload
            yield self._match_to_payload(
                match,
                return_scores=return_scores,
            )


__all__ = [
    "CorpusVectorSearchInput",
    "CorpusCrewAIVectorSearchTool",
]
