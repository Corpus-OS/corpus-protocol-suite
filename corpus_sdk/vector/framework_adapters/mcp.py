# corpus_sdk/vector/framework_adapters/mcp_vector.py
# SPDX-License-Identifier: Apache-2.0

"""
MCP vector adapter for Corpus Vector protocol.

This module exposes Corpus `VectorProtocolV1` implementations to MCP-based
tools as an async-only, protocol-first vector service.

Design goals
------------
- Protocol-first:
    * All heavy lifting (backpressure, batching, deadlines, breakers,
      retries, etc.) lives in the adapter + shared `VectorTranslator`,
      not in this layer.
- MCP-aligned:
    * Async-only API that fits naturally into MCP tool execution.
    * Concrete, predictable return types (`MCPVectorResult`) instead of `Any`.
    * Embedding-agnostic: this service assumes embeddings are provided by
      a separate MCP embedding service or upstream caller.
- Thin but useful:
    * Capability-aware validation (top_k, filters, namespaces).
    * Optional client-side score thresholding.
    * Optional MMR search and streaming search.
    * High quality error context via `attach_context` + `VECTOR_ERROR_CODES`.

Typical usage
-------------

    from corpus_sdk.vector.pinecone_adapter import PineconeVectorAdapter
    from corpus_sdk.vector.framework_adapters.mcp_vector import (
        MCPVectorService,
        MCPVectorResult,
    )

    adapter = PineconeVectorAdapter(
        index_name="my-index",
        api_key="...",
        dimensions=1536,
    )

    vector_service = MCPVectorService(
        corpus_adapter=adapter,
        namespace="docs",
        default_top_k=4,
        score_threshold=0.2,
    )

    # Adding vectors (embeddings are computed elsewhere, e.g. MCP embedding tool)
    texts = ["hello world", "goodbye world"]
    embeddings = [
        [...],  # embedding for "hello world"
        [...],  # embedding for "goodbye world"
    ]
    ids = await vector_service.add_texts(
        texts,
        embeddings=embeddings,
        metadatas=[{"kind": "greeting"}, {"kind": "farewell"}],
    )

    # Similarity search (requires query embedding from embedding service)
    query_embedding = [...]
    results = await vector_service.similarity_search(
        query="hello",
        embedding=query_embedding,
        k=4,
    )

    # Streaming search
    async for result in vector_service.similarity_search_stream(
        query="hello",
        embedding=query_embedding,
        k=8,
    ):
        handle(result)

    # MMR search (for diverse results)
    mmr_results = await vector_service.max_marginal_relevance_search(
        query="hello",
        embedding=query_embedding,
        k=4,
        lambda_mult=0.5,
        fetch_k=16,
    )

    # Delete by IDs
    await vector_service.delete(ids=ids)

This layer is intentionally framework-agnostic and MCP-focused:
it does not depend on any MCP runtime types directly; callers can
wire it into MCP tools however they like.
"""

from __future__ import annotations

import logging
import math
import uuid
from functools import cached_property
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
)

from corpus_sdk.core.context_translation import (
    from_mcp as ctx_from_mcp,
    from_dict as ctx_from_dict,
)
from corpus_sdk.core.error_context import attach_context
from corpus_sdk.vector.vector_base import (
    VectorProtocolV1,
    Vector,
    VectorMatch,
    QueryResult,
    UpsertResult,
    DeleteResult,
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

logger = logging.getLogger(__name__)

FRAMEWORK_LABEL = "mcp"

VECTOR_ERROR_CODES = VectorCoercionErrorCodes(framework_label=FRAMEWORK_LABEL)
VECTOR_LIMITS = VectorResourceLimits()
VECTOR_FLAGS = VectorValidationFlags()
TOPK_WARNING_CONFIG = TopKWarningConfig(framework_label=FRAMEWORK_LABEL)


Metadata = Dict[str, Any]
Embeddings = Sequence[Sequence[float]]


class MCPVectorResult(TypedDict):
    """
    MCP-facing vector search result.

    This is intentionally AI-friendly and JS/TS-friendly:
    - `content` is the main text payload.
    - `metadata` contains user / application metadata only
      (internal/system fields are stripped).
    - `confidence` is the underlying adapter similarity score
      (typically 0-1, higher is better).
    - `namespace` is included for multi-tenant / multi-space setups.
    """

    id: str
    content: str
    metadata: Dict[str, Any]
    confidence: float
    namespace: Optional[str]
    source: str


class MCPVectorService:
    """
    MCP-focused vector service over a Corpus `VectorProtocolV1` adapter.

    This class is:

    - Async-only (designed for MCP tool execution).
    - Embedding-agnostic (requires caller-provided embeddings).
    - Thin but capability-aware (top_k limits, filter support, namespaces).
    - Richly instrumented with error context for production debugging.

    It does NOT:
    - Compute embeddings (compose with an embedding MCP service for that).
    - Implement vector database algorithms (delegated to the adapter).
    - Own batching/backpressure logic (delegated to VectorTranslator).
    """

    def __init__(
        self,
        *,
        corpus_adapter: VectorProtocolV1,
        namespace: Optional[str] = "default",
        id_field: str = "id",
        text_field: str = "page_content",
        metadata_field: Optional[str] = None,
        score_threshold: Optional[float] = None,
        batch_size: int = 100,
        default_top_k: int = 4,
    ) -> None:
        self.corpus_adapter: VectorProtocolV1 = corpus_adapter
        self.namespace = namespace

        # Reserved field configuration for metadata layout
        self.id_field = str(id_field)
        self.text_field = str(text_field)
        self.metadata_field = str(metadata_field) if metadata_field else None

        # Ensure uniqueness of reserved metadata fields
        reserved_fields = {self.id_field, self.text_field}
        if self.metadata_field:
            reserved_fields.add(self.metadata_field)
        if len(reserved_fields) != (3 if self.metadata_field else 2):
            raise ValueError(
                f"Reserved metadata fields must be unique: {reserved_fields}"
            )

        # Basic numeric validation (aligned with other framework adapters)
        self.score_threshold = self._validate_score_threshold(score_threshold)
        self.batch_size = self._validate_batch_size(batch_size)
        self.default_top_k = self._validate_default_top_k(default_top_k)

        # Cached capabilities
        self._caps: Optional[VectorCapabilities] = None

    # ------------------------------------------------------------------ #
    # Basic validation helpers
    # ------------------------------------------------------------------ #

    def _validate_batch_size(self, batch_size: int) -> int:
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if batch_size > 10000:
            logger.warning(
                "batch_size %d is unusually large; consider reducing for better performance",
                batch_size,
            )
        return batch_size

    def _validate_default_top_k(self, default_top_k: int) -> int:
        default_top_k = int(default_top_k)
        if default_top_k < 1:
            raise ValueError("default_top_k must be at least 1")
        if default_top_k > 1000:
            logger.warning(
                "default_top_k %d is unusually large; most applications use 4-100",
                default_top_k,
            )
        return default_top_k

    def _validate_score_threshold(
        self,
        score_threshold: Optional[float],
    ) -> Optional[float]:
        if score_threshold is not None:
            score_threshold = float(score_threshold)
            if not (0.0 <= score_threshold <= 1.0):
                raise ValueError("score_threshold must be between 0.0 and 1.0")
            if score_threshold > 0.9:
                logger.warning(
                    "score_threshold %.2f is very high; may filter out relevant results",
                    score_threshold,
                )
        return score_threshold

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
            framework=FRAMEWORK_LABEL,
            translator=framework_translator,
        )

    # ------------------------------------------------------------------ #
    # Capabilities / context helpers
    # ------------------------------------------------------------------ #

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
            attach_context(
                exc,
                framework=FRAMEWORK_LABEL,
                operation="capabilities_async",
                error_codes=VECTOR_ERROR_CODES,
            )
            raise

    async def get_capabilities(self) -> VectorCapabilities:
        """
        Public method for MCP callers to inspect vector capabilities.

        Useful for exposing MCP tools like `vector_get_capabilities`.
        """
        return await self._get_caps_async()

    def _effective_namespace(self, namespace: Optional[str]) -> Optional[str]:
        """
        Resolve namespace using explicit override or store default.

        If the underlying adapter does not support namespaces, this value is
        still passed down; the adapter may ignore it.
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
            framework=FRAMEWORK_LABEL,
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

    def _build_ctx_from_mcp(
        self,
        mcp_context: Optional[Mapping[str, Any]],
    ) -> Optional[OperationContext]:
        """
        Build an OperationContext from MCP-ish context dict.

        Priority:
        1. `mcp_context` interpreted via `ctx_from_mcp` (if it looks like MCP).
        2. Fallback to `ctx_from_dict`.
        3. None (no context).
        """
        if mcp_context is None:
            return None

        # Try MCP-aware translation first
        try:
            ctx = ctx_from_mcp(mcp_context)
            if isinstance(ctx, OperationContext):
                return ctx
        except Exception as exc:  # noqa: BLE001
            logger.debug("ctx_from_mcp failed: %s", exc)

        # Fallback to generic dict-based translation
        try:
            ctx = ctx_from_dict(mcp_context)
            if isinstance(ctx, OperationContext):
                return ctx
        except Exception as exc:  # noqa: BLE001
            logger.debug("ctx_from_dict failed in _build_ctx_from_mcp: %s", exc)

        return None

    # ------------------------------------------------------------------ #
    # Translation helpers: text/embeddings ↔ Corpus Vector
    # ------------------------------------------------------------------ #

    def _normalize_metadatas(
        self,
        n: int,
        metadatas: Optional[List[Metadata]],
    ) -> List[Metadata]:
        """
        Normalize metadata list to length n.

        Behavior:
        - If metadatas is None: return [{} for _ in range(n)].
        - If len(metadatas) == n: return shallow copies.
        - If len(metadatas) == 1 and n > 1: replicate the single metadata.
        - Else: raise BadRequest.
        """
        if metadatas is None:
            return [{} for _ in range(n)]

        if len(metadatas) == n:
            return [dict(m or {}) for m in metadatas]

        if len(metadatas) == 1 and n > 1:
            base = dict(metadatas[0] or {})
            return [dict(base) for _ in range(n)]

        err = BadRequest(
            f"metadatas length {len(metadatas)} does not match texts length {n}",
            code="BAD_METADATA",
            details={"texts": n, "metadatas": len(metadatas)},
        )
        attach_context(
            err,
            framework=FRAMEWORK_LABEL,
            operation="normalize_metadatas",
            error_codes=VECTOR_ERROR_CODES,
            texts_count=n,
            metadatas_count=len(metadatas),
        )
        raise err

    def _normalize_ids(
        self,
        n: int,
        ids: Optional[List[str]],
    ) -> List[str]:
        """
        Normalize IDs list to length n.

        Behavior:
        - If ids is None: generate UUID4 hex IDs.
        - If len(ids) == n: coerce to str list.
        - Else: raise BadRequest.
        """
        if ids is None:
            return [uuid.uuid4().hex for _ in range(n)]
        if len(ids) != n:
            err = BadRequest(
                f"ids length {len(ids)} does not match texts length {n}",
                code="BAD_IDS",
                details={"texts": n, "ids": len(ids)},
            )
            attach_context(
                err,
                framework=FRAMEWORK_LABEL,
                operation="normalize_ids",
                error_codes=VECTOR_ERROR_CODES,
                texts_count=n,
                ids_count=len(ids),
            )
            raise err
        return [str(i) for i in ids]

    def _to_corpus_vectors(
        self,
        texts: List[str],
        embeddings: Embeddings,
        metadatas: List[Metadata],
        ids: List[str],
        namespace: Optional[str],
    ) -> List[Vector]:
        """
        Convert texts + embeddings + metadatas into Corpus `Vector` objects.
        """
        vectors: List[Vector] = []
        ns = self._effective_namespace(namespace)

        for text, emb, meta, vid in zip(texts, embeddings, metadatas, ids):
            if self.metadata_field:
                envelope: Metadata = {}
                if meta:
                    envelope[self.metadata_field] = dict(meta)
                envelope[self.text_field] = text
                envelope[self.id_field] = vid
                metadata_payload: Metadata = envelope
            else:
                metadata_payload = dict(meta or {})
                metadata_payload[self.text_field] = text
                metadata_payload[self.id_field] = vid

            vectors.append(
                Vector(
                    id=str(vid),
                    vector=[float(x) for x in emb],
                    metadata=metadata_payload,
                    namespace=ns,
                    text=None,
                )
            )

        return vectors

    def _truncate_text(self, text: str, max_length: int = 2000) -> str:
        """
        Truncate text for token/response efficiency in MCP tools.

        Keeps word boundaries when possible.
        """
        if len(text) <= max_length:
            return text
        truncated = text[:max_length].rsplit(" ", 1)[0]
        if not truncated:
            truncated = text[:max_length]
        return truncated + "..."

    def _from_corpus_matches(
        self,
        matches: Sequence[VectorMatch],
        namespace: Optional[str],
    ) -> List[MCPVectorResult]:
        """
        Convert Corpus `VectorMatch` objects into MCPVectorResult structures.

        The similarity score returned is `VectorMatch.score`, which is assumed
        to be higher-is-better (normalized similarity as defined by the adapter).
        """
        results: List[MCPVectorResult] = []
        ns = self._effective_namespace(namespace)

        internal_fields = {
            "id",
            "vector",
            "_id",
            "_vector",
            "embedding",
            "timestamp",
            self.text_field,
            self.id_field,
        }

        for m in matches:
            v = m.vector
            raw_meta = dict(v.metadata or {})

            if self.metadata_field and self.metadata_field in raw_meta:
                nested = raw_meta.get(self.metadata_field) or {}
                if isinstance(nested, Mapping):
                    meta = dict(nested)
                else:
                    meta = {}
            else:
                meta = dict(raw_meta)

            text = raw_meta.get(self.text_field) or ""

            # Strip internal/system fields from user-visible metadata
            for k in list(meta.keys()):
                if k in internal_fields:
                    meta.pop(k, None)

            vid = str(v.id or raw_meta.get(self.id_field) or "")

            results.append(
                MCPVectorResult(
                    id=vid,
                    content=self._truncate_text(text),
                    metadata=meta,
                    confidence=float(m.score),
                    namespace=ns,
                    source="vector_database",
                )
            )

        return results

    def _apply_score_threshold(
        self,
        matches: List[VectorMatch],
    ) -> List[VectorMatch]:
        """
        Apply optional client-side score thresholding to a list of matches.
        """
        if self.score_threshold is None:
            return matches
        threshold = float(self.score_threshold)
        return [m for m in matches if float(m.score) >= threshold]

    # ------------------------------------------------------------------ #
    # Raw request builders
    # ------------------------------------------------------------------ #

    def _build_upsert_request(
        self,
        vectors: List[Vector],
        *,
        namespace: Optional[str],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
        ns = self._effective_namespace(namespace)
        raw: Dict[str, Any] = {
            "namespace": ns,
            "ids": [str(i) for i in ids] if ids else None,
            "filter": dict(filter) if filter else None,
        }
        framework_ctx = self._framework_ctx_for_namespace(ns)
        return raw, framework_ctx

    # ------------------------------------------------------------------ #
    # Graceful error recovery for batch operations
    # ------------------------------------------------------------------ #

    def _handle_partial_upsert_failure(
        self,
        result: UpsertResult,
        total_texts: int,
        namespace: Optional[str],
    ) -> None:
        """
        Handle partial failures in batch upsert operations gracefully.

        Logs warnings for partial failures but doesn't raise exceptions for
        non-critical failures, allowing successful operations to proceed.
        Only raises exceptions for complete failures.
        """
        if result.failed_count and result.failed_count > 0:
            successful = result.upserted_count or 0
            failed = result.failed_count

            logger.warning(
                "Partial upsert failure: %d/%d texts succeeded, %d failed in namespace %s",
                successful,
                total_texts,
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

        if (result.upserted_count or 0) == 0 and total_texts > 0:
            err = VectorAdapterError(
                f"All {total_texts} texts failed to upsert",
                code="BATCH_UPSERT_FAILED",
                details={
                    "total_texts": total_texts,
                    "namespace": namespace,
                    "failures": result.failures or [],
                },
            )
            attach_context(
                err,
                framework=FRAMEWORK_LABEL,
                operation="batch_upsert",
                error_codes=VECTOR_ERROR_CODES,
                texts_count=total_texts,
                namespace=self._effective_namespace(namespace),
            )
            raise err

    # ------------------------------------------------------------------ #
    # Validation helpers aligned with capabilities
    # ------------------------------------------------------------------ #

    async def _validate_query_params_async(
        self,
        top_k: int,
        namespace: Optional[str],
        filter: Optional[Mapping[str, Any]],
    ) -> int:
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
                framework=FRAMEWORK_LABEL,
                operation="query_async",
                error_codes=VECTOR_ERROR_CODES,
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
                framework=FRAMEWORK_LABEL,
                operation="query_async",
                error_codes=VECTOR_ERROR_CODES,
                namespace=ns,
                top_k=effective_top_k,
            )
            raise err

        return effective_top_k

    async def _validate_delete_params_async(
        self,
        *,
        ids: Optional[List[str]],
        namespace: Optional[str],
        filter: Optional[Mapping[str, Any]],
    ) -> None:
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
                framework=FRAMEWORK_LABEL,
                operation="delete_async",
                error_codes=VECTOR_ERROR_CODES,
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
                framework=FRAMEWORK_LABEL,
                operation="delete_async",
                error_codes=VECTOR_ERROR_CODES,
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
        if isinstance(result, QueryResult):
            return result

        err = BadRequest(
            f"{operation} returned unsupported type: {type(result).__name__}",
            code="BAD_TRANSLATED_RESULT",
        )
        attach_context(
            err,
            framework=FRAMEWORK_LABEL,
            operation=operation,
            error_codes=VECTOR_ERROR_CODES,
        )
        raise err

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
        query_vec: Sequence[float],  # kept for future use if needed
        candidate_matches: List[VectorMatch],
        k: int,
        lambda_mult: float,
    ) -> List[int]:
        """
        Improved MMR selector that respects original database scores and caches similarities.

        Args:
            query_vec: The query embedding vector (currently unused but kept for extensibility).
            candidate_matches: Candidate matches with original scores and vectors.
            k: Number of results to select.
            lambda_mult: MMR lambda parameter (0-1), higher values favor relevance.

        Returns:
            Indices into candidate_matches for selected results.
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

        original_scores = [float(match.score) for match in candidate_matches]
        candidate_vecs = [match.vector.vector or [] for match in candidate_matches]

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
            first_idx = max(candidates, key=lambda idx: normalized_scores[idx])
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
    # Public async API (MCP-facing)
    # ------------------------------------------------------------------ #

    async def add_texts(
        self,
        texts: Iterable[str],
        *,
        embeddings: Optional[Embeddings] = None,
        metadatas: Optional[List[Metadata]] = None,
        ids: Optional[List[str]] = None,
        namespace: Optional[str] = None,
        mcp_context: Optional[Mapping[str, Any]] = None,
    ) -> List[str]:
        """
        Add texts + embeddings to the vector store (async).

        This service is embedding-agnostic:
        - `embeddings` MUST be provided by the caller.
        - No attempt is made to compute embeddings here.

        Returns:
            List of vector IDs.
        """
        texts_list = [str(t) for t in texts]
        if not texts_list:
            return []

        if embeddings is None:
            err = NotSupported(
                "MCPVectorService.add_texts requires caller-provided embeddings; "
                "compose with an MCP embedding service.",
                code="NO_EMBEDDINGS",
                details={"texts": len(texts_list)},
            )
            attach_context(
                err,
                framework=FRAMEWORK_LABEL,
                operation="add_texts",
                error_codes=VECTOR_ERROR_CODES,
            )
            raise err

        if len(embeddings) != len(texts_list):
            err = BadRequest(
                f"embeddings length {len(embeddings)} does not match texts length {len(texts_list)}",
                code="BAD_EMBEDDINGS",
                details={"texts": len(texts_list), "embeddings": len(embeddings)},
            )
            attach_context(
                err,
                framework=FRAMEWORK_LABEL,
                operation="add_texts",
                error_codes=VECTOR_ERROR_CODES,
                texts_count=len(texts_list),
                embeddings_count=len(embeddings),
            )
            raise err

        ns = namespace or self.namespace
        ctx = self._build_ctx_from_mcp(mcp_context)

        metadatas_norm = self._normalize_metadatas(len(texts_list), metadatas)
        ids_norm = self._normalize_ids(len(texts_list), ids)

        vectors = self._to_corpus_vectors(
            texts=texts_list,
            embeddings=embeddings,
            metadatas=metadatas_norm,
            ids=ids_norm,
            namespace=ns,
        )

        raw_request, framework_ctx = self._build_upsert_request(
            vectors,
            namespace=ns,
        )

        try:
            result = await self._translator.arun_upsert(
                raw_request,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

            if isinstance(result, UpsertResult):
                self._handle_partial_upsert_failure(
                    result,
                    len(texts_list),
                    ns,
                )

        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework=FRAMEWORK_LABEL,
                operation="add_texts",
                error_codes=VECTOR_ERROR_CODES,
                namespace=self._effective_namespace(ns),
                texts_count=len(texts_list),
            )
            raise

        return ids_norm

    async def add_documents(
        self,
        documents: Iterable[Mapping[str, Any]],
        *,
        embeddings: Optional[Embeddings] = None,
        namespace: Optional[str] = None,
        mcp_context: Optional[Mapping[str, Any]] = None,
    ) -> List[str]:
        """
        Add generic document-like objects to the vector store (async).

        Expected shape per document:
            {
                "page_content": str,  # or field matching `text_field`
                "metadata": dict,     # optional
            }

        Embeddings must be provided and must align with the order/length of
        the yielded texts.
        """
        docs = list(documents)
        if not docs:
            return []

        texts: List[str] = []
        metadatas: List[Metadata] = []

        for d in docs:
            text = d.get(self.text_field) or d.get("page_content") or ""
            texts.append(str(text))
            meta = d.get("metadata") or {}
            metadatas.append(dict(meta))

        return await self.add_texts(
            texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=None,
            namespace=namespace,
            mcp_context=mcp_context,
        )

    async def similarity_search(
        self,
        query: str,
        *,
        embedding: Optional[Sequence[float]] = None,
        k: int = 4,
        filter: Optional[Mapping[str, Any]] = None,
        namespace: Optional[str] = None,
        mcp_context: Optional[Mapping[str, Any]] = None,
    ) -> List[MCPVectorResult]:
        """
        Perform similarity search and return MCPVectorResult list (async).

        This service does NOT embed the query; it expects `embedding` to be
        provided by the caller (e.g. MCP embedding service).
        """
        if embedding is None:
            err = NotSupported(
                "MCPVectorService.similarity_search requires caller-provided query embedding; "
                "compose with an MCP embedding service.",
                code="NO_QUERY_EMBEDDING",
                details={"query": query},
            )
            attach_context(
                err,
                framework=FRAMEWORK_LABEL,
                operation="similarity_search",
                error_codes=VECTOR_ERROR_CODES,
            )
            raise err

        ns = namespace or self.namespace
        ctx = self._build_ctx_from_mcp(mcp_context)

        top_k = int(k or self.default_top_k)
        top_k = await self._validate_query_params_async(top_k, ns, filter)

        warn_if_extreme_k(
            top_k,
            framework=FRAMEWORK_LABEL,
            op_name="similarity_search_async",
            warning_config=TOPK_WARNING_CONFIG,
            logger=logger,
        )

        raw_query, framework_ctx = self._build_query_request(
            embedding,
            top_k=top_k,
            namespace=ns,
            filter=filter,
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
                framework=FRAMEWORK_LABEL,
                operation="similarity_search",
                error_codes=VECTOR_ERROR_CODES,
                namespace=self._effective_namespace(ns),
                top_k=top_k,
            )
            raise

        result = self._validate_query_result_type(
            result_any,
            operation="translator.query_async",
        )

        matches_list: List[VectorMatch] = list(result.matches or [])
        matches_list = self._apply_score_threshold(matches_list)

        return self._from_corpus_matches(matches_list, namespace=ns)

    async def similarity_search_with_score(
        self,
        query: str,
        *,
        embedding: Optional[Sequence[float]] = None,
        k: int = 4,
        filter: Optional[Mapping[str, Any]] = None,
        namespace: Optional[str] = None,
        mcp_context: Optional[Mapping[str, Any]] = None,
    ) -> List[Tuple[MCPVectorResult, float]]:
        """
        Similarity search returning (result, confidence) tuples (async).

        Convenience wrapper around `similarity_search`.
        """
        docs = await self.similarity_search(
            query=query,
            embedding=embedding,
            k=k,
            filter=filter,
            namespace=namespace,
            mcp_context=mcp_context,
        )
        return [(doc, float(doc["confidence"])) for doc in docs]

    async def similarity_search_stream(
        self,
        query: str,
        *,
        embedding: Optional[Sequence[float]] = None,
        k: int = 4,
        filter: Optional[Mapping[str, Any]] = None,
        namespace: Optional[str] = None,
        mcp_context: Optional[Mapping[str, Any]] = None,
    ) -> AsyncIterator[MCPVectorResult]:
        """
        Streaming similarity search (async), yielding MCPVectorResult items.

        This is particularly useful for MCP tools that want to stream
        partial results back to the client as they arrive.
        """
        if embedding is None:
            err = NotSupported(
                "MCPVectorService.similarity_search_stream requires caller-provided query embedding; "
                "compose with an MCP embedding service.",
                code="NO_QUERY_EMBEDDING",
                details={"query": query},
            )
            attach_context(
                err,
                framework=FRAMEWORK_LABEL,
                operation="similarity_search_stream",
                error_codes=VECTOR_ERROR_CODES,
            )
            raise err

        ns = namespace or self.namespace
        ctx = self._build_ctx_from_mcp(mcp_context)

        top_k = int(k or self.default_top_k)
        top_k = await self._validate_query_params_async(top_k, ns, filter)

        warn_if_extreme_k(
            top_k,
            framework=FRAMEWORK_LABEL,
            op_name="similarity_search_stream",
            warning_config=TOPK_WARNING_CONFIG,
            logger=logger,
        )

        raw_query, framework_ctx = self._build_query_request(
            embedding,
            top_k=top_k,
            namespace=ns,
            filter=filter,
            include_vectors=False,
        )

        yielded = 0

        try:
            async for chunk in self._translator.query_stream(
                raw_query,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            ):
                raw_matches_obj = getattr(chunk, "matches", None)
                if raw_matches_obj is None and isinstance(chunk, Mapping):
                    raw_matches_obj = chunk.get("matches")
                if raw_matches_obj is None:
                    err = VectorAdapterError(
                        f"VectorTranslator.query_stream yielded unsupported type: {type(chunk).__name__}",
                        code="BAD_STREAM_CHUNK",
                    )
                    attach_context(
                        err,
                        framework=FRAMEWORK_LABEL,
                        operation="similarity_search_stream",
                        error_codes=VECTOR_ERROR_CODES,
                        namespace=self._effective_namespace(ns),
                        top_k=top_k,
                    )
                    raise err

                raw_matches = list(raw_matches_obj or [])
                filtered_matches = self._apply_score_threshold(raw_matches)

                if not filtered_matches:
                    continue

                formatted = self._from_corpus_matches(filtered_matches, namespace=ns)
                for doc in formatted:
                    if yielded >= top_k:
                        return
                    yielded += 1
                    yield doc

        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework=FRAMEWORK_LABEL,
                operation="similarity_search_stream",
                error_codes=VECTOR_ERROR_CODES,
                namespace=self._effective_namespace(ns),
                top_k=top_k,
            )
            raise

    async def max_marginal_relevance_search(
        self,
        query: str,
        *,
        embedding: Optional[Sequence[float]] = None,
        k: int = 4,
        lambda_mult: float = 0.5,
        fetch_k: Optional[int] = None,
        filter: Optional[Mapping[str, Any]] = None,
        namespace: Optional[str] = None,
        mcp_context: Optional[Mapping[str, Any]] = None,
    ) -> List[MCPVectorResult]:
        """
        Perform Maximal Marginal Relevance (MMR) search (async).

        This runs a similarity search with a larger `fetch_k` and then
        selects a subset of results via MMR based on vector geometry and
        original database scores. Returns MCPVectorResult items.
        """
        if embedding is None:
            err = NotSupported(
                "MCPVectorService.max_marginal_relevance_search requires caller-provided query embedding; "
                "compose with an MCP embedding service.",
                code="NO_QUERY_EMBEDDING",
                details={"query": query},
            )
            attach_context(
                err,
                framework=FRAMEWORK_LABEL,
                operation="mmr_search_async",
                error_codes=VECTOR_ERROR_CODES,
            )
            raise err

        if not (0.0 <= lambda_mult <= 1.0):
            err = BadRequest(
                f"lambda_mult must be in [0, 1], got {lambda_mult}",
                code="BAD_MMR_LAMBDA",
            )
            attach_context(
                err,
                framework=FRAMEWORK_LABEL,
                operation="mmr_search_async",
                error_codes=VECTOR_ERROR_CODES,
                lambda_mult=lambda_mult,
            )
            raise err

        ns = namespace or self.namespace
        ctx = self._build_ctx_from_mcp(mcp_context)

        k_eff = int(k or self.default_top_k)
        if k_eff <= 0:
            return []

        fetch_k_eff = int(fetch_k or max(k_eff * 4, k_eff + 5))

        k_eff = await self._validate_query_params_async(k_eff, ns, filter)
        fetch_k_eff = await self._validate_query_params_async(fetch_k_eff, ns, filter)

        warn_if_extreme_k(
            k_eff,
            framework=FRAMEWORK_LABEL,
            op_name="mmr_search_async",
            warning_config=TOPK_WARNING_CONFIG,
            logger=logger,
        )
        warn_if_extreme_k(
            fetch_k_eff,
            framework=FRAMEWORK_LABEL,
            op_name="mmr_search_async_fetch_k",
            warning_config=TOPK_WARNING_CONFIG,
            logger=logger,
        )

        raw_query, framework_ctx = self._build_query_request(
            embedding,
            top_k=fetch_k_eff,
            namespace=ns,
            filter=filter,
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
                framework=FRAMEWORK_LABEL,
                operation="mmr_search_async",
                error_codes=VECTOR_ERROR_CODES,
                namespace=self._effective_namespace(ns),
                top_k=fetch_k_eff,
            )
            raise

        result = self._validate_query_result_type(
            result_any,
            operation="translator.query_mmr_async",
        )

        matches_list: List[VectorMatch] = list(result.matches or [])
        matches_list = self._apply_score_threshold(matches_list)

        if not matches_list:
            return []

        indices = self._mmr_select_indices(
            query_vec=list(embedding),
            candidate_matches=matches_list,
            k=k_eff,
            lambda_mult=lambda_mult,
        )

        selected_matches = [matches_list[i] for i in indices]
        return self._from_corpus_matches(selected_matches, namespace=ns)

    async def delete(
        self,
        ids: Optional[List[str]] = None,
        *,
        filter: Optional[Mapping[str, Any]] = None,
        namespace: Optional[str] = None,
        mcp_context: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Delete vectors by IDs or metadata filter (async).

        At least one of `ids` or `filter` must be provided.
        """
        ns = namespace or self.namespace
        ctx = self._build_ctx_from_mcp(mcp_context)

        await self._validate_delete_params_async(
            ids=ids,
            namespace=ns,
            filter=filter,
        )

        raw_request, framework_ctx = self._build_delete_request(
            ids=ids,
            namespace=ns,
            filter=filter,
        )

        try:
            result = await self._translator.arun_delete(
                raw_request,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            if isinstance(result, DeleteResult):
                logger.debug(
                    "DeleteResult: deleted_count=%s namespace=%s",
                    result.deleted_count,
                    ns,
                )
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework=FRAMEWORK_LABEL,
                operation="delete_async",
                error_codes=VECTOR_ERROR_CODES,
                namespace=self._effective_namespace(ns),
                ids_count=len(ids or []),
            )
            raise


__all__ = [
    "MCPVectorService",
    "MCPVectorResult",
]
