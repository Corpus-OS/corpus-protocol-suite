# corpus_sdk/vector/framework_adapters/crewai.py
# SPDX-License-Identifier: Apache-2.0

"""
CrewAI tool adapter for Corpus Vector protocol.

This module exposes a Corpus `VectorProtocolV1` implementation as a CrewAI
`BaseTool` that performs semantic vector search with:

- Sync + async run APIs (`_run` / `_arun`)
- Optional streaming search via the shared `VectorTranslator` streaming bridge
- Proper integration with Corpus `VectorProtocolV1` via `VectorTranslator`
- Namespace + metadata filter handling (capability-aware for async flows)
- Optional client-side score thresholding
- Optional embedding function integration
- Optional Max Marginal Relevance (MMR) diversification (non-streaming)
- Optional OperationContext propagation via context_translation

Design philosophy
-----------------
- Protocol-first: CrewAI is a thin skin over Corpus vector adapters.
- All heavy lifting (backpressure, deadlines, breakers, etc.) lives in
  the underlying adapter / protocol, not here.
- This layer focuses on:
    * Translating Corpus matches → JSON-serializable payloads
    * Respecting VectorCapabilities (namespaces, filters, top_k limits)
      where available (async paths)
    * Propagating OperationContext when provided (crewai task or dict)
    * Using `VectorTranslator` for all sync/async query and streaming orchestration
"""

from __future__ import annotations

import logging
import math
from functools import cached_property, wraps
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
    Callable,
)

from pydantic import BaseModel, Field
from crewai.tools import BaseTool

from corpus_sdk.vector.vector_base import (
    VectorProtocolV1,
    QueryResult,
    QueryChunk,
    OperationContext,
    VectorCapabilities,
    # Errors
    BadRequest,
    NotSupported,
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
from corpus_sdk.vector.framework_adapters.common.vector_translation import (
    DefaultVectorFrameworkTranslator,
    VectorTranslator,
)
from corpus_sdk.core.error_context import attach_context
from corpus_sdk.core.context_translation import (
    from_crewai as ctx_from_crewai,
    from_dict as ctx_from_dict,
)

logger = logging.getLogger(__name__)

Embeddings = Sequence[Sequence[float]]
Metadata = Dict[str, Any]


# --------------------------------------------------------------------------- #
# Vector framework-utils configuration for CrewAI adapter
# --------------------------------------------------------------------------- #

VECTOR_ERROR_CODES = VectorCoercionErrorCodes(
    framework_label="crewai",
)
VECTOR_LIMITS = VectorResourceLimits()
VECTOR_FLAGS = VectorValidationFlags()
TOPK_WARNING_CONFIG = TopKWarningConfig(
    framework_label="crewai",
)


# --------------------------------------------------------------------------- #
# Error codes and decorators for richer error context
# --------------------------------------------------------------------------- #


class ErrorCodes:
    BAD_OPERATION_CONTEXT = "BAD_OPERATION_CONTEXT"
    BAD_QUERY_RESULT = "BAD_QUERY_RESULT"
    BAD_STREAM_CHUNK = "BAD_STREAM_CHUNK"
    BAD_EMBEDDINGS = "BAD_EMBEDDINGS"
    NO_EMBEDDING_FUNCTION = "NO_EMBEDDING_FUNCTION"
    EMBEDDING_ERROR = "EMBEDDING_ERROR"
    BAD_TOP_K = "BAD_TOP_K"
    FILTER_NOT_SUPPORTED = "FILTER_NOT_SUPPORTED"


def with_error_context(
    operation: str,
    **context_kwargs: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to automatically attach error context to synchronous exceptions.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                enhanced_context = dict(context_kwargs)
                attach_context(
                    exc,
                    framework="crewai",
                    operation=operation,
                    **enhanced_context,
                )
                raise

        return wrapper

    return decorator


def with_async_error_context(
    operation: str,
    **context_kwargs: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to automatically attach error context to async exceptions.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as exc:
                enhanced_context = dict(context_kwargs)
                attach_context(
                    exc,
                    framework="crewai",
                    operation=operation,
                    **enhanced_context,
                )
                raise

        return wrapper

    return decorator


# --------------------------------------------------------------------------- #
# Input schema
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# CrewAI Tool implementation
# --------------------------------------------------------------------------- #


class CorpusCrewAIVectorSearchTool(BaseTool):
    """
    CrewAI `BaseTool` implementation backed by a Corpus `VectorProtocolV1`.

    This tool performs semantic vector search against a Corpus-backed index
    and returns a list of JSON objects:

        [
            {"text": "...", "metadata": {...}, "score": 0.87, "id": "..."},
            ...
        ]

    Key behaviors
    -------------
    - Uses an embedding function (if provided) to embed query text.
    - Uses `VectorTranslator` for all sync + async queries and streaming.
    - Respects VectorCapabilities (namespaces, filters, top_k limits) in async flows.
    - Supports optional MMR re-ranking for diversity (non-streaming paths).
    - Optionally thresholds matches by score_threshold.
    - Optionally propagates OperationContext using context_translation.
    - Offers an advanced `stream_search(...)` API backed by VectorTranslator's
      streaming bridge.
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
    corpus_adapter: VectorProtocolV1
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

    # Cached capabilities (used by async flows)
    _caps: Optional[VectorCapabilities] = None

    # Optional static OperationContext for advanced scenarios
    static_operation_context: Optional[OperationContext] = None

    # ------------------------------------------------------------------ #
    # Translator setup
    # ------------------------------------------------------------------ #

    class _CrewAIVectorFrameworkTranslator(DefaultVectorFrameworkTranslator):
        """
        CrewAI-specific VectorFrameworkTranslator.

        This translator reuses the default translator for spec construction
        and context handling, but deliberately *does not* reshape core
        protocol results:

        - QueryResult is returned as-is
        - QueryChunk is returned as-is
        """

        def translate_query_result(
            self,
            result: QueryResult,
            *,
            op_ctx: OperationContext,
            framework_ctx: Optional[Any] = None,
        ) -> QueryResult:
            return result

        def translate_query_chunk(
            self,
            chunk: QueryChunk,
            *,
            op_ctx: OperationContext,
            framework_ctx: Optional[Any] = None,
        ) -> QueryChunk:
            return chunk

    @cached_property
    def _translator(self) -> VectorTranslator:
        """
        Lazily construct and cache the `VectorTranslator`.

        All sync and async search operations, including streaming, are
        performed through this translator to centralize async↔sync bridging
        and error-context handling.
        """
        framework_translator = self._CrewAIVectorFrameworkTranslator()
        return VectorTranslator(
            adapter=self.corpus_adapter,
            framework="crewai",
            translator=framework_translator,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers: capabilities / context / validation
    # ------------------------------------------------------------------ #

    async def _get_caps_async(self) -> VectorCapabilities:
        """
        Async capability fetch with caching.

        Uses the underlying adapter's capabilities() API and attaches
        rich error context on failure. Only used by async flows to
        preserve the original hardening.
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
                    ctx = ctx_from_dict(call_context)
                    if not isinstance(ctx, OperationContext):
                        raise BadRequest(
                            f"from_dict produced unsupported context type: {type(ctx).__name__}",
                            code=ErrorCodes.BAD_OPERATION_CONTEXT,
                        )
                    return ctx
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "CorpusCrewAIVectorSearchTool: from_dict context translation failed: %s",
                    exc,
                )

            # Fall back to CrewAI-specific translation
            try:
                ctx = ctx_from_crewai(call_context)
                if not isinstance(ctx, OperationContext):
                    raise BadRequest(
                        f"from_crewai produced unsupported context type: {type(ctx).__name__}",
                        code=ErrorCodes.BAD_OPERATION_CONTEXT,
                    )
                return ctx
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "CorpusCrewAIVectorSearchTool: from_crewai context translation failed: %s",
                    exc,
                )

        # Static context configured on the tool instance
        if isinstance(self.static_operation_context, OperationContext):
            return self.static_operation_context

        return None

    def _framework_ctx_for_namespace(
        self,
        namespace: Optional[str],
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build a normalized framework_ctx mapping that is consistent with the
        shared vector framework utilities.

        This ensures:
        - vector context keys (namespace, index_name, tenant_id, etc.) are
          normalized via `normalize_vector_context`
        - the resulting values are attached into a generic framework_ctx dict
          via `attach_vector_context_to_framework_ctx`
        """
        ns = self._effective_namespace(namespace)

        raw_ctx: Dict[str, Any] = {}
        if ns is not None:
            raw_ctx["namespace"] = ns
        if extra_context:
            raw_ctx.update(extra_context)

        vector_ctx = normalize_vector_context(
            raw_ctx,
            framework="crewai",
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

    @staticmethod
    def _validate_query_result(
        result: Any,
        *,
        operation: str,
    ) -> QueryResult:
        """
        Validate that the translator returned a QueryResult.
        """
        if not isinstance(result, QueryResult):
            raise BadRequest(
                f"{operation} returned unsupported type: {type(result).__name__}",
                code=ErrorCodes.BAD_QUERY_RESULT,
            )
        return result

    @staticmethod
    def _validate_stream_chunk(
        chunk: Any,
        *,
        operation: str,
    ) -> QueryChunk:
        """
        Validate that the translator returned a QueryChunk for streaming.
        """
        if not isinstance(chunk, QueryChunk):
            raise BadRequest(
                f"{operation} yielded unsupported chunk type: {type(chunk).__name__}",
                code=ErrorCodes.BAD_STREAM_CHUNK,
            )
        return chunk

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
                code=ErrorCodes.NO_EMBEDDING_FUNCTION,
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
                code=ErrorCodes.EMBEDDING_ERROR,
            )

        if not embs or len(embs) != 1:
            raise BadRequest(
                "embedding_function must return exactly one embedding for a single query",
                code=ErrorCodes.BAD_EMBEDDINGS,
            )

        return [float(x) for x in embs[0]]

    @staticmethod
    def _get_match_score(match: Any) -> float:
        """
        Robustly extract a numeric score from a match object or mapping.
        """
        if isinstance(match, Mapping):
            value = match.get("score", 0.0)
        else:
            value = getattr(match, "score", 0.0)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _get_match_vector(match: Any) -> List[float]:
        """
        Robustly extract an embedding vector from a match.

        Supports:
        - Mapping with 'embedding' key
        - Objects with `vector.vector` attribute (VectorMatch-style)
        """
        vec_raw: Any = None

        if isinstance(match, Mapping) and "embedding" in match:
            vec_raw = match.get("embedding")
        elif hasattr(match, "vector") and getattr(match, "vector") is not None:
            vec_obj = getattr(match, "vector")
            if hasattr(vec_obj, "vector"):
                vec_raw = getattr(vec_obj, "vector")

        if vec_raw is None:
            return []

        try:
            return [float(x) for x in vec_raw]
        except (TypeError, ValueError):
            return []

    def _filter_matches_by_score(
        self,
        matches: Sequence[Any],
    ) -> List[Any]:
        """
        Apply optional client-side score threshold to matches.
        """
        if self.score_threshold is None:
            return list(matches)

        threshold = float(self.score_threshold)
        return [
            m for m in matches
            if self._get_match_score(m) >= threshold
        ]

    def _match_to_payload(
        self,
        match: Any,
        *,
        return_scores: bool,
    ) -> Dict[str, Any]:
        """
        Convert a single match into a JSON-serializable payload.

        Supports both:
        - Mapping-based matches: {'metadata': {...}, 'score': ..., 'id': ..., 'text': ...}
        - VectorMatch-style objects: match.vector.metadata, match.score, match.vector.id
        """
        # Extract metadata
        if isinstance(match, Mapping):
            meta_full = dict(match.get("metadata") or {})
        else:
            v = getattr(match, "vector", None)
            meta_full = dict(getattr(v, "metadata", {}) or {}) if v is not None else {}

        # Handle metadata envelope if configured
        if self.metadata_field and self.metadata_field in meta_full:
            nested = meta_full.get(self.metadata_field) or {}
            if isinstance(nested, Mapping):
                user_meta: Dict[str, Any] = dict(nested)
            else:
                user_meta = {}
        else:
            user_meta = dict(meta_full)

        # Extract text and id
        text_value: Any = meta_full.get(self.text_field)
        text = text_value if isinstance(text_value, str) else ""

        id_value: Any
        if isinstance(match, Mapping):
            id_value = match.get("id")
        else:
            v = getattr(match, "vector", None)
            id_value = getattr(v, "id", None) if v is not None else None
            if id_value is None:
                id_value = meta_full.get(self.id_field)

        # Remove internal keys from user metadata
        user_meta.pop(self.text_field, None)
        user_meta.pop(self.id_field, None)

        payload: Dict[str, Any] = {
            "text": text,
            "metadata": user_meta,
        }
        if id_value is not None:
            payload["id"] = str(id_value)
        if return_scores:
            payload["score"] = float(self._get_match_score(match))

        return payload

    # ------------------------------------------------------------------ #
    # Query spec builders for the VectorTranslator
    # ------------------------------------------------------------------ #

    def _build_raw_query(
        self,
        embedding: Sequence[float],
        *,
        k: int,
        namespace: Optional[str],
        filter: Optional[Mapping[str, Any]],
        include_vectors: bool,
    ) -> Mapping[str, Any]:
        """
        Build a raw query mapping suitable for VectorTranslator.

        The common VectorTranslator expects a mapping with:
            - 'vector': list[float]
            - 'top_k': int
            - 'filters': optional mapping
            - 'namespace': optional str
            - 'include_metadata': bool
            - 'include_vectors': bool
        """
        ns = self._effective_namespace(namespace)
        raw: Dict[str, Any] = {
            "vector": [float(x) for x in embedding],
            "top_k": int(k),
            "filters": dict(filter) if filter is not None else None,
            "namespace": ns,
            "include_metadata": True,
            "include_vectors": bool(include_vectors),
        }
        return raw

    # ------------------------------------------------------------------ #
    # MMR utilities (manual, on top of translator results)
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
        candidate_matches: List[Any],
        k: int,
        lambda_mult: float,
    ) -> List[int]:
        """
        Improved MMR selector that respects original database scores and caches similarities.

        Args:
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
        original_scores = [self._get_match_score(m) for m in candidate_matches]
        candidate_vecs: List[List[float]] = [
            self._get_match_vector(m) for m in candidate_matches
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

    @with_async_error_context("vector_search_async")
    async def _asearch_simple(
        self,
        args: CorpusVectorSearchInput,
        *,
        caps: VectorCapabilities,
    ) -> List[Dict[str, Any]]:
        """
        Simple top-k search without MMR (async).
        """
        top_k = int(args.k or self.default_top_k)
        if caps.max_top_k is not None and top_k > caps.max_top_k:
            raise BadRequest(
                f"top_k {top_k} exceeds maximum of {caps.max_top_k}",
                code=ErrorCodes.BAD_TOP_K,
            )

        # Soft warning for extreme top_k
        warn_if_extreme_k(
            top_k,
            framework="crewai",
            op_name="vector_search_async_simple",
            warning_config=TOPK_WARNING_CONFIG,
            logger=logger,
        )

        ns = args.namespace
        ctx = self._resolve_operation_context(args.context)

        if args.filter and not caps.supports_metadata_filtering:
            raise NotSupported(
                "metadata filtering is not supported by the underlying vector adapter",
                code=ErrorCodes.FILTER_NOT_SUPPORTED,
                details={"namespace": self._effective_namespace(ns)},
            )

        query_emb = self._embed_query(args.query, embedding=args.embedding)
        raw_query = self._build_raw_query(
            embedding=query_emb,
            k=top_k,
            namespace=ns,
            filter=args.filter,
            include_vectors=False,
        )

        result = await self._translator.arun_query(
            raw_query,
            op_ctx=ctx,
            framework_ctx=self._framework_ctx_for_namespace(ns),
            mmr_config=None,
        )
        result_qr = self._validate_query_result(
            result,
            operation="VectorTranslator.arun_query",
        )

        matches_all = list(result_qr.matches or [])
        matches = self._filter_matches_by_score(matches_all)

        return_scores = bool(args.return_scores)
        return [self._match_to_payload(m, return_scores=return_scores) for m in matches]

    @with_async_error_context("vector_search_mmr_async")
    async def _asearch_with_mmr(
        self,
        args: CorpusVectorSearchInput,
        *,
        caps: VectorCapabilities,
    ) -> List[Dict[str, Any]]:
        """
        MMR-based search that first fetches candidates and then re-ranks them (async).
        """
        top_k = int(args.k or self.default_top_k)
        if top_k <= 0:
            return []

        # Soft warning for extreme user-visible k
        warn_if_extreme_k(
            top_k,
            framework="crewai",
            op_name="vector_search_async_mmr",
            warning_config=TOPK_WARNING_CONFIG,
            logger=logger,
        )

        ns = args.namespace
        ctx = self._resolve_operation_context(args.context)

        if args.filter and not caps.supports_metadata_filtering:
            raise NotSupported(
                "metadata filtering is not supported by the underlying vector adapter",
                code=ErrorCodes.FILTER_NOT_SUPPORTED,
                details={"namespace": self._effective_namespace(ns)},
            )

        lambda_mult = (
            float(args.mmr_lambda)
            if args.mmr_lambda is not None
            else float(self.mmr_lambda)
        )
        lambda_mult = max(0.0, min(1.0, lambda_mult))

        # Fetch more candidates than we will return
        fetch_k = args.fetch_k or max(top_k * 4, top_k + 5)
        if caps.max_top_k is not None and fetch_k > caps.max_top_k:
            raise BadRequest(
                f"fetch_k {fetch_k} exceeds maximum of {caps.max_top_k}",
                code=ErrorCodes.BAD_TOP_K,
            )

        # Soft warning for internal fetch_k as well
        warn_if_extreme_k(
            fetch_k,
            framework="crewai",
            op_name="vector_search_async_mmr_fetch",
            warning_config=TOPK_WARNING_CONFIG,
            logger=logger,
        )

        query_emb = self._embed_query(args.query, embedding=args.embedding)
        raw_query = self._build_raw_query(
            embedding=query_emb,
            k=int(fetch_k),
            namespace=ns,
            filter=args.filter,
            include_vectors=True,
        )

        result = await self._translator.arun_query(
            raw_query,
            op_ctx=ctx,
            framework_ctx=self._framework_ctx_for_namespace(ns),
            mmr_config=None,  # manual MMR applied here
        )
        result_qr = self._validate_query_result(
            result,
            operation="VectorTranslator.arun_query",
        )

        matches_all = list(result_qr.matches or [])
        matches_all = self._filter_matches_by_score(matches_all)

        if not matches_all:
            return []

        indices = self._mmr_select_indices(
            candidate_matches=matches_all,
            k=top_k,
            lambda_mult=lambda_mult,
        )

        return_scores = bool(args.return_scores)
        return [
            self._match_to_payload(matches_all[i], return_scores=return_scores)
            for i in indices
        ]

    @with_async_error_context("vector_search_dispatch_async")
    async def _asearch(
        self,
        args: CorpusVectorSearchInput,
    ) -> List[Dict[str, Any]]:
        """
        Unified async search entry point, dispatching to simple or MMR-based search.
        """
        caps = await self._get_caps_async()

        use_mmr = (
            bool(args.use_mmr)
            if args.use_mmr is not None
            else bool(self.use_mmr_by_default)
        )

        if use_mmr:
            return await self._asearch_with_mmr(args, caps=caps)
        return await self._asearch_simple(args, caps=caps)

    # ------------------------------------------------------------------ #
    # High-level sync search flows (simple vs MMR)
    # ------------------------------------------------------------------ #

    @with_error_context("vector_search_sync")
    def _search_simple_sync(
        self,
        args: CorpusVectorSearchInput,
    ) -> List[Dict[str, Any]]:
        """
        Simple top-k search without MMR (sync).

        Uses VectorTranslator.query directly (no direct AsyncBridge usage).
        """
        top_k = int(args.k or self.default_top_k)
        if top_k <= 0:
            return []

        # Soft warning for extreme k in sync path as well
        warn_if_extreme_k(
            top_k,
            framework="crewai",
            op_name="vector_search_sync_simple",
            warning_config=TOPK_WARNING_CONFIG,
            logger=logger,
        )

        ns = args.namespace
        ctx = self._resolve_operation_context(args.context)

        query_emb = self._embed_query(args.query, embedding=args.embedding)
        raw_query = self._build_raw_query(
            embedding=query_emb,
            k=top_k,
            namespace=ns,
            filter=args.filter,
            include_vectors=False,
        )

        result = self._translator.query(
            raw_query,
            op_ctx=ctx,
            framework_ctx=self._framework_ctx_for_namespace(ns),
            mmr_config=None,
        )
        result_qr = self._validate_query_result(
            result,
            operation="VectorTranslator.query",
        )

        matches_all = list(result_qr.matches or [])
        matches = self._filter_matches_by_score(matches_all)

        return_scores = bool(args.return_scores)
        return [self._match_to_payload(m, return_scores=return_scores) for m in matches]

    @with_error_context("vector_search_mmr_sync")
    def _search_with_mmr_sync(
        self,
        args: CorpusVectorSearchInput,
    ) -> List[Dict[str, Any]]:
        """
        MMR-based search that fetches candidates and then re-ranks them (sync).
        """
        top_k = int(args.k or self.default_top_k)
        if top_k <= 0:
            return []

        # Soft warning for user-visible k
        warn_if_extreme_k(
            top_k,
            framework="crewai",
            op_name="vector_search_sync_mmr",
            warning_config=TOPK_WARNING_CONFIG,
            logger=logger,
        )

        ns = args.namespace
        ctx = self._resolve_operation_context(args.context)

        lambda_mult = (
            float(args.mmr_lambda)
            if args.mmr_lambda is not None
            else float(self.mmr_lambda)
        )
        lambda_mult = max(0.0, min(1.0, lambda_mult))

        fetch_k = args.fetch_k or max(top_k * 4, top_k + 5)

        # Soft warning for internal fetch_k as well
        warn_if_extreme_k(
            fetch_k,
            framework="crewai",
            op_name="vector_search_sync_mmr_fetch",
            warning_config=TOPK_WARNING_CONFIG,
            logger=logger,
        )

        query_emb = self._embed_query(args.query, embedding=args.embedding)
        raw_query = self._build_raw_query(
            embedding=query_emb,
            k=int(fetch_k),
            namespace=ns,
            filter=args.filter,
            include_vectors=True,
        )

        result = self._translator.query(
            raw_query,
            op_ctx=ctx,
            framework_ctx=self._framework_ctx_for_namespace(ns),
            mmr_config=None,  # manual MMR applied here
        )
        result_qr = self._validate_query_result(
            result,
            operation="VectorTranslator.query",
        )

        matches_all = list(result_qr.matches or [])
        matches_all = self._filter_matches_by_score(matches_all)

        if not matches_all:
            return []

        indices = self._mmr_select_indices(
            candidate_matches=matches_all,
            k=top_k,
            lambda_mult=lambda_mult,
        )

        return_scores = bool(args.return_scores)
        return [
            self._match_to_payload(matches_all[i], return_scores=return_scores)
            for i in indices
        ]

    # ------------------------------------------------------------------ #
    # CrewAI Tool API: sync + async
    # ------------------------------------------------------------------ #

    @with_error_context("tool_run")
    def _run(self, **kwargs: Any) -> List[Dict[str, Any]]:
        """
        Synchronous tool entrypoint used by CrewAI.

        Internally delegates to the sync search implementation which uses
        VectorTranslator for all vector operations.
        """
        args = self.args_schema(**kwargs)  # type: ignore[arg-type]

        use_mmr = (
            bool(args.use_mmr)
            if args.use_mmr is not None
            else bool(self.use_mmr_by_default)
        )

        if use_mmr:
            return self._search_with_mmr_sync(args)
        return self._search_simple_sync(args)

    @with_async_error_context("tool_arun")
    async def _arun(self, **kwargs: Any) -> List[Dict[str, Any]]:
        """
        Async tool entrypoint for advanced users.

        CrewAI may call this in some configurations, but typical usage
        goes through `_run`.
        """
        args = self.args_schema(**kwargs)  # type: ignore[arg-type]
        return await self._asearch(args)

    # ------------------------------------------------------------------ #
    # Advanced: streaming search using VectorTranslator.query_stream
    # ------------------------------------------------------------------ #

    @with_error_context("stream_search")
    def stream_search(self, **kwargs: Any) -> Iterator[Dict[str, Any]]:
        """
        Streaming search API for advanced callers (outside CrewAI's planner).

        This method:
        - Uses the same input schema as `_run` / `_arun`.
        - Bridges the underlying async stream via VectorTranslator.query_stream.
        - Applies score thresholding per match.
        - For correctness and consistency with the shared translator layer,
          MMR is not applied to streaming results (matching the protocol
          design that MMR requires the full result set).
        """
        args = self.args_schema(**kwargs)  # type: ignore[arg-type]

        top_k = int(args.k or self.default_top_k)
        if top_k <= 0:
            # Empty iterator
            return iter(())

        # Soft warning for streaming top_k
        warn_if_extreme_k(
            top_k,
            framework="crewai",
            op_name="vector_search_stream",
            warning_config=TOPK_WARNING_CONFIG,
            logger=logger,
        )

        ns = args.namespace
        ctx = self._resolve_operation_context(args.context)
        return_scores = bool(args.return_scores)

        # Embed query
        query_emb = self._embed_query(args.query, embedding=args.embedding)

        # Build raw query for streaming; we still send top_k to the backend
        raw_query = self._build_raw_query(
            embedding=query_emb,
            k=top_k,
            namespace=ns,
            filter=args.filter,
            include_vectors=False,
        )

        # Note: We deliberately do not apply MMR here to align with the
        # shared VectorTranslator design, which does not apply MMR to
        # streaming results.
        yielded = 0
        for chunk in self._translator.query_stream(
            raw_query,
            op_ctx=ctx,
            framework_ctx=self._framework_ctx_for_namespace(ns),
        ):
            chunk_qc = self._validate_stream_chunk(
                chunk,
                operation="VectorTranslator.query_stream",
            )

            raw_matches = list(chunk_qc.matches or [])
            filtered_matches = self._filter_matches_by_score(raw_matches)

            for match in filtered_matches:
                if yielded >= top_k:
                    return
                yield self._match_to_payload(
                    match,
                    return_scores=return_scores,
                )
                yielded += 1


__all__ = [
    "CorpusVectorSearchInput",
    "CorpusCrewAIVectorSearchTool",
    "with_error_context",
    "with_async_error_context",
]
