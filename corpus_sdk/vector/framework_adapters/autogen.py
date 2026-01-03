# corpus_sdk/vector/framework_adapters/autogen.py
# SPDX-License-Identifier: Apache-2.0

"""
AutoGen adapter for Corpus Vector protocol (v1, translator-based).

This module exposes Corpus `VectorProtocolV1` implementations as
AutoGen-friendly vector stores and retrievers, with:

- Sync + async add/search APIs
- Streaming similarity search via VectorTranslator.query_stream
- Proper integration with Corpus VectorProtocolV1 via VectorTranslator
- Namespace + metadata filter handling (through the common FilterTranslator)
- Optional client-side score thresholding
- Optional embedding function integration
- Optional max marginal relevance (MMR) search (via MMRConfig on VectorTranslator)
- Simple retriever/tool wrapper for AutoGen agents

Design philosophy
-----------------
- Protocol-first: AutoGen is a thin skin over Corpus vector adapters.
- All heavy lifting (backpressure, deadlines, breakers, async↔sync, streaming)
  lives in the underlying adapter and the shared `VectorTranslator`.
- This layer focuses on:
    * Translating simple Python data structures ↔ framework-agnostic vector shapes
    * Propagating AutoGen conversation context into OperationContext
    * Converting generic matches into AutoGen-friendly document objects
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from functools import cached_property, wraps
from threading import RLock
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
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
)

from corpus_sdk.vector.vector_base import (
    VectorProtocolV1,
    OperationContext,
    BadRequest,
    NotSupported,
)
from corpus_sdk.core.context_translation import (
    from_autogen as core_ctx_from_autogen,
)
from corpus_sdk.core.error_context import attach_context
from corpus_sdk.vector.framework_adapters.common.vector_translation import (
    MMRConfig,
    VectorTranslator,
    create_vector_translator,
)
from corpus_sdk.vector.framework_adapters.common.framework_utils import (
    VectorCoercionErrorCodes,
    VectorResourceLimits,
    VectorValidationFlags,
    coerce_hits,
    warn_if_extreme_k,
)

logger = logging.getLogger(__name__)

Embeddings = Sequence[Sequence[float]]
Metadata = Dict[str, Any]

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# Error codes & coercion configuration
# --------------------------------------------------------------------------- #


class ErrorCodes:
    BAD_OPERATION_CONTEXT = "BAD_OPERATION_CONTEXT"
    BAD_TRANSLATED_QUERY_RESULT = "BAD_TRANSLATED_QUERY_RESULT"
    BAD_TRANSLATED_STREAM_CHUNK = "BAD_TRANSLATED_STREAM_CHUNK"
    BAD_UPSERT_RESULT = "BAD_UPSERT_RESULT"
    BAD_DELETE_REQUEST = "BAD_DELETE_REQUEST"
    BAD_EMBEDDINGS = "BAD_EMBEDDINGS"
    BAD_METADATA = "BAD_METADATA"
    BAD_IDS = "BAD_IDS"
    NO_EMBEDDING_FUNCTION = "NO_EMBEDDING_FUNCTION"
    BAD_TEXTS = "BAD_TEXTS"
    UNKNOWN_VECTOR_DIMENSION = "UNKNOWN_VECTOR_DIMENSION"
    VECTOR_DIM_MISMATCH = "VECTOR_DIM_MISMATCH"


# Bundle of error codes for vector coercion utilities
VECTOR_COERCION_ERROR_CODES: VectorCoercionErrorCodes = VectorCoercionErrorCodes(
    invalid_vector_result="INVALID_VECTOR_RESULT",
    invalid_hit_result="INVALID_VECTOR_HIT_RESULT",
    empty_result="EMPTY_VECTOR_RESULT",
    conversion_error="VECTOR_CONVERSION_ERROR",
    score_out_of_range="VECTOR_SCORE_OUT_OF_RANGE",
    vector_dimension_exceeded="VECTOR_DIMENSION_EXCEEDED",
    vector_norm_invalid="VECTOR_VECTOR_NORM_INVALID",
    framework_label="autogen",
)

# Defaults: we are mostly advisory here, not enforcing strict limits
VECTOR_LIMITS: VectorResourceLimits = VectorResourceLimits()
VECTOR_FLAGS: VectorValidationFlags = VectorValidationFlags()


def _coerce_hits_safe(result: Any) -> List[Mapping[str, Any]]:
    """
    Thin wrapper around common coerce_hits with EMPTY_RESULT treated as [].

    This allows us to use the shared vector coercion logic while preserving
    the existing behavior of returning an empty list when there are no hits,
    instead of raising.
    """
    try:
        return coerce_hits(
            result,
            framework="autogen",
            error_codes=VECTOR_COERCION_ERROR_CODES,
            limits=VECTOR_LIMITS,
            flags=VECTOR_FLAGS,
            logger=logger,
        )
    except ValueError as exc:
        msg = str(exc)
        if VECTOR_COERCION_ERROR_CODES.empty_result in msg:
            return []
        raise


def _warn_if_extreme_k(k: int, op_name: str) -> None:
    """
    Shared wrapper so all k-based operations emit consistent soft warnings.
    """
    warn_if_extreme_k(
        k,
        framework="autogen",
        op_name=op_name,
        logger=logger,
    )


# --------------------------------------------------------------------------- #
# Event-loop guard
# --------------------------------------------------------------------------- #


def _ensure_not_in_event_loop(sync_api_name: str) -> None:
    """
    Prevent deadlocks from calling sync APIs in active event loops.

    This is a defensive guard for environments where users might
    accidentally call sync vector APIs from async frameworks.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop -> safe to use sync API.
        return
    raise RuntimeError(
        f"{sync_api_name} was called from inside an active asyncio event loop. "
        f"Use the async variant instead (e.g. 'a{sync_api_name}')."
    )


# --------------------------------------------------------------------------- #
# Error decorators with operation-specific context builders
# --------------------------------------------------------------------------- #


def _build_add_context(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build additional error-context fields for add_* style operations.

    Extracts:
    - texts_count
    - vectors_count (alias of texts_count for vector ops)
    - total_content_chars (sum of source text lengths, best-effort)
    - metadatas_count
    - ids_count
    - has_embeddings
    """
    extra: Dict[str, Any] = {}
    try:
        texts = None
        if len(args) >= 2:
            texts = args[1]
        else:
            texts = kwargs.get("texts") or kwargs.get("documents")

        texts_list: Optional[List[Any]] = None
        if texts is not None:
            try:
                texts_list = list(texts)
                extra["texts_count"] = len(texts_list)
                extra["vectors_count"] = len(texts_list)
                # Best-effort sum of text lengths if they are strings.
                total_chars = 0
                for t in texts_list:
                    if isinstance(t, str):
                        total_chars += len(t)
                    else:
                        total_chars += len(str(t))
                extra["total_content_chars"] = total_chars
            except Exception:
                # Last-resort: treat as a single payload
                extra["texts_count"] = 1
                extra["vectors_count"] = 1

        metadatas = kwargs.get("metadatas")
        if isinstance(metadatas, list):
            extra["metadatas_count"] = len(metadatas)

        ids = kwargs.get("ids")
        if isinstance(ids, list):
            extra["ids_count"] = len(ids)

        if "embeddings" in kwargs and kwargs["embeddings"] is not None:
            extra["has_embeddings"] = True
    except Exception:
        # Metrics must never break error-context attachment
        pass
    return extra


def _build_search_context(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build additional error-context fields for search/query/MMR operations.

    Extracts:
    - query_chars (when a text query is present)
    - total_content_chars (alias of query_chars for vector search operations)
    - k
    - fetch_k
    - lambda_mult
    """
    extra: Dict[str, Any] = {}
    try:
        query = None
        if len(args) >= 2:
            query = args[1]
        else:
            query = kwargs.get("query")

        if isinstance(query, str):
            qc = len(query)
            extra["query_chars"] = qc
            extra["total_content_chars"] = qc

        k = kwargs.get("k")
        if isinstance(k, int):
            extra["k"] = k

        fetch_k = kwargs.get("fetch_k")
        if isinstance(fetch_k, int):
            extra["fetch_k"] = fetch_k

        lambda_mult = kwargs.get("lambda_mult")
        if isinstance(lambda_mult, (int, float)):
            extra["lambda_mult"] = float(lambda_mult)
    except Exception:
        pass
    return extra


def _build_delete_context(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build additional error-context fields for delete operations.

    Extracts:
    - ids_count
    - has_filter
    """
    extra: Dict[str, Any] = {}
    try:
        ids = kwargs.get("ids")
        if isinstance(ids, list):
            extra["ids_count"] = len(ids)

        if "filter" in kwargs:
            extra["has_filter"] = kwargs["filter"] is not None
    except Exception:
        pass
    return extra


def _build_vectorize_context(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build additional error-context fields for vectorize operations (__call__/acall).

    Extracts:
    - vectors_count
    - total_content_chars
    - has_empty_items
    """
    extra: Dict[str, Any] = {}
    try:
        payload = None
        if len(args) >= 2:
            payload = args[1]
        else:
            payload = kwargs.get("texts")

        if isinstance(payload, str):
            extra["vectors_count"] = 1
            extra["total_content_chars"] = len(payload)
            extra["has_empty_items"] = (not payload.strip())
            return extra

        batch = list(payload or [])
        extra["vectors_count"] = len(batch)
        total_chars = 0
        has_empty = False
        for t in batch:
            s = t if isinstance(t, str) else str(t)
            total_chars += len(s)
            if not s.strip():
                has_empty = True
        extra["total_content_chars"] = total_chars
        extra["has_empty_items"] = has_empty
    except Exception:
        pass
    return extra


_OPERATION_CONTEXT_BUILDERS: Dict[
    str, Callable[[Tuple[Any, ...], Dict[str, Any]], Dict[str, Any]]
] = {
    # Add operations
    "add_texts_sync": _build_add_context,
    "add_texts_async": _build_add_context,
    "add_documents_sync": _build_add_context,
    "add_documents_async": _build_add_context,
    "from_texts_sync": _build_add_context,
    "from_documents_sync": _build_add_context,
    # Search/query/MMR operations
    "similarity_search_sync": _build_search_context,
    "similarity_search_async": _build_search_context,
    "similarity_search_with_score_sync": _build_search_context,
    "similarity_search_with_score_async": _build_search_context,
    "similarity_search_stream_sync": _build_search_context,
    "similarity_search_stream_async": _build_search_context,
    "mmr_search_sync": _build_search_context,
    "mmr_search_async": _build_search_context,
    "query_sync": _build_search_context,
    "query_async": _build_search_context,
    # Vectorize operations
    "call_sync": _build_vectorize_context,
    "call_async": _build_vectorize_context,
    # Delete operations
    "delete_sync": _build_delete_context,
    "delete_async": _build_delete_context,
}


def _build_dynamic_error_context(
    operation: str,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    base_context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a rich dynamic error context for vector operations.

    This function is shared by both sync and async decorators so the
    behavior is identical across both paths.
    """
    extra: Dict[str, Any] = dict(base_context)
    try:
        # Generic parameters available for most operations
        if "k" in kwargs and "k" not in extra:
            extra["k"] = kwargs["k"]
        if "namespace" in kwargs:
            extra["namespace"] = kwargs["namespace"]
        if "filter" in kwargs and "has_filter" not in extra:
            extra["has_filter"] = kwargs["filter"] is not None

        # Self-introspection: default namespace & framework_version & dim hint
        if args:
            self_obj = args[0]
            default_ns = getattr(self_obj, "namespace", None)
            if default_ns is not None:
                extra.setdefault("default_namespace", default_ns)
            fv = getattr(self_obj, "_framework_version", None)
            if fv is not None:
                extra.setdefault("framework_version", fv)
            dh = getattr(self_obj, "_vector_dim_hint", None)
            if isinstance(dh, int):
                extra.setdefault("vector_dimension_hint", dh)

        # Operation-specific builder hook (best-effort)
        builder = _OPERATION_CONTEXT_BUILDERS.get(operation)
        if builder is not None:
            op_specific = builder(args, kwargs)
            if op_specific:
                extra.update(op_specific)
    except Exception:
        # Error-context enrichment must never be fatal
        pass

    return extra


def with_error_context(operation: str, **context_kwargs: Any):
    """
    Decorator to automatically attach error context to sync exceptions.

    Keeps the wrapper itself small and delegates dynamic context building
    to `_build_dynamic_error_context` plus operation-specific builders.
    """

    def decorator(fn: T) -> T:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                extra = _build_dynamic_error_context(
                    operation,
                    args,
                    kwargs,
                    base_context=dict(context_kwargs),
                )
                attach_context(
                    exc,
                    framework="autogen",
                    operation=f"vector_{operation}",
                    error_codes=VECTOR_COERCION_ERROR_CODES,
                    **extra,
                )
                raise

        return wrapper  # type: ignore[return-value]

    return decorator


def with_async_error_context(operation: str, **context_kwargs: Any):
    """
    Decorator to automatically attach error context to async exceptions.

    Keeps the wrapper itself small and delegates dynamic context building
    to `_build_dynamic_error_context` plus operation-specific builders.
    """

    def decorator(fn: T) -> T:
        @wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any):
            try:
                return await fn(*args, **kwargs)
            except Exception as exc:
                extra = _build_dynamic_error_context(
                    operation,
                    args,
                    kwargs,
                    base_context=dict(context_kwargs),
                )
                attach_context(
                    exc,
                    framework="autogen",
                    operation=f"vector_{operation}",
                    error_codes=VECTOR_COERCION_ERROR_CODES,
                    **extra,
                )
                raise

        return wrapper  # type: ignore[return-value]

    return decorator


# --------------------------------------------------------------------------- #
# Simple AutoGen document representation
# --------------------------------------------------------------------------- #


@dataclass
class AutoGenDocument:
    """
    Simple document representation for AutoGen integrations.

    Attributes
    ----------
    page_content:
        The primary text content of the document.

    metadata:
        Arbitrary key-value metadata associated with the document.
    """

    page_content: str
    metadata: Dict[str, Any]


# --------------------------------------------------------------------------- #
# Vector store based on VectorTranslator
# --------------------------------------------------------------------------- #


class CorpusAutoGenVectorStore:
    """
    AutoGen-friendly vector store backed by a Corpus `VectorProtocolV1`
    via the common `VectorTranslator`.

    This class is a thin integration layer:

    - Raw texts + embeddings are mapped to generic vector document dicts
      that the `DefaultVectorFrameworkTranslator` understands.
    - Similarity search calls map to `VectorTranslator.query(...)`.
    - Maximal Marginal Relevance (MMR) search is delegated to the translator
      via `MMRConfig`.
    - Namespaces + metadata filters are handled by the translator's
      `FilterTranslator`.
    - Sync/async and streaming orchestration is fully handled inside
      `VectorTranslator`; this module does not use AsyncBridge or
      SyncStreamBridge directly.

    Attributes
    ----------
    corpus_adapter:
        Underlying Corpus vector adapter implementing `VectorProtocolV1`.

    namespace:
        Default namespace for all operations, when not explicitly overridden.

    id_field:
        Metadata key under which the logical document ID will be stored.

    text_field:
        Metadata key under which the document text/page_content is stored.

    metadata_field:
        Optional "envelope" key under which user metadata dict is stored.
        If None, metadata is stored directly on the vector metadata.

    score_threshold:
        Optional minimum similarity score required for a match to be returned.
        Applied client-side on the generic record's "score" field.

    default_top_k:
        Default K used in similarity search when the caller does not specify.

    embedding_function:
        Optional function used to embed raw texts into vectors.

        Signature:
            embedding_function(texts: List[str]) -> List[List[float]]

    async_embedding_function:
        Optional async embedding function for use in async flows.

        Signature:
            async_embedding_function(texts: List[str]) -> Awaitable[List[List[float]]]

    own_adapter:
        If True, this store owns the lifecycle of corpus_adapter and will attempt
        to close it during close()/aclose(). Defaults to False for backwards
        compatibility and to preserve caller-managed lifecycles.
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
        default_top_k: int = 4,
        embedding_function: Optional[Callable[[List[str]], Embeddings]] = None,
        async_embedding_function: Optional[
            Callable[[List[str]], Awaitable[Embeddings]]
        ] = None,
        framework_version: Optional[str] = None,
        own_adapter: bool = False,
    ) -> None:
        self.corpus_adapter: VectorProtocolV1 = corpus_adapter
        self.namespace = namespace

        self.id_field = id_field
        self.text_field = text_field
        self.metadata_field = metadata_field

        self.score_threshold = score_threshold
        self.default_top_k = default_top_k

        # Embedding integration
        self.embedding_function = embedding_function
        self.async_embedding_function = async_embedding_function

        # Observability / context
        self._framework_version: Optional[str] = framework_version

        # Vector dimension hint (best-effort, thread-safe first-write-wins)
        self._vector_dim_hint: Optional[int] = None
        self._dim_lock: RLock = RLock()

        # Lifecycle ownership for underlying adapter (opt-in)
        self._own_adapter: bool = bool(own_adapter)

    # ------------------------------------------------------------------ #
    # Translator (lazy, cached)
    # ------------------------------------------------------------------ #

    @cached_property
    def _translator(self) -> VectorTranslator:
        """
        Lazily construct and cache the `VectorTranslator`.

        Uses the common `create_vector_translator`, which will:
        - Use any registered "autogen" translator factory if present, or
        - Fall back to `DefaultVectorFrameworkTranslator`.
        """
        return create_vector_translator(
            adapter=self.corpus_adapter,
            framework="autogen",
            translator=None,
        )

    # ------------------------------------------------------------------ #
    # Standardized context building (core + framework + orchestrator)
    # ------------------------------------------------------------------ #

    def _build_core_context(
        self,
        *,
        conversation: Optional[Any],
        extra_context: Optional[Mapping[str, Any]],
    ) -> Optional[OperationContext]:
        """
        Build an OperationContext from an AutoGen-style conversation plus
        optional extra context.

        - Returns None only when there is no input context at all.
        - Raises (with attached context) on translation failures.
        """
        extra: Dict[str, Any] = dict(extra_context or {})
        if conversation is None and not extra:
            return None

        try:
            ctx = core_ctx_from_autogen(
                conversation,
                framework_version=self._framework_version,
                **extra,
            )
        except Exception as exc:
            attach_context(
                exc,
                framework="autogen",
                operation="vector_context_translation",
                framework_version=self._framework_version,
                error_codes=VECTOR_COERCION_ERROR_CODES,
            )
            raise

        if not isinstance(ctx, OperationContext):
            err = BadRequest(
                f"from_autogen produced unsupported context type: {type(ctx).__name__}",
                code=ErrorCodes.BAD_OPERATION_CONTEXT,
            )
            attach_context(
                err,
                framework="autogen",
                operation="vector_context_translation",
                framework_version=self._framework_version,
                error_codes=VECTOR_COERCION_ERROR_CODES,
                produced_type=type(ctx).__name__,
            )
            raise err

        return ctx

    def _build_framework_context(
        self,
        core_ctx: Optional[OperationContext],
        *,
        operation: str,
        namespace: Optional[str],
    ) -> Mapping[str, Any]:
        """
        Build framework_ctx metadata for the vector translator.

        This remains stable and framework-only:
        - namespace (effective)
        - framework_version (if present)
        - operation name (for observability/debugging)
        """
        ns = self._effective_namespace(namespace)
        ctx: Dict[str, Any] = {"operation": str(operation)}
        if ns is not None:
            ctx["namespace"] = ns
        if self._framework_version is not None:
            ctx["framework_version"] = self._framework_version
        return ctx

    def _build_contexts(
        self,
        *,
        operation: str,
        conversation: Optional[Any],
        extra_context: Optional[Mapping[str, Any]],
        namespace: Optional[str],
    ) -> Tuple[Optional[OperationContext], Mapping[str, Any]]:
        """
        Orchestrate both core OperationContext and framework_ctx building.

        This is the single entrypoint used by all public APIs so behavior is
        consistent everywhere.
        """
        core = self._build_core_context(conversation=conversation, extra_context=extra_context)
        fw = self._build_framework_context(core, operation=operation, namespace=namespace)
        return core, fw

    # Backwards-compatible alias for existing internal callers (kept intentionally).
    def _build_ctx(
        self,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Optional[OperationContext]:
        """
        Backwards-compatible wrapper around _build_core_context.

        (Kept to avoid changing internal call patterns elsewhere; public
        methods use _build_contexts for standardized behavior.)
        """
        return self._build_core_context(conversation=conversation, extra_context=extra_context)

    # ------------------------------------------------------------------ #
    # Namespace helper
    # ------------------------------------------------------------------ #

    def _effective_namespace(self, namespace: Optional[str]) -> Optional[str]:
        """
        Resolve namespace using explicit override or store default.
        """
        return namespace if namespace is not None else self.namespace

    # ------------------------------------------------------------------ #
    # Dimension hint helpers
    # ------------------------------------------------------------------ #

    def _update_dim_hint(self, embedding_dim: Optional[int]) -> None:
        """
        Thread-safe, best-effort update of the vector dimension hint.

        The first successful write wins; subsequent calls are no-ops.
        """
        if embedding_dim is None or embedding_dim <= 0:
            return
        if self._vector_dim_hint is not None:
            return
        with self._dim_lock:
            if self._vector_dim_hint is None:
                self._vector_dim_hint = int(embedding_dim)

    def _maybe_check_dim(self, vec: Sequence[float], *, where: str) -> None:
        """
        Validate vector dimensionality against the current hint (if set).

        This is a lightweight consistency check that prevents shape drift
        (e.g., mixed-embedding models) from silently producing inconsistent
        results.
        """
        hint = self._vector_dim_hint
        if hint is None:
            return
        if len(vec) != hint:
            err = BadRequest(
                f"vector dimension mismatch in {where}: got {len(vec)}, expected {hint}",
                code=ErrorCodes.VECTOR_DIM_MISMATCH,
                details={"got": len(vec), "expected": hint, "where": where},
            )
            attach_context(
                err,
                framework="autogen",
                operation=f"vector_{where}",
                error_codes=VECTOR_COERCION_ERROR_CODES,
                vector_dimension_hint=hint,
            )
            raise err

    def _zero_vector(self) -> List[float]:
        """
        Construct a deterministic zero vector based on known vector dimension.

        Raises if vector dimension is unknown.
        """
        dim = self._vector_dim_hint
        if dim is None or dim <= 0:
            err = BadRequest(
                "vector dimension is unknown; cannot produce deterministic zero vectors",
                code=ErrorCodes.UNKNOWN_VECTOR_DIMENSION,
            )
            attach_context(
                err,
                framework="autogen",
                operation="vector_zero_vector",
                error_codes=VECTOR_COERCION_ERROR_CODES,
                vector_dimension_hint=dim,
            )
            raise err
        return [0.0] * int(dim)

    # ------------------------------------------------------------------ #
    # Normalization helpers
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

        raise BadRequest(
            f"metadatas length {len(metadatas)} does not match texts length {n}",
            code=ErrorCodes.BAD_METADATA,
            details={"texts": n, "metadatas": len(metadatas)},
        )

    def _normalize_ids(
        self,
        n: int,
        ids: Optional[List[str]],
    ) -> List[str]:
        """
        Normalize IDs list to length n.

        Behavior:
        - If ids is None: generate simple string IDs based on index.
          (Callers can override by passing explicit IDs.)
        - If len(ids) == n: coerce to str list.
        - Else: raise BadRequest.
        """
        if ids is None:
            # We avoid UUID generation here to keep this adapter deterministic
            # and leave any strong ID semantics to upstream callers.
            return [f"doc-{i}" for i in range(n)]

        if len(ids) != n:
            raise BadRequest(
                f"ids length {len(ids)} does not match texts length {n}",
                code=ErrorCodes.BAD_IDS,
                details={"texts": n, "ids": len(ids)},
            )
        return [str(i) for i in ids]

    # ------------------------------------------------------------------ #
    # Embedding helpers (sync + async)
    # ------------------------------------------------------------------ #

    def _validate_embedding_batch_shape(self, embeddings: Embeddings, *, where: str) -> None:
        """
        Validate embedding batch shapes for internal consistency and against
        any established dimension hint.

        - Ensures all vectors have the same length (if possible).
        - Enforces current hint, if set.
        - Updates hint (first-write-wins) based on the first vector.
        """
        try:
            if not embeddings:
                return
            first = embeddings[0]
            if first is None:
                return
            dim0 = len(first)
            if dim0 > 0:
                self._update_dim_hint(dim0)
            # Enforce against hint if present
            if self._vector_dim_hint is not None:
                self._maybe_check_dim(first, where=where)
            # Ensure uniformity across batch (best-effort)
            for idx, v in enumerate(embeddings):
                if v is None:
                    continue
                if len(v) != dim0:
                    err = BadRequest(
                        f"embedding batch contains inconsistent vector sizes: "
                        f"index 0 has {dim0}, index {idx} has {len(v)}",
                        code=ErrorCodes.VECTOR_DIM_MISMATCH,
                        details={"first_dim": dim0, "mismatch_dim": len(v), "index": idx},
                    )
                    attach_context(
                        err,
                        framework="autogen",
                        operation=f"vector_{where}",
                        error_codes=VECTOR_COERCION_ERROR_CODES,
                        vector_dimension_hint=self._vector_dim_hint,
                    )
                    raise err
        except Exception:
            # If validation itself fails unexpectedly, we allow the original flow
            # to proceed. Any downstream coercion/translator checks will still run.
            raise

    def _ensure_embeddings(
        self,
        texts: List[str],
        embeddings: Optional[Embeddings],
    ) -> Embeddings:
        """
        Ensure embeddings are available for a batch of texts (sync).

        Behavior:
        - If embeddings are provided, verify length and validate shapes.
        - Else, if `embedding_function` is set, compute embeddings and validate shapes.
        - Else, raise NotSupported.
        """
        if embeddings is not None:
            if len(embeddings) != len(texts):
                raise BadRequest(
                    f"embeddings length {len(embeddings)} does not match texts length {len(texts)}",
                    code=ErrorCodes.BAD_EMBEDDINGS,
                    details={"texts": len(texts), "embeddings": len(embeddings)},
                )
            self._validate_embedding_batch_shape(embeddings, where="embed_documents")
            return embeddings

        if self.embedding_function is None:
            raise NotSupported(
                "No embedding_function configured; caller must supply embeddings",
                code=ErrorCodes.NO_EMBEDDING_FUNCTION,
                details={"texts": len(texts)},
            )

        try:
            computed = self.embedding_function(texts)
        except Exception as exc:
            err = BadRequest(
                f"embedding_function failed: {exc}",
                code=ErrorCodes.BAD_EMBEDDINGS,
            )
            attach_context(
                err,
                framework="autogen",
                operation="vector_embed_documents",
                framework_version=self._framework_version,
                error_codes=VECTOR_COERCION_ERROR_CODES,
            )
            raise err

        if len(computed) != len(texts):
            raise BadRequest(
                f"embedding_function returned {len(computed)} embeddings for {len(texts)} texts",
                code=ErrorCodes.BAD_EMBEDDINGS,
            )

        self._validate_embedding_batch_shape(computed, where="embed_documents")
        return computed

    async def _ensure_embeddings_async(
        self,
        texts: List[str],
        embeddings: Optional[Embeddings],
    ) -> Embeddings:
        """
        Async-safe version of _ensure_embeddings.

        Behavior:
        - If embeddings are provided, verify length and validate shapes.
        - Else, if async_embedding_function is set, await it and validate shapes.
        - Else, if embedding_function is set, run it in a worker thread and validate shapes.
        - Else, raise NotSupported.
        """
        if embeddings is not None:
            if len(embeddings) != len(texts):
                raise BadRequest(
                    f"embeddings length {len(embeddings)} does not match texts length {len(texts)}",
                    code=ErrorCodes.BAD_EMBEDDINGS,
                    details={"texts": len(texts), "embeddings": len(embeddings)},
                )
            self._validate_embedding_batch_shape(embeddings, where="embed_documents_async")
            return embeddings

        if self.async_embedding_function is not None:
            try:
                computed = await self.async_embedding_function(texts)
            except Exception as exc:
                err = BadRequest(
                    f"async_embedding_function failed: {exc}",
                    code=ErrorCodes.BAD_EMBEDDINGS,
                )
                attach_context(
                    err,
                    framework="autogen",
                    operation="vector_embed_documents_async",
                    framework_version=self._framework_version,
                    error_codes=VECTOR_COERCION_ERROR_CODES,
                )
                raise err
        else:
            if self.embedding_function is None:
                raise NotSupported(
                    "No embedding_function/async_embedding_function configured; caller must supply embeddings",
                    code=ErrorCodes.NO_EMBEDDING_FUNCTION,
                    details={"texts": len(texts)},
                )
            try:
                computed = await asyncio.to_thread(self.embedding_function, texts)
            except Exception as exc:
                err = BadRequest(
                    f"embedding_function failed: {exc}",
                    code=ErrorCodes.BAD_EMBEDDINGS,
                )
                attach_context(
                    err,
                    framework="autogen",
                    operation="vector_embed_documents_async",
                    framework_version=self._framework_version,
                    error_codes=VECTOR_COERCION_ERROR_CODES,
                )
                raise err

        if len(computed) != len(texts):
            raise BadRequest(
                f"embedding function returned {len(computed)} embeddings for {len(texts)} texts",
                code=ErrorCodes.BAD_EMBEDDINGS,
            )

        self._validate_embedding_batch_shape(computed, where="embed_documents_async")
        return computed

    def _embed_query(
        self,
        query: str,
        *,
        embedding: Optional[Sequence[float]] = None,
    ) -> List[float]:
        """
        Ensure a single query embedding is available (sync).

        Behavior:
        - If `embedding` is provided, use it (and validate dimension vs hint).
        - Else, if query is empty/whitespace:
            * return deterministic zero vector if dimension is known
            * otherwise raise (existing behavior preserved when dim unknown)
        - Else, if `embedding_function` is set, compute embedding for [query].
        - Else, raise NotSupported.
        """
        if embedding is not None:
            vec = [float(x) for x in embedding]
            if not vec:
                err = BadRequest(
                    "provided query embedding cannot be empty",
                    code=ErrorCodes.BAD_EMBEDDINGS,
                )
                attach_context(
                    err,
                    framework="autogen",
                    operation="vector_embed_query",
                    query=query,
                    framework_version=self._framework_version,
                    error_codes=VECTOR_COERCION_ERROR_CODES,
                    vector_dimension_hint=self._vector_dim_hint,
                )
                raise err
            self._update_dim_hint(len(vec))
            self._maybe_check_dim(vec, where="embed_query")
            return vec

        if not query or not str(query).strip():
            # Deterministic empty-input handling: return zeros if dimension is known.
            if self._vector_dim_hint is not None:
                return self._zero_vector()

            err = BadRequest(
                "query cannot be empty when no embedding is supplied",
                code=ErrorCodes.BAD_TEXTS,
            )
            attach_context(
                err,
                framework="autogen",
                operation="vector_embed_query",
                query=query,
                framework_version=self._framework_version,
                error_codes=VECTOR_COERCION_ERROR_CODES,
            )
            raise err

        if self.embedding_function is None:
            exc = NotSupported(
                "No embedding_function configured; caller must supply query embedding",
                code=ErrorCodes.NO_EMBEDDING_FUNCTION,
            )
            attach_context(
                exc,
                framework="autogen",
                operation="vector_embed_query",
                query=query,
                framework_version=self._framework_version,
                error_codes=VECTOR_COERCION_ERROR_CODES,
            )
            raise exc

        try:
            embs = self.embedding_function([query])
        except Exception as exc:
            err = BadRequest(
                f"embedding_function failed for query: {exc}",
                code=ErrorCodes.BAD_EMBEDDINGS,
            )
            attach_context(
                err,
                framework="autogen",
                operation="vector_embed_query",
                query=query,
                framework_version=self._framework_version,
                error_codes=VECTOR_COERCION_ERROR_CODES,
            )
            raise err

        if not embs or len(embs) != 1:
            err = BadRequest(
                "embedding_function must return exactly one embedding for a single query",
                code=ErrorCodes.BAD_EMBEDDINGS,
            )
            attach_context(
                err,
                framework="autogen",
                operation="vector_embed_query",
                query=query,
                framework_version=self._framework_version,
                error_codes=VECTOR_COERCION_ERROR_CODES,
            )
            raise err

        vec = [float(x) for x in embs[0]]
        if not vec:
            err = BadRequest(
                "embedding_function returned an empty vector for query",
                code=ErrorCodes.BAD_EMBEDDINGS,
            )
            attach_context(
                err,
                framework="autogen",
                operation="vector_embed_query",
                query=query,
                framework_version=self._framework_version,
                error_codes=VECTOR_COERCION_ERROR_CODES,
            )
            raise err

        self._update_dim_hint(len(vec))
        self._maybe_check_dim(vec, where="embed_query")
        return vec

    async def _embed_query_async(
        self,
        query: str,
        *,
        embedding: Optional[Sequence[float]] = None,
    ) -> List[float]:
        """
        Async-safe query embedding helper.

        Behavior:
        - If `embedding` is provided, use it (and validate dimension vs hint).
        - Else, if query is empty/whitespace:
            * return deterministic zero vector if dimension is known
            * otherwise raise (existing behavior preserved when dim unknown)
        - Else, if `async_embedding_function` is set, await it.
        - Else, if `embedding_function` is set, run it in a worker thread.
        - Else, raise NotSupported.
        """
        if embedding is not None:
            vec = [float(x) for x in embedding]
            if not vec:
                err = BadRequest(
                    "provided query embedding cannot be empty",
                    code=ErrorCodes.BAD_EMBEDDINGS,
                )
                attach_context(
                    err,
                    framework="autogen",
                    operation="vector_embed_query_async",
                    query=query,
                    framework_version=self._framework_version,
                    error_codes=VECTOR_COERCION_ERROR_CODES,
                    vector_dimension_hint=self._vector_dim_hint,
                )
                raise err
            self._update_dim_hint(len(vec))
            self._maybe_check_dim(vec, where="embed_query_async")
            return vec

        if not query or not str(query).strip():
            # Deterministic empty-input handling: return zeros if dimension is known.
            if self._vector_dim_hint is not None:
                return self._zero_vector()

            err = BadRequest(
                "query cannot be empty when no embedding is supplied",
                code=ErrorCodes.BAD_TEXTS,
            )
            attach_context(
                err,
                framework="autogen",
                operation="vector_embed_query_async",
                query=query,
                framework_version=self._framework_version,
                error_codes=VECTOR_COERCION_ERROR_CODES,
            )
            raise err

        if self.async_embedding_function is not None:
            try:
                embs = await self.async_embedding_function([query])
            except Exception as exc:
                err = BadRequest(
                    f"async_embedding_function failed for query: {exc}",
                    code=ErrorCodes.BAD_EMBEDDINGS,
                )
                attach_context(
                    err,
                    framework="autogen",
                    operation="vector_embed_query_async",
                    query=query,
                    framework_version=self._framework_version,
                    error_codes=VECTOR_COERCION_ERROR_CODES,
                )
                raise err
        else:
            if self.embedding_function is None:
                exc = NotSupported(
                    "No embedding_function/async_embedding_function configured; caller must supply query embedding",
                    code=ErrorCodes.NO_EMBEDDING_FUNCTION,
                )
                attach_context(
                    exc,
                    framework="autogen",
                    operation="vector_embed_query_async",
                    query=query,
                    framework_version=self._framework_version,
                    error_codes=VECTOR_COERCION_ERROR_CODES,
                )
                raise exc
            try:
                embs = await asyncio.to_thread(self.embedding_function, [query])
            except Exception as exc:
                err = BadRequest(
                    f"embedding_function failed for query: {exc}",
                    code=ErrorCodes.BAD_EMBEDDINGS,
                )
                attach_context(
                    err,
                    framework="autogen",
                    operation="vector_embed_query_async",
                    query=query,
                    framework_version=self._framework_version,
                    error_codes=VECTOR_COERCION_ERROR_CODES,
                )
                raise err

        if not embs or len(embs) != 1:
            err = BadRequest(
                "embedding function must return exactly one embedding for a single query",
                code=ErrorCodes.BAD_EMBEDDINGS,
            )
            attach_context(
                err,
                framework="autogen",
                operation="vector_embed_query_async",
                query=query,
                framework_version=self._framework_version,
                error_codes=VECTOR_COERCION_ERROR_CODES,
            )
            raise err

        vec = [float(x) for x in embs[0]]
        if not vec:
            err = BadRequest(
                "embedding function returned an empty vector for query",
                code=ErrorCodes.BAD_EMBEDDINGS,
            )
            attach_context(
                err,
                framework="autogen",
                operation="vector_embed_query_async",
                query=query,
                framework_version=self._framework_version,
                error_codes=VECTOR_COERCION_ERROR_CODES,
            )
            raise err

        self._update_dim_hint(len(vec))
        self._maybe_check_dim(vec, where="embed_query_async")
        return vec

    # ------------------------------------------------------------------ #
    # Upsert result validation (uses ErrorCodes.BAD_UPSERT_RESULT)
    # ------------------------------------------------------------------ #

    def _validate_upsert_result(
        self,
        result: Any,
        total_texts: int,
        namespace: Optional[str],
    ) -> None:
        """
        Best-effort validation of an upsert result.

        We look at the *translated* upsert result (typically a mapping like
        {"ids": [...], "count": int}) and ensure that at least one record was
        successfully written.

        If it appears that all records failed, we raise a BadRequest with
        ErrorCodes.BAD_UPSERT_RESULT.

        This is intentionally soft:
        - If the shape of the result is unexpected, we just log and do not raise.
        """
        if total_texts <= 0:
            return
        if result is None:
            logger.debug("Upsert result is None; skipping validation")
            return

        if not isinstance(result, Mapping):
            logger.debug(
                "Skipping upsert result validation for non-mapping result: %r",
                type(result).__name__,
            )
            return

        try:
            if "count" in result:
                count = int(result.get("count") or 0)
            else:
                ids = result.get("ids") or []
                if isinstance(ids, (list, tuple)):
                    count = len(ids)
                else:
                    count = 0
        except Exception:
            logger.debug(
                "Skipping upsert result validation for unexpected shape",
                exc_info=True,
            )
            return

        if count <= 0:
            raise BadRequest(
                "All documents failed to upsert into vector index",
                code=ErrorCodes.BAD_UPSERT_RESULT,
                details={"namespace": namespace, "total": total_texts},
            )

    # ------------------------------------------------------------------ #
    # Translation helpers: match records → AutoGenDocument
    # ------------------------------------------------------------------ #

    def _from_matches(
        self,
        matches: Sequence[Mapping[str, Any]],
    ) -> List[Tuple[AutoGenDocument, float]]:
        """
        Convert generic match records (from VectorTranslator/coerce_hits) into
        (AutoGenDocument, score) tuples.

        Expected match shape (canonicalized by coerce_hits):
            {
                "id": str,
                "score": float,
                "metadata": { ... },
                "vector": [...],      # if include_vectors=True
                ...
            }
        """
        results: List[Tuple[AutoGenDocument, float]] = []

        for m in matches:
            if not isinstance(m, Mapping):
                logger.debug("Skipping non-mapping match: %r", type(m).__name__)
                continue

            meta_full = dict(m.get("metadata") or {})
            score = float(m.get("score", 0.0))

            # Optional score threshold
            if self.score_threshold is not None and score < float(self.score_threshold):
                continue

            # Extract user-facing metadata & text
            if self.metadata_field and self.metadata_field in meta_full:
                nested = meta_full.get(self.metadata_field) or {}
                if isinstance(nested, Mapping):
                    nested_meta = dict(nested)
                else:
                    nested_meta = {}
            else:
                nested_meta = dict(meta_full)

            text_val = meta_full.get(self.text_field) or ""
            nested_meta.pop(self.text_field, None)
            nested_meta.pop(self.id_field, None)

            doc = AutoGenDocument(page_content=str(text_val), metadata=nested_meta)
            results.append((doc, score))

        return results

    # ------------------------------------------------------------------ #
    # Internal query helpers
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
        Build a framework-agnostic raw query dict consumed by
        DefaultVectorFrameworkTranslator.build_query_spec.

        Fields:
            vector: list[float]
            top_k: int
            filters: optional mapping / DSL
            namespace: optional
            include_metadata: bool
            include_vectors: bool
        """
        ns = self._effective_namespace(namespace)
        vec = [float(x) for x in embedding]
        self._update_dim_hint(len(vec) if vec else None)
        if vec:
            self._maybe_check_dim(vec, where="build_raw_query")

        raw: Dict[str, Any] = {
            "vector": vec,
            "top_k": int(k),
            "filters": dict(filter) if filter is not None else None,
            "namespace": ns,
            "include_metadata": True,
            "include_vectors": bool(include_vectors),
        }
        return raw

    # Backwards-compatible helper retained; public calls now use _build_contexts
    # (kept to avoid removing anything, and to preserve internal utility).
    def _framework_ctx_for_namespace(self, namespace: Optional[str]) -> Mapping[str, Any]:
        """
        Minimal framework_ctx that hints preferred namespace to the translator.
        """
        ns = self._effective_namespace(namespace)
        ctx: Dict[str, Any] = {}
        if ns is not None:
            ctx["namespace"] = ns
        if self._framework_version is not None:
            ctx["framework_version"] = self._framework_version
        return ctx

    def _extract_matches_from_result(self, result: Any) -> List[Mapping[str, Any]]:
        """
        Coerce translator query result into a canonical list of hit mappings.
        """
        # result is typically {"matches": [...], "namespace": ..., ...}
        return _coerce_hits_safe(result)

    def _extract_matches_from_chunk(self, chunk: Any) -> List[Mapping[str, Any]]:
        """
        Coerce translator streaming chunk into a canonical list of hit mappings.
        """
        # chunk is typically {"matches": [...], "is_final": bool, ...}
        return _coerce_hits_safe(chunk)

    # ------------------------------------------------------------------ #
    # Callable interface: vector function (__call__ / acall)
    # ------------------------------------------------------------------ #

    @with_error_context("call_sync")
    def __call__(self, texts: Any) -> Any:
        """
        Callable embedding function for integrations that expect a vector_function.

        Input:
        - str -> returns List[float]
        - Sequence[str] -> returns List[List[float]]

        Empty-input handling:
        - Empty batch -> returns [] (existing style)
        - Empty/whitespace items -> deterministic zero vectors *if* dimension is known.
          If dimension is unknown and all items are empty, raises with attached context.
        """
        _ensure_not_in_event_loop("__call__")

        single = False
        if isinstance(texts, str):
            single = True
            batch = [texts]
        else:
            batch = list(texts or [])

        if not batch:
            return [] if not single else []

        empties: List[int] = []
        nonempty: List[str] = []
        for i, t in enumerate(batch):
            s = t if isinstance(t, str) else str(t)
            if not s.strip():
                empties.append(i)
            else:
                nonempty.append(s)

        computed_vectors: List[List[float]] = []
        if nonempty:
            computed = self._ensure_embeddings(nonempty, embeddings=None)
            computed_vectors = [[float(x) for x in v] for v in computed]
            if computed_vectors and computed_vectors[0]:
                self._update_dim_hint(len(computed_vectors[0]))

        # If we have empty items and still do not know dimension, we cannot
        # deterministically produce zero vectors.
        if empties and self._vector_dim_hint is None:
            err = BadRequest(
                "cannot vectorize empty/whitespace items because vector dimension is unknown",
                code=ErrorCodes.UNKNOWN_VECTOR_DIMENSION,
                details={"empty_items": len(empties), "total_items": len(batch)},
            )
            attach_context(
                err,
                framework="autogen",
                operation="vector_call_sync",
                error_codes=VECTOR_COERCION_ERROR_CODES,
                vectors_count=len(batch),
                total_content_chars=sum(len((t if isinstance(t, str) else str(t))) for t in batch),
                vector_dimension_hint=self._vector_dim_hint,
            )
            raise err

        # Reconstruct output aligned to original indices.
        out: List[List[float]] = []
        it = iter(computed_vectors)
        empty_set = set(empties)
        for i, _ in enumerate(batch):
            if i in empty_set:
                out.append(self._zero_vector())
            else:
                v = next(it)
                self._maybe_check_dim(v, where="__call__")
                out.append(v)

        return out[0] if single else out

    @with_async_error_context("call_async")
    async def acall(self, texts: Any) -> Any:
        """
        Async callable embedding function.

        Mirrors __call__ behavior but uses async embedding computation when available.
        """
        single = False
        if isinstance(texts, str):
            single = True
            batch = [texts]
        else:
            batch = list(texts or [])

        if not batch:
            return [] if not single else []

        empties: List[int] = []
        nonempty: List[str] = []
        for i, t in enumerate(batch):
            s = t if isinstance(t, str) else str(t)
            if not s.strip():
                empties.append(i)
            else:
                nonempty.append(s)

        computed_vectors: List[List[float]] = []
        if nonempty:
            computed = await self._ensure_embeddings_async(nonempty, embeddings=None)
            computed_vectors = [[float(x) for x in v] for v in computed]
            if computed_vectors and computed_vectors[0]:
                self._update_dim_hint(len(computed_vectors[0]))

        if empties and self._vector_dim_hint is None:
            err = BadRequest(
                "cannot vectorize empty/whitespace items because vector dimension is unknown",
                code=ErrorCodes.UNKNOWN_VECTOR_DIMENSION,
                details={"empty_items": len(empties), "total_items": len(batch)},
            )
            attach_context(
                err,
                framework="autogen",
                operation="vector_call_async",
                error_codes=VECTOR_COERCION_ERROR_CODES,
                vectors_count=len(batch),
                total_content_chars=sum(len((t if isinstance(t, str) else str(t))) for t in batch),
                vector_dimension_hint=self._vector_dim_hint,
            )
            raise err

        out: List[List[float]] = []
        it = iter(computed_vectors)
        empty_set = set(empties)
        for i, _ in enumerate(batch):
            if i in empty_set:
                out.append(self._zero_vector())
            else:
                v = next(it)
                self._maybe_check_dim(v, where="acall")
                out.append(v)

        return out[0] if single else out

    # ------------------------------------------------------------------ #
    # Public add APIs (sync + async)
    # ------------------------------------------------------------------ #

    @with_error_context("add_texts_sync")
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Metadata]] = None,
        ids: Optional[List[str]] = None,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
        embeddings: Optional[Embeddings] = None,
        namespace: Optional[str] = None,
    ) -> List[str]:
        """
        Add texts to the vector store (sync).

        Behavior:
        - Uses `embedding_function` if configured, unless explicit `embeddings`
          are supplied.
        - Delegates upsert orchestration to VectorTranslator.
        """
        _ensure_not_in_event_loop("add_texts")

        texts_list = [str(t) for t in texts]
        if not texts_list:
            return []

        # Enforce non-empty texts for writes
        for idx, t in enumerate(texts_list):
            if not t.strip():
                raise BadRequest(
                    f"text at index {idx} is empty or whitespace-only",
                    code=ErrorCodes.BAD_TEXTS,
                    details={"index": idx},
                )

        op_ctx, fw_ctx = self._build_contexts(
            operation="add_texts",
            conversation=conversation,
            extra_context=extra_context,
            namespace=namespace,
        )
        metadatas_norm = self._normalize_metadatas(len(texts_list), metadatas)
        ids_norm = self._normalize_ids(len(texts_list), ids)
        emb = self._ensure_embeddings(texts_list, embeddings)

        ns = self._effective_namespace(namespace)

        raw_documents: List[Mapping[str, Any]] = []
        for text, vec, meta, vid in zip(texts_list, emb, metadatas_norm, ids_norm):
            vec_f = [float(x) for x in vec]
            if vec_f:
                self._update_dim_hint(len(vec_f))
                self._maybe_check_dim(vec_f, where="add_texts")
            if self.metadata_field:
                envelope: Dict[str, Any] = {}
                if meta:
                    envelope[self.metadata_field] = dict(meta)
                envelope[self.text_field] = text
                envelope[self.id_field] = vid
                metadata_payload = envelope
            else:
                metadata_payload = dict(meta or {})
                metadata_payload[self.text_field] = text
                metadata_payload[self.id_field] = vid

            raw_documents.append(
                {
                    "id": vid,
                    "vector": vec_f,
                    "metadata": metadata_payload,
                    "namespace": ns,
                }
            )

        # We return logical IDs, but still validate the translated upsert result
        # to detect "everything failed" scenarios.
        result = self._translator.upsert(
            raw_documents=raw_documents,
            op_ctx=op_ctx,
            framework_ctx=fw_ctx,
        )

        self._validate_upsert_result(
            result=result,
            total_texts=len(texts_list),
            namespace=ns,
        )

        return ids_norm

    @with_async_error_context("add_texts_async")
    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Metadata]] = None,
        ids: Optional[List[str]] = None,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
        embeddings: Optional[Embeddings] = None,
        namespace: Optional[str] = None,
    ) -> List[str]:
        """
        Add texts to the vector store (async).
        """
        texts_list = [str(t) for t in texts]
        if not texts_list:
            return []

        # Enforce non-empty texts for writes
        for idx, t in enumerate(texts_list):
            if not t.strip():
                raise BadRequest(
                    f"text at index {idx} is empty or whitespace-only",
                    code=ErrorCodes.BAD_TEXTS,
                    details={"index": idx},
                )

        op_ctx, fw_ctx = self._build_contexts(
            operation="aadd_texts",
            conversation=conversation,
            extra_context=extra_context,
            namespace=namespace,
        )
        metadatas_norm = self._normalize_metadatas(len(texts_list), metadatas)
        ids_norm = self._normalize_ids(len(texts_list), ids)
        emb = await self._ensure_embeddings_async(texts_list, embeddings)

        ns = self._effective_namespace(namespace)

        raw_documents: List[Mapping[str, Any]] = []
        for text, vec, meta, vid in zip(texts_list, emb, metadatas_norm, ids_norm):
            vec_f = [float(x) for x in vec]
            if vec_f:
                self._update_dim_hint(len(vec_f))
                self._maybe_check_dim(vec_f, where="aadd_texts")
            if self.metadata_field:
                envelope: Dict[str, Any] = {}
                if meta:
                    envelope[self.metadata_field] = dict(meta)
                envelope[self.text_field] = text
                envelope[self.id_field] = vid
                metadata_payload = envelope
            else:
                metadata_payload = dict(meta or {})
                metadata_payload[self.text_field] = text
                metadata_payload[self.id_field] = vid

            raw_documents.append(
                {
                    "id": vid,
                    "vector": vec_f,
                    "metadata": metadata_payload,
                    "namespace": ns,
                }
            )

        result = await self._translator.arun_upsert(
            raw_documents=raw_documents,
            op_ctx=op_ctx,
            framework_ctx=fw_ctx,
        )

        self._validate_upsert_result(
            result=result,
            total_texts=len(texts_list),
            namespace=ns,
        )

        return ids_norm

    @with_error_context("add_documents_sync")
    def add_documents(
        self,
        documents: Iterable[AutoGenDocument],
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
        embeddings: Optional[Embeddings] = None,
        namespace: Optional[str] = None,
    ) -> List[str]:
        """
        Add AutoGenDocuments to the vector store (sync).
        """
        _ensure_not_in_event_loop("add_documents")

        docs_list = list(documents)
        if not docs_list:
            return []

        texts = [str(d.page_content) for d in docs_list]
        metadatas = [dict(d.metadata or {}) for d in docs_list]
        return self.add_texts(
            texts,
            metadatas=metadatas,
            ids=None,
            conversation=conversation,
            extra_context=extra_context,
            embeddings=embeddings,
            namespace=namespace,
        )

    @with_async_error_context("add_documents_async")
    async def aadd_documents(
        self,
        documents: Iterable[AutoGenDocument],
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
        embeddings: Optional[Embeddings] = None,
        namespace: Optional[str] = None,
    ) -> List[str]:
        """
        Add AutoGenDocuments to the vector store (async).
        """
        docs_list = list(documents)
        if not docs_list:
            return []

        texts = [str(d.page_content) for d in docs_list]
        metadatas = [dict(d.metadata or {}) for d in docs_list]
        return await self.aadd_texts(
            texts,
            metadatas=metadatas,
            ids=None,
            conversation=conversation,
            extra_context=extra_context,
            embeddings=embeddings,
            namespace=namespace,
        )

    # ------------------------------------------------------------------ #
    # Query APIs (sync + async)
    # ------------------------------------------------------------------ #

    @with_error_context("similarity_search_sync")
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Mapping[str, Any]] = None,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
        embedding: Optional[Sequence[float]] = None,
        namespace: Optional[str] = None,
    ) -> List[AutoGenDocument]:
        """
        Perform similarity search and return AutoGenDocuments (sync).
        """
        _ensure_not_in_event_loop("similarity_search")

        op_ctx, fw_ctx = self._build_contexts(
            operation="similarity_search",
            conversation=conversation,
            extra_context=extra_context,
            namespace=namespace,
        )
        query_emb = self._embed_query(query, embedding=embedding)

        top_k = k or self.default_top_k
        _warn_if_extreme_k(top_k, op_name="similarity_search")

        raw_query = self._build_raw_query(
            embedding=query_emb,
            k=top_k,
            namespace=namespace,
            filter=filter,
            include_vectors=False,
        )

        result = self._translator.query(
            raw_query,
            op_ctx=op_ctx,
            framework_ctx=fw_ctx,
            mmr_config=None,
        )
        matches = self._extract_matches_from_result(result)
        docs_scores = self._from_matches(matches)
        return [doc for doc, _ in docs_scores]

    @with_error_context("similarity_search_with_score_sync")
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Mapping[str, Any]] = None,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
        embedding: Optional[Sequence[float]] = None,
        namespace: Optional[str] = None,
    ) -> List[Tuple[AutoGenDocument, float]]:
        """
        Similarity search returning (AutoGenDocument, score) tuples (sync).
        """
        _ensure_not_in_event_loop("similarity_search_with_score")

        op_ctx, fw_ctx = self._build_contexts(
            operation="similarity_search_with_score",
            conversation=conversation,
            extra_context=extra_context,
            namespace=namespace,
        )
        query_emb = self._embed_query(query, embedding=embedding)

        top_k = k or self.default_top_k
        _warn_if_extreme_k(top_k, op_name="similarity_search_with_score")

        raw_query = self._build_raw_query(
            embedding=query_emb,
            k=top_k,
            namespace=namespace,
            filter=filter,
            include_vectors=False,
        )

        result = self._translator.query(
            raw_query,
            op_ctx=op_ctx,
            framework_ctx=fw_ctx,
            mmr_config=None,
        )
        matches = self._extract_matches_from_result(result)
        return self._from_matches(matches)

    @with_async_error_context("similarity_search_async")
    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Mapping[str, Any]] = None,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
        embedding: Optional[Sequence[float]] = None,
        namespace: Optional[str] = None,
    ) -> List[AutoGenDocument]:
        """
        Perform similarity search and return AutoGenDocuments (async).
        """
        op_ctx, fw_ctx = self._build_contexts(
            operation="asimilarity_search",
            conversation=conversation,
            extra_context=extra_context,
            namespace=namespace,
        )
        query_emb = await self._embed_query_async(query, embedding=embedding)

        top_k = k or self.default_top_k
        _warn_if_extreme_k(top_k, op_name="asimilarity_search")

        raw_query = self._build_raw_query(
            embedding=query_emb,
            k=top_k,
            namespace=namespace,
            filter=filter,
            include_vectors=False,
        )

        result = await self._translator.arun_query(
            raw_query,
            op_ctx=op_ctx,
            framework_ctx=fw_ctx,
            mmr_config=None,
        )
        matches = self._extract_matches_from_result(result)
        docs_scores = self._from_matches(matches)
        return [doc for doc, _ in docs_scores]

    @with_async_error_context("similarity_search_with_score_async")
    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Mapping[str, Any]] = None,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
        embedding: Optional[Sequence[float]] = None,
        namespace: Optional[str] = None,
    ) -> List[Tuple[AutoGenDocument, float]]:
        """
        Similarity search returning (AutoGenDocument, score) tuples (async).
        """
        op_ctx, fw_ctx = self._build_contexts(
            operation="asimilarity_search_with_score",
            conversation=conversation,
            extra_context=extra_context,
            namespace=namespace,
        )
        query_emb = await self._embed_query_async(query, embedding=embedding)

        top_k = k or self.default_top_k
        _warn_if_extreme_k(top_k, op_name="asimilarity_search_with_score")

        raw_query = self._build_raw_query(
            embedding=query_emb,
            k=top_k,
            namespace=namespace,
            filter=filter,
            include_vectors=False,
        )

        result = await self._translator.arun_query(
            raw_query,
            op_ctx=op_ctx,
            framework_ctx=fw_ctx,
            mmr_config=None,
        )
        matches = self._extract_matches_from_result(result)
        return self._from_matches(matches)

    # ------------------------------------------------------------------ #
    # Streaming similarity search (sync + async)
    # ------------------------------------------------------------------ #

    @with_error_context("similarity_search_stream_sync")
    def similarity_search_stream(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Mapping[str, Any]] = None,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
        embedding: Optional[Sequence[float]] = None,
        namespace: Optional[str] = None,
    ) -> Iterator[AutoGenDocument]:
        """
        Streaming similarity search (sync), yielding AutoGenDocuments one by one.

        This delegates streaming orchestration to `VectorTranslator.query_stream`
        which uses SyncStreamBridge internally; this adapter does not directly
        instantiate any async↔sync bridges.
        """
        _ensure_not_in_event_loop("similarity_search_stream")

        op_ctx, fw_ctx = self._build_contexts(
            operation="similarity_search_stream",
            conversation=conversation,
            extra_context=extra_context,
            namespace=namespace,
        )
        query_emb = self._embed_query(query, embedding=embedding)

        top_k = k or self.default_top_k
        _warn_if_extreme_k(top_k, op_name="similarity_search_stream")

        raw_query = self._build_raw_query(
            embedding=query_emb,
            k=top_k,
            namespace=namespace,
            filter=filter,
            include_vectors=False,
        )

        for chunk in self._translator.query_stream(
            raw_query,
            op_ctx=op_ctx,
            framework_ctx=fw_ctx,
        ):
            matches = self._extract_matches_from_chunk(chunk)
            for doc, _ in self._from_matches(matches):
                yield doc

    @with_async_error_context("similarity_search_stream_async")
    async def asimilarity_search_stream(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Mapping[str, Any]] = None,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
        embedding: Optional[Sequence[float]] = None,
        namespace: Optional[str] = None,
    ) -> AsyncIterator[AutoGenDocument]:
        """
        Streaming similarity search (async), yielding AutoGenDocuments one by one.

        Delegates streaming orchestration to `VectorTranslator.arun_query_stream`.
        """
        op_ctx, fw_ctx = self._build_contexts(
            operation="asimilarity_search_stream",
            conversation=conversation,
            extra_context=extra_context,
            namespace=namespace,
        )
        query_emb = await self._embed_query_async(query, embedding=embedding)

        top_k = k or self.default_top_k
        _warn_if_extreme_k(top_k, op_name="asimilarity_search_stream")

        raw_query = self._build_raw_query(
            embedding=query_emb,
            k=top_k,
            namespace=namespace,
            filter=filter,
            include_vectors=False,
        )

        async for chunk in self._translator.arun_query_stream(
            raw_query,
            op_ctx=op_ctx,
            framework_ctx=fw_ctx,
        ):
            matches = self._extract_matches_from_chunk(chunk)
            for doc, _ in self._from_matches(matches):
                yield doc

    # ------------------------------------------------------------------ #
    # Low-level raw query API (precomputed embeddings)
    # ------------------------------------------------------------------ #

    @with_error_context("query_sync")
    def query(
        self,
        embedding: Sequence[float],
        k: int = 4,
        *,
        filter: Optional[Mapping[str, Any]] = None,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
        namespace: Optional[str] = None,
        include_vectors: bool = False,
    ) -> List[Mapping[str, Any]]:
        """
        Low-level similarity query using a precomputed embedding (sync).

        Returns raw match records from the translator, instead of
        AutoGenDocument objects.
        """
        _ensure_not_in_event_loop("query")

        op_ctx, fw_ctx = self._build_contexts(
            operation="query",
            conversation=conversation,
            extra_context=extra_context,
            namespace=namespace,
        )
        top_k = k or self.default_top_k
        _warn_if_extreme_k(top_k, op_name="query")

        raw_query = self._build_raw_query(
            embedding=embedding,
            k=top_k,
            namespace=namespace,
            filter=filter,
            include_vectors=include_vectors,
        )

        result = self._translator.query(
            raw_query,
            op_ctx=op_ctx,
            framework_ctx=fw_ctx,
            mmr_config=None,
        )
        return self._extract_matches_from_result(result)

    @with_async_error_context("query_async")
    async def aquery(
        self,
        embedding: Sequence[float],
        k: int = 4,
        *,
        filter: Optional[Mapping[str, Any]] = None,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
        namespace: Optional[str] = None,
        include_vectors: bool = False,
    ) -> List[Mapping[str, Any]]:
        """
        Low-level similarity query using a precomputed embedding (async).

        Returns raw match records from the translator, instead of
        AutoGenDocument objects.
        """
        op_ctx, fw_ctx = self._build_contexts(
            operation="aquery",
            conversation=conversation,
            extra_context=extra_context,
            namespace=namespace,
        )
        top_k = k or self.default_top_k
        _warn_if_extreme_k(top_k, op_name="aquery")

        raw_query = self._build_raw_query(
            embedding=embedding,
            k=top_k,
            namespace=namespace,
            filter=filter,
            include_vectors=include_vectors,
        )

        result = await self._translator.arun_query(
            raw_query,
            op_ctx=op_ctx,
            framework_ctx=fw_ctx,
            mmr_config=None,
        )
        return self._extract_matches_from_result(result)

    # ------------------------------------------------------------------ #
    # MMR search (sync + async) via MMRConfig
    # ------------------------------------------------------------------ #

    @with_error_context("mmr_search_sync")
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        lambda_mult: float = 0.5,
        filter: Optional[Mapping[str, Any]] = None,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
        embedding: Optional[Sequence[float]] = None,
        namespace: Optional[str] = None,
        fetch_k: Optional[int] = None,
    ) -> List[AutoGenDocument]:
        """
        Perform Maximal Marginal Relevance (MMR) search (sync).

        This runs a similarity search with a larger `fetch_k` and delegates
        MMR re-ranking to VectorTranslator via MMRConfig.
        """
        _ensure_not_in_event_loop("max_marginal_relevance_search")

        op_ctx, fw_ctx = self._build_contexts(
            operation="max_marginal_relevance_search",
            conversation=conversation,
            extra_context=extra_context,
            namespace=namespace,
        )
        query_emb = self._embed_query(query, embedding=embedding)

        base_k = k or self.default_top_k
        actual_fetch_k = fetch_k or max(base_k * 4, base_k + 5)

        _warn_if_extreme_k(base_k, op_name="max_marginal_relevance_search_k")
        _warn_if_extreme_k(actual_fetch_k, op_name="max_marginal_relevance_search_fetch_k")

        raw_query = self._build_raw_query(
            embedding=query_emb,
            k=actual_fetch_k,
            namespace=namespace,
            filter=filter,
            include_vectors=True,  # embeddings required for MMR
        )

        mmr_config = MMRConfig(
            enabled=True,
            k=base_k,
            lambda_mult=lambda_mult,
            # We assume the adapter exposes "score" and "embedding" fields for matches.
            score_key="score",
            vector_key="embedding",
            invert_score=False,
        )

        result = self._translator.query(
            raw_query,
            op_ctx=op_ctx,
            framework_ctx=fw_ctx,
            mmr_config=mmr_config,
        )
        matches = self._extract_matches_from_result(result)
        docs_scores = self._from_matches(matches)
        return [doc for doc, _ in docs_scores][:base_k]

    @with_async_error_context("mmr_search_async")
    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        lambda_mult: float = 0.5,
        filter: Optional[Mapping[str, Any]] = None,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
        embedding: Optional[Sequence[float]] = None,
        namespace: Optional[str] = None,
        fetch_k: Optional[int] = None,
    ) -> List[AutoGenDocument]:
        """
        Perform Maximal Marginal Relevance (MMR) search (async).
        """
        op_ctx, fw_ctx = self._build_contexts(
            operation="amax_marginal_relevance_search",
            conversation=conversation,
            extra_context=extra_context,
            namespace=namespace,
        )
        query_emb = await self._embed_query_async(query, embedding=embedding)

        base_k = k or self.default_top_k
        actual_fetch_k = fetch_k or max(base_k * 4, base_k + 5)

        _warn_if_extreme_k(base_k, op_name="amax_marginal_relevance_search_k")
        _warn_if_extreme_k(actual_fetch_k, op_name="amax_marginal_relevance_search_fetch_k")

        raw_query = self._build_raw_query(
            embedding=query_emb,
            k=actual_fetch_k,
            namespace=namespace,
            filter=filter,
            include_vectors=True,
        )

        mmr_config = MMRConfig(
            enabled=True,
            k=base_k,
            lambda_mult=lambda_mult,
            score_key="score",
            vector_key="embedding",
            invert_score=False,
        )

        result = await self._translator.arun_query(
            raw_query,
            op_ctx=op_ctx,
            framework_ctx=fw_ctx,
            mmr_config=mmr_config,
        )
        matches = self._extract_matches_from_result(result)
        docs_scores = self._from_matches(matches)
        return [doc for doc, _ in docs_scores][:base_k]

    # ------------------------------------------------------------------ #
    # Delete API (sync + async)
    # ------------------------------------------------------------------ #

    @with_error_context("delete_sync")
    def delete(
        self,
        ids: Optional[List[str]] = None,
        *,
        filter: Optional[Mapping[str, Any]] = None,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
        namespace: Optional[str] = None,
    ) -> None:
        """
        Delete vectors by IDs or metadata filter (sync).

        If both ids and filter are provided, filter takes precedence.
        """
        _ensure_not_in_event_loop("delete")

        op_ctx, fw_ctx = self._build_contexts(
            operation="delete",
            conversation=conversation,
            extra_context=extra_context,
            namespace=namespace,
        )
        ns = self._effective_namespace(namespace)

        if filter is not None:
            raw_filter_or_ids: Any = dict(filter)
        elif ids is not None:
            raw_filter_or_ids = list(ids)
        else:
            raise BadRequest(
                "must provide ids or filter for delete",
                code=ErrorCodes.BAD_DELETE_REQUEST,
            )

        self._translator.delete(
            raw_filter_or_ids,
            op_ctx=op_ctx,
            framework_ctx=fw_ctx,
        )

    @with_async_error_context("delete_async")
    async def adelete(
        self,
        ids: Optional[List[str]] = None,
        *,
        filter: Optional[Mapping[str, Any]] = None,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
        namespace: Optional[str] = None,
    ) -> None:
        """
        Delete vectors by IDs or metadata filter (async).

        If both ids and filter are provided, filter takes precedence.
        """
        op_ctx, fw_ctx = self._build_contexts(
            operation="adelete",
            conversation=conversation,
            extra_context=extra_context,
            namespace=namespace,
        )
        ns = self._effective_namespace(namespace)

        if filter is not None:
            raw_filter_or_ids: Any = dict(filter)
        elif ids is not None:
            raw_filter_or_ids = list(ids)
        else:
            raise BadRequest(
                "must provide ids or filter for delete",
                code=ErrorCodes.BAD_DELETE_REQUEST,
            )

        await self._translator.arun_delete(
            raw_filter_or_ids,
            op_ctx=op_ctx,
            framework_ctx=fw_ctx,
        )

    # ------------------------------------------------------------------ #
    # Health / capabilities via translator only (no adapter fallback)
    # ------------------------------------------------------------------ #

    @with_error_context("capabilities_sync")
    def capabilities(self, **kwargs: Any) -> Mapping[str, Any]:
        """
        Synchronous capabilities query.

        Translator-only:
        - `self._translator.capabilities(**kwargs)` MUST be implemented.
        - No fallback to corpus_adapter.capabilities to keep a strict
          translator-first contract.
        """
        _ensure_not_in_event_loop("capabilities")

        translator_capabilities = getattr(self._translator, "capabilities", None)
        if not callable(translator_capabilities):
            raise NotSupported(
                "VectorTranslator for framework='autogen' must implement "
                "capabilities(); no adapter fallback is allowed.",
            )
        return translator_capabilities(**kwargs)  # type: ignore[misc]

    @with_async_error_context("capabilities_async")
    async def acapabilities(self, **kwargs: Any) -> Mapping[str, Any]:
        """
        Async capabilities query.

        Translator-only resolution:
        1. self._translator.acapabilities(**kwargs)
        2. self._translator.capabilities(**kwargs) via worker thread

        If neither is implemented, this is treated as a configuration error.
        """
        translator_acapabilities = getattr(self._translator, "acapabilities", None)
        if callable(translator_acapabilities):
            return await translator_acapabilities(**kwargs)  # type: ignore[misc]

        translator_capabilities = getattr(self._translator, "capabilities", None)
        if callable(translator_capabilities):
            return await asyncio.to_thread(translator_capabilities, **kwargs)

        raise NotSupported(
            "VectorTranslator for framework='autogen' must implement "
            "acapabilities() or capabilities(); no adapter fallback is allowed.",
        )

    @with_error_context("health_sync")
    def health(self, **kwargs: Any) -> Mapping[str, Any]:
        """
        Synchronous health check.

        Translator-only:
        - `self._translator.health(**kwargs)` MUST be implemented.
        - No fallback to corpus_adapter.health to avoid legacy/adapter coupling.
        """
        _ensure_not_in_event_loop("health")

        translator_health = getattr(self._translator, "health", None)
        if not callable(translator_health):
            raise NotSupported(
                "VectorTranslator for framework='autogen' must implement "
                "health(); no adapter fallback is allowed.",
            )
        return translator_health(**kwargs)  # type: ignore[misc]

    @with_async_error_context("health_async")
    async def ahealth(self, **kwargs: Any) -> Mapping[str, Any]:
        """
        Async health check.

        Translator-only resolution:
        1. self._translator.ahealth(**kwargs)
        2. self._translator.health(**kwargs) via worker thread

        If neither is implemented, this is treated as a configuration error.
        """
        translator_ahealth = getattr(self._translator, "ahealth", None)
        if callable(translator_ahealth):
            return await translator_ahealth(**kwargs)  # type: ignore[misc]

        translator_health = getattr(self._translator, "health", None)
        if callable(translator_health):
            return await asyncio.to_thread(translator_health, **kwargs)

        raise NotSupported(
            "VectorTranslator for framework='autogen' must implement "
            "ahealth() or health(); no adapter fallback is allowed.",
        )

    # ------------------------------------------------------------------ #
    # Resource cleanup (close / aclose / context managers)
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """
        Best-effort synchronous cleanup of translator/adapter resources.

        This method never raises; it only logs warnings on failure.

        Ownership semantics:
        - Always attempts to close the VectorTranslator if it supports close().
        - If own_adapter=True, also attempts to close the underlying corpus_adapter.
        """
        try:
            translator_close = getattr(self._translator, "close", None)
        except Exception:
            translator_close = None

        if callable(translator_close):
            try:
                translator_close()
            except Exception:
                logger.warning(
                    "Error while closing VectorTranslator synchronously",
                    exc_info=True,
                )

        if self._own_adapter:
            try:
                adapter_close = getattr(self.corpus_adapter, "close", None)
            except Exception:
                adapter_close = None
            if callable(adapter_close):
                try:
                    adapter_close()
                except Exception:
                    logger.warning(
                        "Error while closing corpus_adapter synchronously",
                        exc_info=True,
                    )

    async def aclose(self) -> None:
        """
        Best-effort asynchronous cleanup of translator/adapter resources.

        This method never raises; it only logs warnings on failure.

        Ownership semantics:
        - Prefers translator.aclose() if available; falls back to translator.close().
        - If own_adapter=True, also closes/acloses the underlying corpus_adapter
          using the same preference order.
        """
        # Close translator
        try:
            translator_aclose = getattr(self._translator, "aclose", None)
        except Exception:
            translator_aclose = None

        if callable(translator_aclose):
            try:
                await translator_aclose()
            except Exception:
                logger.warning(
                    "Error while closing VectorTranslator asynchronously",
                    exc_info=True,
                )
        else:
            try:
                translator_close = getattr(self._translator, "close", None)
            except Exception:
                translator_close = None
            if callable(translator_close):
                try:
                    if asyncio.iscoroutinefunction(translator_close):
                        await translator_close()
                    else:
                        await asyncio.to_thread(translator_close)
                except Exception:
                    logger.warning(
                        "Error while closing VectorTranslator from async context",
                        exc_info=True,
                    )

        # Close underlying adapter if owned
        if self._own_adapter:
            try:
                adapter_aclose = getattr(self.corpus_adapter, "aclose", None)
            except Exception:
                adapter_aclose = None

            if callable(adapter_aclose):
                try:
                    await adapter_aclose()
                    return
                except Exception:
                    logger.warning(
                        "Error while closing corpus_adapter asynchronously",
                        exc_info=True,
                    )
            else:
                try:
                    adapter_close = getattr(self.corpus_adapter, "close", None)
                except Exception:
                    adapter_close = None
                if callable(adapter_close):
                    try:
                        if asyncio.iscoroutinefunction(adapter_close):
                            await adapter_close()
                        else:
                            await asyncio.to_thread(adapter_close)
                    except Exception:
                        logger.warning(
                            "Error while closing corpus_adapter from async context",
                            exc_info=True,
                        )

    def __enter__(self) -> "CorpusAutoGenVectorStore":
        """
        Synchronous context manager entry; returns self.
        """
        _ensure_not_in_event_loop("__enter__")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """
        Synchronous context manager exit; best-effort close().
        """
        try:
            self.close()
        except Exception:
            logger.warning(
                "Error while closing CorpusAutoGenVectorStore in __exit__",
                exc_info=True,
            )

    async def __aenter__(self) -> "CorpusAutoGenVectorStore":
        """
        Async context manager entry; returns self.
        """
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """
        Async context manager exit; best-effort aclose().
        """
        try:
            await self.aclose()
        except Exception:
            logger.warning(
                "Error while closing CorpusAutoGenVectorStore in __aexit__",
                exc_info=True,
            )

    # ------------------------------------------------------------------ #
    # Convenience constructors
    # ------------------------------------------------------------------ #

    @classmethod
    @with_error_context("from_texts_sync")
    def from_texts(
        cls,
        texts: List[str],
        *,
        corpus_adapter: VectorProtocolV1,
        metadatas: Optional[List[Metadata]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "CorpusAutoGenVectorStore":
        """
        Create a store from texts, then add them immediately (sync).
        """
        _ensure_not_in_event_loop("from_texts")

        store = cls(corpus_adapter=corpus_adapter, **kwargs)
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store

    @classmethod
    @with_error_context("from_documents_sync")
    def from_documents(
        cls,
        documents: List[AutoGenDocument],
        *,
        corpus_adapter: VectorProtocolV1,
        **kwargs: Any,
    ) -> "CorpusAutoGenVectorStore":
        """
        Create a store from AutoGenDocuments, then add them immediately (sync).
        """
        _ensure_not_in_event_loop("from_documents")

        store = cls(corpus_adapter=corpus_adapter, **kwargs)
        store.add_documents(documents)
        return store


# --------------------------------------------------------------------------- #
# AutoGen retriever tool wrapper
# --------------------------------------------------------------------------- #


class CorpusAutoGenRetrieverTool:
    """
    Thin wrapper to expose a `CorpusAutoGenVectorStore` as an AutoGen tool.

    This is just a callable object that AutoGen agents can invoke to retrieve
    relevant documents. It is intentionally minimal and framework-agnostic.

    Typical registration in AutoGen:

        retriever_tool = CorpusAutoGenRetrieverTool(
            vector_store=store,
            name="corpus_vector_search",
            description="Retrieve relevant documents from Corpus vector index.",
            search_kwargs={"k": 4},
        )

        # Then register `retriever_tool` as a tool/function in your AutoGen setup.
    """

    def __init__(
        self,
        *,
        vector_store: CorpusAutoGenVectorStore,
        name: str = "corpus_vector_search",
        description: str = "Retrieve relevant documents from a Corpus-backed vector index.",
        search_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.vector_store = vector_store
        self.name = name
        self.description = description
        self.search_kwargs: Dict[str, Any] = dict(search_kwargs or {})

    def __call__(
        self,
        query: str,
        *,
        k: Optional[int] = None,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
        filter: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Invoke the retriever as an AutoGen tool (sync).

        Returns a list of plain dicts with `page_content` and `metadata`
        fields, which are easy for AutoGen agents to consume.
        """
        _ensure_not_in_event_loop("CorpusAutoGenRetrieverTool.__call__")

        effective_kwargs: Dict[str, Any] = dict(self.search_kwargs)
        effective_kwargs.update(kwargs)

        if k is not None:
            effective_kwargs["k"] = k

        docs = self.vector_store.similarity_search(
            query,
            filter=filter,
            conversation=conversation,
            extra_context=extra_context,
            **effective_kwargs,
        )
        return [
            {
                "page_content": d.page_content,
                "metadata": dict(d.metadata or {}),
            }
            for d in docs
        ]


__all__ = [
    "AutoGenDocument",
    "CorpusAutoGenVectorStore",
    "CorpusAutoGenRetrieverTool",
    "ErrorCodes",
    "with_error_context",
    "with_async_error_context",
]
