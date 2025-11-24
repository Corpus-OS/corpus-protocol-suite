# corpus_sdk/vector/framework_adapters/autogen.py
# SPDX-License-Identifier: Apache-2.0

"""
AutoGen adapter for Corpus Vector protocol (v2, translator-based).

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

from adapter_sdk.vector_base import (
    VectorProtocolV1,
    OperationContext,
    BadRequest,
    NotSupported,
    VectorAdapterError,
    UpsertResult,
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


# Bundle of error codes for vector coercion utilities
VECTOR_COERCION_ERROR_CODES: VectorCoercionErrorCodes = VectorCoercionErrorCodes(
    invalid_vector_result="INVALID_VECTOR_RESULT",
    invalid_hit_result="INVALID_VECTOR_HIT_RESULT",
    empty_result="EMPTY_VECTOR_RESULT",
    conversion_error="VECTOR_CONVERSION_ERROR",
    score_out_of_range="VECTOR_SCORE_OUT_OF_RANGE",
    vector_dimension_exceeded="VECTOR_DIMENSION_EXCEEDED",
    vector_norm_invalid="VECTOR_NORM_INVALID",
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
# Error decorators
# --------------------------------------------------------------------------- #


def with_error_context(operation: str, **context_kwargs: Any):
    """
    Decorator to automatically attach error context to sync exceptions.
    """

    def decorator(fn: T) -> T:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                extra = dict(context_kwargs)
                attach_context(
                    exc,
                    framework="autogen",
                    operation=f"vector_{operation}",
                    **extra,
                )
                raise

        return wrapper  # type: ignore[return-value]

    return decorator


def with_async_error_context(operation: str, **context_kwargs: Any):
    """
    Decorator to automatically attach error context to async exceptions.
    """

    def decorator(fn: T) -> T:
        @wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any):
            try:
                return await fn(*args, **kwargs)
            except Exception as exc:
                extra = dict(context_kwargs)
                attach_context(
                    exc,
                    framework="autogen",
                    operation=f"vector_{operation}",
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
    # Context & namespace helpers
    # ------------------------------------------------------------------ #

    def _build_ctx(
        self,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Optional[OperationContext]:
        """
        Build an OperationContext from an AutoGen-style conversation plus
        optional extra context.

        If both are None/empty, returns None (translator will construct an
        "empty" OperationContext as needed).
        """
        extra: Dict[str, Any] = dict(extra_context or {})

        if conversation is None and not extra:
            return None

        try:
            ctx = core_ctx_from_autogen(conversation, **extra)
        except Exception as exc:
            attach_context(
                exc,
                framework="autogen",
                operation="vector_context_translation",
            )
            raise

        if not isinstance(ctx, OperationContext):
            raise BadRequest(
                f"from_autogen produced unsupported context type: {type(ctx).__name__}",
                code=ErrorCodes.BAD_OPERATION_CONTEXT,
            )

        return ctx

    def _effective_namespace(self, namespace: Optional[str]) -> Optional[str]:
        """
        Resolve namespace using explicit override or store default.
        """
        return namespace if namespace is not None else self.namespace

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

    def _ensure_embeddings(
        self,
        texts: List[str],
        embeddings: Optional[Embeddings],
    ) -> Embeddings:
        """
        Ensure embeddings are available for a batch of texts (sync).

        Behavior:
        - If embeddings are provided, verify length.
        - Else, if `embedding_function` is set, compute embeddings.
        - Else, raise NotSupported.
        """
        if embeddings is not None:
            if len(embeddings) != len(texts):
                raise BadRequest(
                    f"embeddings length {len(embeddings)} does not match texts length {len(texts)}",
                    code=ErrorCodes.BAD_EMBEDDINGS,
                    details={"texts": len(texts), "embeddings": len(embeddings)},
                )
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
            )
            raise err

        if len(computed) != len(texts):
            raise BadRequest(
                f"embedding_function returned {len(computed)} embeddings for {len(texts)} texts",
                code=ErrorCodes.BAD_EMBEDDINGS,
            )
        return computed

    async def _ensure_embeddings_async(
        self,
        texts: List[str],
        embeddings: Optional[Embeddings],
    ) -> Embeddings:
        """
        Async-safe version of _ensure_embeddings.

        Behavior:
        - If embeddings are provided, verify length.
        - Else, if async_embedding_function is set, await it.
        - Else, if embedding_function is set, run it in a worker thread.
        - Else, raise NotSupported.
        """
        if embeddings is not None:
            if len(embeddings) != len(texts):
                raise BadRequest(
                    f"embeddings length {len(embeddings)} does not match texts length {len(texts)}",
                    code=ErrorCodes.BAD_EMBEDDINGS,
                    details={"texts": len(texts), "embeddings": len(embeddings)},
                )
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
                )
                raise err

        if len(computed) != len(texts):
            raise BadRequest(
                f"embedding function returned {len(computed)} embeddings for {len(texts)} texts",
                code=ErrorCodes.BAD_EMBEDDINGS,
            )
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
        - If `embedding` is provided, use it.
        - Else, if `embedding_function` is set, compute embedding for [query].
        - Else, raise NotSupported.
        """
        if embedding is not None:
            return [float(x) for x in embedding]

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
            )
            raise err

        return [float(x) for x in embs[0]]

    async def _embed_query_async(
        self,
        query: str,
        *,
        embedding: Optional[Sequence[float]] = None,
    ) -> List[float]:
        """
        Async-safe query embedding helper.

        Behavior:
        - If `embedding` is provided, use it.
        - Else, if `async_embedding_function` is set, await it.
        - Else, if `embedding_function` is set, run it in a worker thread.
        - Else, raise NotSupported.
        """
        if embedding is not None:
            return [float(x) for x in embedding]

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
            )
            raise err

        return [float(x) for x in embs[0]]

    # ------------------------------------------------------------------ #
    # Partial upsert failure handling
    # ------------------------------------------------------------------ #

    def _handle_partial_upsert_failure(
        self,
        result: UpsertResult,
        total_texts: int,
        namespace: Optional[str],
    ) -> None:
        """
        Best-effort partial failure handling for upserts.

        - Logs a warning if some (but not all) records failed.
        - Logs a debug sample of individual failures if available.
        - Only raises if *all* records failed.
        """
        try:
            upserted = int(getattr(result, "upserted_count", 0))
            failed = int(getattr(result, "failed_count", 0))
            failures = getattr(result, "failures", None) or []
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "UpsertResult introspection failed in AutoGen adapter: %r", exc
            )
            return

        if total_texts <= 0:
            return

        if failed > 0:
            logger.warning(
                "CorpusAutoGenVectorStore upsert partial failure: %s/%s succeeded, %s failed (namespace=%r)",
                upserted,
                total_texts,
                failed,
                namespace,
            )
            for failure in list(failures)[:5]:
                logger.debug("Upsert failure detail (sample): %r", failure)

        if upserted == 0 and failed >= total_texts:
            # Treat as hard failure: nothing made it into the index.
            raise VectorAdapterError(
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
    # Internal query helper (sync + async)
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

        raw: Dict[str, Any] = {
            "vector": [float(x) for x in embedding],
            "top_k": int(k),
            "filters": dict(filter) if filter is not None else None,
            "namespace": ns,
            "include_metadata": True,
            "include_vectors": bool(include_vectors),
        }
        return raw

    def _framework_ctx_for_namespace(self, namespace: Optional[str]) -> Mapping[str, Any]:
        """
        Minimal framework_ctx that hints preferred namespace to the translator.
        """
        ns = self._effective_namespace(namespace)
        return {"namespace": ns} if ns is not None else {}

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
        texts_list = list(texts)
        if not texts_list:
            return []

        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        metadatas_norm = self._normalize_metadatas(len(texts_list), metadatas)
        ids_norm = self._normalize_ids(len(texts_list), ids)
        emb = self._ensure_embeddings(texts_list, embeddings)

        ns = self._effective_namespace(namespace)

        raw_documents: List[Mapping[str, Any]] = []
        for text, vec, meta, vid in zip(texts_list, emb, metadatas_norm, ids_norm):
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
                    "vector": [float(x) for x in vec],
                    "metadata": metadata_payload,
                    "namespace": ns,
                }
            )

        # We intentionally ignore the returned ID list; we return our logical IDs.
        result = self._translator.upsert(
            raw_documents=raw_documents,
            op_ctx=ctx,
            framework_ctx=self._framework_ctx_for_namespace(ns),
        )

        if isinstance(result, UpsertResult):
            self._handle_partial_upsert_failure(result, len(texts_list), ns)

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
        texts_list = list(texts)
        if not texts_list:
            return []

        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        metadatas_norm = self._normalize_metadatas(len(texts_list), metadatas)
        ids_norm = self._normalize_ids(len(texts_list), ids)
        emb = await self._ensure_embeddings_async(texts_list, embeddings)

        ns = self._effective_namespace(namespace)

        raw_documents: List[Mapping[str, Any]] = []
        for text, vec, meta, vid in zip(texts_list, emb, metadatas_norm, ids_norm):
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
                    "vector": [float(x) for x in vec],
                    "metadata": metadata_payload,
                    "namespace": ns,
                }
            )

        result = await self._translator.arun_upsert(
            raw_documents=raw_documents,
            op_ctx=ctx,
            framework_ctx=self._framework_ctx_for_namespace(ns),
        )

        if isinstance(result, UpsertResult):
            self._handle_partial_upsert_failure(result, len(texts_list), ns)

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
        docs_list = list(documents)
        if not docs_list:
            return []

        texts = [d.page_content for d in docs_list]
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

        texts = [d.page_content for d in docs_list]
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
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
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
            op_ctx=ctx,
            framework_ctx=self._framework_ctx_for_namespace(namespace),
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
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
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
            op_ctx=ctx,
            framework_ctx=self._framework_ctx_for_namespace(namespace),
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
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
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
            op_ctx=ctx,
            framework_ctx=self._framework_ctx_for_namespace(namespace),
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
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
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
            op_ctx=ctx,
            framework_ctx=self._framework_ctx_for_namespace(namespace),
            mmr_config=None,
        )
        matches = self._extract_matches_from_result(result)
        return self._from_matches(matches)

    # ------------------------------------------------------------------ #
    # Streaming similarity search (sync)
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
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
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
            op_ctx=ctx,
            framework_ctx=self._framework_ctx_for_namespace(namespace),
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
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
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
            op_ctx=ctx,
            framework_ctx=self._framework_ctx_for_namespace(namespace),
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
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
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
            op_ctx=ctx,
            framework_ctx=self._framework_ctx_for_namespace(namespace),
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
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
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
            op_ctx=ctx,
            framework_ctx=self._framework_ctx_for_namespace(namespace),
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
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
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
            op_ctx=ctx,
            framework_ctx=self._framework_ctx_for_namespace(namespace),
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
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
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
            op_ctx=ctx,
            framework_ctx=self._framework_ctx_for_namespace(ns),
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
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
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
            op_ctx=ctx,
            framework_ctx=self._framework_ctx_for_namespace(ns),
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
