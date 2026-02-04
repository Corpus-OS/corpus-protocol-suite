# corpus_sdk/vector/framework_adapters/langchain.py
# SPDX-License-Identifier: Apache-2.0

"""
LangChain adapter for Corpus Vector protocol.

This module exposes Corpus `BaseVectorAdapter` implementations as
`langchain_core` vector stores and retrievers, with:

- Sync + async add/search APIs (mirroring LangChain's VectorStore)
- Proper integration with Corpus VectorProtocolV1
- Namespace + metadata filter handling (capability-aware)
- Batch upserts and deletes that respect backend limits (via VectorTranslator)
- Optional client-side score thresholding
- Optional embedding function integration (sync + async)
- Optional max marginal relevance (MMR) search

Design philosophy
-----------------
- Protocol-first: LangChain is a thin skin over Corpus vector adapters.
- All heavy lifting (backpressure, deadlines, breakers, batching, etc.) lives in
  the underlying adapter + the shared VectorTranslator, not here.
- This layer focuses on:
    * Translating LangChain Documents ↔ Corpus Vector objects
    * Respecting VectorCapabilities (namespaces, filters, batch sizes)
    * Delegating sync/async orchestration to VectorTranslator
    * Propagating LangChain config into OperationContext

Usage
-----

    from corpus_sdk.vector.pinecone_adapter import PineconeVectorAdapter
    from corpus_sdk.vector.framework_adapters.langchain import (
        CorpusLangChainVectorStore,
        CorpusLangChainRetriever,
    )

    adapter = PineconeVectorAdapter(
        index_name="my-index",
        api_key="...",
        dimensions=1536,
    )

    # Provide an embedding function that maps List[str] -> List[List[float]]
    def embed_texts(texts: list[str]) -> list[list[float]]:
        ...

    store = CorpusLangChainVectorStore(
        corpus_adapter=adapter,
        embedding_function=embed_texts,
        namespace="docs",
        default_top_k=4,
    )

    # Add texts
    store.add_texts(
        ["hello world", "another doc"],
        metadatas=[{"topic": "greeting"}, {"topic": "other"}],
    )

    # Similarity search
    docs = store.similarity_search("hello", k=3)

    # Streaming similarity search
    for doc in store.similarity_search_stream("hello", k=3):
        ...

    # As a retriever
    retriever = CorpusLangChainRetriever(vector_store=store, search_kwargs={"k": 3})
    relevant_docs = retriever.get_relevant_documents("hello")
"""

from __future__ import annotations

import asyncio
import logging
import math
import uuid
from functools import cached_property, wraps
from threading import RLock
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

# --------------------------------------------------------------------------- #
# Optional LangChain imports (soft dependency)
# --------------------------------------------------------------------------- #

try:  # pragma: no cover - optional dependency
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.vectorstores import VectorStore

    LANGCHAIN_AVAILABLE = True
except ImportError:  # pragma: no cover - environments without LangChain
    LANGCHAIN_AVAILABLE = False

    class VectorStore:  # type: ignore[no-redef]
        """Minimal stub to keep imports working when LangChain is absent."""

        pass

    class BaseRetriever:  # type: ignore[no-redef]
        """Minimal stub to keep imports working when LangChain is absent."""

        pass

    class Document:  # type: ignore[no-redef]
        """Minimal stub Document; real usage requires langchain-core."""

        def __init__(self, page_content: str, metadata: Optional[dict] = None) -> None:
            self.page_content = page_content
            self.metadata = metadata or {}

    class CallbackManagerForRetrieverRun:  # type: ignore[no-redef]
        """Minimal stub; callbacks are no-ops without LangChain."""

        pass


from corpus_sdk.vector.vector_base import (
    VectorProtocolV1,
    Vector,
    VectorMatch,
    QueryResult,
    UpsertResult,
    DeleteResult,  # kept for API parity (even if unused directly here)
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
    from_langchain as ctx_from_langchain,
    from_dict as ctx_from_dict,
)
from corpus_sdk.core.error_context import attach_context

logger = logging.getLogger(__name__)


def _coerce_to_vector_operation_context(ctx: Any) -> OperationContext:
    """Coerce a context-like object into the vector protocol's OperationContext.

    `corpus_sdk.core.context_translation` returns the core OperationContext.
    Vector adapters use the vector protocol OperationContext.
    """
    if isinstance(ctx, OperationContext):
        return ctx

    return OperationContext(
        request_id=getattr(ctx, "request_id", None),
        idempotency_key=getattr(ctx, "idempotency_key", None),
        deadline_ms=getattr(ctx, "deadline_ms", None),
        traceparent=getattr(ctx, "traceparent", None),
        tenant=getattr(ctx, "tenant", None),
        attrs=getattr(ctx, "attrs", None) or {},
    )

Embeddings = Sequence[Sequence[float]]
Metadata = Dict[str, Any]


class LangChainVectorFrameworkTranslator(DefaultVectorFrameworkTranslator):
    """LangChain VectorStore expects protocol-native result objects.

    The generic DefaultVectorFrameworkTranslator translates QueryResult into
    framework-neutral dict shapes. For LangChain's VectorStore integration,
    we keep QueryResult intact and let the adapter map it into Documents.
    """

    def translate_query_result(
        self,
        result: QueryResult,
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Any:
        return result


# --------------------------------------------------------------------------- #
# Protocol client wrapper
# --------------------------------------------------------------------------- #


class CorpusLangChainVectorClient:
    """VectorProtocolV1-shaped wrapper used by the conformance test suite."""

    def __init__(self, *, adapter: VectorProtocolV1) -> None:
        self._translator = VectorTranslator(
            adapter=adapter,
            framework="langchain",
            translator=DefaultVectorFrameworkTranslator(),
        )

    def capabilities(self, *, config: Optional[Any] = None) -> Any:
        return self._translator.capabilities(framework_ctx=config)

    def health(self, *, config: Optional[Any] = None) -> Any:
        return self._translator.health(framework_ctx=config)

    def query(self, raw_query: Any, *, config: Optional[Any] = None) -> Any:
        return self._translator.query(raw_query, framework_ctx=config)

    def batch_query(self, raw_queries: Any, *, config: Optional[Any] = None) -> Any:
        return self._translator.batch_query(raw_queries, framework_ctx=config)

    def upsert(self, raw_documents: Any, *, config: Optional[Any] = None) -> Any:
        return self._translator.upsert(raw_documents, framework_ctx=config)

    def delete(self, raw_filter_or_ids: Any, *, config: Optional[Any] = None) -> Any:
        return self._translator.delete(raw_filter_or_ids, framework_ctx=config)

    def create_namespace(self, name: str, *, config: Optional[Any] = None) -> Any:
        return self._translator.create_namespace(name, framework_ctx=config)

    def delete_namespace(self, name: str, *, config: Optional[Any] = None) -> Any:
        return self._translator.delete_namespace(name, framework_ctx=config)


# --------------------------------------------------------------------------- #
# Shared vector framework constants
# --------------------------------------------------------------------------- #

VECTOR_ERROR_CODES = VectorCoercionErrorCodes(
    # Vector-flavored codes (framework scoped)
    invalid_vector_result="LANGCHAIN_VECTOR_INVALID_VECTOR_RESULT",
    invalid_hit_result="LANGCHAIN_VECTOR_INVALID_HIT_RESULT",
    empty_result="LANGCHAIN_VECTOR_EMPTY_RESULT",
    conversion_error="LANGCHAIN_VECTOR_CONVERSION_ERROR",
    score_out_of_range="LANGCHAIN_VECTOR_SCORE_OUT_OF_RANGE",
    vector_dimension_exceeded="LANGCHAIN_VECTOR_DIMENSION_EXCEEDED",
    vector_norm_invalid="LANGCHAIN_VECTOR_NORM_INVALID",
    framework_label="langchain",
)
VECTOR_LIMITS = VectorResourceLimits()
VECTOR_FLAGS = VectorValidationFlags()
TOPK_WARNING_CONFIG = TopKWarningConfig(
    framework_label="langchain",
)


# --------------------------------------------------------------------------- #
# Local adapter error codes (framework-facing)
# --------------------------------------------------------------------------- #


class ErrorCodes:
    BAD_OPERATION_CONTEXT = "BAD_OPERATION_CONTEXT"
    BAD_TRANSLATED_RESULT = "BAD_TRANSLATED_RESULT"
    BAD_TRANSLATED_CHUNK = "BAD_TRANSLATED_CHUNK"
    BAD_EMBEDDINGS = "BAD_EMBEDDINGS"
    NO_EMBEDDING_FUNCTION = "NO_EMBEDDING_FUNCTION"
    EMBEDDING_ERROR = "EMBEDDING_ERROR"
    EMPTY_INPUT_DIM_UNKNOWN = "EMPTY_INPUT_DIM_UNKNOWN"
    BAD_MMR_LAMBDA = "BAD_MMR_LAMBDA"
    BAD_DELETE = "BAD_DELETE"
    BAD_TOP_K = "BAD_TOP_K"
    FILTER_NOT_SUPPORTED = "FILTER_NOT_SUPPORTED"
    CAPABILITIES_NOT_AVAILABLE = "CAPABILITIES_NOT_AVAILABLE"


# --------------------------------------------------------------------------- #
# Sync-in-async guard (prevents deadlocks in notebook / server environments)
# --------------------------------------------------------------------------- #


def _ensure_not_in_event_loop(sync_api_name: str) -> None:
    """
    Prevent calling sync APIs from inside an active asyncio loop.

    This is a hard guard to avoid deadlocks or unexpected re-entrancy.

    Enhanced behavior:
    - The raised error is context-attached for observability consistency.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return

    err = RuntimeError(
        f"{sync_api_name} was called from inside an active asyncio event loop. "
        f"Use the async variant instead."
    )
    attach_context(
        err,
        framework="langchain",
        operation=sync_api_name,
        error_codes=VECTOR_ERROR_CODES,
        event_loop_active=True,
        sync_api_name=sync_api_name,
    )
    raise err


# --------------------------------------------------------------------------- #
# Error-context decorators (parallel to other framework adapters)
# --------------------------------------------------------------------------- #


def _safe_len(obj: Any) -> Optional[int]:
    try:
        return len(obj)  # type: ignore[arg-type]
    except Exception:
        return None


def _sum_text_chars(texts: Sequence[str]) -> int:
    total = 0
    for t in texts:
        try:
            total += len(t)
        except Exception:
            pass
    return total


def _build_dynamic_error_context(
    operation: str,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    base_context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Add best-effort dynamic context for richer observability without changing behavior.

    Note: This must never raise.
    """
    extra: Dict[str, Any] = dict(base_context)

    try:
        # Add namespace hints where possible.
        if args:
            self_obj = args[0]
            default_ns = getattr(self_obj, "namespace", None)
            if default_ns is not None:
                extra.setdefault("default_namespace", default_ns)
            dim_hint = getattr(self_obj, "_vector_dim_hint", None)
            if dim_hint is not None:
                extra.setdefault("vector_dimension_hint", dim_hint)

        if "namespace" in kwargs:
            extra.setdefault("namespace", kwargs.get("namespace"))
        if "filter" in kwargs:
            extra.setdefault("has_filter", kwargs.get("filter") is not None)

        # Op-specific hints.
        if operation in {"add_texts_sync", "add_texts_async"}:
            texts_obj = args[1] if len(args) >= 2 else None
            if texts_obj is not None:
                try:
                    texts_list = [str(t) for t in list(texts_obj)]
                    extra.setdefault("texts_count", len(texts_list))
                    extra.setdefault("vectors_count", len(texts_list))
                    extra.setdefault("total_content_chars", _sum_text_chars(texts_list))
                except Exception:
                    pass
            ids_obj = kwargs.get("ids")
            if isinstance(ids_obj, list):
                extra.setdefault("ids_count", len(ids_obj))
            if kwargs.get("embeddings") is not None:
                extra.setdefault("has_embeddings", True)

        if operation in {
            "similarity_search_sync",
            "similarity_search_async",
            "similarity_search_stream_sync",
            "similarity_search_with_score_sync",
            "similarity_search_with_score_async",
            "mmr_search_sync",
            "mmr_search_async",
        }:
            query_obj = args[1] if len(args) >= 2 else None
            if isinstance(query_obj, str):
                extra.setdefault("query_chars", len(query_obj))
            k = kwargs.get("k")
            if isinstance(k, int):
                extra.setdefault("k", k)
            fetch_k = kwargs.get("fetch_k")
            if isinstance(fetch_k, int):
                extra.setdefault("fetch_k", fetch_k)
            if kwargs.get("embedding") is not None:
                extra.setdefault("has_embedding", True)

        if operation in {"delete_sync", "delete_async"}:
            ids_obj = kwargs.get("ids")
            if isinstance(ids_obj, list):
                extra.setdefault("ids_count", len(ids_obj))
            extra.setdefault("has_filter", kwargs.get("filter") is not None)
    except Exception:
        pass

    return extra


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
                extra = _build_dynamic_error_context(
                    operation=operation,
                    args=args,
                    kwargs=kwargs,
                    base_context=dict(context_kwargs),
                )
                attach_context(
                    exc,
                    framework="langchain",
                    operation=operation,  # keep original operation string
                    error_codes=VECTOR_ERROR_CODES,
                    **extra,
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
                extra = _build_dynamic_error_context(
                    operation=operation,
                    args=args,
                    kwargs=kwargs,
                    base_context=dict(context_kwargs),
                )
                attach_context(
                    exc,
                    framework="langchain",
                    operation=operation,  # keep original operation string
                    error_codes=VECTOR_ERROR_CODES,
                    **extra,
                )
                raise

        return wrapper

    return decorator


class CorpusLangChainVectorStore(VectorStore):
    """
    LangChain `VectorStore` implementation backed by a Corpus `VectorProtocolV1`.

    This class is a thin integration layer:
    - Documents are mapped to Corpus VectorProtocol `Vector` objects.
    - Similarity search calls map to translator-level `query()` calls.
    - Namespaces + metadata filters are honored based on VectorCapabilities
      (as enforced by the shared VectorTranslator).
    - All sync/async orchestration is delegated to `VectorTranslator`, so this
      adapter does not use any AsyncBridge or sync-stream utilities directly.
    """

    corpus_adapter: VectorProtocolV1
    namespace: Optional[str] = "default"

    id_field: str = "id"
    text_field: str = "page_content"
    metadata_field: Optional[str] = None

    score_threshold: Optional[float] = None
    batch_size: int = 100
    max_query_batch_size: Optional[int] = None
    default_top_k: int = 4

    # Optional embedding integration (sync + async)
    embedding_function: Optional[Callable[[List[str]], Embeddings]] = None
    async_embedding_function: Optional[
        Callable[[List[str]], Awaitable[Embeddings]]
    ] = None

    # Optional custom similarity function for MMR diversity term
    mmr_similarity_fn: Optional[
        Callable[[Sequence[float], Sequence[float]], float]
    ] = None

    # Cached capabilities
    _caps: Optional[VectorCapabilities] = None

    # Dimension hint (best-effort, thread-safe first-write-wins)
    _vector_dim_hint: Optional[int] = None
    _dim_lock: RLock = RLock()

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
        max_query_batch_size: Optional[int] = None,
        default_top_k: int = 4,
        embedding_function: Optional[Callable[[List[str]], Embeddings]] = None,
        async_embedding_function: Optional[
            Callable[[List[str]], Awaitable[Embeddings]]
        ] = None,
        mmr_similarity_fn: Optional[
            Callable[[Sequence[float], Sequence[float]], float]
        ] = None,
    ) -> None:
        """
        Initialize the VectorStore.

        We fail fast with a clear error if LangChain is not installed, but keep
        imports soft and module-import safe.

        Enhanced behavior:
        - The raised error is context-attached for consistency with all framework-facing failures.
        """
        if not LANGCHAIN_AVAILABLE:
            err = RuntimeError(
                "CorpusLangChainVectorStore requires `langchain-core` to be installed. "
                "Install it with `pip install langchain-core`."
            )
            attach_context(
                err,
                framework="langchain",
                operation="init_vector_store",
                error_codes=VECTOR_ERROR_CODES,
                langchain_available=False,
            )
            raise err

        if corpus_adapter is None:
            raise TypeError("corpus_adapter must be provided")

        if score_threshold is not None and not (0.0 <= float(score_threshold) <= 1.0):
            raise ValueError("score_threshold must be between 0.0 and 1.0")
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if int(default_top_k) <= 0:
            raise ValueError("default_top_k must be positive")
        if max_query_batch_size is not None and int(max_query_batch_size) <= 0:
            raise ValueError("max_query_batch_size must be positive")

        self.corpus_adapter = corpus_adapter
        self.namespace = namespace
        self.id_field = id_field
        self.text_field = text_field
        self.metadata_field = metadata_field
        self.score_threshold = score_threshold
        self.batch_size = int(batch_size)
        self.max_query_batch_size = (
            int(max_query_batch_size) if max_query_batch_size is not None else None
        )
        self.default_top_k = int(default_top_k)
        self.embedding_function = embedding_function
        self.async_embedding_function = async_embedding_function
        self.mmr_similarity_fn = mmr_similarity_fn

    # ------------------------------------------------------------------ #
    # Translator + VectorStore-required properties / metadata
    # ------------------------------------------------------------------ #

    @cached_property
    def _translator(self) -> VectorTranslator:
        """
        Lazily construct and cache the `VectorTranslator`.
        """
        framework_translator = LangChainVectorFrameworkTranslator()
        return VectorTranslator(
            adapter=self.corpus_adapter,
            framework="langchain",
            translator=framework_translator,
        )

    @property
    def _vectorstore_type(self) -> str:
        """Identifier used by LangChain in serialization / introspection."""
        return "corpus"

    # ------------------------------------------------------------------ #
    # Capabilities / context helpers
    # ------------------------------------------------------------------ #

    def _get_caps_sync(self) -> VectorCapabilities:
        """
        Synchronously fetch and cache VectorCapabilities.
        """
        if self._caps is not None:
            return self._caps
        try:
            caps = self._translator.capabilities()
            self._caps = caps
            return caps
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="langchain",
                operation="capabilities",
                error_codes=VECTOR_ERROR_CODES,
            )
            raise

    async def _get_caps_async(self) -> VectorCapabilities:
        """
        Async capability fetch with caching.

        Prefer translator.acapabilities(); fall back to translator.capabilities()
        via a thread if needed.
        """
        if self._caps is not None:
            return self._caps

        # Prefer async capabilities if available
        try:
            translator_acapabilities = getattr(self._translator, "acapabilities", None)
        except Exception:
            translator_acapabilities = None

        if callable(translator_acapabilities):
            try:
                caps = await translator_acapabilities()
                self._caps = caps
                return caps
            except Exception as exc:  # noqa: BLE001
                attach_context(
                    exc,
                    framework="langchain",
                    operation="capabilities",
                    error_codes=VECTOR_ERROR_CODES,
                )
                raise

        # Fallback to sync capabilities in a worker thread
        try:
            translator_capabilities = getattr(self._translator, "capabilities", None)
        except Exception:
            translator_capabilities = None

        if callable(translator_capabilities):
            try:
                caps = await asyncio.to_thread(translator_capabilities)
                self._caps = caps
                return caps
            except Exception as exc:  # noqa: BLE001
                attach_context(
                    exc,
                    framework="langchain",
                    operation="capabilities",
                    error_codes=VECTOR_ERROR_CODES,
                )
                raise

        err = NotSupported(
            "VectorTranslator for framework='langchain' must implement "
            "acapabilities() or capabilities(); none found.",
            code=ErrorCodes.CAPABILITIES_NOT_AVAILABLE,
        )
        attach_context(
            err,
            framework="langchain",
            operation="capabilities",
            error_codes=VECTOR_ERROR_CODES,
        )
        raise err

    # --------------------------- #
    # Separated context building
    # --------------------------- #

    def _build_core_context(self, config: Any) -> OperationContext:
        """
        Build an OperationContext from a LangChain config-like object.

        Strict behavior:
        - config is None: return OperationContext() (normal).
        - config is OperationContext: return it.
        - config is Mapping: ctx_from_dict must succeed and return OperationContext.
        - otherwise: ctx_from_langchain must succeed and return OperationContext.
        - If translation is attempted and fails or returns wrong type: raise with attach_context.
        """
        if config is None:
            return OperationContext()

        if isinstance(config, OperationContext):
            return config

        if isinstance(config, Mapping):
            try:
                ctx = ctx_from_dict(config)
            except Exception as exc:  # noqa: BLE001
                attach_context(
                    exc,
                    framework="langchain",
                    operation="vector_context_from_dict",
                    error_codes=VECTOR_ERROR_CODES,
                )
                raise
            try:
                return _coerce_to_vector_operation_context(ctx)
            except Exception as exc:  # noqa: BLE001
                err = BadRequest(
                    f"from_dict produced unsupported context type: {type(ctx).__name__}",
                    code=ErrorCodes.BAD_OPERATION_CONTEXT,
                )
                attach_context(
                    err,
                    framework="langchain",
                    operation="vector_context_from_dict",
                    error_codes=VECTOR_ERROR_CODES,
                )
                raise err from exc

        try:
            ctx = ctx_from_langchain(config)
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="langchain",
                operation="vector_context_from_langchain",
                error_codes=VECTOR_ERROR_CODES,
            )
            raise
        try:
            return _coerce_to_vector_operation_context(ctx)
        except Exception as exc:  # noqa: BLE001
            err = BadRequest(
                f"from_langchain produced unsupported context type: {type(ctx).__name__}",
                code=ErrorCodes.BAD_OPERATION_CONTEXT,
            )
            attach_context(
                err,
                framework="langchain",
                operation="vector_context_from_langchain",
                error_codes=VECTOR_ERROR_CODES,
            )
            raise err from exc

    def _build_ctx(self, **kwargs: Any) -> OperationContext:
        """
        Backward-compatible context builder retained for API stability.

        This method delegates to the strict core context builder.
        """
        config = kwargs.get("config")
        return self._build_core_context(config)

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
        Build a framework_ctx carrying normalized vector context (namespace, etc.)
        for the shared VectorTranslator and downstream adapters.
        """
        ns = self._effective_namespace(namespace)

        raw_ctx: Dict[str, Any] = {}
        if ns is not None:
            raw_ctx["namespace"] = ns
        if extra_context:
            raw_ctx.update(extra_context)

        vector_ctx = normalize_vector_context(
            raw_ctx,
            framework="langchain",
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

    def _build_framework_context(
        self,
        core_ctx: OperationContext,
        *,
        operation: str,
        namespace: Optional[str],
        filter: Optional[Mapping[str, Any]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build framework_ctx metadata for the VectorTranslator.

        Requirements:
        - Depends only on stable, public pieces.
        - Must never raise; on failure, fall back to minimal metadata.
        - Carries vector_context (namespace, etc.) plus stable operation metadata.
        """
        try:
            ns = self._effective_namespace(namespace)
            ctx = self._framework_ctx_for_namespace(ns, extra_context=extra_context)
            # Stable framework-level metadata (non-breaking, best-effort).
            ctx.setdefault("vector_operation", operation)
            ctx.setdefault("has_filter", bool(filter))
            # core_ctx is passed separately as op_ctx; translator already consumes it.
            _ = core_ctx  # keep explicit reference for clarity; no behavior change
            return ctx
        except Exception:
            return {"vector_operation": operation, "has_filter": bool(filter)}

    def _build_contexts(
        self,
        *,
        operation: str,
        namespace: Optional[str],
        filter: Optional[Mapping[str, Any]] = None,
        extra_framework_context: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[OperationContext, Dict[str, Any]]:
        """
        Orchestrate both context layers for all public APIs:
        - _build_core_context(...) -> OperationContext
        - _build_framework_context(...) -> framework_ctx dict for the translator

        This provides a single consistent entry point for context handling.
        """
        core_ctx = self._build_core_context(kwargs.get("config"))
        fw_ctx = self._build_framework_context(
            core_ctx,
            operation=operation,
            namespace=namespace,
            filter=filter,
            extra_context=extra_framework_context,
        )
        return core_ctx, fw_ctx

    # ------------------------------------------------------------------ #
    # Dimension hint + deterministic zero vectors
    # ------------------------------------------------------------------ #

    def _update_dim_hint(self, dim: Optional[int]) -> None:
        """
        Update vector dimension hint (thread-safe, first write wins).
        """
        if dim is None or dim <= 0:
            return
        if self._vector_dim_hint is not None:
            return
        with self._dim_lock:
            if self._vector_dim_hint is None:
                self._vector_dim_hint = int(dim)

    def _zero_vector(self, dim: int) -> List[float]:
        """
        Deterministic zero vector used for empty inputs when dimension is known.
        """
        return [0.0] * int(dim)

    # ------------------------------------------------------------------ #
    # Translation helpers: LC Documents ↔ Corpus Vector
    # ------------------------------------------------------------------ #

    def _ensure_embeddings(
        self,
        texts: List[str],
        embeddings: Optional[Embeddings],
    ) -> Embeddings:
        """
        Ensure embeddings are available for a batch of texts (sync path).

        Updated behavior (empty-input hardening):
        - If embeddings are provided, validate length and update dim hint best-effort.
        - If embeddings must be computed:
            * Compute embeddings only for non-empty texts.
            * Fill empty/whitespace texts with deterministic zero vectors.
            * If all texts are empty and dimension is unknown, raise.
        """
        if not texts:
            return []

        if embeddings is not None:
            if len(embeddings) != len(texts):
                err = BadRequest(
                    f"embeddings length {len(embeddings)} does not match texts length {len(texts)}",
                    code=ErrorCodes.BAD_EMBEDDINGS,
                    details={"texts": len(texts), "embeddings": len(embeddings)},
                )
                attach_context(
                    err,
                    framework="langchain",
                    operation="ensure_embeddings",
                    texts_count=len(texts),
                    embeddings_count=len(embeddings),
                    error_codes=VECTOR_ERROR_CODES,
                )
                raise err
            # Dim hint best-effort from first embedding
            try:
                if embeddings and embeddings[0] is not None:
                    self._update_dim_hint(_safe_len(embeddings[0]))
            except Exception:
                pass
            return embeddings

        empty_idx: List[int] = []
        non_empty_texts: List[str] = []
        non_empty_idx: List[int] = []

        for i, t in enumerate(texts):
            s = str(t)
            if not s.strip():
                empty_idx.append(i)
            else:
                non_empty_texts.append(s)
                non_empty_idx.append(i)

        if not non_empty_texts:
            dim = self._vector_dim_hint
            if dim is None:
                err = BadRequest(
                    "cannot embed empty texts when vector dimension is unknown",
                    code=ErrorCodes.EMPTY_INPUT_DIM_UNKNOWN,
                    details={"texts": len(texts)},
                )
                attach_context(
                    err,
                    framework="langchain",
                    operation="ensure_embeddings",
                    texts_count=len(texts),
                    empty_texts_count=len(texts),
                    error_codes=VECTOR_ERROR_CODES,
                )
                raise err
            return [self._zero_vector(dim) for _ in texts]

        if self.embedding_function is None:
            err = NotSupported(
                "No embedding_function configured; caller must supply embeddings",
                code=ErrorCodes.NO_EMBEDDING_FUNCTION,
                details={"texts": len(texts)},
            )
            attach_context(
                err,
                framework="langchain",
                operation="ensure_embeddings",
                texts_count=len(texts),
                non_empty_texts_count=len(non_empty_texts),
                empty_texts_count=len(empty_idx),
                error_codes=VECTOR_ERROR_CODES,
            )
            raise err

        try:
            computed_non_empty = self.embedding_function(non_empty_texts)
        except Exception as exc:  # noqa: BLE001
            err = BadRequest(
                f"embedding_function failed: {exc}",
                code=ErrorCodes.EMBEDDING_ERROR,
                details={"texts": len(texts), "non_empty_texts": len(non_empty_texts)},
            )
            attach_context(
                err,
                framework="langchain",
                operation="ensure_embeddings",
                texts_count=len(texts),
                non_empty_texts_count=len(non_empty_texts),
                empty_texts_count=len(empty_idx),
                error_codes=VECTOR_ERROR_CODES,
            )
            raise err

        if len(computed_non_empty) != len(non_empty_texts):
            err = BadRequest(
                f"embedding_function returned {len(computed_non_empty)} embeddings for {len(non_empty_texts)} texts",
                code=ErrorCodes.BAD_EMBEDDINGS,
                details={"texts": len(texts), "returned": len(computed_non_empty)},
            )
            attach_context(
                err,
                framework="langchain",
                operation="ensure_embeddings",
                texts_count=len(texts),
                embeddings_count=len(computed_non_empty),
                error_codes=VECTOR_ERROR_CODES,
            )
            raise err

        inferred_dim: Optional[int] = None
        try:
            if computed_non_empty and computed_non_empty[0] is not None:
                inferred_dim = _safe_len(computed_non_empty[0])
        except Exception:
            inferred_dim = None

        if inferred_dim is None or inferred_dim <= 0:
            inferred_dim = self._vector_dim_hint

        if inferred_dim is None or inferred_dim <= 0:
            err = BadRequest(
                "embedding_function produced embeddings with unknown dimension",
                code=ErrorCodes.BAD_EMBEDDINGS,
            )
            attach_context(
                err,
                framework="langchain",
                operation="ensure_embeddings",
                texts_count=len(texts),
                error_codes=VECTOR_ERROR_CODES,
            )
            raise err

        self._update_dim_hint(inferred_dim)

        # Assemble full embeddings aligned to original order
        full: List[List[float]] = [self._zero_vector(inferred_dim) for _ in range(len(texts))]
        for idx, emb in zip(non_empty_idx, computed_non_empty):
            full[idx] = [float(x) for x in emb]

        return full

    async def _ensure_embeddings_async(
        self,
        texts: List[str],
        embeddings: Optional[Embeddings],
    ) -> Embeddings:
        """
        Async-safe embedding helper for a batch of texts.

        Updated behavior mirrors sync version (empty-input hardening).
        """
        if embeddings is not None:
            if len(embeddings) != len(texts):
                err = BadRequest(
                    f"embeddings length {len(embeddings)} does not match texts length {len(texts)}",
                    code=ErrorCodes.BAD_EMBEDDINGS,
                    details={"texts": len(texts), "embeddings": len(embeddings)},
                )
                attach_context(
                    err,
                    framework="langchain",
                    operation="ensure_embeddings_async",
                    texts_count=len(texts),
                    embeddings_count=len(embeddings),
                    error_codes=VECTOR_ERROR_CODES,
                )
                raise err
            try:
                if embeddings and embeddings[0] is not None:
                    self._update_dim_hint(_safe_len(embeddings[0]))
            except Exception:
                pass
            return embeddings

        empty_idx: List[int] = []
        non_empty_texts: List[str] = []
        non_empty_idx: List[int] = []

        for i, t in enumerate(texts):
            s = str(t)
            if not s.strip():
                empty_idx.append(i)
            else:
                non_empty_texts.append(s)
                non_empty_idx.append(i)

        if not non_empty_texts:
            dim = self._vector_dim_hint
            if dim is None:
                err = BadRequest(
                    "cannot embed empty texts when vector dimension is unknown",
                    code=ErrorCodes.EMPTY_INPUT_DIM_UNKNOWN,
                    details={"texts": len(texts)},
                )
                attach_context(
                    err,
                    framework="langchain",
                    operation="ensure_embeddings_async",
                    texts_count=len(texts),
                    empty_texts_count=len(texts),
                    error_codes=VECTOR_ERROR_CODES,
                )
                raise err
            return [self._zero_vector(dim) for _ in texts]

        if self.async_embedding_function is not None:
            try:
                computed_non_empty = await self.async_embedding_function(non_empty_texts)
            except Exception as exc:  # noqa: BLE001
                err = BadRequest(
                    f"async_embedding_function failed: {exc}",
                    code=ErrorCodes.EMBEDDING_ERROR,
                    details={"texts": len(texts), "non_empty_texts": len(non_empty_texts)},
                )
                attach_context(
                    err,
                    framework="langchain",
                    operation="ensure_embeddings_async",
                    texts_count=len(texts),
                    non_empty_texts_count=len(non_empty_texts),
                    empty_texts_count=len(empty_idx),
                    error_codes=VECTOR_ERROR_CODES,
                )
                raise err
        else:
            if self.embedding_function is None:
                err = NotSupported(
                    "No embedding_function/async_embedding_function configured; caller must supply embeddings",
                    code=ErrorCodes.NO_EMBEDDING_FUNCTION,
                    details={"texts": len(texts)},
                )
                attach_context(
                    err,
                    framework="langchain",
                    operation="ensure_embeddings_async",
                    texts_count=len(texts),
                    error_codes=VECTOR_ERROR_CODES,
                )
                raise err
            try:
                computed_non_empty = await asyncio.to_thread(self.embedding_function, non_empty_texts)
            except Exception as exc:  # noqa: BLE001
                err = BadRequest(
                    f"embedding_function failed: {exc}",
                    code=ErrorCodes.EMBEDDING_ERROR,
                    details={"texts": len(texts), "non_empty_texts": len(non_empty_texts)},
                )
                attach_context(
                    err,
                    framework="langchain",
                    operation="ensure_embeddings_async",
                    texts_count=len(texts),
                    non_empty_texts_count=len(non_empty_texts),
                    empty_texts_count=len(empty_idx),
                    error_codes=VECTOR_ERROR_CODES,
                )
                raise err

        if len(computed_non_empty) != len(non_empty_texts):
            err = BadRequest(
                f"embedding function returned {len(computed_non_empty)} embeddings for {len(non_empty_texts)} texts",
                code=ErrorCodes.BAD_EMBEDDINGS,
                details={"texts": len(texts), "returned": len(computed_non_empty)},
            )
            attach_context(
                err,
                framework="langchain",
                operation="ensure_embeddings_async",
                texts_count=len(texts),
                embeddings_count=len(computed_non_empty),
                error_codes=VECTOR_ERROR_CODES,
            )
            raise err

        inferred_dim: Optional[int] = None
        try:
            if computed_non_empty and computed_non_empty[0] is not None:
                inferred_dim = _safe_len(computed_non_empty[0])
        except Exception:
            inferred_dim = None

        if inferred_dim is None or inferred_dim <= 0:
            inferred_dim = self._vector_dim_hint

        if inferred_dim is None or inferred_dim <= 0:
            err = BadRequest(
                "embedding function produced embeddings with unknown dimension",
                code=ErrorCodes.BAD_EMBEDDINGS,
            )
            attach_context(
                err,
                framework="langchain",
                operation="ensure_embeddings_async",
                texts_count=len(texts),
                error_codes=VECTOR_ERROR_CODES,
            )
            raise err

        self._update_dim_hint(inferred_dim)

        full: List[List[float]] = [self._zero_vector(inferred_dim) for _ in range(len(texts))]
        for idx, emb in zip(non_empty_idx, computed_non_empty):
            full[idx] = [float(x) for x in emb]

        return full

    def _normalize_metadatas(
        self,
        n: int,
        metadatas: Optional[List[Metadata]],
    ) -> List[Metadata]:
        """
        Normalize metadata list to length n.
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
            framework="langchain",
            operation="normalize_metadatas",
            texts_count=n,
            metadatas_count=len(metadatas),
            error_codes=VECTOR_ERROR_CODES,
        )
        raise err

    def _normalize_ids(
        self,
        n: int,
        ids: Optional[List[str]],
    ) -> List[str]:
        """
        Normalize IDs list to length n.
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
                framework="langchain",
                operation="normalize_ids",
                texts_count=n,
                ids_count=len(ids),
                error_codes=VECTOR_ERROR_CODES,
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
        Convert LangChain texts + embeddings + metadatas into Corpus `Vector` objects.
        """
        vectors: List[Vector] = []
        ns = self._effective_namespace(namespace)

        for text, emb, meta, vid in zip(texts, embeddings, metadatas, ids):
            # Build metadata payload
            if self.metadata_field:
                # Wrap metadata dict under a single key.
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

            vec = [float(x) for x in emb]
            self._update_dim_hint(len(vec))

            vectors.append(
                Vector(
                    id=str(vid),
                    vector=vec,
                    metadata=metadata_payload,
                    namespace=ns,
                    text=None,
                )
            )

        return vectors

    def _from_corpus_matches(
        self,
        matches: Sequence[VectorMatch],
    ) -> List[Tuple[Document, float]]:
        """
        Convert Corpus `VectorMatch` objects into LangChain Documents + scores.
        """
        results: List[Tuple[Document, float]] = []
        for m in matches:
            v = m.vector
            meta = dict(v.metadata or {})

            # Extract envelope metadata if applicable
            if self.metadata_field and self.metadata_field in meta:
                nested = meta.get(self.metadata_field) or {}
                if isinstance(nested, Mapping):
                    nested_meta = dict(nested)
                else:
                    nested_meta = {}
            else:
                nested_meta = meta

            # Extract text from metadata
            text = meta.get(self.text_field)
            if text is None:
                text = ""

            # Remove internal keys
            nested_meta.pop(self.text_field, None)
            nested_meta.pop(self.id_field, None)

            doc = Document(page_content=str(text), metadata=nested_meta)
            results.append((doc, float(m.score)))
        return results

    def _apply_score_threshold(
        self,
        matches: Sequence[VectorMatch],
    ) -> List[VectorMatch]:
        """
        Apply optional client-side score thresholding to a list of matches.
        """
        if self.score_threshold is None:
            return list(matches)
        threshold = float(self.score_threshold)
        return [m for m in matches if float(m.score) >= threshold]

    # ------------------------------------------------------------------ #
    # Raw request-building helpers for VectorTranslator
    # ------------------------------------------------------------------ #

    def _build_query_request(
        self,
        embedding: Sequence[float],
        *,
        k: int,
        namespace: Optional[str],
        filter: Optional[Mapping[str, Any]],
        include_vectors: bool,
    ) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
        """
        Build a raw query mapping suitable for VectorTranslator.
        """
        ns = self._effective_namespace(namespace)
        raw_query: Dict[str, Any] = {
            "vector": [float(x) for x in embedding],
            "top_k": int(k),
            "namespace": ns,
            "filters": dict(filter) if filter else None,
            "include_metadata": True,
            "include_vectors": bool(include_vectors),
        }
        framework_ctx = self._framework_ctx_for_namespace(ns)
        return raw_query, framework_ctx

    def _build_upsert_request(
        self,
        vectors: List[Vector],
        *,
        namespace: Optional[str],
    ) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
        """
        Build a raw upsert mapping suitable for VectorTranslator.
        """
        ns = self._effective_namespace(namespace)
        raw_request: Dict[str, Any] = {
            "namespace": ns,
            "vectors": vectors,
        }
        framework_ctx = self._framework_ctx_for_namespace(ns)
        return raw_request, framework_ctx

    def _build_delete_request(
        self,
        *,
        ids: Optional[List[str]],
        namespace: Optional[str],
        filter: Optional[Mapping[str, Any]],
    ) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
        """
        Build a raw delete mapping suitable for VectorTranslator.
        """
        ns = self._effective_namespace(namespace)

        if not ids and not filter:
            err = BadRequest(
                "must provide ids or filter for delete",
                code=ErrorCodes.BAD_DELETE,
            )
            attach_context(
                err,
                framework="langchain",
                operation="delete",
                namespace=ns,
                ids_count=0,
                error_codes=VECTOR_ERROR_CODES,
            )
            raise err

        raw_request: Dict[str, Any] = {
            "namespace": ns,
            "ids": [str(i) for i in ids] if ids else None,
            # Translator uses "filters" consistently for metadata filters.
            "filters": dict(filter) if filter else None,
        }
        framework_ctx = self._framework_ctx_for_namespace(ns)
        return raw_request, framework_ctx

    # ------------------------------------------------------------------ #
    # Graceful error recovery for batch operations (partial upserts)
    # ------------------------------------------------------------------ #

    def _handle_partial_upsert_failure(
        self,
        result: UpsertResult,
        total_texts: int,
        namespace: Optional[str],
    ) -> None:
        """
        Handle partial failures in batch upsert operations gracefully.

        Enhanced behavior:
        - If all texts failed, raise a context-attached error (consistent with other failures).
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
                framework="langchain",
                operation="batch_upsert",
                error_codes=VECTOR_ERROR_CODES,
                vectors_count=total_texts,
                namespace=namespace,
            )
            raise err

    # ------------------------------------------------------------------ #
    # Embedding helpers (KEEP original names; add empty-query hardening)
    # ------------------------------------------------------------------ #

    def _embed_query(
        self,
        query: str,
        *,
        embedding: Optional[Sequence[float]] = None,
    ) -> List[float]:
        """
        Ensure a single query embedding is available (sync path).

        Updated behavior:
        - If embedding is provided: use it, update dimension hint.
        - If query is empty/whitespace:
            * If dimension hint is known -> return zero vector.
            * Else -> raise (cannot determine dimension).
        - Otherwise: compute embedding via embedding_function (unchanged semantics).
        """
        if embedding is not None:
            vec = [float(x) for x in embedding]
            self._update_dim_hint(len(vec))
            return vec

        q = "" if query is None else str(query)
        if not q.strip():
            dim = self._vector_dim_hint
            if dim is None:
                err = BadRequest(
                    "query cannot be empty when vector dimension is unknown",
                    code=ErrorCodes.EMPTY_INPUT_DIM_UNKNOWN,
                )
                attach_context(
                    err,
                    framework="langchain",
                    operation="embed_query",
                    error_codes=VECTOR_ERROR_CODES,
                )
                raise err
            return self._zero_vector(dim)

        if self.embedding_function is None:
            err = NotSupported(
                "No embedding_function configured; caller must supply query embedding",
                code=ErrorCodes.NO_EMBEDDING_FUNCTION,
            )
            attach_context(
                err,
                framework="langchain",
                operation="embed_query",
                error_codes=VECTOR_ERROR_CODES,
            )
            raise err

        try:
            embs = self.embedding_function([q])
        except Exception as exc:  # noqa: BLE001
            err = BadRequest(
                f"embedding_function failed for query: {exc}",
                code=ErrorCodes.EMBEDDING_ERROR,
            )
            attach_context(
                err,
                framework="langchain",
                operation="embed_query",
                error_codes=VECTOR_ERROR_CODES,
            )
            raise err
        if not embs or len(embs) != 1:
            err = BadRequest(
                "embedding_function must return exactly one embedding for a single query",
                code=ErrorCodes.BAD_EMBEDDINGS,
                details={"returned": len(embs) if embs is not None else 0},
            )
            attach_context(
                err,
                framework="langchain",
                operation="embed_query",
                error_codes=VECTOR_ERROR_CODES,
            )
            raise err

        vec = [float(x) for x in embs[0]]
        self._update_dim_hint(len(vec))
        return vec

    async def _embed_query_async(
        self,
        query: str,
        *,
        embedding: Optional[Sequence[float]] = None,
    ) -> List[float]:
        """
        Async-safe query embedding helper.

        Updated behavior mirrors sync version for empty-query handling.
        """
        if embedding is not None:
            vec = [float(x) for x in embedding]
            self._update_dim_hint(len(vec))
            return vec

        q = "" if query is None else str(query)
        if not q.strip():
            dim = self._vector_dim_hint
            if dim is None:
                err = BadRequest(
                    "query cannot be empty when vector dimension is unknown",
                    code=ErrorCodes.EMPTY_INPUT_DIM_UNKNOWN,
                )
                attach_context(
                    err,
                    framework="langchain",
                    operation="embed_query_async",
                    error_codes=VECTOR_ERROR_CODES,
                )
                raise err
            return self._zero_vector(dim)

        if self.async_embedding_function is not None:
            try:
                embs = await self.async_embedding_function([q])
            except Exception as exc:  # noqa: BLE001
                err = BadRequest(
                    f"async_embedding_function failed for query: {exc}",
                    code=ErrorCodes.EMBEDDING_ERROR,
                )
                attach_context(
                    err,
                    framework="langchain",
                    operation="embed_query_async",
                    error_codes=VECTOR_ERROR_CODES,
                )
                raise err
        else:
            if self.embedding_function is None:
                err = NotSupported(
                    "No embedding_function/async_embedding_function configured; caller must supply query embedding",
                    code=ErrorCodes.NO_EMBEDDING_FUNCTION,
                )
                attach_context(
                    err,
                    framework="langchain",
                    operation="embed_query_async",
                    error_codes=VECTOR_ERROR_CODES,
                )
                raise err
            try:
                embs = await asyncio.to_thread(self.embedding_function, [q])
            except Exception as exc:  # noqa: BLE001
                err = BadRequest(
                    f"embedding_function failed for query: {exc}",
                    code=ErrorCodes.EMBEDDING_ERROR,
                )
                attach_context(
                    err,
                    framework="langchain",
                    operation="embed_query_async",
                    error_codes=VECTOR_ERROR_CODES,
                )
                raise err

        if not embs or len(embs) != 1:
            err = BadRequest(
                "embedding function must return exactly one embedding for a single query",
                code=ErrorCodes.BAD_EMBEDDINGS,
                details={"returned": len(embs) if embs is not None else 0},
            )
            attach_context(
                err,
                framework="langchain",
                operation="embed_query_async",
                error_codes=VECTOR_ERROR_CODES,
            )
            raise err

        vec = [float(x) for x in embs[0]]
        self._update_dim_hint(len(vec))
        return vec

    # ------------------------------------------------------------------ #
    # Optional: callable/vectorize helpers (ADDITIVE; do not replace originals)
    # ------------------------------------------------------------------ #

    def vectorize_documents(
        self,
        texts: Iterable[str],
        *,
        embeddings: Optional[Embeddings] = None,
    ) -> List[List[float]]:
        """
        Convenience embedding API (sync). Does not change core adapter usage.
        """
        _ensure_not_in_event_loop("vectorize_documents")
        texts_list = [str(t) for t in texts]
        if not texts_list:
            return []
        embs = self._ensure_embeddings(texts_list, embeddings)
        out: List[List[float]] = []
        for e in embs:
            vec = [float(x) for x in e]
            self._update_dim_hint(len(vec))
            out.append(vec)
        return out

    async def avectorize_documents(
        self,
        texts: Iterable[str],
        *,
        embeddings: Optional[Embeddings] = None,
    ) -> List[List[float]]:
        """
        Convenience embedding API (async).
        """
        texts_list = [str(t) for t in texts]
        if not texts_list:
            return []
        embs = await self._ensure_embeddings_async(texts_list, embeddings)
        out: List[List[float]] = []
        for e in embs:
            vec = [float(x) for x in e]
            self._update_dim_hint(len(vec))
            out.append(vec)
        return out

    def vectorize_query(
        self,
        query: str,
        *,
        embedding: Optional[Sequence[float]] = None,
    ) -> List[float]:
        """
        Convenience query embedding API (sync). Delegates to _embed_query.
        """
        _ensure_not_in_event_loop("vectorize_query")
        return self._embed_query(query, embedding=embedding)

    async def avectorize_query(
        self,
        query: str,
        *,
        embedding: Optional[Sequence[float]] = None,
    ) -> List[float]:
        """
        Convenience query embedding API (async). Delegates to _embed_query_async.
        """
        return await self._embed_query_async(query, embedding=embedding)

    def __call__(self, inputs: Any, **kwargs: Any) -> Any:
        """
        Callable interface (additive):

        - If inputs is a string: returns query embedding.
        - If inputs is an iterable of strings: returns embeddings for documents.
        """
        if isinstance(inputs, str):
            return self.vectorize_query(inputs, embedding=kwargs.get("embedding"))
        try:
            seq = list(inputs)
        except Exception:
            seq = [inputs]
        return self.vectorize_documents(seq, embeddings=kwargs.get("embeddings"))

    # ------------------------------------------------------------------ #
    # LangChain VectorStore sync API
    # ------------------------------------------------------------------ #

    @with_error_context("add_texts_sync")
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Metadata]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Add texts to the vector store (sync).
        """
        _ensure_not_in_event_loop("add_texts")

        texts_list = list(texts)
        if not texts_list:
            return []

        embeddings: Optional[Embeddings] = kwargs.get("embeddings")
        namespace: Optional[str] = kwargs.get("namespace")

        ctx, framework_ctx = self._build_contexts(
            operation="add_texts_sync",
            namespace=namespace,
            **kwargs,
        )

        metadatas_norm = self._normalize_metadatas(len(texts_list), metadatas)
        ids_norm = self._normalize_ids(len(texts_list), ids)
        emb = self._ensure_embeddings([str(t) for t in texts_list], embeddings)

        vectors = self._to_corpus_vectors(
            texts=[str(t) for t in texts_list],
            embeddings=emb,
            metadatas=metadatas_norm,
            ids=ids_norm,
            namespace=namespace,
        )

        raw_request, framework_ctx2 = self._build_upsert_request(
            vectors,
            namespace=namespace,
        )
        # Merge orchestrated framework ctx with request-specific vector_context (request wins).
        framework_ctx.update(framework_ctx2)

        result = self._translator.upsert(
            raw_request,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )

        if isinstance(result, UpsertResult):
            self._handle_partial_upsert_failure(
                result,
                total_texts=len(texts_list),
                namespace=namespace,
            )

        return ids_norm

    @with_async_error_context("add_texts_async")
    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Metadata]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Add texts to the vector store (async).
        """
        texts_list = list(texts)
        if not texts_list:
            return []

        embeddings: Optional[Embeddings] = kwargs.get("embeddings")
        namespace: Optional[str] = kwargs.get("namespace")

        ctx, framework_ctx = self._build_contexts(
            operation="add_texts_async",
            namespace=namespace,
            **kwargs,
        )

        metadatas_norm = self._normalize_metadatas(len(texts_list), metadatas)
        ids_norm = self._normalize_ids(len(texts_list), ids)
        emb = await self._ensure_embeddings_async([str(t) for t in texts_list], embeddings)

        vectors = self._to_corpus_vectors(
            texts=[str(t) for t in texts_list],
            embeddings=emb,
            metadatas=metadatas_norm,
            ids=ids_norm,
            namespace=namespace,
        )

        raw_request, framework_ctx2 = self._build_upsert_request(
            vectors,
            namespace=namespace,
        )
        framework_ctx.update(framework_ctx2)

        result = await self._translator.arun_upsert(
            raw_request,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )

        if isinstance(result, UpsertResult):
            self._handle_partial_upsert_failure(
                result,
                total_texts=len(texts_list),
                namespace=namespace,
            )

        return ids_norm

    def add_documents(
        self,
        documents: List[Document],
        **kwargs: Any,
    ) -> List[str]:
        """
        Add LangChain Documents to the vector store (sync).
        """
        _ensure_not_in_event_loop("add_documents")
        texts = [d.page_content for d in documents]
        metadatas = [dict(d.metadata or {}) for d in documents]
        return self.add_texts(texts, metadatas=metadatas, **kwargs)

    async def aadd_documents(
        self,
        documents: List[Document],
        **kwargs: Any,
    ) -> List[str]:
        """
        Add LangChain Documents to the vector store (async).
        """
        texts = [d.page_content for d in documents]
        metadatas = [dict(d.metadata or {}) for d in documents]
        return await self.aadd_texts(texts, metadatas=metadatas, **kwargs)

    @with_error_context("similarity_search_sync")
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Perform similarity search and return Documents (sync).
        """
        _ensure_not_in_event_loop("similarity_search")

        embedding: Optional[Sequence[float]] = kwargs.get("embedding")
        namespace: Optional[str] = kwargs.get("namespace")

        ctx, framework_ctx = self._build_contexts(
            operation="similarity_search_sync",
            namespace=namespace,
            filter=filter,
            **kwargs,
        )

        query_emb = self._embed_query(query, embedding=embedding)
        top_k = k or self.default_top_k

        caps = self._get_caps_sync()
        if caps.max_top_k is not None and top_k > caps.max_top_k:
            err = BadRequest(
                f"top_k {top_k} exceeds maximum of {caps.max_top_k}",
                code=ErrorCodes.BAD_TOP_K,
            )
            attach_context(
                err,
                framework="langchain",
                operation="similarity_search_sync",
                error_codes=VECTOR_ERROR_CODES,
            )
            raise err

        if filter and not caps.supports_metadata_filtering:
            err = NotSupported(
                "metadata filtering is not supported by the underlying vector adapter",
                code=ErrorCodes.FILTER_NOT_SUPPORTED,
                details={"namespace": self._effective_namespace(namespace)},
            )
            attach_context(
                err,
                framework="langchain",
                operation="similarity_search_sync",
                error_codes=VECTOR_ERROR_CODES,
            )
            raise err

        warn_if_extreme_k(
            top_k,
            framework="langchain",
            op_name="similarity_search_sync",
            warning_config=TOPK_WARNING_CONFIG,
            logger=logger,
        )

        raw_query, framework_ctx2 = self._build_query_request(
            query_emb,
            k=top_k,
            namespace=namespace,
            filter=filter,
            include_vectors=False,
        )
        framework_ctx.update(framework_ctx2)

        result_any = self._translator.query(
            raw_query,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )

        if not isinstance(result_any, QueryResult):
            err = VectorAdapterError(
                f"VectorTranslator.query returned unsupported type: {type(result_any).__name__}",
                code=ErrorCodes.BAD_TRANSLATED_RESULT,
            )
            attach_context(
                err,
                framework="langchain",
                operation="similarity_search",
                error_codes=VECTOR_ERROR_CODES,
            )
            raise err

        matches = self._apply_score_threshold(list(result_any.matches or []))
        docs_scores = self._from_corpus_matches(matches)
        return [doc for doc, _ in docs_scores]

    @with_error_context("similarity_search_stream_sync")
    def similarity_search_stream(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Iterator[Document]:
        """
        Streaming similarity search (sync), yielding Documents one by one.
        """
        _ensure_not_in_event_loop("similarity_search_stream")

        embedding: Optional[Sequence[float]] = kwargs.get("embedding")
        namespace: Optional[str] = kwargs.get("namespace")

        ctx, framework_ctx = self._build_contexts(
            operation="similarity_search_stream_sync",
            namespace=namespace,
            filter=filter,
            **kwargs,
        )

        query_emb = self._embed_query(query, embedding=embedding)
        top_k = k or self.default_top_k

        caps = self._get_caps_sync()
        if caps.max_top_k is not None and top_k > caps.max_top_k:
            err = BadRequest(
                f"top_k {top_k} exceeds maximum of {caps.max_top_k}",
                code=ErrorCodes.BAD_TOP_K,
            )
            attach_context(
                err,
                framework="langchain",
                operation="similarity_search_stream_sync",
                error_codes=VECTOR_ERROR_CODES,
            )
            raise err

        if filter and not caps.supports_metadata_filtering:
            err = NotSupported(
                "metadata filtering is not supported by the underlying vector adapter",
                code=ErrorCodes.FILTER_NOT_SUPPORTED,
                details={"namespace": self._effective_namespace(namespace)},
            )
            attach_context(
                err,
                framework="langchain",
                operation="similarity_search_stream_sync",
                error_codes=VECTOR_ERROR_CODES,
            )
            raise err

        warn_if_extreme_k(
            top_k,
            framework="langchain",
            op_name="similarity_search_stream_sync",
            warning_config=TOPK_WARNING_CONFIG,
            logger=logger,
        )

        raw_query, framework_ctx2 = self._build_query_request(
            query_emb,
            k=top_k,
            namespace=namespace,
            filter=filter,
            include_vectors=False,
        )
        framework_ctx.update(framework_ctx2)

        for chunk in self._translator.query_stream(
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
                    code=ErrorCodes.BAD_TRANSLATED_CHUNK,
                )
                attach_context(
                    err,
                    framework="langchain",
                    operation="similarity_search_stream",
                    error_codes=VECTOR_ERROR_CODES,
                )
                raise err

            raw_matches = list(raw_matches_obj or [])
            filtered_matches = self._apply_score_threshold(raw_matches)

            for match in filtered_matches:
                docs_scores = self._from_corpus_matches([match])
                if docs_scores:
                    yield docs_scores[0][0]

    @with_async_error_context("similarity_search_async")
    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Perform similarity search and return Documents (async).
        """
        embedding: Optional[Sequence[float]] = kwargs.get("embedding")
        namespace: Optional[str] = kwargs.get("namespace")

        ctx, framework_ctx = self._build_contexts(
            operation="similarity_search_async",
            namespace=namespace,
            filter=filter,
            **kwargs,
        )

        query_emb = await self._embed_query_async(query, embedding=embedding)
        top_k = k or self.default_top_k

        caps = await self._get_caps_async()
        if caps.max_top_k is not None and top_k > caps.max_top_k:
            err = BadRequest(
                f"top_k {top_k} exceeds maximum of {caps.max_top_k}",
                code=ErrorCodes.BAD_TOP_K,
            )
            attach_context(
                err,
                framework="langchain",
                operation="similarity_search_async",
                error_codes=VECTOR_ERROR_CODES,
            )
            raise err

        if filter and not caps.supports_metadata_filtering:
            err = NotSupported(
                "metadata filtering is not supported by the underlying vector adapter",
                code=ErrorCodes.FILTER_NOT_SUPPORTED,
                details={"namespace": self._effective_namespace(namespace)},
            )
            attach_context(
                err,
                framework="langchain",
                operation="similarity_search_async",
                error_codes=VECTOR_ERROR_CODES,
            )
            raise err

        warn_if_extreme_k(
            top_k,
            framework="langchain",
            op_name="similarity_search_async",
            warning_config=TOPK_WARNING_CONFIG,
            logger=logger,
        )

        raw_query, framework_ctx2 = self._build_query_request(
            query_emb,
            k=top_k,
            namespace=namespace,
            filter=filter,
            include_vectors=False,
        )
        framework_ctx.update(framework_ctx2)

        result_any = await self._translator.arun_query(
            raw_query,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )

        if not isinstance(result_any, QueryResult):
            err = VectorAdapterError(
                f"VectorTranslator.arun_query returned unsupported type: {type(result_any).__name__}",
                code=ErrorCodes.BAD_TRANSLATED_RESULT,
            )
            attach_context(
                err,
                framework="langchain",
                operation="asimilarity_search",
                error_codes=VECTOR_ERROR_CODES,
            )
            raise err

        matches = self._apply_score_threshold(list(result_any.matches or []))
        docs_scores = self._from_corpus_matches(matches)
        return [doc for doc, _ in docs_scores]

    @with_error_context("similarity_search_with_score_sync")
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Similarity search returning (Document, score) tuples (sync).
        """
        _ensure_not_in_event_loop("similarity_search_with_score")

        embedding: Optional[Sequence[float]] = kwargs.get("embedding")
        namespace: Optional[str] = kwargs.get("namespace")

        ctx, framework_ctx = self._build_contexts(
            operation="similarity_search_with_score_sync",
            namespace=namespace,
            filter=filter,
            **kwargs,
        )

        query_emb = self._embed_query(query, embedding=embedding)
        top_k = k or self.default_top_k

        caps = self._get_caps_sync()
        if caps.max_top_k is not None and top_k > caps.max_top_k:
            err = BadRequest(
                f"top_k {top_k} exceeds maximum of {caps.max_top_k}",
                code=ErrorCodes.BAD_TOP_K,
            )
            attach_context(
                err,
                framework="langchain",
                operation="similarity_search_with_score_sync",
                error_codes=VECTOR_ERROR_CODES,
            )
            raise err

        if filter and not caps.supports_metadata_filtering:
            err = NotSupported(
                "metadata filtering is not supported by the underlying vector adapter",
                code=ErrorCodes.FILTER_NOT_SUPPORTED,
                details={"namespace": self._effective_namespace(namespace)},
            )
            attach_context(
                err,
                framework="langchain",
                operation="similarity_search_with_score_sync",
                error_codes=VECTOR_ERROR_CODES,
            )
            raise err

        warn_if_extreme_k(
            top_k,
            framework="langchain",
            op_name="similarity_search_with_score_sync",
            warning_config=TOPK_WARNING_CONFIG,
            logger=logger,
        )

        raw_query, framework_ctx2 = self._build_query_request(
            query_emb,
            k=top_k,
            namespace=namespace,
            filter=filter,
            include_vectors=False,
        )
        framework_ctx.update(framework_ctx2)

        result_any = self._translator.query(
            raw_query,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )

        if not isinstance(result_any, QueryResult):
            err = VectorAdapterError(
                f"VectorTranslator.query returned unsupported type: {type(result_any).__name__}",
                code=ErrorCodes.BAD_TRANSLATED_RESULT,
            )
            attach_context(
                err,
                framework="langchain",
                operation="similarity_search_with_score",
                error_codes=VECTOR_ERROR_CODES,
            )
            raise err

        matches = self._apply_score_threshold(list(result_any.matches or []))
        return self._from_corpus_matches(matches)

    @with_async_error_context("similarity_search_with_score_async")
    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Similarity search returning (Document, score) tuples (async).
        """
        embedding: Optional[Sequence[float]] = kwargs.get("embedding")
        namespace: Optional[str] = kwargs.get("namespace")

        ctx, framework_ctx = self._build_contexts(
            operation="similarity_search_with_score_async",
            namespace=namespace,
            filter=filter,
            **kwargs,
        )

        query_emb = await self._embed_query_async(query, embedding=embedding)
        top_k = k or self.default_top_k

        caps = await self._get_caps_async()
        if caps.max_top_k is not None and top_k > caps.max_top_k:
            err = BadRequest(
                f"top_k {top_k} exceeds maximum of {caps.max_top_k}",
                code=ErrorCodes.BAD_TOP_K,
            )
            attach_context(
                err,
                framework="langchain",
                operation="similarity_search_with_score_async",
                error_codes=VECTOR_ERROR_CODES,
            )
            raise err

        if filter and not caps.supports_metadata_filtering:
            err = NotSupported(
                "metadata filtering is not supported by the underlying vector adapter",
                code=ErrorCodes.FILTER_NOT_SUPPORTED,
                details={"namespace": self._effective_namespace(namespace)},
            )
            attach_context(
                err,
                framework="langchain",
                operation="similarity_search_with_score_async",
                error_codes=VECTOR_ERROR_CODES,
            )
            raise err

        warn_if_extreme_k(
            top_k,
            framework="langchain",
            op_name="similarity_search_with_score_async",
            warning_config=TOPK_WARNING_CONFIG,
            logger=logger,
        )

        raw_query, framework_ctx2 = self._build_query_request(
            query_emb,
            k=top_k,
            namespace=namespace,
            filter=filter,
            include_vectors=False,
        )
        framework_ctx.update(framework_ctx2)

        result_any = await self._translator.arun_query(
            raw_query,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )

        if not isinstance(result_any, QueryResult):
            err = VectorAdapterError(
                f"VectorTranslator.arun_query returned unsupported type: {type(result_any).__name__}",
                code=ErrorCodes.BAD_TRANSLATED_RESULT,
            )
            attach_context(
                err,
                framework="langchain",
                operation="asimilarity_search_with_score",
                error_codes=VECTOR_ERROR_CODES,
            )
            raise err

        matches = self._apply_score_threshold(list(result_any.matches or []))
        return self._from_corpus_matches(matches)

    # ------------------------------------------------------------------ #
    # MMR search (improved + configurable similarity)
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

    def _similarity_for_mmr(
        self,
        a: Sequence[float],
        b: Sequence[float],
    ) -> float:
        """
        Compute similarity between two vectors for MMR.
        """
        if self.mmr_similarity_fn is not None:
            try:
                value = self.mmr_similarity_fn(a, b)
                if isinstance(value, (int, float)):
                    return float(value)
                logger.debug(
                    "mmr_similarity_fn returned non-numeric value %r; falling back to cosine",
                    value,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "mmr_similarity_fn raised %r; falling back to cosine similarity",
                    exc,
                )
        return self._cosine_sim(a, b)

    def _mmr_select_indices(
        self,
        query_vec: Sequence[float],
        candidate_matches: List[VectorMatch],
        k: int,
        lambda_mult: float,
    ) -> List[int]:
        """
        Improved MMR selector that respects original database scores.
        """
        if not candidate_matches or k <= 0:
            return []

        k = min(k, len(candidate_matches))
        if k == 0:
            return []

        if lambda_mult >= 1.0:
            scores = [float(match.score) for match in candidate_matches]
            sorted_indices = sorted(
                range(len(candidate_matches)),
                key=lambda i: scores[i],
                reverse=True,
            )
            return sorted_indices[:k]

        original_scores = [float(match.score) for match in candidate_matches]

        candidate_vecs: List[List[float]] = []
        dim = len(query_vec)
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
                sim = self._similarity_for_mmr(vec_i, vec_j)

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

    @with_error_context("mmr_search_sync")
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        lambda_mult: float = 0.5,
        filter: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Perform Maximal Marginal Relevance (MMR) search (sync).
        """
        _ensure_not_in_event_loop("max_marginal_relevance_search")

        if k <= 0:
            return []

        if not (0.0 <= lambda_mult <= 1.0):
            err = BadRequest(
                f"lambda_mult must be in [0, 1], got {lambda_mult}",
                code=ErrorCodes.BAD_MMR_LAMBDA,
            )
            attach_context(
                err,
                framework="langchain",
                operation="mmr_search",
                lambda_mult=lambda_mult,
                error_codes=VECTOR_ERROR_CODES,
            )
            raise err

        caps = self._get_caps_sync()

        if caps.max_top_k is not None and k > caps.max_top_k:
            err = BadRequest(
                f"k {k} exceeds maximum of {caps.max_top_k}",
                code=ErrorCodes.BAD_TOP_K,
            )
            attach_context(
                err,
                framework="langchain",
                operation="mmr_search_sync",
                error_codes=VECTOR_ERROR_CODES,
            )
            raise err

        if filter and not caps.supports_metadata_filtering:
            err = NotSupported(
                "metadata filtering is not supported by the underlying vector adapter",
                code=ErrorCodes.FILTER_NOT_SUPPORTED,
                details={"namespace": self._effective_namespace(kwargs.get("namespace"))},
            )
            attach_context(
                err,
                framework="langchain",
                operation="mmr_search_sync",
                error_codes=VECTOR_ERROR_CODES,
            )
            raise err

        warn_if_extreme_k(
            k,
            framework="langchain",
            op_name="mmr_search_sync",
            warning_config=TOPK_WARNING_CONFIG,
            logger=logger,
        )

        fetch_k: int = kwargs.get("fetch_k") or max(k * 4, k + 5)

        if caps.max_top_k is not None and fetch_k > caps.max_top_k:
            err = BadRequest(
                f"fetch_k {fetch_k} exceeds maximum of {caps.max_top_k}",
                code=ErrorCodes.BAD_TOP_K,
            )
            attach_context(
                err,
                framework="langchain",
                operation="mmr_search_sync",
                error_codes=VECTOR_ERROR_CODES,
            )
            raise err

        warn_if_extreme_k(
            fetch_k,
            framework="langchain",
            op_name="mmr_search_sync_fetch_k",
            warning_config=TOPK_WARNING_CONFIG,
            logger=logger,
        )

        embedding: Optional[Sequence[float]] = kwargs.get("embedding")
        namespace: Optional[str] = kwargs.get("namespace")

        ctx, framework_ctx = self._build_contexts(
            operation="mmr_search_sync",
            namespace=namespace,
            filter=filter,
            **kwargs,
        )

        query_emb = self._embed_query(query, embedding=embedding)

        raw_query, framework_ctx2 = self._build_query_request(
            query_emb,
            k=fetch_k,
            namespace=namespace,
            filter=filter,
            include_vectors=True,
        )
        framework_ctx.update(framework_ctx2)

        result_any = self._translator.query(
            raw_query,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )

        if not isinstance(result_any, QueryResult):
            err = VectorAdapterError(
                f"VectorTranslator.query returned unsupported type: {type(result_any).__name__}",
                code=ErrorCodes.BAD_TRANSLATED_RESULT,
            )
            attach_context(
                err,
                framework="langchain",
                operation="mmr_search",
                error_codes=VECTOR_ERROR_CODES,
            )
            raise err

        candidate_matches = self._apply_score_threshold(list(result_any.matches or []))
        if not candidate_matches:
            return []

        indices = self._mmr_select_indices(
            query_vec=query_emb,
            candidate_matches=candidate_matches,
            k=k,
            lambda_mult=lambda_mult,
        )

        docs_scores = self._from_corpus_matches(candidate_matches)
        return [docs_scores[i][0] for i in indices]

    @with_async_error_context("mmr_search_async")
    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        lambda_mult: float = 0.5,
        filter: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Perform Maximal Marginal Relevance (MMR) search (async).
        """
        if k <= 0:
            return []

        if not (0.0 <= lambda_mult <= 1.0):
            err = BadRequest(
                f"lambda_mult must be in [0, 1], got {lambda_mult}",
                code=ErrorCodes.BAD_MMR_LAMBDA,
            )
            attach_context(
                err,
                framework="langchain",
                operation="mmr_search_async",
                lambda_mult=lambda_mult,
                error_codes=VECTOR_ERROR_CODES,
            )
            raise err

        caps = await self._get_caps_async()

        if caps.max_top_k is not None and k > caps.max_top_k:
            err = BadRequest(
                f"k {k} exceeds maximum of {caps.max_top_k}",
                code=ErrorCodes.BAD_TOP_K,
            )
            attach_context(
                err,
                framework="langchain",
                operation="mmr_search_async",
                error_codes=VECTOR_ERROR_CODES,
            )
            raise err

        if filter and not caps.supports_metadata_filtering:
            err = NotSupported(
                "metadata filtering is not supported by the underlying vector adapter",
                code=ErrorCodes.FILTER_NOT_SUPPORTED,
                details={"namespace": self._effective_namespace(kwargs.get("namespace"))},
            )
            attach_context(
                err,
                framework="langchain",
                operation="mmr_search_async",
                error_codes=VECTOR_ERROR_CODES,
            )
            raise err

        warn_if_extreme_k(
            k,
            framework="langchain",
            op_name="mmr_search_async",
            warning_config=TOPK_WARNING_CONFIG,
            logger=logger,
        )

        fetch_k: int = kwargs.get("fetch_k") or max(k * 4, k + 5)

        if caps.max_top_k is not None and fetch_k > caps.max_top_k:
            err = BadRequest(
                f"fetch_k {fetch_k} exceeds maximum of {caps.max_top_k}",
                code=ErrorCodes.BAD_TOP_K,
            )
            attach_context(
                err,
                framework="langchain",
                operation="mmr_search_async",
                error_codes=VECTOR_ERROR_CODES,
            )
            raise err

        warn_if_extreme_k(
            fetch_k,
            framework="langchain",
            op_name="mmr_search_async_fetch_k",
            warning_config=TOPK_WARNING_CONFIG,
            logger=logger,
        )

        embedding: Optional[Sequence[float]] = kwargs.get("embedding")
        namespace: Optional[str] = kwargs.get("namespace")

        ctx, framework_ctx = self._build_contexts(
            operation="mmr_search_async",
            namespace=namespace,
            filter=filter,
            **kwargs,
        )

        query_emb = await self._embed_query_async(query, embedding=embedding)

        raw_query, framework_ctx2 = self._build_query_request(
            query_emb,
            k=fetch_k,
            namespace=namespace,
            filter=filter,
            include_vectors=True,
        )
        framework_ctx.update(framework_ctx2)

        result_any = await self._translator.arun_query(
            raw_query,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )

        if not isinstance(result_any, QueryResult):
            err = VectorAdapterError(
                f"VectorTranslator.arun_query returned unsupported type: {type(result_any).__name__}",
                code=ErrorCodes.BAD_TRANSLATED_RESULT,
            )
            attach_context(
                err,
                framework="langchain",
                operation="mmr_search_async",
                error_codes=VECTOR_ERROR_CODES,
            )
            raise err

        candidate_matches = self._apply_score_threshold(list(result_any.matches or []))
        if not candidate_matches:
            return []

        indices = self._mmr_select_indices(
            query_vec=query_emb,
            candidate_matches=candidate_matches,
            k=k,
            lambda_mult=lambda_mult,
        )

        docs_scores = self._from_corpus_matches(candidate_matches)
        return [docs_scores[i][0] for i in indices]

    # ------------------------------------------------------------------ #
    # Delete API
    # ------------------------------------------------------------------ #

    @with_error_context("delete_sync")
    def delete(
        self,
        ids: Optional[List[str]] = None,
        *,
        filter: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Delete vectors by IDs or metadata filter (sync).
        """
        _ensure_not_in_event_loop("delete")

        namespace: Optional[str] = kwargs.get("namespace")

        ctx, framework_ctx = self._build_contexts(
            operation="delete_sync",
            namespace=namespace,
            filter=filter,
            **kwargs,
        )

        raw_request, framework_ctx2 = self._build_delete_request(
            ids=ids,
            namespace=namespace,
            filter=filter,
        )
        framework_ctx.update(framework_ctx2)

        self._translator.delete(
            raw_request,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )

    @with_async_error_context("delete_async")
    async def adelete(
        self,
        ids: Optional[List[str]] = None,
        *,
        filter: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Delete vectors by IDs or metadata filter (async).
        """
        namespace: Optional[str] = kwargs.get("namespace")

        ctx, framework_ctx = self._build_contexts(
            operation="delete_async",
            namespace=namespace,
            filter=filter,
            **kwargs,
        )

        raw_request, framework_ctx2 = self._build_delete_request(
            ids=ids,
            namespace=namespace,
            filter=filter,
        )
        framework_ctx.update(framework_ctx2)

        await self._translator.arun_delete(
            raw_request,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )

    # ------------------------------------------------------------------ #
    # Resource cleanup (full ownership: translator + underlying adapter)
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """
        Best-effort cleanup of translator and underlying adapter resources. Never raises.

        Cleanup order:
        1) Translator (framework orchestration layer)
        2) Underlying adapter/backend (DB/HTTP client, etc.), if it exposes close()
        """
        try:
            close_fn = getattr(self._translator, "close", None)
        except Exception:
            close_fn = None
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                logger.warning("Error while closing VectorTranslator", exc_info=True)

        try:
            adapter_close = getattr(self.corpus_adapter, "close", None)
        except Exception:
            adapter_close = None
        if callable(adapter_close):
            try:
                adapter_close()
            except Exception:
                logger.warning("Error while closing corpus_adapter", exc_info=True)

    async def aclose(self) -> None:
        """
        Best-effort async cleanup of translator and underlying adapter resources. Never raises.

        Cleanup order:
        1) Translator (prefer aclose(), then close() via thread)
        2) Underlying adapter/backend (prefer aclose(), then close() via thread)
        """
        # --- Translator ---
        try:
            aclose_fn = getattr(self._translator, "aclose", None)
        except Exception:
            aclose_fn = None
        if callable(aclose_fn):
            try:
                await aclose_fn()
            except Exception:
                logger.warning("Error while closing VectorTranslator asynchronously", exc_info=True)
        else:
            try:
                close_fn = getattr(self._translator, "close", None)
            except Exception:
                close_fn = None
            if callable(close_fn):
                try:
                    await asyncio.to_thread(close_fn)
                except Exception:
                    logger.warning("Error while closing VectorTranslator (threaded)", exc_info=True)

        # --- Underlying adapter/backend ---
        try:
            adapter_aclose = getattr(self.corpus_adapter, "aclose", None)
        except Exception:
            adapter_aclose = None
        if callable(adapter_aclose):
            try:
                await adapter_aclose()
                return
            except Exception:
                logger.warning("Error while closing corpus_adapter asynchronously", exc_info=True)

        try:
            adapter_close = getattr(self.corpus_adapter, "close", None)
        except Exception:
            adapter_close = None
        if callable(adapter_close):
            try:
                await asyncio.to_thread(adapter_close)
            except Exception:
                logger.warning("Error while closing corpus_adapter (threaded)", exc_info=True)

    def __enter__(self) -> "CorpusLangChainVectorStore":
        _ensure_not_in_event_loop("__enter__")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    async def __aenter__(self) -> "CorpusLangChainVectorStore":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    # ------------------------------------------------------------------ #
    # Convenience constructors
    # ------------------------------------------------------------------ #

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        *,
        corpus_adapter: VectorProtocolV1,
        metadatas: Optional[List[Metadata]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "CorpusLangChainVectorStore":
        """
        Create a store from texts, then add them immediately (sync).
        """
        _ensure_not_in_event_loop("from_texts")
        store = cls(corpus_adapter=corpus_adapter, **kwargs)
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        *,
        corpus_adapter: VectorProtocolV1,
        **kwargs: Any,
    ) -> "CorpusLangChainVectorStore":
        """
        Create a store from Documents, then add them immediately (sync).
        """
        _ensure_not_in_event_loop("from_documents")
        store = cls(corpus_adapter=corpus_adapter, **kwargs)
        store.add_documents(documents)
        return store


class CorpusLangChainRetriever(BaseRetriever):
    """
    LangChain `BaseRetriever` implementation backed by a `CorpusLangChainVectorStore`.
    """

    vector_store: CorpusLangChainVectorStore
    search_kwargs: Dict[str, Any] = {}
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the retriever.

        Enhanced behavior:
        - The raised error is context-attached for consistency with all framework-facing failures.
        """
        if not LANGCHAIN_AVAILABLE:
            err = RuntimeError(
                "CorpusLangChainRetriever requires `langchain-core` to be installed. "
                "Install it with `pip install langchain-core`."
            )
            attach_context(
                err,
                framework="langchain",
                operation="init_retriever",
                error_codes=VECTOR_ERROR_CODES,
                langchain_available=False,
            )
            raise err
        super().__init__(*args, **kwargs)

    @with_error_context("retriever_sync")
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        **kwargs: Any,
    ) -> List[Document]:
        kwargs_combined = {**self.search_kwargs, **kwargs}
        try:
            docs = self.vector_store.similarity_search(query, **kwargs_combined)
            if run_manager is not None:
                # Optional callback hooks could go here.
                pass
            return docs
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="langchain",
                operation="retriever_sync",
                query=query,
                error_codes=VECTOR_ERROR_CODES,
            )
            if run_manager is not None:
                pass
            raise

    @with_async_error_context("retriever_async")
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        **kwargs: Any,
    ) -> List[Document]:
        kwargs_combined = {**self.search_kwargs, **kwargs}
        try:
            docs = await self.vector_store.asimilarity_search(query, **kwargs_combined)
            if run_manager is not None:
                pass
            return docs
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="langchain",
                operation="retriever_async",
                query=query,
                error_codes=VECTOR_ERROR_CODES,
            )
            if run_manager is not None:
                pass
            raise


__all__ = [
    "CorpusLangChainVectorStore",
    "CorpusLangChainRetriever",
    "with_error_context",
    "with_async_error_context",
]
