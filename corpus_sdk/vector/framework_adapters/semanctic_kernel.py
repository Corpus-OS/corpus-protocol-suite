# corpus_sdk/vector/framework_adapters/semantic_kernel.py
# SPDX-License-Identifier: Apache-2.0

"""
Semantic Kernel adapter for Corpus Vector protocol.

This module exposes Corpus `VectorProtocolV1` implementations to
Semantic Kernel in two layers:

1. Core Python API:
   - `CorpusSemanticKernelVectorStore`: protocol-first vector store that
     talks to a Corpus `VectorProtocolV1` via VectorTranslator.
   - Sync + async add/search/delete APIs (mirroring Semantic Kernel patterns)
   - Proper integration with Corpus VectorProtocolV1 via VectorTranslator
   - Namespace + metadata filter handling (capability-aware)
   - Batch upserts and deletes that respect backend limits (enforced in translator)
   - Optional client-side score thresholding
   - Optional embedding function integration (Sync and Async support)
   - Optional streaming search via VectorTranslator.query_stream
   - Optional Maximal Marginal Relevance (MMR) search
   - Comprehensive configuration validation with runtime checks
   - Graceful error recovery for partial batch failures

2. Semantic Kernel plugin:
   - `CorpusSemanticKernelVectorPlugin`: a plugin object that can be
     imported into a Semantic Kernel as a plugin via
       KernelPlugin.from_object(...)
     or `Kernel.add_plugin(...)`.
   - Provides `@kernel_function`-decorated functions that SK can expose
     to the model for tool/function-calling.
   - Full OperationContext propagation via `corpus_sdk.core.context_translation.from_semantic_kernel`
   - Rich error context via `corpus_sdk.core.error_context.attach_context`

Design philosophy
-----------------
- Protocol-first: Semantic Kernel is a thin skin over Corpus vector adapters.
- All heavy lifting (backpressure, deadlines, breakers, batching, etc.) lives in
  the underlying adapter + the shared VectorTranslator, not here.
- This layer focuses on:
    * Translating SK-friendly documents ↔ Corpus Vector objects
    * Respecting VectorCapabilities (namespaces, filters, batch sizes)
    * Delegating sync/async orchestration to VectorTranslator
    * Providing AI-optimized functions for Semantic Kernel tool calling
    * Supporting SK-specific patterns (streaming, context propagation, memory)

Typical usage
-------------

    from semantic_kernel import Kernel
    from semantic_kernel.functions import KernelPlugin
    from corpus_sdk.vector.pinecone_adapter import PineconeVectorAdapter
    from corpus_sdk.vector.framework_adapters.semantic_kernel import (
        CorpusSemanticKernelVectorStore,
        CorpusSemanticKernelVectorPlugin,
    )

    adapter = PineconeVectorAdapter(
        index_name="my-index",
        api_key="...",
        dimensions=1536,
    )

    # Sync embedding function
    def embed_texts(texts: list[str]) -> list[list[float]]:
        ...

    # Async embedding function (optional)
    async def aembed_texts(texts: list[str]) -> list[list[float]]:
        ...

    store = CorpusSemanticKernelVectorStore(
        corpus_adapter=adapter,
        embedding_function=embed_texts,
        async_embedding_function=aembed_texts,  # Optional async support
        namespace="docs",
        default_top_k=4,
    )

    plugin = CorpusSemanticKernelVectorPlugin(vector_store=store)

    kernel = Kernel()
    kernel.add_plugin(plugin, plugin_name="corpus_vector")

    # Now the model can call:
    #   corpus_vector.vector_search(...)
    #   corpus_vector.vector_search_stream(...)
    #   corpus_vector.vector_mmr_search(...)
    #   corpus_vector.vector_store_document(...)
    #   corpus_vector.vector_get_capabilities(...)
"""

from __future__ import annotations

import asyncio
import logging
import math
import uuid
from functools import cached_property
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

from corpus_sdk.core.context_translation import (
    from_semantic_kernel as context_from_semantic_kernel,
    from_dict as context_from_dict,
)
from corpus_sdk.vector.vector_base import (
    VectorProtocolV1,
    Vector,
    VectorMatch,
    QueryResult,
    QueryChunk,
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
from corpus_sdk.core.error_context import attach_context

# Semantic Kernel imports are optional; if SK is not installed, we fall back
# to a no-op decorator so the rest of the SDK remains usable.
try:  # pragma: no cover - import guard
    from semantic_kernel.functions import kernel_function
    from semantic_kernel.exceptions import KernelFunctionException
except Exception:  # pragma: no cover - SK not installed

    def kernel_function(func: Any = None, **_: Any):  # type: ignore[override]
        """
        Fallback decorator when Semantic Kernel is not available.

        Behaves like an identity decorator so this module remains importable
        without SK installed. This is not a placeholder; it is a deliberate
        compatibility shim.
        """

        def _wrap(f: Any) -> Any:
            return f

        # Support both @kernel_function and @kernel_function(...)
        if callable(func):
            return func
        return _wrap

    class KernelFunctionException(Exception):  # type: ignore
        """Fallback exception when SK is not available."""
        pass


logger = logging.getLogger(__name__)

Embeddings = Sequence[Sequence[float]]
Metadata = Dict[str, Any]


# --------------------------------------------------------------------------- #
# Shared vector framework configuration / limits
# --------------------------------------------------------------------------- #

VECTOR_ERROR_CODES = VectorCoercionErrorCodes(framework_label="semantic_kernel")
VECTOR_LIMITS = VectorResourceLimits()
VECTOR_FLAGS = VectorValidationFlags()
TOPK_WARNING_CONFIG = TopKWarningConfig(framework_label="semantic_kernel")


class CorpusSemanticKernelVectorStore:
    """
    Corpus vector store integration for Semantic Kernel.

    This class is framework-agnostic; it does not depend on SK types.
    It provides sync + async APIs optimized for Semantic Kernel patterns:

    - AI-friendly document formats and return types
    - Streaming support for real-time applications
    - MMR search for balanced relevance and diversity
    - Comprehensive configuration validation with runtime enforcement
    - Graceful error recovery for batch operations
    - Capability-aware operations respecting backend limits
    - Optional async embedding function support

    All heavy lifting is delegated to VectorTranslator for consistent
    orchestration across all framework adapters.
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
        embedding_function: Optional[Callable[[List[str]], Embeddings]] = None,
        async_embedding_function: Optional[Callable[[List[str]], Awaitable[Embeddings]]] = None,
    ) -> None:
        # Validate and set configuration with runtime checks
        self.corpus_adapter: VectorProtocolV1 = corpus_adapter
        self.namespace = namespace

        # Validate and set fields with uniqueness check
        self.id_field = str(id_field)
        self.text_field = str(text_field)
        self.metadata_field = str(metadata_field) if metadata_field else None

        # Validate field uniqueness
        reserved_fields = {self.id_field, self.text_field}
        if self.metadata_field:
            reserved_fields.add(self.metadata_field)
        if len(reserved_fields) != (3 if self.metadata_field else 2):
            raise ValueError(
                f"Reserved metadata fields must be unique: {reserved_fields}"
            )

        # Validate and set numeric parameters with bounds checking
        self.score_threshold = self._validate_score_threshold(score_threshold)
        self.batch_size = self._validate_batch_size(batch_size)
        self.default_top_k = self._validate_default_top_k(default_top_k)
        
        # Embedding integration - both sync and async supported
        self.embedding_function = embedding_function
        self.async_embedding_function = async_embedding_function

        # Cached capabilities
        self._caps: Optional[VectorCapabilities] = None

    def _validate_batch_size(self, batch_size: int) -> int:
        """Ensure batch_size is reasonable for vector operations."""
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
        """Ensure default_top_k is reasonable for similarity search."""
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
        """Ensure score_threshold is a valid similarity score."""
        if score_threshold is not None:
            score_threshold = float(score_threshold)
            if score_threshold < 0.0 or score_threshold > 1.0:
                raise ValueError("score_threshold must be between 0.0 and 1.0")
            if score_threshold > 0.9:
                logger.warning(
                    "score_threshold %.2f is very high; may filter out relevant results",
                    score_threshold,
                )
        return score_threshold

    # ------------------------------------------------------------------ #
    # VectorTranslator integration
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
            framework="semantic_kernel",
            translator=framework_translator,
        )

    # ------------------------------------------------------------------ #
    # Capability helpers
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
            attach_context(
                exc,
                framework="semantic_kernel",
                operation="capabilities_sync",
            )
            raise

    async def _get_caps_async(self) -> VectorCapabilities:
        """Async capability fetch with caching via VectorTranslator."""
        if self._caps is not None:
            return self._caps
        try:
            caps = await self._translator.arun_capabilities()
            self._caps = caps
            return caps
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="semantic_kernel",
                operation="capabilities_async",
            )
            raise

    def get_capabilities(self) -> VectorCapabilities:
        """
        Public method to get capabilities without breaking encapsulation.

        Used by the plugin layer to expose capabilities to AI models.
        """
        return self._get_caps_sync()

    async def aget_capabilities(self) -> VectorCapabilities:
        """
        Async public method to get capabilities without breaking encapsulation.
        """
        return await self._get_caps_async()

    def _effective_namespace(self, namespace: Optional[str]) -> Optional[str]:
        """
        Resolve namespace using explicit override or store default.

        If the underlying adapter does not support namespaces, this value is
        still passed down; the adapter may ignore it.
        """
        return namespace if namespace is not None else self.namespace

    def _build_ctx(self, **kwargs: Any) -> Optional[OperationContext]:
        """
        Build an OperationContext from various inputs.

        Priority:
        1. Explicit `ctx` (already an OperationContext)
        2. Semantic Kernel context/settings via `from_semantic_kernel`
           when `sk_context` or `sk_settings` are provided.
        3. Plain dict via `context_dict` and `from_dict`.
        """
        ctx = kwargs.get("ctx")
        if isinstance(ctx, OperationContext):
            return ctx

        sk_context = kwargs.get("sk_context")
        sk_settings = kwargs.get("sk_settings")
        if sk_context is not None or sk_settings is not None:
            try:
                return context_from_semantic_kernel(
                    sk_context,
                    settings=sk_settings,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "context_from_semantic_kernel failed: %s",
                    exc,
                )

        context_dict = kwargs.get("context_dict")
        if isinstance(context_dict, Mapping):
            try:
                return context_from_dict(context_dict)
            except Exception as exc:  # noqa: BLE001
                logger.debug("context_from_dict failed: %s", exc)

        return None

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
            framework="semantic_kernel",
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
    # Translation helpers: text/embeddings ↔ Corpus Vector
    # ------------------------------------------------------------------ #

    def _ensure_embeddings(
        self,
        texts: List[str],
        embeddings: Optional[Embeddings],
    ) -> Embeddings:
        """
        Ensure embeddings are available for a batch of texts.

        Behavior:
        - If embeddings are provided, verify length.
        - Else, if `embedding_function` is set, compute embeddings.
        - Else, raise NotSupported.
        """
        if embeddings is not None:
            if len(embeddings) != len(texts):
                err = BadRequest(
                    f"embeddings length {len(embeddings)} does not match texts length {len(texts)}",
                    code="BAD_EMBEDDINGS",
                    details={"texts": len(texts), "embeddings": len(embeddings)},
                )
                attach_context(
                    err,
                    framework="semantic_kernel",
                    operation="ensure_embeddings",
                    texts_count=len(texts),
                    embeddings_count=len(embeddings),
                )
                raise err
            return embeddings

        if self.embedding_function is None:
            err = NotSupported(
                "No embedding_function configured; caller must supply embeddings",
                code="NO_EMBEDDING_FUNCTION",
                details={"texts": len(texts)},
            )
            attach_context(
                err,
                framework="semantic_kernel",
                operation="ensure_embeddings",
                texts_count=len(texts),
            )
            raise err

        try:
            computed = self.embedding_function(texts)
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="semantic_kernel",
                operation="embedding_function",
            )
            err = BadRequest(
                f"embedding_function failed: {exc}",
                code="EMBEDDING_ERROR",
            )
            attach_context(
                err,
                framework="semantic_kernel",
                operation="ensure_embeddings",
                texts_count=len(texts),
            )
            raise err
        if len(computed) != len(texts):
            err = BadRequest(
                f"embedding_function returned {len(computed)} embeddings for {len(texts)} texts",
                code="BAD_EMBEDDINGS",
            )
            attach_context(
                err,
                framework="semantic_kernel",
                operation="ensure_embeddings",
                texts_count=len(texts),
                embeddings_count=len(computed),
            )
            raise err
        return computed

    async def _ensure_embeddings_async(
        self,
        texts: List[str],
        embeddings: Optional[Embeddings],
    ) -> Embeddings:
        """
        Ensure embeddings are available for a batch of texts (Async path).

        Behavior:
        - If embeddings are provided, verify length.
        - Else, if `async_embedding_function` is set, await it.
        - Else, if `embedding_function` is set, run in thread pool.
        - Else, raise NotSupported.
        """
        if embeddings is not None:
            if len(embeddings) != len(texts):
                err = BadRequest(
                    f"embeddings length {len(embeddings)} does not match texts length {len(texts)}",
                    code="BAD_EMBEDDINGS",
                    details={"texts": len(texts), "embeddings": len(embeddings)},
                )
                attach_context(
                    err,
                    framework="semantic_kernel",
                    operation="ensure_embeddings_async",
                    texts_count=len(texts),
                    embeddings_count=len(embeddings),
                )
                raise err
            return embeddings

        if self.async_embedding_function is not None:
            try:
                computed = await self.async_embedding_function(texts)
            except Exception as exc:  # noqa: BLE001
                attach_context(
                    exc,
                    framework="semantic_kernel",
                    operation="async_embedding_function",
                )
                err = BadRequest(
                    f"async_embedding_function failed: {exc}",
                    code="EMBEDDING_ERROR",
                )
                attach_context(
                    err,
                    framework="semantic_kernel",
                    operation="ensure_embeddings_async",
                    texts_count=len(texts),
                )
                raise err
            if len(computed) != len(texts):
                err = BadRequest(
                    f"async_embedding_function returned {len(computed)} embeddings for {len(texts)} texts",
                    code="BAD_EMBEDDINGS",
                )
                attach_context(
                    err,
                    framework="semantic_kernel",
                    operation="ensure_embeddings_async",
                    texts_count=len(texts),
                    embeddings_count=len(computed),
                )
                raise err
            return computed

        if self.embedding_function is not None:
            try:
                # Run sync embedding function in thread pool to avoid blocking
                computed = await asyncio.to_thread(self.embedding_function, texts)
            except Exception as exc:  # noqa: BLE001
                attach_context(
                    exc,
                    framework="semantic_kernel",
                    operation="embedding_function_thread",
                )
                err = BadRequest(
                    f"embedding_function failed in thread: {exc}",
                    code="EMBEDDING_ERROR",
                )
                attach_context(
                    err,
                    framework="semantic_kernel",
                    operation="ensure_embeddings_async",
                    texts_count=len(texts),
                )
                raise err
            if len(computed) != len(texts):
                err = BadRequest(
                    f"embedding_function returned {len(computed)} embeddings for {len(texts)} texts",
                    code="BAD_EMBEDDINGS",
                )
                attach_context(
                    err,
                    framework="semantic_kernel",
                    operation="ensure_embeddings_async",
                    texts_count=len(texts),
                    embeddings_count=len(computed),
                )
                raise err
            return computed

        err = NotSupported(
            "No embedding_function or async_embedding_function configured; caller must supply embeddings",
            code="NO_EMBEDDING_FUNCTION",
            details={"texts": len(texts)},
        )
        attach_context(
            err,
            framework="semantic_kernel",
            operation="ensure_embeddings_async",
            texts_count=len(texts),
        )
        raise err

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
            framework="semantic_kernel",
            operation="normalize_metadatas",
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
                framework="semantic_kernel",
                operation="normalize_ids",
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

    def _from_corpus_matches(
        self,
        matches: Sequence[VectorMatch],
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        Convert Corpus `VectorMatch` objects into (text, metadata, score) tuples.

        The similarity score returned is `VectorMatch.score`, which is assumed
        to be higher-is-better (normalized similarity as defined by the adapter).
        """
        results: List[Tuple[str, Dict[str, Any], float]] = []
        for m in matches:
            v = m.vector
            meta = dict(v.metadata or {})

            if self.metadata_field and self.metadata_field in meta:
                nested = meta.get(self.metadata_field) or {}
                if isinstance(nested, Mapping):
                    nested_meta = dict(nested)
                else:
                    nested_meta = {}
            else:
                nested_meta = meta

            text = meta.get(self.text_field)
            if text is None:
                text = ""

            nested_meta.pop(self.text_field, None)
            nested_meta.pop(self.id_field, None)

            results.append((text, nested_meta, float(m.score)))
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

    def _format_for_ai_model(self, matches: List[VectorMatch]) -> List[Dict[str, Any]]:
        """
        Format results for optimal consumption by AI models in SK.

        Returns consistent structure with clear confidence scores and
        token-efficient content formatting. Filters out internal system fields.
        """

        def _truncate_for_tokens(text: str, max_length: int = 500) -> str:
            """Truncate text for token efficiency in AI model responses."""
            if len(text) <= max_length:
                return text
            return text[:max_length].rsplit(" ", 1)[0] + "..."

        # Define fields that should be filtered out as they're internal/system
        internal_fields = {"id", "vector", "_id", "_vector", "embedding", "timestamp"}

        return [
            {
                "content": _truncate_for_tokens(text),
                "metadata": {k: v for k, v in meta.items() if k not in internal_fields},
                "confidence": round(score, 3),
                "source": "vector_database",
            }
            for text, meta, score in self._from_corpus_matches(matches)
        ]

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
        Only raises exceptions for complete failures or critical errors.
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

            # Log details of individual failures for debugging
            if result.failures:
                for failure in result.failures[:5]:  # Log first 5 failures
                    logger.debug("Upsert failure: %s", failure)
                if len(result.failures) > 5:
                    logger.debug(
                        "... and %d more failures", len(result.failures) - 5
                    )

        # Only raise exception if no texts were upserted at all
        if (result.upserted_count or 0) == 0 and total_texts > 0:
            raise VectorAdapterError(
                f"All {total_texts} texts failed to upsert",
                code="BATCH_UPSERT_FAILED",
                details={
                    "total_texts": total_texts,
                    "namespace": namespace,
                    "failures": result.failures or [],
                },
            )

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

        Returns the effective top_k value (possibly reduced to max_top_k).
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
                framework="semantic_kernel",
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
                framework="semantic_kernel",
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

        Returns the effective top_k value (possibly reduced to max_top_k).
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
                framework="semantic_kernel",
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
                framework="semantic_kernel",
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
                framework="semantic_kernel",
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
                framework="semantic_kernel",
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
                framework="semantic_kernel",
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
                framework="semantic_kernel",
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
            framework="semantic_kernel",
            operation=operation,
        )
        raise err

    # ------------------------------------------------------------------ #
    # Embedding helper for queries
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
        - If `embedding` is provided, use it.
        - Else, if `embedding_function` is set, compute embedding for [query].
        - Else, raise NotSupported.
        """
        if embedding is not None:
            return [float(x) for x in embedding]

        if self.embedding_function is None:
            exc = NotSupported(
                "No embedding_function configured; caller must supply query embedding",
                code="NO_EMBEDDING_FUNCTION",
            )
            attach_context(
                exc,
                framework="semantic_kernel",
                operation="embed_query",
            )
            raise exc

        try:
            embs = self.embedding_function([query])
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="semantic_kernel",
                operation="embed_query",
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

    async def _embed_query_async(
        self,
        query: str,
        *,
        embedding: Optional[Sequence[float]] = None,
    ) -> List[float]:
        """
        Async-safe query embedding helper.

        Behavior:
        - If `embedding` provided, use it.
        - If `async_embedding_function` set, await it.
        - If `embedding_function` set, run in thread pool.
        - Else raise NotSupported.
        """
        if embedding is not None:
            return [float(x) for x in embedding]

        if self.async_embedding_function is not None:
            try:
                embs = await self.async_embedding_function([query])
            except Exception as exc:  # noqa: BLE001
                attach_context(
                    exc,
                    framework="semantic_kernel",
                    operation="async_embed_query",
                )
                raise BadRequest(
                    f"async_embedding_function failed for query: {exc}",
                    code="EMBEDDING_ERROR",
                )
            if not embs or len(embs) != 1:
                raise BadRequest(
                    "async_embedding_function must return exactly one embedding for a single query",
                    code="BAD_EMBEDDINGS",
                )
            return [float(x) for x in embs[0]]

        if self.embedding_function is not None:
            try:
                # Run sync embedding function in thread pool
                embs = await asyncio.to_thread(self.embedding_function, [query])
            except Exception as exc:  # noqa: BLE001
                attach_context(
                    exc,
                    framework="semantic_kernel",
                    operation="embed_query_thread",
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

        exc = NotSupported(
            "No embedding_function or async_embedding_function configured; caller must supply query embedding",
            code="NO_EMBEDDING_FUNCTION",
        )
        attach_context(
            exc,
            framework="semantic_kernel",
            operation="embed_query_async",
        )
        raise exc

    # ------------------------------------------------------------------ #
    # Public sync/async API using VectorTranslator
    # ------------------------------------------------------------------ #

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Metadata]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Add texts to the vector store (sync).

        Behavior:
        - Uses `embedding_function` if configured, unless explicit `embeddings`
          are passed via kwargs.
        - Delegates batching + capability-aware upserts to VectorTranslator.
        - Handles partial failures gracefully via _handle_partial_upsert_failure.
        """
        texts_list = list(texts)
        if not texts_list:
            return []

        embeddings: Optional[Embeddings] = kwargs.get("embeddings")
        namespace: Optional[str] = kwargs.get("namespace")
        ctx = self._build_ctx(**kwargs)

        metadatas_norm = self._normalize_metadatas(len(texts_list), metadatas)
        ids_norm = self._normalize_ids(len(texts_list), ids)
        emb = self._ensure_embeddings(texts_list, embeddings)

        vectors = self._to_corpus_vectors(
            texts=texts_list,
            embeddings=emb,
            metadatas=metadatas_norm,
            ids=ids_norm,
            namespace=namespace,
        )

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
                self._handle_partial_upsert_failure(
                    result,
                    len(texts_list),
                    namespace,
                )

        except Exception:
            # Errors already have context attached by the translator.
            raise

        return ids_norm

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
        ctx = self._build_ctx(**kwargs)

        metadatas_norm = self._normalize_metadatas(len(texts_list), metadatas)
        ids_norm = self._normalize_ids(len(texts_list), ids)
        
        # Use async-safe embedding resolution
        emb = await self._ensure_embeddings_async(texts_list, embeddings)

        vectors = self._to_corpus_vectors(
            texts=texts_list,
            embeddings=emb,
            metadatas=metadatas_norm,
            ids=ids_norm,
            namespace=namespace,
        )

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
                self._handle_partial_upsert_failure(
                    result,
                    len(texts_list),
                    namespace,
                )

        except Exception:
            # Translator/adapter already annotated the error.
            raise

        return ids_norm

    def add_documents(
        self,
        documents: Iterable[Mapping[str, Any]],
        **kwargs: Any,
    ) -> List[str]:
        """
        Add generic document-like objects to the vector store (sync).

        Expected shape:
            {"page_content": str, "metadata": dict}
        """
        texts: List[str] = []
        metadatas: List[Metadata] = []
        for d in documents:
            texts.append(str(d.get("page_content", "")))
            meta = d.get("metadata") or {}
            metadatas.append(dict(meta))
        return self.add_texts(texts, metadatas=metadatas, **kwargs)

    async def aadd_documents(
        self,
        documents: Iterable[Mapping[str, Any]],
        **kwargs: Any,
    ) -> List[str]:
        """
        Add generic document-like objects to the vector store (async).
        """
        texts: List[str] = []
        metadatas: List[Metadata] = []
        for d in documents:
            texts.append(str(d.get("page_content", "")))
            meta = d.get("metadata") or {}
            metadatas.append(dict(meta))
        return await self.aadd_texts(texts, metadatas=metadatas, **kwargs)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search and return AI-optimized document dicts (sync).

        Return shape:
            [
                {
                    "content": str,
                    "metadata": dict,
                    "confidence": float,
                    "source": "vector_database",
                },
                ...
            ]
        """
        embedding: Optional[Sequence[float]] = kwargs.get("embedding")
        namespace: Optional[str] = kwargs.get("namespace")
        ctx = self._build_ctx(**kwargs)

        query_emb = self._embed_query(query, embedding=embedding)
        top_k = k or self.default_top_k
        top_k = self._validate_query_params_sync(top_k, namespace, filter)

        warn_if_extreme_k(
            top_k,
            framework="semantic_kernel",
            op_name="similarity_search_sync",
            warning_config=TOPK_WARNING_CONFIG,
            logger=logger,
        )

        raw_query, framework_ctx = self._build_query_request(
            query_emb,
            top_k=top_k,
            namespace=namespace,
            filter=filter,
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
                framework="semantic_kernel",
                operation="similarity_search_sync",
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

        return self._format_for_ai_model(matches_list)

    def similarity_search_stream(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Iterator[Dict[str, Any]]:
        """
        Streaming similarity search (sync), yielding AI-optimized document dicts.

        Each yielded item has the shape:
            {
                "content": str,
                "metadata": dict,
                "confidence": float,
                "source": "vector_database",
            }
        """
        embedding: Optional[Sequence[float]] = kwargs.get("embedding")
        namespace: Optional[str] = kwargs.get("namespace")
        ctx = self._build_ctx(**kwargs)
        top_k = k or self.default_top_k
        top_k = self._validate_query_params_sync(top_k, namespace, filter)

        warn_if_extreme_k(
            top_k,
            framework="semantic_kernel",
            op_name="similarity_search_stream",
            warning_config=TOPK_WARNING_CONFIG,
            logger=logger,
        )

        query_emb = self._embed_query(query, embedding=embedding)

        raw_query, framework_ctx = self._build_query_request(
            query_emb,
            top_k=top_k,
            namespace=namespace,
            filter=filter,
            include_vectors=False,
        )

        try:
            for chunk in self._translator.query_stream(
                raw_query,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            ):
                if not isinstance(chunk, QueryChunk):
                    err = VectorAdapterError(
                        f"VectorTranslator.query_stream yielded unsupported type: {type(chunk).__name__}",
                        code="BAD_STREAM_CHUNK",
                    )
                    attach_context(
                        err,
                        framework="semantic_kernel",
                        operation="similarity_search_stream",
                        namespace=self._effective_namespace(namespace),
                        top_k=top_k,
                    )
                    raise err

                raw_matches = list(chunk.matches or [])
                filtered_matches = self._apply_score_threshold(raw_matches)

                for match in filtered_matches:
                    formatted = self._format_for_ai_model([match])
                    if formatted:
                        yield formatted[0]
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="semantic_kernel",
                operation="similarity_search_stream",
                namespace=self._effective_namespace(namespace),
                top_k=top_k,
            )
            raise

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search and return AI-optimized document dicts (async).

        Return shape:
            [
                {
                    "content": str,
                    "metadata": dict,
                    "confidence": float,
                    "source": "vector_database",
                },
                ...
            ]
        """
        embedding: Optional[Sequence[float]] = kwargs.get("embedding")
        namespace: Optional[str] = kwargs.get("namespace")
        ctx = self._build_ctx(**kwargs)

        query_emb = await self._embed_query_async(query, embedding=embedding)
        top_k = k or self.default_top_k
        top_k = await self._validate_query_params_async(top_k, namespace, filter)

        warn_if_extreme_k(
            top_k,
            framework="semantic_kernel",
            op_name="similarity_search_async",
            warning_config=TOPK_WARNING_CONFIG,
            logger=logger,
        )

        raw_query, framework_ctx = self._build_query_request(
            query_emb,
            top_k=top_k,
            namespace=namespace,
            filter=filter,
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

        return self._format_for_ai_model(matches_list)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Similarity search returning (doc_dict, confidence) tuples (sync).

        Each doc_dict has the AI-optimized shape:
            {"content", "metadata", "confidence", "source"}.
        """
        docs = self.similarity_search(query, k=k, filter=filter, **kwargs)
        return [(doc, float(doc["confidence"])) for doc in docs]

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Similarity search returning (doc_dict, confidence) tuples (async).

        Each doc_dict has the AI-optimized shape:
            {"content", "metadata", "confidence", "source"}.
        """
        docs = await self.asimilarity_search(query, k=k, filter=filter, **kwargs)
        return [(doc, float(doc["confidence"])) for doc in docs]

    # ------------------------------------------------------------------ #
    # MMR search (improved implementation)
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

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        lambda_mult: float = 0.5,
        filter: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Perform Maximal Marginal Relevance (MMR) search (sync).

        This runs a similarity search with a larger `fetch_k` and then
        selects a subset of results via MMR based on vector geometry and
        original database scores. Returns AI-optimized document dicts:
            {"content", "metadata", "confidence", "source"}.
        """
        if not (0.0 <= lambda_mult <= 1.0):
            err = BadRequest(
                f"lambda_mult must be in [0, 1], got {lambda_mult}",
                code="BAD_MMR_LAMBDA",
            )
            attach_context(
                err,
                framework="semantic_kernel",
                operation="mmr_search_sync",
                lambda_mult=lambda_mult,
            )
            raise err

        fetch_k: int = kwargs.get("fetch_k") or max(k * 4, k + 5)
        embedding: Optional[Sequence[float]] = kwargs.get("embedding")
        namespace: Optional[str] = kwargs.get("namespace")
        ctx = self._build_ctx(**kwargs)

        # Validate fetch_k against capabilities (treating it as top_k for the query)
        fetch_k = self._validate_query_params_sync(fetch_k, namespace, filter)

        warn_if_extreme_k(
            k,
            framework="semantic_kernel",
            op_name="mmr_search_sync",
            warning_config=TOPK_WARNING_CONFIG,
            logger=logger,
        )

        warn_if_extreme_k(
            fetch_k,
            framework="semantic_kernel",
            op_name="mmr_search_sync_fetch_k",
            warning_config=TOPK_WARNING_CONFIG,
            logger=logger,
        )

        query_emb = self._embed_query(query, embedding=embedding)

        raw_query, framework_ctx = self._build_query_request(
            query_emb,
            top_k=fetch_k,
            namespace=namespace,
            filter=filter,
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
            return []

        indices = self._mmr_select_indices(
            query_vec=query_emb,
            candidate_matches=matches_list,
            k=k,
            lambda_mult=lambda_mult,
        )

        selected_matches = [matches_list[i] for i in indices]
        return self._format_for_ai_model(selected_matches)

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        lambda_mult: float = 0.5,
        filter: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Perform Maximal Marginal Relevance (MMR) search (async).

        This runs a similarity search with a larger `fetch_k` and then
        selects a subset of results via MMR based on vector geometry and
        original database scores. Returns AI-optimized document dicts:
            {"content", "metadata", "confidence", "source"}.
        """
        if not (0.0 <= lambda_mult <= 1.0):
            err = BadRequest(
                f"lambda_mult must be in [0, 1], got {lambda_mult}",
                code="BAD_MMR_LAMBDA",
            )
            attach_context(
                err,
                framework="semantic_kernel",
                operation="mmr_search_async",
                lambda_mult=lambda_mult,
            )
            raise err

        fetch_k: int = kwargs.get("fetch_k") or max(k * 4, k + 5)
        embedding: Optional[Sequence[float]] = kwargs.get("embedding")
        namespace: Optional[str] = kwargs.get("namespace")
        ctx = self._build_ctx(**kwargs)

        # Validate fetch_k against capabilities (treating it as top_k for the query)
        fetch_k = await self._validate_query_params_async(fetch_k, namespace, filter)

        warn_if_extreme_k(
            k,
            framework="semantic_kernel",
            op_name="mmr_search_async",
            warning_config=TOPK_WARNING_CONFIG,
            logger=logger,
        )

        warn_if_extreme_k(
            fetch_k,
            framework="semantic_kernel",
            op_name="mmr_search_async_fetch_k",
            warning_config=TOPK_WARNING_CONFIG,
            logger=logger,
        )

        query_emb = await self._embed_query_async(query, embedding=embedding)

        raw_query, framework_ctx = self._build_query_request(
            query_emb,
            top_k=fetch_k,
            namespace=namespace,
            filter=filter,
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
            return []

        indices = self._mmr_select_indices(
            query_vec=query_emb,
            candidate_matches=matches_list,
            k=k,
            lambda_mult=lambda_mult,
        )

        selected_matches = [matches_list[i] for i in indices]
        return self._format_for_ai_model(selected_matches)

    # ------------------------------------------------------------------ #
    # Delete API
    # ------------------------------------------------------------------ #

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
        namespace: Optional[str] = kwargs.get("namespace")
        ctx = self._build_ctx(**kwargs)

        self._validate_delete_params_sync(
            ids=ids,
            namespace=namespace,
            filter=filter,
        )

        raw_request, framework_ctx = self._build_delete_request(
            ids=ids,
            namespace=namespace,
            filter=filter,
        )

        try:
            self._translator.delete(
                raw_request,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
        except Exception:
            # Translator/adapter already annotated the error.
            raise

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
        ctx = self._build_ctx(**kwargs)

        await self._validate_delete_params_async(
            ids=ids,
            namespace=namespace,
            filter=filter,
        )

        raw_request, framework_ctx = self._build_delete_request(
            ids=ids,
            namespace=namespace,
            filter=filter,
        )

        await self._translator.arun_delete(
            raw_request,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )

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
    ) -> "CorpusSemanticKernelVectorStore":
        """
        Create a store from texts, then add them immediately (sync).
        """
        store = cls(corpus_adapter=corpus_adapter, **kwargs)
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Mapping[str, Any]],
        *,
        corpus_adapter: VectorProtocolV1,
        **kwargs: Any,
    ) -> "CorpusSemanticKernelVectorStore":
        """
        Create a store from documents, then add them immediately (sync).

        Expected document shape:
            {"page_content": str, "metadata": dict}
        """
        store = cls(corpus_adapter=corpus_adapter, **kwargs)
        store.add_documents(documents)
        return store


class CorpusSemanticKernelVectorPlugin:
    """
    Semantic Kernel plugin exposing Corpus vector search.

    This plugin wraps a `CorpusSemanticKernelVectorStore` and provides
    `@kernel_function`-decorated methods that the model can call via
    SK's function-calling / tool mechanisms.

    The plugin is intentionally thin; all heavy lifting is delegated to
    the vector store, which talks to the underlying `VectorProtocolV1`.
    """

    def __init__(
        self,
        *,
        vector_store: CorpusSemanticKernelVectorStore,
        framework_version: Optional[str] = None,
    ) -> None:
        self.vector_store = vector_store
        self.framework_version = framework_version

    def _build_sk_aware_context(
        self,
        sk_context: Optional[Any],
        sk_settings: Optional[Any],
    ) -> OperationContext:
        """
        Enhanced context translation that understands SK-specific patterns.

        Includes SK variables, skill context, and other SK-specific metadata
        for comprehensive tracing and debugging.
        """
        try:
            base_ctx = context_from_semantic_kernel(
                sk_context,
                settings=sk_settings,
                framework_version=self.framework_version,
            )

            # Add SK-specific context extensions
            if hasattr(sk_context, "variables") and sk_context.variables:
                base_ctx.extra_context["sk_variables"] = dict(sk_context.variables)
            if hasattr(sk_context, "skill_name") and sk_context.skill_name:
                base_ctx.extra_context["sk_skill"] = sk_context.skill_name
            if hasattr(sk_context, "function_name") and sk_context.function_name:
                base_ctx.extra_context["sk_function"] = sk_context.function_name

            return base_ctx
        except Exception as exc:  # noqa: BLE001
            logger.debug("context_from_semantic_kernel failed in plugin: %s", exc)
            # Fall back to empty context
            return context_from_dict({})

    def _handle_sk_plugin_error(
        self,
        exc: Exception,
        operation: str,
        **context: Any,
    ) -> None:
        """
        SK plugins need particularly good error handling since they're often
        called autonomously by AI models without human intervention.
        """
        attach_context(
            exc,
            framework="semantic_kernel",
            operation=operation,
            **context,
        )

        # Convert to SK-friendly error format
        if isinstance(exc, NotSupported):
            raise KernelFunctionException(
                f"Vector operation not supported: {exc}",
                exc.code if hasattr(exc, "code") else "NOT_SUPPORTED",
            )
        elif isinstance(exc, BadRequest):
            raise KernelFunctionException(
                f"Invalid vector operation: {exc}",
                exc.code if hasattr(exc, "code") else "BAD_REQUEST",
            )
        elif isinstance(exc, VectorAdapterError):
            raise KernelFunctionException(
                f"Vector database error: {exc}",
                exc.code if hasattr(exc, "code") else "VECTOR_ERROR",
            )
        else:
            # Generic error for unexpected exceptions
            raise KernelFunctionException(
                f"Vector search failed: {exc}",
                "INTERNAL_ERROR",
            )

    @kernel_function(
        name="vector_search",
        description=(
            "Search for semantically similar documents in the vector database. "
            "Use this when you need to find relevant information based on meaning "
            "rather than keyword matching. Returns documents with content, metadata, and confidence scores."
        ),
    )
    async def vector_search(
        self,
        query: str,
        k: int = 4,
        *,
        filter: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
        sk_context: Optional[Any] = None,
        sk_settings: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic Kernel-exposed similarity search (async).

        This is the primary function SK will call during tool/function-calling.
        Returns AI-optimized document format for easy consumption by models.
        """
        ctx = self._build_sk_aware_context(sk_context, sk_settings)

        try:
            docs = await self.vector_store.asimilarity_search(
                query,
                k=k,
                filter=filter,
                namespace=namespace,
                ctx=ctx,
            )
            return docs
        except Exception as exc:  # noqa: BLE001
            self._handle_sk_plugin_error(
                exc,
                operation="plugin_vector_search",
                query=query,
                k=k,
                namespace=namespace,
            )

    @kernel_function(
        name="vector_search_stream",
        description=(
            "Streaming variant of vector_search. Yields documents one by one "
            "as they are retrieved. Each item has content, metadata, and confidence score. "
            "Useful for real-time applications and progressive UI updates."
        ),
    )
    def vector_search_stream(
        self,
        query: str,
        k: int = 4,
        *,
        filter: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
        sk_context: Optional[Any] = None,
        sk_settings: Optional[Any] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Semantic Kernel-exposed streaming similarity search (sync generator).

        SK will detect that this is a streaming kernel function because it
        returns an iterator. Uses VectorTranslator.query_stream under the hood.
        """
        ctx = self._build_sk_aware_context(sk_context, sk_settings)

        try:
            for doc in self.vector_store.similarity_search_stream(
                query,
                k=k,
                filter=filter,
                namespace=namespace,
                ctx=ctx,
            ):
                yield doc
        except Exception as exc:  # noqa: BLE001
            self._handle_sk_plugin_error(
                exc,
                operation="plugin_vector_search_stream",
                query=query,
                k=k,
                namespace=namespace,
            )

    @kernel_function(
        name="vector_mmr_search",
        description=(
            "Maximal Marginal Relevance (MMR) search over the Corpus vector index. "
            "Balances relevance and diversity to provide a varied set of results "
            "while maintaining high relevance. Returns a list of documents with "
            "content, metadata, and confidence scores."
        ),
    )
    async def vector_mmr_search(
        self,
        query: str,
        k: int = 4,
        lambda_mult: float = 0.5,
        *,
        filter: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
        sk_context: Optional[Any] = None,
        sk_settings: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic Kernel-exposed MMR search (async).

        Provides balanced results between relevance and diversity, useful for
        exploration and avoiding redundant information.
        """
        ctx = self._build_sk_aware_context(sk_context, sk_settings)

        try:
            docs = await self.vector_store.amax_marginal_relevance_search(
                query,
                k=k,
                lambda_mult=lambda_mult,
                filter=filter,
                namespace=namespace,
                ctx=ctx,
            )
            return docs
        except Exception as exc:  # noqa: BLE001
            self._handle_sk_plugin_error(
                exc,
                operation="plugin_vector_mmr_search",
                query=query,
                k=k,
                lambda_mult=lambda_mult,
                namespace=namespace,
            )

    @kernel_function(
        name="vector_store_document",
        description=(
            "Store a document or piece of information in the vector database for later retrieval. "
            "Use this to add new knowledge or memories that can be searched later."
        ),
    )
    async def store_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        namespace: Optional[str] = None,
        sk_context: Optional[Any] = None,
        sk_settings: Optional[Any] = None,
    ) -> str:
        """
        Semantic Kernel-exposed document storage (async).

        AI-friendly function for storing documents in vector memory.
        Returns the ID of the stored document for future reference.
        """
        ctx = self._build_sk_aware_context(sk_context, sk_settings)

        try:
            ids = await self.vector_store.aadd_texts(
                texts=[content],
                metadatas=[metadata or {}],
                ids=[document_id] if document_id else None,
                namespace=namespace,
                ctx=ctx,
            )
            return ids[0] if ids else "unknown"
        except Exception as exc:  # noqa: BLE001
            self._handle_sk_plugin_error(
                exc,
                operation="plugin_store_document",
                content_length=len(content),
                namespace=namespace,
            )

    @kernel_function(
        name="vector_get_capabilities",
        description=(
            "Get information about what the vector database supports, including "
            "maximum batch sizes, filtering capabilities, and other features."
        ),
    )
    async def get_capabilities(
        self,
        sk_context: Optional[Any] = None,
        sk_settings: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Semantic Kernel-exposed capability query (async).

        Lets AI models understand what operations are available and their limits.
        Uses public methods instead of private ones for proper encapsulation.
        """
        _ = self._build_sk_aware_context(sk_context, sk_settings)

        try:
            caps = await self.vector_store.aget_capabilities()
            return {
                "max_batch_size": caps.max_batch_size,
                "max_top_k": caps.max_top_k,
                "supports_metadata_filtering": caps.supports_metadata_filtering,
                "supports_namespaces": caps.supports_namespaces,
                "description": "Corpus VectorProtocol vector database",
            }
        except Exception as exc:  # noqa: BLE001
            self._handle_sk_plugin_error(
                exc,
                operation="plugin_get_capabilities",
            )


__all__ = [
    "CorpusSemanticKernelVectorStore",
    "CorpusSemanticKernelVectorPlugin",
]
