# corpus_sdk/vector/framework_adapters/semantic_kernel.py
# SPDX-License-Identifier: Apache-2.0

"""
Semantic Kernel adapter for Corpus Vector protocol.

This module exposes Corpus `BaseVectorAdapter` implementations to
Semantic Kernel in two layers:

1. Core Python API:
   - `CorpusSemanticKernelVectorStore`: protocol-first vector store that
     talks to a Corpus `BaseVectorAdapter` using VectorProtocolV1.
   - Sync + async add/search/delete APIs.
   - Optional streaming search via SyncStreamBridge.
   - Optional Maximal Marginal Relevance (MMR) search.
   - Optional client-side score thresholding.
   - Capability-aware handling of namespaces, metadata filters, and
     backend batch limits.
   - Integration with OperationContext via `corpus_sdk.core.context_translation`.

2. Semantic Kernel plugin:
   - `CorpusSemanticKernelVectorPlugin`: a plugin object that can be
     imported into a Semantic Kernel as a plugin via
       KernelPlugin.from_object(...)
     or `Kernel.add_plugin(...)`.
   - Provides `@kernel_function`-decorated functions that SK can expose
     to the model for tool/function-calling.

Design philosophy
-----------------
- Protocol-first: all heavy lifting lives in `BaseVectorAdapter`.
- Framework-agnostic core: the vector store does not depend on SK types.
- SK-specific layer: plugin functions that translate SK context into
  OperationContext using `from_semantic_kernel`.

Usage (simplified)
------------------

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

    def embed_texts(texts: list[str]) -> list[list[float]]:
        ...

    store = CorpusSemanticKernelVectorStore(
        corpus_adapter=adapter,
        embedding_function=embed_texts,
        namespace="docs",
        default_top_k=4,
    )

    plugin = CorpusSemanticKernelVectorPlugin(vector_store=store)

    kernel = Kernel()
    kernel.add_plugin(plugin, plugin_name="corpus_vector")

    # Now the model can call:
    #   corpus_vector.vector_search(...)
    #   corpus_vector.vector_search_stream(...)
"""

from __future__ import annotations

import logging
import math
import uuid
from typing import (
    Any,
    AsyncIterator,
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
from corpus_sdk.core.sync_bridge import sync_stream
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
from corpus_sdk.core.async_bridge import AsyncBridge
from corpus_sdk.core.error_context import attach_context

# Semantic Kernel imports are optional; if SK is not installed, we fall back
# to a no-op decorator so the rest of the SDK remains usable.
try:  # pragma: no cover - import guard
    from semantic_kernel.functions import kernel_function
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


logger = logging.getLogger(__name__)

Embeddings = Sequence[Sequence[float]]
Metadata = Dict[str, Any]


class CorpusSemanticKernelVectorStore:
    """
    Corpus vector store integration for Semantic Kernel.

    This class is framework-agnostic; it does not depend on SK types.
    It provides sync + async APIs for:

    - Adding texts/documents
    - Similarity search (with or without scores)
    - Streaming similarity search (sync, via SyncStreamBridge)
    - Maximal Marginal Relevance (MMR) search
    - Delete by ID or metadata filter

    Semantic Kernel-specific context is translated into an
    `OperationContext` via `from_semantic_kernel` in the plugin layer.
    This class accepts either:

    - `ctx: OperationContext` directly, or
    - `sk_context` / `sk_settings` / `context_dict` for on-the-fly
      translation via `from_semantic_kernel` / `from_dict`.
    """

    corpus_adapter: BaseVectorAdapter
    namespace: Optional[str] = "default"

    id_field: str = "id"
    text_field: str = "page_content"
    metadata_field: Optional[str] = None

    score_threshold: Optional[float] = None
    batch_size: int = 100
    max_query_batch_size: Optional[int] = None
    default_top_k: int = 4

    # Optional embedding integration
    embedding_function: Optional[Any] = None  # Callable[[List[str]], Embeddings]

    # Cached capabilities
    _caps: Optional[VectorCapabilities] = None

    # Pydantic-style config (kept for symmetry with other adapters)
    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        *,
        corpus_adapter: BaseVectorAdapter,
        namespace: Optional[str] = "default",
        id_field: str = "id",
        text_field: str = "page_content",
        metadata_field: Optional[str] = None,
        score_threshold: Optional[float] = None,
        batch_size: int = 100,
        max_query_batch_size: Optional[int] = None,
        default_top_k: int = 4,
        embedding_function: Optional[Any] = None,
    ) -> None:
        self.corpus_adapter = corpus_adapter
        self.namespace = namespace
        self.id_field = id_field
        self.text_field = text_field
        self.metadata_field = metadata_field
        self.score_threshold = score_threshold
        self.batch_size = int(batch_size)
        self.max_query_batch_size = max_query_batch_size
        self.default_top_k = int(default_top_k)
        self.embedding_function = embedding_function
        self._caps = None

    # ------------------------------------------------------------------ #
    # Capability helpers
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
            attach_context(exc, framework="semantic_kernel", operation="capabilities")
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
            attach_context(exc, framework="semantic_kernel", operation="capabilities")
            raise

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
                raise BadRequest(
                    f"embeddings length {len(embeddings)} does not match texts length {len(texts)}",
                    code="BAD_EMBEDDINGS",
                    details={"texts": len(texts), "embeddings": len(embeddings)},
                )
            return embeddings

        if self.embedding_function is None:
            raise NotSupported(
                "No embedding_function configured; caller must supply embeddings",
                code="NO_EMBEDDING_FUNCTION",
                details={"texts": len(texts)},
            )

        try:
            computed = self.embedding_function(texts)
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="semantic_kernel",
                operation="embedding_function",
            )
            raise BadRequest(
                f"embedding_function failed: {exc}",
                code="EMBEDDING_ERROR",
            )
        if len(computed) != len(texts):
            raise BadRequest(
                f"embedding_function returned {len(computed)} embeddings for {len(texts)} texts",
                code="BAD_EMBEDDINGS",
            )
        return computed

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
            code="BAD_METADATA",
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
        - If ids is None: generate UUID4 hex IDs.
        - If len(ids) == n: coerce to str list.
        - Else: raise BadRequest.
        """
        if ids is None:
            return [uuid.uuid4().hex for _ in range(n)]
        if len(ids) != n:
            raise BadRequest(
                f"ids length {len(ids)} does not match texts length {n}",
                code="BAD_IDS",
                details={"texts": n, "ids": len(ids)},
            )
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
                    framework="semantic_kernel",
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
                framework="semantic_kernel",
                operation="query",
                namespace=ns,
                top_k=k,
            )
            raise

        matches = list(result.matches or [])

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
                framework="semantic_kernel",
                operation="delete",
                namespace=ns,
                ids_count=len(ids or []),
            )
            raise

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

    # ------------------------------------------------------------------ #
    # Public sync/async API
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

        try:
            AsyncBridge.run_async(
                self._aupsert_vectors(
                    vectors,
                    namespace=namespace,
                    ctx=ctx,
                )
            )
        except Exception:
            # Errors already have context attached.
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
        emb = self._ensure_embeddings(texts_list, embeddings)

        vectors = self._to_corpus_vectors(
            texts=texts_list,
            embeddings=emb,
            metadatas=metadatas_norm,
            ids=ids_norm,
            namespace=namespace,
        )

        await self._aupsert_vectors(
            vectors,
            namespace=namespace,
            ctx=ctx,
        )
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
        Perform similarity search and return document dicts (sync).

        Return shape:
            [{"page_content": str, "metadata": dict, "score": float}, ...]
        """
        embedding: Optional[Sequence[float]] = kwargs.get("embedding")
        namespace: Optional[str] = kwargs.get("namespace")
        ctx = self._build_ctx(**kwargs)

        query_emb = self._embed_query(query, embedding=embedding)
        try:
            matches = AsyncBridge.run_async(
                self._aquery_embedding(
                    query_emb,
                    k=k or self.default_top_k,
                    namespace=namespace,
                    filter=filter,
                    include_vectors=False,
                    ctx=ctx,
                )
            )
        except Exception:
            raise

        docs = []
        for text, meta, score in self._from_corpus_matches(matches):
            docs.append(
                {
                    "page_content": text,
                    "metadata": meta,
                    "score": score,
                }
            )
        return docs

    def similarity_search_stream(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Iterator[Dict[str, Any]]:
        """
        Streaming similarity search (sync), yielding document dicts one by one.

        This uses SyncStreamBridge under the hood via `sync_stream` to bridge
        the async query into a synchronous iterator.
        """
        embedding: Optional[Sequence[float]] = kwargs.get("embedding")
        namespace: Optional[str] = kwargs.get("namespace")
        ctx = self._build_ctx(**kwargs)
        top_k = k or self.default_top_k

        query_emb = self._embed_query(query, embedding=embedding)

        async def _stream_coro() -> AsyncIterator[VectorMatch]:
            matches = await self._aquery_embedding(
                query_emb,
                k=top_k,
                namespace=namespace,
                filter=filter,
                include_vectors=False,
                ctx=ctx,
            )
            for m in matches:
                yield m

        for match in sync_stream(
            _stream_coro,
            framework="semantic_kernel",
            error_context={
                "operation": "similarity_search_stream",
                "namespace": namespace,
                "top_k": top_k,
            },
        ):
            text, meta, score = self._from_corpus_matches([match])[0]
            yield {
                "page_content": text,
                "metadata": meta,
                "score": score,
            }

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search and return document dicts (async).
        """
        embedding: Optional[Sequence[float]] = kwargs.get("embedding")
        namespace: Optional[str] = kwargs.get("namespace")
        ctx = self._build_ctx(**kwargs)

        query_emb = self._embed_query(query, embedding=embedding)
        matches = await self._aquery_embedding(
            query_emb,
            k=k or self.default_top_k,
            namespace=namespace,
            filter=filter,
            include_vectors=False,
            ctx=ctx,
        )
        docs = []
        for text, meta, score in self._from_corpus_matches(matches):
            docs.append(
                {
                    "page_content": text,
                    "metadata": meta,
                    "score": score,
                }
            )
        return docs

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Similarity search returning (doc_dict, score) tuples (sync).
        """
        docs = self.similarity_search(query, k=k, filter=filter, **kwargs)
        return [(doc, float(doc["score"])) for doc in docs]

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Similarity search returning (doc_dict, score) tuples (async).
        """
        docs = await self.asimilarity_search(query, k=k, filter=filter, **kwargs)
        return [(doc, float(doc["score"])) for doc in docs]

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
        original database scores.
        """
        fetch_k: int = kwargs.get("fetch_k") or max(k * 4, k + 5)
        embedding: Optional[Sequence[float]] = kwargs.get("embedding")
        namespace: Optional[str] = kwargs.get("namespace")
        ctx = self._build_ctx(**kwargs)

        query_emb = self._embed_query(query, embedding=embedding)

        matches = AsyncBridge.run_async(
            self._aquery_embedding(
                query_emb,
                k=fetch_k,
                namespace=namespace,
                filter=filter,
                include_vectors=True,
                ctx=ctx,
            )
        )

        if not matches:
            return []

        indices = self._mmr_select_indices(
            query_vec=query_emb,
            candidate_matches=matches,
            k=k,
            lambda_mult=lambda_mult,
        )

        docs: List[Dict[str, Any]] = []
        flattened = self._from_corpus_matches(matches)
        for idx in indices:
            text, meta, score = flattened[idx]
            docs.append(
                {
                    "page_content": text,
                    "metadata": meta,
                    "score": score,
                }
            )
        return docs

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
        """
        fetch_k: int = kwargs.get("fetch_k") or max(k * 4, k + 5)
        embedding: Optional[Sequence[float]] = kwargs.get("embedding")
        namespace: Optional[str] = kwargs.get("namespace")
        ctx = self._build_ctx(**kwargs)

        query_emb = self._embed_query(query, embedding=embedding)

        matches = await self._aquery_embedding(
            query_emb,
            k=fetch_k,
            namespace=namespace,
            filter=filter,
            include_vectors=True,
            ctx=ctx,
        )

        if not matches:
            return []

        indices = self._mmr_select_indices(
            query_vec=query_emb,
            candidate_matches=matches,
            k=k,
            lambda_mult=lambda_mult,
        )

        docs: List[Dict[str, Any]] = []
        flattened = self._from_corpus_matches(matches)
        for idx in indices:
            text, meta, score = flattened[idx]
            docs.append(
                {
                    "page_content": text,
                    "metadata": meta,
                    "score": score,
                }
            )
        return docs

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
        AsyncBridge.run_async(
            self._adelete_vectors(
                ids=ids,
                namespace=namespace,
                filter=filter,
                ctx=ctx,
            )
        )

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
        await self._adelete_vectors(
            ids=ids,
            namespace=namespace,
            filter=filter,
            ctx=ctx,
        )

    # ------------------------------------------------------------------ #
    # Convenience constructors
    # ------------------------------------------------------------------ #

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        *,
        corpus_adapter: BaseVectorAdapter,
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
        corpus_adapter: BaseVectorAdapter,
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
    the vector store, which talks to the underlying `BaseVectorAdapter`.
    """

    def __init__(
        self,
        *,
        vector_store: CorpusSemanticKernelVectorStore,
        framework_version: Optional[str] = None,
    ) -> None:
        self.vector_store = vector_store
        self.framework_version = framework_version

    def _build_ctx_from_sk(
        self,
        sk_context: Optional[Any],
        sk_settings: Optional[Any],
    ) -> OperationContext:
        """
        Translate SK context/settings into OperationContext.

        Any translation errors are logged and result in an empty context
        rather than failing the call outright.
        """
        try:
            return context_from_semantic_kernel(
                sk_context,
                settings=sk_settings,
                framework_version=self.framework_version,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("context_from_semantic_kernel failed in plugin: %s", exc)
            # Fall back to empty context
            return context_from_dict({})

    @kernel_function(
        name="vector_search",
        description=(
            "Retrieve the most relevant documents from the Corpus vector index "
            "using semantic similarity. Returns a list of objects with "
            "page_content, metadata, and score."
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
        """
        ctx = self._build_ctx_from_sk(sk_context, sk_settings)

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
            attach_context(
                exc,
                framework="semantic_kernel",
                operation="plugin_vector_search",
                query=query,
            )
            raise

    @kernel_function(
        name="vector_search_stream",
        description=(
            "Streaming variant of vector_search. Yields documents one by one "
            "as they are retrieved. Each item has page_content, metadata, and score."
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
        returns an iterator. Under the hood, this uses SyncStreamBridge via
        `vector_store.similarity_search_stream`.
        """
        ctx = self._build_ctx_from_sk(sk_context, sk_settings)

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
            attach_context(
                exc,
                framework="semantic_kernel",
                operation="plugin_vector_search_stream",
                query=query,
            )
            raise

    @kernel_function(
        name="vector_mmr_search",
        description=(
            "Maximal Marginal Relevance (MMR) search over the Corpus vector index. "
            "Balances relevance and diversity; returns a list of documents."
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
        """
        ctx = self._build_ctx_from_sk(sk_context, sk_settings)

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
            attach_context(
                exc,
                framework="semantic_kernel",
                operation="plugin_vector_mmr_search",
                query=query,
            )
            raise


__all__ = [
    "CorpusSemanticKernelVectorStore",
    "CorpusSemanticKernelVectorPlugin",
]
