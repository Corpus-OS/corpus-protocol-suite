# corpus_sdk/vector/framework_adapters/langchain.py
# SPDX-License-Identifier: Apache-2.0

"""
LangChain adapter for Corpus Vector protocol.

This module exposes Corpus `BaseVectorAdapter` implementations as
`langchain_core` vector stores and retrievers, with:

- Sync + async add/search APIs (mirroring LangChain's VectorStore)
- Proper integration with Corpus VectorProtocolV1
- Namespace + metadata filter handling (capability-aware)
- Batch upserts and deletes that respect backend limits
- Optional client-side score thresholding
- Optional embedding function integration
- Optional max marginal relevance (MMR) search

Design philosophy
-----------------
- Protocol-first: LangChain is a thin skin over Corpus vector adapters.
- All heavy lifting (backpressure, deadlines, breakers, etc.) lives in
  the underlying `BaseVectorAdapter`, not here.
- This layer focuses on:
    * Translating LangChain Documents ↔ Corpus Vector objects
    * Respecting VectorCapabilities (namespaces, filters, batch sizes)
    * Bridging async Corpus APIs into LangChain's sync interface via AsyncBridge

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

import logging
import math
import uuid
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

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
from corpus_sdk.llm.framework_adapters.common.context_translation import (
    from_langchain as context_from_langchain,
)
from corpus_sdk.llm.framework_adapters.common.error_context import attach_context
from corpus_sdk.core.sync_stream_bridge import sync_stream

logger = logging.getLogger(__name__)

Embeddings = Sequence[Sequence[float]]
Metadata = Dict[str, Any]


class CorpusLangChainVectorStore(VectorStore):
    """
    LangChain `VectorStore` implementation backed by a Corpus `BaseVectorAdapter`.

    This class is a thin integration layer:
    - Documents are mapped to Corpus VectorProtocol `Vector` objects.
    - Similarity search calls map to adapter `query()` calls.
    - Namespaces + metadata filters are honored based on VectorCapabilities.
    - Sync APIs are implemented via `AsyncBridge` on top of async Corpus methods.

    Attributes
    ----------
    corpus_adapter:
        Underlying Corpus vector adapter implementing VectorProtocolV1.

    namespace:
        Default namespace for all operations, when not explicitly overridden.
        If the underlying adapter does not support namespaces, this is ignored.

    id_field:
        Metadata key under which the logical document ID will be stored.

    text_field:
        Metadata key under which the document text/page_content is stored.

    metadata_field:
        Optional "envelope" key under which LangChain metadata dict is stored.
        If None, metadata is stored directly on Vector.metadata.

    score_threshold:
        Optional minimum similarity score required for a match to be returned.
        This is applied client-side on the `VectorMatch.score` field.

    batch_size:
        Planning hint for upsert/delete batches. The effective batch size is
        `min(batch_size, capabilities.max_batch_size)` when the backend reports
        a max batch size; otherwise `batch_size`.

    max_query_batch_size:
        Placeholder for future multi-query APIs. Currently unused.

    default_top_k:
        Default K used in `similarity_search` when the caller does not specify.

    embedding_function:
        Optional function used to embed raw texts into vectors. If provided,
        `add_texts` and `similarity_search` can accept raw strings. If not
        provided, callers must supply embeddings explicitly via kwargs.

        Signature:
            embedding_function(texts: List[str]) -> List[List[float]]

    model_config:
        Pydantic v2-style config: allow arbitrary types like BaseVectorAdapter.
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

    # Pydantic v2-style config
    model_config = {"arbitrary_types_allowed": True}

    # ------------------------------------------------------------------ #
    # VectorStore-required properties / metadata
    # ------------------------------------------------------------------ #

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

        Uses AsyncBridge to call the async adapter.
        """
        if self._caps is not None:
            return self._caps
        try:
            caps = AsyncBridge.run_async(self.corpus_adapter.capabilities())
            self._caps = caps
            return caps
        except Exception as exc:  # noqa: BLE001
            attach_context(exc, framework="langchain", operation="capabilities")
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
            attach_context(exc, framework="langchain", operation="capabilities")
            raise

    def _build_ctx(self, **kwargs: Any) -> Optional[OperationContext]:
        """
        Build an OperationContext from a LangChain config-like dict in kwargs.

        Expected kwargs:
            - config: RunnableConfig or similar (optional)
        """
        config = kwargs.get("config")
        if config is None:
            return None
        try:
            return context_from_langchain(config)
        except Exception as exc:  # noqa: BLE001
            logger.debug("context_from_langchain failed: %s", exc)
            return None

    def _effective_namespace(self, namespace: Optional[str]) -> Optional[str]:
        """
        Resolve namespace using explicit override or store default.

        If the underlying adapter does not support namespaces, this value is
        still passed down, but the adapter may ignore it.
        """
        return namespace if namespace is not None else self.namespace

    # ------------------------------------------------------------------ #
    # Translation helpers: LC Documents ↔ Corpus Vector
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

            vectors.append(
                Vector(
                    id=str(vid),
                    vector=[float(x) for x in emb],
                    metadata=metadata_payload,
                    namespace=ns,
                    text=None,  # text is stored in metadata; docstore is handled upstream
                )
            )

        return vectors

    def _from_corpus_matches(
        self,
        matches: Sequence[VectorMatch],
    ) -> List[Tuple[Document, float]]:
        """
        Convert Corpus `VectorMatch` objects into LangChain Documents + scores.

        The similarity score returned is `VectorMatch.score`, which is assumed
        to be higher-is-better (normalized similarity as defined by the adapter).
        """
        results: List[Tuple[Document, float]] = []
        for m in matches:
            v = m.vector
            meta = dict(v.metadata or {})

            # Extract envelope metadata if applicable
            if self.metadata_field and self.metadata_field in meta:
                nested = meta.get(self.metadata_field) or {}
                if isinstance(nested, Mapping):
                    # Merge nested metadata into the top-level dict, but keep
                    # reserved keys like text/id from the top-level.
                    nested_meta = dict(nested)
                else:
                    nested_meta = {}
            else:
                nested_meta = meta

            # Extract text from metadata
            text = meta.get(self.text_field)
            if text is None:
                # Fallback: if text is not found, use empty string.
                text = ""

            # Remove internal keys to avoid leaking implementation details
            nested_meta.pop(self.text_field, None)
            nested_meta.pop(self.id_field, None)

            doc = Document(page_content=text, metadata=nested_meta)
            results.append((doc, float(m.score)))
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
                    framework="langchain",
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
                framework="langchain",
                operation="query",
                namespace=ns,
                top_k=k,
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
                framework="langchain",
                operation="delete",
                namespace=ns,
                ids_count=len(ids or []),
            )
            raise

    # ------------------------------------------------------------------ #
    # LangChain VectorStore sync API
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
        - Respects backend batch size limits when performing upserts.
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
            # Errors already have context attached in _aupsert_vectors.
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
        documents: List[Document],
        **kwargs: Any,
    ) -> List[str]:
        """
        Add LangChain Documents to the vector store (sync).
        """
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
            raise NotSupported(
                "No embedding_function configured; caller must supply query embedding",
                code="NO_EMBEDDING_FUNCTION",
            )

        try:
            embs = self.embedding_function([query])
        except Exception as exc:  # noqa: BLE001
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

        docs_scores = self._from_corpus_matches(matches)
        return [doc for doc, _ in docs_scores]

    def similarity_search_stream(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Iterator[Document]:
        """
        Streaming similarity search (sync), yielding Documents one by one.

        This uses SyncStreamBridge under the hood via `sync_stream` to bridge
        the async query into a synchronous iterator. The backend query itself
        is still a single async call; this just exposes results incrementally.
        """
        embedding: Optional[Sequence[float]] = kwargs.get("embedding")
        namespace: Optional[str] = kwargs.get("namespace")
        ctx = self._build_ctx(**kwargs)

        query_emb = self._embed_query(query, embedding=embedding)
        top_k = k or self.default_top_k

        async def _stream_coro():
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
            framework="langchain",
            error_context={
                "operation": "similarity_search_stream",
                "namespace": namespace,
                "top_k": top_k,
            },
        ):
            docs_scores = self._from_corpus_matches([match])
            if docs_scores:
                yield docs_scores[0][0]

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
        docs_scores = self._from_corpus_matches(matches)
        return [doc for doc, _ in docs_scores]

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
        embedding: Optional[Sequence[float]] = kwargs.get("embedding")
        namespace: Optional[str] = kwargs.get("namespace")
        ctx = self._build_ctx(**kwargs)

        query_emb = self._embed_query(query, embedding=embedding)
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
        return self._from_corpus_matches(matches)

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
        return self._from_corpus_matches(matches)

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
            first_idx = max(candidates, key=lambda i: normalized_scores[i])
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

        This runs a similarity search with a larger `fetch_k` and then
        selects a subset of results via MMR based on vector geometry and
        original database scores.

        Args:
            query: Query string
            k: Number of results to return
            lambda_mult: MMR lambda parameter (0-1), higher values favor relevance
            filter: Optional metadata filter
            **kwargs: Additional arguments including:
                - fetch_k: Number of candidates to fetch for MMR selection
                - embedding: Optional precomputed query embedding
                - namespace: Optional namespace override

        Returns:
            List of Documents selected via MMR
        """
        fetch_k: int = kwargs.get("fetch_k") or max(k * 4, k + 5)
        embedding: Optional[Sequence[float]] = kwargs.get("embedding")
        namespace: Optional[str] = kwargs.get("namespace")
        ctx = self._build_ctx(**kwargs)

        query_emb = self._embed_query(query, embedding=embedding)

        # We need vectors back to compute MMR.
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

        # Use improved MMR that respects original scores
        indices = self._mmr_select_indices(
            query_vec=query_emb,
            candidate_matches=matches,
            k=k,
            lambda_mult=lambda_mult,
        )

        docs_scores = self._from_corpus_matches(matches)
        return [docs_scores[i][0] for i in indices]

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

        docs_scores = self._from_corpus_matches(matches)
        return [docs_scores[i][0] for i in indices]

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
    ) -> "CorpusLangChainVectorStore":
        """
        Create a store from texts, then add them immediately (sync).
        """
        store = cls(corpus_adapter=corpus_adapter, **kwargs)
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        *,
        corpus_adapter: BaseVectorAdapter,
        **kwargs: Any,
    ) -> "CorpusLangChainVectorStore":
        """
        Create a store from Documents, then add them immediately (sync).
        """
        store = cls(corpus_adapter=corpus_adapter, **kwargs)
        store.add_documents(documents)
        return store


class CorpusLangChainRetriever(BaseRetriever):
    """
    LangChain `BaseRetriever` implementation backed by a `CorpusLangChainVectorStore`.

    This is a light wrapper around `similarity_search`, useful when you want
    the VectorStore to be the primary retrieval mechanism in a LangChain graph.

    Attributes
    ----------
    vector_store:
        Underlying `CorpusLangChainVectorStore` instance.

    search_kwargs:
        Keyword arguments forwarded to `vector_store.similarity_search`, e.g.:
        {"k": 4, "filter": {...}, "namespace": "docs"}.
    """

    vector_store: CorpusLangChainVectorStore
    search_kwargs: Dict[str, Any] = {}
    model_config = {"arbitrary_types_allowed": True}

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
                # LangChain does not mandate callbacks here, but we can optionally
                # add `on_retriever_end` analogues if desired. For now, we no-op.
                pass
            return docs
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="langchain",
                operation="retriever_sync",
                query=query,
            )
            if run_manager is not None:
                # No explicit retriever-level error callback in core API; we simply re-raise.
                pass
            raise

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
                # Same note as sync version.
                pass
            return docs
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="langchain",
                operation="retriever_async",
                query=query,
            )
            if run_manager is not None:
                pass
            raise


__all__ = [
    "CorpusLangChainVectorStore",
    "CorpusLangChainRetriever",
]
