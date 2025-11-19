# corpus_sdk/vector/framework_adapters/autogen.py
# SPDX-License-Identifier: Apache-2.0

"""
AutoGen adapter for Corpus Vector protocol.

This module exposes Corpus `BaseVectorAdapter` implementations as
AutoGen-friendly vector stores and retrievers, with:

- Sync + async add/search APIs
- Streaming similarity search via SyncStreamBridge
- Proper integration with Corpus VectorProtocolV1
- Namespace + metadata filter handling (capability-aware)
- Batch upserts and deletes that respect backend limits
- Optional client-side score thresholding
- Optional embedding function integration
- Optional max marginal relevance (MMR) search
- Simple retriever/tool wrapper for AutoGen agents

Design philosophy
-----------------
- Protocol-first: AutoGen is a thin skin over Corpus vector adapters.
- All heavy lifting (backpressure, deadlines, breakers, etc.) lives in
  the underlying `BaseVectorAdapter`, not here.
- This layer focuses on:
    * Translating simple Python data structures ↔ Corpus Vector objects
    * Respecting VectorCapabilities (namespaces, filters, batch sizes)
    * Bridging async Corpus APIs into sync AutoGen tools via AsyncBridge
      and SyncStreamBridge
    * Propagating AutoGen conversation context into OperationContext

Usage
-----

    from corpus_sdk.vector.pinecone_adapter import PineconeVectorAdapter
    from corpus_sdk.vector.framework_adapters.autogen import (
        CorpusAutoGenVectorStore,
        CorpusAutoGenRetrieverTool,
    )

    adapter = PineconeVectorAdapter(
        index_name="my-index",
        api_key="...",
        dimensions=1536,
    )

    # Provide an embedding function that maps List[str] -> List[List[float]]
    def embed_texts(texts: list[str]) -> list[list[float]]:
        ...

    store = CorpusAutoGenVectorStore(
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

    # Similarity search (sync)
    docs = store.similarity_search("hello", k=3)

    # Streaming similarity search (sync)
    for doc in store.similarity_search_stream("hello", k=3):
        handle(doc)

    # Register as an AutoGen tool
    retriever_tool = CorpusAutoGenRetrieverTool(
        vector_store=store,
        name="corpus_vector_search",
        description="Retrieve relevant documents from Corpus vector index.",
        search_kwargs={"k": 4},
    )

    # In AutoGen, you can register `retriever_tool` as a callable tool.
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
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
from corpus_sdk.llm.framework_adapters.common.error_context import attach_context
from corpus_sdk.core.context_translation import (
    from_autogen as context_from_autogen,
)
from corpus_sdk.core.sync_stream_bridge import SyncStreamBridge

logger = logging.getLogger(__name__)

Embeddings = Sequence[Sequence[float]]
Metadata = Dict[str, Any]


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


class CorpusAutoGenVectorStore:
    """
    AutoGen-friendly vector store backed by a Corpus `BaseVectorAdapter`.

    This class is a thin integration layer:
    - Raw texts + embeddings are mapped to Corpus VectorProtocol `Vector` objects.
    - Similarity search calls map directly to adapter `query()` calls.
    - Namespaces + metadata filters are honored based on VectorCapabilities.
    - Sync APIs are implemented via `AsyncBridge` on top of async Corpus methods.
    - Streaming APIs are implemented via `SyncStreamBridge`.

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
        Optional "envelope" key under which metadata dict is stored.
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
            attach_context(exc, framework="autogen", operation="capabilities")
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
            attach_context(exc, framework="autogen", operation="capabilities")
            raise

    def _build_ctx(
        self,
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Optional[OperationContext]:
        """
        Build an OperationContext from an AutoGen conversation + extra metadata.

        Args
        ----
        conversation:
            AutoGen conversation object (or None).

        extra_context:
            Optional dict of extra context fields that should flow into
            OperationContext via `from_autogen`.

        Returns
        -------
        OperationContext | None
        """
        if conversation is None and not extra_context:
            return None
        try:
            extra = dict(extra_context or {})
            return context_from_autogen(conversation, **extra)
        except Exception as exc:  # noqa: BLE001
            logger.debug("context_from_autogen failed: %s", exc)
            return None

    def _effective_namespace(self, namespace: Optional[str]) -> Optional[str]:
        """
        Resolve namespace using explicit override or store default.

        If the underlying adapter does not support namespaces, this value is
        still passed down, but the adapter may ignore it.
        """
        return namespace if namespace is not None else self.namespace

    # ------------------------------------------------------------------ #
    # Translation helpers: AutoGenDocument ↔ Corpus Vector
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
        Convert texts + embeddings + metadatas into Corpus `Vector` objects.
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
                    text=None,  # text is stored in metadata; docstore handled upstream
                )
            )

        return vectors

    def _from_corpus_matches(
        self,
        matches: Sequence[VectorMatch],
    ) -> List[Tuple[AutoGenDocument, float]]:
        """
        Convert Corpus `VectorMatch` objects into AutoGenDocuments + scores.

        The similarity score returned is `VectorMatch.score`, which is assumed
        to be higher-is-better (normalized similarity as defined by the adapter).
        """
        results: List[Tuple[AutoGenDocument, float]] = []
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

            # Remove internal keys to avoid leaking implementation details
            nested_meta.pop(self.text_field, None)
            nested_meta.pop(self.id_field, None)

            doc = AutoGenDocument(page_content=text, metadata=nested_meta)
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
                    framework="autogen",
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
                framework="autogen",
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
                framework="autogen",
                operation="delete",
                namespace=ns,
                ids_count=len(ids or []),
            )
            raise

    # ------------------------------------------------------------------ #
    # Public add APIs
    # ------------------------------------------------------------------ #

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
        - Respects backend batch size limits when performing upserts.
        """
        texts_list = list(texts)
        if not texts_list:
            return []

        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)

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
                framework="autogen",
                operation="embed_query",
                query=query,
            )
            raise exc

        try:
            embs = self.embedding_function([query])
        except Exception as exc:  # noqa: BLE001
            err = BadRequest(
                f"embedding_function failed for query: {exc}",
                code="EMBEDDING_ERROR",
            )
            attach_context(
                err,
                framework="autogen",
                operation="embed_query",
                query=query,
            )
            raise err
        if not embs or len(embs) != 1:
            err = BadRequest(
                "embedding_function must return exactly one embedding for a single query",
                code="BAD_EMBEDDINGS",
            )
            attach_context(
                err,
                framework="autogen",
                operation="embed_query",
                query=query,
            )
            raise err
        return [float(x) for x in embs[0]]

    # ------------------------------------------------------------------ #
    # Query APIs (sync + async + streaming + raw)
    # ------------------------------------------------------------------ #

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
        queue_maxsize: int = 0,
    ) -> Iterator[AutoGenDocument]:
        """
        Streaming similarity search (sync), yielding AutoGenDocuments one by one.

        This uses SyncStreamBridge to bridge the async query into a synchronous
        iterator. The backend query itself is still a single async call; this
        just exposes results incrementally to the caller.
        """
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
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

            async def _inner() -> AsyncIterator[VectorMatch]:
                for m in matches:
                    yield m

            return _inner()

        bridge = SyncStreamBridge(
            coro_factory=_stream_coro,
            queue_maxsize=queue_maxsize,
            framework="autogen",
            error_context={
                "operation": "similarity_search_stream",
                "namespace": namespace,
                "top_k": top_k,
            },
        )

        for match in bridge.run():
            docs_scores = self._from_corpus_matches([match])
            if docs_scores:
                yield docs_scores[0][0]

    # Low-level raw query API for advanced users

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
    ) -> List[VectorMatch]:
        """
        Low-level similarity query using a precomputed embedding (sync).

        Returns raw `VectorMatch` objects instead of AutoGenDocuments.
        """
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        try:
            matches = AsyncBridge.run_async(
                self._aquery_embedding(
                    embedding,
                    k=k or self.default_top_k,
                    namespace=namespace,
                    filter=filter,
                    include_vectors=include_vectors,
                    ctx=ctx,
                )
            )
            return matches
        except Exception:
            raise

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
    ) -> List[VectorMatch]:
        """
        Low-level similarity query using a precomputed embedding (async).

        Returns raw `VectorMatch` objects instead of AutoGenDocuments.
        """
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        return await self._aquery_embedding(
            embedding,
            k=k or self.default_top_k,
            namespace=namespace,
            filter=filter,
            include_vectors=include_vectors,
            ctx=ctx,
        )

    # ------------------------------------------------------------------ #
    # MMR search (same strategy as LangChain/LlamaIndex adapters)
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
        candidate_matches: List[VectorMatch],
        k: int,
        lambda_mult: float,
    ) -> List[int]:
        """
        MMR selector that respects original database scores and caches similarities.

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
        *,
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
        embedding: Optional[Sequence[float]] = None,
        namespace: Optional[str] = None,
        fetch_k: Optional[int] = None,
    ) -> List[AutoGenDocument]:
        """
        Perform Maximal Marginal Relevance (MMR) search (sync).

        This runs a similarity search with a larger `fetch_k` and then
        selects a subset of results via MMR based on vector geometry and
        original database scores.
        """
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
        query_emb = self._embed_query(query, embedding=embedding)

        actual_fetch_k = fetch_k or max(k * 4, k + 5)

        matches = AsyncBridge.run_async(
            self._aquery_embedding(
                query_emb,
                k=actual_fetch_k,
                namespace=namespace,
                filter=filter,
                include_vectors=True,
                ctx=ctx,
            )
        )

        if not matches:
            return []

        indices = self._mmr_select_indices(
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
        query_emb = self._embed_query(query, embedding=embedding)

        actual_fetch_k = fetch_k or max(k * 4, k + 5)

        matches = await self._aquery_embedding(
            query_emb,
            k=actual_fetch_k,
            namespace=namespace,
            filter=filter,
            include_vectors=True,
            ctx=ctx,
        )

        if not matches:
            return []

        indices = self._mmr_select_indices(
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
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
        namespace: Optional[str] = None,
    ) -> None:
        """
        Delete vectors by IDs or metadata filter (sync).
        """
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
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
        conversation: Optional[Any] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
        namespace: Optional[str] = None,
    ) -> None:
        """
        Delete vectors by IDs or metadata filter (async).
        """
        ctx = self._build_ctx(conversation=conversation, extra_context=extra_context)
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
    ) -> "CorpusAutoGenVectorStore":
        """
        Create a store from texts, then add them immediately (sync).
        """
        store = cls(corpus_adapter=corpus_adapter, **kwargs)
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store

    @classmethod
    def from_documents(
        cls,
        documents: List[AutoGenDocument],
        *,
        corpus_adapter: BaseVectorAdapter,
        **kwargs: Any,
    ) -> "CorpusAutoGenVectorStore":
        """
        Create a store from AutoGenDocuments, then add them immediately (sync).
        """
        store = cls(corpus_adapter=corpus_adapter, **kwargs)
        store.add_documents(documents)
        return store


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
]
