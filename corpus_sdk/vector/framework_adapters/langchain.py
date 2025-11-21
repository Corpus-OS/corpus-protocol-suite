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
- Optional embedding function integration
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

import logging
import math
import uuid
from functools import cached_property
from typing import (
    Any,
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

from corpus_sdk.llm.framework_adapters.common.context_translation import (
    from_langchain as context_from_langchain,
)
from corpus_sdk.llm.framework_adapters.common.error_context import attach_context

logger = logging.getLogger(__name__)

Embeddings = Sequence[Sequence[float]]
Metadata = Dict[str, Any]


class CorpusLangChainVectorStore(VectorStore):
    """
    LangChain `VectorStore` implementation backed by a Corpus `BaseVectorAdapter`.

    This class is a thin integration layer:
    - Documents are mapped to Corpus VectorProtocol `Vector` objects.
    - Similarity search calls map to translator-level `query()` calls.
    - Namespaces + metadata filters are honored based on VectorCapabilities
      (as enforced by the shared VectorTranslator).
    - All sync/async orchestration is delegated to `VectorTranslator`, so this
      adapter does not use any AsyncBridge or sync-stream utilities directly.

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
        enforced inside the shared `VectorTranslator` via VectorCapabilities.

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

    mmr_similarity_fn:
        Optional custom similarity function for the MMR diversity term.
        If not provided, cosine similarity is used.

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
    embedding_function: Optional[Callable[[List[str]], Embeddings]] = None

    # Optional custom similarity function for MMR diversity term
    mmr_similarity_fn: Optional[
        Callable[[Sequence[float], Sequence[float]], float]
    ] = None

    # Cached capabilities
    _caps: Optional[VectorCapabilities] = None

    # Pydantic v2-style config
    model_config = {"arbitrary_types_allowed": True}

    # ------------------------------------------------------------------ #
    # Translator + VectorStore-required properties / metadata
    # ------------------------------------------------------------------ #

    @cached_property
    def _translator(self) -> VectorTranslator:
        """
        Lazily construct and cache the `VectorTranslator`.

        All sync/async bridging, batching, and capability-aware orchestration
        is delegated to this shared translator. This adapter only:
        - Builds raw request shapes
        - Translates between LangChain and Corpus types
        - Applies framework-specific post-processing (e.g., MMR).
        """
        framework_translator = DefaultVectorFrameworkTranslator()
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

        Uses the VectorTranslator's sync capabilities API. All underlying
        async behavior and bridging is owned by the translator, not here.
        """
        if self._caps is not None:
            return self._caps
        try:
            caps = self._translator.capabilities()
            self._caps = caps
            return caps
        except Exception as exc:  # noqa: BLE001
            attach_context(exc, framework="langchain", operation="capabilities")
            raise

    async def _get_caps_async(self) -> VectorCapabilities:
        """
        Async capability fetch with caching.

        Uses the VectorTranslator's async capabilities API.
        """
        if self._caps is not None:
            return self._caps
        try:
            caps = await self._translator.arun_capabilities()
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

    def _framework_ctx_for_namespace(
        self,
        namespace: Optional[str],
    ) -> Mapping[str, Any]:
        """
        Build a minimal framework_ctx mapping that lets the translator know
        which logical namespace is being targeted.
        """
        ns = self._effective_namespace(namespace)
        return {"namespace": ns} if ns is not None else {}

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
                err = BadRequest(
                    f"embeddings length {len(embeddings)} does not match texts length {len(texts)}",
                    code="BAD_EMBEDDINGS",
                    details={"texts": len(texts), "embeddings": len(embeddings)},
                )
                attach_context(
                    err,
                    framework="langchain",
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
                framework="langchain",
                operation="ensure_embeddings",
                texts_count=len(texts),
            )
            raise err

        try:
            computed = self.embedding_function(texts)
        except Exception as exc:  # noqa: BLE001
            err = BadRequest(
                f"embedding_function failed: {exc}",
                code="EMBEDDING_ERROR",
                details={"texts": len(texts)},
            )
            attach_context(
                err,
                framework="langchain",
                operation="ensure_embeddings",
                texts_count=len(texts),
            )
            raise err
        if len(computed) != len(texts):
            err = BadRequest(
                f"embedding_function returned {len(computed)} embeddings for {len(texts)} texts",
                code="BAD_EMBEDDINGS",
                details={"texts": len(texts), "embeddings": len(computed)},
            )
            attach_context(
                err,
                framework="langchain",
                operation="ensure_embeddings",
                texts_count=len(texts),
                embeddings_count=len(computed),
            )
            raise err
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
                framework="langchain",
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

        The translator is responsible for converting this into a QuerySpec and
        executing the backend query with proper capability checks.
        """
        ns = self._effective_namespace(namespace)
        raw_query: Dict[str, Any] = {
            "vector": [float(x) for x in embedding],
            "top_k": int(k),
            "namespace": ns,
            "filter": dict(filter) if filter else None,
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

        The translator handles batching and capability-aware limits.
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

        Local validation ensures the caller provides at least ids or filter;
        capability-specific checks are handled by the translator.
        """
        ns = self._effective_namespace(namespace)

        if not ids and not filter:
            err = BadRequest(
                "must provide ids or filter for delete",
                code="BAD_DELETE",
            )
            attach_context(
                err,
                framework="langchain",
                operation="delete",
                namespace=ns,
                ids_count=0,
            )
            raise err

        raw_request: Dict[str, Any] = {
            "namespace": ns,
            "ids": [str(i) for i in ids] if ids else None,
            "filter": dict(filter) if filter else None,
        }
        framework_ctx = self._framework_ctx_for_namespace(ns)
        return raw_request, framework_ctx

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
        - Delegates batching + capability-aware upserts to VectorTranslator.
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
            # All sync/async bridging and batching is inside the translator.
            result: UpsertResult = self._translator.upsert(
                raw_request,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            # We intentionally ignore result contents here; failures are surfaced
            # via exceptions (with context) from the translator/adapter.
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

        await self._translator.arun_upsert(
            raw_request,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
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
            err = NotSupported(
                "No embedding_function configured; caller must supply query embedding",
                code="NO_EMBEDDING_FUNCTION",
            )
            attach_context(
                err,
                framework="langchain",
                operation="embed_query",
            )
            raise err

        try:
            embs = self.embedding_function([query])
        except Exception as exc:  # noqa: BLE001
            err = BadRequest(
                f"embedding_function failed for query: {exc}",
                code="EMBEDDING_ERROR",
            )
            attach_context(
                err,
                framework="langchain",
                operation="embed_query",
            )
            raise err
        if not embs or len(embs) != 1:
            err = BadRequest(
                "embedding_function must return exactly one embedding for a single query",
                code="BAD_EMBEDDINGS",
                details={"returned": len(embs) if embs is not None else 0},
            )
            attach_context(
                err,
                framework="langchain",
                operation="embed_query",
            )
            raise err
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
        top_k = k or self.default_top_k

        raw_query, framework_ctx = self._build_query_request(
            query_emb,
            k=top_k,
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
        except Exception:
            # Errors are already enriched by the translator.
            raise

        if not isinstance(result_any, QueryResult):
            err = VectorAdapterError(
                f"VectorTranslator.query returned unsupported type: {type(result_any).__name__}",
                code="BAD_TRANSLATED_RESULT",
            )
            attach_context(
                err,
                framework="langchain",
                operation="similarity_search",
            )
            raise err

        matches = self._apply_score_threshold(list(result_any.matches or []))
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

        This delegates streaming orchestration to VectorTranslator.query_stream.
        The adapter only:
        - Builds the raw query
        - Applies client-side score thresholding
        - Converts matches to LangChain Documents.
        """
        embedding: Optional[Sequence[float]] = kwargs.get("embedding")
        namespace: Optional[str] = kwargs.get("namespace")
        ctx = self._build_ctx(**kwargs)

        query_emb = self._embed_query(query, embedding=embedding)
        top_k = k or self.default_top_k

        raw_query, framework_ctx = self._build_query_request(
            query_emb,
            k=top_k,
            namespace=namespace,
            filter=filter,
            include_vectors=False,
        )

        for item in self._translator.query_stream(
            raw_query,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        ):
            # The translator should stream VectorMatch items directly.
            if not isinstance(item, VectorMatch):
                err = VectorAdapterError(
                    f"VectorTranslator.query_stream yielded unsupported type: {type(item).__name__}",
                    code="BAD_TRANSLATED_CHUNK",
                )
                attach_context(
                    err,
                    framework="langchain",
                    operation="similarity_search_stream",
                )
                raise err

            match = item
            # Client-side score thresholding per chunk
            if self.score_threshold is not None:
                if float(match.score) < float(self.score_threshold):
                    continue

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
        top_k = k or self.default_top_k

        raw_query, framework_ctx = self._build_query_request(
            query_emb,
            k=top_k,
            namespace=namespace,
            filter=filter,
            include_vectors=False,
        )

        result_any = await self._translator.arun_query(
            raw_query,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )

        if not isinstance(result_any, QueryResult):
            err = VectorAdapterError(
                f"VectorTranslator.arun_query returned unsupported type: {type(result_any).__name__}",
                code="BAD_TRANSLATED_RESULT",
            )
            attach_context(
                err,
                framework="langchain",
                operation="asimilarity_search",
            )
            raise err

        matches = self._apply_score_threshold(list(result_any.matches or []))
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
        top_k = k or self.default_top_k

        raw_query, framework_ctx = self._build_query_request(
            query_emb,
            k=top_k,
            namespace=namespace,
            filter=filter,
            include_vectors=False,
        )

        result_any = self._translator.query(
            raw_query,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )

        if not isinstance(result_any, QueryResult):
            err = VectorAdapterError(
                f"VectorTranslator.query returned unsupported type: {type(result_any).__name__}",
                code="BAD_TRANSLATED_RESULT",
            )
            attach_context(
                err,
                framework="langchain",
                operation="similarity_search_with_score",
            )
            raise err

        matches = self._apply_score_threshold(list(result_any.matches or []))
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
        top_k = k or self.default_top_k

        raw_query, framework_ctx = self._build_query_request(
            query_emb,
            k=top_k,
            namespace=namespace,
            filter=filter,
            include_vectors=False,
        )

        result_any = await self._translator.arun_query(
            raw_query,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )

        if not isinstance(result_any, QueryResult):
            err = VectorAdapterError(
                f"VectorTranslator.arun_query returned unsupported type: {type(result_any).__name__}",
                code="BAD_TRANSLATED_RESULT",
            )
            attach_context(
                err,
                framework="langchain",
                operation="asimilarity_search_with_score",
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

        Uses `mmr_similarity_fn` if provided; otherwise falls back to cosine
        similarity. Any error or non-numeric return from the custom function
        is logged and treated as a signal to fall back to cosine.
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
        Improved MMR selector that respects original database scores and allows
        a configurable similarity metric for the diversity term.

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

        # If lambda_mult is 1.0, MMR reduces to pure relevance ranking:
        # skip diversity computation for better performance.
        if lambda_mult >= 1.0:
            scores = [float(match.score) for match in candidate_matches]
            sorted_indices = sorted(
                range(len(candidate_matches)),
                key=lambda i: scores[i],
                reverse=True,
            )
            return sorted_indices[:k]

        # Use original scores from database as relevance measure
        original_scores = [float(match.score) for match in candidate_matches]

        # Build candidate vector list, handling missing or mismatched dimensions
        candidate_vecs: List[List[float]] = []
        dim = len(query_vec)
        for match in candidate_matches:
            vec = match.vector.vector or []
            if not vec or (dim > 0 and len(vec) != dim):
                # Treat missing or dimensionally inconsistent vectors as zero vectors,
                # so MMR gracefully falls back toward original scores.
                candidate_vecs.append([])
            else:
                candidate_vecs.append([float(x) for x in vec])

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
                sim = self._similarity_for_mmr(vec_i, vec_j)

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
        if k <= 0:
            return []

        if not (0.0 <= lambda_mult <= 1.0):
            err = BadRequest(
                f"lambda_mult must be in [0, 1], got {lambda_mult}",
                code="BAD_MMR_LAMBDA",
            )
            attach_context(
                err,
                framework="langchain",
                operation="mmr_search",
                lambda_mult=lambda_mult,
            )
            raise err

        fetch_k: int = kwargs.get("fetch_k") or max(k * 4, k + 5)
        embedding: Optional[Sequence[float]] = kwargs.get("embedding")
        namespace: Optional[str] = kwargs.get("namespace")
        ctx = self._build_ctx(**kwargs)

        query_emb = self._embed_query(query, embedding=embedding)

        raw_query, framework_ctx = self._build_query_request(
            query_emb,
            k=fetch_k,
            namespace=namespace,
            filter=filter,
            include_vectors=True,  # MMR needs vectors
        )

        result_any = self._translator.query(
            raw_query,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )

        if not isinstance(result_any, QueryResult):
            err = VectorAdapterError(
                f"VectorTranslator.query returned unsupported type: {type(result_any).__name__}",
                code="BAD_TRANSLATED_RESULT",
            )
            attach_context(
                err,
                framework="langchain",
                operation="mmr_search",
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
                code="BAD_MMR_LAMBDA",
            )
            attach_context(
                err,
                framework="langchain",
                operation="mmr_search_async",
                lambda_mult=lambda_mult,
            )
            raise err

        fetch_k: int = kwargs.get("fetch_k") or max(k * 4, k + 5)
        embedding: Optional[Sequence[float]] = kwargs.get("embedding")
        namespace: Optional[str] = kwargs.get("namespace")
        ctx = self._build_ctx(**kwargs)

        query_emb = self._embed_query(query, embedding=embedding)

        raw_query, framework_ctx = self._build_query_request(
            query_emb,
            k=fetch_k,
            namespace=namespace,
            filter=filter,
            include_vectors=True,
        )

        result_any = await self._translator.arun_query(
            raw_query,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )

        if not isinstance(result_any, QueryResult):
            err = VectorAdapterError(
                f"VectorTranslator.arun_query returned unsupported type: {type(result_any).__name__}",
                code="BAD_TRANSLATED_RESULT",
            )
            attach_context(
                err,
                framework="langchain",
                operation="mmr_search_async",
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

        raw_request, framework_ctx = self._build_delete_request(
            ids=ids,
            namespace=namespace,
            filter=filter,
        )

        try:
            result: DeleteResult = self._translator.delete(
                raw_request,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            # Result is returned for completeness; current LangChain interface
            # only requires that deletion executes or raises.
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