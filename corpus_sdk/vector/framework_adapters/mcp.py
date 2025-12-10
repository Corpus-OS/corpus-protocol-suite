# corpus_sdk/vector/mcp_vector_service.py
# SPDX-License-Identifier: Apache-2.0

"""
MCP Vector Translation Service

This module provides the MCP-facing vector service implementation for
Corpus VectorProtocolV1. It mirrors the structure, design, and
responsibilities of MCPLLMTranslationService and MCPGraphTranslationService:

- Async-only API surface (no sync methods)
- No batching, no caching, no retries, no circuit breakers
- Thin translation layer around VectorTranslator + VectorProtocolV1
- Uses OperationContext consistently for tracing
- Uses shared error utils for consistent diagnostics
- Does not add algorithms (MMR, normalization, ranking, etc.)
- Accepts raw embeddings + metadata (MCP-native shape)
- Exposes minimal high-level operations required by MCP:
    * capabilities()
    * similarity_search()
    * stream_similarity_search()
    * add_texts()
    * add_documents()
    * delete()

Design Philosophy
-----------------
- MCP services must be predictable, minimal, and fully aligned.
- All complexity lives inside the underlying adapter or VectorTranslator.
- This file NEVER reinvents protocol logic, token logic, batching, or shaping.
- Errors are wrapped identically to other MCP services via attach_context().
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Iterable, AsyncIterator, Mapping

from corpus_sdk.vector.vector_base import (
    VectorProtocolV1,
    VectorCapabilities,
    QueryResult,
    QueryChunk,
    OperationContext,
    BadRequest,
    NotSupported,
)
from corpus_sdk.core.context_translation import from_dict as ctx_from_dict
from corpus_sdk.core.error_context import attach_context
from corpus_sdk.vector.framework_adapters.common.vector_translation import (
    VectorTranslator,
    DefaultVectorFrameworkTranslator,
)

# Shared framework utils for consistency across MCP services
from corpus_sdk.vector.framework_adapters.common.framework_utils import (
    VectorCoercionErrorCodes,
    VectorResourceLimits,
    VectorValidationFlags,
    normalize_vector_context,
    attach_vector_context_to_framework_ctx,
)

import logging
logger = logging.getLogger(__name__)


VECTOR_ERROR_CODES = VectorCoercionErrorCodes(framework_label="mcp")
VECTOR_LIMITS = VectorResourceLimits()
VECTOR_FLAGS = VectorValidationFlags()


class MCPVectorTranslationService:
    """
    MCP-facing vector service for Corpus.

    This class strictly mirrors the architecture of MCPLLMTranslationService
    and MCPGraphTranslationService:

    - Translate MCP request → raw vector spec dict → VectorTranslator call
    - Wrap errors using attach_context()
    - Use OperationContext consistently
    - Return raw dicts suitable for JSON serialization
    - NO sync methods
    - NO MMR
    - NO batching logic
    - NO custom transformations of embeddings or metadata
    """

    def __init__(
        self,
        *,
        adapter: VectorProtocolV1,
        namespace: Optional[str] = None,
        default_top_k: int = 4,
    ) -> None:
        self._adapter = adapter
        self._namespace = namespace
        self._default_top_k = int(default_top_k)
        self._caps: Optional[VectorCapabilities] = None

        # Translator identical pattern to MCPLLM + MCPGraph
        self._translator = VectorTranslator(
            adapter=self._adapter,
            framework="mcp",
            translator=DefaultVectorFrameworkTranslator(),
        )

    # ----------------------------
    # Internal helpers
    # ----------------------------

    def _resolve_namespace(self, ns: Optional[str]) -> Optional[str]:
        return ns if ns is not None else self._namespace

    def _resolve_op_ctx(self, ctx: Optional[Any]) -> Optional[OperationContext]:
        if ctx is None:
            return None
        if isinstance(ctx, OperationContext):
            return ctx
        try:
            translated = ctx_from_dict(ctx)
            if isinstance(translated, OperationContext):
                return translated
        except Exception as exc:
            attach_context(
                exc,
                framework="mcp",
                operation="context_translation",
            )
            raise
        return None

    def _framework_ctx(self, namespace: Optional[str]) -> Dict[str, Any]:
        raw = {}
        if namespace:
            raw["namespace"] = namespace

        vector_ctx = normalize_vector_context(
            raw,
            framework="mcp",
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

    async def _get_caps(self) -> VectorCapabilities:
        if self._caps is None:
            try:
                self._caps = await self._translator.arun_capabilities()
            except Exception as exc:
                attach_context(exc, framework="mcp", operation="capabilities")
                raise
        return self._caps

    def _validate_query_result(self, result: Any, *, op: str) -> QueryResult:
        if not isinstance(result, QueryResult):
            raise BadRequest(
                f"{op} returned invalid result type: {type(result).__name__}",
                code="BAD_QUERY_RESULT",
            )
        return result

    def _validate_stream_chunk(self, chunk: Any, *, op: str) -> QueryChunk:
        if not isinstance(chunk, QueryChunk):
            raise BadRequest(
                f"{op} yielded invalid chunk type: {type(chunk).__name__}",
                code="BAD_STREAM_CHUNK",
            )
        return chunk

    # ----------------------------
    # Public API (async-only)
    # ----------------------------

    async def capabilities(self) -> VectorCapabilities:
        """Return vector store capabilities."""
        return await self._get_caps()

    async def add_texts(
        self,
        texts: Iterable[str],
        *,
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Mapping[str, Any]]] = None,
        namespace: Optional[str] = None,
        ctx: Optional[Any] = None,
    ) -> List[str]:
        """
        Add raw text + embedding rows to the vector store.
        MCP-native shape → list of:
        {
            "id": "string",
            "vector": [...],
            "metadata": {},
            "namespace": ...
        }

        This method does:
        - Validate lengths
        - Shape raw dicts
        - Let VectorTranslator handle the protocol call
        """

        texts_list = list(texts)
        n = len(texts_list)
        if n == 0:
            return []

        if embeddings is None or len(embeddings) != n:
            raise BadRequest(
                "embeddings must be provided and match text count",
                code="BAD_EMBEDDINGS",
            )

        meta_list: List[Mapping[str, Any]] = []
        if metadatas is None:
            meta_list = [{} for _ in range(n)]
        else:
            if len(metadatas) != n:
                raise BadRequest(
                    "metadatas length mismatch",
                    code="BAD_METADATA",
                )
            meta_list = list(metadatas)

        ns = self._resolve_namespace(namespace)
        op_ctx = self._resolve_op_ctx(ctx)

        # Build raw vector rows directly for translator
        vectors = []
        for text, emb, meta in zip(texts_list, embeddings, meta_list):
            vectors.append(
                {
                    "id": meta.get("id") or meta.get("doc_id") or None,
                    "vector": [float(x) for x in emb],
                    "metadata": dict(meta),
                    "namespace": ns,
                    "text": text,
                }
            )

        raw_request = {
            "namespace": ns,
            "vectors": vectors,
        }

        try:
            await self._translator.arun_upsert(
                raw_request,
                op_ctx=op_ctx,
                framework_ctx=self._framework_ctx(ns),
            )
        except Exception as exc:
            attach_context(exc, framework="mcp", operation="add_texts")
            raise

        # IDs may come from metadata or may be assigned by backend
        ids = [str(meta.get("id") or meta.get("doc_id") or "") for meta in meta_list]
        return ids

    async def add_documents(
        self,
        documents: Iterable[Mapping[str, Any]],
        *,
        embeddings: Optional[List[List[float]]] = None,
        namespace: Optional[str] = None,
        ctx: Optional[Any] = None,
    ) -> List[str]:
        """
        Add general documents.
        Document shape:
            {
                "text": "...",
                "metadata": {...}
            }
        """
        docs = list(documents)
        texts = [str(d.get("text") or "") for d in docs]
        metas = [dict(d.get("metadata") or {}) for d in docs]
        return await self.add_texts(
            texts,
            embeddings=embeddings,
            metadatas=metas,
            namespace=namespace,
            ctx=ctx,
        )

    async def similarity_search(
        self,
        *,
        vector: List[float],
        k: Optional[int] = None,
        namespace: Optional[str] = None,
        filter: Optional[Mapping[str, Any]] = None,
        ctx: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Return search results as raw dicts matching MCP expectations."""
        ns = self._resolve_namespace(namespace)
        op_ctx = self._resolve_op_ctx(ctx)
        top_k = int(k or self._default_top_k)

        raw_request = {
            "vector": [float(x) for x in vector],
            "top_k": top_k,
            "namespace": ns,
            "filters": dict(filter) if filter else None,
            "include_metadata": True,
            "include_vectors": False,
        }

        try:
            result_any = await self._translator.arun_query(
                raw_request,
                op_ctx=op_ctx,
                framework_ctx=self._framework_ctx(ns),
            )
        except Exception as exc:
            attach_context(exc, framework="mcp", operation="similarity_search")
            raise

        result = self._validate_query_result(result_any, op="similarity_search")
        matches = list(result.matches or [])
        payload = []

        for m in matches:
            vector_obj = m.vector
            meta = dict(vector_obj.metadata or {})
            payload.append(
                {
                    "id": str(vector_obj.id),
                    "score": float(m.score),
                    "metadata": meta,
                    "text": meta.get("text", ""),
                }
            )
        return payload

    async def stream_similarity_search(
        self,
        *,
        vector: List[float],
        k: Optional[int] = None,
        namespace: Optional[str] = None,
        filter: Optional[Mapping[str, Any]] = None,
        ctx: Optional[Any] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Streaming version of similarity_search."""
        ns = self._resolve_namespace(namespace)
        op_ctx = self._resolve_op_ctx(ctx)
        top_k = int(k or self._default_top_k)

        raw_request = {
            "vector": [float(x) for x in vector],
            "top_k": top_k,
            "namespace": ns,
            "filters": dict(filter) if filter else None,
            "include_metadata": True,
            "include_vectors": False,
        }

        try:
            async for chunk in self._translator.arun_query_stream(
                raw_request,
                op_ctx=op_ctx,
                framework_ctx=self._framework_ctx(ns),
            ):
                chunk_qc = self._validate_stream_chunk(chunk, op="stream_similarity_search")

                for m in chunk_qc.matches or []:
                    vector_obj = m.vector
                    meta = dict(vector_obj.metadata or {})
                    yield {
                        "id": str(vector_obj.id),
                        "score": float(m.score),
                        "metadata": meta,
                        "text": meta.get("text", ""),
                    }

        except Exception as exc:
            attach_context(exc, framework="mcp", operation="stream_similarity_search")
            raise

    async def delete(
        self,
        *,
        ids: Optional[List[str]] = None,
        namespace: Optional[str] = None,
        filter: Optional[Mapping[str, Any]] = None,
        ctx: Optional[Any] = None,
    ) -> None:
        """Delete vectors by IDs or filter."""
        ns = self._resolve_namespace(namespace)
        op_ctx = self._resolve_op_ctx(ctx)

        raw_request = {
            "namespace": ns,
            "ids": ids,
            "filter": dict(filter) if filter else None,
        }

        try:
            await self._translator.arun_delete(
                raw_request,
                op_ctx=op_ctx,
                framework_ctx=self._framework_ctx(ns),
            )
        except Exception as exc:
            attach_context(exc, framework="mcp", operation="delete")
            raise