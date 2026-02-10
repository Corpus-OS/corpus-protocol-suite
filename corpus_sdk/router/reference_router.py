# corpus_sdk/router/reference_router.py
# SPDX-License-Identifier: Apache-2.0

"""
Reference Router Implementation

Purpose
-------
A minimal, single-process router demonstrating the Corpus Protocol Suite's
unified envelope contract. This implementation shows how to:

- Dispatch across multiple protocols (LLM, Embedding, Vector, Graph)
- Apply uniform middleware (auth, deadlines, metrics)
- Handle both unary and streaming operations

IMPORTANT: Production Limitations
----------------------------------
This reference implementation is intentionally simple and NOT suitable for
production use at scale. It lacks:

- Distributed coordination (single-process only)
- Advanced routing logic (model selection, fallbacks, hedging)
- Cost optimization (caching, batching, request coalescing)
- Multi-tenant isolation and quota enforcement
- Sophisticated retry/backoff strategies
- Load balancing and health-aware routing
- Query planning and optimization
- Cross-protocol orchestration

This reference implementation is provided to:
1. Demonstrate protocol interoperability
2. Enable rapid prototyping and development
3. Serve as a learning resource for custom router implementations
4. Validate the unified envelope contract design

Design Philosophy
-----------------
The router is deliberately "thin" - it does minimal work:
- Protocol dispatch based on operation prefix
- Streaming vs unary operation detection
- Error propagation (no transformation)

All intelligence (model selection, cost optimization, retry logic) belongs
in higher-level routing layers or the commercial Corpus Router.
"""

from typing import Any, Dict, Optional, AsyncIterator, Union, overload
import logging

from corpus_sdk.llm.llm_base import WireLLMHandler, LLMProtocolV1
from corpus_sdk.embedding.embedding_base import WireEmbeddingHandler, EmbeddingProtocolV1
from corpus_sdk.vector.vector_base import WireVectorHandler, VectorProtocolV1
from corpus_sdk.graph.graph_base import WireGraphHandler, GraphProtocolV1

LOG = logging.getLogger(__name__)


class ReferenceRouter:
    """
    Reference implementation of a unified protocol router.
    
    This router demonstrates the minimal surface area needed to dispatch
    requests across the Corpus Protocol Suite. It is intentionally simple
    and NOT production-ready.
    
    Supported Protocols:
        - LLM Protocol V1: complete, stream, count_tokens, health
        - Embedding Protocol V1: embed, stream_embed, embed_batch, count_tokens, health
        - Vector Protocol V1: query, batch_query, upsert, delete, create/delete namespace, health
        - Graph Protocol V1: query, stream_query, upsert_nodes, upsert_edges, delete, health
    
    Usage:
        # Development/Testing
        router = ReferenceRouter(
            llm_adapter=OpenAIAdapter(),
            embedding_adapter=OpenAIEmbeddingAdapter(),
            vector_adapter=PineconeAdapter(),
            graph_adapter=Neo4jAdapter(),
        )
        
        # Unary operation
        result = await router.route({
            "op": "llm.complete",
            "ctx": {"tenant": "dev", "deadline_ms": 1234567890},
            "args": {"messages": [{"role": "user", "content": "Hello"}]}
        })
        
        # Streaming operation
        async for chunk in await router.route({
            "op": "llm.stream",
            "ctx": {"tenant": "dev"},
            "args": {"messages": [...]}
        }):
            print(chunk)
        
        # Cross-protocol workflow (RAG example)
        # 1. Generate embedding
        embed_result = await router.route({
            "op": "embedding.embed",
            "ctx": ctx,
            "args": {"text": query, "model": "text-embedding-3-large"}
        })
        
        # 2. Vector similarity search
        search_result = await router.route({
            "op": "vector.query",
            "ctx": ctx,
            "args": {
                "vector": embed_result["result"]["embedding"]["vector"],
                "top_k": 5,
                "namespace": "docs"
            }
        })
        
        # 3. Graph traversal for context enrichment
        graph_result = await router.route({
            "op": "graph.query",
            "ctx": ctx,
            "args": {
                "cypher": "MATCH (d:Document)-[:RELATES_TO]->(c:Concept) WHERE d.id IN $doc_ids RETURN c",
                "params": {"doc_ids": [m["vector"]["id"] for m in search_result["result"]["matches"]]}
            }
        })
        
        # 4. LLM completion with enriched context
        final_result = await router.route({
            "op": "llm.complete",
            "ctx": ctx,
            "args": {
                "messages": build_messages(query, search_result, graph_result),
                "model": "gpt-4"
            }
        })
    
    For production use cases requiring advanced routing, model selection,
    cost optimization, and multi-tenant isolation, see Corpus Router:
        https://www.corpusos.com/products/router
    """
    
    def __init__(
        self,
        *,
        llm_adapter: Optional[LLMProtocolV1] = None,
        embedding_adapter: Optional[EmbeddingProtocolV1] = None,
        vector_adapter: Optional[VectorProtocolV1] = None,
        graph_adapter: Optional[GraphProtocolV1] = None,
    ):
        """
        Initialize the reference router with protocol adapters.
        
        Args:
            llm_adapter: Adapter implementing LLMProtocolV1
            embedding_adapter: Adapter implementing EmbeddingProtocolV1
            vector_adapter: Adapter implementing VectorProtocolV1
            graph_adapter: Adapter implementing GraphProtocolV1
        
        Note:
            At least one adapter must be provided. Operations for protocols
            without an adapter will raise NotSupported.
            
            Each adapter is wrapped in its corresponding WireHandler to provide
            the canonical envelope-based interface. Adapters can be swapped at
            runtime for testing or A/B experiments.
        """
        self._llm = WireLLMHandler(llm_adapter) if llm_adapter else None
        self._embedding = WireEmbeddingHandler(embedding_adapter) if embedding_adapter else None
        self._vector = WireVectorHandler(vector_adapter) if vector_adapter else None
        self._graph = WireGraphHandler(graph_adapter) if graph_adapter else None
        
        # Track which protocols are available for introspection
        self._available_protocols = set()
        if self._llm:
            self._available_protocols.add("llm")
        if self._embedding:
            self._available_protocols.add("embedding")
        if self._vector:
            self._available_protocols.add("vector")
        if self._graph:
            self._available_protocols.add("graph")
        
        if not self._available_protocols:
            LOG.warning("ReferenceRouter initialized with no adapters - all requests will fail")
    
    def available_protocols(self) -> set:
        """
        Return set of available protocol names.
        
        Useful for capability discovery and routing decisions.
        
        Returns:
            Set of protocol names: {"llm", "embedding", "vector", "graph"}
        """
        return self._available_protocols.copy()
    
    @overload
    async def route(self, envelope: Dict[str, Any]) -> Dict[str, Any]: ...
    
    @overload
    async def route(self, envelope: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]: ...
    
    async def route(
        self,
        envelope: Dict[str, Any]
    ) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
        """
        Route a request envelope to the appropriate protocol handler.
        
        Supports both unary and streaming operations. For streaming operations,
        returns an AsyncIterator that must be consumed by the caller.
        
        Protocol Dispatch Rules:
            - Operations starting with "llm." → LLM protocol
            - Operations starting with "embedding." → Embedding protocol
            - Operations starting with "vector." → Vector protocol
            - Operations starting with "graph." → Graph protocol
        
        Streaming Operations (return AsyncIterator):
            - llm.stream → LLM streaming completion
            - embedding.stream_embed → Embedding streaming
            - graph.stream_query → Graph streaming query
        
        Unary Operations (return Dict):
            - All other operations
        
        Args:
            envelope: Canonical envelope with required keys: op, ctx, args
                {
                    "op": "<protocol>.<operation>",
                    "ctx": {
                        "request_id": "...",
                        "deadline_ms": 1234567890,
                        "tenant": "...",
                        ...
                    },
                    "args": {
                        # Operation-specific arguments
                    }
                }
        
        Returns:
            For unary operations: Response envelope Dict[str, Any]
                {
                    "ok": true,
                    "code": "OK",
                    "ms": 123.45,
                    "result": { ... }
                }
            
            For streaming operations: AsyncIterator[Dict[str, Any]]
                Yields zero or more chunk envelopes:
                {
                    "ok": true,
                    "code": "STREAMING",
                    "ms": 123.45,
                    "chunk": { ... }
                }
                
                Followed by either:
                - Final success chunk (is_final: true)
                - Error envelope (ok: false)
        
        Raises:
            BadRequest: If envelope is malformed or missing required fields
            NotSupported: If protocol or operation is not available
            Various protocol-specific errors from adapters
        
        Examples:
            # Unary LLM completion
            result = await router.route({
                "op": "llm.complete",
                "ctx": {"tenant": "acme"},
                "args": {"messages": [...], "model": "gpt-4"}
            })
            
            # Streaming LLM completion
            stream = await router.route({
                "op": "llm.stream",
                "ctx": {"tenant": "acme"},
                "args": {"messages": [...], "model": "gpt-4"}
            })
            async for chunk in stream:
                print(chunk["chunk"]["text"])
            
            # Vector query
            result = await router.route({
                "op": "vector.query",
                "ctx": {"tenant": "acme"},
                "args": {"vector": [...], "top_k": 5}
            })
            
            # Graph query with streaming
            stream = await router.route({
                "op": "graph.stream_query",
                "ctx": {"tenant": "acme"},
                "args": {"cypher": "MATCH (n) RETURN n LIMIT 1000"}
            })
            async for chunk in stream:
                process_nodes(chunk["chunk"]["nodes"])
        """
        # Validate envelope structure
        if not isinstance(envelope, dict):
            from corpus_sdk.llm import BadRequest
            raise BadRequest("envelope must be a dictionary")
        
        op = envelope.get("op")
        if not isinstance(op, str):
            from corpus_sdk.llm import BadRequest
            raise BadRequest("envelope must include string 'op' field")
        
        # LLM Protocol
        if op.startswith("llm."):
            if self._llm is None:
                from corpus_sdk.llm import NotSupported
                raise NotSupported(
                    "LLM protocol not configured - provide llm_adapter to ReferenceRouter"
                )
            
            # Streaming operation
            if op == "llm.stream":
                return self._llm.handle_stream(envelope)
            
            # Unary operations: llm.complete, llm.count_tokens, llm.health, llm.capabilities
            return await self._llm.handle(envelope)
        
        # Embedding Protocol
        elif op.startswith("embedding."):
            if self._embedding is None:
                from corpus_sdk.embedding import NotSupported
                raise NotSupported(
                    "Embedding protocol not configured - provide embedding_adapter to ReferenceRouter"
                )
            
            # Streaming operation
            if op == "embedding.stream_embed":
                return self._embedding.handle_stream(envelope)
            
            # Unary operations: embedding.embed, embedding.embed_batch, embedding.count_tokens, etc.
            return await self._embedding.handle(envelope)
        
        # Vector Protocol
        elif op.startswith("vector."):
            if self._vector is None:
                from corpus_sdk.vector import NotSupported
                raise NotSupported(
                    "Vector protocol not configured - provide vector_adapter to ReferenceRouter"
                )
            
            # Vector protocol has no streaming operations (queries return complete result sets)
            return await self._vector.handle(envelope)
        
        # Graph Protocol
        elif op.startswith("graph."):
            if self._graph is None:
                from corpus_sdk.graph import NotSupported
                raise NotSupported(
                    "Graph protocol not configured - provide graph_adapter to ReferenceRouter"
                )
            
            # Streaming operation
            if op == "graph.stream_query":
                return self._graph.handle_stream(envelope)
            
            # Unary operations: graph.query, graph.upsert_nodes, graph.upsert_edges, etc.
            return await self._graph.handle(envelope)
        
        # Unknown protocol
        else:
            from corpus_sdk.llm import NotSupported
            raise NotSupported(
                f"unknown protocol in operation: {op}. "
                f"Available protocols: {sorted(self._available_protocols)}"
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check health of all configured protocol adapters.
        
        Calls the health() method on each configured adapter and aggregates
        results. Useful for readiness probes and monitoring.
        
        Returns:
            Dictionary mapping protocol names to their health status:
            {
                "llm": {"ok": true, "server": "openai", "version": "v1"},
                "embedding": {"ok": true, "server": "openai", "version": "v1"},
                "vector": {"ok": true, "server": "pinecone", "version": "2.0"},
                "graph": {"ok": true, "server": "neo4j", "version": "5.0"}
            }
        
        Note:
            Individual protocol health check failures do not raise exceptions;
            they are included in the result with ok: false. This allows
            partial health status reporting.
        """
        health = {}
        
        if self._llm:
            try:
                result = await self._llm.handle({
                    "op": "llm.health",
                    "ctx": {},
                    "args": {}
                })
                health["llm"] = result.get("result", {})
            except Exception as e:
                health["llm"] = {"ok": False, "error": str(e)}
        
        if self._embedding:
            try:
                result = await self._embedding.handle({
                    "op": "embedding.health",
                    "ctx": {},
                    "args": {}
                })
                health["embedding"] = result.get("result", {})
            except Exception as e:
                health["embedding"] = {"ok": False, "error": str(e)}
        
        if self._vector:
            try:
                result = await self._vector.handle({
                    "op": "vector.health",
                    "ctx": {},
                    "args": {}
                })
                health["vector"] = result.get("result", {})
            except Exception as e:
                health["vector"] = {"ok": False, "error": str(e)}
        
        if self._graph:
            try:
                result = await self._graph.handle({
                    "op": "graph.health",
                    "ctx": {},
                    "args": {}
                })
                health["graph"] = result.get("result", {})
            except Exception as e:
                health["graph"] = {"ok": False, "error": str(e)}
        
        return health
    
    async def __aenter__(self):
        """Support async context manager usage."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up adapters on context exit."""
        await self.close()
    
    async def close(self):
        """
        Close all configured adapters.
        
        Calls close() on each adapter if available. Useful for cleaning up
        HTTP sessions, connection pools, and other resources.
        """
        if self._llm and hasattr(self._llm._adapter, "close"):
            try:
                await self._llm._adapter.close()
            except Exception as e:
                LOG.warning(f"Error closing LLM adapter: {e}")
        
        if self._embedding and hasattr(self._embedding._adapter, "close"):
            try:
                await self._embedding._adapter.close()
            except Exception as e:
                LOG.warning(f"Error closing Embedding adapter: {e}")
        
        if self._vector and hasattr(self._vector._adapter, "close"):
            try:
                await self._vector._adapter.close()
            except Exception as e:
                LOG.warning(f"Error closing Vector adapter: {e}")
        
        if self._graph and hasattr(self._graph._adapter, "close"):
            try:
                await self._graph._adapter.close()
            except Exception as e:
                LOG.warning(f"Error closing Graph adapter: {e}")


__all__ = ["ReferenceRouter"]
