# Corpus OS Quickstart

**Build Production-Ready Corpus Protocol Adapters in 15 Minutes**

**Table of Contents**
- [0. Mental Model (What You're Actually Building)](#0-mental-model-what-youre-actually-building)
- [0.5 When to Implement Which Operations](#05-when-to-implement-which-operations)
- [1. Prerequisites & Setup](#1-prerequisites--setup)
- [2. Testing Your Adapter (Certification Suite)](#2-testing-your-adapter-certification-suite)
- [3. Hello World: Complete Reference Adapters](#3-hello-world-complete-reference-adapters)
  - [3.1 Embedding Adapter (OpenAI/Cohere Style)](#31-embedding-adapter-openaicohere-style)
  - [3.2 LLM Adapter (Chat Completion Style)](#32-llm-adapter-chat-completion-style)
  - [3.3 Vector Adapter (Pinecone/Qdrant Style)](#33-vector-adapter-pineconeqdrant-style)
  - [3.4 Graph Adapter (Neo4j/JanusGraph Style)](#34-graph-adapter-neo4jjanusgraph-style)
- [4. Running Certification Tests](#4-running-certification-tests)
- [5. Understanding Certification Results](#5-understanding-certification-results)
- [6. What to Read Next](#6-what-to-read-next)
- [7. Protocol-Specific Requirements & Pitfalls](#7-protocol-specific-requirements--pitfalls)
  - [7.1 Embedding Protocol](#71-embedding-protocol)
  - [7.2 LLM Protocol](#72-llm-protocol)
  - [7.3 Vector Protocol](#73-vector-protocol)
  - [7.4 Graph Protocol](#74-graph-protocol)
- [8. Certification Checklist](#8-certification-checklist)
- [Appendix A: Common Pitfalls by Component](#appendix-a-common-pitfalls-by-component)
- [Appendix B: Glossary](#appendix-b-glossary)
- [Appendix C: Debugging & Troubleshooting](#appendix-c-debugging--troubleshooting)

---

> **Goal:** Get a Gold-certified adapter speaking **any Corpus Protocol v1.0** (Embedding, LLM, Vector, or Graph) in **under 15 minutes**.  
> **Audience:** SDK / adapter authors for embedding providers, LLM APIs, vector databases, and graph databases.  
> **You'll build:** A complete, certified adapter with streaming, batch operations, error mapping, and full conformance.

**By the end of this guide you will have:**
- ✅ A fully tested adapter implementation for your chosen protocol
- ✅ Streaming and batch operation support (where applicable)
- ✅ Proper error mapping and deadline propagation
- ✅ Cache invalidation (Vector/Graph) or idempotency (Embedding)
- ✅ **Gold certification** from the official conformance suite
- ✅ **Full compliance with Corpus Protocol v1.0 specification**

---

## 0. Mental Model (What You're Actually Building)

An **adapter** is a thin translation layer that converts between Corpus Protocol and your provider's native API:

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Your Provider  │◄────┤  YourAdapter │◄────┤  Corpus Base    │
│  (OpenAI, etc.) │     │  (_do_* hooks)│     │  (infrastructure)│
└─────────────────┘     └──────────────┘     └─────────────────┘
```

**You implement only:**
- `_do_capabilities()` - What your adapter supports (**MUST include `protocol` field**)
- `_do_embed()` / `_do_complete()` / `_do_query()` / etc. - Core operation
- `_do_stream_*()` - Streaming (if supported)
- `_do_health()` - Liveness check
- `_do_get_stats()` - Service statistics (optional but recommended)

**The base class provides automatically:**
- ✅ JSON envelope parsing/serialization
- ✅ Deadline enforcement & timeout propagation
- ✅ Circuit breaker patterns
- ✅ Rate limiting
- ✅ Read-path caching (standalone mode)
- ✅ Metrics emission (tenant-hashed, SIEM-safe)
- ✅ Error normalization to canonical codes
- ✅ Batch operation fallbacks

**Critical insight:** The base class is *not* abstract—it provides working fallbacks. You only override what your provider does *better* than the default.

The full protocol specification is embedded in the docstrings of each base class:
- [`../../corpus_sdk/embedding/embedding_base.py`](../../corpus_sdk/embedding/embedding_base.py) — Embedding Protocol V1
- [`../../corpus_sdk/llm/llm_base.py`](../../corpus_sdk/llm/llm_base.py) — LLM Protocol V1
- [`../../corpus_sdk/vector/vector_base.py`](../../corpus_sdk/vector/vector_base.py) — Vector Protocol V1
- [`../../corpus_sdk/graph/graph_base.py`](../../corpus_sdk/graph/graph_base.py) — Graph Protocol V1

---

## 0.5 When to Implement Which Operations

| Operation | When to Override |
|-----------|------------------|
| `_do_capabilities()` | **ALWAYS** (REQUIRED) |
| `_do_embed()` / `_do_complete()` / `_do_query()` / etc. | **ALWAYS** (REQUIRED) |
| `_do_health()` | **ALWAYS** (REQUIRED) |
| `_do_stream_*()` | If your provider supports streaming |
| `_do_batch_*()` | If your provider supports batching |
| `_do_count_tokens()` | If you have a tokenizer (Embedding/LLM) |
| `_do_get_stats()` | Optional - for observability |
| `_do_create_namespace()` | If your vector store supports namespaces |
| `_do_transaction()` | If your graph database supports transactions |

---

## 1. Prerequisites & Setup

### Requirements
- Python 3.10+
- `corpus-sdk` ≥ 1.0.0
- `pytest` ≥ 7.0 (for certification)

### Installation

```bash
pip install corpus-sdk
pip install pytest pytest-asyncio  # Certification dependencies
pip install httpx                   # For real HTTP clients (recommended)
```

---

## 2. Testing Your Adapter (Certification Suite)

The Corpus OS certification suite ships with the SDK. You do not need to create test files or conftest.py—they are already installed with `corpus-sdk`.

### Step 1: Set Your Adapter

```bash
export CORPUS_ADAPTER=my_project.adapters:MyEmbeddingAdapter
```

This tells the conformance suite which adapter to test. The format is `module:ClassName`.

### Step 2: Run the Tests

```bash
# Run embedding protocol tests directly from the installed SDK
pytest $(python -c "import corpus_sdk; print(corpus_sdk.__path__[0])")/tests/embedding/ -v

# Or run all protocols
pytest $(python -c "import corpus_sdk; print(corpus_sdk.__path__[0])")/tests/ -v
```

### Step 3: Watch It Fail

```bash
_________________________________ FAILURE __________________________________
NotImplementedError: _do_capabilities not implemented
```

Each failure tells you exactly what to implement next. Keep running tests until you see:

```
================== 47 passed in 1.2s ==================
CORPUS PROTOCOL SUITE - GOLD CERTIFIED
```

**That's it.** No conftest.py to write. No test files to copy. The certification framework is already installed and ready to test your adapter.

---

## 3. Hello World: Complete Reference Adapters

This section provides **four complete, specification-compliant reference implementations**—one for each protocol. **Choose the one that matches your provider type.**

> **Important:** These adapters show real implementation patterns for connecting to actual providers. They implement only the `_do_*()` hooks required by the base classes.

---

### 3.1 Embedding Adapter (OpenAI/Cohere Style)

Create `adapters/hello_embedding.py`:

```python
import asyncio
import hashlib
from typing import AsyncIterator, Optional, List, Dict, Any
import httpx

from corpus_sdk.embedding.embedding_base import (
    BaseEmbeddingAdapter,
    EmbeddingCapabilities,
    EmbedSpec,
    BatchEmbedSpec,
    EmbedResult,
    BatchEmbedResult,
    EmbeddingVector,
    EmbedChunk,
    OperationContext,
    # Canonical errors
    BadRequest,
    ResourceExhausted,
    AuthError,
    Unavailable,
    DeadlineExceeded,
    NotSupported,
)

class HelloEmbeddingAdapter(BaseEmbeddingAdapter):
    """
    Production-ready embedding adapter for a hypothetical provider.
    
    This demonstrates real patterns:
    - HTTP client with timeout propagation
    - Idempotency key storage
    - Error mapping from provider responses
    - Streaming support
    """
    
    def __init__(self, api_key: str, endpoint: Optional[str] = None, mode: str = "standalone"):
        """Initialize with real provider credentials."""
        super().__init__(mode=mode)
        self.api_key = api_key
        self.endpoint = endpoint or "https://api.example.com/v1/embeddings"
        self.client = httpx.AsyncClient(timeout=30.0)
        self._idempotency_cache = {}  # Replace with Redis in production

    async def _do_capabilities(self) -> EmbeddingCapabilities:
        """Advertise what this adapter supports."""
        return EmbeddingCapabilities(
            server="hello-embedding",
            protocol="embedding/v1.0",  # ✅ REQUIRED
            version="1.0.0",
            supported_models=("text-embedding-001", "text-embedding-002"),
            max_batch_size=100,
            max_text_length=8192,
            max_dimensions=1536,
            supports_normalization=True,
            normalizes_at_source=False,
            supports_truncation=True,
            supports_token_counting=True,
            supports_streaming=True,
            supports_batch_embedding=True,
            supports_deadline=True,
            idempotent_writes=True,      # ✅ REQUIRED
            supports_multi_tenant=True,
            truncation_mode="base",
        )

    async def _do_embed(
        self,
        spec: EmbedSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> EmbedResult:
        """Generate embedding by calling provider API."""
        # Idempotency check
        if ctx and ctx.idempotency_key and ctx.tenant:
            cache_key = f"idem:{ctx.tenant}:{ctx.idempotency_key}"
            cached = self._idempotency_cache.get(cache_key)
            if cached:
                return cached

        # Deadline propagation
        timeout = None
        if ctx and ctx.deadline_ms:
            remaining = ctx.remaining_ms()
            if remaining <= 0:
                raise DeadlineExceeded("deadline already expired")
            timeout = remaining / 1000.0

        try:
            # Call provider API with timeout
            response = await self.client.post(
                self.endpoint,
                json={
                    "model": spec.model,
                    "input": spec.text,
                    "truncate": spec.truncate,
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()

            # Map provider response to Corpus format
            vector = data["embedding"]
            tokens = data.get("usage", {}).get("prompt_tokens", len(spec.text) // 4)

            result = EmbedResult(
                embedding=EmbeddingVector(
                    vector=vector,
                    text=spec.text,
                    model=spec.model,
                    dimensions=len(vector),
                ),
                model=spec.model,
                text=spec.text,
                tokens_used=tokens,
                truncated=data.get("truncated", False),
            )

            # Store idempotency result
            if ctx and ctx.idempotency_key and ctx.tenant:
                self._idempotency_cache[cache_key] = result

            return result

        except httpx.TimeoutException:
            raise DeadlineExceeded("provider timeout")
        except httpx.HTTPStatusError as e:
            raise self._map_provider_error(e)
        except Exception as e:
            raise Unavailable(f"provider error: {str(e)}")

    async def _do_stream_embed(
        self,
        spec: EmbedSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> AsyncIterator[EmbedChunk]:
        """Stream embedding generation (if provider supports it)."""
        # Similar to _do_embed but yields chunks progressively
        # Implementation depends on provider streaming API
        pass

    async def _do_embed_batch(
        self,
        spec: BatchEmbedSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> BatchEmbedResult:
        """Batch embed multiple texts."""
        embeddings = []
        failures = []

        for idx, text in enumerate(spec.texts):
            try:
                # Call provider for each text (or use batch API if supported)
                result = await self._do_embed(
                    EmbedSpec(model=spec.model, text=text, truncate=spec.truncate),
                    ctx=ctx,
                )
                embeddings.append(
                    EmbeddingVector(
                        vector=result.embedding.vector,
                        text=text,
                        model=spec.model,
                        dimensions=len(result.embedding.vector),
                        index=idx,  # ✅ REQUIRED for correlation
                    )
                )
            except Exception as e:
                failures.append({
                    "index": idx,
                    "error": type(e).__name__,
                    "code": getattr(e, "code", "UNKNOWN"),
                    "message": str(e),
                })

        return BatchEmbedResult(
            embeddings=embeddings,
            model=spec.model,
            total_texts=len(spec.texts),
            total_tokens=sum(len(t) // 4 for t in spec.texts),
            failures=failures,  # ✅ REQUIRED field name
        )

    async def _do_count_tokens(
        self,
        text: str,
        model: str,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> int:
        """Count tokens using provider's tokenizer or approximation."""
        # In production, use tiktoken or similar
        return len(text) // 4

    async def _do_health(self, *, ctx: Optional[OperationContext] = None) -> Dict[str, Any]:
        """Check provider health."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.endpoint}/health",
                    timeout=5.0,
                )
                if response.status_code == 200:
                    return {"ok": True, "status": "ok", "server": "hello-embedding", "version": "1.0.0"}
                return {"ok": False, "status": "degraded", "server": "hello-embedding", "version": "1.0.0"}
        except Exception:
            return {"ok": False, "status": "down", "server": "hello-embedding", "version": "1.0.0"}

    def _map_provider_error(self, error: httpx.HTTPStatusError) -> Exception:
        """Map provider HTTP errors to canonical Corpus errors."""
        status = error.response.status_code
        if status == 429:
            retry = int(error.response.headers.get("Retry-After", 5)) * 1000
            return ResourceExhausted("rate limit exceeded", retry_after_ms=retry)
        if status == 401:
            return AuthError("invalid API key")
        if status == 400:
            return BadRequest(error.response.text)
        if status >= 500:
            return Unavailable("provider unavailable", retry_after_ms=1000)
        return error
```

**What makes this specification-compliant:**
- ✅ **REQUIRED** `protocol="embedding/v1.0"` in capabilities
- ✅ **REQUIRED** `idempotent_writes=True` in capabilities
- ✅ **REQUIRED** Batch field name `failures` (not `failed_texts`)
- ✅ **REQUIRED** Index field for batch correlation
- ✅ **REQUIRED** Idempotency key deduplication (24-hour retention)
- ✅ **REQUIRED** Constructor accepts `endpoint=None`
- ✅ **REQUIRED** Deadline propagation using `ctx.remaining_ms()`
- ✅ **REQUIRED** Graded health status (`ok`/`degraded`/`down`)

---

### 3.2 LLM Adapter (Chat Completion Style)

Create `adapters/hello_llm.py`:

```python
import asyncio
import secrets
from typing import AsyncIterator, Optional, List, Dict, Any, Union, Mapping
import httpx

from corpus_sdk.llm.llm_base import (
    BaseLLMAdapter,
    LLMCapabilities,
    LLMCompletion,
    LLMChunk,
    TokenUsage,
    ToolCall,
    ToolCallFunction,
    OperationContext,
    # Canonical errors
    BadRequest,
    ResourceExhausted,
    AuthError,
    Unavailable,
    DeadlineExceeded,
    NotSupported,
)

class HelloLLMAdapter(BaseLLMAdapter):
    """
    Production-ready LLM adapter for a hypothetical provider.
    
    Demonstrates:
    - Chat completion API integration
    - Streaming support
    - Tool calling passthrough
    """
    
    def __init__(self, api_key: str, endpoint: Optional[str] = None, mode: str = "standalone"):
        super().__init__(mode=mode)
        self.api_key = api_key
        self.endpoint = endpoint or "https://api.example.com/v1/chat/completions"
        self.client = httpx.AsyncClient(timeout=30.0)

    async def _do_capabilities(self) -> LLMCapabilities:
        """Advertise LLM capabilities."""
        return LLMCapabilities(
            server="hello-llm",
            protocol="llm/v1.0",  # ✅ REQUIRED
            version="1.0.0",
            model_family="gpt-4",  # ✅ REQUIRED
            max_context_length=8192,
            supports_streaming=True,
            supports_roles=True,
            supports_json_output=True,
            supports_tools=True,
            supports_parallel_tool_calls=True,
            supports_tool_choice=True,
            max_tool_calls_per_turn=5,
            idempotent_writes=False,
            supports_multi_tenant=True,
            supports_system_message=True,
            supports_deadline=True,
            supports_count_tokens=True,
            supported_models=("gpt-4", "gpt-3.5-turbo"),
        )

    async def _do_complete(
        self,
        *,
        messages: List[Mapping[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        model: Optional[str] = None,
        system_message: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        ctx: Optional[OperationContext] = None,
    ) -> LLMCompletion:
        """Call provider chat completion API."""
        # Deadline propagation
        timeout = None
        if ctx and ctx.deadline_ms:
            remaining = ctx.remaining_ms()
            if remaining <= 0:
                raise DeadlineExceeded("deadline expired")
            timeout = remaining / 1000.0

        # Build provider request
        request_messages = list(messages)
        if system_message:
            request_messages.insert(0, {"role": "system", "content": system_message})

        payload = {
            "model": model or "gpt-4",
            "messages": request_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stop": stop_sequences,
        }

        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice

        try:
            response = await self.client.post(
                self.endpoint,
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()

            # Parse response
            choice = data["choices"][0]
            message = choice["message"]
            
            tool_calls = []
            if "tool_calls" in message:
                for tc in message["tool_calls"]:
                    tool_calls.append(
                        ToolCall(
                            id=tc["id"],
                            type="function",
                            function=ToolCallFunction(
                                name=tc["function"]["name"],
                                arguments=tc["function"]["arguments"],
                            ),
                        )
                    )

            usage = TokenUsage(
                prompt_tokens=data["usage"]["prompt_tokens"],
                completion_tokens=data["usage"]["completion_tokens"],
                total_tokens=data["usage"]["total_tokens"],
            )

            return LLMCompletion(
                text=message.get("content", ""),
                model=model or "gpt-4",
                model_family="gpt-4",
                usage=usage,
                finish_reason=choice["finish_reason"],
                tool_calls=tool_calls,
            )

        except httpx.TimeoutException:
            raise DeadlineExceeded("provider timeout")
        except httpx.HTTPStatusError as e:
            raise self._map_provider_error(e)

    async def _do_stream(
        self,
        *,
        messages: List[Mapping[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        model: Optional[str] = None,
        system_message: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        ctx: Optional[OperationContext] = None,
    ) -> AsyncIterator[LLMChunk]:
        """Stream chat completion with server-sent events."""
        # Implementation depends on provider's streaming API
        # Usually involves setting stream=True and parsing SSE
        pass

    async def _do_count_tokens(
        self,
        text: str,
        model: Optional[str] = None,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> int:
        """Count tokens using provider's tokenizer."""
        # Use tiktoken or similar in production
        return len(text) // 4

    async def _do_health(self, *, ctx: Optional[OperationContext] = None) -> Dict[str, Any]:
        """Check provider health."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.endpoint}/health",
                    timeout=5.0,
                )
                if response.status_code == 200:
                    return {"ok": True, "status": "ok", "server": "hello-llm", "version": "1.0.0"}
                return {"ok": False, "status": "degraded", "server": "hello-llm", "version": "1.0.0"}
        except Exception:
            return {"ok": False, "status": "down", "server": "hello-llm", "version": "1.0.0"}

    def _map_provider_error(self, error: httpx.HTTPStatusError) -> Exception:
        """Map provider errors to canonical Corpus errors."""
        status = error.response.status_code
        if status == 429:
            retry = int(error.response.headers.get("Retry-After", 5)) * 1000
            return ResourceExhausted("rate limit exceeded", retry_after_ms=retry)
        if status == 401:
            return AuthError("invalid API key")
        if status == 400:
            return BadRequest(error.response.text)
        if status >= 500:
            return Unavailable("provider unavailable", retry_after_ms=1000)
        return error
```

---

### 3.3 Vector Adapter (Pinecone/Qdrant Style)

Create `adapters/hello_vector.py`:

```python
from typing import Optional, List, Dict, Any
import httpx

from corpus_sdk.vector.vector_base import (
    BaseVectorAdapter,
    VectorCapabilities,
    QuerySpec,
    BatchQuerySpec,
    UpsertSpec,
    DeleteSpec,
    NamespaceSpec,
    QueryResult,
    UpsertResult,
    DeleteResult,
    NamespaceResult,
    Vector,
    VectorID,
    VectorMatch,
    OperationContext,
    # Canonical errors
    BadRequest,
    ResourceExhausted,
    AuthError,
    Unavailable,
    DeadlineExceeded,
    DimensionMismatch,
    IndexNotReady,
)

class HelloVectorAdapter(BaseVectorAdapter):
    """
    Production-ready vector adapter for a hypothetical vector database.
    
    Demonstrates:
    - REST API integration
    - Namespace management
    - Query with filtering
    - Cache invalidation
    """
    
    def __init__(self, api_key: str, endpoint: Optional[str] = None, mode: str = "standalone"):
        super().__init__(mode=mode)
        self.api_key = api_key
        self.endpoint = endpoint or "https://api.example.com/v1/vectors"
        self.client = httpx.AsyncClient(timeout=30.0)

    async def _do_capabilities(self) -> VectorCapabilities:
        """Advertise vector database capabilities."""
        return VectorCapabilities(
            server="hello-vector",
            protocol="vector/v1.0",  # ✅ REQUIRED
            version="1.0.0",
            max_dimensions=1536,
            supported_metrics=("cosine", "euclidean", "dotproduct"),
            supports_namespaces=True,
            supports_metadata_filtering=True,
            supports_batch_operations=True,
            max_batch_size=100,
            supports_deadline=True,
            text_storage_strategy="metadata",
            supports_batch_queries=True,
        )

    async def _do_query(
        self,
        spec: QuerySpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> QueryResult:
        """Execute vector similarity search."""
        # Deadline propagation
        timeout = None
        if ctx and ctx.deadline_ms:
            remaining = ctx.remaining_ms()
            if remaining <= 0:
                raise DeadlineExceeded("deadline expired")
            timeout = remaining / 1000.0

        try:
            response = await self.client.post(
                f"{self.endpoint}/query",
                json={
                    "namespace": spec.namespace,
                    "vector": spec.vector,
                    "top_k": spec.top_k,
                    "filter": spec.filter,
                    "include_metadata": spec.include_metadata,
                    "include_vectors": spec.include_vectors,
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()

            matches = []
            for m in data["matches"]:
                matches.append(
                    VectorMatch(
                        vector=Vector(
                            id=VectorID(m["id"]),
                            vector=m.get("vector", []),
                            metadata=m.get("metadata"),
                            namespace=spec.namespace,
                            text=m.get("text"),
                        ),
                        score=m["score"],
                        distance=m.get("distance", 0.0),
                    )
                )

            return QueryResult(
                matches=matches,
                query_vector=spec.vector,
                namespace=spec.namespace,
                total_matches=data.get("total", len(matches)),
            )

        except httpx.TimeoutException:
            raise DeadlineExceeded("provider timeout")
        except httpx.HTTPStatusError as e:
            raise self._map_provider_error(e)

    async def _do_batch_query(
        self,
        spec: BatchQuerySpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> List[QueryResult]:
        """Execute multiple queries in one batch."""
        results = []
        for query in spec.queries:
            # Ensure query namespace matches batch namespace
            if query.namespace != spec.namespace:
                query.namespace = spec.namespace  # Canonicalize
            result = await self._do_query(query, ctx=ctx)
            results.append(result)
        return results

    async def _do_upsert(
        self,
        spec: UpsertSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> UpsertResult:
        """Upsert vectors."""
        # Prepare vectors - ensure namespace matches spec
        vectors = []
        for v in spec.vectors:
            # ✅ CRITICAL: Canonicalize namespace
            vectors.append({
                "id": str(v.id),
                "vector": v.vector,
                "metadata": v.metadata,
                "text": v.text,
                "namespace": spec.namespace,  # Force to spec namespace
            })

        try:
            response = await self.client.post(
                f"{self.endpoint}/upsert",
                json={
                    "namespace": spec.namespace,
                    "vectors": vectors,
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            response.raise_for_status()
            data = response.json()

            # ✅ CORRECT: Invalidate cache AFTER successful write
            if data.get("upserted_count", 0) > 0:
                await self._invalidate_namespace_cache(spec.namespace)

            return UpsertResult(
                upserted_count=data.get("upserted_count", 0),
                failed_count=data.get("failed_count", 0),
                failures=data.get("failures", []),
            )

        except httpx.HTTPStatusError as e:
            raise self._map_provider_error(e)

    async def _do_delete(
        self,
        spec: DeleteSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> DeleteResult:
        """Delete vectors by ID or filter."""
        try:
            response = await self.client.post(
                f"{self.endpoint}/delete",
                json={
                    "namespace": spec.namespace,
                    "ids": [str(id) for id in spec.ids] if spec.ids else None,
                    "filter": spec.filter,
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            response.raise_for_status()
            data = response.json()

            # ✅ CORRECT: Invalidate cache AFTER successful delete
            if data.get("deleted_count", 0) > 0:
                await self._invalidate_namespace_cache(spec.namespace)

            return DeleteResult(
                deleted_count=data.get("deleted_count", 0),
                failed_count=data.get("failed_count", 0),
                failures=data.get("failures", []),
            )

        except httpx.HTTPStatusError as e:
            raise self._map_provider_error(e)

    async def _do_create_namespace(
        self,
        spec: NamespaceSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> NamespaceResult:
        """Create a new namespace/collection."""
        try:
            response = await self.client.post(
                f"{self.endpoint}/namespaces",
                json={
                    "namespace": spec.namespace,
                    "dimensions": spec.dimensions,
                    "metric": spec.distance_metric,
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            response.raise_for_status()
            data = response.json()
            return NamespaceResult(
                success=True,
                namespace=spec.namespace,
                details=data,
            )
        except httpx.HTTPStatusError as e:
            raise self._map_provider_error(e)

    async def _do_health(self, *, ctx: Optional[OperationContext] = None) -> Dict[str, Any]:
        """Check vector store health."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.endpoint}/health",
                    timeout=5.0,
                )
                if response.status_code == 200:
                    return {"ok": True, "status": "ok", "server": "hello-vector", "version": "1.0.0"}
                return {"ok": False, "status": "degraded", "server": "hello-vector", "version": "1.0.0"}
        except Exception:
            return {"ok": False, "status": "down", "server": "hello-vector", "version": "1.0.0"}

    def _map_provider_error(self, error: httpx.HTTPStatusError) -> Exception:
        """Map provider HTTP errors to canonical Corpus errors."""
        status = error.response.status_code
        if status == 429:
            retry = int(error.response.headers.get("Retry-After", 5)) * 1000
            return ResourceExhausted("rate limit exceeded", retry_after_ms=retry)
        if status == 401:
            return AuthError("invalid API key")
        if status == 400:
            return BadRequest(error.response.text)
        if status >= 500:
            return Unavailable("provider unavailable", retry_after_ms=1000)
        return error
```

---

### 3.4 Graph Adapter (Neo4j/JanusGraph Style)

Create `adapters/hello_graph.py`:

```python
from typing import AsyncIterator, Optional, List, Dict, Any, Union, Mapping
import httpx

from corpus_sdk.graph.graph_base import (
    BaseGraphAdapter,
    GraphCapabilities,
    GraphQuerySpec,
    UpsertNodesSpec,
    UpsertEdgesSpec,
    DeleteNodesSpec,
    DeleteEdgesSpec,
    BatchOperation,
    GraphTraversalSpec,
    QueryResult,
    QueryChunk,
    UpsertResult,
    DeleteResult,
    BatchResult,
    TraversalResult,
    GraphSchema,
    Node,
    Edge,
    GraphID,
    OperationContext,
    # Canonical errors
    BadRequest,
    AuthError,
    ResourceExhausted,
    Unavailable,
    DeadlineExceeded,
    NotSupported,
)

class HelloGraphAdapter(BaseGraphAdapter):
    """
    Production-ready graph adapter for a hypothetical graph database.
    
    Demonstrates:
    - Query execution
    - Node/edge operations
    - Transactions with cache invalidation
    """
    
    def __init__(self, api_key: str, endpoint: Optional[str] = None, mode: str = "standalone"):
        super().__init__(mode=mode)
        self.api_key = api_key
        self.endpoint = endpoint or "https://api.example.com/v1/graph"
        self.client = httpx.AsyncClient(timeout=30.0)

    async def _do_capabilities(self) -> GraphCapabilities:
        """Advertise graph database capabilities."""
        return GraphCapabilities(
            server="hello-graph",
            protocol="graph/v1.0",  # ✅ REQUIRED
            version="1.0.0",
            supports_stream_query=True,
            supported_query_dialects=("cypher", "gremlin"),
            supports_namespaces=True,
            supports_property_filters=True,
            supports_bulk_vertices=True,
            supports_batch=True,
            supports_schema=True,
            idempotent_writes=False,
            supports_multi_tenant=True,
            supports_deadline=True,
            max_batch_ops=100,
            supports_transaction=True,
            supports_traversal=True,
            max_traversal_depth=10,
            supports_path_queries=True,
        )

    async def _do_query(
        self,
        spec: GraphQuerySpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> QueryResult:
        """Execute graph query."""
        # Deadline propagation
        timeout = None
        if ctx and ctx.deadline_ms:
            remaining = ctx.remaining_ms()
            if remaining <= 0:
                raise DeadlineExceeded("deadline expired")
            timeout = remaining / 1000.0

        # Validate dialect
        caps = await self._do_capabilities()
        if spec.dialect and spec.dialect not in caps.supported_query_dialects:
            raise NotSupported(f"dialect '{spec.dialect}' not supported")

        try:
            response = await self.client.post(
                f"{self.endpoint}/query",
                json={
                    "text": spec.text,
                    "dialect": spec.dialect,
                    "params": spec.params,
                    "namespace": spec.namespace,
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()

            return QueryResult(
                records=data.get("records", []),
                summary=data.get("summary", {}),
                dialect=spec.dialect,
                namespace=spec.namespace,
            )

        except httpx.TimeoutException:
            raise DeadlineExceeded("provider timeout")
        except httpx.HTTPStatusError as e:
            raise self._map_provider_error(e)

    async def _do_stream_query(
        self,
        spec: GraphQuerySpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> AsyncIterator[QueryChunk]:
        """Stream query results."""
        # Implementation depends on provider's streaming API
        pass

    async def _do_upsert_nodes(
        self,
        spec: UpsertNodesSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> UpsertResult:
        """Upsert nodes."""
        try:
            response = await self.client.post(
                f"{self.endpoint}/nodes",
                json={
                    "namespace": spec.namespace,
                    "nodes": [
                        {
                            "id": str(n.id),
                            "labels": list(n.labels),
                            "properties": n.properties,
                        }
                        for n in spec.nodes
                    ],
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            response.raise_for_status()
            data = response.json()
            return UpsertResult(
                upserted_count=data.get("upserted_count", len(spec.nodes)),
                failed_count=data.get("failed_count", 0),
                failures=data.get("failures", []),
            )
        except httpx.HTTPStatusError as e:
            raise self._map_provider_error(e)

    async def _do_upsert_edges(
        self,
        spec: UpsertEdgesSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> UpsertResult:
        """Upsert edges."""
        try:
            response = await self.client.post(
                f"{self.endpoint}/edges",
                json={
                    "namespace": spec.namespace,
                    "edges": [
                        {
                            "id": str(e.id),
                            "src": str(e.src),
                            "dst": str(e.dst),
                            "label": e.label,
                            "properties": e.properties,
                        }
                        for e in spec.edges
                    ],
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            response.raise_for_status()
            data = response.json()
            return UpsertResult(
                upserted_count=data.get("upserted_count", len(spec.edges)),
                failed_count=data.get("failed_count", 0),
                failures=data.get("failures", []),
            )
        except httpx.HTTPStatusError as e:
            raise self._map_provider_error(e)

    async def _do_transaction(
        self,
        operations: List[BatchOperation],
        *,
        ctx: Optional[OperationContext] = None,
    ) -> BatchResult:
        """Execute atomic transaction."""
        try:
            response = await self.client.post(
                f"{self.endpoint}/transaction",
                json={
                    "operations": [
                        {"op": op.op, "args": op.args}
                        for op in operations
                    ],
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            response.raise_for_status()
            data = response.json()

            # ✅ CORRECT: Invalidate cache AFTER successful commit
            if data.get("success", False):
                # Collect affected namespaces
                namespaces = set()
                for op in operations:
                    ns = op.args.get("namespace")
                    if ns:
                        namespaces.add(ns)
                for ns in namespaces:
                    await self._invalidate_namespace_cache(ns)

            return BatchResult(
                results=data.get("results", []),
                success=data.get("success", False),
                error=data.get("error"),
                transaction_id=data.get("transaction_id"),
            )

        except httpx.HTTPStatusError as e:
            raise self._map_provider_error(e)

    async def _do_health(self, *, ctx: Optional[OperationContext] = None) -> Dict[str, Any]:
        """Check graph database health."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.endpoint}/health",
                    timeout=5.0,
                )
                if response.status_code == 200:
                    return {"ok": True, "status": "ok", "server": "hello-graph", "version": "1.0.0"}
                return {"ok": False, "status": "degraded", "server": "hello-graph", "version": "1.0.0"}
        except Exception:
            return {"ok": False, "status": "down", "server": "hello-graph", "version": "1.0.0"}

    def _map_provider_error(self, error: httpx.HTTPStatusError) -> Exception:
        """Map provider HTTP errors to canonical Corpus errors."""
        status = error.response.status_code
        if status == 429:
            retry = int(error.response.headers.get("Retry-After", 5)) * 1000
            return ResourceExhausted("rate limit exceeded", retry_after_ms=retry)
        if status == 401:
            return AuthError("invalid API key")
        if status == 400:
            return BadRequest(error.response.text)
        if status >= 500:
            return Unavailable("provider unavailable", retry_after_ms=1000)
        return error
```

---

## 4. Running Certification Tests

Now run the official certification suite against your adapter. **Choose the section that matches your protocol.**

### 4.1 Embedding Certification

```bash
# Test embedding protocol only
export CORPUS_ADAPTER=adapters.hello_embedding:HelloEmbeddingAdapter
pytest $(python -c "import corpus_sdk; print(corpus_sdk.__path__[0])")/tests/embedding/ -v

# Incremental test order (if you want to run specific files)
pytest $(python -c "import corpus_sdk; print(corpus_sdk.__path__[0])")/tests/embedding/test_capabilities.py -v
pytest $(python -c "import corpus_sdk; print(corpus_sdk.__path__[0])")/tests/embedding/test_embed.py -v
pytest $(python -c "import corpus_sdk; print(corpus_sdk.__path__[0])")/tests/embedding/test_streaming.py -v
pytest $(python -c "import corpus_sdk; print(corpus_sdk.__path__[0])")/tests/embedding/test_batch.py -v
pytest $(python -c "import corpus_sdk; print(corpus_sdk.__path__[0])")/tests/embedding/test_deadlines.py -v
pytest $(python -c "import corpus_sdk; print(corpus_sdk.__path__[0])")/tests/embedding/test_errors.py -v
```

### 4.2 LLM Certification

```bash
export CORPUS_ADAPTER=adapters.hello_llm:HelloLLMAdapter
pytest $(python -c "import corpus_sdk; print(corpus_sdk.__path__[0])")/tests/llm/ -v
```

### 4.3 Vector Certification

```bash
export CORPUS_ADAPTER=adapters.hello_vector:HelloVectorAdapter
pytest $(python -c "import corpus_sdk; print(corpus_sdk.__path__[0])")/tests/vector/ -v
```

### 4.4 Graph Certification

```bash
export CORPUS_ADAPTER=adapters.hello_graph:HelloGraphAdapter
pytest $(python -c "import corpus_sdk; print(corpus_sdk.__path__[0])")/tests/graph/ -v
```

---

## 5. Understanding Certification Results

The certification suite provides **tiered scoring**. When you run the full suite, look for this summary:

```
================================================================================
CORPUS PROTOCOL SUITE - GOLD CERTIFIED
🔌 Adapter: adapters.hello_embedding:HelloEmbeddingAdapter | ⚖️ Strict: off

Protocol & Framework Conformance Status (scored / collected):
  ✅ PASS Embedding Protocol V1.0: Gold (135/135 scored; 150 collected)

🎯 Status: Ready for production deployment
⏱️ Completed in 1.2s
```

### Certification Tiers

| Tier | Score | Meaning | Production Ready? |
|------|-------|---------|------------------|
| 🥇 **Gold** | 100% | Perfect protocol conformance | ✅ Yes |
| 🥈 **Silver** | ≥80% | Integration testing ready | ⚠️ No |
| 🔬 **Development** | ≥50% | Early implementation | ❌ No |
| ❌ **None** | <50% | Not yet functional | ❌ No |

**Your goal:** Gold certification for your protocol.

### Reading Failure Output

When tests fail, the certification suite provides detailed guidance:

```
--------------------------------------------------
🟥 FAILURES & ERRORS
Embedding Protocol V1.0:
  ❌ Failure Wire Contract & Routing: 2 issue(s)
      Specification: §4.1 Wire-First Canonical Form
      Test: test_wire_envelope_validation
      Quick fix: Wire envelope missing required fields per §4.1
```

**Each failure includes:**
- **Specification section** (§4.1, §7.2, etc.)
- **Quick fix** - Exactly what to change

**Do not guess.** The error guidance is authoritative.

For complete certification requirements, see [`CONFORMANCE_GUIDE.md`](CONFORMANCE_GUIDE.md).

---

## 6. What to Read Next

**You now have a Gold-certified adapter.** Choose your path:

| Guide | Purpose | When to Read |
|-------|---------|--------------|
| **[DEPLOYMENT.md](DEPLOYMENT.md)** | Expose your adapter as an HTTP service (FastAPI, Lambda, etc.) | You need a standalone microservice |
| **[IMPLEMENTATION.md](IMPLEMENTATION.md)** | Deep dive on `_do_*` semantics and advanced features | You need custom deadline policies, circuit breakers, etc. |
| **[CONFORMANCE_GUIDE.md](CONFORMANCE_GUIDE.md)** | Debugging certification failures | Tests are failing and you're stuck |
| **[ADAPTER_RECIPES.md](ADAPTER_RECIPES.md)** | Multi-cloud and RAG scenarios | You're building complex pipelines |

**The conformance tests in [`../../tests/`](../../tests/) are the source of truth.** When this document and the tests disagree, **the tests are correct.**

---

## 7. Protocol-Specific Requirements & Pitfalls

### 7.1 Embedding Protocol

```python
# ✅ REQUIRED: protocol="embedding/v1.0" in capabilities
return EmbeddingCapabilities(protocol="embedding/v1.0")

# ✅ REQUIRED: idempotent_writes=True
return EmbeddingCapabilities(idempotent_writes=True)

# ✅ REQUIRED: batch field name "failures" (not "failed_texts")
return BatchEmbedResult(failures=[])

# ✅ REQUIRED: index field for batch correlation
EmbeddingVector(index=idx)

# ✅ REQUIRED: idempotency key deduplication (24h)
if ctx.idempotency_key:
    cached = await redis.get(f"idem:{ctx.tenant}:{ctx.idempotency_key}")
    if cached: return cached
```

### 7.2 LLM Protocol

```python
# ✅ REQUIRED: protocol="llm/v1.0" in capabilities
return LLMCapabilities(protocol="llm/v1.0")

# ✅ REQUIRED: model_family must be set
return LLMCapabilities(model_family="gpt-4")

# ⚠️ CRITICAL: Never implement tool execution
if chunk.tool_calls:
    yield chunk  # Pass through, don't execute!

# ✅ REQUIRED: streaming with usage in final chunk
LLMChunk(is_final=True, usage_so_far=TokenUsage(...))

# ✅ REQUIRED: ToolCall with generated ID
ToolCall(id=f"call_{secrets.token_hex(8)}", ...)

# ❌ NEVER: system_message if not supported
if not caps.supports_system_message:
    raise NotSupported("system_message not supported")
```

### 7.3 Vector Protocol

```python
# ✅ REQUIRED: protocol="vector/v1.0" in capabilities
return VectorCapabilities(protocol="vector/v1.0")

# ⚠️ CRITICAL: Namespace footgun prevention
if v.namespace is not None and v.namespace != spec.namespace:
    raise BadRequest("vector.namespace must match spec.namespace")

# ✅ REQUIRED: Canonicalize to spec namespace
v.namespace = spec.namespace  # Always set to spec.namespace

# ⚠️ CRITICAL: Cache invalidation AFTER successful write
result = await self._do_upsert(spec, ctx)  # Success!
await self._invalidate_namespace_cache(spec.namespace)  # Then invalidate

# ❌ WRONG: Invalidate before commit (cache will be stale if commit fails)
await self._invalidate_namespace_cache(namespace)  # Too early!
await client.upsert(vectors)  # If this fails, cache is now wrong
```

### 7.4 Graph Protocol

```python
# ✅ REQUIRED: protocol="graph/v1.0" in capabilities
return GraphCapabilities(protocol="graph/v1.0")

# ⚠️ CRITICAL: Transaction cache invalidation
# ✅ CORRECT: Invalidate after successful commit
result = await txn.commit()  # Atomic commit succeeds
if result.success:
    await self._invalidate_namespace_cache(namespace)

# ❌ WRONG: Invalidate during transaction (commit may fail)
await self._cache.invalidate_pattern(...)  # Premature!
await txn.commit()  # If this fails, cache is now inconsistent

# ✅ REQUIRED: Batch operation success detection
if self._batch_op_succeeded(op, batch_result, idx):
    # Only invalidate if this op actually changed data
    await self._invalidate_namespace_cache(namespace)

# ✅ REQUIRED: Query dialect validation
if spec.dialect and caps.supported_query_dialects:
    if spec.dialect not in caps.supported_query_dialects:
        raise NotSupported(f"dialect '{spec.dialect}' not supported")
```

---

## 8. Certification Checklist

### Universal Requirements (All Protocols)

- [ ] **REQUIRED:** Constructor accepts `endpoint=None`
- [ ] **REQUIRED:** `_do_capabilities()` declares `protocol="{component}/v1.0"`
- [ ] **REQUIRED:** `ctx.remaining_ms()` used in all `_do_*` methods
- [ ] **REQUIRED:** No raw tenant IDs in logs/metrics (use tenant hashing)
- [ ] **REQUIRED:** Gold certification achieved: `pytest tests/{protocol}/ -v` shows 100% pass
- [ ] **RECOMMENDED:** `_do_get_stats()` implemented for service observability
- [ ] **RECOMMENDED:** Health endpoint returns graded `status: "ok"|"degraded"|"down"`

### Embedding-Specific Checklist

- [ ] **REQUIRED:** `_do_capabilities()` declares `idempotent_writes=True`
- [ ] **REQUIRED:** Batch operations use field name `failures` (not `failed_texts`)
- [ ] **REQUIRED:** Batch success items include `index` field for correlation
- [ ] **REQUIRED:** Idempotency keys deduplicated for ≥24 hours
- [ ] **REQUIRED:** `_do_count_tokens()` implemented
- [ ] **RECOMMENDED:** Streaming implemented (if `supports_streaming=True`)

### LLM-Specific Checklist

- [ ] **REQUIRED:** `_do_capabilities()` declares `model_family` (not just `model`)
- [ ] **REQUIRED:** Never implement tool execution - only pass through tool calls
- [ ] **REQUIRED:** Tool calls include generated IDs (`secrets.token_hex()`)
- [ ] **REQUIRED:** Streaming includes `usage_so_far` in final chunk
- [ ] **REQUIRED:** `supports_system_message` accurately reflects capability
- [ ] **REQUIRED:** `_do_count_tokens()` implemented
- [ ] **RECOMMENDED:** Support for `stop_sequences`, `frequency_penalty`, `presence_penalty`

### Vector-Specific Checklist

- [ ] **REQUIRED:** Namespace canonicalization enforced (vector.namespace == spec.namespace)
- [ ] **REQUIRED:** Cache invalidation performed AFTER successful writes
- [ ] **REQUIRED:** `max_dimensions` validated on upsert and query
- [ ] **REQUIRED:** Batch query support (if `supports_batch_queries=True`)
- [ ] **RECOMMENDED:** Metadata filtering support with `supports_metadata_filtering=True`

### Graph-Specific Checklist

- [ ] **REQUIRED:** Cache invalidation performed ONLY after successful transaction commit
- [ ] **REQUIRED:** Query dialect validation against `supported_query_dialects`
- [ ] **REQUIRED:** Transaction support requires atomic batch operations
- [ ] **REQUIRED:** Batch operation success detection for targeted invalidation
- [ ] **REQUIRED:** `supports_transaction` accurately reflects capability
- [ ] **RECOMMENDED:** Traversal support with `supports_traversal=True`

---

## Appendix A: Common Pitfalls by Component

### Embedding

```python
# ❌ WRONG: Missing REQUIRED protocol field
return EmbeddingCapabilities(
    server="hello-embedding",
    version="1.0.0",
    # missing protocol="embedding/v1.0"  # WILL FAIL CERTIFICATION
)

# ❌ WRONG: Wrong batch field name
return BatchEmbedResult(
    embeddings=embeddings,
    failed_texts=failures,  # ❌ MUST be "failures"
)

# ❌ WRONG: Assuming batch results align 1:1 with inputs
for i, text in enumerate(spec.texts):
    assert result.embeddings[i].text == text  # MAY FAIL!

# ✅ CORRECT: Use index field for correlation
for emb in result.embeddings:
    original_text = spec.texts[emb.index]  # SAFE
```

### LLM

```python
# ❌ WRONG: Missing REQUIRED protocol field
return LLMCapabilities(
    server="hello-llm",
    version="1.0.0",
    # missing protocol="llm/v1.0"  # WILL FAIL CERTIFICATION
)

# ❌ WRONG: Missing model_family
return LLMCapabilities(
    protocol="llm/v1.0",
    # missing model_family  # WILL FAIL CERTIFICATION
)

# ❌ WRONG: Implementing tool execution in adapter
if tool_calls:
    result = await execute_tools(tool_calls)  # NO - that's orchestration!

# ✅ CORRECT: Just pass through tool calls
return LLMCompletion(tool_calls=tool_calls)  # Router's job to execute
```

### Vector

```python
# ❌ WRONG: Missing REQUIRED protocol field
return VectorCapabilities(
    server="hello-vector",
    version="1.0.0",
    # missing protocol="vector/v1.0"  # WILL FAIL CERTIFICATION
)

# ❌ WRONG: Ignoring namespace mismatch
vector = Vector(id="123", vector=[...], namespace="user-space")
spec = UpsertSpec(vectors=[vector], namespace="default")  # WILL FAIL!

# ✅ CORRECT: Canonicalize to spec namespace
vector.namespace = spec.namespace  # Must match

# ❌ WRONG: Cache invalidation before write
await self._invalidate_namespace_cache(spec.namespace)  # Too early!
result = await self._do_upsert(spec, ctx)  # If this fails, cache is stale

# ✅ CORRECT: Invalidate after successful write
result = await self._do_upsert(spec, ctx)
if result.upserted_count > 0:
    await self._invalidate_namespace_cache(spec.namespace)  # SAFE
```

### Graph

```python
# ❌ WRONG: Missing REQUIRED protocol field
return GraphCapabilities(
    server="hello-graph",
    version="1.0.0",
    # missing protocol="graph/v1.0"  # WILL FAIL CERTIFICATION
)

# ❌ WRONG: Cache invalidation before commit
await self._cache.invalidate_pattern(...)
await txn.commit()  # If commit fails, cache is stale!

# ✅ CORRECT: Invalidate after successful commit
await txn.commit()
await self._invalidate_namespace_cache(namespace)  # SAFE

# ❌ WRONG: Not validating dialects
spec = GraphQuerySpec(dialect="cypher")
caps = await self.capabilities()
if "cypher" not in caps.supported_query_dialects:
    # MISSING: Should raise NotSupported
    pass
```

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **Adapter** | A class that implements `_do_*` hooks to connect a provider to Corpus Protocol |
| **Base Class** | `BaseEmbeddingAdapter`, `BaseLLMAdapter`, `BaseVectorAdapter`, `BaseGraphAdapter` |
| **Certification Suite** | The conformance tests in `tests/embedding/`, `tests/llm/`, etc. |
| **Gold Certification** | 100% pass rate in a single protocol |
| **Wire Envelope** | The JSON `{op, ctx, args}` structure all Corpus services speak |
| **Protocol Field** | REQUIRED field in capabilities: `"protocol": "{component}/v1.0"` |
| **Idempotent Writes** | REQUIRED capability for embedding: `idempotent_writes: true` |
| **Failures Field** | REQUIRED field name for embedding batch errors (not `failed_texts`) |
| **Model Family** | REQUIRED field in LLM capabilities: `model_family` |
| **Namespace Canonicalization** | REQUIRED behavior for Vector: enforce namespace match |
| **Transaction Atomicity** | REQUIRED for Graph: all or nothing |
| **CORPUS_ADAPTER** | Environment variable: `module:ClassName` for dynamic loading |

---

## Appendix C: Debugging & Troubleshooting

### Enable Full Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("corpus_sdk").setLevel(logging.DEBUG)
```

### Common Errors & Fixes (All Protocols)

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| `AdapterValidationError: Failed to instantiate adapter` | Constructor doesn't accept `endpoint=None` | Add `endpoint=None` to `__init__` |
| `capabilities missing required field: protocol` | Missing `protocol="{component}/v1.0"` | Add protocol field to capabilities |
| `DEADLINE_EXCEEDED not raised` | Deadline not checked before provider call | Call `ctx.remaining_ms()` and raise if 0 |
| `retry_after_ms missing from 429 responses` | Error mapping incomplete | Map provider rate limits with `retry_after_ms` |

### Protocol-Specific Errors

**Embedding:**
| Error | Fix |
|-------|-----|
| `capabilities missing required field: idempotent_writes` | Add `idempotent_writes=True` |
| `Batch result missing field: failures` | Rename `failed_texts` → `failures` |
| `Batch success missing index` | Set `index=idx` on EmbeddingVector |
| `Idempotency test failed` | Implement idempotency cache with 24h TTL |

**LLM:**
| Error | Fix |
|-------|-----|
| `capabilities missing required field: model_family` | Add `model_family` to capabilities |
| `Tool execution detected` | Remove tool execution logic - only pass through |
| `Missing tool call ID` | Generate IDs with `secrets.token_hex(8)` |
| `Missing usage_so_far in final chunk` | Add TokenUsage to final streaming chunk |

**Vector:**
| Error | Fix |
|-------|-----|
| `Namespace mismatch` | Canonicalize vector.namespace = spec.namespace |
| `Cache invalidation order` | Move invalidation AFTER successful write |
| `Batch query not implemented` | Implement `_do_batch_query()` if `supports_batch_queries=True` |

**Graph:**
| Error | Fix |
|-------|-----|
| `Cache invalidation before commit` | Move invalidation AFTER successful commit |
| `Dialect not validated` | Check `spec.dialect` against `caps.supported_query_dialects` |
| `Transaction not atomic` | Ensure all operations in transaction commit or rollback together |

### Debugging Test Failures

```bash
# Run with full traceback
pytest tests/{protocol}/test_file.py -v --tb=long

# Stop on first failure
pytest tests/{protocol}/ -v --maxfail=1

# Run only tests that failed last time
pytest tests/{protocol}/ -v --lf

# See which tests are available
pytest tests/{protocol}/ --collect-only
```

---

**Maintainers:** Corpus SDK Team  
**Last Updated:** 2026-02-13  
**Scope:** Complete adapter authoring reference for all Corpus Protocols v1.0 (Embedding, LLM, Vector, Graph).

**The conformance tests in [`../../tests/`](../../tests/) are the source of truth.** When this document and the tests disagree, **the tests are correct.**