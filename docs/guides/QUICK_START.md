# Corpus OS Quickstart

**Build Production-Ready Corpus Protocol Adapters in Minutes**

**Table of Contents**
- [0. Mental Model (What You're Actually Building)](#0-mental-model-what-youre-actually-building)
- [1. Prerequisites & Setup](#1-prerequisites--setup)
- [2. Conformance-First Development (The Right Way)](#2-conformance-first-development-the-right-way)
- [3. Hello World Embedding Adapter (Complete)](#3-hello-world-embedding-adapter-complete)
- [4. Test-Driven Iteration](#4-test-driven-iteration)
- [5. Expose It Over HTTP](#5-expose-it-over-http)
- [6. Other Protocol Variants (LLM/Vector/Graph)](#6-other-protocol-variants-llmvectorgraph)
- [7. Production Readiness](#7-production-readiness)
- [8. Full Conformance Suite](#8-full-conformance-suite)
- [9. What to Read Next](#9-what-to-read-next)
- [10. Adapter Launch Checklist](#10-adapter-launch-checklist)
- [Appendix A: Common Pitfalls by Component](#appendix-a-common-pitfalls-by-component)
- [Appendix B: Glossary](#appendix-b-glossary)
- [Appendix C: Debugging & Troubleshooting](#appendix-c-debugging--troubleshooting)

---

> **Goal:** Get a real, production-ready adapter speaking the Corpus Protocol v1.0 in under 15 minutes.  
> **Audience:** SDK / adapter authors (LLM, Embedding, Vector, Graph).  
> **You'll build:** A complete Embedding adapter with streaming, batch operations, error mapping, and full conformance—then adapt the pattern for LLM/Vector/Graph.

**By the end of this guide you will have:**
- ✅ A fully tested adapter implementation
- ✅ Streaming and batch operation support
- ✅ Proper error mapping and deadline propagation
- ✅ Cache invalidation (where applicable)
- ✅ Passing conformance tests
- ✅ A reusable pattern for all four protocol variants

---

## 0. Mental Model (What You're Actually Building)

An **adapter** is a thin translation layer that:

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Your Provider  │◄────┤  YourAdapter │◄────┤  Corpus Base    │
│  (OpenAI, etc.) │     │  (_do_* hooks)│     │  (infrastructure)│
└─────────────────┘     └──────────────┘     └─────────────────┘
                                                      │
                                                      ▼
                                            ┌─────────────────┐
                                            │  WireHandler    │
                                            │  (JSON envelope)│
                                            └─────────────────┘
```

**You implement only:**
- `_do_capabilities()` - What your adapter supports
- `_do_embed()` / `_do_complete()` / etc. - Core operation
- `_do_stream_*()` - Streaming (if supported)
- `_do_health()` - Liveness check

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

---

## 1. Prerequisites & Setup

### Requirements
- Python 3.10+ (async/await, `asyncio`)
- `corpus-sdk` ≥ 1.0.0
- Your favorite async HTTP client (aiohttp, httpx) for provider calls

### Installation

```bash
pip install corpus-sdk
# Optional but recommended for Vector docstore
pip install msgpack  # for RedisDocStore
```

### Project Layout (Recommended)

```
your-adapter-repo/
├── adapters/
│   ├── __init__.py
│   ├── hello_embedding.py        # Your adapter
│   └── conftest.py              # Pytest fixtures
├── services/
│   └── embedding_service.py     # HTTP/gRPC entrypoint
├── tests/
│   ├── test_adapter.py          # Your unit tests
│   └── conformance/             # Copied from SDK
├── spec/                        # Protocol documentation
│   ├── IMPLEMENTATION.md
│   └── BEHAVIORAL_CONFORMANCE.md
└── conformance/                 # Conformance test runner
    └── run_conformance.py
```

### Enable Debug Logging (Development)

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("corpus_sdk").setLevel(logging.DEBUG)
# See cache hits/misses, deadline decisions, wire envelopes
```

---

## 2. Conformance-First Development (The Right Way)

**Stop writing code before you have a failing test.**  

The Corpus SDK ships with *off-the-shelf* conformance tests. Your job is to make them pass, one by one.

### Step 1: Copy the Conformance Tests

```bash
# Copy the official conformance suite into your test directory
cp -r $(python -c "import corpus_sdk; print(corpus_sdk.__path__[0])")/conformance/embedding ./tests/conformance
```

### Step 2: Write a Minimal Test Fixture

`tests/conftest.py`:
```python
import pytest
from adapters.hello_embedding import HelloEmbeddingAdapter

@pytest.fixture
def adapter():
    """Return an UNIMPLEMENTED adapter - tests should fail!"""
    return HelloEmbeddingAdapter(mode="thin")  # Start with thin mode
```

### Step 3: Run a Single Test and Watch It Fail

```bash
pytest tests/conformance/test_capabilities.py -v -k test_capabilities_basic

# Expected: NotImplementedError from _do_capabilities
# This is GOOD - you now have a target.
```

**Why this matters:** You're not guessing what "correct" means. The conformance tests are the normative specification. When they all pass, your adapter is done.

---

## 3. Hello World Embedding Adapter (Complete)

This is the **minimum viable implementation** that passes all Embedding conformance tests, including:

- ✅ Unary embedding
- ✅ Streaming embedding
- ✅ Batch embedding with partial success
- ✅ Token counting (when supported)
- ✅ Deadline propagation
- ✅ Error mapping

Create `adapters/hello_embedding.py`:

```python
import asyncio
from typing import AsyncIterator, Optional, List, Dict, Any
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
    # Canonical errors - USE THESE, don't raise raw exceptions
    BadRequest,
    ResourceExhausted,
    AuthError,
    Unavailable,
    NotSupported,
    DeadlineExceeded,
)


class HelloEmbeddingAdapter(BaseEmbeddingAdapter):
    """
    Production-ready demo adapter.

    Features:
        - 8-dimensional embeddings (text length as first dimension)
        - Streaming support with chunked responses
        - Batch operations with partial success
        - Token counting (fake, but present)
        - Full deadline propagation
        - Complete error mapping
    """

    async def _do_capabilities(self) -> EmbeddingCapabilities:
        """Advertise what this adapter can do - BE HONEST."""
        return EmbeddingCapabilities(
            server="hello-embedding",
            version="1.0.0",
            supported_models=("hello-1", "hello-2"),  # Explicit list
            max_batch_size=10,
            max_text_length=1000,
            max_dimensions=8,
            supports_normalization=False,      # We don't normalize
            supports_truncation=True,          # We can truncate
            supports_token_counting=True,      # We provide fake counts
            supports_streaming=True,           # We support streaming!
            supports_batch_embedding=True,     # Native batch support
            normalizes_at_source=False,
            truncation_mode="base",            # Let base handle truncation
        )

    # ------------------------------------------------------------------------
    # SINGLE EMBEDDING (Unary)
    # ------------------------------------------------------------------------

    async def _do_embed(
        self,
        spec: EmbedSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> EmbedResult:
        """
        Generate embedding for a single text.

        ⚠️ CRITICAL: Always use ctx.remaining_ms() for provider timeouts.
        ⚠️ CRITICAL: Never swallow deadline errors - let them propagate.
        """
        # 1. Provider-specific client call with deadline
        try:
            # Calculate timeout from context
            timeout = None
            if ctx and ctx.deadline_ms:
                remaining = ctx.remaining_ms()
                if remaining is not None and remaining > 0:
                    timeout = remaining / 1000.0
                elif remaining == 0:
                    raise DeadlineExceeded("deadline already expired")

            # Simulate provider call (replace with your actual client)
            await asyncio.sleep(0.01)  # Fake latency
            
            # Our "model": first dimension = text length, rest zeros
            vec = [float(len(spec.text))] + [0.0] * 7

            embedding = EmbeddingVector(
                vector=vec,
                text=spec.text,
                model=spec.model,
                dimensions=len(vec),
                # index=None - not needed for single embed
            )

            return EmbedResult(
                embedding=embedding,
                model=spec.model,
                text=spec.text,
                tokens_used=len(spec.text) // 4,  # Fake token count
                truncated=False,  # Base will set this if truncation occurred
            )

        # 2. MAP PROVIDER ERRORS TO CANONICAL ONES
        except asyncio.TimeoutError:
            # Provider timeout -> DeadlineExceeded
            raise DeadlineExceeded("provider timeout")
        except Exception as e:
            # Use the mapping helper (defined below)
            raise self._map_provider_error(e)

    # ------------------------------------------------------------------------
    # STREAMING EMBEDDING (Required for conformance)
    # ------------------------------------------------------------------------

    async def _do_stream_embed(
        self,
        spec: EmbedSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> AsyncIterator[EmbedChunk]:
        """
        Stream embedding generation in chunks.

        Rules:
            - Yield EmbedChunk objects
            - Last chunk MUST have is_final=True
            - You MAY include usage_so_far in final chunk
            - Never yield an error - raise it instead
        """
        # Simulate streaming by sending vector in 4 chunks
        base_vec = [float(len(spec.text))] + [0.0] * 7
        
        # Chunk 1: first 2 dimensions
        yield EmbedChunk(
            embeddings=[
                EmbeddingVector(
                    vector=base_vec[:2] + [0.0] * 6,
                    text=spec.text,
                    model=spec.model,
                    dimensions=8,
                )
            ],
            is_final=False,
        )
        await asyncio.sleep(0.005)
        
        # Chunk 2: next 2 dimensions
        yield EmbedChunk(
            embeddings=[
                EmbeddingVector(
                    vector=base_vec[:4] + [0.0] * 4,
                    text=spec.text,
                    model=spec.model,
                    dimensions=8,
                )
            ],
            is_final=False,
        )
        await asyncio.sleep(0.005)
        
        # Final chunk: complete vector + usage
        yield EmbedChunk(
            embeddings=[
                EmbeddingVector(
                    vector=base_vec,
                    text=spec.text,
                    model=spec.model,
                    dimensions=8,
                )
            ],
            is_final=True,
            usage={"prompt_tokens": len(spec.text) // 4, "total_tokens": len(spec.text) // 4},
            model=spec.model,
        )

    # ------------------------------------------------------------------------
    # BATCH EMBEDDING (With Partial Success)
    # ------------------------------------------------------------------------

    async def _do_embed_batch(
        self,
        spec: BatchEmbedSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> BatchEmbedResult:
        """
        Batch embedding with partial success semantics.

        ⚠️ CRITICAL: 
            - Do NOT assume 1:1 alignment between input texts and output embeddings
            - Set EmbeddingVector.index on each success for correlation
            - Populate failed_texts for failures
        """
        embeddings: List[EmbeddingVector] = []
        failures: List[Dict[str, Any]] = []

        for idx, text in enumerate(spec.texts):
            try:
                # Validate each item (provider-specific)
                if not text or len(text) > 1000:
                    raise BadRequest(f"text too long or empty")

                # Generate embedding
                vec = [float(len(text))] + [0.0] * 7
                
                embeddings.append(
                    EmbeddingVector(
                        vector=vec,
                        text=text,
                        model=spec.model,
                        dimensions=8,
                        index=idx,  # ⚠️ CRITICAL: Enable correlation
                    )
                )
            except Exception as e:
                # Record failure with stable error code and original index
                failures.append({
                    "index": idx,
                    "text": text,
                    "error": type(e).__name__,
                    "code": getattr(e, "code", "UNKNOWN"),
                    "message": str(e),
                })

        return BatchEmbedResult(
            embeddings=embeddings,  # May be shorter than input texts!
            model=spec.model,
            total_texts=len(spec.texts),
            total_tokens=sum(len(t) // 4 for t in spec.texts),
            failed_texts=failures,
        )

    # ------------------------------------------------------------------------
    # TOKEN COUNTING (Even if Fake)
    # ------------------------------------------------------------------------

    async def _do_count_tokens(
        self,
        text: str,
        model: str,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> int:
        """
        Token counting implementation.

        Even if your provider doesn't support this, implement a reasonable
        approximation. Many routers depend on this for cost estimation.
        """
        # Simple approximation: 4 chars ≈ 1 token
        return len(text) // 4

    # ------------------------------------------------------------------------
    # HEALTH CHECK
    # ------------------------------------------------------------------------

    async def _do_health(self, *, ctx: Optional[OperationContext] = None) -> Dict[str, Any]:
        """Lightweight liveness probe."""
        return {
            "ok": True,
            "server": "hello-embedding",
            "version": "1.0.0",
            "status": "healthy",
        }

    # ------------------------------------------------------------------------
    # ERROR MAPPING (Centralized)
    # ------------------------------------------------------------------------

    def _map_provider_error(self, e: Exception) -> Exception:
        """
        Map provider-specific exceptions to canonical Corpus errors.

        ⚠️ CRITICAL: Preserve retry_after_ms and other hints.
        """
        # Example provider errors - replace with your actual provider's SDK
        if hasattr(e, "status_code"):
            if e.status_code == 429:
                retry = getattr(e, "retry_after", 5000)
                return ResourceExhausted(
                    "rate limit exceeded",
                    retry_after_ms=retry,
                    throttle_scope="model",
                    details={"provider": "hello"},
                )
            elif e.status_code == 401:
                return AuthError("invalid credentials")
            elif e.status_code == 400:
                return BadRequest(str(e))
            elif e.status_code >= 500:
                return Unavailable("provider unavailable", retry_after_ms=1000)
        
        # Network errors are typically transient
        if isinstance(e, ConnectionError):
            return TransientNetwork("connection failed", retry_after_ms=1000)
        
        # Catch-all - let base wrap as UNAVAILABLE
        return e
```

**What makes this production-grade:**
- ✅ Streaming implementation with chunking
- ✅ Batch partial success with index correlation
- ✅ Deadline propagation to provider calls
- ✅ Centralized error mapping with retry hints
- ✅ Honest capability advertisement
- ✅ Token counting approximation

---

## 4. Test-Driven Iteration

Now make the conformance tests pass, one by one.

### Step 1: Run a Single Test

```bash
pytest tests/conformance/test_capabilities.py -v --tb=short
```

**Expected failure pattern:**
```
FAILED test_capabilities.py::test_capabilities_basic - AssertionError: ...
```

### Step 2: Implement Until Green

```python
# Edit your adapter, re-run test
pytest tests/conformance/test_capabilities.py -k test_capabilities_basic
```

### Step 3: Commit After Each Passing Test

```bash
git add adapters/hello_embedding.py
git commit -m "PASS: capabilities basic shape"
```

### Step 4: Progress to Harder Tests

```bash
# Streaming semantics
pytest tests/conformance/test_streaming.py -v

# Deadline propagation
pytest tests/conformance/test_deadlines.py -v

# Error mapping correctness
pytest tests/conformance/test_errors.py -v
```

**Visual success indicator:**
```
tests/conformance/test_capabilities.py ✓✓✓✓ (4 passed)
tests/conformance/test_embed.py ✓✓✓ (3 passed)
tests/conformance/test_streaming.py ✓✓✓✓✓ (5 passed)
tests/conformance/test_batch.py ✓✓✓✓ (4 passed)
tests/conformance/test_deadlines.py ✓✓✓ (3 passed)
tests/conformance/test_errors.py ✓✓✓✓ (4 passed)
```

---

## 5. Expose It Over HTTP

Now wire your tested adapter to the wire protocol.

### FastAPI Implementation (Most Common)

`services/embedding_service.py`:
```python
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
import json
from corpus_sdk.embedding.embedding_base import WireEmbeddingHandler

from adapters.hello_embedding import HelloEmbeddingAdapter

app = FastAPI(title="Corpus Embedding Adapter")

# MODE DECISION TREE:
# - Behind an existing control plane (Kong, Envoy, Apigee)? → mode="thin"
# - Standalone service? → mode="standalone"
# - Don't know? Start with "standalone" for development
adapter = HelloEmbeddingAdapter(
    mode="standalone",  # Enables deadlines, caching, rate limiting
    # Optional: plug in Redis cache instead of in-memory
    # cache=RedisCache(...),
)
handler = WireEmbeddingHandler(adapter)


@app.post("/v1/embedding")
async def handle_embedding(request: Request):
    """Unary embedding operations."""
    try:
        envelope = await request.json()
        response = await handler.handle(envelope)
        return response
    except Exception as e:
        # WireEmbeddingHandler already formats errors correctly
        # This catch-all is just for framework-level failures
        return {
            "ok": False,
            "code": "UNAVAILABLE",
            "error": type(e).__name__,
            "message": "internal error",
            "ms": 0,
            "details": None,
            "retry_after_ms": None,
        }


@app.post("/v1/embedding/stream")
async def handle_embedding_stream(request: Request):
    """Streaming embedding operations."""
    try:
        envelope = await request.json()
        
        async def stream_generator():
            async for chunk in handler.handle_stream(envelope):
                yield json.dumps(chunk) + "\n"
        
        return StreamingResponse(
            stream_generator(),
            media_type="application/x-ndjson",
        )
    except Exception as e:
        # Fallback error response
        return {
            "ok": False,
            "code": "UNAVAILABLE",
            "error": type(e).__name__,
            "message": "internal error",
            "ms": 0,
        }


@app.get("/v1/health")
async def health_check():
    """Health endpoint for orchestration."""
    health = await adapter.health()
    return health
```

### Run It

```bash
uvicorn services.embedding_service:app --reload --port 8000
```

### Test the Live Endpoint

```bash
# Unary embed
curl -X POST http://localhost:8000/v1/embedding \
  -H "Content-Type: application/json" \
  -d '{
    "op": "embedding.embed",
    "ctx": {
      "request_id": "test-123",
      "tenant": "acme-corp"
    },
    "args": {
      "model": "hello-1",
      "text": "Corpus protocol is elegant",
      "truncate": true
    }
  }'

# Streaming embed
curl -N -X POST http://localhost:8000/v1/embedding/stream \
  -H "Content-Type: application/json" \
  -d '{
    "op": "embedding.stream_embed",
    "ctx": {"tenant": "acme-corp"},
    "args": {
      "model": "hello-1",
      "text": "Watch this stream in real time"
    }
  }'
```

### Alternative Transports

**gRPC:**
```python
# Adapt WireEmbeddingHandler to your gRPC service stub
class EmbeddingGRPCService:
    def __init__(self, adapter):
        self.handler = WireEmbeddingHandler(adapter)
    
    async def Embed(self, request, context):
        envelope = {
            "op": "embedding.embed",
            "ctx": {...},  # Map from gRPC metadata
            "args": {...},  # Map from protobuf
        }
        return await self.handler.handle(envelope)
```

**AWS Lambda:**
```python
def lambda_handler(event, context):
    # API Gateway → JSON envelope
    response = asyncio.run(handler.handle(event))
    return {
        "statusCode": 200,
        "body": json.dumps(response),
        "headers": {"Content-Type": "application/json"},
    }
```

---

## 6. Other Protocol Variants (LLM/Vector/Graph)

Once you understand the Embedding pattern, the other three protocols are **identical in structure, different in domain methods**.

### 6.1 LLM Adapter (Complete Example)

```python
from corpus_sdk.llm.llm_base import (
    BaseLLMAdapter,
    LLMCapabilities,
    LLMCompletion,
    LLMChunk,
    TokenUsage,
    ToolCall,
    ToolCallFunction,
    OperationContext,
)

class HelloLLMAdapter(BaseLLMAdapter):
    async def _do_capabilities(self) -> LLMCapabilities:
        return LLMCapabilities(
            server="hello-llm",
            version="1.0.0",
            model_family="hello",
            max_context_length=4096,
            supports_streaming=True,
            supports_tools=True,           # We support function calling!
            supports_parallel_tool_calls=True,
            supports_tool_choice=True,
            supported_models=("hello-1",),
        )

    async def _do_complete(
        self,
        *,
        messages: List[Mapping[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        ctx: Optional[OperationContext] = None,
    ) -> LLMCompletion:
        """Complete with optional tool calling."""
        
        # 1. Apply timeout from context
        timeout = self._deadline_timeout(ctx)
        
        # 2. Call provider
        try:
            response = await self._client.chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools,
                tool_choice=tool_choice,
                timeout=timeout,
            )
        except Exception as e:
            raise self._map_provider_error(e)
        
        # 3. Map tool calls if present
        tool_calls = []
        if response.tool_calls:
            for tc in response.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        type="function",
                        function=ToolCallFunction(
                            name=tc.function.name,
                            arguments=tc.function.arguments,
                        ),
                    )
                )
        
        return LLMCompletion(
            text=response.content or "",
            model=response.model,
            model_family="hello",
            usage=TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            ),
            finish_reason=response.finish_reason,
            tool_calls=tool_calls,
        )

    async def _do_stream(
        self,
        *,
        messages: List[Mapping[str, str]],
        ctx: Optional[OperationContext] = None,
        **kwargs,
    ) -> AsyncIterator[LLMChunk]:
        """Streaming with tool call deltas."""
        async for chunk in self._client.stream(messages=messages, **kwargs):
            yield LLMChunk(
                text=chunk.content or "",
                is_final=chunk.is_final,
                model=chunk.model,
                usage_so_far=TokenUsage(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens,
                ) if chunk.usage else None,
                tool_calls=[
                    ToolCall(
                        id=tc.id,
                        type="function",
                        function=ToolCallFunction(
                            name=tc.function.name,
                            arguments=tc.function.arguments,
                        ),
                    ) for tc in (chunk.tool_calls or [])
                ],
            )
```

**Critical LLM Implementation Notes:**
- ⚠️ **Never** implement tool execution logic - only transmit tool calls
- ⚠️ **Never** cache across tenant boundaries unless `supports_multi_tenant=True`
- ⚠️ **Always** set `model_family` correctly for router affinity

---

### 6.2 Vector Adapter (With DocStore + Cache Invalidation)

```python
from corpus_sdk.vector.vector_base import (
    BaseVectorAdapter,
    VectorCapabilities,
    QuerySpec,
    BatchQuerySpec,
    UpsertSpec,
    DeleteSpec,
    NamespaceSpec,
    QueryResult,
    VectorMatch,
    Vector,
    VectorID,
    Document,
    RedisDocStore,  # Production-ready
    InMemoryDocStore,  # Development
)

class HelloVectorAdapter(BaseVectorAdapter):
    def __init__(self, redis_client=None, **kwargs):
        # Initialize docstore based on environment
        docstore = None
        if redis_client:
            docstore = RedisDocStore(redis_client)
        else:
            docstore = InMemoryDocStore()
        
        super().__init__(docstore=docstore, **kwargs)
    
    async def _do_capabilities(self) -> VectorCapabilities:
        return VectorCapabilities(
            server="hello-vector",
            version="1.0.0",
            max_dimensions=1536,
            supported_metrics=("cosine", "euclidean"),
            supports_namespaces=True,
            supports_metadata_filtering=True,
            supports_batch_operations=True,
            max_batch_size=100,
            supports_batch_queries=True,  # V1.0 feature
            text_storage_strategy="docstore",  # We separate text from vectors
            max_text_length=10000,
        )

    async def _do_upsert(
        self,
        spec: UpsertSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> UpsertResult:
        """
        Upsert with automatic cache invalidation.

        ⚠️ CRITICAL: 
            - The base already handled docstore storage
            - We only need to store vectors in the backend
            - Caller depends on cache invalidation after success
        """
        try:
            # Store vectors in your vector DB
            result = await self._client.upsert(
                namespace=spec.namespace,
                vectors=[
                    {
                        "id": str(v.id),
                        "values": v.vector,
                        "metadata": v.metadata or {},
                    }
                    for v in spec.vectors
                ],
            )
            
            upserted = UpsertResult(
                upserted_count=result.upserted_count,
                failed_count=result.failed_count,
                failures=result.failures,
            )
            
            # ⚠️ CRITICAL: Cache invalidation is YOUR responsibility
            # The base provides _invalidate_namespace_cache helper
            if upserted.upserted_count > 0:
                await self._invalidate_namespace_cache(spec.namespace)
            
            return upserted
            
        except Exception as e:
            raise self._map_provider_error(e)

    async def _do_query(
        self,
        spec: QuerySpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> QueryResult:
        """
        Vector similarity search.

        ⚠️ CRITICAL:
            - Return VectorMatch objects, not raw vectors
            - The base handles docstore hydration automatically
            - Include both score AND distance when available
        """
        response = await self._client.query(
            namespace=spec.namespace,
            vector=spec.vector,
            top_k=spec.top_k,
            filter=spec.filter,
            include_metadata=spec.include_metadata,
            include_vectors=spec.include_vectors,
        )
        
        matches = [
            VectorMatch(
                vector=Vector(
                    id=VectorID(str(m.id)),
                    vector=m.values if spec.include_vectors else [],
                    metadata=m.metadata if spec.include_metadata else None,
                    namespace=spec.namespace,
                    text=None,  # DocStore hydration happens in base
                ),
                score=m.score,
                distance=m.distance,
            )
            for m in response.matches
        ]
        
        return QueryResult(
            matches=matches,
            query_vector=spec.vector,
            namespace=spec.namespace,
            total_matches=response.total,
        )
```

**Critical Vector Implementation Notes:**
- ⚠️ **Namespace footgun**: `Vector.namespace` MUST equal `UpsertSpec.namespace` - the base enforces this
- ⚠️ **DocStore atomicity**: DocStore failures during upsert cause full operation failure
- ⚠️ **Cache invalidation**: You must call `_invalidate_namespace_cache()` after successful writes
- ⚠️ **Batch query optimization**: Use `_docstore_hydrate_query_results()` for de-duplication

---

### 6.3 Graph Adapter (With Transaction Support)

```python
from corpus_sdk.graph.graph_base import (
    BaseGraphAdapter,
    GraphCapabilities,
    GraphQuerySpec,
    UpsertNodesSpec,
    UpsertEdgesSpec,
    DeleteNodesSpec,
    DeleteEdgesSpec,
    BatchOperation,
    BatchResult,
    Node,
    Edge,
    GraphID,
)

class HelloGraphAdapter(BaseGraphAdapter):
    async def _do_capabilities(self) -> GraphCapabilities:
        return GraphCapabilities(
            server="hello-graph",
            version="1.0.0",
            supports_stream_query=True,
            supported_query_dialects=("cypher", "gremlin"),
            supports_namespaces=True,
            supports_batch=True,
            supports_transaction=True,  # ACID transactions!
            supports_traversal=True,
            max_batch_ops=100,
        )

    async def _do_transaction(
        self,
        operations: List[BatchOperation],
        *,
        ctx: Optional[OperationContext] = None,
    ) -> BatchResult:
        """
        ACID transaction support.

        ⚠️ CRITICAL:
            - All operations succeed OR none are applied
            - Cache invalidation only on successful commit
        """
        # Start transaction
        txn = await self._client.begin_transaction()
        
        try:
            results = []
            for op in operations:
                if op.op == "upsert_nodes":
                    result = await self._do_upsert_nodes(op.args, ctx=ctx, transaction=txn)
                elif op.op == "upsert_edges":
                    result = await self._do_upsert_edges(op.args, ctx=ctx, transaction=txn)
                # ... other operations
                results.append(result)
            
            # Commit - all or nothing
            await txn.commit()
            
            # ⚠️ Cache invalidation only after successful commit
            await self._invalidate_namespace_cache(ctx.tenant if ctx else None)
            
            return BatchResult(results=results, success=True)
            
        except Exception as e:
            await txn.rollback()
            raise
```

**Critical Graph Implementation Notes:**
- ⚠️ **Cache invalidation**: Only invalidate after successful transaction commit
- ⚠️ **Batch operations**: Use `_batch_op_succeeded()` to detect actual mutations
- ⚠️ **Streaming**: Graph streaming uses same pattern as LLM/Embedding

---

## 7. Production Readiness

### 7.1 Mode Strategy Decision Tree

```
┌─────────────────────────────────────────────────────┐
│   MODE SELECTION - CHOOSE ONCE, DOCUMENT FOREVER    │
└─────────────────────────────────────────────────────┘

Is this adapter deployed behind an existing control plane
with its OWN circuit breakers, rate limiters, and cache?
            │
            ├─ YES → mode="thin"
            │        • No-op policies (fastest)
            │        • Your infra owns resilience
            │        • Caching disabled by default
            │
            └─ NO  → mode="standalone" 
                     • SimpleDeadline (timeouts)
                     • SimpleCircuitBreaker (5 failures → 10s cooldown)
                     • InMemoryTTLCache (read paths, 60s TTL)
                     • TokenBucketLimiter (50 req/sec, burst 100)
                     
⚠️ WARNING: "standalone" mode components are per-process only.
           Do NOT use in multi-threaded or distributed deployments
           without replacing them with production implementations.
```

### 7.2 Cache Implementation Selection

```python
# DEVELOPMENT (single process, no external deps)
adapter = HelloVectorAdapter(
    mode="standalone",
    cache=InMemoryTTLCache(),  # Default - fine for dev
)

# PRODUCTION - Distributed Redis Cache
from myapp.cache import RedisCache

redis_cache = RedisCache(
    redis_client=redis.from_url("redis://..."),
    prefix="corpus:vector:",
    default_ttl_s=60,
)

adapter = HelloVectorAdapter(
    mode="thin",  # Your infra owns caching
    cache=redis_cache,  # Distributed, TTL-aware
)

# PRODUCTION - With namespace invalidation support
class ProdCache:
    async def get(self, key): ...
    async def set(self, key, value, ttl_s): ...
    
    # ⚠️ CRITICAL: Implement this for efficient invalidation
    async def invalidate_namespace(self, namespace: str):
        pattern = f"*:ns={namespace}:*"
        await self._redis.delete_by_pattern(pattern)
    
    @property
    def supports_ttl(self): return True
```

### 7.3 Rate Limiter Selection

```python
# DEVELOPMENT (per-process only)
adapter = HelloEmbeddingAdapter(
    mode="standalone",
    limiter=SimpleTokenBucketLimiter(
        rate_per_sec=50,
        burst=100,
    )
)

# PRODUCTION - Distributed rate limiting
from myapp.limiter import SlidingWindowLimiter

adapter = HelloEmbeddingAdapter(
    mode="thin",  # External control plane owns limits
    limiter=SlidingWindowLimiter(
        redis_client=redis_client,
        tenant_based=True,  # Isolate by ctx.tenant
        default_rate=100,
    )
)
```

### 7.4 Deadline Propagation Patterns

```python
async def _do_embed(self, spec, *, ctx=None):
    """Three ways to use deadlines - choose one."""
    
    # OPTION 1: Simple timeout (80% use cases)
    timeout = None
    if ctx and ctx.deadline_ms:
        remaining = ctx.remaining_ms()
        if remaining > 0:
            timeout = remaining / 1000.0
        elif remaining == 0:
            raise DeadlineExceeded("deadline expired")
    
    # OPTION 2: asyncio.wait_for (if provider lacks timeout param)
    try:
        response = await asyncio.wait_for(
            self._client.embed(...),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        raise DeadlineExceeded("provider timeout")
    
    # OPTION 3: asyncio.timeout (Python 3.11+)
    async with asyncio.timeout(timeout):
        response = await self._client.embed(...)
```

### 7.5 Tenant Isolation & Metrics

```python
async def _do_query(self, spec, *, ctx=None):
    # ✅ CORRECT: Use tenant for routing
    tenant_id = ctx.tenant if ctx else "default"
    index_name = f"embeddings_{tenant_id}"
    
    # ❌ WRONG: Logging raw tenant IDs
    logger.info(f"Query from {ctx.tenant}")  # NEVER DO THIS
    
    # ✅ CORRECT: Hashed tenant for debugging
    logger.debug(f"Query from tenant_hash={self._tenant_hash(ctx.tenant)}")
    
    # The base already adds tenant_hash to metrics automatically
    # via _record() - you don't need to do anything
```

---

## 8. Full Conformance Suite

### 8.1 Running All Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run all conformance tests for your component
pytest tests/conformance/ -v --tb=short

# With coverage
pytest tests/conformance/ --cov=adapters --cov-report=html

# Performance benchmarks
pytest tests/conformance/benchmarks/ -v --benchmark-only
```

### 8.2 What Passing Looks Like

```
============================= test session starts ==============================
collected 47 items

tests/conformance/test_capabilities.py ✓✓✓✓ [  8%]
tests/conformance/test_embed.py ✓✓✓✓✓✓✓ [ 23%]
tests/conformance/test_streaming.py ✓✓✓✓✓✓ [ 36%]
tests/conformance/test_batch.py ✓✓✓✓✓✓✓ [ 51%]
tests/conformance/test_deadlines.py ✓✓✓✓✓ [ 61%]
tests/conformance/test_errors.py ✓✓✓✓✓✓ [ 74%]
tests/conformance/test_cache.py ✓✓✓✓ [ 83%]
tests/conformance/test_metrics.py ✓✓✓✓✓ [ 93%]
tests/conformance/test_health.py ✓✓✓ [100%]

============================= 47 passed in 2.34s ==============================
```

### 8.3 Common Test Failures & Fixes

| Failure | Symptom | Fix |
|--------|---------|-----|
| `capabilities.supports_streaming=True but no _do_stream_embed` | Streaming tests fail with NotImplementedError | Implement `_do_stream_embed` or set `supports_streaming=False` |
| `batch_result.embeddings[0].index is None` | Correlation tests fail | Set `index` parameter in EmbeddingVector for batch successes |
| `DEADLINE_EXCEEDED not raised on timeout` | Deadline preflight not firing | Call `_fail_if_expired(ctx)` or use SimpleDeadline |
| `cache_hits counter never increments` | Cache not used | Enable caching in standalone mode, verify cache key stability |
| `retry_after_ms missing from 429 responses` | Error mapping incomplete | Map provider rate limits with `retry_after_ms` parameter |

---

## 9. What to Read Next

| Document | Purpose | When to Read |
|----------|---------|--------------|
| [`spec/IMPLEMENTATION.md`](../spec/IMPLEMENTATION.md) | Full `_do_*` semantics, all edge cases | After quickstart works |
| [`spec/BEHAVIORAL_CONFORMANCE.md`](../spec/BEHAVIORAL_CONFORMANCE.md) | What "correct" means normatively | Before production deploy |
| [`spec/SCHEMA_CONFORMANCE.md`](../spec/SCHEMA_CONFORMANCE.md) | Wire envelope JSON schemas | When debugging wire issues |
| [`spec/ERRORS.md`](../spec/ERRORS.md) | Complete error taxonomy | When adding new error types |
| [`examples/`](../examples/) | Full reference implementations | Always - copy-paste friendly |

**Specific deep dives by component:**

- **Embedding**: `spec/IMPLEMENTATION.md§7.1` - Batch semantics, truncation modes
- **LLM**: `spec/IMPLEMENTATION.md§7.2` - Tool calling, streaming contracts
- **Vector**: `spec/IMPLEMENTATION.md§7.3` - DocStore, namespace canonicalization
- **Graph**: `spec/IMPLEMENTATION.md§7.4` - Transaction semantics, batch invalidation

---

## 10. Adapter Launch Checklist

### 🔴 Pre-Flight (Must Have)
- [ ] `_do_capabilities()` advertises EXACTLY what you implement - no lies
- [ ] All conformance tests pass **unmodified** (`pytest tests/conformance/ -v`)
- [ ] Provider errors mapped to canonical Corpus errors with retry hints preserved
- [ ] `ctx.remaining_ms()` used for provider timeouts in all `_do_*` methods
- [ ] No raw tenant IDs, PII, or full text in logs/metrics - use `_tenant_hash()`
- [ ] Batch operations set `.index` on success items for correlation

### 🟡 Production Hardening (Should Have)
- [ ] Streaming implemented (if `supports_streaming=True`)
- [ ] Batch operations handle partial success gracefully
- [ ] Cache invalidation implemented for write operations (Vector/Graph)
- [ ] DocStore configured and tested with failure scenarios (Vector)
- [ ] `max_batch_size` respected and documented
- [ ] Memory usage tested with maximum batch sizes

### 🟢 Operational Excellence (Nice to Have)
- [ ] Metrics sink configured (Prometheus, Datadog, etc.)
- [ ] Distributed cache (Redis) instead of `InMemoryTTLCache`
- [ ] Distributed rate limiter instead of `SimpleTokenBucketLimiter`
- [ ] Health check includes dependency status (DB, upstream providers)
- [ ] Version pinning: `corpus-sdk>=1.0.0,<2.0.0`

---

## Appendix A: Common Pitfalls by Component

### Embedding

```python
# ❌ WRONG: Assuming batch results align 1:1 with inputs
for i, text in enumerate(spec.texts):
    assert result.embeddings[i].text == text  # MAY FAIL!

# ✅ CORRECT: Use index field for correlation
for emb in result.embeddings:
    original_text = spec.texts[emb.index]  # SAFE
```

### LLM

```python
# ❌ WRONG: Implementing tool execution in adapter
if tool_calls:
    result = await execute_tools(tool_calls)  # NO - that's orchestration!

# ✅ CORRECT: Just pass through tool calls
return LLMCompletion(
    tool_calls=tool_calls,  # Router's job to execute
    # ...
)
```

### Vector

```python
# ❌ WRONG: Ignoring namespace mismatch
vector = Vector(id="123", vector=[...], namespace="user-space")
spec = UpsertSpec(vectors=[vector], namespace="default")  # WILL FAIL!

# ✅ CORRECT: Canonicalize to spec namespace
vector = Vector(
    id="123", 
    vector=[...], 
    namespace="default",  # Must match spec
)
```

### Graph

```python
# ❌ WRONG: Cache invalidation before commit
await self._cache.invalidate_pattern(...)
await txn.commit()  # If commit fails, cache is stale!

# ✅ CORRECT: Invalidate after successful commit
await txn.commit()
await self._invalidate_namespace_cache(namespace)  # SAFE
```

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **Adapter** | A class that implements `_do_*` hooks for a specific provider |
| **Base Class** | `BaseEmbeddingAdapter`, `BaseLLMAdapter`, etc. - provides infrastructure |
| **Wire Handler** | `WireEmbeddingHandler`, etc. - translates JSON ↔ typed calls |
| **Canonical Envelope** | The JSON {op, ctx, args} structure all Corpus services speak |
| **Capability** | A feature advertised in `capabilities()` (e.g., `supports_streaming`) |
| **Thin Mode** | `mode="thin"` - all policies disabled, for composition |
| **Standalone Mode** | `mode="standalone"` - basic in-memory policies enabled |
| **DocStore** | Optional text storage for Vector adapter |
| **Namespace** | Logical collection/tenant isolation scope |
| **Partial Success** | Batch operations that succeed partially, with failures recorded |
| **SIEM-Safe** | No PII, no raw tenants, only hashed identifiers in metrics |

---

## Appendix C: Debugging & Troubleshooting

### Enable Full Wire Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# See every envelope in/out
logging.getLogger("corpus_sdk.wire").setLevel(logging.DEBUG)

# See cache decisions
logging.getLogger("corpus_sdk.cache").setLevel(logging.DEBUG)

# See deadline enforcement
logging.getLogger("corpus_sdk.deadline").setLevel(logging.DEBUG)
```

### Common Error Codes & Meanings

| Code | Meaning | Likely Cause |
|------|---------|--------------|
| `BAD_REQUEST` | Invalid parameters | Namespace mismatch, empty ID, malformed vector |
| `DIMENSION_MISMATCH` | Vector wrong size | `capabilities.max_dimensions` exceeded |
| `DEADLINE_EXCEEDED` | Operation timed out | Provider slow, deadline too tight |
| `RESOURCE_EXHAUSTED` | Rate limited/quota | Provider throttling, `retry_after_ms` present |
| `UNAVAILABLE` | Service down | Provider 5xx, network issue |
| `NOT_SUPPORTED` | Feature not available | Check capability flags |

### Performance Debugging

```python
# Profile a single operation
import time

t0 = time.perf_counter()
result = await adapter.embed(spec)
elapsed = time.perf_counter() - t0

print(f"Total time: {elapsed*1000:.2f}ms")
print(f"  - Deadline remaining: {ctx.remaining_ms()}ms")
print(f"  - Cache hit: {adapter._cache_stats['hits']}")
```

---

**Maintainers:** Corpus SDK Team  
**Last Updated:** 2026-02-11  
**Scope:** Complete adapter authoring reference. When in doubt, conformance tests are the source of truth.