# Corpus OS Quickstart

**Build Production-Ready Corpus Protocol Adapters in Minutes**

**Table of Contents**
- [0. Mental Model (What You're Actually Building)](#0-mental-model-what-youre-actually-building)
- [1. Prerequisites & Setup](#1-prerequisites--setup)
- [2. Conformance-First Development (The Right Way)](#2-conformance-first-development-the-right-way)
- [3. Hello World: Complete Reference Adapters](#3-hello-world-complete-reference-adapters)
  - [3.1 Embedding Adapter (OpenAI/Cohere Style)](#31-embedding-adapter-openaicohere-style)
  - [3.2 LLM Adapter (Chat Completion Style)](#32-llm-adapter-chat-completion-style)
  - [3.3 Vector Adapter (Pinecone/Qdrant Style)](#33-vector-adapter-pineconeqdrant-style)
  - [3.4 Graph Adapter (Neo4j/JanusGraph Style)](#34-graph-adapter-neo4jjanusgraph-style)
- [4. Testing Your Adapter (Certification Suite)](#4-testing-your-adapter-certification-suite)
  - [4.1 Embedding Certification](#41-embedding-certification)
  - [4.2 LLM Certification](#42-llm-certification)
  - [4.3 Vector Certification](#43-vector-certification)
  - [4.4 Graph Certification](#44-graph-certification)
- [5. Understanding Certification Results](#5-understanding-certification-results)
- [6. Expose It Over HTTP](#6-expose-it-over-http)
  - [6.1 Embedding Service (FastAPI)](#61-embedding-service-fastapi)
  - [6.2 LLM Service (FastAPI)](#62-llm-service-fastapi)
  - [6.3 Vector Service (FastAPI)](#63-vector-service-fastapi)
  - [6.4 Graph Service (FastAPI)](#64-graph-service-fastapi)
- [7. Protocol-Specific Requirements & Pitfalls](#7-protocol-specific-requirements--pitfalls)
  - [7.1 Embedding Protocol](#71-embedding-protocol)
  - [7.2 LLM Protocol](#72-llm-protocol)
  - [7.3 Vector Protocol](#73-vector-protocol)
  - [7.4 Graph Protocol](#74-graph-protocol)
- [8. Production Readiness](#8-production-readiness)
- [9. What to Read Next](#9-what-to-read-next)
- [10. Adapter Launch Checklist](#10-adapter-launch-checklist)
  - [10.1 Universal Requirements (All Protocols)](#101-universal-requirements-all-protocols)
  - [10.2 Embedding-Specific Checklist](#102-embedding-specific-checklist)
  - [10.3 LLM-Specific Checklist](#103-llm-specific-checklist)
  - [10.4 Vector-Specific Checklist](#104-vector-specific-checklist)
  - [10.5 Graph-Specific Checklist](#105-graph-specific-checklist)
- [Appendix A: Common Pitfalls by Component](#appendix-a-common-pitfalls-by-component)
- [Appendix B: Glossary](#appendix-b-glossary)
- [Appendix C: Debugging & Troubleshooting](#appendix-c-debugging--troubleshooting)

---

> **Goal:** Get a real, production-ready adapter speaking **any Corpus Protocol v1.0** (Embedding, LLM, Vector, or Graph) in under 15 minutes.  
> **Audience:** SDK / adapter authors for embedding providers, LLM APIs, vector databases, and graph databases.  
> **You'll build:** A complete, certified adapter with streaming, batch operations, error mapping, and full conformance—then reuse the same pattern across all four protocols.

**By the end of this guide you will have:**
- ✅ A fully tested adapter implementation for your chosen protocol
- ✅ Streaming and batch operation support (where applicable)
- ✅ Proper error mapping and deadline propagation
- ✅ Cache invalidation (Vector/Graph) or idempotency (Embedding)
- ✅ **Gold certification** from the official conformance suite
- ✅ **Full compliance with Corpus Protocol v1.0 specification**
- ✅ A reusable pattern that works for all four protocol variants

---

## 0. Mental Model (What You're Actually Building)

An **adapter** is a thin translation layer that converts between Corpus Protocol and your provider's native API:

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
- `_do_capabilities()` - What your adapter supports (**MUST include `protocol` field**)
- `_do_embed()` / `_do_complete()` / `_do_query()` / etc. - Core operation
- `_do_stream_*()` - Streaming (if supported) - **MUST use canonical envelope format**
- `_do_health()` - Liveness check
- `_do_get_stats()` - Service statistics (optional but recommended)
- `build_*_envelope()` - Test fixture support (**REQUIRED for certification**)

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
- [`corpus_sdk/embedding/embedding_base.py`](../corpus_sdk/embedding/embedding_base.py) — Embedding Protocol V1
- [`corpus_sdk/llm/llm_base.py`](../corpus_sdk/llm/llm_base.py) — LLM Protocol V1
- [`corpus_sdk/vector/vector_base.py`](../corpus_sdk/vector/vector_base.py) — Vector Protocol V1
- [`corpus_sdk/graph/graph_base.py`](../corpus_sdk/graph/graph_base.py) — Graph Protocol V1

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
pip install fastapi uvicorn        # Optional: for HTTP services
pip install redis msgpack          # Optional: for RedisDocStore
```

---

## 2. Conformance-First Development (The Right Way)

**Stop writing code before you have a failing test.**

The Corpus certification suite evaluates adapters against the protocol specification. **You copy the tests into your project and run them locally.**

### Step 1: Copy the Official Conformance Tests for Your Protocol

```bash
# For Embedding adapters
cp -r $(python -c "import corpus_sdk; print(corpus_sdk.__path__[0])")/tests/embedding ./tests/

# For LLM adapters
cp -r $(python -c "import corpus_sdk; print(corpus_sdk.__path__[0])")/tests/llm ./tests/

# For Vector adapters  
cp -r $(python -c "import corpus_sdk; print(corpus_sdk.__path__[0])")/tests/vector ./tests/

# For Graph adapters
cp -r $(python -c "import corpus_sdk; print(corpus_sdk.__path__[0])")/tests/graph ./tests/

# Copy live endpoint tests and schema validation (all protocols)
cp -r $(python -c "import corpus_sdk; print(corpus_sdk.__path__[0])")/tests/live ./tests/
cp -r $(python -c "import corpus_sdk; print(corpus_sdk.__path__[0])")/tests/schema ./tests/
```

Your `tests/` directory now contains the official, unmodified certification suite for your chosen protocol.

### Step 2: Create Your Test Fixture

Create `tests/conftest.py`:

```python
import pytest
from adapters.my_adapter import MyAdapter  # Your adapter

@pytest.fixture
def adapter():
    """Return your adapter instance for certification testing."""
    return MyAdapter(mode="thin")  # Use "thin" for certification
```

**Do not modify the copied test files.** They are the source of truth.

### Step 3: Run a Single Test and Watch It Fail

```bash
# For Embedding
pytest tests/embedding/test_capabilities.py -v -k test_capabilities_basic

# For LLM
pytest tests/llm/test_capabilities.py -v -k test_capabilities_basic

# For Vector
pytest tests/vector/test_capabilities.py -v -k test_capabilities_basic

# For Graph
pytest tests/graph/test_capabilities.py -v -k test_capabilities_basic
```

**Expected output:**
```
_________________________________ FAILURE __________________________________
NotImplementedError: _do_capabilities not implemented
```

This is **GOOD**. You now have a target.

### Why This Works

| What you DO | Why |
|------------|-----|
| ✅ Copy tests into `./tests/` | Tests run locally, fast iteration |
| ✅ Keep tests unmodified | Tests are the normative specification |
| ✅ One fixture in `conftest.py` | Pytest automatically loads it |
| ✅ Run tests against your adapter | You see exactly what's broken |

**The conformance tests are the source of truth.** When they all pass, your adapter is done.

For detailed guidance on running and interpreting conformance tests, see [`CONFORMANCE_GUIDE.md`](CONFORMANCE_GUIDE.md).

---

## 3. Hello World: Complete Reference Adapters

This section provides **four complete, specification-compliant reference implementations**—one for each protocol. **Choose the one that matches your provider type.**

---

### 3.1 Embedding Adapter (OpenAI/Cohere Style)

Create `adapters/hello_embedding.py`:

```python
import asyncio
import time
import hashlib
import secrets
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
    Production-ready, specification-compliant embedding adapter.
    
    ⚠️ CRITICAL: Constructor must accept endpoint=None for certification.
    ⚠️ CRITICAL: All wire formats exactly match Corpus Protocol v1.0.
    """
    def __init__(self, endpoint: Optional[str] = None, mode: str = "standalone"):
        """Initialize adapter. endpoint is provided by CORPUS_ENDPOINT when set."""
        super().__init__(mode=mode)
        self.endpoint = endpoint
        # Initialize your client here
        
        # Idempotency cache (24-hour retention per specification)
        self._idempotency_cache = {}  # Replace with Redis in production

    async def _do_capabilities(self) -> EmbeddingCapabilities:
        """
        Advertise what this adapter supports.
        
        ⚠️ REQUIRED FIELDS per specification:
        - protocol: MUST be "embedding/v1.0"
        - idempotent_writes: MUST be true for embedding operations
        """
        return EmbeddingCapabilities(
            server="hello-embedding",
            protocol="embedding/v1.0",  # ✅ REQUIRED - exact format
            version="1.0.0",
            supported_models=("hello-1", "hello-2"),
            max_batch_size=10,
            max_text_length=1000,
            max_dimensions=8,
            supports_normalization=False,
            normalizes_at_source=False,
            supports_truncation=True,
            supports_token_counting=True,
            supports_streaming=True,
            supports_batch_embedding=True,
            supports_deadline=True,
            idempotent_writes=True,      # ✅ REQUIRED - embedding ops are idempotent
            supports_multi_tenant=True,
            truncation_mode="base",
        )

    async def _do_embed(
        self,
        spec: EmbedSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> EmbedResult:
        """Generate embedding for a single text."""
        # --------------------------------------------------------------------
        # IDEMPOTENCY CHECK (Required per specification §6.1)
        # --------------------------------------------------------------------
        if ctx and ctx.idempotency_key and ctx.tenant:
            cache_key = f"idem:{ctx.tenant}:{ctx.idempotency_key}"
            cached = self._idempotency_cache.get(cache_key)
            if cached:
                return cached
        
        # --------------------------------------------------------------------
        # DEADLINE PROPAGATION (Required per specification §6.1)
        # --------------------------------------------------------------------
        timeout = None
        if ctx and ctx.deadline_ms:
            remaining = ctx.remaining_ms()
            if remaining > 0:
                timeout = remaining / 1000.0
            elif remaining == 0:
                raise DeadlineExceeded("deadline already expired")

        try:
            # Simulate provider call with deadline
            await asyncio.sleep(0.01)
            
            # Generate embedding
            vec = [float(len(spec.text))] + [0.0] * 7

            result = EmbedResult(
                embedding=EmbeddingVector(
                    vector=vec,
                    text=spec.text,
                    model=spec.model,
                    dimensions=len(vec),
                ),
                model=spec.model,
                text=spec.text,
                tokens_used=len(spec.text) // 4,
                truncated=False,
            )
            
            # --------------------------------------------------------------------
            # IDEMPOTENCY STORE (24-hour retention)
            # --------------------------------------------------------------------
            if ctx and ctx.idempotency_key and ctx.tenant:
                self._idempotency_cache[cache_key] = result
                # In production: await redis.setex(cache_key, 86400, result)
            
            return result
            
        except asyncio.TimeoutError:
            raise DeadlineExceeded("provider timeout")
        except Exception as e:
            raise self._map_provider_error(e)

    async def _do_stream_embed(
        self,
        spec: EmbedSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream embedding generation in chunks.
        
        ⚠️ CRITICAL: Must use canonical streaming envelope format:
        { "ok": true, "code": "STREAMING", "ms": 12.3, "chunk": { ... } }
        """
        base_vec = [float(len(spec.text))] + [0.0] * 7
        start_time = time.time()
        
        # Chunk 1: first 2 dimensions
        elapsed_ms = (time.time() - start_time) * 1000
        yield {
            "ok": True,
            "code": "STREAMING",  # ✅ REQUIRED - must be exactly "STREAMING"
            "ms": elapsed_ms,
            "chunk": {
                "embeddings": [
                    {
                        "vector": base_vec[:2] + [0.0] * 6,
                        "text": spec.text,
                        "model": spec.model,
                        "dimensions": 8,
                    }
                ],
                "is_final": False,
            }
        }
        await asyncio.sleep(0.005)
        
        # Final chunk: complete vector + usage
        elapsed_ms = (time.time() - start_time) * 1000
        yield {
            "ok": True,
            "code": "STREAMING",
            "ms": elapsed_ms,
            "chunk": {
                "embeddings": [
                    {
                        "vector": base_vec,
                        "text": spec.text,
                        "model": spec.model,
                        "dimensions": 8,
                    }
                ],
                "is_final": True,
                "usage": {"prompt_tokens": len(spec.text) // 4},
                "model": spec.model,
            }
        }

    async def _do_embed_batch(
        self,
        spec: BatchEmbedSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> BatchEmbedResult:
        """
        Batch embedding with partial success.
        
        ⚠️ CRITICAL: 
        - Field name MUST be "failures" (not "failed_texts")
        - Each success MUST include "index" for correlation
        """
        embeddings = []
        failures = []

        for idx, text in enumerate(spec.texts):
            try:
                # Validate each item
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
                        index=idx,  # ⚠️ REQUIRED for correlation
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
        """Token counting approximation."""
        return len(text) // 4

    async def _do_health(
        self,
        *,
        ctx: Optional[OperationContext] = None
    ) -> Dict[str, Any]:
        """
        Liveness probe with graded status.
        
        Status MUST be one of: "ok", "degraded", "down"
        """
        return {
            "ok": True,
            "status": "ok",  # ✅ REQUIRED - one of: ok, degraded, down
            "server": "hello-embedding",
            "version": "1.0.0",
            "reason": None,
        }

    async def _do_get_stats(
        self,
        *,
        ctx: Optional[OperationContext] = None
    ) -> Dict[str, Any]:
        """
        Retrieve embedding service statistics.
        
        Optional but recommended per specification.
        """
        return {
            "total_requests": 12345,
            "total_tokens": 987654,
            "total_errors": 42,
            "uptime_ms": 86400000,
            "models": {
                "hello-1": {
                    "requests": 10000,
                    "tokens": 800000,
                    "errors": 30,
                },
                "hello-2": {
                    "requests": 2345,
                    "tokens": 187654,
                    "errors": 12,
                }
            },
            "cache_hit_rate": 0.85,
        }

    # ------------------------------------------------------------------------
    # TEST FIXTURE SUPPORT (REQUIRED FOR CERTIFICATION)
    # ------------------------------------------------------------------------
    
    def build_embedding_embed_envelope(
        self,
        model: str = "hello-1",
        text: str = "Hello world",
        truncate: bool = True,
        normalize: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Return wire envelope for single embedding operation."""
        return {
            "op": "embedding.embed",
            "ctx": {
                "request_id": "test-123",
                "tenant": "test-tenant",
                "deadline_ms": 5000,
                "idempotency_key": "test-idem-123",
            },
            "args": {
                "model": model,
                "text": text,
                "truncate": truncate,
                "normalize": normalize,
                **kwargs
            }
        }

    def build_embedding_stream_embed_envelope(
        self,
        model: str = "hello-1",
        text: str = "Hello world",
        truncate: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Return wire envelope for streaming embedding operation."""
        return {
            "op": "embedding.stream_embed",
            "ctx": {
                "request_id": "test-123",
                "tenant": "test-tenant",
                "deadline_ms": 5000,
            },
            "args": {
                "model": model,
                "text": text,
                "truncate": truncate,
                **kwargs
            }
        }

    def build_embedding_batch_embed_envelope(
        self,
        model: str = "hello-1",
        texts: List[str] = None,
        truncate: bool = True,
        normalize: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Return wire envelope for batch embedding operation."""
        if texts is None:
            texts = ["Hello", "world", "test"]
        return {
            "op": "embedding.embed_batch",
            "ctx": {
                "request_id": "test-123",
                "tenant": "test-tenant",
                "deadline_ms": 5000,
            },
            "args": {
                "model": model,
                "texts": texts,
                "truncate": truncate,
                "normalize": normalize,
                **kwargs
            }
        }

    def build_embedding_capabilities_envelope(
        self,
        **kwargs
    ) -> Dict[str, Any]:
        """Return wire envelope for capabilities discovery."""
        return {
            "op": "embedding.capabilities",
            "ctx": {
                "request_id": "test-123",
                "tenant": "test-tenant",
            },
            "args": {}
        }

    def build_embedding_health_envelope(
        self,
        **kwargs
    ) -> Dict[str, Any]:
        """Return wire envelope for health check."""
        return {
            "op": "embedding.health",
            "ctx": {
                "request_id": "test-123",
                "tenant": "test-tenant",
            },
            "args": {}
        }

    def build_embedding_count_tokens_envelope(
        self,
        text: str = "Hello world",
        model: str = "hello-1",
        **kwargs
    ) -> Dict[str, Any]:
        """Return wire envelope for token counting."""
        return {
            "op": "embedding.count_tokens",
            "ctx": {
                "request_id": "test-123",
                "tenant": "test-tenant",
            },
            "args": {
                "text": text,
                "model": model,
                **kwargs
            }
        }

    def build_embedding_get_stats_envelope(
        self,
        **kwargs
    ) -> Dict[str, Any]:
        """Return wire envelope for stats retrieval."""
        return {
            "op": "embedding.get_stats",
            "ctx": {
                "request_id": "test-123",
                "tenant": "test-tenant",
            },
            "args": {}
        }

    # ------------------------------------------------------------------------
    # ERROR MAPPING
    # ------------------------------------------------------------------------

    def _map_provider_error(self, e: Exception) -> Exception:
        """Map provider exceptions to canonical Corpus errors."""
        if hasattr(e, "status_code"):
            if e.status_code == 429:
                retry = getattr(e, "retry_after", 5000)
                return ResourceExhausted(
                    "rate limit exceeded",
                    retry_after_ms=retry,
                    throttle_scope="model",
                )
            elif e.status_code == 401:
                return AuthError("invalid credentials")
            elif e.status_code == 400:
                return BadRequest(str(e))
            elif e.status_code >= 500:
                return Unavailable(
                    "provider unavailable",
                    retry_after_ms=1000,
                )
        return e

    # ------------------------------------------------------------------------
    # TENANT HASHING (SIEM-Safe)
    # ------------------------------------------------------------------------
    
    def _tenant_hash(self, tenant: Optional[str]) -> Optional[str]:
        """Return irreversible hash of tenant identifier."""
        if not tenant:
            return None
        salt = "corpus-salt"  # In production: os.environ.get("CORPUS_TENANT_SALT")
        return hashlib.sha256(f"{salt}:{tenant}".encode()).hexdigest()[:16]
```

**What makes this specification-compliant:**
- ✅ **REQUIRED** `protocol="embedding/v1.0"` in capabilities
- ✅ **REQUIRED** `idempotent_writes=True` in capabilities
- ✅ **REQUIRED** Canonical streaming envelope `{ok, code, ms, chunk}`
- ✅ **REQUIRED** Batch field name `failures` (not `failed_texts`)
- ✅ **REQUIRED** Idempotency key deduplication (24-hour retention)
- ✅ **REQUIRED** Constructor accepts `endpoint=None`
- ✅ **REQUIRED** All `build_*_envelope()` methods for test fixture
- ✅ **REQUIRED** Graded health status (`ok`/`degraded`/`down`)

---

### 3.2 LLM Adapter (Chat Completion Style)

Create `adapters/hello_llm.py`:

```python
import asyncio
import time
import secrets
from typing import AsyncIterator, Optional, List, Dict, Any, Union, Mapping
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
    ModelOverloaded,
)

class HelloLLMAdapter(BaseLLMAdapter):
    """
    Production-ready, specification-compliant LLM adapter.
    
    ⚠️ CRITICAL: Never implement tool execution - only pass through tool calls.
    ⚠️ CRITICAL: Constructor must accept endpoint=None for certification.
    """
    def __init__(self, endpoint: Optional[str] = None, mode: str = "standalone"):
        super().__init__(mode=mode)
        self.endpoint = endpoint

    async def _do_capabilities(self) -> LLMCapabilities:
        """
        Advertise LLM capabilities.
        
        ⚠️ REQUIRED FIELDS:
        - protocol: MUST be "llm/v1.0"
        - model_family: MUST be set (e.g., "gpt-4", "claude-3")
        """
        return LLMCapabilities(
            server="hello-llm",
            protocol="llm/v1.0",  # ✅ REQUIRED
            version="1.0.0",
            model_family="hello-family",  # ✅ REQUIRED
            max_context_length=8192,
            supports_streaming=True,
            supports_roles=True,
            supports_json_output=False,
            supports_tools=True,
            supports_parallel_tool_calls=False,
            supports_tool_choice=True,
            max_tool_calls_per_turn=1,
            idempotent_writes=False,
            supports_multi_tenant=True,
            supports_system_message=True,
            supports_deadline=True,
            supports_count_tokens=True,
            supported_models=("hello-1", "hello-2"),
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
        """Generate LLM completion."""
        # Deadline propagation
        if ctx and ctx.deadline_ms:
            remaining = ctx.remaining_ms()
            if remaining == 0:
                raise DeadlineExceeded("deadline expired")

        # Simulate generation
        await asyncio.sleep(0.02)
        
        # If tools requested, return a tool call
        if tools:
            tool_calls = [
                ToolCall(
                    id=f"call_{secrets.token_hex(8)}",  # ✅ REQUIRED: generated ID
                    type="function",
                    function=ToolCallFunction(
                        name=tools[0].get("function", {}).get("name", "unknown"),
                        arguments='{"query": "hello"}'
                    )
                )
            ]
        else:
            tool_calls = []

        return LLMCompletion(
            text="Hello, I'm a helpful assistant!",
            model=model or "hello-1",
            model_family="hello-family",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            finish_reason="stop" if not tool_calls else "tool_calls",
            tool_calls=tool_calls,
        )

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
        """Stream LLM completion."""
        chunks = [
            LLMChunk(text="Hello, ", is_final=False),
            LLMChunk(text="I'm ", is_final=False),
            LLMChunk(text="a helpful ", is_final=False),
            LLMChunk(text="assistant!", is_final=False),
        ]
        
        for chunk in chunks:
            await asyncio.sleep(0.01)
            yield chunk
        
        # Final chunk with usage (✅ REQUIRED)
        yield LLMChunk(
            text="",
            is_final=True,
            model=model or "hello-1",
            usage_so_far=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    async def _do_count_tokens(
        self,
        text: str,
        model: Optional[str] = None,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> int:
        """Count tokens in text."""
        return len(text) // 4

    async def _do_health(
        self,
        *,
        ctx: Optional[OperationContext] = None
    ) -> Dict[str, Any]:
        """Health check with graded status."""
        return {
            "ok": True,
            "status": "ok",
            "server": "hello-llm",
            "version": "1.0.0",
            "reason": None,
        }

    # ------------------------------------------------------------------------
    # TEST FIXTURE SUPPORT (REQUIRED FOR CERTIFICATION)
    # ------------------------------------------------------------------------
    
    def build_llm_complete_envelope(
        self,
        model: str = "hello-1",
        messages: List[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Return wire envelope for LLM completion."""
        if messages is None:
            messages = [{"role": "user", "content": "Hello"}]
        return {
            "op": "llm.complete",
            "ctx": {
                "request_id": "test-123",
                "tenant": "test-tenant",
                "deadline_ms": 5000,
            },
            "args": {
                "model": model,
                "messages": messages,
                **kwargs
            }
        }

    def build_llm_stream_envelope(
        self,
        model: str = "hello-1",
        messages: List[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Return wire envelope for streaming LLM completion."""
        if messages is None:
            messages = [{"role": "user", "content": "Hello"}]
        return {
            "op": "llm.stream",
            "ctx": {
                "request_id": "test-123",
                "tenant": "test-tenant",
                "deadline_ms": 5000,
            },
            "args": {
                "model": model,
                "messages": messages,
                **kwargs
            }
        }

    def build_llm_capabilities_envelope(self, **kwargs) -> Dict[str, Any]:
        """Return wire envelope for capabilities discovery."""
        return {
            "op": "llm.capabilities",
            "ctx": {"request_id": "test-123", "tenant": "test-tenant"},
            "args": {}
        }

    def build_llm_health_envelope(self, **kwargs) -> Dict[str, Any]:
        """Return wire envelope for health check."""
        return {
            "op": "llm.health",
            "ctx": {"request_id": "test-123", "tenant": "test-tenant"},
            "args": {}
        }

    def build_llm_count_tokens_envelope(
        self,
        text: str = "Hello world",
        model: str = "hello-1",
        **kwargs
    ) -> Dict[str, Any]:
        """Return wire envelope for token counting."""
        return {
            "op": "llm.count_tokens",
            "ctx": {"request_id": "test-123", "tenant": "test-tenant"},
            "args": {"text": text, "model": model, **kwargs}
        }
```

---

### 3.3 Vector Adapter (Pinecone/Qdrant Style)

Create `adapters/hello_vector.py`:

```python
import asyncio
import time
from typing import Optional, List, Dict, Any
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
    NotSupported,
    DimensionMismatch,
    IndexNotReady,
)

class HelloVectorAdapter(BaseVectorAdapter):
    """
    Production-ready, specification-compliant vector adapter.
    
    ⚠️ CRITICAL: Namespace footgun prevention - always canonicalize.
    ⚠️ CRITICAL: Cache invalidation AFTER successful writes.
    """
    def __init__(self, endpoint: Optional[str] = None, mode: str = "standalone"):
        super().__init__(mode=mode)
        self.endpoint = endpoint
        self._vectors = {}  # Simulated storage

    async def _do_capabilities(self) -> VectorCapabilities:
        """
        Advertise vector database capabilities.
        
        ⚠️ REQUIRED FIELDS:
        - protocol: MUST be "vector/v1.0"
        """
        return VectorCapabilities(
            server="hello-vector",
            protocol="vector/v1.0",  # ✅ REQUIRED
            version="1.0.0",
            max_dimensions=1024,
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
        if ctx and ctx.deadline_ms:
            remaining = ctx.remaining_ms()
            if remaining == 0:
                raise DeadlineExceeded("deadline expired")

        # Simulate search
        await asyncio.sleep(0.01)
        
        matches = [
            VectorMatch(
                vector=Vector(
                    id=VectorID("vec-1"),
                    vector=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    metadata={"key": "value"},
                    namespace=spec.namespace,
                    text="Sample text",
                ),
                score=0.95,
                distance=0.05,
            )
        ]

        return QueryResult(
            matches=matches,
            query_vector=spec.vector,
            namespace=spec.namespace,
            total_matches=1,
        )

    async def _do_batch_query(
        self,
        spec: BatchQuerySpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> List[QueryResult]:
        """Execute batch vector similarity search."""
        results = []
        for query in spec.queries:
            result = await self._do_query(query, ctx=ctx)
            results.append(result)
        return results

    async def _do_upsert(
        self,
        spec: UpsertSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> UpsertResult:
        """
        Upsert vectors with namespace canonicalization.
        
        ⚠️ CRITICAL: Namespace footgun prevention already handled by base class.
        """
        upserted = 0
        failures = []

        for v in spec.vectors:
            try:
                self._vectors[f"{spec.namespace}:{v.id}"] = v
                upserted += 1
            except Exception as e:
                failures.append({
                    "index": len(failures),
                    "id": str(v.id),
                    "error": type(e).__name__,
                    "code": "UPSERT_FAILED",
                    "message": str(e),
                })

        # ✅ CORRECT: Invalidate AFTER successful write
        if upserted > 0:
            await self._invalidate_namespace_cache(spec.namespace)

        return UpsertResult(
            upserted_count=upserted,
            failed_count=len(failures),
            failures=failures,
        )

    async def _do_delete(
        self,
        spec: DeleteSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> DeleteResult:
        """Delete vectors."""
        deleted = 0
        failures = []

        for vid in spec.ids:
            key = f"{spec.namespace}:{vid}"
            if key in self._vectors:
                del self._vectors[key]
                deleted += 1
            else:
                failures.append({
                    "index": len(failures),
                    "id": str(vid),
                    "error": "NotFound",
                    "code": "NOT_FOUND",
                    "message": f"Vector {vid} not found",
                })

        # ✅ CORRECT: Invalidate AFTER successful delete
        if deleted > 0:
            await self._invalidate_namespace_cache(spec.namespace)

        return DeleteResult(
            deleted_count=deleted,
            failed_count=len(failures),
            failures=failures,
        )

    async def _do_create_namespace(
        self,
        spec: NamespaceSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> NamespaceResult:
        """Create namespace/collection."""
        return NamespaceResult(
            success=True,
            namespace=spec.namespace,
            details={"dimensions": spec.dimensions, "metric": spec.distance_metric},
        )

    async def _do_delete_namespace(
        self,
        namespace: str,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> NamespaceResult:
        """Delete namespace/collection."""
        result = NamespaceResult(
            success=True,
            namespace=namespace,
            details={},
        )
        
        # ✅ CORRECT: Invalidate AFTER successful delete
        if result.success:
            await self._invalidate_namespace_cache(namespace)
            
        return result

    async def _do_health(
        self,
        *,
        ctx: Optional[OperationContext] = None
    ) -> Dict[str, Any]:
        """Health check with graded status."""
        return {
            "ok": True,
            "status": "ok",
            "server": "hello-vector",
            "version": "1.0.0",
            "namespaces": {"default": "ready"},
        }

    # ------------------------------------------------------------------------
    # TEST FIXTURE SUPPORT (REQUIRED FOR CERTIFICATION)
    # ------------------------------------------------------------------------
    
    def build_vector_query_envelope(
        self,
        namespace: str = "default",
        vector: List[float] = None,
        top_k: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """Return wire envelope for vector query."""
        if vector is None:
            vector = [1.0, 0.0, 0.0, 0.0]
        return {
            "op": "vector.query",
            "ctx": {
                "request_id": "test-123",
                "tenant": "test-tenant",
                "deadline_ms": 5000,
            },
            "args": {
                "namespace": namespace,
                "vector": vector,
                "top_k": top_k,
                **kwargs
            }
        }

    def build_vector_upsert_envelope(
        self,
        namespace: str = "default",
        vectors: List[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Return wire envelope for vector upsert."""
        if vectors is None:
            vectors = [{
                "id": "vec-1",
                "vector": [1.0, 0.0, 0.0, 0.0],
                "text": "Hello",
                "metadata": {"source": "test"}
            }]
        return {
            "op": "vector.upsert",
            "ctx": {
                "request_id": "test-123",
                "tenant": "test-tenant",
                "deadline_ms": 5000,
            },
            "args": {
                "namespace": namespace,
                "vectors": vectors,
                **kwargs
            }
        }

    def build_vector_delete_envelope(
        self,
        namespace: str = "default",
        ids: List[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Return wire envelope for vector delete."""
        if ids is None:
            ids = ["vec-1"]
        return {
            "op": "vector.delete",
            "ctx": {
                "request_id": "test-123",
                "tenant": "test-tenant",
                "deadline_ms": 5000,
            },
            "args": {
                "namespace": namespace,
                "ids": ids,
                **kwargs
            }
        }

    def build_vector_capabilities_envelope(self, **kwargs) -> Dict[str, Any]:
        """Return wire envelope for capabilities discovery."""
        return {
            "op": "vector.capabilities",
            "ctx": {"request_id": "test-123", "tenant": "test-tenant"},
            "args": {}
        }

    def build_vector_health_envelope(self, **kwargs) -> Dict[str, Any]:
        """Return wire envelope for health check."""
        return {
            "op": "vector.health",
            "ctx": {"request_id": "test-123", "tenant": "test-tenant"},
            "args": {}
        }

    def build_vector_create_namespace_envelope(
        self,
        namespace: str = "test",
        dimensions: int = 8,
        distance_metric: str = "cosine",
        **kwargs
    ) -> Dict[str, Any]:
        """Return wire envelope for namespace creation."""
        return {
            "op": "vector.create_namespace",
            "ctx": {
                "request_id": "test-123",
                "tenant": "test-tenant",
                "deadline_ms": 5000,
            },
            "args": {
                "namespace": namespace,
                "dimensions": dimensions,
                "distance_metric": distance_metric,
                **kwargs
            }
        }

    def build_vector_delete_namespace_envelope(
        self,
        namespace: str = "test",
        **kwargs
    ) -> Dict[str, Any]:
        """Return wire envelope for namespace deletion."""
        return {
            "op": "vector.delete_namespace",
            "ctx": {
                "request_id": "test-123",
                "tenant": "test-tenant",
                "deadline_ms": 5000,
            },
            "args": {
                "namespace": namespace,
                **kwargs
            }
        }

    def build_vector_batch_query_envelope(
        self,
        namespace: str = "default",
        queries: List[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Return wire envelope for batch query."""
        if queries is None:
            queries = [
                {"vector": [1.0, 0.0, 0.0, 0.0], "top_k": 5},
                {"vector": [0.0, 1.0, 0.0, 0.0], "top_k": 5}
            ]
        return {
            "op": "vector.batch_query",
            "ctx": {
                "request_id": "test-123",
                "tenant": "test-tenant",
                "deadline_ms": 5000,
            },
            "args": {
                "namespace": namespace,
                "queries": queries,
                **kwargs
            }
        }
```

---

### 3.4 Graph Adapter (Neo4j/JanusGraph Style)

Create `adapters/hello_graph.py`:

```python
import asyncio
import time
from typing import AsyncIterator, Optional, List, Dict, Any, Union, Mapping
from corpus_sdk.graph.graph_base import (
    BaseGraphAdapter,
    GraphCapabilities,
    GraphQuerySpec,
    UpsertNodesSpec,
    UpsertEdgesSpec,
    DeleteNodesSpec,
    DeleteEdgesSpec,
    BulkVerticesSpec,
    BatchOperation,
    GraphTraversalSpec,
    QueryResult,
    QueryChunk,
    UpsertResult,
    DeleteResult,
    BulkVerticesResult,
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
    Production-ready, specification-compliant graph adapter.
    
    ⚠️ CRITICAL: Cache invalidation only AFTER successful transaction commit.
    ⚠️ CRITICAL: Never implement query rewriting - pass through raw queries.
    """
    def __init__(self, endpoint: Optional[str] = None, mode: str = "standalone"):
        super().__init__(mode=mode)
        self.endpoint = endpoint
        self._nodes = {}
        self._edges = {}

    async def _do_capabilities(self) -> GraphCapabilities:
        """
        Advertise graph database capabilities.
        
        ⚠️ REQUIRED FIELDS:
        - protocol: MUST be "graph/v1.0"
        """
        return GraphCapabilities(
            server="hello-graph",
            protocol="graph/v1.0",  # ✅ REQUIRED
            version="1.0.0",
            supports_stream_query=True,
            supported_query_dialects=("cypher", "gremlin"),
            supports_namespaces=True,
            supports_property_filters=True,
            supports_bulk_vertices=False,
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
        if ctx and ctx.deadline_ms:
            remaining = ctx.remaining_ms()
            if remaining == 0:
                raise DeadlineExceeded("deadline expired")

        await asyncio.sleep(0.01)
        
        return QueryResult(
            records=[{"result": "data"}],
            summary={"query_time_ms": 10},
            dialect=spec.dialect,
            namespace=spec.namespace,
        )

    async def _do_stream_query(
        self,
        spec: GraphQuerySpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> AsyncIterator[QueryChunk]:
        """Stream graph query results."""
        chunks = [
            QueryChunk(records=[{"chunk": 1}], is_final=False),
            QueryChunk(records=[{"chunk": 2}], is_final=False),
            QueryChunk(records=[], is_final=True, summary={"total": 2}),
        ]
        
        for chunk in chunks:
            await asyncio.sleep(0.01)
            yield chunk

    async def _do_upsert_nodes(
        self,
        spec: UpsertNodesSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> UpsertResult:
        """Upsert nodes."""
        upserted = 0
        failures = []
        
        for node in spec.nodes:
            key = f"{spec.namespace}:{node.id}"
            self._nodes[key] = node
            upserted += 1
            
        return UpsertResult(
            upserted_count=upserted,
            failed_count=len(failures),
            failures=failures,
        )

    async def _do_upsert_edges(
        self,
        spec: UpsertEdgesSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> UpsertResult:
        """Upsert edges."""
        upserted = 0
        failures = []
        
        for edge in spec.edges:
            key = f"{spec.namespace}:{edge.id}"
            self._edges[key] = edge
            upserted += 1
            
        return UpsertResult(
            upserted_count=upserted,
            failed_count=len(failures),
            failures=failures,
        )

    async def _do_delete_nodes(
        self,
        spec: DeleteNodesSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> DeleteResult:
        """Delete nodes."""
        deleted = 0
        failures = []
        
        for node_id in spec.ids:
            key = f"{spec.namespace}:{node_id}"
            if key in self._nodes:
                del self._nodes[key]
                deleted += 1
            else:
                failures.append({
                    "index": len(failures),
                    "id": str(node_id),
                    "error": "NotFound",
                    "code": "NOT_FOUND",
                    "message": f"Node {node_id} not found",
                })
                
        return DeleteResult(
            deleted_count=deleted,
            failed_count=len(failures),
            failures=failures,
        )

    async def _do_delete_edges(
        self,
        spec: DeleteEdgesSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> DeleteResult:
        """Delete edges."""
        deleted = 0
        failures = []
        
        for edge_id in spec.ids:
            key = f"{spec.namespace}:{edge_id}"
            if key in self._edges:
                del self._edges[key]
                deleted += 1
            else:
                failures.append({
                    "index": len(failures),
                    "id": str(edge_id),
                    "error": "NotFound",
                    "code": "NOT_FOUND",
                    "message": f"Edge {edge_id} not found",
                })
                
        return DeleteResult(
            deleted_count=deleted,
            failed_count=len(failures),
            failures=failures,
        )

    async def _do_batch(
        self,
        ops: List[BatchOperation],
        *,
        ctx: Optional[OperationContext] = None,
    ) -> BatchResult:
        """Execute batch operations."""
        results = []
        success = True
        
        for op in ops:
            try:
                if op.op == "upsert_nodes":
                    nodes = [Node(**n) for n in op.args.get("nodes", [])]
                    spec = UpsertNodesSpec(nodes=nodes, namespace=op.args.get("namespace"))
                    result = await self._do_upsert_nodes(spec, ctx=ctx)
                    results.append(result)
                elif op.op == "upsert_edges":
                    edges = [Edge(**e) for e in op.args.get("edges", [])]
                    spec = UpsertEdgesSpec(edges=edges, namespace=op.args.get("namespace"))
                    result = await self._do_upsert_edges(spec, ctx=ctx)
                    results.append(result)
                else:
                    results.append({"success": True})
            except Exception as e:
                success = False
                results.append({"error": str(e)})
                
        return BatchResult(
            results=results,
            success=success,
            transaction_id="batch-123" if success else None,
        )

    async def _do_transaction(
        self,
        operations: List[BatchOperation],
        *,
        ctx: Optional[OperationContext] = None,
    ) -> BatchResult:
        """Execute atomic transaction."""
        # Simulate atomic commit
        result = await self._do_batch(operations, ctx=ctx)
        
        # ✅ CORRECT: Invalidate AFTER successful commit
        if result.success:
            namespaces = set()
            for op in operations:
                ns = op.args.get("namespace")
                if ns:
                    namespaces.add(ns)
            for ns in namespaces:
                await self._invalidate_namespace_cache(ns)
                
        return result

    async def _do_traversal(
        self,
        spec: GraphTraversalSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> TraversalResult:
        """Execute graph traversal."""
        nodes = [
            Node(
                id=GraphID("node-1"),
                labels=("Person",),
                properties={"name": "Alice"},
                namespace=spec.namespace,
            )
        ]
        
        edges = [
            Edge(
                id=GraphID("edge-1"),
                src=GraphID("node-1"),
                dst=GraphID("node-2"),
                label="KNOWS",
                namespace=spec.namespace,
            )
        ]

        return TraversalResult(
            nodes=nodes,
            relationships=edges,
            paths=[[nodes[0], edges[0]]],
            summary={"depth": spec.max_depth},
            namespace=spec.namespace,
        )

    async def _do_get_schema(
        self,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> GraphSchema:
        """Return graph schema."""
        return GraphSchema(
            nodes={"Person": {"properties": ["name", "age"]}},
            edges={"KNOWS": {"properties": ["since"]}},
            metadata={"version": "1.0"},
        )

    async def _do_health(
        self,
        *,
        ctx: Optional[OperationContext] = None
    ) -> Dict[str, Any]:
        """Health check with graded status."""
        return {
            "ok": True,
            "status": "ok",
            "server": "hello-graph",
            "version": "1.0.0",
            "namespaces": {"default": "ready"},
        }

    # ------------------------------------------------------------------------
    # TEST FIXTURE SUPPORT (REQUIRED FOR CERTIFICATION)
    # ------------------------------------------------------------------------
    
    def build_graph_query_envelope(
        self,
        query: str = "MATCH (n) RETURN n",
        dialect: str = "cypher",
        namespace: str = "default",
        **kwargs
    ) -> Dict[str, Any]:
        """Return wire envelope for graph query."""
        return {
            "op": "graph.query",
            "ctx": {
                "request_id": "test-123",
                "tenant": "test-tenant",
                "deadline_ms": 5000,
            },
            "args": {
                "text": query,
                "dialect": dialect,
                "namespace": namespace,
                **kwargs
            }
        }

    def build_graph_stream_query_envelope(
        self,
        query: str = "MATCH (n) RETURN n",
        dialect: str = "cypher",
        namespace: str = "default",
        **kwargs
    ) -> Dict[str, Any]:
        """Return wire envelope for streaming graph query."""
        return {
            "op": "graph.stream_query",
            "ctx": {
                "request_id": "test-123",
                "tenant": "test-tenant",
                "deadline_ms": 5000,
            },
            "args": {
                "text": query,
                "dialect": dialect,
                "namespace": namespace,
                **kwargs
            }
        }

    def build_graph_upsert_nodes_envelope(
        self,
        namespace: str = "default",
        nodes: List[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Return wire envelope for node upsert."""
        if nodes is None:
            nodes = [{
                "id": "node-1",
                "labels": ["Person"],
                "properties": {"name": "Alice"}
            }]
        return {
            "op": "graph.upsert_nodes",
            "ctx": {
                "request_id": "test-123",
                "tenant": "test-tenant",
                "deadline_ms": 5000,
            },
            "args": {
                "namespace": namespace,
                "nodes": nodes,
                **kwargs
            }
        }

    def build_graph_upsert_edges_envelope(
        self,
        namespace: str = "default",
        edges: List[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Return wire envelope for edge upsert."""
        if edges is None:
            edges = [{
                "id": "edge-1",
                "src": "node-1",
                "dst": "node-2",
                "label": "KNOWS",
                "properties": {"since": 2020}
            }]
        return {
            "op": "graph.upsert_edges",
            "ctx": {
                "request_id": "test-123",
                "tenant": "test-tenant",
                "deadline_ms": 5000,
            },
            "args": {
                "namespace": namespace,
                "edges": edges,
                **kwargs
            }
        }

    def build_graph_delete_nodes_envelope(
        self,
        namespace: str = "default",
        ids: List[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Return wire envelope for node deletion."""
        if ids is None:
            ids = ["node-1"]
        return {
            "op": "graph.delete_nodes",
            "ctx": {
                "request_id": "test-123",
                "tenant": "test-tenant",
                "deadline_ms": 5000,
            },
            "args": {
                "namespace": namespace,
                "ids": ids,
                **kwargs
            }
        }

    def build_graph_delete_edges_envelope(
        self,
        namespace: str = "default",
        ids: List[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Return wire envelope for edge deletion."""
        if ids is None:
            ids = ["edge-1"]
        return {
            "op": "graph.delete_edges",
            "ctx": {
                "request_id": "test-123",
                "tenant": "test-tenant",
                "deadline_ms": 5000,
            },
            "args": {
                "namespace": namespace,
                "ids": ids,
                **kwargs
            }
        }

    def build_graph_capabilities_envelope(self, **kwargs) -> Dict[str, Any]:
        """Return wire envelope for capabilities discovery."""
        return {
            "op": "graph.capabilities",
            "ctx": {"request_id": "test-123", "tenant": "test-tenant"},
            "args": {}
        }

    def build_graph_health_envelope(self, **kwargs) -> Dict[str, Any]:
        """Return wire envelope for health check."""
        return {
            "op": "graph.health",
            "ctx": {"request_id": "test-123", "tenant": "test-tenant"},
            "args": {}
        }

    def build_graph_batch_envelope(
        self,
        ops: List[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Return wire envelope for batch operations."""
        if ops is None:
            ops = [{
                "op": "upsert_nodes",
                "args": {"nodes": [{"id": "node-1"}]}
            }]
        return {
            "op": "graph.batch",
            "ctx": {
                "request_id": "test-123",
                "tenant": "test-tenant",
                "deadline_ms": 5000,
            },
            "args": {
                "ops": ops,
                **kwargs
            }
        }

    def build_graph_transaction_envelope(
        self,
        operations: List[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Return wire envelope for atomic transaction."""
        if operations is None:
            operations = [{
                "op": "upsert_nodes",
                "args": {"nodes": [{"id": "node-1"}]}
            }]
        return {
            "op": "graph.transaction",
            "ctx": {
                "request_id": "test-123",
                "tenant": "test-tenant",
                "deadline_ms": 5000,
            },
            "args": {
                "operations": operations,
                **kwargs
            }
        }

    def build_graph_traversal_envelope(
        self,
        start_nodes: List[str] = None,
        max_depth: int = 2,
        direction: str = "OUTGOING",
        namespace: str = "default",
        **kwargs
    ) -> Dict[str, Any]:
        """Return wire envelope for graph traversal."""
        if start_nodes is None:
            start_nodes = ["node-1"]
        return {
            "op": "graph.traversal",
            "ctx": {
                "request_id": "test-123",
                "tenant": "test-tenant",
                "deadline_ms": 5000,
            },
            "args": {
                "start_nodes": start_nodes,
                "max_depth": max_depth,
                "direction": direction,
                "namespace": namespace,
                **kwargs
            }
        }

    def build_graph_get_schema_envelope(self, **kwargs) -> Dict[str, Any]:
        """Return wire envelope for schema retrieval."""
        return {
            "op": "graph.get_schema",
            "ctx": {"request_id": "test-123", "tenant": "test-tenant"},
            "args": {}
        }
```

---

## 4. Testing Your Adapter (Certification Suite)

Now run the official certification suite against your adapter. **Choose the section that matches your protocol.**

---

### 4.1 Embedding Certification

**Test fixture** (`tests/conftest.py`):
```python
import pytest
from adapters.hello_embedding import HelloEmbeddingAdapter

@pytest.fixture
def adapter():
    return HelloEmbeddingAdapter(mode="thin")
```

**Run tests:**
```bash
# Test embedding protocol only
pytest tests/embedding/ -v

# Test with specific adapter via environment variable
export CORPUS_ADAPTER=adapters.hello_embedding:HelloEmbeddingAdapter
pytest tests/embedding/ -v

# Test wire envelope conformance
pytest tests/live/ -v

# Test schema validation
pytest tests/schema/ -v

# Incremental test order
pytest tests/embedding/test_capabilities.py -v
pytest tests/embedding/test_embed.py -v
pytest tests/embedding/test_streaming.py -v
pytest tests/embedding/test_batch.py -v
pytest tests/embedding/test_deadlines.py -v
pytest tests/embedding/test_errors.py -v
```

---

### 4.2 LLM Certification

**Test fixture** (`tests/conftest.py`):
```python
import pytest
from adapters.hello_llm import HelloLLMAdapter

@pytest.fixture
def adapter():
    return HelloLLMAdapter(mode="thin")
```

**Run tests:**
```bash
# Test LLM protocol
pytest tests/llm/ -v

# Incremental test order
pytest tests/llm/test_capabilities.py -v
pytest tests/llm/test_complete.py -v
pytest tests/llm/test_stream.py -v
pytest tests/llm/test_tools.py -v
pytest tests/llm/test_deadlines.py -v
pytest tests/llm/test_errors.py -v
```

---

### 4.3 Vector Certification

**Test fixture** (`tests/conftest.py`):
```python
import pytest
from adapters.hello_vector import HelloVectorAdapter

@pytest.fixture
def adapter():
    return HelloVectorAdapter(mode="thin")
```

**Run tests:**
```bash
# Test vector protocol
pytest tests/vector/ -v

# Incremental test order
pytest tests/vector/test_capabilities.py -v
pytest tests/vector/test_query.py -v
pytest tests/vector/test_upsert.py -v
pytest tests/vector/test_delete.py -v
pytest tests/vector/test_namespace.py -v
pytest tests/vector/test_batch_query.py -v
pytest tests/vector/test_deadlines.py -v
```

---

### 4.4 Graph Certification

**Test fixture** (`tests/conftest.py`):
```python
import pytest
from adapters.hello_graph import HelloGraphAdapter

@pytest.fixture
def adapter():
    return HelloGraphAdapter(mode="thin")
```

**Run tests:**
```bash
# Test graph protocol
pytest tests/graph/ -v

# Incremental test order
pytest tests/graph/test_capabilities.py -v
pytest tests/graph/test_query.py -v
pytest tests/graph/test_stream_query.py -v
pytest tests/graph/test_nodes.py -v
pytest tests/graph/test_edges.py -v
pytest tests/graph/test_batch.py -v
pytest tests/graph/test_transaction.py -v
pytest tests/graph/test_traversal.py -v
pytest tests/graph/test_schema.py -v
pytest tests/graph/test_deadlines.py -v
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
⏱️ Completed in 2.3s
```

### Certification Tiers

| Tier | Score | Meaning | Production Ready? |
|------|-------|---------|------------------|
| 🥇 **Gold** | 100% in SINGLE protocol | Ready for focused deployment | ✅ Yes |
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
      Examples: See build_embedding_embed_envelope() in reference adapter
```

**Each failure includes:**
- **Specification section** (§4.1, §7.2, etc.)
- **Quick fix** - Exactly what to change
- **Examples** - Where to look for correct implementation

**Do not guess.** The error guidance is authoritative.

For complete certification requirements, see [`CONFORMANCE_GUIDE.md`](CONFORMANCE_GUIDE.md).

---

## 6. Expose It Over HTTP

Now wire your certified adapter to the wire protocol. **Choose the service implementation that matches your protocol.**

---

### 6.1 Embedding Service (FastAPI)

`services/embedding_service.py`:
```python
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import json
import time
from corpus_sdk.embedding.embedding_base import WireEmbeddingHandler
from adapters.hello_embedding import HelloEmbeddingAdapter

app = FastAPI(title="Corpus Embedding Adapter")
adapter = HelloEmbeddingAdapter(mode="standalone")
handler = WireEmbeddingHandler(adapter)

@app.post("/v1/embedding")
async def handle_embedding(request: Request):
    """Handle unary embedding operations."""
    # Protocol version negotiation
    protocol = request.headers.get("x-adapter-protocol")
    if protocol and protocol != "embedding/v1.0":
        return JSONResponse(
            status_code=400,
            content={
                "ok": False,
                "code": "NOT_SUPPORTED",
                "error": "NotSupported",
                "message": f"Protocol {protocol} not supported. Use embedding/v1.0",
                "ms": 0,
                "details": {"supported_protocols": ["embedding/v1.0"]}
            }
        )
    
    try:
        envelope = await request.json()
        start = time.time()
        response = await handler.handle(envelope)
        if "ms" not in response:
            response["ms"] = (time.time() - start) * 1000
        return response
    except Exception as e:
        return {
            "ok": False,
            "code": "UNAVAILABLE",
            "error": "Unavailable",
            "message": "internal error",
            "ms": (time.time() - start) * 1000,
            "retry_after_ms": None,
            "details": None,
        }

@app.post("/v1/embedding/stream")
async def handle_embedding_stream(request: Request):
    """Handle streaming embedding operations."""
    protocol = request.headers.get("x-adapter-protocol")
    if protocol and protocol != "embedding/v1.0":
        return JSONResponse(status_code=400, content={
            "ok": False, "code": "NOT_SUPPORTED",
            "message": f"Protocol {protocol} not supported"
        })
    
    try:
        envelope = await request.json()
        
        async def stream_generator():
            start_time = time.time()
            async for chunk in handler.handle_stream(envelope):
                if isinstance(chunk, dict) and "chunk" not in chunk:
                    chunk = {
                        "ok": True,
                        "code": "STREAMING",
                        "ms": (time.time() - start_time) * 1000,
                        "chunk": chunk
                    }
                yield json.dumps(chunk) + "\n"
        
        return StreamingResponse(
            stream_generator(),
            media_type="application/x-ndjson",
            headers={
                "X-Adapter-Protocol": "embedding/v1.0",
                "Cache-Control": "no-cache"
            }
        )
    except Exception as e:
        return {
            "ok": False,
            "code": "UNAVAILABLE",
            "error": "Unavailable",
            "message": "internal error",
            "ms": 0,
        }

@app.get("/v1/health")
async def health_check():
    """Health endpoint with graded status."""
    health = await adapter.health()
    status_code = 200 if health.get("status") in ["ok", "degraded"] else 503
    return JSONResponse(status_code=status_code, content=health)

@app.get("/v1/stats")
async def get_stats():
    """Service statistics endpoint."""
    stats = await adapter._do_get_stats()
    return stats

@app.get("/v1/capabilities")
async def get_capabilities():
    """Capabilities discovery endpoint."""
    caps = await adapter.capabilities()
    from dataclasses import asdict
    return asdict(caps)
```

---

### 6.2 LLM Service (FastAPI)

`services/llm_service.py`:
```python
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import json
import time
from corpus_sdk.llm.llm_base import WireLLMHandler
from adapters.hello_llm import HelloLLMAdapter

app = FastAPI(title="Corpus LLM Adapter")
adapter = HelloLLMAdapter(mode="standalone")
handler = WireLLMHandler(adapter)

@app.post("/v1/llm/complete")
async def handle_complete(request: Request):
    """Handle LLM completion requests."""
    protocol = request.headers.get("x-adapter-protocol")
    if protocol and protocol != "llm/v1.0":
        return JSONResponse(status_code=400, content={
            "ok": False, "code": "NOT_SUPPORTED",
            "message": f"Protocol {protocol} not supported"
        })
    
    try:
        envelope = await request.json()
        start = time.time()
        response = await handler.handle(envelope)
        if "ms" not in response:
            response["ms"] = (time.time() - start) * 1000
        return response
    except Exception:
        return {"ok": False, "code": "UNAVAILABLE", "message": "internal error", "ms": 0}

@app.post("/v1/llm/stream")
async def handle_stream(request: Request):
    """Handle streaming LLM completions."""
    try:
        envelope = await request.json()
        
        async def stream_generator():
            start = time.time()
            async for chunk in handler.handle_stream(envelope):
                if "ms" not in chunk:
                    chunk["ms"] = (time.time() - start) * 1000
                yield json.dumps(chunk) + "\n"
        
        return StreamingResponse(
            stream_generator(),
            media_type="application/x-ndjson",
            headers={"X-Adapter-Protocol": "llm/v1.0"}
        )
    except Exception:
        return {"ok": False, "code": "UNAVAILABLE", "message": "internal error", "ms": 0}

@app.get("/v1/health")
async def health_check():
    health = await adapter.health()
    status_code = 200 if health.get("status") in ["ok", "degraded"] else 503
    return JSONResponse(status_code=status_code, content=health)

@app.post("/v1/llm/count_tokens")
async def count_tokens(request: Request):
    """Count tokens in text."""
    try:
        envelope = await request.json()
        start = time.time()
        response = await handler.handle(envelope)
        if "ms" not in response:
            response["ms"] = (time.time() - start) * 1000
        return response
    except Exception:
        return {"ok": False, "code": "UNAVAILABLE", "ms": 0}

@app.get("/v1/capabilities")
async def get_capabilities():
    caps = await adapter.capabilities()
    from dataclasses import asdict
    return asdict(caps)
```

---

### 6.3 Vector Service (FastAPI)

`services/vector_service.py`:
```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import time
from corpus_sdk.vector.vector_base import WireVectorHandler
from adapters.hello_vector import HelloVectorAdapter

app = FastAPI(title="Corpus Vector Adapter")
adapter = HelloVectorAdapter(mode="standalone")
handler = WireVectorHandler(adapter)

@app.post("/v1/vector/query")
async def handle_query(request: Request):
    """Execute vector similarity search."""
    protocol = request.headers.get("x-adapter-protocol")
    if protocol and protocol != "vector/v1.0":
        return JSONResponse(status_code=400, content={
            "ok": False, "code": "NOT_SUPPORTED",
            "message": f"Protocol {protocol} not supported"
        })
    
    try:
        envelope = await request.json()
        start = time.time()
        response = await handler.handle(envelope)
        if "ms" not in response:
            response["ms"] = (time.time() - start) * 1000
        return response
    except Exception:
        return {"ok": False, "code": "UNAVAILABLE", "message": "internal error", "ms": 0}

@app.post("/v1/vector/upsert")
async def handle_upsert(request: Request):
    """Upsert vectors."""
    try:
        envelope = await request.json()
        start = time.time()
        response = await handler.handle(envelope)
        if "ms" not in response:
            response["ms"] = (time.time() - start) * 1000
        return response
    except Exception:
        return {"ok": False, "code": "UNAVAILABLE", "ms": 0}

@app.post("/v1/vector/delete")
async def handle_delete(request: Request):
    """Delete vectors."""
    try:
        envelope = await request.json()
        start = time.time()
        response = await handler.handle(envelope)
        if "ms" not in response:
            response["ms"] = (time.time() - start) * 1000
        return response
    except Exception:
        return {"ok": False, "code": "UNAVAILABLE", "ms": 0}

@app.post("/v1/vector/batch_query")
async def handle_batch_query(request: Request):
    """Execute batch vector queries."""
    try:
        envelope = await request.json()
        start = time.time()
        response = await handler.handle(envelope)
        if "ms" not in response:
            response["ms"] = (time.time() - start) * 1000
        return response
    except Exception:
        return {"ok": False, "code": "UNAVAILABLE", "ms": 0}

@app.post("/v1/vector/create_namespace")
async def handle_create_namespace(request: Request):
    """Create namespace."""
    try:
        envelope = await request.json()
        start = time.time()
        response = await handler.handle(envelope)
        if "ms" not in response:
            response["ms"] = (time.time() - start) * 1000
        return response
    except Exception:
        return {"ok": False, "code": "UNAVAILABLE", "ms": 0}

@app.post("/v1/vector/delete_namespace")
async def handle_delete_namespace(request: Request):
    """Delete namespace."""
    try:
        envelope = await request.json()
        start = time.time()
        response = await handler.handle(envelope)
        if "ms" not in response:
            response["ms"] = (time.time() - start) * 1000
        return response
    except Exception:
        return {"ok": False, "code": "UNAVAILABLE", "ms": 0}

@app.get("/v1/health")
async def health_check():
    health = await adapter.health()
    status_code = 200 if health.get("status") in ["ok", "degraded"] else 503
    return JSONResponse(status_code=status_code, content=health)

@app.get("/v1/capabilities")
async def get_capabilities():
    caps = await adapter.capabilities()
    from dataclasses import asdict
    return asdict(caps)
```

---

### 6.4 Graph Service (FastAPI)

`services/graph_service.py`:
```python
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import json
import time
from corpus_sdk.graph.graph_base import WireGraphHandler
from adapters.hello_graph import HelloGraphAdapter

app = FastAPI(title="Corpus Graph Adapter")
adapter = HelloGraphAdapter(mode="standalone")
handler = WireGraphHandler(adapter)

@app.post("/v1/graph/query")
async def handle_query(request: Request):
    """Execute graph query."""
    protocol = request.headers.get("x-adapter-protocol")
    if protocol and protocol != "graph/v1.0":
        return JSONResponse(status_code=400, content={
            "ok": False, "code": "NOT_SUPPORTED",
            "message": f"Protocol {protocol} not supported"
        })
    
    try:
        envelope = await request.json()
        start = time.time()
        response = await handler.handle(envelope)
        if "ms" not in response:
            response["ms"] = (time.time() - start) * 1000
        return response
    except Exception:
        return {"ok": False, "code": "UNAVAILABLE", "message": "internal error", "ms": 0}

@app.post("/v1/graph/stream")
async def handle_stream_query(request: Request):
    """Stream graph query results."""
    try:
        envelope = await request.json()
        
        async def stream_generator():
            start = time.time()
            async for chunk in handler.handle_stream(envelope):
                if "ms" not in chunk:
                    chunk["ms"] = (time.time() - start) * 1000
                yield json.dumps(chunk) + "\n"
        
        return StreamingResponse(
            stream_generator(),
            media_type="application/x-ndjson",
            headers={"X-Adapter-Protocol": "graph/v1.0"}
        )
    except Exception:
        return {"ok": False, "code": "UNAVAILABLE", "ms": 0}

@app.post("/v1/graph/upsert_nodes")
async def handle_upsert_nodes(request: Request):
    """Upsert nodes."""
    try:
        envelope = await request.json()
        start = time.time()
        response = await handler.handle(envelope)
        if "ms" not in response:
            response["ms"] = (time.time() - start) * 1000
        return response
    except Exception:
        return {"ok": False, "code": "UNAVAILABLE", "ms": 0}

@app.post("/v1/graph/upsert_edges")
async def handle_upsert_edges(request: Request):
    """Upsert edges."""
    try:
        envelope = await request.json()
        start = time.time()
        response = await handler.handle(envelope)
        if "ms" not in response:
            response["ms"] = (time.time() - start) * 1000
        return response
    except Exception:
        return {"ok": False, "code": "UNAVAILABLE", "ms": 0}

@app.post("/v1/graph/delete_nodes")
async def handle_delete_nodes(request: Request):
    """Delete nodes."""
    try:
        envelope = await request.json()
        start = time.time()
        response = await handler.handle(envelope)
        if "ms" not in response:
            response["ms"] = (time.time() - start) * 1000
        return response
    except Exception:
        return {"ok": False, "code": "UNAVAILABLE", "ms": 0}

@app.post("/v1/graph/delete_edges")
async def handle_delete_edges(request: Request):
    """Delete edges."""
    try:
        envelope = await request.json()
        start = time.time()
        response = await handler.handle(envelope)
        if "ms" not in response:
            response["ms"] = (time.time() - start) * 1000
        return response
    except Exception:
        return {"ok": False, "code": "UNAVAILABLE", "ms": 0}

@app.post("/v1/graph/batch")
async def handle_batch(request: Request):
    """Execute batch operations."""
    try:
        envelope = await request.json()
        start = time.time()
        response = await handler.handle(envelope)
        if "ms" not in response:
            response["ms"] = (time.time() - start) * 1000
        return response
    except Exception:
        return {"ok": False, "code": "UNAVAILABLE", "ms": 0}

@app.post("/v1/graph/transaction")
async def handle_transaction(request: Request):
    """Execute atomic transaction."""
    try:
        envelope = await request.json()
        start = time.time()
        response = await handler.handle(envelope)
        if "ms" not in response:
            response["ms"] = (time.time() - start) * 1000
        return response
    except Exception:
        return {"ok": False, "code": "UNAVAILABLE", "ms": 0}

@app.post("/v1/graph/traversal")
async def handle_traversal(request: Request):
    """Execute graph traversal."""
    try:
        envelope = await request.json()
        start = time.time()
        response = await handler.handle(envelope)
        if "ms" not in response:
            response["ms"] = (time.time() - start) * 1000
        return response
    except Exception:
        return {"ok": False, "code": "UNAVAILABLE", "ms": 0}

@app.get("/v1/graph/schema")
async def get_schema(request: Request):
    """Get graph schema."""
    try:
        envelope = {"op": "graph.get_schema", "ctx": {}, "args": {}}
        start = time.time()
        response = await handler.handle(envelope)
        if "ms" not in response:
            response["ms"] = (time.time() - start) * 1000
        return response
    except Exception:
        return {"ok": False, "code": "UNAVAILABLE", "ms": 0}

@app.get("/v1/health")
async def health_check():
    health = await adapter.health()
    status_code = 200 if health.get("status") in ["ok", "degraded"] else 503
    return JSONResponse(status_code=status_code, content=health)

@app.get("/v1/capabilities")
async def get_capabilities():
    caps = await adapter.capabilities()
    from dataclasses import asdict
    return asdict(caps)
```

---

### Validate Your HTTP Endpoint

```bash
# Start your service (choose the right one)
uvicorn services.embedding_service:app --port 8000
# OR
uvicorn services.llm_service:app --port 8000
# OR
uvicorn services.vector_service:app --port 8000
# OR
uvicorn services.graph_service:app --port 8000

# Test with protocol header
curl -X POST http://localhost:8000/v1/embedding \
  -H "X-Adapter-Protocol: embedding/v1.0" \
  -H "Content-Type: application/json" \
  -d '{
    "op": "embedding.embed",
    "ctx": {"request_id": "test-123", "tenant": "acme"},
    "args": {"model": "hello-1", "text": "Hello world"}
  }'

# Test against wire conformance suite
export CORPUS_ENDPOINT=http://localhost:8000
pytest tests/live/ -v
```

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

# ✅ REQUIRED: canonical streaming envelope
{
    "ok": True,
    "code": "STREAMING",  # Exactly "STREAMING"
    "ms": 12.34,
    "chunk": {...}
}

# ✅ REQUIRED: idempotency key deduplication (24h)
if ctx.idempotency_key:
    cached = await redis.get(f"idem:{ctx.tenant}:{ctx.idempotency_key}")
    if cached: return cached
```

Full specification: [`corpus_sdk/embedding/embedding_base.py`](../corpus_sdk/embedding/embedding_base.py)

---

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

Full specification: [`corpus_sdk/llm/llm_base.py`](../corpus_sdk/llm/llm_base.py)

---

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

# ✅ REQUIRED: Docstore hydration is best-effort, never fail query
try:
    text = await docstore.get(doc_id)
except Exception:
    text = None  # Graceful degradation
```

Full specification: [`corpus_sdk/vector/vector_base.py`](../corpus_sdk/vector/vector_base.py)

---

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

Full specification: [`corpus_sdk/graph/graph_base.py`](../corpus_sdk/graph/graph_base.py)

---

## 8. Production Readiness

### Mode Decision Tree

```
Is adapter deployed behind a control plane with its own circuit breakers, rate limiters, and cache?
            │
            ├─ YES → mode="thin" (no-op policies, fastest)
            │
            └─ NO  → mode="standalone" (default for most deployments)
                     • SimpleDeadline (timeouts)
                     • SimpleCircuitBreaker (5 failures → 10s cooldown)
                     • InMemoryTTLCache (read paths, 60s TTL)
                     • TokenBucketLimiter (50 req/sec, burst 100)
```

### Deadline Propagation (Required for All Protocols)

```python
# ALWAYS use ctx.remaining_ms() per specification §6.1
timeout = None
if ctx and ctx.deadline_ms:
    remaining = ctx.remaining_ms()
    if remaining > 0:
        timeout = remaining / 1000.0
    elif remaining == 0:
        raise DeadlineExceeded("deadline expired")
```

### Idempotency Key Storage (Embedding Only)

```python
# MUST deduplicate identical requests for at least 24 hours per §6.1
async def _check_idempotency(ctx):
    if ctx and ctx.idempotency_key and ctx.tenant:
        key = f"idem:{ctx.tenant}:{ctx.idempotency_key}"
        cached = await redis.get(key)
        if cached:
            return cached
    return None
```

### Tenant Isolation (Required for All Protocols)

```python
# ✅ CORRECT: Use tenant for routing
tenant_id = ctx.tenant if ctx else "default"

# ❌ WRONG: Logging raw tenant IDs (per §14.1, §15)
logger.info(f"Query from {ctx.tenant}")  # NEVER DO THIS

# ✅ CORRECT: Hashed tenant for debugging
logger.debug(f"tenant_hash={self._tenant_hash(ctx.tenant)}")
```

### Cache Invalidation (Vector/Graph Only)

```python
# ✅ CORRECT: Invalidate AFTER successful write (Vector)
result = await self._do_upsert(spec, ctx)
if result.upserted_count > 0:
    await self._invalidate_namespace_cache(spec.namespace)

# ✅ CORRECT: Transaction - invalidate AFTER commit (Graph)
if await txn.commit():
    await self._invalidate_namespace_cache(namespace)
```

---

## 9. What to Read Next

| Document | Purpose | Location |
|----------|---------|----------|
| **Protocol Specifications** | Normative behavior for each protocol | [`corpus_sdk/embedding/embedding_base.py`](../corpus_sdk/embedding/embedding_base.py)<br>[`corpus_sdk/llm/llm_base.py`](../corpus_sdk/llm/llm_base.py)<br>[`corpus_sdk/vector/vector_base.py`](../corpus_sdk/vector/vector_base.py)<br>[`corpus_sdk/graph/graph_base.py`](../corpus_sdk/graph/graph_base.py) |
| **Implementation Guide** | Deep dive on `_do_*` semantics | [`IMPLEMENTATION.md`](IMPLEMENTATION.md) |
| **Conformance Guide** | Running certification suites | [`CONFORMANCE_GUIDE.md`](CONFORMANCE_GUIDE.md) |
| **Adapter Recipes** | Multi-cloud and RAG scenarios | [`ADAPTER_RECIPES.md`](ADAPTER_RECIPES.md) *(coming soon)* |
| **Error Taxonomy** | Complete error hierarchy | [`ERRORS.md`](ERRORS.md) *(in spec docstrings)* |
| **Metrics Schema** | SIEM-safe observability | [`METRICS.md`](METRICS.md) *(in `MetricsSink` protocol)* |

**The conformance tests in `tests/` are the source of truth.** When this document and the tests disagree, **the tests are correct.**

---

## 10. Adapter Launch Checklist

### 10.1 Universal Requirements (All Protocols)

- [ ] **REQUIRED:** Constructor accepts `endpoint=None`
- [ ] **REQUIRED:** `_do_capabilities()` declares `protocol="{component}/v1.0"`
- [ ] **REQUIRED:** All `build_*_envelope()` methods implemented for your protocol
- [ ] **REQUIRED:** `ctx.remaining_ms()` used in all `_do_*` methods
- [ ] **REQUIRED:** No raw tenant IDs in logs/metrics (use `_tenant_hash()`)
- [ ] **REQUIRED:** Gold certification achieved: `pytest tests/{protocol}/ -v` shows 100% pass
- [ ] **REQUIRED:** Wire conformance tests pass: `pytest tests/live/ -v`
- [ ] **REQUIRED:** Schema validation tests pass: `pytest tests/schema/ -v`
- [ ] **RECOMMENDED:** `_do_get_stats()` implemented for service observability
- [ ] **RECOMMENDED:** Health endpoint returns graded `status: "ok"|"degraded"|"down"`
- [ ] **RECOMMENDED:** HTTP service validates `X-Adapter-Protocol` header

### 10.2 Embedding-Specific Checklist

- [ ] **REQUIRED:** `_do_capabilities()` declares `idempotent_writes=True`
- [ ] **REQUIRED:** Batch operations use field name `failures` (not `failed_texts`)
- [ ] **REQUIRED:** Batch success items include `index` field for correlation
- [ ] **REQUIRED:** Streaming operations use canonical `{ok,code,ms,chunk}` envelope
- [ ] **REQUIRED:** Idempotency keys deduplicated for ≥24 hours
- [ ] **REQUIRED:** `_do_count_tokens()` implemented
- [ ] **RECOMMENDED:** Streaming implemented (if `supports_streaming=True`)

### 10.3 LLM-Specific Checklist

- [ ] **REQUIRED:** `_do_capabilities()` declares `model_family` (not just `model`)
- [ ] **REQUIRED:** Never implement tool execution - only pass through tool calls
- [ ] **REQUIRED:** Tool calls include generated IDs (`secrets.token_hex()`)
- [ ] **REQUIRED:** Streaming includes `usage_so_far` in final chunk
- [ ] **REQUIRED:** `supports_system_message` accurately reflects capability
- [ ] **REQUIRED:** `_do_count_tokens()` implemented
- [ ] **RECOMMENDED:** Support for `stop_sequences`, `frequency_penalty`, `presence_penalty`
- [ ] **RECOMMENDED:** Tool calling support with `supports_tools=True`

### 10.4 Vector-Specific Checklist

- [ ] **REQUIRED:** Namespace canonicalization enforced (vector.namespace == spec.namespace)
- [ ] **REQUIRED:** Cache invalidation performed AFTER successful writes
- [ ] **REQUIRED:** Docstore hydration never fails the query (graceful degradation)
- [ ] **REQUIRED:** `max_dimensions` validated on upsert and query
- [ ] **REQUIRED:** Batch query support (if `supports_batch_queries=True`)
- [ ] **RECOMMENDED:** Metadata filtering support with `supports_metadata_filtering=True`
- [ ] **RECOMMENDED:** Auto-normalization for cosine similarity
- [ ] **RECOMMENDED:** RedisDocStore for production text storage

### 10.5 Graph-Specific Checklist

- [ ] **REQUIRED:** Cache invalidation performed ONLY after successful transaction commit
- [ ] **REQUIRED:** Query dialect validation against `supported_query_dialects`
- [ ] **REQUIRED:** Transaction support requires atomic batch operations
- [ ] **REQUIRED:** Batch operation success detection for targeted invalidation
- [ ] **REQUIRED:** `supports_transaction` accurately reflects capability
- [ ] **RECOMMENDED:** Traversal support with `supports_traversal=True`
- [ ] **RECOMMENDED:** Schema introspection with `supports_schema=True`
- [ ] **RECOMMENDED:** Bulk vertex scanning with `supports_bulk_vertices=True`

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

| Term | Definition | Location |
|------|------------|----------|
| **Adapter** | A class that implements `_do_*` hooks and `build_*_envelope()` methods | This guide |
| **Base Class** | `BaseEmbeddingAdapter`, `BaseLLMAdapter`, `BaseVectorAdapter`, `BaseGraphAdapter` | `corpus_sdk/*/*_base.py` |
| **Certification Suite** | The conformance tests in `tests/embedding/`, `tests/llm/`, etc. | `tests/` directory |
| **Gold Certification** | 100% pass rate in a single protocol | Section 5 |
| **build_*_envelope()** | Test fixture methods that return wire envelopes (REQUIRED) | Section 3 |
| **Wire Envelope** | The JSON `{op, ctx, args}` structure all Corpus services speak | Base class docstrings |
| **Canonical Streaming Envelope** | `{ok: true, code: "STREAMING", ms: number, chunk: object}` | `*_base.py` |
| **Protocol Field** | REQUIRED field in capabilities: `"protocol": "{component}/v1.0"` | `*Capabilities` classes |
| **Idempotent Writes** | REQUIRED capability for embedding: `idempotent_writes: true` | `EmbeddingCapabilities` |
| **Failures Field** | REQUIRED field name for embedding batch errors (not `failed_texts`) | `BatchEmbedResult` |
| **Model Family** | REQUIRED field in LLM capabilities: `model_family` | `LLMCapabilities` |
| **Namespace Canonicalization** | REQUIRED behavior for Vector: enforce namespace match | `BaseVectorAdapter` |
| **Transaction Atomicity** | REQUIRED for Graph: all or nothing | `BaseGraphAdapter` |
| **CORPUS_ADAPTER** | Environment variable: `module:ClassName` for dynamic loading | Section 4 |
| **CORPUS_ENDPOINT** | Environment variable: URL for live endpoint testing | Section 6 |
| **X-Adapter-Protocol** | Header for protocol version negotiation | Section 6 |

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
| `Missing build_*_envelope` | No test fixture methods | Implement all required `build_*_envelope()` methods |
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
| `Streaming envelope missing required fields` | Use `{ok,code,ms,chunk}` envelope |
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
| `Docstore failure caused query failure` | Make docstore hydration best-effort, catch exceptions |
| `Batch query not implemented` | Implement `_do_batch_query()` if `supports_batch_queries=True` |

**Graph:**
| Error | Fix |
|-------|-----|
| `Cache invalidation before commit` | Move invalidation AFTER successful commit |
| `Dialect not validated` | Check `spec.dialect` against `caps.supported_query_dialects` |
| `Transaction not atomic` | Ensure all operations in transaction commit or rollback together |
| `Batch success detection missing` | Implement `_batch_op_succeeded()` for targeted invalidation |

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

### Getting Help From the Suite

The certification suite's error messages are authoritative. When you see:

```
Specification: §4.2.2 Version Identification
Quick fix: Capabilities missing required field 'protocol'
```

**Do not search the internet.** Open the specification in the base class docstring and read §4.2.2. The answer is there.

---

**Maintainers:** Corpus SDK Team  
**Last Updated:** 2026-02-11  
**Scope:** Complete adapter authoring reference for all Corpus Protocols v1.0 (Embedding, LLM, Vector, Graph).

**The conformance tests in `tests/` are the source of truth.** When this document and the tests disagree, **the tests are correct.**

---
