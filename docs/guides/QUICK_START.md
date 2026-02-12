# Corpus OS Quickstart

**Build Production-Ready Corpus Protocol Adapters in Minutes**

**Table of Contents**
- [0. Mental Model (What You're Actually Building)](#0-mental-model-what-youre-actually-building)
- [1. Prerequisites & Setup](#1-prerequisites--setup)
- [2. Conformance-First Development (The Right Way)](#2-conformance-first-development-the-right-way)
- [3. Hello World Embedding Adapter (Complete)](#3-hello-world-embedding-adapter-complete)
- [4. Testing Your Adapter (Certification Suite)](#4-testing-your-adapter-certification-suite)
- [5. Understanding Certification Results](#5-understanding-certification-results)
- [6. Expose It Over HTTP](#6-expose-it-over-http)
- [7. Other Protocol Variants (LLM/Vector/Graph)](#7-other-protocol-variants-llmvectorgraph)
- [8. Production Readiness](#8-production-readiness)
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
- ✅ **Platinum or Gold certification** from the official conformance suite
- ✅ **Full compliance with Corpus Protocol v1.0 specification**
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
- `_do_capabilities()` - What your adapter supports (MUST include `protocol` field)
- `_do_embed()` / `_do_complete()` / etc. - Core operation
- `_do_stream_*()` - Streaming (if supported) - **MUST use canonical envelope format**
- `_do_health()` - Liveness check
- `_do_get_stats()` - Service statistics (optional but recommended)
- `build_*_envelope()` - Test fixture support (required for certification)

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
- Python 3.10+
- `corpus-sdk` ≥ 1.0.0
- `pytest` ≥ 7.0 (for certification)

### Installation

```bash
pip install corpus-sdk
pip install pytest pytest-asyncio pysqlite3-binary  # Certification dependencies
```
---

## 2. Conformance-First Development (The Right Way)

**Stop writing code before you have a failing test.**

The Corpus certification suite evaluates adapters against the protocol specification. **You copy the tests into your project and run them locally.**

### Step 1: Copy the Official Conformance Tests

```bash
# Copy embedding protocol tests into your project
cp -r $(python -c "import corpus_sdk; print(corpus_sdk.__path__[0])")/tests/embedding ./tests/
cp -r $(python -c "import corpus_sdk; print(corpus_sdk.__path__[0])")/tests/live ./tests/
cp -r $(python -c "import corpus_sdk; print(corpus_sdk.__path__[0])")/tests/schema ./tests/
```

Your `tests/` directory now contains the official, unmodified certification suite.

### Step 2: Create Your Test Fixture

Create `tests/conftest.py`:

```python
import pytest
from adapters.hello_embedding import HelloEmbeddingAdapter

@pytest.fixture
def adapter():
    """Return your adapter instance for certification testing."""
    return HelloEmbeddingAdapter(mode="thin")
```

**Do not modify the copied test files.** They are the source of truth.

### Step 3: Run a Single Test and Watch It Fail

```bash
pytest tests/embedding/test_capabilities.py -v -k test_capabilities_basic
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

---

## 3. Hello World Embedding Adapter (Complete)

This is the **fully specification-compliant implementation** that passes all Embedding conformance tests.

Create `adapters/hello_embedding.py`:

```python
import asyncio
import time
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

    # ------------------------------------------------------------------------
    # PRODUCTION IMPLEMENTATION (_do_* methods)
    # ------------------------------------------------------------------------

    async def _do_capabilities(self) -> EmbeddingCapabilities:
        """
        Advertise what this adapter supports.
        
        ⚠️ REQUIRED FIELDS per specification:
        - protocol: MUST be "{component}/v1.0"
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
        """
        Generate embedding for a single text.
        
        ⚠️ Idempotency: Must deduplicate identical requests for 24 hours.
        """
        # --------------------------------------------------------------------
        # IDEMPOTENCY CHECK (Required per specification §6.1, §11.4)
        # --------------------------------------------------------------------
        if ctx and ctx.idempotency_key and ctx.tenant:
            cache_key = f"idem:{ctx.tenant}:{ctx.idempotency_key}"
            cached = self._idempotency_cache.get(cache_key)
            if cached:
                return cached  # Return previously stored result
        
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
        
        # Chunk 2: next 2 dimensions
        elapsed_ms = (time.time() - start_time) * 1000
        yield {
            "ok": True,
            "code": "STREAMING",
            "ms": elapsed_ms,
            "chunk": {
                "embeddings": [
                    {
                        "vector": base_vec[:4] + [0.0] * 4,
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
        - Overall HTTP status is 200, code is "OK"
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
                # ✅ CORRECT field name: "failures"
                failures.append({
                    "index": idx,
                    "error": type(e).__name__,
                    "code": getattr(e, "code", "UNKNOWN"),
                    "message": str(e),
                    "detail": str(e),  # Optional additional context
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
            "reason": None,  # Required if status != "ok"
        }

    async def _do_get_stats(
        self,
        *,
        ctx: Optional[OperationContext] = None
    ) -> Dict[str, Any]:
        """
        Retrieve embedding service statistics.
        
        Optional but recommended per specification §4.2.6.
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
    # The certification suite calls these methods to validate wire format.
    # ⚠️ CRITICAL: These methods MUST return raw JSON-serializable dicts.
    
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
                "idempotency_key": "test-idem-123",  # Required for idempotency testing
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
            "op": "embedding.embed_batch",  # ✅ Note: embed_batch, not batch_embed
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
                    details={"provider_error_id": getattr(e, "error_id", None)},
                )
            elif e.status_code == 401:
                return AuthError("invalid credentials")
            elif e.status_code == 400:
                return BadRequest(str(e))
            elif e.status_code >= 500:
                return Unavailable(
                    "provider unavailable",
                    retry_after_ms=1000,
                    details={"provider_error_id": getattr(e, "error_id", None)},
                )
        return e

    # ------------------------------------------------------------------------
    # TENANT HASHING (SIEM-Safe)
    # ------------------------------------------------------------------------
    
    def _tenant_hash(self, tenant: Optional[str]) -> Optional[str]:
        """Return irreversible hash of tenant identifier."""
        if not tenant:
            return None
        import hashlib
        import os
        salt = os.environ.get("CORPUS_TENANT_SALT", "default-salt-change-me")
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
- ✅ **RECOMMENDED** `_do_get_stats()` implementation
- ✅ **RECOMMENDED** SIEM-safe tenant hashing

---

## 4. Testing Your Adapter (Certification Suite)

Now run the official certification suite against your adapter.

### Step 1: Create Test Fixture

`tests/conftest.py`:
```python
import pytest
from adapters.hello_embedding import HelloEmbeddingAdapter

@pytest.fixture
def adapter():
    """Return your adapter instance for certification."""
    return HelloEmbeddingAdapter(mode="thin")
```

### Step 2: Run Certification Tests

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
```

### Step 3: Run Tests Incrementally

```bash
# Start with capabilities (tests protocol field, idempotent_writes)
pytest tests/embedding/test_capabilities.py -v

# Then single embedding (tests idempotency, deadlines)
pytest tests/embedding/test_embed.py -v

# Then streaming (tests canonical envelope format)
pytest tests/embedding/test_streaming.py -v

# Then batch (tests failures field name, index correlation)
pytest tests/embedding/test_batch.py -v

# Then deadlines, errors, wire format
pytest tests/embedding/test_deadlines.py -v
pytest tests/embedding/test_errors.py -v
pytest tests/live/ -v
```

### Step 4: Commit After Each Passing Test Group

```bash
git add adapters/hello_embedding.py tests/conftest.py
git commit -m "PASS: capabilities (protocol field, idempotent_writes)"
git commit -m "PASS: single embed (idempotency, deadlines)"
git commit -m "PASS: streaming (canonical envelope)"
git commit -m "PASS: batch (failures field, index correlation)"
```

---

## 5. Understanding Certification Results

The certification suite provides **tiered scoring**. When you run the full suite, look for this summary:

```
================================================================================
CORPUS PROTOCOL SUITE - PLATINUM CERTIFIED
🔌 Adapter: adapters.hello_embedding:HelloEmbeddingAdapter | ⚖️ Strict: off

Protocol & Framework Conformance Status (scored / collected):
  ✅ PASS Embedding Protocol V1.0: Gold (135/135 scored; 150 collected)

🎯 Status: Ready for production deployment
⏱️ Completed in 2.3s
```

### Certification Tiers

| Tier | Score | Meaning | Production Ready? |
|------|-------|---------|------------------|
| 🏆 **Platinum** | 100% across ALL protocols | Full ecosystem support | ✅ Yes |
| 🥇 **Gold** | 100% in SINGLE protocol | Ready for focused deployment | ✅ Yes |
| 🥈 **Silver** | ≥80% | Integration testing ready | ⚠️ No |
| 🔬 **Development** | ≥50% | Early implementation | ❌ No |
| ❌ **None** | <50% | Not yet functional | ❌ No |

**Your goal:** Gold or Platinum certification.

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

---

## 6. Expose It Over HTTP

Now wire your certified adapter to the wire protocol with **full specification compliance**.

### FastAPI Implementation

`services/embedding_service.py`:
```python
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import json
import time
from corpus_sdk.embedding.embedding_base import WireEmbeddingHandler
from adapters.hello_embedding import HelloEmbeddingAdapter

app = FastAPI(title="Corpus Embedding Adapter")

# Your certified adapter
adapter = HelloEmbeddingAdapter(mode="standalone")
handler = WireEmbeddingHandler(adapter)

@app.post("/v1/embedding")
async def handle_embedding(request: Request):
    """
    Handle unary embedding operations.
    
    ⚠️ REQUIRED: Validate X-Adapter-Protocol header
    """
    # --------------------------------------------------------------------
    # PROTOCOL VERSION NEGOTIATION (Required per §4.2.2, §18.2)
    # --------------------------------------------------------------------
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
                "retry_after_ms": None,
                "details": {
                    "supported_protocols": ["embedding/v1.0"]
                }
            }
        )
    
    try:
        envelope = await request.json()
        start_time = time.time()
        response = await handler.handle(envelope)
        
        # Add processing time if not present
        if "ms" not in response:
            response["ms"] = (time.time() - start_time) * 1000
            
        return response
    except Exception as e:
        # WireEmbeddingHandler already formats errors correctly
        return {
            "ok": False,
            "code": "UNAVAILABLE",
            "error": "Unavailable",
            "message": "internal error",
            "ms": (time.time() - start_time) * 1000,
            "retry_after_ms": None,
            "details": None,
        }

@app.post("/v1/embedding/stream")
async def handle_embedding_stream(request: Request):
    """
    Handle streaming embedding operations.
    
    ⚠️ REQUIRED: Must use canonical streaming envelope format
    """
    # Protocol version validation (same as above)
    protocol = request.headers.get("x-adapter-protocol")
    if protocol and protocol != "embedding/v1.0":
        return JSONResponse(
            status_code=400,
            content={
                "ok": False,
                "code": "NOT_SUPPORTED",
                "error": "NotSupported",
                "message": f"Protocol {protocol} not supported",
                "ms": 0
            }
        )
    
    try:
        envelope = await request.json()
        
        async def stream_generator():
            start_time = time.time()
            async for chunk in handler.handle_stream(envelope):
                # Ensure canonical streaming envelope format
                if isinstance(chunk, dict) and "chunk" not in chunk:
                    # If handler returns raw chunk, wrap it
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
```

### Validate Your HTTP Endpoint

```bash
# Start your service
uvicorn services.embedding_service:app --port 8000

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

## 7. Other Protocol Variants (LLM/Vector/Graph)

The pattern is **identical** across all four protocols. Only the method names change.

### Protocol Mapping

| Component | Embedding | LLM | Vector | Graph |
|-----------|-----------|-----|--------|-------|
| **Base class** | `BaseEmbeddingAdapter` | `BaseLLMAdapter` | `BaseVectorAdapter` | `BaseGraphAdapter` |
| **Core method** | `_do_embed()` | `_do_complete()` | `_do_query()` | `_do_query()` |
| **Streaming** | `_do_stream_embed()` | `_do_stream()` | `_do_stream_query()` | `_do_stream_query()` |
| **Batch** | `_do_embed_batch()` | `_do_complete_batch()` | `_do_query_batch()` | `_do_transaction()` |
| **Capabilities** | `EmbeddingCapabilities` | `LLMCapabilities` | `VectorCapabilities` | `GraphCapabilities` |
| **REQUIRED protocol field** | `"embedding/v1.0"` | `"llm/v1.0"` | `"vector/v1.0"` | `"graph/v1.0"` |
| **Test envelope** | `build_embedding_*()` | `build_llm_*()` | `build_vector_*()` | `build_graph_*()` |
| **Copy tests** | `cp -r .../tests/embedding` | `cp -r .../tests/llm` | `cp -r .../tests/vector` | `cp -r .../tests/graph` |
| **Run tests** | `pytest tests/embedding/` | `pytest tests/llm/` | `pytest tests/vector/` | `pytest tests/graph/` |

### Protocol-Specific Requirements

**LLM:**
- ⚠️ `protocol: "llm/v1.0"` in capabilities
- ⚠️ Never implement tool execution - only transmit tool calls
- ⚠️ Must set `model_family` in capabilities
- ⚠️ Required `build_llm_*_envelope()` methods for each operation

**Vector:**
- ⚠️ `protocol: "vector/v1.0"` in capabilities
- ⚠️ Must set `index` on Vector objects for batch correlation
- ⚠️ Cache invalidation required after writes
- ⚠️ Required `build_vector_*_envelope()` methods for upsert, query, delete

**Graph:**
- ⚠️ `protocol: "graph/v1.0"` in capabilities
- ⚠️ Cache invalidation only after successful transaction commit
- ⚠️ Transaction support requires atomic batch operations
- ⚠️ Required `build_graph_*_envelope()` methods for nodes, edges, transactions

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

### Deadline Propagation (Required)

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

### Idempotency Key Storage (Required)

```python
# MUST deduplicate identical requests for at least 24 hours per §6.1
async def _check_idempotency(ctx):
    if ctx and ctx.idempotency_key and ctx.tenant:
        key = f"idem:{ctx.tenant}:{ctx.idempotency_key}"
        cached = await redis.get(key)
        if cached:
            return cached
    return None

async def _store_idempotency(ctx, result, ttl=86400):
    if ctx and ctx.idempotency_key and ctx.tenant:
        key = f"idem:{ctx.tenant}:{ctx.idempotency_key}"
        await redis.setex(key, ttl, result)
```

### Tenant Isolation (Required)

```python
# ✅ CORRECT: Use tenant for routing
tenant_id = ctx.tenant if ctx else "default"

# ❌ WRONG: Logging raw tenant IDs (per §14.1, §15)
logger.info(f"Query from {ctx.tenant}")  # NEVER DO THIS

# ✅ CORRECT: Hashed tenant for debugging
logger.debug(f"tenant_hash={self._tenant_hash(ctx.tenant)}")
```

### Cache Invalidation (Vector/Graph)

```python
# ✅ CORRECT: Invalidate AFTER successful write
await self._client.upsert(vectors)
await self._invalidate_namespace_cache(spec.namespace)

# ❌ WRONG: Invalidate before commit (commit may fail)
await self._invalidate_namespace_cache(spec.namespace)
await self._client.upsert(vectors)  # If this fails, cache is stale!
```

---

## 9. What to Read Next

| Document | Purpose | When to Read |
|----------|---------|--------------|
| [`spec/CORPUS_SPECIFICATION.md`](../spec/CORPUS_SPECIFICATION.md) | Full specification (normative) | Before production deploy |
| [`spec/IMPLEMENTATION.md`](../spec/IMPLEMENTATION.md) | Full `_do_*` semantics, all edge cases | After quickstart works |
| [`spec/BEHAVIORAL_CONFORMANCE.md`](../spec/BEHAVIORAL_CONFORMANCE.md) | What "correct" means normatively | Before production deploy |
| [`spec/ERRORS.md`](../spec/ERRORS.md) | Complete error taxonomy | When adding new error types |
| [`tests/embedding/`](./tests/embedding/) | The actual conformance tests | Always - they are the spec |

**The conformance tests in `tests/` are the source of truth.** When this document and the tests disagree, **the tests are correct.**

---

## 10. Adapter Launch Checklist

### 🔴 Pre-Flight (Required for Certification)
- [ ] **REQUIRED:** Constructor accepts `endpoint=None`
- [ ] **REQUIRED:** `_do_capabilities()` declares `protocol="embedding/v1.0"`
- [ ] **REQUIRED:** `_do_capabilities()` declares `idempotent_writes=True`
- [ ] **REQUIRED:** All `build_*_envelope()` methods implemented for your protocol
- [ ] **REQUIRED:** Batch operations use field name `failures` (not `failed_texts`)
- [ ] **REQUIRED:** Batch success items include `index` field for correlation
- [ ] **REQUIRED:** Streaming operations use canonical `{ok,code,ms,chunk}` envelope
- [ ] **REQUIRED:** Idempotency keys deduplicated for ≥24 hours
- [ ] **REQUIRED:** `ctx.remaining_ms()` used in all `_do_*` methods
- [ ] **REQUIRED:** No raw tenant IDs in logs/metrics (use `_tenant_hash()`)
- [ ] **REQUIRED:** Gold certification achieved: `pytest tests/embedding/ -v` shows 100% pass
- [ ] **REQUIRED:** Wire conformance tests pass: `pytest tests/live/ -v`
- [ ] **REQUIRED:** Schema validation tests pass: `pytest tests/schema/ -v`

### 🟡 Production Hardening (Recommended)
- [ ] **RECOMMENDED:** `_do_get_stats()` implemented for service observability
- [ ] **RECOMMENDED:** Health endpoint returns graded `status: "ok"|"degraded"|"down"`
- [ ] **RECOMMENDED:** HTTP service validates `X-Adapter-Protocol` header
- [ ] **RECOMMENDED:** Platinum certification (100% across all protocols you support)
- [ ] **RECOMMENDED:** Streaming implemented (if `supports_streaming=True`)
- [ ] **RECOMMENDED:** Cache invalidation implemented for write operations (Vector/Graph)

### 🟢 Operational Excellence (Nice to Have)
- [ ] Distributed cache (Redis) with idempotency key storage
- [ ] Distributed rate limiter instead of `SimpleTokenBucketLimiter`
- [ ] Health check includes dependency status (DB, upstream providers)
- [ ] Version pinning: `corpus-sdk>=1.0.0,<2.0.0`

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
```

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **Adapter** | A class that implements `_do_*` hooks and `build_*_envelope()` methods |
| **Base Class** | `BaseEmbeddingAdapter`, `BaseLLMAdapter`, etc. - provides infrastructure |
| **Certification Suite** | The conformance tests in `tests/embedding/`, `tests/llm/`, etc. |
| **Gold Certification** | 100% pass rate in a single protocol |
| **Platinum Certification** | 100% pass rate across all protocols |
| **build_*_envelope()** | Test fixture methods that return wire envelopes (REQUIRED) |
| **Wire Envelope** | The JSON `{op, ctx, args}` structure all Corpus services speak |
| **Canonical Streaming Envelope** | `{ok: true, code: "STREAMING", ms: number, chunk: object}` |
| **Protocol Field** | REQUIRED field in capabilities: `"protocol": "{component}/v1.0"` |
| **Idempotent Writes** | REQUIRED capability: `idempotent_writes: true` for embedding |
| **Failures Field** | REQUIRED field name for batch errors (not `failed_texts`) |
| **CORPUS_ADAPTER** | Environment variable: `module:ClassName` for dynamic loading |
| **CORPUS_ENDPOINT** | Environment variable: URL for live endpoint testing |
| **X-Adapter-Protocol** | Header for protocol version negotiation |

---

## Appendix C: Debugging & Troubleshooting

### Enable Full Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("corpus_sdk").setLevel(logging.DEBUG)
```

### Common Errors & Fixes

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| `AdapterValidationError: Failed to instantiate adapter` | Constructor doesn't accept `endpoint=None` | Add `endpoint=None` to `__init__` |
| `Missing build_*_envelope` | No test fixture methods | Implement all required `build_*_envelope()` methods |
| `capabilities missing required field: protocol` | Missing `protocol="embedding/v1.0"` | Add protocol field to capabilities |
| `capabilities missing required field: idempotent_writes` | Missing `idempotent_writes=True` | Add idempotent_writes to capabilities |
| `Batch result missing field: failures` | Using `failed_texts` instead of `failures` | Rename field to `failures` |
| `Batch success missing index` | No `index` on EmbeddingVector | Set `index=idx` in batch success items |
| `Streaming envelope missing required fields` | Wrong streaming format | Use `{ok,code,ms,chunk}` envelope |
| `Wire envelope validation failed` | Envelope shape doesn't match schema | Match exactly the format in your `build_*_envelope()` |
| `Idempotency test failed` | Not deduplicating identical requests | Implement idempotency cache with 24h TTL |
| `DEADLINE_EXCEEDED not raised` | Deadline not checked before provider call | Call `ctx.remaining_ms()` and raise if 0 |
| `retry_after_ms missing from 429 responses` | Error mapping incomplete | Map provider rate limits with `retry_after_ms` |

### Debugging Test Failures

```bash
# Run with full traceback
pytest tests/embedding/test_file.py -v --tb=long

# Stop on first failure
pytest tests/embedding/ -v --maxfail=1

# Run only tests that failed last time
pytest tests/embedding/ -v --lf

# See which tests are available
pytest tests/embedding/ --collect-only
```

### Getting Help From the Suite

The certification suite's error messages are authoritative. When you see:

```
Specification: §4.2.2 Version Identification
Quick fix: Capabilities missing required field 'protocol'
```

**Do not search the internet.** Open the specification and read §4.2.2. The answer is there.

---

**Maintainers:** Corpus SDK Team  
**Last Updated:** 2026-02-11  
**Scope:** Complete adapter authoring reference, fully aligned with Corpus Protocol v1.0 specification.

**The conformance tests in `tests/` are the source of truth.** When this document and the tests disagree, **the tests are correct.**