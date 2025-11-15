# IMPLEMENTATION

**Table of Contents**
- [🚀 Hello World Adapter (30-second start)](#-hello-world-adapter-30-second-start)
- [1. Purpose & Scope](#1-purpose--scope)
- [2. System Layout (What You Implement vs What You Get)](#2-system-layout-what-you-implement-vs-what-you-get)
- [3. Context & Identity (OperationContext)](#3-context--identity-operationcontext)
- [4. Error Taxonomy & Mapping](#4-error-taxonomy--mapping)
- [5. Modes: thin vs standalone](#5-modes-thin-vs-standalone)
- [6. Deadlines & Cancellation](#6-deadlines--cancellation)
- [7. Component Runtime Semantics](#7-component-runtime-semantics)
- [8. Caching & Key Design](#8-caching--key-design)
- [9. Metrics & Observability](#9-metrics--observability)
- [10. Wire Handlers & Canonical Envelopes](#10-wire-handlers--canonical-envelopes)
- [11. Streaming Rules (LLM & Graph)](#11-streaming-rules-llm--graph)
- [12. Conformance Tests & Mocks (High-Level)](#12-conformance-tests--mocks-high-level)
- [13. Environment, Configuration, and Modes](#13-environment-configuration-and-modes)
- [14. Partial Failures & Batch Behavior](#14-partial-failures--batch-behavior)
- [15. Extending Adapters Safely & Common Pitfalls](#15-extending-adapters-safely--common-pitfalls)
- [16. Debugging Conformance Failures](#16-debugging-conformance-failures)
- [17. Implementation Checklists](#17-implementation-checklists)

---

Corpus Protocol (v1.0) — Adapter Implementation Guide (Runtime Behavior)

> **Scope:** How to implement **real adapters** against the Corpus Protocol base SDKs  
> **Components:** **LLM • Embedding • Vector • Graph**  
> **Out of scope:** JSON schema shape (see `SCHEMA_CONFORMANCE.md`)

---

## 🚀 Hello World Adapter (30-second start)

```python
from corpus_sdk.embedding.embedding_base import BaseEmbeddingAdapter, EmbeddingCapabilities, EmbedSpec, BatchEmbedSpec, EmbedResult, BatchEmbedResult, EmbeddingVector

class HelloEmbeddingAdapter(BaseEmbeddingAdapter):
    async def _do_capabilities(self):
        return EmbeddingCapabilities(
            server="hello", version="1.0.0", supported_models=("hello-1",),
            max_batch_size=10, max_text_length=100, max_dimensions=8,
            supports_normalization=False, supports_truncation=True,
            supports_token_counting=False,  # No token counting support
        )
    
    async def _do_embed(self, spec, *, ctx=None):
        vec = [float(len(spec.text))] + [0.0] * 7
        return EmbedResult(
            embedding=EmbeddingVector(vector=vec, text=spec.text, model=spec.model, dimensions=8),
            model=spec.model, text=spec.text
        )
    
    async def _do_embed_batch(self, spec, *, ctx=None):
        embeddings = [await self._do_embed(EmbedSpec(text=t, model=spec.model)) for t in spec.texts]
        return BatchEmbedResult(embeddings=[e.embedding for e in embeddings], model=spec.model, total_texts=len(spec.texts))
    
    async def _do_health(self, *, ctx=None):
        return {"ok": True}

# Use it:
from corpus_sdk.embedding.embedding_base import WireEmbeddingHandler
adapter = HelloEmbeddingAdapter(mode="standalone")
handler = WireEmbeddingHandler(adapter)
# handler.handle(envelope) → full Corpus protocol!
```

**Quick Recipe:**
- **Want LLM?** Override `_do_complete`, `_do_stream`, `_do_count_tokens`
- **Want Embedding?** Override `_do_embed`, `_do_embed_batch` (and `_do_count_tokens` if `supports_token_counting=True`)
- **Want Vector?** Override `_do_query`, `_do_upsert`, `_do_delete`
- **Check:** Capabilities describe your true limits
- **Check:** Map provider errors → canonical errors
- **Check:** Use `ctx.remaining_ms()` for timeouts

---

## 1. Purpose & Scope

This document explains how to implement **production-grade adapters** on top of the Corpus SDK base classes:

- `corpus_sdk.llm.llm_base.BaseLLMAdapter`
- `corpus_sdk.embedding.embedding_base.BaseEmbeddingAdapter`
- `corpus_sdk.vector.vector_base.BaseVectorAdapter`
- `corpus_sdk.graph.graph_base.BaseGraphAdapter` 

It ties together:

- The **runtime behavior** required by `BEHAVIORAL_CONFORMANCE.md`
- The **base classes** and **wire handlers**
- The **mock adapters** and **conformance test suites**

If you follow this file, your adapter will:

- Pass **schema conformance** (with `SCHEMA_CONFORMANCE.md`)
- Pass **behavioral conformance** (with `BEHAVIORAL_CONFORMANCE.md`)
- Behave correctly under deadlines, backpressure, and error conditions  
- Be safe to run in production routing systems

---

## 2. System Layout (What You Implement vs What You Get)

### 2.1 Core SDK modules

You get four protocol "bases," one per component:

- **LLM**  
  `corpus_sdk.llm.llm_base`  
  - `LLMProtocolV1`
  - `BaseLLMAdapter`
  - `WireLLMHandler`

- **Embedding**  
  `corpus_sdk.embedding.embedding_base`  
  - `EmbeddingProtocolV1`
  - `BaseEmbeddingAdapter`
  - `WireEmbeddingHandler`

- **Vector**  
  `corpus_sdk.vector.vector_base`  
  - `VectorProtocolV1`
  - `BaseVectorAdapter`
  - `WireVectorHandler`

- **Graph**  
  `corpus_sdk.graph.graph_base`  
  - `GraphProtocolV1`
  - `BaseGraphAdapter`
  - `WireGraphHandler`

Each base class:

- Implements **all public methods** (`capabilities`, core ops, `health`)
- Exposes **backend hooks** (`_do_*`) that you implement
- Handles **deadlines, metrics, circuit breaking, rate limiting, caching**
- Normalizes **errors** into canonical codes

Each module also ships a **wire handler**:

- Accepts canonical envelopes:  
  `{ "op": "<prefix>.<operation>", "ctx": {...}, "args": {...} }`
- Returns canonical responses:  
  `{ "ok": true/false, "code": "...", "ms": <float>, "result|error": ... }`

You **do not** write your own wire protocol. You plug provider logic into the base class and let the wire handler do the JSON work.

---

### 2.2 What you implement (minimal mental model)

For each provider, you typically:

- Create a subclass, e.g. `MyProviderLLMAdapter(BaseLLMAdapter)`
- Implement a small set of `_do_*` methods:

**LLM**

- `_do_capabilities`
- `_do_complete`
- `_do_stream`
- `_do_count_tokens`
- `_do_health`

**Embedding**

- `_do_capabilities`
- `_do_embed`
- `_do_embed_batch`
- `_do_count_tokens` (if `supports_token_counting=True`)
- `_do_health`

**Vector**

- `_do_capabilities`
- `_do_query`
- `_do_upsert`
- `_do_delete`
- `_do_create_namespace`
- `_do_delete_namespace`
- `_do_health`

**Graph**

- `_do_capabilities`
- `_do_query`
- `_do_stream_query`
- `_do_batch`
- `_do_health`
- plus any CRUD helpers used by batch/query

Everything else (envelopes, metrics, deadlines, caches, circuit breakers) is handled by the base class.

---

### 2.3 Quick-start: your first adapter (end-to-end mini example)

**Example: Tiny Embedding adapter + wire handler + HTTP**

```python
# my_embedding_adapter.py
from typing import Dict, Any
from corpus_sdk.embedding.embedding_base import (
    BaseEmbeddingAdapter,
    EmbeddingCapabilities,
    EmbedSpec,
    BatchEmbedSpec,
    EmbedResult,
    BatchEmbedResult,
    EmbeddingVector,
    ModelNotAvailable,
)

class MyToyEmbeddingAdapter(BaseEmbeddingAdapter):
    """
    Example adapter that returns a deterministic fake embedding.
    This is enough to pass the basic behavior + schema suites.
    """

    async def _do_capabilities(self) -> EmbeddingCapabilities:
        return EmbeddingCapabilities(
            server="mytoy-embedding",
            version="1.0.0",
            supported_models=("toy-embed-1",),
            max_batch_size=16,
            max_text_length=1024,
            max_dimensions=64,
            supports_normalization=True,
            supports_truncation=True,
            supports_token_counting=False,
            normalizes_at_source=False,
        )

    async def _do_embed(
        self,
        spec: EmbedSpec,
        *,
        ctx=None,
    ) -> EmbedResult:
        caps = await self._do_capabilities()
        if spec.model not in caps.supported_models:
            raise ModelNotAvailable(f"Model '{spec.model}' is not supported")

        # Simple deterministic embedding: encode text length into vector shape
        dim = min(caps.max_dimensions or 64, (len(spec.text) % 8) + 4)
        vec = [float((i % 3) - 1) for i in range(dim)]

        ev = EmbeddingVector(
            vector=vec,
            text=spec.text,
            model=spec.model,
            dimensions=len(vec),
        )
        return EmbedResult(
            embedding=ev,
            model=spec.model,
            text=spec.text,
            tokens_used=None,
            truncated=False,  # base sets this if truncation happened
        )

    async def _do_embed_batch(
        self,
        spec: BatchEmbedSpec,
        *,
        ctx=None,
    ) -> BatchEmbedResult:
        # Simple implementation built on top of _do_embed
        embeddings = []
        for text in spec.texts:
            single = await self._do_embed(
                EmbedSpec(
                    text=text,
                    model=spec.model,
                    truncate=spec.truncate,
                    normalize=False,  # base handles normalization
                ),
                ctx=ctx,
            )
            embeddings.append(single.embedding)

        return BatchEmbedResult(
            embeddings=embeddings,
            model=spec.model,
            total_texts=len(spec.texts),
            total_tokens=None,
            failed_texts=[],
        )

    async def _do_count_tokens(
        self,
        text: str,
        model: str,
        *,
        ctx=None,
    ) -> int:
        # Not needed because capabilities() says supports_token_counting=False
        raise NotImplementedError

    async def _do_health(self, *, ctx=None) -> Dict[str, Any]:
        return {
            "ok": True,
            "server": "mytoy-embedding",
            "version": "1.0.0",
            "models": {"toy-embed-1": {"status": "ready"}},
        }

# wiring into HTTP using FastAPI
from fastapi import FastAPI, Request
from corpus_sdk.embedding.embedding_base import WireEmbeddingHandler
from my_embedding_adapter import MyToyEmbeddingAdapter

app = FastAPI()
adapter = MyToyEmbeddingAdapter(mode="standalone")  # demo mode w/ caching, limits
handler = WireEmbeddingHandler(adapter)

@app.post("/embedding")
async def embedding_endpoint(request: Request):
    envelope = await request.json()
    resp = await handler.handle(envelope)
    return resp
```

That's the pattern you repeat for LLM, Vector, and Graph: implement _do_*, drop the adapter into the wire handler, and your service speaks the Corpus protocol.

---

## 3. Context & Identity (OperationContext)

All components share the same context pattern:

- `request_id: Optional[str]`
- `idempotency_key: Optional[str]`
- `deadline_ms: Optional[int]` (epoch ms)
- `traceparent: Optional[str]` (W3C trace context)
- `tenant: Optional[str]`
- `attrs: Mapping[str, Any]` (always a dict after `__post_init__`)

Your `_do_*` hooks receive `ctx: Optional[OperationContext]`. You should:

- Use `ctx.deadline_ms` / `ctx.remaining_ms()` to set provider timeouts.
- Use `ctx.tenant` for multi-tenant partitioning (indexes, projects, DBs).
- Propagate tracing metadata (`traceparent`, `request_id`) into logs and provider SDKs.

The base classes already:

- Hash tenant for metrics (no raw tenant IDs).
- Treat `attrs` as opaque and pass-through.

---

## 4. Error Taxonomy & Mapping

Each component defines a canonical error tree:

- **LLM:** `LLMAdapterError` + `BadRequest`, `AuthError`, `ResourceExhausted`, `ModelOverloaded`, `Unavailable`, `NotSupported`, `DeadlineExceeded`, etc.
- **Embedding:** `EmbeddingAdapterError` + `BadRequest`, `AuthError`, `ResourceExhausted`, `TextTooLong`, `ModelNotAvailable`, `TransientNetwork`, `Unavailable`, `NotSupported`, `DeadlineExceeded`, etc.
- **Vector:** `VectorAdapterError` + `BadRequest`, `AuthError`, `ResourceExhausted`, `DimensionMismatch`, `IndexNotReady`, `TransientNetwork`, `Unavailable`, `NotSupported`, `DeadlineExceeded`, etc.
- **Graph:** `GraphAdapterError` + graph-specific variants like `InvalidQuery`, `DialectNotSupported`, etc.

You are expected to:

1. Catch provider/SDK errors inside `_do_*`.
2. Map them into canonical errors by raising the right subclass with:
   - `message`
   - optional `code` (if overriding default)
   - optional `retry_after_ms`
   - optional `resource_scope` or `throttle_scope`
   - optional `suggested_batch_reduction`
   - optional JSON-safe `details`
3. Avoid leaking provider internals (full HTTP bodies, stack traces, PII) in `message` or `details`.

Wire handlers call `_error_to_wire` to emit the canonical envelope:

```json
{
  "ok": false,
  "code": "RESOURCE_EXHAUSTED",
  "error": "ResourceExhausted",
  "message": "rate limit exceeded",
  "retry_after_ms": 5000,
  "details": { "scope": "model" },
  "ms": 12.34
}
```

Key rule: the same provider error must always map to the same (code, class, retryability).

### 4.1 Example: provider → canonical error mapping (Embedding)

```python
from corpus_sdk.embedding.embedding_base import (
    EmbeddingAdapterError,
    BadRequest,
    AuthError,
    ResourceExhausted,
    TextTooLong,
    ModelNotAvailable,
    TransientNetwork,
    Unavailable,
)

# Example provider error types (replace with real ones)
class ProviderRateLimitError(Exception): ...
class ProviderAuthError(Exception): ...
class ProviderInvalidRequest(Exception): ...
class ProviderTextTooLong(Exception): ...
class ProviderModelError(Exception): ...
class ProviderTimeout(Exception): ...
class ProviderServerError(Exception): ...

def map_provider_error(e: Exception) -> EmbeddingAdapterError:
    if isinstance(e, ProviderRateLimitError):
        return ResourceExhausted(
            "rate limit exceeded",
            retry_after_ms=5000,
            resource_scope="rate_limit",
        )
    if isinstance(e, ProviderAuthError):
        return AuthError("invalid credentials")
    if isinstance(e, ProviderInvalidRequest):
        return BadRequest("invalid parameters")
    if isinstance(e, ProviderTextTooLong):
        return TextTooLong("text exceeds provider limit")
    if isinstance(e, ProviderModelError):
        return ModelNotAvailable("model not available")
    if isinstance(e, ProviderTimeout):
        return TransientNetwork("upstream timeout")
    if isinstance(e, ProviderServerError):
        return Unavailable("provider unavailable")
    # Last resort fallback
    return Unavailable("unknown provider error")
```

Then in `_do_*`:

```python
try:
    resp = await self._client.embed(...)
except Exception as e:
    raise map_provider_error(e)
```

---

## 5. Modes: thin vs standalone

Every base adapter supports two modes:

- `mode="thin"` (default)
  - For use under an external control plane.
  - Deadline policy = no-op
  - Circuit breaker = no-op
  - Rate limiter = no-op
  - Cache = no-op

- `mode="standalone"`
  - For direct use, demos, and light production:
  - Enforces deadlines
  - Uses a per-process circuit breaker
  - Uses a token-bucket rate limiter
  - Uses in-memory TTL cache (read-paths + capabilities)
  - Logs a warning if running standalone with `NoopMetrics`

You pick the mode when constructing your adapter:

```python
adapter = MyRealEmbeddingAdapter(
    mode="standalone",
    metrics=my_metrics_sink,
)
```

You can also override individual policies explicitly (see §§6–8).

---

## 6. Deadlines & Cancellation

Each base uses a `DeadlinePolicy` to enforce `ctx.deadline_ms`:

- **LLM:** `NoopDeadline` or `SimpleDeadline`
- **Embedding:** `NoopDeadline` or `EnforcingDeadline`
- **Vector:** `NoopDeadline` or `SimpleDeadline`
- **Graph:** same pattern

### 6.1 Preflight

Bases call `_fail_if_expired(ctx)`:

- If `ctx.deadline_ms` is set and `ctx.remaining_ms() <= 0`
- → raise `DeadlineExceeded("deadline already exceeded")` before hitting the provider.

### 6.2 Wrapping provider calls

All provider awaits are wrapped via `_apply_deadline() → deadline_policy.wrap(awaitable, ctx)`:

- If `asyncio.wait_for` times out, the base raises `DeadlineExceeded("operation timed out")`.
- You generally should not call `asyncio.wait_for` on your own for the same operation.

### 6.3 Propagating budgets to providers

Inside `_do_*`, you should:

```python
async def _do_embed(self, spec, *, ctx=None) -> EmbedResult:
    timeout_s = None
    if ctx is not None:
        rem = ctx.remaining_ms()
        if rem is not None and rem > 0:
            timeout_s = rem / 1000.0

    resp = await self._client.embed(
        model=spec.model,
        text=spec.text,
        timeout=timeout_s,
    )
    ...
```

Guidelines:

- Never pass a negative or zero timeout to providers.
- If the provider has its own deadline / context object, convert `remaining_ms` accordingly.

### 6.4 Metrics & deadlines

Bases tag metrics with `deadline_bucket` when a deadline is present:

- `<1s, <5s, <15s, <60s, >=60s`

This gives you a quick view of how tight caller budgets are.

---

## 7. Component Runtime Semantics

This section explains the runtime behavior for each component and illustrates patterns with code.

---

## 7.1 LLM (BaseLLMAdapter)

### 7.1.1 Core operations

- `capabilities() -> LLMCapabilities`
- `complete(request, ctx) -> LLMCompletion`
- `stream(request, ctx) -> AsyncIterator[LLMChunk]`
- `count_tokens(text, model, ctx) -> int`
- `health(ctx) -> Mapping[str, Any]`

### 7.1.2 Implementation pattern

**What you override:**
- `_do_capabilities()` - Describe what you support
- `_do_complete()` - Single request → response
- `_do_stream()` - Streaming response chunks
- `_do_count_tokens()` - Token counting (optional)
- `_do_health()` - Health checks

**When this will break conformance:**
- Not respecting `supported_models` from capabilities
- Streaming: multiple final chunks or data after error
- Ignoring `ctx.remaining_ms()` for provider timeouts
- Not mapping provider errors to canonical errors

### 7.1.3 Example wire payload (unary completion)

```json
{
  "op": "llm.complete",
  "ctx": {
    "request_id": "abc-123",
    "deadline_ms": 1731456000000,
    "tenant": "tenant-a"
  },
  "args": {
    "model": "my-llm-1",
    "messages": [
      { "role": "system", "content": "You are a helpful assistant." },
      { "role": "user", "content": "Summarize this text." }
    ],
    "max_tokens": 256,
    "temperature": 0.7
  }
}
```

### 7.1.4 Validation & context window

Base-level behavior:

- Validates messages as a non-empty list of `{role, content}` mappings.
- Validates sampling params ranges.
- If `caps.supports_count_tokens` is true:
  - Builds an internal prompt representation.
  - Calls `_do_count_tokens` with deadlines enforced.
  - Enforces `prompt_tokens + max_tokens <= caps.max_context_length`.
  - Raises `BadRequest` if the request is too big.

### 7.1.5 Model gating & caching

- If `caps.supported_models` is non-empty:
  - `request.model` must be in that set.
- If in-memory cache is present (`InMemoryTTLCache` in standalone mode):
  - `complete` can re-use cached responses based on:
    - model
    - full messages fingerprint
    - sampling params
    - stop sequences
    - tenant hash
  - Cache hits increment `cache_hits` counter.

### 7.1.6 Streaming semantics

- `stream()` uses a streaming gate (`_with_gates_stream`):
  - Deadline preflight
  - Rate limit acquire/release
  - Circuit breaker
  - Periodic deadline checks while streaming
- You implement `_do_stream(request, ctx) -> AsyncIterator[LLMChunk]`:
  - Yield chunks with `text`, `is_final`, `model` (and any other fields used in tests).
  - Exactly one chunk with `is_final=True`.
  - No chunks after `is_final=True` or an error.

**Example: simple LLM adapter**

```python
from typing import AsyncIterator, Dict, Any
from corpus_sdk.llm.llm_base import (
    BaseLLMAdapter,
    LLMCapabilities,
    LLMCompletion,
    LLMChunk,
    BadRequest,
)

class MyLLMAdapter(BaseLLMAdapter):
    async def _do_capabilities(self) -> LLMCapabilities:
        return LLMCapabilities(
            server="my-llm-provider",
            version="1.0.0",
            supported_models=("my-llm-1",),
            max_context_length=8192,
            supports_streaming=True,
            supports_count_tokens=False,
        )

    async def _do_complete(self, request, *, ctx=None) -> LLMCompletion:
        prompt = "\n".join(m["content"] for m in request.messages)
        text = f"[LLM reply to]: {prompt[:64]}..."
        return LLMCompletion(
            text=text,
            model=request.model,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    async def _do_stream(
        self,
        request,
        *,
        ctx=None,
    ) -> AsyncIterator[LLMChunk]:
        # Single-chunk streaming example
        completion = await self._do_complete(request, ctx=ctx)
        yield LLMChunk(text=completion.text, is_final=True, model=request.model)

    async def _do_count_tokens(self, text: str, model: str, *, ctx=None) -> int:
        raise BadRequest("count_tokens not supported by this adapter")

    async def _do_health(self, *, ctx=None) -> Dict[str, Any]:
        return {"ok": True, "server": "my-llm-provider", "version": "1.0.0"}
```

---

## 7.2 Embedding (BaseEmbeddingAdapter)

### 7.2.1 Implementation pattern

**What you override:**
- `_do_capabilities()` - Describe limits and support
- `_do_embed()` - Single text → vector
- `_do_embed_batch()` - Batch texts → vectors (or raise `NotSupported`)
- `_do_count_tokens()` - Token counting (if `supports_token_counting=True`)
- `_do_health()` - Health checks

**When this will break conformance:**
- Capabilities don't match actual provider limits
- Not using base truncation for `max_text_length`
- Returning normalized vectors but `normalizes_at_source=False`
- Batch: not reporting partial failures correctly

Base handles truncation, normalization, caching, deadlines, partial batch fallback, and metrics.

### 7.2.2 Core types & operations

**Types:**

- `EmbedSpec`, `BatchEmbedSpec`
- `EmbedResult`, `BatchEmbedResult`
- `EmbeddingVector`
- `EmbeddingCapabilities`

**Operations:**

- `capabilities() -> EmbeddingCapabilities`
- `embed(spec, ctx) -> EmbedResult`
- `embed_batch(spec, ctx) -> BatchEmbedResult`
- `count_tokens(text, model, ctx) -> int`
- `health(ctx) -> Dict[str, Any]`

### 7.2.3 Capabilities fields

Important fields in `EmbeddingCapabilities`:

- `supported_models: Tuple[str, ...]`
- `max_batch_size: Optional[int]`
- `max_text_length: Optional[int]`
- `max_dimensions: Optional[int]`
- `supports_normalization: bool`
- `supports_truncation: bool`
- `supports_token_counting: bool`
- `normalizes_at_source: bool`
- `truncation_mode: str` ("base" or "adapter")
- `supports_deadline: bool`

### 7.2.4 Truncation semantics

- If `max_text_length` is set and `len(text) > max_text_length`:
  - Base uses `SimpleCharTruncation.apply(text, max_len, allow=spec.truncate)`.
  - If `truncate=True`: text is truncated, `EmbedResult.truncated=True`.
  - If `truncate=False`: raise `TextTooLong`.

You usually do not need to manually truncate inside `_do_embed`.

### 7.2.5 Normalization semantics

- If `spec.normalize=True`:
  - If `caps.supports_normalization=False`: raise `NotSupported`.
  - Else if `caps.normalizes_at_source=False`:
    - Base applies L2 normalization (`L2Normalization`) to outbound vectors:
      - `norm = sqrt(sum(v*v))` or `1.0`
      - `v[i] = v[i] / norm`

### 7.2.6 Batch semantics & partial success

`embed_batch`:

- Validates model and batch size.
- Validates and truncates each text.
- Attempts `_do_embed_batch` first.
- If `_do_embed_batch` raises `NotSupported`:
  - Base falls back to per-text `_do_embed`.
  - Successful embeddings go into `BatchEmbedResult.embeddings`.
  - Failures go into `BatchEmbedResult.failed_texts` with:
    - `index`, `text`, `error`, `code`, `message`.

### 7.2.7 Caching & token counting

- `embed` uses a tenant-aware, content-addressed cache when an `InMemoryTTLCache` is configured:

  ```
  embedding:embed:tenant=<hash|global>:model=<model>:norm=<0|1>:text=<sha256(text)>
  ```

- `count_tokens`:
  - Requires model in `caps.supported_models`.
  - Requires `caps.supports_token_counting=True`.
  - Otherwise, base raises `NotSupported`.

**Example: "real-ish" Embedding adapter**

```python
from typing import Dict, Any
from corpus_sdk.embedding.embedding_base import (
    BaseEmbeddingAdapter,
    EmbeddingCapabilities,
    EmbedSpec,
    BatchEmbedSpec,
    EmbedResult,
    BatchEmbedResult,
    EmbeddingVector,
    ModelNotAvailable,
    EmbeddingAdapterError,
    NotSupported,
)

class MyRealEmbeddingAdapter(BaseEmbeddingAdapter):
    def __init__(self, provider_client, **kwargs):
        super().__init__(mode="thin", **kwargs)
        self._client = provider_client

    async def _do_capabilities(self) -> EmbeddingCapabilities:
        return EmbeddingCapabilities(
            server="my-embed-provider",
            version="2025-01-01",
            supported_models=("embed-large", "embed-small"),
            max_batch_size=256,
            max_text_length=8192,
            max_dimensions=1536,
            supports_normalization=True,
            supports_truncation=True,
            supports_token_counting=True,
            normalizes_at_source=False,
        )

    async def _do_embed(self, spec: EmbedSpec, *, ctx=None) -> EmbedResult:
        caps = await self._do_capabilities()
        if spec.model not in caps.supported_models:
            raise ModelNotAvailable(f"Model '{spec.model}' is not supported")

        timeout_s = None
        if ctx is not None:
            rem = ctx.remaining_ms()
            if rem and rem > 0:
                timeout_s = rem / 1000.0

        try:
            resp = await self._client.embed(
                model=spec.model,
                text=spec.text,
                timeout=timeout_s,
            )
        except Exception as e:
            from .error_mapping import map_provider_error
            raise map_provider_error(e)

        vec = resp["vector"]
        ev = EmbeddingVector(
            vector=vec,
            text=spec.text,
            model=spec.model,
            dimensions=len(vec),
        )
        return EmbedResult(
            embedding=ev,
            model=spec.model,
            text=spec.text,
            tokens_used=resp.get("tokens"),
            truncated=False,  # base will set if truncation occurred
        )

    async def _do_embed_batch(
        self,
        spec: BatchEmbedSpec,
        *,
        ctx=None,
    ) -> BatchEmbedResult:
        caps = await self._do_capabilities()
        if spec.model not in caps.supported_models:
            raise ModelNotAvailable(f"Model '{spec.model}' is not supported")

        timeout_s = None
        if ctx is not None:
            rem = ctx.remaining_ms()
            if rem and rem > 0:
                timeout_s = rem / 1000.0

        try:
            resp = await self._client.embed_batch(
                model=spec.model,
                texts=spec.texts,
                timeout=timeout_s,
            )
        except NotImplementedError:
            # Force base fallback with per-item partial success
            raise NotSupported("provider batch not supported")
        except Exception as e:
            from .error_mapping import map_provider_error
            raise map_provider_error(e)

        embeddings = [
            EmbeddingVector(
                vector=vec,
                text=text,
                model=spec.model,
                dimensions=len(vec),
            )
            for text, vec in zip(spec.texts, resp["vectors"])
        ]

        return BatchEmbedResult(
            embeddings=embeddings,
            model=spec.model,
            total_texts=len(spec.texts),
            total_tokens=resp.get("total_tokens"),
            failed_texts=[],
        )

    async def _do_count_tokens(self, text: str, model: str, *, ctx=None) -> int:
        timeout_s = None
        if ctx is not None:
            rem = ctx.remaining_ms()
            if rem and rem > 0:
                timeout_s = rem / 1000.0
        try:
            return await self._client.count_tokens(
                model=model,
                text=text,
                timeout=timeout_s,
            )
        except Exception as e:
            from .error_mapping import map_provider_error
            raise map_provider_error(e)

    async def _do_health(self, *, ctx=None) -> Dict[str, Any]:
        return {
            "ok": True,
            "server": "my-embed-provider",
            "version": "2025-01-01",
            "models": {"embed-large": {"status": "ready"}},
        }
```

---

## 7.3 Vector (BaseVectorAdapter)

### 7.3.1 Core types & operations

**Types:**

- `VectorID`, `Vector`, `VectorMatch`
- `QuerySpec`, `UpsertSpec`, `DeleteSpec`, `NamespaceSpec`
- `QueryResult`, `UpsertResult`, `DeleteResult`, `NamespaceResult`
- `VectorCapabilities`

**Operations:**

- `capabilities() -> VectorCapabilities`
- `query(spec, ctx) -> QueryResult`
- `upsert(spec, ctx) -> UpsertResult`
- `delete(spec, ctx) -> DeleteResult`
- `create_namespace(spec, ctx) -> NamespaceResult`
- `delete_namespace(namespace, ctx) -> NamespaceResult`
- `health(ctx) -> Dict[str, Any]`

### 7.3.2 Semantics

Base behavior:

- Validates:
  - Vectors as non-empty lists of numeric values.
  - Filters as mappings that are JSON-serializable.
- Enforces capabilities:
  - `max_dimensions`: any vector dimension must be ≤ this value.
  - `max_top_k`: `top_k` must be ≤ this value.
  - `max_batch_size`: enforced for upsert/delete.
  - `supports_metadata_filtering`: required to use filter.
- Caching:
  - Standalone mode with `InMemoryTTLCache`:
  - Query cache key includes:
    - vector hash, filter hash
    - namespace, top_k, include_metadata, include_vectors
    - backend identity (server:version)
    - tenant hash

### 7.3.3 Wire example

```json
{
  "op": "vector.query",
  "ctx": { "request_id": "r-1", "tenant": "tenant-a" },
  "args": {
    "vector": [0.1, 0.2, 0.3],
    "top_k": 5,
    "namespace": "docs",
    "filter": { "doc_type": "pdf" },
    "include_metadata": true,
    "include_vectors": false
  }
}
```

**Example: in-memory Vector adapter**

```python
from typing import Dict, Any, List
from corpus_sdk.vector.vector_base import (
    BaseVectorAdapter,
    VectorCapabilities,
    QuerySpec,
    UpsertSpec,
    DeleteSpec,
    NamespaceSpec,
    QueryResult,
    UpsertResult,
    DeleteResult,
    NamespaceResult,
    Vector,
    VectorMatch,
    VectorID,
)

class InMemoryVectorAdapter(BaseVectorAdapter):
    def __init__(self, **kwargs):
        super().__init__(mode="thin", **kwargs)
        # namespace -> { id -> Vector }
        self._store: Dict[str, Dict[str, Vector]] = {}

    async def _do_capabilities(self) -> VectorCapabilities:
        return VectorCapabilities(
            server="vector-in-memory",
            version="1.0.0",
            max_dimensions=2048,
            max_batch_size=1024,
            supports_metadata_filtering=False,
            supports_index_management=True,
        )

    async def _do_query(self, spec: QuerySpec, *, ctx=None) -> QueryResult:
        ns = self._store.get(spec.namespace, {})
        matches: List[VectorMatch] = []
        for v in ns.values():
            # Super naive similarity: length difference
            score = 1.0 / (1.0 + abs(len(v.vector) - len(spec.vector)))
            matches.append(
                VectorMatch(
                    vector=v,
                    score=score,
                    distance=1.0 - score,
                )
            )
        matches.sort(key=lambda m: m.score, reverse=True)
        top = matches[: spec.top_k]
        return QueryResult(
            matches=top,
            query_vector=spec.vector,
            namespace=spec.namespace,
            total_matches=len(matches),
        )

    async def _do_upsert(self, spec: UpsertSpec, *, ctx=None) -> UpsertResult:
        ns = self._store.setdefault(spec.namespace, {})
        failures: List[Dict[str, Any]] = []
        for v in spec.vectors:
            ns[str(v.id)] = v
        return UpsertResult(
            upserted_count=len(spec.vectors),
            failed_count=len(failures),
            failures=failures,
        )

    async def _do_delete(self, spec: DeleteSpec, *, ctx=None) -> DeleteResult:
        ns = self._store.setdefault(spec.namespace, {})
        failures: List[Dict[str, Any]] = []
        deleted = 0
        for vid in spec.ids:
            if str(vid) in ns:
                ns.pop(str(vid))
                deleted += 1
        return DeleteResult(
            deleted_count=deleted,
            failed_count=len(failures),
            failures=failures,
        )

    async def _do_create_namespace(
        self,
        spec: NamespaceSpec,
        *,
        ctx=None,
    ) -> NamespaceResult:
        self._store.setdefault(spec.namespace, {})
        return NamespaceResult(success=True, namespace=spec.namespace, details={})

    async def _do_delete_namespace(
        self,
        namespace: str,
        *,
        ctx=None,
    ) -> NamespaceResult:
        self._store.pop(namespace, None)
        return NamespaceResult(success=True, namespace=namespace, details={})

    async def _do_health(self, *, ctx=None) -> Dict[str, Any]:
        return {"ok": True, "server": "vector-in-memory", "version": "1.0.0"}
```

---

## 7.4 Graph (BaseGraphAdapter)

The Graph base mirrors the same design:

- Async-first, canonical envelopes, `_do_*` hooks.
- Supports:
  - `query` (unary)
  - `stream_query` (streaming)
  - `batch` (mixed vertex/edge operations)
  - Vertex/edge CRUD as needed
  - `capabilities`, `health`

### 7.4.1 Expectations

- **Streaming:**
  - Exactly one terminal event (end or error).
  - No rows/events after terminal.
  - Heartbeats allowed but must respect pacing/backpressure.
- **Batch semantics:**
  - Partial success is explicitly represented.
  - Only successful operations mutate state.
  - Input ordering and IDs are preserved.

**Example: Graph streaming skeleton**

```python
from typing import AsyncIterator, Dict, Any
from corpus_sdk.graph.graph_base import (
    BaseGraphAdapter,
    GraphCapabilities,
    GraphRow,
    GraphStreamEvent,
)

class MyGraphAdapter(BaseGraphAdapter):
    async def _do_capabilities(self) -> GraphCapabilities:
        return GraphCapabilities(
            server="my-graph",
            version="1.0.0",
            dialect="cypher",
            supports_streaming=True,
        )

    async def _do_query(self, spec, *, ctx=None):
        # Simple unary example
        rows = [
            GraphRow(data={"id": "u-1", "name": "Alice"}),
            GraphRow(data={"id": "u-2", "name": "Bob"}),
        ]
        return {"rows": rows}

    async def _do_stream_query(
        self,
        spec,
        *,
        ctx=None,
    ) -> AsyncIterator[GraphStreamEvent]:
        rows = await self._do_query(spec, ctx=ctx)
        for row in rows["rows"]:
            yield GraphStreamEvent(data=row, kind="row")
        # Exactly one terminal event
        yield GraphStreamEvent(data=None, kind="end")

    async def _do_batch(self, spec, *, ctx=None):
        # Must match BEHAVIORAL_CONFORMANCE expectations for batch responses
        ...

    async def _do_health(self, *, ctx=None) -> Dict[str, Any]:
        return {"ok": True, "server": "my-graph", "version": "1.0.0"}
```

---

## 8. Caching & Key Design

The base classes implement per-process caches for:

- LLM complete
- Embedding embed
- Vector query
- Capabilities for all components

**Properties:**

- **Content-addressed:**
  - Use hashes of content; no raw text or vectors in keys.
- **Tenant-aware:**
  - Include tenant hash when `ctx.tenant` is present.
- **Param-sensitive:**
  - Include model, flags, and relevant parameters (normalization, top_k, etc.).
- **TTL-based:**
  - Configurable TTLs (`cache_ttl_s`, `cache_embed_ttl_s`, `cache_query_ttl_s`, `cache_caps_ttl_s`).

If you need additional caching:

- Prefer wrapping adapters or higher-level caches.
- Avoid adding conflicting caches inside `_do_*`.

---

## 9. Metrics & Observability

All base classes use a `MetricsSink` with:

```python
def observe(
    *,
    component: str,
    op: str,
    ms: float,
    ok: bool,
    code: str = "OK",
    extra: Optional[Mapping[str, Any]] = None,
) -> None: ...

def counter(
    *,
    component: str,
    name: str,
    value: int = 1,
    extra: Optional[Mapping[str, Any]] = None,
) -> None: ...
```

**Behavior:**

- Exactly one timing observation per op.
- Additional counters for:
  - queries, vectors_upserted, vectors_deleted
  - texts_embedded, tokens_processed, count_tokens_calls
  - cache_hits, errors_total, etc.
- Tenant IDs are hashed (`tenant_hash`).
- Deadlines reported as `deadline_bucket` when present.

**Example: simple Prometheus sink**

```python
class PromMetricsSink:
    def __init__(self, registry):
        self._latency = ...
        self._counter = ...

    def observe(self, *, component, op, ms, ok, code="OK", extra=None):
        labels = {
            "component": component,
            "op": op,
            "ok": "true" if ok else "false",
            "code": code,
        }
        if extra:
            for k in ("tenant_hash", "deadline_bucket", "model"):
                if k in extra:
                    labels[k] = extra[k]
        self._latency.labels(**labels).observe(ms / 1000.0)

    def counter(self, *, component, name, value=1, extra=None):
        labels = {"component": component, "name": name}
        self._counter.labels(**labels).inc(value)
```

---

## 10. Wire Handlers & Canonical Envelopes

Each component exposes a wire handler:

- `WireLLMHandler`
- `WireEmbeddingHandler`
- `WireVectorHandler`
- `WireGraphHandler`

### 10.1 Envelopes

**Input:**

```json
{
  "op": "<prefix>.<operation>",
  "ctx": { ... },
  "args": { ... }
}
```

**Success output:**

```json
{
  "ok": true,
  "code": "OK",
  "ms": 12.3,
  "result": { ... }
}
```

**Error output:**

```json
{
  "ok": false,
  "code": "BAD_REQUEST",
  "error": "BadRequest",
  "message": "explanation",
  "retry_after_ms": null,
  "details": { ... } | null,
  "ms": 1.2
}
```

Wire handlers:

- Parse `ctx` into `OperationContext` via `_ctx_from_wire`.
- Construct typed specs (`EmbedSpec`, `QuerySpec`, etc.) from `args`.
- Call your adapter methods.
- Normalize results with `_success_to_wire`.
- Normalize errors with `_error_to_wire`.
- For unknown op, raise `NotSupported("unknown operation '<op>'")`.

**Example: embedding wire endpoint**

```python
from fastapi import FastAPI, Request
from corpus_sdk.embedding.embedding_base import WireEmbeddingHandler
from my_embedding_adapter import MyRealEmbeddingAdapter

app = FastAPI()
adapter = MyRealEmbeddingAdapter(mode="standalone", metrics=PromMetricsSink(...))
handler = WireEmbeddingHandler(adapter)

@app.post("/embedding")
async def embedding_endpoint(request: Request):
    envelope = await request.json()
    resp = await handler.handle(envelope)
    return resp
```

---

## 11. Streaming Rules (LLM & Graph)

Streaming operations (`llm.stream`, `graph.stream_query`) must obey:

- Exactly one terminal envelope:
  - Final chunk with `is_final=True` (LLM), or
  - Terminal end event (Graph), or
  - Single error envelope.
- No data after terminal.
- Mid-stream error is terminal.
- Heartbeats are allowed but must not flood or ignore backpressure.

**Streaming State Machine:**
```
START -> DATA* -> (END | ERROR) -> STOP
            ^           |
            |-----------|
(No DATA after END/ERROR; ERROR is terminal)
```

The base enforces:

- Deadline preflight.
- Rate limiting.
- Circuit breaker integration.
- Deadline checks across long-running streams.

You implement `_do_stream` / `_do_stream_query` to produce chunks/events that the wire handler wraps.

---

## 12. Conformance Tests & Mocks (High-Level)

Three main conformance pillars:

- `SPECIFICATION.md` — normative protocol description.
- `SCHEMA_CONFORMANCE.md` — JSON schemas & wire shapes.
- `BEHAVIORAL_CONFORMANCE.md` — semantic behavior & edge cases.

Plus:

- Mock adapters for LLM, Embedding, Vector, Graph.
- Conformance test suites for:
  - Schema conformance.
  - Behavioral conformance.

**Adapter author workflow:**

1. Implement your adapter on top of the base class.
2. Run schema verify targets (`make verify-schema`).
3. Run behavioral suites:
   - `make test-llm-conformance`
   - `make test-embedding-conformance`
   - `make test-vector-conformance`
   - `make test-graph-conformance`
4. Iterate until your adapter passes all suites unmodified.

---

## 13. Environment, Configuration, and Modes

Typical knobs:

- `mode`: "thin" or "standalone".
- `metrics`: your MetricsSink implementation.
- Policy overrides:
  - `deadline_policy`
  - `breaker`
  - `cache`
  - `limiter`
- Embedding-specific: truncation, normalization.
- Cache TTLs:
  - LLM: `cache_ttl_s`
  - Embedding: `cache_embed_ttl_s`, `cache_caps_ttl_s`
  - Vector: `cache_query_ttl_s`, `cache_caps_ttl_s`

**Example: wiring from env**

```python
import os
from corpus_sdk.embedding.embedding_base import InMemoryTTLCache, TokenBucketLimiter

cache_ttl = int(os.getenv("EMBED_CACHE_TTL_S", "60"))
rate = float(os.getenv("EMBED_RATE_PER_SEC", "50"))

adapter = MyRealEmbeddingAdapter(
    mode=os.getenv("ADAPTER_MODE", "thin"),
    metrics=PromMetricsSink(...),
    cache=InMemoryTTLCache(),
    limiter=TokenBucketLimiter(rate=rate, burst=100),
    cache_embed_ttl_s=cache_ttl,
)
```

---

## 14. Partial Failures & Batch Behavior

The system prefers explicit partial success over all-or-nothing semantics.

- **Embedding:**
  - `BatchEmbedResult.failed_texts` carries:
    - `index`, `text`, `error`, `code`, `message`.
- **Vector:**
  - `UpsertResult`, `DeleteResult` include:
    - `upserted_count` / `deleted_count`
    - `failed_count`
    - `failures: List[Dict[str, Any]]`
- **Graph:**
  - Batch ops encode per-op success/failure as required by `BEHAVIORAL_CONFORMANCE.md`.

**Rules:**

- Never silently drop failed items.
- Counts must match actual successes and failures.
- Input order and indices must be preserved.

---

## 15. Extending Adapters Safely & Common Pitfalls

You can extend adapters:

- **Inside `_do_*`:** add provider features and advanced options.
- **Outside the base:** wrap adapters with higher-level routing, logging, or policy layers.

The canonical contract is defined by:

- Base classes
- Wire handlers
- Conformance tests

As long as those pass unchanged, you are within spec.

### 15.1 Common Pitfalls

**LLM**

- Forgetting to enforce `supported_models` → requests for unknown models slip through.
- Violating streaming rules (`is_final` multiple times or trailing chunks).
- Ignoring deadlines, causing upstream provider timeouts that never surface as `DeadlineExceeded`.

**Embedding**

- Re-implementing truncation inside `_do_embed` instead of relying on base behavior.
- Returning already-normalized vectors but incorrectly setting `normalizes_at_source=False`.
- Forgetting to set `supports_token_counting` and implementing `_do_count_tokens` anyway (base may never call it).
- Not including model in `supported_models`, causing `ModelNotAvailable` to fire unexpectedly.

**Vector**

- Returning vectors with mismatched dimensions relative to capabilities.
- Using filters when `supports_metadata_filtering=False`.
- Ignoring `max_batch_size` and relying on provider failures instead of raising `BadRequest`.
- Using raw vector contents in cache keys or logs.

**Graph**

- Emitting extra data after an end event or error in streaming.
- Not preserving row order or IDs in batch responses.
- Collapsing partial failures into a single global error instead of per-op details.

**Wire / Metrics / Security**

- Forking wire handlers instead of using the provided ones.
- Logging entire envelopes (including text/embeddings) without redaction.
- Exposing raw tenant IDs in metrics or logs instead of hashed identifiers.

If conformance tests are failing in confusing ways, check this section first.

---

## 16. Debugging Conformance Failures

**Diagnostic Flow:**
- **Schema failures?** → Run `make verify-schema`
- **Behavioral failures?** → Check the specific test file from `BEHAVIORAL_CONFORMANCE.md §5`
- **Streaming issues?** → Review **Streaming State Machine** above
- **Error mapping wrong?** → Review **Error Taxonomy & Mapping** (§4)

**Testing Best Practices:**
- Seed randomness and log the seed on failure
- Log adapter version + capabilities snapshot at test start  
- Isolate or clear caches between tests
- Use `PYTEST_JOBS=auto` for CI parallelism

**Environment Profiles:**
- **Local dev:** generous timeouts, low parallelism
- **CI:** explicit budgets, no external dependencies  
- **Stress (opt-in):** short deadlines, forced backoffs (`@slow` tests)

---

## 17. Implementation Checklists

Use these as the "ready for conformance" gates for each adapter.

---

## 17.1 LLM Adapter Checklist

- [ ] **Implements:**
  - `_do_capabilities() -> LLMCapabilities`
  - `_do_complete(...) -> LLMCompletion`
  - `_do_stream(...) -> AsyncIterator[LLMChunk]`
  - `_do_count_tokens(text, model, ctx) -> int`
  - `_do_health(ctx) -> Mapping[str, Any]`
- [ ] **Honors:**
  - `supported_models` gating.
  - `max_context_length` via token counting preflight when supported.
  - Flags like `supports_streaming`, `supports_count_tokens`.
- [ ] **Deadlines:**
  - Uses `ctx.remaining_ms()` for provider timeouts.
  - Does not call upstream if deadline already exceeded.
- [ ] **Streaming:**
  - Exactly one terminal chunk with `is_final=True`.
  - No chunks after final or error.
- [ ] **Error mapping:**
  - Provider errors map to `LLMAdapterError` subclasses.
  - Retryability matches your mapping table.
- [ ] **Metrics:**
  - One observe per op.
  - Token counters wired via `usage.total_tokens`.
- [ ] **Conformance:**
  - Passes all LLM schema + behavioral suites unmodified.

---

## 17.2 Vector Adapter Checklist

- [ ] **Implements:**
  - `_do_capabilities() -> VectorCapabilities`
  - `_do_query(spec, ctx) -> QueryResult`
  - `_do_upsert(spec, ctx) -> UpsertResult`
  - `_do_delete(spec, ctx) -> DeleteResult`
  - `_do_create_namespace(spec, ctx) -> NamespaceResult`
  - `_do_delete_namespace(namespace, ctx) -> NamespaceResult`
  - `_do_health(ctx) -> Dict[str, Any]`
- [ ] **Enforces:**
  - Vector shape (non-empty, numeric).
  - `max_dimensions`, `max_top_k`, `max_batch_size`.
  - `supports_metadata_filtering` before using filter.
- [ ] **Caching:**
  - Query cache keys include tenant hash, namespace, params.
  - No raw vectors in cache keys.
- [ ] **Deadlines & breaker:**
  - Uses `ctx.remaining_ms()` with providers.
  - Uses circuit breaker hooks consistently (`allow`, `on_success`, `on_error`).
- [ ] **Partial behavior:**
  - Upsert/delete results report per-item failures when applicable.
- [ ] **Conformance:**
  - Passes vector schema + behavioral suites unmodified.

---

## 17.3 Embedding Adapter Checklist

- [ ] **Implements:**
  - `_do_capabilities() -> EmbeddingCapabilities`
  - `_do_embed(spec: EmbedSpec, ctx) -> EmbedResult`
  - `_do_embed_batch(spec: BatchEmbedSpec, ctx) -> BatchEmbedResult` (or raises `NotSupported` to use base fallback)
  - `_do_count_tokens(text, model, ctx) -> int` (if `supports_token_counting=True`)
  - `_do_health(ctx) -> Dict[str, Any]`
- [ ] **Capabilities:**
  - `supported_models` matches provider reality.
  - `max_text_length`, `max_batch_size`, `max_dimensions` set truthfully.
  - `supports_normalization`, `normalizes_at_source`, `supports_token_counting` correct.
- [ ] **Truncation:**
  - `max_text_length` enforced via base truncation.
  - `truncate=True` → truncated text + `EmbedResult.truncated=True`.
  - `truncate=False` → `TextTooLong` when over limit.
- [ ] **Normalization:**
  - `normalize=True` and `supports_normalization=False` → `NotSupported`.
  - If vectors are not normalized at source, base L2-normalization is sufficient.
- [ ] **Batch behavior:**
  - `max_batch_size` enforced.
  - On `_do_embed_batch` `NotSupported`, base per-item fallback:
    - Embeddings in `embeddings`.
    - Failures in `failed_texts` with `index`, `code`, `message`.
- [ ] **Caching:**
  - Uses base embed cache (tenant-scoped, content-addressed).
  - No raw text or tenant IDs in any cache key you add.
- [ ] **Deadlines & metrics:**
  - Provider timeouts use `ctx.remaining_ms()`.
  - Metrics are SIEM-safe; tenant hash only.
- [ ] **Conformance:**
  - Passes all embedding schema + behavioral suites unmodified.

---

## ✅ Adapter Ready — Corpus Protocol (v1.0)

```
Components: <LLM|Embedding|Vector|Graph>
Adapter:    <YourAdapterName>
Mode(s):    ["thin", "standalone"]
Commit:     <git sha>
CI Run:     <link to conformance run>

Status:
  - SCHEMA_CONFORMANCE:    PASS
  - BEHAVIORAL_CONFORMANCE: PASS
  - COV_FAIL_UNDER:        <N>% or higher
```

**Maintainers:** Corpus SDK Team  
**Last Updated:** 2025-11-12  
**Scope:** Adapter runtime behavior; see `SCHEMA_CONFORMANCE.md` and `BEHAVIORAL_CONFORMANCE.md` for normative contracts.