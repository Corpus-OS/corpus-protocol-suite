# Adapter Quickstart

> **Goal:** Get a real adapter speaking the Corpus Protocol v1.0 in minutes.  
> **Audience:** SDK / adapter authors (LLM, Embedding, Vector, Graph).  
> **You'll build:** A tiny Embedding adapter + wire handler you can swap for LLM / Vector / Graph / Embedding

**In this guide you will:**
- Implement a minimal Embedding adapter
- Expose it over HTTP  
- Call it with a real Corpus envelope
- See where to go for LLM / Vector / Graph / Embedding

---

## 0. Mental Model (What You're Actually Building)

An **adapter** is a thin layer that:

1. Implements a small set of `_do_*` methods against *your* provider (OpenAI, Vertex, in-house, etc).
2. Plugs into a **base class** (`BaseLLMAdapter`, `BaseEmbeddingAdapter`, etc).
3. Is exposed via a **wire handler** (`WireLLMHandler`, `WireEmbeddingHandler`, …) that speaks the Corpus Protocol JSON envelopes.

You **do not** write:

- JSON envelopes  
- Deadline handling  
- Metrics, rate limiting, circuit breaking  
- Caching or error normalization  

The base classes + wire handlers already do that.

---

## 1. Prerequisites

- Python 3.10+ (async-friendly)
- `corpus-sdk` installed:

```bash
pip install corpus-sdk
```

Recommended layout:

```text
your-repo/
  adapters/
    hello_embedding_adapter.py
  services/
    embedding_service.py
  spec/
    IMPLEMENTATION.md
    BEHAVIORAL_CONFORMANCE.md
    SCHEMA_CONFORMANCE.md
  conformance/
    ...
```

---

## 2. Hello World Embedding Adapter (Single File)

This is the smallest useful adapter that:

* Announces its capabilities.
* Implements unary + batch embed.
* Exposes a health endpoint.
* Can pass basic conformance tests.

Create `adapters/hello_embedding_adapter.py`:

```python
from corpus_sdk.embedding.embedding_base import (
    BaseEmbeddingAdapter,
    EmbeddingCapabilities,
    EmbedSpec,
    BatchEmbedSpec,
    EmbedResult,
    BatchEmbedResult,
    EmbeddingVector,
)


class HelloEmbeddingAdapter(BaseEmbeddingAdapter):
    """
    Tiny demo adapter.

    Embedding = [len(text)] + zeros → 8-dim vector.
    Just enough to exercise the protocol and tests.
    """

    async def _do_capabilities(self) -> EmbeddingCapabilities:
        return EmbeddingCapabilities(
            server="hello-embedding",
            version="1.0.0",
            supported_models=("hello-1",),
            max_batch_size=10,
            max_text_length=100,
            max_dimensions=8,
            supports_normalization=False,
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
        # Simple: encode text length as first element of vector.
        vec = [float(len(spec.text))] + [0.0] * 7

        embedding = EmbeddingVector(
            vector=vec,
            text=spec.text,
            model=spec.model,
            dimensions=len(vec),
        )
        return EmbedResult(
            embedding=embedding,
            model=spec.model,
            text=spec.text,
            tokens_used=None,
            truncated=False,  # base will flip this if truncation happened
        )

    async def _do_embed_batch(
        self,
        spec: BatchEmbedSpec,
        *,
        ctx=None,
    ) -> BatchEmbedResult:
        embeddings = []
        for t in spec.texts:
            single = await self._do_embed(
                EmbedSpec(
                    text=t,
                    model=spec.model,
                    truncate=spec.truncate,
                    normalize=False,
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
        # Not used because supports_token_counting=False
        raise NotImplementedError

    async def _do_health(self, *, ctx=None):
        return {"ok": True, "server": "hello-embedding", "version": "1.0.0"}
```

That's it. You've implemented the "provider side" of the adapter.

---

## 3. Expose It Over HTTP (Wire Handler)

Now wire the adapter to HTTP using the wire handler.
Create `services/embedding_service.py`:

```python
from fastapi import FastAPI, Request
from corpus_sdk.embedding.embedding_base import WireEmbeddingHandler

from adapters.hello_embedding_adapter import HelloEmbeddingAdapter

app = FastAPI()

# mode="standalone" → deadlines, caching, rate limiting, breaker
adapter = HelloEmbeddingAdapter(mode="standalone")
handler = WireEmbeddingHandler(adapter)


@app.post("/embedding")
async def embedding_endpoint(request: Request):
    envelope = await request.json()
    resp = await handler.handle(envelope)
    return resp
```

Run it:

```bash
uvicorn services.embedding_service:app --reload
```

*If your app layout differs, just update the uvicorn import path accordingly.*

Send a minimal Corpus envelope:

```bash
curl -X POST http://localhost:8000/embedding \
  -H "Content-Type: application/json" \
  -d '{
    "op": "embedding.embed",
    "ctx": { "tenant": "demo-tenant" },
    "args": {
      "model": "hello-1",
      "text": "hello corpus!"
    }
  }'
```

You should see a response like:

```json
{
  "ok": true,
  "code": "OK",
  "ms": 0.42,
  "result": {
    "embedding": {
      "vector": [13.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      "text": "hello corpus!",
      "model": "hello-1",
      "dimensions": 8
    },
    "model": "hello-1",
    "text": "hello corpus!",
    "tokens_used": null,
    "truncated": false
  }
}
```

Congrats — your service now speaks the Corpus Protocol.

---

## 4. "I Want LLM / Vector / Graph Instead"

Once you understand the Embedding example, the others are the same pattern:

### LLM adapters

Subclass `BaseLLMAdapter` and override:

* `_do_capabilities`
* `_do_complete`
* `_do_stream`
* `_do_count_tokens` (optional; only if `supports_count_tokens=True`)
* `_do_health`

Wire handler:

```python
from corpus_sdk.llm.llm_base import WireLLMHandler

adapter = MyLLMAdapter(mode="standalone")
handler = WireLLMHandler(adapter)
```

### Vector adapters

Subclass `BaseVectorAdapter` and override:

* `_do_capabilities`
* `_do_query`
* `_do_upsert`
* `_do_delete`
* `_do_create_namespace`
* `_do_delete_namespace`
* `_do_health`

Wire handler:

```python
from corpus_sdk.vector.vector_base import WireVectorHandler
```

### Graph adapters

Subclass `BaseGraphAdapter` and override:

* `_do_capabilities`
* `_do_query`
* `_do_stream_query`
* `_do_batch`
* `_do_health`

Wire handler:

```python
from corpus_sdk.graph.graph_base import WireGraphHandler
```

> When in doubt: **pick the base class for your component, implement its `_do_*` hooks, and drop it into the matching `Wire*Handler`.**

For the full method lists & semantics, see `spec/IMPLEMENTATION.md` §7.x.

---

## 5. Timeouts, Tenants, and Errors (60-Second Version)

You'll see `ctx` passed into your `_do_*` methods as an `OperationContext`.
The only things you really need to care about on day one:

### 5.1 Deadlines (`ctx.deadline_ms`)

Use it for provider timeouts:

```python
async def _do_embed(self, spec, *, ctx=None):
    timeout_s = None
    if ctx is not None:
        rem = ctx.remaining_ms()
        if rem and rem > 0:
            timeout_s = rem / 1000.0

    resp = await self._client.embed(
        model=spec.model,
        text=spec.text,
        timeout=timeout_s,
    )
    ...
```

The base:

* Fast-fails if the deadline is already expired.
* Wraps provider calls with its own timeout enforcement.
* Emits the correct `DEADLINE_EXCEEDED` canonical error.

### 5.2 Tenants (`ctx.tenant`)

Use it to select indexes, databases, or projects:

```python
tenant_id = ctx.tenant or "default"
index_name = f"embed-{tenant_id}"
```

The base:

* Hashes tenants for metrics (`tenant_hash`) so you never emit raw IDs.

### 5.3 Error mapping

Inside `_do_*`, catch provider exceptions and map to canonical ones, e.g.:

```python
from corpus_sdk.embedding.embedding_base import (
    ResourceExhausted,
    AuthError,
    BadRequest,
    Unavailable,
)

def map_provider_error(e: Exception):
    if isinstance(e, ProviderRateLimitError):
        return ResourceExhausted("rate limit exceeded", retry_after_ms=5000)
    if isinstance(e, ProviderAuthError):
        return AuthError("invalid credentials")
    if isinstance(e, ProviderInvalidRequest):
        return BadRequest("invalid parameters")
    return Unavailable("provider unavailable")


async def _do_embed(self, spec, *, ctx=None):
    try:
        resp = await self._client.embed(...)
    except Exception as e:
        raise map_provider_error(e)
```

The wire handler will turn this into a structured error envelope.

For the full taxonomy, see `spec/ERRORS.md` and `spec/BEHAVIORAL_CONFORMANCE.md` §6.3.

---

## 6. Run the Conformance Tests

As soon as your adapter can answer basic requests, wire it into the conformance tests.
Typical Makefile targets (adjust to your repo):

```bash
# Everything (schema + behavioral) for all components
make test-conformance

# Per component
make test-llm-conformance
make test-embedding-conformance
make test-vector-conformance
make test-graph-conformance
```

These suites check:

* Wire shapes match the JSON schemas.
* Deadlines, streaming, and error semantics.
* Token counting, truncation, normalization (where applicable).
* Caching and idempotency behavior.
* Observability / SIEM hygiene.

If a test fails, open the reported test file — they're designed to be self-documenting.

---

## 7. What to Read Next

Once the quickstart is working, deepen the adapter:

* **Runtime behavior & patterns**
  `spec/IMPLEMENTATION.md`
  → Full walkthrough of `_do_*` semantics, deadlines, caches, metrics, streaming, batch semantics.

* **Behavioral semantics (normative)**
  `spec/BEHAVIORAL_CONFORMANCE.md`
  → What "correct" looks like: deadlines, error taxonomy, streaming rules, token counting, normalization, etc.

* **Wire shapes & schemas**
  `spec/SCHEMA_CONFORMANCE.md`
  → Canonical envelopes and JSON schemas for all operations.

---

## 8. Adapter Launch Checklist (TL;DR)

Before you ship:

* [ ] Your adapter's `_do_*` methods are implemented against your provider.
* [ ] `capabilities()` values match reality (models, limits, flags).
* [ ] You use `ctx.remaining_ms()` for provider timeouts.
* [ ] Provider errors are mapped to canonical adapter errors.
* [ ] No raw tenant IDs, PII, or full texts are logged or put into metrics.
* [ ] `make test-conformance` passes **unmodified**.

If all those are true, you're ready to plug into a production Corpus routing stack.

---
## 9. What's Next?
- **Deep dive**: See `IMPLEMENTATION.md` for production patterns, error handling, and advanced features
- **Validation**: Run `make test-conformance` to verify your adapter

---
**Maintainers:** Corpus SDK Team

**Scope:** Adapter author quickstart — see `spec/IMPLEMENTATION.md`, `spec/BEHAVIORAL_CONFORMANCE.md`, and `spec/SCHEMA_CONFORMANCE.md` for full details.
