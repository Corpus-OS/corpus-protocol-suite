# Corpus Protocol (v1.0) — Adapter Recipes

> **Goal:** Give adapter authors copy-pasteable patterns for real-world adapters.  
> **Audience:** People who finished `Adapter Quickstart` and now want “how do I do X with my provider?”.

**Read this *after*:**

- `QUICKSTART_ADAPTERS.md` — hello world Embedding adapter
- `IMPLEMENTATION.md` — full runtime semantics & base class behavior
- `BEHAVIORAL_CONFORMANCE.md` — normative behavioral rules
- `SCHEMA_CONFORMANCE.md` — JSON schemas & wire shapes

---

## 0. How to Use This File

Each section is a **recipe**:

- **Pick** the component you care about (LLM / Embedding / Vector / Graph)
- **Copy** the closest recipe into your repo
- **Replace** the fake provider client / error types with your real ones
- **Run** the conformance tests and fix anything they complain about

Conventions:

- All adapters subclass:
  - `BaseLLMAdapter`
  - `BaseEmbeddingAdapter`
  - `BaseVectorAdapter`
  - `BaseGraphAdapter`
- We always use **canonical error types** (e.g. `BadRequest`, `AuthError`, `ResourceExhausted`, …)
- Example provider types are placeholders (`MyProvider*`, `AcmeClient`, etc.)

---

## 1. LLM Adapter Recipes

### 1.1 Minimal Non-Toy LLM Adapter

**Scenario:** Wrap a single upstream LLM with basic completion + streaming.  
**Good for:** “I just want this thing to talk Corpus ASAP.”

```python
# adapters/my_llm_adapter.py
from typing import AsyncIterator, Dict, Any

from corpus_sdk.llm.llm_base import (
    BaseLLMAdapter,
    LLMCapabilities,
    LLMCompletion,
    LLMChunk,
    BadRequest,
)

# Fake provider client + errors; replace with your SDK.
class MyProviderClient:
    async def complete(self, *, model: str, messages, **kwargs) -> Dict[str, Any]:
        # Pretend this hits a real upstream.
        joined = "\n".join(m["content"] for m in messages)
        return {"text": f"[reply to]: {joined[:80]}..."}


class MyLLMAdapter(BaseLLMAdapter):
    def __init__(self, client: MyProviderClient, **kwargs):
        super().__init__(**kwargs)
        self._client = client

    async def _do_capabilities(self) -> LLMCapabilities:
        return LLMCapabilities(
            server="my-llm",
            version="1.0.0",
            supported_models=("my-llm-1",),
            max_context_length=8192,
            supports_streaming=True,
            supports_count_tokens=False,
        )

    async def _do_complete(self, request, *, ctx=None) -> LLMCompletion:
        caps = await self._do_capabilities()
        if request.model not in caps.supported_models:
            raise BadRequest(f"unknown model '{request.model}'")

        resp = await self._client.complete(
            model=request.model,
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        return LLMCompletion(
            text=resp["text"],
            model=request.model,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    async def _do_stream(
        self,
        request,
        *,
        ctx=None,
    ) -> AsyncIterator[LLMChunk]:
        # Easiest legal streaming: one final chunk.
        completion = await self._do_complete(request, ctx=ctx)
        yield LLMChunk(
            text=completion.text,
            is_final=True,
            model=request.model,
        )

    async def _do_count_tokens(self, text: str, model: str, *, ctx=None) -> int:
        # Not used because supports_count_tokens=False
        raise NotImplementedError

    async def _do_health(self, *, ctx=None) -> Dict[str, Any]:
        return {"ok": True, "server": "my-llm", "version": "1.0.0"}
````

Wire it with `WireLLMHandler` exactly like in the quickstart.

---

### 1.2 LLM with Provider Token Counting & Context Limits

**Scenario:** Your provider exposes `count_tokens` and has a hard context window.
**Goal:** Reject over-long prompts *before* hitting the model.

```python
# adapters/my_llm_with_tokens.py
from corpus_sdk.llm.llm_base import (
    BaseLLMAdapter,
    LLMCapabilities,
    LLMCompletion,
    LLMChunk,
    BadRequest,
)

class ProviderClient:
    async def complete(self, *, model, messages, timeout=None, **kwargs):
        ...
    async def count_tokens(self, *, model, text, timeout=None):
        ...

class MyLLMWithTokens(BaseLLMAdapter):
    def __init__(self, client: ProviderClient, **kwargs):
        super().__init__(**kwargs)
        self._client = client

    async def _do_capabilities(self) -> LLMCapabilities:
        return LLMCapabilities(
            server="my-llm-with-tokens",
            version="1.0.0",
            supported_models=("my-llm-ctx8k",),
            max_context_length=8192,
            supports_streaming=True,
            supports_count_tokens=True,  # important
        )

    async def _do_count_tokens(
        self,
        text: str,
        model: str,
        *,
        ctx=None,
    ) -> int:
        timeout_s = None
        if ctx is not None:
            rem = ctx.remaining_ms()
            if rem and rem > 0:
                timeout_s = rem / 1000.0

        resp = await self._client.count_tokens(
            model=model,
            text=text,
            timeout=timeout_s,
        )
        return int(resp["tokens"])

    async def _do_complete(self, request, *, ctx=None) -> LLMCompletion:
        # Base class will call _do_count_tokens and enforce max_context_length.
        timeout_s = None
        if ctx is not None:
            rem = ctx.remaining_ms()
            if rem and rem > 0:
                timeout_s = rem / 1000.0

        resp = await self._client.complete(
            model=request.model,
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            timeout=timeout_s,
        )

        return LLMCompletion(
            text=resp["text"],
            model=request.model,
            usage=resp.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
        )

    async def _do_stream(self, request, *, ctx=None):
        # Could stream here; see 1.3 for a full example.
        raise BadRequest("streaming not implemented yet")

    async def _do_health(self, *, ctx=None):
        return {"ok": True, "server": "my-llm-with-tokens", "version": "1.0.0"}
```

---

### 1.3 LLM Streaming Adapter (Chunked Provider Response)

**Scenario:** Provider streams tokens/chunks; you want full Corpus streaming semantics.

```python
# adapters/my_llm_streaming.py
from typing import AsyncIterator, Dict, Any

from corpus_sdk.llm.llm_base import (
    BaseLLMAdapter,
    LLMCapabilities,
    LLMCompletion,
    LLMChunk,
    Unavailable,
)

class ProviderStreamError(Exception):
    ...

class StreamingProviderClient:
    async def stream_complete(self, *, model, messages, timeout=None, **kwargs):
        """
        Yields dicts like:
        {"text": "...", "is_final": False}
        and one final: {"text": "...", "is_final": True}
        """
        ...

class MyStreamingLLMAdapter(BaseLLMAdapter):
    def __init__(self, client: StreamingProviderClient, **kwargs):
        super().__init__(**kwargs)
        self._client = client

    async def _do_capabilities(self) -> LLMCapabilities:
        return LLMCapabilities(
            server="my-llm-streaming",
            version="1.0.0",
            supported_models=("my-llm-stream-1",),
            max_context_length=8192,
            supports_streaming=True,
            supports_count_tokens=False,
        )

    async def _do_complete(self, request, *, ctx=None) -> LLMCompletion:
        # Simple “gather from streaming” implementation.
        text_parts = []
        async for chunk in self._do_stream(request, ctx=ctx):
            text_parts.append(chunk.text)
        full_text = "".join(text_parts)
        return LLMCompletion(
            text=full_text,
            model=request.model,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    async def _do_stream(
        self,
        request,
        *,
        ctx=None,
    ) -> AsyncIterator[LLMChunk]:
        timeout_s = None
        if ctx is not None:
            rem = ctx.remaining_ms()
            if rem and rem > 0:
                timeout_s = rem / 1000.0

        try:
            async for prov_chunk in self._client.stream_complete(
                model=request.model,
                messages=request.messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                timeout=timeout_s,
            ):
                yield LLMChunk(
                    text=prov_chunk.get("text", ""),
                    is_final=bool(prov_chunk.get("is_final", False)),
                    model=request.model,
                )
        except ProviderStreamError as e:
            # Base will convert this to a canonical error envelope.
            raise Unavailable(str(e))

    async def _do_count_tokens(self, text: str, model: str, *, ctx=None) -> int:
        raise NotImplementedError

    async def _do_health(self, *, ctx=None) -> Dict[str, Any]:
        return {"ok": True, "server": "my-llm-streaming", "version": "1.0.0"}
```

---

### 1.4 LLM Error Mapping Table

**Scenario:** Provider has many HTTP codes / error types; you want one *central* mapping.

```python
# adapters/provider_errors_llm.py
from corpus_sdk.llm.llm_base import (
    LLMAdapterError,
    BadRequest,
    AuthError,
    ResourceExhausted,
    Unavailable,
    DeadlineExceeded,
)

class ProviderRateLimitError(Exception): ...
class ProviderAuthError(Exception): ...
class ProviderBadRequest(Exception): ...
class ProviderTimeout(Exception): ...
class ProviderServerError(Exception): ...

def map_provider_error_llm(e: Exception) -> LLMAdapterError:
    if isinstance(e, ProviderRateLimitError):
        return ResourceExhausted("rate limit exceeded", retry_after_ms=5000)
    if isinstance(e, ProviderAuthError):
        return AuthError("invalid credentials")
    if isinstance(e, ProviderBadRequest):
        return BadRequest("invalid request")
    if isinstance(e, ProviderTimeout):
        return DeadlineExceeded("upstream timeout")
    if isinstance(e, ProviderServerError):
        return Unavailable("upstream server error")
    # Fallback
    return Unavailable("unknown provider error")
```

Usage inside your adapter:

```python
from .provider_errors_llm import map_provider_error_llm

async def _do_complete(self, request, *, ctx=None):
    try:
        resp = await self._client.complete(...)
    except Exception as e:
        raise map_provider_error_llm(e)
    ...
```

---

## 2. Embedding Adapter Recipes

### 2.1 Single-Model Embedding with Truncation

**Scenario:** Provider has a strict `max_text_length`; you want the base to handle it.

```python
# adapters/my_embedding_single.py
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

class ProviderEmbeddingClient:
    async def embed(self, *, model, text, timeout=None) -> Dict[str, Any]:
        ...

class MySingleModelEmbeddingAdapter(BaseEmbeddingAdapter):
    def __init__(self, client: ProviderEmbeddingClient, **kwargs):
        super().__init__(**kwargs)
        self._client = client

    async def _do_capabilities(self) -> EmbeddingCapabilities:
        return EmbeddingCapabilities(
            server="my-embed-single",
            version="1.0.0",
            supported_models=("embed-1",),
            max_batch_size=64,
            max_text_length=2048,  # base enforces + truncates
            max_dimensions=768,
            supports_normalization=True,
            supports_truncation=True,
            supports_token_counting=False,
            normalizes_at_source=False,
        )

    async def _do_embed(self, spec: EmbedSpec, *, ctx=None) -> EmbedResult:
        caps = await self._do_capabilities()
        if spec.model not in caps.supported_models:
            raise ModelNotAvailable(f"model '{spec.model}' is not supported")

        timeout_s = None
        if ctx is not None:
            rem = ctx.remaining_ms()
            if rem and rem > 0:
                timeout_s = rem / 1000.0

        resp = await self._client.embed(
            model=spec.model,
            text=spec.text,  # already truncated by base if needed
            timeout=timeout_s,
        )

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
            truncated=False,  # base flips this if truncation happened
        )

    async def _do_embed_batch(self, spec: BatchEmbedSpec, *, ctx=None) -> BatchEmbedResult:
        # For simplicity, build on single-embed path.
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

    async def _do_count_tokens(self, text: str, model: str, *, ctx=None) -> int:
        raise NotImplementedError

    async def _do_health(self, *, ctx=None) -> Dict[str, Any]:
        return {"ok": True, "server": "my-embed-single", "version": "1.0.0"}
```

---

### 2.2 Multi-Model Embedding (Small + Large)

**Scenario:** Two models with different limits; you want one adapter.

```python
# adapters/my_embedding_multi.py
from dataclasses import dataclass

from corpus_sdk.embedding.embedding_base import (
    BaseEmbeddingAdapter,
    EmbeddingCapabilities,
    EmbedSpec,
    EmbedResult,
    BatchEmbedSpec,
    BatchEmbedResult,
    EmbeddingVector,
    ModelNotAvailable,
)

@dataclass(frozen=True)
class EmbeddingModelConfig:
    name: str
    max_text_length: int
    dimensions: int

class ProviderClient:
    async def embed(self, *, model, text, timeout=None):
        ...

class MultiModelEmbeddingAdapter(BaseEmbeddingAdapter):
    def __init__(self, client: ProviderClient, **kwargs):
        super().__init__(**kwargs)
        self._client = client
        self._models = {
            "embed-small": EmbeddingModelConfig("embed-small", max_text_length=1024, dimensions=384),
            "embed-large": EmbeddingModelConfig("embed-large", max_text_length=4096, dimensions=1536),
        }

    async def _do_capabilities(self) -> EmbeddingCapabilities:
        # Take the strictest (smallest) max_text_length across models, or document per-model behavior.
        max_len = min(cfg.max_text_length for cfg in self._models.values())
        max_dims = max(cfg.dimensions for cfg in self._models.values())
        return EmbeddingCapabilities(
            server="my-embed-multi",
            version="1.0.0",
            supported_models=tuple(self._models.keys()),
            max_batch_size=128,
            max_text_length=max_len,
            max_dimensions=max_dims,
            supports_normalization=True,
            supports_truncation=True,
            supports_token_counting=False,
            normalizes_at_source=False,
        )

    def _get_model_config(self, model_name: str) -> EmbeddingModelConfig:
        try:
            return self._models[model_name]
        except KeyError:
            raise ModelNotAvailable(f"model '{model_name}' is not supported")

    async def _do_embed(self, spec: EmbedSpec, *, ctx=None) -> EmbedResult:
        cfg = self._get_model_config(spec.model)

        timeout_s = None
        if ctx is not None:
            rem = ctx.remaining_ms()
            if rem and rem > 0:
                timeout_s = rem / 1000.0

        resp = await self._client.embed(
            model=cfg.name,
            text=spec.text,
            timeout=timeout_s,
        )
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
            truncated=False,
        )

    async def _do_embed_batch(self, spec: BatchEmbedSpec, *, ctx=None) -> BatchEmbedResult:
        cfg = self._get_model_config(spec.model)
        embeddings = []
        for t in spec.texts:
            single = await self._do_embed(
                EmbedSpec(
                    text=t,
                    model=cfg.name,
                    truncate=spec.truncate,
                    normalize=False,
                ),
                ctx=ctx,
            )
            embeddings.append(single.embedding)

        return BatchEmbedResult(
            embeddings=embeddings,
            model=cfg.name,
            total_texts=len(spec.texts),
            total_tokens=None,
            failed_texts=[],
        )

    async def _do_count_tokens(self, text: str, model: str, *, ctx=None) -> int:
        raise NotImplementedError

    async def _do_health(self, *, ctx=None):
        return {"ok": True, "server": "my-embed-multi", "version": "1.0.0"}
```

---

### 2.3 Batch Embedding with Native Batch API + Partial Failure

```python
# adapters/my_embedding_batch.py
from typing import Dict, Any, List

from corpus_sdk.embedding.embedding_base import (
    BaseEmbeddingAdapter,
    EmbeddingCapabilities,
    BatchEmbedSpec,
    BatchEmbedResult,
    EmbedSpec,
    EmbedResult,
    EmbeddingVector,
    EmbeddingAdapterError,
)

class ProviderBatchError(Exception):
    """Wraps partial failures:
    .failures: List[{"index": int, "error": Exception}]
    """
    def __init__(self, failures: List[Dict[str, Any]]):
        self.failures = failures

class ProviderClient:
    async def embed_batch(self, *, model, texts, timeout=None):
        ...

class BatchEmbeddingAdapter(BaseEmbeddingAdapter):
    def __init__(self, client: ProviderClient, **kwargs):
        super().__init__(**kwargs)
        self._client = client

    async def _do_capabilities(self) -> EmbeddingCapabilities:
        return EmbeddingCapabilities(
            server="my-embed-batch",
            version="1.0.0",
            supported_models=("embed-batch-1",),
            max_batch_size=256,
            max_text_length=8192,
            max_dimensions=1536,
            supports_normalization=True,
            supports_truncation=True,
            supports_token_counting=False,
            normalizes_at_source=False,
        )

    async def _do_embed(self, spec: EmbedSpec, *, ctx=None) -> EmbedResult:
        # Used if you want fallback behavior; can be trivial.
        batch = await self._do_embed_batch(
            BatchEmbedSpec(
                texts=[spec.text],
                model=spec.model,
                truncate=spec.truncate,
                normalize=spec.normalize,
            ),
            ctx=ctx,
        )
        return EmbedResult(
            embedding=batch.embeddings[0],
            model=batch.model,
            text=spec.text,
            tokens_used=None,
            truncated=False,
        )

    async def _do_embed_batch(
        self,
        spec: BatchEmbedSpec,
        *,
        ctx=None,
    ) -> BatchEmbedResult:
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
        except ProviderBatchError as e:
            # Map per-item failures; succeed others.
            embeddings: List[EmbeddingVector] = []
            failed = []
            for idx, text in enumerate(spec.texts):
                failure = next((f for f in e.failures if f["index"] == idx), None)
                if failure:
                    err: EmbeddingAdapterError = failure["error"]
                    failed.append(
                        {
                            "index": idx,
                            "text": text,
                            "code": err.code,
                            "error": err.__class__.__name__,
                            "message": str(err),
                        }
                    )
                else:
                    # In a real integration you'd pull the vector from resp.
                    # Here we just fake one.
                    embeddings.append(
                        EmbeddingVector(
                            vector=[1.0],
                            text=text,
                            model=spec.model,
                            dimensions=1,
                        )
                    )

            return BatchEmbedResult(
                embeddings=embeddings,
                model=spec.model,
                total_texts=len(spec.texts),
                total_tokens=None,
                failed_texts=failed,
            )

        # Happy path: all good.
        vectors = resp["vectors"]
        embeddings = [
            EmbeddingVector(
                vector=v,
                text=t,
                model=spec.model,
                dimensions=len(v),
            )
            for t, v in zip(spec.texts, vectors)
        ]
        return BatchEmbedResult(
            embeddings=embeddings,
            model=spec.model,
            total_texts=len(spec.texts),
            total_tokens=resp.get("total_tokens"),
            failed_texts=[],
        )
```

---

### 2.4 Embedding + Normalization + Cache

**Scenario:** You want normalized vectors and a small in-process cache.

```python
# wiring in your service, not in the adapter itself
from corpus_sdk.embedding.embedding_base import (
    InMemoryTTLCache,
)
from adapters.my_embedding_single import MySingleModelEmbeddingAdapter

cache = InMemoryTTLCache(max_items=10_000)

adapter = MySingleModelEmbeddingAdapter(
    client=ProviderEmbeddingClient(),
    mode="standalone",
    cache=cache,
    cache_embed_ttl_s=60,
)

# Base will:
# - Cache embed() calls (tenant-aware, content-addressed)
# - Normalize vectors when spec.normalize=True and caps.supports_normalization=True
```

(See `IMPLEMENTATION.md` for cache key details.)

---

## 3. Vector Adapter Recipes

### 3.1 In-Memory Vector Store (Great for Tests)

```python
# adapters/in_memory_vector_adapter.py
from typing import Dict, Any, List
from math import sqrt

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
)

def _l2(a: List[float], b: List[float]) -> float:
    return sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

class InMemoryVectorAdapter(BaseVectorAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # namespace -> id -> Vector
        self._store: Dict[str, Dict[str, Vector]] = {}

    async def _do_capabilities(self) -> VectorCapabilities:
        return VectorCapabilities(
            server="vector-in-memory",
            version="1.0.0",
            max_dimensions=2048,
            max_top_k=100,
            max_batch_size=1024,
            supports_metadata_filtering=False,
            supports_index_management=True,
        )

    async def _do_query(self, spec: QuerySpec, *, ctx=None) -> QueryResult:
        ns = self._store.get(spec.namespace, {})
        matches: List[VectorMatch] = []
        for v in ns.values():
            # Super naive: L2 distance + convert to [0,1] score.
            d = _l2(spec.vector, v.vector)
            score = 1.0 / (1.0 + d)
            matches.append(
                VectorMatch(
                    vector=v,
                    score=score,
                    distance=d,
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
        for v in spec.vectors:
            ns[str(v.id)] = v
        return UpsertResult(
            upserted_count=len(spec.vectors),
            failed_count=0,
            failures=[],
        )

    async def _do_delete(self, spec: DeleteSpec, *, ctx=None) -> DeleteResult:
        ns = self._store.setdefault(spec.namespace, {})
        deleted = 0
        for vid in spec.ids:
            if str(vid) in ns:
                del ns[str(vid)]
                deleted += 1
        return DeleteResult(
            deleted_count=deleted,
            failed_count=0,
            failures=[],
        )

    async def _do_create_namespace(self, spec: NamespaceSpec, *, ctx=None) -> NamespaceResult:
        self._store.setdefault(spec.namespace, {})
        return NamespaceResult(success=True, namespace=spec.namespace, details={})

    async def _do_delete_namespace(self, namespace: str, *, ctx=None) -> NamespaceResult:
        self._store.pop(namespace, None)
        return NamespaceResult(success=True, namespace=namespace, details={})

    async def _do_health(self, *, ctx=None) -> Dict[str, Any]:
        return {"ok": True, "server": "vector-in-memory", "version": "1.0.0"}
```

Great for unit tests and CI; don’t use for serious workloads.

---

### 3.2 Hosted Vector DB Wrapper (Shape)

**Scenario:** You’re wrapping a real vector DB (Pinecone/Weaviate/pgvector-like).
**Pattern:** Namespace + capabilities enforcement.

```python
# adapters/hosted_vector_adapter.py
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
    BadRequest,
)

class HostedVectorClient:
    async def query(self, *, index_name, vector, top_k, filter=None, **kwargs):
        ...
    async def upsert(self, *, index_name, vectors: List[Dict[str, Any]], **kwargs):
        ...
    async def delete(self, *, index_name, ids: List[str], **kwargs):
        ...
    async def create_index(self, *, name: str, dimension: int, **kwargs):
        ...
    async def delete_index(self, *, name: str, **kwargs):
        ...

class HostedVectorAdapter(BaseVectorAdapter):
    def __init__(self, client: HostedVectorClient, **kwargs):
        super().__init__(**kwargs)
        self._client = client

    async def _do_capabilities(self) -> VectorCapabilities:
        return VectorCapabilities(
            server="hosted-vector",
            version="1.0.0",
            max_dimensions=1536,
            max_top_k=100,
            max_batch_size=500,
            supports_metadata_filtering=True,
            supports_index_management=True,
        )

    def _index_name(self, namespace: str) -> str:
        return f"corpus-{namespace or 'default'}"

    async def _do_query(self, spec: QuerySpec, *, ctx=None) -> QueryResult:
        caps = await self._do_capabilities()
        if len(spec.vector) > caps.max_dimensions:
            raise BadRequest("vector dimensions exceed max_dimensions")

        resp = await self._client.query(
            index_name=self._index_name(spec.namespace),
            vector=spec.vector,
            top_k=spec.top_k,
            filter=spec.filter if caps.supports_metadata_filtering else None,
        )

        matches = [
            VectorMatch(
                vector=Vector(
                    id=str(item["id"]),
                    vector=item["vector"] if spec.include_vectors else [],
                    metadata=item.get("metadata") if spec.include_metadata else None,
                ),
                score=item["score"],
                distance=item.get("distance"),
            )
            for item in resp["matches"]
        ]
        return QueryResult(
            matches=matches,
            query_vector=spec.vector,
            namespace=spec.namespace,
            total_matches=resp.get("total_matches", len(matches)),
        )

    async def _do_upsert(self, spec: UpsertSpec, *, ctx=None) -> UpsertResult:
        caps = await self._do_capabilities()
        if len(spec.vectors) > caps.max_batch_size:
            raise BadRequest("too many vectors in one batch")

        payload = [
            {
                "id": str(v.id),
                "vector": v.vector,
                "metadata": v.metadata or {},
            }
            for v in spec.vectors
        ]

        await self._client.upsert(
            index_name=self._index_name(spec.namespace),
            vectors=payload,
        )
        return UpsertResult(
            upserted_count=len(spec.vectors),
            failed_count=0,
            failures=[],
        )

    async def _do_delete(self, spec: DeleteSpec, *, ctx=None) -> DeleteResult:
        await self._client.delete(
            index_name=self._index_name(spec.namespace),
            ids=[str(i) for i in spec.ids],
        )
        return DeleteResult(
            deleted_count=len(spec.ids),
            failed_count=0,
            failures=[],
        )

    async def _do_create_namespace(self, spec: NamespaceSpec, *, ctx=None) -> NamespaceResult:
        caps = await self._do_capabilities()
        await self._client.create_index(
            name=self._index_name(spec.namespace),
            dimension=caps.max_dimensions,
        )
        return NamespaceResult(success=True, namespace=spec.namespace, details={})

    async def _do_delete_namespace(self, namespace: str, *, ctx=None) -> NamespaceResult:
        await self._client.delete_index(name=self._index_name(namespace))
        return NamespaceResult(success=True, namespace=namespace, details={})

    async def _do_health(self, *, ctx=None) -> Dict[str, Any]:
        return {"ok": True, "server": "hosted-vector", "version": "1.0.0"}
```

---

### 3.3 Multi-Tenant Vector Patterns

Two common patterns:

1. **Namespace per tenant**

   * `namespace = f"{ctx.tenant}--{logical_namespace}"`
   * Good isolation; more indexes.

2. **Single index, tenant in metadata**

   * Shared index; filter on `tenant_id`
   * Fewer indexes; more complex filters.

Both are valid; pick based on your provider and scale.

---

### 3.4 Partial Failure Reporting (Upsert/Delete)

When provider returns per-item status, map failures into `failures` list:

```python
# inside _do_upsert
failures = []
success_count = 0
for idx, item in enumerate(provider_result["items"]):
    if item["status"] == "ok":
        success_count += 1
    else:
        failures.append(
            {
                "index": idx,
                "id": spec.vectors[idx].id,
                "code": "UNAVAILABLE",
                "message": item.get("error", "unknown"),
            }
        )

return UpsertResult(
    upserted_count=success_count,
    failed_count=len(failures),
    failures=failures,
)
```

Same pattern applies to `DeleteResult`.

---

## 4. Graph Adapter Recipes

### 4.1 Unary Graph Query (Cypher / SQL-ish)

```python
# adapters/my_graph_adapter.py
from typing import Dict, Any, List

from corpus_sdk.graph.graph_base import (
    BaseGraphAdapter,
    GraphCapabilities,
    GraphRow,
    GraphStreamEvent,
)

class GraphDBClient:
    async def query(self, *, dialect: str, text: str, params: Dict[str, Any], timeout=None):
        ...

class MyGraphAdapter(BaseGraphAdapter):
    def __init__(self, client: GraphDBClient, **kwargs):
        super().__init__(**kwargs)
        self._client = client

    async def _do_capabilities(self) -> GraphCapabilities:
        return GraphCapabilities(
            server="my-graph",
            version="1.0.0",
            dialect="cypher",
            supports_streaming=True,
        )

    async def _do_query(self, spec, *, ctx=None):
        timeout_s = None
        if ctx is not None:
            rem = ctx.remaining_ms()
            if rem and rem > 0:
                timeout_s = rem / 1000.0

        rows = await self._client.query(
            dialect="cypher",
            text=spec.query,
            params=spec.params,
            timeout=timeout_s,
        )

        # rows is e.g. List[Dict[str, Any]]
        graph_rows = [GraphRow(data=row) for row in rows]
        return {"rows": graph_rows}

    async def _do_stream_query(self, spec, *, ctx=None):
        # See 4.2 for a full streaming example
        rows = (await self._do_query(spec, ctx=ctx))["rows"]
        for row in rows:
            yield GraphStreamEvent(data=row, kind="row")
        yield GraphStreamEvent(data=None, kind="end")

    async def _do_batch(self, spec, *, ctx=None):
        # Implement according to BEHAVIORAL_CONFORMANCE.md
        raise NotImplementedError

    async def _do_health(self, *, ctx=None) -> Dict[str, Any]:
        return {"ok": True, "server": "my-graph", "version": "1.0.0"}
```

---

### 4.2 Streaming Graph Query

```python
# adapters/my_graph_streaming.py
from typing import AsyncIterator

from corpus_sdk.graph.graph_base import (
    BaseGraphAdapter,
    GraphCapabilities,
    GraphStreamEvent,
    GraphRow,
)

class StreamingGraphClient:
    async def stream(self, *, query: str, params: dict, timeout=None):
        """
        async generator yielding rows (dicts).
        """
        ...

class StreamingGraphAdapter(BaseGraphAdapter):
    def __init__(self, client: StreamingGraphClient, **kwargs):
        super().__init__(**kwargs)
        self._client = client

    async def _do_capabilities(self) -> GraphCapabilities:
        return GraphCapabilities(
            server="my-graph-streaming",
            version="1.0.0",
            dialect="cypher",
            supports_streaming=True,
        )

    async def _do_stream_query(
        self,
        spec,
        *,
        ctx=None,
    ) -> AsyncIterator[GraphStreamEvent]:
        timeout_s = None
        if ctx is not None:
            rem = ctx.remaining_ms()
            if rem and rem > 0:
                timeout_s = rem / 1000.0

        async for row in self._client.stream(
            query=spec.query,
            params=spec.params,
            timeout=timeout_s,
        ):
            yield GraphStreamEvent(
                data=GraphRow(data=row),
                kind="row",
            )

        # Exactly one terminal event.
        yield GraphStreamEvent(data=None, kind="end")
```

(Implement `_do_query`, `_do_batch`, `_do_health` as in 4.1.)

---

## 5. Cross-Cutting Recipes

### 5.1 Deadline Helper

Use the same helper everywhere:

```python
# adapters/utils_time.py
from typing import Optional
from corpus_sdk.common.context import OperationContext  # adjust import if needed

def timeout_from_ctx(ctx: Optional[OperationContext]) -> Optional[float]:
    if ctx is None:
        return None
    rem = ctx.remaining_ms()
    if rem is None or rem <= 0:
        return None
    return rem / 1000.0
```

Usage:

```python
from .utils_time import timeout_from_ctx

async def _do_embed(self, spec, *, ctx=None):
    timeout_s = timeout_from_ctx(ctx)
    resp = await self._client.embed(..., timeout=timeout_s)
    ...
```

---

### 5.2 Simple Retry + Backoff Wrapper

**Be careful:** only retry when safe (idempotent operations or when your provider guarantees idempotency).

```python
# adapters/utils_retry.py
import asyncio
from typing import Awaitable, Callable, TypeVar, Iterable

from corpus_sdk.embedding.embedding_base import (
    TransientNetwork,
    Unavailable,
    ResourceExhausted,
)

T = TypeVar("T")

def _is_retryable(e: Exception) -> bool:
    if isinstance(e, (TransientNetwork, Unavailable)):
        return True
    if isinstance(e, ResourceExhausted) and getattr(e, "retry_after_ms", None):
        return True
    return False

async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay_s: float = 0.1,
) -> T:
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            return await fn()
        except Exception as e:
            if not _is_retryable(e) or i == attempts - 1:
                raise
            last_exc = e
            delay = base_delay_s * (2 ** i)
            await asyncio.sleep(delay)
    assert False, "unreachable"
```

Usage:

```python
from .utils_retry import retry_async

async def _do_embed(self, spec, *, ctx=None):
    async def call():
        return await self._client.embed(...)

    try:
        resp = await retry_async(call)
    except Exception as e:
        raise map_provider_error_embedding(e)
```

---

### 5.3 Tenant-Safe Logging & Metrics

**Pattern:**

* Use `tenant_hash` in metrics (base already does).
* Do **not** log raw tenant IDs or full texts/embeddings.

Example wrapper:

```python
# adapters/utils_logging.py
import hashlib
from typing import Optional

def hash_tenant(tenant: Optional[str]) -> str:
    if not tenant:
        return "none"
    return hashlib.sha256(tenant.encode("utf-8")).hexdigest()[:16]

def log_request(component: str, op: str, tenant: Optional[str]):
    th = hash_tenant(tenant)
    # Replace with your logger
    print(f"[{component}.{op}] tenant_hash={th}")
```

Usage inside `_do_*`:

```python
from .utils_logging import log_request

async def _do_embed(self, spec, *, ctx=None):
    log_request("embedding", "embed", getattr(ctx, "tenant", None))
    ...
```

---

## 6. “Real-ish” Provider: Multiple Adapters, Shared Pieces

**Suggested layout:**

```text
adapters/
  acme/
    __init__.py
    client.py               # AcmeClient (raw HTTP/gRPC)
    errors.py               # provider → Corpus error mapping
    utils_time.py           # timeout helper
    utils_retry.py          # retry helper

    llm_adapter.py          # AcmeLLMAdapter(BaseLLMAdapter)
    embedding_adapter.py    # AcmeEmbeddingAdapter(BaseEmbeddingAdapter)
    vector_adapter.py       # AcmeVectorAdapter(BaseVectorAdapter)
    graph_adapter.py        # AcmeGraphAdapter(BaseGraphAdapter)
```

Shared pieces:

* `client.py` — base “AcmeClient” used by all adapters.
* `errors.py` — `map_error_llm`, `map_error_embedding`, etc.
* `utils_time.py` — `timeout_from_ctx`.
* `utils_retry.py` — safe retry wrapper.

Then each adapter file is ~50–100 lines of *just*:

* Capabilities
* `_do_*` implementations
* `AcmeClient` calls
* Error mapping

Everything else (deadlines, metrics, circuit breaking, caching, envelopes) comes from the SDK.

---

## 7. Choosing the Right Recipe

Quick “what do I start from?” table:

| Use case                            | Start with section             |
| ----------------------------------- | ------------------------------ |
| Simple LLM, no streaming, no tokens | **1.1 Minimal LLM**            |
| LLM with strict context window      | **1.2 LLM + token counting**   |
| Streaming LLM completions           | **1.3 Streaming LLM**          |
| Simple Embedding backend            | **2.1 Single-model Embedding** |
| Embedding small+large models        | **2.2 Multi-model Embedding**  |
| Provider has batch Embedding API    | **2.3 Batch Embedding**        |
| Test/CI-only vector backend         | **3.1 In-memory Vector**       |
| Hosted vector DB integration        | **3.2 Hosted Vector**          |
| Graph DB unary queries              | **4.1 Unary Graph**            |
| Graph DB with streaming             | **4.2 Streaming Graph**        |

---

## 8. Next Steps

Once you’ve cloned a recipe:

1. Swap out the **provider client** and **error types** for your real ones.
2. Make sure `capabilities()` truly matches what the provider can do.
3. Run the conformance tests:

   * `make test-llm-conformance`
   * `make test-embedding-conformance`
   * `make test-vector-conformance`
   * `make test-graph-conformance`
4. Only after all suites pass **unmodified**, wire your adapter into a production routing stack.

---

**Maintainers:** Corpus SDK Team
**Scope:** Practical patterns for implementing adapters; see `IMPLEMENTATION.md`, `BEHAVIORAL_CONFORMANCE.md`, and `SCHEMA_CONFORMANCE.md` for normative details.

```

