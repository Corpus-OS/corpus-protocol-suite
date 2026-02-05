# tests/frameworks/vector/test_semantickernel_vector_adapter.py

"""
Semantic Kernel Vector framework adapter tests.

These tests are written against the current public API in
`corpus_sdk.vector.framework_adapters.semantic_kernel`, which exposes a
Semantic Kernel adapter layer backed by Corpus VectorProtocolV1.

Update note (alignment):
- The adapter surface under test is now:
  - `CorpusSemanticKernelVectorStore` (core framework-agnostic store)
  - `CorpusSemanticKernelVectorPlugin` (Semantic Kernel tool/plugin wrapper)

- Prior IMemoryStore/MemoryRecord-based tests have been replaced with store/plugin
  contract tests aligned to the current implementation. All existing section
  headers and documentation were preserved and expanded for clarity.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import math
import sys
import uuid
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Awaitable, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple
from unittest.mock import Mock, patch

import pytest

import corpus_sdk.vector.framework_adapters.semantic_kernel as sk_adapter_module
from corpus_sdk.vector.framework_adapters.semantic_kernel import (
    CorpusSemanticKernelVectorPlugin,
    CorpusSemanticKernelVectorStore,
)
from corpus_sdk.vector.vector_base import (
    BadRequest,
    NotSupported,
    OperationContext,
    QueryResult,
    QuerySpec,
    UpsertResult,
    UpsertSpec,
    Vector,
    VectorAdapterError,
    VectorCapabilities,
    VectorMatch,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_TEXT = "hello from semantic kernel vector tests"
SAMPLE_EMBEDDING = [0.1, 0.2, 0.3, 0.4]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vector_match(
    *,
    vid: str,
    score: float,
    text_field: str,
    id_field: str,
    text: str,
    embedding: Optional[Sequence[float]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    namespace: Optional[str] = "default",
) -> VectorMatch:
    """
    Build a realistic VectorMatch with a Corpus `Vector` embedded in it.

    The production adapter reads content from:
      - match.vector.metadata[text_field]
    and strips reserved fields:
      - text_field, id_field
    """
    meta = dict(metadata or {})
    meta[text_field] = text
    meta[id_field] = vid

    vec = Vector(
        id=vid,
        vector=list(embedding or []),
        metadata=meta,
        namespace=namespace,
        text=None,
    )
    return VectorMatch(vector=vec, score=score, distance=1.0 - score)


def _make_dummy_translator(
    *,
    text_field: str = "page_content",
    id_field: str = "id",
    include_low_score: bool = False,
    long_text: bool = False,
    include_internal_fields: bool = False,
    include_vectors: bool = True,
) -> Any:
    """
    Factory for creating a standard dummy translator for tests.

    The DummyTranslator mimics the subset of VectorTranslator methods used by
    the adapter layer:
      - upsert/arun_upsert
      - query/arun_query
      - query_stream (streaming)
      - delete/arun_delete
      - capabilities/arun_capabilities
    """

    # Build a single match with expected metadata shape.
    match_text = ("x" * 650) if long_text else "test"
    user_meta: Dict[str, Any] = {"custom": "data"}

    # Add internal fields that should be filtered out from AI-facing output.
    if include_internal_fields:
        user_meta.update(
            {
                "id": "internal-id",
                "_id": "internal-_id",
                "vector": [9, 9, 9],
                "_vector": [8, 8, 8],
                "embedding": [7, 7, 7],
                "timestamp": "2024-01-01T00:00:00Z",
            }
        )

    matches: List[VectorMatch] = [
        _make_vector_match(
            vid="key-0",
            score=0.95,
            text_field=text_field,
            id_field=id_field,
            text=match_text,
            embedding=[0.1, 0.2] if include_vectors else None,
            metadata=user_meta,
            namespace="default",
        )
    ]

    if include_low_score:
        matches.append(
            _make_vector_match(
                vid="key-1",
                score=0.3,
                text_field=text_field,
                id_field=id_field,
                text="low",
                embedding=[0.3, 0.4] if include_vectors else None,
                metadata={"custom": "low"},
                namespace="default",
            )
        )

    class DummyTranslator:
        def __init__(self) -> None:
            # Captured requests for assertions in tests.
            self.last_upsert_raw: Optional[Dict[str, Any]] = None
            self.last_query_raw: Optional[Dict[str, Any]] = None
            self.last_delete_raw: Optional[Dict[str, Any]] = None
            self.last_framework_ctx: Optional[Dict[str, Any]] = None
            self.last_op_ctx: Optional[OperationContext] = None

        def upsert(self, raw_request: Any, *, op_ctx: Any = None, framework_ctx: Any = None, **_: Any) -> Any:
            self.last_upsert_raw = raw_request
            self.last_framework_ctx = framework_ctx
            self.last_op_ctx = op_ctx
            # raw_request is now a list of document dicts
            count = len(raw_request) if isinstance(raw_request, list) else len(raw_request.get("vectors", []) or [])
            return UpsertResult(
                upserted_count=count,
                failed_count=0,
                failures=[],
            )

        async def arun_upsert(self, raw_request: Any, *, op_ctx: Any = None, framework_ctx: Any = None, **_: Any) -> Any:
            self.last_upsert_raw = raw_request
            self.last_framework_ctx = framework_ctx
            self.last_op_ctx = op_ctx
            # raw_request is now a list of document dicts
            count = len(raw_request) if isinstance(raw_request, list) else len(raw_request.get("vectors", []) or [])
            return UpsertResult(
                upserted_count=count,
                failed_count=0,
                failures=[],
            )

        def query(self, raw_query: Any, *, op_ctx: Any = None, framework_ctx: Any = None, **_: Any) -> Any:
            self.last_query_raw = raw_query
            self.last_framework_ctx = framework_ctx
            self.last_op_ctx = op_ctx
            return QueryResult(
                matches=matches, 
                namespace=raw_query.get("namespace") or "default",
                query_vector=raw_query.get("query_vector") or [],
                total_matches=len(matches)
            )

        async def arun_query(self, raw_query: Any, *, op_ctx: Any = None, framework_ctx: Any = None, **_: Any) -> Any:
            self.last_query_raw = raw_query
            self.last_framework_ctx = framework_ctx
            self.last_op_ctx = op_ctx
            return QueryResult(
                matches=matches, 
                namespace=raw_query.get("namespace") or "default",
                query_vector=raw_query.get("query_vector") or [],
                total_matches=len(matches)
            )

        def query_stream(self, raw_query: Any, *, op_ctx: Any = None, framework_ctx: Any = None, **_: Any) -> Iterator[Any]:
            self.last_query_raw = raw_query
            self.last_framework_ctx = framework_ctx
            self.last_op_ctx = op_ctx
            # Stream a single chunk for deterministic tests.
            yield QueryResult(
                matches=matches, 
                namespace=raw_query.get("namespace") or "default",
                query_vector=raw_query.get("query_vector") or [],
                total_matches=len(matches)
            )

        def delete(self, raw_request: Any, *, op_ctx: Any = None, framework_ctx: Any = None, **_: Any) -> Any:
            self.last_delete_raw = raw_request
            self.last_framework_ctx = framework_ctx
            self.last_op_ctx = op_ctx
            return {"deleted": len(raw_request.get("ids") or [])}

        async def arun_delete(self, raw_request: Any, *, op_ctx: Any = None, framework_ctx: Any = None, **_: Any) -> Any:
            self.last_delete_raw = raw_request
            self.last_framework_ctx = framework_ctx
            self.last_op_ctx = op_ctx
            return {"deleted": len(raw_request.get("ids") or [])}

        def capabilities(self) -> VectorCapabilities:
            # Provide realistic capability defaults used in validation logic.
            return VectorCapabilities(
                server="test",
                version="1.0",
                supports_metadata_filtering=True,
                supports_namespaces=True,
                max_batch_size=10_000,
                max_top_k=100,
            )

        async def arun_capabilities(self) -> VectorCapabilities:
            return VectorCapabilities(
                server="test",
                version="1.0",
                supports_metadata_filtering=True,
                supports_namespaces=True,
                max_batch_size=10_000,
                max_top_k=100,
            )

    return DummyTranslator()


def _reload_adapter_module_with_import_block(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """
    Reload the adapter module while forcing Semantic Kernel imports to fail.

    This is intentionally robust:
    - If semantic_kernel *is* installed, we simulate it being unavailable.
    - If semantic_kernel is not installed, behavior remains consistent.

    This allows testing the fallback decorator/exception path without skips.
    """
    original_import = __import__

    def blocked_import(name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0) -> Any:
        if name.startswith("semantic_kernel"):
            raise ImportError("blocked semantic_kernel import for test")
        return original_import(name, globals, locals, fromlist, level)

    # Ensure we force the guarded import path to execute again by reloading.
    monkeypatch.setattr("builtins.__import__", blocked_import)

    # Remove cached semantic_kernel modules so the import guard is exercised.
    for k in list(sys.modules.keys()):
        if k.startswith("semantic_kernel"):
            sys.modules.pop(k, None)

    # Reload the module under test.
    return importlib.reload(sk_adapter_module)


# ---------------------------------------------------------------------------
# E2E Helpers (real adapter + deterministic embeddings)
# ---------------------------------------------------------------------------


def _e2e_embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Deterministic, lightweight embedding function used for E2E tests.

    This produces a fixed 8-dimensional vector based on character codes and
    then L2-normalizes it. It is intentionally *not* semantically meaningful,
    but it is deterministic, cheap, and stable across environments.
    """
    out: List[List[float]] = []
    for t in texts:
        buckets = [0.0] * 8
        if not t:
            out.append(buckets)
            continue
        for i, ch in enumerate(t):
            buckets[i % 8] += (ord(ch) % 97) / 97.0
        norm = math.sqrt(sum(x * x for x in buckets)) or 1.0
        out.append([x / norm for x in buckets])
    return out


async def _e2e_aembed_texts(texts: List[str]) -> List[List[float]]:
    """
    Async wrapper embedding function used for E2E tests.

    This keeps the async execution path real (and avoids relying on any internal
    threading fallback behavior in the store).
    """
    # Yield control to ensure we genuinely exercise async paths.
    await asyncio.sleep(0)
    return _e2e_embed_texts(texts)


def _e2e_dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def _e2e_l2norm(a: Sequence[float]) -> float:
    return float(math.sqrt(sum(x * x for x in a)))


def _e2e_match_filter(metadata: Mapping[str, Any], filt: Optional[Mapping[str, Any]]) -> bool:
    """
    Minimal metadata filter matcher sufficient for E2E tests.

    Supported patterns:
      - {"field": "value"}
      - {"field": {"$in": [..]}}
      - {"$and": [<filters>...]}
      - {"$or": [<filters>...]}
    """
    if not filt:
        return True

    if "$and" in filt:
        items = filt.get("$and") or []
        return all(_e2e_match_filter(metadata, f) for f in items if isinstance(f, Mapping))

    if "$or" in filt:
        items = filt.get("$or") or []
        return any(_e2e_match_filter(metadata, f) for f in items if isinstance(f, Mapping))

    for k, v in filt.items():
        if isinstance(v, Mapping):
            if "$in" in v:
                allowed = v.get("$in") or []
                if metadata.get(k) not in allowed:
                    return False
            else:
                # Unknown operator: treat as non-match for safety.
                return False
        else:
            if metadata.get(k) != v:
                return False

    return True


@dataclass
class _E2EStoredVector:
    """
    Internal representation of stored vectors for the in-memory adapter.

    The adapter retains:
      - vector values
      - metadata
      - namespace
    """
    vector: List[float]
    metadata: Dict[str, Any]
    namespace: Optional[str]


@dataclass
class E2EInMemoryVectorAdapter:
    """
    In-memory VectorProtocolV1-style adapter for E2E tests.

    Purpose:
      - Provide a deterministic, dependency-free backend that the real
        VectorTranslator can call into without patching/mocking.

    Interface coverage:
      - upsert / arun_upsert
      - query / arun_query
      - delete / arun_delete
      - capabilities / arun_capabilities
      - health / ahealth
      - query_stream (optional; used when translator supports streaming)
    """
    _db: Dict[str, _E2EStoredVector] = field(default_factory=dict)

    _caps: VectorCapabilities = field(
        default_factory=lambda: VectorCapabilities(
            server="in-memory",
            version="1.0",
            supports_metadata_filtering=True,
            supports_namespaces=True,
            max_batch_size=10_000,
            max_top_k=1_000,
        )
    )

    async def capabilities(self) -> VectorCapabilities:
        return self._caps

    async def arun_capabilities(self) -> VectorCapabilities:
        # Legacy alias for backwards compatibility
        return await self.capabilities()

    async def health(self) -> Mapping[str, Any]:
        return {"status": "ok"}

    async def ahealth(self) -> Mapping[str, Any]:
        # Legacy alias for backwards compatibility
        return await self.health()

    def close(self) -> None:
        # Nothing to close for in-memory storage.
        return

    async def aclose(self) -> None:
        return

    async def upsert(self, spec: UpsertSpec, **_: Any) -> UpsertResult:
        """
        Upsert vectors into the in-memory store.

        Accepts a UpsertSpec with vectors and namespace.
        """
        await asyncio.sleep(0)  # Yield control to test async path
        namespace = spec.namespace
        vectors = spec.vectors or []

        upserted_ids: List[str] = []
        failures: List[Dict[str, Any]] = []

        for v in vectors:
            try:
                vid = str(v.id)
                ns = namespace if namespace is not None else getattr(v, "namespace", None)
                self._db[vid] = _E2EStoredVector(
                    vector=list(v.vector or []),
                    metadata=dict(v.metadata or {}),
                    namespace=ns,
                )
                upserted_ids.append(vid)
            except Exception as exc:  # noqa: BLE001
                failures.append({"id": getattr(v, "id", "unknown"), "error": str(exc)})

        return UpsertResult(
            upserted_count=len(upserted_ids),
            failed_count=len(failures),
            failures=failures,
        )

    async def arun_upsert(self, spec: UpsertSpec, **kwargs: Any) -> UpsertResult:
        # Legacy alias for backwards compatibility
        return await self.upsert(spec, **kwargs)

    async def query(self, spec: QuerySpec, **_: Any) -> QueryResult:
        """
        Cosine similarity query across stored vectors in the namespace.

        Accepts a QuerySpec with vector, top_k, namespace, filters, etc.
        """
        await asyncio.sleep(0)  # Yield control to test async path
        qv = list(spec.vector or [])
        namespace = spec.namespace
        top_k = int(spec.top_k or 4)
        filters = spec.filter
        include_metadata = bool(spec.include_metadata)
        include_vectors = bool(spec.include_vectors)

        qn = _e2e_l2norm(qv) or 1.0
        qv = [x / qn for x in qv]

        matches: List[VectorMatch] = []
        for vid, sv in self._db.items():
            if namespace is not None and sv.namespace != namespace:
                continue
            if not _e2e_match_filter(sv.metadata, filters):
                continue

            dv = list(sv.vector)
            dn = _e2e_l2norm(dv) or 1.0
            dv = [x / dn for x in dv]
            score = _e2e_dot(qv, dv)

            match_vec = Vector(
                id=vid,
                vector=(list(sv.vector) if include_vectors else []),
                metadata=(dict(sv.metadata) if include_metadata else {}),
                namespace=sv.namespace,
                text=None,
            )
            distance = 1.0 - score  # Convert similarity score to distance
            matches.append(VectorMatch(vector=match_vec, score=float(score), distance=distance))

        matches.sort(key=lambda m: float(m.score), reverse=True)
        return QueryResult(
            matches=matches[:top_k], 
            namespace=namespace or "default",
            query_vector=qv,
            total_matches=len(matches)
        )

    async def arun_query(self, spec: QuerySpec, **kwargs: Any) -> QueryResult:
        # Legacy alias for backwards compatibility
        return await self.query(spec, **kwargs)

    def delete(self, raw_request: Mapping[str, Any], **_: Any) -> Mapping[str, Any]:
        """
        Delete by ids and optional namespace.

        The real VectorTranslator typically sends raw_request that includes:
          - ids: List[str]
          - namespace: str (optional)
          - filter: Mapping (optional; not required for these E2E tests)
        """
        namespace = raw_request.get("namespace")
        ids = raw_request.get("ids") or []
        filt = raw_request.get("filter")

        deleted = 0

        # Delete by explicit ids.
        if ids:
            for vid in list(ids):
                sv = self._db.get(str(vid))
                if sv is None:
                    continue
                if namespace is not None and sv.namespace != namespace:
                    continue
                del self._db[str(vid)]
                deleted += 1
            return {"deleted": deleted}

        # Delete by filter (supported for completeness).
        if filt:
            to_delete: List[str] = []
            for vid, sv in self._db.items():
                if namespace is not None and sv.namespace != namespace:
                    continue
                if _e2e_match_filter(sv.metadata, filt):
                    to_delete.append(vid)
            for vid in to_delete:
                del self._db[vid]
                deleted += 1
            return {"deleted": deleted}

        return {"deleted": 0}

    async def arun_delete(self, raw_request: Mapping[str, Any], **kwargs: Any) -> Mapping[str, Any]:
        await asyncio.sleep(0)
        return self.delete(raw_request, **kwargs)

    def query_stream(self, raw_query: Mapping[str, Any], **_: Any) -> Iterator[QueryResult]:
        """
        Optional streaming query support.

        If the VectorTranslator uses adapter-level streaming, this method
        enables store.similarity_search_stream() to exercise a real streaming path.
        """
        result = self.query(raw_query)
        yield QueryResult(
            matches=result.matches or [], 
            namespace=result.namespace,
            query_vector=result.query_vector,
            total_matches=result.total_matches
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def adapter() -> Any:
    """Create a minimal test adapter implementing the VectorProtocolV1 shape needed by VectorTranslator."""
    class TestAdapter:
        def capabilities(self) -> VectorCapabilities:
            return VectorCapabilities(
                supports_metadata_filtering=True,
                supports_namespaces=True,
                max_top_k=100,
            )
    return TestAdapter()


@pytest.fixture
def store(adapter: Any) -> CorpusSemanticKernelVectorStore:
    """Default store fixture."""
    return CorpusSemanticKernelVectorStore(corpus_adapter=adapter)


# ---------------------------------------------------------------------------
# Construction / Initialization Tests
# ---------------------------------------------------------------------------


def test_store_init_requires_corpus_adapter() -> None:
    """Adapter must be provided."""
    with pytest.raises(TypeError):
        # corpus_adapter is keyword-only and required.
        CorpusSemanticKernelVectorStore()  # type: ignore[misc]


def test_store_init_stores_config_attributes(adapter: Any) -> None:
    """Store should keep key config attributes accessible."""
    s = CorpusSemanticKernelVectorStore(
        corpus_adapter=adapter,
        namespace="docs",
        batch_size=50,
        default_top_k=10,
        score_threshold=0.8,
        id_field="custom_id",
        text_field="custom_text",
        metadata_field="custom_metadata",
    )
    assert s.namespace == "docs"
    assert s.batch_size == 50
    assert s.default_top_k == 10
    assert s.score_threshold == 0.8
    assert s.id_field == "custom_id"
    assert s.text_field == "custom_text"
    assert s.metadata_field == "custom_metadata"


def test_store_init_validates_batch_size(adapter: Any) -> None:
    """batch_size must be positive."""
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        CorpusSemanticKernelVectorStore(corpus_adapter=adapter, batch_size=0)


def test_store_init_validates_default_top_k(adapter: Any) -> None:
    """default_top_k must be positive."""
    with pytest.raises(ValueError, match="default_top_k must be at least 1"):
        CorpusSemanticKernelVectorStore(corpus_adapter=adapter, default_top_k=0)


def test_store_init_validates_score_threshold_range(adapter: Any) -> None:
    """score_threshold must be between 0.0 and 1.0."""
    with pytest.raises(ValueError, match="score_threshold must be between"):
        CorpusSemanticKernelVectorStore(corpus_adapter=adapter, score_threshold=1.5)

    with pytest.raises(ValueError, match="score_threshold must be between"):
        CorpusSemanticKernelVectorStore(corpus_adapter=adapter, score_threshold=-0.1)


def test_store_init_validates_reserved_fields_unique(adapter: Any) -> None:
    """Reserved metadata fields must be unique."""
    with pytest.raises(ValueError, match="Reserved metadata fields must be unique"):
        CorpusSemanticKernelVectorStore(
            corpus_adapter=adapter,
            id_field="same",
            text_field="same",  # Duplicate
        )


def test_store_init_validates_reserved_fields_unique_with_metadata_field(adapter: Any) -> None:
    """Reserved fields must remain unique when metadata envelope is enabled."""
    with pytest.raises(ValueError, match="Reserved metadata fields must be unique"):
        CorpusSemanticKernelVectorStore(
            corpus_adapter=adapter,
            id_field="id",
            text_field="text",
            metadata_field="id",  # Duplicate with id_field
        )


# ---------------------------------------------------------------------------
# Translator Wiring Tests
# ---------------------------------------------------------------------------


def test_translator_created_with_framework_semantic_kernel(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Translator factory should be called with framework='semantic_kernel'."""
    captured: Dict[str, Any] = {}

    class FakeTranslator:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def capabilities(self) -> VectorCapabilities:
            return VectorCapabilities(server="test", version="1.0")

    with patch.object(sk_adapter_module, "VectorTranslator", FakeTranslator):
        s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter)
        _ = s._translator  # force creation

    assert captured.get("framework") == "semantic_kernel"
    assert captured.get("adapter") is adapter


def test_translator_uses_default_framework_translator(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Should use SemanticKernelVectorFrameworkTranslator."""
    captured: Dict[str, Any] = {}

    class FakeTranslator:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def capabilities(self) -> VectorCapabilities:
            return VectorCapabilities(server="test", version="1.0")

    with patch.object(sk_adapter_module, "VectorTranslator", FakeTranslator):
        s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter)
        _ = s._translator

    translator_obj = captured.get("translator")
    assert translator_obj is not None
    assert translator_obj.__class__.__name__ == "SemanticKernelVectorFrameworkTranslator"


def test_translator_cached_property_reused(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Multiple accesses to _translator should return same instance."""
    dummy = _make_dummy_translator()
    with patch.object(sk_adapter_module, "VectorTranslator", return_value=dummy):
        s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter)
        t1 = s._translator
        t2 = s._translator
        assert t1 is t2


# ---------------------------------------------------------------------------
# Capability Tests
# ---------------------------------------------------------------------------


def test_get_capabilities_sync_delegates_and_caches(adapter: Any) -> None:
    """Should delegate to translator.capabilities() and cache the result."""
    dummy = _make_dummy_translator()
    with patch.object(sk_adapter_module, "VectorTranslator", return_value=dummy):
        s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter)
        caps1 = s.get_capabilities()
        caps2 = s.get_capabilities()
        assert isinstance(caps1, VectorCapabilities)
        assert caps1 is caps2  # cached object identity


@pytest.mark.asyncio
async def test_get_capabilities_async_delegates_and_caches(adapter: Any) -> None:
    """Should delegate to translator.arun_capabilities() and cache the result."""
    dummy = _make_dummy_translator()
    with patch.object(sk_adapter_module, "VectorTranslator", return_value=dummy):
        s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter)
        caps1 = await s.aget_capabilities()
        caps2 = await s.aget_capabilities()
        assert isinstance(caps1, VectorCapabilities)
        assert caps1 is caps2


def test_caps_error_attaches_context_sync(adapter: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Errors during sync capability fetch should attach error context."""
    captured: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured.update(ctx)

    class FailingTranslator:
        def capabilities(self) -> VectorCapabilities:
            raise RuntimeError("capabilities failed")

    monkeypatch.setattr(sk_adapter_module, "attach_context", fake_attach_context)
    with patch.object(sk_adapter_module, "VectorTranslator", return_value=FailingTranslator()):
        s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter)
        with pytest.raises(RuntimeError, match="capabilities failed"):
            _ = s.get_capabilities()

    assert captured.get("framework") == "semantic_kernel"
    assert captured.get("operation") == "capabilities_sync"


@pytest.mark.asyncio
async def test_caps_error_attaches_context_async(adapter: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Errors during async capability fetch should attach error context."""
    captured: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured.update(ctx)

    class FailingTranslator:
        async def arun_capabilities(self) -> VectorCapabilities:
            raise RuntimeError("acapabilities failed")

    monkeypatch.setattr(sk_adapter_module, "attach_context", fake_attach_context)
    with patch.object(sk_adapter_module, "VectorTranslator", return_value=FailingTranslator()):
        s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter)
        with pytest.raises(RuntimeError, match="acapabilities failed"):
            _ = await s.aget_capabilities()

    assert captured.get("framework") == "semantic_kernel"
    assert captured.get("operation") == "capabilities_async"


# ---------------------------------------------------------------------------
# Embedding Resolution Tests
# ---------------------------------------------------------------------------


def test_ensure_embeddings_uses_provided_embeddings_when_lengths_match(adapter: Any) -> None:
    """Provided embeddings should be accepted when length matches."""
    s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter)
    out = s._ensure_embeddings(["a", "b"], embeddings=[[0.1], [0.2]])
    assert len(out) == 2


def test_ensure_embeddings_raises_when_embeddings_length_mismatch(adapter: Any) -> None:
    """Mismatched embeddings count should raise BadRequest with strong error context."""
    s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter)
    with pytest.raises(BadRequest, match="embeddings length"):
        _ = s._ensure_embeddings(["a", "b"], embeddings=[[0.1]])


def test_ensure_embeddings_raises_when_no_embedding_function_and_no_embeddings(adapter: Any) -> None:
    """If no embedding function is configured, caller must supply embeddings."""
    s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter, embedding_function=None)
    with pytest.raises(NotSupported, match="No embedding_function configured"):
        _ = s._ensure_embeddings(["a"], embeddings=None)


def test_ensure_embeddings_calls_embedding_function_when_needed(adapter: Any) -> None:
    """Embedding function should be invoked when embeddings are not provided."""
    called: Dict[str, Any] = {"count": 0}

    def embed(texts: List[str]) -> Sequence[Sequence[float]]:
        called["count"] += 1
        return [[0.1] for _ in texts]

    s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter, embedding_function=embed)
    out = s._ensure_embeddings(["a", "b"], embeddings=None)
    assert called["count"] == 1
    assert len(out) == 2


def test_ensure_embeddings_wraps_embedding_function_exception_as_BadRequest(adapter: Any) -> None:
    """Embedding errors should be normalized into BadRequest(code=EMBEDDING_ERROR)."""
    def embed(_: List[str]) -> Sequence[Sequence[float]]:
        raise RuntimeError("boom")

    s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter, embedding_function=embed)
    with pytest.raises(BadRequest, match="embedding_function failed"):
        _ = s._ensure_embeddings(["a"], embeddings=None)


def test_ensure_embeddings_raises_when_embedding_function_returns_wrong_count(adapter: Any) -> None:
    """Embedding function must return the same number of embeddings as input texts."""
    def embed(_: List[str]) -> Sequence[Sequence[float]]:
        return [[0.1], [0.2]]  # wrong count for single input

    s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter, embedding_function=embed)
    with pytest.raises(BadRequest, match="returned 2 embeddings"):
        _ = s._ensure_embeddings(["a"], embeddings=None)


@pytest.mark.asyncio
async def test_ensure_embeddings_async_prefers_async_embedding_function(adapter: Any) -> None:
    """Async embedding function should take precedence over sync embedding function."""
    called: Dict[str, int] = {"async": 0, "sync": 0}

    async def aembed(texts: List[str]) -> Sequence[Sequence[float]]:
        called["async"] += 1
        return [[0.1] for _ in texts]

    def embed(texts: List[str]) -> Sequence[Sequence[float]]:
        called["sync"] += 1
        return [[0.2] for _ in texts]

    s = CorpusSemanticKernelVectorStore(
        corpus_adapter=adapter,
        embedding_function=embed,
        async_embedding_function=aembed,
    )
    out = await s._ensure_embeddings_async(["a", "b"], embeddings=None)
    assert len(out) == 2
    assert called["async"] == 1
    assert called["sync"] == 0


@pytest.mark.asyncio
async def test_ensure_embeddings_async_falls_back_to_threaded_sync_embedding_function(adapter: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """When no async embedding function exists, the sync one is executed via asyncio.to_thread."""
    called: Dict[str, Any] = {"to_thread": 0}

    def embed(texts: List[str]) -> Sequence[Sequence[float]]:
        return [[0.1] for _ in texts]

    async def fake_to_thread(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        called["to_thread"] += 1
        return fn(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter, embedding_function=embed)
    out = await s._ensure_embeddings_async(["a"], embeddings=None)
    assert len(out) == 1
    assert called["to_thread"] == 1


@pytest.mark.asyncio
async def test_ensure_embeddings_async_raises_when_none_configured_and_no_embeddings(adapter: Any) -> None:
    """If no embedding functions exist, caller must provide embeddings."""
    s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter, embedding_function=None, async_embedding_function=None)
    with pytest.raises(NotSupported, match="No embedding_function or async_embedding_function configured"):
        _ = await s._ensure_embeddings_async(["a"], embeddings=None)


# ---------------------------------------------------------------------------
# Metadata & ID Normalization Tests
# ---------------------------------------------------------------------------


def test_normalize_metadatas_none_creates_empty_dicts(adapter: Any) -> None:
    """None metadata list should be normalized to empty dicts."""
    s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter)
    out = s._normalize_metadatas(3, metadatas=None)
    assert out == [{}, {}, {}]


def test_normalize_metadatas_singleton_replicates(adapter: Any) -> None:
    """Singleton metadata should replicate across all texts."""
    s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter)
    out = s._normalize_metadatas(3, metadatas=[{"a": 1}])
    assert out == [{"a": 1}, {"a": 1}, {"a": 1}]


def test_normalize_metadatas_length_mismatch_raises_BadRequest(adapter: Any) -> None:
    """Mismatched metadata length should raise a structured BadRequest."""
    s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter)
    with pytest.raises(BadRequest, match="metadatas length"):
        _ = s._normalize_metadatas(2, metadatas=[{"a": 1}, {"b": 2}, {"c": 3}])


def test_normalize_ids_none_generates_uuid_hex(adapter: Any) -> None:
    """When ids are None, UUID4 hex strings should be generated."""
    s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter)
    out = s._normalize_ids(5, ids=None)
    assert len(out) == 5
    assert len(set(out)) == 5  # uniqueness is a safety property for writes


def test_normalize_ids_length_mismatch_raises_BadRequest(adapter: Any) -> None:
    """Mismatched ids length should raise a structured BadRequest."""
    s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter)
    with pytest.raises(BadRequest, match="ids length"):
        _ = s._normalize_ids(2, ids=["a"])


# ---------------------------------------------------------------------------
# Vector Translation Helper Tests
# ---------------------------------------------------------------------------


def test_to_corpus_vectors_without_metadata_envelope_sets_text_and_id_in_metadata(adapter: Any) -> None:
    """Without metadata_field, user metadata remains flat but reserved fields are injected."""
    s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter, metadata_field=None, text_field="page_content", id_field="id")
    vectors = s._to_corpus_vectors(
        texts=["hello"],
        embeddings=[[0.1, 0.2]],
        metadatas=[{"custom": "x"}],
        ids=["doc-1"],
        namespace="docs",
    )
    assert vectors[0].metadata["custom"] == "x"
    assert vectors[0].metadata["page_content"] == "hello"
    assert vectors[0].metadata["id"] == "doc-1"
    assert vectors[0].namespace == "docs"


def test_to_corpus_vectors_with_metadata_envelope_nests_user_metadata(adapter: Any) -> None:
    """With metadata_field, user metadata is nested for safer reserved-field separation."""
    s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter, metadata_field="user_metadata")
    vectors = s._to_corpus_vectors(
        texts=["hello"],
        embeddings=[[0.1, 0.2]],
        metadatas=[{"custom": "x"}],
        ids=["doc-1"],
        namespace=None,
    )
    assert vectors[0].metadata["user_metadata"] == {"custom": "x"}
    assert vectors[0].metadata[s.text_field] == "hello"
    assert vectors[0].metadata[s.id_field] == "doc-1"


def test_from_corpus_matches_strips_reserved_fields_without_envelope(adapter: Any) -> None:
    """Match conversion should drop text/id reserved fields from metadata returned to callers."""
    s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter, metadata_field=None, text_field="page_content", id_field="id")
    matches = [
        _make_vector_match(
            vid="doc-1",
            score=0.9,
            text_field="page_content",
            id_field="id",
            text="hello",
            embedding=[0.1, 0.2],
            metadata={"custom": "x"},
        )
    ]
    out = s._from_corpus_matches(matches)
    assert out[0][0] == "hello"
    assert out[0][1] == {"custom": "x"}  # reserved fields removed


def test_from_corpus_matches_unwraps_envelope(adapter: Any) -> None:
    """When metadata_field is set, returned metadata should be unwrapped to user payload."""
    s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter, metadata_field="user_metadata")
    match = _make_vector_match(
        vid="doc-1",
        score=0.9,
        text_field=s.text_field,
        id_field=s.id_field,
        text="hello",
        embedding=[0.1, 0.2],
        metadata={"user_metadata": {"custom": "x"}},
    )
    out = s._from_corpus_matches([match])
    assert out[0][1] == {"custom": "x"}


def test_from_corpus_matches_missing_text_returns_empty_string(adapter: Any) -> None:
    """Missing text field should yield empty string rather than raising."""
    s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter)
    vec = Vector(id="doc-1", vector=[0.1], metadata={"id": "doc-1"}, namespace="default", text=None)
    match = VectorMatch(vector=vec, score=0.9, distance=0.1)
    out = s._from_corpus_matches([match])
    assert out[0][0] == ""


# ---------------------------------------------------------------------------
# Request Builder Tests
# ---------------------------------------------------------------------------


def test_build_upsert_request_includes_namespace_and_vectors(adapter: Any) -> None:
    """Upsert request must be a list of document dicts with id, vector, metadata, namespace."""
    s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter, namespace="default")
    vecs = [Vector(id="v1", vector=[0.1], metadata={"key": "val"}, namespace="default", text=None)]
    raw_documents, framework_ctx = s._build_upsert_request(vecs, namespace=None)
    
    assert isinstance(raw_documents, list)
    assert len(raw_documents) == 1
    doc = raw_documents[0]
    assert doc["id"] == "v1"
    assert doc["vector"] == [0.1]
    assert doc["metadata"] == {"key": "val"}
    assert doc["namespace"] == "default"
    assert isinstance(framework_ctx, dict)


def test_build_query_request_includes_all_params(adapter: Any) -> None:
    """Query request must include the expected raw shape and flags."""
    s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter, namespace="ns")
    raw, _ = s._build_query_request(
        embedding=[0.1, 0.2],
        top_k=5,
        namespace=None,
        filter={"a": 1},
        include_vectors=True,
    )
    assert raw["vector"] == [0.1, 0.2]
    assert raw["top_k"] == 5
    assert raw["namespace"] == "ns"
    assert raw["filters"] == {"a": 1}
    assert raw["include_metadata"] is True  # Always true for this adapter layer
    assert raw["include_vectors"] is True


def test_build_delete_request_includes_namespace_ids_or_filter(adapter: Any) -> None:
    """Delete request must include namespace and ids/filter keys."""
    s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter, namespace="ns")
    raw, _ = s._build_delete_request(ids=["a", "b"], namespace=None, filter=None)
    assert raw["namespace"] == "ns"
    assert raw["ids"] == ["a", "b"]
    assert raw["filter"] is None


# ---------------------------------------------------------------------------
# Query Parameter Validation Tests
# ---------------------------------------------------------------------------


def test_validate_query_params_sync_raises_on_exceeded_max_top_k(adapter: Any) -> None:
    """Should raise if top_k exceeds max_top_k."""
    dummy = _make_dummy_translator()
    with patch.object(sk_adapter_module, "VectorTranslator", return_value=dummy):
        s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter)
        with pytest.raises(BadRequest, match="exceeds maximum"):
            _ = s._validate_query_params_sync(200, namespace=None, filter=None)


def test_validate_query_params_sync_raises_on_unsupported_filter(adapter: Any) -> None:
    """Should raise if metadata filtering is not supported."""
    class NoFilterTranslator:
        def capabilities(self) -> VectorCapabilities:
            return VectorCapabilities(server="test", version="1.0", supports_metadata_filtering=False, max_top_k=100)

    with patch.object(sk_adapter_module, "VectorTranslator", return_value=NoFilterTranslator()):
        s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter)
        with pytest.raises(NotSupported, match="metadata filtering"):
            _ = s._validate_query_params_sync(10, namespace=None, filter={"a": 1})


@pytest.mark.asyncio
async def test_validate_query_params_async_raises_on_exceeded_max_top_k(adapter: Any) -> None:
    """Should raise if top_k exceeds max_top_k (async)."""
    dummy = _make_dummy_translator()
    with patch.object(sk_adapter_module, "VectorTranslator", return_value=dummy):
        s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter)
        with pytest.raises(BadRequest, match="exceeds maximum"):
            _ = await s._validate_query_params_async(200, namespace=None, filter=None)


# ---------------------------------------------------------------------------
# Delete Parameter Validation Tests
# ---------------------------------------------------------------------------


def test_validate_delete_params_sync_raises_without_ids_and_filter(adapter: Any) -> None:
    """Delete must include ids or filter to avoid accidental broad deletes."""
    dummy = _make_dummy_translator()
    with patch.object(sk_adapter_module, "VectorTranslator", return_value=dummy):
        s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter)
        with pytest.raises(BadRequest, match="must provide ids or filter"):
            s._validate_delete_params_sync(ids=None, namespace=None, filter=None)


def test_validate_delete_params_sync_raises_when_filter_not_supported(adapter: Any) -> None:
    """Delete by filter must respect capabilities."""
    class NoFilterTranslator:
        def capabilities(self) -> VectorCapabilities:
            return VectorCapabilities(server="test", version="1.0", supports_metadata_filtering=False, max_top_k=100)

    with patch.object(sk_adapter_module, "VectorTranslator", return_value=NoFilterTranslator()):
        s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter)
        with pytest.raises(NotSupported, match="delete by metadata filter"):
            s._validate_delete_params_sync(ids=None, namespace=None, filter={"a": 1})


@pytest.mark.asyncio
async def test_validate_delete_params_async_raises_without_ids_and_filter(adapter: Any) -> None:
    """Async delete validation should enforce ids-or-filter as well."""
    dummy = _make_dummy_translator()
    with patch.object(sk_adapter_module, "VectorTranslator", return_value=dummy):
        s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter)
        with pytest.raises(BadRequest, match="must provide ids or filter"):
            await s._validate_delete_params_async(ids=None, namespace=None, filter=None)


# ---------------------------------------------------------------------------
# Add / Upsert Tests
# ---------------------------------------------------------------------------


def test_add_texts_empty_input_returns_empty_list(adapter: Any) -> None:
    """Empty input should not hit translator and should return empty list."""
    s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter)
    assert s.add_texts([]) == []


def test_add_texts_calls_translator_upsert_and_returns_ids(adapter: Any) -> None:
    """add_texts should upsert vectors and return ids (caller-visible)."""
    dummy = _make_dummy_translator()
    with patch.object(sk_adapter_module, "VectorTranslator", return_value=dummy):
        s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter, embedding_function=lambda t: [[0.1] for _ in t])
        ids = s.add_texts(["a", "b"], metadatas=[{"m": 1}, {"m": 2}], ids=["id1", "id2"], namespace="ns")
        assert ids == ["id1", "id2"]
        assert dummy.last_upsert_raw is not None
        # raw_upsert_request is now a list of document dicts
        assert isinstance(dummy.last_upsert_raw, list)
        assert len(dummy.last_upsert_raw) == 2
        assert all(doc.get("namespace") == "ns" for doc in dummy.last_upsert_raw)


def test_add_texts_passes_ctx_and_framework_ctx_to_translator(adapter: Any) -> None:
    """The adapter should pass op_ctx and framework_ctx through to VectorTranslator."""
    dummy = _make_dummy_translator()
    ctx = OperationContext(request_id="r1", tenant="t1", attrs={})

    with patch.object(sk_adapter_module, "VectorTranslator", return_value=dummy):
        s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter, embedding_function=lambda t: [[0.1] for _ in t])
        _ = s.add_texts(["a"], ids=["id1"], ctx=ctx, namespace="ns")

        assert dummy.last_op_ctx is ctx
        assert isinstance(dummy.last_framework_ctx, dict)


def test_add_texts_handles_partial_failure_logs_but_returns_ids(adapter: Any, caplog: pytest.LogCaptureFixture) -> None:
    """Partial failures should log warnings but still return ids."""
    class PartialFailureTranslator:
        def upsert(self, raw_request: Any, **_: Any) -> Any:
            return UpsertResult(
                upserted_count=1,
                failed_count=1,
                failures=[{"id": "id2", "error": "boom"}],
            )

        def capabilities(self) -> VectorCapabilities:
            return VectorCapabilities(server="test", version="1.0", supports_metadata_filtering=True, max_top_k=100)

    with patch.object(sk_adapter_module, "VectorTranslator", return_value=PartialFailureTranslator()):
        s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter, embedding_function=lambda t: [[0.1] for _ in t])
        with caplog.at_level(logging.WARNING):
            ids = s.add_texts(["a", "b"], ids=["id1", "id2"], metadatas=[{}, {}], namespace="ns")
            assert ids == ["id1", "id2"]
            assert any("Partial upsert failure" in r.message for r in caplog.records)


def test_add_texts_raises_when_all_upserts_failed(adapter: Any) -> None:
    """If nothing succeeds, the adapter raises VectorAdapterError(code=BATCH_UPSERT_FAILED)."""
    class AllFailureTranslator:
        def upsert(self, raw_request: Any, **_: Any) -> Any:
            # raw_request is now a list of document dicts
            count = len(raw_request) if isinstance(raw_request, list) else len(raw_request.get("vectors") or [])
            return UpsertResult(
                upserted_count=0,
                failed_count=count,
                failures=[{"id": "id1", "error": "boom"}],
            )

        def capabilities(self) -> VectorCapabilities:
            return VectorCapabilities(server="test", version="1.0", supports_metadata_filtering=True, max_top_k=100)

    with patch.object(sk_adapter_module, "VectorTranslator", return_value=AllFailureTranslator()):
        s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter, embedding_function=lambda t: [[0.1] for _ in t])
        with pytest.raises(VectorAdapterError, match="All 2 texts failed to upsert"):
            _ = s.add_texts(["a", "b"], ids=["id1", "id2"], metadatas=[{}, {}], namespace="ns")


@pytest.mark.asyncio
async def test_aadd_texts_calls_translator_arun_upsert_and_returns_ids(adapter: Any) -> None:
    """Async add should use arun_upsert and return ids."""
    dummy = _make_dummy_translator()
    with patch.object(sk_adapter_module, "VectorTranslator", return_value=dummy):
        s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter, async_embedding_function=lambda t: asyncio.sleep(0, result=[[0.1] for _ in t]))
        ids = await s.aadd_texts(["a"], ids=["id1"], metadatas=[{}], namespace="ns")
        assert ids == ["id1"]
        assert dummy.last_upsert_raw is not None
        # raw_upsert_request is now a list of document dicts
        assert isinstance(dummy.last_upsert_raw, list)
        assert all(doc.get("namespace") == "ns" for doc in dummy.last_upsert_raw)


def test_add_documents_extracts_page_content_and_metadata(adapter: Any) -> None:
    """add_documents should extract 'page_content' and 'metadata' keys and delegate."""
    dummy = _make_dummy_translator()
    with patch.object(sk_adapter_module, "VectorTranslator", return_value=dummy):
        s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter, embedding_function=lambda t: [[0.1] for _ in t])
        _ = s.add_documents([{"page_content": "hello", "metadata": {"a": 1}}], namespace="ns")
        assert dummy.last_upsert_raw is not None
        # raw_upsert_request is now a list of document dicts
        assert isinstance(dummy.last_upsert_raw, list)
        assert len(dummy.last_upsert_raw) == 1
        doc = dummy.last_upsert_raw[0]
        assert doc.get("metadata", {}).get("page_content") == "hello"
        assert doc.get("metadata", {}).get("a") == 1


@pytest.mark.asyncio
async def test_aadd_documents_extracts_page_content_and_metadata(adapter: Any) -> None:
    """aadd_documents should extract and delegate to async path."""
    dummy = _make_dummy_translator()
    with patch.object(sk_adapter_module, "VectorTranslator", return_value=dummy):
        s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter, async_embedding_function=lambda t: asyncio.sleep(0, result=[[0.1] for _ in t]))
        _ = await s.aadd_documents([{"page_content": "hello", "metadata": {"a": 1}}], namespace="ns")
        assert dummy.last_upsert_raw is not None


# ---------------------------------------------------------------------------
# Similarity Search Tests
# ---------------------------------------------------------------------------


def test_similarity_search_calls_translator_query_and_formats_results(adapter: Any) -> None:
    """similarity_search returns AI-optimized dict output."""
    dummy = _make_dummy_translator()
    with patch.object(sk_adapter_module, "VectorTranslator", return_value=dummy):
        s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter, embedding_function=lambda t: [[0.1] for _ in t])
        docs = s.similarity_search("q", k=1)
        assert isinstance(docs, list)
        assert docs and set(docs[0].keys()) == {"content", "metadata", "confidence", "source"}
        assert docs[0]["source"] == "vector_database"


def test_similarity_search_applies_score_threshold(adapter: Any) -> None:
    """Score threshold should filter low-confidence results."""
    dummy = _make_dummy_translator(include_low_score=True)
    with patch.object(sk_adapter_module, "VectorTranslator", return_value=dummy):
        s = CorpusSemanticKernelVectorStore(
            corpus_adapter=adapter,
            embedding_function=lambda t: [[0.1] for _ in t],
            score_threshold=0.5,
        )
        docs = s.similarity_search("q", k=10)
        assert len(docs) == 1
        assert docs[0]["content"] != "low"


def test_similarity_search_truncates_long_content(adapter: Any) -> None:
    """Long content should be truncated for token efficiency."""
    dummy = _make_dummy_translator(long_text=True)
    with patch.object(sk_adapter_module, "VectorTranslator", return_value=dummy):
        s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter, embedding_function=lambda t: [[0.1] for _ in t])
        docs = s.similarity_search("q", k=1)
        assert len(docs[0]["content"]) <= 503  # 500 + "..." (approximate safety)
        assert docs[0]["content"].endswith("...")


def test_similarity_search_filters_internal_metadata_fields(adapter: Any) -> None:
    """Internal/system fields should be removed from AI-facing metadata."""
    dummy = _make_dummy_translator(include_internal_fields=True)
    with patch.object(sk_adapter_module, "VectorTranslator", return_value=dummy):
        s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter, embedding_function=lambda t: [[0.1] for _ in t])
        docs = s.similarity_search("q", k=1)
        md = docs[0]["metadata"]
        assert "id" not in md
        assert "_id" not in md
        assert "vector" not in md
        assert "_vector" not in md
        assert "embedding" not in md
        assert "timestamp" not in md
        assert md.get("custom") == "data"


def test_similarity_search_raises_on_non_QueryResult_translator_return(adapter: Any) -> None:
    """Translator must return QueryResult; otherwise the adapter raises BadRequest."""
    class BadTranslator:
        def query(self, *_: Any, **__: Any) -> Any:
            return {"not": "a QueryResult"}

        def capabilities(self) -> VectorCapabilities:
            return VectorCapabilities(server="test", version="1.0", supports_metadata_filtering=True, max_top_k=100)

    with patch.object(sk_adapter_module, "VectorTranslator", return_value=BadTranslator()):
        s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter, embedding_function=lambda t: [[0.1] for _ in t])
        with pytest.raises(BadRequest, match="returned unsupported type"):
            _ = s.similarity_search("q", k=1)


@pytest.mark.asyncio
async def test_asimilarity_search_calls_translator_arun_query_and_formats_results(adapter: Any) -> None:
    """Async similarity search returns AI-optimized dict output."""
    dummy = _make_dummy_translator()
    with patch.object(sk_adapter_module, "VectorTranslator", return_value=dummy):
        s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter, async_embedding_function=lambda t: asyncio.sleep(0, result=[[0.1] for _ in t]))
        docs = await s.asimilarity_search("q", k=1)
        assert docs and "confidence" in docs[0]


# ---------------------------------------------------------------------------
# Streaming Similarity Search Tests
# ---------------------------------------------------------------------------


def test_similarity_search_stream_yields_items_one_by_one(adapter: Any) -> None:
    """Streaming search yields docs incrementally."""
    dummy = _make_dummy_translator()
    with patch.object(sk_adapter_module, "VectorTranslator", return_value=dummy):
        s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter, embedding_function=lambda t: [[0.1] for _ in t])
        out = list(s.similarity_search_stream("q", k=2))
        assert out
        assert all("content" in d and "confidence" in d for d in out)


def test_similarity_search_stream_raises_on_non_QueryResult(adapter: Any) -> None:
    """Streaming must yield QueryResult; otherwise BAD_STREAM_CHUNK is raised."""
    class BadStreamTranslator:
        def query_stream(self, *_: Any, **__: Any) -> Iterator[Any]:
            yield {"not": "a chunk"}

        def capabilities(self) -> VectorCapabilities:
            return VectorCapabilities(server="test", version="1.0", supports_metadata_filtering=True, max_top_k=100)

    with patch.object(sk_adapter_module, "VectorTranslator", return_value=BadStreamTranslator()):
        s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter, embedding_function=lambda t: [[0.1] for _ in t])
        with pytest.raises(VectorAdapterError) as exc_info:
            _ = list(s.similarity_search_stream("q", k=1))
        
        assert exc_info.value.code == "BAD_STREAM_CHUNK"
        assert "yielded unsupported type" in str(exc_info.value)


# ---------------------------------------------------------------------------
# MMR Search Tests
# ---------------------------------------------------------------------------


def test_mmr_search_raises_on_lambda_out_of_range(adapter: Any) -> None:
    """lambda_mult must be in [0,1] to prevent invalid scoring behavior."""
    s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter, embedding_function=lambda t: [[0.1] for _ in t])
    with pytest.raises(BadRequest, match="lambda_mult must be in"):
        _ = s.max_marginal_relevance_search("q", k=2, lambda_mult=2.0)


def test_mmr_search_requests_include_vectors_true(adapter: Any) -> None:
    """MMR requires vectors to compute diversity; request must set include_vectors=True."""
    dummy = _make_dummy_translator(include_vectors=True)
    with patch.object(sk_adapter_module, "VectorTranslator", return_value=dummy):
        s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter, embedding_function=lambda t: [[0.1, 0.2] for _ in t])
        _ = s.max_marginal_relevance_search("q", k=1, lambda_mult=0.5, fetch_k=5)
        assert dummy.last_query_raw is not None
        assert dummy.last_query_raw.get("include_vectors") is True


def test_mmr_search_lambda_1_returns_pure_relevance_order(adapter: Any) -> None:
    """lambda_mult==1.0 should behave like pure relevance ranking."""
    # Provide two candidates with different scores.
    def custom_translator() -> Any:
        class T:
            def __init__(self) -> None:
                self.last_query_raw: Optional[Dict[str, Any]] = None

            def query(self, raw_query: Any, **_: Any) -> Any:
                self.last_query_raw = raw_query
                matches = [
                    _make_vector_match(
                        vid="a",
                        score=0.2,
                        text_field="page_content",
                        id_field="id",
                        text="low-score",
                        embedding=[1.0, 0.0],
                        metadata={"custom": "a"},
                        namespace="default",
                    ),
                    _make_vector_match(
                        vid="b",
                        score=0.9,
                        text_field="page_content",
                        id_field="id",
                        text="high-score",
                        embedding=[0.0, 1.0],
                        metadata={"custom": "b"},
                        namespace="default",
                    ),
                ]
                return QueryResult(
                    matches=matches, 
                    namespace="default",
                    query_vector=[0.1, 0.2],
                    total_matches=len(matches)
                )

            def capabilities(self) -> VectorCapabilities:
                return VectorCapabilities(server="test", version="1.0", supports_metadata_filtering=True, max_top_k=100)

        return T()

    t = custom_translator()
    with patch.object(sk_adapter_module, "VectorTranslator", return_value=t):
        s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter, embedding_function=lambda t: [[0.1, 0.2] for _ in t])
        docs = s.max_marginal_relevance_search("q", k=1, lambda_mult=1.0, fetch_k=2)
        assert len(docs) == 1
        assert docs[0]["content"] == "high-score"


@pytest.mark.asyncio
async def test_amax_marginal_relevance_search_requests_include_vectors_true(adapter: Any) -> None:
    """Async MMR also must request include_vectors=True."""
    dummy = _make_dummy_translator(include_vectors=True)
    with patch.object(sk_adapter_module, "VectorTranslator", return_value=dummy):
        s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter, async_embedding_function=lambda t: asyncio.sleep(0, result=[[0.1, 0.2] for _ in t]))
        _ = await s.amax_marginal_relevance_search("q", k=1, lambda_mult=0.5, fetch_k=5)
        assert dummy.last_query_raw is not None
        assert dummy.last_query_raw.get("include_vectors") is True


# ---------------------------------------------------------------------------
# Delete API Tests
# ---------------------------------------------------------------------------


def test_delete_calls_translator_delete_with_ids(adapter: Any) -> None:
    """delete() should build a safe delete request and delegate to translator.delete()."""
    dummy = _make_dummy_translator()
    with patch.object(sk_adapter_module, "VectorTranslator", return_value=dummy):
        s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter)
        s.delete(ids=["a", "b"], namespace="ns")
        assert dummy.last_delete_raw is not None
        assert dummy.last_delete_raw["namespace"] == "ns"
        assert dummy.last_delete_raw["ids"] == ["a", "b"]


@pytest.mark.asyncio
async def test_adelete_calls_translator_arun_delete_with_ids(adapter: Any) -> None:
    """adelete() should delegate to translator.arun_delete()."""
    dummy = _make_dummy_translator()
    with patch.object(sk_adapter_module, "VectorTranslator", return_value=dummy):
        s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter)
        await s.adelete(ids=["a"], namespace="ns")
        assert dummy.last_delete_raw is not None
        assert dummy.last_delete_raw["ids"] == ["a"]


def test_delete_raises_when_no_ids_and_no_filter(adapter: Any) -> None:
    """Safety guard: delete must never allow empty criteria."""
    dummy = _make_dummy_translator()
    with patch.object(sk_adapter_module, "VectorTranslator", return_value=dummy):
        s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter)
        with pytest.raises(BadRequest, match="must provide ids or filter"):
            s.delete(ids=None, filter=None)


# ---------------------------------------------------------------------------
# Context Builder Tests
# ---------------------------------------------------------------------------


def test_build_ctx_returns_operation_context_when_provided(store: CorpusSemanticKernelVectorStore) -> None:
    """Explicit OperationContext must pass through unchanged."""
    ctx = OperationContext(request_id="test", tenant="test", attrs={})
    out = store._build_ctx(ctx=ctx)
    assert out is ctx


def test_build_ctx_uses_context_from_dict_when_context_dict_present(adapter: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """context_dict should build an OperationContext using context_from_dict."""
    base = OperationContext(request_id="from-dict", tenant="from-dict", attrs={})

    def fake_from_dict(_: Any) -> OperationContext:
        return base

    monkeypatch.setattr(sk_adapter_module, "context_from_dict", fake_from_dict)
    s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter)
    out = s._build_ctx(context_dict={"a": 1})
    assert isinstance(out, OperationContext)
    assert out is base


def test_build_ctx_returns_none_when_translation_fails(adapter: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Translation failures should be swallowed (debug logged) and return None."""
    def fake_from_dict(_: Any) -> OperationContext:
        raise RuntimeError("boom")

    monkeypatch.setattr(sk_adapter_module, "context_from_dict", fake_from_dict)
    s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter)
    out = s._build_ctx(context_dict={"a": 1})
    assert out is None


# ---------------------------------------------------------------------------
# Semantic Kernel Optionality Tests (No Skips)
# ---------------------------------------------------------------------------


def test_import_without_semantic_kernel_installed_uses_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    If Semantic Kernel is unavailable, the module must still be importable and provide:
    - a no-op `kernel_function` decorator
    - a fallback `KernelFunctionException`
    """
    m = _reload_adapter_module_with_import_block(monkeypatch)

    # The fallback decorator should behave like identity.
    @m.kernel_function
    def f() -> str:
        return "ok"

    assert f() == "ok"
    assert hasattr(m, "KernelFunctionException")


def test_import_with_semantic_kernel_available_uses_real_decorator_or_fallback() -> None:
    """
    This test is intentionally flexible:
    - If semantic_kernel is installed in the environment, we should have imported SK's decorator.
    - If it's not installed, we should be on the fallback path.
    Either outcome is acceptable as long as the module remains functional.
    """
    assert hasattr(sk_adapter_module, "kernel_function")
    assert hasattr(sk_adapter_module, "KernelFunctionException")


# ---------------------------------------------------------------------------
# Plugin Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plugin_vector_search_calls_store_asimilarity_search_and_returns_docs(adapter: Any) -> None:
    """Plugin vector_search should delegate to store.asimilarity_search and return docs."""
    s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter, async_embedding_function=lambda t: asyncio.sleep(0, result=[[0.1] for _ in t]))
    plugin = CorpusSemanticKernelVectorPlugin(vector_store=s)

    dummy = _make_dummy_translator()
    with patch.object(sk_adapter_module, "VectorTranslator", return_value=dummy):
        # Force translator to be created on store (cached_property).
        _ = s._translator

        docs = await plugin.vector_search("q", k=1, namespace="ns")
        assert isinstance(docs, list)
        assert docs and docs[0]["source"] == "vector_database"


def test_plugin_vector_search_stream_yields_docs(adapter: Any) -> None:
    """Plugin vector_search_stream should yield documents from the store streaming API."""
    s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter, embedding_function=lambda t: [[0.1] for _ in t])
    plugin = CorpusSemanticKernelVectorPlugin(vector_store=s)

    dummy = _make_dummy_translator()
    with patch.object(sk_adapter_module, "VectorTranslator", return_value=dummy):
        _ = s._translator
        out = list(plugin.vector_search_stream("q", k=1, namespace="ns"))
        assert out and out[0]["source"] == "vector_database"


@pytest.mark.asyncio
async def test_plugin_vector_mmr_search_calls_store_amax_marginal_relevance_search(adapter: Any) -> None:
    """Plugin vector_mmr_search should call the store async MMR path."""
    s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter, async_embedding_function=lambda t: asyncio.sleep(0, result=[[0.1, 0.2] for _ in t]))
    plugin = CorpusSemanticKernelVectorPlugin(vector_store=s)

    dummy = _make_dummy_translator(include_vectors=True)
    with patch.object(sk_adapter_module, "VectorTranslator", return_value=dummy):
        _ = s._translator
        docs = await plugin.vector_mmr_search("q", k=1, lambda_mult=0.5, namespace="ns")
        assert isinstance(docs, list)


@pytest.mark.asyncio
async def test_plugin_store_document_calls_store_aadd_texts_and_returns_id(adapter: Any) -> None:
    """store_document should return the created/assigned document id."""
    s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter, async_embedding_function=lambda t: asyncio.sleep(0, result=[[0.1] for _ in t]))
    plugin = CorpusSemanticKernelVectorPlugin(vector_store=s)

    dummy = _make_dummy_translator()
    with patch.object(sk_adapter_module, "VectorTranslator", return_value=dummy):
        _ = s._translator
        doc_id = await plugin.store_document("content", metadata={"a": 1}, document_id="doc-1", namespace="ns")
        assert doc_id == "doc-1"


@pytest.mark.asyncio
async def test_plugin_get_capabilities_returns_expected_shape(adapter: Any) -> None:
    """Plugin get_capabilities should return a stable AI-facing dict."""
    s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter)
    plugin = CorpusSemanticKernelVectorPlugin(vector_store=s)

    dummy = _make_dummy_translator()
    with patch.object(sk_adapter_module, "VectorTranslator", return_value=dummy):
        _ = s._translator
        caps = await plugin.get_capabilities()
        assert set(caps.keys()) == {
            "max_batch_size",
            "max_top_k",
            "supports_metadata_filtering",
            "supports_namespaces",
            "description",
        }


@pytest.mark.asyncio
async def test_plugin_error_mapping_NotSupported_to_KernelFunctionException(adapter: Any) -> None:
    """NotSupported should map to KernelFunctionException with a safe message."""
    s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter, async_embedding_function=lambda t: asyncio.sleep(0, result=[[0.1] for _ in t]))
    plugin = CorpusSemanticKernelVectorPlugin(vector_store=s)

    # Force store.asimilarity_search to raise NotSupported.
    async def fail(*_: Any, **__: Any) -> Any:
        raise NotSupported("nope", code="NOPE")

    with patch.object(s, "asimilarity_search", side_effect=fail):
        with pytest.raises(sk_adapter_module.KernelFunctionException):
            _ = await plugin.vector_search("q")


@pytest.mark.asyncio
async def test_plugin_error_mapping_BadRequest_to_KernelFunctionException(adapter: Any) -> None:
    """BadRequest should map to KernelFunctionException with a safe message."""
    s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter, async_embedding_function=lambda t: asyncio.sleep(0, result=[[0.1] for _ in t]))
    plugin = CorpusSemanticKernelVectorPlugin(vector_store=s)

    async def fail(*_: Any, **__: Any) -> Any:
        raise BadRequest("bad", code="BAD")

    with patch.object(s, "asimilarity_search", side_effect=fail):
        with pytest.raises(sk_adapter_module.KernelFunctionException):
            _ = await plugin.vector_search("q")


@pytest.mark.asyncio
async def test_plugin_error_mapping_VectorAdapterError_to_KernelFunctionException(adapter: Any) -> None:
    """VectorAdapterError should map to KernelFunctionException with a safe message."""
    s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter, async_embedding_function=lambda t: asyncio.sleep(0, result=[[0.1] for _ in t]))
    plugin = CorpusSemanticKernelVectorPlugin(vector_store=s)

    async def fail(*_: Any, **__: Any) -> Any:
        raise VectorAdapterError("db down", code="DB_DOWN")

    with patch.object(s, "asimilarity_search", side_effect=fail):
        with pytest.raises(sk_adapter_module.KernelFunctionException):
            _ = await plugin.vector_search("q")


@pytest.mark.asyncio
async def test_plugin_error_mapping_generic_exception_to_KernelFunctionException(adapter: Any) -> None:
    """Unexpected exceptions should be wrapped safely into KernelFunctionException."""
    s = CorpusSemanticKernelVectorStore(corpus_adapter=adapter, async_embedding_function=lambda t: asyncio.sleep(0, result=[[0.1] for _ in t]))
    plugin = CorpusSemanticKernelVectorPlugin(vector_store=s)

    async def fail(*_: Any, **__: Any) -> Any:
        raise RuntimeError("boom")

    with patch.object(s, "asimilarity_search", side_effect=fail):
        with pytest.raises(sk_adapter_module.KernelFunctionException):
            _ = await plugin.vector_search("q")


# ---------------------------------------------------------------------------
# Semantic Kernel End-to-End Integration Tests (6 tests - NO SKIPS)
# ---------------------------------------------------------------------------


def test_e2e_semantic_kernel_is_installed_and_importable() -> None:
    """
    E2E: Confirm Semantic Kernel is installed in the test environment.

    This is intentionally a hard requirement for E2E coverage: if SK is not
    installed, these tests should fail loudly rather than being skipped.
    """
    import semantic_kernel  # noqa: F401
    from semantic_kernel import Kernel  # noqa: F401

    assert True


def test_e2e_store_add_texts_and_similarity_search_end_to_end() -> None:
    """
    E2E: Real store + real translator + real adapter upsert/query path.

    Validates that:
      - texts are embedded and written
      - similarity_search queries back results
      - returned docs conform to AI-facing schema
    """
    adapter_e2e = E2EInMemoryVectorAdapter()
    store_e2e = CorpusSemanticKernelVectorStore(
        corpus_adapter=adapter_e2e,
        embedding_function=_e2e_embed_texts,
        namespace="docs",
        default_top_k=4,
        score_threshold=None,
    )

    ids = store_e2e.add_texts(
        ["alpha document", "beta document", "gamma note"],
        metadatas=[{"category": "a"}, {"category": "b"}, {"category": "c"}],
        # Provide explicit ids to make assertions stable.
        ids=["alpha", "beta", "gamma"],
        namespace="docs",
    )
    assert ids == ["alpha", "beta", "gamma"]

    results = store_e2e.similarity_search("alpha", k=2, namespace="docs")
    assert isinstance(results, list)
    assert len(results) >= 1
    assert set(results[0].keys()) == {"content", "metadata", "confidence", "source"}
    assert results[0]["source"] == "vector_database"


@pytest.mark.asyncio
async def test_e2e_store_async_add_and_search_end_to_end() -> None:
    """
    E2E: Real async write + read path (no patching).

    This verifies:
      - async embedding function path is exercised
      - translator async methods execute correctly against the adapter
    """
    adapter_e2e = E2EInMemoryVectorAdapter()
    store_e2e = CorpusSemanticKernelVectorStore(
        corpus_adapter=adapter_e2e,
        async_embedding_function=_e2e_aembed_texts,
        namespace="async_docs",
        default_top_k=4,
        score_threshold=None,
    )

    ids = await store_e2e.aadd_texts(
        ["async one", "async two"],
        metadatas=[{"tag": "one"}, {"tag": "two"}],
        ids=["one", "two"],
        namespace="async_docs",
    )
    assert ids == ["one", "two"]

    results = await store_e2e.asimilarity_search("async one", k=1, namespace="async_docs")
    assert isinstance(results, list)
    assert len(results) == 1
    assert set(results[0].keys()) == {"content", "metadata", "confidence", "source"}


def test_e2e_store_streaming_similarity_search_end_to_end() -> None:
    """
    E2E: Streaming query path against a real adapter.

    This validates:
      - similarity_search_stream returns an iterator of AI-facing dicts
      - results are produced without buffering everything in caller code
    """
    adapter_e2e = E2EInMemoryVectorAdapter()
    store_e2e = CorpusSemanticKernelVectorStore(
        corpus_adapter=adapter_e2e,
        embedding_function=_e2e_embed_texts,
        namespace="stream_docs",
    )

    _ = store_e2e.add_texts(
        ["stream alpha", "stream beta"],
        metadatas=[{"s": 1}, {"s": 2}],
        ids=["sa", "sb"],
        namespace="stream_docs",
    )

    streamed = list(store_e2e.similarity_search_stream("stream", k=2, namespace="stream_docs"))
    assert isinstance(streamed, list)
    assert len(streamed) >= 1
    assert "content" in streamed[0]
    assert "confidence" in streamed[0]


@pytest.mark.asyncio
async def test_e2e_plugin_can_be_added_to_kernel_and_invoked_vector_search() -> None:
    """
    E2E: Semantic Kernel Kernel + plugin registration + kernel.invoke.

    This exercises the full SK integration surface:
      Kernel -> plugin registry -> kernel_function -> plugin method -> store -> translator -> adapter.
    """
    from semantic_kernel import Kernel
    from semantic_kernel.functions.kernel_arguments import KernelArguments

    adapter_e2e = E2EInMemoryVectorAdapter()
    store_e2e = CorpusSemanticKernelVectorStore(
        corpus_adapter=adapter_e2e,
        embedding_function=_e2e_embed_texts,
        namespace="plugin_docs",
    )
    plugin = CorpusSemanticKernelVectorPlugin(vector_store=store_e2e)

    # Seed documents through the store (real upsert chain).
    _ = store_e2e.add_texts(
        ["the quick brown fox", "jumps over the lazy dog"],
        metadatas=[{"animal": "fox"}, {"animal": "dog"}],
        ids=["fox", "dog"],
        namespace="plugin_docs",
    )

    kernel = Kernel()
    kernel.add_plugin(plugin, plugin_name="corpus_vector")

    # Access the function from the plugin registry (common SK pattern).
    sk_func = kernel.plugins["corpus_vector"]["vector_search"]

    result = await kernel.invoke(sk_func, KernelArguments(query="quick fox", k=1, namespace="plugin_docs"))
    value = getattr(result, "value", result)

    assert isinstance(value, list)
    assert len(value) == 1
    assert "content" in value[0]
    assert "metadata" in value[0]
    assert value[0]["source"] == "vector_database"


@pytest.mark.asyncio
async def test_e2e_plugin_store_document_and_get_capabilities_via_kernel() -> None:
    """
    E2E: Plugin helper functions invoked through SK.

    This validates:
      - store_document writes via plugin surface
      - get_capabilities returns an AI-facing dict surface via plugin
    """
    from semantic_kernel import Kernel
    from semantic_kernel.functions.kernel_arguments import KernelArguments

    adapter_e2e = E2EInMemoryVectorAdapter()
    store_e2e = CorpusSemanticKernelVectorStore(
        corpus_adapter=adapter_e2e,
        embedding_function=_e2e_embed_texts,
        namespace="plugin2_docs",
        default_top_k=3,
    )
    plugin = CorpusSemanticKernelVectorPlugin(vector_store=store_e2e)

    kernel = Kernel()
    kernel.add_plugin(plugin, plugin_name="corpus_vector")

    store_doc_fn = kernel.plugins["corpus_vector"]["vector_store_document"]
    caps_fn = kernel.plugins["corpus_vector"]["vector_get_capabilities"]

    # Store via plugin surface.
    doc_id_result = await kernel.invoke(
        store_doc_fn,
        KernelArguments(
            content="stored via plugin",
            metadata={"source": "e2e"},
            # Provide a stable id for test determinism when the plugin supports it.
            document_id="doc-e2e-1",
            namespace="plugin2_docs",
        ),
    )
    doc_id = getattr(doc_id_result, "value", doc_id_result)
    assert isinstance(doc_id, str)
    assert doc_id  # non-empty

    caps_result = await kernel.invoke(caps_fn, KernelArguments())
    caps = getattr(caps_result, "value", caps_result)

    assert isinstance(caps, dict)
    assert caps.get("supports_metadata_filtering") is True
    assert caps.get("supports_namespaces") is True
    expected_caps = await adapter_e2e.capabilities()
    assert int(caps.get("max_top_k") or 0) == int(expected_caps.max_top_k or 0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
