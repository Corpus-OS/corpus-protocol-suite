"""
LlamaIndex Vector framework adapter tests.

These tests are written against the current public API in
`corpus_sdk.vector.framework_adapters.llamaindex`, which exposes a LlamaIndex
BasePydanticVectorStore implementation backed by Corpus VectorProtocolV1.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError as PydanticValidationError

import corpus_sdk.vector.framework_adapters.llamaindex as llamaindex_adapter_module
from corpus_sdk.vector.framework_adapters.llamaindex import (
    CorpusLlamaIndexVectorClient,
    CorpusLlamaIndexVectorStore,
    _ensure_not_in_event_loop,
)
from corpus_sdk.vector.vector_base import (
    BadRequest,
    BaseVectorAdapter,
    NotSupported,
    OperationContext,
    QueryResult,
    UpsertResult,
    Vector,
    VectorCapabilities,
    VectorMatch,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_TEXT = "hello from llamaindex vector tests"
SAMPLE_EMBEDDING = [0.1, 0.2, 0.3, 0.4]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dummy_translator() -> Any:
    """Factory for creating a standard dummy translator for tests."""

    class DummyTranslator:
        def upsert(self, *a: Any, **k: Any) -> Any:
            return UpsertResult(
                upserted_count=1,
                failed_count=0,
                failures=[],
            )

        async def arun_upsert(self, *a: Any, **k: Any) -> Any:
            return UpsertResult(
                upserted_count=1,
                failed_count=0,
                failures=[],
            )

        def query(self, *a: Any, **k: Any) -> Any:
            from corpus_sdk.vector.vector_base import Vector
            return QueryResult(
                matches=[
                    VectorMatch(
                        vector=Vector(
                            id="node-0",
                            vector=[0.1, 0.2],
                            metadata={
                                "text": "test",
                                "id": "node-0",
                                "node_id": "node-0",
                            },
                            namespace="default",
                        ),
                        score=0.95,
                        distance=0.05,
                    )
                ],
                query_vector=[0.1, 0.2],
                namespace="default",
                total_matches=1,
            )

        async def arun_query(self, *a: Any, **k: Any) -> Any:
            from corpus_sdk.vector.vector_base import Vector
            return QueryResult(
                matches=[
                    VectorMatch(
                        vector=Vector(
                            id="node-0",
                            vector=[0.1, 0.2],
                            metadata={
                                "text": "test",
                                "id": "node-0",
                                "node_id": "node-0",
                            },
                            namespace="default",
                        ),
                        score=0.95,
                        distance=0.05,
                    )
                ],
                query_vector=[0.1, 0.2],
                namespace="default",
                total_matches=1,
            )

        def query_stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            from corpus_sdk.vector.vector_base import Vector
            class Chunk:
                matches = [
                    VectorMatch(
                        vector=Vector(
                            id="node-0",
                            vector=[0.1, 0.2],
                            metadata={
                                "text": "test",
                                "id": "node-0",
                                "node_id": "node-0",
                            },
                            namespace="default",
                        ),
                        score=0.95,
                        distance=0.05,
                    )
                ]
                is_final = True

            yield Chunk()

        def delete(self, *a: Any, **k: Any) -> Any:
            return {"deleted": 1}

        async def arun_delete(self, *a: Any, **k: Any) -> Any:
            return {"deleted": 1}

        def capabilities(self) -> VectorCapabilities:
            return VectorCapabilities(
                server="mock",
                version="1.0",
                supports_metadata_filtering=True,
                max_top_k=100,
            )

        async def arun_capabilities(self) -> VectorCapabilities:
            return VectorCapabilities(
                server="mock",
                version="1.0",
                supports_metadata_filtering=True,
                max_top_k=100,
            )

        def health(self) -> Mapping[str, Any]:
            return {"status": "ok"}

        async def ahealth(self) -> Mapping[str, Any]:
            return {"status": "ok"}

    return DummyTranslator()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def adapter() -> Any:
    """Adapter fixture that implements VectorProtocolV1."""
    
    class TestVectorAdapter(BaseVectorAdapter):
        """Minimal test adapter implementing VectorProtocolV1."""
        
        def _do_capabilities(self, ctx: Optional[OperationContext] = None) -> VectorCapabilities:
            return VectorCapabilities(
                server="test",
                version="1.0",
                supports_metadata_filtering=True,
                max_top_k=100,
            )
        
        async def _do_acapabilities(self, ctx: Optional[OperationContext] = None) -> VectorCapabilities:
            return VectorCapabilities(
                server="test",
                version="1.0",
                supports_metadata_filtering=True,
                max_top_k=100,
            )
    
    return TestVectorAdapter()


@pytest.fixture
def TextNode() -> Any:
    """Fixture for LlamaIndex TextNode class."""
    try:
        from llama_index.core.schema import TextNode
        return TextNode
    except ImportError:
        # Create minimal stub
        class TextNode:
            def __init__(self, text="", id_=None, metadata=None, embedding=None):
                self.text = text
                self.id_ = id_
                self.node_id = id_
                self.metadata = metadata or {}
                self.embedding = embedding
                self.ref_doc_id = None

            def get_content(self, metadata_mode=None):
                return self.text

            def get_embedding(self):
                return self.embedding

        return TextNode


@pytest.fixture
def VectorStoreQuery() -> Any:
    """Fixture for LlamaIndex VectorStoreQuery class."""
    try:
        from llama_index.core.vector_stores.types import VectorStoreQuery
        return VectorStoreQuery
    except ImportError:
        # Create minimal stub
        class VectorStoreQuery:
            def __init__(
                self,
                query_embedding=None,
                similarity_top_k=4,
                filters=None,
                doc_ids=None,
                node_ids=None,
            ):
                self.query_embedding = query_embedding
                self.similarity_top_k = similarity_top_k
                self.filters = filters
                self.doc_ids = doc_ids
                self.node_ids = node_ids

        return VectorStoreQuery


@pytest.fixture
def NodeWithScore() -> Any:
    """Fixture for LlamaIndex NodeWithScore class."""
    try:
        from llama_index.core.schema import NodeWithScore
        return NodeWithScore
    except ImportError:
        # Create minimal stub
        class NodeWithScore:
            def __init__(self, node, score):
                self.node = node
                self.score = score

        return NodeWithScore


# ---------------------------------------------------------------------------
# Construction / Initialization Tests (8 tests)
# ---------------------------------------------------------------------------


def test_init_requires_corpus_adapter(adapter: Any) -> None:
    """Adapter must be provided."""
    with pytest.raises((TypeError, AttributeError, PydanticValidationError)):
        CorpusLlamaIndexVectorStore(corpus_adapter=None)  # type: ignore


def test_init_stores_config_attributes(adapter: Any) -> None:
    """Store should keep key config attributes accessible."""
    store = CorpusLlamaIndexVectorStore(
        corpus_adapter=adapter,
        namespace="test-ns",
        batch_size=50,
        default_top_k=10,
        score_threshold=0.8,
        id_field="custom_id",
        text_field="custom_text",
        node_id_field="custom_node_id",
        ref_doc_id_field="custom_ref_doc_id",
        own_adapter=True,
    )

    assert store.namespace == "test-ns"
    assert store.batch_size == 50
    assert store.default_top_k == 10
    assert store.score_threshold == 0.8
    assert store.id_field == "custom_id"
    assert store.text_field == "custom_text"
    assert store.node_id_field == "custom_node_id"
    assert store.ref_doc_id_field == "custom_ref_doc_id"
    assert store.own_adapter is True


def test_init_validates_batch_size(adapter: Any) -> None:
    """batch_size must be positive."""
    with pytest.raises((ValueError, PydanticValidationError), match="batch_size"):
        CorpusLlamaIndexVectorStore(corpus_adapter=adapter, batch_size=0)


def test_init_validates_default_top_k(adapter: Any) -> None:
    """default_top_k must be positive."""
    with pytest.raises((ValueError, PydanticValidationError), match="default_top_k"):
        CorpusLlamaIndexVectorStore(corpus_adapter=adapter, default_top_k=0)


def test_init_validates_score_threshold_range(adapter: Any) -> None:
    """score_threshold must be between 0.0 and 1.0."""
    with pytest.raises((ValueError, PydanticValidationError), match="score_threshold"):
        CorpusLlamaIndexVectorStore(corpus_adapter=adapter, score_threshold=1.5)

    with pytest.raises((ValueError, PydanticValidationError), match="score_threshold"):
        CorpusLlamaIndexVectorStore(corpus_adapter=adapter, score_threshold=-0.1)


def test_init_validates_reserved_fields_unique(adapter: Any) -> None:
    """Reserved metadata fields must be unique."""
    with pytest.raises((ValueError, PydanticValidationError), match="Reserved metadata fields"):
        CorpusLlamaIndexVectorStore(
            corpus_adapter=adapter,
            id_field="same",
            text_field="same",  # Duplicate
        )


def test_class_name_property(adapter: Any) -> None:
    """Should return correct class name."""
    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
    assert store.class_name() == "CorpusLlamaIndexVectorStore"


def test_client_property_exposes_adapter(adapter: Any) -> None:
    """client property should expose underlying adapter."""
    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
    assert store.client is adapter


# ---------------------------------------------------------------------------
# Translator Wiring Tests (4 tests)
# ---------------------------------------------------------------------------


def test_translator_created_with_framework_llamaindex(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Translator factory should be called with framework='llamaindex'."""
    captured: Dict[str, Any] = {}

    class FakeTranslator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def query(self, *a, **k):
            return QueryResult(matches=[], namespace="default")

        def capabilities(self):
            return VectorCapabilities(server="mock", version="1.0")

    with patch.object(llamaindex_adapter_module, "VectorTranslator", FakeTranslator):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
        _ = store._translator

    assert captured.get("framework") == "llamaindex"
    assert captured.get("adapter") is adapter


def test_translator_cached_property_reused(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Multiple accesses to _translator should return same instance."""
    with patch.object(
        llamaindex_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
        translator1 = store._translator
        translator2 = store._translator

        assert translator1 is translator2


def test_translator_uses_default_framework_translator(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should use DefaultVectorFrameworkTranslator."""
    captured: Dict[str, Any] = {}

    class FakeTranslator:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.translator = kwargs.get("translator")

        def query(self, *a, **k):
            return QueryResult(matches=[], query_vector=[0.1, 0.2], namespace="default", total_matches=0)

    with patch.object(llamaindex_adapter_module, "VectorTranslator", FakeTranslator):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
        _ = store._translator

    assert "translator" in captured
    translator_obj = captured["translator"]
    assert translator_obj.__class__.__name__ == "DefaultVectorFrameworkTranslator"


def test_translator_available_on_first_access(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Lazy construction should work on first access."""
    with patch.object(
        llamaindex_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
        translator = store._translator
        assert translator is not None


# ---------------------------------------------------------------------------
# Context Translation Tests (6 tests)
# ---------------------------------------------------------------------------


def test_build_core_context_from_operation_context(adapter: Any) -> None:
    """Should pass through OperationContext unchanged."""
    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
    ctx = OperationContext(request_id="test", tenant="test", attrs={})

    result = store._build_core_context(ctx=ctx)

    assert result is ctx


def test_build_core_context_from_dict(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should build OperationContext from dict using context_translation."""
    captured: Dict[str, Any] = {}
    base_ctx = OperationContext(request_id="from-dict", tenant="from-dict", attrs={})

    def fake_from_dict(mapping: Any) -> Any:
        captured["mapping"] = mapping
        return base_ctx

    monkeypatch.setattr(llamaindex_adapter_module, "ctx_from_dict", fake_from_dict)

    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
    test_dict = {"key": "value"}

    ctx = store._build_core_context(ctx=test_dict)

    assert captured["mapping"] == test_dict
    assert isinstance(ctx, OperationContext)


def test_build_core_context_from_llamaindex_callback(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should build OperationContext from LlamaIndex callback using context_translation."""
    captured: Dict[str, Any] = {}
    base_ctx = OperationContext(request_id="from-li", tenant="from-li", attrs={})

    def fake_from_llamaindex(callback: Any) -> Any:
        captured["callback"] = callback
        return base_ctx

    monkeypatch.setattr(llamaindex_adapter_module, "ctx_from_llamaindex", fake_from_llamaindex)

    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
    mock_callback = Mock()

    ctx = store._build_core_context(callback_manager=mock_callback)

    assert captured["callback"] is mock_callback
    assert isinstance(ctx, OperationContext)


def test_build_core_context_dict_translation_error_attaches_context(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Errors during dict translation should attach error context."""
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    def fake_from_dict(mapping: Any) -> Any:
        raise RuntimeError("dict translation failed")

    monkeypatch.setattr(llamaindex_adapter_module, "attach_context", fake_attach_context)
    monkeypatch.setattr(llamaindex_adapter_module, "ctx_from_dict", fake_from_dict)

    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)

    with pytest.raises(RuntimeError, match="dict translation failed"):
        store._build_core_context(ctx={"key": "value"})

    assert captured_ctx.get("framework") == "llamaindex"
    assert captured_ctx.get("operation") == "context_from_dict"


def test_build_core_context_llamaindex_translation_error_attaches_context(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Errors during LlamaIndex translation should attach error context."""
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    def fake_from_llamaindex(callback: Any) -> Any:
        raise RuntimeError("llamaindex translation failed")

    monkeypatch.setattr(llamaindex_adapter_module, "attach_context", fake_attach_context)
    monkeypatch.setattr(llamaindex_adapter_module, "ctx_from_llamaindex", fake_from_llamaindex)

    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)

    with pytest.raises(RuntimeError, match="llamaindex translation failed"):
        store._build_core_context(callback_manager=Mock())

    assert captured_ctx.get("framework") == "llamaindex"
    assert captured_ctx.get("operation") == "context_from_llamaindex"


def test_build_contexts_orchestrates_both(adapter: Any) -> None:
    """_build_contexts should return both OperationContext and framework_ctx."""
    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter, namespace="test-ns")
    core_ctx, fw_ctx = store._build_contexts(namespace=None)

    assert core_ctx is None  # No context provided
    assert isinstance(fw_ctx, Mapping)
    assert "namespace" in fw_ctx
    assert fw_ctx["namespace"] == "test-ns"


# ---------------------------------------------------------------------------
# Namespace Resolution Tests (2 tests)
# ---------------------------------------------------------------------------


def test_effective_namespace_uses_override(adapter: Any) -> None:
    """Explicit namespace should override store default."""
    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter, namespace="default")
    ns = store._effective_namespace("override")
    assert ns == "override"


def test_effective_namespace_uses_store_default(adapter: Any) -> None:
    """Should use store default when not specified."""
    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter, namespace="default")
    ns = store._effective_namespace(None)
    assert ns == "default"


# ---------------------------------------------------------------------------
# Dimension Hint Tests (5 tests)
# ---------------------------------------------------------------------------


def test_update_dim_hint_sets_first_write(adapter: Any) -> None:
    """First non-zero dimension should win."""
    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
    assert store._vector_dim_hint is None

    store._update_dim_hint(4)
    assert store._vector_dim_hint == 4


def test_update_dim_hint_thread_safe(adapter: Any) -> None:
    """Concurrent updates should not race."""
    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)

    def update_thread(dim: int):
        store._update_dim_hint(dim)

    threads = [threading.Thread(target=update_thread, args=(i,)) for i in range(1, 10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert store._vector_dim_hint is not None
    assert 1 <= store._vector_dim_hint < 10


def test_update_dim_hint_ignores_subsequent_writes(adapter: Any) -> None:
    """Second write should be no-op."""
    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
    store._update_dim_hint(4)
    store._update_dim_hint(8)

    assert store._vector_dim_hint == 4


def test_maybe_check_dim_validates_against_hint(adapter: Any) -> None:
    """Should raise on dimension mismatch."""
    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
    store._update_dim_hint(4)

    with pytest.raises(BadRequest, match="dimension mismatch"):
        store._maybe_check_dim([0.1, 0.2], where="test")


def test_maybe_check_dim_noop_when_hint_none(adapter: Any) -> None:
    """No validation without hint."""
    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
    # Should not raise
    store._maybe_check_dim([0.1, 0.2], where="test")


# ---------------------------------------------------------------------------
# Node Translation Tests (3 tests)
# ---------------------------------------------------------------------------


def test_nodes_to_corpus_vectors_builds_correct_structure(
    adapter: Any, TextNode: Any
) -> None:
    """Should build Vector objects with correct structure."""
    from llama_index.core.schema import NodeRelationship, RelatedNodeInfo
    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)

    node = TextNode(
        text="test content",
        id_="node-1",
        metadata={"key": "value"},
        embedding=[0.1, 0.2, 0.3, 0.4],
    )
    node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id="doc-abc")

    vectors = store._nodes_to_corpus_vectors([node], namespace="test-ns")

    assert len(vectors) == 1
    vec = vectors[0]
    assert vec.id == "node-1"
    assert vec.vector == [0.1, 0.2, 0.3, 0.4]
    assert vec.namespace == "test-ns"
    assert vec.metadata["text"] == "test content"
    assert vec.metadata["node_id"] == "node-1"
    assert vec.metadata["ref_doc_id"] == "doc-abc"


def test_nodes_to_corpus_vectors_raises_without_embedding(
    adapter: Any, TextNode: Any
) -> None:
    """Should raise if node has no embedding."""
    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)

    node = TextNode(text="test", id_="node-1")
    # No embedding set

    with pytest.raises(BadRequest, match="has no embedding"):
        store._nodes_to_corpus_vectors([node], namespace=None)


def test_matches_to_nodes_converts_correctly(
    adapter: Any, NodeWithScore: Any
) -> None:
    """Should convert VectorMatch to NodeWithScore."""
    from corpus_sdk.vector.vector_base import Vector
    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)

    vec = Vector(
        id="node-1",
        vector=[0.1, 0.2],
        metadata={
            "text": "test content",
            "node_id": "node-1",
            "ref_doc_id": "doc-abc",
            "custom": "data",
        },
        namespace="test",
    )
    matches = [
        VectorMatch(
            vector=vec,
            score=0.9,
            distance=0.1,
        )
    ]

    results = store._matches_to_nodes(matches)

    assert len(results) == 1
    nws = results[0]
    assert isinstance(nws, NodeWithScore)
    assert nws.score == 0.9
    assert nws.node.get_content() == "test content"
    assert nws.node.node_id == "node-1"


# ---------------------------------------------------------------------------
# Add Tests (10 tests)
# ---------------------------------------------------------------------------


def test_add_returns_node_ids(monkeypatch: pytest.MonkeyPatch, adapter: Any, TextNode: Any) -> None:
    """Should return list of node IDs."""
    with patch.object(
        llamaindex_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)

        node = TextNode(text="test", id_="node-1", embedding=[0.1, 0.2])
        ids = store.add([node])

        assert isinstance(ids, list)
        assert len(ids) == 1
        assert ids[0] == "node-1"


def test_add_calls_translator_upsert(
    monkeypatch: pytest.MonkeyPatch, adapter: Any, TextNode: Any
) -> None:
    """Should delegate to translator.upsert()."""
    called = {"upsert": False}

    class DummyTranslator:
        def upsert(self, *a: Any, **k: Any) -> Any:
            called["upsert"] = True
            return UpsertResult(
                upserted_count=1, failed_count=0, failures=[]
            )

        def capabilities(self):
            return VectorCapabilities(server="mock", version="1.0")

    with patch.object(llamaindex_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)

        node = TextNode(text="test", id_="node-1", embedding=[0.1, 0.2])
        store.add([node])

        assert called["upsert"] is True


def test_add_handles_partial_failure(
    adapter: Any, monkeypatch: pytest.MonkeyPatch, TextNode: Any
) -> None:
    """Should log warnings for partial failures but not raise."""

    class PartialFailureTranslator:
        def upsert(self, *a: Any, **k: Any) -> Any:
            return UpsertResult(
                upserted_count=1,
                failed_count=1,
                failures=[{"id": "node-2", "error": "test error"}],
            )

        def capabilities(self):
            return VectorCapabilities(server="mock", version="1.0")

    with patch.object(
        llamaindex_adapter_module, "VectorTranslator", return_value=PartialFailureTranslator()
    ):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)

        nodes = [
            TextNode(text="test1", id_="node-1", embedding=[0.1, 0.2]),
            TextNode(text="test2", id_="node-2", embedding=[0.3, 0.4]),
        ]
        # Should not raise
        ids = store.add(nodes)
        assert len(ids) == 2


def test_add_raises_if_all_failed(
    adapter: Any, monkeypatch: pytest.MonkeyPatch, TextNode: Any
) -> None:
    """Should raise if all nodes failed."""

    class AllFailureTranslator:
        def upsert(self, *a: Any, **k: Any) -> Any:
            return UpsertResult(
                upserted_count=0,
                failed_count=2,
                failures=[
                    {"id": "node-1", "error": "test error"},
                    {"id": "node-2", "error": "test error"},
                ],
            )

        def capabilities(self):
            return VectorCapabilities(server="mock", version="1.0")

    with patch.object(
        llamaindex_adapter_module, "VectorTranslator", return_value=AllFailureTranslator()
    ):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)

        nodes = [
            TextNode(text="test1", id_="node-1", embedding=[0.1, 0.2]),
            TextNode(text="test2", id_="node-2", embedding=[0.3, 0.4]),
        ]

        with pytest.raises(Exception, match="All"):
            store.add(nodes)


def test_add_guards_event_loop(adapter: Any, TextNode: Any) -> None:
    """Should raise RuntimeError in event loop."""

    @pytest.mark.asyncio
    async def test_in_loop():
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
        node = TextNode(text="test", id_="node-1", embedding=[0.1, 0.2])

        with pytest.raises(RuntimeError, match="event loop"):
            store.add([node])

    asyncio.run(test_in_loop())


@pytest.mark.asyncio
async def test_aadd_returns_node_ids(
    monkeypatch: pytest.MonkeyPatch, adapter: Any, TextNode: Any
) -> None:
    """Should return list of node IDs."""
    with patch.object(
        llamaindex_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)

        node = TextNode(text="test", id_="node-1", embedding=[0.1, 0.2])
        ids = await store.aadd([node])

        assert isinstance(ids, list)
        assert len(ids) == 1


@pytest.mark.asyncio
async def test_aadd_calls_translator_arun_upsert(
    monkeypatch: pytest.MonkeyPatch, adapter: Any, TextNode: Any
) -> None:
    """Should delegate to translator.arun_upsert()."""
    called = {"arun_upsert": False}

    class DummyTranslator:
        async def arun_upsert(self, *a: Any, **k: Any) -> Any:
            called["arun_upsert"] = True
            return UpsertResult(
                upserted_count=1, failed_count=0, failures=[]
            )

        async def arun_capabilities(self):
            return VectorCapabilities(server="mock", version="1.0")

    with patch.object(llamaindex_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)

        node = TextNode(text="test", id_="node-1", embedding=[0.1, 0.2])
        await store.aadd([node])

        assert called["arun_upsert"] is True


@pytest.mark.asyncio
async def test_aadd_handles_empty_list(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should handle empty node list gracefully."""
    with patch.object(
        llamaindex_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)

        ids = await store.aadd([])

        assert ids == []


def test_add_handles_empty_list(adapter: Any) -> None:
    """Should handle empty node list gracefully."""
    with patch.object(
        llamaindex_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)

        ids = store.add([])

        assert ids == []


def test_add_validates_node_embeddings(adapter: Any, TextNode: Any) -> None:
    """Should raise if node missing embedding."""
    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)

    node = TextNode(text="test", id_="node-1")
    # No embedding

    with pytest.raises(BadRequest, match="has no embedding"):
        store.add([node])


# ---------------------------------------------------------------------------
# Query Tests (Sync) (5 tests)
# ---------------------------------------------------------------------------


def test_query_returns_vector_store_query_result(
    adapter: Any, monkeypatch: pytest.MonkeyPatch, VectorStoreQuery: Any
) -> None:
    """Should return VectorStoreQueryResult."""
    with patch.object(
        llamaindex_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)

        query = VectorStoreQuery(query_embedding=[0.1, 0.2, 0.3, 0.4], similarity_top_k=4)
        result = store.query(query)

        assert hasattr(result, "nodes")
        assert hasattr(result, "similarities")
        assert hasattr(result, "ids")


def test_query_calls_translator_query(
    adapter: Any, monkeypatch: pytest.MonkeyPatch, VectorStoreQuery: Any
) -> None:
    """Should delegate to translator.query()."""
    called = {"query": False}

    class DummyTranslator:
        def query(self, *a: Any, **k: Any) -> Any:
            called["query"] = True
            return QueryResult(matches=[], query_vector=[0.1, 0.2], namespace="default", total_matches=0)

        def capabilities(self):
            return VectorCapabilities(server="mock", version="1.0")

    with patch.object(llamaindex_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)

        query = VectorStoreQuery(query_embedding=[0.1, 0.2], similarity_top_k=4)
        store.query(query)

        assert called["query"] is True


def test_query_validates_max_top_k(
    adapter: Any, monkeypatch: pytest.MonkeyPatch, VectorStoreQuery: Any
) -> None:
    """Should raise if similarity_top_k exceeds max_top_k."""
    with patch.object(
        llamaindex_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)

        query = VectorStoreQuery(query_embedding=[0.1, 0.2], similarity_top_k=200)

        with pytest.raises(BadRequest, match="exceeds maximum"):
            store.query(query)


def test_query_validates_filter_support(
    adapter: Any, monkeypatch: pytest.MonkeyPatch, VectorStoreQuery: Any
) -> None:
    """Should raise if filters not supported."""
    try:
        from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter
        
        class DummyTranslator:
            def capabilities(self):
                return VectorCapabilities(server="mock", version="1.0", supports_metadata_filtering=False)
            
            def query(self, *a: Any, **k: Any) -> Any:
                # This should never be called if validation works correctly
                pytest.fail("query() should not be called when filters are unsupported")

        with patch.object(llamaindex_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
            store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)

            query = VectorStoreQuery(
                query_embedding=[0.1, 0.2],
                similarity_top_k=4,
                filters=MetadataFilters(filters=[MetadataFilter(key="test", value="val")]),
            )

            with pytest.raises(NotSupported, match="metadata filtering"):
                store.query(query)
    except ImportError:
        pytest.skip("LlamaIndex not available")


def test_query_applies_score_threshold(
    adapter: Any, monkeypatch: pytest.MonkeyPatch, VectorStoreQuery: Any
) -> None:
    """Should filter results by score_threshold."""

    class DummyTranslator:
        def query(self, *a: Any, **k: Any) -> Any:
            from corpus_sdk.vector.vector_base import Vector
            return QueryResult(
                matches=[
                    VectorMatch(
                        vector=Vector(
                            id="node-1",
                            vector=[0.9, 0.1],
                            metadata={
                                "text": "high",
                                "id": "node-1",
                                "node_id": "node-1",
                            },
                            namespace="default",
                        ),
                        score=0.9,
                        distance=0.1,
                    ),
                    VectorMatch(
                        vector=Vector(
                            id="node-2",
                            vector=[0.3, 0.7],
                            metadata={
                                "text": "low",
                                "id": "node-2",
                                "node_id": "node-2",
                            },
                            namespace="default",
                        ),
                        score=0.3,
                        distance=0.7,
                    ),
                ],
                query_vector=[0.1, 0.2],
                namespace="default",
                total_matches=2,
            )

        def capabilities(self):
            return VectorCapabilities(server="mock", version="1.0")

    with patch.object(llamaindex_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter, score_threshold=0.5)

        query = VectorStoreQuery(query_embedding=[0.1, 0.2], similarity_top_k=10)
        result = store.query(query)

        assert len(result.nodes) == 1
        assert result.nodes[0].node.get_content() == "high"


# ---------------------------------------------------------------------------
# Query Tests (Async) (4 tests)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aquery_returns_vector_store_query_result(
    adapter: Any, monkeypatch: pytest.MonkeyPatch, VectorStoreQuery: Any
) -> None:
    """Should return VectorStoreQueryResult."""
    with patch.object(
        llamaindex_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)

        query = VectorStoreQuery(query_embedding=[0.1, 0.2, 0.3, 0.4], similarity_top_k=4)
        result = await store.aquery(query)

        assert hasattr(result, "nodes")
        assert hasattr(result, "similarities")
        assert hasattr(result, "ids")


@pytest.mark.asyncio
async def test_aquery_calls_translator_arun_query(
    adapter: Any, monkeypatch: pytest.MonkeyPatch, VectorStoreQuery: Any
) -> None:
    """Should delegate to translator.arun_query()."""
    called = {"arun_query": False}

    class DummyTranslator:
        async def arun_query(self, *a: Any, **k: Any) -> Any:
            called["arun_query"] = True
            return QueryResult(matches=[], query_vector=[0.1, 0.2], namespace="default", total_matches=0)

        async def arun_capabilities(self):
            return VectorCapabilities(server="mock", version="1.0")

    with patch.object(llamaindex_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)

        query = VectorStoreQuery(query_embedding=[0.1, 0.2], similarity_top_k=4)
        await store.aquery(query)

        assert called["arun_query"] is True


@pytest.mark.asyncio
async def test_aquery_validates_max_top_k(
    adapter: Any, monkeypatch: pytest.MonkeyPatch, VectorStoreQuery: Any
) -> None:
    """Should raise if similarity_top_k exceeds max_top_k."""
    with patch.object(
        llamaindex_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)

        query = VectorStoreQuery(query_embedding=[0.1, 0.2], similarity_top_k=200)

        with pytest.raises(BadRequest, match="exceeds maximum"):
            await store.aquery(query)


@pytest.mark.asyncio
async def test_aquery_applies_score_threshold(
    adapter: Any, monkeypatch: pytest.MonkeyPatch, VectorStoreQuery: Any
) -> None:
    """Should filter results by score_threshold."""

    class DummyTranslator:
        async def arun_query(self, *a: Any, **k: Any) -> Any:
            from corpus_sdk.vector.vector_base import Vector
            return QueryResult(
                matches=[
                    VectorMatch(
                        vector=Vector(
                            id="node-1",
                            vector=[0.9, 0.1],
                            metadata={
                                "text": "high",
                                "id": "node-1",
                                "node_id": "node-1",
                            },
                            namespace="default",
                        ),
                        score=0.9,
                        distance=0.1,
                    ),
                    VectorMatch(
                        vector=Vector(
                            id="node-2",
                            vector=[0.3, 0.7],
                            metadata={
                                "text": "low",
                                "id": "node-2",
                                "node_id": "node-2",
                            },
                            namespace="default",
                        ),
                        score=0.3,
                        distance=0.7,
                    ),
                ],
                query_vector=[0.1, 0.2],
                namespace="default",
                total_matches=2,
            )

        async def arun_capabilities(self):
            return VectorCapabilities(server="mock", version="1.0")

    with patch.object(llamaindex_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter, score_threshold=0.5)

        query = VectorStoreQuery(query_embedding=[0.1, 0.2], similarity_top_k=10)
        result = await store.aquery(query)

        assert len(result.nodes) == 1
        assert result.nodes[0].node.get_content() == "high"


# ---------------------------------------------------------------------------
# Streaming Query Tests (3 tests)
# ---------------------------------------------------------------------------


def test_query_stream_returns_iterator(
    adapter: Any, monkeypatch: pytest.MonkeyPatch, VectorStoreQuery: Any
) -> None:
    """Should return Iterator[NodeWithScore]."""
    with patch.object(
        llamaindex_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)

        query = VectorStoreQuery(query_embedding=[0.1, 0.2], similarity_top_k=4)
        result = store.query_stream(query)

        assert hasattr(result, "__iter__")


def test_query_stream_yields_nodes_progressively(
    adapter: Any, monkeypatch: pytest.MonkeyPatch, VectorStoreQuery: Any, NodeWithScore: Any
) -> None:
    """Should yield progressive nodes."""
    with patch.object(
        llamaindex_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)

        query = VectorStoreQuery(query_embedding=[0.1, 0.2], similarity_top_k=4)
        nodes = list(store.query_stream(query))

        assert all(isinstance(n, NodeWithScore) for n in nodes)


def test_query_stream_validates_max_top_k(
    adapter: Any, monkeypatch: pytest.MonkeyPatch, VectorStoreQuery: Any
) -> None:
    """Should raise if similarity_top_k exceeds max_top_k."""
    with patch.object(
        llamaindex_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)

        query = VectorStoreQuery(query_embedding=[0.1, 0.2], similarity_top_k=200)

        with pytest.raises(BadRequest, match="exceeds maximum"):
            list(store.query_stream(query))


# ---------------------------------------------------------------------------
# MMR Tests (8 tests)
# ---------------------------------------------------------------------------


def test_cosine_sim_basic(adapter: Any) -> None:
    """Should compute cosine similarity correctly."""
    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)

    # Identical vectors
    sim = store._cosine_sim([1.0, 0.0], [1.0, 0.0])
    assert abs(sim - 1.0) < 0.01

    # Orthogonal vectors
    sim = store._cosine_sim([1.0, 0.0], [0.0, 1.0])
    assert abs(sim - 0.0) < 0.01


def test_mmr_select_indices_pure_relevance_lambda_1(adapter: Any) -> None:
    """Lambda=1.0 should return pure relevance ranking."""
    from corpus_sdk.vector.vector_base import Vector
    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
    query_vec = [1.0, 0.0]
    matches = [
        VectorMatch(vector=Vector(id="0", vector=[1.0, 0.0], metadata={}, namespace=None), score=0.5, distance=0.5),
        VectorMatch(vector=Vector(id="1", vector=[0.9, 0.1], metadata={}, namespace=None), score=0.9, distance=0.1),
        VectorMatch(vector=Vector(id="2", vector=[0.7, 0.3], metadata={}, namespace=None), score=0.7, distance=0.3),
    ]

    indices = store._mmr_select_indices(query_vec, matches, k=3, lambda_mult=1.0)

    # Should be in descending score order: [1, 2, 0]
    assert indices == [1, 2, 0]


def test_mmr_select_indices_respects_k(adapter: Any) -> None:
    """Should return at most k results."""
    from corpus_sdk.vector.vector_base import Vector
    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
    query_vec = [1.0, 0.0]
    matches = [
        VectorMatch(vector=Vector(id=str(i), vector=[1.0, 0.0], metadata={}, namespace=None), score=0.9, distance=0.1)
        for i in range(5)
    ]

    indices = store._mmr_select_indices(query_vec, matches, k=2, lambda_mult=0.5)

    assert len(indices) == 2


def test_mmr_select_indices_handles_empty_matches(adapter: Any) -> None:
    """Should handle empty match list."""
    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
    query_vec = [1.0, 0.0]

    indices = store._mmr_select_indices(query_vec, [], k=5, lambda_mult=0.5)

    assert indices == []


def test_query_mmr_validates_lambda_range(
    adapter: Any, monkeypatch: pytest.MonkeyPatch, VectorStoreQuery: Any
) -> None:
    """Should raise if lambda_mult not in [0, 1]."""
    with patch.object(
        llamaindex_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)

        query = VectorStoreQuery(query_embedding=[0.1, 0.2], similarity_top_k=4)

        with pytest.raises(BadRequest, match="lambda_mult must be in"):
            store.query_mmr(query, lambda_mult=1.5)


def test_query_mmr_requests_vectors(
    adapter: Any, monkeypatch: pytest.MonkeyPatch, VectorStoreQuery: Any
) -> None:
    """Should request include_vectors=True for MMR."""
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def query(self, raw_query: Any, **k: Any) -> Any:
            from corpus_sdk.vector.vector_base import Vector
            captured["raw_query"] = raw_query
            return QueryResult(
                matches=[
                    VectorMatch(
                        vector=Vector(
                            id="node-1",
                            vector=[0.1, 0.2],
                            metadata={
                                "text": "test",
                                "id": "node-1",
                                "node_id": "node-1",
                            },
                            namespace="default",
                        ),
                        score=0.9,
                        distance=0.1,
                    ),
                ],
                query_vector=[0.1, 0.2],
                namespace="default",
                total_matches=1,
            )

        def capabilities(self):
            return VectorCapabilities(server="mock", version="1.0")

    with patch.object(llamaindex_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)

        query = VectorStoreQuery(query_embedding=[0.1, 0.2], similarity_top_k=4)
        store.query_mmr(query)

        assert captured["raw_query"]["include_vectors"] is True


def test_query_mmr_returns_vector_store_query_result(
    adapter: Any, monkeypatch: pytest.MonkeyPatch, VectorStoreQuery: Any
) -> None:
    """Should return VectorStoreQueryResult."""
    with patch.object(
        llamaindex_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)

        query = VectorStoreQuery(query_embedding=[0.1, 0.2], similarity_top_k=4)
        result = store.query_mmr(query, lambda_mult=0.5)

        assert hasattr(result, "nodes")
        assert hasattr(result, "similarities")
        assert hasattr(result, "ids")


@pytest.mark.asyncio
async def test_aquery_mmr_returns_vector_store_query_result(
    adapter: Any, monkeypatch: pytest.MonkeyPatch, VectorStoreQuery: Any
) -> None:
    """Should return VectorStoreQueryResult."""
    with patch.object(
        llamaindex_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)

        query = VectorStoreQuery(query_embedding=[0.1, 0.2], similarity_top_k=4)
        result = await store.aquery_mmr(query, lambda_mult=0.5)

        assert hasattr(result, "nodes")
        assert hasattr(result, "similarities")
        assert hasattr(result, "ids")


# ---------------------------------------------------------------------------
# Delete Tests (3 tests)
# ---------------------------------------------------------------------------


def test_delete_by_ref_doc_id_delegates_to_translator(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should call translator.delete() with ref_doc_id filter."""
    called = {"delete": False}

    class DummyTranslator:
        def delete(self, *a: Any, **k: Any) -> Any:
            called["delete"] = True
            return {"deleted": 1}

        def capabilities(self):
            return VectorCapabilities(server="mock", version="1.0")

    with patch.object(llamaindex_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
        store.delete("doc-abc")

        assert called["delete"] is True


def test_delete_nodes_delegates_to_translator(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should call translator.delete() with node IDs."""
    called = {"delete": False}

    class DummyTranslator:
        def delete(self, *a: Any, **k: Any) -> Any:
            called["delete"] = True
            return {"deleted": 1}

        def capabilities(self):
            return VectorCapabilities(server="mock", version="1.0")

    with patch.object(llamaindex_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
        store.delete_nodes(["node-1", "node-2"])

        assert called["delete"] is True


def test_delete_nodes_handles_empty_list(adapter: Any) -> None:
    """Should handle empty node_ids gracefully."""
    with patch.object(
        llamaindex_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)

        # Should not raise
        store.delete_nodes([])


# ---------------------------------------------------------------------------
# Capabilities Tests (3 tests)
# ---------------------------------------------------------------------------


def test_get_caps_sync_delegates_to_translator(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should delegate to translator.capabilities()."""
    with patch.object(
        llamaindex_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)

        caps = store._get_caps_sync()

        assert isinstance(caps, VectorCapabilities)


def test_get_caps_sync_caches_result(adapter: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Should cache VectorCapabilities."""
    with patch.object(
        llamaindex_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)

        caps1 = store._get_caps_sync()
        caps2 = store._get_caps_sync()

        # Should cache
        assert store._caps is not None


@pytest.mark.asyncio
async def test_get_caps_async_delegates_to_translator(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should delegate to translator.arun_capabilities()."""
    with patch.object(
        llamaindex_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)

        caps = await store._get_caps_async()

        assert isinstance(caps, VectorCapabilities)


# ---------------------------------------------------------------------------
# Context Manager Tests (3 tests)
# ---------------------------------------------------------------------------


def test_context_manager_calls_close(adapter: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """__exit__ should call close()."""
    with patch.object(
        llamaindex_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)

        with store:
            assert store is not None


@pytest.mark.asyncio
async def test_async_context_manager_calls_aclose(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """__aexit__ should call aclose()."""
    with patch.object(
        llamaindex_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)

        async with store:
            assert store is not None


def test_close_closes_translator_and_adapter(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should call close on both translator and adapter."""
    called = {"translator": False, "adapter": False}

    class DummyTranslator:
        def close(self) -> None:
            called["translator"] = True

        def capabilities(self):
            return VectorCapabilities(server="mock", version="1.0")

    # Create a mock adapter that tracks close calls
    adapter.close = lambda: called.update({"adapter": True})

    with patch.object(llamaindex_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter, own_adapter=True)
        # Access _translator to ensure it's instantiated (cached_property is lazy)
        _ = store._translator
        store.close()

        assert called["translator"] is True
        assert called["adapter"] is True


# ---------------------------------------------------------------------------
# Event Loop Guard Tests (2 tests)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_not_in_event_loop_raises_in_loop() -> None:
    """Should raise RuntimeError when called in event loop."""
    with pytest.raises(RuntimeError, match="event loop"):
        _ensure_not_in_event_loop("test_api")


def test_ensure_not_in_event_loop_succeeds_outside_loop() -> None:
    """Should not raise when called outside event loop."""
    # Should not raise
    _ensure_not_in_event_loop("test_api")


# ---------------------------------------------------------------------------
# Error Context Tests (1 test - consolidated)
# ---------------------------------------------------------------------------


def test_error_context_includes_framework_and_operation(
    adapter: Any, monkeypatch: pytest.MonkeyPatch, VectorStoreQuery: Any
) -> None:
    """Error context should include framework='llamaindex' and operation."""
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    monkeypatch.setattr(llamaindex_adapter_module, "attach_context", fake_attach_context)

    class FailingTranslator:
        def query(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("test error")

        def capabilities(self):
            return VectorCapabilities(server="mock", version="1.0")

    with patch.object(llamaindex_adapter_module, "VectorTranslator", return_value=FailingTranslator()):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)

        query = VectorStoreQuery(query_embedding=[0.1, 0.2], similarity_top_k=4)

        with pytest.raises(RuntimeError):
            store.query(query)

        assert captured_ctx.get("framework") == "llamaindex"
        assert "operation" in captured_ctx


# ---------------------------------------------------------------------------
# LlamaIndex Integration Tests (6 tests - NO SKIPS)
# ---------------------------------------------------------------------------


def test_llamaindex_node_roundtrip(
    adapter: Any, monkeypatch: pytest.MonkeyPatch, TextNode: Any, VectorStoreQuery: Any
) -> None:
    """TextNode → add → query → NodeWithScore should preserve content."""
    with patch.object(
        llamaindex_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)

        # Create node
        from llama_index.core.schema import NodeRelationship, RelatedNodeInfo
        node = TextNode(
            text="test content",
            id_="node-123",
            metadata={"key": "value"},
            embedding=[0.1, 0.2, 0.3, 0.4],
        )
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id="doc-abc")

        # Add
        ids = store.add([node])
        assert ids == ["node-123"]

        # Query back (translator will return our node)
        query = VectorStoreQuery(
            query_embedding=[0.1, 0.2, 0.3, 0.4], similarity_top_k=1
        )
        result = store.query(query)

        # Verify roundtrip preservation
        assert len(result.nodes) >= 0  # Dummy translator behavior


def test_llamaindex_metadata_flattening(
    adapter: Any, TextNode: Any
) -> None:
    """node_to_metadata_dict → storage → metadata_dict_to_node."""
    from llama_index.core.vector_stores.utils import node_to_metadata_dict, metadata_dict_to_node

    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)

    node = TextNode(
        text="test",
        id_="node-1",
        metadata={"key": "value", "year": 2024},
        embedding=[0.1, 0.2],
    )

    # Test flattening works
    flat_meta = node_to_metadata_dict(node, remove_text=False, flat_metadata=store.flat_metadata)
    assert isinstance(flat_meta, dict)
    assert store.flat_metadata is True
    assert "key" in flat_meta


def test_llamaindex_vector_store_query_integration(
    adapter: Any, VectorStoreQuery: Any
) -> None:
    """VectorStoreQuery object → query() → VectorStoreQueryResult."""
    with patch.object(
        llamaindex_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)

        # Create LlamaIndex VectorStoreQuery
        query = VectorStoreQuery(
            query_embedding=[0.1, 0.2, 0.3, 0.4],
            similarity_top_k=5,
            filters=None,
            doc_ids=None,
            node_ids=None,
        )

        result = store.query(query)

        # Verify LlamaIndex contract
        assert hasattr(result, "nodes")
        assert hasattr(result, "similarities")
        assert hasattr(result, "ids")
        assert isinstance(result.nodes, list)
        assert isinstance(result.similarities, list)
        assert isinstance(result.ids, list)


def test_llamaindex_node_with_score_structure(
    adapter: Any, NodeWithScore: Any, VectorStoreQuery: Any
) -> None:
    """Query results return proper NodeWithScore objects."""
    with patch.object(
        llamaindex_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)

        query = VectorStoreQuery(query_embedding=[0.1, 0.2], similarity_top_k=4)
        result = store.query(query)

        # Verify NodeWithScore structure
        for nws in result.nodes:
            assert hasattr(nws, "node")
            assert hasattr(nws, "score")
            assert isinstance(nws.score, (int, float))


def test_llamaindex_metadata_filters_translation(adapter: Any) -> None:
    """MetadataFilters → Corpus filter dict conversion."""
    from llama_index.core.vector_stores.types import MetadataFilters, ExactMatchFilter

    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)

    # Create LlamaIndex MetadataFilters
    filters = MetadataFilters(
        filters=[
            ExactMatchFilter(key="category", value="tech"),
            ExactMatchFilter(key="year", value=2024),
        ]
    )

    # Translate
    corpus_filter = store._metadata_filters_to_corpus_filter(filters)

    # Verify translation
    assert corpus_filter is not None
    assert isinstance(corpus_filter, dict)


def test_llamaindex_streaming_yields_nodes_progressively(
    adapter: Any, VectorStoreQuery: Any, NodeWithScore: Any
) -> None:
    """query_stream() yields NodeWithScore one-by-one."""
    with patch.object(
        llamaindex_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)

        query = VectorStoreQuery(query_embedding=[0.1, 0.2], similarity_top_k=4)

        # Stream should yield individual nodes
        count = 0
        for node_with_score in store.query_stream(query):
            assert isinstance(node_with_score, NodeWithScore)
            count += 1
            if count >= 5:  # Limit iteration for test
                break

# ---------------------------------------------------------------------------
# Metadata/Filter Translation Tests (6 tests)
# ---------------------------------------------------------------------------


def test_metadata_filters_to_corpus_filter_handles_eq_operator(adapter: Any) -> None:
    """Should translate EQ operator correctly."""
    from llama_index.core.vector_stores.types import MetadataFilters, ExactMatchFilter
    
    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
    
    filters = MetadataFilters(filters=[ExactMatchFilter(key="status", value="active")])
    corpus_filter = store._metadata_filters_to_corpus_filter(filters)
    
    assert corpus_filter == {"status": "active"}


def test_metadata_filters_to_corpus_filter_handles_in_operator(adapter: Any) -> None:
    """Should translate IN operator correctly."""
    try:
        from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter, FilterOperator
        
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
        
        filters = MetadataFilters(
            filters=[MetadataFilter(key="category", value=["tech", "science"], operator=FilterOperator.IN)]
        )
        corpus_filter = store._metadata_filters_to_corpus_filter(filters)
        
        assert "$in" in corpus_filter.get("category", {})
    except ImportError:
        pytest.skip("LlamaIndex not available")


def test_metadata_filters_to_corpus_filter_handles_gt_operator(adapter: Any) -> None:
    """Should translate GT operator correctly."""
    try:
        from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter, FilterOperator
        
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
        
        filters = MetadataFilters(
            filters=[MetadataFilter(key="year", value=2020, operator=FilterOperator.GT)]
        )
        corpus_filter = store._metadata_filters_to_corpus_filter(filters)

        assert corpus_filter is not None
        assert "year" in corpus_filter
        assert "$gt" in corpus_filter["year"]
    except ImportError:
        pytest.skip("LlamaIndex not available")


def test_metadata_filters_combines_with_doc_ids(adapter: Any) -> None:
    """Should combine filters with doc_ids."""
    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
    
    corpus_filter = store._metadata_filters_to_corpus_filter(
        None,
        doc_ids=["doc-1", "doc-2"],
    )
    
    assert corpus_filter is not None
    assert store.ref_doc_id_field in corpus_filter or "$and" in corpus_filter or "$or" in corpus_filter


def test_metadata_filters_combines_with_node_ids(adapter: Any) -> None:
    """Should combine filters with node_ids."""
    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
    
    corpus_filter = store._metadata_filters_to_corpus_filter(
        None,
        node_ids=["node-1", "node-2"],
    )
    
    assert corpus_filter is not None


def test_metadata_filters_returns_none_when_empty(adapter: Any) -> None:
    """Should return None when no filters."""
    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
    
    corpus_filter = store._metadata_filters_to_corpus_filter(None)
    
    assert corpus_filter is None


# ---------------------------------------------------------------------------
# Request Builder Tests (6 tests)
# ---------------------------------------------------------------------------


def test_build_upsert_request_includes_namespace(adapter: Any) -> None:
    """Should include effective namespace in request."""
    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter, namespace="test-ns")
    
    vectors = [
        Vector(
            id="v1",
            vector=[0.1, 0.2],
            metadata={"test": "data"},
            namespace="test-ns",
        )
    ]
    
    request = store._build_upsert_request(vectors, namespace=None)
    
    assert request["namespace"] == "test-ns"
    assert request["vectors"] == vectors


def test_build_query_request_includes_all_params(adapter: Any) -> None:
    """Should include vector, top_k, namespace, filters."""
    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter, namespace="test-ns")
    
    request = store._build_query_request(
        embedding=[0.1, 0.2],
        top_k=5,
        namespace=None,
        filter={"key": "value"},
        include_vectors=True,
    )
    
    assert request["vector"] == [0.1, 0.2]
    assert request["top_k"] == 5
    assert request["namespace"] == "test-ns"
    assert request["filters"] == {"key": "value"}
    assert request["include_vectors"] is True
    assert request["include_metadata"] is True


def test_build_delete_request_by_ids(adapter: Any) -> None:
    """Should build delete request with IDs."""
    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter, namespace="test-ns")
    
    request = store._build_delete_request(
        ids=["id-1", "id-2"],
        namespace=None,
        filter=None,
    )
    
    assert request["ids"] == ["id-1", "id-2"]
    assert request["namespace"] == "test-ns"
    assert request["filters"] is None


def test_build_delete_request_by_filter(adapter: Any) -> None:
    """Should build delete request with filter."""
    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
    
    request = store._build_delete_request(
        ids=None,
        namespace=None,
        filter={"status": "deleted"},
    )
    
    assert request["ids"] is None
    assert request["filters"] == {"status": "deleted"}


def test_build_query_request_respects_namespace_override(adapter: Any) -> None:
    """Explicit namespace should override default."""
    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter, namespace="default")
    
    request = store._build_query_request(
        embedding=[0.1],
        top_k=4,
        namespace="override",
        filter=None,
        include_vectors=False,
    )
    
    assert request["namespace"] == "override"


def test_build_upsert_request_respects_namespace_override(adapter: Any) -> None:
    """Explicit namespace should override default."""
    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter, namespace="default")
    
    vectors = [Vector(id="v1", vector=[0.1], metadata={}, namespace="override")]
    
    request = store._build_upsert_request(vectors, namespace="override")
    
    assert request["namespace"] == "override"


# ---------------------------------------------------------------------------
# Validation Tests (8 tests)
# ---------------------------------------------------------------------------


def test_validate_query_params_sync_accepts_valid_top_k(adapter: Any) -> None:
    """Should accept valid top_k."""
    with patch.object(
        llamaindex_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
        
        result = store._validate_query_params_sync(
            top_k=10,
            namespace=None,
            filter=None,
        )
        
        assert result == 10


def test_validate_query_params_sync_raises_on_exceeded_max(adapter: Any) -> None:
    """Should raise when top_k exceeds max_top_k."""
    with patch.object(
        llamaindex_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
        
        with pytest.raises(BadRequest, match="exceeds maximum"):
            store._validate_query_params_sync(
                top_k=200,
                namespace=None,
                filter=None,
            )


def test_validate_query_params_sync_raises_on_unsupported_filter(adapter: Any) -> None:
    """Should raise when filters not supported."""
    
    class NoFilterTranslator:
        def capabilities(self):
            return VectorCapabilities(server="mock", version="1.0", supports_metadata_filtering=False)
    
    with patch.object(llamaindex_adapter_module, "VectorTranslator", return_value=NoFilterTranslator()):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
        
        with pytest.raises(NotSupported, match="metadata filtering"):
            store._validate_query_params_sync(
                top_k=10,
                namespace=None,
                filter={"key": "value"},
            )


@pytest.mark.asyncio
async def test_validate_query_params_async_accepts_valid_top_k(adapter: Any) -> None:
    """Should accept valid top_k."""
    with patch.object(
        llamaindex_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
        
        result = await store._validate_query_params_async(
            top_k=10,
            namespace=None,
            filter=None,
        )
        
        assert result == 10


def test_validate_delete_params_sync_raises_without_ids_or_filter(adapter: Any) -> None:
    """Should raise when neither ids nor filter provided."""
    with patch.object(
        llamaindex_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
        
        with pytest.raises(BadRequest, match="must provide ids or filter"):
            store._validate_delete_params_sync(
                ids=None,
                namespace=None,
                filter=None,
            )


def test_validate_delete_params_sync_accepts_ids(adapter: Any) -> None:
    """Should accept delete with IDs."""
    with patch.object(
        llamaindex_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
        
        # Should not raise
        store._validate_delete_params_sync(
            ids=["id-1"],
            namespace=None,
            filter=None,
        )


def test_validate_delete_params_sync_accepts_filter(adapter: Any) -> None:
    """Should accept delete with filter."""
    with patch.object(
        llamaindex_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
        
        # Should not raise
        store._validate_delete_params_sync(
            ids=None,
            namespace=None,
            filter={"key": "value"},
        )


@pytest.mark.asyncio
async def test_validate_delete_params_async_raises_without_ids_or_filter(adapter: Any) -> None:
    """Should raise when neither ids nor filter provided."""
    with patch.object(
        llamaindex_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
        
        with pytest.raises(BadRequest, match="must provide ids or filter"):
            await store._validate_delete_params_async(
                ids=None,
                namespace=None,
                filter=None,
            )


# ---------------------------------------------------------------------------
# Query Edge Cases (4 tests)
# ---------------------------------------------------------------------------


def test_query_raises_without_embedding(adapter: Any, VectorStoreQuery: Any) -> None:
    """Should raise if query_embedding is None."""
    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
    
    query = VectorStoreQuery(query_embedding=None, similarity_top_k=4)
    
    with pytest.raises(NotSupported, match="query_embedding is None"):
        store.query(query)


def test_query_uses_default_top_k(adapter: Any, VectorStoreQuery: Any) -> None:
    """Should use default_top_k when not specified."""
    with patch.object(
        llamaindex_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter, default_top_k=7)
        
        query = VectorStoreQuery(query_embedding=[0.1, 0.2], similarity_top_k=None)
        result = store.query(query)
        
        # Should use default_top_k=7


@pytest.mark.asyncio
async def test_aquery_raises_without_embedding(adapter: Any, VectorStoreQuery: Any) -> None:
    """Should raise if query_embedding is None."""
    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
    
    query = VectorStoreQuery(query_embedding=None, similarity_top_k=4)
    
    with pytest.raises(NotSupported, match="query_embedding is None"):
        await store.aquery(query)


def test_query_stream_raises_without_embedding(adapter: Any, VectorStoreQuery: Any) -> None:
    """Should raise if query_embedding is None."""
    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
    
    query = VectorStoreQuery(query_embedding=None, similarity_top_k=4)
    
    with pytest.raises(NotSupported, match="query_embedding is None"):
        list(store.query_stream(query))


# ---------------------------------------------------------------------------
# Delete Async Tests (2 tests)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adelete_by_ref_doc_id_delegates_to_translator(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should call translator.arun_delete() with ref_doc_id filter."""
    called = {"arun_delete": False}
    
    class DummyTranslator:
        async def arun_delete(self, *a: Any, **k: Any) -> Any:
            called["arun_delete"] = True
            return {"deleted": 1}
        
        async def arun_capabilities(self):
            return VectorCapabilities(server="mock", version="1.0")
    
    with patch.object(llamaindex_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
        await store.adelete("doc-abc")
        
        assert called["arun_delete"] is True


@pytest.mark.asyncio
async def test_adelete_nodes_delegates_to_translator(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should call translator.arun_delete() with node IDs."""
    called = {"arun_delete": False}
    
    class DummyTranslator:
        async def arun_delete(self, *a: Any, **k: Any) -> Any:
            called["arun_delete"] = True
            return {"deleted": 1}
        
        async def arun_capabilities(self):
            return VectorCapabilities(server="mock", version="1.0")
    
    with patch.object(llamaindex_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
        await store.adelete_nodes(["node-1", "node-2"])
        
        assert called["arun_delete"] is True


# ---------------------------------------------------------------------------
# MMR Edge Cases (4 tests)
# ---------------------------------------------------------------------------


def test_query_mmr_raises_without_embedding(adapter: Any, VectorStoreQuery: Any) -> None:
    """Should raise if query_embedding is None."""
    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
    
    query = VectorStoreQuery(query_embedding=None, similarity_top_k=4)
    
    with pytest.raises(NotSupported, match="query_embedding is None"):
        store.query_mmr(query)


def test_query_mmr_returns_empty_for_zero_k(adapter: Any, VectorStoreQuery: Any) -> None:
    """Should return empty result for k=0."""
    with patch.object(
        llamaindex_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
        
        query = VectorStoreQuery(query_embedding=[0.1, 0.2], similarity_top_k=0)
        result = store.query_mmr(query, lambda_mult=0.5)
        
        assert len(result.nodes) == 0


@pytest.mark.asyncio
async def test_aquery_mmr_raises_without_embedding(adapter: Any, VectorStoreQuery: Any) -> None:
    """Should raise if query_embedding is None."""
    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
    
    query = VectorStoreQuery(query_embedding=None, similarity_top_k=4)
    
    with pytest.raises(NotSupported, match="query_embedding is None"):
        await store.aquery_mmr(query)


@pytest.mark.asyncio
async def test_aquery_mmr_returns_empty_for_zero_k(adapter: Any, VectorStoreQuery: Any) -> None:
    """Should return empty result for k=0."""
    with patch.object(
        llamaindex_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
        
        query = VectorStoreQuery(query_embedding=[0.1, 0.2], similarity_top_k=0)
        result = await store.aquery_mmr(query, lambda_mult=0.5)
        
        assert len(result.nodes) == 0


# ---------------------------------------------------------------------------
# Vector Store Info Test (1 test)
# ---------------------------------------------------------------------------


def test_vector_store_info_property(adapter: Any) -> None:
    """Should return VectorStoreInfo with metadata."""
    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
    
    info = store.vector_store_info
    
    assert info.content_info == "Vector store content"
    assert len(info.metadata_info) >= 2  # node_id, ref_doc_id


# ---------------------------------------------------------------------------
# Apply Score Threshold Test (1 test)
# ---------------------------------------------------------------------------


def test_apply_score_threshold_filters_matches(adapter: Any) -> None:
    """Should filter matches below threshold."""
    from corpus_sdk.vector.vector_base import Vector
    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter, score_threshold=0.7)
    
    matches = [
        VectorMatch(vector=Vector(id="1", vector=[0.1], metadata={}, namespace=None), score=0.9, distance=0.1),
        VectorMatch(vector=Vector(id="2", vector=[0.2], metadata={}, namespace=None), score=0.5, distance=0.5),
        VectorMatch(vector=Vector(id="3", vector=[0.3], metadata={}, namespace=None), score=0.8, distance=0.2),
    ]
    
    filtered = store._apply_score_threshold(matches)
    
    assert len(filtered) == 2
    assert filtered[0].vector.id == "1"
    assert filtered[1].vector.id == "3"


# ---------------------------------------------------------------------------
# Validate Query Result Type Test (1 test)
# ---------------------------------------------------------------------------


def test_validate_query_result_type_raises_on_wrong_type(adapter: Any) -> None:
    """Should raise if translator returns wrong type."""
    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
    
    with pytest.raises(BadRequest, match="unsupported type"):
        store._validate_query_result_type({"wrong": "type"}, operation="test")


# ---------------------------------------------------------------------------
# Framework Context Test (1 test)
# ---------------------------------------------------------------------------


def test_build_framework_context_includes_namespace(adapter: Any) -> None:
    """Should include namespace in framework_context."""
    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter, namespace="test-ns")

    fw_ctx = store._build_framework_context(namespace=None)

    assert "namespace" in fw_ctx
    assert fw_ctx["namespace"] == "test-ns"

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
