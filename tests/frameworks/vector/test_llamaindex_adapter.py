# tests/frameworks/vector/test_llamaindex_adapter.py

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, Set
import inspect
import pytest
from unittest.mock import Mock, AsyncMock, patch, call

import corpus_sdk.vector.framework_adapters.llamaindex as llamaindex_adapter_module
from corpus_sdk.vector.framework_adapters.llamaindex import (
    CorpusLlamaIndexVectorStore,
    with_error_context,
    with_async_error_context,
)


# ---------------------------------------------------------------------------
# Test Fixtures and Helpers
# ---------------------------------------------------------------------------

class DummyVectorStoreQuery:
    """Minimal stand-in for llama_index.core.vector_stores.types.VectorStoreQuery."""
    def __init__(
        self,
        query_embedding: Sequence[float] | None = None,
        similarity_top_k: int | None = None,
        filters: Any | None = None,
        doc_ids: Sequence[str] | None = None,
        node_ids: Sequence[str] | None = None,
    ) -> None:
        self.query_embedding = query_embedding
        self.similarity_top_k = similarity_top_k
        self.filters = filters
        self.doc_ids = list(doc_ids) if doc_ids is not None else None
        self.node_ids = list(node_ids) if node_ids is not None else None


class DummyMetadataFilter:
    """Minimal stand-in for llama_index.core.vector_stores.types.MetadataFilter."""
    def __init__(
        self,
        key: str,
        value: Any,
        operator: str = "==",
    ) -> None:
        self.key = key
        self.value = value
        self.operator = operator


class DummyMetadataFilters:
    """Minimal stand-in for llama_index.core.vector_stores.types.MetadataFilters."""
    def __init__(
        self,
        filters: list[DummyMetadataFilter] | None = None,
        condition: str = "AND",
    ) -> None:
        self.filters = filters or []
        self.condition = condition


def _make_store(adapter: Any = None, **kwargs: Any) -> CorpusLlamaIndexVectorStore:
    """Construct a CorpusLlamaIndexVectorStore instance."""
    if adapter is None:
        adapter = Mock()
        adapter.capabilities = Mock(return_value=_make_dummy_caps())
    return CorpusLlamaIndexVectorStore(corpus_adapter=adapter, **kwargs)


def _make_dummy_caps(**overrides: Any) -> Any:
    """Create dummy VectorCapabilities."""
    caps = Mock()
    caps.max_top_k = 1000
    caps.supports_metadata_filtering = True
    caps.supports_namespaces = True
    caps.max_batch_size = 1000
    for key, value in overrides.items():
        setattr(caps, key, value)
    return caps


def _make_dummy_node(node_id: str = "test-node", ref_doc_id: Optional[str] = "doc-123") -> Any:
    """Create a dummy LlamaIndex node."""
    node = Mock()
    node.node_id = node_id
    node.ref_doc_id = ref_doc_id
    node.get_embedding = Mock(return_value=[0.1, 0.2, 0.3])
    node.get_content = Mock(return_value="Test content")
    node.metadata = {"topic": "science"}
    return node


# ---------------------------------------------------------------------------
# Test Suite: 40 Comprehensive Tests
# ---------------------------------------------------------------------------

# 1. Framework-specific metadata field handling
def test_llamaindex_reserved_metadata_fields_are_unique_and_configurable():
    """Test that reserved metadata fields are configurable and unique."""
    store = _make_store()
    reserved_fields = {
        store.id_field,
        store.text_field,
        store.node_id_field,
        store.ref_doc_id_field,
    }
    assert len(reserved_fields) == 4, "Reserved metadata fields must be unique"
    
    store_custom = _make_store(
        id_field="doc_id",
        text_field="content",
        node_id_field="llama_node_id",
        ref_doc_id_field="ref_doc",
    )
    assert store_custom.id_field == "doc_id"
    assert store_custom.text_field == "content"
    assert store_custom.node_id_field == "llama_node_id"
    assert store_custom.ref_doc_id_field == "ref_doc"


def test_vector_store_info_includes_llamaindex_metadata_fields():
    """Test that VectorStoreInfo includes LlamaIndex-specific metadata fields."""
    store = _make_store()
    info = store.vector_store_info
    
    assert info.name == "corpus"
    assert "Corpus VectorProtocol" in info.description
    
    field_names = {mi.name for mi in info.metadata_info}
    assert store.node_id_field in field_names
    assert store.ref_doc_id_field in field_names


def test_reserved_fields_conflict_validation():
    """Test validation fails when reserved fields are not unique."""
    with pytest.raises(ValueError) as excinfo:
        _make_store(
            id_field="same",
            text_field="same",
            node_id_field="node_id",
            ref_doc_id_field="ref_doc_id",
        )
    assert "must be unique" in str(excinfo.value)


# 2. Node translation and reconstruction
def test_node_translation_preserves_llamaindex_specific_fields():
    """Test that node translation preserves LlamaIndex-specific fields."""
    store = _make_store()
    node = _make_dummy_node()
    
    vectors = store._nodes_to_corpus_vectors([node], namespace="test-ns")
    
    assert len(vectors) == 1
    v = vectors[0]
    meta = v.metadata
    
    assert meta[store.node_id_field] == "test-node"
    assert meta[store.ref_doc_id_field] == "doc-123"
    assert meta[store.text_field] == "Test content"
    assert meta[store.id_field] == "test-node"
    assert meta["topic"] == "science"


def test_node_translation_handles_missing_ref_doc_id():
    """Test node translation when ref_doc_id is None."""
    store = _make_store()
    node = _make_dummy_node(ref_doc_id=None)
    
    vectors = store._nodes_to_corpus_vectors([node], namespace="test-ns")
    meta = vectors[0].metadata
    
    assert store.ref_doc_id_field not in meta


def test_node_translation_requires_embedding():
    """Test that node translation requires embeddings."""
    store = _make_store()
    node = Mock()
    node.node_id = "test-node"
    node.get_embedding = Mock(return_value=None)
    node.embedding = None
    
    with pytest.raises(Exception) as excinfo:
        store._nodes_to_corpus_vectors([node], namespace="test-ns")
    assert "NO_EMBEDDING" in str(excinfo.value.code) or "embedding" in str(excinfo.value)


def test_node_reconstruction_uses_llamaindex_utilities():
    """Test that node reconstruction uses LlamaIndex utilities."""
    store = _make_store()
    
    with patch('corpus_sdk.vector.framework_adapters.llamaindex.metadata_dict_to_node') as mock_modern:
        with patch('corpus_sdk.vector.framework_adapters.llamaindex.legacy_metadata_dict_to_node') as mock_legacy:
            mock_modern.return_value = Mock()
            
            vector_match = Mock()
            vector_match.vector = Mock()
            vector_match.vector.id = "vec-123"
            vector_match.vector.metadata = {
                store.node_id_field: "node-123",
                store.text_field: "Test text",
                store.id_field: "vec-123",
            }
            vector_match.score = 0.9
            
            store._matches_to_nodes([vector_match])
            
            mock_modern.assert_called_once()
            mock_legacy.assert_not_called()


def test_node_reconstruction_falls_back_to_legacy():
    """Test that node reconstruction falls back to legacy utility."""
    store = _make_store()
    
    with patch('corpus_sdk.vector.framework_adapters.llamaindex.metadata_dict_to_node', side_effect=Exception("modern failed")):
        with patch('corpus_sdk.vector.framework_adapters.llamaindex.legacy_metadata_dict_to_node') as mock_legacy:
            mock_legacy.return_value = Mock()
            
            vector_match = Mock()
            vector_match.vector = Mock()
            vector_match.vector.id = "vec-123"
            vector_match.vector.metadata = {
                store.node_id_field: "node-123",
                store.text_field: "Test text",
                store.id_field: "vec-123",
            }
            vector_match.score = 0.9
            
            store._matches_to_nodes([vector_match])
            
            mock_legacy.assert_called_once()


def test_node_reconstruction_falls_back_to_textnode():
    """Test that node reconstruction falls back to TextNode when all else fails."""
    store = _make_store()
    
    with patch('corpus_sdk.vector.framework_adapters.llamaindex.metadata_dict_to_node', side_effect=Exception("modern failed")):
        with patch('corpus_sdk.vector.framework_adapters.llamaindex.legacy_metadata_dict_to_node', side_effect=Exception("legacy failed")):
            
            vector_match = Mock()
            vector_match.vector = Mock()
            vector_match.vector.id = "vec-123"
            vector_match.vector.metadata = {
                store.node_id_field: "node-123",
                store.text_field: "Test text",
                store.id_field: "vec-123",
            }
            vector_match.score = 0.9
            
            nodes = store._matches_to_nodes([vector_match])
            
            assert len(nodes) == 1
            assert nodes[0].node.id_ == "node-123"
            assert nodes[0].node.text == "Test text"


# 3. Metadata filter translation
def test_metadata_filters_translation_supports_llamaindex_operators():
    """Test metadata filter operator translation."""
    store = _make_store()
    
    # Test EQ operator
    filters = DummyMetadataFilters([
        DummyMetadataFilter("topic", "science", "==")
    ])
    result = store._metadata_filters_to_corpus_filter(filters)
    assert result == {"topic": "science"}
    
    # Test NE operator
    filters = DummyMetadataFilters([
        DummyMetadataFilter("status", "archived", "!=")
    ])
    result = store._metadata_filters_to_corpus_filter(filters)
    assert result == {"status": {"$ne": "archived"}}
    
    # Test IN operator
    filters = DummyMetadataFilters([
        DummyMetadataFilter("category", ["a", "b", "c"], "in")
    ])
    result = store._metadata_filters_to_corpus_filter(filters)
    assert result == {"category": {"$in": ["a", "b", "c"]}}
    
    # Test GT operator
    filters = DummyMetadataFilters([
        DummyMetadataFilter("year", 2020, ">")
    ])
    result = store._metadata_filters_to_corpus_filter(filters)
    assert result == {"year": {"$gt": 2020}}


def test_metadata_filters_with_doc_ids_and_node_ids():
    """Test integration of doc_ids and node_ids into metadata filters."""
    store = _make_store()
    
    # Test with doc_ids only
    result = store._metadata_filters_to_corpus_filter(
        None,
        doc_ids=["doc-1", "doc-2"],
        node_ids=None,
    )
    assert result == {store.ref_doc_id_field: {"$in": ["doc-1", "doc-2"]}}
    
    # Test with node_ids only
    result = store._metadata_filters_to_corpus_filter(
        None,
        doc_ids=None,
        node_ids=["node-1", "node-2"],
    )
    assert result == {store.node_id_field: {"$in": ["node-1", "node-2"]}}


def test_metadata_filter_condition_handling():
    """Test AND/OR condition handling in metadata filters."""
    store = _make_store()
    
    # Test AND condition (default)
    filters = DummyMetadataFilters([
        DummyMetadataFilter("topic", "science", "=="),
        DummyMetadataFilter("year", 2023, ">="),
    ], condition="AND")
    result = store._metadata_filters_to_corpus_filter(filters)
    assert result == {
        "$and": [
            {"topic": "science"},
            {"year": {"$gte": 2023}}
        ]
    }
    
    # Test OR condition
    filters = DummyMetadataFilters([
        DummyMetadataFilter("topic", "science", "=="),
        DummyMetadataFilter("topic", "technology", "=="),
    ], condition="OR")
    result = store._metadata_filters_to_corpus_filter(filters)
    assert result == {
        "$or": [
            {"topic": "science"},
            {"topic": "technology"}
        ]
    }


def test_metadata_filter_skips_unsupported_operators():
    """Test that unsupported operators are skipped with warning."""
    store = _make_store()
    
    with patch.object(store, 'logger') as mock_logger:
        filters = DummyMetadataFilters([
            DummyMetadataFilter("field", "value", "UNSUPPORTED")
        ])
        result = store._metadata_filters_to_corpus_filter(filters)
        
        # Should skip unsupported operator
        assert result is None
        mock_logger.debug.assert_called()


# 4. Query validation and execution
def test_query_requires_llamaindex_embedding():
    """Test that query requires VectorStoreQuery.query_embedding."""
    store = _make_store()
    query = DummyVectorStoreQuery(query_embedding=None)
    
    with pytest.raises(Exception) as excinfo:
        store.query(query)
    assert "NO_QUERY_EMBEDDING" in str(excinfo.value.code)


def test_query_uses_llamaindex_similarity_top_k():
    """Test that query respects similarity_top_k or uses default."""
    store = _make_store(default_top_k=10)
    
    # Mock translator to capture parameters
    mock_translator = Mock()
    mock_translator.capabilities.return_value = _make_dummy_caps()
    mock_translator.query.return_value = Mock(matches=[])
    store._translator = mock_translator
    
    # Test explicit similarity_top_k
    query = DummyVectorStoreQuery(
        query_embedding=[0.1, 0.2, 0.3],
        similarity_top_k=5,
    )
    store.query(query)
    
    # Check top_k was passed correctly
    call_args = mock_translator.query.call_args[0][0]
    assert call_args["top_k"] == 5


def test_query_validates_against_capabilities():
    """Test that query validates against backend capabilities."""
    store = _make_store()
    
    # Mock capabilities with low max_top_k
    mock_caps = _make_dummy_caps(max_top_k=10)
    store._get_caps_sync = Mock(return_value=mock_caps)
    
    query = DummyVectorStoreQuery(
        query_embedding=[0.1, 0.2, 0.3],
        similarity_top_k=20,
    )
    
    with pytest.raises(Exception) as excinfo:
        store.query(query)
    assert "BAD_TOP_K" in str(excinfo.value.code)


def test_query_with_unsupported_filtering():
    """Test query fails when backend doesn't support filtering."""
    store = _make_store()
    
    # Mock capabilities without filtering support
    mock_caps = _make_dummy_caps(supports_metadata_filtering=False)
    store._get_caps_sync = Mock(return_value=mock_caps)
    
    filters = DummyMetadataFilters([
        DummyMetadataFilter("topic", "science", "==")
    ])
    query = DummyVectorStoreQuery(
        query_embedding=[0.1, 0.2, 0.3],
        similarity_top_k=5,
        filters=filters,
    )
    
    with pytest.raises(Exception) as excinfo:
        store.query(query)
    assert "FILTER_NOT_SUPPORTED" in str(excinfo.value.code)


# 5. Error context and framework labeling
def test_error_context_includes_llamaindex_framework_label():
    """Test that errors include framework="llamaindex" in context."""
    captured_context = []
    
    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_context.append(ctx)
    
    with patch('corpus_sdk.vector.framework_adapters.llamaindex.attach_context', fake_attach_context):
        store = _make_store()
        query = DummyVectorStoreQuery(query_embedding=None)
        
        try:
            store.query(query)
        except Exception:
            pass
        
        assert captured_context
        assert any(ctx.get("framework") == "llamaindex" for ctx in captured_context)


def test_operation_names_are_llamaindex_specific():
    """Test that operation names are specific to LlamaIndex."""
    captured_operations = []
    
    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        if "operation" in ctx:
            captured_operations.append(ctx["operation"])
    
    with patch('corpus_sdk.vector.framework_adapters.llamaindex.attach_context', fake_attach_context):
        store = _make_store()
        
        # Trigger query error
        query = DummyVectorStoreQuery(query_embedding=None)
        try:
            store.query(query)
        except Exception:
            pass
        
        # Check for LlamaIndex-specific operation names
        llama_ops = {"query_sync", "add_sync", "delete_sync"}
        assert any(op in captured_operations for op in llama_ops)


def test_error_context_decorators_work():
    """Test that error context decorators properly wrap methods."""
    # Test sync decorator
    @with_error_context("test_operation", test_key="test_value")
    def failing_func():
        raise ValueError("Test error")
    
    with patch('corpus_sdk.vector.framework_adapters.llamaindex.attach_context') as mock_attach:
        try:
            failing_func()
        except ValueError:
            pass
        
        mock_attach.assert_called_once()
        call_kwargs = mock_attach.call_args[1]
        assert call_kwargs["operation"] == "test_operation"
        assert call_kwargs["test_key"] == "test_value"
        assert call_kwargs["framework"] == "llamaindex"


# 6. Async API parity
@pytest.mark.asyncio
async def test_async_methods_exist_and_return_correct_types():
    """Test that all required async methods exist."""
    store = _make_store()
    
    assert hasattr(store, "aadd")
    assert hasattr(store, "adelete")
    assert hasattr(store, "adelete_nodes")
    assert hasattr(store, "aquery")
    assert hasattr(store, "aquery_mmr")
    
    assert inspect.iscoroutinefunction(store.aadd)
    assert inspect.iscoroutinefunction(store.adelete)
    assert inspect.iscoroutinefunction(store.adelete_nodes)
    assert inspect.iscoroutinefunction(store.aquery)
    assert inspect.iscoroutinefunction(store.aquery_mmr)


@pytest.mark.asyncio
async def test_async_query_requires_llamaindex_embedding():
    """Test that aquery() requires query embedding."""
    store = _make_store()
    query = DummyVectorStoreQuery(query_embedding=None)
    
    with pytest.raises(Exception) as excinfo:
        await store.aquery(query)
    assert "NO_QUERY_EMBEDDING" in str(excinfo.value.code)


@pytest.mark.asyncio
async def test_async_capabilities_caching():
    """Test that capabilities are cached for async operations."""
    store = _make_store()
    
    mock_caps = _make_dummy_caps()
    store._translator = Mock()
    store._translator.arun_capabilities = AsyncMock(return_value=mock_caps)
    
    # First call should fetch
    caps1 = await store._get_caps_async()
    assert caps1 is mock_caps
    store._translator.arun_capabilities.assert_called_once()
    
    # Second call should use cache
    caps2 = await store._get_caps_async()
    assert caps2 is mock_caps
    assert store._translator.arun_capabilities.call_count == 1


# 7. MMR integration
def test_mmr_query_uses_llamaindex_parameters():
    """Test that MMR query integrates with LlamaIndex parameters."""
    store = _make_store()
    
    query = DummyVectorStoreQuery(
        query_embedding=[0.1, 0.2, 0.3],
        similarity_top_k=4,
        filters=DummyMetadataFilters([
            DummyMetadataFilter("topic", "science", "==")
        ]),
        doc_ids=["doc-1"],
    )
    
    # Mock translator
    mock_translator = Mock()
    mock_translator.capabilities.return_value = _make_dummy_caps()
    mock_translator.query.return_value = Mock(matches=[Mock()] * 10)
    store._translator = mock_translator
    
    result = store.query_mmr(query, lambda_mult=0.7, fetch_k=20)
    
    # Verify translator was called with correct parameters
    call_args = mock_translator.query.call_args[0][0]
    assert call_args["top_k"] == 20  # fetch_k
    assert call_args["filters"] is not None


def test_mmr_algorithm_edge_cases():
    """Test MMR algorithm with edge cases."""
    store = _make_store()
    
    # Test with empty candidates
    indices = store._mmr_select_indices(
        query_vec=[0.1, 0.2, 0.3],
        candidate_matches=[],
        k=5,
        lambda_mult=0.5,
    )
    assert indices == []
    
    # Test with lambda_mult=1.0 (pure relevance)
    mock_matches = [Mock() for _ in range(5)]
    for i, match in enumerate(mock_matches):
        match.score = 1.0 - (i * 0.1)  # Decreasing scores
        match.vector = Mock()
        match.vector.vector = [float(i)] * 3
    
    indices = store._mmr_select_indices(
        query_vec=[0.1, 0.2, 0.3],
        candidate_matches=mock_matches,
        k=3,
        lambda_mult=1.0,
    )
    # Should select first 3 (highest scores)
    assert indices == [0, 1, 2]


def test_mmr_validation():
    """Test MMR parameter validation."""
    store = _make_store()
    
    query = DummyVectorStoreQuery(
        query_embedding=[0.1, 0.2, 0.3],
        similarity_top_k=4,
    )
    
    # Invalid lambda_mult
    with pytest.raises(Exception) as excinfo:
        store.query_mmr(query, lambda_mult=1.5)
    assert "BAD_MMR_LAMBDA" in str(excinfo.value.code)


# 8. Configuration validation
def test_configuration_validation():
    """Test configuration validation for LlamaIndex usage."""
    # Valid configurations should work
    store1 = _make_store(batch_size=50, default_top_k=20, score_threshold=0.8)
    assert store1.batch_size == 50
    assert store1.default_top_k == 20
    assert store1.score_threshold == 0.8
    
    # Invalid configurations should raise
    with pytest.raises(ValueError):
        _make_store(batch_size=0)
    
    with pytest.raises(ValueError):
        _make_store(default_top_k=0)
    
    with pytest.raises(ValueError):
        _make_store(score_threshold=1.5)


def test_configuration_warnings():
    """Test configuration warnings for extreme values."""
    import logging
    
    with patch.object(logging.getLogger('corpus_sdk.vector.framework_adapters.llamaindex'), 'warning') as mock_warning:
        _make_store(batch_size=15000)  # Unusually large
        mock_warning.assert_called()
        
    with patch.object(logging.getLogger('corpus_sdk.vector.framework_adapters.llamaindex'), 'warning') as mock_warning:
        _make_store(default_top_k=1500)  # Unusually large
        mock_warning.assert_called()
        
    with patch.object(logging.getLogger('corpus_sdk.vector.framework_adapters.llamaindex'), 'warning') as mock_warning:
        _make_store(score_threshold=0.95)  # Very high
        mock_warning.assert_called()


# 9. Streaming queries
def test_streaming_query_yields_node_with_score_format():
    """Test that streaming query yields NodeWithScore objects."""
    store = _make_store()
    
    # Mock translator stream
    mock_translator = Mock()
    mock_translator.capabilities.return_value = _make_dummy_caps()
    
    class MockChunk:
        def __init__(self):
            self.matches = []
    
    mock_chunk = MockChunk()
    mock_match = Mock()
    mock_match.vector = Mock()
    mock_match.vector.id = "vec-1"
    mock_match.vector.metadata = {
        store.node_id_field: "node-1",
        store.text_field: "Streamed text",
        store.id_field: "vec-1",
    }
    mock_match.score = 0.85
    mock_chunk.matches.append(mock_match)
    
    mock_translator.query_stream.return_value = [mock_chunk]
    store._translator = mock_translator
    
    query = DummyVectorStoreQuery(
        query_embedding=[0.1, 0.2, 0.3],
        similarity_top_k=5,
    )
    
    results = list(store.query_stream(query))
    
    assert len(results) == 1
    assert hasattr(results[0], "node")
    assert hasattr(results[0], "score")
    assert results[0].score == 0.85


def test_streaming_query_respects_top_k():
    """Test that streaming query stops after yielding top_k results."""
    store = _make_store()
    
    mock_translator = Mock()
    mock_translator.capabilities.return_value = _make_dummy_caps()
    
    # Create stream with more matches than top_k
    class MockChunk:
        def __init__(self, num_matches: int):
            self.matches = [Mock() for _ in range(num_matches)]
            for i, match in enumerate(self.matches):
                match.vector = Mock()
                match.vector.id = f"vec-{i}"
                match.vector.metadata = {
                    store.node_id_field: f"node-{i}",
                    store.text_field: f"Text {i}",
                    store.id_field: f"vec-{i}",
                }
                match.score = 0.9 - (i * 0.1)
    
    mock_translator.query_stream.return_value = [MockChunk(10)]
    store._translator = mock_translator
    
    query = DummyVectorStoreQuery(
        query_embedding=[0.1, 0.2, 0.3],
        similarity_top_k=3,  # Only want 3 results
    )
    
    results = list(store.query_stream(query))
    assert len(results) == 3  # Should stop at top_k


# 10. Context building
def test_context_building_from_llamaindex_callback_manager():
    """Test context building from LlamaIndex callback manager."""
    store = _make_store()
    
    mock_callback_manager = Mock()
    mock_operation_context = Mock()
    
    with patch('corpus_sdk.vector.framework_adapters.llamaindex.ctx_from_llamaindex', return_value=mock_operation_context):
        ctx = store._build_ctx(callback_manager=mock_callback_manager)
        assert ctx is mock_operation_context


def test_context_building_priority():
    """Test context building priority (OperationContext > dict > callback_manager)."""
    store = _make_store()
    
    # Priority 1: OperationContext
    mock_op_ctx = Mock(spec=llamaindex_adapter_module.OperationContext)
    ctx1 = store._build_ctx(ctx=mock_op_ctx)
    assert ctx1 is mock_op_ctx
    
    # Priority 2: Dict context
    mock_from_dict_result = Mock()
    with patch('corpus_sdk.vector.framework_adapters.llamaindex.ctx_from_dict', return_value=mock_from_dict_result):
        ctx2 = store._build_ctx(ctx={"key": "value"})
        assert ctx2 is mock_from_dict_result
    
    # Priority 3: Callback manager
    mock_callback_manager = Mock()
    mock_from_llamaindex_result = Mock()
    with patch('corpus_sdk.vector.framework_adapters.llamaindex.ctx_from_llamaindex', return_value=mock_from_llamaindex_result):
        ctx3 = store._build_ctx(callback_manager=mock_callback_manager)
        assert ctx3 is mock_from_llamaindex_result


# 11. Delete operations
def test_delete_by_ref_doc_id_uses_llamaindex_field():
    """Test delete uses configured ref_doc_id_field."""
    store = _make_store(ref_doc_id_field="llama_ref_doc")
    
    mock_translator = Mock()
    mock_translator.capabilities.return_value = _make_dummy_caps(supports_metadata_filtering=True)
    store._translator = mock_translator
    
    store.delete("doc-123")
    
    # Verify filter uses correct field name
    call_args = mock_translator.delete.call_args[0][0]
    assert call_args["filter"] == {"llama_ref_doc": "doc-123"}


def test_delete_nodes_handles_empty_list():
    """Test delete_nodes handles empty list gracefully."""
    store = _make_store()
    store.delete_nodes([])  # Should not raise


def test_delete_validation_requires_ids_or_filter():
    """Test delete validation requires ids or filter."""
    store = _make_store()
    
    # Mock capabilities
    mock_caps = _make_dummy_caps(supports_metadata_filtering=True)
    store._get_caps_sync = Mock(return_value=mock_caps)
    
    with pytest.raises(Exception) as excinfo:
        store._validate_delete_params_sync(ids=None, namespace=None, filter=None)
    assert "BAD_DELETE" in str(excinfo.value.code)


# 12. Score threshold filtering
def test_score_threshold_filtering():
    """Test client-side score threshold filtering."""
    store = _make_store(score_threshold=0.7)
    
    mock_matches = []
    for i in range(5):
        match = Mock()
        match.score = 0.6 + (i * 0.1)  # Scores: 0.6, 0.7, 0.8, 0.9, 1.0
        mock_matches.append(match)
    
    filtered = store._apply_score_threshold(mock_matches)
    
    # Should filter out scores < 0.7
    assert len(filtered) == 4  # 0.7, 0.8, 0.9, 1.0
    assert all(m.score >= 0.7 for m in filtered)


def test_no_score_threshold_returns_all():
    """Test that no threshold returns all matches."""
    store = _make_store(score_threshold=None)
    
    mock_matches = [Mock(score=0.5), Mock(score=0.8)]
    filtered = store._apply_score_threshold(mock_matches)
    
    assert len(filtered) == 2


# 13. Framework utilities
def test_uses_llamaindex_framework_utilities_for_warnings():
    """Test framework utilities are called with framework="llamaindex"."""
    store = _make_store()
    
    with patch('corpus_sdk.vector.framework_adapters.llamaindex.warn_if_extreme_k') as mock_warn:
        query = DummyVectorStoreQuery(
            query_embedding=[0.1, 0.2, 0.3],
            similarity_top_k=1000,
        )
        
        # Mock translator to avoid actual query
        mock_translator = Mock()
        mock_translator.capabilities.return_value = _make_dummy_caps()
        mock_translator.query.return_value = Mock(matches=[])
        store._translator = mock_translator
        
        try:
            store.query(query)
        except Exception:
            pass
        
        mock_warn.assert_called()
        call_kwargs = mock_warn.call_args[1]
        assert call_kwargs["framework"] == "llamaindex"
        assert call_kwargs["op_name"] == "query_sync"


# 14. Partial batch failure handling
def test_partial_upsert_failure_handling():
    """Test graceful handling of partial batch failures."""
    store = _make_store()
    
    mock_result = Mock()
    mock_result.failed_count = 2
    mock_result.upserted_count = 3
    mock_result.failures = ["err1", "err2"]
    
    # Should log warning but not raise
    with patch.object(store.logger, 'warning') as mock_warning:
        store._handle_partial_upsert_failure(mock_result, total_nodes=5, namespace="test")
        mock_warning.assert_called()


def test_complete_upsert_failure_raises():
    """Test that complete batch failure raises exception."""
    store = _make_store()
    
    mock_result = Mock()
    mock_result.failed_count = 5
    mock_result.upserted_count = 0
    mock_result.failures = ["err1", "err2", "err3", "err4", "err5"]
    
    with pytest.raises(Exception) as excinfo:
        store._handle_partial_upsert_failure(mock_result, total_nodes=5, namespace="test")
    assert "BATCH_UPSERT_FAILED" in str(excinfo.value.code)


# 15. Client property exposure
def test_client_property_exposes_corpus_adapter():
    """Test that client property exposes the underlying adapter."""
    mock_adapter = Mock()
    store = _make_store(adapter=mock_adapter)
    
    assert store.client is mock_adapter


# 16. Namespace handling
def test_effective_namespace_resolution():
    """Test namespace resolution with overrides."""
    store = _make_store(namespace="default-ns")
    
    # Use store default
    assert store._effective_namespace(None) == "default-ns"
    
    # Use explicit override
    assert store._effective_namespace("custom-ns") == "custom-ns"


def test_framework_context_includes_namespace():
    """Test that framework context includes namespace."""
    store = _make_store(namespace="test-ns")
    
    framework_ctx = store._framework_ctx_for_namespace(None)
    
    # Should include namespace in vector context
    assert "vector_context" in framework_ctx
    assert framework_ctx["vector_context"]["namespace"] == "test-ns"


# 17. Translator integration
def test_translator_is_cached_property():
    """Test that translator is a cached property."""
    mock_adapter = Mock()
    store = _make_store(adapter=mock_adapter)
    
    translator1 = store._translator
    translator2 = store._translator
    
    assert translator1 is translator2
    assert translator1.adapter is mock_adapter


# 18. VectorStoreInfo completeness
def test_vector_store_info_has_correct_structure():
    """Test that VectorStoreInfo has complete structure."""
    store = _make_store()
    info = store.vector_store_info
    
    assert hasattr(info, "name")
    assert hasattr(info, "description")
    assert hasattr(info, "metadata_info")
    assert isinstance(info.metadata_info, list)
    
    # Check all metadata info items have required fields
    for mi in info.metadata_info:
        assert hasattr(mi, "name")
        assert hasattr(mi, "description")
        assert hasattr(mi, "type")


# 19. Class name identification
def test_class_name_is_correct():
    """Test that class_name() returns correct identifier."""
    assert CorpusLlamaIndexVectorStore.class_name() == "CorpusLlamaIndexVectorStore"


# 20. VectorStore flags
def test_vector_store_flags_are_set():
    """Test that LlamaIndex VectorStore flags are properly set."""
    store = _make_store()
    
    assert store.stores_text is True
    assert store.flat_metadata is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
