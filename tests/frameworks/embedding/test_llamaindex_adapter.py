# tests/frameworks/embedding/test_llamaindex_adapter.py

from __future__ import annotations

from collections.abc import Sequence, Mapping
from typing import Any, Dict

import inspect
import pytest
from unittest.mock import Mock, patch
import concurrent.futures
import asyncio

import corpus_sdk.embedding.framework_adapters.llamaindex as llamaindex_adapter_module
from corpus_sdk.embedding.framework_adapters.llamaindex import (
    CorpusLlamaIndexEmbeddings,
    LLAMAINDEX_AVAILABLE,
    configure_llamaindex_embeddings,
    register_with_llamaindex,
    ErrorCodes,
    LlamaIndexAdapterConfig,
)
from corpus_sdk.embedding.embedding_base import OperationContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_embedding_matrix_shape(result: Any, expected_rows: int) -> None:
    """Validate that a result looks like a 2D embedding matrix."""
    assert isinstance(
        result, Sequence
    ), f"Expected sequence, got {type(result).__name__}"
    assert len(result) == expected_rows, (
        f"Expected {expected_rows} rows, got {len(result)}"
    )

    for row in result:
        assert isinstance(
            row, Sequence
        ), f"Row is not a sequence: {type(row).__name__}"
        for val in row:
            assert isinstance(
                val, (int, float)
            ), f"Embedding value is not numeric: {val!r}"


def _assert_embedding_vector_shape(result: Any) -> None:
    """Validate that a result looks like a 1D embedding vector."""
    assert isinstance(
        result, Sequence
    ), f"Expected sequence, got {type(result).__name__}"
    for val in result:
        assert isinstance(
            val, (int, float)
        ), f"Embedding value is not numeric: {val!r}"


def _make_embeddings(adapter: Any, **kwargs: Any) -> CorpusLlamaIndexEmbeddings:
    """
    Construct a CorpusLlamaIndexEmbeddings instance from the generic adapter.

    If the adapter doesn't implement get_embedding_dimension, we provide
    embedding_dimension explicitly to satisfy the constructor's contract.
    """
    base_kwargs: dict[str, Any] = {"corpus_adapter": adapter}
    if not hasattr(adapter, "get_embedding_dimension"):
        base_kwargs["embedding_dimension"] = 8
    base_kwargs.update(kwargs)
    return CorpusLlamaIndexEmbeddings(**base_kwargs)


# ---------------------------------------------------------------------------
# Constructor / validation behavior
# ---------------------------------------------------------------------------


def test_constructor_rejects_adapter_without_embed() -> None:
    """
    CorpusLlamaIndexEmbeddings should enforce that corpus_adapter exposes
    an `embed` method; otherwise __init__ should raise TypeError.
    """

    class BadAdapter:
        # deliberately missing `embed`
        def __init__(self) -> None:
            pass

    with pytest.raises(TypeError) as exc_info:
        CorpusLlamaIndexEmbeddings(corpus_adapter=BadAdapter(), embedding_dimension=8)

    msg = str(exc_info.value)
    assert "must implement an EmbeddingProtocolV1-compatible interface" in msg


def test_embedding_dimension_required_without_get_embedding_dimension() -> None:
    """
    If the corpus_adapter does not implement get_embedding_dimension(),
    the constructor should require embedding_dimension to be provided.
    """

    class NoDimAdapter:
        def embed(self, *args: Any, **kwargs: Any) -> list[list[float]]:
            return [[0.0, 0.0]]

    # Missing embedding_dimension -> error
    with pytest.raises(ValueError) as exc_info:
        CorpusLlamaIndexEmbeddings(corpus_adapter=NoDimAdapter())
    assert "Embedding dimension is unknown" in str(exc_info.value)

    # Providing embedding_dimension -> allowed
    embeddings = CorpusLlamaIndexEmbeddings(
        corpus_adapter=NoDimAdapter(),
        embedding_dimension=16,
    )
    assert isinstance(embeddings, CorpusLlamaIndexEmbeddings)
    assert embeddings.embedding_dimension == 16


def test_embedding_dimension_reads_from_adapter_when_available() -> None:
    """
    If the adapter exposes get_embedding_dimension(), embedding_dimension
    should be derived from it (unless explicitly overridden).
    """

    class DimAdapter:
        def embed(self, *args: Any, **kwargs: Any) -> list[list[float]]:
            return [[0.0] * 4]

        def get_embedding_dimension(self) -> int:
            return 4

    embeddings = CorpusLlamaIndexEmbeddings(corpus_adapter=DimAdapter())
    assert embeddings.embedding_dimension == 4

    # Explicit override should win
    embeddings_override = CorpusLlamaIndexEmbeddings(
        corpus_adapter=DimAdapter(),
        embedding_dimension=12,
    )
    assert embeddings_override.embedding_dimension == 12


def test_configure_and_register_helpers_return_embeddings(adapter: Any) -> None:
    """
    configure_llamaindex_embeddings and register_with_llamaindex should both
    return CorpusLlamaIndexEmbeddings instances wired to the given adapter.
    """
    emb1 = configure_llamaindex_embeddings(
        corpus_adapter=adapter,
        model_name="cfg-model",
    )
    assert isinstance(emb1, CorpusLlamaIndexEmbeddings)
    assert emb1.corpus_adapter is adapter

    emb2 = register_with_llamaindex(
        corpus_adapter=adapter,
        model_name="reg-model",
    )
    assert isinstance(emb2, CorpusLlamaIndexEmbeddings)
    assert emb2.corpus_adapter is adapter


def test_LLAMAINDEX_AVAILABLE_is_bool() -> None:
    """
    LLAMAINDEX_AVAILABLE flag should always be a boolean, regardless of
    whether LlamaIndex is actually installed.
    """
    assert isinstance(LLAMAINDEX_AVAILABLE, bool)


def test_llamaindex_interface_compatibility(adapter: Any) -> None:
    """
    Verify that CorpusLlamaIndexEmbeddings implements the expected LlamaIndex
    BaseEmbedding interface when LlamaIndex is available.
    """
    embeddings = _make_embeddings(adapter)

    # Core methods should always exist
    assert hasattr(embeddings, "_get_query_embedding")
    assert hasattr(embeddings, "_get_text_embedding")
    assert hasattr(embeddings, "_get_text_embeddings")
    assert hasattr(embeddings, "_aget_query_embedding")
    assert hasattr(embeddings, "_aget_text_embedding")
    assert hasattr(embeddings, "_aget_text_embeddings")

    if not LLAMAINDEX_AVAILABLE:
        pytest.skip(
            "LlamaIndex is not available; cannot assert base class compatibility",
        )

    try:
        from llama_index.core.embeddings import BaseEmbedding  # type: ignore[import]
    except Exception:
        pytest.skip(
            "LLAMAINDEX_AVAILABLE is True but importing BaseEmbedding failed",
        )

    assert isinstance(
        embeddings,
        BaseEmbedding,
    ), "CorpusLlamaIndexEmbeddings should subclass LlamaIndex BaseEmbedding when available"


# ---------------------------------------------------------------------------
# Context translation / LlamaIndexContext mapping
# ---------------------------------------------------------------------------


def test_llamaindex_context_passed_to_context_translation(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    Verify that kwargs used as LlamaIndexContext are passed through to
    context_from_llamaindex when embedding.
    """
    captured: Dict[str, Any] = {}

    def fake_from_llamaindex(ctx: Dict[str, Any]) -> None:
        captured["ctx"] = ctx
        # Returning None is allowed; adapter will just skip OperationContext.
        return None

    # Patch the imported symbol inside the module under test
    monkeypatch.setattr(
        llamaindex_adapter_module,
        "context_from_llamaindex",
        fake_from_llamaindex,
    )

    embeddings = _make_embeddings(adapter)

    llama_ctx = {
        "node_ids": ["n1", "n2"],
        "index_id": "idx-123",
        "trace_id": "trace-xyz",
        "workflow": "unit-test",
    }

    # Use the public batch embedding implementation
    result = embeddings._get_text_embeddings(["foo", "bar"], **llama_ctx)
    _assert_embedding_matrix_shape(result, expected_rows=2)

    assert captured.get("ctx") is not None
    assert captured["ctx"] == llama_ctx


# ---------------------------------------------------------------------------
# Sync semantics
# ---------------------------------------------------------------------------


def test_sync_query_and_text_embedding_basic(adapter: Any) -> None:
    """
    Basic smoke test for sync _get_query_embedding / _get_text_embedding /
    _get_text_embeddings behavior: they should accept text input and return
    numeric shapes.
    """
    embeddings = _make_embeddings(adapter)

    query = "llama-query"
    text = "llama-text"
    texts = ["llama-text-1", "llama-text-2", "llama-text-3"]

    q_vec = embeddings._get_query_embedding(query)
    _assert_embedding_vector_shape(q_vec)

    t_vec = embeddings._get_text_embedding(text)
    _assert_embedding_vector_shape(t_vec)

    t_mat = embeddings._get_text_embeddings(texts)
    _assert_embedding_matrix_shape(t_mat, expected_rows=len(texts))


def test_single_text_embedding_consistency(adapter: Any) -> None:
    """
    _get_text_embedding should be consistent with _get_text_embeddings
    for a single text input (at least in dimensionality, typically in values).
    """
    embeddings = _make_embeddings(adapter)

    text = "llama-single-text"

    single_result = embeddings._get_text_embedding(text)
    _assert_embedding_vector_shape(single_result)

    batch_result = embeddings._get_text_embeddings([text])
    _assert_embedding_matrix_shape(batch_result, expected_rows=1)

    # Dimensions must match; if either is empty, it's too weak a signal.
    if not batch_result or len(single_result) == 0 or len(batch_result[0]) == 0:
        pytest.skip("Zero-dimension embeddings; cannot assert consistency")

    assert len(single_result) == len(batch_result[0]), (
        "Single-text embedding dimension does not match batch-of-one row dimension"
    )


def test_empty_text_returns_zero_vector(adapter: Any) -> None:
    """
    Empty or whitespace-only texts should be handled via _handle_empty_text
    and return an all-zero vector of the correct dimension.
    """
    embeddings = _make_embeddings(adapter)

    dim = embeddings.embedding_dimension

    q_vec = embeddings._get_query_embedding("")
    t_vec = embeddings._get_text_embedding("   ")

    assert len(q_vec) == dim
    assert len(t_vec) == dim
    assert all(val == 0.0 for val in q_vec)
    assert all(val == 0.0 for val in t_vec)


def test_large_batch_sync_shape(adapter: Any) -> None:
    """
    Larger batches should still produce N rows for N inputs. This lightly
    stresses translator batching behavior.
    """
    embeddings = _make_embeddings(adapter)

    texts = [f"node-text-{i}" for i in range(40)]
    result = embeddings._get_text_embeddings(texts)
    _assert_embedding_matrix_shape(result, expected_rows=len(texts))


# ---------------------------------------------------------------------------
# Async semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_query_and_text_embedding_basic(adapter: Any) -> None:
    """
    Async _aget_query_embedding / _aget_text_embedding / _aget_text_embeddings
    should be coroutine functions and produce shapes compatible with sync API.
    """
    embeddings = _make_embeddings(adapter)

    # Ensure async methods exist and are coroutine functions
    assert hasattr(embeddings, "_aget_query_embedding")
    assert hasattr(embeddings, "_aget_text_embedding")
    assert hasattr(embeddings, "_aget_text_embeddings")

    assert inspect.iscoroutinefunction(embeddings._aget_query_embedding)
    assert inspect.iscoroutinefunction(embeddings._aget_text_embedding)
    assert inspect.iscoroutinefunction(embeddings._aget_text_embeddings)

    query = "async-llama-query"
    text = "async-llama-text"
    texts = ["async-text-1", "async-text-2"]

    q_vec = await embeddings._aget_query_embedding(query)
    _assert_embedding_vector_shape(q_vec)

    t_vec = await embeddings._aget_text_embedding(text)
    _assert_embedding_vector_shape(t_vec)

    t_mat = await embeddings._aget_text_embeddings(texts)
    _assert_embedding_matrix_shape(t_mat, expected_rows=len(texts))


@pytest.mark.asyncio
async def test_async_and_sync_same_dimension(adapter: Any) -> None:
    """
    Check that sync and async embeddings for the same input produce vectors
    of the same dimensionality (not necessarily identical values).
    """
    embeddings = _make_embeddings(adapter)

    texts = ["same-dim-1", "same-dim-2"]
    query = "same-dim-query"

    sync_q = embeddings._get_query_embedding(query)
    async_q = await embeddings._aget_query_embedding(query)

    sync_mat = embeddings._get_text_embeddings(texts)
    async_mat = await embeddings._aget_text_embeddings(texts)

    # Query dimensions
    assert len(sync_q) == len(async_q)

    # Batch row counts
    assert len(sync_mat) == len(async_mat) == len(texts)

    if sync_mat and async_mat:
        sync_dim = len(sync_mat[0])
        async_dim = len(async_mat[0])
        assert sync_dim == async_dim


# ---------------------------------------------------------------------------
# NEW TESTS: Error Context & Error Message Quality
# ---------------------------------------------------------------------------

def test_error_context_includes_llamaindex_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    When an error occurs, error context should include LlamaIndex-specific metadata
    via attach_context().
    """
    captured_context: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_context.update(ctx)

    monkeypatch.setattr(
        llamaindex_adapter_module,
        "attach_context",
        fake_attach_context,
    )

    class FailingAdapter:
        def embed(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("test error from llamaindex adapter: Check model configuration and API keys")

    embeddings = CorpusLlamaIndexEmbeddings(
        corpus_adapter=FailingAdapter(),
        embedding_dimension=8,
        model_name="err-model",
    )

    llama_ctx = {
        "node_ids": ["n1", "n2", "n3"],
        "index_id": "idx-123",
        "trace_id": "trace-xyz",
    }

    with pytest.raises(RuntimeError, match="test error from llamaindex adapter") as exc_info:
        embeddings._get_text_embedding("test text", **llama_ctx)
    
    # Verify error contains actionable information
    error_str = str(exc_info.value)
    assert "Check model configuration" in error_str or "API keys" in error_str
    
    # Verify some context was attached
    assert captured_context, "attach_context was not called"
    
    # Framework tagging should be present
    assert "framework" in captured_context
    assert captured_context.get("framework") == "llamaindex"
    
    # LlamaIndex-specific fields should be present in the context
    assert "node_ids" in captured_context
    assert captured_context.get("index_id") == "idx-123"
    assert captured_context.get("trace_id") == "trace-xyz"
    
    # Verify context contains debugging breadcrumbs
    assert "operation" in captured_context
    assert captured_context["operation"] == "embedding_text"
    
    # Verify error codes are attached for proper categorization
    assert "error_codes" in captured_context
    assert captured_context["error_codes"] == llamaindex_adapter_module.EMBEDDING_COERCION_ERROR_CODES


def test_embedding_error_context_includes_node_ids(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """
    Error context should include node_ids (truncated to max limit).
    """
    captured: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured.update(ctx)

    monkeypatch.setattr(llamaindex_adapter_module, "attach_context", fake_attach_context)

    class FailingTranslator:
        def embed(self, raw_texts: Any, op_ctx: Any = None, framework_ctx: Any = None) -> Any:
            raise RuntimeError("translator failed: Check model configuration")

    embeddings = CorpusLlamaIndexEmbeddings(
        corpus_adapter=adapter,
        embedding_dimension=8,
        model_name="test-model",
        llamaindex_config={"max_node_ids_in_context": 2},  # Limit to 2 nodes
    )

    with monkeypatch.context() as m:
        m.setattr(embeddings, "_translator", FailingTranslator())
        
        # Create many node_ids to test truncation
        llama_ctx = {
            "node_ids": [f"node-{i}" for i in range(10)],  # 10 nodes
            "index_id": "idx-123",
        }

        with pytest.raises(RuntimeError) as exc_info:
            embeddings._get_text_embedding("test text", **llama_ctx)
        
        error_str = str(exc_info.value)
        assert "translator failed" in error_str
        assert "Check model configuration" in error_str

        ctx = captured
        assert ctx["framework"] == "llamaindex"
        assert ctx["operation"] == "embedding_text"
        # Should include truncated node_ids
        assert "node_ids" in ctx
        assert len(ctx["node_ids"]) == 2  # Truncated to max limit
        assert ctx["node_ids"] == ["node-0", "node-1"]
        assert ctx["node_count"] == 10  # Total count should still be accurate
        assert ctx.get("node_ids_truncated") is True


def test_invalid_llamaindex_context_type_is_ignored(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """
    Non-mapping llamaindex_context should be ignored gracefully.
    """
    class DummyTranslator:
        def embed(self, raw_texts: Any, op_ctx: Any = None, framework_ctx: Any = None) -> Any:
            if isinstance(raw_texts, list):
                return [[0.0, 1.0] for _ in raw_texts]
            return [0.0, 1.0]

        async def arun_embed(self, raw_texts: Any, op_ctx: Any = None, framework_ctx: Any = None) -> Any:
            return self.embed(raw_texts, op_ctx, framework_ctx)

    monkeypatch.setattr(llamaindex_adapter_module, "create_embedding_translator", lambda **_: DummyTranslator())

    embeddings = _make_embeddings(adapter)

    # Non-mapping context should not break embeddings
    result = embeddings._get_text_embedding("x", **{"not-a-mapping": True})  # type: ignore[arg-type]
    _assert_embedding_vector_shape(result)


def test_context_from_llamaindex_failure_attaches_context(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """
    If context_from_llamaindex raises, attach_context should be called.
    """
    calls = {"attached": False}

    def boom(ctx: Dict[str, Any]) -> Any:
        raise RuntimeError("ctx boom")

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        if ctx.get("operation") == "context_build":
            calls["attached"] = True

    monkeypatch.setattr(llamaindex_adapter_module, "context_from_llamaindex", boom)
    monkeypatch.setattr(llamaindex_adapter_module, "attach_context", fake_attach_context)

    embeddings = _make_embeddings(adapter)

    llama_ctx = {"node_ids": ["n1"], "index_id": "idx-123"}
    result = embeddings._get_text_embedding("x", **llama_ctx)
    _assert_embedding_vector_shape(result)
    assert calls["attached"] is True


def test_embed_documents_error_context_includes_llamaindex_fields(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """
    Sync errors should include LlamaIndex context fields.
    """
    captured: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured.update(ctx)

    class FailingTranslator:
        def embed(self, raw_texts: Any, op_ctx: Any = None, framework_ctx: Any = None) -> Any:
            raise RuntimeError("translator failed: Check model configuration and API limits")

    monkeypatch.setattr(llamaindex_adapter_module, "attach_context", fake_attach_context)

    embeddings = CorpusLlamaIndexEmbeddings(
        corpus_adapter=adapter,
        embedding_dimension=8,
        model_name="test-model",
    )

    with monkeypatch.context() as m:
        m.setattr(embeddings, "_translator", FailingTranslator())
        
        llama_ctx = {
            "node_ids": ["doc-node-1", "doc-node-2"],
            "index_id": "doc-index",
            "workflow": "document-indexing",
            "callback_manager": Mock(),
        }

        with pytest.raises(RuntimeError) as exc_info:
            embeddings._get_text_embeddings(["doc1", "doc2"], **llama_ctx)
        
        error_str = str(exc_info.value)
        assert "translator failed" in error_str
        assert "Check model configuration" in error_str

        ctx = captured
        assert ctx["framework"] == "llamaindex"
        assert ctx["operation"] == "embedding_texts"
        assert ctx["model_name"] == "test-model"
        assert ctx["node_ids"] == ["doc-node-1", "doc-node-2"]
        assert ctx["index_id"] == "doc-index"
        assert ctx["workflow"] == "document-indexing"
        assert ctx.get("has_callback_manager") is True


@pytest.mark.asyncio
async def test_async_error_context_includes_llamaindex_fields(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """
    Async errors should include LlamaIndex context fields.
    """
    captured: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured.update(ctx)

    class FailingTranslator:
        async def arun_embed(self, raw_texts: Any, op_ctx: Any = None, framework_ctx: Any = None) -> Any:
            raise RuntimeError("async translator failed: Verify API key and model access permissions")

    monkeypatch.setattr(llamaindex_adapter_module, "attach_context", fake_attach_context)

    embeddings = CorpusLlamaIndexEmbeddings(
        corpus_adapter=adapter,
        embedding_dimension=8,
        model_name="async-test-model",
    )

    with monkeypatch.context() as m:
        m.setattr(embeddings, "_translator", FailingTranslator())
        
        llama_ctx = {"node_ids": ["async-node-1"], "trace_id": "async-trace"}

        with pytest.raises(RuntimeError) as exc_info:
            await embeddings._aget_text_embeddings(["async-doc1", "async-doc2"], **llama_ctx)
        
        error_str = str(exc_info.value)
        assert "async translator failed" in error_str
        assert "Verify API key" in error_str

        ctx = captured
        assert ctx["framework"] == "llamaindex"
        assert ctx["operation"] == "embedding_texts"
        assert ctx.get("node_ids") == ["async-node-1"]
        assert ctx.get("trace_id") == "async-trace"
        assert ctx.get("model_name") == "async-test-model"


def test_error_message_quality_for_invalid_inputs(adapter: Any) -> None:
    """
    Error messages should be actionable (not cryptic Python errors).
    """
    embeddings = _make_embeddings(adapter)

    # Test non-string query
    with pytest.raises(TypeError) as exc:
        embeddings._get_query_embedding(123)  # type: ignore[arg-type]
    
    error_msg = str(exc.value)
    assert "embedding_query" in error_msg or "embedding" in error_msg
    # Error should indicate what went wrong
    assert "str" in error_msg or "string" in error_msg


# ---------------------------------------------------------------------------
# NEW TESTS: Input Validation
# ---------------------------------------------------------------------------

def test_get_text_embeddings_rejects_non_string_items(adapter: Any) -> None:
    """
    _get_text_embeddings should reject non-string items when strict_text_types=True.
    """
    # Create embeddings with strict_text_types=True (default)
    embeddings = CorpusLlamaIndexEmbeddings(
        corpus_adapter=adapter,
        embedding_dimension=8,
        model_name="strict-model",
        llamaindex_config={"strict_text_types": True},
    )

    with pytest.raises(TypeError) as exc:
        embeddings._get_text_embeddings(["ok", 123, "ok2"])  # type: ignore[list-item]
    
    error_msg = str(exc.value)
    assert "_get_text_embeddings expects Sequence[str]" in error_msg
    assert "item 1 is int" in error_msg


@pytest.mark.asyncio
async def test_async_get_text_embeddings_rejects_non_string_items(adapter: Any) -> None:
    """
    _aget_text_embeddings should reject non-string items when strict_text_types=True.
    """
    embeddings = CorpusLlamaIndexEmbeddings(
        corpus_adapter=adapter,
        embedding_dimension=8,
        model_name="async-strict-model",
        llamaindex_config={"strict_text_types": True},
    )

    with pytest.raises(TypeError) as exc:
        await embeddings._aget_text_embeddings(["ok", object(), "ok2"])  # type: ignore[list-item]
    
    error_msg = str(exc.value)
    assert "_aget_text_embeddings expects Sequence[str]" in error_msg
    assert "item 1 is object" in error_msg


def test_llamaindex_config_rejects_non_mapping() -> None:
    """
    llamaindex_config must be a Mapping; non-mapping values should raise.
    """
    class MockAdapter:
        def embed(self, *args: Any, **kwargs: Any) -> list[list[float]]:
            return [[0.0] * 8]

    with pytest.raises(ValueError) as exc_info:
        CorpusLlamaIndexEmbeddings(
            corpus_adapter=MockAdapter(),
            embedding_dimension=8,
            llamaindex_config="not-a-mapping",  # type: ignore[arg-type]
        )

    msg = str(exc_info.value)
    assert "LLAMAINDEX_CONFIG_INVALID" in msg
    assert "llamaindex_config must be a Mapping" in msg


def test_embed_batch_size_validation() -> None:
    """
    embed_batch_size must be positive.
    """
    class MockAdapter:
        def embed(self, *args: Any, **kwargs: Any) -> list[list[float]]:
            return [[0.0] * 8]

    # Zero batch size should fail
    with pytest.raises(ValueError) as exc_info:
        CorpusLlamaIndexEmbeddings(
            corpus_adapter=MockAdapter(),
            embedding_dimension=8,
            embed_batch_size=0,
        )
    assert "embed_batch_size must be positive" in str(exc_info.value)

    # Negative batch size should fail
    with pytest.raises(ValueError) as exc_info:
        CorpusLlamaIndexEmbeddings(
            corpus_adapter=MockAdapter(),
            embedding_dimension=8,
            embed_batch_size=-1,
        )
    assert "embed_batch_size must be positive" in str(exc_info.value)

    # Positive batch size should work
    embeddings = CorpusLlamaIndexEmbeddings(
        corpus_adapter=MockAdapter(),
        embedding_dimension=8,
        embed_batch_size=100,
    )
    assert embeddings._embed_batch_size == 100


# ---------------------------------------------------------------------------
# NEW TESTS: Config/Context Validation
# ---------------------------------------------------------------------------

def test_llamaindex_config_defaults_and_bool_coercion(adapter: Any) -> None:
    """
    llamaindex_config should be normalized with defaults and booleans coerced.
    """
    embeddings = CorpusLlamaIndexEmbeddings(
        corpus_adapter=adapter,
        embedding_dimension=8,
        llamaindex_config={
            "enable_operation_context_propagation": 1,  # truthy -> bool
            "strict_text_types": 0,  # falsy -> bool
            # leave max_node_ids_in_context unset
        },
    )

    cfg = embeddings.llamaindex_config
    # Defaults filled in
    assert "enable_operation_context_propagation" in cfg
    assert "strict_text_types" in cfg
    assert "max_node_ids_in_context" in cfg

    # Bool coercion
    assert isinstance(cfg["enable_operation_context_propagation"], bool)
    assert isinstance(cfg["strict_text_types"], bool)
    assert isinstance(cfg["max_node_ids_in_context"], int)

    # Specific values
    assert cfg["enable_operation_context_propagation"] is True
    assert cfg["strict_text_types"] is False
    assert cfg["max_node_ids_in_context"] == 100  # default


def test_enable_operation_context_propagation_flag(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """
    enable_operation_context_propagation controls _operation_context inclusion.
    """
    def fake_from_llamaindex(ctx: Dict[str, Any]) -> OperationContext:
        return OperationContext(request_id="r1")

    monkeypatch.setattr(llamaindex_adapter_module, "context_from_llamaindex", fake_from_llamaindex)

    llama_ctx = {"node_ids": ["n1"]}

    # Default/True case
    emb_default = CorpusLlamaIndexEmbeddings(
        corpus_adapter=adapter,
        embedding_dimension=8,
        llamaindex_config={"enable_operation_context_propagation": True},
    )
    core_ctx, framework_ctx = emb_default._build_contexts(llamaindex_context=llama_ctx)
    assert isinstance(core_ctx, OperationContext)
    assert framework_ctx.get("_operation_context") is core_ctx

    # Disabled case
    emb_disabled = CorpusLlamaIndexEmbeddings(
        corpus_adapter=adapter,
        embedding_dimension=8,
        llamaindex_config={"enable_operation_context_propagation": False},
    )
    core_ctx2, framework_ctx2 = emb_disabled._build_contexts(llamaindex_context=llama_ctx)
    assert isinstance(core_ctx2, OperationContext)
    assert "_operation_context" not in framework_ctx2


def test_strict_text_types_flag_behavior(adapter: Any) -> None:
    """
    strict_text_types flag should control validation strictness.
    """
    # With strict_text_types=True (default), should reject
    emb_strict = CorpusLlamaIndexEmbeddings(
        corpus_adapter=adapter,
        embedding_dimension=8,
        llamaindex_config={"strict_text_types": True},
    )

    with pytest.raises(TypeError):
        emb_strict._get_text_embeddings(["ok", 123])  # type: ignore[list-item]

    # With strict_text_types=False, should handle gracefully
    emb_lenient = CorpusLlamaIndexEmbeddings(
        corpus_adapter=adapter,
        embedding_dimension=8,
        llamaindex_config={"strict_text_types": False},
    )

    # Should not raise, should handle non-string as empty
    result = emb_lenient._get_text_embeddings(["ok", 123, "ok2"])  # type: ignore[list-item]
    _assert_embedding_matrix_shape(result, expected_rows=3)
    # All rows should have the same dimension
    assert all(len(row) == 8 for row in result)


# ---------------------------------------------------------------------------
# NEW TESTS: Capabilities/Health Passthrough
# ---------------------------------------------------------------------------

def test_capabilities_passthrough_when_underlying_provides() -> None:
    """
    Should surface capabilities from underlying adapter.
    """
    class CapabilitiesAdapter:
        def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
            return [[0.0] * 8 for _ in texts]

        def capabilities(self) -> Dict[str, Any]:
            return {"supported_models": ["model-a", "model-b"], "max_tokens": 8192}

    adapter = CapabilitiesAdapter()
    embeddings = CorpusLlamaIndexEmbeddings(
        corpus_adapter=adapter,
        embedding_dimension=8,
        model_name="cap-model",
    )

    caps = embeddings.capabilities()
    assert isinstance(caps, dict)
    assert caps.get("supported_models") == ["model-a", "model-b"]
    assert caps.get("max_tokens") == 8192


def test_health_passthrough_when_underlying_provides() -> None:
    """
    Should surface health from underlying adapter.
    """
    class HealthAdapter:
        def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
            return [[0.0] * 8 for _ in texts]

        def health(self) -> Dict[str, Any]:
            return {"status": "healthy", "uptime_seconds": 3600}

    adapter = HealthAdapter()
    embeddings = CorpusLlamaIndexEmbeddings(
        corpus_adapter=adapter,
        embedding_dimension=8,
        model_name="health-model",
    )

    health = embeddings.health()
    assert isinstance(health, dict)
    assert health.get("status") == "healthy"
    assert health.get("uptime_seconds") == 3600


def test_capabilities_empty_when_missing() -> None:
    """
    Should return empty dict when adapter has no capabilities.
    """
    class NoCapAdapter:
        def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
            return [[0.0] * 8 for _ in texts]

    embeddings = CorpusLlamaIndexEmbeddings(
        corpus_adapter=NoCapAdapter(),
        embedding_dimension=8,
        model_name="no-cap-model",
    )

    caps = embeddings.capabilities()
    assert isinstance(caps, dict)
    assert caps == {}


def test_health_empty_when_missing() -> None:
    """
    Should return empty dict when adapter has no health.
    """
    class NoHealthAdapter:
        def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
            return [[0.0] * 8 for _ in texts]

    embeddings = CorpusLlamaIndexEmbeddings(
        corpus_adapter=NoHealthAdapter(),
        embedding_dimension=8,
        model_name="no-health-model",
    )

    health = embeddings.health()
    assert isinstance(health, dict)
    assert health == {}


# ---------------------------------------------------------------------------
# NEW TESTS: Resource Management
# ---------------------------------------------------------------------------

def test_context_manager_closes_underlying_adapter() -> None:
    """
    __enter__/__exit__ should call close on underlying adapter.
    """
    class ClosingAdapter:
        def __init__(self) -> None:
            self.closed = False
            self.aclosed = False

        def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
            return [[0.0] * 8 for _ in texts]

        def close(self) -> None:
            self.closed = True

        async def aclose(self) -> None:
            self.aclosed = True

    adapter = ClosingAdapter()
    embeddings = CorpusLlamaIndexEmbeddings(
        corpus_adapter=adapter,
        embedding_dimension=8,
        model_name="ctx-model",
    )

    with embeddings as emb:
        _ = emb._get_text_embedding("x")  # smoke test

    assert adapter.closed is True


@pytest.mark.asyncio
async def test_async_context_manager_closes_underlying_adapter() -> None:
    """
    __aenter__/__aexit__ should call aclose on underlying adapter.
    """
    class ClosingAdapter:
        def __init__(self) -> None:
            self.closed = False
            self.aclosed = False

        def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
            return [[0.0] * 8 for _ in texts]

        def close(self) -> None:
            self.closed = True

        async def aclose(self) -> None:
            self.aclosed = True

    adapter = ClosingAdapter()
    embeddings = CorpusLlamaIndexEmbeddings(
        corpus_adapter=adapter,
        embedding_dimension=8,
        model_name="async-ctx-model",
    )

    async with embeddings:
        _ = await embeddings._aget_text_embedding("y")

    assert adapter.aclosed is True


# ---------------------------------------------------------------------------
# NEW TESTS: Concurrency
# ---------------------------------------------------------------------------

@pytest.mark.concurrency
def test_shared_embedder_thread_safety(adapter: Any) -> None:
    """
    Shared embedder should be thread-safe for concurrent access.
    """
    embedder = configure_llamaindex_embeddings(
        corpus_adapter=adapter,
        model_name="concurrent-model",
        embedding_dimension=8,
    )

    def embed_query(text: str) -> Any:
        return embedder._get_query_embedding(text)

    texts = [f"query {i}" for i in range(10)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(embed_query, text) for text in texts]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(results) == len(texts)
    for result in results:
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)


@pytest.mark.asyncio
@pytest.mark.concurrency
async def test_concurrent_async_embedding(adapter: Any) -> None:
    """
    Async embedding should support concurrent operations.
    """
    embedder = configure_llamaindex_embeddings(
        corpus_adapter=adapter,
        model_name="async-concurrent-model",
        embedding_dimension=8,
    )

    async def embed_async(text: str) -> Any:
        return await embedder._aget_query_embedding(text)

    texts = [f"async query {i}" for i in range(5)]
    tasks = [embed_async(text) for text in texts]
    results = await asyncio.gather(*tasks)

    assert len(results) == len(texts)
    for result in results:
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)


# ---------------------------------------------------------------------------
# NEW TESTS: Integration
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestLlamaIndexIntegration:
    """
    Integration tests with real LlamaIndex objects.
    """
    
    @pytest.fixture
    def llamaindex_available(self) -> bool:
        try:
            import llama_index
            return True
        except ImportError:
            pytest.skip("LlamaIndex not installed - skipping integration tests")
    
    def test_can_use_with_llamaindex_settings(self, llamaindex_available: bool, adapter: Any) -> None:
        """
        Should work with LlamaIndex Settings.embed_model.
        """
        if not LLAMAINDEX_AVAILABLE:
            pytest.skip("LLAMAINDEX_AVAILABLE is False - skipping Settings integration")

        embedder = configure_llamaindex_embeddings(
            corpus_adapter=adapter,
            model_name="integration-model",
            embedding_dimension=8,
        )

        # Test basic embedding functionality
        docs = ["LlamaIndex is a framework.", "Embeddings convert text to vectors."]
        doc_vecs = embedder._get_text_embeddings(docs)
        _assert_embedding_matrix_shape(doc_vecs, expected_rows=len(docs))

        query_vec = embedder._get_query_embedding("What is LlamaIndex?")
        _assert_embedding_vector_shape(query_vec)
    
    def test_embeddings_work_in_llamaindex_pipelines(self, llamaindex_available: bool, adapter: Any) -> None:
        """
        Should work in LlamaIndex document processing pipelines.
        """
        if not LLAMAINDEX_AVAILABLE:
            pytest.skip("LLAMAINDEX_AVAILABLE is False - skipping pipeline integration")

        embedder = configure_llamaindex_embeddings(
            corpus_adapter=adapter,
            model_name="pipeline-model",
            embedding_dimension=8,
        )

        # Simulate LlamaIndex pipeline context
        llama_ctx = {
            "node_ids": ["doc-node-1", "doc-node-2"],
            "index_id": "pipeline-index",
            "workflow": "document-ingestion",
        }

        # Test with LlamaIndex context
        embeddings = embedder._get_text_embeddings(["doc1", "doc2"], **llama_ctx)
        _assert_embedding_matrix_shape(embeddings, expected_rows=2)
    
    def test_error_handling_in_llamaindex_workflow(self, llamaindex_available: bool) -> None:
        """
        Error handling in LlamaIndex context should be actionable.
        """
        class FailingAdapter:
            def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
                raise RuntimeError("Rate limit exceeded: Please wait 60 seconds before retrying")
            
            def get_embedding_dimension(self) -> int:
                return 8

        adapter = FailingAdapter()
        failing_embedder = CorpusLlamaIndexEmbeddings(
            corpus_adapter=adapter,
            model_name="failing-model",
        )

        with pytest.raises(RuntimeError) as exc_info:
            failing_embedder._get_text_embeddings(["test document"])

        error_str = str(exc_info.value)
        assert "rate limit" in error_str.lower() or "exceeded" in error_str.lower()
        assert "wait 60 seconds" in error_str.lower() or "retry" in error_str.lower()
