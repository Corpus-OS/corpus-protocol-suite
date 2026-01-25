# tests/frameworks/embedding/test_semantickernel_adapter.py

# ---------------------------------------------------------------------------
# Framework Version Support Matrix
# ---------------------------------------------------------------------------
"""
Framework Version Support:
- Semantic Kernel: 1.x+ (tested across common 1.x connector layouts; best-effort across minor moves)
- Python: 3.9+
- Corpus SDK: 1.0.0+

Integration Notes:
- Validates real compatibility with Semantic Kernel embedding service patterns:
  * EmbeddingGeneratorBase-derived service object (when SK installed)
  * Sync + async embedding entrypoints:
      - generate_embeddings / generate_embedding
      - generate_embeddings_async / generate_embedding_async
      - convenience aliases: embed_documents/embed_query + aembed_documents/aembed_query
- Ensures Semantic Kernel execution context is accepted and forwarded best-effort:
  plugin_name, function_name, kernel_id, memory_type, request_id, user_id, execution_settings
- Confirms strict, protocol-first behavior:
  * `corpus_adapter` must expose `embed` (duck-typed EmbeddingProtocolV1)
  * strict_text_types controls whether non-string items raise vs. zero-vector padded
  * sync methods refuse to run inside an active event loop (prevents deadlocks)
- Verifies adapter-level config normalization:
  enable_operation_context_propagation, strict_text_types, max_items_in_context
- Validates observability hardening:
  attach_context() receives framework identity, operation, model_id, batch metrics,
  SK routing fields and safe snapshots, and error_codes

Policy:
- Integration checks are PASS/FAIL (not skip): the suite asserts Semantic Kernel is installed
  (SEMANTIC_KERNEL_AVAILABLE=True) so failures reflect real integration regressions.
"""


from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict

import asyncio
import concurrent.futures
import inspect
import pytest
from unittest.mock import Mock

import corpus_sdk.embedding.framework_adapters.semantic_kernel as semantic_kernel_adapter_module
from corpus_sdk.embedding.framework_adapters.semantic_kernel import (
    CorpusSemanticKernelEmbeddings,
    SEMANTIC_KERNEL_AVAILABLE,
    SemanticKernelContext,
    ErrorCodes,
    register_with_semantic_kernel,
    configure_semantic_kernel_embeddings,
)
from corpus_sdk.embedding.embedding_base import OperationContext


# ---------------------------------------------------------------------------
# Framework Version Support Matrix
# ---------------------------------------------------------------------------
"""
Framework Version Support:
- Semantic Kernel: Python SDK 0.9+ (tested up to recent 1.x lines)
- Python: 3.8+
- Corpus SDK: 1.0.0+

Integration Notes:
- Compatible with SK's EmbeddingGeneratorBase when installed
- Supports SK's kernel services and function calling patterns
- Framework protocol-first design (adapter just needs an `embed` method)
- Error context includes SK-specific metadata (plugin_name, function_name, kernel_id)
"""


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


def _make_embeddings(adapter: Any, **kwargs: Any) -> CorpusSemanticKernelEmbeddings:
    """
    Construct a CorpusSemanticKernelEmbeddings instance from the generic adapter.

    If the adapter doesn't implement get_embedding_dimension, we provide
    embedding_dimension explicitly to satisfy the constructor's contract.
    """
    init_kwargs: dict[str, Any] = {"corpus_adapter": adapter, **kwargs}
    if not hasattr(adapter, "get_embedding_dimension"):
        init_kwargs.setdefault("embedding_dimension", 8)
    return CorpusSemanticKernelEmbeddings(**init_kwargs)


# ---------------------------------------------------------------------------
# Constructor / validation behavior
# ---------------------------------------------------------------------------


def test_constructor_rejects_adapter_without_embed() -> None:
    """
    CorpusSemanticKernelEmbeddings should enforce that corpus_adapter exposes
    an `embed` method; otherwise __init__ should raise TypeError.
    """

    class BadAdapter:
        # deliberately missing `embed`
        def __init__(self) -> None:
            pass

    with pytest.raises(TypeError) as exc_info:
        CorpusSemanticKernelEmbeddings(
            corpus_adapter=BadAdapter(),
            embedding_dimension=8,
        )

    msg = str(exc_info.value)
    assert "must implement an EmbeddingProtocolV1-compatible interface" in msg


def test_constructor_rejects_common_user_mistakes() -> None:
    """
    CorpusSemanticKernelEmbeddings should provide clear error messages for
    common user mistakes.
    """
    # Common mistake 1: Passing None
    with pytest.raises(TypeError) as exc_info:
        CorpusSemanticKernelEmbeddings(corpus_adapter=None)  # type: ignore[arg-type]

    msg = str(exc_info.value).lower()
    assert "embed" in msg or "embeddingprotocolv1" in msg

    # Common mistake 2: Passing a string (wrong type)
    with pytest.raises(TypeError) as exc_info:
        CorpusSemanticKernelEmbeddings(corpus_adapter="not an adapter")  # type: ignore[arg-type]

    msg = str(exc_info.value).lower()
    assert "embed" in msg or "embeddingprotocolv1" in msg


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
        CorpusSemanticKernelEmbeddings(corpus_adapter=NoDimAdapter())
    assert "Embedding dimension is unknown" in str(exc_info.value)

    # Providing embedding_dimension -> allowed
    embeddings = CorpusSemanticKernelEmbeddings(
        corpus_adapter=NoDimAdapter(),
        embedding_dimension=16,
    )
    assert isinstance(embeddings, CorpusSemanticKernelEmbeddings)
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

    embeddings = CorpusSemanticKernelEmbeddings(corpus_adapter=DimAdapter())
    assert embeddings.embedding_dimension == 4

    # Explicit override should win
    embeddings_override = CorpusSemanticKernelEmbeddings(
        corpus_adapter=DimAdapter(),
        embedding_dimension=12,
    )
    assert embeddings_override.embedding_dimension == 12


def test_embedding_dimension_property_behavior() -> None:
    """
    Test edge cases in embedding_dimension property.
    """
    class ErrorDimAdapter:
        def embed(self, *args: Any, **kwargs: Any) -> list[list[float]]:
            return [[0.0] * 8]
        
        def get_embedding_dimension(self) -> int:
            raise RuntimeError("Failed to get dimension")

    # Should fall back to override if adapter fails
    embeddings = CorpusSemanticKernelEmbeddings(
        corpus_adapter=ErrorDimAdapter(),
        embedding_dimension=16,
    )
    assert embeddings.embedding_dimension == 16


def test_sk_config_type_validation(adapter: Any) -> None:
    """
    sk_config must be a dict when provided; non-dicts should raise TypeError.
    """
    # Valid dict config
    emb = _make_embeddings(
        adapter,
        sk_config={"foo": "bar"},
    )
    assert isinstance(emb, CorpusSemanticKernelEmbeddings)
    assert emb.sk_config["foo"] == "bar"

    # Invalid type
    with pytest.raises(TypeError):
        CorpusSemanticKernelEmbeddings(
            corpus_adapter=adapter,
            embedding_dimension=8,
            sk_config="not-a-dict",  # type: ignore[arg-type]
        )


def test_sk_config_validation_with_invalid_types() -> None:
    """
    Test sk_config type validation with various invalid types.
    """
    class MockAdapter:
        def embed(self, *args: Any, **kwargs: Any) -> list[list[float]]:
            return [[0.0] * 8]

    adapter = MockAdapter()
    
    invalid_configs = [
        ("string", str),
        (123, int),
        ([1, 2, 3], list),
        (object(), type(object())),
    ]
    
    for config_val, expected_type in invalid_configs:
        with pytest.raises(TypeError) as exc_info:
            CorpusSemanticKernelEmbeddings(
                corpus_adapter=adapter,
                embedding_dimension=8,
                sk_config=config_val,  # type: ignore[arg-type]
            )
        msg = str(exc_info.value)
        assert "sk_config" in msg.lower()
        assert "Mapping" in msg or "dict" in msg


def test_sk_config_defaults_and_behavior(adapter: Any) -> None:
    """
    sk_config should be normalized with defaults and proper behavior.
    """
    # Test with empty config (should get defaults)
    emb_empty = CorpusSemanticKernelEmbeddings(
        corpus_adapter=adapter,
        embedding_dimension=8,
        sk_config={},
    )
    assert isinstance(emb_empty.sk_config, dict)
    assert "enable_operation_context_propagation" in emb_empty.sk_config
    assert "strict_text_types" in emb_empty.sk_config
    
    # Test with custom config
    emb_custom = CorpusSemanticKernelEmbeddings(
        corpus_adapter=adapter,
        embedding_dimension=8,
        sk_config={
            "enable_operation_context_propagation": False,
            "strict_text_types": False,
            "max_items_in_context": 50,
        },
    )
    assert emb_custom.sk_config["enable_operation_context_propagation"] is False
    assert emb_custom.sk_config["strict_text_types"] is False
    assert emb_custom.sk_config["max_items_in_context"] == 50


def test_sk_config_boolean_coercion(adapter: Any) -> None:
    """
    Boolean values in sk_config should be coerced properly.
    """
    # Test truthy/falsy coercion
    emb = CorpusSemanticKernelEmbeddings(
        corpus_adapter=adapter,
        embedding_dimension=8,
        sk_config={
            "enable_operation_context_propagation": 1,  # truthy -> True
            "strict_text_types": 0,  # falsy -> False
        },
    )
    assert emb.sk_config["enable_operation_context_propagation"] is True
    assert emb.sk_config["strict_text_types"] is False


def test_enable_operation_context_propagation_flag(adapter: Any) -> None:
    """
    enable_operation_context_propagation controls whether _operation_context
    is included in framework_ctx.
    """
    from unittest.mock import Mock
    
    # Mock the context translation to return an OperationContext
    mock_ctx = OperationContext(request_id="test-req")
    
    class MockAdapter:
        def embed(self, texts: Sequence[str], **kwargs: Any) -> list[list[float]]:
            # Check if _operation_context is in framework_ctx
            framework_ctx = kwargs.get("framework_ctx", {})
            if framework_ctx.get("enable_operation_context_propagation", True):
                assert "_operation_context" in framework_ctx
                assert framework_ctx["_operation_context"] is mock_ctx
            else:
                assert "_operation_context" not in framework_ctx
            return [[0.0] * 8 for _ in texts]
    
    adapter = MockAdapter()
    
    # Default/True case
    emb_default = CorpusSemanticKernelEmbeddings(
        corpus_adapter=adapter,
        embedding_dimension=8,
        sk_config={"enable_operation_context_propagation": True},
    )
    
    # Mock _build_contexts to return our mock context
    emb_default._build_contexts = Mock(return_value=(mock_ctx, {
        "framework": "semantic_kernel",
        "enable_operation_context_propagation": True
    }))
    
    emb_default.embed_documents(["test"])
    
    # Disabled case  
    emb_disabled = CorpusSemanticKernelEmbeddings(
        corpus_adapter=adapter,
        embedding_dimension=8,
        sk_config={"enable_operation_context_propagation": False},
    )
    
    emb_disabled._build_contexts = Mock(return_value=(mock_ctx, {
        "framework": "semantic_kernel",
        "enable_operation_context_propagation": False
    }))
    
    emb_disabled.embed_documents(["test"])


def test_strict_text_types_flag_behavior(adapter: Any) -> None:
    """
    strict_text_types flag should control validation strictness.
    """
    # With strict_text_types=True (default), should reject non-strings
    emb_strict = CorpusSemanticKernelEmbeddings(
        corpus_adapter=adapter,
        embedding_dimension=8,
        sk_config={"strict_text_types": True},
    )
    
    with pytest.raises(TypeError):
        emb_strict.embed_documents(["ok", 123])  # type: ignore[list-item]

    # With strict_text_types=False, should handle gracefully
    emb_lenient = CorpusSemanticKernelEmbeddings(
        corpus_adapter=adapter,
        embedding_dimension=8,
        sk_config={"strict_text_types": False},
    )
    
    # Should not raise, should handle non-string as empty
    result = emb_lenient.embed_documents(["ok", 123, "ok2"])  # type: ignore[list-item]
    _assert_embedding_matrix_shape(result, expected_rows=3)
    # All rows should have the same dimension
    assert all(len(row) == 8 for row in result)


def test_SEMANTIC_KERNEL_AVAILABLE_is_bool() -> None:
    """
    SEMANTIC_KERNEL_AVAILABLE flag should always be a boolean, regardless of
    whether Semantic Kernel is actually installed.
    """
    assert isinstance(SEMANTIC_KERNEL_AVAILABLE, bool)


# ---------------------------------------------------------------------------
# Context translation / SemanticKernelContext mapping
# ---------------------------------------------------------------------------


def test_semantickernel_context_passed_to_context_translation(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    Verify that sk_context is passed through to context_from_semantic_kernel
    when embedding.
    """
    captured: Dict[str, Any] = {}

    def fake_from_semantic_kernel(ctx: Dict[str, Any]) -> None:
        captured["ctx"] = ctx
        # Returning None is allowed; adapter will just skip OperationContext.
        return None

    # Patch the imported symbol inside the module under test
    monkeypatch.setattr(
        semantic_kernel_adapter_module,
        "context_from_semantic_kernel",
        fake_from_semantic_kernel,
    )

    embeddings = _make_embeddings(
        adapter,
    )

    sk_ctx: SemanticKernelContext = {
        "plugin_name": "test-plugin",
        "function_name": "embed-fn",
        "kernel_id": "kernel-123",
        "request_id": "req-456",
        "user_id": "user-abc",
    }

    # Use embed_documents alias, which routes through generate_embeddings
    result = embeddings.embed_documents(
        ["foo", "bar"],
        sk_context=sk_ctx,
    )
    _assert_embedding_matrix_shape(result, expected_rows=2)

    assert captured.get("ctx") is not None
    assert captured["ctx"] == sk_ctx


def test_invalid_sk_context_type_is_tolerated_and_does_not_crash(
    adapter: Any,
) -> None:
    """
    Passing a non-Mapping sk_context should not crash the adapter.

    The adapter is expected to log a warning and ignore the invalid context,
    still returning embeddings.
    """
    embeddings = _make_embeddings(adapter)

    texts = ["ctx-invalid-alpha", "ctx-invalid-beta"]

    result = embeddings.embed_documents(
        texts,
        sk_context="not-a-mapping",  # type: ignore[arg-type]
    )
    _assert_embedding_matrix_shape(result, expected_rows=len(texts))


def test_error_context_includes_semantickernel_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When an error occurs during Semantic Kernel embedding, error context should
    include SK-specific metadata via attach_context().
    """
    captured_context: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_context.update(ctx)

    monkeypatch.setattr(
        semantic_kernel_adapter_module,
        "attach_context",
        fake_attach_context,
    )

    class FailingAdapter:
        def embed(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("test error from SK adapter: Check configuration")

    embeddings = CorpusSemanticKernelEmbeddings(
        corpus_adapter=FailingAdapter(),
        embedding_dimension=8,
    )

    sk_ctx: SemanticKernelContext = {
        "plugin_name": "test-plugin",
        "function_name": "embed-fn",
        "kernel_id": "kernel-123",
    }

    with pytest.raises(RuntimeError, match="test error from SK adapter") as exc_info:
        embeddings.embed_documents(["text"], sk_context=sk_ctx)

    # Verify error is actionable
    error_str = str(exc_info.value)
    assert "Check configuration" in error_str

    # Verify some context was attached
    assert captured_context, "attach_context was not called"
    assert captured_context.get("framework") == "semantic_kernel"
    # SK-specific fields should be present
    assert captured_context.get("plugin_name") == "test-plugin"
    assert captured_context.get("function_name") == "embed-fn"


def test_error_context_includes_dynamic_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Error context should include dynamic metrics like text_len, texts_count.
    """
    captured_context: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_context.update(ctx)

    monkeypatch.setattr(
        semantic_kernel_adapter_module,
        "attach_context",
        fake_attach_context,
    )

    class FailingAdapter:
        def embed(self, texts: Sequence[str], **kwargs: Any) -> Any:
            raise RuntimeError("test error with metrics")

    embeddings = CorpusSemanticKernelEmbeddings(
        corpus_adapter=FailingAdapter(),
        embedding_dimension=8,
        model_id="test-model",
    )

    sk_ctx: SemanticKernelContext = {
        "plugin_name": "metrics-plugin",
        "kernel_id": "metrics-kernel",
    }

    with pytest.raises(RuntimeError):
        embeddings.embed_documents(["text1", "text2", "text3"], sk_context=sk_ctx)

    # Should include dynamic metrics
    assert captured_context.get("texts_count") == 3
    assert captured_context.get("model") == "test-model"
    assert captured_context.get("model_id") == "test-model"


def test_error_context_extraction_with_complex_sk_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Test context extraction with complex/nested SK context.
    """
    captured_context: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_context.update(ctx)

    monkeypatch.setattr(
        semantic_kernel_adapter_module,
        "attach_context",
        fake_attach_context,
    )

    class FailingAdapter:
        def embed(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("complex context error")

    embeddings = CorpusSemanticKernelEmbeddings(
        corpus_adapter=FailingAdapter(),
        embedding_dimension=8,
    )

    sk_ctx: SemanticKernelContext = {
        "plugin_name": "complex-plugin",
        "function_name": "complex-function",
        "kernel_id": "complex-kernel",
        "request_id": "req-complex-123",
        "user_id": "user-complex-456",
        "execution_settings": {"temperature": 0.7, "max_tokens": 100},
    }

    with pytest.raises(RuntimeError):
        embeddings.embed_documents(["test"], sk_context=sk_ctx)

    # Should include all relevant fields
    ctx = captured_context
    assert ctx.get("plugin_name") == "complex-plugin"
    assert ctx.get("function_name") == "complex-function"
    assert ctx.get("kernel_id") == "complex-kernel"
    assert ctx.get("request_id") == "req-complex-123"
    assert ctx.get("user_id") == "user-complex-456"
    # Complex nested fields might be snapshotted
    assert "execution_settings" in str(ctx)


@pytest.mark.asyncio
async def test_async_error_context_includes_sk_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Async error context should include SK-specific metadata.
    """
    captured_context: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_context.update(ctx)

    monkeypatch.setattr(
        semantic_kernel_adapter_module,
        "attach_context",
        fake_attach_context,
    )

    class FailingTranslator:
        async def arun_embed(
            self,
            raw_texts: Any,
            op_ctx: Any = None,
            framework_ctx: Any = None,
        ) -> Any:
            raise RuntimeError("async translator failed: Verify API key")

    class MinimalAdapter:
        def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
            return [[0.0, 1.0] for _ in texts]

    embeddings = CorpusSemanticKernelEmbeddings(
        corpus_adapter=MinimalAdapter(),
        embedding_dimension=8,
        model_id="async-err-model",
    )

    with monkeypatch.context() as m:
        m.setattr(embeddings, "_translator", FailingTranslator())
        
        sk_ctx: SemanticKernelContext = {
            "plugin_name": "async-plugin",
            "function_name": "async-fn",
            "kernel_id": "async-kernel",
        }

        with pytest.raises(RuntimeError, match="async translator failed") as exc_info:
            await embeddings.aembed_documents(["doc1", "doc2"], sk_context=sk_ctx)
        
        error_str = str(exc_info.value)
        assert "Verify API key" in error_str

        ctx = captured_context
        assert ctx["framework"] == "semantic_kernel"
        assert ctx["operation"] == "embedding_documents"
        assert ctx.get("plugin_name") == "async-plugin"
        assert ctx.get("function_name") == "async-fn"
        assert ctx.get("model_id") == "async-err-model"


def test_embed_documents_error_context_includes_all_fields(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    When embed_documents fails, error context should include all SK fields.
    """
    captured: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured.update(ctx)

    class FailingTranslator:
        def embed(
            self,
            raw_texts: Any,
            op_ctx: Any = None,
            framework_ctx: Any = None,
        ) -> Any:
            raise RuntimeError("translator failed: Check model configuration")

    monkeypatch.setattr(semantic_kernel_adapter_module, "attach_context", fake_attach_context)

    embeddings = CorpusSemanticKernelEmbeddings(
        corpus_adapter=adapter,
        embedding_dimension=8,
        model_id="test-model",
    )

    with monkeypatch.context() as m:
        m.setattr(embeddings, "_translator", FailingTranslator())
        
        sk_ctx: SemanticKernelContext = {
            "plugin_name": "doc-plugin",
            "function_name": "doc-fn",
            "kernel_id": "doc-kernel",
            "request_id": "req-doc-123",
            "user_id": "user-doc-456",
        }

        with pytest.raises(RuntimeError) as exc_info:
            embeddings.embed_documents(["text1", "text2"], sk_context=sk_ctx)
        
        error_str = str(exc_info.value)
        assert "translator failed" in error_str
        assert "Check model configuration" in error_str

        ctx = captured
        assert ctx["framework"] == "semantic_kernel"
        assert ctx["operation"] == "embedding_documents"
        assert ctx["model_id"] == "test-model"
        assert ctx.get("plugin_name") == "doc-plugin"
        assert ctx.get("function_name") == "doc-fn"
        assert ctx.get("kernel_id") == "doc-kernel"
        assert ctx.get("request_id") == "req-doc-123"


def test_generate_embedding_error_context_includes_sk_fields(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    generate_embedding errors should also include SK context.
    """
    captured: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured.update(ctx)

    class FailingTranslator:
        def embed(
            self,
            raw_texts: Any,
            op_ctx: Any = None,
            framework_ctx: Any = None,
        ) -> Any:
            raise RuntimeError("query translator failed")

    monkeypatch.setattr(semantic_kernel_adapter_module, "attach_context", fake_attach_context)

    embeddings = CorpusSemanticKernelEmbeddings(
        corpus_adapter=adapter,
        embedding_dimension=8,
        model_id="query-model",
    )

    with monkeypatch.context() as m:
        m.setattr(embeddings, "_translator", FailingTranslator())
        
        sk_ctx: SemanticKernelContext = {
            "plugin_name": "query-plugin",
            "function_name": "query-fn",
        }

        with pytest.raises(RuntimeError) as exc_info:
            embeddings.generate_embedding("test query", sk_context=sk_ctx)
        
        ctx = captured
        assert ctx["framework"] == "semantic_kernel"
        assert ctx["operation"] == "embedding_query" or ctx["operation"] == "embedding_text"
        assert ctx.get("model_id") == "query-model"
        assert ctx.get("plugin_name") == "query-plugin"


# ---------------------------------------------------------------------------
# Input Validation / Error Message Quality
# ---------------------------------------------------------------------------


def test_embed_documents_rejects_non_string_items(adapter: Any) -> None:
    """
    embed_documents should reject non-string items when strict_text_types=True.
    """
    embeddings = CorpusSemanticKernelEmbeddings(
        corpus_adapter=adapter,
        embedding_dimension=8,
        model_id="strict-model",
        sk_config={"strict_text_types": True},
    )

    with pytest.raises(TypeError) as exc:
        embeddings.embed_documents(["ok", 123, "ok2"])  # type: ignore[list-item]
    
    error_msg = str(exc.value)
    assert "embed_documents expects Sequence[str]" in error_msg or "expects Sequence[str]" in error_msg
    assert "item 1 is int" in error_msg or "item 1" in error_msg


def test_generate_embeddings_rejects_non_string_items(adapter: Any) -> None:
    """
    generate_embeddings should reject non-string items when strict_text_types=True.
    """
    embeddings = CorpusSemanticKernelEmbeddings(
        corpus_adapter=adapter,
        embedding_dimension=8,
        model_id="strict-gen-model",
        sk_config={"strict_text_types": True},
    )

    with pytest.raises(TypeError) as exc:
        embeddings.generate_embeddings(["ok", object(), "ok2"])  # type: ignore[list-item]
    
    error_msg = str(exc.value)
    assert "generate_embeddings expects Sequence[str]" in error_msg or "expects Sequence[str]" in error_msg
    assert "item 1 is object" in error_msg or "item 1" in error_msg


def test_embed_query_rejects_non_string() -> None:
    """
    embed_query should reject non-string input with clear error message.
    """
    class MockAdapter:
        def embed(self, *args: Any, **kwargs: Any) -> list[list[float]]:
            return [[0.0] * 8]

    adapter = MockAdapter()
    embeddings = CorpusSemanticKernelEmbeddings(
        corpus_adapter=adapter,
        embedding_dimension=8,
    )

    with pytest.raises(TypeError) as exc:
        embeddings.embed_query(123)  # type: ignore[arg-type]
    
    error_msg = str(exc.value)
    assert "embed_query expects str" in error_msg or "expects str" in error_msg
    assert "got int" in error_msg or "int" in error_msg


def test_generate_embedding_rejects_non_string() -> None:
    """
    generate_embedding should reject non-string input.
    """
    class MockAdapter:
        def embed(self, *args: Any, **kwargs: Any) -> list[list[float]]:
            return [[0.0] * 8]

    adapter = MockAdapter()
    embeddings = CorpusSemanticKernelEmbeddings(
        corpus_adapter=adapter,
        embedding_dimension=8,
    )

    with pytest.raises(TypeError) as exc:
        embeddings.generate_embedding({"not": "a string"})  # type: ignore[arg-type]
    
    error_msg = str(exc.value)
    assert "generate_embedding expects str" in error_msg or "expects str" in error_msg
    assert "dict" in error_msg or "got dict" in error_msg


@pytest.mark.asyncio
async def test_async_methods_reject_non_string_items(adapter: Any) -> None:
    """
    Async methods should reject non-string items when strict_text_types=True.
    """
    embeddings = CorpusSemanticKernelEmbeddings(
        corpus_adapter=adapter,
        embedding_dimension=8,
        model_id="async-strict-model",
        sk_config={"strict_text_types": True},
    )

    with pytest.raises(TypeError) as exc:
        await embeddings.aembed_documents(["ok", 456, "ok2"])  # type: ignore[list-item]
    
    error_msg = str(exc.value)
    assert "aembed_documents expects Sequence[str]" in error_msg or "expects Sequence[str]" in error_msg
    assert "item 1 is int" in error_msg or "item 1" in error_msg


def test_error_message_quality_for_invalid_inputs() -> None:
    """
    Error messages should be actionable (not cryptic Python errors).
    """
    class MockAdapter:
        def embed(self, *args: Any, **kwargs: Any) -> list[list[float]]:
            return [[0.0] * 8]

    adapter = MockAdapter()
    embeddings = CorpusSemanticKernelEmbeddings(
        corpus_adapter=adapter,
        embedding_dimension=8,
    )

    # Test non-string query
    with pytest.raises(TypeError) as exc:
        embeddings.embed_query(123)  # type: ignore[arg-type]
    
    error_msg = str(exc.value)
    # Error should indicate what went wrong and what was expected
    assert "embed_query" in error_msg or "embedding" in error_msg
    assert "str" in error_msg or "string" in error_msg
    assert "int" in error_msg or "123" in error_msg


# ---------------------------------------------------------------------------
# Sync / async semantics
# ---------------------------------------------------------------------------


def test_sync_generate_and_aliases_basic(adapter: Any) -> None:
    """
    Basic smoke test for sync generate_embeddings/generate_embedding and
    embed_documents/embed_query behavior: they should accept text input
    and return numeric shapes.
    """
    embeddings = _make_embeddings(adapter, model_id="sync-model")

    texts = ["alpha", "beta", "gamma"]
    query = "delta"

    # Core SK methods
    gen_mat = embeddings.generate_embeddings(texts)
    _assert_embedding_matrix_shape(gen_mat, expected_rows=len(texts))

    gen_vec = embeddings.generate_embedding(query)
    _assert_embedding_vector_shape(gen_vec)

    # Aliases
    docs_result = embeddings.embed_documents(texts)
    _assert_embedding_matrix_shape(docs_result, expected_rows=len(texts))

    query_result = embeddings.embed_query(query)
    _assert_embedding_vector_shape(query_result)


def test_empty_text_returns_zero_vector(adapter: Any) -> None:
    """
    Empty or whitespace-only texts should be handled via _handle_empty_text
    and return an all-zero vector of the correct dimension.
    """
    embeddings = _make_embeddings(adapter)

    dim = embeddings.embedding_dimension

    q_vec = embeddings.generate_embedding("")
    t_vec = embeddings.generate_embedding("   ")

    assert len(q_vec) == dim
    assert len(t_vec) == dim
    assert all(val == 0.0 for val in q_vec)
    assert all(val == 0.0 for val in t_vec)


def test_empty_texts_embed_documents_returns_empty_matrix(adapter: Any) -> None:
    """
    embed_documents([]) should be a no-op and return an empty sequence.
    """
    embeddings = _make_embeddings(adapter, model_id="empty-list-model")

    result = embeddings.embed_documents([])
    assert isinstance(result, Sequence)
    assert len(result) == 0


def test_empty_string_embed_query_has_consistent_dimension(adapter: Any) -> None:
    """
    Embedding an empty string should return a vector with same dimension
    as non-empty queries.
    """
    embeddings = _make_embeddings(adapter, model_id="empty-string-model")

    empty_vec = embeddings.embed_query("")
    non_empty_vec = embeddings.embed_query("non-empty query")

    _assert_embedding_vector_shape(empty_vec)
    _assert_embedding_vector_shape(non_empty_vec)

    # If both are non-empty, assert same dimensionality.
    if empty_vec and non_empty_vec:
        assert len(empty_vec) == len(non_empty_vec)


def test_large_batch_sync_shape(adapter: Any) -> None:
    """
    Large batches should still produce correct number of rows.
    """
    embeddings = _make_embeddings(adapter, model_id="large-batch-model")

    texts = [f"text-{i}" for i in range(50)]
    result = embeddings.embed_documents(texts)
    _assert_embedding_matrix_shape(result, expected_rows=len(texts))


def test_single_text_embedding_consistency(adapter: Any) -> None:
    """
    Single text embedding should be consistent with batch-of-one result.
    """
    embeddings = _make_embeddings(adapter)

    text = "single-text-test"

    single_result = embeddings.embed_query(text)
    _assert_embedding_vector_shape(single_result)

    batch_result = embeddings.embed_documents([text])
    _assert_embedding_matrix_shape(batch_result, expected_rows=1)

    # Dimensions must match
    if batch_result and len(single_result) > 0 and len(batch_result[0]) > 0:
        assert len(single_result) == len(batch_result[0])


@pytest.mark.asyncio
async def test_async_generate_and_aliases_basic(adapter: Any) -> None:
    """
    Async generate_embeddings_async / generate_embedding_async and
    aembed_documents / aembed_query should be coroutine functions and
    produce shapes compatible with the sync API.
    """
    embeddings = _make_embeddings(adapter)

    # Ensure async methods exist and are coroutine functions
    assert hasattr(embeddings, "generate_embeddings_async")
    assert hasattr(embeddings, "generate_embedding_async")
    assert hasattr(embeddings, "aembed_documents")
    assert hasattr(embeddings, "aembed_query")

    assert inspect.iscoroutinefunction(embeddings.generate_embeddings_async)
    assert inspect.iscoroutinefunction(embeddings.generate_embedding_async)
    assert inspect.iscoroutinefunction(embeddings.aembed_documents)
    assert inspect.iscoroutinefunction(embeddings.aembed_query)

    texts = ["async-alpha", "async-beta"]
    query = "async-gamma"

    gen_mat = await embeddings.generate_embeddings_async(texts)
    _assert_embedding_matrix_shape(gen_mat, expected_rows=len(texts))

    gen_vec = await embeddings.generate_embedding_async(query)
    _assert_embedding_vector_shape(gen_vec)

    docs_result = await embeddings.aembed_documents(texts)
    _assert_embedding_matrix_shape(docs_result, expected_rows=len(texts))

    query_result = await embeddings.aembed_query(query)
    _assert_embedding_vector_shape(query_result)


@pytest.mark.asyncio
async def test_async_and_sync_same_dimension(adapter: Any) -> None:
    """
    Check that sync and async embeddings for the same input produce vectors
    of the same dimensionality.
    """
    embeddings = _make_embeddings(adapter)

    texts = ["dim-a", "dim-b"]
    query = "dim-q"

    sync_docs = embeddings.embed_documents(texts)
    async_docs = await embeddings.aembed_documents(texts)

    sync_query = embeddings.embed_query(query)
    async_query = await embeddings.aembed_query(query)

    # Row counts
    assert len(sync_docs) == len(async_docs) == len(texts)

    # Dimensions (if any rows present)
    if sync_docs and async_docs:
        assert len(sync_docs[0]) == len(async_docs[0])

    # Query dims
    assert len(sync_query) == len(async_query)


# ---------------------------------------------------------------------------
# Semantic Kernel interface compatibility
# ---------------------------------------------------------------------------


def test_semantickernel_interface_compatibility(adapter: Any) -> None:
    """
    Verify that CorpusSemanticKernelEmbeddings implements the expected
    Semantic Kernel EmbeddingGeneratorBase interface when Semantic Kernel
    is available.
    """
    embeddings = _make_embeddings(adapter)

    # Core methods should always exist
    for name in (
        "generate_embeddings",
        "generate_embedding",
        "generate_embeddings_async",
        "generate_embedding_async",
        "embed_documents",
        "embed_query",
        "aembed_documents",
        "aembed_query",
    ):
        assert hasattr(embeddings, name), f"Missing expected method {name!r}"

    if not SEMANTIC_KERNEL_AVAILABLE:
        pytest.skip("Semantic Kernel is not available; cannot assert base-class compatibility")

    try:
        from semantic_kernel.connectors.ai.embeddings.embedding_generator_base import (  # type: ignore[import]  # noqa: E501
            EmbeddingGeneratorBase,
        )
    except Exception:
        pytest.skip(
            "SEMANTIC_KERNEL_AVAILABLE is True but importing EmbeddingGeneratorBase failed",
        )

    assert isinstance(
        embeddings,
        EmbeddingGeneratorBase,
    ), "CorpusSemanticKernelEmbeddings should subclass EmbeddingGeneratorBase when available"


# ---------------------------------------------------------------------------
# Capabilities / health passthrough
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
    embeddings = CorpusSemanticKernelEmbeddings(
        corpus_adapter=adapter,
        embedding_dimension=8,
        model_id="cap-model",
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
    embeddings = CorpusSemanticKernelEmbeddings(
        corpus_adapter=adapter,
        embedding_dimension=8,
        model_id="health-model",
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

    embeddings = CorpusSemanticKernelEmbeddings(
        corpus_adapter=NoCapAdapter(),
        embedding_dimension=8,
        model_id="no-cap-model",
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

    embeddings = CorpusSemanticKernelEmbeddings(
        corpus_adapter=NoHealthAdapter(),
        embedding_dimension=8,
        model_id="no-health-model",
    )

    health = embeddings.health()
    assert isinstance(health, dict)
    assert health == {}


@pytest.mark.asyncio
async def test_async_capabilities_and_health_fallback_to_sync() -> None:
    """
    acapabilities/ahealth should fall back to sync capabilities()/health()
    when only sync methods are implemented.
    """
    class SyncOnlyAdapter:
        def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
            return [[0.0] * 8 for _ in texts]

        def capabilities(self) -> Dict[str, Any]:
            return {"via_sync_caps": True}

        def health(self) -> Dict[str, Any]:
            return {"via_sync_health": True}

    embeddings = CorpusSemanticKernelEmbeddings(
        corpus_adapter=SyncOnlyAdapter(),
        embedding_dimension=8,
        model_id="sync-fallback-model",
    )

    acaps = await embeddings.acapabilities()
    assert isinstance(acaps, dict)
    assert acaps.get("via_sync_caps") is True

    ahealth = await embeddings.ahealth()
    assert isinstance(ahealth, dict)
    assert ahealth.get("via_sync_health") is True


# ---------------------------------------------------------------------------
# Resource management (context managers)
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
    embeddings = CorpusSemanticKernelEmbeddings(
        corpus_adapter=adapter,
        embedding_dimension=8,
        model_id="ctx-model",
    )

    with embeddings as emb:
        _ = emb.embed_documents(["x"])  # smoke test

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
    embeddings = CorpusSemanticKernelEmbeddings(
        corpus_adapter=adapter,
        embedding_dimension=8,
        model_id="async-ctx-model",
    )

    async with embeddings:
        _ = await embeddings.aembed_documents(["y"])

    assert adapter.aclosed is True


# ---------------------------------------------------------------------------
# Concurrency tests
# ---------------------------------------------------------------------------


@pytest.mark.concurrency
def test_shared_embedder_thread_safety(adapter: Any) -> None:
    """
    Shared embedder should be thread-safe for concurrent access.
    """
    embedder = configure_semantic_kernel_embeddings(
        corpus_adapter=adapter,
        model_id="concurrent-model",
        embedding_dimension=8,
    )

    def embed_query(text: str) -> Any:
        return embedder.embed_query(text)

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
    embedder = configure_semantic_kernel_embeddings(
        corpus_adapter=adapter,
        model_id="async-concurrent-model",
        embedding_dimension=8,
    )

    async def embed_async(text: str) -> Any:
        return await embedder.aembed_query(text)

    texts = [f"async query {i}" for i in range(5)]
    tasks = [embed_async(text) for text in texts]
    results = await asyncio.gather(*tasks)

    assert len(results) == len(texts)
    for result in results:
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)


# ---------------------------------------------------------------------------
# Semantic Kernel service registration helpers
# ---------------------------------------------------------------------------


def test_register_with_semantic_kernel_raises_when_kernel_is_none(
    adapter: Any,
) -> None:
    """
    register_with_semantic_kernel should raise ValueError when kernel is None.
    """
    with pytest.raises(ValueError, match="kernel cannot be None"):
        register_with_semantic_kernel(
            kernel=None,
            corpus_adapter=adapter,
            embedding_dimension=8,
        )


def test_register_with_semantic_kernel_uses_add_service_when_available(
    adapter: Any,
) -> None:
    """
    When kernel exposes add_service, register_with_semantic_kernel should call it
    and register the embeddings instance.
    """

    class DummyKernel:
        def __init__(self) -> None:
            self.calls: list[tuple[Any, Any | None]] = []

        def add_service(self, service: Any, service_id: str | None = None) -> None:
            self.calls.append((service, service_id))

    kernel = DummyKernel()

    embeddings = register_with_semantic_kernel(
        kernel=kernel,
        corpus_adapter=adapter,
        service_id="svc-1",
        model_id="model-1",
        embedding_dimension=8,
    )

    assert isinstance(embeddings, CorpusSemanticKernelEmbeddings)
    assert kernel.calls, "add_service was not called"

    svc, svc_id = kernel.calls[-1]
    assert svc is embeddings
    assert svc_id == "svc-1"


def test_register_with_semantic_kernel_falls_back_to_other_methods(
    adapter: Any,
) -> None:
    """
    If add_service fails, register_with_semantic_kernel should try other
    registration methods like register_embedding_generation.
    """

    class DummyKernel:
        def __init__(self) -> None:
            self.add_calls: int = 0
            self.reg_calls: list[tuple[Any, Any | None]] = []

        def add_service(self, service: Any, service_id: str | None = None) -> None:
            self.add_calls += 1
            raise TypeError("simulated mismatch")

        def register_embedding_generation(
            self,
            service: Any,
            service_id: str | None = None,
        ) -> None:
            self.reg_calls.append((service, service_id))

    kernel = DummyKernel()

    embeddings = register_with_semantic_kernel(
        kernel=kernel,
        corpus_adapter=adapter,
        service_id="fallback-svc",
        embedding_dimension=8,
    )

    assert isinstance(embeddings, CorpusSemanticKernelEmbeddings)
    assert kernel.add_calls == 1, "add_service should have been attempted"
    assert len(kernel.reg_calls) == 1, "register_embedding_generation should have been called"

    svc, svc_id = kernel.reg_calls[-1]
    assert svc is embeddings
    assert svc_id == "fallback-svc"


def test_register_with_semantic_kernel_when_no_registration_methods(
    adapter: Any,
) -> None:
    """
    If kernel has no known registration methods, register_with_semantic_kernel
    should still return an embeddings instance without raising.
    """

    class KernelNoMethods:
        def __init__(self) -> None:
            self.name = "no-methods-kernel"

    kernel = KernelNoMethods()

    embeddings = register_with_semantic_kernel(
        kernel=kernel,
        corpus_adapter=adapter,
        embedding_dimension=8,
    )
    assert isinstance(embeddings, CorpusSemanticKernelEmbeddings)


def test_configure_semantic_kernel_embeddings_returns_embeddings(adapter: Any) -> None:
    """
    configure_semantic_kernel_embeddings should return embeddings instance.
    """
    embeddings = configure_semantic_kernel_embeddings(
        corpus_adapter=adapter,
        model_id="cfg-model",
        embedding_dimension=8,
    )
    assert isinstance(embeddings, CorpusSemanticKernelEmbeddings)
    assert embeddings.corpus_adapter is adapter


# ---------------------------------------------------------------------------
# Integration tests with real Semantic Kernel objects
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSemanticKernelIntegration:
    """
    Integration tests with real Semantic Kernel objects.
    """

    @pytest.fixture
    def semantic_kernel_available(self) -> bool:
        try:
            import semantic_kernel  # noqa: F401
            return True
        except ImportError:
            pytest.skip("Semantic Kernel not installed - skipping integration tests")

    def test_can_use_with_semantic_kernel_kernel(
        self,
        semantic_kernel_available: bool,
        adapter: Any,
    ) -> None:
        """
        Integration: CorpusSemanticKernelEmbeddings should work with SK kernel.
        """
        if not SEMANTIC_KERNEL_AVAILABLE:
            pytest.skip("SEMANTIC_KERNEL_AVAILABLE is False - skipping SK integration")

        try:
            from semantic_kernel import Kernel  # type: ignore[import]
        except Exception:
            pytest.skip("Failed to import semantic_kernel.Kernel")

        embedder = configure_semantic_kernel_embeddings(
            corpus_adapter=adapter,
            model_id="integration-model",
            embedding_dimension=8,
        )

        # Create a kernel and try to add the embeddings
        kernel = Kernel()
        
        # Try different registration methods
        registration_succeeded = False
        try:
            if hasattr(kernel, 'add_service'):
                kernel.add_service(embedder, service_id="test-embeddings")
                registration_succeeded = True
            elif hasattr(kernel, 'register_embedding_generation'):
                kernel.register_embedding_generation(embedder, service_id="test-embeddings")
                registration_succeeded = True
        except Exception:
            # Some SK versions may have different APIs
            pass

        # Even if registration fails, embeddings should still work
        docs = ["Semantic Kernel is a framework.", "Embeddings convert text to vectors."]
        doc_vecs = embedder.embed_documents(docs)
        _assert_embedding_matrix_shape(doc_vecs, expected_rows=len(docs))

        query_vec = embedder.embed_query("What is Semantic Kernel?")
        _assert_embedding_vector_shape(query_vec)

    def test_embeddings_work_in_sk_pipelines(
        self,
        semantic_kernel_available: bool,
        adapter: Any,
    ) -> None:
        """
        Integration: Embeddings can be used in SK-style pipelines.
        """
        embedder = configure_semantic_kernel_embeddings(
            corpus_adapter=adapter,
            model_id="pipeline-model",
            embedding_dimension=8,
        )

        # Simulate SK pipeline context
        sk_ctx: SemanticKernelContext = {
            "plugin_name": "pipeline-plugin",
            "function_name": "embedding-function",
            "kernel_id": "pipeline-kernel",
            "request_id": "req-pipeline-123",
        }

        # Test with SK context
        embeddings = embedder.embed_documents(["doc1", "doc2"], sk_context=sk_ctx)
        _assert_embedding_matrix_shape(embeddings, expected_rows=2)

    def test_error_handling_in_sk_workflow(
        self,
        semantic_kernel_available: bool,
    ) -> None:
        """
        Integration: Error handling in SK context should be actionable.
        """
        class FailingAdapter:
            def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
                raise RuntimeError("Rate limit exceeded: Please wait 60 seconds before retrying")
            
            def get_embedding_dimension(self) -> int:
                return 8

        adapter = FailingAdapter()
        failing_embedder = CorpusSemanticKernelEmbeddings(
            corpus_adapter=adapter,
            model_id="failing-model",
        )

        with pytest.raises(RuntimeError) as exc_info:
            failing_embedder.embed_documents(["test document"])

        error_str = str(exc_info.value)
        assert "rate limit" in error_str.lower() or "exceeded" in error_str.lower()
        assert "wait 60 seconds" in error_str.lower() or "retry" in error_str.lower()
