# tests/frameworks/embedding/test_semantickernel_adapter.py

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict

import inspect
import pytest

import corpus_sdk.embedding.framework_adapters.semantic_kernel as semantic_kernel_adapter_module
from corpus_sdk.embedding.framework_adapters.semantic_kernel import (
    CorpusSemanticKernelEmbeddings,
    SEMANTIC_KERNEL_AVAILABLE,
    SemanticKernelContext,
    register_with_semantic_kernel,
)


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
            raise RuntimeError("test error from SK adapter")
        
        async def embed_batch(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("test error from SK adapter")

    embeddings = CorpusSemanticKernelEmbeddings(
        corpus_adapter=FailingAdapter(),
        embedding_dimension=8,
    )

    sk_ctx: SemanticKernelContext = {
        "plugin_name": "test-plugin",
        "function_name": "embed-fn",
        "kernel_id": "kernel-123",
    }

    with pytest.raises(RuntimeError, match="test error from SK adapter"):
        embeddings.embed_documents(["text"], sk_context=sk_ctx)

    # Verify some context was attached
    assert captured_context, "attach_context was not called"
    assert captured_context.get("framework") == "semantic_kernel"
    # SK-specific fields should be present
    assert captured_context.get("plugin_name") == "test-plugin"
    assert captured_context.get("function_name") == "embed-fn"


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
    of the same dimensionality (not necessarily identical values).
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

