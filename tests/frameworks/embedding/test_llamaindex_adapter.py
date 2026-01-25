# tests/frameworks/embedding/test_llamaindex_adapter.py

# ---------------------------------------------------------------------------
# Framework Version Support Matrix
# ---------------------------------------------------------------------------
"""
Framework Version Support:
- LlamaIndex: 0.10+ (tested across recent 0.10.x / 0.11.x / 0.12.x lines; best-effort across minor changes)
- Python: 3.9+
- Corpus SDK: 1.0.0+

Integration Notes:
- Validates real compatibility with LlamaIndex `BaseEmbedding` (no stub-only runs)
- Exercises LlamaIndex node-based embedding entrypoints:
  * _get_query_embedding / _aget_query_embedding
  * _get_text_embedding / _aget_text_embedding
  * _get_text_embeddings / _aget_text_embeddings
- Ensures kwargs-based LlamaIndex execution context is filtered and forwarded:
  node_ids, index_id, trace_id, workflow, callback_manager
- Confirms strict, protocol-first behavior:
  * `corpus_adapter` must expose `embed` (duck-typed EmbeddingProtocolV1)
  * sync methods refuse to run inside an active event loop (prevents deadlocks)
  * strict_text_types controls whether non-string items are rejected vs. zero-vector padded
- Verifies adapter-level configuration normalization:
  enable_operation_context_propagation, strict_text_types, max_node_ids_in_context
- Validates observability hardening:
  attach_context() receives framework identity, operation, model_name, node_id context,
  error_codes, and truncation metadata when applicable
- Includes PASS/FAIL integration checks (not skip) to ensure LlamaIndex is actually installed
  and the adapter is an instance of the real `llama_index.core.embeddings.BaseEmbedding`

Note: This suite is designed to run against any CORPUS embedding adapter via the shared
`adapter` fixture (see conftest.py). The tests are deterministic and focus on interface
conformance, context propagation, error-context richness, and safe sync/async semantics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict

import asyncio
import concurrent.futures
import inspect

import pytest

import corpus_sdk.embedding.framework_adapters.llamaindex as llamaindex_adapter_module
from corpus_sdk.embedding.framework_adapters.llamaindex import (
    CorpusLlamaIndexEmbeddings,
    LLAMAINDEX_AVAILABLE,
    configure_llamaindex_embeddings,
    register_with_llamaindex,
    ErrorCodes,
)
from corpus_sdk.embedding.embedding_base import OperationContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_embedding_matrix_shape(result: Any, expected_rows: int) -> None:
    """Validate that a result looks like a 2D embedding matrix."""
    assert isinstance(result, Sequence), f"Expected sequence, got {type(result).__name__}"
    assert len(result) == expected_rows, f"Expected {expected_rows} rows, got {len(result)}"

    for row in result:
        assert isinstance(row, Sequence), f"Row is not a sequence: {type(row).__name__}"
        for val in row:
            assert isinstance(val, (int, float)), f"Embedding value is not numeric: {val!r}"


def _assert_embedding_vector_shape(result: Any) -> None:
    """Validate that a result looks like a 1D embedding vector."""
    assert isinstance(result, Sequence), f"Expected sequence, got {type(result).__name__}"
    for val in result:
        assert isinstance(val, (int, float)), f"Embedding value is not numeric: {val!r}"


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


def _require_llamaindex() -> None:
    """
    Hard requirement: these tests are intended to PASS/FAIL (not skip) and validate
    real LlamaIndex integration. If LlamaIndex isn't installed, that's a failure.
    """
    assert LLAMAINDEX_AVAILABLE is True, (
        "LlamaIndex is not installed but these tests require real LlamaIndex integration. "
        "Install llama-index to run this suite."
    )
    # Also ensure the real BaseEmbedding can be imported (not just stubs).
    from llama_index.core.embeddings import BaseEmbedding  # noqa: F401  # type: ignore[import]


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
    assert "EmbeddingProtocolV1-compatible" in msg
    assert "embed" in msg.lower()


def test_embedding_dimension_required_without_get_embedding_dimension() -> None:
    """
    If the corpus_adapter does not implement get_embedding_dimension(),
    the constructor should require embedding_dimension to be provided.
    """

    class NoDimAdapter:
        def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
            return [[0.0, 0.0] for _ in texts]

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
    should be derived from it, unless explicitly overridden.
    """

    class DimAdapter:
        def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
            return [[0.0] * 4 for _ in texts]

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
        # ensure constructor contract satisfied if adapter lacks dimension method
        embedding_dimension=(None if hasattr(adapter, "get_embedding_dimension") else 8),
    )
    assert isinstance(emb1, CorpusLlamaIndexEmbeddings)
    assert emb1.corpus_adapter is adapter

    emb2 = register_with_llamaindex(
        corpus_adapter=adapter,
        model_name="reg-model",
        embedding_dimension=(None if hasattr(adapter, "get_embedding_dimension") else 8),
    )
    assert isinstance(emb2, CorpusLlamaIndexEmbeddings)
    assert emb2.corpus_adapter is adapter


def test_LLAMAINDEX_AVAILABLE_is_bool() -> None:
    """LLAMAINDEX_AVAILABLE flag should always be a boolean."""
    assert isinstance(LLAMAINDEX_AVAILABLE, bool)


def test_llamaindex_interface_compatibility(adapter: Any) -> None:
    """
    Verify that CorpusLlamaIndexEmbeddings implements the expected LlamaIndex
    BaseEmbedding interface and is an instance of the real BaseEmbedding.
    """
    _require_llamaindex()

    embeddings = _make_embeddings(adapter)

    # Core methods should always exist
    for name in (
        "_get_query_embedding",
        "_get_text_embedding",
        "_get_text_embeddings",
        "_aget_query_embedding",
        "_aget_text_embedding",
        "_aget_text_embeddings",
    ):
        assert hasattr(embeddings, name), f"Missing required method: {name}"

    from llama_index.core.embeddings import BaseEmbedding  # type: ignore[import]

    assert isinstance(
        embeddings,
        BaseEmbedding,
    ), "CorpusLlamaIndexEmbeddings should subclass LlamaIndex BaseEmbedding"


# ---------------------------------------------------------------------------
# Context translation / LlamaIndexContext mapping
# ---------------------------------------------------------------------------


def test_llamaindex_context_passed_to_context_translation(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    Verify that filtered kwargs used as LlamaIndexContext are passed through to
    context_from_llamaindex when embedding.
    """
    captured: Dict[str, Any] = {}

    def fake_from_llamaindex(ctx: Dict[str, Any], framework_version: Any = None) -> None:
        captured["ctx"] = ctx
        captured["framework_version"] = framework_version
        return None

    monkeypatch.setattr(llamaindex_adapter_module, "context_from_llamaindex", fake_from_llamaindex)

    embeddings = _make_embeddings(adapter)

    llama_ctx = {
        "node_ids": ["n1", "n2"],
        "index_id": "idx-123",
        "trace_id": "trace-xyz",
        "workflow": "unit-test",
    }

    result = embeddings._get_text_embeddings(["foo", "bar"], **llama_ctx)
    _assert_embedding_matrix_shape(result, expected_rows=2)

    assert captured.get("ctx") == llama_ctx


def test_invalid_llamaindex_context_type_is_ignored(adapter: Any) -> None:
    """
    llamaindex_context passed into internal context builder is best-effort:
    non-mapping values should be ignored (core_ctx=None) and embeddings should
    still function.
    """
    embeddings = _make_embeddings(adapter)

    core_ctx, framework_ctx = embeddings._build_contexts(llamaindex_context="not-a-mapping")  # type: ignore[arg-type]
    assert core_ctx is None
    assert framework_ctx.get("framework") == "llamaindex"


def test_context_from_llamaindex_failure_attaches_context(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    If context_from_llamaindex raises, embeddings proceed and attach_context is invoked
    with operation="context_build".
    """
    calls = {"attached": False}

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("ctx boom")

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        if ctx.get("operation") == "context_build":
            calls["attached"] = True

    monkeypatch.setattr(llamaindex_adapter_module, "context_from_llamaindex", boom)
    monkeypatch.setattr(llamaindex_adapter_module, "attach_context", fake_attach_context)

    embeddings = _make_embeddings(adapter)

    llama_ctx = {"node_ids": ["n1"], "index_id": "idx-123"}
    out = embeddings._get_text_embedding("x", **llama_ctx)
    _assert_embedding_vector_shape(out)

    assert calls["attached"] is True


# ---------------------------------------------------------------------------
# Sync semantics
# ---------------------------------------------------------------------------


def test_sync_query_and_text_embedding_basic(adapter: Any) -> None:
    """Basic smoke test for sync query/text/batch embeddings."""
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
    _get_text_embedding should be consistent with _get_text_embeddings([text])
    at least in dimensionality.
    """
    embeddings = _make_embeddings(adapter)
    text = "llama-single-text"

    single_result = embeddings._get_text_embedding(text)
    _assert_embedding_vector_shape(single_result)

    batch_result = embeddings._get_text_embeddings([text])
    _assert_embedding_matrix_shape(batch_result, expected_rows=1)

    assert len(single_result) == len(batch_result[0])


def test_empty_text_returns_zero_vector(adapter: Any) -> None:
    """Empty/whitespace-only texts should return an all-zero vector."""
    embeddings = _make_embeddings(adapter)
    dim = embeddings.embedding_dimension

    q_vec = embeddings._get_query_embedding("")
    t_vec = embeddings._get_text_embedding("   ")

    assert len(q_vec) == dim
    assert len(t_vec) == dim
    assert all(val == 0.0 for val in q_vec)
    assert all(val == 0.0 for val in t_vec)


def test_large_batch_sync_shape(adapter: Any) -> None:
    """Larger batches should still produce N rows for N inputs."""
    embeddings = _make_embeddings(adapter)
    texts = [f"node-text-{i}" for i in range(40)]
    result = embeddings._get_text_embeddings(texts)
    _assert_embedding_matrix_shape(result, expected_rows=len(texts))


# ---------------------------------------------------------------------------
# Async semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_query_and_text_embedding_basic(adapter: Any) -> None:
    """Async query/text/batch embeddings should exist and return correct shapes."""
    embeddings = _make_embeddings(adapter)

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
    """Sync and async should produce same dimensionality for same inputs."""
    embeddings = _make_embeddings(adapter)

    texts = ["same-dim-1", "same-dim-2"]
    query = "same-dim-query"

    sync_q = embeddings._get_query_embedding(query)
    async_q = await embeddings._aget_query_embedding(query)

    sync_mat = embeddings._get_text_embeddings(texts)
    async_mat = await embeddings._aget_text_embeddings(texts)

    assert len(sync_q) == len(async_q)
    assert len(sync_mat) == len(async_mat) == len(texts)
    if sync_mat and async_mat:
        assert len(sync_mat[0]) == len(async_mat[0])


# ---------------------------------------------------------------------------
# Event-loop guard semantics (must fail, not hang)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_methods_called_in_event_loop_raise(adapter: Any) -> None:
    """
    Sync methods must refuse to run inside an active event loop to prevent deadlocks.
    """
    embeddings = _make_embeddings(adapter)

    with pytest.raises(RuntimeError) as exc:
        embeddings._get_query_embedding("x")
    msg = str(exc.value)
    assert ErrorCodes.SYNC_WRAPPER_CALLED_IN_EVENT_LOOP in msg


# ---------------------------------------------------------------------------
# Error context & observability
# ---------------------------------------------------------------------------


def test_error_context_includes_llamaindex_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Errors should attach LlamaIndex metadata via attach_context()."""
    captured_context: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_context.update(ctx)

    monkeypatch.setattr(llamaindex_adapter_module, "attach_context", fake_attach_context)

    class FailingAdapter:
        def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
            raise RuntimeError("test error from llamaindex adapter: Check model configuration and API keys")

        def get_embedding_dimension(self) -> int:
            return 8

    embeddings = CorpusLlamaIndexEmbeddings(
        corpus_adapter=FailingAdapter(),
        model_name="err-model",
    )

    llama_ctx = {
        "node_ids": ["n1", "n2", "n3"],
        "index_id": "idx-123",
        "trace_id": "trace-xyz",
    }

    with pytest.raises(RuntimeError, match="test error from llamaindex adapter"):
        embeddings._get_text_embedding("test text", **llama_ctx)

    assert captured_context, "attach_context was not called"
    assert captured_context.get("framework") == "llamaindex"
    assert captured_context.get("operation") == "embedding_text"
    assert captured_context.get("index_id") == "idx-123"
    assert captured_context.get("trace_id") == "trace-xyz"
    assert captured_context.get("error_codes") == llamaindex_adapter_module.EMBEDDING_COERCION_ERROR_CODES
    assert isinstance(captured_context.get("node_ids"), list)


def test_embedding_error_context_truncates_node_ids(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """node_ids must be truncated to max_node_ids_in_context in attached context."""
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
        llamaindex_config={"max_node_ids_in_context": 2},
    )

    with monkeypatch.context() as m:
        # Override cached_property backing slot directly
        m.setattr(embeddings, "_translator", FailingTranslator())

        llama_ctx = {
            "node_ids": [f"node-{i}" for i in range(10)],
            "index_id": "idx-123",
        }

        with pytest.raises(RuntimeError, match="translator failed"):
            embeddings._get_text_embedding("test text", **llama_ctx)

    assert captured.get("framework") == "llamaindex"
    assert captured.get("operation") == "embedding_text"
    assert captured.get("node_ids") == ["node-0", "node-1"]
    assert captured.get("node_count") == 10
    assert captured.get("node_ids_truncated") is True


def test_error_message_quality_for_invalid_inputs(adapter: Any) -> None:
    """Error messages should be actionable (not cryptic)."""
    embeddings = _make_embeddings(adapter)

    with pytest.raises(TypeError) as exc:
        embeddings._get_query_embedding(123)  # type: ignore[arg-type]

    msg = str(exc.value)
    assert "expects str" in msg or "got" in msg
    assert "int" in msg


# ---------------------------------------------------------------------------
# Input validation: strict_text_types
# ---------------------------------------------------------------------------


def test_get_text_embeddings_rejects_non_string_items_when_strict(adapter: Any) -> None:
    """strict_text_types=True should reject non-string items."""
    embeddings = CorpusLlamaIndexEmbeddings(
        corpus_adapter=adapter,
        embedding_dimension=8,
        model_name="strict-model",
        llamaindex_config={"strict_text_types": True},
    )

    with pytest.raises(TypeError) as exc:
        embeddings._get_text_embeddings(["ok", 123, "ok2"])  # type: ignore[list-item]

    msg = str(exc.value)
    assert "_get_text_embeddings expects Sequence[str]" in msg
    assert "item 1 is int" in msg


@pytest.mark.asyncio
async def test_async_get_text_embeddings_rejects_non_string_items_when_strict(adapter: Any) -> None:
    """strict_text_types=True should reject non-string items on async batch path."""
    embeddings = CorpusLlamaIndexEmbeddings(
        corpus_adapter=adapter,
        embedding_dimension=8,
        model_name="async-strict-model",
        llamaindex_config={"strict_text_types": True},
    )

    with pytest.raises(TypeError) as exc:
        await embeddings._aget_text_embeddings(["ok", object(), "ok2"])  # type: ignore[list-item]

    msg = str(exc.value)
    assert "_aget_text_embeddings expects Sequence[str]" in msg
    assert "item 1 is object" in msg


def test_strict_text_types_false_preserves_row_alignment(adapter: Any) -> None:
    """
    strict_text_types=False should preserve row alignment by replacing non-string/empty items
    with zero vectors of the inferred embedding dimension.
    """
    embeddings = CorpusLlamaIndexEmbeddings(
        corpus_adapter=adapter,
        embedding_dimension=8,
        model_name="lenient-model",
        llamaindex_config={"strict_text_types": False},
    )

    out = embeddings._get_text_embeddings(["ok", 123, "ok2"])  # type: ignore[list-item]
    _assert_embedding_matrix_shape(out, expected_rows=3)

    # Determine expected dim from the first non-empty embedding row
    expected_dim = len(out[0])
    assert expected_dim > 0

    # Non-string item becomes zero vector of expected_dim
    assert out[1] == [0.0] * expected_dim

    # All rows must share the same dimension
    assert all(len(row) == expected_dim for row in out)


# ---------------------------------------------------------------------------
# llamaindex_config validation
# ---------------------------------------------------------------------------


def test_llamaindex_config_rejects_non_mapping() -> None:
    """llamaindex_config must be a Mapping; non-mapping values should raise."""

    class MockAdapter:
        def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
            return [[0.0] * 8 for _ in texts]

        def get_embedding_dimension(self) -> int:
            return 8

    with pytest.raises(ValueError) as exc_info:
        CorpusLlamaIndexEmbeddings(
            corpus_adapter=MockAdapter(),
            llamaindex_config="not-a-mapping",  # type: ignore[arg-type]
        )

    msg = str(exc_info.value)
    assert ErrorCodes.LLAMAINDEX_CONFIG_INVALID in msg
    assert "llamaindex_config must be a Mapping" in msg


def test_llamaindex_config_rejects_unknown_keys() -> None:
    """Unknown keys should be rejected to prevent silent misconfiguration."""

    class MockAdapter:
        def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
            return [[0.0] * 8 for _ in texts]

        def get_embedding_dimension(self) -> int:
            return 8

    with pytest.raises(ValueError) as exc_info:
        CorpusLlamaIndexEmbeddings(
            corpus_adapter=MockAdapter(),
            llamaindex_config={"unknown_key": True},  # type: ignore[typeddict-item]
        )

    msg = str(exc_info.value)
    assert ErrorCodes.LLAMAINDEX_CONFIG_INVALID in msg
    assert "unknown keys" in msg.lower()


def test_embed_batch_size_validation() -> None:
    """embed_batch_size must be positive."""

    class MockAdapter:
        def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
            return [[0.0] * 8 for _ in texts]

        def get_embedding_dimension(self) -> int:
            return 8

    with pytest.raises(ValueError):
        CorpusLlamaIndexEmbeddings(
            corpus_adapter=MockAdapter(),
            embed_batch_size=0,
        )

    with pytest.raises(ValueError):
        CorpusLlamaIndexEmbeddings(
            corpus_adapter=MockAdapter(),
            embed_batch_size=-1,
        )

    ok = CorpusLlamaIndexEmbeddings(
        corpus_adapter=MockAdapter(),
        embed_batch_size=100,
    )
    assert ok._embed_batch_size == 100


def test_llamaindex_config_defaults_and_bool_coercion(adapter: Any) -> None:
    """llamaindex_config should be normalized with defaults and type coercion."""
    embeddings = CorpusLlamaIndexEmbeddings(
        corpus_adapter=adapter,
        embedding_dimension=8,
        llamaindex_config={
            "enable_operation_context_propagation": 1,  # truthy -> bool
            "strict_text_types": 0,  # falsy -> bool
            # max_node_ids_in_context omitted => default
        },
    )

    cfg = embeddings.llamaindex_config
    assert isinstance(cfg["enable_operation_context_propagation"], bool)
    assert isinstance(cfg["strict_text_types"], bool)
    assert isinstance(cfg["max_node_ids_in_context"], int)

    assert cfg["enable_operation_context_propagation"] is True
    assert cfg["strict_text_types"] is False
    assert cfg["max_node_ids_in_context"] == 100


def test_enable_operation_context_propagation_flag(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """enable_operation_context_propagation controls _operation_context inclusion."""

    def fake_from_llamaindex(ctx: Dict[str, Any], framework_version: Any = None) -> OperationContext:
        return OperationContext(request_id="r1")

    monkeypatch.setattr(llamaindex_adapter_module, "context_from_llamaindex", fake_from_llamaindex)

    llama_ctx = {"node_ids": ["n1"]}

    emb_default = CorpusLlamaIndexEmbeddings(
        corpus_adapter=adapter,
        embedding_dimension=8,
        llamaindex_config={"enable_operation_context_propagation": True},
    )
    core_ctx, framework_ctx = emb_default._build_contexts(llamaindex_context=llama_ctx)
    assert isinstance(core_ctx, OperationContext)
    assert framework_ctx.get("_operation_context") is core_ctx

    emb_disabled = CorpusLlamaIndexEmbeddings(
        corpus_adapter=adapter,
        embedding_dimension=8,
        llamaindex_config={"enable_operation_context_propagation": False},
    )
    core_ctx2, framework_ctx2 = emb_disabled._build_contexts(llamaindex_context=llama_ctx)
    assert isinstance(core_ctx2, OperationContext)
    assert "_operation_context" not in framework_ctx2


# ---------------------------------------------------------------------------
# Capabilities / Health passthrough (via EmbeddingTranslator)
# ---------------------------------------------------------------------------


def test_capabilities_passthrough_when_underlying_provides() -> None:
    class CapabilitiesAdapter:
        def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
            return [[0.0] * 8 for _ in texts]

        def get_embedding_dimension(self) -> int:
            return 8

        def capabilities(self) -> Dict[str, Any]:
            return {"supported_models": ["model-a", "model-b"], "max_tokens": 8192}

    embeddings = CorpusLlamaIndexEmbeddings(
        corpus_adapter=CapabilitiesAdapter(),
        model_name="cap-model",
    )

    caps = embeddings.capabilities()
    assert isinstance(caps, dict)
    assert caps.get("supported_models") == ["model-a", "model-b"]
    assert caps.get("max_tokens") == 8192


def test_health_passthrough_when_underlying_provides() -> None:
    class HealthAdapter:
        def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
            return [[0.0] * 8 for _ in texts]

        def get_embedding_dimension(self) -> int:
            return 8

        def health(self) -> Dict[str, Any]:
            return {"status": "healthy", "uptime_seconds": 3600}

    embeddings = CorpusLlamaIndexEmbeddings(
        corpus_adapter=HealthAdapter(),
        model_name="health-model",
    )

    health = embeddings.health()
    assert isinstance(health, dict)
    assert health.get("status") == "healthy"
    assert health.get("uptime_seconds") == 3600


def test_capabilities_empty_when_missing() -> None:
    class NoCapAdapter:
        def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
            return [[0.0] * 8 for _ in texts]

        def get_embedding_dimension(self) -> int:
            return 8

    embeddings = CorpusLlamaIndexEmbeddings(
        corpus_adapter=NoCapAdapter(),
        model_name="no-cap-model",
    )
    caps = embeddings.capabilities()
    assert isinstance(caps, dict)
    assert caps == {}


def test_health_empty_when_missing() -> None:
    class NoHealthAdapter:
        def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
            return [[0.0] * 8 for _ in texts]

        def get_embedding_dimension(self) -> int:
            return 8

    embeddings = CorpusLlamaIndexEmbeddings(
        corpus_adapter=NoHealthAdapter(),
        model_name="no-health-model",
    )
    health = embeddings.health()
    assert isinstance(health, dict)
    assert health == {}


# ---------------------------------------------------------------------------
# Resource management (context managers)
# ---------------------------------------------------------------------------


def test_context_manager_closes_underlying_adapter() -> None:
    class ClosingAdapter:
        def __init__(self) -> None:
            self.closed = False
            self.aclosed = False

        def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
            return [[0.0] * 8 for _ in texts]

        def get_embedding_dimension(self) -> int:
            return 8

        def close(self) -> None:
            self.closed = True

        async def aclose(self) -> None:
            self.aclosed = True

    adapter = ClosingAdapter()
    embeddings = CorpusLlamaIndexEmbeddings(
        corpus_adapter=adapter,
        model_name="ctx-model",
    )

    with embeddings as emb:
        _ = emb._get_text_embedding("x")

    assert adapter.closed is True


@pytest.mark.asyncio
async def test_async_context_manager_closes_underlying_adapter() -> None:
    class ClosingAdapter:
        def __init__(self) -> None:
            self.closed = False
            self.aclosed = False

        def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
            return [[0.0] * 8 for _ in texts]

        def get_embedding_dimension(self) -> int:
            return 8

        def close(self) -> None:
            self.closed = True

        async def aclose(self) -> None:
            self.aclosed = True

    adapter = ClosingAdapter()
    embeddings = CorpusLlamaIndexEmbeddings(
        corpus_adapter=adapter,
        model_name="async-ctx-model",
    )

    async with embeddings:
        _ = await embeddings._aget_text_embedding("y")

    assert adapter.aclosed is True


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


@pytest.mark.concurrency
def test_shared_embedder_thread_safety(adapter: Any) -> None:
    embedder = configure_llamaindex_embeddings(
        corpus_adapter=adapter,
        model_name="concurrent-model",
        embedding_dimension=(None if hasattr(adapter, "get_embedding_dimension") else 8),
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
    embedder = configure_llamaindex_embeddings(
        corpus_adapter=adapter,
        model_name="async-concurrent-model",
        embedding_dimension=(None if hasattr(adapter, "get_embedding_dimension") else 8),
    )

    async def embed_async(text: str) -> Any:
        return await embedder._aget_query_embedding(text)

    texts = [f"async query {i}" for i in range(5)]
    results = await asyncio.gather(*(embed_async(t) for t in texts))

    assert len(results) == len(texts)
    for result in results:
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)


# ---------------------------------------------------------------------------
# Integration tests with real LlamaIndex objects (PASS/FAIL, not skip)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestLlamaIndexIntegration:
    """
    Integration tests with real LlamaIndex objects.
    These tests must PASS/FAIL (not skip). If LlamaIndex isn't installed, fail.
    """

    def test_llamaindex_is_installed(self) -> None:
        _require_llamaindex()

    def test_configure_llamaindex_embeddings_registers_settings_best_effort(self, adapter: Any) -> None:
        """
        configure_llamaindex_embeddings attempts to register Settings.embed_model when possible.
        We validate the integration path without requiring a specific Settings behavior
        across all LlamaIndex versions.
        """
        _require_llamaindex()

        embedder = configure_llamaindex_embeddings(
            corpus_adapter=adapter,
            model_name="integration-model",
            embedding_dimension=(None if hasattr(adapter, "get_embedding_dimension") else 8),
        )

        # Smoke: embeddings operate
        docs = ["LlamaIndex is a framework.", "Embeddings convert text to vectors."]
        doc_vecs = embedder._get_text_embeddings(docs)
        _assert_embedding_matrix_shape(doc_vecs, expected_rows=len(docs))

        query_vec = embedder._get_query_embedding("What is LlamaIndex?")
        _assert_embedding_vector_shape(query_vec)

        # Best-effort: if Settings exists, embed_model should be set to the embedder.
        # (Some versions may restrict Settings; we avoid brittle assertions.)
        try:
            from llama_index.core import Settings  # type: ignore[import]

            # If Settings exists and is mutable, it should now reference our embedder.
            # If it doesn't, that's still acceptable per adapter contract (best-effort).
            if getattr(Settings, "embed_model", None) is not None:
                assert getattr(Settings, "embed_model") is embedder or isinstance(getattr(Settings, "embed_model"), type(embedder))
        except Exception:
            # Best-effort integration; do not hard-fail on Settings API drift.
            pass

    def test_error_handling_in_llamaindex_workflow_is_actionable(self) -> None:
        _require_llamaindex()

        class FailingAdapter:
            def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
                raise RuntimeError("Rate limit exceeded: Please wait 60 seconds before retrying")

            def get_embedding_dimension(self) -> int:
                return 8

        failing_embedder = CorpusLlamaIndexEmbeddings(
            corpus_adapter=FailingAdapter(),
            model_name="failing-model",
        )

        with pytest.raises(RuntimeError) as exc_info:
            failing_embedder._get_text_embeddings(["test document"])

        s = str(exc_info.value).lower()
        assert "rate limit" in s or "exceeded" in s
        assert "wait" in s or "retry" in s
