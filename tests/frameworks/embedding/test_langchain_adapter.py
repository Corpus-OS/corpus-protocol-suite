from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict

import asyncio
import concurrent.futures
import inspect
import pytest
from pydantic import TypeAdapter, ValidationError

import corpus_sdk.embedding.framework_adapters.langchain as langchain_adapter_module
from corpus_sdk.embedding.framework_adapters.langchain import (
    CorpusLangChainEmbeddings,
    LANGCHAIN_AVAILABLE,
    configure_langchain_embeddings,
    register_with_langchain,
)
from corpus_sdk.embedding.embedding_base import OperationContext


# ---------------------------------------------------------------------------
# Framework Version Support Matrix
# ---------------------------------------------------------------------------
"""
Framework Version Support:
- LangChain: 0.1.0+ (tested up to recent 0.1.x / 0.2.x lines)
- Python: 3.8+
- Corpus SDK: 1.0.0+

Integration Notes:
- Compatible with LangChain's Embeddings base class when installed
- Supports LangChain's RunnableConfig-like patterns via `config`
  (run_id, run_name, tags, metadata, configurable, etc.)
- Framework protocol-first design (adapter just needs an `embed` method)

Note: LangChain compatibility is validated via duck typing and protocol
behavior, not strict inheritance. This keeps the adapter resilient even
as LangChain evolves its base classes.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_embedding_matrix_shape(
    result: Any,
    expected_rows: int,
) -> None:
    """Validate that a result looks like a 2D embedding matrix."""
    # NOTE: strings/bytes are Sequences but are never valid embedding matrices.
    assert isinstance(result, Sequence) and not isinstance(
        result, (str, bytes)
    ), f"Expected sequence (non-str), got {type(result).__name__}"
    assert len(result) == expected_rows, (
        f"Expected {expected_rows} rows, got {len(result)}"
    )

    for row in result:
        assert isinstance(row, Sequence) and not isinstance(
            row, (str, bytes)
        ), f"Row is not a sequence (non-str): {type(row).__name__}"
        for val in row:
            assert isinstance(
                val,
                (int, float),
            ), f"Embedding value is not numeric: {val!r}"


def _assert_embedding_vector_shape(result: Any) -> None:
    """Validate that a result looks like a 1D embedding vector."""
    # NOTE: strings/bytes are Sequences but are never valid embedding vectors.
    assert isinstance(result, Sequence) and not isinstance(
        result, (str, bytes)
    ), f"Expected sequence (non-str), got {type(result).__name__}"
    for val in result:
        assert isinstance(
            val,
            (int, float),
        ), f"Embedding value is not numeric: {val!r}"


def _make_embeddings(adapter: Any, **kwargs: Any) -> CorpusLangChainEmbeddings:
    """Construct a CorpusLangChainEmbeddings instance from the adapter."""
    return CorpusLangChainEmbeddings(corpus_adapter=adapter, **kwargs)


@pytest.fixture(scope="session")
def require_langchain() -> bool:
    """
    Pass/fail gating for *real* LangChain integration tests.

    Policy: no skip. If langchain-core isn't installed or importable, fail fast.
    """
    try:
        import langchain_core  # noqa: F401
        from langchain_core.embeddings import Embeddings  # noqa: F401
    except Exception as exc:
        pytest.fail(
            "LangChain integration tests require langchain-core. Install with:\n"
            "  pip install -U langchain-core\n"
            f"Import error: {exc!r}",
            pytrace=False,
        )

    # If imports succeed but adapter flag is false, that's an internal inconsistency.
    if not LANGCHAIN_AVAILABLE:
        pytest.fail(
            "LANGCHAIN_AVAILABLE is False but langchain_core imports succeeded. "
            "This indicates an internal inconsistency in the adapter module.",
            pytrace=False,
        )

    return True


# ---------------------------------------------------------------------------
# Pydantic / construction behavior
# ---------------------------------------------------------------------------


def test_pydantic_rejects_adapter_without_embed() -> None:
    """
    CorpusLangChainEmbeddings should validate that corpus_adapter has an
    `embed` method; otherwise Pydantic validation should fail.
    """

    class BadAdapter:
        # deliberately missing `embed`
        def __init__(self) -> None:
            pass

    with pytest.raises(ValidationError) as exc_info:
        CorpusLangChainEmbeddings(corpus_adapter=BadAdapter())

    # Make sure our custom ValueError propagated into the ValidationError
    msg = str(exc_info.value)
    assert "embed" in msg
    assert "EmbeddingProtocolV1" in msg or "must implement" in msg


def test_constructor_rejects_common_user_mistakes() -> None:
    """
    CorpusLangChainEmbeddings should provide clear error messages for
    common user mistakes.

    Error Message Quality: Users get helpful error messages, not
    cryptic Python errors.
    """
    # Common mistake 1: Passing None
    with pytest.raises((ValidationError, TypeError)) as exc_info:
        CorpusLangChainEmbeddings(corpus_adapter=None)  # type: ignore[arg-type]

    msg = str(exc_info.value).lower()
    assert "embed" in msg or "embeddingprotocolv1" in msg
    assert "none" in msg or "null" in msg

    # Common mistake 2: Passing a string (wrong type)
    with pytest.raises((ValidationError, TypeError)) as exc_info:
        CorpusLangChainEmbeddings(corpus_adapter="not an adapter")  # type: ignore[arg-type]

    msg = str(exc_info.value).lower()
    assert "embed" in msg or "embeddingprotocolv1" in msg
    assert "str" in msg or "string" in msg

    # Common mistake 3: Passing an object without embed() method
    class MockLLM:
        """Looks like an LLM but not an embedding adapter."""

        def invoke(self) -> None:
            pass

    with pytest.raises((ValidationError, TypeError)) as exc_info:
        CorpusLangChainEmbeddings(corpus_adapter=MockLLM())

    msg = str(exc_info.value).lower()
    assert "embed" in msg or "embeddingprotocolv1" in msg


def test_pydantic_accepts_valid_corpus_adapter(adapter: Any) -> None:
    """
    A valid corpus_adapter implementing `embed` should be accepted and
    stored as-is on the model.
    """
    embeddings = CorpusLangChainEmbeddings(
        corpus_adapter=adapter,
        model="test-model",
    )

    assert embeddings.corpus_adapter is adapter
    assert embeddings.model == "test-model"


def test_configure_and_register_helpers_return_embeddings(adapter: Any) -> None:
    """
    configure_langchain_embeddings and register_with_langchain should both
    return CorpusLangChainEmbeddings instances wired to the given adapter.
    """
    emb1 = configure_langchain_embeddings(
        corpus_adapter=adapter,
        model="cfg-model",
    )
    assert isinstance(emb1, CorpusLangChainEmbeddings)
    assert emb1.corpus_adapter is adapter

    emb2 = register_with_langchain(
        corpus_adapter=adapter,
        model="reg-model",
    )
    assert isinstance(emb2, CorpusLangChainEmbeddings)
    assert emb2.corpus_adapter is adapter


def test_LANGCHAIN_AVAILABLE_is_bool() -> None:
    """
    LANGCHAIN_AVAILABLE flag should always be a boolean, regardless of
    whether LangChain is actually installed.
    """
    assert isinstance(LANGCHAIN_AVAILABLE, bool)


def test_typed_dicts_are_pydantic_compatible_on_py_lt_312() -> None:
    """
    Regression test for Pydantic v2 + Python < 3.12.

    If the adapter defines TypedDicts using `typing.TypedDict` on Python < 3.12,
    Pydantic raises `PydanticUserError` during schema generation (often at import time).
    This test ensures LangChainConfig and LangChainAdapterConfig remain schema-compatible.
    """
    # These will raise PydanticUserError if TypedDict was imported from `typing`
    # (instead of `typing_extensions`) on Python < 3.12.
    TypeAdapter(langchain_adapter_module.LangChainConfig).json_schema()
    TypeAdapter(langchain_adapter_module.LangChainAdapterConfig).json_schema()


# ---------------------------------------------------------------------------
# LangChain interface compatibility
# ---------------------------------------------------------------------------


def test_langchain_interface_compatibility(adapter: Any) -> None:
    """
    Verify that CorpusLangChainEmbeddings implements the expected LangChain
    Embeddings interface when LangChain is available, and that the core
    embedding methods are present regardless.

    IMPORTANT POLICY (Path A): no pytest.skip here.
    - If LANGCHAIN_AVAILABLE is True, assert real langchain_core Embeddings compatibility.
    - If LANGCHAIN_AVAILABLE is False, assert fallback base class compatibility.
    """
    embeddings = _make_embeddings(adapter, model="iface-model")

    # Core methods should always exist
    assert hasattr(embeddings, "embed_documents")
    assert hasattr(embeddings, "embed_query")
    assert hasattr(embeddings, "aembed_documents")
    assert hasattr(embeddings, "aembed_query")

    if LANGCHAIN_AVAILABLE:
        # Adapter is implemented against langchain_core, not legacy langchain.embeddings.base.
        try:
            from langchain_core.embeddings import Embeddings  # type: ignore[import]
        except Exception as exc:
            pytest.fail(
                "LANGCHAIN_AVAILABLE is True but importing langchain_core.embeddings.Embeddings failed. "
                f"Import error: {exc!r}",
                pytrace=False,
            )

        assert isinstance(
            embeddings,
            Embeddings,
        ), "CorpusLangChainEmbeddings should be an Embeddings instance when LangChain is available"
    else:
        # When LangChain isn't installed, adapter still uses module-local fallback Embeddings base.
        assert isinstance(embeddings, langchain_adapter_module.Embeddings)


# ---------------------------------------------------------------------------
# RunnableConfig / context mapping
# ---------------------------------------------------------------------------


def test_runnable_config_passed_to_context_translation(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    Verify that the `config` kwarg is passed through to context_from_langchain
    for embed_documents().
    """
    captured: Dict[str, Any] = {}

    def fake_from_langchain(
        config: Dict[str, Any],
        framework_version: str | None = None,
    ) -> None:
        captured["config"] = config
        captured["framework_version"] = framework_version
        # Returning None is allowed; the adapter will just skip OperationContext usage.
        return None

    # Patch the imported symbol inside the module under test
    monkeypatch.setattr(
        langchain_adapter_module,
        "context_from_langchain",
        fake_from_langchain,
    )

    embeddings = CorpusLangChainEmbeddings(
        corpus_adapter=adapter,
        model="cfg-model",
        framework_version="lc-test-version",
    )

    config = {
        "run_id": "run-123",
        "run_name": "test-run",
        "tags": ["tag-a", "tag-b"],
        "metadata": {"pipeline": "unit-test"},
        "configurable": {"tenant": "acme"},
    }

    # Just ensure call succeeds and our fake context translator sees the config
    result = embeddings.embed_documents(["one", "two"], config=config)
    _assert_embedding_matrix_shape(result, expected_rows=2)

    assert captured.get("config") is config
    assert captured.get("framework_version") == "lc-test-version"


def test_runnable_config_passed_to_context_translation_for_embed_query(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    Verify that the `config` kwarg is also passed through to context_from_langchain
    for embed_query(), not just embed_documents().
    """
    captured: Dict[str, Any] = {}

    def fake_from_langchain(
        config: Dict[str, Any],
        framework_version: str | None = None,
    ) -> None:
        captured["config"] = config
        captured["framework_version"] = framework_version
        return None

    monkeypatch.setattr(
        langchain_adapter_module,
        "context_from_langchain",
        fake_from_langchain,
    )

    embeddings = CorpusLangChainEmbeddings(
        corpus_adapter=adapter,
        model="cfg-model-query",
        framework_version="lc-test-version-query",
    )

    config = {
        "run_id": "run-query-1",
        "run_name": "test-run-query",
    }

    result = embeddings.embed_query("some query text", config=config)
    _assert_embedding_vector_shape(result)

    assert captured.get("config") is config
    assert captured.get("framework_version") == "lc-test-version-query"


def test_config_to_operation_context_when_translator_returns_operation_context(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    Parity: when context_from_langchain returns an OperationContext, the
    adapter should treat it as such and pass it through core_ctx.
    """

    def fake_from_langchain(
        config: Dict[str, Any],
        framework_version: str | None = None,
    ) -> OperationContext:
        return OperationContext(request_id="r1")

    monkeypatch.setattr(
        langchain_adapter_module,
        "context_from_langchain",
        fake_from_langchain,
    )

    embeddings = CorpusLangChainEmbeddings(
        corpus_adapter=adapter,
        model="ctx-ok-model",
        framework_version="lc-ctx-ok",
    )

    config = {"run_id": "run-ok"}

    core_ctx, framework_ctx = embeddings._build_contexts(config=config)

    assert isinstance(core_ctx, OperationContext)
    assert framework_ctx["framework"] == "langchain"
    # If implementation exposes _operation_context, it must match
    if "_operation_context" in framework_ctx:
        assert framework_ctx["_operation_context"] is core_ctx


def test_context_from_langchain_failure_still_embeds(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    If context_from_langchain raises, embeddings proceed without OperationContext,
    and attach_context is invoked with operation="context_build".
    """
    calls: Dict[str, Any] = {}

    def boom(config: Dict[str, Any], framework_version: str | None = None) -> None:
        raise RuntimeError("ctx boom")

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        # Only capture the context_build calls to avoid noise
        if ctx.get("operation") == "context_build":
            calls["exc"] = exc
            calls["ctx"] = ctx

    monkeypatch.setattr(langchain_adapter_module, "context_from_langchain", boom)
    monkeypatch.setattr(langchain_adapter_module, "attach_context", fake_attach_context)

    embeddings = CorpusLangChainEmbeddings(
        corpus_adapter=adapter,
        model="ctx-fail-model",
        framework_version="lc-ctx-fail",
    )

    config = {"run_id": "run-ctx-fail"}

    result = embeddings.embed_documents(["x"], config=config)
    _assert_embedding_matrix_shape(result, expected_rows=1)

    # We should have attached error context
    assert "ctx" in calls
    ctx = calls["ctx"]
    assert ctx.get("framework") == "langchain"
    assert ctx.get("operation") == "context_build"
    # Snapshot of config should be present under some key
    assert any(
        key in ctx for key in ("langchain_config_snapshot", "config_snapshot")
    )


def test_invalid_config_type_is_ignored(adapter: Any) -> None:
    """
    config parameter that's not a Mapping should be handled gracefully:
    ignored and not fatal for embeddings.
    """
    embeddings = CorpusLangChainEmbeddings(
        corpus_adapter=adapter,
        model="invalid-config-model",
    )

    # Deliberately pass a non-mapping config; this should not raise
    result = embeddings.embed_documents(["x"], config="not-a-mapping")  # type: ignore[arg-type]
    _assert_embedding_matrix_shape(result, expected_rows=1)


# ---------------------------------------------------------------------------
# Adapter-level config / OperationContext propagation parity
# ---------------------------------------------------------------------------


def test_langchain_adapter_config_defaults_and_bool_coercion(adapter: Any) -> None:
    """
    langchain_config should be normalized with defaults and booleans coerced.

    Parity: Mirrors CrewAI config tests but for LangChain-specific config.
    """
    embeddings = CorpusLangChainEmbeddings(
        corpus_adapter=adapter,
        langchain_config={
            "fallback_to_simple_context": 1,  # truthy -> bool
            # let enable_operation_context_propagation default
        },
    )

    cfg = embeddings.langchain_config

    # Defaults filled in
    assert "fallback_to_simple_context" in cfg
    assert "enable_operation_context_propagation" in cfg

    # Bool coercion
    assert isinstance(cfg["fallback_to_simple_context"], bool)
    assert isinstance(cfg["enable_operation_context_propagation"], bool)

    # Specific values
    assert cfg["fallback_to_simple_context"] is True
    assert cfg["enable_operation_context_propagation"] is True


def test_langchain_adapter_config_rejects_non_mapping(adapter: Any) -> None:
    """
    langchain_config must be a Mapping; non-mapping values should raise
    a clear error mentioning LANGCHAIN_CONFIG_INVALID.
    """
    with pytest.raises(ValidationError) as exc_info:
        CorpusLangChainEmbeddings(
            corpus_adapter=adapter,
            langchain_config="not-a-mapping",  # type: ignore[arg-type]
        )

    msg = str(exc_info.value)
    assert "LANGCHAIN_CONFIG_INVALID" in msg or "langchain_config must be a Mapping" in msg


def test_fallback_to_simple_context_true_uses_default_operation_context(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    When context_from_langchain returns a non-OperationContext and
    fallback_to_simple_context is True, we should use a default OperationContext().
    """

    class WeirdCtx:
        pass

    def fake_from_langchain(
        config: Dict[str, Any],
        framework_version: str | None = None,
    ) -> Any:
        return WeirdCtx()

    monkeypatch.setattr(langchain_adapter_module, "context_from_langchain", fake_from_langchain)

    embeddings = CorpusLangChainEmbeddings(
        corpus_adapter=adapter,
        langchain_config={"fallback_to_simple_context": True},
    )

    config = {"run_id": "run-fallback-true"}

    core_ctx, framework_ctx = embeddings._build_contexts(config=config)

    assert isinstance(core_ctx, OperationContext)
    assert framework_ctx["framework"] == "langchain"


def test_fallback_to_simple_context_false_leaves_core_ctx_none(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    When context_from_langchain returns a non-OperationContext and
    fallback_to_simple_context is False, core_ctx should remain None.
    """

    class WeirdCtx:
        pass

    def fake_from_langchain(
        config: Dict[str, Any],
        framework_version: str | None = None,
    ) -> Any:
        return WeirdCtx()

    monkeypatch.setattr(langchain_adapter_module, "context_from_langchain", fake_from_langchain)

    embeddings = CorpusLangChainEmbeddings(
        corpus_adapter=adapter,
        langchain_config={"fallback_to_simple_context": False},
    )

    config = {"run_id": "run-fallback-false"}

    core_ctx, framework_ctx = embeddings._build_contexts(config=config)

    assert core_ctx is None
    assert framework_ctx["framework"] == "langchain"


def test_enable_operation_context_propagation_flag_controls_operation_context(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    enable_operation_context_propagation controls whether _operation_context
    is included in framework_ctx.
    """

    def fake_from_langchain(
        config: Dict[str, Any],
        framework_version: str | None = None,
    ) -> OperationContext:
        return OperationContext(request_id="r2")

    monkeypatch.setattr(langchain_adapter_module, "context_from_langchain", fake_from_langchain)

    config = {"run_id": "run-opctx"}

    # Default / True case
    emb_default = CorpusLangChainEmbeddings(
        corpus_adapter=adapter,
        langchain_config={"enable_operation_context_propagation": True},
    )
    core_ctx, framework_ctx = emb_default._build_contexts(config=config)
    assert isinstance(core_ctx, OperationContext)
    assert framework_ctx.get("_operation_context") is core_ctx

    # Disabled case
    emb_disabled = CorpusLangChainEmbeddings(
        corpus_adapter=adapter,
        langchain_config={"enable_operation_context_propagation": False},
    )
    core_ctx2, framework_ctx2 = emb_disabled._build_contexts(config=config)
    assert isinstance(core_ctx2, OperationContext)
    assert "_operation_context" not in framework_ctx2


def test_build_contexts_includes_framework_metadata(adapter: Any) -> None:
    """
    _build_contexts should include framework name, error_codes, and
    langchain_config in framework_ctx.
    """
    embeddings = CorpusLangChainEmbeddings(
        corpus_adapter=adapter,
        model="meta-model",
        langchain_config={"fallback_to_simple_context": True},
    )

    core_ctx, framework_ctx = embeddings._build_contexts(config={"run_id": "meta-run"})

    assert framework_ctx["framework"] == "langchain"
    assert "error_codes" in framework_ctx
    assert "langchain_config" in framework_ctx


# ---------------------------------------------------------------------------
# Error Message Quality: input validation
# ---------------------------------------------------------------------------


def test_embed_documents_rejects_non_string_items(adapter: Any) -> None:
    """Error Message Quality: Clear error for type mismatches."""
    embeddings = CorpusLangChainEmbeddings(corpus_adapter=adapter)

    with pytest.raises(TypeError) as exc:
        embeddings.embed_documents(["ok", 123])  # type: ignore[list-item]

    error_msg = str(exc.value)
    # Verify error is actionable
    assert "embed_documents expects Sequence[str]" in error_msg or "expects Sequence[str]" in error_msg
    assert "item 1 is int" in error_msg or "item 1" in error_msg


def test_embed_query_rejects_non_string(adapter: Any) -> None:
    """Error Message Quality: Clear error for type mismatches."""
    embeddings = CorpusLangChainEmbeddings(corpus_adapter=adapter)

    with pytest.raises(TypeError) as exc:
        embeddings.embed_query(123)  # type: ignore[arg-type]

    error_msg = str(exc.value)
    # Verify error is actionable
    assert "embed_query expects str" in error_msg
    assert "got int" in error_msg or "int" in error_msg


@pytest.mark.asyncio
async def test_aembed_documents_rejects_non_string_items(adapter: Any) -> None:
    """Error Message Quality: Consistent error messages for async API."""
    embeddings = CorpusLangChainEmbeddings(corpus_adapter=adapter)

    with pytest.raises(TypeError) as exc:
        await embeddings.aembed_documents(["ok", 123])  # type: ignore[list-item]

    error_msg = str(exc.value)
    # Verify error is actionable and consistent with sync version
    assert "aembed_documents expects Sequence[str]" in error_msg or "expects Sequence[str]" in error_msg
    assert "item 1 is int" in error_msg or "item 1" in error_msg


@pytest.mark.asyncio
async def test_aembed_query_rejects_non_string(adapter: Any) -> None:
    """Error Message Quality: Consistent error messages for async API."""
    embeddings = CorpusLangChainEmbeddings(corpus_adapter=adapter)

    with pytest.raises(TypeError) as exc:
        await embeddings.aembed_query(123)  # type: ignore[arg-type]

    error_msg = str(exc.value)
    # Verify error is actionable and consistent with sync version
    assert "aembed_query expects str" in error_msg
    assert "got int" in error_msg or "int" in error_msg


# ---------------------------------------------------------------------------
# Error-context decorator behavior
# ---------------------------------------------------------------------------


def test_error_context_includes_langchain_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When an error occurs during LangChain embedding, error context should
    include LangChain-specific metadata via attach_context().
    """
    captured_context: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_context.update(ctx)

    monkeypatch.setattr(
        langchain_adapter_module,
        "attach_context",
        fake_attach_context,
    )

    class FailingAdapter:
        def embed(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(
                "test error from langchain adapter: Check model configuration and API keys",
            )

    embeddings = CorpusLangChainEmbeddings(
        corpus_adapter=FailingAdapter(),
        model="err-model",
    )

    config = {
        "run_id": "run-ctx",
        "run_name": "error-test",
        "tags": ["tag-a"],
        "metadata": {"pipeline": "test"},
    }

    with pytest.raises(RuntimeError, match="test error from langchain adapter") as exc_info:
        embeddings.embed_documents(["x", "y"], config=config)

    # Verify error is actionable
    error_str = str(exc_info.value)
    assert "Check model configuration" in error_str or "API keys" in error_str

    # Verify some context was attached
    assert captured_context, "attach_context was not called"
    assert captured_context.get("framework") == "langchain"
    assert captured_context.get("operation") == "embedding_documents"
    # Best-effort propagation of config metadata
    assert captured_context.get("run_id") == "run-ctx"
    assert captured_context.get("run_name") == "error-test"


@pytest.mark.asyncio
async def test_async_error_context_includes_langchain_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Async embedding errors should also attach LangChain-specific metadata
    via attach_context(), mirroring the sync error-context behavior.

    NOTE: The Corpus embedding stack is protocol-first: the underlying adapter is
    only required to implement `embed()`. Async behavior may be provided by the
    translator (e.g., offloading sync embed to a thread), so this test MUST NOT
    assert that an adapter-level `aembed()` is invoked.
    """
    captured_context: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_context.update(ctx)

    monkeypatch.setattr(
        langchain_adapter_module,
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
            raise RuntimeError(
                "async test error from langchain translator: Verify API key and model access permissions",
            )

    class MinimalAdapter:
        # Must satisfy Pydantic validation (has `embed`), but it won't be used
        # because we patch the translator below.
        def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
            return [[0.0, 1.0] for _ in texts]

    embeddings = CorpusLangChainEmbeddings(
        corpus_adapter=MinimalAdapter(),
        model="err-async-model",
    )

    config = {
        "run_id": "run-ctx-async",
        "run_name": "error-test-async",
    }

    # Patch the translator so the async path fails deterministically.
    with monkeypatch.context() as m:
        m.setattr(embeddings, "_translator", FailingTranslator())

        with pytest.raises(
            RuntimeError,
            match="async test error from langchain translator",
        ) as exc_info:
            await embeddings.aembed_documents(["x", "y"], config=config)

    error_str = str(exc_info.value)
    assert "Verify API key" in error_str or "access permissions" in error_str

    assert captured_context, "attach_context was not called"
    assert captured_context.get("framework") == "langchain"
    assert captured_context.get("operation") == "embedding_documents"
    # Best-effort propagation of RunnableConfig metadata
    assert captured_context.get("run_id") == "run-ctx-async"
    assert captured_context.get("run_name") == "error-test-async"


def test_embed_documents_error_context_includes_langchain_fields(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    When embed_documents fails inside the translator, error context should include
    LangChain config fields (run_id, run_name, tags, metadata).
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
            raise RuntimeError(
                "translator failed: Check model configuration and API limits",
            )

    monkeypatch.setattr(langchain_adapter_module, "attach_context", fake_attach_context)

    embeddings = CorpusLangChainEmbeddings(
        corpus_adapter=adapter,
        model="test-model",
    )

    with monkeypatch.context() as m:
        m.setattr(embeddings, "_translator", FailingTranslator())

        config = {
            "run_id": "run-123",
            "run_name": "test-run",
            "tags": ["tag-a"],
            "metadata": {"pipeline": "test"},
        }

        with pytest.raises(RuntimeError) as exc_info:
            embeddings.embed_documents(["text"], config=config)

        error_str = str(exc_info.value)
        assert "translator failed" in error_str
        assert "Check model configuration" in error_str or "API limits" in error_str

        ctx = captured
        assert ctx["framework"] == "langchain"
        assert ctx["operation"] == "embedding_documents"
        assert ctx["model"] == "test-model"
        # RunnableConfig fields should be surfaced where possible
        assert ctx.get("run_id") == "run-123"
        assert ctx.get("run_name") == "test-run"


@pytest.mark.asyncio
async def test_aembed_query_error_context_includes_langchain_fields(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """Error Message Quality: Async query errors capture LangChain config context."""
    captured: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured.update(ctx)

    class FailingTranslator:
        async def arun_embed(
            self,
            raw_texts: Any,
            op_ctx: Any = None,
            framework_ctx: Any = None,
        ) -> Any:
            raise RuntimeError(
                "translator failed: Verify API key and model access permissions",
            )

    monkeypatch.setattr(langchain_adapter_module, "attach_context", fake_attach_context)

    embeddings = CorpusLangChainEmbeddings(
        corpus_adapter=adapter,
        model="async-test-model",
    )

    with monkeypatch.context() as m:
        m.setattr(embeddings, "_translator", FailingTranslator())

        config = {"run_id": "run-async-1", "run_name": "async-run"}

        with pytest.raises(RuntimeError) as exc_info:
            await embeddings.aembed_query("hello", config=config)

        error_str = str(exc_info.value)
        assert "translator failed" in error_str
        assert "Verify API key" in error_str or "access permissions" in error_str

        ctx = captured
        assert ctx["framework"] == "langchain"
        assert ctx["operation"] == "embedding_query"
        assert ctx.get("run_id") == "run-async-1"
        assert ctx.get("model") == "async-test-model"


# ---------------------------------------------------------------------------
# Sync / async semantics
# ---------------------------------------------------------------------------


def test_sync_embed_documents_and_query_basic(adapter: Any) -> None:
    """
    Basic smoke test for sync embed_documents and embed_query behavior:
    they should accept simple text input and return numeric shapes.
    """
    embeddings = configure_langchain_embeddings(
        corpus_adapter=adapter,
        model="sync-model",
    )

    texts = ["alpha", "beta", "gamma"]
    query_text = "delta"

    docs_result = embeddings.embed_documents(texts)
    _assert_embedding_matrix_shape(docs_result, expected_rows=len(texts))

    query_result = embeddings.embed_query(query_text)
    _assert_embedding_vector_shape(query_result)


def test_empty_texts_embed_documents_returns_empty_matrix(adapter: Any) -> None:
    """
    embed_documents([]) should be a no-op and return an empty sequence,
    not raise.
    """
    embeddings = configure_langchain_embeddings(
        corpus_adapter=adapter,
        model="empty-list-model",
    )

    result = embeddings.embed_documents([])
    assert isinstance(result, Sequence)
    assert len(result) == 0


def test_empty_string_embed_query_has_consistent_dimension(adapter: Any) -> None:
    """
    Embedding an empty string should still return a numeric vector, and its
    dimensionality should match that of a non-empty query.
    """
    embeddings = configure_langchain_embeddings(
        corpus_adapter=adapter,
        model="empty-string-model",
    )

    empty_vec = embeddings.embed_query("")
    non_empty_vec = embeddings.embed_query("non-empty query")

    _assert_embedding_vector_shape(empty_vec)
    _assert_embedding_vector_shape(non_empty_vec)

    # If both are non-empty, assert same dimensionality.
    if empty_vec and non_empty_vec:
        assert len(empty_vec) == len(non_empty_vec)


@pytest.mark.asyncio
async def test_async_embed_documents_and_query_basic(adapter: Any) -> None:
    """
    Async aembed_documents / aembed_query should be coroutine functions and
    produce shapes compatible with the sync API.
    """
    embeddings = configure_langchain_embeddings(
        corpus_adapter=adapter,
        model="async-model",
    )

    # Ensure we actually have async methods and they are coroutine functions
    assert hasattr(embeddings, "aembed_documents")
    assert hasattr(embeddings, "aembed_query")
    assert inspect.iscoroutinefunction(embeddings.aembed_documents)
    assert inspect.iscoroutinefunction(embeddings.aembed_query)

    texts = ["alpha-async", "beta-async"]
    query_text = "gamma-async"

    docs_result = await embeddings.aembed_documents(texts)
    _assert_embedding_matrix_shape(docs_result, expected_rows=len(texts))

    query_result = await embeddings.aembed_query(query_text)
    _assert_embedding_vector_shape(query_result)


@pytest.mark.asyncio
async def test_async_and_sync_same_dimension(adapter: Any) -> None:
    """
    Check that sync and async embeddings for the same input produce vectors
    of the same dimensionality (not necessarily identical values).

    NOTE: Some adapters enforce sync loop-guards. If your adapter raises when sync
    methods are called inside an event loop, run sync calls via asyncio.to_thread.
    """
    embeddings = configure_langchain_embeddings(
        corpus_adapter=adapter,
        model="dim-model",
    )

    texts = ["same-dim-1", "same-dim-2"]
    query = "same-dim-query"

    # Prefer safe parity pattern: run sync calls off-loop to avoid loop-guard deadlocks.
    sync_docs = await asyncio.to_thread(embeddings.embed_documents, texts)
    sync_query = await asyncio.to_thread(embeddings.embed_query, query)

    async_docs = await embeddings.aembed_documents(texts)
    async_query = await embeddings.aembed_query(query)

    # Compare dimensions (len of row vectors), if non-empty
    assert len(sync_docs) == len(async_docs) == len(texts)

    if sync_docs and async_docs:
        sync_dim = len(sync_docs[0])
        async_dim = len(async_docs[0])
        assert sync_dim == async_dim

    # Query dimensions
    assert len(sync_query) == len(async_query)


def test_large_batch_sync_shape(adapter: Any) -> None:
    """
    Large-ish batches should still produce N rows for N inputs.
    This is a light stress test around translator batching.
    """
    embeddings = configure_langchain_embeddings(
        corpus_adapter=adapter,
        model="large-batch-model",
    )

    texts = [f"text-{i}" for i in range(50)]
    result = embeddings.embed_documents(texts)
    _assert_embedding_matrix_shape(result, expected_rows=len(texts))


# ---------------------------------------------------------------------------
# Capabilities / health passthrough
# ---------------------------------------------------------------------------


def test_capabilities_and_health_passthrough_when_underlying_provides() -> None:
    """
    When the underlying adapter implements capabilities/acapabilities and
    health/ahealth, CorpusLangChainEmbeddings should surface them.
    """

    class FullAdapter:
        def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
            return [[0.0, 1.0] for _ in texts]

        def capabilities(self) -> Dict[str, Any]:
            return {"ok": True}

        async def acapabilities(self) -> Dict[str, Any]:
            return {"ok_async": True}

        def health(self) -> Dict[str, Any]:
            return {"status": "healthy"}

        async def ahealth(self) -> Dict[str, Any]:
            return {"status_async": "healthy"}

    embeddings = CorpusLangChainEmbeddings(
        corpus_adapter=FullAdapter(),
        model="cap-model",
    )

    # Sync passthrough
    caps = embeddings.capabilities()
    assert isinstance(caps, dict)
    assert caps.get("ok") is True

    health = embeddings.health()
    assert isinstance(health, dict)
    assert health.get("status") == "healthy"

    # Async passthrough via event loop
    acaps = asyncio.run(embeddings.acapabilities())
    assert isinstance(acaps, dict)
    assert acaps.get("ok_async") is True

    ahealth = asyncio.run(embeddings.ahealth())
    assert isinstance(ahealth, dict)
    assert ahealth.get("status_async") == "healthy"


@pytest.mark.asyncio
async def test_async_capabilities_and_health_fallback_to_sync() -> None:
    """
    acapabilities/ahealth should fall back to sync capabilities()/health()
    when only sync methods are implemented on the underlying adapter.
    """

    class CapHealthAdapter:
        def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
            return [[0.0, 1.0] for _ in texts]

        def capabilities(self) -> Dict[str, Any]:
            return {"via_sync_caps": True}

        def health(self) -> Dict[str, Any]:
            return {"via_sync_health": True}

    embeddings = CorpusLangChainEmbeddings(
        corpus_adapter=CapHealthAdapter(),
        model="cap-fallback-model",
    )

    acaps = await embeddings.acapabilities()
    assert isinstance(acaps, dict)
    assert acaps.get("via_sync_caps") is True

    ahealth = await embeddings.ahealth()
    assert isinstance(ahealth, dict)
    assert ahealth.get("via_sync_health") is True


def test_capabilities_and_health_return_empty_when_missing() -> None:
    """
    If the underlying adapter has no capabilities()/health(), the LangChain
    adapter should return an empty dict rather than raising.
    """

    class NoCapHealthAdapter:
        def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
            return [[0.0] * 3 for _ in texts]

    embeddings = CorpusLangChainEmbeddings(
        corpus_adapter=NoCapHealthAdapter(),
        model="no-cap-health-model",
    )

    caps = embeddings.capabilities()
    assert isinstance(caps, dict)
    assert caps == {}

    health = embeddings.health()
    assert isinstance(health, dict)
    assert health == {}

    # Async variants should also return empty mapping
    acaps = asyncio.run(embeddings.acapabilities())
    assert isinstance(acaps, dict)
    assert acaps == {}

    ahealth = asyncio.run(embeddings.ahealth())
    assert isinstance(ahealth, dict)
    assert ahealth == {}



# ---------------------------------------------------------------------------
# Resource management (context managers)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_manager_closes_underlying_adapter() -> None:
    """
    __enter__/__exit__ and __aenter__/__aexit__ should call close/aclose on
    the underlying adapter when those methods exist.
    """
    # IMPORTANT:
    # The Corpus embedding stack calls adapter.embed() with an EmbedSpec-like object,
    # and expects an EmbedResult / BatchEmbedResult flow. So this test MUST use a
    # protocol-conformant adapter. We wrap the existing mock adapter and only add
    # close()/aclose() flags.
    from tests.mock.mock_embedding_adapter import MockEmbeddingAdapter

    class SyncClosingAdapter(MockEmbeddingAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class AsyncClosingAdapter(MockEmbeddingAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.aclosed = False

        async def aclose(self) -> None:
            self.aclosed = True

    # Sync context manager
    adapter = SyncClosingAdapter()

    # IMPORTANT:
    # This test is async, so we must not call sync APIs inside the running event loop.
    # Run the sync context manager + sync embed_documents off-loop.
    def _run_sync_context_manager() -> None:
        with CorpusLangChainEmbeddings(corpus_adapter=adapter, model="ctx-model") as emb:
            _ = emb.embed_documents(["x"])  # smoke

    await asyncio.to_thread(_run_sync_context_manager)

    assert adapter.closed is True

    # Async context manager
    adapter2 = AsyncClosingAdapter()
    emb2 = CorpusLangChainEmbeddings(corpus_adapter=adapter2, model="ctx-model-2")

    async with emb2:
        _ = await emb2.aembed_documents(["y"])

    assert adapter2.aclosed is True


# ---------------------------------------------------------------------------
# Concurrency tests
# ---------------------------------------------------------------------------


@pytest.mark.concurrency
def test_shared_embedder_thread_safety(adapter: Any) -> None:
    """
    Shared embedder is thread-safe for concurrent access.
    """
    embedder = configure_langchain_embeddings(
        corpus_adapter=adapter,
        model="concurrent-model",
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
        assert all(isinstance(x, (int, float)) for x in result)


@pytest.mark.asyncio
@pytest.mark.concurrency
async def test_concurrent_async_embedding(adapter: Any) -> None:
    """
    Async embedding supports concurrent operations.
    """
    embedder = configure_langchain_embeddings(
        corpus_adapter=adapter,
        model="async-concurrent-model",
    )

    async def embed_async(text: str) -> Any:
        return await embedder.aembed_query(text)

    texts = [f"async query {i}" for i in range(5)]
    tasks = [embed_async(text) for text in texts]
    results = await asyncio.gather(*tasks)

    assert len(results) == len(texts)
    for result in results:
        assert isinstance(result, list)
        assert all(isinstance(x, (int, float)) for x in result)


# ---------------------------------------------------------------------------
# Integration tests with real LangChain objects
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestLangChainIntegration:
    """
    Integration tests with real LangChain objects.

    Policy (Path A):
    - No pytest.skip inside the integration tests.
    - Pass/fail gating is handled by require_langchain fixture.
    """

    def test_can_use_with_langchain_embeddings_base(
        self,
        require_langchain: bool,
        adapter: Any,
    ) -> None:
        """
        Integration: CorpusLangChainEmbeddings should be usable anywhere
        LangChain's Embeddings base class is expected.
        """
        from langchain_core.embeddings import Embeddings  # type: ignore[import]

        embedder = configure_langchain_embeddings(
            corpus_adapter=adapter,
            model="integration-model",
        )

        assert isinstance(embedder, Embeddings)

        docs = ["LangChain is a framework.", "Embeddings convert text to vectors."]
        doc_vecs = embedder.embed_documents(docs)
        _assert_embedding_matrix_shape(doc_vecs, expected_rows=len(docs))

        query_vec = embedder.embed_query("What is LangChain?")
        _assert_embedding_vector_shape(query_vec)

    def test_embeddings_work_in_runnable_chain(
        self,
        require_langchain: bool,
        adapter: Any,
    ) -> None:
        """
        Integration: Embeddings can be used inside a simple Runnable chain,
        if langchain_core is available.
        """
        from langchain_core.runnables import RunnableLambda  # type: ignore[import]

        embedder = configure_langchain_embeddings(
            corpus_adapter=adapter,
            model="chain-model",
        )

        chain = RunnableLambda(lambda s: embedder.embed_query(s))
        vec = chain.invoke("test query in chain")
        _assert_embedding_vector_shape(vec)

    def test_integration_error_propagation_is_actionable(
        self,
        require_langchain: bool,
    ) -> None:
        """
        Integration: Errors from a failing adapter used in a LangChain-style
        workflow are actionable (e.g., rate limit / API key guidance).
        """

        class FailingAdapter:
            def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
                raise RuntimeError(
                    "API limit reached: Please upgrade your plan or wait before retrying",
                )

        failing_embedder = CorpusLangChainEmbeddings(
            corpus_adapter=FailingAdapter(),
            model="failing-model",
        )

        with pytest.raises(RuntimeError) as exc_info:
            failing_embedder.embed_documents(["test"])

        error_str = str(exc_info.value).lower()
        assert "api limit" in error_str or "rate limit" in error_str
        assert "upgrade" in error_str or "wait" in error_str or "retry" in error_str
