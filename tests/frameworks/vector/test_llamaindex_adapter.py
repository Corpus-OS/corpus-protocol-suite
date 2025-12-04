# tests/frameworks/vector/test_llamaindex_adapter.py

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict

import inspect
import pytest

import corpus_sdk.vector.framework_adapters.llamaindex as llamaindex_adapter_module
from corpus_sdk.vector.framework_adapters.llamaindex import (
    CorpusLlamaIndexVectorStore,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class DummyVectorStoreQuery:
    """
    Minimal stand-in for llama_index.core.vector_stores.types.VectorStoreQuery.

    The adapter only relies on these attributes:
    - query_embedding
    - similarity_top_k
    - filters
    - doc_ids
    - node_ids
    """

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


def _make_store(adapter: Any, **kwargs: Any) -> CorpusLlamaIndexVectorStore:
    """
    Construct a CorpusLlamaIndexVectorStore instance from the generic adapter.
    """
    return CorpusLlamaIndexVectorStore(corpus_adapter=adapter, **kwargs)


def _make_fake_node(
    node_id: str = "n-1",
    ref_doc_id: str = "doc-1",
    embedding: Sequence[float] | None = None,
    text: str = "node-text",
    metadata: dict[str, Any] | None = None,
) -> Any:
    """
    Minimal BaseNode-like object for add()/aadd() tests.
    """

    class FakeNode:
        def __init__(self) -> None:
            self.node_id = node_id
            self.ref_doc_id = ref_doc_id
            self._embedding = list(embedding or [0.1, 0.2, 0.3])
            self._text = text
            self.metadata = dict(metadata or {})

        def get_embedding(self) -> Sequence[float]:
            return self._embedding

        def get_content(self, metadata_mode: Any | None = None) -> str:  # noqa: ARG002
            return self._text

    return FakeNode()


# ---------------------------------------------------------------------------
# 1. Translator wiring / default construction
# ---------------------------------------------------------------------------


def test_default_translator_constructed_with_llamaindex_framework(adapter: Any) -> None:
    """
    CorpusLlamaIndexVectorStore should construct a VectorTranslator with
    framework='llamaindex' and the given corpus adapter.
    """
    captured: Dict[str, Any] = {}

    # Patch VectorTranslator.__init__ inside the adapter module.
    VectorTranslator = llamaindex_adapter_module.VectorTranslator  # type: ignore[attr-defined]

    def fake_init(self, adapter: Any, framework: str, translator: Any) -> None:  # noqa: D401, ANN001
        captured["adapter"] = adapter
        captured["framework"] = framework
        captured["translator"] = translator
        # No need to call the real __init__ for this test.

    orig_init = VectorTranslator.__init__
    try:
        VectorTranslator.__init__ = fake_init  # type: ignore[assignment]
        store = _make_store(adapter)
        _ = store._translator  # noqa: SLF001 - trigger cached_property
    finally:
        VectorTranslator.__init__ = orig_init  # type: ignore[assignment]

    assert captured["adapter"] is adapter
    assert captured["framework"] == "llamaindex"
    # Translator should be some framework translator instance
    assert captured["translator"] is not None


# ---------------------------------------------------------------------------
# 2. Context building & framework_ctx wiring
# ---------------------------------------------------------------------------


def test_build_ctx_uses_dict_via_ctx_from_dict(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """
    When ctx is a dict, _build_ctx should call ctx_from_dict and use its return
    value as the OperationContext.
    """
    captured: Dict[str, Any] = {}

    class DummyOperationContext:
        def __init__(self, **kwargs: Any) -> None:
            self.attrs = kwargs

    monkeypatch.setattr(
        llamaindex_adapter_module,
        "OperationContext",
        DummyOperationContext,
    )

    def fake_ctx_from_dict(d: Mapping[str, Any]) -> DummyOperationContext:
        captured["from_dict"] = dict(d)
        return DummyOperationContext(source="from_dict")

    monkeypatch.setattr(
        llamaindex_adapter_module,
        "ctx_from_dict",
        fake_ctx_from_dict,
    )

    store = _make_store(adapter)

    # Use private helper directly to avoid translator side-effects.
    ctx = store._build_ctx(ctx={"request_id": "req-1"})  # noqa: SLF001
    assert isinstance(ctx, DummyOperationContext)
    assert captured["from_dict"] == {"request_id": "req-1"}


def test_build_ctx_uses_callback_manager_when_no_ctx(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """
    When ctx is not provided but callback_manager is, _build_ctx should call
    ctx_from_llamaindex.
    """
    captured: Dict[str, Any] = {}

    class DummyOperationContext:
        def __init__(self, **kwargs: Any) -> None:
            self.attrs = kwargs

    monkeypatch.setattr(
        llamaindex_adapter_module,
        "OperationContext",
        DummyOperationContext,
    )

    def fake_ctx_from_llamaindex(callback_manager: Any) -> DummyOperationContext:
        captured["callback_manager"] = callback_manager
        return DummyOperationContext(source="from_llamaindex")

    monkeypatch.setattr(
        llamaindex_adapter_module,
        "ctx_from_llamaindex",
        fake_ctx_from_llamaindex,
    )

    store = _make_store(adapter)

    cb_manager = object()
    ctx = store._build_ctx(callback_manager=cb_manager)  # noqa: SLF001
    assert isinstance(ctx, DummyOperationContext)
    assert captured["callback_manager"] is cb_manager


def test_framework_ctx_for_namespace_uses_normalize_and_attach(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """
    _framework_ctx_for_namespace should normalize vector context and then
    attach it to framework_ctx via attach_vector_context_to_framework_ctx.
    """
    captured: Dict[str, Any] = {}

    def fake_normalize_vector_context(raw_ctx: Mapping[str, Any], framework: str, logger: Any) -> Dict[str, Any]:  # noqa: ARG001
        captured["raw_ctx"] = dict(raw_ctx)
        captured["framework"] = framework
        # Inject an extra flag into the normalized context.
        return {"normalized": True, **raw_ctx}

    def fake_attach_vector_context_to_framework_ctx(
        framework_ctx: Dict[str, Any],
        *,
        vector_context: Mapping[str, Any],
        limits: Any,  # noqa: ARG001
        flags: Any,  # noqa: ARG001
    ) -> None:
        framework_ctx.update(vector_context)
        captured["framework_ctx_after_attach"] = dict(framework_ctx)

    monkeypatch.setattr(
        llamaindex_adapter_module,
        "normalize_vector_context",
        fake_normalize_vector_context,
    )
    monkeypatch.setattr(
        llamaindex_adapter_module,
        "attach_vector_context_to_framework_ctx",
        fake_attach_vector_context_to_framework_ctx,
    )

    store = _make_store(adapter, namespace="default-ns")

    fw_ctx = store._framework_ctx_for_namespace("explicit-ns")  # noqa: SLF001

    assert captured["framework"] == "llamaindex"
    assert captured["raw_ctx"] == {"namespace": "explicit-ns"}
    assert fw_ctx["normalized"] is True
    assert fw_ctx["namespace"] == "explicit-ns"


# ---------------------------------------------------------------------------
# 3. Capabilities validation (top_k, filters)
# ---------------------------------------------------------------------------


def test_sync_query_top_k_exceeds_caps_raises_badrequest_with_context(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    If top_k exceeds caps.max_top_k, query() should raise BadRequest and call
    attach_context with useful metadata.
    """
    captured_ctx: Dict[str, Any] = {}

    class DummyTranslator:
        def capabilities(self) -> Any:
            class Caps:
                max_top_k = 10
                supports_metadata_filtering = True

            return Caps()

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    monkeypatch.setattr(
        llamaindex_adapter_module,
        "attach_context",
        fake_attach_context,
    )

    store = _make_store(adapter)
    # Override translator instance.
    store._translator = DummyTranslator()  # type: ignore[assignment, attr-defined]  # noqa: SLF001

    query = DummyVectorStoreQuery(
        query_embedding=[0.1, 0.2, 0.3],
        similarity_top_k=100,  # exceeds max_top_k
    )

    with pytest.raises(Exception) as exc_info:
        store.query(query)

    err = exc_info.value
    # Don't assert exact type, but it should look like a BadRequest-style error.
    assert "BAD_TOP_K" in getattr(err, "code", "") or "top_k" in str(err).lower()

    assert captured_ctx.get("framework") == "llamaindex"
    assert captured_ctx.get("operation") == "query_sync"
    assert captured_ctx.get("top_k") == 100


def test_sync_query_metadata_filter_not_supported_raises_notsupported(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    If metadata filters are provided but capabilities say filtering is not
    supported, query() should raise NotSupported and attach context.
    """
    captured_ctx: Dict[str, Any] = {}

    class DummyTranslator:
        def capabilities(self) -> Any:
            class Caps:
                max_top_k = None
                supports_metadata_filtering = False

            return Caps()

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    monkeypatch.setattr(
        llamaindex_adapter_module,
        "attach_context",
        fake_attach_context,
    )

    store = _make_store(adapter)
    store._translator = DummyTranslator()  # type: ignore[assignment, attr-defined]  # noqa: SLF001

    # Minimal filter object with .filters list and .condition.
    class DummyFilters:
        def __init__(self) -> None:
            self.filters = []
            self.condition = None

    q = DummyVectorStoreQuery(
        query_embedding=[0.1, 0.2],
        similarity_top_k=4,
        filters=DummyFilters(),
    )

    with pytest.raises(Exception) as exc_info:
        store.query(q, namespace="ns-filters")

    err = exc_info.value
    assert "FILTER_NOT_SUPPORTED" in getattr(err, "code", "") or "filter" in str(err).lower()

    assert captured_ctx.get("framework") == "llamaindex"
    assert captured_ctx.get("operation") == "query_sync"
    assert captured_ctx.get("namespace") == "ns-filters"


# ---------------------------------------------------------------------------
# 4. Delete validation (ids / filters)
# ---------------------------------------------------------------------------


def test_delete_by_ref_doc_id_respects_filter_capabilities(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    delete(ref_doc_id=...) internally uses a metadata filter; if the backend
    does not support metadata filtering, it should raise NotSupported and
    attach context.
    """
    captured_ctx: Dict[str, Any] = {}

    class DummyTranslator:
        def capabilities(self) -> Any:
            class Caps:
                max_top_k = None
                supports_metadata_filtering = False

            return Caps()

        def delete(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            pytest.fail("translator.delete should not be called when validation fails")

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    monkeypatch.setattr(
        llamaindex_adapter_module,
        "attach_context",
        fake_attach_context,
    )

    store = _make_store(adapter)
    store._translator = DummyTranslator()  # type: ignore[assignment, attr-defined]  # noqa: SLF001

    with pytest.raises(Exception) as exc_info:
        store.delete("doc-xyz", namespace="del-ns")

    err = exc_info.value
    assert "FILTER_NOT_SUPPORTED" in getattr(err, "code", "") or "filter" in str(err).lower()

    assert captured_ctx.get("framework") == "llamaindex"
    assert captured_ctx.get("operation") == "delete_sync"
    assert captured_ctx.get("namespace") == "del-ns"


def test_delete_nodes_empty_noop(adapter: Any) -> None:
    """
    delete_nodes([]) should be a no-op and must not raise.
    """
    store = _make_store(adapter)
    store.delete_nodes([])  # Should not raise.


# ---------------------------------------------------------------------------
# 5. Upsert / add behavior
# ---------------------------------------------------------------------------


def test_add_builds_vectors_and_calls_translator(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """
    add() should:

    - Translate nodes into Vector objects via _nodes_to_corpus_vectors,
    - Build an upsert request mapping with namespace and vectors,
    - Call translator.upsert with op_ctx/framework_ctx, and
    - Return node IDs as strings.
    """
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def upsert(self, raw_request: Mapping[str, Any], *, op_ctx: Any, framework_ctx: Mapping[str, Any]) -> Any:  # noqa: D401, ARG001
            captured["raw_request"] = dict(raw_request)
            captured["op_ctx"] = op_ctx
            captured["framework_ctx"] = dict(framework_ctx)
            # Return something that is not an UpsertResult so _handle_partial_upsert_failure is skipped.
            return object()

    store = _make_store(adapter, namespace="default-ns")
    store._translator = DummyTranslator()  # type: ignore[assignment, attr-defined]  # noqa: SLF001

    node1 = _make_fake_node(node_id="n1", text="t1")
    node2 = _make_fake_node(node_id="n2", text="t2")
    ids = store.add([node1, node2], namespace="add-ns")

    assert ids == ["n1", "n2"]

    raw = captured["raw_request"]
    assert raw["namespace"] == "add-ns"
    assert isinstance(raw["vectors"], list)
    assert len(raw["vectors"]) == 2

    fw_ctx = captured["framework_ctx"]
    assert fw_ctx.get("framework") == "llamaindex"
    assert fw_ctx.get("namespace") == "add-ns"


def test_add_raises_badrequest_when_node_missing_embedding(adapter: Any) -> None:
    """
    If a node has no embedding, add() should raise a BadRequest-like error
    with code NO_EMBEDDING.
    """

    class NodeWithoutEmbedding:
        def __init__(self) -> None:
            self.node_id = "missing-emb"

        def get_content(self, metadata_mode: Any | None = None) -> str:  # noqa: ARG002
            return "text"

    store = _make_store(adapter)

    with pytest.raises(Exception) as exc_info:
        store.add([NodeWithoutEmbedding()])

    err = exc_info.value
    assert "NO_EMBEDDING" in getattr(err, "code", "") or "embedding" in str(err).lower()


def test_handle_partial_upsert_failure_raises_vectoradaptererror() -> None:
    """
    _handle_partial_upsert_failure should raise VectorAdapterError when all
    nodes fail to upsert.
    """
    store = _make_store(adapter=object())  # adapter not used here

    class DummyResult:
        def __init__(self) -> None:
            self.upserted_count = 0
            self.failed_count = 3
            self.failures = [{"id": "n1"}, {"id": "n2"}, {"id": "n3"}]

    result = DummyResult()

    with pytest.raises(Exception) as exc_info:
        store._handle_partial_upsert_failure(result, total_nodes=3, namespace="ns")  # noqa: SLF001

    err = exc_info.value
    assert "BATCH_UPSERT_FAILED" in getattr(err, "code", "") or "upsert" in str(err).lower()


# ---------------------------------------------------------------------------
# 6. Query / stream / MMR wiring + error context
# ---------------------------------------------------------------------------


def test_sync_query_basic_smoke_uses_translator(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """
    Basic smoke test that query() calls translator.query with a raw request and
    returns a VectorStoreQueryResult-like object via a stubbed validator.
    """
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def capabilities(self) -> Any:
            class Caps:
                max_top_k = None
                supports_metadata_filtering = True

            return Caps()

        def query(self, raw_request: Mapping[str, Any], *, op_ctx: Any, framework_ctx: Mapping[str, Any]) -> Any:  # noqa: D401, ARG001
            captured["raw_request"] = dict(raw_request)
            captured["op_ctx"] = op_ctx
            captured["framework_ctx"] = dict(framework_ctx)

            class DummyResult:
                def __init__(self) -> None:
                    self.matches = []

            return DummyResult()

    def fake_validate(self, result: Any, *, operation: str) -> Any:  # noqa: D401, ARG002
        # Return the result unchanged so query() can proceed.
        return result

    monkeypatch.setattr(
        CorpusLlamaIndexVectorStore,
        "_validate_query_result_type",
        fake_validate,
    )

    store = _make_store(adapter, namespace="store-ns")
    store._translator = DummyTranslator()  # type: ignore[assignment, attr-defined]  # noqa: SLF001

    q = DummyVectorStoreQuery(query_embedding=[0.1, 0.2, 0.3], similarity_top_k=4)
    result = store.query(q, namespace="q-ns")

    # We don't assert exact type, but it should have basic VectorStoreQueryResult attrs.
    assert hasattr(result, "nodes")
    assert hasattr(result, "similarities")
    assert hasattr(result, "ids")

    raw = captured["raw_request"]
    assert raw["namespace"] == "q-ns"
    assert raw["top_k"] == 4
    assert raw["include_metadata"] is True
    assert raw["include_vectors"] is False


def test_query_stream_invalid_chunk_type_raises_vectoradaptererror_with_context(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    If translator.query_stream yields a non-QueryChunk object, query_stream()
    should raise a VectorAdapterError and attach context via attach_context.
    """
    captured_ctx: Dict[str, Any] = {}

    class BadChunkTranslator:
        def capabilities(self) -> Any:
            class Caps:
                max_top_k = None
                supports_metadata_filtering = True

            return Caps()

        def query_stream(self, raw_request: Mapping[str, Any], *, op_ctx: Any, framework_ctx: Mapping[str, Any]):  # noqa: D401, ARG001
            # Emit a blatantly wrong type as the first "chunk".
            yield "not-a-query-chunk"

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    # We don't need validator for stream chunks here; the adapter itself
    # does the isinstance(QueryChunk) check.
    monkeypatch.setattr(
        llamaindex_adapter_module,
        "attach_context",
        fake_attach_context,
    )

    store = _make_store(adapter, namespace="stream-ns")
    store._translator = BadChunkTranslator()  # type: ignore[assignment, attr-defined]  # noqa: SLF001

    q = DummyVectorStoreQuery(query_embedding=[0.1, 0.2], similarity_top_k=2)

    with pytest.raises(Exception) as exc_info:
        for _ in store.query_stream(q, namespace="stream-ns"):  # consume one chunk
            break

    err = exc_info.value
    assert "BAD_STREAM_CHUNK" in getattr(err, "code", "") or "chunk" in str(err).lower()

    assert captured_ctx.get("framework") == "llamaindex"
    assert captured_ctx.get("operation") == "query_stream"
    assert captured_ctx.get("namespace") == "stream-ns"


def test_query_mmr_rejects_invalid_lambda_and_attaches_context(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    query_mmr() should validate lambda_mult in [0, 1] and attach context when
    invalid.
    """
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    monkeypatch.setattr(
        llamaindex_adapter_module,
        "attach_context",
        fake_attach_context,
    )

    store = _make_store(adapter)

    q = DummyVectorStoreQuery(query_embedding=[0.1, 0.2, 0.3], similarity_top_k=4)

    with pytest.raises(Exception) as exc_info:
        store.query_mmr(q, lambda_mult=2.0)  # invalid

    err = exc_info.value
    assert "BAD_MMR_LAMBDA" in getattr(err, "code", "") or "lambda" in str(err).lower()

    assert captured_ctx.get("framework") == "llamaindex"
    assert captured_ctx.get("operation") == "query_mmr_sync"
    assert captured_ctx.get("lambda_mult") == 2.0


# ---------------------------------------------------------------------------
# 7. Score threshold behavior
# ---------------------------------------------------------------------------


def test_score_threshold_filters_matches_after_query_validation(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    When score_threshold is set, results below the threshold should be filtered
    out after translator.query returns.
    """
    class DummyMatch:
        def __init__(self, score: float) -> None:
            self.score = score

            class DummyVector:
                def __init__(self) -> None:
                    self.metadata = {}
                    self.id = "id"

                vector = []

            self.vector = DummyVector()

    class DummyResult:
        def __init__(self) -> None:
            self.matches = [DummyMatch(0.2), DummyMatch(0.5), DummyMatch(0.9)]

    class DummyTranslator:
        def capabilities(self) -> Any:
            class Caps:
                max_top_k = None
                supports_metadata_filtering = True

            return Caps()

        def query(self, raw_request: Mapping[str, Any], *, op_ctx: Any, framework_ctx: Mapping[str, Any]) -> Any:  # noqa: D401, ARG001
            return DummyResult()

    def fake_validate(self, result: Any, *, operation: str) -> Any:  # noqa: D401, ARG002
        return result

    monkeypatch.setattr(
        CorpusLlamaIndexVectorStore,
        "_validate_query_result_type",
        fake_validate,
    )

    # Only keep scores >= 0.6
    store = _make_store(adapter, score_threshold=0.6)
    store._translator = DummyTranslator()  # type: ignore[assignment, attr-defined]  # noqa: SLF001

    q = DummyVectorStoreQuery(query_embedding=[0.1, 0.2, 0.3], similarity_top_k=10)
    result = store.query(q)

    # Only the 0.9 match should survive; Node reconstruction is stubbed via TextNode fallback,
    # but we only care about similarities length here.
    assert len(result.similarities) == 1


# ---------------------------------------------------------------------------
# 8. Node ↔ Vector translation
# ---------------------------------------------------------------------------


def test_nodes_to_corpus_vectors_populates_reserved_metadata(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    _nodes_to_corpus_vectors should:

    - Call node_to_metadata_dict,
    - Populate node_id_field/ref_doc_id_field/text_field/id_field,
    - Use effective namespace.
    """
    captured: Dict[str, Any] = {}

    def fake_node_to_metadata_dict(node: Any, remove_text: bool, mode: Any) -> Dict[str, Any]:  # noqa: ARG002
        captured["node"] = node
        return {"custom": "value"}

    monkeypatch.setattr(
        llamaindex_adapter_module,
        "node_to_metadata_dict",
        fake_node_to_metadata_dict,
    )

    store = _make_store(
        adapter,
        namespace="store-ns",
        id_field="id",
        text_field="text",
        node_id_field="node_id",
        ref_doc_id_field="ref_doc_id",
    )

    node = _make_fake_node(node_id="node-123", ref_doc_id="doc-abc", text="hello")
    vectors = store._nodes_to_corpus_vectors([node], namespace="ns-override")  # noqa: SLF001

    assert len(vectors) == 1
    v = vectors[0]
    assert v.namespace == "ns-override"
    assert v.id == "node-123"

    meta = v.metadata or {}
    assert meta["custom"] == "value"
    assert meta["node_id"] == "node-123"
    assert meta["ref_doc_id"] == "doc-abc"
    assert meta["text"] == "hello"
    assert meta["id"] == "node-123"


def test_matches_to_nodes_uses_metadata_dict_to_node_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    _matches_to_nodes should reconstruct nodes from metadata via
    metadata_dict_to_node / legacy_metadata_dict_to_node / TextNode fallback.
    """
    captured: Dict[str, Any] = {}

    class DummyNode:
        def __init__(self, text: str, node_id: str) -> None:
            self.text = text
            self.node_id = node_id

    def fake_metadata_dict_to_node(meta: Mapping[str, Any], *, text: str | None, node_id: str) -> Any:  # noqa: D401
        captured["meta"] = dict(meta)
        captured["text"] = text
        captured["node_id"] = node_id
        return DummyNode(text=text or "", node_id=node_id)

    # Ensure legacy path is not used in this test.
    def fake_legacy_metadata_dict_to_node(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        pytest.fail("legacy_metadata_dict_to_node should not be called in this test")

    monkeypatch.setattr(
        llamaindex_adapter_module,
        "metadata_dict_to_node",
        fake_metadata_dict_to_node,
    )
    monkeypatch.setattr(
        llamaindex_adapter_module,
        "legacy_metadata_dict_to_node",
        fake_legacy_metadata_dict_to_node,
    )

    store = _make_store(
        adapter,
        text_field="text",
        node_id_field="node_id",
        ref_doc_id_field="ref_doc_id",
    )

    class DummyVector:
        def __init__(self) -> None:
            self.id = "node-xyz"
            self.metadata = {
                "text": "hello-world",
                "node_id": "node-xyz",
                "ref_doc_id": "doc-123",
                "foo": "bar",
            }

    class DummyMatch:
        def __init__(self) -> None:
            self.vector = DummyVector()
            self.score = 0.99

    nodes_with_scores = store._matches_to_nodes([DummyMatch()])  # noqa: SLF001
    assert len(nodes_with_scores) == 1
    nws = nodes_with_scores[0]

    assert isinstance(nws.node, DummyNode)
    assert nws.score == pytest.approx(0.99)

    assert captured["node_id"] == "node-xyz"
    assert captured["text"] == "hello-world"


# ---------------------------------------------------------------------------
# 9. Async vs sync parity (basic)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_and_sync_query_return_compatible_shapes(adapter: Any) -> None:
    """
    Basic parity check that query() and aquery() both succeed and return
    VectorStoreQueryResult-like objects for the same input.
    """
    store = _make_store(adapter)

    # We rely on the real translator + test adapter here; generic contract
    # tests already validate deep semantics, so this is just a smoke check.
    q = DummyVectorStoreQuery(query_embedding=[0.1, 0.2, 0.3], similarity_top_k=2)

    sync_result = store.query(q)
    assert hasattr(sync_result, "nodes")

    coro = store.aquery(q)
    assert inspect.isawaitable(coro)
    async_result = await coro
    assert hasattr(async_result, "nodes")

    # At minimum, both should expose .ids with the same length.
    assert isinstance(sync_result.ids, Sequence)
    assert isinstance(async_result.ids, Sequence)
    assert len(sync_result.ids) == len(async_result.ids)
