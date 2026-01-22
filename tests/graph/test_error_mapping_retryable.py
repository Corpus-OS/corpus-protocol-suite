# SPDX-License-Identifier: Apache-2.0
"""
Graph Conformance — Error mapping & retry hints.

Asserts (Spec refs):
  • Retryable errors include retry_after_ms hints             (§6.3, §12.1, §12.4)
  • Unknown dialects rejected with NotSupported               (§7.4)
  • Input validation surfaces BadRequest                      (§17.2, §7.4)
  • Normalized error taxonomy codes are stable                (§12.4)
"""
from __future__ import annotations

import pytest

from corpus_sdk.graph.graph_base import (
    OperationContext as GraphContext,
    GraphAdapterError,
    # Normalized errors
    BadRequest,
    NotSupported,
    AuthError,
    ResourceExhausted,
    TransientNetwork,
    Unavailable,
    DeadlineExceeded,
    # Ops/types used by adapter-driven checks
    GraphQuerySpec,
    UpsertEdgesSpec,
    Edge,
    GraphID,
    BaseGraphAdapter,
)

# NOTE: Do NOT set a global pytestmark=asyncio here because this file contains
# both sync taxonomy-level tests and async adapter-surface tests. Mark only the
# async tests to avoid PytestWarning spam.

# ---------------------------------------------------------------------------
# Taxonomy-level (adapter-independent) checks
# ---------------------------------------------------------------------------

def test_error_handling_retryable_errors_with_hints():
    err = GraphAdapterError("retryable", retry_after_ms=123)
    assert isinstance(err.retry_after_ms, int) and err.retry_after_ms >= 0


def test_graph_adapter_error_details_is_mapping():
    err = GraphAdapterError("x", details={"k": "v"})
    assert isinstance(err.details, dict)
    assert err.details["k"] == "v"


@pytest.mark.parametrize(
    "exc, expected_code",
    [
        (BadRequest("x"), "BAD_REQUEST"),
        (AuthError("x"), "AUTH_ERROR"),
        (ResourceExhausted("x"), "RESOURCE_EXHAUSTED"),
        (TransientNetwork("x"), "TRANSIENT_NETWORK"),
        (Unavailable("x"), "UNAVAILABLE"),
        (NotSupported("x"), "NOT_SUPPORTED"),
        (DeadlineExceeded("x"), "DEADLINE_EXCEEDED"),
    ],
)
def test_normalized_error_default_codes(exc: GraphAdapterError, expected_code: str):
    """
    Validate that each normalized error subclass sets a stable default .code.
    This is critical for machine-actionable routing and wire mapping.
    """
    assert isinstance(exc.code, str)
    assert exc.code == expected_code


@pytest.mark.parametrize(
    "exc_type",
    [Unavailable, ResourceExhausted, TransientNetwork],
)
def test_retryable_error_types_accept_retry_after_and_details(exc_type):
    """
    Retryable-ish errors should be able to carry retry_after_ms and SIEM-safe details.
    (Even if not every backend uses all of them, the type contract must support it.)
    """
    e = exc_type("try later", retry_after_ms=500, details={"hint": "backoff"})
    assert e.retry_after_ms == 500
    assert isinstance(e.details, dict)
    assert e.details.get("hint") == "backoff"


def test_error_string_includes_code_when_present():
    """
    Keep this light: just ensure __str__ includes the code tag when present.
    Avoid overfitting exact formatting.
    """
    e = Unavailable("down")
    s = str(e)
    assert "code=" in s


# ---------------------------------------------------------------------------
# Adapter-surface checks (capability-driven)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_error_handling_bad_request_on_empty_edge_label(adapter: BaseGraphAdapter):
    """
    Invalid edge label must surface BadRequest from BaseGraphAdapter validation.
    """
    ctx = GraphContext(request_id="t_err_badreq", tenant="t")
    spec = UpsertEdgesSpec(
        edges=[
            Edge(
                id=GraphID("e1"),
                src=GraphID("v:1"),
                dst=GraphID("v:2"),
                label="",  # invalid
                properties={},
                namespace="ns",
            )
        ],
        namespace="ns",
    )
    with pytest.raises(BadRequest):
        await adapter.upsert_edges(spec, ctx=ctx)


@pytest.mark.asyncio
async def test_not_supported_on_unknown_dialect_when_declared(adapter: BaseGraphAdapter):
    """
    If supported_query_dialects is declared non-empty, unknown dialects must raise NotSupported.
    If dialect list is empty/opaque, adapter may accept or reject; if it rejects, NotSupported is acceptable.
    """
    caps = await adapter.capabilities()
    declared = tuple(getattr(caps, "supported_query_dialects", ()) or ())
    ctx = GraphContext(request_id="t_err_notsup_dialect", tenant="t")

    unknown = "__sparql_like__"
    spec = GraphQuerySpec(text="RETURN 1", dialect=unknown)

    if declared and unknown not in declared:
        with pytest.raises(NotSupported):
            await adapter.query(spec, ctx=ctx)
        return

    # Opaque dialect set: allow either success or NotSupported.
    try:
        await adapter.query(spec, ctx=ctx)
    except NotSupported:
        pass


@pytest.mark.asyncio
async def test_error_message_includes_dialect_name_when_rejected_due_to_declared_list(adapter: BaseGraphAdapter):
    """
    When dialect is rejected due to declared supported_query_dialects, include the dialect in the message.
    """
    caps = await adapter.capabilities()
    declared = tuple(getattr(caps, "supported_query_dialects", ()) or ())
    if not declared:
        return

    dialect = "__nope__"
    if dialect in declared:
        return

    ctx = GraphContext(request_id="t_err_dialect_msg", tenant="t")
    with pytest.raises(NotSupported) as ei:
        await adapter.query(GraphQuerySpec(text="RETURN 1", dialect=dialect), ctx=ctx)
    assert dialect in str(ei.value)
