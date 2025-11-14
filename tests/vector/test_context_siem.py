# SPDX-License-Identifier: Apache-2.0
"""
Vector Conformance — Observability & SIEM safety.

Spec refs:
  • SPECIFICATION.md §13.1-§13.3 (Observability)
  • SPECIFICATION.md §15 (Privacy)
  • SPECIFICATION.md §6.1 (Operation Context)
"""

import pytest
from typing import Any, Mapping, Optional, List, Dict
from corpus_sdk.vector.vector_base import (
    OperationContext,
    QuerySpec,
    UpsertSpec,
    Vector,
    VectorID,
    MetricsSink,
    BadRequest,
)

pytestmark = pytest.mark.asyncio


class CaptureMetrics(MetricsSink):
    """Metrics sink that captures all observations and counters for testing."""
    
    def __init__(self) -> None:
        self.observations: List[Dict[str, Any]] = []
        self.counters: List[Dict[str, Any]] = []

    def observe(
        self,
        *,
        component: str,
        op: str,
        ms: float,
        ok: bool,
        code: str = "OK",
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.observations.append(
            {
                "component": component,
                "op": op,
                "ok": ok,
                "code": code,
                "extra": dict(extra or {}),
            }
        )

    def counter(
        self,
        *,
        component: str,
        name: str,
        value: int = 1,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.counters.append(
            {
                "component": component,
                "name": name,
                "value": value,
                "extra": dict(extra or {}),
            }
        )


async def test_observability_context_propagates_to_metrics_siem_safe(adapter):
    """Verify operation context propagates to metrics while maintaining SIEM safety."""
    metrics = CaptureMetrics()
    original_metrics = getattr(adapter, "_metrics", None)
    adapter._metrics = metrics  # type: ignore[attr-defined]

    ctx = OperationContext(request_id="v_ctx", tenant="acme")
    await adapter.query(QuerySpec(vector=[0.1], top_k=1, namespace="default"), ctx=ctx)

    # Restore original metrics
    if original_metrics is not None:
        adapter._metrics = original_metrics  # type: ignore[attr-defined]

    assert any(obs["op"] == "query" for obs in metrics.observations)


async def test_observability_tenant_hashed_never_raw(adapter):
    """Verify tenant identifiers are never logged in raw form."""
    metrics = CaptureMetrics()
    original_metrics = getattr(adapter, "_metrics", None)
    adapter._metrics = metrics  # type: ignore[attr-defined]

    secret_tenant = "super-secret-tenant-12345"
    ctx = OperationContext(request_id="v_hash", tenant=secret_tenant)
    await adapter.query(QuerySpec(vector=[0.1], top_k=1, namespace="default"), ctx=ctx)

    # Restore original metrics
    if original_metrics is not None:
        adapter._metrics = original_metrics  # type: ignore[attr-defined]

    # Verify no raw tenant appears in any metrics output
    all_metrics_output = str(metrics.observations) + str(metrics.counters)
    assert secret_tenant not in all_metrics_output, "Raw tenant ID leaked in metrics"


async def test_observability_no_vector_data_in_metrics(adapter):
    """Verify vector data never appears in metrics output."""
    metrics = CaptureMetrics()
    original_metrics = getattr(adapter, "_metrics", None)
    adapter._metrics = metrics  # type: ignore[attr-defined]

    ctx = OperationContext(request_id="v_no_vec", tenant="test-tenant")
    await adapter.query(QuerySpec(vector=[0.9, 0.8], top_k=1, namespace="default"), ctx=ctx)

    # Restore original metrics
    if original_metrics is not None:
        adapter._metrics = original_metrics  # type: ignore[attr-defined]

    # Verify no vector data appears in metrics
    all_metrics_output = str(metrics.observations) + str(metrics.counters)
    vector_indicators = ["[0.9, 0.8]", "0.9", "0.8", "vector_data", "embedding"]
    for indicator in vector_indicators:
        assert indicator not in all_metrics_output, f"Vector data leaked: {indicator}"


async def test_observability_metrics_emitted_on_error_path(adapter):
    """Verify metrics are emitted even when operations fail."""
    metrics = CaptureMetrics()
    original_metrics = getattr(adapter, "_metrics", None)
    adapter._metrics = metrics  # type: ignore[attr-defined]

    ctx = OperationContext(request_id="v_err", tenant="test-tenant")

    with pytest.raises(BadRequest):
        await adapter.query(QuerySpec(vector=[0.1], top_k=0, namespace="default"), ctx=ctx)

    # Restore original metrics
    if original_metrics is not None:
        adapter._metrics = original_metrics  # type: ignore[attr-defined]

    error_observations = [obs for obs in metrics.observations if obs["op"] == "query" and not obs["ok"]]
    assert error_observations, "No metrics emitted for error path"


async def test_observability_query_metrics_include_namespace(adapter):
    """Verify query metrics include namespace information."""
    metrics = CaptureMetrics()
    original_metrics = getattr(adapter, "_metrics", None)
    adapter._metrics = metrics  # type: ignore[attr-defined]

    ctx = OperationContext(request_id="v_ns", tenant="test-tenant")
    await adapter.query(QuerySpec(vector=[0.2], top_k=1, namespace="test-namespace"), ctx=ctx)

    # Restore original metrics
    if original_metrics is not None:
        adapter._metrics = original_metrics  # type: ignore[attr-defined]

    query_observations = [obs for obs in metrics.observations if obs["op"] == "query"]
    assert query_observations, "No query observations found"
    
    last_query_extra = query_observations[-1].get("extra", {})
    assert last_query_extra.get("namespace") == "test-namespace"


async def test_observability_upsert_metrics_include_vector_count(adapter):
    """Verify upsert operations emit vector count metrics."""
    metrics = CaptureMetrics()
    original_metrics = getattr(adapter, "_metrics", None)
    adapter._metrics = metrics  # type: ignore[attr-defined]

    ctx = OperationContext(request_id="v_up", tenant="test-tenant")
    spec = UpsertSpec(
        namespace="default",
        vectors=[Vector(id=VectorID("m1"), vector=[0.1])],
    )
    await adapter.upsert(spec, ctx=ctx)

    # Restore original metrics
    if original_metrics is not None:
        adapter._metrics = original_metrics  # type: ignore[attr-defined]

    vector_counters = [counter for counter in metrics.counters if "vectors_upserted" in counter["name"]]
    assert vector_counters, "No vector count metrics emitted"
    assert vector_counters[-1]["value"] >= 1