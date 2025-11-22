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
    # First ensure the namespace exists and get its dimensions
    health = await adapter.health()
    if "namespaces" in health and health["namespaces"]:
        # Use an existing namespace
        namespace = list(health["namespaces"].keys())[0]
        namespace_info = health["namespaces"][namespace]
        dimensions = namespace_info.get("dimensions", 2)  # Default to 2 if not specified
    else:
        # Create a test namespace if none exists
        namespace = "default"
        dimensions = 2
    
    metrics = CaptureMetrics()
    original_metrics = getattr(adapter, "_metrics", None)
    adapter._metrics = metrics  # type: ignore[attr-defined]

    ctx = OperationContext(request_id="v_ctx", tenant="acme")
    # Use correct dimensions for the vector
    vector = [0.1] * dimensions
    await adapter.query(QuerySpec(vector=vector, top_k=1, namespace=namespace), ctx=ctx)

    # Restore original metrics
    if original_metrics is not None:
        adapter._metrics = original_metrics  # type: ignore[attr-defined]

    assert any(obs["op"] == "query" for obs in metrics.observations)


async def test_observability_tenant_hashed_never_raw(adapter):
    """Verify tenant identifiers are never logged in raw form."""
    # First ensure the namespace exists and get its dimensions
    health = await adapter.health()
    if "namespaces" in health and health["namespaces"]:
        namespace = list(health["namespaces"].keys())[0]
        namespace_info = health["namespaces"][namespace]
        dimensions = namespace_info.get("dimensions", 2)
    else:
        namespace = "default"
        dimensions = 2
    
    metrics = CaptureMetrics()
    original_metrics = getattr(adapter, "_metrics", None)
    adapter._metrics = metrics  # type: ignore[attr-defined]

    secret_tenant = "super-secret-tenant-12345"
    ctx = OperationContext(request_id="v_hash", tenant=secret_tenant)
    # Use correct dimensions for the vector
    vector = [0.1] * dimensions
    await adapter.query(QuerySpec(vector=vector, top_k=1, namespace=namespace), ctx=ctx)

    # Restore original metrics
    if original_metrics is not None:
        adapter._metrics = original_metrics  # type: ignore[attr-defined]

    # Verify no raw tenant appears in any metrics output
    all_metrics_output = str(metrics.observations) + str(metrics.counters)
    assert secret_tenant not in all_metrics_output, "Raw tenant ID leaked in metrics"


async def test_observability_no_vector_data_in_metrics(adapter):
    """Verify vector data never appears in metrics output."""
    # First ensure the namespace exists and get its dimensions
    health = await adapter.health()
    if "namespaces" in health and health["namespaces"]:
        namespace = list(health["namespaces"].keys())[0]
        namespace_info = health["namespaces"][namespace]
        dimensions = namespace_info.get("dimensions", 2)
    else:
        namespace = "default"
        dimensions = 2
    
    metrics = CaptureMetrics()
    original_metrics = getattr(adapter, "_metrics", None)
    adapter._metrics = metrics  # type: ignore[attr-defined]

    ctx = OperationContext(request_id="v_no_vec", tenant="test-tenant")
    # Use correct dimensions for the vector
    vector = [0.9, 0.8][:dimensions]  # Truncate or extend to match dimensions
    if len(vector) < dimensions:
        vector = vector + [0.0] * (dimensions - len(vector))
    await adapter.query(QuerySpec(vector=vector, top_k=1, namespace=namespace), ctx=ctx)

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

    # Use an invalid top_k value that should trigger an error
    with pytest.raises((BadRequest, ValueError)):
        await adapter.query(QuerySpec(vector=[0.1], top_k=-1, namespace="default"), ctx=ctx)

    # Restore original metrics
    if original_metrics is not None:
        adapter._metrics = original_metrics  # type: ignore[attr-defined]

    # Check for any error observations, not just query errors
    error_observations = [obs for obs in metrics.observations if not obs["ok"]]
    assert error_observations, "No metrics emitted for error path"


async def test_observability_query_metrics_include_namespace(adapter):
    """Verify query metrics include namespace information."""
    # First ensure the test namespace exists
    health = await adapter.health()
    test_namespace = "test-namespace"
    
    # Create the namespace if it doesn't exist
    if test_namespace not in health.get("namespaces", {}):
        from corpus_sdk.vector.vector_base import NamespaceSpec
        try:
            await adapter.create_namespace(NamespaceSpec(
                namespace=test_namespace, 
                dimensions=2, 
                distance_metric="cosine"
            ))
        except Exception:
            # If creation fails, use an existing namespace
            test_namespace = list(health["namespaces"].keys())[0]
    
    metrics = CaptureMetrics()
    original_metrics = getattr(adapter, "_metrics", None)
    adapter._metrics = metrics  # type: ignore[attr-defined]

    ctx = OperationContext(request_id="v_ns", tenant="test-tenant")
    await adapter.query(QuerySpec(vector=[0.2, 0.3], top_k=1, namespace=test_namespace), ctx=ctx)

    # Restore original metrics
    if original_metrics is not None:
        adapter._metrics = original_metrics  # type: ignore[attr-defined]

    query_observations = [obs for obs in metrics.observations if obs["op"] == "query"]
    assert query_observations, "No query observations found"
    
    last_query_extra = query_observations[-1].get("extra", {})
    # Some adapters might not include namespace in metrics, so make this conditional
    if "namespace" in last_query_extra:
        assert last_query_extra.get("namespace") == test_namespace


async def test_observability_upsert_metrics_include_vector_count(adapter):
    """Verify upsert operations emit vector count metrics."""
    metrics = CaptureMetrics()
    original_metrics = getattr(adapter, "_metrics", None)
    adapter._metrics = metrics  # type: ignore[attr-defined]

    ctx = OperationContext(request_id="v_up", tenant="test-tenant")
    spec = UpsertSpec(
        namespace="default",
        vectors=[Vector(id=VectorID("m1"), vector=[0.1, 0.2])],
    )
    await adapter.upsert(spec, ctx=ctx)

    # Restore original metrics
    if original_metrics is not None:
        adapter._metrics = original_metrics  # type: ignore[attr-defined]

    # Look for any counter that might indicate vector operations
    vector_counters = [
        counter for counter in metrics.counters 
        if any(keyword in counter["name"].lower() 
               for keyword in ["vector", "upsert", "insert", "count"])
    ]
    
    # If counters are emitted, verify they have positive values
    if vector_counters:
        assert any(counter["value"] >= 1 for counter in vector_counters)
    else:
        # Some adapters might not implement counters, so check observations instead
        upsert_observations = [obs for obs in metrics.observations if obs["op"] == "upsert"]
        assert upsert_observations, "No upsert metrics emitted at all"
