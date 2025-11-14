# SPDX-License-Identifier: Apache-2.0
"""
Embedding Conformance — Health endpoint.

Spec refs:
  • §10.3 health() — Embedding health contract
  • §6.4 Common — Health surfaces MUST be small, stable, and SIEM-safe
  • §13.3 Observability & Privacy
"""

import pytest

from corpus_sdk.embedding.embedding_base import BaseEmbeddingAdapter, OperationContext
from examples.common.ctx import make_ctx

pytestmark = pytest.mark.asyncio


async def test_health_returns_required_fields(adapter: BaseEmbeddingAdapter):
    """§10.3: health() must return all required fields."""
    ctx = make_ctx(OperationContext, request_id="t_health_ok", tenant="t")
    h = await adapter.health(ctx=ctx)

    assert isinstance(h, dict)
    assert "ok" in h, "Missing 'ok' field"
    assert "server" in h, "Missing 'server' field"
    assert "version" in h, "Missing 'version' field"
    assert "models" in h, "Missing 'models' field"


async def test_health_ok_is_boolean(adapter: BaseEmbeddingAdapter):
    """§10.3: 'ok' field must be boolean."""
    h = await adapter.health()
    assert isinstance(h["ok"], bool), f"'ok' must be boolean, got {type(h['ok'])}"


async def test_health_models_dict_shape(adapter: BaseEmbeddingAdapter):
    """§10.3: 'models' must be dict with SIEM-safe status strings."""
    h = await adapter.health()
    models = h["models"]

    assert isinstance(models, dict), f"'models' must be dict, got {type(models)}"
    assert len(models) > 0, "'models' dict cannot be empty"
    
    # Values must be simple, SIEM-safe status strings
    for model_name, status in models.items():
        assert isinstance(model_name, str) and model_name, f"Model name must be non-empty string, got {model_name}"
        assert isinstance(status, str) and status, f"Status must be non-empty string, got {status}"
        # Status should be a simple word, not complex data
        assert len(status) <= 50, f"Status too long: {status}"
        assert "\n" not in status, f"Status contains newline: {status}"
        assert "{" not in status and "}" not in status, f"Status contains JSON: {status}"


async def test_health_server_version_strings(adapter: BaseEmbeddingAdapter):
    """§10.3: 'server' and 'version' must be non-empty strings."""
    h = await adapter.health()
    
    assert isinstance(h["server"], str) and h["server"], f"'server' must be non-empty string, got {h['server']}"
    assert isinstance(h["version"], str) and h["version"], f"'version' must be non-empty string, got {h['version']}"
    
    # Should be reasonable lengths
    assert len(h["server"]) <= 100, f"'server' too long: {h['server']}"
    assert len(h["version"]) <= 50, f"'version' too long: {h['version']}"


async def test_health_shape_consistent_on_error_like_response(adapter: BaseEmbeddingAdapter):
    """§6.4: Health response must maintain canonical shape even when degraded."""
    h = await adapter.health()

    # Must have exactly these four fields
    assert set(h.keys()) == {"ok", "server", "version", "models"}, f"Unexpected health fields: {set(h.keys())}"
    
    # All fields must have correct types regardless of health status
    assert isinstance(h["ok"], bool)
    assert isinstance(h["server"], str)
    assert isinstance(h["version"], str) 
    assert isinstance(h["models"], dict)


async def test_health_models_includes_supported_models(adapter: BaseEmbeddingAdapter):
    """§10.3: Health models should include adapter's supported models."""
    h = await adapter.health()
    caps = adapter.capabilities
    
    # Health should include all supported models
    for supported_model in caps.supported_models:
        assert supported_model in h["models"], f"Supported model {supported_model} missing from health"
        
        # Status should be a meaningful string
        status = h["models"][supported_model]
        assert status in ["healthy", "ready", "available", "ok"] or any(
            term in status.lower() for term in ['up', 'ok', 'ready', 'healthy']
        ), f"Unclear health status for {supported_model}: {status}"


async def test_health_context_propagation(adapter: BaseEmbeddingAdapter):
    """§6.1: Health should respect operation context."""
    from unittest.mock import Mock
    mock_metrics = Mock()
    
    ctx = make_ctx(
        OperationContext,
        request_id="health_ctx_test",
        tenant="test-tenant",
        metrics=mock_metrics,
    )
    
    h = await adapter.health(ctx=ctx)
    
    # Should return valid health response
    assert isinstance(h, dict)
    assert "ok" in h
    
    # If adapter implements metrics, they should be called
    if hasattr(adapter, '_emit_metrics') or mock_metrics.method_calls:
        health_calls = [call for call in mock_metrics.method_calls if 'health' in str(call).lower()]
        assert health_calls or not mock_metrics.method_calls, "Expected health metrics"


async def test_health_siem_safe_no_sensitive_data(adapter: BaseEmbeddingAdapter):
    """§13.3: Health response must not contain sensitive data."""
    h = await adapter.health()
    
    # Serialize health response to string for inspection
    health_str = str(h).lower()
    
    # Should not contain sensitive patterns
    sensitive_terms = [
        "password", "secret", "key", "token", "auth",
        "credential", "private", "internal", "localhost",
        "127.0.0.1", "0.0.0.0", "admin", "root"
    ]
    
    for term in sensitive_terms:
        assert term not in health_str, f"Health response contains sensitive term: {term}"


async def test_health_performance_reasonable(adapter: BaseEmbeddingAdapter):
    """§6.4: Health checks should be fast and lightweight."""
    import time
    
    start_time = time.time()
    h = await adapter.health()
    duration = time.time() - start_time
    
    # Health checks should be very fast (under 5 seconds)
    assert duration < 5.0, f"Health check too slow: {duration:.2f}s"
    
    # Response should still be valid
    assert isinstance(h, dict)
    assert "ok" in h


async def test_health_idempotent(adapter: BaseEmbeddingAdapter):
    """§6.2: Health checks should be idempotent."""
    h1 = await adapter.health()
    h2 = await adapter.health()
    
    # Structure should be identical
    assert h1.keys() == h2.keys()
    
    # Core fields should be consistent
    assert h1["ok"] == h2["ok"]
    assert h1["server"] == h2["server"]
    assert h1["version"] == h2["version"]
    
    # Models dict should have same keys (status might change)
    assert h1["models"].keys() == h2["models"].keys()