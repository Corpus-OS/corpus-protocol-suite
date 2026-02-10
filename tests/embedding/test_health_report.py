# SPDX-License-Identifier: Apache-2.0
"""
Embedding Conformance — Health endpoint.

Spec refs:
  • §10.3 health() — Embedding health contract
  • §6.4 Common — Health surfaces MUST be small, stable, and SIEM-safe
  • §13.3 Observability & Privacy

Notes:
- OperationContext does not carry metrics; context tests validate acceptance and stability.
"""

import pytest

from corpus_sdk.embedding.embedding_base import BaseEmbeddingAdapter, OperationContext

pytestmark = pytest.mark.asyncio


async def test_health_returns_required_fields(adapter: BaseEmbeddingAdapter):
    """§10.3: health() must return all required fields."""
    ctx = OperationContext(request_id="t_health_ok", tenant="t")
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

    for model_name, status in models.items():
        assert isinstance(model_name, str) and model_name
        assert isinstance(status, str) and status
        assert len(status) <= 50
        assert "\n" not in status
        assert "{" not in status and "}" not in status


async def test_health_server_version_strings(adapter: BaseEmbeddingAdapter):
    """§10.3: 'server' and 'version' must be non-empty strings."""
    h = await adapter.health()
    assert isinstance(h["server"], str) and h["server"]
    assert isinstance(h["version"], str) and h["version"]
    assert len(h["server"]) <= 100
    assert len(h["version"]) <= 50


async def test_health_shape_consistent_on_error_like_response(adapter: BaseEmbeddingAdapter):
    """§6.4: Health response must maintain canonical shape even when degraded."""
    h = await adapter.health()

    assert set(h.keys()) == {"ok", "server", "version", "models"}
    assert isinstance(h["ok"], bool)
    assert isinstance(h["server"], str)
    assert isinstance(h["version"], str)
    assert isinstance(h["models"], dict)


async def test_health_models_includes_supported_models(adapter: BaseEmbeddingAdapter):
    """§10.3: Health models should include adapter's supported models."""
    h = await adapter.health()
    caps = await adapter.capabilities()

    for supported_model in caps.supported_models:
        assert supported_model in h["models"], f"Supported model {supported_model} missing from health"

        status = h["models"][supported_model]
        assert isinstance(status, str) and status
        # Accept a broad set of SIEM-safe statuses; do not require a specific vocabulary.
        assert len(status) <= 50


async def test_health_context_propagation(adapter: BaseEmbeddingAdapter):
    """§6.1: Health should accept operation context and preserve response shape."""
    ctx = OperationContext(
        request_id="health_ctx_test",
        tenant="test-tenant",
        deadline_ms=int(__import__("time").time() * 1000) + 5000,
        attrs={"health": "degraded"},
    )

    h = await adapter.health(ctx=ctx)
    assert isinstance(h, dict)
    assert set(h.keys()) == {"ok", "server", "version", "models"}


async def test_health_siem_safe_no_sensitive_data(adapter: BaseEmbeddingAdapter):
    """§13.3: Health response must not contain sensitive data."""
    h = await adapter.health()
    health_str = str(h).lower()

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

    assert duration < 5.0, f"Health check too slow: {duration:.2f}s"
    assert isinstance(h, dict)
    assert "ok" in h


async def test_health_idempotent(adapter: BaseEmbeddingAdapter):
    """§6.2: Health checks should be idempotent in shape and stable fields."""
    h1 = await adapter.health()
    h2 = await adapter.health()

    assert h1.keys() == h2.keys()
    assert h1["server"] == h2["server"]
    assert h1["version"] == h2["version"]
    assert h1["models"].keys() == h2["models"].keys()
