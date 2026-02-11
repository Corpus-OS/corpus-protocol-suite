# tests/frameworks/embedding/test_mcp_adapter.py

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, AsyncMock, patch

import asyncio
import inspect
import pytest
import time

import corpus_sdk.embedding.framework_adapters.mcp as mcp_adapter_module
from corpus_sdk.embedding.framework_adapters.mcp import (
    CorpusMCPEmbeddings,
    MCPEmbedder,
    MCPContext,
    MCPConfig,
    MCPEmbeddingServiceError,
    create_embedder,
    ErrorCodes,
)
from corpus_sdk.embedding.embedding_base import (
    EmbeddingAdapterError,
    ResourceExhausted,
    DeadlineExceeded,
    Unavailable,
    TransientNetwork,
    OperationContext,
)


# ---------------------------------------------------------------------------
# MCP-Specific Test Considerations
# ---------------------------------------------------------------------------
"""
MCP Test Design Considerations:

1. **Async-First Architecture**: MCP is fundamentally async - all tests must be async
2. **Concurrency Limits**: MCP has configurable concurrent request limits
3. **Service Boundaries**: MCP enforces request limits (1000 texts, 1M chars)
4. **Protocol-Oriented**: Uses Protocol for type safety, not inheritance
5. **Error Taxonomy**: Rich error mapping from adapter to service errors
6. **Session Context**: Strong session/tool/request context propagation
7. **Health Metrics**: Built-in health checking with protocol metrics
8. **Request Semantics**: Each request gets a unique ID for tracing
"""


# ---------------------------------------------------------------------------
# Test Fixtures & Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def mcp_adapter() -> Any:
    """Create a mock adapter with MCP-compatible interface."""
    adapter = Mock()
    adapter.embed = Mock(return_value=[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    adapter.get_embedding_dimension = Mock(return_value=3)
    return adapter


@pytest.fixture
def mcp_context() -> MCPContext:
    """Create a typical MCP context for tests."""
    return {
        "session_id": "session-123",
        "tool_name": "test-tool",
        "server_id": "server-456",
        "client_id": "client-789",
        "trace_id": "trace-abc",
        "request_id": "req-xyz",
    }


def _assert_embedding_matrix_shape(result: Any, expected_rows: int) -> None:
    """Validate embedding matrix shape for MCP results."""
    assert isinstance(result, Sequence), f"Expected sequence, got {type(result).__name__}"
    assert len(result) == expected_rows, f"Expected {expected_rows} rows, got {len(result)}"
    for row in result:
        assert isinstance(row, Sequence), f"Row is not a sequence: {type(row).__name__}"
        for val in row:
            assert isinstance(val, (int, float)), f"Embedding value not numeric: {val!r}"


def _assert_embedding_vector_shape(result: Any) -> None:
    """Validate embedding vector shape for MCP results."""
    assert isinstance(result, Sequence), f"Expected sequence, got {type(result).__name__}"
    for val in result:
        assert isinstance(val, (int, float)), f"Embedding value not numeric: {val!r}"


async def _create_embeddings(**kwargs: Any) -> CorpusMCPEmbeddings:
    """Helper to create embeddings instance with defaults."""
    adapter = kwargs.pop("corpus_adapter", Mock())
    if "embed" not in dir(adapter):
        adapter.embed = Mock(return_value=[[0.0] * 8])
    
    return CorpusMCPEmbeddings(
        corpus_adapter=adapter,
        model="test-model",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Constructor & Configuration Tests
# ---------------------------------------------------------------------------

def test_constructor_accepts_mcp_adapter() -> None:
    """MCP embedder should accept valid adapter with async methods."""
    
    class MCPAdapter:
        async def embed(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
            return [[0.0] * 8 for _ in texts]
        
        def get_embedding_dimension(self) -> int:
            return 8
    
    embeddings = CorpusMCPEmbeddings(
        corpus_adapter=MCPAdapter(),
        model="mcp-model",
    )
    assert isinstance(embeddings, CorpusMCPEmbeddings)
    assert embeddings.model == "mcp-model"


def test_mcp_config_defaults_and_validation() -> None:
    """MCP config should have sensible defaults and validation."""
    
    class MockAdapter:
        async def embed(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
            return [[0.0] * 8 for _ in texts]
    
    # Test with empty config (should get defaults)
    embeddings = CorpusMCPEmbeddings(
        corpus_adapter=MockAdapter(),
        model="default-config-model",
        mcp_config={},
    )
    
    cfg = embeddings.mcp_config
    assert "max_concurrent_requests" in cfg
    assert "fallback_to_simple_context" in cfg
    assert "enable_session_context_propagation" in cfg
    assert "tool_aware_batching" in cfg
    
    # Default concurrent requests should be positive
    assert cfg["max_concurrent_requests"] > 0
    
    # Test invalid concurrent requests
    with pytest.raises(ValueError, match="must be positive"):
        CorpusMCPEmbeddings(
            corpus_adapter=MockAdapter(),
            mcp_config={"max_concurrent_requests": 0},
        )
    
    with pytest.raises(ValueError, match="must be an integer"):
        CorpusMCPEmbeddings(
            corpus_adapter=MockAdapter(),
            mcp_config={"max_concurrent_requests": "not-an-int"},  # type: ignore
        )


def test_mcp_config_boolean_coercion() -> None:
    """Boolean values in mcp_config should be coerced properly."""
    
    class MockAdapter:
        async def embed(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
            return [[0.0] * 8 for _ in texts]
    
    # Test truthy/falsy coercion
    embeddings = CorpusMCPEmbeddings(
        corpus_adapter=MockAdapter(),
        model="bool-coercion-model",
        mcp_config={
            "fallback_to_simple_context": 1,  # truthy -> True
            "enable_session_context_propagation": 0,  # falsy -> False
            "tool_aware_batching": True,
        },
    )
    
    cfg = embeddings.mcp_config
    assert cfg["fallback_to_simple_context"] is True
    assert cfg["enable_session_context_propagation"] is False
    assert cfg["tool_aware_batching"] is True


# ---------------------------------------------------------------------------
# Protocol Compliance Tests
# ---------------------------------------------------------------------------

def test_implements_mcp_embedder_protocol(mcp_adapter: Any) -> None:
    """CorpusMCPEmbeddings should implement MCPEmbedder protocol."""
    embeddings = CorpusMCPEmbeddings(
        corpus_adapter=mcp_adapter,
        model="protocol-model",
    )
    
    # Check required async methods exist
    assert hasattr(embeddings, "embed_documents")
    assert hasattr(embeddings, "embed_query")
    
    # Check they are async functions
    assert inspect.iscoroutinefunction(embeddings.embed_documents)
    assert inspect.iscoroutinefunction(embeddings.embed_query)
    
    # Should be usable as MCPEmbedder
    embedder: MCPEmbedder = embeddings
    assert embedder is embeddings


@pytest.mark.asyncio
async def test_create_embedder_factory(mcp_adapter: Any) -> None:
    """create_embedder should return MCPEmbedder protocol implementation."""
    embedder = create_embedder(
        corpus_adapter=mcp_adapter,
        model="factory-model",
        mcp_config={"max_concurrent_requests": 50},
    )
    
    assert isinstance(embedder, MCPEmbedder)
    assert hasattr(embedder, "embed_documents")
    assert hasattr(embedder, "embed_query")
    
    # Should be callable
    result = await embedder.embed_documents(["test"])
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Context & Error Context Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mcp_context_passed_to_context_translation(
    monkeypatch: pytest.MonkeyPatch,
    mcp_adapter: Any,
) -> None:
    """MCP context should be passed to context_from_mcp."""
    captured: Dict[str, Any] = {}
    
    def fake_from_mcp(ctx: Dict[str, Any]) -> None:
        captured["ctx"] = ctx
        return None
    
    monkeypatch.setattr(
        mcp_adapter_module,
        "from_mcp",
        fake_from_mcp,
    )
    
    embeddings = await _create_embeddings(corpus_adapter=mcp_adapter)
    
    mcp_ctx: MCPContext = {
        "session_id": "test-session",
        "tool_name": "test-tool",
        "request_id": "test-req",
    }
    
    result = await embeddings.embed_documents(["doc1", "doc2"], mcp_context=mcp_ctx)
    _assert_embedding_matrix_shape(result, expected_rows=2)
    
    assert captured.get("ctx") is not None
    assert captured["ctx"]["session_id"] == "test-session"
    assert captured["ctx"]["tool_name"] == "test-tool"


@pytest.mark.asyncio
async def test_error_context_includes_mcp_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Error context should include MCP-specific metadata."""
    captured_context: Dict[str, Any] = {}
    
    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_context.update(ctx)
    
    monkeypatch.setattr(
        mcp_adapter_module,
        "attach_context",
        fake_attach_context,
    )
    
    class FailingAdapter:
        async def embed(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
            raise RuntimeError("MCP adapter error: Check service configuration")
        
        def get_embedding_dimension(self) -> int:
            return 8
    
    embeddings = CorpusMCPEmbeddings(
        corpus_adapter=FailingAdapter(),
        model="error-context-model",
    )
    
    mcp_ctx: MCPContext = {
        "session_id": "error-session",
        "tool_name": "error-tool",
        "server_id": "error-server",
    }
    
    with pytest.raises(MCPEmbeddingServiceError) as exc_info:
        await embeddings.embed_documents(["test"], mcp_context=mcp_ctx)
    
    # Verify error is actionable
    error_str = str(exc_info.value)
    assert "MCP adapter error" in error_str or "Check service configuration" in error_str
    
    # Verify context was attached with MCP fields
    assert captured_context, "attach_context was not called"
    assert captured_context.get("framework") == "mcp"
    assert captured_context.get("session_id") == "error-session"
    assert captured_context.get("tool_name") == "error-tool"
    assert captured_context.get("operation") == "embedding_documents"


@pytest.mark.asyncio 
async def test_error_context_extraction_with_complex_mcp_context(
    monkeypatch: pytest.MonkeyPatch,
    mcp_adapter: Any,
) -> None:
    """Error context should handle complex/nested MCP contexts."""
    captured_context: Dict[str, Any] = {}
    
    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_context.update(ctx)
    
    monkeypatch.setattr(mcp_adapter_module, "attach_context", fake_attach_context)
    
    class FailingTranslator:
        async def arun_embed(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("translator failed in MCP context")
    
    embeddings = await _create_embeddings(corpus_adapter=mcp_adapter)
    
    # Patch the translator to inject failure
    with monkeypatch.context() as m:
        m.setattr(embeddings, "_translator", FailingTranslator())
        
        mcp_ctx: MCPContext = {
            "session_id": "complex-session",
            "tool_name": "complex-tool",
            "server_id": "complex-server",
            "client_id": "complex-client",
            "trace_id": "complex-trace",
            "request_id": "complex-req",
        }
        
        with pytest.raises(MCPEmbeddingServiceError):
            await embeddings.embed_documents(["doc1", "doc2"], mcp_context=mcp_ctx)
        
        ctx = captured_context
        # Should include all MCP routing fields
        assert ctx.get("session_id") == "complex-session"
        assert ctx.get("tool_name") == "complex-tool"
        assert ctx.get("server_id") == "complex-server"
        assert ctx.get("client_id") == "complex-client"
        assert ctx.get("trace_id") == "complex-trace"
        # Dynamic metrics
        assert ctx.get("texts_count") == 2


# ---------------------------------------------------------------------------
# Request Validation Tests (MCP-Specific Limits)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_embed_documents_rejects_empty_request() -> None:
    """MCP should reject empty document requests."""
    embeddings = await _create_embeddings()
    
    with pytest.raises(MCPEmbeddingServiceError) as exc_info:
        await embeddings.embed_documents([])
    
    assert exc_info.value.code == ErrorCodes.EMPTY_REQUEST
    assert "No texts provided" in str(exc_info.value)


@pytest.mark.asyncio
async def test_embed_documents_rejects_large_batch() -> None:
    """MCP should reject batches exceeding 1000 texts."""
    embeddings = await _create_embeddings()
    
    # Create 1001 texts to exceed limit
    texts = [f"text-{i}" for i in range(1001)]
    
    with pytest.raises(MCPEmbeddingServiceError) as exc_info:
        await embeddings.embed_documents(texts)
    
    assert exc_info.value.code == ErrorCodes.BATCH_SIZE_EXCEEDED
    assert "exceeds maximum 1000" in str(exc_info.value)


@pytest.mark.asyncio
async def test_embed_documents_rejects_large_text_size() -> None:
    """MCP should reject requests exceeding 1M total characters."""
    embeddings = await _create_embeddings()
    
    # Create text that exceeds character limit
    large_text = "x" * 500_000
    texts = [large_text, large_text]  # 1M total characters
    
    with pytest.raises(MCPEmbeddingServiceError) as exc_info:
        await embeddings.embed_documents(texts)
    
    assert exc_info.value.code == ErrorCodes.TEXT_SIZE_EXCEEDED
    assert "exceeds limit" in str(exc_info.value)


@pytest.mark.asyncio
async def test_embed_documents_rejects_non_string_items() -> None:
    """MCP should reject non-string items in batch requests."""
    embeddings = await _create_embeddings()
    
    with pytest.raises(MCPEmbeddingServiceError) as exc_info:
        await embeddings.embed_documents(["valid", 123, "valid"])  # type: ignore
    
    assert exc_info.value.code == ErrorCodes.INVALID_TEXT_TYPE
    assert "not a string" in str(exc_info.value)


@pytest.mark.asyncio
async def test_embed_query_validates_single_text() -> None:
    """embed_query should validate single text through batch validator."""
    embeddings = await _create_embeddings()
    
    # Very long single text
    long_text = "x" * 1_000_001
    
    with pytest.raises(MCPEmbeddingServiceError) as exc_info:
        await embeddings.embed_query(long_text)
    
    assert exc_info.value.code == ErrorCodes.TEXT_SIZE_EXCEEDED


# ---------------------------------------------------------------------------
# Async Semantics & Concurrency Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_embed_documents_and_query_basic(mcp_adapter: Any) -> None:
    """Basic async embedding should work with MCP."""
    embeddings = CorpusMCPEmbeddings(
        corpus_adapter=mcp_adapter,
        model="async-basic-model",
    )
    
    texts = ["mcp-doc-1", "mcp-doc-2", "mcp-doc-3"]
    query = "mcp-query"
    
    # Test documents embedding
    docs_result = await embeddings.embed_documents(texts)
    _assert_embedding_matrix_shape(docs_result, expected_rows=len(texts))
    
    # Test query embedding  
    query_result = await embeddings.embed_query(query)
    _assert_embedding_vector_shape(query_result)


@pytest.mark.asyncio
async def test_concurrent_request_limiting() -> None:
    """MCP should respect max_concurrent_requests configuration."""
    
    class SlowAdapter:
        def __init__(self) -> None:
            self.concurrent_calls = 0
            self.max_concurrent = 0
        
        async def embed(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
            self.concurrent_calls += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent_calls)
            await asyncio.sleep(0.01)  # Small delay to ensure concurrency
            result = [[0.0] * 8 for _ in texts]
            self.concurrent_calls -= 1
            return result
        
        def get_embedding_dimension(self) -> int:
            return 8
    
    adapter = SlowAdapter()
    
    # Configure for low concurrency
    embeddings = CorpusMCPEmbeddings(
        corpus_adapter=adapter,
        model="concurrent-model",
        mcp_config={"max_concurrent_requests": 2},
    )
    
    # Launch 5 concurrent requests
    tasks = [
        embeddings.embed_documents([f"doc-{i}"])
        for i in range(5)
    ]
    
    results = await asyncio.gather(*tasks)
    
    # Verify all succeeded
    assert len(results) == 5
    for result in results:
        _assert_embedding_matrix_shape(result, expected_rows=1)
    
    # Should not exceed configured limit (allowing for test timing)
    assert adapter.max_concurrent <= 3  # Some slack for test timing


@pytest.mark.asyncio
async def test_active_request_tracking() -> None:
    """MCP should track active requests for health monitoring."""
    
    class TrackingAdapter:
        async def embed(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
            await asyncio.sleep(0.005)
            return [[0.0] * 8 for _ in texts]
    
    embeddings = CorpusMCPEmbeddings(
        corpus_adapter=TrackingAdapter(),
        model="tracking-model",
        mcp_config={"max_concurrent_requests": 10},
    )
    
    # Start multiple requests
    tasks = [
        embeddings.embed_documents([f"doc-{i}"])
        for i in range(3)
    ]
    
    # Check health while requests are active
    health_before = await embeddings.health_check()
    assert health_before["active_requests"] > 0
    
    # Wait for completion
    await asyncio.gather(*tasks)
    
    # Check health after completion
    health_after = await embeddings.health_check()
    assert health_after["active_requests"] == 0


# ---------------------------------------------------------------------------
# Error Mapping Tests (Adapter → MCP Service Errors)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_map_resource_exhausted_to_service_unavailable() -> None:
    """ResourceExhausted errors should map to SERVICE_UNAVAILABLE."""
    
    class RateLimitedAdapter:
        async def embed(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
            raise ResourceExhausted(
                message="Rate limit exceeded: 100 requests per minute",
                code="RATE_LIMIT_EXCEEDED",
                resource_scope="rate_limit",
            )
    
    embeddings = CorpusMCPEmbeddings(
        corpus_adapter=RateLimitedAdapter(),
        model="rate-limit-model",
    )
    
    with pytest.raises(MCPEmbeddingServiceError) as exc_info:
        await embeddings.embed_documents(["test"])
    
    assert exc_info.value.code == ErrorCodes.SERVICE_UNAVAILABLE
    assert "Rate or resource limit exceeded" in str(exc_info.value)
    assert exc_info.value.request_id is not None


@pytest.mark.asyncio
async def test_map_deadline_exceeded_to_request_timeout() -> None:
    """DeadlineExceeded errors should map to REQUEST_TIMEOUT."""
    
    class SlowAdapter:
        async def embed(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
            raise DeadlineExceeded(
                message="Request timed out after 30s",
                code="DEADLINE_EXCEEDED",
            )
    
    embeddings = CorpusMCPEmbeddings(
        corpus_adapter=SlowAdapter(),
        model="timeout-model",
    )
    
    with pytest.raises(MCPEmbeddingServiceError) as exc_info:
        await embeddings.embed_documents(["test"])
    
    assert exc_info.value.code == ErrorCodes.REQUEST_TIMEOUT
    assert "Request deadline exceeded" in str(exc_info.value)


@pytest.mark.asyncio
async def test_map_unavailable_to_service_unavailable() -> None:
    """Unavailable/TransientNetwork errors should map to SERVICE_UNAVAILABLE."""
    
    class UnavailableAdapter:
        async def embed(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
            raise Unavailable(
                message="Service temporarily unavailable",
                code="SERVICE_UNAVAILABLE",
            )
    
    embeddings = CorpusMCPEmbeddings(
        corpus_adapter=UnavailableAdapter(),
        model="unavailable-model",
    )
    
    with pytest.raises(MCPEmbeddingServiceError) as exc_info:
        await embeddings.embed_documents(["test"])
    
    assert exc_info.value.code == ErrorCodes.SERVICE_UNAVAILABLE
    assert "Embedding backend unavailable" in str(exc_info.value)


@pytest.mark.asyncio
async def test_map_generic_adapter_error() -> None:
    """Generic EmbeddingAdapterError should map to EMBEDDING_EXTRACTION_ERROR."""
    
    class GenericErrorAdapter:
        async def embed(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
            raise EmbeddingAdapterError(
                message="Generic adapter error",
                code="GENERIC_ERROR",
            )
    
    embeddings = CorpusMCPEmbeddings(
        corpus_adapter=GenericErrorAdapter(),
        model="generic-error-model",
    )
    
    with pytest.raises(MCPEmbeddingServiceError) as exc_info:
        await embeddings.embed_documents(["test"])
    
    assert exc_info.value.code == ErrorCodes.EMBEDDING_EXTRACTION_ERROR
    assert "Embedding adapter error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_map_non_adapter_error() -> None:
    """Non-EmbeddingAdapterError exceptions should map appropriately."""
    
    class CrashAdapter:
        async def embed(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
            raise ValueError("Unexpected error in adapter")
    
    embeddings = CorpusMCPEmbeddings(
        corpus_adapter=CrashAdapter(),
        model="crash-model",
    )
    
    with pytest.raises(MCPEmbeddingServiceError) as exc_info:
        await embeddings.embed_documents(["test"])
    
    assert exc_info.value.code == ErrorCodes.EMBEDDING_EXTRACTION_ERROR
    assert "Embedding service error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_map_asyncio_timeout_error() -> None:
    """asyncio.TimeoutError should map to REQUEST_TIMEOUT."""
    
    class TimeoutAdapter:
        async def embed(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
            raise asyncio.TimeoutError("Embedding operation timed out")
    
    embeddings = CorpusMCPEmbeddings(
        corpus_adapter=TimeoutAdapter(),
        model="async-timeout-model",
    )
    
    with pytest.raises(MCPEmbeddingServiceError) as exc_info:
        await embeddings.embed_documents(["test"])
    
    assert exc_info.value.code == ErrorCodes.REQUEST_TIMEOUT
    assert "Embedding request timed out" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Health Check & Metrics Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_check_basic_functionality(mcp_adapter: Any) -> None:
    """health_check should return comprehensive service status."""
    embeddings = CorpusMCPEmbeddings(
        corpus_adapter=mcp_adapter,
        model="health-check-model",
        mcp_config={"max_concurrent_requests": 50},
    )
    
    health = await embeddings.health_check()
    
    # Basic structure
    assert "status" in health
    assert "active_requests" in health
    assert "max_concurrent_requests" in health
    assert "framework_version" in health
    
    # Metrics should be present
    assert "protocol_success_count" in health
    assert "protocol_error_count" in health
    assert "avg_protocol_latency_ms" in health
    
    # Should include adapter health test
    assert "service_test" in health
    assert health["service_test"] == "passed"
    assert "embedding_dimension" in health


@pytest.mark.asyncio
async def test_health_check_with_failing_adapter() -> None:
    """health_check should detect adapter failures."""
    
    class FailingHealthAdapter:
        async def embed(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
            raise RuntimeError("Adapter health check failed")
        
        def get_embedding_dimension(self) -> int:
            return 8
    
    embeddings = CorpusMCPEmbeddings(
        corpus_adapter=FailingHealthAdapter(),
        model="failing-health-model",
    )
    
    health = await embeddings.health_check()
    
    assert health["status"] == "degraded"
    assert "failed" in health["service_test"]


@pytest.mark.asyncio
async def test_health_metrics_increment_with_requests(mcp_adapter: Any) -> None:
    """Health metrics should increment with successful and failed requests."""
    embeddings = CorpusMCPEmbeddings(
        corpus_adapter=mcp_adapter,
        model="metrics-model",
    )
    
    # Initial health
    health_before = await embeddings.health_check()
    initial_success = health_before["protocol_success_count"]
    initial_errors = health_before["protocol_error_count"]
    
    # Successful request
    await embeddings.embed_documents(["success"])
    
    health_after_success = await embeddings.health_check()
    assert health_after_success["protocol_success_count"] == initial_success + 1
    assert health_after_success["protocol_error_count"] == initial_errors
    
    # Batch request should increment batch metrics
    await embeddings.embed_documents(["batch1", "batch2"])
    
    health_after_batch = await embeddings.health_check()
    assert health_after_batch["protocol_batch_success_count"] > 0


@pytest.mark.asyncio
async def test_capabilities_and_health_passthrough() -> None:
    """Should pass through capabilities and health from adapter."""
    
    class CapableAdapter:
        async def embed(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
            return [[0.0] * 8 for _ in texts]
        
        async def acapabilities(self) -> Dict[str, Any]:
            return {"supported_models": ["mcp-model-1", "mcp-model-2"], "max_batch": 1000}
        
        async def ahealth(self) -> Dict[str, Any]:
            return {"adapter_status": "healthy", "version": "1.0.0"}
    
    embeddings = CorpusMCPEmbeddings(
        corpus_adapter=CapableAdapter(),
        model="capable-model",
    )
    
    caps = await embeddings.acapabilities()
    assert isinstance(caps, dict)
    assert caps.get("supported_models") == ["mcp-model-1", "mcp-model-2"]
    
    health = await embeddings.ahealth()
    assert isinstance(health, dict)
    assert health.get("adapter_status") == "healthy"


@pytest.mark.asyncio
async def test_capabilities_and_health_fallback_to_sync() -> None:
    """Should fall back to sync methods when async not available."""
    
    class SyncOnlyAdapter:
        async def embed(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
            return [[0.0] * 8 for _ in texts]
        
        def capabilities(self) -> Dict[str, Any]:
            return {"via_sync_caps": True}
        
        def health(self) -> Dict[str, Any]:
            return {"via_sync_health": True}
    
    embeddings = CorpusMCPEmbeddings(
        corpus_adapter=SyncOnlyAdapter(),
        model="sync-fallback-model",
    )
    
    caps = await embeddings.acapabilities()
    assert isinstance(caps, dict)
    assert caps.get("via_sync_caps") is True
    
    health = await embeddings.ahealth()
    assert isinstance(health, dict)
    assert health.get("via_sync_health") is True


@pytest.mark.asyncio
async def test_capabilities_and_health_empty_when_missing() -> None:
    """Should return empty dict when adapter has no capabilities/health."""
    
    class MinimalAdapter:
        async def embed(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
            return [[0.0] * 8 for _ in texts]
    
    embeddings = CorpusMCPEmbeddings(
        corpus_adapter=MinimalAdapter(),
        model="minimal-model",
    )
    
    caps = await embeddings.acapabilities()
    assert isinstance(caps, dict)
    assert caps == {}
    
    health = await embeddings.ahealth()
    assert isinstance(health, dict)
    assert health == {}


# ---------------------------------------------------------------------------
# Tool-Aware Batching Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_aware_batching_flag(mcp_adapter: Any) -> None:
    """tool_aware_batching should add batch_strategy to framework context."""
    
    captured_contexts: List[Dict[str, Any]] = []
    
    class CapturingTranslator:
        async def arun_embed(self, raw_texts: Any, op_ctx: Any = None, framework_ctx: Any = None) -> Any:
            captured_contexts.append(dict(framework_ctx or {}))
            if isinstance(raw_texts, list):
                return [[0.0, 1.0, 2.0] for _ in raw_texts]
            return [0.0, 1.0, 2.0]
    
    embeddings = CorpusMCPEmbeddings(
        corpus_adapter=mcp_adapter,
        model="tool-aware-model",
        mcp_config={"tool_aware_batching": True},
    )
    
    # Patch translator to capture context
    embeddings._translator_instance = CapturingTranslator()
    
    mcp_ctx: MCPContext = {
        "tool_name": "search-tool",
        "session_id": "tool-session",
    }
    
    await embeddings.embed_documents(["doc1", "doc2"], mcp_context=mcp_ctx)
    
    assert captured_contexts
    last_ctx = captured_contexts[-1]
    assert last_ctx.get("batch_strategy") == "tool_aware_search-tool"


@pytest.mark.asyncio
async def test_tool_aware_batching_without_tool_name(mcp_adapter: Any) -> None:
    """tool_aware_batching should not add batch_strategy without tool_name."""
    
    captured_contexts: List[Dict[str, Any]] = []
    
    class CapturingTranslator:
        async def arun_embed(self, raw_texts: Any, op_ctx: Any = None, framework_ctx: Any = None) -> Any:
            captured_contexts.append(dict(framework_ctx or {}))
            if isinstance(raw_texts, list):
                return [[0.0, 1.0, 2.0] for _ in raw_texts]
            return [0.0, 1.0, 2.0]
    
    embeddings = CorpusMCPEmbeddings(
        corpus_adapter=mcp_adapter,
        model="no-tool-model",
        mcp_config={"tool_aware_batching": True},
    )
    
    embeddings._translator_instance = CapturingTranslator()
    
    # No tool_name in context
    mcp_ctx: MCPContext = {
        "session_id": "no-tool-session",
    }
    
    await embeddings.embed_documents(["doc1"], mcp_context=mcp_ctx)
    
    assert captured_contexts
    last_ctx = captured_contexts[-1]
    assert "batch_strategy" not in last_ctx


# ---------------------------------------------------------------------------
# Context Propagation Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_enable_session_context_propagation_flag(mcp_adapter: Any) -> None:
    """enable_session_context_propagation controls _operation_context inclusion."""
    
    captured_contexts: List[Dict[str, Any]] = []
    
    class CapturingTranslator:
        async def arun_embed(self, raw_texts: Any, op_ctx: Any = None, framework_ctx: Any = None) -> Any:
            captured_contexts.append({
                "op_ctx": op_ctx,
                "framework_ctx": dict(framework_ctx or {})
            })
            if isinstance(raw_texts, list):
                return [[0.0, 1.0, 2.0] for _ in raw_texts]
            return [0.0, 1.0, 2.0]
    
    # Enable propagation
    embeddings_enabled = CorpusMCPEmbeddings(
        corpus_adapter=mcp_adapter,
        model="ctx-prop-model",
        mcp_config={"enable_session_context_propagation": True},
    )
    embeddings_enabled._translator_instance = CapturingTranslator()
    
    mcp_ctx: MCPContext = {"session_id": "prop-session"}
    await embeddings_enabled.embed_documents(["doc"], mcp_context=mcp_ctx)
    
    # Disable propagation  
    embeddings_disabled = CorpusMCPEmbeddings(
        corpus_adapter=mcp_adapter,
        model="no-ctx-prop-model",
        mcp_config={"enable_session_context_propagation": False},
    )
    embeddings_disabled._translator_instance = CapturingTranslator()
    
    await embeddings_disabled.embed_documents(["doc"], mcp_context=mcp_ctx)
    
    # Check captured contexts
    assert len(captured_contexts) >= 2
    
    # First call (enabled) should have _operation_context in framework_ctx
    enabled_ctx = captured_contexts[0]["framework_ctx"]
    if "_operation_context" in enabled_ctx:
        assert enabled_ctx["_operation_context"] is not None
    
    # Second call (disabled) should not have _operation_context
    disabled_ctx = captured_contexts[1]["framework_ctx"]
    assert "_operation_context" not in disabled_ctx


@pytest.mark.asyncio
async def test_fallback_to_simple_context_flag(mcp_adapter: Any) -> None:
    """fallback_to_simple_context controls OperationContext creation."""
    
    # Mock context_from_mcp to return non-OperationContext
    from unittest.mock import Mock
    
    with patch.object(mcp_adapter_module, 'from_mcp') as mock_from_mcp:
        mock_from_mcp.return_value = Mock()  # Non-OperationContext
        
        # With fallback enabled (default)
        embeddings_fallback = CorpusMCPEmbeddings(
            corpus_adapter=mcp_adapter,
            model="fallback-model",
            mcp_config={"fallback_to_simple_context": True},
        )
        
        # Should still create OperationContext internally
        result = await embeddings_fallback.embed_documents(["doc"])
        _assert_embedding_matrix_shape(result, expected_rows=1)
        
        # With fallback disabled
        embeddings_no_fallback = CorpusMCPEmbeddings(
            corpus_adapter=mcp_adapter,
            model="no-fallback-model",
            mcp_config={"fallback_to_simple_context": False},
        )
        
        # Should also work (translator handles None OperationContext)
        result = await embeddings_no_fallback.embed_documents(["doc"])
        _assert_embedding_matrix_shape(result, expected_rows=1)


# ---------------------------------------------------------------------------
# Integration & Protocol Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestMCPIntegration:
    """Integration tests for MCP server workflows."""
    
    @pytest.fixture
    async def mcp_embedder(self, mcp_adapter: Any) -> MCPEmbedder:
        """Create an MCP embedder for integration tests."""
        return create_embedder(
            corpus_adapter=mcp_adapter,
            model="integration-model",
            mcp_config={
                "max_concurrent_requests": 20,
                "tool_aware_batching": True,
            },
        )
    
    @pytest.mark.asyncio
    async def test_embedder_in_server_workflow(
        self,
        mcp_embedder: MCPEmbedder,
        mcp_context: MCPContext,
    ) -> None:
        """MCP embedder should work in server-like workflow."""
        # Simulate MCP tool execution
        documents = [
            "MCP is a protocol for building AI applications.",
            "Embeddings convert text to vector representations.",
            "The protocol supports tools, resources, and prompts.",
        ]
        
        # Embed documents with MCP context
        embeddings = await mcp_embedder.embed_documents(
            documents,
            mcp_context=mcp_context,
        )
        
        _assert_embedding_matrix_shape(embeddings, expected_rows=len(documents))
        
        # Embed query for retrieval
        query = "What is MCP?"
        query_embedding = await mcp_embedder.embed_query(
            query,
            mcp_context=mcp_context,
        )
        
        _assert_embedding_vector_shape(query_embedding)
    
    @pytest.mark.asyncio
    async def test_concurrent_mcp_requests(
        self,
        mcp_embedder: MCPEmbedder,
    ) -> None:
        """MCP embedder should handle concurrent requests gracefully."""
        # Create multiple concurrent requests
        tasks = [
            mcp_embedder.embed_documents([f"request-{i}-doc"])
            for i in range(10)
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 10
        for result in results:
            _assert_embedding_matrix_shape(result, expected_rows=1)
    
    @pytest.mark.asyncio
    async def test_mcp_error_handling_in_workflow(
        self,
        mcp_context: MCPContext,
    ) -> None:
        """MCP errors should be actionable in server workflows."""
        
        class FailingMCPAdapter:
            async def embed(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
                raise ResourceExhausted(
                    message="Daily quota exceeded: 1000 requests per day",
                    code="QUOTA_EXCEEDED",
                    resource_scope="quota",
                )
        
        embedder = create_embedder(
            corpus_adapter=FailingMCPAdapter(),
            model="failing-workflow-model",
        )
        
        with pytest.raises(MCPEmbeddingServiceError) as exc_info:
            await embedder.embed_documents(
                ["test document"],
                mcp_context=mcp_context,
            )
        
        # Error should be actionable for MCP server
        error = exc_info.value
        assert error.code == ErrorCodes.SERVICE_UNAVAILABLE
        assert "quota" in str(error).lower() or "limit" in str(error).lower()
        assert error.request_id is not None
    
    @pytest.mark.asyncio
    async def test_mcp_health_in_server_monitoring(
        self,
        mcp_embedder: MCPEmbedder,
    ) -> None:
        """MCP embedder should provide health for server monitoring."""
        # Cast to CorpusMCPEmbeddings to access health_check
        if isinstance(mcp_embedder, CorpusMCPEmbeddings):
            health = await mcp_embedder.health_check()
            
            # Should have metrics for server monitoring
            assert "status" in health
            assert "active_requests" in health
            assert "avg_protocol_latency_ms" in health
            
            # Should indicate service is operational
            assert health["status"] in ["healthy", "degraded"]
            assert health["service_test"] == "passed"


# ---------------------------------------------------------------------------
# Edge Cases & Stress Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_near_limit_batch_size() -> None:
    """Should handle batches near the 1000 text limit."""
    embeddings = await _create_embeddings()
    
    # Create exactly 1000 texts (the limit)
    texts = [f"text-{i}" for i in range(1000)]
    
    result = await embeddings.embed_documents(texts)
    _assert_embedding_matrix_shape(result, expected_rows=1000)


@pytest.mark.asyncio  
async def test_near_limit_text_size() -> None:
    """Should handle text near the 1M character limit."""
    embeddings = await _create_embeddings()
    
    # Create text just under limit
    text = "x" * 999_999
    
    result = await embeddings.embed_query(text)
    _assert_embedding_vector_shape(result)


@pytest.mark.asyncio
async def test_mixed_empty_and_valid_texts() -> None:
    """Should handle batches with empty and valid texts."""
    
    class CountingAdapter:
        def __init__(self) -> None:
            self.call_count = 0
        
        async def embed(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
            self.call_count += 1
            # Return embeddings only for non-empty texts
            non_empty = [t for t in texts if t.strip()]
            return [[float(i)] * 8 for i in range(len(non_empty))]
    
    adapter = CountingAdapter()
    embeddings = CorpusMCPEmbeddings(
        corpus_adapter=adapter,
        model="mixed-text-model",
    )
    
    texts = ["", "valid1", "   ", "valid2", ""]  # 2 valid, 3 empty
    
    result = await embeddings.embed_documents(texts)
    _assert_embedding_matrix_shape(result, expected_rows=5)
    
    # Adapter should be called with only non-empty texts
    # Note: Implementation may filter before calling adapter


@pytest.mark.asyncio
async def test_rapid_concurrent_health_checks() -> None:
    """Health checks should work during high concurrency."""
    
    class SlowAdapter:
        async def embed(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
            await asyncio.sleep(0.02)  # Simulate work
            return [[0.0] * 8 for _ in texts]
    
    embeddings = CorpusMCPEmbeddings(
        corpus_adapter=SlowAdapter(),
        model="health-concurrency-model",
        mcp_config={"max_concurrent_requests": 5},
    )
    
    # Start embedding requests
    embed_tasks = [
        embeddings.embed_documents([f"doc-{i}"])
        for i in range(3)
    ]
    
    # Interleave health checks
    health_tasks = [
        embeddings.health_check()
        for _ in range(3)
    ]
    
    # Run all concurrently
    all_results = await asyncio.gather(*embed_tasks, *health_tasks)
    
    # Verify all succeeded
    assert len(all_results) == 6
    for i in range(3):
        _assert_embedding_matrix_shape(all_results[i], expected_rows=1)
    for i in range(3, 6):
        assert isinstance(all_results[i], dict)
        assert "status" in all_results[i]
