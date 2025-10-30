# adapter_sdk/graph_base.py
# SPDX-License-Identifier: Apache-2.0
"""
Adapter SDK — Graph Protocol V1
A minimal, production-quality surface for building graph adapters.
"""

from __future__ import annotations
import hashlib
import time
from dataclasses import dataclass
from typing import (
    Any, Dict, Iterable, List, Mapping, Optional, Protocol, Tuple, 
    runtime_checkable, AsyncIterator, NewType
)

GRAPH_PROTOCOL_VERSION = "1.0.0"
KNOWN_DIALECTS: Tuple[str, ...] = ("cypher", "opencypher", "gremlin", "gql")

# Explicit type for graph identifiers - enhances protocol clarity
GraphID = NewType('GraphID', str)

class HealthStatus:
    """Standard health status constants."""
    OK = "ok"
    DEGRADED = "degraded" 
    UNAVAILABLE = "unavailable"
    READ_ONLY = "read_only"

class AdapterError(Exception):
    """
    Base exception for all graph adapter errors.
    
    Provides structured error information for clients including
    retry guidance, throttling information, and operational context.
    """
    def __init__(
        self,
        message: str = "",
        *,
        code: Optional[str] = None,
        retry_after_ms: Optional[int] = None,
        throttle_scope: Optional[str] = None,
        suggested_batch_reduction: Optional[int] = None,
        details: Optional[Mapping[str, Any]] = None,
        operation: Optional[str] = None,
        dialect: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.retry_after_ms = retry_after_ms
        self.throttle_scope = throttle_scope
        self.suggested_batch_reduction = suggested_batch_reduction
        self.details = dict(details or {})
        self.operation = operation
        self.dialect = dialect

    def asdict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        result = {
            "message": self.message,
            "code": self.code,
            "retry_after_ms": self.retry_after_ms,
            "throttle_scope": self.throttle_scope,
            "suggested_batch_reduction": self.suggested_batch_reduction,
            "details": {k: self.details[k] for k in sorted(self.details)},
        }
        if self.operation:
            result["operation"] = self.operation
        if self.dialect:
            result["dialect"] = self.dialect
        return result

class BadRequest(AdapterError):
    """Client sent an invalid request (4xx)."""
    pass

class AuthError(AdapterError):
    """Authentication or authorization failed."""
    pass

class ResourceExhausted(AdapterError):
    """Quota, rate limit, or resource constraints exceeded."""
    pass

class TransientNetwork(AdapterError):
    """Transient network failure that may be retried."""
    pass

class Unavailable(AdapterError):
    """Service is temporarily unavailable."""
    pass

class NotSupported(AdapterError):
    """Requested operation or dialect is not supported."""
    pass

@dataclass(frozen=True)
class OperationContext:
    """
    Context for graph operations providing tracing, deadlines, and metadata.
    
    Attributes:
        request_id: Unique identifier for the request chain
        idempotency_key: Key for ensuring idempotent operations
        deadline_ms: Operation timeout in milliseconds
        traceparent: W3C Trace Context header
        tenant: Multi-tenant isolation scope
        attrs: Additional operation attributes for extensibility
    """
    request_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    deadline_ms: Optional[int] = None
    traceparent: Optional[str] = None
    tenant: Optional[str] = None
    attrs: Mapping[str, Any] = None

    def __post_init__(self) -> None:
        if self.attrs is None:
            object.__setattr__(self, "attrs", {})
        
    @property
    def is_timed_out(self) -> bool:
        """Check if operation has exceeded deadline (placeholder implementation)."""
        if self.deadline_ms is None:
            return False
        # In real implementation, this would compare against request start time
        return False
        
    def with_attrs(self, **new_attrs: Any) -> OperationContext:
        """Return new context with additional attributes."""
        return OperationContext(
            request_id=self.request_id,
            idempotency_key=self.idempotency_key,
            deadline_ms=self.deadline_ms,
            traceparent=self.traceparent,
            tenant=self.tenant,
            attrs={**self.attrs, **new_attrs}
        )

class LogSink(Protocol):
    """Protocol for logging implementations."""
    def debug(self, message: str, *, extra: Optional[Mapping[str, Any]] = None) -> None: ...
    def info(self, message: str, *, extra: Optional[Mapping[str, Any]] = None) -> None: ...
    def warning(self, message: str, *, extra: Optional[Mapping[str, Any]] = None) -> None: ...
    def error(self, message: str, *, extra: Optional[Mapping[str, Any]] = None) -> None: ...

class NoopLogSink:
    """No-operation log sink for default or testing scenarios."""
    def debug(self, message: str, **_: Any) -> None: ...
    def info(self, message: str, **_: Any) -> None: ...
    def warning(self, message: str, **_: Any) -> None: ...
    def error(self, message: str, **_: Any) -> None: ...

class MetricsSink(Protocol):
    """Protocol for metrics collection implementations."""
    def observe(
        self,
        *,
        component: str,
        op: str,
        ms: float,
        ok: bool,
        code: str = "OK",
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None: ...
    def counter(
        self,
        *,
        component: str,
        name: str,
        value: int = 1,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None: ...

class NoopMetrics:
    """No-operation metrics sink for default or testing scenarios."""
    def observe(self, **_: Any) -> None: ...
    def counter(self, **_: Any) -> None: ...

@dataclass(frozen=True)
class GraphCapabilities:
    """
    Describes the capabilities and limitations of a graph adapter implementation.
    
    Attributes:
        server: Backend server identifier
        version: Backend server version
        dialects: Supported query dialects
        supports_txn: Whether transactions are supported
        supports_schema_ops: Whether schema operations are supported
        max_batch_ops: Maximum operations per batch (None for unlimited)
        retryable_codes: Which error codes are retryable
        rate_limit_unit: Unit for rate limiting (e.g., "requests_per_second")
        max_qps: Maximum queries per second (None for unlimited)
        idempotent_writes: Whether write operations are idempotent
        supports_multi_tenant: Whether multi-tenancy is supported
        supports_streaming: Whether streaming queries are supported
        supports_bulk_ops: Whether bulk operations are supported
    """
    server: str
    version: str
    dialects: Tuple[str, ...] = ("cypher",)
    supports_txn: bool = True
    supports_schema_ops: bool = True
    max_batch_ops: Optional[int] = None
    retryable_codes: Tuple[str, ...] = ()
    rate_limit_unit: str = "requests_per_second"
    max_qps: Optional[int] = None
    idempotent_writes: bool = False
    supports_multi_tenant: bool = False
    supports_streaming: bool = False
    supports_bulk_ops: bool = False

class BatchOperations:
    """Helper methods for constructing batch operations."""
    
    @staticmethod
    def create_vertex_op(label: str, props: Mapping[str, Any]) -> Dict[str, Any]:
        """Create a vertex creation batch operation."""
        return {"type": "create_vertex", "label": label, "props": dict(props)}
    
    @staticmethod
    def create_edge_op(label: str, from_id: GraphID, to_id: GraphID, props: Mapping[str, Any]) -> Dict[str, Any]:
        """Create an edge creation batch operation."""
        return {
            "type": "create_edge", 
            "label": label, 
            "from_id": str(from_id), 
            "to_id": str(to_id), 
            "props": dict(props)
        }
    
    @staticmethod
    def query_op(dialect: str, text: str, params: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Create a query batch operation."""
        return {"type": "query", "dialect": dialect, "text": text, "params": dict(params or {})}

class ProtocolVersion:
    """Helper for protocol version compatibility checks."""
    
    def __init__(self, version_str: str):
        self.major, self.minor, self.patch = map(int, version_str.split('.'))
    
    def is_compatible_with(self, other: str) -> bool:
        """Check if this version is compatible with another version."""
        other_ver = ProtocolVersion(other)
        return self.major == other_ver.major and self.minor >= other_ver.minor

@runtime_checkable
class GraphProtocolV1(Protocol):
    """
    Protocol defining the Graph Protocol V1 interface.
    
    Implement this protocol to create compatible graph adapters.
    All methods are async and should be implemented by concrete adapters.
    """
    
    async def capabilities(self) -> GraphCapabilities: 
        """Get the capabilities of this graph adapter."""
        ...
        
    async def create_vertex(
        self, label: str, props: Mapping[str, Any], *, ctx: Optional[OperationContext] = None
    ) -> GraphID: 
        """Create a new vertex with the given label and properties."""
        ...
        
    async def create_edge(
        self, label: str, from_id: GraphID, to_id: GraphID, props: Mapping[str, Any], *, ctx: Optional[OperationContext] = None
    ) -> GraphID: 
        """Create a new edge between two vertices."""
        ...
        
    async def delete_vertex(self, vertex_id: GraphID, *, ctx: Optional[OperationContext] = None) -> None: 
        """Delete a vertex by its identifier."""
        ...
        
    async def delete_edge(self, edge_id: GraphID, *, ctx: Optional[OperationContext] = None) -> None: 
        """Delete an edge by its identifier."""
        ...
        
    async def query(
        self,
        *,
        dialect: str,
        text: str,
        params: Optional[Mapping[str, Any]] = None,
        ctx: Optional[OperationContext] = None,
    ) -> List[Mapping[str, Any]]: 
        """Execute a query and return all results."""
        ...
        
    async def stream_query(
        self,
        *,
        dialect: str,
        text: str,
        params: Optional[Mapping[str, Any]] = None,
        ctx: Optional[OperationContext] = None,
    ) -> AsyncIterator[Mapping[str, Any]]: 
        """Execute a query and stream results as they arrive."""
        ...
        
    async def bulk_vertices(
        self, vertices: Iterable[Tuple[str, Mapping[str, Any]]], *, ctx: Optional[OperationContext] = None
    ) -> List[GraphID]: 
        """Create multiple vertices in a single operation."""
        ...
        
    async def batch(
        self,
        ops: Iterable[Mapping[str, Any]],
        *,
        ctx: Optional[OperationContext] = None,
    ) -> List[Mapping[str, Any]]: 
        """Execute multiple operations in a single batch."""
        ...
        
    async def get_schema(self, *, ctx: Optional[OperationContext] = None) -> Dict[str, Any]: 
        """Get the graph schema information."""
        ...
        
    async def health(self, *, ctx: Optional[OperationContext] = None) -> Dict[str, Any]: 
        """Check the health status of the graph backend."""
        ...

class _Base:
    """
    Base functionality shared by graph adapters.
    
    Provides common validation, metrics recording, and utility methods.
    """
    _component = "graph"
    _noop_log_sink = NoopLogSink()

    def __init__(self, *, metrics: Optional[MetricsSink] = None, logs: Optional[LogSink] = None) -> None:
        self._metrics: MetricsSink = metrics or NoopMetrics()
        self._logs: LogSink = logs or self._noop_log_sink

    @staticmethod
    def _require_non_empty(name: str, value: str) -> None:
        """Validate that a string value is non-empty."""
        if not isinstance(value, str) or not value.strip():
            raise BadRequest(f"{name} must be a non-empty string")

    def _validate_dialect(self, dialect: str) -> None:
        """Validate that a dialect is known and supported."""
        self._require_non_empty("dialect", dialect)
        if dialect not in KNOWN_DIALECTS:
            raise NotSupported(
                f"Dialect '{dialect}' not supported. Known dialects: {KNOWN_DIALECTS}",
                dialect=dialect
            )

    def _validate_properties(self, props: Mapping[str, Any]) -> Dict[str, Any]:
        """Convert and validate property types for graph compatibility."""
        if props is None:
            return {}
        return {str(k): v for k, v in props.items()}

    @staticmethod
    def _bucket_ms(ms: Optional[int]) -> Optional[str]:
        """Bucket milliseconds for metrics categorization."""
        if ms is None or ms < 0: return None
        if ms < 1000: return "<1s"
        if ms < 5000: return "<5s"
        if ms < 15000: return "<15s"
        if ms < 60000: return "<60s"
        return ">=60s"

    @staticmethod
    def _tenant_hash(tenant: Optional[str]) -> Optional[str]:
        """Create a privacy-preserving hash of tenant identifier."""
        if not tenant: return None
        return hashlib.sha256(tenant.encode("utf-8")).hexdigest()[:12]

    def _record(
        self,
        op: str,
        t0: float,
        ok: bool,
        *,
        code: str = "OK",
        ctx: Optional[OperationContext] = None,
        **extra: Any,
    ) -> None:
        """Record operation metrics with context."""
        try:
            dt_ms = (time.monotonic() - t0) * 1000.0
            x = dict(extra or {})
            if ctx:
                x["deadline_bucket"] = self._bucket_ms(ctx.deadline_ms)
                x["tenant"] = self._tenant_hash(ctx.tenant)
            self._metrics.observe(
                component=self._component, op=op, ms=dt_ms, ok=ok, code=code, extra=x or None
            )
        except Exception:
            # Never let metrics recording break the operation
            pass

class BaseGraphAdapter(_Base, GraphProtocolV1):
    """
    Base class for implementing Graph Protocol V1 adapters.
    
    Implementers should override the `_do_*` methods to provide
    backend-specific functionality while getting validation,
    observability, and error handling for free.
    
    Example:
        class MyGraphAdapter(BaseGraphAdapter):
            async def _do_create_vertex(self, label: str, props: Dict[str, Any], *, ctx: Optional[OperationContext]) -> GraphID:
                # Implementation here
                return GraphID("vertex-123")
    """

    async def capabilities(self) -> GraphCapabilities:
        """Get the capabilities of this graph adapter."""
        return await self._do_capabilities()

    async def create_vertex(
        self, label: str, props: Mapping[str, Any], *, ctx: Optional[OperationContext] = None
    ) -> GraphID:
        """
        Create a new vertex with the given label and properties.
        
        Args:
            label: The vertex type/label (must be non-empty)
            props: Vertex properties as key-value pairs
            ctx: Operation context for tracing, deadlines, etc.
            
        Returns:
            GraphID: The unique identifier for the created vertex
            
        Raises:
            BadRequest: For invalid arguments
            AuthError: For authentication failures
            ResourceExhausted: For quota/rate limiting
            AdapterError: For other backend-specific errors
        """
        self._require_non_empty("label", label)
        validated_props = self._validate_properties(props)
        t0 = time.monotonic()
        try:
            vid = await self._do_create_vertex(label, validated_props, ctx=ctx)
            self._record("create_vertex", t0, True, ctx=ctx)
            return GraphID(str(vid))  # Explicit type conversion
        except AdapterError as e:
            self._record("create_vertex", t0, False, code=type(e).__name__, ctx=ctx)
            raise

    async def create_edge(
        self, label: str, from_id: GraphID, to_id: GraphID, props: Mapping[str, Any], *, ctx: Optional[OperationContext] = None
    ) -> GraphID:
        """
        Create a new edge between two vertices.
        
        Args:
            label: The edge type/label (must be non-empty)
            from_id: Source vertex identifier
            to_id: Target vertex identifier  
            props: Edge properties as key-value pairs
            ctx: Operation context for tracing, deadlines, etc.
            
        Returns:
            GraphID: The unique identifier for the created edge
            
        Raises:
            BadRequest: For invalid arguments
            AuthError: For authentication failures  
            ResourceExhausted: For quota/rate limiting
            AdapterError: For other backend-specific errors
        """
        for n, v in (("label", label), ("from_id", from_id), ("to_id", to_id)):
            self._require_non_empty(n, str(v))
        validated_props = self._validate_properties(props)
        t0 = time.monotonic()
        try:
            eid = await self._do_create_edge(label, str(from_id), str(to_id), validated_props, ctx=ctx)
            self._record("create_edge", t0, True, ctx=ctx)
            return GraphID(str(eid))  # Explicit type conversion
        except AdapterError as e:
            self._record("create_edge", t0, False, code=type(e).__name__, ctx=ctx)
            raise

    async def delete_vertex(self, vertex_id: GraphID, *, ctx: Optional[OperationContext] = None) -> None:
        """Delete a vertex by its identifier."""
        self._require_non_empty("vertex_id", str(vertex_id))
        t0 = time.monotonic()
        try:
            await self._do_delete_vertex(str(vertex_id), ctx=ctx)
            self._record("delete_vertex", t0, True, ctx=ctx)
        except AdapterError as e:
            self._record("delete_vertex", t0, False, code=type(e).__name__, ctx=ctx)
            raise

    async def delete_edge(self, edge_id: GraphID, *, ctx: Optional[OperationContext] = None) -> None:
        """Delete an edge by its identifier."""
        self._require_non_empty("edge_id", str(edge_id))
        t0 = time.monotonic()
        try:
            await self._do_delete_edge(str(edge_id), ctx=ctx)
            self._record("delete_edge", t0, True, ctx=ctx)
        except AdapterError as e:
            self._record("delete_edge", t0, False, code=type(e).__name__, ctx=ctx)
            raise

    async def query(
        self,
        *,
        dialect: str,
        text: str,
        params: Optional[Mapping[str, Any]] = None,
        ctx: Optional[OperationContext] = None,
    ) -> List[Mapping[str, Any]]:
        """Execute a query and return all results."""
        self._validate_dialect(dialect)
        self._require_non_empty("text", text)
        validated_params = self._validate_properties(params or {})
        t0 = time.monotonic()
        try:
            res = await self._do_query(dialect=dialect, text=text, params=validated_params, ctx=ctx)
            self._record("query", t0, True, ctx=ctx, dialect=dialect, rows=len(res))
            return res
        except AdapterError as e:
            self._record("query", t0, False, code=type(e).__name__, ctx=ctx, dialect=dialect)
            raise

    async def stream_query(
        self,
        *,
        dialect: str,
        text: str,
        params: Optional[Mapping[str, Any]] = None,
        ctx: Optional[OperationContext] = None,
    ) -> AsyncIterator[Mapping[str, Any]]:
        """Execute a query and stream results as they arrive."""
        self._validate_dialect(dialect)
        self._require_non_empty("text", text)
        validated_params = self._validate_properties(params or {})
        t0 = time.monotonic()
        try:
            count = 0
            async for row in self._do_stream_query(dialect=dialect, text=text, params=validated_params, ctx=ctx):
                count += 1
                yield row
            self._record("stream_query", t0, True, ctx=ctx, dialect=dialect, rows=count)
        except AdapterError as e:
            self._record("stream_query", t0, False, code=type(e).__name__, ctx=ctx, dialect=dialect)
            raise

    async def bulk_vertices(
        self, vertices: Iterable[Tuple[str, Mapping[str, Any]]], *, ctx: Optional[OperationContext] = None
    ) -> List[GraphID]:
        """Create multiple vertices in a single operation."""
        vertex_list = list(vertices)
        for label, props in vertex_list:
            self._require_non_empty("label", label)
        validated_vertices = [(label, self._validate_properties(props)) for label, props in vertex_list]
        t0 = time.monotonic()
        try:
            ids = await self._do_bulk_vertices(validated_vertices, ctx=ctx)
            self._record("bulk_vertices", t0, True, ctx=ctx, count=len(vertex_list))
            return [GraphID(str(id)) for id in ids]  # Explicit type conversion
        except AdapterError as e:
            self._record("bulk_vertices", t0, False, code=type(e).__name__, ctx=ctx, count=len(vertex_list))
            raise

    async def batch(
        self,
        ops: Iterable[Mapping[str, Any]],
        *,
        ctx: Optional[OperationContext] = None,
    ) -> List[Mapping[str, Any]]:
        """Execute multiple operations in a single batch."""
        op_list = list(ops)
        t0 = time.monotonic()
        try:
            res = await self._do_batch(op_list, ctx=ctx)
            self._record("batch", t0, True, ctx=ctx, ops=len(op_list))
            return res
        except AdapterError as e:
            self._record("batch", t0, False, code=type(e).__name__, ctx=ctx, ops=len(op_list))
            raise

    async def get_schema(self, *, ctx: Optional[OperationContext] = None) -> Dict[str, Any]:
        """Get the graph schema information."""
        t0 = time.monotonic()
        try:
            schema = await self._do_get_schema(ctx=ctx)
            self._record("get_schema", t0, True, ctx=ctx)
            return dict(schema)
        except AdapterError as e:
            self._record("get_schema", t0, False, code=type(e).__name__, ctx=ctx)
            raise

    async def health(self, *, ctx: Optional[OperationContext] = None) -> Dict[str, Any]:
        """Check the health status of the graph backend."""
        t0 = time.monotonic()
        try:
            h = await self._do_health(ctx=ctx)
            self._record("health", t0, True, ctx=ctx)
            return {
                "status": h.get("status", HealthStatus.OK),
                "read_only": bool(h.get("read_only", False)),
                "degraded": bool(h.get("degraded", False)),
                "version": str(h.get("version", "")),
                "server": str(h.get("server", "")),
                "details": dict(h.get("details", {})),
            }
        except AdapterError as e:
            self._record("health", t0, False, code=type(e).__name__, ctx=ctx)
            raise

    # ============ ABSTRACT METHODS TO BE IMPLEMENTED ============

    async def _do_capabilities(self) -> GraphCapabilities:
        """Implement to return adapter-specific capabilities."""
        raise NotImplementedError

    async def _do_create_vertex(self, label: str, props: Dict[str, Any], *, ctx: Optional[OperationContext]) -> GraphID:
        """Implement vertex creation with validated inputs."""
        raise NotImplementedError

    async def _do_create_edge(
        self, label: str, from_id: str, to_id: str, props: Dict[str, Any], *, ctx: Optional[OperationContext]
    ) -> GraphID:
        """Implement edge creation with validated inputs."""
        raise NotImplementedError

    async def _do_delete_vertex(self, vertex_id: str, *, ctx: Optional[OperationContext]) -> None:
        """Implement vertex deletion."""
        raise NotImplementedError

    async def _do_delete_edge(self, edge_id: str, *, ctx: Optional[OperationContext]) -> None:
        """Implement edge deletion."""
        raise NotImplementedError

    async def _do_query(
        self, *, dialect: str, text: str, params: Mapping[str, Any], ctx: Optional[OperationContext]
    ) -> List[Mapping[str, Any]]:
        """Implement query execution."""
        raise NotImplementedError

    async def _do_stream_query(
        self, *, dialect: str, text: str, params: Mapping[str, Any], ctx: Optional[OperationContext]
    ) -> AsyncIterator[Mapping[str, Any]]:
        """Implement streaming query execution."""
        raise NotImplementedError

    async def _do_bulk_vertices(
        self, vertices: List[Tuple[str, Mapping[str, Any]]], *, ctx: Optional[OperationContext]
    ) -> List[GraphID]:
        """Implement bulk vertex creation."""
        raise NotImplementedError

    async def _do_batch(self, ops: List[Mapping[str, Any]], *, ctx: Optional[OperationContext]) -> List[Mapping[str, Any]]:
        """Implement batch operation execution."""
        raise NotImplementedError

    async def _do_get_schema(self, *, ctx: Optional[OperationContext]) -> Dict[str, Any]:
        """Implement schema retrieval."""
        raise NotImplementedError

    async def _do_health(self, *, ctx: Optional[OperationContext]) -> Dict[str, Any]:
        """Implement health check."""
        raise NotImplementedError

__all__ = [
    "GRAPH_PROTOCOL_VERSION", "KNOWN_DIALECTS", "GraphID", "AdapterError", 
    "BadRequest", "AuthError", "ResourceExhausted", "TransientNetwork", 
    "Unavailable", "NotSupported", "OperationContext", "LogSink", 
    "NoopLogSink", "MetricsSink", "NoopMetrics", "GraphCapabilities", 
    "GraphProtocolV1", "BaseGraphAdapter", "BatchOperations", "HealthStatus",
    "ProtocolVersion"
]
