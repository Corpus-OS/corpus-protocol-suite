# adapter_sdk/graph_base.py
# SPDX-License-Identifier: Apache-2.0
"""
Adapter SDK — Graph Protocol V1 (public contract + production-grade base)

Purpose
-------
A minimal, production-quality surface for building graph adapters that plug into a
control plane. This protocol enables seamless integration with any graph database
while maintaining production-grade observability, security, and operational rigor.

Design Philosophy
-----------------
- Minimal surface area: Core graph operations only, no vendor-specific extensions
- Async-first: All operations are non-blocking for high-concurrency environments
- Production hardened: Built-in metrics, error taxonomy, and context propagation
- Extensible: Capability discovery allows for database-specific features
- Query agnostic: Supports multiple graph query dialects through unified interface

Deliberate Non-Goals
--------------------
- No retries, hedging, circuit breakers, failover, pooling, or dynamic config.
- No secret fetching or policy evaluation.
- No backend-specific optimizations beyond validated calls and timing.
- No query planning, optimization, or result post-processing.

Those behaviors live in the control plane and upper routing layers.

Versioning
----------
Follow SemVer against GRAPH_PROTOCOL_VERSION. Minor versions are strictly additive.
- Patch (x.y.Z): Editorial clarifications, non-breaking fixes
- Minor (x.Y.z): New optional parameters, capabilities, or methods  
- Major (X.y.z): Breaking changes to signatures or behavior
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

# =============================================================================
# Core Type Definitions
# =============================================================================

GraphID = NewType('GraphID', str)
"""
Type alias for graph identifiers providing explicit type safety.

Using GraphID instead of raw string enhances protocol clarity and enables
better IDE support and type checking while maintaining string compatibility.
"""

class HealthStatus:
    """Standard health status constants for consistent health reporting."""
    OK = "ok"
    DEGRADED = "degraded" 
    UNAVAILABLE = "unavailable"
    READ_ONLY = "read_only"

# =============================================================================
# Normalized Errors (with retry hints and operational guidance)
# =============================================================================

class AdapterError(Exception):
    """
    Base exception for all graph adapter errors.
    
    Provides structured error information including retry guidance, throttling context,
    and operational suggestions for callers to handle failures gracefully.
    
    Attributes:
        message: Human-readable error description
        code: Machine-readable error code for programmatic handling
        retry_after_ms: Suggested delay before retry (None if not retryable)
        throttle_scope: Scope of throttling ("tenant", "cluster", "query_complexity")
        suggested_batch_reduction: Percentage reduction suggestion for batch size
        details: Additional context-specific error details
        operation: Operation that failed (for debugging and metrics)
        dialect: Query dialect in use during failure (if applicable)
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
        """Convert error to dictionary for serialization and logging."""
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
    """Client sent an invalid request (malformed parameters, invalid queries)."""
    pass

class AuthError(AdapterError):
    """Authentication or authorization failed (invalid credentials, permissions)."""
    pass

class ResourceExhausted(AdapterError):
    """Quota, rate limit, or resource constraints exceeded."""
    pass

class TransientNetwork(AdapterError):
    """Transient network failure that may succeed on retry."""
    pass

class Unavailable(AdapterError):
    """Service is temporarily unavailable or overloaded."""
    pass

class NotSupported(AdapterError):
    """Requested operation, dialect, or parameter is not supported."""
    pass

# =============================================================================
# Context (used for deadlines, identity, SIEM-safe metrics)
# =============================================================================

@dataclass(frozen=True)
class OperationContext:
    """
    Context for graph operations providing tracing, deadlines, and multi-tenant isolation.
    
    All context information is propagated through the call chain and used for
    observability, security, and operational control without exposing sensitive data.
    
    Attributes:
        request_id: Unique identifier for the request chain (correlation ID)
        idempotency_key: Key for ensuring idempotent operations (when supported)
        deadline_ms: Absolute epoch milliseconds when operation should timeout
        traceparent: W3C Trace Context header for distributed tracing
        tenant: Multi-tenant isolation scope (NEVER logged or exposed in metrics)
        attrs: Additional operation attributes for extensibility and middleware
    """
    request_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    deadline_ms: Optional[int] = None
    traceparent: Optional[str] = None
    tenant: Optional[str] = None
    attrs: Mapping[str, Any] = None

    def __post_init__(self) -> None:
        """Ensure attrs is always a valid dictionary."""
        if self.attrs is None:
            object.__setattr__(self, "attrs", {})
        
    @property
    def is_timed_out(self) -> bool:
        """
        Check if operation has exceeded deadline.
        
        Note: This is a placeholder implementation. In production, this would
        compare against the actual request start time stored in context.
        """
        if self.deadline_ms is None:
            return False
        # In real implementation, this would compare against request start time
        return False
        
    def with_attrs(self, **new_attrs: Any) -> OperationContext:
        """
        Return new context with additional attributes.
        
        Args:
            **new_attrs: Additional attributes to merge into existing attrs
            
        Returns:
            OperationContext: New context with merged attributes
        """
        return OperationContext(
            request_id=self.request_id,
            idempotency_key=self.idempotency_key,
            deadline_ms=self.deadline_ms,
            traceparent=self.traceparent,
            tenant=self.tenant,
            attrs={**self.attrs, **new_attrs}
        )

# =============================================================================
# Observability Interfaces (SIEM-safe, low-cardinality)
# =============================================================================

class LogSink(Protocol):
    """
    Protocol for logging implementations.
    
    Used for structured logging without exposing sensitive information.
    All log data must avoid PII and use hashed tenant identifiers.
    """
    def debug(self, message: str, *, extra: Optional[Mapping[str, Any]] = None) -> None: 
        """Log debug-level message with optional structured data."""
        ...
    def info(self, message: str, *, extra: Optional[Mapping[str, Any]] = None) -> None: 
        """Log info-level message with optional structured data."""
        ...
    def warning(self, message: str, *, extra: Optional[Mapping[str, Any]] = None) -> None: 
        """Log warning-level message with optional structured data."""
        ...
    def error(self, message: str, *, extra: Optional[Mapping[str, Any]] = None) -> None: 
        """Log error-level message with optional structured data."""
        ...

class NoopLogSink:
    """No-operation log sink for testing or when logging is disabled."""
    def debug(self, message: str, **_: Any) -> None: ...
    def info(self, message: str, **_: Any) -> None: ...
    def warning(self, message: str, **_: Any) -> None: ...
    def error(self, message: str, **_: Any) -> None: ...

class MetricsSink(Protocol):
    """
    Protocol for metrics collection implementations.
    
    Used for operational monitoring without exposing sensitive information.
    All metrics must be low-cardinality and never include PII or tenant identifiers.
    """
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
        """
        Record operation timing and status.
        
        Args:
            component: Component name (e.g., "graph")
            op: Operation name (e.g., "create_vertex", "query")
            ms: Operation duration in milliseconds
            ok: Whether operation succeeded
            code: Status code (error class name or "OK")
            extra: Additional low-cardinality dimensions
        """
        ...
    def counter(
        self,
        *,
        component: str,
        name: str,
        value: int = 1,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None: 
        """
        Increment a counter metric.
        
        Args:
            component: Component name (e.g., "graph") 
            name: Counter name (e.g., "requests", "vertices_created")
            value: Increment value
            extra: Additional low-cardinality dimensions
        """
        ...

class NoopMetrics:
    """No-operation metrics sink for testing or when metrics are disabled."""
    def observe(self, **_: Any) -> None: ...
    def counter(self, **_: Any) -> None: ...

# =============================================================================
# Capabilities (dynamic discovery for routing and planning)
# =============================================================================

@dataclass(frozen=True)
class GraphCapabilities:
    """
    Describes the capabilities and limitations of a graph adapter implementation.
    
    Used by routing layers for intelligent database selection, query planning,
    and feature compatibility checking across different graph database backends.
    
    Attributes:
        server: Backend server identifier (e.g., "neo4j", "janusgraph", "tigergraph")
        version: Backend server version string
        dialects: Supported query dialects ("cypher", "gremlin", "gql", etc.)
        supports_txn: Whether ACID transactions are supported
        supports_schema_ops: Whether schema operations are supported
        max_batch_ops: Maximum operations per batch (None for unlimited)
        retryable_codes: Which error codes are retryable
        rate_limit_unit: Unit for rate limiting ("requests_per_second", "tokens_per_minute")
        max_qps: Maximum queries per second (None for unlimited)
        idempotent_writes: Whether write operations are idempotent with idempotency_key
        supports_multi_tenant: Whether multi-tenant isolation is supported
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

# =============================================================================
# Helper Classes (utilities for common operations)
# =============================================================================

class BatchOperations:
    """
    Helper methods for constructing batch operations.
    
    Provides type-safe utilities for creating batch operation dictionaries
    that can be passed to the batch() method.
    """
    
    @staticmethod
    def create_vertex_op(label: str, props: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Create a vertex creation batch operation.
        
        Args:
            label: Vertex type/label
            props: Vertex properties as key-value pairs
            
        Returns:
            Dictionary representing a create_vertex batch operation
        """
        return {"type": "create_vertex", "label": label, "props": dict(props)}
    
    @staticmethod
    def create_edge_op(label: str, from_id: GraphID, to_id: GraphID, props: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Create an edge creation batch operation.
        
        Args:
            label: Edge type/label
            from_id: Source vertex identifier
            to_id: Target vertex identifier
            props: Edge properties as key-value pairs
            
        Returns:
            Dictionary representing a create_edge batch operation
        """
        return {
            "type": "create_edge", 
            "label": label, 
            "from_id": str(from_id), 
            "to_id": str(to_id), 
            "props": dict(props)
        }
    
    @staticmethod
    def query_op(dialect: str, text: str, params: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a query batch operation.
        
        Args:
            dialect: Query dialect ("cypher", "gremlin", etc.)
            text: Query text
            params: Query parameters
            
        Returns:
            Dictionary representing a query batch operation
        """
        return {"type": "query", "dialect": dialect, "text": text, "params": dict(params or {})}

class ProtocolVersion:
    """
    Helper for protocol version compatibility checks.
    
    Provides semantic version parsing and compatibility checking
    to ensure adapter and caller version alignment.
    """
    
    def __init__(self, version_str: str):
        """
        Initialize with a semantic version string.
        
        Args:
            version_str: Semantic version string (e.g., "1.0.0")
        """
        self.major, self.minor, self.patch = map(int, version_str.split('.'))
    
    def is_compatible_with(self, other: str) -> bool:
        """
        Check if this version is compatible with another version.
        
        Compatibility rules:
        - Major versions must match exactly
        - Minor version must be >= the other minor version
        - Patch versions don't affect compatibility
        
        Args:
            other: Other version string to check compatibility with
            
        Returns:
            bool: True if versions are compatible
        """
        other_ver = ProtocolVersion(other)
        return self.major == other_ver.major and self.minor >= other_ver.minor

# =============================================================================
# Stable Protocol Interface (async, versioned contract)
# =============================================================================

@runtime_checkable
class GraphProtocolV1(Protocol):
    """
    Protocol defining the Graph Protocol V1 interface.
    
    Implement this protocol to create compatible graph adapters. All methods are async
    and designed for high-concurrency environments. The protocol is runtime-checkable
    for dynamic adapter validation.
    """
    
    async def capabilities(self) -> GraphCapabilities: 
        """
        Get the capabilities of this graph adapter.
        
        Returns:
            GraphCapabilities: Description of supported features and limitations
            
        Note:
            This method is async to support dynamic capability discovery in
            distributed systems where capabilities may change or require
            network calls to determine.
        """
        ...
        
    async def create_vertex(
        self, label: str, props: Mapping[str, Any], *, ctx: Optional[OperationContext] = None
    ) -> GraphID: 
        """
        Create a new vertex with the given label and properties.
        
        Args:
            label: The vertex type/label (must be non-empty string)
            props: Vertex properties as key-value pairs
            ctx: Operation context for tracing, deadlines, and multi-tenancy
            
        Returns:
            GraphID: The unique identifier for the created vertex
            
        Raises:
            BadRequest: For invalid arguments or malformed parameters
            AuthError: For authentication or authorization failures
            ResourceExhausted: For quota or rate limit exceeded
            NotSupported: If vertex creation is not supported
            TransientNetwork: For retryable network failures
            Unavailable: For service unavailable errors
        """
        ...
        
    async def create_edge(
        self, label: str, from_id: GraphID, to_id: GraphID, props: Mapping[str, Any], *, ctx: Optional[OperationContext] = None
    ) -> GraphID: 
        """
        Create a new edge between two vertices.
        
        Args:
            label: The edge type/label (must be non-empty string)
            from_id: Source vertex identifier (must exist)
            to_id: Target vertex identifier (must exist)
            props: Edge properties as key-value pairs
            ctx: Operation context for tracing, deadlines, and multi-tenancy
            
        Returns:
            GraphID: The unique identifier for the created edge
            
        Raises:
            BadRequest: For invalid arguments or non-existent vertices
            AuthError: For authentication or authorization failures
            ResourceExhausted: For quota or rate limit exceeded
            NotSupported: If edge creation is not supported
            TransientNetwork: For retryable network failures
            Unavailable: For service unavailable errors
        """
        ...
        
    async def delete_vertex(self, vertex_id: GraphID, *, ctx: Optional[OperationContext] = None) -> None: 
        """
        Delete a vertex by its identifier.
        
        Args:
            vertex_id: Vertex identifier to delete
            ctx: Operation context for tracing, deadlines, and multi-tenancy
            
        Raises:
            BadRequest: For invalid vertex identifier
            AuthError: For authentication or authorization failures
            ResourceExhausted: For quota or rate limit exceeded
            NotSupported: If vertex deletion is not supported
            TransientNetwork: For retryable network failures
            Unavailable: For service unavailable errors
        """
        ...
        
    async def delete_edge(self, edge_id: GraphID, *, ctx: Optional[OperationContext] = None) -> None: 
        """
        Delete an edge by its identifier.
        
        Args:
            edge_id: Edge identifier to delete
            ctx: Operation context for tracing, deadlines, and multi-tenancy
            
        Raises:
            BadRequest: For invalid edge identifier
            AuthError: For authentication or authorization failures
            ResourceExhausted: For quota or rate limit exceeded
            NotSupported: If edge deletion is not supported
            TransientNetwork: For retryable network failures
            Unavailable: For service unavailable errors
        """
        ...
        
    async def query(
        self,
        *,
        dialect: str,
        text: str,
        params: Optional[Mapping[str, Any]] = None,
        ctx: Optional[OperationContext] = None,
    ) -> List[Mapping[str, Any]]: 
        """
        Execute a query and return all results.
        
        Args:
            dialect: Query dialect ("cypher", "gremlin", "gql", etc.)
            text: Query text in the specified dialect
            params: Query parameters as key-value pairs
            ctx: Operation context for tracing, deadlines, and multi-tenancy
            
        Returns:
            List of result mappings. Each result is a dictionary representing
            a node, edge, or projection from the query.
            
        Raises:
            BadRequest: For invalid query, dialect, or parameters
            AuthError: For authentication or authorization failures
            ResourceExhausted: For quota or rate limit exceeded
            NotSupported: If the dialect or query type is not supported
            TransientNetwork: For retryable network failures
            Unavailable: For service unavailable errors
        """
        ...
        
    async def stream_query(
        self,
        *,
        dialect: str,
        text: str,
        params: Optional[Mapping[str, Any]] = None,
        ctx: Optional[OperationContext] = None,
    ) -> AsyncIterator[Mapping[str, Any]]: 
        """
        Execute a query and stream results as they arrive.
        
        Args:
            dialect: Query dialect ("cypher", "gremlin", "gql", etc.)
            text: Query text in the specified dialect
            params: Query parameters as key-value pairs
            ctx: Operation context for tracing, deadlines, and multi-tenancy
            
        Yields:
            Result mappings as they become available. Each result is a dictionary
            representing a node, edge, or projection from the query.
            
        Raises:
            BadRequest: For invalid query, dialect, or parameters
            AuthError: For authentication or authorization failures
            ResourceExhausted: For quota or rate limit exceeded
            NotSupported: If streaming or the dialect is not supported
            TransientNetwork: For retryable network failures
            Unavailable: For service unavailable errors
        """
        ...
        
    async def bulk_vertices(
        self, vertices: Iterable[Tuple[str, Mapping[str, Any]]], *, ctx: Optional[OperationContext] = None
    ) -> List[GraphID]: 
        """
        Create multiple vertices in a single operation.
        
        Args:
            vertices: Iterable of (label, properties) tuples for vertices to create
            ctx: Operation context for tracing, deadlines, and multi-tenancy
            
        Returns:
            List of GraphIDs for the created vertices in the same order as input
            
        Raises:
            BadRequest: For invalid labels or properties
            AuthError: For authentication or authorization failures
            ResourceExhausted: For quota or rate limit exceeded
            NotSupported: If bulk operations are not supported
            TransientNetwork: For retryable network failures
            Unavailable: For service unavailable errors
        """
        ...
        
    async def batch(
        self,
        ops: Iterable[Mapping[str, Any]],
        *,
        ctx: Optional[OperationContext] = None,
    ) -> List[Mapping[str, Any]]: 
        """
        Execute multiple operations in a single batch.
        
        Args:
            ops: Iterable of operation dictionaries (use BatchOperations helpers)
            ctx: Operation context for tracing, deadlines, and multi-tenancy
            
        Returns:
            List of results corresponding to each operation in the input batch
            
        Raises:
            BadRequest: For invalid operations or batch size exceeded
            AuthError: For authentication or authorization failures
            ResourceExhausted: For quota or rate limit exceeded
            NotSupported: If batching is not supported
            TransientNetwork: For retryable network failures
            Unavailable: For service unavailable errors
        """
        ...
        
    async def get_schema(self, *, ctx: Optional[OperationContext] = None) -> Dict[str, Any]: 
        """
        Get the graph schema information.
        
        Args:
            ctx: Operation context for tracing, deadlines, and multi-tenancy
            
        Returns:
            Dictionary containing schema information (structure varies by backend)
            
        Raises:
            AuthError: For authentication or authorization failures
            NotSupported: If schema operations are not supported
            TransientNetwork: For retryable network failures
            Unavailable: For service unavailable errors
        """
        ...
        
    async def health(self, *, ctx: Optional[OperationContext] = None) -> Dict[str, Any]: 
        """
        Check the health status of the graph backend.
        
        Args:
            ctx: Operation context for tracing and multi-tenancy
            
        Returns:
            Dictionary with health information including:
            - status: Overall status (HealthStatus constants)
            - read_only: Whether backend is in read-only mode
            - degraded: Whether backend is degraded but operational
            - version: Backend version information
            - server: Backend server identifier
            - details: Additional backend-specific health details
            
        Raises:
            Unavailable: If the health check fails or backend is unreachable
        """
        ...

# =============================================================================
# Base Instrumented Adapter (validation, metrics, error handling)
# =============================================================================

class _Base:
    """
    Base functionality shared by graph adapters.
    
    Provides common validation, metrics recording, utility methods, and
    SIEM-safe observability. This class contains all shared infrastructure
    without implementing the protocol methods.
    """
    _component = "graph"
    _noop_log_sink = NoopLogSink()

    def __init__(self, *, metrics: Optional[MetricsSink] = None, logs: Optional[LogSink] = None) -> None:
        """
        Initialize the graph adapter with observability instrumentation.
        
        Args:
            metrics: Metrics sink for operational monitoring. Uses NoopMetrics if None.
            logs: Log sink for structured logging. Uses NoopLogSink if None.
        """
        self._metrics: MetricsSink = metrics or NoopMetrics()
        self._logs: LogSink = logs or self._noop_log_sink

    @staticmethod
    def _require_non_empty(name: str, value: str) -> None:
        """
        Validate that a string value is non-empty.
        
        Args:
            name: Parameter name for error messages
            value: Value to validate
            
        Raises:
            BadRequest: If value is empty or not a string
        """
        if not isinstance(value, str) or not value.strip():
            raise BadRequest(f"{name} must be a non-empty string")

    def _validate_dialect(self, dialect: str) -> None:
        """
        Validate that a dialect is known and supported.
        
        Args:
            dialect: Dialect to validate
            
        Raises:
            BadRequest: If dialect is empty
            NotSupported: If dialect is not in KNOWN_DIALECTS
        """
        self._require_non_empty("dialect", dialect)
        if dialect not in KNOWN_DIALECTS:
            raise NotSupported(
                f"Dialect '{dialect}' not supported. Known dialects: {KNOWN_DIALECTS}",
                dialect=dialect
            )

    def _validate_properties(self, props: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Convert and validate property types for graph compatibility.
        
        Args:
            props: Properties to validate and convert
            
        Returns:
            Validated properties dictionary with string keys
        """
        if props is None:
            return {}
        return {str(k): v for k, v in props.items()}

    @staticmethod
    def _bucket_ms(ms: Optional[int]) -> Optional[str]:
        """
        Bucket milliseconds for metrics categorization.
        
        Args:
            ms: Milliseconds to bucket
            
        Returns:
            Bucketed time range string or None
        """
        if ms is None or ms < 0: return None
        if ms < 1000: return "<1s"
        if ms < 5000: return "<5s"
        if ms < 15000: return "<15s"
        if ms < 60000: return "<60s"
        return ">=60s"

    @staticmethod
    def _tenant_hash(tenant: Optional[str]) -> Optional[str]:
        """
        Create privacy-preserving hash of tenant identifier for metrics.
        
        Args:
            tenant: Raw tenant identifier
            
        Returns:
            Hashed tenant identifier (first 12 chars of SHA256) or None
        """
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
        """
        Record operation metrics with context and tenant hashing.
        
        Never exposes raw tenant identifiers in metrics. Safe for SIEM systems.
        
        Args:
            op: Operation name
            t0: Start time from time.monotonic()
            ok: Whether operation succeeded
            code: Status code for metrics
            ctx: Operation context for tenant and deadline information
            **extra: Additional metric dimensions
        """
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
    
    Provides common validation, metrics instrumentation, error handling, and
    SIEM-safe observability. Implementers should override the `_do_*` methods
    to provide backend-specific functionality while getting production-ready
    infrastructure for free.
    
    Example:
        class Neo4jAdapter(BaseGraphAdapter):
            async def _do_create_vertex(self, label: str, props: Dict[str, Any], *, ctx: Optional[OperationContext]) -> GraphID:
                # Neo4j-specific implementation using driver sessions
                async with self._driver.session() as session:
                    result = await session.run(
                        "CREATE (v:$label) SET v = $props RETURN id(v) as id",
                        label=label, props=props
                    )
                    record = await result.single()
                    return GraphID(str(record["id"]))
    """

    async def capabilities(self) -> GraphCapabilities:
        """Get the capabilities of this graph adapter."""
        return await self._do_capabilities()

    async def create_vertex(
        self, label: str, props: Mapping[str, Any], *, ctx: Optional[OperationContext] = None
    ) -> GraphID:
        """
        Create a new vertex with the given label and properties.
        
        See GraphProtocolV1.create_vertex for full documentation.
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
        
        See GraphProtocolV1.create_edge for full documentation.
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
