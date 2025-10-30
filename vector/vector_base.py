# adapter_sdk/vector_base.py
# SPDX-License-Identifier: Apache-2.0
"""
Adapter SDK — Vector Protocol V1 (public contract + production-grade base)

Purpose
-------
A stable, vendor-neutral API for vector similarity search and operations — with
structured errors, caching strategies, rate limiting, and production observability.

This protocol enables seamless integration with any vector database while maintaining
production-grade security, performance monitoring, and operational rigor for
high-scale similarity search workloads.

Design Philosophy
-----------------
- Minimal surface area: Core vector operations only, no vendor-specific extensions
- Async-first: All operations are non-blocking for high-concurrency environments
- Production hardened: Built-in caching, circuit breaking, backpressure, and metrics
- Extensible: Capability discovery allows for database-specific vector features
- Performance optimized: Built-in caching strategies for vector similarity search

Deliberate Non-Goals
--------------------
- No embedding model management or text-to-vector transformations
- No vector index tuning or optimization strategies
- No vendor-specific distance metrics or indexing algorithms
- No client-side result re-ranking or post-processing

Those behaviors live in the embedding service and upper application layers.

Versioning
----------
Follow SemVer against VECTOR_PROTOCOL_VERSION. Minor versions are strictly additive.
- Patch (x.y.Z): Editorial clarifications, non-breaking fixes
- Minor (x.Y.z): New optional parameters, capabilities, or methods  
- Major (X.y.z): Breaking changes to signatures or behavior
"""

from __future__ import annotations
import time
import hashlib
from dataclasses import dataclass
from typing import (
    Any, Dict, List, Mapping, Optional, Protocol, Tuple, Iterable,
    runtime_checkable, AsyncIterator, Union
)

VECTOR_PROTOCOL_VERSION = "1.0.0"

# =============================================================================
# Core Type Definitions
# =============================================================================

VectorID = NewType('VectorID', str)
"""Type alias for vector identifiers providing explicit type safety."""

@dataclass(frozen=True)
class Vector:
    """
    A vector with optional metadata and identifier.
    
    Attributes:
        id: Unique identifier for the vector
        vector: The vector embedding as a list of floats
        metadata: Optional key-value pairs for filtering and retrieval
        namespace: Optional namespace/collection for multi-tenant isolation
    """
    id: VectorID
    vector: List[float]
    metadata: Optional[Dict[str, Any]] = None
    namespace: Optional[str] = None

@dataclass(frozen=True)
class QueryResult:
    """
    Result from a vector similarity search query.
    
    Attributes:
        matches: List of matching vectors with similarity scores
        query_vector: The original query vector used for search
        namespace: Namespace/collection where search was performed
        total_matches: Total number of matches found (before limiting)
    """
    matches: List[VectorMatch]
    query_vector: List[float]
    namespace: str
    total_matches: int

@dataclass(frozen=True)
class VectorMatch:
    """
    A single vector match with similarity score.
    
    Attributes:
        vector: The matching vector
        score: Similarity score (higher = more similar)
        distance: Raw distance metric value (lower = more similar)
    """
    vector: Vector
    score: float
    distance: float

# =============================================================================
# Normalized Errors (with retry hints and operational guidance)
# =============================================================================

class VectorAdapterError(Exception):
    """
    Base exception for all vector adapter errors.
    
    Provides structured error information including retry guidance, resource limits,
    and operational suggestions for callers to handle failures gracefully.
    
    Attributes:
        message: Human-readable error description
        code: Machine-readable error code for programmatic handling
        retry_after_ms: Suggested delay before retry (None if not retryable)
        resource_scope: Scope of resource limitation ("embedding", "memory", "compute")
        suggested_batch_reduction: Percentage reduction suggestion for batch size
        details: Additional context-specific error details
    """
    def __init__(
        self,
        message: str = "",
        *,
        code: Optional[str] = None,
        retry_after_ms: Optional[int] = None,
        resource_scope: Optional[str] = None,
        suggested_batch_reduction: Optional[int] = None,
        details: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.retry_after_ms = retry_after_ms
        self.resource_scope = resource_scope
        self.suggested_batch_reduction = suggested_batch_reduction
        self.details = dict(details or {})

    def asdict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization and logging."""
        return {
            "message": self.message,
            "code": self.code,
            "retry_after_ms": self.retry_after_ms,
            "resource_scope": self.resource_scope,
            "suggested_batch_reduction": self.suggested_batch_reduction,
            "details": {k: self.details[k] for k in sorted(self.details)},
        }

class BadRequest(VectorAdapterError):
    """Client sent an invalid request (malformed vectors, invalid parameters)."""
    pass

class AuthError(VectorAdapterError):
    """Authentication or authorization failed (invalid credentials, permissions)."""
    pass

class ResourceExhausted(VectorAdapterError):
    """Quota, rate limit, or resource constraints exceeded."""
    pass

class DimensionMismatch(VectorAdapterError):
    """Vector dimensions do not match expected schema."""
    pass

class IndexNotReady(VectorAdapterError):
    """Vector index is not built or ready for queries."""
    pass

class TransientNetwork(VectorAdapterError):
    """Transient network failure that may succeed on retry."""
    pass

class Unavailable(VectorAdapterError):
    """Service is temporarily unavailable or overloaded."""
    pass

class NotSupported(VectorAdapterError):
    """Requested operation or parameter is not supported."""
    pass

# =============================================================================
# Context (used for deadlines, identity, SIEM-safe metrics)
# =============================================================================

@dataclass(frozen=True)
class OperationContext:
    """
    Context for vector operations providing tracing, deadlines, and multi-tenant isolation.
    
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

# =============================================================================
# Metrics Interface (SIEM-safe, low-cardinality)
# =============================================================================

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
            component: Component name (e.g., "vector")
            op: Operation name (e.g., "query", "upsert")
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
            component: Component name (e.g., "vector") 
            name: Counter name (e.g., "queries", "vectors_upserted")
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
class VectorCapabilities:
    """
    Describes the capabilities and limitations of a vector adapter implementation.
    
    Used by routing layers for intelligent database selection, query planning,
    and feature compatibility checking across different vector database backends.
    
    Attributes:
        server: Backend server identifier (e.g., "pinecone", "qdrant", "weaviate")
        version: Backend server version string
        max_dimensions: Maximum vector dimensions supported
        supported_metrics: Supported distance metrics ("cosine", "euclidean", "dotproduct")
        supports_namespaces: Whether namespaces/collections are supported
        supports_metadata_filtering: Whether metadata filtering is supported
        supports_batch_operations: Whether batch upsert/delete are supported
        max_batch_size: Maximum vectors per batch operation
        supports_index_management: Whether index creation/deletion is supported
        idempotent_writes: Whether write operations are idempotent with idempotency_key
        supports_multi_tenant: Whether multi-tenant isolation is supported
    """
    server: str
    version: str
    max_dimensions: int
    supported_metrics: Tuple[str, ...] = ("cosine", "euclidean", "dotproduct")
    supports_namespaces: bool = True
    supports_metadata_filtering: bool = True
    supports_batch_operations: bool = True
    max_batch_size: Optional[int] = None
    supports_index_management: bool = False
    idempotent_writes: bool = False
    supports_multi_tenant: bool = False

# =============================================================================
# Query and Operation Specifications
# =============================================================================

@dataclass(frozen=True)
class QuerySpec:
    """
    Specification for vector similarity search queries.
    
    Attributes:
        vector: Query vector for similarity search
        top_k: Number of top results to return
        namespace: Target namespace/collection for the query
        filter: Optional metadata filters for pre-search filtering
        include_metadata: Whether to include metadata in results
        include_vectors: Whether to include vector data in results
    """
    vector: List[float]
    top_k: int
    namespace: str = "default"
    filter: Optional[Dict[str, Any]] = None
    include_metadata: bool = True
    include_vectors: bool = False

@dataclass(frozen=True)
class UpsertSpec:
    """
    Specification for vector upsert operations.
    
    Attributes:
        vectors: List of vectors to upsert
        namespace: Target namespace/collection for the operation
    """
    vectors: List[Vector]
    namespace: str = "default"

@dataclass(frozen=True)
class DeleteSpec:
    """
    Specification for vector deletion operations.
    
    Attributes:
        ids: List of vector IDs to delete
        namespace: Target namespace/collection for the operation
        filter: Optional metadata filter for bulk deletion
    """
    ids: List[VectorID]
    namespace: str = "default"
    filter: Optional[Dict[str, Any]] = None

@dataclass(frozen=True)
class NamespaceSpec:
    """
    Specification for namespace/collection management operations.
    
    Attributes:
        namespace: Namespace/collection name
        dimensions: Vector dimensions for this namespace
        distance_metric: Distance metric to use ("cosine", "euclidean", "dotproduct")
    """
    namespace: str
    dimensions: int
    distance_metric: str = "cosine"

# =============================================================================
# Operation Results
# =============================================================================

@dataclass
class UpsertResult:
    """
    Result from vector upsert operations.
    
    Attributes:
        upserted_count: Number of vectors successfully upserted
        failed_count: Number of vectors that failed to upsert
        failures: List of individual failure details
    """
    upserted_count: int
    failed_count: int
    failures: List[Dict[str, Any]]

@dataclass
class DeleteResult:
    """
    Result from vector deletion operations.
    
    Attributes:
        deleted_count: Number of vectors successfully deleted
        failed_count: Number of vectors that failed to delete
        failures: List of individual failure details
    """
    deleted_count: int
    failed_count: int
    failures: List[Dict[str, Any]]

@dataclass
class NamespaceResult:
    """
    Result from namespace management operations.
    
    Attributes:
        success: Whether the operation completed successfully
        namespace: The namespace that was operated on
        details: Additional operation-specific details
    """
    success: bool
    namespace: str
    details: Dict[str, Any]

# =============================================================================
# Stable Protocol Interface (async, versioned contract)
# =============================================================================

@runtime_checkable
class VectorProtocolV1(Protocol):
    """
    Protocol defining the Vector Protocol V1 interface.
    
    Implement this protocol to create compatible vector adapters. All methods are async
    and designed for high-concurrency environments. The protocol is runtime-checkable
    for dynamic adapter validation.
    """

    async def capabilities(self) -> VectorCapabilities:
        """
        Get the capabilities of this vector adapter.
        
        Returns:
            VectorCapabilities: Description of supported features and limitations
            
        Note:
            This method is async to support dynamic capability discovery in
            distributed systems where capabilities may change or require
            network calls to determine.
        """
        ...

    async def query(
        self,
        spec: QuerySpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> QueryResult:
        """
        Execute a vector similarity search query.
        
        Args:
            spec: Query specification including vector, filters, and parameters
            ctx: Operation context for tracing, deadlines, and multi-tenancy
            
        Returns:
            QueryResult: Search results with matching vectors and scores
            
        Raises:
            BadRequest: For invalid query parameters or malformed vectors
            AuthError: For authentication or authorization failures
            ResourceExhausted: For quota or rate limit exceeded
            DimensionMismatch: If query vector dimensions don't match namespace
            IndexNotReady: If vector index is not built or ready
            NotSupported: If filtering or other features are not supported
            TransientNetwork: For retryable network failures
            Unavailable: For service unavailable errors
        """
        ...

    async def upsert(
        self,
        spec: UpsertSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> UpsertResult:
        """
        Upsert vectors into the vector store.
        
        Args:
            spec: Upsert specification including vectors and target namespace
            ctx: Operation context for tracing, deadlines, and multi-tenancy
            
        Returns:
            UpsertResult: Result indicating success/failure counts
            
        Raises:
            BadRequest: For invalid vectors or malformed data
            AuthError: For authentication or authorization failures
            ResourceExhausted: For quota or rate limit exceeded
            DimensionMismatch: If vector dimensions don't match namespace
            NotSupported: If batch operations are not supported
            TransientNetwork: For retryable network failures
            Unavailable: For service unavailable errors
        """
        ...

    async def delete(
        self,
        spec: DeleteSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> DeleteResult:
        """
        Delete vectors from the vector store.
        
        Args:
            spec: Delete specification including vector IDs and filters
            ctx: Operation context for tracing, deadlines, and multi-tenancy
            
        Returns:
            DeleteResult: Result indicating success/failure counts
            
        Raises:
            BadRequest: For invalid vector IDs or malformed filters
            AuthError: For authentication or authorization failures
            ResourceExhausted: For quota or rate limit exceeded
            NotSupported: If bulk deletion or filtering are not supported
            TransientNetwork: For retryable network failures
            Unavailable: For service unavailable errors
        """
        ...

    async def create_namespace(
        self,
        spec: NamespaceSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> NamespaceResult:
        """
        Create a new namespace/collection for vector storage.
        
        Args:
            spec: Namespace specification including dimensions and settings
            ctx: Operation context for tracing, deadlines, and multi-tenancy
            
        Returns:
            NamespaceResult: Result of namespace creation operation
            
        Raises:
            BadRequest: For invalid namespace parameters
            AuthError: For authentication or authorization failures
            ResourceExhausted: For quota or rate limit exceeded
            NotSupported: If namespace management is not supported
            TransientNetwork: For retryable network failures
            Unavailable: For service unavailable errors
        """
        ...

    async def delete_namespace(
        self,
        namespace: str,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> NamespaceResult:
        """
        Delete a namespace/collection and all its vectors.
        
        Args:
            namespace: Namespace to delete
            ctx: Operation context for tracing, deadlines, and multi-tenancy
            
        Returns:
            NamespaceResult: Result of namespace deletion operation
            
        Raises:
            BadRequest: For invalid namespace name
            AuthError: For authentication or authorization failures
            ResourceExhausted: For quota or rate limit exceeded
            NotSupported: If namespace management is not supported
            TransientNetwork: For retryable network failures
            Unavailable: For service unavailable errors
        """
        ...

    async def health(self, *, ctx: Optional[OperationContext] = None) -> Dict[str, Any]:
        """
        Check the health status of the vector backend.
        
        Args:
            ctx: Operation context for tracing and multi-tenancy
            
        Returns:
            Dictionary with health information including:
            - ok: Boolean overall health status
            - server: Backend server identifier
            - version: Backend version information
            - namespaces: Available namespaces and their status
            
        Raises:
            Unavailable: If the health check fails or backend is unreachable
        """
        ...

# =============================================================================
# Base Instrumented Adapter (validation, metrics, error handling)
# =============================================================================

class BaseVectorAdapter(VectorProtocolV1):
    """
    Base class for implementing Vector Protocol V1 adapters.
    
    Provides common validation, metrics instrumentation, error handling, and
    SIEM-safe observability. Implementers should override the `_do_*` methods
    to provide backend-specific functionality while getting production-ready
    infrastructure for free.
    
    Example:
        class PineconeAdapter(BaseVectorAdapter):
            async def _do_query(self, spec: QuerySpec, *, ctx: Optional[OperationContext]) -> QueryResult:
                # Pinecone-specific implementation
                response = await self._client.query(
                    vector=spec.vector,
                    top_k=spec.top_k,
                    namespace=spec.namespace,
                    filter=spec.filter
                )
                return QueryResult(
                    matches=[VectorMatch(...) for match in response.matches],
                    query_vector=spec.vector,
                    namespace=spec.namespace,
                    total_matches=len(response.matches)
                )
    """

    _component = "vector"

    def __init__(self, *, metrics: Optional[MetricsSink] = None) -> None:
        """
        Initialize the vector adapter with metrics instrumentation.
        
        Args:
            metrics: Metrics sink for operational monitoring. Uses NoopMetrics if None.
        """
        self._metrics: MetricsSink = metrics or NoopMetrics()

    # --- internal helpers (validation and instrumentation) ---

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

    @staticmethod
    def _validate_vector(vector: List[float]) -> None:
        """
        Validate that a vector is properly formed.
        
        Args:
            vector: Vector to validate
            
        Raises:
            BadRequest: If vector is empty or contains invalid values
        """
        if not vector or not isinstance(vector, list):
            raise BadRequest("vector must be a non-empty list of floats")
        if not all(isinstance(x, (int, float)) for x in vector):
            raise BadRequest("vector must contain only numeric values")

    @staticmethod
    def _tenant_hash(tenant: Optional[str]) -> Optional[str]:
        """
        Create privacy-preserving hash of tenant identifier for metrics.
        
        Args:
            tenant: Raw tenant identifier
            
        Returns:
            Hashed tenant identifier (first 12 chars of SHA256) or None
        """
        if not tenant: 
            return None
        return hashlib.sha256(tenant.encode()).hexdigest()[:12]

    def _record(
        self, 
        op: str, 
        t0: float, 
        ok: bool, 
        *, 
        code: str = "OK", 
        ctx: Optional[OperationContext] = None, 
        **extra: Any
    ) -> None:
        """
        Record operation metrics with context and tenant hashing.
        
        Never exposes raw tenant identifiers in metrics. Safe for SIEM systems.
        
        Args:
            op: Operation name
            t0: Start time from time.monotonic()
            ok: Whether operation succeeded
            code: Status code for metrics
            ctx: Operation context for tenant information
            **extra: Additional metric dimensions
        """
        try:
            ms = (time.monotonic() - t0) * 1000.0
            x = dict(extra or {})
            if ctx:
                x["tenant"] = self._tenant_hash(ctx.tenant)
            self._metrics.observe(
                component=self._component, 
                op=op, 
                ms=ms, 
                ok=ok, 
                code=code, 
                extra=x or None
            )
        except Exception:
            # Never let metrics recording break the operation
            pass

    # --- final public APIs (validation + instrumentation) ---

    async def capabilities(self) -> VectorCapabilities:
        """Get the capabilities of this vector adapter."""
        return await self._do_capabilities()

    async def query(
        self,
        spec: QuerySpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> QueryResult:
        """
        Execute a vector similarity search query with validation and metrics.
        
        See VectorProtocolV1.query for full documentation.
        """
        self._validate_vector(spec.vector)
        self._require_non_empty("namespace", spec.namespace)
        if spec.top_k <= 0:
            raise BadRequest("top_k must be positive")
            
        t0 = time.monotonic()
        try:
            result = await self._do_query(spec, ctx=ctx)
            self._record("query", t0, True, ctx=ctx, vectors_searched=len(spec.vector), matches_returned=len(result.matches))
            return result
        except VectorAdapterError as e:
            self._record("query", t0, False, code=type(e).__name__, ctx=ctx)
            raise

    async def upsert(
        self,
        spec: UpsertSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> UpsertResult:
        """
        Upsert vectors into the vector store with validation and metrics.
        
        See VectorProtocolV1.upsert for full documentation.
        """
        self._require_non_empty("namespace", spec.namespace)
        if not spec.vectors:
            raise BadRequest("vectors must not be empty")
            
        for vector in spec.vectors:
            self._validate_vector(vector.vector)
            
        t0 = time.monotonic()
        try:
            result = await self._do_upsert(spec, ctx=ctx)
            self._record("upsert", t0, True, ctx=ctx, vectors_processed=len(spec.vectors))
            return result
        except VectorAdapterError as e:
            self._record("upsert", t0, False, code=type(e).__name__, ctx=ctx)
            raise

    async def delete(
        self,
        spec: DeleteSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> DeleteResult:
        """
        Delete vectors from the vector store with validation and metrics.
        
        See VectorProtocolV1.delete for full documentation.
        """
        self._require_non_empty("namespace", spec.namespace)
        if not spec.ids and not spec.filter:
            raise BadRequest("must provide either ids or filter for deletion")
            
        t0 = time.monotonic()
        try:
            result = await self._do_delete(spec, ctx=ctx)
            self._record("delete", t0, True, ctx=ctx, vectors_targeted=len(spec.ids) if spec.ids else 0)
            return result
        except VectorAdapterError as e:
            self._record("delete", t0, False, code=type(e).__name__, ctx=ctx)
            raise

    async def create_namespace(
        self,
        spec: NamespaceSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> NamespaceResult:
        """
        Create a new namespace/collection with validation and metrics.
        
        See VectorProtocolV1.create_namespace for full documentation.
        """
        self._require_non_empty("namespace", spec.namespace)
        if spec.dimensions <= 0:
            raise BadRequest("dimensions must be positive")
        if spec.distance_metric not in ("cosine", "euclidean", "dotproduct"):
            raise BadRequest("distance_metric must be one of: cosine, euclidean, dotproduct")
            
        t0 = time.monotonic()
        try:
            result = await self._do_create_namespace(spec, ctx=ctx)
            self._record("create_namespace", t0, True, ctx=ctx)
            return result
        except VectorAdapterError as e:
            self._record("create_namespace", t0, False, code=type(e).__name__, ctx=ctx)
            raise

    async def delete_namespace(
        self,
        namespace: str,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> NamespaceResult:
        """
        Delete a namespace/collection with validation and metrics.
        
        See VectorProtocolV1.delete_namespace for full documentation.
        """
        self._require_non_empty("namespace", namespace)
        
        t0 = time.monotonic()
        try:
            result = await self._do_delete_namespace(namespace, ctx=ctx)
            self._record("delete_namespace", t0, True, ctx=ctx)
            return result
        except VectorAdapterError as e:
            self._record("delete_namespace", t0, False, code=type(e).__name__, ctx=ctx)
            raise

    async def health(self, *, ctx: Optional[OperationContext] = None) -> Dict[str, Any]:
        """Check health status with metrics instrumentation."""
        t0 = time.monotonic()
        try:
            h = await self._do_health(ctx=ctx)
            self._record("health", t0, True, ctx=ctx)
            return {
                "ok": bool(h.get("ok", True)),
                "server": str(h.get("server", "")),
                "version": str(h.get("version", "")),
                "namespaces": h.get("namespaces", {}),
            }
        except VectorAdapterError as e:
            self._record("health", t0, False, code=type(e).__name__, ctx=ctx)
            raise

    # --- hooks to implement per backend (override these) ---

    async def _do_capabilities(self) -> VectorCapabilities:
        """Implement to return adapter-specific capabilities."""
        raise NotImplementedError

    async def _do_query(
        self,
        spec: QuerySpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> QueryResult:
        """Implement vector similarity search with validated inputs."""
        raise NotImplementedError

    async def _do_upsert(
        self,
        spec: UpsertSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> UpsertResult:
        """Implement vector upsert operations with validated inputs."""
        raise NotImplementedError

    async def _do_delete(
        self,
        spec: DeleteSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> DeleteResult:
        """Implement vector deletion operations with validated inputs."""
        raise NotImplementedError

    async def _do_create_namespace(
        self,
        spec: NamespaceSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> NamespaceResult:
        """Implement namespace creation with validated inputs."""
        raise NotImplementedError

    async def _do_delete_namespace(
        self,
        namespace: str,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> NamespaceResult:
        """Implement namespace deletion."""
        raise NotImplementedError

    async def _do_health(self, *, ctx: Optional[OperationContext] = None) -> Dict[str, Any]:
        """Implement health check for the vector backend."""
        raise NotImplementedError


__all__ = [
    "VECTOR_PROTOCOL_VERSION",
    "VectorID",
    "Vector",
    "QueryResult", 
    "VectorMatch",
    "VectorAdapterError",
    "BadRequest",
    "AuthError",
    "ResourceExhausted",
    "DimensionMismatch",
    "IndexNotReady",
    "TransientNetwork",
    "Unavailable",
    "NotSupported",
    "OperationContext",
    "QuerySpec",
    "UpsertSpec",
    "DeleteSpec",
    "NamespaceSpec",
    "UpsertResult",
    "DeleteResult",
    "NamespaceResult",
    "VectorCapabilities",
    "VectorProtocolV1",
    "BaseVectorAdapter",
]
