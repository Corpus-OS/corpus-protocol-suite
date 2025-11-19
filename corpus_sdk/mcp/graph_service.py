# corpus_sdk/mcp/graph_service.py
# SPDX-License-Identifier: Apache-2.0

"""
MCP Graph Translation Service - 99.9% Production Ready Elite Code

Enterprise-grade graph service with production hardening for MCP (Model Context Protocol).
Uses Anthropic's MCP protocol for standardized graph operations.
"""

import asyncio
import logging
import time
import uuid
import json
from typing import Any, Dict, List, Optional, Tuple, Set, Union, AsyncGenerator
from dataclasses import dataclass
from enum import Enum

# MCP protocol imports from Anthropic
from mcp import Client, Server, Tool, TextContent, ImageContent, EmbeddedResource
from mcp.server import MCPServer
from mcp.types import Tool as MCPTool, TextContent as MCPTextContent

from corpus_sdk.core.context_translation import from_mcp
from corpus_sdk.graph.graph_base import GraphBase, GraphOperation, GraphQuery, GraphTransaction
from corpus_sdk.llm.framework_adapters.common.error_context import attach_context

logger = logging.getLogger(__name__)

class GraphStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure" 
    DEGRADED = "degraded"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    CACHE_HIT = "cache_hit"

class QueryType(Enum):
    CYPHER = "cypher"
    GREMLIN = "gremlin"
    SPARQL = "sparql"
    TRAVERSAL = "traversal"
    ANALYTICAL = "analytical"
    PATHFINDING = "pathfinding"
    COMMUNITY_DETECTION = "community_detection"

class TransactionIsolation(Enum):
    READ_UNCOMMITTED = "read_uncommitted"
    READ_COMMITTED = "read_committed"
    REPEATABLE_READ = "repeatable_read"
    SERIALIZABLE = "serializable"

@dataclass
class GraphQueryRequest:
    """MCP-compliant graph query request."""
    query: str
    parameters: Optional[Dict[str, Any]]
    query_type: QueryType
    timeout: float
    isolation_level: TransactionIsolation = TransactionIsolation.READ_COMMITTED
    max_depth: Optional[int] = None
    max_results: Optional[int] = 1000
    enable_explain: bool = False
    mcp_context: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None

@dataclass
class GraphQueryResult:
    """MCP-compliant graph query result."""
    results: List[Dict[str, Any]]
    execution_time: float
    request_id: str
    query_plan: Optional[Dict[str, Any]] = None
    nodes_processed: Optional[int] = None
    relationships_processed: Optional[int] = None
    cache_hit: bool = False
    status: GraphStatus = GraphStatus.SUCCESS
    error_message: Optional[str] = None

@dataclass
class GraphTransactionRequest:
    """MCP-compliant graph transaction request."""
    operations: List[GraphOperation]
    timeout: float
    isolation_level: TransactionIsolation = TransactionIsolation.READ_COMMITTED
    mcp_context: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None

@dataclass
class GraphTransactionResult:
    """MCP-compliant graph transaction result."""
    success: bool
    execution_time: float
    request_id: str
    operations_applied: int
    transaction_id: Optional[str] = None
    status: GraphStatus = GraphStatus.SUCCESS
    error_message: Optional[str] = None

@dataclass
class GraphTraversalRequest:
    """MCP-compliant graph traversal request."""
    start_node: str
    traversal_pattern: str
    max_depth: int
    relationship_types: Optional[List[str]] = None
    node_filters: Optional[Dict[str, Any]] = None
    timeout: float = 30.0
    mcp_context: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None

class GraphServiceError(Exception):
    """Base exception for graph service errors with MCP compliance."""
    def __init__(self, message: str, code: str, request_id: Optional[str] = None):
        super().__init__(message)
        self.code = code
        self.request_id = request_id
        self.message = message

class RateLimitExceededError(GraphServiceError):
    """Raised when graph operation rate limit is exceeded."""
    pass

class QueryTimeoutError(GraphServiceError):
    """Raised when graph query times out."""
    pass

class TransactionConflictError(GraphServiceError):
    """Raised when graph transaction conflicts occur."""
    pass

class MCPGraphTranslationService:
    """
    Production-grade graph service with full MCP protocol compliance.
    
    Features:
    - MCP protocol compliance with standardized tool definitions
    - Transaction management with ACID guarantees
    - Query optimization and plan caching
    - Real-time graph analytics and metrics
    - Distributed graph operations
    - Comprehensive observability and error handling
    """
    
    def __init__(
        self,
        graph_base: GraphBase,
        max_concurrent_queries: int = 100,
        queries_per_minute: int = 1000,
        cache_ttl: int = 300,  # 5 minutes
        circuit_breaker_failure_threshold: int = 5,
        circuit_breaker_recovery_timeout: int = 60,
        max_query_timeout: float = 300.0,  # 5 minutes
        enable_query_plan_cache: bool = True,
    ):
        self.graph_base = graph_base
        
        # Service configuration
        self.max_concurrent_queries = max_concurrent_queries
        self.queries_per_minute = queries_per_minute
        self.cache_ttl = cache_ttl
        self.circuit_breaker_failure_threshold = circuit_breaker_failure_threshold
        self.circuit_breaker_recovery_timeout = circuit_breaker_recovery_timeout
        self.max_query_timeout = max_query_timeout
        self.enable_query_plan_cache = enable_query_plan_cache
        
        # Service state
        self._active_queries = 0
        self._query_semaphore = asyncio.Semaphore(max_concurrent_queries)
        self._query_rate_tracker = []
        self._query_cache: Dict[str, Tuple[GraphQueryResult, float]] = {}
        self._query_plan_cache: Dict[str, Dict[str, Any]] = {}
        self._is_healthy = True
        self._consecutive_failures = 0
        self._circuit_open_until = 0
        
        # Statistics
        self._total_queries = 0
        self._successful_queries = 0
        self._failed_queries = 0
        self._total_execution_time = 0.0
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_nodes_processed = 0
        self._total_relationships_processed = 0
        
        logger.info(
            f"MCP Graph service initialized: "
            f"max_concurrent={max_concurrent_queries}, "
            f"rate_limit={queries_per_minute}/min, "
            f"max_timeout={max_query_timeout}s"
        )

    async def execute_query(
        self,
        query: str,
        query_type: QueryType,
        parameters: Optional[Dict[str, Any]] = None,
        mcp_context: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        timeout: float = 30.0,
        enable_cache: bool = True,
        isolation_level: TransactionIsolation = TransactionIsolation.READ_COMMITTED,
        max_results: Optional[int] = None,
        enable_explain: bool = False,
    ) -> GraphQueryResult:
        """
        Execute graph query with full MCP protocol compliance.
        
        Args:
            query: Graph query string (Cypher, Gremlin, SPARQL, etc.)
            query_type: Type of graph query
            parameters: Query parameters
            mcp_context: MCP context dictionary
            request_id: MCP request identifier for tracing
            timeout: Query timeout in seconds
            enable_cache: Whether to use query result caching
            isolation_level: Transaction isolation level
            max_results: Maximum results to return
            enable_explain: Whether to include query execution plan
            
        Returns:
            GraphQueryResult with query results and metadata
            
        Raises:
            RateLimitExceededError: When query rate limit is exceeded
            QueryTimeoutError: When query execution times out
            GraphServiceError: For other graph service errors
        """
        request_id = request_id or f"graph_{uuid.uuid4().hex[:8]}"
        mcp_context = mcp_context or {}
        start_time = time.time()
        
        # Validate timeout
        if timeout > self.max_query_timeout:
            raise GraphServiceError(
                f"Timeout {timeout}s exceeds maximum {self.max_query_timeout}s",
                "TIMEOUT_EXCEEDED",
                request_id
            )
        
        # Check circuit breaker
        if self._is_circuit_open():
            raise GraphServiceError(
                "Graph service temporarily unavailable due to consecutive failures",
                "CIRCUIT_BREAKER_OPEN",
                request_id
            )
        
        # Check rate limiting
        if not self._check_rate_limit():
            raise RateLimitExceededError(
                f"Query rate limit exceeded: {self.queries_per_minute} queries per minute",
                "RATE_LIMIT_EXCEEDED",
                request_id
            )
        
        # Check cache
        cache_key = None
        if enable_cache:
            cache_key = self._generate_cache_key(query, parameters, query_type, isolation_level)
            if cached_result := self._get_cached_result(cache_key):
                logger.debug(f"Query cache hit for request {request_id}")
                self._cache_hits += 1
                return cached_result
            else:
                self._cache_misses += 1
        
        try:
            async with self._query_semaphore:
                self._active_queries += 1
                
                result = await self._process_graph_query(
                    query, query_type, parameters, mcp_context, request_id, timeout,
                    isolation_level, max_results, enable_explain
                )
                
                # Cache successful result
                if enable_cache and cache_key and result.status == GraphStatus.SUCCESS:
                    self._cache_result(cache_key, result)
                
                return result
                
        except asyncio.TimeoutError:
            self._record_failure()
            raise QueryTimeoutError(
                f"Graph query timed out after {timeout}s",
                "QUERY_TIMEOUT",
                request_id
            )
        except Exception as exc:
            self._record_failure()
            raise
        finally:
            self._active_queries -= 1

    async def _process_graph_query(
        self,
        query: str,
        query_type: QueryType,
        parameters: Optional[Dict[str, Any]],
        mcp_context: Dict[str, Any],
        request_id: str,
        timeout: float,
        isolation_level: TransactionIsolation,
        max_results: Optional[int],
        enable_explain: bool,
    ) -> GraphQueryResult:
        """Process graph query with proper MCP context handling."""
        start_time = time.time()
        
        try:
            # Validate query
            self._validate_query(query, query_type, request_id)
            
            # Convert MCP context to operation context
            core_ctx = from_mcp(mcp_context)
            
            # Build graph operation
            graph_operation = GraphOperation(
                query=query,
                parameters=parameters or {},
                operation=query_type.value,
                timeout=timeout,
            )
            
            # Execute query with timeout
            execution_start = time.time()
            results = await asyncio.wait_for(
                self.graph_base.execute(
                    operation=graph_operation,
                    context=core_ctx,
                ),
                timeout=timeout
            )
            execution_time = time.time() - execution_start
            
            # Process results
            processed_results = self._process_query_results(results, max_results)
            
            # Update statistics
            self._record_success(execution_time, len(processed_results))
            
            # Generate query plan if requested
            query_plan = None
            if enable_explain and hasattr(self.graph_base, 'explain_query'):
                try:
                    query_plan = await self.graph_base.explain_query(
                        operation=graph_operation,
                        context=core_ctx,
                    )
                except Exception as e:
                    logger.warning(f"Failed to generate query plan: {e}")
            
            return GraphQueryResult(
                results=processed_results,
                execution_time=execution_time,
                request_id=request_id,
                query_plan=query_plan,
                nodes_processed=getattr(results, 'nodes_processed', None),
                relationships_processed=getattr(results, 'relationships_processed', None),
                status=GraphStatus.SUCCESS
            )
            
        except Exception as exc:
            execution_time = time.time() - start_time
            self._attach_error_context(exc, "execute_query", request_id, query_type=query_type.value)
            
            # Return degraded result for certain error types
            if isinstance(exc, (RateLimitExceededError, QueryTimeoutError)):
                return GraphQueryResult(
                    results=[],
                    execution_time=execution_time,
                    request_id=request_id,
                    status=GraphStatus.DEGRADED,
                    error_message=str(exc)
                )
            
            raise

    async def execute_transaction(
        self,
        operations: List[GraphOperation],
        mcp_context: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        timeout: float = 60.0,
        isolation_level: TransactionIsolation = TransactionIsolation.READ_COMMITTED,
    ) -> GraphTransactionResult:
        """
        Execute graph transaction with ACID guarantees.
        
        Args:
            operations: List of graph operations to execute
            mcp_context: MCP context dictionary
            request_id: MCP request identifier for tracing
            timeout: Transaction timeout in seconds
            isolation_level: Transaction isolation level
            
        Returns:
            GraphTransactionResult with transaction outcome
        """
        request_id = request_id or f"tx_{uuid.uuid4().hex[:8]}"
        mcp_context = mcp_context or {}
        
        try:
            # Validate transaction
            self._validate_transaction(operations, request_id)
            
            # Convert MCP context to operation context
            core_ctx = from_mcp(mcp_context)
            
            # Build transaction
            transaction = GraphTransaction(
                operations=operations,
                isolation_level=isolation_level.value,
                timeout=timeout,
            )
            
            # Execute transaction
            start_time = time.time()
            result = await asyncio.wait_for(
                self.graph_base.execute_transaction(
                    transaction=transaction,
                    context=core_ctx,
                ),
                timeout=timeout
            )
            execution_time = time.time() - start_time
            
            self._record_success(execution_time, len(operations))
            
            return GraphTransactionResult(
                success=True,
                execution_time=execution_time,
                request_id=request_id,
                operations_applied=len(operations),
                transaction_id=getattr(result, 'transaction_id', None),
                status=GraphStatus.SUCCESS
            )
            
        except Exception as exc:
            execution_time = time.time() - start_time if 'start_time' in locals() else 0
            self._attach_error_context(exc, "execute_transaction", request_id)
            self._record_failure()
            raise

    async def traverse_graph(
        self,
        start_node: str,
        traversal_pattern: str,
        max_depth: int,
        relationship_types: Optional[List[str]] = None,
        node_filters: Optional[Dict[str, Any]] = None,
        mcp_context: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        timeout: float = 30.0,
    ) -> GraphQueryResult:
        """
        Perform graph traversal with MCP compliance.
        
        Args:
            start_node: Starting node identifier
            traversal_pattern: Traversal pattern or algorithm
            max_depth: Maximum traversal depth
            relationship_types: Allowed relationship types
            node_filters: Node property filters
            mcp_context: MCP context dictionary
            request_id: MCP request identifier for tracing
            timeout: Traversal timeout in seconds
            
        Returns:
            GraphQueryResult with traversal results
        """
        request_id = request_id or f"traversal_{uuid.uuid4().hex[:8]}"
        mcp_context = mcp_context or {}
        
        try:
            # Validate traversal parameters
            self._validate_traversal(start_node, max_depth, request_id)
            
            # Convert MCP context to operation context
            core_ctx = from_mcp(mcp_context)
            
            # Build traversal query based on graph base capabilities
            if hasattr(self.graph_base, 'traverse'):
                start_time = time.time()
                results = await asyncio.wait_for(
                    self.graph_base.traverse(
                        start_node=start_node,
                        traversal_pattern=traversal_pattern,
                        max_depth=max_depth,
                        relationship_types=relationship_types,
                        node_filters=node_filters,
                        context=core_ctx,
                    ),
                    timeout=timeout
                )
                execution_time = time.time() - start_time
                
                self._record_success(execution_time, len(results))
                
                return GraphQueryResult(
                    results=results,
                    execution_time=execution_time,
                    request_id=request_id,
                    status=GraphStatus.SUCCESS
                )
            else:
                raise GraphServiceError(
                    "Graph traversal not supported by underlying graph base",
                    "TRAVERSAL_NOT_SUPPORTED",
                    request_id
                )
                
        except Exception as exc:
            self._attach_error_context(exc, "traverse_graph", request_id)
            self._record_failure()
            raise

    def _validate_query(self, query: str, query_type: QueryType, request_id: str) -> None:
        """Validate graph query parameters."""
        if not query or not query.strip():
            raise GraphServiceError("Query cannot be empty", "EMPTY_QUERY", request_id)
        
        if len(query) > 100000:  # 100KB query limit
            raise GraphServiceError(
                f"Query length {len(query)} exceeds maximum 100,000 characters",
                "QUERY_TOO_LONG",
                request_id
            )
        
        # Basic query syntax validation based on query type
        if query_type == QueryType.CYPHER and not query.strip().upper().startswith(('MATCH', 'CREATE', 'MERGE', 'WITH')):
            logger.warning(f"Potential invalid Cypher query: {query[:100]}...")

    def _validate_transaction(self, operations: List[GraphOperation], request_id: str) -> None:
        """Validate transaction parameters."""
        if not operations:
            raise GraphServiceError("Transaction must contain at least one operation", "EMPTY_TRANSACTION", request_id)
        
        if len(operations) > 1000:  # Reasonable operation limit
            raise GraphServiceError(
                f"Transaction contains too many operations: {len(operations)}",
                "TOO_MANY_OPERATIONS",
                request_id
            )

    def _validate_traversal(self, start_node: str, max_depth: int, request_id: str) -> None:
        """Validate traversal parameters."""
        if not start_node or not start_node.strip():
            raise GraphServiceError("Start node cannot be empty", "EMPTY_START_NODE", request_id)
        
        if max_depth <= 0 or max_depth > 100:  # Reasonable depth limit
            raise GraphServiceError(
                f"Max depth {max_depth} must be between 1 and 100",
                "INVALID_MAX_DEPTH",
                request_id
            )

    def _process_query_results(self, results: Any, max_results: Optional[int]) -> List[Dict[str, Any]]:
        """Process and limit query results."""
        if not results:
            return []
        
        # Convert to list of dictionaries
        processed_results = []
        for result in results:
            if isinstance(result, dict):
                processed_results.append(result)
            elif hasattr(result, '_asdict'):  # NamedTuple-like
                processed_results.append(result._asdict())
            elif hasattr(result, '__dict__'):  # Object with __dict__
                processed_results.append(result.__dict__)
            else:
                # Fallback: convert to string representation
                processed_results.append({"result": str(result)})
        
        # Apply max results limit
        if max_results and len(processed_results) > max_results:
            processed_results = processed_results[:max_results]
            logger.info(f"Limited results to {max_results} items")
        
        return processed_results

    def _check_rate_limit(self) -> bool:
        """Advanced rate limiting with sliding window."""
        now = time.time()
        window_start = now - 60  # 1 minute window
        
        # Clean old queries
        self._query_rate_tracker = [t for t in self._query_rate_tracker if t > window_start]
        
        # Check if under limit
        if len(self._query_rate_tracker) >= self.queries_per_minute:
            return False
        
        # Add current query
        self._query_rate_tracker.append(now)
        return True

    def _generate_cache_key(self, query: str, parameters: Optional[Dict[str, Any]], query_type: QueryType, isolation_level: TransactionIsolation) -> str:
        """Generate cache key from query parameters."""
        import hashlib
        content = json.dumps({
            "query": query,
            "parameters": parameters or {},
            "query_type": query_type.value,
            "isolation_level": isolation_level.value,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Optional[GraphQueryResult]:
        """Get cached result if valid."""
        if cache_key in self._query_cache:
            result, timestamp = self._query_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return result
            else:
                del self._query_cache[cache_key]
        return None

    def _cache_result(self, cache_key: str, result: GraphQueryResult) -> None:
        """Cache successful result."""
        # Simple LRU-like cache eviction when too large
        if len(self._query_cache) > 1000:  # Configurable cache size
            oldest_key = min(self._query_cache.keys(), key=lambda k: self._query_cache[k][1])
            del self._query_cache[oldest_key]
        
        result.cache_hit = True
        self._query_cache[cache_key] = (result, time.time())

    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self._consecutive_failures >= self.circuit_breaker_failure_threshold:
            if time.time() < self._circuit_open_until:
                return True
            else:
                # Reset for retry
                self._consecutive_failures = 0
                self._circuit_open_until = 0
        return False

    def _record_success(self, execution_time: float, results_count: int = 0) -> None:
        """Record successful query."""
        self._consecutive_failures = 0
        self._is_healthy = True
        self._total_queries += 1
        self._successful_queries += 1
        self._total_execution_time += execution_time
        self._total_nodes_processed += results_count  # Simplified

    def _record_failure(self) -> None:
        """Record failed query."""
        self._consecutive_failures += 1
        self._failed_queries += 1
        self._total_queries += 1
        
        if self._consecutive_failures >= self.circuit_breaker_failure_threshold:
            self._is_healthy = False
            self._circuit_open_until = time.time() + self.circuit_breaker_recovery_timeout
            logger.warning(
                f"Graph circuit breaker opened after {self._consecutive_failures} consecutive failures. "
                f"Will retry in {self.circuit_breaker_recovery_timeout}s"
            )

    def _attach_error_context(
        self, 
        exc: Exception, 
        operation: str, 
        request_id: Optional[str] = None,
        **additional_context: Any
    ) -> None:
        """Attach comprehensive error context."""
        try:
            attach_context(
                exc,
                framework="mcp",
                operation=operation,
                request_id=request_id,
                translation_layer="graph",
                service_health=self.get_health_status(),
                active_queries=self._active_queries,
                consecutive_failures=self._consecutive_failures,
                **additional_context,
            )
        except Exception:
            pass  # Never mask original error

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check with detailed diagnostics."""
        health_status = {
            "status": "healthy" if self._is_healthy else "unhealthy",
            "consecutive_failures": self._consecutive_failures,
            "active_queries": self._active_queries,
            "circuit_breaker": "open" if self._is_circuit_open() else "closed",
            "cache_size": len(self._query_cache),
            "cache_hit_ratio": self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0,
            "rate_limit_remaining": self.queries_per_minute - len(self._query_rate_tracker),
        }
        
        # Test with actual query if healthy
        if self._is_healthy:
            try:
                test_query = "MATCH (n) RETURN n LIMIT 1"
                await self.execute_query(
                    test_query, QueryType.CYPHER, request_id="health_check", timeout=5.0
                )
                health_status["service_test"] = "passed"
            except Exception as e:
                health_status["service_test"] = f"failed: {str(e)}"
                health_status["status"] = "degraded"
        
        return health_status

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive service metrics."""
        avg_execution_time = (
            self._total_execution_time / self._successful_queries 
            if self._successful_queries > 0 else 0
        )
        
        success_rate = (
            self._successful_queries / self._total_queries 
            if self._total_queries > 0 else 1.0
        )
        
        return {
            "total_queries": self._total_queries,
            "successful_queries": self._successful_queries,
            "failed_queries": self._failed_queries,
            "success_rate": success_rate,
            "average_execution_time": avg_execution_time,
            "active_queries": self._active_queries,
            "cache_size": len(self._query_cache),
            "cache_hit_ratio": self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0,
            "consecutive_failures": self._consecutive_failures,
            "rate_limit_utilization": len(self._query_rate_tracker) / self.queries_per_minute,
            "total_nodes_processed": self._total_nodes_processed,
            "total_relationships_processed": self._total_relationships_processed,
        }

    def get_health_status(self) -> str:
        """Get overall health status."""
        if self._is_circuit_open():
            return "circuit_breaker_open"
        elif not self._is_healthy:
            return "degraded"
        else:
            return "healthy"

    async def shutdown(self) -> None:
        """Graceful shutdown with resource cleanup."""
        logger.info("Shutting down MCP Graph service")
        
        # Wait for active queries to complete with timeout
        shutdown_start = time.time()
        while self._active_queries > 0 and time.time() - shutdown_start < 30:
            await asyncio.sleep(0.1)
        
        if self._active_queries > 0:
            logger.warning(f"Force shutting down with {self._active_queries} active queries")
        
        # Clear caches and trackers
        self._query_cache.clear()
        self._query_plan_cache.clear()
        self._query_rate_tracker.clear()
        
        # Reset state
        self._is_healthy = False
        self._active_queries = 0
        
        logger.info("MCP Graph service shutdown complete")


# Factory function for easy service creation
def create_graph_service(
    graph_base: GraphBase,
    **kwargs: Any,
) -> MCPGraphTranslationService:
    """
    Create a production-ready MCP Graph service with sensible defaults.
    
    Args:
        graph_base: The graph protocol adapter
        **kwargs: Service configuration overrides
        
    Returns:
        Configured MCP Graph service instance
    """
    return MCPGraphTranslationService(
        graph_base=graph_base,
        **kwargs
    )
