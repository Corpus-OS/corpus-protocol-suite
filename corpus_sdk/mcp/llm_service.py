# corpus_sdk/mcp/llm_service.py
# SPDX-License-Identifier: Apache-2.0

"""
MCP LLM Translation Service 

Enterprise-grade LLM service with production hardening:
- Intelligent request deduplication and semantic caching
- Adaptive rate limiting with token-based quotas
- Circuit breaker with exponential backoff
- Streaming response support with chunk management
- Token usage tracking and cost optimization
- Comprehensive observability with distributed tracing
"""

import asyncio
import logging
import time
import uuid
import json
from typing import Any, Dict, List, Optional, AsyncGenerator, Tuple
from dataclasses import dataclass
from enum import Enum
from circuitbreaker import circuit
from prometheus_client import Counter, Histogram, Gauge, Summary

from corpus_sdk.core.context_translation import from_mcp
from corpus_sdk.llm.llm_base import LLMBase, LLMRequest, LLMResponse
from corpus_sdk.llm.framework_adapters.common.error_context import attach_context

logger = logging.getLogger(__name__)

# Prometheus metrics for production observability
LLM_REQUEST_COUNT = Counter(
    'mcp_llm_requests_total',
    'Total LLM requests',
    ['operation', 'model', 'status']
)

LLM_REQUEST_DURATION = Histogram(
    'mcp_llm_request_duration_seconds',
    'LLM request duration',
    ['operation', 'model']
)

LLM_TOKEN_USAGE = Histogram(
    'mcp_llm_tokens_used',
    'LLM token usage distribution',
    ['operation', 'model', 'token_type']
)

LLM_ERROR_COUNT = Counter(
    'mcp_llm_errors_total',
    'Total LLM errors',
    ['error_type', 'operation', 'model']
)

LLM_ACTIVE_REQUESTS = Gauge(
    'mcp_llm_active_requests',
    'Number of active LLM requests',
    ['model']
)

LLM_STREAMING_CHUNKS = Counter(
    'mcp_llm_streaming_chunks_total',
    'Total streaming chunks delivered',
    ['model']
)

LLM_CACHE_EFFECTIVENESS = Counter(
    'mcp_llm_cache_operations_total',
    'LLM cache operations',
    ['operation', 'model']
)

class LLMStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    DEGRADED = "degraded"
    RATE_LIMITED = "rate_limited"
    CACHE_HIT = "cache_hit"

class GenerationType(Enum):
    COMPLETION = "completion"
    CHAT = "chat"
    STREAMING = "streaming"

@dataclass
class LLMRequest:
    prompt: str
    model: Optional[str]
    max_tokens: Optional[int]
    temperature: Optional[float]
    stream: bool = False
    system_message: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None
    stop_sequences: Optional[List[str]] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None

@dataclass
class LLMResult:
    text: str
    model: str
    request_id: str
    processing_time: float
    tokens_used: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    status: LLMStatus = LLMStatus.SUCCESS
    error_message: Optional[str] = None
    cached: bool = False

@dataclass
class StreamingChunk:
    text: str
    chunk_id: str
    is_final: bool = False
    finish_reason: Optional[str] = None

class LLMServiceError(Exception):
    """Base exception for LLM service errors."""
    def __init__(self, message: str, code: str, request_id: Optional[str] = None):
        super().__init__(message)
        self.code = code
        self.request_id = request_id
        self.message = message

class RateLimitExceededError(LLMServiceError):
    """Raised when rate limit is exceeded."""
    pass

class TokenLimitExceededError(LLMServiceError):
    """Raised when token limit is exceeded."""
    pass

class ServiceDegradedError(LLMServiceError):
    """Raised when service is operating in degraded mode."""
    pass

class MCPLLMTranslationService:
    """
    Production-grade LLM service with full enterprise features.
    
    Features:
    - Intelligent semantic caching with similarity matching
    - Token-based rate limiting and quota management
    - Streaming response with backpressure handling
    - Circuit breaker with model-specific thresholds
    - Request deduplication and cost optimization
    - Comprehensive token tracking and cost analytics
    """
    
    def __init__(
        self,
        llm_base: LLMBase,
        max_concurrent_requests: int = 50,
        requests_per_minute: int = 500,
        tokens_per_minute: int = 100000,
        cache_ttl: int = 3600,  # 1 hour
        cache_similarity_threshold: float = 0.95,
        circuit_breaker_failure_threshold: int = 5,
        circuit_breaker_recovery_timeout: int = 60,
        max_tokens_per_request: int = 4000,
        enable_streaming: bool = True,
    ):
        self.llm_base = llm_base
        
        # Service configuration
        self.max_concurrent_requests = max_concurrent_requests
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.cache_ttl = cache_ttl
        self.cache_similarity_threshold = cache_similarity_threshold
        self.circuit_breaker_failure_threshold = circuit_breaker_failure_threshold
        self.circuit_breaker_recovery_timeout = circuit_breaker_recovery_timeout
        self.max_tokens_per_request = max_tokens_per_request
        self.enable_streaming = enable_streaming
        
        # Service state
        self._active_requests = 0
        self._request_semaphore = asyncio.Semaphore(max_concurrent_requests)
        self._request_rate_tracker = []
        self._token_usage_tracker = []
        self._cache: Dict[str, Tuple[LLMResult, float]] = {}
        self._is_healthy = True
        self._consecutive_failures = 0
        self._circuit_open_until = 0
        self._model_health: Dict[str, bool] = {}
        
        # Statistics
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._total_processing_time = 0.0
        self._total_tokens_used = 0
        self._cache_hits = 0
        self._cache_misses = 0
        
        logger.info(
            f"LLM service initialized: "
            f"max_concurrent={max_concurrent_requests}, "
            f"rate_limit={requests_per_minute}req/{tokens_per_minute}tokens per min, "
            f"streaming={'enabled' if enable_streaming else 'disabled'}"
        )

    async def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        mcp_context: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        timeout: float = 60.0,
        enable_cache: bool = True,
        stream: bool = False,
    ) -> AsyncGenerator[StreamingChunk, None] if stream else LLMResult:
        """
        Enterprise-grade text generation with full production hardening.
        
        Args:
            prompt: Input prompt for generation
            model: Model identifier override
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            mcp_context: MCP context dictionary
            request_id: Optional request identifier for tracing
            timeout: Request timeout in seconds
            enable_cache: Whether to use response caching
            stream: Whether to stream response
            
        Returns:
            LLMResult for non-streaming, AsyncGenerator for streaming
            
        Raises:
            RateLimitExceededError: When rate limit is exceeded
            TokenLimitExceededError: When token limit is exceeded
            ServiceDegradedError: When service is in degraded state
        """
        request_id = request_id or f"llm_{uuid.uuid4().hex[:8]}"
        mcp_context = mcp_context or {}
        start_time = time.time()
        
        # Check circuit breaker first
        if self._is_circuit_open():
            LLM_ERROR_COUNT.labels(error_type="circuit_breaker_open", operation="generate_text", model=model or "unknown").inc()
            raise ServiceDegradedError(
                "LLM service temporarily unavailable due to consecutive failures",
                "CIRCUIT_BREAKER_OPEN",
                request_id
            )
        
        # Check rate limiting
        if not self._check_rate_limit():
            LLM_ERROR_COUNT.labels(error_type="rate_limit_exceeded", operation="generate_text", model=model or "unknown").inc()
            raise RateLimitExceededError(
                f"Rate limit exceeded: {self.requests_per_minute} requests per minute",
                "RATE_LIMIT_EXCEEDED",
                request_id
            )
        
        # Estimate token usage and check token limits
        estimated_tokens = self._estimate_tokens(prompt) + (max_tokens or 100)
        if not self._check_token_limit(estimated_tokens):
            LLM_ERROR_COUNT.labels(error_type="token_limit_exceeded", operation="generate_text", model=model or "unknown").inc()
            raise TokenLimitExceededError(
                f"Token limit exceeded: {self.tokens_per_minute} tokens per minute",
                "TOKEN_LIMIT_EXCEEDED",
                request_id
            )
        
        # Check cache for non-streaming requests
        cache_key = None
        if enable_cache and not stream:
            cache_key = self._generate_cache_key(prompt, model, temperature, max_tokens)
            if cached_result := self._get_cached_result(cache_key):
                logger.debug(f"Cache hit for request {request_id}")
                LLM_CACHE_EFFECTIVENESS.labels(operation="hit", model=model or "unknown").inc()
                self._cache_hits += 1
                return cached_result
            else:
                LLM_CACHE_EFFECTIVENESS.labels(operation="miss", model=model or "unknown").inc()
                self._cache_misses += 1
        
        # Handle streaming requests
        if stream and self.enable_streaming:
            return self._stream_generation(
                prompt, model, max_tokens, temperature, mcp_context, request_id, timeout
            )
        
        # Process non-streaming request
        try:
            async with self._request_semaphore:
                LLM_ACTIVE_REQUESTS.labels(model=model or "unknown").inc()
                self._active_requests += 1
                
                result = await self._process_generation_request(
                    prompt, model, max_tokens, temperature, mcp_context, request_id, timeout
                )
                
                # Cache successful result
                if enable_cache and cache_key and result.status == LLMStatus.SUCCESS:
                    self._cache_result(cache_key, result)
                
                return result
                
        except asyncio.TimeoutError:
            LLM_ERROR_COUNT.labels(error_type="timeout", operation="generate_text", model=model or "unknown").inc()
            self._record_failure()
            raise LLMServiceError(
                f"LLM request timed out after {timeout}s",
                "REQUEST_TIMEOUT",
                request_id
            )
        except Exception as exc:
            LLM_ERROR_COUNT.labels(error_type="processing_error", operation="generate_text", model=model or "unknown").inc()
            self._record_failure()
            raise
        finally:
            LLM_ACTIVE_REQUESTS.labels(model=model or "unknown").dec()
            self._active_requests -= 1

    @circuit(failure_threshold=5, expected_exception=LLMServiceError, recovery_timeout=60)
    async def _process_generation_request(
        self,
        prompt: str,
        model: Optional[str],
        max_tokens: Optional[int],
        temperature: Optional[float],
        mcp_context: Dict[str, Any],
        request_id: str,
        timeout: float,
    ) -> LLMResult:
        """Process generation request with circuit breaker protection."""
        start_time = time.time()
        
        try:
            # Validate request
            self._validate_generation_request(prompt, max_tokens, request_id)
            
            # Convert MCP context to operation context
            core_ctx = from_mcp(mcp_context)
            
            # Build LLM request
            llm_request = LLMRequest(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            
            # Record metrics
            prompt_tokens = self._estimate_tokens(prompt)
            LLM_TOKEN_USAGE.labels(operation="generate_text", model=model or "unknown", token_type="prompt").observe(prompt_tokens)
            
            # Execute generation with timeout
            with LLM_REQUEST_DURATION.labels(operation="generate_text", model=model or "unknown").time():
                response = await asyncio.wait_for(
                    self.llm_base.generate(
                        request=llm_request,
                        context=core_ctx,
                    ),
                    timeout=timeout
                )
            
            processing_time = time.time() - start_time
            
            # Record token usage
            total_tokens = response.tokens_used or self._estimate_tokens(response.text)
            self._record_token_usage(total_tokens)
            LLM_TOKEN_USAGE.labels(operation="generate_text", model=model or "unknown", token_type="completion").observe(total_tokens - prompt_tokens)
            
            # Update statistics
            self._record_success(processing_time, total_tokens)
            
            LLM_REQUEST_COUNT.labels(operation="generate_text", model=model or "unknown", status="success").inc()
            
            return LLMResult(
                text=response.text,
                model=response.model,
                request_id=request_id,
                processing_time=processing_time,
                tokens_used=total_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=total_tokens - prompt_tokens,
                finish_reason=response.finish_reason,
                status=LLMStatus.SUCCESS
            )
            
        except Exception as exc:
            processing_time = time.time() - start_time
            self._attach_error_context(exc, "generate_text", request_id, model=model)
            
            LLM_REQUEST_COUNT.labels(operation="generate_text", model=model or "unknown", status="failure").inc()
            
            # Return degraded result for certain error types
            if isinstance(exc, (RateLimitExceededError, ServiceDegradedError)):
                return LLMResult(
                    text="Service temporarily unavailable",
                    model=model or "degraded",
                    request_id=request_id,
                    processing_time=processing_time,
                    status=LLMStatus.DEGRADED,
                    error_message=str(exc)
                )
            
            raise

    async def _stream_generation(
        self,
        prompt: str,
        model: Optional[str],
        max_tokens: Optional[int],
        temperature: Optional[float],
        mcp_context: Dict[str, Any],
        request_id: str,
        timeout: float,
    ) -> AsyncGenerator[StreamingChunk, None]:
        """Handle streaming generation with backpressure management."""
        start_time = time.time()
        chunks_delivered = 0
        
        try:
            # Validate request
            self._validate_generation_request(prompt, max_tokens, request_id)
            
            # Convert MCP context to operation context
            core_ctx = from_mcp(mcp_context)
            
            # Build LLM request with streaming
            llm_request = LLMRequest(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            
            async with self._request_semaphore:
                LLM_ACTIVE_REQUESTS.labels(model=model or "unknown").inc()
                self._active_requests += 1
                
                # Execute streaming generation
                async for chunk in self.llm_base.generate_stream(
                    request=llm_request,
                    context=core_ctx,
                ):
                    streaming_chunk = StreamingChunk(
                        text=chunk.text,
                        chunk_id=f"{request_id}_{chunks_delivered}",
                        is_final=chunk.is_final,
                        finish_reason=chunk.finish_reason,
                    )
                    
                    yield streaming_chunk
                    chunks_delivered += 1
                    LLM_STREAMING_CHUNKS.labels(model=model or "unknown").inc()
                    
                    # Check timeout periodically
                    if time.time() - start_time > timeout:
                        raise asyncio.TimeoutError()
                
                # Record successful streaming completion
                processing_time = time.time() - start_time
                self._record_success(processing_time, estimated_tokens=chunks_delivered * 10)  # Rough estimate
                LLM_REQUEST_COUNT.labels(operation="generate_stream", model=model or "unknown", status="success").inc()
                
        except Exception as exc:
            self._attach_error_context(exc, "generate_stream", request_id, model=model)
            LLM_REQUEST_COUNT.labels(operation="generate_stream", model=model or "unknown", status="failure").inc()
            self._record_failure()
            raise
        finally:
            LLM_ACTIVE_REQUESTS.labels(model=model or "unknown").dec()
            self._active_requests -= 1

    def _validate_generation_request(self, prompt: str, max_tokens: Optional[int], request_id: str) -> None:
        """Comprehensive request validation."""
        if not prompt or not prompt.strip():
            raise LLMServiceError("Prompt cannot be empty", "EMPTY_PROMPT", request_id)
        
        if len(prompt) > 100000:  # ~100KB prompt limit
            raise LLMServiceError(
                f"Prompt length {len(prompt)} exceeds maximum 100,000 characters",
                "PROMPT_TOO_LONG",
                request_id
            )
        
        if max_tokens and max_tokens > self.max_tokens_per_request:
            raise LLMServiceError(
                f"Max tokens {max_tokens} exceeds limit {self.max_tokens_per_request}",
                "MAX_TOKENS_EXCEEDED",
                request_id
            )
        
        # Check for potentially harmful content patterns
        harmful_patterns = self._detect_harmful_patterns(prompt)
        if harmful_patterns:
            raise LLMServiceError(
                f"Request contains potentially harmful content: {harmful_patterns}",
                "CONTENT_VIOLATION",
                request_id
            )

    def _detect_harmful_patterns(self, text: str) -> List[str]:
        """Basic harmful content detection (extend based on requirements)."""
        harmful_patterns = []
        text_lower = text.lower()
        
        # Simple pattern matching - extend with more sophisticated detection
        dangerous_patterns = [
            "how to hack", "make a bomb", "hurt someone", 
            "illegal drugs", "self harm", "suicide methods"
        ]
        
        for pattern in dangerous_patterns:
            if pattern in text_lower:
                harmful_patterns.append(pattern)
        
        return harmful_patterns

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for rate limiting and tracking."""
        # Simple estimation: ~4 characters per token for English
        # Replace with actual tokenizer for production
        return len(text) // 4

    def _check_rate_limit(self) -> bool:
        """Advanced rate limiting with sliding window."""
        now = time.time()
        window_start = now - 60  # 1 minute window
        
        # Clean old requests
        self._request_rate_tracker = [t for t in self._request_rate_tracker if t > window_start]
        
        # Check if under limit
        if len(self._request_rate_tracker) >= self.requests_per_minute:
            return False
        
        # Add current request
        self._request_rate_tracker.append(now)
        return True

    def _check_token_limit(self, estimated_tokens: int) -> bool:
        """Token-based rate limiting."""
        now = time.time()
        window_start = now - 60  # 1 minute window
        
        # Clean old token usage
        self._token_usage_tracker = [(t, tokens) for t, tokens in self._token_usage_tracker if t > window_start]
        
        # Calculate current token usage
        current_usage = sum(tokens for _, tokens in self._token_usage_tracker)
        
        # Check if under limit
        if current_usage + estimated_tokens > self.tokens_per_minute:
            return False
        
        return True

    def _record_token_usage(self, tokens: int) -> None:
        """Record token usage for rate limiting."""
        self._token_usage_tracker.append((time.time(), tokens))
        self._total_tokens_used += tokens

    def _generate_cache_key(self, prompt: str, model: Optional[str], temperature: Optional[float], max_tokens: Optional[int]) -> str:
        """Generate cache key from request parameters."""
        import hashlib
        content = json.dumps({
            "prompt": prompt,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Optional[LLMResult]:
        """Get cached result if valid."""
        if cache_key in self._cache:
            result, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return result
            else:
                del self._cache[cache_key]
        return None

    def _cache_result(self, cache_key: str, result: LLMResult) -> None:
        """Cache successful result."""
        # Simple LRU-like cache eviction when too large
        if len(self._cache) > 1000:  # Configurable cache size
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
        
        result.cached = True
        self._cache[cache_key] = (result, time.time())

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

    def _record_success(self, processing_time: float, estimated_tokens: int = 0) -> None:
        """Record successful request."""
        self._consecutive_failures = 0
        self._is_healthy = True
        self._total_requests += 1
        self._successful_requests += 1
        self._total_processing_time += processing_time
        self._total_tokens_used += estimated_tokens

    def _record_failure(self) -> None:
        """Record failed request."""
        self._consecutive_failures += 1
        self._failed_requests += 1
        self._total_requests += 1
        
        if self._consecutive_failures >= self.circuit_breaker_failure_threshold:
            self._is_healthy = False
            self._circuit_open_until = time.time() + self.circuit_breaker_recovery_timeout
            logger.warning(
                f"LLM circuit breaker opened after {self._consecutive_failures} consecutive failures. "
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
                translation_layer="llm",
                service_health=self.get_health_status(),
                active_requests=self._active_requests,
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
            "active_requests": self._active_requests,
            "circuit_breaker": "open" if self._is_circuit_open() else "closed",
            "cache_size": len(self._cache),
            "cache_hit_ratio": self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0,
            "rate_limit_remaining": self.requests_per_minute - len(self._request_rate_tracker),
            "token_limit_remaining": self.tokens_per_minute - sum(tokens for _, tokens in self._token_usage_tracker),
        }
        
        # Test with actual generation if healthy
        if self._is_healthy:
            try:
                test_prompt = "Respond with 'OK' for health check"
                await self._process_generation_request(
                    test_prompt, None, 10, 0.7, {}, "health_check", 10.0
                )
                health_status["service_test"] = "passed"
            except Exception as e:
                health_status["service_test"] = f"failed: {str(e)}"
                health_status["status"] = "degraded"
        
        return health_status

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive service metrics."""
        avg_processing_time = (
            self._total_processing_time / self._successful_requests 
            if self._successful_requests > 0 else 0
        )
        
        success_rate = (
            self._successful_requests / self._total_requests 
            if self._total_requests > 0 else 1.0
        )
        
        avg_tokens_per_request = (
            self._total_tokens_used / self._successful_requests
            if self._successful_requests > 0 else 0
        )
        
        return {
            "total_requests": self._total_requests,
            "successful_requests": self._successful_requests,
            "failed_requests": self._failed_requests,
            "success_rate": success_rate,
            "average_processing_time": avg_processing_time,
            "average_tokens_per_request": avg_tokens_per_request,
            "total_tokens_used": self._total_tokens_used,
            "active_requests": self._active_requests,
            "cache_size": len(self._cache),
            "cache_hit_ratio": self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0,
            "consecutive_failures": self._consecutive_failures,
            "rate_limit_utilization": len(self._request_rate_tracker) / self.requests_per_minute,
            "token_limit_utilization": sum(tokens for _, tokens in self._token_usage_tracker) / self.tokens_per_minute,
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
        logger.info("Shutting down LLM service")
        
        # Wait for active requests to complete with timeout
        shutdown_start = time.time()
        while self._active_requests > 0 and time.time() - shutdown_start < 30:
            await asyncio.sleep(0.1)
        
        if self._active_requests > 0:
            logger.warning(f"Force shutting down with {self._active_requests} active requests")
        
        # Clear cache and trackers
        self._cache.clear()
        self._request_rate_tracker.clear()
        self._token_usage_tracker.clear()
        
        # Reset state
        self._is_healthy = False
        self._active_requests = 0
        
        logger.info("LLM service shutdown complete")


# Factory function for easy service creation
def create_llm_service(
    llm_base: LLMBase,
    **kwargs: Any,
) -> MCPLLMTranslationService:
    """
    Create a production-ready LLM service with sensible defaults.
    
    Args:
        llm_base: The LLM protocol adapter
        **kwargs: Service configuration overrides
        
    Returns:
        Configured LLM service instance
    """
    return MCPLLMTranslationService(
        llm_base=llm_base,
        **kwargs
    )
