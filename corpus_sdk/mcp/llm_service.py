# corpus_sdk/mcp/llm_service.py
# SPDX-License-Identifier: Apache-2.0

"""
MCP LLM Translation Service

Enterprise-oriented LLM service with:
- Context-aware MCP → LLM translation via LLMTranslator
- Intelligent request caching
- Token-based rate limiting
- Concurrent request limiting
- Streaming + non-streaming support
- Basic safety / harmful-content validation (configurable)
- Error context attachment for observability
- Enhanced configuration validation
- Improved token estimation with fallback (pluggable)
- Distributed cache interface support
- Comprehensive request prioritization

Health & metrics
----------------
- Health checks are implemented in a way that does NOT consume request/token
  budget or affect normal metrics (no double-counting).
- Metrics access is synchronous and never awaits.

Harmful content policy
----------------------
Harmful-content detection is intentionally lightweight and configurable:

- Default patterns are a small set of dangerous phrases.
- Callers can inject custom patterns or disable detection entirely via
  MCPLLMTranslationService constructor / ServiceConfig.

Token estimation
----------------
Token estimation uses a robust, multi-tiered strategy:

1. Model-specific tiktoken encoding (cached with failure tracking).
2. Default encoding for GPT-style models.
3. Conservative character-based fallback when encodings are unavailable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    Deque,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import tiktoken

from corpus_sdk.core.context_translation import from_mcp
from corpus_sdk.core.error_context import attach_context
from corpus_sdk.llm.llm_base import LLMProtocolV1  # for type hints
from corpus_sdk.llm.framework_adapters.common.llm_translation import (
    LLMTranslator,
    LLMFrameworkTranslator,
    LLMPostProcessingConfig,
    create_llm_translator,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Cache Interface for Distributed Caching
# =============================================================================


class CacheBackend(ABC):
    """Abstract base class for cache backends supporting distributed caching."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache by key."""
        raise NotImplementedError

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int) -> None:
        """Set value in cache with TTL."""
        raise NotImplementedError

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete value from cache by key."""
        raise NotImplementedError

    @abstractmethod
    async def clear(self) -> None:
        """Clear all cached values."""
        raise NotImplementedError

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        raise NotImplementedError


class InMemoryCache(CacheBackend):
    """
    In-memory cache implementation with LRU eviction.

    Notes
    -----
    - LRU eviction is implemented with a simple access-time map and min() scan.
      This is O(N) on eviction, which is acceptable for moderate max_size.
    - This backend is intended primarily for single-process usage; for
      production distributed setups, plug in a distributed CacheBackend.
    """

    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._max_size = max_size
        self._access_times: Dict[str, float] = {}

    async def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            value, expiry = self._cache[key]
            if time.time() < expiry:
                self._access_times[key] = time.time()
                return value
            await self.delete(key)
        return None

    async def set(self, key: str, value: Any, ttl: int) -> None:
        current_time = time.time()

        # Evict if needed (LRU)
        if len(self._cache) >= self._max_size and key not in self._cache:
            if self._access_times:
                lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
                await self.delete(lru_key)

        self._cache[key] = (value, current_time + ttl)
        self._access_times[key] = current_time

    async def delete(self, key: str) -> None:
        self._cache.pop(key, None)
        self._access_times.pop(key, None)

    async def clear(self) -> None:
        self._cache.clear()
        self._access_times.clear()

    async def exists(self, key: str) -> bool:
        if key not in self._cache:
            return False
        _, expiry = self._cache[key]
        return time.time() < expiry


# =============================================================================
# Configuration Validation
# =============================================================================


class ServiceConfigError(ValueError):
    """Raised for invalid LLM service configuration values."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ServiceConfig:
    """Validated service configuration with sensible defaults and constraints."""

    def __init__(
        self,
        max_concurrent_requests: int = 50,
        requests_per_minute: int = 500,
        tokens_per_minute: int = 100_000,
        cache_ttl: int = 3600,
        max_tokens_per_request: int = 4000,
        enable_streaming: bool = True,
        cache_max_size: int = 1000,
        request_timeout: float = 60.0,
        health_check_timeout: float = 5.0,
        circuit_breaker_threshold: int = 5,
        harmful_patterns: Optional[List[str]] = None,
        enable_harmful_content_detection: bool = True,
    ):
        # Validate and set parameters with constraints
        if max_concurrent_requests <= 0:
            raise ServiceConfigError("max_concurrent_requests must be positive")
        self.max_concurrent_requests = max_concurrent_requests

        if requests_per_minute <= 0:
            raise ServiceConfigError("requests_per_minute must be positive")
        self.requests_per_minute = requests_per_minute

        if tokens_per_minute <= 0:
            raise ServiceConfigError("tokens_per_minute must be positive")
        self.tokens_per_minute = tokens_per_minute

        if cache_ttl < 0:
            raise ServiceConfigError("cache_ttl must be non-negative")
        self.cache_ttl = cache_ttl

        if max_tokens_per_request <= 0:
            raise ServiceConfigError("max_tokens_per_request must be positive")
        self.max_tokens_per_request = max_tokens_per_request

        if cache_max_size <= 0:
            raise ServiceConfigError("cache_max_size must be positive")
        self.cache_max_size = cache_max_size

        if request_timeout <= 0:
            raise ServiceConfigError("request_timeout must be positive")
        self.request_timeout = request_timeout

        if health_check_timeout <= 0:
            raise ServiceConfigError("health_check_timeout must be positive")
        self.health_check_timeout = health_check_timeout

        if circuit_breaker_threshold <= 0:
            raise ServiceConfigError("circuit_breaker_threshold must be positive")
        self.circuit_breaker_threshold = circuit_breaker_threshold

        self.enable_streaming = enable_streaming
        self.harmful_patterns = harmful_patterns
        self.enable_harmful_content_detection = enable_harmful_content_detection


# =============================================================================
# Token Estimation with Fallback
# =============================================================================


class TokenEstimator:
    """Advanced token estimation with multiple fallback strategies."""

    def __init__(self):
        self._encoding_cache: Dict[str, tiktoken.Encoding] = {}
        self._encoding_failures: Set[str] = set()
        self._default_encoding: Optional[tiktoken.Encoding] = self._init_default_encoding()

    def _init_default_encoding(self) -> Optional[tiktoken.Encoding]:
        """
        Initialize a safe default encoding.

        Attempts a GPT-style model-specific encoding first, then falls back
        to a generic cl100k_base encoding. If all attempts fail, returns None.
        """
        try:
            return tiktoken.encoding_for_model("gpt-3.5-turbo")
        except Exception:
            try:
                return tiktoken.get_encoding("cl100k_base")
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "TokenEstimator could not initialize default encoding; "
                    "falling back to character-based estimation: %s",
                    exc,
                )
                return None

    def _get_encoding_for_model(self, model: str) -> Optional[tiktoken.Encoding]:
        """Get appropriate encoding for model with caching and failure tracking."""
        if model in self._encoding_cache:
            return self._encoding_cache[model]

        if model in self._encoding_failures:
            return None

        try:
            if "gpt-4" in model or "gpt-3.5" in model:
                encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            elif "claude" in model:
                # Claude uses different tokenization; fall back to conservative estimate
                encoding = None
            else:
                encoding = tiktoken.get_encoding("cl100k_base")

            if encoding is not None:
                self._encoding_cache[model] = encoding
            else:
                self._encoding_failures.add(model)

            return encoding
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to get encoding for model %s: %s", model, exc)
            self._encoding_failures.add(model)
            return None

    def estimate_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Estimate tokens with multiple fallback strategies.

        Strategy priority:
        1. Model-specific tiktoken encoding
        2. Default tiktoken encoding
        3. Conservative character-based estimation
        """
        if not text:
            return 0

        try:
            if model:
                encoding = self._get_encoding_for_model(model)
                if encoding:
                    return len(encoding.encode(text))

            if self._default_encoding:
                return len(self._default_encoding.encode(text))
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Token estimation failed for model %s: %s; "
                "falling back to character-based estimation",
                model,
                exc,
            )

        # Ultimate fallback: conservative character-based estimation
        # Use 3.5 chars per token for safety (more conservative than 4)
        return max(1, len(text) // 3)


# =============================================================================
# Request Prioritization
# =============================================================================


class RequestPriority(Enum):
    HIGH = 0
    NORMAL = 1
    LOW = 2
    BATCH = 3


@dataclass
class PrioritizedRequest:
    """Request with priority metadata."""
    priority: RequestPriority
    created_at: float
    request_id: str
    data: Any

    def __lt__(self, other: "PrioritizedRequest") -> bool:
        """Priority queue ordering (lower value = higher priority, FIFO within same priority)."""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.created_at < other.created_at


class PrioritySemaphore:
    """
    Semaphore with priority-based acquisition.

    Notes
    -----
    - This implementation assumes a single asyncio event loop (no cross-thread use).
    - Higher-priority requests are favored and may starve lower-priority
      requests under sustained high load. This is intentional and should be
      accounted for at the call site.
    """

    def __init__(self, value: int):
        self._value = value
        self._priority_queue: List[PrioritizedRequest] = []

    async def acquire(
        self,
        priority: RequestPriority = RequestPriority.NORMAL,
        request_id: str = "",
    ) -> bool:
        """Acquire semaphore with priority."""
        while self._value <= 0:
            fut: asyncio.Future = asyncio.get_event_loop().create_future()
            self._priority_queue.append(
                PrioritizedRequest(priority, time.time(), request_id, fut)
            )
            self._priority_queue.sort()
            try:
                await fut
            except Exception:  # noqa: BLE001
                self._remove_waiter(fut)
                raise
        self._value -= 1
        return True

    def release(self) -> None:
        """Release semaphore, waking highest priority waiter."""
        self._value += 1
        if self._priority_queue:
            request = self._priority_queue.pop(0)
            if not request.data.done():
                request.data.set_result(True)

    def _remove_waiter(self, fut: asyncio.Future) -> None:
        """Remove waiter from priority queue."""
        self._priority_queue = [r for r in self._priority_queue if r.data is not fut]

    @property
    def value(self) -> int:
        return self._value


# =============================================================================
# Public enums + result types
# =============================================================================


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
class LLMResult:
    """
    High-level result returned to MCP callers for non-streaming generation.
    """

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
    """
    High-level streaming chunk returned to MCP callers.
    """

    text: str
    chunk_id: str
    is_final: bool = False
    finish_reason: Optional[str] = None


# =============================================================================
# Error types
# =============================================================================


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


# =============================================================================
# MCP LLM Translation Service
# =============================================================================


class MCPLLMTranslationService:
    """
    MCP-facing LLM service built on top of the framework-agnostic LLMTranslator.

    Key responsibilities:
    - Convert MCP context into core OperationContext via from_mcp(...)
    - Apply basic pre-flight validation and safety checks
    - Enforce request + token rate limiting
    - Provide streaming and non-streaming interfaces
    - Offer a simple in-memory cache keyed by prompt/model/params
    - Attach rich error context via corpus_sdk.core.error_context.attach_context

    Streaming semantics
    -------------------
    - For `stream=True`, timeout is applied to the total duration of the stream,
      not per-chunk. A timeout results in a STREAM_TIMEOUT error.

    Rate limiting semantics
    -----------------------
    - Request rate limit: sliding 60-second window, per-instance.
    - Token rate limit: sliding 60-second window, estimating completion size
      pessimistically as prompt tokens + (max_tokens or 100).
    """

    def __init__(
        self,
        llm_translator: LLMTranslator,
        *,
        max_concurrent_requests: int = 50,
        requests_per_minute: int = 500,
        tokens_per_minute: int = 100_000,
        cache_ttl: int = 3600,
        max_tokens_per_request: int = 4000,
        enable_streaming: bool = True,
        cache_backend: Optional[CacheBackend] = None,
        cache_max_size: int = 1000,
        request_timeout: float = 60.0,
        health_check_timeout: float = 5.0,
        circuit_breaker_threshold: int = 5,
        harmful_content_patterns: Optional[List[str]] = None,
        enable_harmful_content_detection: bool = True,
        token_estimator: Optional[TokenEstimator] = None,
    ) -> None:
        """
        Args:
            llm_translator:
                Instance of LLMTranslator that wraps an LLMProtocolV1 adapter.
            max_concurrent_requests:
                Maximum concurrent LLM operations.
            requests_per_minute:
                Sliding-window request rate limit.
            tokens_per_minute:
                Sliding-window token rate limit (prompt + completion).
            cache_ttl:
                TTL for cached responses in seconds. Enforced both at cache
                backend level (if supported) and at service level.
            max_tokens_per_request:
                Hard limit for max_tokens parameter.
            enable_streaming:
                Whether streaming APIs are enabled.
            cache_backend:
                Optional distributed cache backend.
            cache_max_size:
                Maximum cache size for in-memory cache.
            request_timeout:
                Default request timeout in seconds for non-streaming calls.
            health_check_timeout:
                Health check timeout in seconds for LLM probe.
            circuit_breaker_threshold:
                Consecutive failures before circuit breaker trips.
            harmful_content_patterns:
                Optional list of additional harmful-content patterns to match.
            enable_harmful_content_detection:
                Whether to perform basic harmful-content screening on prompts.
            token_estimator:
                Optional custom TokenEstimator implementation.
        """
        # Validate configuration
        self._config = ServiceConfig(
            max_concurrent_requests=max_concurrent_requests,
            requests_per_minute=requests_per_minute,
            tokens_per_minute=tokens_per_minute,
            cache_ttl=cache_ttl,
            max_tokens_per_request=max_tokens_per_request,
            enable_streaming=enable_streaming,
            cache_max_size=cache_max_size,
            request_timeout=request_timeout,
            health_check_timeout=health_check_timeout,
            circuit_breaker_threshold=circuit_breaker_threshold,
            harmful_patterns=harmful_content_patterns,
            enable_harmful_content_detection=enable_harmful_content_detection,
        )

        self._llm = llm_translator
        self._token_estimator = token_estimator or TokenEstimator()

        # Cache setup
        self._cache_backend = cache_backend or InMemoryCache(cache_max_size)

        # Concurrency + rate limiting state
        self._active_requests: int = 0
        self._request_semaphore = PrioritySemaphore(max_concurrent_requests)
        self._request_rate_tracker: Deque[float] = deque()  # timestamps of recent requests
        self._token_usage_tracker: Deque[Tuple[float, int]] = deque()  # (timestamp, tokens)

        # Simple health / statistics
        self._is_healthy: bool = True
        self._consecutive_failures: int = 0
        self._total_requests: int = 0
        self._successful_requests: int = 0
        self._failed_requests: int = 0
        self._total_processing_time: float = 0.0
        self._total_tokens_used: int = 0
        self._cache_hits: int = 0
        self._cache_misses: int = 0

        logger.info(
            "MCPLLMTranslationService initialized: "
            "max_concurrent=%s, rate_limit=%s req/min, token_limit=%s tok/min, "
            "streaming=%s, cache_backend=%s",
            max_concurrent_requests,
            requests_per_minute,
            tokens_per_minute,
            "enabled" if enable_streaming else "disabled",
            cache_backend.__class__.__name__ if cache_backend else "InMemoryCache",
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def generate_text(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        mcp_context: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        timeout: Optional[float] = None,
        enable_cache: bool = True,
        stream: bool = False,
        system_message: Optional[str] = None,
        priority: RequestPriority = RequestPriority.NORMAL,
    ) -> Union[LLMResult, AsyncGenerator[StreamingChunk, None]]:
        """
        High-level MCP entrypoint for text generation.

        If `stream` is False:
            Returns an LLMResult.

        If `stream` is True and streaming is enabled:
            Returns an async generator yielding StreamingChunk instances.

        Timeout semantics:
            - Non-streaming: applies to the total completion call.
            - Streaming: applies to the total duration of the stream.

        Raises:
            RateLimitExceededError
            TokenLimitExceededError
            ServiceDegradedError
            LLMServiceError
        """
        request_id = request_id or f"llm_{uuid.uuid4().hex[:8]}"
        mcp_context = mcp_context or {}
        timeout = timeout or self._config.request_timeout
        start_time = time.time()

        # Basic health gate. We preserve the health signal, but do not rely on any
        # external circuit breaker library.
        if not self._is_healthy:
            raise ServiceDegradedError(
                "LLM service is currently in degraded state",
                "SERVICE_DEGRADED",
                request_id,
            )

        # Rate limiting: request-level
        if not self._check_rate_limit():
            self._record_failure()
            raise RateLimitExceededError(
                f"Rate limit exceeded: {self._config.requests_per_minute} requests per minute",
                "RATE_LIMIT_EXCEEDED",
                request_id,
            )

        # Estimate tokens and enforce token-level rate limit
        # We pessimistically assume a completion size of (max_tokens or 100).
        estimated_tokens = self._estimate_tokens(prompt, model) + (max_tokens or 100)
        if not self._check_token_limit(estimated_tokens):
            self._record_failure()
            raise TokenLimitExceededError(
                f"Token limit exceeded: {self._config.tokens_per_minute} tokens per minute",
                "TOKEN_LIMIT_EXCEEDED",
                request_id,
            )

        # Cache lookup for non-streaming requests
        cache_key: Optional[str] = None
        if enable_cache and not stream:
            cache_key = self._generate_cache_key(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                system_message=system_message,
            )
            cached_result = await self._get_cached_result(cache_key)
            if cached_result is not None:
                logger.debug("Cache hit for request_id=%s", request_id)
                cached_result.status = LLMStatus.CACHE_HIT
                cached_result.cached = True
                self._cache_hits += 1
                return cached_result
            self._cache_misses += 1

        if stream:
            if not self._config.enable_streaming:
                raise LLMServiceError(
                    "Streaming is disabled for this service",
                    "STREAMING_DISABLED",
                    request_id,
                )
            # Streaming path
            return self._stream_generation(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                mcp_context=mcp_context,
                request_id=request_id,
                timeout=timeout,
                system_message=system_message,
                priority=priority,
            )

        # Non-streaming path
        try:
            await self._request_semaphore.acquire(priority, request_id)
            self._active_requests += 1
            result = await self._process_generation_request(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                mcp_context=mcp_context,
                request_id=request_id,
                timeout=timeout,
                system_message=system_message,
                start_time=start_time,
            )

            # Cache successful result
            if enable_cache and cache_key and result.status == LLMStatus.SUCCESS:
                await self._cache_result(cache_key, result)

            return result

        except asyncio.TimeoutError:
            self._record_failure()
            raise LLMServiceError(
                f"LLM request timed out after {timeout}s",
                "REQUEST_TIMEOUT",
                request_id,
            )
        except Exception as exc:  # noqa: BLE001
            self._attach_error_context(
                exc,
                operation="generate_text",
                request_id=request_id,
                model=model,
            )
            self._record_failure()
            raise
        finally:
            self._request_semaphore.release()
            self._active_requests = max(0, self._active_requests - 1)

    # -------------------------------------------------------------------------
    # Internal non-streaming execution
    # -------------------------------------------------------------------------

    async def _process_generation_request(
        self,
        *,
        prompt: str,
        model: Optional[str],
        max_tokens: Optional[int],
        temperature: Optional[float],
        mcp_context: Dict[str, Any],
        request_id: str,
        timeout: float,
        system_message: Optional[str],
        start_time: float,
    ) -> LLMResult:
        """
        Execute a single non-streaming generation request via LLMTranslator.
        """
        # Validate request & enforce per-request constraints
        self._validate_generation_request(prompt, max_tokens, request_id)

        # Convert MCP context into core OperationContext for the LLMTranslator
        op_ctx = from_mcp(mcp_context)

        # Build chat-style messages. We intentionally keep it simple:
        # optional system message + single user message.
        messages: List[Dict[str, str]] = [{"role": "user", "content": prompt}]

        prompt_tokens = self._estimate_tokens(prompt, model)

        try:
            # Enforce timeout at coroutine level
            completion_obj = await asyncio.wait_for(
                self._llm.arun_complete(
                    raw_messages=messages,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system_message=system_message,
                    op_ctx=op_ctx,
                    framework_ctx=None,
                ),
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001
            processing_time = time.time() - start_time
            self._attach_error_context(
                exc,
                operation="generate_text",
                request_id=request_id,
                model=model,
                processing_time=processing_time,
            )
            raise

        processing_time = time.time() - start_time
        # Normalize translator result into the MCP-level LLMResult
        result = self._normalize_completion_result(
            completion=completion_obj,
            request_id=request_id,
            processing_time=processing_time,
            prompt_tokens=prompt_tokens,
            model_hint=model,
        )
        return result

    def _normalize_completion_result(
        self,
        *,
        completion: Any,
        request_id: str,
        processing_time: float,
        prompt_tokens: int,
        model_hint: Optional[str],
    ) -> LLMResult:
        """
        Normalize whatever the LLMTranslator returns into an LLMResult.

        The default LLMTranslator returns a dict:
            {
                "text": ...,
                "model": ...,
                "model_family": ...,
                "usage": {
                    "prompt_tokens": ...,
                    "completion_tokens": ...,
                    "total_tokens": ...,
                },
                "finish_reason": ...,
                "tool_calls": [...],
            }

        But we defensively handle attribute-style objects as well.
        """
        text: str = ""
        model: str = model_hint or "unknown"
        finish_reason: Optional[str] = None
        total_tokens: Optional[int] = None
        completion_tokens: Optional[int] = None
        effective_prompt_tokens: Optional[int] = prompt_tokens

        if isinstance(completion, dict):
            text = str(
                completion.get("text")
                or completion.get("content")
                or completion.get("message", {}).get("content", "")
            )
            model = str(completion.get("model") or model_hint or "unknown")
            finish_reason = completion.get("finish_reason")

            usage = completion.get("usage") or {}
            pt = usage.get("prompt_tokens")
            if isinstance(pt, int):
                effective_prompt_tokens = pt
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")
            if (
                total_tokens is None
                and effective_prompt_tokens is not None
                and completion_tokens is not None
            ):
                total_tokens = effective_prompt_tokens + completion_tokens
        else:
            # Attribute-based object
            text = str(getattr(completion, "text", "") or "")
            model = str(getattr(completion, "model", model_hint or "unknown"))
            finish_reason = getattr(completion, "finish_reason", None)

            usage = getattr(completion, "usage", None)
            if usage is not None:
                pt = getattr(usage, "prompt_tokens", None)
                if isinstance(pt, int):
                    effective_prompt_tokens = pt
                completion_tokens = getattr(usage, "completion_tokens", None)
                total_tokens = getattr(usage, "total_tokens", None)
                if (
                    total_tokens is None
                    and effective_prompt_tokens is not None
                    and completion_tokens is not None
                ):
                    total_tokens = effective_prompt_tokens + completion_tokens

        # Fallback token estimation if not provided
        if total_tokens is None:
            total_tokens = (effective_prompt_tokens or 0) + self._estimate_tokens(text, model)
        if completion_tokens is None and effective_prompt_tokens is not None:
            completion_tokens = max(0, total_tokens - effective_prompt_tokens)

        # Record token usage + success statistics (avoid double-counting)
        self._record_token_usage(total_tokens)
        self._record_success(processing_time)

        return LLMResult(
            text=text,
            model=model,
            request_id=request_id,
            processing_time=processing_time,
            tokens_used=total_tokens,
            prompt_tokens=effective_prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            status=LLMStatus.SUCCESS,
            error_message=None,
            cached=False,
        )

    # -------------------------------------------------------------------------
    # Streaming execution
    # -------------------------------------------------------------------------

    async def _stream_generation(
        self,
        *,
        prompt: str,
        model: Optional[str],
        max_tokens: Optional[int],
        temperature: Optional[float],
        mcp_context: Dict[str, Any],
        request_id: str,
        timeout: float,
        system_message: Optional[str],
        priority: RequestPriority,
    ) -> AsyncGenerator[StreamingChunk, None]:
        """
        Streaming generation via LLMTranslator.arun_stream.

        Timeout applies to the total duration of the stream.
        """
        self._validate_generation_request(prompt, max_tokens, request_id)

        op_ctx = from_mcp(mcp_context)
        messages: List[Dict[str, str]] = [{"role": "user", "content": prompt}]
        start_time = time.time()
        chunks_delivered = 0

        async def _stream() -> AsyncGenerator[StreamingChunk, None]:
            nonlocal chunks_delivered
            await self._request_semaphore.acquire(priority, request_id)
            self._active_requests += 1
            try:
                agen = await self._llm.arun_stream(
                    raw_messages=messages,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system_message=system_message,
                    op_ctx=op_ctx,
                    framework_ctx=None,
                )

                async for chunk in agen:
                    streaming_chunk = self._normalize_stream_chunk(
                        chunk_obj=chunk,
                        request_id=request_id,
                        index=chunks_delivered,
                    )
                    yield streaming_chunk
                    chunks_delivered += 1

                    # Timeout check on total streaming duration
                    if time.time() - start_time > timeout:
                        raise asyncio.TimeoutError()

                # Rough token estimate for stats
                processing_time = time.time() - start_time
                estimated_tokens = self._estimate_tokens(prompt, model) + chunks_delivered * 10
                self._record_token_usage(estimated_tokens)
                self._record_success(processing_time)

            except asyncio.TimeoutError:
                self._record_failure()
                raise LLMServiceError(
                    f"Streaming request timed out after {timeout}s",
                    "STREAM_TIMEOUT",
                    request_id,
                )
            except Exception as exc:  # noqa: BLE001
                self._attach_error_context(
                    exc,
                    operation="generate_stream",
                    request_id=request_id,
                    model=model,
                )
                self._record_failure()
                raise
            finally:
                self._request_semaphore.release()
                self._active_requests = max(0, self._active_requests - 1)

        return _stream()

    def _normalize_stream_chunk(
        self,
        *,
        chunk_obj: Any,
        request_id: str,
        index: int,
    ) -> StreamingChunk:
        """
        Normalize whatever the translator emits during streaming into StreamingChunk.
        """
        text = ""
        is_final = False
        finish_reason: Optional[str] = None

        if isinstance(chunk_obj, dict):
            text = str(chunk_obj.get("text") or chunk_obj.get("delta", ""))
            is_final = bool(chunk_obj.get("is_final", False))
            finish_reason = chunk_obj.get("finish_reason")
        else:
            text = str(getattr(chunk_obj, "text", "") or getattr(chunk_obj, "delta", ""))
            is_final = bool(getattr(chunk_obj, "is_final", False))
            finish_reason = getattr(chunk_obj, "finish_reason", None)

        chunk_id = f"{request_id}_{index}"
        return StreamingChunk(
            text=text,
            chunk_id=chunk_id,
            is_final=is_final,
            finish_reason=finish_reason,
        )

    # -------------------------------------------------------------------------
    # Validation + safety
    # -------------------------------------------------------------------------

    def _validate_generation_request(
        self,
        prompt: str,
        max_tokens: Optional[int],
        request_id: str,
    ) -> None:
        """
        Basic validation of prompt and token parameters, plus
        very conservative harmful-content screening.
        """
        if not prompt or not prompt.strip():
            raise LLMServiceError("Prompt cannot be empty", "EMPTY_PROMPT", request_id)

        if len(prompt) > 100_000:
            raise LLMServiceError(
                f"Prompt length {len(prompt)} exceeds maximum 100,000 characters",
                "PROMPT_TOO_LONG",
                request_id,
            )

        if max_tokens is not None and max_tokens > self._config.max_tokens_per_request:
            raise LLMServiceError(
                f"max_tokens {max_tokens} exceeds limit {self._config.max_tokens_per_request}",
                "MAX_TOKENS_EXCEEDED",
                request_id,
            )

        harmful_patterns = self._detect_harmful_patterns(prompt)
        if harmful_patterns:
            raise LLMServiceError(
                f"Request contains potentially harmful content: {harmful_patterns}",
                "CONTENT_VIOLATION",
                request_id,
            )

    def _detect_harmful_patterns(self, text: str) -> List[str]:
        """
        Extremely lightweight harmful-content detection.

        This is intentionally conservative and configurable; more sophisticated
        policies should be implemented at higher layers if needed.
        """
        if not self._config.enable_harmful_content_detection:
            return []

        harmful_patterns: List[str] = []
        text_lower = text.lower()

        # Default patterns plus any configured ones.
        default_patterns = [
            "how to hack",
            "make a bomb",
            "hurt someone",
            "illegal drugs",
            "self harm",
            "suicide methods",
        ]
        configured_patterns = self._config.harmful_patterns or []
        patterns_to_check = list(dict.fromkeys(default_patterns + configured_patterns))

        for pattern in patterns_to_check:
            if pattern.lower() in text_lower:
                harmful_patterns.append(pattern)

        return harmful_patterns

    # -------------------------------------------------------------------------
    # Rate limiting + token accounting
    # -------------------------------------------------------------------------

    def _estimate_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Advanced token estimation with model-specific encoding and fallbacks.
        """
        return self._token_estimator.estimate_tokens(text, model)

    def _check_rate_limit(self) -> bool:
        """
        Sliding-window request rate limiting over the last 60 seconds.

        Assumes single-threaded asyncio event loop; no explicit locking needed.
        """
        now = time.time()
        window_start = now - 60.0

        # Prune old entries
        while self._request_rate_tracker and self._request_rate_tracker[0] <= window_start:
            self._request_rate_tracker.popleft()

        if len(self._request_rate_tracker) >= self._config.requests_per_minute:
            return False

        self._request_rate_tracker.append(now)
        return True

    def _check_token_limit(self, estimated_tokens: int) -> bool:
        """
        Sliding-window token rate limiting over the last 60 seconds.

        Assumes single-threaded asyncio event loop; no explicit locking needed.
        """
        now = time.time()
        window_start = now - 60.0

        # Prune old entries
        while self._token_usage_tracker and self._token_usage_tracker[0][0] <= window_start:
            self._token_usage_tracker.popleft()

        current_usage = sum(tokens for _, tokens in self._token_usage_tracker)
        if current_usage + estimated_tokens > self._config.tokens_per_minute:
            return False

        return True

    def _record_token_usage(self, tokens: int) -> None:
        """
        Record token usage for rate limiting and aggregate metrics.

        This is the single source of truth for per-request token accounting
        to avoid double-counting tokens across metrics.
        """
        self._token_usage_tracker.append((time.time(), tokens))
        self._total_tokens_used += tokens

    # -------------------------------------------------------------------------
    # Caching
    # -------------------------------------------------------------------------

    def _generate_cache_key(
        self,
        *,
        prompt: str,
        model: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
        system_message: Optional[str],
    ) -> str:
        """
        Generate a cache key from request parameters.

        We intentionally include system_message as it can change semantics
        even with identical prompts.
        """
        import hashlib

        content = json.dumps(
            {
                "prompt": prompt,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "system_message": system_message,
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    async def _get_cached_result(self, cache_key: str) -> Optional[LLMResult]:
        """
        Return cached result if it is still within TTL, else None.

        Both the cache backend's TTL semantics and the service-level TTL
        are respected; whichever expires first will invalidate the entry.
        """
        try:
            entry = await self._cache_backend.get(cache_key)
            if entry is not None:
                result, timestamp = entry
                if time.time() - timestamp <= self._config.cache_ttl:
                    return result
                await self._cache_backend.delete(cache_key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cache get operation failed: %s", exc)
        return None

    async def _cache_result(self, cache_key: str, result: LLMResult) -> None:
        """
        Cache a successful result.
        """
        try:
            result.cached = True
            await self._cache_backend.set(
                cache_key,
                (result, time.time()),
                self._config.cache_ttl,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cache set operation failed: %s", exc)

    # -------------------------------------------------------------------------
    # Health + statistics
    # -------------------------------------------------------------------------

    def _record_success(self, processing_time: float) -> None:
        """
        Record a successful request.

        Token usage is tracked exclusively via _record_token_usage to avoid
        double-counting across metrics.
        """
        self._consecutive_failures = 0
        self._is_healthy = True
        self._total_requests += 1
        self._successful_requests += 1
        self._total_processing_time += processing_time

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        self._failed_requests += 1
        self._total_requests += 1
        # If failures accumulate, mark service as degraded; callers can
        # query get_health_status() for diagnostics.
        if self._consecutive_failures >= self._config.circuit_breaker_threshold:
            self._is_healthy = False
            logger.warning(
                "MCPLLMTranslationService marked as degraded after %s failures",
                self._consecutive_failures,
            )

    def get_metrics(self) -> Dict[str, Any]:
        """
        Return aggregated in-memory metrics for inspection/logging.

        This method is synchronous and never awaits, so it can be safely
        called from any context.
        """
        avg_processing_time = (
            self._total_processing_time / self._successful_requests
            if self._successful_requests > 0
            else 0.0
        )
        success_rate = (
            self._successful_requests / self._total_requests
            if self._total_requests > 0
            else 1.0
        )
        avg_tokens_per_request = (
            self._total_tokens_used / self._successful_requests
            if self._successful_requests > 0
            else 0
        )
        rate_limit_utilization = (
            len(self._request_rate_tracker) / self._config.requests_per_minute
            if self._config.requests_per_minute > 0
            else 0.0
        )
        token_limit_utilization = (
            sum(tokens for _, tokens in self._token_usage_tracker)
            / self._config.tokens_per_minute
            if self._config.tokens_per_minute > 0
            else 0.0
        )
        cache_hit_ratio = (
            self._cache_hits
            / (self._cache_hits + self._cache_misses)
            if (self._cache_hits + self._cache_misses) > 0
            else 0.0
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
            "cache_size": self._get_cache_size(),
            "cache_hit_ratio": cache_hit_ratio,
            "consecutive_failures": self._consecutive_failures,
            "rate_limit_utilization": rate_limit_utilization,
            "token_limit_utilization": token_limit_utilization,
        }

    def _get_cache_size(self) -> int:
        """
        Get approximate cache size without awaiting.

        For in-memory cache backends, we introspect the internal _cache dict.
        For other backends, this returns 0 unless they choose to expose a
        compatible attribute.
        """
        try:
            cache = getattr(self._cache_backend, "_cache", None)
            if isinstance(cache, dict):
                return len(cache)
        except Exception:  # noqa: BLE001
            # Best-effort only; never let cache size introspection break callers.
            pass
        return 0

    def get_health_status(self) -> str:
        """
        Return a coarse-grained health status string.
        """
        if not self._is_healthy:
            return "degraded"
        if self._consecutive_failures > 0:
            return "warning"
        return "healthy"

    async def _run_health_probe(self) -> bool:
        """
        Run a synthetic LLM call for health probing.

        This intentionally:
        - Does NOT consume request/token rate limit budget.
        - Does NOT affect normal success/failure metrics.
        """
        test_prompt = "Respond with 'OK' for health check."
        op_ctx = from_mcp({"id": "health_check", "method": "health_check"})
        messages: List[Dict[str, str]] = [{"role": "user", "content": test_prompt}]

        try:
            completion_obj = await asyncio.wait_for(
                self._llm.arun_complete(
                    raw_messages=messages,
                    model=None,
                    max_tokens=4,
                    temperature=0.0,
                    system_message="You are a health-check probe.",
                    op_ctx=op_ctx,
                    framework_ctx=None,
                ),
                timeout=self._config.health_check_timeout,
            )

            # Basic sanity check: ensure some text is returned, but do not
            # enforce specific content to avoid coupling to model behavior.
            if isinstance(completion_obj, dict):
                _ = str(completion_obj.get("text") or "")
            else:
                _ = str(getattr(completion_obj, "text", "") or "")
            return True
        except Exception as exc:  # noqa: BLE001
            self._attach_error_context(
                exc,
                operation="health_check_probe",
                request_id="health_check",
                model=None,
            )
            logger.warning("Health probe LLM call failed: %s", exc)
            return False

    async def health_check(self) -> Dict[str, Any]:
        """
        Simple health probe with optional synthetic LLM call.

        Health probes are designed not to interfere with normal rate limits
        or success/failure metrics.
        """
        status = self.get_health_status()
        health: Dict[str, Any] = {
            "status": status,
            "consecutive_failures": self._consecutive_failures,
            "active_requests": self._active_requests,
            "cache_size": self._get_cache_size(),
        }

        # Only attempt a real LLM call if we appear healthy
        if status == "healthy":
            probe_ok = await self._run_health_probe()
            if probe_ok:
                health["service_test"] = "passed"
            else:
                health["service_test"] = "failed"
                health["status"] = "degraded"

        return health

    # -------------------------------------------------------------------------
    # Error context
    # -------------------------------------------------------------------------

    def _attach_error_context(
        self,
        exc: Exception,
        *,
        operation: str,
        request_id: Optional[str],
        model: Optional[str] = None,
        **additional_context: Any,
    ) -> None:
        """
        Attach rich error context without ever masking the original exception.
        """
        try:
            attach_context(
                exc,
                framework="mcp",
                resource_type="llm",
                operation=operation,
                request_id=request_id,
                translation_layer="llm",
                service_health=self.get_health_status(),
                active_requests=self._active_requests,
                consecutive_failures=self._consecutive_failures,
                model=model,
                **additional_context,
            )
        except Exception:  # noqa: BLE001
            # Never allow error-context attachment to change behavior
            pass

    # -------------------------------------------------------------------------
    # Shutdown
    # -------------------------------------------------------------------------

    async def shutdown(self) -> None:
        """
        Graceful shutdown – allow active requests to complete up to 30s,
        then clear internal state.
        """
        logger.info("Shutting down MCPLLMTranslationService")
        shutdown_start = time.time()
        while self._active_requests > 0 and time.time() - shutdown_start < 30.0:
            await asyncio.sleep(0.1)

        if self._active_requests > 0:
            logger.warning(
                "Force shutdown with %s active requests still in-flight",
                self._active_requests,
            )

        await self._cache_backend.clear()
        self._request_rate_tracker.clear()
        self._token_usage_tracker.clear()
        self._is_healthy = False
        self._active_requests = 0
        logger.info("MCPLLMTranslationService shutdown complete")


# =============================================================================
# Factory
# =============================================================================


def create_llm_service(
    adapter: LLMProtocolV1,
    *,
    framework: str = "mcp",
    translator: Optional[LLMFrameworkTranslator] = None,
    post_processing_config: Optional[LLMPostProcessingConfig] = None,
    cache_backend: Optional[CacheBackend] = None,
    token_estimator: Optional[TokenEstimator] = None,
    **kwargs: Any,
) -> MCPLLMTranslationService:
    """
    Convenience factory to build an MCPLLMTranslationService from an LLM adapter.

    Args:
        adapter:
            An implementation of LLMProtocolV1.
        framework:
            Framework identifier passed to create_llm_translator; defaults to "mcp".
        translator:
            Optional framework-specific LLMFrameworkTranslator implementation.
        post_processing_config:
            Optional LLMPostProcessingConfig for post-processing behavior.
        cache_backend:
            Optional distributed cache backend implementation.
        token_estimator:
            Optional custom TokenEstimator implementation.
        **kwargs:
            Additional MCPLLMTranslationService configuration parameters.

    Returns:
        Configured MCPLLMTranslationService instance.
    """
    llm_translator = create_llm_translator(
        adapter=adapter,
        framework=framework,
        translator=translator,
        post_processing_config=post_processing_config,
    )
    return MCPLLMTranslationService(
        llm_translator=llm_translator,
        cache_backend=cache_backend,
        token_estimator=token_estimator,
        **kwargs,
    )
