# SPDX-License-Identifier: Apache-2.0
"""
Generic async retry helpers with exponential backoff.

This module is protocol-agnostic and uses class-name checks aligned with the
normalized error taxonomy used across the SDK (LLM/Embedding/Vector/Graph).

Randomization (jitter) can be toggled off for deterministic testing.

Usage:
    from corpus_sdk.examples.common.retry import RetryPolicy, retry_async

    policy = RetryPolicy(max_attempts=5, base_ms=200, max_ms=5_000)
    result = await retry_async(
        lambda: adapter.complete(messages=[...], ctx=ctx),
        policy=policy,
    )
"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

__all__ = [
    "RetryPolicy", 
    "RetryStats", 
    "retry_async", 
    "is_retryable_normalized",
    "CircuitBreakerOpenError"
]

# ------------------------------------------------------------------------------
# Classification sets (open-source safe)
# ------------------------------------------------------------------------------

# Retryable by class name, per normalized taxonomy
_DEFAULT_RETRYABLE_NAMES = {
    "TransientNetwork",
    "Unavailable",
    "ResourceExhausted",
    "ModelOverloaded",
    "IndexNotReady",
}

# Non-retryable by class name
_DEFAULT_NON_RETRYABLE_NAMES = {
    "BadRequest",
    "AuthError",
    "NotSupported",
    "DimensionMismatch",
    "ContentFiltered",
    "TextTooLong",
}


# ------------------------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------------------------

class CircuitBreakerOpenError(Exception):
    """Raised when the circuit breaker is open and requests are blocked."""
    def __init__(self, reset_after: float):
        super().__init__(f"Circuit breaker is open. Resets in {reset_after:.1f}s")
        self.reset_after = reset_after


# ------------------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------------------

@dataclass(frozen=True)
class RetryStats:
    """
    Statistics about the retry operation.
    
    Attributes:
        attempts: Number of attempts made (including successful one)
        total_delay: Total time spent waiting between retries (seconds)
        last_exception: The last exception encountered before success or final failure
    """
    attempts: int
    total_delay: float
    last_exception: Optional[BaseException] = None


@dataclass(frozen=True)
class RetryPolicy:
    """
    Retry configuration with exponential backoff.

    Attributes:
        max_attempts: Total tries including the first attempt.
        base_ms:      Initial backoff in milliseconds.
        max_ms:       Maximum backoff cap in milliseconds.
        multiplier:   Exponential growth factor per attempt.
        use_jitter:   Randomize sleep in [0, backoff] for distributed retry spacing.
        circuit_breaker_threshold: Consecutive failures to open circuit breaker (0 to disable)
        circuit_breaker_reset_ms: How long circuit breaker stays open (milliseconds)
    """

    max_attempts: int = 4
    base_ms: int = 150
    max_ms: int = 10_000
    multiplier: float = 2.0
    use_jitter: bool = True
    circuit_breaker_threshold: int = 0  # 0 = disabled
    circuit_breaker_reset_ms: int = 30_000

    def __post_init__(self):
        """Validate configuration on initialization."""
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_ms <= 0 or self.max_ms <= 0:
            raise ValueError("Backoff times must be positive")
        if self.multiplier < 1.0:
            raise ValueError("Multiplier must be >= 1.0")
        if self.base_ms > self.max_ms:
            raise ValueError("base_ms cannot exceed max_ms")
        if self.circuit_breaker_threshold < 0:
            raise ValueError("circuit_breaker_threshold must be >= 0")
        if self.circuit_breaker_threshold > 0 and self.circuit_breaker_reset_ms <= 0:
            raise ValueError("circuit_breaker_reset_ms must be positive when circuit breaker is enabled")

    def backoff_ms(self, attempt_index: int) -> int:
        """Compute exponential backoff for a given retry index."""
        raw = int(self.base_ms * (self.multiplier ** attempt_index))
        return min(raw, self.max_ms)


# ------------------------------------------------------------------------------
# Circuit breaker state
# ------------------------------------------------------------------------------

class CircuitBreakerState:
    """Thread-safe circuit breaker state management."""
    __slots__ = ('failure_count', 'last_failure_time', 'open_until')
    
    def __init__(self):
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.open_until = 0.0

    def record_success(self):
        """Reset failure count on successful operation."""
        self.failure_count = 0
        self.open_until = 0.0

    def record_failure(self, policy: RetryPolicy) -> Optional[float]:
        """
        Record a failure and check if circuit breaker should open.
        
        Returns:
            None if circuit is closed, reset timestamp if open
        """
        now = time.monotonic()
        self.failure_count += 1
        self.last_failure_time = now
        
        # Check if we should open the circuit breaker
        if (policy.circuit_breaker_threshold > 0 and 
            self.failure_count >= policy.circuit_breaker_threshold):
            self.open_until = now + (policy.circuit_breaker_reset_ms / 1000.0)
            return self.open_until
        
        return None

    def check_state(self) -> Optional[float]:
        """Check if circuit breaker is open. Returns reset time if open."""
        if self.open_until > 0:
            now = time.monotonic()
            if now < self.open_until:
                return self.open_until
            else:
                # Circuit breaker reset period has passed
                self.open_until = 0.0
                self.failure_count = 0
        return None


# ------------------------------------------------------------------------------
# Retry logic helpers
# ------------------------------------------------------------------------------

def is_retryable_normalized(exc: BaseException) -> bool:
    """
    Decide if an exception should be retried.

    Uses the normalized error taxonomy — matching by class name.
    Defaults to non-retryable unless explicitly in retryable set or
    defines `retry_after_ms` or `retry_after`.
    """
    # Special case: never retry circuit breaker errors
    if isinstance(exc, CircuitBreakerOpenError):
        return False
        
    name = type(exc).__name__
    if name in _DEFAULT_NON_RETRYABLE_NAMES:
        return False
    if name in _DEFAULT_RETRYABLE_NAMES:
        return True
    
    # Enhanced retry-after support
    retry_after_ms = getattr(exc, "retry_after_ms", None)
    if retry_after_ms is not None:
        return True
        
    # Support for retry-after seconds as well
    retry_after_sec = getattr(exc, "retry_after", None)
    return retry_after_sec is not None


async def retry_async(
    fn: Callable[[], Awaitable[Any]],
    *,
    policy: RetryPolicy = RetryPolicy(),
    is_retryable: Callable[[BaseException], bool] = is_retryable_normalized,
    on_backoff: Optional[Callable[[int, float, BaseException], None]] = None,
    return_stats: bool = False,
    circuit_breaker_state: Optional[_CircuitBreakerState] = None,
) -> Any | tuple[Any, RetryStats]:
    """
    Execute an async operation with retries on retryable errors.

    Args:
        fn:           Zero-arg coroutine factory to invoke each attempt.
        policy:       RetryPolicy controlling backoff.
        is_retryable: Predicate to decide if a given exception is retryable.
        on_backoff:   Optional callback (attempt_no, sleep_seconds, exc) before sleeping.
        return_stats: If True, returns (result, stats) instead of just result.
        circuit_breaker_state: Shared state for circuit breaker across calls.

    Returns:
        The result of `fn()` if it eventually succeeds.
        If return_stats=True, returns (result, RetryStats).

    Raises:
        The last exception if all attempts fail or a non-retryable error occurs.
        CircuitBreakerOpenError if circuit breaker is active.
    """
    # Initialize circuit breaker state if needed
    if circuit_breaker_state is None and policy.circuit_breaker_threshold > 0:
        circuit_breaker_state = _CircuitBreakerState()
    
    # Check circuit breaker state
    if circuit_breaker_state:
        reset_time = circuit_breaker_state.check_state()
        if reset_time is not None:
            raise CircuitBreakerOpenError(reset_time - time.monotonic())

    attempts = max(1, policy.max_attempts)
    total_delay = 0.0
    last_exception = None

    for attempt in range(1, attempts + 1):
        try:
            result = await fn()
            
            # Record success for circuit breaker
            if circuit_breaker_state:
                circuit_breaker_state.record_success()
                
            if return_stats:
                return result, RetryStats(
                    attempts=attempt,
                    total_delay=total_delay,
                    last_exception=last_exception
                )
            return result
            
        except BaseException as exc:
            last_exception = exc
            
            # Check circuit breaker
            if circuit_breaker_state:
                reset_time = circuit_breaker_state.record_failure(policy)
                if reset_time is not None:
                    # Circuit breaker just opened
                    raise CircuitBreakerOpenError(reset_time - time.monotonic())

            retry_ok = is_retryable(exc)
            if not retry_ok or attempt >= attempts:
                raise

            backoff = policy.backoff_ms(attempt_index=attempt - 1) / 1000.0
            sleep_for = random.random() * backoff if policy.use_jitter else backoff
            total_delay += sleep_for

            if on_backoff:
                try:
                    on_backoff(attempt, sleep_for, exc)
                except Exception:
                    # Don't allow metrics or hooks to break retry loop
                    pass

            await asyncio.sleep(sleep_for)


# ------------------------------------------------------------------------------
# Simplified version without advanced features
# ------------------------------------------------------------------------------

async def simple_retry_async(
    fn: Callable[[], Awaitable[Any]],
    *,
    max_attempts: int = 4,
    base_delay: float = 0.15,
    max_delay: float = 10.0,
) -> Any:
    """
    Simplified version for basic use cases.
    
    Args:
        fn: Zero-arg coroutine factory
        max_attempts: Maximum number of attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        
    Returns:
        Result of the successful operation
    """
    policy = RetryPolicy(
        max_attempts=max_attempts,
        base_ms=int(base_delay * 1000),
        max_ms=int(max_delay * 1000)
    )
    return await retry_async(fn, policy=policy)


# ------------------------------------------------------------------------------
# Usage examples
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # Example usage
    async def demo():
        policy = RetryPolicy(
            max_attempts=3,
            base_ms=100,
            max_ms=2000,
            circuit_breaker_threshold=5  # Open after 5 consecutive failures
        )
        
        # Shared circuit breaker state across related operations
        cb_state = _CircuitBreakerState()
        
        try:
            result, stats = await retry_async(
                lambda: some_async_operation(),
                policy=policy,
                return_stats=True,
                circuit_breaker_state=cb_state,
                on_backoff=lambda attempt, sleep, exc: 
                    print(f"Attempt {attempt} failed, retrying in {sleep:.2f}s: {exc}")
            )
            print(f"Success after {stats.attempts} attempts, total delay: {stats.total_delay:.2f}s")
        except CircuitBreakerOpenError as e:
            print(f"Circuit breaker open: {e}")
        except Exception as e:
            print(f"All attempts failed: {e}")

    # asyncio.run(demo())
