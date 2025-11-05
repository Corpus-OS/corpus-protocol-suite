# SPDX-License-Identifier: Apache-2.0
"""
Generic async retry helpers with exponential backoff.

This module is protocol-agnostic and uses class-name checks aligned with the
normalized error taxonomy used across the SDK (LLM/Embedding/Vector/Graph).
Randomization (jitter) can be toggled off for deterministic backoff.

Usage:
    from corpus_sdk.examples.common.retry import RetryPolicy, retry_async

    policy = RetryPolicy(max_attempts=5, base_ms=200, max_ms=5_000)
    result = await retry_async(
        lambda: adapter.complete(messages=[...], ctx=ctx),
        policy=policy
    )
"""
from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

__all__ = ["RetryPolicy", "retry_async", "is_retryable_normalized"]

# Retryable (by class name) per normalized taxonomy
_DEFAULT_RETRYABLE_NAMES = {
    "TransientNetwork",
    "Unavailable",
    "ResourceExhausted",
    "ModelOverloaded",
    "IndexNotReady",
}

# Non-retryable guardrails
_DEFAULT_NON_RETRYABLE_NAMES = {
    "BadRequest",
    "AuthError",
    "NotSupported",
    "DimensionMismatch",
    "ContentFiltered",
    "TextTooLong",
}

@dataclass(frozen=True)
class RetryPolicy:
    """
    Retry configuration with exponential backoff.

    Attributes:
        max_attempts: Total tries including the first attempt.
        base_ms:      Initial backoff in milliseconds.
        max_ms:       Maximum backoff cap in milliseconds.
        multiplier:   Exponential growth factor.
        use_jitter:   If True, randomize each sleep in [0, backoff]; if False, sleep=backoff.
    """
    max_attempts: int = 4
    base_ms: int = 150
    max_ms: int = 10_000
    multiplier: float = 2.0
    use_jitter: bool = True

    def backoff_ms(self, attempt_index: int) -> int:
        """
        Compute backoff for attempt_index (0-based, i.e., first retry -> 0).
        """
        raw = int(self.base_ms * (self.multiplier ** attempt_index))
        return min(raw, self.max_ms)

def is_retryable_normalized(exc: BaseException) -> bool:
    """
    Heuristic: treat exceptions by *class name* according to the normalized taxonomy.
    Unknown classes default to non-retryable unless they expose a retry hint.
    """
    name = type(exc).__name__
    if name in _DEFAULT_NON_RETRYABLE_NAMES:
        return False
    if name in _DEFAULT_RETRYABLE_NAMES:
        return True
    retry_after = getattr(exc, "retry_after_ms", None)
    return retry_after is not None

async def retry_async(
    fn: Callable[[], Awaitable],
    *,
    policy: RetryPolicy = RetryPolicy(),
    is_retryable: Callable[[BaseException], bool] = is_retryable_normalized,
    on_backoff: Optional[Callable[[int, float, BaseException], None]] = None,
) -> any:
    """
    Execute an async operation with retries on retryable errors.

    Args:
        fn:           Zero-arg coroutine factory to invoke each attempt.
        policy:       RetryPolicy controlling backoff.
        is_retryable: Predicate deciding whether to retry a given exception.
        on_backoff:   Optional callback (attempt_no, sleep_seconds, exc) before sleeping.

    Returns:
        The result of `fn()` if it eventually succeeds.

    Raises:
        The last exception if all attempts fail or a non-retryable error occurs.
    """
    attempts = int(policy.max_attempts)
    if attempts <= 0:
        raise ValueError("max_attempts must be >= 1")

    for attempt in range(1, attempts + 1):
        try:
            return await fn()
        except BaseException as exc:
            if attempt >= attempts or not is_retryable(exc):
                raise
            backoff = policy.backoff_ms(attempt_index=attempt - 1) / 1000.0
            sleep_for = (random.random() * backoff) if policy.use_jitter else backoff
            if on_backoff:
                try:
                    on_backoff(attempt, sleep_for, exc)
                except Exception:
                    pass
            await asyncio.sleep(sleep_for)

