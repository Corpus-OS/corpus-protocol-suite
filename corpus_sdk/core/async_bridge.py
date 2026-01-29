# corpus_sdk/core/async_bridge.py
# SPDX-License-Identifier: Apache-2.0

"""
Async bridge utilities for Corpus adapters.

This module bridges async-first Corpus SDK APIs to sync-centric frameworks
(e.g., LangChain, LlamaIndex, custom apps) in a safe and predictable way.

Key concerns handled here:

- Running async code from synchronous call sites
- Nested event loops (Jupyter, asyncio-based apps)
- Threaded execution when a loop is already running
- Timeouts and cancellation
- Thread safety for concurrent access
- Context variable (contextvars) propagation across threads
- Resource management and backpressure
- Circuit breaker for fault tolerance
- Comprehensive metrics and observability

This is *protocol infrastructure*, not business logic. Adapters should use
these helpers to expose both async and sync entry points on top of async
Corpus APIs.

Design Philosophy
-----------------
- Minimal public API surface: `run_async` and `sync_wrapper`.
- No framework-specific behavior.
- Avoid shared global event loops to prevent cross-thread issues.
- Use a shared ThreadPoolExecutor only when we must run async code from a
  thread that already has a running event loop (nested loop avoidance).
- Preserve Python contextvars (e.g., tracing, logging context) when hopping
  to worker threads.
- Fail fast under resource exhaustion.
- Provide observability through metrics.
"""

from __future__ import annotations

import asyncio
import contextvars
from concurrent.futures import Future, ThreadPoolExecutor
import functools
import inspect
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Optional, TypeVar, Set
from time import perf_counter
import os
import sys

# ---------------------------------------------------------------------------
# Python compatibility: ParamSpec
# ---------------------------------------------------------------------------
# ParamSpec is in typing starting Python 3.10.
# For Python 3.9 (and some 3.8 builds), it must come from typing_extensions.
try:
    from typing import ParamSpec  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    from typing_extensions import ParamSpec  # type: ignore[no-redef]

T = TypeVar("T")
P = ParamSpec("P")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Default number of worker threads used when we need to run async code
#: from within an already-running event loop (e.g., Jupyter).
DEFAULT_MAX_WORKERS: int = int(os.environ.get("CORPUS_ASYNC_MAX_WORKERS", "4"))

#: Optional global timeout (in seconds) applied by default in run_async
#: when no explicit timeout is provided. None means "no global timeout".
DEFAULT_RUN_TIMEOUT: Optional[float] = None

#: Maximum number of pending tasks before rejecting new requests
DEFAULT_MAX_PENDING_TASKS: int = int(os.environ.get("CORPUS_ASYNC_MAX_PENDING", "1000"))

#: Circuit breaker failure threshold
DEFAULT_CIRCUIT_BREAKER_THRESHOLD: int = 5

#: Circuit breaker recovery timeout (seconds)
DEFAULT_CIRCUIT_BREAKER_TIMEOUT: float = 60.0

# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class AsyncBridgeTimeoutError(TimeoutError):
    """Raised when an async operation exceeds its timeout in AsyncBridge."""


class AsyncBridgeResourceError(RuntimeError):
    """Raised when the async bridge is overloaded or out of resources."""


class AsyncBridgeCircuitOpenError(RuntimeError):
    """Raised when the circuit breaker is open and blocking requests."""


# ---------------------------------------------------------------------------
# Metrics and Monitoring
# ---------------------------------------------------------------------------


@dataclass
class BridgeMetrics:
    """Metrics for monitoring async bridge performance and health."""
    calls_total: int = 0
    calls_with_threads: int = 0
    timeouts: int = 0
    errors: int = 0
    resource_errors: int = 0
    circuit_breaker_trips: int = 0
    avg_duration: float = 0.0
    max_duration: float = 0.0
    min_duration: float = float('inf')
    last_call_time: float = 0.0


class CircuitBreaker:
    """Simple circuit breaker pattern for fault tolerance.

    Notes
    -----
    - Uses time.monotonic() for failure timing to avoid wall-clock skew issues.
    - last_failure_time is stored as a monotonic timestamp.
    """

    def __init__(
        self,
        failure_threshold: int = DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
        recovery_timeout: float = DEFAULT_CIRCUIT_BREAKER_TIMEOUT,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        # Monotonic timestamp of last failure (or None if fully healthy)
        self.last_failure_time: Optional[float] = None
        self._lock = threading.Lock()

    def should_try(self) -> bool:
        """Check if requests should be allowed through."""
        now = time.monotonic()
        with self._lock:
            if self.failures < self.failure_threshold:
                return True
            if self.last_failure_time and (now - self.last_failure_time) > self.recovery_timeout:
                # Reset after recovery timeout
                self.failures = 0
                self.last_failure_time = None
                return True
            return False

    def record_failure(self) -> None:
        """Record a failure and potentially trip the circuit breaker."""
        with self._lock:
            self.failures += 1
            self.last_failure_time = time.monotonic()

    def record_success(self) -> None:
        """Record a success and reset failure count."""
        with self._lock:
            self.failures = 0
            self.last_failure_time = None

    @property
    def is_open(self) -> bool:
        """Check if circuit breaker is currently open."""
        now = time.monotonic()
        with self._lock:
            return (
                self.failures >= self.failure_threshold
                and self.last_failure_time is not None
                and (now - self.last_failure_time) <= self.recovery_timeout
            )


# ---------------------------------------------------------------------------
# Core bridge
# ---------------------------------------------------------------------------


class AsyncBridge:
    """
    Helper for safely running async code from sync contexts.

    Core Functionality
    ------------------
    - run_async: Execute a coroutine and return its result.
    - sync_wrapper: Decorator to convert async callables into sync ones.

    Event Loop Strategy
    -------------------
    - If there is *already* a running event loop in the current thread:
        * Spawn a background thread and use `asyncio.run(...)` there.
        * This avoids illegal nested loop usage in environments like Jupyter.
        * The current `contextvars.Context` is captured and propagated to
          the worker thread so tracing/logging context is preserved.
    - If there is *no* running loop in the current thread:
        * Run the coroutine directly via `asyncio.run`.

    Timeouts
    --------
    - Optional per-call timeout (seconds) can be provided to `run_async`.
    - If omitted, `DEFAULT_RUN_TIMEOUT` is used.
    - If both are None, no timeout is enforced.
    - Timeouts are applied via `asyncio.wait_for`, and surfaced as
      `AsyncBridgeTimeoutError`.

    Resource Management
    -------------------
    - Maximum pending tasks limit to prevent resource exhaustion.
    - Circuit breaker pattern to fail fast during downstream issues.
    - Comprehensive metrics for monitoring and alerting.

    Thread Safety
    -------------
    - Executor creation and configuration is protected by a class-level lock.
    - `run_async` and `sync_wrapper` are safe to use from multiple threads.

    Lifecycle
    ---------
    - The shared ThreadPoolExecutor is created lazily.
    - It lives for the process lifetime unless `shutdown()` is called.
    - It is always safe not to call `shutdown`; the OS will reclaim resources
      on process exit.

    Example
    -------
        # Direct usage
        result = AsyncBridge.run_async(some_coro(), timeout=5.0)

        # Decorator usage
        class MyAdapter:
            async def afetch(self, url: str) -> str:
                ...

            fetch = AsyncBridge.sync_wrapper(afetch)
    """

    # Shared state - all access protected by _lock.
    _lock = threading.RLock()
    _executor: Optional[ThreadPoolExecutor] = None
    _max_workers: int = DEFAULT_MAX_WORKERS
    _max_pending_tasks: int = DEFAULT_MAX_PENDING_TASKS
    _pending_futures: Set[Future] = set()

    # Monitoring and fault tolerance
    _metrics = BridgeMetrics()
    _metrics_lock = threading.Lock()
    _circuit_breaker = CircuitBreaker()

    # ------------------------------------------------------------------ #
    # Configuration
    # ------------------------------------------------------------------ #

    @classmethod
    def configure(
        cls,
        max_workers: Optional[int] = None,
        max_pending_tasks: Optional[int] = None,
        circuit_breaker_threshold: Optional[int] = None,
        circuit_breaker_timeout: Optional[float] = None,
    ) -> None:
        """
        Configure the async bridge runtime parameters.

        This only affects future executor creation. If an executor already
        exists, it will continue using its current thread count.

        Args:
            max_workers: Maximum worker threads for executor
            max_pending_tasks: Maximum pending tasks before rejection
            circuit_breaker_threshold: Failure threshold for circuit breaker
            circuit_breaker_timeout: Recovery timeout for circuit breaker

        Raises:
            ValueError: If any parameter is invalid.
        """
        with cls._lock:
            if max_workers is not None:
                if max_workers <= 0:
                    raise ValueError("max_workers must be a positive integer")
                cls._max_workers = max_workers

            if max_pending_tasks is not None:
                if max_pending_tasks <= 0:
                    raise ValueError("max_pending_tasks must be a positive integer")
                cls._max_pending_tasks = max_pending_tasks

            if circuit_breaker_threshold is not None or circuit_breaker_timeout is not None:
                if circuit_breaker_threshold is not None and circuit_breaker_threshold <= 0:
                    raise ValueError("circuit_breaker_threshold must be positive")
                if circuit_breaker_timeout is not None and circuit_breaker_timeout <= 0:
                    raise ValueError("circuit_breaker_timeout must be positive")

                # Create new circuit breaker with updated settings
                cls._circuit_breaker = CircuitBreaker(
                    failure_threshold=circuit_breaker_threshold or DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
                    recovery_timeout=circuit_breaker_timeout or DEFAULT_CIRCUIT_BREAKER_TIMEOUT,
                )

            if cls._executor is not None:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "AsyncBridge.configure: executor already created; "
                        "new settings will apply only to future executors."
                    )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @classmethod
    def _get_or_create_executor(cls) -> ThreadPoolExecutor:
        """
        Lazily create and return the shared ThreadPoolExecutor.

        Caller must hold cls._lock.
        """
        if cls._executor is None:
            cls._executor = ThreadPoolExecutor(
                max_workers=cls._max_workers,
                thread_name_prefix="corpus_async_",
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "AsyncBridge: created ThreadPoolExecutor(max_workers=%d)",
                    cls._max_workers,
                )
        return cls._executor

    @staticmethod
    async def _with_timeout(
        coro: Coroutine[Any, Any, T],
        timeout: Optional[float],
    ) -> T:
        """
        Await a coroutine with an optional timeout.

        Raises:
            AsyncBridgeTimeoutError: if the coroutine exceeds the timeout.
        """
        if timeout is None:
            return await coro
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise AsyncBridgeTimeoutError(
                f"Async operation exceeded timeout={timeout!r} seconds"
            ) from exc

    @staticmethod
    def _run_in_context(
        coro: Coroutine[Any, Any, T],
        timeout: Optional[float],
        context: contextvars.Context,
    ) -> T:
        """
        Run coroutine in a specific contextvars context.

        This wrapper ensures exceptions from asyncio.run are properly handled
        and don't crash the worker thread.
        """
        try:
            # Actually run in the captured context to preserve contextvars
            return context.run(asyncio.run, AsyncBridge._with_timeout(coro, timeout))
        except Exception:
            # Let all exceptions propagate to the caller thread
            raise

    @classmethod
    def _check_resource_limits(cls) -> None:
        """Check if we're at resource limits and should reject the request."""
        with cls._lock:
            if len(cls._pending_futures) >= cls._max_pending_tasks:
                raise AsyncBridgeResourceError(
                    f"AsyncBridge queue full ({len(cls._pending_futures)}/{cls._max_pending_tasks} tasks pending). "
                    "Consider increasing max_pending_tasks or reducing concurrent calls."
                )

    @classmethod
    def _attach_error_context(
        cls,
        exc: Exception,
        timeout: Optional[float],
        threaded_execution: bool,
    ) -> None:
        """Safely attach context to exceptions for better debugging."""
        try:
            from corpus_sdk.core.error_context import attach_context

            attach_context(
                exc,
                origin="async_bridge",
                timeout=timeout,
                threaded_execution=threaded_execution,
                pending_tasks=len(cls._pending_futures),
                circuit_breaker_open=cls._circuit_breaker.is_open,
            )
        except Exception as attach_error:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Failed to attach error context: %s", attach_error)

    @classmethod
    def _update_metrics(
        cls,
        duration: float,
        threaded: bool,
        timed_out: bool = False,
        errored: bool = False,
        resource_error: bool = False,
        circuit_trip: bool = False,
    ) -> None:
        """Update performance and health metrics."""
        with cls._metrics_lock:
            cls._metrics.calls_total += 1
            if threaded:
                cls._metrics.calls_with_threads += 1
            if timed_out:
                cls._metrics.timeouts += 1
            if errored:
                cls._metrics.errors += 1
            if resource_error:
                cls._metrics.resource_errors += 1
            if circuit_trip:
                cls._metrics.circuit_breaker_trips += 1

            # Update duration statistics using Welford's method for numerical stability
            new_avg = cls._metrics.avg_duration + (duration - cls._metrics.avg_duration) / cls._metrics.calls_total
            cls._metrics.avg_duration = new_avg
            cls._metrics.max_duration = max(cls._metrics.max_duration, duration)
            cls._metrics.min_duration = min(cls._metrics.min_duration, duration)
            cls._metrics.last_call_time = time.time()

    @staticmethod
    def _dispose_unawaited_coroutine_best_effort(coro: Any) -> None:
        """
        Best-effort disposal for coroutine/awaitable objects that will not be awaited.

        Why this exists:
        - In certain fail-fast branches (e.g., circuit breaker open, resource rejection),
          we may have already been handed a coroutine object by the caller.
        - If we raise before executing/awaiting it, Python will emit
          "coroutine was never awaited" warnings during garbage collection.
        - Closing coroutine objects is the correct, low-cost, non-blocking way
          to prevent those warnings while preserving fail-fast behavior.

        Safety notes:
        - We only attempt to close/cancel; we never block on completion.
        - If the object is not a coroutine/awaitable with a close/cancel API,
          this function becomes a no-op.
        """
        try:
            # Coroutine objects created by calling an `async def` implement `.close()`.
            if inspect.iscoroutine(coro):
                coro.close()
                return

            # Futures/Tasks implement `.cancel()`. This is safe and non-blocking.
            cancel = getattr(coro, "cancel", None)
            if callable(cancel):
                cancel()
                return

            # As a last resort, if a third-party awaitable provides `.close()`, use it.
            close = getattr(coro, "close", None)
            if callable(close):
                close()
        except Exception:
            # Never allow cleanup to mask the real error path.
            return

    @classmethod
    def _is_circuit_breaker_failure(cls, exc: BaseException) -> bool:
        """
        Determine whether an exception should contribute to circuit breaker failures.

        IMPORTANT DISTINCTION:
        - This module is protocol infrastructure, not business logic.
        - Exceptions raised *by the coroutine's business logic* (e.g., adapter/runtime errors)
          are expected to propagate to callers and should not automatically poison the
          bridge circuit breaker.
        - The circuit breaker is intended to protect against *infrastructure-level*
          failures such as timeouts and execution environment instability.

        Current policy (conservative by design):
        - Count AsyncBridgeTimeoutError as a failure (represents infrastructure timeout).
        - Count asyncio.CancelledError as a failure (represents cancellation at the bridge boundary).
        - Count specific known event-loop/execution RuntimeError patterns as failures.

        This policy prevents expected downstream exceptions from opening the circuit,
        while still allowing the breaker to engage for genuine infrastructure trouble.
        """
        if isinstance(exc, AsyncBridgeTimeoutError):
            return True

        # Cancellation at the bridge boundary is treated as an infrastructure signal.
        if isinstance(exc, asyncio.CancelledError):
            return True

        # Some RuntimeErrors indicate execution environment problems rather than user code.
        if isinstance(exc, RuntimeError):
            msg = str(exc)
            infra_markers = (
                "asyncio.run() cannot be called from a running event loop",
                "Event loop is closed",
                "cannot schedule new futures after shutdown",
            )
            if any(marker in msg for marker in infra_markers):
                return True

        return False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @classmethod
    def run_async(
        cls,
        coro: Coroutine[Any, Any, T],
        timeout: Optional[float] = None,
    ) -> T:
        """
        Execute a coroutine from synchronous code and return its result.

        Behavior by environment:
        - If an event loop is already running in this thread:
            * Run the coroutine in a background thread using `asyncio.run`,
              propagating the current contextvars.Context.
        - If no event loop is running:
            * Run the coroutine directly via `asyncio.run`.

        Args:
            coro:
                The coroutine to execute. It must not have been awaited before.
            timeout:
                Optional timeout in seconds. If None, uses DEFAULT_RUN_TIMEOUT.
                If both are None, no timeout is enforced.

        Returns:
            The result of the coroutine.

        Raises:
            AsyncBridgeTimeoutError: On timeout.
            AsyncBridgeResourceError: When resource limits are exceeded.
            AsyncBridgeCircuitOpenError: When circuit breaker is open.
            Any exception raised by the coroutine itself.

        Thread Safety:
            Safe to call from multiple threads concurrently.
        """
        start_time = perf_counter()
        effective_timeout = timeout if timeout is not None else DEFAULT_RUN_TIMEOUT
        threaded_execution = False
        circuit_tripped = False

        # Check circuit breaker first
        if not cls._circuit_breaker.should_try():
            circuit_tripped = True

            # IMPORTANT:
            # Fail-fast here means we will not execute/await the coroutine.
            # Close it to avoid "coroutine was never awaited" warnings.
            cls._dispose_unawaited_coroutine_best_effort(coro)

            exc = AsyncBridgeCircuitOpenError(
                "AsyncBridge circuit breaker is open due to repeated failures"
            )
            cls._attach_error_context(exc, effective_timeout, False)
            cls._update_metrics(
                duration=perf_counter() - start_time,
                threaded=False,
                errored=True,
                circuit_trip=True,
            )
            raise exc

        try:
            # Detect a running event loop in this thread.
            try:
                asyncio.get_running_loop()
                loop_running = True
            except RuntimeError:
                loop_running = False

            # Case 1: Running loop - use a background thread to avoid nested loops.
            if loop_running:
                threaded_execution = True
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "AsyncBridge.run_async: running loop detected; using executor with contextvars"
                    )

                # Check resource limits before proceeding.
                # If we reject here, we must dispose the coroutine because it will not be awaited.
                try:
                    cls._check_resource_limits()
                except AsyncBridgeResourceError:
                    cls._dispose_unawaited_coroutine_best_effort(coro)
                    raise

                # Capture the current contextvars.Context so tracing/logging/context
                # is preserved when we hop to the worker thread.
                ctx = contextvars.copy_context()

                # CRITICAL FIX #1: All operations under lock to prevent race condition
                # between future creation, tracking, and callback registration.
                with cls._lock:
                    executor = cls._get_or_create_executor()
                    try:
                        future = executor.submit(cls._run_in_context, coro, effective_timeout, ctx)
                    except RuntimeError as e:
                        # Executor may have been shut down or is unavailable.
                        # Dispose coroutine because it will not be awaited after this failure.
                        cls._dispose_unawaited_coroutine_best_effort(coro)

                        resource_exc = AsyncBridgeResourceError(
                            "AsyncBridge executor is unavailable or shut down"
                        )
                        cls._attach_error_context(resource_exc, effective_timeout, True)
                        raise resource_exc from e

                    cls._pending_futures.add(future)

                    def cleanup_future(f: Future) -> None:
                        with cls._lock:
                            cls._pending_futures.discard(f)

                    future.add_done_callback(cleanup_future)

                # Let exceptions from the coroutine surface naturally.
                try:
                    result = future.result()
                    cls._circuit_breaker.record_success()
                    return result
                except Exception as exc:
                    # Only count infrastructure-level failures toward the circuit breaker.
                    if cls._is_circuit_breaker_failure(exc):
                        cls._circuit_breaker.record_failure()
                    cls._attach_error_context(exc, effective_timeout, True)
                    raise

            # Case 2: No running loop - use asyncio.run directly.
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("AsyncBridge.run_async: no running loop; using asyncio.run")
            try:
                result = asyncio.run(cls._with_timeout(coro, effective_timeout))
                cls._circuit_breaker.record_success()
                return result
            except AsyncBridgeTimeoutError:
                # Re-raise timeout errors as-is; timeouts are infrastructure failures.
                cls._circuit_breaker.record_failure()
                raise
            except Exception as exc:
                # Only count infrastructure-level failures toward the circuit breaker.
                if cls._is_circuit_breaker_failure(exc):
                    cls._circuit_breaker.record_failure()
                cls._attach_error_context(exc, effective_timeout, False)
                raise

        except AsyncBridgeTimeoutError:
            # Re-raise timeout errors with metrics
            duration = perf_counter() - start_time
            cls._update_metrics(
                duration=duration,
                threaded=threaded_execution,
                timed_out=True,
                errored=True,
            )
            raise
        except AsyncBridgeResourceError:
            # Resource errors are not circuit breaker failures
            duration = perf_counter() - start_time
            cls._update_metrics(
                duration=duration,
                threaded=threaded_execution,
                resource_error=True,
                errored=True,
            )
            raise
        except Exception:
            # Update metrics for other errors
            duration = perf_counter() - start_time
            cls._update_metrics(
                duration=duration,
                threaded=threaded_execution,
                errored=True,
            )
            raise
        else:
            # Success case - no exception occurred
            duration = perf_counter() - start_time
            cls._update_metrics(
                duration=duration,
                threaded=threaded_execution,
            )

    @classmethod
    def sync_wrapper(
        cls,
        async_func: Callable[P, Coroutine[Any, Any, T]],
        *,
        timeout: Optional[float] = None,
    ) -> Callable[P, T]:
        """
        Decorator: wrap an async function to create a sync version.

        Args:
            async_func:
                The async function to wrap.
            timeout:
                Optional default timeout (seconds) for all calls to the
                wrapped sync function. If None, uses DEFAULT_RUN_TIMEOUT.

        Returns:
            A synchronous function with the same signature as `async_func`.

        Example
        -------
            class MyAdapter:
                async def acomplete(self, prompt: str) -> str:
                    ...

                complete = AsyncBridge.sync_wrapper(acomplete)
        """

        @functools.wraps(async_func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            coro = async_func(*args, **kwargs)
            return cls.run_async(coro, timeout=timeout)

        return wrapper

    # ------------------------------------------------------------------ #
    # Monitoring and Inspection
    # ------------------------------------------------------------------ #

    @classmethod
    def get_metrics(cls) -> BridgeMetrics:
        """Get current bridge metrics for monitoring."""
        with cls._metrics_lock:
            # Return a copy to prevent external modification
            return BridgeMetrics(**cls._metrics.__dict__)

    @classmethod
    def get_status(cls) -> dict[str, Any]:
        """Get comprehensive status information."""
        with cls._lock:
            executor_info = {
                "max_workers": cls._max_workers,
                "max_pending_tasks": cls._max_pending_tasks,
                "pending_tasks": len(cls._pending_futures),
                "executor_created": cls._executor is not None,
            }

        metrics = cls.get_metrics()
        circuit_breaker = {
            "is_open": cls._circuit_breaker.is_open,
            "failures": cls._circuit_breaker.failures,
            "failure_threshold": cls._circuit_breaker.failure_threshold,
        }

        return {
            "executor": executor_info,
            "circuit_breaker": circuit_breaker,
            "metrics": metrics.__dict__,
        }

    # ------------------------------------------------------------------ #
    # Optional cleanup
    # ------------------------------------------------------------------ #

    @classmethod
    def shutdown(cls, *, wait: bool = True, cancel_futures: bool = False) -> None:
        """
        Best-effort cleanup of the shared executor.

        This is entirely optional and need not be called in normal usage.
        It exists for host environments that prefer explicit teardown
        (e.g. tests, long-lived worker processes).

        Args:
            wait:
                If True, block until all queued futures are completed.
                If False, do not wait.
            cancel_futures:
                If True, cancel all pending futures (best effort).

        It is safe to call multiple times.
        """
        with cls._lock:
            # CRITICAL FIX #2: Make shutdown idempotent and exception-safe.
            # Clear executor reference first, before shutdown() which might raise.
            executor = cls._executor
            cls._executor = None

            if executor is None:
                # Already shut down or never created
                return

            try:
                if cancel_futures:
                    # Cancel pending futures
                    for future in cls._pending_futures.copy():
                        future.cancel()
                    cls._pending_futures.clear()

                executor.shutdown(wait=wait)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "AsyncBridge: executor shutdown (wait=%s, cancel_futures=%s)",
                        wait,
                        cancel_futures,
                    )
            except Exception as e:
                logger.error(
                    "AsyncBridge: error during executor shutdown: %s",
                    e,
                    exc_info=True,
                )
                # Don't re-raise - we've already cleared the executor reference
                # Subsequent shutdown calls will be no-ops


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def run_async(
    coro: Coroutine[Any, Any, T],
    timeout: Optional[float] = None,
) -> T:
    """
    Execute a coroutine from synchronous code.

    Convenience wrapper around AsyncBridge.run_async.
    """
    return AsyncBridge.run_async(coro, timeout=timeout)


def sync_wrapper(
    async_func: Callable[P, Coroutine[Any, Any, T]],
    *,
    timeout: Optional[float] = None,
) -> Callable[P, T]:
    """
    Decorator to convert async functions to sync.

    Convenience wrapper around AsyncBridge.sync_wrapper.
    """
    return AsyncBridge.sync_wrapper(async_func, timeout=timeout)


__all__ = [
    "AsyncBridge",
    "AsyncBridgeTimeoutError",
    "AsyncBridgeResourceError",
    "AsyncBridgeCircuitOpenError",
    "BridgeMetrics",
    "DEFAULT_MAX_WORKERS",
    "DEFAULT_RUN_TIMEOUT",
    "DEFAULT_MAX_PENDING_TASKS",
    "run_async",
    "sync_wrapper",
]
