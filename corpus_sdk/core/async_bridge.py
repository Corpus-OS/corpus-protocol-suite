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
"""

from __future__ import annotations

import asyncio
import contextvars
from concurrent.futures import ThreadPoolExecutor
import functools
import logging
import threading
from typing import Any, Callable, Coroutine, Optional, TypeVar, ParamSpec

T = TypeVar("T")
P = ParamSpec("P")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Default number of worker threads used when we need to run async code
#: from within an already-running event loop (e.g., Jupyter).
DEFAULT_MAX_WORKERS: int = 4

#: Optional global timeout (in seconds) applied by default in run_async
#: when no explicit timeout is provided. None means "no global timeout".
DEFAULT_RUN_TIMEOUT: Optional[float] = None


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class AsyncBridgeTimeoutError(TimeoutError):
    """Raised when an async operation exceeds its timeout in AsyncBridge."""


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

    # ------------------------------------------------------------------ #
    # Configuration
    # ------------------------------------------------------------------ #

    @classmethod
    def configure(cls, max_workers: int) -> None:
        """
        Configure the maximum number of worker threads for the shared executor.

        This only affects future executor creation. If an executor already
        exists, it will continue using its current thread count.

        This method is safe to call from any thread at any time.

        Args:
            max_workers: Must be a positive integer.

        Raises:
            ValueError: If max_workers <= 0.
        """
        if max_workers <= 0:
            raise ValueError("max_workers must be a positive integer")
        with cls._lock:
            if cls._executor is not None:
                logger.debug(
                    "AsyncBridge.configure: executor already created; "
                    "new max_workers will apply only to future executors."
                )
            cls._max_workers = max_workers

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
            Any exception raised by the coroutine itself.

        Thread Safety:
            Safe to call from multiple threads concurrently.
        """
        effective_timeout = timeout if timeout is not None else DEFAULT_RUN_TIMEOUT

        # Detect a running event loop in this thread.
        try:
            asyncio.get_running_loop()
            loop_running = True
        except RuntimeError:
            loop_running = False

        # Case 1: Running loop - use a background thread to avoid nested loops.
        if loop_running:
            logger.debug(
                "AsyncBridge.run_async: running loop detected; using executor with contextvars"
            )

            # Capture the current contextvars.Context so tracing/logging/context
            # is preserved when we hop to the worker thread.
            ctx = contextvars.copy_context()

            def _runner() -> T:
                # Each call gets its own fresh loop via asyncio.run, executed
                # inside the captured contextvars.Context.
                return asyncio.run(cls._with_timeout(coro, effective_timeout))

            with cls._lock:
                executor = cls._get_or_create_executor()

            future = executor.submit(ctx.run, _runner)
            # Let exceptions from the coroutine surface naturally.
            return future.result()

        # Case 2: No running loop - use asyncio.run directly.
        logger.debug("AsyncBridge.run_async: no running loop; using asyncio.run")
        try:
            return asyncio.run(cls._with_timeout(coro, effective_timeout))
        except KeyboardInterrupt:
            # Let KeyboardInterrupt propagate; asyncio.run handles loop teardown.
            logger.debug("AsyncBridge.run_async: KeyboardInterrupt during asyncio.run")
            raise

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
    # Optional cleanup
    # ------------------------------------------------------------------ #

    @classmethod
    def shutdown(cls, *, wait: bool = False) -> None:
        """
        Best-effort cleanup of the shared executor.

        This is entirely optional and need not be called in normal usage.
        It exists for host environments that prefer explicit teardown
        (e.g. tests, long-lived worker processes).

        Args:
            wait:
                If True, block until all queued futures are completed.
                If False (default), do not wait.

        It is safe to call multiple times.
        """
        with cls._lock:
            if cls._executor is not None:
                cls._executor.shutdown(wait=wait)
                cls._executor = None
                logger.debug(
                    "AsyncBridge: executor shutdown (wait=%s)",
                    wait,
                )


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
    "DEFAULT_MAX_WORKERS",
    "DEFAULT_RUN_TIMEOUT",
    "run_async",
    "sync_wrapper",
]
