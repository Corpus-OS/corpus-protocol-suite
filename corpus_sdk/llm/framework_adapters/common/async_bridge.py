# corpus_sdk/llm/framework_adapters/common/async_bridge.py
# SPDX-License-Identifier: Apache-2.0

"""
Async bridge utilities for Corpus LLM adapters.

This module bridges async-first Corpus SDK APIs to sync-centric frameworks
(e.g., LangChain, LlamaIndex) in a safe and predictable way.

Key concerns handled here:

- Nested event loops (Jupyter, asyncio-based apps)
- Threaded execution when a loop is already running
- Timeouts and cancellation
- Thread safety for concurrent access

This is *protocol infrastructure*, not business logic. Adapters should use
these helpers to expose both async and sync entry points on top of async
Corpus APIs.

Design Philosophy
-----------------
Minimal API surface. If a feature isn't essential for async/sync bridging,
it doesn't belong here.

Resources (executor, event loop) live for the process lifetime. The OS
handles cleanup on exit.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools
import threading
from typing import Any, Callable, Coroutine, Optional, TypeVar

T = TypeVar("T")

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
    - run_async: Execute a coroutine and return its result
    - sync_wrapper: Decorator to convert async functions to sync

    Thread Safety
    -------------
    All operations are thread-safe. A single RLock protects shared state.

    Lifecycle
    ---------
    Resources are created lazily and live for the process lifetime.
    No explicit cleanup is provided or needed.

    Example:
        # Direct usage
        result = AsyncBridge.run_async(some_coro(), timeout=5.0)
        
        # Decorator usage
        class MyAdapter:
            async def afetch(self, url: str) -> str:
                ...
            
            fetch = AsyncBridge.sync_wrapper(afetch)
    """

    # Shared state - all access protected by _lock
    _lock = threading.RLock()
    _loop: Optional[asyncio.AbstractEventLoop] = None
    _loop_owned: bool = False
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

        Args:
            max_workers: Must be a positive integer

        Raises:
            ValueError: If max_workers <= 0
        """
        if max_workers <= 0:
            raise ValueError("max_workers must be a positive integer")
        with cls._lock:
            cls._max_workers = max_workers

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @classmethod
    def _get_or_create_loop(cls) -> asyncio.AbstractEventLoop:
        """
        Get or create the primary event loop for sync contexts.
        
        Caller must hold cls._lock.
        """
        if cls._loop is not None and not cls._loop.is_closed():
            return cls._loop

        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Current event loop is closed")
            cls._loop = loop
            cls._loop_owned = False
            return loop
        except (RuntimeError, AssertionError):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            cls._loop = loop
            cls._loop_owned = True
            return loop

    @classmethod
    def _get_or_create_executor(cls) -> ThreadPoolExecutor:
        """
        Lazily create and return the shared ThreadPoolExecutor.
        
        Caller must hold cls._lock.
        """
        if cls._executor is None:
            cls._executor = ThreadPoolExecutor(
                max_workers=cls._max_workers,
                thread_name_prefix="corpus_async_"
            )
        return cls._executor

    @staticmethod
    async def _with_timeout(
        coro: Coroutine[Any, Any, T],
        timeout: Optional[float],
    ) -> T:
        """Await a coroutine with optional timeout."""
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

        If an event loop is already running in this thread, executes the
        coroutine in a background thread to avoid nested loop issues.

        Args:
            coro: The coroutine to execute
            timeout: Optional timeout in seconds

        Returns:
            The result of the coroutine

        Raises:
            AsyncBridgeTimeoutError: On timeout
            Any exception raised by the coroutine

        Thread Safety:
            Safe to call from multiple threads concurrently.

        Example:
            >>> async def fetch_data():
            ...     return "data"
            >>> result = AsyncBridge.run_async(fetch_data(), timeout=5.0)
        """
        effective_timeout = timeout if timeout is not None else DEFAULT_RUN_TIMEOUT

        # Check if we're in a running loop
        try:
            loop = asyncio.get_event_loop()
            loop_running = loop.is_running()
        except RuntimeError:
            loop_running = False

        # Case 1: Running loop - use background thread
        if loop_running:
            def _runner() -> T:
                return asyncio.run(cls._with_timeout(coro, effective_timeout))

            with cls._lock:
                executor = cls._get_or_create_executor()
            
            return executor.submit(_runner).result()

        # Case 2: No running loop - use managed loop
        with cls._lock:
            loop = cls._get_or_create_loop()

        try:
            return loop.run_until_complete(cls._with_timeout(coro, effective_timeout))
        except KeyboardInterrupt:
            # Cancel pending tasks on interrupt
            for task in asyncio.all_tasks(loop=loop):
                task.cancel()
            raise

    @classmethod
    def sync_wrapper(
        cls,
        async_func: Callable[..., Coroutine[Any, Any, T]],
        *,
        timeout: Optional[float] = None,
    ) -> Callable[..., T]:
        """
        Decorator: wrap an async function to create a sync version.

        Args:
            async_func: The async function to wrap
            timeout: Optional default timeout for all calls

        Returns:
            A synchronous function with the same signature

        Example:
            >>> class Adapter:
            ...     async def acomplete(self, prompt: str) -> str:
            ...         return f"response: {prompt}"
            ...     
            ...     # Create sync version
            ...     complete = AsyncBridge.sync_wrapper(acomplete)
            ...     
            ...     # Or as a decorator
            ...     @AsyncBridge.sync_wrapper
            ...     async def astream(self, prompt: str):
            ...         ...
        """
        @functools.wraps(async_func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            coro = async_func(*args, **kwargs)
            return cls.run_async(coro, timeout=timeout)
        return wrapper


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

    Example:
        >>> from corpus_sdk.llm.framework_adapters.common.async_bridge import run_async
        >>> result = run_async(some_coro(), timeout=5.0)
    """
    return AsyncBridge.run_async(coro, timeout=timeout)


def sync_wrapper(
    async_func: Callable[..., Coroutine[Any, Any, T]],
    *,
    timeout: Optional[float] = None,
) -> Callable[..., T]:
    """
    Decorator to convert async functions to sync.

    Convenience wrapper around AsyncBridge.sync_wrapper.

    Example:
        >>> from corpus_sdk.llm.framework_adapters.common.async_bridge import sync_wrapper
        >>> 
        >>> @sync_wrapper
        ... async def fetch(url: str) -> str:
        ...     ...
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
