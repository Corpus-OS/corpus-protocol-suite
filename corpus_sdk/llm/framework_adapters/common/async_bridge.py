"""
Bridges async-first Corpus SDK APIs to sync-centric frameworks
(LangChain, LlamaIndex, etc.) safely and predictably.

Key concerns handled here:

- Nested event loops (e.g., Jupyter, asyncio-based apps)
- Threaded execution when a loop is already running
- Timeouts and cancellation
- Clean shutdown / resource reuse

This module is *protocol infrastructure*, not business logic. It provides
a small set of primitives that adapters can use to expose both async and
sync entry points on top of async Corpus APIs.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools
import threading
from typing import Any, Callable, Coroutine, Optional, TypeVar, Generic

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


class AsyncBridge(Generic[T]):
    """
    Helper for safely running async code from sync contexts.

    Responsibilities
    ----------------
    - Manage an event loop for "normal" synchronous code paths.
    - Detect when an event loop is already running and fall back to
      running the coroutine in a dedicated thread.
    - Provide decorators for wrapping async functions into sync APIs.
    - Manage a shared ThreadPoolExecutor for nested-loop use cases.
    """

    # Shared state
    _loop: Optional[asyncio.AbstractEventLoop] = None
    _loop_owned: bool = False  # whether this class created the loop
    _executor: Optional[ThreadPoolExecutor] = None
    _max_workers: int = DEFAULT_MAX_WORKERS
    _lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Loop / executor management
    # ------------------------------------------------------------------ #

    @classmethod
    def configure_executor(cls, max_workers: int) -> None:
        """
        Configure the maximum number of worker threads for the shared executor.

        If the executor already exists, this does not resize it; it only
        affects future creation. Call `shutdown()` first if you want to
        recreate it with a new size.
        """
        if max_workers <= 0:
            raise ValueError("max_workers must be a positive integer")
        with cls._lock:
            cls._max_workers = max_workers

    @classmethod
    def get_loop(cls) -> asyncio.AbstractEventLoop:
        """
        Get or create the primary event loop for sync contexts.

        Behavior:
        - If a cached loop exists and is not closed, return it.
        - Else, try `asyncio.get_event_loop()`.
        - If that fails or returns a closed loop, create a new loop via
          `asyncio.new_event_loop()` and install it with `asyncio.set_event_loop()`.

        Note:
            This helper is for *non-nested* sync contexts. In nested-loop
            situations (loop already running), `run_async` will detect
            that and avoid using `run_until_complete` on the same loop.
        """
        with cls._lock:
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
                # No current loop or unusable; create our own.
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                cls._loop = loop
                cls._loop_owned = True
                return loop

    @classmethod
    def _ensure_executor(cls) -> ThreadPoolExecutor:
        """
        Lazily create and return the shared ThreadPoolExecutor.

        Used when we need to run async code while an event loop is
        already running in the current thread (nested-loop case).
        """
        with cls._lock:
            if cls._executor is None:
                cls._executor = ThreadPoolExecutor(max_workers=cls._max_workers)
            return cls._executor

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    async def _maybe_with_timeout(
        coro: Coroutine[Any, Any, T],
        timeout: Optional[float],
    ) -> T:
        """
        Await a coroutine, optionally wrapping with asyncio.wait_for.

        Raises AsyncBridgeTimeoutError on timeout.
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
    # Public API: run_async
    # ------------------------------------------------------------------ #

    @classmethod
    def run_async(
        cls,
        coro: Coroutine[Any, Any, T],
        timeout: Optional[float] = None,
    ) -> T:
        """
        Execute a coroutine from synchronous code and return its result.

        Behavior:
        - If an event loop is NOT running in this thread:
            - Use the managed loop and `run_until_complete`.
        - If an event loop IS running:
            - Use a background thread with `asyncio.run` to avoid
              nested event loops.

        Args:
            coro:
                The coroutine to execute.
            timeout:
                Optional per-call timeout in seconds. If None, falls
                back to DEFAULT_RUN_TIMEOUT; if that is also None, no
                timeout is applied.

        Raises:
            AsyncBridgeTimeoutError on timeout.
            Any exception raised by the coroutine itself.
        """
        effective_timeout = timeout if timeout is not None else DEFAULT_RUN_TIMEOUT

        loop = None
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # No current loop in this thread; we'll use get_loop() below.
            pass

        # ------------------------------------------------------------------
        # Case 1: Loop is already running (nested-loop environment).
        # ------------------------------------------------------------------
        if loop is not None and loop.is_running():
            executor = cls._ensure_executor()

            def _runner() -> T:
                async def _inner() -> T:
                    return await cls._maybe_with_timeout(coro, effective_timeout)

                # New event loop in this background thread.
                return asyncio.run(_inner())

            future = executor.submit(_runner)
            # Let any exception propagate naturally.
            return future.result()

        # ------------------------------------------------------------------
        # Case 2: No running loop in this thread; use our managed loop.
        # ------------------------------------------------------------------
        loop = cls.get_loop()

        async def _inner() -> T:
            return await cls._maybe_with_timeout(coro, effective_timeout)

        try:
            return loop.run_until_complete(_inner())
        except KeyboardInterrupt:
            # Best-effort cancellation: cancel all pending tasks on this loop.
            # This avoids leaving dangling tasks when a user interrupts.
            pending = asyncio.all_tasks(loop=loop)
            for task in pending:
                task.cancel()
            raise

    # ------------------------------------------------------------------ #
    # Decorators: sync_wrapper / sync_method
    # ------------------------------------------------------------------ #

    @classmethod
    def sync_wrapper(
        cls,
        async_func: Callable[..., Coroutine[Any, Any, T]],
    ) -> Callable[..., T]:
        """
        Decorator: wrap an async function to expose a sync function
        that uses AsyncBridge.run_async under the hood.

        Example
        -------
            class Adapter:
                async def acomplete(self, prompt: str) -> str:
                    ...

                complete = AsyncBridge.sync_wrapper(acomplete)
        """

        @functools.wraps(async_func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            coro = async_func(*args, **kwargs)
            return cls.run_async(coro)

        return wrapper

    @classmethod
    def sync_method(
        cls,
        async_func: Callable[..., Coroutine[Any, Any, T]],
    ) -> Callable[..., T]:
        """
        Decorator variant for instance methods.

        This is essentially the same as sync_wrapper, but exists as a
        semantic hint that the wrapped function is likely a method
        (i.e., first argument is `self`). Usage is identical.

        Example
        -------
            class Adapter:
                @AsyncBridge.sync_method
                async def acomplete(self, prompt: str) -> str:
                    ...

                # complete is a sync twin of acomplete
                def complete(self, prompt: str) -> str:
                    return self.acomplete_sync(prompt)

                # or directly:
                complete = AsyncBridge.sync_method(acomplete)
        """

        @functools.wraps(async_func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            coro = async_func(*args, **kwargs)
            return cls.run_async(coro)

        return wrapper

    # ------------------------------------------------------------------ #
    # Optional utilities
    # ------------------------------------------------------------------ #

    @classmethod
    def spawn(cls, coro: Coroutine[Any, Any, T]) -> asyncio.Future:
        """
        Schedule a coroutine on the managed loop without waiting for it.

        Returns:
            asyncio.Future associated with the scheduled task.

        Notes:
            - This is primarily for background side-effects (metrics,
              logging, etc.).
            - Exceptions will be stored on the Future and need to be
              observed somewhere to avoid "unobserved exception" logs.
        """
        loop = cls.get_loop()
        # If loop is running in this thread, schedule directly.
        if loop.is_running():
            return loop.create_task(coro)

        # If loop is not running, start the task by running once around it.
        # Caller is responsible for ensuring this makes sense in their
        # environment; often this is used only when the server is already
        # managing the loop.
        def _start_task() -> asyncio.Future:
            return loop.create_task(coro)

        return _start_task()

    @classmethod
    def run_in_thread(cls, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Run a synchronous callable in the shared ThreadPoolExecutor and
        wait for its result.

        This is a convenience helper for CPU or I/O-bound sync work that
        you want to offload from the main thread, using the same executor
        infrastructure as nested async calls.
        """
        executor = cls._ensure_executor()
        future = executor.submit(func, *args, **kwargs)
        return future.result()

    @classmethod
    def shutdown(cls) -> None:
        """
        Gracefully shut down the executor and (if owned) the managed loop.

        Intended primarily for tests or short-lived scripts. In long-lived
        processes (servers, notebooks) you usually don't want to call this
        unless you're explicitly tearing down all async infrastructure.
        """
        with cls._lock:
            if cls._executor is not None:
                cls._executor.shutdown(wait=True)
                cls._executor = None

            if cls._loop is not None and cls._loop_owned and not cls._loop.is_closed():
                cls._loop.call_soon_threadsafe(cls._loop.stop)
                cls._loop.close()
            cls._loop = None
            cls._loop_owned = False


# ---------------------------------------------------------------------------
# Module-level convenience aliases
# ---------------------------------------------------------------------------

def run_async(
    coro: Coroutine[Any, Any, T],
    timeout: Optional[float] = None,
) -> T:
    """
    Module-level convenience wrapper around AsyncBridge.run_async.
    """
    return AsyncBridge.run_async(coro, timeout=timeout)


def sync_wrapper(
    async_func: Callable[..., Coroutine[Any, Any, T]],
) -> Callable[..., T]:
    """
    Module-level convenience wrapper around AsyncBridge.sync_wrapper.
    """
    return AsyncBridge.sync_wrapper(async_func)


def sync_method(
    async_func: Callable[..., Coroutine[Any, Any, T]],
) -> Callable[..., T]:
    """
    Module-level convenience wrapper around AsyncBridge.sync_method.
    """
    return AsyncBridge.sync_method(async_func)
