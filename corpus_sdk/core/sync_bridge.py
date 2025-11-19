# corpus_sdk/core/sync_stream_bridge.py
# SPDX-License-Identifier: Apache-2.0

"""
SyncStreamBridge

Utility for bridging an async streaming coroutine into a synchronous
iterator, used by Corpus adapters across protocols (LLM, vector, tools)
and frameworks (LangChain, LlamaIndex, etc.).

Design goals
------------
- Framework-agnostic: no LangChain / LlamaIndex imports.
- Protocol-agnostic: does not know about LLMChunks, VectorMatches, etc.
- Safe: no nested event loops; uses a dedicated worker thread.
- Observable: attaches rich error context via error_context.attach_context.
- Resilient: optional transient retry with exponential backoff.
- Bounded: queue-full handling with configurable retry limit.

Typical usage
-------------

    bridge = SyncStreamBridge(
        coro_factory=lambda: some_async_streaming_call(...),
        framework="langchain",
        error_context={"operation": "stream"},
    )

    for item in bridge.run():
        handle(item)

The async side is expected to look like:

    async def some_async_streaming_call(...) -> AsyncIterator[T]:
        async for chunk in ...:
            yield chunk

This module does not depend on any specific Corpus protocol; it is
reusable for LLM, vector, and future streaming APIs.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import queue
import threading
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Iterator,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

from corpus_sdk.core.error_context import attach_context

logger = logging.getLogger(__name__)

T = TypeVar("T")


class SyncStreamBridgeError(RuntimeError):
    """Raised when the bridge encounters an internal failure (e.g., queue full)."""


class SyncStreamBridge:
    """
    Bridge an async streaming coroutine into a synchronous iterator.

    High-level behavior
    -------------------
    - Spawns a dedicated worker thread.
    - In that thread, runs an asyncio event loop with `asyncio.run`.
    - Calls `coro_factory()` to obtain an awaitable that yields an
      `AsyncIterator[T]`, then iterates and pushes items into a queue.
    - The main thread calls `run()` and synchronously consumes items
      from the queue as they arrive.
    - On error in the worker, the exception is stored and a sentinel is
      enqueued; the main thread raises the stored exception after the
      iteration completes.

    Transient retry
    ---------------
    If `max_transient_retries > 0` and `transient_error_types` is
    provided, errors of those types are treated as transient:

    - The worker sleeps with exponential backoff in the async loop.
    - Re-runs `coro_factory()` and retries streaming.
    - Once streaming has successfully yielded items, subsequent errors
      are treated as terminal and are surfaced immediately.

    Queue full handling
    -------------------
    Items are pushed into a bounded or unbounded queue. If the queue
    remains full for too long, the bridge:

    - Logs a warning.
    - Stores a `SyncStreamBridgeError` as the bridge error.
    - Enqueues a sentinel (best-effort) and terminates the worker.

    Parameters
    ----------
    coro_factory:
        Callable that returns an awaitable which resolves to an
        `AsyncIterator[T]` when awaited.

    queue_maxsize:
        Max queue size between worker and consumer. `<= 0` means unbounded.

    poll_timeout_s:
        Timeout (seconds) used for both:
        - queue.put() in the worker (when queue is full)
        - queue.get() in the main thread (when waiting for items)

    join_timeout_s:
        Timeout (seconds) for joining the worker thread during shutdown.
        Prevents hangs in pathological cases.

    cancel_event:
        Optional external cancellation event. If provided and set, both
        worker and consumer will exit as soon as practical.

    framework:
        Framework identifier used only for logging / error context
        (e.g., "langchain", "llamaindex", "semantic_kernel").

    error_context:
        Optional dict of extra fields to attach to any propagated error
        via `attach_context`.

    max_transient_retries:
        Number of retry attempts for errors matching transient_error_types.

    transient_backoff_s:
        Initial backoff delay (seconds) for transient retries. Delay grows
        exponentially: attempt N sleeps for `backoff * (2 ** (N - 1))`.

    transient_error_types:
        Tuple of exception types considered transient (e.g.,
        (TransientNetwork, Unavailable)).

    max_queue_full_retries:
        Maximum number of times the worker will retry queue.put() when the
        queue is full. Each attempt uses `poll_timeout_s` as the wait
        timeout. None means "unbounded retries" (not recommended).

    thread_name:
        Optional name for the worker thread. If None, defaults to
        `f"corpus_sync_stream_{framework}"`.
    """

    def __init__(
        self,
        *,
        coro_factory: Callable[[], Awaitable[AsyncIterator[T]]],
        queue_maxsize: int = 0,
        poll_timeout_s: float = 0.1,
        join_timeout_s: float = 5.0,
        cancel_event: Optional[threading.Event] = None,
        framework: str = "unknown",
        error_context: Optional[Dict[str, Any]] = None,
        max_transient_retries: int = 0,
        transient_backoff_s: float = 0.25,
        transient_error_types: Tuple[Type[BaseException], ...] = (),
        max_queue_full_retries: Optional[int] = 100,
        thread_name: Optional[str] = None,
    ) -> None:
        self._coro_factory = coro_factory
        self._queue: "queue.Queue[Any]" = queue.Queue(
            maxsize=max(queue_maxsize, 0)
        )
        self._poll_timeout_s = float(poll_timeout_s)
        self._join_timeout_s = float(join_timeout_s)
        self._cancel_event = cancel_event or threading.Event()
        self._framework = framework or "unknown"
        self._error_context: Dict[str, Any] = dict(error_context or {})
        self._max_transient_retries = max(0, int(max_transient_retries))
        self._transient_backoff_s = max(0.0, float(transient_backoff_s))
        self._transient_error_types = transient_error_types
        self._max_queue_full_retries = (
            int(max_queue_full_retries)
            if max_queue_full_retries is not None and max_queue_full_retries > 0
            else None
        )
        self._thread_name = (
            thread_name or f"corpus_sync_stream_{self._framework}"
        )

        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[BaseException] = None
        self._sentinel: object = object()
        # Capture contextvars so tracing/logging context propagates into the worker.
        self._parent_context = contextvars.copy_context()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(self) -> Iterator[T]:
        """
        Run the bridge and yield streaming items synchronously.

        This method:
        - Starts the worker thread on first call.
        - Consumes items from the queue until a sentinel is received or
          cancellation is signaled.
        - After iteration, raises any error that occurred in the worker.

        Returns
        -------
        Iterator[T]
            Synchronous iterator over streaming items.

        Raises
        ------
        BaseException
            Any exception raised by the underlying async stream (or
            internal bridge errors like queue saturation) will be re-raised
            in the calling thread after iteration completes.
        """
        self._ensure_worker_started()

        try:
            while True:
                # Early exit if canceled and nothing is left to consume.
                if self._cancel_event.is_set() and self._queue.empty():
                    break

                try:
                    item = self._queue.get(timeout=self._poll_timeout_s)
                except queue.Empty:
                    # If worker is dead and queue is empty, we are done.
                    with self._lock:
                        thread = self._thread
                    if thread is not None and not thread.is_alive():
                        if self._queue.empty():
                            break
                    continue

                if item is self._sentinel:
                    # Worker signaled completion.
                    break

                # Normal item
                yield item  # type: ignore[misc]
        finally:
            self._shutdown_worker()

        # Propagate worker error, if any.
        if self._error is not None:
            raise self._error

    # ------------------------------------------------------------------ #
    # Worker management
    # ------------------------------------------------------------------ #

    def _ensure_worker_started(self) -> None:
        """Start the worker thread if it is not already running."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return

            def _thread_target() -> None:
                # Run worker logic under the captured contextvars.
                try:
                    self._parent_context.run(self._worker_main)
                except BaseException as exc:  # noqa: BLE001
                    # Last-resort: if something blows up at top-level, capture it.
                    logger.debug(
                        "SyncStreamBridge worker crashed in framework=%s: %s",
                        self._framework,
                        exc,
                    )
                    self._push_error(exc)

            self._thread = threading.Thread(
                target=_thread_target,
                name=self._thread_name,
                daemon=True,
            )
            self._thread.start()

    def _shutdown_worker(self) -> None:
        """Signal cancellation and join the worker thread (best-effort)."""
        with self._lock:
            thread = self._thread

        # Signal cancellation
        self._cancel_event.set()

        if thread is None:
            return

        thread.join(self._join_timeout_s)
        if thread.is_alive():
            logger.debug(
                "SyncStreamBridge worker thread %s did not exit within %.3fs",
                thread.name,
                self._join_timeout_s,
            )

    # ------------------------------------------------------------------ #
    # Worker logic
    # ------------------------------------------------------------------ #

    def _worker_main(self) -> None:
        """Worker entrypoint, runs an async loop with asyncio.run()."""
        try:
            asyncio.run(self._worker_coro())
        except BaseException as exc:  # noqa: BLE001
            # If the async runner itself fails, propagate error.
            self._push_error(exc)
            self._safe_enqueue_sentinel()

    async def _worker_coro(self) -> None:
        """
        Core async worker logic with optional transient retry.

        Calls `coro_factory()` to get an `AsyncIterator[T]` and streams
        items into the queue.
        """
        attempt = 0
        while True:
            if self._cancel_event.is_set():
                return

            try:
                stream = await self._coro_factory()
                async for item in stream:
                    if self._cancel_event.is_set():
                        break
                    self._put_queue_item(item)

                # Normal completion
                self._put_queue_item(self._sentinel)
                return

            except BaseException as exc:  # noqa: BLE001
                is_transient = (
                    bool(self._transient_error_types)
                    and isinstance(exc, self._transient_error_types)
                )
                if (
                    is_transient
                    and attempt < self._max_transient_retries
                    and not self._cancel_event.is_set()
                ):
                    attempt += 1
                    delay = self._transient_backoff_s * (2 ** (attempt - 1))
                    logger.warning(
                        "SyncStreamBridge transient error in framework=%s "
                        "attempt=%d/%d; backing off for %.3fs: %s",
                        self._framework,
                        attempt,
                        self._max_transient_retries,
                        delay,
                        exc,
                    )
                    try:
                        await asyncio.sleep(delay)
                    except Exception:  # noqa: BLE001
                        # Ignore sleep errors; we'll retry immediately.
                        pass
                    continue

                # Non-transient or out-of-retries: surface error.
                self._push_error(exc)
                self._put_queue_item(self._sentinel)
                return

    # ------------------------------------------------------------------ #
    # Queue helpers
    # ------------------------------------------------------------------ #

    def _put_queue_item(self, item: Any) -> None:
        """
        Put an item into the queue with bounded retry when full.

        If the queue remains full after `max_queue_full_retries` attempts,
        records a `SyncStreamBridgeError`, enqueues a sentinel (best-effort),
        and returns without enqueuing the item.
        """
        retries = 0
        while True:
            if self._cancel_event.is_set():
                return

            try:
                self._queue.put(item, timeout=self._poll_timeout_s)
                return
            except queue.Full:
                retries += 1
                logger.debug(
                    "SyncStreamBridge queue full for framework=%s; "
                    "retry=%d (max=%s)",
                    self._framework,
                    retries,
                    self._max_queue_full_retries,
                )

                if (
                    self._max_queue_full_retries is not None
                    and retries >= self._max_queue_full_retries
                ):
                    err = SyncStreamBridgeError(
                        f"SyncStreamBridge queue remained full after "
                        f"{retries} attempts (framework={self._framework})"
                    )
                    self._push_error(err)
                    self._safe_enqueue_sentinel()
                    return

    def _safe_enqueue_sentinel(self) -> None:
        """Best-effort enqueue of the sentinel without blocking."""
        try:
            self._queue.put_nowait(self._sentinel)
        except queue.Full:
            logger.debug(
                "SyncStreamBridge: queue full when enqueuing sentinel "
                "for framework=%s",
                self._framework,
            )

    # ------------------------------------------------------------------ #
    # Error handling
    # ------------------------------------------------------------------ #

    def _push_error(self, exc: BaseException) -> None:
        """
        Record an error for later propagation and attach debug context.

        - Only the first error is recorded; subsequent errors are ignored.
        - Error context attachment failures are logged and swallowed, but
          do not override or replace the original exception.
        """
        with self._lock:
            if self._error is None:
                self._error = exc
                if self._error_context:
                    try:
                        attach_context(
                            exc,
                            framework=self._framework,
                            **self._error_context,
                        )
                    except Exception as ctx_exc:  # noqa: BLE001
                        # We explicitly *only* swallow context attachment
                        # failures here; the original error is still stored
                        # and will be raised in the main thread.
                        logger.debug(
                            "SyncStreamBridge: failed to attach error context "
                            "for framework=%s: %s",
                            self._framework,
                            ctx_exc,
                        )

            # Ensure the consumer unblocks and exits.
            self._safe_enqueue_sentinel()


__all__ = [
    "SyncStreamBridge",
    "SyncStreamBridgeError",
]
