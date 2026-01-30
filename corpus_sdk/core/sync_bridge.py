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
import inspect
import logging
import queue
import threading
import time
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
    Union,
    cast,
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

        Note:
        The preferred contract is "awaitable -> AsyncIterator". For
        maximum adapter compatibility, this bridge also accepts factories
        that directly return an AsyncIterator. This is validated and
        handled explicitly in the worker without weakening error safety.

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

        Important:
        The bridge will *not* set this external event; it is treated as
        input-only. Internally, the bridge uses its own cancellation event
        for shutdown to avoid surprising side effects on callers.

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
        coro_factory: Callable[
            [],
            Union[
                Awaitable[AsyncIterator[T]],
                AsyncIterator[T],
            ],
        ],
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
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=max(queue_maxsize, 0))
        self._poll_timeout_s = float(poll_timeout_s)
        self._join_timeout_s = float(join_timeout_s)

        # External cancellation is treated as input-only (never set by the bridge).
        self._external_cancel_event = cancel_event

        # Internal cancellation is controlled by the bridge for clean shutdown.
        self._internal_cancel_event = threading.Event()

        self._framework = framework or "unknown"
        self._error_context: Dict[str, Any] = dict(error_context or {})
        self._max_transient_retries = max(0, int(max_transient_retries))
        self._transient_backoff_s = max(0.0, float(transient_backoff_s))
        self._transient_error_types = transient_error_types

        # None => unbounded retries. If the user supplies 0 or negative, treat it as None
        # (unbounded), matching the existing "None means unbounded" behavior and keeping
        # semantics stable for callers who intentionally pass 0.
        self._max_queue_full_retries = (
            int(max_queue_full_retries)
            if max_queue_full_retries is not None and max_queue_full_retries > 0
            else None
        )

        self._thread_name = thread_name or f"corpus_sync_stream_{self._framework}"

        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[BaseException] = None

        # Sentinel is a control marker in the data queue indicating completion.
        # The bridge also uses an explicit done event (below) as a robust fallback
        # in case sentinel insertion cannot complete under extreme backpressure.
        self._sentinel: object = object()
        self._sentinel_enqueued = False

        # Worker lifecycle events:
        # - _ready_event: set once the worker has either successfully created the stream
        #   (or produced a terminal error) so that run() can propagate "call-time" errors
        #   deterministically without using arbitrary sleep.
        # - _done_event: set when the worker is done producing items (success or error).
        self._ready_event = threading.Event()
        self._done_event = threading.Event()

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
        - Checks for immediate call-time errors (brief wait).
        - Returns an iterator that consumes items from the queue.
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
            in the calling thread, either immediately (for call-time errors)
            or after iteration completes (for in-stream errors).
        """
        self._ensure_worker_started()

        # Brief wait to catch call-time errors before returning iterator.
        # This ensures that if the adapter's stream method raises immediately,
        # we propagate that exception synchronously rather than deferring it
        # until the first iteration attempt.
        #
        # Unlike a fixed sleep, waiting on _ready_event is deterministic:
        # the worker sets _ready_event once it has either created the stream
        # or recorded an error (including immediate failures).
        calltime_check_timeout = min(0.05, self._poll_timeout_s)
        self._ready_event.wait(timeout=calltime_check_timeout)

        # Check if worker hit an immediate error
        if self._error is not None:
            raise self._error

        def _iterator() -> Iterator[T]:
            try:
                while True:
                    # Check cancellation with proper queue state handling.
                    # Cancellation may come from an external event (caller-controlled)
                    # or the internal event (bridge-controlled shutdown).
                    if self._is_cancelled():
                        # Drain any remaining items before exiting.
                        # This preserves as much already-produced data as practical.
                        try:
                            while True:
                                item = self._queue.get_nowait()
                                if item is self._sentinel:
                                    break
                                yield cast(T, item)
                        except queue.Empty:
                            pass
                        break

                    try:
                        item = self._queue.get(timeout=self._poll_timeout_s)
                    except queue.Empty:
                        # If worker is done and the queue is empty, we can exit.
                        # This is a robust fallback even if sentinel delivery fails
                        # under extreme backpressure.
                        if self._done_event.is_set() and self._queue.empty():
                            break

                        # Also check thread liveness as an additional safety net.
                        # If thread died and the queue is empty, we should exit.
                        with self._lock:
                            thread = self._thread
                            if thread is not None and not thread.is_alive():
                                if self._sentinel_enqueued or self._queue.empty():
                                    break
                        continue

                    if item is self._sentinel:
                        # Worker signaled completion
                        with self._lock:
                            self._sentinel_enqueued = True
                        break

                    # Normal item
                    yield cast(T, item)
            finally:
                self._shutdown_worker()

            # Propagate worker error, if any.
            if self._error is not None:
                raise self._error

        return _iterator()

    # ------------------------------------------------------------------ #
    # Worker management
    # ------------------------------------------------------------------ #

    def _ensure_worker_started(self) -> None:
        """Start the worker thread if it is not already running."""
        with self._lock:
            # Double-check pattern: another thread might have started it
            if self._thread is not None and self._thread.is_alive():
                return

            # If we are restarting (or starting fresh), reset bridge-controlled state.
            # Note: external cancellation state is not modified.
            self._internal_cancel_event.clear()
            self._ready_event.clear()
            self._done_event.clear()
            self._sentinel_enqueued = False

            # Preserve "first error wins" behavior, but allow restart if the previous
            # attempt ended. If callers re-use the instance intentionally, they likely
            # expect a clean state. This is safe because errors are always surfaced
            # to the caller via run().
            self._error = None

            def _thread_target() -> None:
                # Run worker logic under the captured contextvars.
                try:
                    self._parent_context.run(self._worker_main)
                except BaseException as exc:  # noqa: BLE001
                    # Last-resort: if something blows up at top-level, capture it.
                    logger.error(
                        "SyncStreamBridge worker crashed in framework=%s: %s",
                        self._framework,
                        exc,
                        exc_info=True,
                    )
                    self._push_error(exc)

            self._thread = threading.Thread(
                target=_thread_target,
                name=self._thread_name,
                daemon=True,
            )
            self._thread.start()
            logger.debug(
                "SyncStreamBridge: started worker thread %s for framework=%s",
                self._thread.name,
                self._framework,
            )

    def _shutdown_worker(self) -> None:
        """Signal cancellation and join the worker thread (best-effort)."""
        # Signal bridge-controlled cancellation first (do not mutate external cancel_event).
        self._internal_cancel_event.set()

        with self._lock:
            thread = self._thread

        if thread is None:
            return

        # Give worker time to clean up
        thread.join(self._join_timeout_s)
        if thread.is_alive():
            logger.warning(
                "SyncStreamBridge worker thread %s did not exit within %.3fs (framework=%s)",
                thread.name,
                self._join_timeout_s,
                self._framework,
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
            logger.error(
                "SyncStreamBridge asyncio.run failed in framework=%s: %s",
                self._framework,
                exc,
                exc_info=True,
            )
            self._push_error(exc)
        finally:
            # Mark done before attempting sentinel insertion so the consumer
            # can exit even if the queue is saturated.
            self._done_event.set()

            # Always ensure sentinel is enqueued (best-effort).
            self._ensure_sentinel_enqueued()

    async def _worker_coro(self) -> None:
        """
        Core async worker logic with optional transient retry.

        Calls `coro_factory()` to get an `AsyncIterator[T]` and streams
        items into the queue.
        """
        attempt = 0
        items_yielded = False

        while True:
            if self._is_cancelled():
                return

            try:
                result = self._coro_factory()

                # Explicit, safe normalization:
                # - If the factory returns an awaitable, await it to get the stream.
                # - Otherwise, treat it as an AsyncIterator directly (supported for
                #   adapter compatibility, as documented).
                if inspect.isawaitable(result):
                    stream = await cast(Awaitable[AsyncIterator[T]], result)
                else:
                    # Validate that it looks like an async iterator.
                    # This avoids accidental acceptance of arbitrary objects.
                    if not (hasattr(result, "__aiter__") and hasattr(result, "__anext__")):
                        raise TypeError(
                            "SyncStreamBridge coro_factory must return an awaitable "
                            "resolving to an AsyncIterator, or an AsyncIterator directly; "
                            f"got {type(result)!r}."
                        )
                    stream = cast(AsyncIterator[T], result)

                # Signal readiness after stream creation; if coro_factory fails,
                # we record the error and set readiness via _push_error.
                self._ready_event.set()

                async for item in stream:
                    if self._is_cancelled():
                        return
                    self._put_queue_item(item)
                    items_yielded = True

                # Normal completion - stream exhausted
                return

            except BaseException as exc:  # noqa: BLE001
                # Ensure call-time readiness is signaled even on immediate errors.
                # This allows run() to deterministically check and raise without sleeping.
                self._ready_event.set()

                # Once we've successfully yielded items, treat all errors as terminal
                if items_yielded:
                    logger.error(
                        "SyncStreamBridge error after yielding items in framework=%s: %s",
                        self._framework,
                        exc,
                        exc_info=True,
                    )
                    self._push_error(exc)
                    return

                # Check if error is transient and we have retries left
                is_transient = (
                    bool(self._transient_error_types)
                    and isinstance(exc, self._transient_error_types)
                )
                if (
                    is_transient
                    and attempt < self._max_transient_retries
                    and not self._is_cancelled()
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
                logger.error(
                    "SyncStreamBridge terminal error in framework=%s: %s",
                    self._framework,
                    exc,
                    exc_info=True,
                )
                self._push_error(exc)
                return

    # ------------------------------------------------------------------ #
    # Queue helpers
    # ------------------------------------------------------------------ #

    def _put_queue_item(self, item: Any) -> None:
        """
        Put an item into the queue with bounded retry when full.

        If the queue remains full after `max_queue_full_retries` attempts,
        records a `SyncStreamBridgeError` and returns without enqueuing.
        """
        retries = 0
        while True:
            if self._is_cancelled():
                return

            try:
                self._queue.put(item, timeout=self._poll_timeout_s)
                return
            except queue.Full:
                retries += 1

                if (
                    self._max_queue_full_retries is not None
                    and retries >= self._max_queue_full_retries
                ):
                    logger.error(
                        "SyncStreamBridge queue remained full after %d attempts (framework=%s); "
                        "consumer may be blocked or too slow",
                        retries,
                        self._framework,
                    )
                    err = SyncStreamBridgeError(
                        f"SyncStreamBridge queue remained full after "
                        f"{retries} attempts (framework={self._framework}). "
                        f"Consumer is not keeping up with producer."
                    )
                    self._push_error(err)
                    return

                if retries % 10 == 0:  # Log every 10th retry to avoid spam
                    logger.warning(
                        "SyncStreamBridge queue full for framework=%s; "
                        "retry=%d (max=%s)",
                        self._framework,
                        retries,
                        self._max_queue_full_retries,
                    )

    def _ensure_sentinel_enqueued(self) -> None:
        """
        Ensure sentinel is enqueued with aggressive retry.

        This is critical for consumer to exit properly. Uses multiple
        strategies to ensure sentinel delivery:
        1. Try non-blocking put
        2. If full, drain one item and retry
        3. If still failing after max attempts, set cancel event

        Clarification:
        The bridge preserves stream ordering and avoids dropping items. For that
        reason, it does not actually drain and reinsert items (which could reorder
        or lose data under contention). Instead, it retries sentinel insertion with
        bounded waits and relies on `_done_event` as a robust fallback so the
        consumer can still terminate when the worker completes.
        """
        with self._lock:
            if self._sentinel_enqueued:
                return
            self._sentinel_enqueued = True

        max_attempts = 100
        for attempt in range(max_attempts):
            if self._is_cancelled():
                return

            # Strategy: Fast-path non-blocking put.
            try:
                self._queue.put_nowait(self._sentinel)
                logger.debug(
                    "SyncStreamBridge: sentinel enqueued for framework=%s",
                    self._framework,
                )
                return
            except queue.Full:
                # Fall back to a timed put to avoid busy-waiting while preserving ordering.
                pass

            # Strategy: Timed put with small waits; does not reorder or drop items.
            try:
                self._queue.put(self._sentinel, timeout=self._poll_timeout_s)
                logger.debug(
                    "SyncStreamBridge: sentinel enqueued after waiting for space (framework=%s)",
                    self._framework,
                )
                return
            except queue.Full:
                if attempt % 10 == 0 and attempt > 0:
                    logger.warning(
                        "SyncStreamBridge: struggling to enqueue sentinel "
                        "(attempt %d/%d) for framework=%s",
                        attempt + 1,
                        max_attempts,
                        self._framework,
                    )

                # Small sleep to let consumer catch up.
                # This is in the worker thread (not the event loop thread), and is only
                # used in this terminal control-path; it avoids hot-spinning when the
                # queue is saturated.
                time.sleep(0.001)

        # Last resort: set internal cancel event to unblock consumer paths that are polling.
        # Note that `_done_event` is already set in _worker_main(), which allows the consumer
        # to exit even without sentinel delivery once the queue drains.
        logger.error(
            "SyncStreamBridge: failed to enqueue sentinel after %d attempts "
            "for framework=%s; setting internal cancel event as fallback",
            max_attempts,
            self._framework,
        )
        self._internal_cancel_event.set()

    # ------------------------------------------------------------------ #
    # Error handling
    # ------------------------------------------------------------------ #

    def _push_error(self, exc: BaseException) -> None:
        """
        Record an error for later propagation and attach debug context.

        - Only the first error is recorded; subsequent errors are ignored.
        - Error context attachment failures are logged and swallowed.
        - Ensures sentinel is enqueued exactly once per error.
        """
        sentinel_needed = False

        with self._lock:
            if self._error is None:
                self._error = exc
                sentinel_needed = True

                # Attach error context if available
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

        # Enqueue sentinel only if we were the first to record an error
        # This prevents multiple sentinels from being enqueued
        if sentinel_needed:
            self._ensure_sentinel_enqueued()

    # ------------------------------------------------------------------ #
    # Cancellation helpers
    # ------------------------------------------------------------------ #

    def _is_cancelled(self) -> bool:
        """
        Composite cancellation check.

        This preserves the existing behavior that an external cancellation event
        can terminate both producer and consumer quickly, while also ensuring
        the bridge can shut itself down without mutating the caller's event.
        """
        if self._internal_cancel_event.is_set():
            return True
        if self._external_cancel_event is not None and self._external_cancel_event.is_set():
            return True
        return False


__all__ = [
    "SyncStreamBridge",
    "SyncStreamBridgeError",
]
