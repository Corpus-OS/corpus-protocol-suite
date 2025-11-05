# SPDX-License-Identifier: Apache-2.0
"""
Context helpers for examples.

These utilities construct domain-specific OperationContext objects without
tight-coupling to any one protocol file. You pass in the context *factory*
from the specific domain (e.g., corpus_sdk.llm.llm_base.OperationContext),
and we instantiate it with sensible defaults and deadline plumbing.

Usage:
    from corpus_sdk.llm.llm_base import OperationContext as LLMContext
    from corpus_sdk.examples.common.ctx import make_ctx, remaining_budget_ms

    ctx = make_ctx(LLMContext, tenant="acme", timeout_ms=30_000)
    print(remaining_budget_ms(ctx))
"""
from __future__ import annotations

import time
import uuid
import dataclasses
from typing import Any, Callable, Mapping, Optional, Type, TypeVar, Dict
from functools import lru_cache

CtxT = TypeVar("CtxT")

__all__ = [
    "make_ctx",
    "clone_ctx", 
    "bump_deadline",
    "extend_budget_pct",
    "remaining_budget_ms",
    "now_ms",
    "is_expired",
    "with_timeout",
    "create_child_ctx",
    "parse_traceparent",
]

# ------------------------------------------------------------------------------
# Core utilities
# ------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _cached_now_ms() -> int:
    """Cached time for performance in tight loops."""
    return int(time.time() * 1000)

def now_ms() -> int:
    """Return current epoch time in milliseconds."""
    return _cached_now_ms()

def clear_time_cache():
    """Clear the time cache (useful for tests or long-running processes)."""
    _cached_now_ms.cache_clear()

def _default_request_id() -> str:
    """Generate a unique request ID."""
    return f"req_{uuid.uuid4().hex[:16]}"

def _default_traceparent() -> str:
    """Generate W3C Trace Context (version 00) with random trace/span ids."""
    trace_id = uuid.uuid4().hex
    span_id = uuid.uuid4().hex[:16]
    return f"00-{trace_id}-{span_id}-01"  # Sampled = true

def parse_traceparent(traceparent: str) -> Optional[Dict[str, str]]:
    """
    Parse W3C traceparent header for debugging and validation.
    
    Returns:
        Dict with version, trace_id, span_id, flags, sampled or None if invalid
    """
    try:
        parts = traceparent.split('-')
        if len(parts) != 4:
            return None
        version, trace_id, span_id, flags = parts
        # Validate format
        if (len(version) == 2 and len(trace_id) == 32 and 
            len(span_id) == 16 and len(flags) == 2):
            return {
                "version": version,
                "trace_id": trace_id,
                "span_id": span_id,
                "flags": flags,
                "sampled": flags == "01"
            }
    except Exception:
        pass
    return None

# ------------------------------------------------------------------------------
# Context creation and manipulation
# ------------------------------------------------------------------------------

def make_ctx(
    factory: Type[CtxT],
    *,
    request_id: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    deadline_ms: Optional[int] = None,
    timeout_ms: Optional[int] = None,
    traceparent: Optional[str] = None,
    tenant: Optional[str] = None,
    attrs: Optional[Mapping[str, Any]] = None,
) -> CtxT:
    """
    Create a domain-specific OperationContext using the provided factory.

    Args:
        factory: The OperationContext class from the domain module.
        request_id: Correlation id; auto-generated if absent.
        idempotency_key: Optional exactly-once token for mutating ops.
        deadline_ms: Absolute epoch ms deadline. Mutually exclusive with timeout_ms.
        timeout_ms: Relative budget in ms from *now* (used if deadline_ms is None).
        traceparent: W3C trace context; auto-generated if absent.
        tenant: Multi-tenant isolation hint.
        attrs: Extra attributes to propagate.

    Returns:
        Instance of the provided OperationContext class.

    Raises:
        ValueError: If both deadline_ms and timeout_ms are provided, or if times are invalid.
    """
    # Validate inputs
    if deadline_ms is not None and timeout_ms is not None:
        raise ValueError("Cannot specify both deadline_ms and timeout_ms")
    if timeout_ms is not None and timeout_ms <= 0:
        raise ValueError("timeout_ms must be positive")
    if deadline_ms is not None and deadline_ms <= now_ms():
        raise ValueError("deadline_ms must be in the future")
    
    # Generate defaults
    rid = request_id or _default_request_id()
    tp = traceparent or _default_traceparent()
    
    # Convert relative timeout to absolute deadline
    if deadline_ms is None and timeout_ms is not None:
        deadline_ms = now_ms() + int(timeout_ms)
    
    # Prepare attributes
    payload: Dict[str, Any] = dict(attrs or {})
    
    return factory(
        request_id=rid,
        idempotency_key=idempotency_key,
        deadline_ms=deadline_ms,
        traceparent=tp,
        tenant=tenant,
        attrs=payload,
    )

def clone_ctx(factory: Type[CtxT], ctx: Any, **overrides: Any) -> CtxT:
    """
    Clone an OperationContext with overrides using the same factory.
    Works with frozen dataclasses (Corpus default).

    Example:
        new_ctx = clone_ctx(LLMContext, ctx, deadline_ms=now_ms()+5000)
    """
    if dataclasses.is_dataclass(ctx):
        base = dataclasses.asdict(ctx)
    else:
        base = {
            "request_id": getattr(ctx, "request_id", None),
            "idempotency_key": getattr(ctx, "idempotency_key", None),
            "deadline_ms": getattr(ctx, "deadline_ms", None),
            "traceparent": getattr(ctx, "traceparent", None),
            "tenant": getattr(ctx, "tenant", None),
            "attrs": dict(getattr(ctx, "attrs", {}) or {}),
        }
    base.update(overrides)
    return factory(**base)

def bump_deadline(factory: Type[CtxT], ctx: Any, *, add_ms: int) -> CtxT:
    """
    Return a copy of ctx with deadline extended by add_ms (if present),
    or set a new deadline add_ms from *now* if none exists.
    Also records the bump in attrs (for observability).
    """
    if add_ms <= 0:
        raise ValueError("add_ms must be positive")
        
    current = getattr(ctx, "deadline_ms", None)
    new_deadline = (current + int(add_ms)) if current is not None else now_ms() + int(add_ms)
    
    new_attrs = dict(getattr(ctx, "attrs", {}) or {})
    new_attrs["deadline_bumped_ms"] = int(add_ms)
    new_attrs["deadline_previous_ms"] = current
    
    new_ctx = clone_ctx(factory, ctx, deadline_ms=new_deadline, attrs=new_attrs)
    return new_ctx

def extend_budget_pct(factory: Type[CtxT], ctx: Any, pct: float) -> CtxT:
    """
    Extend the remaining time budget by a percentage (e.g., 0.5 to add 50%).
    If no deadline present, returns the original context unchanged.
    """
    if pct <= 0:
        raise ValueError("pct must be positive")
        
    remaining = remaining_budget_ms(ctx)
    if remaining is None or remaining <= 0:
        return ctx
        
    add_ms = int(remaining * pct)
    return bump_deadline(factory, ctx, add_ms=add_ms)

def with_timeout(factory: Type[CtxT], base_ctx: Any, timeout_ms: int) -> CtxT:
    """Create a new context with a specific timeout, inheriting other properties."""
    if timeout_ms <= 0:
        raise ValueError("timeout_ms must be positive")
        
    return make_ctx(
        factory, 
        request_id=getattr(base_ctx, "request_id", None),
        traceparent=getattr(base_ctx, "traceparent", None),
        tenant=getattr(base_ctx, "tenant", None),
        attrs=dict(getattr(base_ctx, "attrs", {}) or {}),
        timeout_ms=timeout_ms
    )

def create_child_ctx(factory: Type[CtxT], parent_ctx: Any, operation: str) -> CtxT:
    """Create a child context with new span ID for distributed tracing."""
    traceparent = getattr(parent_ctx, "traceparent", None)
    if traceparent:
        # Generate new span ID while preserving trace ID
        parsed = parse_traceparent(traceparent)
        if parsed:
            new_span_id = uuid.uuid4().hex[:16]
            traceparent = f"{parsed['version']}-{parsed['trace_id']}-{new_span_id}-{parsed['flags']}"
    
    attrs = dict(getattr(parent_ctx, "attrs", {}) or {})
    attrs["parent_operation"] = operation
    attrs["parent_request_id"] = getattr(parent_ctx, "request_id", None)
    
    return make_ctx(
        factory,
        request_id=_default_request_id(),  # New request ID for child
        traceparent=traceparent,
        tenant=getattr(parent_ctx, "tenant", None),
        attrs=attrs
    )

# ------------------------------------------------------------------------------
# Context inspection
# ------------------------------------------------------------------------------

def remaining_budget_ms(ctx: Any) -> Optional[int]:
    """
    Compute remaining time budget from ctx.deadline_ms.

    Returns:
        Remaining ms (>= 0), or None if no deadline is set.
    """
    deadline = getattr(ctx, "deadline_ms", None)
    if deadline is None:
        return None
    return max(0, int(deadline - now_ms()))

def is_expired(ctx: Any) -> bool:
    """Check if the context's deadline has expired."""
    remaining = remaining_budget_ms(ctx)
    return remaining is not None and remaining <= 0
