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

CtxT = TypeVar("CtxT")

__all__ = [
    "make_ctx",
    "clone_ctx",
    "bump_deadline",
    "extend_budget_pct",
    "remaining_budget_ms",
    "now_ms",
]

def now_ms() -> int:
    """Return current epoch time in milliseconds."""
    return int(time.time() * 1000)

def _default_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:16]}"

def _default_traceparent() -> str:
    # Minimal W3C Trace Context (version 00) with random trace/span ids
    trace_id = uuid.uuid4().hex
    span_id = uuid.uuid4().hex[:16]
    return f"00-{trace_id}-{span_id}-01"

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
    """
    rid = request_id or _default_request_id()
    tp = traceparent or _default_traceparent()
    if deadline_ms is None and timeout_ms is not None:
        deadline_ms = now_ms() + int(timeout_ms)
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
    current = getattr(ctx, "deadline_ms", None)
    new_deadline = (current + int(add_ms)) if current is not None else now_ms() + int(add_ms)
    new_attrs = dict(getattr(ctx, "attrs", {}) or {})
    new_attrs["deadline_bumped_ms"] = int(add_ms)
    new_ctx = clone_ctx(factory, ctx, deadline_ms=new_deadline, attrs=new_attrs)
    return new_ctx

def extend_budget_pct(factory: Type[CtxT], ctx: Any, pct: float) -> CtxT:
    """
    Extend the remaining time budget by a percentage (e.g., 0.5 to add 50%).
    If no deadline present, returns the original context unchanged.
    """
    remaining = remaining_budget_ms(ctx)
    if remaining is None:
        return ctx
    add_ms = int(remaining * pct)
    return bump_deadline(factory, ctx, add_ms=add_ms)

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
