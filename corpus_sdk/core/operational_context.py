# corpus_sdk/core/operation_context.py
# SPDX-License-Identifier: Apache-2.0

"""
Core OperationContext type for Corpus SDK.

This module defines a protocol-agnostic `OperationContext` used across
all Corpus protocol layers (LLM, vector, embedding, graph, etc.) and
framework adapters.

Goals
-----
- Provide a single, lightweight carrier for request-scoped metadata.
- Be safe to construct directly or via framework translation helpers.
- Keep the shape stable across protocol layers.
- Avoid any dependencies on specific backends or frameworks.

Typical usage
-------------

    from corpus_sdk.core.operation_context import OperationContext

    # Create directly
    ctx = OperationContext(
        request_id="req-123",
        tenant="tenant-a",
        deadline_ms=5_000,
        traceparent="00-...-...",
        attrs={"tags": ["demo"], "route": "llm"},
    )

    # Use helpers
    child = ctx.with_updates(
        attrs={"sub_route": "rerank"},
        deadline_ms=3_000,  # override deadline
    )

    # Serialize
    as_dict = ctx.to_dict()

    # Deserialize
    ctx2 = OperationContext.from_dict(as_dict)

Notes
-----
- `attrs` is the escape hatch for protocol- or framework-specific data.
  It should remain mostly JSON-serializable, but this is not enforced.
- The core fields (request_id, tenant, deadline_ms, traceparent) are
  intentionally minimal and stable; avoid adding protocol-specific
  fields here. Use `attrs` instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Mapping, Optional


@dataclass
class OperationContext:
    """
    Protocol-agnostic request context for Corpus operations.

    Fields
    ------
    request_id:
        Stable identifier for the logical request or run. Typically used
        for tracing, logging, and idempotency. Optional.

    tenant:
        Multi-tenant identifier (customer, workspace, project, etc.).
        This may be sensitive; loggers should treat it accordingly.
        Optional.

    deadline_ms:
        Optional absolute deadline in milliseconds *from "now"* or
        relative timeout, depending on how callers interpret it.
        Common pattern: interpreter converts this into an absolute
        timestamp when scheduling work.

    traceparent:
        W3C traceparent header value, if present. Used for distributed
        tracing propagation. Optional.

    attrs:
        Free-form attribute bag for framework- or protocol-specific
        metadata. Recommended keys include:
            - "tags": list[str]
            - "framework": str
            - "framework_version": str
            - "route": str
            - "component": str
            - "caller": str
        But any additional keys are allowed.
    """

    request_id: Optional[str] = None
    tenant: Optional[str] = None
    deadline_ms: Optional[int] = None
    traceparent: Optional[str] = None
    attrs: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Core helpers
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this context into a normalized dict.

        Shape:
            {
                "request_id": str | None,
                "tenant": str | None,
                "deadline_ms": int | None,
                "traceparent": str | None,
                "attrs": dict,
            }
        """
        return {
            "request_id": self.request_id,
            "tenant": self.tenant,
            "deadline_ms": self.deadline_ms,
            "traceparent": self.traceparent,
            "attrs": dict(self.attrs) if self.attrs is not None else {},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OperationContext":
        """
        Create an OperationContext from a normalized dict.

        Missing keys default to None / {}.
        """
        if data is None:
            return cls()
        return cls(
            request_id=data.get("request_id"),
            tenant=data.get("tenant"),
            deadline_ms=data.get("deadline_ms"),
            traceparent=data.get("traceparent"),
            attrs=dict(data.get("attrs") or {}),
        )

    # ------------------------------------------------------------------ #
    # Non-destructive mutation helpers
    # ------------------------------------------------------------------ #

    def with_updates(
        self,
        *,
        request_id: Optional[str] = None,
        tenant: Optional[str] = None,
        deadline_ms: Optional[int] = None,
        traceparent: Optional[str] = None,
        attrs: Optional[Mapping[str, Any]] = None,
        merge_attrs: bool = True,
    ) -> "OperationContext":
        """
        Return a new OperationContext with optional overrides.

        Parameters
        ----------
        request_id, tenant, deadline_ms, traceparent:
            If not None, replace the corresponding field. If None, the
            existing value is preserved.

        attrs:
            If provided:
                - when merge_attrs=True (default): merged into a copy of
                  the existing attrs, with `attrs` taking precedence.
                - when merge_attrs=False: replaces attrs entirely.

        merge_attrs:
            Whether to merge attrs dicts instead of replacing them.

        Returns
        -------
        OperationContext
            New instance with requested updates applied.
        """
        new_attrs: Dict[str, Any]
        if attrs is None:
            new_attrs = dict(self.attrs)
        elif merge_attrs:
            new_attrs = dict(self.attrs)
            new_attrs.update(attrs)
        else:
            new_attrs = dict(attrs)

        return replace(
            self,
            request_id=request_id if request_id is not None else self.request_id,
            tenant=tenant if tenant is not None else self.tenant,
            deadline_ms=deadline_ms if deadline_ms is not None else self.deadline_ms,
            traceparent=traceparent if traceparent is not None else self.traceparent,
            attrs=new_attrs,
        )

    def with_attr(self, key: str, value: Any) -> "OperationContext":
        """
        Convenience helper to set a single attribute in attrs, returning
        a new OperationContext.

        Example
        -------
            ctx = ctx.with_attr("route", "llm.complete")
        """
        new_attrs = dict(self.attrs)
        new_attrs[key] = value
        return replace(self, attrs=new_attrs)

    def with_tags(self, *tags: str, merge: bool = True) -> "OperationContext":
        """
        Add or replace tags in attrs.

        Parameters
        ----------
        *tags:
            Tags to add.

        merge:
            If True (default), merges with existing tags and de-duplicates.
            If False, replaces any existing tags entirely.

        Returns
        -------
        OperationContext
        """
        existing = self.attrs.get("tags") if isinstance(self.attrs.get("tags"), list) else []
        if merge:
            combined = list(dict.fromkeys([*existing, *tags]))
        else:
            combined = list(tags)

        new_attrs = dict(self.attrs)
        new_attrs["tags"] = combined
        return replace(self, attrs=new_attrs)

    # ------------------------------------------------------------------ #
    # Introspection helpers
    # ------------------------------------------------------------------ #

    def get_attr(self, key: str, default: Any = None) -> Any:
        """Safe lookup helper for attrs."""
        return self.attrs.get(key, default) if self.attrs is not None else default

    def copy(self) -> "OperationContext":
        """Explicit shallow copy."""
        return replace(self, attrs=dict(self.attrs))

    # You can add deadline-based helpers here later if you introduce a
    # stronger convention for absolute vs. relative deadlines.


__all__ = [
    "OperationContext",
]
