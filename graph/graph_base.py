# adapter_sdk/graph_base.py
# SPDX-License-Identifier: Apache-2.0
"""
Adapter SDK — Graph Protocol V1
A minimal, production-quality surface for building graph adapters.
"""

from __future__ import annotations
import hashlib
import time
from dataclasses import dataclass
from typing import (
    Any, Dict, Iterable, List, Mapping, Optional, Protocol, Tuple, 
    runtime_checkable, AsyncIterator
)

GRAPH_PROTOCOL_VERSION = "1.0.0"
KNOWN_DIALECTS: Tuple[str, ...] = ("cypher", "opencypher", "gremlin", "gql")

class AdapterError(Exception):
    def __init__(
        self,
        message: str = "",
        *,
        code: Optional[str] = None,
        retry_after_ms: Optional[int] = None,
        throttle_scope: Optional[str] = None,
        suggested_batch_reduction: Optional[int] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.retry_after_ms = retry_after_ms
        self.throttle_scope = throttle_scope
        self.suggested_batch_reduction = suggested_batch_reduction
        self.details = dict(details or {})

    def asdict(self) -> Dict[str, Any]:
        return {
            "message": self.message,
            "code": self.code,
            "retry_after_ms": self.retry_after_ms,
            "throttle_scope": self.throttle_scope,
            "suggested_batch_reduction": self.suggested_batch_reduction,
            "details": {k: self.details[k] for k in sorted(self.details)},
        }

class BadRequest(AdapterError): ...
class AuthError(AdapterError): ...
class ResourceExhausted(AdapterError): ...
class TransientNetwork(AdapterError): ...
class Unavailable(AdapterError): ...
class NotSupported(AdapterError): ...

@dataclass(frozen=True)
class OperationContext:
    request_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    deadline_ms: Optional[int] = None
    traceparent: Optional[str] = None
    tenant: Optional[str] = None
    attrs: Mapping[str, Any] = None

    def __post_init__(self) -> None:
        if self.attrs is None:
            object.__setattr__(self, "attrs", {})

class LogSink(Protocol):
    def debug(self, message: str, *, extra: Optional[Mapping[str, Any]] = None) -> None: ...
    def info(self, message: str, *, extra: Optional[Mapping[str, Any]] = None) -> None: ...
    def warning(self, message: str, *, extra: Optional[Mapping[str, Any]] = None) -> None: ...
    def error(self, message: str, *, extra: Optional[Mapping[str, Any]] = None) -> None: ...

class NoopLogSink:
    def debug(self, message: str, **_: Any) -> None: ...
    def info(self, message: str, **_: Any) -> None: ...
    def warning(self, message: str, **_: Any) -> None: ...
    def error(self, message: str, **_: Any) -> None: ...

class MetricsSink(Protocol):
    def observe(
        self,
        *,
        component: str,
        op: str,
        ms: float,
        ok: bool,
        code: str = "OK",
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None: ...
    def counter(
        self,
        *,
        component: str,
        name: str,
        value: int = 1,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None: ...

class NoopMetrics:
    def observe(self, **_: Any) -> None: ...
    def counter(self, **_: Any) -> None: ...

@dataclass(frozen=True)
class GraphCapabilities:
    server: str
    version: str
    dialects: Tuple[str, ...] = ("cypher",)
    supports_txn: bool = True
    supports_schema_ops: bool = True
    max_batch_ops: Optional[int] = None
    retryable_codes: Tuple[str, ...] = ()
    rate_limit_unit: str = "requests_per_second"
    max_qps: Optional[int] = None
    idempotent_writes: bool = False
    supports_multi_tenant: bool = False
    supports_streaming: bool = False
    supports_bulk_ops: bool = False

@runtime_checkable
class GraphProtocolV1(Protocol):
    async def capabilities(self) -> GraphCapabilities: ...
    async def create_vertex(
        self, label: str, props: Mapping[str, Any], *, ctx: Optional[OperationContext] = None
    ) -> str: ...
    async def create_edge(
        self, label: str, from_id: str, to_id: str, props: Mapping[str, Any], *, ctx: Optional[OperationContext] = None
    ) -> str: ...
    async def delete_vertex(self, vertex_id: str, *, ctx: Optional[OperationContext] = None) -> None: ...
    async def delete_edge(self, edge_id: str, *, ctx: Optional[OperationContext] = None) -> None: ...
    async def query(
        self,
        *,
        dialect: str,
        text: str,
        params: Optional[Mapping[str, Any]] = None,
        ctx: Optional[OperationContext] = None,
    ) -> List[Mapping[str, Any]]: ...
    async def stream_query(
        self,
        *,
        dialect: str,
        text: str,
        params: Optional[Mapping[str, Any]] = None,
        ctx: Optional[OperationContext] = None,
    ) -> AsyncIterator[Mapping[str, Any]]: ...
    async def bulk_vertices(
        self, vertices: Iterable[Tuple[str, Mapping[str, Any]]], *, ctx: Optional[OperationContext] = None
    ) -> List[str]: ...
    async def batch(
        self,
        ops: Iterable[Mapping[str, Any]],
        *,
        ctx: Optional[OperationContext] = None,
    ) -> List[Mapping[str, Any]]: ...
    async def get_schema(self, *, ctx: Optional[OperationContext] = None) -> Dict[str, Any]: ...
    async def health(self, *, ctx: Optional[OperationContext] = None) -> Dict[str, Any]: ...

class _Base:
    _component = "graph"
    _noop_log_sink = NoopLogSink()

    def __init__(self, *, metrics: Optional[MetricsSink] = None, logs: Optional[LogSink] = None) -> None:
        self._metrics: MetricsSink = metrics or NoopMetrics()
        self._logs: LogSink = logs or self._noop_log_sink

    @staticmethod
    def _require_non_empty(name: str, value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise BadRequest(f"{name} must be a non-empty string")

    @staticmethod
    def _bucket_ms(ms: Optional[int]) -> Optional[str]:
        if ms is None or ms < 0: return None
        if ms < 1000: return "<1s"
        if ms < 5000: return "<5s"
        if ms < 15000: return "<15s"
        if ms < 60000: return "<60s"
        return ">=60s"

    @staticmethod
    def _tenant_hash(tenant: Optional[str]) -> Optional[str]:
        if not tenant: return None
        return hashlib.sha256(tenant.encode("utf-8")).hexdigest()[:12]

    def _record(
        self,
        op: str,
        t0: float,
        ok: bool,
        *,
        code: str = "OK",
        ctx: Optional[OperationContext] = None,
        **extra: Any,
    ) -> None:
        try:
            dt_ms = (time.monotonic() - t0) * 1000.0
            x = dict(extra or {})
            if ctx:
                x["deadline_bucket"] = self._bucket_ms(ctx.deadline_ms)
                x["tenant"] = self._tenant_hash(ctx.tenant)
            self._metrics.observe(
                component=self._component, op=op, ms=dt_ms, ok=ok, code=code, extra=x or None
            )
        except Exception:
            pass

class BaseGraphAdapter(_Base, GraphProtocolV1):
    async def capabilities(self) -> GraphCapabilities:
        return await self._do_capabilities()

    async def create_vertex(
        self, label: str, props: Mapping[str, Any], *, ctx: Optional[OperationContext] = None
    ) -> str:
        self._require_non_empty("label", label)
        t0 = time.monotonic()
        try:
            vid = await self._do_create_vertex(label, dict(props), ctx=ctx)
            self._record("create_vertex", t0, True, ctx=ctx)
            return str(vid)
        except AdapterError as e:
            self._record("create_vertex", t0, False, code=type(e).__name__, ctx=ctx)
            raise

    async def create_edge(
        self, label: str, from_id: str, to_id: str, props: Mapping[str, Any], *, ctx: Optional[OperationContext] = None
    ) -> str:
        for n, v in (("label", label), ("from_id", from_id), ("to_id", to_id)):
            self._require_non_empty(n, v)
        t0 = time.monotonic()
        try:
            eid = await self._do_create_edge(label, str(from_id), str(to_id), dict(props), ctx=ctx)
            self._record("create_edge", t0, True, ctx=ctx)
            return str(eid)
        except AdapterError as e:
            self._record("create_edge", t0, False, code=type(e).__name__, ctx=ctx)
            raise

    async def delete_vertex(self, vertex_id: str, *, ctx: Optional[OperationContext] = None) -> None:
        self._require_non_empty("vertex_id", vertex_id)
        t0 = time.monotonic()
        try:
            await self._do_delete_vertex(str(vertex_id), ctx=ctx)
            self._record("delete_vertex", t0, True, ctx=ctx)
        except AdapterError as e:
            self._record("delete_vertex", t0, False, code=type(e).__name__, ctx=ctx)
            raise

    async def delete_edge(self, edge_id: str, *, ctx: Optional[OperationContext] = None) -> None:
        self._require_non_empty("edge_id", edge_id)
        t0 = time.monotonic()
        try:
            await self._do_delete_edge(str(edge_id), ctx=ctx)
            self._record("delete_edge", t0, True, ctx=ctx)
        except AdapterError as e:
            self._record("delete_edge", t0, False, code=type(e).__name__, ctx=ctx)
            raise

    async def query(
        self,
        *,
        dialect: str,
        text: str,
        params: Optional[Mapping[str, Any]] = None,
        ctx: Optional[OperationContext] = None,
    ) -> List[Mapping[str, Any]]:
        self._require_non_empty("dialect", dialect)
        self._require_non_empty("text", text)
        t0 = time.monotonic()
        try:
            res = await self._do_query(dialect=dialect, text=text, params=params or {}, ctx=ctx)
            self._record("query", t0, True, ctx=ctx, dialect=dialect, rows=len(res))
            return res
        except AdapterError as e:
            self._record("query", t0, False, code=type(e).__name__, ctx=ctx, dialect=dialect)
            raise

    async def stream_query(
        self,
        *,
        dialect: str,
        text: str,
        params: Optional[Mapping[str, Any]] = None,
        ctx: Optional[OperationContext] = None,
    ) -> AsyncIterator[Mapping[str, Any]]:
        self._require_non_empty("dialect", dialect)
        self._require_non_empty("text", text)
        t0 = time.monotonic()
        try:
            count = 0
            async for row in self._do_stream_query(dialect=dialect, text=text, params=params or {}, ctx=ctx):
                count += 1
                yield row
            self._record("stream_query", t0, True, ctx=ctx, dialect=dialect, rows=count)
        except AdapterError as e:
            self._record("stream_query", t0, False, code=type(e).__name__, ctx=ctx, dialect=dialect)
            raise

    async def bulk_vertices(
        self, vertices: Iterable[Tuple[str, Mapping[str, Any]]], *, ctx: Optional[OperationContext] = None
    ) -> List[str]:
        vertex_list = list(vertices)
        t0 = time.monotonic()
        try:
            ids = await self._do_bulk_vertices(vertex_list, ctx=ctx)
            self._record("bulk_vertices", t0, True, ctx=ctx, count=len(vertex_list))
            return [str(id) for id in ids]
        except AdapterError as e:
            self._record("bulk_vertices", t0, False, code=type(e).__name__, ctx=ctx, count=len(vertex_list))
            raise

    async def batch(
        self,
        ops: Iterable[Mapping[str, Any]],
        *,
        ctx: Optional[OperationContext] = None,
    ) -> List[Mapping[str, Any]]:
        op_list = list(ops)
        t0 = time.monotonic()
        try:
            res = await self._do_batch(op_list, ctx=ctx)
            self._record("batch", t0, True, ctx=ctx, ops=len(op_list))
            return res
        except AdapterError as e:
            self._record("batch", t0, False, code=type(e).__name__, ctx=ctx, ops=len(op_list))
            raise

    async def get_schema(self, *, ctx: Optional[OperationContext] = None) -> Dict[str, Any]:
        t0 = time.monotonic()
        try:
            schema = await self._do_get_schema(ctx=ctx)
            self._record("get_schema", t0, True, ctx=ctx)
            return dict(schema)
        except AdapterError as e:
            self._record("get_schema", t0, False, code=type(e).__name__, ctx=ctx)
            raise

    async def health(self, *, ctx: Optional[OperationContext] = None) -> Dict[str, Any]:
        t0 = time.monotonic()
        try:
            h = await self._do_health(ctx=ctx)
            self._record("health", t0, True, ctx=ctx)
            return {
                "ok": bool(h.get("ok", True)),
                "read_only": bool(h.get("read_only", False)),
                "degraded": bool(h.get("degraded", False)),
                "version": str(h.get("version", "")),
                "server": str(h.get("server", "")),
            }
        except AdapterError as e:
            self._record("health", t0, False, code=type(e).__name__, ctx=ctx)
            raise

    async def _do_capabilities(self) -> GraphCapabilities:
        raise NotImplementedError

    async def _do_create_vertex(self, label: str, props: Dict[str, Any], *, ctx: Optional[OperationContext]) -> str:
        raise NotImplementedError

    async def _do_create_edge(
        self, label: str, from_id: str, to_id: str, props: Dict[str, Any], *, ctx: Optional[OperationContext]
    ) -> str:
        raise NotImplementedError

    async def _do_delete_vertex(self, vertex_id: str, *, ctx: Optional[OperationContext]) -> None:
        raise NotImplementedError

    async def _do_delete_edge(self, edge_id: str, *, ctx: Optional[OperationContext]) -> None:
        raise NotImplementedError

    async def _do_query(
        self, *, dialect: str, text: str, params: Mapping[str, Any], ctx: Optional[OperationContext]
    ) -> List[Mapping[str, Any]]:
        raise NotImplementedError

    async def _do_stream_query(
        self, *, dialect: str, text: str, params: Mapping[str, Any], ctx: Optional[OperationContext]
    ) -> AsyncIterator[Mapping[str, Any]]:
        raise NotImplementedError

    async def _do_bulk_vertices(
        self, vertices: List[Tuple[str, Mapping[str, Any]]], *, ctx: Optional[OperationContext]
    ) -> List[str]:
        raise NotImplementedError

    async def _do_batch(self, ops: List[Mapping[str, Any]], *, ctx: Optional[OperationContext]) -> List[Mapping[str, Any]]:
        raise NotImplementedError

    async def _do_get_schema(self, *, ctx: Optional[OperationContext]) -> Dict[str, Any]:
        raise NotImplementedError

    async def _do_health(self, *, ctx: Optional[OperationContext]) -> Dict[str, Any]:
        raise NotImplementedError

__all__ = [
    "GRAPH_PROTOCOL_VERSION", "KNOWN_DIALECTS", "AdapterError", "BadRequest", "AuthError", 
    "ResourceExhausted", "TransientNetwork", "Unavailable", "NotSupported", "OperationContext",
    "LogSink", "NoopLogSink", "MetricsSink", "NoopMetrics", "GraphCapabilities", 
    "GraphProtocolV1", "BaseGraphAdapter",
]
