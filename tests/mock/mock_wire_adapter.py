"""
wire_adapter.py

Production-ready (protocol-faithful) wire adapter for Corpus Protocol Suite v1.0.

This module is intentionally dependency-light and does NOT require importing any provider/base adapters.
It focuses only on building/validating canonical wire envelopes as defined in PROTOCOLS.md v1.0.

Key features:
- Canonical request envelopes: { "op": str, "ctx": {...}, "args": {...} }
- Canonical success envelopes: { "ok": True, "code": "OK", "ms": float, "result": ... }
- Canonical error envelopes: { "ok": False, "code": "...", "error": "...", "message": "...", "retry_after_ms"?: int, "details"?: {...}, "ms": float }
- Canonical streaming frames: { "ok": True/False, "code": "...", "ms": float, "chunk": ... } (success) or error envelope (error)
- Minimal JSON-serializability checks for envelopes built here

Notes:
- This adapter does not perform provider calls, routing, caching, or rate limiting.
- Deadline enforcement is provided as an optional helper (build_deadline_exceeded_error) but is not imposed automatically.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Mapping, Optional, Union

JSON = Union[None, bool, int, float, str, list, dict]


def tenant_hash(tenant: str) -> str:
    """First 12 chars of SHA256(tenant) as required for telemetry use."""
    digest = hashlib.sha256(tenant.encode("utf-8")).hexdigest()
    return digest[:12]


def _is_jsonable(value: Any) -> bool:
    try:
        json.dumps(value)
        return True
    except Exception:
        return False


def _require_jsonable(value: Any, what: str) -> None:
    if not _is_jsonable(value):
        raise TypeError(f"{what} must be JSON-serializable")


def _to_mapping(obj: Any) -> Dict[str, Any]:
    """
    Convert a variety of context-like objects into a plain dict suitable for JSON.
    Accepts:
    - dict-like mappings
    - dataclasses
    - simple objects with __dict__
    """
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if isinstance(obj, Mapping):
        return dict(obj)
    if is_dataclass(obj):
        return asdict(obj)
    # Fall back to attribute dict (best-effort)
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict):
        return dict(d)
    raise TypeError(
        f"Unsupported ctx/args type: {type(obj)!r}. Expected mapping-like or dataclass."
    )


def _normalize_ctx(ctx: Any) -> Dict[str, Any]:
    """
    Normalize OperationContext to the canonical wire ctx shape.

    We do not force presence of optional fields; we simply pass through known fields if present.
    Unknown fields are preserved (attrs/extensions are allowed by spec).
    
    Auto-generates a request_id if not provided (required by wire protocol).
    """
    d = _to_mapping(ctx)
    
    # Auto-generate request_id if missing (required by protocol)
    if "request_id" not in d or d["request_id"] is None:
        d["request_id"] = f"req_{uuid.uuid4().hex[:16]}"
    
    # Ensure nested attrs is a mapping if present
    if "attrs" in d and d["attrs"] is not None and not isinstance(d["attrs"], Mapping):
        raise TypeError("ctx.attrs must be a mapping when provided")
    _require_jsonable(d, "ctx")
    return d


def _normalize_args(args: Any) -> Dict[str, Any]:
    d = _to_mapping(args)
    _require_jsonable(d, "args")
    return d


class WireAdapter:
    """
    Protocol v1.0 Wire Adapter.

    Public methods:
    - build_request_envelope(...)
    - build_success_envelope(...)
    - build_error_envelope(...)
    - build_stream_frame(...)
    - operation-specific convenience builders (build_<protocol>_<op>_envelope)
    """

    # -----------------------
    # Core envelope builders
    # -----------------------

    def build_request_envelope(
        self, op: str, ctx: Any = None, args: Any = None
    ) -> Dict[str, Any]:
        if not isinstance(op, str) or not op:
            raise ValueError("op must be a non-empty string")
        env = {
            "op": op,
            "ctx": _normalize_ctx(ctx),
            "args": _normalize_args(args),
        }
        self._validate_request_envelope(env)
        return env

    def build_success_envelope(
        self, result: Any, ms: float, code: str = "OK"
    ) -> Dict[str, Any]:
        if not isinstance(code, str) or not code:
            raise ValueError("code must be a non-empty string")
        env = {"ok": True, "code": code, "ms": float(ms), "result": result}
        _require_jsonable(env, "success envelope")
        self._validate_success_envelope(env)
        return env

    def build_error_envelope(
        self,
        *,
        error: str,
        message: str,
        code: str,
        ms: float,
        retry_after_ms: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not isinstance(code, str) or not code:
            raise ValueError("code must be a non-empty string")
        if not isinstance(error, str) or not error:
            raise ValueError("error must be a non-empty string (PascalCase type name)")
        if not isinstance(message, str):
            raise ValueError("message must be a string")
        env: Dict[str, Any] = {
            "ok": False,
            "code": code,
            "error": error,
            "message": message,
            "ms": float(ms),
        }
        if retry_after_ms is not None:
            env["retry_after_ms"] = int(retry_after_ms)
        if details is not None:
            if not isinstance(details, dict):
                raise TypeError("details must be a dict when provided")
            env["details"] = details

        _require_jsonable(env, "error envelope")
        self._validate_error_envelope(env)
        return env

    def build_stream_frame(
        self, chunk: Any, ms: float, code: str = "OK", ok: bool = True
    ) -> Dict[str, Any]:
        if not isinstance(code, str) or not code:
            raise ValueError("code must be a non-empty string")
        env = {"ok": bool(ok), "code": code, "ms": float(ms), "chunk": chunk}
        _require_jsonable(env, "stream frame")
        self._validate_stream_frame(env)
        return env

    # Optional helper for a common error case described in the spec.
    def build_deadline_exceeded_error(
        self,
        *,
        ms: float,
        message: str = "deadline already exceeded",
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.build_error_envelope(
            error="DeadlineExceeded",
            message=message,
            code="DEADLINE_EXCEEDED",
            ms=ms,
            retry_after_ms=None,
            details=details,
        )

    # -----------------------
    # Validation (wire-level)
    # -----------------------

    def _validate_request_envelope(self, env: Dict[str, Any]) -> None:
        if set(env.keys()) != {"op", "ctx", "args"}:
            raise ValueError("request envelope must have exactly keys: op, ctx, args")
        if not isinstance(env["op"], str) or not env["op"]:
            raise ValueError("request.op must be non-empty string")
        if not isinstance(env["ctx"], dict):
            raise ValueError("request.ctx must be an object")
        if not isinstance(env["args"], dict):
            raise ValueError("request.args must be an object")

    def _validate_success_envelope(self, env: Dict[str, Any]) -> None:
        if env.get("ok") is not True:
            raise ValueError("success envelope ok must be true")
        for k in ("code", "ms", "result"):
            if k not in env:
                raise ValueError(f"success envelope missing {k}")
        if not isinstance(env["code"], str) or not env["code"]:
            raise ValueError("success.code must be a non-empty string")
        if not isinstance(env["ms"], (int, float)):
            raise ValueError("success.ms must be a number")

    def _validate_error_envelope(self, env: Dict[str, Any]) -> None:
        if env.get("ok") is not False:
            raise ValueError("error envelope ok must be false")
        for k in ("code", "error", "message", "ms"):
            if k not in env:
                raise ValueError(f"error envelope missing {k}")
        if not isinstance(env["code"], str) or not env["code"]:
            raise ValueError("error.code must be a non-empty string")
        if not isinstance(env["error"], str) or not env["error"]:
            raise ValueError("error.error must be a non-empty string")
        if not isinstance(env["message"], str):
            raise ValueError("error.message must be a string")
        if not isinstance(env["ms"], (int, float)):
            raise ValueError("error.ms must be a number")
        if (
            "retry_after_ms" in env
            and env["retry_after_ms"] is not None
            and not isinstance(env["retry_after_ms"], int)
        ):
            raise ValueError("error.retry_after_ms must be an int when provided")
        if (
            "details" in env
            and env["details"] is not None
            and not isinstance(env["details"], dict)
        ):
            raise ValueError("error.details must be an object when provided")

    def _validate_stream_frame(self, env: Dict[str, Any]) -> None:
        for k in ("ok", "code", "ms", "chunk"):
            if k not in env:
                raise ValueError(f"stream frame missing {k}")
        if not isinstance(env["ok"], bool):
            raise ValueError("stream.ok must be boolean")
        if not isinstance(env["code"], str) or not env["code"]:
            raise ValueError("stream.code must be non-empty string")
        if not isinstance(env["ms"], (int, float)):
            raise ValueError("stream.ms must be a number")

    # -----------------------------------------
    # Operation-specific convenience builders
    # -----------------------------------------
    # Naming: build_<protocol>_<op>_envelope
    # Op strings match PROTOCOLS.md v1.0 exactly.

    # Graph (11)
    def build_graph_capabilities_envelope(
        self, ctx: Any = None, args: Any = None
    ) -> Dict[str, Any]:
        return self.build_request_envelope("graph.capabilities", ctx, args or {})

    def build_graph_upsert_nodes_envelope(
        self, ctx: Any = None, args: Any = None
    ) -> Dict[str, Any]:
        # upsert_nodes requires nodes array
        default_args = {
            "nodes": [
                {
                    "id": "node1",
                    "label": "Person",
                    "props": {"name": "Alice", "age": 30}
                }
            ]
        }
        return self.build_request_envelope("graph.upsert_nodes", ctx, args or default_args)

    def build_graph_upsert_edges_envelope(
        self, ctx: Any = None, args: Any = None
    ) -> Dict[str, Any]:
        # create_edge requires label, from_id, to_id, props (single edge)
        default_args = {
            "label": "KNOWS",
            "from_id": "node1",
            "to_id": "node2",
            "props": {"since": 2020}
        }
        return self.build_request_envelope("graph.create_edge", ctx, args or default_args)

    def build_graph_delete_nodes_envelope(
        self, ctx: Any = None, args: Any = None
    ) -> Dict[str, Any]:
        # delete_nodes requires ids or label (adding namespace to disambiguate from delete_edge)
        default_args = {
            "ids": ["node1"],
            "namespace": "default"
        }
        return self.build_request_envelope("graph.delete_nodes", ctx, args or default_args)

    def build_graph_delete_edges_envelope(
        self, ctx: Any = None, args: Any = None
    ) -> Dict[str, Any]:
        # delete_edge requires ids or type (using type to disambiguate from delete_nodes)
        default_args = {
            "type": "KNOWS"
        }
        return self.build_request_envelope("graph.delete_edge", ctx, args or default_args)

    def build_graph_query_envelope(
        self, ctx: Any = None, args: Any = None
    ) -> Dict[str, Any]:
        # query requires dialect and text
        default_args = {
            "dialect": "cypher",
            "text": "MATCH (n) RETURN n LIMIT 10",
            "page_size": 10
        }
        return self.build_request_envelope("graph.query", ctx, args or default_args)

    def build_graph_stream_query_envelope(
        self, ctx: Any = None, args: Any = None
    ) -> Dict[str, Any]:
        # stream_query requires dialect and text
        default_args = {
            "dialect": "cypher",
            "text": "MATCH (n) RETURN n",
            "page_size": 10
        }
        return self.build_request_envelope("graph.stream_query", ctx, args or default_args)

    def build_graph_bulk_vertices_envelope(
        self, ctx: Any = None, args: Any = None
    ) -> Dict[str, Any]:
        # bulk_vertices requires ids or label
        default_args = {
            "ids": ["node1", "node2"]
        }
        return self.build_request_envelope("graph.bulk_vertices", ctx, args or default_args)

    def build_graph_batch_envelope(
        self, ctx: Any = None, args: Any = None
    ) -> Dict[str, Any]:
        # batch requires ops array with proper op types
        default_args = {
            "ops": [
                {
                    "op": "create_vertex",
                    "label": "Person",
                    "props": {"name": "Alice"}
                }
            ]
        }
        return self.build_request_envelope("graph.batch", ctx, args or default_args)

    def build_graph_get_schema_envelope(
        self, ctx: Any = None, args: Any = None
    ) -> Dict[str, Any]:
        return self.build_request_envelope("graph.get_schema", ctx, args or {})

    def build_graph_health_envelope(
        self, ctx: Any = None, args: Any = None
    ) -> Dict[str, Any]:
        return self.build_request_envelope("graph.health", ctx, args or {})

    # LLM (5)
    def build_llm_capabilities_envelope(
        self, ctx: Any = None, args: Any = None
    ) -> Dict[str, Any]:
        # capabilities operations typically have empty args
        return self.build_request_envelope("llm.capabilities", ctx, args or {})

    def build_llm_complete_envelope(
        self, ctx: Any = None, args: Any = None
    ) -> Dict[str, Any]:
        # Provide minimal valid args for llm.complete
        default_args = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-3.5-turbo",
        }
        return self.build_request_envelope("llm.complete", ctx, args or default_args)

    def build_llm_stream_envelope(
        self, ctx: Any = None, args: Any = None
    ) -> Dict[str, Any]:
        # Same as complete but for streaming
        default_args = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-3.5-turbo",
            "stream": True,
        }
        return self.build_request_envelope("llm.stream", ctx, args or default_args)

    def build_llm_count_tokens_envelope(
        self, ctx: Any = None, args: Any = None
    ) -> Dict[str, Any]:
        # count_tokens requires text or messages
        default_args = {
            "text": "Hello world",
            "model": "gpt-3.5-turbo",
        }
        return self.build_request_envelope("llm.count_tokens", ctx, args or default_args)

    def build_llm_health_envelope(
        self, ctx: Any = None, args: Any = None
    ) -> Dict[str, Any]:
        # health operations typically have empty args
        return self.build_request_envelope("llm.health", ctx, args or {})

    # Vector (7)
    def build_vector_capabilities_envelope(
        self, ctx: Any = None, args: Any = None
    ) -> Dict[str, Any]:
        return self.build_request_envelope("vector.capabilities", ctx, args or {})

    def build_vector_query_envelope(
        self, ctx: Any = None, args: Any = None
    ) -> Dict[str, Any]:
        # query requires vector or text
        default_args = {
            "vector": [0.1, 0.2, 0.3],
            "k": 5,
            "namespace": "default",
        }
        return self.build_request_envelope("vector.query", ctx, args or default_args)

    def build_vector_upsert_envelope(
        self, ctx: Any = None, args: Any = None
    ) -> Dict[str, Any]:
        # upsert requires vectors
        default_args = {
            "vectors": [
                {"id": "vec1", "values": [0.1, 0.2, 0.3]},
            ],
            "namespace": "default",
        }
        return self.build_request_envelope("vector.upsert", ctx, args or default_args)

    def build_vector_delete_envelope(
        self, ctx: Any = None, args: Any = None
    ) -> Dict[str, Any]:
        # delete requires ids, filter, or delete_all
        default_args = {
            "ids": ["vec1"],
            "namespace": "default",
        }
        return self.build_request_envelope("vector.delete", ctx, args or default_args)

    def build_vector_create_namespace_envelope(
        self, ctx: Any = None, args: Any = None
    ) -> Dict[str, Any]:
        default_args = {
            "namespace": "test_namespace",
            "dimension": 3,
        }
        return self.build_request_envelope("vector.namespace_create", ctx, args or default_args)

    def build_vector_delete_namespace_envelope(
        self, ctx: Any = None, args: Any = None
    ) -> Dict[str, Any]:
        default_args = {
            "namespace": "test_namespace",
        }
        return self.build_request_envelope("vector.namespace_delete", ctx, args or default_args)

    def build_vector_health_envelope(
        self, ctx: Any = None, args: Any = None
    ) -> Dict[str, Any]:
        return self.build_request_envelope("vector.health", ctx, args or {})

    # Embedding (5)
    def build_embedding_capabilities_envelope(
        self, ctx: Any = None, args: Any = None
    ) -> Dict[str, Any]:
        return self.build_request_envelope("embedding.capabilities", ctx, args or {})

    def build_embedding_embed_envelope(
        self, ctx: Any = None, args: Any = None
    ) -> Dict[str, Any]:
        # embed requires text or texts
        default_args = {
            "text": "Hello, world!",
            "model": "text-embedding-ada-002",
        }
        return self.build_request_envelope("embedding.embed", ctx, args or default_args)

    def build_embedding_embed_batch_envelope(
        self, ctx: Any = None, args: Any = None
    ) -> Dict[str, Any]:
        # embed_batch requires texts array
        default_args = {
            "texts": ["Hello", "World"],
            "model": "text-embedding-ada-002",
        }
        return self.build_request_envelope("embedding.embed_batch", ctx, args or default_args)

    def build_embedding_count_tokens_envelope(
        self, ctx: Any = None, args: Any = None
    ) -> Dict[str, Any]:
        # count_tokens requires text
        default_args = {
            "text": "Hello, world!",
            "model": "text-embedding-ada-002",
        }
        return self.build_request_envelope("embedding.count_tokens", ctx, args or default_args)

    def build_embedding_health_envelope(
        self, ctx: Any = None, args: Any = None
    ) -> Dict[str, Any]:
        return self.build_request_envelope("embedding.health", ctx, args or {})
