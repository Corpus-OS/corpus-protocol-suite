# SPDX-License-Identifier: Apache-2.0
"""
wire_adapter.py

Production-ready (schema-faithful) wire adapter for Corpus Protocol Suite v1.0.

This module is intentionally dependency-light and does NOT require importing any provider/base adapters.
It focuses only on building/validating canonical wire envelopes as defined by SCHEMA.md (normative).

Key features (SCHEMA.md-aligned):
- Canonical request envelopes (extensible):
    { "op": str, "ctx": {...}, "args": {...}, ...optional extra keys... }
  Wire-level invariants enforced here:
    - op is a non-empty string
    - ctx is an object (mapping)
    - args is an object (mapping)

- Canonical success envelopes (closed):
    { "ok": true, "code": "OK", "ms": number>=0, "result": any }
  NOTE: success envelopes are closed and code is const "OK".

- Canonical error envelopes (closed):
    {
      "ok": false,
      "code": "SOME_CODE",              # ^[A-Z_]+$
      "error": "ErrorTypeName",
      "message": "Human readable",
      "retry_after_ms": int>=0 | null,  # REQUIRED key (nullable)
      "details": object | null,         # REQUIRED key (nullable)
      "ms": number>=0
    }

- Canonical streaming success frames (closed):
    { "ok": true, "code": "STREAMING", "ms": number>=0, "chunk": any }
  Stream failures are represented by the standard error envelope (not a special streaming error frame).

- Minimal JSON-serializability checks for envelopes built here

Notes:
- This adapter does not perform provider calls, routing, caching, or rate limiting.
- Deadline enforcement is provided as an optional helper (build_deadline_exceeded_error) but is not imposed automatically.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Mapping, Optional, Union

JSON = Union[None, bool, int, float, str, list, dict]

_ERROR_CODE_RE = re.compile(r"^[A-Z_]+$")


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
    Normalize OperationContext to a canonical wire ctx shape.

    SCHEMA.md allows ctx to contain unknown fields (additionalProperties: true).
    This function:
      - converts ctx to a plain dict
      - (optionally) generates a request_id if absent (allowed by SCHEMA.md)
      - validates ctx.attrs is a mapping if present
      - validates JSON-serializability
    """
    d = _to_mapping(ctx)

    # Optional convenience: request_id is nullable/optional in SCHEMA.md, but helpful for correlation.
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
    - build_stream_frame(...)                 # streaming success frames only (STREAMING)
    - build_deadline_exceeded_error(...)
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

    def build_success_envelope(self, result: Any, ms: float, code: str = "OK") -> Dict[str, Any]:
        # SCHEMA.md: code is const "OK" for success envelopes.
        if code != "OK":
            raise ValueError("success.code must be 'OK' per SCHEMA.md")
        env = {"ok": True, "code": "OK", "ms": float(ms), "result": result}
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
        # SCHEMA.md: required keys include retry_after_ms and details (nullable).
        if not isinstance(code, str) or not code:
            raise ValueError("code must be a non-empty string")
        if not _ERROR_CODE_RE.match(code):
            raise ValueError("error.code must match ^[A-Z_]+$ per SCHEMA.md")
        if not isinstance(error, str) or not error:
            raise ValueError("error must be a non-empty string (type name)")
        if not isinstance(message, str):
            raise ValueError("message must be a string")

        if retry_after_ms is not None:
            retry_after_ms = int(retry_after_ms)
            if retry_after_ms < 0:
                raise ValueError("retry_after_ms must be >= 0 when provided")

        if details is not None and not isinstance(details, dict):
            raise TypeError("details must be a dict when provided")

        env: Dict[str, Any] = {
            "ok": False,
            "code": code,
            "error": error,
            "message": message,
            "retry_after_ms": retry_after_ms if retry_after_ms is not None else None,
            "details": details if details is not None else None,
            "ms": float(ms),
        }

        _require_jsonable(env, "error envelope")
        self._validate_error_envelope(env)
        return env

    def build_stream_frame(
        self, chunk: Any, ms: float, code: str = "STREAMING", ok: bool = True
    ) -> Dict[str, Any]:
        """
        Build a streaming *success* frame per SCHEMA.md:
          { ok:true, code:"STREAMING", ms:number>=0, chunk:any }

        Stream failures must use build_error_envelope() instead of ok=false streaming frames.
        The signature is retained for compatibility, but conflicting inputs are rejected.
        """
        if ok is not True:
            raise ValueError("streaming success frames must have ok=true per SCHEMA.md")
        if code != "STREAMING":
            raise ValueError("streaming success frames must have code='STREAMING' per SCHEMA.md")

        env = {"ok": True, "code": "STREAMING", "ms": float(ms), "chunk": chunk}
        _require_jsonable(env, "stream frame")
        self._validate_stream_frame(env)
        return env

    # Optional helper for a common error case described in the spec/schema.
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
        # SCHEMA.md: request envelope is extensible; do not require exact keys.
        for k in ("op", "ctx", "args"):
            if k not in env:
                raise ValueError(f"request envelope missing required key: {k}")

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
        if env.get("code") != "OK":
            raise ValueError("success.code must be 'OK' per SCHEMA.md")
        if not isinstance(env["ms"], (int, float)):
            raise ValueError("success.ms must be a number")
        if float(env["ms"]) < 0:
            raise ValueError("success.ms must be >= 0")

        # SCHEMA.md: success envelope is closed (additionalProperties: false)
        allowed = {"ok", "code", "ms", "result"}
        extra = set(env.keys()) - allowed
        if extra:
            raise ValueError(f"success envelope contains unexpected keys: {sorted(extra)}")

    def _validate_error_envelope(self, env: Dict[str, Any]) -> None:
        if env.get("ok") is not False:
            raise ValueError("error envelope ok must be false")
        for k in ("code", "error", "message", "retry_after_ms", "details", "ms"):
            if k not in env:
                raise ValueError(f"error envelope missing {k}")

        if not isinstance(env["code"], str) or not env["code"] or not _ERROR_CODE_RE.match(env["code"]):
            raise ValueError("error.code must be a non-empty string matching ^[A-Z_]+$")
        if not isinstance(env["error"], str) or not env["error"]:
            raise ValueError("error.error must be a non-empty string")
        if not isinstance(env["message"], str):
            raise ValueError("error.message must be a string")

        if not isinstance(env["ms"], (int, float)):
            raise ValueError("error.ms must be a number")
        if float(env["ms"]) < 0:
            raise ValueError("error.ms must be >= 0")

        ram = env["retry_after_ms"]
        if ram is not None:
            if not isinstance(ram, int):
                raise ValueError("error.retry_after_ms must be an int or null")
            if ram < 0:
                raise ValueError("error.retry_after_ms must be >= 0 when not null")

        det = env["details"]
        if det is not None and not isinstance(det, dict):
            raise ValueError("error.details must be an object or null")

        # SCHEMA.md: error envelope is closed (additionalProperties: false)
        allowed = {"ok", "code", "error", "message", "retry_after_ms", "details", "ms"}
        extra = set(env.keys()) - allowed
        if extra:
            raise ValueError(f"error envelope contains unexpected keys: {sorted(extra)}")

    def _validate_stream_frame(self, env: Dict[str, Any]) -> None:
        for k in ("ok", "code", "ms", "chunk"):
            if k not in env:
                raise ValueError(f"stream frame missing {k}")
        if env.get("ok") is not True:
            raise ValueError("stream.ok must be true for streaming success frames")
        if env.get("code") != "STREAMING":
            raise ValueError("stream.code must be 'STREAMING' for streaming success frames")
        if not isinstance(env["ms"], (int, float)):
            raise ValueError("stream.ms must be a number")
        if float(env["ms"]) < 0:
            raise ValueError("stream.ms must be >= 0")

        # SCHEMA.md: stream success envelope is closed (additionalProperties: false)
        allowed = {"ok", "code", "ms", "chunk"}
        extra = set(env.keys()) - allowed
        if extra:
            raise ValueError(f"stream frame contains unexpected keys: {sorted(extra)}")

    # -----------------------------------------
    # Operation-specific convenience builders
    # -----------------------------------------
    # Naming: build_<protocol>_<op>_envelope
    # Op strings match SCHEMA.md / PROTOCOLS.md v1.0 exactly.

    # -----------------------
    # Graph (v1.0)
    # -----------------------

    def build_graph_capabilities_envelope(self, ctx: Any = None, args: Any = None) -> Dict[str, Any]:
        return self.build_request_envelope("graph.capabilities", ctx, args or {})

    def build_graph_upsert_nodes_envelope(self, ctx: Any = None, args: Any = None) -> Dict[str, Any]:
        # SCHEMA.md: graph.upsert_nodes args: {nodes:[GraphNode...], namespace?:string}
        default_args = {
            "nodes": [
                {
                    "id": "node1",
                    "labels": ["Person"],
                    "properties": {"name": "Alice", "age": 30},
                }
            ]
        }
        return self.build_request_envelope("graph.upsert_nodes", ctx, args or default_args)

    def build_graph_upsert_edges_envelope(self, ctx: Any = None, args: Any = None) -> Dict[str, Any]:
        # SCHEMA.md: graph.upsert_edges args: {edges:[GraphEdge...], namespace?:string}
        default_args = {
            "edges": [
                {
                    "id": "edge1",
                    "src": "node1",
                    "dst": "node2",
                    "label": "KNOWS",
                    "properties": {"since": 2020},
                }
            ]
        }
        return self.build_request_envelope("graph.upsert_edges", ctx, args or default_args)

    def build_graph_delete_nodes_envelope(self, ctx: Any = None, args: Any = None) -> Dict[str, Any]:
        # SCHEMA.md: ids OR filter, optional namespace
        default_args = {"ids": ["node1"], "namespace": "default"}
        return self.build_request_envelope("graph.delete_nodes", ctx, args or default_args)

    def build_graph_delete_edges_envelope(self, ctx: Any = None, args: Any = None) -> Dict[str, Any]:
        # SCHEMA.md: ids OR filter, optional namespace
        default_args = {"ids": ["edge1"], "namespace": "default"}
        return self.build_request_envelope("graph.delete_edges", ctx, args or default_args)

    def build_graph_query_envelope(self, ctx: Any = None, args: Any = None) -> Dict[str, Any]:
        # SCHEMA.md graph.types.query_spec.json is closed (additionalProperties: false)
        default_args = {
            "text": "MATCH (n) RETURN n LIMIT 10",
            "dialect": "cypher",
            "params": {},
            "namespace": None,
            "timeout_ms": 1000,
            "stream": False,
        }
        return self.build_request_envelope("graph.query", ctx, args or default_args)

    def build_graph_stream_query_envelope(self, ctx: Any = None, args: Any = None) -> Dict[str, Any]:
        # Same query_spec; op indicates streaming; stream flag is allowed and can be true.
        default_args = {
            "text": "MATCH (n) RETURN n",
            "dialect": "cypher",
            "params": {},
            "namespace": None,
            "timeout_ms": 1000,
            "stream": True,
        }
        return self.build_request_envelope("graph.stream_query", ctx, args or default_args)

    def build_graph_bulk_vertices_envelope(self, ctx: Any = None, args: Any = None) -> Dict[str, Any]:
        # SCHEMA.md bulk_vertices_spec is closed; ids are not part of the spec.
        default_args = {"namespace": None, "limit": 100, "cursor": None, "filter": None}
        return self.build_request_envelope("graph.bulk_vertices", ctx, args or default_args)

    def build_graph_batch_envelope(self, ctx: Any = None, args: Any = None) -> Dict[str, Any]:
        # SCHEMA.md batch args: {ops:[{op:str, args:object}...]}
        default_args = {
            "ops": [
                {
                    "op": "graph.upsert_nodes",
                    "args": {
                        "nodes": [
                            {
                                "id": "node1",
                                "labels": ["Person"],
                                "properties": {"name": "Alice"},
                            }
                        ]
                    },
                }
            ]
        }
        return self.build_request_envelope("graph.batch", ctx, args or default_args)

    def build_graph_get_schema_envelope(self, ctx: Any = None, args: Any = None) -> Dict[str, Any]:
        return self.build_request_envelope("graph.get_schema", ctx, args or {})

    def build_graph_health_envelope(self, ctx: Any = None, args: Any = None) -> Dict[str, Any]:
        return self.build_request_envelope("graph.health", ctx, args or {})

    def build_graph_transaction_envelope(self, ctx: Any = None, args: Any = None) -> Dict[str, Any]:
        # SCHEMA.md transaction args: {operations:[{op:str, args:object}...]}
        default_args = {
            "operations": [
                {
                    "op": "graph.upsert_nodes",
                    "args": {
                        "nodes": [
                            {
                                "id": "node1",
                                "labels": ["Person"],
                                "properties": {"name": "Alice"},
                            }
                        ]
                    },
                }
            ]
        }
        return self.build_request_envelope("graph.transaction", ctx, args or default_args)

    def build_graph_traversal_envelope(self, ctx: Any = None, args: Any = None) -> Dict[str, Any]:
        # SCHEMA.md traversal_spec is closed.
        default_args = {
            "start_nodes": ["node1"],
            "max_depth": 1,
            "direction": "OUTGOING",
            "relationship_types": None,
            "node_filters": None,
            "relationship_filters": None,
            "return_properties": None,
            "namespace": None,
        }
        return self.build_request_envelope("graph.traversal", ctx, args or default_args)

    # -----------------------
    # LLM (v1.0)
    # -----------------------

    def build_llm_capabilities_envelope(self, ctx: Any = None, args: Any = None) -> Dict[str, Any]:
        return self.build_request_envelope("llm.capabilities", ctx, args or {})

    def build_llm_complete_envelope(self, ctx: Any = None, args: Any = None) -> Dict[str, Any]:
        default_args = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-3.5-turbo",
        }
        return self.build_request_envelope("llm.complete", ctx, args or default_args)

    def build_llm_stream_envelope(self, ctx: Any = None, args: Any = None) -> Dict[str, Any]:
        # SCHEMA.md stream spec does not require a 'stream' flag; op signals streaming.
        default_args = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-3.5-turbo",
        }
        return self.build_request_envelope("llm.stream", ctx, args or default_args)

    def build_llm_count_tokens_envelope(self, ctx: Any = None, args: Any = None) -> Dict[str, Any]:
        default_args = {"text": "Hello world", "model": "gpt-3.5-turbo"}
        return self.build_request_envelope("llm.count_tokens", ctx, args or default_args)

    def build_llm_health_envelope(self, ctx: Any = None, args: Any = None) -> Dict[str, Any]:
        return self.build_request_envelope("llm.health", ctx, args or {})

    # -----------------------
    # Vector (v1.0)
    # -----------------------

    def build_vector_capabilities_envelope(self, ctx: Any = None, args: Any = None) -> Dict[str, Any]:
        return self.build_request_envelope("vector.capabilities", ctx, args or {})

    def build_vector_query_envelope(self, ctx: Any = None, args: Any = None) -> Dict[str, Any]:
        default_args = {
            "vector": [0.1, 0.2, 0.3],
            "top_k": 5,
            "namespace": "default",
            "filter": None,
            "include_metadata": True,
            "include_vectors": False,
        }
        return self.build_request_envelope("vector.query", ctx, args or default_args)

    def build_vector_batch_query_envelope(self, ctx: Any = None, args: Any = None) -> Dict[str, Any]:
        default_args = {
            "queries": [
                {
                    "vector": [0.1, 0.2, 0.3],
                    "top_k": 5,
                    "namespace": "default",
                    "filter": None,
                    "include_metadata": True,
                    "include_vectors": False,
                }
            ],
            "namespace": "default",
        }
        return self.build_request_envelope("vector.batch_query", ctx, args or default_args)

    def build_vector_upsert_envelope(self, ctx: Any = None, args: Any = None) -> Dict[str, Any]:
        default_args = {
            "vectors": [
                {"id": "vec1", "vector": [0.1, 0.2, 0.3], "metadata": None, "namespace": None, "text": None}
            ],
            "namespace": "default",
        }
        return self.build_request_envelope("vector.upsert", ctx, args or default_args)

    def build_vector_delete_envelope(self, ctx: Any = None, args: Any = None) -> Dict[str, Any]:
        default_args = {"ids": ["vec1"], "namespace": "default"}
        return self.build_request_envelope("vector.delete", ctx, args or default_args)

    def build_vector_create_namespace_envelope(self, ctx: Any = None, args: Any = None) -> Dict[str, Any]:
        default_args = {"namespace": "test_namespace", "dimensions": 3, "distance_metric": "cosine"}
        return self.build_request_envelope("vector.create_namespace", ctx, args or default_args)

    def build_vector_delete_namespace_envelope(self, ctx: Any = None, args: Any = None) -> Dict[str, Any]:
        default_args = {"namespace": "test_namespace"}
        return self.build_request_envelope("vector.delete_namespace", ctx, args or default_args)

    def build_vector_health_envelope(self, ctx: Any = None, args: Any = None) -> Dict[str, Any]:
        return self.build_request_envelope("vector.health", ctx, args or {})

    # -----------------------
    # Embedding (v1.0)
    # -----------------------

    def build_embedding_capabilities_envelope(self, ctx: Any = None, args: Any = None) -> Dict[str, Any]:
        return self.build_request_envelope("embedding.capabilities", ctx, args or {})

    def build_embedding_embed_envelope(self, ctx: Any = None, args: Any = None) -> Dict[str, Any]:
        # SCHEMA.md: embed is unary; stream must be absent or false if present.
        default_args = {"text": "Hello, world!", "model": "text-embedding-ada-002"}
        return self.build_request_envelope("embedding.embed", ctx, args or default_args)

    def build_embedding_stream_embed_envelope(self, ctx: Any = None, args: Any = None) -> Dict[str, Any]:
        default_args = {"text": "Hello, world!", "model": "text-embedding-ada-002"}
        return self.build_request_envelope("embedding.stream_embed", ctx, args or default_args)

    def build_embedding_embed_batch_envelope(self, ctx: Any = None, args: Any = None) -> Dict[str, Any]:
        default_args = {"texts": ["Hello", "World"], "model": "text-embedding-ada-002"}
        return self.build_request_envelope("embedding.embed_batch", ctx, args or default_args)

    def build_embedding_count_tokens_envelope(self, ctx: Any = None, args: Any = None) -> Dict[str, Any]:
        default_args = {"text": "Hello, world!", "model": "text-embedding-ada-002"}
        return self.build_request_envelope("embedding.count_tokens", ctx, args or default_args)

    def build_embedding_health_envelope(self, ctx: Any = None, args: Any = None) -> Dict[str, Any]:
        return self.build_request_envelope("embedding.health", ctx, args or {})

    def build_embedding_get_stats_envelope(self, ctx: Any = None, args: Any = None) -> Dict[str, Any]:
        return self.build_request_envelope("embedding.get_stats", ctx, args or {})
