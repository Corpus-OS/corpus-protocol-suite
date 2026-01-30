# corpus_sdk/graph/framework_adapters/common/graph_translation.py
# SPDX-License-Identifier: Apache-2.0
"""
Framework-agnostic Graph → Framework translation layer.

Purpose
-------
Provide a high-level orchestration and translation layer between:

- The Corpus Graph Protocol V1 (`GraphProtocolV1` / `BaseGraphAdapter`), and
- Framework-specific graph integrations (LangChain, LlamaIndex, SK, AutoGen, CrewAI, custom).

This module is intentionally *framework-neutral* and focuses on:

- Building `GraphQuerySpec` / mutation specs from framework-level inputs
- Translating `QueryResult` / `QueryChunk` / mutation results back to framework-facing shapes
- Applying optional MMR / diversification on query results
- Handling rich metadata filters (including $and / $or / range operators)
- Providing sync + async APIs, including streaming via a sync bridge
- Attaching rich error context for observability

Context translation
-------------------
This module does **not** parse framework configs directly. Instead:

- `corpus_sdk.core.context_translation` is responsible for taking framework-native
  contexts (LangChain RunnableConfig, LlamaIndex CallbackManager, etc.) and producing
  a core `OperationContext`.
- Callers pass either an `OperationContext` or a simple dict-like context into
  the methods here; we normalize that via `from_dict` into the graph adapter's
  `OperationContext`.

MMR / diversification
---------------------
This layer optionally supports Maximal Marginal Relevance (MMR) re-ranking on
graph query results that expose:

- A relevance score field (e.g. "score")
- An embedding field (e.g. "embedding")

The similarity metric for diversity is configurable (defaults to cosine).

Filter handling
---------------
Metadata / property filters can use a rich, adapter-agnostic DSL:

- Logical combinators:    {"$and": [...]} / {"$or": [...]}
- Range operators:        {"field": {"$gt": v, "$lt": v2}} or tuple ["field", ">", v]
- Simple equality:        {"field": value}

The normalized filter is passed through to the graph adapter, which may further
interpret or constrain semantics based on its own capabilities.

Streaming
---------
For streaming queries, this module exposes:

- An async API that yields translated framework chunks, and
- A sync API that wraps the async generator via `SyncStreamBridge`, preserving
  proper cancellation and error propagation.

Registry
--------
A small registry lets you register per-framework graph translators:

- `register_graph_translator("my_framework", factory)`
- `create_graph_translator("my_framework", adapter, ...)`

This makes it straightforward to plug in framework-specific behaviors while
reusing the common orchestration logic here.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, asdict
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    Union,
)

from corpus_sdk.graph.graph_base import (
    BatchOperation,
    BatchResult,
    BulkVerticesResult,
    BulkVerticesSpec,
    DeleteEdgesSpec,
    DeleteNodesSpec,
    GraphCapabilities,
    GraphQuerySpec,
    GraphSchema,
    GraphProtocolV1,
    GraphTraversalSpec,
    OperationContext,
    QueryChunk,
    QueryResult,
    TraversalResult,
    UpsertEdgesSpec,
    UpsertNodesSpec,
    BadRequest,
)
from corpus_sdk.core.context_translation import from_dict as ctx_from_dict
from corpus_sdk.core.sync_bridge import SyncStreamBridge
from corpus_sdk.core.async_bridge import AsyncBridge
from corpus_sdk.core.error_context import attach_context

LOG = logging.getLogger(__name__)

R = TypeVar("R")
Record = Mapping[str, Any]


# =============================================================================
# Helpers: OperationContext normalization
# =============================================================================


def _ensure_operation_context(
    ctx: Optional[Union[OperationContext, Mapping[str, Any]]],
) -> OperationContext:
    """
    Normalize various context shapes into a graph OperationContext.

    Accepts:
        - None: returns an "empty" context
        - OperationContext: returned as-is
        - Mapping[str, Any]: interpreted via context_translation.from_dict,
          then adapted into a graph OperationContext.
        - Duck-typed context: any object with request_id/attrs attributes.
    """
    # 1) No context → build from empty dict
    if ctx is None:
        core_ctx = ctx_from_dict({})
        return OperationContext(
            request_id=getattr(core_ctx, "request_id", None),
            idempotency_key=getattr(core_ctx, "idempotency_key", None),
            deadline_ms=getattr(core_ctx, "deadline_ms", None),
            traceparent=getattr(core_ctx, "traceparent", None),
            tenant=getattr(core_ctx, "tenant", None),
            attrs=getattr(core_ctx, "attrs", None) or {},
        )

    # 2) Already our graph OperationContext → just use it
    if isinstance(ctx, OperationContext):
        return ctx

    # 3) Mapping → go through core context translation
    if isinstance(ctx, Mapping):
        core_ctx = ctx_from_dict(ctx)
        return OperationContext(
            request_id=getattr(core_ctx, "request_id", None),
            idempotency_key=getattr(core_ctx, "idempotency_key", None),
            deadline_ms=getattr(core_ctx, "deadline_ms", None),
            traceparent=getattr(core_ctx, "traceparent", None),
            tenant=getattr(core_ctx, "tenant", None),
            attrs=getattr(core_ctx, "attrs", None) or {},
        )

    # 4) Duck-typed context (covers cases where another layer already wrapped it)
    if hasattr(ctx, "request_id") or hasattr(ctx, "attrs"):
        return OperationContext(
            request_id=getattr(ctx, "request_id", None),
            idempotency_key=getattr(ctx, "idempotency_key", None),
            deadline_ms=getattr(ctx, "deadline_ms", None),
            traceparent=getattr(ctx, "traceparent", None),
            tenant=getattr(ctx, "tenant", None),
            attrs=getattr(ctx, "attrs", None) or {},
        )

    # 5) Everything else → hard error
    raise BadRequest(
        f"Unsupported context type: {type(ctx).__name__}",
        code="BAD_OPERATION_CONTEXT",
    )


# =============================================================================
# MMR configuration
# =============================================================================


SimilarityFn = Callable[[Sequence[float], Sequence[float]], float]


@dataclass(frozen=True)
class MMRConfig:
    """
    Configuration for Maximal Marginal Relevance re-ranking.

    Attributes:
        enabled:
            Whether to apply MMR. If False, results are returned as-is.

        k:
            Number of results to keep after diversification. If None, the full
            set of records is re-ordered but not truncated.

        lambda_mult:
            Tradeoff between relevance and diversity, in [0, 1].
            Higher → more weight on original relevance scores.
            Lower  → more weight on diversity between results.

        score_key:
            Record field name containing the original relevance score.
            Must be convertible to float.

        vector_key:
            Record field name containing the embedding vector for diversity.
            Must be a sequence of floats.

        similarity_fn:
            Optional custom similarity metric between two embedding vectors.
            If None, cosine similarity is used.
    """

    enabled: bool = False
    k: Optional[int] = None
    lambda_mult: float = 0.5
    score_key: str = "score"
    vector_key: str = "embedding"
    similarity_fn: Optional[SimilarityFn] = None

    def __post_init__(self) -> None:
        """Validate MMR configuration parameters."""
        if self.enabled:
            if not (0.0 <= self.lambda_mult <= 1.0):
                raise ValueError(f"lambda_mult must be in [0, 1], got {self.lambda_mult}")
            if self.k is not None and self.k < 0:
                raise ValueError(f"k must be non-negative, got {self.k}")

    def effective_k(self, n: int) -> int:
        """Resolve final k against the number of available records."""
        if not self.enabled:
            return n
        if self.k is None or self.k <= 0:
            return n
        return min(self.k, n)


# =============================================================================
# Filter translation helpers
# =============================================================================


class FilterTranslator:
    """
    Helper for normalizing framework-level filter DSLs into a mapping suitable
    for GraphProtocol filters.

    Supported input shapes (examples):
        - {"field": "value"}
        - {"$and": [ {...}, {...} ]}
        - {"$or":  [ {...}, {...} ]}
        - ["field", ">", 10]          -> {"field": {"$gt": 10}}
        - [ ["age", ">", 18], ["age", "<", 65] ]
        - [{"field": "v1"}, {"field": "v2"}]  -> {"$and": [...]}

    This module intentionally does not enforce the semantics of these operators;
    adapters can interpret or restrict them as needed.
    """

    _RANGE_OP_MAP: Mapping[str, str] = {
        ">": "$gt",
        "<": "$lt",
        ">=": "$gte",
        "<=": "$lte",
        "==": "$eq",
        "=": "$eq",
        "!=": "$ne",
    }

    def normalize(self, raw: Any) -> Optional[Mapping[str, Any]]:
        """Normalize arbitrary filter DSL into a mapping, or None."""
        if raw is None:
            return None

        if isinstance(raw, Mapping):
            return self._normalize_mapping(raw)

        if isinstance(raw, (list, tuple)):
            return self._normalize_sequence(raw)

        raise BadRequest(
            f"Unsupported filter type: {type(raw).__name__}",
            code="BAD_FILTER",
        )

    def _normalize_mapping(self, raw: Mapping[str, Any]) -> Mapping[str, Any]:
        # Logical combinators
        if "$and" in raw or "$or" in raw:
            out: Dict[str, Any] = {}
            for key, value in raw.items():
                if key in ("$and", "$or") and isinstance(value, (list, tuple)):
                    normalized_parts: List[Mapping[str, Any]] = []
                    for idx, v in enumerate(value):
                        nv = self.normalize(v)
                        if nv is None:
                            continue
                        if not isinstance(nv, Mapping):
                            raise BadRequest(
                                f"Nested filter at index {idx} must be a mapping",
                                code="BAD_FILTER",
                                details={"index": idx},
                            )
                        normalized_parts.append(nv)
                    out[key] = normalized_parts
                else:
                    out[key] = value
            return out

        # Simple equality / basic range map
        out2: Dict[str, Any] = {}
        for key, value in raw.items():
            if isinstance(value, (list, tuple)) and len(value) == 2:
                op, v = value
                mapped = self._RANGE_OP_MAP.get(str(op))
                if mapped:
                    out2[key] = {mapped: v}
                else:
                    LOG.warning(
                        "FilterTranslator: unrecognized range operator %r for field %r; "
                        "keeping original value",
                        op,
                        key,
                    )
                    out2[key] = value
            else:
                out2[key] = value
        return out2

    def _normalize_sequence(self, raw: Sequence[Any]) -> Mapping[str, Any]:
        # Tuple condition: ["age", ">", 18]
        # FIX: Explicitly check if the middle element is a known string operator
        # to distinguish from data lists.
        if (
            len(raw) == 3
            and isinstance(raw[1], str)
            and raw[1] in self._RANGE_OP_MAP
            and not isinstance(raw[0], (list, tuple, Mapping))
        ):
            field, op, value = raw
            mapped = self._RANGE_OP_MAP[str(op)]
            return {str(field): {mapped: value}}

        # Otherwise, assume a list of filters combined via AND.
        parts: List[Mapping[str, Any]] = []
        for idx, part in enumerate(raw):
            nv = self.normalize(part)
            if nv is None:
                continue
            if not isinstance(nv, Mapping):
                raise BadRequest(
                    f"Filter at index {idx} must be a mapping",
                    code="BAD_FILTER",
                    details={"index": idx},
                )
            parts.append(nv)
        return {"$and": parts}


# =============================================================================
# Framework-agnostic translator protocol
# =============================================================================


class GraphFrameworkTranslator(Protocol):
    """
    Per-framework translator contract.

    Implementations are responsible for:
        - Converting framework-level query/mutation inputs into Graph*Spec types
        - Converting Graph results into framework-level outputs
        - (Optionally) applying MMR or other post-processing steps
    """

    # ---- query translation ----

    def build_query_spec(
        self,
        raw_query: Any,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
        stream: bool = False,
    ) -> GraphQuerySpec:
        ...

    def translate_query_result(
        self,
        result: QueryResult,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        ...

    def translate_query_chunk(
        self,
        chunk: QueryChunk,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        ...

    # ---- transaction translation ----

    def build_transaction_spec(
        self,
        raw_transaction: Any,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> List[BatchOperation]:
        ...

    def translate_transaction_result(
        self,
        result: BatchResult,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        ...

    # ---- traversal translation ----

    def build_traversal_spec(
        self,
        raw_traversal: Any,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> GraphTraversalSpec:
        ...

    def translate_traversal_result(
        self,
        result: TraversalResult,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        ...

    # ---- mutation translation ----

    def build_upsert_nodes_spec(
        self,
        raw_nodes: Any,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> UpsertNodesSpec:
        ...

    def build_upsert_edges_spec(
        self,
        raw_edges: Any,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> UpsertEdgesSpec:
        ...

    def build_delete_nodes_spec(
        self,
        raw_filter_or_ids: Any,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> DeleteNodesSpec:
        ...

    def build_delete_edges_spec(
        self,
        raw_filter_or_ids: Any,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> DeleteEdgesSpec:
        ...

    # ---- bulk / batch / schema ----

    def build_bulk_vertices_spec(
        self,
        raw_request: Any,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> BulkVerticesSpec:
        ...

    def translate_bulk_vertices_result(
        self,
        result: BulkVerticesResult,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        ...

    def build_batch_ops(
        self,
        raw_batch_ops: Any,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> List[BatchOperation]:
        ...

    def translate_batch_result(
        self,
        result: BatchResult,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        ...

    def translate_schema(
        self,
        schema: GraphSchema,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        ...

    # ---- capabilities / health ----

    def translate_capabilities(
        self,
        capabilities: GraphCapabilities,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        ...

    def translate_health(
        self,
        health: Mapping[str, Any],
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        ...

    # ---- optional hooks ----

    def preferred_namespace(
        self,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Optional[str]:
        """
        Optional hook for translators to derive a default namespace.

        This can come from:
            - framework_ctx (e.g., which index/graph is active)
            - op_ctx.attrs (e.g., "graph_namespace" key)
        """
        ...


# =============================================================================
# Default generic translator implementation
# =============================================================================


class DefaultGraphFrameworkTranslator:
    """
    Generic, framework-neutral translator implementation.
    """

    def __init__(self) -> None:
        self._filter_translator = FilterTranslator()

    # ---- namespace helper ----

    def preferred_namespace(
        self,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Optional[str]:
        # Priority: explicit framework_ctx > graph_namespace attr > namespace attr
        if isinstance(framework_ctx, Mapping):
            ns = framework_ctx.get("namespace")
            if ns is not None and str(ns).strip():
                return str(ns)

        attrs = op_ctx.attrs or {}

        ns = attrs.get("graph_namespace")
        if ns is not None and str(ns).strip():
            return str(ns)

        ns = attrs.get("namespace")
        if ns is not None and str(ns).strip():
            return str(ns)

        return None

    # ---- query translation ----

    def build_query_spec(
        self,
        raw_query: Any,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
        stream: bool = False,
    ) -> GraphQuerySpec:
        namespace = self.preferred_namespace(op_ctx=op_ctx, framework_ctx=framework_ctx)

        if isinstance(raw_query, str):
            return GraphQuerySpec(
                text=raw_query,
                dialect=None,
                params={},
                namespace=namespace,
                timeout_ms=None,
                stream=stream,
            )

        if isinstance(raw_query, Mapping):
            text = raw_query.get("text")
            if not isinstance(text, str) or not text.strip():
                raise BadRequest(
                    "raw_query.text must be a non-empty string",
                    code="BAD_QUERY",
                )

            dialect = raw_query.get("dialect")
            params = raw_query.get("params") or {}
            rq_namespace = raw_query.get("namespace")
            timeout_ms = raw_query.get("timeout_ms")

            return GraphQuerySpec(
                text=text,
                dialect=str(dialect) if dialect is not None else None,
                params=dict(params),
                namespace=str(rq_namespace) if rq_namespace is not None else namespace,
                timeout_ms=int(timeout_ms) if timeout_ms is not None else None,
                stream=bool(raw_query.get("stream", stream)),
            )

        raise BadRequest(
            f"Unsupported raw_query type: {type(raw_query).__name__}",
            code="BAD_QUERY",
        )

    def translate_query_result(
        self,
        result: QueryResult,
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Any:
        return {
            "records": list(result.records or []),
            "summary": dict(result.summary or {}),
            "dialect": result.dialect,
            "namespace": result.namespace,
        }

    def translate_query_chunk(
        self,
        chunk: QueryChunk,
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Any:
        return {
            "records": list(chunk.records or []),
            "is_final": bool(chunk.is_final),
            "summary": dict(chunk.summary or {}) if chunk.summary is not None else None,
        }

    # ---- transaction translation ----

    def build_transaction_spec(
        self,
        raw_transaction: Any,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> List[BatchOperation]:
        return self.build_batch_ops(
            raw_transaction,
            op_ctx=op_ctx,
            framework_ctx=framework_ctx,
        )

    def translate_transaction_result(
        self,
        result: BatchResult,
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Any:
        return {
            "success": bool(getattr(result, "success", True)),
            "results": list(result.results or []),
            "error": getattr(result, "error", None),
            "transaction_id": getattr(result, "transaction_id", None),
        }

    # ---- traversal translation ----

    def build_traversal_spec(
        self,
        raw_traversal: Any,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> GraphTraversalSpec:
        namespace = self.preferred_namespace(op_ctx=op_ctx, framework_ctx=framework_ctx)

        if isinstance(raw_traversal, Mapping):
            start_nodes = raw_traversal.get("start_nodes", [])
            max_depth = raw_traversal.get("max_depth", 1)
            direction = raw_traversal.get("direction", "OUTGOING")
            relationship_types = raw_traversal.get("relationship_types")
            node_filters = raw_traversal.get("node_filters")
            relationship_filters = raw_traversal.get("relationship_filters")
            return_properties = raw_traversal.get("return_properties")
            traversal_namespace = raw_traversal.get("namespace", namespace)

            normalized_node_filters = (
                self._filter_translator.normalize(node_filters) if node_filters else None
            )
            normalized_rel_filters = (
                self._filter_translator.normalize(relationship_filters)
                if relationship_filters
                else None
            )

            return GraphTraversalSpec(
                start_nodes=[str(node_id) for node_id in start_nodes],
                max_depth=int(max_depth),
                direction=str(direction),
                relationship_types=tuple(relationship_types)
                if relationship_types
                else None,
                node_filters=normalized_node_filters,
                relationship_filters=normalized_rel_filters,
                return_properties=tuple(return_properties)
                if return_properties
                else None,
                namespace=str(traversal_namespace) if traversal_namespace else namespace,
            )

        raise BadRequest(
            f"Unsupported raw_traversal type: {type(raw_traversal).__name__}",
            code="BAD_TRAVERSAL",
        )

    def translate_traversal_result(
        self,
        result: TraversalResult,
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Any:
        return {
            "nodes": list(result.nodes or []),
            "relationships": list(result.relationships or []),
            "paths": list(result.paths or []),
            "summary": dict(result.summary or {}),
            "namespace": result.namespace,
        }

    # ---- mutation translation ----

    def build_upsert_nodes_spec(
        self,
        raw_nodes: Any,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> UpsertNodesSpec:
        from corpus_sdk.graph.graph_base import Node  # local import to avoid cycles

        namespace = self.preferred_namespace(op_ctx=op_ctx, framework_ctx=framework_ctx)

        if isinstance(raw_nodes, Mapping):
            raw_nodes = [raw_nodes]
        if not isinstance(raw_nodes, Iterable):
            raise BadRequest(
                "raw_nodes must be a mapping or iterable",
                code="BAD_NODES",
            )

        nodes: List[Node] = []
        for idx, item in enumerate(raw_nodes):
            if isinstance(item, Node):
                nodes.append(item)
                continue
            if not isinstance(item, Mapping):
                raise BadRequest(
                    f"raw_nodes[{idx}] must be a mapping or Node",
                    code="BAD_NODES",
                    details={"index": idx, "type": type(item).__name__},
                )
            node_ns = item.get("namespace", namespace)
            node = Node(
                id=item.get("id"),
                labels=tuple(item.get("labels") or ()),
                properties=item.get("properties") or {},
                namespace=node_ns,
                created_at=item.get("created_at"),
                updated_at=item.get("updated_at"),
            )
            nodes.append(node)

        return UpsertNodesSpec(nodes=nodes, namespace=namespace)

    def build_upsert_edges_spec(
        self,
        raw_edges: Any,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> UpsertEdgesSpec:
        from corpus_sdk.graph.graph_base import Edge, GraphID  # local import

        namespace = self.preferred_namespace(op_ctx=op_ctx, framework_ctx=framework_ctx)

        if isinstance(raw_edges, Mapping):
            raw_edges = [raw_edges]
        if not isinstance(raw_edges, Iterable):
            raise BadRequest(
                "raw_edges must be a mapping or iterable",
                code="BAD_EDGES",
            )

        edges: List[Edge] = []
        for idx, item in enumerate(raw_edges):
            if isinstance(item, Edge):
                edges.append(item)
                continue
            if not isinstance(item, Mapping):
                raise BadRequest(
                    f"raw_edges[{idx}] must be a mapping or Edge",
                    code="BAD_EDGES",
                    details={"index": idx, "type": type(item).__name__},
                )
            edge_ns = item.get("namespace", namespace)
            edge = Edge(
                id=GraphID(str(item.get("id"))),
                src=GraphID(str(item.get("src"))),
                dst=GraphID(str(item.get("dst"))),
                label=str(item.get("label")),
                properties=item.get("properties") or {},
                namespace=edge_ns,
                created_at=item.get("created_at"),
                updated_at=item.get("updated_at"),
            )
            edges.append(edge)

        return UpsertEdgesSpec(edges=edges, namespace=namespace)

    def build_delete_nodes_spec(
        self,
        raw_filter_or_ids: Any,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> DeleteNodesSpec:
        from corpus_sdk.graph.graph_base import GraphID  # local import

        namespace = self.preferred_namespace(op_ctx=op_ctx, framework_ctx=framework_ctx)

        ids: List[GraphID] = []
        filter_expr: Optional[Mapping[str, Any]] = None

        if isinstance(raw_filter_or_ids, Mapping):
            filter_expr = self._filter_translator.normalize(raw_filter_or_ids)
        elif isinstance(raw_filter_or_ids, (list, tuple)):
            if not raw_filter_or_ids:
                ids = []
                filter_expr = None
            else:
                first = raw_filter_or_ids[0]
                if isinstance(first, Mapping):
                    filter_expr = self._filter_translator.normalize(raw_filter_or_ids)
                else:
                    ids = [GraphID(str(x)) for x in raw_filter_or_ids]
        else:
            ids = [GraphID(str(raw_filter_or_ids))]

        return DeleteNodesSpec(ids=ids, namespace=namespace, filter=filter_expr)

    def build_delete_edges_spec(
        self,
        raw_filter_or_ids: Any,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> DeleteEdgesSpec:
        from corpus_sdk.graph.graph_base import GraphID  # local import

        namespace = self.preferred_namespace(op_ctx=op_ctx, framework_ctx=framework_ctx)

        ids: List[GraphID] = []
        filter_expr: Optional[Mapping[str, Any]] = None

        if isinstance(raw_filter_or_ids, Mapping):
            filter_expr = self._filter_translator.normalize(raw_filter_or_ids)
        elif isinstance(raw_filter_or_ids, (list, tuple)):
            if not raw_filter_or_ids:
                ids = []
                filter_expr = None
            else:
                first = raw_filter_or_ids[0]
                if isinstance(first, Mapping):
                    filter_expr = self._filter_translator.normalize(raw_filter_or_ids)
                else:
                    ids = [GraphID(str(x)) for x in raw_filter_or_ids]
        else:
            ids = [GraphID(str(raw_filter_or_ids))]

        return DeleteEdgesSpec(ids=ids, namespace=namespace, filter=filter_expr)

    # ---- bulk / batch / schema ----

    def build_bulk_vertices_spec(
        self,
        raw_request: Any,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> BulkVerticesSpec:
        namespace = self.preferred_namespace(op_ctx=op_ctx, framework_ctx=framework_ctx)

        if raw_request is None:
            return BulkVerticesSpec(namespace=namespace)

        if not isinstance(raw_request, Mapping):
            raise BadRequest(
                "bulk_vertices request must be a mapping or None",
                code="BAD_BULK_VERTICES",
            )

        limit = raw_request.get("limit", 100)
        cursor = raw_request.get("cursor")
        filter_expr = self._filter_translator.normalize(raw_request.get("filter"))

        return BulkVerticesSpec(
            namespace=raw_request.get("namespace", namespace),
            limit=int(limit),
            cursor=cursor,
            filter=filter_expr,
        )

    def translate_bulk_vertices_result(
        self,
        result: BulkVerticesResult,
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Any:
        return {
            "nodes": list(result.nodes or []),
            "next_cursor": result.next_cursor,
            "has_more": bool(result.has_more),
        }

    def build_batch_ops(
        self,
        raw_batch_ops: Any,
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> List[BatchOperation]:
        if raw_batch_ops is None:
            raise BadRequest(
                "raw_batch_ops must not be None",
                code="BAD_BATCH",
            )
        if isinstance(raw_batch_ops, Mapping):
            raw_batch_ops = [raw_batch_ops]
        if not isinstance(raw_batch_ops, Iterable):
            raise BadRequest(
                "raw_batch_ops must be a mapping or iterable",
                code="BAD_BATCH",
            )

        ops: List[BatchOperation] = []
        for idx, item in enumerate(raw_batch_ops):
            if not isinstance(item, Mapping):
                raise BadRequest(
                    f"raw_batch_ops[{idx}] must be a mapping",
                    code="BAD_BATCH_OP",
                    details={"index": idx, "type": type(item).__name__},
                )
            op_name = item.get("op")
            args = item.get("args") or {}
            if not isinstance(op_name, str) or not op_name:
                raise BadRequest(
                    f"raw_batch_ops[{idx}].op must be a non-empty string",
                    code="BAD_BATCH_OP",
                    details={"index": idx},
                )
            if not isinstance(args, Mapping):
                raise BadRequest(
                    f"raw_batch_ops[{idx}].args must be a mapping",
                    code="BAD_BATCH_OP",
                    details={"index": idx, "type": type(args).__name__},
                )
            ops.append(BatchOperation(op=str(op_name), args=dict(args)))
        return ops

    def translate_batch_result(
        self,
        result: BatchResult,
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Any:
        return list(result.results or [])

    def translate_schema(
        self,
        schema: GraphSchema,
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Any:
        return schema

    # ---- capabilities / health ----

    def translate_capabilities(
        self,
        capabilities: GraphCapabilities,
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Any:
        # Simple default: expose as plain dict
        return asdict(capabilities)

    def translate_health(
        self,
        health: Mapping[str, Any],
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Any:
        # Simple default: normalize to dict
        return dict(health or {})


# =============================================================================
# Graph Translator Orchestrator
# =============================================================================


class GraphTranslator:
    """
    Framework-agnostic orchestrator for graph operations.

    This class:
        - Accepts framework-level inputs and a normalized OperationContext
        - Delegates to a GraphFrameworkTranslator to build specs and translate results
        - Calls into a GraphProtocolV1 adapter to execute operations
        - Provides sync + async variants for all core operations
        - Handles streaming via SyncStreamBridge for sync callers
        - Optionally applies MMR re-ranking before returning results
        - Attaches rich error context for diagnostics
    """

    def __init__(
        self,
        *,
        adapter: GraphProtocolV1,
        framework: str = "generic",
        translator: Optional[GraphFrameworkTranslator] = None,
    ) -> None:
        self._adapter = adapter
        self._framework = framework
        self._translator: GraphFrameworkTranslator = (
            translator or DefaultGraphFrameworkTranslator()
        )

    # --------------------------------------------------------------------- #
    # Internal MMR helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))

        if na <= 0.0 or nb <= 0.0:
            return 0.0
        return float(dot / (na * nb))

    def _apply_mmr_to_query_result(
        self,
        result: QueryResult,
        mmr_config: Optional[MMRConfig],
    ) -> QueryResult:
        """
        Apply Maximal Marginal Relevance re-ranking to a QueryResult.
        """
        if mmr_config is None or not mmr_config.enabled:
            return result

        records = list(result.records or [])
        n = len(records)
        if n <= 1:
            return result

        scores: List[float] = []
        vectors: List[List[float]] = []

        for idx, rec in enumerate(records):
            if not isinstance(rec, Mapping):
                LOG.debug("MMR skipped: record[%d] is not a mapping", idx)
                return result
            if mmr_config.score_key not in rec or mmr_config.vector_key not in rec:
                LOG.debug(
                    "MMR skipped: record[%d] missing score or vector key (%s/%s)",
                    idx,
                    mmr_config.score_key,
                    mmr_config.vector_key,
                )
                return result
            try:
                scores.append(float(rec[mmr_config.score_key]))
            except (TypeError, ValueError):
                LOG.debug("MMR skipped: record[%d] score is not numeric", idx)
                return result

            vec_raw = rec[mmr_config.vector_key]
            if not isinstance(vec_raw, (list, tuple)):
                LOG.debug("MMR skipped: record[%d] vector is not a sequence", idx)
                return result
            try:
                vec = [float(v) for v in vec_raw]
            except (TypeError, ValueError):
                LOG.debug("MMR skipped: record[%d] vector elements not numeric", idx)
                return result
            vectors.append(vec)

        dim = len(vectors[0])
        if dim == 0 or any(len(v) != dim for v in vectors):
            LOG.debug("MMR skipped: inconsistent or zero-length vector dimensions")
            return result

        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score

        if score_range <= 0.0:
            normalized_scores = [1.0 for _ in scores]
        else:
            normalized_scores = [(s - min_score) / score_range for s in scores]

        sim_fn: SimilarityFn = mmr_config.similarity_fn or self._cosine_similarity
        k = mmr_config.effective_k(n)
        lambda_mult = float(mmr_config.lambda_mult)

        selected_indices: List[int] = []
        candidates: List[int] = list(range(n))

        similarity_cache: Dict[tuple[int, int], float] = {}

        def get_similarity(i: int, j: int) -> float:
            key = (min(i, j), max(i, j))
            if key not in similarity_cache:
                similarity_cache[key] = sim_fn(vectors[i], vectors[j])
            return similarity_cache[key]

        if candidates:
            best_first = max(candidates, key=lambda idx: normalized_scores[idx])
            selected_indices.append(best_first)
            candidates.remove(best_first)

        while candidates and len(selected_indices) < k:
            best_idx: Optional[int] = None
            best_mmr_score = -float("inf")

            for idx in candidates:
                relevance = normalized_scores[idx]
                diversity_penalty = 0.0
                if selected_indices:
                    diversity_penalty = max(
                        get_similarity(idx, j) for j in selected_indices
                    )
                mmr_score = lambda_mult * relevance - (1.0 - lambda_mult) * diversity_penalty
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = idx

            if best_idx is None:
                break

            selected_indices.append(best_idx)
            candidates.remove(best_idx)

        reordered: List[Record] = [records[i] for i in selected_indices]
        if k < n:
            remaining = [records[i] for i in range(n) if i not in selected_indices]
            reordered.extend(remaining)

        LOG.debug(
            "MMR applied: %d records → %d selected with lambda=%.2f",
            n,
            len(selected_indices),
            lambda_mult,
        )

        return QueryResult(
            records=reordered,
            summary=result.summary,
            dialect=result.dialect,
            namespace=result.namespace,
        )

    # --------------------------------------------------------------------- #
    # Internal executor helpers
    # --------------------------------------------------------------------- #

    def _run_operation(
        self,
        *,
        op_name: str,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]],
        sync: bool,
        logic: Callable[[OperationContext], Awaitable[R]],
    ) -> Union[R, Awaitable[R]]:
        """Centralized execution for non-streaming operations."""
        ctx = _ensure_operation_context(op_ctx)

        async def _wrapped() -> R:
            try:
                return await logic(ctx)
            except Exception as exc:
                attach_context(
                    exc,
                    framework=self._framework,
                    operation=f"graph.{op_name}",
                    request_id=ctx.request_id,
                    tenant=ctx.tenant,
                )
                raise

        if sync:
            timeout = ctx.deadline_ms / 1000.0 if ctx.deadline_ms else None
            return AsyncBridge.run_async(_wrapped(), timeout=timeout)
        return _wrapped()

    def _run_stream_operation(
        self,
        *,
        op_name: str,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]],
        sync: bool,
        factory: Callable[[OperationContext], AsyncIterator[R]],
    ) -> Union[Iterator[R], AsyncIterator[R]]:
        """
        Centralized execution for streaming operations.

        Args:
            factory: A function that takes context and returns an AsyncIterator.
        """
        ctx = _ensure_operation_context(op_ctx)

        async def _async_gen() -> AsyncIterator[R]:
            # Call factory outside the iteration try/except so call-time failures can be handled
            # distinctly. We still attach error context for call-time failures to preserve
            # observability and conformance expectations for "early" errors.
            #
            # Factory returns async iterator directly (no await).
            try:
                aiter = factory(ctx)
            except Exception as exc:
                attach_context(
                    exc,
                    framework=self._framework,
                    operation=f"graph.{op_name}",
                    request_id=ctx.request_id,
                    tenant=ctx.tenant,
                )
                raise

            try:
                async for chunk in aiter:
                    yield chunk
            except Exception as exc:
                attach_context(
                    exc,
                    framework=self._framework,
                    operation=f"graph.{op_name}",
                    request_id=ctx.request_id,
                    tenant=ctx.tenant,
                )
                raise

        if sync:
            # SyncStreamBridge expects a coro_factory that returns an *awaitable* which resolves
            # to an AsyncIterator. An async generator function (like _async_gen) returns an
            # AsyncIterator directly and is not awaitable; therefore we wrap it in a small
            # coroutine that returns the async iterator. This preserves cancellation and error
            # propagation semantics in SyncStreamBridge without changing adapter behavior.
            async def _coro_factory() -> AsyncIterator[R]:
                return _async_gen()

            return SyncStreamBridge(
                coro_factory=_coro_factory,
                framework=self._framework,
                error_context={
                    "operation": f"graph.{op_name}",
                    "request_id": ctx.request_id,
                    "tenant": ctx.tenant,
                },
            ).run()

        return _async_gen()

    # Helper to build a query-stream factory
    def _prepare_query_stream_factory(
        self,
        raw_query: Any,
        framework_ctx: Any,
    ) -> Callable[[OperationContext], AsyncIterator[Any]]:

        async def _factory(ctx: OperationContext) -> AsyncIterator[Any]:
            spec = self._translator.build_query_spec(
                raw_query,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
                stream=True,
            )
            async for chunk in self._adapter.stream_query(spec, ctx=ctx):
                yield self._translator.translate_query_chunk(
                    chunk,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )

        return _factory

    # --------------------------------------------------------------------- #
    # Capabilities / Health APIs
    # --------------------------------------------------------------------- #

    def capabilities(
        self,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """
        Synchronous capabilities API.

        Wraps adapter.capabilities() with error-context and translation.
        """

        async def _logic(ctx: OperationContext) -> Any:
            caps = await self._adapter.capabilities()

            if not isinstance(caps, GraphCapabilities):
                raise BadRequest(
                    f"adapter.capabilities returned unsupported type: "
                    f"{type(caps).__name__}",
                    code="BAD_ADAPTER_RESULT",
                )

            return self._translator.translate_capabilities(
                caps,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

        return self._run_operation(
            op_name="capabilities",
            op_ctx=op_ctx,
            sync=True,
            logic=_logic,
        )

    async def arun_capabilities(
        self,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """
        Async capabilities API.
        """

        async def _logic(ctx: OperationContext) -> Any:
            caps = await self._adapter.capabilities()

            if not isinstance(caps, GraphCapabilities):
                raise BadRequest(
                    f"adapter.capabilities returned unsupported type: "
                    f"{type(caps).__name__}",
                    code="BAD_ADAPTER_RESULT",
                )

            return self._translator.translate_capabilities(
                caps,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

        return await self._run_operation(
            op_name="capabilities",
            op_ctx=op_ctx,
            sync=False,
            logic=_logic,
        )

    def health(
        self,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """
        Synchronous health API.

        Wraps adapter.health(ctx=...) with error-context and translation.
        """

        async def _logic(ctx: OperationContext) -> Any:
            res = await self._adapter.health(ctx=ctx)

            if not isinstance(res, Mapping):
                raise BadRequest(
                    f"adapter.health returned unsupported type: {type(res).__name__}",
                    code="BAD_ADAPTER_RESULT",
                )

            return self._translator.translate_health(
                res,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

        return self._run_operation(
            op_name="health",
            op_ctx=op_ctx,
            sync=True,
            logic=_logic,
        )

    async def arun_health(
        self,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """
        Async health API.
        """

        async def _logic(ctx: OperationContext) -> Any:
            res = await self._adapter.health(ctx=ctx)

            if not isinstance(res, Mapping):
                raise BadRequest(
                    f"adapter.health returned unsupported type: {type(res).__name__}",
                    code="BAD_ADAPTER_RESULT",
                )

            return self._translator.translate_health(
                res,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

        return await self._run_operation(
            op_name="health",
            op_ctx=op_ctx,
            sync=False,
            logic=_logic,
        )

    # --------------------------------------------------------------------- #
    # Sync Query APIs
    # --------------------------------------------------------------------- #

    def query(
        self,
        raw_query: Any,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
        mmr_config: Optional[MMRConfig] = None,
    ) -> Any:
        """Synchronous query API."""

        async def _logic(ctx: OperationContext) -> Any:
            spec = self._translator.build_query_spec(
                raw_query,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
                stream=False,
            )
            result = await self._adapter.query(spec, ctx=ctx)

            if not isinstance(result, QueryResult):
                raise BadRequest(
                    f"adapter.query returned unsupported type: {type(result).__name__}",
                    code="BAD_ADAPTER_RESULT",
                )

            result_mmr = self._apply_mmr_to_query_result(result, mmr_config=mmr_config)
            return self._translator.translate_query_result(
                result_mmr,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

        return self._run_operation(
            op_name="query",
            op_ctx=op_ctx,
            sync=True,
            logic=_logic,
        )

    # --------------------------------------------------------------------- #
    # Async Query APIs
    # --------------------------------------------------------------------- #

    async def arun_query(
        self,
        raw_query: Any,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
        mmr_config: Optional[MMRConfig] = None,
    ) -> Any:
        """Async query API."""

        async def _logic(ctx: OperationContext) -> Any:
            spec = self._translator.build_query_spec(
                raw_query,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
                stream=False,
            )
            result = await self._adapter.query(spec, ctx=ctx)

            if not isinstance(result, QueryResult):
                raise BadRequest(
                    f"adapter.query returned unsupported type: {type(result).__name__}",
                    code="BAD_ADAPTER_RESULT",
                )

            result_mmr = self._apply_mmr_to_query_result(result, mmr_config=mmr_config)
            return self._translator.translate_query_result(
                result_mmr,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

        return await self._run_operation(
            op_name="query",
            op_ctx=op_ctx,
            sync=False,
            logic=_logic,
        )

    # --------------------------------------------------------------------- #
    # Transaction APIs
    # --------------------------------------------------------------------- #

    def transaction(
        self,
        raw_transaction: Any,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """Synchronous transaction API."""

        async def _logic(ctx: OperationContext) -> Any:
            ops = self._translator.build_transaction_spec(
                raw_transaction,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            result = await self._adapter.transaction(ops, ctx=ctx)

            if not isinstance(result, BatchResult):
                raise BadRequest(
                    f"adapter.transaction returned unsupported type: {type(result).__name__}",
                    code="BAD_ADAPTER_RESULT",
                )

            return self._translator.translate_transaction_result(
                result,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

        return self._run_operation(
            op_name="transaction",
            op_ctx=op_ctx,
            sync=True,
            logic=_logic,
        )

    async def arun_transaction(
        self,
        raw_transaction: Any,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """Async transaction API."""

        async def _logic(ctx: OperationContext) -> Any:
            ops = self._translator.build_transaction_spec(
                raw_transaction,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            result = await self._adapter.transaction(ops, ctx=ctx)

            if not isinstance(result, BatchResult):
                raise BadRequest(
                    f"adapter.transaction returned unsupported type: {type(result).__name__}",
                    code="BAD_ADAPTER_RESULT",
                )

            return self._translator.translate_transaction_result(
                result,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

        return await self._run_operation(
            op_name="transaction",
            op_ctx=op_ctx,
            sync=False,
            logic=_logic,
        )

    # --------------------------------------------------------------------- #
    # Traversal APIs
    # --------------------------------------------------------------------- #

    def traversal(
        self,
        raw_traversal: Any,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """Synchronous traversal API."""

        async def _logic(ctx: OperationContext) -> Any:
            spec = self._translator.build_traversal_spec(
                raw_traversal,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            result = await self._adapter.traversal(spec, ctx=ctx)

            if not isinstance(result, TraversalResult):
                raise BadRequest(
                    f"adapter.traversal returned unsupported type: {type(result).__name__}",
                    code="BAD_ADAPTER_RESULT",
                )

            return self._translator.translate_traversal_result(
                result,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

        return self._run_operation(
            op_name="traversal",
            op_ctx=op_ctx,
            sync=True,
            logic=_logic,
        )

    async def arun_traversal(
        self,
        raw_traversal: Any,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """Async traversal API."""

        async def _logic(ctx: OperationContext) -> Any:
            spec = self._translator.build_traversal_spec(
                raw_traversal,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            result = await self._adapter.traversal(spec, ctx=ctx)

            if not isinstance(result, TraversalResult):
                raise BadRequest(
                    f"adapter.traversal returned unsupported type: {type(result).__name__}",
                    code="BAD_ADAPTER_RESULT",
                )

            return self._translator.translate_traversal_result(
                result,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

        return await self._run_operation(
            op_name="traversal",
            op_ctx=op_ctx,
            sync=False,
            logic=_logic,
        )

    # --------------------------------------------------------------------- #
    # Streaming Query APIs
    # --------------------------------------------------------------------- #

    def query_stream(
        self,
        raw_query: Any,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Iterator[Any]:
        """Synchronous streaming query API."""
        return self._run_stream_operation(
            op_name="query_stream",
            op_ctx=op_ctx,
            sync=True,
            factory=self._prepare_query_stream_factory(raw_query, framework_ctx),
        )

    def arun_query_stream(
        self,
        raw_query: Any,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> AsyncIterator[Any]:
        """
        Async streaming query API.
        Returns an AsyncIterator yielding framework-level chunks.
        """
        # Note: This returns the AsyncIterator directly (no await)
        return self._run_stream_operation(
            op_name="query_stream",
            op_ctx=op_ctx,
            sync=False,
            factory=self._prepare_query_stream_factory(raw_query, framework_ctx),
        )

    # --------------------------------------------------------------------- #
    # Mutation APIs
    # --------------------------------------------------------------------- #

    def upsert_nodes(
        self,
        raw_nodes: Any,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """Synchronous upsert_nodes API."""

        async def _logic(ctx: OperationContext) -> Any:
            spec = self._translator.build_upsert_nodes_spec(
                raw_nodes,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            return await self._adapter.upsert_nodes(spec, ctx=ctx)

        return self._run_operation(
            op_name="upsert_nodes",
            op_ctx=op_ctx,
            sync=True,
            logic=_logic,
        )

    async def arun_upsert_nodes(
        self,
        raw_nodes: Any,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """Async upsert_nodes API."""

        async def _logic(ctx: OperationContext) -> Any:
            spec = self._translator.build_upsert_nodes_spec(
                raw_nodes,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            return await self._adapter.upsert_nodes(spec, ctx=ctx)

        return await self._run_operation(
            op_name="upsert_nodes",
            op_ctx=op_ctx,
            sync=False,
            logic=_logic,
        )

    def upsert_edges(
        self,
        raw_edges: Any,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """Synchronous upsert_edges API."""

        async def _logic(ctx: OperationContext) -> Any:
            spec = self._translator.build_upsert_edges_spec(
                raw_edges,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            return await self._adapter.upsert_edges(spec, ctx=ctx)

        return self._run_operation(
            op_name="upsert_edges",
            op_ctx=op_ctx,
            sync=True,
            logic=_logic,
        )

    async def arun_upsert_edges(
        self,
        raw_edges: Any,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """Async upsert_edges API."""

        async def _logic(ctx: OperationContext) -> Any:
            spec = self._translator.build_upsert_edges_spec(
                raw_edges,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            return await self._adapter.upsert_edges(spec, ctx=ctx)

        return await self._run_operation(
            op_name="upsert_edges",
            op_ctx=op_ctx,
            sync=False,
            logic=_logic,
        )

    def delete_nodes(
        self,
        raw_filter_or_ids: Any,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """Synchronous delete_nodes API."""

        async def _logic(ctx: OperationContext) -> Any:
            spec = self._translator.build_delete_nodes_spec(
                raw_filter_or_ids,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            return await self._adapter.delete_nodes(spec, ctx=ctx)

        return self._run_operation(
            op_name="delete_nodes",
            op_ctx=op_ctx,
            sync=True,
            logic=_logic,
        )

    async def arun_delete_nodes(
        self,
        raw_filter_or_ids: Any,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """Async delete_nodes API."""

        async def _logic(ctx: OperationContext) -> Any:
            spec = self._translator.build_delete_nodes_spec(
                raw_filter_or_ids,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            return await self._adapter.delete_nodes(spec, ctx=ctx)

        return await self._run_operation(
            op_name="delete_nodes",
            op_ctx=op_ctx,
            sync=False,
            logic=_logic,
        )

    def delete_edges(
        self,
        raw_filter_or_ids: Any,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """Synchronous delete_edges API."""

        async def _logic(ctx: OperationContext) -> Any:
            spec = self._translator.build_delete_edges_spec(
                raw_filter_or_ids,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            return await self._adapter.delete_edges(spec, ctx=ctx)

        return self._run_operation(
            op_name="delete_edges",
            op_ctx=op_ctx,
            sync=True,
            logic=_logic,
        )

    async def arun_delete_edges(
        self,
        raw_filter_or_ids: Any,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """Async delete_edges API."""

        async def _logic(ctx: OperationContext) -> Any:
            spec = self._translator.build_delete_edges_spec(
                raw_filter_or_ids,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            return await self._adapter.delete_edges(spec, ctx=ctx)

        return await self._run_operation(
            op_name="delete_edges",
            op_ctx=op_ctx,
            sync=False,
            logic=_logic,
        )

    # --------------------------------------------------------------------- #
    # Bulk vertices / batch / schema
    # --------------------------------------------------------------------- #

    def bulk_vertices(
        self,
        raw_request: Any,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """Synchronous bulk_vertices API."""

        async def _logic(ctx: OperationContext) -> Any:
            spec = self._translator.build_bulk_vertices_spec(
                raw_request,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            result = await self._adapter.bulk_vertices(spec, ctx=ctx)

            if not isinstance(result, BulkVerticesResult):
                raise BadRequest(
                    f"adapter.bulk_vertices returned unsupported type: {type(result).__name__}",
                    code="BAD_ADAPTER_RESULT",
                )

            return self._translator.translate_bulk_vertices_result(
                result,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

        return self._run_operation(
            op_name="bulk_vertices",
            op_ctx=op_ctx,
            sync=True,
            logic=_logic,
        )

    async def arun_bulk_vertices(
        self,
        raw_request: Any,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """Async bulk_vertices API."""

        async def _logic(ctx: OperationContext) -> Any:
            spec = self._translator.build_bulk_vertices_spec(
                raw_request,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            result = await self._adapter.bulk_vertices(spec, ctx=ctx)

            if not isinstance(result, BulkVerticesResult):
                raise BadRequest(
                    f"adapter.bulk_vertices returned unsupported type: {type(result).__name__}",
                    code="BAD_ADAPTER_RESULT",
                )

            return self._translator.translate_bulk_vertices_result(
                result,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

        return await self._run_operation(
            op_name="bulk_vertices",
            op_ctx=op_ctx,
            sync=False,
            logic=_logic,
        )

    def batch(
        self,
        raw_batch_ops: Any,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """Synchronous batch API."""

        async def _logic(ctx: OperationContext) -> Any:
            ops = self._translator.build_batch_ops(
                raw_batch_ops,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            result = await self._adapter.batch(ops, ctx=ctx)

            if not isinstance(result, BatchResult):
                raise BadRequest(
                    f"adapter.batch returned unsupported type: {type(result).__name__}",
                    code="BAD_ADAPTER_RESULT",
                )

            return self._translator.translate_batch_result(
                result,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

        return self._run_operation(
            op_name="batch",
            op_ctx=op_ctx,
            sync=True,
            logic=_logic,
        )

    async def arun_batch(
        self,
        raw_batch_ops: Any,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """Async batch API."""

        async def _logic(ctx: OperationContext) -> Any:
            ops = self._translator.build_batch_ops(
                raw_batch_ops,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            result = await self._adapter.batch(ops, ctx=ctx)

            if not isinstance(result, BatchResult):
                raise BadRequest(
                    f"adapter.batch returned unsupported type: {type(result).__name__}",
                    code="BAD_ADAPTER_RESULT",
                )

            return self._translator.translate_batch_result(
                result,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

        return await self._run_operation(
            op_name="batch",
            op_ctx=op_ctx,
            sync=False,
            logic=_logic,
        )

    def get_schema(
        self,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """Synchronous get_schema API."""

        async def _logic(ctx: OperationContext) -> Any:
            schema = await self._adapter.get_schema(ctx=ctx)

            if not isinstance(schema, GraphSchema):
                raise BadRequest(
                    f"adapter.get_schema returned unsupported type: {type(schema).__name__}",
                    code="BAD_ADAPTER_RESULT",
                )

            return self._translator.translate_schema(
                schema,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

        return self._run_operation(
            op_name="get_schema",
            op_ctx=op_ctx,
            sync=True,
            logic=_logic,
        )

    async def arun_get_schema(
        self,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """Async get_schema API."""

        async def _logic(ctx: OperationContext) -> Any:
            schema = await self._adapter.get_schema(ctx=ctx)

            if not isinstance(schema, GraphSchema):
                raise BadRequest(
                    f"adapter.get_schema returned unsupported type: {type(schema).__name__}",
                    code="BAD_ADAPTER_RESULT",
                )

            return self._translator.translate_schema(
                schema,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

        return await self._run_operation(
            op_name="get_schema",
            op_ctx=op_ctx,
            sync=False,
            logic=_logic,
        )


# =============================================================================
# Registry for per-framework translators
# =============================================================================


_TranslatorFactory = Callable[[GraphProtocolV1], GraphFrameworkTranslator]
_GRAPH_TRANSLATOR_FACTORIES: Dict[str, _TranslatorFactory] = {}
_REGISTRY_LOCK = threading.Lock()


def register_graph_translator(
    framework: str,
    factory: _TranslatorFactory,
) -> None:
    """
    Register or override a GraphFrameworkTranslator factory for a given framework.
    """
    if not framework or not isinstance(framework, str):
        raise BadRequest(
            "framework name must be a non-empty string",
            code="BAD_TRANSLATOR_REGISTRATION",
        )
    if not callable(factory):
        raise BadRequest(
            "translator factory must be callable",
            code="BAD_TRANSLATOR_REGISTRATION",
        )
    with _REGISTRY_LOCK:
        _GRAPH_TRANSLATOR_FACTORIES[framework] = factory
    LOG.debug("Registered graph translator factory for framework=%s", framework)


def get_graph_translator_factory(framework: str) -> Optional[_TranslatorFactory]:
    """Return a previously registered translator factory for a framework, if any."""
    with _REGISTRY_LOCK:
        return _GRAPH_TRANSLATOR_FACTORIES.get(framework)


def create_graph_translator(
    *,
    adapter: GraphProtocolV1,
    framework: str = "generic",
    translator: Optional[GraphFrameworkTranslator] = None,
) -> GraphTranslator:
    """
    Convenience helper to construct a GraphTranslator for a given framework.

    Behavior:
        - If `translator` is provided explicitly, it is used as-is.
        - Else, if a factory is registered for `framework`, it is used.
        - Else, DefaultGraphFrameworkTranslator is used.
    """
    if translator is None:
        factory = get_graph_translator_factory(framework)
        if factory is not None:
            translator = factory(adapter)
        else:
            translator = DefaultGraphFrameworkTranslator()
    return GraphTranslator(adapter=adapter, framework=framework, translator=translator)


__all__ = [
    "MMRConfig",
    "SimilarityFn",
    "FilterTranslator",
    "GraphFrameworkTranslator",
    "DefaultGraphFrameworkTranslator",
    "GraphTranslator",
    "register_graph_translator",
    "get_graph_translator_factory",
    "create_graph_translator",
]
