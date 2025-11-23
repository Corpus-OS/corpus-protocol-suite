# corpus_sdk/graph/framework_adapters/common/framework_utils.py
# SPDX-License-Identifier: Apache-2.0
"""
Shared utilities for framework-specific *graph* adapters.

This module centralizes common logic used across all graph adapters:

- Normalizing provider / runtime results into canonical graph shapes:
  * Nodes → List[GraphNode-like Mapping[str, Any]]
  * Edges → List[GraphEdge-like Mapping[str, Any]]
- Enforcing graph-level resource limits (node/edge counts, depth, fan-out)
- Emitting consistent, framework-aware warnings
- Best-effort depth estimation and cycle detection
- Basic state-size estimation for safety
- Normalizing graph execution context and attaching it to framework_ctx
- Optional streaming helpers for graph event streams

It intentionally stays *framework-neutral* and uses only:

- Standard library types
- Simple, caller-provided error codes and limits
- No direct dependencies on specific graph runtimes

Adapters remain responsible for:

- Choosing framework names and passing them in
- Supplying appropriate error code bundles
- Deciding which limits/flags to use per environment
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error codes, limits, validation flags
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphCoercionErrorCodes:
    """
    Structured bundle of error codes used during graph coercion / validation.

    These codes are surfaced in exception messages so individual frameworks
    can attach or filter on them in higher-level error handlers.

    Attributes
    ----------
    invalid_node_result:
        Code used when the node container or node rows are invalid.

    invalid_edge_result:
        Code used when the edge container or edge rows are invalid.

    empty_graph_result:
        Code used when no valid nodes remain after processing.

    conversion_error:
        Code used when value conversion fails (reserved for future numeric
        conversions; currently used for structural consistency).

    graph_depth_exceeded:
        Code used when a graph's depth exceeds configured limits.

    graph_fanout_exceeded:
        Code used when fan-out or per-run counts exceed configured limits.

    graph_cycle_detected:
        Code used when cycles are detected in graphs that must be acyclic.

    state_size_exceeded:
        Code used when per-node state exceeds configured size limits.

    framework_label:
        Default framework label used when no explicit framework name is passed.
    """

    invalid_node_result: str = "INVALID_GRAPH_NODE_RESULT"
    invalid_edge_result: str = "INVALID_GRAPH_EDGE_RESULT"
    empty_graph_result: str = "EMPTY_GRAPH_RESULT"
    conversion_error: str = "GRAPH_CONVERSION_ERROR"
    graph_depth_exceeded: str = "GRAPH_DEPTH_EXCEEDED"
    graph_fanout_exceeded: str = "GRAPH_FANOUT_EXCEEDED"
    graph_cycle_detected: str = "GRAPH_CYCLE_DETECTED"
    state_size_exceeded: str = "GRAPH_STATE_SIZE_EXCEEDED"
    framework_label: str = "graph"


@dataclass(frozen=True)
class GraphResourceLimits:
    """
    Resource limits for graph size and complexity.

    All fields are optional; if None, the limit is effectively disabled.

    Attributes
    ----------
    max_nodes_per_step:
        Maximum number of nodes returned in a single step.

    max_edges_per_step:
        Maximum number of edges returned in a single step.

    max_nodes_per_run:
        Maximum total number of nodes for a graph run.

    max_edges_per_run:
        Maximum total number of edges for a graph run.

    max_graph_depth:
        Maximum allowed depth of the graph (longest path length).

    max_fanout_per_node:
        Maximum number of outgoing edges from any single node.

    max_state_size_bytes:
        Maximum approximate size, in bytes, for per-node state blobs.

    max_paths_per_run:
        Optional limit on number of distinct logical paths (for path-heavy graphs).

    max_total_stream_events:
        Maximum number of events processed by streaming helpers.
    """

    max_nodes_per_step: Optional[int] = None
    max_edges_per_step: Optional[int] = None
    max_nodes_per_run: Optional[int] = None
    max_edges_per_run: Optional[int] = None
    max_graph_depth: Optional[int] = None
    max_fanout_per_node: Optional[int] = None
    max_state_size_bytes: Optional[int] = None
    max_paths_per_run: Optional[int] = None
    max_total_stream_events: Optional[int] = 10_000


@dataclass(frozen=True)
class GraphValidationFlags:
    """
    Flags controlling how strict graph validation should be.

    These can be tuned per-adapter or per-environment (dev vs prod).
    """

    validate_ids: bool = True
    validate_state_size: bool = True
    validate_acyclic: bool = False
    warn_on_excess_fanout: bool = True
    fail_on_limit_exceeded: bool = False
    validate_error_codes: bool = True
    validate_framework_name: bool = True
    # Streaming behavior
    strict_stream_limits: bool = False


# ---------------------------------------------------------------------------
# Internal helpers: framework + error code validation
# ---------------------------------------------------------------------------


def _infer_framework_name(
    error_codes: Optional[GraphCoercionErrorCodes],
    framework: Optional[str],
    *,
    source: Optional[Any] = None,
    flags: Optional[GraphValidationFlags] = None,
) -> str:
    """
    Normalize or infer a framework name for logging / diagnostics.

    Priority chain:
    1. Explicit `framework` argument (if valid).
    2. `error_codes.framework_label` (if present and non-empty).
    3. Class-name inference from `source` (e.g. adapter instance).
    4. Fallback to "graph".

    If validation is enabled and the explicit framework is invalid, raises
    ValueError. Everything else is best-effort.
    """
    flags = flags or GraphValidationFlags()

    # 1) Explicit framework
    if framework is not None:
        value = str(framework).strip()
        if not value:
            if flags.validate_framework_name:
                raise ValueError("framework must be a non-empty string when provided")
            return "graph"
        return value.lower()

    # 2) error_codes.framework_label
    if error_codes is not None:
        label = getattr(error_codes, "framework_label", None)
        if isinstance(label, str) and label.strip():
            return label.strip().lower()

    # 3) Class-name inference from source
    if source is not None:
        try:
            cls = getattr(source, "__class__", type(source))
            name = getattr(cls, "__name__", "") or ""
            if name:
                return name.lower()
        except Exception:  # noqa: BLE001
            # Best-effort only; ignore failures.
            pass

    # 4) Fallback
    return "graph"


def _validate_error_codes(
    error_codes: GraphCoercionErrorCodes,
    *,
    logger: Optional[logging.Logger] = None,
    flags: Optional[GraphValidationFlags] = None,
) -> None:
    """
    Ensure the provided error code bundle looks structurally valid.

    This is intentionally lightweight and only checks for non-empty strings.
    """
    flags = flags or GraphValidationFlags()
    log = logger or LOG

    if not flags.validate_error_codes:
        return

    required_fields = (
        "invalid_node_result",
        "invalid_edge_result",
        "empty_graph_result",
        "conversion_error",
        "graph_depth_exceeded",
        "graph_fanout_exceeded",
        "graph_cycle_detected",
        "state_size_exceeded",
    )
    missing: List[str] = []
    empty: List[str] = []

    for field in required_fields:
        if not hasattr(error_codes, field):
            missing.append(field)
            continue
        value = getattr(error_codes, field)
        if not isinstance(value, str) or not value.strip():
            empty.append(field)

    if missing or empty:
        message = (
            f"GraphCoercionErrorCodes is missing fields={missing} "
            f"or has empty fields={empty}"
        )
        if flags.fail_on_limit_exceeded:
            raise TypeError(message)
        log.warning("Invalid error_codes configuration: %s", message)


# ---------------------------------------------------------------------------
# Extraction helpers for node / edge containers
# ---------------------------------------------------------------------------


def _extract_nodes_object(result: Any) -> Any:
    """
    Extract the underlying nodes object from a variety of result shapes.

    Supported shapes:
    - Mapping with "nodes": {"nodes": [...], ...}
    - Object with `.nodes` attribute
    - Raw list / sequence of node-like objects: [{...}, {...}]
    """
    if isinstance(result, Mapping) and "nodes" in result:
        return result["nodes"]
    if hasattr(result, "nodes"):
        return getattr(result, "nodes")
    return result


def _extract_edges_object(result: Any) -> Any:
    """
    Extract the underlying edges object from a variety of result shapes.

    Supported shapes:
    - Mapping with "edges": {"edges": [...], ...}
    - Object with `.edges` attribute
    - Raw list / sequence of edge-like objects: [{...}, {...}]
    """
    if isinstance(result, Mapping) and "edges" in result:
        return result["edges"]
    if hasattr(result, "edges"):
        return getattr(result, "edges")
    return result


def _as_mapping(obj: Any) -> Optional[Mapping[str, Any]]:
    """
    Best-effort conversion of an arbitrary object to a Mapping[str, Any].

    - If already a Mapping → returned as-is
    - If has __dict__ → return that dict
    - Otherwise, returns None and the caller decides how to react
    """
    if isinstance(obj, Mapping):
        return obj
    if hasattr(obj, "__dict__"):
        try:
            return dict(vars(obj))
        except Exception:  # noqa: BLE001
            return None
    return None


# ---------------------------------------------------------------------------
# Graph coercion helpers
# ---------------------------------------------------------------------------


def coerce_nodes(
    result: Any,
    *,
    framework: Optional[str],
    error_codes: GraphCoercionErrorCodes,
    limits: Optional[GraphResourceLimits] = None,
    flags: Optional[GraphValidationFlags] = None,
    logger: Optional[logging.Logger] = None,
) -> List[Mapping[str, Any]]:
    """
    Coerce a generic graph result into a canonical list of node mappings.

    Parameters
    ----------
    result:
        Arbitrary result returned by a graph runtime / adapter.
    framework:
        Framework label for logging / diagnostics.
    error_codes:
        Bundle of error codes to embed in exception messages.
    limits:
        Optional resource limits; if provided, node limits may be enforced.
    flags:
        Optional validation flags controlling strictness.
    logger:
        Optional logger; if omitted, the module-level logger is used.

    Returns
    -------
    List[Mapping[str, Any]]
        Canonical node representations.

    Raises
    ------
    TypeError, ValueError
        If the result is not structurally valid and limits require failures.
    """
    flags = flags or GraphValidationFlags()
    log = logger or LOG
    framework_name = _infer_framework_name(error_codes, framework, flags=flags)
    _validate_error_codes(error_codes, logger=log, flags=flags)

    nodes_obj = _extract_nodes_object(result)

    if nodes_obj is None:
        raise TypeError(
            f"[{error_codes.invalid_node_result}] "
            f"{framework_name}: graph result does not contain nodes"
        )

    if isinstance(nodes_obj, (str, bytes)) or not isinstance(nodes_obj, Sequence):
        raise TypeError(
            f"[{error_codes.invalid_node_result}] "
            f"{framework_name}: nodes container must be a sequence, "
            f"got {type(nodes_obj).__name__}"
        )

    coerced: List[Mapping[str, Any]] = []

    for idx, raw_node in enumerate(nodes_obj):
        mapping = _as_mapping(raw_node)
        if mapping is None:
            message = (
                f"[{error_codes.invalid_node_result}] "
                f"{framework_name}: node at index {idx} is not mapping-like "
                f"(type={type(raw_node).__name__})"
            )
            if flags.fail_on_limit_exceeded:
                raise TypeError(message)
            log.warning(message)
            continue

        if flags.validate_ids:
            node_id = mapping.get("id")
            if not isinstance(node_id, str) or not node_id.strip():
                message = (
                    f"[{error_codes.invalid_node_result}] "
                    f"{framework_name}: node at index {idx} missing valid 'id'"
                )
                if flags.fail_on_limit_exceeded:
                    raise ValueError(message)
                log.warning(message)
                # We still keep the node but mark it; adapters may choose to drop it.
        coerced.append(mapping)

    if not coerced:
        raise ValueError(
            f"[{error_codes.empty_graph_result}] "
            f"{framework_name}: no valid nodes found in graph result"
        )

    log.debug(
        "%s: successfully coerced %d nodes (original_type=%s)",
        framework_name,
        len(coerced),
        type(nodes_obj).__name__,
    )

    # Step-level limits, if configured
    if limits and limits.max_nodes_per_step is not None:
        if len(coerced) > limits.max_nodes_per_step:
            message = (
                f"[{error_codes.graph_fanout_exceeded}] "
                f"{framework_name}: nodes_per_step={len(coerced)} exceeds "
                f"max_nodes_per_step={limits.max_nodes_per_step}"
            )
            if flags.fail_on_limit_exceeded:
                raise ValueError(message)
            log.warning(message)
            coerced = coerced[: limits.max_nodes_per_step]

    return coerced


def coerce_edges(
    result: Any,
    *,
    framework: Optional[str],
    error_codes: GraphCoercionErrorCodes,
    limits: Optional[GraphResourceLimits] = None,
    flags: Optional[GraphValidationFlags] = None,
    logger: Optional[logging.Logger] = None,
) -> List[Mapping[str, Any]]:
    """
    Coerce a generic graph result into a canonical list of edge mappings.

    Parameters mirror `coerce_nodes` but operate on edge containers.
    """
    flags = flags or GraphValidationFlags()
    log = logger or LOG
    framework_name = _infer_framework_name(error_codes, framework, flags=flags)
    _validate_error_codes(error_codes, logger=log, flags=flags)

    edges_obj = _extract_edges_object(result)

    if edges_obj is None:
        # Edges are allowed to be absent; treat as empty list.
        return []

    if isinstance(edges_obj, (str, bytes)) or not isinstance(edges_obj, Sequence):
        raise TypeError(
            f"[{error_codes.invalid_edge_result}] "
            f"{framework_name}: edges container must be a sequence, "
            f"got {type(edges_obj).__name__}"
        )

    coerced: List[Mapping[str, Any]] = []

    for idx, raw_edge in enumerate(edges_obj):
        mapping = _as_mapping(raw_edge)
        if mapping is None:
            message = (
                f"[{error_codes.invalid_edge_result}] "
                f"{framework_name}: edge at index {idx} is not mapping-like "
                f"(type={type(raw_edge).__name__})"
            )
            if flags.fail_on_limit_exceeded:
                raise TypeError(message)
            log.warning(message)
            continue

        if flags.validate_ids:
            from_id = mapping.get("from_id")
            to_id = mapping.get("to_id")
            if not isinstance(from_id, str) or not from_id.strip():
                message = (
                    f"[{error_codes.invalid_edge_result}] "
                    f"{framework_name}: edge at index {idx} missing valid 'from_id'"
                )
                if flags.fail_on_limit_exceeded:
                    raise ValueError(message)
                log.warning(message)
            if not isinstance(to_id, str) or not to_id.strip():
                message = (
                    f"[{error_codes.invalid_edge_result}] "
                    f"{framework_name}: edge at index {idx} missing valid 'to_id'"
                )
                if flags.fail_on_limit_exceeded:
                    raise ValueError(message)
                log.warning(message)

        coerced.append(mapping)

    # Step-level limits, if configured
    if limits and limits.max_edges_per_step is not None:
        if len(coerced) > limits.max_edges_per_step:
            message = (
                f"[{error_codes.graph_fanout_exceeded}] "
                f"{framework_name}: edges_per_step={len(coerced)} exceeds "
                f"max_edges_per_step={limits.max_edges_per_step}"
            )
            if flags.fail_on_limit_exceeded:
                raise ValueError(message)
            log.warning(message)
            coerced = coerced[: limits.max_edges_per_step]

    log.debug(
        "%s: successfully coerced %d edges (original_type=%s)",
        framework_name,
        len(coerced),
        type(edges_obj).__name__,
    )
    return coerced


def coerce_graph(
    result: Any,
    *,
    framework: Optional[str],
    error_codes: GraphCoercionErrorCodes,
    limits: Optional[GraphResourceLimits] = None,
    flags: Optional[GraphValidationFlags] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]]]:
    """
    Coerce a generic result into a `(nodes, edges)` tuple.

    This unified entry-point tries to extract nodes and edges in one shot.
    """
    flags = flags or GraphValidationFlags()
    log = logger or LOG

    nodes = coerce_nodes(
        result,
        framework=framework,
        error_codes=error_codes,
        limits=limits,
        flags=flags,
        logger=log,
    )
    edges = coerce_edges(
        result,
        framework=framework,
        error_codes=error_codes,
        limits=limits,
        flags=flags,
        logger=log,
    )

    warn_if_extreme_graph_size(
        nodes,
        edges,
        framework=framework,
        op_name="coerce_graph",
        limits=limits,
        logger=log,
    )

    return nodes, edges


# ---------------------------------------------------------------------------
# Graph size & shape validation
# ---------------------------------------------------------------------------


def warn_if_extreme_graph_size(
    nodes: Sequence[Any],
    edges: Sequence[Any],
    *,
    framework: Optional[str],
    op_name: str,
    limits: Optional[GraphResourceLimits] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Emit a soft warning if a graph is unusually large.

    This does *not* enforce any hard limits; it is purely diagnostic.
    """
    log = logger or LOG
    framework_name = (framework or "graph").lower()
    limits = limits or GraphResourceLimits()

    node_count = len(nodes)
    edge_count = len(edges)

    threshold_nodes = limits.max_nodes_per_run or 50_000
    threshold_edges = limits.max_edges_per_run or 100_000

    if node_count > threshold_nodes or edge_count > threshold_edges:
        log.warning(
            "%s: %s produced a large graph (nodes=%d, edges=%d, "
            "threshold_nodes=%d, threshold_edges=%d). Ensure your "
            "runtime can handle this safely.",
            framework_name,
            op_name,
            node_count,
            edge_count,
            threshold_nodes,
            threshold_edges,
        )


def enforce_graph_limits(
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    *,
    framework: Optional[str],
    op_name: str,
    error_codes: GraphCoercionErrorCodes,
    limits: Optional[GraphResourceLimits],
    flags: Optional[GraphValidationFlags] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]]]:
    """
    Enforce global graph limits on node/edge counts.

    Returns truncated lists when limits are exceeded (unless configured to fail).
    """
    flags = flags or GraphValidationFlags()
    log = logger or LOG
    framework_name = _infer_framework_name(error_codes, framework, flags=flags)

    if not limits:
        return list(nodes), list(edges)

    nodes_list = list(nodes)
    edges_list = list(edges)

    # Node limits
    if limits.max_nodes_per_run is not None and len(nodes_list) > limits.max_nodes_per_run:
        message = (
            f"[{error_codes.graph_fanout_exceeded}] "
            f"{framework_name}: {op_name} nodes_per_run={len(nodes_list)} "
            f"exceeds max_nodes_per_run={limits.max_nodes_per_run}"
        )
        if flags.fail_on_limit_exceeded:
            raise ValueError(message)
        log.warning(message)
        nodes_list = nodes_list[: limits.max_nodes_per_run]

    # Edge limits
    if limits.max_edges_per_run is not None and len(edges_list) > limits.max_edges_per_run:
        message = (
            f"[{error_codes.graph_fanout_exceeded}] "
            f"{framework_name}: {op_name} edges_per_run={len(edges_list)} "
            f"exceeds max_edges_per_run={limits.max_edges_per_run}"
        )
        if flags.fail_on_limit_exceeded:
            raise ValueError(message)
        log.warning(message)
        edges_list = edges_list[: limits.max_edges_per_run]

    return nodes_list, edges_list


# ---------------------------------------------------------------------------
# Graph depth / cycle helpers
# ---------------------------------------------------------------------------


def estimate_graph_depth(
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    *,
    framework: Optional[str],
    logger: Optional[logging.Logger] = None,
) -> int:
    """
    Best-effort estimate of the maximum path length (graph depth).

    This function intentionally ignores edge weights and simply treats
    the graph as directed from `from_id` → `to_id`.
    """
    log = logger or LOG
    framework_name = (framework or "graph").lower()

    # Build adjacency list
    adj: Dict[str, List[str]] = {}
    for edge in edges:
        from_id = edge.get("from_id")
        to_id = edge.get("to_id")
        if not isinstance(from_id, str) or not isinstance(to_id, str):
            continue
        adj.setdefault(from_id, []).append(to_id)

    visited: Dict[str, int] = {}  # node_id -> max depth seen
    max_depth = 0

    def dfs(node_id: str, depth: int, seen: set[str]) -> None:
        nonlocal max_depth
        if node_id in seen:
            return
        if node_id in visited and visited[node_id] >= depth:
            return
        visited[node_id] = depth
        max_depth = max(max_depth, depth)
        if node_id not in adj:
            return
        seen.add(node_id)
        for child in adj[node_id]:
            dfs(child, depth + 1, seen)
        seen.remove(node_id)

    # Start DFS from all nodes; in practice we might restrict to roots, but
    # this is best-effort and still bounded by graph size.
    for node in nodes:
        node_id = node.get("id")
        if not isinstance(node_id, str):
            continue
        dfs(node_id, 1, set())

    log.debug("%s: estimated graph depth=%d", framework_name, max_depth)
    return max_depth


def detect_cycles(
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    *,
    framework: Optional[str],
    error_codes: GraphCoercionErrorCodes,
    limits: Optional[GraphResourceLimits] = None,
    flags: Optional[GraphValidationFlags] = None,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """
    Best-effort cycle detection.

    Returns True if cycles are detected, False otherwise.

    Behavior on cycles:
    - If validate_acyclic AND fail_on_limit_exceeded → raises.
    - Otherwise → logs warning and returns True.
    """
    flags = flags or GraphValidationFlags()
    log = logger or LOG
    framework_name = _infer_framework_name(error_codes, framework, flags=flags)
    _validate_error_codes(error_codes, logger=log, flags=flags)

    # Build adjacency list
    adj: Dict[str, List[str]] = {}
    for edge in edges:
        from_id = edge.get("from_id")
        to_id = edge.get("to_id")
        if not isinstance(from_id, str) or not isinstance(to_id, str):
            continue
        adj.setdefault(from_id, []).append(to_id)

    visited: set[str] = set()
    stack: set[str] = set()

    def dfs(node_id: str) -> bool:
        if node_id in stack:
            return True
        if node_id in visited:
            return False
        visited.add(node_id)
        stack.add(node_id)
        for child in adj.get(node_id, ()):
            if dfs(child):
                return True
        stack.remove(node_id)
        return False

    # Run DFS from all nodes
    node_ids = [n.get("id") for n in nodes if isinstance(n.get("id"), str)]
    has_cycle = False
    for node_id in node_ids:
        if dfs(node_id):
            has_cycle = True
            break

    if has_cycle:
        message = (
            f"[{error_codes.graph_cycle_detected}] "
            f"{framework_name}: cycle detected in graph"
        )
        if flags.validate_acyclic and flags.fail_on_limit_exceeded:
            raise ValueError(message)
        log.warning(message)
        return True

    return False


# ---------------------------------------------------------------------------
# Node / edge ID and state helpers
# ---------------------------------------------------------------------------


def validate_node_ids(
    nodes: Sequence[Mapping[str, Any]],
    *,
    framework: Optional[str],
    error_codes: GraphCoercionErrorCodes,
    flags: Optional[GraphValidationFlags] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Validate that each node has a non-empty string 'id'.
    """
    flags = flags or GraphValidationFlags()
    log = logger or LOG
    framework_name = _infer_framework_name(error_codes, framework, flags=flags)
    _validate_error_codes(error_codes, logger=log, flags=flags)

    for idx, node in enumerate(nodes):
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id.strip():
            message = (
                f"[{error_codes.invalid_node_result}] "
                f"{framework_name}: node at index {idx} has invalid 'id'"
            )
            if flags.fail_on_limit_exceeded:
                raise ValueError(message)
            log.warning(message)


def validate_edge_ids(
    edges: Sequence[Mapping[str, Any]],
    *,
    framework: Optional[str],
    error_codes: GraphCoercionErrorCodes,
    flags: Optional[GraphValidationFlags] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Validate that each edge has non-empty string 'from_id' and 'to_id'.
    """
    flags = flags or GraphValidationFlags()
    log = logger or LOG
    framework_name = _infer_framework_name(error_codes, framework, flags=flags)
    _validate_error_codes(error_codes, logger=log, flags=flags)

    for idx, edge in enumerate(edges):
        from_id = edge.get("from_id")
        to_id = edge.get("to_id")
        if not isinstance(from_id, str) or not from_id.strip():
            message = (
                f"[{error_codes.invalid_edge_result}] "
                f"{framework_name}: edge at index {idx} has invalid 'from_id'"
            )
            if flags.fail_on_limit_exceeded:
                raise ValueError(message)
            log.warning(message)
        if not isinstance(to_id, str) or not to_id.strip():
            message = (
                f"[{error_codes.invalid_edge_result}] "
                f"{framework_name}: edge at index {idx} has invalid 'to_id'"
            )
            if flags.fail_on_limit_exceeded:
                raise ValueError(message)
            log.warning(message)


def estimate_state_size_bytes(
    state: Any,
    *,
    max_depth: int = 4,
) -> int:
    """
    Rough, shallow-ish estimate of state size in bytes.

    This is intentionally approximate and bounded by `max_depth` to avoid
    pathological recursion. It is suitable for limit enforcement, not billing.
    """
    visited_ids: set[int] = set()

    def _estimate(obj: Any, depth: int) -> int:
        if depth <= 0:
            return 0
        obj_id = id(obj)
        if obj_id in visited_ids:
            return 0
        visited_ids.add(obj_id)

        if obj is None:
            return 0
        if isinstance(obj, (bool, int, float)):
            return 32
        if isinstance(obj, (str, bytes)):
            return len(obj)

        if isinstance(obj, Mapping):
            size = 0
            for k, v in obj.items():
                size += _estimate(k, depth - 1)
                size += _estimate(v, depth - 1)
            return size

        if isinstance(obj, (list, tuple, set, frozenset)):
            size = 0
            for item in obj:
                size += _estimate(item, depth - 1)
            return size

        # Fallback for arbitrary objects
        if hasattr(obj, "__dict__"):
            try:
                return _estimate(vars(obj), depth - 1)
            except Exception:  # noqa: BLE001
                return 128

        return 64

    return _estimate(state, max_depth)


def enforce_state_size_limit(
    state: Any,
    *,
    framework: Optional[str],
    error_codes: GraphCoercionErrorCodes,
    limits: Optional[GraphResourceLimits],
    flags: Optional[GraphValidationFlags] = None,
    logger: Optional[logging.Logger] = None,
) -> Any:
    """
    Enforce a maximum allowed state size, if configured.

    Returns the original state if within limits. If the limit is exceeded:
    - If fail_on_limit_exceeded → raises.
    - Otherwise → logs a warning and returns the state unchanged (adapters may
      choose to transform/truncate separately).
    """
    flags = flags or GraphValidationFlags()
    log = logger or LOG
    framework_name = _infer_framework_name(error_codes, framework, flags=flags)
    _validate_error_codes(error_codes, logger=log, flags=flags)

    if not limits or limits.max_state_size_bytes is None:
        return state

    size = estimate_state_size_bytes(state)
    if size <= limits.max_state_size_bytes:
        return state

    message = (
        f"[{error_codes.state_size_exceeded}] "
        f"{framework_name}: state size {size} bytes exceeds "
        f"max_state_size_bytes={limits.max_state_size_bytes}"
    )
    if flags.fail_on_limit_exceeded:
        raise ValueError(message)

    log.warning(message)
    return state


# ---------------------------------------------------------------------------
# Graph context helpers
# ---------------------------------------------------------------------------


def normalize_graph_context(
    graph_context: Optional[Mapping[str, Any]],
    *,
    framework: Optional[str],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Normalize graph context into a consistently-shaped dict.

    This helper is tolerant of:
    - Missing or None context
    - Alternate key casing (graphId → graph_id, etc.)
    """
    log = logger or LOG
    framework_name = (framework or "graph").lower()

    if graph_context is None:
        return {}

    if not isinstance(graph_context, Mapping):
        log.warning(
            "%s: graph_context must be a Mapping, got %s; ignoring context",
            framework_name,
            type(graph_context).__name__,
        )
        return {}

    # Copy + normalize common id fields
    ctx: Dict[str, Any] = {}
    for key, value in graph_context.items():
        ctx[key] = value

    def _maybe_move(src: str, dst: str) -> None:
        if src in ctx and dst not in ctx:
            ctx[dst] = ctx[src]

    # Normalize variants
    _maybe_move("graphId", "graph_id")
    _maybe_move("runId", "run_id")
    _maybe_move("stepId", "step_id")
    _maybe_move("nodeId", "node_id")
    _maybe_move("edgeId", "edge_id")
    _maybe_move("pathId", "path_id")
    _maybe_move("subgraphId", "subgraph_id")
    _maybe_move("traceId", "trace_id")
    _maybe_move("tenantId", "tenant_id")
    _maybe_move("userId", "user_id")

    return ctx


def attach_graph_context_to_framework_ctx(
    framework_ctx: MutableMapping[str, Any],
    *,
    graph_context: Mapping[str, Any],
    limits: Optional[GraphResourceLimits] = None,
    flags: Optional[GraphValidationFlags] = None,
) -> None:
    """
    Attach normalized graph context to a framework-level context dict.

    This helper mutates `framework_ctx` in-place.
    """
    flags = flags or GraphValidationFlags()

    for key in (
        "graph_id",
        "run_id",
        "step_id",
        "node_id",
        "edge_id",
        "path_id",
        "stage",
        "subgraph_id",
        "tenant_id",
        "user_id",
        "trace_id",
    ):
        if key in graph_context:
            framework_ctx[key] = graph_context[key]

    # Optional "batch strategy" hint for downstream batching logic
    stage = graph_context.get("stage")
    path_id = graph_context.get("path_id")
    if limits and flags.warn_on_excess_fanout:
        # This is purely metadata; exact strategy is chosen by adapters
        if stage:
            framework_ctx.setdefault("batch_strategy", f"stage_{stage}")
        elif path_id:
            framework_ctx.setdefault("batch_strategy", f"path_{path_id}")


# ---------------------------------------------------------------------------
# Streaming / event helpers
# ---------------------------------------------------------------------------


def iter_graph_events(
    events: Iterable[Any],
    *,
    framework: Optional[str],
    op_name: str,
    limits: Optional[GraphResourceLimits] = None,
    flags: Optional[GraphValidationFlags] = None,
    logger: Optional[logging.Logger] = None,
) -> Iterator[Any]:
    """
    Wrap an event stream with basic safety limits.

    - Counts events and enforces `max_total_stream_events` if configured.
    - Logs at most one warning when the limit is exceeded.
    - When `strict_stream_limits` is True, stops iteration after limit.
    """
    flags = flags or GraphValidationFlags()
    log = logger or LOG
    framework_name = (framework or "graph").lower()
    limits = limits or GraphResourceLimits()

    max_events = limits.max_total_stream_events
    count = 0
    warned = False

    for event in events:
        count += 1
        if max_events is not None and count > max_events:
            if not warned:
                log.warning(
                    "%s: %s exceeded max_total_stream_events=%d; "
                    "subsequent events will %s.",
                    framework_name,
                    op_name,
                    max_events,
                    "be dropped" if flags.strict_stream_limits else "still be yielded",
                )
                warned = True
            if flags.strict_stream_limits:
                break
            # If not strict, still yield but we don't spam warnings
        yield event


__all__ = [
    "GraphCoercionErrorCodes",
    "GraphResourceLimits",
    "GraphValidationFlags",
    "coerce_nodes",
    "coerce_edges",
    "coerce_graph",
    "warn_if_extreme_graph_size",
    "enforce_graph_limits",
    "estimate_graph_depth",
    "detect_cycles",
    "validate_node_ids",
    "validate_edge_ids",
    "estimate_state_size_bytes",
    "enforce_state_size_limit",
    "normalize_graph_context",
    "attach_graph_context_to_framework_ctx",
    "iter_graph_events",
]
