# corpus_sdk/vector/framework_adapters/crewai.py
# SPDX-License-Identifier: Apache-2.0

"""
CrewAI tool adapter for Corpus Vector protocol.

This module exposes a Corpus `VectorProtocolV1` implementation as a CrewAI
`BaseTool` that performs semantic vector search with:

- Sync + async run APIs (`_run` / `_arun`)
- Optional streaming search via the shared `VectorTranslator` streaming bridge
- Proper integration with Corpus `VectorProtocolV1` via `VectorTranslator`
- Namespace + metadata filter handling (capability-aware for async flows)
- Optional client-side score thresholding
- Optional embedding function integration
- Optional Max Marginal Relevance (MMR) diversification (non-streaming)
- Optional OperationContext propagation via context_translation

Design philosophy
-----------------
- Protocol-first: CrewAI is a thin skin over Corpus vector adapters.
- All heavy lifting (backpressure, deadlines, breakers, etc.) lives in
  the underlying adapter / protocol, not here.
- This layer focuses on:
    * Translating Corpus matches → JSON-serializable payloads
    * Respecting VectorCapabilities (namespaces, filters, top_k limits)
      where available (async paths)
    * Propagating OperationContext when provided (crewai task or dict)
    * Using `VectorTranslator` for all sync/async query and streaming orchestration
"""

from __future__ import annotations

import asyncio
import logging
import math
from functools import cached_property, wraps
from threading import RLock
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from pydantic import BaseModel, Field, PrivateAttr

# --------------------------------------------------------------------------- #
# Optional CrewAI import (soft dependency)
# --------------------------------------------------------------------------- #

try:  # pragma: no cover - optional dependency
    from crewai.tools import BaseTool

    CREWAI_AVAILABLE = True
except ImportError:  # pragma: no cover - environments without CrewAI
    CREWAI_AVAILABLE = False

    class BaseTool(BaseModel):  # type: ignore[no-redef]
        """Minimal stub BaseTool; real usage requires `crewai`."""

        pass


from corpus_sdk.vector.vector_base import (
    VectorProtocolV1,
    QueryResult,
    OperationContext,
    VectorCapabilities,
    # Errors
    BadRequest,
    NotSupported,
)
from corpus_sdk.vector.framework_adapters.common.framework_utils import (
    VectorCoercionErrorCodes,
    VectorResourceLimits,
    VectorValidationFlags,
    TopKWarningConfig,
    warn_if_extreme_k,
    normalize_vector_context,
    attach_vector_context_to_framework_ctx,
)
from corpus_sdk.vector.framework_adapters.common.vector_translation import (
    DefaultVectorFrameworkTranslator,
    VectorTranslator,
)
from corpus_sdk.core.error_context import attach_context
from corpus_sdk.core.context_translation import (
    from_crewai as ctx_from_crewai,
    from_dict as ctx_from_dict,
)

logger = logging.getLogger(__name__)

Embeddings = Sequence[Sequence[float]]
Metadata = Dict[str, Any]


def _coerce_to_vector_operation_context(ctx: Any) -> OperationContext:
    """Coerce a context-like object into the vector protocol's OperationContext.

    The framework context translation utilities in `corpus_sdk.core.context_translation`
    return the core `OperationContext`, which is protocol-agnostic. Vector adapters
    (and `VectorTranslator`) use the vector protocol `OperationContext` defined in
    `corpus_sdk.vector.vector_base`.

    This helper accepts either type (or any object with a compatible attribute
    surface) and reconstructs the vector OperationContext.
    """
    if isinstance(ctx, OperationContext):
        return ctx

    return OperationContext(
        request_id=getattr(ctx, "request_id", None),
        idempotency_key=getattr(ctx, "idempotency_key", None),
        deadline_ms=getattr(ctx, "deadline_ms", None),
        traceparent=getattr(ctx, "traceparent", None),
        tenant=getattr(ctx, "tenant", None),
        attrs=getattr(ctx, "attrs", None) or {},
    )


# --------------------------------------------------------------------------- #
# Protocol client wrapper
# --------------------------------------------------------------------------- #


class CorpusCrewAIVectorClient:
    """VectorProtocolV1-shaped wrapper used by the conformance test suite.

    CrewAI's primary integration surface is a tool (`CorpusCrewAIVectorSearchTool`),
    but conformance tests expect a wrapper exposing the strict vector surface.
    """

    def __init__(self, *, adapter: VectorProtocolV1) -> None:
        self._translator = VectorTranslator(
            adapter=adapter,
            framework="crewai",
            translator=DefaultVectorFrameworkTranslator(),
        )

    def _call_with_optional_framework_ctx(self, method_name: str, *args: Any, task: Optional[Any] = None) -> Any:
        method = getattr(self._translator, method_name)
        if task is None:
            return method(*args)
        try:
            return method(*args, framework_ctx=task)
        except TypeError:
            # Some translator implementations / test doubles don't accept framework_ctx.
            return method(*args)

    def capabilities(self, *, task: Optional[Any] = None) -> Any:
        return self._call_with_optional_framework_ctx("capabilities", task=task)

    def health(self, *, task: Optional[Any] = None) -> Any:
        return self._call_with_optional_framework_ctx("health", task=task)

    def query(self, raw_query: Any, *, task: Optional[Any] = None) -> Any:
        return self._call_with_optional_framework_ctx("query", raw_query, task=task)

    def batch_query(self, raw_queries: Any, *, task: Optional[Any] = None) -> Any:
        return self._call_with_optional_framework_ctx("batch_query", raw_queries, task=task)

    def upsert(self, raw_documents: Any, *, task: Optional[Any] = None) -> Any:
        return self._call_with_optional_framework_ctx("upsert", raw_documents, task=task)

    def delete(self, raw_filter_or_ids: Any, *, task: Optional[Any] = None) -> Any:
        return self._call_with_optional_framework_ctx("delete", raw_filter_or_ids, task=task)

    def create_namespace(self, name: str, *, task: Optional[Any] = None) -> Any:
        return self._call_with_optional_framework_ctx("create_namespace", name, task=task)

    def delete_namespace(self, name: str, *, task: Optional[Any] = None) -> Any:
        return self._call_with_optional_framework_ctx("delete_namespace", name, task=task)


# --------------------------------------------------------------------------- #
# Vector framework-utils configuration for CrewAI adapter
# --------------------------------------------------------------------------- #

# Vector coercion error codes (vector-flavored, framework-labeled).
# These are passed into attach_context for consistent observability.
VECTOR_ERROR_CODES: VectorCoercionErrorCodes = VectorCoercionErrorCodes(
    invalid_vector_result="CREWAI_VECTOR_INVALID_RESULT",
    invalid_hit_result="CREWAI_VECTOR_INVALID_HIT_RESULT",
    empty_result="CREWAI_VECTOR_EMPTY_RESULT",
    conversion_error="CREWAI_VECTOR_CONVERSION_ERROR",
    score_out_of_range="CREWAI_VECTOR_SCORE_OUT_OF_RANGE",
    vector_dimension_exceeded="CREWAI_VECTOR_DIMENSION_EXCEEDED",
    vector_norm_invalid="CREWAI_VECTOR_NORM_INVALID",
    framework_label="crewai",
)

VECTOR_LIMITS = VectorResourceLimits()
VECTOR_FLAGS = VectorValidationFlags()
TOPK_WARNING_CONFIG = TopKWarningConfig(
    framework_label="crewai",
)


# --------------------------------------------------------------------------- #
# Event-loop guard (sync entrypoints)
# --------------------------------------------------------------------------- #


def _ensure_not_in_event_loop(sync_api_name: str) -> None:
    """
    Prevent deadlocks from calling sync APIs in active event loops.

    This is a defensive guard for environments where users might accidentally
    call sync tool/vector APIs from async frameworks.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop -> safe to use sync API.
        return
    raise RuntimeError(
        f"{sync_api_name} was called from inside an active asyncio event loop. "
        f"Use the async variant instead."
    )


# --------------------------------------------------------------------------- #
# Error codes and decorators for richer error context
# --------------------------------------------------------------------------- #


class ErrorCodes:
    BAD_OPERATION_CONTEXT = "BAD_OPERATION_CONTEXT"
    BAD_QUERY_RESULT = "BAD_QUERY_RESULT"
    BAD_STREAM_CHUNK = "BAD_STREAM_CHUNK"
    BAD_EMBEDDINGS = "BAD_EMBEDDINGS"
    NO_EMBEDDING_FUNCTION = "NO_EMBEDDING_FUNCTION"
    EMBEDDING_ERROR = "EMBEDDING_ERROR"
    BAD_TOP_K = "BAD_TOP_K"
    FILTER_NOT_SUPPORTED = "FILTER_NOT_SUPPORTED"
    BAD_QUERY_TEXT = "BAD_QUERY_TEXT"
    BAD_VECTOR_DIMENSION = "BAD_VECTOR_DIMENSION"
    CAPABILITIES_NOT_AVAILABLE = "CAPABILITIES_NOT_AVAILABLE"


def _build_dynamic_error_context(
    operation: str,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    base_context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build an enriched, vector-flavored error context for tool/vector operations.

    Best-effort only: this must never raise.
    """
    extra: Dict[str, Any] = dict(base_context)
    try:
        extra.setdefault("operation", operation)

        # Many internal helpers receive the input schema as a positional arg
        # (e.g., _search_simple_sync(self, args)). Extract common fields from it.
        input_payload: Dict[str, Any] = {}
        if len(args) >= 2:
            candidate = args[1]
            try:
                if isinstance(candidate, Mapping):
                    input_payload.update(candidate)
                else:
                    for key in (
                        "query",
                        "k",
                        "fetch_k",
                        "namespace",
                        "filter",
                        "use_mmr",
                        "mmr_lambda",
                        "return_scores",
                        "embedding",
                        "texts",
                    ):
                        if hasattr(candidate, key):
                            input_payload[key] = getattr(candidate, key)
            except Exception:
                # Best-effort only.
                pass

        # Self-introspection: defaults & basic metadata
        if args:
            self_obj = args[0]
            default_ns = getattr(self_obj, "namespace", None)
            if default_ns is not None:
                extra.setdefault("default_namespace", default_ns)
            score_threshold = getattr(self_obj, "score_threshold", None)
            if score_threshold is not None:
                extra.setdefault("score_threshold", score_threshold)
            default_top_k = getattr(self_obj, "default_top_k", None)
            if default_top_k is not None:
                extra.setdefault("default_top_k", default_top_k)
            dim_hint = getattr(self_obj, "_vector_dim_hint", None)
            if dim_hint is not None:
                extra.setdefault("vector_dim_hint", dim_hint)

        # Common query parameters (kwargs win over schema payload)
        query = kwargs.get("query") if "query" in kwargs else input_payload.get("query")
        if isinstance(query, str):
            extra.setdefault("query_chars", len(query))

        k = kwargs.get("k") if "k" in kwargs else input_payload.get("k")
        if isinstance(k, int):
            extra.setdefault("k", k)

        fetch_k = kwargs.get("fetch_k") if "fetch_k" in kwargs else input_payload.get("fetch_k")
        if isinstance(fetch_k, int):
            extra.setdefault("fetch_k", fetch_k)

        namespace = kwargs.get("namespace") if "namespace" in kwargs else input_payload.get("namespace")
        if namespace is not None:
            extra.setdefault("namespace", namespace)

        if "filter" in kwargs or "filter" in input_payload:
            fval = kwargs.get("filter") if "filter" in kwargs else input_payload.get("filter")
            extra.setdefault("has_filter", fval is not None)

        if "return_scores" in kwargs or "return_scores" in input_payload:
            rval = kwargs.get("return_scores") if "return_scores" in kwargs else input_payload.get("return_scores")
            extra.setdefault("return_scores", bool(rval))

        # Embedding / vectorization inputs
        emb = kwargs.get("embedding") if "embedding" in kwargs else input_payload.get("embedding")
        if emb is not None:
            if isinstance(emb, Sequence):
                extra.setdefault("embedding_dim", len(emb))

        texts = kwargs.get("texts") if "texts" in kwargs else input_payload.get("texts")
        if texts is not None:
            try:
                texts_list = list(texts)  # may raise for non-iterables
                extra.setdefault("vectors_count", len(texts_list))
                total_chars = 0
                for t in texts_list:
                    if isinstance(t, str):
                        total_chars += len(t)
                    else:
                        total_chars += len(str(t))
                extra.setdefault("total_content_chars", total_chars)
            except Exception:
                extra.setdefault("vectors_count", 1)

        # If this wrapper was invoked via CrewAI tool `_run/_arun`,
        # the kwargs often represent the input schema fields.
        # Attach the most common ones.
        if "use_mmr" in kwargs or "use_mmr" in input_payload:
            extra.setdefault("use_mmr", kwargs.get("use_mmr") if "use_mmr" in kwargs else input_payload.get("use_mmr"))
        if "mmr_lambda" in kwargs or "mmr_lambda" in input_payload:
            mmr_lambda_val = kwargs.get("mmr_lambda") if "mmr_lambda" in kwargs else input_payload.get("mmr_lambda")
            extra.setdefault("mmr_lambda", mmr_lambda_val)

            # Convenience: attach the internal parameter name used by MMR selection.
            # This is the clamped value that downstream logic uses.
            try:
                self_obj = args[0] if args else None
                default_mmr_lambda = getattr(self_obj, "mmr_lambda", None) if self_obj is not None else None
                raw_lambda = mmr_lambda_val if mmr_lambda_val is not None else default_mmr_lambda
                if raw_lambda is not None:
                    lambda_mult = float(raw_lambda)
                    lambda_mult = max(0.0, min(1.0, lambda_mult))
                    extra.setdefault("lambda_mult", lambda_mult)
            except Exception:
                pass
    except Exception:
        # Error-context enrichment must never be fatal.
        pass
    return extra


def with_error_context(
    operation: str,
    **context_kwargs: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to automatically attach error context to synchronous exceptions.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                enhanced_context = _build_dynamic_error_context(
                    operation=operation,
                    args=args,
                    kwargs=kwargs,
                    base_context=dict(context_kwargs),
                )
                attach_context(
                    exc,
                    framework="crewai",
                    error_codes=VECTOR_ERROR_CODES,
                    **enhanced_context,
                )
                raise

        return wrapper

    return decorator


def with_async_error_context(
    operation: str,
    **context_kwargs: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to automatically attach error context to async exceptions.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as exc:
                enhanced_context = _build_dynamic_error_context(
                    operation=operation,
                    args=args,
                    kwargs=kwargs,
                    base_context=dict(context_kwargs),
                )
                attach_context(
                    exc,
                    framework="crewai",
                    error_codes=VECTOR_ERROR_CODES,
                    **enhanced_context,
                )
                raise

        return wrapper

    return decorator


# --------------------------------------------------------------------------- #
# Input schema
# --------------------------------------------------------------------------- #


class CorpusVectorSearchInput(BaseModel):
    """
    Input schema for CorpusCrewAIVectorSearchTool.

    Fields are designed so that CrewAI can auto-generate tool calls,
    while still allowing advanced users to override behavior.
    """

    query: str = Field(
        ...,
        description="Natural language search query.",
    )

    k: Optional[int] = Field(
        default=None,
        description="Number of results to return. Defaults to the tool's default_top_k.",
    )

    namespace: Optional[str] = Field(
        default=None,
        description=(
            "Optional namespace override for the vector index. "
            "If omitted, the tool's configured namespace is used."
        ),
    )

    filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional metadata filter to restrict matches. "
            "Filter semantics are defined by the underlying Corpus adapter."
        ),
    )

    use_mmr: Optional[bool] = Field(
        default=None,
        description=(
            "Whether to apply Maximal Marginal Relevance (MMR) re-ranking "
            "for diversity. If omitted, the tool's use_mmr_by_default is used."
        ),
    )

    mmr_lambda: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "MMR lambda parameter in [0, 1]. Higher values favor relevance, "
            "lower values favor diversity. Defaults to the tool's mmr_lambda."
        ),
    )

    fetch_k: Optional[int] = Field(
        default=None,
        gt=0,
        description=(
            "Number of initial candidates to fetch before MMR re-ranking. "
            "Defaults to max(k * 4, k + 5) when MMR is enabled."
        ),
    )

    return_scores: Optional[bool] = Field(
        default=False,
        description="If true, include similarity scores in the results.",
    )

    embedding: Optional[List[float]] = Field(
        default=None,
        description=(
            "Optional precomputed query embedding. If provided, "
            "embedding_function is not used."
        ),
    )

    context: Optional[Any] = Field(
        default=None,
        description=(
            "Optional context used to build a Corpus OperationContext. "
            "For advanced use, this can be either:\n"
            "- A CrewAI Task-like object (translated via context_translation.from_crewai), or\n"
            "- A plain dict (translated via context_translation.from_dict)."
        ),
    )


# --------------------------------------------------------------------------- #
# CrewAI Tool implementation
# --------------------------------------------------------------------------- #


class CorpusCrewAIVectorSearchTool(BaseTool):
    """
    CrewAI `BaseTool` implementation backed by a Corpus `VectorProtocolV1`.

    This tool performs semantic vector search against a Corpus-backed index
    and returns a list of JSON objects:

        [
            {"text": "...", "metadata": {...}, "score": 0.87, "id": "..."},
            ...
        ]

    Key behaviors
    -------------
    - Uses an embedding function (if provided) to embed query text.
    - Uses `VectorTranslator` for all sync + async queries and streaming.
    - Respects VectorCapabilities (namespaces, filters, top_k limits) in async flows.
    - Supports optional MMR re-ranking for diversity (non-streaming paths).
    - Optionally thresholds matches by score_threshold.
    - Optionally propagates OperationContext using context_translation.
    - Offers an advanced `stream_search(...)` API backed by VectorTranslator's
      streaming bridge.
    """

    # CrewAI metadata
    name: str = "corpus_vector_search"
    description: str = (
        "Semantic vector search over a Corpus-backed index. "
        "Returns a JSON list of the most relevant text snippets and metadata "
        "for the given query, with optional MMR diversification."
    )

    # Tool input schema
    args_schema: Type[BaseModel] = CorpusVectorSearchInput

    # Core configuration
    corpus_adapter: VectorProtocolV1
    namespace: Optional[str] = "default"

    id_field: str = "id"
    text_field: str = "page_content"
    metadata_field: Optional[str] = None

    score_threshold: Optional[float] = None
    default_top_k: int = 4

    # Optional embedding integration
    embedding_function: Optional[Callable[[List[str]], Embeddings]] = None
    async_embedding_function: Optional[
        Callable[[List[str]], Awaitable[Embeddings]]
    ] = None

    # Optional MMR configuration
    use_mmr_by_default: bool = False
    mmr_lambda: float = 0.5

    # Optional static OperationContext for advanced scenarios
    static_operation_context: Optional[OperationContext] = None

    # Ownership model for cleanup: by default, the caller owns the adapter.
    own_adapter: bool = False

    # Internal / cached state (private attrs; not part of tool schema)
    _caps: Optional[VectorCapabilities] = PrivateAttr(default=None)
    _vector_dim_hint: Optional[int] = PrivateAttr(default=None)
    _dim_lock: RLock = PrivateAttr(default_factory=RLock)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the tool.

        We keep imports soft but fail loudly if someone tries to use the
        tool without CrewAI installed.
        """
        if not CREWAI_AVAILABLE:
            raise RuntimeError(
                "CorpusCrewAIVectorSearchTool requires `crewai` to be installed. "
                "Install it with `pip install crewai`."
            )
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------ #
    # Translator setup
    # ------------------------------------------------------------------ #

    class _CrewAIVectorFrameworkTranslator(DefaultVectorFrameworkTranslator):
        """
        CrewAI-specific VectorFrameworkTranslator.

        This translator reuses the default translator for spec construction
        and context handling, but deliberately *does not* reshape core
        protocol results:

        - QueryResult is returned as-is
        - Query chunks are returned as-is
        """

        def translate_query_result(
            self,
            result: QueryResult,
            *,
            op_ctx: OperationContext,
            framework_ctx: Optional[Any] = None,
        ) -> QueryResult:
            return result

        def translate_query_chunk(
            self,
            chunk: Any,
            *,
            op_ctx: OperationContext,
            framework_ctx: Optional[Any] = None,
        ) -> Any:
            return chunk

    @cached_property
    def _translator(self) -> VectorTranslator:
        """
        Lazily construct and cache the `VectorTranslator`.

        All sync and async search operations, including streaming, are
        performed through this translator to centralize async↔sync bridging
        and error-context handling.
        """
        framework_translator = self._CrewAIVectorFrameworkTranslator()
        return VectorTranslator(
            adapter=self.corpus_adapter,
            framework="crewai",
            translator=framework_translator,
        )

    # ------------------------------------------------------------------ #
    # Context building (separated core + framework contexts)
    # ------------------------------------------------------------------ #

    def _build_core_context(self, call_context: Optional[Any]) -> Optional[OperationContext]:
        """
        Build an OperationContext from an incoming CrewAI context payload.

        Contract:
        - If translation is attempted and fails, we raise with attached context.
        - We do not silently fall back to "no context" on translation failures.
        - If call_context is None, we may return the configured static_operation_context.
        """
        if isinstance(call_context, OperationContext):
            return call_context

        if call_context is None:
            if self.static_operation_context is None:
                return None
            return _coerce_to_vector_operation_context(self.static_operation_context)

        # Prefer explicit mapping translation when a dict-like is supplied.
        if isinstance(call_context, Mapping):
            try:
                ctx = ctx_from_dict(call_context)
            except Exception as exc:  # noqa: BLE001
                attach_context(
                    exc,
                    framework="crewai",
                    operation="context_translation_from_dict",
                    error_codes=VECTOR_ERROR_CODES,
                )
                raise
            try:
                return _coerce_to_vector_operation_context(ctx)
            except Exception as exc:  # noqa: BLE001
                err = BadRequest(
                    f"from_dict produced unsupported context type: {type(ctx).__name__}",
                    code=ErrorCodes.BAD_OPERATION_CONTEXT,
                )
                attach_context(
                    err,
                    framework="crewai",
                    operation="context_translation_from_dict",
                    error_codes=VECTOR_ERROR_CODES,
                )
                raise err from exc

        # Otherwise attempt CrewAI-specific translation for Task-like objects.
        try:
            ctx = ctx_from_crewai(call_context)
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="crewai",
                operation="context_translation_from_crewai",
                error_codes=VECTOR_ERROR_CODES,
            )
            raise

        try:
            return _coerce_to_vector_operation_context(ctx)
        except Exception as exc:  # noqa: BLE001
            err = BadRequest(
                f"from_crewai produced unsupported context type: {type(ctx).__name__}",
                code=ErrorCodes.BAD_OPERATION_CONTEXT,
            )
            attach_context(
                err,
                framework="crewai",
                operation="context_translation_from_crewai",
                error_codes=VECTOR_ERROR_CODES,
            )
            raise err from exc

    def _effective_namespace(self, namespace: Optional[str]) -> Optional[str]:
        """Resolve namespace using explicit override or tool default."""
        return namespace if namespace is not None else self.namespace

    def _build_framework_context(
        self,
        *,
        namespace: Optional[str],
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build a normalized framework_ctx mapping consistent with shared vector utilities.

        This ensures vector context keys (namespace, index_name, tenant_id, etc.) are
        normalized and attached into a generic framework_ctx dict.
        """
        ns = self._effective_namespace(namespace)

        raw_ctx: Dict[str, Any] = {}
        if ns is not None:
            raw_ctx["namespace"] = ns
        if extra_context:
            raw_ctx.update(extra_context)

        vector_ctx = normalize_vector_context(
            raw_ctx,
            framework="crewai",
            logger=logger,
        )

        framework_ctx: Dict[str, Any] = {}
        attach_vector_context_to_framework_ctx(
            framework_ctx,
            vector_context=vector_ctx,
            limits=VECTOR_LIMITS,
            flags=VECTOR_FLAGS,
        )
        return framework_ctx

    def _build_contexts(
        self,
        *,
        call_context: Optional[Any],
        namespace: Optional[str],
        extra_framework_ctx: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[Optional[OperationContext], Dict[str, Any]]:
        """
        Orchestrate core + framework context building for all public operations.
        """
        op_ctx = self._build_core_context(call_context)
        fw_ctx = self._build_framework_context(
            namespace=namespace,
            extra_context=extra_framework_ctx,
        )
        return op_ctx, fw_ctx

    # ------------------------------------------------------------------ #
    # Capabilities ownership (translator-first; no adapter fallback)
    # ------------------------------------------------------------------ #

    async def _get_caps_async(self) -> VectorCapabilities:
        """
        Async capability fetch with caching (translator-owned).

        Contract:
        - Capabilities must be provided by the VectorTranslator (preferred),
          not directly by the underlying adapter.
        - If translator does not implement capabilities, raise NotSupported.
        """
        if self._caps is not None:
            return self._caps

        # Translator-owned capability resolution:
        # 1) acapabilities()
        # 2) capabilities() via worker thread
        translator_acapabilities = getattr(self._translator, "acapabilities", None)
        if callable(translator_acapabilities):
            try:
                caps = await translator_acapabilities()
                self._caps = caps
                return caps
            except Exception as exc:  # noqa: BLE001
                attach_context(
                    exc,
                    framework="crewai",
                    operation="translator_acapabilities",
                    error_codes=VECTOR_ERROR_CODES,
                )
                raise

        translator_capabilities = getattr(self._translator, "capabilities", None)
        if callable(translator_capabilities):
            try:
                caps = await asyncio.to_thread(translator_capabilities)
                self._caps = caps
                return caps
            except Exception as exc:  # noqa: BLE001
                attach_context(
                    exc,
                    framework="crewai",
                    operation="translator_capabilities",
                    error_codes=VECTOR_ERROR_CODES,
                )
                raise

        err = NotSupported(
            "VectorTranslator for framework='crewai' must implement "
            "acapabilities() or capabilities(); no adapter fallback is allowed.",
            code=ErrorCodes.CAPABILITIES_NOT_AVAILABLE,
        )
        attach_context(
            err,
            framework="crewai",
            operation="capabilities",
            error_codes=VECTOR_ERROR_CODES,
        )
        raise err

    # ------------------------------------------------------------------ #
    # Dimension hint helpers (thread-safe first-write-wins)
    # ------------------------------------------------------------------ #

    def _update_dim_hint(self, dim: Optional[int]) -> None:
        """
        Thread-safe, best-effort update of the vector dimension hint.

        First successful write wins; subsequent calls are no-ops.
        """
        if dim is None:
            return
        try:
            d = int(dim)
        except Exception:
            return
        if d <= 0:
            return
        if self._vector_dim_hint is not None:
            return
        with self._dim_lock:
            if self._vector_dim_hint is None:
                self._vector_dim_hint = d

    def _zero_vector(self) -> List[float]:
        """
        Deterministic zero vector based on known dimension hint.
        """
        dim = self._vector_dim_hint
        if dim is None or dim <= 0:
            raise BadRequest(
                "vector dimension is unknown; cannot create deterministic zero vector",
                code=ErrorCodes.BAD_VECTOR_DIMENSION,
            )
        return [0.0] * int(dim)

    def _validate_embedding_dimension(self, vec: Sequence[float]) -> None:
        """
        Validate embedding dimension against the dimension hint (when known).

        If the hint is not set yet, this method updates it.
        """
        dim = len(vec)
        if dim <= 0:
            return
        hint = self._vector_dim_hint
        if hint is None:
            self._update_dim_hint(dim)
            return
        if int(hint) != int(dim):
            raise BadRequest(
                f"embedding dimension {dim} does not match expected {hint}",
                code=ErrorCodes.BAD_VECTOR_DIMENSION,
                details={"expected": int(hint), "actual": int(dim)},
            )

    # ------------------------------------------------------------------ #
    # Result validation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_query_result(
        result: Any,
        *,
        operation: str,
    ) -> QueryResult:
        """Validate that the translator returned a QueryResult."""
        if not isinstance(result, QueryResult):
            raise BadRequest(
                f"{operation} returned unsupported type: {type(result).__name__}",
                code=ErrorCodes.BAD_QUERY_RESULT,
            )
        return result

    @staticmethod
    def _validate_stream_chunk(
        chunk: Any,
        *,
        operation: str,
    ) -> Any:
        """Validate that the translator returned a usable streaming chunk."""
        matches = getattr(chunk, "matches", None)
        if matches is None and isinstance(chunk, Mapping):
            matches = chunk.get("matches")
        if matches is None:
            raise BadRequest(
                f"{operation} yielded unsupported chunk type: {type(chunk).__name__}",
                code=ErrorCodes.BAD_STREAM_CHUNK,
            )
        return chunk

    # ------------------------------------------------------------------ #
    # Embedding + vectorization helpers
    # ------------------------------------------------------------------ #

    def vectorize_query(
        self,
        query: str,
        *,
        embedding: Optional[Sequence[float]] = None,
    ) -> List[float]:
        """
        Vectorize a single query (sync).

        - Guards against event-loop usage.
        - Empty query returns deterministic zero vector when dimension is known.
        """
        _ensure_not_in_event_loop("vectorize_query")
        return self._embed_query(query, embedding=embedding)

    async def avectorize_query(
        self,
        query: str,
        *,
        embedding: Optional[Sequence[float]] = None,
    ) -> List[float]:
        """
        Vectorize a single query (async).

        Empty query returns deterministic zero vector when dimension is known.
        """
        return await self._embed_query_async(query, embedding=embedding)

    def vectorize_documents(
        self,
        texts: Sequence[str],
        *,
        embeddings: Optional[Embeddings] = None,
    ) -> List[List[float]]:
        """
        Vectorize a batch of documents (sync).

        - Guards against event-loop usage.
        - Empty texts return deterministic zero vectors when dimension is known.
        """
        _ensure_not_in_event_loop("vectorize_documents")

        texts_list = [str(t) for t in texts]
        if not texts_list:
            return []

        # Pre-handle empties for deterministic output.
        # If all texts are empty, we can return zero vectors without calling any embedding provider.
        empty_mask: List[bool] = [not str(t).strip() for t in texts_list]
        if all(empty_mask):
            z = self._zero_vector()
            return [list(z) for _ in texts_list]

        # If embeddings are supplied, validate and coerce.
        if embeddings is not None:
            if len(embeddings) != len(texts_list):
                raise BadRequest(
                    f"embeddings length {len(embeddings)} does not match texts length {len(texts_list)}",
                    code=ErrorCodes.BAD_EMBEDDINGS,
                )
            out: List[List[float]] = []
            for i, e in enumerate(embeddings):
                vec = [float(x) for x in (e or [])]
                if not vec:
                    # Deterministic shape for empty vectors if dim known.
                    vec = self._zero_vector()
                self._validate_embedding_dimension(vec)
                out.append(vec)
            return out

        # Use embedding function if configured.
        if self.embedding_function is None:
            raise NotSupported(
                "No embedding_function configured; caller must supply embeddings",
                code=ErrorCodes.NO_EMBEDDING_FUNCTION,
                details={"texts": len(texts_list)},
            )

        non_empty_texts: List[str] = [t for t, is_empty in zip(texts_list, empty_mask) if not is_empty]
        try:
            computed = self.embedding_function(non_empty_texts)
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="crewai",
                operation="vectorize_documents_embedding_function",
                error_codes=VECTOR_ERROR_CODES,
                vectors_count=len(texts_list),
                total_content_chars=sum(len(t) for t in texts_list),
            )
            raise BadRequest(
                f"embedding_function failed for documents: {exc}",
                code=ErrorCodes.EMBEDDING_ERROR,
            )

        if len(computed) != len(non_empty_texts):
            raise BadRequest(
                f"embedding_function returned {len(computed)} embeddings for {len(non_empty_texts)} texts",
                code=ErrorCodes.BAD_EMBEDDINGS,
            )

        out_vectors: List[List[float]] = []
        idx_non_empty = 0
        for is_empty in empty_mask:
            if is_empty:
                vec = self._zero_vector()
            else:
                vec = [float(x) for x in computed[idx_non_empty]]
                idx_non_empty += 1
                if not vec:
                    vec = self._zero_vector()
            self._validate_embedding_dimension(vec)
            out_vectors.append(vec)

        return out_vectors

    async def avectorize_documents(
        self,
        texts: Sequence[str],
        *,
        embeddings: Optional[Embeddings] = None,
    ) -> List[List[float]]:
        """
        Vectorize a batch of documents (async).

        Empty texts return deterministic zero vectors when dimension is known.
        """
        texts_list = [str(t) for t in texts]
        if not texts_list:
            return []

        if embeddings is not None:
            if len(embeddings) != len(texts_list):
                raise BadRequest(
                    f"embeddings length {len(embeddings)} does not match texts length {len(texts_list)}",
                    code=ErrorCodes.BAD_EMBEDDINGS,
                )
            out: List[List[float]] = []
            for e in embeddings:
                vec = [float(x) for x in (e or [])]
                if not vec:
                    vec = self._zero_vector()
                self._validate_embedding_dimension(vec)
                out.append(vec)
            return out

        # Pre-handle empties deterministically.
        empty_mask: List[bool] = [not str(t).strip() for t in texts_list]
        if all(empty_mask):
            z = self._zero_vector()
            return [list(z) for _ in texts_list]

        non_empty_texts: List[str] = [t for t, is_empty in zip(texts_list, empty_mask) if not is_empty]

        # Prefer async embedding function when provided.
        if self.async_embedding_function is not None:
            try:
                computed = await self.async_embedding_function(non_empty_texts)
            except Exception as exc:  # noqa: BLE001
                attach_context(
                    exc,
                    framework="crewai",
                    operation="avectorize_documents_async_embedding_function",
                    error_codes=VECTOR_ERROR_CODES,
                    vectors_count=len(texts_list),
                    total_content_chars=sum(len(t) for t in texts_list),
                )
                raise BadRequest(
                    f"async_embedding_function failed for documents: {exc}",
                    code=ErrorCodes.EMBEDDING_ERROR,
                )
        else:
            if self.embedding_function is None:
                raise NotSupported(
                    "No embedding_function/async_embedding_function configured; caller must supply embeddings",
                    code=ErrorCodes.NO_EMBEDDING_FUNCTION,
                    details={"texts": len(texts_list)},
                )
            try:
                computed = await asyncio.to_thread(self.embedding_function, non_empty_texts)
            except Exception as exc:  # noqa: BLE001
                attach_context(
                    exc,
                    framework="crewai",
                    operation="avectorize_documents_embedding_function",
                    error_codes=VECTOR_ERROR_CODES,
                    vectors_count=len(texts_list),
                    total_content_chars=sum(len(t) for t in texts_list),
                )
                raise BadRequest(
                    f"embedding_function failed for documents: {exc}",
                    code=ErrorCodes.EMBEDDING_ERROR,
                )

        if len(computed) != len(non_empty_texts):
            raise BadRequest(
                f"embedding function returned {len(computed)} embeddings for {len(non_empty_texts)} texts",
                code=ErrorCodes.BAD_EMBEDDINGS,
            )

        out_vectors: List[List[float]] = []
        idx_non_empty = 0
        for is_empty in empty_mask:
            if is_empty:
                vec = self._zero_vector()
            else:
                vec = [float(x) for x in computed[idx_non_empty]]
                idx_non_empty += 1
                if not vec:
                    vec = self._zero_vector()
            self._validate_embedding_dimension(vec)
            out_vectors.append(vec)

        return out_vectors

    # ------------------------------------------------------------------ #
    # Compatibility: stable vector-action method names (no renames/removals)
    # ------------------------------------------------------------------ #
    #
    # These are strict aliases/wrappers around the existing hardened implementations
    # above. They exist to preserve cross-adapter consistency (e.g., shared base
    # layers that call embed_query/embed_documents).
    #
    # IMPORTANT: These wrappers do not change behavior, validation, or performance;
    # they only provide stable method names.
    #

    @with_error_context("embed_query_sync")
    def embed_query(
        self,
        text: str,
        *,
        embedding: Optional[Sequence[float]] = None,
    ) -> List[float]:
        """
        Compatibility alias for query embedding (sync).

        Mirrors the common adapter surface name: embed_query(text, embedding=...).
        """
        return self.vectorize_query(str(text), embedding=embedding)

    @with_async_error_context("embed_query_async")
    async def aembed_query(
        self,
        text: str,
        *,
        embedding: Optional[Sequence[float]] = None,
    ) -> List[float]:
        """
        Compatibility alias for query embedding (async).

        Mirrors the common adapter surface name: aembed_query(text, embedding=...).
        """
        return await self.avectorize_query(str(text), embedding=embedding)

    # Optional alternate async naming some integrations use.
    # Kept as an alias for maximum compatibility.
    @with_async_error_context("embed_query_async_alias")
    async def embed_query_async(
        self,
        text: str,
        *,
        embedding: Optional[Sequence[float]] = None,
    ) -> List[float]:
        """
        Compatibility alias (async) for embed_query in frameworks that prefer
        the '*_async' naming convention.
        """
        return await self.avectorize_query(str(text), embedding=embedding)

    @with_error_context("embed_documents_sync")
    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        embeddings: Optional[Embeddings] = None,
    ) -> List[List[float]]:
        """
        Compatibility alias for document embedding (sync).

        Mirrors the common adapter surface name: embed_documents(texts, embeddings=...).
        """
        return self.vectorize_documents(texts, embeddings=embeddings)

    @with_async_error_context("embed_documents_async")
    async def aembed_documents(
        self,
        texts: Sequence[str],
        *,
        embeddings: Optional[Embeddings] = None,
    ) -> List[List[float]]:
        """
        Compatibility alias for document embedding (async).

        Mirrors the common adapter surface name: aembed_documents(texts, embeddings=...).
        """
        return await self.avectorize_documents(texts, embeddings=embeddings)

    # Optional alternate async naming some integrations use.
    @with_async_error_context("embed_documents_async_alias")
    async def embed_documents_async(
        self,
        texts: Sequence[str],
        *,
        embeddings: Optional[Embeddings] = None,
    ) -> List[List[float]]:
        """
        Compatibility alias (async) for embed_documents in frameworks that prefer
        the '*_async' naming convention.
        """
        return await self.avectorize_documents(texts, embeddings=embeddings)

    # Optional "embed_texts" naming used in some codebases; kept as an alias.
    @with_error_context("embed_texts_sync")
    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        embeddings: Optional[Embeddings] = None,
    ) -> List[List[float]]:
        """
        Compatibility alias (sync) for embed_documents.
        """
        return self.vectorize_documents(texts, embeddings=embeddings)

    @with_async_error_context("embed_texts_async")
    async def aembed_texts(
        self,
        texts: Sequence[str],
        *,
        embeddings: Optional[Embeddings] = None,
    ) -> List[List[float]]:
        """
        Compatibility alias (async) for aembed_documents.
        """
        return await self.avectorize_documents(texts, embeddings=embeddings)

    def _embed_query(
        self,
        query: str,
        *,
        embedding: Optional[Sequence[float]] = None,
    ) -> List[float]:
        """
        Ensure a single query embedding is available (sync path).

        Behavior:
        - If `embedding` is provided, coerce to float list and validate dimension.
        - If query is empty/whitespace:
            - return deterministic zero vector if dimension hint is known
            - otherwise raise BAD_QUERY_TEXT (avoids inconsistent provider behavior)
        - Else, if `embedding_function` is set, compute embedding for [query].
        - Else, raise NotSupported.
        """
        if embedding is not None:
            vec = [float(x) for x in embedding]
            if not vec:
                vec = self._zero_vector()
            self._validate_embedding_dimension(vec)
            return vec

        if not str(query).strip():
            # Deterministic behavior for empty queries
            try:
                vec = self._zero_vector()
            except Exception as exc:
                err = BadRequest(
                    "query cannot be empty when vector dimension is unknown",
                    code=ErrorCodes.BAD_QUERY_TEXT,
                    details={"query_preview": str(query)[:64]},
                )
                attach_context(
                    err,
                    framework="crewai",
                    operation="vectorize_query_empty",
                    error_codes=VECTOR_ERROR_CODES,
                )
                raise err from exc
            self._validate_embedding_dimension(vec)
            return vec

        if self.embedding_function is None:
            raise NotSupported(
                "No embedding_function configured; caller must supply query embedding",
                code=ErrorCodes.NO_EMBEDDING_FUNCTION,
                details={"framework": "crewai", "query_preview": query[:64]},
            )

        try:
            embs = self.embedding_function([query])
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework="crewai",
                operation="embedding_function",
                query_preview=query[:128],
                error_codes=VECTOR_ERROR_CODES,
            )
            raise BadRequest(
                f"embedding_function failed for query: {exc}",
                code=ErrorCodes.EMBEDDING_ERROR,
            )

        if not embs or len(embs) != 1:
            raise BadRequest(
                "embedding_function must return exactly one embedding for a single query",
                code=ErrorCodes.BAD_EMBEDDINGS,
            )

        vec = [float(x) for x in embs[0]]
        if not vec:
            vec = self._zero_vector()
        self._validate_embedding_dimension(vec)
        return vec

    async def _embed_query_async(
        self,
        query: str,
        *,
        embedding: Optional[Sequence[float]] = None,
    ) -> List[float]:
        """
        Async-safe query embedding helper.

        Behavior:
        - If `embedding` is provided, coerce to float list and validate dimension.
        - If query is empty/whitespace:
            - return deterministic zero vector if dimension hint is known
            - otherwise raise BAD_QUERY_TEXT
        - Else, if `async_embedding_function` is set, await it.
        - Else, if `embedding_function` is set, run it in a worker thread.
        - Else, raise NotSupported.
        """
        if embedding is not None:
            vec = [float(x) for x in embedding]
            if not vec:
                vec = self._zero_vector()
            self._validate_embedding_dimension(vec)
            return vec

        if not str(query).strip():
            try:
                vec = self._zero_vector()
            except Exception as exc:
                err = BadRequest(
                    "query cannot be empty when vector dimension is unknown",
                    code=ErrorCodes.BAD_QUERY_TEXT,
                    details={"query_preview": str(query)[:64]},
                )
                attach_context(
                    err,
                    framework="crewai",
                    operation="avectorize_query_empty",
                    error_codes=VECTOR_ERROR_CODES,
                )
                raise err from exc
            self._validate_embedding_dimension(vec)
            return vec

        if self.async_embedding_function is not None:
            try:
                embs = await self.async_embedding_function([query])
            except Exception as exc:  # noqa: BLE001
                attach_context(
                    exc,
                    framework="crewai",
                    operation="embed_query_async",
                    query_preview=query[:128],
                    error_codes=VECTOR_ERROR_CODES,
                )
                raise BadRequest(
                    f"async_embedding_function failed for query: {exc}",
                    code=ErrorCodes.EMBEDDING_ERROR,
                )
        else:
            if self.embedding_function is None:
                raise NotSupported(
                    "No embedding_function/async_embedding_function configured; "
                    "caller must supply query embedding",
                    code=ErrorCodes.NO_EMBEDDING_FUNCTION,
                    details={"framework": "crewai", "query_preview": query[:64]},
                )
            try:
                embs = await asyncio.to_thread(self.embedding_function, [query])
            except Exception as exc:  # noqa: BLE001
                attach_context(
                    exc,
                    framework="crewai",
                    operation="embed_query_async",
                    query_preview=query[:128],
                    error_codes=VECTOR_ERROR_CODES,
                )
                raise BadRequest(
                    f"embedding_function failed for query: {exc}",
                    code=ErrorCodes.EMBEDDING_ERROR,
                )

        if not embs or len(embs) != 1:
            raise BadRequest(
                "embedding function must return exactly one embedding for a single query",
                code=ErrorCodes.BAD_EMBEDDINGS,
            )

        vec = [float(x) for x in embs[0]]
        if not vec:
            vec = self._zero_vector()
        self._validate_embedding_dimension(vec)
        return vec

    # ------------------------------------------------------------------ #
    # Match translation helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_match_score(match: Any) -> float:
        """Robustly extract a numeric score from a match object or mapping."""
        if isinstance(match, Mapping):
            value = match.get("score", 0.0)
        else:
            value = getattr(match, "score", 0.0)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _extract_match_vector_raw(match: Any) -> Any:
        """
        Extract a raw vector payload from a match, without coercion.

        Supports:
        - Mapping with 'embedding' key
        - Objects with `vector.vector` attribute (VectorMatch-style)
        """
        if isinstance(match, Mapping) and "embedding" in match:
            return match.get("embedding")

        if hasattr(match, "vector") and getattr(match, "vector") is not None:
            vec_obj = getattr(match, "vector")
            if hasattr(vec_obj, "vector"):
                return getattr(vec_obj, "vector")

        return None

    def _get_match_vector(self, match: Any) -> List[float]:
        """
        Robustly extract an embedding vector from a match and update dimension hint.
        """
        vec_raw = self._extract_match_vector_raw(match)
        if vec_raw is None:
            return []

        try:
            vec = [float(x) for x in vec_raw]
        except (TypeError, ValueError):
            return []

        if vec:
            # Best-effort: update dimension hint from returned vectors.
            self._update_dim_hint(len(vec))
        return vec

    def _filter_matches_by_score(self, matches: Sequence[Any]) -> List[Any]:
        """Apply optional client-side score threshold to matches."""
        if self.score_threshold is None:
            return list(matches)

        threshold = float(self.score_threshold)
        return [m for m in matches if self._get_match_score(m) >= threshold]

    def _match_to_payload(
        self,
        match: Any,
        *,
        return_scores: bool,
    ) -> Dict[str, Any]:
        """
        Convert a single match into a JSON-serializable payload.

        Supports both:
        - Mapping-based matches: {'metadata': {...}, 'score': ..., 'id': ..., 'text': ...}
        - VectorMatch-style objects: match.vector.metadata, match.score, match.vector.id
        """
        # Extract metadata
        if isinstance(match, Mapping):
            meta_full = dict(match.get("metadata") or {})
        else:
            v = getattr(match, "vector", None)
            meta_full = dict(getattr(v, "metadata", {}) or {}) if v is not None else {}

        # Handle metadata envelope if configured
        if self.metadata_field and self.metadata_field in meta_full:
            nested = meta_full.get(self.metadata_field) or {}
            if isinstance(nested, Mapping):
                user_meta: Dict[str, Any] = dict(nested)
            else:
                user_meta = {}
        else:
            user_meta = dict(meta_full)

        # Extract text and id
        text_value: Any = meta_full.get(self.text_field)
        text = text_value if isinstance(text_value, str) else ""

        id_value: Any
        if isinstance(match, Mapping):
            id_value = match.get("id")
        else:
            v = getattr(match, "vector", None)
            id_value = getattr(v, "id", None) if v is not None else None
            if id_value is None:
                id_value = meta_full.get(self.id_field)

        # Remove internal keys from user metadata
        user_meta.pop(self.text_field, None)
        user_meta.pop(self.id_field, None)

        payload: Dict[str, Any] = {
            "text": text,
            "metadata": user_meta,
        }
        if id_value is not None:
            payload["id"] = str(id_value)
        if return_scores:
            payload["score"] = float(self._get_match_score(match))

        return payload

    # ------------------------------------------------------------------ #
    # Query spec builders for the VectorTranslator
    # ------------------------------------------------------------------ #

    def _build_raw_query(
        self,
        embedding: Sequence[float],
        *,
        k: int,
        namespace: Optional[str],
        filter: Optional[Mapping[str, Any]],
        include_vectors: bool,
    ) -> Mapping[str, Any]:
        """
        Build a raw query mapping suitable for VectorTranslator.

        The common VectorTranslator expects a mapping with:
            - 'vector': list[float]
            - 'top_k': int
            - 'filters': optional mapping
            - 'namespace': optional str
            - 'include_metadata': bool
            - 'include_vectors': bool
        """
        ns = self._effective_namespace(namespace)
        raw: Dict[str, Any] = {
            "vector": [float(x) for x in embedding],
            "top_k": int(k),
            "filters": dict(filter) if filter is not None else None,
            "namespace": ns,
            "include_metadata": True,
            "include_vectors": bool(include_vectors),
        }
        return raw

    # ------------------------------------------------------------------ #
    # MMR utilities (manual, on top of translator results)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cosine_sim(a: Sequence[float], b: Sequence[float]) -> float:
        """Simple cosine similarity helper."""
        dot = 0.0
        na = 0.0
        nb = 0.0
        for x, y in zip(a, b):
            dot += x * y
            na += x * x
            nb += y * y
        if na <= 0.0 or nb <= 0.0:
            return 0.0
        return float(dot / (math.sqrt(na) * math.sqrt(nb)))

    def _mmr_select_indices(
        self,
        candidate_matches: List[Any],
        k: int,
        lambda_mult: float,
    ) -> List[int]:
        """
        Improved MMR selector that respects original database scores and caches similarities.
        """
        if not candidate_matches or k <= 0:
            return []

        k = min(k, len(candidate_matches))
        if k == 0:
            return []

        # Optimization: lambda=1.0 (or above) → pure relevance ranking, no diversity term.
        if lambda_mult >= 1.0:
            indices = sorted(
                range(len(candidate_matches)),
                key=lambda idx: self._get_match_score(candidate_matches[idx]),
                reverse=True,
            )
            return indices[:k]

        original_scores = [self._get_match_score(m) for m in candidate_matches]
        candidate_vecs: List[List[float]] = [self._get_match_vector(m) for m in candidate_matches]

        # Normalize original scores to 0-1 range for consistency
        max_orig_score = max(original_scores) if original_scores else 1.0
        if max_orig_score <= 0.0:
            normalized_scores = [0.0] * len(original_scores)
        else:
            normalized_scores = [score / max_orig_score for score in original_scores]

        similarity_cache: Dict[Tuple[int, int], float] = {}

        def get_similarity(i: int, j: int) -> float:
            if (i, j) in similarity_cache:
                return similarity_cache[(i, j)]
            if (j, i) in similarity_cache:
                return similarity_cache[(j, i)]

            vec_i = candidate_vecs[i]
            vec_j = candidate_vecs[j]
            if not vec_i or not vec_j or len(vec_i) != len(vec_j):
                sim = 0.0
            else:
                sim = self._cosine_sim(vec_i, vec_j)

            similarity_cache[(i, j)] = sim
            return sim

        selected: List[int] = []
        candidates = list(range(len(candidate_matches)))

        # Start with the most relevant document based on original score
        if candidates:
            first_idx = max(candidates, key=lambda idx: normalized_scores[idx])
            selected.append(first_idx)
            candidates.remove(first_idx)

        while candidates and len(selected) < k:
            best_idx = None
            best_score = -float("inf")

            for idx in candidates:
                relevance = normalized_scores[idx]

                max_similarity = 0.0
                for sel_idx in selected:
                    similarity = get_similarity(idx, sel_idx)
                    max_similarity = max(max_similarity, similarity)

                mmr_score = lambda_mult * relevance - (1.0 - lambda_mult) * max_similarity

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx is None:
                break

            selected.append(best_idx)
            candidates.remove(best_idx)

        return selected

    # ------------------------------------------------------------------ #
    # High-level async search flows (simple vs MMR)
    # ------------------------------------------------------------------ #

    @with_async_error_context("vector_search_async")
    async def _asearch_simple(
        self,
        args: CorpusVectorSearchInput,
        *,
        caps: VectorCapabilities,
    ) -> List[Dict[str, Any]]:
        """
        Simple top-k search without MMR (async).
        """
        top_k = int(self.default_top_k if args.k is None else args.k)
        if top_k <= 0:
            return []

        if caps.max_top_k is not None and top_k > caps.max_top_k:
            raise BadRequest(
                f"top_k {top_k} exceeds maximum of {caps.max_top_k}",
                code=ErrorCodes.BAD_TOP_K,
            )

        warn_if_extreme_k(
            top_k,
            framework="crewai",
            op_name="vector_search_async_simple",
            warning_config=TOPK_WARNING_CONFIG,
            logger=logger,
        )

        ns = args.namespace
        op_ctx, fw_ctx = self._build_contexts(
            call_context=args.context,
            namespace=ns,
        )

        if args.filter and not caps.supports_metadata_filtering:
            raise NotSupported(
                "metadata filtering is not supported by the underlying vector adapter",
                code=ErrorCodes.FILTER_NOT_SUPPORTED,
                details={"namespace": self._effective_namespace(ns)},
            )

        query_emb = await self._embed_query_async(args.query, embedding=args.embedding)
        raw_query = self._build_raw_query(
            embedding=query_emb,
            k=top_k,
            namespace=ns,
            filter=args.filter,
            include_vectors=False,
        )

        result = await self._translator.arun_query(
            raw_query,
            op_ctx=op_ctx,
            framework_ctx=fw_ctx,
            mmr_config=None,
        )
        result_qr = self._validate_query_result(result, operation="VectorTranslator.arun_query")

        matches_all = list(result_qr.matches or [])
        matches = self._filter_matches_by_score(matches_all)

        return_scores = bool(args.return_scores)
        return [self._match_to_payload(m, return_scores=return_scores) for m in matches]

    @with_async_error_context("vector_search_mmr_async")
    async def _asearch_with_mmr(
        self,
        args: CorpusVectorSearchInput,
        *,
        caps: VectorCapabilities,
    ) -> List[Dict[str, Any]]:
        """
        MMR-based search that first fetches candidates and then re-ranks them (async).
        """
        top_k = int(self.default_top_k if args.k is None else args.k)
        if top_k <= 0:
            return []

        warn_if_extreme_k(
            top_k,
            framework="crewai",
            op_name="vector_search_async_mmr",
            warning_config=TOPK_WARNING_CONFIG,
            logger=logger,
        )

        ns = args.namespace
        op_ctx, fw_ctx = self._build_contexts(
            call_context=args.context,
            namespace=ns,
        )

        if args.filter and not caps.supports_metadata_filtering:
            raise NotSupported(
                "metadata filtering is not supported by the underlying vector adapter",
                code=ErrorCodes.FILTER_NOT_SUPPORTED,
                details={"namespace": self._effective_namespace(ns)},
            )

        lambda_mult = float(args.mmr_lambda) if args.mmr_lambda is not None else float(self.mmr_lambda)
        lambda_mult = max(0.0, min(1.0, lambda_mult))

        fetch_k = args.fetch_k or max(top_k * 4, top_k + 5)
        if caps.max_top_k is not None and fetch_k > caps.max_top_k:
            raise BadRequest(
                f"fetch_k {fetch_k} exceeds maximum of {caps.max_top_k}",
                code=ErrorCodes.BAD_TOP_K,
            )

        warn_if_extreme_k(
            fetch_k,
            framework="crewai",
            op_name="vector_search_async_mmr_fetch",
            warning_config=TOPK_WARNING_CONFIG,
            logger=logger,
        )

        query_emb = await self._embed_query_async(args.query, embedding=args.embedding)
        raw_query = self._build_raw_query(
            embedding=query_emb,
            k=int(fetch_k),
            namespace=ns,
            filter=args.filter,
            include_vectors=True,
        )

        result = await self._translator.arun_query(
            raw_query,
            op_ctx=op_ctx,
            framework_ctx=fw_ctx,
            mmr_config=None,  # manual MMR applied here
        )
        result_qr = self._validate_query_result(result, operation="VectorTranslator.arun_query")

        matches_all = list(result_qr.matches or [])
        matches_all = self._filter_matches_by_score(matches_all)
        if not matches_all:
            return []

        indices = self._mmr_select_indices(
            candidate_matches=matches_all,
            k=top_k,
            lambda_mult=lambda_mult,
        )

        return_scores = bool(args.return_scores)
        return [self._match_to_payload(matches_all[i], return_scores=return_scores) for i in indices]

    @with_async_error_context("vector_search_dispatch_async")
    async def _asearch(self, args: CorpusVectorSearchInput) -> List[Dict[str, Any]]:
        """Unified async search entry point, dispatching to simple or MMR-based search."""
        caps = await self._get_caps_async()

        use_mmr = bool(args.use_mmr) if args.use_mmr is not None else bool(self.use_mmr_by_default)
        if use_mmr:
            return await self._asearch_with_mmr(args, caps=caps)
        return await self._asearch_simple(args, caps=caps)

    # ------------------------------------------------------------------ #
    # High-level sync search flows (simple vs MMR)
    # ------------------------------------------------------------------ #

    @with_error_context("vector_search_sync")
    def _search_simple_sync(self, args: CorpusVectorSearchInput) -> List[Dict[str, Any]]:
        """
        Simple top-k search without MMR (sync).

        Uses VectorTranslator.query directly (no direct AsyncBridge usage).
        """
        _ensure_not_in_event_loop("CorpusCrewAIVectorSearchTool._search_simple_sync")

        top_k = int(self.default_top_k if args.k is None else args.k)
        if top_k <= 0:
            return []

        warn_if_extreme_k(
            top_k,
            framework="crewai",
            op_name="vector_search_sync_simple",
            warning_config=TOPK_WARNING_CONFIG,
            logger=logger,
        )

        ns = args.namespace
        op_ctx, fw_ctx = self._build_contexts(
            call_context=args.context,
            namespace=ns,
        )

        query_emb = self._embed_query(args.query, embedding=args.embedding)
        raw_query = self._build_raw_query(
            embedding=query_emb,
            k=top_k,
            namespace=ns,
            filter=args.filter,
            include_vectors=False,
        )

        result = self._translator.query(
            raw_query,
            op_ctx=op_ctx,
            framework_ctx=fw_ctx,
            mmr_config=None,
        )
        result_qr = self._validate_query_result(result, operation="VectorTranslator.query")

        matches_all = list(result_qr.matches or [])
        matches = self._filter_matches_by_score(matches_all)

        return_scores = bool(args.return_scores)
        return [self._match_to_payload(m, return_scores=return_scores) for m in matches]

    @with_error_context("vector_search_mmr_sync")
    def _search_with_mmr_sync(self, args: CorpusVectorSearchInput) -> List[Dict[str, Any]]:
        """
        MMR-based search that fetches candidates and then re-ranks them (sync).
        """
        _ensure_not_in_event_loop("CorpusCrewAIVectorSearchTool._search_with_mmr_sync")

        top_k = int(self.default_top_k if args.k is None else args.k)
        if top_k <= 0:
            return []

        warn_if_extreme_k(
            top_k,
            framework="crewai",
            op_name="vector_search_sync_mmr",
            warning_config=TOPK_WARNING_CONFIG,
            logger=logger,
        )

        ns = args.namespace
        op_ctx, fw_ctx = self._build_contexts(
            call_context=args.context,
            namespace=ns,
        )

        lambda_mult = float(args.mmr_lambda) if args.mmr_lambda is not None else float(self.mmr_lambda)
        lambda_mult = max(0.0, min(1.0, lambda_mult))

        fetch_k = args.fetch_k or max(top_k * 4, top_k + 5)

        warn_if_extreme_k(
            fetch_k,
            framework="crewai",
            op_name="vector_search_sync_mmr_fetch",
            warning_config=TOPK_WARNING_CONFIG,
            logger=logger,
        )

        query_emb = self._embed_query(args.query, embedding=args.embedding)
        raw_query = self._build_raw_query(
            embedding=query_emb,
            k=int(fetch_k),
            namespace=ns,
            filter=args.filter,
            include_vectors=True,
        )

        result = self._translator.query(
            raw_query,
            op_ctx=op_ctx,
            framework_ctx=fw_ctx,
            mmr_config=None,  # manual MMR applied here
        )
        result_qr = self._validate_query_result(result, operation="VectorTranslator.query")

        matches_all = list(result_qr.matches or [])
        matches_all = self._filter_matches_by_score(matches_all)
        if not matches_all:
            return []

        indices = self._mmr_select_indices(
            candidate_matches=matches_all,
            k=top_k,
            lambda_mult=lambda_mult,
        )

        return_scores = bool(args.return_scores)
        return [self._match_to_payload(matches_all[i], return_scores=return_scores) for i in indices]

    # ------------------------------------------------------------------ #
    # CrewAI Tool API: sync + async
    # ------------------------------------------------------------------ #

    @with_error_context("tool_run")
    def _run(self, **kwargs: Any) -> List[Dict[str, Any]]:
        """
        Synchronous tool entrypoint used by CrewAI.

        Internally delegates to the sync search implementation which uses
        VectorTranslator for all vector operations.
        """
        _ensure_not_in_event_loop("CorpusCrewAIVectorSearchTool._run")

        args = self.args_schema(**kwargs)  # type: ignore[arg-type]

        use_mmr = bool(args.use_mmr) if args.use_mmr is not None else bool(self.use_mmr_by_default)
        if use_mmr:
            return self._search_with_mmr_sync(args)
        return self._search_simple_sync(args)

    @with_async_error_context("tool_arun")
    async def _arun(self, **kwargs: Any) -> List[Dict[str, Any]]:
        """
        Async tool entrypoint for advanced users.

        CrewAI may call this in some configurations, but typical usage
        goes through `_run`.
        """
        args = self.args_schema(**kwargs)  # type: ignore[arg-type]
        return await self._asearch(args)

    # ------------------------------------------------------------------ #
    # Optional callable interface (preserves tool semantics)
    # ------------------------------------------------------------------ #

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Dual-purpose callable interface that preserves CrewAI tool semantics.

        - If called like a CrewAI tool (query/k/namespace/filter/use_mmr/...),
          delegates to `_run`.
        - If called with explicit `texts=[...]` or `text="..."`, performs vectorization.

        This avoids breaking common `tool("query")` usage while still providing
        a vector-function interface for advanced integrations.
        """
        _ensure_not_in_event_loop("CorpusCrewAIVectorSearchTool.__call__")

        # Vectorization mode (explicit keywords only).
        if "texts" in kwargs:
            texts = kwargs.get("texts") or []
            embeddings = kwargs.get("embeddings")
            return self.vectorize_documents(texts, embeddings=embeddings)
        if "text" in kwargs:
            text = kwargs.get("text") or ""
            embedding = kwargs.get("embedding")
            return self.vectorize_query(str(text), embedding=embedding)

        # Tool mode:
        if args and not kwargs:
            # Common convenience: tool("query") -> _run(query="query")
            if len(args) == 1 and isinstance(args[0], str):
                return self._run(query=args[0])
            # Otherwise fall back to schema parsing through kwargs-only.
            raise TypeError(
                "Tool call expects keyword arguments (or a single positional query string). "
                "Vectorization expects keywords: texts=[...], text='...'."
            )

        return self._run(**kwargs)

    # ------------------------------------------------------------------ #
    # Advanced: streaming search using VectorTranslator.query_stream
    # ------------------------------------------------------------------ #

    @with_error_context("stream_search")
    def stream_search(self, **kwargs: Any) -> Iterator[Dict[str, Any]]:
        """
        Streaming search API for advanced callers (outside CrewAI's planner).

        This method:
        - Uses the same input schema as `_run` / `_arun`.
        - Bridges the underlying async stream via VectorTranslator.query_stream.
        - Applies score thresholding per match.
        - For correctness and consistency with the shared translator layer,
          MMR is not applied to streaming results (matching the protocol
          design that MMR requires the full result set).
        """
        _ensure_not_in_event_loop("CorpusCrewAIVectorSearchTool.stream_search")

        args = self.args_schema(**kwargs)  # type: ignore[arg-type]

        top_k = int(self.default_top_k if args.k is None else args.k)
        if top_k <= 0:
            return iter(())

        warn_if_extreme_k(
            top_k,
            framework="crewai",
            op_name="vector_search_stream",
            warning_config=TOPK_WARNING_CONFIG,
            logger=logger,
        )

        ns = args.namespace
        op_ctx, fw_ctx = self._build_contexts(
            call_context=args.context,
            namespace=ns,
        )
        return_scores = bool(args.return_scores)

        query_emb = self._embed_query(args.query, embedding=args.embedding)

        raw_query = self._build_raw_query(
            embedding=query_emb,
            k=top_k,
            namespace=ns,
            filter=args.filter,
            include_vectors=False,
        )

        yielded = 0
        for chunk in self._translator.query_stream(
            raw_query,
            op_ctx=op_ctx,
            framework_ctx=fw_ctx,
        ):
            chunk_qc = self._validate_stream_chunk(chunk, operation="VectorTranslator.query_stream")

            raw_matches = list(chunk_qc.matches or [])
            filtered_matches = self._filter_matches_by_score(raw_matches)

            for match in filtered_matches:
                if yielded >= top_k:
                    return
                yield self._match_to_payload(match, return_scores=return_scores)
                yielded += 1

    # ------------------------------------------------------------------ #
    # Resource cleanup (close / aclose / context managers)
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """
        Best-effort synchronous cleanup of translator resources.

        This method never raises; it only logs warnings on failure.
        By default we do not close the underlying adapter (caller-owned),
        unless `own_adapter=True`.
        """
        try:
            translator_close = getattr(self._translator, "close", None)
        except Exception:
            translator_close = None

        if callable(translator_close):
            try:
                translator_close()
            except Exception:
                logger.warning("Error while closing VectorTranslator synchronously", exc_info=True)

        if self.own_adapter:
            try:
                adapter_close = getattr(self.corpus_adapter, "close", None)
            except Exception:
                adapter_close = None

            if callable(adapter_close):
                try:
                    adapter_close()
                except Exception:
                    logger.warning("Error while closing corpus_adapter synchronously", exc_info=True)

    async def aclose(self) -> None:
        """
        Best-effort asynchronous cleanup of translator resources.

        This method never raises; it only logs warnings on failure.
        By default we do not close the underlying adapter (caller-owned),
        unless `own_adapter=True`.
        """
        try:
            translator_aclose = getattr(self._translator, "aclose", None)
        except Exception:
            translator_aclose = None

        if callable(translator_aclose):
            try:
                await translator_aclose()
            except Exception:
                logger.warning("Error while closing VectorTranslator asynchronously", exc_info=True)
        else:
            try:
                translator_close = getattr(self._translator, "close", None)
            except Exception:
                translator_close = None

            if callable(translator_close):
                try:
                    if asyncio.iscoroutinefunction(translator_close):
                        await translator_close()
                    else:
                        await asyncio.to_thread(translator_close)
                except Exception:
                    logger.warning("Error while closing VectorTranslator from async context", exc_info=True)

        if self.own_adapter:
            try:
                adapter_aclose = getattr(self.corpus_adapter, "aclose", None)
            except Exception:
                adapter_aclose = None

            if callable(adapter_aclose):
                try:
                    await adapter_aclose()
                    return
                except Exception:
                    logger.warning("Error while closing corpus_adapter asynchronously", exc_info=True)

            try:
                adapter_close = getattr(self.corpus_adapter, "close", None)
            except Exception:
                adapter_close = None

            if callable(adapter_close):
                try:
                    if asyncio.iscoroutinefunction(adapter_close):
                        await adapter_close()
                    else:
                        await asyncio.to_thread(adapter_close)
                except Exception:
                    logger.warning("Error while closing corpus_adapter from async context", exc_info=True)

    def __enter__(self) -> "CorpusCrewAIVectorSearchTool":
        """Synchronous context manager entry; returns self."""
        _ensure_not_in_event_loop("CorpusCrewAIVectorSearchTool.__enter__")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Synchronous context manager exit; best-effort close()."""
        try:
            self.close()
        except Exception:
            logger.warning("Error while closing tool in __exit__", exc_info=True)

    async def __aenter__(self) -> "CorpusCrewAIVectorSearchTool":
        """Async context manager entry; returns self."""
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Async context manager exit; best-effort aclose()."""
        try:
            await self.aclose()
        except Exception:
            logger.warning("Error while closing tool in __aexit__", exc_info=True)


__all__ = [
    "CorpusVectorSearchInput",
    "CorpusCrewAIVectorSearchTool",
    "with_error_context",
    "with_async_error_context",
]
