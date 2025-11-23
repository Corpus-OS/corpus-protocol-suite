# corpus_sdk/embedding/framework_adapters/common/embedding_translation.py
# SPDX-License-Identifier: Apache-2.0
"""
Framework-agnostic Embedding → Framework translation layer.

Purpose
-------
Provide a high-level orchestration and translation layer between:

- The Corpus Embedding Protocol V1 (`EmbeddingProtocolV1` / `BaseEmbeddingAdapter`), and
- Framework-specific embedding integrations (LangChain, LlamaIndex, SK, AutoGen, CrewAI, custom).

This module is intentionally *framework-neutral* and focuses on:

- Building `EmbedSpec` / `BatchEmbedSpec` from framework-level inputs
- Translating `EmbedResult` / `EmbedChunk` / `BatchEmbedResult` back to framework-facing shapes
- Handling text normalization (truncation, cleaning, batching hints)
- Providing sync + async APIs, including streaming via a sync bridge
- Attaching rich error context for observability
- Passing through token usage / stats data from adapters
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Union,
)

from corpus_sdk.embedding.embedding_base import (
    EmbeddingProtocolV1,
    OperationContext,
    EmbedSpec,
    BatchEmbedSpec,
    EmbedResult,
    EmbedChunk,
    BatchEmbedResult,
    EmbeddingStats,
    BadRequest,
)

from corpus_sdk.core.context_translation import from_dict as ctx_from_dict
from corpus_sdk.core.sync_bridge import SyncStreamBridge
from corpus_sdk.core.async_bridge import AsyncBridge
from corpus_sdk.llm.framework_adapters.common.error_context import attach_context

LOG = logging.getLogger(__name__)


# =============================================================================
# Helpers: OperationContext normalization
# =============================================================================


def _ensure_operation_context(
    ctx: Optional[Union[OperationContext, Mapping[str, Any]]],
) -> OperationContext:
    """
    Normalize various context shapes into an embedding OperationContext.

    Accepts:
        - None: returns an "empty" context (via context_translation)
        - OperationContext: returned as-is
        - Mapping[str, Any]: normalized via context_translation.from_dict
    """
    if ctx is None:
        core_ctx = ctx_from_dict({})
    elif isinstance(ctx, OperationContext):
        return ctx
    elif isinstance(ctx, Mapping):
        core_ctx = ctx_from_dict(ctx)
    else:
        raise BadRequest(
            f"Unsupported context type: {type(ctx).__name__}",
            code="BAD_OPERATION_CONTEXT",
        )

    return OperationContext(
        request_id=getattr(core_ctx, "request_id", None),
        idempotency_key=getattr(core_ctx, "idempotency_key", None),
        deadline_ms=getattr(core_ctx, "deadline_ms", None),
        traceparent=getattr(core_ctx, "traceparent", None),
        tenant=getattr(core_ctx, "tenant", None),
        metrics=getattr(core_ctx, "metrics", None),
        attrs=getattr(core_ctx, "attrs", None) or {},
    )


# =============================================================================
# Batching configuration
# =============================================================================


@dataclass(frozen=True)
class BatchConfig:
    """
    Configuration for batching behavior.

    Note
    ----
    This class *does not* implement batching logic itself. It is a shared
    configuration object that can be passed to components that actually
    perform batching (e.g., adapters or higher-level orchestrators).

    The EmbeddingTranslator stores this config for inspection / wiring, but it
    never splits or groups texts on its own.
    """

    enabled: bool = True
    max_batch_size: int = 32
    max_tokens_per_batch: Optional[int] = None
    sort_by_length: bool = True
    retry_on_partial_failure: bool = True

    def __post_init__(self) -> None:
        """Validate batch configuration parameters."""
        if self.enabled:
            if self.max_batch_size <= 0:
                raise ValueError(f"max_batch_size must be positive, got {self.max_batch_size}")
            if self.max_tokens_per_batch is not None and self.max_tokens_per_batch <= 0:
                raise ValueError(
                    f"max_tokens_per_batch must be positive, got {self.max_tokens_per_batch}"
                )


# =============================================================================
# Text normalization configuration
# =============================================================================


@dataclass(frozen=True)
class TextNormalizationConfig:
    """
    Configuration for text preprocessing before embedding.

    Attributes:
        normalize_whitespace:
            Collapse multiple whitespace characters to single spaces and strip
            leading/trailing whitespace.

        remove_empty:
            If True, filter out empty or whitespace-only texts.

        max_length:
            Optional character limit. Texts longer than this are truncated.

        truncate_strategy:
            How to truncate long texts:
            - "end": Keep beginning, truncate end (default)
            - "start": Keep end, truncate beginning
            - "middle": Keep beginning and end, truncate middle.

        encoding:
            Text encoding to validate/enforce. Default "utf-8".

        lowercase:
            If True, convert all text to lowercase.
    """

    normalize_whitespace: bool = True
    remove_empty: bool = True
    max_length: Optional[int] = None
    truncate_strategy: str = "end"
    encoding: str = "utf-8"
    lowercase: bool = False

    def __post_init__(self) -> None:
        """Validate text normalization configuration."""
        if self.truncate_strategy not in ("end", "start", "middle"):
            raise ValueError(
                "truncate_strategy must be 'end', 'start', or 'middle', "
                f"got {self.truncate_strategy!r}"
            )
        if self.max_length is not None and self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")


# =============================================================================
# Text normalization helpers
# =============================================================================


class TextNormalizer:
    """Helper for normalizing text before embedding."""

    def __init__(self, config: TextNormalizationConfig) -> None:
        self.config = config

    def normalize(self, text: str) -> Optional[str]:
        """
        Normalize a single text according to configuration.

        Returns:
            Normalized text, or None if text should be filtered out.
        """
        if not isinstance(text, str):
            LOG.warning("TextNormalizer: non-string text %r, converting", type(text))
            text = str(text)

        # Validate / enforce encoding if requested
        if self.config.encoding:
            try:
                text = text.encode(self.config.encoding).decode(self.config.encoding)
            except (UnicodeEncodeError, UnicodeDecodeError):
                LOG.debug("TextNormalizer: encoding error, using replacement chars")
                text = text.encode(self.config.encoding, errors="replace").decode(
                    self.config.encoding
                )

        # Normalize whitespace
        if self.config.normalize_whitespace:
            import re

            text = re.sub(r"\s+", " ", text).strip()

        # Remove empty
        if self.config.remove_empty and not text.strip():
            return None

        # Lowercase
        if self.config.lowercase:
            text = text.lower()

        # Truncate
        if self.config.max_length is not None and len(text) > self.config.max_length:
            text = self._truncate(
                text, self.config.max_length, self.config.truncate_strategy
            )

        return text

    def normalize_batch(self, texts: Sequence[str]) -> List[str]:
        """
        Normalize a batch of texts, filtering out None results.

        Returns:
            List of normalized texts (may be shorter than input if texts filtered).
        """
        normalized: List[str] = []
        for text in texts:
            result = self.normalize(text)
            if result is not None:
                normalized.append(result)
        return normalized

    @staticmethod
    def _truncate(text: str, max_length: int, strategy: str) -> str:
        """Truncate text according to strategy."""
        if len(text) <= max_length:
            return text

        if strategy == "end":
            return text[:max_length]
        if strategy == "start":
            return text[-max_length:]
        if strategy == "middle":
            keep_each = max_length // 2
            return text[:keep_each] + text[-keep_each:]

        # Fallback (should not be hit if validated in config)
        return text[:max_length]


# =============================================================================
# Framework-agnostic translator protocol
# =============================================================================


class EmbeddingFrameworkTranslator(Protocol):
    """
    Per-framework translator contract.

    Implementations are responsible for:
        - Converting framework-level embed inputs into Embed*Spec types
        - Converting embedding results into framework-level outputs
        - Handling framework-specific document/text representations
    """

    # ---- embed translation ----

    def build_embed_spec(
        self,
        raw_texts: Any,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
        stream: bool = False,
    ) -> EmbedSpec:
        ...

    def translate_embed_result(
        self,
        result: EmbedResult,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        ...

    def translate_embed_chunk(
        self,
        chunk: EmbedChunk,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        ...

    # ---- batch embed translation ----

    def build_batch_embed_spec(
        self,
        raw_batch: Any,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> BatchEmbedSpec:
        ...

    def translate_batch_embed_result(
        self,
        result: BatchEmbedResult,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        ...

    # ---- stats / inspection ----

    def translate_stats(
        self,
        stats: EmbeddingStats,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        ...

    # ---- optional hooks ----

    def preferred_model(
        self,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Optional[str]:
        """
        Optional hook for translators to derive a default model name.

        This can come from:
            - framework_ctx (e.g., configured model)
            - op_ctx.attrs (e.g., "embedding_model" key)
        """
        ...


# =============================================================================
# Default generic translator implementation
# =============================================================================


class DefaultEmbeddingFrameworkTranslator:
    """
    Generic, framework-neutral translator implementation.

    Behaviors:
        - `embed` is single-text oriented (string or mapping with "text")
        - `batch_embed` handles lists/tuples or mappings with "texts"
        - Applies text normalization if configured
        - Results are translated into simple dicts that mirror dataclasses
    """

    def __init__(
        self,
        *,
        text_normalizer: Optional[TextNormalizer] = None,
    ) -> None:
        self._text_normalizer = text_normalizer or TextNormalizer(
            TextNormalizationConfig()
        )

    # ---- model helper ----

    def preferred_model(
        self,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Optional[str]:
        # Priority: explicit framework_ctx > embedding_model attr > model attr
        if isinstance(framework_ctx, Mapping):
            model = framework_ctx.get("model")
            if model is not None and str(model).strip():
                return str(model)

        attrs = op_ctx.attrs or {}
        # Try embedding_model first (most specific)
        model = attrs.get("embedding_model")
        if model is not None and str(model).strip():
            return str(model)

        # Fall back to generic model
        model = attrs.get("model")
        if model is not None and str(model).strip():
            return str(model)

        return None

    # ---- embed translation ----

    def _extract_single_text(self, raw_texts: Any) -> str:
        if isinstance(raw_texts, str):
            return raw_texts

        if isinstance(raw_texts, Mapping):
            if "text" in raw_texts:
                return str(raw_texts["text"])
            raise BadRequest(
                "Mapping input for embed must contain 'text'",
                code="BAD_TEXTS",
            )

        raise BadRequest(
            "embed expects a single string or mapping with 'text'; "
            "use batch_embed for multiple texts",
            code="BAD_TEXTS",
        )

    def build_embed_spec(
        self,
        raw_texts: Any,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
        stream: bool = False,
    ) -> EmbedSpec:
        model = self.preferred_model(op_ctx=op_ctx, framework_ctx=framework_ctx)

        if isinstance(raw_texts, Mapping):
            req_model = raw_texts.get("model")
            truncate = bool(raw_texts.get("truncate", True))
            normalize_vec = bool(raw_texts.get("normalize", False))
            text = self._extract_single_text(raw_texts)
            normalized = self._text_normalizer.normalize(text)
            if normalized is None:
                raise BadRequest(
                    "Text was filtered out during normalization (empty after processing)",
                    code="BAD_TEXT_EMPTY",
                )
            return EmbedSpec(
                text=normalized,
                model=str(req_model) if req_model is not None else model or "",
                truncate=truncate,
                normalize=normalize_vec,
                metadata=None,
                stream=bool(raw_texts.get("stream", stream)),
            )

        text = self._extract_single_text(raw_texts)
        normalized = self._text_normalizer.normalize(text)
        if normalized is None:
            raise BadRequest(
                "Text was filtered out during normalization (empty after processing)",
                code="BAD_TEXT_EMPTY",
            )

        return EmbedSpec(
            text=normalized,
            model=model or "",
            truncate=True,
            normalize=False,
            metadata=None,
            stream=stream,
        )

    def translate_embed_result(
        self,
        result: EmbedResult,
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Any:
        # Return a simple dict mirroring EmbedResult + embedding vector
        ev = result.embedding
        return {
            "embedding": ev.vector,
            "dimensions": ev.dimensions,
            "model": result.model,
            "text": result.text,
            "tokens_used": result.tokens_used,
            "truncated": result.truncated,
            "metadata": ev.metadata,
        }

    def translate_embed_chunk(
        self,
        chunk: EmbedChunk,
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Any:
        # Generic behavior: return flattened vectors plus flags
        return {
            "embeddings": [e.vector for e in (chunk.embeddings or [])],
            "model": chunk.model,
            "is_final": bool(chunk.is_final),
            "usage": dict(chunk.usage or {}) if chunk.usage is not None else None,
        }

    # ---- batch embed translation ----

    def _extract_text_list(self, raw_batch: Any) -> List[str]:
        if isinstance(raw_batch, Mapping):
            texts = raw_batch.get("texts")
            if texts is None:
                raise BadRequest(
                    "raw_batch mapping must contain 'texts'",
                    code="BAD_BATCH",
                )
        else:
            texts = raw_batch

        if isinstance(texts, str):
            texts = [texts]

        if not isinstance(texts, (list, tuple)):
            raise BadRequest(
                "texts must be a list (or tuple) of strings",
                code="BAD_BATCH",
            )

        if not texts:
            raise BadRequest(
                "texts must contain at least one item",
                code="BAD_BATCH_EMPTY",
            )

        return [str(t) for t in texts]

    def build_batch_embed_spec(
        self,
        raw_batch: Any,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> BatchEmbedSpec:
        model = self.preferred_model(op_ctx=op_ctx, framework_ctx=framework_ctx)

        texts = self._extract_text_list(raw_batch)
        normalized = self._text_normalizer.normalize_batch(texts)
        if not normalized:
            raise BadRequest(
                "All texts were filtered out during normalization",
                code="BAD_BATCH_ALL_EMPTY",
            )

        truncate = True
        normalize_vec = False
        req_model: Optional[str] = None

        if isinstance(raw_batch, Mapping):
            truncate = bool(raw_batch.get("truncate", True))
            normalize_vec = bool(raw_batch.get("normalize", False))
            req_model = raw_batch.get("model")

        return BatchEmbedSpec(
            texts=normalized,
            model=str(req_model) if req_model is not None else model or "",
            truncate=truncate,
            normalize=normalize_vec,
            metadatas=None,
        )

    def translate_batch_embed_result(
        self,
        result: BatchEmbedResult,
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Any:
        return {
            "embeddings": [e.vector for e in (result.embeddings or [])],
            "model": result.model,
            "total_texts": result.total_texts,
            "total_tokens": result.total_tokens,
            "failed_texts": list(result.failed_texts or []),
        }

    def translate_stats(
        self,
        stats: EmbeddingStats,
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Any:
        return asdict(stats)


# =============================================================================
# Embedding Translator Orchestrator
# =============================================================================


class EmbeddingTranslator:
    """
    Framework-agnostic orchestrator for embedding operations.

    This class:
        - Accepts framework-level inputs and a normalized OperationContext
        - Delegates to an EmbeddingFrameworkTranslator to build specs and translate results
        - Calls into an EmbeddingProtocolV1 adapter to execute operations
        - Provides sync + async variants for all core operations
        - Handles streaming via SyncStreamBridge for sync callers
        - Attaches rich error context for diagnostics

    Note: This layer does not perform automatic batch splitting. BatchConfig
    is a hint object for callers / adapters.
    """

    def __init__(
        self,
        *,
        adapter: EmbeddingProtocolV1,
        framework: str = "generic",
        translator: Optional[EmbeddingFrameworkTranslator] = None,
        batch_config: Optional[BatchConfig] = None,
    ) -> None:
        self._adapter = adapter
        self._framework = framework
        self._translator: EmbeddingFrameworkTranslator = (
            translator or DefaultEmbeddingFrameworkTranslator()
        )
        self._batch_config = batch_config or BatchConfig()

    # --------------------------------------------------------------------- #
    # Sync Embed APIs (use AsyncBridge)
    # --------------------------------------------------------------------- #

    def embed(
        self,
        raw_texts: Any,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """
        Synchronous embed API.

        Ergonomics:
            - If raw_texts is a single text / mapping → adapter.embed
            - If raw_texts is a list/tuple → routed to batch_embed for convenience
        """
        if isinstance(raw_texts, (list, tuple)) and not isinstance(raw_texts, Mapping):
            return self.batch_embed(
                {"texts": list(raw_texts)},
                op_ctx=op_ctx,
                framework_ctx=framework_ctx,
            )

        async def _embed_coro() -> Any:
            ctx = _ensure_operation_context(op_ctx)
            try:
                spec = self._translator.build_embed_spec(
                    raw_texts,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                    stream=False,
                )
                result = await self._adapter.embed(spec, ctx=ctx)

                if not isinstance(result, EmbedResult):
                    raise BadRequest(
                        f"adapter.embed returned unsupported type: "
                        f"{type(result).__name__}",
                        code="BAD_ADAPTER_RESULT",
                    )

                return self._translator.translate_embed_result(
                    result,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )
            except Exception as exc:
                attach_context(
                    exc,
                    framework=self._framework,
                    embedding_operation="embed",
                    request_id=ctx.request_id,
                    tenant=ctx.tenant,
                )
                raise

        ctx = _ensure_operation_context(op_ctx)
        timeout = ctx.deadline_ms / 1000.0 if ctx.deadline_ms else None
        return AsyncBridge.run_async(_embed_coro(), timeout=timeout)

    # --------------------------------------------------------------------- #
    # Async Embed APIs
    # --------------------------------------------------------------------- #

    async def arun_embed(
        self,
        raw_texts: Any,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """
        Async embed API (preferred for async applications).

        Ergonomics:
            - If raw_texts is a single text / mapping → adapter.embed
            - If raw_texts is a list/tuple → routed to arun_batch_embed
        """
        if isinstance(raw_texts, (list, tuple)) and not isinstance(raw_texts, Mapping):
            return await self.arun_batch_embed(
                {"texts": list(raw_texts)},
                op_ctx=op_ctx,
                framework_ctx=framework_ctx,
            )

        ctx = _ensure_operation_context(op_ctx)
        try:
            spec = self._translator.build_embed_spec(
                raw_texts,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
                stream=False,
            )
            result = await self._adapter.embed(spec, ctx=ctx)

            if not isinstance(result, EmbedResult):
                raise BadRequest(
                    f"adapter.embed returned unsupported type: "
                    f"{type(result).__name__}",
                    code="BAD_ADAPTER_RESULT",
                )

            return self._translator.translate_embed_result(
                result,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
        except Exception as exc:
            attach_context(
                exc,
                framework=self._framework,
                embedding_operation="embed",
                request_id=ctx.request_id,
                tenant=ctx.tenant,
            )
            raise

    # --------------------------------------------------------------------- #
    # Streaming Embed APIs
    # --------------------------------------------------------------------- #

    def embed_stream(
        self,
        raw_texts: Any,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Iterator[Any]:
        """
        Synchronous streaming embed API.

        Exposes a sync iterator that yields framework-level chunks by
        bridging adapter.embed(..., stream=True) via SyncStreamBridge.

        Only single-text streaming is supported; lists must use batch_embed
        or per-item streaming at a higher layer.
        """
        if isinstance(raw_texts, (list, tuple)) and not isinstance(raw_texts, Mapping):
            raise BadRequest(
                "embed_stream only supports single-text inputs; "
                "stream batches at a higher level",
                code="BAD_STREAM_INPUT",
            )

        ctx = _ensure_operation_context(op_ctx)

        async def _stream_factory() -> AsyncIterator[Any]:
            spec = self._translator.build_embed_spec(
                raw_texts,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
                stream=True,
            )
            try:
                stream_or_result = await self._adapter.embed(spec, ctx=ctx)
                if isinstance(stream_or_result, EmbedResult):
                    # Adapter misconfiguration: streaming requested but unary returned.
                    # Treat as a single final chunk.
                    chunk = EmbedChunk(
                        embeddings=[stream_or_result.embedding],
                        is_final=True,
                        usage=None,
                        model=stream_or_result.model,
                    )
                    yield self._translator.translate_embed_chunk(
                        chunk,
                        op_ctx=ctx,
                        framework_ctx=framework_ctx,
                    )
                    return

                stream = stream_or_result
                if not hasattr(stream, "__aiter__"):
                    raise BadRequest(
                        "adapter.embed did not return a streaming iterator",
                        code="BAD_ADAPTER_RESULT",
                    )

                async for chunk in stream:
                    if not isinstance(chunk, EmbedChunk):
                        raise BadRequest(
                            f"adapter.embed stream yielded unsupported type: "
                            f"{type(chunk).__name__}",
                            code="BAD_ADAPTER_RESULT",
                        )
                    yield self._translator.translate_embed_chunk(
                        chunk,
                        op_ctx=ctx,
                        framework_ctx=framework_ctx,
                    )
            except Exception as exc:
                attach_context(
                    exc,
                    framework=self._framework,
                    embedding_operation="stream_embed",
                    request_id=ctx.request_id,
                    tenant=ctx.tenant,
                )
                raise

        bridge = SyncStreamBridge(
            coro_factory=_stream_factory,
            framework=self._framework,
            error_context={
                "operation": "embedding.embed_stream",
                "request_id": ctx.request_id,
                "tenant": ctx.tenant,
            },
        )
        return bridge.run()

    async def arun_embed_stream(
        self,
        raw_texts: Any,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> AsyncIterator[Any]:
        """
        Async streaming embed API.

        Returns an async iterator yielding framework-level chunks.
        """
        if isinstance(raw_texts, (list, tuple)) and not isinstance(raw_texts, Mapping):
            raise BadRequest(
                "arun_embed_stream only supports single-text inputs; "
                "stream batches at a higher level",
                code="BAD_STREAM_INPUT",
            )

        ctx = _ensure_operation_context(op_ctx)
        spec = self._translator.build_embed_spec(
            raw_texts,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
            stream=True,
        )

        try:
            stream_or_result = await self._adapter.embed(spec, ctx=ctx)
            if isinstance(stream_or_result, EmbedResult):
                # Streaming requested but unary returned: yield a single final chunk.
                chunk = EmbedChunk(
                    embeddings=[stream_or_result.embedding],
                    is_final=True,
                    usage=None,
                    model=stream_or_result.model,
                )
                yield self._translator.translate_embed_chunk(
                    chunk,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )
                return

            stream = stream_or_result
            if not hasattr(stream, "__aiter__"):
                raise BadRequest(
                    "adapter.embed did not return a streaming iterator",
                    code="BAD_ADAPTER_RESULT",
                )

            async for chunk in stream:
                if not isinstance(chunk, EmbedChunk):
                    raise BadRequest(
                        f"adapter.embed stream yielded unsupported type: "
                        f"{type(chunk).__name__}",
                        code="BAD_ADAPTER_RESULT",
                    )
                yield self._translator.translate_embed_chunk(
                    chunk,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )
        except Exception as exc:
            attach_context(
                exc,
                framework=self._framework,
                embedding_operation="stream_embed",
                request_id=ctx.request_id,
                tenant=ctx.tenant,
            )
            raise

    async def arun_embed_stream_collect(
        self,
        raw_texts: Any,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Convenience helper: run a streaming embed and return a single
        aggregated embedding-like structure.

        Behavior (generic):
            - Flattens all chunk["embeddings"] vectors into one list
            - Returns final metadata from the last chunk
        """
        vectors: List[List[float]] = []
        last_chunk: Optional[Mapping[str, Any]] = None
        async for chunk in self.arun_embed_stream(
            raw_texts, op_ctx=op_ctx, framework_ctx=framework_ctx
        ):
            if isinstance(chunk, Mapping):
                embeddings = chunk.get("embeddings") or []
                if isinstance(embeddings, list):
                    for v in embeddings:
                        vectors.append(v)
            last_chunk = chunk

        return {
            "embedding": vectors,
            "last_chunk": last_chunk,
        }

    # --------------------------------------------------------------------- #
    # Batch Embed APIs
    # --------------------------------------------------------------------- #

    def batch_embed(
        self,
        raw_batch: Any,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """
        Synchronous batch_embed API (uses AsyncBridge).

        Expects framework-level input that can be translated into a BatchEmbedSpec
        by the configured translator.
        """

        async def _batch_coro() -> Any:
            ctx = _ensure_operation_context(op_ctx)
            try:
                spec = self._translator.build_batch_embed_spec(
                    raw_batch,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )
                result = await self._adapter.embed_batch(spec, ctx=ctx)

                if not isinstance(result, BatchEmbedResult):
                    raise BadRequest(
                        f"adapter.embed_batch returned unsupported type: "
                        f"{type(result).__name__}",
                        code="BAD_ADAPTER_RESULT",
                    )

                return self._translator.translate_batch_embed_result(
                    result,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )
            except Exception as exc:
                attach_context(
                    exc,
                    framework=self._framework,
                    embedding_operation="batch_embed",
                    request_id=ctx.request_id,
                    tenant=ctx.tenant,
                )
                raise

        ctx = _ensure_operation_context(op_ctx)
        timeout = ctx.deadline_ms / 1000.0 if ctx.deadline_ms else None
        return AsyncBridge.run_async(_batch_coro(), timeout=timeout)

    async def arun_batch_embed(
        self,
        raw_batch: Any,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """Async batch_embed API."""
        ctx = _ensure_operation_context(op_ctx)
        try:
            spec = self._translator.build_batch_embed_spec(
                raw_batch,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            result = await self._adapter.embed_batch(spec, ctx=ctx)

            if not isinstance(result, BatchEmbedResult):
                raise BadRequest(
                    f"adapter.embed_batch returned unsupported type: "
                    f"{type(result).__name__}",
                    code="BAD_ADAPTER_RESULT",
                )

            return self._translator.translate_batch_embed_result(
                result,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
        except Exception as exc:
            attach_context(
                exc,
                framework=self._framework,
                embedding_operation="batch_embed",
                request_id=ctx.request_id,
                tenant=ctx.tenant,
            )
            raise

    # --------------------------------------------------------------------- #
    # Stats / Inspection
    # --------------------------------------------------------------------- #

    def get_stats(
        self,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """Synchronous get_stats (uses AsyncBridge)."""

        async def _stats_coro() -> Any:
            ctx = _ensure_operation_context(op_ctx)
            try:
                stats = await self._adapter.get_stats(ctx=ctx)

                if not isinstance(stats, EmbeddingStats):
                    raise BadRequest(
                        f"adapter.get_stats returned unsupported type: "
                        f"{type(stats).__name__}",
                        code="BAD_ADAPTER_RESULT",
                    )

                return self._translator.translate_stats(
                    stats,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )
            except Exception as exc:
                attach_context(
                    exc,
                    framework=self._framework,
                    embedding_operation="get_stats",
                    request_id=ctx.request_id,
                    tenant=ctx.tenant,
                )
                raise

        ctx = _ensure_operation_context(op_ctx)
        timeout = ctx.deadline_ms / 1000.0 if ctx.deadline_ms else None
        return AsyncBridge.run_async(_stats_coro(), timeout=timeout)

    async def arun_get_stats(
        self,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """Async get_stats."""
        ctx = _ensure_operation_context(op_ctx)
        try:
            stats = await self._adapter.get_stats(ctx=ctx)

            if not isinstance(stats, EmbeddingStats):
                raise BadRequest(
                    f"adapter.get_stats returned unsupported type: "
                    f"{type(stats).__name__}",
                    code="BAD_ADAPTER_RESULT",
                )

            return self._translator.translate_stats(
                stats,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
        except Exception as exc:
            attach_context(
                exc,
                framework=self._framework,
                embedding_operation="get_stats",
                request_id=ctx.request_id,
                tenant=ctx.tenant,
            )
            raise


# =============================================================================
# Registry for per-framework translators
# =============================================================================


_TranslatorFactory = Callable[[EmbeddingProtocolV1], EmbeddingFrameworkTranslator]
_EMBEDDING_TRANSLATOR_FACTORIES: Dict[str, _TranslatorFactory] = {}


def register_embedding_translator(
    framework: str,
    factory: _TranslatorFactory,
) -> None:
    """
    Register or override an EmbeddingFrameworkTranslator factory for a given framework.

    Example
    -------
        def make_langchain_translator(
            adapter: EmbeddingProtocolV1,
        ) -> EmbeddingFrameworkTranslator:
            return LangChainEmbeddingTranslator(adapter=adapter)

        register_embedding_translator("langchain", make_langchain_translator)
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
    _EMBEDDING_TRANSLATOR_FACTORIES[framework] = factory
    LOG.debug("Registered embedding translator factory for framework=%s", framework)


def get_embedding_translator_factory(framework: str) -> Optional[_TranslatorFactory]:
    """Return a previously registered translator factory for a framework, if any."""
    return _EMBEDDING_TRANSLATOR_FACTORIES.get(framework)


def create_embedding_translator(
    *,
    adapter: EmbeddingProtocolV1,
    framework: str = "generic",
    translator: Optional[EmbeddingFrameworkTranslator] = None,
    batch_config: Optional[BatchConfig] = None,
    text_normalization_config: Optional[TextNormalizationConfig] = None,
) -> EmbeddingTranslator:
    """
    Convenience helper to construct an EmbeddingTranslator for a given framework.

    Behavior:
        - If `translator` is provided explicitly, it is used as-is.
        - Else, if a factory is registered for `framework`, it is used.
        - Else, DefaultEmbeddingFrameworkTranslator is used with optional
          text normalization.
    """
    if translator is None:
        factory = get_embedding_translator_factory(framework)
        if factory is not None:
            translator = factory(adapter)
        else:
            text_normalizer = None
            if text_normalization_config is not None:
                text_normalizer = TextNormalizer(text_normalization_config)
            translator = DefaultEmbeddingFrameworkTranslator(
                text_normalizer=text_normalizer
            )

    return EmbeddingTranslator(
        adapter=adapter,
        framework=framework,
        translator=translator,
        batch_config=batch_config,
    )


__all__ = [
    "BatchConfig",
    "TextNormalizationConfig",
    "TextNormalizer",
    "EmbeddingFrameworkTranslator",
    "DefaultEmbeddingFrameworkTranslator",
    "EmbeddingTranslator",
    "register_embedding_translator",
    "get_embedding_translator_factory",
    "create_embedding_translator",
]
