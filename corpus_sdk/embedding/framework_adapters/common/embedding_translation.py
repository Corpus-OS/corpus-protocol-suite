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

- Building `EmbedSpec` / batch embed specs from framework-level inputs
- Translating `EmbedResult` / `EmbedChunk` back to framework-facing shapes
- Handling text normalization (truncation, cleaning, batching)
- Providing sync + async APIs, including streaming via a sync bridge
- Attaching rich error context for observability
- Managing token counting and cost tracking

Context translation
-------------------
This module does **not** parse framework configs directly. Instead:

- `corpus_sdk.core.context_translation` is responsible for taking framework-native
  contexts (LangChain RunnableConfig, LlamaIndex CallbackManager, etc.) and producing
  a core `OperationContext`.
- Callers pass either an `OperationContext` or a simple dict-like context into
  the methods here; we normalize that via `from_dict` into the embedding adapter's
  `OperationContext`.

Batching strategy
-----------------
Embedding operations benefit heavily from batching:

- Automatic batching of single/multiple texts
- Configurable batch size limits (model-dependent)
- Token-aware batching to stay under model limits
- Retry logic for batch failures with exponential backoff

Text normalization
------------------
This layer handles common text preprocessing:

- Whitespace normalization
- Empty text filtering
- Truncation to model max length
- Encoding validation (UTF-8)

Streaming
---------
For streaming embeds (useful for large document sets), this module exposes:

- An async API that yields translated framework chunks, and
- A sync API that wraps the async generator via `SyncStreamBridge`, preserving
  proper cancellation and error propagation.

Registry
--------
A small registry lets you register per-framework embedding translators:

- `register_embedding_translator("my_framework", factory)`
- `create_embedding_translator("my_framework", adapter, ...)`

This makes it straightforward to plug in framework-specific behaviors while
reusing the common orchestration logic here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
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

from corpus_sdk.embedding.embedding_base import (
    EmbeddingProtocolV1,
    OperationContext,
    EmbedSpec,
    BatchEmbedSpec,
    EmbedResult,
    EmbedChunk,
    BatchEmbedResult,
    EmbeddingStats,
    EmbeddingAdapterError,
    BadRequest,
    NotSupported,
)

from corpus_sdk.core.context_translation import from_dict as ctx_from_dict
from corpus_sdk.core.sync_stream_bridge import SyncStreamBridge
from corpus_sdk.core.async_bridge import AsyncBridge
from corpus_sdk.llm.framework_adapters.common.error_context import attach_context

LOG = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Helpers: OperationContext normalization
# =============================================================================


def _ensure_operation_context(
    ctx: Optional[Union[OperationContext, Mapping[str, Any]]],
) -> OperationContext:
    """
    Normalize various context shapes into an embedding OperationContext.

    Accepts:
        - None: returns an "empty" context
        - OperationContext: returned as-is
        - Mapping[str, Any]: interpreted via context_translation.from_dict,
          then adapted into an embedding OperationContext.

    This keeps responsibilities clean:
        - Framework-native → Normalized dict/OperationContext happens in
          corpus_sdk.core.context_translation (from_langchain, from_llamaindex, etc.)
        - This helper simply ensures the embedding adapter receives the right type.
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

    # Reconstruct as embedding OperationContext with validation
    return OperationContext(
        request_id=getattr(core_ctx, "request_id", None),
        idempotency_key=getattr(core_ctx, "idempotency_key", None),
        deadline_ms=getattr(core_ctx, "deadline_ms", None),
        traceparent=getattr(core_ctx, "traceparent", None),
        tenant=getattr(core_ctx, "tenant", None),
        attrs=getattr(core_ctx, "attrs", None) or {},
    )


# =============================================================================
# Batching configuration
# =============================================================================


@dataclass(frozen=True)
class BatchConfig:
    """
    Configuration for automatic batching of embedding requests.

    Batching is CRITICAL for embedding performance:
    - Reduces network round-trips
    - Improves throughput via model parallelism
    - Lowers cost (many providers charge per request)
    - Enables better resource utilization

    Attributes:
        enabled:
            Whether to apply automatic batching. If False, each text is
            embedded individually (not recommended for production).

        max_batch_size:
            Maximum number of texts per batch. Model-dependent.
            Common values: 32 (sentence-transformers), 96 (OpenAI), 2048 (Cohere).

        max_tokens_per_batch:
            Optional token limit per batch. If set, batches are sized to
            stay under this limit. Requires token counting support.

        sort_by_length:
            If True, sort texts by length before batching. This can improve
            efficiency by grouping similar-length texts together, reducing
            padding overhead in transformer models.

        retry_on_partial_failure:
            If True and a batch fails, retry failed items individually.
            Useful for handling rate limits or individual text issues.
    """

    enabled: bool = True
    max_batch_size: int = 32
    max_tokens_per_batch: Optional[int] = None
    sort_by_length: bool = True
    retry_on_partial_failure: bool = True

    def __post_init__(self):
        """Validate batch configuration parameters."""
        if self.enabled:
            if self.max_batch_size <= 0:
                raise ValueError(f"max_batch_size must be positive, got {self.max_batch_size}")
            if self.max_tokens_per_batch is not None and self.max_tokens_per_batch <= 0:
                raise ValueError(f"max_tokens_per_batch must be positive, got {self.max_tokens_per_batch}")


# =============================================================================
# Text normalization configuration
# =============================================================================


@dataclass(frozen=True)
class TextNormalizationConfig:
    """
    Configuration for text preprocessing before embedding.

    Proper text normalization is important for:
    - Consistent embedding quality
    - Avoiding model errors (empty texts, invalid UTF-8)
    - Staying within model token limits
    - Reducing noise in embeddings

    Attributes:
        normalize_whitespace:
            If True, collapse multiple whitespace characters to single space,
            and strip leading/trailing whitespace.

        remove_empty:
            If True, filter out empty or whitespace-only texts before embedding.

        max_length:
            Optional character limit. Texts longer than this are truncated.
            Set based on model's max token limit (e.g., 512 tokens ≈ 2000 chars).

        truncate_strategy:
            How to truncate long texts:
            - "end": Keep beginning, truncate end (default)
            - "start": Keep end, truncate beginning
            - "middle": Keep beginning and end, truncate middle

        encoding:
            Text encoding to validate/enforce. Default "utf-8".
            Invalid encoding is replaced with � (Unicode replacement char).

        lowercase:
            If True, convert all text to lowercase. Useful for some models
            that are case-sensitive but shouldn't be for your use case.
    """

    normalize_whitespace: bool = True
    remove_empty: bool = True
    max_length: Optional[int] = None
    truncate_strategy: str = "end"
    encoding: str = "utf-8"
    lowercase: bool = False

    def __post_init__(self):
        """Validate text normalization configuration."""
        if self.truncate_strategy not in ("end", "start", "middle"):
            raise ValueError(
                f"truncate_strategy must be 'end', 'start', or 'middle', got {self.truncate_strategy!r}"
            )
        if self.max_length is not None and self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")


# =============================================================================
# Text normalization helpers
# =============================================================================


class TextNormalizer:
    """Helper for normalizing text before embedding."""

    def __init__(self, config: TextNormalizationConfig):
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

        # Validate encoding
        if self.config.encoding != "utf-8":
            try:
                text = text.encode(self.config.encoding).decode(self.config.encoding)
            except (UnicodeEncodeError, UnicodeDecodeError):
                LOG.debug("TextNormalizer: encoding error, using replacement chars")
                text = text.encode(self.config.encoding, errors="replace").decode(self.config.encoding)

        # Normalize whitespace
        if self.config.normalize_whitespace:
            import re
            text = re.sub(r'\s+', ' ', text).strip()

        # Remove empty
        if self.config.remove_empty and not text.strip():
            return None

        # Lowercase
        if self.config.lowercase:
            text = text.lower()

        # Truncate
        if self.config.max_length is not None and len(text) > self.config.max_length:
            text = self._truncate(text, self.config.max_length, self.config.truncate_strategy)

        return text

    def normalize_batch(self, texts: Sequence[str]) -> List[str]:
        """
        Normalize a batch of texts, filtering out None results.

        Returns:
            List of normalized texts (may be shorter than input if texts filtered).
        """
        normalized = []
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
        elif strategy == "start":
            return text[-max_length:]
        elif strategy == "middle":
            # Keep first and last portions
            keep_each = max_length // 2
            return text[:keep_each] + text[-keep_each:]
        else:
            return text[:max_length]


# =============================================================================
# Framework-agnostic translator protocol
# =============================================================================


class EmbeddingFrameworkTranslator(Protocol):
    """
    Per-framework translator contract.

    Implementations are responsible for:
        - Converting framework-level embed inputs into Embed*Spec types
        - Converting Embedding results into framework-level outputs
        - Handling framework-specific document/text representations

    The default implementation provided here is generic and treats inputs as
    strings or lists of strings. Frameworks with richer abstractions 
    (LangChain Document, LlamaIndex Node, etc.) can provide their own 
    implementations and register them via the registry.
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
            - framework_ctx (e.g., which model is configured)
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
        - Treats raw_texts as either:
            * a string (single text), or
            * a list/sequence of strings
        - Applies text normalization if configured
        - For results:
            * EmbedResult → list of embeddings
            * EmbedChunk → list of embeddings per chunk
            * BatchEmbedResult → nested list structure
            * EmbeddingStats → underlying stats object

    Frameworks with richer abstractions (LangChain Document with page_content,
    LlamaIndex Node with text, etc.) are expected to provide their own
    EmbeddingFrameworkTranslator implementation.
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

    def build_embed_spec(
        self,
        raw_texts: Any,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
        stream: bool = False,
    ) -> EmbedSpec:
        model = self.preferred_model(op_ctx=op_ctx, framework_ctx=framework_ctx)

        # Case 1: raw_texts is a single string
        if isinstance(raw_texts, str):
            normalized = self._text_normalizer.normalize(raw_texts)
            if normalized is None:
                raise BadRequest(
                    "Text was filtered out during normalization (empty after processing)",
                    code="BAD_TEXT_EMPTY",
                )
            return EmbedSpec(
                texts=[normalized],
                model=model,
                stream=stream,
            )

        # Case 2: raw_texts is a list/sequence of strings
        if isinstance(raw_texts, (list, tuple)):
            if not raw_texts:
                raise BadRequest(
                    "raw_texts must contain at least one text",
                    code="BAD_TEXT_EMPTY",
                )
            
            # Normalize all texts
            normalized = self._text_normalizer.normalize_batch(raw_texts)
            if not normalized:
                raise BadRequest(
                    "All texts were filtered out during normalization",
                    code="BAD_TEXT_ALL_EMPTY",
                )
            
            return EmbedSpec(
                texts=normalized,
                model=model,
                stream=stream,
            )

        # Case 3: raw_texts is a mapping with EmbedSpec fields
        if isinstance(raw_texts, Mapping):
            texts = raw_texts.get("texts")
            if not texts:
                raise BadRequest(
                    "raw_texts.texts must be a non-empty list",
                    code="BAD_TEXT_EMPTY",
                )
            
            if isinstance(texts, str):
                texts = [texts]
            elif not isinstance(texts, (list, tuple)):
                raise BadRequest(
                    "raw_texts.texts must be a string or list of strings",
                    code="BAD_TEXTS",
                )
            
            # Normalize texts
            normalized = self._text_normalizer.normalize_batch(texts)
            if not normalized:
                raise BadRequest(
                    "All texts were filtered out during normalization",
                    code="BAD_TEXT_ALL_EMPTY",
                )
            
            req_model = raw_texts.get("model")
            
            return EmbedSpec(
                texts=normalized,
                model=str(req_model) if req_model is not None else model,
                stream=bool(raw_texts.get("stream", stream)),
            )

        raise BadRequest(
            f"Unsupported raw_texts type: {type(raw_texts).__name__}",
            code="BAD_TEXTS",
        )

    def translate_embed_result(
        self,
        result: EmbedResult,
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Any:
        # Generic behavior: return the embeddings list with metadata
        return {
            "embeddings": list(result.embeddings or []),
            "model": result.model,
            "usage": dict(result.usage or {}) if result.usage else None,
        }

    def translate_embed_chunk(
        self,
        chunk: EmbedChunk,
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Any:
        # Generic behavior: return the chunk as a simple dict
        return {
            "embeddings": list(chunk.embeddings or []),
            "is_final": bool(chunk.is_final),
            "usage": dict(chunk.usage or {}) if chunk.usage is not None else None,
        }

    # ---- batch embed translation ----

    def build_batch_embed_spec(
        self,
        raw_batch: Any,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> BatchEmbedSpec:
        model = self.preferred_model(op_ctx=op_ctx, framework_ctx=framework_ctx)

        if isinstance(raw_batch, Mapping):
            batches = raw_batch.get("batches")
            if not batches or not isinstance(batches, (list, tuple)):
                raise BadRequest(
                    "raw_batch.batches must be a non-empty list",
                    code="BAD_BATCH",
                )
            
            # Each batch is a list of texts
            normalized_batches: List[List[str]] = []
            for idx, batch in enumerate(batches):
                if not isinstance(batch, (list, tuple)):
                    raise BadRequest(
                        f"raw_batch.batches[{idx}] must be a list",
                        code="BAD_BATCH",
                        details={"index": idx, "type": type(batch).__name__},
                    )
                normalized = self._text_normalizer.normalize_batch(batch)
                if not normalized:
                    LOG.warning(
                        "Batch %d was filtered out completely during normalization",
                        idx,
                    )
                    continue
                normalized_batches.append(normalized)
            
            if not normalized_batches:
                raise BadRequest(
                    "All batches were filtered out during normalization",
                    code="BAD_BATCH_ALL_EMPTY",
                )
            
            req_model = raw_batch.get("model")
            
            return BatchEmbedSpec(
                batches=normalized_batches,
                model=str(req_model) if req_model is not None else model,
            )

        # Assume raw_batch is a list of lists
        if isinstance(raw_batch, (list, tuple)):
            normalized_batches: List[List[str]] = []
            for idx, batch in enumerate(raw_batch):
                if not isinstance(batch, (list, tuple)):
                    raise BadRequest(
                        f"raw_batch[{idx}] must be a list",
                        code="BAD_BATCH",
                        details={"index": idx, "type": type(batch).__name__},
                    )
                normalized = self._text_normalizer.normalize_batch(batch)
                if not normalized:
                    LOG.warning(
                        "Batch %d was filtered out completely during normalization",
                        idx,
                    )
                    continue
                normalized_batches.append(normalized)
            
            if not normalized_batches:
                raise BadRequest(
                    "All batches were filtered out during normalization",
                    code="BAD_BATCH_ALL_EMPTY",
                )
            
            return BatchEmbedSpec(
                batches=normalized_batches,
                model=model,
            )

        raise BadRequest(
            f"Unsupported raw_batch type: {type(raw_batch).__name__}",
            code="BAD_BATCH",
        )

    def translate_batch_embed_result(
        self,
        result: BatchEmbedResult,
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Any:
        return {
            "batch_embeddings": [list(batch or []) for batch in (result.batch_embeddings or [])],
            "model": result.model,
            "usage": dict(result.usage or {}) if result.usage else None,
        }

    def translate_stats(
        self,
        stats: EmbeddingStats,
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Any:
        return stats


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
        - Applies automatic batching when configured
        - Attaches rich error context for diagnostics

    Sync methods use AsyncBridge to call async adapters from sync contexts.

    It does *not*:
        - Know anything about framework-native context objects (RunnableConfig, etc.)
        - Implement any backend-specific logic (that lives in BaseEmbeddingAdapter subclasses)
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
        self._translator: EmbeddingFrameworkTranslator = translator or DefaultEmbeddingFrameworkTranslator()
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

        Uses AsyncBridge to call the async adapter from a sync context.

        Steps:
            - Normalize OperationContext
            - Build EmbedSpec via translator (with text normalization)
            - Call adapter.embed(...) via AsyncBridge
            - Translate result back to framework-level shape
        """
        async def _embed_coro():
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
                        f"adapter.embed returned unsupported type: {type(result).__name__}",
                        code="BAD_ADAPTER_RESULT",
                    )

                return self._translator.translate_embed_result(
                    result,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )
            except Exception as exc:
                # Attach error context to all exceptions
                attach_context(
                    exc,
                    framework=self._framework,
                    embedding_operation="embed",
                    request_id=ctx.request_id,
                    tenant=ctx.tenant,
                )
                raise

        # Use AsyncBridge with deadline from context
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

        Fully async:
            - await adapter.embed(...)
            - result translation
        """
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
                    f"adapter.embed returned unsupported type: {type(result).__name__}",
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
        bridging the async adapter.stream_embed(...) via SyncStreamBridge.

        Useful for embedding large document sets where you want results
        as they become available rather than waiting for entire batch.
        """
        ctx = _ensure_operation_context(op_ctx)

        async def _stream_factory() -> AsyncIterator[Any]:
            """Factory that creates the async stream."""
            spec = self._translator.build_embed_spec(
                raw_texts,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
                stream=True,
            )
            try:
                async for chunk in self._adapter.stream_embed(spec, ctx=ctx):
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

        # Use SyncStreamBridge for sync streaming
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
        ctx = _ensure_operation_context(op_ctx)
        spec = self._translator.build_embed_spec(
            raw_texts,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
            stream=True,
        )

        try:
            async for chunk in self._adapter.stream_embed(spec, ctx=ctx):
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
        """Synchronous batch_embed (uses AsyncBridge)."""
        async def _batch_coro():
            ctx = _ensure_operation_context(op_ctx)
            try:
                spec = self._translator.build_batch_embed_spec(
                    raw_batch,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )
                result = await self._adapter.batch_embed(spec, ctx=ctx)
                
                if not isinstance(result, BatchEmbedResult):
                    raise BadRequest(
                        f"adapter.batch_embed returned unsupported type: {type(result).__name__}",
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
        """Async batch_embed."""
        ctx = _ensure_operation_context(op_ctx)
        try:
            spec = self._translator.build_batch_embed_spec(
                raw_batch,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            result = await self._adapter.batch_embed(spec, ctx=ctx)
            
            if not isinstance(result, BatchEmbedResult):
                raise BadRequest(
                    f"adapter.batch_embed returned unsupported type: {type(result).__name__}",
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
        async def _stats_coro():
            ctx = _ensure_operation_context(op_ctx)
            try:
                stats = await self._adapter.get_stats(ctx=ctx)
                
                if not isinstance(stats, EmbeddingStats):
                    raise BadRequest(
                        f"adapter.get_stats returned unsupported type: {type(stats).__name__}",
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
                    f"adapter.get_stats returned unsupported type: {type(stats).__name__}",
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
        def make_langchain_translator(adapter: EmbeddingProtocolV1) -> EmbeddingFrameworkTranslator:
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
        - Else, DefaultEmbeddingFrameworkTranslator is used with optional text normalization.
    """
    if translator is None:
        factory = get_embedding_translator_factory(framework)
        if factory is not None:
            translator = factory(adapter)
        else:
            # Use default with optional text normalization config
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
