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
- Applying text normalization (whitespace, encoding, truncation)
- Providing sync + async APIs, including streaming via a sync bridge
- Attaching rich error context for observability
- Passing through token usage / stats data from adapters

Text normalization
------------------
This layer supports configurable text preprocessing:

- Whitespace normalization and cleaning
- Encoding validation and enforcement
- Length truncation with multiple strategies
- Empty text filtering
- Case normalization

Batch handling
--------------
Batch configuration is passed through to adapters, but this layer does not
implement automatic batch splitting. The `BatchConfig` is a hint object
for adapters or higher-level orchestrators to use.

Streaming
---------
For streaming embeddings, this module exposes:

- An async API that yields translated framework chunks, and
- A sync API that wraps the async generator via `SyncStreamBridge`, preserving
  proper cancellation and error propagation.

Note that some adapters may ignore `stream=True` and return unary results;
this layer handles that gracefully by wrapping the result as a single chunk.

Registry
--------
A small registry lets you register per-framework embedding translators:

- `register_embedding_translator("my_framework", factory)`
- `create_embedding_translator("my_framework", adapter, ...)`

This makes it straightforward to plug in framework-specific behaviors while
reusing the common orchestration logic here.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
import threading
from dataclasses import asdict, dataclass, replace
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    Union,
)

from corpus_sdk.core.async_bridge import AsyncBridge
from corpus_sdk.core.context_translation import from_dict as ctx_from_dict
from corpus_sdk.core.error_context import attach_context
from corpus_sdk.core.sync_bridge import SyncStreamBridge
from corpus_sdk.embedding.embedding_base import (
    BadRequest,
    BatchEmbedResult,
    BatchEmbedSpec,
    EmbedChunk,
    EmbedResult,
    EmbedSpec,
    EmbeddingProtocolV1,
    EmbeddingStats,
    OperationContext,
)

LOG = logging.getLogger(__name__)

R = TypeVar("R")


# =============================================================================
# Helpers: OperationContext normalization
# =============================================================================


def _operation_context_kwargs(**kwargs: Any) -> Dict[str, Any]:
    """
    Filter kwargs to only those accepted by OperationContext.__init__.

    This keeps this translation layer compatible across OperationContext
    revisions (e.g., older versions without `metrics`).
    """
    try:
        params = set(inspect.signature(OperationContext).parameters.keys())
    except Exception:  # noqa: BLE001
        # Conservative fallback to the most common stable fields.
        params = {
            "request_id",
            "idempotency_key",
            "deadline_ms",
            "traceparent",
            "tenant",
            "attrs",
        }

    params.discard("self")
    return {k: v for k, v in kwargs.items() if k in params}


def _ensure_operation_context(
    ctx: Optional[Union[OperationContext, Mapping[str, Any]]],
) -> OperationContext:
    """Normalize various context shapes into an embedding OperationContext."""
    # 1) No context → build from empty dict
    if ctx is None:
        core_ctx = ctx_from_dict({})
        return OperationContext(
            **_operation_context_kwargs(
                request_id=getattr(core_ctx, "request_id", None),
                idempotency_key=getattr(core_ctx, "idempotency_key", None),
                deadline_ms=getattr(core_ctx, "deadline_ms", None),
                traceparent=getattr(core_ctx, "traceparent", None),
                tenant=getattr(core_ctx, "tenant", None),
                metrics=getattr(core_ctx, "metrics", None),
                attrs=getattr(core_ctx, "attrs", None) or {},
            )
        )

    # 2) Already our embedding OperationContext → just use it
    if isinstance(ctx, OperationContext):
        return ctx

    # 3) Mapping → go through core context translation
    if isinstance(ctx, Mapping):
        core_ctx = ctx_from_dict(ctx)
        return OperationContext(
            **_operation_context_kwargs(
                request_id=getattr(core_ctx, "request_id", None),
                idempotency_key=getattr(core_ctx, "idempotency_key", None),
                deadline_ms=getattr(core_ctx, "deadline_ms", None),
                traceparent=getattr(core_ctx, "traceparent", None),
                tenant=getattr(core_ctx, "tenant", None),
                metrics=getattr(core_ctx, "metrics", None),
                attrs=getattr(core_ctx, "attrs", None) or {},
            )
        )

    # 4) Duck-typed context
    if hasattr(ctx, "request_id") or hasattr(ctx, "attrs"):
        return OperationContext(
            **_operation_context_kwargs(
                request_id=getattr(ctx, "request_id", None),
                idempotency_key=getattr(ctx, "idempotency_key", None),
                deadline_ms=getattr(ctx, "deadline_ms", None),
                traceparent=getattr(ctx, "traceparent", None),
                tenant=getattr(ctx, "tenant", None),
                metrics=getattr(ctx, "metrics", None),
                attrs=getattr(ctx, "attrs", None) or {},
            )
        )

    # 5) Everything else → hard error
    raise BadRequest(
        f"Unsupported context type: {type(ctx).__name__}",
        code="BAD_OPERATION_CONTEXT",
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

    Important
    ---------
    - `max_tokens_per_batch` requires tokenization support from the underlying
      adapter. Without tokenization hints, this setting may be ignored.
    - Batch splitting based on token limits is typically handled by the
      underlying embedding provider or a dedicated batching layer.
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

        strict_encoding:
            If True, raise an error when text cannot be encoded/decoded with
            the specified encoding. If False, use replacement characters.

        strict_type:
            If True, reject non-string inputs with an error. If False,
            attempt to convert to string.
    """

    normalize_whitespace: bool = True
    remove_empty: bool = True
    max_length: Optional[int] = None
    truncate_strategy: str = "end"
    encoding: str = "utf-8"
    lowercase: bool = False
    strict_encoding: bool = True
    strict_type: bool = True

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

        Raises:
            BadRequest: If strict_type=True and input is not a string,
                        or if strict_encoding=True and encoding fails.
        """
        # Type handling
        if not isinstance(text, str):
            if self.config.strict_type:
                raise BadRequest(
                    f"Text must be a string, got {type(text).__name__}",
                    code="BAD_TEXT_TYPE",
                    details={"type": type(text).__name__},
                )
            else:
                LOG.warning("TextNormalizer: non-string text %r, converting", type(text))
                text = str(text)

        # Validate / enforce encoding
        if self.config.encoding:
            try:
                text.encode(self.config.encoding, errors="strict")
                if not self.config.strict_encoding:
                    text = text.encode(self.config.encoding, errors="replace").decode(self.config.encoding)
            except UnicodeEncodeError:
                if self.config.strict_encoding:
                    raise BadRequest(
                        f"Text cannot be encoded as {self.config.encoding}",
                        code="BAD_TEXT_ENCODING",
                        details={"encoding": self.config.encoding},
                    )
                else:
                    LOG.debug("TextNormalizer: encoding error, using replacement chars")
                    text = text.encode(self.config.encoding, errors="replace").decode(self.config.encoding)

        if self.config.normalize_whitespace:
            text = re.sub(r"\s+", " ", text).strip()

        if self.config.remove_empty and not text.strip():
            return None

        if self.config.lowercase:
            text = text.lower()

        if self.config.max_length is not None and len(text) > self.config.max_length:
            text = self._truncate(text, self.config.max_length, self.config.truncate_strategy)

        return text

    def normalize_batch(self, texts: Sequence[str]) -> List[str]:
        """
        Normalize a batch of texts, filtering out None results.

        Returns:
            List of normalized texts (may be shorter than input if texts filtered).

        Raises:
            BadRequest: If any text fails normalization based on strict settings.
        """
        normalized: List[str] = []
        for idx, text in enumerate(texts):
            try:
                result = self.normalize(text)
                if result is not None:
                    normalized.append(result)
            except BadRequest as e:
                raise BadRequest(
                    f"Text at index {idx} failed normalization: {str(e)}",
                    code=e.code,
                    details={**e.details, "index": idx} if e.details else {"index": idx},
                ) from e
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
            remaining = max_length - (keep_each * 2)
            return text[: keep_each + remaining] + text[-keep_each:]

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

    def translate_stats(
        self,
        stats: EmbeddingStats,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        ...

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

        Returns:
            Model name string, or None if no model is preferred.
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
        self._text_normalizer = text_normalizer or TextNormalizer(TextNormalizationConfig())

    def preferred_model(
        self,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Optional[str]:
        if isinstance(framework_ctx, Mapping):
            model = framework_ctx.get("model")
            if model is not None:
                return str(model)

        attrs = op_ctx.attrs or {}
        model = attrs.get("embedding_model")
        if model is not None:
            return str(model)

        model = attrs.get("model")
        if model is not None:
            return str(model)

        return None

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
        fallback_model = model or ""

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
                model=str(req_model) if req_model is not None else fallback_model,
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
            model=fallback_model,
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
        return {
            "embeddings": [e.vector for e in (chunk.embeddings or [])],
            "model": chunk.model,
            "is_final": bool(chunk.is_final),
            "usage": dict(chunk.usage or {}) if chunk.usage is not None else None,
        }

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
        fallback_model = model or ""

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
            model=str(req_model) if req_model is not None else fallback_model,
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
    is a hint object stored here for adapters/orchestrators to inspect.
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
        self._translator = translator or DefaultEmbeddingFrameworkTranslator()
        self._batch_config = batch_config or BatchConfig()

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    async def _get_supported_models(self) -> Optional[Sequence[str]]:
        """
        Best-effort supported model discovery.

        Uses adapter.acapabilities()/capabilities() if present.
        """
        adapter = self._adapter
        caps = None

        acaps = getattr(adapter, "acapabilities", None)
        if callable(acaps):
            try:
                caps = await acaps()
            except Exception:
                caps = None

        if caps is None:
            caps_fn = getattr(adapter, "capabilities", None)
            if callable(caps_fn):
                try:
                    if asyncio.iscoroutinefunction(caps_fn):
                        caps = await caps_fn()
                    else:
                        caps = await asyncio.to_thread(caps_fn)
                except Exception:
                    caps = None

        if caps is None:
            return None

        if hasattr(caps, "supported_models"):
            sm = getattr(caps, "supported_models")
            if isinstance(sm, (list, tuple)) and all(isinstance(x, str) for x in sm):
                return sm

        if isinstance(caps, Mapping):
            sm = caps.get("supported_models") or caps.get("models")
            if isinstance(sm, (list, tuple)) and all(isinstance(x, str) for x in sm):
                return sm

        return None

    async def _coerce_model_unary(self, spec: EmbedSpec) -> EmbedSpec:
        """
        If spec.model is not supported, deterministically fall back to the first supported model.
        """
        supported = await self._get_supported_models()
        if not supported:
            return spec
        if spec.model in supported:
            return spec

        fallback = supported[0]
        attach_context(
            Exception("model_fallback"),
            framework=self._framework,
            operation="embedding.model_fallback",
            requested_model=spec.model,
            fallback_model=fallback,
        )
        return replace(spec, model=fallback)

    async def _coerce_model_batch(self, spec: BatchEmbedSpec) -> BatchEmbedSpec:
        supported = await self._get_supported_models()
        if not supported:
            return spec
        if spec.model in supported:
            return spec

        fallback = supported[0]
        attach_context(
            Exception("model_fallback"),
            framework=self._framework,
            operation="embedding.model_fallback",
            requested_model=spec.model,
            fallback_model=fallback,
        )
        return replace(spec, model=fallback)

    def _extract_vector_from_translated(self, translated: Any) -> List[float]:
        """
        Best-effort extraction for unary->batch fallback.
        """
        if isinstance(translated, list) and all(isinstance(x, (int, float)) for x in translated):
            return [float(x) for x in translated]

        if isinstance(translated, Mapping):
            emb = translated.get("embedding")
            if isinstance(emb, list) and all(isinstance(x, (int, float)) for x in emb):
                return [float(x) for x in emb]

            embs = translated.get("embeddings")
            if isinstance(embs, list) and embs and isinstance(embs[0], list):
                first = embs[0]
                if all(isinstance(x, (int, float)) for x in first):
                    return [float(x) for x in first]

        raise BadRequest(
            "Unable to coerce unary translated result into embedding vector",
            code="BAD_TRANSLATION_RESULT",
        )

    # --------------------------------------------------------------------- #
    # Internal execution helpers (The "Executor Pattern")
    # --------------------------------------------------------------------- #

    def _run_operation(
        self,
        *,
        op_name: str,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]],
        sync: bool,
        logic: Callable[[OperationContext], Awaitable[R]],
    ) -> Union[R, Awaitable[R]]:
        """
        Centralized execution for non-streaming operations.

        - Normalizes OperationContext
        - Wraps `logic` with consistent error-context attachment
        - Uses AsyncBridge for sync variants, returns coroutine for async variants
        """
        ctx = _ensure_operation_context(op_ctx)

        async def _wrapped() -> R:
            try:
                return await logic(ctx)
            except Exception as exc:
                attach_context(
                    exc,
                    framework=self._framework,
                    operation=f"embedding.{op_name}",
                    request_id=getattr(ctx, "request_id", None),
                    tenant=getattr(ctx, "tenant", None),
                )
                raise

        if sync:
            timeout = getattr(ctx, "deadline_ms", None)
            timeout_s = (timeout / 1000.0) if isinstance(timeout, (int, float)) else None
            return AsyncBridge.run_async(_wrapped(), timeout=timeout_s)
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
            try:
                async for chunk in factory(ctx):
                    yield chunk
            except Exception as exc:
                attach_context(
                    exc,
                    framework=self._framework,
                    operation=f"embedding.{op_name}",
                    request_id=getattr(ctx, "request_id", None),
                    tenant=getattr(ctx, "tenant", None),
                )
                raise

        if sync:
            return SyncStreamBridge(
                coro_factory=_async_gen,
                framework=self._framework,
                error_context={
                    "operation": f"embedding.{op_name}",
                    "request_id": getattr(ctx, "request_id", None),
                    "tenant": getattr(ctx, "tenant", None),
                },
            ).run()
        return _async_gen()

    # --------------------------------------------------------------------- #
    # Embed APIs
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
            - If raw_texts is a list/tuple:
                * If adapter has embed_batch → use batch_embed
                * Else → unary-per-item fallback and return batch-shaped output
        """
        if isinstance(raw_texts, (list, tuple)) and not isinstance(raw_texts, Mapping):
            # Empty list should be a no-op and return empty matrix (frameworks expect this).
            if not raw_texts:
                return {
                    "embeddings": [],
                    "model": None,
                    "total_texts": 0,
                    "total_tokens": 0,
                    "failed_texts": [],
                }

            if hasattr(self._adapter, "embed_batch") and callable(getattr(self._adapter, "embed_batch", None)):
                return self.batch_embed(
                    {"texts": list(raw_texts)},
                    op_ctx=op_ctx,
                    framework_ctx=framework_ctx,
                )

            async def _logic_many(ctx: OperationContext) -> Any:
                vectors: List[List[float]] = []
                for t in list(raw_texts):
                    spec = self._translator.build_embed_spec(
                        t,
                        op_ctx=ctx,
                        framework_ctx=framework_ctx,
                        stream=False,
                    )
                    spec = await self._coerce_model_unary(spec)
                    result = await self._adapter.embed(spec, ctx=ctx)
                    if not isinstance(result, EmbedResult):
                        raise BadRequest(
                            f"adapter.embed returned unsupported type: {type(result).__name__}",
                            code="BAD_ADAPTER_RESULT",
                        )
                    translated = self._translator.translate_embed_result(
                        result,
                        op_ctx=ctx,
                        framework_ctx=framework_ctx,
                    )
                    vectors.append(self._extract_vector_from_translated(translated))

                return {
                    "embeddings": vectors,
                    "model": getattr(result, "model", None) if vectors else None,
                    "total_texts": len(vectors),
                    "total_tokens": 0,
                    "failed_texts": [],
                }

            return self._run_operation(
                op_name="embed_many_unary_fallback",
                op_ctx=op_ctx,
                sync=True,
                logic=_logic_many,
            )

        async def _logic(ctx: OperationContext) -> Any:
            spec = self._translator.build_embed_spec(
                raw_texts,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
                stream=False,
            )
            spec = await self._coerce_model_unary(spec)
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

        return self._run_operation(
            op_name="embed",
            op_ctx=op_ctx,
            sync=True,
            logic=_logic,
        )

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
            - If raw_texts is a list/tuple:
                * If adapter has embed_batch → use arun_batch_embed
                * Else → unary-per-item fallback and return batch-shaped output
        """
        if isinstance(raw_texts, (list, tuple)) and not isinstance(raw_texts, Mapping):
            if not raw_texts:
                return {
                    "embeddings": [],
                    "model": None,
                    "total_texts": 0,
                    "total_tokens": 0,
                    "failed_texts": [],
                }

            if hasattr(self._adapter, "embed_batch") and callable(getattr(self._adapter, "embed_batch", None)):
                return await self.arun_batch_embed(
                    {"texts": list(raw_texts)},
                    op_ctx=op_ctx,
                    framework_ctx=framework_ctx,
                )

            async def _logic_many(ctx: OperationContext) -> Any:
                vectors: List[List[float]] = []
                last_model: Optional[str] = None
                for t in list(raw_texts):
                    spec = self._translator.build_embed_spec(
                        t,
                        op_ctx=ctx,
                        framework_ctx=framework_ctx,
                        stream=False,
                    )
                    spec = await self._coerce_model_unary(spec)
                    result = await self._adapter.embed(spec, ctx=ctx)
                    if not isinstance(result, EmbedResult):
                        raise BadRequest(
                            f"adapter.embed returned unsupported type: {type(result).__name__}",
                            code="BAD_ADAPTER_RESULT",
                        )
                    last_model = result.model
                    translated = self._translator.translate_embed_result(
                        result,
                        op_ctx=ctx,
                        framework_ctx=framework_ctx,
                    )
                    vectors.append(self._extract_vector_from_translated(translated))

                return {
                    "embeddings": vectors,
                    "model": last_model,
                    "total_texts": len(vectors),
                    "total_tokens": 0,
                    "failed_texts": [],
                }

            return await self._run_operation(
                op_name="embed_many_unary_fallback",
                op_ctx=op_ctx,
                sync=False,
                logic=_logic_many,
            )

        async def _logic(ctx: OperationContext) -> Any:
            spec = self._translator.build_embed_spec(
                raw_texts,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
                stream=False,
            )
            spec = await self._coerce_model_unary(spec)
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

        return await self._run_operation(
            op_name="embed",
            op_ctx=op_ctx,
            sync=False,
            logic=_logic,
        )

    # --------------------------------------------------------------------- #
    # Streaming Embed APIs (unchanged)
    # --------------------------------------------------------------------- #

    def _prepare_stream_factory(
        self,
        raw_texts: Any,
        framework_ctx: Any,
    ) -> Callable[[OperationContext], AsyncIterator[Any]]:
        """Shared logic to prepare the stream generator."""
        if isinstance(raw_texts, (list, tuple)) and not isinstance(raw_texts, Mapping):
            raise BadRequest(
                "embed_stream only supports single-text inputs. "
                "For multiple texts:\n"
                "- Use batch_embed() for non-streaming batch embedding\n"
                "- Iterate and call embed_stream() per text for streaming multiple texts",
                code="BAD_STREAM_BATCH",
            )

        async def _factory(ctx: OperationContext) -> AsyncIterator[Any]:
            spec = self._translator.build_embed_spec(
                raw_texts,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
                stream=True,
            )
            spec = await self._coerce_model_unary(spec)
            stream_or_result = await self._adapter.embed(spec, ctx=ctx)

            if isinstance(stream_or_result, EmbedResult):
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

            if not hasattr(stream_or_result, "__aiter__"):
                raise BadRequest(
                    "adapter.embed did not return a streaming iterator",
                    code="BAD_ADAPTER_RESULT",
                )

            async for chunk in stream_or_result:
                if not isinstance(chunk, EmbedChunk):
                    raise BadRequest(
                        f"adapter.embed stream yielded unsupported type: {type(chunk).__name__}",
                        code="BAD_ADAPTER_RESULT",
                    )
                yield self._translator.translate_embed_chunk(
                    chunk,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )

        return _factory

    def embed_stream(
        self,
        raw_texts: Any,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Iterator[Any]:
        return self._run_stream_operation(
            op_name="embed_stream",
            op_ctx=op_ctx,
            sync=True,
            factory=self._prepare_stream_factory(raw_texts, framework_ctx),
        )

    def arun_embed_stream(
        self,
        raw_texts: Any,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> AsyncIterator[Any]:
        return self._run_stream_operation(
            op_name="embed_stream",
            op_ctx=op_ctx,
            sync=False,
            factory=self._prepare_stream_factory(raw_texts, framework_ctx),
        )

    async def arun_embed_stream_collect(
        self,
        raw_texts: Any,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Dict[str, Any]:
        vectors: List[List[float]] = []
        last_chunk: Optional[Mapping[str, Any]] = None

        async for chunk in self.arun_embed_stream(
            raw_texts,
            op_ctx=op_ctx,
            framework_ctx=framework_ctx,
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
        async def _logic(ctx: OperationContext) -> Any:
            spec = self._translator.build_batch_embed_spec(
                raw_batch,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            spec = await self._coerce_model_batch(spec)
            result = await self._adapter.embed_batch(spec, ctx=ctx)

            if not isinstance(result, BatchEmbedResult):
                raise BadRequest(
                    f"adapter.embed_batch returned unsupported type: {type(result).__name__}",
                    code="BAD_ADAPTER_RESULT",
                )

            return self._translator.translate_batch_embed_result(
                result,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

        return self._run_operation(
            op_name="batch_embed",
            op_ctx=op_ctx,
            sync=True,
            logic=_logic,
        )

    async def arun_batch_embed(
        self,
        raw_batch: Any,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        async def _logic(ctx: OperationContext) -> Any:
            spec = self._translator.build_batch_embed_spec(
                raw_batch,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            spec = await self._coerce_model_batch(spec)
            result = await self._adapter.embed_batch(spec, ctx=ctx)

            if not isinstance(result, BatchEmbedResult):
                raise BadRequest(
                    f"adapter.embed_batch returned unsupported type: {type(result).__name__}",
                    code="BAD_ADAPTER_RESULT",
                )

            return self._translator.translate_batch_embed_result(
                result,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

        return await self._run_operation(
            op_name="batch_embed",
            op_ctx=op_ctx,
            sync=False,
            logic=_logic,
        )

    # --------------------------------------------------------------------- #
    # Capabilities / Health (unchanged)
    # --------------------------------------------------------------------- #

    def capabilities(
        self,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Mapping[str, Any]:
        async def _logic(ctx: OperationContext) -> Mapping[str, Any]:
            caps = getattr(self._adapter, "capabilities", None)
            acaps = getattr(self._adapter, "acapabilities", None)

            if callable(acaps):
                result = await acaps()  # type: ignore[no-any-return]
            elif callable(caps):
                if asyncio.iscoroutinefunction(caps):
                    result = await caps()  # type: ignore[no-any-return]
                else:
                    result = caps()  # type: ignore[no-any-return]
            else:
                return {}

            if not isinstance(result, Mapping):
                raise BadRequest(
                    f"adapter.capabilities returned unsupported type: {type(result).__name__}",
                    code="BAD_ADAPTER_RESULT",
                )
            return dict(result)

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
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Mapping[str, Any]:
        async def _logic(ctx: OperationContext) -> Mapping[str, Any]:
            caps = getattr(self._adapter, "capabilities", None)
            acaps = getattr(self._adapter, "acapabilities", None)

            if callable(acaps):
                result = await acaps()  # type: ignore[no-any-return]
            elif callable(caps):
                if asyncio.iscoroutinefunction(caps):
                    result = await caps()  # type: ignore[no-any-return]
                else:
                    result = await asyncio.to_thread(caps)  # type: ignore[arg-type]
            else:
                return {}

            if not isinstance(result, Mapping):
                raise BadRequest(
                    f"adapter.capabilities returned unsupported type: {type(result).__name__}",
                    code="BAD_ADAPTER_RESULT",
                )
            return dict(result)

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
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Mapping[str, Any]:
        async def _logic(ctx: OperationContext) -> Mapping[str, Any]:
            health = getattr(self._adapter, "health", None)
            ahealth = getattr(self._adapter, "ahealth", None)

            if callable(ahealth):
                result = await ahealth()  # type: ignore[no-any-return]
            elif callable(health):
                if asyncio.iscoroutinefunction(health):
                    result = await health()  # type: ignore[no-any-return]
                else:
                    result = health()  # type: ignore[no-any-return]
            else:
                return {}

            if not isinstance(result, Mapping):
                raise BadRequest(
                    f"adapter.health returned unsupported type: {type(result).__name__}",
                    code="BAD_ADAPTER_RESULT",
                )
            return dict(result)

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
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Mapping[str, Any]:
        async def _logic(ctx: OperationContext) -> Mapping[str, Any]:
            health = getattr(self._adapter, "health", None)
            ahealth = getattr(self._adapter, "ahealth", None)

            if callable(ahealth):
                result = await ahealth()  # type: ignore[no-any-return]
            elif callable(health):
                if asyncio.iscoroutinefunction(health):
                    result = await health()  # type: ignore[no-any-return]
                else:
                    result = await asyncio.to_thread(health)  # type: ignore[arg-type]
            else:
                return {}

            if not isinstance(result, Mapping):
                raise BadRequest(
                    f"adapter.health returned unsupported type: {type(result).__name__}",
                    code="BAD_ADAPTER_RESULT",
                )
            return dict(result)

        return await self._run_operation(
            op_name="health",
            op_ctx=op_ctx,
            sync=False,
            logic=_logic,
        )

    # --------------------------------------------------------------------- #
    # Resource Management / Cleanup (unchanged)
    # --------------------------------------------------------------------- #

    async def aclose(
        self,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> None:
        async def _logic(ctx: OperationContext) -> None:  # noqa: ARG001
            adapter = self._adapter
            aclose = getattr(adapter, "aclose", None)
            if callable(aclose):
                await aclose()
                return

            close = getattr(adapter, "close", None)
            if not callable(close):
                return

            if asyncio.iscoroutinefunction(close):
                await close()
            else:
                await asyncio.to_thread(close)

        await self._run_operation(
            op_name="close",
            op_ctx=op_ctx,
            sync=False,
            logic=_logic,
        )

    def close(
        self,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> None:
        async def _logic(ctx: OperationContext) -> None:  # noqa: ARG001
            adapter = self._adapter
            aclose = getattr(adapter, "aclose", None)
            if callable(aclose):
                await aclose()
                return

            close = getattr(adapter, "close", None)
            if not callable(close):
                return

            if asyncio.iscoroutinefunction(close):
                await close()
            else:
                await asyncio.to_thread(close)

        self._run_operation(
            op_name="close",
            op_ctx=op_ctx,
            sync=True,
            logic=_logic,
        )

    # --------------------------------------------------------------------- #
    # Stats / Inspection (unchanged)
    # --------------------------------------------------------------------- #

    def get_stats(
        self,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        async def _logic(ctx: OperationContext) -> Any:
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

        return self._run_operation(
            op_name="get_stats",
            op_ctx=op_ctx,
            sync=True,
            logic=_logic,
        )

    async def arun_get_stats(
        self,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        async def _logic(ctx: OperationContext) -> Any:
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

        return await self._run_operation(
            op_name="get_stats",
            op_ctx=op_ctx,
            sync=False,
            logic=_logic,
        )


# =============================================================================
# Registry for per-framework translators
# =============================================================================


_TranslatorFactory = Callable[[EmbeddingProtocolV1], EmbeddingFrameworkTranslator]
_EMBEDDING_TRANSLATOR_FACTORIES: Dict[str, _TranslatorFactory] = {}
_REGISTRY_LOCK = threading.Lock()


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

    with _REGISTRY_LOCK:
        _EMBEDDING_TRANSLATOR_FACTORIES[framework] = factory
    LOG.debug("Registered embedding translator factory for framework=%s", framework)


def get_embedding_translator_factory(framework: str) -> Optional[_TranslatorFactory]:
    """Return a previously registered translator factory for a framework, if any."""
    with _REGISTRY_LOCK:
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
            translator = DefaultEmbeddingFrameworkTranslator(text_normalizer=text_normalizer)

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
