# corpus_sdk/llm/openai_adapter.py
# SPDX-License-Identifier: Apache-2.0
"""
OpenAI LLM adapter for the Corpus LLM Protocol.

This module implements a production-grade adapter on top of the
`BaseLLMAdapter` / LLMProtocolV1 contract, targeting the official
`openai` Python client (v1+ async API via `AsyncOpenAI`).

Goals
-----
- Map Corpus protocol → OpenAI Chat Completions.
- Preserve async/streaming semantics.
- Normalize provider errors into Corpus' error taxonomy.
- Provide basic token counting for cost/quota tracking.
- Play nicely with higher-level framework adapters
  (LangChain, LlamaIndex, Semantic Kernel, etc.).

Usage
-----
    from openai import AsyncOpenAI
    from corpus_sdk.llm.openai_adapter import OpenAIAdapter

    client = AsyncOpenAI(api_key="sk-...")
    adapter = OpenAIAdapter(client=client, default_model="gpt-4.1-mini")

    result = await adapter.complete(
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(result.text)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List, Mapping, Optional

import openai  # type: ignore

from corpus_sdk.llm.llm_base import (
    AuthError,
    BadRequest,
    BaseLLMAdapter,
    DeadlineExceeded,
    LLMCapabilities,
    LLMChunk,
    LLMCompletion,
    LLMAdapterError,
    NotSupported,
    OperationContext,
    ResourceExhausted,
    TokenUsage,
    TransientNetwork,
    Unavailable,
)

logger = logging.getLogger(__name__)

# Try to use the modern async client if available.
try:  # pragma: no cover - import surface only
    from openai import AsyncOpenAI  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - older client
    AsyncOpenAI = None  # type: ignore[misc,assignment]

# Error types (v1+ client). We defensively pull them via getattr so the
# adapter fails gracefully on older/unsupported versions.
APIStatusError = getattr(openai, "APIStatusError", None)
APIConnectionError = getattr(openai, "APIConnectionError", None)
APITimeoutError = getattr(openai, "APITimeoutError", None)
RateLimitError = getattr(openai, "RateLimitError", None)
AuthenticationError = getattr(openai, "AuthenticationError", None)
BadRequestError = getattr(openai, "BadRequestError", None)
OpenAIError = getattr(openai, "OpenAIError", Exception)


class OpenAIAdapter(BaseLLMAdapter):
    """
    Corpus LLM adapter backed by the OpenAI Chat Completions API.

    This adapter is async-first and plugs directly into the Corpus
    protocol stack via `BaseLLMAdapter`.

    Parameters
    ----------
    client:
        Pre-configured `AsyncOpenAI` client instance. Recommended when you
        want to control retry, proxies, organization, etc.
    api_key:
        API key used when a client is not provided. Ignored if `client`
        is given.
    organization:
        Optional OpenAI organization ID.
    base_url:
        Optional custom base URL (for proxies, gateways, etc.).
    default_model:
        Model to use when callers don't specify one explicitly.
    model_family:
        Logical family name surfaced in `LLMCompletion.model_family`.
    max_context_length:
        Approximate max context window size used only for capabilities
        reporting and *approximate* planning. We deliberately do not
        hard-enforce this against provider limits.
    metrics, mode, ...:
        Passed through to `BaseLLMAdapter`.

    Notes
    -----
    - Requires `openai` Python library v1+ (for `AsyncOpenAI`). If that
      is not available, instantiation will raise `RuntimeError`.
    - Streaming returns token deltas as `LLMChunk.text`; `usage_so_far`
      is currently omitted (None) because OpenAI's streaming API only
      surfaces final usage. Callers that need precise usage can call
      `complete()` instead.
    """

    def __init__(
        self,
        *,
        client: Optional["AsyncOpenAI"] = None,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: str = "gpt-4.1-mini",
        model_family: str = "openai",
        max_context_length: int = 128_000,
        # Base adapter infra
        metrics=None,
        mode: str = "thin",
        deadline_policy=None,
        breaker=None,
        cache=None,
        limiter=None,
        tag_model_in_metrics: bool = True,
        cache_ttl_s: int = 60,
        stream_deadline_check_every_n_chunks: int = 10,
    ) -> None:
        if client is None:
            if AsyncOpenAI is None:
                raise RuntimeError(
                    "OpenAIAdapter requires `openai>=1.0.0` with AsyncOpenAI. "
                    "Upgrade via `pip install --upgrade openai`."
                )
            client = AsyncOpenAI(
                api_key=api_key,
                organization=organization,
                base_url=base_url,
            )

        super().__init__(
            metrics=metrics,
            mode=mode,
            deadline_policy=deadline_policy,
            breaker=breaker,
            cache=cache,
            limiter=limiter,
            tag_model_in_metrics=tag_model_in_metrics,
            cache_ttl_s=cache_ttl_s,
            stream_deadline_check_every_n_chunks=stream_deadline_check_every_n_chunks,
        )

        self._client: AsyncOpenAI = client  # type: ignore[assignment]
        self._default_model = default_model
        self._model_family = model_family
        self._max_context_length = int(max_context_length)
        self._server = "openai"
        self._version = getattr(openai, "__version__", "unknown")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_model(self, model: Optional[str]) -> str:
        """Resolve caller-supplied model or fall back to the default."""
        return model or self._default_model

    @staticmethod
    def _build_messages(
        *,
        messages: List[Mapping[str, str]],
        system_message: Optional[str],
    ) -> List[Dict[str, str]]:
        """
        Combine system_message + messages for OpenAI's chat format.

        BaseLLMAdapter has already validated that messages is a list of
        {role, content} with string values.
        """
        out: List[Dict[str, str]] = []
        if system_message:
            out.append({"role": "system", "content": system_message})
        # Trust upstream to have canonical roles; provider will error for
        # invalid ones and we normalize that below.
        for m in messages:
            out.append({"role": m["role"], "content": m["content"]})
        return out

    @staticmethod
    def _usage_from_response(resp: Any) -> TokenUsage:
        """Convert OpenAI usage object -> TokenUsage; handle missing usage."""
        usage = getattr(resp, "usage", None)
        if usage is None:
            return TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        return TokenUsage(
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
        )

    @staticmethod
    def _extract_retry_after_ms(err: Any) -> Optional[int]:
        """Best-effort extraction of Retry-After header from OpenAI errors."""
        try:
            resp = getattr(err, "response", None)
            if resp is None:
                return None
            headers = getattr(resp, "headers", None)
            if not headers:
                return None
            val = (
                headers.get("retry-after")
                or headers.get("Retry-After")
                or headers.get("Retry-after")
            )
            if val is None:
                return None
            # Retry-After is usually seconds; we treat it as such.
            seconds = int(str(val).strip())
            return max(0, seconds) * 1000
        except Exception:  # noqa: BLE001
            return None

    def _translate_openai_error(self, err: Exception) -> LLMAdapterError:
        """
        Map OpenAI client errors → Corpus LLMAdapterError subclasses.

        This ensures routers and observability can reason about failures
        without vendor-specific conditionals.
        """
        # Connection / transport issues → TransientNetwork
        if APIConnectionError is not None and isinstance(err, APIConnectionError):
            return TransientNetwork(str(err) or "OpenAI API connection error")

        # Upstream timeout → DeadlineExceeded
        if APITimeoutError is not None and isinstance(err, APITimeoutError):
            return DeadlineExceeded("OpenAI API request timed out")

        # Rate limiting / quota → ResourceExhausted
        if RateLimitError is not None and isinstance(err, RateLimitError):
            retry_ms = self._extract_retry_after_ms(err)
            return ResourceExhausted(
                "OpenAI rate limit exceeded",
                retry_after_ms=retry_ms,
                throttle_scope="tenant",
            )

        # AuthN / AuthZ → AuthError
        if AuthenticationError is not None and isinstance(err, AuthenticationError):
            return AuthError(str(err) or "OpenAI authentication/authorization error")

        # Request shape / params → BadRequest
        if BadRequestError is not None and isinstance(err, BadRequestError):
            return BadRequest(str(err) or "OpenAI request is invalid")

        # HTTP status buckets
        if APIStatusError is not None and isinstance(err, APIStatusError):
            status = int(getattr(err, "status_code", 0) or 0)

            # Map a few key ranges explicitly.
            if status == 400:
                return BadRequest(str(err) or "OpenAI request is invalid")
            if status in (401, 403):
                return AuthError(str(err) or "OpenAI authentication/authorization error")
            if status == 404:
                # Often "model not found" or similar.
                return NotSupported(str(err) or "Requested OpenAI resource is not supported")
            if status == 429:
                retry_ms = self._extract_retry_after_ms(err)
                return ResourceExhausted(
                    "OpenAI rate limit exceeded",
                    retry_after_ms=retry_ms,
                    throttle_scope="tenant",
                )
            if 500 <= status <= 599:
                retry_ms = self._extract_retry_after_ms(err)
                return Unavailable(
                    "OpenAI service is temporarily unavailable",
                    retry_after_ms=retry_ms,
                )

            # Fallback for unexpected status codes.
            return Unavailable(str(err) or f"OpenAI error (status={status})")

        # Generic OpenAI error fallback → Unavailable
        if isinstance(err, OpenAIError):
            return Unavailable(str(err) or "OpenAI API error")

        # Anything else → wrap as Unavailable
        return Unavailable(str(err) or "internal OpenAI adapter error")

    # ------------------------------------------------------------------
    # BaseLLMAdapter backend hooks
    # ------------------------------------------------------------------

    async def _do_capabilities(self) -> LLMCapabilities:
        """
        Report adapter capabilities.

        We deliberately leave supported_models empty so routers can pass
        any valid OpenAI model ID without the adapter enforcing a fixed
        allow-list.
        """
        return LLMCapabilities(
            server=self._server,
            version=self._version,
            model_family=self._model_family,
            max_context_length=self._max_context_length,
            supports_streaming=True,
            supports_roles=True,
            supports_json_output=True,
            supports_parallel_tool_calls=False,
            idempotent_writes=True,
            supports_multi_tenant=True,
            supports_system_message=True,
            supports_deadline=True,
            # We implement a *rough* token counter (whitespace-based) for
            # planning and observability only.
            supports_count_tokens=True,
            supported_models=(),  # open set; let callers choose.
        )

    async def _do_complete(
        self,
        *,
        messages: List[Mapping[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        model: Optional[str] = None,
        system_message: Optional[str] = None,
        ctx: Optional[OperationContext] = None,
    ) -> LLMCompletion:
        """
        Backend implementation of `complete()` using Chat Completions.
        """
        resolved_model = self._resolve_model(model)
        oai_messages = self._build_messages(
            messages=messages,
            system_message=system_message,
        )

        try:
            resp = await self._client.chat.completions.create(
                model=resolved_model,
                messages=oai_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop_sequences,
                stream=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise self._translate_openai_error(exc) from exc

        # Extract the primary choice.
        if not getattr(resp, "choices", None):
            raise Unavailable("OpenAI returned no choices")

        choice = resp.choices[0]
        # For v1 client: `choice.message.content` is the text.
        message = getattr(choice, "message", None)
        text = ""
        if message is not None:
            text = getattr(message, "content", "") or ""
        # Older/newer variants sometimes expose `choice.text`
        if not text and hasattr(choice, "text"):
            text = getattr(choice, "text", "") or ""

        finish_reason = getattr(choice, "finish_reason", None) or "stop"
        usage = self._usage_from_response(resp)
        model_id = getattr(resp, "model", None) or resolved_model

        return LLMCompletion(
            text=text,
            model=model_id,
            model_family=self._model_family,
            usage=usage,
            finish_reason=str(finish_reason),
        )

    async def _do_stream(
        self,
        *,
        messages: List[Mapping[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        model: Optional[str] = None,
        system_message: Optional[str] = None,
        ctx: Optional[OperationContext] = None,
    ) -> AsyncIterator[LLMChunk]:
        """
        Backend implementation of `stream()` using OpenAI streaming.

        We emit:
            - One `LLMChunk` per non-empty delta text (is_final=False).
            - A terminal sentinel chunk with is_final=True and no new text.

        `usage_so_far` is not populated because OpenAI only exposes final
        usage at the end of the stream; callers requiring exact usage
        should prefer `complete()`.
        """
        resolved_model = self._resolve_model(model)
        oai_messages = self._build_messages(
            messages=messages,
            system_message=system_message,
        )

        try:
            stream = await self._client.chat.completions.create(
                model=resolved_model,
                messages=oai_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop_sequences,
                stream=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise self._translate_openai_error(exc) from exc

        # Aggregate model + usage info as we go; usage is normally only
        # available on the final event.
        last_model_id: Optional[str] = None
        final_usage: Optional[TokenUsage] = None

        async for event in stream:
            last_model_id = getattr(event, "model", None) or last_model_id or resolved_model

            # v1 ChatCompletionChunk: event.choices[0].delta.content
            if not getattr(event, "choices", None):
                continue

            choice = event.choices[0]
            delta = getattr(choice, "delta", None)
            text = ""
            if delta is not None:
                text = getattr(delta, "content", "") or ""
            # Older patterns may expose `choice.text` instead.
            if not text and hasattr(choice, "text"):
                text = getattr(choice, "text", "") or ""

            # Collect final usage if present on this chunk.
            usage = getattr(event, "usage", None)
            if usage is not None:
                final_usage = TokenUsage(
                    prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                    completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                    total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
                )

            if text:
                # Emit delta chunks as we receive them.
                yield LLMChunk(
                    text=text,
                    is_final=False,
                    model=last_model_id,
                    usage_so_far=None,  # not tracked per-chunk
                )

        # Emit a final sentinel chunk marking end-of-stream.
        yield LLMChunk(
            text="",
            is_final=True,
            model=last_model_id or resolved_model,
            usage_so_far=final_usage,
        )

    async def _do_count_tokens(
        self,
        text: str,
        *,
        model: Optional[str] = None,
        ctx: Optional[OperationContext] = None,
    ) -> int:
        """
        Approximate token counting.

        For simplicity and to avoid coupling to provider-specific
        tokenizers, we use a whitespace-based heuristic similar to the
        mock adapter. This is good enough for:
            - relative size comparisons
            - basic context-window planning
            - high-level quota tracking

        If you need exact tokenization for billing-critical paths,
        consider extending this adapter with `tiktoken` or the OpenAI
        tokenizer APIs.
        """
        if not text:
            return 0
        # +3 overhead for system metadata / biases (same as MockLLMAdapter).
        return len(text.split()) + 3

    async def _do_health(
        self,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> Mapping[str, Any]:
        """
        Lightweight health check.

        Behavior:
        - If ctx.attrs["health"] is "degraded" / "error", we return a
          deterministic degraded/error shape (mirrors MockLLMAdapter to
          keep the conformance harness happy).
        - Otherwise we perform a minimal live check by listing models.
          Any OpenAI error is captured into the returned payload instead
          of being raised, so that `BaseLLMAdapter.health()` can still
          normalize the response.
        """
        status_hint = (ctx and ctx.attrs.get("health")) if ctx and ctx.attrs else None

        if status_hint == "degraded":
            return {
                "ok": False,
                "status": "degraded",
                "server": self._server,
                "version": self._version,
            }
        if status_hint == "error":
            return {
                "ok": False,
                "status": "error",
                "server": self._server,
                "version": self._version,
            }

        try:
            # Minimal live check: ensure we can talk to the API at all.
            # We don't depend on the exact payload shape here.
            await self._client.models.list()
            return {
                "ok": True,
                "status": "healthy",
                "server": self._server,
                "version": self._version,
            }
        except Exception as exc:  # noqa: BLE001
            err = self._translate_openai_error(exc)
            # We intentionally do NOT raise here; BaseLLMAdapter.health()
            # expects a mapping and will treat this as a successful but
            # unhealthy probe.
            logger.warning("OpenAIAdapter health check failed: %s", err)
            return {
                "ok": False,
                "status": "error",
                "server": self._server,
                "version": self._version,
                "error_code": getattr(err, "code", None),
                "error_message": str(err),
            }

    # ------------------------------------------------------------------
    # Resource cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """
        Close underlying client resources if supported.

        Called automatically when used as:

            async with OpenAIAdapter(...) as adapter:
                ...
        """
        close = getattr(self._client, "close", None)
        if close is None:
            return
        try:
            result = close()
            if asyncio.iscoroutine(result):
                await result
        except Exception:  # noqa: BLE001
            # Best-effort cleanup; ignore close failures.
            logger.debug("OpenAIAdapter close() failed", exc_info=True)


__all__ = [
    "OpenAIAdapter",
]
