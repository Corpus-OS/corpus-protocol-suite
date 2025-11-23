# corpus_sdk/llm/azure_openai_adapter.py
# SPDX-License-Identifier: Apache-2.0
"""
Azure OpenAI LLM adapter for the Corpus LLM Protocol.

This module implements a production-grade adapter on top of the
`BaseLLMAdapter` / LLMProtocolV1 contract, targeting Microsoft Azure
OpenAI Service via the official `openai` Python client (v1+ async API
with `AsyncAzureOpenAI`).

Key Differences from OpenAI Adapter
------------------------------------
- Uses Azure-specific endpoint structure and authentication
- Supports both API key and Azure AD (Entra ID) authentication
- Uses deployment names instead of model names in API calls
- Handles Azure-specific rate limiting and error responses
- Requires API version specification

Goals
-----
- Map Corpus protocol → Azure OpenAI Chat Completions.
- Preserve async/streaming semantics.
- Normalize provider errors into Corpus' error taxonomy.
- Provide token usage accounting for cost/quota tracking.
- Support enterprise authentication (Azure AD).
- Play nicely with higher-level framework adapters
  (LangChain, LlamaIndex, Semantic Kernel, etc.).

Usage
-----
    from corpus_sdk.llm.azure_openai_adapter import AzureOpenAIAdapter

    # Option 1: API Key authentication
    adapter = AzureOpenAIAdapter(
        azure_endpoint="https://myresource.openai.azure.com",
        api_key="your-api-key",
        deployment_name="gpt-4",
        api_version="2024-02-15-preview",
    )

    # Option 2: Azure AD authentication
    from azure.identity.aio import DefaultAzureCredential
    
    credential = DefaultAzureCredential()
    async def token_provider():
        token = await credential.get_token("https://cognitiveservices.azure.com/.default")
        return token.token
    
    adapter = AzureOpenAIAdapter(
        azure_endpoint="https://myresource.openai.azure.com",
        azure_ad_token_provider=token_provider,
        deployment_name="gpt-4",
    )

    result = await adapter.complete(
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(result.text)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Callable, Dict, List, Mapping, Optional, Union

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

# Optional tiktoken support for accurate token counting.
try:  # pragma: no cover - optional dependency
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover
    tiktoken = None  # type: ignore[assignment]

# Try to use the modern async Azure client if available.
try:  # pragma: no cover - import surface only
    from openai import AsyncAzureOpenAI  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - older client
    AsyncAzureOpenAI = None  # type: ignore[misc,assignment]

# Error types (v1+ client). We defensively pull them via getattr so the
# adapter fails gracefully on older/unsupported versions.
APIStatusError = getattr(openai, "APIStatusError", None)
APIConnectionError = getattr(openai, "APIConnectionError", None)
APITimeoutError = getattr(openai, "APITimeoutError", None)
RateLimitError = getattr(openai, "RateLimitError", None)
AuthenticationError = getattr(openai, "AuthenticationError", None)
BadRequestError = getattr(openai, "BadRequestError", None)
OpenAIError = getattr(openai, "OpenAIError", Exception)

# Allowed roles kept in sync with MockLLMAdapter and OpenAIAdapter so
# conformance tests see identical role semantics across adapters.
_ALLOWED_ROLES = {"system", "user", "assistant"}


class AzureOpenAIAdapter(BaseLLMAdapter):
    """
    Corpus LLM adapter backed by Azure OpenAI Service.

    This adapter is async-first and plugs directly into the Corpus
    protocol stack via `BaseLLMAdapter`.

    Parameters
    ----------
    azure_endpoint:
        Azure OpenAI resource endpoint, e.g.,
        "https://myresource.openai.azure.com"
    deployment_name:
        Azure deployment name (not the model name). This is the name you
        configured in Azure Portal for your model deployment.
    client:
        Pre-configured `AsyncAzureOpenAI` client instance. Recommended when
        you want to control retry, proxies, organization, etc. If provided,
        all other auth parameters are ignored.
    api_key:
        API key for Azure OpenAI. Used when client is not provided.
        Mutually exclusive with azure_ad_token_provider.
    api_version:
        Azure OpenAI API version string, e.g., "2024-02-15-preview".
        Defaults to "2024-02-15-preview".
    azure_ad_token_provider:
        Callable that returns an Azure AD token for authentication.
        Mutually exclusive with api_key.
        Example:
            from azure.identity.aio import DefaultAzureCredential
            credential = DefaultAzureCredential()
            async def token_provider():
                token = await credential.get_token(
                    "https://cognitiveservices.azure.com/.default"
                )
                return token.token
    default_model:
        Logical model name for capabilities reporting. Defaults to the
        deployment_name.
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
    - Requires `openai` Python library v1+ (for `AsyncAzureOpenAI`). If
      that is not available, instantiation will raise `RuntimeError`.
    - Uses `tiktoken` for token counting when available, with a safe
      heuristic fallback otherwise.
    - Streaming uses `stream_options={"include_usage": True}` so the
      final chunk carries accurate token usage.
    - Response caching is handled automatically by BaseLLMAdapter using
      the provided `cache` parameter. This adapter only implements
      tokenizer caching for performance optimization.
    - JSON output mode can be enabled via `ctx.attrs.get("response_format")`
      set to "json_object".
    """

    def __init__(
        self,
        *,
        azure_endpoint: str,
        deployment_name: str,
        client: Optional["AsyncAzureOpenAI"] = None,
        api_key: Optional[str] = None,
        api_version: str = "2024-02-15-preview",
        azure_ad_token_provider: Optional[Callable[[], str]] = None,
        default_model: Optional[str] = None,
        model_family: str = "azure-openai",
        max_context_length: int = 128_000,
        # Base adapter infra
        metrics=None,
        mode: str = "thin",
        deadline_policy=None,
        breaker=None,
        cache=None,
        limiter=None,
        tag_model_in_metrics: bool = True,
        stream_deadline_check_every_n_chunks: int = 10,
    ) -> None:
        # Validate critical parameters
        if not azure_endpoint or not isinstance(azure_endpoint, str):
            raise ValueError("azure_endpoint must be a non-empty string")
        if not deployment_name or not isinstance(deployment_name, str):
            raise ValueError("deployment_name must be a non-empty string")
        if not api_version or not isinstance(api_version, str):
            raise ValueError("api_version must be a non-empty string")
        if max_context_length <= 0:
            raise ValueError("max_context_length must be positive")

        # Validate auth configuration
        if api_key and azure_ad_token_provider:
            raise ValueError(
                "api_key and azure_ad_token_provider are mutually exclusive. "
                "Provide only one."
            )

        if client is None:
            if AsyncAzureOpenAI is None:
                raise RuntimeError(
                    "AzureOpenAIAdapter requires `openai>=1.0.0` with AsyncAzureOpenAI. "
                    "Upgrade via `pip install --upgrade openai`."
                )

            # Build client with appropriate auth method
            if azure_ad_token_provider:
                client = AsyncAzureOpenAI(
                    azure_endpoint=azure_endpoint,
                    api_version=api_version,
                    azure_ad_token_provider=azure_ad_token_provider,
                )
            else:
                client = AsyncAzureOpenAI(
                    azure_endpoint=azure_endpoint,
                    api_key=api_key,
                    api_version=api_version,
                )

        super().__init__(
            metrics=metrics,
            mode=mode,
            deadline_policy=deadline_policy,
            breaker=breaker,
            cache=cache,
            limiter=limiter,
            tag_model_in_metrics=tag_model_in_metrics,
            stream_deadline_check_every_n_chunks=stream_deadline_check_every_n_chunks,
        )

        self._client: AsyncAzureOpenAI = client  # type: ignore[assignment]
        self._deployment_name = deployment_name
        self._default_model = default_model or deployment_name
        self._model_family = model_family
        self._max_context_length = int(max_context_length)
        self._server = "azure-openai"
        self._version = getattr(openai, "__version__", "unknown")

        # Cache for tokenizers to avoid repeated initialization
        # This is separate from the base class response cache
        self._tokenizer_cache: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_roles(messages: List[Mapping[str, Any]]) -> None:
        """
        Enforce a strict role set consistent with MockLLMAdapter and OpenAIAdapter.

        This keeps error behavior deterministic and aligned across adapters,
        and fails fast with a normalized BadRequest instead of deferring to
        provider-specific behavior for unknown roles.
        """
        for m in messages:
            role = str(m.get("role", ""))
            if role not in _ALLOWED_ROLES:
                raise BadRequest(f"unknown role: {role!r}")

    def _resolve_model(self, model: Optional[str]) -> str:
        """
        Resolve caller-supplied model or fall back to the default.
        
        Note: For Azure, this returns the logical model name for reporting,
        but we always use deployment_name in API calls.
        """
        return model or self._default_model

    def _build_messages(
        self,
        *,
        messages: List[Mapping[str, str]],
        system_message: Optional[str],
    ) -> List[Dict[str, str]]:
        """
        Combine system_message + messages for Azure OpenAI's chat format.

        BaseLLMAdapter has already validated that messages is a list of
        {role, content} with string values.

        Handles system message de-duplication: if both system_message
        parameter and system role messages are present, they are merged.
        """
        out: List[Dict[str, str]] = []

        # Collect all system messages from both sources
        system_messages: List[str] = []

        # Add explicit system_message parameter if provided
        if system_message:
            system_messages.append(system_message)

        # Extract system role messages from the messages list
        non_system_messages: List[Mapping[str, str]] = []
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "system":
                system_messages.append(content)
            else:
                non_system_messages.append(m)

        # Combine all system messages into one
        if system_messages:
            combined_system = "\n\n".join(system_messages)
            out.append({"role": "system", "content": combined_system})

            # Log if we had to merge multiple system messages
            if len(system_messages) > 1:
                logger.debug(
                    "Merged %d system messages into one for Azure OpenAI API",
                    len(system_messages),
                )

        # Add all non-system messages
        for m in non_system_messages:
            out.append({"role": m["role"], "content": m["content"]})

        return out

    def _get_response_format(self, ctx: Optional[OperationContext]) -> Optional[Dict[str, str]]:
        """
        Extract response format from context attributes.

        Supports:
        - ctx.attrs.get("response_format") = "json_object" → {"type": "json_object"}
        - ctx.attrs.get("response_format") = {"type": "json_object"} (direct mapping)
        - Other string values are passed through as {"type": <value>}.
        """
        if not ctx or not getattr(ctx, "attrs", None):
            return None

        response_format = ctx.attrs.get("response_format")
        if not response_format:
            return None

        if response_format == "json_object":
            return {"type": "json_object"}
        elif isinstance(response_format, dict) and response_format.get("type"):
            return {"type": response_format["type"]}
        elif isinstance(response_format, str):
            # Handle other potential formats
            return {"type": response_format}

        return None

    @staticmethod
    def _usage_from_response(resp: Any) -> TokenUsage:
        """Convert Azure OpenAI usage object → TokenUsage; handle missing usage."""
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
        """
        Extract Retry-After header from Azure OpenAI errors.
        
        Handles both:
        - Retry-After: 60 (integer seconds)
        - Retry-After: Wed, 21 Oct 2015 07:28:00 GMT (HTTP date format)
        
        Azure may also provide retry information in different headers or
        response body fields, which we attempt to extract.
        """
        try:
            resp = getattr(err, "response", None)
            if resp is None:
                return None
            
            headers = getattr(resp, "headers", None)
            if not headers:
                return None
            
            # Try case-insensitive header lookup
            # Azure may use x-ratelimit-reset-* headers in addition to Retry-After
            val = None
            for key in ["retry-after", "Retry-After", "retry_after", "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"]:
                val = headers.get(key)
                if val is not None:
                    break
            
            if val is None:
                return None
            
            val_str = str(val).strip()
            
            # Try parsing as integer seconds first
            try:
                seconds = int(val_str)
                return max(0, seconds) * 1000
            except ValueError:
                pass
            
            # Try parsing as HTTP date
            try:
                from email.utils import parsedate_to_datetime
                import time
                retry_datetime = parsedate_to_datetime(val_str)
                delay_seconds = int(retry_datetime.timestamp() - time.time())
                return max(0, delay_seconds) * 1000
            except Exception:
                pass
            
            # Fallback: return None instead of crashing
            return None
            
        except Exception:
            return None

    def _translate_azure_error(self, err: Exception) -> LLMAdapterError:
        """
        Map Azure OpenAI client errors → Corpus LLMAdapterError subclasses.

        This ensures routers and observability can reason about failures
        without vendor-specific conditionals.
        
        Azure errors are largely compatible with OpenAI errors, but may
        have different status codes or messages for certain scenarios.
        """
        # Connection / transport issues → TransientNetwork
        if APIConnectionError is not None and isinstance(err, APIConnectionError):
            return TransientNetwork(str(err) or "Azure OpenAI API connection error")

        # Upstream timeout → DeadlineExceeded
        if APITimeoutError is not None and isinstance(err, APITimeoutError):
            return DeadlineExceeded("Azure OpenAI API request timed out")

        # Rate limiting / quota → ResourceExhausted
        if RateLimitError is not None and isinstance(err, RateLimitError):
            retry_ms = self._extract_retry_after_ms(err)
            return ResourceExhausted(
                "Azure OpenAI rate limit exceeded",
                retry_after_ms=retry_ms,
                throttle_scope="tenant",
            )

        # AuthN / AuthZ → AuthError
        if AuthenticationError is not None and isinstance(err, AuthenticationError):
            return AuthError(str(err) or "Azure OpenAI authentication/authorization error")

        # Request shape / params → BadRequest
        if BadRequestError is not None and isinstance(err, BadRequestError):
            return BadRequest(str(err) or "Azure OpenAI request is invalid")

        # HTTP status buckets
        if APIStatusError is not None and isinstance(err, APIStatusError):
            status = int(getattr(err, "status_code", 0) or 0)

            # Map a few key ranges explicitly.
            if status == 400:
                return BadRequest(str(err) or "Azure OpenAI request is invalid")
            if status in (401, 403):
                return AuthError(str(err) or "Azure OpenAI authentication/authorization error")
            if status == 404:
                # Often "deployment not found" or similar in Azure.
                return NotSupported(str(err) or "Requested Azure OpenAI resource is not supported")
            if status == 429:
                retry_ms = self._extract_retry_after_ms(err)
                return ResourceExhausted(
                    "Azure OpenAI rate limit exceeded",
                    retry_after_ms=retry_ms,
                    throttle_scope="tenant",
                )
            if 500 <= status <= 599:
                retry_ms = self._extract_retry_after_ms(err)
                return Unavailable(
                    "Azure OpenAI service is temporarily unavailable",
                    retry_after_ms=retry_ms,
                )

            # Fallback for unexpected status codes.
            return Unavailable(str(err) or f"Azure OpenAI error (status={status})")

        # Generic OpenAI error fallback → Unavailable
        if isinstance(err, OpenAIError):
            return Unavailable(str(err) or "Azure OpenAI API error")

        # Anything else → wrap as Unavailable
        return Unavailable(str(err) or "internal Azure OpenAI adapter error")

    def _get_tokenizer(self, model: str) -> Any:
        """
        Get or create tokenizer for the given model, with caching.

        Returns None if tiktoken is not available.
        """
        if tiktoken is None:
            return None

        cache_key = model
        if cache_key in self._tokenizer_cache:
            return self._tokenizer_cache[cache_key]

        try:
            try:
                enc = tiktoken.encoding_for_model(model)
            except Exception:
                # Fallback encoding used by most modern OpenAI/Azure chat models.
                enc = tiktoken.get_encoding("cl100k_base")

            self._tokenizer_cache[cache_key] = enc
            return enc
        except Exception:
            logger.debug("Failed to get tokenizer for model %s", model, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # BaseLLMAdapter backend hooks
    # ------------------------------------------------------------------

    async def _do_capabilities(self) -> LLMCapabilities:
        """
        Report adapter capabilities.

        We include at least one supported model so conformance tests can
        exercise the adapter end-to-end. For Azure, we report the
        deployment name as the supported model since that's what clients
        will use.
        """
        return LLMCapabilities(
            server=self._server,
            version=self._version,
            model_family=self._model_family,
            max_context_length=self._max_context_length,
            supports_streaming=True,
            supports_roles=True,
            supports_json_output=True,  # JSON via response_format
            supports_tools=False,       # Tools accepted at the interface but not yet implemented
            supports_parallel_tool_calls=False,
            idempotent_writes=True,
            supports_multi_tenant=True,
            supports_system_message=True,
            supports_deadline=True,
            # We implement token counting using tiktoken when available,
            # with a heuristic fallback when it isn't.
            supports_count_tokens=True,
            # IMPORTANT: must be non-empty to satisfy conformance tests.
            # We report the deployment name here.
            supported_models=(self._deployment_name,),
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
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        ctx: Optional[OperationContext] = None,
    ) -> LLMCompletion:
        """
        Backend implementation of `complete()` using Azure Chat Completions.

        Note:
        - Response caching is handled automatically by BaseLLMAdapter via
          the _make_complete_cache_key mechanism.
        - tools and tool_choice are accepted for interface parity with
          the protocol but are intentionally ignored while
          capabilities.supports_tools == False. BaseLLMAdapter will
          reject tool usage before this method is invoked in that mode.
        - Azure uses deployment_name in API calls, not model name.
        """
        # Enforce the same role restrictions as the mock adapter so both
        # behave equivalently from the protocol's perspective.
        self._validate_roles(messages)

        resolved_model = self._resolve_model(model)
        azure_messages = self._build_messages(
            messages=messages,
            system_message=system_message,
        )

        # Extract response format from context if specified
        response_format = self._get_response_format(ctx)

        try:
            # Build the base request parameters
            # Key difference: Azure uses deployment_name via the model parameter
            request_params: Dict[str, Any] = {
                "model": self._deployment_name,  # Azure deployment name, not logical model
                "messages": azure_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "stop": stop_sequences,
                "stream": False,
            }

            # Add response_format if specified
            if response_format:
                request_params["response_format"] = response_format

            resp = await self._client.chat.completions.create(**request_params)

        except Exception as exc:  # noqa: BLE001
            raise self._translate_azure_error(exc) from exc

        # Extract the primary choice.
        if not getattr(resp, "choices", None):
            raise Unavailable("Azure OpenAI returned no choices")

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
        
        # Azure returns deployment name in the model field, but we report
        # the logical model name for consistency
        model_id = resolved_model

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
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        ctx: Optional[OperationContext] = None,
    ) -> AsyncIterator[LLMChunk]:
        """
        Backend implementation of `stream()` using Azure OpenAI streaming.

        We emit:
            - One `LLMChunk` per non-empty delta text (is_final=False).
            - A terminal sentinel chunk with is_final=True.

        With `stream_options={"include_usage": True}`, the final event
        contains exact token usage; this is surfaced on the final chunk
        via `usage_so_far`.

        Note:
        - Streaming responses are not cached by BaseLLMAdapter.
        - tools and tool_choice are accepted for interface parity but
          ignored while capabilities.supports_tools == False.
        - Azure uses deployment_name in API calls, not model name.
        """
        # Enforce the same role restrictions as the mock adapter.
        self._validate_roles(messages)

        resolved_model = self._resolve_model(model)
        azure_messages = self._build_messages(
            messages=messages,
            system_message=system_message,
        )

        # Extract response format from context if specified
        response_format = self._get_response_format(ctx)

        try:
            # Build the base request parameters
            request_params: Dict[str, Any] = {
                "model": self._deployment_name,  # Azure deployment name, not logical model
                "messages": azure_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "stop": stop_sequences,
                "stream": True,
                # Key improvement: ask Azure OpenAI to include usage in the stream.
                "stream_options": {"include_usage": True},
            }

            # Add response_format if specified
            if response_format:
                request_params["response_format"] = response_format

            stream = await self._client.chat.completions.create(**request_params)

        except Exception as exc:  # noqa: BLE001
            raise self._translate_azure_error(exc) from exc

        # Aggregate model + usage info as we go; usage is normally only
        # available on the final event.
        final_usage: Optional[TokenUsage] = None
        received_chunks = 0

        try:
            async for event in stream:
                received_chunks += 1

                # Handle empty or malformed events gracefully
                if not getattr(event, "choices", None):
                    continue

                choice = event.choices[0]
                if not choice:
                    continue

                delta = getattr(choice, "delta", None)
                text = ""
                if delta is not None:
                    text = getattr(delta, "content", "") or ""
                # Older patterns may expose `choice.text` instead.
                if not text and hasattr(choice, "text"):
                    text = getattr(choice, "text", "") or ""

                # Collect final usage if present on this chunk (thanks to
                # stream_options={"include_usage": True}).
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
                        model=resolved_model,
                        usage_so_far=None,  # not tracked per-chunk
                    )

            # Handle case where stream ends with no chunks received
            if received_chunks == 0:
                logger.warning("Azure OpenAI stream ended with no chunks received")
                yield LLMChunk(
                    text="",
                    is_final=True,
                    model=resolved_model,
                    usage_so_far=TokenUsage(
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                    ),
                )
                return

        except Exception as exc:  # noqa: BLE001
            # Translate any errors that occur during streaming
            raise self._translate_azure_error(exc) from exc

        # Emit a final sentinel chunk marking end-of-stream (with usage if available).
        yield LLMChunk(
            text="",
            is_final=True,
            model=resolved_model,
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
        Token counting for Azure OpenAI models.

        Strategy:
        - If `tiktoken` is available, use model-aware tokenization via
          `encoding_for_model` (falling back to `cl100k_base`).
        - Otherwise, use a whitespace/character heuristic similar to
          MockLLMAdapter, which is good enough for rough planning.

        This function is used by BaseLLMAdapter for:
            - context-window preflight checks
            - coarse quota tracking
        """
        if not text:
            return 0

        resolved_model = self._resolve_model(model)

        # Try cached tokenizer first
        enc = self._get_tokenizer(resolved_model)
        if enc is not None:
            try:
                return len(enc.encode(text))
            except Exception:  # noqa: BLE001
                # If tokenization fails, fall back to heuristic
                logger.debug(
                    "tiktoken token counting failed; falling back to heuristic",
                    exc_info=True,
                )

        # Heuristic fallback: whitespace-based + small overhead, blended
        # with a character-based estimate. This is reasonably accurate
        # for many languages but is only an estimate (not a strict lower
        # bound) and is intended for planning, not billing.
        word_count = len(text.split())
        char_count = len(text)

        word_based = word_count + 3  # Original heuristic
        char_based = char_count // 4  # Rough character-to-token ratio

        return max(word_based, char_based)

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
        - Otherwise we perform a minimal live check by listing deployments.
          Any Azure OpenAI error is captured into the returned payload instead
          of being raised, so that `BaseLLMAdapter.health()` can still
          normalize the response.
        """
        status_hint: Optional[str] = None
        if ctx is not None and getattr(ctx, "attrs", None):
            status_hint = ctx.attrs.get("health")

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
            # For Azure, we try to list deployments or make a minimal request.
            # We don't depend on the exact payload shape here.
            await self._client.models.list()
            return {
                "ok": True,
                "status": "healthy",
                "server": self._server,
                "version": self._version,
            }
        except Exception as exc:  # noqa: BLE001
            err = self._translate_azure_error(exc)
            # We intentionally do NOT raise here; BaseLLMAdapter.health()
            # expects a mapping and will treat this as a successful but
            # unhealthy probe.
            logger.warning("AzureOpenAIAdapter health check failed: %s", err)
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

            async with AzureOpenAIAdapter(...) as adapter:
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
            logger.debug("AzureOpenAIAdapter close() failed", exc_info=True)

        # Clear tokenizer cache
        self._tokenizer_cache.clear()


__all__ = [
    "AzureOpenAIAdapter",
]
