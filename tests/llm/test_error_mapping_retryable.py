# SPDX-License-Identifier: Apache-2.0
"""
LLM Conformance — Error mapping & retryability (enhanced).

Specification references:
  • §12.1 (Retry Semantics): retryable / conditionally retryable / non-retryable
  • §12.4 (Error Handling and Resilience — Error Mapping Table): taxonomy + client guidance, retry_after_ms hints
  • §8.3 (LLM Protocol V1 — Operations / parameter validation): BadRequest on invalid sampling ranges
  • §6.1 (Common Foundation — Operation Context): deadline semantics (pre-expired budgets)

Covers (normative + robustness):
  • Retryable errors (Unavailable, ResourceExhausted) are raised via adapter methods and expose retry_after_ms hints when present
  • Non-retryable BadRequest on invalid sampling params (temperature out of range) has no retry_after_ms
  • DeadlineExceeded on pre-expired budgets is raised when supports_deadline=True; not enforced when supports_deadline=False
  • Error objects expose informative message text and string `code`
"""

import pytest

from corpus_sdk.llm.llm_base import (
    BaseLLMAdapter,
    OperationContext,
    DeadlineExceeded,
    ResourceExhausted,
    Unavailable,
    BadRequest,
    LLMCapabilities,
    LLMCompletion,
    TokenUsage,
)

pytestmark = pytest.mark.asyncio


class RaisingAdapter(BaseLLMAdapter):
    """
    Minimal BaseLLMAdapter-backed adapter for deterministic error triggering
    via the public protocol methods (complete/health/stream/count_tokens).

    We use this to avoid directly instantiating error classes in tests; instead
    tests call adapter.complete(...) and assert on raised normalized errors.
    """

    def __init__(self, *, exc: Exception, supports_deadline: bool = True):
        super().__init__(mode="thin", stream_deadline_check_every_n_chunks=1)
        self._exc = exc
        self._caps = LLMCapabilities(
            server="test",
            version="1.0.0",
            model_family="test",
            max_context_length=4096,
            supports_streaming=True,
            supports_roles=True,
            supports_json_output=False,
            supports_tools=False,
            supports_parallel_tool_calls=False,
            supports_tool_choice=False,
            idempotent_writes=False,
            supports_multi_tenant=False,
            supports_system_message=True,
            supports_deadline=bool(supports_deadline),
            supports_count_tokens=True,
            supported_models=("test-model",),
        )

    async def _do_capabilities(self) -> LLMCapabilities:
        return self._caps

    async def _do_complete(self, **kwargs) -> LLMCompletion:
        raise self._exc

    async def _do_stream(self, **kwargs):
        raise self._exc

    async def _do_count_tokens(self, text: str, **kwargs) -> int:
        raise self._exc

    async def _do_health(self, **kwargs):
        raise self._exc


async def test_error_handling_retryable_errors_with_hints(adapter):
    """
    §12.1, §12.4 — Retryable classification and hints.

    Retryable errors (Unavailable, ResourceExhausted) SHOULD include `retry_after_ms`
    to guide client backoff. If present, it MUST be a non-negative integer and
    SHOULD be reasonable (not minutes-long in normal cases).

    This test triggers ResourceExhausted via adapter.complete().
    """
    a = RaisingAdapter(exc=ResourceExhausted("rate limited", retry_after_ms=123), supports_deadline=False)
    caps = await a.capabilities()

    with pytest.raises(ResourceExhausted) as excinfo:
        await a.complete(
            messages=[{"role": "user", "content": "x"}],
            model=caps.supported_models[0],
            ctx=OperationContext(tenant="t", request_id="err-hints-1"),
        )

    err = excinfo.value

    assert (getattr(err, "message", None) or str(err)).strip(), "error message should be non-empty"

    code = getattr(err, "code", None)
    if code is not None:
        assert isinstance(code, str) and code.strip(), "error code should be a non-empty string"

    ra = getattr(err, "retry_after_ms", None)
    assert (ra is None) or (isinstance(ra, int) and ra >= 0), "retry_after_ms must be non-negative int or None"
    if ra is not None:
        assert ra < 300_000, f"retry_after_ms ({ra} ms) unreasonably long"


async def test_error_handling_bad_request_is_non_retryable_and_no_retry_after(adapter):
    """
    §8.3 — Parameter validation must produce BadRequest via adapter.complete().
    §12.1/§12.4 — BadRequest is non-retryable; should not carry retry_after_ms.
    """
    ctx = OperationContext(request_id="t_err_bad_request", tenant="test")
    caps = await adapter.capabilities()

    with pytest.raises(BadRequest) as excinfo:
        await adapter.complete(
            messages=[{"role": "user", "content": "oops"}],
            model=caps.supported_models[0],
            temperature=3.0,  # out of range (valid range [0, 2])
            ctx=ctx,
        )

    err = excinfo.value
    assert getattr(err, "retry_after_ms", None) in (None, 0), "BadRequest should not suggest retries"
    assert (getattr(err, "message", None) or str(err)).strip(), "BadRequest should include reason text"


async def test_error_handling_deadline_exceeded_is_conditionally_retryable_with_no_chunks(adapter):
    """
    §6.1 — Pre-expired budgets MUST fail fast when supports_deadline=True.
    §12.1/§12.4 — DeadlineExceeded is conditionally retryable.
    """
    caps = await adapter.capabilities()
    ctx = OperationContext(deadline_ms=0, tenant="test")  # epoch 0 guarantees elapsed deadline

    if caps.supports_deadline:
        with pytest.raises(DeadlineExceeded):
            await adapter.complete(
                messages=[{"role": "user", "content": "late"}],
                model=caps.supported_models[0],
                ctx=ctx,
            )
    else:
        # If the adapter explicitly reports no deadline enforcement, pre-expired deadlines must not be enforced.
        res = await adapter.complete(
            messages=[{"role": "user", "content": "late"}],
            model=caps.supported_models[0],
            ctx=ctx,
        )
        assert hasattr(res, "finish_reason")


async def test_error_handling_retryable_error_attributes_minimum_shape(adapter):
    """
    §12.4 — Normalized error objects SHOULD provide programmatic attributes.

    Trigger Unavailable via adapter.complete() (not by instantiating and asserting directly).
    """
    a = RaisingAdapter(
        exc=Unavailable(
            "backend unavailable",
            retry_after_ms=42,
            details={"hint": "retry later"},
            throttle_scope="global",
        ),
        supports_deadline=False,
    )
    caps = await a.capabilities()

    with pytest.raises(Unavailable) as excinfo:
        await a.complete(
            messages=[{"role": "user", "content": "x"}],
            model=caps.supported_models[0],
            ctx=OperationContext(tenant="t", request_id="err-attrs-1"),
        )

    err = excinfo.value

    code = getattr(err, "code", None)
    if code is not None:
        assert isinstance(code, str) and code.strip(), "error.code should be a non-empty string"

    details = getattr(err, "details", None)
    if details is not None:
        assert isinstance(details, dict), "error.details should be a dict when present"

    throttle_scope = getattr(err, "throttle_scope", None)
    if throttle_scope is not None:
        assert isinstance(throttle_scope, str) and throttle_scope.strip(), "throttle_scope should be a non-empty string"


async def test_error_handling_deadline_capability_alignment(adapter):
    """
    Capability↔behavior alignment:
      - supports_deadline=True  => expired deadline MUST raise DeadlineExceeded
      - supports_deadline=False => expired deadline MUST NOT be enforced
    """
    # Use Base-backed adapter to force supports_deadline True/False deterministically.
    ok_completion = LLMCompletion(
        text="ok",
        model="test-model",
        model_family="test",
        usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        finish_reason="stop",
    )

    class OkAdapter(BaseLLMAdapter):
        def __init__(self, supports_deadline: bool):
            super().__init__(mode="thin", stream_deadline_check_every_n_chunks=1)
            self._caps = LLMCapabilities(
                server="ok",
                version="1.0.0",
                model_family="ok",
                max_context_length=4096,
                supports_streaming=True,
                supports_roles=True,
                supports_deadline=bool(supports_deadline),
                supports_count_tokens=True,
                supported_models=("test-model",),
            )

        async def _do_capabilities(self) -> LLMCapabilities:
            return self._caps

        async def _do_complete(self, **kwargs) -> LLMCompletion:
            return ok_completion

        async def _do_stream(self, **kwargs):
            yield  # pragma: no cover

        async def _do_count_tokens(self, text: str, **kwargs) -> int:
            return 0

        async def _do_health(self, **kwargs):
            return {"ok": True, "server": "ok", "version": "1.0.0"}

    expired = OperationContext(deadline_ms=0, tenant="test")

    a_true = OkAdapter(supports_deadline=True)
    with pytest.raises(DeadlineExceeded) as excinfo:
        await a_true.health(ctx=expired)
    assert excinfo.value.code in ("DEADLINE", "DEADLINE_EXCEEDED")

    a_false = OkAdapter(supports_deadline=False)
    h = await a_false.health(ctx=expired)
    assert isinstance(h, dict) and "ok" in h
