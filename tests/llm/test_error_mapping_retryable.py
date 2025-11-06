# SPDX-License-Identifier: Apache-2.0
"""
LLM Conformance — Error mapping & retryability (enhanced).

Specification references:
  • §12.1 (Retry Semantics): which classes are retryable / conditionally retryable / non-retryable
  • §12.4 (Error Handling and Resilience — Error Mapping Table): taxonomy + client guidance, retry_after_ms hints
  • §8.3 (LLM Protocol V1 — Operations / parameter validation): BadRequest on invalid sampling ranges
  • §6.1 (Common Foundation — Operation Context): deadline semantics (pre-expired budgets)

Covers (normative + robustness):
  • Retryable errors (Unavailable, ResourceExhausted) include sane retry_after_ms hints when present
  • Non-retryable BadRequest on invalid sampling params (temperature out of range) has no retry_after_ms
  • DeadlineExceeded on pre-expired budgets is raised (conditionally retryable only if deadline/work adjusted)
  • Error objects expose informative message text and string `code`
"""

import random
import pytest

from corpus_sdk.examples.llm.mock_llm_adapter import MockLLMAdapter
from corpus_sdk.llm.llm_base import OperationContext, DeadlineExceeded
from corpus_sdk.examples.common.ctx import make_ctx
from corpus_sdk.examples.common.errors import (
    ResourceExhausted,
    Unavailable,
    BadRequest,
)

pytestmark = pytest.mark.asyncio


async def test_retryable_errors_with_hints():
    """
    §12.1, §12.4 — Retryable classification and hints.

    Retryable errors (Unavailable, ResourceExhausted) SHOULD include `retry_after_ms`
    to guide client backoff. If present, it MUST be a non-negative integer and
    SHOULD be reasonable (not minutes-long in normal cases).
    """
    # Deterministic path: always trigger failure branch
    random.seed(1337)
    adapter = MockLLMAdapter(failure_rate=1.0)  # force failure path
    ctx = make_ctx(OperationContext, request_id="t_err_retryable", tenant="test")

    with pytest.raises((Unavailable, ResourceExhausted)) as excinfo:
        await adapter.complete(
            messages=[{"role": "user", "content": "overload"}],
            model="mock-model",
            ctx=ctx,
        )

    err = excinfo.value

    # Message should be informative
    assert (getattr(err, "message", None) or str(err)).strip(), "error message should be non-empty"

    # Code is a string if present (taxonomy key, §12.4)
    code = getattr(err, "code", None)
    if code is not None:
        assert isinstance(code, str) and code.strip(), "error code should be a non-empty string"

    # retry_after_ms sanity checks (when provided)
    ra = getattr(err, "retry_after_ms", None)
    assert (ra is None) or (isinstance(ra, int) and ra >= 0), "retry_after_ms must be non-negative int or None"
    if ra is not None:
        # Keep generous but bounded: < 5 minutes in mocks
        assert ra < 300_000, f"retry_after_ms ({ra} ms) unreasonably long for mock"


async def test_bad_request_is_non_retryable_and_no_retry_after():
    """
    §8.3 — Parameter validation must produce BadRequest.
    §12.1/§12.4 — BadRequest is non-retryable; should not carry retry_after_ms.
    """
    adapter = MockLLMAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_err_bad_request", tenant="test")

    # temperature out of range per §8.3 (valid range [0, 2])
    with pytest.raises(BadRequest) as excinfo:
        await adapter.complete(
            messages=[{"role": "user", "content": "oops"}],
            model="mock-model",
            temperature=3.0,
            ctx=ctx,
        )

    err = excinfo.value
    # No retry_after_ms for non-retryable client errors
    assert getattr(err, "retry_after_ms", None) in (None, 0), "BadRequest should not suggest retries"
    # Informative message expected
    assert (getattr(err, "message", None) or str(err)).strip(), "BadRequest should include reason text"


async def test_deadline_exceeded_is_conditionally_retryable_with_no_chunks():
    """
    §6.1 — Pre-expired budgets MUST fail fast.
    §12.1/§12.4 — DeadlineExceeded is conditionally retryable (only if deadline/work adjusted).
    """
    adapter = MockLLMAdapter(failure_rate=0.0)

    # Pre-expired: absolute epoch 0 guarantees elapsed deadline
    ctx = OperationContext(deadline_ms=0, tenant="test")

    with pytest.raises(DeadlineExceeded):
        await adapter.complete(
            messages=[{"role": "user", "content": "late"}],
            model="mock-model",
            ctx=ctx,
        )


async def test_retryable_error_attributes_minimum_shape():
    """
    §12.4 — Normalized error objects SHOULD provide programmatic attributes.
    We check presence & types without overfitting to specific provider codes.
    """
    random.seed(2025)
    adapter = MockLLMAdapter(failure_rate=1.0)  # force failure
    ctx = make_ctx(OperationContext, request_id="t_err_shape", tenant="test")

    with pytest.raises((Unavailable, ResourceExhausted)) as excinfo:
        await adapter.complete(
            messages=[{"role": "user", "content": "please fail"}],
            model="mock-model",
            ctx=ctx,
        )
    err = excinfo.value

    # Expect a string code if set
    code = getattr(err, "code", None)
    if code is not None:
        assert isinstance(code, str) and code.strip(), "error code should be a non-empty string"

    # Optional structured details should be a mapping if present
    details = getattr(err, "details", None)
    if details is not None:
        assert isinstance(details, dict), "error.details should be a dict when present"

    # Optional resource scope (taxonomy hint) remains a string if present
    scope = getattr(err, "resource_scope", None)
    if scope is not None:
        assert isinstance(scope, str) and scope.strip(), "resource_scope should be a non-empty string"
