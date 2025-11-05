# SPDX-License-Identifier: Apache-2.0
"""
LLM Conformance — Error mapping & retryability.

Asserts:
  • Mock adapter maps "overload" content to retryable classes
  • retry_after_ms (if present) is sane
"""

import random
import pytest

from corpus_sdk.examples.llm.mock_llm_adapter import MockLLMAdapter
from corpus_sdk.llm.llm_base import OperationContext
from corpus_sdk.examples.common.ctx import make_ctx
from corpus_sdk.examples.common.errors import ResourceExhausted, Unavailable

pytestmark = pytest.mark.asyncio


async def test_retryable_errors_with_hints():
    # Deterministic path: always trigger failure branch
    random.seed(1337)
    adapter = MockLLMAdapter(failure_rate=1.0)  # force failure path
    ctx = make_ctx(OperationContext, request_id="t_err_retryable", tenant="test")

    with pytest.raises((Unavailable, ResourceExhausted)) as excinfo:
        await adapter.complete(messages=[{"role": "user", "content": "overload"}], ctx=ctx)

    err = excinfo.value
    # retryable class as per example taxonomy
    assert isinstance(err, (Unavailable, ResourceExhausted))

    # optional retry_after_ms is int-like if present
    ra = getattr(err, "retry_after_ms", None)
    assert (ra is None) or (isinstance(ra, int) and ra >= 0)
