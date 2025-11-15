# SPDX-License-Identifier: Apache-2.0
"""
Example 07 — SDK Deadline Enforcement (LLM)

Demonstrates CORRECT use of the SDK's deadline mechanism:
- Use OperationContext.deadline_ms / timeout_ms
- Do NOT wrap with asyncio.wait_for (competes with SDK deadlines)
"""

from __future__ import annotations
import asyncio

from examples.llm.mock_llm_adapter import MockLLMAdapter
from corpus_sdk.llm.llm_base import OperationContext
from examples.common.ctx import make_ctx, remaining_budget_ms
from examples.common.printing import box, print_kv

# Prefer the example taxonomy for clarity; falls back to SDK if needed.
try:
    from corpus_sdk.examples.common.errors import DeadlineExceeded
except Exception:  # pragma: no cover
    # If your normalized errors are exposed from the SDK directly:
    from corpus_sdk.llm.llm_base import DeadlineExceeded  # type: ignore


async def main() -> None:
    box("Example 07 — SDK Deadline Enforcement")

    adapter = MockLLMAdapter()
    # Set a very small budget so the adapter's internal work exceeds it.
    ctx = make_ctx(
        OperationContext,
        request_id="deadline-demo",
        tenant="examples",
        timeout_ms=10,  # SDK-enforced deadline (absolute is computed internally)
    )

    try:
        _ = await adapter.complete(
            messages=[{"role": "user", "content": "simulate slow operation"}],
            model="mock-model",
            ctx=ctx,
        )
        # If we reached here, operation finished before the SDK deadline.
        print_kv({"status": "completed_within_deadline", "remaining_ms": remaining_budget_ms(ctx)})
    except DeadlineExceeded as e:
        # Correct handling: the SDK raised a normalized deadline error
        print_kv(
            {
                "status": "deadline_exceeded",
                "error": e.__class__.__name__,
                "message": getattr(e, "message", str(e)),
                "remaining_ms": remaining_budget_ms(ctx) or 0,
            }
        )
    except Exception as e:
        # Any other error path is handled separately
        print_kv({"status": "other_error", "type": e.__class__.__name__, "message": str(e)})


if __name__ == "__main__":
    asyncio.run(main())
