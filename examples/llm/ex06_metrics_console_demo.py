# SPDX-License-Identifier: Apache-2.0
"""
Example 06 — Console Metrics Demo (LLM)

Shows metrics emitted by the adapter on:
  • a successful completion
  • a simulated overload error (triggered via message content)

We share a single ConsoleMetrics sink across two MockLLMAdapter instances:
  - success_adapter: failure_rate=0.0 for deterministic success
  - fail_adapter:    failure_rate=1.0 to force an error; "overload" message
"""

from __future__ import annotations
import asyncio

from corpus_sdk.examples.llm.mock_llm_adapter import MockLLMAdapter
from corpus_sdk.examples.common.metrics_console import ConsoleMetrics
from corpus_sdk.examples.common.ctx import make_ctx
from corpus_sdk.llm.llm_base import OperationContext
from corpus_sdk.examples.common.printing import box, print_kv

# Prefer the example taxonomy (clearer for OSS examples).
try:
    from corpus_sdk.examples.common.errors import Unavailable, ResourceExhausted
except Exception:  # pragma: no cover
    from corpus_sdk.common.errors import Unavailable, ResourceExhausted  # type: ignore


async def main() -> None:
    box("Example 06 — Console Metrics Demo")

    # One metrics sink shared by both adapters
    metrics = ConsoleMetrics(name="llm-demo", colored=False)

    # Deterministic success / failure adapters
    success_adapter = MockLLMAdapter(failure_rate=0.0, metrics=metrics)
    fail_adapter = MockLLMAdapter(failure_rate=1.0, metrics=metrics)

    # --- Successful request (emits OBS/CTR lines) ---
    ctx_ok = make_ctx(OperationContext, request_id="metrics-success", tenant="examples")
    ok = await success_adapter.complete(
        messages=[{"role": "user", "content": "Summarize Corpus SDK"}],
        model="mock-model",
        ctx=ctx_ok,
    )
    print_kv({"success.text": ok.text, "success.tokens": ok.usage.total_tokens})

    # --- Simulated overload path (emits OBS with error code, counters, etc.) ---
    # Using content "overload" so MockLLMAdapter raises Unavailable on failure
    ctx_err = make_ctx(OperationContext, request_id="metrics-failure", tenant="examples")
    try:
        await fail_adapter.complete(
            messages=[{"role": "user", "content": "overload"}],
            model="mock-model",
            ctx=ctx_err,
        )
    except (Unavailable, ResourceExhausted) as e:
        print_kv(
            {
                "failure.error": e.__class__.__name__,
                "failure.message": getattr(e, "message", str(e)),
            }
        )

    print("\n✅ Metrics were emitted above as structured lines (OBS/CTR/GAU).")


if __name__ == "__main__":
    asyncio.run(main())
