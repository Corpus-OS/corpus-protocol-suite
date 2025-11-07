# SPDX-License-Identifier: Apache-2.0
"""
Example 05 — Error Handling and Retryability

Demonstrates how to handle errors using the example taxonomy.
Shows how retryable errors (e.g., ResourceExhausted, TransientNetwork)
are classified distinctly from non-retryable ones.
"""

import asyncio
from corpus_sdk.examples.llm.mock_llm_adapter import MockLLMAdapter
from corpus_sdk.examples.common.errors import ResourceExhausted, TransientNetwork, BadRequest
from corpus_sdk.examples.common.ctx import make_ctx
from corpus_sdk.llm.llm_base import OperationContext
from corpus_sdk.examples.common.printing import box, print_kv


async def main() -> None:
    box("Example 05 — Error Handling Demo")

    adapter = MockLLMAdapter()
    ctx = make_ctx(OperationContext, request_id="error-demo", tenant="demo")

    # Simulate catching different errors
    try:
        # Force a retryable error
        raise ResourceExhausted("Rate limit hit", retry_after_ms=1500)
    except (ResourceExhausted, TransientNetwork) as e:
        print_kv({
            "Type": e.__class__.__name__,
            "Retryable": True,
            "Message": str(e),
            "Retry After (ms)": getattr(e, "retry_after_ms", None)
        })
    except BadRequest as e:
        print_kv({
            "Type": e.__class__.__name__,
            "Retryable": False,
            "Message": str(e)
        })
    except Exception as e:
        print_kv({
            "Type": type(e).__name__,
            "Retryable": False,
            "Message": str(e)
        })


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
