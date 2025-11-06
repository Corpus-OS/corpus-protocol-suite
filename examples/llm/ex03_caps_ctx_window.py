# SPDX-License-Identifier: Apache-2.0
"""
Example 03 — Model Capabilities and Context Window

Demonstrates how to query and display model capabilities from the
MockLLMAdapter. Prints available models and their context window sizes.
"""

import asyncio
from corpus_sdk.examples.llm.mock_llm_adapter import MockLLMAdapter
from corpus_sdk.examples.common.printing import box, print_kv, print_table


async def main() -> None:
    box("Example 03 — Model Capabilities")

    adapter = MockLLMAdapter()
    caps = await adapter.capabilities()

    # Display core adapter-level metadata
    print_kv({
        "Server": caps.server,
        "Version": caps.version,
        "Family": caps.model_family,
        "Max Context": caps.max_context_length,
    })

    # Display supported models table
    print("\nSupported Models:")
    print_table(
        [{"Model": m, "Context Window": caps.max_context_length} for m in caps.supported_models]
    )

    print("\n✅ Capabilities loaded successfully.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
