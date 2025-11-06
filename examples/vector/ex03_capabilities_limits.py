# SPDX-License-Identifier: Apache-2.0
"""
Vector ex03 — Capabilities & Limits
Spec refs: §6.2 (Capability Discovery), §9.2 (Data Types)

Demonstrates:
  • Fetch VectorCapabilities and print fields useful for routing/planning
"""

import asyncio
from corpus_sdk.examples.vector.mock_vector_adapter import MockVectorAdapter
from corpus_sdk.examples.common.printing import box, print_kv

async def main():
    box("Vector ex03 — Capabilities & Limits")
    adapter = MockVectorAdapter(failure_rate=0.0)
    caps = await adapter.capabilities()
    print_kv({
        "server": caps.server,
        "version": caps.version,
        "supported_metrics": caps.supported_metrics,
        "max_dimensions": caps.max_dimensions,
        "max_top_k": caps.max_top_k,
        "max_batch_size": caps.max_batch_size,
        "supports_metadata_filtering": caps.supports_metadata_filtering,
        "supports_index_management": caps.supports_index_management,
    })
    print("\n[lesson] ex03: capability discovery enables safe validation before calls.")

if __name__ == "__main__":
    asyncio.run(main())

