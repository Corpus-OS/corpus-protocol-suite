Here’s the upgraded, “all four protocols” universal prompt you can drop straight into your docs.

You are helping me generate a **production-ready Corpus provider adapter**.

This adapter may implement any subset of the following Corpus protocols:
- LLM
- Embedding
- Vector
- Graph

You will generate a single adapter class plus a configuration checklist.

===============================================================================
1. Corpus Protocols and Required Overrides
===============================================================================

For each protocol, the adapter subclasses the corresponding base:

- LLM:      corpus_sdk.llm.llm_base.BaseLLMAdapter
- Embedding: corpus_sdk.embedding.embedding_base.BaseEmbeddingAdapter
- Vector:   corpus_sdk.vector.vector_base.BaseVectorAdapter
- Graph:    adapter_sdk.graph_base.BaseGraphAdapter

You must implement **only** the methods required for the protocols I specify.

----------------------------------------
LLM Protocol (BaseLLMAdapter)
----------------------------------------
Required overrides:
- _do_complete(...)
- _do_stream(...)
- _do_count_tokens(...), if supports_count_tokens=True
- _do_capabilities(...)
- _do_health(...)

Behavior:
- Deterministic given same inputs + ctx.request_id where applicable
- Streaming and non-streaming must share the same planning path so aggregate
  streamed text matches the complete() text
- Token counting must be consistent with how you bill / estimate tokens

----------------------------------------
Embedding Protocol (BaseEmbeddingAdapter)
----------------------------------------
Required overrides:
- _do_capabilities(...)
- _do_health(...)
- _do_embed(...)
- _do_embed_batch(...), if native batch is supported
- _do_count_tokens(...), if supports_token_counting=True

Behavior:
- Single-embed and batch-embed must be consistent (same vectors for same input)
- Batch should support partial failures where possible (per-item errors)
- Normalization and truncation semantics should follow capabilities

----------------------------------------
Vector Protocol (BaseVectorAdapter)
----------------------------------------
Required overrides:
- _do_capabilities(...)
- _do_health(...)
- _do_query(...)
- _do_upsert(...)
- _do_delete(...)
- _do_create_namespace(...), if index management is supported
- _do_delete_namespace(...), if index management is supported

Behavior:
- Query must respect namespace schema (dimensions, metrics)
- Upsert must support partial failures when possible
- Delete must be idempotent for unknown IDs
- Filtering semantics should be consistent across query & delete

----------------------------------------
Graph Protocol (BaseGraphAdapter, Graph V1.0)
----------------------------------------
Required overrides:
- _do_capabilities(...)
- _do_health(...)
- _do_query(...)
- _do_stream_query(...)
- _do_upsert_nodes(...)
- _do_upsert_edges(...)
- _do_delete_nodes(...)
- _do_delete_edges(...)
- _do_bulk_vertices(...)
- _do_batch(...)
- _do_get_schema(...), if schema is supported

Behavior:
- Query and stream_query must use equivalent semantics (streamed summary matches)
- Upserts should support per-item validation and failure recording
- Deletes should be idempotent
- bulk_vertices must implement deterministic pagination (cursor, has_more)
- batch must return per-op result objects with clear success/failure info

===============================================================================
2. Canonical Error Mapping (MANDATORY)
===============================================================================

You must map provider-specific errors into Corpus canonical error types.

Common canonical errors:

- BadRequest
  - Invalid parameters, malformed queries, unsupported options
- NotSupported
  - Unsupported operations, dialects, or metrics
- ModelNotAvailable
  - Unknown/disabled model (LLM/Embedding/Vector, if applicable)
- DimensionMismatch
  - Vector dimension mismatch vs namespace/index schema
- IndexNotReady
  - Vector index exists but is not ready / not populated
- Unavailable
  - Backend down, overloaded, non-retryable temporary failures
- ResourceExhausted
  - Rate limits, quota exhausted, “too many requests”
  - Include retry_after_ms where possible
- TransientNetwork
  - Network blips, timeouts, connection resets, etc.
- DeadlineExceeded
  - When a request deadline has passed before or during an operation

Your generated code must:
- Catch the provider’s SDK/HTTP errors
- Re-raise as these canonical errors with useful messages and details

===============================================================================
3. Deadline Handling (MANDATORY)
===============================================================================

All provider calls must respect the Corpus operation context:

    ctx.remaining_ms()

Rules:
- If ctx is provided and exposes remaining_ms(), use this for timeouts.
- Subtract a small safety buffer (e.g. 50–100 ms).
- Apply this timeout to every provider API call:
  - LLM completions and streams
  - Embedding single/batch calls
  - Vector queries/upserts/deletes
  - Graph queries, streams, upserts, deletes, bulk scans, batch, schema fetch

If ctx or remaining_ms() is not available, use a sensible default timeout.

===============================================================================
4. Capabilities MUST Reflect Reality
===============================================================================

For each protocol, your _do_capabilities() implementation must accurately describe
the provider’s real behavior. This includes:

LLM:
- supported_models
- max_context_length
- supports_streaming
- supports_roles
- supports_system_message
- supports_json_output, supports_parallel_tool_calls (if any)
- supports_deadline
- supports_count_tokens

Embedding:
- supported_models
- max_text_length
- max_batch_size
- max_dimensions
- supports_normalization
- supports_truncation
- supports_token_counting
- supports_multi_tenant
- supports_deadline
- normalizes_at_source

Vector:
- max_dimensions
- supported_metrics
- supports_namespaces
- supports_metadata_filtering
- supports_batch_operations
- max_batch_size
- supports_index_management
- supports_multi_tenant
- supports_deadline
- max_top_k
- max_filter_terms

Graph:
- supported_query_dialects
- supports_stream_query
- supports_namespaces
- supports_property_filters
- supports_bulk_vertices
- supports_batch
- supports_schema
- supports_multi_tenant
- idempotent_writes
- supports_deadline

You MUST NOT claim features the provider does not support.

===============================================================================
5. Mock Adapters as Canonical References
===============================================================================

Use the following mock adapters as the **authoritative references** for
structure, semantics, and behavior for each protocol.

They show:
- How to subclass the base adapters
- How to implement capabilities & health
- How to implement query/upsert/delete, single/batch, and streaming
- How to do partial failures and deterministic behavior
- How to raise canonical errors consistently

Paste these directly into the prompt when you use it:

----------------------------------------
Mock LLM Adapter (BaseLLMAdapter)
----------------------------------------
```python
{{PASTE MockLLMAdapter CODE HERE}}


⸻

Mock Embedding Adapter (BaseEmbeddingAdapter)

{{PASTE MockEmbeddingAdapter CODE HERE}}


⸻

Mock Vector Adapter (BaseVectorAdapter)

{{PASTE MockVectorAdapter CODE HERE}}


⸻

Mock Graph Adapter (BaseGraphAdapter)

{{PASTE MockGraphAdapter CODE HERE}}

When implementing a real provider, follow these patterns but swap in actual
backend calls and real capabilities.

===============================================================================
6. Provider-Specific Details (FILL THESE IN)

I will now describe the specific provider and what I need:
	•	Provider name: {{PROVIDER_NAME}}
	•	Adapter class name: {{ProviderName}}Adapter
	•	Protocols to implement (choose any subset of LLM, Embedding, Vector, Graph):
{{e.g. “LLM + Embedding + Vector”}}
	•	Provider API documentation:
{{URL(s), OpenAPI/Swagger refs, or a concise description}}

For LLM (if selected):
	•	Supported models and their properties:
{{MODEL_LIST_AND_NOTES}}
	•	Max context length per model:
{{LIMITS}}
	•	Streaming support (which models support streaming?):
{{DETAILS}}
	•	Token counting support (how to obtain it):
{{DETAILS}}

For Embedding (if selected):
	•	Supported embedding models:
{{EMBED_MODEL_LIST}}
	•	Embedding dimensions per model:
{{DIMENSION_MAP}}
	•	Max text length / input limits:
{{LIMITS}}
	•	Whether provider returns token counts:
{{YES/NO AND HOW}}

For Vector (if selected):
	•	Index / namespace concept in this provider:
{{DESCRIBE HOW}}
	•	Supported distance metrics:
{{METRICS}}
	•	Max dimensions:
{{MAX_DIMENSIONS}}
	•	Max batch sizes / top_k limits:
{{LIMITS}}
	•	Filter capabilities (equality-only? richer?):
{{DETAILS}}
	•	Whether the provider manages namespaces / collections directly:
{{DETAILS}}

For Graph (if selected):
	•	Underlying graph engine: {{e.g. Neo4j, Neptune, Cosmos, etc.}}
	•	Supported query dialects:
{{e.g. “cypher”, “opencypher”, “gql”}}
	•	How namespaces/tenants map to the backend:
{{DETAILS}}
	•	Whether server supports streaming query responses:
{{YES/NO AND HOW}}
	•	Whether bulk vertex scans are supported, and limits:
{{DETAILS}}
	•	How to fetch schema (if supported) or whether it must be synthesized:
{{DETAILS}}
	•	Max operations per batch request:
{{MAX_OPS}}

===============================================================================
7. Output Requirements

You must produce:

⸻

A. Single Python Adapter Class

Generate a single adapter class:

class {{ProviderName}}Adapter(...):
    ...

Requirements:
	•	Subclass the correct base adapter classes for the selected protocols:
	•	BaseLLMAdapter for LLM
	•	BaseEmbeddingAdapter for Embedding
	•	BaseVectorAdapter for Vector
	•	BaseGraphAdapter for Graph
	•	Implement ONLY the do* methods required for those protocols.
	•	Wire them to real {{PROVIDER_NAME}} API calls (no mock/random behavior).
	•	Apply canonical error mapping everywhere.
	•	Use ctx.remaining_ms() for all provider timeouts.
	•	Populate capabilities() accurately for each protocol.
	•	Produce clean, idiomatic, production-appropriate Python.

⸻

B. Configuration Checklist

After the code, output a concise checklist of things I must configure manually:
	•	Environment variables / API keys / secrets
	•	Endpoint / region / project / database / graph identifiers
	•	Model names and dimensions (if applicable)
	•	Index / namespace / collection names (for Vector/Graph)
	•	Limits (batch sizes, top_k, timeouts) and any provider hard limits
	•	Dialects and schema configuration for Graph
	•	Any provider-specific quirks or required parameters

===============================================================================
8. Begin

Using everything above and the mock adapters as reference, generate:
	1.	The complete {{ProviderName}}Adapter class, and
	2.	The configuration checklist.

Do not add extra prose beyond the code and the checklist.

