Corpus SDK Specification

Abstract

This specification defines the Corpus Protocol Suite: a vendor-neutral set of production-grade interfaces for graph databases, large language models, vector databases, and text embeddings. The suite standardizes contracts for heterogeneous AI infrastructure with built-in observability, error handling, and operational rigor. Protocols are minimal yet expressive, async-first, and extensible via negotiated capabilities. This document includes normative contracts, wire-compatible data shapes, an error taxonomy and mapping, resilience semantics, privacy and security guidance, and compatibility/versioning rules for enterprise-scale deployments.

Keywords: Graph Database, Large Language Model, Vector Search, Embeddings, Observability, Multi-Tenancy, Capability Discovery, Semantic Versioning, SIEM-Safe Telemetry, BCP 14

Status of This Memo

This document is not an Internet Standards Track specification; it is published for informational/standards-style guidance for the Corpus Protocol Suite. Distribution of this memo is unlimited.

Copyright Notice

Copyright © 2025 Corpus Protocol Suite.
SPDX-License-Identifier: Apache-2.0

⸻

Table of Contents
	1.	Introduction
1.1 Motivation • 1.2 Scope • 1.3 Design Philosophy
	2.	Requirements Language
	3.	Terminology
	4.	Conventions and Notation
	5.	Architecture Overview
5.1 Protocol Relationships • 5.2 Layered Architecture • 5.3 Implementation Profiles (Informative)
	6.	Common Foundation
6.1 Operation Context • 6.2 Capability Discovery • 6.3 Error Taxonomy • 6.4 Observability Interfaces
	7.	Graph Protocol V1 Specification
7.1 Overview • 7.2 Data Types • 7.3 Operations • 7.4 Dialects • 7.5 Schema Operations (Optional) • 7.6 Health
	8.	LLM Protocol V1 Specification
8.1 Overview • 8.2 Data Types • 8.3 Operations • 8.4 Model Discovery • 8.5 LLM-Specific Errors
	9.	Vector Protocol V1 Specification
9.1 Overview • 9.2 Data Types • 9.3 Operations • 9.4 Distance Metrics • 9.5 Vector-Specific Errors
	10.	Embedding Protocol V1 Specification
10.1 Overview • 10.2 Data Types (Normative) • 10.3 Operations (Normative) • 10.4 Errors (Embedding-Specific) • 10.5 Capabilities Shape (Normative) • 10.6 Validation & Numerical Requirements (Normative)
	11.	Cross-Protocol Patterns
	12.	Error Handling and Resilience
12.1 Retry Semantics • 12.2 Backoff and Jitter (RECOMMENDED) • 12.3 Circuit Breaking • 12.4 Error Mapping Table (Normative) • 12.5 Partial Failure Contracts • 12.6 Backpressure Integration
	13.	Observability and Monitoring
13.1 Metrics Taxonomy (MUST) • 13.2 Structured Logging (MUST) • 13.3 Distributed Tracing (SHOULD)
	14.	Security Considerations
14.1 Tenant Isolation (MUST) • 14.2 Authentication and Authorization (MUST) • 14.3 Threat Model (SHOULD)
	15.	Privacy Considerations
	16.	Performance Characteristics
16.1 Latency Targets (Indicative) • 16.2 Concurrency Limits • 16.3 Caching Strategies
	17.	Implementation Guidelines
	18.	Versioning and Compatibility
18.1 Semantic Versioning (MUST) • 18.2 Version Identification and Negotiation • 18.3 Backward Compatibility • 18.4 Deprecation Policy
	19.	IANA Considerations
	20.	References
20.1 Normative References • 20.2 Informative References
	21.	Author’s Address

Appendices:
A — End-to-End Example (Normative) • B — Capability Shapes (Illustrative) • C — Wire-Level Envelopes (Optional) • D — Content Redaction Patterns (Normative) • E — Implementation Status (Non-Normative) • F — Change Log / Revision History (Non-Normative)

⸻

1. Introduction

1.1. Motivation

The proliferation of AI infrastructure has created a fragmented landscape of proprietary APIs and inconsistent interfaces. Fragmentation increases integration complexity, reduces operational visibility, and creates vendor lock-in. Enterprise teams need cohesive, auditable, and performance-predictable interfaces that allow swapping providers without rewriting core application code or telemetry pipelines.

1.2. Scope

This specification defines four complementary protocols:
	•	Graph Protocol V1.1 — Vertex/edge CRUD, traversal, and multi-dialect query execution.
	•	LLM Protocol V1.1 — Chat-style completion, streaming tokens, usage accounting.
	•	Vector Protocol V1.1 — Vector upsert/delete, similarity search, and namespace management.
	•	Embedding Protocol V1.1 — Text embedding generation (single/batch), token counting, capability discovery, and health reporting.

All protocols share a Common Foundation (context propagation, capability discovery, error taxonomy, observability, resilience).

1.3. Design Philosophy
	•	Minimal Surface Area (MUST). Only essential operations are standardized. Vendor extensions appear via capabilities, not new methods.
	•	Async-First (MUST). All operations are non-blocking and concurrency-safe.
	•	Production-Hardened (MUST). Observability, error taxonomy, and resilience are first-class.
	•	Extensible (SHOULD). Capability negotiation enables optional features without breaking compatibility.
	•	Type-Safe (SHOULD). Strong typing and runtime validation minimize undefined behavior.
	•	Privacy by Design (MUST). SIEM-safe telemetry, data minimization, and redaction are defaults.

⸻

2. Requirements Language

The key words “MUST”, “MUST NOT”, “REQUIRED”, “SHALL”, “SHALL NOT”, “SHOULD”, “SHOULD NOT”, “RECOMMENDED”, “NOT RECOMMENDED”, “MAY”, and “OPTIONAL” in this document are to be interpreted as described in BCP 14 [RFC2119] [RFC8174] when, and only when, they appear in all capitals.

⸻

3. Terminology

Adapter — Concrete implementation of a protocol for a specific provider/backend.
Protocol — Interface contract that adapters MUST implement.
Operation Context — Metadata container for tracing, deadlines, and tenancy.
Capabilities — Dynamically discoverable features and limits of an adapter.
SIEM-Safe — Observability that excludes PII and uses privacy-preserving identifiers.
Idempotency Key — Client-provided token guaranteeing idempotent semantics.
Tenant Isolation — Logical separation of data/control plane in multi-tenant deployments.
Backpressure — Cooperative throttling to keep systems within safe operating limits.

⸻

4. Conventions and Notation
	•	JSON keys are case-sensitive; unknown keys MUST be ignored by clients and servers.
	•	Durations are expressed in milliseconds unless otherwise specified.
	•	Examples are non-normative unless explicitly marked (Normative).
	•	Field names use lower_snake_case unless specified.
	•	Error code values use UPPER_SNAKE_CASE.
	•	“tenant_hash” denotes a deterministic, irreversible hash of the tenant identifier.
	•	Unless otherwise stated, scores are higher is better (cosine/dot); distance metrics MAY be inverted to scores.

⸻

5. Architecture Overview

5.1. Protocol Relationships

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Graph Protocol│    │   LLM Protocol  │    │  Vector Protocol│    │ Embedding Proto │
│  • CRUD/Query   │    │  • Completion   │    │  • Search       │    │  • Single/Batch │
│  • Dialects     │    │  • Streaming    │    │  • Upsert/Delete│    │  • Token Count  │
│                 │    │  • Token Usage  │    │  • Namespaces   │    │  • Health       │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │                       │
         └───────────────────────┼───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────────┐
                    │  Common Foundation  │
                    │ • Context           │
                    │ • Errors            │
                    │ • Observability     │
                    │ • Resilience        │
                    └─────────────────────┘

5.2. Layered Architecture

┌─────────────────────────────────────────┐
│            Application Layer            │
│ (Orchestrates Graph/LLM/Vector/Embed)   │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│             Protocol Layer              │
│ (Uniform contracts with capabilities)   │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│              Adapter Layer              │
│ (Provider-specific implementations)     │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│               Backend Layer             │
│ (Databases, models, vector & embed svc) │
└─────────────────────────────────────────┘

5.3. Implementation Profiles (Informative)

Two implementation profiles are recognized. Profiles MUST NOT alter wire contracts or semantics.
	•	Thin (default): All infra hooks (cache, limiter, breaker) are no-ops; deadlines are propagated to downstream backends.
	•	Standalone: Optional local enforcement for deadlines, a small circuit breaker, a simple token-bucket limiter, and short-TTL in-memory caches (eligible for Embedding and LLM complete only).

⸻

6. Common Foundation

6.1. Operation Context

from dataclasses import dataclass
from typing import Any, Mapping, Optional

@dataclass(frozen=True)
class OperationContext:
    request_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    deadline_ms: Optional[int] = None
    traceparent: Optional[str] = None  # W3C Trace Context
    tenant: Optional[str] = None       # never logged raw
    attrs: Optional[Mapping[str, Any]] = None

Normative behavior:
	•	request_id (SHOULD) uniquely ties together end-to-end operations.
	•	idempotency_key (MAY) be supplied on mutating operations; when present, adapters MUST guarantee exactly-once external effects or return a consistent prior result.
	•	deadline_ms (SHOULD) be treated as an absolute deadline; adapters MUST propagate remaining budget downstream when possible. If the budget is already elapsed at call time, adapters SHOULD fail fast with DeadlineExceeded (LLM/Embedding) or Unavailable (Graph/Vector).
	•	traceparent (MUST) be forwarded unchanged for distributed tracing.
	•	tenant (MUST) enforce isolation; telemetry MUST NOT log raw tenant values.
	•	attrs (MUST) be treated as a map; if absent, treat as empty.

6.2. Capability Discovery

capabilities() MUST return a structured description:

{
  "server": "example-backend",
  "version": "5.17.1",
  "protocol": "graph/v1",
  "features": {
    "dialects": ["cypher", "opencypher"],
    "supports_txn": true,
    "supports_schema_ops": true,
    "supports_streaming": false,
    "supports_deadline": true
  },
  "limits": {
    "max_batch_ops": 5000,
    "rate_limit_qps": 200,
    "concurrency": 256,
    "max_query_length": 100000
  },
  "extensions": {
    "vendor:neo4j.routing": "cluster",
    "vendor:read_your_writes": true,
    "tag_model_in_metrics": false
  }
}

	•	Unknown features, limits, or extensions keys MUST be ignored by clients.
	•	extensions keys MUST be namespaced (e.g., vendor:foo.bar).
	•	Semantic changes to published keys MUST NOT occur without a major protocol version change.

6.3. Error Taxonomy

AdapterError (base)
├─ BadRequest              # 400 client errors (validation, schema)
├─ AuthError               # 401/403 authentication/authorization
├─ ResourceExhausted       # 429 quotas, rate limits (retry-after)
├─ TransientNetwork        # 5xx gateway/timeouts; retryable
├─ Unavailable             # 503 backend temporarily unavailable
├─ NotSupported            # 501/400 operation unsupported
└─ DeadlineExceeded        # 504 budget/deadline exhausted

LLM, Vector, and Embedding add specific subtypes (see §§8.5, 9.5, 10.4).

Errors MUST include machine-readable metadata when applicable:

{
  "error": "ResourceExhausted",
  "message": "Rate limit exceeded",
  "code": "RATE_LIMIT",
  "retry_after_ms": 1200,
  "throttle_scope": "tenant:acme:llm",
  "hint": "Reduce concurrency or increase backoff"
}

6.4. Observability Interfaces

from typing import Protocol, Mapping, Optional, Any

class MetricsSink(Protocol):
    def observe(
        self, *, component: str, op: str, ms: float, ok: bool,
        code: str = "OK", extra: Optional[Mapping[str, Any]] = None
    ) -> None: ...
    def counter(
        self, *, component: str, name: str, value: int = 1,
        extra: Optional[Mapping[str, Any]] = None
    ) -> None: ...

	•	component MUST be one of graph|llm|vector|embedding.
	•	Adapters MUST emit at least one observe per operation (streaming ops SHOULD emit a final outcome observe).
	•	Telemetry MUST be SIEM-safe: no raw tenant IDs, request bodies, prompts, vectors, or source texts.
	•	Implementations MAY include low-cardinality fields in extra: tenant_hash, deadline_bucket (<1s|<5s|<15s|<60s|>=60s), cache_hit (0/1), rows, batch_size, model (when extensions.tag_model_in_metrics=true).

⸻

7. Graph Protocol V1 Specification

7.1. Overview

Vendor-neutral interface for graph databases (Cypher, OpenCypher, Gremlin, GQL), standardizing CRUD, queries (sync/streaming), batch operations, and (optionally) schema management.

7.2. Data Types

from typing import NewType, Tuple, Optional
from dataclasses import dataclass

GraphID = NewType('GraphID', str)

@dataclass(frozen=True)
class GraphCapabilities:
    server: str
    version: str
    dialects: Tuple[str, ...] = ("cypher",)
    supports_txn: bool = True
    supports_schema_ops: bool = True
    max_batch_ops: Optional[int] = None
    retryable_codes: Tuple[str, ...] = ()
    rate_limit_unit: str = "requests_per_second"  # or "tokens_per_minute"
    max_qps: Optional[int] = None
    idempotent_writes: bool = False
    supports_multi_tenant: bool = False
    supports_streaming: bool = False
    supports_bulk_ops: bool = False
    # Optional limits/extensions:
    max_query_length: Optional[int] = None

7.3. Operations

7.3.1. Vertex/Edge CRUD

async def create_vertex(label: str, props: Mapping[str, Any], *, ctx: Optional[OperationContext]=None) -> GraphID
async def delete_vertex(vertex_id: GraphID, *, ctx: Optional[OperationContext]=None) -> None

async def create_edge(
    label: str, from_id: GraphID, to_id: GraphID, props: Mapping[str, Any], *,
    ctx: Optional[OperationContext]=None
) -> GraphID
async def delete_edge(edge_id: GraphID, *, ctx: Optional[OperationContext]=None) -> None

Semantics (Normative):
	•	props keys MUST be strings; values MUST be JSON-serializable.
	•	Create operations SHOULD accept idempotency_key and MUST be idempotent when provided.
	•	Deletes MUST be idempotent: deleting a non-existent ID returns success.

7.3.2. Queries

async def query(*, dialect: str, text: str, params: Optional[Mapping[str, Any]]=None,
                ctx: Optional[OperationContext]=None) -> List[Mapping[str, Any]]

async def stream_query(*, dialect: str, text: str, params: Optional[Mapping[str, Any]]=None,
                       ctx: Optional[OperationContext]=None) -> AsyncIterator[Mapping[str, Any]]

	•	dialect MUST be one advertised in capabilities.
	•	params MUST be bound safely; adapters MUST prevent injection by disallowing string interpolation.
	•	stream_query yields rows progressively; iterator close MUST free resources.
	•	Deadline behavior: if budget elapses mid-stream, adapters SHOULD terminate promptly with Unavailable (Thin) or DeadlineExceeded (Standalone).

7.3.3. Batch Operations

async def bulk_vertices(
    vertices: Iterable[Tuple[str, Mapping[str, Any]]], *,
    ctx: Optional[OperationContext]=None
) -> List[GraphID]

async def batch(
    ops: Iterable[Mapping[str, Any]], *,
    ctx: Optional[OperationContext]=None
) -> List[Mapping[str, Any]]

	•	Item-level atomicity: partial failures MUST be reported per item.
	•	batch supports vendor-optimized pipelines; supported op codes MUST be documented in capabilities.extensions.

7.4. Dialects

Supported (non-exhaustive): Cypher, OpenCypher, Gremlin, GQL (ISO). Unknown dialects MUST yield NotSupported.

7.5. Schema Operations (Optional)

If supports_schema_ops=true, adapters MAY expose:

async def create_index(label: str, property: str, unique: bool=False, *, ctx: Optional[OperationContext]=None) -> None
async def drop_index(label: str, property: str, *, ctx: Optional[OperationContext]=None) -> None

Index creation events MUST NOT log sample property values.

7.6. Health

Adapters SHOULD normalize unexpected failures in health() to Unavailable("health check failed") and still report server/version when known.

⸻

8. LLM Protocol V1 Specification

8.1. Overview

Standardized interface for chat-style completions with synchronous and streaming modes, token accounting, and deterministic sampling control. Capability-based model discovery provides portability.

8.2. Data Types

from dataclasses import dataclass
from typing import Optional

@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

@dataclass
class LLMCompletion:
    text: str
    model: str
    model_family: str       # e.g., "gpt", "llama", "mistral"
    usage: TokenUsage
    finish_reason: str      # "stop" | "length" | "tool_call" | "content_filter"

@dataclass
class LLMChunk:
    text: str
    is_final: bool = False
    model: Optional[str] = None
    usage_so_far: Optional[TokenUsage] = None

8.3. Operations

from typing import Optional, AsyncIterator

async def complete(
    *, messages: list[dict[str, str]],
    max_tokens: Optional[int]=None, temperature: Optional[float]=None,
    top_p: Optional[float]=None, model: Optional[str]=None,
    system_message: Optional[str]=None, frequency_penalty: Optional[float]=None,
    presence_penalty: Optional[float]=None, ctx: Optional[OperationContext]=None
) -> LLMCompletion

async def stream(
    *, messages: list[dict[str, str]],
    max_tokens: Optional[int]=None, temperature: Optional[float]=None,
    model: Optional[str]=None, system_message: Optional[str]=None,
    ctx: Optional[OperationContext]=None
) -> AsyncIterator[LLMChunk]

async def count_tokens(text: str, *, model: Optional[str]=None,
                       ctx: Optional[OperationContext]=None) -> int

Message format (MUST):

[
  {"role":"system","content":"You are a helpful assistant."},
  {"role":"user","content":"Summarize this..."}
]

Validation (Normative):
	•	Roles are from {system,user,assistant,tool}; unknown roles → BadRequest.
	•	temperature ∈ [0,2]; top_p ∈ (0,1]; frequency_penalty, presence_penalty ∈ [-2,2]; invalid ranges → BadRequest.
	•	count_tokens is gated by capability supports_count_tokens; if false, return NotSupported.

Deadline semantics: If budget is pre-expired or elapses during execution, return DeadlineExceeded. Streaming MUST set is_final=true on the terminal chunk; if stopping due to deadline, emit a final outcome metric reflecting the error.

8.4. Model Discovery

capabilities() MUST include:

{
  "models": [
    {"name":"gpt-4.1-mini","family":"gpt","context_window":128000,"supports_tools":true}
  ],
  "sampling": {"temperature_range":[0.0,2.0],"top_p_range":[0.0,1.0]},
  "features": {
    "supports_streaming": true,
    "supports_roles": true,
    "supports_json_output": false,
    "supports_parallel_tool_calls": false,
    "supports_deadline": true,
    "supports_count_tokens": true
  },
  "limits": {"max_context_length": 128000}
}

Clients MAY preflight prompt_tokens + max_tokens ≤ max_context_length when token counting is supported.

8.5. LLM-Specific Errors
	•	ModelOverloaded (subtype Unavailable): transient capacity pressure.
	•	ContentFiltered (subtype BadRequest): provider content policy triggered.
	•	DeadlineExceeded: budget exhausted; retryable only if deadline/size/sampling adjusted.

⸻

9. Vector Protocol V1 Specification

9.1. Overview

Standardized vector storage, search, and namespace isolation across providers with configurable distance metrics and metadata filtering.

9.2. Data Types

from typing import Any, Optional, NewType, List, Dict
from dataclasses import dataclass

VectorID = NewType('VectorID', str)

@dataclass(frozen=True)
class Vector:
    id: VectorID
    vector: List[float]
    metadata: Optional[Dict[str, Any]] = None
    namespace: Optional[str] = None

@dataclass(frozen=True)
class VectorMatch:
    vector: Vector
    score: float
    distance: float

@dataclass(frozen=True)
class QueryResult:
    matches: List[VectorMatch]
    query_vector: List[float]
    namespace: str
    total_matches: int

Spec types (aligned):

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass(frozen=True)
class QuerySpec:
    vector: List[float]
    top_k: int
    namespace: str = "default"
    filter: Optional[Dict[str, Any]] = None
    include_metadata: bool = True
    include_vectors: bool = False   # if False, VectorMatch.vector.vector MAY be omitted

@dataclass(frozen=True)
class UpsertSpec:
    vectors: List[Vector]
    namespace: str = "default"

@dataclass(frozen=True)
class DeleteSpec:
    ids: List[VectorID]
    namespace: str = "default"
    filter: Optional[Dict[str, Any]] = None

9.3. Operations

async def query(spec: QuerySpec, *, ctx: Optional[OperationContext]=None) -> QueryResult
async def upsert(spec: UpsertSpec, *, ctx: Optional[OperationContext]=None) -> dict
async def delete(spec: DeleteSpec, *, ctx: Optional[OperationContext]=None) -> dict
async def create_namespace(spec: dict, *, ctx: Optional[OperationContext]=None) -> dict
async def delete_namespace(namespace: str, *, ctx: Optional[OperationContext]=None) -> dict

Semantics (Normative):
	•	Vector dimension MUST match the configured index dimension; mismatches MUST raise DimensionMismatch.
	•	query.top_k MUST be > 0 and bounded by limits.max_top_k if present.
	•	Metadata filters MUST be applied pre-search where supported; otherwise adapters MUST document post-filter behavior.
	•	Scores MUST reflect the “higher is better” convention unless explicitly documented.
	•	include_vectors=false MAY suppress returning raw vectors in matches; IDs/metadata remain available.

9.4. Distance Metrics
	•	Cosine (default)
	•	Euclidean (L2)
	•	Dot Product

Adapters MUST advertise supported metrics and scoring conventions.

9.5. Vector-Specific Errors
	•	DimensionMismatch — query/upsert vector dimension incompatible (non-retryable).
	•	IndexNotReady — background index build incomplete (retryable with retry_after_ms).

⸻

10. Embedding Protocol V1 Specification

10.1. Overview

Vendor-neutral, production-grade interface for generating text embeddings (single/batch), counting tokens, discovering capabilities, and health reporting. Aligns with the Common Foundation and resilience semantics.

Deliberate non-goals (Informative): preprocessing/chunking, fine-tuning, provider-specific vector post-processing; those belong to adjacent layers.

⸻

10.2. Data Types (Normative)

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

@dataclass(frozen=True)
class EmbeddingVector:
    """
    A single embedding result for an input text.
    Raw input text MUST NOT be included; text_hash is an optional fingerprint.
    """
    vector: List[float]                # length == dimensions, finite, non-NaN
    dimensions: int                    # > 0
    model: str                         # non-empty; must match result.model
    index: int                         # position in the input batch (0-based)
    text_hash: Optional[str] = None    # deterministic fingerprint of input text
    metadata: Optional[Dict[str, Any]] = None  # provider-specific, low-cardinality only

@dataclass(frozen=True)
class ItemFailure:
    """
    Per-item failure record for non-atomic batches.
    """
    index: int                         # 0-based input position
    error: str                         # normalized error class (e.g., "BadRequest", "ResourceExhausted")
    code: Optional[str] = None         # machine-readable code (UPPER_SNAKE_CASE)
    message: Optional[str] = None      # human-readable explanation
    retry_after_ms: Optional[int] = None
    suggested_batch_reduction: Optional[int] = None  # percentage [1..100]

@dataclass(frozen=True)
class EmbeddingResult:
    """
    Canonical embedding result. For single-item calls, len(embeddings) == 1.
    For batches, embeddings and failures share the same input index space.
    """
    embeddings: List[EmbeddingVector]  # may be empty if all items failed
    model: str                         # model actually used
    total_tokens: Optional[int] = None # total over all inputs, if available
    processing_time_ms: Optional[float] = None
    failures: Optional[List[ItemFailure]] = None    # present only if any failed

@dataclass(frozen=True)
class EmbeddingCapabilities:
    server: str                        # backend identity
    version: str                       # backend version
    supported_models: Tuple[str, ...]  # non-empty list of model identifiers
    # Limits (MAY be None if unknown)
    max_batch_size: Optional[int] = None
    max_text_length: Optional[int] = None           # characters or tokens (see flags)
    max_dimensions: Optional[int] = None
    # Features/flags
    supports_normalization: bool = False            # can return normalized vectors at source
    normalizes_at_source: bool = False              # if True, vectors are already L2-normalized
    supports_truncation: bool = True
    length_is_tokens: bool = True                   # True: max_text_length counts tokens; False: characters
    supports_token_counting: bool = True
    supports_deadline: bool = True
    idempotent_operations: bool = True
    supports_multi_tenant: bool = True
    # Optional extensions (namespaced keys)
    extensions: Optional[Dict[str, Any]] = None

Type and value constraints (MUST):
	•	All vector elements MUST be finite floats (no NaN, Inf); dimensions MUST equal len(vector) and be > 0.
	•	model MUST be a non-empty string and MUST be one of supported_models (unless a documented alias maps to one).
	•	For batches, EmbeddingVector.index and ItemFailure.index MUST reference the same 0-based input positions with no gaps or duplicates.
	•	failures MUST be omitted or empty when all items succeed; when present, each failure MUST correspond to an input item not present in embeddings for that index.
	•	text_hash (if provided) MUST be a deterministic, irreversible fingerprint (e.g., sha256(text)) and MUST NOT reveal the raw text.

⸻

10.3. Operations (Normative)

from dataclasses import dataclass
from typing import Optional, List, Mapping, Any, Dict

@dataclass(frozen=True)
class EmbedSpec:
    text: str
    model: str
    truncate: bool = True              # if False and input exceeds max_text_length => TextTooLong
    normalize: bool = False            # request L2-normalized output (if supported)
    attrs: Optional[Mapping[str, Any]] = None       # advisory, non-normative

@dataclass(frozen=True)
class BatchEmbedSpec:
    texts: List[str]                   # length > 0
    model: str
    truncate: bool = True
    normalize: bool = False
    attrs: Optional[Mapping[str, Any]] = None

# Interface
async def capabilities(*, ctx: Optional[OperationContext] = None) -> EmbeddingCapabilities: ...
async def health(*, ctx: Optional[OperationContext] = None) -> Dict[str, Any]: ...
async def embed(spec: EmbedSpec, *, ctx: Optional[OperationContext] = None) -> EmbeddingResult: ...
async def embed_batch(spec: BatchEmbedSpec, *, ctx: Optional[OperationContext] = None) -> EmbeddingResult: ...
async def count_tokens(text: str, *, model: Optional[str] = None,
                       ctx: Optional[OperationContext] = None) -> int: ...

Semantics (MUST/SHOULD):
	•	capabilities() MUST return a fully populated EmbeddingCapabilities; unknown extensions MUST be ignored by clients (Common §6.2).
	•	health() MUST return { ok: bool, server: str, version: str, models?: string[] }. Shape MUST be stable in degraded states (Common §7.6).
	•	embed() MUST return exactly one EmbeddingVector in embeddings on success; failures MUST be omitted.
	•	embed_batch() is non-atomic per item: successful items MUST appear in embeddings with their original indices; failed items MUST appear in failures with matching indices. Implementations SHOULD compute suggested_batch_reduction (percentage) on overload or resource errors (Common §12.5).
	•	count_tokens() MUST return a non-negative integer; if unsupported (supports_token_counting == False), adapters MUST raise NotSupported.
	•	If normalize=True and supports_normalization == False, adapters MUST raise NotSupported.
	•	If truncate=False and an input exceeds max_text_length, adapters MUST raise TextTooLong (Embedding-specific).
	•	If length_is_tokens=True, max_text_length refers to tokens; otherwise characters. Adapters SHOULD document tokenization behavior.
	•	Deadline semantics follow Common §6.1 and §12.4: pre-expired budgets MUST fail fast with DeadlineExceeded; elapsed budgets during execution SHOULD terminate promptly. No raw content may appear in errors or telemetry.

⸻

10.4. Errors (Embedding-Specific)
	•	TextTooLong (subtype of BadRequest) — input exceeds model maximum when truncate=False.
Client guidance: enable truncation or split text.
	•	ModelNotAvailable (subtype of NotSupported or Unavailable) — requested model not present/disabled.
Client guidance: select a model from supported_models or retry if transient.
	•	DeadlineExceeded — budget exhausted; conditionally retryable only if deadline extended or inputs reduced.

Mitigation hints (SHOULD): retry_after_ms, resource_scope ("model"|"token_limit"|"rate_limit"), suggested_batch_reduction (percentage). Error objects SHOULD include code (string) and informative message (Common §6.3, §12.4).

⸻

10.5. Capabilities Shape (Normative)

capabilities() MUST serialize to the following JSON-compatible envelope (illustrative fields shown):

{
  "server": "embed-service",
  "version": "2025-01-15",
  "supported_models": ["example-embed-1", "example-embed-2"],
  "max_batch_size": 512,
  "max_text_length": 16000,
  "max_dimensions": 1536,
  "supports_normalization": true,
  "normalizes_at_source": false,
  "supports_truncation": true,
  "supports_token_counting": true,
  "supports_deadline": true,
  "idempotent_operations": true,
  "supports_multi_tenant": true,
  "length_is_tokens": true,
  "extensions": {
    "vendor:tokenizer": "tiktoken/example-embed-1"
  }
}

Client rules (MUST):
	•	Treat unknown keys as ignorable; treat missing optional keys as None/unsupported.
	•	Enforce max_batch_size and max_text_length preflight where known.
	•	When normalizes_at_source == True, clients MUST NOT assume raw (unnormalized) vectors.

⸻

10.6. Validation & Numerical Requirements (Normative)
	•	All floats in EmbeddingVector.vector MUST satisfy isfinite(x); vectors MUST NOT contain NaN, ±Inf, or subnormal values outside IEEE-754 normal range if the backend documents such constraints.
	•	If normalize=True, returned vectors MUST be L2-normalized within a tolerance ε ≤ 1e-6.
	•	Dimensions MUST be consistent for all outputs of a single call; if multiple models are internally used (vendor ensembles), adapters MUST expose a single model string and a single dimensions value to callers.

⸻

11. Cross-Protocol Patterns

11.1. Unified Error Handling

Centralize error mapping to the normalized taxonomy; use mitigation hints (retry_after_ms, suggested_*) for adaptive clients.

11.2. Consistent Observability

Emit observe and counter across components (graph|llm|vector|embedding) with common labels (op, code, tenant_hash). Never log raw prompts, vectors, source texts, or tenant identifiers.

11.3. Context Propagation

Create one OperationContext at ingress and pass through all protocol calls; update remaining time budget between calls.

11.4. Idempotency and Exactly-Once

Where idempotency_key is accepted, mutations MUST be exactly-once or return the prior committed result.

11.5. Pagination and Streaming

Graph MAY stream rows; LLM MUST stream with a terminal chunk; Vector pagination is capability-gated; Embedding is request/response (no streaming) in V1.

11.6. Caching (Implementation Guidance)
	•	Eligible at base: Embedding (deterministic outputs), LLM complete (key includes sampling params).
	•	Typically router/infra: Vector/Graph result caching due to workload variability.
	•	Keying guidance: use content hashes; do not include raw content; include model/params.

⸻

12. Error Handling and Resilience

12.1. Retry Semantics

Retryable: TransientNetwork, ResourceExhausted (respect retry_after_ms), Unavailable, IndexNotReady.
Conditionally Retryable: DeadlineExceeded (only if deadline extended or work reduced).
Non-Retryable: BadRequest, AuthError, NotSupported, DimensionMismatch, ContentFiltered, TextTooLong (unless truncation enabled).

12.2. Backoff and Jitter (RECOMMENDED)

Exponential backoff (100–500 ms base, ×2 factor, 10–30 s cap) with full jitter. Prefer server-provided retry_after_ms when present.

12.3. Circuit Breaking

Fail fast with Unavailable("circuit open") when breaker is open; include retry_after_ms where possible to reduce thundering herds. Breakers are an implementation profile feature (§5.3).

12.4. Error Mapping Table (Normative)

Error Class	HTTP Mapping	Retryable	Client Guidance
BadRequest	400	No	Fix parameters; do not retry
AuthError	401/403	No	Refresh credentials; verify scopes
ResourceExhausted	429	Yes	Back off; honor retry_after_ms; reduce concurrency/batch
TransientNetwork	502/504	Yes	Exponential backoff + jitter; consider failover
Unavailable	503	Yes	Trip/bias breaker; failover if possible
NotSupported	501/400	No	Probe with capabilities(); use alternative feature
DimensionMismatch*	400	No	Align dimensions to index
IndexNotReady*	503	Yes	Retry after retry_after_ms
ModelOverloaded**	503	Yes	Reduce rate; try alternate family/model
ContentFiltered**	400	No	Sanitize/adjust prompt
TextTooLong***	400	No	Enable truncation or split text
DeadlineExceeded	504	It depends	Extend deadline or reduce work (max_tokens, batch size, etc.)

	•	Vector-specific. ** LLM-specific. *** Embedding-specific.

12.5. Partial Failure Contracts

Batch APIs MUST report per-item status. Non-atomic batches MUST NOT fail the entire batch due to a single item.

12.6. Backpressure Integration

Expose per-tenant semaphores to bound concurrency:

async with backpressure.acquire(f"{tenant}:{component}:{op}"):
    return await adapter.operation(...)


⸻

13. Observability and Monitoring

13.1. Metrics Taxonomy (MUST)

Operational: Latency (p50/p90/p99) by component+op+code, error rate by class, concurrency/queue length.
Business: LLM tokens processed, vector upserts/searches, graph ops executed, texts embedded.
Resource: Cache hit ratios, rate-limit utilization, breaker state.

13.2. Structured Logging (MUST)

SIEM-safe logs (examples):

{
  "kind": "vector.audit",
  "op": "query",
  "tenant_hash": "7d9f53d2f1ab",
  "trace_id": "abc-123",
  "status": "ok",
  "latency_ms": 45.2,
  "vectors_searched": 1,
  "matches_returned": 10
}

{
  "kind": "embedding.audit",
  "op": "embed_batch",
  "tenant_hash": "7d9f53d2f1ab",
  "trace_id": "def-456",
  "status": "ok",
  "latency_ms": 37.8,
  "texts": 32,
  "model": "example-embed-1"
}

13.3. Distributed Tracing (SHOULD)

Propagate traceparent. Use standard span attributes (component, op, tenant_hash, model, counts). For LLM streaming, emit child events per chunk. Record one final stream outcome observe after completion/error.

⸻

14. Security Considerations

14.1. Tenant Isolation (MUST)
	•	Graph: separate DBs/schemas or RBAC-scoped labels.
	•	LLM: per-tenant keys or dedicated instances.
	•	Vector: per-tenant namespaces/collections with ACLs.
	•	Embedding: per-tenant API keys; no cross-tenant caches without isolation keys.

14.2. Authentication and Authorization (MUST)

Credentials managed at adapter init; rotate via secret stores; never emit secrets in telemetry or errors.

14.3. Threat Model (SHOULD)

Address idempotency-key spoofing, prompt/graph injection, vector/embedding poisoning, and unbounded traversals via rate limiting, schema constraints, and timeouts.

⸻

15. Privacy Considerations

Do not log prompts, source texts, vectors, or raw tenant IDs. Hash tenant identifiers, time-bound log retention (≤30 days recommended), and require explicit, access-controlled opt-in for content retention. Provide DSAR-compatible export/delete pathways where applicable.

⸻

16. Performance Characteristics

16.1. Latency Targets (Indicative)
	•	Graph: CRUD 1–10 ms; queries 10–1000 ms; batch 100–5000 ms.
	•	LLM: token counting 1–5 ms; completion 100–30000 ms; streaming progressive.
	•	Vector: search 1–100 ms; batch upsert 10–1000 ms; index 1000–60000 ms.
	•	Embedding: single 5–50 ms; batch 10–1000 ms; token counting 1–5 ms.

Adapters SHOULD surface p90/p99 per op in capabilities().limits.

16.2. Concurrency Limits

Expose concurrency, rate_limit_qps, max_batch_ops/top_k, and memory considerations in capabilities.

16.3. Caching Strategies
	•	LLM: cache keyed by normalized messages + sampling params + system hash.
	•	Vector: generally router-layer; if implemented, fingerprint the full QuerySpec.
	•	Graph: fingerprint dialect + text + params.
	•	Embedding: content-addressable (model, normalize, sha256(text)); record only hashes in telemetry.

⸻

17. Implementation Guidelines

17.1. Adapter Pattern

Use base classes to centralize validation, error normalization, and metrics; focus provider code on business logic.

17.2. Validation (MUST)

Reject empty labels/texts, negative top_k, NaN/Inf vectors; enforce JSON-serializable props/metadata; validate message roles and max_tokens vs. window; enforce embedding max_text_length and max_batch_size; enforce sampling parameter ranges.

17.3. Testing

Unit: dimension mismatch, role/parameter validation, error mapping, batching limits.
Integration: end-to-end pipelines (Graph → LLM → Vector → Embedding).
Chaos: simulate Unavailable, timeouts, and rate-limit storms; verify backoff and breaker behavior; ensure idempotence and stream resource release.

⸻

18. Versioning and Compatibility

18.1. Semantic Versioning (MUST)

MAJOR (breaking), MINOR (additive), PATCH (non-breaking fixes/docs).

18.2. Version Identification and Negotiation

Clients MAY specify X-Adapter-Protocol: {component}/v{major}. Adapters MUST reject incompatible majors with NotSupported and SHOULD advertise supported versions in capabilities.protocol.

18.3. Backward Compatibility

Guaranteed for additive parameters/methods and new capability flags. Not guaranteed for changing required params, removing elements, or altering error semantics outside a major.

18.4. Deprecation Policy

Announce, warn at runtime (where feasible), maintain ≥1 major, then remove in the subsequent major.

⸻

19. IANA Considerations

No IANA actions required.

⸻

20. References

20.1. Normative References
	•	[RFC2119] S. Bradner, “Key words for use in RFCs to Indicate Requirement Levels,” BCP 14.
	•	[RFC8174] B. Leiba, “Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words,” BCP 14 update.
	•	[W3C-Trace-Context] W3C Recommendation, “Trace Context.”
	•	[OpenTelemetry-Spec] OpenTelemetry Specification.
	•	[SemVer] Semantic Versioning 2.0.0.

20.2. Informative References
	•	Corpus GitHub Repository — https://github.com/adapter-sdk

⸻

21. Author’s Address

Corpus Working Group
Email: standards@adaptersdk.org
GitHub: https://github.com/adapter-sdk/standards

⸻

Appendix A — End-to-End Example (Normative)

import time, random, asyncio
ctx = OperationContext(
    request_id="req_01HZX...",
    idempotency_key="idem_64f0...",
    deadline_ms=int(time.time()*1000)+30000,
    traceparent="00-4bf9...-00f0...-01",
    tenant="acme-corp",
    attrs={"user":"u_12345"}
)

# 1) Graph query for related documents
graph_rows = await graph_adapter.query(
    dialect="cypher",
    text="MATCH (u:User {id:$uid})-[:READ]->(d:Doc) RETURN d.id AS doc_id LIMIT 20",
    params={"uid":"u_12345"},
    ctx=ctx
)

# 2) LLM summarization
summary = await llm_adapter.complete(
    messages=[{"role":"system","content":"Summarize tersely."},
              {"role":"user","content":f"Summarize docs: {[r['doc_id'] for r in graph_rows]}"}],
    max_tokens=256,
    temperature=0.2,
    model="gpt-4.1-mini",
    ctx=ctx
)

# 3) Embedding + Vector search
embedding = embed(summary.text)  # implementation-specific
qr = await vector_adapter.query(
    QuerySpec(vector=embedding, top_k=10, namespace="acme.docs",
              filter={"doc_type":"kb","lang":{"$in":["en"]}}),
    ctx=ctx
)

Resilience loop (backoff + jitter):

for attempt in range(5):
    try:
        res = await vector_adapter.upsert(UpsertSpec(...), ctx=ctx)
        break
    except ResourceExhausted as e:
        await asyncio.sleep((e.retry_after_ms or 500)/1000)
    except (TransientNetwork, Unavailable):
        sleep = min(2**attempt * 0.2, 5.0) * random.random()
        await asyncio.sleep(sleep)
    except DeadlineExceeded:
        # increase deadline or reduce work before retrying
        break


⸻

Appendix B — Capability Shapes (Illustrative)

Graph

{
  "server": "janusgraph",
  "version": "1.0.0",
  "protocol": "graph/v1",
  "features": {
    "dialects":["gremlin"],
    "supports_txn":true,
    "supports_schema_ops":false,
    "supports_streaming":true,
    "supports_deadline":true
  },
  "limits": {"max_batch_ops": 2000, "concurrency": 128},
  "extensions": {"vendor:storage":"cassandra","read_after_write_consistency":"session"}
}

LLM

{
  "server": "chat-gateway",
  "version": "2024-10-01",
  "protocol": "llm/v1",
  "features": {
    "supports_tools":true,
    "supports_streaming":true,
    "supports_deadline":true,
    "supports_count_tokens":true
  },
  "limits": {"max_context_length": 128000},
  "sampling":{"temperature_range":[0.0,2.0],"top_p_range":[0.0,1.0]},
  "models": [
    {"name":"gpt-4.1-mini","family":"gpt","context_window":128000,"supports_tools":true}
  ]
}

Vector

{
  "server": "pinecone",
  "version": "2.2",
  "protocol": "vector/v1",
  "features": {"supports_pagination":true,"supports_filters":true,"supports_deadline":true},
  "limits": {"dimension":1536,"max_top_k":1000,"max_batch":1000},
  "extensions": {"metric_default":"cosine"}
}

Embedding

{
  "server": "embed-service",
  "version": "2025-01-15",
  "protocol": "embedding/v1",
  "supported_models": ["example-embed-1", "example-embed-2"],
  "max_batch_size": 512,
  "max_text_length": 16000,
  "max_dimensions": 1536,
  "supports_normalization": true,
  "normalizes_at_source": false,
  "supports_truncation": true,
  "supports_token_counting": true,
  "supports_deadline": true,
  "idempotent_operations": true,
  "supports_multi_tenant": true
}


⸻

Appendix C — Wire-Level Envelopes (Optional)

Request

{
  "op": "embedding.embed_batch",
  "ctx": {"request_id":"req_abc","deadline_ms":1730312345123,"tenant":"acme"},
  "args": {"texts":["a","b","c"],"model":"example-embed-1","normalize":true}
}

Response

{
  "ok": true,
  "code": "OK",
  "ms": 38.4,
  "result": {"embeddings":[{"dimensions":1536,"model":"example-embed-1"}], "model":"example-embed-1"}
}

Error

{
  "ok": false,
  "code": "DEADLINE_EXCEEDED",
  "error": "DeadlineExceeded",
  "message": "Operation budget exhausted",
  "retry_after_ms": null,
  "resource_scope": "time_budget"
}


⸻

Appendix D — Content Redaction Patterns (Normative)
	•	Replace user/tenant identifiers with irreversible hashes before logging.
	•	Replace prompts and graph query text with SHA-256 fingerprints; store full content only when explicit debug sampling is enabled and access-controlled.
	•	For vectors and embeddings, log only dimension and norm statistics (mean/std); never raw vectors or source texts.
	•	Telemetry exporters MUST implement field-level redaction lists configurable per deployment.

⸻

Appendix E — Implementation Status (Non-Normative)
	•	Reference Adapters: at least one open-source adapter per protocol family is RECOMMENDED for interoperability testing.
	•	Interop Suite: a conformance test suite SHOULD validate error mapping, capability negotiation, and streaming semantics.
	•	Release Quality Bar: adapters SHOULD demonstrate stability under soak tests (24h) and chaos scenarios (network partitions, rate-limit storms).

⸻

Appendix F — Change Log / Revision History (Non-Normative)
	•	v1.1 — Alignment & Deadlines:
	•	Added DeadlineExceeded error class and mapping (§6.3, §12.4).
	•	Clarified deadline semantics and streaming finalization across protocols.
	•	Extended capability flags: supports_deadline (all), supports_count_tokens (LLM), expanded Graph fields (supports_streaming, supports_bulk_ops, retryable_codes, rate_limit_unit, max_qps).
	•	Aligned Vector types: VectorMatch carries full Vector, plus score and distance; clarified include_vectors behavior.
	•	Added Implementation Profiles (§5.3) and observability enrichments (deadline_bucket, final stream outcome).
	•	v1.1 — Embedding Added (from previous RFC note): Added Embedding Protocol V1, expanded observability/security/privacy coverage.
	•	v1.0 — Initial RFC-Style: Introduced BCP 14 requirements language, IANA Considerations, split Normative/Informative references, explicit Privacy Considerations, Conventions and Notation, error-mapping table, capability namespacing rules, and appendices for examples, redaction, and wire envelopes.

⸻

End of Document