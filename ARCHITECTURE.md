```markdown
# Architecture

## Design Philosophy

The Adapter SDK follows these core principles:

### 1. Minimal Surface Area
Only core operations are included in each protocol. No vendor-specific extensions, no convenience wrappers, no opinionated abstractions. Each protocol defines the smallest possible interface that enables production-grade adapter implementations.

### 2. Async-First
All operations are non-blocking and designed for high-concurrency environments. Every method returns an awaitable, enabling efficient resource utilization in production systems handling thousands of concurrent requests.

### 3. Production Hardened
Built-in support for:
- **Structured error taxonomy** with retry hints and operational guidance
- **Context propagation** for distributed tracing and deadlines
- **SIEM-safe metrics** that never expose PII or raw tenant identifiers
- **Capability discovery** for intelligent routing and feature detection

### 4. Extensible
Protocols use capability discovery to expose backend-specific features without polluting the core interface. Adapters declare what they support; routing layers make intelligent decisions.

### 5. Vendor Agnostic
No assumptions about specific databases, models, or providers. Works equally well with OpenAI, Anthropic, Neo4j, Pinecone, or any custom implementation.

---

## Protocol Suite

The Adapter SDK consists of four independent protocols, each optimized for its domain:

### Graph Protocol V1
**Purpose:** Vendor-neutral interface for graph databases

**Core Operations:**
- `create_vertex()` / `create_edge()` - Entity creation
- `delete_vertex()` / `delete_edge()` - Entity deletion
- `query()` / `stream_query()` - Query execution with dialect support
- `bulk_vertices()` - Bulk operations
- `batch()` - Atomic batch execution
- `get_schema()` - Schema introspection
- `capabilities()` - Feature discovery
- `health()` - Health monitoring

**Domain-Specific Features:**
- Multi-dialect query support (Cypher, Gremlin, GQL)
- Schema operations for graph structure management
- Batch operations for atomic multi-entity changes
- Streaming query results for large result sets

**Supported Backends:** Neo4j, Neptune, JanusGraph, TigerGraph, ArangoDB, etc.

---

### LLM Protocol V1
**Purpose:** Vendor-neutral interface for Large Language Models

**Core Operations:**
- `complete()` - Full response generation with token accounting
- `stream()` - Streaming response generation
- `count_tokens()` - Token counting for cost estimation
- `capabilities()` - Model feature discovery
- `health()` - Service health monitoring

**Domain-Specific Features:**
- Token usage accounting (prompt + completion)
- Progressive streaming with metadata
- Model family routing (GPT-4, Claude, Gemini)
- Finish reason tracking (stop, length, error, tool_call)

**Supported Backends:** OpenAI, Anthropic, Google, Cohere, local models, etc.

---

### Vector Protocol V1
**Purpose:** Vendor-neutral interface for vector similarity search

**Core Operations:**
- `query()` - Similarity search with filtering
- `upsert()` - Insert or update vectors
- `delete()` - Remove vectors by ID or filter
- `create_namespace()` / `delete_namespace()` - Namespace management
- `capabilities()` - Feature discovery
- `health()` - Service health monitoring

**Domain-Specific Features:**
- Namespace/collection isolation for multi-tenancy
- Metadata filtering for pre-search constraints
- Multiple distance metrics (cosine, euclidean, dot product)
- Dimension-aware operations
- Partial failure handling in batch operations

**Supported Backends:** Pinecone, Qdrant, Weaviate, Milvus, Chroma, etc.

---

### Embedding Protocol V1
**Purpose:** Vendor-neutral interface for text embedding generation

**Core Operations:**
- `embed()` - Single text embedding generation
- `embed_batch()` - Batch text embedding with partial failure support
- `count_tokens()` - Token counting for cost estimation
- `capabilities()` - Model feature discovery
- `health()` - Service health monitoring

**Domain-Specific Features:**
- Native batch operations (embeddings are naturally parallel)
- Multi-model support per adapter
- Text length validation and truncation
- Vector normalization support
- Token usage tracking for cost management

**Supported Backends:** OpenAI, Cohere, HuggingFace, Voyage, local models, etc.

---

## Architectural Layers

The Adapter SDK defines a clear separation between protocol (open source) and platform (your internal control plane):

```
┌─────────────────────────────────────────────────────────┐
│  Application Layer                                      │
│  (Your business logic)                                  │
└─────────────────────────────────────────────────────────┘
                           ▲
                           │
┌──────────────────────────┴──────────────────────────────┐
│  Control Plane (Internal - Closed Source)               │
│                                                          │
│  • BaseGraphProvider                                    │
│  • BaseLLMProvider                                      │
│  • BaseVectorProvider                                   │
│  • BaseEmbeddingProvider                                │
│                                                          │
│  Enhancement Layer:                                     │
│  - Redis cache managers (SWR + singleflight)           │
│  - Lane schedulers (batching/coalescing/DRR)           │
│  - Circuit breakers + backpressure                      │
│  - Distributed rate limiting                            │
│  - Retry engines (deadline + retry-after aware)        │
│  - Safety gates (content filtering)                     │
│  - Audit logging                                        │
└─────────────────────────────────────────────────────────┘
                           ▲
                           │ implements
                           │
┌──────────────────────────┴──────────────────────────────┐
│  Adapter SDK (Open Source Protocols)                    │
│                                                          │
│  • GraphProtocolV1                                      │
│  • LLMProtocolV1                                        │
│  • VectorProtocolV1                                     │
│  • EmbeddingProtocolV1                                  │
│                                                          │
│  Defines: Core operations, error taxonomy, types        │
└─────────────────────────────────────────────────────────┘
                           ▲
                           │ implements
                           │
┌──────────────────────────┴──────────────────────────────┐
│  Vendor Adapters (Community + Internal)                 │
│                                                          │
│  • Neo4jAdapter, PineconeAdapter, OpenAIAdapter, ...   │
└─────────────────────────────────────────────────────────┘
```

### Protocol Layer (Open Source)
**Responsibility:** Define the contract
- Minimal interface (core operations only)
- Error taxonomy with retry semantics
- Type definitions and validation
- Base adapter class with instrumentation
- Zero runtime dependencies

**What it does NOT do:**
- Caching strategies
- Retry logic
- Rate limiting
- Circuit breaking
- Routing decisions
- Policy enforcement

### Control Plane Layer (Internal)
**Responsibility:** Production enhancements
- Cache management (Redis with SWR)
- Request scheduling and batching
- Resilience patterns (circuit breakers, backpressure)
- Rate limiting (distributed + local)
- Retry with absolute deadlines
- Safety gates and content filtering
- Structured audit logging
- Cost tracking and quota management

---

## Error Handling Philosophy

### Structured Error Taxonomy

All protocols share a consistent error hierarchy designed for production systems:

```python
AdapterError (base)
├── BadRequest          # Client error - do not retry
├── AuthError           # Authentication failure - do not retry
├── ResourceExhausted   # Rate limit - retry with backoff
├── TransientNetwork    # Network failure - retry immediately
├── Unavailable         # Service down - retry with backoff
└── NotSupported        # Feature unavailable - do not retry
```

Each protocol adds domain-specific errors:
- **Graph:** (uses base taxonomy)
- **LLM:** `ModelOverloaded` (model-specific capacity issues)
- **Vector:** `DimensionMismatch`, `IndexNotReady`
- **Embedding:** `TextTooLong`, `ModelNotAvailable`

### Retry Hints

Errors include machine-readable retry guidance:

```python
class AdapterError(Exception):
    def __init__(
        self,
        message: str,
        *,
        retry_after_ms: Optional[int] = None,      # Suggested retry delay
        throttle_scope: Optional[str] = None,       # "tenant", "model", "global"
        suggested_batch_reduction: Optional[int] = None,  # Adaptive load shedding
        details: Optional[Mapping[str, Any]] = None,
    ):
        ...
```

This enables intelligent retry strategies in the control plane:

```python
try:
    result = await adapter.query(...)
except ResourceExhausted as e:
    if e.retry_after_ms:
        await asyncio.sleep(e.retry_after_ms / 1000)
        # Retry with server-specified delay
    if e.suggested_batch_reduction:
        # Reduce batch size by suggested percentage
        new_batch_size = batch_size * (1 - e.suggested_batch_reduction / 100)
```

---

## Context Propagation

All operations accept an `OperationContext` for:

### Distributed Tracing
```python
@dataclass(frozen=True)
class OperationContext:
    request_id: Optional[str] = None      # Correlation ID
    traceparent: Optional[str] = None     # W3C Trace Context
```

### Deadline Management
```python
    deadline_ms: Optional[int] = None     # Absolute epoch milliseconds
```

Deadlines are **absolute** (not relative) to prevent timeout compounding:
```python
# ❌ Wrong: Relative timeouts compound
timeout = 5.0  # 5 seconds
for retry in range(3):
    await call_with_timeout(fn, timeout)  # 5s + 5s + 5s = 15s total

# ✅ Right: Absolute deadline
deadline_ms = time.time() * 1000 + 5000  # 5 seconds from now
for retry in range(3):
    remaining = (deadline_ms - time.time() * 1000) / 1000
    await call_with_timeout(fn, remaining)  # Always respects 5s total
```

### Multi-Tenant Isolation
```python
    tenant: Optional[str] = None          # NEVER logged or exposed in metrics
```

Tenant identifiers are hashed in metrics (SHA256, truncated to 12 chars) for GDPR/compliance:

```python
def _tenant_hash(tenant: Optional[str]) -> Optional[str]:
    if not tenant:
        return None
    return hashlib.sha256(tenant.encode()).hexdigest()[:12]
```

### Idempotency
```python
    idempotency_key: Optional[str] = None # For at-most-once semantics
```

### Extensibility
```python
    attrs: Mapping[str, Any] = None       # Custom attributes for middleware
```

---

## Observability

### SIEM-Safe Metrics

All protocols include a `MetricsSink` protocol for operational monitoring:

```python
class MetricsSink(Protocol):
    def observe(
        self, *, component: str, op: str, ms: float,
        ok: bool, code: str = "OK",
        extra: Optional[Mapping[str, Any]] = None
    ) -> None:
        """Record operation timing and status."""
        ...
    
    def counter(
        self, *, component: str, name: str,
        value: int = 1,
        extra: Optional[Mapping[str, Any]] = None
    ) -> None:
        """Increment a counter metric."""
        ...
```

**Privacy Guarantees:**
- ✅ Tenant identifiers are hashed (never raw)
- ✅ Low-cardinality dimensions only
- ✅ No PII in metrics
- ✅ GDPR/CCPA compliant by design

**Example Usage:**
```python
metrics.observe(
    component="llm",
    op="complete",
    ms=1234.5,
    ok=True,
    code="OK",
    extra={
        "model": "gpt-4",
        "tenant": _tenant_hash(tenant_id),  # Hashed!
        "tokens": 150
    }
)
```

### Structured Logging

Base adapters automatically record:
- Operation start/end
- Success/failure status
- Duration (milliseconds)
- Error codes (for failed operations)
- Low-cardinality dimensions

---

## Capability Discovery

All protocols implement capability discovery for runtime feature detection:

```python
@dataclass(frozen=True)
class GraphCapabilities:
    server: str                           # "neo4j", "neptune", etc.
    version: str                          # Backend version
    dialects: Tuple[str, ...]            # Supported query languages
    supports_txn: bool = True            # ACID transactions
    supports_schema_ops: bool = True     # Schema management
    max_batch_ops: Optional[int] = None  # Batch size limits
    ...
```

**Enables intelligent routing:**

```python
# Routing layer can choose optimal backend
caps = await adapter.capabilities()

if query_requires_cypher and "cypher" not in caps.dialects:
    # Route to Neo4j adapter instead
    adapter = neo4j_adapter

if batch_size > caps.max_batch_ops:
    # Split into multiple batches
    ...
```

---

## Versioning Strategy

### Protocol Versions

Each protocol is independently versioned following SemVer:

```python
GRAPH_PROTOCOL_VERSION = "1.0.0"
LLM_PROTOCOL_VERSION = "1.1.0"
VECTOR_PROTOCOL_VERSION = "1.0.0"
EMBEDDING_PROTOCOL_VERSION = "1.0.0"
```

### Version Compatibility

**MAJOR version (X.y.z)** - Breaking changes:
- Method signature changes
- Required parameter additions
- Behavior changes
- Return type structure changes

**MINOR version (x.Y.z)** - Backwards-compatible additions:
- New optional parameters
- New capabilities
- New helper methods
- New error classes (if additive)

**PATCH version (x.y.Z)** - Backwards-compatible fixes:
- Documentation clarifications
- Bug fixes in base classes
- Performance improvements
- Type hint corrections

### Adapter Compatibility

Adapters implicitly declare protocol compatibility by implementing the interface:

```python
class Neo4jAdapter(BaseGraphAdapter):
    """Compatible with GraphProtocolV1."""
    
    async def _do_capabilities(self) -> GraphCapabilities:
        return GraphCapabilities(
            server="neo4j",
            version="5.14.0",
            dialects=("cypher", "opencypher"),
            ...
        )
```

### Version Checking

Runtime compatibility validation:

```python
from graph import GRAPH_PROTOCOL_VERSION, ProtocolVersion

protocol = ProtocolVersion(GRAPH_PROTOCOL_VERSION)
if not protocol.is_compatible_with("1.0.0"):
    raise RuntimeError(f"Adapter requires protocol >= 1.0.0")
```

---

## Extension Points

### 1. Custom Error Coercers

Protocols allow registration of custom error mapping:

```python
def coerce_neo4j_error(err: Exception) -> Optional[Exception]:
    """Map Neo4j-specific errors to protocol errors."""
    if "Neo.ClientError.Statement.SyntaxError" in str(err):
        return BadRequest(str(err), code="INVALID_CYPHER")
    if "Neo.TransientError.Transaction.DeadlockDetected" in str(err):
        return TransientNetwork(str(err), retry_after_ms=100)
    return None

# Register with adapter
error_registry.register(coerce_neo4j_error)
```

### 2. Custom Metrics Implementations

Adapters accept any `MetricsSink` implementation:

```python
class PrometheusMetrics:
    """Prometheus metrics implementation."""
    
    def __init__(self, registry):
        self.latency = Histogram('adapter_latency_ms', ...)
        self.requests = Counter('adapter_requests_total', ...)
    
    def observe(self, *, component, op, ms, ok, code, extra=None):
        self.latency.labels(
            component=component,
            op=op,
            status="ok" if ok else "error"
        ).observe(ms)
        
        self.requests.labels(
            component=component,
            op=op,
            code=code
        ).inc()

adapter = Neo4jAdapter(metrics=PrometheusMetrics(registry))
```

### 3. Context Attributes

Operation context supports arbitrary attributes for middleware:

```python
ctx = OperationContext(
    request_id="req-123",
    tenant="tenant-456",
    attrs={
        "user_id": "user-789",
        "experiment": "new-ranking-v2",
        "priority": "high"
    }
)

result = await adapter.query(spec, ctx=ctx)
```

---

## Security Considerations

### 1. Tenant Isolation

**Raw tenant identifiers NEVER appear in:**
- Metrics (hashed with SHA256)
- Logs (use correlation IDs instead)
- Error messages returned to clients

**Tenant identifiers ARE used for:**
- Routing to correct database/namespace
- Rate limiting per tenant
- Quota enforcement
- Audit logs (internal only)

### 2. Credential Management

Protocols do **NOT** handle credentials. Adapters receive pre-configured clients:

```python
# ❌ Wrong: Adapter manages credentials
class Neo4jAdapter:
    def __init__(self, uri: str, username: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

# ✅ Right: Adapter receives authenticated client
class Neo4jAdapter:
    def __init__(self, driver: Driver, metrics: MetricsSink):
        self.driver = driver  # Already authenticated
        self._metrics = metrics
```

Credential management lives in your control plane:
- Secret retrieval (AWS Secrets Manager, Vault, etc.)
- Credential rotation
- Connection pooling

### 3. Input Validation

Base adapters validate all inputs before delegation:

```python
async def create_vertex(self, label: str, props: Mapping[str, Any], *, ctx):
    # ✅ Validation before backend call
    self._require_non_empty("label", label)
    validated_props = self._validate_properties(props)
    
    # Delegate to backend
    result = await self._do_create_vertex(label, validated_props, ctx=ctx)
    return result
```

Prevents injection attacks and malformed requests from reaching backends.

### 4. Rate Limiting Hints

Protocols include throttling information in errors:

```python
raise ResourceExhausted(
    "Rate limit exceeded",
    retry_after_ms=5000,
    throttle_scope="tenant"  # Per-tenant limit, not global
)
```

Control plane uses this for:
- Per-tenant rate limiting
- Adaptive backpressure
- Cost control

---

## Performance Considerations

### 1. Async Throughout

All operations are async to maximize concurrency:

```python
# ✅ Handle 1000s of concurrent requests efficiently
results = await asyncio.gather(
    adapter.query(spec1, ctx=ctx1),
    adapter.query(spec2, ctx=ctx2),
    adapter.query(spec3, ctx=ctx3),
    # ... thousands more
)
```

### 2. Streaming for Large Results

Graph and LLM protocols support streaming:

```python
# ✅ Stream large result sets without buffering
async for row in adapter.stream_query(dialect="cypher", text=query):
    process(row)  # Process incrementally
```

### 3. Batch Operations

Protocols encourage batching where appropriate:

```python
# Graph: Atomic batch operations
results = await adapter.batch([
    {"type": "create_vertex", "label": "Person", ...},
    {"type": "create_vertex", "label": "Company", ...},
    {"type": "create_edge", "label": "WORKS_AT", ...},
])

# Embedding: Native batch support
results = await adapter.embed_batch(
    BatchEmbedSpec(texts=["text1", "text2", ...], model="ada-002")
)
```

### 4. No Blocking Operations

Protocols prohibit synchronous/blocking operations:

```python
# ❌ Never do this in adapters
def sync_query(self, query):
    return requests.post(...)  # Blocking!

# ✅ Always async
async def query(self, spec, *, ctx):
    async with aiohttp.ClientSession() as session:
        return await session.post(...)  # Non-blocking
```

---

## Testing Strategy

### 1. Protocol Compliance Tests

Each protocol includes a compliance test suite:

```python
# tests/test_graph_compliance.py
class GraphProtocolComplianceTests:
    """Run this suite against your adapter to verify compliance."""
    
    @pytest.mark.asyncio
    async def test_create_vertex_returns_graph_id(self, adapter):
        result = await adapter.create_vertex("Person", {"name": "Alice"})
        assert isinstance(result, str)  # GraphID is NewType of str
    
    @pytest.mark.asyncio
    async def test_bad_request_on_empty_label(self, adapter):
        with pytest.raises(BadRequest):
            await adapter.create_vertex("", {"name": "Alice"})
```

### 2. Adapter Testing

Adapter authors run compliance tests:

```bash
pytest tests/test_graph_compliance.py --adapter=my_adapter.Neo4jAdapter
```

### 3. Integration Testing

Control plane tests with real adapters:

```python
@pytest.mark.integration
async def test_graph_provider_with_neo4j():
    adapter = Neo4jAdapter(driver=neo4j_driver)
    provider = BaseGraphProvider(
        connector=adapter,
        settings=settings,
        breaker=breaker,
        cache_manager=cache_manager
    )
    
    result = await provider.create_vertex(
        label="Person",
        props={"name": "Alice"},
        tenant_id="test-tenant"
    )
    assert result.id
```

---

## Migration Guide

### From Vendor SDKs to Adapter SDK

**Before (Direct SDK Usage):**
```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

with driver.session() as session:
    result = session.run("CREATE (p:Person {name: $name}) RETURN id(p)", name="Alice")
    vertex_id = result.single()[0]
```

**After (Using Adapter SDK):**
```python
from graph import Neo4jAdapter, OperationContext

adapter = Neo4jAdapter(driver=driver, metrics=metrics)

ctx = OperationContext(
    request_id="req-123",
    tenant="tenant-456",
    deadline_ms=int(time.time() * 1000) + 5000
)

vertex_id = await adapter.create_vertex(
    label="Person",
    props={"name": "Alice"},
    ctx=ctx
)
```

**Benefits:**
- ✅ Consistent error handling
- ✅ Built-in metrics and tracing
- ✅ Multi-tenant support
- ✅ Deadline management
- ✅ Vendor portability

### From Other Adapter Patterns

**LangChain-style:**
```python
# LangChain: Opinionated abstractions
from langchain.vectorstores import Pinecone
db = Pinecone.from_documents(docs, embeddings)
results = db.similarity_search(query, k=5)

# Adapter SDK: Minimal protocol
from vector import PineconeAdapter
adapter = PineconeAdapter(client=pinecone_client)
results = await adapter.query(
    QuerySpec(vector=query_vector, top_k=5, namespace="default")
)
```

**Haystack-style:**
```python
# Haystack: Document-oriented abstractions
from haystack.document_stores import PineconeDocumentStore
store = PineconeDocumentStore(api_key="...")
store.write_documents(documents)

# Adapter SDK: Vector-oriented protocol
from vector import PineconeAdapter
adapter = PineconeAdapter(client=pinecone_client)
await adapter.upsert(
    UpsertSpec(vectors=[Vector(id="1", vector=[...], metadata={...})])
)
```

**Key Differences:**
- Adapter SDK: Minimal surface, production-grade
- LangChain/Haystack: Rich abstractions, experimental focus

---

## Best Practices

### 1. Always Use Context
```python
# ✅ Good: Propagate context for observability
ctx = OperationContext(
    request_id=generate_request_id(),
    tenant=tenant_id,
    deadline_ms=compute_deadline(),
    traceparent=extract_traceparent(headers)
)
result = await adapter.query(spec, ctx=ctx)

# ❌ Bad: No context
result = await adapter.query(spec)
```

### 2. Handle Partial Failures
```python
# ✅ Good: Check for partial failures in batch operations
result = await adapter.embed_batch(spec)
if result.failed_texts:
    logger.warning(f"Failed to embed {len(result.failed_texts)} texts")
    for failure in result.failed_texts:
        logger.error(f"Failed: {failure}")
```

### 3. Use Capabilities for Routing
```python
# ✅ Good: Check capabilities before routing
caps = await adapter.capabilities()
if query_tokens > caps.max_context_length:
    # Use different model or chunk the input
    adapter = long_context_adapter
```

### 4. Respect Retry Hints
```python
# ✅ Good: Honor server retry-after
try:
    result = await adapter.query(spec)
except ResourceExhausted as e:
    if e.retry_after_ms:
        await asyncio.sleep(e.retry_after_ms / 1000)
        result = await adapter.query(spec)  # Retry
```

### 5. Validate Early
```python
# ✅ Good: Validate before expensive operations
if not spec.texts:
    raise BadRequest("texts cannot be empty")

if len(spec.texts) > 1000:
    raise BadRequest("batch size exceeds maximum of 1000")

result = await adapter.embed_batch(spec)
```

---

## Future Directions

### Planned Protocol Extensions (V2)

**1. Structured Query Language (SQL for Graphs)**
- Unified query language across backends
- Compile to backend-specific dialects

**2. Transaction Protocol**
- Explicit `begin()` / `commit()` / `rollback()`
- Cross-backend distributed transactions

**3. Schema Evolution**
- Versioned schema changes
- Migration APIs

**4. Streaming Embeddings**
- Generate embeddings incrementally
- Useful for very large document sets

### Community Contributions

We welcome:
- New adapter implementations
- Protocol improvement proposals
- Example applications
- Documentation enhancements
- Compliance test coverage

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## References

### Related Standards
- **W3C Trace Context:** https://www.w3.org/TR/trace-context/
- **OpenTelemetry:** https://opentelemetry.io/
- **Semantic Versioning:** https://semver.org/

### Inspiration
- **DBAPI (PEP 249):** Python database API specification
- **SQLAlchemy Core:** Database abstraction layer
- **AWS SDK Retry Logic:** Production-grade error handling
- **gRPC:** Context propagation and deadlines

### Graph Query Languages
- **Cypher:** Neo4j's declarative query language
- **Gremlin:** Apache TinkerPop traversal language
- **GQL:** ISO standard graph query language

---

## License

All protocols in the Adapter SDK are licensed under Apache 2.0.

See [LICENSE](LICENSE) for full text.
```
