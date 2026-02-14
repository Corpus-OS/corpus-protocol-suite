# Deploying Your Certified Adapter

**Take Your Gold-Certified Adapter to Production**

---

> **📋 Document Status**
>
> This guide provides **practical deployment examples** (informative). For normative specifications:
>
> - **Wire Protocol** → [SCHEMA.md](../spec/SCHEMA.md) - Envelope format, fields, error codes
> - **Protocol Semantics** → [PROTOCOL.md](../spec/PROTOCOL.md) - Operation contracts, versioning
>
> **Validation:** Wire conformance tests in `tests/live/` validate against specs.  
> **Conflicts:** When this guide differs from SCHEMA.md or PROTOCOL.md, **the specs are correct**.

---

**Table of Contents**
- [0. Mental Model: Adapter → Service](#0-mental-model-adapter--service)
- [1. Prerequisites](#1-prerequisites)
- [2. Standalone HTTP Services (FastAPI)](#2-standalone-http-services-fastapi)
  - [2.1 Embedding Service](#21-embedding-service)
  - [2.2 LLM Service](#22-llm-service)
  - [2.3 Vector Service](#23-vector-service)
  - [2.4 Graph Service](#24-graph-service)
- [3. Testing Your HTTP Service](#3-testing-your-http-service)
- [4. Alternative Frameworks](#4-alternative-frameworks)
  - [4.1 Starlette (Async-First)](#41-starlette-async-first)
  - [4.2 Flask (Synchronous)](#42-flask-synchronous)
  - [4.3 Quart (Async Flask)](#43-quart-async-flask)
- [5. Deployment Patterns](#5-deployment-patterns)
  - [5.1 Docker + Kubernetes](#51-docker--kubernetes)
  - [5.2 AWS Lambda (Serverless)](#52-aws-lambda-serverless)
  - [5.3 Google Cloud Run](#53-google-cloud-run)
  - [5.4 Cloudflare Workers](#54-cloudflare-workers)
- [6. Production Considerations](#6-production-considerations)
  - [6.1 Environment Configuration](#61-environment-configuration)
  - [6.2 Logging & Observability](#62-logging--observability)
  - [6.3 Metrics & Monitoring](#63-metrics--monitoring)
  - [6.4 Rate Limiting (Beyond BaseAdapter)](#64-rate-limiting-beyond-baseadapter)
  - [6.5 Security (mTLS, API Keys)](#65-security-mtls-api-keys)
  - [6.6 Load Balancing & Scaling](#66-load-balancing--scaling)
- [7. Production Checklist](#7-production-checklist)
- [Appendix A: Full Service Examples](#appendix-a-full-service-examples)
- [Appendix B: Common Deployment Pitfalls](#appendix-b-common-deployment-pitfalls)
- [Appendix C: Debugging Production Issues](#appendix-c-debugging-production-issues)

---
> **Prerequisite:** You have a **Gold-certified adapter** from the [Quickstart](QUICKSTART.md).  
> **Goal:** Expose your adapter as a production-ready HTTP service.  
> **Time:** 20-30 minutes (depending on deployment target)
---

## 0. Mental Model: Adapter → Service

Your certified adapter is a Python class that implements `_do_*()` hooks. To expose it over HTTP, you need:

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   HTTP      │────►│  WireHandler │────►│  YourAdapter    │
│   Request   │◄────│  (envelope   │◄────│  (_do_* hooks)  │
└─────────────┘     │   parse/serialize)│ └─────────────────┘
                    └──────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  Provider   │
                    │    API      │
                    └─────────────┘
```

**Three layers:**
1. **HTTP Layer** (FastAPI/Starlette/Flask) - Receives requests, handles headers
2. **WireHandler** (from `corpus-sdk`) - Parses/serializes canonical envelopes
3. **YourAdapter** - Your certified implementation

**Critical:** The WireHandler does **all** the heavy lifting. Your HTTP layer is just a thin wrapper.

---

## 1. Prerequisites

```bash
# Install production dependencies
pip install fastapi uvicorn  # For HTTP services
pip install httpx             # For provider calls (already in adapter)
pip install redis             # Optional: for production idempotency cache
pip install prometheus-client # Optional: for metrics
pip install python-json-logger # Optional: for structured logging

# Your certified adapter from Quickstart
# Copy adapters/hello_embedding.py (or your protocol) to your project
```

---

## 2. Standalone HTTP Services (FastAPI)

Choose the service implementation that matches your protocol.

### 2.1 Embedding Service

`services/embedding_service.py`:

```python
import os
import time
import logging
from typing import Dict, Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

from corpus_sdk.embedding.embedding_base import WireEmbeddingHandler
from adapters.hello_embedding import HelloEmbeddingAdapter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("embedding-service")

# Initialize adapter with environment variables
adapter = HelloEmbeddingAdapter(
    api_key=os.environ.get("PROVIDER_API_KEY", ""),
    endpoint=os.environ.get("PROVIDER_ENDPOINT", None),
    mode=os.environ.get("ADAPTER_MODE", "standalone"),
)

# Wire handler does envelope parsing/serialization
handler = WireEmbeddingHandler(adapter)

# FastAPI app
app = FastAPI(
    title="Corpus Embedding Adapter",
    version="1.0.0",
    description="Production-ready embedding service",
)

@app.middleware("http")
async def add_protocol_header(request: Request, call_next):
    """Ensure protocol header is present."""
    response = await call_next(request)
    response.headers["X-Adapter-Protocol"] = "embedding/v1.0"
    return response

@app.post("/v1/embedding")
async def handle_embedding(request: Request) -> Dict[str, Any]:
    """Handle unary embedding operations."""
    # Protocol version negotiation
    protocol = request.headers.get("x-adapter-protocol")
    if protocol and protocol != "embedding/v1.0":
        return {
            "ok": False,
            "code": "NOT_SUPPORTED",
            "error": "NotSupported",
            "message": f"Protocol {protocol} not supported. Use embedding/v1.0",
            "ms": 0,
            "retry_after_ms": None,
            "details": {"supported_protocols": ["embedding/v1.0"]}
        }
    
    try:
        envelope = await request.json()
        start = time.time()
        response = await handler.handle(envelope)
        
        # Add timing if not present
        if "ms" not in response:
            response["ms"] = (time.time() - start) * 1000
            
        return response
        
    except Exception as e:
        logger.exception("Unhandled error in embedding endpoint")
        return {
            "ok": False,
            "code": "UNAVAILABLE",
            "error": "Unavailable",
            "message": "internal server error",
            "ms": (time.time() - start) * 1000 if 'start' in locals() else 0,
            "retry_after_ms": None,
            "details": None,
        }

@app.post("/v1/embedding/stream")
async def handle_embedding_stream(request: Request) -> StreamingResponse:
    """Handle streaming embedding operations."""
    protocol = request.headers.get("x-adapter-protocol")
    if protocol and protocol != "embedding/v1.0":
        return JSONResponse(
            status_code=400,
            content={
                "ok": False,
                "code": "NOT_SUPPORTED",
                "error": "NotSupported",
                "message": f"Protocol {protocol} not supported",
                "ms": 0,
                "retry_after_ms": None,
                "details": {"supported_protocols": ["embedding/v1.0"]}
            }
        )
    
    try:
        envelope = await request.json()
        
        async def stream_generator():
            start_time = time.time()
            async for chunk in handler.handle_stream(envelope):
                # Ensure canonical streaming format
                if isinstance(chunk, dict) and "chunk" not in chunk:
                    chunk = {
                        "ok": True,
                        "code": "STREAMING",  # Must be exactly "STREAMING"
                        "ms": (time.time() - start_time) * 1000,
                        "chunk": chunk
                    }
                yield json.dumps(chunk) + "\n"
                
        return StreamingResponse(
            stream_generator(),
            media_type="application/x-ndjson",
            headers={
                "X-Adapter-Protocol": "embedding/v1.0",
                "Cache-Control": "no-cache"
            }
        )
        
    except Exception as e:
        logger.exception("Error in streaming endpoint")
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "code": "UNAVAILABLE",
                "error": "Unavailable",
                "message": "streaming error",
                "ms": 0,
                "retry_after_ms": None,
                "details": None,
            }
        )

@app.get("/v1/health")
async def health_check() -> Dict[str, Any]:
    """Health endpoint with graded status."""
    health = await adapter.health()
    # Status codes per specification:
    # - 200: ok or degraded
    # - 503: down
    status_code = 200 if health.get("status") in ["ok", "degraded"] else 503
    return JSONResponse(status_code=status_code, content=health)

@app.get("/v1/capabilities")
async def get_capabilities() -> Dict[str, Any]:
    """Capabilities discovery endpoint."""
    caps = await adapter.capabilities()
    from dataclasses import asdict
    return asdict(caps)

@app.get("/v1/stats")
async def get_stats() -> Dict[str, Any]:
    """Service statistics endpoint."""
    stats = await adapter._do_get_stats()
    return stats

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    from prometheus_client import generate_latest, REGISTRY
    return Response(content=generate_latest(REGISTRY), media_type="text/plain")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "services.embedding_service:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
```

### 2.2 LLM Service

`services/llm_service.py`:

```python
import os
import time
import logging
from typing import Dict, Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

from corpus_sdk.llm.llm_base import WireLLMHandler
from adapters.hello_llm import HelloLLMAdapter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm-service")

# Initialize adapter
adapter = HelloLLMAdapter(
    api_key=os.environ.get("PROVIDER_API_KEY", ""),
    endpoint=os.environ.get("PROVIDER_ENDPOINT", None),
    mode=os.environ.get("ADAPTER_MODE", "standalone"),
)

handler = WireLLMHandler(adapter)

app = FastAPI(title="Corpus LLM Adapter", version="1.0.0")

@app.middleware("http")
async def add_protocol_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Adapter-Protocol"] = "llm/v1.0"
    return response

@app.post("/v1/llm/complete")
async def handle_complete(request: Request) -> Dict[str, Any]:
    """Handle LLM completion requests."""
    protocol = request.headers.get("x-adapter-protocol")
    if protocol and protocol != "llm/v1.0":
        return {
            "ok": False,
            "code": "NOT_SUPPORTED",
            "error": "NotSupported",
            "message": f"Protocol {protocol} not supported",
            "ms": 0,
            "retry_after_ms": None,
            "details": {"supported_protocols": ["llm/v1.0"]}
        }
    
    try:
        envelope = await request.json()
        start = time.time()
        response = await handler.handle(envelope)
        if "ms" not in response:
            response["ms"] = (time.time() - start) * 1000
        return response
    except Exception as e:
        logger.exception("Error in completion endpoint")
        return {
            "ok": False,
            "code": "UNAVAILABLE",
            "error": "Unavailable",
            "message": "internal error",
            "ms": (time.time() - start) * 1000 if 'start' in locals() else 0,
            "retry_after_ms": None,
            "details": None,
        }

@app.post("/v1/llm/stream")
async def handle_stream(request: Request) -> StreamingResponse:
    """Handle streaming LLM completions."""
    try:
        envelope = await request.json()
        
        async def stream_generator():
            start = time.time()
            async for chunk in handler.handle_stream(envelope):
                if "ms" not in chunk:
                    chunk["ms"] = (time.time() - start) * 1000
                yield json.dumps(chunk) + "\n"
                
        return StreamingResponse(
            stream_generator(),
            media_type="application/x-ndjson",
            headers={"X-Adapter-Protocol": "llm/v1.0"}
        )
    except Exception as e:
        logger.exception("Error in streaming endpoint")
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "code": "UNAVAILABLE",
                "error": "Unavailable",
                "message": "streaming error",
                "ms": 0,
                "retry_after_ms": None,
                "details": None,
            }
        )

@app.get("/v1/health")
async def health_check():
    health = await adapter.health()
    status_code = 200 if health.get("status") in ["ok", "degraded"] else 503
    return JSONResponse(status_code=status_code, content=health)

@app.post("/v1/llm/count_tokens")
async def count_tokens(request: Request) -> Dict[str, Any]:
    """Count tokens in text."""
    try:
        envelope = await request.json()
        start = time.time()
        response = await handler.handle(envelope)
        if "ms" not in response:
            response["ms"] = (time.time() - start) * 1000
        return response
    except Exception as e:
        logger.exception("Error in count_tokens endpoint")
        return {
            "ok": False,
            "code": "UNAVAILABLE",
            "error": "Unavailable",
            "message": "internal error",
            "ms": (time.time() - start) * 1000 if 'start' in locals() else 0,
            "retry_after_ms": None,
            "details": None,
        }

@app.get("/v1/capabilities")
async def get_capabilities():
    caps = await adapter.capabilities()
    from dataclasses import asdict
    return asdict(caps)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "services.llm_service:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
```

### 2.3 Vector Service

`services/vector_service.py`:

```python
import os
import time
import logging
from typing import Dict, Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
import uvicorn

from corpus_sdk.vector.vector_base import WireVectorHandler
from adapters.hello_vector import HelloVectorAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vector-service")

adapter = HelloVectorAdapter(
    api_key=os.environ.get("PROVIDER_API_KEY", ""),
    endpoint=os.environ.get("PROVIDER_ENDPOINT", None),
    mode=os.environ.get("ADAPTER_MODE", "standalone"),
)

handler = WireVectorHandler(adapter)

app = FastAPI(title="Corpus Vector Adapter", version="1.0.0")

@app.middleware("http")
async def add_protocol_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Adapter-Protocol"] = "vector/v1.0"
    return response

@app.post("/v1/vector/query")
async def handle_query(request: Request) -> Dict[str, Any]:
    """Execute vector similarity search."""
    protocol = request.headers.get("x-adapter-protocol")
    if protocol and protocol != "vector/v1.0":
        return {
            "ok": False,
            "code": "NOT_SUPPORTED",
            "error": "NotSupported",
            "message": f"Protocol {protocol} not supported",
            "ms": 0,
            "retry_after_ms": None,
            "details": {"supported_protocols": ["vector/v1.0"]}
        }
    
    try:
        envelope = await request.json()
        start = time.time()
        response = await handler.handle(envelope)
        if "ms" not in response:
            response["ms"] = (time.time() - start) * 1000
        return response
    except Exception as e:
        logger.exception("Error in query endpoint")
        return {
            "ok": False,
            "code": "UNAVAILABLE",
            "error": "Unavailable",
            "message": "internal error",
            "ms": (time.time() - start) * 1000 if 'start' in locals() else 0,
            "retry_after_ms": None,
            "details": None,
        }

@app.post("/v1/vector/upsert")
async def handle_upsert(request: Request) -> Dict[str, Any]:
    """Upsert vectors."""
    try:
        envelope = await request.json()
        start = time.time()
        response = await handler.handle(envelope)
        if "ms" not in response:
            response["ms"] = (time.time() - start) * 1000
        return response
    except Exception as e:
        logger.exception("Error in upsert endpoint")
        return {
            "ok": False,
            "code": "UNAVAILABLE",
            "error": "Unavailable",
            "message": "internal error",
            "ms": (time.time() - start) * 1000 if 'start' in locals() else 0,
            "retry_after_ms": None,
            "details": None,
        }

@app.post("/v1/vector/delete")
async def handle_delete(request: Request) -> Dict[str, Any]:
    """Delete vectors."""
    try:
        envelope = await request.json()
        start = time.time()
        response = await handler.handle(envelope)
        if "ms" not in response:
            response["ms"] = (time.time() - start) * 1000
        return response
    except Exception as e:
        logger.exception("Error in delete endpoint")
        return {
            "ok": False,
            "code": "UNAVAILABLE",
            "error": "Unavailable",
            "message": "internal error",
            "ms": (time.time() - start) * 1000 if 'start' in locals() else 0,
            "retry_after_ms": None,
            "details": None,
        }

@app.post("/v1/vector/batch_query")
async def handle_batch_query(request: Request) -> Dict[str, Any]:
    """Execute batch vector queries."""
    try:
        envelope = await request.json()
        start = time.time()
        response = await handler.handle(envelope)
        if "ms" not in response:
            response["ms"] = (time.time() - start) * 1000
        return response
    except Exception as e:
        logger.exception("Error in batch_query endpoint")
        return {
            "ok": False,
            "code": "UNAVAILABLE",
            "error": "Unavailable",
            "message": "internal error",
            "ms": (time.time() - start) * 1000 if 'start' in locals() else 0,
            "retry_after_ms": None,
            "details": None,
        }

@app.post("/v1/vector/create_namespace")
async def handle_create_namespace(request: Request) -> Dict[str, Any]:
    """Create namespace."""
    try:
        envelope = await request.json()
        start = time.time()
        response = await handler.handle(envelope)
        if "ms" not in response:
            response["ms"] = (time.time() - start) * 1000
        return response
    except Exception as e:
        logger.exception("Error in create_namespace endpoint")
        return {
            "ok": False,
            "code": "UNAVAILABLE",
            "error": "Unavailable",
            "message": "internal error",
            "ms": (time.time() - start) * 1000 if 'start' in locals() else 0,
            "retry_after_ms": None,
            "details": None,
        }

@app.post("/v1/vector/delete_namespace")
async def handle_delete_namespace(request: Request) -> Dict[str, Any]:
    """Delete namespace."""
    try:
        envelope = await request.json()
        start = time.time()
        response = await handler.handle(envelope)
        if "ms" not in response:
            response["ms"] = (time.time() - start) * 1000
        return response
    except Exception as e:
        logger.exception("Error in delete_namespace endpoint")
        return {
            "ok": False,
            "code": "UNAVAILABLE",
            "error": "Unavailable",
            "message": "internal error",
            "ms": (time.time() - start) * 1000 if 'start' in locals() else 0,
            "retry_after_ms": None,
            "details": None,
        }

@app.get("/v1/health")
async def health_check():
    health = await adapter.health()
    status_code = 200 if health.get("status") in ["ok", "degraded"] else 503
    return JSONResponse(status_code=status_code, content=health)

@app.get("/v1/capabilities")
async def get_capabilities():
    caps = await adapter.capabilities()
    from dataclasses import asdict
    return asdict(caps)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "services.vector_service:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
```

### 2.4 Graph Service

`services/graph_service.py`:

```python
import os
import time
import logging
from typing import Dict, Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

from corpus_sdk.graph.graph_base import WireGraphHandler
from adapters.hello_graph import HelloGraphAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("graph-service")

adapter = HelloGraphAdapter(
    api_key=os.environ.get("PROVIDER_API_KEY", ""),
    endpoint=os.environ.get("PROVIDER_ENDPOINT", None),
    mode=os.environ.get("ADAPTER_MODE", "standalone"),
)

handler = WireGraphHandler(adapter)

app = FastAPI(title="Corpus Graph Adapter", version="1.0.0")

@app.middleware("http")
async def add_protocol_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Adapter-Protocol"] = "graph/v1.0"
    return response

@app.post("/v1/graph/query")
async def handle_query(request: Request) -> Dict[str, Any]:
    """Execute graph query."""
    protocol = request.headers.get("x-adapter-protocol")
    if protocol and protocol != "graph/v1.0":
        return {
            "ok": False,
            "code": "NOT_SUPPORTED",
            "error": "NotSupported",
            "message": f"Protocol {protocol} not supported",
            "ms": 0,
            "retry_after_ms": None,
            "details": {"supported_protocols": ["graph/v1.0"]}
        }
    
    try:
        envelope = await request.json()
        start = time.time()
        response = await handler.handle(envelope)
        if "ms" not in response:
            response["ms"] = (time.time() - start) * 1000
        return response
    except Exception as e:
        logger.exception("Error in query endpoint")
        return {
            "ok": False,
            "code": "UNAVAILABLE",
            "error": "Unavailable",
            "message": "internal error",
            "ms": (time.time() - start) * 1000 if 'start' in locals() else 0,
            "retry_after_ms": None,
            "details": None,
        }

@app.post("/v1/graph/stream")
async def handle_stream_query(request: Request) -> StreamingResponse:
    """Stream graph query results."""
    try:
        envelope = await request.json()
        
        async def stream_generator():
            start = time.time()
            async for chunk in handler.handle_stream(envelope):
                if "ms" not in chunk:
                    chunk["ms"] = (time.time() - start) * 1000
                yield json.dumps(chunk) + "\n"
                
        return StreamingResponse(
            stream_generator(),
            media_type="application/x-ndjson",
            headers={"X-Adapter-Protocol": "graph/v1.0"}
        )
    except Exception as e:
        logger.exception("Error in streaming endpoint")
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "code": "UNAVAILABLE",
                "error": "Unavailable",
                "message": "streaming error",
                "ms": 0,
                "retry_after_ms": None,
                "details": None,
            }
        )

@app.post("/v1/graph/upsert_nodes")
async def handle_upsert_nodes(request: Request) -> Dict[str, Any]:
    """Upsert nodes."""
    try:
        envelope = await request.json()
        start = time.time()
        response = await handler.handle(envelope)
        if "ms" not in response:
            response["ms"] = (time.time() - start) * 1000
        return response
    except Exception as e:
        logger.exception("Error in upsert_nodes endpoint")
        return {
            "ok": False,
            "code": "UNAVAILABLE",
            "error": "Unavailable",
            "message": "internal error",
            "ms": (time.time() - start) * 1000 if 'start' in locals() else 0,
            "retry_after_ms": None,
            "details": None,
        }

@app.post("/v1/graph/upsert_edges")
async def handle_upsert_edges(request: Request) -> Dict[str, Any]:
    """Upsert edges."""
    try:
        envelope = await request.json()
        start = time.time()
        response = await handler.handle(envelope)
        if "ms" not in response:
            response["ms"] = (time.time() - start) * 1000
        return response
    except Exception as e:
        logger.exception("Error in upsert_edges endpoint")
        return {
            "ok": False,
            "code": "UNAVAILABLE",
            "error": "Unavailable",
            "message": "internal error",
            "ms": (time.time() - start) * 1000 if 'start' in locals() else 0,
            "retry_after_ms": None,
            "details": None,
        }

@app.post("/v1/graph/delete_nodes")
async def handle_delete_nodes(request: Request) -> Dict[str, Any]:
    """Delete nodes."""
    try:
        envelope = await request.json()
        start = time.time()
        response = await handler.handle(envelope)
        if "ms" not in response:
            response["ms"] = (time.time() - start) * 1000
        return response
    except Exception as e:
        logger.exception("Error in delete_nodes endpoint")
        return {
            "ok": False,
            "code": "UNAVAILABLE",
            "error": "Unavailable",
            "message": "internal error",
            "ms": (time.time() - start) * 1000 if 'start' in locals() else 0,
            "retry_after_ms": None,
            "details": None,
        }

@app.post("/v1/graph/delete_edges")
async def handle_delete_edges(request: Request) -> Dict[str, Any]:
    """Delete edges."""
    try:
        envelope = await request.json()
        start = time.time()
        response = await handler.handle(envelope)
        if "ms" not in response:
            response["ms"] = (time.time() - start) * 1000
        return response
    except Exception as e:
        logger.exception("Error in delete_edges endpoint")
        return {
            "ok": False,
            "code": "UNAVAILABLE",
            "error": "Unavailable",
            "message": "internal error",
            "ms": (time.time() - start) * 1000 if 'start' in locals() else 0,
            "retry_after_ms": None,
            "details": None,
        }

@app.post("/v1/graph/batch")
async def handle_batch(request: Request) -> Dict[str, Any]:
    """Execute batch operations."""
    try:
        envelope = await request.json()
        start = time.time()
        response = await handler.handle(envelope)
        if "ms" not in response:
            response["ms"] = (time.time() - start) * 1000
        return response
    except Exception as e:
        logger.exception("Error in batch endpoint")
        return {
            "ok": False,
            "code": "UNAVAILABLE",
            "error": "Unavailable",
            "message": "internal error",
            "ms": (time.time() - start) * 1000 if 'start' in locals() else 0,
            "retry_after_ms": None,
            "details": None,
        }

@app.post("/v1/graph/transaction")
async def handle_transaction(request: Request) -> Dict[str, Any]:
    """Execute atomic transaction."""
    try:
        envelope = await request.json()
        start = time.time()
        response = await handler.handle(envelope)
        if "ms" not in response:
            response["ms"] = (time.time() - start) * 1000
        return response
    except Exception as e:
        logger.exception("Error in transaction endpoint")
        return {
            "ok": False,
            "code": "UNAVAILABLE",
            "error": "Unavailable",
            "message": "internal error",
            "ms": (time.time() - start) * 1000 if 'start' in locals() else 0,
            "retry_after_ms": None,
            "details": None,
        }

@app.post("/v1/graph/traversal")
async def handle_traversal(request: Request) -> Dict[str, Any]:
    """Execute graph traversal."""
    try:
        envelope = await request.json()
        start = time.time()
        response = await handler.handle(envelope)
        if "ms" not in response:
            response["ms"] = (time.time() - start) * 1000
        return response
    except Exception as e:
        logger.exception("Error in traversal endpoint")
        return {
            "ok": False,
            "code": "UNAVAILABLE",
            "error": "Unavailable",
            "message": "internal error",
            "ms": (time.time() - start) * 1000 if 'start' in locals() else 0,
            "retry_after_ms": None,
            "details": None,
        }

@app.get("/v1/graph/schema")
async def get_schema(request: Request) -> Dict[str, Any]:
    """Get graph schema."""
    try:
        envelope = {"op": "graph.get_schema", "ctx": {}, "args": {}}
        start = time.time()
        response = await handler.handle(envelope)
        if "ms" not in response:
            response["ms"] = (time.time() - start) * 1000
        return response
    except Exception as e:
        logger.exception("Error in schema endpoint")
        return {
            "ok": False,
            "code": "UNAVAILABLE",
            "error": "Unavailable",
            "message": "internal error",
            "ms": (time.time() - start) * 1000 if 'start' in locals() else 0,
            "retry_after_ms": None,
            "details": None,
        }

@app.get("/v1/health")
async def health_check():
    health = await adapter.health()
    status_code = 200 if health.get("status") in ["ok", "degraded"] else 503
    return JSONResponse(status_code=status_code, content=health)

@app.get("/v1/capabilities")
async def get_capabilities():
    caps = await adapter.capabilities()
    from dataclasses import asdict
    return asdict(caps)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "services.graph_service:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
```

---

## 3. Testing Your HTTP Service

The Corpus SDK includes a comprehensive wire conformance test suite in `tests/live/`. Use it to validate your HTTP service:

```bash
# Start your service
uvicorn services.embedding_service:app --port 8000 &

# Run wire conformance tests against your live endpoint
export CORPUS_ENDPOINT=http://localhost:8000
pytest tests/live/test_wire_conformance.py -v

# Test specific protocol
pytest tests/live/test_wire_conformance.py -v -m "embedding"

# Test with verbose output
pytest tests/live/test_wire_conformance.py -v --conformance-verbose

# Skip schema validation for faster iteration
pytest tests/live/test_wire_conformance.py -v --skip-schema
```

The test suite will validate:
- ✅ Canonical envelope format (`{op, ctx, args}`)
- ✅ Required fields and types
- ✅ Protocol-specific args validation
- ✅ JSON serializability
- ✅ Schema conformance (when enabled)

**No need to write custom HTTP tests** - the existing suite covers everything.

---

## 4. Alternative Frameworks

### 4.1 Starlette (Async-First)

`services/starlette_embedding.py`:

```python
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
import uvicorn
import time
import json
import os

from corpus_sdk.embedding.embedding_base import WireEmbeddingHandler
from adapters.hello_embedding import HelloEmbeddingAdapter

adapter = HelloEmbeddingAdapter(api_key=os.environ["PROVIDER_API_KEY"])
handler = WireEmbeddingHandler(adapter)

async def handle_embedding(request: Request):
    protocol = request.headers.get("x-adapter-protocol")
    if protocol and protocol != "embedding/v1.0":
        return JSONResponse({
            "ok": False,
            "code": "NOT_SUPPORTED",
            "error": "NotSupported",
            "message": f"Protocol {protocol} not supported",
            "ms": 0,
            "retry_after_ms": None,
            "details": {"supported_protocols": ["embedding/v1.0"]}
        })
    
    try:
        envelope = await request.json()
        start = time.time()
        response = await handler.handle(envelope)
        if "ms" not in response:
            response["ms"] = (time.time() - start) * 1000
        return JSONResponse(response)
    except Exception as e:
        return JSONResponse({
            "ok": False,
            "code": "UNAVAILABLE",
            "error": "Unavailable",
            "message": "internal error",
            "ms": 0,
            "retry_after_ms": None,
            "details": None,
        }, status_code=500)

async def health_check(request: Request):
    health = await adapter.health()
    status_code = 200 if health.get("status") in ["ok", "degraded"] else 503
    return JSONResponse(health, status_code=status_code)

routes = [
    Route("/v1/embedding", handle_embedding, methods=["POST"]),
    Route("/v1/health", health_check, methods=["GET"]),
]

app = Starlette(routes=routes)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 4.2 Flask (Synchronous)

`services/flask_embedding.py`:

```python
from flask import Flask, request, jsonify
import time
import asyncio
import os

from corpus_sdk.embedding.embedding_base import WireEmbeddingHandler
from adapters.hello_embedding import HelloEmbeddingAdapter

app = Flask(__name__)

# Initialize adapter and handler
adapter = HelloEmbeddingAdapter(api_key=os.environ["PROVIDER_API_KEY"])
handler = WireEmbeddingHandler(adapter)

# Helper to run async code in Flask
def run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

@app.route('/v1/embedding', methods=['POST'])
def handle_embedding():
    protocol = request.headers.get('X-Adapter-Protocol')
    if protocol and protocol != 'embedding/v1.0':
        return jsonify({
            "ok": False,
            "code": "NOT_SUPPORTED",
            "error": "NotSupported",
            "message": f"Protocol {protocol} not supported",
            "ms": 0,
            "retry_after_ms": None,
            "details": {"supported_protocols": ["embedding/v1.0"]}
        })
    
    try:
        envelope = request.get_json()
        start = time.time()
        response = run_async(handler.handle(envelope))
        if "ms" not in response:
            response["ms"] = (time.time() - start) * 1000
        return jsonify(response)
    except Exception as e:
        app.logger.exception("Error in embedding endpoint")
        return jsonify({
            "ok": False,
            "code": "UNAVAILABLE",
            "error": "Unavailable",
            "message": "internal error",
            "ms": (time.time() - start) * 1000 if 'start' in locals() else 0,
            "retry_after_ms": None,
            "details": None,
        })

@app.route('/v1/health', methods=['GET'])
def health_check():
    health = run_async(adapter.health())
    status_code = 200 if health.get("status") in ["ok", "degraded"] else 503
    return jsonify(health), status_code

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

### 4.3 Quart (Async Flask)

`services/quart_embedding.py`:

```python
from quart import Quart, request, jsonify
import time
import os

from corpus_sdk.embedding.embedding_base import WireEmbeddingHandler
from adapters.hello_embedding import HelloEmbeddingAdapter

app = Quart(__name__)
adapter = HelloEmbeddingAdapter(api_key=os.environ["PROVIDER_API_KEY"])
handler = WireEmbeddingHandler(adapter)

@app.route('/v1/embedding', methods=['POST'])
async def handle_embedding():
    protocol = request.headers.get('X-Adapter-Protocol')
    if protocol and protocol != 'embedding/v1.0':
        return jsonify({
            "ok": False,
            "code": "NOT_SUPPORTED",
            "error": "NotSupported",
            "message": f"Protocol {protocol} not supported",
            "ms": 0,
            "retry_after_ms": None,
            "details": {"supported_protocols": ["embedding/v1.0"]}
        })
    
    try:
        envelope = await request.get_json()
        start = time.time()
        response = await handler.handle(envelope)
        if "ms" not in response:
            response["ms"] = (time.time() - start) * 1000
        return jsonify(response)
    except Exception as e:
        app.logger.exception("Error")
        return jsonify({
            "ok": False,
            "code": "UNAVAILABLE",
            "error": "Unavailable",
            "message": "internal error",
            "ms": (time.time() - start) * 1000 if 'start' in locals() else 0,
            "retry_after_ms": None,
            "details": None,
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

---

## 5. Deployment Patterns

### 5.1 Docker + Kubernetes

**Dockerfile:**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY adapters/ ./adapters/
COPY services/ ./services/

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Run the service
CMD ["uvicorn", "services.embedding_service:app", "--host", "0.0.0.0", "--port", "8000"]
```

**requirements.txt:**

```
corpus-sdk>=1.0.0
fastapi>=0.104.0
uvicorn>=0.24.0
httpx>=0.25.0
prometheus-client>=0.19.0
redis>=5.0.0
```

**kubernetes/deployment.yaml:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: embedding-adapter
spec:
  replicas: 3
  selector:
    matchLabels:
      app: embedding-adapter
  template:
    metadata:
      labels:
        app: embedding-adapter
    spec:
      containers:
      - name: adapter
        image: your-registry/embedding-adapter:latest
        ports:
        - containerPort: 8000
        env:
        - name: PROVIDER_API_KEY
          valueFrom:
            secretKeyRef:
              name: provider-secrets
              key: api-key
        - name: ADAPTER_MODE
          value: "standalone"
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /v1/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /v1/health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: embedding-adapter
spec:
  selector:
    app: embedding-adapter
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### 5.2 AWS Lambda (Serverless)

`lambda_function.py`:

```python
import os
import time
import json
from adapters.hello_embedding import HelloEmbeddingAdapter
from corpus_sdk.embedding.embedding_base import WireEmbeddingHandler

# Initialize outside handler for reuse
adapter = HelloEmbeddingAdapter(
    api_key=os.environ["PROVIDER_API_KEY"],
    mode="standalone"
)
handler = WireEmbeddingHandler(adapter)

def lambda_handler(event, context):
    """Handle API Gateway Lambda proxy events."""
    try:
        # Parse API Gateway event
        body = json.loads(event.get("body", "{}"))
        protocol = event.get("headers", {}).get("x-adapter-protocol")
        
        # Protocol validation
        if protocol and protocol != "embedding/v1.0":
            return {
                "statusCode": 200,
                "headers": {"X-Adapter-Protocol": "embedding/v1.0"},
                "body": json.dumps({
                    "ok": False,
                    "code": "NOT_SUPPORTED",
                    "error": "NotSupported",
                    "message": f"Protocol {protocol} not supported",
                    "ms": 0,
                    "retry_after_ms": None,
                    "details": {"supported_protocols": ["embedding/v1.0"]}
                })
            }
        
        # Process request
        start = time.time()
        response = handler.handle(body)
        if "ms" not in response:
            response["ms"] = (time.time() - start) * 1000
            
        return {
            "statusCode": 200,
            "headers": {"X-Adapter-Protocol": "embedding/v1.0"},
            "body": json.dumps(response)
        }
        
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"X-Adapter-Protocol": "embedding/v1.0"},
            "body": json.dumps({
                "ok": False,
                "code": "UNAVAILABLE",
                "error": "Unavailable",
                "message": "internal error",
                "ms": 0,
                "retry_after_ms": None,
                "details": None,
            })
        }
```

**serverless.yml:**

```yaml
service: corpus-embedding-adapter

provider:
  name: aws
  runtime: python3.11
  environment:
    PROVIDER_API_KEY: ${env:PROVIDER_API_KEY}
    ADAPTER_MODE: "standalone"

functions:
  embedding:
    handler: lambda_function.lambda_handler
    events:
      - http:
          path: /v1/embedding
          method: post
          cors: true
      - http:
          path: /v1/health
          method: get
          cors: true
```

### 5.3 Google Cloud Run

`Dockerfile` (same as above) + `cloudbuild.yaml`:

```yaml
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '-t', 'gcr.io/$PROJECT_ID/embedding-adapter', '.']
- name: 'gcr.io/cloud-builders/docker'
  args: ['push', 'gcr.io/$PROJECT_ID/embedding-adapter']
- name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
  entrypoint: gcloud
  args:
  - 'run'
  - 'deploy'
  - 'embedding-adapter'
  - '--image'
  - 'gcr.io/$PROJECT_ID/embedding-adapter'
  - '--platform'
  - 'managed'
  - '--region'
  - 'us-central1'
  - '--allow-unauthenticated'
  - '--memory'
  - '512Mi'
```

### 5.4 Cloudflare Workers

`worker.js` (for lightweight HTTP routing to your service):

```javascript
export default {
  async fetch(request) {
    // Validate protocol
    const protocol = request.headers.get('X-Adapter-Protocol');
    if (protocol && protocol !== 'embedding/v1.0') {
      return new Response(JSON.stringify({
        ok: false,
        code: 'NOT_SUPPORTED',
        error: 'NotSupported',
        message: `Protocol ${protocol} not supported`,
        ms: 0,
        retry_after_ms: null,
        details: { supported_protocols: ['embedding/v1.0'] }
      }), {
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // Forward to your backend service
    const url = new URL(request.url);
    const backendUrl = `https://your-backend.com${url.pathname}`;
    
    return fetch(backendUrl, {
      method: request.method,
      headers: request.headers,
      body: request.body
    });
  }
}
```

---

## 6. Production Considerations

### 6.1 Environment Configuration

`.env` file (never commit to git):

```bash
# Provider credentials
PROVIDER_API_KEY=sk-...
PROVIDER_ENDPOINT=https://api.openai.com/v1

# Adapter mode (standalone or thin)
ADAPTER_MODE=standalone

# Redis for idempotency cache (embedding only)
REDIS_URL=redis://:password@redis.example.com:6379/0
REDIS_TTL_SECONDS=86400  # 24 hours

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Server
PORT=8000
HOST=0.0.0.0
WORKERS=4
```

Load with python-dotenv:

```python
from dotenv import load_dotenv
load_dotenv()
```

### 6.2 Logging & Observability

**Structured Logging (JSON):**

```python
import logging
import json
from pythonjsonlogger import jsonlogger

logger = logging.getLogger()
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter(
    fmt='%(asctime)s %(levelname)s %(name)s %(message)s'
)
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

# In your endpoints
logger.info("Request processed", extra={
    "op": op,
    "tenant_hash": tenant_hash,
    "duration_ms": duration_ms,
    "status_code": 200
})
```

**Request ID Propagation:**

```python
import uuid
from fastapi import Request

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

# In handler
logger.info("Processing", extra={"request_id": request.state.request_id})
```

### 6.3 Metrics & Monitoring

**Prometheus Metrics:**

```python
from prometheus_client import Counter, Histogram, generate_latest
import time

# Define metrics
requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    
    requests_total.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    request_duration.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    return response

@app.get("/metrics")
async def metrics():
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )
```

### 6.4 Rate Limiting (Beyond BaseAdapter)

**Redis-based rate limiter:**

```python
import redis
import time
from fastapi import HTTPException

redis_client = redis.from_url(os.environ["REDIS_URL"])

async def check_rate_limit(tenant: str, limit: int = 100, window: int = 60):
    """Check rate limit per tenant."""
    key = f"rate_limit:{tenant}"
    current = redis_client.get(key)
    
    if current and int(current) >= limit:
        raise HTTPException(
            status_code=429,
            detail={
                "ok": False,
                "code": "RATE_LIMITED",
                "error": "ResourceExhausted",
                "message": "Rate limit exceeded",
                "retry_after_ms": 5000,
                "details": {"limit": limit, "window": window}
            }
        )
    
    pipe = redis_client.pipeline()
    pipe.incr(key)
    pipe.expire(key, window)
    pipe.execute()
```

### 6.5 Security (mTLS, API Keys)

**API Key Authentication:**

```python
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    api_key = credentials.credentials
    valid_keys = os.environ.get("API_KEYS", "").split(",")
    
    if api_key not in valid_keys:
        raise HTTPException(
            status_code=401,
            detail={
                "ok": False,
                "code": "UNAUTHENTICATED",
                "error": "AuthError",
                "message": "Invalid API key",
                "ms": 0,
                "retry_after_ms": None,
                "details": None
            }
        )
    return api_key

@app.post("/v1/embedding")
async def handle_embedding(
    request: Request,
    api_key: str = Security(verify_api_key)
):
    # ... endpoint logic
```

**mTLS with Kubernetes:**

```yaml
# Ingress with mTLS
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: embedding-adapter
  annotations:
    nginx.ingress.kubernetes.io/auth-tls-verify-client: "on"
    nginx.ingress.kubernetes.io/auth-tls-secret: "default/ca-secret"
spec:
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /v1/embedding
        pathType: Prefix
        backend:
          service:
            name: embedding-adapter
            port:
              number: 80
```

### 6.6 Load Balancing & Scaling

**Horizontal Pod Autoscaling (Kubernetes):**

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: embedding-adapter
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: embedding-adapter
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

---

## 7. Production Checklist

### Pre-Deployment

- [ ] Adapter has **Gold certification** from Quickstart
- [ ] All `_do_*()` methods use `ctx.remaining_ms()` for deadlines
- [ ] Idempotency cache uses Redis (not in-memory) for embedding adapters
- [ ] Error mapping covers all provider error codes
- [ ] No raw tenant IDs in logs (hashed only)
- [ ] Health endpoint returns graded status (`ok`/`degraded`/`down`)

### HTTP Service

- [ ] Protocol header validation (`X-Adapter-Protocol`)
- [ ] Request ID propagation for tracing
- [ ] Structured logging (JSON format)
- [ ] Prometheus metrics endpoint (`/metrics`)
- [ ] Graceful shutdown handling
- [ ] CORS configuration if needed
- [ ] Rate limiting per tenant (if required)

### Security

- [ ] API key authentication or mTLS
- [ ] Secrets managed via environment/vault (not in code)
- [ ] HTTPS/TLS termination
- [ ] Network policies restrict access
- [ ] Regular dependency updates

### Infrastructure

- [ ] Containerized with Docker
- [ ] Resource limits set (CPU/memory)
- [ ] Liveness and readiness probes configured
- [ ] Horizontal autoscaling configured
- [ ] Monitoring and alerting set up
- [ ] Disaster recovery plan

### Observability

- [ ] Request duration metrics
- [ ] Error rate monitoring
- [ ] Provider latency tracking
- [ ] Cache hit/miss ratios
- [ ] Tenant usage dashboards
- [ ] Log aggregation (ELK/CloudWatch)

---

## Appendix A: Full Service Examples

See the complete service implementations in:
- `services/embedding_service.py` (FastAPI)
- `services/llm_service.py` (FastAPI)
- `services/vector_service.py` (FastAPI)
- `services/graph_service.py` (FastAPI)

These are production-ready templates with:
- Protocol header validation
- Error handling
- Metrics
- Structured logging
- Health checks
- Graceful shutdown

---

## Appendix B: Common Deployment Pitfalls

### 1. Missing Protocol Header Validation

```python
# ❌ WRONG: No protocol validation
@app.post("/v1/embedding")
async def handle(request):
    return await handler.handle(await request.json())

# ✅ CORRECT: Validate protocol
protocol = request.headers.get("x-adapter-protocol")
if protocol and protocol != "embedding/v1.0":
    return error_response("NOT_SUPPORTED")
```

### 2. In-memory Cache in Production

```python
# ❌ WRONG: In-memory cache (lost on restart)
self._idempotency_cache = {}

# ✅ CORRECT: Redis cache
self.redis = redis.from_url(os.environ["REDIS_URL"])
async def _check_idempotency(key):
    return await self.redis.get(key)
```

### 3. Missing Deadline Propagation

```python
# ❌ WRONG: No timeout passed to provider
response = await client.post(self.endpoint, json=payload)

# ✅ CORRECT: Use remaining deadline
if ctx and ctx.deadline_ms:
    remaining = ctx.remaining_ms()
    if remaining > 0:
        timeout = remaining / 1000.0
        response = await client.post(..., timeout=timeout)
```

### 4. Raw Tenant IDs in Logs

```python
# ❌ WRONG: Logs PII
logger.info(f"Request from tenant {ctx.tenant}")

# ✅ CORRECT: Logs hash only
tenant_hash = hashlib.sha256(ctx.tenant.encode()).hexdigest()[:16]
logger.info(f"Request from tenant {tenant_hash}")
```

### 5. Cache Invalidation Order (Vector/Graph)

```python
# ❌ WRONG: Invalidate before write
await self._invalidate_namespace_cache(ns)
await client.upsert(vectors)  # If this fails, cache is stale

# ✅ CORRECT: Invalidate after successful write
result = await client.upsert(vectors)
if result.success:
    await self._invalidate_namespace_cache(ns)
```

---

## Appendix C: Debugging Production Issues

### Common HTTP Status Codes

| Status | Meaning | Common Causes |
|--------|---------|---------------|
| 200 | Success | - |
| 400 | Bad Request | Invalid envelope, missing fields |
| 401 | Unauthorized | Invalid/missing API key |
| 429 | Rate Limited | Too many requests |
| 500 | Internal Error | Provider outage, bug |
| 503 | Unavailable | Provider down, circuit open |

### Debugging Commands

```bash
# Check health
curl -v http://localhost:8000/v1/health

# Test with valid request
curl -X POST http://localhost:8000/v1/embedding \
  -H "X-Adapter-Protocol: embedding/v1.0" \
  -H "Content-Type: application/json" \
  -d '{"op":"embedding.embed","ctx":{"request_id":"test"},"args":{"text":"hello","model":"text-embedding-001"}}'

# Test streaming
curl -N -X POST http://localhost:8000/v1/embedding/stream \
  -H "X-Adapter-Protocol: embedding/v1.0" \
  -H "Content-Type: application/json" \
  -d '{"op":"embedding.stream_embed","ctx":{},"args":{"text":"hello","model":"text-embedding-001"}}'

# Check metrics
curl http://localhost:8000/metrics

# Run wire conformance tests
export CORPUS_ENDPOINT=http://localhost:8000
pytest tests/live/test_wire_conformance.py -v -m "embedding"

# Tail logs with request ID
kubectl logs -l app=embedding-adapter --tail=100 | grep "req_abc123"
```

### Profiling

```python
import cProfile
import pstats
from io import StringIO

@app.middleware("http")
async def profile_request(request: Request, call_next):
    profiler = cProfile.Profile()
    profiler.enable()
    
    response = await call_next(request)
    
    profiler.disable()
    s = StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    stats.print_stats(20)
    
    logger.debug(f"Profile for {request.url.path}:\n{s.getvalue()}")
    return response
```

### Memory Profiling

```python
from memory_profiler import profile

@profile
async def handle_large_batch(self, spec):
    # Profile memory usage of batch operations
    embeddings = []
    for text in spec.texts:
        vec = await self._generate_embedding(text)
        embeddings.append(vec)
    return embeddings
```

---

**Maintainers:** Corpus SDK Team  
**Last Updated:** 2026-02-13  
**Scope:** Production deployment guide for Corpus Protocol adapters.

**Next:** Return to [QUICKSTART.md](QUICKSTART.md) if you haven't certified your adapter yet.