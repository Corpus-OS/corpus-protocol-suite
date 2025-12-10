# Corpus SDK Framework Adapter Quick Start

Goal: Make Corpus Protocol–speaking services usable from 6 popular frameworks (Autogen, CrewAI, LangChain, LlamaIndex, Semantic Kernel, MCP) across all 4 domains (LLM, Embedding, Vector, Graph) – with thin, honest translation layers, not fantasy SDKs.

⸻

Table of Contents
	•	0. Mental Model (What You’re Actually Building)
	•	1. Prerequisites & Shared Wire Client
	•	2. Autogen Integration
	•	2.1 LLM (Chat Completion)
	•	2.2 Embedding-backed “Memory”
	•	2.3 Vector Search Tool
	•	2.4 Graph Query Tool
	•	2.5 Wiring Agents
	•	3. CrewAI Integration
	•	3.1 CrewAI LLM Backend
	•	3.2 CrewAI Embedding & Vector-backed Tools
	•	3.3 CrewAI Graph Tool
	•	4. LangChain Integration
	•	4.1 Chat Model (LLM)
	•	4.2 Embeddings
	•	4.3 VectorStore
	•	4.4 Graph Tooling
	•	5. LlamaIndex Integration
	•	5.1 LLM
	•	5.2 Embedding Model
	•	5.3 Vector Index / Storage
	•	5.4 Graph-backed Query Engine
	•	6. Semantic Kernel Integration
	•	6.1 Text Embedding Service
	•	6.2 Chat Completion Service
	•	6.3 Vector / Graph Connectors
	•	7. MCP Integration
	•	7.1 MCP Server Backed by Corpus Protocols
	•	7.2 Using Corpus from MCP-aware Frameworks
	•	8. Contexts, Deadlines, Tenants & Error Mapping
	•	9. Framework Conformance & Smoke Tests
	•	10. Integration Launch Checklist (TL;dr)

⸻

0. Mental Model (What You’re Actually Building)

You already have server-side adapters:

[BaseLLMAdapter / BaseEmbeddingAdapter / ...]
          ↓
[WireLLMHandler / WireEmbeddingHandler / ...]
          ↓
      HTTP endpoints (speak Corpus Protocol)

This document is about the client-side:

[Framework-native API]  (Autogen, CrewAI, LangChain, etc.)
        ↓
[Framework adapter]     ← very thin, framework-specific glue
        ↓
[CorpusClient]          ← sends/receives Corpus envelopes over HTTP
        ↓
[Your HTTP services]    ← built with Base*Adapter + Wire*Handler

Design rules for framework adapters:
	1.	Thin – 50–200 LOC per adapter. No business logic.
	2.	Honest – no fake SDK modules; everything in this doc is code YOU define.
	3.	Protocol-correct – envelopes must match SCHEMA.md.
	4.	Framework-native – return the types each framework expects (or very close).

⸻

1. Prerequisites & Shared Wire Client

You need a tiny, shared CorpusClient that understands your envelopes and endpoints.

Assumption:
You’ve deployed four HTTP endpoints (names are up to you), each expecting/returning Corpus envelopes:
	•	llm_url – handles op: "llm.complete" etc.
	•	embedding_url – handles op: "embedding.embed" / embedding.embed_batch
	•	vector_url – handles op: "vector.query", vector.upsert, …
	•	graph_url – handles op: "graph.query" / graph.stream.query"

Create something like frameworks/corpus_client.py:

# frameworks/corpus_client.py
from __future__ import annotations
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx  # you choose sync vs async; here: async


def _now_ms() -> int:
    return int(time.time() * 1000)


class CorpusError(RuntimeError):
    """Raised when the Corpus service returns ok=false."""


class CorpusClient:
    """
    Minimal client that speaks Corpus envelopes (SCHEMA.md).
    This is the ONLY thing that knows about wire format.
    """

    def __init__(
        self,
        *,
        llm_url: Optional[str] = None,
        embedding_url: Optional[str] = None,
        vector_url: Optional[str] = None,
        graph_url: Optional[str] = None,
        tenant: Optional[str] = None,
        default_deadline_ms: int = 20_000,
        client: Optional[httpx.AsyncClient] = None,
    ):
        self.llm_url = llm_url
        self.embedding_url = embedding_url
        self.vector_url = vector_url
        self.graph_url = graph_url
        self.tenant = tenant
        self.default_deadline_ms = default_deadline_ms
        self._client = client or httpx.AsyncClient(timeout=None)

    # --- core envelope helpers ------------------------------------------------

    def _make_ctx(self, deadline_ms: Optional[int] = None) -> Dict[str, Any]:
        return {
            "request_id": str(uuid.uuid4()),
            "deadline_ms": deadline_ms or (_now_ms() + self.default_deadline_ms),
            "tenant": self.tenant,
        }

    async def _post(
        self,
        url: str,
        op: str,
        args: Dict[str, Any],
        *,
        deadline_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        envelope = {
            "op": op,
            "ctx": self._make_ctx(deadline_ms),
            "args": args,
        }
        resp = await self._client.post(url, json=envelope)
        resp.raise_for_status()
        data = resp.json()

        # Common envelope shape per SCHEMA.md
        if not isinstance(data, dict) or "ok" not in data:
            raise CorpusError(f"Malformed response from {url}: {data!r}")

        if not data["ok"]:
            code = data.get("code", "UNKNOWN")
            msg = data.get("message", "Unknown error")
            raise CorpusError(f"Corpus error {code}: {msg}")

        return data["result"]

    # --- LLM ------------------------------------------------------------------

    async def llm_complete(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        **extra_args: Any,
    ) -> Dict[str, Any]:
        if not self.llm_url:
            raise RuntimeError("llm_url not configured")
        args: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if temperature is not None:
            args["temperature"] = temperature
        if max_tokens is not None:
            args["max_tokens"] = max_tokens
        if response_format is not None:
            args["response_format"] = response_format
        args.update(extra_args)
        return await self._post(self.llm_url, "llm.complete", args)

    # --- Embedding ------------------------------------------------------------

    async def embed(
        self,
        *,
        model: str,
        text: str,
        **extra_args: Any,
    ) -> Dict[str, Any]:
        if not self.embedding_url:
            raise RuntimeError("embedding_url not configured")
        args: Dict[str, Any] = {"model": model, "text": text}
        args.update(extra_args)
        return await self._post(self.embedding_url, "embedding.embed", args)

    async def embed_batch(
        self,
        *,
        model: str,
        texts: List[str],
        **extra_args: Any,
    ) -> Dict[str, Any]:
        if not self.embedding_url:
            raise RuntimeError("embedding_url not configured")
        args: Dict[str, Any] = {"model": model, "texts": texts}
        args.update(extra_args)
        return await self._post(self.embedding_url, "embedding.embed_batch", args)

    # --- Vector ---------------------------------------------------------------

    async def vector_query(
        self,
        *,
        namespace: str,
        vector: Optional[List[float]] = None,
        text: Optional[str] = None,
        top_k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **extra_args: Any,
    ) -> Dict[str, Any]:
        if not self.vector_url:
            raise RuntimeError("vector_url not configured")
        if vector is None and text is None:
            raise ValueError("vector_query requires either vector or text")
        args: Dict[str, Any] = {
            "namespace": namespace,
            "top_k": top_k,
        }
        if vector is not None:
            args["vector"] = vector
        if text is not None:
            args["text"] = text
        if filter is not None:
            args["filter"] = filter
        args.update(extra_args)
        return await self._post(self.vector_url, "vector.query", args)

    # --- Graph ----------------------------------------------------------------

    async def graph_query(
        self,
        *,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        **extra_args: Any,
    ) -> Dict[str, Any]:
        if not self.graph_url:
            raise RuntimeError("graph_url not configured")
        args: Dict[str, Any] = {"query": query}
        if params is not None:
            args["params"] = params
        args.update(extra_args)
        return await self._post(self.graph_url, "graph.query", args)

This client is the only place that knows about op, ctx, and exact args.
Every framework adapter below only deals in messages, texts, vectors, etc. and just calls CorpusClient.

⸻

2. Autogen Integration

The key idea:
	•	Wrap CorpusClient.llm_complete in an object Autogen can call as an LLM.
	•	Use CorpusClient.embed/embed_batch for “memory”.
	•	Use CorpusClient.vector_query and graph_query behind tools.

2.1 LLM (Chat Completion)

File: frameworks/autogen_llm.py

# frameworks/autogen_llm.py
from __future__ import annotations
from typing import Any, Dict, List, Optional

import asyncio

from .corpus_client import CorpusClient


class CorpusAutogenLLMClient:
    """
    Autogen-compatible LLM client.

    This is NOT part of corpus_sdk; it's your glue class.
    It exposes a method Autogen can call that looks roughly
    like an OpenAI chat completion client.
    """

    def __init__(
        self,
        corpus: CorpusClient,
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ):
        self._corpus = corpus
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Autogen expects something OpenAI-ish:

        {
          "choices": [{
            "message": {"role": "assistant", "content": "..."},
            "finish_reason": "stop"
          }],
          "usage": {...},
          "model": "..."
        }
        """
        result = await self._corpus.llm_complete(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )

        # result is the Corpus "completion" payload from llm.types.completion.json
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": result["text"],
                    },
                    "finish_reason": result.get("finish_reason"),
                }
            ],
            "usage": result.get("usage", {}),
            "model": result.get("model", self.model),
        }

    # Small helper if you need sync bridging in some places
    def create_chat_completion_sync(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return asyncio.run(self.create_chat_completion(messages, **kwargs))

Usage in Autogen (pseudo):

from autogen import ConversableAgent

from frameworks.corpus_client import CorpusClient
from frameworks.autogen_llm import CorpusAutogenLLMClient

corpus = CorpusClient(
    llm_url="http://localhost:8001/llm",
    embedding_url="http://localhost:8002/embedding",
    vector_url="http://localhost:8003/vector",
    graph_url="http://localhost:8004/graph",
    tenant="demo-tenant",
)

llm_client = CorpusAutogenLLMClient(
    corpus=corpus,
    model="gpt-4.1-mini",
    temperature=0.5,
)

agent = ConversableAgent(
    "assistant",
    llm_config={
        "config_list": [
            {
                "model": "corpus-llm",  # arbitrary name
                "client": llm_client,   # must expose create_chat_completion
            }
        ]
    },
)

Adjust the exact config fields to match the Autogen version you use, but the shape (create_chat_completion → Corpus envelope) stays the same.

⸻

2.2 Embedding-backed “Memory”

Use CorpusClient.embed and vector_query to back Autogen’s retrieval agents.

File: frameworks/autogen_memory.py

# frameworks/autogen_memory.py
from __future__ import annotations
from typing import Any, Dict, List

import asyncio

from .corpus_client import CorpusClient


class CorpusMemoryStore:
    """
    Tiny 'memory store' backed by Corpus Embedding + Vector.

    You can wrap this into Autogen's retrieve agents or plug
    it into your own Autogen tooling.
    """

    def __init__(
        self,
        corpus: CorpusClient,
        *,
        namespace: str = "autogen-memory",
        embedding_model: str = "embed-1",
    ):
        self._corpus = corpus
        self.namespace = namespace
        self.embedding_model = embedding_model

    async def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] | None = None) -> None:
        metadatas = metadatas or [{} for _ in texts]

        # 1) embed batch
        batch = await self._corpus.embed_batch(
            model=self.embedding_model,
            texts=texts,
        )
        vectors = batch["embeddings"]  # matches embedding.partial_success.result.json-ish

        # 2) upsert into your vector service (you must define the op)
        # This assumes your vector endpoint has an op "vector.upsert"
        # You can extend CorpusClient._post or create another helper here.
        payload = {
            "namespace": self.namespace,
            "vectors": [
                {
                    "id": meta.get("id") or f"mem-{i}",
                    "vector": vec["vector"],
                    "metadata": meta,
                }
                for i, (vec, meta) in enumerate(zip(vectors, metadatas))
            ],
        }
        # Using _post directly for brevity:
        await self._corpus._post(
            self._corpus.vector_url,
            "vector.upsert",
            payload,
        )

    async def similarity_search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        # 1) embed query
        e = await self._corpus.embed(
            model=self.embedding_model,
            text=query,
        )
        vec = e["embedding"]["vector"]

        # 2) vector query
        result = await self._corpus.vector_query(
            namespace=self.namespace,
            vector=vec,
            top_k=k,
        )
        # matches vector.types.query_result.json
        return result["matches"]

    # Sync helpers (Autogen likes sync in some spots)
    def similarity_search_sync(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        return asyncio.run(self.similarity_search(query, k=k))

You can now plug CorpusMemoryStore into whatever retrieval hooks Autogen exposes.

⸻

2.3 Vector Search Tool

Turn similarity_search_sync into a callable tool.

# frameworks/autogen_vector_tool.py
from __future__ import annotations
from typing import List

from .autogen_memory import CorpusMemoryStore


class CorpusVectorSearchTool:
    """
    Simple Python callable you register as a tool.

    Autogen will call this with natural-language queries.
    """

    def __init__(self, store: CorpusMemoryStore):
        self._store = store

    def __call__(self, query: str, k: int = 4) -> str:
        matches = self._store.similarity_search_sync(query, k=k)
        # assume each match has metadata["text"] or similar
        chunks: List[str] = []
        for m in matches:
            meta = m.get("vector", {}).get("metadata", m.get("metadata", {}))
            text = meta.get("text") or meta.get("content") or "<no text>"
            score = m.get("score")
            chunks.append(f"[score={score:.3f}] {text}")
        return "\n\n".join(chunks)

Then register this callable with the Autogen tooling API you’re using.

⸻

2.4 Graph Query Tool

Same pattern, but for graph:

# frameworks/autogen_graph_tool.py
from __future__ import annotations

import asyncio
from typing import Any, Dict

from .corpus_client import CorpusClient


class CorpusGraphQueryTool:
    def __init__(self, corpus: CorpusClient):
        self._corpus = corpus

    async def aquery(self, query: str) -> Dict[str, Any]:
        return await self._corpus.graph_query(query=query)

    def __call__(self, query: str) -> str:
        """
        Sync wrapper so Autogen can call it as a normal function.
        """
        result = asyncio.run(self.aquery(query))
        # result.data.records is per graph.types.row.json / graph.stream.frame.data.json
        rows = result.get("rows") or result.get("records") or []
        if not rows:
            return "No results."
        return "\n".join(str(r) for r in rows[:20])


⸻

2.5 Wiring Agents

Tie all the above together:

# frameworks/autogen_system.py
from __future__ import annotations

from .corpus_client import CorpusClient
from .autogen_llm import CorpusAutogenLLMClient
from .autogen_memory import CorpusMemoryStore
from .autogen_vector_tool import CorpusVectorSearchTool
from .autogen_graph_tool import CorpusGraphQueryTool


def build_autogen_stack():
    corpus = CorpusClient(
        llm_url="http://localhost:8001/llm",
        embedding_url="http://localhost:8002/embedding",
        vector_url="http://localhost:8003/vector",
        graph_url="http://localhost:8004/graph",
        tenant="demo-tenant",
    )

    llm_client = CorpusAutogenLLMClient(
        corpus=corpus,
        model="gpt-4.1-mini",
    )

    memory_store = CorpusMemoryStore(
        corpus=corpus,
        namespace="autogen-memory",
        embedding_model="embed-1",
    )

    vector_tool = CorpusVectorSearchTool(memory_store)
    graph_tool = CorpusGraphQueryTool(corpus)

    # From here, you plug llm_client, vector_tool, graph_tool
    # into Autogen's agent definitions (depends on the Autogen
    # version and style you use).
    return {
        "corpus": corpus,
        "llm_client": llm_client,
        "memory_store": memory_store,
        "vector_tool": vector_tool,
        "graph_tool": graph_tool,
    }

Everything here is plain Python you own; no made-up corpus modules.

⸻

3. CrewAI Integration

CrewAI typically wants an LLM object plus BaseTool implementations.

Again, we wrap CorpusClient.

3.1 CrewAI LLM Backend

# frameworks/crewai_llm.py
from __future__ import annotations
import asyncio
from typing import Any

from .corpus_client import CorpusClient


class CorpusCrewAILLM:
    """
    Minimal CrewAI-compatible LLM wrapper.

    CrewAI currently can use either LiteLLM or an OpenAI-compatible client.
    Here we show a simple 'call(prompt) -> str' style wrapper that you
    can plug into CrewAI's LLM config.
    """

    def __init__(
        self,
        corpus: CorpusClient,
        *,
        model: str,
        temperature: float = 0.7,
    ):
        self._corpus = corpus
        self.model = model
        self.temperature = temperature

    async def acomplete(self, prompt: str, **kwargs: Any) -> str:
        messages = [{"role": "user", "content": prompt}]
        result = await self._corpus.llm_complete(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
        )
        return result["text"]

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        # CrewAI's LLM hooks are typically sync
        return asyncio.run(self.acomplete(prompt, **kwargs))

You then pass CorpusCrewAILLM to CrewAI where it expects an LLM backend (exact wiring depends on your CrewAI version/config).

⸻

3.2 CrewAI Embedding & Vector-backed Tools

# frameworks/crewai_vector_tool.py
from __future__ import annotations
import asyncio
from typing import Any, Dict, List

from .corpus_client import CorpusClient


class CorpusCrewAIVectorSearchTool:
    """
    CrewAI Tool-like object: __call__(query: str) -> str.

    If you're subclassing CrewAI's BaseTool, you can adapt this shape easily.
    """

    def __init__(
        self,
        corpus: CorpusClient,
        *,
        namespace: str,
        embedding_model: str,
    ):
        self._corpus = corpus
        self.namespace = namespace
        self.embedding_model = embedding_model

    async def _asearch(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        # 1) embed
        e = await self._corpus.embed(
            model=self.embedding_model,
            text=query,
        )
        vec = e["embedding"]["vector"]

        # 2) vector.query
        result = await self._corpus.vector_query(
            namespace=self.namespace,
            vector=vec,
            top_k=k,
        )
        return result["matches"]

    def __call__(self, query: str, k: int = 4) -> str:
        matches = asyncio.run(self._asearch(query, k=k))
        parts: List[str] = []
        for m in matches:
            meta = m.get("vector", {}).get("metadata", m.get("metadata", {}))
            text = meta.get("text") or meta.get("content") or "<no text>"
            score = m.get("score")
            parts.append(f"[score={score:.3f}] {text}")
        return "\n\n".join(parts)

You can wrap this class into CrewAI’s BaseTool if you want the nicer metadata/description.

⸻

3.3 CrewAI Graph Tool

# frameworks/crewai_graph_tool.py
from __future__ import annotations
import asyncio
from typing import Any

from .corpus_client import CorpusClient


class CorpusCrewAIGraphTool:
    def __init__(self, corpus: CorpusClient):
        self._corpus = corpus

    async def _aquery(self, query: str) -> Any:
        return await self._corpus.graph_query(query=query)

    def __call__(self, query: str) -> str:
        result = asyncio.run(self._aquery(query))
        rows = result.get("rows") or result.get("records") or []
        if not rows:
            return "No results from graph."
        return "\n".join(str(r) for r in rows[:20])

Again: CrewAI expects tools with a certain interface; you wrap this accordingly (e.g., BaseTool.run() calling self(query)).

⸻

4. LangChain Integration

LangChain is stricter about base class methods, but still easy to adapt.

4.1 Chat Model (LLM)

# frameworks/langchain_llm.py
from __future__ import annotations
import asyncio
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from .corpus_client import CorpusClient


class CorpusLangChainChat(BaseChatModel):
    """
    LangChain ChatModel backed by Corpus LLM endpoint.
    """

    model_name: str
    temperature: float

    def __init__(
        self,
        corpus: CorpusClient,
        *,
        model_name: str,
        temperature: float = 0.7,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._corpus = corpus
        self.model_name = model_name
        self.temperature = temperature

    @property
    def _llm_type(self) -> str:
        return "corpus-llm"

    def _convert_to_raw(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        raw: List[Dict[str, str]] = []
        for m in messages:
            if isinstance(m, HumanMessage):
                raw.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                raw.append({"role": "assistant", "content": m.content})
            elif isinstance(m, SystemMessage):
                raw.append({"role": "system", "content": m.content})
            else:
                # Fallback to role attr if present
                role = getattr(m, "role", "user")
                raw.append({"role": role, "content": m.content})
        return raw

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        # LangChain calls this sync; we bridge to async
        return asyncio.run(self._agenerate(messages, stop=stop, **kwargs))

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        raw = self._convert_to_raw(messages)
        result = await self._corpus.llm_complete(
            model=self.model_name,
            messages=raw,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens"),
        )

        msg = AIMessage(
            content=result["text"],
            additional_kwargs={"finish_reason": result.get("finish_reason")},
        )

        from langchain_core.outputs import ChatResult, ChatGeneration

        return ChatResult(
            generations=[ChatGeneration(message=msg)],
            llm_output={
                "model": result.get("model", self.model_name),
                "usage": result.get("usage", {}),
            },
        )


⸻

4.2 Embeddings

# frameworks/langchain_embeddings.py
from __future__ import annotations
import asyncio
from typing import List

from langchain_core.embeddings import Embeddings

from .corpus_client import CorpusClient


class CorpusLangChainEmbeddings(Embeddings):
    """
    LangChain Embeddings implementation using Corpus embedding endpoint.
    """

    def __init__(
        self,
        corpus: CorpusClient,
        *,
        model: str,
    ):
        self._corpus = corpus
        self.model = model

    async def _aembed_documents(self, texts: List[str]) -> List[List[float]]:
        result = await self._corpus.embed_batch(model=self.model, texts=texts)
        return [e["vector"] for e in result["embeddings"]]

    async def _aembed_query(self, text: str) -> List[float]:
        result = await self._corpus.embed(model=self.model, text=text)
        return result["embedding"]["vector"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return asyncio.run(self._aembed_documents(texts))

    def embed_query(self, text: str) -> List[float]:
        return asyncio.run(self._aembed_query(text))


⸻

4.3 VectorStore

# frameworks/langchain_vectorstore.py
from __future__ import annotations
import asyncio
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from .corpus_client import CorpusClient
from .langchain_embeddings import CorpusLangChainEmbeddings


class CorpusLangChainVectorStore(VectorStore):
    """
    VectorStore backed by Corpus embedding + vector endpoints.
    """

    def __init__(
        self,
        corpus: CorpusClient,
        *,
        namespace: str,
        embedding_model: str,
    ):
        self._corpus = corpus
        self.namespace = namespace
        self._emb = CorpusLangChainEmbeddings(corpus, model=embedding_model)

    async def _aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        texts = list(texts)
        metadatas = metadatas or [{} for _ in texts]
        ids = ids or [f"doc-{i}" for i in range(len(texts))]

        embeddings = await self._emb._aembed_documents(texts)

        vectors = [
            {
                "id": id_,
                "vector": vec,
                "metadata": meta,
            }
            for id_, vec, meta in zip(ids, embeddings, metadatas)
        ]

        await self._corpus._post(
            self._corpus.vector_url,
            "vector.upsert",
            {
                "namespace": self.namespace,
                "vectors": vectors,
            },
        )
        return ids

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        return asyncio.run(self._aadd_texts(texts, metadatas, ids))

    async def _asimilarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        qvec = await self._emb._aembed_query(query)
        res = await self._corpus.vector_query(
            namespace=self.namespace,
            vector=qvec,
            top_k=k,
        )
        docs: List[Document] = []
        for m in res["matches"]:
            vec = m.get("vector") or {}
            meta = vec.get("metadata", {})
            content = meta.get("text") or meta.get("content") or ""
            docs.append(
                Document(page_content=content, metadata=meta),
            )
        return docs

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        return asyncio.run(self._asimilarity_search(query, k=k, **kwargs))


⸻

4.4 Graph Tooling

# frameworks/langchain_graph_tool.py
from __future__ import annotations
import asyncio
from typing import Any

from langchain_core.tools import BaseTool

from .corpus_client import CorpusClient


class CorpusLangChainGraphTool(BaseTool):
    name = "corpus_graph_query"
    description = "Query the Corpus-backed knowledge graph using a query string."

    def __init__(self, corpus: CorpusClient):
        super().__init__()
        self._corpus = corpus

    async def _aquery(self, query: str) -> Any:
        return await self._corpus.graph_query(query=query)

    def _run(self, query: str) -> str:
        result = asyncio.run(self._aquery(query))
        rows = result.get("rows") or result.get("records") or []
        if not rows:
            return "No graph results."
        return "\n".join(str(r) for r in rows[:20])

    async def _arun(self, query: str) -> str:
        result = await self._aquery(query)
        rows = result.get("rows") or result.get("records") or []
        if not rows:
            return "No graph results."
        return "\n".join(str(r) for r in rows[:20])


⸻

5. LlamaIndex Integration

LlamaIndex has its own base classes, but same pattern: wrap CorpusClient.

5.1 LLM

# frameworks/llamaindex_llm.py
from __future__ import annotations
import asyncio
from typing import Any

from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata

from .corpus_client import CorpusClient


class CorpusLlamaIndexLLM(CustomLLM):
    """
    LlamaIndex CustomLLM backed by Corpus LLM endpoint.
    """

    def __init__(
        self,
        corpus: CorpusClient,
        *,
        model: str,
        temperature: float = 0.7,
    ):
        super().__init__()
        self._corpus = corpus
        self.model = model
        self.temperature = temperature

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model,
            is_chat_model=True,
        )

    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        messages = [{"role": "user", "content": prompt}]
        result = await self._corpus.llm_complete(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
        )
        return CompletionResponse(
            text=result["text"],
            raw={
                "model": result.get("model", self.model),
                "usage": result.get("usage", {}),
            },
        )

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        return asyncio.run(self.acomplete(prompt, **kwargs))


⸻

5.2 Embedding Model

# frameworks/llamaindex_embeddings.py
from __future__ import annotations
from typing import List

import asyncio
from llama_index.core.embeddings import BaseEmbedding

from .corpus_client import CorpusClient


class CorpusLlamaIndexEmbedding(BaseEmbedding):
    """
    LlamaIndex embedding model backed by Corpus embedding endpoint.
    """

    def __init__(self, corpus: CorpusClient, *, model: str):
        super().__init__()
        self._corpus = corpus
        self.model = model

    async def _aget_text_embedding(self, text: str) -> List[float]:
        res = await self._corpus.embed(model=self.model, text=text)
        return res["embedding"]["vector"]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        res = await self._corpus.embed_batch(model=self.model, texts=texts)
        return [e["vector"] for e in res["embeddings"]]


⸻

5.3 Vector Index / Storage

# frameworks/llamaindex_vectorstore.py
from __future__ import annotations
from typing import Any, Dict, List, Optional

from llama_index.core.vector_stores import VectorStore, VectorStoreQuery, VectorStoreQueryResult
from llama_index.core.schema import TextNode

import asyncio

from .corpus_client import CorpusClient
from .llamaindex_embeddings import CorpusLlamaIndexEmbedding


class CorpusLlamaIndexVectorStore(VectorStore):
    """
    VectorStore backed by Corpus embedding + vector endpoints.
    """

    def __init__(
        self,
        corpus: CorpusClient,
        *,
        namespace: str,
        embedding_model: str,
    ):
        self._corpus = corpus
        self.namespace = namespace
        self._emb = CorpusLlamaIndexEmbedding(corpus, model=embedding_model)

    async def _aadd(
        self,
        nodes: List[TextNode],
        **kwargs: Any,
    ) -> List[str]:
        texts = [n.text for n in nodes]
        embeddings = await self._emb._aget_text_embeddings(texts)
        metadatas: List[Dict[str, Any]] = [dict(n.metadata) for n in nodes]
        ids: List[str] = [n.id_ for n in nodes]

        vectors = [
            {"id": id_, "vector": vec, "metadata": meta}
            for id_, vec, meta in zip(ids, embeddings, metadatas)
        ]

        await self._corpus._post(
            self._corpus.vector_url,
            "vector.upsert",
            {"namespace": self.namespace, "vectors": vectors},
        )
        return ids

    def add(self, nodes: List[TextNode], **kwargs: Any) -> List[str]:
        return asyncio.run(self._aadd(nodes, **kwargs))

    async def _aquery(self, query: VectorStoreQuery) -> VectorStoreQueryResult:
        if query.query_embedding is None:
            # embed query text
            emb = await self._emb._aget_text_embedding(query.query_str or "")
            qvec = emb
        else:
        # user gave embedding directly
            qvec = list(query.query_embedding)

        res = await self._corpus.vector_query(
            namespace=self.namespace,
            vector=qvec,
            top_k=query.similarity_top_k or 4,
        )

        matches = res["matches"]
        nodes: List[TextNode] = []
        scores: List[float] = []

        for m in matches:
            vec = m.get("vector") or {}
            meta = vec.get("metadata", {})
            text = meta.get("text") or meta.get("content") or ""
            nodes.append(TextNode(text=text, metadata=meta))
            scores.append(m.get("score", 0.0))

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=scores,
        )

    def query(self, query: VectorStoreQuery) -> VectorStoreQueryResult:
        return asyncio.run(self._aquery(query))


⸻

5.4 Graph-backed Query Engine

# frameworks/llamaindex_graph_query.py
from __future__ import annotations
import asyncio
from typing import Any

from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.response.schema import Response

from .corpus_client import CorpusClient


class CorpusGraphQueryEngine(CustomQueryEngine):
    """
    LlamaIndex QueryEngine that delegates to Corpus graph endpoint.
    """

    def __init__(self, corpus: CorpusClient):
        self._corpus = corpus

    async def _aquery(self, query_str: str) -> Response:
        res = await self._corpus.graph_query(query=query_str)
        rows = res.get("rows") or res.get("records") or []
        txt = "\n".join(str(r) for r in rows[:50]) or "No graph results."
        return Response(response=txt)

    def query(self, query_str: str) -> Response:
        return asyncio.run(self._aquery(query_str))


⸻

6. Semantic Kernel Integration

Semantic Kernel wants connector classes that implement their base interfaces. Same plan.

6.1 Text Embedding Service

# frameworks/sk_embeddings.py
from __future__ import annotations
from typing import List

import asyncio
from semantic_kernel.connectors.ai import TextEmbeddingBase

from .corpus_client import CorpusClient


class CorpusSKEmbedding(TextEmbeddingBase):
    """
    Semantic Kernel TextEmbeddingBase backed by Corpus embedding endpoint.
    """

    def __init__(self, corpus: CorpusClient, *, model_id: str):
        super().__init__(model_id=model_id)
        self._corpus = corpus

    async def _generate_embeddings_async(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        res = await self._corpus.embed_batch(model=self.model_id, texts=texts)
        return [e["vector"] for e in res["embeddings"]]

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        return asyncio.run(self._generate_embeddings_async(texts))


⸻

6.2 Chat Completion Service

# frameworks/sk_chat.py
from __future__ import annotations
import asyncio
from typing import Any, List

from semantic_kernel.connectors.ai import ChatCompletionClientBase
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatHistory, ChatMessage

from .corpus_client import CorpusClient


class CorpusSKChat(ChatCompletionClientBase):
    """
    Semantic Kernel ChatCompletionClientBase backed by Corpus LLM endpoint.
    """

    def __init__(self, corpus: CorpusClient, *, model_id: str, temperature: float = 0.7):
        super().__init__(model_id=model_id)
        self._corpus = corpus
        self.temperature = temperature

    async def _acomplete_chat(
        self,
        chat_history: ChatHistory,
        **kwargs: Any,
    ) -> str:
        messages: List[dict] = []
        for m in chat_history.messages:
            messages.append({"role": m.role.value.lower(), "content": m.content})

        res = await self._corpus.llm_complete(
            model=self.model_id,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
        )
        return res["text"]

    def complete_chat(
        self,
        chat_history: ChatHistory,
        **kwargs: Any,
    ) -> str:
        return asyncio.run(self._acomplete_chat(chat_history, **kwargs))

(If your SK version expects a richer return type, wrap the string accordingly – but the core “messages → CorpusClient → text” shape is correct.)

⸻

6.3 Vector / Graph Connectors

For memory / vector stores, SK typically uses a MemoryStoreBase interface. Wrap CorpusClient.vector_query:

# frameworks/sk_memory.py
from __future__ import annotations
from typing import List, Tuple

import asyncio
from semantic_kernel.memory import MemoryStoreBase, MemoryRecord

from .corpus_client import CorpusClient


class CorpusSKMemoryStore(MemoryStoreBase):
    """
    Semantic Kernel memory store backed by Corpus vector endpoint.
    """

    def __init__(self, corpus: CorpusClient, *, namespace: str):
        self._corpus = corpus
        self.namespace = namespace

    async def _aget_nearest_matches(
        self,
        embedding: List[float],
        limit: int,
    ) -> List[Tuple[MemoryRecord, float]]:
        res = await self._corpus.vector_query(
            namespace=self.namespace,
            vector=embedding,
            top_k=limit,
        )
        matches = res["matches"]
        out: List[Tuple[MemoryRecord, float]] = []
        for m in matches:
            vec = m.get("vector") or {}
            meta = vec.get("metadata", {})
            text = meta.get("text") or meta.get("content") or ""
            record = MemoryRecord(
                id=meta.get("id") or vec.get("id"),
                text=text,
                description=meta.get("description"),
                external_source_name=meta.get("source"),
                additional_metadata=str(meta),
            )
            out.append((record, m.get("score", 0.0)))
        return out

    def get_nearest_matches(
        self,
        collection: str,  # maps to namespace if you want
        embedding: List[float],
        limit: int = 1,
        min_relevance_score: float = 0.0,
    ) -> List[Tuple[MemoryRecord, float]]:
        return asyncio.run(self._aget_nearest_matches(embedding, limit))

(You can add a graph connector similarly if SK wants one, but the vector one above is the main piece.)

⸻

7. MCP Integration

Here, Corpus becomes the backend of an MCP server. Tools call CorpusClient.

7.1 MCP Server Backed by Corpus Protocols

# mcp_server.py
from __future__ import annotations
import asyncio
from typing import Any, Dict

from mcp.server import Server

from frameworks.corpus_client import CorpusClient


async def main() -> None:
    corpus = CorpusClient(
        llm_url="http://localhost:8001/llm",
        embedding_url="http://localhost:8002/embedding",
        vector_url="http://localhost:8003/vector",
        graph_url="http://localhost:8004/graph",
        tenant="demo-tenant",
    )

    server = Server("corpus-mcp")

    @server.list_tools()
    async def list_tools() -> list[dict]:
        return [
            {
                "name": "corpus_complete",
                "description": "LLM completion via Corpus",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string"},
                        "model": {"type": "string"},
                    },
                    "required": ["prompt"],
                },
            },
            {
                "name": "corpus_vector_search",
                "description": "Vector search via Corpus vector endpoint",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "k": {"type": "integer"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "corpus_graph_query",
                "description": "Graph query via Corpus graph endpoint",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                },
            },
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> list[dict]:
        if name == "corpus_complete":
            prompt = arguments["prompt"]
            model = arguments.get("model", "gpt-4.1-mini")
            res = await corpus.llm_complete(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            return [{"type": "text", "text": res["text"]}]

        if name == "corpus_vector_search":
            query = arguments["query"]
            k = arguments.get("k", 4)
            # embed query
            e = await corpus.embed(model="embed-1", text=query)
            vec = e["embedding"]["vector"]
            r = await corpus.vector_query(
                namespace="mcp-docs",
                vector=vec,
                top_k=k,
            )
            lines = []
            for m in r["matches"]:
                meta = m.get("vector", {}).get("metadata", {})
                text = meta.get("text") or meta.get("content") or ""
                score = m.get("score")
                lines.append(f"[score={score:.3f}] {text}")
            return [{"type": "text", "text": "\n\n".join(lines)}]

        if name == "corpus_graph_query":
            q = arguments["query"]
            r = await corpus.graph_query(query=q)
            rows = r.get("rows") or r.get("records") or []
            txt = "\n".join(str(row) for row in rows[:50]) or "No graph results."
            return [{"type": "text", "text": txt}]

        return [{"type": "text", "text": f"Unknown tool {name}"}]

    async with server.run_stdio():
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())


⸻

7.2 Using Corpus from MCP-aware Frameworks

Once this server is wired, any MCP-aware client (Claude Desktop, Cursor, etc.) can use:
	•	corpus_complete → your LLM
	•	corpus_vector_search → embedding + vector
	•	corpus_graph_query → graph

Config is client-specific (e.g., JSON config telling it to run python mcp_server.py), but the underlying logic is the same: MCP tools call CorpusClient.

⸻

8. Contexts, Deadlines, Tenants & Error Mapping

Your server side enforces deadlines, tenants, error taxonomy. On the framework side:
	•	Always send ctx.deadline_ms reasonably (CorpusClient already does).
	•	Always send ctx.tenant (set once in CorpusClient).
	•	Convert framework-specific error types to/from CorpusError as needed.

Example thin error wrapper for LangChain:

# frameworks/errors.py
from __future__ import annotations

from .corpus_client import CorpusError


class FrameworkRateLimitError(RuntimeError):
    ...


class FrameworkUnavailableError(RuntimeError):
    ...


def map_corpus_error_to_framework(err: CorpusError) -> Exception:
    msg = str(err)
    if "RESOURCE_EXHAUSTED" in msg or "rate limit" in msg.lower():
        return FrameworkRateLimitError(msg)
    if "UNAVAILABLE" in msg:
        return FrameworkUnavailableError(msg)
    return err

Then use in adapters:

# inside CorpusLangChainChat._agenerate
try:
    result = await self._corpus.llm_complete(...)
except CorpusError as e:
    raise map_corpus_error_to_framework(e) from e

That way, framework users see framework-appropriate errors, but everything still travels over Corpus envelopes.

⸻

9. Framework Conformance & Smoke Tests

You don’t need a giant test matrix – just cheap, targeted smoke tests per framework.

Example (LangChain) in tests/test_langchain_llm.py:

from __future__ import annotations
import pytest
from unittest.mock import AsyncMock

from frameworks.corpus_client import CorpusClient, CorpusError
from frameworks.langchain_llm import CorpusLangChainChat


class DummyCorpusClient(CorpusClient):
    async def llm_complete(self, **kwargs):
        # Return a minimal valid 'completion' result
        return {
            "text": "hello from corpus",
            "model": kwargs["model"],
            "usage": {"prompt_tokens": 1, "completion_tokens": 3, "total_tokens": 4},
            "finish_reason": "stop",
        }


@pytest.mark.asyncio
async def test_langchain_chat_generates():
    corpus = DummyCorpusClient()
    llm = CorpusLangChainChat(corpus, model_name="test-model")
    result = await llm._agenerate([])
    assert result.generations[0].message.content == "hello from corpus"

Do similarly small tests for:
	•	Autogen: create_chat_completion returns expected dict shape.
	•	CrewAI: calling the tool / LLM returns a string.
	•	LlamaIndex: complete() returns CompletionResponse.
	•	Semantic Kernel: embedding/chat connectors run end-to-end.
	•	MCP: tool handlers return { "type": "text", "text": ... }.

⸻

10. Integration Launch Checklist (TL;DR)

By framework × domain:
	•	Autogen
	•	LLM wrapper (CorpusAutogenLLMClient)
	•	Embedding+Vector “memory” (CorpusMemoryStore)
	•	Vector tool (CorpusVectorSearchTool)
	•	Graph tool (CorpusGraphQueryTool)
	•	CrewAI
	•	LLM backend (CorpusCrewAILLM or equivalent)
	•	Vector tool (CorpusCrewAIVectorSearchTool)
	•	Graph tool (CorpusCrewAIGraphTool)
	•	LangChain
	•	ChatModel (CorpusLangChainChat)
	•	Embeddings (CorpusLangChainEmbeddings)
	•	VectorStore (CorpusLangChainVectorStore)
	•	Graph tool (CorpusLangChainGraphTool)
	•	LlamaIndex
	•	LLM (CorpusLlamaIndexLLM)
	•	Embedding model (CorpusLlamaIndexEmbedding)
	•	VectorStore (CorpusLlamaIndexVectorStore)
	•	Graph QueryEngine (CorpusGraphQueryEngine)
	•	Semantic Kernel
	•	TextEmbeddingBase (CorpusSKEmbedding)
	•	ChatCompletionClientBase (CorpusSKChat)
	•	MemoryStoreBase (CorpusSKMemoryStore)
	•	MCP
	•	MCP server exposing:
	•	corpus_complete
	•	corpus_vector_search
	•	corpus_graph_query

Cross-cutting:
	•	Shared CorpusClient is the single source of truth for envelopes.
	•	No framework adapter knows about op/ctx/args details; they just call CorpusClient.
	•	Every adapter returns framework-native types (or a very thin wrapper).
	•	Timeouts & tenants are consistently set in CorpusClient._make_ctx.
	•	Schema-level invariants (e.g., total_tokens = prompt + completion) are enforced server-side, not in framework glue.
	•	Smoke tests exist for each framework integration.
