Corpus SDK Framework Adapter Quick Start

Goal: Make Corpus Protocol–speaking services usable from 6 popular frameworks (Autogen, CrewAI, LangChain, LlamaIndex, Semantic Kernel, MCP) across all 4 domains (LLM, Embedding, Vector, Graph) – with thin, honest translation layers, not fantasy SDKs.

⸻

Table of Contents
	•		0.	Mental Model (What You’re Actually Building)
	•		1.	Prerequisites & Shared Wire Client
	•		2.	Autogen Integration
	•	2.1 LLM (Chat Completion)
	•	2.2 Embedding-backed “Memory”
	•	2.3 Vector Search Tool
	•	2.4 Graph Query Tool
	•	2.5 Wiring Agents
	•		3.	CrewAI Integration
	•	3.1 CrewAI LLM Backend
	•	3.2 CrewAI Embedding & Vector-backed Tools
	•	3.3 CrewAI Graph Tool
	•		4.	LangChain Integration
	•	4.1 Chat Model (LLM)
	•	4.2 Embeddings
	•	4.3 VectorStore
	•	4.4 Graph Tooling
	•		5.	LlamaIndex Integration
	•	5.1 LLM
	•	5.2 Embedding Model
	•	5.3 Vector Index / Storage
	•	5.4 Graph-backed Query Engine
	•		6.	Semantic Kernel Integration
	•	6.1 Text Embedding Service
	•	6.2 Chat Completion Service
	•	6.3 Vector / Graph Connectors
	•		7.	MCP Integration
	•	7.1 MCP Server Backed by Corpus Protocols
	•	7.2 Using Corpus from MCP-aware Frameworks
	•		8.	Contexts, Deadlines, Tenants & Error Mapping
	•		9.	Framework Conformance & Smoke Tests
	•		10.	Integration Launch Checklist (TL;dr)

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

Create:

# corpus_sdk/framework_client/client.py
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

Key idea:
	•	Wrap CorpusClient.llm_complete in an object Autogen can call as an LLM.
	•	Use CorpusClient.embed / embed_batch for “memory”.
	•	Use CorpusClient.vector_query and graph_query behind tools.

2.1 LLM (Chat Completion)

File: corpus_sdk/llm/framework_adapters/autogen.py

from __future__ import annotations
from typing import Any, Dict, List, Optional
import asyncio

from corpus_sdk.framework_client.client import CorpusClient
from corpus_sdk.llm.framework_adapters.common.llm_translation import create_llm_translator


class CorpusAutogenLLMClient:
    """
    Autogen-compatible LLM client.
    Exposes a method Autogen can call that looks roughly
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
        self._translator = create_llm_translator()

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
        corpus_args = self._translator.to_corpus(
            raw_messages=messages,
            model=self.model,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )

        result = await self._corpus.llm_complete(**corpus_args)

        return self._translator.to_autogen(result)

    def create_chat_completion_sync(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return asyncio.run(self.create_chat_completion(messages, **kwargs))

Usage in Autogen (pseudo):

from autogen import ConversableAgent

from corpus_sdk.framework_client.client import CorpusClient
from corpus_sdk.llm.framework_adapters.autogen import CorpusAutogenLLMClient

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

⸻

2.2 Embedding-backed “Memory”

Use CorpusClient.embed and vector_query to back Autogen’s retrieval agents.

File: corpus_sdk/embedding/framework_adapters/autogen.py

from __future__ import annotations
from typing import Any, Dict, List
import asyncio

from corpus_sdk.framework_client.client import CorpusClient


class CorpusMemoryStore:
    """
    Tiny 'memory store' backed by Corpus Embedding + Vector.
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

    async def add_texts(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]] | None = None,
    ) -> None:
        metadatas = metadatas or [{} for _ in texts]

        batch = await self._corpus.embed_batch(
            model=self.embedding_model,
            texts=texts,
        )
        vectors = batch["embeddings"]

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

        await self._corpus._post(
            self._corpus.vector_url,
            "vector.upsert",
            payload,
        )

    async def similarity_search(
        self,
        query: str,
        k: int = 4,
    ) -> List[Dict[str, Any]]:
        e = await self._corpus.embed(
            model=self.embedding_model,
            text=query,
        )
        vec = e["embedding"]["vector"]

        result = await self._corpus.vector_query(
            namespace=self.namespace,
            vector=vec,
            top_k=k,
        )
        return result["matches"]

    def similarity_search_sync(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        return asyncio.run(self.similarity_search(query, k=k))

⸻

2.3 Vector Search Tool

Turn similarity_search_sync into a callable tool.

File: corpus_sdk/vector/framework_adapters/autogen.py

from __future__ import annotations
from typing import List

from corpus_sdk.embedding.framework_adapters.autogen import CorpusMemoryStore


class CorpusVectorSearchTool:
    """
    Simple Python callable you register as an Autogen tool.
    """

    def __init__(self, store: CorpusMemoryStore):
        self._store = store

    def __call__(self, query: str, k: int = 4) -> str:
        matches = self._store.similarity_search_sync(query, k=k)
        chunks: List[str] = []
        for m in matches:
            meta = m.get("vector", {}).get("metadata", m.get("metadata", {}))
            text = meta.get("text") or meta.get("content") or "<no text>"
            score = m.get("score")
            chunks.append(f"[score={score:.3f}] {text}")
        return "\n\n".join(chunks)

⸻

2.4 Graph Query Tool

File: corpus_sdk/graph/framework_adapters/autogen.py

from __future__ import annotations
import asyncio
from typing import Any, Dict

from corpus_sdk.framework_client.client import CorpusClient


class CorpusGraphQueryTool:
    def __init__(self, corpus: CorpusClient):
        self._corpus = corpus

    async def aquery(self, query: str) -> Dict[str, Any]:
        return await self._corpus.graph_query(query=query)

    def __call__(self, query: str) -> str:
        result = asyncio.run(self.aquery(query))
        rows = result.get("rows") or result.get("records") or []
        if not rows:
            return "No results."
        return "\n".join(str(r) for r in rows[:20])

⸻

2.5 Wiring Agents

Tie all the above together:

File: corpus_sdk/autogen_system.py

from __future__ import annotations

from corpus_sdk.framework_client.client import CorpusClient
from corpus_sdk.llm.framework_adapters.autogen import CorpusAutogenLLMClient
from corpus_sdk.embedding.framework_adapters.autogen import CorpusMemoryStore
from corpus_sdk.vector.framework_adapters.autogen import CorpusVectorSearchTool
from corpus_sdk.graph.framework_adapters.autogen import CorpusGraphQueryTool


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

    return {
        "corpus": corpus,
        "llm_client": llm_client,
        "memory_store": memory_store,
        "vector_tool": vector_tool,
        "graph_tool": graph_tool,
    }

⸻

3. CrewAI Integration

CrewAI typically wants an LLM object plus BaseTool implementations. Same pattern: wrap CorpusClient.

3.1 CrewAI LLM Backend

File: corpus_sdk/llm/framework_adapters/crewai.py

from __future__ import annotations
import asyncio
from typing import Any

from corpus_sdk.framework_client.client import CorpusClient
from corpus_sdk.llm.framework_adapters.common.llm_translation import create_llm_translator


class CorpusCrewAILLM:
    """
    Minimal CrewAI-compatible LLM wrapper.
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
        self._translator = create_llm_translator()

    async def acomplete(self, prompt: str, **kwargs: Any) -> str:
        corpus_args = self._translator.to_corpus(
            raw_messages=[{"role": "user", "content": prompt}],
            model=self.model,
            temperature=kwargs.get("temperature", self.temperature),
        )
        result = await self._corpus.llm_complete(**corpus_args)
        return result["text"]

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        return asyncio.run(self.acomplete(prompt, **kwargs))

⸻

3.2 CrewAI Embedding & Vector-backed Tools

File: corpus_sdk/vector/framework_adapters/crewai.py

from __future__ import annotations
import asyncio
from typing import Any, Dict, List

from corpus_sdk.framework_client.client import CorpusClient


class CorpusCrewAIVectorSearchTool:
    """
    CrewAI Tool-like object: __call__(query: str) -> str.
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
        e = await self._corpus.embed(
            model=self.embedding_model,
            text=query,
        )
        vec = e["embedding"]["vector"]

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

⸻

3.3 CrewAI Graph Tool

File: corpus_sdk/graph/framework_adapters/crewai.py

from __future__ import annotations
import asyncio
from typing import Any

from corpus_sdk.framework_client.client import CorpusClient


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

⸻

4. LangChain Integration

LangChain is stricter about base class methods, but same pattern.

4.1 Chat Model (LLM)

File: corpus_sdk/llm/framework_adapters/langchain.py

from __future__ import annotations
import asyncio
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration

from corpus_sdk.framework_client.client import CorpusClient
from corpus_sdk.llm.framework_adapters.common.llm_translation import create_llm_translator


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
        self._translator = create_llm_translator()

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
                role = getattr(m, "role", "user")
                raw.append({"role": role, "content": m.content})
        return raw

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        return asyncio.run(self._agenerate(messages, stop=stop, **kwargs))

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        raw = self._convert_to_raw(messages)
        corpus_args = self._translator.to_corpus(
            raw_messages=raw,
            model=self.model_name,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens"),
        )
        result = await self._corpus.llm_complete(**corpus_args)

        msg = AIMessage(
            content=result["text"],
            additional_kwargs={"finish_reason": result.get("finish_reason")},
        )

        return ChatResult(
            generations=[ChatGeneration(message=msg)],
            llm_output={
                "model": result.get("model", self.model_name),
                "usage": result.get("usage", {}),
            },
        )

⸻

4.2 Embeddings

File: corpus_sdk/embedding/framework_adapters/langchain.py

from __future__ import annotations
import asyncio
from typing import List

from langchain_core.embeddings import Embeddings

from corpus_sdk.framework_client.client import CorpusClient


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

File: corpus_sdk/vector/framework_adapters/langchain.py

from __future__ import annotations
import asyncio
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from corpus_sdk.framework_client.client import CorpusClient
from corpus_sdk.embedding.framework_adapters.langchain import CorpusLangChainEmbeddings


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

File: corpus_sdk/graph/framework_adapters/langchain.py

from __future__ import annotations
import asyncio
from typing import Any

from langchain_core.tools import BaseTool

from corpus_sdk.framework_client.client import CorpusClient


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

Same pattern: wrap CorpusClient in LlamaIndex base classes.

5.1 LLM

File: corpus_sdk/llm/framework_adapters/llamaindex.py

from __future__ import annotations
import asyncio
from typing import Any

from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata

from corpus_sdk.framework_client.client import CorpusClient
from corpus_sdk.llm.framework_adapters.common.llm_translation import create_llm_translator


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
        self._translator = create_llm_translator()

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model,
            is_chat_model=True,
        )

    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        corpus_args = self._translator.to_corpus(
            raw_messages=[{"role": "user", "content": prompt}],
            model=self.model,
            temperature=kwargs.get("temperature", self.temperature),
        )
        result = await self._corpus.llm_complete(**corpus_args)
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

File: corpus_sdk/embedding/framework_adapters/llamaindex.py

from __future__ import annotations
from typing import List
import asyncio

from llama_index.core.embeddings import BaseEmbedding

from corpus_sdk.framework_client.client import CorpusClient


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

File: corpus_sdk/vector/framework_adapters/llamaindex.py

from __future__ import annotations
from typing import Any, Dict, List, Optional
import asyncio

from llama_index.core.vector_stores import VectorStore, VectorStoreQuery, VectorStoreQueryResult
from llama_index.core.schema import TextNode

from corpus_sdk.framework_client.client import CorpusClient
from corpus_sdk.embedding.framework_adapters.llamaindex import CorpusLlamaIndexEmbedding


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
            emb = await self._emb._aget_text_embedding(query.query_str or "")
            qvec = emb
        else:
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

File: corpus_sdk/graph/framework_adapters/llamaindex.py

from __future__ import annotations
import asyncio
from typing import Any

from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.response.schema import Response

from corpus_sdk.framework_client.client import CorpusClient


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

Semantic Kernel wants connector classes that implement their base interfaces.

6.1 Text Embedding Service

File: corpus_sdk/embedding/framework_adapters/semantic_kernel.py

from __future__ import annotations
from typing import List
import asyncio

from semantic_kernel.connectors.ai import TextEmbeddingBase

from corpus_sdk.framework_client.client import CorpusClient


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

File: corpus_sdk/llm/framework_adapters/semantic_kernel.py

from __future__ import annotations
import asyncio
from typing import Any, List

from semantic_kernel.connectors.ai import ChatCompletionClientBase
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatHistory

from corpus_sdk.framework_client.client import CorpusClient
from corpus_sdk.llm.framework_adapters.common.llm_translation import create_llm_translator


class CorpusSKChat(ChatCompletionClientBase):
    """
    Semantic Kernel ChatCompletionClientBase backed by Corpus LLM endpoint.
    """

    def __init__(self, corpus: CorpusClient, *, model_id: str, temperature: float = 0.7):
        super().__init__(model_id=model_id)
        self._corpus = corpus
        self.temperature = temperature
        self._translator = create_llm_translator()

    async def _acomplete_chat(
        self,
        chat_history: ChatHistory,
        **kwargs: Any,
    ) -> str:
        messages: List[dict] = [
            {"role": m.role.value.lower(), "content": m.content}
            for m in chat_history.messages
        ]

        corpus_args = self._translator.to_corpus(
            raw_messages=messages,
            model=self.model_id,
            temperature=kwargs.get("temperature", self.temperature),
        )

        res = await self._corpus.llm_complete(**corpus_args)
        return res["text"]

    def complete_chat(
        self,
        chat_history: ChatHistory,
        **kwargs: Any,
    ) -> str:
        return asyncio.run(self._acomplete_chat(chat_history, **kwargs))

⸻

6.3 Vector / Graph Connectors

For memory / vector stores, SK typically uses MemoryStoreBase.

File: corpus_sdk/vector/framework_adapters/semantic_kernel.py

from __future__ import annotations
from typing import List, Tuple
import asyncio

from semantic_kernel.memory import MemoryStoreBase, MemoryRecord

from corpus_sdk.framework_client.client import CorpusClient


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
        collection: str,
        embedding: List[float],
        limit: int = 1,
        min_relevance_score: float = 0.0,
    ) -> List[Tuple[MemoryRecord, float]]:
        return asyncio.run(self._aget_nearest_matches(embedding, limit))

⸻

7. MCP Integration

Corpus becomes the backend of an MCP server. Tools call CorpusClient.

7.1 MCP Server Backed by Corpus Protocols

File: mcp_server.py

from __future__ import annotations
import asyncio
from typing import Any, Dict

from mcp.server import Server

from corpus_sdk.framework_client.client import CorpusClient


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

File: corpus_sdk/framework_client/errors.py

from __future__ import annotations

from corpus_sdk.framework_client.client import CorpusError


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

Usage inside CorpusLangChainChat._agenerate:

from corpus_sdk.framework_client.errors import map_corpus_error_to_framework
from corpus_sdk.framework_client.client import CorpusError

# inside _agenerate
try:
    result = await self._corpus.llm_complete(**corpus_args)
except CorpusError as e:
    raise map_corpus_error_to_framework(e) from e

⸻

9. Framework Conformance & Smoke Tests

You don’t need a giant test matrix – just targeted smoke tests per framework.

Example (LangChain):

File: tests/test_langchain_llm.py

from __future__ import annotations
import pytest

from corpus_sdk.framework_client.client import CorpusClient
from corpus_sdk.llm.framework_adapters.langchain import CorpusLangChainChat


class DummyCorpusClient(CorpusClient):
    async def llm_complete(self, **kwargs):
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
	•	LLM wrapper (corpus_sdk/llm/framework_adapters/autogen.py)
	•	Embedding+Vector “memory” (corpus_sdk/embedding/framework_adapters/autogen.py)
	•	Vector tool (corpus_sdk/vector/framework_adapters/autogen.py)
	•	Graph tool (corpus_sdk/graph/framework_adapters/autogen.py)
	•	CrewAI
	•	LLM backend (corpus_sdk/llm/framework_adapters/crewai.py)
	•	Vector tool (corpus_sdk/vector/framework_adapters/crewai.py)
	•	Graph tool (corpus_sdk/graph/framework_adapters/crewai.py)
	•	LangChain
	•	ChatModel (corpus_sdk/llm/framework_adapters/langchain.py)
	•	Embeddings (corpus_sdk/embedding/framework_adapters/langchain.py)
	•	VectorStore (corpus_sdk/vector/framework_adapters/langchain.py)
	•	Graph tool (corpus_sdk/graph/framework_adapters/langchain.py)
	•	LlamaIndex
	•	LLM (corpus_sdk/llm/framework_adapters/llamaindex.py)
	•	Embedding model (corpus_sdk/embedding/framework_adapters/llamaindex.py)
	•	VectorStore (corpus_sdk/vector/framework_adapters/llamaindex.py)
	•	Graph QueryEngine (corpus_sdk/graph/framework_adapters/llamaindex.py)
	•	Semantic Kernel
	•	TextEmbeddingBase (corpus_sdk/embedding/framework_adapters/semantic_kernel.py)
	•	ChatCompletionClientBase (corpus_sdk/llm/framework_adapters/semantic_kernel.py)
	•	MemoryStoreBase (corpus_sdk/vector/framework_adapters/semantic_kernel.py)
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