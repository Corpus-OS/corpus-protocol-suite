# Optional additional imports (put these near the top of the file, guarded if needed)
try:
    from langchain.tools import BaseTool
except ImportError:  # pragma: no cover - only used when LangChain is installed
    BaseTool = object  # type: ignore[misc]


class CorpusGraphTool(BaseTool):
    """
    LangChain Tool wrapper around `CorpusLangChainGraphClient`.

    This allows the Corpus graph client to be used directly in LangChain
    agents / tools-based workflows.

    By default it exposes a very simple interface:

        - `query`: required graph query string
        - `params`: optional mapping of query parameters
        - `dialect`: optional query dialect override
        - `namespace`: optional namespace override
        - `timeout_ms`: optional per-call timeout

    You can subclass this Tool to tighten or reshape the input schema as
    needed for a particular agent.
    """

    # LangChain BaseTool core attributes
    name: str = "corpus_graph"
    description: str = (
        "Run graph queries against the Corpus graph service. "
        "Input is a JSON-encoded object with fields: "
        "`query` (required string), and optional `params`, "
        "`dialect`, `namespace`, `timeout_ms`."
    )

    def __init__(
        self,
        graph_client: CorpusLangChainGraphClient,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        # BaseTool may or may not be a real class depending on import guard above
        if hasattr(super(), "__init__"):
            super().__init__()  # type: ignore[call-arg]

        self._graph_client = graph_client
        if name is not None:
            self.name = name
        if description is not None:
            self.description = description

    def _parse_input(self, tool_input: Any) -> Dict[str, Any]:
        """
        Normalize LangChain tool input into a dict with the shape:

            {
                "query": str,
                "params": Optional[Mapping[str, Any]],
                "dialect": Optional[str],
                "namespace": Optional[str],
                "timeout_ms": Optional[int],
            }

        Accepts:
        - Plain string: treated as `query`
        - Mapping: expects `query` plus optional fields
        - JSON-encoded string: parsed into a mapping (best-effort)
        """
        if isinstance(tool_input, str):
            # Try to parse JSON if it looks structured; fall back to raw query.
            stripped = tool_input.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    import json

                    as_obj = json.loads(stripped)
                    if isinstance(as_obj, Mapping) and "query" in as_obj:
                        return dict(as_obj)  # type: ignore[arg-type]
                except Exception:  # noqa: BLE001
                    # Fall back to treating input as raw query text
                    pass

            return {"query": tool_input}

        if isinstance(tool_input, Mapping):
            return dict(tool_input)

        raise BadRequest(
            f"Unsupported tool input type for CorpusGraphTool: {type(tool_input).__name__}",
            code=ErrorCodes.BAD_ADAPTER_RESULT,
        )

    # -------------------------- sync --------------------------------- #

    def _run(self, tool_input: Any, *args: Any, **kwargs: Any) -> str:
        """
        Synchronous execution of the graph tool.

        Returns a stringified version of the `QueryResult` so it can be
        consumed by standard LangChain agents. Callers who need structured
        results should directly use `CorpusLangChainGraphClient`.
        """
        parsed = self._parse_input(tool_input)
        query = parsed.get("query")
        if not isinstance(query, str) or not query.strip():
            raise BadRequest(
                "CorpusGraphTool input must include a non-empty 'query' field",
                code=ErrorCodes.BAD_ADAPTER_RESULT,
            )

        params = parsed.get("params")
        dialect = parsed.get("dialect")
        namespace = parsed.get("namespace")
        timeout_ms = parsed.get("timeout_ms")

        # Delegate to the underlying graph client.
        result = self._graph_client.query(
            query=query,
            params=params if isinstance(params, Mapping) else None,
            dialect=str(dialect) if dialect is not None else None,
            namespace=str(namespace) if namespace is not None else None,
            timeout_ms=int(timeout_ms) if timeout_ms is not None else None,
            config=None,
            extra_context=None,
        )

        # We return a string so that generic agents can consume the output.
        try:
            import json

            return json.dumps(result.to_dict() if hasattr(result, "to_dict") else result, default=str)
        except Exception:  # noqa: BLE001
            return str(result)

    # -------------------------- async -------------------------------- #

    async def _arun(self, tool_input: Any, *args: Any, **kwargs: Any) -> str:
        """
        Asynchronous execution of the graph tool.

        Mirrors `_run` but delegates to `aquery` on the underlying client.
        """
        parsed = self._parse_input(tool_input)
        query = parsed.get("query")
        if not isinstance(query, str) or not query.strip():
            raise BadRequest(
                "CorpusGraphTool input must include a non-empty 'query' field",
                code=ErrorCodes.BAD_ADAPTER_RESULT,
            )

        params = parsed.get("params")
        dialect = parsed.get("dialect")
        namespace = parsed.get("namespace")
        timeout_ms = parsed.get("timeout_ms")

        result = await self._graph_client.aquery(
            query=query,
            params=params if isinstance(params, Mapping) else None,
            dialect=str(dialect) if dialect is not None else None,
            namespace=str(namespace) if namespace is not None else None,
            timeout_ms=int(timeout_ms) if timeout_ms is not None else None,
            config=None,
            extra_context=None,
        )

        try:
            import json

            return json.dumps(result.to_dict() if hasattr(result, "to_dict") else result, default=str)
        except Exception:  # noqa: BLE001
            return str(result)


def create_corpus_graph_tool(
    *,
    graph_adapter: GraphProtocolV1,
    default_dialect: Optional[str] = None,
    default_namespace: Optional[str] = None,
    default_timeout_ms: Optional[int] = None,
    framework_version: Optional[str] = None,
    name: str = "corpus_graph",
    description: Optional[str] = None,
    framework_translator: Optional[GraphFrameworkTranslator] = None,
) -> CorpusGraphTool:
    """
    Convenience factory to create a `CorpusLangChainGraphClient` and wrap it
    in a `CorpusGraphTool` in one go.

    Example
    -------
        graph_adapter = MyGraphAdapter(...)
        graph_tool = create_corpus_graph_tool(
            graph_adapter=graph_adapter,
            default_namespace="prod",
        )
        agent = initialize_agent(
            tools=[graph_tool],
            llm=...,
            agent_type=AgentType.OPENAI_FUNCTIONS,
        )
    """
    client = CorpusLangChainGraphClient(
        graph_adapter=graph_adapter,
        default_dialect=default_dialect,
        default_namespace=default_namespace,
        default_timeout_ms=default_timeout_ms,
        framework_version=framework_version,
        framework_translator=framework_translator,
    )
    return CorpusGraphTool(
        graph_client=client,
        name=name,
        description=description,
    )
