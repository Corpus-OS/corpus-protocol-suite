# corpus_sdk/embedding/framework_adapters/crewai.py
# SPDX-License-Identifier: Apache-2.0

"""
CrewAI adapter for Corpus Embedding protocol.

This module exposes Corpus `EmbeddingProtocolV1` implementations as
embedding services within CrewAI agents and workflows, with:

- Seamless integration with CrewAI agent `embedder` attribute
- Support for CrewAI knowledge sources and RAG workflows
- Context normalization for CrewAI-specific execution context
- Framework-agnostic orchestration via `EmbeddingTranslator`
- Async → sync bridging using `AsyncBridge`
- Rich error context attachment for observability

The design follows CrewAI's adapter patterns while maintaining the
protocol-first Corpus embedding stack.
"""

from __future__ import annotations

import logging
from functools import cached_property, wraps
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Protocol,
    TypeVar,
    Callable,
    TypedDict,
)

from corpus_sdk.core.context_translation import (
    from_crewai as context_from_crewai,
)
from corpus_sdk.core.error_context import attach_context
from corpus_sdk.embedding.embedding_base import (
    EmbeddingProtocolV1,
    OperationContext,
)
from corpus_sdk.embedding.framework_adapters.common.embedding_translation import (
    EmbeddingTranslator,
    BatchConfig,
    TextNormalizationConfig,
    create_embedding_translator,
)

logger = logging.getLogger(__name__)

# Type variables for decorators
T = TypeVar("T")

# Error code constants
class ErrorCodes:
    INVALID_EMBEDDING_RESULT = "INVALID_EMBEDDING_RESULT"
    EMPTY_EMBEDDING_RESULT = "EMPTY_EMBEDDING_RESULT"
    EMBEDDING_CONVERSION_ERROR = "EMBEDDING_CONVERSION_ERROR"
    CREWAI_CONTEXT_INVALID = "CREWAI_CONTEXT_INVALID"


class CrewAIContext(TypedDict, total=False):
    """Structured type for CrewAI execution context."""
    agent_role: Optional[str]
    task_id: Optional[str]
    workflow: Optional[str]
    agent_id: Optional[str]
    crew_id: Optional[str]
    process_id: Optional[str]


class CrewAIEmbedder(Protocol):
    """
    Protocol representing the embedder interface expected by CrewAI agents.

    This allows type-safe integration with CrewAI's agent embedder system
    without requiring a hard dependency on CrewAI at type-check time.
    """

    def embed_documents(
        self,
        texts: List[str],
        *,
        crewai_context: Optional[CrewAIContext] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """Embed multiple documents for CrewAI RAG workflows."""
        ...

    def embed_query(
        self,
        text: str,
        *,
        crewai_context: Optional[CrewAIContext] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[float]:
        """Embed a single query for CrewAI retrieval."""
        ...


def with_embedding_error_context(
    operation: str,
    **context_kwargs: Any,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to automatically attach error context to embedding exceptions.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                enhanced_context = context_kwargs.copy()
                attach_context(
                    exc,
                    framework="crewai",
                    operation=f"embedding_{operation}",
                    **enhanced_context,
                )
                raise
        return wrapper
    return decorator


def with_async_embedding_error_context(
    operation: str,
    **context_kwargs: Any,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to automatically attach error context to async embedding exceptions.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return await func(*args, **kwargs)
            except Exception as exc:
                enhanced_context = context_kwargs.copy()
                attach_context(
                    exc,
                    framework="crewai",
                    operation=f"embedding_{operation}",
                    **enhanced_context,
                )
                raise
        return wrapper
    return decorator


class CrewAIConfig(TypedDict, total=False):
    """Structured configuration for CrewAI-specific settings."""
    max_embedding_retries: int
    fallback_to_simple_context: bool
    enable_agent_context_propagation: bool
    task_aware_batching: bool


class CorpusCrewAIEmbeddings:
    """
    CrewAI embedding service backed by a Corpus `EmbeddingProtocolV1` adapter.

    This class implements the CrewAI embedder interface and can be directly
    assigned to CrewAI agents via the `embedder` attribute.

    Example:
    ```python
    from crewai import Agent, Task, Crew
    from corpus_sdk.embedding.framework_adapters.crewai import create_embedder

    # Create embedder with optimized CrewAI configuration
    embedder = create_embedder(
        corpus_adapter=my_adapter,
        model="text-embedding-3-large",
        crewai_config={
            "max_embedding_retries": 3,
            "task_aware_batching": True
        }
    )

    # Use with CrewAI agent
    researcher = Agent(
        role="Senior Research Analyst",
        goal="Uncover breakthrough AI research insights",
        backstory="Expert analyst with deep technical understanding",
        embedder=embedder,
        tools=[web_search_tool, document_processor]
    )

    # Embedder automatically receives CrewAI context during execution
    task = Task(
        description="Research latest LLM architectures and their performance characteristics",
        agent=researcher,
        expected_output="Comprehensive analysis report"
    )

    crew = Crew(agents=[researcher], tasks=[task])
    ```

    Error Handling Example:
    ```python
    try:
        embeddings = embedder.embed_documents(
            texts=research_docs,
            crewai_context={"agent_role": "Researcher", "task_id": "llm_analysis"}
        )
    except Exception as e:
        # Rich error context automatically attached
        logger.error(f"Embedding failed with context: {e.__corpus_context__}")
    ```

    Attributes
    ----------
    corpus_adapter: Underlying Corpus embedding protocol adapter
    model: Optional default model identifier
    batch_config: Optional batching configuration
    text_normalization_config: Optional text normalization settings
    crewai_config: CrewAI-specific configuration with validation
    """

    def __init__(
        self,
        corpus_adapter: EmbeddingProtocolV1,
        model: Optional[str] = None,
        batch_config: Optional[BatchConfig] = None,
        text_normalization_config: Optional[TextNormalizationConfig] = None,
        crewai_config: Optional[CrewAIConfig] = None,
    ):
        self.corpus_adapter = corpus_adapter
        self.model = model
        self.batch_config = batch_config
        self.text_normalization_config = text_normalization_config
        self.crewai_config = self._validate_crewai_config(crewai_config or {})

        logger.info(
            f"CorpusCrewAIEmbeddings initialized with model: {model or 'default'}, "
            f"config: {self.crewai_config}"
        )

    def _validate_crewai_config(self, config: CrewAIConfig) -> CrewAIConfig:
        """Validate and normalize CrewAI configuration with sensible defaults."""
        validated = config.copy()
        
        # Set defaults for missing values
        if "max_embedding_retries" not in validated:
            validated["max_embedding_retries"] = 3
        if "fallback_to_simple_context" not in validated:
            validated["fallback_to_simple_context"] = True
        if "enable_agent_context_propagation" not in validated:
            validated["enable_agent_context_propagation"] = True
        if "task_aware_batching" not in validated:
            validated["task_aware_batching"] = False

        # Validate numeric ranges
        if validated["max_embedding_retries"] < 0:
            logger.warning("max_embedding_retries cannot be negative, setting to 0")
            validated["max_embedding_retries"] = 0

        return validated

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @cached_property
    def _translator(self) -> EmbeddingTranslator:
        """
        Lazily construct and cache the `EmbeddingTranslator`.
        
        Uses `cached_property` for thread safety and optimal performance
        in multi-agent CrewAI environments.
        """
        return create_embedding_translator(
            adapter=self.corpus_adapter,
            framework="crewai",
            translator=None,
            batch_config=self.batch_config,
            text_normalization_config=self.text_normalization_config,
        )

    def _build_contexts(
        self,
        *,
        crewai_context: Optional[CrewAIContext] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[Optional[OperationContext], Dict[str, Any], Dict[str, Any]]:
        """
        Build contexts for CrewAI execution environment with comprehensive validation.

        Returns
        -------
        Tuple of:
        - core_ctx: core OperationContext or None if no/invalid context
        - op_ctx_dict: normalized dict for embedding layer
        - framework_ctx: CrewAI-specific context for translator
        """
        core_ctx: Optional[OperationContext] = None
        op_ctx_dict: Dict[str, Any] = {}
        framework_ctx: Dict[str, Any] = {
            "framework": "crewai",
            "config": self.crewai_config,
        }

        # Convert CrewAI context to core OperationContext with comprehensive error handling
        if crewai_context is not None:
            try:
                # Validate CrewAI context structure
                self._validate_crewai_context_structure(crewai_context)
                
                core_ctx_candidate = context_from_crewai(crewai_context)
                if isinstance(core_ctx_candidate, OperationContext):
                    core_ctx = core_ctx_candidate
                    logger.debug(
                        "Successfully created OperationContext from CrewAI context "
                        "for agent: %s, task: %s",
                        crewai_context.get('agent_role', 'unknown'),
                        crewai_context.get('task_id', 'unknown')
                    )
                else:
                    logger.warning(
                        "context_from_crewai returned non-OperationContext type: %s. "
                        "Using empty context.",
                        type(core_ctx_candidate).__name__,
                    )
                    if self.crewai_config["fallback_to_simple_context"]:
                        # Create minimal context as fallback
                        core_ctx = OperationContext()
            except Exception as e:
                logger.warning(
                    "Failed to create OperationContext from crewai_context: %s. "
                    "Using empty context.",
                    e,
                )
                attach_context(
                    e,
                    framework="crewai",
                    operation="context_build",
                    crewai_context_snapshot=dict(crewai_context),
                    config=self.crewai_config,
                )

        # Build comprehensive context dictionary preserving rich OperationContext
        if core_ctx is not None:
            # Preserve the rich OperationContext object for maximum fidelity
            op_ctx_dict = {"_operation_context": core_ctx}
            
            # Include structured dict representation for compatibility
            if hasattr(core_ctx, "to_dict"):
                op_ctx_dict.update(core_ctx.to_dict())
            elif hasattr(core_ctx, "__dict__"):
                op_ctx_dict.update(core_ctx.__dict__)
        else:
            op_ctx_dict = {}

        # Framework-level context for CrewAI-specific optimizations
        effective_model = model or self.model
        if effective_model:
            framework_ctx["model"] = effective_model

        # Add rich CrewAI-specific context for observability and optimization
        if crewai_context:
            framework_ctx.update({
                "agent_role": crewai_context.get("agent_role"),
                "task_id": crewai_context.get("task_id"),
                "workflow": crewai_context.get("workflow"),
                "agent_id": crewai_context.get("agent_id"),
                "crew_id": crewai_context.get("crew_id"),
                "process_id": crewai_context.get("process_id"),
            })

            # Enable task-aware batching if configured
            if self.crewai_config["task_aware_batching"] and crewai_context.get("task_id"):
                framework_ctx["batch_strategy"] = f"task_aware_{crewai_context['task_id']}"

        # Include any extra call-specific hints while preserving structure
        framework_ctx.update({k: v for k, v in kwargs.items() if not k.startswith('_')})
        
        return core_ctx, op_ctx_dict, framework_ctx

    def _validate_crewai_context_structure(self, context: CrewAIContext) -> None:
        """Validate CrewAI context structure and log warnings for anomalies."""
        if not isinstance(context, dict):
            raise ValueError(
                f"[{ErrorCodes.CREWAI_CONTEXT_INVALID}] "
                f"CrewAI context must be a dictionary, got {type(context).__name__}"
            )

        # Check for required context fields in production scenarios
        if not context.get('agent_role') and not context.get('task_id'):
            logger.debug(
                "CrewAI context missing both agent_role and task_id - "
                "reduced observability for embeddings"
            )

    def _coerce_embedding_matrix(self, result: Any) -> List[List[float]]:
        """
        Coerce translator result into embedding matrix with comprehensive validation.

        Supports:
        - {"embeddings": [[...], [...]], "model": "...", "usage": {...}}
        - Direct matrix: [[...], [...]]
        - EmbedResult-like with `.embeddings` attribute

        Implemented without `match`/`case` for Python 3.9+ compatibility.
        """
        embeddings_obj: Any

        if isinstance(result, Mapping) and "embeddings" in result:
            embeddings_obj = result["embeddings"]
        elif hasattr(result, "embeddings"):
            embeddings_obj = getattr(result, "embeddings")
        else:
            embeddings_obj = result

        if not isinstance(embeddings_obj, Sequence):
            raise TypeError(
                f"[{ErrorCodes.INVALID_EMBEDDING_RESULT}] "
                f"Translator result does not contain valid embeddings sequence: "
                f"type={type(embeddings_obj).__name__}"
            )

        matrix: List[List[float]] = []
        for i, row in enumerate(embeddings_obj):
            if not isinstance(row, Sequence):
                raise TypeError(
                    f"[{ErrorCodes.INVALID_EMBEDDING_RESULT}] "
                    f"Expected embedding row to be sequence, got {type(row).__name__} at index {i}"
                )
            
            # Validate row is not empty
            if len(row) == 0:
                logger.warning(f"Empty embedding row at index {i}, skipping")
                continue
                
            try:
                embedding_vector = [float(x) for x in row]
                matrix.append(embedding_vector)
            except (TypeError, ValueError) as e:
                raise TypeError(
                    f"[{ErrorCodes.EMBEDDING_CONVERSION_ERROR}] "
                    f"Failed to convert embedding values to float at row {i}: {e}"
                ) from e

        if not matrix:
            raise ValueError(
                f"[{ErrorCodes.EMPTY_EMBEDDING_RESULT}] "
                "Translator returned no valid embedding rows"
            )

        logger.debug("Successfully coerced embedding matrix with %d rows", len(matrix))
        return matrix

    def _coerce_embedding_vector(self, result: Any) -> List[float]:
        """
        Coerce translator result for single-text embed with validation.
        """
        matrix = self._coerce_embedding_matrix(result)

        if len(matrix) > 1:
            logger.warning(
                "Expected single embedding for query, got %d rows; using first row. "
                "Context: %s",
                len(matrix),
                {"multiple_embeddings_received": len(matrix)}
            )

        return matrix[0]

    # ------------------------------------------------------------------ #
    # Core Embedding API (CrewAI Compatible)
    # ------------------------------------------------------------------ #

    @with_embedding_error_context("documents")
    def embed_documents(
        self,
        texts: List[str],
        *,
        crewai_context: Optional[CrewAIContext] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """
        Sync embedding for multiple documents.

        Used by CrewAI agents during RAG operations and knowledge processing.

        Parameters
        ----------
        texts: List of document texts to embed
        crewai_context: Optional CrewAI execution context for agent/task awareness
        model: Optional model override for this specific call
        **kwargs: Additional framework-specific parameters

        Returns
        -------
        List of embedding vectors, one per input text

        Raises
        ------
        TypeError: If embedding result format is invalid
        ValueError: If no embeddings are generated
        Exception: Any underlying embedding errors with enriched context
        """
        # Comprehensive batch size validation with CrewAI-aware thresholds
        if isinstance(texts, Sequence) and not isinstance(texts, (str, bytes)):
            batch_size = len(texts)
            
            # CrewAI-specific batch size recommendations
            if batch_size > 10_000:
                logger.warning(
                    "Large batch size %d detected for CrewAI workflow - "
                    "consider splitting across multiple tasks for better performance",
                    batch_size
                )
            
            if (
                batch_size > 50_000
                and (self.batch_config is None or getattr(self.batch_config, "max_batch_size", None) is None)
            ):
                logger.warning(
                    "embed_documents called with batch_size=%d and no batch_config.max_batch_size; "
                    "this may impact CrewAI agent responsiveness",
                    batch_size,
                )

        core_ctx, op_ctx_dict, framework_ctx = self._build_contexts(
            crewai_context=crewai_context,
            model=model,
            **kwargs,
        )

        # Pass the rich OperationContext through for maximum fidelity and observability
        if core_ctx is not None and self.crewai_config["enable_agent_context_propagation"]:
            framework_ctx["_operation_context"] = core_ctx

        logger.debug(
            "Embedding %d documents for CrewAI agent: %s, task: %s",
            len(texts),
            crewai_context.get('agent_role', 'unknown') if crewai_context else 'unknown',
            crewai_context.get('task_id', 'unknown') if crewai_context else 'unknown'
        )

        translated = self._translator.embed(
            raw_texts=texts,
            op_ctx=op_ctx_dict,
            framework_ctx=framework_ctx,
        )
        return self._coerce_embedding_matrix(translated)

    @with_embedding_error_context("query")
    def embed_query(
        self,
        text: str,
        *,
        crewai_context: Optional[CrewAIContext] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[float]:
        """
        Sync embedding for a single query.

        Used by CrewAI for query understanding and retrieval in agent workflows.

        Parameters
        ----------
        text: Query text to embed
        crewai_context: Optional CrewAI execution context
        model: Optional model override
        **kwargs: Additional parameters

        Returns
        -------
        Single embedding vector for the query text
        """
        core_ctx, op_ctx_dict, framework_ctx = self._build_contexts(
            crewai_context=crewai_context,
            model=model,
            **kwargs,
        )

        # Ensure rich context propagation for query understanding
        if core_ctx is not None and self.crewai_config["enable_agent_context_propagation"]:
            framework_ctx["_operation_context"] = core_ctx

        logger.debug(
            "Embedding query for CrewAI agent: %s",
            crewai_context.get('agent_role', 'unknown') if crewai_context else 'unknown'
        )

        translated = self._translator.embed(
            raw_texts=text,
            op_ctx=op_ctx_dict,
            framework_ctx=framework_ctx,
        )
        return self._coerce_embedding_vector(translated)

    # ------------------------------------------------------------------ #
    # Async API for CrewAI Flows
    # ------------------------------------------------------------------ #

    @with_async_embedding_error_context("documents")
    async def aembed_documents(
        self,
        texts: List[str],
        *,
        crewai_context: Optional[CrewAIContext] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """
        Async embedding for multiple documents.

        Designed for async CrewAI workflows and parallel agent execution.
        """
        if isinstance(texts, Sequence) and not isinstance(texts, (str, bytes)):
            batch_size = len(texts)
            if (
                batch_size > 10_000
                and (self.batch_config is None or getattr(self.batch_config, "max_batch_size", None) is None)
            ):
                logger.warning(
                    "aembed_documents called with batch_size=%d and no batch_config.max_batch_size; "
                    "consider enabling task_aware_batching in crewai_config",
                    batch_size,
                )

        core_ctx, op_ctx_dict, framework_ctx = self._build_contexts(
            crewai_context=crewai_context,
            model=model,
            **kwargs,
        )

        if core_ctx is not None and self.crewai_config["enable_agent_context_propagation"]:
            framework_ctx["_operation_context"] = core_ctx

        logger.debug(
            "Async embedding %d documents for CrewAI task: %s",
            len(texts),
            crewai_context.get('task_id', 'unknown') if crewai_context else 'unknown'
        )

        translated = await self._translator.arun_embed(
            raw_texts=texts,
            op_ctx=op_ctx_dict,
            framework_ctx=framework_ctx,
        )
        return self._coerce_embedding_matrix(translated)

    @with_async_embedding_error_context("query")
    async def aembed_query(
        self,
        text: str,
        *,
        crewai_context: Optional[CrewAIContext] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[float]:
        """
        Async embedding for a single query.

        Optimized for async CrewAI agent interactions and real-time workflows.
        """
        core_ctx, op_ctx_dict, framework_ctx = self._build_contexts(
            crewai_context=crewai_context,
            model=model,
            **kwargs,
        )

        if core_ctx is not None and self.crewai_config["enable_agent_context_propagation"]:
            framework_ctx["_operation_context"] = core_ctx

        translated = await self._translator.arun_embed(
            raw_texts=text,
            op_ctx=op_ctx_dict,
            framework_ctx=framework_ctx,
        )
        return self._coerce_embedding_vector(translated)


def create_embedder(
    corpus_adapter: EmbeddingProtocolV1,
    model: Optional[str] = None,
    **kwargs: Any,
) -> CrewAIEmbedder:
    """
    Create a CrewAI-compatible embedder for seamless agent integration.

    Example:
    ```python
    from crewai import Agent, Task, Crew
    from corpus_sdk.embedding.framework_adapters.crewai import create_embedder

    # Create optimized embedder for research crew
    research_embedder = create_embedder(
        corpus_adapter=research_adapter,
        model="text-embedding-3-large",
        crewai_config={
            "task_aware_batching": True,
            "max_embedding_retries": 3
        }
    )

    # Create different embedder for analysis crew with specialized config
    analysis_embedder = create_embedder(
        corpus_adapter=analysis_adapter, 
        model="text-embedding-3-small",
        crewai_config={
            "enable_agent_context_propagation": False  # Lighter weight for simple tasks
        }
    )

    research_agent = Agent(
        role="Research Specialist",
        goal="Conduct deep research analysis",
        backstory="Expert researcher",
        embedder=research_embedder
    )

    analysis_agent = Agent(
        role="Data Analyst", 
        goal="Analyze research findings",
        backstory="Statistical expert",
        embedder=analysis_embedder
    )
    ```

    Parameters
    ----------
    corpus_adapter: Corpus embedding protocol adapter
    model: Model identifier for embedding operations
    **kwargs: Additional arguments for CorpusCrewAIEmbeddings

    Returns
    -------
    CrewAIEmbedder compatible embedder instance optimized for CrewAI workflows
    """
    embedder = CorpusCrewAIEmbeddings(
        corpus_adapter=corpus_adapter,
        model=model,
        **kwargs,
    )

    logger.info(
        "CrewAI embedder created successfully with model: %s",
        model or "default"
    )
    
    return embedder


__all__ = [
    "CorpusCrewAIEmbeddings",
    "CrewAIEmbedder", 
    "CrewAIContext",
    "CrewAIConfig",
    "create_embedder",
    "ErrorCodes",
]