Phase 1 – Core plumbing

tests/frameworks/registries/embedding_registry.py

Define the basic data structure for embedding frameworks (even if you just fill in one, e.g. LangChain, to start).

Fields: framework name, module path, adapter class, batch/query method names, context kwarg, async support.

tests/frameworks/conftest.py

Add fixtures that:

Load the embedding registry (EMBEDDING_FRAMEWORKS).

Parametrize a fixture like embedding_framework_descriptor.

Add a simple mock EmbeddingProtocolV1 adapter fixture (just enough to let tests instantiate adapters).

Once these two exist, you can already start writing contract tests.

Phase 2 – Interface contract first

test_contract_interface_conformance.py

Get this file working first. It should:

Instantiate each adapter using the registry + mock backend.

Assert interface bits: batch method exists, query method exists, types of return values, optional async methods, basic context manager support.

At this point, you already have a minimal, runnable “does this even look like a framework adapter?” check.

Phase 3 – Shapes + batching

test_contract_shapes_and_batching.py

Add tests that use the same fixtures to assert:

Matrix vs vector shapes are correct.

Extreme batch warning semantics kick in (just assert that your warn_if_extreme_batch hook was called).

Now the “happy path” behavior is covered generically.

Phase 4 – Context + error context

test_contract_context_and_error_context.py

Add tests to confirm, for each framework:

Passing a valid context mapping builds an OperationContext (using your translation helpers).

Passing a bogus context type doesn’t explode.

When the mock backend throws, the attached error has framework, operation, and the expected error codes.

This is where the observability contract is enforced.

Phase 5 – Stress with bad backends

test_with_mock_backends.py

Introduce “evil” mock backends / translators:

Return invalid shapes.

Return empty results.

Raise exceptions.

Assert each adapter’s public methods respond consistently (coercion errors, attached context, etc.).

Now you’ve hardened behavior against misbehaving adapters.

Phase 6 – Per-framework integration tests

Do these last, once the generic contract is stable. One file at a time:

test_langchain_adapter.py

Pydantic validation, RunnableConfig mapping, LANGCHAIN_AVAILABLE behavior, sync/async semantics.

test_llamaindex_adapter.py

Dimension enforcement, zero-vector behavior, Settings.embed_model registration, LlamaIndex context fields.

test_semantickernel_adapter.py

Dimension/zeros, Semantic Kernel context mapping, registration via multiple kernel APIs.

test_autogen_adapter.py

__call__ delegating to embed_documents, health/capabilities passthrough, create_retriever wiring.

test_crewai_adapter.py

CrewAIConfig defaults, context translation + task-aware batching, register_with_crewai attaching .embedder.

Phase 7 – Clean-up / docs (optional but nice)

Add a short markdown doc (even just in your repo’s docs or README section) explaining:

“To add a new embedding framework adapter, do X, then add a descriptor in embedding_registry.py, then run these tests.”
