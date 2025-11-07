.PHONY: \
	test-conformance \
	test-all-conformance \
	test-llm-conformance \
	test-vector-conformance \
	test-graph-conformance \
	test-embedding-conformance

# Run ALL protocol conformance suites (LLM + Vector + Graph + Embedding)
test-conformance test-all-conformance:
	pytest \
		tests/llm \
		tests/vector \
		tests/graph \
		tests/embedding \
		-v \
		--cov=corpus_sdk \
		--cov-report=term \
		--cov-report=html:conformance_coverage_report

# LLM Protocol V1 conformance
test-llm-conformance:
	pytest tests/llm -v \
		--cov=corpus_sdk.llm \
		--cov-report=term \
		--cov-report=html:llm_coverage_report

# Vector Protocol V1 conformance
test-vector-conformance:
	pytest tests/vector -v \
		--cov=corpus_sdk.vector \
		--cov-report=term \
		--cov-report=html:vector_coverage_report

# Graph Protocol V1 conformance
test-graph-conformance:
	pytest tests/graph -v \
		--cov=corpus_sdk.graph \
		--cov-report=term \
		--cov-report=html:graph_coverage_report

# Embedding Protocol V1 conformance
test-embedding-conformance:
	pytest tests/embedding -v \
		--cov=corpus_sdk.embedding \
		--cov-report=term \
		--cov-report=html:embedding_coverage_report
