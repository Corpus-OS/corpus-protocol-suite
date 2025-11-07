.PHONY: test-embedding-conformance

test-embedding-conformance:
	pytest tests/embedding -v \
		--cov=corpus_sdk.embedding \
		--cov-report=term \
		--cov-report=html:embedding_coverage_report
