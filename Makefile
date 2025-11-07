.PHONY: \
	test-conformance \
	test-all-conformance \
	test-llm-conformance \
	test-vector-conformance \
	test-graph-conformance \
	test-embedding-conformance \
	check-deps \
	clean \
	help

# Configuration
PYTEST := pytest
PYTEST_ARGS ?= -v
PYTEST_JOBS ?= auto
COV_FAIL_UNDER ?= 80

# Derived configuration
PYTEST_PARALLEL := $(if $(filter-out 1,$(PYTEST_JOBS)),-n $(PYTEST_JOBS),)
COV_REPORT_TERM := --cov-report=term
COV_REPORT_HTML := --cov-report=html:$(PROTOCOL)_coverage_report
COV_THRESHOLD := --cov-fail-under=$(COV_FAIL_UNDER)

# Test directories (validate existence)
TEST_DIRS := tests/llm tests/vector tests/graph tests/embedding
$(foreach dir,$(TEST_DIRS),$(if $(wildcard $(dir)),,$(error Test directory $(dir) not found)))

# Dependency check
check-deps:
	@echo "🔍 Checking test dependencies..."
	@python -c "import pytest, corpus_sdk" 2>/dev/null || \
		(echo "❌ Error: Test dependencies not installed."; \
		 echo "   Please run: pip install .[test]"; exit 1)
	@echo "✅ Dependencies OK"

# Run ALL protocol conformance suites (LLM + Vector + Graph + Embedding)
test-conformance test-all-conformance: check-deps
	@echo "🚀 Running ALL protocol conformance suites..."
	@echo "   Protocols: LLM, Vector, Graph, Embedding"
	@echo "   Parallel jobs: $(PYTEST_JOBS)"
	@echo "   Coverage threshold: $(COV_FAIL_UNDER)%"
	$(PYTEST) \
		$(TEST_DIRS) \
		$(PYTEST_ARGS) \
		$(PYTEST_PARALLEL) \
		--cov=corpus_sdk \
		$(COV_THRESHOLD) \
		$(COV_REPORT_TERM) \
		--cov-report=html:conformance_coverage_report

# LLM Protocol V1 conformance
test-llm-conformance: check-deps
	@echo "🚀 Running LLM Protocol V1 conformance tests..."
	@echo "   Parallel jobs: $(PYTEST_JOBS)"
	@echo "   Coverage threshold: $(COV_FAIL_UNDER)%"
	PROTOCOL=llm $(PYTEST) tests/llm $(PYTEST_ARGS) $(PYTEST_PARALLEL) \
		--cov=corpus_sdk.llm \
		$(COV_THRESHOLD) \
		$(COV_REPORT_TERM) \
		$(COV_REPORT_HTML)

# Vector Protocol V1 conformance
test-vector-conformance: check-deps
	@echo "🚀 Running Vector Protocol V1 conformance tests..."
	@echo "   Parallel jobs: $(PYTEST_JOBS)"
	@echo "   Coverage threshold: $(COV_FAIL_UNDER)%"
	PROTOCOL=vector $(PYTEST) tests/vector $(PYTEST_ARGS) $(PYTEST_PARALLEL) \
		--cov=corpus_sdk.vector \
		$(COV_THRESHOLD) \
		$(COV_REPORT_TERM) \
		$(COV_REPORT_HTML)

# Graph Protocol V1 conformance
test-graph-conformance: check-deps
	@echo "🚀 Running Graph Protocol V1 conformance tests..."
	@echo "   Parallel jobs: $(PYTEST_JOBS)"
	@echo "   Coverage threshold: $(COV_FAIL_UNDER)%"
	PROTOCOL=graph $(PYTEST) tests/graph $(PYTEST_ARGS) $(PYTEST_PARALLEL) \
		--cov=corpus_sdk.graph \
		$(COV_THRESHOLD) \
		$(COV_REPORT_TERM) \
		$(COV_REPORT_HTML)

# Embedding Protocol V1 conformance
test-embedding-conformance: check-deps
	@echo "🚀 Running Embedding Protocol V1 conformance tests..."
	@echo "   Parallel jobs: $(PYTEST_JOBS)"
	@echo "   Coverage threshold: $(COV_FAIL_UNDER)%"
	PROTOCOL=embedding $(PYTEST) tests/embedding $(PYTEST_ARGS) $(PYTEST_PARALLEL) \
		--cov=corpus_sdk.embedding \
		$(COV_THRESHOLD) \
		$(COV_REPORT_TERM) \
		$(COV_REPORT_HTML)

# Fast test runs (no coverage, parallel)
test-fast: check-deps
	@echo "⚡ Running fast tests (no coverage, parallel)..."
	$(PYTEST) $(TEST_DIRS) $(PYTEST_ARGS) -n auto --no-cov

# Clean up generated files
clean:
	@echo "🧹 Cleaning up generated files..."
	rm -rf \
		*_coverage_report \
		.coverage \
		.pytest_cache \
		htmlcov \
		.mypy_cache \
		__pycache__ \
		*/__pycache__ \
		*/*/__pycache__

# Help target
help:
	@echo "Corpus SDK Conformance Test Targets:"
	@echo ""
	@echo "  test-conformance       Run ALL protocol suites (LLM + Vector + Graph + Embedding)"
	@echo "  test-all-conformance   Alias for test-conformance"
	@echo "  test-llm-conformance   Run only LLM Protocol V1 tests"
	@echo "  test-vector-conformance Run only Vector Protocol V1 tests"
	@echo "  test-graph-conformance Run only Graph Protocol V1 tests"
	@echo "  test-embedding-conformance Run only Embedding Protocol V1 tests"
	@echo "  test-fast              Run tests quickly (no coverage, parallel)"
	@echo "  check-deps            Verify test dependencies are installed"
	@echo "  clean                 Remove all generated files and caches"
	@echo "  help                  Show this help message"
	@echo ""
	@echo "Configuration (override via environment/make args):"
	@echo "  PYTEST_ARGS=-x         Stop on first failure"
	@echo "  PYTEST_ARGS=--tb=short Shorter tracebacks"
	@echo "  PYTEST_JOBS=4          Run 4 parallel jobs"
	@echo "  PYTEST_JOBS=1          Disable parallel execution"
	@echo "  COV_FAIL_UNDER=90      Require 90% coverage"
	@echo ""
	@echo "Examples:"
	@echo "  make test-conformance                    # Run all tests"
	@echo "  make test-llm-conformance               # Run only LLM tests"
	@echo "  make PYTEST_JOBS=4 test-conformance     # Run with 4 parallel jobs"
	@echo "  make COV_FAIL_UNDER=90 test-conformance # Require 90% coverage"
	@echo "  make test-fast                         # Fast iteration (no coverage)"
	@echo "  make clean test-vector-conformance     # Clean then run Vector tests"
	@echo ""

# Default target
.DEFAULT_GOAL := help