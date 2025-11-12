.PHONY: \
	test-conformance \
	test-all-conformance \
	test-llm-conformance \
	test-vector-conformance \
	test-graph-conformance \
	test-embedding-conformance \
	test-fast \
	test-fast-llm \
	test-fast-vector \
	test-fast-graph \
	test-fast-embedding \
	check-deps \
	verify \
	clean \
	help

# Configuration
PYTEST := pytest
PYTEST_ARGS ?= -v
PYTEST_JOBS ?= auto
COV_FAIL_UNDER ?= 80

# Protocols and directories
PROTOCOLS := llm vector graph embedding
TEST_DIRS := $(foreach p,$(PROTOCOLS),tests/$(p))

# Derived configuration
PYTEST_PARALLEL := $(if $(filter-out 1,$(PYTEST_JOBS)),-n $(PYTEST_JOBS),)
COV_REPORT_TERM := --cov-report=term
COV_THRESHOLD := --cov-fail-under=$(COV_FAIL_UNDER)

# Validate test directories exist
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
	@echo "   Protocols: $(PROTOCOLS)"
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

# --------------------------------------------------------------------------- #
# Per-Protocol Conformance (Dynamic Targets)
# --------------------------------------------------------------------------- #

# Single target to handle all protocol conformance tests
test-%-conformance: check-deps
	@echo "🚀 Running $(shell echo $* | tr 'a-z' 'A-Z') Protocol V1 conformance tests..."
	@echo "   Parallel jobs: $(PYTEST_JOBS)"
	@echo "   Coverage threshold: $(COV_FAIL_UNDER)%"
	$(PYTEST) tests/$* $(PYTEST_ARGS) $(PYTEST_PARALLEL) \
		--cov=corpus_sdk.$* \
		$(COV_THRESHOLD) \
		$(COV_REPORT_TERM) \
		--cov-report=html:$*_coverage_report

# --------------------------------------------------------------------------- #
# Fast Test Runs (No Coverage)
# --------------------------------------------------------------------------- #

# Fast all tests
test-fast: check-deps
	@echo "⚡ Running fast tests (no coverage, skipping slow tests)..."
	$(PYTEST) $(TEST_DIRS) $(PYTEST_ARGS) $(PYTEST_PARALLEL) -m "not slow" --no-cov

# Fast per-protocol tests
test-fast-%: check-deps
	@echo "⚡ Running fast $(shell echo $* | tr 'a-z' 'A-Z') tests (no coverage, skipping slow)..."
	$(PYTEST) tests/$* $(PYTEST_ARGS) $(PYTEST_PARALLEL) -m "not slow" --no-cov

# --------------------------------------------------------------------------- #
# Verification & Utilities
# --------------------------------------------------------------------------- #

# Verify command (alias for test-conformance with better messaging)
verify: check-deps
	@echo "🔍 Running Corpus Protocol Conformance Suite..."
	@echo "   Protocols: $(PROTOCOLS)"
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

# Clean up generated files
clean:
	@echo "🧹 Cleaning up generated files..."
	rm -rf \
		*_coverage_report \
		conformance_coverage_report \
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
	@echo "  test-conformance       Run ALL protocol suites ($(PROTOCOLS))"
	@echo "  test-all-conformance   Alias for test-conformance"
	@echo "  verify                 Alias for test-conformance with verification messaging"
	@echo ""
	@echo "Per-Protocol Conformance:"
	@echo "  test-llm-conformance   Run only LLM Protocol V1 tests"
	@echo "  test-vector-conformance Run only Vector Protocol V1 tests"
	@echo "  test-graph-conformance Run only Graph Protocol V1 tests"
	@echo "  test-embedding-conformance Run only Embedding Protocol V1 tests"
	@echo ""
	@echo "Fast Testing (No Coverage):"
	@echo "  test-fast              Run all tests quickly (no coverage, skip slow)"
	@echo "  test-fast-llm          Run only LLM tests quickly"
	@echo "  test-fast-vector       Run only Vector tests quickly"
	@echo "  test-fast-graph        Run only Graph tests quickly"
	@echo "  test-fast-embedding    Run only Embedding tests quickly"
	@echo ""
	@echo "Utilities:"
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
	@echo "  make test-fast-vector                   # Run Vector tests quickly"
	@echo "  make PYTEST_JOBS=4 test-conformance     # Run with 4 parallel jobs"
	@echo "  make COV_FAIL_UNDER=90 verify           # Verify with 90% coverage"
	@echo "  make clean test-vector-conformance      # Clean then run Vector tests"
	@echo ""

# Default target
.DEFAULT_GOAL := help