.PHONY: \
	test-conformance \
	test-all-conformance \
	test-llm-conformance \
	test-vector-conformance \
	test-graph-conformance \
	test-embedding-conformance \
	test-llm-frameworks \
	test-vector-frameworks \
	test-embedding-frameworks \
	test-graph-frameworks \
	test-fast \
	test-fast-llm \
	test-fast-vector \
	test-fast-graph \
	test-fast-embedding \
	test-fast-llm-frameworks \
	test-fast-vector-frameworks \
	test-fast-embedding-frameworks \
	test-fast-graph-frameworks \
	test-schema \
	test-schema-fast \
	test-golden \
	test-golden-fast \
	verify-schema \
	validate-env \
	quick-check \
	conformance-report \
	test-docker \
	test-ci \
	test-ci-fast \
	setup-test-env \
	upload-results \
	safety-check \
	check-deps \
	check-versions \
	verify \
	test-cli \
	test-root-conformance \
	test-wire \
	test-wire-fast \
	clean \
	help \
	validate-test-dirs

# Use bash because this Makefile relies on bash-isms (e.g., `read -p`).
SHELL := /bin/bash

# Configuration
PYTEST := pytest
PYTEST_ARGS ?= -v
PYTEST_JOBS ?= auto
COV_FAIL_UNDER ?= 80

# Protocols and directories (core protocol conformance suites)
# Keep in sync with Python PROTOCOL_PATHS core entries
PROTOCOLS := llm vector graph embedding
TEST_DIRS := $(foreach p,$(PROTOCOLS),tests/$(p))

# Extra non-protocol conformance/CLI tests
EXTRA_TEST_FILES := tests/cli.py tests/run_conformance.py

# Schema / Golden / Wire directories (matches Python PROTOCOL_PATHS)
SCHEMA_TEST_DIR := tests/schema
GOLDEN_TEST_DIR := tests/golden
WIRE_TEST_DIR := tests/live

# Framework adapter test directories (optional)
LLM_FRAMEWORKS_DIR := tests/frameworks/llm
VECTOR_FRAMEWORKS_DIR := tests/frameworks/vector
EMBEDDING_FRAMEWORKS_DIR := tests/frameworks/embedding
GRAPH_FRAMEWORKS_DIR := tests/frameworks/graph

# Derived configuration
PYTEST_PARALLEL := $(if $(filter-out 1,$(PYTEST_JOBS)),-n $(PYTEST_JOBS),)
COV_REPORT_TERM := --cov-report=term
COV_THRESHOLD := --cov-fail-under=$(COV_FAIL_UNDER)

# --------------------------------------------------------------------------- #
# Directory Validation
# --------------------------------------------------------------------------- #
# IMPORTANT: Do NOT fail at parse-time. This allows `make help`, `make clean`, etc.
# Validation is enforced only for targets that actually need core protocol test dirs.
validate-test-dirs:
	@missing=0; \
	for d in $(TEST_DIRS); do \
		if [ ! -d "$$d" ]; then \
			echo "❌ Test directory $$d not found"; \
			missing=1; \
		fi; \
	done; \
	if [ "$$missing" -ne 0 ]; then \
		echo ""; \
		echo "💡 If you're running from a partial checkout, ensure core protocol tests exist:"; \
		echo "   Expected: $(TEST_DIRS)"; \
		exit 1; \
	fi

# Soft-guard: warn (do not fail) if schema/golden/wire/framework dirs are missing
$(if $(wildcard $(SCHEMA_TEST_DIR)),,$(warning ⚠️  Schema test directory '$(SCHEMA_TEST_DIR)' not found))
$(if $(wildcard $(GOLDEN_TEST_DIR)),,$(warning ⚠️  Golden test directory '$(GOLDEN_TEST_DIR)' not found))
$(if $(wildcard $(WIRE_TEST_DIR)),,$(warning ⚠️  Wire test directory '$(WIRE_TEST_DIR)' not found))
$(if $(wildcard $(LLM_FRAMEWORKS_DIR)),,$(warning ⚠️  LLM framework test directory '$(LLM_FRAMEWORKS_DIR)' not found))
$(if $(wildcard $(VECTOR_FRAMEWORKS_DIR)),,$(warning ⚠️  Vector framework test directory '$(VECTOR_FRAMEWORKS_DIR)' not found))
$(if $(wildcard $(EMBEDDING_FRAMEWORKS_DIR)),,$(warning ⚠️  Embedding framework test directory '$(EMBEDDING_FRAMEWORKS_DIR)' not found))
$(if $(wildcard $(GRAPH_FRAMEWORKS_DIR)),,$(warning ⚠️  Graph framework test directory '$(GRAPH_FRAMEWORKS_DIR)' not found))

# --------------------------------------------------------------------------- #
# Dependency check
# --------------------------------------------------------------------------- #
# Keep the simple UX, but validate the things we actually rely on:
# - pytest
# - corpus_sdk
# - pytest-cov (for coverage flags used in most suites)
# - pytest-xdist (only required when parallelism is enabled)
check-deps:
	@echo "🔍 Checking test dependencies..."
	@python - <<'PY'
import sys, importlib

def must(mod):
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False

missing = []
for m in ("pytest", "corpus_sdk", "pytest_cov"):
    if not must(m):
        missing.append(m)

# xdist is only required if -n is used (PYTEST_JOBS != 1)
# We can't reliably read Make variables inside Python without env, so just check and warn here.
# The Makefile will still fail later if -n is used without xdist; this warning makes it obvious.
xdist_ok = must("xdist") or must("pytest_xdist")

if missing:
    print("❌ Error: Test dependencies not installed.")
    print("   Missing:", ", ".join(missing))
    print("   Please run: pip install .[test]")
    sys.exit(1)

if not xdist_ok:
    print("⚠️  Note: pytest-xdist not detected. Parallel runs (-n) may fail.")
    print("   Install with: pip install .[test]  (or pip install pytest-xdist)")

print("✅ Dependencies OK")
PY

# Optional: show key tool versions (useful in CI logs)
check-versions:
	@echo "📦 Checking critical dependency versions..."
	@python -c "import sys; print(f'Python {sys.version}')"
	@python - <<'PY'
import importlib
def ver(mod):
	try:
		m = importlib.import_module(mod)
		v = getattr(m, '__version__', 'unknown')
	except Exception as e:
		v = f'not installed ({e})'
	print(f'{mod} {v}')
for m in ('pytest','jsonschema','rfc3339_validator'):
	ver(m)
PY

# --------------------------------------------------------------------------- #
# Environment Validation & Setup
# --------------------------------------------------------------------------- #

# Validate critical environment variables
validate-env:
	@echo "🔧 Validating test environment..."
	@if [ -z "$${CORPUS_TEST_ENV}" ]; then \
		echo "⚠️  CORPUS_TEST_ENV not set, using default"; \
	fi
	@if [ -z "$${CORPUS_ENDPOINT}" ]; then \
		echo "⚠️  CORPUS_ENDPOINT not set, using default test endpoint"; \
	fi
	@echo "✅ Environment validation complete"

# Interactive test environment setup
# Fix: prior `command -v read` check is unreliable because `read` is a shell builtin.
# Use a TTY check instead; if non-interactive, print manual instructions.
setup-test-env:
	@echo "🎯 Setting up test environment..."
	@if [ -t 0 ]; then \
		read -p "Test endpoint [http://localhost:8080]: " endpoint; \
		endpoint=$${endpoint:-http://localhost:8080}; \
		read -p "API key [test-key]: " key; \
		key=$${key:-test-key}; \
		echo "CORPUS_ENDPOINT=$$endpoint" > .testenv; \
		echo "CORPUS_API_KEY=$$key" >> .testenv; \
		echo "✅ Test environment saved to .testenv"; \
		echo "   Load with: export $$(cat .testenv | xargs)"; \
	else \
		echo "❌ Non-interactive shell detected - manual setup required"; \
		echo "   Create .testenv with:"; \
		echo "   CORPUS_ENDPOINT=http://your-endpoint"; \
		echo "   CORPUS_API_KEY=your-api-key"; \
	fi

# Safety check for production environments
safety-check: validate-env
	@if [ "$${CORPUS_TEST_ENV}" = "production" ]; then \
		echo "❌ Cannot run full test suite in production"; \
		echo "   Use: make quick-check"; \
		exit 1; \
	fi

# --------------------------------------------------------------------------- #
# Run ALL protocol conformance suites (LLM + Vector + Graph + Embedding)
# + top-level orchestration/CLI tests
# NOTE: Framework adapter suites are opt-in via their own targets.
# --------------------------------------------------------------------------- #
test-conformance test-all-conformance: check-deps safety-check validate-test-dirs
	@echo "🚀 Running ALL protocol conformance suites..."
	@echo "   Protocols: $(PROTOCOLS)"
	@echo "   Extra test files: $(EXTRA_TEST_FILES)"
	@echo "   Parallel jobs: $(PYTEST_JOBS)"
	@echo "   Coverage threshold: $(COV_FAIL_UNDER)%"
	@echo "   Environment: $${CORPUS_TEST_ENV:-default}"
	$(PYTEST) \
		$(TEST_DIRS) \
		$(EXTRA_TEST_FILES) \
		$(PYTEST_ARGS) \
		$(PYTEST_PARALLEL) \
		--cov=corpus_sdk \
		$(COV_THRESHOLD) \
		$(COV_REPORT_TERM) \
		--cov-report=html:conformance_coverage_report \
		--cov-report=xml:conformance_coverage.xml \
		--junitxml=conformance_results.xml

# --------------------------------------------------------------------------- #
# Per-Protocol Conformance (Dynamic Targets)
# --------------------------------------------------------------------------- #

# Single target to handle all protocol conformance tests
test-%-conformance: check-deps validate-test-dirs
	@echo "🚀 Running $(shell echo $* | tr 'a-z' 'A-Z') Protocol V1 conformance tests..."
	@echo "   Parallel jobs: $(PYTEST_JOBS)"
	@echo "   Coverage threshold: $(COV_FAIL_UNDER)%"
	@echo "   Environment: $${CORPUS_TEST_ENV:-default}"
	$(PYTEST) tests/$* $(PYTEST_ARGS) $(PYTEST_PARALLEL) \
		--cov=corpus_sdk.$* \
		$(COV_THRESHOLD) \
		$(COV_REPORT_TERM) \
		--cov-report=html:$*_coverage_report \
		--junitxml=$*_results.xml

# Framework adapter suites: explicit targets (paths differ from tests/$*)
test-llm-frameworks: check-deps
	@echo "🚀 Running LLM Framework Adapters conformance tests..."
	@echo "   Directory: $(LLM_FRAMEWORKS_DIR)"
	@echo "   Parallel jobs: $(PYTEST_JOBS)"
	@echo "   Coverage threshold: $(COV_FAIL_UNDER)%"
	$(PYTEST) $(LLM_FRAMEWORKS_DIR) $(PYTEST_ARGS) $(PYTEST_PARALLEL) \
		--cov=corpus_sdk.llm.framework_adapters \
		$(COV_THRESHOLD) \
		$(COV_REPORT_TERM) \
		--cov-report=html:llm_frameworks_coverage_report \
		--junitxml=llm_frameworks_results.xml

test-vector-frameworks: check-deps
	@echo "🚀 Running Vector Framework Adapters conformance tests..."
	@echo "   Directory: $(VECTOR_FRAMEWORKS_DIR)"
	@echo "   Parallel jobs: $(PYTEST_JOBS)"
	@echo "   Coverage threshold: $(COV_FAIL_UNDER)%"
	$(PYTEST) $(VECTOR_FRAMEWORKS_DIR) $(PYTEST_ARGS) $(PYTEST_PARALLEL) \
		--cov=corpus_sdk.vector.framework_adapters \
		$(COV_THRESHOLD) \
		$(COV_REPORT_TERM) \
		--cov-report=html:vector_frameworks_coverage_report \
		--junitxml=vector_frameworks_results.xml

test-embedding-frameworks: check-deps
	@echo "🚀 Running Embedding Framework Adapters conformance tests..."
	@echo "   Directory: $(EMBEDDING_FRAMEWORKS_DIR)"
	@echo "   Parallel jobs: $(PYTEST_JOBS)"
	@echo "   Coverage threshold: $(COV_FAIL_UNDER)%"
	$(PYTEST) $(EMBEDDING_FRAMEWORKS_DIR) $(PYTEST_ARGS) $(PYTEST_PARALLEL) \
		--cov=corpus_sdk.embedding.framework_adapters \
		$(COV_THRESHOLD) \
		$(COV_REPORT_TERM) \
		--cov-report=html:embedding_frameworks_coverage_report \
		--junitxml=embedding_frameworks_results.xml

test-graph-frameworks: check-deps
	@echo "🚀 Running Graph Framework Adapters conformance tests..."
	@echo "   Directory: $(GRAPH_FRAMEWORKS_DIR)"
	@echo "   Parallel jobs: $(PYTEST_JOBS)"
	@echo "   Coverage threshold: $(COV_FAIL_UNDER)%"
	$(PYTEST) $(GRAPH_FRAMEWORKS_DIR) $(PYTEST_ARGS) $(PYTEST_PARALLEL) \
		--cov=corpus_sdk.graph.framework_adapters \
		$(COV_THRESHOLD) \
		$(COV_REPORT_TERM) \
		--cov-report=html:graph_frameworks_coverage_report \
		--junitxml=graph_frameworks_results.xml

# --------------------------------------------------------------------------- #
# Schema / Golden Conformance (Additive Targets)
# --------------------------------------------------------------------------- #

# Schema meta-lint only (no coverage—schema validation isn't code coverage)
test-schema: check-deps
	@echo "🧩 Running Schema Meta-Lint (JSON Schema Draft 2020-12)..."
	$(PYTEST) $(SCHEMA_TEST_DIR) $(PYTEST_ARGS) $(PYTEST_PARALLEL) --no-cov --junitxml=schema_results.xml

# Faster schema run (skip @slow)
test-schema-fast: check-deps
	@echo "⚡ Running fast Schema Meta-Lint..."
	$(PYTEST) $(SCHEMA_TEST_DIR) $(PYTEST_ARGS) $(PYTEST_PARALLEL) -m "not slow" --no-cov

# Golden wire-message validation (envelopes, frames, invariants). No coverage by default.
test-golden: check-deps
	@echo "🧪 Running Golden Wire-Message Validation..."
	$(PYTEST) $(GOLDEN_TEST_DIR) $(PYTEST_ARGS) $(PYTEST_PARALLEL) --no-cov --junitxml=golden_results.xml

# Fast golden run (skips tests marked 'slow')
test-golden-fast: check-deps
	@echo "⚡ Running fast Golden Wire-Message Validation..."
	$(PYTEST) $(GOLDEN_TEST_DIR) $(PYTEST_ARGS) $(PYTEST_PARALLEL) -m "not slow" --no-cov

# Convenience alias: schema meta-lint first, then golden messages
verify-schema: check-deps
	@echo "🔍 Verifying Schema Conformance (schema meta-lint + golden validation)..."
	$(PYTEST) $(SCHEMA_TEST_DIR) $(PYTEST_ARGS) $(PYTEST_PARALLEL) --no-cov && \
	$(PYTEST) $(GOLDEN_TEST_DIR) $(PYTEST_ARGS) $(PYTEST_PARALLEL) --no-cov

# --------------------------------------------------------------------------- #
# Wire Envelope Conformance (separate suite)
# --------------------------------------------------------------------------- #

test-wire: check-deps
	@echo "🧪 Running Wire Envelope Conformance tests (tests/live/test_wire_conformance.py)..."
	$(PYTEST) $(WIRE_TEST_DIR)/test_wire_conformance.py \
		$(PYTEST_ARGS) \
		$(PYTEST_PARALLEL) \
		--no-cov \
		--junitxml=wire_results.xml

test-wire-fast: check-deps
	@echo "⚡ Running fast Wire Envelope Conformance tests (no coverage, skipping slow)..."
	$(PYTEST) $(WIRE_TEST_DIR)/test_wire_conformance.py \
		$(PYTEST_ARGS) \
		$(PYTEST_PARALLEL) \
		-m "not slow" \
		--no-cov

# --------------------------------------------------------------------------- #
# Quick Verification / Smoke
# --------------------------------------------------------------------------- #

# Quick health check (smoke test)
quick-check: check-deps validate-test-dirs
	@echo "🔍 Quick health check..."
	$(PYTEST) tests/ -k "test_golden_validates or test_schema_meta" -v --no-cov -x

# --------------------------------------------------------------------------- #
# Additional dedicated targets for extra test files
# --------------------------------------------------------------------------- #

test-cli: check-deps
	@echo "🧪 Running CLI tests (tests/cli.py)..."
	$(PYTEST) tests/cli.py $(PYTEST_ARGS) $(PYTEST_PARALLEL) --no-cov

test-root-conformance: check-deps
	@echo "🧪 Running top-level conformance runner tests (tests/run_conformance.py)..."
	$(PYTEST) tests/run_conformance.py $(PYTEST_ARGS) $(PYTEST_PARALLEL) --no-cov

# --------------------------------------------------------------------------- #
# Reports & CI Integration
# --------------------------------------------------------------------------- #

# Generate conformance report (Delegated to python module)
conformance-report: test-conformance
	@echo "📊 Generating detailed conformance report..."
	@python -m tests.run_conformance conformance-report

# Upload results to conformance service
upload-results:
	@if [ -f conformance_report.json ]; then \
		echo "📤 Uploading conformance results..."; \
		curl -f -X POST https://api.corpus.io/conformance \
			-H "Content-Type: application/json" \
			-H "Authorization: Bearer $${CORPUS_API_KEY}" \
			-d @conformance_report.json > /dev/null 2>&1 && \
		echo "✅ Results uploaded successfully" || \
		echo "⚠️  Upload failed (service may be unavailable)"; \
	else \
		echo "❌ No conformance report found - run 'make conformance-report' first"; \
	fi

# --------------------------------------------------------------------------- #
# CI-Optimized Targets
# --------------------------------------------------------------------------- #

# Full CI pipeline (wire tests first for fast feedback, then full conformance)
test-ci: check-deps validate-env validate-test-dirs
	@echo "🏗️  Running CI-optimized conformance suite..."
	@echo "   Step 1: Wire envelope conformance (fast feedback)"
	$(PYTEST) $(WIRE_TEST_DIR)/test_wire_conformance.py \
		$(PYTEST_ARGS) \
		$(PYTEST_PARALLEL) \
		--no-cov \
		--junitxml=wire_results.xml || exit $$?
	@echo "   Step 2: Protocol conformance suites"
	@make test-conformance
	@make conformance-report
	@echo "✅ CI conformance suite complete"

# Fast CI pipeline (for PR validation) - includes wire tests
test-ci-fast: check-deps validate-test-dirs
	@echo "⚡ Running fast CI validation (includes wire tests)..."
	@make test-wire-fast
	@make test-fast
	@echo "✅ Fast CI validation complete"

# --------------------------------------------------------------------------- #
# Docker Support
# --------------------------------------------------------------------------- #

# Run tests in Docker
test-docker:
	@echo "🐳 Running tests in Docker..."
	docker build -t corpus-conformance .
	docker run --rm \
		-e CORPUS_TEST_ENV=$${CORPUS_TEST_ENV} \
		-e CORPUS_ENDPOINT=$${CORPUS_ENDPOINT} \
		-e CORPUS_API_KEY=$${CORPUS_API_KEY} \
		corpus-conformance make test-ci

# --------------------------------------------------------------------------- #
# Fast Test Runs (No Coverage)
# --------------------------------------------------------------------------- #

# Fast all tests (protocol suites + wire tests + extra test files)
test-fast: check-deps validate-test-dirs
	@echo "⚡ Running fast tests (no coverage, skipping slow tests)..."
	$(PYTEST) \
		$(TEST_DIRS) \
		$(WIRE_TEST_DIR)/test_wire_conformance.py \
		$(EXTRA_TEST_FILES) \
		$(PYTEST_ARGS) \
		$(PYTEST_PARALLEL) \
		-m "not slow" \
		--no-cov

# Fast per-protocol tests (explicit wire test still available via test-wire-fast)
test-fast-%: check-deps validate-test-dirs
	@echo "⚡ Running fast $(shell echo $* | tr 'a-z' 'A-Z') tests (no coverage, skipping slow)..."
	$(PYTEST) tests/$* $(PYTEST_ARGS) $(PYTEST_PARALLEL) -m "not slow" --no-cov

# Fast framework adapter suites
test-fast-llm-frameworks: check-deps
	@echo "⚡ Running fast LLM Framework Adapters tests (no coverage, skipping slow)..."
	$(PYTEST) $(LLM_FRAMEWORKS_DIR) $(PYTEST_ARGS) $(PYTEST_PARALLEL) -m "not slow" --no-cov

test-fast-vector-frameworks: check-deps
	@echo "⚡ Running fast Vector Framework Adapters tests (no coverage, skipping slow)..."
	$(PYTEST) $(VECTOR_FRAMEWORKS_DIR) $(PYTEST_ARGS) $(PYTEST_PARALLEL) -m "not slow" --no-cov

test-fast-embedding-frameworks: check-deps
	@echo "⚡ Running fast Embedding Framework Adapters tests (no coverage, skipping slow)..."
	$(PYTEST) $(EMBEDDING_FRAMEWORKS_DIR) $(PYTEST_ARGS) $(PYTEST_PARALLEL) -m "not slow" --no-cov

test-fast-graph-frameworks: check-deps
	@echo "⚡ Running fast Graph Framework Adapters tests (no coverage, skipping slow)..."
	$(PYTEST) $(GRAPH_FRAMEWORKS_DIR) $(PYTEST_ARGS) $(PYTEST_PARALLEL) -m "not slow" --no-cov

# --------------------------------------------------------------------------- #
# Verification & Utilities
# --------------------------------------------------------------------------- #

# Verify command (alias for test-conformance with better messaging)
verify: check-deps safety-check validate-test-dirs
	@echo "🔍 Running Corpus Protocol Conformance Suite..."
	@echo "   Protocols: $(PROTOCOLS)"
	@echo "   Extra test files: $(EXTRA_TEST_FILES)"
	@echo "   Parallel jobs: $(PYTEST_JOBS)"
	@echo "   Coverage threshold: $(COV_FAIL_UNDER)%"
	@echo "   Environment: $${CORPUS_TEST_ENV:-default}"
	$(PYTEST) \
		$(TEST_DIRS) \
		$(EXTRA_TEST_FILES) \
		$(PYTEST_ARGS) \
		$(PYTEST_PARALLEL) \
		--cov=corpus_sdk \
		$(COV_THRESHOLD) \
		$(COV_REPORT_TERM) \
		--cov-report=html:conformance_coverage_report \
		--cov-report=xml:conformance_coverage.xml \
		--junitxml=conformance_results.xml

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
		*/*/__pycache__ \
		*.xml \
		*.json \
		.testenv

# Help target
help:
	@echo "Corpus SDK Conformance Test Targets:"
	@echo ""
	@echo "  test-conformance           Run ALL protocol suites ($(PROTOCOLS)) plus extra tests ($(EXTRA_TEST_FILES))"
	@echo "  test-all-conformance       Alias for test-conformance"
	@echo "  verify                     Alias for test-conformance with verification messaging"
	@echo ""
	@echo "Per-Protocol Conformance:"
	@echo "  test-llm-conformance              Run only LLM Protocol V1 tests (tests/llm)"
	@echo "  test-vector-conformance           Run only Vector Protocol V1 tests (tests/vector)"
	@echo "  test-graph-conformance            Run only Graph Protocol V1 tests (tests/graph)"
	@echo "  test-embedding-conformance        Run only Embedding Protocol V1 tests (tests/embedding)"
	@echo ""
	@echo "Framework Adapter Suites:"
	@echo "  test-llm-frameworks               Run LLM Framework Adapters tests (tests/frameworks/llm)"
	@echo "  test-vector-frameworks            Run Vector Framework Adapters tests (tests/frameworks/vector)"
	@echo "  test-embedding-frameworks         Run Embedding Framework Adapters tests (tests/frameworks/embedding)"
	@echo "  test-graph-frameworks             Run Graph Framework Adapters tests (tests/frameworks/graph)"
	@echo "  test-fast-llm-frameworks          Fast LLM Framework Adapters tests (no coverage, skip slow)"
	@echo "  test-fast-vector-frameworks       Fast Vector Framework Adapters tests (no coverage, skip slow)"
	@echo "  test-fast-embedding-frameworks    Fast Embedding Framework Adapters tests (no coverage, skip slow)"
	@echo "  test-fast-graph-frameworks        Fast Graph Framework Adapters tests (no coverage, skip slow)"
	@echo ""
	@echo "Schema & Golden Conformance:"
	@echo "  test-schema                       Run schema meta-lint (Draft 2020-12, \$$id/\$$ref checks)"
	@echo "  test-golden                       Validate golden wire messages (envelopes/streams/invariants)"
	@echo "  verify-schema                     Run schema meta-lint + golden validation"
	@echo "  test-schema-fast                  Fast schema meta-lint (no coverage, skip slow)"
	@echo "  test-golden-fast                  Fast golden validation (no coverage, skip slow)"
	@echo ""
	@echo "Wire Envelope Conformance:"
	@echo "  test-wire                         Run wire-level envelope conformance (tests/live/test_wire_conformance.py)"
	@echo "  test-wire-fast                    Fast wire conformance (no coverage, skip slow)"
	@echo ""
	@echo "Extra Top-Level Tests:"
	@echo "  test-cli                          Run CLI wrapper tests (tests/cli.py)"
	@echo "  test-root-conformance             Run top-level conformance runner tests (tests/run_conformance.py)"
	@echo ""
	@echo "CI & Automation:"
	@echo "  test-ci                           Full CI pipeline (wire + conformance + report)"
	@echo "  test-ci-fast                      Fast CI pipeline (wire-fast + test-fast)"
	@echo "  conformance-report                Generate JSON summary with JUnit XML parsing"
	@echo "  upload-results                    Upload results to conformance service"
	@echo "  setup-test-env                    Interactive test environment configuration"
	@echo ""
	@echo "Quick / Docker / Environment:"
	@echo "  quick-check                       Smoke test subset (schema+golden)"
	@echo "  test-docker                       Build and run tests inside Docker"
	@echo "  validate-env                      Validate required environment variables"
	@echo "  safety-check                      Block full runs in production envs (use quick-check)"
	@echo "  check-versions                    Print key dependency versions"
	@echo ""
	@echo "Fast Testing (No Coverage):"
	@echo "  test-fast                         Run all tests quickly (protocol + wire + extra, no coverage, skip slow)"
	@echo "  test-fast-llm                     Run only LLM tests quickly"
	@echo "  test-fast-vector                  Run Vector tests quickly"
	@echo "  test-fast-graph                   Run Graph tests quickly"
	@echo "  test-fast-embedding               Run Embedding tests quickly"
	@echo "  test-fast-llm-frameworks          Run LLM Framework Adapters tests quickly"
	@echo "  test-fast-vector-frameworks       Run Vector Framework Adapters tests quickly"
	@echo "  test-fast-embedding-frameworks    Run Embedding Framework Adapters tests quickly"
	@echo "  test-fast-graph-frameworks        Run Graph Framework Adapters tests quickly"
	@echo "  test-wire-fast                    Run wire tests quickly (also available via test-fast)"
	@echo ""
	@echo "Utilities:"
	@echo "  check-deps                        Verify test dependencies are installed"
	@echo "  clean                             Remove all generated files and caches"
	@echo "  help                              Show this help message"
	@echo ""
	@echo "Configuration (override via environment/make args):"
	@echo "  PYTEST_ARGS=-x                    Stop on first failure"
	@echo "  PYTEST_ARGS=--tb=short            Shorter tracebacks"
	@echo "  PYTEST_JOBS=4                     Run 4 parallel jobs"
	@echo "  PYTEST_JOBS=1                     Disable parallel execution"
	@echo "  COV_FAIL_UNDER=90                 Require 90% coverage"
	@echo ""
	@echo "Examples:"
	@echo "  make test-conformance                      # Run all protocol + extra tests"
	@echo "  make test-llm-conformance                  # Run only LLM tests"
	@echo "  make test-wire                             # Run wire envelope conformance"
	@echo "  make test-llm-frameworks                   # Run all LLM framework adapter tests"
	@echo "  make test-vector-frameworks                # Run all Vector framework adapter tests"
	@echo "  make test-embedding-frameworks             # Run all embedding framework adapter tests"
	@echo "  make test-graph-frameworks                 # Run all graph framework adapter tests"
	@echo "  make test-fast                             # Run all tests quickly (protocol + wire + extra)"
	@echo "  make test-ci                               # Full CI pipeline (wire + conformance)"
	@echo "  make setup-test-env                        # Configure test environment"
	@echo "  make test-ci-fast                          # Fast CI pipeline"
	@echo "  make conformance-report upload-results     # Generate and upload report"
	@echo "  make test-docker                           # Run in Docker"
	@echo "  make PYTEST_JOBS=4 test-conformance        # Run with 4 parallel jobs"
	@echo "  make COV_FAIL_UNDER=90 verify              # Verify with 90% coverage"
	@echo "  make clean test-vector-conformance         # Clean then run Vector tests"
	@echo ""
	@echo "For advanced wire testing with adapter selection, filtering, and watch mode:"
	@echo "  Use the corpus-sdk CLI:"
	@echo "    corpus-sdk test-wire --adapter=openai"
	@echo "    corpus-sdk test-wire --watch"
	@echo "    corpus-sdk wire-list --component llm"

# Default target
.DEFAULT_GOAL := help
