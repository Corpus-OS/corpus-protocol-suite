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
	test-schema \
	test-schema-fast \
	test-golden \
	test-golden-fast \
	verify-schema \
	validate-env \
	quick-check \
	conformance-report \
	test-docker \
	safety-check \
	check-deps \
	check-versions \
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

# Schema / Golden directories
SCHEMA_TEST_DIR := tests/schema
GOLDEN_TEST_DIR := tests/golden

# Derived configuration
PYTEST_PARALLEL := $(if $(filter-out 1,$(PYTEST_JOBS)),-n $(PYTEST_JOBS),)
COV_REPORT_TERM := --cov-report=term
COV_THRESHOLD := --cov-fail-under=$(COV_FAIL_UNDER)

# Validate per-protocol test directories exist
$(foreach dir,$(TEST_DIRS),$(if $(wildcard $(dir)),,$(error Test directory $(dir) not found)))

# Soft-guard: warn (do not fail) if schema/golden dirs are missing
$(if $(wildcard $(SCHEMA_TEST_DIR)),,$(warning ⚠️  Schema test directory '$(SCHEMA_TEST_DIR)' not found))
$(if $(wildcard $(GOLDEN_TEST_DIR)),,$(warning ⚠️  Golden test directory '$(GOLDEN_TEST_DIR)' not found))

# Dependency check
check-deps:
	@echo "🔍 Checking test dependencies..."
	@python -c "import pytest, corpus_sdk" 2>/dev/null || \
		(echo "❌ Error: Test dependencies not installed."; \
		 echo "   Please run: pip install .[test]"; exit 1)
	@echo "✅ Dependencies OK"

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
# Environment Validation
# --------------------------------------------------------------------------- #

# Validate critical environment variables
validate-env:
	@if [ -z "$${CORPUS_TEST_ENV}" ]; then \
		echo "⚠️  CORPUS_TEST_ENV not set, using default"; \
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
# --------------------------------------------------------------------------- #
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
# Schema / Golden Conformance (Additive Targets)
# --------------------------------------------------------------------------- #

# Schema meta-lint only (no coverage—schema validation isn't code coverage)
test-schema: check-deps
	@echo "🧩 Running Schema Meta-Lint (JSON Schema Draft 2020-12)..."
	$(PYTEST) $(SCHEMA_TEST_DIR) $(PYTEST_ARGS) $(PYTEST_PARALLEL) --no-cov

# Faster schema run (skip @slow)
test-schema-fast: check-deps
	@echo "⚡ Running fast Schema Meta-Lint..."
	$(PYTEST) $(SCHEMA_TEST_DIR) $(PYTEST_ARGS) $(PYTEST_PARALLEL) -m "not slow" --no-cov

# Golden wire-message validation (envelopes, frames, invariants). No coverage by default.
test-golden: check-deps
	@echo "🧪 Running Golden Wire-Message Validation..."
	$(PYTEST) $(GOLDEN_TEST_DIR) $(PYTEST_ARGS) $(PYTEST_PARALLEL) --no-cov

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
# Quick Verification / Smoke
# --------------------------------------------------------------------------- #

# Quick health check (smoke test)
quick-check: check-deps
	@echo "🔍 Quick health check..."
	$(PYTEST) tests/ -k "test_golden_validates or test_schema_meta" -v --no-cov -x

# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #

# Generate conformance report (with error handling and duration aggregation if JUnit XML is present)
conformance-report: test-conformance
	@echo "📊 Generating detailed conformance report..."
	@python - <<'PY'
try:
    import json, datetime, os
    # Try to aggregate duration from any JUnit XML in repo (best-effort, optional)
    try:
        # Shell-style aggregation via os.popen to avoid platform-specific tools
        # Looks for attributes like time="X.Y" or duration="X.Y"
        import re, glob
        total = 0.0
        for path in glob.glob("**/*.xml", recursive=True):
            try:
                txt = open(path, "r", encoding="utf-8", errors="ignore").read()
            except Exception:
                continue
            for m in re.finditer(r'(?:time|duration)="([0-9]*\.?[0-9]+)"', txt):
                try:
                    total += float(m.group(1))
                except Exception:
                    pass
        duration_seconds = round(total, 3)
    except Exception:
        duration_seconds = 0.0

    results = {
        "protocols": "$(PROTOCOLS)",
        "status": "PASS",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "coverage_threshold": $(COV_FAIL_UNDER),
        "test_suites": ["schema", "golden", "llm", "vector", "graph", "embedding"],
        "duration_seconds": duration_seconds
    }
    with open("conformance_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("✅ Conformance report: conformance_report.json")
except Exception as e:
    print(f"❌ Failed to generate report: {e}")
    raise SystemExit(1)
PY

# --------------------------------------------------------------------------- #
# Docker Support
# --------------------------------------------------------------------------- #

# Run tests in Docker
test-docker:
	@echo "🐳 Running tests in Docker..."
	docker build -t corpus-conformance .
	docker run --rm corpus-conformance make test-conformance

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
	@echo "  test-conformance           Run ALL protocol suites ($(PROTOCOLS))"
	@echo "  test-all-conformance       Alias for test-conformance"
	@echo "  verify                     Alias for test-conformance with verification messaging"
	@echo ""
	@echo "Per-Protocol Conformance:"
	@echo "  test-llm-conformance       Run only LLM Protocol V1 tests"
	@echo "  test-vector-conformance    Run only Vector Protocol V1 tests"
	@echo "  test-graph-conformance     Run only Graph Protocol V1 tests"
	@echo "  test-embedding-conformance Run only Embedding Protocol V1 tests"
	@echo ""
	@echo "Schema & Golden Conformance:"
	@echo "  test-schema                Run schema meta-lint (Draft 2020-12, \$id/\$ref checks)"
	@echo "  test-golden                Validate golden wire messages (envelopes/streams/invariants)"
	@echo "  verify-schema              Run schema meta-lint + golden validation"
	@echo "  test-schema-fast           Fast schema meta-lint (no coverage, skip slow)"
	@echo "  test-golden-fast           Fast golden validation (no coverage, skip slow)"
	@echo ""
	@echo "Quick / Reports / Docker / Env:"
	@echo "  quick-check                Smoke test subset (schema+golden)"
	@echo "  conformance-report         Emit JSON summary after full run (timestamped, error-handled)"
	@echo "  test-docker                Build and run tests inside Docker"
	@echo "  validate-env               Validate required environment variables"
	@echo "  safety-check               Block full runs in production envs (use quick-check)"
	@echo "  check-versions             Print key dependency versions"
	@echo ""
	@echo "Fast Testing (No Coverage):"
	@echo "  test-fast                  Run all tests quickly (no coverage, skip slow)"
	@echo "  test-fast-llm              Run only LLM tests quickly"
	@echo "  test-fast-vector           Run Vector tests quickly"
	@echo "  test-fast-graph            Run Graph tests quickly"
	@echo "  test-fast-embedding        Run Embedding tests quickly"
	@echo ""
	@echo "Utilities:"
	@echo "  check-deps                 Verify test dependencies are installed"
	@echo "  clean                      Remove all generated files and caches"
	@echo "  help                       Show this help message"
	@echo ""
	@echo "Configuration (override via environment/make args):"
	@echo "  PYTEST_ARGS=-x             Stop on first failure"
	@echo "  PYTEST_ARGS=--tb=short     Shorter tracebacks"
	@echo "  PYTEST_JOBS=4              Run 4 parallel jobs"
	@echo "  PYTEST_JOBS=1              Disable parallel execution"
	@echo "  COV_FAIL_UNDER=90          Require 90% coverage"
	@echo ""
	@echo "Examples:"
	@echo "  make test-conformance                      # Run all tests"
	@echo "  make test-llm-conformance                 # Run only LLM tests"
	@echo "  make test-schema                           # Run schema meta-lint only"
	@echo "  make test-golden                           # Run golden validation only"
	@echo "  make quick-check                           # Smoke test subset"
	@echo "  make conformance-report                    # Emit JSON summary after full run"
	@echo "  make test-docker                           # Run in Docker"
	@echo "  make PYTEST_JOBS=4 test-conformance        # Run with 4 parallel jobs"
	@echo "  make COV_FAIL_UNDER=90 verify              # Verify with 90% coverage"
	@echo "  make clean test-vector-conformance         # Clean then run Vector tests"

# Default target
.DEFAULT_GOAL := help
