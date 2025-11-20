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

# Extra non-protocol conformance/CLI tests
EXTRA_TEST_FILES := tests/cli.py tests/run_conformance.py

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
	@python -c "import importlib; print('pytest', getattr(importlib.import_module('pytest'), '__version__', 'unknown'))" 2>/dev/null || echo "pytest not installed"
	@python -c "import importlib; print('jsonschema', getattr(importlib.import_module('jsonschema'), '__version__', 'unknown'))" 2>/dev/null || echo "jsonschema not installed"  
	@python -c "import importlib; print('rfc3339_validator', getattr(importlib.import_module('rfc3339_validator'), '__version__', 'unknown'))" 2>/dev/null || echo "rfc3339_validator not installed"

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
setup-test-env:
	@echo "🎯 Setting up test environment..."
	@if command -v read > /dev/null 2>&1; then \
		read -p "Test endpoint [http://localhost:8080]: " endpoint; \
		endpoint=$${endpoint:-http://localhost:8080}; \
		read -p "API key [test-key]: " key; \
		key=$${key:-test-key}; \
		echo "CORPUS_ENDPOINT=$$endpoint" > .testenv; \
		echo "CORPUS_API_KEY=$$key" >> .testenv; \
		echo "✅ Test environment saved to .testenv"; \
		echo "   Load with: export \$$(cat .testenv | xargs)"; \
	else \
		echo "❌ 'read' command not available - manual setup required"; \
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
# --------------------------------------------------------------------------- #
test-conformance test-all-conformance: check-deps safety-check
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
test-%-conformance: check-deps
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
# Quick Verification / Smoke
# --------------------------------------------------------------------------- #

# Quick health check (smoke test)
quick-check: check-deps
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

# Generate conformance report (with error handling and duration aggregation if JUnit XML is present)
conformance-report: test-conformance
	@echo "📊 Generating detailed conformance report..."
	@echo 'import json, datetime, os, glob, re' > .report_gen.py
	@echo 'from xml.etree import ElementTree' >> .report_gen.py
	@echo 'total_tests = 0; total_failures = 0; total_errors = 0; total_time = 0.0' >> .report_gen.py
	@echo 'for xml_file in glob.glob("*_results.xml"):' >> .report_gen.py
	@echo '    try:' >> .report_gen.py
	@echo '        tree = ElementTree.parse(xml_file)' >> .report_gen.py
	@echo '        root = tree.getroot()' >> .report_gen.py
	@echo '        total_tests += int(root.get("tests", 0))' >> .report_gen.py
	@echo '        total_failures += int(root.get("failures", 0))' >> .report_gen.py
	@echo '        total_errors += int(root.get("errors", 0))' >> .report_gen.py
	@echo '        total_time += float(root.get("time", 0))' >> .report_gen.py
	@echo '    except Exception as e:' >> .report_gen.py
	@echo '        print(f"⚠️  Could not parse {xml_file}: {e}")' >> .report_gen.py
	@echo 'status = "PASS" if total_failures == 0 and total_errors == 0 else "FAIL"' >> .report_gen.py
	@echo 'results = {"protocols": "$(PROTOCOLS)", "status": status, "timestamp": datetime.datetime.utcnow().isoformat() + "Z", "summary": {"total_tests": total_tests, "failures": total_failures, "errors": total_errors, "duration_seconds": round(total_time, 3)}, "coverage_threshold": $(COV_FAIL_UNDER), "test_suites": ["schema", "golden", "llm", "vector", "graph", "embedding", "cli", "root_conformance"], "environment": os.getenv("CORPUS_TEST_ENV", "default")}' >> .report_gen.py
	@echo 'with open("conformance_report.json", "w", encoding="utf-8") as f:' >> .report_gen.py
	@echo '    json.dump(results, f, indent=2)' >> .report_gen.py
	@echo 'print("✅ Conformance report: conformance_report.json")' >> .report_gen.py
	@echo 'print(f"📈 Summary: {total_tests} tests, {total_failures} failures, {total_errors} errors")' >> .report_gen.py
	@echo 'print(f"⏱️  Duration: {round(total_time, 3)}s")' >> .report_gen.py
	@echo 'print(f"🎯 Status: {status}")' >> .report_gen.py
	@python .report_gen.py || (echo "❌ Failed to generate report" && exit 1)
	@rm -f .report_gen.py

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

# Full CI pipeline
test-ci: check-deps validate-env
	@echo "🏗️  Running CI-optimized conformance suite..."
	@make test-conformance
	@make conformance-report
	@echo "✅ CI conformance suite complete"

# Fast CI pipeline (for PR validation)
test-ci-fast: check-deps
	@echo "⚡ Running fast CI validation..."
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

# Fast all tests (protocol suites + extra test files)
test-fast: check-deps
	@echo "⚡ Running fast tests (no coverage, skipping slow tests)..."
	$(PYTEST) $(TEST_DIRS) $(EXTRA_TEST_FILES) $(PYTEST_ARGS) $(PYTEST_PARALLEL) -m "not slow" --no-cov

# Fast per-protocol tests
test-fast-%: check-deps
	@echo "⚡ Running fast $(shell echo $* | tr 'a-z' 'A-Z') tests (no coverage, skipping slow)..."
	$(PYTEST) tests/$* $(PYTEST_ARGS) $(PYTEST_PARALLEL) -m "not slow" --no-cov

# --------------------------------------------------------------------------- #
# Verification & Utilities
# --------------------------------------------------------------------------- #

# Verify command (alias for test-conformance with better messaging)
verify: check-deps safety-check
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
	@echo "  test-llm-conformance       Run only LLM Protocol V1 tests (tests/llm)"
	@echo "  test-vector-conformance    Run only Vector Protocol V1 tests (tests/vector)"
	@echo "  test-graph-conformance     Run only Graph Protocol V1 tests (tests/graph)"
	@echo "  test-embedding-conformance Run only Embedding Protocol V1 tests (tests/embedding)"
	@echo ""
	@echo "Schema & Golden Conformance:"
	@echo "  test-schema                Run schema meta-lint (Draft 2020-12, \$id/\$ref checks)"
	@echo "  test-golden                Validate golden wire messages (envelopes/streams/invariants)"
	@echo "  verify-schema              Run schema meta-lint + golden validation"
	@echo "  test-schema-fast           Fast schema meta-lint (no coverage, skip slow)"
	@echo "  test-golden-fast           Fast golden validation (no coverage, skip slow)"
	@echo ""
	@echo "Extra Top-Level Tests:"
	@echo "  test-cli                   Run CLI wrapper tests (tests/cli.py)"
	@echo "  test-root-conformance      Run top-level conformance runner tests (tests/run_conformance.py)"
	@echo ""
	@echo "CI & Automation:"
	@echo "  test-ci                    Full CI pipeline (deps+env+test+report)"
	@echo "  test-ci-fast               Fast CI pipeline for PR validation"
	@echo "  conformance-report         Generate JSON summary with JUnit XML parsing"
	@echo "  upload-results             Upload results to conformance service"
	@echo "  setup-test-env             Interactive environment configuration"
	@echo ""
	@echo "Quick / Docker / Environment:"
	@echo "  quick-check                Smoke test subset (schema+golden)"
	@echo "  test-docker                Build and run tests inside Docker"
	@echo "  validate-env               Validate required environment variables"
	@echo "  safety-check               Block full runs in production envs (use quick-check)"
	@echo "  check-versions             Print key dependency versions"
	@echo ""
	@echo "Fast Testing (No Coverage):"
	@echo "  test-fast                  Run all tests quickly (protocol + extra, no coverage, skip slow)"
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
	@echo "  make test-conformance                      # Run all protocol + extra tests"
	@echo "  make test-llm-conformance                 # Run only LLM tests"
	@echo "  make test-ci                              # Full CI pipeline"
	@echo "  make setup-test-env                       # Configure test environment"
	@echo "  make test-ci-fast                         # Fast CI for PRs"
	@echo "  make conformance-report upload-results    # Generate and upload report"
	@echo "  make test-docker                          # Run in Docker"
	@echo "  make PYTEST_JOBS=4 test-conformance       # Run with 4 parallel jobs"
	@echo "  make COV_FAIL_UNDER=90 verify             # Verify with 90% coverage"
	@echo "  make clean test-vector-conformance        # Clean then run Vector tests"

# Default target
.DEFAULT_GOAL := help
