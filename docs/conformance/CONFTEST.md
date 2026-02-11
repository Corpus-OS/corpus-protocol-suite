# CORPUS Protocol Conformance Test Configuration V1.0

This `conftest.py` file provides the test configuration and fixtures for the CORPUS Protocol Conformance Framework. It integrates all components of the certification system and enables the unified testing of protocol adapters and frameworks.

## Quick Start

To run the CORPUS conformance tests with the default mock adapter:

```bash
# Install dependencies (if not already installed)
pip install pytest pysqlite3-binary

# Run all tests
pytest tests/ -v

# Test a specific protocol
pytest tests/llm -v

# Test with your own adapter
export CORPUS_ADAPTER=my_project.adapters:MyLLMAdapter
pytest tests/llm -v
```

## Overview

The `conftest.py` is a pytest configuration file that:
- Registers the CORPUS Protocol Plugin for certification scoring
- Provides the `adapter` fixture for testing any CORPUS implementation
- Configures test markers and command-line options
- Sets up the wire request case registry for conformance testing

## Installation

The configuration is automatically loaded when pytest runs tests in the CORPUS codebase. No manual setup is required beyond installing dependencies:

```bash
pip install -r requirements.txt
```

## Version Compatibility Matrix

### Python & Dependency Requirements
From the codebase analysis:

**Python Compatibility** (`conftest.py`):
```python
# SQLite workaround for ChromaDB/CrewAI compatibility
# ChromaDB requires SQLite >= 3.35.0, but system may have older version
# Use pysqlite3-binary which includes a newer SQLite version
import sys
try:
    import pysqlite3
    # Override sqlite3 module with pysqlite3 before any imports
    sys.modules['sqlite3'] = sys.modules['pysqlite3']
except ImportError:
    # If pysqlite3 not available, use system sqlite3
    # (some tests may skip due to version requirements)
    pass
```

**Minimum Requirements**:
- **Python**: 3.8+ (based on type hints and dataclass usage in the file)
- **SQLite**: ≥3.35.0 (or `pysqlite3-binary` package)
- **pytest**: 7.0+ (based on plugin API usage)

**JSON Schema Requirements** (implied by protocol configurations):
- **jsonschema**: Draft 2020-12 support required
- **Schema $id format**: Must be valid URI (http/https/urn/tag)

### Protocol Version ↔ Test Suite Mapping
From protocol configurations in `conftest.py`:

**Protocol Versions Supported**:
- **LLM Protocol V1.0** → `tests/llm/` and `tests/frameworks/llm/`
- **Vector Protocol V1.0** → `tests/vector/` and `tests/frameworks/vector/`
- **Graph Protocol V1.0** → `tests/graph/` and `tests/frameworks/graph/`
- **Embedding Protocol V1.0** → `tests/embedding/` and `tests/frameworks/embedding/`
- **Wire Request Suite** → `tests/live/` (tests all wire envelopes)
- **Schema Conformance Suite** → `tests/schema/` (validates all schemas)

### Breaking Changes History
From protocol configurations in `conftest.py`:
- **All protocols**: Require proper wire envelope structure with `op`, `ctx`, `args`
- **Context fields**: `ctx.deadline_ms` must be integer|null with minimum 0
- **Schema validation**: All wire envelopes must validate against JSON Schemas
- **Error handling**: Must follow specific error patterns per protocol

## Environment Variables

### Required
- **`CORPUS_ADAPTER`** (optional, default: `tests.mock.mock_llm_adapter:MockLLMAdapter`)
  - Specifies the adapter class to test in format `module:ClassName`
  - Example: `CORPUS_ADAPTER=my_project.adapters:MyLLMAdapter`

### Optional
- **`CORPUS_ENDPOINT`** - Optional endpoint URL for adapter instantiation
- **`CORPUS_STRICT`** - Set to `"1"` for strict scoring mode (skips/xfails count against certification)
- **`CORPUS_MAX_FAILURES`** - Limit failure output per category (prevents log spam)
- **`CORPUS_REPORT_JSON`** - Write JSON summary to this path
- **`CORPUS_REPORT_DIR`** - Write summary.json to this directory
- **`CORPUS_PLAIN_OUTPUT`** - Disable emoji output for CI environments

## Command Line Options

Run with `pytest` and these additional options:

```bash
# Basic usage
pytest tests/ -v

# Test specific protocol
pytest tests/llm -v
pytest tests/vector -v
pytest tests/graph -v
pytest tests/embedding -v

# Test framework adapters
pytest tests/frameworks/llm -v
pytest tests/frameworks/vector -v
pytest tests/frameworks/graph -v
pytest tests/frameworks/embedding -v

# Test specific components
pytest tests/live -v  # Wire conformance tests
pytest tests/schema -v  # Schema validation tests

# Custom adapter via command line
pytest tests/llm -v --adapter=my_project.adapters:MyLLMAdapter

# Filter by marker
pytest -m "llm" -v  # Only LLM protocol tests
pytest -m "vector" -v  # Only vector protocol tests
pytest -m "graph" -v  # Only graph protocol tests
pytest -m "embedding" -v  # Only embedding protocol tests
pytest -m "wire" -v  # Only wire conformance tests
pytest -m "schema" -v  # Only schema validation tests
pytest -m "not slow" -v  # Skip slow tests
pytest -m "conformance" -v  # All protocol conformance tests
```

## Available Markers

The following pytest markers are defined in `conftest.py`:

- **`llm`** - LLM Protocol V1.0 conformance tests
- **`vector`** - Vector Protocol V1.0 conformance tests  
- **`graph`** - Graph Protocol V1.0 conformance tests
- **`embedding`** - Embedding Protocol V1.0 conformance tests
- **`llm_frameworks`** - LLM framework adapter conformance tests
- **`vector_frameworks`** - Vector framework adapter conformance tests
- **`embedding_frameworks`** - Embedding framework adapter conformance tests
- **`graph_frameworks`** - Graph framework adapter conformance tests
- **`wire`** - Wire Request Conformance tests (`tests/live/`)
- **`schema`** - Schema conformance validation tests
- **`slow`** - Tests that take longer to run (skip with `-m 'not slow'`)
- **`conformance`** - All protocol conformance tests

## Fixtures

### `adapter` (session-scoped)

The primary fixture for testing CORPUS protocol adapters:

```python
def test_llm_completion(adapter):
    """Example test using the adapter fixture."""
    # The adapter is instantiated based on CORPUS_ADAPTER environment variable
    result = adapter.build_llm_complete_envelope(
        messages=[{"role": "user", "content": "Hello"}]
    )
    assert result["op"] == "llm.complete"
```

**How it works:**
1. Reads `CORPUS_ADAPTER` environment variable (or uses default mock adapter)
2. Imports the specified class using `module:ClassName` syntax
3. Instantiates with `CORPUS_ENDPOINT` if provided, otherwise uses no-arg constructor
4. Provides the same instance to all tests in the session for efficiency

## Certification Framework

The plugin automatically provides certification scoring with these tiers:

### Certification Tiers
- **Platinum** 🏆 - 100% passing across all protocols (production-ready)
- **Gold** 🥇 - 100% passing within a single protocol  
- **Silver** 🥈 - ≥80% passing (integration-ready)
- **Development** 🔬 - ≥50% passing (early development)
- **Below Development** ❌ - <50% passing

### Scoring Policies
- **Default mode**: Skipped/xfailed tests excluded from denominator
- **Strict mode** (`CORPUS_STRICT=1`): All collected tests count toward score

### Example Outputs

**Platinum Certification Success**:
```
================================================================================
CORPUS PROTOCOL SUITE - PLATINUM CERTIFIED
🔌 Adapter: tests.mock.mock_llm_adapter:MockLLMAdapter | CORPUS_ENDPOINT: not set | ⚖️ Strict: off

Protocol & Framework Conformance Status (scored / collected):
  ✅ PASS LLM Protocol V1.0: Gold (132/132 scored; 145 collected)
  ✅ PASS Vector Protocol V1.0: Gold (108/108 scored; 120 collected)
  ✅ PASS Graph Protocol V1.0: Gold (99/99 scored; 110 collected)
  ✅ PASS Embedding Protocol V1.0: Gold (135/135 scored; 150 collected)

🎯 Status: Ready for production deployment
⏱️ Completed in 42.3s
```

**Gold Certification (Partial Success)**:
```
================================================================================
CORPUS PROTOCOL SUITE - GOLD CERTIFIED
🔌 Adapter: tests.mock.mock_llm_adapter:MockLLMAdapter | CORPUS_ENDPOINT: not set | ⚖️ Strict: off

Protocol & Framework Conformance Status (scored / collected):
  ✅ PASS LLM Protocol V1.0: Gold (132/132 scored; 145 collected)
  ⚠️ WARN Vector Protocol V1.0: Silver (86/108 scored; 120 collected; 22 to Gold)
  ⚠️ WARN Graph Protocol V1.0: Development (60/99 scored; 110 collected; 20 to Silver)
  ✅ PASS Embedding Protocol V1.0: Gold (135/135 scored; 150 collected)

🎯 Focus on protocols below Gold to reach Platinum (100% scored pass).
ℹ️ Not Platinum because: 22 failed, 1 error, 39 skipped
⏱️ Completed in 42.3s
```

**Failure Analysis**:
```
================================================================================
❌ CORPUS PROTOCOL CONFORMANCE ANALYSIS
🔌 Adapter: tests.mock.mock_llm_adapter:MockLLMAdapter | CORPUS_ENDPOINT: not set | ⚖️ Strict: off
ℹ️ Not Platinum because: 15 failed, 2 error, 3 xpassed

Found 20 issue(s) across protocols.

--------------------------------------------------
🟥 FAILURES & ERRORS
LLM Protocol V1.0:
  ❌ Failure Wire Contract & Routing: 2 issue(s)
      Specification: §4.1 Wire-First Canonical Form
      Test: test_wire_envelope_validation
      Quick fix: Ensure all wire envelopes include required fields with correct types
      Detected: Wire envelope missing required fields per §4.1
...
```

## Writing Tests

### Protocol Conformance Tests
Place tests in the appropriate directory:
- `tests/llm/` - LLM protocol tests
- `tests/vector/` - Vector protocol tests
- `tests/graph/` - Graph protocol tests
- `tests/embedding/` - Embedding protocol tests

### Framework Adapter Tests
- `tests/frameworks/llm/` - LLM framework adapter tests
- `tests/frameworks/vector/` - Vector framework adapter tests
- `tests/frameworks/graph/` - Graph framework adapter tests
- `tests/frameworks/embedding/` - Embedding framework adapter tests

### Wire Conformance Tests
Use the wire case registry for envelope validation:

```python
from tests.live.wire_cases import get_pytest_params

@pytest.mark.parametrize("case", get_pytest_params(), ids=lambda c: c.id)
def test_wire_request_envelope(case, adapter):
    """Parameterized test for all wire request cases."""
    builder = getattr(adapter, case.build_method, None)
    if builder is None:
        pytest.skip(f"Adapter missing {case.build_method}")
    
    envelope = builder()
    # Validation happens automatically via test_wire_conformance.py
```

## Custom Adapter Implementation

To test your own adapter:

1. **Create an adapter class** that implements the required methods:
```python
# my_llm_adapter.py
class MyLLMAdapter:
    def build_llm_complete_envelope(self):
        return {
            "op": "llm.complete",
            "ctx": {"request_id": "test-123"},
            "args": {"messages": [{"role": "user", "content": "Hello"}]}
        }
    
    # Implement other build_* methods as needed
```

2. **Set environment variable**:
```bash
export CORPUS_ADAPTER=my_llm_adapter:MyLLMAdapter
```

3. **Run tests**:
```bash
pytest tests/llm -v
```

## Getting Unstuck: Common Errors & Solutions

### Troubleshooting Flowchart

```
Test Failures → Check Certification Output → Identify Protocol → Review Error Guidance
      ↓                                         ↓                       ↓
Adapter Issues                           Protocol Issues          Specification Issues
      ↓                                         ↓                       ↓
Check CORPUS_ADAPTER                    Check spec sections      Review error patterns
Check constructor signature             Review test categories   Check examples
Check endpoint configuration           Validate wire envelopes   See § references
```

### 1. **Adapter Instantiation Errors**

**Error Messages from `conftest.py`**:
```
AdapterValidationError: Failed to instantiate adapter 'module:ClassName' with endpoint.
AdapterValidationError: Failed to instantiate adapter 'module:ClassName' without arguments.
```

**Quick Fixes**:
```python
# If using CORPUS_ENDPOINT, ensure your adapter accepts it:
class MyAdapter:
    def __init__(self, endpoint=None):  # ✓ Accepts endpoint parameter
        pass
    
    # OR for no endpoint:
    def __init__(self):  # ✓ No arguments required
        pass
```

### 2. **Wire Validation Failures**

**Common Patterns** (based on protocol configurations):
- Missing required fields in wire envelope (`op`, `ctx`, `args`)
- Invalid field types (`deadline_ms` must be integer|null)
- Schema validation failures against JSON Schema

**Debug with**:
```bash
# See exact validation failure
pytest tests/live/test_wire_conformance.py::test_wire_request_envelope -v

# Run specific protocol tests
pytest tests/llm -v --tb=short
```

### 3. **Test Categorization Issues**

**From `conftest.py` categorization logic**:
```python
# Tests categorized by directory pattern:
# - tests/llm/ → "llm" protocol
# - tests/vector/ → "vector" protocol  
# - tests/frameworks/llm/ → "llm_frameworks"
# - tests/live/ → "wire" protocol
# - tests/schema/ → "schema" protocol
```

**Solution**: Place tests in correct directory according to protocol.

### 4. **Certification Level Not Reached**

**Common Issues**:
- **Failed tests**: Always block Platinum certification
- **XPassed tests**: Unexpected passes block certification (stale xfail markers)
- **Strict mode**: Skipped and xfailed tests also block Platinum when `CORPUS_STRICT=1`

**Solutions**:
1. Fix failing tests first
2. Remove or update xfail markers for xpassed tests
3. Consider running in non-strict mode during development: `CORPUS_STRICT=0`

### 5. **Performance Issues**

**Cache Statistics** (reported in terminal output):
```
🔧 Performance: 1423 cache hits, 87 misses (cache size: 256)
```

**Low hit rate?** Tests might not be following naming conventions. Check test paths match protocol directories.

### 6. **Error Guidance References**

Each protocol has detailed error guidance in `conftest.py`:
```python
error_guidance={
    "wire_contract": {
        "test_wire_envelope_validation": {
            "error_patterns": {
                "missing_required_fields": "Wire envelope missing required fields per §4.1",
                "invalid_field_types": "Field types don't match canonical form requirements",
            },
            "quick_fix": "Ensure all wire envelopes include required fields with correct types",
            "examples": "See §4.1 for wire envelope format and field requirements",
        }
    },
    # ... more guidance per protocol and category
}
```

## CI/CD Integration Examples

### GitHub Actions
```yaml
name: CORPUS Conformance Tests
on: [push, pull_request]
jobs:
  conformance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install pytest pysqlite3-binary
      - name: Run conformance tests
        env:
          CORPUS_ADAPTER: ${{ secrets.CORPUS_ADAPTER }}
          CORPUS_STRICT: '1'
          CORPUS_REPORT_JSON: test-results/summary.json
        run: pytest tests/ -v --junitxml=test-results/results.xml
      - name: Upload test results
        uses: actions/upload-artifact@v3
        with:
          name: test-results
          path: test-results/
```

### GitLab CI
```yaml
stages:
  - test

conformance:
  stage: test
  image: python:3.10
  before_script:
    - pip install pytest pysqlite3-binary
  script:
    - export CORPUS_ADAPTER=$CORPUS_ADAPTER
    - export CORPUS_STRICT=1
    - export CORPUS_REPORT_JSON=summary.json
    - pytest tests/ -v --junitxml=report.xml
  artifacts:
    paths:
      - summary.json
      - report.xml
    reports:
      junit: report.xml
```

### Jenkins Pipeline
```groovy
pipeline {
    agent any
    environment {
        CORPUS_ADAPTER = credentials('corpus-adapter')
        CORPUS_STRICT = '1'
    }
    stages {
        stage('Test') {
            steps {
                sh '''
                    python -m pip install pytest pysqlite3-binary
                    pytest tests/ -v --junitxml=test-results.xml
                '''
            }
            post {
                always {
                    junit 'test-results.xml'
                    archiveArtifacts 'summary.json'
                }
            }
        }
    }
}
```

## JSON Report Output

When `CORPUS_REPORT_JSON` or `CORPUS_REPORT_DIR` is set, a JSON summary is written:

```json
{
  "version": 2,
  "generated_at_epoch_s": 1678901234,
  "policy": {"strict": false, "max_failures": null},
  "adapter": {"spec": "tests.mock.mock_llm_adapter:MockLLMAdapter", "endpoint_set": false},
  "duration_s": 42.3,
  "protocols": {
    "llm": {
      "display_name": "LLM Protocol V1.0",
      "reference_levels": {"gold": 132, "silver": 106, "development": 66},
      "outcomes": {"passed": 132, "failed": 0, "error": 0, "skipped": 13, "xfailed": 0, "xpassed": 0},
      "collected_total": 145,
      "scored_total": 132,
      "scored_passed": 132,
      "level": "Gold",
      "tests_needed_to_next_level": 0
    },
    "vector": {
      "display_name": "Vector Protocol V1.0",
      "reference_levels": {"gold": 108, "silver": 87, "development": 54},
      "outcomes": {"passed": 108, "failed": 0, "error": 0, "skipped": 12, "xfailed": 0, "xpassed": 0},
      "collected_total": 120,
      "scored_total": 108,
      "scored_passed": 108,
      "level": "Gold",
      "tests_needed_to_next_level": 0
    }
  },
  "platinum_certified": true,
  "why_not_platinum": {
    "blocked": false,
    "totals": {"failed": 0, "error": 0, "skipped": 25, "xfailed": 0, "xpassed": 0},
    "strict": false,
    "reasons": []
  }
}
```

## Best Practices

1. **Use the `adapter` fixture** - Don't instantiate adapters directly in tests
2. **Mark slow tests** - Use `@pytest.mark.slow` for tests >1 second
3. **Filter with markers** - Use markers to run specific test subsets
4. **Check certification output** - Review the terminal summary after test runs
5. **Export JSON reports** - For CI/CD integration and compliance tracking
6. **Use strict mode in CI** - Set `CORPUS_STRICT=1` in production pipelines
7. **Implement all required methods** - Adapters should implement all `build_*_envelope` methods for their protocol
8. **Review error guidance** - Check the `error_guidance` section in `conftest.py` for specific test failures

## Architecture Integration

This `conftest.py` integrates with the complete CORPUS conformance framework:

- **Protocol Registry** - Central configuration for all 11 protocol suites
- **Test Categorizer** - Automatically categorizes tests by protocol and category
- **Certification Plugin** - Provides Platinum/Gold/Silver/Development scoring
- **Adapter System** - Dynamic loading and validation of adapter implementations

Together, they provide a comprehensive certification system for CORPUS Protocol implementations.

---

*For more details, see the CORPUS Protocol specification and conformance documentation.*
