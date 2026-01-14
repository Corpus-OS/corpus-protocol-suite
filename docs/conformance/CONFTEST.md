# `conftest.py` - CORPUS Protocol Conformance Test Configuration

This `conftest.py` file provides the test configuration and fixtures for the CORPUS Protocol Conformance Framework. It integrates all components of the certification system and enables the unified testing of protocol adapters.

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
- **`CORPUS_SCHEMAS_ROOT`** - Override schemas directory location
- **`CORPUS_SCHEMA_BASE_URL`** - Override schema $id base URL

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
pytest tests/golden -v  # Golden sample tests

# Custom adapter via command line
pytest tests/llm -v --adapter=my_project.adapters:MyLLMAdapter

# Skip schema validation for faster iteration
pytest tests/live -v --skip-schema

# Verbose output for debugging
pytest tests/llm -v --conformance-verbose

# Filter by marker
pytest -m "llm" -v  # Only LLM protocol tests
pytest -m "vector and not batch" -v  # Vector tests excluding batch operations
pytest -m "core" -v  # Core operations only
pytest -m "streaming" -v  # Streaming operations only
pytest -m "not slow" -v  # Skip slow tests
```

## Available Markers

The following pytest markers are defined for filtered test execution:

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
- **`golden`** - Golden wire message validation tests
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

### `test_config` (session-scoped)

Provides the `ConformanceTestConfig` for wire conformance tests:

```python
def test_wire_validation(test_config):
    assert test_config.enable_metrics == True
    assert test_config.skip_schema_validation == False
```

### `session_metrics` (session-scoped)

Provides `ValidationMetrics` for aggregate reporting:

```python
def test_metrics_collection(session_metrics):
    # Metrics are automatically recorded by test_wire_conformance.py
    assert session_metrics.total_runs > 0
```

### `case_registry` (session-scoped)

Provides access to the wire request case registry:

```python
def test_case_coverage(case_registry):
    cases = case_registry.filter(component="llm")
    assert len(cases) > 0
    assert "llm_complete" in case_registry
```

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

### Example Output
```
================================================================================
CORPUS PROTOCOL SUITE - PLATINUM CERTIFIED
🔌 Adapter: tests.mock.mock_llm_adapter:MockLLMAdapter | CORPUS_ENDPOINT: not set | ⚖️ Strict: off

Protocol & Framework Conformance Status (scored / collected):
  ✅ PASS LLM Protocol V1.0: Gold (111/111 scored; 120 collected)
  ✅ PASS Vector Protocol V1.0: Gold (73/73 scored; 80 collected)
  ✅ PASS Graph Protocol V1.0: Gold (68/68 scored; 75 collected)
  ✅ PASS Embedding Protocol V1.0: Gold (75/75 scored; 82 collected)

🎯 Status: Ready for production deployment
⏱️ Completed in 42.3s
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

## Troubleshooting

### Adapter Instantiation Failed
```
❌ Adapter configuration error: Failed to instantiate adapter 'my_module:MyAdapter'
```
**Solution**: Ensure your adapter has a no-arg constructor or set `CORPUS_ENDPOINT`.

### Schema Not Found
```
KeyError: Schema not found: https://corpusos.com/schemas/llm/llm.envelope.request.json
```
**Solution**: Set `CORPUS_SCHEMAS_ROOT` to point to your schemas directory.

### Test Categorization Issues
Tests appearing under "Other (non-CORPUS conformance tests)"
**Solution**: Place tests in correct directory structure or update categorization patterns in `conftest.py`.

### Performance Issues
**Solution**: Use `--skip-schema` for faster iteration during development.

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
      "reference_levels": {"gold": 111, "silver": 89, "development": 56},
      "outcomes": {"passed": 111, "failed": 0, "error": 0, "skipped": 9, "xfailed": 0, "xpassed": 0},
      "collected_total": 120,
      "scored_total": 111,
      "scored_passed": 111,
      "level": "Gold",
      "tests_needed_to_next_level": 0
    }
  },
  "platinum_certified": true,
  "why_not_platinum": {
    "blocked": false,
    "totals": {"failed": 0, "error": 0, "skipped": 36, "xfailed": 0, "xpassed": 0},
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

## Architecture Integration

This `conftest.py` integrates with the complete CORPUS conformance framework:

- **Schema Registry** (`schema_registry.py`) - JSON Schema validation
- **Wire Validators** (`wire_validators.py`) - Structural and semantic validation
- **Wire Cases** (`wire_cases.py`) - Canonical test case registry
- **Conformance Tests** (`test_wire_conformance.py`) - Wire envelope validation

Together, they provide a comprehensive certification system for CORPUS Protocol implementations.

---

*For more details, see the CORPUS Protocol specification and conformance documentation.*
