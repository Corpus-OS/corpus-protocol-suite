# SPDX-License-Identifier: Apache-2.0
"""
Complete module entrypoint for Corpus Protocol conformance testing.

Provides full protocol coverage with rich reporting, certification levels,
and CI integration. For advanced workflows and watch mode, use the corpus-sdk CLI.

Usage:
    # Basic run (all protocols)
    python -m tests.run_conformance
    
    # Fast mode (no coverage, skip slow tests)
    python -m tests.run_conformance --fast
    
    # Specific protocols only
    python -m tests.run_conformance --protocol llm --protocol vector
    
    # Schema conformance only
    python -m tests.run_conformance --protocol schema
    python -m tests.run_conformance --protocol golden
    python -m tests.run_conformance --verify-schema
    
    # CI mode with reporting
    python -m tests.run_conformance --ci --report
    
    # Verbose output with JUnit XML
    python -m tests.run_conformance --verbose --junit
    
    # JSON output (for CI/machines)
    python -m tests.run_conformance --json
    
    # Watch mode (TDD)
    python -m tests.run_conformance --watch
    
    # With configuration
    PYTEST_JOBS=4 COV_FAIL_UNDER=90 python -m tests.run_conformance

Environment variables:
    PYTEST_JOBS      - Parallel jobs (default: auto)
    COV_FAIL_UNDER   - Coverage threshold % (default: 80)
    PYTEST_ARGS      - Additional pytest arguments
    JUNIT_OUTPUT     - Generate JUnit XML (default: true)
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest


# Complete protocol configuration
PROTOCOLS = ["llm", "vector", "graph", "embedding", "schema", "golden"]

PROTOCOL_DISPLAY_NAMES = {
    "llm": "LLM Protocol V1.0",
    "vector": "Vector Protocol V1.0", 
    "graph": "Graph Protocol V1.0",
    "embedding": "Embedding Protocol V1.0", 
    "schema": "Schema Conformance",
    "golden": "Golden Wire Validation"
}

CONFORMANCE_LEVELS = {
    "llm": {"gold": 61, "silver": 49, "development": 31},
    "vector": {"gold": 72, "silver": 58, "development": 36},
    "graph": {"gold": 68, "silver": 54, "development": 34},
    "embedding": {"gold": 75, "silver": 60, "development": 38},
    "schema": {"gold": 86, "silver": 69, "development": 43},
    "golden": {"gold": 73, "silver": 58, "development": 37},
}


@dataclass
class TestResult:
    """Structured test results for JSON output."""
    success: bool
    protocols: List[str]
    duration: float
    test_count: int = 0
    failure_count: int = 0
    error_message: Optional[str] = None
    coverage_percent: Optional[float] = None
    certification_level: Optional[str] = None
    protocol_results: Optional[Dict[str, Dict]] = None


def _validate_protocols(protocols: List[str]) -> bool:
    """Validate that protocol names are known."""
    invalid = [p for p in protocols if p not in PROTOCOLS]
    if invalid:
        print(f"error: unknown protocols: {', '.join(invalid)}", file=sys.stderr)
        print(f"  available: {', '.join(PROTOCOLS)}", file=sys.stderr)
        return False
    return True


def _validate_test_dirs(protocols: list[str]) -> bool:
    """Ensure specified protocol test directories exist."""
    if not _validate_protocols(protocols):
        return False
        
    test_dirs = [f"tests/{p}" for p in protocols]
    missing = [d for d in test_dirs if not os.path.isdir(d)]
    if missing:
        print(f"error: test directories not found: {', '.join(missing)}", file=sys.stderr)
        print("  Please run from repo root with 'tests/' directory", file=sys.stderr)
        return False
    return True


def _validate_environment() -> tuple[bool, str]:
    """Validate test environment for safety checks."""
    issues = []
    
    if not os.getenv("CORPUS_TEST_ENV"):
        issues.append("CORPUS_TEST_ENV not set, using default")
    
    if os.getenv("CORPUS_TEST_ENV") == "production":
        return False, "Cannot run full test suite in production. Use quick-check instead."
    
    return True, "; ".join(issues) if issues else "Environment OK"


def _check_dependencies() -> bool:
    """Check test dependencies are installed."""
    try:
        import corpus_sdk
        return True
    except ImportError:
        print(
            "❌ Error: Test dependencies not installed.\n"
            "   Please run: pip install .[test]",
            file=sys.stderr,
        )
        return False


def _check_json_plugin() -> bool:
    """Check if pytest-json-report plugin is available."""
    try:
        import pytest_jsonreport
        return True
    except ImportError:
        return False


def _calculate_conformance_level(protocol: str, passed_count: int) -> tuple[str, int]:
    """Calculate conformance level and progress to next level."""
    levels = CONFORMANCE_LEVELS.get(protocol, {})
    
    if passed_count >= levels.get("gold", 0):
        return "🥇 Gold", 0
    elif passed_count >= levels.get("silver", 0):
        needed = levels.get("gold", 0) - passed_count
        return "🥈 Silver", needed
    elif passed_count >= levels.get("development", 0):
        needed = levels.get("silver", 0) - passed_count
        return "🔬 Development", needed
    else:
        needed = levels.get("development", 0) - passed_count
        return "❌ Below Development", needed


def _parse_junit_results() -> Dict[str, Any]:
    """Parse JUnit XML results to get real test data."""
    try:
        import xml.etree.ElementTree as ET
        import glob
        
        results = {
            "total_tests": 0,
            "failures": 0,
            "errors": 0,
            "duration": 0.0,
            "protocol_results": {}
        }
        
        # Look for all JUnit XML files
        for xml_file in glob.glob("*_results.xml") + glob.glob("conformance_results.xml"):
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # Aggregate overall stats
                results["total_tests"] += int(root.get("tests", 0))
                results["failures"] += int(root.get("failures", 0))
                results["errors"] += int(root.get("errors", 0))
                results["duration"] += float(root.get("time", 0))
                
                # Parse protocol-specific results from test cases
                for testcase in root.findall(".//testcase"):
                    classname = testcase.get("classname", "")
                    # Extract protocol from classname/path
                    for proto in PROTOCOLS:
                        if f"tests/{proto}" in classname or f"test_{proto}" in classname:
                            if proto not in results["protocol_results"]:
                                results["protocol_results"][proto] = {"passed": 0, "failed": 0}
                            
                            if testcase.find("failure") is None and testcase.find("error") is None:
                                results["protocol_results"][proto]["passed"] += 1
                            else:
                                results["protocol_results"][proto]["failed"] += 1
                            break
                            
            except (ET.ParseError, IOError) as e:
                print(f"⚠️  Could not parse {xml_file}: {e}")
                continue
                
        return results
        
    except ImportError:
        # Fallback if XML parsing not available
        return {}


def _parse_coverage_data() -> Optional[float]:
    """Parse coverage percentage from coverage.xml if available."""
    try:
        import xml.etree.ElementTree as ET
        
        if os.path.exists("coverage.xml"):
            tree = ET.parse("coverage.xml")
            root = tree.getroot()
            
            # Try different coverage XML formats
            coverage_elem = root.find(".//coverage")
            if coverage_elem is not None:
                line_rate = coverage_elem.get("line-rate")
                if line_rate:
                    return float(line_rate) * 100
            
            # Alternative format
            for elem in root.findall(".//*"):
                if elem.get("type") == "line" and "rate" in elem.attrib:
                    return float(elem.get("rate")) * 100
                    
    except (ImportError, ET.ParseError, IOError, ValueError):
        pass
        
    return None


def _build_pytest_args(
    protocols: list[str],
    fast_mode: bool = False,
    verbose_mode: bool = False,
    json_mode: bool = False,
    junit_mode: bool = True,
    pytest_extra_args: list[str] | None = None,
    pytest_jobs: str = "auto",
    cov_fail_under: str = "80",
) -> list[str]:
    """Build pytest arguments with consistent configuration."""
    if pytest_extra_args is None:
        pytest_extra_args = []

    test_dirs = [f"tests/{p}" for p in protocols]
    
    args = [
        *test_dirs,
        *pytest_extra_args,
    ]

    # JSON output format (only if plugin available)
    if json_mode and _check_json_plugin():
        args.extend(["--json-report", "--json-report-file=test_report.json"])
    elif json_mode:
        print("⚠️  pytest-json-report plugin not installed, JSON output disabled", file=sys.stderr)

    # JUnit XML output (for CI)
    if junit_mode and not fast_mode:
        args.extend(["--junitxml", "conformance_results.xml"])

    # Verbosity
    if verbose_mode:
        args.append("-vv")
    elif not json_mode:  # Don't use -v in JSON mode (interferes with parsing)
        args.append("-v")

    # Parallel execution
    if pytest_jobs != "1":
        args.extend(["-n", pytest_jobs])

    # Fast mode: skip coverage and slow tests
    if fast_mode:
        args.extend(["-m", "not slow", "--no-cov"])
    else:
        # Coverage configuration
        args.extend([
            "--cov=corpus_sdk",
            f"--cov-fail-under={cov_fail_under}",
            "--cov-report=term",
            "--cov-report=xml:coverage.xml"
        ])
        if not json_mode:  # HTML report only in non-JSON mode
            args.append("--cov-report=html:conformance_coverage_report")

    return args


def _run_watch_mode(
    protocols: list[str],
    passthrough_args: list[str],
    fast_mode: bool = False,
    verbose_mode: bool = False,
) -> int:
    """Run pytest in watch mode for TDD workflows."""
    try:
        import pytest_watch
    except ImportError:
        print(
            "error: pytest-watch is required for watch mode.\n"
            "Install watch dependencies via:\n"
            "    pip install pytest-watch",
            file=sys.stderr,
        )
        return 1

    print("👀 Starting watch mode... (Ctrl+C to stop)")
    protocol_names = [PROTOCOL_DISPLAY_NAMES.get(p, p.upper()) for p in protocols]
    print(f"   Watching: {', '.join(protocol_names)}")
    
    # Build watch paths from protocol directories
    watch_paths = [f"tests/{p}" for p in protocols]
    watch_paths.append("corpus_sdk")  # Also watch source code
    
    # Build pytest args for watch mode
    pytest_args = _build_pytest_args(
        protocols=protocols,
        fast_mode=fast_mode,
        verbose_mode=verbose_mode,
        pytest_extra_args=passthrough_args,
        pytest_jobs="1",  # No parallel in watch mode
        junit_mode=False,  # No JUnit in watch mode
    )
    
    # Remove paths from pytest args (watch handles paths separately)
    pytest_args = [arg for arg in pytest_args if not arg.startswith("tests/")]
    
    try:
        return pytest_watch.watch(
            paths=watch_paths,
            args=pytest_args,
            on_pass=lambda: print("✅ Tests passed"),
            on_fail=lambda: print("❌ Tests failed"),
        )
    except KeyboardInterrupt:
        print("\n👋 Watch mode stopped.")
        return 0


def _parse_json_report() -> Dict[str, Any]:
    """Parse pytest-json-report output if available."""
    report_file = "test_report.json"
    if os.path.exists(report_file):
        try:
            with open(report_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def _generate_conformance_report(protocols: List[str], duration: float, success: bool, junit_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive conformance report with real data."""
    try:
        # Calculate certification levels from real test data
        protocol_results = {}
        overall_certification = "❌ Below Development"
        
        if junit_data and "protocol_results" in junit_data:
            all_gold = True
            any_below_dev = False
            
            for proto in protocols:
                proto_data = junit_data["protocol_results"].get(proto, {"passed": 0, "failed": 0})
                passed = proto_data["passed"]
                level, needed = _calculate_conformance_level(proto, passed)
                
                protocol_results[proto] = {
                    "passed": passed,
                    "failed": proto_data["failed"],
                    "level": level,
                    "needed_for_next": needed,
                    "gold_threshold": CONFORMANCE_LEVELS[proto]["gold"]
                }
                
                if "Gold" not in level:
                    all_gold = False
                if "Below Development" in level:
                    any_below_dev = True
            
            # Determine overall certification
            if all_gold:
                overall_certification = "🏆 Platinum"
            elif not any_below_dev:
                overall_certification = "🥇 Gold"
            elif any(proto_data["passed"] >= CONFORMANCE_LEVELS[proto]["development"] for proto, proto_data in protocol_results.items()):
                overall_certification = "🔬 Development"
        
        report = {
            "protocols": protocols,
            "status": "PASS" if success else "FAIL",
            "certification_level": overall_certification,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "duration_seconds": round(duration, 3),
            "summary": {
                "total_tests": junit_data.get("total_tests", 0),
                "failures": junit_data.get("failures", 0),
                "errors": junit_data.get("errors", 0),
            },
            "protocol_results": protocol_results,
            "environment": os.getenv("CORPUS_TEST_ENV", "default"),
            "test_suites": ["schema", "golden", "llm", "vector", "graph", "embedding"],
        }
        
        # Add coverage data if available
        coverage = _parse_coverage_data()
        if coverage is not None:
            report["coverage_percent"] = round(coverage, 1)
        
        with open("conformance_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        
        return report
        
    except Exception as e:
        print(f"⚠️  Could not generate detailed report: {e}")
        # Fallback to basic report
        return {
            "protocols": protocols,
            "status": "PASS" if success else "FAIL", 
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "duration_seconds": round(duration, 3),
            "error": str(e)
        }


def _create_test_result(
    success: bool,
    protocols: List[str],
    duration: float,
    json_report: Dict[str, Any],
    junit_data: Dict[str, Any],
) -> TestResult:
    """Create structured test results from real test data."""
    # Use JUnit data for accurate counts (fallback to JSON report)
    test_count = junit_data.get("total_tests", json_report.get("summary", {}).get("total", 0))
    failure_count = junit_data.get("failures", json_report.get("summary", {}).get("failed", 0))
    
    # Extract coverage from multiple sources
    coverage = None
    if "coverage" in json_report:
        coverage = json_report["coverage"].get("percent_covered")
    if coverage is None:
        coverage = _parse_coverage_data()
    
    # Calculate certification levels from real protocol results
    protocol_results = {}
    certification_level = "❌ Needs Improvement"
    
    if junit_data and "protocol_results" in junit_data:
        all_gold = True
        
        for proto in protocols:
            proto_data = junit_data["protocol_results"].get(proto, {"passed": 0, "failed": 0})
            passed = proto_data["passed"]
            level, needed = _calculate_conformance_level(proto, passed)
            
            protocol_results[proto] = {
                "passed": passed,
                "failed": proto_data["failed"], 
                "level": level,
                "needed_for_next": needed
            }
            
            if "Gold" not in level:
                all_gold = False
        
        certification_level = "🏆 Platinum" if all_gold else "🥇 Gold"
    
    return TestResult(
        success=success,
        protocols=protocols,
        duration=duration,
        test_count=test_count,
        failure_count=failure_count,
        coverage_percent=coverage,
        certification_level=certification_level,
        protocol_results=protocol_results,
    )


def _print_help() -> int:
    """Print help message and return success."""
    print(__doc__.strip())
    return 0


def _print_json_result(result: TestResult) -> None:
    """Print test results in JSON format."""
    output = {
        "success": result.success,
        "protocols": result.protocols,
        "duration": round(result.duration, 2),
        "test_count": result.test_count,
        "failure_count": result.failure_count,
        "coverage_percent": result.coverage_percent,
        "certification_level": result.certification_level,
        "protocol_results": result.protocol_results,
    }
    print(json.dumps(output, indent=2))


def _print_rich_summary(result: TestResult, protocols: List[str], verbose: bool = False) -> None:
    """Print rich terminal summary with real certification levels."""
    protocol_names = [PROTOCOL_DISPLAY_NAMES.get(p, p.upper()) for p in protocols]
    
    if result.success:
        print(f"✅ {result.certification_level} - All protocols conformant")
        print(f"   Protocols: {', '.join(protocol_names)}")
        print(f"   Duration: {result.duration:.1f}s")
        if result.coverage_percent is not None:
            print(f"   Coverage: {result.coverage_percent:.1f}%")
        if result.test_count > 0:
            print(f"   Tests: {result.test_count} passed")
        
        # Show protocol details in verbose mode
        if verbose and result.protocol_results:
            print("\nProtocol Details:")
            for proto, stats in result.protocol_results.items():
                display_name = PROTOCOL_DISPLAY_NAMES.get(proto, proto.upper())
                level = stats.get('level', '❌ Unknown')
                passed = stats.get('passed', 0)
                gold_threshold = CONFORMANCE_LEVELS[proto]["gold"]
                print(f"   - {display_name}: {level} ({passed}/{gold_threshold} tests)")
    else:
        print(f"❌ Conformance failures detected")
        print(f"   Protocols: {', '.join(protocol_names)}")
        print(f"   Duration: {result.duration:.1f}s")
        if result.failure_count > 0:
            print(f"   Failures: {result.failure_count} tests")
        
        # Show detailed protocol status
        if result.protocol_results:
            print("\nProtocol Status:")
            for proto, stats in result.protocol_results.items():
                display_name = PROTOCOL_DISPLAY_NAMES.get(proto, proto.upper())
                level = stats.get('level', '❌ Unknown')
                passed = stats.get('passed', 0)
                failed = stats.get('failed', 0)
                needed = stats.get('needed_for_next', 0)
                
                if "Gold" in level:
                    print(f"   ✅ {display_name}: {level}")
                elif needed > 0:
                    print(f"   ⚠️  {display_name}: {level} ({needed} tests to next level)")
                else:
                    print(f"   ❌ {display_name}: {level} ({passed} passed, {failed} failed)")


def run_all_tests(
    fast: bool = False,
    verbose: bool = False,
    quiet: bool = False,
    json_output: bool = False,
    watch: bool = False,
    junit_output: bool = True,
    generate_report: bool = False,
    validate_environment: bool = True,
    protocols: list[str] | None = None,
    pytest_args: list[str] | None = None,
    cov_threshold: int = 80,
) -> int:
    """
    Run protocol conformance tests programmatically.
    
    Args:
        fast: Skip coverage and slow tests for faster iteration
        verbose: Enable detailed output (-vv)
        quiet: Minimal output (quiet mode)
        json_output: Output results in JSON format
        watch: Run in watch mode (TDD)
        junit_output: Generate JUnit XML reports
        generate_report: Create conformance report
        validate_environment: Perform safety checks
        protocols: List of protocols to test (default: all)
        pytest_args: Additional pytest arguments
        cov_threshold: Coverage threshold percentage
        
    Returns:
        Exit code from pytest (0 = success, non-zero = failure)
    """
    # This file lives in tests/
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.dirname(here)
    os.chdir(repo_root)

    # Use all protocols if none specified
    if protocols is None:
        protocols = PROTOCOLS

    # Validate environment
    if validate_environment and not fast:
        env_ok, env_msg = _validate_environment()
        if not env_ok:
            print(f"❌ {env_msg}")
            return 1

    # Check dependencies
    if not _check_dependencies():
        return 1

    # Check JSON plugin if needed
    if json_output and not _check_json_plugin():
        if not quiet:
            print("❌ pytest-json-report plugin required for JSON output")
            print("   Install with: pip install pytest-json-report")
        return 1

    # Validate test directories exist
    if not _validate_test_dirs(protocols):
        return 1

    # Watch mode (handles its own execution)
    if watch:
        return _run_watch_mode(
            protocols=protocols,
            passthrough_args=pytest_args or [],
            fast_mode=fast,
            verbose_mode=verbose,
        )

    # Configuration from environment
    pytest_jobs = os.environ.get("PYTEST_JOBS", "auto")
    env_cov_threshold = os.environ.get("COV_FAIL_UNDER", str(cov_threshold))
    env_pytest_args = os.environ.get("PYTEST_ARGS", "").split()
    env_junit_output = os.environ.get("JUNIT_OUTPUT", "true").lower() == "true"
    
    # Use environment values if not explicitly set
    if cov_threshold == 80:  # Default value
        cov_threshold = int(env_cov_threshold)
    if junit_output:  # Default is True, respect environment
        junit_output = env_junit_output
    
    # Combine environment and provided args
    all_pytest_args = [arg for arg in env_pytest_args if arg]
    if pytest_args:
        all_pytest_args.extend(pytest_args)

    # User feedback (unless quiet or JSON mode)
    if not quiet and not json_output:
        mode = "fast" if fast else "full"
        protocol_names = [PROTOCOL_DISPLAY_NAMES.get(p, p.upper()) for p in protocols]
        print(f"🚀 Running Corpus Protocol Conformance Tests ({mode} mode)...")
        print(f"   Protocols: {', '.join(protocol_names)}")
        print(f"   Parallel jobs: {pytest_jobs}")
        if not fast:
            print(f"   Coverage threshold: {cov_threshold}%")
            if junit_output:
                print(f"   JUnit output: enabled")
        if all_pytest_args:
            print(f"   Extra args: {' '.join(all_pytest_args)}")

    # Build pytest arguments
    args = _build_pytest_args(
        protocols=protocols,
        fast_mode=fast,
        verbose_mode=verbose,
        json_mode=json_output,
        junit_mode=junit_output and not fast,  # No JUnit in fast mode
        pytest_extra_args=all_pytest_args,
        pytest_jobs=pytest_jobs,
        cov_fail_under=str(cov_threshold),
    )

    start_time = time.time()
    
    try:
        result = pytest.main(args)
        elapsed = time.time() - start_time
        
        # Parse results from multiple sources
        json_report = _parse_json_report() if json_output else {}
        junit_data = _parse_junit_results() if junit_output and not fast else {}
        
        # Generate conformance report if requested
        report_data = None
        if generate_report:
            report_data = _generate_conformance_report(protocols, elapsed, result == 0, junit_data)
        
        # Create structured result with real data
        test_result = _create_test_result(
            success=result == 0,
            protocols=[PROTOCOL_DISPLAY_NAMES.get(p, p.upper()) for p in protocols],
            duration=elapsed,
            json_report=json_report,
            junit_data=junit_data,
        )
        
        # Output results
        if json_output:
            _print_json_result(test_result)
        elif not quiet:
            _print_rich_summary(test_result, protocols, verbose)
            
        return result
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        if json_output:
            error_result = TestResult(
                success=False,
                protocols=[PROTOCOL_DISPLAY_NAMES.get(p, p.upper()) for p in protocols],
                duration=0,
                error_message=error_msg,
            )
            _print_json_result(error_result)
        else:
            print("\n❌ pytest execution failed unexpectedly:", file=sys.stderr)
            print(f"  {error_msg}", file=sys.stderr)
        return 1


def _handle_verify_schema() -> int:
    """Handle verify-schema command (schema + golden validation)."""
    print("🔍 Verifying Schema Conformance (schema meta-lint + golden validation)...")
    
    # Run schema tests
    rc1 = run_all_tests(
        protocols=["schema"],
        quiet=False,
        verbose=False,
        junit_output=True,
        generate_report=False,
    )
    
    if rc1 != 0:
        return rc1
    
    # Run golden tests  
    rc2 = run_all_tests(
        protocols=["golden"],
        quiet=False,
        verbose=False,
        junit_output=True,
        generate_report=False,
    )
    
    return rc2 if rc2 != 0 else 0


def _handle_ci_mode() -> int:
    """Handle CI pipeline mode."""
    print("🏗️  Running CI-optimized conformance suite...")
    
    if not _check_dependencies():
        return 1
    
    env_ok, env_msg = _validate_environment()
    if not env_ok:
        print(f"❌ {env_msg}")
        return 1
    
    return run_all_tests(
        fast=False,
        verbose=False,  # CI doesn't need verbose
        quiet=False,
        json_output=False,
        junit_output=True,
        generate_report=True,
        validate_environment=True,
        protocols=None,  # All protocols
        cov_threshold=80,
    )


def main(argv: list[str] | None = None) -> int:
    """
    Command line entrypoint for running conformance tests.
    
    Args:
        argv: Command line arguments. Uses sys.argv[1:] if None.
        
    Returns:
        Exit code from pytest
    """
    if argv is None:
        argv = sys.argv[1:]

    # Handle help flag
    if not argv or "-h" in argv or "--help" in argv:
        return _print_help()
    
    # Handle advanced commands
    if argv and argv[0] in ["verify-schema", "test-schema", "test-golden", "test-ci"]:
        command = argv[0]
        remaining_args = argv[1:]
        
        if command == "verify-schema":
            return _handle_verify_schema()
        elif command == "test-schema":
            return run_all_tests(protocols=["schema"], pytest_args=remaining_args)
        elif command == "test-golden":
            return run_all_tests(protocols=["golden"], pytest_args=remaining_args)
        elif command == "test-ci":
            return _handle_ci_mode()
    
    # Parse simple flags
    fast = "--fast" in argv
    verbose = "--verbose" in argv or "-v" in argv
    watch = "--watch" in argv
    json_output = "--json" in argv
    junit_output = "--junit" in argv
    report = "--report" in argv
    ci_mode = "--ci" in argv
    
    # Protocol selection
    protocols = None
    if "--protocol" in argv:
        proto_indexes = [i for i, arg in enumerate(argv) if arg == "--protocol"]
        protocols = []
        for idx in proto_indexes:
            if idx + 1 < len(argv) and not argv[idx + 1].startswith("-"):
                protocols.append(argv[idx + 1])
    
    # Filter out only our flags from passthrough args
    our_flags = {
        "--fast", "--verbose", "-v", "--watch", "--json", "--junit", 
        "--report", "--ci", "--protocol", "-h", "--help"
    }
    passthrough = [arg for i, arg in enumerate(argv) 
                  if arg not in our_flags and (i == 0 or argv[i-1] != "--protocol")]
    
    # Handle CI mode
    if ci_mode:
        return _handle_ci_mode()
    
    return run_all_tests(
        fast=fast, 
        verbose=verbose, 
        quiet=False,    # CLI is never quiet by default
        json_output=json_output,
        watch=watch,
        junit_output=junit_output,
        generate_report=report,
        protocols=protocols,
        pytest_args=passthrough
    )


if __name__ == "__main__":
    raise SystemExit(main())