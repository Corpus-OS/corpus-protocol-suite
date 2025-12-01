# SPDX-License-Identifier: Apache-2.0
"""
Complete module entrypoint for Corpus Protocol conformance testing.

Provides full protocol coverage with rich reporting, certification levels,
and CI integration. For advanced workflows and watch mode, use the corpus-sdk CLI.

Usage:
    # Basic run (all core protocols)
    python -m tests.run_conformance

    # Fast mode (no coverage, skip slow tests)
    python -m tests.run_conformance --fast

    # Specific protocols only
    python -m tests.run_conformance --protocol llm --protocol vector

    # Schema conformance only
    python -m tests.run_conformance --protocol schema
    python -m tests.run_conformance --protocol golden
    python -m tests.run_conformance verify-schema

    # Wire conformance (envelope validation)
    python -m tests.run_conformance --protocol wire

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


# --------------------------------------------------------------------------- #
# Protocol configuration
# --------------------------------------------------------------------------- #

# Mapping from protocol name -> pytest test path
# NOTE: keep this in sync with corpus_sdk/cli.py
PROTOCOL_PATHS: Dict[str, str] = {
    "llm": "tests/llm",
    "vector": "tests/vector",
    "graph": "tests/graph",
    "embedding": "tests/embedding",
    "schema": "tests/schema",
    "golden": "tests/golden",
    # Wire-level envelope conformance suite
    "wire": "tests/live",
}

# Core protocol set used when no explicit protocols are passed.
PROTOCOLS: List[str] = ["llm", "vector", "graph", "embedding", "schema", "golden"]

# All known protocol-like suites this runner understands.
ALL_PROTOCOLS: List[str] = list(PROTOCOL_PATHS.keys())

PROTOCOL_DISPLAY_NAMES = {
    "llm": "LLM Protocol V1.0",
    "vector": "Vector Protocol V1.0",
    "graph": "Graph Protocol V1.0",
    "embedding": "Embedding Protocol V1.0",
    "schema": "Schema Conformance",
    "golden": "Golden Wire Validation",
    "wire": "Wire Envelope Conformance",
}

# Reference conformance levels (used for reporting only).
# Scoring is computed dynamically from collected tests.
CONFORMANCE_LEVELS: Dict[str, Dict[str, int]] = {
    # Reference: currently 111 llm tests in the suite
    "llm": {"gold": 111, "silver": 89, "development": 56},
    # Reference: 73 vector tests
    "vector": {"gold": 73, "silver": 58, "development": 36},
    "graph": {"gold": 68, "silver": 54, "development": 34},
    "embedding": {"gold": 75, "silver": 60, "development": 38},
    # 13 schema meta-lint tests
    "schema": {"gold": 13, "silver": 10, "development": 7},
    # 78 golden tests (55 parametrized + 23 standalone)
    "golden": {"gold": 78, "silver": 62, "development": 39},
    # Wire suite is dynamic; we treat its thresholds as "total collected" at runtime.
    # No static entry needed here.
}


@dataclass
class TestResult:
    """Structured test results for JSON output."""
    success: bool
    protocols: List[str]  # canonical protocol IDs (llm, vector, ...)
    duration: float
    test_count: int = 0
    failure_count: int = 0
    error_message: Optional[str] = None
    coverage_percent: Optional[float] = None
    certification_level: Optional[str] = None
    protocol_results: Optional[Dict[str, Dict]] = None


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #

def _validate_protocols(protocols: List[str]) -> bool:
    """Validate that protocol names are known."""
    invalid = [p for p in protocols if p not in ALL_PROTOCOLS]
    if invalid:
        print(f"error: unknown protocols: {', '.join(invalid)}", file=sys.stderr)
        print(f"  available: {', '.join(ALL_PROTOCOLS)}", file=sys.stderr)
        return False
    return True


def _validate_test_dirs(protocols: List[str]) -> bool:
    """Ensure specified protocol test directories exist."""
    if not _validate_protocols(protocols):
        return False

    test_dirs = [PROTOCOL_PATHS[p] for p in protocols]
    missing = [d for d in test_dirs if not os.path.isdir(d)]
    if missing:
        print(f"error: test directories not found: {', '.join(missing)}", file=sys.stderr)
        print("  Please run from repo root with 'tests/' directory", file=sys.stderr)
        return False
    return True


def _validate_environment() -> tuple[bool, str]:
    """Validate test environment for safety checks."""
    issues: List[str] = []

    if not os.getenv("CORPUS_TEST_ENV"):
        issues.append("CORPUS_TEST_ENV not set, using default")

    if os.getenv("CORPUS_TEST_ENV") == "production":
        return False, "Cannot run full test suite in production. Use quick-check instead."

    return True, "; ".join(issues) if issues else "Environment OK"


def _check_dependencies() -> bool:
    """Check test dependencies are installed."""
    try:
        import corpus_sdk  # noqa: F401
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
        import pytest_jsonreport  # noqa: F401
        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------- #
# Conformance level calculation (dynamic scoring)
# --------------------------------------------------------------------------- #

def _calculate_conformance_level(
    protocol: str,
    passed_count: int,
    total_collected: int,
) -> tuple[str, int]:
    """
    Dynamically calculate level based on percentage of collected tests.

    - Gold: 100% passing
    - Silver: >= 80% passing
    - Development: >= 50% passing
    - Below Development: < 50% passing

    The integer returned is "tests needed to reach the next level" in terms of
    additional passing tests, relative to the collected count.
    """
    if total_collected == 0:
        return "⚠️ No Tests Found", 0

    if passed_count == total_collected:
        # Perfect run
        return "🥇 Gold", 0

    silver_threshold = int(total_collected * 0.80)
    dev_threshold = int(total_collected * 0.50)

    if passed_count >= silver_threshold:
        # On Silver; next stop is Gold (i.e. all tests passing)
        needed = total_collected - passed_count
        return "🥈 Silver", max(0, needed)

    if passed_count >= dev_threshold:
        # On Development; next stop is Silver
        needed = silver_threshold - passed_count
        return "🔬 Development", max(0, needed)

    # Below Development; aim for Development threshold
    needed = dev_threshold - passed_count
    return "❌ Below Development", max(0, needed)


# --------------------------------------------------------------------------- #
# Result parsing helpers
# --------------------------------------------------------------------------- #

def _parse_junit_results() -> Dict[str, Any]:
    """Parse JUnit XML results to get real test data."""
    try:
        import xml.etree.ElementTree as ET
        import glob

        results: Dict[str, Any] = {
            "total_tests": 0,
            "failures": 0,
            "errors": 0,
            "duration": 0.0,
            "protocol_results": {},  # proto -> {passed, failed}
        }

        # Pre-initialize known protocols for convenience
        for proto in ALL_PROTOCOLS:
            results["protocol_results"][proto] = {"passed": 0, "failed": 0}

        # Look for all JUnit XML files we might have created
        xml_files = glob.glob("*_results.xml") + glob.glob("conformance_results.xml")

        for xml_file in xml_files:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()

                # Aggregate overall stats
                results["total_tests"] += int(root.get("tests", 0))
                results["failures"] += int(root.get("failures", 0))
                results["errors"] += int(root.get("errors", 0))
                try:
                    results["duration"] += float(root.get("time", 0))
                except (TypeError, ValueError):
                    pass

                # Parse per-testcase stats
                for testcase in root.findall(".//testcase"):
                    classname = testcase.get("classname", "") or ""
                    file_attr = testcase.get("file", "") or ""
                    context = f"{classname} {file_attr}"

                    # Determine which protocol suite this belongs to
                    proto_for_case: Optional[str] = None
                    for proto in ALL_PROTOCOLS:
                        if proto == "wire":
                            # Wire tests are typically in tests/conformance or tests/live
                            patterns = [
                                "tests.conformance.",
                                "tests/conformance/",
                                "tests.live.",
                                "tests/live/",
                                "test_wire_conformance",
                            ]
                        else:
                            patterns = [
                                f"tests.{proto}.",
                                f"tests/{proto}/",
                            ]

                        if any(pat in context for pat in patterns):
                            proto_for_case = proto
                            break

                    if not proto_for_case:
                        continue  # Not a known protocol suite

                    proto_stats = results["protocol_results"].setdefault(
                        proto_for_case, {"passed": 0, "failed": 0}
                    )

                    if testcase.find("failure") is None and testcase.find("error") is None:
                        proto_stats["passed"] += 1
                    else:
                        proto_stats["failed"] += 1

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

            # Try a couple of common coverage XML layouts
            coverage_elem = root.find(".//coverage")
            if coverage_elem is not None:
                line_rate = coverage_elem.get("line-rate")
                if line_rate:
                    return float(line_rate) * 100.0

            for elem in root.findall(".//*"):
                if elem.get("type") == "line" and "rate" in elem.attrib:
                    return float(elem.get("rate")) * 100.0

    except (ImportError, ET.ParseError, IOError, ValueError):
        pass

    return None


# --------------------------------------------------------------------------- #
# Pytest invocation helpers
# --------------------------------------------------------------------------- #

def _build_pytest_args(
    protocols: List[str],
    fast_mode: bool = False,
    verbose_mode: bool = False,
    json_mode: bool = False,
    junit_mode: bool = True,
    pytest_extra_args: Optional[List[str]] = None,
    pytest_jobs: str = "auto",
    cov_fail_under: str = "80",
) -> List[str]:
    """Build pytest arguments with consistent configuration."""
    if pytest_extra_args is None:
        pytest_extra_args = []

    test_dirs = [PROTOCOL_PATHS[p] for p in protocols]

    args: List[str] = [
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
    elif not json_mode:  # Don't add -v in JSON mode, it just clutters output
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
            "--cov-report=xml:coverage.xml",
        ])
        if not json_mode:
            args.append("--cov-report=html:conformance_coverage_report")

    return args


def _run_watch_mode(
    protocols: List[str],
    passthrough_args: List[str],
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

    # Watch test directories and corpus_sdk itself
    watch_paths = [PROTOCOL_PATHS[p] for p in protocols]
    watch_paths.append("corpus_sdk")

    pytest_args = _build_pytest_args(
        protocols=protocols,
        fast_mode=fast_mode,
        verbose_mode=verbose_mode,
        json_mode=False,
        junit_mode=False,        # No JUnit in watch mode
        pytest_extra_args=passthrough_args,
        pytest_jobs="1",         # Avoid xdist in watch mode
    )

    # Remove the explicit test paths for watch mode; pytest-watch passes them separately.
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
            with open(report_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


# --------------------------------------------------------------------------- #
# Conformance report generation & TestResult assembly
# --------------------------------------------------------------------------- #

def _generate_conformance_report(
    protocols: List[str],
    duration: float,
    success: bool,
    junit_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate comprehensive conformance report with real data."""
    try:
        # Calculate certification levels from real test data
        protocol_results: Dict[str, Dict[str, Any]] = {}
        overall_certification = "❌ Below Development"

        if junit_data and "protocol_results" in junit_data:
            all_gold = True
            any_below_dev = False

            for proto in protocols:
                proto_stats = junit_data["protocol_results"].get(
                    proto, {"passed": 0, "failed": 0}
                )
                passed = proto_stats["passed"]
                failed = proto_stats["failed"]
                total = passed + failed

                level, needed = _calculate_conformance_level(proto, passed, total)

                # For reporting, show the "expected" gold threshold if we have it;
                # otherwise default to the number of collected tests.
                ref_gold = CONFORMANCE_LEVELS.get(proto, {}).get("gold", total)

                protocol_results[proto] = {
                    "passed": passed,
                    "failed": failed,
                    "total": total,
                    "level": level,
                    "needed_for_next": needed,
                    "gold_threshold": ref_gold,
                }

                if "Gold" not in level:
                    all_gold = False
                if "Below Development" in level:
                    any_below_dev = True

            # Determine overall certification
            if all_gold and protocols:
                overall_certification = "🏆 Platinum"
            elif not any_below_dev:
                overall_certification = "🥇 Gold"
            elif any(
                (proto_stats["passed"] >= int((proto_stats["passed"] + proto_stats["failed"]) * 0.5))
                for proto, proto_stats in protocol_results.items()
            ):
                overall_certification = "🔬 Development"

        report: Dict[str, Any] = {
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
        }

        # Stable suite ordering, including wire if present
        ordered_suites = ["schema", "golden", "wire", "llm", "vector", "graph", "embedding"]
        seen = set(protocols) | set(junit_data.get("protocol_results", {}).keys())
        report["test_suites"] = [p for p in ordered_suites if p in seen]

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
            "error": str(e),
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
    protocol_results: Dict[str, Dict[str, Any]] = {}
    certification_level = "❌ Needs Improvement"

    if junit_data and "protocol_results" in junit_data:
        all_gold = True

        for proto in protocols:
            proto_stats = junit_data["protocol_results"].get(
                proto, {"passed": 0, "failed": 0}
            )
            passed = proto_stats["passed"]
            failed = proto_stats["failed"]
            total = passed + failed

            level, needed = _calculate_conformance_level(proto, passed, total)
            protocol_results[proto] = {
                "passed": passed,
                "failed": failed,
                "total": total,
                "level": level,
                "needed_for_next": needed,
            }

            if "Gold" not in level:
                all_gold = False

        certification_level = "🏆 Platinum" if all_gold and protocols else "🥇 Gold"

    return TestResult(
        success=success,
        protocols=protocols,
        duration=duration,
        test_count=test_count,
        failure_count=failure_count,
        coverage_percent=coverage,
        certification_level=certification_level,
        protocol_results=protocol_results or None,
    )


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #

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
        "error_message": result.error_message,
    }
    print(json.dumps(output, indent=2))


def _print_rich_summary(
    result: TestResult,
    protocols: List[str],
    verbose: bool = False,
) -> None:
    """Print rich terminal summary with real certification levels."""
    protocol_names = [PROTOCOL_DISPLAY_NAMES.get(p, p.upper()) for p in protocols]

    if result.success:
        print(f"✅ {result.certification_level} - All selected protocols conformant")
        print(f"   Protocols: {', '.join(protocol_names)}")
        print(f"   Duration: {result.duration:.1f}s")
        if result.coverage_percent is not None:
            print(f"   Coverage: {result.coverage_percent:.1f}%")
        if result.test_count > 0:
            print(f"   Tests: {result.test_count} passed")

        if verbose and result.protocol_results:
            print("\nProtocol Details:")
            for proto, stats in result.protocol_results.items():
                display_name = PROTOCOL_DISPLAY_NAMES.get(proto, proto.upper())
                level = stats.get("level", "❌ Unknown")
                passed = stats.get("passed", 0)
                total = stats.get("total", passed)
                print(f"   - {display_name}: {level} ({passed}/{total} tests)")
    else:
        print("❌ Conformance failures detected")
        print(f"   Protocols: {', '.join(protocol_names)}")
        print(f"   Duration: {result.duration:.1f}s")
        if result.failure_count > 0:
            print(f"   Failures: {result.failure_count} tests")

        if result.protocol_results:
            print("\nProtocol Status:")
            for proto, stats in result.protocol_results.items():
                display_name = PROTOCOL_DISPLAY_NAMES.get(proto, proto.upper())
                level = stats.get("level", "❌ Unknown")
                passed = stats.get("passed", 0)
                failed = stats.get("failed", 0)
                needed = stats.get("needed_for_next", 0)

                if "Gold" in level:
                    print(f"   ✅ {display_name}: {level}")
                elif needed > 0:
                    print(f"   ⚠️  {display_name}: {level} ({needed} tests to next level)")
                else:
                    print(f"   ❌ {display_name}: {level} ({passed} passed, {failed} failed)")


# --------------------------------------------------------------------------- #
# Public runner
# --------------------------------------------------------------------------- #

def run_all_tests(
    fast: bool = False,
    verbose: bool = False,
    quiet: bool = False,
    json_output: bool = False,
    watch: bool = False,
    junit_output: bool = True,
    generate_report: bool = False,
    validate_environment: bool = True,
    protocols: Optional[List[str]] = None,
    pytest_args: Optional[List[str]] = None,
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
        generate_report: Create conformance_report.json
        validate_environment: Perform safety checks
        protocols: List of protocol IDs to test (default: core PROTOCOLS)
        pytest_args: Additional pytest arguments
        cov_threshold: Coverage threshold percentage

    Returns:
        Exit code from pytest (0 = success, non-zero = failure)
    """
    # This file lives in tests/
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.dirname(here)
    os.chdir(repo_root)

    # Use default protocol set if none specified
    if protocols is None:
        protocols = PROTOCOLS

    # Validate environment
    if validate_environment and not fast:
        env_ok, env_msg = _validate_environment()
        if not env_ok:
            print(f"❌ {env_msg}")
            return 1
        if not quiet and env_msg != "Environment OK":
            print(f"ℹ️  Environment: {env_msg}")

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

    # Use environment values if not explicitly overridden
    if cov_threshold == 80:  # Default value; allow env to override
        cov_threshold = int(env_cov_threshold)
    if junit_output:  # Respect environment toggle
        junit_output = env_junit_output

    # Combine environment and provided args
    all_pytest_args = [arg for arg in env_pytest_args if arg]
    if pytest_args:
        all_pytest_args.extend(pytest_args)

    # User feedback (unless quiet or JSON-only)
    if not quiet and not json_output:
        mode = "fast" if fast else "full"
        protocol_names = [PROTOCOL_DISPLAY_NAMES.get(p, p.upper()) for p in protocols]
        print(f"🚀 Running Corpus Protocol Conformance Tests ({mode} mode)...")
        print(f"   Protocols: {', '.join(protocol_names)}")
        print(f"   Parallel jobs: {pytest_jobs}")
        if not fast:
            print(f"   Coverage threshold: {cov_threshold}%")
            print(f"   JUnit output: {'enabled' if junit_output else 'disabled'}")
        if all_pytest_args:
            print(f"   Extra args: {' '.join(all_pytest_args)}")

    # Build pytest arguments
    args = _build_pytest_args(
        protocols=protocols,
        fast_mode=fast,
        verbose_mode=verbose,
        json_mode=json_output,
        junit_mode=junit_output and not fast,
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
        if generate_report:
            _generate_conformance_report(protocols, elapsed, result == 0, junit_data)

        # Create structured result with real data
        test_result = _create_test_result(
            success=result == 0,
            protocols=protocols,
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
                protocols=protocols,
                duration=0.0,
                error_message=error_msg,
            )
            _print_json_result(error_result)
        else:
            print("\n❌ pytest execution failed unexpectedly:", file=sys.stderr)
            print(f"  {error_msg}", file=sys.stderr)
        return 1


# --------------------------------------------------------------------------- #
# Convenience command handlers
# --------------------------------------------------------------------------- #

def _handle_verify_schema() -> int:
    """Handle verify-schema command (schema + golden validation)."""
    print("🔍 Verifying Schema Conformance (schema meta-lint + golden validation)...")

    rc1 = run_all_tests(
        protocols=["schema"],
        quiet=False,
        verbose=False,
        junit_output=True,
        generate_report=False,
    )

    if rc1 != 0:
        return rc1

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
    if env_msg != "Environment OK":
        print(f"ℹ️  Environment: {env_msg}")

    # CI defaults to core protocols only here; wire is handled separately via CLI if desired.
    return run_all_tests(
        fast=False,
        verbose=False,
        quiet=False,
        json_output=False,
        junit_output=True,
        generate_report=True,
        validate_environment=True,
        protocols=None,
        cov_threshold=80,
    )


# --------------------------------------------------------------------------- #
# CLI entrypoint
# --------------------------------------------------------------------------- #

def main(argv: Optional[List[str]] = None) -> int:
    """
    Command line entrypoint for running conformance tests.

    Args:
        argv: Command line arguments. Uses sys.argv[1:] if None.

    Returns:
        Exit code from pytest
    """
    if argv is None:
        argv = sys.argv[1:]

    # Help
    if not argv or "-h" in argv or "--help" in argv:
        return _print_help()

    # Advanced commands
    if argv[0] in ["verify-schema", "test-schema", "test-golden", "test-ci"]:
        command = argv[0]
        remaining_args = argv[1:]

        if command == "verify-schema":
            return _handle_verify_schema()
        if command == "test-schema":
            return run_all_tests(protocols=["schema"], pytest_args=remaining_args)
        if command == "test-golden":
            return run_all_tests(protocols=["golden"], pytest_args=remaining_args)
        if command == "test-ci":
            return _handle_ci_mode()

    # Simple flags
    fast = "--fast" in argv
    verbose = "--verbose" in argv or "-v" in argv
    watch = "--watch" in argv
    json_output = "--json" in argv
    junit_output = "--junit" in argv
    report = "--report" in argv
    ci_mode = "--ci" in argv

    # Protocol selection
    protocols: Optional[List[str]] = None
    if "--protocol" in argv:
        proto_indexes = [i for i, arg in enumerate(argv) if arg == "--protocol"]
        protocols = []
        for idx in proto_indexes:
            if idx + 1 < len(argv) and not argv[idx + 1].startswith("-"):
                protocols.append(argv[idx + 1])

    # Filter out our own flags from passthrough args
    our_flags = {
        "--fast",
        "--verbose",
        "-v",
        "--watch",
        "--json",
        "--junit",
        "--report",
        "--ci",
        "--protocol",
        "-h",
        "--help",
    }
    passthrough = [
        arg
        for i, arg in enumerate(argv)
        if arg not in our_flags and (i == 0 or argv[i - 1] != "--protocol")
    ]

    # CI mode goes through dedicated handler
    if ci_mode:
        return _handle_ci_mode()

    return run_all_tests(
        fast=fast,
        verbose=verbose,
        quiet=False,
        json_output=json_output,
        watch=watch,
        junit_output=junit_output,
        generate_report=report,
        protocols=protocols,
        pytest_args=passthrough,
    )


if __name__ == "__main__":
    raise SystemExit(main())
