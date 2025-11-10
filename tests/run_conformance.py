# SPDX-License-Identifier: Apache-2.0
"""
Lightweight module entrypoint for basic conformance testing.

For full-featured testing with per-protocol runs, watch mode, and better UX,
use the corpus-sdk CLI instead.

Usage:
    # Basic run (all protocols)
    python -m tests.run_conformance
    
    # Fast mode (no coverage, skip slow tests)
    python -m tests.run_conformance --fast
    
    # Verbose output
    python -m tests.run_conformance --verbose
    
    # Watch mode (TDD)
    python -m tests.run_conformance --watch
    
    # JSON output (for CI/machines)
    python -m tests.run_conformance --json
    
    # With configuration
    PYTEST_JOBS=4 COV_FAIL_UNDER=90 python -m tests.run_conformance
    
    # Programmatic usage (full features)
    from tests.run_conformance import run_all_tests
    exit_code = run_all_tests(fast=True, verbose=True, protocols=["llm", "vector"])

Environment variables:
    PYTEST_JOBS      - Parallel jobs (default: auto)
    COV_FAIL_UNDER   - Coverage threshold % (default: 80)
    PYTEST_ARGS      - Additional pytest arguments
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest


# Protocol configuration
PROTOCOLS = ["llm", "vector", "graph", "embedding"]


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


def _build_pytest_args(
    protocols: list[str],
    fast_mode: bool = False,
    verbose_mode: bool = False,
    json_mode: bool = False,
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

    # JSON output format
    if json_mode:
        args.extend(["--json-report", "--json-report-file=test_report.json"])

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
    protocol_names = [p.upper() for p in protocols]
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


def _create_test_result(
    success: bool,
    protocols: List[str],
    duration: float,
    json_report: Dict[str, Any],
) -> TestResult:
    """Create structured test results from pytest output."""
    # Extract test counts from JSON report if available
    test_count = json_report.get("summary", {}).get("total", 0)
    failure_count = json_report.get("summary", {}).get("failed", 0)
    
    # Extract coverage from JSON report if available
    coverage = json_report.get("coverage", {}).get("percent_covered")
    
    return TestResult(
        success=success,
        protocols=protocols,
        duration=duration,
        test_count=test_count,
        failure_count=failure_count,
        coverage_percent=coverage,
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
    }
    print(json.dumps(output, indent=2))


def run_all_tests(
    fast: bool = False,
    verbose: bool = False,
    quiet: bool = False,
    json_output: bool = False,
    watch: bool = False,
    protocols: list[str] | None = None,
    pytest_args: list[str] | None = None,
) -> int:
    """
    Run protocol conformance tests programmatically.
    
    Args:
        fast: Skip coverage and slow tests for faster iteration
        verbose: Enable detailed output (-vv)
        quiet: Minimal output (quiet mode)
        json_output: Output results in JSON format
        watch: Run in watch mode (TDD)
        protocols: List of protocols to test (default: all)
        pytest_args: Additional pytest arguments
        
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
    cov_fail_under = os.environ.get("COV_FAIL_UNDER", "80")
    env_pytest_args = os.environ.get("PYTEST_ARGS", "").split()
    
    # Combine environment and provided args
    all_pytest_args = [arg for arg in env_pytest_args if arg]
    if pytest_args:
        all_pytest_args.extend(pytest_args)

    # User feedback (unless quiet or JSON mode)
    if not quiet and not json_output:
        mode = "fast" if fast else "full"
        protocol_names = [p.upper() for p in protocols]
        print(f"🚀 Running Corpus Protocol Conformance Tests ({mode} mode)...")
        print(f"   Protocols: {', '.join(protocol_names)}")
        print(f"   Parallel jobs: {pytest_jobs}")
        if not fast:
            print(f"   Coverage threshold: {cov_fail_under}%")
        if all_pytest_args:
            print(f"   Extra args: {' '.join(all_pytest_args)}")

    # Build pytest arguments
    args = _build_pytest_args(
        protocols=protocols,
        fast_mode=fast,
        verbose_mode=verbose,
        json_mode=json_output,
        pytest_extra_args=all_pytest_args,
        pytest_jobs=pytest_jobs,
        cov_fail_under=cov_fail_under,
    )

    start_time = time.time()
    
    try:
        result = pytest.main(args)
        elapsed = time.time() - start_time
        
        # Parse JSON report if available
        json_report = _parse_json_report() if json_output else {}
        
        # Create structured result
        test_result = _create_test_result(
            success=result == 0,
            protocols=[p.upper() for p in protocols],
            duration=elapsed,
            json_report=json_report,
        )
        
        # Output results
        if json_output:
            _print_json_result(test_result)
        elif not quiet:
            protocol_names = [p.upper() for p in protocols]
            if result == 0:
                print(f"✅ All protocols conformant ({elapsed:.1f}s)")
                print(f"   Protocols: {', '.join(protocol_names)}")
                if test_result.coverage_percent is not None:
                    print(f"   Coverage: {test_result.coverage_percent:.1f}%")
            else:
                print(f"❌ Conformance failures detected ({elapsed:.1f}s)")
                print(f"   Failed protocols: {', '.join(protocol_names)}")
                if test_result.failure_count > 0:
                    print(f"   Failures: {test_result.failure_count} tests")
        else:
            # Quiet mode: just success/failure
            if result == 0:
                print("✅ All protocols conformant")
            else:
                print("❌ Conformance failures")
            
        return result
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        if json_output:
            error_result = TestResult(
                success=False,
                protocols=[p.upper() for p in protocols],
                duration=0,
                error_message=error_msg,
            )
            _print_json_result(error_result)
        else:
            print("\nerror: pytest execution failed unexpectedly:", file=sys.stderr)
            print(f"  {error_msg}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    """
    Command line entrypoint for running ALL conformance tests.
    
    Args:
        argv: Command line arguments. Uses sys.argv[1:] if None.
        
    Returns:
        Exit code from pytest
    """
    if argv is None:
        argv = sys.argv[1:]

    # Handle help flag
    if "-h" in argv or "--help" in argv:
        return _print_help()
    
    # Parse simple flags
    fast = "--fast" in argv
    verbose = "--verbose" in argv or "-v" in argv
    watch = "--watch" in argv
    json_output = "--json" in argv
    
    # Filter out only our simple flags from passthrough args
    our_flags = {"--fast", "--verbose", "-v", "--watch", "--json", "-h", "--help"}
    passthrough = [arg for arg in argv if arg not in our_flags]
    
    # The CLI entrypoint always runs all protocols
    # Use corpus-sdk CLI for per-protocol testing
    return run_all_tests(
        fast=fast, 
        verbose=verbose, 
        quiet=False,    # CLI is never quiet by default
        json_output=json_output,
        watch=watch,
        protocols=None, # None means all protocols
        pytest_args=passthrough
    )


if __name__ == "__main__":
    raise SystemExit(main())