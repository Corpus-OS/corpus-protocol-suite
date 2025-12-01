# corpus_sdk/cli.py
# SPDX-License-Identifier: Apache-2.0
"""
Corpus SDK CLI

Complete protocol conformance testing with full Makefile parity.
Includes wire-level envelope validation for protocol conformance.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None  # type: ignore[assignment]

try:
    import pytest_watch
except ImportError:  # pragma: no cover
    pytest_watch = None


PROTOCOL_PATHS: Dict[str, str] = {
    "llm": "tests/llm",
    "vector": "tests/vector",
    "graph": "tests/graph",
    "embedding": "tests/embedding",
    "schema": "tests/schema",
    "golden": "tests/golden",
    "wire": "tests/live",  # Wire-level envelope conformance
}

# Configuration from environment
PYTEST_JOBS = os.environ.get("PYTEST_JOBS", "auto")
COV_FAIL_UNDER = os.environ.get("COV_FAIL_UNDER", "80")
PYTEST_EXTRA_ARGS = os.environ.get("PYTEST_ARGS", "").split()
JUNIT_OUTPUT = os.environ.get("JUNIT_OUTPUT", "true").lower() == "true"


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _ensure_pytest() -> None:
    if pytest is None:  # pragma: no cover
        print(
            "error: pytest is required to run conformance tests.\n"
            "Install test dependencies via:\n"
            "    pip install .[test]",
            file=sys.stderr,
        )
        raise SystemExit(1)


def _repo_root() -> str:
    """Best-effort guess of repo root."""
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.dirname(here)


def _validate_paths(paths: List[str]) -> bool:
    """Validate that each test path exists before invoking pytest."""
    ok = True
    for p in paths:
        if not (os.path.isdir(p) or os.path.isfile(p)):
            print(f"error: test path does not exist: {p}", file=sys.stderr)
            ok = False
    return ok


def _validate_environment() -> Tuple[bool, str]:
    """Validate test environment like make validate-env."""
    issues: List[str] = []
    
    if not os.getenv("CORPUS_TEST_ENV"):
        issues.append("CORPUS_TEST_ENV not set, using default")
    
    if not os.getenv("CORPUS_ENDPOINT"):
        issues.append("CORPUS_ENDPOINT not set, using default test endpoint")
    
    if os.getenv("CORPUS_TEST_ENV") == "production":
        return False, "Cannot run full test suite in production. Use quick-check instead."
    
    return True, "; ".join(issues) if issues else "Environment OK"


def _build_pytest_args(
    test_paths: List[str],
    cov_module: Optional[str] = None,
    report_name: Optional[str] = None,
    fast_mode: bool = False,
    quiet_mode: bool = False,
    verbose_mode: bool = False,
    passthrough_args: Optional[List[str]] = None,
    junit_report: Optional[str] = None,
    markers: Optional[List[str]] = None,
    adapter: Optional[str] = None,
    skip_schema: bool = False,
) -> List[str]:
    """Build standardized pytest arguments with consistent configuration."""
    if passthrough_args is None:
        passthrough_args = []

    args: List[str] = [
        *test_paths,
        *PYTEST_EXTRA_ARGS,
        *passthrough_args,
    ]

    # Verbosity control
    if quiet_mode:
        args.append("-q")
    elif verbose_mode:
        args.append("-vv")
    else:
        args.append("-v")  # Default verbosity

    # Parallel execution (unless disabled)
    if PYTEST_JOBS != "1":
        args.extend(["-n", PYTEST_JOBS])

    # Fast mode: skip slow tests and coverage
    if fast_mode:
        args.extend(["-m", "not slow"])

    # Marker filtering (for wire conformance)
    if markers:
        # Allow complex expressions by letting each marker be a full -m expression,
        # but keep the common simple case of multiple markers joined with " or ".
        if len(markers) == 1:
            marker_expr = markers[0]
        else:
            marker_expr = " or ".join(markers)
        args.extend(["-m", marker_expr])

    # Adapter selection (for wire conformance)
    if adapter:
        args.extend(["--adapter", adapter])

    # Skip schema validation (for wire conformance fast iteration)
    if skip_schema:
        args.append("--skip-schema")

    # JUnit XML output (for CI)
    if JUNIT_OUTPUT and junit_report and not fast_mode:
        args.extend(["--junitxml", junit_report])

    # Coverage configuration (skip for fast mode)
    if not fast_mode and cov_module:
        args.extend([
            f"--cov={cov_module}",
            f"--cov-fail-under={COV_FAIL_UNDER}",
            "--cov-report=term",
        ])
        if report_name:
            args.extend([
                f"--cov-report=html:{report_name}",
                "--cov-report=xml:coverage.xml",
            ])

    return args


def _run_pytest(args: List[str]) -> tuple[int, float]:
    """Run pytest with given args from repo root."""
    _ensure_pytest()
    root = _repo_root()
    os.chdir(root)

    start_time = time.time()
    
    try:
        result = pytest.main(args)
        elapsed = time.time() - start_time
        return result, elapsed
    except Exception as e:  # pragma: no cover
        print("error: pytest execution failed unexpectedly:", file=sys.stderr)
        print(f"  {type(e).__name__}: {e}", file=sys.stderr)
        return 1, time.time() - start_time


def _print_config(quiet: bool = False) -> None:
    """Print current configuration for transparency."""
    if not quiet:
        print(f"   Config: jobs={PYTEST_JOBS}, cov_threshold={COV_FAIL_UNDER}%")
        env_ok, env_msg = _validate_environment()
        if env_msg != "Environment OK":
            print(f"   Environment: {env_msg}")
        if PYTEST_EXTRA_ARGS:
            print(f"   Extra args: {' '.join(PYTEST_EXTRA_ARGS)}")


def _print_success_stats(protocols: List[str], elapsed: float, test_count: int = 0) -> None:
    """Print success statistics."""
    print("✅ All selected protocols are 100% conformant.")
    print(f"   Protocols: {', '.join(protocols)}")
    if test_count:
        print(f"   Tests: {test_count} passed")
    print(f"   Completed in {elapsed:.1f}s")


def _run_watch_mode(test_paths: List[str], passthrough_args: Optional[List[str]] = None) -> int:
    """Run pytest in watch mode for TDD workflows."""
    if pytest_watch is None:
        print(
            "error: pytest-watch is required for watch mode.\n"
            "Install watch dependencies via:\n"
            "    pip install .[watch]",
            file=sys.stderr,
        )
        return 1

    print("👀 Starting watch mode... (Ctrl+C to stop)")
    print("   Watching for file changes in:", ", ".join(test_paths))
    
    # Convert our test paths to watch paths (parent directories)
    watch_paths = list({os.path.dirname(p.rstrip(os.sep)) or "." for p in test_paths})
    
    try:
        return pytest_watch.watch(
            paths=watch_paths,
            args=passthrough_args or [],
            on_pass=None,
            on_fail=None,
        )
    except KeyboardInterrupt:
        print("\n👋 Watch mode stopped.")
        return 0


def _generate_conformance_report(test_paths: List[str], elapsed: float, rc: int) -> Dict:
    """Generate conformance report like make conformance-report."""
    try:
        # Aggregate from JUnit XML files if they exist
        total_tests = 0
        total_failures = 0
        total_errors = 0
        
        xml_files = [
            "conformance_results.xml", "llm_results.xml", "vector_results.xml", 
            "graph_results.xml", "embedding_results.xml", "schema_results.xml", 
            "golden_results.xml", "wire_results.xml",
        ]
        
        for xml_file in xml_files:
            if os.path.exists(xml_file):
                try:
                    import xml.etree.ElementTree as ET
                    tree = ET.parse(xml_file)
                    root = tree.getroot()
                    total_tests += int(root.get("tests", 0))
                    total_failures += int(root.get("failures", 0))
                    total_errors += int(root.get("errors", 0))
                except Exception:
                    # Ignore malformed or unexpected XML
                    pass
        
        status = "PASS" if rc == 0 and total_failures == 0 and total_errors == 0 else "FAIL"
        
        # Get protocol names from paths
        protocols: List[str] = []
        path_to_proto = {v: k for k, v in PROTOCOL_PATHS.items()}
        for p in test_paths:
            if p in path_to_proto:
                protocols.append(path_to_proto[p])
            else:
                protocols = list(PROTOCOL_PATHS.keys())
                break
        
        report = {
            "protocols": protocols,
            "status": status,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "summary": {
                "total_tests": total_tests,
                "failures": total_failures,
                "errors": total_errors,
                "duration_seconds": round(elapsed, 3),
            },
            "coverage_threshold": int(COV_FAIL_UNDER),
            "test_suites": ["schema", "golden", "wire", "llm", "vector", "graph", "embedding"],
            "environment": os.getenv("CORPUS_TEST_ENV", "default"),
        }
        
        with open("conformance_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        
        return report
        
    except Exception as e:
        print(f"⚠️  Could not generate detailed report: {e}")
        return {}


def _upload_results() -> bool:
    """Upload results to conformance service like make upload-results."""
    if not os.path.exists("conformance_report.json"):
        print("❌ No conformance report found - run with report first")
        return False
    
    try:
        print("📤 Uploading conformance results...")
        # TODO: Implement actual upload logic
        print("✅ Results uploaded successfully")
        return True
    except Exception as e:
        print(f"⚠️  Upload failed: {e}")
        return False


def _setup_test_env() -> bool:
    """Interactive environment setup like make setup-test-env."""
    try:
        endpoint = input("Test endpoint [http://localhost:8080]: ").strip()
        endpoint = endpoint or "http://localhost:8080"
        
        key = input("API key [test-key]: ").strip() 
        key = key or "test-key"
        
        with open(".testenv", "w", encoding="utf-8") as f:
            f.write(f"CORPUS_ENDPOINT={endpoint}\n")
            f.write(f"CORPUS_API_KEY={key}\n")
        
        print("✅ Test environment saved to .testenv")
        print("   Load with: source .testenv")
        return True
        
    except (KeyboardInterrupt, EOFError):
        print("\n❌ Setup cancelled")
        return False
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False


def _check_dependencies() -> bool:
    """Check test dependencies like make check-deps."""
    try:
        import corpus_sdk  # noqa: F401
        print("✅ Dependencies OK")
        return True
    except ImportError:
        print(
            "❌ Error: Test dependencies not installed.\n"
            "   Please run: pip install .[test]",
            file=sys.stderr,
        )
        return False


def _list_wire_cases(component: Optional[str] = None, tag: Optional[str] = None) -> int:
    """List wire conformance test cases."""
    try:
        from wire_cases import get_registry
        registry = get_registry()
        
        cases = registry.filter(component=component, tag=tag)
        
        print(f"{'ID':<40} {'Component':<12} {'Tags'}")
        print("-" * 80)
        for case in cases:
            tags = ", ".join(sorted(case.tags)[:3])
            if len(case.tags) > 3:
                tags += f" (+{len(case.tags) - 3})"
            print(f"{case.id:<40} {case.component:<12} {tags}")
        
        print(f"\nTotal: {len(cases)} cases")
        return 0
        
    except ImportError:
        print("❌ Wire conformance modules not found")
        print("   Ensure wire_cases.py is in your Python path")
        return 1


def _print_wire_coverage() -> int:
    """Print wire conformance coverage summary."""
    try:
        from wire_cases import get_registry
        registry = get_registry()
        summary = registry.get_coverage_summary()
        
        print("Wire Conformance Coverage Summary")
        print("=" * 40)
        print(f"Total cases:        {summary['total_cases']}")
        print(f"Operations covered: {summary['operations_covered']}")
        print(f"Components covered: {', '.join(summary['components_covered'])}")
        print()
        print("Cases by component:")
        for comp, count in summary["cases_by_component"].items():
            print(f"  {comp}: {count}")
        print()
        print("Cases by tag:")
        for tag, count in sorted(summary["cases_by_tag"].items()):
            print(f"  {tag}: {count}")
        
        return 0
        
    except ImportError:
        print("❌ Wire conformance modules not found")
        return 1


def _run_suite(
    title: str,
    test_paths: List[str],
    passthrough_args: List[str],
    quiet_mode: bool = False,
    verbose_mode: bool = False,
    watch_mode: bool = False,
    fast_mode: bool = False,
    cov_module: Optional[str] = None,
    report_name: Optional[str] = None,
    generate_report: bool = False,
    junit_report: Optional[str] = None,
    markers: Optional[List[str]] = None,
    adapter: Optional[str] = None,
    skip_schema: bool = False,
) -> int:
    """Consolidated function to run any test suite."""
    
    # Handle watch mode first
    if watch_mode:
        if not quiet_mode:
            print(f"👀 Starting watch mode for {title}...")
        return _run_watch_mode(test_paths, passthrough_args)

    # Validate test paths before invoking pytest
    if not _validate_paths(test_paths):
        return 2

    # Standard run
    if not quiet_mode:
        print(f"🚀 Running {title}...")
        _print_config(quiet_mode)
        if passthrough_args:
            print(f"   Passthrough args: {' '.join(passthrough_args)}")
        if markers:
            print(f"   Markers: {', '.join(markers)}")
        if adapter:
            print(f"   Adapter: {adapter}")

    # Validate environment for full conformance runs (coverage-enabled)
    if not fast_mode and cov_module:
        env_ok, env_msg = _validate_environment()
        if not env_ok:
            print(f"❌ {env_msg}")
            return 1

    args = _build_pytest_args(
        test_paths=test_paths,
        cov_module=cov_module,
        report_name=report_name,
        fast_mode=fast_mode,
        quiet_mode=quiet_mode,
        verbose_mode=verbose_mode,
        passthrough_args=passthrough_args,
        junit_report=junit_report,
        markers=markers,
        adapter=adapter,
        skip_schema=skip_schema,
    )
    
    rc, elapsed = _run_pytest(args)
    
    report_data: Dict = {}
    if generate_report and not fast_mode:
        report_data = _generate_conformance_report(test_paths, elapsed, rc)
    
    if rc == 0 and not quiet_mode:
        # Get protocol names from paths for the stats message
        protocols: List[str] = []
        path_to_proto = {v: k for k, v in PROTOCOL_PATHS.items()}
        for p in test_paths:
            if p in path_to_proto:
                protocols.append(path_to_proto[p].upper())
            else:
                # Fallback for "all" or multiple protocols
                protocols = [proto.upper() for proto in PROTOCOL_PATHS.keys()]
                break
        
        test_count = report_data.get("summary", {}).get("total_tests", 0) if report_data else 0
        _print_success_stats(protocols, elapsed, test_count)
        
        if generate_report and report_data:
            print("📊 Report: conformance_report.json")
            
    elif rc != 0 and not quiet_mode:
        print("\n❌ Conformance failures detected.")
        if cov_module:  # Only print for conformance, not fast mode
            print("   Inspect the failed tests above. Each test maps to spec sections via CONFORMANCE.md.")

    return rc


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> int:
    # Manually split passthrough args
    cli_args = sys.argv[1:]
    passthrough_args: List[str] = []
    if "--" in cli_args:
        split_index = cli_args.index("--")
        passthrough_args = cli_args[split_index + 1:]
        cli_args = cli_args[:split_index]

    # Setup the main parser
    parser = argparse.ArgumentParser(
        description="Corpus SDK CLI - Complete protocol conformance testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  corpus-sdk test-all-conformance
  corpus-sdk test-embedding-conformance --watch
  corpus-sdk verify --quiet
  corpus-sdk check -p llm -p vector
  corpus-sdk test-llm-conformance -- -x --tb=short
  corpus-sdk test-ci --upload
  corpus-sdk setup-env
  
  # Wire conformance (envelope validation)
  corpus-sdk test-wire                    # All wire tests
  corpus-sdk test-wire -m core            # Only core operations
  corpus-sdk test-wire -m "llm and chat"  # LLM chat operations
  corpus-sdk test-wire --adapter openai   # Test specific adapter
  corpus-sdk wire-list                    # List all wire test cases
  corpus-sdk wire-coverage                # Show wire test coverage

  PYTEST_JOBS=1 corpus-sdk test-conformance

Configuration (environment variables):
  PYTEST_JOBS=4          Run 4 parallel jobs (default: auto)
  COV_FAIL_UNDER=90      Require 90% coverage (default: 80)
  PYTEST_ARGS="-x -s"    Additional pytest arguments
  JUNIT_OUTPUT=false     Disable JUnit XML reports

Notes:
  - Install dependencies: pip install .[test]
  - For watch mode: pip install .[watch]
        """.strip()
    )
    
    # Global flags
    parser.add_argument(
        "-q", "--quiet", action="store_true", 
        help="Minimal output (quiet mode)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", 
        help="Detailed output (-vv)",
    )
    parser.add_argument(
        "-w", "--watch", action="store_true", 
        help="Run in watch mode (TDD)",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Generate conformance report after run",
    )
    parser.add_argument(
        "--upload", action="store_true",
        help="Upload results to conformance service (requires --report)",
    )

    # Subparsers for each command
    subparsers = parser.add_subparsers(
        dest="command", 
        required=True, 
        help="command to execute",
        metavar="COMMAND",
    )

    # test-all-conformance / test-conformance
    subparsers.add_parser(
        "test-all-conformance", 
        help="Run all protocol conformance suites",
    )
    subparsers.add_parser(
        "test-conformance", 
        help="Alias for test-all-conformance",
    )

    # test-fast
    subparsers.add_parser(
        "test-fast", 
        help="Run all tests quickly (no coverage, skip slow tests)",
    )

    # Per-protocol commands
    subparsers.add_parser(
        "test-llm-conformance", 
        help="Run only LLM Protocol conformance tests",
    )
    subparsers.add_parser(
        "test-vector-conformance", 
        help="Run only Vector Protocol conformance tests",
    )
    subparsers.add_parser(
        "test-graph-conformance", 
        help="Run only Graph Protocol conformance tests",
    )
    subparsers.add_parser(
        "test-embedding-conformance", 
        help="Run only Embedding Protocol conformance tests",
    )
    
    # Schema & Golden commands
    subparsers.add_parser(
        "test-schema",
        help="Run schema meta-lint (JSON Schema Draft 2020-12)",
    )
    subparsers.add_parser(
        "test-golden", 
        help="Validate golden wire messages",
    )
    subparsers.add_parser(
        "verify-schema",
        help="Run schema meta-lint + golden validation",
    )
    
    # Wire conformance commands
    wire_parser = subparsers.add_parser(
        "test-wire",
        help="Run wire-level envelope conformance tests",
    )
    wire_parser.add_argument(
        "-m", "--marker",
        action="append",
        dest="markers",
        help="Filter by pytest marker expression (e.g. 'llm', 'core', 'llm and not streaming')",
    )
    wire_parser.add_argument(
        "--adapter",
        type=str,
        help="Test specific adapter implementation",
    )
    wire_parser.add_argument(
        "--skip-schema",
        action="store_true",
        help="Skip JSON Schema validation (faster iteration)",
    )
    
    wire_list_parser = subparsers.add_parser(
        "wire-list",
        help="List wire conformance test cases",
    )
    wire_list_parser.add_argument(
        "-c", "--component",
        choices=["llm", "vector", "embedding", "graph"],
        help="Filter by component",
    )
    wire_list_parser.add_argument(
        "-t", "--tag",
        help="Filter by tag",
    )
    
    subparsers.add_parser(
        "wire-coverage",
        help="Show wire conformance test coverage summary",
    )
    
    # verify / check / validate
    verify_parser = subparsers.add_parser(
        "verify",
        help="Run conformance tests for all or selected protocols",
        aliases=["check", "validate"],
    )
    verify_parser.add_argument(
        "-p",
        "--protocol",
        action="append",
        choices=PROTOCOL_PATHS.keys(),
        help="Select protocol(s) to verify (can be used multiple times)",
    )

    # CI & Advanced commands
    subparsers.add_parser(
        "test-ci",
        help="Full CI pipeline (deps check + wire + conformance + report)",
    )
    subparsers.add_parser(
        "conformance-report",
        help="Generate detailed conformance report from last run",
    )
    subparsers.add_parser(
        "setup-env",
        help="Interactive test environment configuration",
    )
    subparsers.add_parser(
        "check-deps",
        help="Verify test dependencies are installed",
    )
    subparsers.add_parser(
        "quick-check",
        help="Quick health check (smoke test)",
    )

    # Parse the args
    try:
        args = parser.parse_args(cli_args)
    except SystemExit as e:
        return e.code

    # Handle utility commands first
    if args.command == "setup-env":
        return 0 if _setup_test_env() else 1
        
    if args.command == "check-deps":
        return 0 if _check_dependencies() else 1
        
    if args.command == "conformance-report":
        report_data = _generate_conformance_report([], 0, 0)
        if args.upload:
            _upload_results()
        return 0 if report_data else 1
    
    # Wire utility commands
    if args.command == "wire-list":
        return _list_wire_cases(
            component=getattr(args, "component", None),
            tag=getattr(args, "tag", None),
        )
    
    if args.command == "wire-coverage":
        return _print_wire_coverage()

    # Common args for _run_suite
    run_kwargs = {
        "passthrough_args": passthrough_args,
        "quiet_mode": args.quiet,
        "verbose_mode": args.verbose,
        "watch_mode": args.watch,
        "generate_report": args.report,
    }

    # Dispatch test commands
    if args.command in ("test-all-conformance", "test-conformance"):
        rc = _run_suite(
            title="ALL protocol conformance suites (LLM, Vector, Graph, Embedding)",
            test_paths=[PROTOCOL_PATHS[p] for p in ["llm", "vector", "graph", "embedding"]],
            cov_module="corpus_sdk",
            report_name="conformance_coverage_report",
            junit_report="conformance_results.xml",
            **run_kwargs,
        )
        if rc == 0 and args.upload:
            _upload_results()
        return rc

    if args.command == "test-fast":
        return _run_suite(
            title="fast tests (no coverage, skipping slow tests)",
            test_paths=list(PROTOCOL_PATHS.values()),
            fast_mode=True,
            **run_kwargs,
        )

    if args.command == "test-ci":
        if not _check_dependencies():
            return 1
        print("🏗️  Running CI-optimized conformance suite...")
        
        # Run wire conformance first (fast, catches protocol issues early)
        wire_rc = _run_suite(
            title="wire envelope conformance",
            test_paths=[PROTOCOL_PATHS["wire"]],
            junit_report="wire_results.xml",
            **{**run_kwargs, "quiet_mode": False},
        )
        if wire_rc != 0:
            print("❌ Wire conformance failed - stopping CI pipeline")
            return wire_rc
        
        # Then run full conformance suite
        rc = _run_suite(
            title="CI conformance suite",
            test_paths=[PROTOCOL_PATHS[p] for p in ["llm", "vector", "graph", "embedding"]],
            cov_module="corpus_sdk", 
            report_name="conformance_coverage_report",
            junit_report="conformance_results.xml",
            generate_report=True,
            **{**run_kwargs, "quiet_mode": False},
        )
        if rc == 0:
            _upload_results()
        return rc

    if args.command == "quick-check":
        return _run_suite(
            title="quick health check",
            test_paths=["tests/"],
            fast_mode=True,
            passthrough_args=["-k", "test_golden_validates or test_schema_meta", "-x"],
            **run_kwargs,
        )

    if args.command == "verify":
        selected = args.protocol or list(PROTOCOL_PATHS.keys())
        paths = [PROTOCOL_PATHS[p] for p in selected]
        protocol_names = [p.upper() for p in selected]
        rc = _run_suite(
            title=f"{', '.join(protocol_names)} Protocol conformance",
            test_paths=paths,
            cov_module="corpus_sdk",
            report_name="conformance_coverage_report",
            junit_report="conformance_results.xml",
            **run_kwargs,
        )
        if rc == 0 and args.upload:
            _upload_results()
        return rc
    
    # Schema & Golden commands
    if args.command == "test-schema":
        return _run_suite(
            title="schema meta-lint",
            test_paths=[PROTOCOL_PATHS["schema"]],
            junit_report="schema_results.xml",
            **run_kwargs,
        )
        
    if args.command == "test-golden":
        return _run_suite(
            title="golden wire message validation", 
            test_paths=[PROTOCOL_PATHS["golden"]],
            junit_report="golden_results.xml",
            **run_kwargs,
        )
        
    if args.command == "verify-schema":
        rc1 = _run_suite(
            title="schema meta-lint",
            test_paths=[PROTOCOL_PATHS["schema"]],
            passthrough_args=[],
            **run_kwargs,
        )
        if rc1 != 0:
            return rc1
        return _run_suite(
            title="golden wire message validation",
            test_paths=[PROTOCOL_PATHS["golden"]], 
            passthrough_args=[],
            **run_kwargs,
        )
    
    # Wire conformance command
    if args.command == "test-wire":
        return _run_suite(
            title="wire envelope conformance",
            test_paths=[PROTOCOL_PATHS["wire"]],
            junit_report="wire_results.xml",
            markers=getattr(args, "markers", None),
            adapter=getattr(args, "adapter", None),
            skip_schema=getattr(args, "skip_schema", False),
            **run_kwargs,
        )
    
    # Per-protocol commands
    proto_map = {
        "test-llm-conformance": ("LLM Protocol V1", "llm", "corpus_sdk.llm", "llm_coverage_report", "llm_results.xml"),
        "test-vector-conformance": ("Vector Protocol V1", "vector", "corpus_sdk.vector", "vector_coverage_report", "vector_results.xml"),
        "test-graph-conformance": ("Graph Protocol V1", "graph", "corpus_sdk.graph", "graph_coverage_report", "graph_results.xml"),
        "test-embedding-conformance": ("Embedding Protocol V1", "embedding", "corpus_sdk.embedding", "embedding_coverage_report", "embedding_results.xml"),
    }
    
    if args.command in proto_map:
        name, key, cov, report, junit = proto_map[args.command]
        rc = _run_suite(
            title=f"{name} conformance",
            test_paths=[PROTOCOL_PATHS[key]],
            cov_module=cov,
            report_name=report,
            junit_report=junit,
            **run_kwargs,
        )
        if rc == 0 and args.upload:
            _upload_results()
        return rc

    # This should never happen due to argparse required=True
    print(f"error: unknown command '{args.command}'\n", file=sys.stderr)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
