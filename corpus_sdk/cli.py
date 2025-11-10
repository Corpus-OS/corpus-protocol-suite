# corpus_sdk/cli.py
# SPDX-License-Identifier: Apache-2.0
"""
Corpus SDK CLI

Lightweight entrypoint to run protocol conformance suites with one command.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional

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
}

# Configuration from environment
PYTEST_JOBS = os.environ.get("PYTEST_JOBS", "auto")
COV_FAIL_UNDER = os.environ.get("COV_FAIL_UNDER", "80")
PYTEST_EXTRA_ARGS = os.environ.get("PYTEST_ARGS", "").split()


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
    """Validate that each path we intend pytest to run exists."""
    ok = True
    for p in paths:
        if p.startswith("-"):
            continue
        if not (os.path.isdir(p) or os.path.isfile(p)):
            print(f"error: test path does not exist: {p}", file=sys.stderr)
            ok = False
    return ok


def _build_pytest_args(
    test_paths: List[str],
    cov_module: Optional[str] = None,
    report_name: Optional[str] = None,
    fast_mode: bool = False,
    quiet_mode: bool = False,
    verbose_mode: bool = False,
    passthrough_args: List[str] = None,
) -> List[str]:
    """Build standardized pytest arguments with consistent configuration."""
    if passthrough_args is None:
        passthrough_args = []

    args = [
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
        args.append("-m")
        args.append("not slow")

    # Coverage configuration (skip for fast mode)
    if not fast_mode and cov_module:
        args.extend([
            f"--cov={cov_module}",
            f"--cov-fail-under={COV_FAIL_UNDER}",
            "--cov-report=term",
        ])
        if report_name:
            args.append(f"--cov-report=html:{report_name}")

    return args


def _run_pytest(args: List[str]) -> tuple[int, float]:
    """Run pytest with given args from repo root."""
    _ensure_pytest()
    root = _repo_root()
    os.chdir(root)

    # Validate paths before invoking pytest
    path_like = [a for a in args if not a.startswith("-")]
    if not _validate_paths(path_like):
        return 2, 0.0

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
        if PYTEST_EXTRA_ARGS:
            print(f"   Extra args: {' '.join(PYTEST_EXTRA_ARGS)}")


def _print_success_stats(protocols: List[str], elapsed: float) -> None:
    """Print success statistics."""
    print(f"✅ All selected protocols are 100% conformant.")
    print(f"   Protocols: {', '.join(protocols)}")
    print(f"   Completed in {elapsed:.1f}s")


def _run_watch_mode(test_paths: List[str], passthrough_args: List[str] = None) -> int:
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
    watch_paths = list(set(os.path.dirname(p) for p in test_paths))
    
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
) -> int:
    """Consolidated function to run any test suite."""
    
    # Handle watch mode first
    if watch_mode:
        if not quiet_mode:
            print(f"👀 Starting watch mode for {title}...")
        return _run_watch_mode(test_paths, passthrough_args)

    # Standard run
    if not quiet_mode:
        print(f"🚀 Running {title}...")
        _print_config(quiet_mode)
        if passthrough_args:
            print(f"   Passthrough args: {' '.join(passthrough_args)}")

    args = _build_pytest_args(
        test_paths,
        cov_module=cov_module,
        report_name=report_name,
        fast_mode=fast_mode,
        quiet_mode=quiet_mode,
        verbose_mode=verbose_mode,
        passthrough_args=passthrough_args,
    )
    
    rc, elapsed = _run_pytest(args)
    
    if rc == 0 and not quiet_mode:
        # Get protocol names from paths for the stats message
        protocols = []
        path_to_proto = {v: k for k, v in PROTOCOL_PATHS.items()}
        for p in test_paths:
            if p in path_to_proto:
                protocols.append(path_to_proto[p].upper())
            else:
                # Fallback for "all" or multiple protocols
                protocols = [proto.upper() for proto in PROTOCOL_PATHS.keys()]
                break
        _print_success_stats(protocols, elapsed)
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
    passthrough_args = []
    if "--" in cli_args:
        split_index = cli_args.index("--")
        passthrough_args = cli_args[split_index + 1:]
        cli_args = cli_args[:split_index]

    # Setup the main parser
    parser = argparse.ArgumentParser(
        description="Corpus SDK CLI - Protocol conformance testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  corpus-sdk test-all-conformance
  corpus-sdk test-embedding-conformance --watch
  corpus-sdk verify --quiet
  corpus-sdk check -p llm -p vector
  corpus-sdk test-llm-conformance -- -x --tb=short
  PYTEST_JOBS=1 corpus-sdk test-conformance

Configuration (environment variables):
  PYTEST_JOBS=4          Run 4 parallel jobs (default: auto)
  COV_FAIL_UNDER=90      Require 90% coverage (default: 80)
  PYTEST_ARGS="-x -s"    Additional pytest arguments

Notes:
  - Install dependencies: pip install .[test]
  - For watch mode: pip install .[watch]
        """.strip()
    )
    
    # Global flags
    parser.add_argument(
        "-q", "--quiet", action="store_true", 
        help="Minimal output (quiet mode)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", 
        help="Detailed output (-vv)"
    )
    parser.add_argument(
        "-w", "--watch", action="store_true", 
        help="Run in watch mode (TDD)"
    )

    # Subparsers for each command
    subparsers = parser.add_subparsers(
        dest="command", 
        required=True, 
        help="command to execute",
        metavar="COMMAND"
    )

    # test-all-conformance / test-conformance
    all_parser = subparsers.add_parser(
        "test-all-conformance", 
        help="Run all protocol conformance suites"
    )
    subparsers.add_parser(
        "test-conformance", 
        help="Alias for test-all-conformance"
    )

    # test-fast
    subparsers.add_parser(
        "test-fast", 
        help="Run all tests quickly (no coverage, skip slow tests)"
    )

    # Per-protocol commands
    subparsers.add_parser(
        "test-llm-conformance", 
        help="Run only LLM Protocol conformance tests"
    )
    subparsers.add_parser(
        "test-vector-conformance", 
        help="Run only Vector Protocol conformance tests"
    )
    subparsers.add_parser(
        "test-graph-conformance", 
        help="Run only Graph Protocol conformance tests"
    )
    subparsers.add_parser(
        "test-embedding-conformance", 
        help="Run only Embedding Protocol conformance tests"
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

    # Parse the args
    try:
        args = parser.parse_args(cli_args)
    except SystemExit as e:
        return e.code

    # Common args for _run_suite
    run_kwargs = {
        "passthrough_args": passthrough_args,
        "quiet_mode": args.quiet,
        "verbose_mode": args.verbose,
        "watch_mode": args.watch,
    }

    # Dispatch commands
    if args.command in ("test-all-conformance", "test-conformance"):
        return _run_suite(
            title="ALL protocol conformance suites (LLM, Vector, Graph, Embedding)",
            test_paths=list(PROTOCOL_PATHS.values()),
            cov_module="corpus_sdk",
            report_name="conformance_coverage_report",
            **run_kwargs,
        )

    if args.command == "test-fast":
        return _run_suite(
            title="fast tests (no coverage, skipping slow tests)",
            test_paths=list(PROTOCOL_PATHS.values()),
            fast_mode=True,
            **run_kwargs,
        )

    if args.command == "verify":
        selected = args.protocol or list(PROTOCOL_PATHS.keys())
        paths = [PROTOCOL_PATHS[p] for p in selected]
        protocol_names = [p.upper() for p in selected]
        return _run_suite(
            title=f"{', '.join(protocol_names)} Protocol conformance",
            test_paths=paths,
            cov_module="corpus_sdk",
            report_name="conformance_coverage_report",
            **run_kwargs,
        )
    
    # Per-protocol commands
    proto_map = {
        "test-llm-conformance": ("LLM Protocol V1", "llm", "corpus_sdk.llm", "llm_coverage_report"),
        "test-vector-conformance": ("Vector Protocol V1", "vector", "corpus_sdk.vector", "vector_coverage_report"),
        "test-graph-conformance": ("Graph Protocol V1", "graph", "corpus_sdk.graph", "graph_coverage_report"),
        "test-embedding-conformance": ("Embedding Protocol V1", "embedding", "corpus_sdk.embedding", "embedding_coverage_report"),
    }
    
    if args.command in proto_map:
        name, key, cov, report = proto_map[args.command]
        return _run_suite(
            title=f"{name} conformance",
            test_paths=[PROTOCOL_PATHS[key]],
            cov_module=cov,
            report_name=report,
            **run_kwargs,
        )

    # This should never happen due to argparse required=True
    print(f"error: unknown command '{args.command}'\n", file=sys.stderr)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())