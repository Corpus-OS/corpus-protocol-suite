# SPDX-License-Identifier: Apache-2.0
"""
Module entrypoint to run all protocol conformance tests.

Usage:
    # Basic run
    python -m tests.run_conformance
    
    # With configuration
    PYTEST_JOBS=4 COV_FAIL_UNDER=90 python -m tests.run_conformance
    
    # With pytest arguments
    python -m tests.run_conformance -x --tb=short
    
    # Fast mode (no coverage)
    python -m tests.run_conformance --no-cov -n auto

    # Programmatic usage
    from tests.run_conformance import main
    exit_code = main(["-x", "--tb=short"])

Environment variables:
    PYTEST_JOBS      - Parallel jobs (default: auto, set to 1 to disable)
    COV_FAIL_UNDER   - Coverage threshold % (default: 80)
    PYTEST_ARGS      - Additional pytest arguments (space-separated)

Note:
    For more features (per-protocol runs, verify mode, etc.),
    use the corpus-sdk CLI tool instead:
        corpus-sdk test-conformance
        corpus-sdk test-llm-conformance
        corpus-sdk verify -p llm -p vector
"""

from __future__ import annotations

import os
import sys

import pytest


def _validate_test_dirs() -> bool:
    """Ensure all protocol test directories exist."""
    test_dirs = ["tests/llm", "tests/vector", "tests/graph", "tests/embedding"]
    missing = [d for d in test_dirs if not os.path.isdir(d)]
    if missing:
        print(f"error: test directories not found: {', '.join(missing)}", file=sys.stderr)
        return False
    return True


def _build_pytest_args(
    argv: list[str],
    pytest_extra_args: list[str],
    pytest_jobs: str,
    cov_fail_under: str,
) -> list[str]:
    """Build pytest arguments with consistent configuration."""
    args = [
        "tests",
        "-v",
        *pytest_extra_args,
        *argv,
    ]
    
    # Parallel execution (unless disabled)
    if pytest_jobs != "1":
        args.extend(["-n", pytest_jobs])
    
    # Coverage configuration (skip if --no-cov in args)
    if "--no-cov" not in argv and "--no-cov" not in pytest_extra_args:
        args.extend([
            "--cov=corpus_sdk",
            f"--cov-fail-under={cov_fail_under}",
            "--cov-report=term",
            "--cov-report=html:conformance_coverage_report",
        ])
    
    return args


def main(argv: list[str] | None = None) -> int:
    """
    Run all protocol conformance tests.
    
    Args:
        argv: Command line arguments. Uses sys.argv[1:] if None.
        
    Returns:
        Exit code from pytest (0 = success, non-zero = failure)
    """
    if argv is None:
        argv = sys.argv[1:]
    
    # This file lives in tests/
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.dirname(here)
    os.chdir(repo_root)

    # Validate test directories exist
    if not _validate_test_dirs():
        return 1

    # Configuration from environment
    pytest_jobs = os.environ.get("PYTEST_JOBS", "auto")
    cov_fail_under = os.environ.get("COV_FAIL_UNDER", "80")
    pytest_extra_args = os.environ.get("PYTEST_ARGS", "").split()
    
    # Filter out empty strings from split
    pytest_extra_args = [arg for arg in pytest_extra_args if arg]
    
    # User feedback
    print("🚀 Running Corpus Protocol Conformance Tests...")
    print(f"   Config: jobs={pytest_jobs}, cov_threshold={cov_fail_under}%")
    if pytest_extra_args:
        print(f"   Extra args (PYTEST_ARGS): {' '.join(pytest_extra_args)}")
    if argv:
        print(f"   Command line args: {' '.join(argv)}")

    # Build pytest arguments
    args = _build_pytest_args(argv, pytest_extra_args, pytest_jobs, cov_fail_under)

    try:
        return pytest.main(args)
    except Exception as e:  # pragma: no cover
        print("\nerror: pytest execution failed unexpectedly:", file=sys.stderr)
        print(f"  {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())