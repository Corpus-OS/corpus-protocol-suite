# corpus_sdk/cli.py
# SPDX-License-Identifier: Apache-2.0
"""
Corpus SDK CLI

Lightweight entrypoint to run protocol conformance suites with one command.

Configured via pyproject.toml:

    [project.scripts]
    corpus-sdk = "corpus_sdk.cli:main"

Usage (from repo root or any checkout with tests/):

    # Help
    corpus-sdk
    corpus-sdk --help

    # Run all conformance tests (LLM + Vector + Graph + Embedding)
    corpus-sdk test-all-conformance
    corpus-sdk test-conformance

    # Run per-protocol suites
    corpus-sdk test-llm-conformance
    corpus-sdk test-vector-conformance
    corpus-sdk test-graph-conformance
    corpus-sdk test-embedding-conformance

    # Fast mode (no coverage, parallel)
    corpus-sdk test-fast

    # Run verify (all or filtered)
    corpus-sdk verify
    corpus-sdk verify -p llm -p vector

    # Passthrough pytest args
    corpus-sdk test-llm-conformance -- -x --tb=short

Configuration via environment:
    PYTEST_JOBS=4        # Parallel jobs (default: auto)
    COV_FAIL_UNDER=90    # Coverage threshold (default: 80)
    PYTEST_ARGS=-v       # Additional pytest args

Notes:
- Assumes a top-level `tests/` directory:
      tests/llm/
      tests/vector/
      tests/graph/
      tests/embedding/
- Uses pytest directly; install test extras first:
      pip install .[test]
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None  # type: ignore[assignment]


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
    """
    Best-effort guess of repo root:
    - This file lives in corpus_sdk/cli.py
    - Root is its parent directory.
    """
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.dirname(here)


def _validate_paths(paths: List[str]) -> bool:
    """
    Validate that each path we intend pytest to run exists.

    Only checks arguments that look like paths (no leading '-').
    """
    ok = True
    for p in paths:
        if p.startswith("-"):
            continue
        # Allow either directories or files.
        if not (os.path.isdir(p) or os.path.isfile(p)):
            print(f"error: test path does not exist: {p}", file=sys.stderr)
            ok = False
    return ok


def _build_pytest_args(
    test_paths: List[str],
    cov_module: Optional[str] = None,
    report_name: Optional[str] = None,
    fast_mode: bool = False,
    passthrough_args: List[str] = None,
) -> List[str]:
    """
    Build standardized pytest arguments with consistent configuration.
    """
    if passthrough_args is None:
        passthrough_args = []

    args = [
        *test_paths,
        "-v",
        *PYTEST_EXTRA_ARGS,
        *passthrough_args,
    ]

    # Parallel execution (unless disabled or fast mode)
    if PYTEST_JOBS != "1":
        args.extend(["-n", PYTEST_JOBS])

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


def _run_pytest(args: List[str]) -> int:
    """
    Run pytest with given args from repo root.

    - Ensures pytest is installed.
    - Ensures target test paths exist (nice DX).
    - Prints clean error on unexpected pytest exceptions.
    """
    _ensure_pytest()
    root = _repo_root()
    os.chdir(root)

    # Validate paths before invoking pytest (only non-flag args).
    path_like = [a for a in args if not a.startswith("-")]
    if not _validate_paths(path_like):
        return 2

    try:
        return pytest.main(args)
    except Exception as e:  # pragma: no cover
        print("error: pytest execution failed unexpectedly:", file=sys.stderr)
        print(f"  {type(e).__name__}: {e}", file=sys.stderr)
        return 1


def _print_config() -> None:
    """Print current configuration for transparency."""
    print(f"   Config: jobs={PYTEST_JOBS}, cov_threshold={COV_FAIL_UNDER}%")
    if PYTEST_EXTRA_ARGS:
        print(f"   Extra args: {' '.join(PYTEST_EXTRA_ARGS)}")


def _usage() -> int:
    msg = """Corpus SDK CLI

Usage:
  corpus-sdk test-all-conformance
  corpus-sdk test-conformance
      Run all protocol conformance suites (LLM + Vector + Graph + Embedding).

  corpus-sdk test-llm-conformance
      Run only LLM Protocol V1 conformance tests (tests/llm).

  corpus-sdk test-vector-conformance
      Run only Vector Protocol V1 conformance tests (tests/vector).

  corpus-sdk test-graph-conformance
      Run only Graph Protocol V1 conformance tests (tests/graph).

  corpus-sdk test-embedding-conformance
      Run only Embedding Protocol V1 conformance tests (tests/embedding).

  corpus-sdk test-fast
      Run all tests quickly (no coverage, maximum parallelism).

  corpus-sdk verify [-p llm] [-p vector] [-p graph] [-p embedding]
      Run conformance tests for all (default) or selected protocols.

Configuration (environment variables):
  PYTEST_JOBS=4          Run 4 parallel jobs (default: auto)
  COV_FAIL_UNDER=90      Require 90% coverage (default: 80)
  PYTEST_ARGS="-x -s"    Additional pytest arguments

Examples:
  corpus-sdk test-all-conformance
  corpus-sdk test-embedding-conformance
  corpus-sdk verify
  corpus-sdk verify -p llm -p vector
  corpus-sdk test-llm-conformance -- -x --tb=short  # Passthrough args
  PYTEST_JOBS=1 corpus-sdk test-conformance         # Sequential execution

Notes:
  - Must be run from a checkout (or env) where the `tests/` tree exists.
  - Install dev/test deps first:
        pip install .[test]
"""
    print(msg.strip())
    return 0


# --------------------------------------------------------------------------- #
# verify command
# --------------------------------------------------------------------------- #

def _run_verify(argv: List[str]) -> int:
    """
    Implementation for:

        corpus-sdk verify [-p llm] [-p vector] [-p graph] [-p embedding]

    - No flags       -> run all protocol suites.
    - One/more -p    -> run only those protocol directories.
    """
    _ensure_pytest()
    root = _repo_root()
    os.chdir(root)

    # Parse -p/--protocol flags and passthrough args
    selected: List[str] = []
    passthrough: List[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("-p", "--protocol"):
            i += 1
            if i >= len(argv):
                print("error: --protocol requires a value", file=sys.stderr)
                return 2
            proto = argv[i]
            if proto not in PROTOCOL_PATHS:
                print(f"error: unknown protocol '{proto}'", file=sys.stderr)
                return 2
            if proto not in selected:
                selected.append(proto)
        elif arg == "--":
            # Everything after -- is passthrough
            passthrough.extend(argv[i+1:])
            break
        else:
            print(f"error: unknown option '{arg}'", file=sys.stderr)
            return 2
        i += 1

    if not selected:
        selected = list(PROTOCOL_PATHS.keys())

    paths = [PROTOCOL_PATHS[p] for p in selected]

    if not _validate_paths(paths):
        return 2

    print("🔍 Running Corpus Protocol Conformance Suite...")
    print(f"   Protocols: {', '.join(selected)}")
    _print_config()
    if passthrough:
        print(f"   Passthrough args: {' '.join(passthrough)}")

    try:
        args = _build_pytest_args(
            paths,
            cov_module="corpus_sdk",
            report_name="conformance_coverage_report",
            passthrough_args=passthrough,
        )
        rc = pytest.main(args)
    except Exception as e:  # pragma: no cover
        print("error: pytest execution failed unexpectedly:", file=sys.stderr)
        print(f"  {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    if rc == 0:
        print("\n✅ All selected protocols are 100% conformant.")
    else:
        print("\n❌ Conformance failures detected.")
        print("   Inspect the failed tests above. Each test maps to spec sections via CONFORMANCE.md.")

    return rc


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> int:
    if len(sys.argv) == 1 or sys.argv[1] in ("-h", "--help", "help"):
        return _usage()

    cmd = sys.argv[1]
    passthrough_args = []

    # Extract passthrough args (everything after --)
    if "--" in sys.argv:
        dash_index = sys.argv.index("--")
        passthrough_args = sys.argv[dash_index + 1:]
        sys.argv = sys.argv[:dash_index]

    # Unified "all" / legacy alias
    if cmd in ("test-all-conformance", "test-conformance"):
        print("🚀 Running ALL protocol conformance suites (LLM, Vector, Graph, Embedding)...")
        _print_config()
        if passthrough_args:
            print(f"   Passthrough args: {' '.join(passthrough_args)}")
        return _run_pytest(
            _build_pytest_args(
                list(PROTOCOL_PATHS.values()),
                cov_module="corpus_sdk",
                report_name="conformance_coverage_report",
                passthrough_args=passthrough_args,
            )
        )

    # Fast mode (no coverage, parallel)
    if cmd == "test-fast":
        print("⚡ Running fast tests (no coverage, maximum parallelism)...")
        _print_config()
        if passthrough_args:
            print(f"   Passthrough args: {' '.join(passthrough_args)}")
        return _run_pytest(
            _build_pytest_args(
                list(PROTOCOL_PATHS.values()),
                fast_mode=True,
                passthrough_args=passthrough_args,
            )
        )

    # Per-protocol wrappers with nice progress messages
    protocol_commands = {
        "test-llm-conformance": ("LLM", "llm", "corpus_sdk.llm", "llm_coverage_report"),
        "test-vector-conformance": ("Vector", "vector", "corpus_sdk.vector", "vector_coverage_report"),
        "test-graph-conformance": ("Graph", "graph", "corpus_sdk.graph", "graph_coverage_report"),
        "test-embedding-conformance": ("Embedding", "embedding", "corpus_sdk.embedding", "embedding_coverage_report"),
    }

    if cmd in protocol_commands:
        protocol_name, protocol_key, cov_module, report_name = protocol_commands[cmd]
        print(f"🚀 Running {protocol_name} Protocol V1 conformance tests...")
        _print_config()
        if passthrough_args:
            print(f"   Passthrough args: {' '.join(passthrough_args)}")
        return _run_pytest(
            _build_pytest_args(
                [PROTOCOL_PATHS[protocol_key]],
                cov_module=cov_module,
                report_name=report_name,
                passthrough_args=passthrough_args,
            )
        )

    # verify (all or subset)
    if cmd == "verify":
        return _run_verify(sys.argv[2:] + (["--"] + passthrough_args if passthrough_args else []))

    # Unknown command -> usage + non-zero
    print(f"error: unknown command '{cmd}'\n", file=sys.stderr)
    _usage()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())