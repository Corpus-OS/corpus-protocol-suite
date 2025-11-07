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

    # Run all conformance tests (LLM + Vector + Graph + Embedding)
    corpus-sdk test-all-conformance
    corpus-sdk test-conformance

    # Run per-protocol suites
    corpus-sdk test-llm-conformance
    corpus-sdk test-vector-conformance
    corpus-sdk test-graph-conformance
    corpus-sdk test-embedding-conformance

    # Run verify (all or filtered)
    corpus-sdk verify
    corpus-sdk verify -p llm -p vector

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
from typing import Dict, Iterable, List

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


def _validate_paths(paths: Iterable[str]) -> bool:
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

  corpus-sdk verify [-p llm] [-p vector] [-p graph] [-p embedding]
      Run conformance tests for all (default) or selected protocols.

Examples:
  corpus-sdk test-all-conformance
  corpus-sdk test-embedding-conformance
  corpus-sdk verify
  corpus-sdk verify -p llm -p vector

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

    # Tiny manual arg parse (only -p/--protocol).
    selected: List[str] = []
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
    for p in selected:
        print(f"   • {p}: {PROTOCOL_PATHS[p]}")

    try:
        rc = pytest.main([*paths, "-v"])
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
    if len(sys.argv) == 1:
        return _usage()

    cmd = sys.argv[1]

    # Unified “all” / legacy alias
    if cmd in ("test-all-conformance", "test-conformance"):
        print("🚀 Running ALL protocol conformance suites (LLM, Vector, Graph, Embedding)...")
        return _run_pytest(
            [
                "tests/llm",
                "tests/vector",
                "tests/graph",
                "tests/embedding",
                "-v",
                "--cov=corpus_sdk",
                "--cov-report=term",
                "--cov-report=html:conformance_coverage_report",
            ]
        )

    # Per-protocol wrappers with nice progress messages
    if cmd == "test-llm-conformance":
        print("🚀 Running LLM Protocol V1 conformance tests...")
        return _run_pytest(
            [
                "tests/llm",
                "-v",
                "--cov=corpus_sdk.llm",
                "--cov-report=term",
                "--cov-report=html:llm_coverage_report",
            ]
        )

    if cmd == "test-vector-conformance":
        print("🚀 Running Vector Protocol V1 conformance tests...")
        return _run_pytest(
            [
                "tests/vector",
                "-v",
                "--cov=corpus_sdk.vector",
                "--cov-report=term",
                "--cov-report=html:vector_coverage_report",
            ]
        )

    if cmd == "test-graph-conformance":
        print("🚀 Running Graph Protocol V1 conformance tests...")
        return _run_pytest(
            [
                "tests/graph",
                "-v",
                "--cov=corpus_sdk.graph",
                "--cov-report=term",
                "--cov-report=html:graph_coverage_report",
            ]
        )

    if cmd == "test-embedding-conformance":
        print("🚀 Running Embedding Protocol V1 conformance tests...")
        return _run_pytest(
            [
                "tests/embedding",
                "-v",
                "--cov=corpus_sdk.embedding",
                "--cov-report=term",
                "--cov-report=html:embedding_coverage_report",
            ]
        )

    # verify (all or subset)
    if cmd == "verify":
        return _run_verify(sys.argv[2:])

    # Unknown command -> usage + non-zero
    print(f"error: unknown command '{cmd}'\n", file=sys.stderr)
    _usage()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

