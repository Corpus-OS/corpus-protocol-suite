# corpus_sdk/cli.py
# SPDX-License-Identifier: Apache-2.0
"""
Corpus SDK CLI

Lightweight entrypoint to run protocol conformance suites with one command.

Configured via pyproject.toml:

    [project.scripts]
    corpus-sdk = "corpus_sdk.cli:main"

Usage (from repo root):

    # Show help
    corpus-sdk

    # Run all conformance tests (LLM + Vector + Graph + Embedding)
    corpus-sdk test-all-conformance

    # Run per-protocol suites
    corpus-sdk test-llm-conformance
    corpus-sdk test-vector-conformance
    corpus-sdk test-graph-conformance
    corpus-sdk test-embedding-conformance

Notes:
- Assumes the standard layout with a top-level `tests/` directory:
    tests/llm/
    tests/vector/
    tests/graph/
    tests/embedding/
- Uses pytest directly; requires the "test" extra:
    pip install .[test]
"""

from __future__ import annotations

import os
import sys
from typing import List

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None  # type: ignore[assignment]


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


def _run_pytest(args: List[str]) -> int:
    _ensure_pytest()
    root = _repo_root()
    os.chdir(root)
    return pytest.main(args)


def _usage() -> int:
    msg = """Corpus SDK CLI

Usage:
  corpus-sdk test-all-conformance
      Run all protocol conformance suites (LLM + Vector + Graph + Embedding).

  corpus-sdk test-llm-conformance
      Run only LLM Protocol V1 conformance tests (tests/llm).

  corpus-sdk test-vector-conformance
      Run only Vector Protocol V1 conformance tests (tests/vector).

  corpus-sdk test-graph-conformance
      Run only Graph Protocol V1 conformance tests (tests/graph).

  corpus-sdk test-embedding-conformance
      Run only Embedding Protocol V1 conformance tests (tests/embedding).

Examples:
  corpus-sdk test-all-conformance
  corpus-sdk test-embedding-conformance

Notes:
  - Must be run from a checkout (or environment) where the `tests/` tree exists.
  - Install dev/test deps first:
        pip install .[test]
"""
    print(msg.strip())
    return 0


def main() -> int:
    if len(sys.argv) == 1:
        return _usage()

    cmd = sys.argv[1]

    if cmd == "test-all-conformance":
        # Full suite across all protocols
        return _run_pytest(
            [
                "tests",
                "-v",
                "--cov=corpus_sdk",
                "--cov-report=term",
                "--cov-report=html:conformance_coverage_report",
            ]
        )

    if cmd == "test-embedding-conformance":
        return _run_pytest(
            [
                "tests/embedding",
                "-v",
                "--cov=corpus_sdk.embedding",
                "--cov-report=term",
                "--cov-report=html:embedding_coverage_report",
            ]
        )

    if cmd == "test-llm-conformance":
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
        return _run_pytest(
            [
                "tests/graph",
                "-v",
                "--cov=corpus_sdk.graph",
                "--cov-report=term",
                "--cov-report=html:graph_coverage_report",
            ]
        )

    # Unknown command -> usage + non-zero
    print(f"error: unknown command '{cmd}'\n", file=sys.stderr)
    _usage()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
