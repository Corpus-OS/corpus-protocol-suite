# corpus_sdk/cli.py
# SPDX-License-Identifier: Apache-2.0
"""
Corpus SDK CLI

Lightweight entrypoint to run protocol conformance suites with one command.

Configured via pyproject.toml:

    [project.scripts]
    corpus-sdk = "corpus_sdk.cli:main"

Usage (from repo root or any environment with tests/ available):

    # Show help
    corpus-sdk

    # Run ALL protocol conformance suites (LLM + Vector + Graph + Embedding)
    corpus-sdk test-all-conformance
    corpus-sdk verify

    # Run per-protocol suites
    corpus-sdk test-llm-conformance
    corpus-sdk test-vector-conformance
    corpus-sdk test-graph-conformance
    corpus-sdk test-embedding-conformance

    # Flexible verify mode:
    corpus-sdk verify --protocol llm --protocol vector
    corpus-sdk verify -p embedding

Notes:
- Assumes a top-level `tests/` directory with:
      tests/llm/
      tests/vector/
      tests/graph/
      tests/embedding/
- Uses pytest directly; requires the "test" extra:
      pip install .[test]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None  # type: ignore[assignment]


# Map protocol names → test directory paths
PROTOCOL_PATHS: Dict[str, str] = {
    "llm": "tests/llm",
    "vector": "tests/vector",
    "graph": "tests/graph",
    "embedding": "tests/embedding",
}


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


# ----- command handlers ------------------------------------------------------


def _cmd_test_all_conformance(_args: argparse.Namespace) -> int:
    # Full suite across all protocols with coverage over the whole SDK
    return _run_pytest(
        [
            "tests",
            "-v",
            "--cov=corpus_sdk",
            "--cov-report=term",
            "--cov-report=html:conformance_coverage_report",
        ]
    )


def _cmd_test_single_protocol(proto: str, cov_pkg: str, report_dir: str) -> int:
    path = PROTOCOL_PATHS[proto]
    return _run_pytest(
        [
            path,
            "-v",
            f"--cov={cov_pkg}",
            "--cov-report=term",
            f"--cov-report=html:{report_dir}",
        ]
    )


def _cmd_test_llm_conformance(_args: argparse.Namespace) -> int:
    return _cmd_test_single_protocol("llm", "corpus_sdk.llm", "llm_coverage_report")


def _cmd_test_vector_conformance(_args: argparse.Namespace) -> int:
    return _cmd_test_single_protocol("vector", "corpus_sdk.vector", "vector_coverage_report")


def _cmd_test_graph_conformance(_args: argparse.Namespace) -> int:
    return _cmd_test_single_protocol("graph", "corpus_sdk.graph", "graph_coverage_report")


def _cmd_test_embedding_conformance(_args: argparse.Namespace) -> int:
    return _cmd_test_single_protocol(
        "embedding", "corpus_sdk.embedding", "embedding_coverage_report"
    )


def _cmd_verify(args: argparse.Namespace) -> int:
    """
    Flexible 'verify' command:

    - No --protocol → run all protocol suites (same as test-all-conformance)
    - With one or more --protocol → run only those suites.

    Examples:
        corpus-sdk verify
        corpus-sdk verify --protocol llm --protocol vector
        corpus-sdk verify -p embedding
    """
    # Determine which protocols to run
    if args.protocol:
        invalid = [p for p in args.protocol if p not in PROTOCOL_PATHS]
        if invalid:
            print(
                f"error: unknown protocol(s): {', '.join(invalid)}\n"
                f"valid choices: {', '.join(PROTOCOL_PATHS.keys())}",
                file=sys.stderr,
            )
            return 2
        protos = args.protocol
    else:
        protos = list(PROTOCOL_PATHS.keys())

    # Build pytest args:
    # - one or more test paths
    # - verbose
    # - coverage over the whole SDK (simpler + stable)
    pytest_args: List[str] = []
    for p in protos:
        pytest_args.append(PROTOCOL_PATHS[p])

    pytest_args.extend(
        [
            "-v",
            "--cov=corpus_sdk",
            "--cov-report=term",
            "--cov-report=html:conformance_coverage_report",
        ]
    )

    print("🔍 Running Corpus Protocol Conformance Suite...")
    print(f"   Protocols: {', '.join(protos)}")
    rc = _run_pytest(pytest_args)

    if rc == 0:
        print("\n✅ All selected protocols are 100% conformant.")
    else:
        print("\n❌ Conformance failures detected.")
        print("   Inspect failed tests above; each one maps to a spec section via the per-protocol CONFORMANCE.md.")

    return rc


# ----- main / argument parsing ----------------------------------------------


def main(argv: List[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="corpus-sdk",
        description="Corpus SDK conformance test runner",
    )
    sub = parser.add_subparsers(dest="cmd")

    # test-all-conformance
    p_all = sub.add_parser(
        "test-all-conformance",
        help="Run ALL protocol conformance suites (LLM, Vector, Graph, Embedding)",
    )
    p_all.set_defaults(func=_cmd_test_all_conformance)

    # Per-protocol commands (simple, explicit)
    p_llm = sub.add_parser("test-llm-conformance", help="Run LLM Protocol V1 conformance tests")
    p_llm.set_defaults(func=_cmd_test_llm_conformance)

    p_vec = sub.add_parser("test-vector-conformance", help="Run Vector Protocol V1 conformance tests")
    p_vec.set_defaults(func=_cmd_test_vector_conformance)

    p_graph = sub.add_parser("test-graph-conformance", help="Run Graph Protocol V1 conformance tests")
    p_graph.set_defaults(func=_cmd_test_graph_conformance)

    p_emb = sub.add_parser(
        "test-embedding-conformance",
        help="Run Embedding Protocol V1 conformance tests",
    )
    p_emb.set_defaults(func=_cmd_test_embedding_conformance)

    # verify (nice UX alias, supports filtering)
    v = sub.add_parser(
        "verify",
        help=(
            "Run protocol conformance suites. "
            "Use without args for all, or -p/--protocol to filter."
        ),
    )
    v.add_argument(
        "-p",
        "--protocol",
        choices=sorted(PROTOCOL_PATHS.keys()),
        action="append",
        help="Limit verification to one or more protocols "
             "(can be repeated: -p llm -p vector)",
    )
    v.set_defaults(func=_cmd_verify)

    # No subcommand → print help
    if not argv:
        parser.print_help()
        return 0

    args = parser.parse_args(argv)

    if not getattr(args, "func", None):
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
