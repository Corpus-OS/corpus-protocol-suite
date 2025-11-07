#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Run all protocol conformance test suites (LLM, Vector, Graph, Embedding).

Usage:
    python scripts/test_all_conformance.py
"""

import os
import sys

try:
    import pytest
except ImportError:  # pragma: no cover
    print("pytest is required to run the conformance tests.", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    # scripts/ -> repo root
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    args = [
        "tests",
        "-v",
        "--cov=corpus_sdk",
        "--cov-report=term",
        "--cov-report=html:conformance_coverage_report",
    ]
    return pytest.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
