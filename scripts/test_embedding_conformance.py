#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Run all Embedding Protocol V1.0 conformance tests in one shot.

Usage:
    python scripts/test_embedding_conformance.py
"""

import os
import sys

try:
    import pytest
except ImportError:  # pragma: no cover
    print("pytest is required to run the conformance tests.", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    args = [
        "tests/embedding",
        "-v",
        "--cov=corpus_sdk.embedding",
        "--cov-report=term",
        "--cov-report=html:embedding_coverage_report",
    ]
    return pytest.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
