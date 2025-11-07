#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""
Run all Embedding Protocol V1.0 conformance tests.

Usage (from repo root):
    python scripts/test_embedding_conformance.py

Or:
    pytest tests/embedding -v
"""

import os
import sys

try:
    import pytest
except ImportError:  # pragma: no cover
    print("pytest is required to run the embedding conformance suite.", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    repo_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
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
