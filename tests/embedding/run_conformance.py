# SPDX-License-Identifier: Apache-2.0
"""
Helper entrypoint to run the Embedding Protocol V1.0 conformance suite.

This is *not* a test module. It's a convenience runner.

Usage (from repo root):
    python -m tests.embedding.run_conformance

or:
    python tests/embedding/run_conformance.py
"""

import os
import sys

try:
    import pytest
except ImportError:  # pragma: no cover
    print("pytest is required to run the conformance tests.", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    # If executed as a module, cwd may not be repo root; normalize it.
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
