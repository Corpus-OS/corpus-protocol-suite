# SPDX-License-Identifier: Apache-2.0
"""
Module entrypoint to run all Graph Protocol conformance tests.

Usage:
    python -m tests.graph.run_conformance
"""

import os
import sys

import pytest


def main() -> int:
    # This file lives in tests/graph/
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.dirname(os.path.dirname(here))
    os.chdir(repo_root)

    args = [
        "tests/graph",
        "-v",
        "--cov=corpus_sdk.graph",
        "--cov-report=term",
        "--cov-report=html:graph_coverage_report",
    ]
    return pytest.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
