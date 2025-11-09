# SPDX-License-Identifier: Apache-2.0
"""
Module entrypoint to run all Vector Protocol conformance tests.

Usage:
    python -m tests.vector.run_conformance
"""

import os
import sys

import pytest


def main() -> int:
    # This file lives in tests/vector/
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.dirname(os.path.dirname(here))
    os.chdir(repo_root)

    args = [
        "tests/vector",
        "-v",
        "--cov=corpus_sdk.vector",
        "--cov-report=term",
        "--cov-report=html:vector_coverage_report",
    ]
    return pytest.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
