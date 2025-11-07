# SPDX-License-Identifier: Apache-2.0
"""
Module entrypoint to run all protocol conformance tests.

Usage:
    python -m tests.run_conformance
"""

import os
import sys

import pytest


def main() -> int:
    # This file lives in tests/
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.dirname(here)
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
