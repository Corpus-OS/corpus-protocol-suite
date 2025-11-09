# SPDX-License-Identifier: Apache-2.0
"""
Pytest plugin: pretty terminal summary for Corpus Protocol conformance.

Drop this into `tests/conftest.py` (or any file auto-discovered by pytest)
to get a per-protocol summary at the end of a run.

It:
- Prints a ✅ banner if everything passed
- Otherwise groups failures by protocol:
    - tests/llm/
    - tests/vector/
    - tests/graph/
    - tests/embedding/
- Includes collection/runtime errors in the counts
"""

from __future__ import annotations


def pytest_terminal_summary(terminalreporter, exitstatus):
    # Collect both test failures and internal/errors (import errors, etc.)
    failed_reports = []
    for key in ("failed", "error"):
        failed_reports.extend(terminalreporter.stats.get(key, []))

    # All green ✅
    if not failed_reports:
        terminalreporter.write_sep("=", "✅ Corpus Protocol Conformance: ALL PASS")
        terminalreporter.write_line(
            "All LLM, Vector, Graph, and Embedding conformance tests passed."
        )
        return

    # Some failures ❌
    terminalreporter.write_sep("=", "❌ Corpus Protocol Conformance Summary")

    by_protocol = {
        "llm": 0,
        "vector": 0,
        "graph": 0,
        "embedding": 0,
        "other": 0,
    }

    for rep in failed_reports:
        nodeid = getattr(rep, "nodeid", "") or ""

        # Handle both POSIX and Windows paths
        if "tests/llm/" in nodeid or "tests\\llm\\" in nodeid:
            by_protocol["llm"] += 1
        elif "tests/vector/" in nodeid or "tests\\vector\\" in nodeid:
            by_protocol["vector"] += 1
        elif "tests/graph/" in nodeid or "tests\\graph\\" in nodeid:
            by_protocol["graph"] += 1
        elif "tests/embedding/" in nodeid or "tests\\embedding\\" in nodeid:
            by_protocol["embedding"] += 1
        else:
            by_protocol["other"] += 1

    for proto, count in by_protocol.items():
        if count:
            label = proto if proto != "other" else "other / non-conformance tests"
            terminalreporter.write_line(f"  - {label}: {count} failing test(s)")

    terminalreporter.write_line(
        "See protocol-specific CONFORMANCE.md files for the mapping from tests → spec sections."
    )
