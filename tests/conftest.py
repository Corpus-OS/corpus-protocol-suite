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
- Shows total test counts and duration
- Provides actionable debugging guidance
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional


# Protocol configuration
PROTOCOLS = ["llm", "vector", "graph", "embedding"]


class CorpusProtocolPlugin:
    """Pytest plugin for Corpus Protocol conformance reporting."""
    
    def __init__(self):
        self.start_time: Optional[float] = None
        
    def pytest_sessionstart(self, session):
        """Record session start time for duration calculation."""
        self.start_time = time.time()
    
    def _get_test_counts(self, terminalreporter) -> Dict[str, int]:
        """Get counts of passed, failed, skipped tests."""
        return {
            "passed": len(terminalreporter.stats.get("passed", [])),
            "failed": len(terminalreporter.stats.get("failed", [])),
            "skipped": len(terminalreporter.stats.get("skipped", [])),
            "error": len(terminalreporter.stats.get("error", [])),
            "xfailed": len(terminalreporter.stats.get("xfailed", [])),
            "xpassed": len(terminalreporter.stats.get("xpassed", [])),
        }
    
    def _get_total_tests(self, counts: Dict[str, int]) -> int:
        """Calculate total number of tests run."""
        return sum(counts.values())
    
    def _get_duration(self) -> float:
        """Calculate test session duration."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def _categorize_failures(self, failed_reports: List) -> Dict[str, int]:
        """Categorize failed tests by protocol."""
        by_protocol = {proto: 0 for proto in PROTOCOLS}
        by_protocol["other"] = 0
        
        for rep in failed_reports:
            nodeid = getattr(rep, "nodeid", "") or ""
            
            # Handle both POSIX and Windows paths
            categorized = False
            for proto in PROTOCOLS:
                if f"tests/{proto}/" in nodeid or f"tests\\{proto}\\" in nodeid:
                    by_protocol[proto] += 1
                    categorized = True
                    break
            
            if not categorized:
                by_protocol["other"] += 1
                
        return by_protocol
    
    def _format_protocol_name(self, proto: str) -> str:
        """Format protocol name for display."""
        if proto == "other":
            return "other / non-conformance tests"
        return proto.upper()
    
    def _print_success_summary(self, terminalreporter, counts: Dict[str, int], duration: float):
        """Print summary when all tests pass."""
        total_tests = self._get_total_tests(counts)
        protocol_names = [proto.upper() for proto in PROTOCOLS]
        
        terminalreporter.write_sep("=", "✅ Corpus Protocol Conformance: ALL PASS")
        terminalreporter.write_line(
            f"All {total_tests} tests across {len(PROTOCOLS)} protocols passed."
        )
        terminalreporter.write_line(
            f"Protocols: {', '.join(protocol_names)}"
        )
        terminalreporter.write_line(
            f"Completed in {duration:.2f}s"
        )
    
    def _print_failure_summary(self, terminalreporter, by_protocol: Dict[str, int], duration: float):
        """Print summary when there are test failures."""
        terminalreporter.write_sep("=", "❌ Corpus Protocol Conformance Summary")
        
        # Show failing protocols first
        failing_protocols = {k: v for k, v in by_protocol.items() if v > 0}
        
        for proto, count in failing_protocols.items():
            label = self._format_protocol_name(proto)
            terminalreporter.write_line(f"  - {label}: {count} failing test(s)")
        
        terminalreporter.write_line("")
        terminalreporter.write_line(
            "See protocol-specific CONFORMANCE.md files for the mapping from tests → spec sections."
        )
        terminalreporter.write_line(
            f"Completed in {duration:.2f}s"
        )
    
    def pytest_terminal_summary(self, terminalreporter, exitstatus, config):
        """Generate Corpus Protocol conformance summary."""
        # Collect both test failures and internal errors
        failed_reports = []
        for key in ("failed", "error"):
            failed_reports.extend(terminalreporter.stats.get(key, []))
        
        # Get test counts and duration
        counts = self._get_test_counts(terminalreporter)
        duration = self._get_duration()
        
        # All tests passed ✅
        if not failed_reports:
            self._print_success_summary(terminalreporter, counts, duration)
            return
        
        # Some failures ❌ - categorize by protocol
        by_protocol = self._categorize_failures(failed_reports)
        self._print_failure_summary(terminalreporter, by_protocol, duration)
    
    def pytest_runtest_logstart(self, nodeid, location):
        """Optional: Show which protocol is currently being tested."""
        # This provides progress feedback during test execution
        for proto in PROTOCOLS:
            if f"tests/{proto}/" in nodeid or f"tests\\{proto}\\" in nodeid:
                # Could add progress reporting here if desired
                break


# Instantiate the plugin
corpus_protocol_plugin = CorpusProtocolPlugin()

# Pytest hook functions (delegate to plugin instance)
def pytest_sessionstart(session):
    corpus_protocol_plugin.pytest_sessionstart(session)

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    corpus_protocol_plugin.pytest_terminal_summary(terminalreporter, exitstatus, config)

def pytest_runtest_logstart(nodeid, location):
    corpus_protocol_plugin.pytest_runtest_logstart(nodeid, location)
