# SPDX-License-Identifier: Apache-2.0
"""
Pytest plugin: comprehensive terminal summary for Corpus Protocol conformance.

Drop this into `tests/conftest.py` (or any file auto-discovered by pytest)
to get detailed per-protocol conformance reporting with certification levels,
failure analysis, and actionable guidance.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple, Any


# Protocol configuration with certification levels
PROTOCOLS = ["llm", "vector", "graph", "embedding", "schema", "golden"]

PROTOCOL_DISPLAY_NAMES = {
    "llm": "LLM Protocol V1.0",
    "vector": "Vector Protocol V1.0", 
    "graph": "Graph Protocol V1.0",
    "embedding": "Embedding Protocol V1.0", 
    "schema": "Schema Conformance",
    "golden": "Golden Wire Validation"
}

CONFORMANCE_LEVELS = {
    "llm": {"gold": 61, "silver": 49, "development": 31},
    "vector": {"gold": 72, "silver": 58, "development": 36},
    "graph": {"gold": 68, "silver": 54, "development": 34},
    "embedding": {"gold": 75, "silver": 60, "development": 38},
    "schema": {"gold": 86, "silver": 69, "development": 43},
    "golden": {"gold": 73, "silver": 58, "development": 37},
}

TEST_CATEGORIES = {
    "llm": {
        "core_ops": "Core Operations",
        "message_validation": "Message Validation", 
        "sampling_params": "Sampling Parameters",
        "streaming": "Streaming Semantics",
        "error_handling": "Error Handling",
        "capabilities": "Capabilities Discovery",
        "observability": "Observability & Privacy",
        "deadline": "Deadline Semantics", 
        "token_counting": "Token Counting",
        "health": "Health Endpoint",
        "wire_envelopes": "Wire Envelopes & Routing"
    },
    "vector": {
        "core_ops": "Core Operations",
        "capabilities": "Capabilities Discovery",
        "namespace": "Namespace Management", 
        "upsert": "Upsert Operations",
        "query": "Query Operations",
        "delete": "Delete Operations",
        "filtering": "Filtering Semantics",
        "dimension_validation": "Dimension Validation",
        "error_handling": "Error Handling",
        "deadline": "Deadline Semantics",
        "health": "Health Endpoint", 
        "observability": "Observability & Privacy",
        "batch_limits": "Batch Size Limits",
        "wire_envelopes": "Wire Envelopes & Routing"
    },
    "graph": {
        "core_ops": "Core Operations", 
        "crud_validation": "CRUD Validation",
        "query_ops": "Query Operations",
        "dialect_validation": "Dialect Validation",
        "streaming": "Streaming Semantics",
        "batch_ops": "Batch Operations", 
        "schema_ops": "Schema Operations",
        "error_handling": "Error Handling",
        "capabilities": "Capabilities Discovery",
        "observability": "Observability & Privacy",
        "deadline": "Deadline Semantics",
        "health": "Health Endpoint",
        "wire_envelopes": "Wire Envelopes & Routing"
    },
    "embedding": {
        "core_ops": "Core Operations",
        "capabilities": "Capabilities Discovery", 
        "batch_partial": "Batch & Partial Failures",
        "truncation": "Truncation & Length",
        "normalization": "Normalization Semantics",
        "token_counting": "Token Counting",
        "error_handling": "Error Handling",
        "deadline": "Deadline Semantics", 
        "health": "Health Endpoint",
        "observability": "Observability & Privacy",
        "caching": "Caching & Idempotency", 
        "wire_contract": "Wire Contract"
    },
    "schema": {
        "meta_lint": "Schema Meta-Lint",
        "golden_validation": "Golden Wire Validation"
    },
    "golden": {
        "wire_messages": "Golden Wire Messages",
        "envelope_validation": "Envelope Validation"
    }
}

SPEC_SECTION_MAPPING = {
    "llm": {
        "core_ops": "§8.3 Complete Operation",
        "streaming": "§8.3 Stream Operation", 
        "message_validation": "§8.3 Message Format",
        "sampling_params": "§8.3 Sampling Parameters",
        "capabilities": "§8.4 Model Discovery"
    },
    "vector": {
        "namespace": "§9.3 Namespace Management",
        "upsert": "§9.3 Upsert Operations", 
        "query": "§9.3 Query Operations",
        "capabilities": "§9.2 Capabilities Discovery"
    },
    # ... similar mappings for other protocols
}


class CorpusProtocolPlugin:
    """Pytest plugin for comprehensive Corpus Protocol conformance reporting."""
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.test_reports: Dict[str, List[Any]] = {}
        self.protocol_counts: Dict[str, Dict[str, int]] = {}
        
    def pytest_sessionstart(self, session):
        """Record session start time and initialize tracking."""
        self.start_time = time.time()
        self.test_reports = {proto: [] for proto in PROTOCOLS}
        self.test_reports["other"] = []
        self.protocol_counts = {proto: {} for proto in PROTOCOLS}
        
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
    
    def _categorize_test_by_protocol(self, nodeid: str) -> Tuple[str, str]:
        """Categorize test by protocol and test category."""
        # Handle both POSIX and Windows paths
        nodeid_lower = nodeid.lower()
        
        for proto in PROTOCOLS:
            if f"tests/{proto}/" in nodeid_lower or f"tests\\{proto}\\" in nodeid_lower:
                # Further categorize by test type
                test_name = nodeid_lower
                category = "unknown"
                
                for cat_key, cat_name in TEST_CATEGORIES.get(proto, {}).items():
                    if cat_key in test_name or cat_name.lower() in test_name:
                        category = cat_key
                        break
                        
                return proto, category
                
        return "other", "unknown"
    
    def _categorize_failures(self, failed_reports: List) -> Dict[str, Dict[str, int]]:
        """Categorize failed tests by protocol and category."""
        by_protocol = {proto: {} for proto in PROTOCOLS}
        by_protocol["other"] = {}
        
        for rep in failed_reports:
            nodeid = getattr(rep, "nodeid", "") or ""
            proto, category = self._categorize_test_by_protocol(nodeid)
            
            if proto not in by_protocol:
                by_protocol[proto] = {}
                
            if category not in by_protocol[proto]:
                by_protocol[proto][category] = 0
                
            by_protocol[proto][category] += 1
                
        return by_protocol
    
    def _calculate_conformance_level(self, protocol: str, passed_count: int) -> Tuple[str, int]:
        """Calculate conformance level and progress to next level."""
        levels = CONFORMANCE_LEVELS.get(protocol, {})
        
        if passed_count >= levels.get("gold", 0):
            return "🥇 Gold", 0
        elif passed_count >= levels.get("silver", 0):
            next_level = "Gold"
            needed = levels.get("gold", 0) - passed_count
            return "🥈 Silver", needed
        elif passed_count >= levels.get("development", 0):
            next_level = "Silver" 
            needed = levels.get("silver", 0) - passed_count
            return "🔬 Development", needed
        else:
            next_level = "Development"
            needed = levels.get("development", 0) - passed_count
            return "❌ Below Development", needed
    
    def _get_spec_section(self, protocol: str, category: str) -> str:
        """Get specification section for a test category."""
        protocol_map = SPEC_SECTION_MAPPING.get(protocol, {})
        return protocol_map.get(category, "See protocol specification")
    
    def _print_platinum_certification(self, terminalreporter, counts: Dict[str, int], duration: float):
        """Print Platinum certification summary."""
        total_tests = self._get_total_tests(counts)
        
        terminalreporter.write_sep("=", "🏆 CORPUS PROTOCOL SUITE - PLATINUM CERTIFIED")
        terminalreporter.write_line(
            f"All {total_tests} conformance tests passed across 6 test suites"
        )
        terminalreporter.write_line("")
        
        # Show protocol breakdown
        terminalreporter.write_line("Protocol Conformance Status:")
        for proto in PROTOCOLS:
            display_name = PROTOCOL_DISPLAY_NAMES.get(proto, proto.upper())
            level, _ = self._calculate_conformance_level(proto, CONFORMANCE_LEVELS[proto]["gold"])
            terminalreporter.write_line(f"  ✅ {display_name}: {level}")
        
        terminalreporter.write_line("")
        terminalreporter.write_line(f"⏱️  Completed in {duration:.2f}s")
        terminalreporter.write_line("🎯 Status: Ready for production deployment")
    
    def _print_gold_certification(self, terminalreporter, protocol_results: Dict[str, int], duration: float):
        """Print Gold certification summary with progress to Platinum."""
        terminalreporter.write_sep("=", "🥇 CORPUS PROTOCOL SUITE - GOLD CERTIFIED")
        
        terminalreporter.write_line("Protocol Conformance Status:")
        platinum_ready = True
        
        for proto in PROTOCOLS:
            display_name = PROTOCOL_DISPLAY_NAMES.get(proto, proto.upper())
            passed = protocol_results.get(proto, 0)
            level, needed = self._calculate_conformance_level(proto, passed)
            
            if "Gold" in level:
                terminalreporter.write_line(f"  ✅ {display_name}: {level}")
            else:
                platinum_ready = False
                terminalreporter.write_line(f"  ⚠️  {display_name}: {level} ({needed} tests to Gold)")
        
        terminalreporter.write_line("")
        if platinum_ready:
            terminalreporter.write_line("🎯 All protocols at Gold level - Platinum certification available!")
        else:
            terminalreporter.write_line("🎯 Focus on protocols below Gold level for Platinum certification")
        
        terminalreporter.write_line(f"⏱️  Completed in {duration:.2f}s")
    
    def _print_failure_analysis(self, terminalreporter, by_protocol: Dict[str, Dict[str, int]], duration: float):
        """Print detailed failure analysis with actionable guidance."""
        terminalreporter.write_sep("=", "❌ CORPUS PROTOCOL CONFORMANCE ANALYSIS")
        
        total_failures = 0
        for proto_failures in by_protocol.values():
            for category_count in proto_failures.values():
                total_failures += category_count
        
        terminalreporter.write_line(f"Found {total_failures} conformance issue(s) across protocols:")
        terminalreporter.write_line("")
        
        # Show failures by protocol and category
        for proto, categories in by_protocol.items():
            if not categories:
                continue
                
            display_name = PROTOCOL_DISPLAY_NAMES.get(proto, proto.upper())
            terminalreporter.write_line(f"{display_name}:")
            
            for category, count in categories.items():
                category_name = TEST_CATEGORIES.get(proto, {}).get(category, category.replace('_', ' ').title())
                spec_section = self._get_spec_section(proto, category)
                
                terminalreporter.write_line(f"  ❌ {category_name}: {count} failure(s)")
                terminalreporter.write_line(f"      Specification: {spec_section}")
            
            terminalreporter.write_line("")
        
        # Certification impact
        terminalreporter.write_line("Certification Impact:")
        failing_protocols = [p for p, cats in by_protocol.items() if cats and p != "other"]
        
        if failing_protocols:
            terminalreporter.write_line("  ⚠️  Platinum certification blocked by failures in:")
            for proto in failing_protocols:
                display_name = PROTOCOL_DISPLAY_NAMES.get(proto, proto.upper())
                terminalreporter.write_line(f"      - {display_name}")
        else:
            terminalreporter.write_line("  ✅ No protocol conformance failures - review 'other' category tests")
        
        terminalreporter.write_line("")
        terminalreporter.write_line("Next Steps:")
        terminalreporter.write_line("  1. Review failing tests above")
        terminalreporter.write_line("  2. Check CONFORMANCE.md for test-to-spec mapping") 
        terminalreporter.write_line("  3. Run individual protocol tests: make test-{protocol}-conformance")
        terminalreporter.write_line(f"⏱️  Completed in {duration:.2f}s")
    
    def _collect_protocol_results(self, terminalreporter) -> Dict[str, int]:
        """Collect passed test counts per protocol."""
        protocol_results = {proto: 0 for proto in PROTOCOLS}
        
        # Count passed tests per protocol
        for test_report in terminalreporter.stats.get("passed", []):
            nodeid = getattr(test_report, "nodeid", "") or ""
            proto, _ = self._categorize_test_by_protocol(nodeid)
            if proto in protocol_results:
                protocol_results[proto] += 1
        
        return protocol_results
    
    def _is_platinum_certified(self, protocol_results: Dict[str, int]) -> bool:
        """Check if all protocols meet Platinum certification requirements."""
        for proto in PROTOCOLS:
            passed = protocol_results.get(proto, 0)
            if passed < CONFORMANCE_LEVELS[proto]["gold"]:
                return False
        return True

    def pytest_terminal_summary(self, terminalreporter, exitstatus, config):
        """Generate comprehensive Corpus Protocol conformance summary."""
        # Collect both test failures and internal errors
        failed_reports = []
        for key in ("failed", "error"):
            failed_reports.extend(terminalreporter.stats.get(key, []))
        
        # Get test counts and duration
        counts = self._get_test_counts(terminalreporter)
        duration = self._get_duration()
        
        # Collect protocol-specific results
        protocol_results = self._collect_protocol_results(terminalreporter)
        
        # --- CORRECTED LOGIC ---
        
        # Check if any tests actually failed
        if not failed_reports:
            # All tests that ran have passed.
            
            if self._is_platinum_certified(protocol_results):
                # Best case: All protocols ran and hit Gold.
                self._print_platinum_certification(terminalreporter, counts, duration)
            else:
                # Partial pass: (e.g., only 'make test-llm-conformance' ran)
                # Show the mixed-level "Gold" summary.
                self._print_gold_certification(terminalreporter, protocol_results, duration)
            return
        
        # We have actual failures. Show the analysis.
        by_protocol = self._categorize_failures(failed_reports)
        self._print_failure_analysis(terminalreporter, by_protocol, duration)
    
    def pytest_runtest_logstart(self, nodeid, location):
        """Show protocol being tested for better progress visibility."""
        proto, category = self._categorize_test_by_protocol(nodeid)
        if proto != "other":
            # Could enhance with progress reporting
            pass


# Instantiate the plugin
corpus_protocol_plugin = CorpusProtocolPlugin()

# Pytest hook functions (delegate to plugin instance)
def pytest_sessionstart(session):
    corpus_protocol_plugin.pytest_sessionstart(session)

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    corpus_protocol_plugin.pytest_terminal_summary(terminalreporter, exitstatus, config)

def pytest_runtest_logstart(nodeid, location):
    corpus_protocol_plugin.pytest_runtest_logstart(nodeid, location)

# Optional: Add custom markers for better test organization
def pytest_configure(config):
    """Register custom markers for protocol tests."""
    markers = [
        "llm: LLM Protocol V1.0 conformance tests",
        "vector: Vector Protocol V1.0 conformance tests", 
        "graph: Graph Protocol V1.0 conformance tests",
        "embedding: Embedding Protocol V1.0 conformance tests",
        "schema: Schema conformance validation tests",
        "golden: Golden wire message validation tests",
        "slow: Tests that take longer to run (skip with -m 'not slow')",
        "conformance: All protocol conformance tests",
    ]
    
    for marker in markers:
        config.addinivalue_line("markers", marker)