# SPDX-License-Identifier: Apache-2.0
"""
A simple console MetricsSink for examples and local debugging.

Implements the same shape used by the SDK bases:
  - observe(component, op, ms, ok, code="OK", extra=None)
  - counter(component, name, value=1, extra=None)
  - gauge(component, name, value, extra=None)

Features:
  • Thread-safe printing
  • Optional ANSI colors
  • Low-cardinality guarding of 'extra'
  • Human-readable, machine-parseable lines
  • Input validation and safety
"""

from __future__ import annotations

import json
import sys
import threading
import time
from typing import Any, Mapping, Optional, TextIO

__all__ = ["ConsoleMetrics"]

_LOCK = threading.Lock()
_JSON_ENCODER = json.JSONEncoder(separators=(",", ":"), ensure_ascii=False, check_circular=False)


class ConsoleMetrics:
    """
    Drop-in example metrics sink that prints structured lines to stdout.

    Args:
        colored: Enable ANSI colors.
        flush:   Force flush() on each write (useful in CI).
        name:    Optional instance name to include in lines.
        output_file: File-like object to write to (default: stdout).
        max_extra_fields: Maximum number of extra fields to include.
    """

    def __init__(
        self, 
        *, 
        colored: bool = True, 
        flush: bool = True, 
        name: Optional[str] = None,
        output_file: Optional[TextIO] = None,
        max_extra_fields: int = 10
    ) -> None:
        self.colored = colored and (output_file or sys.stdout).isatty()
        self.flush = flush
        self.name = name
        self.output_file = output_file or sys.stdout
        self.max_extra_fields = max_extra_fields

    # ------------------------------------------------------------------
    # Public API matching the Corpus SDK Metrics Protocol
    # ------------------------------------------------------------------

    def observe(
        self,
        *,
        component: str,
        op: str,
        ms: float,
        ok: bool,
        code: str = "OK",
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Record an observation (typically latency).
        """
        # Input validation
        if not component or not op:
            return
            
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "component": component,
            "op": op,
            "ms": round(max(0.0, float(ms)), 3),
            "ok": bool(ok),
            "code": str(code or "OK"),
            **({"instance": self.name} if self.name else {}),
        }
        
        safe_extra = self._safe_extra(extra)
        if safe_extra:
            payload["extra"] = safe_extra

        self._write(self._format_line("OBS", payload, ok))

    def counter(
        self,
        *,
        component: str,
        name: str,
        value: int = 1,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Increment a counter by `value`.
        """
        if not component or not name:
            return
            
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "component": component,
            "name": name,
            "value": max(0, int(value)),
            **({"instance": self.name} if self.name else {}),
        }
        
        safe_extra = self._safe_extra(extra)
        if safe_extra:
            payload["extra"] = safe_extra

        self._write(self._format_line("CTR", payload, True))

    def gauge(
        self,
        *,
        component: str,
        name: str,
        value: float,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Record a gauge value.
        """
        if not component or not name:
            return
            
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "component": component,
            "name": name,
            "value": float(value),
            "type": "gauge",
            **({"instance": self.name} if self.name else {}),
        }
        
        safe_extra = self._safe_extra(extra)
        if safe_extra:
            payload["extra"] = safe_extra

        self._write(self._format_line("GAU", payload, True))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _write(self, s: str) -> None:
        """Thread-safe write to output."""
        with _LOCK:
            print(s, file=self.output_file, flush=self.flush)

    def _format_line(self, kind: str, payload: Mapping[str, Any], ok: bool) -> str:
        """Format structured log line with optional color."""
        prefix = self._color(kind, ok)
        body = _JSON_ENCODER.encode(payload)
        return f"{prefix} {body}"

    def _color(self, kind: str, ok: bool) -> str:
        """Return ANSI-colored prefix if enabled."""
        if not self.colored:
            return f"[{kind}]"
        
        colors = {
            "CTR": "\x1b[36m",  # cyan
            "GAU": "\x1b[35m",  # magenta
            "OBS": "\x1b[32m" if ok else "\x1b[31m"  # green/red
        }
        color_code = colors.get(kind, "\x1b[0m")
        return f"{color_code}[{kind}]\x1b[0m"

    def _safe_extra(self, extra: Optional[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
        """Create a safe, low-cardinality version of extra fields."""
        if not extra:
            return None
            
        safe_extra = {}
        for i, (k, v) in enumerate(sorted(extra.items())):
            if i >= self.max_extra_fields:
                break
                
            if not isinstance(k, str) or len(k) > 100:
                continue
                
            # Only allow simple, low-cardinality values
            if v is None or isinstance(v, (str, int, float, bool)):
                str_val = str(v)
                if len(str_val) <= 1000:  # Limit value size
                    safe_extra[k] = v
                    
        return safe_extra or None
