# SPDX-License-Identifier: Apache-2.0
"""
A simple console MetricsSink for examples and local debugging.

Implements the same shape used by the SDK bases:
  - observe(component, op, ms, ok, code="OK", extra=None)
  - counter(component, name, value=1, extra=None)

Features:
  • Thread-safe printing
  • Optional ANSI colors
  • Low-cardinality guarding of 'extra'
  • Human-readable, machine-parseable lines
"""

from __future__ import annotations

import json
import sys
import threading
import time
from typing import Any, Mapping, Optional

__all__ = ["ConsoleMetrics"]

_LOCK = threading.Lock()


class ConsoleMetrics:
    """
    Drop-in example metrics sink that prints structured lines to stdout.

    Args:
        colored: Enable ANSI colors.
        flush:   Force flush() on each write (useful in CI).
        name:    Optional instance name to include in lines.
    """

    def __init__(self, *, colored: bool = True, flush: bool = True, name: Optional[str] = None) -> None:
        self.colored = colored and sys.stdout.isatty()
        self.flush = flush
        self.name = name

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
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "component": component,
            "op": op,
            "ms": round(float(ms), 3),
            "ok": bool(ok),
            "code": str(code),
            **({"instance": self.name} if self.name else {}),
        }
        if extra:
            # Sort and filter for low cardinality safety
            payload["extra"] = {k: extra[k] for k in sorted(extra.keys()) if isinstance(k, str)}

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
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "component": component,
            "name": name,
            "value": int(value),
            **({"instance": self.name} if self.name else {}),
        }
        if extra:
            payload["extra"] = {k: extra[k] for k in sorted(extra.keys()) if isinstance(k, str)}

        self._write(self._format_line("CTR", payload, True))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _write(self, s: str) -> None:
        """Thread-safe write to stdout."""
        with _LOCK:
            print(s, file=sys.stdout, flush=self.flush)

    def _format_line(self, kind: str, payload: Mapping[str, Any], ok: bool) -> str:
        """Format structured log line with optional color."""
        prefix = self._color(kind, ok)
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        return f"{prefix} {body}"

    def _color(self, kind: str, ok: bool) -> str:
        """Return ANSI-colored prefix if enabled."""
        if not self.colored:
            return f"[{kind}]"
        if kind == "CTR":
            return "\x1b[36m[CTR]\x1b[0m"  # cyan
        return ("\x1b[32m" if ok else "\x1b[31m") + f"[{kind}]" + "\x1b[0m"
