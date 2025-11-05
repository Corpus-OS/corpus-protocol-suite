# SPDX-License-Identifier: Apache-2.0
"""
Tiny, dependency-free pretty-print helpers for examples.

Includes:
  • box          — simple boxed section headers
  • print_kv     — aligned key/value output
  • print_json   — compact or pretty JSON
  • print_table  — fixed-width ASCII table (auto-fit / truncate)
"""
from __future__ import annotations

import json
import shutil
from typing import Any, Iterable, List, Mapping, Sequence

__all__ = ["box", "print_kv", "print_json", "print_table"]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _term_width(default: int = 100) -> int:
    """Detect terminal width with a safe fallback."""
    try:
        cols = shutil.get_terminal_size((default, 20)).columns
    except Exception:
        cols = default
    return max(40, min(cols, 200))


def _to_str(x: Any) -> str:
    """Convert any value to a safe string."""
    try:
        if isinstance(x, (dict, list, tuple)):
            return json.dumps(x, ensure_ascii=False)
        return str(x)
    except Exception:
        return "<unprintable>"


# ----------------------------------------------------------------------
# Public functions
# ----------------------------------------------------------------------

def box(title: str, *, fill: str = "─") -> None:
    """Print a single-line boxed title."""
    width = _term_width()
    title = f" {title.strip()} "
    bar = fill * min(len(title), width - 4)
    print(f"\n┌{bar}┐")
    print(f"│{title.center(width - 2)}│")
    print(f"└{bar}┘\n")


def print_kv(
    pairs: Mapping[str, Any] | Sequence[tuple[str, Any]],
    *,
    indent: int = 2,
) -> None:
    """Print aligned key/value pairs."""
    items = list(pairs.items()) if isinstance(pairs, Mapping) else list(pairs)
    if not items:
        return
    k_width = max(len(str(k)) for k, _ in items)
    for k, v in items:
        print(" " * indent + f"{str(k).rjust(k_width)}: {v}")


def print_json(obj: Any, *, pretty: bool = True, indent: int = 2) -> None:
    """Print an object as JSON (compact or pretty)."""
    if pretty:
        s = json.dumps(obj, indent=indent, ensure_ascii=False)
    else:
        s = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
    print(s)


def print_table(
    rows: Iterable[Mapping[str, Any]] | Iterable[Sequence[Any]],
    headers: Sequence[str] | None = None,
    *,
    max_width: int | None = None,
    truncate_marker: str = "…",
) -> None:
    """
    Print a fixed-width ASCII table.

    Works with dict or sequence rows. Automatically infers headers if not provided.
    """
    data: List[List[str]] = []
    rows = list(rows)
    if not rows:
        return

    # infer headers if needed
    if headers is None:
        first = rows[0]
        if isinstance(first, Mapping):
            headers = sorted(first.keys())
            data = [[_to_str(r.get(h, "")) for h in headers] for r in rows]
        else:
            headers = [f"col{i}" for i in range(len(first))]
            data = [[_to_str(x) for x in r] for r in rows]
    else:
        for r in rows:
            if isinstance(r, Mapping):
                data.append([_to_str(r.get(h, "")) for h in headers])
            else:
                data.append([_to_str(x) for x in r])

    # determine column widths
    cols = len(headers)
    col_widths = [max(len(h), *(len(row[i]) for row in data)) for i, h in enumerate(headers)]
    width_limit = max_width or _term_width()
    total_width = sum(col_widths) + 3 * (cols - 1)
    if total_width > width_limit:
        scale = (width_limit - 3 * (cols - 1)) / max(1, sum(col_widths))
        col_widths = [max(4, int(w * scale)) for w in col_widths]

    # fit function
    def fit(cell: str, w: int) -> str:
        if len(cell) <= w:
            return cell.ljust(w)
        if w <= len(truncate_marker):
            return truncate_marker[:w]
        return cell[: w - len(truncate_marker)] + truncate_marker

    # render
    header_line = " | ".join(fit(h, col_widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * col_widths[i] for i in range(cols))
    print(header_line)
    print(sep_line)
    for row in data:
        print(" | ".join(fit(row[i], col_widths[i]) for i in range(cols)))
