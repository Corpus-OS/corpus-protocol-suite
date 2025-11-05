# SPDX-License-Identifier: Apache-2.0
"""
Tiny, dependency-free pretty-print helpers for examples.

Includes:
  • print_kv:     aligned key/value output
  • print_json:   compact or pretty JSON
  • print_table:  fixed-width ASCII table (auto-fit / truncate)
  • box:          simple boxed section headers
"""
from __future__ import annotations

import json
import shutil
from typing import Any, Iterable, List, Mapping, Sequence

__all__ = ["print_kv", "print_json", "print_table", "box"]

def _term_width(default: int = 100) -> int:
    try:
        cols = shutil.get_terminal_size((default, 20)).columns
    except Exception:
        cols = default
    return max(40, min(cols, 200))

def box(title: str, *, fill: str = "─") -> None:
    """Print a single-line boxed title."""
    width = _term_width()
    title = f" {title.strip()} "
    pad = max(0, width - len(title) - 2)
    print(f"┌{fill* (len(title))}┐")
    print(f"│{title}│")
    print(f"└{fill* (len(title))}┘")

def print_kv(pairs: Mapping[str, Any] | Sequence[tuple[str, Any]], *, indent: int = 2) -> None:
    """Print aligned key/value pairs."""
    if isinstance(pairs, Mapping):
        items = list(pairs.items())
    else:
        items = list(pairs)
    if not items:
        return
    k_width = max(len(str(k)) for k, _ in items)
    for k, v in items:
        key = str(k).rjust(k_width)
        print(" " * indent + f"{key}: {v}")

def print_json(obj: Any, *, pretty: bool = True, indent: int = 2) -> None:
    """Print JSON safely (ensuring ASCII where helpful)."""
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
    Print a simple fixed-width table. Works with dict-rows or list/tuple rows.

    If dict-rows are provided and headers=None, headers are inferred (sorted keys).
    """
    data: List[List[str]] = []
    if headers is None:
        rows = list(rows)
        if not rows:
            return
        first = rows[0]
        if isinstance(first, Mapping):
            headers = sorted(first.keys())
            for r in rows:
                data.append([_to_str(r.get(h, "")) for h in headers])
        else:
            headers = [f"col{i}" for i in range(len(first))]  # type: ignore[arg-type]
            for r in rows:  # type: ignore[assignment]
                data.append([_to_str(x) for x in r])  # type: ignore[index]
    else:
        for r in rows:
            if isinstance(r, Mapping):
                data.append([_to_str(r.get(h, "")) for h in headers])
            else:
                data.append([_to_str(x) for x in r])  # type: ignore[index]

    cols = len(headers)
    col_widths = [max(len(h), *(len(row[i]) for row in data)) for i, h in enumerate(headers)]
    width_limit = max_width or _term_width()
    table_width = sum(col_widths) + 3 * (cols - 1)
    if table_width > width_limit:
        scale = (width_limit - 3 * (cols - 1)) / max(1, sum(col_widths))
        col_widths = [max(4, int(w * scale)) for w in col_widths]

    def fit(cell: str, w: int) -> str:
        if len(cell) <= w:
            return cell.ljust(w)
        if w <= 1:
            return cell[:w]
        return cell[: max(0, w - len(truncate_marker))] + truncate_marker

    header_line = " | ".join(fit(h, col_widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * col_widths[i] for i in range(cols))
    print(header_line)
    print(sep_line)
    for row in data:
        print(" | ".join(fit(row[i], col_widths[i]) for i in range(cols)))

def _to_str(x: Any) -> str:
    try:
        if isinstance(x, (dict, list, tuple)):
            return json.dumps(x, ensure_ascii=False)
        return str(x)
    except Exception:
        return "<unprintable>"

