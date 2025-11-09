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

def box(title: str, *, fill: str = "─", color: str | None = None) -> None:
    """
    Print a single-line boxed title.
    
    Args:
        title: Text to display in the box
        fill: Character to use for the horizontal line
        color: Optional ANSI color code (e.g., '36' for cyan)
    """
    width = _term_width()
    title = f" {title.strip()} "
    bar = fill * min(len(title), width - 4)
    
    # Apply color if specified
    color_start = f"\033[{color}m" if color else ""
    color_end = "\033[0m" if color else ""
    
    print(f"\n{color_start}┌{bar}┐{color_end}")
    print(f"{color_start}│{title.center(width - 2)}│{color_end}")
    print(f"{color_start}└{bar}┘{color_end}\n")


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


def print_json(
    obj: Any, 
    *, 
    pretty: bool = True, 
    indent: int = 2,
    compact_arrays: bool = False
) -> None:
    """
    Print an object as JSON (compact or pretty).
    
    Args:
        obj: Object to serialize as JSON
        pretty: Whether to use pretty-printing
        indent: Indentation level for pretty-printing
        compact_arrays: Format arrays on single line when possible
    """
    if compact_arrays and isinstance(obj, (list, tuple)) and pretty:
        # Compact array formatting for better readability
        if all(not isinstance(x, (dict, list, tuple)) for x in obj):
            compact_json = json.dumps(obj, ensure_ascii=False)
            print(compact_json)
            return
    
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
    preserve_order: bool = True,
) -> None:
    """
    Print a fixed-width ASCII table.

    Works with dict or sequence rows. Automatically infers headers if not provided.
    
    Args:
        rows: Data rows as either mappings or sequences
        headers: Column headers (inferred if None)
        max_width: Maximum table width (defaults to terminal width)
        truncate_marker: String to indicate truncated content
        preserve_order: For dict rows, maintain key order instead of sorting
    """
    data: List[List[str]] = []
    rows = list(rows)
    if not rows:
        return

    # infer headers if needed
    if headers is None:
        first = rows[0]
        if isinstance(first, Mapping):
            # Preserve insertion order by default, fall back to sorted
            headers = list(first.keys()) if preserve_order and hasattr(first, 'keys') else sorted(first.keys())
            data = [[_to_str(r.get(h, "")) for h in headers] for r in rows]
        else:
            headers = [f"col{i}" for i in range(len(first))]
            data = [[_to_str(x) for x in r] for r in rows]
    else:
        headers = list(headers)
        expected_cols = len(headers)
        for r in rows:
            if isinstance(r, Mapping):
                data.append([_to_str(r.get(h, "")) for h in headers])
            else:
                # Ensure consistent column count for sequence rows
                row_data = [_to_str(x) for x in r]
                if len(row_data) < expected_cols:
                    row_data.extend([""] * (expected_cols - len(row_data)))
                elif len(row_data) > expected_cols:
                    row_data = row_data[:expected_cols]
                data.append(row_data)

    # Ensure we have data to display
    if not data:
        return

    # determine column widths
    cols = len(headers)
    col_widths = [max(len(h), *(len(row[i]) for row in data if i < len(row))) for i, h in enumerate(headers)]
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
        print(" | ".join(fit(row[i] if i < len(row) else "", col_widths[i]) for i in range(cols)))


# ----------------------------------------------------------------------
# Usage examples
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Demo the functionality
    box("Demo Output", color="36")  # Cyan color
    
    print("Key-Value Pairs:")
    print_kv([("Name", "Alice"), ("Age", 30), ("City", "New York")])
    
    print("\nJSON Output:")
    data = {
        "users": [
            {"id": 1, "name": "Alice", "tags": ["admin", "user"]},
            {"id": 2, "name": "Bob", "tags": ["user"]}
        ],
        "status": "success"
    }
    print_json(data, pretty=True)
    
    print("\nCompact Arrays:")
    print_json([1, 2, 3, 4, 5], compact_arrays=True)
    
    print("\nTable Output:")
    users = [
        {"id": 1, "name": "Alice Johnson", "email": "alice@example.com", "role": "Administrator"},
        {"id": 2, "name": "Bob Smith", "email": "bob.smith@company.org", "role": "User"},
        {"id": 3, "name": "Charlie Brown", "email": "charlie@test.io", "role": "Moderator"},
    ]
    print_table(users, headers=["ID", "Name", "Email", "Role"])
    
    print("\nSequence Rows:")
    data_rows = [
        [1, "Short", "This is a medium length description"],
        [2, "Longer title here", "Short"],
        [3, "Medium", "A very long description that will likely get truncated in most terminals"]
    ]
    print_table(data_rows, headers=["ID", "Title", "Description"])
