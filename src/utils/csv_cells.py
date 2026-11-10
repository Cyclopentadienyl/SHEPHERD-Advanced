"""
CSV cells: the one set of rules every CSV this project writes follows.
=====================================================================
Research comparison §5 Stage 8 ("one serializer for CSV and JSON; the
Diagnosis CSV stays byte-identical"). The rules were the Diagnosis export's
(`src/webui/components/diagnosis_panel.py`), and the results export
(`src/api/services/results.py`) writes its cells by the same functions, so
two CSVs of this project never defuse, prefix or spell a value differently:

- **A text cell a spreadsheet would run as a formula is defused** (OWASP,
  "CSV Injection"): one that starts with ``=``, ``+``, ``-``, ``@``, a tab or a
  carriage return gets a leading apostrophe (`text_cell`). A number written as
  a number never passes through it, so a negative number stays a number.
- **A SHA-256 is written ``sha256:<hex>``** (`digest_cell`): the prefix names
  the algorithm and stops a spreadsheet reading an all-digit digest as a
  number.
- **A JSON scalar as text** (`plain_text`): ``true``/``false`` for booleans,
  numbers as Python prints them (``0.7``, ``1.0``), unrounded.
- **A value that is absent is a word, never an empty cell** (`value_cell`):
  the caller names the word for null and for an empty text.

Standard library only, no I/O.

Module: src/utils/csv_cells.py
"""
from __future__ import annotations

import re
from typing import Any, Optional

#: A cell a spreadsheet would run as a formula starts with one of these (OWASP,
#: "CSV Injection").
FORMULA_START = ("=", "+", "-", "@", "\t", "\r")

#: A SHA-256 as the records hold it: 64 lowercase hexadecimal digits.
_SHA256 = re.compile(r"[0-9a-f]{64}")


def is_sha256(value: Any) -> bool:
    """Whether `value` is a SHA-256 as the records hold it."""
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def plain_text(value: Any) -> str:
    """A JSON scalar as text: ``true``/``false`` for booleans, floats as Python
    prints them (``0.7``, ``1.0``)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def text_cell(value: Any) -> str:
    """A free-text cell, defused: one that a spreadsheet would run as a formula
    gets a leading apostrophe. None is an empty text."""
    text = "" if value is None else str(value)
    return "'" + text if text.startswith(FORMULA_START) else text


def digest_cell(value: Any) -> Optional[str]:
    """A SHA-256 as a cell, ``sha256:<hex>``; None for anything else, which the
    caller writes by its own rule."""
    return f"sha256:{value}" if is_sha256(value) else None


def value_cell(value: Any, *, null_word: str, empty_word: str) -> str:
    """One JSON scalar as a cell, never empty: `null_word` for null,
    `empty_word` for an empty text, a text defused (`text_cell`), anything
    else as `plain_text` (a number unrounded, a boolean as ``true``/``false``)."""
    if value is None:
        return null_word
    if isinstance(value, str):
        return text_cell(value) if value else empty_word
    return plain_text(value)


__all__ = ["FORMULA_START", "digest_cell", "is_sha256", "plain_text", "text_cell", "value_cell"]
