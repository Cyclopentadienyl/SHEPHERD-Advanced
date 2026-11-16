"""
The R10 table as a test reads it (`renderers.r10_table_html`).
================================================================
`r10_cells` parses the one R10 rendering the Test page's result, Models'
recorded results and Models › Results' Compare share into its cells, so a test
checks a cell's mark, words, count and reason without depending on the page
around it.

Module: tests/fixtures/r10_table.py
"""
from __future__ import annotations

import html as html_lib
import re
from typing import Dict


def cell_text(markup: str) -> str:
    """A cell's words as a reader sees them: each tag a space."""
    return " ".join(html_lib.unescape(re.sub(r"<[^>]+>", " ", markup)).split())


def r10_cells(markup: str) -> Dict[str, Dict[str, str]]:
    """The first R10 table in `markup`, by row head (Training, Validation):
    each row's ``evidence`` words (under its head) and each column's cell as
    HTML, by its column head (Disease overlap, Phenotype-set overlap)."""
    table = re.search(r'<table class="shep-r10"[^>]*>(.*?)</table>', markup, re.S)
    assert table, "no R10 table"
    heads = [cell_text(h) for h in re.findall(r'<th scope="col">(.*?)</th>', table.group(1))]
    rows: Dict[str, Dict[str, str]] = {}
    for head, cells in re.findall(r'<tr><th scope="row">(.*?)</th>(.*?)</tr>', table.group(1),
                                  re.S):
        name, evidence = re.match(r'(.*?)<div class="shep-r10-n">(.*?)</div>', head).groups()
        row = {"evidence": cell_text(evidence)}
        row.update(zip(heads[1:], re.findall(r"<td>(.*?)</td>", cells, re.S)))
        rows[cell_text(name)] = row
    return rows


def without_r10(markup: str) -> str:
    """`markup` without its R10 tables: an overlap's amber ▲ and its column
    heads ("Disease overlap") are R10's own, not another state's or a
    mode's words."""
    return re.sub(r'<table class="shep-r10".*?</table>', "", markup, flags=re.S)


__all__ = ["cell_text", "r10_cells", "without_r10"]
