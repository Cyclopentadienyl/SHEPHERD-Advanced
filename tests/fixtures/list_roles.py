"""
The ARIA list structure of a rendered fragment.
===============================================
For the WebUI's list pins (the WebUI skill's §10, correct roles): each
``role="list"`` element of an HTML fragment, its ``aria-label`` and its direct
children (their role, class and text), so a test checks that a list holds list
items only and counts its entries, never its headings or notes.

Module: tests/fixtures/list_roles.py
"""
from __future__ import annotations

from html.parser import HTMLParser
from typing import Any, Dict, List

_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source",
         "track", "wbr"}


class _Tree(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root: Dict[str, Any] = {"tag": None, "attrs": {}, "children": [], "text": ""}
        self.stack = [self.root]

    def handle_starttag(self, tag: str, attrs: List[Any]) -> None:
        node = {"tag": tag, "attrs": dict(attrs), "children": [], "text": ""}
        self.stack[-1]["children"].append(node)
        if tag not in _VOID:
            self.stack.append(node)

    def handle_endtag(self, tag: str) -> None:
        for depth in range(len(self.stack) - 1, 0, -1):
            if self.stack[depth]["tag"] == tag:
                del self.stack[depth:]
                return

    def handle_data(self, data: str) -> None:
        for node in self.stack:
            node["text"] += data


def lists(html: str) -> List[Dict[str, Any]]:
    """Each ``role="list"`` element in document order, as ``{"label": its
    aria-label, "children": [{"role", "class", "text"}, ...]}`` (its direct
    children only)."""
    tree = _Tree()
    tree.feed(html)
    found: List[Dict[str, Any]] = []

    def walk(node: Dict[str, Any]) -> None:
        if node["attrs"].get("role") == "list":
            found.append({"label": node["attrs"].get("aria-label"),
                          "children": [{"role": child["attrs"].get("role"),
                                        "class": child["attrs"].get("class") or "",
                                        "text": " ".join(child["text"].split())}
                                       for child in node["children"]]})
        for child in node["children"]:
            walk(child)

    walk(tree.root)
    return found


__all__ = ["lists"]
