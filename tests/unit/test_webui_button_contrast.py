"""
Primary buttons keep their words readable (M5-2): text contrast at least 4.5:1.
==============================================================================
The dashboard's theme (Gradio's Soft) draws a primary button white on indigo-500
(4.47:1) and, on hover and when disabled, on indigo-400 (2.98:1, measured on
Run Diagnosis). The shared stylesheet (`src/webui/style.py`) sets the primary
button's colours from the dashboard's own tokens, and draws a disabled primary
button grey with grey text. These tests compute the contrast (WCAG 2.x) of what
a primary button actually gets: the stylesheet's value where it sets one, the
theme's otherwise, so removing the stylesheet's rules fails them.

Module: tests/unit/test_webui_button_contrast.py
"""
from __future__ import annotations

import re

import pytest

gr = pytest.importorskip("gradio")

from src.webui import style  # noqa: E402

_TOKEN = re.compile(r"(--shep-[a-z0-9-]+)\s*:\s*(#[0-9a-fA-F]{6})")


def _tokens() -> dict:
    return dict(_TOKEN.findall(style.SHELL_CSS))


def _block(selector: str) -> str:
    """The declarations of every rule whose whole selector is `selector`."""
    rules = re.findall(r"(?m)^" + re.escape(selector) + r"\s*\{([^}]*)\}", style.SHELL_CSS)
    return ";".join(rules)


def _resolve(value: str, tokens: dict) -> str:
    value = value.strip().removesuffix("!important").strip()
    found = re.fullmatch(r"var\((--shep-[a-z0-9-]+)\)", value)
    if found:
        return tokens[found.group(1)]
    return {"white": "#ffffff"}.get(value, value)


def _theme_value(name: str) -> str:
    from src.webui.app import _dashboard_theme

    theme = _dashboard_theme()
    value = getattr(theme, name)
    while isinstance(value, str) and value.startswith("*"):
        value = getattr(theme, value[1:])
    return {"white": "#ffffff"}.get(value, value)


def _effective(variable: str) -> str:
    """What a primary button gets for one of Gradio's button variables: the
    stylesheet's value on ``.gradio-container`` if it sets one, else the
    theme's."""
    declarations = {
        name.strip(): value.strip()
        for name, _, value in (part.partition(":") for part in _block(".gradio-container").split(";"))
        if name.strip()}
    if variable in declarations:
        return _resolve(declarations[variable], _tokens())
    return _theme_value(variable.removeprefix("--").replace("-", "_"))


def _luminance(hex_colour: str) -> float:
    def channel(c: int) -> float:
        s = c / 255
        return s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4

    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def contrast(one: str, other: str) -> float:
    a, b = sorted((_luminance(one), _luminance(other)), reverse=True)
    return (a + 0.05) / (b + 0.05)


def test_the_themes_own_primary_button_is_below_the_threshold():
    """Why the stylesheet sets them: the theme alone fails, at rest and on hover."""
    assert contrast("#ffffff", _theme_value("button_primary_background_fill")) < 4.5
    assert contrast("#ffffff", _theme_value("button_primary_background_fill_hover")) < 3.0


@pytest.mark.parametrize("background, text", [
    ("--button-primary-background-fill", "--button-primary-text-color"),
    ("--button-primary-background-fill-hover", "--button-primary-text-color-hover"),
])
def test_a_primary_buttons_text_has_at_least_four_and_a_half_to_one(background, text):
    assert contrast(_effective(text), _effective(background)) >= 4.5


def test_a_disabled_primary_button_is_grey_with_readable_grey_text():
    """Gradio draws a disabled primary button on the hover colour at half
    opacity; the stylesheet draws it grey, opaque, with its text at 4.5:1."""
    rule = _block(".gradio-container button.primary[disabled]")
    declarations = {name.strip(): value for name, _, value in
                    (part.partition(":") for part in rule.split(";")) if name.strip()}
    tokens = _tokens()
    background = _resolve(declarations["background"], tokens)
    text = _resolve(declarations["color"], tokens)
    assert contrast(text, background) >= 4.5
    assert "opacity" in declarations and declarations["opacity"].strip().startswith("1")
