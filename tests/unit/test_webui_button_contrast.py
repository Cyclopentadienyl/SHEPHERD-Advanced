"""
Primary buttons, field labels and checkboxes stay readable (M5-2, M5-3).
=========================================================================
The dashboard's theme (Gradio's Soft) draws a primary button white on indigo-500
(4.47:1) and, on hover and when disabled, on indigo-400 (2.98:1, measured on
Run Diagnosis); a field's label chip (HPO terms, Patient ID, Workspace)
indigo-500 on indigo-100 (3.63:1); and an unchecked checkbox's border grey-100
on a grey-50 fill (1.1:1 against the panel, so the box is nearly invisible;
grey-300 on hover). The shared stylesheet (`src/webui/style.py`) sets these
colours from the dashboard's own tokens and draws a disabled primary button
grey with grey text. These tests compute the contrast (WCAG 2.x) of what each
actually gets: the stylesheet's value where it sets one, the theme's
otherwise, so removing the stylesheet's rules fails them. Text needs 4.5:1;
the checkbox's border, a control's boundary, 3:1 against its fill and against
every background a checkbox sits on (WCAG 1.4.11).

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


def test_the_themes_own_label_chip_and_checkbox_border_are_below_the_threshold():
    """Why the stylesheet sets them."""
    assert contrast(_theme_value("block_title_text_color"),
                    _theme_value("block_title_background_fill")) < 4.5
    assert contrast(_theme_value("checkbox_border_color"), "#ffffff") < 3.0
    assert contrast(_theme_value("checkbox_border_color_hover"), "#ffffff") < 3.0


@pytest.mark.parametrize("text, fill", [
    ("--block-title-text-color", "--block-title-background-fill"),
    ("--block-label-text-color", "--block-label-background-fill"),
])
def test_a_field_labels_text_has_at_least_four_and_a_half_to_one_on_its_chip(text, fill):
    assert contrast(_effective(text), _effective(fill)) >= 4.5


@pytest.mark.parametrize("border", ["--checkbox-border-color", "--checkbox-border-color-hover"])
def test_an_unchecked_checkboxs_border_shows_against_its_fill_and_every_background(border):
    tokens = _tokens()
    around = {"its fill": _effective("--checkbox-background-color"), "white": "#ffffff",
              "panel": tokens["--shep-panel"], "panel-2": tokens["--shep-panel-2"],
              "page": tokens["--shep-bg"]}
    for name, colour in around.items():
        assert contrast(_effective(border), colour) >= 3.0, name


#: Gradio 6.30's own hover rules for a checkbox and a radio (its frontend
#: stylesheet): they ignore `disabled`, so a read-only box turned the accent on
#: hover exactly like one that can be clicked.
_GRADIO_HOVER = {
    "checkbox": ".gradio-container-6-30-0 .gradio-style input[type=checkbox]:hover",
    "radio": ".gradio-container-6-30-0 .gradio-style input[type=radio]:hover",
}


def _specificity(selector: str) -> tuple:
    """(ids, classes + attributes + pseudo-classes, elements) of a selector
    without pseudo-elements; `:not(x)` counts as x."""
    ids = len(re.findall(r"#[\w-]+", selector))
    classes = (len(re.findall(r"\.[A-Za-z_][\w-]*", selector))
               + len(re.findall(r"\[[^\]]*\]", selector))
               + len([p for p in re.findall(r":([\w-]+)", selector) if p != "not"]))
    elements = len(re.findall(r"(?:^|[\s>+~])([A-Za-z][\w-]*)", selector))
    return ids, classes, elements


def _rules() -> list:
    css = re.sub(r"/\*.*?\*/", "", style.SHELL_CSS, flags=re.S)
    return [([part.strip() for part in selectors.split(",")], body)
            for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css)]


@pytest.mark.parametrize("kind", ["checkbox", "radio"])
def test_a_disabled_unchecked_box_does_not_answer_hover(kind):
    """Read-only, a disabled checkbox or radio keeps its resting border and
    fill on hover (only the not-allowed cursor answers); the stylesheet's rule
    outranks Gradio's hover rule, which ignores `disabled`. A checked one is
    left to Gradio's checked look (`:not(:checked)`)."""
    from pathlib import Path

    gradio_css = "".join(path.read_text(encoding="utf-8") for path in
                         (Path(gr.__file__).parent / "templates" / "frontend").rglob("*.css"))
    assert _GRADIO_HOVER[kind] + "{border-color:var(--checkbox-border-color-hover)" in gradio_css

    selector = f".gradio-container input[type={kind}]:disabled:not(:checked):hover"
    bodies = [body for selectors, body in _rules() if selector in selectors]
    assert bodies, f"no rule for {selector}"
    declarations = {name.strip(): value.strip() for name, _, value in
                    (part.partition(":") for part in ";".join(bodies).split(";")) if name.strip()}
    assert declarations.get("border-color") == "var(--checkbox-border-color)"
    assert declarations.get("background-color") == "var(--checkbox-background-color)"
    assert _specificity(selector) > _specificity(_GRADIO_HOVER[kind])
