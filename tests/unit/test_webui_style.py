"""R-WIDTH (the owner's D12 of 2026-11-24): a field's width comes from its role.

At 2560 px Fine-tune's Epochs box was about 440 px wide for a two-digit number while a name or a
path had no floor of its own: one cap for every field, or none. The shared stylesheet now holds a
width per role (``--shep-w-num``, ``--shep-w-name``, ``--shep-w-grow-min``, in ``ch`` of each
field's own type) and one class per role, and the one field of a learning setting
(`training_console.setting_field`) takes its role from its field spec's type, never from a
page.
"""

from __future__ import annotations

import re

import pytest

from src.webui.style import SHELL_CSS

ROLES = ("shep-w-num", "shep-w-name", "shep-w-path", "shep-w-search")


def _root() -> str:
    return SHELL_CSS[SHELL_CSS.index(":root {"):].split("}", 1)[0]


def test_each_role_has_its_width_token_and_its_class():
    root = _root()
    for token in ("--shep-w-num", "--shep-w-name", "--shep-w-grow-min"):
        found = re.search(rf"{token}:\s*(\d+)ch;", root)
        assert found, token
    # A short number is narrower than a name, and a path or a search grows from its floor.
    widths = {t: int(re.search(rf"{t}:\s*(\d+)ch;", root).group(1))
              for t in ("--shep-w-num", "--shep-w-name", "--shep-w-grow-min")}
    assert widths["--shep-w-num"] < widths["--shep-w-grow-min"] <= widths["--shep-w-name"]
    for cls in ROLES:
        assert re.search(rf"\.{cls}\b[^{{]*\{{[^}}]*(max|min)-width", SHELL_CSS), cls
    # The search label reads as every field label (amendment 6): the label tokens.
    rule = SHELL_CSS[SHELL_CSS.index(".shep-search-lbl, .shep-bar-lbl {"):].split("}", 1)[0]
    for token in ("--shep-label-size", "--shep-label-weight", "--shep-label-color"):
        assert f"var({token})" in rule, token


@pytest.mark.parametrize("name, role", [("num_epochs", "shep-w-num"), ("learning_rate", "shep-w-num")])
def test_a_number_setting_takes_the_number_role_from_its_spec(name, role):
    from src.webui.components import training_console as tc

    field = tc.setting_field(name, tc.by_name(name).default, elem_id=f"t-{name}")
    assert role in (field.elem_classes or []), field.elem_classes


def test_a_setting_that_is_not_a_number_takes_no_number_role():
    from src.config.training_fields import FIELDS
    from src.webui.components import training_console as tc

    seen = set()
    for spec in FIELDS:
        if spec.name not in tc.r.SETTING_LABELS:
            continue
        field = tc.setting_field(spec.name, spec.default, elem_id=f"t-{spec.name}")
        # A number's role by the spec's type (a list of whole numbers is text of its own kind).
        numeric = spec.kind in ("int", "float") and not spec.choices
        assert ("shep-w-num" in (field.elem_classes or [])) is numeric, (spec.name, spec.kind)
        seen.add(numeric)
    assert seen == {True, False}
