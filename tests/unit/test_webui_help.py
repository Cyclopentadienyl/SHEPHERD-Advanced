"""
The shared ⓘ (M5-2, UI-A): one help control for every ordinary help text.
=========================================================================
`src/webui/components/help.py`. What these pin:

- **the registry**: one text per key, a key that can be an HTML id, every text
  escaped in the one hidden container, and a ⓘ that names its text by
  ``aria-describedby``;
- **the built dashboard**: every registered field exists (by ``elem_id``) and
  its title is the field's label; no component keeps ordinary help in
  ``info=`` (only the AMP refusal, which is state, is set there at runtime);
  every ⓘ the shared renderers draw names a registered text;
- **the browser side** fills the bubble with ``textContent`` and is given to
  the mount's ``head`` (Gradio 6 drops a head given to ``gr.Blocks``);
- **the AMP refusal** stays visible text under HGT and is cleared with ``""``
  (Gradio 6.30 keeps an info sent as None).

Module: tests/unit/test_webui_help.py
"""
from __future__ import annotations

import pytest

gr = pytest.importorskip("gradio")

from src.webui.components import help as h  # noqa: E402


# --------------------------------------------------------------------------- the registry
def test_a_field_and_a_topic_are_registered_once_each():
    registry = h.HelpRegistry()
    assert registry.field("num_epochs", "Epochs", "Smoke test: 2-5") == "num_epochs"
    registry.field("num_epochs", "Epochs", "Smoke test: 2-5")          # the same again: fine
    registry.topic("dataset-fingerprint", "Dataset fingerprint", "The SHA-256 …")
    assert registry.targets() == ("num_epochs",)
    assert "dataset-fingerprint" in registry and "num_epochs" in registry
    with pytest.raises(ValueError, match="twice"):
        registry.field("num_epochs", "Epochs", "other words")


@pytest.mark.parametrize("key", ["", "1abc", "a b", 'x"><script>', "a/b", None])
def test_a_key_must_be_able_to_be_an_html_id(key):
    with pytest.raises(ValueError):
        h.HelpRegistry().topic(key, "T", "text")


def test_a_title_and_a_text_are_required():
    with pytest.raises(ValueError, match="title and a text"):
        h.HelpRegistry().topic("k", "", "text")


def test_every_text_is_escaped_once_in_the_hidden_container():
    registry = h.HelpRegistry()
    registry.field("dx_architecture", "Arch <b>", "the latest run under {data}/runs/<arch>/ & more")
    html = registry.texts_html()
    assert html.startswith('<div class="shep-help-texts" hidden>')
    assert 'id="shep-help-text-dx_architecture"' in html
    assert 'data-help-target="dx_architecture"' in html
    assert "{data}/runs/&lt;arch&gt;/ &amp; more" in html
    assert 'data-help-title="Arch &lt;b&gt;"' in html
    assert "<arch>" not in html and "<b>" not in html


def test_a_help_button_is_a_button_that_names_its_text():
    button = h.help_button("dataset-fingerprint", 'Fingerprint "x"')
    assert button.startswith('<button type="button" class="shep-i"')
    assert 'aria-describedby="shep-help-text-dataset-fingerprint"' in button
    assert 'aria-expanded="false"' in button
    assert 'aria-label="Help: Fingerprint &quot;x&quot;"' in button
    own = h.help_button("chip-d8", "Operator controls", describedby="shep-chip-d8-text")
    assert 'aria-describedby="shep-chip-d8-text"' in own
    with pytest.raises(ValueError):
        h.help_button("not a key", "x")


def test_copy_carries_the_full_value_escaped():
    button = h.copy_button('a"b' + "0" * 62, "the full dataset fingerprint")
    assert 'data-copy="a&quot;b' in button and button.startswith('<button type="button"')
    assert "shep-copy-icon" in h.copy_button("0" * 64, "x", icon_only=True)


def test_the_browser_side_reads_texts_as_text_and_goes_to_the_mount():
    from src.webui.app import mount_options

    head = mount_options()["head"]
    assert h.HELP_HEAD in head
    # The bubble is filled with textContent: no help text is parsed as HTML.
    assert "textContent = textOf(btn)" in h.HELP_HEAD
    assert "innerHTML = textOf" not in h.HELP_HEAD
    for behaviour in ("mouseover", "focusin", "'click'", "Escape", "MutationObserver",
                      "aria-expanded", "aria-describedby"):
        assert behaviour in h.HELP_HEAD, behaviour


def test_a_click_on_a_fields_title_still_reaches_the_field():
    """The ⓘ is a <button> inside the field's <label>: without ``for``, it would
    become the label's control and a click on "Epochs" would open the help
    instead of focusing the input. The label names its input by id."""
    assert "function keepLabel(target, key)" in h.HELP_HEAD
    assert "label.htmlFor = input.id" in h.HELP_HEAD
    assert "input.id = 'shep-in-' + key" in h.HELP_HEAD
    # kept on every pass, for a field Gradio renders again
    assert "{ keepLabel(target, key); continue; }" in h.HELP_HEAD


def test_the_bubble_is_in_the_dashboards_font_and_never_taller_than_the_window():
    bubble = h.HELP_CSS[h.HELP_CSS.index(".shep-bubble {"):]
    bubble = bubble[:bubble.index("}")]
    assert "font-family: 'IBM Plex Sans'" in bubble        # it lives outside .gradio-container
    assert "max-height: calc(100vh - 16px)" in bubble
    assert "overflow-y: auto" in h.HELP_CSS[h.HELP_CSS.index(".shep-bubble-bd {"):]
    assert "roomBelow >= roomAbove" in h.HELP_HEAD and "bubble.style.maxHeight" in h.HELP_HEAD


# --------------------------------------------------------------------------- the built dashboard
@pytest.fixture(scope="module")
def built():
    """The dashboard as `mount_dashboard` builds it, and its help registry."""
    from src.webui import shell
    from src.webui.app import PAGES

    registry = h.HelpRegistry()
    with gr.Blocks() as demo:
        shell.build_shell(demo, PAGES, help_registry=registry)
        texts = shell.help_texts(registry)
    return demo, registry, texts


def test_every_registered_help_target_exists_in_the_built_app(built):
    """Structural, so a page that adds or removes a field (UI-B, UI-C, UI-D)
    changes no test here: every page module the app builds that declares
    ``FIELD_HELP`` has all of it registered, and every registered target is a
    component of the built app whose label is the help's title."""
    import sys

    from src.webui.app import PAGES

    demo, registry, _texts = built
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    modules = {sys.modules[builder.__module__] for builder in PAGES.values()}
    declared = [m.FIELD_HELP for m in modules if isinstance(getattr(m, "FIELD_HELP", None), dict)]
    assert declared, "no page declares field help"
    for field_help in declared:
        assert set(field_help) <= set(registry.targets())
    for entry in registry.entries():
        if entry.target is None:
            continue
        assert entry.target in by_id, entry.target
        component = by_id[entry.target]
        assert getattr(component, "label", None) == entry.title, entry.target


def test_no_component_keeps_ordinary_help_in_info(built):
    """Only state-bearing text travels in ``info=``, and none is set at build
    time: the AMP refusal is set when HGT is chosen."""
    demo, _registry, _texts = built
    with_info = [(getattr(b, "elem_id", None), b.info) for b in demo.blocks.values()
                 if getattr(b, "info", None)]
    assert with_info == []


def test_the_hidden_container_holds_every_registered_text(built):
    _demo, registry, texts = built
    for entry in registry.entries():
        assert f'id="{h.text_id(entry.key)}"' in texts.value


def test_every_renderer_help_key_is_registered(built):
    """The ⓘ the shared renderers draw name registered texts (the shell
    registers RENDERER_HELP)."""
    from src.webui.components import renderers as r

    _demo, registry, _texts = built
    workspace = {"name": "w", "dataset_fingerprint": "a" * 64, "trainable": True,
                 "dataset_record": {"binding": "verified", "origin": "files"},
                 "build": {"state": "not_recorded"}, "declared_releases": ()}
    item = {"workspace": "w", "architecture": "gat", "run": "20261017-091500_run_x",
            "filename": "best.pt", "sha256": "b" * 64, "epoch": 3,
            "validation": {"metric": "val_mrr", "value": 0.3}, "dataset_fingerprint": "a" * 64,
            "problem": None}
    rendered = (r.dataset_identity_html(workspace)
                + r.model_facts_html(r.facts_from_checkpoint(item, [workspace]))
                + r.fingerprint_html("model fingerprint", "b" * 64, help_key="model-fingerprint"))
    keys = h.registered_keys_in(rendered)
    assert set(keys) >= {"dataset-fingerprint", "validation-trainer", "model-fingerprint"}
    for key in keys:
        assert key in registry, key


# --------------------------------------------------------------------------- the AMP refusal
def test_hgt_shows_the_amp_refusal_and_any_other_architecture_clears_it():
    from src.webui.components import training_console as tc

    hgt = tc._on_conv_type_change("hgt")
    assert hgt["info"] == tc.AMP_HGT_REFUSAL == (
        "Disabled: HGT requires full float32 (pyg_lib limitation).")
    assert hgt["interactive"] is False and hgt["value"] == "Off"
    for other in ("gat", "sage"):
        cleared = tc._on_conv_type_change(other)
        assert cleared["info"] == "" and cleared["interactive"] is True
        assert "value" not in cleared


def test_the_architecture_change_updates_the_amp_radio_once():
    """One listener: two would send two requests per change, both writing
    the AMP radio."""
    from src.webui.components import training_console as tc
    from src.webui.shell import standalone_context

    with gr.Blocks() as demo:
        tc.create_training_tab(standalone_context(demo))
    listeners = [fn for fn in demo.fns.values() if fn.fn is tc._on_conv_type_change]
    assert len(listeners) == 1


def test_the_amp_radio_has_its_ordinary_help_behind_the_i():
    from src.webui.components import training_console as tc

    title, text = tc.FIELD_HELP["amp_mode"]
    assert title == "Mixed Precision (AMP)" and text.startswith("Faster & less VRAM")
