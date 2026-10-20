"""
The Training form takes Datasets' workspace until the run bar does (M5-2, UI-A).
================================================================================
`src/webui/components/training_console.py`: `_from_datasets` and
`_keep_or_clear_workspace`, the temporary bridge from Datasets' "Use for
training ›" (``ctx.training_workspace``) to today's Data Directory. UI-C's run
bar replaces both and this file with them. What these pin:

- "Use for training ›" fills this session's Data Directory with the listed
  workspace and opens Data Paths; a name the server does not list changes
  nothing;
- a workspace outside the project would be an absolute server path: only a
  request D8 allows is given it (the one that can Start); a refused request
  gets a note and the field is left;
- the Data Directory changed to anything else (by hand, or Reset) clears the
  session's workspace, so Datasets stops marking it "in your Training run
  bar" and choosing the same workspace again fills the field again.

Module: tests/unit/test_training_console_datasets_bridge.py
"""
from __future__ import annotations

import shutil

import pytest

gr = pytest.importorskip("gradio")

from src.webui.components import training_console as tc  # noqa: E402
from tests.fixtures.operator_access import gradio_request, operator_request  # noqa: E402


@pytest.fixture(scope="module")
def built_workspace(tmp_path_factory):
    from tests.fixtures.registered_cohort import write_demo_workspace

    return write_demo_workspace(tmp_path_factory.mktemp("bridge") / "demo")


@pytest.fixture
def root(built_workspace, tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    shutil.copytree(built_workspace, root / "resampled")
    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(root))
    return root


def test_the_bridge_is_wired_to_the_sessions_workspace(root):
    from src.webui import shell
    from src.webui.app import PAGES

    with gr.Blocks() as demo:
        ctx = shell.build_shell(demo, PAGES)
    (bridge,) = [fn for fn in demo.fns.values() if fn.fn is tc._from_datasets]
    assert bridge.inputs == [ctx.training_workspace]
    data_dir, accordion, message = bridge.outputs
    assert data_dir.elem_id == "data_dir" and isinstance(accordion, gr.Accordion)
    assert isinstance(message, gr.Markdown)
    (clear,) = [fn for fn in demo.fns.values() if fn.fn is tc._keep_or_clear_workspace]
    assert clear.targets == [(data_dir._id, "change")]
    assert clear.inputs == [data_dir, ctx.training_workspace]
    assert clear.outputs == [ctx.training_workspace]


def test_a_workspace_inside_the_project_is_shown_relative_to_every_request(root, monkeypatch):
    from src.api.services import training_manager as tm

    monkeypatch.setattr(tm, "PROJECT_ROOT", root.parent)
    value, accordion, note = tc._from_datasets("resampled", gradio_request())
    assert value["value"] == "SHEPHERD-Advanced/root/resampled" and accordion.open is True
    assert note == ""


def test_an_absolute_server_path_goes_only_to_a_request_d8_allows(root, monkeypatch):
    monkeypatch.delenv("SHEPHERD_DEPLOYMENT_MODE", raising=False)
    value, accordion, note = tc._from_datasets("resampled", gradio_request())
    assert value == gr.update() and accordion.open is True
    assert "resampled is selected" in note and str(root) not in note
    value, _accordion, note = tc._from_datasets("resampled", operator_request(monkeypatch))
    assert value["value"] == str((root / "resampled").resolve()) and note == ""


def test_a_name_the_server_does_not_list_changes_nothing(root, monkeypatch):
    assert tc._from_datasets("../etc", operator_request(monkeypatch)) == (
        gr.update(), gr.update(), gr.update())
    assert tc._from_datasets(None, operator_request(monkeypatch)) == (
        gr.update(), gr.update(), gr.update())


def test_a_data_directory_changed_to_anything_else_clears_the_workspace(root):
    shown = tc._workspace_field("resampled")
    assert tc._keep_or_clear_workspace(shown, "resampled") == gr.update()
    assert tc._keep_or_clear_workspace(f"  {shown} ", "resampled") == gr.update()
    assert tc._keep_or_clear_workspace("SHEPHERD-Advanced/data/workspaces/other",
                                       "resampled") is None
    assert tc._keep_or_clear_workspace("anything", None) == gr.update()
