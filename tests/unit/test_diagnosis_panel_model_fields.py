"""
The model-configuration fields follow the loaded pipeline (BACKLOG 17, F5).
===========================================================================
The Data Directory and Checkpoint fields are what Load / Reload sends. They used
to show the saved UI preferences whatever was served, so after a start
configured by `SHEPHERD_*` the tab named one workspace while another was served,
and pressing Reload would have switched to the one shown. What is pinned here:

- a page load sets both fields from the one status read it makes, to the
  workspace and checkpoint file the loaded pipeline was built from, an
  automatically chosen checkpoint included; with nothing loaded, or no status
  read, the fields keep the saved preferences;
- a start from `SHEPHERD_KG_PATH` alone blanks the Data Directory, with a
  placeholder saying so, and never shows the saved preference there;
- a reload that succeeded sets both fields to what is now served; a refused
  one keeps what the user typed, even though its response carries the live
  pipeline's status;
- the status names the served paths on one line; after a reload that line is
  the only place it names the checkpoint, with how it was chosen below;
- Save writes what the fields show and says when saved paths are used.

Module: tests/unit/test_diagnosis_panel_model_fields.py
"""
import json

import pytest

pytest.importorskip("gradio")
import gradio as gr  # noqa: E402

from src.webui.components import diagnosis_panel as dp  # noqa: E402
from tests.unit.test_diagnosis_panel import _build_tab, _loaded_status  # noqa: E402

WORKSPACE = "/srv/workspaces/hpo_2026_0929"
CHOSEN = f"{WORKSPACE}/runs/hgt/run-1/checkpoints/best.pt"
SAVED = {"data_dir": "data/workspaces/saved_earlier", "checkpoint_path": "/saved/old.pt"}
#: What a handler returns for a field it leaves as it is.
UNCHANGED = gr.update()
NO_RECORDS = {"state": "unavailable", "error": "not read in these tests"}


def _served(data_dir=WORKSPACE, checkpoint=CHOSEN):
    return _loaded_status(current_data_dir=data_dir, current_checkpoint_path=checkpoint)


def _page_load(monkeypatch, status):
    """Run the page-load handler over ``status``; returns its outputs and the
    number of status reads it made."""
    reads = []

    def read_status():
        reads.append(1)
        return status

    monkeypatch.setattr(dp, "_get_pipeline_status", read_status)
    monkeypatch.setattr(dp, "_get_loaded_model_evaluations", lambda: NO_RECORDS)
    return dp._on_load_status(), len(reads)


def _reload_answering(monkeypatch, response):
    """Make the reload POST answer ``response``; returns the requests sent."""
    sent = []

    def reload(data_dir, checkpoint_path, conv_type="auto"):
        sent.append((data_dir, checkpoint_path, conv_type))
        return response

    monkeypatch.setattr(dp, "_reload_pipeline", reload)
    monkeypatch.setattr(dp, "_get_loaded_model_evaluations", lambda: NO_RECORDS)
    return sent


def _reload(monkeypatch, *fields):
    """The Reload handler as Gradio runs it (``async``), for a request D8
    allows (mode M1, loopback)."""
    import asyncio

    from tests.fixtures.operator_access import operator_request

    return asyncio.run(dp._on_reload_pipeline(*fields, operator_request(monkeypatch)))


def _path_lines(rendered):
    return [line for line in rendered.splitlines()
            if line.startswith(("- Workspace", "- No workspace"))]


def _path_line(rendered):
    (line,) = _path_lines(rendered)
    return line


# ---------------------------------------------------------------- page load
def test_a_page_load_sets_both_fields_from_its_one_status_read(monkeypatch):
    (rendered, data_dir, checkpoint), reads = _page_load(monkeypatch, _served())

    assert reads == 1
    assert data_dir == gr.update(value=WORKSPACE, placeholder="")
    assert checkpoint == gr.update(value=CHOSEN)
    assert _path_line(rendered) == f"- Workspace `{WORKSPACE}` · Checkpoint file `{CHOSEN}`"


def test_a_pipeline_serving_no_checkpoint_empties_the_checkpoint_field(monkeypatch):
    """Path reasoning from a workspace with no model: the field is blank, so a
    Reload asks for automatic selection rather than a file that is not served."""
    (rendered, data_dir, checkpoint), _ = _page_load(monkeypatch, _served(checkpoint=None))

    assert data_dir["value"] == WORKSPACE
    assert checkpoint == gr.update(value="")
    assert _path_line(rendered) == f"- Workspace `{WORKSPACE}` · no checkpoint file"


def test_with_nothing_loaded_the_saved_preferences_stay(monkeypatch, tmp_path):
    config = tmp_path / ".shepherd_ui_config.json"
    config.write_text(json.dumps(SAVED))
    monkeypatch.setattr(dp, "CONFIG_FILE", config)
    demo = _build_tab(monkeypatch)
    fields = {block.label: block for block in demo.blocks.values()
              if isinstance(block, gr.Textbox)}
    assert fields["Data Directory"].value == SAVED["data_dir"]
    assert fields["Checkpoint Path (optional)"].value == SAVED["checkpoint_path"]

    (rendered, data_dir, checkpoint), reads = _page_load(monkeypatch, {"initialized": False})

    assert reads == 1
    assert data_dir == UNCHANGED and checkpoint == UNCHANGED
    assert "Pipeline not loaded" in rendered and _path_lines(rendered) == []


def test_an_unreadable_status_leaves_the_fields_alone(monkeypatch):
    """It says nothing about what is served, so it must not blank a field."""
    status = {"initialized": False, "error": "API not reachable"}

    (rendered, data_dir, checkpoint), _ = _page_load(monkeypatch, status)

    assert data_dir == UNCHANGED and checkpoint == UNCHANGED
    assert "Pipeline status unavailable" in rendered and _path_lines(rendered) == []


def test_paths_the_status_did_not_report_leave_the_fields_alone(monkeypatch):
    (rendered, data_dir, checkpoint), _ = _page_load(monkeypatch, _loaded_status())

    assert data_dir == UNCHANGED and checkpoint == UNCHANGED
    assert _path_line(rendered) == "- Workspace not reported · checkpoint file not reported"


@pytest.mark.parametrize("checkpoint_path, named", [
    (None, "no checkpoint file"),
    ("/models/best.pt", "Checkpoint file `/models/best.pt`"),
])
def test_a_path_only_start_blanks_the_data_directory(monkeypatch, checkpoint_path, named):
    """Started from SHEPHERD_KG_PATH without SHEPHERD_DATA_DIR: no workspace is
    served, so the field is blank, never the saved preference, and says why;
    Reload then needs a directory typed."""
    status = _served(data_dir=None, checkpoint=checkpoint_path)

    (rendered, data_dir, checkpoint), _ = _page_load(monkeypatch, status)

    assert data_dir == gr.update(value="", placeholder=dp.PATH_ONLY_PLACEHOLDER)
    assert "SHEPHERD_KG_PATH" in dp.PATH_ONLY_PLACEHOLDER
    assert checkpoint == gr.update(value=checkpoint_path or "")
    assert _path_line(rendered) == (
        f"- No workspace: graph served from `SHEPHERD_KG_PATH` · {named}"
    )


# ------------------------------------------------------------------- reload
def test_a_successful_reload_sets_both_fields_to_what_is_served(monkeypatch):
    sent = _reload_answering(monkeypatch, {
        "success": True, "message": "Pipeline reloaded successfully.",
        "status": _served(), "checkpoint_path": CHOSEN,
    })

    status_md, message, data_dir, checkpoint = _reload(
        monkeypatch, f" {WORKSPACE} ", f" {CHOSEN} ", "auto")

    assert sent == [(WORKSPACE, CHOSEN, "auto")]
    assert message == "Pipeline reloaded successfully."
    # The placeholder a path-only status may have set is cleared.
    assert data_dir == gr.update(value=WORKSPACE, placeholder="")
    assert checkpoint == gr.update(value=CHOSEN)
    assert f"Workspace `{WORKSPACE}` · Checkpoint file `{CHOSEN}`" in status_md


def test_a_successful_reload_names_its_checkpoint_once_and_how_it_was_chosen(monkeypatch):
    """The response's `checkpoint_path` is the string the status's served-paths
    line already names; the status adds the selection reason, not the path a
    second time (BACKLOG 27: clean and uncrowded)."""
    reason = "auto: latest run 'run-1' (hgt); best val_mrr=0.4100"
    _reload_answering(monkeypatch, {
        "success": True, "message": f"Pipeline reloaded successfully. ({reason})",
        "status": _served(), "checkpoint_path": CHOSEN, "selection_reason": reason,
    })

    status_md, *_ = _reload(monkeypatch, WORKSPACE, "", "auto")

    assert status_md.count(CHOSEN) == 1
    assert _path_line(status_md) == f"- Workspace `{WORKSPACE}` · Checkpoint file `{CHOSEN}`"
    assert f"<sub>Checkpoint selection: {reason}</sub>" in status_md.splitlines()


def test_an_automatically_chosen_checkpoint_is_shown_and_reloaded_as_shown(monkeypatch):
    """A blank field asks the server to choose; the field then names the file it
    chose, so the next Reload reloads exactly that until the field is cleared."""
    sent = _reload_answering(monkeypatch, {
        "success": True, "message": "Pipeline reloaded successfully.",
        "status": _served(), "checkpoint_path": CHOSEN, "selection_reason": "auto",
    })

    _, _, data_dir, checkpoint = _reload(monkeypatch, WORKSPACE, "", "auto")
    assert checkpoint["value"] == CHOSEN
    _reload(monkeypatch, data_dir["value"], checkpoint["value"], "auto")

    assert sent == [(WORKSPACE, "", "auto"), (WORKSPACE, CHOSEN, "auto")]


def test_a_refused_reload_keeps_what_was_typed(monkeypatch):
    """A refusal carries the live pipeline's status (the old pipeline is still
    served); taking the fields from it would discard what the user typed."""
    live = _served(data_dir="/srv/workspaces/still_served", checkpoint="/srv/old.pt")
    _reload_answering(monkeypatch, {
        "success": False, "message": "Missing required files", "status": live,
    })

    status_md, message, data_dir, checkpoint = _reload(
        monkeypatch, "/typed/workspace", "/typed/best.pt", "auto")

    assert status_md == "❌ Missing required files" and message == "Missing required files"
    assert data_dir == UNCHANGED and checkpoint == UNCHANGED


def test_a_reload_without_a_data_directory_keeps_the_fields(monkeypatch):
    sent = _reload_answering(monkeypatch, AssertionError("nothing is sent"))

    status_md, message, data_dir, checkpoint = _reload(monkeypatch, " ", "/typed.pt", "auto")

    assert sent == []
    assert status_md == "⚠️ Data directory is required." and message == ""
    assert data_dir == UNCHANGED and checkpoint == UNCHANGED


# ------------------------------------------------------- the line and Save
def test_the_paths_are_one_line_under_the_counts():
    rendered = dp._format_pipeline_status(_served())

    lines = rendered.splitlines()
    gnn = next(i for i, line in enumerate(lines) if line.startswith("- GNN:"))
    assert lines[gnn + 1] == _path_line(rendered)


def test_the_checkpoint_help_says_how_to_ask_for_automatic_selection(monkeypatch):
    demo = _build_tab(monkeypatch)
    (field,) = [block for block in demo.blocks.values()
                if isinstance(block, gr.Textbox) and block.label == "Checkpoint Path (optional)"]

    assert "the served file" in field.info
    assert "Clear it to auto-select" in field.info


def test_save_writes_what_the_fields_show_and_says_so(monkeypatch, tmp_path):
    config = tmp_path / ".shepherd_ui_config.json"
    monkeypatch.setattr(dp, "CONFIG_FILE", config)

    from tests.fixtures.operator_access import operator_request

    request = operator_request(monkeypatch)
    message = dp._on_save_config(WORKSPACE, CHOSEN, request)

    assert json.loads(config.read_text()) == {"data_dir": WORKSPACE, "checkpoint_path": CHOSEN}
    assert message == dp.CONFIG_SAVED
    assert "Saved the paths shown" in message
    assert "only while no pipeline is loaded" in message

    dp._on_save_config(WORKSPACE, "", request)
    assert json.loads(config.read_text())["checkpoint_path"] is None


def test_save_is_refused_without_m1_and_writes_nothing(monkeypatch, tmp_path):
    """Save Config is an operator control (D8): checked on its own request,
    before the one writer is reached."""
    from src.api.routes import pipeline
    from tests.fixtures.operator_access import gradio_request

    config = tmp_path / ".shepherd_ui_config.json"
    monkeypatch.setattr(dp, "CONFIG_FILE", config)
    monkeypatch.setattr(pipeline, "write_ui_config",
                        lambda *a, **k: pytest.fail("a refused save reached the writer"))

    message = dp._on_save_config(WORKSPACE, CHOSEN, gradio_request())
    assert message.startswith("Refused: SHEPHERD_DEPLOYMENT_MODE is not set")
    assert not config.exists()
