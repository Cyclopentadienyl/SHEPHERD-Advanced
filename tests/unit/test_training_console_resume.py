"""
Unit tests for architecture-aware resume in the Training Console.

Covers the resume read path aligning with the run layout (contract M3a-2):
refresh lists the selected conv_type's runs through the shared listing, and the
_on_resume containment guard.
"""
from pathlib import Path

import pytest

# training_console imports gradio (and the api/service stack) at module load.
pytest.importorskip("gradio")

from src.api.services.training_manager import training_manager  # noqa: E402
from src.webui.components import training_console as tc  # noqa: E402


def _run_file(ws, arch, run, name):
    path = ws / "runs" / arch / run / "checkpoints" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")
    return path


# --------------------------------------------------------------- containment guard
def test_is_within_dir(tmp_path):
    parent = tmp_path / "checkpoints" / "gat"
    parent.mkdir(parents=True)
    inside = parent / "last.pt"
    inside.write_text("x")
    outside = tmp_path / "checkpoints" / "hgt" / "last.pt"
    outside.parent.mkdir(parents=True)
    outside.write_text("x")
    assert tc._is_within_dir(inside, parent) is True
    assert tc._is_within_dir(outside, parent) is False  # different architecture dir


def test_is_within_dir_handles_missing_paths(tmp_path):
    parent = tmp_path / "checkpoints" / "gat"  # does not exist
    assert tc._is_within_dir(parent / "last.pt", parent) is True  # still contained
    assert tc._is_within_dir(tmp_path / "elsewhere.pt", parent) is False


# --------------------------------------------------------------- refresh relink
def test_refresh_lists_the_selected_architectures_runs(tmp_path):
    ws = tmp_path / "ws"
    hgt = _run_file(ws, "hgt", "20261001-000000_a_111111", "last.pt")
    gat = _run_file(ws, "gat", "20261002-000000_b_222222", "model-1-0.9.pt")

    update = tc._refresh_checkpoints(str(ws), "hgt")
    assert [value for _, value in update["choices"]] == [str(hgt)]
    assert update["choices"][0][0].startswith("20261001-000000_a_111111 / last.pt")

    update = tc._refresh_checkpoints(str(ws), "gat")  # conv_type change -> relink
    assert [value for _, value in update["choices"]] == [str(gat)]
    assert update["value"] == str(gat)


def test_refresh_does_not_list_the_old_layout(tmp_path):
    ws = tmp_path / "ws"
    (ws / "checkpoints" / "hgt").mkdir(parents=True)
    (ws / "checkpoints" / "hgt" / "last.pt").write_text("x")

    update = tc._refresh_checkpoints(str(ws), "hgt")
    assert update["choices"] == [] and update["value"] is None


def test_the_manager_lists_every_architecture_with_its_run(tmp_path):
    ws = tmp_path / "ws"
    import os

    newer = _run_file(ws, "hgt", "20261001-000000_a_111111", "last.pt")
    older = _run_file(ws, "gat", "20261002-000000_b_222222", "last.pt")
    os.utime(newer, (2_000, 2_000))
    os.utime(older, (1_000, 1_000))

    listed = training_manager.get_checkpoints(str(ws))
    assert [(c["architecture"], c["run"]) for c in listed] == [
        ("hgt", "20261001-000000_a_111111"), ("gat", "20261002-000000_b_222222")]


# --------------------------------------------------------------- docstring
def test_cli_resume_docstring_names_the_run_layout():
    text = Path("scripts/train_model.py").read_text(encoding="utf-8")
    assert "--resume checkpoints/last.pt" not in text  # stale flat example gone
    assert "/checkpoints/hgt/last.pt" not in text      # the M2 layout is gone too
    assert "runs/hgt/<run>/checkpoints/last.pt" in text
