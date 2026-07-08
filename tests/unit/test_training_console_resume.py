"""
Unit tests for architecture-aware resume in the Training Console.

Covers the resume read path aligning with the architecture-scoped checkpoint
write layout: refresh relinks to the selected conv_type, explicit checkpoint_dir
wins, and the _on_resume containment guard.
"""
from pathlib import Path

import pytest

# training_console imports gradio (and the api/service stack) at module load.
pytest.importorskip("gradio")

from src.api.services.training_manager import training_manager  # noqa: E402
from src.webui.components import training_console as tc  # noqa: E402


# --------------------------------------------------------------- dir resolution
def test_resolve_resume_dir_architecture_scoped():
    assert tc._resolve_resume_dir("ws", "hgt", "") == Path("ws/checkpoints/hgt")
    assert tc._resolve_resume_dir("ws", "gat", "") == Path("ws/checkpoints/gat")


def test_resolve_resume_dir_explicit_dir_wins():
    # explicit checkpoint_dir is used verbatim (same rule as the write path)
    assert tc._resolve_resume_dir("ws", "hgt", "/custom/ckpts") == Path("/custom/ckpts")


def test_resolve_resume_dir_strips_display_prefix():
    assert tc._resolve_resume_dir("SHEPHERD-Advanced/ws", "hgt", "") == Path(
        "ws/checkpoints/hgt"
    )


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
def test_refresh_relinks_to_selected_architecture(tmp_path):
    ws = tmp_path / "ws"
    (ws / "checkpoints" / "hgt").mkdir(parents=True)
    (ws / "checkpoints" / "hgt" / "last.pt").write_text("x")
    (ws / "checkpoints" / "gat").mkdir(parents=True)
    (ws / "checkpoints" / "gat" / "model-1-0.9.pt").write_text("x")

    tc._refresh_checkpoints(str(ws), "hgt", "")
    assert training_manager.checkpoint_dir == ws / "checkpoints" / "hgt"
    assert {c["filename"] for c in training_manager.get_checkpoints()} == {"last.pt"}

    tc._refresh_checkpoints(str(ws), "gat", "")  # conv_type change -> relink
    assert training_manager.checkpoint_dir == ws / "checkpoints" / "gat"
    assert {c["filename"] for c in training_manager.get_checkpoints()} == {"model-1-0.9.pt"}


def test_refresh_explicit_checkpoint_dir_wins(tmp_path):
    custom = tmp_path / "custom"
    custom.mkdir()
    (custom / "last.pt").write_text("x")
    tc._refresh_checkpoints(str(tmp_path / "ws"), "hgt", str(custom))
    assert training_manager.checkpoint_dir == custom


# --------------------------------------------------------------- docstring
def test_cli_resume_docstring_is_architecture_aware():
    text = Path("scripts/train_model.py").read_text(encoding="utf-8")
    assert "--resume checkpoints/last.pt" not in text  # stale flat example gone
    assert "checkpoints/hgt/last.pt" in text
