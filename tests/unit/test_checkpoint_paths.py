"""
Unit tests for architecture-scoped checkpoint path derivation & selection.

Dependency-light — checkpoint_paths imports only the torch-free constants module.
"""
import os
from pathlib import Path

from src.utils.checkpoint_paths import (
    normalize_conv_type,
    resolve_checkpoint_dir,
    select_auto_checkpoint,
    select_checkpoint_in_dir,
)


def _touch(path: Path, mtime: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    os.utime(path, (mtime, mtime))
    return path


# --------------------------------------------------------------- normalize
def test_normalize_conv_type():
    assert normalize_conv_type("HGT") == "hgt"
    assert normalize_conv_type("  gat ") == "gat"
    assert normalize_conv_type("sage") == "sage"
    assert normalize_conv_type("bogus") == "gat"  # default
    assert normalize_conv_type(None) == "gat"
    assert normalize_conv_type("") == "gat"


# --------------------------------------------------------------- resolve dir
def test_resolve_checkpoint_dir_auto():
    assert resolve_checkpoint_dir("ws", "hgt") == Path("ws/checkpoints/hgt")
    assert resolve_checkpoint_dir("ws", "GAT") == Path("ws/checkpoints/gat")
    assert resolve_checkpoint_dir("ws", None) == Path("ws/checkpoints/gat")  # default


def test_resolve_checkpoint_dir_explicit_wins_verbatim():
    # explicit dir is returned unchanged — no conv_type appended
    assert resolve_checkpoint_dir("ws", "hgt", "/custom/ckpts") == Path("/custom/ckpts")


# --------------------------------------------------------------- select in dir
def test_select_prefers_best_val(tmp_path):
    d = tmp_path / "hgt"
    _touch(d / "last.pt", 100)
    _touch(d / "model-05-9.1000.pt", 101)
    _touch(d / "model-11-8.6500.pt", 102)  # best (lowest val)
    _touch(d / "model-14-8.9000.pt", 103)
    assert select_checkpoint_in_dir(d).name == "model-11-8.6500.pt"


def test_select_falls_back_to_last(tmp_path):
    d = tmp_path / "hgt"
    _touch(d / "last.pt", 100)  # no model-*.pt present
    assert select_checkpoint_in_dir(d).name == "last.pt"


def test_select_falls_back_to_newest(tmp_path):
    d = tmp_path / "hgt"
    _touch(d / "epoch_a.pt", 100)
    _touch(d / "epoch_b.pt", 200)  # newest, no last/model-*
    assert select_checkpoint_in_dir(d).name == "epoch_b.pt"


def test_select_none_when_empty_or_missing(tmp_path):
    assert select_checkpoint_in_dir(tmp_path / "nope") is None
    (tmp_path / "empty").mkdir()
    assert select_checkpoint_in_dir(tmp_path / "empty") is None


# --------------------------------------------------------------- auto selection
def test_auto_picks_latest_trained_architecture(tmp_path):
    base = tmp_path / "checkpoints"
    _touch(base / "hgt" / "last.pt", 100)
    _touch(base / "hgt" / "model-10-8.7.pt", 101)
    _touch(base / "gat" / "last.pt", 500)  # gat trained more recently
    _touch(base / "gat" / "model-10-8.5.pt", 501)
    path, arch, reason = select_auto_checkpoint(base)
    assert arch == "gat"
    assert path.name == "model-10-8.5.pt"  # best within the latest arch
    assert "gat" in reason


def test_auto_ignores_non_architecture_dirs(tmp_path):
    base = tmp_path / "checkpoints"
    _touch(base / "junk" / "last.pt", 999)  # not a supported arch name
    _touch(base / "hgt" / "last.pt", 100)
    path, arch, _ = select_auto_checkpoint(base)
    assert arch == "hgt"
    assert path.name == "last.pt"


def test_auto_none_without_architecture_checkpoints(tmp_path):
    base = tmp_path / "checkpoints"
    base.mkdir()
    _touch(base / "legacy.pt", 100)  # flat file, no arch subdir -> not auto-selected
    path, arch, reason = select_auto_checkpoint(base)
    assert path is None and arch is None
    assert "no architecture" in reason
