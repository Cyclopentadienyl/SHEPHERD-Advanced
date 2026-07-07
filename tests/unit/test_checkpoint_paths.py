"""
Unit tests for architecture-scoped checkpoint path derivation & selection.

Dependency-light — checkpoint_paths imports only the torch-free constants module.
"""
import os
from pathlib import Path

import pytest

from src.utils.checkpoint_paths import (
    normalize_conv_type,
    ranking_score_detail,
    ranking_score_from_logs,
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
def test_normalize_conv_type_valid_and_missing():
    assert normalize_conv_type("HGT") == "hgt"
    assert normalize_conv_type("  gat ") == "gat"
    assert normalize_conv_type("sage") == "sage"
    # missing/empty -> default is fine
    assert normalize_conv_type(None) == "gat"
    assert normalize_conv_type("") == "gat"
    assert normalize_conv_type("   ") == "gat"


@pytest.mark.parametrize("bad", ["bogus", "gatv2", "gps", "hg", "GATT"])
def test_normalize_conv_type_explicit_invalid_raises(bad):
    # An explicit but unsupported value must fail loud, not silently become GAT.
    with pytest.raises(ValueError, match="Unsupported conv_type"):
        normalize_conv_type(bad)


# --------------------------------------------------------------- resolve dir
def test_resolve_checkpoint_dir_auto():
    assert resolve_checkpoint_dir("ws", "hgt") == Path("ws/checkpoints/hgt")
    assert resolve_checkpoint_dir("ws", "GAT") == Path("ws/checkpoints/gat")
    assert resolve_checkpoint_dir("ws", None) == Path("ws/checkpoints/gat")  # default


def test_resolve_checkpoint_dir_invalid_conv_raises():
    with pytest.raises(ValueError, match="Unsupported conv_type"):
        resolve_checkpoint_dir("ws", "gatv2")


def test_resolve_checkpoint_dir_explicit_wins_verbatim():
    # explicit dir is returned unchanged — no conv_type appended, no validation
    assert resolve_checkpoint_dir("ws", "hgt", "/custom/ckpts") == Path("/custom/ckpts")
    # even an odd conv_type is irrelevant when an explicit dir is given
    assert resolve_checkpoint_dir("ws", "gatv2", "/custom") == Path("/custom")


# ---------------------------------------------------------- ranking score reader
def test_ranking_score_key_priority():
    assert ranking_score_detail({"val_mrr": 0.7, "val_hits@10": 0.9}) == ("val_mrr", 0.7)
    # val_mrr missing -> falls back to val_hits@10 (Codex case 7)
    assert ranking_score_detail({"val_hits@10": 0.75, "val_hits@1": 0.2}) == (
        "val_hits@10", 0.75,
    )
    assert ranking_score_detail({"val_hits@1": 0.3}) == ("val_hits@1", 0.3)


def test_ranking_score_ignores_invalid():
    assert ranking_score_detail({"val_loss": 8.6}) is None  # loss is not a ranking metric
    assert ranking_score_detail({"val_mrr": float("nan")}) is None  # NaN ignored
    assert ranking_score_detail({"val_mrr": True}) is None  # bool is not a numeric score
    assert ranking_score_detail({}) is None
    assert ranking_score_detail(None) is None
    assert ranking_score_from_logs({"val_hits@10": 0.75}) == 0.75


# --------------------------------------------------------------- select in dir
def _score_map(mapping):
    """Build a score_fn from {filename: score | 'raise'} — no torch needed."""
    def fn(path):
        value = mapping.get(path.name)
        if value == "raise":
            raise RuntimeError("unreadable checkpoint")
        return value
    return fn


def test_select_by_score_picks_highest(tmp_path):
    # The event case: epoch-0 has the LOWEST ranking score and must not be chosen.
    d = tmp_path / "hgt"
    for name in ("model-00-0.0174.pt", "model-21-0.9200.pt", "last.pt"):
        _touch(d / name, 100)
    fn = _score_map({
        "model-00-0.0174.pt": 0.0174,
        "model-21-0.9200.pt": 0.9200,
        "last.pt": 0.9000,
    })
    assert select_checkpoint_in_dir(d, score_fn=fn).name == "model-21-0.9200.pt"


def test_select_trusts_metadata_not_filename(tmp_path):
    # Filename number is misleading; the metadata score must win (locks in the fix).
    d = tmp_path / "hgt"
    _touch(d / "model-00-9999.pt", 100)
    _touch(d / "model-10-0001.pt", 100)
    fn = _score_map({"model-00-9999.pt": 0.01, "model-10-0001.pt": 0.90})
    assert select_checkpoint_in_dir(d, score_fn=fn).name == "model-10-0001.pt"


def test_select_falls_back_to_last(tmp_path):
    d = tmp_path / "hgt"
    _touch(d / "model-01-x.pt", 100)
    _touch(d / "last.pt", 100)
    # no score_fn -> last.pt
    assert select_checkpoint_in_dir(d).name == "last.pt"


def test_select_all_none_scores_fall_back_to_last(tmp_path):
    d = tmp_path / "hgt"
    _touch(d / "model-01-x.pt", 100)
    _touch(d / "last.pt", 100)
    assert select_checkpoint_in_dir(d, score_fn=_score_map({})).name == "last.pt"


def test_select_bad_checkpoint_does_not_block_others(tmp_path):
    d = tmp_path / "hgt"
    _touch(d / "bad.pt", 100)
    _touch(d / "good.pt", 100)
    fn = _score_map({"bad.pt": "raise", "good.pt": 0.8})
    assert select_checkpoint_in_dir(d, score_fn=fn).name == "good.pt"


def test_select_nan_score_ignored(tmp_path):
    d = tmp_path / "hgt"
    _touch(d / "nan.pt", 100)
    _touch(d / "good.pt", 100)
    fn = _score_map({"nan.pt": float("nan"), "good.pt": 0.7})
    assert select_checkpoint_in_dir(d, score_fn=fn).name == "good.pt"


def test_select_falls_back_to_newest(tmp_path):
    d = tmp_path / "hgt"
    _touch(d / "epoch_a.pt", 100)
    _touch(d / "epoch_b.pt", 200)  # newest, no last/score
    assert select_checkpoint_in_dir(d).name == "epoch_b.pt"


def test_select_none_when_empty_or_missing(tmp_path):
    assert select_checkpoint_in_dir(tmp_path / "nope") is None
    (tmp_path / "empty").mkdir()
    assert select_checkpoint_in_dir(tmp_path / "empty") is None


# --------------------------------------------------------------- auto selection
def test_auto_picks_latest_arch_then_best_within(tmp_path):
    # auto = latest-trained architecture, then best-scoring checkpoint within it.
    # hgt has a HIGHER score but is older -> not chosen (metrics not compared
    # across architectures).
    base = tmp_path / "checkpoints"
    _touch(base / "hgt" / "model-40-hi.pt", 100)   # score 0.95 but older arch
    _touch(base / "gat" / "model-05-lo.pt", 500)
    _touch(base / "gat" / "model-20-hi.pt", 501)   # newer arch, its best
    fn = _score_map({
        "model-40-hi.pt": 0.95,
        "model-05-lo.pt": 0.30,
        "model-20-hi.pt": 0.85,
    })
    path, arch, reason = select_auto_checkpoint(base, score_fn=fn)
    assert arch == "gat"
    assert path.name == "model-20-hi.pt"
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
