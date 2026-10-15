"""
Unit tests for the run layout's path derivation & selection (contract M3a-2).

Dependency-light — checkpoint_paths imports only the torch-free constants module.
"""
import os
from pathlib import Path

import pytest

from src.utils.checkpoint_paths import (
    normalize_conv_type,
    ranking_score_detail,
    ranking_score_from_logs,
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


# --------------------------------------------------------------- run directories (M3a-2)
def test_a_run_directory_is_new_named_and_claimed(tmp_path, monkeypatch):
    from datetime import datetime

    from src.utils import checkpoint_paths as cp

    first = cp.allocate_run_dir(tmp_path, "HGT", "Temp Cosine / Pilot!",
                                now=datetime(2026, 10, 15, 9, 30, 0))
    assert first.parent == tmp_path / "runs" / "hgt"
    assert first.name.startswith("20261015-093000_temp-cosine-pilot_")
    assert sorted(p.name for p in first.iterdir()) == ["checkpoints", "logs"]

    second = cp.allocate_run_dir(tmp_path, "hgt")
    assert second != first and "_run_" in second.name


def test_a_colliding_run_id_is_drawn_again_never_reused(tmp_path, monkeypatch):
    from datetime import datetime

    from src.utils import checkpoint_paths as cp

    ids = iter(["aaaaaa", "aaaaaa", "bbbbbb"])
    monkeypatch.setattr(cp.secrets, "token_hex", lambda n: next(ids))
    when = datetime(2026, 10, 15, 9, 30, 0)

    a = cp.allocate_run_dir(tmp_path, "gat", "x", now=when)
    b = cp.allocate_run_dir(tmp_path, "gat", "x", now=when)

    assert a.name.endswith("_aaaaaa") and b.name.endswith("_bbbbbb")


def test_runs_list_most_recently_trained_first(tmp_path):
    root = tmp_path / "runs"
    _touch(root / "gat" / "20261001-000000_a_111111" / "checkpoints" / "last.pt", 100)
    _touch(root / "hgt" / "20261002-000000_b_222222" / "checkpoints" / "last.pt", 300)
    _touch(root / "hgt" / "20261003-000000_c_333333" / "checkpoints" / "last.pt", 200)
    (root / "sage" / "20261004-000000_empty_444444").mkdir(parents=True)

    from src.utils.checkpoint_paths import list_runs

    names = [r.name for r in list_runs(tmp_path)]
    assert names[:3] == ["20261002-000000_b_222222", "20261003-000000_c_333333",
                         "20261001-000000_a_111111"]
    assert [r.name for r in list_runs(tmp_path, "hgt")] == [
        "20261002-000000_b_222222", "20261003-000000_c_333333"]


def test_selection_is_the_latest_run_then_the_existing_in_run_order(tmp_path):
    """Runs are not ranked against each other by metric: an older run's higher
    score does not win."""
    from src.utils.checkpoint_paths import select_auto_run_checkpoint, select_run_checkpoint

    root = tmp_path / "runs"
    _touch(root / "hgt" / "20261001-000000_old_111111" / "checkpoints" / "model-40-hi.pt", 100)
    newer = root / "hgt" / "20261002-000000_new_222222" / "checkpoints"
    _touch(newer / "model-05-lo.pt", 500)
    _touch(newer / "model-20-hi.pt", 501)
    fn = _score_map({"model-40-hi.pt": 0.95, "model-05-lo.pt": 0.30, "model-20-hi.pt": 0.85})

    path, reason = select_run_checkpoint(tmp_path, "hgt", score_fn=fn)
    assert path == newer / "model-20-hi.pt"
    assert "20261002-000000_new_222222" in reason

    path, architecture, _ = select_auto_run_checkpoint(tmp_path, score_fn=fn)
    assert (path, architecture) == (newer / "model-20-hi.pt", "hgt")


def test_selection_finds_nothing_outside_the_run_layout(tmp_path):
    from src.utils.checkpoint_paths import select_auto_run_checkpoint, select_run_checkpoint

    _touch(tmp_path / "checkpoints" / "hgt" / "last.pt", 100)

    assert select_auto_run_checkpoint(tmp_path)[0] is None
    assert select_run_checkpoint(tmp_path, "hgt")[0] is None


def test_equal_recency_is_ordered_by_name_not_by_the_filesystem(tmp_path):
    """The order is a total key (newest checkpoint, then the run's name), so
    which run is selected never depends on `iterdir()`'s order, which ext4
    gives by directory hash."""
    from src.utils.checkpoint_paths import list_runs, select_auto_run_checkpoint

    root = tmp_path / "runs"
    for arch, name in (("sage", "20261001-000000_a_111111"),
                       ("gat", "20261003-000000_c_333333"),
                       ("hgt", "20261002-000000_b_222222")):
        _touch(root / arch / name / "checkpoints" / "last.pt", 1_700_000_000)

    assert [r.name for r in list_runs(tmp_path)] == [
        "20261003-000000_c_333333", "20261002-000000_b_222222", "20261001-000000_a_111111"]
    assert select_auto_run_checkpoint(tmp_path)[1] == "gat"
    _touch(root / "sage" / "20261001-000000_a_111111" / "checkpoints" / "last.pt",
           1_700_000_900)
    assert select_auto_run_checkpoint(tmp_path)[1] == "sage", "newer training wins over the name"


def test_an_explicit_run_directory_must_be_free(tmp_path):
    from src.utils.checkpoint_paths import RunDirectoryTaken, check_run_dir_free, claim_run_dir

    assert check_run_dir_free(tmp_path / "absent") == tmp_path / "absent"
    (tmp_path / "empty").mkdir()
    assert claim_run_dir(tmp_path / "empty") == tmp_path / "empty"
    assert (tmp_path / "empty" / "checkpoints").is_dir()

    used = tmp_path / "used"
    _touch(used / "config.yaml", 100)
    with pytest.raises(RunDirectoryTaken, match="already holds files"):
        check_run_dir_free(used)
    with pytest.raises(RunDirectoryTaken):
        claim_run_dir(used)
    assert sorted(p.name for p in used.iterdir()) == ["config.yaml"], "a refusal created files"

    _touch(tmp_path / "a-file", 100)
    with pytest.raises(RunDirectoryTaken, match="not a directory"):
        check_run_dir_free(tmp_path / "a-file")


def test_of_two_claims_on_one_free_directory_only_one_wins(tmp_path, monkeypatch):
    """The free check and the creation are two steps; the claim is the
    creation of `checkpoints/`, so a run that passed the check second is still
    refused."""
    from src.utils import checkpoint_paths as cp

    target = tmp_path / "shared"
    real_check = cp.check_run_dir_free

    def check_then_lose_the_race(run_dir):
        path = real_check(run_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "checkpoints").mkdir(exist_ok=True)   # the other run claims it here
        return path

    monkeypatch.setattr(cp, "check_run_dir_free", check_then_lose_the_race)
    with pytest.raises(cp.RunDirectoryTaken, match="claimed by another run"):
        cp.claim_run_dir(target)


def test_the_listing_is_by_run_then_newest_file(tmp_path):
    from src.utils.checkpoint_paths import list_run_checkpoints

    root = tmp_path / "runs"
    old = root / "gat" / "20261001-000000_a_111111"
    new = root / "hgt" / "20261002-000000_b_222222"
    _touch(old / "checkpoints" / "last.pt", 100)
    _touch(new / "checkpoints" / "model-1.pt", 200)
    _touch(new / "checkpoints" / "last.pt", 300)
    _touch(tmp_path / "checkpoints" / "gat" / "last.pt", 900)   # old layout: not listed

    assert [(r.name, p.name) for r, p in list_run_checkpoints(tmp_path)] == [
        (new.name, "last.pt"), (new.name, "model-1.pt"), (old.name, "last.pt")]
    assert [p.name for _, p in list_run_checkpoints(tmp_path, "gat")] == ["last.pt"]
