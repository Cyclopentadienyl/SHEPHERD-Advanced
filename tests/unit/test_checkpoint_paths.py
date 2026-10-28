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


def test_only_accepted_files_are_candidates_the_fallback_included(tmp_path):
    """Automatic selection's record rule (provenance §9.3, hazard 3): a file
    `accept` refuses is never chosen, not even as the `last.pt` fallback; a
    judge that raises refuses the file it judged."""
    d = tmp_path / "gatv2"
    _touch(d / "last.pt", 100)
    _touch(d / "model.pt", 50)
    _touch(d / "unjudged.pt", 200)
    fn = _score_map({"last.pt": 0.9})

    def accept(path):
        if path.name == "unjudged.pt":
            raise RuntimeError("cannot be judged")
        return path.name != "last.pt"

    assert select_checkpoint_in_dir(d, score_fn=fn, accept=accept).name == "model.pt"
    assert select_checkpoint_in_dir(d, accept=lambda p: False) is None


def test_a_file_removed_between_its_read_and_the_ranking_is_never_fatal(tmp_path):
    """Training prunes its top-k files while a reload ranks them: a file gone
    since its score was read sorts last, and the selection still answers."""
    d = tmp_path / "gatv2"
    _touch(d / "last.pt", 100)
    _touch(d / "epoch_a.pt", 200)
    _touch(d / "epoch_b.pt", 300)

    def pruning_score(path):
        if path.name == "epoch_b.pt":
            path.unlink()
            return 0.9
        return 0.5 if path.name == "epoch_a.pt" else None

    assert select_checkpoint_in_dir(d, score_fn=pruning_score).name == "epoch_b.pt"
    e = tmp_path / "hgt"
    _touch(e / "epoch_a.pt", 100)
    gone = _touch(e / "epoch_b.pt", 200)

    def pruning_accept(path):
        if path == gone:
            path.unlink()
        return True

    assert select_checkpoint_in_dir(e, accept=pruning_accept).name == "epoch_a.pt"


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


def test_with_accept_the_record_decides_and_later_runs_are_never_judged(tmp_path):
    """The latest run holding an accepted file wins among the layout keys
    searched (the ShepherdGNN conv types unless told: a task's search names
    every run-layout key); the runs after it are not judged (`accept` may read
    every file it is asked about)."""
    from src.utils.checkpoint_paths import (
        RUN_LAYOUT_KEYS,
        select_auto_run_checkpoint,
        select_run_checkpoint,
    )

    root = tmp_path / "runs"
    _touch(root / "gat" / "20261004-000000_new_444444" / "checkpoints" / "last.pt", 400)
    child = _touch(root / "gatv2" / "20261003-000000_mid_333333" / "checkpoints" / "ok.pt", 300)
    _touch(root / "sage" / "20261002-000000_sage_222222" / "checkpoints" / "ok.pt", 200)
    _touch(root / "gatv2" / "20261001-000000_old_111111" / "checkpoints" / "ok.pt", 100)
    judged = []

    def accept(path):
        judged.append(path.parent.parent.name)
        return path.name == "ok.pt"

    path, key, reason = select_auto_run_checkpoint(tmp_path, accept=accept, keys=RUN_LAYOUT_KEYS)
    assert (path, key) == (child, "gatv2") and "20261003-000000_mid_333333" in reason
    assert set(judged) == {"20261004-000000_new_444444", "20261003-000000_mid_333333"}
    judged.clear()
    # The conv types only (the ShepherdGNN reload): a gatv2 run is never read.
    assert select_auto_run_checkpoint(tmp_path, accept=accept)[1] == "sage"
    assert set(judged) == {"20261004-000000_new_444444", "20261002-000000_sage_222222"}
    assert select_auto_run_checkpoint(tmp_path)[1] == "gat", "without accept: the newest run"
    assert select_run_checkpoint(tmp_path, "sage", accept=accept)[0].parent.parent.name == (
        "20261002-000000_sage_222222")
    assert select_run_checkpoint(tmp_path, "gat", accept=accept) == (
        None, "no run of 'gat' has a checkpoint")
    assert select_auto_run_checkpoint(tmp_path, accept=lambda p: False)[0] is None


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


# ---------------------------------------------------------------------------
# The listing, read (M5-1, BACKLOG 19.3)
# ---------------------------------------------------------------------------
class _Pickled:
    """A pickled class, which torch's weights-only loader refuses."""


def _write_checkpoint(path, payload):
    from src.utils.checkpoint_io import write_checkpoint

    path.parent.mkdir(parents=True, exist_ok=True)
    return write_checkpoint(payload, path)


def test_the_read_listing_keeps_the_order_and_each_reads_digest(tmp_path):
    from src.utils.checkpoint_paths import list_run_checkpoints, read_run_checkpoints
    from src.utils.fingerprint import file_sha256

    old = tmp_path / "runs" / "gat" / "20261001-000000_a_111111" / "checkpoints"
    new = tmp_path / "runs" / "hgt" / "20261002-000000_b_222222" / "checkpoints"
    digest = _write_checkpoint(new / "last.pt", {
        "checkpoint_kind": "training_state", "epoch": 4,
        "logs": {"val_loss": 0.3, "val_mrr": 0.25},
        "training_input_digests": {"split_manifest": "1" * 64, "kg": "2" * 64}})
    _write_checkpoint(old / "last.pt", {"checkpoint_kind": "weights", "epoch": 1})
    os.utime(old / "last.pt", (100, 100))
    os.utime(new / "last.pt", (200, 200))

    entries = read_run_checkpoints(tmp_path)
    assert [(e.run_dir, e.path) for e in entries] == list_run_checkpoints(tmp_path)
    first, second = entries
    assert first.path == new / "last.pt"
    assert first.sha256 == digest == file_sha256(new / "last.pt")
    assert (first.checkpoint_kind, first.epoch) == ("training_state", 4)
    assert first.logs == {"val_loss": 0.3, "val_mrr": 0.25}
    assert first.validation == ("val_mrr", 0.25)
    assert first.dataset_fingerprint == "1" * 64 and first.problem is None
    # A weights file records no logs and no validation: none, never zero.
    assert second.logs == {} and second.validation is None
    assert second.dataset_fingerprint is None and second.problem is None


def test_a_file_the_read_refuses_is_listed_with_its_reason(tmp_path):
    import torch

    from src.utils.checkpoint_paths import read_run_checkpoints

    run = tmp_path / "runs" / "gat" / "20261001-000000_a_111111" / "checkpoints"
    run.mkdir(parents=True)
    (run / "garbage.pt").write_bytes(b"not a checkpoint")
    torch.save({"model": _Pickled()}, run / "pickled.pt")
    torch.save([1, 2, 3], run / "list.pt")

    entries = {e.path.name: e for e in read_run_checkpoints(tmp_path)}
    assert set(entries) == {"garbage.pt", "pickled.pt", "list.pt"}
    for name in ("garbage.pt", "pickled.pt"):
        entry = entries[name]
        assert entry.sha256 is None and entry.logs is None and entry.validation is None
        assert entry.problem and "\n" not in entry.problem and len(entry.problem) <= 330
    assert "UnpicklingError" in entries["pickled.pt"].problem
    # Loads, but is not a record: its digest is kept, its metadata is not there.
    assert entries["list.pt"].sha256 is not None and entries["list.pt"].logs is None
    assert "not a checkpoint record" in entries["list.pt"].problem


def test_the_read_is_weights_only_through_read_checkpoint(tmp_path, monkeypatch):
    import src.utils.checkpoint_io as checkpoint_io
    from src.utils.checkpoint_paths import read_run_checkpoints

    run = tmp_path / "runs" / "sage" / "20261001-000000_a_111111" / "checkpoints"
    _write_checkpoint(run / "a.pt", {"epoch": 1})
    _write_checkpoint(run / "b.pt", {"epoch": 2})
    seen = []
    original = checkpoint_io.read_checkpoint

    def spy(path, **kwargs):
        seen.append((path.name, kwargs))
        return original(path, **kwargs)

    monkeypatch.setattr(checkpoint_io, "read_checkpoint", spy)
    read_run_checkpoints(tmp_path, "sage")
    assert sorted(seen) == [("a.pt", {"map_location": "cpu", "weights_only": True}),
                            ("b.pt", {"map_location": "cpu", "weights_only": True})]


def _counted_reads(monkeypatch):
    import src.utils.checkpoint_io as checkpoint_io

    seen = []
    original = checkpoint_io.read_checkpoint

    def spy(path, **kwargs):
        seen.append(path.name)
        return original(path, **kwargs)

    monkeypatch.setattr(checkpoint_io, "read_checkpoint", spy)
    return seen


def test_a_second_listing_of_unchanged_files_opens_none_of_them(tmp_path, monkeypatch):
    """M5-3 review UI-2: every opening of a model list read every checkpoint
    of the workspace again. An unchanged file is now shown from its last read
    (the same digest and record); a changed one is read again."""
    from collections import OrderedDict

    import src.utils.checkpoint_paths as cp
    from src.utils.fingerprint import file_sha256

    monkeypatch.setattr(cp, "_READS", OrderedDict())
    monkeypatch.setattr(cp, "SETTLED_NS", 0)  # these files were written just now
    run = tmp_path / "runs" / "gat" / "20261001-000000_a_111111" / "checkpoints"
    _write_checkpoint(run / "a.pt", {"epoch": 1, "logs": {"val_mrr": 0.2}})
    _write_checkpoint(run / "b.pt", {"epoch": 2})
    seen = _counted_reads(monkeypatch)

    first = cp.read_run_checkpoints(tmp_path, within=tmp_path)
    assert sorted(seen) == ["a.pt", "b.pt"]
    first_a = next(e for e in first if e.path.name == "a.pt")
    first_a.logs["val_mrr"] = 99.0  # a caller's change never reaches the kept read
    again = cp.read_run_checkpoints(tmp_path, within=tmp_path)
    assert sorted(seen) == ["a.pt", "b.pt"], "an unchanged file was opened again"
    assert [(e.path, e.sha256, e.epoch, e.size_bytes) for e in again] == [
        (e.path, e.sha256, e.epoch, e.size_bytes) for e in first]
    assert next(e for e in again if e.path.name == "a.pt").logs == {"val_mrr": 0.2}

    _write_checkpoint(run / "a.pt", {"epoch": 3, "logs": {"val_mrr": 0.3, "val_loss": 1.0}})
    changed = {e.path.name: e for e in cp.read_run_checkpoints(tmp_path, within=tmp_path)}
    assert sorted(seen) == ["a.pt", "a.pt", "b.pt"]
    assert changed["a.pt"].epoch == 3 and changed["a.pt"].sha256 == file_sha256(run / "a.pt")


def test_a_file_changed_just_before_its_read_is_read_again_every_time(tmp_path, monkeypatch):
    """A rewrite within one tick of the file system's clock could leave the
    file's `stat` as it was, so a read is kept only for a file that had not
    changed for `SETTLED_NS` before it."""
    from collections import OrderedDict

    import src.utils.checkpoint_paths as cp

    monkeypatch.setattr(cp, "_READS", OrderedDict())
    run = tmp_path / "runs" / "gat" / "20261001-000000_a_111111" / "checkpoints"
    _write_checkpoint(run / "a.pt", {"epoch": 1})
    seen = _counted_reads(monkeypatch)
    cp.read_run_checkpoints(tmp_path)
    cp.read_run_checkpoints(tmp_path)
    assert seen == ["a.pt", "a.pt"]


def test_the_kept_reads_are_bounded_and_an_outside_file_is_never_kept(tmp_path, monkeypatch):
    from collections import OrderedDict

    import src.utils.checkpoint_paths as cp

    monkeypatch.setattr(cp, "_READS", OrderedDict())
    monkeypatch.setattr(cp, "SETTLED_NS", 0)
    monkeypatch.setattr(cp, "READS_KEPT", 2)
    run = tmp_path / "ws" / "runs" / "gat" / "20261001-000000_a_111111" / "checkpoints"
    for name in ("a.pt", "b.pt", "c.pt"):
        _write_checkpoint(run / name, {"epoch": 1})
    outside = tmp_path / "elsewhere.pt"
    _write_checkpoint(outside, {"epoch": 9})
    (run / "link.pt").symlink_to(outside)
    cp.read_run_checkpoints(tmp_path / "ws", within=tmp_path / "ws")
    assert len(cp._READS) == 2
    assert all(not key.endswith("link.pt") for key in cp._READS)


def test_importing_the_listing_does_not_import_torch():
    """`checkpoint_paths` promises no torch at import, so the API's lighter
    callers can import it; the read imports torch only when it runs."""
    import subprocess
    import sys

    code = ("import sys; import src.utils.checkpoint_paths; "
            "sys.exit(1 if 'torch' in sys.modules else 0)")
    done = subprocess.run([sys.executable, "-c", code], cwd=str(Path(__file__).parents[2]))
    assert done.returncode == 0


def test_a_link_whose_target_is_gone_is_listed_not_fatal(tmp_path):
    """A `.pt` link whose target is gone, or a checkpoint training removes
    beyond `save_top_k` between the listing and its `stat`, sorts last and stays
    listed; the read says what became of it (M5-1 review)."""
    from src.utils.checkpoint_paths import (
        list_run_checkpoints,
        list_runs,
        read_run_checkpoints,
    )

    run = tmp_path / "runs" / "gat" / "20261001-000000_a_111111"
    _write_checkpoint(run / "checkpoints" / "last.pt", {"epoch": 1})
    (run / "checkpoints" / "gone.pt").symlink_to(run / "checkpoints" / "removed.pt")

    assert list_runs(tmp_path) == [run]
    assert [p.name for _, p in list_run_checkpoints(tmp_path)] == ["last.pt", "gone.pt"]
    good, gone = read_run_checkpoints(tmp_path)
    assert good.problem is None and good.sha256 is not None
    assert gone.path.name == "gone.pt" and gone.sha256 is None
    assert gone.problem.startswith("FileNotFoundError") and gone.size_bytes is None


def test_the_size_is_the_listed_files(tmp_path):
    from src.utils.checkpoint_paths import read_run_checkpoints

    path = tmp_path / "runs" / "gat" / "20261001-000000_a_111111" / "checkpoints" / "last.pt"
    _write_checkpoint(path, {"epoch": 1})
    (entry,) = read_run_checkpoints(tmp_path)
    assert entry.size_bytes == path.stat().st_size > 0


class TestWithinBoundsWhatIsRead:
    """`within`: a file that resolves outside the workspace is listed and never
    opened, so nothing of a file elsewhere is read or shown (M5-1 review)."""

    def _layout(self, tmp_path):
        workspace = tmp_path / "ws"
        run = workspace / "runs" / "gat" / "20261001-000000_a_111111" / "checkpoints"
        _write_checkpoint(run / "inside.pt", {"epoch": 1})
        elsewhere = tmp_path / "elsewhere"
        _write_checkpoint(elsewhere / "secret.pt", {
            "epoch": 9, "logs": {"val_mrr": 0.9},
            "training_input_digests": {"split_manifest": "1" * 64}})
        (run / "link.pt").symlink_to(elsewhere / "secret.pt")
        far = workspace / "runs" / "sage" / "20261002-000000_b_222222"
        far.parent.mkdir(parents=True)
        far.symlink_to(elsewhere.parent / "far_run")
        _write_checkpoint(elsewhere.parent / "far_run" / "checkpoints" / "far.pt", {"epoch": 2})
        return workspace

    def _reads(self, monkeypatch):
        import src.utils.checkpoint_io as checkpoint_io

        original = checkpoint_io.read_checkpoint
        reads = []

        def counting(path, **kwargs):
            reads.append(Path(path).name)
            return original(path, **kwargs)

        monkeypatch.setattr(checkpoint_io, "read_checkpoint", counting)
        return reads

    def test_outside_entries_are_listed_and_never_opened(self, tmp_path, monkeypatch):
        from src.utils.checkpoint_paths import read_run_checkpoints

        workspace = self._layout(tmp_path)
        reads = self._reads(monkeypatch)
        entries = {e.path.name: e for e in read_run_checkpoints(workspace, within=workspace)}
        assert set(entries) == {"inside.pt", "link.pt", "far.pt"}
        assert reads == ["inside.pt"]
        assert entries["inside.pt"].problem is None and entries["inside.pt"].sha256
        for name in ("link.pt", "far.pt"):
            entry = entries[name]
            assert "resolves outside the workspace, so it is not read" in entry.problem
            assert (entry.sha256, entry.epoch, entry.logs, entry.validation,
                    entry.dataset_fingerprint, entry.size_bytes) == (None,) * 6

    def test_without_within_every_file_is_read(self, tmp_path, monkeypatch):
        """The training service's own listing is unchanged: it passes none."""
        from src.utils.checkpoint_paths import read_run_checkpoints

        workspace = self._layout(tmp_path)
        reads = self._reads(monkeypatch)
        read_run_checkpoints(workspace)
        assert sorted(reads) == ["far.pt", "inside.pt", "link.pt"]
