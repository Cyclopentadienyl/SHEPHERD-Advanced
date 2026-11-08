"""
Unit tests for the run layout's path derivation & selection (contract M3a-2).

Dependency-light — checkpoint_paths imports only the torch-free constants module.
"""
import os
from pathlib import Path

import pytest

from src.utils.checkpoint_paths import (
    ranking_score_detail,
    select_checkpoint_in_dir,
)


def _touch(path: Path, mtime: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    os.utime(path, (mtime, mtime))
    return path


# --------------------------------------------------------------- resolve dir
# ---------------------------------------------------------- ranking score reader
def test_ranking_score_key_priority():
    assert ranking_score_detail({"val_pooled_mrr": 0.7, "val_lp_accuracy": 0.9}) == (
        "val_pooled_mrr", 0.7)
    assert ranking_score_detail({"val_lp_accuracy": 0.75}) == ("val_lp_accuracy", 0.75)


def test_the_ranking_keys_are_the_stages_selection_metrics_and_never_a_loss():
    """N9 (K8-7b): a checkpoint is ranked by the figure its stage selects by,
    a base's LP accuracy and a task model's pooled MRR, and by nothing else: a
    log holding only figures no stage of this build selects by (an earlier
    trainer's MRR or Hits@K) gives no ranking figure, and a loss never does."""
    from src.training.finetune_task import FINETUNE_SELECTION_METRIC
    from src.training.lp_task import BASE_SELECTION_METRIC
    from src.utils.checkpoint_paths import RANKING_SCORE_KEYS

    assert set(RANKING_SCORE_KEYS) == {BASE_SELECTION_METRIC, FINETUNE_SELECTION_METRIC}
    assert "val_loss" not in RANKING_SCORE_KEYS
    assert ranking_score_detail({"val_mrr": 0.7, "val_hits@10": 0.9, "val_loss": 0.1}) is None


def test_ranking_score_ignores_invalid():
    assert ranking_score_detail({"val_loss": 8.6}) is None  # loss is not a ranking metric
    assert ranking_score_detail({"val_pooled_mrr": float("nan")}) is None  # NaN ignored
    assert ranking_score_detail({"val_pooled_mrr": True}) is None  # bool is not a numeric score
    # An integer past a float's range (the safe loader reads one) has no finite
    # figure; `math.isfinite` raised OverflowError out of the list.
    assert ranking_score_detail({"val_pooled_mrr": 2 ** 2000, "val_lp_accuracy": 2 ** 200}) == (
        "val_lp_accuracy", float(2 ** 200))
    assert ranking_score_detail({}) is None
    assert ranking_score_detail(None) is None


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

    first = cp.claim_run_dir(cp.propose_run_dir(tmp_path, "HGT", "Temp Cosine / Pilot!",
                                                now=datetime(2026, 10, 15, 9, 30, 0)),
                             data_dir=tmp_path)
    assert first.parent == tmp_path / "runs" / "hgt"
    assert first.name.startswith("20261015-093000_temp-cosine-pilot_")
    assert sorted(p.name for p in first.iterdir()) == ["checkpoints", "logs"]

    second = cp.claim_run_dir(cp.propose_run_dir(tmp_path, "hgt"), data_dir=tmp_path)
    assert second != first and "_run_" in second.name


@pytest.mark.parametrize("key", ["", None, "  "])
def test_a_run_without_a_layout_key_is_never_filed_under_a_default(tmp_path, key):
    """TM4: a run is filed under its model spec's encoder kind. An empty key
    used to file it under the ShepherdGNN default `gat`; it raises, and no
    directory is proposed or made."""
    from src.utils import checkpoint_paths as cp

    for call in (lambda: cp.normalize_run_layout_key(key),
                 lambda: cp.runs_root(tmp_path, key),
                 lambda: cp.propose_run_dir(tmp_path, key, "x")):
        with pytest.raises(ValueError, match="layout key is required"):
            call()
    assert not (tmp_path / "runs").exists()


def test_a_colliding_run_id_is_drawn_again_never_reused(tmp_path, monkeypatch):
    from datetime import datetime

    from src.utils import checkpoint_paths as cp

    ids = iter(["aaaaaa", "aaaaaa", "bbbbbb"])
    monkeypatch.setattr(cp.secrets, "token_hex", lambda n: next(ids))
    when = datetime(2026, 10, 15, 9, 30, 0)

    a = cp.claim_run_dir(cp.propose_run_dir(tmp_path, "gatv2", "x", now=when), data_dir=tmp_path)
    b = cp.propose_run_dir(tmp_path, "gatv2", "x", now=when)

    assert a.name.endswith("_aaaaaa") and b.name.endswith("_bbbbbb")


def test_runs_list_most_recently_trained_first(tmp_path):
    root = tmp_path / "runs"
    _touch(root / "gatv2" / "20261001-000000_a_111111" / "checkpoints" / "last.pt", 100)
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
    from src.utils.checkpoint_paths import select_auto_run_checkpoint

    root = tmp_path / "runs"
    _touch(root / "hgt" / "20261001-000000_old_111111" / "checkpoints" / "model-40-hi.pt", 100)
    newer = root / "hgt" / "20261002-000000_new_222222" / "checkpoints"
    _touch(newer / "model-05-lo.pt", 500)
    _touch(newer / "model-20-hi.pt", 501)
    fn = _score_map({"model-40-hi.pt": 0.95, "model-05-lo.pt": 0.30, "model-20-hi.pt": 0.85})

    path, architecture, reason = select_auto_run_checkpoint(tmp_path, score_fn=fn)
    assert (path, architecture) == (newer / "model-20-hi.pt", "hgt")
    assert "20261002-000000_new_222222" in reason


def test_with_accept_the_record_decides_and_later_runs_are_never_judged(tmp_path):
    """The latest run holding an accepted file wins among every run-layout
    key; the runs after it are not judged (`accept` may read every file it is
    asked about). The default searches every key (K8-4 retired the ShepherdGNN
    conv-type default), so the accepted `gatv2` child is chosen over an older
    accepted `sage` run."""
    from src.utils.checkpoint_paths import select_auto_run_checkpoint

    root = tmp_path / "runs"
    _touch(root / "hgt" / "20261004-000000_new_444444" / "checkpoints" / "last.pt", 400)
    child = _touch(root / "gatv2" / "20261003-000000_mid_333333" / "checkpoints" / "ok.pt", 300)
    _touch(root / "sage" / "20261002-000000_sage_222222" / "checkpoints" / "ok.pt", 200)
    _touch(root / "gatv2" / "20261001-000000_old_111111" / "checkpoints" / "ok.pt", 100)
    judged = []

    def accept(path):
        judged.append(path.parent.parent.name)
        return path.name == "ok.pt"

    path, key, reason = select_auto_run_checkpoint(tmp_path, accept=accept)
    assert (path, key) == (child, "gatv2") and "20261003-000000_mid_333333" in reason
    assert set(judged) == {"20261004-000000_new_444444", "20261003-000000_mid_333333"}
    assert select_auto_run_checkpoint(tmp_path)[1] == "hgt", "without accept: the newest run"
    assert select_auto_run_checkpoint(tmp_path, accept=lambda p: False)[0] is None


def test_selection_finds_nothing_outside_the_run_layout(tmp_path):
    from src.utils.checkpoint_paths import select_auto_run_checkpoint

    _touch(tmp_path / "checkpoints" / "hgt" / "last.pt", 100)

    assert select_auto_run_checkpoint(tmp_path)[0] is None


class TestSelectionKeepsTheListingsBoundary:
    """Automatic selection reads only inside the workspace, as the listing does
    (`checkpoint_not_read`, the reviewer's K8-1 recheck): before the fix the listing
    refused a checkpoint that resolves outside the workspace while selection
    judged, read and chose it. A run directory or a layout directory linked
    out of the workspace is no candidate, its files are never handed to
    `accept` (which reads them) and the reason names each in the listing's
    words; a run inside is chosen as before."""

    REASON = "resolves outside the workspace, so it is not read"

    def _accept(self):
        judged = []

        def accept(path):
            judged.append(path.name)
            return True

        return judged, accept

    def test_a_run_linked_outside_is_no_candidate(self, tmp_path):
        from src.utils.checkpoint_paths import select_auto_run_checkpoint

        workspace = tmp_path / "ws"
        inside = _touch(workspace / "runs" / "hgt" / "20261001-000000_in_111111"
                        / "checkpoints" / "inside.pt", 100)
        _touch(tmp_path / "elsewhere" / "far_run" / "checkpoints" / "far.pt", 900)
        link = workspace / "runs" / "hgt" / "20261009-000000_far_999999"
        link.symlink_to(tmp_path / "elsewhere" / "far_run", target_is_directory=True)
        judged, accept = self._accept()

        path, key, reason = select_auto_run_checkpoint(workspace, accept=accept)
        assert (path, key) == (inside, "hgt") and judged == ["inside.pt"]
        assert f"not read: 20261009-000000_far_999999/far.pt {self.REASON}" in reason

    def test_a_layout_directory_linked_outside_is_no_candidate(self, tmp_path):
        from src.utils.checkpoint_paths import select_auto_run_checkpoint

        workspace = tmp_path / "ws"
        (workspace / "runs").mkdir(parents=True)
        _touch(tmp_path / "elsewhere" / "gatv2" / "20261009-000000_far_999999"
               / "checkpoints" / "last.pt", 900)
        (workspace / "runs" / "gatv2").symlink_to(tmp_path / "elsewhere" / "gatv2",
                                                  target_is_directory=True)
        judged, accept = self._accept()

        path, key, reason = select_auto_run_checkpoint(workspace, accept=accept)
        assert (path, key) == (None, None) and judged == []
        assert reason == (f"no run has a checkpoint; not read: "
                          f"20261009-000000_far_999999/last.pt {self.REASON}")

    def test_a_reason_names_a_few_files_and_counts_the_rest(self, tmp_path):
        """A runs directory linked out of the workspace with many runs gives a
        reason of bounded length: the first `NOT_READ_NAMED` files, then how
        many more; a list given as `not_read` holds every one."""
        from src.utils.checkpoint_paths import NOT_READ_NAMED, select_auto_run_checkpoint

        workspace = tmp_path / "ws"
        workspace.mkdir()
        for index in range(NOT_READ_NAMED + 4):
            _touch(tmp_path / "elsewhere" / "hgt" / f"2026100{index}-000000_r_00000{index}"
                   / "checkpoints" / "last.pt", 100 + index)
        (workspace / "runs").symlink_to(tmp_path / "elsewhere", target_is_directory=True)

        not_read = []
        path, key, reason = select_auto_run_checkpoint(workspace, not_read=not_read)
        assert (path, key) == (None, None) and len(not_read) == NOT_READ_NAMED + 4
        assert reason.count(self.REASON) == NOT_READ_NAMED
        assert reason.endswith("; and 4 more file(s) not read")

    def test_the_listing_and_the_selection_give_one_reason(self, tmp_path):
        from src.utils.checkpoint_paths import read_run_checkpoints, select_auto_run_checkpoint

        workspace = tmp_path / "ws"
        run = workspace / "runs" / "hgt" / "20261001-000000_a_111111" / "checkpoints"
        run.mkdir(parents=True)
        _write_checkpoint(tmp_path / "elsewhere.pt", {"epoch": 1})
        (run / "link.pt").symlink_to(tmp_path / "elsewhere.pt")

        (entry,) = read_run_checkpoints(workspace, within=workspace)
        path, _, reason = select_auto_run_checkpoint(workspace)
        assert path is None
        assert reason == f"no run has a checkpoint; not read: {entry.problem}"


def test_equal_recency_is_ordered_by_name_not_by_the_filesystem(tmp_path):
    """The order is a total key (newest checkpoint, then the run's name), so
    which run is selected never depends on `iterdir()`'s order, which ext4
    gives by directory hash."""
    from src.utils.checkpoint_paths import list_runs, select_auto_run_checkpoint

    root = tmp_path / "runs"
    for arch, name in (("sage", "20261001-000000_a_111111"),
                       ("gatv2", "20261003-000000_c_333333"),
                       ("hgt", "20261002-000000_b_222222")):
        _touch(root / arch / name / "checkpoints" / "last.pt", 1_700_000_000)

    assert [r.name for r in list_runs(tmp_path)] == [
        "20261003-000000_c_333333", "20261002-000000_b_222222", "20261001-000000_a_111111"]
    assert select_auto_run_checkpoint(tmp_path)[1] == "gatv2"
    _touch(root / "sage" / "20261001-000000_a_111111" / "checkpoints" / "last.pt",
           1_700_000_900)
    assert select_auto_run_checkpoint(tmp_path)[1] == "sage", (
        "newer training wins over the name")


@pytest.mark.parametrize("reverse", [False, True])
def test_one_run_name_in_two_directories_keeps_one_order(tmp_path, monkeypatch, reverse):
    """N5 (C2): every directory under runs/ is listed, so one run name can sit
    in two of them at one recency. The order is a total key (newest checkpoint,
    the run's name, then its directory's name), so the list and automatic
    selection are the same whichever order `iterdir()` gives."""
    from src.utils.checkpoint_paths import list_runs, select_auto_run_checkpoint

    root = tmp_path / "runs"
    for layout in ("gatv2", "hgt"):
        _touch(root / layout / "20261001-000000_run_aaaaaa" / "checkpoints" / "last.pt", 100)
    original = Path.iterdir

    def iterdir(self):
        entries = sorted(original(self))
        return iter(entries[::-1] if reverse else entries)

    monkeypatch.setattr(Path, "iterdir", iterdir)
    assert [(r.parent.name, r.name) for r in list_runs(tmp_path)] == [
        ("hgt", "20261001-000000_run_aaaaaa"), ("gatv2", "20261001-000000_run_aaaaaa")]
    assert select_auto_run_checkpoint(tmp_path)[1] == "hgt"


def test_any_directory_under_runs_is_listed_and_a_name_is_never_joined(tmp_path):
    """N1/N4 at the listing (C1, C9): a run under a directory no writer of this
    build uses (an earlier build's ``gat``, one an operator named) is listed
    like any other, and a name given to keep one directory is matched against
    the listed names, never joined onto a path: ``.``, ``..``, ``a/b`` and
    ``""`` name no directory."""
    from src.utils.checkpoint_paths import list_run_checkpoints, list_runs

    ws = tmp_path / "ws"
    root = ws / "runs"
    for layout, day, mtime in (("gat", 1, 100), ("my_runs", 2, 200), ("gatv2", 3, 300)):
        _touch(root / layout / f"2026100{day}-000000_r_aaaaaa" / "checkpoints" / "last.pt",
               mtime)
    (root / "not_a_directory").write_text("x", encoding="utf-8")
    assert [r.parent.name for r in list_runs(ws)] == ["gatv2", "my_runs", "gat"]
    assert [r.parent.name for r in list_runs(ws, "my_runs")] == ["my_runs"]
    for name in (".", "..", "a/b", "", "runs", "my_runs/..", "GAT"):
        assert list_runs(ws, name) == [], name
        assert list_run_checkpoints(ws, name) == [], name
    assert list_runs(tmp_path / "no_such_workspace") == []


def test_new_runs_are_filed_under_an_encoder_kind_only(tmp_path):
    """N3: the writer's closed set is the model spec's encoder kinds
    (`RUN_LAYOUT_KEYS == ENCODER_KINDS`); any other name, an operator's
    directory name included, is never proposed for a new run, while the
    listing reads it."""
    from src.config.model_types import ENCODER_KINDS
    from src.utils import checkpoint_paths as cp

    assert cp.RUN_LAYOUT_KEYS == ENCODER_KINDS
    for kind in ENCODER_KINDS:
        assert cp.propose_run_dir(tmp_path, kind).parent == tmp_path / "runs" / kind
    with pytest.raises(ValueError, match="Unsupported run layout key 'my_runs'"):
        cp.propose_run_dir(tmp_path, "my_runs")
    assert not (tmp_path / "runs" / "my_runs").exists()


def test_an_explicit_run_directory_must_be_free(tmp_path):
    from src.utils.checkpoint_paths import RunDirectoryTaken, check_run_dir_free, claim_run_dir

    workspace = tmp_path / "ws"
    assert check_run_dir_free(tmp_path / "absent", data_dir=workspace) == tmp_path / "absent"
    (tmp_path / "empty").mkdir()
    assert claim_run_dir(tmp_path / "empty", data_dir=workspace) == tmp_path / "empty"
    assert (tmp_path / "empty" / "checkpoints").is_dir()

    used = tmp_path / "used"
    _touch(used / "config.yaml", 100)
    with pytest.raises(RunDirectoryTaken, match="already holds files"):
        check_run_dir_free(used, data_dir=workspace)
    with pytest.raises(RunDirectoryTaken):
        claim_run_dir(used, data_dir=workspace)
    assert sorted(p.name for p in used.iterdir()) == ["config.yaml"], "a refusal created files"

    _touch(tmp_path / "a-file", 100)
    with pytest.raises(RunDirectoryTaken, match="not a directory"):
        check_run_dir_free(tmp_path / "a-file", data_dir=workspace)


class TestWritingKeepsTheWorkspaceBoundary:
    """A run directory placed in the workspace must resolve inside it, the
    boundary the listing and automatic selection read by (the reviewer's K8-1
    recheck: since bf7cd22 selection skipped a run behind a link out of the
    workspace, while `propose_run_dir` and `claim_run_dir` still created one
    there, so a finetune wrote a run the reload never chose). Refused before
    anything is written, saying why and what to do; a run inside the
    workspace, and an explicit run directory outside it, are as before."""

    REASON = "resolves outside the workspace, so it is not written"

    def _linked(self, tmp_path, linked):
        workspace = tmp_path / "ws"
        elsewhere = tmp_path / "elsewhere"
        if linked == "runs":
            workspace.mkdir()
            (elsewhere / "runs").mkdir(parents=True)
            (workspace / "runs").symlink_to(elsewhere / "runs", target_is_directory=True)
        else:
            (workspace / "runs").mkdir(parents=True)
            (elsewhere / "gatv2").mkdir(parents=True)
            (workspace / "runs" / "gatv2").symlink_to(elsewhere / "gatv2",
                                                      target_is_directory=True)
        return workspace, elsewhere

    @pytest.mark.parametrize("linked", ["runs", "layout"])
    def test_a_run_behind_a_link_out_of_the_workspace_is_refused(self, tmp_path, linked):
        from src.utils.checkpoint_paths import (
            check_run_dir_free,
            claim_run_dir,
            propose_run_dir,
            runs_root,
        )

        workspace, elsewhere = self._linked(tmp_path, linked)
        before = sorted(elsewhere.rglob("*"))
        with pytest.raises(ValueError, match=self.REASON) as refused:
            propose_run_dir(workspace, "gatv2", "child")
        assert "Keep the workspace's runs/ and its layout directories inside the workspace" \
            in str(refused.value)
        assert "explicit checkpoint_path" in str(refused.value)
        # An explicit run directory under the linked layout: refused at the
        # start's check and at the claim, before anything is created.
        named = runs_root(workspace, "gatv2") / "20261101-000000_child_abcdef"
        with pytest.raises(ValueError, match=self.REASON):
            check_run_dir_free(named, data_dir=workspace)
        with pytest.raises(ValueError, match=self.REASON):
            claim_run_dir(named, data_dir=workspace)
        assert sorted(elsewhere.rglob("*")) == before, "a refused claim wrote outside"

    def test_a_run_inside_the_workspace_is_claimed_as_before(self, tmp_path):
        from src.utils.checkpoint_paths import claim_run_dir, propose_run_dir, run_dir_outside

        workspace = tmp_path / "ws"
        run = claim_run_dir(propose_run_dir(workspace, "gatv2", "child"), data_dir=workspace)
        assert run.parent == workspace / "runs" / "gatv2" and (run / "checkpoints").is_dir()
        # A workspace reached through a link of its own is inside itself.
        (tmp_path / "current").symlink_to(workspace, target_is_directory=True)
        assert claim_run_dir(propose_run_dir(tmp_path / "current", "gatv2"),
                             data_dir=tmp_path / "current").is_dir()
        assert run_dir_outside(workspace / "runs" / "gatv2" / "x", workspace) is None

    def test_a_run_directory_behind_a_link_loop_is_refused_with_the_loop_named(self, tmp_path):
        """A loop on the run's path is a broken link, not a runs/ placed on
        another disk: the reason says so and gives the advice that fits."""
        from src.utils.checkpoint_paths import run_dir_outside

        workspace = tmp_path / "ws"
        (workspace / "runs").mkdir(parents=True)
        loop = workspace / "runs" / "gatv2"
        loop.symlink_to(loop, target_is_directory=True)
        reason = run_dir_outside(loop / "child", workspace)
        assert reason is not None and "cannot be resolved" in reason
        assert "is not written" in reason and "loops or is broken" in reason
        assert "layout directories inside the workspace" not in reason

    def test_an_explicit_run_directory_outside_the_workspace_is_as_before(self, tmp_path):
        """Not in the layout: nothing selects from it automatically, and its
        checkpoints are served by an explicit path, as before."""
        from src.utils.checkpoint_paths import claim_run_dir

        assert claim_run_dir(tmp_path / "runs-elsewhere" / "r",
                             data_dir=tmp_path / "ws").is_dir()


def test_of_two_claims_on_one_free_directory_only_one_wins(tmp_path, monkeypatch):
    """The free check and the creation are two steps; the claim is the
    creation of `checkpoints/`, so a run that passed the check second is still
    refused."""
    from src.utils import checkpoint_paths as cp

    target = tmp_path / "shared"
    real_check = cp.check_run_dir_free

    def check_then_lose_the_race(run_dir, **kwargs):
        path = real_check(run_dir, **kwargs)
        path.mkdir(parents=True, exist_ok=True)
        (path / "checkpoints").mkdir(exist_ok=True)   # the other run claims it here
        return path

    monkeypatch.setattr(cp, "check_run_dir_free", check_then_lose_the_race)
    with pytest.raises(cp.RunDirectoryTaken, match="claimed by another run"):
        cp.claim_run_dir(target, data_dir=tmp_path)


def test_the_listing_is_by_run_then_newest_file(tmp_path):
    from src.utils.checkpoint_paths import list_run_checkpoints

    root = tmp_path / "runs"
    old = root / "gatv2" / "20261001-000000_a_111111"
    new = root / "hgt" / "20261002-000000_b_222222"
    _touch(old / "checkpoints" / "last.pt", 100)
    _touch(new / "checkpoints" / "model-1.pt", 200)
    _touch(new / "checkpoints" / "last.pt", 300)
    _touch(tmp_path / "checkpoints" / "gat" / "last.pt", 900)   # old layout: not listed

    assert [(r.name, p.name) for r, p in list_run_checkpoints(tmp_path)] == [
        (new.name, "last.pt"), (new.name, "model-1.pt"), (old.name, "last.pt")]
    assert [p.name for _, p in list_run_checkpoints(tmp_path, "gatv2")] == ["last.pt"]


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

    old = tmp_path / "runs" / "gatv2" / "20261001-000000_a_111111" / "checkpoints"
    new = tmp_path / "runs" / "hgt" / "20261002-000000_b_222222" / "checkpoints"
    digest = _write_checkpoint(new / "last.pt", {
        "checkpoint_kind": "training_state", "epoch": 4,
        "logs": {"val_loss": 0.3, "val_pooled_mrr": 0.25},
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
    assert first.logs == {"val_loss": 0.3, "val_pooled_mrr": 0.25}
    assert first.validation == ("val_pooled_mrr", 0.25)
    assert first.dataset_fingerprint == "1" * 64 and first.problem is None
    # A weights file records no logs and no validation: none, never zero.
    assert second.logs == {} and second.validation is None
    assert second.dataset_fingerprint is None and second.problem is None


def test_a_file_the_read_refuses_is_listed_with_its_reason(tmp_path):
    import torch

    from src.utils.checkpoint_paths import read_run_checkpoints

    run = tmp_path / "runs" / "gatv2" / "20261001-000000_a_111111" / "checkpoints"
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
    """Each file through `read_checkpoint`, weights only, on the CPU; bound to
    the workspace by the file it opens exactly when the listing is (the
    reviewer's K8-4r recheck, P2-A), unbounded otherwise."""
    from collections import OrderedDict

    import src.utils.checkpoint_io as checkpoint_io
    import src.utils.checkpoint_paths as cp
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
    for within, bound in ((None, None), (tmp_path, tmp_path.resolve())):
        monkeypatch.setattr(cp, "_READS", OrderedDict())
        seen.clear()
        read_run_checkpoints(tmp_path, "sage", within=within)
        options = {"map_location": "cpu", "weights_only": True, "within": bound}
        assert sorted(seen, key=lambda s: s[0]) == [("a.pt", options), ("b.pt", options)]


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
    run = tmp_path / "runs" / "gatv2" / "20261001-000000_a_111111" / "checkpoints"
    _write_checkpoint(run / "a.pt", {"epoch": 1, "logs": {"val_pooled_mrr": 0.2}})
    _write_checkpoint(run / "b.pt", {"epoch": 2})
    seen = _counted_reads(monkeypatch)

    first = cp.read_run_checkpoints(tmp_path, within=tmp_path)
    assert sorted(seen) == ["a.pt", "b.pt"]
    first_a = next(e for e in first if e.path.name == "a.pt")
    first_a.logs["val_pooled_mrr"] = 99.0  # a caller's change never reaches the kept read
    again = cp.read_run_checkpoints(tmp_path, within=tmp_path)
    assert sorted(seen) == ["a.pt", "b.pt"], "an unchanged file was opened again"
    assert [(e.path, e.sha256, e.epoch, e.size_bytes) for e in again] == [
        (e.path, e.sha256, e.epoch, e.size_bytes) for e in first]
    assert next(e for e in again if e.path.name == "a.pt").logs == {"val_pooled_mrr": 0.2}

    _write_checkpoint(run / "a.pt", {"epoch": 3, "logs": {"val_pooled_mrr": 0.3, "val_loss": 1.0}})
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
    run = tmp_path / "runs" / "gatv2" / "20261001-000000_a_111111" / "checkpoints"
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
    run = tmp_path / "ws" / "runs" / "gatv2" / "20261001-000000_a_111111" / "checkpoints"
    for name in ("a.pt", "b.pt", "c.pt"):
        _write_checkpoint(run / name, {"epoch": 1})
    outside = tmp_path / "elsewhere.pt"
    _write_checkpoint(outside, {"epoch": 9})
    (run / "link.pt").symlink_to(outside)
    cp.read_run_checkpoints(tmp_path / "ws", within=tmp_path / "ws")
    assert len(cp._READS) == 2
    assert all(not path.endswith("link.pt") for path, _ in cp._READS)


def test_a_bounded_listing_never_shows_what_an_unbounded_read_kept_of_an_outside_file(
        tmp_path, monkeypatch):
    """The K8-4r2 verifier's cache recall: the training list (unbounded) read
    a name that was a link to a file outside the workspace and kept that read.
    The name then became a file inside, so the server's list (bounded) passed
    its name check; between that check and its `stat` the name was a link to
    the outside file again, and the list showed the kept read, the outside
    file's digest with no problem, without opening anything. A kept read is
    now recalled only by a listing bound as that read was, so the bounded list
    reads the name itself, bound to the workspace: not read, no digest."""
    from collections import OrderedDict

    import src.utils.checkpoint_paths as cp
    from src.utils.fingerprint import file_sha256

    monkeypatch.setattr(cp, "_READS", OrderedDict())
    monkeypatch.setattr(cp, "SETTLED_NS", 0)  # stands for files settled before their reads
    ws = tmp_path / "ws"
    run = "20261001-000000_a_111111"
    name = ws / "runs" / "gatv2" / run / "checkpoints" / "last.pt"
    outside = tmp_path / "elsewhere" / "last.pt"
    _write_checkpoint(outside, {"epoch": 9})
    outside_sha = file_sha256(outside)
    name.parent.mkdir(parents=True)
    name.symlink_to(outside)
    unbounded = cp.read_run_checkpoints(ws)
    assert unbounded[0].sha256 == outside_sha, "the unbounded list read the outside file"

    name.unlink()
    _write_checkpoint(name, {"epoch": 1})
    real_identity, swapped = cp._identity, []

    def identity_after_swap(path):
        if path == name and not swapped:
            swapped.append(path)
            pending = name.with_suffix(".pending")
            pending.symlink_to(outside)
            os.replace(pending, name)
        return real_identity(path)

    monkeypatch.setattr(cp, "_identity", identity_after_swap)
    row, = cp.read_run_checkpoints(ws, within=ws)

    assert swapped == [name], "the swap fell between the name check and the stat"
    assert row.sha256 is None and row.model_record is None and row.size_bytes is None
    assert row.problem == f"{run}/last.pt was opened outside the workspace, so it is not read"
    assert outside_sha not in repr(row)


def test_a_read_whose_values_nest_past_what_is_copied_is_listed_and_never_kept(
        tmp_path, monkeypatch):
    """A kept read is copied in and out, one Python frame or more per level of
    nesting, while the safe loader reads a value nested any depth. Before the
    fix the list copied a record and a kept read's logs whole, so a file with
    a value nested a thousand deep raised `RecursionError` out of the list.
    Now such a file is listed from its own read, which nothing else holds, and
    is read again at every listing rather than kept; a shallow one is kept.
    (Its record nests: logs hold single figures, so deep logs are refused,
    `test_a_file_whose_listed_metadata_is_not_of_its_kind_is_listed_unread`.)"""
    from collections import OrderedDict

    import src.utils.checkpoint_paths as cp

    monkeypatch.setattr(cp, "_READS", OrderedDict())
    monkeypatch.setattr(cp, "SETTLED_NS", 0)
    nested = []
    for _ in range(1000):
        nested = [nested]
    run = tmp_path / "runs" / "gatv2" / "20261001-000000_a_111111" / "checkpoints"
    _write_checkpoint(run / "deep.pt", {"epoch": 1, "logs": {"val_pooled_mrr": 0.1},
                                        "model_spec": {}, "data_roles": {"notes": nested}})
    _write_checkpoint(run / "shallow.pt", {"epoch": 2, "logs": {"val_pooled_mrr": 0.2}})
    seen = _counted_reads(monkeypatch)

    for _ in range(2):
        rows = {e.path.name: e for e in cp.read_run_checkpoints(tmp_path, within=tmp_path)}
        assert rows["deep.pt"].problem is None and rows["deep.pt"].epoch == 1
        assert rows["deep.pt"].model_record["data_roles"]["notes"] is not None
    assert sorted(seen) == ["deep.pt", "deep.pt", "shallow.pt"]


#: A file's metadata as a training state writes it, which the listing shows.
_LISTED = {"checkpoint_kind": "training_state", "epoch": 3,
           "logs": {"val_pooled_mrr": 0.2, "val_loss": float("nan"), "steps": 2 ** 200},
           "progress": {"completed_epochs": 4, "end": "finished"},
           "training_input_digests": {"split_manifest": "1" * 64}}


def _with(field, value):
    import copy

    meta = copy.deepcopy(_LISTED)
    *parents, leaf = field.split(".")
    part = meta
    for key in parents:
        part = part[key]
    part[leaf] = value
    return meta


@pytest.mark.parametrize("field, value, named", [
    ("checkpoint_kind", "\udc80", "its checkpoint_kind is not text: character 0 is a lone "
                                   "surrogate"),
    ("checkpoint_kind", "best\ud800", "its checkpoint_kind is not text: character 4"),
    ("checkpoint_kind", 5, "its checkpoint_kind is a int, not text"),
    ("epoch", 1.5, "its epoch is a float, not a whole number"),
    ("epoch", "x", "its epoch is a str, not a whole number"),
    ("epoch", True, "its epoch is a bool, not a whole number"),
    ("progress.completed_epochs", "4", "its progress.completed_epochs is a str, not a whole "
                                       "number"),
    ("progress.end", "\udc80", "its progress.end is not text: character 0"),
    ("progress.end", 1, "its progress.end is a int, not text"),
    ("training_input_digests.split_manifest", "\udc80" * 64,
     "its training_input_digests.split_manifest is not text: character 0"),
    ("logs", {"\udc80": 0.1}, "its logs hold a name that is not text: character 0"),
    ("logs", {1: 0.1}, "its logs hold a name that is a int, not text"),
    ("logs.val_pooled_mrr", "\ud800", "its logs entry 'val_pooled_mrr' is not text"),
    ("logs.val_pooled_mrr", [0.1], "its logs entry 'val_pooled_mrr' is a list, not a number"),
], ids=["kind-low-surrogate", "kind-high-surrogate", "kind-number", "epoch-fraction",
        "epoch-text", "epoch-boolean", "completed-text", "end-surrogate", "end-number",
        "fingerprint-surrogate", "logs-name-surrogate", "logs-name-number",
        "logs-figure-surrogate", "logs-figure-list"])
def test_a_file_whose_listed_metadata_is_not_of_its_kind_is_listed_unread(
        tmp_path, field, value, named):
    """The K8-4r3 verifier, left open 1 and 2: the listing put a file's kind,
    epochs, end, logs and dataset fingerprint in every list's answer as it
    read them, so one string no answer can encode (a lone surrogate, high or
    low) or one value not of its field's kind (an epoch of 1.5, a kind that is
    a number) made a whole list answer HTTP 500. Now that file alone is listed
    with its digest and the reason naming the field (`text_problem` for each
    string), none of its metadata shown, and the file beside it is listed as
    it reads; text with a NUL or 200,000 characters, a non-finite or huge
    figure, are of their kinds and listed."""
    import torch

    from src.utils.checkpoint_paths import read_run_checkpoints
    from src.utils.fingerprint import file_sha256

    run = tmp_path / "runs" / "gatv2" / "20261001-000000_a_111111" / "checkpoints"
    good = _write_checkpoint(run / "good.pt", {**_LISTED, "checkpoint_kind": "a\x00b" * 3,
                                               "progress": {"end": "x" * 200_000}})
    run.mkdir(parents=True, exist_ok=True)
    torch.save(_with(field, value), run / "bad.pt")

    rows = {e.path.name: e for e in read_run_checkpoints(tmp_path, within=tmp_path)}

    bad = rows["bad.pt"]
    assert bad.sha256 == file_sha256(run / "bad.pt") and bad.problem is not None
    assert bad.problem.startswith(named), bad.problem
    assert bad.problem.endswith(", so none of its metadata is shown")
    assert (bad.checkpoint_kind, bad.epoch, bad.logs, bad.validation, bad.end,
            bad.dataset_fingerprint, bad.model_record) == (None,) * 7
    listed = rows["good.pt"]
    assert listed.problem is None and listed.sha256 == good
    assert listed.checkpoint_kind == "a\x00b" * 3 and listed.end == "x" * 200_000
    assert listed.epoch == 3 and listed.logs["steps"] == 2 ** 200
    assert listed.validation == ("val_pooled_mrr", 0.2)


@pytest.mark.parametrize("protocol, shown", [
    ("\udc80proto", "\\udc80proto"),
    ("a\x00b\tc", "a\\x00b\\tc"),
    ("\x00" * 400, "\\x00" * 67 + "\\x0..."),
], ids=["lone-surrogate", "nul-and-tab", "long"])
def test_a_refused_read_is_listed_with_its_message_escaped_and_bounded(tmp_path, protocol, shown):
    """The K8-4r4 verifier, defect 1: torch refuses a legacy (non-zip) file
    whose protocol version is not its own with a message naming the file's
    value, and the listing copied that message into the file's reason, so a
    lone surrogate there made every list answer HTTP 500 and a NUL reached
    the reason raw. Now each character that is not printable is written as
    its escape, and the escaped line is bounded (`refusal_line`); the file
    beside it is listed as it reads."""
    import pickle

    from src.utils.checkpoint_paths import read_run_checkpoints

    run = tmp_path / "runs" / "gatv2" / "20261001-000000_a_111111" / "checkpoints"
    good = _write_checkpoint(run / "good.pt", _LISTED)
    (run / "legacy.pt").write_bytes(pickle.dumps(0x1950A86A20F9469CFC6C, protocol=2)
                                    + pickle.dumps(protocol, protocol=2))

    rows = {e.path.name: e for e in read_run_checkpoints(tmp_path, within=tmp_path)}

    legacy = rows["legacy.pt"]
    assert legacy.problem.isprintable() and legacy.sha256 is None   # shown escaped if not
    assert legacy.problem == f"RuntimeError: Invalid protocol version: {shown}"
    assert rows["good.pt"].problem is None and rows["good.pt"].sha256 == good


def test_a_problem_line_is_the_first_line_escaped_and_bounded():
    """K8-6e.1: `problem_line` is the one rule a reason line follows, split out
    of `refusal_line`, whose behaviour is unchanged: the first non-empty line,
    each character that is not printable escaped, at most 300 characters.
    HEAD: no `problem_line`."""
    from src.utils.checkpoint_paths import problem_line, refusal_line

    assert problem_line("\n  \nfirst line \nsecond") == "first line"
    assert problem_line("a\udc80b\x00") == "a\\udc80b\\x00"
    long = problem_line("k" * 500)
    assert len(long) == 300 and long.endswith("...")
    assert problem_line("") == "" and problem_line("\n\n") == ""
    assert refusal_line(ValueError("bad\nmore")) == "ValueError: bad"
    assert refusal_line(ValueError("")) == "ValueError"


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

    run = tmp_path / "runs" / "gatv2" / "20261001-000000_a_111111"
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

    path = tmp_path / "runs" / "gatv2" / "20261001-000000_a_111111" / "checkpoints" / "last.pt"
    _write_checkpoint(path, {"epoch": 1})
    (entry,) = read_run_checkpoints(tmp_path)
    assert entry.size_bytes == path.stat().st_size > 0


class TestWithinBoundsWhatIsRead:
    """`within`: a file that resolves outside the workspace is listed and never
    opened, so nothing of a file elsewhere is read or shown (M5-1 review)."""

    def _layout(self, tmp_path):
        workspace = tmp_path / "ws"
        run = workspace / "runs" / "gatv2" / "20261001-000000_a_111111" / "checkpoints"
        _write_checkpoint(run / "inside.pt", {"epoch": 1})
        elsewhere = tmp_path / "elsewhere"
        _write_checkpoint(elsewhere / "secret.pt", {
            "epoch": 9, "logs": {"val_pooled_mrr": 0.9},
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
