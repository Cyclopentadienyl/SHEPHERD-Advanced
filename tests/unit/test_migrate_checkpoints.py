"""
`scripts/migrate_checkpoints.py`: the owner's one-time move into the run layout
(contract M3a-2, decision 9).

- The plan moves nothing; `--apply` moves each file by its record: M2's checks
  pass -> an imported run, labelled; M2 refuses -> `archive/`; the check cannot
  be completed -> left in place, listed as unverifiable.
- Bytes are kept (every digest still names the file), nothing is deleted, and
  a destination that exists stops the whole move before anything moves.
- An imported run is found by the run listing and the reload's selection, and
  continuing from it is refused for want of M3 resume state.
- Saved settings naming a moved file are listed.
- Run once as a subprocess, so a broken import fails the test rather than
  hiding in a per-file handler.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tests.fixtures.generated_workspace import write_generated_workspace  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "migrate_checkpoints.py"
IDS = {"train_ids": [0, 1, 2], "val_ids": [3]}


class _NotPlain:
    """Pickled by reference: the safe loader refuses it."""


def _snapshot(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture
def workspace(tmp_path):
    from scripts.train_model import read_training_inputs, training_input_digests

    ws = write_generated_workspace(tmp_path / "ws", training_fields=True, **IDS)[0]
    record = training_input_digests(read_training_inputs(ws))
    hgt = {"config": {"model_config": {"conv_type": "hgt"}}}

    def save(name, contents):
        path = ws / "checkpoints" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": {"w": torch.zeros(2)}, **contents}, path)

    save("hgt/good.pt", {**hgt, "training_input_digests": record})
    save("hgt/crossed.pt", {**hgt, "training_input_digests": {**record, "kg": "0" * 64}})
    save("old.pt", hgt)                                           # no record at all
    save("mystery.pt", {"training_input_digests": record})        # architecture unknown
    torch.save({"thing": _NotPlain()}, ws / "checkpoints" / "hgt" / "unreadable.pt")
    return ws


def _run(ws, *flags):
    import scripts.migrate_checkpoints as tool

    return tool.main([str(ws), *flags])


def test_the_plan_moves_nothing(workspace, capsys):
    before = _snapshot(workspace)
    assert _run(workspace) == 0
    out = capsys.readouterr().out
    assert _snapshot(workspace) == before
    assert "PLAN (nothing is moved without --apply)" in out
    assert "hgt/good.pt  ->  runs/hgt/" in out
    assert "checkpoints/old.pt  ->  archive/checkpoints/old.pt" in out
    assert "unverifiable" in out and "the safe loader cannot open it" in out


def test_apply_moves_by_record_keeps_bytes_and_deletes_nothing(workspace):
    from src.utils.checkpoint_paths import list_runs

    before = _snapshot(workspace)
    assert _run(workspace, "--apply") == 0
    after = _snapshot(workspace)

    runs = list_runs(workspace)
    assert len(runs) == 1 and "_imported-hgt_" in runs[0].name
    imported = runs[0]
    moved = imported / "checkpoints" / "good.pt"
    assert after[moved.relative_to(workspace).as_posix()] == before["checkpoints/hgt/good.pt"]
    label = json.loads((imported / "imported.json").read_text())
    assert label["label"] == ("M2 record checks pass; no M3 resume state; "
                              "servability not judged")
    assert label["files"] == [{"file": "good.pt", "moved_from": "checkpoints/hgt/good.pt",
                               "sha256": before["checkpoints/hgt/good.pt"]}]

    assert after["archive/checkpoints/old.pt"] == before["checkpoints/old.pt"]
    assert after["archive/checkpoints/hgt/crossed.pt"] == before["checkpoints/hgt/crossed.pt"]
    # Unverifiable files stay where they were.
    for stays in ("checkpoints/hgt/unreadable.pt", "checkpoints/mystery.pt"):
        assert after[stays] == before[stays]
    # Every byte is still somewhere; the only new file is the label.
    assert sorted(before.values()) == sorted(
        v for k, v in after.items() if not k.endswith("imported.json"))


def test_an_existing_destination_stops_the_move_before_anything_moves(workspace):
    blocker = workspace / "archive" / "checkpoints" / "old.pt"
    blocker.parent.mkdir(parents=True)
    blocker.write_bytes(b"already here")
    before = _snapshot(workspace)

    with pytest.raises(SystemExit, match="Stopped before moving anything"):
        _run(workspace, "--apply")
    assert _snapshot(workspace) == before


def test_an_imported_run_is_listed_and_selected_but_not_continued(workspace):
    from src.utils.run_plan import RunPlanRefused, plan_continuation
    from src.utils.checkpoint_paths import select_run_checkpoint

    _run(workspace, "--apply")
    selected, reason = select_run_checkpoint(workspace, "hgt")
    assert selected.name == "good.pt" and "imported-hgt" in reason

    record = torch.load(selected, map_location="cpu", weights_only=True)
    with pytest.raises(RunPlanRefused, match="before contract M3a"):
        plan_continuation(record, selected, {}, set(), device="cpu")


def test_saved_settings_that_name_a_moved_file_are_listed(workspace, monkeypatch, tmp_path, capsys):
    import scripts.migrate_checkpoints as tool

    settings = tmp_path / "repo"
    settings.mkdir()
    (settings / ".shepherd_ui_config.json").write_text(json.dumps({
        "checkpoint_path": str(workspace / "checkpoints" / "hgt" / "good.pt")}))
    monkeypatch.setattr(tool, "REPO_ROOT", settings)

    _run(workspace)
    out = capsys.readouterr().out
    assert ".shepherd_ui_config.json: checkpoint_path" in out and "now " in out


def test_the_script_runs_as_an_operator_runs_it(workspace):
    result = subprocess.run([sys.executable, str(SCRIPT), str(workspace)],
                            capture_output=True, text=True, cwd=REPO_ROOT, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "ImportError" not in result.stdout + result.stderr
    assert "runs/hgt/" in result.stdout
