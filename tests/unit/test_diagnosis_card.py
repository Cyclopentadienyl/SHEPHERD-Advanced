"""
Diagnosis: the "Model in use" card and its Details (M5-2, UI-B).
================================================================
`src/webui/components/diagnosis_card.py`. What these pin:

- **one status read with a 5 s limit**, through the dashboard's one reader; a
  read that does not answer in time is "timed out (5 s)", one that fails says
  why; nothing raises;
- **what is served**, as the card names it: no model loaded, a graph without
  a trained model, a model supplied in memory, a checkpoint;
- **the served checkpoint's place** in the serving workspace's runs, from the
  status's served paths, found again in the server's list, **no file opened**;
- **trained on: the checkpoint's own record**, never the served workspace's
  manifest, read from the served file alone (never the other checkpoints of
  its runs) only for a request D8 allows and only on a queued refresh, and
  only when the bytes read are the served ones, compared on that read before
  torch is handed them (another model, or none, is said so); a read-only
  request is told it is not shown, and nothing is opened;
- **the session's state**: a failed read keeps the last known status (grey,
  never current); another served identity than this session's last read sets
  the change notice; what the served checkpoint records is kept while the
  served identity is the same (the file is opened once), read again on ↻; the
  recorded tests and the listed workspaces are read again on every queued
  refresh, never kept under a fresh "checked" time; the timer reads the status
  alone;
- **the card's states** (current, cannot check, stale, no model, no trained
  model) in words, the mode chips only when true, the recorded tests a state;
  a failed read's reason in the card's few words, so its row fits the column;
  Run Diagnosis off only for a read that found no model;
- **Details**: the shared Model Facts with what a served model adds (the
  served graph's and record's digests, the structural fingerprint and GNN
  line, the link to the full records), the trained-on dataset said "not shown"
  to a read-only request and "not read yet" before it is read, never "not
  recorded", in every section; the structural warnings' list inside its box;
- **escaped**: a workspace name with markup.

Module: tests/unit/test_diagnosis_card.py
"""
from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

pytest.importorskip("gradio")

from src.utils.fingerprint import file_sha256  # noqa: E402
from src.webui.components import diagnosis_card as card  # noqa: E402
from tests.fixtures.clock_zone import local_zone, utc_clock  # noqa: E402,F401

RUN = "20261017-091500_run_3fa9c1"
NEWER = "20261018-091200_run_8c2e7a"
NOW = datetime(2026, 10, 20, 10, 14, 5)


# The local stamps below are written in UTC, the zone the card shows.
pytestmark = pytest.mark.usefixtures("utc_clock")


@pytest.fixture(scope="module")
def demo_workspace(tmp_path_factory):
    from tests.fixtures.registered_cohort import write_demo_workspace

    return write_demo_workspace(tmp_path_factory.mktemp("cardbuild") / "demo")


@pytest.fixture
def root(demo_workspace, tmp_path, monkeypatch):
    """A workspaces root: ``demo`` with two gat runs; the older run's
    checkpoint records a dataset fingerprint that is NOT the workspace's
    (an in-place rebuild since), so the two can be told apart."""
    from src.utils.checkpoint_io import write_checkpoint

    root = tmp_path / "root"
    workspace = shutil.copytree(demo_workspace, root / "demo")
    fingerprint = file_sha256(workspace / "split_manifest.json")
    recorded = "9" * 64
    shas = {}
    now = time.time()
    for index, (run, name, epoch, fp) in enumerate([(NEWER, "model-12.pt", 12, fingerprint),
                                                     (RUN, "model-05.pt", 5, recorded)]):
        directory = workspace / "runs" / "gat" / run / "checkpoints"
        directory.mkdir(parents=True)
        shas[name] = write_checkpoint({"checkpoint_kind": "weights", "epoch": epoch,
                                       "logs": {"val_mrr": 0.24},
                                       "training_input_digests": {"split_manifest": fp}},
                                      directory / name)
        os.utime(directory / name, (now - index * 3600, now - index * 3600))
    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(root))
    monkeypatch.delenv("SHEPHERD_REGISTRY", raising=False)
    return {"root": root, "workspace": workspace, "fingerprint": fingerprint,
            "recorded": recorded, "shas": shas}


def _status(root, served="model-05.pt", run=RUN, **extra):
    sha = root["shas"].get(served)
    kg = json.loads((root["workspace"] / "split_manifest.json").read_text())["artifacts"]["kg"]
    status = {
        "initialized": True, "gnn_ready": True,
        "current_data_dir": str(root["workspace"]),
        "current_checkpoint_path": str(root["workspace"] / "runs" / "gat" / run / "checkpoints"
                                       / served),
        "checkpoint_meta": {"epoch": 5, "params": 6569, "val_mrr": 0.241},
        "fingerprint_warnings": [],
        "served_identity": {
            "scoring_mode": "gnn_only",
            "checkpoint": {"state": "verified", "sha256": sha},
            "graph": {"state": "verified", "kg_sha256": kg,
                      "split_manifest_sha256": root["fingerprint"]},
            "shortest_paths": {"state": "absent", "build_id": None, "max_hops": None},
            "dataset_record": {"state": "verified", "origin": "synthetic", "sha256": "e" * 64},
            "settings": {"device": "cpu", "device_requested": "cpu", "eta": 1.0,
                         "max_path_length": 4, "path_length_penalty": 0.5,
                         "aggregation_method": "max", "max_paths_per_candidate": 10,
                         "use_direct_path_optimization": True,
                         "phenotype_normalisation_version": 1},
        },
    }
    status.update(extra)
    return status


@pytest.fixture
def no_checkpoint_opened(monkeypatch):
    """Fails the test if any checkpoint file is opened."""
    from src.api.services import server_lists
    from src.utils import checkpoint_io, checkpoint_paths

    def refuse(*args, **kwargs):
        raise AssertionError("a checkpoint was opened")

    monkeypatch.setattr(server_lists, "list_checkpoints", refuse)
    monkeypatch.setattr(checkpoint_paths, "read_run_checkpoints", refuse)
    monkeypatch.setattr(checkpoint_io, "read_checkpoint", refuse)


def _state(root, *, status=None, operator=True, open_checkpoints=True, state=None, now=NOW):
    return card.advance(state, status if status is not None else _status(root), None, now=now,
                        operator=operator, d8_short=None if operator else "request not local",
                        open_checkpoints=open_checkpoints)


# --------------------------------------------------------------------------- the read
def test_a_read_within_its_limit_is_the_readers_answer():
    assert card.read_status_within(1, reader=lambda: ({"initialized": False}, None)) == (
        {"initialized": False}, None)
    assert card.read_status_within(1, reader=lambda: (None, "the pipeline status could not be "
                                                            "read")) == (
        None, "the pipeline status could not be read")


def test_a_read_that_does_not_answer_in_time_is_said_to_have_timed_out():
    def slow():
        time.sleep(1.0)
        return {"initialized": True}, None

    assert card.read_status_within(0.1, reader=slow) == (None, "timed out (0.1 s)")
    assert card.STATUS_TIMEOUT_SECONDS == 5.0


def test_a_reader_that_raises_is_a_failed_read_never_an_exception():
    def broken():
        raise RuntimeError("boom")

    status, problem = card.read_status_within(1, reader=broken)
    assert status is None and "boom" not in problem and problem


# --------------------------------------------------------------------------- what is served
def test_what_is_served_is_named_by_its_kind(root):
    assert card.served_kind({"initialized": False}) == "no_model"
    assert card.served_kind(None) == "unknown"
    assert card.served_kind(_status(root)) == "model"
    absent = _status(root)
    absent["served_identity"]["checkpoint"] = {"state": "absent", "sha256": None}
    assert card.served_kind(absent) == "no_trained"
    memory = _status(root)
    memory["served_identity"]["checkpoint"] = {"state": "supplied_in_memory", "sha256": None}
    assert card.served_kind(memory) == "in_memory"


def test_the_served_checkpoint_is_found_in_the_runs_without_opening_it(root, no_checkpoint_opened):
    item, why = card.served_item(_status(root), "demo")
    assert why is None
    assert (item["architecture"], item["run"], item["filename"]) == ("gat", RUN, "model-05.pt")
    assert item["sha256"] == root["shas"]["model-05.pt"] and item["epoch"] == 5
    assert item["validation"] == {"metric": "val_mrr", "value": 0.241}


def test_a_checkpoint_outside_the_runs_or_unlisted_is_said_to_be(root, tmp_path):
    outside = _status(root, current_checkpoint_path=str(tmp_path / "elsewhere.pt"))
    assert card.served_item(outside, "demo") == (
        None, "the checkpoint in use is not in the serving workspace's runs")
    missing = _status(root, current_checkpoint_path=str(
        root["workspace"] / "runs" / "gat" / RUN / "checkpoints" / "gone.pt"))
    assert card.served_item(missing, "demo")[1] == (
        "the checkpoint in use is not in the serving workspace's list")
    assert card.served_item(_status(root), None)[1] == (
        "the serving workspace is not a listed workspace")


# --------------------------------------------------------------------------- trained on
def test_trained_on_is_the_checkpoints_own_record_not_the_served_manifest(root):
    state = _state(root)
    trained = state["facts"]["trained"]
    assert trained["state"] == "read" and trained["fingerprint"] == root["recorded"]
    assert trained["fingerprint"] != root["fingerprint"]
    body = card.body_html(state)
    assert "matches no listed workspace" in body
    assert root["recorded"][:12] in body
    assert f'title="{root["fingerprint"]}"' not in body


def test_only_the_served_checkpoint_is_opened(root, monkeypatch):
    """The other checkpoints of the workspace's runs stay closed: a workspace
    with hundreds of them costs the card one read, not one per file."""
    from src.api.services import server_lists
    from src.utils import checkpoint_io, checkpoint_paths

    opened = []
    real = checkpoint_io.read_checkpoint

    def counting(path, *args, **kwargs):
        opened.append(Path(path).name)
        return real(path, *args, **kwargs)

    def refuse(*args, **kwargs):
        raise AssertionError("the whole checkpoint list was read")

    monkeypatch.setattr(checkpoint_io, "read_checkpoint", counting)
    monkeypatch.setattr(server_lists, "list_checkpoints", refuse)
    monkeypatch.setattr(checkpoint_paths, "read_run_checkpoints", refuse)

    state = _state(root)

    assert opened == ["model-05.pt"]
    assert state["facts"]["trained"] == {"state": "read", "fingerprint": root["recorded"],
                                         "problem": None}
    assert state["facts"]["item"]["dataset_fingerprint"] == root["recorded"]


def test_a_read_only_request_opens_no_checkpoint_and_is_told_why(root, no_checkpoint_opened):
    state = _state(root, operator=False)
    assert state["facts"]["trained"]["state"] == "not_shown"
    body = card.body_html(state)
    assert card.TRAINED_NOT_SHOWN in body
    assert "not recorded" not in body
    assert card.change_button(state)["interactive"] is False
    # The reason is visible text in two word groups: a line breaks between
    # them, never inside "request not local" (M5-3: "to M1" alone on a line).
    assert card.change_reason_html(state) == (
        '<p class="shep-dx-why"><span class="shep-wg">🔒 Operator controls only:</span> '
        '<span class="shep-wg">request not local</span></p>')


def test_the_timer_never_opens_a_checkpoint(root, no_checkpoint_opened):
    state = _state(root, open_checkpoints=False)
    assert state["facts"]["trained"]["state"] == "pending"
    assert card.TRAINED_NOT_READ_YET in card.body_html(state)


def test_a_file_rewritten_since_it_was_loaded_is_not_read_as_the_models(root):
    from src.utils.checkpoint_io import write_checkpoint

    path = root["workspace"] / "runs" / "gat" / RUN / "checkpoints" / "model-05.pt"
    status = _status(root)
    write_checkpoint({"checkpoint_kind": "weights", "epoch": 6,
                      "training_input_digests": {"split_manifest": root["fingerprint"]}}, path)
    trained = _state(root, status=status)["facts"]["trained"]
    assert trained["state"] == "not_read" and trained["fingerprint"] is None
    assert "not the one loaded" in trained["problem"]


@pytest.mark.parametrize("bytes_now", ["another checkpoint", "torn", "not a dict", "gone"])
def test_other_bytes_at_the_served_file_are_said_not_the_model_loaded_before_torch_sees_them(
    root, monkeypatch, bytes_now
):
    """Round 3 of the review fixes (INT-R1's class, its verification's A-2): the
    card's one read of the served file is bound to the served digest
    (`read_checkpoint(expected_sha256=)`), so whatever is there when it reads
    -- another valid checkpoint, torn bytes, a pickle that is not a checkpoint,
    or nothing -- is "not read" as another model than the one loaded, or as
    gone, compared before torch is handed the bytes: never torch's error, never
    "could not be read". The control: the same served file unchanged is read."""
    import io

    import torch

    from src.utils import checkpoint_io
    from tests.fixtures.replacement import hook_read_once, replace_file

    status = _status(root)
    control = _state(root, status=status)["facts"]["trained"]
    assert control == {"state": "read", "fingerprint": root["recorded"], "problem": None}

    path = root["workspace"] / "runs" / "gat" / RUN / "checkpoints" / "model-05.pt"
    data = path.read_bytes()
    if bytes_now == "another checkpoint":
        buffer = io.BytesIO()
        torch.save({"checkpoint_kind": "weights", "epoch": 6,
                    "training_input_digests": {"split_manifest": root["fingerprint"]}}, buffer)
        data = buffer.getvalue()
    elif bytes_now == "torn":
        data = data[:len(data) // 2]
    elif bytes_now == "not a dict":
        buffer = io.BytesIO()
        torch.save([1, 2, 3], buffer)
        data = buffer.getvalue()
    fired = []

    def on_entry(read_path):
        # Right before the card's read of the served file, after the server's
        # list found it: the window between the list and the read.
        if fired or read_path.resolve() != path.resolve():
            return
        fired.append(read_path)
        if bytes_now == "gone":
            path.unlink()
        else:
            replace_file(path, data, "rename")

    hook_read_once(monkeypatch, on_entry=on_entry)
    monkeypatch.setattr(checkpoint_io, "BytesIO",
                        lambda *a, **k: pytest.fail("torch was handed the file's bytes"))

    trained = _state(root, status=status)["facts"]["trained"]

    assert fired, "the card never read the served file"
    problem = ("the file the model was loaded from is gone" if bytes_now == "gone" else
               f"the file now holds model {file_sha256(path)[:12]}, not the one loaded")
    assert trained == {"state": "not_read", "fingerprint": None, "problem": problem}


# --------------------------------------------------------------------------- the session's state
def test_a_failed_read_keeps_the_last_known_status_grey_and_never_current(root):
    good = _state(root)
    failed = card.advance(good, None, "timed out (5 s)", now=NOW + timedelta(seconds=5),
                          operator=True, d8_short=None, open_checkpoints=False)
    assert failed["status"] == good["status"] and failed["good_at"] == good["good_at"]
    assert card.card_mode(failed) == "cannot_check"
    body = card.body_html(failed)
    assert "cannot check" in body and "timed out (5 s)" in body
    assert 'title="timed out after 5 s"' in body
    assert "last known 10:14:05 UTC · not current" in body
    assert "verified</span>" not in body          # no ✓ on a last known identity
    assert "Recorded tests: not read" in body
    assert card.time_html(failed) == ""
    assert not card.run_blocked(failed)


def test_a_failed_read_says_its_reason_in_few_words_on_the_card(root):
    """Seen in review: the real reason ("the pipeline status could not be
    read") wrapped the first grey row to three lines in the 300-320 px column
    and scrolled the input body at 1707x870. The card says it short; the full
    reason is in the row's title (and the badge's line and Details)."""
    from src.webui.components.served_status import STATUS_NOT_READ

    failed = card.advance(_state(root), None, STATUS_NOT_READ, now=NOW, operator=True,
                          d8_short=None, open_checkpoints=False)
    body = card.body_html(failed)
    assert ">status not read</span>" in body and f'title="{STATUS_NOT_READ}"' in body
    assert f">{STATUS_NOT_READ}<" not in body
    first_row = body[body.index('class="shep-dx-state"'):body.index("</div>")]
    assert "cannot check" in first_row and "↻ Retry" in first_row
    never = card.advance(None, None, STATUS_NOT_READ, now=NOW, operator=True, d8_short=None,
                         open_checkpoints=False)
    assert ">status not read</span>" in card.body_html(never)
    assert f"not current: {STATUS_NOT_READ}." in card.served_facts_html(failed)
    assert card.card_problem("timed out (5 s)") == "timed out (5 s)"
    assert card.problem_sentence("timed out (5 s)") == "timed out after 5 s"
    assert card.problem_sentence(STATUS_NOT_READ) == STATUS_NOT_READ


def test_a_read_older_than_the_result_is_stale(root):
    state = _state(root)
    produced = (NOW + timedelta(seconds=30)).isoformat()
    assert card.card_mode(state, produced) == "stale"
    body = card.body_html(state, produced)
    assert "read 10:14:05" in body and "older than this result" in body
    assert card.card_mode(state, (NOW - timedelta(seconds=30)).isoformat()) == "current"
    assert "checked 10:14:05" in card.time_html(state)


def test_a_read_later_in_the_same_second_as_the_result_is_current(root):
    """Seen in the browser: the status re-read after a diagnosis came 0.4 s
    after the result's timestamp, in the same second, and showed stale."""
    state = _state(root, now=NOW.replace(microsecond=700_000))
    produced = NOW.replace(microsecond=300_000).isoformat()
    assert card.card_mode(state, produced) == "current"
    assert "checked 10:14:05" in card.time_html(state, produced)


def test_another_served_identity_sets_the_change_notice(root):
    first = _state(root)
    assert first["change"] is None
    later = card.advance(first, _status(root, "model-12.pt", NEWER), None,
                         now=datetime(2026, 10, 20, 10, 31, 12), operator=True, d8_short=None,
                         open_checkpoints=True)
    old, new = root["shas"]["model-05.pt"][:12], root["shas"]["model-12.pt"][:12]
    assert later["change"] == {"at": "10:31 UTC", "was": old, "now": new, "same_model": False}
    assert f"Model changed at 10:31 UTC</b> — was {old}, now {new}" in card.body_html(later)
    assert card.dismiss_change(later)["change"] is None
    # Another part of what is served (the SP table), same model: said so.
    sp = _status(root)
    sp["served_identity"]["shortest_paths"] = {"state": "verified", "build_id": "b1",
                                               "max_hops": 3}
    other = card.advance(first, sp, None, now=NOW, operator=True, d8_short=None,
                         open_checkpoints=False)
    assert other["change"]["same_model"] is True
    assert "same model" in card.body_html(other)


def test_what_the_checkpoint_records_is_kept_while_the_same_thing_is_served(root):
    calls = []

    def facts(status, **kwargs):
        calls.append(kwargs)
        return {**card.served_facts(status, **kwargs)}

    first = card.advance(None, _status(root), None, now=NOW, operator=True, d8_short=None,
                         open_checkpoints=True, facts=facts)
    assert calls[-1]["trained"] is None
    # A queued refresh reads the rest again, with what the checkpoint records
    # carried (the file is not opened again).
    again = card.advance(first, _status(root), None, now=NOW, operator=True, d8_short=None,
                         open_checkpoints=True, facts=facts)
    assert len(calls) == 2 and calls[-1]["trained"] == first["facts"]["trained"]
    assert again["facts"] == first["facts"]
    # The timer's read keeps it all: no listing, no ledger, no file.
    ticked = card.advance(again, _status(root), None, now=NOW, operator=True, d8_short=None,
                          open_checkpoints=False, facts=facts)
    assert len(calls) == 2 and ticked["facts"] == first["facts"]
    # ↻ reads the checkpoint again too.
    card.advance(again, _status(root), None, now=NOW, operator=True, d8_short=None,
                 open_checkpoints=True, reread=True, facts=facts)
    assert len(calls) == 3 and calls[-1]["trained"] is None
    # Another D8 answer reads everything again.
    card.advance(again, _status(root), None, now=NOW, operator=False, d8_short="x",
                 open_checkpoints=True, facts=facts)
    assert len(calls) == 4 and calls[-1]["trained"] is None
    # A pending trained-on dataset is read when a queued refresh may open files.
    pending = card.advance(None, _status(root), None, now=NOW, operator=True, d8_short=None,
                           open_checkpoints=False, facts=facts)
    filled = card.advance(pending, _status(root), None, now=NOW, operator=True, d8_short=None,
                          open_checkpoints=True, facts=facts)
    assert filled["facts"]["trained"]["state"] == "read"


def test_a_queued_refresh_reads_the_recorded_tests_again_without_opening_the_checkpoint(
        root, monkeypatch):
    """Seen in review: a test recorded after the first read (Research › Test)
    stayed "no ledger" under a fresh "checked HH:MM:SS" when Diagnosis was
    shown again, after a diagnosis and on the timer, until ↻."""
    from src.utils import checkpoint_io

    first = _state(root)
    assert "Recorded tests: no ledger" in card.body_html(first)
    ledger = root["workspace"] / "runs" / "gat" / RUN / "checkpoints" / "evaluations.json"
    ledger.write_text("{not json")

    def refuse(*args, **kwargs):
        raise AssertionError("the checkpoint was opened again")

    monkeypatch.setattr(checkpoint_io, "read_checkpoint", refuse)
    later = NOW + timedelta(hours=3)
    # The timer reads the status alone: its answer is the first read's.
    ticked = card.advance(first, _status(root), None, now=later, operator=True, d8_short=None,
                          open_checkpoints=False)
    assert "Recorded tests: no ledger" in card.body_html(ticked)
    # Diagnosis shown again (a queued refresh, not ↻): the ledger is read again.
    shown = card.advance(ticked, _status(root), None, now=later, operator=True, d8_short=None,
                         open_checkpoints=True)
    body = card.body_html(shown)
    assert "Recorded tests: ledger not readable" in body and "no ledger" not in body
    assert "checked 13:14:05" in card.time_html(shown)
    # What the checkpoint records is kept (it was not opened again).
    assert shown["facts"]["trained"] == first["facts"]["trained"]
    assert shown["facts"]["item"]["dataset_fingerprint"] == root["recorded"]
    assert f">{root['recorded'][:12]}</code>" in body


# --------------------------------------------------------------------------- the card's states
def test_the_normal_card_names_the_model_its_fingerprint_and_its_states(root):
    body = card.body_html(_state(root))
    assert "GAT · run 2026-10-17 09:15 · epoch 5" in body
    assert f'>{root["shas"]["model-05.pt"][:12]}</code>' in body and "verified</span>" in body
    assert "synthetic data" in body and "no SP table" in body
    assert "no trained model" not in body and "unverified" not in body
    assert "Recorded tests: no ledger" in body


def test_mode_chips_appear_only_when_true(root):
    files = _status(root)
    files["served_identity"]["dataset_record"]["origin"] = "files"
    files["served_identity"]["shortest_paths"] = {"state": "verified", "build_id": "b",
                                                  "max_hops": 3}
    body = card.body_html(_state(root, status=files))
    assert "synthetic data" not in body and "no SP table" not in body
    no_gnn = _status(root, gnn_ready=False)
    assert "GNN not loaded" in card.body_html(_state(root, status=no_gnn))


def test_no_model_loaded_turns_run_off_and_offers_a_load(root):
    state = _state(root, status={"initialized": False})
    assert card.run_blocked(state)
    assert "No model loaded" in card.body_html(state)
    button = card.change_button(state)
    assert button["value"] == "🔒 Load a model…" and button["variant"] == "primary"
    assert not card.details_visible(state)


def test_a_graph_without_a_trained_model_keeps_run_on(root):
    status = _status(root)
    status["served_identity"]["checkpoint"] = {"state": "absent", "sha256": None}
    status["gnn_ready"] = False
    state = _state(root, status=status)
    assert not card.run_blocked(state)
    body = card.body_html(state)
    assert "No trained model" in body and "path reasoning only" in body
    assert "graph from workspace <b>demo</b>" in body and "no trained model</span>" in body
    assert "GNN not loaded" not in body and "no SP table" not in body


def test_names_are_escaped(root, tmp_path, monkeypatch):
    odd = root["root"] / "<b>odd&name"
    shutil.copytree(root["workspace"], odd, ignore=shutil.ignore_patterns("runs"))
    status = _status(root, "model-12.pt", NEWER)
    body = card.body_html(_state(root, status=status))
    assert "&lt;b&gt;odd&amp;name" in body and "<b>odd" not in body


# --------------------------------------------------------------------------- Details
def test_details_add_what_a_served_model_has_to_model_facts(root):
    html = card.served_facts_html(_state(root))
    identity = html[html.index('data-sec="identity"'):html.index('data-sec="uses"')]
    for text in ("Knowledge graph served (kg.json)", "Dataset record served (kg.provenance.json)",
                 "origin synthetic", "6,569 params", root["recorded"]):
        assert text in identity
    assert "Validation MRR (Trainer, top-20) 0.2410" in html
    # The card reads no model record: no Task or Configuration row claims one
    # is unreadable (K6 decision 4 names the task in the name line instead).
    assert "Record not" not in html and 'class="shep-k">Task<' not in identity
    assert 'class="shep-k">Configuration<' not in identity
    assert 'data-key="models">Full records: Research › Models ›' in html
    assert "Graph structural fingerprint: no warnings · GNN loaded" in html
    warnings = html[html.index('data-sec="warnings"'):]
    assert "SYNTHETIC WORKSPACE" in warnings and "NO SHORTEST-PATH TABLE" in warnings
    assert "TRAINED ON A DATASET NO LISTED WORKSPACE HOLDS" in warnings


def _section(html, key):
    start = html.index(f'data-sec="{key}"')
    return html[start:html.index("</details>", start)]


def test_details_for_a_read_only_request_say_not_shown_never_not_recorded(root,
                                                                          no_checkpoint_opened):
    html = card.served_facts_html(_state(root, operator=False))
    assert card.TRAINED_NOT_SHOWN in html
    assert "DATASET NOT RECORDED" not in html
    assert "Trained on not recorded" not in html
    # Seen in review: the Data section kept "dataset fingerprint not recorded".
    for key in ("identity", "data"):
        assert "not recorded" not in _section(html, key), key
    data = _section(html, "data")
    assert f"Trained on: {card.TRAINED_NOT_SHOWN}" in data
    assert 'dataset fingerprint</span> <span class="shep-muted">not shown</span>' in data


def test_details_before_the_checkpoint_is_read_say_not_read_yet(root, no_checkpoint_opened):
    html = card.served_facts_html(_state(root, open_checkpoints=False))
    for key in ("identity", "data"):
        assert "not recorded" not in _section(html, key), key
    data = _section(html, "data")
    assert f"Trained on: {card.TRAINED_NOT_READ_YET}" in data
    assert 'dataset fingerprint</span> <span class="shep-muted">not read yet</span>' in data
    assert "DATASET NOT RECORDED" not in html


def test_details_say_not_recorded_when_the_checkpoint_records_no_dataset(root):
    """"not recorded" stays where it is true: the checkpoint was read and
    records no dataset fingerprint."""
    from src.utils.checkpoint_io import write_checkpoint

    run = "20261019-080000_run_0d0d0d"
    directory = root["workspace"] / "runs" / "gat" / run / "checkpoints"
    directory.mkdir(parents=True)
    root["shas"]["model-01.pt"] = write_checkpoint({"checkpoint_kind": "weights", "epoch": 1},
                                                   directory / "model-01.pt")
    state = _state(root, status=_status(root, "model-01.pt", run))
    assert state["facts"]["trained"] == {"state": "read", "fingerprint": None, "problem": None}
    html = card.served_facts_html(state)
    assert "not recorded" in _section(html, "identity") and "not recorded" in _section(html, "data")
    assert "DATASET NOT RECORDED" in html


def test_structural_warnings_are_named_for_what_they_compare(root):
    status = _status(root, fingerprint_warnings=["Node count mismatch for 'gene': 3 vs 4"])
    html = card.served_facts_html(_state(root, status=status))
    assert "structural fingerprint</b> (1; node types and counts" in html
    assert "not the graph's identity" in html and "Node count mismatch" in html
    assert "KG version mismatch" not in html
    # The list takes the page's list class (its markers inside the amber box).
    assert '<ul class="shep-dx-list"><li>Node count mismatch' in html


def test_the_structural_line_counts_differences_and_says_when_nothing_was_compared(root):
    """With the comparator's own answers (`verify_fingerprint`): its headline
    is not counted or listed as a difference, and a checkpoint with no
    structural fingerprint (seen in the demo workspace) is "not compared",
    never "differs"."""
    from src.utils.fingerprint import verify_fingerprint

    differs = verify_fingerprint(
        {"data_fingerprint": {"node_types": [], "node_counts": {"gene": 3}, "num_edge_types": 0,
                              "feature_dims": {}}},
        {"num_nodes_dict": {"gene": 4}})
    assert len(differs) == 2                      # the headline and one difference
    line = card._structure_line(_status(root, fingerprint_warnings=differs))
    assert "structural fingerprint</b> (1; node types" in line
    assert line.count("<li>") == 1 and "Node count mismatch for &#x27;gene&#x27;" in line
    legacy = verify_fingerprint({}, {})
    line = card._structure_line(_status(root, fingerprint_warnings=legacy))
    assert "Graph structure not compared</b> (the checkpoint records no structural fingerprint" in line
    assert "differs" not in line and "compared, not" not in line
    many = [differs[0]] + [f"Node count mismatch for 't{i}': checkpoint expects 1, current data has 2"
                           for i in range(5)]
    line = card._structure_line(_status(root, fingerprint_warnings=many))
    assert "(5; " in line and line.count("<li>") == 3 and "and 2 more" in line


def test_details_open_at_the_section_asked_for(root):
    html = card.served_facts_html(_state(root), open_section="evidence")
    assert 'data-sec="evidence" open' in html and 'data-sec="identity" open' not in html


def test_details_of_a_failed_read_say_what_is_last_known(root):
    failed = card.advance(_state(root), None, "timed out (5 s)", now=NOW, operator=True,
                          d8_short=None, open_checkpoints=False)
    assert "Last known at 10:14:05 UTC, not current: timed out (5 s)." in (
        card.served_facts_html(failed))
    assert "could not be read: timed out (5 s)" in card.served_facts_html(
        card.advance(None, None, "timed out (5 s)", now=NOW, operator=True, d8_short=None,
                     open_checkpoints=False))


def test_the_cards_times_are_utc_and_say_so_on_a_server_in_another_zone(root):
    """M5-3 review UI-5: on a server whose zone is not UTC (the homelab),
    Diagnosis showed its local time unlabeled ("checked 23:50:35") beside the
    chip's "since 15:50 UTC" and the Test job's UTC times."""
    with local_zone("Asia/Taipei"):
        state = _state(root, now=datetime(2026, 10, 21, 23, 50, 35))
        assert "checked 15:50:35 UTC" in card.time_html(state)
        assert card.hhmmss("2026-10-21T23:50:30") == "15:50:30 UTC"
        later = card.advance(state, _status(root, "model-12.pt", NEWER), None,
                             now=datetime(2026, 10, 21, 23, 52, 0), operator=True,
                             d8_short=None, open_checkpoints=True)
        assert later["change"]["at"] == "15:52 UTC"
        assert "Model changed at 15:52 UTC</b>" in card.body_html(later)
