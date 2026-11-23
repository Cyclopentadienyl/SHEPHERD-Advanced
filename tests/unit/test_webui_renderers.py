"""
The shared renderers (M5-2, UI-A): one way to show a dataset, a model and its facts.
====================================================================================
`src/webui/components/renderers.py`: pure functions from the server's list
items to escaped HTML. What these pin:

- **the dataset identity**: the name; "built … UTC" or "build time not
  recorded · sources …" (never a file time); "dataset fingerprint" with 12 hex
  shown and the full 64 only in ``title`` and on Copy; amber "synthetic";
  "record verified" / each other record state in its own words; "not
  trainable: <why>"; more
  digits for two listed workspaces that share their first 12;
- **the model row**: the name line from the run id and the epoch; "model
  fingerprint"; "trained on" every listed workspace with the full manifest
  digest the checkpoint records, or "matches no listed workspace"; the
  synthetic chip; the recorded tests as a state, never a result's number; an
  unreadable checkpoint greyed "Metadata not readable: <why>. It cannot be loaded.";
- **Model Facts**: Identity, Uses (the code's fixed statement and the synthetic
  line), Data, Evidence ("selection metric on val, not an independent test";
  recorded tests as states) and Warnings, always open;
- **one vocabulary** (K6 decisions 6 and 8): a listed checkpoint's kind word
  (`record_kind_text`, on `caveats.TASK_NAMES`, the model spec's tasks), a
  recorded metric named by its task and never as a mode (`caveats.metric_text`;
  Mode C's and Mode D's history names unchanged), "GATv2" for a model bundle's
  encoder, the Trainer's validation figures of a base and of a task model by
  name, and Model Facts' Task and Configuration rows;
- **everything escaped**: a workspace or run name may hold markup.

Module: tests/unit/test_webui_renderers.py
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from src.webui.components import renderers as r

FP_A = "2be1f04c9a83" + "a" * 52
FP_B = "b83f2d6a1c95" + "b" * 52
SHA = "15f098277b54" + "c" * 52


def workspace(name="demo", fingerprint=FP_A, origin="synthetic", binding="verified",
              built=None, releases=(), trainable=True, not_trainable=None):
    return {"name": name, "dataset_fingerprint": fingerprint, "trainable": trainable,
            "not_trainable": not_trainable,
            "dataset_record": {"binding": binding, "origin": origin, "sha256": None,
                               "problem": None},
            "build": ({"state": "recorded", "built_at": built} if built
                      else {"state": "not_recorded", "built_at": None}),
            "declared_releases": releases, "train_samples": 20, "val_samples": 5}


def checkpoint(fingerprint=FP_A, sha=SHA, problem=None, epoch=5, run="20261017-091500_run_3fa9c1",
               validation=0.3180):
    return {"workspace": "demo", "architecture": "gatv2", "run": run,
            "filename": "model-05-0.2410.pt", "sha256": sha, "size_bytes": 10,
            "checkpoint_kind": "weights", "epoch": epoch,
            "validation": {"metric": "val_pooled_mrr", "value": validation} if validation else None,
            "dataset_fingerprint": fingerprint, "trained_on_this_dataset": True,
            "problem": problem}


def text_of(html: str) -> str:
    """The words as read: tags dropped, a no-break space (a separator's,
    `renderers.SEP`, or a mark's, `renderers.MARK_SPACE`) read as the space
    it shows."""
    return re.sub(r"<[^>]+>", "", html).replace("\u00a0", " ")


# --------------------------------------------------------------------------- digests
def test_twelve_hex_by_default_and_more_when_two_share_them():
    assert r.short_hex(FP_A) == "2be1f04c9a83" and r.short_hex(None) == "not recorded"
    assert r.distinguishing_digits([FP_A, FP_B]) == 12
    twin = FP_A[:12] + "f" * 52
    assert r.distinguishing_digits([FP_A, twin, FP_A]) == 16
    assert r.distinguishing_digits([FP_A, FP_A[:60] + "ffff"]) == 64


def test_a_fingerprint_shows_12_hex_and_copies_all_64():
    html = r.fingerprint_html("dataset fingerprint", FP_A, help_key="dataset-fingerprint")
    assert '<span class="shep-fpk">dataset fingerprint</span>' in html
    assert f'<code class="shep-fpv" title="{FP_A}">2be1f04c9a83</code>' in html
    assert f'data-copy="{FP_A}"' in html
    assert text_of(html).count(FP_A) == 0          # the full value is not printed
    assert "not recorded" in r.fingerprint_html("dataset fingerprint", None)


# --------------------------------------------------------------------------- the dataset identity
def test_a_recorded_build_time_is_shown_in_utc():
    html = r.dataset_identity_html(workspace(built="2026-10-15T09:41:00Z"))
    assert "built 2026-10-15 09:41 UTC" in html
    assert r.built_text({"state": "recorded", "built_at": "2026-10-15T11:41:00+02:00"}) == (
        "built 2026-10-15 09:41 UTC")


def test_without_a_build_record_the_sources_releases_are_shown():
    html = r.dataset_identity_html(workspace(origin="files", releases=(
        {"role": "hpo", "declared_version": "2026-09-01"},
        {"role": "mondo", "declared_version": None})))
    assert "build time not recorded · sources HPO 2026-09-01 · MONDO release not declared" in html
    assert "synthetic" not in html


def test_a_workspaces_sources_counts_and_build_time_read_one_way_wherever_shown():
    """R-53 (the WebUI ledger): Research › Datasets showed a synthetic
    build's sources as "—" in its table and "none (synthetic)" in its
    details. One renderer now says them (`sources_html`): each declared
    release, "none (synthetic)", or the record's own missing state, never
    "—"; train / val as one value (`pair_text`), the build time without its
    zone, said once in the key (`built_date`)."""
    assert r.sources_html(workspace()) == r.SYNTHETIC_SOURCES == "none (synthetic)"
    declared = workspace(origin="files", releases=({"role": "hpo", "declared_version": "2026-09-01"},))
    assert r.sources_html(declared) == "HPO 2026-09-01"
    assert r.sources_html(workspace(origin="files")) == r.status_mark("neutral", r.NOT_RECORDED)
    assert r.sources_html(workspace(origin=None, binding="unrecorded")) == r.status_mark(
        "neutral", r.NOT_RECORDED)
    assert r.sources_html(workspace(origin=None, binding=None)) == r.status_mark("neutral",
                                                                                 r.UNKNOWN)
    absent = {**workspace(origin=None, binding=None), "manifest": "absent"}
    assert r.sources_html(absent) == r.status_mark("neutral", r.NOT_RECORDED)
    assert r.sources_html(workspace(origin="files", releases=({"role": "<b>x", "declared_version":
                                                               "1"},))) == "&lt;B&gt;X 1"
    assert r.pair_text(200, 50) == "200 / 50" and r.pair_text(1200, None) == "1,200 / not recorded"
    assert r.pair_text(None, None) == "not recorded"
    assert r.built_date({"state": "recorded", "built_at": "2026-10-15T09:41:00Z"}) == (
        "2026-10-15 09:41")
    assert r.built_date({"state": "not_recorded", "built_at": None}) is None


def test_origin_record_and_not_trainable_are_one_renderer_for_the_run_bar_and_datasets():
    """R-53: the run bar's identity line and Research › Datasets' selected
    workspace draw origin and record by one renderer (`dataset_marks_html`),
    and "not trainable" by one (`not_trainable_chip`, its whole reason in its
    title), so a state never reads two ways on one screen."""
    item = workspace()
    marks = r.dataset_marks_html(item)
    assert marks == r.chip("synthetic", "amber") + r.record_chip(item)
    assert marks in r.dataset_identity_html(item)
    assert r.dataset_marks_html(workspace(origin="files")) == r.record_chip(workspace(origin="files"))
    assert r.dataset_marks_html(workspace(origin=None, fingerprint=None, binding=None)) == ""
    refused = workspace(trainable=False, not_trainable="files its <b>manifest</b> binds are missing")
    chip = r.not_trainable_chip(refused)
    assert chip == r.chip("not trainable: files its <b>manifest</b> binds are missing", "grey",
                          title="files its <b>manifest</b> binds are missing")
    assert "<b>" not in chip and chip in r.dataset_identity_html(refused)


def test_marks_synthetic_verified_and_not_trainable():
    html = r.dataset_identity_html(workspace())
    assert 'shep-tag-amber">synthetic<' in html and "record verified" in html
    html = r.dataset_identity_html(workspace(binding="unverifiable", origin="files"))
    assert "record cannot be verified" in html and "record not verified" not in html
    html = r.dataset_identity_html(workspace(trainable=False, not_trainable="files its "
                                             "split_manifest.json binds are missing (kg.json)"))
    assert "not trainable: files its split_manifest.json binds are missing (kg.json)" in html
    html = r.dataset_identity_html(workspace(fingerprint=None, trainable=False,
                                             not_trainable="no manifest at /srv/x"))
    assert "not trainable: no readable split manifest" in text_of(html)


def test_a_name_with_markup_is_escaped():
    html = r.dataset_identity_html(workspace(name='<img src=x onerror="alert(1)">'))
    assert "<img" not in html and "&lt;img src=x" in html


# --------------------------------------------------------------------------- the model row
def test_the_name_line_comes_from_the_run_id_and_the_epoch():
    """A file's title is its run by the one run words (`run_words`, what the
    Run selects say) and how far it trained in the one wording
    (`epoch_words`): the epochs its progress record says it completed; an
    older file without one, the epoch index it records, never a count made
    from it. One " · " (the run words'), the progress after a comma."""
    assert r.model_name_line({**checkpoint(), "completed_epochs": 6}) == (
        "2026-10-17 09:15 · GATv2, 6 epochs completed")
    assert r.model_name_line(checkpoint()) == "2026-10-17 09:15 · GATv2, recorded epoch 5"
    assert r.model_name_line(checkpoint(epoch=None, run="odd")) == (
        "odd · GATv2, epochs not recorded")
    named = checkpoint(run="20261024-173909_ui-live-stop-02_c1d9cf")
    assert r.model_name_line(named) == "ui-live-stop-02 · GATv2 2026-10-24 17:39, recorded epoch 5"
    assert r.run_words("gatv2", named["run"]) in r.model_name_line(named)
    # A title fits an option's length; the details say a long name whole.
    long = checkpoint(run="20261028-100000_smoke-ft-patients_like_me-very-long-name_abcdef")
    assert "…" in r.model_name_line(long)
    assert "smoke-ft-patients_like_me-very-long-name" in r.model_name_line(long, fit=False)
    for line in (r.model_name_line(named), r.model_name_line(long, fit=False)):
        assert line.count(" · ") == 1, line


def test_trained_on_every_listed_workspace_with_the_full_digest():
    listed = [workspace("demo"), workspace("demo_copy"), workspace("other", FP_B)]
    plain = text_of(r.trained_on_html(FP_A, listed))
    assert "trained on demo, demo_copy · dataset fingerprint 2be1f04c9a83" in plain
    assert r.is_synthetic_training(FP_A, listed)
    # a twin that shares 12 hex only is not a match: comparisons are on the full digest
    twin = [workspace("twin", FP_A[:12] + "f" * 52)]
    assert "matches no listed workspace" in text_of(r.trained_on_html(FP_A, twin))
    assert not r.is_synthetic_training(FP_A, twin)
    # The sweep's L7 (SW-LK7): its facts are joined by the shared separator (a no-break space
    # before the dot), so no line of the Diagnosis card starts with "·" ("trained on demo, other
    # ·" then "dataset fingerprint …" read as two facts parted mid-way).
    for html in (r.trained_on_html(FP_A, listed), r.trained_on_html(FP_A, twin)):
        assert html.count(r.SEP) == 1 and " ·" not in re.sub(r"<[^>]+>", "", html), html


def test_the_facts_show_recorded_tests_as_a_state_never_a_number():
    facts = r.facts_from_checkpoint(checkpoint(validation=0.3180), [workspace()],
                                    recorded=r.RecordedTests("records", 3, 1))
    evidence = text_of(r.model_facts_html(facts).split('data-sec="evidence"')[1])
    assert "3 · runs differ for 1" in evidence
    assert "0.318" in evidence and "Recorded results" in evidence  # the Trainer's figure apart
    assert r.recorded_tests_text(r.RecordedTests("none")) == "Recorded results: none in ledger"
    assert r.recorded_tests_text(r.RecordedTests("no_ledger")) == "Recorded results: none kept"
    assert r.recorded_tests_text(r.RecordedTests("not_read")) == (
        "Recorded results: ledger not readable")
    assert r.recorded_tests_text(r.RecordedTests("records", 2, None)) == (
        "Recorded results: 2 · comparison incomplete")


def test_records_that_cannot_be_read_are_named_apart_from_the_results():
    """K8-6e.1 (R3): a record the shared check cannot read is never counted as
    a recorded test; the row says how many with one " · ", before the group
    descriptions, and "none in ledger" only when there is neither. HEAD: no
    count of them, so 0 results and 1 unreadable read "none in ledger"."""
    text = r.recorded_tests_text
    assert text(r.RecordedTests("records", 3, 0, unreadable=1)) == (
        "Recorded results: 3 · 1 not readable")
    assert text(r.RecordedTests("records", 0, 0, unreadable=1)) == (
        "Recorded results: 0 · 1 not readable")
    assert text(r.RecordedTests("records", 0, None, unreadable=2)) == (
        "Recorded results: 0 · 2 not readable")
    assert text(r.RecordedTests("records", 3, 1, unchecked=1, unreadable=2)) == (
        "Recorded results: 3 · 2 not readable, runs differ for 1, comparison incomplete for 1")
    assert text(r.RecordedTests("records", 3, 1, unreadable=2)).count(" · ") == 1
    facts = r.facts_from_checkpoint(checkpoint(), [workspace()],
                                    recorded=r.RecordedTests("records", 0, 0, unreadable=1))
    html = text_of(r.model_facts_html(facts))
    assert "0 · 1 not readable" in html and "none in ledger" not in html


def test_an_unreadable_checkpoint_is_greyed_and_cannot_be_loaded():
    """NEW-models-results-4: an unreadable file's row is greyed and says its
    state and reason; it is chosen like any other row (never Loaded), so its
    details say the reason whole, the loader's reference with it, in visible
    text: Models' detail and the drawer's facts. It was ``aria-disabled``, so
    the details that held the reference were never reached."""
    from src.webui.components import model_list as ml
    from src.webui.components import models_page as mp

    html = _models_tree(checkpoint(sha=None, problem="UnpicklingError: no"))
    assert "shep-row-dis" in html and 'aria-disabled="true"' not in html
    assert "Metadata not readable: UnpicklingError: no. It cannot be loaded." in html
    # A refused read's plain words on the row, the loader's reference left to
    # its details (`renderers.reason_lead`).
    refused = "the file cannot be read as a checkpoint; Ref: UnpicklingError: Weights only"
    assert r.reason_lead(refused) == "the file cannot be read as a checkpoint"
    item = {**checkpoint(sha=None, problem=refused), "size_bytes": 10}
    assert "Ref:" not in _models_tree(item)
    listing = {"items": [item], "workspaces": [], "workspace": "demo"}
    key = ml.item_key(item)
    detail = mp.detail_parts(listing, {}, key, {})
    shown = " ".join(re.sub(r"<[^>]+>", " ", detail["top"]).split())
    assert ("State ▲ Metadata not readable: cannot be loaded Reason the file cannot be read "
            "as a checkpoint Ref: UnpicklingError: Weights only") in shown
    assert detail["seg"] is None and detail["records"] == ""
    # Load stays off, its state said beside it.
    assert ml.load_button(listing, item)["interactive"] is False
    assert "Metadata not readable: it cannot be loaded." in ml.load_reason_html(item, listing)
    assert "UnpicklingError" in ml.facts_html(listing, key)


def test_in_use_marks_the_row():
    from src.webui.components import model_list as ml

    item = checkpoint()
    listing = {"items": [item], "workspaces": [workspace()], "served_sha": item["sha256"]}
    html = ml.list_html(listing, None, ml.PRESETS["research"])
    assert html.count('shep-tag-accent">In use (Diagnosis)<') == 1


# --------------------------------------------------------------------------- Model Facts
def test_model_facts_has_its_five_sections_always_open_with_warnings_first():
    """R-28: Model Facts were an accordion inside the drawer that shows them
    (Identity open, Uses, Data and Evidence collapsed), so most facts were a
    second click away. Every section is open now, Warnings first when there
    are any, its headline in sentence case (R-26)."""
    facts = r.facts_from_checkpoint(checkpoint(), [workspace()],
                                    recorded=r.RecordedTests("none"),
                                    serving_workspace=workspace())
    html = r.model_facts_html(facts)
    assert "<details" not in html and "<summary" not in html
    order = [html.index(f'data-sec="{key}"')
             for key in ("warnings", "identity", "uses", "data", "evidence")]
    assert order == sorted(order), order
    # Each title one level under the h2 of every place that draws the facts
    # (verification of fcc3dea, V-AUD-2: an h4 under the Load drawer's h2).
    for title in ("Warnings", "Identity", "Uses", "Data", "Evidence"):
        assert f"<h3>{title}</h3>" in html
    assert "<h4" not in html
    assert '<section class="shep-facts-sec shep-facts-warn" data-sec="warnings">' in html
    assert "data-shep-focus" not in html
    assert '<section class="shep-facts-sec" data-sec="data" data-shep-focus="true">' in (
        r.model_facts_html(facts, focus="data"))
    # Without Uses (a page that says the device line itself): the rest stays.
    without = r.model_facts_html(facts, uses=False)
    assert 'data-sec="uses"' not in without and r.NOT_A_MEDICAL_DEVICE not in without
    assert 'data-sec="data"' in without
    plain = text_of(html)
    assert ("Not a certified medical device; decision support for a qualified clinician. "
            "Synthetic workspaces are for testing only.") in plain
    # Data and Evidence are summary lists too: a row per fact, one " · " at
    # most (the skill's §4, §5), the words kept.
    assert ("Validationpooled MRR (Trainer) 0.3180 — selection metric on val, not an "
            "independent test") in plain
    assert "Recorded resultsnone in ledger" in plain
    data = html[html.index('data-sec="data"'):html.index('data-sec="evidence"')]
    assert '<dl class="shep-kv">' in data and "Trained ondemo" in text_of(data)
    assert ('<dt class="shep-k">Serving</dt><dd class="shep-v">reads workspace demo: the same '
            'dataset</dd>') in data
    assert "Synthetic workspace: not a clinical result." in plain
    assert "runs/gatv2/20261017-091500_run_3fa9c1/checkpoints/model-05-0.2410.pt" in plain
    assert SHA in plain and FP_A in plain           # Identity: the full digests, with Copy


def test_model_facts_say_a_file_trained_on_the_current_dataset_once():
    """T08-flow-5 (rule 1.2): the Load drawer's and Models' Model Facts of a
    listed file trained on the selected workspace's current dataset said
    "Trained on the selected workspace's current dataset" twice, Identity's
    last row and the Data section's only one (5 of 5 files in the T08 flow).
    Said once now, in Data, where what it was trained on is read; Identity
    keeps the model's own identity. A file trained on another dataset keeps
    its dataset fingerprint whole with its Copy in Identity, and Data says
    which workspaces hold it."""
    from html import unescape

    def words(markup):
        return unescape(text_of(markup))

    facts = r.facts_from_checkpoint(checkpoint(), [workspace()], listed=True)
    html = r.model_facts_html(facts)
    plain = words(html)
    assert plain.count(r.CURRENT_DATASET) == 1, plain
    data = html[html.index('data-sec="data"'):html.index('data-sec="evidence"')]
    assert r.CURRENT_DATASET in words(data)
    identity = html[html.index('data-sec="identity"'):html.index('data-sec="uses"')]
    assert "Trained on" not in words(identity) and FP_A not in plain
    # Diagnosis' drawer draws the same facts without Uses: still once.
    assert words(r.model_facts_html(facts, uses=False)).count(r.CURRENT_DATASET) == 1
    # Another dataset: its fingerprint whole in Identity, its workspaces in Data.
    other = r.facts_from_checkpoint({**checkpoint(fingerprint=FP_B),
                                     "trained_on_this_dataset": False},
                                    [workspace(), workspace("o", FP_B)], listed=True)
    html = r.model_facts_html(other)
    assert r.CURRENT_DATASET not in words(html) and FP_B in words(html)
    assert "Trained ono" in words(html[html.index('data-sec="data"'):])


def test_model_facts_warns_of_a_dataset_no_listed_workspace_holds():
    facts = r.facts_from_checkpoint(checkpoint(fingerprint="9b1f04e6d2c8" + "0" * 52),
                                    [workspace()])
    plain = text_of(r.model_facts_html(facts))
    assert r.TRAINED_ON_UNLISTED in plain and r.TRAINED_ON_UNLISTED == (
        "Trained on a dataset no listed workspace holds.")
    assert "Trained onmatches no listed workspace" in plain
    facts = r.facts_from_checkpoint(checkpoint(fingerprint=FP_B), [workspace("o", FP_B, "files")])
    html = r.model_facts_html(facts)
    # No warning: Warnings says so, last.
    assert "No warnings." in text_of(html)
    assert html.index('data-sec="evidence"') < html.index('data-sec="warnings"')


def test_the_medical_device_line_is_the_one_diagnosis_uses():
    from src.webui.components import diagnosis_panel as dp

    assert dp.NOT_A_MEDICAL_DEVICE is r.NOT_A_MEDICAL_DEVICE
    assert dp._IDENTITY_BANNERS is r.IDENTITY_BANNERS


def test_every_warning_headline_is_in_sentence_case():
    """R-26: the banners were in capitals ("SYNTHETIC WORKSPACE: not a
    clinical result."); their mark and tone make them stand out. Every
    headline the banners and Model Facts' Warnings draw starts with one
    capital and has no word in capitals, its words otherwise kept."""
    import re

    headlines = [head for head, _detail in r.IDENTITY_BANNERS.values()]
    headlines += [r.METADATA_NOT_READABLE, r.TRAINED_ON_UNLISTED, r.DATASET_NOT_RECORDED]
    for head in headlines:
        assert head[0].isupper() and not re.search(r"\b[A-Z]{2,}\b", head), head
    assert r.IDENTITY_BANNERS["synthetic"][0] == "Synthetic workspace: not a clinical result."
    assert r.IDENTITY_BANNERS["not_reported"][0] == "What produced this result was not reported."


@pytest.mark.parametrize("name", ['<script>alert(1)</script>', 'a"b&c'])
def test_model_facts_escape_names(name):
    item = {**checkpoint(), "run": name, "workspace": name}
    html = r.model_facts_html(r.facts_from_checkpoint(item, [workspace(name)]))
    assert "<script>" not in html


# --------------------------------------------------------------------------- shared by several pages
def _option_rules(choices, values):
    """The WebUI skill §8 for one select: unique labels, at most 40
    characters and one separator each, and each value the full key."""
    labels = [label for label, _value in choices]
    assert len(set(labels)) == len(labels), labels
    for label in labels:
        assert len(label) <= r.OPTION_CHARS and label.count(" · ") <= 1, label
    assert [value for _label, value in choices] == list(values)


def test_an_option_is_never_longer_than_the_rule_and_never_lengthens_its_token():
    """`fit_option`, under every select (the skill's §8): at most
    `OPTION_CHARS` characters and one separator whatever the lengths of its
    two parts and its mark; the distinguishing token is cut in its middle and
    never made longer than it was ("a" never reads "a…a"); past the room that
    leaves it at least a few characters, what follows it is cut in its middle
    too, never kept whole while the token alone is cut away."""
    assert r.fit_option("ab", "x" * 37) == "ab · " + "x" * 17 + "…" + "x" * 17
    assert r.fit_option("a", "EXPERIMENTS_FROM_OCTOBER 2026-10-24 17:39").startswith("a · ")
    assert r.fit_option("a" * 30, "x" * 80).startswith("aa…aa · ")
    assert r.middle_cut("abc", 1) == "…" and r.middle_cut("abc", 3) == "abc"
    for what in ("", "a", "ab", "w" * 12, "w" * 70):
        for rest in (None, "r", "r" * 20, "r" * 37, "r" * 70):
            for mark in ("", " (2)", " (13)"):
                option = r.fit_option(what, rest, mark)
                assert len(option) <= r.OPTION_CHARS and option.count(" · ") <= 1, option
                token = option.split(" · ")[0].removesuffix(mark)
                assert len(token) <= len(what) or token == what, option


#: A ZWJ family emoji (MAN, ZWJ, WOMAN, ZWJ, GIRL): five code points, one
#: character on screen.
FAMILY = "\U0001F468‍\U0001F469‍\U0001F467"
#: A flag (two regional indicators) and a letter with a combining accent.
FLAG, ACCENTED = "\U0001F1EF\U0001F1F5", "é"


def _split_sequences(text):
    """What is left of `text` once every whole family, flag and accented
    letter is taken out: any part of one left over means a cut split it."""
    left = text.replace(FAMILY, "").replace(FLAG, "").replace(ACCENTED, "")
    return [ch for ch in left if ch in FAMILY + FLAG + "́"]


def test_a_cut_never_splits_a_character_drawn_from_several_code_points():
    """B72-N4: `middle_cut` sliced by code point, so a name holding the ZWJ
    family emoji cut to 20 characters kept a lone GIRL after the ellipsis,
    and `fit_option` gave "run-👨‍👩…👩‍👧-end · …", a split sequence. Every cut
    (`middle_cut`, so `fit_option` and `fit_options`; `short_reason`) is
    made between two grapheme clusters (`grapheme_ends`): a family, a flag
    and a letter with its accent each stay whole on its side of the "…" or
    are left out whole, and every option stays within `OPTION_CHARS`."""
    name = f"run-{FAMILY}-{'x' * 30}{FAMILY}-end"
    assert r.middle_cut(name, 20) == f"run-{FAMILY}…x{FAMILY}-end"
    for room in range(1, len(name) + 1):
        for head in (0, 4, 6, 8):
            cut = r.middle_cut(name, room, head=head)
            assert len(cut) <= room and not _split_sequences(cut), (room, head, cut)
    # The one cut under it, at every start a caller may try (Diagnosis'
    # cohort options try each).
    for keep in range(len(name)):
        for head in range(keep + 1):
            cut = r.cut_at(name, head, keep)
            assert len(cut) <= keep + 1 and not _split_sequences(cut), (keep, head, cut)
    for what in (f"run-{FAMILY}-end", f"run-{FAMILY * 3}-end", f"{FAMILY}{FLAG}caf{ACCENTED}" * 3):
        for rest in (None, "r" * 20, "r" * 37, f"{FAMILY}-{'r' * 40}"):
            for mark in ("", " (2)"):
                option = r.fit_option(what, rest, mark)
                assert len(option) <= r.OPTION_CHARS and not _split_sequences(option), option
    labels = [label for label, _v in r.fit_options([(f"a{FAMILY * 12}b", None, 1),
                                                     (f"a{FAMILY * 12}c", None, 2)])]
    assert len(set(labels)) == 2 and not any(_split_sequences(label) for label in labels), labels
    assert r.short_reason("a" * 78 + FAMILY + "b", 80) == "a" * 78 + "…"
    assert r.grapheme_ends(f"{ACCENTED}{FLAG}\U0001F1FA\U0001F1F8\r\n한") == [0, 2, 4, 6, 8, 9]


#: The full fine-tune and its frozen-backbone control of the round's
#: checkpoint B flow, whose names differ only in their middle.
ARMS = {"gatv2/20261114-025700_cpb-dis-full-disease_265be0":
        ("gatv2", "20261114-025700_cpb-dis-full-disease_265be0"),
        "gatv2/20261114-025600_cpb-dis-frozen-disease_1a2b3c":
        ("gatv2", "20261114-025600_cpb-dis-frozen-disease_1a2b3c"),
        "gatv2/20261114-025300_run_bb03cc": ("gatv2", "20261114-025300_run_bb03cc")}


def test_runs_whose_names_cut_alike_keep_where_they_differ():
    """The closing check of the ops round (checkpoint B's flow): the full
    fine-tune and its frozen control, "cpb-dis-full-disease" and
    "cpb-dis-frozen-disease", both read "cpb-dis…disease" in every Run
    select and the Models table, told apart only by their start minute. Shown
    together, a name cut alike with another's keeps its start through the
    first two characters where they differ (`fit_apart`, `run_names`, used by
    every Run select through `model_list.run_options` and by the Models
    table); alone it keeps the plain middle cut; the values stay the full
    keys."""
    from src.webui.components import model_list as ml

    names = r.run_names(ARMS)
    full, frozen, base = ARMS
    assert names[full] == "cpb-dis-ful…ase · GATv2 2026-11-14 02:57"
    assert names[frozen] == "cpb-dis-fro…ase · GATv2 2026-11-14 02:56"
    assert names[base] == r.run_words(*ARMS[base])
    assert r.run_names({full: ARMS[full]})[full] == r.run_words(*ARMS[full]) == (
        "cpb-dis…disease · GATv2 2026-11-14 02:57")
    choices = ml.run_options(ARMS)
    _option_rules(choices, list(ARMS))
    assert [label for label, _key in choices] == [names[key] for key in ARMS]
    # The rule is every select's (`fit_options`): two file names alike at
    # both ends keep where they differ; numbering stays the last resort.
    one, two = ("m" * 25 + "A" + "z" * 25, "m" * 25 + "B" + "z" * 25)
    assert r.fit_option(one, None) == r.fit_option(two, None)
    labels = [label for label, _v in r.fit_options([(one, None, 1), (two, None, 2)])]
    assert labels[0] != labels[1] and "A" in labels[0] and "B" in labels[1]
    assert [label for label, _v in r.fit_options([(one, None, 1), (one, None, 2)])] == [
        r.fit_option(one, None, " (1)"), r.fit_option(one, None, " (2)")]
    assert r.middle_cut("abcdefghij", 5, head=3) == "abc…j"
    assert r.middle_cut("abcdefghij", 5, head=9) == "abcd…"


def test_the_models_table_and_the_load_plan_tell_the_two_arms_apart():
    """The Models tree names its runs whole (it has the room an option has
    not; runs that would read alike told apart, `renderers.distinct_names`),
    and the load plan's From and To, which compare two files, name each run
    whole too: the closing check read "cpb-dis…disease · GATv2 2026-11-14
    02:56 → cpb-dis…disease · GATv2 2026-11-14 02:57"."""
    from src.webui.components import model_list as ml

    items = [{**checkpoint(run=run, sha=digit * 64), "key": f"{key}/last.pt", "run_key": key,
              "filename": "last.pt", "record": "model", "learning_stage": "finetune",
              "task": "disease"}
             for (key, (_arch, run)), digit in zip(ARMS.items(), "abc")]
    listing = {"items": items, "workspaces": [workspace("demo")], "workspace": "demo"}
    table = text_of(ml.list_html(listing, None, ml.PRESETS["research"]))
    assert "cpb-dis-full-disease · GATv2 2026-11-14 02:57" in table
    assert "cpb-dis-frozen-disease · GATv2 2026-11-14 02:56" in table
    assert "cpb-dis…disease" not in table
    served = {"initialized": True, "checkpoint_meta": {},
              "served_identity": {"checkpoint": {"sha256": "b" * 64}}}
    frm = ml._from_side(served, {"items": items, "workspaces": listing["workspaces"]})
    to = ml._to_side(items[0], "demo", Path("."), listing["workspaces"])
    assert frm["model"] == "cpb-dis-frozen-disease · GATv2 2026-11-14 02:56, recorded epoch 5"
    assert to["model"] == "cpb-dis-full-disease · GATv2 2026-11-14 02:57, recorded epoch 5"


def test_the_workspace_options_are_one_label_for_every_page():
    """Diagnosis' drawer, Models, Test, Fine-tune and Training's run bar label
    a listed workspace the same way, its name first (the skill's §8): a
    trainable workspace by its name, its identity (fingerprint, synthetic) the
    one beside the select; one that cannot be trained adds why in a few
    words. It read "demo · dataset fingerprint <16 hex> · synthetic", 55
    characters and two separators."""
    twin = FP_A[:12] + "f" * 52
    listed = [workspace("demo"), workspace("twin", fingerprint=twin, origin="files"),
              {**workspace("scratch", fingerprint=None, trainable=False), "manifest": "absent"},
              workspace("broken", fingerprint=None, trainable=False),
              {**workspace("far", fingerprint=None, trainable=False), "manifest": "not_checked",
               "not_trainable": "far resolves outside the workspaces root"}]
    choices = r.workspace_choices(listed)
    assert choices == [("demo", "demo"), ("twin", "twin"),
                       ("scratch · no split manifest", "scratch"),
                       ("broken · no readable split manifest", "broken"),
                       ("far · not trainable", "far")]
    _option_rules(choices, [w["name"] for w in listed])


def test_alike_workspace_names_get_their_distinguishing_digits_and_no_more():
    """A long name is shortened in its middle; only two that would then read
    alike get the synthetic flag where it applies and their fingerprints'
    distinguishing digits (more than 12 where those share 12), and two copies
    of one dataset, alike still, are numbered; each value stays the full
    name."""
    head, tail = "hospital-cohort-build-", "-for-the-research-comparison-round"
    names = [f"{head}{x}{tail}" for x in ("alpha", "bravo", "charlie", "delta")]
    twin = FP_A[:12] + "f" * 52
    listed = [workspace(names[0], fingerprint=FP_A),                       # synthetic
              workspace(names[1], fingerprint=twin, origin="files"),
              workspace(names[2], fingerprint=FP_B, origin="files"),
              workspace(names[3], fingerprint=FP_B, origin="files"),
              workspace("short"), workspace("hospital-cohort-1")]
    choices = r.workspace_choices(listed)
    _option_rules(choices, [w["name"] for w in listed])
    labels = [label for label, _value in choices]
    digits = r.distinguishing_digits([FP_A, twin, FP_B])
    assert digits == 16
    assert all(label.startswith("hospi") and "…" in label for label in labels[:4]), labels
    assert labels[0].endswith(f" · synthetic {FP_A[:digits]}"), labels
    assert labels[1].endswith(f" · {twin[:digits]}"), labels
    assert labels[2].endswith(f"(1) · {FP_B[:digits]}"), labels
    assert labels[3].endswith(f"(2) · {FP_B[:digits]}"), labels
    assert labels[4:] == ["short", "hospital-cohort-1"]


def test_not_trainable_tells_an_absent_manifest_from_an_unreadable_one():
    assert r.not_trainable_short({"dataset_fingerprint": None, "manifest": "absent"}) == (
        "no split manifest")
    assert r.not_trainable_short({"dataset_fingerprint": None, "manifest": "unreadable"}) == (
        "no readable split manifest")
    assert r.not_trainable_short({"dataset_fingerprint": None, "manifest": "not_checked",
                                  "not_trainable": "x resolves outside the workspaces root"}) == (
        "x resolves outside the workspaces root")
    assert r.not_trainable_short({"dataset_fingerprint": FP_A,
                                  "not_trainable": "files are missing (kg.json)"}) == (
        "files are missing (kg.json)")
    html = r.dataset_identity_html({**workspace("s", fingerprint=None, trainable=False),
                                    "manifest": "absent"})
    assert "not trainable: no split manifest" in html


def _models_tree(item, workspaces=()):
    """Models' tree (the shared list's research preset) with `item` its one row."""
    from src.webui.components import model_list as ml

    return ml.list_html({"items": [item], "workspaces": list(workspaces)}, None,
                        ml.PRESETS["research"])


def test_an_entry_never_read_does_not_claim_a_failed_read():
    """NEW-models-results-5: an entry the listing never opened (a run or file
    whose name is not text, a directory that cannot be listed, a file outside
    the workspace: no digest and no size, `ListedCheckpoint`) reads "not
    read", never "metadata not readable", on the shared list's row and the
    research table's; a file whose read was made and failed (its size known)
    or that gave no record (its digest known) still reads "metadata not
    readable"."""
    never = {**checkpoint(sha=None, problem="'bad\\xff' is not text"), "size_bytes": None,
             "run": "bad\\xff", "filename": "RUNS"}
    html = _models_tree(never)
    assert "Not read: " in html and ". It cannot be loaded." in html
    assert "metadata not readable" not in html.lower()
    assert r.unreadable_words(never) == "not read"
    failed = checkpoint(sha=None, problem="UnpicklingError: …")
    loaded = checkpoint(problem="it loads as list, not a checkpoint record")
    for item in (failed, loaded):
        assert r.unreadable_words(item) == "metadata not readable"
        assert "Metadata not readable: " in _models_tree(item)


def test_one_run_reads_one_way_on_every_page():
    """The ops-wiring round's B3 (BACKLOG 72): one run words for every page
    (`run_words`), so a run is never "GATv2 · run …" in one place and its
    name in another; the run-label helpers retired with it."""
    from src.webui.components import model_list as ml

    run = "20261024-173909_ui-live-stop-02_c1d9cf"
    item = {**checkpoint(run=run), "completed_epochs": 6}
    words = r.run_words("gatv2", run)
    assert words == "ui-live-stop-02 · GATv2 2026-10-24 17:39"
    assert r.model_name_line(item).startswith(words)                       # titles, rows
    assert words in text_of(_models_tree(item))                             # Models' tree
    assert ml.run_choices({"k": item})[0][0] == words                       # the Run selects
    assert not hasattr(r, "run_label") and not hasattr(r, "run_label_html")
    assert not hasattr(ml, "run_words") and not hasattr(ml, "_run_parts")


def test_an_identifier_is_kept_whole_with_its_value_as_title_and_is_escaped():
    """M5-3: "research_" / "selection" broke a role code in two. An identifier
    is one unbreakable code element (no break opportunity inside it), which
    ends in a titled ellipsis where it does not fit."""
    assert r.ident_html("research_selection") == (
        '<code class="shep-ident" title="research_selection">research_selection</code>')
    assert "<wbr>" not in r.ident_html("model-00-0.8500.pt")
    assert r.ident_html('a_<b>"') == '<code class="shep-ident" title="a_&lt;b&gt;&quot;">a_&lt;b&gt;&quot;</code>'
    assert r.ident_html(None) == '<code class="shep-ident" title=""></code>'
    from src.webui.style import SHELL_CSS
    rule = SHELL_CSS[SHELL_CSS.index("code.shep-ident {"):].split("}", 1)[0]
    assert "white-space: nowrap" in rule and "text-overflow: ellipsis" in rule


def test_word_groups_break_only_between_groups_and_are_escaped():
    """M5-3: "…: deployment mode not set" / "to M1" left two words alone; each
    group is one inline block, so a line breaks between groups."""
    assert r.word_groups("🔒 Operator controls only:", "a<b>") == (
        '<span class="shep-wg">🔒 Operator controls only:</span> <span class="shep-wg">a&lt;b&gt;</span>')
    assert r.word_groups("one", "") == '<span class="shep-wg">one</span>'
    from src.webui.style import SHELL_CSS
    assert ".shep-wg { display: inline-block;" in SHELL_CSS


def test_the_run_suffix_and_a_validation_figure_by_its_name():
    assert r.run_suffix("20261017-091500_run_3fa9c1") == "run_3fa9c1"
    assert r.run_suffix("my-run") is None
    # The column says "Validation (Trainer)": each cell names its own figure.
    assert r.validation_cell({"metric": "val_pooled_mrr", "value": 0.24101}) == (
        '0.2410<div class="shep-sub">Validation pooled MRR (Trainer)</div>')
    assert "Validation LP accuracy (Trainer)" in r.validation_cell(
        {"metric": "val_lp_accuracy", "value": 0.5})
    # The names are those of the figures selection ranks by, and no other.
    from src.utils.checkpoint_paths import RANKING_SCORE_KEYS
    assert set(r.VALIDATION_LABELS) == set(RANKING_SCORE_KEYS)
    assert "—" not in r.validation_cell(None)
    assert r.validation_cell(None) == r.status_mark("neutral", "Not recorded")
    assert r.validation_cell(None, read=False) == r.status_mark("neutral", "Unknown")


def test_every_clock_time_is_utc_with_its_label():
    """One form for every displayed clock time (M5-3 review UI-5): a naive
    stamp is this server's local time, converted; an aware one converted."""
    from datetime import datetime, timedelta, timezone

    from tests.fixtures.clock_zone import local_zone

    with local_zone("Asia/Taipei"):
        assert r.clock_utc(datetime(2026, 10, 21, 23, 50, 35)) == "15:50:35 UTC"
        assert r.clock_utc(datetime(2026, 10, 21, 23, 50, 35), seconds=False) == "15:50 UTC"
        aware = datetime(2026, 10, 21, 12, 0, tzinfo=timezone(timedelta(hours=-5)))
        assert r.clock_utc(aware) == "17:00:00 UTC"
    assert r.clock_utc(None) is None



def test_model_facts_digest_rows_take_a_line_over_their_full_value():
    """M5-3: Model Facts scrolled at 1920 x 1080 (977 px in 895): a digest's
    long label folded into the label column took four or five lines. Each
    digest's label takes its own line, its full value and an icon-only Copy
    the line under it; the trained-on workspaces follow the label."""
    facts = r.ModelFacts(
        name_line="GATv2 · run 2026-10-21 18:03 · epoch 2", run="gatv2/r",
        checkpoint_file="runs/x.pt", workspace="demo", model_fingerprint="a" * 64, trained_on_fingerprint="b" * 64,
        trained_on_names=("demo",), synthetic=True, sources="", validation=None, recorded=None,
        warnings=())
    html_ = r.model_facts_html(facts)
    assert (f'<dt class="shep-k shep-kv-wide">{r.FACTS_MODEL_FP}</dt>'
            f'<dd class="shep-v shep-kv-wide"><code class="shep-full" title="{"a" * 64}">'
            f'{"a" * 64}</code>') in html_
    assert f'<dt class="shep-k shep-kv-wide">{r.e(r.FACTS_TRAINED_FP)} · demo</dt>' in html_
    assert html_.count("shep-copy shep-copy-icon") == 2
    assert '<dt class="shep-k">Model</dt>' in html_       # short labels keep the grid
    # The sweep's L7 (SW-LK7): the run and the file are tokens, broken only before their own
    # separators ("gatv2/20261002-000000_…" ended a line of the Load drawer at a hyphen).
    run = "gatv2/20261002-000000_disease_bbbbbb"
    tokens = r.model_facts_html(facts._replace(run=run, checkpoint_file=f"runs/{run}/last.pt"))
    assert r.token_html(run) in tokens and r.token_html(f"runs/{run}/last.pt") in tokens
    assert f"<code>{run}" not in tokens and f"<code>runs/{run}" not in tokens
    from src.webui.style import SHELL_CSS

    assert ".shep-kv > .shep-kv-wide { grid-column: 1 / -1;" in SHELL_CSS


# --------------------------------------------------------------------------- one vocabulary
def _record(record="model", stage="finetune", task="disease", differences=(), encoder="gatv2"):
    return {**checkpoint(), "architecture": "gatv2", "record": record,
            "learning_stage": stage if record == "model" else None,
            "task": task if record == "model" else None,
            "encoder": encoder if record == "model" else None,
            "differences": differences if record == "model" else None,
            "reference": (not differences) if record == "model" else None}


def test_a_listed_checkpoint_leads_with_its_kind_word():
    """Decision 6: one word per kind, the task names `caveats.TASK_NAMES`'."""
    from src.evaluation.caveats import TASK_NAMES
    from src.models.bundle.spec import TASKS

    assert tuple(TASK_NAMES) == TASKS
    assert r.record_kind_text(_record(stage="base", task=None)) == "Base"
    assert [r.record_kind_text(_record(task=t)) for t in TASKS] == [
        "Disease", "Gene", "Patients-like-me"]
    # A file without a model spec is a refused record, in the refused word.
    assert r.record_kind_text(_record("refused")) == "Record not readable"
    # A record never read (its file was not read) is Unknown, the skill's §6
    # word for a fact asked for and not read, apart from a refused one.
    assert r.record_kind_text({**checkpoint(problem="torn"), "record": None}) == (
        "Record unknown")


#: A record's weight transfer, by arm (`train_model._transfer_parent_weights`'s
#: shapes): the frozen-backbone control, the full fine-tune, a rules-1 child
#: (no parts, every part trainable) and a freeze of another set of parts.
GRAPH_SIDE_FROZEN = {"trainable": ["patient_encoder", "task_head"], "optimizer": "new",
                     "parts": {"input": "frozen", "encoder": "frozen", "lp_head": "frozen",
                               "patient_encoder": "trainable", "task_head": "trainable"},
                     "frozen_forward": "eval"}
FULL = {"trainable": "all", "optimizer": "new", "frozen_forward": None,
        "parts": dict.fromkeys(GRAPH_SIDE_FROZEN["parts"], "trainable")}
RULES_1 = {"loaded_from_parent": ["encoder", "input", "lp_head"],
           "initialised_new": ["patient_encoder", "task_head"], "trainable": "all",
           "optimizer": "new"}
ENCODER_ONLY = {**GRAPH_SIDE_FROZEN, "parts": {**GRAPH_SIDE_FROZEN["parts"],
                                               "lp_head": "trainable"}}


@pytest.mark.parametrize("transfer, words", [
    (GRAPH_SIDE_FROZEN, "frozen graph backbone"),
    (FULL, None),
    (RULES_1, None),
    (ENCODER_ONLY, None),
    (None, None),
    ("all", None),
])
def test_the_frozen_backbone_control_is_named_by_its_recorded_freeze(transfer, words):
    """Ops-wiring round, decision F1: a task model whose record names exactly
    the graph side frozen (the input module, the encoder and the LP head) is
    "Disease · frozen graph backbone" wherever a model is described
    (`model_kind_text`), read by the continuation's own reading of the
    record (`run_plan.recorded_frozen_parts`); a full fine-tune, a rules-1
    child ("all"), another freeze and no transfer keep the kind alone.
    `record_kind_text` keeps its one word, the grouping key."""
    from src.config.recipe import FROZEN_PARTS
    from src.utils.run_plan import recorded_frozen_parts

    item = {**_record(), "weight_transfer": transfer}
    assert r.freeze_words(item) == words
    assert r.record_kind_text(item) == "Disease"
    assert r.model_kind_text(item) == ("Disease · frozen graph backbone" if words else "Disease")
    frozen = recorded_frozen_parts(transfer)
    assert (frozen == tuple(sorted(FROZEN_PARTS["graph_side"]))) is (words is not None)
    # A result's model (`results._model`) carries the same field.
    assert r.freeze_words({"weight_transfer": transfer}) == words


def test_the_model_kind_text_fits_an_option_for_every_task_and_keeps_other_words():
    """At most one separator and 40 characters for every task (skill §5,
    §8); a base, a refused record and one never read read as their own word,
    whatever a transfer says."""
    from src.models.bundle.spec import TASKS
    OPTION_CHARS = r.OPTION_CHARS
    for task in TASKS:
        text = r.model_kind_text({**_record(task=task), "weight_transfer": GRAPH_SIDE_FROZEN})
        assert text.endswith(" · frozen graph backbone") and text.count(" · ") == 1, text
        assert len(text) <= OPTION_CHARS, text
    assert r.model_kind_text(_record(stage="base", task=None)) == "Base"
    assert r.model_kind_text(_record("refused")) == "Record not readable"
    assert r.model_kind_text({**checkpoint(problem="torn"), "record": None}) == (
        "Record unknown")


def test_a_task_metric_is_named_by_its_task_never_as_a_mode():
    """K6 decision 6: one vocabulary. A task measurement's metric reads "Gene
    pooled MRR", never "Mode gene pooled_mrr"; any other mode's metric, a kept
    Mode C or D record's included, is its recorded mode and its name as
    recorded (R2, K8-6e.2: no C/D words); the task names are the model spec's
    tasks."""
    from src.evaluation.caveats import TASK_NAMES, metric_text, task_list_text
    from src.models.bundle.spec import TASKS

    assert tuple(TASK_NAMES) == TASKS
    # One way to name a set of tasks (the runner's refusals, the Fine-tune page).
    assert [task_list_text(TASK_NAMES[t] for t in TASKS[:n]) for n in (1, 2, 3)] == [
        "Disease", "Disease and Gene", "Disease, Gene and Patients-like-me"]
    assert metric_text("gene", "pooled_mrr") == "Gene pooled MRR"
    assert [metric_text("disease", name) for name in (
        "pooled_top10", "macro_recall@5", "macro_precision@20", "macro_mrr",
        "best_positive_mrr", "conservative_pooled_mrr", "coverage")] == [
        "Disease pooled top-10", "Disease macro recall@5", "Disease macro precision@20",
        "Disease macro MRR", "Disease best-positive MRR", "Disease conservative pooled MRR",
        "Disease coverage"]
    assert metric_text("patients_like_me", "macro_recall@10") == (
        "Patients-like-me macro recall@10")
    assert (metric_text("C", "untruncated_mrr"), metric_text("C", "hits@10"),
            metric_text("D", "post_import_mrr"), metric_text("D", "coverage"),
            metric_text("D", "reached_mean_rank"), metric_text(None, "pooled_mrr")) == (
        "Mode C untruncated_mrr", "Mode C hits@10", "Mode D post_import_mrr",
        "Mode D coverage", "Mode D reached_mean_rank", "pooled MRR (mode not recorded)")
    assert metric_text("zz_never_seen", "mrr") == "Mode zz_never_seen mrr"


def test_a_bundle_run_reads_gatv2_and_a_shepherdgnn_run_its_upper_case():
    assert r.run_words("gatv2", "20261017-091500_run_3fa9c1") == "2026-10-17 09:15 · GATv2"
    assert r.run_words("gat", "20261017-091500_run_3fa9c1") == "2026-10-17 09:15 · GAT"
    assert r.model_name_line(_record()) == "2026-10-17 09:15 · GATv2, recorded epoch 5"
    assert (r.architecture_label("gat"), r.architecture_label("hgt"),
            r.architecture_label(None)) == ("GAT", "HGT", "MODEL")


def test_a_fingerprint_in_a_line_is_the_one_digest_value_after_its_label():
    """The review of B0c: `fingerprint_html` kept its own copy of the digest
    value's rule beside `fingerprint_value`; it is that value after its label,
    the markup unchanged, its missing state "not recorded"."""
    digest = "ab" * 32
    for value, copy in ((digest, True), (digest, False), (None, True)):
        assert r.fingerprint_html("model fingerprint", value, copy=copy) == (
            '<span class="shep-fp"><span class="shep-fpk">model fingerprint</span> '
            + r.fingerprint_value(value, what="model fingerprint", copy=copy) + "</span>")
    shown = r.fingerprint_html("model fingerprint", digest)
    assert f'<code class="shep-fpv" title="{digest}">{digest[:12]}</code> ' in shown
    assert 'data-copy="' + digest in shown and "the full model fingerprint" in shown
    assert '<span class="shep-muted">not recorded</span>' in r.fingerprint_html("x", None)


@pytest.mark.parametrize("metric, label", [
    ("val_pooled_mrr", "Validation pooled MRR (Trainer)"),
    ("val_lp_accuracy", "Validation LP accuracy (Trainer)"),
])
def test_a_base_and_a_task_models_validation_figures_are_named(metric, label):
    """G4: a bundle file's figure read "Validation val_pooled_mrr (Trainer)"."""
    assert r.VALIDATION_LABELS[metric] == label
    assert r.validation_cell({"metric": metric, "value": 0.5}) == (
        f'0.5000<div class="shep-sub">{label}</div>')
    assert r.validation_text((metric, 0.5)).startswith(f"{label} 0.5000 — selection metric")
    from src.webui.components import model_list as ml
    from src.webui.components import training_console as tc

    assert not hasattr(tc, "_METRIC_WORDS"), "one map, the renderers'"
    item = {"filename": "last.pt", "sha256": SHA, "completed_epochs": 1,
            "validation": {"metric": metric, "value": 0.5}}
    # The figure is the chosen file's details on Training, not its option (K6b:
    # the skill's 40 characters and one separator), its Validation row's value
    # the one rule every details list gives it (the review of B0c).
    assert r.validation_figure(item) == (metric, 0.5)
    assert r.validation_value(r.validation_figure(item)) == (
        f"{label.removeprefix('Validation ')} 0.5000 — selection metric on val, "
        "not an independent test")
    assert not hasattr(r, "validation_words")
    assert ml.checkpoint_words(item) == "Latest save · 1 epoch completed"


def test_model_facts_show_the_task_and_the_configuration_escaped():
    """Axis 1 in the facts: the kind word, then the encoder and whether the
    spec is the reference recipe; a recipe changed in N fields counts them."""
    def identity(item):
        return text_of(r.model_facts_html(r.facts_from_checkpoint(item, [workspace()])))

    reference = r.facts_from_checkpoint(_record(task="gene"), [workspace()])
    assert (reference.task, reference.configuration) == ("Gene", "GATv2 · reference recipe")
    assert "TaskGene" in identity(_record(task="gene"))
    assert "ConfigurationGATv2 · reference recipe" in identity(_record(task="gene"))
    changed = r.facts_from_checkpoint(
        _record(stage="base", task=None, differences=("encoder.heads", "input.dim")),
        [workspace()])
    assert (changed.task, changed.configuration) == ("Base", "GATv2 · recipe changed in 2 fields")
    one = r.configuration_text(_record(differences=("input.dim",)))
    assert one == "GATv2 · recipe changed in 1 field"
    refused = r.facts_from_checkpoint(_record("refused"), [workspace()])
    assert (refused.task, refused.configuration) == ("Record not readable", "Not readable")
    assert r.configuration_text({**checkpoint(), "record": None}) == "Not read"
    # An item built without reading a record (the Diagnosis card's served item)
    # shows neither row: the card names its task in the name line (decision 4).
    unread = r.facts_from_checkpoint(checkpoint(), [workspace()])
    assert (unread.task, unread.configuration) == (None, None)
    unread_html = r.model_facts_html(unread)
    assert 'class="shep-k">Task<' not in unread_html
    assert 'class="shep-k">Configuration<' not in unread_html
    odd = r.model_facts_html(r.facts_from_checkpoint(_record(encoder="<b>odd"), [workspace()]))
    assert "<b>odd" not in odd and r.chain_html("<B>ODD", "reference recipe") in odd
    # Each fact of a " · " line one unit (L7 SW-LK7: "recipe changed in 5
    # fields" and the name line's "GATv2 …, 1 epoch complete" broke inside).
    assert '<span class="shep-fact">&lt;B&gt;ODD</span>' in odd
    # A tail (the Details › that opens the last fact) ends that fact's unit,
    # escaped facts and empty ones left out as before.
    assert r.chain_html("a<", "", "b", tail="<i>t</i>") == (
        f'<span class="shep-fact">a&lt;</span>{r.SEP}<span class="shep-fact">b <i>t</i></span>')
    assert r.chain_html("", tail="<i>t</i>") == ""
    bare = r.model_facts_html(r.ModelFacts(
        name_line="n", run=None, checkpoint_file=None, workspace=None, model_fingerprint=None,
        trained_on_fingerprint=None, trained_on_names=(), synthetic=False, sources="",
        validation=None, recorded=None, warnings=(), task="<i>x</i>", configuration=None))
    assert "&lt;i&gt;x&lt;/i&gt;" in bare and "<i>x</i>" not in bare
    assert '<span class="shep-k">Configuration</span>' not in bare  # None: no row, no dash


def test_an_identifier_shown_short_keeps_its_whole_value_as_its_title():
    """A run's name part shown for its whole "arch/run" id (the Fine-tune job
    card) goes through the one identifier markup, the whole id in its title."""
    html = r.ident_html("smoke-ft-disease", title="gatv2/20261028-140013_smoke-ft-disease_0e8106")
    assert html == ('<code class="shep-ident" title="gatv2/20261028-140013_smoke-ft-disease_0e8106">'
                    "smoke-ft-disease</code>")
    assert r.ident_html("<b>x", title="<i>y") == (
        '<code class="shep-ident" title="&lt;i&gt;y">&lt;b&gt;x</code>')


# --------------------------------------------------------------------------- state words and marks
@pytest.mark.parametrize("kind, shape, token", [
    ("ok", "●", "ok"), ("neutral", "○", "neutral"), ("warn", "▲", "warn"),
    ("error", "✕", "error"), ("running", None, "info"), ("stopping", "■", "info"),
])
def test_a_status_mark_is_colour_shape_and_word_with_the_word_always_shown(kind, shape, token):
    """The one status indicator (the WebUI skill §6): the state's
    ``--shep-status-*`` colour, its shape hidden from a screen reader, and its
    word, escaped, always beside it; as a chip it reads the tag colours."""
    from src.webui import style

    html = r.status_mark(kind, "<b>word</b>")
    assert html.startswith(f'<span class="shep-st shep-st-{token}">')
    assert '<span class="shep-st-w">&lt;b&gt;word&lt;/b&gt;</span>' in html and "<b>" not in html
    if shape is None:
        assert '<span class="shep-spinner" aria-hidden="true"></span>' in html
    else:
        assert f'<span class="shep-st-shape" aria-hidden="true">{shape}</span>' in html
    assert f".shep-st-{token} {{ color: var(--shep-status-{token}); }}" in style.SHELL_CSS
    built = r.status_mark(kind, "<b>word</b>", words_html=True)
    assert '<span class="shep-st-w"><b>word</b></span>' in built
    chip = r.status_mark(kind, "word", as_chip=True)
    assert chip.startswith(f'<span class="shep-tag shep-tag-{r._STATUS_TAG[token]} shep-st-chip">')
    # The shape, the space and the word, one run of text (`MARK_SPACE`).
    for drawn in (html, chip):
        assert (f'</span>{r.MARK_SPACE}<span class="shep-st-w">' in drawn) is (shape is not None)
    assert '<span class="shep-st-in">' in chip and '<span class="shep-st-in">' not in html


#: Each status mark drawn in a page with the shell's style sheet, as text and
#: as a chip, in 14 and 15 px text: its ``innerText``, the text a copy of it
#: takes, its shape's box and its word's, and the name a screen reader gives
#: it (the accessibility tree's text of the mark). Reduced motion holds the
#: spinner still (the style sheet's own rule): turning, its box is the turned
#: square's, up to 17 px wide, so the gap measured depended on the frame.
_MARKS_JS = r"""
const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch();
  const out = {};
  for (const [w, h] of [[1850, 960], [1280, 720]]) {
    const page = await (await browser.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 1,
                                                   reducedMotion: 'reduce' })).newPage();
    await page.goto('file://' + process.argv[2]);
    out[`${w}x${h}`] = await page.evaluate(() => [...document.querySelectorAll('[data-mark]')].map((p) => {
      const m = p.firstElementChild;
      const shape = m.querySelector('.shep-st-shape, .shep-spinner').getBoundingClientRect();
      const word = m.querySelector('.shep-st-w').getBoundingClientRect();
      const sel = window.getSelection();
      sel.removeAllRanges();
      const range = document.createRange();
      range.selectNodeContents(m);
      sel.addRange(range);
      return { mark: p.dataset.mark, inner: m.innerText, copied: sel.toString(),
               gap: +(word.left - shape.right).toFixed(2), width: +m.getBoundingClientRect().width.toFixed(2),
               hidden: [...m.querySelectorAll('[aria-hidden="true"]')].map((x) => x.textContent) };
    }));
    // Each shaped mark alone in a box narrower than its shape: the word's first line stands on
    // the shape's line (a line never parts them).
    out[`narrow-${w}x${h}`] = await page.evaluate(() => [...document.querySelectorAll('[data-narrow]')].map((p) => {
      const shape = p.querySelector('.shep-st-shape').getBoundingClientRect();
      const first = p.querySelector('.shep-st-w').getClientRects()[0];
      return { mark: p.dataset.narrow, sameLine: first.top < shape.bottom && first.bottom > shape.top,
               shapeTop: shape.top, wordTop: first.top };
    }));
    if (process.argv[3]) { await page.screenshot({ path: `${process.argv[3]}_${w}x${h}.png`, fullPage: true }); }
  }
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def status_marks_page(path):
    """A page of every status mark, as text and as a chip, in 14 and 15 px
    text, with the shell's style sheet (`_MARKS_JS` reads it)."""
    from src.webui.style import SHELL_CSS

    rows = []
    for kind, word in (("ok", "Saved"), ("neutral", "Not loaded"), ("warn", "Needs attention"),
                       ("error", "Not usable"), ("running", "Running"), ("stopping", "Stopping")):
        for chip in (False, True):
            for size in (14, 15):
                rows.append(f'<p data-mark="{kind}-{int(chip)}-{size}" style="font-size: {size}px">'
                            f'{r.status_mark(kind, word, as_chip=chip)} after</p>')
            if r.STATUS_MARKS[kind][0] is not None:
                rows.append(f'<p data-narrow="{kind}-{int(chip)}" style="width: 1px">'
                            f'{r.status_mark(kind, word, as_chip=chip)}</p>')
    path.write_text(f'<!doctype html><html><head><meta charset="utf-8"><style>{SHELL_CSS}'
                    f'</style></head><body style="padding: 16px">{"".join(rows)}</body></html>',
                    encoding="utf-8")
    return path


def test_a_status_marks_text_reads_shape_space_word_and_its_look_is_kept(tmp_path):
    """The verification of fcc3dea: a mark's shape was directly followed by
    its word, so its text read "○Saved", "✕Not usable" (copied or as
    ``innerText``), and a chip's read the shape and the word as two lines.
    Now every mark, as text and as a chip, reads shape, space, word, copied
    and as ``innerText`` (a chip copied alone ends its line, as any box of
    its own does), at 1850 x 960 and the display floor's 1280 x 720; the shape stays
    hidden from a screen reader, so it reads the word; and the look is kept:
    4 px from the shape to the word (6 px after the spinner), as on fcc3dea,
    the space having no width. The sweep's L7 (SW-LK7): the space was a
    breakable one, so "model fingerprint 60adc7154078 ●" ended a line of the
    Diagnosis card and "verified" began the next; it is a no-break space, so
    in a box narrower than the shape the word's first line still stands on
    the shape's line."""
    from tests.fixtures.browser import node_with_playwright, run_node

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    seen = run_node(_MARKS_JS, tmp_path, str(status_marks_page(tmp_path / "marks.html")))
    shapes = {kind: shape or "" for kind, (shape, _token) in r.STATUS_MARKS.items()}
    words = {"ok": "Saved", "neutral": "Not loaded", "warn": "Needs attention",
             "error": "Not usable", "running": "Running", "stopping": "Stopping"}
    narrow = {size: seen.pop(size) for size in list(seen) if size.startswith("narrow-")}
    for size, marks in narrow.items():
        assert len(marks) == 10 and all(m["sameLine"] for m in marks), (size, marks)
    for size, marks in seen.items():
        assert len(marks) == 24, size
        for mark in marks:
            kind, chip, _px = mark["mark"].split("-")
            # Shape, the no-break space (`MARK_SPACE`), word; the spinner has no space.
            said = f"{shapes[kind]}\u00a0{words[kind]}" if shapes[kind] else words[kind]
            # A chip is a box of its own: a copy of it alone ends its line.
            copied = mark["copied"][:-1] if chip == "1" and mark["copied"].endswith("\n") else (
                mark["copied"])
            assert mark["inner"] == said and copied == said, (size, mark)
            assert mark["hidden"] == [shapes[kind]], (size, mark)
            gap = 6 if kind == "running" and chip == "0" else 4
            assert mark["gap"] == gap, (size, mark)


def test_each_missing_state_has_its_own_word_and_a_dash_never_stands_for_not_recorded():
    """The skill's §6: Off, Not loaded, Not applicable, Unknown, Cannot be
    verified and Not recorded say different things; none is "—", and the
    shared renderers that read a record's value say "not recorded" (or why it
    is unknown), never the dash, which means only "no value here"."""
    words = (r.OFF, r.NOT_LOADED, r.NOT_APPLICABLE, r.UNKNOWN, r.CANNOT_BE_VERIFIED,
             r.NOT_RECORDED)
    assert words == ("Off", "Not loaded", "Not applicable", "Unknown", "Cannot be verified",
                     "Not recorded")
    assert len(set(w.lower() for w in words)) == len(words)
    assert r.NO_VALUE == "—" and r.NO_VALUE not in words
    absent = checkpoint(epoch=None, validation=None, sha=SHA)
    shown = [r.short_hex(None), r.count_text(None), r.recorded_count_text(None),
             r.model_name_line(absent), r.validation_cell(None),
             r.validation_cell(None, read=False), _models_tree(absent, [workspace()])]
    for text in shown:
        assert "—" not in text, text
    assert r.count_text(None) == "not recorded" and r.count_text(1200) == "1,200"
    # A record's count and a manifest's count: one word for the same state.
    assert r.recorded_count_text(None) == r.count_text(None)
    assert r.recorded_count_text(1200) == "1200"
    assert "epochs not recorded" in _models_tree(absent, [workspace()])
    # The record's states, each its own words: absent apart from unverifiable.
    states = {binding: r.record_words(workspace(binding=binding))
              for binding in ("verified", "unrecorded", "unverifiable", "mismatch", None)}
    assert states == {"verified": ("ok", "record verified"),
                      "unrecorded": ("neutral", "record not recorded"),
                      "unverifiable": ("neutral", "record cannot be verified"),
                      "mismatch": ("error", "record does not match"),
                      None: ("neutral", "record unknown")}
    assert r.record_chip(workspace(binding="unrecorded")) == r.status_mark(
        "neutral", "record not recorded", as_chip=True)
    # Verified is the shared OK mark too (the WebUI skill's §6; review of the
    # ops-wiring round's B3: it was the old "✓" svg chip beside "● Bound to
    # this graph" on the Test page's Data panel and Models' workspace line).
    verified = r.record_chip(workspace(binding="verified"))
    assert verified == r.status_mark("ok", "record verified", as_chip=True)
    # The page's own check glyph retired once no page drew it (the ops round's
    # B5): every OK state is this one mark.
    assert not hasattr(r, "CHECK") and "CHECK" not in r.__all__


# --------------------------------------------------------------------------- shared details
# The one summary list and the one value-words rule (ops-wiring round B0c;
# the WebUI skill's §2, §4, §5, §6).
HOSTILE = "<img src=x onerror=alert(1)>"
WORLDLY = "中文標籤 🧬✨"


def _shown_text(html_: str) -> str:
    import html as html_lib

    return " ".join(html_lib.unescape(re.sub(r"<[^>]+>", " ", html_)).split())


def test_the_summary_list_is_one_dl_its_keys_escaped_its_values_as_built():
    html_ = r.summary_list([("Run", "<code>gatv2/r</code>"), (HOSTILE, r.e(HOSTILE)),
                            (r.FACTS_MODEL_FP, r.fingerprint_value("a" * 64, full=True))],
                           wide=(r.FACTS_MODEL_FP,), titles={"Run": "run_dir"},
                           helps={"Run": ("model-fingerprint", "Run")}, classes="shep-x")
    assert html_.startswith('<dl class="shep-kv shep-x">') and html_.endswith("</dl>")
    assert '<dt class="shep-k" title="run_dir">Run<button' in html_
    assert '<dd class="shep-v"><code>gatv2/r</code></dd>' in html_
    assert HOSTILE not in html_ and html_.count("&lt;img src=x onerror=alert(1)&gt;") == 2
    assert f'<dt class="shep-k shep-kv-wide">{r.FACTS_MODEL_FP}</dt>' in html_
    # A digest: its 12 hex (all 64 when full) and an icon Copy; the missing
    # state's own word, never a dash.
    short = r.fingerprint_value("b" * 64, what="model fingerprint")
    assert f'<code class="shep-fpv" title="{"b" * 64}">{"b" * 12}</code>' in short
    assert 'aria-label="Copy the full model fingerprint"' in short or "the full model" in short
    assert r.fingerprint_value(None) == '<span class="shep-muted">not recorded</span>'
    assert r.fingerprint_value("x", missing="not read") == '<span class="shep-muted">not read</span>'
    # 12 hex digits are a known prefix only where the caller says the value
    # is one (a report's run id names its model so); from any other caller
    # they are no digest, its missing state (review of B3).
    assert r.fingerprint_value("c" * 12) == '<span class="shep-muted">not recorded</span>'
    assert r.fingerprint_value("c" * 12, prefix=True) == (
        f'<code class="shep-fpv" title="{r.e(r.PREFIX_TITLE)}">{"c" * 12}</code>')
    assert r.fingerprint_value("c" * 64, prefix=True) == r.fingerprint_value("c" * 64)


def test_the_value_words_say_each_setting_by_its_meaning_never_by_repr():
    """One rule for every page (B0c): on/off for a switch, a None by its
    meaning per setting (the field spec's), a list as its items with -1 as
    every neighbour, a number as typed, a closed value by its field's words;
    never Python's repr, never a bare None."""
    words = r.setting_words
    assert (words("use_amp", True), words("use_amp", False)) == ("on", "off")
    assert words("early_stopping_patience", None) == "off"
    assert words("seed", None) == r.NOT_SET == "not set"
    # A task's loss weight takes no null (OPS-R2): a None is no reference.
    assert words("task_loss_weight", None, stage="finetune", task="disease") == r.NOT_SET
    for name in ("task_loss_weight", "candidate_sample_size", "task"):
        assert words(name, None, stage="base") == r.NOT_TAKEN_BY_A_BASE, name
    assert words("candidate_sample_size", None, stage="finetune", task="gene") == (
        "not taken by the Gene task")
    assert words("num_neighbors", [15, 10, 5]) == "15, 10, 5"
    assert words("num_neighbors", [15, -1]) == "15, all neighbours"
    assert words("exclude_relations", []) == "none"
    assert (words("learning_rate", 0.0001), words("learning_rate", 1e-05),
            words("max_grad_norm", 1.0), words("num_epochs", 30)) == ("0.0001", "1e-05",
                                                                     "1.0", "30")
    assert words("frozen_parts", "graph_side") == "Frozen"
    assert words("frozen_parts", "none") == "Trained"
    assert words("scheduler_type", "cosine") == "cosine"
    assert words("gene", {"eta": 0.5, "augmentation": True}) == "eta: 0.5, augmentation: on"
    assert words("run_name", WORLDLY) == WORLDLY                # text as it is
    assert words("run_name", HOSTILE) == HOSTILE                # plain text: the caller escapes
    for value in (None, True, [None, "a"], {"k": None}, ("x",), 0.5):
        text = words("x", value)
        assert "'" not in text and "None" not in text and "True" not in text, (value, text)


def test_a_nones_words_are_what_the_field_spec_says_it_means():
    """The words of a None (`SETTING_UNSET_WORDS`) are the field spec's, never
    a page's own: a setting has words of its own only where a null is one of
    its values (`FieldSpec.null_means`: early stopping's patience, "off: no
    early stopping", reads "off"), and so does the one model spec field whose
    None is a state the form says "off" under the field (the relation edge
    dimension: edges carry no relation), so a record view and the form say
    one word. A setting that takes no null has none (an unset seed is "not
    set"): a task's loss weights and sample sizes no longer read "the task's
    reference" for a None, which the child would fail on (OPS-R2)."""
    from src.config.training_fields import FIELDS, by_name
    from src.webui.components import training_console as tc

    names = {f.name for f in FIELDS}
    off = {key for key, words in tc.FIELD_FORMATS.items() if '"off"' in words}
    assert off == {"early_stopping_patience", "encoder.relation_edge_dim"}
    assert r.SETTING_UNSET_WORDS == {key: r.OFF.lower() for key in off}
    for name in set(r.SETTING_UNSET_WORDS) & names:
        assert by_name(name).null_means.startswith(r.SETTING_UNSET_WORDS[name]), name
    assert r.setting_words("encoder.relation_edge_dim", None) == "off"
    for name in ("task_loss_weight", "lp_loss_weight", "candidate_sample_size",
                 "sparse_sample_size"):
        assert not by_name(name).null_means
        assert r.setting_words(name, None, stage="finetune", task="disease") == r.NOT_SET
    assert r.setting_words("candidate_upsample_size", None, stage="finetune",
                           task="gene") == r.NOT_SET


def _spec(stage="base", task=None, **changes):
    from src.models.bundle.spec import reference_spec

    data = reference_spec(stage, task).to_dict()
    for path, value in changes.items():
        section, field = path.split("__")
        data[section][field] = value
    return data


def test_a_model_spec_is_said_by_its_differences_its_whole_one_click_away():
    """Continue's spec row (BACKLOG 72: the whole spec with its sources, about
    3,000 px): "Reference GATv2 spec", or the fields that differ with the
    reference's values, the first few named and the rest counted; every field
    behind one "Details ›", one level, the differing ones marked; a part its
    stage has not is "not taken by a base"; a spec this build cannot read
    says Unknown and why, escaped. Each field is named in words, its
    section's then its own ("Encoder › Layers", as Compare names a nested
    entry), its dotted path the title, and a recorded identifier is kept
    whole (the review of B0c: the record's paths, broken mid-word)."""
    words, details = r.model_spec_html(_spec())
    assert words == "Reference GATv2 spec"
    assert details.count("<details") == 1 and "<summary>Details ›</summary>" in details
    assert ('<dt class="shep-k" title="encoder.layers">Encoder › Layers</dt>'
            '<dd class="shep-v">3</dd>') in details
    assert ('<dt class="shep-k" title="patient_encoder">Patient encoder</dt>'
            '<dd class="shep-v">not taken by a base</dd>') in details
    assert ('<dt class="shep-k" title="lp_head.relation_init">LP head › Relation '
            'initialisation</dt><dd class="shep-v"><code class="shep-ident" '
            'title="xavier_uniform_leaky_relu">xavier_uniform_leaky_relu</code></dd>') in details
    keys = re.findall(r'<dt class="shep-k"[^>]*>([^<]*)</dt>', details)
    assert not [key for key in keys if "_" in key or "." in key], keys
    # The reference recipe's provenance notes are not this model's values.
    assert "sources" not in details and "mims-harvard" not in details
    assert "None" not in _shown_text(details) and "'" not in _shown_text(details)
    words, details = r.model_spec_html(_spec(encoder__layers=2, encoder__hidden=64,
                                             encoder__heads=2, encoder__output=32,
                                             input__dim=8))
    assert words.startswith("Differs from the reference GATv2 spec: Input › Dimension 8 "
                            "(2048), Encoder › ")
    assert words.endswith(" and 1 more") and words.count("(") == 4
    assert '<dd class="shep-v">2 (reference 3)</dd>' in details
    disease = _spec("finetune", "disease")
    words, details = r.model_spec_html(disease)
    assert words == "Reference GATv2 spec"
    assert ('<dt class="shep-k" title="task_head.gene">Task head › Gene</dt>'
            '<dd class="shep-v">not taken by the Disease task</dd>') in details
    unread, none = r.model_spec_html(_spec(encoder__kind=HOSTILE))
    assert none == "" and "Unknown: not read as a model spec" in unread
    assert HOSTILE not in unread and "&lt;img" in unread
    assert r.model_spec_html(None) == (r.status_mark(
        "neutral", "Unknown: the record holds no model spec"), "")
    # A listed spec (`record.listed_record`'s, without the recipe's notes)
    # is read by the one reader of a listed spec (`record.listed_spec`) and
    # said alike (review of B2: it was "Unknown ... lacks ['sources']").
    for data in (_spec(), _spec(encoder__layers=2, input__dim=8), disease):
        listed = {key: value for key, value in data.items() if key != "sources"}
        assert r.model_spec_html(listed) == r.model_spec_html(data)
        assert "Unknown" not in r.model_spec_html(listed)[0]


#: A Python repr in shown text: a quoted value ('none', ['a']) or a bare
#: None, True or False (an apostrophe inside a word, "task's", is English).
_REPR = re.compile(r"(?<!\w)'[^'\n]*'(?!\w)|\bNone\b|\bTrue\b|\bFalse\b")


def _bordered_classes(css: str) -> set:
    """The classes a rule draws a border around (``border:`` not 0/none),
    by the last compound of each selector."""
    bordered = set()
    for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        if not re.search(r"(?<![-\w])border\s*:(?!\s*(?:0|none)\b)", body):
            continue
        for selector in selectors.split(","):
            parts = selector.split()
            if parts:
                bordered.update(re.findall(r"\.([\w-]+)", parts[-1].split(":")[0]))
    return bordered


def _nested_boxes(html_: str, bordered: set) -> list:
    """Each container (div, dl, details, section, table) with a bordered
    class inside another such container: a bordered box inside a box."""
    from html.parser import HTMLParser

    containers = {"div", "dl", "details", "section", "table"}
    found: list = []
    stack: list = []

    class Walk(HTMLParser):
        def handle_starttag(self, tag, attrs):
            if tag not in containers:
                return
            boxed = set((dict(attrs).get("class") or "").split()) & bordered
            if boxed and any(stack):
                found.append(sorted(boxed))
            stack.append(bool(boxed))

        def handle_endtag(self, tag):
            if tag in containers and stack:
                stack.pop()

    Walk().feed(html_)
    return found


def test_a_summary_lists_values_keep_their_regions_size():
    """The review of B0c: the check's lists and the spec's Details set their
    values at 14 px, under the 15 px the summary list's values take (the
    skill's §2: keys 14 px, values 15 px). The spec's Details sets no size of
    its own; the check's lists take the check's (15 px; 14 px where the
    dense drawer sets the whole check so, to fit)."""
    from src.webui.components import training_console as tc
    from src.webui.style import SHELL_CSS

    def body(css, selector):
        found = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", css)
        assert found, selector
        return found.group(1)

    assert "font-size: 15px" in body(SHELL_CSS, ".shep-kv")
    assert "font-size" not in body(SHELL_CSS, ".shep-more > .shep-kv")
    assert "font-size: inherit" in body(tc.PAGE_CSS, ".shep-check .shep-kv")
    assert "font-size: 15px" in body(tc.PAGE_CSS, ".shep-cmp, " + tc._PAGE + " .shep-check")


def test_every_details_lists_validation_row_reads_the_one_way():
    """The review of B0c: Continue's chosen checkpoint and Fine-tune's base
    cut the figure's prefix themselves, so one fact read three ways ("not
    recorded in this file", "not recorded", "not recorded in this
    checkpoint"; the Trainer's note on some pages only). Every details list's
    Validation row is `validation_value` of the item's `validation_figure`."""
    from src.webui.components import finetune_page as ft
    from src.webui.components import test_page as tp
    from src.webui.components import training_console as tc

    def rows(item):
        page = {"workspaces": [], "roots": [], "access": {}}
        models = {"items": [item], "workspace": "demo"}
        shown = (tc.parent_details_html(item), ft.base_summary_html(item),
                 tp.model_html(page, models, tp.item_key(item)),
                 r.model_facts_html(r.facts_from_checkpoint(item, [])))
        return {re.search(r'Validation(?:<[^>]*>)*</dt><dd class="shep-v[^"]*">([^<]*)',
                          html_).group(1) for html_ in shown}

    figure = {**checkpoint(), "checkpoint_kind": "training_state"}
    assert rows(figure) == {"pooled MRR (Trainer) 0.3180 — selection metric on val, "
                            "not an independent test"}
    assert rows({**figure, "validation": None}) == {"not recorded in this checkpoint"}
    assert rows({**figure, "validation": None, "record": "refused"}) == {
        "not read (record not readable)"}


def test_every_details_lists_selection_row_reads_the_one_way():
    """Ops-wiring round B1c (B0c's leftover): Continue's chosen checkpoint and
    Fine-tune's base each cut "Selection " off `selection_words` themselves.
    Every details list's Selection row is `selection_value`, beside
    `validation_value`: the recorded monitor and direction, or "not
    recorded", never "Selection Selection …"."""
    from src.webui.components import finetune_page as ft
    from src.webui.components import training_console as tc

    def rows(item):
        shown = (tc.parent_details_html(item), ft.base_summary_html(item))
        return {re.search(r'Selection</dt><dd class="shep-v[^"]*">([^<]*)', html_).group(1)
                for html_ in shown}

    figure = {**checkpoint(), "checkpoint_kind": "training_state",
              "selection": {"monitor": "val_pooled_mrr", "mode": "max"}}
    assert rows(figure) == {r.selection_value(figure)} == {
        "by validation pooled MRR (Trainer), higher is better"}
    assert rows({**figure, "selection": None}) == {r.selection_value({})} == {"not recorded"}
    for module in (tc, ft):
        assert 'removeprefix("Selection ")' not in inspect.getsource(module)


def test_a_branch_change_reads_in_the_one_value_words_rule():
    """`change_value`: a setting a branch changes, its parent's value and its
    own, each by `setting_words` (never repr), escaped; Models' "Settings
    changed" and the Continue drawer's check say it so."""
    from src.webui.components import models_page as mp

    assert r.change_value("learning_rate", {"parent": 0.0001, "new": 0.0005}) == (
        "<code>0.0001</code> → <code>0.0005</code>")
    assert r.change_value("early_stopping_patience", {"parent": None, "new": 3}) == (
        "<code>off</code> → <code>3</code>")
    assert r.change_value("scheduler_type", {"parent": "<b>x", "new": "none"}) == (
        "<code>&lt;b&gt;x</code> → <code>none</code>")
    assert r.change_value("seed", None) == "<code>not set</code> → <code>not set</code>"
    # One changed setting a line (the owner's decision of 2026-11-23: never "; ").
    assert mp.changes_html({"learning_rate": {"parent": 0.0001, "new": 0.0005}}) == (
        "<div>Learning rate " + r.change_value("learning_rate", {"parent": 0.0001, "new": 0.0005})
        + "</div>")


def test_every_details_list_escapes_keeps_words_says_no_repr_and_draws_one_box():
    """The details B0c routes through the one summary list, each rendered
    from a value that is hostile markup and one in Chinese with emoji: the
    markup is literal text, the Chinese and emoji unchanged, no Python repr
    quote or bare None is shown, and no bordered box sits inside another
    (the skill's §2, §5, §6)."""
    from src.config.recipe import BASE
    from src.models.bundle.spec import reference_spec
    from src.webui.components import finetune_page as ft
    from src.webui.components import models_page as mp
    from src.webui.components import test_page as tp
    from src.webui.components import training_console as tc
    from src.webui.style import SHELL_CSS

    bordered = _bordered_classes(SHELL_CSS + tc.PAGE_CSS + ft.PAGE_CSS + tp.PAGE_CSS
                                 + mp.PAGE_CSS)
    assert {"shep-panel", "shep-card"} <= bordered
    assert not {"shep-kv", "shep-more", "shep-inherit", "shep-ckpt", "shep-ft-base"} & bordered
    # The check result and the dataset comparison sit in the Continue drawer,
    # a bordered panel (`.shep-panel`), so neither draws a border of its own
    # (the review of B0c: both were bordered boxes inside it).
    assert not {"shep-check", "shep-cmp", "shep-check-ok", "shep-check-amber", "shep-cmp-ok",
                "shep-cmp-amber", "shep-cmp-grey"} & bordered
    item = {**checkpoint(sha=SHA), "filename": HOSTILE, "run": WORLDLY, "architecture": HOSTILE,
            "record": "model", "learning_stage": "base", "task": None, "operation": "branch",
            "encoder": "gatv2", "differences": (), "family_id": WORLDLY,
            "parent_sha256": "d" * 64, "parent_run": HOSTILE,
            "parent_changes": {"scheduler_type": {"parent": WORLDLY, "new": HOSTILE},
                               "early_stopping_patience": {"parent": 10, "new": None}},
            "checkpoint_kind": "training_state", "end": "paused",
            "selection": {"monitor": "val_lp_accuracy", "mode": "max"}}
    inherited = {**dict(BASE),
                 "scheduler_type": HOSTILE, "task": None, "task_loss_weight": None,
                 "early_stopping_patience": None, "seed": None, "use_amp": False,
                 "model_spec": reference_spec("base").to_dict()}
    preview = {"ok": True, "parent": "/elsewhere/x.pt", "sha256": SHA, "architecture": "gatv2",
               "action": f"Resume the original training ({WORLDLY})", "inherited": inherited,
               "decoded": {}, "learning_stage": "base", "task": None}
    version = {"label": "v1", "role": WORLDLY, "kind": "supplied", "manifest_sha256": FP_A,
               "counts": {"source_cases": 3, "ranked_eligible": 2,
                          "excluded": {HOSTILE: 1}}, "imported_at": "2026-11-01T10:00:00Z",
               "source_format": HOSTILE}
    page = {"workspaces": [{"name": WORLDLY, "dataset_fingerprint": FP_A}],
            "versions": [version], "roots": []}
    models = {"items": [item], "workspace": WORLDLY}
    facts = r.facts_from_checkpoint(item, page["workspaces"])
    # The Test page's Result identity, from a kept report's record the one
    # reader reads (the review of B0c: no row of it was checked here); its
    # device is one the reader accepts (CUDA or CPU, else it is no result).
    from tests.unit.test_webui_test_page import _task_record

    record = _task_record("disease")
    record["cohort"].update(label=HOSTILE, role=WORLDLY)
    record["cohort_role"] = WORLDLY
    result = tp._record_html(record, {}, {})
    result = result[result.index('<dl class="shep-kv shep-test-ident-rows">'):]
    shown = {
        "parent details": tc.parent_details_html(item),
        "check": tc._check_html(preview, tc.CONTINUE_FROM_PATH),
        "base summary": ft.base_summary_html(item),
        # A model whose dataset is not shown as the workspace's current one
        # names the listed workspace that holds it.
        "Test model": tp.model_html(page, {**models, "items": [
            {**item, "trained_on_this_dataset": None}]}, tp.item_key(item)),
        "Test version": tp.version_html(page, "v1", None),
        "Models identity": mp.identity_html(item, {"parent_state": "elsewhere",
                                                   "parent_workspace": WORLDLY}),
        "Model Facts": r.model_facts_html(facts),
        "Test result identity": result[:result.index("</dl>") + 5],
    }
    # A supplied version's role is its option's (R-70), not a row under it.
    assert [words for words, _value in tp.version_choices(page)] == [f"v1 · role {WORLDLY}"]
    for where, html_ in shown.items():
        assert "<dl class=\"shep-kv" in html_, where
        assert HOSTILE not in html_ and "&lt;img src=x onerror=alert(1)&gt;" in html_, where
        assert WORLDLY in html_ or where == "Test version", where
        text = _shown_text(html_)
        assert not _REPR.search(text), (where, _REPR.search(text).group(0))
        assert not _nested_boxes(html_, bordered), (where, _nested_boxes(html_, bordered))
    check = shown["check"]
    assert "/elsewhere" not in check and "a path outside the workspaces root" in check
    rows = dict(re.findall(r'title="(\w+)">[^<]*</dt><dd class="shep-v">([^<]*)</dd>', check))
    assert rows["early_stopping_patience"] == "off" and rows["seed"] == "not set"
    assert rows["task"] == rows["task_loss_weight"] == "not taken by a base"
    assert rows["use_amp"] == "off" and rows["num_neighbors"] == "15, 10, 5"
    assert "Reference GATv2 spec" in check
    # The result's determinism by the one value-words rule, a row of the one
    # record rendering (`renderers.record_rows`, R-82); a record without the
    # flag holds none.
    rows = result[result.index('<dl class="shep-kv shep-rec-rows">'):]
    ident = dict(re.findall(r'<dt class="shep-k">([^<]*)</dt><dd class="shep-v">([^<]*)</dd>',
                            rows))
    assert ident["Deterministic algorithms"] == "on"
    record["regime"]["deterministic_algorithms"] = False
    assert ('Deterministic algorithms</dt><dd class="shep-v">off</dd>'
            in tp._record_html(record, {}, {}))


def test_every_digest_code_element_is_drawn_by_fingerprint_value():
    """B72-N3: the one digest value, `renderers.fingerprint_value` (its 12 hex
    digits, ``shep-fpv`` with the whole value as its title), draws every
    digest a page shows as a value; no page builds its own ``<code>`` around
    a digest (the load plan's From and To, the Diagnosis reference cohort and
    Stop question, the Test page's version details and the record's split
    manifest each did). A digest said inside a sentence (a notice, an
    option) is plain text and not a code element."""
    import inspect
    from pathlib import Path

    own = re.compile(r"<code[^>]*>\{[^}]*(short_hex|\[:12\]|sha256|digest|\[\"sha|\['sha)")
    allowed = inspect.getsource(r.fingerprint_value)
    found = []
    for path in sorted(Path("src/webui").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if path.name == "renderers.py":
            text = text.replace(allowed, "")
        found += [f"{path}:{n}" for n, line in enumerate(text.splitlines(), 1)
                  if own.search(line)]
    assert found == []
    # The places it drew, through the one value now: its class and title.
    from src.webui.components import model_list as ml

    side = {"sha256": "a" * 64, "graph": "b" * 64, "workspace": "w", "served": True}
    assert ml._cell(side, "sha", [], to=False) == r.fingerprint_value("a" * 64, what="model fingerprint",
                                                            copy=False)
    assert f'title="{"b" * 64}"' in ml._cell(side, "workspace", [], to=False)
    lines = dict(r.data_kind_lines({"cohort": {
        "kind": "generated", "generation": {}, "allocation": {},
        "split_manifest": {"sha256": "c" * 64}, "task_fixture": None}}))
    assert f'<code class="shep-fpv" title="{"c" * 64}">{"c" * 12}</code>' in lines[
        "Split manifest"]


# --------------------------------------------------------------------------- the notice
def _notices():
    from src.config import server_settings as ss
    from src.webui.components import workspaces as w

    return (w.setting_notice(ss.workspaces_root({})), w.lock_notice({"problem": "x", "unset": True}),
            w.setting_notice(ss.cohort_registry({})))


def test_a_settings_notice_is_chosen_by_its_flags_never_by_its_words(tmp_path):
    """`workspaces.setting_notice`: usable, none; unset, its notice (a normal
    setup step: neutral, never red); set and refused, its own short words in
    red with where to see why and no value or path of the setting. The lock
    (`lock_notice`): unset, the GPU notice, which follows from the folder's;
    another problem, its reason as a Details row for an operator only."""
    from src.config import server_settings as ss
    from src.webui.components import workspaces as w

    assert w.setting_notice(ss.workspaces_root({"SHEPHERD_WORKSPACES_ROOT": str(tmp_path)})) is None
    unset = w.setting_notice(ss.workspaces_root({}))
    assert unset["kind"] == "neutral" and unset["key"] == "workspaces_root"
    assert unset["summary"] == ss.WORKSPACES_ROOT_NOTICE.summary
    refused = w.setting_notice(ss.workspaces_root({"SHEPHERD_WORKSPACES_ROOT": str(tmp_path / "gone")}))
    assert refused["kind"] == "error" and refused["key"] == "workspaces_root"
    assert refused["summary"] == "The workspaces folder cannot be used."
    assert refused["action"] == "See why in Settings › Data locations" and refused["details"] == ()
    assert str(tmp_path) not in repr(refused)
    registry = w.setting_notice(ss.cohort_registry({"SHEPHERD_REGISTRY": "relative"}))
    assert registry["kind"] == "error" and registry["summary"] == "The results registry cannot be used."
    assert w.lock_notice(None) is None and w.lock_notice({"problem": None}) is None
    gpu = w.lock_notice({"problem": "x", "unset": True})
    assert gpu["key"] == "gpu_lock" and gpu["cause"] == "workspaces_root" and gpu["kind"] == "neutral"
    other = {"problem": "the lock /srv/x/.shepherd-gpu.lock cannot be opened", "unset": False}
    assert w.lock_notice(other)["details"] == ()
    assert w.lock_notice(other, operator=True)["details"] == (f"Why: {other['problem']}",)
    assert w.lock_notice(other)["summary"] == "The GPU lock cannot be used."


def test_the_workspace_list_carries_its_notice_from_the_same_read(tmp_path, monkeypatch):
    from src.webui.components import workspaces as w

    for name in ("SHEPHERD_WORKSPACES_ROOT", "SHEPHERD_SETTINGS_FILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SHEPHERD_SETTINGS_FILE", str(tmp_path / "none.json"))
    unset = w.read_workspaces({"SHEPHERD_SETTINGS_FILE": str(tmp_path / "none.json")})
    assert unset["rows"] == [] and unset["notice"]["key"] == "workspaces_root"
    (tmp_path / "ws").mkdir()
    usable = w.read_workspaces({"SHEPHERD_WORKSPACES_ROOT": str(tmp_path / "ws")})
    assert usable["notice"] is None and usable["problem"] is None


def test_a_notice_is_one_line_its_link_and_details_one_click_away():
    """`renderers.notice_block` (the text brief's §1, §12): the state with its
    status mark in a column of its own, the page's consequence, the next step
    as a button to its place (``data-shep-go``, ``data-shep-focus``), and the
    why and how under "Details ›": term rows and one Ref line, the notices it
    merges (`also`) said once there and named by ``data-notice-also``."""
    ws, gpu, _registry = _notices()
    html = r.notice_block(ws, consequence="Start is unavailable until one is chosen.", also=(gpu,))
    assert html.startswith('<div class="shep-nb shep-nb-neutral" data-notice="workspaces_root" '
                           'data-notice-also="gpu_lock">')
    line = html.split("</p>", 1)[0]
    assert '<span class="shep-nb-mark shep-st-neutral" aria-hidden="true">○' in line
    assert ">No workspaces folder is set.</span>" in line
    assert '<span class="shep-nb-then">Start is unavailable until one is chosen.</span>' in line
    assert ('<button type="button" class="shep-link shep-nb-go" data-shep-go="settings" '
            'data-shep-focus="shep-dl">Choose one in Settings › Data locations</button>') in line
    assert "GPU jobs are refused" not in html                    # merged: its details only
    more = html.split('<details class="shep-more shep-nb-more"><summary>Details ›</summary>')[1]
    for term in ("Workspaces", "Unset", "Suggested", "Default", "GPU lock", "Taken by", "GPU jobs"):
        assert f'<dt class="shep-k">{term}</dt>' in more, term
    assert ('<code class="shep-tok"><span>.shepherd</span><wbr><span>-gpu</span><wbr>'
            '<span>.lock</span></code>') in more
    assert more.count("Ref:") == 1 and "deployment guide §4.11" in more
    assert re.sub(r"<[^>]+>", "", more).count("SHEPHERD_WORKSPACES_ROOT") == 1
    assert r.notice_block(None) == "" and r.notice_block("") == ""
    # An older caller's whole text: the line alone, no link, no details.
    old = r.notice_block("Something is off.")
    assert "Something is off." in old and "data-shep-go" not in old and "<details" not in old
    # No link where the page asks for none: the step is a sentence.
    assert '<span class="shep-nb-act">Choose one in Settings › Data locations.</span>' in \
        r.notice_block(ws, go=False)


def test_a_notice_with_steps_numbers_them_with_its_action_first():
    ws, _gpu, _registry = _notices()
    html = r.notice_block(ws, steps=("Build a dataset in it: scripts/build_knowledge_graph.py "
                                     "--generate-samples",))
    line, rest = html.split("</p>", 1)
    assert "data-shep-go" not in line
    steps = rest.split('<ol class="shep-nb-steps">')[1].split("</ol>")[0]
    first, second = re.findall(r"<li>(.*?)</li>", steps)
    assert "data-shep-go" in first
    assert second == ('Build a dataset in it: <code class="shep-tok"><span>scripts</span><wbr>'
                      '<span>/build</span><wbr><span>_knowledge</span><wbr><span>_graph</span>'
                      '<wbr><span>.py</span></code> <code class="shep-tok"><span>--generate-samples</span>'
                      '</code>')


def test_several_notices_are_one_block_with_one_action_when_their_steps_agree():
    from src.webui.components import workspaces as w

    ws, _gpu, registry = _notices()
    html = r.notices_block([registry, ws], consequence="Import and Start are unavailable.")
    assert html.count("shep-nb-line") == 2 and html.count("data-shep-go") == 1
    assert 'data-notice="registry workspaces_root"' in html
    assert html.count("<details") == 1 and html.count("Ref:") == 1
    apart = dict(ws, action="Open Datasets", target="datasets")
    split = r.notices_block([registry, apart])
    assert split.count("data-shep-go") == 2 and 'data-shep-go="datasets"' in split
    assert r.notices_block([None, ws]) == r.notice_block(ws)
    assert r.notices_block([]) == ""
    refused = w.setting_notice(__import__("src.config.server_settings", fromlist=["x"])
                               .cohort_registry({"SHEPHERD_REGISTRY": "relative"}))
    assert r.notices_block([refused, ws]).startswith('<div class="shep-nb shep-nb-error"')


def test_several_notices_share_one_step_in_the_sources_words_and_group_their_details():
    """S12 of the sweep's integration (the lanes' L3 R2): several notices whose steps lead to one
    place draw one link for the block, in the words the settings' source gives several notices
    (`server_settings.SEVERAL_ACTIONS`: "Choose them in …", where one notice's "Choose one in …"
    was drawn for two; where their actions differ, the place itself, `PLACE_ACTIONS`), never
    words built from a sentence; and the block's "Details ›" holds each notice's rows under its
    setting's name (`Notice.name`), so a row reads as the fact of the setting it explains, and
    each state is said once in the block."""
    from src.config import server_settings as ss
    from src.webui.components import workspaces as w

    ws, _gpu, registry = _notices()
    html = r.notices_block([registry, ws], consequence="Testing waits for both.")
    links = re.findall(r'<button[^>]*data-shep-go="settings"[^>]*>([^<]*)</button>', html)
    assert links == [r.e(ss.CHOOSE_THEM_IN_DATA_LOCATIONS)], links
    assert ss.CHOOSE_IN_DATA_LOCATIONS not in html
    details = html.split("<details", 1)[1]
    subs = re.findall(r'<p class="shep-nb-sub">([^<]*)</p>', details)
    assert subs == [ss.RESULTS_REGISTRY, ss.WORKSPACES_FOLDER], subs
    for state in (registry["summary"], ws["summary"]):
        assert html.count(r.e(state)) == 1, state
    first, second = details.split('<p class="shep-nb-sub">')[1:3]
    assert "Holds" in first and "Holds" not in second and "Workspaces" in second, details
    assert details.count("Ref:") == 1
    refused = w.setting_notice(ss.cohort_registry({"SHEPHERD_REGISTRY": "relative"}))
    mixed = r.notices_block([refused, ws])
    assert re.findall(r"data-shep-go[^>]*>([^<]*)</button>", mixed) == [r.e(ss.OPEN_DATA_LOCATIONS)]
    both = r.notices_block([refused, w.setting_notice(ss.workspaces_root({"SHEPHERD_WORKSPACES_ROOT": "x"}))])
    assert re.findall(r"data-shep-go[^>]*>([^<]*)</button>", both) == [r.e(w.SEE_WHY)]
    # One notice keeps its own words.
    assert r.e(ss.CHOOSE_IN_DATA_LOCATIONS) in r.notices_block([ws])


def test_a_relation_says_the_notice_short_and_is_not_the_notice_again():
    ws, gpu, _registry = _notices()
    rel = r.see_above("Start", ws)
    assert rel.startswith('<p class="shep-reason shep-nb-rel" data-notice-rel="workspaces_root">')
    assert ">Start disabled: no workspaces folder is set.</span>" in rel
    assert "data-notice=" not in rel and "<details" not in rel
    assert (">Import disabled: GPU jobs need a workspaces folder "
            '<span class="shep-nowrap">(see Data).</span></span>') in r.see_above(
        "Import", gpu, where="Data")
    assert r.relation_words("Import", gpu, where="Data") == (
        "Import disabled: GPU jobs need a workspaces folder (see Data).")
    assert "<details" in r.see_above("Start", gpu, details=True)
    assert r.see_above("Start", None) == "" and r.relation_words("Start", None) == ""


def test_a_disabled_control_is_said_one_way_and_marked_one_way_on_every_page(monkeypatch):
    """C11 (the L6 check): one disabled state read "Start unavailable: …" on
    Training and Datasets, "Start disabled: …" on Fine-tune and Test, and
    "Disabled: …" or "Not available: …" for a page's own reasons; the
    relations led with ○, Training's and Fine-tune's own reasons with a glyph
    of their own (⊖), Test's with none. Now every page's reason beside or
    under a disabled control names it, says "disabled" (`disabled_words`,
    `relation_words`) and leads with one mark: the relation's ○ for a setting
    not yet set, `DISABLED_MARK` (the same ○, grey) for a page's own reason."""
    import re

    from src.webui.components import datasets_page as dsp
    from src.webui.components import finetune_page as fp
    from src.webui.components import test_page as tp
    from src.webui.components import training_console as tc

    ws, _gpu, _registry = _notices()
    relation_mark = re.search(r'<span class="shep-nb-mark[^"]*"[^>]*>[^<]*',
                              r.see_above("Start", ws))
    assert relation_mark and r.DISABLED_MARK.startswith(relation_mark.group(0))
    assert r.disabled_words("Start", "x.") == "Start disabled: x."
    # Every page's lines, the relations and the pages' own reasons alike.
    access = type("A", (), {"allowed": True, "condition": None})()
    training = tc._run_bar_actions({"status": "idle"}, {"choices": []}, {}, access=access,
                                   active=None, gpu=None)
    finetune = fp.actions_view({"status": "running"}, access=access, gpu=None, confirming=False)
    page = {"access": {"allowed": True}, "workspaces": []}
    test_start = tp.start_reason_html(page, True, False, tp.start_state(
        page, {}, None, None, None, None, ("", ""))[1])
    test_import = tp.import_reason_html({"access": {"allowed": True}, "workspaces": []}, None)
    datasets = dsp.use_note_html({"rows": [{"name": "w", "trainable": False,
                                            "manifest": "absent"}]}, "w")
    lines = {"Training": training["reason"], "Training drawer": training["drawer_reason"],
             "Fine-tune": finetune["reason"],
             "Test start": test_start, "Test import": test_import, "Datasets": datasets,
             "relation": r.see_above("Use for training", ws)}
    for where, html in lines.items():
        words = re.sub(r"<[^>]+>", "", html).replace(r.STATUS_MARKS["neutral"][0], "").strip()
        assert re.match(r"[A-Z][\w …]* disabled: \S", words), (where, words)
        assert "unavailable" not in words and "Not available" not in words, (where, words)
        assert html.count(r.STATUS_MARKS["neutral"][0]) == 1, (where, html)
        assert "<svg" not in html, (where, html)


PAYLOAD = "<img src=x onerror=alert(1)>"


def test_every_part_of_a_notice_is_escaped():
    """No markup comes from a notice: each part, the key and the place
    included, is drawn as text (an ``<img src=x onerror=…>`` in every part
    shows as its characters)."""
    bad = {"key": PAYLOAD, "kind": "error", "summary": PAYLOAD, "short": PAYLOAD,
           "action": PAYLOAD, "target": f'{PAYLOAD}#"{PAYLOAD}', "details": (f"Why: {PAYLOAD}",
                                                                              PAYLOAD),
           "ref": (PAYLOAD,), "cause": None}
    html = (r.notice_block(bad, consequence=PAYLOAD, also=(dict(bad, key="b"),), steps=(PAYLOAD,))
            + r.notices_block([bad, dict(bad, key="c", target="x")], consequence=PAYLOAD)
            + r.notices_block([bad, dict(bad, key="d")], consequence=PAYLOAD)
            + r.see_above(PAYLOAD, bad, where=PAYLOAD, details=True)
            + r.notice_block(PAYLOAD) + r.literal_html(PAYLOAD) + r.token_html(PAYLOAD))
    assert "<img" not in html and "onerror=alert(1)>" not in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html
    assert 'data-shep-go="&lt;img' in html


def test_a_token_breaks_only_at_its_own_separators():
    """The text brief's §8: a path or a variable breaks only before its own
    separators, never inside a piece and never with a drawn hyphen: each piece
    is kept on one line (the browser would break after a hyphen by itself)
    and a ``<wbr>`` stands between pieces; a copy reads the token whole."""
    assert r.token_html("SHEPHERD_WORKSPACES_ROOT") == (
        '<code class="shep-tok"><span>SHEPHERD</span><wbr><span>_WORKSPACES</span><wbr>'
        '<span>_ROOT</span></code>')
    assert r.token_html("--registry") == '<code class="shep-tok"><span>--registry</span></code>'
    assert re.sub(r"<[^>]+>", "", r.token_html("a/b-c.d_e")) == "a/b-c.d_e"
    # A command's option is one piece: the command breaks at the space before it, never
    # inside it (the sweep's L7, SW-LK7: "--generate" / "-samples" read as two options).
    command = r.token_html("scripts/build_knowledge_graph.py --generate-samples")
    assert "<span>--generate-samples</span>" in command
    assert re.sub(r"<[^>]+>", "", command) == "scripts/build_knowledge_graph.py --generate-samples"
    assert r.token_html("a/b-c", classes="shep-x") == (
        '<code class="shep-tok shep-x"><span>a</span><wbr><span>/b</span><wbr><span>-c</span></code>')
    assert "​" not in r.literal_html("a/b/c.json and SHEPHERD_X")
    assert r.literal_html("Settings › Data locations, deployment guide §4.11") == \
        "Settings › Data locations, deployment guide §4.11"
    assert r.literal_html("the lock .shepherd-gpu.lock here").count("<code") == 1


def test_a_line_never_parts_a_count_from_its_unit_or_a_date_from_its_time():
    """The sweep's L7 check, C11, and LOOK2's SW-LK7: narrow cells read "1" / "epoch
    completed" (the Diagnosis card and Model Facts), "recorded epoch" / "9" (Models' Epochs
    cell), "GATv2 2026-10-02" / "00:00" and "(see" / "Data)." (Test's relations). A count and
    the word after it, a word and the number that ends its text, a date and its time, and a
    parenthesis of three words at most with its stop are each one piece a line never parts
    (`bound_html`), drawn by every chain fact, status mark, relation and notice sentence; the line
    breaks at any other space, and the text reads as it was given."""
    import html as html_lib

    def pieces(html):
        return re.findall(r'<span class="shep-nowrap">([^<]*)</span>', html)

    def text(html):
        return html_lib.unescape(re.sub(r"<[^>]+>", "", html))

    name = "GATv2 2026-10-02 00:00, 1 epoch completed"
    assert pieces(r.bound_html(name)) == ["2026-10-02 00:00", "1 epoch"]
    assert text(r.bound_html(name)) == name
    assert pieces(r.bound_html("recorded epoch 9")) == ["epoch 9"]
    assert pieces(r.bound_html("Start disabled: no results registry is set (see Data).")) == [
        "(see Data)."]
    assert pieces(r.bound_html("The selected workspace (measured, always)")) == [
        "(measured, always)"]
    # A digest, a version and a longer parenthesis keep their own breaks; words are escaped.
    assert pieces(r.bound_html("3fa9c1e2b7d0 Copy")) == []
    assert pieces(r.bound_html("(a clause of more than three words)")) == []
    assert text(r.bound_html("<b>2 runs</b>")) == "<b>2 runs</b>" and "<b>" not in r.bound_html("<b>")
    # Every chain fact, status mark and relation draws its words so.
    assert '<span class="shep-nowrap">1 epoch</span>' in r.chain_html(name)
    assert '<span class="shep-nowrap">epoch 9</span>' in r.status_mark("neutral", "recorded epoch 9")
    relation = r.see_above("Start", {"key": "k", "kind": "neutral", "short": "no folder is set"},
                           where="Data")
    assert '<span class="shep-nowrap">(see Data).</span>' in relation
    # A notice's sentence too (its Details rows, its steps), its tokens still code.
    said = r.literal_html("Choose one in Settings › Data locations (Create suggested folder)")
    assert pieces(said) == ["(Create suggested folder)"]
    from src.webui.style import SHELL_CSS
    assert ".shep-nowrap { white-space: nowrap; }" in SHELL_CSS



def test_a_fields_line_and_a_muted_line_of_its_own_keep_the_prose_measure():
    """The sweep's L7 check, C12: Settings' torch.compile line (Gradio's field info) ran to
    about 88 characters at 1850 and 106 at 2560, and its Acceleration sentence (a muted line
    after a break) to 67, while the panel's paragraphs held the text brief's measure. The one
    measure (``--shep-measure``) covers both, as it covers a panel's paragraphs."""
    from src.webui.style import SHELL_CSS

    measure = "max-width: var(--shep-measure)"
    for selector in (":where(.shep-panel p, .shep-edge p)", ":where(.gradio-container .info-text)",
                     ":where(br + .shep-muted)"):
        rule = SHELL_CSS[SHELL_CSS.index(selector + " {"):].split("}", 1)[0]
        assert measure in rule, selector
