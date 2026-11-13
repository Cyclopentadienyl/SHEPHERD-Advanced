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
  unreadable checkpoint greyed "metadata not readable · cannot be loaded";
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

import re

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
    return re.sub(r"<[^>]+>", "", html)


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
    """How far it trained in the one wording (`epoch_words`): the epochs its
    progress record says it completed; an older file without one, the epoch
    index it records, never a count made from it."""
    assert r.model_name_line({**checkpoint(), "completed_epochs": 6}) == (
        "GATv2 · run 2026-10-17 09:15 · 6 epochs completed")
    assert r.model_name_line(checkpoint()) == "GATv2 · run 2026-10-17 09:15 · recorded epoch 5"
    assert r.model_name_line(checkpoint(epoch=None, run="odd")) == (
        "GATv2 · run odd · epochs not recorded")


def test_trained_on_every_listed_workspace_with_the_full_digest():
    listed = [workspace("demo"), workspace("demo_copy"), workspace("other", FP_B)]
    html = r.model_row_html(checkpoint(), listed, key="gatv2/r/f")
    plain = text_of(html)
    assert "trained on demo, demo_copy · dataset fingerprint 2be1f04c9a83" in plain
    assert "model fingerprint 15f098277b54" in plain and 'shep-tag-amber">synthetic' in html
    # a twin that shares 12 hex only is not a match: comparisons are on the full digest
    twin = [workspace("twin", FP_A[:12] + "f" * 52)]
    assert "matches no listed workspace" in text_of(r.model_row_html(checkpoint(), twin))


def test_the_row_shows_recorded_tests_as_a_state_never_a_number():
    html = r.model_row_html(checkpoint(validation=0.3180), [workspace()],
                            recorded=r.RecordedTests("records", 3, 1))
    plain = text_of(html)
    assert "Recorded results: 3 · runs differ for 1" in plain
    assert "0.318" not in plain
    assert r.recorded_tests_text(r.RecordedTests("none")) == "Recorded results: none in ledger"
    assert r.recorded_tests_text(r.RecordedTests("no_ledger")) == "Recorded results: no ledger"
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
        "Recorded results: 3 · 2 not readable; runs differ for 1; comparison incomplete for 1")
    assert text(r.RecordedTests("records", 3, 1, unreadable=2)).count(" · ") == 1
    assert r.recorded_tests_cell(r.RecordedTests("records", 3, 0, unreadable=1)) == (
        "3 · 1 not readable")
    html = r.model_row_html(checkpoint(), [workspace()],
                            recorded=r.RecordedTests("records", 0, 0, unreadable=1))
    assert "Recorded results: 0 · 1 not readable" in text_of(html)
    assert "none in ledger" not in text_of(html)


def test_an_unreadable_checkpoint_is_greyed_and_cannot_be_loaded():
    html = r.model_row_html(checkpoint(sha=None, problem="UnpicklingError: no"), [workspace()])
    assert 'aria-disabled="true"' in html and "shep-row-dis" in html
    assert "metadata not readable · cannot be loaded" in html


def test_in_use_marks_the_row():
    html = r.model_row_html(checkpoint(), [workspace()], in_use=True,
                            in_use_label="In use (Diagnosis)")
    assert 'shep-tag-accent">In use (Diagnosis)<' in html


# --------------------------------------------------------------------------- Model Facts
def test_model_facts_has_its_five_sections_and_warnings_always_open():
    facts = r.facts_from_checkpoint(checkpoint(), [workspace()],
                                    recorded=r.RecordedTests("none"),
                                    serving_workspace=workspace())
    html = r.model_facts_html(facts)
    for number, title in enumerate(("Identity", "Uses", "Data", "Evidence"), start=1):
        assert f'<span class="shep-facts-n">{number}</span> {title}' in html
    assert '<details class="shep-facts-sec" data-sec="identity" open>' in html
    assert '<section class="shep-facts-sec shep-facts-warn" data-sec="warnings">' in html
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
    assert "Servingserving reads workspace demo: the same dataset" in text_of(data)
    assert "SYNTHETIC WORKSPACE: not a clinical result." in plain
    assert "serving reads workspace demo: the same dataset" in plain
    assert "runs/gatv2/20261017-091500_run_3fa9c1/checkpoints/model-05-0.2410.pt" in plain
    assert SHA in plain and FP_A in plain           # Identity: the full digests, with Copy


def test_model_facts_warns_of_a_dataset_no_listed_workspace_holds():
    facts = r.facts_from_checkpoint(checkpoint(fingerprint="9b1f04e6d2c8" + "0" * 52),
                                    [workspace()])
    plain = text_of(r.model_facts_html(facts))
    assert "TRAINED ON A DATASET NO LISTED WORKSPACE HOLDS." in plain
    assert "Trained onmatches no listed workspace" in plain
    facts = r.facts_from_checkpoint(checkpoint(fingerprint=FP_B), [workspace("o", FP_B, "files")])
    assert "No warnings." in text_of(r.model_facts_html(facts))


def test_the_medical_device_line_is_the_one_diagnosis_uses():
    from src.webui.components import diagnosis_panel as dp

    assert dp.NOT_A_MEDICAL_DEVICE is r.NOT_A_MEDICAL_DEVICE
    assert dp._IDENTITY_BANNERS is r.IDENTITY_BANNERS


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


def test_the_research_table_row_has_its_columns_and_escapes():
    row = r.model_table_row_html(checkpoint(run="20261017-091500_<b>x"), [workspace()],
                                 recorded=r.RecordedTests("records", 3, 1), in_use=True,
                                 key="gatv2/r/f.pt", in_use_label="In use (Diagnosis)",
                                 extra="<span>regime</span>")
    # A plain table's row: `aria-current` when selected, never `aria-selected` (skill §8).
    assert row.startswith('<tr class="" data-key="gatv2/r/f.pt" tabindex="0">')
    assert "aria-selected" not in row
    chosen = r.model_table_row_html(checkpoint(), [workspace()], selected=True, key="k")
    assert chosen.startswith('<tr class="shep-on" data-key="k" tabindex="0" aria-current="true">')
    cells = re.findall(r"<td[^>]*>(.*?)</td>", row)
    assert len(cells) == 5
    assert 'GATv2 · <span class="shep-nowrap">run 2026-10-17 09:15</span>' in cells[0]
    assert "recorded epoch 5 · &lt;b&gt;x" in cells[0]
    assert "In use (Diagnosis)" in cells[0] and "<b>x" not in row
    assert f'model <code class="shep-fpv" title="{SHA}">{SHA[:12]}</code>' in cells[1]
    assert "dataset <b>demo</b>" in cells[1] and "synthetic" in cells[1]
    assert cells[2] == '0.3180<div class="shep-sub">Validation pooled MRR (Trainer)</div>'
    assert cells[3] == "<span>regime</span>"
    assert cells[4] == "3 · runs differ for 1"
    unmatched = r.model_table_row_html(checkpoint(fingerprint=FP_B), [workspace()])
    assert "matches no listed workspace" in unmatched and unmatched.count("<td") == 4
    broken = r.model_table_row_html(checkpoint(sha=None, problem="bad"), [])
    assert 'aria-disabled="true"' in broken and "metadata not readable · cannot be loaded" in broken


def test_a_runs_start_time_is_kept_on_one_line_in_the_table():
    """Models at 1440x900 broke "GAT · run 2026-10-" / "17 09:15": the start
    time is one no-wrap span, so a line breaks after "GAT ·" only; a run not
    named by its start time is its escaped name, free to wrap."""
    assert r.run_label_html("gat", "20261017-091500_run_3fa9c1") == (
        'GAT · <span class="shep-nowrap">run 2026-10-17 09:15</span>')
    assert r.run_label_html(None, "my<run>") == "MODEL · run my&lt;run&gt;"
    assert r.run_label("gat", "20261017-091500_run_3fa9c1") == "GAT · run 2026-10-17 09:15"


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
    assert r.MODEL_TABLE_COLUMNS[2] == "Validation (Trainer)"


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
    # A record never read keeps its own word (skill §6), apart from a refused one.
    assert r.record_kind_text({**checkpoint(problem="torn"), "record": None}) == (
        "Record not read")


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
        "Record not read")


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
    assert r.run_label("gatv2", "20261017-091500_run_3fa9c1") == "GATv2 · run 2026-10-17 09:15"
    assert r.run_label_html("gatv2", "20261017-091500_run_3fa9c1") == (
        'GATv2 · <span class="shep-nowrap">run 2026-10-17 09:15</span>')
    assert r.model_name_line(_record()) == "GATv2 · run 2026-10-17 09:15 · recorded epoch 5"
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
    assert "<b>odd" not in odd and "&lt;B&gt;ODD · reference recipe" in odd
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
             r.validation_cell(None, read=False), r.model_table_row_html(absent, [workspace()])]
    for text in shown:
        assert "—" not in text, text
    assert r.count_text(None) == "not recorded" and r.count_text(1200) == "1,200"
    # A record's count and a manifest's count: one word for the same state.
    assert r.recorded_count_text(None) == r.count_text(None)
    assert r.recorded_count_text(1200) == "1200"
    assert "epochs not recorded" in r.model_table_row_html(absent, [workspace()])
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


def test_the_value_words_say_each_setting_by_its_meaning_never_by_repr():
    """One rule for every page (B0c): on/off for a switch, a None by its
    meaning per setting (the field spec's), a list as its items with -1 as
    every neighbour, a number as typed, a closed value by its field's words;
    never Python's repr, never a bare None."""
    words = r.setting_words
    assert (words("use_amp", True), words("use_amp", False)) == ("on", "off")
    assert words("early_stopping_patience", None) == "off"
    assert words("seed", None) == r.NOT_SET == "not set"
    assert words("task_loss_weight", None, stage="finetune", task="disease") == (
        "the task's reference")
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
    """The words of an unset setting (`SETTING_UNSET_WORDS`) are the field
    spec's, never a page's own: each setting whose spec says "Unset is the
    task's reference" reads so, the one whose spec says none "is no early
    stopping" reads "off", and no other setting has words of its own (an
    unset seed is "not set")."""
    from src.config.training_fields import FIELDS

    reference = {f.name for f in FIELDS if "Unset is the task's reference" in f.divergence_note}
    no_stopping = {f.name for f in FIELDS if "is no early stopping" in f.description}
    assert reference and no_stopping == {"early_stopping_patience"}
    assert r.SETTING_UNSET_WORDS == {**{name: "the task's reference" for name in reference},
                                     "early_stopping_patience": "off"}


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
        "Test model": tp.model_html(page, models, tp.item_key(item)),
        "Test version": tp.version_html(page, "v1", None),
        "Models identity": mp.identity_html(item, {"parent_state": "elsewhere",
                                                   "parent_workspace": WORLDLY}),
        "Model Facts": r.model_facts_html(facts),
        "Test result identity": result[:result.index("</dl>") + 5],
    }
    for where, html_ in shown.items():
        assert "<dl class=\"shep-kv" in html_, where
        assert HOSTILE not in html_ and "&lt;img src=x onerror=alert(1)&gt;" in html_, where
        assert WORLDLY in html_, where
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
    # The result's determinism by the one value-words rule; a record
    # without the flag holds none.
    ident = dict(re.findall(r'<dt class="shep-k">([^<]*)</dt><dd class="shep-v">([^<]*)</dd>',
                            shown["Test result identity"]))
    assert ident["Deterministic algorithms"] == "on"
    record["regime"]["deterministic_algorithms"] = False
    assert ('Deterministic algorithms</dt><dd class="shep-v">off</dd>'
            in tp._record_html(record, {}, {}))
