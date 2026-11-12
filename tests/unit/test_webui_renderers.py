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
    assert r.model_name_line(checkpoint()) == "GATv2 · run 2026-10-17 09:15 · epoch 5"
    assert r.model_name_line(checkpoint(epoch=None, run="odd")) == (
        "GATv2 · run odd · epoch not recorded")


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
    assert ("Validation pooled MRR (Trainer) 0.3180 — selection metric on val, not an "
            "independent test") in plain
    assert "Recorded results: none in ledger" in plain
    assert "SYNTHETIC WORKSPACE: not a clinical result." in plain
    assert "serving reads workspace demo: the same dataset" in plain
    assert "runs/gatv2/20261017-091500_run_3fa9c1/checkpoints/model-05-0.2410.pt" in plain
    assert SHA in plain and FP_A in plain           # Identity: the full digests, with Copy


def test_model_facts_warns_of_a_dataset_no_listed_workspace_holds():
    facts = r.facts_from_checkpoint(checkpoint(fingerprint="9b1f04e6d2c8" + "0" * 52),
                                    [workspace()])
    plain = text_of(r.model_facts_html(facts))
    assert "TRAINED ON A DATASET NO LISTED WORKSPACE HOLDS." in plain
    assert "Trained on matches no listed workspace" in plain
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
def test_the_workspace_options_are_one_label_for_every_page():
    """Diagnosis' drawer, Models, Test and Training's run bar label a listed
    workspace the same way: name, the fingerprint's distinguishing digits,
    synthetic; or why it has none."""
    twin = FP_A[:12] + "f" * 52
    choices = r.workspace_choices([
        workspace("demo"), workspace("twin", fingerprint=twin, origin="files"),
        {**workspace("scratch", fingerprint=None, trainable=False), "manifest": "absent"},
        workspace("broken", fingerprint=None, trainable=False)])
    assert choices == [
        (f"demo · dataset fingerprint {FP_A[:16]} · synthetic", "demo"),
        (f"twin · dataset fingerprint {twin[:16]}", "twin"),
        ("scratch · no split manifest", "scratch"),
        ("broken · no readable split manifest", "broken")]


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
    assert row.startswith('<tr class="" data-key="gatv2/r/f.pt" tabindex="0" aria-selected="false"')
    cells = re.findall(r"<td[^>]*>(.*?)</td>", row)
    assert len(cells) == 5
    assert 'GATv2 · <span class="shep-nowrap">run 2026-10-17 09:15</span>' in cells[0]
    assert "epoch 5 · &lt;b&gt;x" in cells[0]
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
    assert (f'<span class="shep-k shep-kv-wide">{r.FACTS_MODEL_FP}</span>'
            f'<span class="shep-v shep-kv-wide"><code class="shep-full">{"a" * 64}</code>') in html_
    assert f'<span class="shep-k shep-kv-wide">{r.e(r.FACTS_TRAINED_FP)} · demo</span>' in html_
    assert html_.count("shep-copy shep-copy-icon") == 2
    assert '<span class="shep-k">Model</span>' in html_       # short labels keep the grid
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
    from src.webui.components.training_console import OPTION_CHARS

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
    assert r.model_name_line(_record()) == "GATv2 · run 2026-10-17 09:15 · epoch 5"
    assert (r.architecture_label("gat"), r.architecture_label("hgt"),
            r.architecture_label(None)) == ("GAT", "HGT", "MODEL")


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
    from src.webui.components import training_console as tc

    assert not hasattr(tc, "_METRIC_WORDS"), "one map, the renderers'"
    item = {"filename": "last.pt", "sha256": SHA, "completed_epochs": 1,
            "validation": {"metric": metric, "value": 0.5}}
    # The figure is the chosen file's details on Training, not its option (K6b:
    # the skill's 40 characters and one separator).
    assert tc.validation_words(item) == f"{label} 0.5000"
    assert tc.checkpoint_words(item) == "Latest save · 1 epoch completed"


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
    words = r.MISSING_WORDS
    assert words == ("Off", "Not loaded", "Not applicable", "Unknown", "Cannot be verified",
                     "Not recorded")
    assert len(set(w.lower() for w in words)) == len(words)
    assert r.NO_VALUE == "—" and r.NO_VALUE not in words
    absent = checkpoint(epoch=None, validation=None, sha=SHA)
    shown = [r.short_hex(None), r.count_text(None), r.model_name_line(absent),
             r.validation_cell(None), r.validation_cell(None, read=False),
             r.model_table_row_html(absent, [workspace()])]
    for text in shown:
        assert "—" not in text, text
    assert r.count_text(None) == "not recorded" and r.count_text(1200) == "1,200"
    assert "epoch not recorded" in r.model_table_row_html(absent, [workspace()])
    # The record's states, each its own words: absent apart from unverifiable.
    states = {binding: r.record_words(workspace(binding=binding))
              for binding in ("verified", "unrecorded", "unverifiable", "mismatch", None)}
    assert states == {"verified": ("ok", "record verified"),
                      "unrecorded": ("neutral", "record not recorded"),
                      "unverifiable": ("neutral", "record cannot be verified"),
                      "mismatch": ("error", "record does not match"),
                      None: ("neutral", "record not read")}
    assert r.record_chip(workspace(binding="unrecorded")) == r.status_mark(
        "neutral", "record not recorded", as_chip=True)
