"""
The shared renderers: one way to show a dataset, a model and its facts (M5-2, UI-A).
====================================================================================
Pure functions from the server's list items (`src.api.services.server_lists`) to
escaped HTML, used by every page that shows a dataset or a model: Datasets,
Training's run bar, Test, Models and Diagnosis' model list and card. One place
for the words, so one fact never reads two ways.

**Two identifiers, always labelled** (the mockup's round 2): the model's own is
the "model fingerprint" (the SHA-256 of the checkpoint's bytes), the dataset's
the "dataset fingerprint" (the SHA-256 of the workspace's
``split_manifest.json``, which binds the graph files, both sample files and the
dataset record). Twelve hex digits are shown, in bold monospace with the label
muted; the full 64 are copied (`copy_button`). **Every comparison is on the full
digests**: two listed workspaces whose fingerprints share their first 12 digits
are both shown with as many digits as tell them apart
(`distinguishing_digits`), and "trained on" names every listed workspace whose
full fingerprint equals the one the checkpoint records.

**What is never shown here:** a test result's numbers (a model row carries the
recorded tests' state only), a server path (a checkpoint file is named relative
to its workspace), or anything unescaped: a workspace or run name may hold
markup characters.

Module: src/webui/components/renderers.py
"""
from __future__ import annotations

import html
import math
import re
from collections import Counter
from datetime import datetime, timezone
from typing import (Any, Callable, Dict, Hashable, Iterable, List, Mapping, NamedTuple,
                    Optional, Sequence, Tuple)

from src.config.recipe import FROZEN_PARTS, unread_settings
from src.evaluation.caveats import (
    EXPOSURE_CAVEAT,
    GENERATED_EXPOSURE_NOTE,
    R10_LABEL_TEXT,
    R10_STATE_TEXT,
    R10_UNVERIFIABLE_REASON,
    TASK_NAMES,
    exposure_text,
    recorded_value_text,
)
from src.utils.run_plan import recorded_frozen_parts
from src.webui.components.help import copy_button, help_button

#: Hex digits of a digest shown by default.
SHORT_HEX = 12

#: The code's fixed statement (README; every report and identity block).
NOT_A_MEDICAL_DEVICE = (
    "Not a certified medical device; decision support for a qualified clinician."
)
#: Added to it in Model Facts' "Uses" (the mockup's round 2, item 12).
SYNTHETIC_FOR_TESTING = "Synthetic workspaces are for testing only."

#: (headline, detail) for each state a reader must not miss: the Diagnosis tab's
#: banners and Model Facts' warnings say them in these words.
IDENTITY_BANNERS = {
    "not_reported": (
        "WHAT PRODUCED THIS RESULT WAS NOT REPORTED.",
        "The API's response carried no served identity, so these candidates "
        "cannot be attributed to a model, graph or table.",
    ),
    "synthetic": (
        "SYNTHETIC WORKSPACE: not a clinical result.",
        "The served workspace's dataset record gives its origin as synthetic: a "
        "generated stand-in, not a clinical knowledge graph. Results are for "
        "testing only.",
    ),
}

#: The help texts the renderers' ⓘ name; the shell registers them.
RENDERER_HELP = {
    "dataset-fingerprint": (
        "Dataset fingerprint",
        "The SHA-256 of this workspace's split_manifest.json. The manifest binds by digest "
        "the graph, the three tensors, the train and val samples and the dataset record, "
        "plus the recipe, the seeds and the split. 12 of its 64 hex digits are shown; every "
        "comparison uses all 64. Equal fingerprints mean the same recorded build: the "
        "manifest names the same bytes by digest. The files themselves are checked against "
        "it when a run or a load reads them.",
    ),
    "model-fingerprint": (
        "Model fingerprint",
        "The SHA-256 of the checkpoint file's bytes, as the server read them. 12 of its 64 "
        "hex digits are shown; every comparison uses all 64. A test record, a usage history "
        "entry and the served identity name the model by it.",
    ),
    "kg-exposure": (
        "Exposure (KG edge sets)",
        "How a KG edge set's measured edges stand to the model's recorded graph use.\n"
        + EXPOSURE_CAVEAT,
    ),
    "validation-trainer": (
        "Validation (Trainer)",
        "The Trainer's own figure on the val samples, recorded in the checkpoint.\n"
        "• Base: LP accuracy.\n"
        "• Task model: pooled MRR.\n"
        "Training selects a checkpoint by it, so it is not an independent test; recorded "
        "test results are kept apart from it.",
    ),
}

#: **The validation figures a checkpoint may record, as every page names them**
#: (K6 decision 8: the Models table, Model Facts and Training's checkpoint
#: options read this one map).
VALIDATION_LABELS = {
    "val_pooled_mrr": "Validation pooled MRR (Trainer)",
    "val_lp_accuracy": "Validation LP accuracy (Trainer)",
}
_VALIDATION_NOTE = "selection metric on val, not an independent test"
#: A recorded selection's direction (``callbacks.model_checkpoint.mode``), in
#: words: the Trainer ranks a run's saves by its monitor, highest first for
#: ``max``.
SELECTION_DIRECTIONS = {"max": "higher is better", "min": "lower is better"}

_CHECK = ('<svg width="12" height="12" viewBox="0 0 16 16" aria-hidden="true" focusable="false">'
          '<path d="M3.2 8.6l3 3 6.6-7.4" fill="none" stroke="currentColor" stroke-width="2.2" '
          'stroke-linecap="round" stroke-linejoin="round"/></svg>')
#: The dashboard's check mark (an inline svg that stays on its line), for a
#: page's own "✓ …" (M5-3 review ARCH-4: pages used the private name).
CHECK = _CHECK

# ---------------------------------------------------------------------------
# State words and status marks (the WebUI skill's §6)
# ---------------------------------------------------------------------------
#: **The missing states' own words** (the skill's §6): each says something
#: else, so none is ever merged into another, into "Off" or into "—". Off: a
#: setting or a control switched off. Not loaded: nothing is served. Not
#: applicable: the fact has no meaning for this item. Unknown: the fact was
#: asked for and could not be read. Cannot be verified: it was read, but what
#: it would be checked against is missing or unreadable. Not recorded: the
#: record holds none. A sentence says them in lower case.
OFF = "Off"
NOT_LOADED = "Not loaded"
NOT_APPLICABLE = "Not applicable"
UNKNOWN = "Unknown"
CANNOT_BE_VERIFIED = "Cannot be verified"
NOT_RECORDED = "Not recorded"
#: **A dash's one meaning: no value here**, where nothing else is meant (a
#: row of a kind the column does not describe, a cell with nothing in it).
NO_VALUE = "—"

#: **Each state's shape and ``--shep-status-*`` colour** (the skill's §6
#: table): OK filled, a missing state hollow and grey, a condition to watch a
#: triangle in amber, an error or a refusal a cross in red, Running the
#: spinner (its alone), and a busy job asked to stop the Stop button's own
#: square in blue.
STATUS_MARKS = {
    "ok": ("●", "ok"),
    "neutral": ("○", "neutral"),
    "warn": ("▲", "warn"),
    "error": ("✕", "error"),
    "running": (None, "info"),
    "stopping": ("■", "info"),
}
#: The tag colour of each status colour, for a mark drawn as a chip.
_STATUS_TAG = {"ok": "ok", "neutral": "grey", "warn": "amber", "error": "red", "info": "accent"}


def status_mark(kind: str, words: Any, *, words_html: bool = False,
                as_chip: bool = False) -> str:
    """**The one status indicator** (the skill's §6): the state's colour, its
    shape and its word together, so the colour never carries the meaning
    alone. `kind` is a `STATUS_MARKS` row; `words` the state's word or the
    condition's short sentence, always shown beside the shape, escaped (with
    `words_html`, HTML these renderers built and escaped, a sentence with a
    bold part or a code value). The shape is hidden from a screen reader,
    which reads the word. Drawn as coloured text (``shep-st``), or with
    `as_chip` as a tag: a state to watch, or one that stands apart in a row of
    tags. No page shows a mark without its word."""
    shape, token = STATUS_MARKS[kind]
    mark = ('<span class="shep-spinner" aria-hidden="true"></span>' if shape is None else
            f'<span class="shep-st-shape" aria-hidden="true">{shape}</span>')
    text = str(words) if words_html else e(words)
    classes = (f"shep-tag shep-tag-{_STATUS_TAG[token]} shep-st-chip" if as_chip
               else f"shep-st shep-st-{token}")
    return f'<span class="{classes}">{mark}<span class="shep-st-w">{text}</span></span>'

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_RUN_TIME = re.compile(r"^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})")
#: A report store run id (`cohort_registry.create_report_dir`, its
#: `RUN_ID_PATTERN`): its UTC time, then the checkpoint's 12 hex digits and the
#: run's own 8.
_REPORT_RUN_ID = re.compile(
    r"^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z-[0-9a-f]{12}-[0-9a-f]{8}$")


def e(value: Any) -> str:
    """Escaped text for HTML content or a quoted attribute."""
    return html.escape("" if value is None else str(value), quote=True)


def ident_html(value: Any, *, title: Any = None) -> str:
    """An identifier (a role code, a run, a checkpoint file name) in code style,
    kept whole: never broken inside its name ("research_selection" is never
    "research_" / "selection"), and where it does not fit it ends in an
    ellipsis with its whole value as its title (``.shep-ident``,
    `src/webui/style.py`). `title` names the whole identifier when `value` is
    a short form of it (a run's name part, its whole "arch/run" id)."""
    text = e(value)
    return f'<code class="shep-ident" title="{e(title) if title is not None else text}">{text}</code>'


def word_groups(*groups: str) -> str:
    """Escaped words kept in groups: a line breaks between groups, so no word
    of a group is left alone on a line ("…: deployment mode not set" / "to
    M1"); a group wider than its line still wraps inside itself (``.shep-wg``,
    `src/webui/style.py`)."""
    return " ".join(f'<span class="shep-wg">{e(group)}</span>' for group in groups if group)


def is_digest(value: Any) -> bool:
    return isinstance(value, str) and _HEX64.match(value) is not None


def short_hex(digest: Any, digits: int = SHORT_HEX) -> str:
    """The first `digits` hex digits of a recorded digest, or "not recorded"
    when the record holds none (the skill's §6: never "—", which means "no
    value here"). A caller with nothing to name at all (no model served)
    says so in its own words before it asks."""
    return digest[:digits] if is_digest(digest) else NOT_RECORDED.lower()


def distinguishing_digits(digests: Iterable[Optional[str]], base: int = SHORT_HEX) -> int:
    """How many leading hex digits tell every distinct digest here apart: `base`,
    or more when two distinct ones share their first `base`."""
    distinct = sorted({d for d in digests if is_digest(d)})
    digits = base
    while digits < 64:
        prefixes = {d[:digits] for d in distinct}
        if len(prefixes) == len(distinct):
            return digits
        digits += 4
    return 64


def chip(text: str, kind: str = "neutral", *, title: Optional[str] = None) -> str:
    """A small state chip (amber synthetic, green verified, grey, red, accent)."""
    attr = f' title="{e(title)}"' if title else ""
    return f'<span class="shep-tag shep-tag-{kind}"{attr}>{e(text)}</span>'


def warn_chip(text: str) -> str:
    """An amber chip with ▲ before its words: a condition to watch (the skill's §6)."""
    return status_mark("warn", text, as_chip=True)


def resources_html(resources: Mapping[str, Any]) -> str:
    """The resources reading (`TrainingManager.get_system_resources`, what
    ``GET /api/v1/system/resources`` answers) as one line of 14 px text: each
    GPU's use, memory and temperature (one shared "unified memory" figure on
    a unified-memory machine, never the same bytes twice), and the system
    memory. Training's resources line, and Diagnosis's reading after a stop
    of inference."""
    parts: List[str] = []
    unified = bool(resources.get("unified_memory"))
    gpu = resources.get("gpu") or {}
    if gpu.get("available"):
        for dev in gpu.get("devices") or []:
            idx = dev.get("index", 0)
            util = dev.get("utilization_percent")
            parts.append(f"GPU {idx} {'N/A' if util is None else f'{util:.0f} %'}")
            if not unified:
                used, total = dev.get("memory_used_mb"), dev.get("memory_total_mb")
                parts.append(f"VRAM {used / 1024:.1f} / {total / 1024:.1f} GB"
                             if used is not None and total else "VRAM N/A")
            temp = dev.get("temperature_c")
            parts.append("N/A °C" if temp is None else f"{temp:.0f} °C")
            if dev.get("name"):
                parts.append(str(dev["name"]))
    else:
        parts.append("GPU: not available")
    ram = resources.get("ram") or {}
    label = "unified memory" if unified else "memory"
    if "error" in ram:
        parts.append(f"{label}: {ram.get('error')}")
    elif ram:
        total = ram.get("total_gb", 0) or 0
        used = (total - ram.get("available_gb", 0)) if unified and ram.get(
            "available_gb") is not None else ram.get("used_gb", 0)
        parts.append(f"{label} {used:.1f} / {total:.1f} GB"
                     + (" (shared CPU + GPU pool)" if unified else ""))
    items = '<span class="shep-sep" aria-hidden="true"> · </span>'.join(
        f'<span>{e(p)}</span>' for p in parts)
    return f'<div class="shep-rv-res"><b>Resources</b> {items}</div>'


def fingerprint_html(label: str, digest: Any, *, digits: int = SHORT_HEX, copy: bool = True,
                     help_key: Optional[str] = None) -> str:
    """"<label> <12 hex>" in a line of text, the label muted before the one
    digest value (`fingerprint_value`: bold monospace, the full value in
    ``title`` and on Copy, "not recorded" when there is none)."""
    help_html = help_button(help_key, label[:1].upper() + label[1:]) if help_key else ""
    value = fingerprint_value(digest, what=label, digits=digits, copy=copy)
    return (f'<span class="shep-fp"><span class="shep-fpk">{e(label)}</span> '
            f'{value}{help_html}</span>')


# ---------------------------------------------------------------------------
# Options (the skill's §5 and §8)
# ---------------------------------------------------------------------------
#: **The longest option a select offers** (the skill's §8): every select's
#: options are at most this long, unique, the distinguishing token first, with
#: at most one " · ".
OPTION_CHARS = 40


#: The fewest characters an option's distinguishing token keeps before what
#: follows it is cut too (`fit_option`): two of its start and two of its end
#: around the ellipsis.
_WHAT_ROOM = 5


def middle_cut(text: str, room: int) -> str:
    """`text` in at most `room` characters: whole where it fits, else cut in
    its middle with an ellipsis, so its start and its end stay (where a copy
    or a version differs: "…copy_b.pt", "…67-v1.pt"). Never longer than
    `text`."""
    if len(text) <= room:
        return text
    keep = max(room, 1) - 1
    head = keep // 2
    return text[:head] + "…" + text[len(text) - (keep - head):]


def fit_option(what: str, rest: Optional[str], mark: str = "") -> str:
    """"<what><mark> · <rest>" in at most `OPTION_CHARS` characters with one
    separator: `what` (the distinguishing token: a name, a file's name) is
    shortened in its middle (`middle_cut`); `rest` stays whole while `what`
    keeps at least `_WHAT_ROOM` characters (or all of its own, if fewer), and
    past that is cut in its middle too, so neither is ever the only part cut
    and the option never runs past `OPTION_CHARS`."""
    room = OPTION_CHARS - len(mark)
    if rest:
        room -= 3
        rest = middle_cut(rest, room - min(len(what), _WHAT_ROOM))
        room -= len(rest)
    what = middle_cut(what, room) + mark
    return f"{what} · {rest}" if rest else what


def fit_options(entries: Sequence[Tuple[str, Optional[str], Any]]) -> List[Tuple[str, Any]]:
    """A select's options from ``(what, rest, value)`` entries, in their order:
    each `fit_option`, and those still alike numbered in that order, "(2)",
    the last resort for what nothing else tells apart. Unique, at most
    `OPTION_CHARS` characters and one separator; **each value is its entry's
    own, whatever the words** (Gradio's ``(label, value)``: the skill's §5, a
    label never changes what is chosen)."""
    labels = [fit_option(what, rest) for what, rest, _value in entries]
    seen, nth = Counter(labels), Counter()
    choices = []
    for (what, rest, value), label in zip(entries, labels):
        if seen[label] > 1:
            nth[label] += 1
            label = fit_option(what, rest, f" ({nth[label]})")
        choices.append((label, value))
    return choices


# ---------------------------------------------------------------------------
# The dataset identity
# ---------------------------------------------------------------------------
def _get(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def clock_utc(moment: Optional[datetime], *, seconds: bool = True) -> Optional[str]:
    """"15:50:35 UTC" (or "15:50 UTC"): **the one form of every clock time the
    dashboard shows** (M5-3 review UI-5: Diagnosis showed the server's local
    time unlabeled beside "UTC" times elsewhere). A time with no zone is this
    server's local time (`datetime.now()`, a naive ISO stamp) and is converted;
    None for no time. Exports keep the result's own stamp."""
    if moment is None:
        return None
    return moment.astimezone(timezone.utc).strftime("%H:%M:%S UTC" if seconds else "%H:%M UTC")


def built_text(build: Optional[Mapping[str, Any]]) -> Optional[str]:
    """"built 2026-10-15 09:41 UTC" from a recorded build, else None."""
    if not isinstance(build, Mapping) or build.get("state") != "recorded":
        return None
    raw = build.get("built_at")
    if not isinstance(raw, str):
        return None
    try:
        moment = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        return None
    return "built " + moment.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def sources_text(releases: Any) -> str:
    """The dataset record's sources as each declares its release: "HPO 2026-09-01 ·
    MONDO release not declared"; "none (synthetic)" for a synthetic build."""
    parts = []
    for source in releases or ():
        if not isinstance(source, Mapping):
            continue
        role = str(source.get("role") or "source").upper()
        version = source.get("declared_version")
        parts.append(f"{role} {version}" if version else f"{role} release not declared")
    return " · ".join(parts)


def origin_of(item: Any) -> Optional[str]:
    record = _get(item, "dataset_record") or {}
    origin = record.get("origin") if isinstance(record, Mapping) else None
    return origin if origin in ("files", "synthetic") else None


def record_verified(item: Any) -> bool:
    record = _get(item, "dataset_record") or {}
    return isinstance(record, Mapping) and record.get("binding") == "verified"


#: **A dataset record's binding state in words** (`WorkspaceItem.dataset_record`
#: ``binding``; `kg.artifacts.verify_dataset_record`), each its own (the
#: skill's §6): the manifest binds no record (not recorded), the record it
#: binds cannot be read or checked (cannot be verified), the record
#: contradicts what the manifest binds, or the manifest was not read, so the
#: record's state is unknown (the origin and fingerprint cells' word for the
#: same cause).
RECORD_WORDS = {
    "unrecorded": "record " + NOT_RECORDED.lower(),
    "unverifiable": "record " + CANNOT_BE_VERIFIED.lower(),
    "mismatch": "record does not match",
    None: "record " + UNKNOWN.lower(),
}


def record_words(item: Any) -> Tuple[str, str]:
    """``(kind, words)`` of a workspace's dataset record (`RECORD_WORDS`): its
    `status_mark` row and its words. Verified is OK; a record the manifest
    contradicts is an error (it is refused before a run); a record absent,
    not checkable or unknown is a missing state, each in its own words."""
    record = _get(item, "dataset_record") or {}
    binding = record.get("binding") if isinstance(record, Mapping) else None
    if binding == "verified":
        return "ok", "record verified"
    if binding == "mismatch":
        return "error", RECORD_WORDS["mismatch"]
    return "neutral", RECORD_WORDS.get(binding, RECORD_WORDS[None])


def record_chip(item: Any) -> str:
    """"✓ record verified" (the dataset record checked), else the record's
    own state as a chip (`record_words`): "record not recorded" apart from
    "record cannot be verified", "record does not match" and "record
    unknown"."""
    kind, words = record_words(item)
    if kind == "ok":
        return f'<span class="shep-tag shep-tag-ok">{_CHECK} {e(words)}</span>'
    return status_mark(kind, words, as_chip=True)


def count_text(value: Any) -> str:
    """A recorded count with thousands separators, or "not recorded" where the
    record holds none (never "—", the skill's §6)."""
    return (f"{value:,}" if isinstance(value, int) and not isinstance(value, bool)
            else NOT_RECORDED.lower())


def dataset_details_html(row: Mapping[str, Any], facts: Optional[Mapping[str, Any]], *,
                         problem: Optional[str] = None) -> str:
    """**The one body of a workspace's details** (Research › Datasets' panel and
    Training's "Details ›" popover, each with its own header; M5-3 review
    ARCH-4): the full dataset fingerprint with Copy, what the manifest binds (12
    hex each, `workspaces.manifest_facts`), the split's seeds, the samples, the
    sources, the build time, the path relative to the server's workspace root,
    and why the workspace is not trainable. `problem` says why the manifest was
    not read (without a path); every value is escaped."""
    name = str(row.get("name") or "")
    parts = []
    fingerprint = row.get("dataset_fingerprint")
    if is_digest(fingerprint):
        parts.append(
            '<div class="shep-digest"><span class="shep-dl">Dataset fingerprint · SHA-256 of '
            f'split_manifest.json</span><code class="shep-full">{e(fingerprint)}</code>'
            f'{copy_button(fingerprint, "the full dataset fingerprint of " + name)}</div>')
    rows_kv = []
    if facts is not None:
        bound = summary_list([(label, f"<code>{e(short_hex(digest))}</code>")
                              for label, digest in facts["bound"]], classes="shep-kv-2")
        parts.append('<div class="shep-card"><div class="shep-card-hd">What the manifest binds '
                     '<span class="shep-muted">sha256, first 12 · full values in the manifest'
                     f'</span></div>{bound}</div>')
        split = []
        if facts.get("allocation_seed") is not None:
            split.append(f"allocation seed {facts['allocation_seed']}")
        if facts.get("feature_seed") is not None:
            split.append(f"feature seed {facts['feature_seed']}")
        samples = f"train {count_text(row.get('train_samples'))} · val {count_text(row.get('val_samples'))}"
        if facts.get("train_diseases") is not None:
            samples += (f" · diseases (train / val) {count_text(facts['train_diseases'])} / "
                        f"{count_text(facts.get('val_diseases'))}")
        rows_kv = [("Split", e(" · ".join(split) or "not recorded")), ("Samples", e(samples))]
    else:
        why = problem or row.get("not_trainable") or "it is not readable"
        parts.append(f'<div class="shep-reason">The manifest was not read: {e(why)}</div>')
    sources = sources_text(row.get("declared_releases"))
    built = built_text(row.get("build"))
    rows_kv += [
        ("Sources", e(sources) if sources else
         ("none (synthetic)" if origin_of(row) == "synthetic" else '<span class="shep-muted">—</span>')),
        ("Built", e(built.replace("built ", "", 1)) if built else "not recorded"),
        ("Path", f'<code>{e(name)}</code> <span class="shep-muted">(in the server’s '
                 'workspace root)</span>'),
    ]
    if not row.get("trainable"):
        rows_kv.append(("Not trainable", e(row.get("not_trainable") or "not trainable")))
    parts.append(summary_list(rows_kv))
    return "".join(parts)


def dataset_identity_html(item: Any, *, digits: int = SHORT_HEX, name: bool = True,
                          compact: bool = False) -> str:
    """One workspace's identity: name; "built … UTC" or "build time not recorded ·
    sources …"; the dataset fingerprint; amber "synthetic"; the record's state
    (`record_chip`: "record verified", or the record's own missing or
    contradicted state); "not trainable: <why>" when it is not."""
    parts = []
    if name:
        parts.append(f'<span class="shep-ds-name">{e(_get(item, "name"))}</span>')
    built = built_text(_get(item, "build"))
    if built:
        parts.append(f'<span class="shep-ds-built">{e(built)}</span>')
    else:
        sources = sources_text(_get(item, "declared_releases"))
        text = "build time not recorded" + (f" · sources {sources}" if sources else "")
        parts.append(f'<span class="shep-ds-built shep-muted">{e(text)}</span>')
    parts.append(fingerprint_html("dataset fingerprint", _get(item, "dataset_fingerprint"),
                                  digits=digits, help_key="dataset-fingerprint"))
    marks = []
    if origin_of(item) == "synthetic":
        marks.append(chip("synthetic", "amber"))
    if record_verified(item) or _get(item, "dataset_fingerprint"):
        marks.append(record_chip(item))
    if not _get(item, "trainable", False):
        why = str(_get(item, "not_trainable") or "not trainable")
        marks.append(chip(f"not trainable: {not_trainable_short(item)}", "grey", title=why))
    if marks:
        parts.append('<span class="shep-ds-marks">' + "".join(marks) + "</span>")
    sep = " " if compact else '<span class="shep-sep" aria-hidden="true"> · </span>'
    return f'<span class="shep-ds">{sep.join(parts)}</span>'


def short_reason(text: Any, limit: int = 80) -> str:
    """A reason shortened for a cell, on one line; the full text goes in
    ``title`` or a details panel."""
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[:limit - 1] + "…"


#: Why a workspace without a readable manifest is not trainable, in short words:
#: by whether its ``split_manifest.json`` is absent (an item's ``manifest``,
#: which the page reading the list adds) or present but not readable.
NO_SPLIT_MANIFEST = "no split manifest"
NO_READABLE_SPLIT_MANIFEST = "no readable split manifest"


def not_trainable_short(item: Any, limit: int = 80) -> str:
    """"not trainable: <this>" in short words: "no split manifest" when the
    file is absent, "no readable split manifest" when it is there but not
    read (or when the page did not check which), and the list's own reason,
    shortened, for a workspace whose manifest was not reached at all
    (``manifest`` "not_checked": a name that resolves outside the root) or
    whose manifest was read."""
    state = _get(item, "manifest")
    if not is_digest(_get(item, "dataset_fingerprint")) and state != "not_checked":
        return NO_SPLIT_MANIFEST if state == "absent" else NO_READABLE_SPLIT_MANIFEST
    return short_reason(_get(item, "not_trainable") or "not trainable", limit)


def _workspace_rest(item: Any, digits: int = 0) -> Optional[str]:
    """What a workspace's option says after its name (`workspace_choices`): a
    workspace without a readable manifest in a few words, "no split manifest",
    "no readable split manifest" or "not trainable", read before choosing
    (its whole reason is beside the select); given `digits` (two names that
    would read alike), "synthetic" where it applies and that many of its
    fingerprint's hex digits; else nothing."""
    fingerprint = _get(item, "dataset_fingerprint")
    if not is_digest(fingerprint):
        short = not_trainable_short(item)
        return short if short in (NO_SPLIT_MANIFEST, NO_READABLE_SPLIT_MANIFEST) else "not trainable"
    if not digits:
        return None
    return " ".join([*(["synthetic"] if origin_of(item) == "synthetic" else []),
                     fingerprint[:digits]])


def workspace_choices(workspaces: Sequence[Any]) -> List[Tuple[str, str]]:
    """**The options of a workspace select, one way on every page** (Diagnosis'
    drawer, Models, Test, Fine-tune and Training's run bar): ``(label,
    name)``, the name first, at most `OPTION_CHARS` characters with one
    separator (the skill's §8; a long name shortened in its middle). A
    trainable workspace is its name: its identity (fingerprint, synthetic,
    record) is the chosen workspace's details, said once (the GB10 WebUI
    recheck: said twice side by side): Training's dataset chip, the identity
    line under Fine-tune's, Models' and Test's selects
    (`dataset_identity_html`); the Diagnosis drawer has no such line, and each
    of its checkpoint rows says what it was trained on (`trained_on_html`).
    One that cannot be trained adds why in a few words ("name · no split
    manifest"), read before choosing. Only where two names would read alike
    do they add the synthetic flag where it applies and their fingerprints'
    distinguishing digits (`distinguishing_digits`), then a number where
    nothing recorded tells them apart. The value is the listed name, whatever
    the words."""
    names = [str(_get(w, "name")) for w in workspaces]
    labels = [fit_option(name, _workspace_rest(w)) for name, w in zip(names, workspaces)]
    seen = Counter(labels)
    digits = distinguishing_digits(_get(w, "dataset_fingerprint")
                                   for w, label in zip(workspaces, labels) if seen[label] > 1)
    return fit_options([(name, _workspace_rest(w, digits if seen[label] > 1 else 0), name)
                        for name, w, label in zip(names, workspaces, labels)])


# ---------------------------------------------------------------------------
# The model row
# ---------------------------------------------------------------------------
class RecordedTests(NamedTuple):
    """What a checkpoint's ledger records for its weights, as a state (never a
    number of a result): `state` is ``records``, ``none`` (the ledger holds no
    record for these weights), ``no_ledger`` or ``not_read``; `count` the
    records for these weights that read as results; `contested` how many of
    them are in a condition group whose runs differ in their cohort version's
    report store, and `unchecked` how many whose comparison there is
    incomplete, counted apart, so a known difference is never hidden by a
    comparison that is incomplete (`contested` None: no comparison was read);
    `unreadable` how many records for these weights cannot be read as results
    (K8-6e.1), never counted among them. Descriptions only: they hide, order
    or refuse nothing (policy rev 16)."""

    state: str
    count: int = 0
    contested: Optional[int] = None
    problem: Optional[str] = None
    unchecked: int = 0
    unreadable: int = 0


def recorded_tests_text(recorded: Optional[RecordedTests]) -> str:
    """"Recorded results: 3 · 1 not readable; runs differ for 1; comparison
    incomplete for 1": states, never a number of a result; a record that
    cannot be read is never counted as one (K8-6e.1) and a record whose
    comparison is incomplete is never shown as agreeing. "none in ledger"
    only when the ledger holds neither. One " · " (the skill's §5)."""
    if recorded is None:
        return "Recorded results: not read"
    if recorded.state == "no_ledger":
        return "Recorded results: no ledger"
    if recorded.state == "not_read":
        return "Recorded results: ledger not readable"
    if recorded.state == "none" or (recorded.count == 0 and not recorded.unreadable):
        return "Recorded results: none in ledger"
    contested = recorded.contested or 0
    unchecked = recorded.count if recorded.contested is None else recorded.unchecked
    notes = []
    if recorded.unreadable:
        notes.append(f"{recorded.unreadable} not readable")
    if contested:
        notes.append(f"runs differ for {contested}")
    if unchecked:
        notes.append("comparison incomplete" if unchecked >= recorded.count
                     else f"comparison incomplete for {unchecked}")
    return f"Recorded results: {recorded.count}" + (f" · {'; '.join(notes)}" if notes else "")


#: Run layout keys whose name is not their upper case (a model bundle's
#: encoder kind, as the GATv2 paper writes it).
_ARCHITECTURE_NAMES = {"gatv2": "GATv2"}


def architecture_label(architecture: Any) -> str:
    """A run directory's layout name as a person reads it: "GATv2" for
    ``gatv2``, any other name upper-cased ("HGT", "GAT", a name an operator
    gave), "MODEL" when none. A location, never the model's architecture,
    which its record names."""
    key = str(architecture or "")
    return _ARCHITECTURE_NAMES.get(key, key.upper()) or "MODEL"


def run_label(architecture: Any, run: Any) -> str:
    """"GATv2 · run 2026-10-17 09:15" from a run directory named by its start
    time; the run's own name when it is not."""
    started = run_started(run)
    return f"{architecture_label(architecture)} · run {started if started else run}"


def run_label_html(architecture: Any, run: Any) -> str:
    """`run_label` as HTML for a narrow cell: the run's start time ("run
    2026-10-17 09:15") in one no-wrap span, so a line breaks after "GAT ·" and
    never inside the date; a run not named by its start time is its escaped
    name, free to wrap."""
    started = run_started(run)
    if not started:
        return e(run_label(architecture, run))
    return (f'{e(architecture_label(architecture))} · '
            f'<span class="shep-nowrap">run {e(started)}</span>')


def run_started(run: Any, *, seconds: bool = False) -> Optional[str]:
    """"2026-10-17 09:15" (with `seconds`, "2026-10-17 09:15:00") from a run
    directory named by its start time, else None."""
    match = _RUN_TIME.match(str(run or ""))
    if not match:
        return None
    y, mo, d, h, mi, sec = match.groups()
    return f"{y}-{mo}-{d} {h}:{mi}{':' + sec if seconds else ''}"


def run_suffix(run: Any) -> Optional[str]:
    """"run_3fa9c1" from a run directory named "20261017-091500_run_3fa9c1"
    (what follows its start time); None when it is not named by its start time
    (`run_label` then names it whole)."""
    text = str(run or "")
    match = _RUN_TIME.match(text)
    if not match:
        return None
    rest = text[match.end():].lstrip("_-")
    return rest or None


def run_name(run: Any) -> Optional[str]:
    """"smoke-ft-disease" from a run directory named
    "20261028-140013_smoke-ft-disease_0e8106": its name, between its start
    time and its 6-hex id; None when it is not named by its start time."""
    rest = run_suffix(run)
    return re.sub(r"_[0-9a-f]{6}$", "", rest) if rest else None


def run_time(run_id: Any, *, seconds: bool = False) -> Optional[str]:
    """A run's time from its run id, as the pages name a run: "2026-10-17
    13:40 UTC" (with `seconds`, "2026-10-17 13:40:07 UTC") from a report store
    run id (`create_report_dir` names it by its UTC time), else None."""
    match = _REPORT_RUN_ID.match(str(run_id or ""))
    if not match:
        return None
    y, mo, d, h, mi, s = match.groups()
    return f"{y}-{mo}-{d} {h}:{mi}{':' + s if seconds else ''} UTC"


def distinct_names(items: Mapping[Hashable, Any], name_at: Callable[[Any, int], str], *,
                   top: int = 2) -> Dict[Hashable, str]:
    """Names that read alike, told apart: each item's name, by its key, at
    level 0 (``name_at(item, level)``), a level higher only for the items
    whose names read alike, until no two do or each of them is at `top` (runs
    named by their time to the minute, then to the second, then with the run
    id's own last part: Models › Results' `row_names`). Names still alike at
    `top` are the caller's to tell apart."""
    levels = dict.fromkeys(items, 0)
    while True:
        names = {key: name_at(item, levels[key]) for key, item in items.items()}
        seen = Counter(names.values())
        clashing = [key for key, name in names.items() if seen[name] > 1 and levels[key] < top]
        if not clashing:
            return names
        for key in clashing:
            levels[key] += 1


def record_kind_text(item: Any) -> str:
    """**What a listed checkpoint is, in the one word every page leads with**
    (K6 decision 6), from its model record (`listed_record`): "Base", the task's
    name (`caveats.TASK_NAMES`: "Disease", "Gene", "Patients-like-me"),
    "Record not readable" for a record this build refused (a file without a
    model spec among them, its reason the record's), or "Record not read" for
    a file whose record was never read (skill §6: missing states keep their
    own words)."""
    record = _get(item, "record")
    if record == "model":
        if _get(item, "learning_stage") == "base":
            return "Base"
        task = _get(item, "task")
        if task in TASK_NAMES:
            return TASK_NAMES[task]
    elif record is None:
        return "Record not read"
    return "Record not readable"


#: What a task model whose graph backbone did not train is called after its
#: kind (`model_kind_text`; ops-wiring round, decision F1): never the base's.
FROZEN_BACKBONE_WORDS = "frozen graph backbone"


def freeze_words(item: Any) -> Optional[str]:
    """The words a listed model's freeze adds to its kind
    (`FROZEN_BACKBONE_WORDS`) when its record's weight transfer names exactly
    the graph side frozen (`recipe.FROZEN_PARTS["graph_side"]`: the input
    module, the encoder and the LP head; read by the continuation's own
    reading of that record, `run_plan.recorded_frozen_parts`), else None: a
    full fine-tune, a rules-1 child (every part trainable, "all"), a base and
    a record that names no transfer read as their kind alone. `item` is a
    listing item or a result's model (`results._model`), which carry the
    record's `weight_transfer` alike."""
    frozen = recorded_frozen_parts(_get(item, "weight_transfer"))
    if frozen is not None and frozen == tuple(sorted(FROZEN_PARTS["graph_side"])):
        return FROZEN_BACKBONE_WORDS
    return None


def model_kind_text(item: Any) -> str:
    """**What a listed model is, wherever a model or a result is described**:
    its kind in the one word every page leads with (`record_kind_text`), and
    for the frozen-backbone control "Disease · frozen graph backbone"
    (`freeze_words`): a task model whose graph side kept its base's weights,
    never the base's score. At most one separator and 40 characters. The one
    word alone stays the grouping key and an option's lead
    (`record_kind_text`)."""
    kind, words = record_kind_text(item), freeze_words(item)
    return f"{kind} · {words}" if words else kind


def configuration_text(item: Any) -> str:
    """A listed model's configuration (README §5, axis 1): its encoder and
    whether its spec is the reference recipe, "GATv2 · reference recipe" or
    "GATv2 · recipe changed in 3 fields" (`differences`: which fields is axis
    2's, shown where the full list fits). A record refused or never read says
    which."""
    record = _get(item, "record")
    if record is None:
        return "Not read"
    differences = _get(item, "differences")
    if record != "model" or differences is None:
        return "Not readable"
    if not differences:
        return f"{architecture_label(_get(item, 'encoder'))} · reference recipe"
    count = len(differences)
    return (f"{architecture_label(_get(item, 'encoder'))} · recipe changed in {count} "
            f"field{'s' if count != 1 else ''}")


def _whole(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def epoch_words(item: Any) -> str:
    """**How far a checkpoint trained, in the one wording of every page**
    (Training's and Fine-tune's options, the Models row and detail, Model
    Facts, the Diagnosis card, the Test options): "195 epochs completed" from
    the file's progress record (`completed_epochs`); a file that records no
    progress (an older format) says the epoch index it records, "recorded
    epoch 194", never a count made from it; "epochs not recorded" when it
    records neither (the skill's §6: never "—"). `item` is a listed
    checkpoint, or the served status's `checkpoint_meta`, which carry both
    fields alike."""
    completed, epoch = _get(item, "completed_epochs"), _get(item, "epoch")
    if _whole(completed):
        return f"{completed} epoch{'s' if completed != 1 else ''} completed"
    if _whole(epoch):
        return f"recorded epoch {epoch}"
    return f"epochs {NOT_RECORDED.lower()}"


def model_name_line(item: Any) -> str:
    """"GATv2 · run 2026-10-17 09:15 · 6 epochs completed" (`epoch_words`)."""
    return f"{run_label(_get(item, 'architecture'), _get(item, 'run'))} · {epoch_words(item)}"


# ---------------------------------------------------------------------------
# The validation figure and the selection (the skill's §8)
# ---------------------------------------------------------------------------
def validation_figure(item: Any) -> Optional[Tuple[str, float]]:
    """The Trainer's validation figure a listed checkpoint records, as
    ``(metric, value)`` (`CheckpointItem.validation`), or None when it
    records none: what `validation_text` and `validation_value` say, read
    the one way for every page's details (Model Facts, the Test page's model,
    Continue's chosen checkpoint, Fine-tune's base)."""
    validation = _get(item, "validation")
    if isinstance(validation, Mapping) and isinstance(validation.get("value"), (int, float)):
        return str(validation.get("metric")), float(validation["value"])
    return None


def _selection_label(monitor: Any) -> str:
    """A recorded monitor as the validation figures are named
    (`VALIDATION_LABELS`; any other by its recorded name, as `validation_cell`
    names it)."""
    return VALIDATION_LABELS.get(monitor, f"Validation {monitor} (Trainer)")


def metric_words(monitor: Any) -> str:
    """A recorded monitor in an option's few words: "LP accuracy", "pooled
    MRR" (its `VALIDATION_LABELS` name without "Validation" and "(Trainer)",
    so an LP figure is never called MRR), any other by its recorded name."""
    return _selection_label(monitor).removeprefix("Validation ").removesuffix(" (Trainer)")


def _selection(item: Any) -> Optional[Tuple[str, str]]:
    selection = _get(item, "selection")
    if not isinstance(selection, Mapping):
        return None
    monitor, mode = selection.get("monitor"), selection.get("mode")
    return (monitor, mode) if isinstance(monitor, str) and isinstance(mode, str) else None


def selection_words(item: Any) -> str:
    """**What a run's saves were ranked by, as the file records it**
    (`CheckpointItem.selection`: ``callbacks.model_checkpoint``'s monitor and
    mode): "Selection by validation LP accuracy (Trainer), higher is better"
    for a base, "… validation pooled MRR (Trainer) …" for a task model, "lower
    is better" for a ``min`` monitor; a validation figure, the Trainer's,
    never a test result. A file that records none (a weights-only file, an
    older format) says "Selection not recorded"; a direction this build does
    not know is "direction unknown"."""
    selection = _selection(item)
    if selection is None:
        return f"Selection {NOT_RECORDED.lower()}"
    monitor, mode = selection
    label = _selection_label(monitor)
    direction = SELECTION_DIRECTIONS.get(mode, f"direction {UNKNOWN.lower()}")
    return f"Selection by {label[:1].lower()}{label[1:]}, {direction}"


def best_saves(items: Mapping[Any, Any]) -> set:
    """**The keys of the best saves**, by the run's own recorded selection
    (the skill's §8): within each run (its `run_key`), among the files that
    record the same selection (`selection_words`) and a validation figure of
    that monitor (`validation`, `checkpoint_paths.ranking_score_detail`'s
    read), those whose figure is the best in the recorded direction; files
    that tie all are. Read from the list's metadata only, never from a file's
    name; a file that records no selection, no figure of its monitor or a
    direction this build does not know is none. `items` maps a key to its
    listed checkpoint."""
    groups: Dict[Tuple[Any, str, str], List[Tuple[Any, float]]] = {}
    for key, item in items.items():
        selection = _selection(item)
        validation = _get(item, "validation")
        if selection is None or selection[1] not in SELECTION_DIRECTIONS:
            continue
        if not isinstance(validation, Mapping) or validation.get("metric") != selection[0]:
            continue
        value = validation.get("value")
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)):
            continue
        run = _get(item, "run_key") or (_get(item, "architecture"), _get(item, "run"))
        groups.setdefault((run, *selection), []).append((key, float(value)))
    best = set()
    for (_run, _monitor, mode), scored in groups.items():
        values = [value for _key, value in scored]
        top = max(values) if mode == "max" else min(values)
        best.update(key for key, value in scored if value == top)
    return best


def trained_on(fingerprint: Any, workspaces: Sequence[Any]) -> Tuple[str, ...]:
    """The names of every listed workspace whose full dataset fingerprint equals
    `fingerprint`."""
    if not is_digest(fingerprint):
        return ()
    return tuple(str(_get(w, "name")) for w in workspaces
                 if _get(w, "dataset_fingerprint") == fingerprint)


def trained_on_html(fingerprint: Any, workspaces: Sequence[Any], *, digits: int = SHORT_HEX,
                    copy: bool = False) -> str:
    """"trained on <name> · dataset fingerprint …", or "trained on dataset
    fingerprint … · matches no listed workspace" (one wording, both presets)."""
    names = trained_on(fingerprint, workspaces)
    fp = fingerprint_html("dataset fingerprint", fingerprint, digits=digits, copy=copy)
    if not is_digest(fingerprint):
        return f'<span class="shep-trained">trained on: {fp}</span>'
    if names:
        listed = ", ".join(f"<b>{e(n)}</b>" for n in names)
        return f'<span class="shep-trained">trained on {listed} · {fp}</span>'
    return (f'<span class="shep-trained">trained on {fp} · '
            f'<span class="shep-amber-t">matches no listed workspace</span></span>')


def is_synthetic_training(fingerprint: Any, workspaces: Sequence[Any]) -> bool:
    return any(origin_of(w) == "synthetic" for w in workspaces
               if is_digest(fingerprint) and _get(w, "dataset_fingerprint") == fingerprint)


def model_row_html(item: Any, workspaces: Sequence[Any], *,
                   recorded: Optional[RecordedTests] = None, in_use: bool = False,
                   selected: bool = False, digits: int = SHORT_HEX, key: str = "",
                   in_use_label: str = "In use") -> str:
    """One checkpoint as a row of the shared model list: the name line, "model
    fingerprint" 12 hex, what it was trained on, the synthetic chip and the
    recorded tests' state. An unreadable checkpoint is greyed: "metadata not
    readable · cannot be loaded"."""
    problem = _get(item, "problem")
    unreadable = problem is not None or not is_digest(_get(item, "sha256"))
    classes = ["shep-row"]
    if selected:
        classes.append("shep-row-on")
    if unreadable:
        classes.append("shep-row-dis")
    tag = f' {chip(in_use_label, "accent")}' if in_use else ""
    head = f'<span class="shep-row-main">{e(model_name_line(item))}</span>{tag}'
    if unreadable:
        why = problem or "its digest was not read"
        body = (f'<span class="shep-row-sub">{e(_get(item, "filename"))} · '
                f'<span title="{e(why)}">metadata not readable · cannot be loaded</span></span>')
        return (f'<button type="button" class="{" ".join(classes)}" data-key="{e(key)}" '
                f'aria-disabled="true" aria-pressed="false">{head}{body}</button>')
    fingerprint = _get(item, "dataset_fingerprint")
    sub = [fingerprint_html("model fingerprint", _get(item, "sha256"), digits=digits, copy=False),
           trained_on_html(fingerprint, workspaces, digits=digits)]
    if is_synthetic_training(fingerprint, workspaces):
        sub.append(chip("synthetic", "amber"))
    tests = f'<span class="shep-row-sub">{e(recorded_tests_text(recorded))}</span>'
    return (f'<button type="button" class="{" ".join(classes)}" data-key="{e(key)}" '
            f'aria-pressed="{"true" if selected else "false"}">{head}'
            f'<span class="shep-row-sub">{" · ".join(sub)}</span>{tests}</button>')


def recorded_tests_cell(recorded: Optional[RecordedTests]) -> str:
    """The recorded tests' state for a table cell (`recorded_tests_text`
    without its label): never a number of a result."""
    return recorded_tests_text(recorded).replace("Recorded results: ", "", 1)


def validation_cell(validation: Any, *, read: bool = True) -> str:
    """A checkpoint's own validation figure for the "Validation (Trainer)"
    column, escaped: its value over the figure's name (`VALIDATION_LABELS`),
    since a base, a task model and a file without a model spec each record
    their own figure. With none, why, as `validation_text` says it: "Not
    recorded" for a record read that holds none, "Unknown" for a record this
    build refuses or a file it did not read (`record_read`), which may well
    record one."""
    if not isinstance(validation, Mapping) or not isinstance(validation.get("value"), (int, float)):
        return status_mark("neutral", NOT_RECORDED if read else UNKNOWN)
    metric, value = str(validation.get("metric")), float(validation["value"])
    label = VALIDATION_LABELS.get(metric, f"Validation {metric} (Trainer)")
    return f'{value:.4f}<div class="shep-sub">{e(label)}</div>'


#: The research table's fixed columns (Models): an extra column a page fills
#: goes before the recorded tests.
MODEL_TABLE_COLUMNS = ("Model", "Fingerprints: model · dataset trained on",
                       "Validation (Trainer)", "Recorded results")


def model_table_row_html(item: Any, workspaces: Sequence[Any], *,
                         recorded: Optional[RecordedTests] = None, in_use: bool = False,
                         selected: bool = False, digits: int = SHORT_HEX, key: str = "",
                         in_use_label: str = "In use", extra: Optional[str] = None) -> str:
    """One checkpoint as a row of the research table (Models): the name line
    and run, "model <12 hex>" over "dataset <listed workspace> · <12 hex>" (or
    "· matches no listed workspace"), the synthetic chip, the Trainer's
    validation figure, an optional cell the page fills (`extra`, HTML it built
    with these renderers) and the recorded tests' state. An unreadable
    checkpoint is greyed: "metadata not readable · cannot be loaded"."""
    problem = _get(item, "problem")
    unreadable = problem is not None or not is_digest(_get(item, "sha256"))
    classes = [c for c, on in (("shep-on", selected), ("shep-row-dis", unreadable)) if on]
    tag = f' {chip(in_use_label, "accent")}' if in_use else ""
    suffix = run_suffix(_get(item, "run"))
    name = run_label_html(_get(item, "architecture"), _get(item, "run"))
    model = (f'<div class="shep-w600">{name}</div><div class="shep-sub">{e(epoch_words(item))}'
             f'{" · " + e(suffix) if suffix else ""}</div>{tag}')
    # A plain table's selected row is `aria-current`, never `aria-selected`
    # (the skill's §8).
    attrs = (f'class="{" ".join(classes)}" data-key="{e(key)}" tabindex="0"'
             + (' aria-current="true"' if selected else "")
             + (' aria-disabled="true"' if unreadable else ""))
    extra_cell = "" if extra is None else f"<td>{extra}</td>"
    if unreadable:
        why = problem or "its digest was not read"
        return (f'<tr {attrs}><td>{model}</td><td colspan="2"><span class="shep-sub">'
                f'{e(_get(item, "filename"))} · <span title="{e(why)}">metadata not readable · '
                f'cannot be loaded</span></span></td>{extra_cell}<td>—</td></tr>')
    fingerprint = _get(item, "dataset_fingerprint")
    names = trained_on(fingerprint, workspaces)
    if not is_digest(fingerprint):
        dataset = 'dataset <span class="shep-muted">not recorded</span>'
    elif names:
        dataset = (f'dataset {", ".join(f"<b>{e(n)}</b>" for n in names)} · '
                   f'<code class="shep-fpv" title="{e(fingerprint)}">{e(fingerprint[:digits])}</code>')
    else:
        dataset = (f'dataset <code class="shep-fpv" title="{e(fingerprint)}">'
                   f'{e(fingerprint[:digits])}</code> · '
                   '<span class="shep-amber-t">matches no listed workspace</span>')
    if is_synthetic_training(fingerprint, workspaces):
        dataset += " " + chip("synthetic", "amber")
    sha = _get(item, "sha256")
    fingerprints = (f'<div>model <code class="shep-fpv" title="{e(sha)}">{e(sha[:digits])}</code>'
                    f'</div><div class="shep-sub">{dataset}</div>')
    return (f'<tr {attrs}><td>{model}</td><td>{fingerprints}</td>'
            f'<td class="shep-num">{validation_cell(_get(item, "validation"), read=record_read(item))}'
            f'</td>{extra_cell}'
            f'<td>{e(recorded_tests_cell(recorded))}</td></tr>')


# ---------------------------------------------------------------------------
# Model Facts
# ---------------------------------------------------------------------------
class ModelFacts(NamedTuple):
    """What Model Facts shows for one model, from what the server listed.
    `task` is its record's kind (`record_kind_text`) and `configuration` its
    encoder and relation to the reference recipe (`configuration_text`)."""

    name_line: str
    run: Optional[str]
    checkpoint_file: Optional[str]
    workspace: Optional[str]
    model_fingerprint: Optional[str]
    trained_on_fingerprint: Optional[str]
    trained_on_names: Tuple[str, ...]
    synthetic: bool
    sources: str
    validation: Optional[Tuple[str, float]]
    recorded: Optional[RecordedTests]
    warnings: Tuple[Tuple[str, str], ...]
    #: What serving reads, as ``(label, words)`` rows of plain text, each a
    #: row of Data's summary list (the skill's §5: a fact a row, never a chain
    #: of " · "): the workspace it reads and how its dataset stands to this
    #: file's (`facts_from_checkpoint`), and on the Diagnosis card the graph,
    #: the shortest paths and the dataset record it serves.
    serving_rows: Tuple[Tuple[str, str], ...] = ()
    task: Optional[str] = None
    configuration: Optional[str] = None
    #: Whether the file's record was read: False for a record this build
    #: refuses or a file not read, whose missing figure is "not read", never
    #: "not recorded" (`validation_text`).
    validation_read: bool = True
    #: What only a served model has (the Diagnosis card's: the graph and the
    #: dataset record it serves), as ``(label, escaped HTML)`` rows, each a
    #: wide row at the end of Identity.
    served_rows: Tuple[Tuple[str, str], ...] = ()
    #: Why the dataset it was trained on is not shown, where its checkpoint
    #: was not read (the Diagnosis card's: D8, or a refresh not run yet), as
    #: ``(words, short)``: `words` on the Trained on rows, `short` in the
    #: dataset fingerprint's place, instead of "not recorded", which says the
    #: checkpoint was read and holds none (the skill's §6). None: it was read.
    trained_on_unread: Optional[Tuple[str, str]] = None
    #: HTML under Evidence's rows (the Diagnosis card's link to the full
    #: records on Models), escaped by its builder.
    evidence_more: str = ""


def facts_from_checkpoint(item: Any, workspaces: Sequence[Any], *,
                          recorded: Optional[RecordedTests] = None,
                          serving_workspace: Any = None) -> ModelFacts:
    """Model Facts for a checkpoint the server listed. `serving_workspace` is the
    listed workspace the pipeline serves now, if any, for the Data line."""
    fingerprint = _get(item, "dataset_fingerprint")
    names = trained_on(fingerprint, workspaces)
    matched = [w for w in workspaces if _get(w, "name") in names]
    synthetic = any(origin_of(w) == "synthetic" for w in matched)
    sources = sources_text(_get(matched[0], "declared_releases")) if matched else ""
    figure = validation_figure(item)
    architecture, run, filename = (_get(item, "architecture"), _get(item, "run"),
                                   _get(item, "filename"))
    warnings = []
    if synthetic:
        warnings.append(IDENTITY_BANNERS["synthetic"])
    if _get(item, "problem") is not None:
        warnings.append(("METADATA NOT READABLE: this checkpoint cannot be loaded.",
                         str(_get(item, "problem"))))
    if is_digest(fingerprint) and not names:
        warnings.append(("TRAINED ON A DATASET NO LISTED WORKSPACE HOLDS.",
                         "The dataset fingerprint this checkpoint records equals no listed "
                         "workspace's: its workspace was rebuilt or removed since."))
    if not is_digest(fingerprint) and _get(item, "problem") is None:
        warnings.append(("DATASET NOT RECORDED.",
                         "This checkpoint records no dataset fingerprint, so what it was "
                         "trained on cannot be told."))
    serving_rows: Tuple[Tuple[str, str], ...] = ()
    if serving_workspace is not None:
        same = (is_digest(fingerprint)
                and _get(serving_workspace, "dataset_fingerprint") == fingerprint)
        serving_rows = (("Serving", f"reads workspace {_get(serving_workspace, 'name')}: "
                                    + ("the same dataset" if same else "another dataset")),)
    return ModelFacts(
        name_line=model_name_line(item),
        run=f"{architecture}/{run}" if architecture and run else None,
        checkpoint_file=(f"runs/{architecture}/{run}/checkpoints/{filename}"
                         if architecture and run and filename else None),
        workspace=_get(item, "workspace"),
        model_fingerprint=_get(item, "sha256") if is_digest(_get(item, "sha256")) else None,
        trained_on_fingerprint=fingerprint if is_digest(fingerprint) else None,
        trained_on_names=names,
        synthetic=synthetic,
        sources=sources,
        validation=figure,
        recorded=recorded,
        warnings=tuple(warnings),
        serving_rows=serving_rows,
        # Task and Configuration come from the file's model record; an item
        # built without reading one (the Diagnosis card's served item, which
        # names its task in the name line, K6 decision 4) shows neither row.
        task=record_kind_text(item) if _get(item, "record") is not None else None,
        configuration=(configuration_text(item) if _get(item, "record") is not None
                       else None),
        validation_read=record_read(item),
    )


def record_read(item: Any) -> bool:
    """Whether a listed checkpoint's record was read: its file was read and
    its record is not one this build refuses (an item built without reading
    one, the Diagnosis card's served item, counts as read)."""
    return _get(item, "problem") is None and _get(item, "record") != "refused"


def validation_text(validation: Optional[Tuple[str, float]], *, read: bool = True) -> str:
    """The Trainer's validation figure, or why there is none: "not
    recorded" for a record read that holds none, "not read" for a record this
    build refuses or a file it did not read (`record_read`), which may well
    record a figure this build does not read."""
    if validation is None:
        return ("Validation (Trainer): not recorded in this checkpoint" if read
                else "Validation (Trainer): not read (record not readable)")
    metric, value = validation
    label = VALIDATION_LABELS.get(metric, f"Validation {metric} (Trainer)")
    return f"{label} {value:.4f} — {_VALIDATION_NOTE}"


def validation_value(validation: Optional[Tuple[str, float]], *, read: bool = True) -> str:
    """**The value of every details list's "Validation" row**
    (`summary_list`; Model Facts, the Test page's model, Continue's chosen
    checkpoint, Fine-tune's base: a listed item's figure by
    `validation_figure`, its `read` by `record_read`): `validation_text`
    without the key's words ("pooled MRR (Trainer) 0.2867 — selection metric
    on val, not an independent test", "not recorded in this checkpoint",
    "not read (record not readable)"), so the row never reads "Validation
    Validation …"."""
    return (validation_text(validation, read=read)
            .removeprefix("Validation (Trainer): ").removeprefix("Validation "))


def selection_value(item: Any) -> str:
    """**The value of every details list's "Selection" row**, beside its
    "Validation" row (`validation_value`; Continue's chosen checkpoint,
    Fine-tune's base): `selection_words` without the key's word ("by
    validation LP accuracy (Trainer), higher is better", "not recorded"), so
    the row never reads "Selection Selection …"."""
    return selection_words(item).removeprefix("Selection ")


# ---------------------------------------------------------------------------
# Details: the one summary list and the words of a value (the skill's §4, §5)
# ---------------------------------------------------------------------------
def summary_list(rows: Iterable[Tuple[str, str]], *, wide: Iterable[str] = (),
                 helps: Optional[Mapping[str, Tuple[str, str]]] = None,
                 titles: Optional[Mapping[str, str]] = None, classes: str = "") -> str:
    """**The one key-value summary list** (the skill's §4: one thing, several
    attributes), wherever a page shows one thing's details: a
    ``<dl class="shep-kv">``, each row its key (``dt``, plain text, escaped
    here) and its value (``dd``: HTML these renderers built, every value read
    escaped with `e` and said in its words, `setting_words`; a fingerprint by
    `fingerprint_value`, with its icon Copy). The key column is 8 to 12em wide
    (``.shep-kv``, `src/webui/style.py`); a key in `wide` (a digest's, whose
    full value is long, or a value that needs the width) takes a line of its
    own over the whole list and its value the line under it, as Model Facts
    draws its digests. `helps` gives a key its ⓘ (`help_button`'s topic and
    title), `titles` its title (a setting's record name beside its words).
    `classes` adds a page's own classes for its spacing.

    The list draws no border: it sits in its region's one container, never in
    a bordered box of its own inside one (the skill's §2)."""
    wide, helps, titles = set(wide), helps or {}, titles or {}
    cells = []
    for key, value in rows:
        extra = " shep-kv-wide" if key in wide else ""
        help_html = help_button(*helps[key]) if key in helps else ""
        title = f' title="{e(titles[key])}"' if key in titles else ""
        cells.append(f'<dt class="shep-k{extra}"{title}>{e(key)}{help_html}</dt>'
                     f'<dd class="shep-v{extra}">{value}</dd>')
    more = f" {classes}" if classes else ""
    return f'<dl class="shep-kv{more}">{"".join(cells)}</dl>'


def details_list(rows: Iterable[Tuple[str, str]], *,
                 titles: Optional[Mapping[str, str]] = None, classes: str = "") -> str:
    """**The one "Details ›" disclosure of a summary list** (the skill's
    §1.5: low-priority detail one click away, one level, never inside another
    disclosure or a popover): a closed ``<details class="shep-more">`` over
    `summary_list`'s rows (`titles` and `classes` as there), wherever a line
    names a few of many (`named_words`) and the rest are one click away."""
    return ('<details class="shep-more"><summary>Details ›</summary>'
            f'{summary_list(rows, titles=titles, classes=classes)}</details>')


def named_words(named: Sequence[str], shown: int) -> str:
    """**A line naming a few of many** (the skill's §4: past two lines a text
    gets structure): the first `shown` of `named` joined by commas, the rest
    counted ("Epochs 5 (reference 200), Seed 3 (reference 1) and 4 more"),
    each one then in its Details (`details_list`). A spec's differences
    (`model_spec_html`) and the Training page's "Set by you" line say them so."""
    more = len(named) - shown
    return ", ".join(named[:shown]) + (f" and {more} more" if more > 0 else "")


def fingerprint_value(digest: Any, *, what: str = "fingerprint", digits: int = SHORT_HEX,
                      full: bool = False, missing: str = NOT_RECORDED.lower(),
                      copy: bool = True) -> str:
    """**The one digest value**, a summary list's (`summary_list`) and a line's
    (`fingerprint_html` puts its label before it): its 12 hex digits
    (`full`: all 64, as Model Facts' digest rows show them, so the long value
    keeps one line) in bold monospace, the whole value in its title and on an
    icon Copy named for `what` (none with `copy` False, where the same digest
    is copyable once elsewhere, or in its own object's details); when there
    is none, `missing`, the state's own word ("not recorded" where the record
    holds none, "not read" where nothing was read: the skill's §6)."""
    if not is_digest(digest):
        return f'<span class="shep-muted">{e(missing)}</span>'
    shown, cls = (digest, "shep-full") if full else (digest[:digits], "shep-fpv")
    code = f'<code class="{cls}" title="{e(digest)}">{e(shown)}</code>'
    return code + (" " + copy_button(digest, f"the full {what}", icon_only=True) if copy else "")


#: **The label of each learning setting a form offers**, one name per setting:
#: the Training page's base form (`training_console.BASE_FIELDS`, what a new
#: base takes) and the Fine-tune page's (`finetune_page.FINETUNE_FIELDS`).
#: Every other page names a learning setting by it too (`field_words`), so a
#: label carries the setting's subject and, where it has one, its unit
#: ("Epochs per validation"; the skill's §5: units said once, in the key),
#: which every page then says where it names the setting. Short enough for a
#: form's column with its ⓘ (25 characters or fewer: the grid's narrowest
#: column leaves about 188 px of a label's line beside its ⓘ, and "Final
#: learning-rate ratio" takes 177 at 14 px); the form says a field's format
#: under it too (`training_console.FIELD_FORMATS`).
SETTING_LABELS: Dict[str, str] = {
    "num_epochs": "Epochs",
    "learning_rate": "Learning rate",
    "batch_size": "Batch size",
    "seed": "Seed",
    "split_seed": "Split seed",
    "num_neighbors": "Neighbours per hop",
    "max_grad_norm": "Gradient clip",
    "scheduler_type": "Scheduler",
    "warmup_steps": "Warmup steps",
    "min_lr_ratio": "Final learning-rate ratio",
    "weight_decay": "Weight decay",
    "gradient_accumulation_steps": "Gradient accumulation",
    "use_amp": "Mixed precision",
    "amp_dtype": "Mixed precision dtype",
    "early_stopping_patience": "Early stopping patience",
    "eval_every_n_epochs": "Epochs per validation",
    "save_top_k": "Top-k saves kept",
    "frozen_parts": "Graph backbone",
}
#: **The words of a setting's closed values**, keyed by its field spec's
#: ``choices`` (`training_fields.FieldSpec`): its radio shows the words and
#: sends the value (Gradio's ``(label, value)`` tuples,
#: `training_console.setting_field`), and every page says the words
#: (`setting_words`), so a value is always the spec's and no page keeps a list
#: of its own. A setting not named here shows its values as they are (the
#: scheduler's ``none``, ``cosine`` …).
SETTING_CHOICE_WORDS: Dict[str, Dict[str, str]] = {
    "frozen_parts": {"none": "Trained", "graph_side": "Frozen"},
}
#: **What a setting or a model spec field left unset (None) means**, as its
#: field spec says (`training_fields.FIELDS`: early stopping's patience "off
#: (unset) is no early stopping"; a task setting "unset is the task's
#: reference") or its spec field does (`spec.EncoderSpec.relation_edge_dim`:
#: None, edges carry no relation, the form's "off"). Any other None reads "not
#: set" (an unset seed); one the run's stage or task does not read reads so
#: first (`setting_words`).
SETTING_UNSET_WORDS: Dict[str, str] = {
    "early_stopping_patience": OFF.lower(),
    "encoder.relation_edge_dim": OFF.lower(),
    "task_loss_weight": "the task's reference",
    "lp_loss_weight": "the task's reference",
    "candidate_sample_size": "the task's reference",
    "sparse_sample_size": "the task's reference",
    "candidate_upsample_size": "the task's reference",
}
#: A None nothing else gives a meaning to: the value was not set.
NOT_SET = "not set"
#: A setting or a part a base does not read.
NOT_TAKEN_BY_A_BASE = "not taken by a base"
#: The sampler's every-neighbour hop (``-1``; `training_fields.value_problem`).
ALL_NEIGHBOURS = "all neighbours"
#: An empty list or mapping: a value with no item.
NO_ITEM = "none"
#: The parts of a model spec that a stage or task has not (`spec.ModelSpec`):
#: a base's patient encoder and task head, a task head's gene scoring outside
#: the Gene task, each null in the spec (`model_spec_html`).
_SPEC_PARTS = ("patient_encoder", "task_head", "task_head.gene")


def _not_taken_words(stage: Any, task: Any) -> str:
    """"not taken by a base", or "not taken by the Gene task"."""
    if stage == "base":
        return NOT_TAKEN_BY_A_BASE
    return f"not taken by the {TASK_NAMES.get(task, str(task))} task"


def _not_taken(name: str, stage: Any, task: Any) -> Optional[str]:
    """"not taken by a base" (or "by the Gene task") when the run's stage and
    task do not read the setting `name` (`recipe.unread_settings`; a base has
    no task either), else None."""
    if stage is None:
        return None
    try:
        unread = unread_settings(stage, task)
    except ValueError:
        return None
    if name not in unread and not (name == "task" and stage == "base"):
        return None
    return _not_taken_words(stage, task)


def setting_words(name: Any, value: Any, *, stage: Any = None, task: Any = None) -> str:
    """**The one rule of a value's words** (the skill's §5, §6), wherever a
    page shows a recorded or chosen value of the setting `name` (a training
    setting, a model spec's field): a closed value by its words
    (`SETTING_CHOICE_WORDS`: "Frozen"); a switch on or off; a list as its items,
    "15, 10, 5", a hop of ``-1`` as "all neighbours", an empty one "none"; a
    mapping as its entries, "eta: 0.5"; a number as recorded, unrounded
    (`caveats.recorded_value_text`: "0.0001", "1.0"); text as it is. **A None
    by its meaning**: "not taken by a base" (or by the run's task) where the
    run's `stage` and `task` do not read it, else what the field spec says it
    means (`SETTING_UNSET_WORDS`: "off" for no patience), else "not set". Never
    Python's repr, never a bare None. Plain text: the caller escapes it (`e`)."""
    words = SETTING_CHOICE_WORDS.get(str(name), {})
    if isinstance(value, str) and value in words:
        return words[value]
    if value is None:
        return (_not_taken(str(name), stage, task)
                or SETTING_UNSET_WORDS.get(str(name), NOT_SET))
    if isinstance(value, bool):
        return "on" if value else OFF.lower()
    if isinstance(value, (list, tuple)):
        return ", ".join(ALL_NEIGHBOURS if name == "num_neighbors" and item == -1
                         else setting_words(None, item) for item in value) or NO_ITEM
    if isinstance(value, Mapping):
        return ", ".join(f"{key}: {setting_words(None, item)}"
                         for key, item in value.items()) or NO_ITEM
    if isinstance(value, (int, float)):
        return recorded_value_text(value)
    return str(value)


def change_value(name: Any, change: Any) -> str:
    """**The value of a setting a branch changes** (ops-wiring round,
    decisions B1, B4), wherever a page says one, a recorded branch's
    (Models' "Settings changed") or a planned one's (the Continue drawer's
    check): its parent's value, then its own, each in the one value-words
    rule (`setting_words`, never Python's repr), "<code>0.0001</code> →
    <code>0.001</code>". `change` is the record's ``{"parent", "new"}``;
    anything else reads as two unset values. Escaped."""
    change = change if isinstance(change, Mapping) else {}
    return (f'<code>{e(setting_words(name, change.get("parent")))}</code> → '
            f'<code>{e(setting_words(name, change.get("new")))}</code>')


#: The words of record keys the setting labels (`SETTING_LABELS`) do not name;
#: any other key reads as its own words (`field_words`).
_FIELD_WORDS = {
    "frozen_forward": "Frozen parts' forward mode",
    "loaded_from_parent": "Loaded from the parent",
    # What it counts is its mode's (a KG link-prediction run's pairs, another
    # mode's cases): one row of every column, in words of neither.
    "n_ranked": "Number ranked",
    # The Test page's word for a version's excluded cases in all.
    "n_excluded": "Excluded",
    "augmentation_k": "Augmentation neighbours (K)",
    "val_without_negative": "Validation edges without a negative",
    "terms_without_trainable_parameters": "Terms without a trainable parameter",
}
#: Abbreviations a key's words keep as they are written.
_ABBREVIATIONS = {"amp": "AMP", "auc": "AUC", "cuda": "CUDA", "cudnn": "cuDNN", "gpu": "GPU",
                  "id": "ID", "kg": "KG", "lp": "LP", "mrr": "MRR", "roc": "ROC",
                  "sha256": "SHA-256", "url": "URL", "vcs": "VCS",
                  # A source's toolchain names its tools as they are written.
                  "gcc": "gcc", "g++": "g++", "nvcc": "nvcc"}
#: Short forms a key's words write out in full (skill §5: full words).
_SHORT_FORMS = {"concat": "concatenate", "dim": "dimension", "eps": "epsilon",
                "grad": "gradient", "init": "initialisation", "lr": "learning rate",
                "norm": "normalisation", "sp": "shortest-path", "val": "validation"}


def field_words(key: Any) -> str:
    """**A record key or a setting in words** (skill §5: sentence case, full
    words), wherever a page names one (Compare's rows, Continue's settings): a
    learning setting by its form's label (`SETTING_LABELS`, the one map of
    those: "Learning rate"), a few other keys by their words ("Number
    ranked"), any other by its own words, its abbreviations as written and its
    short forms in full ("downstream_initial_sha256" → "Downstream initial
    SHA-256", "base_lr" → "Base learning rate"). A key with a capital letter is
    a name as recorded, as a build flag's environment variable is
    ("MAX_JOBS"), and reads as written, as a tool's name does. Where a page
    shows the words, the key itself is their title (Compare's row,
    `summary_list`'s `titles`)."""
    text = str(key)
    if text in SETTING_LABELS:
        return SETTING_LABELS[text]
    if text in _FIELD_WORDS:
        return _FIELD_WORDS[text]
    if text != text.lower():
        return text
    words = [_ABBREVIATIONS.get(word) or _SHORT_FORMS.get(word, word)
             for word in text.split("_") if word]
    if not words:
        return text
    if words[0] not in _ABBREVIATIONS.values():
        words[0] = words[0][:1].upper() + words[0][1:]
    return " ".join(words)


def field_path_words(path: Any) -> str:
    """**A record field's dotted path in words** (skill §5): each part's
    words (`field_words`) joined by " › ", as Compare names a nested entry
    ("encoder.heads" → "Encoder › Heads", "lp_head.relation_init" → "LP
    head › Relation initialisation"). Where a page shows them, the path
    itself is their title (`summary_list`'s `titles`), the record's name
    beside its plain label."""
    return " › ".join(field_words(part) for part in str(path).split("."))


def _spec_value_html(path: str, value: Any) -> str:
    """A model spec field's value, escaped: in its words (`setting_words`),
    and a text recorded as it is (an identifier: "graph_layer_norm+batch_norm",
    "reference/v1") kept whole in code style (`ident_html`), never broken
    inside its name."""
    words = setting_words(path, value)
    return ident_html(words) if isinstance(value, str) and words == value else e(words)


def _spec_rows(data: Mapping[str, Any], prefix: str = "") -> List[Tuple[str, Any]]:
    """A model spec's plain data as ``(dotted path, value)`` rows, every
    field of every section in its recorded order (a mapping's entries one row
    each, as `ModelSpec.differences_from_reference` names a field). Its
    ``sources`` are left out: they are the reference recipe's provenance
    notes, carried with every spec whatever its values (a changed field keeps
    the reference's note), so they say nothing of this model's; a model's
    listed spec leaves them out alike (`record.listed_record`)."""
    rows: List[Tuple[str, Any]] = []
    for key, value in data.items():
        if not prefix and key == "sources":
            continue
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping) and value:
            rows.extend(_spec_rows(value, path))
        else:
            rows.append((path, value))
    return rows


#: How many of a spec's differences its words name (`model_spec_html`); the
#: rest are counted, each one marked in its Details.
_SPEC_WORDS_SHOWN = 4


def model_spec_html(data: Any) -> Tuple[str, str]:
    """**A recorded model spec, said by its differences** (the skill's §1.5,
    §4; ops-wiring round B0c): ``(words, details)``, escaped HTML. `words` says
    the spec by its relation to the reference recipe of its own stage and task
    (`ModelSpec.differences_from_reference`): "Reference GATv2 spec", or
    "Differs from the reference GATv2 spec: Encoder › Layers 2 (3), …", each
    field that differs named in words (`field_path_words`) with its value and
    the reference's in brackets, the first `_SPEC_WORDS_SHOWN` of them
    and the others counted; a spec this build cannot read says why
    ("Unknown: …") and has no details. `details` is the full spec one click
    away: a "Details ›" disclosure over every field as recorded (`_spec_rows`,
    without the reference recipe's notes), each named in the same words with
    its dotted path as the title, each that differs with "reference …" beside
    it; one level, never inside another disclosure or a popover. Every value
    in its words (`_spec_value_html`); a part its stage or task has not
    (`_SPEC_PARTS`), "not taken by a base" or by its task."""
    from src.models.bundle.spec import ModelSpec, SpecError

    if not isinstance(data, Mapping):
        return (status_mark("neutral", f"{UNKNOWN}: the record holds no model spec"), "")
    try:
        spec = ModelSpec.from_dict(data)
        differences = spec.differences_from_reference()
    except (SpecError, TypeError, ValueError, KeyError) as exc:
        return (status_mark("neutral", f"{UNKNOWN}: not read as a model spec "
                                       f"({short_reason(exc, 120)})"), "")
    name = f"reference {architecture_label(spec.encoder.kind)} spec"
    if differences:
        words = f"Differs from the {name}: " + named_words(
            [f"{field_path_words(path)} {setting_words(path, mine)} "
             f"({setting_words(path, reference)})"
             for path, (reference, mine) in differences.items()], _SPEC_WORDS_SHOWN)
    else:
        words = name[:1].upper() + name[1:]
    rows, titles = [], {}
    for path, value in _spec_rows(data):
        if value is None and path in _SPEC_PARTS:
            shown = e(_not_taken_words(spec.learning_stage, spec.task))
        else:
            shown = _spec_value_html(path, value)
        if path in differences:
            shown += f" (reference {_spec_value_html(path, differences[path][0])})"
        key = field_path_words(path)
        rows.append((key, shown))
        titles[key] = path
    return e(words), details_list(rows, titles=titles, classes="shep-kv-spec")


def recorded_count_text(value: Any) -> str:
    """A count as its record holds it (the integer as written, with no
    separator), or "not recorded" where the record holds no integer (the
    skill's §6, as `count_text` says it): the words every result surface
    gives a recorded count in (Models' records, the Test page's result,
    `data_kind_lines`, `r10_rows`)."""
    return str(value) if type(value) is int else NOT_RECORDED.lower()


#: An R10 scope's evidence state in words (`record_views.R10Evidence`'s
#: states): whether the inputs the checkpoint records were found.
R10_EVIDENCE_WORDS = {
    "found": "evidence found",
    "not_found": "evidence not found",
    "unreadable": "evidence found, not readable",
    "not_recorded": "evidence not recorded by the checkpoint",
    "search_incomplete": "evidence search incomplete",
}


def r10_state_text(label: str, part: Any, evidence: Any) -> str:
    """One R10 label's state in the caveats' words, as text: a count with its
    denominator only where the state has one; *unverifiable* is a reason
    (by its scope's evidence state, `caveats.R10_UNVERIFIABLE_REASON`), never
    a number; "not reported" where the state does not read. Models ›
    Results' Compare shows it beside the label's name (its row's), and
    `r10_label_text` after it."""
    state = _get(part, "state")
    words = R10_STATE_TEXT.get(state) if isinstance(state, str) else None
    if state == "unverifiable" and isinstance(evidence, str):
        words = R10_UNVERIFIABLE_REASON.get(evidence, words)
    if words is None:
        return "not reported"
    if state not in ("overlap", "none"):
        return words
    unit = "distinct test diseases" if label == "disease_overlap" else "test cases"
    text = (f"{words} — {recorded_count_text(_get(part, 'k'))} of "
            f"{recorded_count_text(_get(part, 'n'))} {unit}")
    if label == "disease_overlap":
        text += f"; {recorded_count_text(_get(part, 'cases'))} test cases with one of them"
    return text


def r10_label_text(label: str, name: str, part: Any, evidence: Any) -> str:
    """One R10 label in the caveats' words, as text: its name, then its state
    (`r10_state_text`). The text of `r10_rows`."""
    return f"{name}: {r10_state_text(label, part, evidence)}"


def r10_rows(r10: Any) -> list:
    """**The one rendering of a supplied cohort's R10** (research comparison
    Stage 8b), as ``(label, escaped HTML)`` rows, one per scope, from the one
    reader's view of a record (its ``r10``: each scope's evidence state and its
    disease and phenotype-set overlap): ``("R10, training", "inputs (evidence
    found): Training disease overlap: … · Training phenotype-set overlap: …")``,
    each label in `r10_label_text`'s words with *unverifiable* in bold. Models'
    records and the Test page's result lay the rows out each their own way,
    as they lay out `data_kind_lines`; the evidence's path is never read."""
    rows = []
    for scope, labels in R10_LABEL_TEXT.items():
        part = _get(r10, scope)
        evidence = _get(_get(part, "evidence"), "state")
        found = R10_EVIDENCE_WORDS.get(evidence) if isinstance(evidence, str) else None
        shown = []
        for label, words in labels.items():
            name = words["name"]
            text = r10_label_text(label, name, _get(part, label), evidence)
            lead = f"{name}: unverifiable"
            shown.append(f"{e(name)}: <b>unverifiable</b>{e(text[len(lead):])}"
                         if _get(_get(part, label), "state") == "unverifiable"
                         and text.startswith(lead) else e(text))
        rows.append((f"R10, {scope}",
                     f'<span class="shep-muted">inputs ({e(found or "evidence not reported")}):'
                     f'</span> {" · ".join(shown)}'))
    return rows


def data_kind_lines(view: Mapping[str, Any]) -> list:
    """**The one rendering of what a result's data kind adds** (research
    comparison Stages 6 and 8b), as ``(label, escaped HTML)`` rows, from the
    one reader's view of a record (`record_views`: a ledger record's record
    view, or a kept report's result view, `result_view`, as a JSON dump), so a
    registered record on Models and a kept report on Test say it alike, each
    page laying the rows out its own way:

    - a KG edge set's exposure in this build's words (`caveats.exposure_text`,
      the ⓘ holding `EXPOSURE_CAVEAT`) and its source (definition version and
      split, or why its block does not read: fields named, never values), its
      message-passing graph and pairs, and each metric that could not be
      computed, as a status;
    - a generated split's generation identity, its split manifest and gene
      fixture, and how it stands to a model's training
      (`caveats.GENERATED_EXPOSURE_NOTE`);
    - nothing for a supplied cohort, whose relation to training is its R10.

    The KG edges' "held out" is never said for a patient cohort, and R10
    never for a KG edge set or a generated split. Every value is escaped."""
    cohort = _get(view, "cohort")
    cohort = cohort if isinstance(cohort, Mapping) else {}
    kind = cohort.get("kind")
    if kind == "generated":
        generation = cohort.get("generation") if isinstance(cohort.get("generation"),
                                                            Mapping) else {}
        allocation = cohort.get("allocation") if isinstance(cohort.get("allocation"),
                                                            Mapping) else {}
        fixture = cohort.get("task_fixture")
        manifest = cohort.get("split_manifest")
        identity = (f"{generation.get('algorithm')} v{generation.get('algorithm_version')}, "
                    f"allocation {allocation.get('algorithm')} v"
                    f"{allocation.get('algorithm_version')} with seed {allocation.get('seed')}")
        record = fixture.get("record") if isinstance(fixture, Mapping) else None
        kept = (f"gene fixture {short_hex(_get(record, 'sha256'))}"
                if isinstance(fixture, Mapping) else "no gene fixture kept")
        return [("Generation", e(identity)),
                ("Split manifest", f"<code>{e(short_hex(_get(manifest, 'sha256')))}</code>; "
                                   f"{e(kept)}"),
                ("Training relation", e(GENERATED_EXPOSURE_NOTE))]
    if kind != "kg_edges":
        return []
    exposure = _get(view, "exposure")
    exposure = exposure if isinstance(exposure, Mapping) else None
    split = exposure.get("split") if exposure is not None else None
    # An exposure block the one reader could not read is null with its
    # problem (fields named, never values): the result stands beside it.
    source = ("Definition " + str(exposure.get("definition_version")) + "; "
              + (split if isinstance(split, str) else "the model record was not read")
              if exposure is not None else
              f"Its block does not read: {_get(view, 'exposure_problem') or 'no reason given'}")
    passing = cohort.get("message_passing") if isinstance(cohort.get("message_passing"),
                                                          Mapping) else {}
    pairs = cohort.get("pairs") if isinstance(cohort.get("pairs"), Mapping) else {}
    graph = (f"Message passing over the split's {passing.get('part')} part, "
             f"{recorded_count_text(passing.get('edges'))} edges; pairs {pairs.get('rule')} "
             f"with seed {pairs.get('seed')}, {recorded_count_text(pairs.get('rows'))} rows")
    measured = _get(view, "measured")
    statuses = measured.get("metrics") if isinstance(measured, Mapping) else None
    not_computed = [str(name) for name, status in (statuses or {}).items()
                    if isinstance(status, Mapping) and status.get("status") != "computed"]
    rows = [("Exposure", e(exposure_text(exposure))
             + help_button("kg-exposure", "Exposure (KG edge sets)")),
            ("Exposure source", e(source)),
            ("Graph and pairs", e(graph))]
    if not_computed:
        rows.append(("Not computable", e(", ".join(not_computed))
                     + " (a status, never a number)"))
    return rows


#: The labels of Model Facts' digest rows (`summary_list`'s wide rows).
FACTS_MODEL_FP = "Model fingerprint (checkpoint sha256)"
FACTS_TRAINED_FP = ("Trained on: dataset fingerprint (split_manifest.json, from the checkpoint's "
                    "record)")


def model_facts_html(facts: ModelFacts, *, open_section: str = "identity") -> str:
    """Model Facts: 1 Identity, 2 Uses, 3 Data, 4 Evidence (each a disclosure
    with a one-line summary) and 5 Warnings, always open. Identity, Data and
    Evidence are each one summary list (`summary_list`: one row per fact,
    the skill's §4, §5); a trained-on dataset whose
    checkpoint was not read says why (`ModelFacts.trained_on_unread`), never
    "not recorded"."""
    def section(key: str, number: int, title: str, summary: str, body: str) -> str:
        opened = " open" if key == open_section else ""
        return (f'<details class="shep-facts-sec" data-sec="{key}"{opened}>'
                f'<summary><span class="shep-facts-n">{number}</span> {e(title)}'
                f'<span class="shep-facts-sum">{summary}</span></summary>'
                f'<div class="shep-facts-bd">{body}</div></details>')

    location = ""
    if facts.checkpoint_file:
        location = (f'<code>{e(facts.checkpoint_file)}</code>'
                    + (f' <span class="shep-muted">(workspace {e(facts.workspace)})</span>'
                       if facts.workspace else ""))
    unread = facts.trained_on_unread
    model_fp = fingerprint_value(facts.model_fingerprint, what="model fingerprint", full=True,
                                 missing="not read")
    data_fp = fingerprint_value(facts.trained_on_fingerprint, what="dataset fingerprint",
                                full=True, missing=unread[0] if unread else NOT_RECORDED.lower())
    # The listed workspaces holding that dataset follow the label, as the
    # mockup draws it, so the full fingerprint and its Copy keep one line.
    trained_label = FACTS_TRAINED_FP + (f" · {', '.join(facts.trained_on_names)}"
                                        if facts.trained_on_names else "")
    identity = summary_list([
        ("Model", e(facts.name_line)),
        *([("Task", e(facts.task))] if facts.task else []),
        *([("Configuration", e(facts.configuration))] if facts.configuration else []),
        ("Run", f"<code>{e(facts.run)}</code>" if facts.run else "—"),
        ("Checkpoint file", location or "—"),
        (FACTS_MODEL_FP, model_fp),
        (trained_label, data_fp),
        *facts.served_rows,
    ], wide=(FACTS_MODEL_FP, trained_label, *(label for label, _value in facts.served_rows)))
    uses_text = f"{NOT_A_MEDICAL_DEVICE} {SYNTHETIC_FOR_TESTING}"
    trained = (unread[0] if unread
               else ", ".join(facts.trained_on_names) if facts.trained_on_names
               else "matches no listed workspace" if facts.trained_on_fingerprint
               else NOT_RECORDED.lower())
    # The dataset fingerprint once more, short and without its Copy: Identity
    # shows it whole with its Copy.
    data_rows = [("Trained on", e(trained) + (" " + chip("synthetic", "amber")
                                              if facts.synthetic else "")),
                 ("Dataset fingerprint",
                  fingerprint_value(facts.trained_on_fingerprint, copy=False,
                                    missing=unread[1] if unread else NOT_RECORDED.lower()))]
    if facts.sources:
        data_rows.append(("Sources", e(facts.sources)))
    data_rows.extend((label, e(words)) for label, words in facts.serving_rows)
    evidence = summary_list([
        ("Validation", e(validation_value(facts.validation, read=facts.validation_read))),
        ("Recorded results",
         e(recorded_tests_text(facts.recorded).removeprefix("Recorded results: "))),
    ], helps={"Validation": ("validation-trainer", "Validation (Trainer)")}) + facts.evidence_more
    if facts.warnings:
        warnings = "".join(
            f'<p class="shep-warn"><b>⚠ {e(h)}</b> <span>{e(d)}</span></p>'
            for h, d in facts.warnings)
    else:
        warnings = '<p class="shep-muted">No warnings.</p>'
    return (
        '<div class="shep-facts">'
        + section("identity", 1, "Identity", "", identity)
        + section("uses", 2, "Uses", f" · {e(uses_text)}", f"<p>{e(uses_text)}</p>")
        + section("data", 3, "Data", "", summary_list(data_rows))
        + section("evidence", 4, "Evidence", "", evidence)
        + '<section class="shep-facts-sec shep-facts-warn" data-sec="warnings">'
          '<h4><span class="shep-facts-n">5</span> Warnings</h4>'
          f'<div class="shep-facts-bd">{warnings}</div></section>'
        + "</div>"
    )


__all__ = [
    "CHECK",
    "FACTS_MODEL_FP",
    "FACTS_TRAINED_FP",
    "FROZEN_BACKBONE_WORDS",
    "IDENTITY_BANNERS",
    "MODEL_TABLE_COLUMNS",
    "ModelFacts",
    "ALL_NEIGHBOURS",
    "NOT_A_MEDICAL_DEVICE",
    "NOT_SET",
    "NOT_TAKEN_BY_A_BASE",
    "NO_READABLE_SPLIT_MANIFEST",
    "NO_SPLIT_MANIFEST",
    "R10_EVIDENCE_WORDS",
    "RENDERER_HELP",
    "RecordedTests",
    "OPTION_CHARS",
    "SELECTION_DIRECTIONS",
    "SETTING_CHOICE_WORDS",
    "SETTING_LABELS",
    "SETTING_UNSET_WORDS",
    "VALIDATION_LABELS",
    "architecture_label",
    "best_saves",
    "clock_utc",
    "count_text",
    "dataset_details_html",
    "details_list",
    "SHORT_HEX",
    "SYNTHETIC_FOR_TESTING",
    "built_text",
    "change_value",
    "chip",
    "configuration_text",
    "data_kind_lines",
    "dataset_identity_html",
    "distinct_names",
    "distinguishing_digits",
    "e",
    "epoch_words",
    "facts_from_checkpoint",
    "field_path_words",
    "field_words",
    "fingerprint_html",
    "fingerprint_value",
    "fit_option",
    "fit_options",
    "freeze_words",
    "ident_html",
    "is_digest",
    "metric_words",
    "model_facts_html",
    "model_kind_text",
    "model_spec_html",
    "model_name_line",
    "named_words",
    "model_row_html",
    "model_table_row_html",
    "not_trainable_short",
    "origin_of",
    "r10_label_text",
    "r10_rows",
    "r10_state_text",
    "record_chip",
    "record_kind_text",
    "record_read",
    "record_verified",
    "recorded_count_text",
    "recorded_tests_cell",
    "recorded_tests_text",
    "resources_html",
    "run_label",
    "run_label_html",
    "run_name",
    "run_started",
    "run_suffix",
    "run_time",
    "selection_value",
    "selection_words",
    "setting_words",
    "short_hex",
    "short_reason",
    "sources_text",
    "summary_list",
    "trained_on",
    "trained_on_html",
    "validation_cell",
    "validation_text",
    "validation_figure",
    "validation_value",
    "word_groups",
    "workspace_choices",
]
