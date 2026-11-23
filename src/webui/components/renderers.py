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
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from typing import (Any, Callable, Dict, Hashable, Iterable, List, Mapping, NamedTuple,
                    Optional, Sequence, Tuple)

from src.config.recipe import FROZEN_PARTS, unread_settings
from src.config.server_settings import PLACE_ACTIONS, SEVERAL_ACTIONS
from src.evaluation.caveats import (
    EXPOSURE_EVIDENCE,
    EXPOSURE_NEGATIVES,
    EXPOSURE_NOT_EXHAUSTIVE,
    EXPOSURE_NOT_PROVEN,
    EXPOSURE_SCOPE,
    GENERATED_EXPOSURE_NOTE,
    LP_METRIC_WORDS,
    R10_LABEL_TEXT,
    RELATION_DISPLAY_NAMES,
    R10_STATE_TEXT,
    R10_UNVERIFIABLE_REASON,
    TASK_NAMES,
    exposure_text,
    metric_text,
    recorded_metric_lines,
    recorded_value_text,
)
from src.utils.run_environment import native_identity_lines
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
#: banners and Model Facts' warnings say them in these words. Headlines are in
#: sentence case (the skill's §5, R-26): the banner's mark and tone make them
#: stand out, never capitals.
IDENTITY_BANNERS = {
    "not_reported": (
        "What produced this result was not reported.",
        "The API's response carried no served identity.\n"
        "So these candidates cannot be attributed to a model, graph or table.",
    ),
    "synthetic": (
        "Synthetic workspace: not a clinical result.",
        "The served workspace's dataset record gives its origin as synthetic.\n"
        "A generated stand-in, not a clinical knowledge graph.\nResults are for testing only.",
    ),
}
#: Model Facts' other warning headlines (`facts_from_checkpoint`), in the
#: banners' sentence case; named, so a caller tells them apart by the constant,
#: never by their spelling.
METADATA_NOT_READABLE = "Metadata not readable: this checkpoint cannot be loaded."
TRAINED_ON_UNLISTED = "Trained on a dataset no listed workspace holds."
DATASET_NOT_RECORDED = "Dataset not recorded."

#: The help texts the renderers' ⓘ name; the shell registers them.
RENDERER_HELP = {
    "dataset-fingerprint": (
        "Dataset fingerprint",
        "The split manifest's SHA-256: equal fingerprints mean the same build.\n\n"
        "- Binds by digest: graph, three tensors, both sample files, dataset record\n"
        "- Records: the recipe, the seeds and the split\n"
        "- Shown: the first 12 of 64 hex digits, all 64 compared\n"
        "- Checked: the files against it, when a run or a load reads them\n"
        "Ref: split_manifest.json",
    ),
    "model-fingerprint": (
        "Model fingerprint",
        "The SHA-256 of the checkpoint file's bytes, as the server read them.\n\n"
        "- Shown: the first 12 of 64 hex digits, all 64 compared\n"
        "- Named by it: a test record, a usage history entry, the served identity",
    ),
    # The exposure caveat's own parts, a line each (`caveats.EXPOSURE_CAVEAT`,
    # which records and the API keep whole): the skill's §13 lets a governed
    # text change its presentation, never its words' meaning. Its parts are
    # sentences longer than a bubble's two-line item, so each is a paragraph.
    "kg-exposure": (
        "Exposure (KG edge sets)",
        f"{EXPOSURE_SCOPE}.\n"
        f"{EXPOSURE_EVIDENCE[:1].upper()}{EXPOSURE_EVIDENCE[1:]}.\n"
        f"{EXPOSURE_NOT_EXHAUSTIVE}.\n"
        f"Negatives: {EXPOSURE_NEGATIVES}.\n"
        f"They are {EXPOSURE_NOT_PROVEN}.",
    ),
    "validation-trainer": (
        "Validation (Trainer)",
        "The Trainer's own figure on the val samples, recorded in the checkpoint.\n"
        "- Base: LP accuracy.\n"
        "- Task model: pooled MRR.\n"
        "Training selects a checkpoint by it, so it is not an independent test.\n"
        "Recorded test results are kept apart from it.",
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
#: **The space between a mark's shape and its word** (`status_mark`): in its
#: text, so a copy and ``innerText`` read "○ Saved", never "○Saved"; of no
#: width (``shep-st-sp``), so the shape's own margin still draws the gap; a
#: no-break space, so a line never parts the shape from its word (the skill's
#: §6; the sweep's L7, SW-LK7: "model fingerprint 60adc7154078 ●" ended a
#: line of the Diagnosis card, "verified" began the next).
MARK_SPACE = '<span class="shep-st-sp">\u00a0</span>'


def status_mark(kind: str, words: Any, *, words_html: bool = False,
                as_chip: bool = False) -> str:
    """**The one status indicator** (the skill's §6): the state's colour, its
    shape and its word together, so the colour never carries the meaning
    alone. `kind` is a `STATUS_MARKS` row; `words` the state's word or the
    condition's short sentence, always shown beside the shape, escaped (with
    `words_html`, HTML these renderers built and escaped, a sentence with a
    bold part or a code value). The shape is hidden from a screen reader,
    which reads the word. **Its text reads shape, space, word** ("○ Saved"),
    copied or read as ``innerText``: a space (`MARK_SPACE`, no width: the
    shape's margin draws the gap) between them, in one run of text. Drawn as
    coloured text (``shep-st``), or with `as_chip` as a tag: a state to watch,
    or one that stands apart in a row of tags; the tag is a flex box, whose
    items read as lines of their own, so its shape, space and word are one
    item (``shep-st-in``). No page shows a mark without its word."""
    shape, token = STATUS_MARKS[kind]
    mark = ('<span class="shep-spinner" aria-hidden="true"></span>' if shape is None else
            f'<span class="shep-st-shape" aria-hidden="true">{shape}</span>')
    text = str(words) if words_html else e(words)
    # The spinner has no text of its own, so nothing to space from.
    said = f'{mark}{"" if shape is None else MARK_SPACE}<span class="shep-st-w">{text}</span>'
    if as_chip:
        return (f'<span class="shep-tag shep-tag-{_STATUS_TAG[token]} shep-st-chip">'
                f'<span class="shep-st-in">{said}</span></span>')
    return f'<span class="shep-st shep-st-{token}">{said}</span>'

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX12 = re.compile(r"^[0-9a-f]{12}$")
#: The title of a digest known by its first 12 hex digits alone (`fingerprint_value`).
PREFIX_TITLE = "the first 12 hex digits, as the report's run id names them"
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


#: **The separator between the facts of one line** (" · ", hidden from a
#: screen reader): joined to the fact before it by a no-break space, so a
#: line that wraps breaks after it and a line never starts with "·" (the L6
#: check: Test's identity line drew "· synthetic …" at 1550 px).
SEP = '<span class="shep-sep" aria-hidden="true">\u00a0· </span>'


def chain_html(*facts: Any, tail: str = "") -> str:
    """**Facts of one line joined by the shared separator** (`SEP`), each
    escaped and drawn as one unit (``shep-fact``): it moves to the next line
    whole where the line cannot hold it, and wraps inside only when it is
    wider than a line (L7 SW-LK7: "GATv2 2026-10-02 00:00, 1 epoch complete"
    and "recipe changed in 5 fields" broke inside though a line held them).
    `tail`, HTML, ends the last fact's unit: the "Details ›" that opens that
    fact moves with it (L7 SW-LK7: Fine-tune's "Details ›" under "recipe
    changed in 5 fields"). Empty facts are left out."""
    shown = [e(fact) for fact in facts if fact]
    if tail and shown:
        shown[-1] += " " + tail
    return SEP.join(f'<span class="shep-fact">{fact}</span>' for fact in shown)


def whole_words(words: Any) -> str:
    """`words` escaped, every word kept whole: a line breaks only at a space,
    never at a word's own hyphen ("Fine-tune", "demo-test-1", "2026-11-16",
    "model-12.pt"), so a narrow frame never cuts a word (D8) and a name never
    reads as hyphenated across two lines (the L6 check: Datasets' versions
    table broke "demo-test-1" as "demo-" / "test-1")."""
    return " ".join(f'<span class="shep-nowrap">{e(word)}</span>' if "-" in word else e(word)
                    for word in str(words).split(" "))


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
    items = SEP.join(f'<span>{e(p)}</span>' for p in parts)
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


def _glues(ch: str) -> bool:
    """Whether `ch` joins the character before it (UAX #29's GB9 and GB9a):
    a combining or spacing mark, a variation selector, a zero-width joiner or
    non-joiner, an emoji skin tone or a tag character (a subdivision flag)."""
    cp = ord(ch)
    return (unicodedata.category(ch) in ("Mn", "Me", "Mc") or ch in "‌‍"
            or 0x1F3FB <= cp <= 0x1F3FF or 0xE0020 <= cp <= 0xE007F)


def _jamo(ch: str) -> str:
    """A Hangul jamo's or syllable's part in a syllable block (UAX #29's L,
    V, T, LV, LVT), or "" for any other character."""
    cp = ord(ch)
    if 0x1100 <= cp <= 0x115F or 0xA960 <= cp <= 0xA97C:
        return "L"
    if 0x1160 <= cp <= 0x11A7 or 0xD7B0 <= cp <= 0xD7C6:
        return "V"
    if 0x11A8 <= cp <= 0x11FF or 0xD7CB <= cp <= 0xD7FB:
        return "T"
    if 0xAC00 <= cp <= 0xD7A3:
        return "LV" if (cp - 0xAC00) % 28 == 0 else "LVT"
    return ""


#: Which Hangul part may follow which in one syllable block (UAX #29's GB6-GB8).
_JAMO_NEXT = {"L": {"L", "V", "LV", "LVT"}, "LV": {"V", "T"}, "V": {"V", "T"},
              "LVT": {"T"}, "T": {"T"}}


def _regional(ch: str) -> bool:
    """A regional indicator (two make a flag)."""
    return 0x1F1E6 <= ord(ch) <= 0x1F1FF


def grapheme_ends(text: str) -> List[int]:
    """**Where `text` may be cut without splitting what a reader sees as one
    character** (a grapheme cluster, UAX #29): every offset between two
    clusters, 0 and ``len(text)`` included. By the rules a name can meet: CR
    LF; a mark, a variation selector, a skin tone or a tag character with
    what it follows; a Hangul syllable block; flags as pairs of regional
    indicators; and nothing after a zero-width joiner (an emoji sequence such
    as the family 👨‍👩‍👧, five code points, is one character). The standard
    library has no grapheme segmentation (``unicodedata`` gives categories,
    not UAX #29's break properties), so this is written here from them; it
    never cuts where UAX #29 would not, and may keep a rarer sequence
    together where UAX #29 would cut it, which only moves a cut to a
    neighbouring cluster."""
    ends, flags = [0], 0
    for i in range(1, len(text)):
        before, ch = text[i - 1], text[i]
        flags = flags + 1 if _regional(before) else 0
        if before == "\r" and ch == "\n":
            continue
        if unicodedata.category(before) in ("Cc", "Zl", "Zp") \
                or unicodedata.category(ch) in ("Cc", "Zl", "Zp"):
            ends.append(i)
        elif _glues(ch) or before == "‍" or _jamo(ch) in _JAMO_NEXT.get(_jamo(before), ()):
            continue
        elif not (_regional(ch) and flags % 2):
            ends.append(i)
    if text:
        ends.append(len(text))
    return ends


def cut_at(text: str, head: int, keep: int, ends: Optional[List[int]] = None) -> str:
    """`text` as its start of up to `head` characters, an ellipsis, then its
    end, `keep` characters of it in all at most (`head` at most `keep`). Cut
    only between two grapheme clusters (`grapheme_ends`, or `ends` when the
    caller has them, B72-N4): a character drawn from several code points (a
    ZWJ emoji sequence, a flag, a letter and its accent) stays whole on its
    side of the "…" or is left out whole, the end given the room the start
    could not use. The one cut of `middle_cut` and of Diagnosis' cohort
    options (`diagnosis_panel._option_text`), which try every `head`."""
    ends = grapheme_ends(text) if ends is None else ends
    first = max(end for end in ends if end <= head)
    last = min(end for end in ends if len(text) - end <= keep - first)
    return text[:first] + "…" + text[last:]


def middle_cut(text: str, room: int, *, head: int = 0) -> str:
    """`text` in at most `room` characters: whole where it fits, else cut in
    its middle with an ellipsis (`cut_at`, between grapheme clusters), so its
    start and its end stay (where a copy or a version differs: "…copy_b.pt",
    "…67-v1.pt"), its start at least `head` characters long where the room
    allows (`fit_apart`: through where two alike names differ). Never longer
    than `text`."""
    if len(text) <= room:
        return text
    keep = max(room, 1) - 1
    return cut_at(text, max(keep // 2, min(head, keep)), keep)


def _fitted(what: str, rest: Optional[str], mark: str = "",
            keep: int = 0) -> Tuple[str, Optional[str]]:
    """`fit_option`'s two parts, each as the option says it: `what` with its
    mark, and `rest`."""
    room = OPTION_CHARS - len(mark)
    if rest:
        room -= 3
        rest = middle_cut(rest, room - min(len(what), _WHAT_ROOM))
        room -= len(rest)
    return middle_cut(what, room, head=keep) + mark, rest


def fit_option(what: str, rest: Optional[str], mark: str = "", *, keep: int = 0) -> str:
    """"<what><mark> · <rest>" in at most `OPTION_CHARS` characters with one
    separator: `what` (the distinguishing token: a name, a file's name) is
    shortened in its middle (`middle_cut`, its start at least `keep`
    characters where it can: `fit_apart`); `rest` stays whole while `what`
    keeps at least `_WHAT_ROOM` characters (or all of its own, if fewer), and
    past that is cut in its middle too, so neither is ever the only part cut
    and the option never runs past `OPTION_CHARS`."""
    what, rest = _fitted(what, rest, mark, keep)
    return f"{what} · {rest}" if rest else what


def _shared_start(one: str, other: str) -> int:
    """How many leading characters `one` and `other` share."""
    return next((i for i, (a, b) in enumerate(zip(one, other)) if a != b),
                min(len(one), len(other)))


def fit_apart(entries: Mapping[Hashable, Tuple[str, Optional[str]]]) -> Dict[Hashable, str]:
    """**Words shown together, each `fit_option`, told apart where a cut would
    hide what tells them apart**, by key, from ``(what, rest)``: entries
    whose different `what`s would be cut alike (the cut keeps their alike
    start and end) each keep their start through the first two characters
    where they differ ("cpb-dis-ful…" and "cpb-dis-fro…", never both
    "cpb-dis…disease"), the end filling the rest of the room. A `what` alike
    in full is the caller's to tell apart (`distinct_names`, `fit_options`)."""
    cuts: Dict[str, List[Hashable]] = {}
    for key, (what, rest) in entries.items():
        cuts.setdefault(_fitted(what, rest)[0], []).append(key)
    keep = dict.fromkeys(entries, 0)
    for keys in cuts.values():
        for key in keys:
            what = entries[key][0]
            for other in keys:
                theirs = entries[other][0]
                if theirs != what:
                    keep[key] = max(keep[key], _shared_start(what, theirs) + 2)
    return {key: fit_option(what, rest, keep=keep[key])
            for key, (what, rest) in entries.items()}


def fit_options(entries: Sequence[Tuple[str, Optional[str], Any]]) -> List[Tuple[str, Any]]:
    """A select's options from ``(what, rest, value)`` entries, in their order:
    each `fit_option`, told apart where a cut would hide their difference
    (`fit_apart`), and those still alike numbered in that order, "(2)", the
    last resort for what nothing else tells apart. Unique, at most
    `OPTION_CHARS` characters and one separator; **each value is its entry's
    own, whatever the words** (Gradio's ``(label, value)``: the skill's §5, a
    label never changes what is chosen)."""
    fitted = fit_apart({place: (what, rest) for place, (what, rest, _value) in enumerate(entries)})
    labels = [fitted[place] for place in range(len(entries))]
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


def _dig(item: Any, *names: str) -> Any:
    """``item[a][b]…`` through mappings, None where a step is not one (the
    one reader's views are plain JSON; a field may be absent or not read)."""
    for name in names:
        if not isinstance(item, Mapping):
            return None
        item = item.get(name)
    return item


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
    MONDO release not declared"; empty when it declares none (`sources_html`
    says why)."""
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
    """A workspace's dataset record as a chip by the one status mark
    (`status_mark`, `record_words`): "● record verified" (the dataset record
    checked), else the record's own state, "record not recorded" apart from
    "record cannot be verified", "record does not match" and "record
    unknown"."""
    return status_mark(*record_words(item), as_chip=True)


def dataset_marks_html(item: Any) -> str:
    """**A workspace's origin and record, one form wherever they show** (the
    run bar's identity line, `dataset_identity_html`, and the selected
    workspace's header on Research › Datasets): "synthetic" in amber, then
    the record's state (`record_chip`) once the manifest was read; empty when
    neither applies."""
    marks = []
    if origin_of(item) == "synthetic":
        marks.append(chip("synthetic", "amber"))
    if record_verified(item) or _get(item, "dataset_fingerprint"):
        marks.append(record_chip(item))
    return "".join(marks)


def not_trainable_chip(item: Any) -> str:
    """"not trainable: <why>" as a grey tag, in short words
    (`not_trainable_short`), its whole reason in its title: the run bar's
    identity line and Research › Datasets' In use column alike."""
    why = str(_get(item, "not_trainable") or "not trainable")
    return chip(f"not trainable: {not_trainable_short(item)}", "grey", title=why)


#: The sources of a synthetic build, which declares none (`sources_html`).
SYNTHETIC_SOURCES = "none (synthetic)"


def sources_html(item: Any) -> str:
    """**A workspace's sources, one form wherever they show** (Research ›
    Datasets' table and Training's dataset details): each declared release
    (`sources_text`); `SYNTHETIC_SOURCES` for a synthetic build; else the
    record's own missing state, "Not recorded" (no split manifest, a manifest
    that binds no record, or a build from files that declares no release)
    apart from "Unknown" (the record was not read): the fingerprint cell's
    word for the same cause. Never "—", which means "no value here"."""
    text = sources_text(_get(item, "declared_releases"))
    if text:
        return e(text)
    origin = origin_of(item)
    if origin == "synthetic":
        return e(SYNTHETIC_SOURCES)
    record = _get(item, "dataset_record") or {}
    unrecorded = (origin == "files" or _get(item, "manifest") == "absent"
                  or isinstance(record, Mapping) and record.get("binding") == "unrecorded")
    return status_mark("neutral", NOT_RECORDED if unrecorded else UNKNOWN)


def built_date(build: Optional[Mapping[str, Any]]) -> Optional[str]:
    """A recorded build's time as "2026-10-15 09:41", its zone said once in
    its key or header ("Built (UTC)"), else None (`built_text`)."""
    built = built_text(build)
    return built[len("built "):-len(" UTC")] if built else None


def pair_text(first: Any, second: Any) -> str:
    """Two recorded counts as one value, "200 / 50" (train / val, its key or
    header says which), each "not recorded" where the record holds none
    (`count_text`), and one "not recorded" when neither is."""
    if count_text(first) == count_text(second) == count_text(None):
        return count_text(None)
    return f"{count_text(first)} / {count_text(second)}"


def count_text(value: Any) -> str:
    """A recorded count with thousands separators, or "not recorded" where the
    record holds none (never "—", the skill's §6)."""
    return (f"{value:,}" if isinstance(value, int) and not isinstance(value, bool)
            else NOT_RECORDED.lower())


def dataset_details_html(row: Mapping[str, Any], facts: Optional[Mapping[str, Any]], *,
                         problem: Optional[str] = None, popover: bool = False) -> str:
    """**The one body of a workspace's details** (Research › Datasets' panel
    and Training's "Details ›" popover, each with its own header; M5-3 review
    ARCH-4): a summary list, one fact per row (`summary_list`), its dataset
    fingerprint first (12 hex with Copy, `fingerprint_value`), the disease
    split and the seeds; then what the manifest binds, the files by plain
    names with their 12 hex (`workspaces.manifest_facts`), behind "Manifest
    contents (N files) ›". Beside its table row (Datasets) it leaves out what
    the row shows (built, sources, samples; the skill's §1.2); in a popover,
    which stands alone, it says them too, and the manifest's files are listed
    open (no disclosure inside a popover, the skill's §1.5). `problem` says
    why the manifest was not read (without a path); every value is
    escaped."""
    name = str(row.get("name") or "")
    rows_kv: List[Tuple[str, str]] = []
    fingerprint = row.get("dataset_fingerprint")
    if is_digest(fingerprint):
        rows_kv.append(("Dataset fingerprint", fingerprint_value(
            fingerprint, what=f"dataset fingerprint of {name}")))
    parts = []
    why = None
    if facts is None:
        why = problem or row.get("not_trainable") or "it is not readable"
        parts.append(f'<div class="shep-reason">The manifest was not read: {e(why)}</div>')
    if popover:
        built = built_date(row.get("build"))
        rows_kv += [("Built (UTC)", e(built) if built else NOT_RECORDED.lower()),
                    ("Sources", sources_html(row))]
        if facts is not None:
            rows_kv.append(("Samples (train / val)",
                            e(pair_text(row.get("train_samples"), row.get("val_samples")))))
    if facts is not None:
        if facts.get("train_diseases") is not None:
            rows_kv.append(("Diseases (train / val)",
                            e(pair_text(facts["train_diseases"], facts.get("val_diseases")))))
        for key, label in (("allocation_seed", "Allocation seed"), ("feature_seed", "Feature seed")):
            value = facts.get(key)
            rows_kv.append((label, e(value) if value is not None else NOT_RECORDED.lower()))
    if not row.get("trainable") and row.get("not_trainable") != why:
        # Its own reason, unless the line above already said it.
        rows_kv.append(("Not trainable", e(row.get("not_trainable") or "not trainable")))
    if rows_kv:
        parts.append(summary_list(rows_kv))
    if facts is not None:
        bound = list(facts["bound"])
        files = summary_list([(label, fingerprint_value(digest, what=f"{label} digest",
                                                         copy=False))
                              for label, digest in bound], classes="shep-kv-2")
        title = f"Manifest contents ({len(bound)} file{'' if len(bound) == 1 else 's'})"
        parts.append(f'<div class="shep-ds-manifest"><b>{e(title)}</b>{files}</div>' if popover
                     else f'<details class="shep-more shep-ds-manifest"><summary>{e(title)} ›'
                          f'</summary>{files}</details>')
    return "".join(parts)


def dataset_identity_html(item: Any, *, digits: int = SHORT_HEX, name: bool = True,
                          compact: bool = False, marks: str = "") -> str:
    """One workspace's identity: name; "built … UTC" or "build time not recorded ·
    sources …"; the dataset fingerprint; its origin and record
    (`dataset_marks_html`: amber "synthetic", "record verified" or the
    record's own missing or contradicted state); "not trainable: <why>" when
    it is not (`not_trainable_chip`); then a page's own `marks` (HTML: Test's
    "serving reads this workspace"), in the one marks group, which moves to a
    line as a whole (L7 SW-LK7: a mark after the group parted from it)."""
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
    group = dataset_marks_html(item)
    if not _get(item, "trainable", False):
        group += not_trainable_chip(item)
    group += marks
    if group:
        parts.append(f'<span class="shep-ds-marks">{group}</span>')
    sep = " " if compact else SEP
    return f'<span class="shep-ds">{sep.join(parts)}</span>'


def short_reason(text: Any, limit: int = 80) -> str:
    """A reason shortened for a cell, on one line; the full text goes in
    ``title`` or a details panel."""
    flat = " ".join(str(text or "").split())
    if len(flat) <= limit:
        return flat
    return flat[:max(end for end in grapheme_ends(flat) if end <= limit - 1)] + "…"


def reason_lead(text: Any) -> str:
    """**A reason's plain words, without its trailing reference**: a refused
    read says "the file cannot be read as a checkpoint; Ref: <the loader's
    words>" (`checkpoint_paths.refusal_line`; the WebUI skill's §5, an
    internal code only after a plain label), and a row says the words before
    the reference; the whole reason is its details'."""
    return " ".join(str(text or "").partition("; Ref: ")[0].split())


# ---------------------------------------------------------------------------
# The notice: a setting's state, said once per screen (one renderer)
# ---------------------------------------------------------------------------
#: **A literal token inside a sentence** (the text brief's §8): an environment
#: variable, a command's option, a file of a kind the dashboard names, or a
#: path; drawn in the monospace token (`literal_html`), never a label.
_LITERAL = re.compile(r"SHEPHERD_[A-Z0-9_]+|(?<![\w-])--[a-z][\w-]*"
                      r"|(?<![\w/])[\w~-]*(?:\.[\w~-]+)*\.(?:lock|json|pt|py|sh|cmd)\b"
                      r"|(?<![\w/])[\w.~-]+(?:/[\w.~-]+)+/?")
#: Where a token may break: before its own separators, never inside a word,
#: and never with a hyphen drawn (`token_html`).
_TOKEN_BREAKS = "/._-"
#: A notice's detail row (`server_settings.Notice.details`): "Term: text".
_ROW = re.compile(r"^([A-Z][^:.!?]{0,31}):\s+(\S.*)$")
#: The word of the disclosure every notice's why and how sit behind.
DETAILS_WORD = "Details ›"


def _pieces(text: str) -> List[str]:
    """`text` cut before each of its own separators (`_TOKEN_BREAKS`, the
    first of a run), a command-line option (a word that starts with "-")
    kept as one piece: where `token_html` and `name_pieces_html` may break."""
    pieces, start, option = [], 0, False
    for i, ch in enumerate(text):
        if ch == " ":
            option = False
            continue
        first = i == 0 or text[i - 1] == " "
        if first:
            option = ch == "-"
        if (i and ch in _TOKEN_BREAKS and text[i - 1] not in _TOKEN_BREAKS
                and (first or not option)):
            pieces.append(text[start:i])
            start = i
    pieces.append(text[start:])
    return [piece for piece in pieces if piece]


def name_pieces_html(name: Any) -> str:
    """**A name in the body's type that breaks like a token** (the text
    brief's §8): escaped, each piece between its own separators kept on one
    line and a ``<wbr>`` between pieces, so its narrowest is its longest
    piece, never an em floor (the sweep's L7, SW-LK3: Datasets' Name column
    held 11.5em for "demo" and wrapped In use and the headers at 1850 x 960),
    and it never reads hyphenated; a piece longer than its box breaks only as
    the last resort (``.shep-brk``)."""
    return ('<span class="shep-brk">'
            + "<wbr>".join(f"<span>{e(piece)}</span>" for piece in _pieces(str(name or "")))
            + "</span>")


def token_html(token: Any, classes: str = "") -> str:
    """**One literal token** (a path, a file name, a variable) escaped and in
    code style, breakable only before its own separators: its pieces each
    kept on one line (a span, ``white-space: nowrap``: the browser's own
    break after a hyphen is not taken) with a ``<wbr>`` between them, so no
    hyphen is drawn and a copy keeps the token whole (never U+200B). A
    command-line option (a word that starts with "-") is one piece: a
    command breaks at the space before it, never inside it (the sweep's L7,
    SW-LK7: "--generate-samples" broke as "--generate" / "-samples" and read
    as two options). `classes` are added to the token's own."""
    text = "" if token is None else str(token)
    names = " ".join(("shep-tok", *classes.split()))
    return (f'<code class="{names}">'
            + "<wbr>".join(f"<span>{e(piece)}</span>" for piece in _pieces(text)) + "</code>")


def literal_html(text: Any) -> str:
    """`text` escaped, with each literal token in it (`_LITERAL`) drawn by
    `token_html`: the one way a sentence of a notice, a step or a Ref line
    shows a path or a variable."""
    words, out, at = "" if text is None else str(text), [], 0
    for found in _LITERAL.finditer(words):
        out.append(e(words[at:found.start()]))
        out.append(token_html(found.group(0)))
        at = found.end()
    out.append(e(words[at:]))
    return "".join(out)


def _as_notice(notice: Any) -> Optional[Dict[str, Any]]:
    """A notice as `notice_block` reads it: the dict a page keeps
    (`workspaces.notice_view`), a `server_settings.Notice`, or a plain string
    (an older caller's whole text: no details, no link); None or "" for none."""
    if not notice:
        return None
    if isinstance(notice, str):
        return {"key": "", "kind": "neutral", "summary": notice, "short": notice, "action": "",
                "target": "", "details": (), "ref": (), "cause": None}
    if hasattr(notice, "_asdict"):
        notice = {**notice._asdict(), "kind": "neutral"}
    found = dict(notice)
    found.setdefault("kind", "neutral")
    found["kind"] = found["kind"] if found["kind"] in STATUS_MARKS else "neutral"
    return found


def _marked(kind: str, inner: str, classes: str, attrs: str = "") -> str:
    """A line led by a status mark in a column of its own, so its wrapped
    lines align with its words, never under the mark (the text brief's §4):
    the mark's shape (hidden from a screen reader, which reads the words),
    then the words, coloured by the state's token as `status_mark` colours
    them. Its text reads shape, space, words (`MARK_SPACE`)."""
    shape, token = STATUS_MARKS[kind]
    mark = (f'<span class="shep-nb-mark shep-st-{token}" aria-hidden="true">'
            f'{shape or ""}{MARK_SPACE}</span>')
    return f'<p class="{classes}"{attrs}>{mark}<span class="shep-nb-text">{inner}</span></p>'


def marked_line(kind: str, words: Any) -> str:
    """**A state said as a sentence** (more than a state's word: "Nothing is
    set yet. …"): a reason line (``shep-reason``, at the measure) led by the
    state's mark in a column of its own (`_marked`), so a long sentence wraps
    at the measure, its lines on its words' edge; `status_mark` is for a
    state's word in a line of other words. `words` escaped."""
    return _marked(kind, e(words), "shep-reason")


def _go(notice: Mapping[str, Any], go: bool) -> str:
    """A notice's next step: a link-styled button to its place in the GUI
    (``data-shep-go``: the page; ``data-shep-focus``: the element whose
    heading takes the focus; the shell's head script opens it, from its own
    list of places), or its words as a sentence when `go` is off or it names
    no place."""
    action = str(notice.get("action") or "")
    if not action:
        return ""
    page, _, focus = str(notice.get("target") or "").partition("#")
    if not go or not page:
        return f'<span class="shep-nb-act">{e(action)}.</span>'
    focus_attr = f' data-shep-focus="{e(focus)}"' if focus else ""
    return (f'<button type="button" class="shep-link shep-nb-go" data-shep-go="{e(page)}"'
            f'{focus_attr}>{e(action)}</button>')


def _detail_rows(notices: Sequence[Mapping[str, Any]]) -> Tuple[str, str]:
    """The Details of `notices`, once each: their term rows as one summary
    list (each value's tokens in code style), and one Ref line of their
    references, each once."""
    rows, refs = [], []
    for notice in notices:
        for row in notice.get("details") or ():
            found = _ROW.match(str(row))
            rows.append((found.group(1), literal_html(found.group(2))) if found
                        else ("", literal_html(row)))
        refs += [ref for ref in notice.get("ref") or () if ref not in refs]
    kv = summary_list(rows, classes="shep-nb-kv") if rows else ""
    ref = (f'<p class="shep-nb-ref">Ref: {", ".join(literal_html(x) for x in refs)}</p>'
           if refs else "")
    return kv, ref


def _details(notices: Sequence[Mapping[str, Any]], *, grouped: bool = False) -> str:
    """The notices' "Details ›": their rows and one Ref line; none when they
    hold no row (a Ref line alone is not worth a disclosure: the notice's link
    leads to where its why is said). `grouped` (several notices said as one
    block): each notice's rows under its setting's name (the notice's
    `name`; its state is said once, on its line above), so a row is read as
    the fact of the setting it explains; one Ref line under them all."""
    kv, ref = _detail_rows(notices)
    if not kv:
        return ""
    if grouped and len(notices) > 1:
        kv = "".join((f'<p class="shep-nb-sub">{e(n["name"])}</p>' if n.get("name") else "") + rows
                     for n in notices for rows in [_detail_rows([n])[0]] if rows)
    return (f'<details class="shep-more shep-nb-more"><summary>{e(DETAILS_WORD)}</summary>'
            f'{kv}{ref}</details>')


def _shared_step(notices: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, str]]:
    """**The one next step of several notices** (`notices_block`): when every
    notice's step leads to one place, that place once, in the words the
    settings' source gives several notices (`server_settings.SEVERAL_ACTIONS`
    for their one action, else `PLACE_ACTIONS` for the place), never words
    built here; None when their places differ or a step has no words."""
    targets = {str(n.get("target") or "") for n in notices}
    actions = list(dict.fromkeys(str(n.get("action") or "") for n in notices))
    if len(targets) != 1 or not all(actions):
        return None
    target = next(iter(targets))
    words = (SEVERAL_ACTIONS.get(actions[0], actions[0]) if len(actions) == 1
             else PLACE_ACTIONS.get(target))
    return {"action": words, "target": target} if words else None


def _keys(notices: Iterable[Mapping[str, Any]]) -> str:
    return " ".join(dict.fromkeys(str(n["key"]) for n in notices if n.get("key")))


def notice_block(notice: Any, *, consequence: Optional[str] = None, also: Iterable[Any] = (),
                 steps: Iterable[str] = (), go: bool = True, details: bool = True) -> str:
    """**The one notice** (a setting's state, said once per screen where it
    empties a region; the text brief's §1 and §10): one line, its status mark
    in a column of its own, the state (`summary`), what it means here
    (`consequence`, the page's own words, one short sentence) and the next
    step as a link to its place in the GUI (`_go`); the why and how one click
    away, under "Details ›" (its term rows and one Ref line). `also`: notices
    that follow from this one (the GPU lock's, from the workspaces
    folder's): their details join this one's and they are never drawn apart
    (``data-notice-also``). `steps`: further steps, the action then step 1
    of a numbered list (each step's tokens in code style). Every value is
    escaped. ``data-notice`` names the key, so a screen can count each
    notice once. `details` off: its "Details ›" is said by the region that
    draws it, in that region's one disclosure (`notice_rows_html`: a card
    whose own reason has a Details › of its own, two "Details ›" a line
    apart; the sweep's L7, C1)."""
    found = _as_notice(notice)
    if found is None:
        return ""
    extra = [n for n in map(_as_notice, also) if n and n.get("key") != found.get("key")]
    kind = found["kind"]
    act = _go(found, go)
    steps = [str(step) for step in steps if step]
    inner = f'<span class="shep-st-w shep-st-{STATUS_MARKS[kind][1]}">{e(found["summary"])}</span>'
    if consequence:
        inner += f' <span class="shep-nb-then">{e(consequence)}</span>'
    if act and not steps:
        inner += f" {act}"
    keys = f' data-notice="{e(found.get("key") or "")}"' if found.get("key") else ""
    also_keys = _keys(extra)
    if also_keys:
        keys += f' data-notice-also="{e(also_keys)}"'
    listed = "".join(f"<li>{literal_html(step)}</li>" for step in steps)
    ol = (f'<ol class="shep-nb-steps">{f"<li>{act}</li>" if act else ""}{listed}</ol>'
          if steps else "")
    return (f'<div class="shep-nb shep-nb-{kind}"{keys}>'
            f'{_marked(kind, inner, "shep-nb-line")}{ol}'
            f'{_details([found, *extra]) if details else ""}</div>')


def notice_rows_html(notice: Any) -> str:
    """**A notice's Details, for a region's own disclosure** (`notice_block`
    with ``details=False``): its term rows as one summary list and its Ref
    line, as its own "Details ›" holds them; "" for no notice or no row."""
    found = _as_notice(notice)
    if found is None:
        return ""
    kv, ref = _detail_rows([found])
    return f"{kv}{ref}" if kv else ""


def notices_block(notices: Iterable[Any], *, consequence: Optional[str] = None,
                  go: bool = True) -> str:
    """**Several notices said once, as one block** (the Test page's data
    panel: the registry and the workspaces folder): each state on a line of
    its own with its mark, then what they mean here and one next step for
    the block when every notice's step leads to one place (`_shared_step`:
    "Choose them in Settings › Data locations"; each its own after its state,
    else); one "Details ›" with each notice's rows under its setting's
    name and one Ref line. One notice is `notice_block`'s; none, ""."""
    found = [n for n in map(_as_notice, notices) if n]
    if len(found) <= 1:
        return notice_block(found[0], consequence=consequence, go=go) if found else ""
    shared = _shared_step(found)
    lines = []
    for n in found:
        inner = (f'<span class="shep-st-w shep-st-{STATUS_MARKS[n["kind"]][1]}">'
                 f'{e(n["summary"])}</span>')
        if shared is None and _go(n, go):
            inner += f" {_go(n, go)}"
        lines.append(_marked(n["kind"], inner, "shep-nb-line"))
    after = " ".join(part for part in (
        f'<span class="shep-nb-then">{e(consequence)}</span>' if consequence else "",
        _go(shared, go) if shared else "") if part)
    next_line = f'<p class="shep-nb-next">{after}</p>' if after else ""
    kind = "error" if any(n["kind"] == "error" for n in found) else "neutral"
    return (f'<div class="shep-nb shep-nb-{kind}" data-notice="{e(_keys(found))}">'
            f'{"".join(lines)}{next_line}{_details(found, grouped=True)}</div>')


#: **A disabled control's state, said one way on every page**: the control's
#: name, this word, then why ("Start disabled: no workspaces folder is set.",
#: `disabled_words`), never "unavailable" on one page and "disabled" on the
#: next; a reason line a page draws itself leads with `DISABLED_MARK`, the mark
#: a relation to a setting not yet set leads with (`see_above`).
DISABLED = "disabled"
#: The mark a disabled control's reason leads with where its page sets the
#: mark in a column of its own: the neutral mark (grey, the skill's §6), its
#: shape and colour as `_marked` draws them, never a page's own glyph.
DISABLED_MARK = (f'<span class="shep-nb-mark shep-st-{STATUS_MARKS["neutral"][1]}" '
                 f'aria-hidden="true">{STATUS_MARKS["neutral"][0]}</span>')


def disabled_words(control: str, cause: str) -> str:
    """"<control> disabled: <cause>": a disabled control's state and why,
    the one form every page says it in (`DISABLED`)."""
    return f"{control} {DISABLED}: {cause}"


def relation_words(control: str, notice: Any, *, where: Optional[str] = None) -> str:
    """`see_above`'s words as plain text, for a reason a page gives as text:
    the control disabled, the notice's short words, and where the notice is
    ("Start disabled: no results registry is set (see Data).")."""
    found = _as_notice(notice)
    if found is None:
        return ""
    return disabled_words(control, found["short"]) + (f" (see {where})" if where else "") + "."


def see_above(control: str, notice: Any, *, where: Optional[str] = None,
              details: bool = False) -> str:
    """**A control's relation to the notice its screen shows** (said once,
    `notice_block`): one line beside the control it explains, its status
    mark, the control disabled and the notice's short words ("Start
    disabled: no workspaces folder is set.", `relation_words`), with where
    the notice is when it is not in view of the control (`where`: "see
    above", "see Data"). With `details`, the notice's "Details ›" under it
    (where the screen shows the notice's line elsewhere but not its
    details). ``data-notice-rel`` names the notice: a relation is not the
    notice said again."""
    found = _as_notice(notice)
    if found is None:
        return ""
    words = relation_words(control, found, where=where)
    rel = f' data-notice-rel="{e(found["key"])}"' if found.get("key") else ""
    inner = f'<span class="shep-st-w shep-st-{STATUS_MARKS[found["kind"]][1]}">{e(words)}</span>'
    line = _marked(found["kind"], inner, "shep-reason shep-nb-rel", rel)
    return line + (_details([found]) if details else "")


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


#: A checkpoint with no ledger beside it, as its recorded results say (R-27).
NO_LEDGER_WORDS = "none kept"


def recorded_tests_text(recorded: Optional[RecordedTests]) -> str:
    """"Recorded results: 3 · 1 not readable; runs differ for 1; comparison
    incomplete for 1": states, never a number of a result; a record that
    cannot be read is never counted as one (K8-6e.1) and a record whose
    comparison is incomplete is never shown as agreeing. "none in ledger"
    only when the ledger holds neither; with no ledger beside the file, "none
    kept" (R-27: never the internal word "no ledger"). One " ·
    " (the skill's §5)."""
    if recorded is None:
        return "Recorded results: not read"
    if recorded.state == "no_ledger":
        return f"Recorded results: {NO_LEDGER_WORDS}"
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
    (`run_words` then names it whole)."""
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


#: A run directory's own id, after its name (`propose_run_dir`'s 6 hex).
_RUN_ID = re.compile(r"_([0-9a-f]{6})$")
#: The longest architecture label a run's option carries (`run_parts`): a
#: layout directory is listed whatever its name (K8-7b), and the label every
#: run of it shares never crowds out the name or time that tells runs apart.
_ARCHITECTURE_CHARS = 8


def run_parts(architecture: Any, run: Any, level: int = 0, *,
              fit: bool = True) -> Tuple[str, Optional[str]]:
    """A run as an option's two parts, the distinguishing token first: its
    name when it was given one, else its start time, else (a directory not
    named by its start time) its whole name; then the architecture (for an
    option, `fit`, past `_ARCHITECTURE_CHARS` shortened in its middle; whole
    in the details), and the start time after a name. An `architecture` of
    None is a run a record names by its directory alone (a parent's, a
    family's root run): nothing is said in its place. `level` tells alike
    runs apart (`distinct_names`; `model_list.run_options`): 1 gives the time
    to the second, 2 adds the run's own 6-hex id."""
    label = "" if architecture is None else architecture_label(architecture)
    arch = middle_cut(label, _ARCHITECTURE_CHARS) if fit else label
    started = run_started(run, seconds=level >= 1)
    if not started:
        return str(run), arch or None
    found = _RUN_ID.search(str(run))
    rid = f" {found.group(1)}" if level >= 2 and found else ""
    name = run_name(run)
    if name and name != "run":
        return name, f"{arch} {started}{rid}".strip()
    return started, f"{arch}{rid}".strip() or None


def run_words(architecture: Any, run: Any, *, fit: bool = True) -> str:
    """**A run as a person names it, its distinguishing token first**, on
    every page that names a run (the Run selects, the Fine-tune base, the
    Training run view, a model's title and details on Models, Test and the
    Diagnosis card (`model_name_line`), a model's parent and family on Models
    and in Results): its name, then the architecture and its start time,
    "ui-live-stop-02 · GATv2 2026-10-24 17:39" from
    ``gatv2/20261024-173909_ui-live-stop-02_c1d9cf``; an unnamed run ("run")
    is its start time and architecture, "2026-10-24 16:38 · GATv2"; a
    directory not named by its start time is its whole name, "my-old-run ·
    HGT". A run a record names without its layout (`architecture` None: a
    parent's or a family's root run) is the same words without one,
    "ui-live-stop-02 · 2026-10-24 17:39", "2026-10-24 16:38". One separator.

    Two lengths of the one rule: as an option or a title (`fit`, the
    default), at most `OPTION_CHARS` characters, a long name or architecture
    shortened in its middle, the run's directory in the chosen item's
    details; in those details and a sentence naming the run (`fit` False),
    the same parts whole, never an option's cut (the skill's §5, §8)."""
    what, rest = run_parts(architecture, run, fit=fit)
    if fit:
        return fit_option(what, rest)
    return f"{what} · {rest}" if rest else what


def run_names(runs: Mapping[Hashable, Tuple[Any, Any]]) -> Dict[Hashable, str]:
    """**Runs shown together, each by its words told apart from the others'**
    (every Run select's options, `model_list.run_options`, and the Models
    table's rows), by key, from ``{key: (architecture, run directory's
    name)}``: each run's words at an option's length (`run_words`), a name cut
    alike with another's keeping its start through where they differ
    (`fit_apart`: "cpb-dis-ful…" beside "cpb-dis-fro…", where both read
    "cpb-dis…disease" and only their start minute told them apart), and runs
    whose words still read alike their start time to the second, then their
    own id (`distinct_names`). Words nothing recorded tells apart are the
    caller's to number (`fit_options`)."""
    return distinct_names(runs, lambda parts, level: run_parts(parts[0], parts[1], level),
                          together=fit_apart)


def run_time(run_id: Any, *, seconds: bool = False) -> Optional[str]:
    """A run's time from its run id, as the pages name a run: "2026-10-17
    13:40 UTC" (with `seconds`, "2026-10-17 13:40:07 UTC") from a report store
    run id (`create_report_dir` names it by its UTC time), else None."""
    match = _REPORT_RUN_ID.match(str(run_id or ""))
    if not match:
        return None
    y, mo, d, h, mi, s = match.groups()
    return f"{y}-{mo}-{d} {h}:{mi}{':' + s if seconds else ''} UTC"


def distinct_names(items: Mapping[Hashable, Any], name_at: Callable[[Any, int], Any], *,
                   top: int = 2,
                   together: Optional[Callable[[Dict[Hashable, Any]], Dict[Hashable, str]]] = None
                   ) -> Dict[Hashable, str]:
    """Names that read alike, told apart: each item's name, by its key, at
    level 0 (``name_at(item, level)``; with `together`, what it makes of every
    item's at once, `run_names`' `fit_apart`), a level higher only for the
    items whose names read alike, until no two do or each of them is at `top`
    (runs named by their time to the minute, then to the second, then with
    the run id's own last part: Models › Results' `row_names`). Names still
    alike at `top` are the caller's to tell apart."""
    levels = dict.fromkeys(items, 0)
    while True:
        names = {key: name_at(item, levels[key]) for key, item in items.items()}
        if together is not None:
            names = together(names)
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
    model spec among them, its reason the record's), or "Record unknown" for
    a file whose record was never read (its file could not be read: the
    skill's §6 Unknown, a state with its own word, never a dash)."""
    record = _get(item, "record")
    if record == "model":
        if _get(item, "learning_stage") == "base":
            return "Base"
        task = _get(item, "task")
        if task in TASK_NAMES:
            return TASK_NAMES[task]
    elif record is None:
        return f"Record {UNKNOWN.lower()}"
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


#: **A run's operation in a word** (the shared model list's run rows, D04,
#: and the Models detail's Operation row: one spelling on a screen), from its
#: record's `operation`: a run started from nothing, one resumed, one
#: extended, a branch and a fine-tune of a parent.
RUN_OPERATION_WORDS = {"new": "New run", "resume": "Resume", "extend": "Extend",
                       "branch": "Branch", "finetune": "Fine-tune"}


def operation_words(item: Any) -> str:
    """**What a run did, in a few words** (the shared model list's run rows,
    D04, and the Models detail's Operation row): `RUN_OPERATION_WORDS` of its
    record's operation, and for a fine-tune what its weight transfer records
    frozen, in one of three words (`run_plan.recorded_frozen_parts`, the one
    reading of that record): "Fine-tune, full" (nothing frozen), "Fine-tune,
    frozen" (the graph side, `recipe.FROZEN_PARTS["graph_side"]`), "Fine-tune,
    partly frozen" (any other parts); a fine-tune that records no transfer is
    "Fine-tune". A model record that names no operation says "Not recorded", a
    record not readable or not read "Unknown" (the skill's §6), never a
    dash."""
    if _get(item, "record") != "model":
        return UNKNOWN
    operation = _get(item, "operation")
    if not isinstance(operation, str) or not operation:
        return NOT_RECORDED
    words = RUN_OPERATION_WORDS.get(operation, operation)
    if operation != "finetune":
        return words
    frozen = recorded_frozen_parts(_get(item, "weight_transfer"))
    if frozen is None:
        return words
    if not frozen:
        return f"{words}, full"
    if frozen == tuple(sorted(FROZEN_PARTS["graph_side"])):
        return f"{words}, frozen"
    return f"{words}, partly frozen"


def entry_place(item: Any) -> Optional[str]:
    """**Where an entry the listing could not list sits, in words** (BACKLOG
    69 L1): a directory under ``runs/`` ("a folder in runs/") or a run
    directory ("a run folder in runs/gatv2/"), told by its identity (an
    entry for a directory is keyed as its run, `CheckpointItem.key` and
    `run_key`; one directly under ``runs/`` by its own name alone), never by
    upper-casing a directory's name as if it named an architecture. None for
    a file, which its run and its own name place."""
    key, run_key = _get(item, "key"), _get(item, "run_key")
    if not isinstance(key, str) or key != run_key:
        return None
    layout = str(_get(item, "architecture") or "")
    if key.startswith(f"{layout}/"):
        return f"a run folder in runs/{layout}/"
    return "a folder in runs/"


def configuration_text(item: Any) -> str:
    """A listed model's configuration (README §5, axis 1): its encoder and
    whether its spec is the reference recipe, "GATv2 · reference recipe" or
    "GATv2 · recipe changed in 3 fields" (`differences`: which fields is axis
    2's, shown where the full list fits). A record refused or never read says
    which."""
    return " · ".join(configuration_facts(item))


def configuration_facts(item: Any) -> Tuple[str, ...]:
    """`configuration_text`'s facts, each a fact of its line (`chain_html`)."""
    record = _get(item, "record")
    if record is None:
        return ("Not read",)
    differences = _get(item, "differences")
    if record != "model" or differences is None:
        return ("Not readable",)
    if not differences:
        return (architecture_label(_get(item, 'encoder')), "reference recipe")
    count = len(differences)
    return (architecture_label(_get(item, 'encoder')),
            f"recipe changed in {count} field{'s' if count != 1 else ''}")


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


def model_name_line(item: Any, *, fit: bool = True) -> str:
    """**A listed file by its run and how far it trained**, the one line every
    page titles a model by (the shared list's rows, Models' detail header,
    the Diagnosis card) and its details say whole (Model Facts, the Test
    page's model, the load plan's From and To, which compare two files, so
    neither run is ever cut alike with the other): its run by the one run words
    (`run_words`; `fit` for a title or a row, whole in details), then its
    progress (`epoch_words`) after a comma, so the line keeps the run words'
    one " · ": "ui-live-stop-02 · GATv2 2026-10-24 17:39, 6 epochs
    completed"."""
    return " · ".join(model_name_facts(item, fit=fit))


def model_name_html(item: Any, *, fit: bool = True) -> str:
    """`model_name_line` drawn as HTML: each of its facts one unit
    (`chain_html`), so a line breaks between them, never inside "GATv2
    2026-10-02 00:00, 1 epoch completed" where a line holds it (the sweep's
    L7 check, C2: Models' details title broke inside the date at 1550,
    1366 and 1280). Every page that draws the name in HTML uses this; plain
    text (a title attribute, an export) keeps `model_name_line`."""
    return chain_html(*model_name_facts(item, fit=fit))


def model_name_facts(item: Any, *, fit: bool = True) -> Tuple[str, ...]:
    """`model_name_line`'s facts, each a fact of its line (`chain_html`): the
    run's distinguishing words, then its architecture and start time with its
    progress; one fact when the run has no second part."""
    what, rest = run_parts(_get(item, "architecture"), _get(item, "run"), fit=fit)
    if fit:
        what, rest = _fitted(what, rest, "", 0)
    progress = epoch_words(item)
    return (what, f"{rest}, {progress}") if rest else (f"{what}, {progress}",)


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
    fingerprint … · matches no listed workspace" (one wording, both presets),
    joined by the shared separator (`SEP`), so a line ends at it, never
    starts with it, and the layout pass hides it where a line breaks."""
    names = trained_on(fingerprint, workspaces)
    fp = fingerprint_html("dataset fingerprint", fingerprint, digits=digits, copy=copy)
    if not is_digest(fingerprint):
        return f'<span class="shep-trained">trained on: {fp}</span>'
    if names:
        listed = ", ".join(f"<b>{e(n)}</b>" for n in names)
        return f'<span class="shep-trained">trained on {listed}{SEP}{fp}</span>'
    return (f'<span class="shep-trained">trained on {fp}{SEP}'
            f'<span class="shep-amber-t">matches no listed workspace</span></span>')


def is_synthetic_training(fingerprint: Any, workspaces: Sequence[Any]) -> bool:
    return any(origin_of(w) == "synthetic" for w in workspaces
               if is_digest(fingerprint) and _get(w, "dataset_fingerprint") == fingerprint)


def unreadable_words(item: Any) -> str:
    """**Why a listed checkpoint shows no metadata, without claiming a read
    that did not happen**: "metadata not readable" for a file whose one read
    was made and failed, or gave no record this build shows (it carries a
    digest or a size, `ListedCheckpoint`); "not read" for an entry never
    opened: a name that is not text, a directory that cannot be listed, a
    file outside the workspace (BACKLOG 66, `read_run_checkpoints`), whose
    reason says why; "not listed again" for a link to a directory listed
    otherwise (B69-L4, `checkpoint_paths.ALIAS_STATE`), whose runs were read
    and are listed there once, never "not read" (the review of 6a14d6b,
    R3-1). Every list, count and detail of a checkpoint says it alike."""
    from src.utils.checkpoint_paths import ALIAS_STATE, is_alias_problem

    if is_alias_problem(_get(item, "problem")):
        return ALIAS_STATE
    read = is_digest(_get(item, "sha256")) or _get(item, "size_bytes") is not None
    return "metadata not readable" if read else "not read"


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


def dataset_exception(item: Any) -> str:
    """**A listed file's dataset, said only where it is not the workspace's
    current one** (the skill's §1.2: the workspace's dataset is said once,
    under its select, and a row trained on it says nothing of it): "▲ Another
    dataset" where the file records a dataset the workspace no longer holds
    (`trained_on_this_dataset` False), with "synthetic" where a listed
    synthetic workspace holds it; "○ Dataset not recorded" where the file
    records none; "○ Dataset cannot be verified" where it records one the
    workspace's own fingerprint could not be compared with. "" for a file
    trained on the workspace's current dataset. Which dataset it is, its
    fingerprint and the workspaces holding it are the file's details
    (`trained_on_html`)."""
    if _get(item, "trained_on_this_dataset") is True:
        return ""
    if not is_digest(_get(item, "dataset_fingerprint")):
        return status_mark("neutral", f"Dataset {NOT_RECORDED.lower()}")
    if _get(item, "trained_on_this_dataset") is False:
        return status_mark("warn", "Another dataset")
    return status_mark("neutral", f"Dataset {CANNOT_BE_VERIFIED.lower()}")


#: **A listed file trained on the chosen workspace's current dataset**, by
#: its relation (`dataset_relation_html`): that dataset is said once, under
#: the workspace's select.
CURRENT_DATASET = "the selected workspace's current dataset"


def dataset_relation_html(item: Any, workspaces: Sequence[Any]) -> str:
    """**A listed file's dataset in its details**, by its relation to the
    chosen workspace (the Models detail's Dataset row, the Test page's
    chosen model): `CURRENT_DATASET` where it is that workspace's current
    dataset; elsewhere its exception (`dataset_exception`) and which dataset
    it is (`trained_on_html`: the listed workspaces holding it, its
    fingerprint), with "synthetic" where a listed synthetic workspace holds
    it; a file that records no dataset fingerprint, its exception alone ("○
    Dataset not recorded", said once)."""
    exception = dataset_exception(item)
    if not exception:
        return e(CURRENT_DATASET)
    fingerprint = _get(item, "dataset_fingerprint")
    if not is_digest(fingerprint):
        # Nothing to name: the exception says the state, once (the skill's
        # §1.2), never again as trained_on_html's own "not recorded".
        return exception
    return (f"{exception} {trained_on_html(fingerprint, workspaces)}"
            + (" " + chip("synthetic", "amber")
               if is_synthetic_training(fingerprint, workspaces) else ""))


def warning_html(head: str, detail: str) -> str:
    """**A warning's line**: its heading as a condition to watch, by the
    shared mark (`status_mark`, ▲ with the words: the skill's §6), then what
    it means; both escaped. Model Facts' Warnings and the Models detail."""
    return (f'<p class="shep-warn"><b>{status_mark("warn", head)}</b> '
            f'<span>{e(detail)}</span></p>')


# ---------------------------------------------------------------------------
# Model Facts
# ---------------------------------------------------------------------------
class ModelFacts(NamedTuple):
    """What Model Facts shows for one model, from what the server listed.
    `task` is what its record says it is (`model_kind_text`) and
    `configuration` its encoder and relation to the reference recipe
    (`configuration_text`)."""

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
    #: What serving reads, as ``(label, escaped HTML)`` rows, each a row of
    #: Data's summary list (the skill's §5: a fact a row, never a chain of
    #: " · "): the workspace it reads and how its dataset stands to this
    #: file's (`facts_from_checkpoint`), and on the Diagnosis card the graph,
    #: the shortest paths and the dataset record it serves, a verified one by
    #: the shared mark (`status_mark`).
    serving_rows: Tuple[Tuple[str, str], ...] = ()
    task: Optional[str] = None
    configuration: Optional[str] = None
    #: `name_line` and `configuration` as their facts (`model_name_facts`,
    #: `configuration_facts`), each drawn as one unit of its line
    #: (`chain_html`); empty: the line is drawn as its text.
    name_facts: Tuple[str, ...] = ()
    configuration_facts: Tuple[str, ...] = ()
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
    #: A listed file trained on the chosen workspace's current dataset
    #: (`facts_from_checkpoint`'s `listed`): its list says that dataset once,
    #: under the workspace select, so the facts name it by that relation
    #: (`CURRENT_DATASET`), never its fingerprint again (the skill's §1.2).
    current_dataset: bool = False


def facts_from_checkpoint(item: Any, workspaces: Sequence[Any], *,
                          recorded: Optional[RecordedTests] = None,
                          serving_workspace: Any = None, listed: bool = False) -> ModelFacts:
    """Model Facts for a checkpoint the server listed. `serving_workspace` is the
    listed workspace the pipeline serves now, if any, for the Data line.
    `listed`: the facts of a file chosen in a list of the chosen workspace's
    files (the Diagnosis load drawer, Models), not of the model served: no
    synthetic banner, which speaks of the served workspace (the list's
    workspace line says "synthetic"), and a file trained on that workspace's
    current dataset is said so by its relation (`ModelFacts.current_dataset`)."""
    fingerprint = _get(item, "dataset_fingerprint")
    names = trained_on(fingerprint, workspaces)
    matched = [w for w in workspaces if _get(w, "name") in names]
    synthetic = any(origin_of(w) == "synthetic" for w in matched)
    sources = sources_text(_get(matched[0], "declared_releases")) if matched else ""
    figure = validation_figure(item)
    architecture, run, filename = (_get(item, "architecture"), _get(item, "run"),
                                   _get(item, "filename"))
    warnings = []
    if synthetic and not listed:
        warnings.append(IDENTITY_BANNERS["synthetic"])
    if _get(item, "problem") is not None:
        warnings.append((METADATA_NOT_READABLE, str(_get(item, "problem"))))
    if is_digest(fingerprint) and not names:
        warnings.append((TRAINED_ON_UNLISTED,
                         "The dataset fingerprint this checkpoint records equals no listed "
                         "workspace's: its workspace was rebuilt or removed since."))
    if not is_digest(fingerprint) and _get(item, "problem") is None:
        warnings.append((DATASET_NOT_RECORDED,
                         "This checkpoint records no dataset fingerprint, so what it was "
                         "trained on cannot be told."))
    serving_rows: Tuple[Tuple[str, str], ...] = ()
    if serving_workspace is not None:
        same = (is_digest(fingerprint)
                and _get(serving_workspace, "dataset_fingerprint") == fingerprint)
        serving_rows = (("Serving", e(f"reads workspace {_get(serving_workspace, 'name')}: "
                                      + ("the same dataset" if same else "another dataset"))),)
    return ModelFacts(
        name_line=model_name_line(item),
        name_facts=model_name_facts(item),
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
        # Task and Configuration come from the file's model record, the task
        # in the words every page describes a model by (`model_kind_text`: a
        # task model with a frozen graph backbone named as such, decision F1);
        # an item built without reading one (the Diagnosis card's served item,
        # whose card fills Task from the status) shows neither row.
        task=model_kind_text(item) if _get(item, "record") is not None else None,
        configuration=(configuration_text(item) if _get(item, "record") is not None
                       else None),
        configuration_facts=(configuration_facts(item) if _get(item, "record") is not None
                             else ()),
        validation_read=record_read(item),
        current_dataset=listed and _get(item, "trained_on_this_dataset") is True,
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
#: The class of a summary list that sits among a form's fields (Test's R10
#: evidence, batch size and device rows): its keys are drawn as the form's
#: field labels, one label style per page (R-16), not a summary's grey keys.
FIELD_ROWS = "shep-kv-field"


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


def details_list(rows: Iterable[Tuple[str, str]], *, of: Any = None,
                 titles: Optional[Mapping[str, str]] = None, classes: str = "") -> str:
    """**The one "Details ›" disclosure of a summary list** (the skill's
    §1.5: low-priority detail one click away, one level, never inside another
    disclosure or a popover): a closed ``<details class="shep-more">`` over
    `summary_list`'s rows (`titles` and `classes` as there), wherever a line
    names a few of many (`named_words`) and the rest are one click away.
    Every dashboard ``gr.HTML`` keeps it as the reader left it when it is
    drawn again (`KEEP_OPEN_JS`, `shell.plain_html`), and where the region
    can show another thing, the thing it is of is named: `of` on the
    disclosure itself, or `keep_as` around it."""
    named = keep_as(of) if of is not None else ""
    return (f'<details class="shep-more"{named}><summary>Details ›</summary>'
            f'{summary_list(rows, titles=titles, classes=classes)}</details>')


#: **Disclosures as the reader left them, across redraws**: the start of the
#: ``js_on_load`` of every ``gr.HTML`` of the dashboard (`shell.plain_html`,
#: and the chart figures), before the element's own script, whatever event
#: draws it again (a poll, a tick, a render, or the reader's own act: a field
#: edited, a Load, an order changed) and whatever it then holds (the
#: verification of fcc3dea: the "Set by you" lines, the Continue drawer's
#: preview and Models' details closed under the reader's own edits). Gradio
#: 6.30 draws HTML again in place and sets each element's attributes to the
#: new HTML's, so a disclosure the reader opened was drawn closed again by the
#: next drawing (T06-C: the Test job card's Details; T08-flow-2: Training's
#: "Details ›" popovers). Both kinds are kept: a ``details`` (its ``open``)
#: and a button with ``aria-expanded`` and ``aria-controls`` naming its
#: popover in the same HTML (the button's ``aria-expanded``, the popover's
#: ``hidden``; how it opens and closes is its own script's). The reader's own change, by any
#: means (a click, Enter or Space, Escape, a click outside, focus leaving), is
#: told from a drawing's by a mark this script sets on each disclosure, which
#: a drawing takes off: a change seen on a marked disclosure is the reader's,
#: and is kept; a disclosure without it was drawn again, and gets the
#: reader's last choice back, with no focus moved (an open by the reader's
#: own click focuses, never this); only where the drawing replaced the
#: control the reader's focus was on in it, and left focus nowhere, is focus
#: put back on the same place, without scrolling (so Escape still reaches the
#: popover); focus the reader moved away, to the page itself included (a
#: click on text), is never brought back. A disclosure is the same one while
#: the region names the same thing (`keep_as`, on an element around it;
#: none: the region's one thing), its summary's or its button's words are the
#: same and so is its place among those; a choice for one no longer drawn
#: (another run, another job) is dropped. **A region's default**: a
#: disclosure the reader has not opened or closed is drawn as its HTML says,
#: unless the region's own script (which runs after this one) gives a default
#: the HTML cannot say, one that hangs on the window: a function
#: ``element.shepKeepDefault(d)`` answering true (open), false (closed) or
#: undefined (as drawn), asked each time such a disclosure is drawn, the
#: drawing the region mounts with included (this script's first pass waits
#: for the region's own script: before, a disclosure already in the HTML as
#: the region mounted was taken as drawn, its default never asked; the review
#: of e1dda3e); the reader's own choice always wins over it (Training's
#: failed-run banner: its output tail, closed on a short side-by-side window;
#: UI ledger V-16). **A region that keeps its own folds** (an element with
#: ``data-shep-folds-own`` around them: the Models family graph's arrows,
#: whose state the server draws from what the operator folded and a search
#: reveals for its own time, `model_list.FAMILY_JS`) is left to its script.
KEEP_OPEN_JS = r"""
(function () {
  var MARK = 'data-shep-kept', chosen = {}, focused = null;
  // The region's default for a disclosure the reader has not acted on.
  function byDefault(d) {
    var given = element.shepKeepDefault;
    return typeof given === 'function' ? given(d) : undefined;
  }
  var CONTROLS = 'a[href], button, input, select, textarea, summary, [tabindex]';
  function popover(b) {
    var p = document.getElementById(b.getAttribute('aria-controls'));
    return p && element.contains(p) ? p : null;
  }
  function isOpen(d) {
    return d.tagName === 'DETAILS' ? d.open : d.getAttribute('aria-expanded') === 'true';
  }
  function setOpen(d, open) {
    if (d.tagName === 'DETAILS') { d.open = open; return; }
    popover(d).hidden = !open;
    d.setAttribute('aria-expanded', open ? 'true' : 'false');
  }
  function words(d) {
    if (d.tagName === 'DETAILS') {
      var s = d.querySelector(':scope > summary');
      return s ? s.textContent : '';
    }
    return d.textContent + '\u0001' + d.getAttribute('aria-controls');
  }
  // A disclosure's controls in order, its summary or its button first.
  function controls(d) {
    var inside = Array.prototype.slice.call(
      (d.tagName === 'DETAILS' ? d : popover(d)).querySelectorAll(CONTROLS));
    return d.tagName === 'DETAILS' ? inside : [d].concat(inside);
  }
  // Each disclosure drawn, by its key: what the region names, its words, its place.
  function disclosures() {
    var seen = {}, all = [];
    element.querySelectorAll('details, [aria-expanded][aria-controls]').forEach(function (d) {
      if (d.tagName !== 'DETAILS' && !popover(d)) { return; }
      // A region that keeps its own folds (the family graph's arrows,
      // `model_list.FAMILY_JS`: the server draws them from its fold state).
      if (d.closest('[data-shep-folds-own]')) { return; }
      var of = d.closest('[data-keep]'), named = of && element.contains(of);
      var key = (named ? of.getAttribute('data-keep') : '') + '\u0000' + words(d);
      seen[key] = (seen[key] || 0) + 1;
      all.push({d: d, key: key + '\u0000' + seen[key]});
    });
    return all;
  }
  function keep() {
    var drawn = {};
    disclosures().forEach(function (x) {
      var d = x.d;
      drawn[x.key] = d;
      if (!d.hasAttribute(MARK)) {
        var want = chosen[x.key] !== undefined ? chosen[x.key] : byDefault(d);
        if (want !== undefined && want !== isOpen(d)) { setOpen(d, want); }
        d.setAttribute(MARK, '');
      } else if (d.__shepOpen !== isOpen(d)) {
        chosen[x.key] = isOpen(d);
      }
      d.__shepOpen = isOpen(d);
    });
    Object.keys(chosen).forEach(function (k) { if (!drawn[k]) { delete chosen[k]; } });
    // A drawing that replaced the control the reader's focus was on, and left
    // focus nowhere, puts it back on the same place in the same disclosure.
    var now = document.activeElement, d = focused && drawn[focused.key];
    if (d && !focused.el.isConnected && (!now || now === document.body)
        && (focused.at === 0 || isOpen(d))) {
      var to = controls(d)[focused.at];
      if (to) { to.focus({preventScroll: true}); }
    }
  }
  element.addEventListener('focusin', function (e) {
    focused = null;
    disclosures().some(function (x) {
      var at = controls(x.d).indexOf(e.target);
      if (at >= 0) { focused = {el: e.target, key: x.key, at: at}; }
      return at >= 0;
    });
  });
  // Focus that went nowhere is the reader's (a click on text or a chart),
  // unless a drawing removed its control: the browser tells the two alike (a
  // focusout, the control still in the page), so the place is forgotten once
  // the script running now (a drawing's, which removes it) is done, where
  // the control is still in the page and focus is not on it.
  element.addEventListener('focusout', function (e) {
    var was = focused;
    if (e.relatedTarget) { focused = null; return; }
    queueMicrotask(function () {
      if (was && focused === was && was.el.isConnected && document.activeElement !== was.el) {
        focused = null;
      }
    });
  });
  new MutationObserver(keep).observe(element, {childList: true, subtree: true, attributes: true,
    attributeFilter: ['open', 'hidden', 'aria-expanded', MARK]});
  // The drawing the region mounts with, once its own script (after this one,
  // in the same js_on_load) has given its default.
  queueMicrotask(keep);
})();
"""


#: **The one trigger of a region's acts** (a ``js_on_load`` of `shell.plain_html`):
#: a click on an element with ``data-act`` sends the region's ``click`` with
#: ``{act, key}``, which its listener reads (`event_act`): cross-links such as
#: Training's "Open Test ›", Fine-tune's "Charts on Research › Training" and
#: Models' "Test this model ›", and Training's banner's Dismiss. An act inside
#: an element of the region that says a model list's generation
#: (``data-gen``: Models' "Saved files", `model_list.ListClock`) is sent with
#: it (``gen``).
TRIGGER_JS = r"""
element.addEventListener('click', function (e) {
  var a = e.target.closest ? e.target.closest('[data-act]') : null;
  if (!a) { return; }
  var data = {act: a.getAttribute('data-act'), key: a.getAttribute('data-key') || ''};
  var gen = a.closest('[data-gen]');
  if (gen && element.contains(gen)) { data.gen = gen.getAttribute('data-gen'); }
  trigger('click', data);
});
"""


def event_act(evt: Any) -> Optional[str]:
    """The ``act`` a region's trigger sent (`TRIGGER_JS`; Training's popover's
    own script sends one too), or None."""
    data = getattr(evt, "_data", None)
    return data.get("act") if isinstance(data, Mapping) else None


def keep_as(identity: Any) -> str:
    """The attribute naming **what a region shows** (a run, a job, a
    workspace), put on the element around its disclosures (or on one
    itself): `KEEP_OPEN_JS` keeps a disclosure as the reader left it only
    while it is the same. Every region that can show another thing names
    it, since the keeper runs on every HTML (`shell.plain_html`): the
    selected model's identity and each of its records (Models), the
    selected candidate (Diagnosis), the selected workspace (Datasets), the
    checkpoint the Continue drawer checked, each result row and the results
    compared (Results), as the Test page's job and result and Training's
    workspace, run and failed job (its banner's output tail) do. Unnamed, a
    Details opened for one model was drawn open for the next (the review of
    5033d29)."""
    return f' data-keep="{e(identity)}"'


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
                      copy: bool = True, prefix: bool = False) -> str:
    """**The one digest value**, a summary list's (`summary_list`) and a line's
    (`fingerprint_html` puts its label before it): its 12 hex digits
    (`full`: all 64, as Model Facts' digest rows show them, so the long value
    keeps one line) in bold monospace, the whole value in its title and on an
    icon Copy named for `what` (none with `copy` False, where the same digest
    is copyable once elsewhere, or in its own object's details); when there
    is none, `missing`, the state's own word ("not recorded" where the record
    holds none, "not read" where nothing was read: the skill's §6). With
    `prefix`, where the caller's value is a digest known by its first 12
    digits alone (a report's run id names its model so: the Test page's
    reports table), 12 hex digits are those digits, titled so, with no Copy;
    any other caller's 12 hex digits are no digest, so a malformed value read
    from a record says its missing state, never a valid-looking prefix."""
    if prefix and isinstance(digest, str) and _HEX12.match(digest):
        # Only its first 12 digits are known (a report's run id names its
        # model so): shown as they are, with nothing whole to copy.
        return f'<code class="shep-fpv" title="{e(PREFIX_TITLE)}">{e(digest)}</code>'
    if not is_digest(digest):
        return f'<span class="shep-muted">{e(missing)}</span>'
    shown, cls = (digest, "shep-full") if full else (digest[:digits], "shep-fpv")
    code = f'<code class="{cls}" title="{e(digest)}">{e(shown)}</code>'
    return code + (" " + copy_button(digest, f"the full {what}", icon_only=True) if copy else "")


#: **The label of each learning setting a form offers**, one name per setting:
#: the Training page's base form (`training_console.BASE_FIELDS`, what a new
#: base takes), its Continue drawer's Branch (`training_console.BRANCH_FIELDS`,
#: what a branch may change) and the Fine-tune page's
#: (`finetune_page.FINETUNE_FIELDS`).
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
    # A task model's own settings, which the Continue drawer's Branch offers
    # for a task model's parent (`recipe.branch_settings`).
    "task_loss_weight": "Task loss weight",
    "lp_loss_weight": "LP loss weight",
    "candidate_sample_size": "Candidate sample size",
    "sparse_sample_size": "Sparse sample size",
    "candidate_upsample_size": "Candidate upsample size",
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
#: **What a setting or a model spec field that holds None means**, where None
#: is one of its values: as its field spec says (`training_fields.FieldSpec.
#: null_means`: early stopping's patience, "off: no early stopping") or its
#: spec field does (`spec.EncoderSpec.relation_edge_dim`: None, edges carry no
#: relation, the form's "off"). Any other None reads "not set" (an unset seed;
#: a task's loss weight, which takes no null: a value left out is the task's
#: reference, never a None shown as one, OPS-R2); one the run's stage or task
#: does not read reads so first (`setting_words`).
SETTING_UNSET_WORDS: Dict[str, str] = {
    "early_stopping_patience": OFF.lower(),
    "encoder.relation_edge_dim": OFF.lower(),
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
    (`_SPEC_PARTS`), "not taken by a base" or by its task. `data` is a
    recorded spec, with its reference notes or as a list holds it without
    them (`listed_record`'s `spec`), read by the one reader of a listed spec
    (`record.listed_spec`)."""
    from src.models.bundle.record import listed_spec
    from src.models.bundle.spec import SpecError

    if not isinstance(data, Mapping):
        return (status_mark("neutral", f"{UNKNOWN}: the record holds no model spec"), "")
    spec, problem = listed_spec(data)
    differences: Dict[str, Any] = {}
    if spec is not None:
        try:
            differences = spec.differences_from_reference()
        except (SpecError, TypeError, ValueError, KeyError) as exc:
            spec, problem = None, str(exc)
    if spec is None:
        return (status_mark("neutral", f"{UNKNOWN}: not read as a model spec "
                                       f"({short_reason(problem, 120)})"), "")
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
    `data_kind_lines`, `r10_table_html`)."""
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


#: **A relation's state in words** (`measurement.RELATION_STATES`: what a
#: registered result's relations hold): verified a filled mark, not applicable
#: a hollow one (the skill's §6).
RELATION_STATE_MARKS = {"verified": ("ok", "Verified"), "not_applicable": ("neutral", NOT_APPLICABLE)}


def relation_rows(relations: Any) -> List[Tuple[str, str]]:
    """**The one reading of a record's relations** (the Test page's result,
    Models' recorded results, both through `relations_html`): one ``(label,
    escaped HTML)`` row each, its meaning first and its code after it ("Model
    trained on this graph (R3)", "Data version bound to this graph":
    `caveats.RELATION_DISPLAY_NAMES`, in that order, then any other by its
    recorded name, once; the skill's §5: a code only beside a plain label),
    its state a status mark with its word (`RELATION_STATE_MARKS`; any other
    state as recorded). A record that holds none reads one "Relations" row,
    "Not recorded"."""
    if not isinstance(relations, Mapping):
        return [("Relations", status_mark("neutral", NOT_RECORDED))]
    names = [*(n for n in RELATION_DISPLAY_NAMES if n in relations),
             *(n for n in relations if n not in RELATION_DISPLAY_NAMES)]
    rows = []
    for name in names:
        display = RELATION_DISPLAY_NAMES.get(name)
        if display is None:
            # A relation this build has no words for: its recorded name, once.
            label = str(name)
        else:
            label = display[:1].upper() + display[1:]
            if re.fullmatch(r"R\d+", str(name)):
                label += f" ({name})"
        state = relations[name]
        mark = RELATION_STATE_MARKS.get(state) if isinstance(state, str) else None
        rows.append((label, status_mark(*mark) if mark else e(state)))
    return rows


def relations_html(relations: Any, *, classes: str = "") -> str:
    """**The one rendering of a record's relations**: `relation_rows` as a
    summary list of its own whose key column takes a relation's whole name on
    one line (``.shep-kv-rel``, up to 32em, `src/webui/style.py`), so no
    relation row wraps where the result's 12em key column would break it into
    two or three lines. The Test page's result shows it under its "Relations"
    key, Models' recorded results under each record's row. `classes` adds a
    page's own classes."""
    return summary_list(relation_rows(relations),
                        classes=f"shep-kv-rel {classes}".strip())


def r10_state_parts(label: str, part: Any, evidence: Any
                    ) -> Tuple[Optional[str], str, Optional[str]]:
    """One R10 label's state in the caveats' words, in its parts: the state
    as recorded (None where it does not read), its words (`R10_STATE_TEXT`;
    an *unverifiable* label's reason by its scope's evidence state,
    `caveats.R10_UNVERIFIABLE_REASON`, never a number; "not reported" where
    the state does not read), and its count with its denominator and unit
    where the state has one ("3 of 5 distinct test diseases; 7 test cases
    with one of them"), else None. **The one reading** of a label's state:
    `r10_state_text` joins the parts, the R10 table (`r10_table_html`) lays
    them out."""
    state = _get(part, "state")
    words = R10_STATE_TEXT.get(state) if isinstance(state, str) else None
    if state == "unverifiable" and isinstance(evidence, str):
        words = R10_UNVERIFIABLE_REASON.get(evidence, words)
    if words is None:
        return None, "not reported", None
    if state not in ("overlap", "none"):
        return state, words, None
    unit = "distinct test diseases" if label == "disease_overlap" else "test cases"
    count = (f"{recorded_count_text(_get(part, 'k'))} of "
             f"{recorded_count_text(_get(part, 'n'))} {unit}")
    if label == "disease_overlap":
        count += f"; {recorded_count_text(_get(part, 'cases'))} test cases with one of them"
    return state, words, count


def r10_state_text(label: str, part: Any, evidence: Any) -> str:
    """One R10 label's state in the caveats' words, as text: its words, then
    its count with its denominator where the state has one
    (`r10_state_parts`). Models › Results' Compare shows it beside the
    label's name (its row's)."""
    _state, words, count = r10_state_parts(label, part, evidence)
    return words if count is None else f"{words} — {count}"


#: The words before an unverifiable label's reason in every caveats text
#: for it (`R10_STATE_TEXT`, `R10_UNVERIFIABLE_REASON`): the table's mark
#: says the state, the reason follows it.
_UNVERIFIABLE_LEAD = "unverifiable: "


def _r10_cell(label: str, part: Any, evidence: Any) -> str:
    """One R10 cell (`r10_table_html`): its state by the shared mark (the
    skill's §6) and, under it, its count or its reason, all in the caveats'
    words (`r10_state_parts`). An overlap is a condition to watch, amber ▲
    (it qualifies the measurement and blocks nothing); an unverifiable label,
    one with no validation inputs recorded and one not reported are missing
    states, grey ○, an unverifiable one with its reason and never a number;
    "none" is quiet, its words alone, never a green pass ("none in the
    inputs this checkpoint records", not "no overlap")."""
    state, words, count = r10_state_parts(label, part, evidence)
    if state == "overlap":
        head = status_mark("warn", words[:1].upper() + words[1:])
    elif state == "none":
        head = f'<span class="shep-r10-w">{e(words[:1].upper() + words[1:])}</span>'
    elif state == "unverifiable" and words.startswith(_UNVERIFIABLE_LEAD):
        reason = words[len(_UNVERIFIABLE_LEAD):]
        head = (status_mark("neutral", "Unverifiable")
                + f'<div class="shep-r10-n">{e(reason[:1].upper() + reason[1:])}</div>')
    else:
        head = status_mark("neutral", words[:1].upper() + words[1:])
    return head + (f'<div class="shep-r10-n">{e(count)}</div>' if count else "")


#: The R10 table's title, the key a page shows it under: its meaning first,
#: its code after (the skill's §5).
R10_TITLE = "Train/test overlap (R10)"
#: The help of the R10 table's ⓘ (the Test page's result, Models' recorded
#: results): what its two columns count, against the inputs each row names.
#: Its definitions are `R10_LABEL_TEXT`'s, each said once (a column's for
#: both rows); the governed meaning is unchanged (the skill's §13).
R10_HELP = (
    R10_TITLE,
    "How much of the test set the recorded training or validation inputs hold.\n\n"
    "Disease overlap: distinct test diseases an input case also has, of all distinct test "
    "diseases, and their test cases.\n"
    "Phenotype-set overlap: test cases matching an input case's disease and phenotype set, of "
    "all test cases.\n"
    "Sets ignore order and repeats, so the scoring inputs may differ.\n"
    "Ref: R10",
)
#: The R10 table's ⓘ, wherever a recorded result shows it (`record_html`).
RENDERER_HELP["record-r10"] = R10_HELP


def r10_table_html(r10: Any) -> str:
    """**The one rendering of a supplied cohort's R10** (research comparison
    Stage 8b; the skill's §4: the R10 overlap matrix is a table), from the
    one reader's view of a record (its ``r10``: each scope's evidence state
    and its disease and phenotype-set overlap): a row per scope (Training,
    Validation) with whether its evidence was found under it
    (`R10_EVIDENCE_WORDS`), a column per label ("Disease overlap",
    "Phenotype-set overlap": the label's name without its scope, which heads
    the row), each cell its state by the shared mark and its count, unit,
    denominator and cases or its reason (`_r10_cell`). The test page's
    result, Models' recorded results and Models › Results' Compare show it
    alike. The governed words are unchanged (the skill's §13, 2026-10-25):
    an unverifiable cell never reads 0, "—" or "no overlap", and an overlap
    blocks nothing. The evidence's path is never read. Every value is
    escaped."""
    scopes = list(R10_LABEL_TEXT.items())
    heads = []
    for label, words in scopes[0][1].items():
        rest = str(words["name"]).split(" ", 1)[-1]
        heads.append(f'<th scope="col">{e(rest[:1].upper() + rest[1:])}</th>')
    rows = []
    for scope, labels in scopes:
        part = _get(r10, scope)
        evidence = _get(_get(part, "evidence"), "state")
        found = R10_EVIDENCE_WORDS.get(evidence) if isinstance(evidence, str) else None
        cells = "".join(f"<td>{_r10_cell(label, _get(part, label), evidence)}</td>"
                        for label in labels)
        rows.append(f'<tr><th scope="row">{e(scope[:1].upper() + scope[1:])}'
                    f'<div class="shep-r10-n">{e(found or "evidence not reported")}</div></th>'
                    f"{cells}</tr>")
    return (f'<table class="shep-r10" aria-label="{e(R10_TITLE)}"><thead><tr>'
            f'<th scope="col">Inputs</th>{"".join(heads)}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def data_kind_lines(view: Mapping[str, Any], *, version_counts: bool = True) -> list:
    """**The one rendering of what a result's data kind adds** (research
    comparison Stages 6 and 8b), as ``(label, escaped HTML)`` rows, from the
    one reader's view of a record (`record_views`: a ledger record's record
    view, or a kept report's result view, `result_view`, as a JSON dump), so a
    registered record on Models and a kept report on Test say it alike, each
    page laying the rows out its own way:

    - a KG edge set's exposure in this build's words (`caveats.exposure_text`,
      the ⓘ saying `EXPOSURE_CAVEAT`'s points) and its source (definition version and
      split, or why its block does not read: fields named, never values), its
      message-passing graph and pairs (their counts, the version's own, left
      out with `version_counts` False, where the page says them for this very
      version: `record_rows`, R-70), and each metric that could not be
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
        kept = ("gene fixture " + fingerprint_value(_get(record, "sha256"),
                                                    what="gene fixture fingerprint", copy=False)
                if isinstance(fixture, Mapping) else e("no gene fixture kept"))
        return [("Generation", e(identity)),
                ("Split manifest", fingerprint_value(_get(manifest, "sha256"),
                                                     what="split manifest fingerprint",
                                                     copy=False) + f"; {kept}"),
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
             f"with seed {pairs.get('seed')}, {recorded_count_text(pairs.get('rows'))} rows"
             if version_counts else
             f"Message passing over the split's {passing.get('part')} part; pairs "
             f"{pairs.get('rule')} with seed {pairs.get('seed')}")
    measured = _get(view, "measured")
    statuses = measured.get("metrics") if isinstance(measured, Mapping) else None
    not_computed = [str(name) for name, status in (statuses or {}).items()
                    if isinstance(status, Mapping) and status.get("status") != "computed"]
    rows = [("Exposure", e(exposure_text(exposure))
             + help_button("kg-exposure", "Exposure (KG edge sets)")),
            ("Exposure source", e(source)),
            ("Graph and pairs", e(graph))]
    # A KG link-prediction record says each metric's status in its own row
    # (`record_figure_rows`), so the list of the ones not computed is said
    # only for a record of another mode.
    if not_computed and _get(view, "mode") != LP_MODE:
        rows.append(("Not computable", e(", ".join(not_computed))
                     + " (a status, never a number)"))
    return rows


# ---------------------------------------------------------------------------
# One recorded result (R-82): the record the Test page and Models both draw
# ---------------------------------------------------------------------------
#: The KG link-prediction measurement's mode (`caveats.MEASUREMENT_NAMES`).
LP_MODE = "kg_link_prediction"
#: What a dataset record's origin means (the Diagnosis panel's words).
ORIGIN_MEANING = {
    "synthetic": "a generated stand-in, not a clinical knowledge graph",
    "files": "built from identified source files",
}
#: A value a record does not hold as a number, in its state's words (the
#: skill's §6: never "—", which means "no value here").
NOT_REPORTED = "not reported"
#: **A task record's answer: its tiles**, in order, by the record's mode (a
#: task measurement, `src.evaluation.task_measurement`): pooled over its
#: positive pairs or a mean over its ranked cases, and its coverage; each
#: named by `caveats.metric_text`. A record of any other mode has none: its
#: figures are its rows (`record_figure_rows`).
RECORD_TILES = {
    "disease": ("pooled_mrr", "pooled_top1", "pooled_top10", "coverage"),
    "gene": ("pooled_mrr", "pooled_top1", "macro_recall@5", "coverage"),
    "patients_like_me": ("pooled_mrr", "pooled_top10", "macro_recall@10", "coverage"),
}
#: A task record's metric families that take a cut-off k, each one row (its k
#: values in order), and its single figures, a row each.
_TASK_FAMILIES = ("pooled_top", "macro_recall@", "macro_precision@")
_TASK_ROWS = ("macro_mrr", "best_positive_mrr")


def record_number(value: Any) -> str:
    """A recorded figure as a result shows it: four decimals (its value as
    recorded is one click away, `recorded_metrics_fold`, and in every
    export); "not reported" where the record holds no number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return NOT_REPORTED
    try:
        return f"{value:.4f}"
    except (OverflowError, ValueError):  # an integer too large for a float
        return NOT_REPORTED


def record_mode(view: Mapping[str, Any]) -> Optional[str]:
    """A record view's measurement mode, or None where it carries none it can
    read (never taken for another mode)."""
    mode = _get(view, "mode")
    return mode if isinstance(mode, str) and mode.strip() else None


def record_kind(view: Mapping[str, Any]) -> str:
    """The kind of data a record view measured (its ``cohort.kind``, by which
    the one reader read it): supplied, kg_edges or generated."""
    kind = _dig(view, "cohort", "kind")
    return kind if isinstance(kind, str) else "supplied"


def dataset_origin_html(origin: Any) -> str:
    """A record's dataset origin, where it is said (`record_rows`' Dataset
    row, Models' records header): "synthetic" in amber, then what it means."""
    meaning = ORIGIN_MEANING.get(origin) if isinstance(origin, str) else None
    word = (f'<b class="shep-amber-t">{e(origin)}</b>' if origin == "synthetic"
            else e(origin) if isinstance(origin, str) and origin else e(NOT_REPORTED))
    return word + (f" — {e(meaning)}" if meaning else "")


def record_coverage_html(view: Mapping[str, Any], *, version_counts: bool = True) -> str:
    """**A record's coverage, the one wording** (the Test page's result and
    Models' records alike): what it counted on its first line, how its data
    stands under it, with no chain of " · " (the skill's §5). A task record's
    ranked cases of the import-eligible ones (PLAN_B05 §4); a KG edge set's
    pairs scored of those stored, and its positive edges; a generated
    split's cases ranked, and its generated patients and diseases; any other
    mode's source cases and exclusions, and the cases its record counts, with
    no claim of what they are. Every count as recorded (`recorded_count_text`).

    The second line is the data version's own counts (source cases and
    exclusions, positive edges, generated patients): with `version_counts`
    False, where the page says them already for this very version (the Test
    page's Data panel, for a result of the selected version; R-70 / BACKLOG
    51 (c)), it is left out, so a count is said once on the screen; a mode
    whose first line counts nothing of its own keeps it."""
    kind, mode = record_kind(view), record_mode(view)
    counts = _get(view, "counts")
    counts = counts if isinstance(counts, Mapping) else {}
    count = recorded_count_text
    task = mode in TASK_NAMES
    if kind == "kg_edges":
        run = _dig(view, "measured", "counts")
        run = run if isinstance(run, Mapping) else {}
        main = (f"<b>{e(count(run.get('pairs_scored')))} of {e(count(counts.get('pairs')))}"
                "</b> stored pairs scored")
        sub = (f"{count(counts.get('positive_edges'))} positive edges; "
               f"{count(counts.get('edges_without_negative'))} without a negative to pair")
    elif kind == "generated":
        diseases = counts.get("diseases")
        main = (f"<b>{e(count(_get(view, 'n_ranked')))}</b> "
                + ("cases ranked by the model" if task else
                   "counted by the record; what they are is not described for this mode"))
        sub = (f"{count(counts.get('samples'))} generated patients of {count(diseases)} "
               f"disease{'' if diseases == 1 else 's'}")
    else:
        excluded = counts.get("excluded")
        reasons = (NOT_REPORTED if not isinstance(excluded, Mapping) else
                   ", ".join(f"{str(name).replace('_', ' ')} {count(value)}"
                             for name, value in excluded.items()) or "none")
        if task:
            main = (f"<b>{e(count(_get(view, 'n_ranked')))} of "
                    f"{e(count(counts.get('ranked_eligible')))}</b> import-eligible cases "
                    "ranked by the model")
            sub = (f"{count(counts.get('source_cases'))} source cases; "
                   f"{count(counts.get('n_excluded'))} excluded at import: {reasons}")
        else:
            main = e(f"{count(counts.get('source_cases'))} source cases; "
                     f"{count(counts.get('n_excluded'))} excluded: {reasons}")
            sub = (f"{count(_get(view, 'n_ranked'))} cases counted by the record; what they "
                   "are is not described for this mode")
            # Its first line is the version's counts: the record's own count stays.
            return f'<div>{main}</div><div class="shep-sub">{e(sub)}</div>'
    if not version_counts:
        return f"<div>{main}</div>"
    return f'<div>{main}</div><div class="shep-sub">{e(sub)}</div>'


def _tile_note(name: str, kind: str) -> str:
    """What a task tile's figure is over: its positive pairs (pooled), its
    ranked cases (macro) or the share of the import-eligible cases ranked
    (coverage; a generated split's cases are not imported, so its share is of
    the cases measured), with no count, since the Coverage row says "n of M"
    once (the skill's §1.2)."""
    if name == "coverage":
        return ("share of import-eligible cases ranked" if kind == "supplied"
                else "share of the cases measured that were ranked")
    if name.startswith("pooled_"):
        return "pooled over positive pairs"
    if name.startswith("macro_"):
        return "mean over the ranked cases"
    return ""


def record_tiles_html(view: Mapping[str, Any], metrics: Mapping[str, Any]) -> str:
    """**A task record's answer**: its tiles (`RECORD_TILES`), each its
    figure's name less the task's, which the record's label above them says
    once (`_figure_words`: "Pooled MRR", not "Disease pooled MRR"; R-71), its
    value to four decimals and what it is over (`_tile_note`); "" for a
    record of any other mode. The tiles share their rows (``.shep-rec-tile``,
    a subgrid of ``.shep-rec-tiles``), so their values stand on one line
    however a name wraps (NEW-test-4)."""
    mode, kind = record_mode(view), record_kind(view)
    tiles = "".join(
        f'<div class="shep-rec-tile"><div class="shep-rec-tile-l">{e(_figure_words(mode, name))}'
        f'</div><div class="shep-rec-tile-v">{e(record_number(metrics.get(name)))}</div>'
        f'<div class="shep-rec-tile-n">{e(_tile_note(name, kind))}</div></div>'
        for name in RECORD_TILES.get(mode or "", ()))
    return f'<div class="shep-rec-tiles">{tiles}</div>' if tiles else ""


def _figure_words(mode: str, name: str) -> str:
    """A task metric's words as a row under the record's label, which names
    the task once: `caveats.metric_text`'s "Disease pooled top-k" less the
    task's name, "Pooled top-k"."""
    words = metric_text(mode, name)[len(TASK_NAMES[mode]) + 1:]
    return words[:1].upper() + words[1:]


def _task_rows(mode: str, metrics: Mapping[str, Any]) -> List[Tuple[str, str]]:
    """A task record's figures past its tiles (README E1, E2, E5): the
    conservative pooled MRR with what it counts, one row for each family that
    takes a cut-off k (its k values in order, a tile's left out), the macro
    and best-positive MRR, then any other figure the record holds, a row
    each. No mean rank: a task reports none."""
    tiles = RECORD_TILES.get(mode, ())
    shown = set(tiles) | {"conservative_pooled_mrr"}
    rows = [(_figure_words(mode, "conservative_pooled_mrr"),
             e(record_number(metrics.get("conservative_pooled_mrr")))
             + '<div class="shep-sub">an unranked case whose truth can be assessed counts 0'
               '</div>')]
    for prefix in _TASK_FAMILIES:
        family = sorted((int(name[len(prefix):]), name) for name in metrics
                        if str(name).startswith(prefix) and str(name)[len(prefix):].isdigit())
        shown.update(name for _k, name in family)
        values = [f"k={k} {record_number(metrics[name])}" for k, name in family
                  if name not in tiles]
        if values:
            rows.append((_figure_words(mode, f"{prefix}k"), e(", ".join(values))))
    for name in _TASK_ROWS:
        if name in metrics:
            shown.add(name)
            rows.append((_figure_words(mode, name), e(record_number(metrics[name]))))
    for name, value in sorted(metrics.items()):
        if name not in shown:
            rows.append((metric_text(mode, str(name)), e(record_number(value))))
    return rows


def lp_metric_names(names: Iterable[Any]) -> List[Any]:
    """**A KG link-prediction record's metrics in the order every surface
    shows them**: the measurement's own (`caveats.LP_METRIC_WORDS`' order)
    among `names`, then any other name in the order given, each once."""
    names = list(dict.fromkeys(names))
    return [*(n for n in LP_METRIC_WORDS if n in names),
            *(n for n in names if n not in LP_METRIC_WORDS)]


def lp_metric_words(name: Any) -> str:
    """A KG link-prediction metric by its plain name, as a row's key or a
    column's head (`caveats.LP_METRIC_WORDS`: "Accuracy (raw > 0)",
    "ROC-AUC", "Average precision"); a name the measurement does not write
    keeps its recorded name."""
    words = LP_METRIC_WORDS.get(name, str(name))
    return words[:1].upper() + words[1:]


def lp_value_html(value: Any, computed: bool = True) -> str:
    """A KG link-prediction metric's value as every surface shows it: four
    decimals (`record_number`), or "Not computable" where its record says it
    could not be computed (a status, never a number)."""
    return e(record_number(value)) if computed else status_mark("neutral", "Not computable")


def _lp_rows(view: Mapping[str, Any], metrics: Mapping[str, Any]) -> List[Tuple[str, str]]:
    """**A KG link-prediction record's figures** (B72-C2): each metric under
    its plain name (`lp_metric_words`), its value to four decimals
    (`lp_value_html`), and under it, once, its unit and its denominator as
    the record's ``measured`` holds them; a metric that could not be
    computed is that status, never a number. The values as recorded are one
    click away (`recorded_metrics_fold`). Models › Results' table reads the
    same three (`results_segment`)."""
    statuses = _dig(view, "measured", "metrics")
    statuses = statuses if isinstance(statuses, Mapping) else {}
    rows = []
    for name in lp_metric_names([*metrics, *statuses]):
        status = statuses.get(name) if isinstance(statuses.get(name), Mapping) else None
        value = lp_value_html(metrics.get(name),
                              status is None or status.get("status") == "computed")
        if status is not None:
            unit = status.get("unit")
            over = _dig(status, "denominator", "count")
            over_unit = _dig(status, "denominator", "unit")
            said = "; ".join(str(part) for part in (
                unit if isinstance(unit, str) and unit else None,
                f"over {recorded_count_text(over)} {over_unit}"
                if isinstance(over_unit, str) and over_unit else None) if part)
            value += f'<div class="shep-sub">{e(said)}</div>' if said else ""
        rows.append((lp_metric_words(name), value))
    return rows


def recorded_metrics_html(metrics: Mapping[str, Any]) -> str:
    """Every metric of a record as recorded, one per line
    (`caveats.recorded_metric_lines`: the name as written, a code, and the
    value's JSON text, unrounded), every value escaped; "no metric recorded"
    when it holds none. A record view holds every metric as a number with a
    fraction ("1.0"); a report keeps a whole number as written ("1")."""
    lines = recorded_metric_lines(metrics)
    if not lines:
        return f'<span class="shep-muted">{e("no metric recorded")}</span>'
    return "".join(f'<div class="shep-rec-ln"><code>{e(name)}</code> {e(value)}</div>'
                   for name, value in lines)


def recorded_metrics_fold(metrics: Mapping[str, Any]) -> str:
    """Every metric as recorded, one click away: a closed native disclosure,
    "Recorded metrics (N)" (the skill's §1.5), or "no metric recorded" when
    there is none. A record's figures in words round them; this keeps each
    value as written."""
    if not metrics:
        return f'<div class="shep-sub shep-rec-fold">{e("no metric recorded")}</div>'
    return (f'<details class="shep-rec-fold"><summary>Recorded metrics ({len(metrics)})'
            f'</summary>{recorded_metrics_html(metrics)}</details>')


def record_figure_rows(view: Mapping[str, Any], metrics: Mapping[str, Any]
                       ) -> List[Tuple[str, str]]:
    """**A record's figures as rows, by its mode**: a task's past its tiles
    (`_task_rows`), a KG link prediction's in plain names with their units
    (`_lp_rows`), and any other mode's every metric as recorded, one row,
    which is then its main result (R2: one display for every mode)."""
    mode = record_mode(view)
    if mode in TASK_NAMES:
        return _task_rows(mode, metrics)
    if mode == LP_MODE:
        return _lp_rows(view, metrics)
    return [("Recorded metrics", recorded_metrics_html(metrics))]


def record_semantics_html(view: Mapping[str, Any]) -> str:
    """The record's ``score_semantics`` exactly as recorded, a code, or "not
    recorded": never split, interpreted or rebuilt."""
    raw = _get(view, "score_semantics")
    if isinstance(raw, str) and raw.strip():
        return f"<code>{e(raw)}</code>"
    return f'<span class="shep-muted">{e(NOT_RECORDED.lower())}</span>'


def record_environment_html(view: Mapping[str, Any]) -> str:
    """The view's environment class (as the one reader classified it) and,
    for a run on CUDA, its GPU and driver on a line of their own."""
    klass = _get(view, "environment_class")
    gpu = _get(view, "gpu")
    text = e(klass if isinstance(klass, str) and klass else NOT_REPORTED)
    if isinstance(gpu, Mapping):
        text += (f'<div class="shep-sub">GPU {e(gpu.get("name") or "name not read")}, driver '
                 f'{e(gpu.get("driver") or "not read")}</div>')
    return text


def record_native_html(view: Mapping[str, Any]) -> str:
    """The view's native build identity (the run environment's one
    projection) in its one wording: a lead with the route and the count, then
    one line per component (`native_identity_lines`); "not recorded" for a
    run recorded before its environment named components."""
    return "".join(f"<div>{e(line)}</div>"
                   for line in native_identity_lines(_get(view, "native_identity")))


def record_regime_rows(view: Mapping[str, Any]) -> List[Tuple[str, str]]:
    """How the measurement ran, as the record holds it: its device (the
    record's setting, `record_views.recorded_settings`, which each page puts
    on the view as ``device``) and whether deterministic algorithms were on
    (the view's ``deterministic_algorithms``, or the page's ``deterministic``
    where the view holds none); "not recorded" where the record holds none."""
    device = _get(view, "device")
    flag = _get(view, "deterministic_algorithms")
    flag = flag if type(flag) is bool else _get(view, "deterministic")
    return [("Device", f"<code>{e(device)}</code>" if isinstance(device, str) and device
             else f'<span class="shep-muted">{e(NOT_RECORDED.lower())}</span>'),
            ("Deterministic algorithms", e(setting_words("deterministic_algorithms", flag))
             if type(flag) is bool else f'<span class="shep-muted">'
             f'{e(NOT_RECORDED.lower())}</span>')]


def _counts_fold(title: str, counts: Any, units: Any) -> str:
    """A version's or a run's counts, each with its unit as recorded, one
    click away: a closed disclosure, "<title> (N)"."""
    counts = counts if isinstance(counts, Mapping) else {}
    units = units if isinstance(units, Mapping) else {}
    lines = "".join(f'<div class="shep-rec-ln"><code>{e(name)}</code> '
                    f'{e(recorded_count_text(value))}: '
                    f'{e(units.get(name) or "unit not recorded")}</div>'
                    for name, value in counts.items())
    return (f'<details class="shep-rec-fold"><summary>{e(title)} ({len(counts)})</summary>'
            f'{lines}</details>')


def record_counts_rows(view: Mapping[str, Any], *,
                       version_counts: bool = True) -> List[Tuple[str, str]]:
    """A KG edge set's or a generated split's counts, each with its unit as
    recorded, one click away (`_counts_fold`): the data version's, unless
    `version_counts` is False (the page says this very version's counts
    already, `record_coverage_html`; R-70), and a KG edge set's run's, its
    own reading, which can differ; none for a supplied cohort, whose counts
    its Coverage row says, or where no fold is left."""
    kind = record_kind(view)
    if kind == "supplied":
        return []
    folds = (_counts_fold("Data version counts", _get(view, "counts"), _get(view, "count_units"))
             if version_counts else "")
    if kind == "kg_edges":
        folds += _counts_fold("Run counts", _dig(view, "measured", "counts"),
                              _dig(view, "measured", "count_units"))
    return [("Counts", folds)] if folds else []


def record_rows(view: Mapping[str, Any], metrics: Mapping[str, Any], *,
                registration: str, dataset: Optional[str] = None,
                version_counts: bool = True) -> List[Tuple[str, str]]:
    """**A recorded result's rows, the one list** (R-82: the Test page's
    result and each of Models' records are drawn from it, in this order):
    its coverage (`record_coverage_html`, with the version's counts unless
    `version_counts` is False, as are its counts' fold, `record_counts_rows`,
    and a KG edge set's graph and pairs, `data_kind_lines`), its figures by
    mode (`record_figure_rows`),
    its score semantics, its relations, a supplied cohort's R10 or another
    kind's lines (`data_kind_lines`), its environment, device, determinism
    and native build, its dataset where the page has not said it by its
    relation (`dataset`: the row's HTML, or None), and its registration
    (`registration`: the row's HTML, which only the page knows: a kept
    report's state on the Test page, a ledger record's sequence and run on
    Models). One row per fact (the skill's §4, §5): never a chain of " · ".
    Which of them are drawn one click away is `record_html`'s
    (`RECORD_DETAIL_ROWS`)."""
    relations = _get(view, "relations")
    rows = [("Coverage", record_coverage_html(view, version_counts=version_counts)),
            *record_figure_rows(view, metrics),
            ("Score semantics", record_semantics_html(view)),
            ("Relations", relations_html(relations) if isinstance(relations, Mapping)
             else status_mark("neutral", NOT_RECORDED)),
            *([(R10_TITLE, r10_table_html(_get(view, "r10")))]
              if record_kind(view) == "supplied"
              else data_kind_lines(view, version_counts=version_counts)),
            *record_counts_rows(view, version_counts=version_counts),
            ("Environment", record_environment_html(view)),
            *record_regime_rows(view),
            ("Native build", record_native_html(view))]
    if dataset is not None:
        rows.append(("Dataset", dataset))
    rows.append(("Registration", registration))
    return rows


#: **A record's engineering rows** (B52-E): how its score was computed and
#: how the measurement ran, which no reading of the result turns on, drawn one
#: click away under its other rows (`record_html`), with the metrics as
#: recorded. Its environment's class, which says whether it can be set beside
#: a CUDA run, stays in view.
RECORD_DETAIL_ROWS = ("Score semantics", "Device", "Deterministic algorithms", "Native build")


def record_html(view: Mapping[str, Any], metrics: Mapping[str, Any], *,
                registration: str, dataset: Optional[str] = None,
                version_counts: bool = True,
                details: Sequence[Tuple[str, str]] = ()) -> str:
    """**The one rendering of a recorded result** (R-82), the Test page's
    result and each of Models' records: its answer first (a task's tiles,
    `record_tiles_html`), then its rows as one summary list (`record_rows`;
    the relations and R10 across the list's width, R10's ⓘ beside its key),
    then **one disclosure** ("Details ›", `details_list`; the skill's §1.5,
    B52-E) holding its engineering rows (`RECORD_DETAIL_ROWS`: the score
    semantics' formula, the device, determinism and the native build's
    components with their digests) and, for a described measurement, its
    metrics as recorded (another mode's are its rows already); `details`,
    the page's own rows there first (the Test page's run id). What it is of
    and how it is named are the page's, around it."""
    rows = record_rows(view, metrics, registration=registration, dataset=dataset,
                       version_counts=version_counts)
    mode = record_mode(view)
    shown = [row for row in rows if row[0] not in RECORD_DETAIL_ROWS]
    more = [*details, *(row for row in rows if row[0] in RECORD_DETAIL_ROWS)]
    if mode in TASK_NAMES or mode == LP_MODE:
        more.append((f"Recorded metrics ({len(metrics)})", recorded_metrics_html(metrics)))
    return (f'<div class="shep-rec-body">{record_tiles_html(view, metrics)}'
            + summary_list(shown, wide=("Relations", R10_TITLE),
                           helps={R10_TITLE: ("record-r10", R10_HELP[0])},
                           classes="shep-rec-rows")
            + details_list(more, classes="shep-rec-rows") + "</div>")


#: The words a note's time is said in: its recorded ISO time as UTC.
def _note_time(raw: Any) -> str:
    return str(raw or "").replace("T", " ").replace("Z", " UTC")


def investigation_notes_html(notes: Sequence[Mapping[str, Any]], *, problems: Sequence[str] = (),
                             expected: bool = False) -> str:
    """**A condition group's investigation notes, the one rendering** (B40:
    the Test page's report and Models' records, the operator's and the
    read-only view's alike): who, when and what, every value escaped; when
    the notes cannot all be read, so (with `problems`' words, the caller's
    already redacted), never "no note is kept"; with `expected` (a group
    whose runs differ or whose comparison is incomplete) and no note, that
    none is kept. Nothing otherwise."""
    parts = []
    if notes:
        items = "".join(f'<li><b>{e(n.get("who"))}</b> · {e(_note_time(n.get("at")))}'
                        f'<div>{e(n.get("what"))}</div></li>' for n in notes)
        parts.append(f'<div class="shep-rec-notes"><div class="shep-w600">Investigation notes '
                     f'({len(notes)})</div><ul>{items}</ul></div>')
    if problems:
        parts.append('<div class="shep-sub">' + status_mark(
            "neutral", "The investigation notes cannot all be read, so a note kept for this "
                       "group may be missing here" + (": " + "; ".join(problems)
                                                      if any(problems) else "."))
            + "</div>")
    elif expected and not notes:
        parts.append(f'<div class="shep-sub">{e("No investigation note is kept for this group.")}'
                     "</div>")
    return "".join(parts)


#: The labels of Model Facts' digest rows (`summary_list`'s wide rows).
FACTS_MODEL_FP = "Model fingerprint (checkpoint sha256)"
FACTS_TRAINED_FP = ("Trained on: dataset fingerprint (split_manifest.json, from the checkpoint's "
                    "record)")


def model_facts_html(facts: ModelFacts, *, focus: Optional[str] = None,
                     uses: bool = True) -> str:
    """Model Facts, **every section always open** (R-28: no disclosure inside
    the drawer that shows them, the skill's §1.5): Warnings first when there
    are any, then Identity, Uses, Data and Evidence; "No warnings." last when
    there are none. Identity, Data and Evidence are each one summary list
    (`summary_list`: one row per fact, the skill's §4, §5); a trained-on
    dataset whose checkpoint was not read says why
    (`ModelFacts.trained_on_unread`), never "not recorded".

    `focus` marks the section a caller opened the facts at
    (``data-shep-focus``: its page scrolls it into view). Without `uses`
    there is no Uses section: for a page whose own line says the device
    statement, always in view, and whose banner says a synthetic workspace is
    for testing (Diagnosis' identity drawer, over its results footer; R-26).

    Each section's title is an h3: every place the facts are drawn (the Load
    drawer, Models' details, Diagnosis' identity drawer) is titled by an h2,
    so no heading level is skipped (verification of fcc3dea, V-AUD-2)."""
    def section(key: str, title: str, body: str, extra: str = "") -> str:
        marked = ' data-shep-focus="true"' if key == focus else ""
        return (f'<section class="shep-facts-sec{extra}" data-sec="{key}"{marked}>'
                f'<h3>{e(title)}</h3><div class="shep-facts-bd">{body}</div></section>')

    location = ""
    if facts.checkpoint_file:
        location = (token_html(facts.checkpoint_file)
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
    # A listed file trained on the chosen workspace's current dataset: by that
    # relation, said once, in Data (T08-flow-5: Identity said it too); the
    # dataset itself is said under the list's workspace select.
    trained_rows = ([] if facts.current_dataset else [(trained_label, data_fp)])
    identity = summary_list([
        ("Model", chain_html(*facts.name_facts) if facts.name_facts else e(facts.name_line)),
        *([("Task", e(facts.task))] if facts.task else []),
        *([("Configuration", chain_html(*facts.configuration_facts) if facts.configuration_facts
                            else e(facts.configuration))] if facts.configuration else []),
        # A run and a file are tokens: broken only before their own separators (the sweep's
        # L7, SW-LK7: "gatv2/20261002-000000_…" ended a line of the Load drawer at a hyphen).
        ("Run", token_html(facts.run) if facts.run else "—"),
        ("Checkpoint file", location or "—"),
        (FACTS_MODEL_FP, model_fp),
        *trained_rows,
        *facts.served_rows,
    ], wide=(FACTS_MODEL_FP, trained_label, *(label for label, _value in facts.served_rows)))
    uses_text = f"{NOT_A_MEDICAL_DEVICE} {SYNTHETIC_FOR_TESTING}"
    trained = (unread[0] if unread
               else ", ".join(facts.trained_on_names) if facts.trained_on_names
               else "matches no listed workspace" if facts.trained_on_fingerprint
               else NOT_RECORDED.lower())
    # The dataset fingerprint once more, short and without its Copy: Identity
    # shows it whole with its Copy.
    data_rows = ([("Trained on", e(CURRENT_DATASET))] if facts.current_dataset else
                 [("Trained on", e(trained) + (" " + chip("synthetic", "amber")
                                               if facts.synthetic else "")),
                  ("Dataset fingerprint",
                   fingerprint_value(facts.trained_on_fingerprint, copy=False,
                                     missing=unread[1] if unread else NOT_RECORDED.lower()))])
    if facts.sources:
        data_rows.append(("Sources", e(facts.sources)))
    data_rows.extend(facts.serving_rows)
    evidence = summary_list([
        ("Validation", e(validation_value(facts.validation, read=facts.validation_read))),
        ("Recorded results",
         e(recorded_tests_text(facts.recorded).removeprefix("Recorded results: "))),
    ], helps={"Validation": ("validation-trainer", "Validation (Trainer)")}) + facts.evidence_more
    body = (section("identity", "Identity", identity)
            + (section("uses", "Uses", f"<p>{e(uses_text)}</p>") if uses else "")
            + section("data", "Data", summary_list(data_rows))
            + section("evidence", "Evidence", evidence))
    if facts.warnings:
        warned = section("warnings", "Warnings",
                         "".join(warning_html(h, d) for h, d in facts.warnings),
                         " shep-facts-warn")
        return f'<div class="shep-facts">{warned}{body}</div>'
    return (f'<div class="shep-facts">{body}'
            + section("warnings", "Warnings", '<p class="shep-muted">No warnings.</p>')
            + "</div>")


__all__ = [
    "CURRENT_DATASET",
    "DETAILS_WORD",
    "DISABLED",
    "DISABLED_MARK",
    "SEP",
    "whole_words",
    "disabled_words",
    "literal_html",
    "marked_line",
    "name_pieces_html",
    "notice_block",
    "notice_rows_html",
    "notices_block",
    "relation_words",
    "see_above",
    "token_html",
    "LP_MODE",
    "lp_metric_names",
    "lp_metric_words",
    "lp_value_html",
    "NOT_REPORTED",
    "ORIGIN_MEANING",
    "RECORD_TILES",
    "dataset_origin_html",
    "investigation_notes_html",
    "record_counts_rows",
    "record_coverage_html",
    "record_environment_html",
    "record_figure_rows",
    "record_html",
    "record_kind",
    "record_mode",
    "record_native_html",
    "record_number",
    "record_regime_rows",
    "record_rows",
    "record_semantics_html",
    "record_tiles_html",
    "recorded_metrics_fold",
    "recorded_metrics_html",
    "FACTS_MODEL_FP",
    "FACTS_TRAINED_FP",
    "FIELD_ROWS",
    "FROZEN_BACKBONE_WORDS",
    "IDENTITY_BANNERS",
    "DATASET_NOT_RECORDED",
    "METADATA_NOT_READABLE",
    "TRAINED_ON_UNLISTED",
    "ModelFacts",
    "ALL_NEIGHBOURS",
    "NOT_A_MEDICAL_DEVICE",
    "NOT_SET",
    "NOT_TAKEN_BY_A_BASE",
    "NO_READABLE_SPLIT_MANIFEST",
    "NO_SPLIT_MANIFEST",
    "R10_EVIDENCE_WORDS",
    "R10_TITLE",
    "RELATION_STATE_MARKS",
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
    "cut_at",
    "dataset_details_html",
    "details_list",
    "TRIGGER_JS",
    "event_act",
    "keep_as",
    "SHORT_HEX",
    "SYNTHETIC_FOR_TESTING",
    "SYNTHETIC_SOURCES",
    "built_date",
    "built_text",
    "change_value",
    "chip",
    "configuration_text",
    "data_kind_lines",
    "dataset_exception",
    "dataset_relation_html",
    "dataset_identity_html",
    "dataset_marks_html",
    "distinct_names",
    "distinguishing_digits",
    "e",
    "epoch_words",
    "facts_from_checkpoint",
    "field_path_words",
    "field_words",
    "fingerprint_html",
    "fingerprint_value",
    "fit_apart",
    "fit_option",
    "fit_options",
    "freeze_words",
    "grapheme_ends",
    "ident_html",
    "is_digest",
    "metric_words",
    "model_facts_html",
    "model_kind_text",
    "NO_LEDGER_WORDS",
    "operation_words",
    "entry_place",
    "RUN_OPERATION_WORDS",
    "model_spec_html",
    "model_name_html",
    "model_name_line",
    "model_name_facts",
    "chain_html",
    "configuration_facts",
    "named_words",
    "not_trainable_chip",
    "not_trainable_short",
    "origin_of",
    "pair_text",
    "r10_state_parts",
    "r10_state_text",
    "r10_table_html",
    "reason_lead",
    "record_chip",
    "record_kind_text",
    "record_read",
    "record_verified",
    "relation_rows",
    "relations_html",
    "recorded_count_text",
    "recorded_tests_text",
    "resources_html",
    "run_name",
    "run_parts",
    "run_started",
    "run_suffix",
    "run_names",
    "run_time",
    "run_words",
    "selection_value",
    "selection_words",
    "setting_words",
    "short_hex",
    "short_reason",
    "sources_html",
    "sources_text",
    "summary_list",
    "trained_on",
    "trained_on_html",
    "unreadable_words",
    "validation_cell",
    "validation_text",
    "validation_figure",
    "validation_value",
    "warning_html",
    "word_groups",
    "workspace_choices",
]
