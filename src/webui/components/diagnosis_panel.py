"""
Diagnosis (clinical): the dashboard's first screen (M5-2, UI-B).
================================================================
`docs/working/PLAN_TEST_RESULTS.md` §6, "M5's boundaries", and the M5 mockups
(round 2): the clinician's page for one question — which rare diseases fit
these phenotypes, and what produced the answer.

**Layout** (the page contract, `src/webui/shell.py`; this module's ``PAGE_CSS``):
wide (at least 1400 × 760 CSS px) one screen in three panels — the input column
| the candidates | the selected candidate's details — each long region
scrolling inside its own panel; narrow, the panels stack and only the page
scrolls.

- **The input column**: HPO terms, Patient ID (a blank field is sent as no id,
  so the API gives the result its own), Options (Top-K, Explanations), then the
  **"Model in use" card** and Run Diagnosis, the last item of the column's
  scroll body and sticky at its bottom: they follow Options when there is room
  and pin when the column overflows. The card (`diagnosis_card`) is the one
  place the served identity appears; the header's "Serving now" strip and the
  Model Configuration accordion (paths, Load/Reload, Save Config, Reset) are
  gone, their gated operations living on in the shared Load
  (`model_list`, through "Change model…"); ``POST /pipeline/config`` stays
  for API clients, gated by WP2.
- **Run Diagnosis** is greyed out and not clickable when no model is loaded
  (a bubble on hover and on keyboard focus says why, and the reason is visible
  text under it) and while a model change is open; while this page's Load
  reloads, the card says "Loading <12 hex> for every session… <elapsed>" and
  Run Diagnosis says the load is in progress (the mockup's state 3; set in the
  list's step before the reload, taken back after its answer). The API's demo
  answer for API clients is unchanged.
- **The candidates**: the results badge ("Produced by the model in use" /
  "Produced by an earlier model or pipeline — N parts differ ›" / "Cannot
  check against the model in use yet" with its one reason), the identity
  banners of the result, its warnings and the ranked table. **The badge
  compares the whole served identity**, part by part, as the identity table
  reads it (`_identity_rows`), the result's against the card's last read; a
  read that failed or is older than the result never leaves "equal" showing.
  The footer holds the exports (unchanged: CSV and report keep the result's own
  identity; not shown until a result exists) and the medical-device line, once.
- **The details** of the selected row: Evidence · Explanation · Matching
  phenotypes · Paths (in-page segments).
- **Two drawers**, one open at a time: the identity drawer ("Model in use |
  This result") over the details column, and "Change model…" (the clinical
  preset of the shared model list and its Load) over the candidates and the
  details. A request D8 refuses gets Change model… disabled with the reason;
  the list is never shown to it.

**Architecture note.** The diagnosis itself goes through the shared API over
HTTP (`_call_diagnose`, `_self_request`), as every frontend's does. What is
served is read through the dashboard's one status reader, in process
(`served_status.read_status`, by the card), and the Load is the shared one
(`model_list`: D8 on its own request, WP2's in-process reload). This module
calls no reload and writes no configuration.

**Per browser session**: the result, the selection, the card's last read and
the drawers are `gr.State`; nothing here is module state (BACKLOG 28).

Module: src/webui/components/diagnosis_panel.py
"""
from __future__ import annotations

import csv
import io
import logging
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import gradio as gr
import requests

from src.core.types import (
    DEMO,
    MIXED_AS_WORDS,
    PATH_REASONING,
    RESULT_KINDS,
    SERVED_MIXTURE,
    SP_NUMERIC_STATUSES,
    SP_STATUS_WORDS,
    result_label,
)
from src.utils.server_address import ServerAddressUnknownError, api_base_url
from src.webui.components import diagnosis_card as card
from src.webui.components.chips import D8_SHORT
from src.webui.components.help import copy_button, help_button, text_id
from src.webui.components.renderers import IDENTITY_BANNERS
from src.webui.components.renderers import NOT_A_MEDICAL_DEVICE as _NOT_A_MEDICAL_DEVICE
from src.webui.components.renderers import e as _e
from src.webui.components.renderers import short_hex as _short_hex
from src.webui.operator_access import operator_access

logger = logging.getLogger(__name__)

#: This page's id in the shell (`src.webui.shell.CLINICAL`; the shell and the
#: model list are imported where the page is built, since the shell imports
#: this package while it loads).
CLINICAL = "clinical"


def _api_url(path: str) -> str:
    """An API URL on this server, at an address it has accepted requests on.

    The tab runs inside the API's own process, so the API is wherever this server
    accepts connections -- port 8264 under the systemd unit, 8000 under a bare
    start. `src/utils/server_address.py` explains why that is the one source; it
    replaces a hard-coded ``http://127.0.0.1:8000``.
    """
    return f"{api_base_url()}{path}"


def _self_request(method: str, path: str, **kwargs: Any) -> requests.Response:
    """One call from this tab to its own server's API, never through a proxy.

    ``trust_env`` is off, so ``HTTP_PROXY`` and friends are ignored. The target is
    this server's own socket -- under a wildcard bind, possibly a LAN address that
    an institutional ``NO_PROXY`` list does not name -- and the diagnose payload
    carries patient phenotypes, which have no business at a forward proxy. A
    session per call, because a shared ``requests.Session`` is not documented as
    safe across Gradio's worker threads, and these calls are few.
    """
    with requests.Session() as session:
        session.trust_env = False
        return session.request(method, _api_url(path), **kwargs)



# Canonical HPO term id, e.g. HP:0001250. Tolerant of case, an optional / missing
# colon, and surrounding whitespace ("hp 0001250", "HP:0001250", "HP_0001250"),
# and a trailing non-digit so it won't grab 7 digits out of a longer number. The
# leading/trailing guards avoid matching inside another token (e.g. "CHP:...").
_HPO_ID_RE = re.compile(r"(?<![A-Za-z0-9])HP[:_\s]*(\d{7})(?![0-9])", re.IGNORECASE)


def _parse_hpo_ids(text: str) -> List[str]:
    """Extract HPO term ids from free-form text, tolerant of formatting.

    Accepts ids separated by newlines, commas, semicolons, spaces or tabs, with
    or without a trailing name, and mixed with stray punctuation — everything the
    strict one-per-line parser used to reject. Normalises to canonical
    ``HP:0000000`` form and preserves order.

    **Repeats are kept.** This used to remove them by string, a second copy of a
    rule the service owns: the pipeline removes repeats by graph node, after
    mapping, and reports how many it removed. Removing them here as well hid them
    from that count and from the request-size limit
    (docs/working/PLAN_PHENOTYPE_NORMALISATION.md, decision W).
    """
    return [f"HP:{match.group(1)}" for match in _HPO_ID_RE.finditer(text or "")]

#: What a candidate's score is, said under the report's table and in the
#: Score column's help: the value its result kind ranks by (`served_score` or
#: `path_score`, Spec 1 §2.1); the words shown say what the number is. The demo
#: answer has no score, and no note.
SCORE_NOTE = ("Score ranks the candidates of this result. It is not a calibrated "
              "disease probability.")

# Evidence label (the API's evidence_label) → (emoji, CSS color)
LABEL_STYLES = {
    "Strong path support": ("🟢", "#22c55e"),
    "Weak path support": ("🟡", "#eab308"),
    "Analogy-based (no direct KG path)": ("🔵", "#3b82f6"),
    "Insufficient evidence": ("⚪", "#9ca3af"),
}


# =============================================================================
# API Client
# =============================================================================
def _call_diagnose(
    phenotypes: List[str],
    patient_id: Optional[str] = None,
    top_k: int = 10,
    include_explanations: bool = True,
) -> Dict[str, Any]:
    """Call the /api/v1/diagnose endpoint and return parsed JSON.

    **A blank Patient ID is sent as None**, not "" and not a fixed placeholder:
    the API then gives the result its own id (``patient_…``), as the field
    promises ("auto-generated if blank"). A fixed placeholder named every blank
    result from every browser as one patient.
    """
    patient = patient_id.strip() if isinstance(patient_id, str) else None
    payload = {
        "patient_id": patient or None,
        "phenotypes": phenotypes,
        "top_k": top_k,
        "include_explanations": include_explanations,
    }
    try:
        resp = _self_request("POST", "/api/v1/diagnose", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except ServerAddressUnknownError as e:
        return {"error": str(e)}
    except requests.ConnectionError:
        return {"error": "API server not reachable. Is uvicorn running?"}
    except requests.HTTPError as e:
        if e.response.status_code == 422:
            return {"error": _describe_refused_request(e.response)}
        refusal = _describe_refused_result(e.response)
        if refusal:
            return {"error": refusal}
        return {"error": f"API error: {e.response.status_code} — {e.response.text}"}
    except Exception as e:
        return {"error": f"Unexpected error: {e}"}


#: The codes the pipeline refuses a result with (`src.core.types.ResultRefused`)
#: rather than write a number it does not have.
_RESULT_REFUSAL_CODES = ("MODEL_TERM_UNAVAILABLE", "NON_FINITE_SCORE")


def _describe_refused_result(response: Any) -> Optional[str]:
    """A result the pipeline refused (Spec 1 §2.1), as one sentence, or None
    for any other failure. The service sends the code first in its message."""
    try:
        body = response.json()
    except ValueError:
        return None
    detail = body.get("error") if isinstance(body, dict) else None
    if not isinstance(detail, str) or detail.split(":", 1)[0] not in _RESULT_REFUSAL_CODES:
        return None
    return (f"The service refused to produce a result, rather than show a number it does "
            f"not have — {detail}. Nothing was ranked.")


#: Shown when a 422 arrives without a reason this panel can read.
_UNREADABLE_REFUSAL = (
    "The API refused the request as invalid (HTTP 422), and its reason could not "
    "be read. Nothing was diagnosed."
)


def _describe_refused_request(response: requests.Response) -> str:
    """A 422 from `/diagnose`, in words a clinician can act on.

    **Read from the API's structured error, never restated.** FastAPI answers a
    request its model refuses with a list of entries, each naming a field, an
    error type and a message. The panel turns each into a line. It does not echo
    the entry's `input`, which for a list over the limit is the whole list.

    **The phenotype limit is the one case with its own message**, and both of
    its numbers come from the server: `ctx.actual_length` and `ctx.max_length`.
    The panel counts nothing and keeps no copy of the limit, so the request model
    stays the one place it is set. If either number is missing or not a whole
    number, the entry falls back to its plain line rather than guessing one.

    Never raises: a body that is not JSON, or JSON of another shape, gets the
    general line.
    """
    try:
        body = response.json()
    except ValueError:
        return _UNREADABLE_REFUSAL
    entries = body.get("detail") if isinstance(body, dict) else None
    if not isinstance(entries, list):
        return _UNREADABLE_REFUSAL

    lines = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        line = _too_many_phenotypes(entry) or _refusal_line(entry)
        if line:
            lines.append(line)
    return "\n\n".join(lines) if lines else _UNREADABLE_REFUSAL


def _too_many_phenotypes(entry: Dict[str, Any]) -> Optional[str]:
    """The message for a phenotype list over the limit, or None if `entry` is
    not one or does not carry both numbers."""
    if entry.get("loc") != ["body", "phenotypes"] or entry.get("type") != "too_long":
        return None
    ctx = entry.get("ctx")
    if not isinstance(ctx, dict):
        return None
    received, limit = ctx.get("actual_length"), ctx.get("max_length")
    # `bool` is an `int`; neither True nor False is a count.
    if not all(isinstance(n, int) and not isinstance(n, bool) for n in (received, limit)):
        return None
    return (
        f"Recognised {received} HPO phenotype entries (repeats included); one "
        f"submission accepts at most {limit}. No diagnosis was run, and the list "
        "was not shortened. Remove repeats or shorten the list, then submit again."
    )


def _refusal_line(entry: Dict[str, Any]) -> Optional[str]:
    """One refused field as `field: reason`, or None if there is no reason."""
    message = entry.get("msg")
    if not isinstance(message, str) or not message.strip():
        return None
    # Pydantic prefixes a validator's ValueError with "Value error, ".
    message = message.removeprefix("Value error, ")
    loc = entry.get("loc")
    path = [str(part) for part in loc[1:]] if isinstance(loc, list) else []
    field = ".".join(path) if path else "request"
    return f"Request refused — {field}: {message}"


# =============================================================================
# The result's kind (Spec 1 §2.1), as every surface below reads it
# =============================================================================
def _label_of(kind: str, result: Mapping[str, Any]) -> Optional[str]:
    """The kind's fixed label (`src.core.types.result_label`), η read from the
    result's own served identity; None where the identity cannot give it."""
    identity = result.get("served_identity")
    identity = identity if isinstance(identity, Mapping) else {}
    settings = identity.get("settings") if isinstance(identity.get("settings"), Mapping) else {}
    try:
        return result_label(kind, identity.get("scoring_mode"), settings.get("eta"))
    except (TypeError, ValueError):
        return None


def _result_kind(result: Any) -> Optional[str]:
    """The result's `result_kind` when it is one this page knows; else None, and
    every surface draws nothing from the result."""
    kind = result.get("result_kind") if isinstance(result, Mapping) else None
    return kind if kind in RESULT_KINDS else None


def _shows_sp(result: Any) -> bool:
    """Whether the result has an SP column: a mixture served with a table. Without
    one every record is "No SP table", which the kind's label already says."""
    identity = result.get("served_identity") if isinstance(result, Mapping) else None
    return (_result_kind(result) == SERVED_MIXTURE and isinstance(identity, Mapping)
            and identity.get("scoring_mode") == "gnn_plus_shortest_path")


#: Said where a response names no kind this page knows (absent, or a newer
#: kind): nothing is drawn from it.
KIND_NOT_REPORTED = ("The response does not name a result kind this page can read "
                     "(result_kind), so its candidates are not shown.")
#: A mixture whose identity does not give its η (only a malformed response):
#: its candidates are shown, and the label says what is missing, never a constant.
MIXTURE_ETA_NOT_REPORTED = ("Served ranking today: diseases reached by path discovery, "
                            "scored by the mixture; the response does not report its η")


def kind_label(result: Any) -> str:
    """The kind's fixed label, η read from the result's own served identity,
    never a constant; KIND_NOT_REPORTED where the kind is not one this page knows."""
    kind = _result_kind(result)
    if kind is None:
        return KIND_NOT_REPORTED
    return _label_of(kind, result) or MIXTURE_ETA_NOT_REPORTED


def _score(candidate: Mapping[str, Any], kind: Optional[str]) -> Any:
    """The value the kind ranks by; None for the demo answer, which has none."""
    if kind == SERVED_MIXTURE:
        return candidate.get("served_score")
    if kind == PATH_REASONING:
        return candidate.get("path_score")
    return None


def sp_text(sp_term: Any, compact: bool = False) -> str:
    """A candidate's SP record in words: the score for a numeric status (with
    the coverage for a partial one), else the status's own words, and "mixed as
    0 (legacy)" where the mixture still mixed it. Never 0 for a missing value.
    `compact` shortens the coverage to "(3/4)" for a table cell, which the Score
    help explains; the details and the report say "(3/4 phenotypes)"."""
    if not isinstance(sp_term, Mapping):
        return "not reported"
    status = sp_term.get("status")
    if status in SP_NUMERIC_STATUSES:
        text = _num(sp_term.get("score"))
        if status == "COMPUTED_PARTIAL":
            text += (f" ({sp_term.get('phenotypes_mapped')}/"
                     f"{sp_term.get('phenotypes_submitted')}"
                     f"{'' if compact else ' phenotypes'})")
        return text
    words = SP_STATUS_WORDS.get(status, "not reported")
    if sp_term.get("mixed_as") is not None:
        words += f", {MIXED_AS_WORDS}"
    return words


# =============================================================================
# Formatting helpers — diagnosis results
# =============================================================================
def _format_results_table(candidates: List[Dict], kind: Optional[str] = None,
                          shows_sp: bool = False) -> str:
    """Format candidates as a Markdown table (the exported report), its columns
    the kind's: the score and model term of a mixture (SP beside them when a
    table is served), the path score with no model, no score in the demo."""
    if kind is None:
        return f"*{KIND_NOT_REPORTED}*"
    if not candidates:
        return "*No candidates found.*"

    columns = ["Rank", "Disease"]
    if kind == SERVED_MIXTURE:
        columns += ["Score", "Model term"] + (["SP"] if shows_sp else [])
    elif kind == PATH_REASONING:
        columns += ["Score"]
    if kind != DEMO:
        columns += ["Evidence"]
    rows = []
    for c in candidates:
        cells = [str(c.get("rank")), f"**{c.get('disease_name')}**"]
        if kind != DEMO:
            cells.append(_num(_score(c, kind)))
        if kind == SERVED_MIXTURE:
            cells.append(_num(c.get("model_term")))
            if shows_sp:
                cells.append(sp_text(c.get("sp_term")))
        if kind != DEMO:
            label = c.get("evidence_label") or "—"
            emoji, _ = LABEL_STYLES.get(label, ("", "#888"))
            cells.append(f"{emoji} {label}".strip())
        rows.append("| " + " | ".join(cells) + " |")
    header = ("| " + " | ".join(columns) + " |\n"
              + "|" + "|".join(":----:" if c == "Rank" else "---" for c in columns) + "|\n")
    table = header + "\n".join(rows)
    return table + ("" if kind == DEMO else "\n\n" + SCORE_NOTE)


def _format_evidence_detail(candidate: Dict) -> str:
    """Format a candidate's evidence package in Markdown (the exported report;
    the page draws it as escaped HTML, `evidence_html`)."""
    pkg = candidate.get("evidence_package")
    if not pkg:
        return "*No evidence package available. Run with `include_explanations=true`.*"

    mode = pkg.get("mode") or "unknown"
    summary = pkg.get("summary") or ""
    label = pkg.get("evidence_label") or "—"
    emoji, _color = LABEL_STYLES.get(label, ("", "#888"))

    parts = [
        f"### {emoji} {label}",
        "",
        f"> {summary}",
        "",
    ]

    if mode == "direct_path":
        paths = pkg.get("direct_paths", [])
        min_hops = pkg.get("min_path_length", "?")
        parts.append(f"**Mode A — Direct Path Evidence** (shortest: {min_hops} hops)")
        parts.append("")
        for i, path in enumerate(paths[:5], 1):
            path_str = " → ".join(
                _node_display_name(n) for n in path
            )
            parts.append(f"{i}. `{path_str}`")

    elif mode == "analogy_based":
        analogies = pkg.get("analogies", [])
        parts.append("**Mode B — Analogy-Based Evidence**")
        parts.append("")
        for a in analogies[:3]:
            sim = a.get("embedding_similarity")
            name = a.get("similar_gene_name") or "?"
            shared = a.get("shared_features", [])
            known_paths = a.get("known_paths", [])
            # A similarity that is not there is left out, never printed as 0.
            parts.append(
                f"- **{name}**"
                + (f" (similarity: {sim:.2f})" if isinstance(sim, (int, float)) else "")
            )
            if shared:
                parts.append(f"  - Shared: {', '.join(shared)}")
            for kp in known_paths[:2]:
                kp_str = " → ".join(_node_display_name(n) for n in kp)
                parts.append(f"  - Path: `{kp_str}`")

    elif mode == "insufficient":
        parts.append("**No usable evidence found.** This candidate should be validated via independent clinical judgment.")

    # Supporting genes
    genes = candidate.get("supporting_genes", [])
    if genes:
        parts.append("")
        parts.append(f"**Supporting genes**: {', '.join(genes)}")

    return "\n".join(parts)


def _node_display_name(node_str: str) -> str:
    """Convert 'mondo:MONDO:0011073' to 'MONDO:0011073' for readability."""
    if ":" in node_str:
        parts = node_str.split(":", 1)
        return parts[1] if len(parts) == 2 else node_str
    return node_str


def _format_full_explanation(candidate: Dict) -> str:
    """Return the pipeline's Markdown explanation (ExplanationGenerator output)."""
    return candidate.get("explanation") or "*No explanation available.*"


# =============================================================================
# What produced a result: the served identity (contract M3c)
# =============================================================================
# **One reading of the identity for every surface.** The report, the identity
# drawer's "This result", the results badge and the CSV all read the
# `served_identity` the API copied from the result it answered with (the card
# reads the loaded pipeline's, from the status). The table rows are built once,
# by `_identity_rows`, so the report, the drawer and the badge's comparison
# cannot describe one result two ways. Every part states its state in the contract's words (verified,
# unverified, in_memory, supplied_in_memory, absent); a value that is not there
# is written as that state and its reason, never left blank and never "unknown".

#: The plan's claim boundary for the UI (PLAN_PROVENANCE_CONTRACT.md §6), verbatim.
#: Verified states carry ✓ in the identity table, so the sentence refers to them.
IDENTITY_CLAIM_BOUNDARY = (
    "Checked: these files are the ones their records name, and the records agree "
    "on the relations marked ✓. Not checked: who changed them, or anything a "
    "person able to edit both files and records could alter."
)
#: Added under it when a served identity, or the demo answer, has no part marked
#: ✓, so it is not read as a claim that something about this result was checked.
IDENTITY_NOTHING_CHECKED = (
    "No relation is marked ✓ for this result: nothing in it was checked against a record."
)
#: The same, said of the loaded pipeline: the identity drawer's "Model in use"
#: describes what is loaded now, not a result.
IDENTITY_NOTHING_CHECKED_LOADED = (
    "No relation is marked ✓ for the loaded pipeline: nothing it serves was "
    "checked against a record."
)
#: Added under it, in place of the line above, for a response that carried no
#: served identity: the panel was not told what was checked, so it can say
#: neither that something was nor that nothing was.
IDENTITY_CHECKS_NOT_REPORTED = (
    "The response did not say what produced it or what was checked."
)
#: README's statement, carried by every report and identity block (one place
#: for the words: the shared renderers, which Model Facts also uses).
NOT_A_MEDICAL_DEVICE = _NOT_A_MEDICAL_DEVICE

#: The word for the demo answer (`served_identity` null: no pipeline is
#: configured, nothing was served). It fills every identity cell and column.
IDENTITY_DEMO = "demo"
#: The word for a response that carried no `served_identity` at all (a server
#: from before contract M3c, or a malformed reply), or a part of one it left out.
#: Never "demo": that would claim the server said no pipeline was configured.
IDENTITY_NOT_REPORTED = "not_reported"
_SERVED = "served"

_VERIFIED_MARK = "✓"
#: Hex characters of a digest shown in a summary line; the full value is in the
#: identity table directly below it.
_SHORT_HEX = 12
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
#: A key the response did not carry, as distinct from one it carried as null.
_MISSING = object()

#: Row labels, in display order: one row per part of the identity.
_IDENTITY_PARTS = (
    "Checkpoint",
    "Knowledge graph (`kg.json`)",
    "Workspace manifest (`split_manifest.json`)",
    "Shortest-path table",
    "Dataset record (`kg.provenance.json`)",
    "Scoring mode",
    "Device (in effect; requested)",
    "Mixing weight eta",
    "Path settings",
    "Phenotype normalisation rule",
)
#: Why nothing can be said, for the two kinds that have no identity to read.
_NO_IDENTITY_REASON = {
    IDENTITY_DEMO: "no pipeline is configured on this server; nothing was served",
    IDENTITY_NOT_REPORTED: "the API's response did not carry a served identity",
}
_SCORING_MODE_WORDS = {
    "gnn_plus_shortest_path": "the GNN score combined with the shortest-path table",
    "gnn_only": "the GNN score alone",
    "path_reasoning_fallback": "path reasoning over the knowledge graph alone",
}
_PATH_SETTING_KEYS = (
    "max_path_length", "path_length_penalty", "aggregation_method",
    "max_paths_per_candidate", "use_direct_path_optimization",
)

#: (headline, detail) for each state a reader must not miss. Words carry the
#: meaning; the tab adds an icon, the report a blockquote, neither relies on it.
#: Defined once in the shared renderers (Model Facts' warnings say the same).
_IDENTITY_BANNERS = IDENTITY_BANNERS
assert set(_IDENTITY_BANNERS) >= {IDENTITY_DEMO, IDENTITY_NOT_REPORTED, "synthetic",
                                  "no_model", "no_sp"}


def _identity_of(container: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]]]:
    """What a diagnosis result says produced it: ``(kind, identity)``.

    ``kind`` is ``"served"`` with the identity dict, ``"demo"`` when the API sent
    ``served_identity: null`` (its demo answer, no pipeline configured), or
    ``"not_reported"`` when the key is missing or is not an object.
    """
    if "served_identity" not in container:
        return IDENTITY_NOT_REPORTED, None
    identity = container["served_identity"]
    if identity is None:
        return IDENTITY_DEMO, None
    if not isinstance(identity, dict):
        return IDENTITY_NOT_REPORTED, None
    return _SERVED, identity


def _field(part: Any, key: str) -> Any:
    """``part[key]``, or `_MISSING` when ``part`` is not a dict or lacks the key."""
    return part.get(key, _MISSING) if isinstance(part, dict) else _MISSING


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _is_synthetic(identity: Dict[str, Any]) -> bool:
    return _field(_field(identity, "dataset_record"), "origin") == "synthetic"


def _has_model(identity: Dict[str, Any]) -> bool:
    return _field(_field(identity, "checkpoint"), "state") in ("verified", "supplied_in_memory")


def _plain_value(value: Any) -> str:
    """A JSON scalar as text: ``true``/``false`` for booleans, floats as Python
    prints them (``0.7``, ``1.0``)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


# ----------------------------------------------------------------- Markdown form
#: Characters that could start Markdown or HTML syntax, or end a table cell.
_MD_SPECIAL_RE = re.compile(r"([\\`*_{}\[\]<>#|~!])")


def _md_text(value: Any) -> str:
    """A value from the API as plain text, safe in a Markdown table cell: one
    line, every character that could start Markdown or HTML escaped."""
    text = " ".join(_plain_value(value).split())
    return _MD_SPECIAL_RE.sub(r"\\\1", text)


def _md_code(value: Any, in_table: bool = True) -> str:
    """A value from the API in a code span, one line; a backtick in the value
    gets a longer fence.

    In a table cell (``in_table``) a pipe is escaped even inside the span: GFM
    splits a row on ``|`` before it parses code spans. Outside a table the
    escape would be shown literally, since a code span keeps backslashes.
    """
    text = " ".join(_plain_value(value).split())
    if in_table:
        text = text.replace("|", "\\|")
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * (longest + 1)
    pad = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{fence}{pad}{text}{pad}{fence}"


def _md_digest(value: Any) -> str:
    """A full digest, ``sha256:``-prefixed, in a code span."""
    return f"`sha256:{value}`" if _is_sha256(value) else _md_code(value)


_NOT_REPORTED_CELLS = (
    "not\\_reported", "not reported — the response did not include this value"
)


def _md_setting(value: Any) -> str:
    if value is _MISSING:
        return "not reported"
    return _md_code(value) if isinstance(value, str) else _md_text(value)


def _checkpoint_cells(identity: Dict[str, Any]) -> Tuple[str, str]:
    part = _field(identity, "checkpoint")
    state = _field(part, "state")
    if state == "verified":
        return f"{_VERIFIED_MARK} verified", _md_digest(_field(part, "sha256"))
    if state == "supplied_in_memory":
        return ("supplied\\_in\\_memory",
                "none — a model object was supplied in memory; no checkpoint bytes were read")
    if state == "absent":
        return ("absent", "absent — no trained model is served; path reasoning ranks "
                "the candidates (path-reasoning fallback)")
    return _NOT_REPORTED_CELLS


def _graph_cells(identity: Dict[str, Any]) -> Tuple[str, str]:
    part = _field(identity, "graph")
    state = _field(part, "state")
    if state == "verified":
        return f"{_VERIFIED_MARK} verified", _md_digest(_field(part, "kg_sha256"))
    if state == "unverified":
        return ("unverified", f"{_md_digest(_field(part, 'kg_sha256'))} — read from a "
                "file, checked against no workspace manifest")
    if state == "in_memory":
        return "in\\_memory", "none — the graph was supplied in memory; no file was read"
    return _NOT_REPORTED_CELLS


def _manifest_cells(identity: Dict[str, Any]) -> Tuple[str, str]:
    part = _field(identity, "graph")
    state = _field(part, "state")
    if state == "verified":
        return (f"{_VERIFIED_MARK} verified",
                _md_digest(_field(part, "split_manifest_sha256")))
    if state == "unverified":
        return ("absent", "absent — no workspace manifest was read (serving without "
                "a checkpoint verifies no workspace)")
    if state == "in_memory":
        return "absent", "absent — the graph was supplied in memory, from no workspace"
    return _NOT_REPORTED_CELLS


def _shortest_path_cells(identity: Dict[str, Any]) -> Tuple[str, str]:
    part = _field(identity, "shortest_paths")
    state = _field(part, "state")
    if state == "verified":
        build_id, max_hops = _field(part, "build_id"), _field(part, "max_hops")
        return (f"{_VERIFIED_MARK} verified",
                f"build\\_id {_md_setting(build_id)} · max\\_hops {_md_setting(max_hops)}")
    if state == "absent":
        if _field(identity, "scoring_mode") == "gnn_only":
            reason = "the GNN score is used alone (gnn\\_only, eta 1.0)"
        elif not _has_model(identity):
            reason = "no model is served to combine it with (path-reasoning fallback)"
        else:
            reason = "it does not contribute to the score"
        return "absent", f"absent — no table is served; {reason}"
    return _NOT_REPORTED_CELLS


#: What a dataset record's origin means, where the panel names one (the
#: served identity).
_DATASET_ORIGIN_MEANING = {
    "synthetic": "a generated stand-in, not a clinical knowledge graph",
    "files": "built from identified source files",
}


def _dataset_record_cells(identity: Dict[str, Any]) -> Tuple[str, str]:
    part = _field(identity, "dataset_record")
    state = _field(part, "state")
    if state == "verified":
        origin = _field(part, "origin")
        # **Never "verified" alone for a synthetic record** (contract M3b): the
        # binding and the origin are stated together.
        meaning = _DATASET_ORIGIN_MEANING.get(origin, "not described by this page")
        return (f"{_VERIFIED_MARK} verified · {_md_text(origin)}",
                f"{_md_digest(_field(part, 'sha256'))} — origin {_md_text(origin)}: {meaning}")
    if state == "absent":
        return "absent", "absent — no workspace was verified, so no dataset record was read"
    return _NOT_REPORTED_CELLS


def _scoring_mode_cells(identity: Dict[str, Any]) -> Tuple[str, str]:
    mode = _field(identity, "scoring_mode")
    if mode is _MISSING or mode is None:
        return _NOT_REPORTED_CELLS
    words = _SCORING_MODE_WORDS.get(mode, "not described by this page")
    return "in effect", f"{_md_code(mode)} — {words}"


def _device_cells(identity: Dict[str, Any]) -> Tuple[str, str]:
    settings = _field(identity, "settings")
    device, requested = _field(settings, "device"), _field(settings, "device_requested")
    if device is _MISSING or requested is _MISSING:
        return _NOT_REPORTED_CELLS
    if device is None and requested is None and not _has_model(identity):
        return "none", "none — no model is served, so nothing runs on a device"
    in_effect = (_md_code(device) if device is not None
                 else "not stated by the model supplied in memory")
    asked = (_md_code(requested) if requested is not None
             else "none (a model supplied in memory stays on its own device)")
    return "in effect", f"{in_effect} in effect; requested: {asked}"


def _eta_cells(identity: Dict[str, Any]) -> Tuple[str, str]:
    eta = _field(_field(identity, "settings"), "eta")
    if eta is _MISSING:
        return _NOT_REPORTED_CELLS
    if eta is None:
        return ("none", "none — no model is served, so there are no GNN and "
                "shortest-path terms to mix")
    note = (" — no shortest-path table, so the GNN score is used alone"
            if _field(identity, "scoring_mode") == "gnn_only" else "")
    return "in effect", f"{_md_text(eta)}{note}"


def _path_settings_cells(identity: Dict[str, Any]) -> Tuple[str, str]:
    # In effect in every mode: candidates are the diseases these paths reach.
    settings = _field(identity, "settings")
    if not isinstance(settings, dict):
        return _NOT_REPORTED_CELLS
    return "in effect", " · ".join(
        f"`{key}` {_md_setting(_field(settings, key))}" for key in _PATH_SETTING_KEYS
    )


def _phenotype_rule_cells(identity: Dict[str, Any]) -> Tuple[str, str]:
    version = _field(_field(identity, "settings"), "phenotype_normalisation_version")
    if version is _MISSING or version is None:
        return _NOT_REPORTED_CELLS
    return "in effect", f"version {_md_text(version)}"


def _identity_rows(kind: str, identity: Optional[Dict[str, Any]]) -> List[Tuple[str, str, str]]:
    """``(part, state, value)`` Markdown cells, one row per part, never blank.

    The one builder of the identity table: the report and the tab's identity
    block both render these rows, so they cannot disagree.
    """
    if kind != _SERVED or identity is None:
        word = _md_text(kind)
        return [(part, word, f"{word} — {_NO_IDENTITY_REASON[kind]}") for part in _IDENTITY_PARTS]
    cells = (
        _checkpoint_cells, _graph_cells, _manifest_cells, _shortest_path_cells,
        _dataset_record_cells, _scoring_mode_cells, _device_cells, _eta_cells,
        _path_settings_cells, _phenotype_rule_cells,
    )
    return [(part, *build(identity)) for part, build in zip(_IDENTITY_PARTS, cells, strict=True)]


def _claim_boundary_lines(kind: str, identity: Optional[Dict[str, Any]],
                          nothing_checked: str = IDENTITY_NOTHING_CHECKED) -> List[str]:
    """The §6 claim boundary and the sentence that qualifies it, if any.

    The one rule for every surface that shows the boundary: the report (through
    `_identity_table_lines`), the identity drawer's "This result", and its
    "Model in use", which passes the sentence worded for the loaded pipeline.
    "Nothing marked ✓" is read from `_identity_rows`, the rows the table shows.
    """
    lines = [IDENTITY_CLAIM_BOUNDARY]
    # **Not reported is not "nothing checked".** A response without a served
    # identity may come from a server that did check its inputs (one from
    # before contract M3c, for example); the panel was only not told, so it
    # says that. The demo answer keeps the line: nothing was served there.
    if kind == IDENTITY_NOT_REPORTED:
        lines.append(IDENTITY_CHECKS_NOT_REPORTED)
    elif not any(state.startswith(_VERIFIED_MARK)
                 for _, state, _ in _identity_rows(kind, identity)):
        lines.append(nothing_checked)
    return lines


def _identity_table_lines(kind: str, identity: Optional[Dict[str, Any]]) -> List[str]:
    """The identity table, the claim boundary under it and the fixed statement."""
    lines = ["| Part | State | Value |", "|------|-------|-------|"]
    lines += [f"| {part} | {state} | {value} |"
              for part, state, value in _identity_rows(kind, identity)]
    for sentence in _claim_boundary_lines(kind, identity):
        lines += ["", sentence]
    lines += ["", NOT_A_MEDICAL_DEVICE]
    return lines


def _identity_banners(kind: str, identity: Optional[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """The (headline, detail) banners for every non-normal state, most severe first."""
    if kind != _SERVED or identity is None:
        return [_IDENTITY_BANNERS[kind]]
    banners = []
    if _is_synthetic(identity):
        banners.append(_IDENTITY_BANNERS["synthetic"])
    if _field(_field(identity, "checkpoint"), "state") == "absent":
        banners.append(_IDENTITY_BANNERS["no_model"])
    elif _has_model(identity) and _field(_field(identity, "shortest_paths"), "state") == "absent":
        banners.append(_IDENTITY_BANNERS["no_sp"])
    return banners


# =============================================================================
# Export helpers — write ALL candidates (not just the previewed one) to a file
# =============================================================================
#: **The candidate columns follow the result's kind** (Spec 1 §2.1): every row
#: names it (`result_kind`), and a kind's file holds only its own values. The SP
#: record's status sits beside its value, and the value cell is empty where the
#: status has no number (the candidate-column convention: an empty cell means
#: no value, never 0). An evidence count is empty when no evidence package was
#: built (explanations off), never 0.
_CSV_BASE_COLUMNS = ["result_kind", "rank", "disease_id", "disease_name"]
_CSV_SCORE_COLUMNS = {
    SERVED_MIXTURE: ["served_score", "model_term", "sp_term_status", "sp_term_score",
                     "sp_term_mean_hop_distance", "sp_term_phenotypes_mapped",
                     "sp_term_phenotypes_submitted", "sp_term_mixed_as"],
    PATH_REASONING: ["path_score"],
    DEMO: [],
}
_CSV_EVIDENCE_COLUMNS = ["evidence_label", "evidence_mode", "evidence_summary",
                         "min_path_length", "num_direct_paths", "num_analogies",
                         "supporting_genes"]


def _candidate_csv_columns(kind: Optional[str]) -> List[str]:
    """The kind's candidate columns, in order: the evidence columns for the two
    served kinds only (the demo answer has none), the matches for every kind."""
    evidence = _CSV_EVIDENCE_COLUMNS if kind in (SERVED_MIXTURE, PATH_REASONING) else []
    return (_CSV_BASE_COLUMNS + _CSV_SCORE_COLUMNS.get(kind, []) + evidence
            + ["matching_phenotypes"])


# **What produced the result, on every row** (contract M3c). Appended after the
# candidate columns, never between them; one header row and no preamble, so the
# file stays a plain table. Each column holds one atomic part, the same on every row.
# Digests are written in full as "sha256:<hex>" -- the prefix names the
# algorithm and stops a spreadsheet reading an all-digit digest as a number. A
# value the identity holds as null is written as a word, never an empty cell:
#   absent        the part is absent: no checkpoint, table, manifest or record
#   none          nothing applies: no device or eta without a model, no digest
#                 for a model or graph supplied in memory, no device requested
#                 for a model supplied in memory
#   not_stated    a model supplied in memory that does not say its device
#   demo          every column of the demo answer (`served_identity` null: no
#                 pipeline configured, nothing served)
#   not_reported  every column of a response with no `served_identity`, or a
#                 value one left out
_SERVED_CSV_COLUMNS = [
    "served_scoring_mode",
    "served_checkpoint_state", "served_checkpoint_sha256",
    "served_kg_state", "served_kg_sha256", "served_split_manifest_sha256",
    "served_sp_state", "served_sp_build_id", "served_sp_max_hops",
    "served_dataset_record_state", "served_dataset_origin",
    "served_dataset_record_sha256",
    "served_device", "served_device_requested", "served_eta",
    "served_max_path_length", "served_path_length_penalty",
    "served_aggregation_method", "served_max_paths_per_candidate",
    "served_direct_path_optimization", "served_phenotype_normalisation_version",
]

def _fmt_num(value: Any) -> str:
    """A number to four places; empty for no value (never 0)."""
    return (f"{value:.4f}" if isinstance(value, (int, float)) and not isinstance(value, bool)
            else "")


#: A cell a spreadsheet would run as a formula starts with one of these (OWASP,
#: "CSV Injection").
_CSV_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


def _csv_text(value: Any) -> str:
    """A free-text cell, defused: one that a spreadsheet would run as a formula
    gets a leading apostrophe. Numeric columns are written by `_fmt_num` and
    never pass through here, so a negative number stays a number."""
    text = "" if value is None else str(value)
    return "'" + text if text.startswith(_CSV_FORMULA_START) else text


def _csv_value(value: Any, null_word: str) -> str:
    """One identity value as a CSV cell, never empty."""
    if value is _MISSING:
        return IDENTITY_NOT_REPORTED
    if value is None:
        return null_word
    if isinstance(value, str):
        return _csv_text(value) if value else IDENTITY_NOT_REPORTED
    return _plain_value(value)


def _csv_digest(value: Any, null_word: str) -> str:
    if _is_sha256(value):
        return f"sha256:{value}"
    return _csv_value(value, null_word)


def _served_csv_values(result: Dict[str, Any]) -> Dict[str, str]:
    """The `_SERVED_CSV_COLUMNS` of a result, the same for every row."""
    kind, identity = _identity_of(result)
    if kind != _SERVED or identity is None:
        return dict.fromkeys(_SERVED_CSV_COLUMNS, kind)

    checkpoint = _field(identity, "checkpoint")
    graph = _field(identity, "graph")
    shortest_paths = _field(identity, "shortest_paths")
    record = _field(identity, "dataset_record")
    settings = _field(identity, "settings")
    checkpoint_state = _field(checkpoint, "state")
    model = _has_model(identity)

    def setting(key: str, null_word: str = "none") -> str:
        return _csv_value(_field(settings, key), null_word)

    return {
        "served_scoring_mode": _csv_value(_field(identity, "scoring_mode"), IDENTITY_NOT_REPORTED),
        "served_checkpoint_state": _csv_value(checkpoint_state, IDENTITY_NOT_REPORTED),
        "served_checkpoint_sha256": _csv_digest(
            _field(checkpoint, "sha256"),
            "none" if checkpoint_state == "supplied_in_memory" else "absent",
        ),
        "served_kg_state": _csv_value(_field(graph, "state"), IDENTITY_NOT_REPORTED),
        "served_kg_sha256": _csv_digest(_field(graph, "kg_sha256"), "none"),
        "served_split_manifest_sha256": _csv_digest(
            _field(graph, "split_manifest_sha256"), "absent"
        ),
        "served_sp_state": _csv_value(_field(shortest_paths, "state"), IDENTITY_NOT_REPORTED),
        "served_sp_build_id": _csv_value(_field(shortest_paths, "build_id"), "absent"),
        "served_sp_max_hops": _csv_value(_field(shortest_paths, "max_hops"), "absent"),
        "served_dataset_record_state": _csv_value(_field(record, "state"), IDENTITY_NOT_REPORTED),
        "served_dataset_origin": _csv_value(_field(record, "origin"), "absent"),
        "served_dataset_record_sha256": _csv_digest(_field(record, "sha256"), "absent"),
        "served_device": setting("device", "not_stated" if model else "none"),
        "served_device_requested": setting("device_requested"),
        "served_eta": setting("eta"),
        "served_max_path_length": setting("max_path_length", IDENTITY_NOT_REPORTED),
        "served_path_length_penalty": setting("path_length_penalty", IDENTITY_NOT_REPORTED),
        "served_aggregation_method": setting("aggregation_method", IDENTITY_NOT_REPORTED),
        "served_max_paths_per_candidate": setting(
            "max_paths_per_candidate", IDENTITY_NOT_REPORTED
        ),
        "served_direct_path_optimization": setting(
            "use_direct_path_optimization", IDENTITY_NOT_REPORTED
        ),
        "served_phenotype_normalisation_version": setting(
            "phenotype_normalisation_version", IDENTITY_NOT_REPORTED
        ),
    }


# KG disease ids are stored as "<source>:<CURIE>", e.g. "mondo:MONDO:0019441".
# The lowercase source tag duplicates the CURIE namespace and reads as redundant.
# Strip it — but conservatively: ONLY when the source maps to the inner namespace,
# so a genuine cross-namespace id like "mondo:OMIM:123" is left untouched.
_DISEASE_SOURCE_TO_NS = {
    "mondo": "MONDO",
    "omim": "OMIM",
    "orpha": "ORPHA",
    "orphanet": "ORPHA",
    "decipher": "DECIPHER",
}


def _clean_disease_id(disease_id: Any) -> str:
    """Drop a redundant lowercase source tag: 'mondo:MONDO:0019441' -> 'MONDO:0019441'.

    Only strips when the source maps to (or literally equals) the inner CURIE
    namespace; anything else — a bare CURIE, a cross-namespace id, or an
    unknown shape — is returned unchanged.
    """
    text = str(disease_id or "").strip()
    parts = text.split(":", 2)
    if len(parts) == 3:
        source, namespace, local = parts
        expected = _DISEASE_SOURCE_TO_NS.get(source.lower())
        if (expected is not None and expected == namespace.upper()) or \
                source.lower() == namespace.lower():
            return f"{namespace}:{local}"
    return text


def _build_results_csv(result: Dict[str, Any]) -> str:
    """One row per candidate — the tabular summary, for spreadsheet analysis —
    in the result kind's columns (`_candidate_csv_columns`), with what produced
    the result (`_SERVED_CSV_COLUMNS`) repeated on each row."""
    kind = _result_kind(result)
    served = _served_csv_values(result)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_candidate_csv_columns(kind) + _SERVED_CSV_COLUMNS,
                            extrasaction="ignore")
    writer.writeheader()
    for c in result.get("candidates", []) if kind is not None else []:
        pkg = c.get("evidence_package")
        sp = c.get("sp_term") if isinstance(c.get("sp_term"), Mapping) else {}
        writer.writerow({
            "result_kind": kind,
            "rank": c.get("rank"),
            "disease_id": _csv_text(_clean_disease_id(c.get("disease_id", ""))),
            "disease_name": _csv_text(c.get("disease_name", "")),
            "served_score": _fmt_num(c.get("served_score")),
            "model_term": _fmt_num(c.get("model_term")),
            "sp_term_status": _csv_text(sp.get("status") or ""),
            "sp_term_score": _fmt_num(sp.get("score")),
            "sp_term_mean_hop_distance": _fmt_num(sp.get("mean_hop_distance")),
            "sp_term_phenotypes_mapped": sp.get("phenotypes_mapped", ""),
            "sp_term_phenotypes_submitted": sp.get("phenotypes_submitted", ""),
            "sp_term_mixed_as": _fmt_num(sp.get("mixed_as")),
            "path_score": _fmt_num(c.get("path_score")),
            "evidence_label": _csv_text(c.get("evidence_label") or ""),
            "evidence_mode": _csv_text((pkg or {}).get("mode") or ""),
            "evidence_summary": _csv_text(((pkg or {}).get("summary") or "")
                                          .replace("\n", " ").strip()),
            "min_path_length": (pkg or {}).get("min_path_length", ""),
            "num_direct_paths": len(pkg.get("direct_paths") or []) if pkg else "",
            "num_analogies": len(pkg.get("analogies") or []) if pkg else "",
            "matching_phenotypes": _csv_text("; ".join(c.get("matching_phenotypes") or [])),
            "supporting_genes": _csv_text("; ".join(c.get("supporting_genes") or [])),
            **served,
        })
    return buf.getvalue()


def _build_results_report_md(result: Dict[str, Any]) -> str:
    """Full human-readable report: session meta, what produced the result (the
    served identity), summary table, and per-candidate evidence and explanation
    for EVERY candidate (reuses the preview formatters)."""
    candidates = result.get("candidates", [])
    result_kind = _result_kind(result)
    elapsed = result.get("inference_time_ms")
    lines = ["# SHEPHERD-Advanced Diagnosis Report", ""]
    for key, value in (
        ("Patient ID", result.get("patient_id")),
        ("Session ID", result.get("session_id")),
        ("Timestamp", result.get("timestamp")),
        # A missing time is left out, never printed as 0.0.
        ("Inference time (ms)", f"{elapsed:.1f}" if isinstance(elapsed, (int, float))
         and not isinstance(elapsed, bool) else None),
        ("Result kind", f"`{result_kind}`" if result_kind else (
            f"`{result.get('result_kind')}` (not readable here)"
            if isinstance(result.get("result_kind"), str) else "not reported")),
        ("Candidates", len(candidates)),
    ):
        if value not in (None, ""):
            lines.append(f"- **{key}**: {value}")
    query = result.get("_query_phenotypes") or []
    if query:
        lines.append(f"- **Query phenotypes**: {', '.join(query)}")
    warnings = result.get("warnings") or []
    if warnings:
        lines.append(f"- **Warnings**: {'; '.join(warnings)}")

    # **What produced this result, before any candidate** (contract M3c). It
    # replaces a "Model version" line that always read 1.0.0. The skip-empty
    # loop above is not used here: an absent part is printed as its state, so
    # an archived report says what was missing rather than omitting it.
    kind, identity = _identity_of(result)
    lines += ["", "## What produced this result", ""]
    for headline, detail in _identity_banners(kind, identity):
        lines += [f"> **{headline}** {detail}", ""]
    lines += _identity_table_lines(kind, identity)

    lines += ["", "## Ranked candidates", ""]
    if result_kind is not None:     # else the table line below says why
        lines += [f"*{kind_label(result).rstrip('.')}.*", ""]
    lines += [_format_results_table(candidates, result_kind, _shows_sp(result)),
              "", "## Per-candidate detail"]
    for c in candidates if result_kind is not None else []:
        score = _score(c, result_kind)
        lines += [
            "",
            f"### #{c.get('rank')} — {c.get('disease_name')}",
            f"`{_clean_disease_id(c.get('disease_id', ''))}`"
            + (f" · score {_num(score)}" if result_kind != DEMO else ""),
            "",
        ]
        if result_kind == DEMO:     # it never carries evidence, whatever the option
            lines += ["*The demo answer carries no evidence or explanation.*", "", "---"]
            continue
        lines += [
            "**Evidence**",
            "",
            _format_evidence_detail(c),
            "",
            "**Full explanation**",
            "",
            _format_full_explanation(c),
            "",
            "---",
        ]
    return "\n".join(lines)


def _slug(value: Any, default: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("_") or default


# One export directory reused across diagnoses in this process. Files are named by
# session signature so they don't collide; periodic cleanup is future work.
_EXPORT_DIR: Optional[Path] = None


def _export_dir() -> Path:
    global _EXPORT_DIR
    if _EXPORT_DIR is None or not _EXPORT_DIR.exists():
        _EXPORT_DIR = Path(tempfile.mkdtemp(prefix="shepherd_export_"))
    return _EXPORT_DIR


def _export_basename(result: Dict[str, Any]) -> str:
    """diagnosis_[SYNTHETIC_|DEMO_]<patient>_<session-signature> — distinct per
    run for easy filing.

    **A file name carries the two states a copied file must not lose** (contract
    M3c): SYNTHETIC for a result from a synthetic workspace, DEMO for the demo
    answer. A CSV has no place for a banner, and a file is often seen by its
    name alone.
    """
    patient = _slug(result.get("patient_id"), "webui_patient")
    session = _slug(result.get("session_id") or result.get("timestamp"), "run")
    kind, identity = _identity_of(result)
    if kind == IDENTITY_DEMO:
        marker = "DEMO_"
    elif kind == _SERVED and identity is not None and _is_synthetic(identity):
        marker = "SYNTHETIC_"
    else:
        marker = ""
    return f"diagnosis_{marker}{patient}_{session}"


def _write_exports(result: Dict[str, Any]) -> Tuple[str, str]:
    """Generate both export files for a result up front (so the DownloadButtons
    can serve them on the first click). Returns (csv_path, md_path)."""
    base = _export_basename(result)
    directory = _export_dir()
    csv_path = directory / f"{base}.csv"
    md_path = directory / f"{base}_report.md"
    csv_path.write_text(_build_results_csv(result), encoding="utf-8")
    md_path.write_text(_build_results_report_md(result), encoding="utf-8")
    return str(csv_path), str(md_path)



# =============================================================================
# The page: help texts and fixed words
# =============================================================================
#: DownloadButton update for the "no downloadable result" state.
#: No result: the exports are not shown at all (the mockup's first screen has
#: none), but stay mounted (``"hidden"``), so the next result only shows them.
_EXPORTS_HIDDEN = gr.update(value=None, interactive=False, visible="hidden")
#: A result whose export files could not be written (or that has no
#: candidate): the exports are shown, disabled, beside the reason.
_DOWNLOAD_DISABLED = gr.update(value=None, interactive=False, visible=True)

#: The fields' help, behind the shared ⓘ (`src/webui/components/help.py`):
#: ``{elem_id: (title, text)}``, registered by the tab builder. Only a
#: state-bearing text stays visible.
FIELD_HELP = {
    "dx_hpo_terms": (
        "HPO terms",
        "Paste HPO ids in any format — one per line, or comma/space-separated. Names and "
        "stray punctuation are ignored; only HP:####### ids are used. Repeats are sent as "
        "typed; the service removes them and says how many.",
    ),
    "dx_patient_id": (
        "Patient ID (optional)",
        "An identifier for this diagnosis, carried by the result and its exports. Left blank, "
        "the server gives the result its own id, so results from different browsers never "
        "share one.",
    ),
    "dx_top_k": (
        "Top-K",
        "Number of candidate diagnoses to return (1 to 20).",
    ),
    "dx_explanations": (
        "Explanations",
        "Ask for each candidate's evidence and written explanation. Off, the ranking is the "
        "same, and Evidence, Explanation and Paths stay empty.",
    ),
}

#: Help that rendered HTML names by key (`help_button`), registered as topics.
TOPIC_HELP = {
    "dx-card": card.CARD_HELP,
    "dx-badge": (
        "Compared with the model in use",
        "The whole served identity is compared, part by part: checkpoint, knowledge graph, "
        "workspace manifest, shortest-path table, dataset record, scoring mode, device, "
        "mixing weight eta, path settings and phenotype rule — never the checkpoint alone. "
        "The model in use is the card's last status read. A read that failed, or one older "
        "than the result, never shows \"Produced by the model in use\". The result, its CSV "
        "and its report keep their own identity whatever the badge says.",
    ),
    "dx-score": (
        "Score",
        SCORE_NOTE + " A score of 0.72 does not mean a 72% chance of the disease.\n\n"
        "The kind above the table says what it is. With a model: η × model term + "
        "(1 − η) × SP, or the model term alone without an SP table. With no model: "
        "the path score.\n\n"
        "SP 0.273 (3/4): from 3 of the 4 phenotypes submitted. SP in words: no number.",
    ),
    "dx-exports": (
        "Exports",
        "CSV = one row per candidate (all Top-K) for spreadsheets, in the result kind's "
        "columns (result_kind on every row) · report = full evidence + explanation for "
        "every candidate. Both say what produced the result (the served_ columns; the "
        "report's identity section).",
    ),
    "dx-candidate": (
        "Candidate details",
        "Select a row of Diagnosis results (click it, or focus it and press Enter) to show "
        "that candidate here. Evidence, explanation, matching phenotypes and paths come from "
        "this result only.",
    ),
    "dx-run-no-model": (
        "Run Diagnosis is disabled",
        "No model loaded: no pipeline is configured on this server, so nothing would compute "
        "a diagnosis. An operator loads one with \"Load a model…\" on the card above. (An API "
        "client asking now gets the demo answer, marked as such.)",
    ),
    "dx-run-change": (
        "Run Diagnosis is disabled",
        "A model change is open: finish it (Reload with …) or cancel it (Keep … or Cancel) "
        "first. Diagnoses wait while the pipeline reloads.",
    ),
    "dx-run-loading": (
        "Run Diagnosis is disabled",
        "This page started a model load: the pipeline reloads for every session, and the server "
        "answers no diagnosis until it finishes. Closing the change-model drawer does not stop "
        "it.",
    ),
    **{f"dx-banner-{key}": (headline, detail) for key, (headline, detail) in IDENTITY_BANNERS.items()},
}

#: Why Run Diagnosis is off, as visible text under it.
RUN_DISABLED_NO_MODEL = "Disabled: no model loaded. Load one with “Load a model…” above."
RUN_DISABLED_NO_MODEL_READ_ONLY = (
    "Disabled: no model loaded. An operator on the server's machine can load one.")
RUN_DISABLED_CHANGE = "Disabled: finish or cancel the model change first."
RUN_DISABLED_LOADING = "Disabled: a model load is in progress; diagnoses wait until it finishes."

#: The change drawer's link to Research › Models: each "›" held to the word
#: before it by a no-break space, so it is never alone on a line.
COMPARE_LINK = "Compare in Research\u00a0› Models\u00a0›"

#: The details segments: (label, value); the counts are added per candidate.
SEGMENTS = (("Evidence", "evidence"), ("Explanation", "explanation"),
            ("Matching phenotypes", "phenotypes"), ("Paths", "paths"))
_SEGMENT_VALUES = tuple(value for _, value in SEGMENTS)
#: The identity drawer's views.
IDENTITY_VIEWS = (("Model in use", "model"), ("This result", "result"))

#: What the matching phenotypes are (the API's own definition, in short).
MATCHING_NOTE = (
    "The query's phenotypes, as used, that the knowledge graph annotates directly to this "
    "disease (the exact term; not ancestors or descendants, not gene-mediated). An empty "
    "list does not mean no clinical overlap."
)

_DOT_CLASS = {
    "Strong path support": "strong",
    "Weak path support": "weak",
    "Analogy-based (no direct KG path)": "analogy",
    "Insufficient evidence": "none",
}
_CHECK_SVG = ('<svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true" focusable="false">'
              '<circle cx="8" cy="8" r="7" fill="currentColor"/><path d="M4.6 8.3l2.3 2.3 4.5-4.9" '
              'fill="none" stroke="#fff" stroke-width="1.9" stroke-linecap="round" '
              'stroke-linejoin="round"/></svg>')


# =============================================================================
# The result, as the page shows it
# =============================================================================
def _num(value: Any) -> str:
    return (f"{value:.3f}" if isinstance(value, (int, float)) and not isinstance(value, bool)
            else "—")


def _produced_at(result: Optional[Mapping[str, Any]]) -> Optional[str]:
    """When the result was produced: the server's timestamp (this machine's
    local time), else when the page received it."""
    if not isinstance(result, Mapping):
        return None
    stamp = result.get("timestamp")
    return stamp if card.parse_local(stamp) is not None else result.get("_received_at")


def results_header_html(result: Optional[Mapping[str, Any]]) -> str:
    """"Diagnosis results" and, at its right, how many candidates, how long and
    when it was produced (or "no result yet")."""
    if not isinstance(result, Mapping):
        right = '<span class="shep-panel-right">no result yet</span>'
    else:
        n = len(result.get("candidates") or [])
        ms = result.get("inference_time_ms")
        took = f" · {ms:.1f} ms" if isinstance(ms, (int, float)) else ""
        right = (f'<span class="shep-panel-right"><b>{n} candidate{"s" if n != 1 else ""}</b>'
                 f'{_e(took)} · produced {_e(card.hhmmss(_produced_at(result)))}</span>')
    return f'<div class="shep-panel-hd"><h2>Diagnosis results</h2>{right}</div>'


def empty_results_html(no_model: bool = False, notice: Optional[str] = None) -> str:
    """The results panel with no result: what to do, or `notice` (a refusal,
    an error, inputs changed since the last run)."""
    if notice:
        return f'<div class="shep-dx-notice">{notice}</div>'
    first = ("Run Diagnosis needs a loaded model. The card on the left says what is loaded."
             if no_model else
             "Enter the HPO terms, check the model in use on the left, then Run Diagnosis.")
    return ('<div class="shep-dx-empty"><p class="shep-dx-empty-h">No result yet</p>'
            f'<p>{_e(first)}</p><p class="shep-note">Each result says what produced it, compared '
            'with the model in use.</p></div>')


def compare_with_served(result: Mapping[str, Any], state: Mapping[str, Any]) -> Dict[str, Any]:
    """The result's served identity against the card's last read, part by part
    (`_identity_rows`, the one reading of an identity): ``equal``, ``differs``
    (with the parts) or ``unknown`` (with one reason)."""
    kind, identity = _identity_of(result)
    total = len(_IDENTITY_PARTS)
    keep = "This result keeps its own identity; its exports use it."
    mode = card.card_mode(state, _produced_at(result))
    if kind == IDENTITY_NOT_REPORTED:
        return {"state": "unknown", "total": total, "differ": [],
                "reason": f"The response did not say what produced it. {keep}"}
    if mode == "checking":
        return {"state": "unknown", "total": total, "differ": [],
                "reason": f"The model in use has not been read yet. {keep}"}
    if mode in ("unknown", "cannot_check"):
        return {"state": "unknown", "total": total, "differ": [],
                "reason": (f"The status read failed ({card.problem_sentence(state.get('problem'))}), "
                           f"so \"Produced by the model in use\" is not shown. {keep}")}
    if mode == "stale":
        return {"state": "unknown", "total": total, "differ": [],
                "reason": (f"The status was last read at {card.hhmmss(state.get('good_at'))}, "
                           f"before this result was produced "
                           f"({card.hhmmss(_produced_at(result))}), so \"Produced by the model in "
                           f"use\" is not shown. {keep}")}
    status = state.get("status") or {}
    served = status.get("served_identity") if status.get("initialized") else None
    served_kind_ = _SERVED if isinstance(served, dict) else IDENTITY_DEMO
    if kind == IDENTITY_DEMO and served_kind_ == IDENTITY_DEMO:
        return {"state": "unknown", "total": total, "differ": [],
                "reason": "This is the demo answer: no pipeline is configured, so no model "
                          "produced it."}
    mine = _identity_rows(kind, identity)
    now = _identity_rows(served_kind_, served if isinstance(served, dict) else None)
    differ = [part for (part, s1, v1), (_, s2, v2) in zip(mine, now, strict=True)
              if (s1, v1) != (s2, v2)]
    return {"state": "differs" if differ else "equal", "total": total, "differ": differ,
            "reason": None}


def _parts_differ(n: int) -> str:
    return f"{n} part differs" if n == 1 else f"{n} parts differ"


def badge_html(result: Optional[Mapping[str, Any]], state: Mapping[str, Any]) -> str:
    """The results badge, in the card's noun; its › opens "This result"."""
    if not isinstance(result, Mapping):
        return ""
    compared = compare_with_served(result, state)
    help_html = help_button("dx-badge", "Compared with the model in use")
    if compared["state"] == "equal":
        return (f'<div class="shep-dx-badge-row"><button type="button" class="shep-dx-badge '
                f'shep-dx-badge-ok" data-key="result">{_CHECK_SVG} Produced by the model in use '
                f'<span aria-hidden="true">›</span></button>{help_html}</div>')
    if compared["state"] == "differs":
        n = len(compared["differ"])
        return (f'<div class="shep-dx-badge-row"><button type="button" class="shep-dx-badge '
                f'shep-dx-badge-amber" data-key="result"><span aria-hidden="true">⇄</span> '
                f'Produced by an earlier model or pipeline — <span class="shep-dx-u">'
                f'{_parts_differ(n)} ›</span></button>{help_html}</div>')
    return (f'<div class="shep-dx-badge-row"><button type="button" class="shep-dx-badge '
            f'shep-dx-badge-grey" data-key="result"><span aria-hidden="true">?</span> Cannot '
            f'check against the model in use yet</button>{help_html}</div>'
            f'<p class="shep-dx-badge-why">{_e(compared["reason"])}</p>')


_BANNER_KEYS = {headline: key for key, (headline, _detail) in IDENTITY_BANNERS.items()}


def banners_html(result: Mapping[str, Any]) -> str:
    """The result's identity banners: each headline visible, its detail behind
    the ⓘ (the banners' own words, `renderers.IDENTITY_BANNERS`)."""
    kind, identity = _identity_of(result)
    out = []
    for headline, _detail in _identity_banners(kind, identity):
        key = _BANNER_KEYS.get(headline, "")
        tone = "red" if key in ("demo", "not_reported", "no_model") else "amber"
        out.append(f'<div class="shep-dx-banner shep-dx-banner-{tone}" role="note">'
                   f'<span class="shep-dx-banner-ic" aria-hidden="true">⚠</span>'
                   f'<b>{_e(headline)}</b>'
                   f'{help_button(f"dx-banner-{key}", headline) if key else ""}</div>')
    return "".join(out)


def candidates_table_html(candidates: List[Mapping[str, Any]], selected: Any,
                          kind: Optional[str], shows_sp: bool = False) -> str:
    """The ranked candidates in the kind's columns; a row is selected by click,
    or by Enter or Space when focused. Every value is escaped; a long name is
    clamped to two lines with the full name in its title. A mixture shows its
    score and model term, and its SP record in words when a table is served
    (`shows_sp`); a path-reasoning result its path score; the demo no score."""
    if kind is None:
        return f'<div class="shep-reason">{_e(KIND_NOT_REPORTED)}</div>'
    if not candidates:
        return '<div class="shep-reason">No candidates found.</div>'
    rows = []
    selected_class = ' class="shep-on"'
    for c in candidates:
        rank = c.get("rank")
        on = rank == selected
        name = _e(c.get("disease_name"))
        cells = (f'<td class="shep-num">{_e(rank)}</td><td class="shep-dx-dis">'
                 f'<span class="shep-dx-clamp" title="{name}">{name}</span></td>')
        if kind != DEMO:
            cells += f'<td class="shep-num">{_num(_score(c, kind))}</td>'
        if kind == SERVED_MIXTURE:
            cells += f'<td class="shep-num">{_num(c.get("model_term"))}</td>'
            if shows_sp:
                # A number stays on one line; a status in words may wrap.
                sp_term = c.get("sp_term")
                words = not (isinstance(sp_term, Mapping)
                             and sp_term.get("status") in SP_NUMERIC_STATUSES)
                cells += (f'<td class="shep-num{" shep-dx-sp" if words else ""}">'
                          f'{_e(sp_text(sp_term, compact=True))}</td>')
        if kind != DEMO:
            label = c.get("evidence_label") or "—"
            cells += (f'<td><span class="shep-dx-dot shep-dx-dot-{_DOT_CLASS.get(label, "none")}"'
                      f' aria-hidden="true"></span>{_e(label)}</td>')
        rows.append(f'<tr data-key="{_e(rank)}" tabindex="0" aria-selected='
                    f'"{"true" if on else "false"}"{selected_class if on else ""}>{cells}</tr>')
    head = '<th scope="col" class="shep-num">Rank</th><th scope="col">Disease</th>'
    if kind != DEMO:
        head += (f'<th scope="col" class="shep-num">Score{help_button("dx-score", "Score")}'
                 '</th>')
    if kind == SERVED_MIXTURE:
        head += '<th scope="col" class="shep-num">Model term</th>'
        if shows_sp:
            head += '<th scope="col" class="shep-num">SP</th>'
    if kind != DEMO:
        head += '<th scope="col">Evidence</th>'
    return ('<div class="shep-tbl-wrap shep-dx-tbl-wrap"><table class="shep-tbl shep-dx-tbl" '
            f'aria-label="Ranked candidates"><thead><tr>{head}</tr></thead><tbody>'
            f'{"".join(rows)}</tbody></table></div>')


def results_body_html(result: Mapping[str, Any], selected: Any) -> str:
    """Under the badge: the banners, the result's warnings, the kind's label
    (what the Score column holds, visible text) and the table."""
    warnings = result.get("warnings") or []
    note = (f'<p class="shep-dx-warnings">⚠ Warnings: {_e("; ".join(str(w) for w in warnings))}'
            '</p>' if warnings else "")
    kind = _result_kind(result)
    label = (f'<p class="shep-dx-kind">{_e(kind_label(result))}</p>'
             if kind is not None else "")
    return banners_html(result) + note + label + candidates_table_html(
        result.get("candidates") or [], selected, kind, _shows_sp(result))


# ----------------------------------------------------------------- details
def _candidate(result: Optional[Mapping[str, Any]], rank: Any) -> Optional[Mapping[str, Any]]:
    if not isinstance(result, Mapping):
        return None
    return next((c for c in result.get("candidates") or [] if c.get("rank") == rank), None)


def _paths(candidate: Mapping[str, Any]) -> List[List[str]]:
    pkg = candidate.get("evidence_package") or {}
    paths = [p for p in pkg.get("direct_paths") or [] if isinstance(p, list)]
    for analogy in pkg.get("analogies") or []:
        paths += [p for p in (analogy or {}).get("known_paths") or [] if isinstance(p, list)]
    return paths


def segment_choices(candidate: Optional[Mapping[str, Any]]) -> List[Tuple[str, str]]:
    """The segments' labels, with the counts this candidate has."""
    if candidate is None:
        return list(SEGMENTS)
    counts = {"phenotypes": len(candidate.get("matching_phenotypes") or []),
              "paths": len(_paths(candidate))}
    return [(f"{label} {counts[value]}" if value in counts else label, value)
            for label, value in SEGMENTS]


def details_header_html(candidate: Optional[Mapping[str, Any]], kind: Optional[str] = None,
                        shows_sp: bool = False) -> str:
    """The selected candidate's name, id and its kind's values: the score and
    model term of a mixture (its SP record on a line of its own when a table is
    served), the score alone with no model, none in the demo answer."""
    if candidate is None:
        return '<div class="shep-panel-hd"><h2>Candidate details</h2></div>'
    values = ""
    if kind in (SERVED_MIXTURE, PATH_REASONING):
        values = f" Score <b>{_num(_score(candidate, kind))}</b>"
    if kind == SERVED_MIXTURE:
        values += f" · Model term {_num(candidate.get('model_term'))}"
    sp_line = (f'<p class="shep-dx-det-sub">SP {_e(sp_text(candidate.get("sp_term")))}</p>'
               if kind == SERVED_MIXTURE and shows_sp else "")
    return (f'<div class="shep-panel-hd shep-dx-det-hd"><h2><span class="shep-dx-rank">'
            f'#{_e(candidate.get("rank"))}</span> {_e(candidate.get("disease_name"))}</h2>'
            f'<p class="shep-dx-det-sub"><code>{_e(_clean_disease_id(candidate.get("disease_id")))}'
            f'</code>{values}</p>{sp_line}</div>')


def details_empty_html() -> str:
    return ('<div class="shep-dx-empty"><p>After a diagnosis, the selected candidate\'s evidence, '
            'explanation, matching phenotypes and paths appear here.</p></div>')


def evidence_html(candidate: Mapping[str, Any], explanations: bool = True) -> str:
    """The candidate's evidence package, escaped: its label and summary, the
    path or analogy evidence, and the supporting genes."""
    pkg = candidate.get("evidence_package")
    if not pkg:
        why = ("Explanations were off for this run (Options), so no evidence package was asked "
               "for." if not explanations else "No evidence package was returned for this "
               "candidate.")
        return f'<p class="shep-muted">{_e(why)}</p>'
    label = candidate.get("evidence_label") or pkg.get("evidence_label") or "—"
    parts = [f'<p class="shep-dx-ev-label"><span class="shep-dx-dot shep-dx-dot-'
             f'{_DOT_CLASS.get(label, "none")}" aria-hidden="true"></span><b>{_e(label)}</b></p>']
    if pkg.get("summary"):
        parts.append(f'<blockquote class="shep-dx-quote">{_e(pkg.get("summary"))}</blockquote>')
    mode = pkg.get("mode")
    if mode == "direct_path":
        n = len(pkg.get("direct_paths") or [])
        hops = pkg.get("min_path_length")
        hop_text = f"{hops} hop{'s' if hops != 1 else ''}" if isinstance(hops, int) else "not stated"
        parts.append('<h3 class="shep-dx-h">Mode A — Direct path evidence</h3>'
                     f'<p>Shortest: {_e(hop_text)} · {n} direct path{"s" if n != 1 else ""}, '
                     'listed under <button type="button" class="shep-dx-link" data-key="seg:paths">'
                     'Paths ›</button></p>')
    elif mode == "analogy_based":
        items = []
        for analogy in pkg.get("analogies") or []:
            analogy = analogy or {}
            similarity = analogy.get("embedding_similarity")
            sim = f" (similarity {similarity:.2f})" if isinstance(similarity, (int, float)) else ""
            shared = analogy.get("shared_features") or []
            items.append(f'<li><b>{_e(analogy.get("similar_gene_name") or "?")}</b>{_e(sim)}'
                         + (f'<br><span class="shep-sub">Shared: {_e(", ".join(map(str, shared)))}'
                            '</span>' if shared else "") + "</li>")
        parts.append('<h3 class="shep-dx-h">Mode B — Analogy-based evidence</h3>'
                     f'<ul class="shep-dx-list">{"".join(items)}</ul>')
    elif mode == "insufficient":
        parts.append('<p><b>No usable evidence found.</b> This candidate should be validated via '
                     'independent clinical judgment.</p>')
    genes = candidate.get("supporting_genes") or []
    if genes:
        parts.append('<h3 class="shep-dx-h">Supporting genes</h3><p class="shep-dx-genes">'
                     + "".join(f'<span class="shep-dx-gene">{_e(g)}</span>' for g in genes) + "</p>")
    return "".join(parts)


def phenotypes_html(candidate: Mapping[str, Any]) -> str:
    found = candidate.get("matching_phenotypes") or []
    listed = ("".join(f'<li><code>{_e(p)}</code></li>' for p in found) if found else "")
    body = (f'<ul class="shep-dx-list">{listed}</ul>' if found else
            '<p class="shep-muted">None of the query\'s phenotypes is annotated directly to this '
            'disease.</p>')
    return body + f'<p class="shep-note">{_e(MATCHING_NOTE)}</p>'


def paths_html(candidate: Mapping[str, Any]) -> str:
    paths = _paths(candidate)
    if not paths:
        return '<p class="shep-muted">No path is listed for this candidate.</p>'
    items = "".join(
        "<li>" + ' <span class="shep-dx-arrow" aria-label="to">→</span> '.join(
            f"<code>{_e(_node_display_name(str(node)))}</code>" for node in path) + "</li>"
        for path in paths)
    return f'<ol class="shep-dx-paths">{items}</ol>'


def details_footer_html(result: Optional[Mapping[str, Any]], rank: Any) -> str:
    if not isinstance(result, Mapping) or _candidate(result, rank) is None:
        return ""
    candidates = result.get("candidates") or []
    index = next((i for i, c in enumerate(candidates, 1) if c.get("rank") == rank), 0)
    return (f'<div class="shep-dx-det-ft"><span>Candidate {index} of {len(candidates)}</span>'
            f'{help_button("dx-candidate", "Candidate details")}</div>')


#: The page's class that hides a view (`PAGE_CSS`), and the views' own classes
#: (never empty: Gradio 6.30 ignores an update to no class at all).
VIEW_OFF = "shep-dx-off"
_HTML_VIEW = ("shep-html",)     # `shell.plain_html`'s class
_MD_VIEW = ("shep-dx-md",)


def view_update(shown: bool, *classes: str, value: Any = _MISSING, **props: Any) -> Dict[str, Any]:
    """A view shown or hidden **by a class, never by ``visible``**: in Gradio
    6.30, ``visible`` unmounts and re-mounts a component (and a column's
    children), and switching it while the content changes leaves detached
    page elements behind each time (measured after garbage collection: 77 and
    114 per switch of the details' and the identity drawer's views, about 200
    and 600 with each view in a column of its own); a class toggle keeps the
    component mounted (flat). `classes` are the component's own (its update
    replaces them all); `value` and `props` are sent only for a view that is
    shown (a hidden view keeps its last content until it is shown again, with
    its new content)."""
    update = {"elem_classes": [*classes] if shown else [*classes, VIEW_OFF]}
    if shown:
        if value is not _MISSING:
            update["value"] = value
        update.update(props)
    return gr.update(**update)


def details_outputs(result: Optional[Mapping[str, Any]], rank: Any,
                    segment: str = "evidence") -> Dict[str, Any]:
    """Everything the details panel shows for one candidate and segment (each
    view shown or hidden by `view_update`)."""
    candidate = _candidate(result, rank)
    if candidate is None:
        return {"details_hd": details_header_html(None),
                "details_seg": view_update(False, "shep-seg"),
                "details_html": view_update(True, *_HTML_VIEW, value=details_empty_html()),
                "details_md": view_update(False, *_MD_VIEW), "details_ft": ""}
    segment = segment if segment in _SEGMENT_VALUES else "evidence"
    explanations = bool(result.get("_explanations", True))
    html_: Optional[str] = None
    if segment == "evidence":
        html_ = evidence_html(candidate, explanations)
    elif segment == "phenotypes":
        html_ = phenotypes_html(candidate)
    elif segment == "paths":
        html_ = paths_html(candidate)
    explanation = candidate.get("explanation")
    if segment == "explanation" and not explanation:
        why = ("No explanation was returned for this candidate." if explanations
               else "Explanations were off for this run (Options).")
        html_ = f'<p class="shep-muted">{why}</p>'
    return {
        "details_hd": details_header_html(candidate, _result_kind(result), _shows_sp(result)),
        "details_seg": view_update(True, "shep-seg", value=segment,
                                   choices=segment_choices(candidate)),
        "details_html": view_update(html_ is not None, *_HTML_VIEW, value=html_),
        "details_md": view_update(html_ is None, *_MD_VIEW, value=explanation),
        "details_ft": details_footer_html(result, rank),
    }


# ----------------------------------------------------------------- identity drawer
def _short_digests(cell: str) -> str:
    return _SHA256_RE.sub(lambda m: m.group(0)[:_SHORT_HEX] + "…", cell)


def result_identity_md(result: Optional[Mapping[str, Any]], state: Mapping[str, Any]) -> str:
    """The identity drawer's "This result": only what the result carries, each
    part beside what is in use now when the card's read can be compared, then
    the claim boundary and the medical-device line."""
    if not isinstance(result, Mapping):
        return "No result yet: run a diagnosis to see what produced it."
    kind, identity = _identity_of(result)
    compared = compare_with_served(result, state)
    mine = _identity_rows(kind, identity)
    lines: List[str] = []
    if compared["state"] in ("equal", "differs"):
        status = state.get("status") or {}
        served = status.get("served_identity") if status.get("initialized") else None
        now = _identity_rows(_SERVED if isinstance(served, dict) else IDENTITY_DEMO,
                             served if isinstance(served, dict) else None)
        when = card.hhmmss(state.get("good_at"))
        if compared["state"] == "differs":
            n = len(compared["differ"])
            verb = "differs" if n == 1 else "differ"
            lines += [f"**This result came from an earlier model or pipeline:** {n} of "
                      f"{compared['total']} parts {verb} from what is in use now (status read "
                      f"{when}). The result and its CSV and report keep their own identity.", ""]
        else:
            lines += [f"**Produced by the model in use:** all {compared['total']} parts are the "
                      f"same as what is in use now (status read {when}).", ""]
        lines += ["| Part | This result | In use now |", "|------|-------------|------------|"]
        for (part, state_, value), (_, now_state, now_value) in zip(mine, now, strict=True):
            this = _short_digests(f"{state_} · {value}")
            there = ("= same" if (state_, value) == (now_state, now_value)
                     else _short_digests(f"{now_state} · {now_value}") + " **· differs**")
            lines.append(f"| {part} | {this} | {there} |")
    else:
        lines += [f"**Cannot check against the model in use yet:** {compared['reason']}", "",
                  "| Part | State | Value |", "|------|-------|-------|"]
        lines += [f"| {part} | {state_} | {_short_digests(value)} |" for part, state_, value in mine]
    for sentence in _claim_boundary_lines(kind, identity):
        lines += ["", sentence]
    lines += ["", f"**{NOT_A_MEDICAL_DEVICE}**"]
    return "\n".join(lines)


def result_digests_html(result: Optional[Mapping[str, Any]]) -> str:
    """"This result in full": the full digests the result carries, each with
    Copy."""
    if not isinstance(result, Mapping):
        return ""
    kind, identity = _identity_of(result)
    if kind != _SERVED or identity is None:
        return ""
    record = _field(identity, "dataset_record")
    rows = [("Checkpoint · model fingerprint", _field(_field(identity, "checkpoint"), "sha256")),
            ("Knowledge graph (kg.json)", _field(_field(identity, "graph"), "kg_sha256")),
            ("Served workspace manifest (split_manifest.json) · its dataset fingerprint",
             _field(_field(identity, "graph"), "split_manifest_sha256")),
            (f"Dataset record (kg.provenance.json) · origin {_field(record, 'origin')}",
             _field(record, "sha256"))]
    items = "".join(
        f'<div class="shep-digest"><span class="shep-dl">{_e(label)}</span>'
        f'<code class="shep-full">{_e(value)}</code>{copy_button(value, "the full " + label)}</div>'
        for label, value in rows if _is_sha256(value))
    if not items:
        return ""
    return ('<div class="shep-dx-digests"><p class="shep-dx-digests-h"><b>This result in full</b> '
            f'<span class="shep-muted">digests to copy (sha256)</span></p>{items}</div>')


def model_identity_html(state: Mapping[str, Any], section: str = "identity") -> str:
    """The identity drawer's "Model in use": Model Facts of the served model
    (`diagnosis_card.served_facts_html`), then the claim boundary for what is
    loaded and the medical-device line."""
    status = state.get("status")
    boundary = ""
    if isinstance(status, Mapping) and status.get("initialized"):
        identity = status.get("served_identity")
        kind = _SERVED if isinstance(identity, dict) else IDENTITY_NOT_REPORTED
        sentences = _claim_boundary_lines(kind, identity if isinstance(identity, dict) else None,
                                          IDENTITY_NOTHING_CHECKED_LOADED)
        boundary = "".join(f"<p>{_e(s)}</p>" for s in sentences)
    return (card.served_facts_html(state, open_section=section)
            + f'<div class="shep-dx-boundary">{boundary}<p><b>{_e(NOT_A_MEDICAL_DEVICE)}</b></p>'
              '</div>')


def identity_outputs(view: Mapping[str, Any], state: Mapping[str, Any],
                     result: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The identity drawer as `view` asks: open or closed, on "Model in use"
    (opened at a section) or "This result"."""
    if not view.get("open"):
        return {"identity_drawer": gr.update(visible=False)}
    on_model = view.get("seg", "model") == "model"
    # Each view shown or hidden by a class (`view_update`); a hidden one is
    # not rendered.
    return {
        "identity_drawer": gr.update(visible=True),
        "identity_seg": gr.update(value="model" if on_model else "result"),
        "id_model": view_update(on_model, *_HTML_VIEW,
                                value=model_identity_html(state, view.get("section") or "identity")
                                if on_model else None),
        "id_result_md": view_update(not on_model, *_MD_VIEW,
                                    value=None if on_model else result_identity_md(result, state)),
        "id_result_html": view_update(not on_model, *_HTML_VIEW,
                                      value=None if on_model else result_digests_html(result)),
    }


# =============================================================================
# The card's outputs, the Run controls
# =============================================================================
def run_controls(state: Mapping[str, Any], phase: Any) -> Dict[str, Any]:
    """Run Diagnosis: off while a model change is open, or when no model is
    loaded; greyed and not clickable, with the reason as visible text under it
    and a bubble on hover and on keyboard focus (a transparent ⓘ over the
    button: the shared help control)."""
    if phase == "loading":
        key, why = "dx-run-loading", RUN_DISABLED_LOADING
    elif phase not in (None, "idle"):
        key, why = "dx-run-change", RUN_DISABLED_CHANGE
    elif card.run_blocked(state):
        key = "dx-run-no-model"
        why = RUN_DISABLED_NO_MODEL if state.get("operator") else RUN_DISABLED_NO_MODEL_READ_ONLY
    else:
        return {"run_btn": gr.update(interactive=True), "run_cover": "", "run_reason": ""}
    title = TOPIC_HELP[key][0]
    cover = (f'<button type="button" class="shep-i shep-i-cover" data-help="{key}" '
             f'data-help-title="{_e(title)}" aria-label="Run Diagnosis is disabled: why?" '
             f'aria-expanded="false" aria-describedby="{text_id(key)}"></button>')
    return {"run_btn": gr.update(interactive=False), "run_cover": cover,
            "run_reason": f'<p class="shep-dx-why"><span aria-hidden="true">⊖</span> {_e(why)}</p>'}


def loading_line_html(plan: Any) -> str:
    """The card's line while this page's Load reloads the pipeline (the
    mockup's state 3): "Loading <12 hex> for every session… 00:00:00", the
    count kept by the page's own script (`model_list.start_wait_js`, which
    also swaps in the second sentence when the page stops waiting after
    `PAGE_WAIT_SECONDS`, as the drawer says). "" without a plan."""
    from src.webui.components.model_list import PAGE_WAIT_SECONDS

    if not isinstance(plan, Mapping) or not isinstance(plan.get("to"), Mapping):
        return ""
    new = _e(_short_hex(plan["to"].get("sha256")))
    return ('<p class="shep-dx-loading" role="status" data-shep-load-card>'
            '<span class="shep-load-on"><span class="shep-spinner" aria-hidden="true"></span> '
            f'Loading <code>{new}</code> for every session…\u00a0'
            '<span class="shep-elapsed" data-shep-load-elapsed>00:00:00</span></span>'
            f'<span class="shep-load-off" hidden>The page stopped waiting for <code>{new}</code> '
            f'after {PAGE_WAIT_SECONDS} s; the server may still finish the reload.</span></p>')


def _run_tuple(state: Mapping[str, Any], phase: Any) -> Tuple[Any, Any, Any]:
    run = run_controls(state, phase)
    return run["run_btn"], run["run_cover"], run["run_reason"]


def card_loading_outputs(plan: Any, state: Any) -> Tuple[Any, ...]:
    """The card's part of the step "Reload with …" runs before the reload (the
    shared list's ``on_loading``): the "Loading …" line and Run Diagnosis off
    with its reason, shown before the reload holds the server. Nothing when
    there was no plan (nothing is loading)."""
    if not plan:
        return gr.update(), gr.update(), gr.update(), gr.update()
    state = state if isinstance(state, Mapping) else card.new_state()
    return (loading_line_html(plan), *_run_tuple(state, "loading"))


def card_loaded_outputs(state: Mapping[str, Any], phase: Any) -> Tuple[Any, ...]:
    """After the reload's answer: no "Loading …" line, and Run Diagnosis as
    the phase now says (idle, or a confirmation shown again)."""
    return ("", *_run_tuple(state, phase))


def card_outputs(state: Mapping[str, Any], result: Optional[Mapping[str, Any]], phase: Any,
                 view: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Every output that follows the card's state, sending only what changed
    since this session's last send (kept in the state's ``rendered``): a
    focused control is never redrawn by the timer for nothing."""
    produced = _produced_at(result)
    button = card.change_button(state)
    run = run_controls(state, phase)
    view = view or {}
    fresh: Dict[str, Any] = {
        "card_time": card.time_html(state, produced),
        "card_body": card.body_html(state, produced),
        "change_btn": button,
        "change_reason": card.change_reason_html(state),
        "details_btn": card.details_visible(state),
        "run_btn": run["run_btn"].get("interactive"),
        "run_cover": run["run_cover"],
        "run_reason": run["run_reason"],
        "badge": badge_html(result, state),
    }
    if not isinstance(result, Mapping):
        fresh["results_body"] = empty_results_html(card.run_blocked(state))
    if view.get("open"):
        for name, update in identity_outputs(view, state, result).items():
            fresh[name] = update
    previous = state.get("rendered") if isinstance(state.get("rendered"), Mapping) else {}
    out: Dict[str, Any] = {}
    for name, value in fresh.items():
        if previous.get(name) == value:
            continue
        if name == "change_btn":
            out[name] = gr.update(**value)
        elif name == "details_btn":
            out[name] = gr.update(visible=value)
        elif name == "run_btn":
            out[name] = gr.update(interactive=value)
        else:
            out[name] = value
    out["card_state"] = {**state, "rendered": {**previous, **fresh}}
    return out


def _operator(request: Any) -> Tuple[bool, Optional[str]]:
    access = operator_access(request)
    return access.allowed, (None if access.allowed
                            else D8_SHORT.get(access.condition, D8_SHORT["unreadable_request"]))


def read_card(state: Any, request: Any, *, open_checkpoints: bool, reread: bool = False,
              now: Optional[datetime] = None) -> Dict[str, Any]:
    """One status read (5 s limit) for this session's card, and its new state."""
    status, problem = card.read_status_within()
    allowed, short = _operator(request)
    return card.advance(state if isinstance(state, Mapping) else card.new_state(), status,
                        problem, now=now or datetime.now(), operator=allowed, d8_short=short,
                        open_checkpoints=open_checkpoints, reread=reread)


def refresh_card(state: Any, result: Any, phase: Any, view: Any, request: Any, *,
                 reread: bool = False) -> Dict[str, Any]:
    """Page load, Diagnosis shown, ↻, a Load: read and show (queued: the
    trained-on dataset may be read from the served checkpoint)."""
    return card_outputs(read_card(state, request, open_checkpoints=True, reread=reread),
                        result, phase, view)


def poll_card(location: Any, state: Any, result: Any, phase: Any, view: Any,
              request: Any) -> Dict[str, Any]:
    """The shell's slow timer, while Diagnosis is shown: one status read, no
    checkpoint opened (`queue=False`); nothing while another page is shown."""
    if location != CLINICAL:
        return {}
    return card_outputs(read_card(state, request, open_checkpoints=False), result, phase, view)


# =============================================================================
# Event handlers
# =============================================================================
def _clamped_top_k(top_k: Any) -> int:
    try:
        value = int(top_k)
    except (TypeError, ValueError):
        value = 10
    return max(1, min(20, value))


def _no_result_outputs(state: Mapping[str, Any], notice: Optional[str]) -> Dict[str, Any]:
    return {
        "results_hd": results_header_html(None), "badge": "",
        "results_body": empty_results_html(card.run_blocked(state), notice),
        "results_state": None, "selected": None,
        "export_csv": _EXPORTS_HIDDEN, "export_report": _EXPORTS_HIDDEN,
        **details_outputs(None, None),
    }


def diagnose(phenotype_text: str, patient_id: str, top_k: Any, explanations: bool,
             state: Any, phase: Any, view: Any, request: Any) -> Dict[str, Any]:
    """Run Diagnosis: the shared API (`_call_diagnose`), then the status read
    again for the badge, so a result never meets an older read. The full
    result is kept in this session's state; the export files are written up
    front, so one click downloads them."""
    state = state if isinstance(state, Mapping) else card.new_state()
    view = view if isinstance(view, Mapping) else {}
    hpo_ids = _parse_hpo_ids(phenotype_text)
    if not hpo_ids:
        out = _no_result_outputs(state, (
            '<p class="shep-dx-warn"><b>No valid HPO terms found.</b> Enter HPO ids like '
            '<code>HP:0001250</code> — one per line, or comma/space-separated; names are '
            'optional.</p>'))
        return out
    explanations = bool(explanations)
    result = _call_diagnose(phenotypes=hpo_ids, patient_id=patient_id, top_k=_clamped_top_k(top_k),
                            include_explanations=explanations)
    if "error" in result:
        error = _e(result["error"]).replace("\n\n", "<br><br>")
        return _no_result_outputs(state, f'<p class="shep-dx-error"><b>Error:</b> {error}</p>')
    result["_query_phenotypes"] = hpo_ids
    result["_explanations"] = explanations
    result["_received_at"] = card.stamp(datetime.now())
    # A response that does not name its kind shows no candidate: nothing is
    # selected and nothing is exported (the body says why).
    candidates = (result.get("candidates") or []) if _result_kind(result) is not None else []
    first = candidates[0].get("rank") if candidates else None
    export_note = ""
    if candidates:
        try:
            csv_path, md_path = _write_exports(result)
            csv_update = gr.update(value=csv_path, interactive=True, visible=True)
            md_update = gr.update(value=md_path, interactive=True, visible=True)
        except Exception as exc:  # noqa: BLE001 — never let export break the result
            logger.warning("Export file generation failed: %s", exc)
            csv_update = md_update = _DOWNLOAD_DISABLED
            export_note = ('<p class="shep-dx-warn">Export files could not be written '
                           '(disk/permissions?). Results are shown; Run Diagnosis again to retry '
                           'the export.</p>')
    else:
        csv_update = md_update = _DOWNLOAD_DISABLED
    new_state = read_card(state, request, open_checkpoints=True)
    out = card_outputs(new_state, result, phase, view)
    out.update({
        "results_hd": results_header_html(result),
        "badge": badge_html(result, new_state),
        "results_body": results_body_html(result, first) + export_note,
        "results_state": result, "selected": first,
        "export_csv": csv_update, "export_report": md_update,
        **details_outputs(result, first),
    })
    out["card_state"]["rendered"] = {**out["card_state"]["rendered"],
                                     "badge": out["badge"]}
    if view.get("open"):
        out.update(identity_outputs(view, new_state, result))
    return out


def phenotypes_changed(result: Any, state: Any) -> Dict[str, Any]:
    """Editing the phenotypes after a run clears the result and its exports,
    so nothing stale can be read or exported. Nothing happens before a
    result exists (typing the first query)."""
    if not result:
        return {}
    state = state if isinstance(state, Mapping) else card.new_state()
    out = _no_result_outputs(state, '<p class="shep-dx-notice-p">Inputs changed — Run Diagnosis '
                                    'to refresh the results.</p>')
    out["card_state"] = {**state, "rendered": {
        **(state.get("rendered") or {}), "badge": ""}}
    return out


def select_candidate(result: Any, segment: Any, key: Any) -> Dict[str, Any]:
    """A row of the table selected: its details, on the segment shown."""
    if not isinstance(result, Mapping):
        return {}
    try:
        rank = int(key)
    except (TypeError, ValueError):
        return {}
    if _candidate(result, rank) is None:
        return {}
    return {"selected": rank,
            "results_body": results_body_html(result, rank),
            **details_outputs(result, rank, segment if segment in _SEGMENT_VALUES else "evidence")}


def open_identity(state: Any, result: Any, seg: str = "model",
                  section: str = "identity", phase: Any = "idle") -> Dict[str, Any]:
    """Details › (Model in use, at Identity), Recorded tests › (at Evidence) or
    the badge (This result): the identity drawer, in the details column's
    slot; the change-model drawer closes. While a model change is open (its
    confirmation, or a reload running: `phase` not idle) the change drawer
    stays and the identity drawer does not open: the confirmation, and the
    reload's outcome, are never hidden behind it (Run Diagnosis says to finish
    or cancel the change first)."""
    if phase not in (None, "idle"):
        return {"change_drawer": gr.update(visible=True)}
    view = {"open": True, "seg": seg, "section": section}
    state = state if isinstance(state, Mapping) else card.new_state()
    return {"identity_view": view, "change_drawer": gr.update(visible=False),
            **identity_outputs(view, state, result)}


def close_identity() -> Dict[str, Any]:
    return {"identity_view": {"open": False}, "identity_drawer": gr.update(visible=False)}


def open_change(request: gr.Request) -> Dict[str, Any]:
    """"🔒 Change model…" / "🔒 Load a model…": D8 on this request first. A
    request D8 refuses gets the reason and no list (listing opens checkpoint
    files); otherwise the change-model drawer opens (its list is read next, by
    the shared list's own refresh, which asks D8 again)."""
    access = operator_access(request)
    if not access.allowed:
        return {"change_reason": f'<p class="shep-dx-why">{_e(access.refusal)}</p>',
                "change_drawer": gr.update(visible=False)}
    return {"change_drawer": gr.update(visible=True), "identity_drawer": gr.update(visible=False),
            "identity_view": {"open": False}}


#: The outputs of `_on_change_model`, in order.
CHANGE_OUTPUTS = ("change_reason", "change_drawer", "identity_drawer", "identity_view")


def _on_change_model(request: gr.Request) -> Tuple[Any, ...]:
    """The "Change model…" event: `open_change` on this event's own request
    (`operator_access(request)` first), as the outputs `CHANGE_OUTPUTS` name."""
    opened = open_change(request)
    return tuple(opened.get(name, gr.update()) for name in CHANGE_OUTPUTS)


#: While this page's reload runs, its Close, × and Cancel close the whole
#: drawer at once in the page (the card keeps the reload's "Loading …" line),
#: and the × is clicked for the server once the reload has answered; "Change
#: model…" meanwhile shows the drawer again, on the reload's status
#: (`model_list.load_guard`).
CLOSE_WHILE_LOADING = {
    "hide": "#shep-dx-change",
    "then": "#shep-dx-change-close button",
    "also": ["#shep-dx-change-close button", "#shep-dx-change-cancel"],
    "openers": ["#shep-dx-change-btn"],
}


def close_change(phase: Any) -> Dict[str, Any]:
    """Cancel or ×: the drawer closes and an open confirmation is dropped
    (nothing was reloaded: the reload runs only on its own button). A close
    while a reload runs is done in the page at once and reaches this only
    after the reload's answer (`CLOSE_WHILE_LOADING`); one that still finds
    `phase` "loading" (sent before the reload began, not from the page) leaves
    the drawer open: the reload cannot be stopped, and its outcome must be
    seen. The confirmation box inside the
    drawer is reset in a second step (`reset_change`): a response that hides a
    column and updates a component inside it leaves detached page elements in
    the browser (Gradio 6.30, measured), so this one changes no component in
    the drawer; ``change_closed`` tells the second step that it closed."""
    if phase == "loading":
        return {"change_drawer": gr.update(visible=True), "change_closed": False}
    return {"change_drawer": gr.update(visible=False), "plan": None, "phase": "idle",
            "change_closed": True}


def reset_change(closed: Any) -> Dict[str, Any]:
    """After `close_change` closed the drawer (`closed`), inside the hidden
    drawer: the confirmation box hidden and emptied, the list's reason
    cleared. Nothing otherwise."""
    if closed is not True:
        return {}
    return {"confirm": gr.update(visible=False), "confirm_html": "", "list_reason": "",
            "change_closed": False}


# =============================================================================
# Tab builder (called by the shell)
# =============================================================================
def _close_html(what: str) -> str:
    return (f'<button type="button" class="shep-dx-close" data-key="close" aria-label="Close the '
            f'{_e(what)}" title="Close">×</button>')


def _event_key(evt: Any) -> Optional[str]:
    raw = getattr(evt, "_data", None)
    key = raw.get("key") if isinstance(raw, Mapping) else None
    return key if isinstance(key, str) else None


def create_diagnosis_tab(ctx: Any) -> None:
    """Build Diagnosis (clinical), the first screen (the page contract:
    `src/webui/shell.py`). The card is read on page load, each time the page
    is shown, on the shell's slow timer while it is shown, after a diagnosis
    and after a model change; its help and the fields' go behind the shared
    ⓘ."""
    from src.webui.components.model_list import MODEL_LIST_JS
    from src.webui.shell import plain_html

    ctx.help.fields(FIELD_HELP)
    ctx.help.topics(TOPIC_HELP)

    results_state = gr.State(None)
    selected = gr.State(None)
    card_state = gr.State(card.new_state())
    identity_view = gr.State({"open": False})
    change_closed = gr.State(False)

    with gr.Row(elem_id="shep-dx", elem_classes=["shep-page", "shep-dx"], equal_height=False):
        # --- the input column ------------------------------------------------
        with gr.Column(elem_id="shep-dx-input", elem_classes=["shep-panel"], min_width=0):
            plain_html('<div class="shep-panel-hd"><h2>Patient phenotypes</h2></div>',
                       elem_id="shep-dx-input-hd")
            with gr.Column(elem_id="shep-dx-input-bd", elem_classes=["shep-dx-scroll"]):
                with gr.Column(elem_id="shep-dx-fields", elem_classes=["shep-dx-fields"]):
                    phenotype_input = gr.Textbox(
                        label="HPO terms", lines=3, max_lines=12, elem_id="dx_hpo_terms",
                        placeholder=("HP:0001250 — Seizure\n"
                                     "HP:0001263, HP:0002376   (commas / spaces are fine too)"))
                    with gr.Column(elem_id="shep-dx-side", elem_classes=["shep-dx-side"]):
                        patient_id_input = gr.Textbox(label="Patient ID (optional)", value="",
                                                      placeholder="(auto-generated if blank)",
                                                      max_lines=1, elem_id="dx_patient_id")
                        with gr.Row(elem_id="shep-dx-options", elem_classes=["shep-dx-options"]):
                            plain_html('<span class="shep-dx-opt">Options</span>',
                                       elem_id="shep-dx-options-label")
                            top_k_input = gr.Number(label="Top-K", value=10, minimum=1,
                                                    maximum=20, step=1, precision=0,
                                                    elem_id="dx_top_k", min_width=0, scale=0)
                            explanations_input = gr.Checkbox(label="Explanations", value=True,
                                                             elem_id="dx_explanations", scale=0,
                                                             min_width=0)
                with gr.Column(elem_id="shep-dx-sticky", elem_classes=["shep-dx-sticky"]):
                    with gr.Column(elem_id="shep-dx-card", elem_classes=["shep-dx-card"]):
                        with gr.Row(elem_id="shep-dx-card-hd", elem_classes=["shep-dx-card-hd"]):
                            plain_html('<span class="shep-dx-card-title">Model in use</span>'
                                       + help_button("dx-card", "Model in use"),
                                       elem_id="shep-dx-card-title")
                            # "checked HH:MM:SS ↻" stay together: on a narrow column
                            # they move under the title as one, at its right.
                            with gr.Row(elem_id="shep-dx-card-when",
                                        elem_classes=["shep-dx-card-when"]):
                                card_time = plain_html("", elem_id="shep-dx-card-time")
                                reread = plain_html(
                                    '<button type="button" class="shep-dx-reread" '
                                    'data-key="refresh" aria-label="Read the model in use again" '
                                    'title="Read again">↻</button>', elem_id="shep-dx-reread",
                                    js_on_load=MODEL_LIST_JS)
                        # "Loading <12 hex> for every session… 00:00:42" while this
                        # page's Load reloads the pipeline (`loading_line_html`).
                        card_loading = plain_html("", elem_id="shep-dx-card-loading")
                        card_body = plain_html(card.body_html(card.new_state()),
                                               elem_id="shep-dx-card-body",
                                               js_on_load=MODEL_LIST_JS)
                        with gr.Row(elem_id="shep-dx-card-actions",
                                    elem_classes=["shep-dx-card-actions"]):
                            details_btn = gr.Button("Details ›", size="sm", visible=False,
                                                    elem_id="shep-dx-details-btn", scale=0,
                                                    min_width=0)
                            change_btn = gr.Button("🔒 Change model…", size="sm",
                                                   interactive=False, scale=0, min_width=0,
                                                   elem_id="shep-dx-change-btn")
                        change_reason = plain_html("", elem_id="shep-dx-change-why")
                    with gr.Row(elem_id="shep-dx-run", elem_classes=["shep-dx-run"]):
                        with gr.Column(elem_id="shep-dx-run-wrap", min_width=0, scale=1):
                            run_btn = gr.Button("▶ Run Diagnosis", variant="primary", size="lg",
                                                elem_id="shep-dx-run-btn")
                            run_cover = plain_html("", elem_id="shep-dx-run-cover")
                        gr.ClearButton(components=[phenotype_input], value="Clear", size="lg",
                                       elem_id="shep-dx-clear", scale=0, min_width=0)
                    run_reason = plain_html("", elem_id="shep-dx-run-why")

        # --- the drawers (wide: over the columns; narrow: in the page's flow) -
        with gr.Column(visible=False, elem_id="shep-dx-identity",
                       elem_classes=["shep-panel", "shep-dx-drawer"], min_width=0) as identity_drawer:
            with gr.Row(elem_classes=["shep-dx-drawer-hd"], elem_id="shep-dx-identity-hd"):
                plain_html("<h2>Identity</h2>", elem_id="shep-dx-identity-title")
                identity_seg = gr.Radio(choices=list(IDENTITY_VIEWS), value="model",
                                        label="Identity view", show_label=False, container=False,
                                        elem_classes=["shep-seg"], elem_id="shep-dx-identity-seg")
                identity_close = plain_html(_close_html("identity drawer"),
                                            elem_id="shep-dx-identity-close",
                                            js_on_load=MODEL_LIST_JS)
            with gr.Column(elem_id="shep-dx-identity-bd", elem_classes=["shep-dx-scroll"]):
                # The views are shown and hidden by a class, never `visible`
                # (`view_update`).
                id_model = plain_html("", elem_id="shep-dx-id-model", js_on_load=MODEL_LIST_JS)
                id_result_md = gr.Markdown("", elem_id="shep-dx-id-result",
                                           elem_classes=[*_MD_VIEW, VIEW_OFF])
                id_result_html = plain_html("", elem_id="shep-dx-id-digests",
                                            elem_classes=[VIEW_OFF])
        with gr.Column(visible=False, elem_id="shep-dx-change",
                       elem_classes=["shep-panel", "shep-dx-drawer"], min_width=0) as change_drawer:
            with gr.Row(elem_classes=["shep-dx-drawer-hd"], elem_id="shep-dx-change-hd"):
                plain_html('<h2>Change model</h2><span class="shep-dx-drawer-note">🔒 operator '
                           'controls · one model is served to every session</span>',
                           elem_id="shep-dx-change-title")
                change_close = plain_html(_close_html("change-model drawer"),
                                          elem_id="shep-dx-change-close", js_on_load=MODEL_LIST_JS)
            with gr.Column(elem_id="shep-dx-change-bd"):
                links: Dict[str, Any] = {}

                def _head() -> Any:
                    # Each "›" keeps to the word before it (a no-break space):
                    # at 1440 x 900 the last one was left alone on a line.
                    links["compare"] = plain_html(
                        f'<button type="button" class="shep-dx-link" data-key="models">'
                        f'{COMPARE_LINK}</button>', elem_id="shep-dx-compare",
                        js_on_load=MODEL_LIST_JS)
                    return links["compare"]

                def _foot() -> Any:
                    links["cancel"] = gr.Button("Cancel", size="sm", scale=0, min_width=0,
                                                elem_id="shep-dx-change-cancel")
                    plain_html('<span class="shep-note">Load reloads the pipeline and asks once, '
                               'from → to. Replaces what every session is served; results already '
                               'produced keep their own identity.</span>',
                               elem_id="shep-dx-change-note")
                    return links["cancel"]

                handle = ctx.create_model_list(
                    "clinical", head=_head, foot=_foot,
                    on_loading={"fn": card_loading_outputs, "inputs": [card_state],
                                "outputs": [card_loading, run_btn, run_cover, run_reason]},
                    close_while_loading=CLOSE_WHILE_LOADING)

        # --- the candidates ---------------------------------------------------
        with gr.Column(elem_id="shep-dx-results", elem_classes=["shep-panel"], min_width=0):
            results_hd = plain_html(results_header_html(None), elem_id="shep-dx-results-hd")
            with gr.Column(elem_id="shep-dx-results-bd", elem_classes=["shep-dx-scroll"]):
                badge = plain_html("", elem_id="shep-dx-badge", js_on_load=MODEL_LIST_JS)
                results_body = plain_html(empty_results_html(), elem_id="shep-dx-results-body",
                                          js_on_load=MODEL_LIST_JS)
            with gr.Row(elem_id="shep-dx-results-ft", elem_classes=["shep-panel-ft"]):
                # Not shown until a result exists (`_EXPORTS_HIDDEN`).
                export_csv = gr.DownloadButton("⬇ Export CSV", size="sm", interactive=False,
                                               scale=0, min_width=0, elem_id="shep-dx-export-csv",
                                               visible="hidden")
                export_report = gr.DownloadButton("⬇ Export report (.md)", size="sm",
                                                  interactive=False, scale=0, min_width=0,
                                                  elem_id="shep-dx-export-md", visible="hidden")
                plain_html(help_button("dx-exports", "Exports"), elem_id="shep-dx-exports-help")
                plain_html(f'<p class="shep-dx-device">{_e(NOT_A_MEDICAL_DEVICE)}</p>',
                           elem_id="shep-dx-device")

        # --- the selected candidate's details ----------------------------------
        with gr.Column(elem_id="shep-dx-details", elem_classes=["shep-panel"], min_width=0):
            details_hd = plain_html(details_header_html(None), elem_id="shep-dx-details-hd")
            # The segmented control and the views are shown and hidden by a
            # class, never `visible` (`view_update`).
            details_seg = gr.Radio(choices=list(SEGMENTS), value="evidence",
                                   label="Candidate details view", show_label=False,
                                   container=False, elem_classes=["shep-seg", VIEW_OFF],
                                   elem_id="shep-dx-seg")
            with gr.Column(elem_id="shep-dx-details-bd", elem_classes=["shep-dx-scroll"]):
                details_html = plain_html(details_empty_html(), elem_id="shep-dx-details-body",
                                          js_on_load=MODEL_LIST_JS)
                details_md = gr.Markdown("", elem_id="shep-dx-explanation",
                                         elem_classes=[*_MD_VIEW, VIEW_OFF])
            details_ft = plain_html("", elem_id="shep-dx-details-ft")

    named = {
        "card_state": card_state, "card_time": card_time, "card_body": card_body,
        "change_btn": change_btn, "change_reason": change_reason, "details_btn": details_btn,
        "run_btn": run_btn, "run_cover": run_cover, "run_reason": run_reason, "badge": badge,
        "results_hd": results_hd, "results_body": results_body, "results_state": results_state,
        "selected": selected, "export_csv": export_csv, "export_report": export_report,
        "details_hd": details_hd, "details_seg": details_seg, "details_html": details_html,
        "details_md": details_md, "details_ft": details_ft, "identity_view": identity_view,
        "identity_drawer": identity_drawer, "identity_seg": identity_seg, "id_model": id_model,
        "id_result_md": id_result_md, "id_result_html": id_result_html,
        "change_drawer": change_drawer, "plan": handle.plan, "confirm": handle.confirm,
        "confirm_html": handle.confirm_html, "list_reason": handle.reason, "phase": handle.phase,
        "change_closed": change_closed,
    }

    def outputs(*names: str) -> List[Any]:
        return [named[n] for n in names]

    def mapped(values: Dict[str, Any]) -> Any:
        # Gradio reads an empty dict as one value; "nothing to update" is skip().
        if not values:
            return gr.skip()
        return {named[n]: v for n, v in values.items()}

    card_names = ("card_state", "card_time", "card_body", "change_btn", "change_reason",
                  "details_btn", "run_btn", "run_cover", "run_reason", "badge", "results_body",
                  "identity_drawer", "identity_seg", "id_model", "id_result_md", "id_result_html")
    card_inputs = [card_state, results_state, handle.phase, identity_view]

    def _card_refresh(state, result, phase, view, request: gr.Request):
        return mapped(refresh_card(state, result, phase, view, request))

    def _card_reread(state, result, phase, view, request: gr.Request):
        return mapped(refresh_card(state, result, phase, view, request, reread=True))

    def _card_poll(location, state, result, phase, view, request: gr.Request):
        return mapped(poll_card(location, state, result, phase, view, request))

    refresh = {"fn": _card_refresh, "inputs": card_inputs, "outputs": outputs(*card_names),
               "show_progress": "hidden"}
    # Diagnosis is shown at page load with no tab select, and on each select.
    ctx.blocks.load(**refresh)
    ctx.on_page_open(CLINICAL, refresh)
    ctx.served_model_changed.change(**refresh)
    ctx.timer_slow.tick(_card_poll, inputs=[ctx.location, *card_inputs],
                        outputs=outputs(*card_names), queue=False, show_progress="hidden")
    reread.click(_card_reread, inputs=card_inputs, outputs=outputs(*card_names),
                 show_progress="hidden")

    def _phase_changed(state, phase):
        run = run_controls(state if isinstance(state, Mapping) else card.new_state(), phase)
        return mapped(run)

    handle.phase.change(_phase_changed, inputs=[card_state, handle.phase],
                        outputs=outputs("run_btn", "run_cover", "run_reason"),
                        show_progress="hidden")

    # The reload's answer (reloaded or not): the card's "Loading …" line goes
    # and Run Diagnosis follows the phase again, whatever the answer was (a
    # refused reload changes nothing served, so no card refresh would run).
    def _reload_answered(state, phase):
        return card_loaded_outputs(state if isinstance(state, Mapping) else card.new_state(),
                                   phase)

    handle.reload_done.then(_reload_answered, inputs=[card_state, handle.phase],
                            outputs=[card_loading, run_btn, run_cover, run_reason],
                            show_progress="hidden")

    # --- the card's own links: dismiss, Retry / read it, Recorded tests ›
    identity_names = ("identity_view", "identity_drawer", "change_drawer", "identity_seg",
                      "id_model", "id_result_md", "id_result_html")

    def _card_link(state, result, phase, view, evt: gr.EventData, request: gr.Request):
        key = _event_key(evt)
        if key == "dismiss":
            return mapped(card_outputs(card.dismiss_change(state), result, phase, view))
        if key == "refresh":
            return mapped(refresh_card(state, result, phase, view, request, reread=True))
        if key == "evidence":
            return mapped(open_identity(state, result, "model", "evidence", phase))
        return gr.skip()

    card_body.click(_card_link, inputs=card_inputs,
                    outputs=outputs(*dict.fromkeys(card_names + identity_names)),
                    show_progress="hidden")

    def _open_details(state, result, phase):
        return mapped(open_identity(state, result, "model", "identity", phase))

    details_btn.click(_open_details, inputs=[card_state, results_state, handle.phase],
                      outputs=outputs(*identity_names), show_progress="hidden")

    def _open_result(state, result, phase, evt: gr.EventData):
        if _event_key(evt) != "result":
            return gr.skip()
        return mapped(open_identity(state, result, "result", phase=phase))

    badge.click(_open_result, inputs=[card_state, results_state, handle.phase],
                outputs=outputs(*identity_names), show_progress="hidden")

    def _switch_identity(state, result, seg, view):
        new_view = {**(view or {}), "open": True, "seg": seg, "section": "identity"}
        return mapped({"identity_view": new_view, **identity_outputs(new_view, state, result)})

    identity_seg.input(_switch_identity,
                       inputs=[card_state, results_state, identity_seg, identity_view],
                       outputs=outputs(*identity_names), show_progress="hidden")

    def _close_identity_drawer(evt: gr.EventData):
        return mapped(close_identity()) if _event_key(evt) == "close" else gr.skip()

    identity_close.click(_close_identity_drawer, inputs=None,
                         outputs=outputs("identity_view", "identity_drawer"),
                         show_progress="hidden")

    def _to_models(evt: gr.EventData):
        return ctx.go("models") if _event_key(evt) == "models" else ctx.no_nav()

    id_model.click(_to_models, inputs=None, outputs=ctx.nav_outputs, show_progress="hidden")
    links["compare"].click(_to_models, inputs=None, outputs=ctx.nav_outputs,
                           show_progress="hidden")

    # --- Change model…: D8 first, then the shared list's own refresh, then the chips
    change_btn.click(_on_change_model, inputs=None, outputs=outputs(*CHANGE_OUTPUTS),
                     show_progress="hidden").then(**handle.refresh).then(**ctx.chip_refresh)

    change_names = ("change_drawer", "plan", "phase", "change_closed")
    reset_names = ("confirm", "confirm_html", "list_reason", "change_closed")

    def _reset_change(closed):
        return mapped(reset_change(closed))

    reset = {"fn": _reset_change, "inputs": [change_closed], "outputs": outputs(*reset_names),
             "show_progress": "hidden"}

    def _close_change_drawer(phase, evt: gr.EventData):
        return mapped(close_change(phase)) if _event_key(evt) == "close" else gr.skip()

    # Each close ends with the shared list's page step: a close done in the
    # page while a reload ran (`CLOSE_WHILE_LOADING`) is now done here too.
    change_close.click(_close_change_drawer, inputs=[handle.phase],
                       outputs=outputs(*change_names), show_progress="hidden").then(
        **reset).then(None, js=handle.after_close_js)

    def _cancel_change(phase):
        return mapped(close_change(phase))

    links["cancel"].click(_cancel_change, inputs=[handle.phase], outputs=outputs(*change_names),
                          show_progress="hidden").then(**reset).then(None, js=handle.after_close_js)

    # --- Run Diagnosis, the phenotypes, the table, the segments
    result_names = ("results_hd", "badge", "results_body", "results_state", "selected",
                    "export_csv", "export_report", "details_hd", "details_seg", "details_html",
                    "details_md", "details_ft")

    def _run_diagnosis(text, patient, top_k, explanations, state, phase, view,
                       request: gr.Request):
        return mapped(diagnose(text, patient, top_k, explanations, state, phase, view, request))

    run_btn.click(_run_diagnosis,
                  inputs=[phenotype_input, patient_id_input, top_k_input, explanations_input,
                          card_state, handle.phase, identity_view],
                  outputs=outputs(*dict.fromkeys(result_names + card_names + identity_names)))

    def _phenotypes_edited(result, state):
        return mapped(phenotypes_changed(result, state))

    phenotype_input.change(_phenotypes_edited, inputs=[results_state, card_state],
                           outputs=outputs(*dict.fromkeys(result_names + ("card_state",))),
                           show_progress="hidden")

    def _select_candidate(result, segment, evt: gr.EventData):
        return mapped(select_candidate(result, segment, _event_key(evt)))

    details_names = ("details_hd", "details_seg", "details_html", "details_md", "details_ft")
    results_body.click(_select_candidate, inputs=[results_state, details_seg],
                       outputs=outputs("selected", "results_body", *details_names),
                       show_progress="hidden")

    def _switch_segment(result, rank, segment):
        return mapped(details_outputs(result, rank, segment))

    details_seg.input(_switch_segment, inputs=[results_state, selected, details_seg],
                      outputs=outputs(*details_names), show_progress="hidden")

    def _details_link(result, rank, evt: gr.EventData):
        key = _event_key(evt) or ""
        if not key.startswith("seg:"):
            return gr.skip()
        return mapped(details_outputs(result, rank, key[len("seg:"):]))

    details_html.click(_details_link, inputs=[results_state, selected],
                       outputs=outputs(*details_names), show_progress="hidden")


# =============================================================================
# The page's style sheet (the page contract: listed in `src/webui/app.py`)
# =============================================================================
#: The page's tab (`src.webui.shell.tab_elem_id(CLINICAL)`).
_TAB = "#shep-tab-clinical"

#: Diagnosis' layout: the input column 370 px wide from 1800 px of window, 320
#: px from 1600, 300 px below; the candidates and the details share the rest
#: 5:4. Wide, each panel scrolls inside and the input column's card and Run
#: Diagnosis stick to the bottom of its scroll body; narrow, the panels stack.
#: While the Load confirmation is open (UI-A's confirmation box is mounted only
#: then) the list's footer is hidden and its rows give up their height to it
#: (all of it below 820 px of height), so the from → to rows and the warning
#: sentence are never scrolled out of view.
PAGE_CSS = f"""
{_TAB} {{ --dx-in-w: 300px; }}
@media (min-width: 1600px) {{ {_TAB} {{ --dx-in-w: 320px; }} }}
@media (min-width: 1800px) {{ {_TAB} {{ --dx-in-w: 370px; }} }}
{_TAB} #shep-dx {{ position: relative; gap: 16px; align-items: stretch; }}
{_TAB} #shep-dx-input {{ flex: 0 0 var(--dx-in-w) !important; width: var(--dx-in-w); max-width: var(--dx-in-w); }}
{_TAB} #shep-dx-results {{ flex: 5 1 0 !important; min-width: 0 !important; }}
{_TAB} #shep-dx-details {{ flex: 4 1 0 !important; min-width: 0 !important; }}
{_TAB} .shep-dx-scroll {{ gap: 12px !important; padding: 14px 16px !important; }}
{_TAB} .shep-dx-scroll > * {{ flex-shrink: 0 !important; }}
{_TAB} #shep-dx-input-bd {{ padding-top: 12px !important; padding-bottom: 0 !important; gap: 0 !important; }}
{_TAB} .shep-dx-fields, {_TAB} .shep-dx-side {{ gap: 8px !important; }}
{_TAB} #shep-dx-fields {{ padding-bottom: 8px; }}
{_TAB} #shep-dx-fields > .form {{ background: none !important; border: 0 !important; box-shadow: none !important; }}
{_TAB} #shep-dx-fields .block.padded {{ padding: 6px 8px !important; }}
{_TAB} #shep-dx-fields [data-testid="block-info"] {{ margin-bottom: 4px !important; }}
{_TAB} #shep-dx-options .form {{ display: contents !important; }}
{_TAB} #shep-dx-options .block.padded {{ padding: 0 !important; }}
{_TAB} #dx_top_k input[type=number] {{ height: 34px; padding: 4px 8px; }}
{_TAB} .shep-dx-options {{ align-items: center !important; flex-wrap: wrap !important; gap: 8px 14px !important;
  border: 1px solid var(--shep-border); border-radius: 8px; background: var(--shep-panel-2); padding: 8px 10px !important; }}
{_TAB} .shep-dx-options > * {{ flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; }}
{_TAB} .shep-dx-options .form {{ display: flex !important; flex-direction: row !important; flex-wrap: wrap; align-items: center;
  gap: 8px 14px; background: none !important; border: 0 !important; box-shadow: none !important; }}
{_TAB} .shep-dx-options .block {{ padding: 0 !important; background: none !important; border: 0 !important; box-shadow: none !important;
  width: auto !important; flex: 0 0 auto !important; overflow: visible !important; }}
{_TAB} .shep-dx-opt {{ font-size: 14px; color: var(--shep-text-2); font-weight: 500; }}
{_TAB} #dx_top_k label.container {{ display: flex; align-items: center; gap: 6px; }}
{_TAB} #dx_top_k [data-testid="block-info"] {{ margin: 0; background: none; padding: 0; font-size: 15px; color: var(--shep-text); }}
{_TAB} #dx_top_k input[type=number] {{ width: 72px; }}
{_TAB} #shep-dx-input-bd > *, {_TAB} #shep-dx-sticky > *, {_TAB} #shep-dx-card > * {{ flex: none !important; }}
{_TAB} .shep-dx-sticky {{ position: sticky; bottom: 0; z-index: 50; gap: 0 !important; margin: 0 -16px !important;
  width: calc(100% + 32px) !important; max-width: none !important; background: var(--shep-panel); }}
{_TAB} .shep-dx-card {{ background: var(--shep-panel-2); border-top: 1px solid var(--shep-border);
  border-bottom: 1px solid var(--shep-border); padding: 5px 16px 6px !important; gap: 3px !important; }}
{_TAB} .shep-dx-card-hd {{ align-items: center !important; flex-wrap: wrap !important; gap: 2px 6px !important; }}
{_TAB} .shep-dx-card-hd > *, {_TAB} .shep-dx-card-when > * {{ flex: 0 0 auto !important; width: auto !important;
  min-width: 0 !important; }}
{_TAB} .shep-dx-card-when {{ margin-left: auto !important; flex-wrap: nowrap !important; gap: 4px !important;
  align-items: center !important; }}
{_TAB} .shep-dx-card-title {{ white-space: nowrap; }}
{_TAB} .shep-dx-card-title {{ font-size: 14px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--shep-text-2); }}
{_TAB} .shep-dx-checked {{ font-size: 14px; color: var(--shep-text-2); white-space: nowrap; }}
{_TAB} button.shep-dx-reread {{ border: 0; background: none; color: var(--shep-accent-text); font-size: 16px; cursor: pointer; padding: 0 4px; line-height: 1; }}
{_TAB} .shep-dx-card-bd {{ display: flex; flex-direction: column; gap: 1px; font-size: 15px; line-height: 1.3; }}
{_TAB} .shep-dx-card-bd p {{ margin: 0; }}
{_TAB} .shep-dx-name {{ font-weight: 600; }}
{_TAB} .shep-dx-sub {{ font-weight: 400; color: var(--shep-text-2); }}
{_TAB} .shep-dx-line {{ overflow-wrap: anywhere; }}
{_TAB} .shep-dx-line .shep-fp {{ white-space: normal; }}
{_TAB} .shep-dx-tail {{ display: flex; flex-wrap: wrap; gap: 4px 8px; align-items: center; }}
{_TAB} .shep-dx-chips {{ display: inline-flex; flex-wrap: wrap; gap: 4px 6px; }}
{_TAB} .shep-dx-tests {{ white-space: normal; }}
{_TAB} .shep-dx-card-grey .shep-dx-name, {_TAB} .shep-dx-card-grey .shep-dx-line {{ color: var(--shep-grey-600); }}
{_TAB} .shep-dx-card-grey .shep-fpv {{ font-weight: 400; color: var(--shep-grey-600); }}
{_TAB} .shep-tag-dashed {{ border-style: dashed !important; }}
{_TAB} .shep-dx-state {{ display: flex; flex-wrap: wrap; align-items: center; gap: 4px; font-size: 14px; color: var(--shep-grey-800); margin-bottom: 2px; }}
{_TAB} .shep-dx-state .shep-dx-retry {{ height: 24px; padding: 0 6px; }}
/* This page's Load in progress (the mockup's state 3): one accent line over the card's model. */
{_TAB} #shep-dx-card-loading:not(:has(.shep-dx-loading)) {{ display: none !important; }}
{_TAB} .shep-dx-loading {{ margin: 0; font-size: 15px; line-height: 1.35; color: var(--shep-accent-text); overflow-wrap: anywhere; }}
{_TAB} .shep-dx-loading .shep-load-off {{ color: var(--shep-grey-800); }}
{_TAB} .shep-dx-change {{ display: flex; align-items: flex-start; gap: 8px; margin-bottom: 2px; padding: 4px 8px; }}
{_TAB} button.shep-dx-x {{ margin-left: auto; border: 0; background: none; color: var(--shep-amber-800); font-size: 18px; line-height: 1; cursor: pointer; padding: 0 2px; }}
{_TAB} button.shep-dx-link {{ border: 0; background: none; padding: 0; color: var(--shep-accent-text); font: inherit; cursor: pointer; }}
{_TAB} button.shep-dx-link:hover {{ text-decoration: underline; }}
{_TAB} .shep-dx-card-actions {{ justify-content: space-between !important; align-items: center !important; flex-wrap: wrap !important; gap: 8px !important; margin-top: 2px; }}
{_TAB} .shep-dx-card-actions > * {{ flex: 0 0 auto !important; width: auto !important; }}
{_TAB} #shep-dx-details-btn {{ border: 0 !important; background: none !important; box-shadow: none !important; color: var(--shep-accent-text) !important; padding: 0 2px !important; font-weight: 500; }}
{_TAB} #shep-dx-change-btn {{ margin-left: auto; }}
{_TAB} .shep-dx-why {{ margin: 2px 0 0; font-size: 14px; color: var(--shep-text-2); text-align: right; }}
{_TAB} #shep-dx-run-why .shep-dx-why {{ text-align: left; margin: 0 16px 8px; }}
{_TAB} .shep-dx-run {{ padding: 6px 16px !important; gap: 10px !important; flex-wrap: nowrap !important; align-items: stretch !important; }}
{_TAB} #shep-dx-run-wrap {{ position: relative; gap: 0 !important; flex: 1 1 auto !important; }}
{_TAB} #shep-dx-clear {{ flex: 0 0 auto !important; min-width: 84px !important; padding: 0 18px !important; }}
{_TAB} #shep-dx-run-cover {{ position: absolute !important; inset: 0; pointer-events: none; z-index: 3; }}
{_TAB} #shep-dx-run-cover .html-container, {_TAB} #shep-dx-run-cover .prose {{ height: 100%; }}
{_TAB} button.shep-i.shep-i-cover {{ position: absolute; inset: 0; width: 100%; height: 100%; margin: 0; border: 0;
  border-radius: 8px; background: transparent !important; cursor: not-allowed; pointer-events: auto; color: transparent; }}
{_TAB} button.shep-i.shep-i-cover:focus-visible {{ outline: 2px solid var(--shep-accent); outline-offset: 2px; }}
{_TAB} #shep-dx-results-hd .shep-panel-right b {{ color: var(--shep-text); }}
{_TAB} .shep-dx-badge-row {{ display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }}
{_TAB} button.shep-dx-badge {{ display: inline-flex; align-items: center; gap: 8px; min-height: 34px; padding: 4px 12px;
  border-radius: 8px; border: 1px solid; font: inherit; font-size: 15px; font-weight: 600; cursor: pointer; text-align: left; }}
{_TAB} .shep-dx-badge-ok {{ background: var(--shep-green-50); border-color: var(--shep-green-200) !important; color: var(--shep-green-800); }}
{_TAB} .shep-dx-badge-amber {{ background: var(--shep-amber-50); border-color: var(--shep-amber-300) !important; color: var(--shep-amber-800); }}
{_TAB} .shep-dx-badge-grey {{ background: var(--shep-grey-50); border-color: var(--shep-grey-200) !important; color: var(--shep-grey-800); }}
{_TAB} .shep-dx-u {{ text-decoration: underline; }}
{_TAB} .shep-dx-badge-why {{ margin: 6px 0 0; font-size: 14px; color: var(--shep-text-2); }}
{_TAB} .shep-dx-banner {{ display: flex; align-items: center; gap: 8px; padding: 7px 12px; border-radius: 6px; margin: 0 0 8px; font-size: 15px; }}
{_TAB} .shep-dx-banner button.shep-i {{ margin-left: auto; }}
{_TAB} .shep-dx-banner-amber {{ background: var(--shep-amber-50); border: 1px solid var(--shep-amber-100); border-left: 4px solid var(--shep-amber-600); color: var(--shep-amber-800); }}
{_TAB} .shep-dx-banner-red {{ background: var(--shep-red-50); border: 1px solid var(--shep-red-200); border-left: 4px solid var(--shep-red-700); color: var(--shep-red-700); }}
{_TAB} .shep-dx-warnings {{ font-size: 14px; color: var(--shep-amber-800); margin: 0 0 8px; overflow-wrap: anywhere; }}
{_TAB} .shep-dx-tbl-wrap {{ margin: 0; }}
{_TAB} .shep-dx-tbl td, {_TAB} .shep-dx-tbl th {{ white-space: nowrap; }}
{_TAB} .shep-dx-tbl td.shep-dx-dis {{ white-space: normal; width: 100%; min-width: 9em; }}
{_TAB} .shep-dx-tbl td.shep-dx-sp {{ white-space: normal; min-width: 5em; }}
{_TAB} .shep-dx-kind {{ margin: 0 0 8px; font-size: 14px; color: var(--shep-text-2); }}
{_TAB} .shep-dx-tbl tbody tr[data-key] {{ cursor: pointer; }}
{_TAB} .shep-dx-tbl tbody tr.shep-on td.shep-dx-dis {{ color: var(--shep-accent-text); font-weight: 600; }}
{_TAB} .shep-dx-clamp {{ display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }}
{_TAB} .shep-dx-dot {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 7px; vertical-align: 1px; }}
{_TAB} .shep-dx-dot-strong {{ background: #1f9d55; }}
{_TAB} .shep-dx-dot-weak {{ background: #d69e00; }}
{_TAB} .shep-dx-dot-analogy {{ background: #3b6fe0; }}
{_TAB} .shep-dx-dot-none {{ background: #8a909b; }}
{_TAB} .shep-dx-empty {{ margin: auto; max-width: 440px; text-align: center; color: var(--shep-text-2); padding: 24px 8px; }}
{_TAB} .shep-dx-empty p {{ margin: 4px 0; }}
{_TAB} .shep-dx-empty-h {{ font-size: 16px; font-weight: 600; color: var(--shep-text); }}
{_TAB} .shep-dx-notice p {{ margin: 0 0 6px; }}
{_TAB} .shep-dx-error {{ color: var(--shep-red-700); overflow-wrap: anywhere; }}
{_TAB} .shep-dx-warn {{ color: var(--shep-amber-800); }}
{_TAB} #shep-dx-results-ft {{ flex-wrap: wrap !important; gap: 8px 10px !important; }}
{_TAB} #shep-dx-results-ft > * {{ flex: 0 0 auto !important; width: auto !important; }}
{_TAB} #shep-dx-device {{ flex-basis: 100% !important; }}
/* No result, no exports: their ⓘ goes with them (Gradio's "hidden" class). */
{_TAB} #shep-dx-results-ft:has(#shep-dx-export-csv.hidden) #shep-dx-exports-help {{ display: none !important; }}
{_TAB} .shep-dx-device {{ margin: 0; font-size: 14px; color: var(--shep-text-2); }}
{_TAB} .shep-dx-det-hd {{ flex-direction: column; align-items: flex-start !important; gap: 2px !important; }}
{_TAB} .shep-dx-det-hd h2 {{ display: flex; align-items: center; gap: 8px; overflow-wrap: anywhere; }}
{_TAB} .shep-dx-rank {{ font-size: 14px; font-weight: 600; color: var(--shep-accent-text); background: var(--shep-accent-50); border: 1px solid var(--shep-accent-100); border-radius: 6px; padding: 1px 7px; }}
{_TAB} .shep-dx-det-sub {{ margin: 0; font-size: 14px; color: var(--shep-text-2); }}
{_TAB} #shep-dx-seg {{ padding: 10px 16px 0 !important; }}
{_TAB} .shep-dx-off {{ display: none !important; }}
{_TAB} .shep-dx-ev-label {{ margin: 0 0 8px; }}
{_TAB} .shep-dx-quote {{ margin: 0 0 12px; padding: 8px 12px; border-left: 3px solid var(--shep-border-2); background: var(--shep-panel-2); color: var(--shep-text-2); overflow-wrap: anywhere; }}
{_TAB} .shep-dx-h {{ font-size: 15px; font-weight: 600; margin: 10px 0 4px; }}
{_TAB} .shep-dx-genes {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 0; }}
{_TAB} .shep-dx-gene {{ font-family: var(--shep-mono); font-size: 14px; border: 1px solid var(--shep-accent-100); background: var(--shep-accent-50); border-radius: 6px; padding: 2px 8px; }}
{_TAB} .shep-dx-list, {_TAB} .shep-dx-paths {{ margin: 0 0 8px; padding-left: 22px; }}
{_TAB} .shep-dx-paths li {{ margin: 0 0 6px; overflow-wrap: anywhere; }}
{_TAB} .shep-dx-arrow {{ color: var(--shep-muted); }}
{_TAB} #shep-dx-details-ft {{ border-top: 1px solid var(--shep-border); }}
{_TAB} .shep-dx-det-ft {{ display: flex; align-items: center; gap: 4px; padding: 10px 16px; font-size: 14px; color: var(--shep-text-2); }}
{_TAB} #shep-dx-explanation {{ font-size: 15px; }}
{_TAB} .shep-dx-drawer {{ box-shadow: 0 10px 30px rgba(20, 24, 40, .14); }}
{_TAB} .shep-dx-drawer-hd {{ align-items: center !important; flex-wrap: wrap !important; gap: 8px 12px !important; padding: 10px 16px !important; border-bottom: 1px solid var(--shep-border); }}
{_TAB} .shep-dx-drawer-hd > * {{ flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; }}
{_TAB} .shep-dx-drawer-hd h2 {{ font-size: 16px; font-weight: 600; margin: 0; display: inline; }}
{_TAB} .shep-dx-drawer-note {{ margin-left: 10px; font-size: 14px; color: var(--shep-text-2); }}
{_TAB} #shep-dx-identity-close, {_TAB} #shep-dx-change-close {{ margin-left: auto; }}
{_TAB} button.shep-dx-close {{ border: 0; background: none; font-size: 22px; line-height: 1; cursor: pointer; color: var(--shep-text-2); padding: 0 4px; }}
{_TAB} .shep-dx-boundary {{ margin-top: 12px; padding-top: 10px; border-top: 1px solid var(--shep-border); font-size: 14px; color: var(--shep-text-2); }}
{_TAB} .shep-dx-boundary p {{ margin: 0 0 6px; }}
{_TAB} .shep-dx-digests {{ display: flex; flex-direction: column; gap: 8px; }}
{_TAB} .shep-dx-digests-h {{ margin: 4px 0 0; }}
{_TAB} #shep-dx-id-result table {{ font-size: 14px; table-layout: fixed; width: 100%; }}
{_TAB} #shep-dx-id-result td, {_TAB} #shep-dx-id-result th {{ overflow-wrap: anywhere; vertical-align: top; }}
/* A code chip is text in its line, breaking only where it must (M5-3: Gradio
   draws a Markdown chip as an inline flex box, a box a line may break before,
   which left "(" alone: "Knowledge graph (" / "kg.json)"). */
{_TAB} #shep-dx-id-result code {{ display: inline; white-space: normal; word-break: normal;
  overflow-wrap: anywhere; -webkit-box-decoration-break: clone; box-decoration-break: clone; }}
{_TAB} #shep-dx-id-result p {{ font-size: 15px; }}
/* The labels of the drawers' Model Facts are short (a digest's long label takes a
   line of its own, `renderers.kv_cells`), so the values get the width. */
{_TAB} .shep-dx-drawer .shep-kv {{ grid-template-columns: minmax(6em, max-content) minmax(0, 1fr); }}
{_TAB} .shep-dx-drawer .shep-k, {_TAB} .shep-dx-drawer .shep-v {{ min-width: 0; overflow-wrap: anywhere; }}
{_TAB} .shep-dx-drawer .shep-facts-sum {{ overflow-wrap: anywhere; }}
{_TAB} #shep-dx-change button.shep-row {{ display: block; }}
{_TAB} #shep-dx-change button.shep-row > .shep-row-sub {{ display: block; margin-top: 2px; }}
{_TAB} #shep-dx-change button.shep-row > .shep-tag {{ margin-left: 6px; }}
{_TAB} #shep-dx-change-bd {{ padding: 12px 16px !important; gap: 10px !important; }}
{_TAB} #shep-dx-change .shep-model-head {{ align-items: flex-end !important; flex-wrap: wrap !important; gap: 8px 14px !important; }}
{_TAB} #shep-dx-change .shep-model-head > * {{ min-width: 0 !important; }}
{_TAB} #shep-dx-change .shep-model-body {{ gap: 14px !important; align-items: flex-start !important; flex-wrap: nowrap !important; }}
{_TAB} #shep-dx-change .shep-model-rows {{ flex: 3 1 0 !important; min-width: 0 !important; }}
{_TAB} #shep-dx-change .shep-model-facts {{ flex: 2 1 0 !important; min-width: 0 !important; }}
{_TAB} #shep-dx-change .shep-model-foot {{ align-items: center !important; flex-wrap: wrap !important; gap: 8px 12px !important;
  border-top: 1px solid var(--shep-border); padding-top: 10px !important; }}
{_TAB} #shep-dx-change .shep-model-foot > * {{ flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; }}
{_TAB} #shep-dx-change-note {{ flex: 1 1 280px !important; }}
{_TAB} #shep-dx-change .shep-model-confirm-box > .row > button {{ flex: 0 0 auto !important; white-space: nowrap; }}
{_TAB} #shep-dx-change .shep-model-list:has(.shep-model-confirm-box) > .shep-model-foot {{ display: none !important; }}
{_TAB} #shep-dx-change .shep-confirm .shep-ftbl th, {_TAB} #shep-dx-change .shep-confirm .shep-ftbl td {{ padding-top: 5px; padding-bottom: 5px; }}
{_TAB} #shep-dx-change .shep-confirm .shep-ftbl tbody th {{ white-space: nowrap; }}
@media (min-width: 1400px) and (min-height: 760px) {{
  {_TAB} > .column {{ height: 100%; box-sizing: border-box; }}
  {_TAB} #shep-dx {{ height: 100%; }}
  {_TAB} #shep-dx > .shep-panel {{ max-height: 100%; }}
  {_TAB} .shep-dx-scroll {{ flex: 1 1 auto !important; min-height: 0; overflow: auto; }}
  {_TAB} #shep-dx-input-bd {{ display: flex; flex-direction: column; }}
  {_TAB} .shep-dx-drawer {{ position: absolute !important; top: 0; bottom: 0; z-index: 20; }}
  {_TAB} #shep-dx-identity {{ right: 0; width: max(600px, calc((100% - var(--dx-in-w) - 32px) * 4 / 9 + 16px)); }}
  {_TAB} #shep-dx-change {{ left: calc(var(--dx-in-w) + 16px); right: 0; }}
  {_TAB} #shep-dx-identity-bd {{ flex: 1 1 auto !important; min-height: 0; overflow: auto; }}
  {_TAB} #shep-dx-change-bd {{ flex: 1 1 auto !important; min-height: 0; display: flex; flex-direction: column; }}
  {_TAB} #shep-dx-change .shep-model-list {{ flex: 1 1 auto !important; min-height: 0; display: flex !important; flex-direction: column; }}
  {_TAB} #shep-dx-change .shep-model-body {{ flex: 1 1 0 !important; min-height: 120px; overflow: auto; }}
  {_TAB} #shep-dx-change .shep-model-confirm-box {{ flex: 0 1 auto !important; min-height: 0; overflow: hidden; }}
  {_TAB} #shep-models-clinical-confirm {{ flex: 0 1 auto !important; min-height: 0; overflow: auto !important; }}
  {_TAB} #shep-dx-change .shep-model-confirm-box > .row {{ flex: none !important; }}
  {_TAB} #shep-dx-change .shep-model-list:has(.shep-model-confirm-box) > .shep-model-body {{ min-height: 0 !important; }}
}}
@media (min-width: 1400px) and (min-height: 760px) and (max-height: 819px) {{
  {_TAB} #shep-dx-change .shep-model-list:has(.shep-model-confirm-box) > .shep-model-body {{ display: none !important; }}
}}
@media (max-width: 1399px), (max-height: 759px) {{
  {_TAB} #shep-dx-input, {_TAB} #shep-dx-results, {_TAB} #shep-dx-details {{ flex: 1 1 auto !important; width: 100% !important; max-width: none !important; }}
  {_TAB} .shep-dx-fields {{ display: grid !important; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 12px 18px !important; }}
  {_TAB} .shep-dx-sticky {{ position: static; }}
  {_TAB} #shep-dx-change .shep-model-body {{ flex-direction: column !important; }}
  {_TAB} #shep-dx-change .shep-model-rows, {_TAB} #shep-dx-change .shep-model-facts {{ width: 100% !important; }}
}}
@media (max-width: 760px) {{
  {_TAB} .shep-dx-fields {{ grid-template-columns: minmax(0, 1fr); }}
}}
"""


__all__ = [
    "CLOSE_WHILE_LOADING",
    "FIELD_HELP",
    "PAGE_CSS",
    "SEGMENTS",
    "TOPIC_HELP",
    "badge_html",
    "card_outputs",
    "compare_with_served",
    "create_diagnosis_tab",
    "details_outputs",
    "diagnose",
    "open_change",
    "poll_card",
    "refresh_card",
    "result_identity_md",
    "results_body_html",
    "run_controls",
]
