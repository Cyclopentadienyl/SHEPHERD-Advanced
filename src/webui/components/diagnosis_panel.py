"""
Diagnosis (clinical): the dashboard's first screen (M5-2, UI-B).
================================================================
`docs/working/PLAN_TEST_RESULTS.md` §6, "M5's boundaries", and the M5 mockups
(round 2): the clinician's page for one question — which rare diseases fit
these phenotypes, and what produced the answer.

**Layout** (the page contract, `src/webui/shell.py`; this module's ``PAGE_CSS``):
side by side (the shell's one layout decision, ``data-shep-layout``: a window
wider than tall in which the three panels fit by their own minimum widths, each
``--shep-panel-min`` here) one screen in three panels — the input column | the
candidates | the selected candidate's details — each long region scrolling
inside its own panel; stacked, the panels stack and only the page scrolls.

- **The input column**: HPO terms, **the served task's own input** (K6
  decision 15: "Candidate genes" for a gene task model, "Reference cohort" for
  a patients-like-me model, chosen from the registry's versions; shown and
  hidden by a class as the card's read of the served task says,
  `diagnosis_card.served_task`), Patient ID (a blank field is sent as no id,
  so the API gives the result its own), Options (Top-K, and Explanations for
  a disease task model only, `asks_explanations`), then the
  **"Model in use" card** and Run Diagnosis, the column's foot: side by side
  the fields above them are the column's one scroll region and the card and
  Run sit in flow under it, never over a field (R-25: a sticky card hid the
  HPO box's lower lines and caret). The card (`diagnosis_card`) is the one
  place the served identity appears; the header's "Serving now" strip and the
  Model Configuration accordion (paths, Load/Reload, Save Config, Reset) are
  gone, their gated operations living on in the shared Load
  (`model_list`, through "Change model…"); Save Config's route,
  ``/pipeline/config``, is retired (its file was read by nothing).
- **Stop inference** (the ops round's I1, I4), on the card while a model
  serves: an operator control (D8 on each event's own request) that asks
  first, naming the model shown, and is bound to it (the question takes the
  place of the card's actions while it is open, OPS-RECHECK-N1S, and a
  focus any drawing of the card hides is placed again by the card's keeper,
  `STOP_FOCUS_HEAD`);
  confirmed, it calls the one stop in process (`run_stop`, which
  ``POST /api/v1/pipeline/stop`` calls): the diagnoses already accepted
  finish, new ones are refused until a model is loaded, and it says what the
  stop said, with the resources reading taken after it (never a promise that
  memory was returned). The card then
  shows the server's state in its own words (`diagnosis_card.serving_state`:
  "Stopped at …", the backend's lead "By an operator", its reason under
  Details ›) and the way back to research ("Back to research ›", Research ›
  Training).
  **The card's reads own what the stop shows**: its question closes, and what
  it said goes, once a read no longer offers the model they were bound to
  (`stop_views_closed`), so nothing the page holds outlives the state it
  describes (a "stopped" notice after a model is loaded again, say).
- **Run Diagnosis** is greyed out and not clickable when nothing is served
  or inference is stopping ("Run Diagnosis disabled: …" under it, a
  bubble on hover and on keyboard focus for each cause; the card above says
  which, once) and while a model change is open; while this page's Load
  reloads, Run Diagnosis waits for its answer (set in the list's step before
  the reload, taken back after its answer), and the load itself is the
  card's Loading, read from the server's status like every state, never a
  line of the page's own. With nothing served the API refuses a diagnosis
  (503) with its reason, for every client.
- **The candidates**: the results badge ("Produced by the model in use" /
  "Produced by an earlier model or pipeline — N parts differ ›" / "Cannot
  check against the model in use yet" with its one reason), the identity
  banners of the result, its warnings and the ranked table, in the result
  kind's columns: diseases, a gene task model's candidate genes, or a
  patients-like-me model's reference patients (with the reference cohort
  version the result ranked). **The badge
  compares the whole served identity**, part by part, as the identity table
  reads it (`_identity_rows`), the result's against the card's last read; a
  read that failed or is older than the result never leaves "equal" showing.
  The footer holds the exports (unchanged: CSV and report keep the result's own
  identity; not shown until a result exists) and the medical-device line, once.
- **The details** of the selected row: Evidence · Explanation · Matching
  phenotypes · Paths (in-page segments) for a disease; a gene's or a reference
  patient's values as one key-value list (no segment: no path is discovered).
- **Two drawers**, one open at a time: the identity drawer ("Model in use |
  This result") over the details column, and "Load for inference" (opened by
  the card's "Load a model…" or "Change model…": the clinical preset of the
  shared model list and its Load, with a Task control: All tasks, Disease,
  Gene, Patients-like-me, and the chosen workspace's identity once under its
  select, `model_list.workspace_line_html`; each run named once, by its
  heading) over the candidates and the details. A request D8 refuses gets Change model… disabled with the reason;
  the list is never shown to it.

**Architecture note.** The diagnosis itself goes through the shared API over
HTTP (`_call_diagnose`, `_self_request`), as every frontend's does. What is
served is read through the dashboard's one status reader, in process
(`served_status.read_status`, by the card), and the Load is the shared one
(`model_list`: D8 on its own request, WP2's in-process reload). This module
calls no reload and writes no configuration; its one lifecycle call is the
stop (`run_stop`, in process, after D8's check on its own request).

**Per browser session**: the result, the selection, the card's last read and
the drawers are `gr.State`; nothing here is module state (BACKLOG 28).

Module: src/webui/components/diagnosis_panel.py
"""
from __future__ import annotations

import csv
import io
import json
import logging
import re
import time
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Mapping, NamedTuple, Optional, Tuple

import gradio as gr
import requests

from src.core.types import (
    DISEASE_TASK,
    DataSource,
    GENE_TASK,
    PATIENTS_LIKE_ME,
    RESULT_KINDS,
    result_label,
)
from src.evaluation.caveats import TASK_NAMES
from src.utils import csv_cells
from src.utils.server_address import ServerAddressUnknownError, api_base_url
from src.webui.components import diagnosis_card as card
from src.webui.components import served_status
from src.webui.components.chips import D8_SHORT
from src.webui.components.help import (HIT_TARGET, KEY_FIRST_CLASS, REGION_CLASS, help_button,
                                       text_id)
from src.webui.components.renderers import IDENTITY_BANNERS
from src.webui.components.renderers import NOT_A_MEDICAL_DEVICE as _NOT_A_MEDICAL_DEVICE
from src.webui.components.renderers import OPTION_CHARS
from src.webui.components.renderers import chain_html as _chain_html
from src.webui.components.renderers import cut_at as _cut_at
from src.webui.components.renderers import need_words as _need_words
from src.webui.components.renderers import e as _e
from src.webui.components.renderers import fingerprint_value as _fingerprint_value
from src.webui.components.renderers import grapheme_ends as _grapheme_ends
from src.webui.components.renderers import is_digest as _is_digest
from src.webui.components.renderers import keep_as as _keep_as
from src.webui.components.renderers import marked_line as _marked_line
from src.webui.components.renderers import resources_html as _resources_html
from src.webui.exports import download_update, export_dir, file_slug
from src.webui.components.renderers import summary_list as _summary_list
from src.webui.components.renderers import short_hex as _short_hex
from src.webui.components.renderers import STATUS_MARKS as _STATUS_MARKS
from src.webui.components.renderers import status_mark as _status_mark
from src.webui.operator_access import operator_access
from src.webui.style import FOLD, FOLD_FLOOR, FOLD_ROOM, FOLDED_ATTR, SEG_WHOLE, SIDE, SIDE_SHORT, STACK

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


_GENE_SEPARATORS = re.compile(r"[\s,;]+")


def _parse_gene_list(text: str) -> List[str]:
    """The Candidate genes box as the list a gene task model ranks: entries
    split on whitespace, commas and semicolons, empty ones dropped. **Order,
    repeats and case are kept**: the service maps each entry
    (`KnowledgeGraph.gene_lookup`) and counts the repeats and the unmapped
    ones in its warning, as it does for phenotypes (decision W)."""
    return [entry for entry in _GENE_SEPARATORS.split(text or "") if entry]


#: Where the HPO box's entries end: a line break, a comma or a semicolon.
_HPO_ENTRIES = re.compile(r"[\n,;]+")
#: At most this many entries that are not ids are named in the live line.
_NOT_IDS_SHOWN = 5


def not_ids(text: str) -> List[str]:
    """**The HPO box's entries that hold no HPO id**, which are not sent
    (UI-15, R-5): the box's text split at line breaks, commas and semicolons,
    each entry with a letter or a digit and no id (`_HPO_ID_RE`), in order.
    An id's name in its own entry ("HP:0001250 Seizure") is that id's; a name
    on its own ("Hypotonia") is dropped, and the page says so."""
    entries = (entry.strip() for entry in _HPO_ENTRIES.split(text or ""))
    return [entry for entry in entries
            if re.search(r"\w", entry) and not _HPO_ID_RE.search(entry)]


def hpo_echo_html(text: str) -> str:
    """The line under the HPO box, live: how many ids are read from it, and
    each entry that is not an id and so is not sent (`not_ids`, the first
    `_NOT_IDS_SHOWN` by name). Nothing while the box is empty. Which ids the
    served model then used, and why any was not, is the result's to say
    (`phenotype_account`)."""
    ids, dropped = _parse_hpo_ids(text), not_ids(text)
    if not ids and not dropped:
        return ""
    read = f"{len(ids)} HPO id{'' if len(ids) == 1 else 's'} read" if ids else "No HPO id read"
    if not dropped:
        return f'<p class="shep-dx-echo">{_e(read)}.</p>'
    named = ", ".join(f"“{_e(entry)}”" for entry in dropped[:_NOT_IDS_SHOWN])
    more = len(dropped) - _NOT_IDS_SHOWN
    if more > 0:
        named += f" and {more} more"
    return (f'<p class="shep-dx-echo">{_e(read)}. '
            + _status_mark("warn", f"Not an id, so not sent: {named}", words_html=True)
            + "</p>")


#: Why a query's entry was not used (`phenotype_account`).
NOT_AN_ID = "not an HPO id, not sent"
NOT_IN_GRAPH = "not in the model's graph"
REPEAT = "a repeat, scored once"


def phenotype_account(result: Mapping[str, Any]
                      ) -> Optional[Tuple[List[str], List[Tuple[str, str]]]]:
    """**What a result used of its query, and why the rest was not used**
    (UI-15, R-5, NEW-diagnosis-5), from the server's account of the list it
    was sent (``phenotype_input``: ``used`` in first-occurrence order and the
    request's spelling, ``unknown``, ``repeats_removed``) and the query the
    page sent (``_query_phenotypes``, ``_query_not_ids``): ``(used,
    [(entry, reason), …])``. An entry sent that is neither used nor unknown
    was removed as a repeat; when that does not add up to the server's count
    of repeats, no entry is called a repeat and the count is said instead.
    None when the result carries no account."""
    account = result.get("phenotype_input")
    if not isinstance(account, Mapping) or not isinstance(account.get("used"), list):
        return None
    used = [str(term) for term in account["used"]]
    unknown = {str(term) for term in account.get("unknown") or ()}
    left = [(entry, NOT_AN_ID) for entry in result.get("_query_not_ids") or ()]
    pending, repeats = list(used), []
    for term in result.get("_query_phenotypes") or ():
        if term in unknown:
            left.append((term, NOT_IN_GRAPH))
        elif pending and term == pending[0]:
            pending.pop(0)
        else:
            repeats.append(term)
    removed = account.get("repeats_removed")
    if repeats and removed == len(repeats) and not pending:
        left += [(term, REPEAT) for term in repeats]
    elif isinstance(removed, int) and removed > 0:
        left.append((f"{removed} repeated entr{'y' if removed == 1 else 'ies'}", REPEAT))
    return used, left


#: What a candidate's score is, said under the report's table and in the
#: Score column's help: the value its result kind ranks by (`score` or
#: `similarity`, Spec 1 §2.1); the words shown say what the number is.
SCORE_NOTE = ("Score ranks the candidates of this result. It is not a calibrated "
              "disease probability.")
#: The same for a gene task model's candidate genes (the report's note and the
#: lead of the gene Score's help).
GENE_SCORE_RANKS = ("Score ranks your candidate genes: η × model similarity + (1 − η) × SP "
                    "similarity.")
GENE_SCORE_NOTE = f"{GENE_SCORE_RANKS} It is not a calibrated probability."
#: The same for a patients-like-me model's reference patients.
SIMILARITY_RANKS = "Similarity ranks the cohort's patients: 0 is closest, lower is less alike."
SIMILARITY_CAVEAT = "It is not a calibrated probability, and a shared causal gene is not implied."
SIMILARITY_NOTE = f"{SIMILARITY_RANKS} {SIMILARITY_CAVEAT}"
#: What a reference patient's recorded diagnosis, causal genes and HPO terms
#: are (the details and the report).
REFERENCE_CONTEXT_NOTE = ("Recorded by the reference cohort: context, never an input of this "
                          "query.")

#: Each evidence label's (the API's evidence_label) status mark, by the one
#: indicator's rows (`renderers.status_mark`, the WebUI skill's §6; R-1c):
#: strong path support ● OK, weak ▲ to watch, insufficient evidence ○
#: neutral, in the status tokens in the page and by the same shapes in the
#: report, never a raw hex or a colour emoji. A label not listed has no mark.
EVIDENCE_MARKS = {
    "Strong path support": "ok",
    "Weak path support": "warn",
    "Insufficient evidence": "neutral",
}


def _evidence_shape(label: Any) -> str:
    """The report's mark for an evidence label: its status shape, or ""."""
    kind = EVIDENCE_MARKS.get(label)
    return _STATUS_MARKS[kind][0] if kind else ""


def _evidence_mark(label: Any, *, strong: bool = False) -> str:
    """The page's mark for an evidence label: the status mark with the label
    as its word (in the strong weight with `strong`), or the escaped label
    alone when it has none."""
    kind = EVIDENCE_MARKS.get(label)
    words = f"<b>{_e(label)}</b>" if strong else _e(label)
    return _status_mark(kind, words, words_html=True) if kind else words


# =============================================================================
# API Client
# =============================================================================
def _call_diagnose(
    phenotypes: List[str],
    patient_id: Optional[str] = None,
    top_k: int = 10,
    include_explanations: bool = True,
    *,
    candidate_genes: Optional[List[str]] = None,
    reference_cohort: Optional[str] = None,
) -> Dict[str, Any]:
    """Call the /api/v1/diagnose endpoint and return parsed JSON.

    **A blank Patient ID is sent as None**, not "" and not a fixed placeholder:
    the API then gives the result its own id (``patient_…``), as the field
    promises ("auto-generated if blank"). A fixed placeholder named every blank
    result from every browser as one patient.

    **A task's own input is sent only when given** (K6 decision 15): the
    candidate genes a gene task model ranks, or the label of the reference
    cohort version a patients-like-me model ranks; `diagnose` gives each only
    to its task, and the server refuses or ignores one sent to another model.
    """
    patient = patient_id.strip() if isinstance(patient_id, str) else None
    payload: Dict[str, Any] = {
        "patient_id": patient or None,
        "phenotypes": phenotypes,
        "top_k": top_k,
        "include_explanations": include_explanations,
    }
    if candidate_genes is not None:
        payload["candidate_genes"] = candidate_genes
    if reference_cohort is not None:
        payload["reference_cohort"] = reference_cohort
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
        said = _error_words(e.response)
        return {"error": f"API error: {e.response.status_code} — "
                         f"{said if said is not None else e.response.text}"}
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

    **A refusal the route raises itself** (``HTTPException``) reaches the page
    as ``{"error": "<detail>"}`` (`src.api.main`): every reference cohort
    refusal (another model, none named, another graph, changed samples, an
    unreadable version). Its words are shown, without a server path
    (`_error_words`).

    Never raises: a body that is not JSON, or JSON of another shape, gets the
    general line.
    """
    said = _error_words(response)
    if said is not None:
        return f"Request refused — {said}"
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


def _error_words(response: Any) -> Optional[str]:
    """A response's own refusal words, ``body["error"]`` when it is a non-blank
    string (the service's answer to an ``HTTPException``), with no server path
    left (`diagnosis_card.redacted`: the registry's files read ``<registry>/…``);
    None for any other body."""
    try:
        body = response.json()
    except ValueError:
        return None
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, str) or not error.strip():
        return None
    return card.redacted(" ".join(error.split()))


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
        f"Recognised {received} HPO phenotype entries (repeats included). One "
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
#: The kinds this page draws: every kind of the result union. Disease rows for
#: the disease task; a gene task model's candidate genes and a patients-like-me
#: model's reference patients in their own columns, details and exports (K6
#: decision 15). A kind not listed (a newer API) is said, never drawn as
#: another kind's rows.
PAGE_KINDS = RESULT_KINDS
#: The task kinds whose rows are not diseases: no evidence, explanation or path
#: is discovered for them, so their details are one key-value list.
TASK_ROW_KINDS = (GENE_TASK, PATIENTS_LIKE_ME)


def _result_kind(result: Any) -> Optional[str]:
    """The result's `result_kind` when it is one this page draws; else None, and
    every surface draws nothing from the result."""
    kind = result.get("result_kind") if isinstance(result, Mapping) else None
    return kind if kind in PAGE_KINDS else None


#: Said where a response names no kind this page knows (absent, or a newer
#: kind): nothing is drawn from it.
KIND_NOT_REPORTED = ("The response does not name a result kind this page can read "
                     "(result_kind), so its candidates are not shown.")


def kind_label(result: Any) -> str:
    """The kind's fixed label (`src.core.types.result_label`); KIND_NOT_REPORTED
    where the kind is not one this page knows."""
    kind = _result_kind(result)
    if kind is None:
        return KIND_NOT_REPORTED
    return result_label(kind)


def _score(candidate: Mapping[str, Any], kind: Optional[str]) -> Any:
    """The value the kind ranks by; None for a kind this page does not know."""
    if kind in (DISEASE_TASK, GENE_TASK):
        return candidate.get("score")
    if kind == PATIENTS_LIKE_ME:
        return candidate.get("similarity")
    return None


# =============================================================================
# Formatting helpers — diagnosis results
# =============================================================================
def _format_results_table(candidates: List[Dict], kind: Optional[str] = None,
                          result: Optional[Mapping[str, Any]] = None) -> str:
    """Format candidates as a Markdown table (the exported report), its columns
    the kind's: the task head's score and Phenotypes k/N (`phenotype_match`,
    from `result`'s query) for a disease task model."""
    if kind is None:
        return f"*{KIND_NOT_REPORTED}*"
    if not candidates:
        return "*No candidates found.*"
    if kind in TASK_ROW_KINDS:
        return _format_task_table(candidates, kind)

    columns = ["Rank", "Disease", "Score", "Phenotypes", "Evidence"]
    rows = []
    for c in candidates:
        match = phenotype_match(result, c) if isinstance(result, Mapping) else None
        cells = [str(c.get("rank")), f"**{c.get('disease_name')}**", _num(_score(c, kind)),
                 f"{match[0]} / {match[1]}" if match else "not reported"]
        label = c.get("evidence_label") or "—"
        cells.append(" ".join(part for part in (_evidence_shape(label), label) if part))
        rows.append("| " + " | ".join(cells) + " |")
    header = ("| " + " | ".join(columns) + " |\n"
              + "|" + "|".join(":----:" if c == "Rank" else "---" for c in columns) + "|\n")
    table = header + "\n".join(rows)
    return table + "\n\n" + SCORE_NOTE


def _md_list(values: Any) -> str:
    """Recorded values for a Markdown cell or line, each escaped, "; "-joined;
    "not recorded" for none."""
    shown = [_md_text(v) for v in values or [] if str(v).strip()]
    return "; ".join(shown) if shown else "not recorded"


def _md_diagnosis(candidate: Mapping[str, Any]) -> str:
    """A reference patient's recorded diagnosis: its name, else its id, else
    "not recorded"."""
    if candidate.get("disease_name"):
        return _md_text(candidate["disease_name"])
    if candidate.get("disease_id"):
        return _md_code(_clean_disease_id(candidate["disease_id"]))
    return "not recorded"


def _format_task_table(candidates: List[Mapping[str, Any]], kind: str) -> str:
    """The report's table of a gene task model's candidate genes or a
    patients-like-me model's reference patients, every value escaped, and the
    kind's note under it."""
    if kind == GENE_TASK:
        columns = ["Rank", "Gene", "Gene id", "Score", "Model similarity", "SP similarity",
                   "Mean hop distance"]
        rows = [[_md_text(c.get("rank")), f"**{_md_text(c.get('gene_name'))}**",
                 _md_code(c.get("gene_id")), _num(c.get("score")), _num(c.get("embsim")),
                 _num(c.get("sp_similarity")), _num(c.get("mean_distance"))]
                for c in candidates]
        note = GENE_SCORE_NOTE
    else:
        columns = ["Rank", "Research ID", "Similarity", "Diagnosis", "Causal genes"]
        rows = [[_md_text(c.get("rank")), _md_code(c.get("case_id")), _num(c.get("similarity")),
                 _md_diagnosis(c), _md_list(c.get("causal_genes"))] for c in candidates]
        note = SIMILARITY_NOTE
    header = ("| " + " | ".join(columns) + " |\n"
              + "|" + "|".join(":----:" if c == "Rank" else "---" for c in columns) + "|\n")
    return header + "\n".join("| " + " | ".join(row) + " |" for row in rows) + "\n\n" + note


def _format_evidence_detail(candidate: Dict, names: Optional[Mapping[str, str]] = None) -> str:
    """Format a candidate's evidence package in Markdown (the exported report;
    the page draws it as escaped HTML, `evidence_html`): its label, the
    server's summary, then the shortest path and the direct paths (each step
    name first, its id after: `_md_term`), with no retired pipeline's mode
    name (R-3)."""
    pkg = candidate.get("evidence_package")
    if not pkg:
        return "*No evidence package available. Run with `include_explanations=true`.*"

    mode = pkg.get("mode") or "unknown"
    summary = pkg.get("summary") or ""
    label = pkg.get("evidence_label") or "—"

    parts = [
        "### " + " ".join(part for part in (_evidence_shape(label), label) if part),
        "",
        f"> {summary}",
        "",
    ]

    if mode == "direct_path":
        paths = sorted((p for p in pkg.get("direct_paths") or [] if isinstance(p, list)), key=len)
        parts.append(f"- **Shortest path**: {_hops(pkg.get('min_path_length'))}")
        parts.append(f"- **Direct paths**: {len(paths)}")
        parts.append("")
        for i, path in enumerate(paths[:5], 1):
            parts.append(f"{i}. " + " → ".join(_md_term(n, names or {}) for n in path))

    # Supporting genes
    genes = candidate.get("supporting_genes", [])
    if genes:
        parts.append("")
        parts.append(f"**Supporting genes**: {', '.join(genes)}")

    return "\n".join(parts)


#: A node's source tags (`DataSource`), the part of a node id a reader drops.
_NODE_SOURCES = frozenset(source.value for source in DataSource)


def _node_display_name(node_str: str) -> str:
    """A node id without its source tag, for readability: 'mondo:MONDO:0011073'
    reads 'MONDO:0011073', 'disgenet:CDKL5' 'CDKL5'. A tag is dropped when it
    is a known source or a namespaced id follows it, so a term as a request
    spells it ('HP:0001250') is kept whole."""
    source, sep, local = node_str.partition(":")
    return local if sep and local and (source in _NODE_SOURCES or ":" in local) else node_str


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
# computed, absent); a value that is not there is written as that state and its
# reason, never left blank and never "unknown".

#: The plan's claim boundary for the UI (PLAN_PROVENANCE_CONTRACT.md §6), verbatim.
#: Verified states carry ✓ in the report's identity table, so the sentence
#: refers to them; the identity drawer draws its ✓ as the page's verified mark
#: (`_boundary_html`).
IDENTITY_CLAIM_BOUNDARY = (
    "Checked: these files are the ones their records name, and the records agree "
    "on the relations marked ✓. Not checked: who changed them, or anything a "
    "person able to edit both files and records could alter."
)
#: Added under it when a served identity has no part marked ✓ (one the response
#: left parts of out), so it is not read as a claim that something about this
#: result was checked.
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

#: The word for a response that carried no `served_identity` (a server from
#: before contract M3c, or a malformed reply), or a part of one it left out.
IDENTITY_NOT_REPORTED = "not_reported"
#: The word for what is in use now when the card's read found no model loaded
#: (the identity drawer's "In use now" side). A result never has it: every
#: result comes from a model.
IDENTITY_NO_MODEL = "no_model"
_SERVED = "served"

#: The report's mark of a verified part (Markdown); the page draws the
#: shared status mark for it (`_part_html`, `_boundary_html`).
_VERIFIED_MARK = "✓"
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
    IDENTITY_NOT_REPORTED: "the API's response did not carry a served identity",
    IDENTITY_NO_MODEL: "no model is loaded on this server now, so nothing is served",
}
_SCORING_MODE_WORDS = {
    "disease_task": "a model finetuned for the disease task, over its graph view",
    "gene_task": "a model finetuned for the gene task: its similarity mixed with "
                 "shortest-path proximity",
    "patients_like_me": "a model finetuned for patients-like-me: reference patients by "
                        "similarity",
}
_PATH_SETTING_KEYS = (
    "max_path_length", "path_length_penalty", "max_paths_per_candidate",
    "use_direct_path_optimization",
)

#: (headline, detail) for each state a reader must not miss. Words carry the
#: meaning; the tab adds an icon, the report a blockquote, neither relies on it.
#: Defined once in the shared renderers (Model Facts' warnings say the same).
_IDENTITY_BANNERS = IDENTITY_BANNERS
assert set(_IDENTITY_BANNERS) >= {IDENTITY_NOT_REPORTED, "synthetic"}


def _identity_of(container: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]]]:
    """What a diagnosis result says produced it: ``(kind, identity)``.

    ``kind`` is ``"served"`` with the identity dict, or ``"not_reported"`` when
    the key is missing or is not an object.
    """
    identity = container.get("served_identity")
    if not isinstance(identity, dict):
        return IDENTITY_NOT_REPORTED, None
    return _SERVED, identity


def _served_now(state: Mapping[str, Any]) -> Tuple[str, Optional[Dict[str, Any]]]:
    """What the card's last read says is in use now: ``(kind, identity)``,
    ``"no_model"`` when no pipeline is loaded, ``"not_reported"`` for a status
    that names a pipeline and no identity."""
    status = state.get("status") or {}
    if not status.get("initialized"):
        return IDENTITY_NO_MODEL, None
    return _identity_of(status)


def _field(part: Any, key: str) -> Any:
    """``part[key]``, or `_MISSING` when ``part`` is not a dict or lacks the key."""
    return part.get(key, _MISSING) if isinstance(part, dict) else _MISSING


#: Whether a value is a SHA-256 (the shared CSV cells' rule, `csv_cells.is_sha256`).
_is_sha256 = csv_cells.is_sha256


def _is_synthetic(identity: Dict[str, Any]) -> bool:
    return _field(_field(identity, "dataset_record"), "origin") == "synthetic"


#: A JSON scalar as text (`csv_cells.plain_text`): ``true``/``false`` for
#: booleans, floats as Python prints them (``0.7``, ``1.0``).
_plain_value = csv_cells.plain_text


# ----------------------------------------------------------------- Markdown form
#: Characters that could start Markdown or HTML syntax, or end a table cell.
_MD_SPECIAL_RE = re.compile(r"([\\`*_{}\[\]<>#|~!])")


def _one_line(value: Any) -> str:
    """A value from the API as plain text on one line (`csv_cells.plain_text`)."""
    return " ".join(_plain_value(value).split())


def _md_escape(text: str) -> str:
    """Text with every character that could start Markdown or HTML escaped."""
    return _MD_SPECIAL_RE.sub(r"\\\1", text)


def _md_text(value: Any) -> str:
    """A value from the API as plain text, safe in a Markdown table cell: one
    line, every character that could start Markdown or HTML escaped."""
    return _md_escape(_one_line(value))


def _md_code(value: Any, in_table: bool = True) -> str:
    """A value from the API in a code span, one line; a backtick in the value
    gets a longer fence.

    In a table cell (``in_table``) a pipe is escaped even inside the span: GFM
    splits a row on ``|`` before it parses code spans. Outside a table the
    escape would be shown literally, since a code span keeps backslashes.
    """
    text = _one_line(value)
    if in_table:
        text = text.replace("|", "\\|")
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * (longest + 1)
    pad = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{fence}{pad}{text}{pad}{fence}"


def _md_digest(value: Any) -> str:
    """A full digest, ``sha256:``-prefixed, in a code span."""
    return f"`sha256:{value}`" if _is_sha256(value) else _md_code(value)


# ----------------------------------------------------------------- identity values
class _Piece(NamedTuple):
    """**One piece of an identity value**, as the one builder of the identity
    (`_identity_rows`) says it: ``text`` (the builder's own words, or a value
    from the API on one line), ``code`` (a value from the API as code) or
    ``digest`` (a full sha256). Its two readers draw the same pieces, each in
    its own form: the report as Markdown (`_value_md`), the identity drawer as
    HTML (`_value_html`). Neither parses the other's form."""

    kind: str
    text: str


#: An identity value: its pieces, in order.
_Value = Tuple[_Piece, ...]


def _words(text: str) -> _Piece:
    """The builder's own words, as they are."""
    return _Piece("text", text)


def _text(value: Any) -> _Piece:
    """A value from the API as words, on one line."""
    return _Piece("text", _one_line(value))


def _code(value: Any) -> _Piece:
    """A value from the API as code, on one line."""
    return _Piece("code", _one_line(value))


def _digest(value: Any) -> _Piece:
    """A full digest; any other value as code."""
    return _Piece("digest", value) if _is_sha256(value) else _code(value)


def _value_md(value: _Value) -> str:
    """An identity value as the report's Markdown table cell: words escaped
    (`_md_escape`), code in a span (`_md_code`), a digest ``sha256:``-prefixed
    in one (`_md_digest`)."""
    return "".join(_md_escape(piece.text) if piece.kind == "text"
                   else _md_digest(piece.text) if piece.kind == "digest"
                   else _md_code(piece.text) for piece in value)


def _value_html(value: _Value) -> str:
    """An identity value as the identity drawer's HTML: words escaped, code as
    code, a digest as the one digest value (`renderers.fingerprint_value`, 12
    hex, the whole in its title; "This result in full" has its Copy). Every
    piece is escaped, so a value from the API is never markup."""
    return "".join(_e(piece.text) if piece.kind == "text"
                   else _fingerprint_value(piece.text, what="digest", copy=False)
                   if piece.kind == "digest"
                   else f"<code>{_e(piece.text)}</code>" for piece in value)


_NOT_REPORTED_CELLS: Tuple[str, _Value] = (
    "not\\_reported", (_words("not reported — the response did not include this value"),)
)


def _setting(value: Any) -> _Piece:
    if value is _MISSING:
        return _words("not reported")
    return _code(value) if isinstance(value, str) else _text(value)


def _checkpoint_cells(identity: Dict[str, Any]) -> Tuple[str, _Value]:
    part = _field(identity, "checkpoint")
    state = _field(part, "state")
    if state == "verified":
        return f"{_VERIFIED_MARK} verified", (_digest(_field(part, "sha256")),)
    return _NOT_REPORTED_CELLS


def _graph_cells(identity: Dict[str, Any]) -> Tuple[str, _Value]:
    part = _field(identity, "graph")
    state = _field(part, "state")
    if state == "verified":
        return f"{_VERIFIED_MARK} verified", (_digest(_field(part, "kg_sha256")),)
    return _NOT_REPORTED_CELLS


def _manifest_cells(identity: Dict[str, Any]) -> Tuple[str, _Value]:
    part = _field(identity, "graph")
    state = _field(part, "state")
    if state == "verified":
        return f"{_VERIFIED_MARK} verified", (_digest(_field(part, "split_manifest_sha256")),)
    return _NOT_REPORTED_CELLS


def _shortest_path_cells(identity: Dict[str, Any]) -> Tuple[str, _Value]:
    part = _field(identity, "shortest_paths")
    state = _field(part, "state")
    if state == "computed":
        return ("computed", (_words("computed — exact hop distances from the model's graph "
                                    "view; no table is read"),))
    if state == "absent":
        if _field(identity, "scoring_mode") == "disease_task":
            reason = "a disease task model scores without one"
        elif _field(identity, "scoring_mode") == "patients_like_me":
            reason = "a patients-like-me model scores without one"
        else:
            reason = "it does not contribute to the score"
        return "absent", (_words(f"absent — no table is served; {reason}"),)
    return _NOT_REPORTED_CELLS


#: What a dataset record's origin means, where the panel names one (the
#: served identity).
_DATASET_ORIGIN_MEANING = {
    "synthetic": "a generated stand-in, not a clinical knowledge graph",
    "files": "built from identified source files",
}


def _dataset_record_cells(identity: Dict[str, Any]) -> Tuple[str, _Value]:
    part = _field(identity, "dataset_record")
    state = _field(part, "state")
    if state == "verified":
        origin = _field(part, "origin")
        # **Never "verified" alone for a synthetic record** (contract M3b): the
        # binding and the origin are stated together.
        meaning = _DATASET_ORIGIN_MEANING.get(origin, "not described by this page")
        return (f"{_VERIFIED_MARK} verified · {_md_text(origin)}",
                (_digest(_field(part, "sha256")), _words(" — origin "), _text(origin),
                 _words(f": {meaning}")))
    return _NOT_REPORTED_CELLS


def _scoring_mode_cells(identity: Dict[str, Any]) -> Tuple[str, _Value]:
    mode = _field(identity, "scoring_mode")
    if mode is _MISSING or mode is None:
        return _NOT_REPORTED_CELLS
    words = _SCORING_MODE_WORDS.get(mode, "not described by this page")
    return "in effect", (_code(mode), _words(f" — {words}"))


def _device_cells(identity: Dict[str, Any]) -> Tuple[str, _Value]:
    settings = _field(identity, "settings")
    device, requested = _field(settings, "device"), _field(settings, "device_requested")
    if device in (_MISSING, None) or requested in (_MISSING, None):
        return _NOT_REPORTED_CELLS
    return "in effect", (_code(device), _words(" in effect; requested: "), _code(requested))


def _eta_cells(identity: Dict[str, Any]) -> Tuple[str, _Value]:
    eta = _field(_field(identity, "settings"), "eta")
    if eta is _MISSING:
        return _NOT_REPORTED_CELLS
    if eta is None and _field(identity, "scoring_mode") == "disease_task":
        return "none", (_words("none — a disease task model has no terms to mix"),)
    if eta is None and _field(identity, "scoring_mode") == "patients_like_me":
        return "none", (_words("none — a patients-like-me model has no terms to mix"),)
    if eta is None:
        return _NOT_REPORTED_CELLS
    if _field(identity, "scoring_mode") == "gene_task":
        return "in effect", (_text(eta), _words(" — the gene model's own: its similarity "
                                                "against shortest-path proximity"))
    return "in effect", (_text(eta),)


#: The scoring modes that discover no path: their rows are genes or reference
#: patients, so the path settings are not applicable to them.
_NO_PATHS = {"gene_task": "a gene task model discovers no paths",
             "patients_like_me": "a patients-like-me model discovers no paths"}


def _path_settings_cells(identity: Dict[str, Any]) -> Tuple[str, _Value]:
    # In effect for a disease task model: paths explain the diseases shown.
    settings = _field(identity, "settings")
    if not isinstance(settings, dict):
        return _NOT_REPORTED_CELLS
    mode = _field(identity, "scoring_mode")
    if mode in _NO_PATHS:
        return "not applicable", (_words(f"not applicable — {_NO_PATHS[mode]}"),)
    pieces: List[_Piece] = []
    for key in _PATH_SETTING_KEYS:
        if pieces:
            pieces.append(_words(" · "))
        pieces += [_code(key), _words(" "), _setting(_field(settings, key))]
    return "in effect", tuple(pieces)


def _phenotype_rule_cells(identity: Dict[str, Any]) -> Tuple[str, _Value]:
    version = _field(_field(identity, "settings"), "phenotype_normalisation_version")
    if version is _MISSING or version is None:
        return _NOT_REPORTED_CELLS
    return "in effect", (_words("version "), _text(version))


def _identity_rows(kind: str, identity: Optional[Dict[str, Any]]) -> List[Tuple[str, str, _Value]]:
    """``(part, state, value)``, one row per part, never blank: ``state`` the
    report's State cell (Markdown; the drawer reads only whether it is
    verified), ``value`` its pieces (`_Piece`).

    The one builder of the identity: the report (`_value_md`) and the identity
    drawer (`_value_html`) both draw these rows, so they cannot disagree.
    """
    if kind != _SERVED or identity is None:
        return [(part, _md_text(kind), (_text(kind), _words(f" — {_NO_IDENTITY_REASON[kind]}")))
                for part in _IDENTITY_PARTS]
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
    # says that.
    if kind == IDENTITY_NOT_REPORTED:
        lines.append(IDENTITY_CHECKS_NOT_REPORTED)
    elif not any(state.startswith(_VERIFIED_MARK)
                 for _, state, _ in _identity_rows(kind, identity)):
        lines.append(nothing_checked)
    return lines


def _identity_table_lines(kind: str, identity: Optional[Dict[str, Any]]) -> List[str]:
    """The identity table, the claim boundary under it and the fixed statement."""
    lines = ["| Part | State | Value |", "|------|-------|-------|"]
    lines += [f"| {part} | {state} | {_value_md(value)} |"
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
    return banners


# =============================================================================
# Export helpers — write ALL candidates (not just the previewed one) to a file
# =============================================================================
#: **The candidate columns follow the result's kind** (Spec 1 §2.1): every row
#: names it (`result_kind`), and a kind's file holds only its own values (the
#: candidate-column convention: an empty cell means no value, never 0). An
#: evidence count is empty when no evidence package was built (explanations
#: off), never 0.
_CSV_BASE_COLUMNS = ["result_kind", "rank", "disease_id", "disease_name"]
_CSV_SCORE_COLUMNS = {
    DISEASE_TASK: ["score"],
}
_CSV_EVIDENCE_COLUMNS = ["evidence_label", "evidence_mode", "evidence_summary",
                         "min_path_length", "num_direct_paths", "supporting_genes"]
#: **The task kinds' own columns** (K6 decision 15): a gene's or a reference
#: patient's values, then what the whole result carries that its rows need to
#: be read (a gene task model's fixed SP bounds; the reference cohort version a
#: patients-like-me result ranked), the same on every row, before the served
#: columns. A reference patient's lists are "; "-joined.
_CSV_TASK_COLUMNS = {
    GENE_TASK: ["result_kind", "rank", "gene_id", "gene_name", "score", "embsim",
                "sp_similarity", "mean_distance", "sp_bounds_lower", "sp_bounds_upper"],
    PATIENTS_LIKE_ME: ["result_kind", "rank", "case_id", "similarity", "disease_id",
                       "disease_name", "causal_genes", "phenotypes", "reference_cohort_label",
                       "reference_cohort_version_sha256", "reference_cohort_samples_sha256",
                       "reference_cohort_patients"],
}


#: **The query's phenotypes as the result used them** (R-5), the same on every
#: row, after the candidate columns and before the served ones: the ids used
#: ("; "-joined), and each entry not used with its reason ("Hypotonia (not an
#: HPO id, not sent)"), as the page's "Used" and "Not used" rows and the
#: report's lines (`phenotype_account`). ``not_reported`` in both for a
#: response that gave no account; "Not used" is empty when every entry was.
_CSV_QUERY_COLUMNS = ["phenotypes_used", "phenotypes_not_used"]


def _query_csv_values(result: Mapping[str, Any]) -> Dict[str, str]:
    account = phenotype_account(result)
    if account is None:
        return dict.fromkeys(_CSV_QUERY_COLUMNS, IDENTITY_NOT_REPORTED)
    used, left = account
    return {"phenotypes_used": _csv_text("; ".join(used)),
            "phenotypes_not_used": _csv_text("; ".join(f"{entry} ({reason})"
                                                       for entry, reason in left))}


def _candidate_csv_columns(kind: Optional[str]) -> List[str]:
    """The kind's candidate columns, in order: a task kind's own
    (`_CSV_TASK_COLUMNS`); for a disease kind, its score columns, the evidence
    columns and the matches."""
    if kind in _CSV_TASK_COLUMNS:
        return list(_CSV_TASK_COLUMNS[kind])
    return (_CSV_BASE_COLUMNS + _CSV_SCORE_COLUMNS.get(kind, []) + _CSV_EVIDENCE_COLUMNS
            + ["matching_phenotypes"])


def _result_csv_values(result: Mapping[str, Any]) -> Dict[str, Any]:
    """What a task result carries for every row of its CSV (`_CSV_TASK_COLUMNS`'
    result-level columns): the gene task's SP bounds, the patients-like-me
    reference cohort version. Numbers by `_fmt_num`, texts defused, digests
    ``sha256:``-prefixed; an absent value is an empty cell, never 0."""
    bounds = result.get("sp_bounds")
    bounds = bounds if isinstance(bounds, (list, tuple)) and len(bounds) == 2 else (None, None)
    reference = result.get("reference_cohort")
    reference = reference if isinstance(reference, Mapping) else {}
    patients = reference.get("patients")
    return {
        "sp_bounds_lower": _fmt_num(bounds[0]), "sp_bounds_upper": _fmt_num(bounds[1]),
        "reference_cohort_label": _csv_text(reference.get("label") or ""),
        "reference_cohort_version_sha256": _csv_digest(reference.get("version_sha256"), ""),
        "reference_cohort_samples_sha256": _csv_digest(reference.get("samples_sha256"), ""),
        "reference_cohort_patients": (patients if isinstance(patients, int)
                                      and not isinstance(patients, bool) else ""),
    }


# **What produced the result, on every row** (contract M3c). Appended after the
# candidate columns, never between them; one header row and no preamble, so the
# file stays a plain table. Each column holds one atomic part, the same on every row.
# Digests are written in full as "sha256:<hex>" -- the prefix names the
# algorithm and stops a spreadsheet reading an all-digit digest as a number. A
# value the identity holds as null is written as a word, never an empty cell:
#   none          nothing applies: no eta for a model that mixes nothing
#   not_reported  every column of a response with no `served_identity`, or a
#                 value one left out
_SERVED_CSV_COLUMNS = [
    "served_scoring_mode",
    "served_checkpoint_state", "served_checkpoint_sha256",
    "served_kg_state", "served_kg_sha256", "served_split_manifest_sha256",
    "served_sp_state",
    "served_dataset_record_state", "served_dataset_origin",
    "served_dataset_record_sha256",
    "served_device", "served_device_requested", "served_eta",
    "served_max_path_length", "served_path_length_penalty",
    "served_max_paths_per_candidate",
    "served_direct_path_optimization", "served_phenotype_normalisation_version",
]

def _fmt_num(value: Any) -> str:
    """A number to four places; empty for no value (never 0)."""
    return (f"{value:.4f}" if isinstance(value, (int, float)) and not isinstance(value, bool)
            else "")


#: A free-text cell, defused by the one rule every CSV here follows
#: (`csv_cells.text_cell`): one a spreadsheet would run as a formula gets a
#: leading apostrophe. Numeric columns are written by `_fmt_num` and never
#: pass through it, so a negative number stays a number.
_csv_text = csv_cells.text_cell


def _csv_value(value: Any, null_word: str) -> str:
    """One identity value as a CSV cell, never empty (`csv_cells.value_cell`):
    a key the response did not carry, or an empty text, is ``not_reported``."""
    if value is _MISSING:
        return IDENTITY_NOT_REPORTED
    return csv_cells.value_cell(value, null_word=null_word, empty_word=IDENTITY_NOT_REPORTED)


def _csv_digest(value: Any, null_word: str) -> str:
    """A digest as ``sha256:<hex>`` (`csv_cells.digest_cell`), anything else
    as `_csv_value`."""
    return csv_cells.digest_cell(value) or _csv_value(value, null_word)


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
    missing = IDENTITY_NOT_REPORTED

    def setting(key: str, null_word: str = missing) -> str:
        return _csv_value(_field(settings, key), null_word)

    return {
        "served_scoring_mode": _csv_value(_field(identity, "scoring_mode"), missing),
        "served_checkpoint_state": _csv_value(_field(checkpoint, "state"), missing),
        "served_checkpoint_sha256": _csv_digest(_field(checkpoint, "sha256"), missing),
        "served_kg_state": _csv_value(_field(graph, "state"), missing),
        "served_kg_sha256": _csv_digest(_field(graph, "kg_sha256"), missing),
        "served_split_manifest_sha256": _csv_digest(
            _field(graph, "split_manifest_sha256"), missing
        ),
        "served_sp_state": _csv_value(_field(shortest_paths, "state"), missing),
        "served_dataset_record_state": _csv_value(_field(record, "state"), missing),
        "served_dataset_origin": _csv_value(_field(record, "origin"), missing),
        "served_dataset_record_sha256": _csv_digest(_field(record, "sha256"), missing),
        "served_device": setting("device"),
        "served_device_requested": setting("device_requested"),
        "served_eta": setting("eta", "none"),
        "served_max_path_length": setting("max_path_length"),
        "served_path_length_penalty": setting("path_length_penalty"),
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
    in the result kind's columns (`_candidate_csv_columns`), with the query's
    phenotypes as used (`_CSV_QUERY_COLUMNS`) and what produced the result
    (`_SERVED_CSV_COLUMNS`) repeated on each row."""
    kind = _result_kind(result)
    served = _served_csv_values(result)
    whole = _result_csv_values(result) if kind in _CSV_TASK_COLUMNS else {}
    buf = io.StringIO()
    query = _query_csv_values(result)
    writer = csv.DictWriter(buf, fieldnames=(_candidate_csv_columns(kind) + _CSV_QUERY_COLUMNS
                                             + _SERVED_CSV_COLUMNS),
                            extrasaction="ignore")
    writer.writeheader()
    for c in result.get("candidates", []) if kind is not None else []:
        pkg = c.get("evidence_package")
        writer.writerow({
            "result_kind": kind,
            "rank": c.get("rank"),
            "disease_id": _csv_text(_clean_disease_id(c.get("disease_id") or "")),
            "disease_name": _csv_text(c.get("disease_name") or ""),
            "gene_id": _csv_text(c.get("gene_id") or ""),
            "gene_name": _csv_text(c.get("gene_name") or ""),
            "embsim": _fmt_num(c.get("embsim")),
            "sp_similarity": _fmt_num(c.get("sp_similarity")),
            "mean_distance": _fmt_num(c.get("mean_distance")),
            "case_id": _csv_text(c.get("case_id") or ""),
            "similarity": _fmt_num(c.get("similarity")),
            "causal_genes": _csv_text("; ".join(map(str, c.get("causal_genes") or []))),
            "phenotypes": _csv_text("; ".join(map(str, c.get("phenotypes") or []))),
            **whole,
            "score": _fmt_num(c.get("score")),
            "evidence_label": _csv_text(c.get("evidence_label") or ""),
            "evidence_mode": _csv_text((pkg or {}).get("mode") or ""),
            "evidence_summary": _csv_text(((pkg or {}).get("summary") or "")
                                          .replace("\n", " ").strip()),
            "min_path_length": (pkg or {}).get("min_path_length", ""),
            "num_direct_paths": len(pkg.get("direct_paths") or []) if pkg else "",
            "matching_phenotypes": _csv_text("; ".join(c.get("matching_phenotypes") or [])),
            "supporting_genes": _csv_text("; ".join(c.get("supporting_genes") or [])),
            **query,
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
    # The response's format version, read as given and by no version rule, so
    # the kept report says which shape its response had (an older one still
    # reads here); left out unless it is a number.
    version = result.get("result_schema_version")
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
        ("Result format", f"result_schema_version {version}" if isinstance(version, int)
         and not isinstance(version, bool) else None),
        ("Candidates", len(candidates)),
    ):
        if value not in (None, ""):
            lines.append(f"- **{key}**: {value}")
    account = phenotype_account(result)
    names = _names(result)
    if account is not None:
        used, left = account
        lines.append(f"- **Phenotypes used**: "
                     f"{', '.join(_md_term(t, names) for t in used) or 'none'}")
        if left:
            lines.append("- **Phenotypes not used**: "
                         + "; ".join(f"{entry} ({reason})" for entry, reason in left))
    elif result.get("_query_phenotypes"):
        lines.append(f"- **Phenotypes sent**: {', '.join(result['_query_phenotypes'])}")
    lines += _task_meta_lines(result, result_kind)
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
    lines += [_format_results_table(candidates, result_kind, result)]
    if result_kind == GENE_TASK:     # the table holds every value of a gene
        return "\n".join(lines)
    lines += ["", "## Per-candidate detail"]
    if result_kind == PATIENTS_LIKE_ME:
        for c in candidates:
            lines += ["", f"### #{_md_text(c.get('rank'))} — {_md_code(c.get('case_id'), False)}",
                      f"Similarity {_num(c.get('similarity'))}", "",
                      f"**Recorded HPO terms**: {_md_list(c.get('phenotypes'))}", "",
                      f"*{REFERENCE_CONTEXT_NOTE}*", "", "---"]
        return "\n".join(lines)
    for c in candidates if result_kind is not None else []:
        score = _score(c, result_kind)
        lines += [
            "",
            f"### #{c.get('rank')} — {c.get('disease_name')}",
            f"`{_clean_disease_id(c.get('disease_id', ''))}` · score {_num(score)}",
            "",
        ]
        match = phenotype_match(result, c)
        if match:
            found = [_md_term(t, names) for t in c.get("matching_phenotypes") or []]
            lines += [f"**Phenotypes annotated to this disease** ({match[0]} of {match[1]} "
                      f"used): {', '.join(found) or 'none'}", ""]
        lines += [
            "**Evidence**",
            "",
            _format_evidence_detail(c, names),
            "",
            "**Full explanation**",
            "",
            _format_full_explanation(c),
            "",
            "---",
        ]
    return "\n".join(lines)


def _task_meta_lines(result: Mapping[str, Any], kind: Optional[str]) -> List[str]:
    """The report's lines for what a task's query carried beside its
    phenotypes, every value escaped: the candidate genes submitted and the gene
    model's SP bounds; the reference cohort version a patients-like-me result
    ranked (label, version and samples digests, patients)."""
    if kind == GENE_TASK:
        lines = []
        genes = result.get("_query_candidate_genes") or []
        if genes:
            lines.append(f"- **Candidate genes submitted**: "
                         f"{', '.join(_md_text(g) for g in genes)}")
        bounds = result.get("sp_bounds")
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
            lines.append(f"- **SP similarity bounds** (the model's own): {_num(bounds[0])} to "
                         f"{_num(bounds[1])}")
        return lines
    if kind == PATIENTS_LIKE_ME:
        reference = result.get("reference_cohort")
        reference = reference if isinstance(reference, Mapping) else {}
        return [f"- **Reference cohort**: {_md_code(reference.get('label'), False)}",
                f"- **Reference cohort version** (cohort.json): "
                f"{_md_digest(reference.get('version_sha256'))}",
                f"- **Reference cohort samples**: {_md_digest(reference.get('samples_sha256'))}",
                f"- **Reference cohort patients**: {_md_text(reference.get('patients'))}"]
    return []


def _export_basename(result: Dict[str, Any]) -> str:
    """diagnosis_[SYNTHETIC_]<patient>_<session-signature> — distinct per run
    for easy filing.

    **A file name carries the state a copied file must not lose** (contract
    M3c): SYNTHETIC for a result from a synthetic workspace. A CSV has no place
    for a banner, and a file is often seen by its name alone.
    """
    patient = file_slug(result.get("patient_id"), "webui_patient")
    session = file_slug(result.get("session_id") or result.get("timestamp"), "run")
    kind, identity = _identity_of(result)
    marker = ("SYNTHETIC_" if kind == _SERVED and identity is not None
              and _is_synthetic(identity) else "")
    return f"diagnosis_{marker}{patient}_{session}"


def _write_exports(result: Dict[str, Any]) -> Tuple[str, str]:
    """Generate both export files for a result up front (so each export's
    link, `exports.offer`, serves its file on the first click). Returns
    (csv_path, md_path)."""
    base = _export_basename(result)
    # The dashboard's one download directory (`src/webui/exports.py`); files
    # are named by session signature so they do not collide.
    directory = export_dir()
    csv_path = directory / f"{base}.csv"
    md_path = directory / f"{base}_report.md"
    csv_path.write_text(_build_results_csv(result), encoding="utf-8")
    md_path.write_text(_build_results_report_md(result), encoding="utf-8")
    return str(csv_path), str(md_path)



# =============================================================================
# The page: help texts and fixed words
# =============================================================================
#: How the exports stand with no file to download (each an
#: `exports.download_update`, new at each answer; OPS-FIX-R1): with no result
#: they are not shown at all (the mockup's first screen has none) but stay
#: mounted (``"hidden"``), so the next result only shows them; a result whose
#: export files could not be written, or that has no candidate, shows them
#: disabled, beside the reason.
EXPORTS_HIDDEN = "hidden"

#: **What must stay visible under a field** (``info=``; the WebUI skill §7: a
#: format is never only in a bubble or a placeholder; UI-15, R-6).
FIELD_INFO = {
    "dx_hpo_terms": "Only HP:####### ids are read. Names and punctuation are ignored.",
    "dx_patient_id": "Optional. Left blank, the server gives the result its own id.",
}

#: The fields' help, behind the shared ⓘ (`src/webui/components/help.py`):
#: ``{elem_id: (title, text)}``, registered by the tab builder. Only a
#: state-bearing text stays visible.
FIELD_HELP = {
    "dx_hpo_terms": (
        "HPO terms",
        "Paste HPO ids in any format: one per line, or separated by commas or spaces.\n\n"
        "- Only HP:####### ids are read. Names and punctuation are ignored.\n"
        "- Repeats are sent as typed. The service removes them and says how many.",
    ),
    "dx_candidate_genes": (
        "Candidate genes",
        "The patient's candidate genes, which a gene task model ranks.\n\n"
        "- Enter gene symbols (SCN1A) or gene ids (disgenet:PTEN).\n"
        "- Put one per line, or separate them by commas, spaces or semicolons.\n"
        "- They are sent as typed, repeats included.\n"
        "- The service counts any that name no gene of the model's graph.",
    ),
    "dx_reference_cohort": (
        "Reference cohort",
        "The cohort version whose patients a patients-like-me model ranks here.\n\n"
        "- Listed: every version in the server's registry\n"
        "- Marked and refused: a version mapped against another graph\n"
        "- Also refused: a KG edge set, a generated split, an unreadable version\n"
        "- Under the select: the chosen version's fingerprint and its graph check",
    ),
    "dx_patient_id": (
        "Patient ID (optional)",
        "An identifier for this diagnosis, carried by the result and its exports.\n\n"
        "- Left blank: the server gives the result its own id\n"
        "- So: results from different browsers never share one",
    ),
    "dx_top_k": (
        "Top-K",
        "Number of candidates to return, from 1 to 20.\n\n"
        "Diagnoses, candidate genes or reference patients, as the model in use ranks them.",
    ),
    "dx_explanations": (
        "Explanations",
        "Ask for each candidate's evidence and written explanation.\n\n"
        "Off, the ranking is the same, and Evidence, Explanation and Paths stay empty.",
    ),
}

#: Help that rendered HTML names by key (`help_button`), registered as topics.
TOPIC_HELP = {
    "dx-card": card.CARD_HELP,
    "dx-badge": (
        "Compared with the model in use",
        "The whole served identity is compared part by part, not the checkpoint alone.\n\n"
        "- Files: checkpoint, knowledge graph, manifest, shortest-path table, record\n"
        "- Settings: scoring mode, device, mixing weight eta, paths, phenotype rule\n"
        "- Model in use: the card's last read, never a failed or older one\n"
        "- Result: its CSV and report keep their own identity, whatever the badge",
    ),
    "dx-score": (
        "Score",
        SCORE_NOTE + "\n\n"
        "- Model: the kind above the table names it\n"
        "- Score: a disease task model's head score, 0 or below\n"
        "- Order: the closest to 0 ranks first",
    ),
    "dx-gene-score": (
        "Score",
        GENE_SCORE_RANKS + "\n\n"
        "- Probability: it is not a calibrated one\n"
        "- Model similarity: the gene head's similarity to the patient, −1 to 1\n"
        "- SP similarity: mean phenotype hops, scaled to set bounds, unclipped\n"
        "- Eta (η): the model's own, under This result › Mixing weight eta",
    ),
    "dx-similarity": (
        "Similarity",
        SIMILARITY_RANKS + "\n\n"
        f"- {SIMILARITY_CAVEAT}\n"
        "- The patient tower compares the query's phenotypes with each patient's terms.\n"
        "- The diagnosis, genes and terms shown are what the cohort recorded.\n"
        "- They are context, never an input of this query.",
    ),
    "dx-exports": (
        "Exports",
        "Two downloads of this result, each saying what produced it.\n\n"
        "- CSV: a row per candidate (all Top-K) in its kind's columns, for spreadsheets\n"
        "- Kind: result_kind on every CSV row\n"
        "- Report: each candidate's details (evidence, explanation, recorded terms)\n"
        "- What produced it: the CSV's served_ columns, the report's identity section",
    ),
    "dx-phenotypes": (
        "Phenotypes",
        "How many query phenotypes the graph annotates directly to this disease.\n\n"
        "- It counts the query's phenotypes as used, not as typed.\n"
        "- Only the exact term counts, never ancestors, descendants or gene links.\n"
        "- A 0 does not mean no clinical overlap.\n"
        "- Matching phenotypes, in the candidate's details, names each.",
    ),
    "dx-candidate": (
        "Candidate details",
        "Select a row of Diagnosis results to show that candidate here.\n\n"
        "- Click it, or focus it and press Enter.\n"
        "- Evidence, explanation, matching phenotypes, paths: this result's own.",
    ),
    "dx-run-no-model": (
        "Run Diagnosis is unavailable",
        "Nothing is served on this server, so nothing would compute a diagnosis.\n\n"
        "- The card above says why.\n"
        "- An operator loads a model with the card's \"Load a model…\".\n"
        "- An API client asking now is refused, with the server's reason.",
    ),
    "dx-run-stopping": (
        "Run Diagnosis is unavailable",
        "Inference is stopping: accepted diagnoses finish, new ones are refused.\n\n"
        "- Then nothing is served until a model is loaded again.\n"
        "- An operator loads one with \"Load a model…\" on the card above.\n"
        "- An API client asking now is refused, with the server's reason.",
    ),
    "dx-run-change": (
        "Run Diagnosis is unavailable",
        "A model change is open: finish it (Reload with …) or cancel it (Keep … or Cancel).\n\n"
        "Until then no result is produced here, as what is served may change.",
    ),
    "dx-run-loading": (
        "Run Diagnosis is unavailable",
        "This page started a model load, which reloads the pipeline for every session.\n\n"
        "- Run Diagnosis waits for the load's answer.\n"
        "- No result is produced here while what is served changes.\n"
        "- The card shows the server's state.\n"
        "- Closing the drawer does not stop the load.",
    ),
    "dx-run-no-cohort": (
        "Run Diagnosis is unavailable",
        "This model ranks cohort patients, and no cohort version can be read.\n\n"
        "- The registry may be not set or refused, or hold no version yet.\n"
        "- It may hold only KG edge sets and versions marked \"cannot be read\".\n"
        "- An operator imports a cohort in Research › Test.\n"
        "- Or an operator corrects the server's cohort registry setting.",
    ),
    **{f"dx-banner-{key}": (headline, detail) for key, (headline, detail) in IDENTITY_BANNERS.items()},
}

#: Why Run Diagnosis is off, as visible text under it, in the one form every
#: page says a dimmed control's local cause in (`renderers.need_words`: led by
#: the need, no control's name, never "disabled"; drawn by
#: `renderers.marked_line` with the neutral mark; L7 SW-LK8: it was a ⊖ and
#: "Disabled until …" of this page's own). Nothing served (no model, or a stop
#: letting the accepted diagnoses finish) is said once, by the card's state and
#: the backend's lead above it; this refers to it.
RUN_LABEL = "Run Diagnosis"
RUN_DISABLED_NO_MODEL = _need_words("load a model first")
RUN_DISABLED_CHANGE = _need_words("finish or cancel the model change first")
RUN_DISABLED_LOADING = _need_words("the model load started here has not answered yet")
#: A patients-like-me model with no reference cohort version to rank: the
#: lead, then the cause in its own words with who can act (skill §9). The
#: registry setting's own words (its name, a command's flag) are an
#: operator's and never reach this reason (Research › Test shows them).
RUN_DISABLED_NO_COHORT = _need_words("no reference cohort version can be read")
NO_COHORT_REGISTRY = ("The server's results registry is not set or cannot be used. An operator "
                      "sets it in Settings\u00a0› Data locations.")
NO_COHORT_VERSION = ("The registry holds no cohort version yet. An operator imports one in "
                     "Research\u00a0› Test.")
NO_READABLE_VERSION = ("Every version in the registry is marked “cannot be read”. An operator "
                       "imports a readable one in Research\u00a0› Test.")
#: Every readable version is of another kind (a KG edge set, a generated
#: split), which is not a supplied cohort (`diagnosis_card.NOT_A_COHORT`).
NO_REFERENCE_VERSION = ("No listed version is a reference cohort: each is a KG edge set, a "
                        "generated split or cannot be read. An operator imports a cohort in "
                        "Research\u00a0› Test.")

#: The change drawer's link to Research › Models: each "›" held to the word
#: before it by a no-break space, so it is never alone on a line.
COMPARE_LINK = "Compare in Research\u00a0› Models\u00a0›"

#: The change drawer's Task control (K6 decision 14): (label, the shared list's
#: view filter). The clinical preset lists task models only, so "All tasks" is
#: every one of them; the task names are `caveats.TASK_NAMES`.
TASK_VIEWS = (("All tasks", "all"), *((name, task) for task, name in TASK_NAMES.items()))

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


def task_view(task: Any) -> Dict[str, str]:
    """The shared list's view for the drawer's Task control (the K6d1 contract:
    ``{"task": <filter>}``); a value the control does not offer is "all"."""
    return {"task": task if task in {value for _label, value in TASK_VIEWS} else "all"}


def results_header_html(result: Optional[Mapping[str, Any]]) -> str:
    """"Diagnosis results" and, at its right, how many candidates and when it
    was produced; nothing at its right before a result (the panel's body says
    "No result yet", once). How long the inference took is not the
    clinician's: the report keeps it ("Inference time (ms)"; R-23)."""
    if not isinstance(result, Mapping):
        right = ""
    else:
        n = len(result.get("candidates") or [])
        right = (f'<span class="shep-panel-right"><b>{n} candidate{"s" if n != 1 else ""}</b>'
                 f' · produced {_e(card.hhmmss(_produced_at(result)))}</span>')
    return f'<div class="shep-panel-hd"><h2>Diagnosis results</h2>{right}</div>'


def empty_results_html(notice: Optional[str] = None) -> str:
    """The results panel with no result: "No result yet" (said once on the
    page) and what to do, or `notice` (a refusal, an error, inputs changed
    since the last run). Whether a model is served is the card's to say."""
    if notice:
        return f'<div class="shep-dx-notice">{notice}</div>'
    return ('<div class="shep-dx-empty"><p class="shep-dx-empty-h">No result yet</p>'
            '<p>Enter the HPO terms, check the model in use, then Run Diagnosis.</p>'
            '<p class="shep-note">Each result says what produced it, compared with the model in '
            'use.</p></div>')


def compare_with_served(result: Mapping[str, Any], state: Mapping[str, Any]) -> Dict[str, Any]:
    """The result's served identity against the card's last read, part by part
    (`_identity_rows`, the one reading of an identity): ``equal``, ``differs``
    (with the parts), ``nothing`` (nothing is served now: the result is
    compared with nothing, never said to differ in every part) or ``unknown``
    (with one reason). Whatever it says, the result keeps its own identity:
    nothing here relabels it."""
    kind, identity = _identity_of(result)
    total = len(_IDENTITY_PARTS)
    keep = "This result keeps its own identity. Its exports use it."
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
    if _served_now(state)[0] == IDENTITY_NO_MODEL:
        return {"state": "nothing", "total": total, "differ": [],
                "reason": ("Nothing is served now (the card says why), so this result is "
                           f"compared with nothing. {keep}")}
    mine = _identity_rows(kind, identity)
    now = _identity_rows(*_served_now(state))
    differ = [part for (part, s1, v1), (_, s2, v2) in zip(mine, now, strict=True)
              if (s1, v1) != (s2, v2)]
    return {"state": "differs" if differ else "equal", "total": total, "differ": differ,
            "reason": None}


def _result_model(result: Mapping[str, Any]) -> str:
    """The model a result names as its own (its served identity's checkpoint
    fingerprint, 12 hex), in the badge's words."""
    _kind, identity = _identity_of(result)
    sha = _field(_field(identity, "checkpoint"), "sha256")
    return f"model {_short_hex(sha)}" if _is_digest(sha) else "a model it does not name"


def _parts_differ(n: int) -> str:
    return f"{n} part differs" if n == 1 else f"{n} parts differ"


def badge_html(result: Optional[Mapping[str, Any]], state: Mapping[str, Any]) -> str:
    """The results badge, in the card's noun; its › opens "This result".
    Its state is the shared status mark (`renderers.status_mark`, the WebUI
    skill's §6): ● produced by the model in use, ▲ by an earlier one, ○ not
    compared or not checkable yet; the page draws no mark of its own."""
    if not isinstance(result, Mapping):
        return ""
    compared = compare_with_served(result, state)
    help_html = help_button("dx-badge", "Compared with the model in use")
    if compared["state"] == "equal":
        return (f'<div class="shep-dx-badge-row"><button type="button" class="shep-dx-badge '
                f'shep-dx-badge-ok" data-key="result">'
                f'{_status_mark("ok", "Produced by the model in use")} '
                f'<span aria-hidden="true">›</span></button>{help_html}</div>')
    if compared["state"] == "differs":
        n = len(compared["differ"])
        words = ('Produced by an earlier model or pipeline — <span class="shep-dx-u">'
                 f"{_e(_parts_differ(n))} ›</span>")
        return (f'<div class="shep-dx-badge-row"><button type="button" class="shep-dx-badge '
                f'shep-dx-badge-amber" data-key="result">'
                f'{_status_mark("warn", words, words_html=True)}</button>{help_html}</div>')
    if compared["state"] == "nothing":
        # Nothing is served now (after a stop, say): the result is named by
        # its own model, never relabelled, and compared with nothing.
        return (f'<div class="shep-dx-badge-row"><button type="button" class="shep-dx-badge '
                f'shep-dx-badge-grey" data-key="result">'
                f'{_status_mark("neutral", f"Produced by {_result_model(result)} · not compared")}'
                f' <span aria-hidden="true">›</span></button>{help_html}</div>')
    return (f'<div class="shep-dx-badge-row"><button type="button" class="shep-dx-badge '
            f'shep-dx-badge-grey" data-key="result">'
            f'{_status_mark("neutral", "Cannot check against the model in use yet")}'
            f'</button>{help_html}</div>'
            f'<p class="shep-dx-badge-why">{_e(compared["reason"])}</p>')


_BANNER_KEYS = {headline: key for key, (headline, _detail) in IDENTITY_BANNERS.items()}


def synthetic_banner_shown(result: Optional[Mapping[str, Any]], state: Mapping[str, Any]) -> bool:
    """Whether the results show a synthetic banner **from the identity the card
    shows** (`compare_with_served`: the same served identity, part by part):
    then the card leaves out its "synthetic data" chip, said once by the
    banner (R-26). A result from another model, or one not compared, keeps
    the chip, which then speaks of what is served now."""
    if not isinstance(result, Mapping):
        return False
    kind, identity = _identity_of(result)
    if IDENTITY_BANNERS["synthetic"] not in _identity_banners(kind, identity):
        return False
    return compare_with_served(result, state)["state"] == "equal"


def banners_html(result: Mapping[str, Any]) -> str:
    """The result's identity banners: each headline visible, its detail behind
    the ⓘ (the banners' own words, `renderers.IDENTITY_BANNERS`)."""
    kind, identity = _identity_of(result)
    out = []
    for headline, _detail in _identity_banners(kind, identity):
        key = _BANNER_KEYS.get(headline, "")
        tone = "red" if key == "not_reported" else "amber"
        # The shared mark (B72-N7): ✕ for an error, ▲ for a condition to watch.
        mark = _status_mark("error" if tone == "red" else "warn", f"<b>{_e(headline)}</b>",
                            words_html=True)
        out.append(f'<div class="shep-dx-banner shep-dx-banner-{tone}" role="note">{mark}'
                   f'{help_button(f"dx-banner-{key}", headline) if key else ""}</div>')
    return "".join(out)


def used_terms(result: Mapping[str, Any]) -> Optional[List[str]]:
    """The query's phenotypes as the result used them: the server's account
    (``phenotype_input.used``), else the distinct ids the page sent (an older
    response); None when neither is known."""
    account = phenotype_account(result)
    if account is not None:
        return account[0]
    sent = result.get("_query_phenotypes")
    return list(dict.fromkeys(str(t) for t in sent)) if isinstance(sent, list) else None


def phenotype_match(result: Mapping[str, Any], candidate: Mapping[str, Any]
                    ) -> Optional[Tuple[int, int]]:
    """**k of N** (R-1d), computed here from the query and the candidate's
    ``matching_phenotypes``: k the query's phenotypes as used that the graph
    annotates directly to this disease, N the query's phenotypes as used
    (`used_terms`); None when the result says neither (no API change)."""
    used = used_terms(result)
    if used is None:
        return None
    found = {str(p) for p in candidate.get("matching_phenotypes") or []}
    return len([t for t in used if t in found]), len(used)


#: The k/N cell's words, for its title and the report.
def _match_words(match: Tuple[int, int]) -> str:
    return f"{match[0]} of the {match[1]} phenotypes used are annotated to this disease"


def _names(result: Optional[Mapping[str, Any]]) -> Mapping[str, str]:
    """The served graph's names the page read for this result (`_names`,
    `served_status.node_names`): bound to the result's own graph digest."""
    names = result.get("_names") if isinstance(result, Mapping) else None
    return names if isinstance(names, Mapping) else {}


def _name_for(names: Mapping[str, str], term: Any) -> Optional[str]:
    """A term's or a path node's name: by its own spelling, else by its
    display id ("hpo:HP:0001250" read as "HP:0001250")."""
    text = str(term)
    return names.get(text) or names.get(_node_display_name(text))


def term_html(term: Any, names: Mapping[str, str]) -> str:
    """A term **name first, its id muted** (R-3, R-5): "Seizure HP:0001250";
    the id alone, as code, when the graph gave no name. Escaped."""
    shown = _node_display_name(str(term))
    name = _name_for(names, term)
    if not name or name == shown:
        return f'<code>{_e(shown)}</code>'
    return f'{_e(name)} <span class="shep-dx-id shep-mono">{_e(shown)}</span>'


def _md_term(term: Any, names: Mapping[str, str]) -> str:
    """The report's form of `term_html`: "Seizure (HP:0001250)", escaped."""
    shown = _node_display_name(str(term))
    name = _name_for(names, term)
    if not name or name == shown:
        return _md_text(shown)
    return f"{_md_text(name)} ({_md_text(shown)})"


def _wanted_names(result: Mapping[str, Any]) -> List[str]:
    """Every term and path node a result names, for `served_status.node_names`:
    the query's phenotypes, the ones used, each candidate's matching
    phenotypes and path nodes, and a reference patient's recorded terms."""
    keys: List[str] = [str(t) for t in result.get("_query_phenotypes") or []]
    account = result.get("phenotype_input")
    if isinstance(account, Mapping):
        keys += [str(t) for t in account.get("used") or []]
    for c in result.get("candidates") or []:
        if not isinstance(c, Mapping):
            continue
        keys += [str(t) for t in c.get("matching_phenotypes") or []]
        keys += [str(t) for t in c.get("phenotypes") or []]
        keys += [str(n) for path in _paths(c) for n in path]
    return list(dict.fromkeys(keys))


def result_names(result: Mapping[str, Any]) -> Dict[str, str]:
    """The served graph's names of the terms `result` names, while the graph
    served is the result's own (its ``served_identity.graph.kg_sha256``);
    ``{}`` otherwise, and the page shows ids alone."""
    _kind, identity = _identity_of(result)
    digest = _field(_field(identity, "graph"), "kg_sha256")
    return served_status.node_names(digest, _wanted_names(result))


def candidates_table_html(candidates: List[Mapping[str, Any]], selected: Any,
                          kind: Optional[str], result: Optional[Mapping[str, Any]] = None
                          ) -> str:
    """The ranked candidates in the kind's columns; a row is selected by click,
    or by Enter or Space when focused. Every value is escaped; a long name is
    clamped to two lines with the full name in its title. A disease task model
    shows its head's score and **Phenotypes k/N** (`phenotype_match`, R-1d)
    from `result`'s query."""
    if kind is None:
        return f'<div class="shep-reason">{_e(KIND_NOT_REPORTED)}</div>'
    if not candidates:
        return '<div class="shep-reason">No candidates found.</div>'
    if kind in TASK_ROW_KINDS:
        return _task_table_html(candidates, selected, kind)
    rows = []
    selected_class = ' class="shep-on" aria-current="true"'
    for c in candidates:
        rank = c.get("rank")
        on = rank == selected
        name = _e(c.get("disease_name"))
        match = phenotype_match(result, c) if isinstance(result, Mapping) else None
        matched = (f'<td class="shep-num" title="{_e(_match_words(match))}">'
                   f'{match[0]}\u2009/\u2009{match[1]}</td>' if match else
                   '<td class="shep-num"><span class="shep-muted">not reported</span></td>')
        cells = (f'<td class="shep-num">{_e(rank)}</td><td class="shep-dx-dis">'
                 f'<span class="shep-dx-clamp" title="{name}">{name}</span></td>'
                 f'<td class="shep-num">{_num(_score(c, kind))}</td>{matched}')
        label = c.get("evidence_label") or "—"
        cells += f'<td class="shep-dx-evcell">{_evidence_mark(label)}</td>'
        # A plain table's selected row is `aria-current` (the skill's §8).
        rows.append(f'<tr data-key="{_e(rank)}" tabindex="0"'
                    f'{selected_class if on else ""}>{cells}</tr>')
    head = ('<th scope="col" class="shep-num">Rank</th><th scope="col">Disease</th>'
            f'<th scope="col" class="shep-num">Score{help_button("dx-score", "Score")}</th>'
            f'<th scope="col" class="shep-num">Phenotypes'
            f'{help_button("dx-phenotypes", "Phenotypes")}</th>'
            '<th scope="col">Evidence</th>')
    return ('<div class="shep-tbl-wrap shep-dx-tbl-wrap"><table class="shep-tbl shep-dx-tbl" '
            f'aria-label="Ranked candidates"><thead><tr>{head}</tr></thead><tbody>'
            f'{"".join(rows)}</tbody></table></div>')


def _not_recorded() -> str:
    return '<span class="shep-muted">not recorded</span>'


def _diagnosis_html(candidate: Mapping[str, Any]) -> str:
    """A reference patient's recorded diagnosis, escaped: its name, else its
    id, else "not recorded" (skill §6: an absent value keeps its own word)."""
    if candidate.get("disease_name"):
        return _e(candidate["disease_name"])
    if candidate.get("disease_id"):
        return f'<code>{_e(_clean_disease_id(candidate["disease_id"]))}</code>'
    return _not_recorded()


def _task_table_html(candidates: List[Mapping[str, Any]], selected: Any, kind: str) -> str:
    """The ranked rows of a task kind, every value escaped: a gene task model's
    candidate genes (Rank, Gene with its id in the title, Score, Model
    similarity, SP similarity) or a patients-like-me model's reference
    patients (Rank, Research ID, Similarity, Diagnosis, Causal genes). The
    selected row is `shep-on` and `aria-current` (a plain table's rows take
    no `aria-selected`, skill §8)."""
    rows = []
    for c in candidates:
        rank = c.get("rank")
        on = rank == selected
        if kind == GENE_TASK:
            name = _e(c.get("gene_name"))
            cells = (f'<td class="shep-num">{_e(rank)}</td><td class="shep-dx-dis">'
                     f'<span class="shep-dx-clamp" title="{_e(c.get("gene_id"))}">{name}</span></td>'
                     f'<td class="shep-num">{_num(c.get("score"))}</td>'
                     f'<td class="shep-num">{_num(c.get("embsim"))}</td>'
                     f'<td class="shep-num">{_num(c.get("sp_similarity"))}</td>')
        else:
            genes = [str(g) for g in c.get("causal_genes") or [] if str(g).strip()]
            cells = (f'<td class="shep-num">{_e(rank)}</td><td>{_e(c.get("case_id"))}</td>'
                     f'<td class="shep-num">{_num(c.get("similarity"))}</td>'
                     f'<td class="shep-dx-dis">{_diagnosis_html(c)}</td>'
                     f'<td class="shep-dx-wrap">{_e(", ".join(genes)) if genes else _not_recorded()}'
                     '</td>')
        current = ' class="shep-on" aria-current="true"' if on else ""
        rows.append(f'<tr data-key="{_e(rank)}" tabindex="0"{current}>{cells}</tr>')
    if kind == GENE_TASK:
        head = ('<th scope="col" class="shep-num">Rank</th><th scope="col">Gene</th>'
                f'<th scope="col" class="shep-num">Score{help_button("dx-gene-score", "Score")}</th>'
                '<th scope="col" class="shep-num">Model similarity</th>'
                '<th scope="col" class="shep-num">SP similarity</th>')
    else:
        head = ('<th scope="col" class="shep-num">Rank</th><th scope="col">Research ID</th>'
                f'<th scope="col" class="shep-num">Similarity'
                f'{help_button("dx-similarity", "Similarity")}</th>'
                '<th scope="col">Diagnosis</th><th scope="col">Causal genes</th>')
    return ('<div class="shep-tbl-wrap shep-dx-tbl-wrap"><table class="shep-tbl shep-dx-tbl" '
            f'aria-label="Ranked candidates"><thead><tr>{head}</tr></thead><tbody>'
            f'{"".join(rows)}</tbody></table></div>')


def reference_line_html(result: Mapping[str, Any]) -> str:
    """A patients-like-me result's reference cohort version, as the result
    names it: "Reference cohort <label> · version <12 hex>" (the full digests
    are in the identity drawer's "This result in full"); "" for another kind."""
    reference = result.get("reference_cohort")
    if _result_kind(result) != PATIENTS_LIKE_ME or not isinstance(reference, Mapping):
        return ""
    return (f'<p class="shep-dx-kind shep-dx-ref">Reference cohort <b>{_e(reference.get("label"))}'
            f'</b> · version '
            + _fingerprint_value(reference.get("version_sha256"), what="version fingerprint",
                                 copy=False) + '</p>')


def answer_mark(result: Mapping[str, Any]) -> str:
    """The inputs generation a result answered (``_inputs_gen``, `diagnose`)
    as the results body that draws it carries it: a hidden element with
    ``data-shep-gen``, which the page's gate reads (`ANSWER_GATE_CSS`); ""
    for a result no run of the page made."""
    gen = result.get("_inputs_gen")
    if not isinstance(gen, int) or isinstance(gen, bool):
        return ""
    return f'<span class="shep-dx-answered" data-shep-gen="{gen}" hidden></span>'


def results_body_html(result: Mapping[str, Any], selected: Any) -> str:
    """Under the badge: the answer's mark (`answer_mark`), the banners, the
    result's warnings, the kind's label (what the Score column holds,
    visible text), a patients-like-me result's reference cohort version, and
    the table."""
    warnings = result.get("warnings") or []
    # One warning a line, each with its own mark (the owner's decision of
    # 2026-11-23: never two of them glued with a semicolon).
    note = "".join(f'<p class="shep-dx-warnings">{_status_mark("warn", str(w))}</p>'
                   for w in warnings)
    kind = _result_kind(result)
    label = (f'<p class="shep-dx-kind">{_e(kind_label(result))}</p>'
             if kind is not None else "")
    return answer_mark(result) + banners_html(result) + note + phenotypes_used_html(result) + \
        label + reference_line_html(result) + \
        candidates_table_html(result.get("candidates") or [], selected, kind, result)


def phenotypes_used_html(result: Mapping[str, Any]) -> str:
    """The query's phenotypes as the result used them (`phenotype_account`):
    "Used (n)", each **name first, its id muted** (`term_html`, R-5), and,
    when any was not, "Not used (n)" with each one's reason; "" for a result
    with no account."""
    account = phenotype_account(result)
    if account is None:
        return ""
    used, left = account
    names = _names(result)
    rows = [(f"Used ({len(used)})",
             ", ".join(term_html(term, names) for term in used) or "none")]
    if left:
        rows.append((f"Not used ({len(left)})", ", ".join(
            f"{_e(entry)} <span class=\"shep-dx-why\">({_e(reason)})</span>"
            for entry, reason in left)))
    return _summary_list(rows, classes="shep-dx-used")


# ----------------------------------------------------------------- details
def _candidate(result: Optional[Mapping[str, Any]], rank: Any) -> Optional[Mapping[str, Any]]:
    if not isinstance(result, Mapping):
        return None
    return next((c for c in result.get("candidates") or [] if c.get("rank") == rank), None)


def _paths(candidate: Mapping[str, Any]) -> List[List[str]]:
    pkg = candidate.get("evidence_package") or {}
    return [p for p in pkg.get("direct_paths") or [] if isinstance(p, list)]


def segment_choices(candidate: Optional[Mapping[str, Any]]) -> List[Tuple[str, str]]:
    """The segments' labels, with the counts this candidate has."""
    if candidate is None:
        return list(SEGMENTS)
    counts = {"phenotypes": len(candidate.get("matching_phenotypes") or []),
              "paths": len(_paths(candidate))}
    return [(f"{label} {counts[value]}" if value in counts else label, value)
            for label, value in SEGMENTS]


def details_header_html(candidate: Optional[Mapping[str, Any]],
                        kind: Optional[str] = None) -> str:
    """The selected candidate as **#rank, name and id only** (R-1b): its
    scores are its table row's, beside it, said once; a reference patient's
    research id is its name, "Reference patient" its kind."""
    if candidate is None:
        return '<div class="shep-panel-hd"><h2>Candidate details</h2></div>'
    if kind == GENE_TASK:
        name, sub = candidate.get("gene_name"), f'<code>{_e(candidate.get("gene_id"))}</code>'
    elif kind == PATIENTS_LIKE_ME:
        name, sub = candidate.get("case_id"), "Reference patient"
    else:
        name = candidate.get("disease_name")
        sub = f'<code>{_e(_clean_disease_id(candidate.get("disease_id")))}</code>'
    return (f'<div class="shep-panel-hd shep-dx-det-hd"><h2><span class="shep-dx-rank">'
            f'#{_e(candidate.get("rank"))}</span> {_e(name)}</h2>'
            f'<p class="shep-dx-det-sub">{sub}</p></div>')


def details_empty_html() -> str:
    return ('<div class="shep-dx-empty"><p>After a diagnosis, the selected candidate\'s evidence, '
            'explanation, matching phenotypes and paths appear here.</p></div>')


def _hops(value: Any) -> str:
    return f"{value} hop{'s' if value != 1 else ''}" if isinstance(value, int) \
        and not isinstance(value, bool) else "not stated"


def evidence_html(candidate: Mapping[str, Any], explanations: bool = True) -> str:
    """The candidate's evidence package as an explanation (the skill's §4,
    R-3): the answer first (its label, with its mark), then its facts as
    one summary list, each said once (NEW-diagnosis-2: the shortest path's
    length is its row's), with no retired pipeline's mode name; the server's
    narrative folded under "Written summary ›". Evidence found by no path
    has no facts to list: its narrative is the answer's sentence."""
    pkg = candidate.get("evidence_package")
    if not pkg:
        why = ("Explanations were off for this run (Options), so no evidence package was asked "
               "for." if not explanations else "No evidence package was returned for this "
               "candidate.")
        return f'<p class="shep-muted">{_e(why)}</p>'
    label = candidate.get("evidence_label") or pkg.get("evidence_label") or "—"
    parts = [f'<p class="shep-dx-ev-label">{_evidence_mark(label, strong=True)}</p>']
    summary = pkg.get("summary")
    genes = candidate.get("supporting_genes") or []
    rows: List[Tuple[str, str]] = []
    if pkg.get("mode") == "direct_path":
        n = len(pkg.get("direct_paths") or [])
        rows += [("Shortest path", _e(_hops(pkg.get("min_path_length")))),
                 ("Direct paths", f'{n} · <button type="button" class="shep-dx-link" '
                  'data-key="seg:paths">listed under Paths ›</button>')]
    elif summary:
        parts.append(f'<p class="shep-dx-ev-why">{_e(summary)}</p>')
        summary = None
    if genes:
        rows.append(("Supporting genes", '<span class="shep-dx-genes">' + "".join(
            f'<span class="shep-dx-gene shep-mono">{_e(g)}</span>' for g in genes) + "</span>"))
    if rows:
        parts.append(_summary_list(rows, classes="shep-dx-ev"))
    if summary:
        parts.append('<details class="shep-more shep-dx-ev-more"><summary>Written summary ›'
                     f'</summary><p>{_e(summary)}</p></details>')
    return "".join(parts)


#: **The Explanation view** (R-1b): the pipeline's written explanation opens
#: with the candidate's name and its score ("Score: -2.844 (a ranking score,
#: …)"), so shown as the view it put the selected row's score on screen
#: twice. The view shows what the explanation covers from the result the page
#: already projects, then the explanation, whole as the API wrote it, under
#: this fold: the Markdown view under the HTML one (`#shep-dx-explanation`),
#: drawn while the fold is open (`PAGE_CSS`).
EXPLANATION_FOLD = "Written explanation ›"
#: The score, referred to by its relation to the table (its number is said
#: once, there), with what it is (`SCORE_NOTE`).
EXPLANATION_SCORE = ("In the table: it ranks this result's candidates and is not a calibrated "
                     "disease probability.")


def explanation_html(candidate: Mapping[str, Any], result: Optional[Mapping[str, Any]] = None,
                     explanations: bool = True) -> str:
    """The Explanation view of a disease candidate (R-1b; the skill's §4: an
    explanation's facts as key-value rows, its narrative folded): what the
    pipeline's written explanation sets out, from the structured result (the
    score by its relation to the table, the matching phenotypes and the paths
    each listed in its own view, the supporting genes), then the written
    explanation itself under "Written explanation ›" (`EXPLANATION_FOLD`; its
    text is the Markdown view, unchanged and uncut). With no explanation, why,
    in its own words."""
    if not candidate.get("explanation"):
        why = ("No explanation was returned for this candidate." if explanations
               else "Explanations were off for this run (Options).")
        return f'<p class="shep-muted">{_e(why)}</p>'
    rows: List[Tuple[str, str]] = [("Score", _e(EXPLANATION_SCORE)),
                                   ("Matching phenotypes",
                                    '<button type="button" class="shep-dx-link" '
                                    'data-key="seg:phenotypes">listed under Matching phenotypes ›'
                                    '</button>')]
    genes = candidate.get("supporting_genes") or []
    if genes:
        rows.append(("Supporting genes", '<span class="shep-dx-genes">' + "".join(
            f'<span class="shep-dx-gene shep-mono">{_e(g)}</span>' for g in genes) + "</span>"))
    paths = _paths(candidate)
    if paths:
        rows.append(("Direct paths", f'{len(paths)} · <button type="button" class="shep-dx-link" '
                     'data-key="seg:paths">listed under Paths ›</button>'))
    return (_summary_list(rows, classes="shep-dx-ev")
            + f'<details class="shep-more shep-dx-written"><summary>{_e(EXPLANATION_FOLD)}'
            '</summary></details>')


def phenotypes_html(candidate: Mapping[str, Any],
                    result: Optional[Mapping[str, Any]] = None) -> str:
    """The query's phenotypes as used, **split by whether the graph annotates
    each directly to this disease**, with the counts, each name first and its
    id muted (R-3): "Annotated to this disease (k)", "Not annotated (n)"
    (`used_terms`; with no account of the query, the matches alone). Then the
    API's definition of a match."""
    names = _names(result)
    found = [str(p) for p in candidate.get("matching_phenotypes") or []]
    used = used_terms(result) if isinstance(result, Mapping) else None
    rest = [t for t in used if t not in set(found)] if used is not None else None

    def listed(terms: List[str]) -> str:
        return ('<ul class="shep-dx-list">'
                + "".join(f"<li>{term_html(t, names)}</li>" for t in terms) + "</ul>")

    if found:
        body = (f'<p class="shep-dx-sub-h"><b>Annotated to this disease ({len(found)})</b></p>'
                + listed(found))
    else:
        body = ('<p class="shep-muted">None of the query\'s phenotypes is annotated directly to '
                'this disease.</p>')
    if rest:
        body += (f'<p class="shep-dx-sub-h"><b>Not annotated ({len(rest)})</b></p>'
                 + listed(rest))
    return body + f'<p class="shep-note">{_e(MATCHING_NOTE)}</p>'


#: How many paths are shown at once; the rest one click away (R-3).
PATHS_SHOWN = 3


def _path_html(path: List[str], names: Mapping[str, str]) -> str:
    """One path as its steps' chips, each name first and its id muted."""
    return ('<li class="shep-dx-path">' + ' <span class="shep-dx-arrow" aria-label="to">→</span> '
            .join(f'<span class="shep-dx-step">{term_html(node, names)}</span>' for node in path)
            + "</li>")


def paths_html(candidate: Mapping[str, Any],
               result: Optional[Mapping[str, Any]] = None) -> str:
    """The candidate's direct paths, **shortest first**: the `PATHS_SHOWN`
    shortest as chips of named steps (each name first, its id muted), the rest
    under "More paths (n) ›" (R-3). The order among paths of one length is
    the server's."""
    paths = sorted(_paths(candidate), key=len)
    if not paths:
        return '<p class="shep-muted">No path is listed for this candidate.</p>'
    names = _names(result)
    shown = "".join(_path_html(path, names) for path in paths[:PATHS_SHOWN])
    html_ = f'<ol class="shep-dx-paths">{shown}</ol>'
    rest = paths[PATHS_SHOWN:]
    if rest:
        more = "".join(_path_html(path, names) for path in rest)
        html_ += (f'<details class="shep-more shep-dx-paths-more"><summary>More paths '
                  f'({len(rest)}) ›</summary><ol class="shep-dx-paths" start="{PATHS_SHOWN + 1}">'
                  f'{more}</ol></details>')
    return html_


def task_details_html(candidate: Mapping[str, Any], kind: str,
                      result: Mapping[str, Any]) -> str:
    """A gene's or a reference patient's values beyond its table row's, as
    one key-value list (`renderers.summary_list`), every value escaped: a
    gene's mean hop distance and the model's SP bounds; a reference patient's
    recorded diagnosis, causal genes and every recorded HPO term (name first,
    id muted: `term_html`), with the note that they are reference context,
    never an input."""
    if kind == GENE_TASK:
        # Its score, model and SP similarities are its table row's, said once
        # there (R-1b); here what the row does not show.
        bounds = result.get("sp_bounds")
        bounds = bounds if isinstance(bounds, (list, tuple)) and len(bounds) == 2 else (None, None)
        rows = [("Mean hop distance", _num(candidate.get("mean_distance"))),
                ("SP bounds (the model's own)", f"{_num(bounds[0])} to {_num(bounds[1])}")]
        return _summary_list(rows)
    genes = [str(g) for g in candidate.get("causal_genes") or [] if str(g).strip()]
    terms = [str(t) for t in candidate.get("phenotypes") or [] if str(t).strip()]
    rows = [("Recorded diagnosis", _diagnosis_html(candidate)
             + (f' <code>{_e(_clean_disease_id(candidate.get("disease_id")))}</code>'
                if candidate.get("disease_name") and candidate.get("disease_id") else "")),
            ("Causal genes", "".join(f'<span class="shep-dx-gene shep-mono">{_e(g)}</span> ' for g in genes)
             .strip() or _not_recorded()),
            (f"Recorded HPO terms ({len(terms)})",
             ", ".join(term_html(t, _names(result)) for t in terms) or _not_recorded())]
    return (_summary_list(rows)
            + f'<p class="shep-note">{_e(REFERENCE_CONTEXT_NOTE)}</p>')


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
    kind = _result_kind(result)
    # The candidate its disclosures are of (`renderers.keep_as`), so a
    # "More paths" or "Written summary" opened for one candidate is never
    # drawn open for the next one chosen.
    named = _keep_as(" ".join(str(candidate.get(k)) for k in (
        "rank", "disease_id", "gene_id", "case_id") if candidate.get(k) is not None))
    if kind in TASK_ROW_KINDS:
        # No segment is offered: a gene or a reference patient has no
        # evidence, explanation or path, only its values.
        body = f"<div{named}>{task_details_html(candidate, kind, result)}</div>"
        return {"details_hd": details_header_html(candidate, kind),
                "details_seg": view_update(False, "shep-seg"),
                "details_html": view_update(True, *_HTML_VIEW, value=body),
                "details_md": view_update(False, *_MD_VIEW),
                "details_ft": details_footer_html(result, rank)}
    segment = segment if segment in _SEGMENT_VALUES else "evidence"
    explanations = bool(result.get("_explanations", True))
    html_: Optional[str] = None
    if segment == "evidence":
        html_ = evidence_html(candidate, explanations)
    elif segment == "phenotypes":
        html_ = phenotypes_html(candidate, result)
    elif segment == "paths":
        html_ = paths_html(candidate, result)
    else:
        html_ = explanation_html(candidate, result, explanations)
    explanation = candidate.get("explanation")
    # The written explanation, whole as the API wrote it, is the Markdown view
    # under the Explanation view's fold (`explanation_html`).
    written = segment == "explanation" and bool(explanation)
    return {
        "details_hd": details_header_html(candidate, _result_kind(result)),
        "details_seg": view_update(True, "shep-seg", value=segment,
                                   choices=segment_choices(candidate)),
        "details_html": view_update(True, *_HTML_VIEW, value=f"<div{named}>{html_}</div>"),
        "details_md": view_update(written, *_MD_VIEW, value=explanation),
        "details_ft": details_footer_html(result, rank),
    }


# ----------------------------------------------------------------- identity drawer
def _part_html(state: str, value: _Value) -> str:
    """One part of an identity as a details value: a verified part by the
    shared mark (`renderers.status_mark`), then its value (`_value_html`,
    the pieces the report draws as Markdown), which says every other state
    in its own words."""
    mark = _status_mark("ok", "verified") + " " if state.startswith(_VERIFIED_MARK) else ""
    return mark + _value_html(value)


def _part_key(part: str) -> str:
    """A part's row label, plain: its file names without code marks."""
    return part.replace("`", "")


def result_identity_html(result: Optional[Mapping[str, Any]], state: Mapping[str, Any]) -> str:
    """The identity drawer's "This result": what the result carries, its own
    served identity, **apart from what is served now**
    (`compare_with_served`), never relabelled by a model change or a stop.
    Answer first (produced by the model in use, by an earlier model or
    pipeline, not compared while nothing is served, or why it cannot be
    checked), then the result's identity as one summary list
    (`renderers.summary_list`, a part a row), **each value said once**
    (the skill's §1.2): a part that differs carries what is in use now under
    its own value ("In use now: …"), the difference beside it, never a second
    copy of the result's value in a table of its own. Then the claim boundary
    (the medical-device line is the results footer's, R-26)."""
    if not isinstance(result, Mapping):
        return '<p class="shep-muted">Run a diagnosis to see what produced its result.</p>'
    kind, identity = _identity_of(result)
    compared = compare_with_served(result, state)
    mine = _identity_rows(kind, identity)
    when = card.hhmmss(state.get("good_at"))
    parts: List[str] = []
    now: Dict[str, Tuple[str, _Value]] = {}
    if compared["state"] == "differs":
        n = len(compared["differ"])
        now = {part: (state_, value) for part, state_, value in _identity_rows(*_served_now(state))
               if part in compared["differ"]}
        parts.append(f'<p class="shep-dx-lead"><b>This result came from an earlier model or '
                     f'pipeline:</b> {n} of {compared["total"]} parts '
                     f'{"differs" if n == 1 else "differ"} from what is in use now (status '
                     f'read {_e(when)}). Each says what is in use now under its own value. The '
                     'result and its CSV and report keep their own identity.</p>')
    elif compared["state"] == "equal":
        parts.append(f'<p class="shep-dx-lead"><b>Produced by the model in use:</b> all '
                     f'{compared["total"]} parts are the same as what is in use now (status '
                     f'read {_e(when)}).</p>')
    elif compared["state"] == "nothing":
        parts.append(f'<p class="shep-dx-lead"><b>Not compared:</b> {_e(compared["reason"])}</p>')
    else:
        parts.append(f'<p class="shep-dx-lead"><b>Cannot check against the model in use yet:'
                     f'</b> {_e(compared["reason"])}</p>')
    parts.append('<p class="shep-dx-sub-h"><b>What produced this result</b></p>')
    parts.append(_summary_list((_part_key(part), _part_html(state_, value) + (
        f'<span class="shep-dx-now"><span class="shep-dx-now-k">In use now:</span> '
        f'{_part_html(*now[part])}</span>' if part in now else ""))
        for part, state_, value in mine))
    return ('<div class="shep-dx-result-id">' + "".join(parts)
            + _boundary_html(_claim_boundary_lines(kind, identity)) + '</div>')


def _boundary_html(sentences: Iterable[str]) -> str:
    """The claim boundary as the identity drawer shows it. Each sentence is
    the report's, word for word, except that its ✓ is drawn as the mark the
    drawer shows a verified part by (`_part_html`, the card's served facts):
    the sentence names the mark the reader sees. **No medical-device line**:
    the results footer says it, in view beside the drawer (R-26); the report
    keeps it under its own boundary."""
    mark = _status_mark("ok", "verified")
    shown = "".join(f"<p>{_e(sentence).replace(_VERIFIED_MARK, mark)}</p>"
                    for sentence in sentences)
    return f'<div class="shep-dx-boundary">{shown}</div>' if shown else ""


def result_digests_html(result: Optional[Mapping[str, Any]]) -> str:
    """"This result in full": the full digests the result carries, each a
    wide row of one summary list with its Copy (`renderers.summary_list`,
    `renderers.fingerprint_value`, as Model Facts draws its digests)."""
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
    reference = result.get("reference_cohort")
    if _result_kind(result) == PATIENTS_LIKE_ME and isinstance(reference, Mapping):
        # What the result ranked: a request input, not served identity, so it
        # is here and never in the badge's comparison.
        label = reference.get("label")
        rows += [(f"Reference cohort {label} · version (cohort.json)",
                  reference.get("version_sha256")),
                 (f"Reference cohort {label} · samples", reference.get("samples_sha256"))]
    shown = [(label, _fingerprint_value(value, what=str(label), full=True))
             for label, value in rows if _is_sha256(value)]
    if not shown:
        return ""
    return ('<div class="shep-dx-digests"><p class="shep-dx-digests-h"><b>This result in full</b> '
            '<span class="shep-muted">digests to copy (sha256)</span></p>'
            f'{_summary_list(shown, wide=[label for label, _value in shown])}</div>')


def model_identity_html(state: Mapping[str, Any], section: Optional[str] = None) -> str:
    """The identity drawer's "Model in use": Model Facts of the served model
    (`diagnosis_card.served_facts_html`, every section open, `section` the
    one it was opened at, scrolled into view), then the claim boundary for
    what is loaded."""
    status = state.get("status")
    sentences: list[str] = []
    if isinstance(status, Mapping) and status.get("initialized"):
        identity = status.get("served_identity")
        kind = _SERVED if isinstance(identity, dict) else IDENTITY_NOT_REPORTED
        sentences = _claim_boundary_lines(kind, identity if isinstance(identity, dict) else None,
                                          IDENTITY_NOTHING_CHECKED_LOADED)
    return card.served_facts_html(state, focus=section) + _boundary_html(sentences)


def identity_outputs(view: Mapping[str, Any], state: Mapping[str, Any],
                     result: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The identity drawer as `view` asks: open or closed, on "Model in use"
    (opened at a section) or "This result". Model in use carries the view's
    ``opened`` token (``data-shep-open``): its page scrolls the drawer once
    for each token (`FACTS_FOCUS_JS`), so each request to open it is drawn
    as asked and a redraw of the same opening moves nothing."""
    if not view.get("open"):
        return {"identity_drawer": gr.update(visible=False)}
    on_model = view.get("seg", "model") == "model"
    # Each view shown or hidden by a class (`view_update`); a hidden one is
    # not rendered.
    return {
        "identity_drawer": gr.update(visible=True),
        "identity_seg": gr.update(value="model" if on_model else "result"),
        "id_model": view_update(on_model, *_HTML_VIEW,
                                value=f'<div data-shep-open="{_e(str(view.get("opened", "")))}">'
                                + model_identity_html(state, view.get("section")) + "</div>"
                                if on_model else None),
        "id_result_html": view_update(not on_model, *_HTML_VIEW,
                                      value=None if on_model
                                      else result_identity_html(result, state)
                                      + result_digests_html(result)),
    }


# =============================================================================
# The served task's inputs (K6 decision 15)
# =============================================================================
#: The task inputs' containers' own class (a view's classes are never empty).
_TASK_BOX = ("shep-dx-task-box",)
#: The marks an option carries for a version the server would refuse: one
#: mapped against another graph, one whose kept files changed after its
#: import, one that cannot be read, and one of another kind, which holds no
#: patients (`card.NOT_A_COHORT`), marked by its kind.
_VERSION_MARKS = {"another": " · another graph", "changed": " · files changed",
                  "unreadable": " · cannot be read", card.NOT_A_COHORT: " · not a cohort"}
_KIND_MARKS = {"kg_edges": " · KG edge set", "generated": " · generated split"}


def _version_mark(version: Mapping[str, Any]) -> str:
    """The mark of a listed version's option (`_VERSION_MARKS`), or none."""
    if version.get("problem"):
        return _VERSION_MARKS["unreadable"]
    binding = version.get("binding")
    if binding == card.NOT_A_COHORT:
        return _KIND_MARKS.get(version.get("kind"), _VERSION_MARKS[binding])
    return _VERSION_MARKS.get(binding, "")


def _reference(version: Mapping[str, Any]) -> bool:
    """Whether a listed version can be ranked as a reference cohort: it can be
    read and is a supplied cohort (not a KG edge set or a generated split); its
    graph is the server's check."""
    return not version.get("problem") and version.get("binding") != card.NOT_A_COHORT


def _cohorts(state: Mapping[str, Any]) -> Dict[str, Any]:
    """The reference cohort versions the card's facts hold
    (`diagnosis_card.reference_cohorts`); none read reads as none listed."""
    facts = state.get("facts") if isinstance(state.get("facts"), Mapping) else {}
    cohorts = facts.get("cohorts")
    return dict(cohorts) if isinstance(cohorts, Mapping) else {"versions": [], "problem": None}


def _fits(text: str, label: str) -> bool:
    """Whether an option's visible `text` (its mark dropped) can be read as
    `label`: each "…" stands for hidden characters."""
    if "…" not in text:
        return text == label
    pattern = ".*".join(re.escape(piece) for piece in text.split("…"))
    return re.fullmatch(pattern, label, re.DOTALL) is not None


def _shared_start(label: str, other: str) -> int:
    """How many leading characters `label` and `other` share."""
    return next((i for i, (a, b) in enumerate(zip(label, other, strict=False)) if a != b),
                min(len(label), len(other)))


def _led_by_difference(label: str, other: str, room: int) -> Optional[str]:
    """`label` within `room` characters, from the start of the token where it
    first differs from `other` (skill §8: the distinguishing token first), a
    "…" for each side cut, each cut between two grapheme clusters
    (`renderers.grapheme_ends`, B72-N4); None when `other` begins with all of
    `label`."""
    differs = _shared_start(label, other)
    if differs >= len(label):
        return None
    ends = _grapheme_ends(label)
    start = differs
    while start and label[start - 1].isalnum():
        start -= 1
    start = min(end for end in ends if end >= start)
    if len(label) - start <= room - (1 if start else 0):
        return ("…" if start else "") + label[start:]
    width = room - (2 if start else 1)
    if differs >= start + width:    # a token longer than the room: its differing character stays
        width = room - 2
        start = min(end for end in ends if end >= differs + 1 - width)
    stop = max(end for end in ends if end <= start + width)
    if stop <= start:              # one character longer than the room
        return None
    return ("…" if start else "") + label[start:stop] + "…"


def _option_text(label: str, mark: str, others: List[str], taken: set) -> str:
    """`label` and its `mark` within `OPTION_CHARS` characters, unique among
    `taken`, and telling the label from every other listed one (`others`): no
    other label fits its visible text (`_fits`). The first of these that
    does: the label whole; cut in the middle; led by the token where it first
    differs from another label (the one sharing its longest start first);
    cut elsewhere. A label no shortening tells apart keeps the middle cut,
    moved until unique; the full label is under the select once it is chosen
    (`cohort_info`). Every cut is between two grapheme clusters (B72-N4): the
    middle cuts are `renderers.cut_at`'s."""
    room = OPTION_CHARS - len(mark)
    keep = room - 1
    first = (keep + 1) // 2
    ends = _grapheme_ends(label)

    def cut(head: int) -> str:
        return _cut_at(label, head, keep, ends)

    if len(label) <= room:
        candidates: List[Optional[str]] = [label]
    else:
        by_start = sorted(others, key=lambda other: -_shared_start(label, other))
        candidates = [cut(first), *(_led_by_difference(label, other, room) for other in by_start),
                      *(cut(head) for head in range(keep + 1))]
    for text in candidates:
        if text is not None and text + mark not in taken \
                and not any(_fits(text, other) for other in others):
            return text + mark
    for head in (first, *range(first - 1, -1, -1), *range(first + 1, keep + 1)):
        if cut(head) + mark not in taken:
            return cut(head) + mark
    raise ValueError(f"no unique option of {OPTION_CHARS} characters for {label!r}")


def cohort_choices(cohorts: Mapping[str, Any], current: Any
                   ) -> Tuple[List[Tuple[str, str]], Optional[str]]:
    """The Reference cohort options and the version chosen. Every listed
    version, as ``(text, full label)``: the text is the label with at most one
    mark ("another graph", "cannot be read", "KG edge set", "generated split"), at most
    `OPTION_CHARS` characters (a long label shortened where it shows what tells
    it from the others, `_option_text`) and unique. The chosen one is the
    current value while it is listed, else the first version mapped against
    the serving workspace's graph, else the first readable cohort; a KG edge
    set or a generated split is never chosen by default."""
    versions = cohorts.get("versions") or []
    labels = [str(v.get("label")) for v in versions]
    options: List[Tuple[str, str]] = []
    taken: set = set()
    for index, version in enumerate(versions):
        label = labels[index]
        mark = _version_mark(version)
        text = _option_text(label, mark, labels[:index] + labels[index + 1:], taken)
        taken.add(text)
        options.append((text, label))
    if current in labels:
        return options, current
    bound = [str(v.get("label")) for v in versions if v.get("binding") == "bound"]
    readable = [str(v.get("label")) for v in versions if _reference(v)]
    return options, (bound or readable or [None])[0]


def cohort_info(cohorts: Mapping[str, Any], chosen: Any) -> str:
    """The chosen version's facts under the select (plain text; the server
    stays the authority): its full label, version fingerprint and whether it
    was mapped against the served graph, or why it cannot be read."""
    versions = cohorts.get("versions") or []
    version = next((v for v in versions if v.get("label") == chosen), None)
    if version is None:
        # What the select holds; why is the Run reason's (`cohort_refusal`).
        return "No version is chosen." if versions else "No version is listed."
    if version.get("problem"):
        return f"{chosen}: cannot be read: {version['problem']}"
    return (f"{chosen}: version {_short_hex(version.get('manifest_sha256'))}, "
            f"{version.get('binding_words') or 'not checked'}")


def cohort_refusal(state: Mapping[str, Any]) -> Optional[str]:
    """Why Run Diagnosis is off for a patients-like-me model, or None: no
    usable registry, no version, no readable one, or no readable cohort (each
    readable version a KG edge set or a generated split), each in its own words
    with who can act
    (the registry's own problem, an operator's text, is not repeated here)."""
    if card.served_task(state.get("status")) != "patients_like_me":
        return None
    cohorts = _cohorts(state)
    versions = cohorts.get("versions") or []
    if cohorts.get("problem"):
        return f"{RUN_DISABLED_NO_COHORT} {NO_COHORT_REGISTRY}"
    if not versions:
        return f"{RUN_DISABLED_NO_COHORT} {NO_COHORT_VERSION}"
    if all(v.get("problem") for v in versions):
        return f"{RUN_DISABLED_NO_COHORT} {NO_READABLE_VERSION}"
    if not any(_reference(v) for v in versions):
        return f"{RUN_DISABLED_NO_COHORT} {NO_REFERENCE_VERSION}"
    return None


# =============================================================================
# The card's outputs, the Run controls
# =============================================================================
def run_controls(state: Mapping[str, Any], phase: Any) -> Dict[str, Any]:
    """Run Diagnosis: off while a model change is open, when no model is
    loaded, or when a patients-like-me model has no reference cohort version
    to rank; greyed and not clickable, with the reason as visible text under
    it and a bubble on hover and on keyboard focus (a transparent ⓘ over the
    button: the shared help control)."""
    no_cohort = cohort_refusal(state)
    if phase == "loading":
        key, why = "dx-run-loading", RUN_DISABLED_LOADING
    elif phase not in (None, "idle"):
        key, why = "dx-run-change", RUN_DISABLED_CHANGE
    elif card.run_blocked(state):
        # One reason for both (the card says which: Stopping, or nothing
        # served), each cause's own bubble.
        key = "dx-run-stopping" if card.stopping(state) else "dx-run-no-model"
        why = RUN_DISABLED_NO_MODEL
    elif no_cohort is not None:
        key, why = "dx-run-no-cohort", no_cohort
    else:
        return {"run_btn": gr.update(interactive=True), "run_cover": "", "run_reason": ""}
    title = TOPIC_HELP[key][0]
    cover = (f'<button type="button" class="shep-i shep-i-cover" data-help="{key}" '
             f'data-help-title="{_e(title)}" aria-label="Why Run Diagnosis is unavailable" '
             f'aria-expanded="false" aria-describedby="{text_id(key)}"></button>')
    return {"run_btn": gr.update(interactive=False), "run_cover": cover,
            "run_reason": _marked_line("neutral", why)}


def _run_tuple(state: Mapping[str, Any], phase: Any) -> Tuple[Any, Any, Any]:
    run = run_controls(state, phase)
    return run["run_btn"], run["run_cover"], run["run_reason"]


def card_loading_outputs(plan: Any, state: Any) -> Tuple[Any, ...]:
    """The page's part of the step "Reload with …" runs before the reload
    (the shared list's ``on_loading``): Run Diagnosis off with its reason,
    shown before the reload holds the server. The load itself is said by
    the card, from the server's status (`diagnosis_card.serving_state`,
    `loading_too`), never by a line of the page's own. Nothing when there
    was no plan (nothing is loading)."""
    if not plan:
        return gr.update(), gr.update(), gr.update()
    state = state if isinstance(state, Mapping) else card.new_state()
    return _run_tuple(state, "loading")


def card_loaded_outputs(state: Mapping[str, Any], phase: Any) -> Tuple[Any, ...]:
    """After the reload's answer: Run Diagnosis as the phase now says (idle,
    or a confirmation shown again)."""
    return _run_tuple(state, phase)


#: The outputs that follow the served task (`card_outputs(..., inputs=True)`):
#: the task inputs' containers and the Reference cohort select. Never outputs
#: of Run Diagnosis, which reads the inputs.
TASK_OUTPUTS = ("genes_box", "cohort_box", "explanations_box", "cohort")


def asks_explanations(task: Optional[str]) -> bool:
    """Whether the served task reads the Explanations option: a disease task
    model explains its rows (evidence, explanation, paths); a gene or a
    patients-like-me model's rows carry their own values and no explanation
    (`TASK_ROW_KINDS`), so it is not asked for them. Shown while the task is
    not known (nothing read yet, or nothing served: Run is off then)."""
    return task not in ("gene", "patients_like_me")


def card_outputs(state: Mapping[str, Any], result: Optional[Mapping[str, Any]], phase: Any,
                 view: Optional[Mapping[str, Any]] = None, *, cohort: Any = None,
                 inputs: bool = True, stop: Any = None) -> Dict[str, Any]:
    """Every output that follows the card's state, sending only what changed
    since this session's last send (kept in the state's ``rendered``): a
    focused control is never redrawn by the timer for nothing.

    With `inputs`, the served task's inputs too (`TASK_OUTPUTS`): the
    Candidate genes box shown for a gene task model and the Reference cohort
    select for a patients-like-me model, each by its container's class; the
    select's options and its chosen version's facts (`cohort_info`). The
    select is sent only when its options or a listed version's facts change,
    or when `cohort` (its value now) is no longer listed: its cache holds the
    server's facts, never the user's pick, so a pick never makes the timer
    redraw it, even when a read that ran beside the pick writes the state
    last (the timer runs outside the queue). Run Diagnosis passes
    ``inputs=False`` (its outputs hold none of them).

    With `stop` (this session's Stop inference binding, ``stop_asked``: the
    card's reads pass it), the Stop inference views this read closes
    (`stop_views_closed`), once for each binding (kept in ``rendered`` as
    ``stop_closed``)."""
    produced = _produced_at(result)
    button = card.change_button(state)
    run = run_controls(state, phase)
    view = view or {}
    fresh: Dict[str, Any] = {
        "card_time": card.time_html(state, produced),
        "card_body": card.body_html(state, produced,
                                    synthetic_said=synthetic_banner_shown(result, state)),
        "change_btn": button,
        "stop_btn": card.stop_button(state),
        "change_reason": card.change_reason_html(state),
        "details_btn": card.details_visible(state),
        "run_btn": run["run_btn"].get("interactive"),
        "run_cover": run["run_cover"],
        "run_reason": run["run_reason"],
        "badge": badge_html(result, state),
    }
    if not isinstance(result, Mapping):
        fresh["results_body"] = empty_results_html()
    if view.get("open"):
        for name, update in identity_outputs(view, state, result).items():
            fresh[name] = update
    chosen, info = cohort, None
    if inputs:
        task = card.served_task(state.get("status"))
        fresh["genes_box"] = task == "gene"
        fresh["cohort_box"] = task == "patients_like_me"
        fresh["explanations_box"] = asks_explanations(task)
        if task == "patients_like_me":    # a hidden select keeps its last options
            cohorts = _cohorts(state)
            options, chosen = cohort_choices(cohorts, cohort)
            info = cohort_info(cohorts, chosen)
            fresh["cohort"] = {"choices": [list(option) for option in options],
                               "facts": {label: cohort_info(cohorts, label)
                                         for _text, label in options}}
    previous = state.get("rendered") if isinstance(state.get("rendered"), Mapping) else {}
    out: Dict[str, Any] = {}
    for name, value in fresh.items():
        if previous.get(name) == value and not (name == "cohort" and chosen != cohort):
            continue
        if name == "change_btn":
            out[name] = gr.update(**value)
        elif name == "stop_btn":
            out[name] = view_update(value["visible"], *_STOP_BTN, value=value["value"],
                                    interactive=value["interactive"])
        elif name == "details_btn":
            out[name] = gr.update(visible=value)
        elif name == "run_btn":
            out[name] = gr.update(interactive=value)
        elif name in ("genes_box", "cohort_box", "explanations_box"):
            out[name] = view_update(value, *_TASK_BOX)
        elif name == "cohort":
            out[name] = gr.update(choices=[tuple(option) for option in value["choices"]],
                                  info=info, **({"value": chosen} if chosen != cohort else {}))
        else:
            out[name] = value
    rendered = {**previous, **fresh}
    closed = stop_views_closed(state, stop) if stop != previous.get("stop_closed") else {}
    if closed:
        out.update(closed)
        rendered["stop_closed"] = stop
    out["card_state"] = {**state, "rendered": rendered}
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
    new = card.advance(state if isinstance(state, Mapping) else card.new_state(), status,
                       problem, now=now or datetime.now(), operator=allowed, d8_short=short,
                       open_checkpoints=open_checkpoints, reread=reread)
    return {**new, "workspaces_notice": _workspaces_notice()}


def _workspaces_notice() -> Optional[Dict[str, Any]]:
    """The workspaces folder's notice as read now (`workspaces.setting_notice`,
    None while it is usable): the card says it (`diagnosis_card.body_html`)."""
    from src.config import server_settings
    from src.webui.components.workspaces import setting_notice

    return setting_notice(server_settings.workspaces_root())


def refresh_card(state: Any, result: Any, phase: Any, view: Any, request: Any, *,
                 reread: bool = False, cohort: Any = None, stop: Any = None) -> Dict[str, Any]:
    """Page load, Diagnosis shown, ↻, a Load, a stop: read and show (queued:
    the trained-on dataset may be read from the served checkpoint, the
    reference cohort versions are listed again); `cohort` is the select's
    value now, `stop` this session's Stop inference binding."""
    return card_outputs(read_card(state, request, open_checkpoints=True, reread=reread),
                        result, phase, view, cohort=cohort, stop=stop)


def poll_card(location: Any, state: Any, result: Any, phase: Any, view: Any,
              request: Any, cohort: Any = None, stop: Any = None) -> Dict[str, Any]:
    """The shell's slow timer, while Diagnosis is shown: one status read, no
    checkpoint opened (`queue=False`); nothing while another page is shown."""
    if location != CLINICAL:
        return {}
    return card_outputs(read_card(state, request, open_checkpoints=False), result, phase, view,
                        cohort=cohort, stop=stop)


# =============================================================================
# Event handlers
# =============================================================================
def _clamped_top_k(top_k: Any) -> int:
    try:
        value = int(top_k)
    except (TypeError, ValueError):
        value = 10
    return max(1, min(20, value))


def _no_result_outputs(notice: Optional[str]) -> Dict[str, Any]:
    return {
        "results_hd": results_header_html(None), "badge": "",
        "results_body": empty_results_html(notice),
        "results_state": None, "selected": None,
        "export_csv": download_update(visible=EXPORTS_HIDDEN),
        "export_report": download_update(visible=EXPORTS_HIDDEN),
        **details_outputs(None, None),
    }


#: Said at the click, with no API call, when a gene task model is served and
#: the Candidate genes box holds none (decision 15: a click-time notice, as for
#: missing HPO terms).
NO_CANDIDATE_GENES = (
    '<p class="shep-dx-warn"><b>No candidate genes entered.</b> The model in use ranks the '
    'genes you list: enter gene symbols or ids under Candidate genes, one per line or '
    'comma-separated.</p>')


def diagnose(phenotype_text: str, patient_id: str, top_k: Any, explanations: bool,
             state: Any, phase: Any, view: Any, request: Any, *, genes_text: str = "",
             cohort: Any = None, gen: Any = 0) -> Dict[str, Any]:
    """Run Diagnosis: the shared API (`_call_diagnose`), then the status read
    again for the badge, so a result never meets an older read. The full
    result is kept in this session's state; the export files are written up
    front, so one click downloads them.

    **The result keeps the inputs generation its run read** (`gen`, the
    page's count of its inputs' edits as Run Diagnosis' own request carries
    it, read in the same step as the boxes' text: `INPUTS_HEAD`,
    `INPUTS_GEN_JS`), and the results body that draws it marks it
    (`answer_mark`): the page draws the answer, and offers its download, only
    while no edit has been made since (`ANSWER_GATE_CSS`, OPS-C-R3).

    **The served task's input goes only to its task** (the card's last read,
    `diagnosis_card.served_task`): a gene task model gets the Candidate genes
    (`_parse_gene_list`; none is a notice at the click, and nothing is asked
    of the server), a patients-like-me model the Reference cohort label;
    any other model neither; Explanations is asked of a disease task model
    only (`asks_explanations`)."""
    state = state if isinstance(state, Mapping) else card.new_state()
    view = view if isinstance(view, Mapping) else {}
    hpo_ids = _parse_hpo_ids(phenotype_text)
    if not hpo_ids:
        return _no_result_outputs(
            '<p class="shep-dx-warn"><b>No valid HPO terms found.</b> Enter HPO ids like '
            '<code>HP:0001250</code> — one per line, or comma/space-separated. Names are '
            'optional.</p>')
    task = card.served_task(state.get("status"))
    genes = _parse_gene_list(genes_text) if task == "gene" else None
    if task == "gene" and not genes:
        return _no_result_outputs(NO_CANDIDATE_GENES)
    reference = cohort if task == "patients_like_me" and isinstance(cohort, str) and cohort \
        else None
    # Asked of a disease task model only (`asks_explanations`).
    explanations = bool(explanations) and asks_explanations(task)
    result = _call_diagnose(phenotypes=hpo_ids, patient_id=patient_id, top_k=_clamped_top_k(top_k),
                            include_explanations=explanations, candidate_genes=genes,
                            reference_cohort=reference)
    if "error" in result:
        error = _e(result["error"]).replace("\n\n", "<br><br>")
        return _no_result_outputs(f'<p class="shep-dx-error"><b>Error:</b> {error}</p>')
    result["_query_phenotypes"] = hpo_ids
    result["_query_not_ids"] = not_ids(phenotype_text)
    # The terms' names, from the served graph while it is the result's own
    # (by its digest): read once, kept with the result (R-3, R-5).
    result["_names"] = result_names(result)
    if genes is not None:
        result["_query_candidate_genes"] = genes
    result["_explanations"] = explanations
    result["_received_at"] = card.stamp(datetime.now())
    # The inputs generation this run read (`answered_inputs`, `answer_mark`).
    result["_inputs_gen"] = inputs_generation_of(gen)
    # A response that does not name its kind shows no candidate: nothing is
    # selected and nothing is exported (the body says why).
    candidates = (result.get("candidates") or []) if _result_kind(result) is not None else []
    first = candidates[0].get("rank") if candidates else None
    export_note = ""
    if candidates:
        try:
            csv_path, md_path = _write_exports(result)
            # Each file by its link: an opaque id, never its path (OPS-C-R4).
            csv_update, md_update = download_update(csv_path), download_update(md_path)
        except Exception as exc:  # noqa: BLE001 — never let export break the result
            logger.warning("Export file generation failed: %s", exc)
            csv_update, md_update = download_update(), download_update()
            export_note = ('<p class="shep-dx-warn">Export files could not be written '
                           '(disk/permissions?). Results are shown. Run Diagnosis again to retry '
                           'the export.</p>')
    else:
        csv_update, md_update = download_update(), download_update()
    new_state = read_card(state, request, open_checkpoints=True)
    out = card_outputs(new_state, result, phase, view, inputs=False)
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


#: What the results panel says of a result its inputs were edited after
#: (`phenotypes_changed`), and while the page holds back an answer to inputs
#: edited since its run (`ANSWER_GATE_CSS`): the same words either way.
INPUTS_CHANGED = "Inputs changed — Run Diagnosis to refresh the results."


def _inputs_changed_html() -> str:
    return f'<p class="shep-dx-notice-p">{_e(INPUTS_CHANGED)}</p>'


def phenotypes_changed(result: Any, state: Any) -> Dict[str, Any]:
    """The cleared result: no result and no exports, the page saying the
    inputs changed, so nothing stale can be read or exported (what
    `answered_inputs` does for an answer of another generation). Nothing
    happens before a result exists (typing the first query)."""
    if not result:
        return {}
    state = state if isinstance(state, Mapping) else card.new_state()
    out = _no_result_outputs(_inputs_changed_html())
    out["card_state"] = {**state, "rendered": {
        **(state.get("rendered") or {}), "badge": ""}}
    return out


def inputs_generation_of(value: Any) -> Optional[int]:
    """The page's inputs generation as a request carries it (the hidden
    ``shep-dx-inputs-gen`` slot, written by the request's own script,
    `INPUTS_GEN_JS`): a count from 0; None for anything else (a page without
    its head's count, `INPUTS_HEAD`), which no answer is kept under."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 \
        else None


def answered_inputs(result: Any, state: Any, gen: Any) -> Dict[str, Any]:
    """**A result is kept only for the inputs it answered** (UI-15, B73),
    decided by the page's inputs generation as the request carries it
    (`gen`, `INPUTS_GEN_JS`): the request of the check chained after Run
    Diagnosis, once its answer is written, and the request of each edit's own
    listener (the HPO box's and the genes box's change, the Reference
    cohort's pick). When the generation the result answered
    (``_inputs_gen``) is not the page's, the result is cleared
    (`phenotypes_changed`); when it is, nothing changes and the result keeps
    its model's identity: an answer to inputs edited while its run was in
    flight is cleared, and an answer to a Run pressed just after an edit is
    kept, whichever of the requests comes first (REV-R3-b).

    The page never waits for these requests to stop drawing such an answer:
    it draws it, and offers its download, only under the generation it
    answered (`ANSWER_GATE_CSS`), whether they come, come late or fail
    (OPS-C-R3); they keep the session's state and the page's words in step
    with it."""
    if not result:
        return {}
    page = inputs_generation_of(gen)
    if (isinstance(result, Mapping) and page is not None
            and result.get("_inputs_gen") == page):
        return {}
    return phenotypes_changed(result, state)


#: The page's two states its gate reads (`ANSWER_GATE_CSS`; ``{gen}`` is the
#: page's inputs generation): the results body draws no answer of this
#: generation (none at all, or one of another, `answer_mark`), and it draws
#: an answer of another generation.
_NOT_THIS_ANSWER = '#shep-dx:not(:has(#shep-dx-results-body [data-shep-gen="{gen}"]))'
_STALE_ANSWER = '#shep-dx:has(#shep-dx-results-body [data-shep-gen]:not([data-shep-gen="{gen}"]))'

#: **An answer is drawn, and its download offered, only under the inputs it
#: answered** (OPS-C-R3, the external review of 6a2d52e; UI-15, B73): the
#: page's gate, a style sheet the page's head script (`INPUTS_HEAD`) writes
#: for the page's inputs generation at each edit, in the edit's own event. What is drawn from an answer (the header's count,
#: the badge, both exports and their ⓘ, the selected candidate's details) is
#: drawn only while the results body draws an answer of this generation; in
#: its place the details' empty state (``#shep-dx-details-stale-hd`` and
#: ``#shep-dx-details-stale``, what a cleared result leaves there, each part
#: where the details draw it). An answer of another generation in the
#: results body is not drawn either, and the page says what the cleared
#: state says (`INPUTS_CHANGED`, ``#shep-dx-stale``). Gradio writes an
#: answer's outputs one by one; a style applies to each as it is written,
#: before the browser draws it, so no part of an answer that arrives after an
#: edit is ever drawn nor offered, whatever order its parts come in and
#: whatever any later request does; an edit made with an answer shown hides
#: it at once. Once the check chained after the run (`answered_inputs`) or an
#: edit's own listener clears it, the page draws the cleared state the gate
#: imitates; a new run answers this generation. Nothing is recomputed and
#: nothing is deleted.
ANSWER_GATE_CSS = (
    _NOT_THIS_ANSWER + " :is(#shep-dx-results-hd .shep-panel-right, "
    "#shep-dx-badge :is(.shep-dx-badge-row, .shep-dx-badge-why), #shep-dx-export-csv, "
    "#shep-dx-export-md, #shep-dx-exports-help, #shep-dx-details-hd, #shep-dx-seg, "
    "#shep-dx-details-body, #shep-dx-explanation, #shep-dx-details-ft) "
    "{ display: none !important; }\n"
    + _NOT_THIS_ANSWER + " :is(#shep-dx-details-stale-hd, #shep-dx-details-stale) "
    "{ display: block !important; }\n"
    + _STALE_ANSWER + " #shep-dx-results-body { display: none !important; }\n"
    + _STALE_ANSWER + " #shep-dx-stale { display: block !important; }")

#: **An edit of an input the answer depends on, counted in the browser in the
#: edit's own event** (REV-R3-a, REV-R3-b: the review of 961d901; OPS-C-R3):
#: the page's head script (in the dashboard's head, `app.dashboard_head`)
#: holds the page's inputs generation, a count from 0, and at each such edit
#: raises it by one and writes the page's gate for it (`ANSWER_GATE_CSS`),
#: before the event's next listener runs. The edits are the ones whose own
#: listeners clear a result: each input event of the HPO box's and the genes
#: box's text area (every change of a box's text Gradio takes is one: its
#: value binding reads the box at that event alone, so a keystroke, a delete,
#: a cut, a paste, a drop and each step of an input method's composition
#: are each counted), the Reference cohort's blur (its pick, `shell.picked`:
#: Gradio sends it from the select's own blur), and a click on Clear (which
#: writes both boxes in the browser, with no input event).
#:
#: Before, the count went up in a listener of the boxes' ``.change`` and the
#: cohort's blur, which Gradio dispatches some time after the box's text has
#: changed (measured 45 to 110 ms; a text box's two ticks, each an animation
#: frame and a zero timeout). An answer arriving in between was drawn, its
#: download offered, under the edited box (REV-R3-a); a Run pressed in
#: between carried the boxes' new text with the count before the edit, so
#: the answer to the text shown was hidden, then cleared (REV-R3-b).
#:
#: ``ShepDx.gen()`` reads the count. Each request that decides on an answer
#: carries it in its last input (`INPUTS_GEN_JS`).
INPUTS_HEAD = "<script>" + r"""
(function () {
  if (window.ShepDx) { return; }
  var GATE = __GATE__, gen = 0;
  // By event: the elements an edit of an input the answer depends on reaches.
  var EDITS = { input: '#dx_hpo_terms textarea, #dx_candidate_genes textarea',
                focusout: '#dx_reference_cohort input', click: '#shep-dx-clear' };
  function edited() {
    gen += 1;
    var gate = document.getElementById('shep-dx-answer-gate');
    if (!gate) {
      gate = document.createElement('style');
      gate.id = 'shep-dx-answer-gate';
      document.head.appendChild(gate);
    }
    gate.textContent = GATE.split('{gen}').join(String(gen));
  }
  Object.keys(EDITS).forEach(function (type) {
    document.addEventListener(type, function (e) {
      var el = e.target;
      if (el && el.closest && el.closest(EDITS[type])) { edited(); }
    }, true);
  });
  window.ShepDx = { gen: function () { return gen; } };
})();
""".replace("__GATE__", json.dumps(ANSWER_GATE_CSS)) + "</script>"

#: **A request carries the page's inputs generation as it is when the request
#: is made** (the listener's ``js``): its last input, the hidden
#: ``shep-dx-inputs-gen`` slot, becomes ``ShepDx.gen()`` (`INPUTS_HEAD`), or
#: null on a page without that head (`inputs_generation_of`). Gradio 6.30 runs
#: it on the inputs it has just read, in the same step, so Run Diagnosis'
#: request carries the count of the edits its text holds, never one before
#: them; the check chained after the run and each edit's own listener read
#: the count as it is when they go.
INPUTS_GEN_JS = ("(...args) => { const page = window.ShepDx; "
                 "args[args.length - 1] = page ? page.gen() : null; return args; }")


def cohort_picked(result: Any, state: Any, cohort: Any, gen: Any = None) -> Dict[str, Any]:
    """A Reference cohort picked (its blur, `shell.picked`, never a card
    update): a result of another inputs generation is cleared as for edited
    phenotypes (`answered_inputs`, `gen` as the pick's request carries it),
    and the select's facts follow the pick (`cohort_info`). The card's state
    is written back only with a cleared result: the select's cache holds the
    server's facts, not the pick (`card_outputs`), so the timer has nothing
    to send again and a read running beside the pick is not written over."""
    state = state if isinstance(state, Mapping) else card.new_state()
    out = answered_inputs(result, state, gen)
    out["cohort"] = gr.update(info=cohort_info(_cohorts(state), cohort))
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
                  section: Optional[str] = None, phase: Any = "idle") -> Dict[str, Any]:
    """Details › (Model in use, at its top), Recorded results › (at Evidence,
    scrolled into view: every section is open) or the badge (This result):
    the identity drawer, in the details column's
    slot; the change-model drawer closes. Each request is an opening of its
    own (``opened``, `identity_outputs`), so it is drawn as asked whether the
    drawer was open or not. While a model change is open (its
    confirmation, or a reload running: `phase` not idle) the change drawer
    stays and the identity drawer does not open: the confirmation, and the
    reload's outcome, are never hidden behind it (Run Diagnosis says to finish
    or cancel the change first)."""
    if phase not in (None, "idle"):
        return {"change_drawer": gr.update(visible=True)}
    view = {"open": True, "seg": seg, "section": section, "opened": time.monotonic_ns()}
    state = state if isinstance(state, Mapping) else card.new_state()
    return {"identity_view": view, "change_drawer": gr.update(visible=False),
            **identity_outputs(view, state, result)}


def close_identity() -> Dict[str, Any]:
    return {"identity_view": {"open": False}, "identity_drawer": gr.update(visible=False)}


def open_change(request: gr.Request) -> Dict[str, Any]:
    """"Change model…" / "Load a model…" (`card.change_button`): D8 on this
    request first. A request D8 refuses gets the reason and no list (listing
    opens checkpoint files); otherwise the change-model drawer opens (its list
    is read next, by the shared list's own refresh, which asks D8 again)."""
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


# =============================================================================
# Stop inference (the ops round's I1, I4): asked first, bound to the model shown
# =============================================================================
#: The outputs of the two Stop inference handlers, in order.
STOP_OUTPUTS = ("stop_text", "stop_asked", "stop_now_btn", "stop_keep_btn", "stop_notice")
#: The Stop inference views the card's reads may close (`stop_views_closed`):
#: every output above but the binding, which only these handlers write.
STOP_VIEWS = ("stop_text", "stop_now_btn", "stop_keep_btn", "stop_notice")


def _bound(**binding: Any) -> Dict[str, Any]:
    """This session's Stop inference binding (``stop_asked``): what its views
    show is bound to the model the card offers to stop (`card.offered_stop`;
    None for none), as ``sha256`` (the question, the only one a confirmation
    stops), ``text_for`` (what the question's line says instead) or
    ``notice_for`` (what a confirmed stop said), with when it was written, so
    two alike are never one."""
    return {**binding, "at": time.monotonic_ns()}


def stop_views_closed(state: Mapping[str, Any], stop: Any) -> Dict[str, Any]:
    """What a card read closes of the Stop inference views: **nothing while
    the card still offers the model they are bound to** (`card.offered_stop`
    of this read, against `stop`, the binding), else the question and its
    buttons, or what a confirmed stop said. The binding itself is never
    written here: a read never replaces what a Stop inference handler
    answered meanwhile. A stop's own answer is bound to what it left served
    (`card.stop_target` of its status), so the read after it keeps it."""
    if not isinstance(stop, Mapping):
        return {}
    offered = card.offered_stop(state)
    if "notice_for" in stop:
        return {} if stop["notice_for"] == offered else {"stop_notice": ""}
    for key in ("sha256", "text_for"):
        if key in stop:
            return {} if stop[key] == offered else {"stop_text": "", **_stop_buttons(False)}
    return {}


def _stop_said(text: str) -> str:
    return f'<p class="shep-dx-stop-said" role="status">{_e(text)}</p>'


def stop_question_html(sha256: str) -> str:
    """Stop inference's question, naming the model it is bound to and what
    the stop does (the skill's §9: what happens, who acts): the diagnoses
    already accepted finish, new ones are refused until a model is loaded,
    and nothing reloads one by itself. Its words were tightened for the
    card's band (OPS-RECHECK-N1S), each of those facts kept."""
    return ('<div class="shep-dx-stop-q">'
            '<b>Stop inference of model '
            + _fingerprint_value(sha256, what="model fingerprint", copy=False) + '?</b> '
            'Diagnoses already accepted finish. New ones are refused until a model is loaded. '
            'Nothing reloads one by itself.</div>')


#: The Stop inference buttons' own classes: each is shown and hidden by a
#: class (`view_update`), never by ``visible``.
_STOP_BTN = ("shep-dx-stop",)
_STOP_NOW = ("shep-dx-stop-now",)
_STOP_KEEP = ("shep-dx-stop-keep",)


def _stop_buttons(shown: bool) -> Dict[str, Any]:
    return {"stop_now_btn": view_update(shown, *_STOP_NOW),
            "stop_keep_btn": view_update(shown, *_STOP_KEEP)}


def _stop_hidden() -> Dict[str, Any]:
    return {"stop_asked": None, **_stop_buttons(False)}


def ask_stop_inference(access: Any, state: Any) -> Dict[str, Any]:
    """Stop inference's first step, for a request D8 allowed (`access`, read
    by the page's handler on its own request): the question, **bound to the
    model the card shows now** (its full fingerprint, kept in this session's
    ``stop_asked``). Nothing is stopped here. Refused, or with no model
    serving on the card, it says why and asks nothing. Whatever it shows
    lasts while the card offers the model it was answered for
    (`stop_views_closed`)."""
    state = state if isinstance(state, Mapping) else card.new_state()
    offered = card.offered_stop(state)
    if not access.allowed:
        return {**_stop_buttons(False), "stop_asked": _bound(text_for=offered),
                "stop_notice": "",
                "stop_text": f'<p class="shep-dx-why">{_e(access.refusal)}</p>'}
    if offered is None:
        return {**_stop_buttons(False), "stop_asked": _bound(text_for=None), "stop_notice": "",
                "stop_text": _stop_said("Nothing to stop: the card shows no model serving. "
                                        "Read it again with ↻.")}
    return {"stop_text": stop_question_html(offered), "stop_asked": _bound(sha256=offered),
            **_stop_buttons(True), "stop_notice": ""}


def memory_after_stop_html(resources: Optional[Mapping[str, Any]] = None) -> str:
    """The resources reading (`TrainingManager.get_system_resources`, what
    ``GET /api/v1/system/resources`` answers), read now, after a stop that
    released the model, in the Training page's words (`renderers.resources_html`).
    **A reading, never a promise** that memory was returned: the stop empties
    the CUDA allocator's unused blocks only, and the GB10's memory is shared
    (I4). A reading that fails says so."""
    try:
        if resources is None:
            from src.api.services.training_manager import training_manager

            resources = training_manager.get_system_resources()
        line = _resources_html(resources)
    except Exception:  # noqa: BLE001 - a reading that fails is said, never raised
        logger.exception("The resources could not be read after a stop")
        return '<p class="shep-dx-line shep-muted">Memory after the stop: not read.</p>'
    return ('<div class="shep-dx-stop-mem"><span class="shep-muted">Read after the stop, a '
            f'reading and not a promise that memory was returned:</span> {line}</div>')


def confirm_stop_inference(access: Any, asked: Any, *, stop: Optional[Callable[..., Any]] = None,
                           resources: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """**The confirmed Stop inference**, for a request D8 allowed (`access`;
    refused, nothing is stopped): the one stop (`run_stop`, in process, the
    function ``POST /api/v1/pipeline/stop`` calls), **bound to the model the
    question named** (``expected_checkpoint_sha256``: another model served
    meanwhile keeps serving, and the stop's words say so). It waits while the
    accepted diagnoses finish, so its handler is a plain ``def`` (Gradio runs
    it in a worker thread). What it says is the stop's own words; after a
    stop that released the model, the resources reading
    (`memory_after_stop_html`). A confirmation never asked on this page stops
    nothing. What it says lasts while the card offers what the stop left
    served (`card.stop_target` of the stop's own status; `stop_views_closed`):
    a "stopped" notice goes once a model is loaded again. `stop` and
    `resources` stand in for tests."""
    out: Dict[str, Any] = {**_stop_buttons(False), "stop_text": ""}
    sha = asked.get("sha256") if isinstance(asked, Mapping) else None
    sha = sha if _is_digest(sha) else None
    if not access.allowed:
        return {**out, "stop_asked": _bound(notice_for=sha),
                "stop_notice": f'<p class="shep-dx-why">{_e(access.refusal)}</p>'}
    if sha is None:
        return {**out, "stop_asked": _bound(notice_for=None), "stop_notice": _stop_said(
            "Nothing was stopped: Stop inference was not asked for a model on this page.")}
    from src.api.routes.pipeline import PipelineStopRequest, run_stop

    outcome = (stop or run_stop)(PipelineStopRequest(expected_checkpoint_sha256=sha))
    said = _stop_said(outcome.message)
    if outcome.released:
        said += memory_after_stop_html(resources)
    left = card.stop_target(outcome.status.model_dump())
    return {**out, "stop_asked": _bound(notice_for=left), "stop_notice": said}


def keep_serving() -> Dict[str, Any]:
    """"Keep serving": the question closed; nothing is done."""
    return {**_stop_hidden(), "stop_text": ""}


#: The card's controls Stop inference's question swaps, as the card draws
#: them: every button of its actions row (Details ›, Change model…, ■ Stop
#: inference), whose place the open question takes, and of its stop row
#: (■ Stop inference now, Keep serving), which the question's answers and the
#: card's reads hide.
STOP_SWAPPED = "#shep-dx-card-actions button, #shep-dx-stop-row button"

#: Where focus goes when a drawing hides the one of those it was on: the
#: first of these drawn and enabled. Keep serving (the safe answer while the
#: question is open), then ■ Stop inference, then Change model… (a stop that
#: left no model to stop).
STOP_FOCUS_ORDER = ("shep-dx-stop-keep", "shep-dx-stop-btn", "shep-dx-change-btn")

#: **One keeper for the card's swapped controls** (OPS-RECHECK-N1S: its
#: review's F3 and the recheck of b10355a), a script of the dashboard's head
#: (`app.dashboard_head`, as `INPUTS_HEAD`) of the shape of the disclosures'
#: keeper (`renderers.KEEP_OPEN_JS`). The root it holds: a drawing hides the
#: card control focus is on, and nothing places focus again, whichever event
#: drew (the question opened, Keep serving, ■ Stop inference now and the
#: steps after it, a card read closing a question a model changed elsewhere
#: left stale). A step chained to the events that swap them (b10355a) left
#: focus on the page's body after the read's close and after a failed step of
#: Stop now's chain. A ``focusin`` on one of `STOP_SWAPPED` remembers it and
#: watches the card's drawing; focus moved elsewhere (a ``focusin`` outside
#: them, a ``focusout`` with a related target) forgets it, and so does focus
#: that went nowhere while the control is still drawn once the script
#: running now is done (the reader clicked on text; a drawing's hide leaves
#: it not drawn), and a press (``pointerdown``) outside them while focus is
#: not on the control, or is on it no longer drawn (a browser may keep focus
#: on a hidden element): a click on text while the body already has focus (a
#: drawing hid the control and left nothing enabled to place focus on, D8
#: refusing as a model changed elsewhere, say) fires no focus event (the
#: review of 6d77a80). A press while focus is still on it, drawn (a touch
#: that scrolls, a scrollbar), moves no focus and forgets nothing. A drawing that
#: leaves the remembered control not drawn (detached or boxless), with focus
#: on nothing, the body or that control (a browser may keep focus on a hidden
#: element), focuses the first of `STOP_FOCUS_ORDER` drawn and enabled,
#: without scrolling (as `renderers.KEEP_OPEN_JS` does: the page stays where
#: the reader left it).
#: Focus on any other element, the HPO box being typed in for one, is not
#: taken; focus on the body is placed only while a card control is the last
#: place the reader focused or pressed; nothing is done while no card
#: control held focus. No ``gr.State``: the script keeps its own ``focused``
#: (the remembered control) and ``watched`` (the card observed).
STOP_FOCUS_HEAD = "<script>" + r"""
(function () {
  if (window.ShepDxStopFocus) { return; }
  window.ShepDxStopFocus = true;
  var SWAPPED = __SWAPPED__, ORDER = __ORDER__, focused = null, watched = null;
  function drawn(el) { return !!el && el.isConnected && el.getClientRects().length > 0; }
  function keep() {
    var was = focused, now = document.activeElement;
    if (!was || drawn(was) || (now && now !== document.body && now !== was)) { return; }
    for (var i = 0; i < ORDER.length; i++) {
      var to = document.getElementById(ORDER[i]);
      if (drawn(to) && !to.disabled) { to.focus({preventScroll: true}); return; }
    }
  }
  var drawing = new MutationObserver(keep);
  document.addEventListener('focusin', function (e) {
    var el = e.target;
    focused = el && el.matches && el.matches(SWAPPED) ? el : null;
    var card = focused && focused.closest('#shep-dx-card');
    if (card && card !== watched) {
      drawing.observe(card, {childList: true, subtree: true, attributes: true});
      watched = card;
    }
  }, true);
  // Focus that went nowhere is the reader's (a click on text), unless a
  // drawing hid its control: the browser tells the two alike (a focusout
  // with no related target), so the control is forgotten once the script
  // running now is done, where it is still drawn and focus is not on it.
  document.addEventListener('focusout', function (e) {
    var was = focused;
    if (!was || e.target !== was) { return; }
    if (e.relatedTarget) { focused = null; return; }
    queueMicrotask(function () {
      if (focused === was && drawn(was) && document.activeElement !== was) { focused = null; }
    });
  }, true);
  // A press elsewhere while focus is not on the control is the reader's, and
  // may move no focus (a click on text while the body has it already).
  document.addEventListener('pointerdown', function (e) {
    var t = e.target;
    if (focused && (document.activeElement !== focused || !drawn(focused))
        && !(t && t.closest && t.closest(SWAPPED))) { focused = null; }
  }, true);
})();
""".replace("__SWAPPED__", json.dumps(STOP_SWAPPED)).replace(
    "__ORDER__", json.dumps(list(STOP_FOCUS_ORDER))) + "</script>"


def _on_stop_inference_request(request: gr.Request, state) -> Tuple[Any, ...]:
    """"Stop inference" on the card: D8 checked first, on this event's own
    request (`ask_stop_inference`), as for the stop itself."""
    asked = ask_stop_inference(operator_access(request), state)
    return tuple(asked.get(name, gr.update()) for name in STOP_OUTPUTS)


def _on_stop_inference(request: gr.Request, asked) -> Tuple[Any, ...]:
    """The confirmed "Stop inference now": an operator control (D8), checked
    on this event's own request (`confirm_stop_inference`); a plain ``def``,
    since the stop waits for the accepted diagnoses."""
    done = confirm_stop_inference(operator_access(request), asked)
    return tuple(done.get(name, gr.update()) for name in STOP_OUTPUTS)


#: While this page's reload runs, its Close, × and Cancel close the whole
#: drawer at once in the page (the card says the server's Loading),
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
#: Model in use drawn as each opening asks (R-28), once for each opening's
#: token (`identity_outputs`' ``data-shep-open``): opened at a section
#: (`renderers.model_facts_html`'s ``data-shep-focus``, "Recorded results ›" at
#: Evidence), that section scrolled into view inside the drawer, since every
#: section is open and the one asked for may be below the drawer's fold;
#: opened at none (Details ›), the drawer's scroll region at its top, where
#: Warnings lead. Checked when the view mounts (a closed drawer opening mounts
#: it with its content) and on each redraw after: Gradio patches the open
#: drawer's view in place, so a new token may be an attribute's change alone.
#: A redraw of the same opening (the card's reads, a diagnosis) keeps its
#: token and moves nothing the reader scrolled.
FACTS_FOCUS_JS = r"""
var shepOpened = null;
var shepFocus = function () {
  var at = element.querySelector('[data-shep-open]');
  if (!at || at.getAttribute('data-shep-open') === shepOpened) { return; }
  shepOpened = at.getAttribute('data-shep-open');
  var sec = at.querySelector('[data-shep-focus]');
  var region = element.closest('.shep-dx-scroll');
  if (sec) { sec.scrollIntoView({block: 'start'}); }
  else if (region) { region.scrollTop = 0; }
};
new MutationObserver(shepFocus).observe(element, {childList: true, subtree: true, attributes: true,
                                                  attributeFilter: ['data-shep-open']});
requestAnimationFrame(shepFocus);
"""


def _close_html(what: str) -> str:
    return (f'<button type="button" class="shep-dx-close" data-key="close" aria-label="Close the '
            f'{_e(what)}" title="Close">×</button>')


def _event_key(evt: Any) -> Optional[str]:
    raw = getattr(evt, "_data", None)
    key = raw.get("key") if isinstance(raw, Mapping) else None
    return key if isinstance(key, str) else None


#: Under the drawer's list, a row of its own under Load (R-27; a message never shares a row
#: with a button): what Load does next, in plain words; the confirmation then shows what
#: changes.
CHANGE_NOTE = ("Choose a model, then Load: you see what changes and confirm it before every "
               "session is served it. Results already produced keep their own identity.")


def create_diagnosis_tab(ctx: Any) -> None:
    """Build Diagnosis (clinical), the first screen (the page contract:
    `src/webui/shell.py`). The card is read on page load, each time the page
    is shown, on the shell's slow timer while it is shown, after a diagnosis
    and after a model change; its help and the fields' go behind the shared
    ⓘ."""
    from src.webui.components.model_list import MODEL_LIST_JS, workspace_line_html
    from src.webui.shell import picked, plain_html

    ctx.help.fields(FIELD_HELP)
    ctx.help.topics(TOPIC_HELP)

    results_state = gr.State(None)
    selected = gr.State(None)
    #: The slot a request carries the page's inputs generation in (its last
    #: input, written by the request's own script, `INPUTS_GEN_JS`; the count
    #: is the page head's, `INPUTS_HEAD`). Nothing writes the slot itself.
    inputs_gen = gr.Number(value=0, precision=0, visible=False, elem_id="shep-dx-inputs-gen")
    card_state = gr.State(card.new_state())
    identity_view = gr.State({"open": False})
    change_closed = gr.State(False)
    #: The model this session's Stop inference question named (its full
    #: fingerprint), or None: the confirmation stops only that model.
    stop_asked = gr.State(None)
    #: The shared "served model changed" token (`ctx.served_model_changed`)
    #: as it was when this page's Reload began (`_reload_began`): its answer
    #: tells from it whether the Load changed what is served
    #: (`_reload_answered`).
    token_at_reload = gr.State(None)

    with gr.Row(elem_id="shep-dx", elem_classes=["shep-page", "shep-dx"], equal_height=False):
        # --- the input column ------------------------------------------------
        with gr.Column(elem_id="shep-dx-input", elem_classes=["shep-panel"], min_width=0):
            plain_html('<div class="shep-panel-hd"><h2>Patient phenotypes</h2></div>',
                       elem_id="shep-dx-input-hd")
            with gr.Column(elem_id="shep-dx-input-bd", elem_classes=["shep-dx-scroll"]):
                # The column's one scroll region, with its floor (`FIELDS_FLOOR`): the
                # card folds its long identity first (`style.FOLD`, OPS-FIX-N1).
                with gr.Column(elem_id="shep-dx-fields", elem_classes=["shep-dx-fields", FOLD_ROOM]):
                    phenotype_input = gr.Textbox(
                        label="HPO terms", lines=3, max_lines=12, elem_id="dx_hpo_terms",
                        info=FIELD_INFO["dx_hpo_terms"], placeholder="HP:0001250, HP:0001263")
                    # Live: how many ids the box holds, and what is not sent.
                    hpo_echo = plain_html("", elem_id="shep-dx-hpo-echo")
                    with gr.Column(elem_id="shep-dx-side", elem_classes=["shep-dx-side"]):
                        # The served task's input, shown by its container's
                        # class (`card_outputs`): containers, so no handler
                        # that redraws them writes to the input inside.
                        with gr.Column(elem_id="shep-dx-genes-box", min_width=0,
                                       elem_classes=[*_TASK_BOX, VIEW_OFF]) as genes_box:
                            genes_input = gr.Textbox(
                                label="Candidate genes", lines=3, max_lines=12,
                                elem_id="dx_candidate_genes",
                                placeholder="SCN1A\ndisgenet:PTEN, MECP2")
                        with gr.Column(elem_id="shep-dx-cohort-box", min_width=0,
                                       elem_classes=[*_TASK_BOX, VIEW_OFF]) as cohort_box:
                            cohort_input = gr.Dropdown(label="Reference cohort", choices=[],
                                                       value=None, interactive=True,
                                                       elem_id="dx_reference_cohort")
                        patient_id_input = gr.Textbox(label="Patient ID (optional)", value="",
                                                      info=FIELD_INFO["dx_patient_id"],
                                                      max_lines=1, elem_id="dx_patient_id")
                        # Drawn as rows of key, ⓘ and control: the help keeps each
                        # control after its ⓘ in the page (`help.KEY_FIRST_CLASS`).
                        with gr.Row(elem_id="shep-dx-options",
                                    elem_classes=["shep-dx-options", KEY_FIRST_CLASS]):
                            plain_html('<span class="shep-dx-opt">Options</span>',
                                       elem_id="shep-dx-options-label")
                            top_k_input = gr.Number(label="Top-K", value=10, minimum=1,
                                                    maximum=20, step=1, precision=0,
                                                    elem_id="dx_top_k", min_width=0, scale=0)
                            # A disease task model's own option (`asks_explanations`),
                            # shown by its container's class.
                            with gr.Column(elem_id="shep-dx-expl-box", min_width=0, scale=0,
                                           elem_classes=[*_TASK_BOX]) as explanations_box:
                                explanations_input = gr.Checkbox(label="Explanations",
                                                                 value=True,
                                                                 elem_id="dx_explanations",
                                                                 min_width=0)
                with gr.Column(elem_id="shep-dx-sticky", elem_classes=["shep-dx-sticky"]):
                    # Its title's ⓘ explains the whole card: its bubble is placed
                    # outside the card (`help.REGION_CLASS`).
                    with gr.Column(elem_id="shep-dx-card",
                                   elem_classes=["shep-dx-card", REGION_CLASS, FOLD]):
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
                        # The poll draws it again: the not-serving reason's
                        # Details stays as the reader left it for that reason
                        # (`plain_html`'s keeper).
                        card_body = plain_html(card.body_html(card.new_state()),
                                               elem_id="shep-dx-card-body",
                                               js_on_load=MODEL_LIST_JS)
                        with gr.Row(elem_id="shep-dx-card-actions",
                                    elem_classes=["shep-dx-card-actions"]):
                            details_btn = gr.Button("Details ›", size="sm", visible=False,
                                                    elem_id="shep-dx-details-btn", scale=0,
                                                    min_width=0)
                            # Change model… and Stop inference, one group (`shep-grp`): where
                            # the card cannot hold its actions on one line they move under
                            # "Details ›" together, so a refused pair has its reason line
                            # (`change_reason`) right under both (the sweep's L7 check, C5:
                            # Stop wrapped alone and Change's reason was two lines under it).
                            with gr.Row(elem_classes=["shep-dx-card-acts", "shep-grp"]):
                                # Labelled by the card's read (`card.change_button`).
                                change_btn = gr.Button("Change model…", size="sm",
                                                       interactive=False, scale=0, min_width=0,
                                                       elem_id="shep-dx-change-btn")
                                # Shown while a model serves (`card.stop_button`).
                                stop_btn = gr.Button("■ Stop inference", size="sm",
                                                     interactive=False, scale=0, min_width=0,
                                                     elem_id="shep-dx-stop-btn",
                                                     elem_classes=[*_STOP_BTN, VIEW_OFF])
                        change_reason = plain_html("", elem_id="shep-dx-change-why")
                        # Stop inference asks first, bound to the model shown.
                        with gr.Row(elem_id="shep-dx-stop-row", elem_classes=["shep-dx-stop-row"]):
                            stop_text = plain_html("", elem_id="shep-dx-stop-text")
                            stop_now_btn = gr.Button("■ Stop inference now", variant="stop",
                                                     size="sm", scale=0, min_width=0,
                                                     elem_id="shep-dx-stop-now",
                                                     elem_classes=[*_STOP_NOW, VIEW_OFF])
                            stop_keep_btn = gr.Button("Keep serving", size="sm", scale=0,
                                                      min_width=0, elem_id="shep-dx-stop-keep",
                                                      elem_classes=[*_STOP_KEEP, VIEW_OFF])
                        stop_notice = plain_html("", elem_id="shep-dx-stop-notice")
                    with gr.Row(elem_id="shep-dx-run", elem_classes=["shep-dx-run"]):
                        with gr.Column(elem_id="shep-dx-run-wrap", min_width=0, scale=1):
                            run_btn = gr.Button("▶ Run Diagnosis", variant="primary", size="lg",
                                                elem_id="shep-dx-run-btn")
                            run_cover = plain_html("", elem_id="shep-dx-run-cover")
                        gr.ClearButton(components=[phenotype_input, genes_input], value="Clear",
                                       size="lg", elem_id="shep-dx-clear", scale=0, min_width=0)
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
                id_model = plain_html("", elem_id="shep-dx-id-model",
                                      js_on_load=MODEL_LIST_JS + FACTS_FOCUS_JS)
                id_result_html = plain_html("", elem_id="shep-dx-id-result",
                                            elem_classes=[VIEW_OFF])
        with gr.Column(visible=False, elem_id="shep-dx-change",
                       elem_classes=["shep-panel", "shep-dx-drawer"], min_width=0) as change_drawer:
            with gr.Row(elem_classes=["shep-dx-drawer-hd"], elem_id="shep-dx-change-hd"):
                # Load for inference: the one drawer for a first load and a
                # change alike (the card's "Load a model…" / "Change model…").
                plain_html('<h2>Load for inference</h2><span class="shep-dx-drawer-note">'
                           + _chain_html("operator controls",
                                         "one model is served to every session") + '</span>',
                           elem_id="shep-dx-change-title")
                change_close = plain_html(_close_html("change-model drawer"),
                                          elem_id="shep-dx-change-close", js_on_load=MODEL_LIST_JS)
            with gr.Column(elem_id="shep-dx-change-bd"):
                links: Dict[str, Any] = {}

                def _head() -> Any:
                    # The Task control: its `.input` sets the shared list's view.
                    links["task"] = gr.Radio(choices=list(TASK_VIEWS), value="all",
                                             label="Task", show_label=False, container=False,
                                             elem_classes=["shep-seg", SEG_WHOLE, "shep-field-aside"],
                                             elem_id="shep-dx-change-task")
                    # Each "›" keeps to the word before it (a no-break space):
                    # at 1440 x 900 the last one was left alone on a line.
                    links["compare"] = plain_html(
                        f'<button type="button" class="shep-dx-link" data-key="models">'
                        f'{COMPARE_LINK}</button>', elem_id="shep-dx-compare",
                        elem_classes=["shep-field-aside"], js_on_load=MODEL_LIST_JS)
                    # The chosen workspace's identity, once, under the select,
                    # as Training, Fine-tune, Models and Test say theirs (the
                    # shared list's line, `model_list.workspace_line_html`).
                    links["wsline"] = plain_html("", elem_id="shep-dx-change-ws")
                    return links["compare"]

                def _foot() -> Any:
                    links["cancel"] = gr.Button("Cancel", size="sm", scale=0, min_width=0,
                                                elem_id="shep-dx-change-cancel")
                    # The next step in plain words (R-27), never "from → to".
                    plain_html(f'<span class="shep-note">{_e(CHANGE_NOTE)}</span>',
                               elem_id="shep-dx-change-note")
                    return links["cancel"]

                def _reload_began(plan, state, token):
                    return (*card_loading_outputs(plan, state), token)

                handle = ctx.create_model_list(
                    "clinical", head=_head, foot=_foot,
                    on_loading={"fn": _reload_began,
                                "inputs": [card_state, ctx.served_model_changed],
                                "outputs": [run_btn, run_cover, run_reason, token_at_reload]},
                    close_while_loading=CLOSE_WHILE_LOADING)

        # --- the candidates ---------------------------------------------------
        with gr.Column(elem_id="shep-dx-results", elem_classes=["shep-panel"], min_width=0):
            results_hd = plain_html(results_header_html(None), elem_id="shep-dx-results-hd")
            with gr.Column(elem_id="shep-dx-results-bd", elem_classes=["shep-dx-scroll"]):
                badge = plain_html("", elem_id="shep-dx-badge", js_on_load=MODEL_LIST_JS)
                results_body = plain_html(empty_results_html(), elem_id="shep-dx-results-body",
                                          js_on_load=MODEL_LIST_JS)
                # Drawn only while the page holds back an answer to inputs
                # edited since its run (`ANSWER_GATE_CSS`).
                plain_html(empty_results_html(_inputs_changed_html()), elem_id="shep-dx-stale")
            with gr.Row(elem_id="shep-dx-results-ft", elem_classes=["shep-panel-ft"]):
                # Not shown until a result exists (`EXPORTS_HIDDEN`).
                # Each export downloads by its link (`exports.offer`: an
                # opaque id, never the file's path; the button draws as a
                # link once it has one).
                export_csv = gr.Button("⬇ Export CSV", size="sm", interactive=False, scale=0,
                                       min_width=0, elem_id="shep-dx-export-csv",
                                       visible="hidden")
                export_report = gr.Button("⬇ Export report (.md)", size="sm", interactive=False,
                                          scale=0, min_width=0, elem_id="shep-dx-export-md",
                                          visible="hidden")
                plain_html(help_button("dx-exports", "Exports"), elem_id="shep-dx-exports-help")
                plain_html(f'<p class="shep-dx-device">{_e(NOT_A_MEDICAL_DEVICE)}</p>',
                           elem_id="shep-dx-device")

        # --- the selected candidate's details ----------------------------------
        with gr.Column(elem_id="shep-dx-details", elem_classes=["shep-panel"], min_width=0):
            # The details' empty state, each part where the details draw it,
            # drawn only while the results body draws no answer of the page's
            # inputs generation (`ANSWER_GATE_CSS`).
            plain_html(details_header_html(None), elem_id="shep-dx-details-stale-hd")
            details_hd = plain_html(details_header_html(None), elem_id="shep-dx-details-hd")
            # The segmented control and the views are shown and hidden by a
            # class, never `visible` (`view_update`).
            details_seg = gr.Radio(choices=list(SEGMENTS), value="evidence",
                                   label="Candidate details view", show_label=False,
                                   container=False, elem_classes=["shep-seg", VIEW_OFF],
                                   elem_id="shep-dx-seg")
            with gr.Column(elem_id="shep-dx-details-bd", elem_classes=["shep-dx-scroll"]):
                plain_html(details_empty_html(), elem_id="shep-dx-details-stale")
                details_html = plain_html(details_empty_html(), elem_id="shep-dx-details-body",
                                          js_on_load=MODEL_LIST_JS)
                details_md = gr.Markdown("", elem_id="shep-dx-explanation",
                                         elem_classes=[*_MD_VIEW, VIEW_OFF])
            details_ft = plain_html("", elem_id="shep-dx-details-ft")

    named = {
        "card_state": card_state, "card_time": card_time, "card_body": card_body,
        "change_btn": change_btn, "change_reason": change_reason, "details_btn": details_btn,
        "stop_btn": stop_btn, "stop_text": stop_text, "stop_asked": stop_asked,
        "stop_now_btn": stop_now_btn, "stop_keep_btn": stop_keep_btn, "stop_notice": stop_notice,
        "run_btn": run_btn, "run_cover": run_cover, "run_reason": run_reason, "badge": badge,
        "results_hd": results_hd, "results_body": results_body, "results_state": results_state,
        "selected": selected, "export_csv": export_csv, "export_report": export_report,
        "details_hd": details_hd, "details_seg": details_seg, "details_html": details_html,
        "details_md": details_md, "details_ft": details_ft, "identity_view": identity_view,
        "identity_drawer": identity_drawer, "identity_seg": identity_seg, "id_model": id_model,
        "id_result_html": id_result_html,
        "change_drawer": change_drawer, "plan": handle.plan, "confirm": handle.confirm,
        "confirm_html": handle.confirm_html, "list_reason": handle.reason, "phase": handle.phase,
        "change_closed": change_closed, "genes_box": genes_box, "cohort_box": cohort_box,
        "explanations_box": explanations_box,
        "cohort": cohort_input, "hpo_echo": hpo_echo,
    }

    def outputs(*names: str) -> List[Any]:
        return [named[n] for n in names]

    def mapped(values: Dict[str, Any]) -> Any:
        # Gradio reads an empty dict as one value; "nothing to update" is skip().
        if not values:
            return gr.skip()
        return {named[n]: v for n, v in values.items()}

    card_names = ("card_state", "card_time", "card_body", "change_btn", "change_reason",
                  "stop_btn", "details_btn", "run_btn", "run_cover", "run_reason", "badge",
                  "results_body",
                  "identity_drawer", "identity_seg", "id_model", "id_result_html")
    # The card's reads also show the served task's inputs (`TASK_OUTPUTS`);
    # Run Diagnosis, which reads those inputs, never writes them. They close
    # the Stop inference views once the card no longer offers the model those
    # are bound to (`STOP_VIEWS`, read from the binding, never writing it).
    read_names = card_names + TASK_OUTPUTS + STOP_VIEWS
    card_inputs = [card_state, results_state, handle.phase, identity_view, cohort_input,
                   stop_asked]

    def _card_refresh(state, result, phase, view, cohort, stop, request: gr.Request):
        return mapped(refresh_card(state, result, phase, view, request, cohort=cohort,
                                   stop=stop))

    def _card_reread(state, result, phase, view, cohort, stop, request: gr.Request):
        return mapped(refresh_card(state, result, phase, view, request, reread=True,
                                   cohort=cohort, stop=stop))

    def _card_poll(location, state, result, phase, view, cohort, stop, request: gr.Request):
        return mapped(poll_card(location, state, result, phase, view, request, cohort, stop))

    refresh = {"fn": _card_refresh, "inputs": card_inputs, "outputs": outputs(*read_names),
               "show_progress": "hidden"}
    # Diagnosis is shown at page load with no tab select, and on each select.
    ctx.blocks.load(**refresh)
    ctx.on_page_open(CLINICAL, refresh)
    ctx.served_model_changed.change(**refresh)
    ctx.timer_slow.tick(_card_poll, inputs=[ctx.location, *card_inputs],
                        outputs=outputs(*read_names), queue=False, show_progress="hidden")
    reread.click(_card_reread, inputs=card_inputs, outputs=outputs(*read_names),
                 show_progress="hidden")

    def _phase_changed(state, phase):
        run = run_controls(state if isinstance(state, Mapping) else card.new_state(), phase)
        return mapped(run)

    handle.phase.change(_phase_changed, inputs=[card_state, handle.phase],
                        outputs=outputs("run_btn", "run_cover", "run_reason"),
                        show_progress="hidden")

    # The reload's answer (reloaded or not): Run Diagnosis follows the phase
    # again, whatever the answer was, and the card reads the status again
    # after every answer, once. A Load that changed what is served set the
    # shared trigger, whose ``.change`` reads it (above); one that changed
    # nothing (refused, shown again, stopped with an error) left the trigger
    # as it was when the Reload began (`token_at_reload`), so the card is read
    # here: the status's last load outcome ("Load refused" and its reason,
    # `diagnosis_card.serving_state`), with nothing said to have changed (the
    # served identity is the same, so no change notice and no result marked).
    def _reload_answered(state, phase, before, token, result, view, cohort, stop,
                         request: gr.Request):
        if token != before:
            run = card_loaded_outputs(state if isinstance(state, Mapping) else card.new_state(),
                                      phase)
            return mapped(dict(zip(("run_btn", "run_cover", "run_reason"), run)))
        out = refresh_card(state, result, phase, view, request, cohort=cohort, stop=stop)
        # Run is sent whatever the card sent last: the step before the reload
        # changed it in the page alone (`card_loading_outputs`).
        return mapped({**out, **run_controls(out["card_state"], phase)})

    handle.reload_done.then(_reload_answered,
                            inputs=[card_state, handle.phase, token_at_reload,
                                    ctx.served_model_changed, results_state, identity_view,
                                    cohort_input, stop_asked],
                            outputs=outputs(*read_names), show_progress="hidden")

    # --- the card's own links: dismiss, Retry / read it, Recorded results ›
    identity_names = ("identity_view", "identity_drawer", "change_drawer", "identity_seg",
                      "id_model", "id_result_html")

    def _card_link(state, result, phase, view, cohort, stop, evt: gr.EventData,
                   request: gr.Request):
        key = _event_key(evt)
        if key == "dismiss":
            return mapped(card_outputs(card.dismiss_change(state), result, phase, view,
                                       cohort=cohort, stop=stop))
        if key == "refresh":
            return mapped(refresh_card(state, result, phase, view, request, reread=True,
                                       cohort=cohort, stop=stop))
        if key == "evidence":
            return mapped(open_identity(state, result, "model", "evidence", phase))
        return gr.skip()

    card_body.click(_card_link, inputs=card_inputs,
                    outputs=outputs(*dict.fromkeys(read_names + identity_names)),
                    show_progress="hidden")

    def _open_details(state, result, phase):
        return mapped(open_identity(state, result, "model", phase=phase))

    details_btn.click(_open_details, inputs=[card_state, results_state, handle.phase],
                      outputs=outputs(*identity_names), show_progress="hidden")

    def _open_result(state, result, phase, evt: gr.EventData):
        if _event_key(evt) != "result":
            return gr.skip()
        return mapped(open_identity(state, result, "result", phase=phase))

    badge.click(_open_result, inputs=[card_state, results_state, handle.phase],
                outputs=outputs(*identity_names), show_progress="hidden")

    def _switch_identity(state, result, seg, view):
        new_view = {**(view or {}), "open": True, "seg": seg, "section": None,
                    "opened": time.monotonic_ns()}
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

    # After a stop, the card's way back to research (`card.BACK_TO_RESEARCH`):
    # Research › Training, where a GPU job starts.
    def _to_research(evt: gr.EventData):
        return ctx.go("training") if _event_key(evt) == "research" else ctx.no_nav()

    card_body.click(_to_research, inputs=None, outputs=ctx.nav_outputs, show_progress="hidden")
    links["compare"].click(_to_models, inputs=None, outputs=ctx.nav_outputs,
                           show_progress="hidden")

    # The workspace line follows the list's listing, whichever event read it.
    handle.listing.change(workspace_line_html, inputs=[handle.listing],
                          outputs=[links["wsline"]], show_progress="hidden")

    # The drawer's Task control: the shared list redraws on its view's change.
    links["task"].input(task_view, inputs=[links["task"]], outputs=[handle.view],
                        show_progress="hidden")

    # --- Change model…: D8 first, then the shared list's own refresh, then the chips
    change_btn.click(_on_change_model, inputs=None, outputs=outputs(*CHANGE_OUTPUTS),
                     show_progress="hidden").then(**handle.refresh).then(**ctx.chip_refresh)

    # --- Stop inference: D8 first, the question bound to the model shown, then
    # the confirmed stop (in process, `run_stop`), the card read again (its
    # reads close these views once the model they are bound to is no longer
    # offered) and the chips (the GPU chip names inference while a CUDA model
    # serves). A focus any drawing of the card hides is placed again by the
    # card's keeper in the page's head (`STOP_FOCUS_HEAD`), never by an event.
    stop_btn.click(_on_stop_inference_request, inputs=[card_state],
                   outputs=outputs(*STOP_OUTPUTS), show_progress="hidden")
    stop_now_btn.click(_on_stop_inference, inputs=[stop_asked], outputs=outputs(*STOP_OUTPUTS),
                       show_progress="hidden").then(**refresh).then(**ctx.chip_refresh)

    def _keep_serving():
        return mapped(keep_serving())

    stop_keep_btn.click(_keep_serving, inputs=None,
                        outputs=outputs("stop_text", "stop_asked", "stop_now_btn",
                                        "stop_keep_btn"), show_progress="hidden")

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

    def _run_diagnosis(text, patient, top_k, explanations, state, phase, view, genes, cohort,
                       gen, request: gr.Request):
        return mapped(diagnose(text, patient, top_k, explanations, state, phase, view, request,
                               genes_text=genes, cohort=cohort, gen=gen))

    def _phenotypes_edited(result, state, gen):
        return mapped(answered_inputs(result, state, gen))

    edited = {"fn": _phenotypes_edited, "inputs": [results_state, card_state, inputs_gen],
              "outputs": outputs(*dict.fromkeys(result_names + ("card_state",))),
              "js": INPUTS_GEN_JS, "show_progress": "hidden"}

    def _answered_inputs(result, state, gen):
        return mapped(answered_inputs(result, state, gen))

    # Run carries the page's inputs generation, read with the boxes' text
    # (`INPUTS_GEN_JS`), which its result keeps and its results body marks
    # (`diagnose`); the page draws the answer only under that generation
    # (`ANSWER_GATE_CSS`, OPS-C-R3). The answer is written, then checked
    # against the generation as it is now (`answered_inputs`): an edit made
    # while the run was in flight clears it from the session's state as any
    # edit does. Outside the queue, so it follows the answer at once.
    run_btn.click(_run_diagnosis,
                  inputs=[phenotype_input, patient_id_input, top_k_input, explanations_input,
                          card_state, handle.phase, identity_view, genes_input, cohort_input,
                          inputs_gen],
                  outputs=outputs(*dict.fromkeys(result_names + card_names + identity_names)),
                  js=INPUTS_GEN_JS
                  ).then(_answered_inputs, inputs=[results_state, card_state, inputs_gen],
                         outputs=edited["outputs"], js=INPUTS_GEN_JS, queue=False,
                         show_progress="hidden")
    # Every change of a box's text clears a result of another generation
    # (UI-15, B73): its ``.change``. Gradio 6.30's Textbox dispatches
    # ``.input`` from a keypress alone, so select all then Delete, a cut, a
    # paste or an input method's text kept the old genes' result. No handler
    # writes either box (the card shows the genes box by its container's
    # class), so a change is the user's edit or Clear. The cohort select
    # clears on its pick, never on a card's write of it (`_cohort_picked`).
    # Each edit is counted in the browser in its own event (`INPUTS_HEAD`),
    # before these requests go: from then on the page draws no answer of an
    # earlier generation, whatever the requests do or when, and each request
    # carries the count as it is when it goes (`INPUTS_GEN_JS`).
    def _phenotypes_typed(result, state, text, gen):
        return mapped({**answered_inputs(result, state, gen), "hpo_echo": hpo_echo_html(text)})

    phenotype_input.change(_phenotypes_typed,
                           inputs=[results_state, card_state, phenotype_input, inputs_gen],
                           outputs=outputs(*dict.fromkeys(result_names
                                                          + ("card_state", "hpo_echo"))),
                           js=INPUTS_GEN_JS, show_progress="hidden")
    genes_input.change(**edited)

    def _cohort_picked(result, state, cohort, gen):
        return mapped(cohort_picked(result, state, cohort, gen))

    picked(cohort_input)(_cohort_picked,
                         inputs=[results_state, card_state, cohort_input, inputs_gen],
                         outputs=outputs(*dict.fromkeys(result_names + ("card_state", "cohort"))),
                         js=INPUTS_GEN_JS, show_progress="hidden")

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
#: 5:4. Side by side (the shell's one layout decision), each panel scrolls
#: inside; in the input column the fields are its one scroll region and the
#: Model-in-use card and Run Diagnosis are its foot, in flow under them, never
#: over them (R-25: a sticky card hid the HPO box's lower lines and its caret
#: at 1536 x 760 once it held 8 lines); the fields have a floor
#: (`FIELDS_FLOOR`) and the card folds its long identity when it would leave
#: them less (`style.FOLD`, measured by the shell's layout pass; OPS-FIX-N1: at
#: 1280 x 600 with a model loaded and two warning lines the fields were 8 px
#: and the focused HPO box was under the card's head); stacked, the panels
#: stack. Every
#: label and control in the input column starts at the panel's one inner edge,
#: and Options is a label over a key/control list, no box of its own (R-24).
#: While the Load confirmation is open (UI-A's confirmation box is mounted only
#: then) the list's footer is hidden and its rows give up their height to it,
#: all of it in a short window (`style.SIDE_SHORT`, under 900 px of height), so
#: the from → to rows and the warning sentence are never scrolled out of view.
#: Decided at 900 (it was 820 before the one layout decision): from 820 to 899
#: px the rows' share holds no whole row (measured 68 px at 1600 x 820, 108 at
#: 1600 x 860, 128 at 1850 x 880: a cut slice of the first row), and the
#: confirmation is whole either way.
#: The Load drawer's two regions side by side (UI-07; OPS-FIX-R2): the gap
#: between them and the details' minimum width (px). The list's region has no
#: fixed minimum: it takes the tree's own narrowest, its box never wider than
#: `DRAWER_TREE_ROOM`, the body's width (``cqw``: the body is its inline-size
#: container) less the details' minimum and the gap.
DRAWER_GAP = 14
DRAWER_FACTS_MIN = 222
DRAWER_TREE_ROOM = f"calc(100cqw - {DRAWER_FACTS_MIN + DRAWER_GAP}px)"
#: **The fields region's floor** (px, side by side; OPS-FIX-N1): the HPO terms
#: field whole, its label, its info line on two lines and its three-line box
#: (166 px measured in the 300 px column); when the fields need more, the
#: region scrolls (the column's one scroll region). The Model-in-use card
#: gives way first: it folds its long identity (`style.FOLD`) when it would
#: leave the region under this floor. The floor is the fold's measure
#: (`style.FOLD_FLOOR`), not a CSS minimum: where the card, folded or with
#: nothing to fold (no model loaded; a read-only viewer's reason line), still
#: leaves less, the region takes what is left and stays the one scroll region,
#: Run Diagnosis in view (a minimum height there put a second scroll region
#: around the fields and Run out of view; the review of 90c375e). Since the
#: sweep's L7 check (C1), the floor also holds the next field's label under the
#: region's cue (206 px: Patient ID's label starts 182 px down): at 166 the
#: edge fell in the gap under the HPO box, so the cue faded nothing drawn and
#: Patient ID read as absent (configured, 1280 x 720, the card unfolded).
#: Since the sweep's L7 check (D7), the floor holds the Patient ID field whole
#: (its label, info line and box: it ends 302 px down in the 390 px column
#: `DX_IN_W` gives at 1366 x 768), so a loaded model's card folds before it
#: hides that box (configured, 1366 x 768: the card unfolded left the fields
#: 296 px, the box's foot under the edge).
FIELDS_FLOOR = 304
#: **The input column's width** (``--dx-in-w``; B-14): about a third of the
#: page's row, never under the HPO field's floor (300 px: its label, info line
#: and box, and the card's lines) nor over 31rem (wider adds only blank beside
#: short fields). It was 300, 320 or 370 px by the window's width (three
#: media queries), so at 1366 and 1280 the column was 300 px whatever the
#: row held beside it. It was a quarter of the row (24 %, 26rem at most)
#: until the sweep's L7 check (D2, D7): the inputs, the page's task before a
#: result, scrolled at the owner's 1550 x 830 (Explanations under the edge,
#: d3040ca drew it) and hid Patient ID's box at 1366 and 1280, while the two
#: panels beside held one empty-state sentence each in the rest of the row.
#: At 32 % the column is 486 px at 1550 and its fields are whole in both
#: states; at 1366 and 1280 Patient ID's box is in view. It takes that third
#: only as far as the results' and the details' minimums leave (`RESULTS_MIN`,
#: `DETAILS_MIN` and the two 16 px gaps): at 1280 x 720 a third ran the row
#: 45 px past the page's right edge.
RESULTS_MIN = 532
DETAILS_MIN = 380
DX_IN_W = f"clamp(300px, min(32%, 100% - {RESULTS_MIN + DETAILS_MIN + 32}px), 31rem)"

PAGE_CSS = f"""
/* **The input column's width, from the row's** (`DX_IN_W`; no window media query): a share of the
   page's row between a floor and a cap. The row is the column's flex container and the drawers'
   containing block (positioned), so its percentage means the same width in each: the Load drawer
   starts after the column, the identity drawer leaves it in view. */
{_TAB} #shep-dx {{ --dx-in-w: {DX_IN_W}; position: relative; gap: 16px; align-items: stretch; }}
/* Drawn only by the page's gate (`ANSWER_GATE_CSS`). */
{_TAB} #shep-dx-stale, {_TAB} #shep-dx-details-stale-hd, {_TAB} #shep-dx-details-stale {{ display: none !important; }}
{_TAB} #shep-dx-input {{ flex: 0 0 var(--dx-in-w) !important; width: var(--dx-in-w); max-width: var(--dx-in-w); }}
{_TAB} #shep-dx-results {{ flex: 5 1 0 !important; }}
{_TAB} #shep-dx-details {{ flex: 4 1 0 !important; }}
/* The three panels' minimum widths (the layout rule's measure, `src/webui/shell.py`): the input
   column's own width at its narrowest; the candidates' table whole from 532 px (its five
   columns' min-content, 495 px measured with the Phenotypes k/N column and the evidence label
   wrapping, plus the region's padding and border); the details. */
{_TAB} #shep-dx-input {{ --shep-panel-min: 300px; }}
{_TAB} #shep-dx-results {{ --shep-panel-min: {RESULTS_MIN}px; }}
{_TAB} #shep-dx-details {{ --shep-panel-min: {DETAILS_MIN}px; }}
{_TAB} .shep-dx-scroll {{ gap: 12px !important; padding: 14px 16px !important; }}
{_TAB} .shep-dx-scroll > * {{ flex-shrink: 0 !important; }}
{_TAB} #shep-dx-input-bd {{ padding-top: 12px !important; padding-bottom: 0 !important; gap: 0 !important; }}
{_TAB} .shep-dx-fields, {_TAB} .shep-dx-side {{ gap: 8px !important; }}
{_TAB} #shep-dx-fields {{ padding: 0 2px 8px; margin: 0 -2px; }}
{_TAB} #shep-dx-fields > .form {{ background: none !important; border: 0 !important; box-shadow: none !important; }}
{_TAB} #shep-dx-fields .block.padded {{ padding: 6px 0 !important; }}
{_TAB} #shep-dx-fields [data-testid="block-info"] {{ margin-bottom: 4px !important; }}
{_TAB} #shep-dx-options .form {{ display: contents !important; }}
{_TAB} #shep-dx-options .block.padded {{ padding: 0 !important; }}
{_TAB} #dx_top_k input[type=number] {{ height: 34px; padding: 4px 8px; }}
/* Options (R-24): its label over its controls, no box of its own; each control a row of one grid,
   its key, its ⓘ and its control in three columns (the keys as wide as the longest), the
   checkbox's words in the key column as Top-K's are. The blocks between the grid and each
   control's label draw no box (display: contents); the label takes the grid's columns. The
   page holds each label's parts in that order too (the help moves the checkbox's box after
   its ⓘ, help.KEY_FIRST_CLASS), so Tab meets them as drawn. */
{_TAB} .shep-dx-options {{ display: grid !important; grid-template-columns: max-content auto 1fr; align-items: center !important;
  gap: 6px 8px !important; padding: 6px 0 !important; }}
{_TAB} #shep-dx-options-label {{ grid-column: 1 / -1; }}
{_TAB} .shep-dx-options #dx_top_k, {_TAB} .shep-dx-options #shep-dx-expl-box:not(.shep-dx-off),
{_TAB} .shep-dx-options #dx_explanations {{ display: contents !important; }}
{_TAB} .shep-dx-options :is(#dx_top_k, #dx_explanations) > label {{ grid-column: 1 / -1; display: grid !important;
  grid-template-columns: subgrid; align-items: center; gap: 0 8px; margin: 0; padding: 0; }}
{_TAB} #dx_explanations label > span {{ margin: 0 !important; font-size: var(--shep-label-size);
  font-weight: var(--shep-label-weight); color: var(--shep-label-color); }}
{_TAB} :is(#dx_top_k, #dx_explanations) label > :is(span, [data-testid="block-info"]) {{ grid-column: 1; grid-row: 1; margin: 0 !important; }}
{_TAB} :is(#dx_top_k, #dx_explanations) label > button.shep-i {{ grid-column: 2; grid-row: 1; }}
{_TAB} :is(#dx_top_k, #dx_explanations) label > input {{ grid-column: 3; grid-row: 1; justify-self: start; margin: 0; }}
{_TAB} .shep-dx-opt {{ font-size: var(--shep-label-size); font-weight: var(--shep-label-weight); color: var(--shep-label-color); }}
/* Top-K's box holds a value of up to two digits with room to type it, as every text box does
   (the walk's floor: ten widths of "0" in its own type, `ch`), beside its spin arrows. */
{_TAB} #dx_top_k input[type=number] {{ width: calc(11ch + 16px); }}
{_TAB} #shep-dx-input-bd > *, {_TAB} #shep-dx-sticky > *, {_TAB} #shep-dx-card > * {{ flex: none !important; }}
{_TAB} .shep-dx-sticky {{ gap: 0 !important; margin: 0 -16px !important;
  width: calc(100% + 32px) !important; max-width: none !important; background: var(--shep-panel); }}
{_TAB} .shep-dx-card {{ background: var(--shep-panel-2); border-top: 1px solid var(--shep-border);
  border-bottom: 1px solid var(--shep-border); padding: 5px 16px 6px !important; gap: 3px !important; }}
{_TAB} .shep-dx-card-hd {{ align-items: center !important; flex-wrap: wrap !important; gap: 2px 6px !important; }}
{_TAB} .shep-dx-card-hd > *, {_TAB} .shep-dx-card-when > * {{ flex: 0 0 auto !important; width: auto !important;
  min-width: 0 !important; }}
{_TAB} .shep-dx-card-when {{ margin-left: auto !important; flex-wrap: nowrap !important; gap: 4px !important;
  align-items: center !important; }}
{_TAB} .shep-dx-card-title {{ white-space: nowrap; }}
{_TAB} .shep-dx-card-title {{ font-size: var(--shep-label-size); font-weight: var(--shep-label-weight); color: var(--shep-label-color); }}
{_TAB} .shep-dx-checked {{ font-size: 14px; color: var(--shep-text-2); white-space: nowrap; }}
{_TAB} button.shep-dx-reread {{ border: 0; background: none; color: var(--shep-accent-text); font-size: 15px; cursor: pointer; padding: 0; line-height: 1;
  display: inline-flex; align-items: center; justify-content: center; min-width: {HIT_TARGET}px; min-height: {HIT_TARGET}px; margin: -4px 0; border-radius: 6px; }}
{_TAB} .shep-dx-card-bd {{ display: flex; flex-direction: column; gap: 1px; font-size: 15px; line-height: 1.3; }}
{_TAB} .shep-dx-card-bd p {{ margin: 0; }}
{_TAB} .shep-dx-name {{ font-weight: 600; }}
{_TAB} .shep-dx-sub {{ font-weight: 400; color: var(--shep-text-2); }}
{_TAB} .shep-dx-line {{ overflow-wrap: anywhere; }}
{_TAB} .shep-dx-tail {{ display: flex; flex-wrap: wrap; gap: 4px 8px; align-items: center; }}
{_TAB} .shep-dx-chips {{ display: inline-flex; flex-wrap: wrap; gap: 4px 6px; }}
{_TAB} .shep-dx-tests {{ white-space: normal; }}
{_TAB} .shep-dx-card-grey .shep-dx-name, {_TAB} .shep-dx-card-grey .shep-dx-line {{ color: var(--shep-grey-600); }}
{_TAB} .shep-dx-card-grey .shep-fpv {{ font-weight: 400; color: var(--shep-grey-600); }}
{_TAB} .shep-tag-dashed {{ border-style: dashed !important; }}
{_TAB} .shep-dx-state {{ display: flex; flex-wrap: wrap; align-items: center; gap: 4px; font-size: 14px; color: var(--shep-grey-800); margin-bottom: 2px; }}
{_TAB} .shep-dx-state .shep-dx-retry {{ height: 24px; padding: 0 6px; }}
/* Nothing served: the backend's lead under the state, its whole reason one click away. */
{_TAB} .shep-dx-serving .shep-st + .shep-st {{ margin-left: 8px; }}
{_TAB} .shep-dx-serving .shep-dx-when {{ white-space: nowrap; }}
{_TAB} .shep-dx-why-more {{ font-size: 14px; }}
{_TAB} .shep-dx-why-more > .shep-dx-line {{ margin-top: 2px; color: var(--shep-grey-800); }}
{_TAB} button.shep-dx-back {{ min-height: 24px; }}
{_TAB} .shep-dx-change {{ display: flex; align-items: flex-start; gap: 8px; margin-bottom: 2px; padding: 4px 8px; }}
{_TAB} button.shep-dx-x {{ margin-left: auto; border: 0; background: none; color: var(--shep-amber-800); font-size: 20px; line-height: 1; cursor: pointer; padding: 0 2px; }}
{_TAB} button.shep-dx-link {{ border: 0; background: none; padding: 0; color: var(--shep-accent-text); font: inherit; cursor: pointer; }}
{_TAB} button.shep-dx-link:hover {{ text-decoration: underline; }}
{_TAB} .shep-dx-card-actions {{ justify-content: space-between !important; align-items: center !important; flex-wrap: wrap !important; gap: 8px !important; margin-top: 2px; }}
{_TAB} .shep-dx-card-actions > * {{ flex: 0 0 auto !important; width: auto !important; }}
{_TAB} #shep-dx-details-btn {{ border: 0 !important; background: none !important; box-shadow: none !important; color: var(--shep-accent-text) !important; padding: 0 2px !important; font-weight: 400; }}
{_TAB} .shep-dx-card-acts {{ margin-left: auto; flex-wrap: nowrap !important; gap: 8px !important; }}
{_TAB} .shep-dx-stop-row {{ flex-wrap: wrap !important; gap: 8px !important; justify-content: flex-end !important; }}
/* The question's buttons keep their words' width (Gradio gives a scale-0
   button a 0% basis: measured 12 px wide, "Keep serving" over "Stop
   inference now", so a click on one hit the other). */
{_TAB} .shep-dx-stop-row > button {{ flex: 0 0 auto !important; width: auto !important; }}
{_TAB} #shep-dx-stop-text {{ flex: 1 1 100% !important; }}
/* Stop inference's question (OPS-RECHECK-N1S): a band across the card (into its 16 px side padding), its
   words the card's whole width, its two buttons under it; while it is open it takes the place of the
   card's actions: Details ›, Change model… and ■ Stop inference are not drawn (the question's own
   buttons answer it), so it adds no row under them. Keep serving, the stop, or a read that no longer
   offers the model it names closes it (`stop_views_closed`), and they are drawn again. (Drawn under the
   actions, a box inside the card's padding, it kept even the folded card 333 px tall: at 1280 x 600 with
   two warning lines the fields were 8 px, the focused HPO box under the card's head.) */
{_TAB} .shep-dx-stop-q {{ background: var(--shep-red-50); border-top: 1px solid var(--shep-red-200);
  border-bottom: 1px solid var(--shep-red-200); margin: 0 -16px; padding: 4px 8px; }}
{_TAB} .shep-dx-stop-row:has(.shep-dx-stop-q) {{ row-gap: 4px !important; }}
{_TAB} #shep-dx-card:has(> #shep-dx-stop-row .shep-dx-stop-q) > .shep-dx-card-actions {{ display: none !important; }}
{_TAB} .shep-dx-stop-said, {_TAB} .shep-dx-stop-mem {{ margin: 2px 0 0; font-size: 14px; }}
/* The card's reasons stand under its actions, at its end edge: on a card wider than the measure (stacked)
   the reason's box, capped at the measure, keeps that edge too (it stood at the card's start, its words
   mid-card under no button; the read-only walk at 1080 x 1920 found Change model… with nothing said). */
{_TAB} .shep-dx-why {{ margin: 2px 0 0 auto; font-size: 14px; color: var(--shep-text-2); text-align: right; }}
{_TAB} .shep-reason.shep-dx-why {{ justify-content: end; }}
{_TAB} #shep-dx-run-why .shep-reason {{ margin: 0 16px 8px; }}
{_TAB} .shep-dx-run {{ padding: 6px 16px !important; gap: 10px !important; flex-wrap: nowrap !important; align-items: stretch !important; }}
{_TAB} #shep-dx-run-wrap {{ position: relative; gap: 0 !important; }}
{_TAB} #shep-dx-clear {{ flex: 0 0 auto !important; min-width: 84px !important; padding: 0 18px !important; }}
{_TAB} #shep-dx-run-cover {{ position: absolute !important; inset: 0; pointer-events: none; z-index: 3; }}
{_TAB} #shep-dx-run-cover .html-container, {_TAB} #shep-dx-run-cover .prose {{ height: 100%; }}
{_TAB} button.shep-i.shep-i-cover {{ position: absolute; inset: 0; width: 100%; height: 100%; margin: 0; border: 0;
  border-radius: 8px; background: transparent !important; cursor: not-allowed; pointer-events: auto; color: transparent; }}
{_TAB} button.shep-i.shep-i-cover::before {{ display: none; }}
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
{_TAB} .shep-dx-echo {{ font-size: 14px; color: var(--shep-text-2); margin: 2px 0 0; padding: 0; overflow-wrap: anywhere; }}
{_TAB} .shep-dx-used {{ margin: 0 0 8px; overflow-wrap: anywhere; }}
{_TAB} .shep-dx-why {{ color: var(--shep-text-2); }}
{_TAB} .shep-dx-tbl-wrap {{ margin: 0; }}
{_TAB} .shep-dx-tbl td, {_TAB} .shep-dx-tbl th {{ white-space: nowrap; }}
/* A disease name breaks anywhere it must (an unbroken token of a long name, 40 characters and
   more, widened the table past its panel at 1440 x 900, Phenotypes and Evidence out of view in
   its sideways scroll: PTR-LAYOUT); its words still break between words first. */
{_TAB} .shep-dx-tbl td.shep-dx-dis {{ white-space: normal; width: 100%; min-width: 9em; overflow-wrap: anywhere; }}
{_TAB} .shep-dx-tbl td.shep-dx-wrap {{ white-space: normal; min-width: 7em; overflow-wrap: anywhere; }}
/* The evidence label wraps before the table scrolls sideways (beside the
   Phenotypes k/N column, R-1d): on one line wherever the panel has room. */
{_TAB} .shep-dx-tbl td.shep-dx-evcell {{ white-space: normal; min-width: 6.5em; }}
{_TAB} .shep-dx-ref code {{ font-family: var(--shep-mono); }}
{_TAB} .shep-dx-task-box {{ gap: 0 !important; }}
{_TAB} #shep-dx-change-ws {{ flex: 1 1 100% !important; margin-top: -4px; }}
{_TAB} #shep-dx-change-ws:not(:has(.shep-ws-line)) {{ display: none !important; }}
{_TAB} .shep-dx-kind {{ margin: 0 0 8px; font-size: 14px; color: var(--shep-text-2); }}
{_TAB} .shep-dx-tbl tbody tr[data-key] {{ cursor: pointer; }}
{_TAB} .shep-dx-tbl tbody tr.shep-on td.shep-dx-dis {{ color: var(--shep-accent-text); font-weight: 600; }}
{_TAB} .shep-dx-clamp {{ display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }}
/* An empty region's words start on the panel's one content edge, under its heading, at the
   measure (the skill's §2; the text brief's §4: empty text starts at the column's one x and
   lines that wrap are never centred): a box centred in the region stood 28-35 px off the
   heading at 1280-1366 and its padding 8 px off at 390, a near miss of the edge. */
{_TAB} .shep-dx-empty {{ max-width: var(--shep-measure); text-align: start; color: var(--shep-text-2); padding: 24px 0; }}
{_TAB} .shep-dx-empty p {{ margin: 4px 0; }}
{_TAB} .shep-dx-empty-h {{ font-size: 15px; font-weight: 600; color: var(--shep-text); }}
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
/* The evidence as an explanation (R-3): the answer, its facts as one summary
   list, the server's narrative folded; a term's name first, its id muted. */
{_TAB} .shep-dx-ev {{ margin: 0 0 8px; }}
{_TAB} .shep-dx-ev-why {{ margin: 0 0 8px; color: var(--shep-text-2); overflow-wrap: anywhere; }}
{_TAB} .shep-dx-ev-more p {{ margin: 4px 0 0; color: var(--shep-text-2); overflow-wrap: anywhere; }}
{_TAB} .shep-dx-id {{ font-size: 14px; color: var(--shep-muted); }}
{_TAB} #shep-dx-details-body p.shep-dx-sub-h {{ margin: 8px 0 4px; font-size: 14px; }}
{_TAB} .shep-dx-step {{ display: inline-block; border: 1px solid var(--shep-border); border-radius: 6px; padding: 1px 6px; background: var(--shep-panel-2); }}
{_TAB} .shep-dx-paths-more > summary {{ margin: 2px 0 6px; }}
{_TAB} .shep-dx-genes {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 0; }}
{_TAB} .shep-dx-gene {{ font-size: 14px; border: 1px solid var(--shep-accent-100); background: var(--shep-accent-50); border-radius: 6px; padding: 2px 8px; }}
{_TAB} .shep-dx-list, {_TAB} .shep-dx-paths {{ margin: 0 0 8px; padding-left: 22px; }}
{_TAB} .shep-dx-paths li {{ margin: 0 0 6px; overflow-wrap: anywhere; }}
{_TAB} .shep-dx-arrow {{ color: var(--shep-muted); }}
{_TAB} #shep-dx-details-ft {{ border-top: 1px solid var(--shep-border); }}
{_TAB} .shep-dx-det-ft {{ display: flex; align-items: center; gap: 4px; padding: 10px 16px; font-size: 14px; color: var(--shep-text-2); }}
{_TAB} #shep-dx-explanation {{ font-size: 15px; }}
/* The written explanation (the Markdown view) is drawn while the Explanation view's fold is
   open, right under its summary (R-1b, `explanation_html`). */
{_TAB} #shep-dx-details-bd:has(details.shep-dx-written:not([open])) #shep-dx-explanation {{ display: none !important; }}
{_TAB} .shep-dx-drawer {{ box-shadow: 0 10px 30px rgba(20, 24, 40, .14); }}
{_TAB} .shep-dx-drawer-hd {{ align-items: center !important; flex-wrap: wrap !important; gap: 8px 12px !important; padding: 10px 16px !important; border-bottom: 1px solid var(--shep-border); }}
/* A drawer's head and the Load drawer's task segment take their words' width and give way to
   the drawer, never past it (V-19: held at their one line's width, the Load drawer's title note
   ran 217 px and its task segment 49 px past the drawer's edge at 390 x 844, cut): a title
   wraps its words; the task segment keeps every label whole (`style.SEG_WHOLE`: one row while
   it fits, else one label a line; the shared segment's one row would end three of its four
   labels in an ellipsis in a phone's 324 px, and wrapping left one alone on a line). */
{_TAB} .shep-dx-drawer-hd > *, {_TAB} #shep-dx-change-task {{ flex: 0 1 auto !important; width: auto !important; min-width: 0 !important; }}
{_TAB} .shep-dx-drawer-hd h2 {{ font-size: var(--shep-title-size); font-weight: 600; margin: 0; display: inline; }}
{_TAB} .shep-dx-drawer-note {{ margin-left: 10px; font-size: 14px; color: var(--shep-text-2); }}
{_TAB} #shep-dx-identity-close, {_TAB} #shep-dx-change-close {{ margin-left: auto; }}
{_TAB} button.shep-dx-close {{ border: 0; background: none; font-size: 20px; line-height: 1; cursor: pointer; color: var(--shep-text-2); padding: 0 4px; }}
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
/* "This result": the answer first, the parts that differ, then the result's
   own identity as one summary list. */
{_TAB} .shep-dx-result-id {{ display: flex; flex-direction: column; gap: 8px; }}
{_TAB} .shep-dx-result-id p.shep-dx-lead, {_TAB} .shep-dx-result-id p.shep-dx-sub-h {{ margin: 0; }}
{_TAB} .shep-dx-result-id .shep-kv {{ font-size: 14px; }}
{_TAB} .shep-dx-result-id .shep-dx-now {{ display: block; margin-top: 2px; }}
{_TAB} .shep-dx-result-id .shep-dx-now-k {{ font-weight: 600; }}
/* The drawers' Model Facts take the one summary list's columns
   (`renderers.summary_list`: a digest's long label on a line of its own). */
{_TAB} #shep-dx-change-bd {{ padding: 12px 16px !important; gap: 10px !important; }}
{_TAB} #shep-dx-change .shep-model-head {{ gap: 8px 14px !important; }}
/* The search and what the list shows, on a line of their own under the head (R-27): the shared
   list's one tools bar (D14), as Models draws it, with nothing of this page's own. */
/* The head is a field row (D03): the workspace select keeps its own minimum width (Gradio's
   min(160px, 100%)), so a drawer too narrow for the row wraps it rather than squeezing the box
   (at 1280 x 900 it drew 26 px, its label cut to "W"); only the words beside it may shrink. */
{_TAB} #shep-dx-change .shep-model-head > :not(.form) {{ min-width: 0 !important; }}
/* **The list and the chosen model's details, two regions** (UI-07: they shared one scroll, so
   reading the details moved the list). Side by side the list takes the tree's own narrowest
   (its min-content width in the type drawn: 599 px with a model of every task in the dashboard's
   type, "Patients-like-me" its Task column's widest word and a 12-hex digest in the host's
   monospace), as far as the details leave room: the tree's box is never wider than the body less
   the details' minimum and the gap (`DRAWER_TREE_ROOM`, against the body's own width), so a
   longer run name scrolls the list sideways in its own region. OPS-FIX-R2: the list's minimum was
   a fixed 616 px, fitted to this host's text, and on the reviewer's host the same tree was 620 px
   in it. The details' summary lists keep 222 px (`--shep-panel-min`: a key and its value, each
   wrapping). Side by side the drawer covers the candidates and the details panels, whose own
   minimums are no narrower than that tree and the details beside it (summed by
   `test_diagnosis_panel.py`); each scrolls on its own, Load in the foot under both. Stacked, the
   page flows: the list, then the details, then Load; choosing a row brings its details into view
   and "↑ Back to the list" returns to the row (`model_list.FACTS_JS`). */
{_TAB} #shep-dx-change .shep-model-body {{ gap: {DRAWER_GAP}px !important; align-items: flex-start !important; flex-wrap: nowrap !important; }}
{_TAB} #shep-dx-change .shep-model-body > .shep-model-rows {{ flex: 3 1 0 !important; min-width: 0 !important; }}
{_TAB} #shep-dx-change .shep-model-body > .shep-model-facts {{ --shep-panel-min: {DRAWER_FACTS_MIN}px; flex: 2 1 0 !important; min-width: 0 !important; }}
{_TAB} #shep-dx-change .shep-model-foot {{ align-items: center !important; flex-wrap: wrap !important; gap: 8px 12px !important;
  border-top: 1px solid var(--shep-border); padding-top: 10px !important; }}
/* Load and Cancel keep their words' width, side by side; what Load does,
   two sentences, is a row of its own under them, from the foot's edge and at the measure, as
   Models says it (a message never shares a row with a button: it stood beside Load). */
{_TAB} #shep-dx-change .shep-model-foot > :not(#shep-dx-change-note) {{ flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; }}
{_TAB} #shep-dx-change .shep-model-foot > #shep-dx-change-note {{ flex: 1 1 100% !important; min-width: 0 !important; order: 1; }}
/* Load's reason (`model_list.load_reason_html`: every disabled Load says why) is a line of its
   own under Load and Cancel, before the note: beside Load it took two lines (a reason beside its
   button is one line or none), and wrapped between them it parted Cancel from Load (390 x 844). */
{_TAB} #shep-dx-change .shep-model-foot > #shep-models-clinical-reason {{ flex: 1 1 100% !important; min-width: 0 !important; order: 1; }}
{_TAB} #shep-dx-change-note .shep-note {{ display: block; max-width: var(--shep-measure); font-size: 14px; }}
{_TAB} #shep-dx-change .shep-model-confirm-box > .row > button {{ flex: 0 0 auto !important; white-space: nowrap; }}
{_TAB} #shep-dx-change .shep-model-list:has(.shep-model-confirm-box) > .shep-model-foot {{ display: none !important; }}
{_TAB} #shep-dx-change .shep-confirm .shep-ftbl th, {_TAB} #shep-dx-change .shep-confirm .shep-ftbl td {{ padding-top: 5px; padding-bottom: 5px; }}
{_TAB} #shep-dx-change .shep-confirm .shep-ftbl tbody th {{ white-space: nowrap; }}
{SIDE} {{
  {_TAB} #shep-dx {{ height: 100%; }}
  {_TAB} #shep-dx > .shep-panel {{ max-height: 100%; }}
  {_TAB} .shep-dx-scroll {{ flex: 1 1 auto !important; min-height: 0; overflow: auto; }}
  {_TAB} #shep-dx-input-bd {{ display: flex; flex-direction: column; flex-wrap: nowrap !important; }}
  {_TAB} #shep-dx-input-bd > #shep-dx-fields {{ flex: 0 1 auto !important; min-height: 0; {FOLD_FLOOR}: {FIELDS_FLOOR}px; overflow: auto; flex-wrap: nowrap !important; }}
  {_TAB} #shep-dx-fields > *, {_TAB} #shep-dx-side > * {{ flex-shrink: 0 !important; }}
  /* The card folded (OPS-FIX-N1; `style.FOLD`): its long identity is in Details ›; the serving
     state with the data's chips, the task's model name and its run, on one line (beside the name
     where both fit; its whole words in its title), stay, with Details ›, Change model… and Stop
     inference. Its lines that say nothing now (no refusal, no Stop question, no stop notice) take
     no gap. */
  {_TAB} #shep-dx-card[{FOLDED_ATTR}] .shep-dx-card-bd {{ flex-flow: row wrap; column-gap: 6px; }}
  {_TAB} #shep-dx-card[{FOLDED_ATTR}] .shep-dx-card-bd > * {{ flex: 1 1 100%; min-width: 0; }}
  {_TAB} #shep-dx-card[{FOLDED_ATTR}] .shep-dx-card-bd > .shep-dx-name {{ flex: 0 1 auto; }}
  {_TAB} #shep-dx-card[{FOLDED_ATTR}] .shep-dx-card-bd > .shep-dx-sub {{ flex: 1 1 8em; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis; }}
  {_TAB} #shep-dx-card[{FOLDED_ATTR}] .shep-dx-sub .shep-fact {{ display: inline; }}
  {_TAB} #shep-dx-card[{FOLDED_ATTR}] > .shep-dx-card-actions {{ row-gap: 4px !important; margin-top: 0; }}
  {_TAB} #shep-dx-card[{FOLDED_ATTR}] > #shep-dx-change-why:not(:has(.shep-dx-why)),
  {_TAB} #shep-dx-card[{FOLDED_ATTR}] > #shep-dx-stop-row:not(:has(.shep-dx-stop-q, .shep-dx-why, button:not(.{VIEW_OFF}))),
  {_TAB} #shep-dx-card[{FOLDED_ATTR}] > #shep-dx-stop-notice:not(:has(.shep-dx-stop-said, .shep-dx-stop-mem, .shep-dx-why)) {{
    display: none !important; }}
  {_TAB} .shep-dx-drawer {{ position: absolute !important; top: 0; bottom: 0; z-index: 20; }}
  {_TAB} #shep-dx-identity {{ right: 0; width: max(600px, calc((100% - var(--dx-in-w) - 32px) * 4 / 9 + 16px)); }}
  {_TAB} #shep-dx-change {{ left: calc(var(--dx-in-w) + 16px); right: 0; }}
  {_TAB} #shep-dx-identity-bd {{ flex: 1 1 auto !important; min-height: 0; overflow: auto; }}
  {_TAB} #shep-dx-change-bd {{ flex: 1 1 auto !important; min-height: 0; display: flex; flex-direction: column; }}
  {_TAB} #shep-dx-change .shep-model-list {{ flex: 1 1 auto !important; min-height: 0; display: flex !important; flex-direction: column; }}
  {_TAB} #shep-dx-change .shep-model-body {{ flex: 1 1 0 !important; min-height: 120px; overflow: hidden; align-items: stretch !important;
    container-type: inline-size; }}
  {_TAB} #shep-dx-change .shep-model-body > .shep-model-rows,
  {_TAB} #shep-dx-change .shep-model-body > .shep-model-facts {{
    min-height: 0; max-height: 100%; overflow: auto !important; overscroll-behavior: contain;
  }}
  {_TAB} #shep-dx-change .shep-model-body > .shep-model-rows {{ min-width: min-content !important; }}
  {_TAB} #shep-dx-change .shep-model-body > .shep-model-facts {{ min-width: var(--shep-panel-min) !important; }}
  /* The tree's header row sticks to the top of the rows' own region (no box scrolling inside it);
     the tree's box at most the room the details leave (`DRAWER_TREE_ROOM`). */
  {_TAB} #shep-dx-change .shep-model-rows .shep-tbl-wrap {{ overflow: visible; max-width: {DRAWER_TREE_ROOM}; }}
  {_TAB} #shep-dx-change .shep-model-confirm-box {{ flex: 0 1 auto !important; min-height: 0; overflow: hidden; }}
  {_TAB} #shep-models-clinical-confirm {{ flex: 0 1 auto !important; min-height: 0; overflow: auto !important; }}
  {_TAB} #shep-dx-change .shep-model-confirm-box > .row {{ flex: none !important; }}
  {_TAB} #shep-dx-change .shep-model-list:has(.shep-model-confirm-box) > .shep-model-body {{ min-height: 0 !important; }}
}}
{SIDE_SHORT} {{
  {_TAB} #shep-dx-change .shep-model-list:has(.shep-model-confirm-box) > .shep-model-body {{ display: none !important; }}
}}
{STACK} {{
  {_TAB} #shep-dx-input, {_TAB} #shep-dx-results, {_TAB} #shep-dx-details {{ flex: 1 1 auto !important; width: 100% !important; max-width: none !important; }}
  /* Stacked, the fields take two columns: the HPO terms (and the count under them) in the first,
     Patient ID and the options beside them in the second (the owner's decision of 2026-11-23 on
     portrait windows: the count's empty block took the second column's first cell, so Patient ID
     and the options fell under the terms and half the panel stood empty). */
  {_TAB} .shep-dx-fields {{ display: grid !important; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 12px 18px !important;
    grid-auto-flow: row dense; align-content: start; }}
  {_TAB} .shep-dx-fields > :not(#shep-dx-side) {{ grid-column: 1; }}
  {_TAB} .shep-dx-fields > #shep-dx-side {{ grid-column: 2; grid-row: 1 / span 2; align-self: start; }}
  {_TAB} #shep-dx-change .shep-model-body {{ flex-direction: column !important; }}
  {_TAB} #shep-dx-change .shep-model-body > * {{ width: 100% !important; flex: none !important; }}
}}
"""


__all__ = [
    "ANSWER_GATE_CSS",
    "CLOSE_WHILE_LOADING",
    "DX_IN_W",
    "EXPORTS_HIDDEN",
    "FIELD_HELP",
    "INPUTS_CHANGED",
    "INPUTS_GEN_JS",
    "INPUTS_HEAD",
    "PAGE_CSS",
    "SEGMENTS",
    "STOP_FOCUS_HEAD",
    "TASK_OUTPUTS",
    "TASK_VIEWS",
    "TOPIC_HELP",
    "answer_mark",
    "answered_inputs",
    "asks_explanations",
    "badge_html",
    "card_outputs",
    "cohort_choices",
    "cohort_info",
    "cohort_picked",
    "cohort_refusal",
    "compare_with_served",
    "create_diagnosis_tab",
    "details_outputs",
    "diagnose",
    "inputs_generation_of",
    "open_change",
    "poll_card",
    "reference_line_html",
    "refresh_card",
    "result_digests_html",
    "result_identity_html",
    "results_body_html",
    "run_controls",
    "task_details_html",
    "task_view",
]
