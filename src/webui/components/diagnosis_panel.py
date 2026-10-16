"""
SHEPHERD-Advanced Diagnosis Panel Component
=============================================
Gradio UI component for clinician-facing rare disease diagnosis (Tab 2).

Module: src/webui/components/diagnosis_panel.py

Purpose:
    Provide the Inference Testing tab with:
    - HPO phenotype input (manual entry with validation)
    - Top-K and options configuration
    - Ranked disease candidate display with scores
    - Evidence panel showing Mode A (direct path) or Mode B (analogy-based)
    - Confidence labels (Strong/Weak/Analogy/Insufficient) as colored badges
    - What produced each result (the served identity, contract M3c): a summary
      line and a full identity table under the results, banners for a synthetic
      workspace, a missing model or the demo answer, and the same identity in
      the CSV and Markdown exports and in the model status
    - The loaded model's recorded test results (Mode C) under the model status,
      read from the read-only `/api/v1/evaluations/loaded-model` and shown only
      for the checkpoint the status names

Architecture Note:
    This component calls the /api/v1/diagnose endpoint via HTTP (requests),
    NOT by importing pipeline directly. This keeps the Gradio UI decoupled
    from the inference layer and ensures the same code path as external
    API clients.

Dependencies:
    - gradio: UI components
    - requests: HTTP client for API calls

Version: 1.0.0
"""
from __future__ import annotations

import csv
import io
import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import requests

from src.evaluation.caveats import (
    MODE_C_RESULT_LABEL,
    PRE_B1_NOTE,
    R10_LABEL_TEXT,
    R10_STATE_TEXT,
    RECORD_ABSENCE,
    RELATION_DISPLAY_NAMES,
)
from src.utils.server_address import ServerAddressUnknownError, api_base_url

logger = logging.getLogger(__name__)


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


# Shown when the pipeline loads but the GNN did not. This is not cosmetic: the
# ranking then comes from path reasoning over the knowledge graph alone, so a
# disease with no qualifying path cannot appear at all, and the confidence
# figures are on a different footing from the GNN-fused ones. The system has
# previously served in this state for a whole session without anyone noticing,
# because the only signals were a tick turning into a cross and one INFO line.
GNN_UNAVAILABLE_TOAST = (
    "GNN not loaded — running in path-reasoning mode. Rankings come from "
    "knowledge-graph paths only, without GNN embedding scores. Check the "
    "checkpoint path, and the runtime status at the bottom of the page."
)
GNN_UNAVAILABLE_BANNER = (
    "🔴 **Running WITHOUT the GNN — path-reasoning mode.** Rankings come from "
    "knowledge-graph paths only; diseases with no qualifying path are not scored, "
    "and confidence values are not comparable with GNN-fused ones. "
    "Check the checkpoint path and the runtime line at the bottom of the page.\n"
)

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

# Confidence label → (emoji, CSS color)
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
    patient_id: str = "",
    top_k: int = 10,
    include_explanations: bool = True,
) -> Dict[str, Any]:
    """Call the /api/v1/diagnose endpoint and return parsed JSON."""
    payload = {
        "patient_id": patient_id or "webui_patient",
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
        return {"error": f"API error: {e.response.status_code} — {e.response.text}"}
    except Exception as e:
        return {"error": f"Unexpected error: {e}"}


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
# Formatting helpers
def _api_get(path: str) -> Tuple[Any, Optional[str]]:
    """One GET to this server's API: ``(body, None)``, or ``(None, reason)``.

    The reason is the real one, so an answer that did arrive -- an HTTP error, a
    body that is not JSON -- is not reported as an API that could not be reached.
    Shared by the status read and the recorded-results read, so the two cannot
    describe one failure two ways.
    """
    try:
        resp = _self_request("GET", path, timeout=5)
        resp.raise_for_status()
        return resp.json(), None
    except ServerAddressUnknownError as e:
        return None, str(e)
    except requests.ConnectionError:
        # ConnectTimeout is a ConnectionError too: the connection was never made.
        return None, "API not reachable"
    except requests.Timeout:
        # Connected, but no answer in time -- a reload holds the server meanwhile.
        return None, "API did not answer within 5 s"
    except requests.HTTPError as e:
        return None, f"API error {e.response.status_code}"
    except requests.exceptions.InvalidJSONError:
        # Narrower than ValueError, which requests also uses for a malformed URL
        # or header -- a request that was never sent is not a reply.
        return None, "the API's reply was not JSON"
    except Exception as e:
        return None, f"unexpected error: {e}"


def _get_pipeline_status() -> Dict[str, Any]:
    """Get pipeline status from API.

    A failure carries its real reason (`_api_get`): the status line shows it
    instead of "Pipeline not loaded".
    """
    body, error = _api_get("/api/v1/pipeline/status")
    if error is not None:
        return {"initialized": False, "error": error}
    return body


def _reload_pipeline(
    data_dir: str, checkpoint_path: str, conv_type: str = "auto"
) -> Dict[str, Any]:
    """Reload pipeline via API."""
    payload: Dict[str, Any] = {"data_dir": data_dir}
    if checkpoint_path and checkpoint_path.strip():
        payload["checkpoint_path"] = checkpoint_path.strip()
    if conv_type:
        payload["conv_type"] = conv_type
    try:
        resp = _self_request("POST", "/api/v1/pipeline/reload", json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json()
    except ServerAddressUnknownError as e:
        return {"success": False, "message": str(e)}
    except requests.ConnectionError:
        return {"success": False, "message": "API server not reachable."}
    except Exception as e:
        return {"success": False, "message": str(e)}


CONFIG_FILE = Path(".shepherd_ui_config.json")
DEFAULT_WORKSPACE = "data/workspaces/default"


def _load_saved_config() -> Dict[str, Any]:
    """Load saved UI config directly from file (not HTTP — avoids startup race)."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"data_dir": DEFAULT_WORKSPACE, "checkpoint_path": None}


def _save_config_to_file(data_dir: str, checkpoint_path: str) -> str:
    """Save current path config directly to file."""
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump({"data_dir": data_dir, "checkpoint_path": checkpoint_path or None}, f, indent=2)
        return "Configuration saved."
    except Exception as e:
        return f"Failed to save: {e}"


#: `checkpoint_meta` entries that describe the run rather than measure the model.
#: Everything else in it is a training-log metric the pipeline chose to copy
#: (`src/inference/pipeline.py:CHECKPOINT_LOG_METRICS`), shown in the order it
#: arrives, so the choice of metrics lives in one place.
_CHECKPOINT_DESCRIPTIVE_KEYS = ("epoch", "params", "device")

#: Shown until the first status read after the page loads. The status is not read
#: while the app is being built: that happens before the server is serving, so
#: what it found was stale by the time anyone looked.
STATUS_CHECKING = "⏳ Checking pipeline status…"


def _checkpoint_metric_label(key: str) -> str:
    """``val_mrr`` -> ``Val MRR``; ``val_hits@10`` -> ``Val Hits@10``."""
    return " ".join("MRR" if w == "mrr" else w[:1].upper() + w[1:] for w in key.split("_"))


def _format_pipeline_status(status_data: Dict[str, Any]) -> str:
    """Format pipeline status as Markdown."""
    if not status_data.get("initialized", False):
        error = status_data.get("error")
        if error:
            # The status could not be read at all. "Not loaded" here would be a
            # guess shown as a fact -- it is how a tab calling the wrong port
            # reported a serving pipeline as unloaded.
            return f"⚠️ **Pipeline status unavailable:** {error}"
        return "⚪ **Pipeline not loaded.** Configure paths below and click Load / Reload Pipeline."

    gnn_ready = bool(status_data.get("gnn_ready"))
    gnn = "✅" if gnn_ready else "❌"
    sp = "✅" if status_data.get("sp_ready") else "❌"
    mode = status_data.get("scoring_mode", "not reported")
    kg_n = status_data.get("kg_nodes", 0)
    kg_e = status_data.get("kg_edges", 0)
    # What the loaded pipeline serves (contract M3c). It describes the pipeline
    # loaded now, which a reload can replace; a result's own identity is shown
    # beside that result, never taken from here.
    identity = status_data.get("served_identity")
    if not isinstance(identity, dict):
        identity = None

    lines = []
    if not gnn_ready:
        # A tick that has turned into a cross is too quiet for this: without the
        # GNN the ranking comes from path reasoning alone, which is a different
        # and narrower answer. The toast on load says so once; this says so for
        # as long as it is true, so it survives a tab change, a refresh, or a
        # screenshot pasted into a note.
        lines.append(GNN_UNAVAILABLE_BANNER)
    if identity is not None and _is_synthetic(identity):
        lines.append(_identity_banner_md(_IDENTITY_BANNERS["synthetic"]))
    lines += [
        f"🟢 **Pipeline loaded** | Mode: `{mode}`",
        f"- GNN: {gnn} | SP: {sp} | KG: {kg_n} nodes, {kg_e} edges",
    ]

    # Checkpoint training info
    ckpt_meta = status_data.get("checkpoint_meta", {})
    if ckpt_meta:
        epoch = ckpt_meta.get("epoch")
        params = ckpt_meta.get("params")
        device = ckpt_meta.get("device", "?")
        meta_parts = []
        if epoch is not None:
            meta_parts.append(f"Epoch {epoch}")
        if params is not None:
            meta_parts.append(f"{params:,} params")
        meta_parts.append(f"device={device}")
        # Training-log metrics, as the pipeline copied them
        for key, val in ckpt_meta.items():
            if key in _CHECKPOINT_DESCRIPTIVE_KEYS:
                continue
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                meta_parts.append(f"{_checkpoint_metric_label(key)}: {val:.4f}")
        lines.append(f"- Checkpoint: {' | '.join(meta_parts)}")

    if identity is not None:
        lines.append(f"- {_identity_summary_md(_SERVED, identity, lead='Serving')}")
        # **The boundary and its qualifier, by the identity block's rule.** A
        # path-only or in-memory pipeline has nothing marked ✓, and the §6 line
        # alone would then read as a claim that its graph file was checked
        # against a record; `_claim_boundary_lines` adds what the report and
        # the tab add for the same identity.
        for sentence in _claim_boundary_lines(_SERVED, identity,
                                              IDENTITY_NOTHING_CHECKED_LOADED):
            lines += ["", f"<sub>{sentence}</sub>"]
    else:
        lines.append(
            "- **Serving:** not reported — the API's status did not say what the "
            "loaded pipeline serves."
        )

    # **Structural fingerprint warnings, not identity.** `verify_fingerprint`
    # compares the node types, node counts, edge-type count and feature sizes the
    # checkpoint recorded with the served graph's. Which graph the model was
    # trained on is a different question, settled by digest before the model is
    # built (contract M2.4) and stated in the served identity above; this heading
    # used to read "KG version mismatch", which a verified graph digest beside it
    # contradicted.
    fp_warns = status_data.get("fingerprint_warnings", [])
    if fp_warns:
        lines.append("")
        lines.append(
            f"⚠️ **Graph structure differs from the checkpoint's structural "
            f"fingerprint** ({len(fp_warns)} warning{'s' if len(fp_warns) != 1 else ''}; "
            "node types and counts, edge-type count and feature sizes compared, "
            "not the graph's identity):"
        )
        for w in fp_warns[:3]:
            lines.append(f"- {w}")

    return "\n".join(lines)


# =============================================================================
# Formatting helpers — diagnosis results
# =============================================================================
def _format_results_table(candidates: List[Dict]) -> str:
    """Format candidates as a Markdown table for the results area."""
    if not candidates:
        return "*No candidates found.*"

    rows = []
    for c in candidates:
        label = c.get("confidence_label") or "—"
        emoji, _ = LABEL_STYLES.get(label, ("", "#888"))
        conf = c.get("confidence_score") or 0
        gnn = c.get("gnn_score") or 0
        sp = c.get("sp_score") or 0
        rows.append(
            f"| {c['rank']} | **{c['disease_name']}** | "
            f"{conf:.3f} | {gnn:.3f} | {sp:.3f} | "
            f"{emoji} {label} |"
        )

    header = (
        "| Rank | Disease | Confidence | GNN | SP | Evidence |\n"
        "|:----:|---------|:----------:|:---:|:--:|----------|\n"
    )
    return header + "\n".join(rows)


def _format_evidence_detail(candidate: Dict) -> str:
    """Format evidence package for a single selected candidate."""
    pkg = candidate.get("evidence_package")
    if not pkg:
        return "*No evidence package available. Run with `include_explanations=true`.*"

    mode = pkg.get("mode") or "unknown"
    summary = pkg.get("summary") or ""
    label = pkg.get("confidence_label") or "—"
    emoji, color = LABEL_STYLES.get(label, ("", "#888"))

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
            sim = a.get("embedding_similarity") or 0
            name = a.get("similar_gene_name") or "?"
            shared = a.get("shared_features", [])
            known_paths = a.get("known_paths", [])
            parts.append(
                f"- **{name}** (similarity: {sim:.2f})"
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
# **One reading of the identity for every surface.** The report, the tab's
# identity block and summary line, the model status and the CSV all read the
# `served_identity` the API copied from the result it answered with (the status
# reads the loaded pipeline's). The table rows are built once, by
# `_identity_rows`, so the report and the tab cannot describe one result two
# ways. Every part states its state in the contract's words (verified,
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
#: The same, said of the loaded pipeline: the model status describes what is
#: loaded now, not a result.
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
#: README's statement, carried by every report and identity block.
NOT_A_MEDICAL_DEVICE = (
    "Not a certified medical device; decision support for a qualified clinician."
)

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
_IDENTITY_BANNERS = {
    IDENTITY_DEMO: (
        "DEMO ANSWER: not computed.",
        "No pipeline is configured on this server, so these candidates are "
        "illustrative, not computed from any model, graph or table. Nothing here "
        "is a diagnosis.",
    ),
    IDENTITY_NOT_REPORTED: (
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
    "no_model": (
        "NO TRAINED MODEL: path-reasoning fallback.",
        "No checkpoint is served, so candidates are ranked by path reasoning over "
        "the knowledge graph alone: a disease with no qualifying path is not "
        "scored, and scores are not comparable with GNN scores.",
    ),
    "no_sp": (
        "NO SHORTEST-PATH TABLE: GNN score alone.",
        "A trained model is served without a shortest-path table, so each score is "
        "the GNN's alone (scoring mode gnn_only, mixing weight eta 1.0) and differs "
        "from one computed with the table.",
    ),
}


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


def _md_short(value: Any) -> str:
    """The first `_SHORT_HEX` characters of a digest or identifier, for a
    summary line (never a table cell)."""
    if _is_sha256(value):
        return f"`sha256:{value[:_SHORT_HEX]}…`"
    text = _plain_value(value)
    return _md_code(text[:_SHORT_HEX] + "…" if len(text) > _SHORT_HEX else text, in_table=False)


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


#: What a dataset record's origin means, wherever the panel names one: the
#: served identity and the recorded results.
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

    The one rule for every surface that shows the boundary: the report and the
    tab's identity block (through `_identity_table_lines`) and the model status,
    which passes the sentence worded for the loaded pipeline.
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


def _identity_banner_md(banner: Tuple[str, str]) -> str:
    """A banner for the tab, built like `GNN_UNAVAILABLE_BANNER`: the words carry
    it, the icon only draws the eye."""
    headline, detail = banner
    return f"⚠️ **{headline}** {detail}\n"


def _identity_summary_md(
    kind: str, identity: Optional[Dict[str, Any]], lead: str = "Produced by"
) -> str:
    """One line naming what produced a result, with short digests.

    The full values are in the identity table (`_identity_rows`); this line is
    for reading at a glance, never for copying a digest from.
    """
    if kind == IDENTITY_DEMO:
        return (f"**{lead}:** demo answer — no pipeline is configured; these "
                "candidates were not computed.")
    if kind != _SERVED or identity is None:
        return (f"**{lead}:** not reported — the API's response did not say what "
                "produced this result.")

    checkpoint = _field(identity, "checkpoint")
    graph = _field(identity, "graph")
    shortest_paths = _field(identity, "shortest_paths")
    record = _field(identity, "dataset_record")
    mode = _field(identity, "scoring_mode")
    device = _field(_field(identity, "settings"), "device")

    parts = []
    state = _field(checkpoint, "state")
    if state == "verified":
        parts.append(f"checkpoint {_md_short(_field(checkpoint, 'sha256'))} {_VERIFIED_MARK}")
    elif state == "supplied_in_memory":
        parts.append("model supplied in memory (no checkpoint)")
    elif state == "absent":
        parts.append("no trained model (checkpoint absent)")
    else:
        parts.append("checkpoint not reported")

    state = _field(graph, "state")
    if state in ("verified", "unverified"):
        mark = _VERIFIED_MARK if state == "verified" else "unverified"
        parts.append(f"graph {_md_short(_field(graph, 'kg_sha256'))} {mark}")
    elif state == "in_memory":
        parts.append("graph in memory (no file)")
    else:
        parts.append("graph not reported")

    state = _field(shortest_paths, "state")
    if state == "verified":
        parts.append(f"SP table {_md_short(_field(shortest_paths, 'build_id'))} {_VERIFIED_MARK}")
    elif state == "absent":
        parts.append("no SP table")
    else:
        parts.append("SP table not reported")

    parts.append(_md_code(mode, in_table=False) if isinstance(mode, str)
                 else "scoring mode not reported")
    if isinstance(device, str):
        parts.append(_md_code(device, in_table=False))

    state = _field(record, "state")
    if state == "verified":
        parts.append(f"dataset {_md_text(_field(record, 'origin'))} {_VERIFIED_MARK}")
    elif state == "absent":
        parts.append("no dataset record")
    else:
        parts.append("dataset record not reported")
    return f"**{lead}:** " + " · ".join(parts)


def _identity_details_md(kind: str, identity: Optional[Dict[str, Any]]) -> str:
    """The full identity table in a closed block under the result (the tab)."""
    return "\n".join([
        "<details><summary>What produced this result — full identity</summary>",
        "",
        *_identity_table_lines(kind, identity),
        "",
        "</details>",
    ])


# =============================================================================
# Recorded test results for the loaded model (PLAN_TEST_RESULTS D8; M4)
# =============================================================================
# **Read from the API, worded by the caveats.** The records come from
# `GET /api/v1/evaluations/loaded-model`, never from the ledger file: this tab
# reads nothing in its own process (architecture note above). What every record
# says beside its numbers -- the Mode C label, the B-1 note, R10's label names,
# states and definitions, what an empty ledger means, the relation names -- is the
# text in `src/evaluation/caveats.py`, rendered and never reworded here. The block
# is appended to the model status Markdown (`_status_with_recorded_results`), so
# the load and reload events keep their one output component.

#: The block's heading.
RECORDED_RESULTS_HEADING = "Recorded test results (Mode C)"
#: Said instead of the records when they describe another checkpoint than the
#: status read just before them: a reload landed between the two reads.
RECORDS_MODEL_CHANGED = (
    "The loaded model changed while its recorded results were read; refresh the page."
)
#: The states the API answers with (`src/api/routes/evaluations.py`).
_RECORD_STATES = (
    "no_model", "not_identified", "no_ledger", "ledger_not_read", "no_records", "records",
)
_R10_EVIDENCE_WORDS = {
    "found": "found",
    "not_found": "not found",
    "unreadable": "found, not readable",
    "not_recorded": "not recorded by the checkpoint",
}
#: A relation named by its contract number (R1, R2, ...), shown with it.
_NUMBERED_RELATION_RE = re.compile(r"R\d+")


def _get_loaded_model_evaluations() -> Dict[str, Any]:
    """The loaded model's recorded results, from one GET.

    Any failure is ``{"state": "unavailable", "error": <reason>}``, with the
    reason `_api_get` gives: the records are read beside the model status, and
    failing to read them must never become a failure of the status.
    """
    body, error = _api_get("/api/v1/evaluations/loaded-model")
    if error is None and not isinstance(body, dict):
        error = "the API's reply was not an object"
    if error is not None:
        return {"state": "unavailable", "error": error}
    return body


def _served_checkpoint_sha(status_data: Dict[str, Any]) -> Optional[str]:
    """The checkpoint digest a status names as served, or ``None`` where it
    names none (no model, a model supplied in memory, no identity reported)."""
    checkpoint = _field(_field(status_data, "served_identity"), "checkpoint")
    sha = _field(checkpoint, "sha256")
    if _field(checkpoint, "state") == "verified" and _is_sha256(sha):
        return sha
    return None


def _record_text(value: Any) -> str:
    """A value from the records reply as Markdown-safe text; "not reported" for
    one the reply left out."""
    return "not reported" if value is _MISSING else _md_text(value)


def _metric_order(name: str) -> Tuple[int, int, str]:
    """MRR, mean rank, Hits@k by k, then anything else by name."""
    if name == "untruncated_mrr":
        return (0, 0, name)
    if name == "mean_rank":
        return (1, 0, name)
    k = name[len("hits@"):]
    if name.startswith("hits@") and k.isdigit():
        return (2, int(k), name)
    return (3, 0, name)


def _mode_c_metric_label(name: str) -> str:
    """``untruncated_mrr`` -> ``Mode C MRR (untruncated)``; ``hits@10`` ->
    ``Mode C Hits@10``.

    **Always "Mode C".** The Checkpoint line above shows the trainer's "Val MRR",
    a top-20 metric on the selection split; a recorded result read under the same
    name would be quoted as that number, or that number as this one.
    """
    if name == "untruncated_mrr":
        return "Mode C MRR (untruncated)"
    if name == "mean_rank":
        return "Mode C mean rank"
    if name.startswith("hits@"):
        return f"Mode C Hits@{_md_text(name[len('hits@'):])}"
    return f"Mode C {_md_text(name)}"


def _metrics_text(metrics: Any) -> str:
    if not isinstance(metrics, dict) or not metrics:
        return "no metrics recorded"
    parts = []
    for name in sorted((str(key) for key in metrics), key=_metric_order):
        value = metrics[name]
        number = isinstance(value, (int, float)) and not isinstance(value, bool)
        parts.append(f"{_mode_c_metric_label(name)} "
                     f"{f'{value:.4f}' if number else _md_text(value)}")
    return " · ".join(parts)


def _coverage_text(record: Any) -> str:
    """Ranked of source, and every exclusion with its count (D6): the
    denominators a coverage figure needs."""
    counts = _field(record, "counts")
    excluded = _field(counts, "excluded")
    if isinstance(excluded, dict):
        reasons = " · ".join(
            f"{_md_text(str(name).replace('_', ' '))} {_record_text(count)}"
            for name, count in excluded.items()
        ) or "none"
    else:
        reasons = "not reported"
    return (
        f"Coverage: {_record_text(_field(record, 'n_ranked'))} of "
        f"{_record_text(_field(counts, 'source_cases'))} source cases ranked "
        f"({_record_text(_field(counts, 'ranked_eligible'))} eligible) · "
        f"{_record_text(_field(counts, 'n_excluded'))} excluded: {reasons}"
    )


def _relations_text(relations: Any) -> str:
    """Each relation's state on the bytes the measurement read: ✓ where it was
    verified, "not applicable" where it does not apply to this kind of cohort."""
    if not isinstance(relations, dict):
        return "not reported"
    names = [*(name for name in RELATION_DISPLAY_NAMES if name in relations),
             *(name for name in relations if name not in RELATION_DISPLAY_NAMES)]
    parts = []
    for name in names:
        display = RELATION_DISPLAY_NAMES.get(name)
        if display is None:
            label = _md_text(name)
        elif _NUMBERED_RELATION_RE.fullmatch(name):
            label = f"{name} {display}"
        else:
            label = display
        state = relations[name]
        if state == "verified":
            parts.append(f"{label} {_VERIFIED_MARK}")
        elif state == "not_applicable":
            parts.append(f"{label}: not applicable")
        else:
            parts.append(f"{label}: {_md_text(state)}")
    return " · ".join(parts) or "none reported"


def _r10_label_text(label: str, name: str, part: Any) -> str:
    """One R10 label: its state in the caveats' words, and its count with the
    denominator only where the state has one.

    **"unverifiable" and "no validation inputs recorded" are words, never a
    number:** neither a 0 that would read as "no overlap" nor a denominator that
    would read as a count.
    """
    state = _field(part, "state")
    words = R10_STATE_TEXT.get(state) if isinstance(state, str) else None
    if words is None:
        return f"{name}: not reported"
    if state not in ("overlap", "none"):
        return f"{name}: {words}"
    unit = "distinct test diseases" if label == "disease_overlap" else "test cases"
    count = f"{_record_text(_field(part, 'k'))} of {_record_text(_field(part, 'n'))} {unit}"
    if label == "disease_overlap":
        count += f"; {_record_text(_field(part, 'cases'))} test cases with one of them"
    return f"{name}: {words} — {count}"


def _r10_scope_text(scope: str, labels: Dict[str, Dict[str, str]], part: Any) -> str:
    evidence = _field(_field(part, "evidence"), "state")
    found = _R10_EVIDENCE_WORDS.get(evidence) if isinstance(evidence, str) else None
    return f"R10, {scope} inputs (evidence {found or 'not reported'}): " + " · ".join(
        _r10_label_text(label, text["name"], _field(part, label))
        for label, text in labels.items()
    )


def _environment_text(record: Any) -> str:
    text = _record_text(_field(record, "environment_class"))
    gpu = _field(record, "gpu")
    if isinstance(gpu, dict):
        name, driver = _field(gpu, "name"), _field(gpu, "driver")
        text += (f" · GPU {_md_text(name) if isinstance(name, str) else 'name not read'}, "
                 f"driver {_md_text(driver) if isinstance(driver, str) else 'not read'}")
    return text


def _dataset_origin_text(origin: Any) -> str:
    meaning = _DATASET_ORIGIN_MEANING.get(origin) if isinstance(origin, str) else None
    if meaning is None:
        return _record_text(origin)
    # Called out, as the synthetic banner calls out the served workspace.
    word = f"**{origin}**" if origin == "synthetic" else origin
    return f"{word} — {meaning}"


def _registration_text(record: Any) -> str:
    registration = _field(record, "registration")
    n = _field(record, "n_corroborating")
    if isinstance(n, int) and not isinstance(n, bool):
        corroborating = ("no corroborating report" if n == 0
                         else f"{n} corroborating report{'s' if n != 1 else ''}")
    else:
        corroborating = "corroborating reports not reported"
    report = _field(record, "source_artifact_digest")
    return (
        f"Registered #{_record_text(_field(registration, 'sequence'))} at "
        f"{_record_text(_field(registration, 'registered_at'))} · {corroborating} · "
        f"from report {'not reported' if report is _MISSING else _md_short(report)}"
    )


def _recorded_result_lines(record: Any) -> List[str]:
    """One record: the cohort role and version, the Mode C metrics over the cases
    ranked, then coverage, relations, R10, environment, dataset and registration."""
    cohort = _field(record, "cohort")
    lines = [
        f"- **{_record_text(_field(cohort, 'role'))} · "
        f"{_record_text(_field(cohort, 'label'))}** — "
        f"{_metrics_text(_field(record, 'metrics'))}, over "
        f"{_record_text(_field(record, 'n_ranked'))} ranked cases",
        f"  - {_coverage_text(record)}",
        f"  - Relations: {_relations_text(_field(record, 'relations'))}",
    ]
    r10 = _field(record, "r10")
    lines += [f"  - {_r10_scope_text(scope, labels, _field(r10, scope))}"
              for scope, labels in R10_LABEL_TEXT.items()]
    lines += [
        f"  - Environment: {_environment_text(record)}",
        f"  - Dataset: {_dataset_origin_text(_field(record, 'dataset_origin'))}",
        f"  - {_registration_text(record)}",
    ]
    return lines


def _recorded_records_lines(evals: Dict[str, Any]) -> List[str]:
    """The "records" state: the label and note, each record shown, the count of
    those not shown, and once for the block the R10 definitions and the §6 line."""
    records = _field(evals, "records")
    records = records if isinstance(records, list) else []
    not_shown = _field(evals, "records_not_shown")
    lines: List[str] = []
    if records:
        lines += [f"{MODE_C_RESULT_LABEL}. {PRE_B1_NOTE}", ""]
        for record in records:
            lines += _recorded_result_lines(record)
    if isinstance(not_shown, int) and not isinstance(not_shown, bool) and not_shown > 0:
        many = not_shown != 1
        lines += ["", (
            f"{not_shown} record{'s' if many else ''} for these weights "
            f"{'are' if many else 'is'} not shown: {'they do' if many else 'it does'} "
            "not have the shape registration writes (the server's log says why)."
        )]
    if records:
        definitions = " · ".join(
            f"{text['name']}: {text['definition']}"
            for labels in R10_LABEL_TEXT.values() for text in labels.values()
        )
        # Once for the block, not per record: the boundary is the same claim for
        # every record, and the relations it refers to are the ones marked ✓.
        lines += ["", f"<sub>R10 labels. {definitions}.</sub>",
                  "", f"<sub>{IDENTITY_CLAIM_BOUNDARY}</sub>"]
    return lines


def _format_recorded_results(evals: Dict[str, Any], served_sha: Optional[str]) -> str:
    """The recorded-results block for the model status, in Markdown.

    ``served_sha`` is the checkpoint digest of the status read just before
    (`_served_checkpoint_sha`). **The block is shown only for that digest:** the
    API echoes the digest it filtered the ledger by, and an answer about another
    checkpoint -- a reload landed between the two reads -- is said to be one,
    never shown under this status.
    """
    heading = f"**{RECORDED_RESULTS_HEADING}**"
    state = _field(evals, "state")
    if state == "unavailable":
        return f"⚠️ {heading} could not be read: {_record_text(_field(evals, 'error'))}"
    echoed = _field(evals, "checkpoint_sha256")
    if state not in _RECORD_STATES or echoed is _MISSING:
        return f"⚠️ {heading}: the API's reply was not in a shape this page reads."
    if echoed != served_sha:
        return f"⚠️ {heading}: {RECORDS_MODEL_CHANGED}"
    if state == "no_model":
        return (f"{heading}: no trained model is loaded, so there are no recorded "
                "results to look up.")
    if state == "not_identified":
        return (f"{heading}: the loaded model was supplied in memory, not read from a "
                "checkpoint file, so it has no digest to look recorded results up by.")

    ledger = _field(evals, "ledger")
    name = _field(ledger, "file")
    file = _md_code(name, in_table=False) if isinstance(name, str) else "not reported"
    lines = [f"{heading} for checkpoint {_md_short(echoed)}", ""]
    if state == "no_ledger":
        lines.append(
            f"No ledger ({file}) is beside the loaded checkpoint, so no recorded result "
            "can be shown; that does not mean these weights were never evaluated."
        )
    elif state == "ledger_not_read":
        lines.append(
            f"The ledger ({file}) beside the loaded checkpoint was not read, so none of "
            f"its records are shown: {_record_text(_field(ledger, 'problem'))}"
        )
    elif state == "no_records":
        lines.append(RECORD_ABSENCE)
    else:
        lines += _recorded_records_lines(evals)
    return "\n".join(lines)


def _status_with_recorded_results(status_data: Dict[str, Any]) -> str:
    """The model status, with the loaded model's recorded results under it.

    The status is read first and the records second, and the block is shown only
    for the checkpoint the status names (`_format_recorded_results`). With no
    pipeline loaded, or no status read, nothing is looked up: there is no loaded
    model to have results, and no second error line to add.
    """
    rendered = _format_pipeline_status(status_data)
    if not status_data.get("initialized", False):
        return rendered
    block = _format_recorded_results(
        _get_loaded_model_evaluations(), _served_checkpoint_sha(status_data)
    )
    return f"{rendered}\n\n{block}"


# =============================================================================
# Export helpers — write ALL candidates (not just the previewed one) to a file
# =============================================================================
_CANDIDATE_CSV_COLUMNS = [
    "rank", "disease_id", "disease_name", "confidence_score", "confidence_label",
    "gnn_score", "sp_score", "reasoning_score", "evidence_mode", "evidence_summary",
    "min_path_length", "num_direct_paths", "num_analogies",
    "matching_phenotypes", "supporting_genes",
]

# **What produced the result, on every row** (contract M3c). Appended after the
# candidate columns, never between them, so a reader that takes columns by
# position is not broken; one header row and no preamble, so the file stays a
# plain table. Each column holds one atomic part, the same on every row.
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

_CSV_COLUMNS = _CANDIDATE_CSV_COLUMNS + _SERVED_CSV_COLUMNS


def _fmt_num(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else ""


#: A cell a spreadsheet would run as a formula starts with one of these (OWASP,
#: "CSV Injection").
_CSV_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


def _csv_text(value: Any) -> str:
    """A free-text cell, defused: one that a spreadsheet would run as a formula
    gets a leading apostrophe. Numeric columns are written by `_fmt_num` and
    never pass through here, so a negative score stays a number."""
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
    with what produced the result (`_SERVED_CSV_COLUMNS`) repeated on each row."""
    served = _served_csv_values(result)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for c in result.get("candidates", []):
        pkg = c.get("evidence_package") or {}
        writer.writerow({
            "rank": c.get("rank"),
            "disease_id": _csv_text(_clean_disease_id(c.get("disease_id", ""))),
            "disease_name": _csv_text(c.get("disease_name", "")),
            "confidence_score": _fmt_num(c.get("confidence_score")),
            "confidence_label": _csv_text(c.get("confidence_label") or ""),
            "gnn_score": _fmt_num(c.get("gnn_score")),
            "sp_score": _fmt_num(c.get("sp_score")),
            "reasoning_score": _fmt_num(c.get("reasoning_score")),
            "evidence_mode": _csv_text(pkg.get("mode") or ""),
            "evidence_summary": _csv_text((pkg.get("summary") or "").replace("\n", " ").strip()),
            "min_path_length": pkg.get("min_path_length", ""),
            "num_direct_paths": len(pkg.get("direct_paths") or []),
            "num_analogies": len(pkg.get("analogies") or []),
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
    lines = ["# SHEPHERD-Advanced Diagnosis Report", ""]
    for key, value in (
        ("Patient ID", result.get("patient_id")),
        ("Session ID", result.get("session_id")),
        ("Timestamp", result.get("timestamp")),
        ("Inference time (ms)", f"{result.get('inference_time_ms', 0):.1f}"),
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

    lines += ["", "## Ranked candidates", "", _format_results_table(candidates),
              "", "## Per-candidate detail"]
    for c in candidates:
        lines += [
            "",
            f"### #{c.get('rank')} — {c.get('disease_name')}",
            f"`{_clean_disease_id(c.get('disease_id', ''))}` · confidence "
            f"{_fmt_num(c.get('confidence_score')) or 'n/a'}",
            "",
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
# Event handlers
# =============================================================================
# DownloadButton update for the "no downloadable result" state.
_DOWNLOAD_DISABLED = gr.update(value=None, interactive=False)


def _on_diagnose(
    phenotype_text: str,
    patient_id: str,
    top_k: int,
):
    """
    Handle the Diagnose button click.

    Returns: (results_table, evidence_detail, explanation,
              candidate_dropdown_update, results_state,
              export_csv_update, export_report_update)

    The full API result is stashed in results_state so the candidate dropdown
    reuses it without re-running inference. Export files are generated up front
    here and set as the DownloadButtons' values, so a single click downloads
    them (avoids the gr.DownloadButton "download current value, then update"
    two-click behaviour).
    """
    hpo_ids = _parse_hpo_ids(phenotype_text)

    if not hpo_ids:
        return (
            "⚠️ **No valid HPO terms found.** Enter HPO ids like `HP:0001250` — "
            "one per line, or comma/space-separated; names are optional.",
            "",
            "",
            gr.update(choices=[], value=None),
            None,
            _DOWNLOAD_DISABLED,
            _DOWNLOAD_DISABLED,
        )

    # Call API
    result = _call_diagnose(
        phenotypes=hpo_ids,
        patient_id=patient_id,
        top_k=int(top_k),
        include_explanations=True,
    )

    if "error" in result:
        return (
            f"❌ **Error**: {result['error']}",
            "",
            "",
            gr.update(choices=[], value=None),
            None,
            _DOWNLOAD_DISABLED,
            _DOWNLOAD_DISABLED,
        )

    candidates = result.get("candidates", [])
    warnings = result.get("warnings", [])
    inference_ms = result.get("inference_time_ms", 0)

    # **What produced this result, attached to it** (contract M3c): the banners
    # above the table, a summary line under it and the full identity in a closed
    # block, all from the result's own `served_identity`. The model status
    # describes whatever is loaded now, which a reload can change; this cannot.
    kind, identity = _identity_of(result)
    banners_md = "".join(
        _identity_banner_md(banner) + "\n" for banner in _identity_banners(kind, identity)
    )

    # Build results table
    table_md = _format_results_table(candidates)
    meta = f"\n\n*{len(candidates)} candidates found in {inference_ms:.1f}ms*"
    meta += f"\n\n{_identity_summary_md(kind, identity)}"
    meta += f"\n\n{_identity_details_md(kind, identity)}"
    if warnings:
        meta += f"\n\n⚠️ Warnings: {'; '.join(warnings)}"

    # Build candidate selector dropdown
    choices = [
        f"#{c['rank']} — {c['disease_name']} ({c['confidence_score']:.3f})"
        for c in candidates
    ]

    # Show evidence for rank #1 by default
    evidence_md = _format_evidence_detail(candidates[0]) if candidates else ""
    explain_md = _format_full_explanation(candidates[0]) if candidates else ""

    # Stash the parsed query so an exported report records what was asked.
    result["_query_phenotypes"] = hpo_ids

    # Pre-generate the export files (single-click download). Export is secondary
    # to showing the diagnosis, so a write failure must never hide the results:
    # keep the downloads disabled and note it — re-running retries the export.
    export_note = ""
    if candidates:
        try:
            csv_path, md_path = _write_exports(result)
            csv_update = gr.update(value=csv_path, interactive=True)
            md_update = gr.update(value=md_path, interactive=True)
        except Exception as exc:  # noqa: BLE001 — never let export break the result
            logger.warning("Export file generation failed: %s", exc)
            csv_update = md_update = _DOWNLOAD_DISABLED
            export_note = (
                "\n\n⚠️ *Export files could not be written (disk/permissions?). "
                "Results are shown above; click 'Run Diagnosis' again to retry the "
                "export.*"
            )
    else:
        csv_update = md_update = _DOWNLOAD_DISABLED

    return (
        banners_md + table_md + meta + export_note,
        evidence_md,
        explain_md,
        gr.update(choices=choices, value=choices[0] if choices else None),
        result,
        csv_update,
        md_update,
    )


def _on_phenotype_change(result_state: Optional[Dict[str, Any]]):
    """Clear stale results/exports when the phenotype input is edited *after* a run.

    Only the phenotype list is watched — it defines the diagnosis, so editing it
    genuinely invalidates the shown/exported results. Patient ID (a per-keystroke
    textbox that doesn't affect the run) and Top-K are intentionally NOT watched.
    No-ops until there is a result to invalidate, so typing the first query
    doesn't disturb the placeholder. Output order matches ``_diagnosis_outputs``.
    """
    if not result_state:
        no_change = gr.update()
        return (no_change, no_change, no_change, no_change, None,
                no_change, no_change)
    return (
        "*Inputs changed — click 'Run Diagnosis' to refresh results.*",
        "",
        "",
        gr.update(choices=[], value=None),
        None,
        _DOWNLOAD_DISABLED,
        _DOWNLOAD_DISABLED,
    )


def _on_candidate_select(
    selection: str,
    result_state: Optional[Dict[str, Any]],
) -> Tuple[str, str]:
    """
    Handle candidate dropdown selection change.

    Reads the already-fetched result from ``result_state`` instead of re-calling
    the API, so switching candidates for detail view no longer re-runs inference.
    """
    if not selection or not result_state:
        return "", ""

    # Extract rank from "#{rank} — ..."
    try:
        rank = int(selection.split("#")[1].split("—")[0].strip())
    except (IndexError, ValueError):
        return "⚠️ Could not parse selection.", ""

    candidates = result_state.get("candidates", [])
    target = next((c for c in candidates if c.get("rank") == rank), None)
    if not target:
        return f"⚠️ Rank #{rank} not found in current results.", ""

    return _format_evidence_detail(target), _format_full_explanation(target)


def _on_reload_pipeline(
    data_dir: str, checkpoint_path: str, conv_type: str
) -> Tuple[str, str]:
    """Handle Reload Pipeline button click."""
    if not data_dir or not data_dir.strip():
        return "⚠️ Data directory is required.", ""

    result = _reload_pipeline(
        data_dir.strip(),
        checkpoint_path.strip() if checkpoint_path else "",
        (conv_type or "auto").strip().lower(),
    )

    status_md = ""
    if result.get("success"):
        status = result.get("status", {})
        # The new model's recorded results, or the stated absence of any: a
        # status naming the new checkpoint over the previous one's records
        # would be the one thing this block must never show.
        status_md = _status_with_recorded_results(status)
        if status.get("initialized") and not status.get("gnn_ready"):
            # Attached to the load action rather than to the status renderer:
            # the renderer also runs on the initial page render, and a toast that
            # repeats is a toast people learn to dismiss unread.
            gr.Warning(GNN_UNAVAILABLE_TOAST, duration=20)
    else:
        status_md = f"❌ {result.get('message', 'Unknown error')}"

    # Which checkpoint was actually loaded (auto must not be opaque).
    sel_path = result.get("checkpoint_path")
    sel_reason = result.get("selection_reason")
    if sel_path:
        status_md += f"\n\n**Loaded checkpoint:** `{sel_path}`"
        if sel_reason:
            status_md += f"  \n<sub>{sel_reason}</sub>"

    # Format file check results
    files = result.get("files_found", {})
    if files:
        file_lines = []
        for fname, exists in files.items():
            if isinstance(exists, bool):
                icon = "✅" if exists else "❌"
                file_lines.append(f"  {icon} {fname}")
        status_md += "\n\n**Files:**\n" + "\n".join(file_lines)

    msg = result.get("message", "")
    return status_md, msg


def _on_save_config(data_dir: str, checkpoint_path: str) -> str:
    """Handle Save Config button click."""
    return _save_config_to_file(data_dir, checkpoint_path)


def _on_reset_defaults() -> Tuple[str, str, str]:
    """Handle Reset Defaults button click."""
    return DEFAULT_WORKSPACE, "", "Paths reset to defaults."


def _on_load_status() -> str:
    """Read the pipeline status for a page that has just loaded.

    Wired to the Blocks' load event, so every page opened shows the server's
    status at that moment rather than what it was while the app was being built,
    with the loaded model's recorded results under it in the same Markdown.
    """
    status_data = _get_pipeline_status()
    return _status_with_recorded_results(status_data)


# =============================================================================
# Tab builder (called from app.py)
# =============================================================================
def create_diagnosis_tab(blocks: gr.Blocks) -> None:
    """
    Build the Diagnosis Panel tab inside a gr.Blocks context.

    ``blocks`` is the enclosing app, which the model status refreshes from on
    every page load (`_on_load_status`).

    Layout:
        Left column (input):
            - Phenotype text input (one per line)
            - Load Demo button
            - Patient ID, Top-K
            - Diagnose button
        Right column (results):
            - Ranked results table (Markdown)
            - Candidate selector dropdown
            - Evidence detail panel (Markdown)
            - Full explanation accordion (Markdown)
    """

    # Holds the full API result of the latest diagnosis so the candidate
    # dropdown and the export buttons reuse it (no re-running inference).
    results_state = gr.State(None)

    # === MODEL CONFIGURATION (top accordion) ===
    # Load saved config for default values
    saved_cfg = _load_saved_config()

    with gr.Accordion("Model Configuration", open=False):
        config_status_md = gr.Markdown(value=STATUS_CHECKING)

        with gr.Row():
            data_dir_input = gr.Textbox(
                label="Data Directory",
                value=saved_cfg.get("data_dir", "data/workspaces/default"),
                info="Folder containing kg.json, node_features.pt, edge_indices.pt, etc.",
            )
            checkpoint_input = gr.Textbox(
                label="Checkpoint Path (optional)",
                value=saved_cfg.get("checkpoint_path") or "",
                placeholder="(auto-detect from data directory)",
                info="Explicit .pt path. Leave blank to auto-select by architecture.",
            )
            arch_input = gr.Dropdown(
                label="Architecture",
                choices=["auto", "hgt", "gat", "sage"],
                value="auto",
                info="Which GNN to serve: the latest run under {data}/runs/<arch>/. "
                "'auto' = latest trained. Ignored if a checkpoint path is given.",
            )

        with gr.Row():
            reload_btn = gr.Button("🔄 Load / Reload Pipeline", variant="primary", size="sm")
            save_cfg_btn = gr.Button("💾 Save Config", variant="secondary", size="sm")
            reset_btn = gr.Button("↩️ Reset Defaults", variant="secondary", size="sm")

        config_msg = gr.Markdown(value="")

    # === MAIN LAYOUT ===
    with gr.Row():
        # === LEFT COLUMN: Input ===
        with gr.Column(scale=1):
            gr.Markdown("### Patient Phenotypes")

            phenotype_input = gr.Textbox(
                label="HPO Terms",
                placeholder=(
                    "HP:0001250 — Seizure\n"
                    "HP:0001263, HP:0002376   (commas / spaces are fine too)"
                ),
                lines=8,
                info=(
                    "Paste HPO ids in any format — one per line, or comma/space-"
                    "separated. Names and stray punctuation are ignored; only "
                    "HP:####### ids are used."
                ),
            )

            clear_btn = gr.ClearButton(
                components=[phenotype_input],
                value="🗑️ Clear",
                size="sm",
                variant="secondary",
            )

            with gr.Accordion("Options", open=False):
                patient_id_input = gr.Textbox(
                    label="Patient ID",
                    value="",
                    placeholder="(auto-generated if blank)",
                    info="Optional identifier for this diagnosis session.",
                )
                top_k_input = gr.Slider(
                    label="Top-K results",
                    minimum=1,
                    maximum=20,
                    value=10,
                    step=1,
                    info="Number of candidate diagnoses to return.",
                )

            diagnose_btn = gr.Button(
                "🔍 Run Diagnosis",
                variant="primary",
                size="lg",
            )

        # === RIGHT COLUMN: Results ===
        with gr.Column(scale=2):
            gr.Markdown("### Diagnosis Results")

            results_md = gr.Markdown(
                value="*Enter phenotypes and click 'Run Diagnosis' to see results.*",
                label="Ranked Candidates",
            )

            with gr.Row():
                export_csv_btn = gr.DownloadButton(
                    "⬇️ Export CSV (all candidates)",
                    size="sm",
                    variant="secondary",
                    interactive=False,
                )
                export_report_btn = gr.DownloadButton(
                    "⬇️ Export report (.md)",
                    size="sm",
                    variant="secondary",
                    interactive=False,
                )
            gr.Markdown(
                "<sub>CSV = one row per candidate (all Top-K) for spreadsheets · "
                "report = full evidence + explanation for every candidate · "
                "both say what produced the result (the served_ columns; the "
                "report's identity section).</sub>"
            )

            candidate_selector = gr.Dropdown(
                label="Select candidate for evidence detail",
                choices=[],
                value=None,
                interactive=True,
            )

            with gr.Accordion("Evidence Panel", open=True):
                evidence_md = gr.Markdown(
                    value="",
                    label="Evidence",
                )

            with gr.Accordion("Full Explanation", open=False):
                explanation_md = gr.Markdown(
                    value="",
                    label="Explanation",
                )

    # === Event wiring ===
    _diagnosis_outputs = [
        results_md, evidence_md, explanation_md, candidate_selector, results_state,
        export_csv_btn, export_report_btn,
    ]
    diagnose_btn.click(
        fn=_on_diagnose,
        inputs=[phenotype_input, patient_id_input, top_k_input],
        outputs=_diagnosis_outputs,
    )

    candidate_selector.change(
        fn=_on_candidate_select,
        inputs=[candidate_selector, results_state],
        outputs=[evidence_md, explanation_md],
    )

    # Editing the phenotype list invalidates the shown/exported results — clear
    # them (and disable the downloads) so nothing stale can be exported.
    phenotype_input.change(
        fn=_on_phenotype_change,
        inputs=[results_state],
        outputs=_diagnosis_outputs,
    )

    # Note: export files are pre-generated in _on_diagnose and set as each
    # DownloadButton's value, so a single click downloads them — no click handler.

    # === Model config event wiring ===
    # Read on each page load, not once at build time: the app is built before
    # the server is serving, so a status computed then is stale on arrival.
    blocks.load(fn=_on_load_status, inputs=[], outputs=[config_status_md])

    reload_btn.click(
        fn=_on_reload_pipeline,
        inputs=[data_dir_input, checkpoint_input, arch_input],
        outputs=[config_status_md, config_msg],
    )

    save_cfg_btn.click(
        fn=_on_save_config,
        inputs=[data_dir_input, checkpoint_input],
        outputs=[config_msg],
    )

    reset_btn.click(
        fn=_on_reset_defaults,
        inputs=[],
        outputs=[data_dir_input, checkpoint_input, config_msg],
    )
