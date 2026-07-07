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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import requests

logger = logging.getLogger(__name__)

API_BASE = "http://127.0.0.1:8000"
PIPELINE_API = f"{API_BASE}/api/v1"

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
    ``HP:0000000`` form, preserves order, and de-duplicates.
    """
    seen: set[str] = set()
    ids: List[str] = []
    for match in _HPO_ID_RE.finditer(text or ""):
        hpo_id = f"HP:{match.group(1)}"
        if hpo_id not in seen:
            seen.add(hpo_id)
            ids.append(hpo_id)
    return ids

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
        resp = requests.post(
            f"{API_BASE}/api/v1/diagnose",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        return {"error": "API server not reachable. Is uvicorn running?"}
    except requests.HTTPError as e:
        return {"error": f"API error: {e.response.status_code} — {e.response.text}"}
    except Exception as e:
        return {"error": f"Unexpected error: {e}"}


# =============================================================================
# Formatting helpers
def _get_pipeline_status() -> Dict[str, Any]:
    """Get pipeline status from API."""
    try:
        resp = requests.get(f"{PIPELINE_API}/pipeline/status", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {"initialized": False, "error": "API not reachable"}


def _reload_pipeline(data_dir: str, checkpoint_path: str) -> Dict[str, Any]:
    """Reload pipeline via API."""
    payload = {"data_dir": data_dir}
    if checkpoint_path and checkpoint_path.strip():
        payload["checkpoint_path"] = checkpoint_path.strip()
    try:
        resp = requests.post(
            f"{PIPELINE_API}/pipeline/reload",
            json=payload,
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json()
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
    try:
        resp = requests.post(
            f"{PIPELINE_API}/pipeline/config",
            json={"data_dir": data_dir, "checkpoint_path": checkpoint_path or None},
            timeout=5,
        )
        resp.raise_for_status()
        return "Configuration saved."
    except Exception as e:
        return f"Failed to save: {e}"


def _format_pipeline_status(status_data: Dict[str, Any]) -> str:
    """Format pipeline status as Markdown."""
    if not status_data.get("initialized", False):
        return "⚪ **Pipeline not loaded.** Configure paths below and click Load / Reload Pipeline."

    gnn = "✅" if status_data.get("gnn_ready") else "❌"
    sp = "✅" if status_data.get("sp_ready") else "❌"
    mode = status_data.get("scoring_mode", "unknown")
    kg_n = status_data.get("kg_nodes", 0)
    kg_e = status_data.get("kg_edges", 0)

    lines = [
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
        # Training metrics if available
        for key in ("val_loss", "train_loss", "mrr", "hits_at_1", "hits_at_10"):
            val = ckpt_meta.get(key)
            if val is not None:
                label = key.replace("_", " ").title()
                meta_parts.append(f"{label}: {val:.4f}")
        lines.append(f"- Checkpoint: {' | '.join(meta_parts)}")

    # Fingerprint warnings
    fp_warns = status_data.get("fingerprint_warnings", [])
    if fp_warns:
        lines.append("")
        lines.append(f"⚠️ **KG version mismatch** ({len(fp_warns)} warning{'s' if len(fp_warns) != 1 else ''}):")
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
# Export helpers — write ALL candidates (not just the previewed one) to a file
# =============================================================================
_CSV_COLUMNS = [
    "rank", "disease_id", "disease_name", "confidence_score", "confidence_label",
    "gnn_score", "sp_score", "reasoning_score", "evidence_mode", "evidence_summary",
    "min_path_length", "num_direct_paths", "num_analogies",
    "matching_phenotypes", "supporting_genes",
]


def _fmt_num(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else ""


def _build_results_csv(result: Dict[str, Any]) -> str:
    """One row per candidate — the tabular summary, for spreadsheet analysis."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for c in result.get("candidates", []):
        pkg = c.get("evidence_package") or {}
        writer.writerow({
            "rank": c.get("rank"),
            "disease_id": c.get("disease_id", ""),
            "disease_name": c.get("disease_name", ""),
            "confidence_score": _fmt_num(c.get("confidence_score")),
            "confidence_label": c.get("confidence_label") or "",
            "gnn_score": _fmt_num(c.get("gnn_score")),
            "sp_score": _fmt_num(c.get("sp_score")),
            "reasoning_score": _fmt_num(c.get("reasoning_score")),
            "evidence_mode": pkg.get("mode") or "",
            "evidence_summary": (pkg.get("summary") or "").replace("\n", " ").strip(),
            "min_path_length": pkg.get("min_path_length", ""),
            "num_direct_paths": len(pkg.get("direct_paths") or []),
            "num_analogies": len(pkg.get("analogies") or []),
            "matching_phenotypes": "; ".join(c.get("matching_phenotypes") or []),
            "supporting_genes": "; ".join(c.get("supporting_genes") or []),
        })
    return buf.getvalue()


def _build_results_report_md(result: Dict[str, Any]) -> str:
    """Full human-readable report: session meta + summary table + per-candidate
    evidence and explanation for EVERY candidate (reuses the preview formatters)."""
    candidates = result.get("candidates", [])
    lines = ["# SHEPHERD-Advanced Diagnosis Report", ""]
    for key, value in (
        ("Patient ID", result.get("patient_id")),
        ("Session ID", result.get("session_id")),
        ("Timestamp", result.get("timestamp")),
        ("Model version", result.get("model_version")),
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

    lines += ["", "## Ranked candidates", "", _format_results_table(candidates),
              "", "## Per-candidate detail"]
    for c in candidates:
        lines += [
            "",
            f"### #{c.get('rank')} — {c.get('disease_name')}",
            f"`{c.get('disease_id', '')}` · confidence "
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


def _slug(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("_") or "session"


def _write_named_temp(content: str, filename: str) -> str:
    """Write content to a uniquely-named temp dir so the download keeps a
    human-friendly filename (tempfile alone would give a random basename)."""
    directory = Path(tempfile.mkdtemp(prefix="shepherd_export_"))
    path = directory / filename
    path.write_text(content, encoding="utf-8")
    return str(path)


def _export_csv(result_state: Optional[Dict[str, Any]]) -> str:
    if not result_state or not result_state.get("candidates"):
        raise gr.Error("Run a diagnosis first — there are no results to export.")
    tag = _slug(result_state.get("patient_id") or result_state.get("session_id"))
    return _write_named_temp(_build_results_csv(result_state), f"diagnosis_{tag}.csv")


def _export_report(result_state: Optional[Dict[str, Any]]) -> str:
    if not result_state or not result_state.get("candidates"):
        raise gr.Error("Run a diagnosis first — there are no results to export.")
    tag = _slug(result_state.get("patient_id") or result_state.get("session_id"))
    return _write_named_temp(
        _build_results_report_md(result_state), f"diagnosis_report_{tag}.md"
    )


# =============================================================================
# Event handlers
# =============================================================================
def _on_diagnose(
    phenotype_text: str,
    patient_id: str,
    top_k: int,
) -> Tuple[str, str, str, "gr.update", Optional[Dict[str, Any]]]:
    """
    Handle the Diagnose button click.

    Returns: (results_table, evidence_detail, explanation,
              candidate_dropdown_update, results_state)
    The full API result is stashed in results_state so the candidate dropdown and
    the export buttons can reuse it without re-running inference.
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
        )

    candidates = result.get("candidates", [])
    warnings = result.get("warnings", [])
    inference_ms = result.get("inference_time_ms", 0)

    # Build results table
    table_md = _format_results_table(candidates)
    meta = f"\n\n*{len(candidates)} candidates found in {inference_ms:.1f}ms*"
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

    return (
        table_md + meta,
        evidence_md,
        explain_md,
        gr.update(choices=choices, value=choices[0] if choices else None),
        result,
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


def _on_reload_pipeline(data_dir: str, checkpoint_path: str) -> Tuple[str, str]:
    """Handle Reload Pipeline button click."""
    if not data_dir or not data_dir.strip():
        return "⚠️ Data directory is required.", ""

    result = _reload_pipeline(data_dir.strip(), checkpoint_path.strip() if checkpoint_path else "")

    status_md = ""
    if result.get("success"):
        status_md = _format_pipeline_status(result.get("status", {}))
    else:
        status_md = f"❌ {result.get('message', 'Unknown error')}"

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
    """Refresh pipeline status display."""
    status_data = _get_pipeline_status()
    return _format_pipeline_status(status_data)


# =============================================================================
# Tab builder (called from app.py)
# =============================================================================
def create_diagnosis_tab() -> None:
    """
    Build the Diagnosis Panel tab inside a gr.Blocks context.

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
        config_status_md = gr.Markdown(
            value=_format_pipeline_status(_get_pipeline_status()),
        )

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
                info="Path to .pt checkpoint file. Leave blank to auto-detect.",
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
                )
                export_report_btn = gr.DownloadButton(
                    "⬇️ Export report (.md)",
                    size="sm",
                    variant="secondary",
                )
            gr.Markdown(
                "<sub>CSV = one row per candidate (all Top-K) for spreadsheets · "
                "report = full evidence + explanation for every candidate.</sub>"
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
    diagnose_btn.click(
        fn=_on_diagnose,
        inputs=[phenotype_input, patient_id_input, top_k_input],
        outputs=[
            results_md, evidence_md, explanation_md, candidate_selector, results_state,
        ],
    )

    candidate_selector.change(
        fn=_on_candidate_select,
        inputs=[candidate_selector, results_state],
        outputs=[evidence_md, explanation_md],
    )

    export_csv_btn.click(
        fn=_export_csv, inputs=[results_state], outputs=[export_csv_btn]
    )
    export_report_btn.click(
        fn=_export_report, inputs=[results_state], outputs=[export_report_btn]
    )

    # === Model config event wiring ===
    reload_btn.click(
        fn=_on_reload_pipeline,
        inputs=[data_dir_input, checkpoint_input],
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
