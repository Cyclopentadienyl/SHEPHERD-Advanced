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

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import requests

logger = logging.getLogger(__name__)

API_BASE = "http://127.0.0.1:8000"
PIPELINE_API = f"{API_BASE}/api/v1"

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
            timeout=60,
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
# Event handlers
# =============================================================================
def _on_diagnose(
    phenotype_text: str,
    patient_id: str,
    top_k: int,
) -> Tuple[str, str, str, gr.update]:
    """
    Handle the Diagnose button click.

    Returns: (results_table, evidence_detail, explanation, candidate_dropdown_update)
    """
    # Parse phenotype input
    lines = [
        line.strip() for line in phenotype_text.strip().split("\n")
        if line.strip()
    ]
    hpo_ids = []
    for line in lines:
        # Accept "HP:0001250 — Seizure" or just "HP:0001250"
        token = line.split("—")[0].split("-")[0].strip()
        if token.startswith("HP:"):
            hpo_ids.append(token.strip())

    if not hpo_ids:
        return (
            "⚠️ **No valid HPO terms found.** Enter one HPO ID per line (e.g., `HP:0001250`).",
            "",
            "",
            gr.update(choices=[], value=None),
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

    return (
        table_md + meta,
        evidence_md,
        explain_md,
        gr.update(choices=choices, value=choices[0] if choices else None),
    )


def _on_candidate_select(
    selection: str,
    phenotype_text: str,
    patient_id: str,
    top_k: int,
) -> Tuple[str, str]:
    """
    Handle candidate dropdown selection change.
    Re-calls API (stateless) and shows evidence for the selected candidate.
    """
    if not selection:
        return "", ""

    # Extract rank from "#{rank} — ..."
    try:
        rank = int(selection.split("#")[1].split("—")[0].strip())
    except (IndexError, ValueError):
        return "⚠️ Could not parse selection.", ""

    # Parse phenotypes again
    lines = [l.strip() for l in phenotype_text.strip().split("\n") if l.strip()]
    hpo_ids = [
        l.split("—")[0].split("-")[0].strip()
        for l in lines if l.strip().startswith("HP:")
    ]

    if not hpo_ids:
        return "⚠️ Phenotype list is empty.", ""

    result = _call_diagnose(
        phenotypes=hpo_ids,
        patient_id=patient_id or "webui_patient",
        top_k=int(top_k),
        include_explanations=True,
    )

    if "error" in result:
        return f"❌ {result['error']}", ""

    candidates = result.get("candidates", [])
    target = next((c for c in candidates if c["rank"] == rank), None)
    if not target:
        return f"⚠️ Rank #{rank} not found in results.", ""

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
                label="HPO Terms (one per line)",
                placeholder="HP:0001250 — Seizure\nHP:0001263 — Global developmental delay",
                lines=8,
                info="Enter HPO IDs, optionally followed by '—' and the name. One per line.",
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
        outputs=[results_md, evidence_md, explanation_md, candidate_selector],
    )

    candidate_selector.change(
        fn=_on_candidate_select,
        inputs=[candidate_selector, phenotype_input, patient_id_input, top_k_input],
        outputs=[evidence_md, explanation_md],
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
