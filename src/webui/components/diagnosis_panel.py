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
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import requests

logger = logging.getLogger(__name__)

API_BASE = "http://127.0.0.1:8000"

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
