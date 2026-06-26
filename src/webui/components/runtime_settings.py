"""
Runtime Settings Tab
====================
Execution/hardware-level settings that affect HOW computation runs on the
hardware — speed, memory, and numerical precision — as opposed to the Training
Console, which configures the model itself.

Some settings here are read once at process startup (e.g. the CUDA memory
allocator) and therefore require a backend restart to take effect; those are
tagged with 🔄 in the UI.

Module: src/webui/components/runtime_settings.py

Apply model:
    Changes are staged in the UI and only persisted when the user clicks
    "Apply Settings". Persisted values live in ``.shepherd_runtime_settings.json``
    at the repo root and are consumed at launch time by
    ``scripts/launch/shep_launch.py`` (allocator) and by the training subprocess
    it spawns (which inherits the environment).
"""
from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

# Persisted at repo root; read by shep_launch.py at startup.
RUNTIME_SETTINGS_FILE = Path(".shepherd_runtime_settings.json")

# Preset name -> PYTORCH_ALLOC_CONF value. "" means "framework default" (no tuning).
# NOTE: this mapping is intentionally duplicated (small + stable) in
# scripts/launch/shep_launch.py so the launcher stays import-light (no gradio).
# Keep the two in sync if presets change.
ALLOCATOR_PRESETS: dict[str, str] = {
    "cuda_async": "backend:cudaMallocAsync",
    "expandable": "expandable_segments:True",
    "native_roundup": "roundup_power2_divisions:4,max_non_split_rounding_mb:512",
    "native": "",
}
DEFAULT_ALLOCATOR = "cuda_async"


def load_runtime_settings() -> dict:
    """Load persisted runtime settings (empty dict if absent/unreadable)."""
    if RUNTIME_SETTINGS_FILE.exists():
        try:
            with open(RUNTIME_SETTINGS_FILE, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_runtime_settings(data: dict) -> None:
    with open(RUNTIME_SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def create_runtime_settings_tab() -> None:
    """Build the Runtime Settings tab (call inside a gr.Tab/gr.Blocks context)."""
    settings = load_runtime_settings()
    current_alloc = settings.get("allocator_preset", DEFAULT_ALLOCATOR)
    if current_alloc not in ALLOCATOR_PRESETS:
        current_alloc = DEFAULT_ALLOCATOR

    gr.Markdown(
        "### Runtime Settings\n"
        "These settings change **how computation runs on the hardware** — speed, "
        "memory, and numerical precision — and are separate from the Training "
        "Console (which configures the model). Changes take effect only after you "
        "click **Apply Settings**; items tagged **🔄 Restart required** also need a "
        "backend restart."
    )

    # Apply bar — placed above all options, as a single explicit action.
    with gr.Row():
        apply_btn = gr.Button("Apply Settings", variant="primary", elem_id="runtime_apply")
    status = gr.Markdown(
        f"Applied allocator: **{current_alloc}**.", elem_id="runtime_status"
    )

    gr.Markdown("#### Memory")
    with gr.Group():
        allocator = gr.Dropdown(
            label="Memory Allocator",
            choices=list(ALLOCATOR_PRESETS.keys()),
            value=current_alloc,
            elem_id="allocator_preset",
        )
        gr.Markdown(
            "🟦 **Memory**  ·  🔄 **Restart required**\n\n"
            "GPU memory allocation strategy — governs fragmentation and peak usage. "
            "The backend also runs CUDA in-process (for diagnosis/inference), so a "
            "change applies only after the backend is restarted."
        )
        with gr.Accordion("Details — what each option does", open=False):
            gr.Markdown(
                "- **cuda_async** *(default)* — NVIDIA driver's stream-ordered memory "
                "pool. Zero-tuning, driver-managed, and improves automatically with "
                "driver upgrades. Note: PyTorch's native memory snapshot stats are "
                "reduced under this backend.\n"
                "- **expandable** — PyTorch growable VMM segments. Bounds fragmentation "
                "for variable-size GNN subgraphs while keeping native memory "
                "observability.\n"
                "- **native_roundup** — native allocator with power-of-2 size rounding "
                "(plus non-split rounding) to cut fragmentation at low overhead.\n"
                "- **native** — framework default (no tuning); can fragment badly on "
                "variable tensor sizes.\n\n"
                "_Measured on this project (HGT, batch 256): cuda_async ≈ expandable "
                "(~26 GB, comparable speed); plain native fragments (≈60→120 GB)._"
            )

    def _on_change(_value):
        return "● **Unsaved changes** — click **Apply Settings** to persist."

    def _on_apply(alloc):
        data = load_runtime_settings()
        previous = data.get("allocator_preset", DEFAULT_ALLOCATOR)
        data["allocator_preset"] = alloc
        _save_runtime_settings(data)
        msg = f"✓ Applied. Memory Allocator = **{alloc}**."
        if alloc != previous:
            msg += (
                "  🔄 **Restart the backend** for the new allocator to take effect "
                "(it is read once at startup)."
            )
        return msg

    allocator.change(fn=_on_change, inputs=[allocator], outputs=[status])
    apply_btn.click(fn=_on_apply, inputs=[allocator], outputs=[status])


__all__ = ["create_runtime_settings_tab", "load_runtime_settings", "ALLOCATOR_PRESETS"]
