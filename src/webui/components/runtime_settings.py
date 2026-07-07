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
    at the repo root (path + presets defined in ``src.runtime_presets``) and are
    consumed at launch time by ``scripts/launch/shep_launch.py`` (allocator) and
    by the training subprocess it spawns (which inherits the environment).
"""
from __future__ import annotations

import os

import gradio as gr

from src.api.services.backend_control import is_training_active, restart_backend
from src.runtime_presets import (
    ALLOCATOR_PRESETS,
    DEFAULT_ALLOCATOR,
    load_runtime_settings,
    save_runtime_settings,
)


def _active_env_override() -> str | None:
    """Return the explicit allocator env override if one is set, else None."""
    return os.environ.get("PYTORCH_ALLOC_CONF") or os.environ.get("PYTORCH_CUDA_ALLOC_CONF")


def _badge(label: str, kind: str) -> str:
    """Render a single coloured pill badge (styled by global CSS in app.py)."""
    return f"<span class='shep-badge shep-badge-{kind}'>{label}</span>"


def _badge_row(*badges: str) -> str:
    """Wrap one or more badges in a flex row."""
    return "<div class='shep-badge-row'>" + "".join(badges) + "</div>"


# Red warning shown beside the Restart button while training is in progress.
_RESTART_LOCKED_HTML = (
    "<span class='shep-restart-locked'>⛔ Training in progress — "
    "restart is locked.</span>"
)

# Browser-side poller chained after the restart click. Re-exec'ing the backend
# gives the new process a fresh Gradio session, so the already-loaded page keeps
# polling against a session the new server doesn't know — System Resources and
# dropdowns hang until a manual refresh. This watches /health and reloads the
# page (an automatic F5) only once the NEW process is up, detected by EITHER the
# uptime resetting (uptime_seconds drops below the value seen at click time) OR a
# down→up transition. If the restart was refused (e.g. training active), neither
# condition fires, the backend never goes down, and the poller just times out
# without reloading.
_RESTART_RELOAD_JS = """
async () => {
    const get = async () => {
        try {
            const r = await fetch('/health', {cache: 'no-store'});
            return r.ok ? await r.json() : null;
        } catch (e) { return null; }
    };
    const first = await get();
    const oldUptime = (first && typeof first.uptime_seconds === 'number')
        ? first.uptime_seconds : null;
    const deadline = Date.now() + 90000;
    let sawDown = false;
    const poll = async () => {
        const h = await get();
        if (h === null) {
            sawDown = true;
        } else if (h.status === 'healthy') {
            const restarted =
                (oldUptime !== null && typeof h.uptime_seconds === 'number'
                    && h.uptime_seconds < oldUptime)
                || sawDown;
            if (restarted) { window.location.reload(); return; }
        }
        if (Date.now() < deadline) { setTimeout(poll, 1000); }
    };
    setTimeout(poll, 2000);
}
"""


def create_runtime_settings_tab() -> None:
    """Build the Runtime Settings tab (call inside a gr.Tab/gr.Blocks context)."""
    settings = load_runtime_settings()
    current_alloc = settings.get("allocator_preset", DEFAULT_ALLOCATOR)
    if current_alloc not in ALLOCATOR_PRESETS:
        current_alloc = DEFAULT_ALLOCATOR
    current_compile = bool(settings.get("torch_compile", False))

    gr.Markdown(
        "### Runtime Settings\n"
        "These settings change **how computation runs on the hardware** — speed, "
        "memory, and numerical precision — and are separate from the Training "
        "Console (which configures the model). Changes take effect only after you "
        "click **Apply Settings**; items tagged **🔄 Restart required** also need a "
        "backend restart.\n\n"
        "_An explicit `PYTORCH_ALLOC_CONF` / `PYTORCH_CUDA_ALLOC_CONF` set in the "
        "launch environment overrides the Memory Allocator chosen here._"
    )

    # If the backend was launched with an explicit allocator env override, the
    # persisted UI choice is being ignored — surface that prominently.
    env_override = _active_env_override()
    if env_override:
        gr.Markdown(
            f"> ⚠️ **An explicit allocator override is active in the environment** "
            f"(`{env_override}`). It takes precedence over the Memory Allocator "
            f"setting below until the backend is restarted without it.",
            elem_id="runtime_env_override",
        )

    # Action bar — the two backend-level actions, kept together above the
    # individual settings: Apply persists the settings; Restart relaunches the
    # backend so a startup-read setting (the Memory Allocator) takes effect. This
    # mirrors the natural workflow (change allocator → Apply → Restart).
    _locked_now = is_training_active()
    with gr.Row():
        apply_btn = gr.Button("Apply Settings", variant="primary", elem_id="runtime_apply")
        restart_btn = gr.Button(
            "Restart Backend",
            variant="secondary",
            interactive=not _locked_now,
            elem_id="runtime_restart",
        )
    status = gr.Markdown(
        f"Applied: allocator **{current_alloc}**, "
        f"torch.compile **{'on' if current_compile else 'off'}**.",
        elem_id="runtime_status",
    )
    # Restart feedback + training lock (kept next to the button it belongs to).
    restart_lock_note = gr.HTML(
        _RESTART_LOCKED_HTML if _locked_now else "",
        visible=_locked_now,
        elem_id="runtime_restart_lock",
    )
    restart_status = gr.Markdown("", elem_id="runtime_restart_status")
    gr.Markdown(
        "<sub>**Restart Backend** relaunches the backend (REST API + this UI) so a "
        "newly applied **Memory Allocator** takes effect. Locked while training is "
        "running; the page reconnects automatically a few seconds after a restart."
        "</sub>"
    )
    # Keep the lock state live without manual refresh (cheap status read).
    restart_lock_timer = gr.Timer(4.0)

    gr.Markdown("#### Memory")
    with gr.Group():
        allocator = gr.Dropdown(
            label="Memory Allocator",
            choices=list(ALLOCATOR_PRESETS.keys()),
            value=current_alloc,
            elem_id="allocator_preset",
        )
        gr.HTML(_badge_row(_badge("Memory", "mem"), _badge("Restart required", "restart")))
        gr.Markdown(
            "GPU memory allocation strategy — governs fragmentation and peak usage. "
            "The backend also runs CUDA in-process (for diagnosis/inference), so a "
            "change applies only after the backend is restarted."
        )
        with gr.Accordion("Details — what each option does", open=False):
            gr.Markdown(
                "- **cuda_async** *(default)* — NVIDIA driver's stream-ordered memory "
                "pool. Zero-tuning and driver-managed; behaviour may change or improve "
                "with CUDA driver/runtime upgrades. Note: PyTorch's native memory "
                "snapshot stats are reduced under this backend.\n"
                "- **expandable** — PyTorch growable VMM segments. Bounds fragmentation "
                "for variable-size GNN subgraphs while keeping native memory "
                "observability.\n"
                "- **native_roundup** — native allocator with power-of-2 size rounding "
                "(plus non-split rounding) to cut fragmentation at low overhead.\n"
                "- **native** — native allocator with no tuning; can fragment badly on "
                "variable tensor sizes.\n\n"
                "_Measured on this project (HGT, batch 256): cuda_async ≈ expandable "
                "(~26 GB, comparable speed); plain native fragments (≈60→120 GB)._"
            )

    gr.Markdown("#### Compute")
    with gr.Group():
        torch_compile = gr.Checkbox(
            label="Enable torch.compile",
            value=current_compile,
            elem_id="torch_compile",
        )
        gr.HTML(
            _badge_row(
                _badge("Speed", "speed"),
                _badge("Precision ⚠", "prec"),
                _badge("Experimental", "exp"),
                _badge("Next run", "nextrun"),
            )
        )
        gr.Markdown(
            "Attempts to compile supported model regions to reduce overhead; dynamic "
            "PyG/HGT workloads may graph-break, fall back to eager, or run slower. "
            "Applies to the **next training run** — no backend restart needed."
        )
        with gr.Accordion("Details", open=False):
            gr.Markdown(
                "⚠️ **Experimental / benchmark-only — keep this off unless you are "
                "explicitly benchmarking it.** Heterogeneous GNNs (HGT/GAT) often "
                "graph-break and gain little, and on **GB10 (DGX Spark), HGT has been "
                "observed to run _slower_ with torch.compile enabled**. There is also a "
                "known sm_121 Triton issue on GB10 (it falls back to eager). Always "
                "compare MRR / Hits@K against an eager run before trusting any result."
            )

    def _refresh_restart_lock():
        locked = is_training_active()
        return (
            gr.update(interactive=not locked),
            gr.update(visible=locked, value=_RESTART_LOCKED_HTML if locked else ""),
        )

    def _on_restart():
        result = restart_backend()
        if not result.get("success"):
            return gr.update(
                value=f"⛔ **{result.get('error', 'Cannot restart right now.')}**"
            )
        return gr.update(
            value="🔄 **Restarting backend…** This page will reload "
            "automatically once the backend is back (a few seconds)."
        )

    restart_lock_timer.tick(
        fn=_refresh_restart_lock, outputs=[restart_btn, restart_lock_note]
    )
    # After the click handler runs, kick off a browser-side poller that reloads
    # the page once the fresh backend is up (a new process means a new Gradio
    # session; without a reload the old page's pollers hang). The poller no-ops
    # if the restart was refused.
    restart_btn.click(fn=_on_restart, outputs=[restart_status]).then(
        fn=None, inputs=None, outputs=None, js=_RESTART_RELOAD_JS
    )

    def _on_change(*_values):
        return "● **Unsaved changes** — click **Apply Settings** to persist."

    def _on_apply(alloc, compile_on):
        data = load_runtime_settings()
        prev_alloc = data.get("allocator_preset", DEFAULT_ALLOCATOR)
        data["allocator_preset"] = alloc
        data["torch_compile"] = bool(compile_on)
        save_runtime_settings(data)

        parts = [
            f"✓ Applied. Memory Allocator = **{alloc}**, "
            f"torch.compile = **{'on' if compile_on else 'off'}**."
        ]
        if alloc != prev_alloc:
            parts.append(
                "🔄 **Restart the backend** for the new allocator to take effect "
                "(read once at startup)."
            )
        parts.append("torch.compile applies to the **next training run** (no restart).")
        if _active_env_override():
            parts.append(
                "⚠️ An explicit `PYTORCH_ALLOC_CONF` env override is active and takes "
                "precedence over the allocator until removed."
            )
        return "  ".join(parts)

    allocator.change(fn=_on_change, inputs=[allocator], outputs=[status])
    torch_compile.change(fn=_on_change, inputs=[torch_compile], outputs=[status])
    apply_btn.click(fn=_on_apply, inputs=[allocator, torch_compile], outputs=[status])


__all__ = ["create_runtime_settings_tab"]
