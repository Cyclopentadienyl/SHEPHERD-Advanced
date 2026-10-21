"""
# ==============================================================================
# Module: tests/unit/test_webui.py
# ==============================================================================
# Purpose: Smoke tests for WebUI components
#
# Tests:
#   - Gradio app creation
#   - Training console tab creation
#   - Import rule compliance (no src.training imports in src.webui)
# ==============================================================================
"""
import ast
import os
from pathlib import Path

import pytest

# Skip all tests if gradio is not available
gr = pytest.importorskip("gradio")
pd = pytest.importorskip("pandas")


# ==============================================================================
# App Creation Tests
# ==============================================================================
class TestGradioApp:
    """Test that the Gradio app can be created."""

    def test_create_gradio_app_returns_blocks(self):
        from src.webui.app import create_gradio_app

        app = create_gradio_app()
        assert isinstance(app, gr.Blocks)

    def test_create_training_tab(self):
        """Training tab can be created inside a Blocks context."""
        from src.webui.components.training_console import create_training_tab
        from src.webui.shell import standalone_context

        with gr.Blocks() as demo:
            create_training_tab(standalone_context(demo))

        # If we get here without error, the tab was created successfully
        assert demo is not None


# ==============================================================================
# Helper Function Tests
# ==============================================================================
class TestTrainingConsoleHelpers:
    """Test helper functions in the training console module."""

    def test_the_status_pill_idle(self):
        from src.webui.components.training_console import pill_html, pill_state

        assert pill_state({"status": "idle"}) == (
            "idle", "Idle · no training job since the server started")
        assert "shep-pill-idle" in pill_html({"status": "idle"})

    def test_the_status_pill_running(self):
        from src.webui.components.training_console import pill_state

        kind, words = pill_state({
            "status": "running",
            "pid": 12345,
            "current_epoch": 5,
            "total_epochs": 100,
            "batch": 340,
            "total_batches": 812,
            "elapsed_seconds": 3661,
        })
        assert kind == "running"
        assert words == "RUNNING · epoch 5/100 · batch 340/812 · 01:01:01"

    def test_format_resources_no_gpu(self):
        from src.webui.components.training_console import _format_resources

        result = _format_resources({
            "gpu": {"available": False, "devices": []},
            "ram": {"total_gb": 32.0, "used_gb": 16.0, "percent": 50.0},
        })
        assert "GPU: not available" in result
        assert "memory 16.0 / 32.0 GB" in result

    def test_format_resources_with_gpu(self):
        from src.webui.components.training_console import _format_resources

        result = _format_resources({
            "gpu": {
                "available": True,
                "devices": [{
                    "index": 0,
                    "name": "RTX 5070 Ti",
                    "utilization_percent": 85.0,
                    "memory_used_mb": 8000.0,
                    "memory_total_mb": 17408.0,
                    "temperature_c": 72.0,
                }],
            },
            "ram": {"total_gb": 64.0, "used_gb": 32.0, "percent": 50.0},
        })
        assert "RTX 5070 Ti" in result
        assert "GPU 0 85 %" in result and "72 °C" in result and "VRAM 7.8 / 17.0 GB" in result

    def test_the_resources_line_has_no_text_under_14px(self):
        """The resource display used 13 px (and 11 px) inline text; it is one
        line now, sized by the page's style sheet (nothing under 14 px)."""
        from src.webui.components.training_console import _format_resources

        result = _format_resources({
            "gpu": {"available": True, "devices": [{"index": 0, "name": "GB10",
                                                     "utilization_percent": None,
                                                     "temperature_c": None}]},
            "ram": {"total_gb": 119.7, "available_gb": 113.6, "used_gb": 2.0},
            "unified_memory": True,
        })
        assert "font-size" not in result and "style=" not in result
        assert "unified memory 6.1 / 119.7 GB (shared CPU + GPU pool)" in result
        assert "VRAM" not in result                     # the shared pool is shown once
        assert "GPU 0 N/A" in result

    def test_the_loss_chart_from_an_empty_log(self):
        from src.webui.components.training_console import training_charts

        loss = training_charts({"train": [], "validation": []})[0]
        assert loss["id"] == "loss" and loss["x"] == [] and loss["series"] == []

    def test_the_loss_chart_with_data(self):
        from src.webui.components.training_console import training_charts

        train_data = [
            {"epoch": 0, "total": 2.5},
            {"epoch": 1, "total": 2.0},
        ]
        val_data = [
            {"epoch": 0, "val_loss": 3.0},
            {"epoch": 1, "val_loss": 2.5},
        ]
        loss = training_charts({"train": train_data, "validation": val_data})[0]
        assert [s["label"] for s in loss["series"]] == ["train", "validation"]
        # Epochs are 1-indexed for display (0-indexed log + 1)
        assert loss["x"] == [1, 2]
        assert loss["series"][0]["values"] == [2.5, 2.0]

    def test_the_hits_chart(self):
        from src.webui.components.training_console import training_charts

        val_data = [
            {"epoch": 0, "val_hits@1": 0.05, "val_hits@10": 0.2},
            {"epoch": 1, "val_hits@1": 0.1, "val_hits@10": 0.3},
        ]
        hits = training_charts({"train": [], "validation": val_data})[2]
        assert [s["label"] for s in hits["series"]] == ["Hits@1", "Hits@10"]
        # Epochs should be 1-indexed for display
        assert hits["x"] == [1, 2]

    def test_collect_config(self, monkeypatch):
        from src.webui.components import training_console as tc
        from src.webui.components.training_console import _collect_config

        # Isolate from any local .shepherd_runtime_settings.json so config["compile"]
        # is deterministic.
        monkeypatch.setattr(tc, "load_runtime_settings", lambda *a, **k: {})

        config = _collect_config(
            # Paths
            data_dir="data/processed", run_name="  Pilot Run  ",
            # Tier 1
            num_epochs=50, learning_rate=0.001, batch_size="32",
            conv_type="gat", device="auto", seed=42,
            # Tier 2
            hidden_dim="256", num_layers=4, dropout=0.1, weight_decay=0.01,
            scheduler_type="cosine", warmup_steps=500, min_lr_ratio=0.1,
            early_stopping_patience=10,
            diagnosis_weight=1.0, link_prediction_weight=0.5,
            contrastive_weight=0.3, ortholog_weight=0.2,
            # Tier 3
            gradient_accumulation_steps=1, max_grad_norm=1.0,
            num_heads="8", use_ortholog_gate=True,
            amp_mode="float16",
            temperature=0.07, label_smoothing=0.1, margin=1.0,
            num_neighbors_str="15, 10, 5", max_subgraph_nodes=5000,
        )
        assert config["num_epochs"] == 50
        assert config["min_lr_ratio"] == 0.1
        assert config["compile"] is False
        assert config["learning_rate"] == 0.001
        assert config["batch_size"] == 32
        assert config["hidden_dim"] == 256
        assert config["num_neighbors"] == [15, 10, 5]
        assert config["data_dir"] == "data/processed"
        assert config["run_name"] == "Pilot Run"
        assert not {"output_dir", "checkpoint_dir", "log_dir"} & set(config)

    def test_collect_config_strips_prefix(self, monkeypatch):
        """Verify that SHEPHERD-Advanced/ display prefix is stripped from paths."""
        from src.webui.components import training_console as tc
        from src.webui.components.training_console import _collect_config

        monkeypatch.setattr(tc, "load_runtime_settings", lambda *a, **k: {})

        config = _collect_config(
            # Paths with display prefix
            data_dir="SHEPHERD-Advanced/data/processed",
            run_name="",
            # Tier 1
            num_epochs=10, learning_rate=0.001, batch_size="32",
            conv_type="gat", device="auto", seed=42,
            # Tier 2
            hidden_dim="256", num_layers=4, dropout=0.1, weight_decay=0.01,
            scheduler_type="cosine", warmup_steps=500, min_lr_ratio=0.01,
            early_stopping_patience=10,
            diagnosis_weight=1.0, link_prediction_weight=0.5,
            contrastive_weight=0.3, ortholog_weight=0.2,
            # Tier 3
            gradient_accumulation_steps=1, max_grad_norm=1.0,
            num_heads="8", use_ortholog_gate=True,
            amp_mode="float16",
            temperature=0.07, label_smoothing=0.1, margin=1.0,
            num_neighbors_str="15, 10, 5", max_subgraph_nodes=5000,
        )
        # Prefix should be stripped — backend uses project-root-relative paths
        assert config["data_dir"] == "data/processed"
        assert config["run_name"] == ""

    def test_collect_config_compile_from_runtime_settings(self, monkeypatch):
        """torch.compile is now sourced from Runtime Settings, not a Training widget."""
        from src.webui.components import training_console as tc

        common = dict(
            data_dir="data/processed", run_name="",
            num_epochs=10, learning_rate=0.001, batch_size="32",
            conv_type="gat", device="auto", seed=42,
            hidden_dim="256", num_layers=4, dropout=0.1, weight_decay=0.01,
            scheduler_type="cosine", warmup_steps=500, min_lr_ratio=0.1,
            early_stopping_patience=10,
            diagnosis_weight=1.0, link_prediction_weight=0.5,
            contrastive_weight=0.3, ortholog_weight=0.2,
            gradient_accumulation_steps=1, max_grad_norm=1.0,
            num_heads="8", use_ortholog_gate=True, amp_mode="float16",
            temperature=0.07, label_smoothing=0.1, margin=1.0,
            num_neighbors_str="15, 10, 5", max_subgraph_nodes=5000,
        )

        monkeypatch.setattr(tc, "load_runtime_settings", lambda *a, **k: {"torch_compile": True})
        assert tc._collect_config(**common)["compile"] is True

        monkeypatch.setattr(tc, "load_runtime_settings", lambda *a, **k: {})
        assert tc._collect_config(**common)["compile"] is False


# ==============================================================================
# Import Rule Compliance Test
# ==============================================================================
class TestImportRuleCompliance:
    """Verify that src.webui does NOT import from src.training."""

    def test_no_training_imports_in_webui(self):
        """
        Scan all Python files in src/webui/ and verify none of them
        import from src.training (import-linter rule).
        """
        webui_dir = Path(__file__).parent.parent.parent / "src" / "webui"
        violations = []

        for py_file in webui_dir.rglob("*.py"):
            with open(py_file, encoding="utf-8") as f:
                try:
                    tree = ast.parse(f.read())
                except SyntaxError:
                    continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("src.training"):
                            violations.append(
                                f"{py_file.relative_to(webui_dir)}: "
                                f"import {alias.name}"
                            )
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.startswith("src.training"):
                        violations.append(
                            f"{py_file.relative_to(webui_dir)}: "
                            f"from {node.module} import ..."
                        )

        assert violations == [], (
            "src.webui must not import from src.training. Violations:\n"
            + "\n".join(violations)
        )


# ==============================================================================
# Test: Error / Warning Banner
# ==============================================================================
class TestErrorBanner:
    """The dedicated banner surfaces hard errors (red) and PyG fallback
    warnings (yellow), separately from the normal status text."""

    def test_no_error_no_warning_is_hidden(self):
        from src.webui.components.training_console import _build_error_banner_payload

        html, key = _build_error_banner_payload({"status": "running"}, None)
        assert html is None and key is None

    def test_error_only_renders_red_block(self):
        from src.webui.components.training_console import _build_error_banner_payload

        html, key = _build_error_banner_payload(
            {"error_message": "RuntimeError: CUDA out of memory"}, None
        )
        assert html is not None
        assert "shep-banner-red" in html and 'role="alert"' in html  # red, announced
        assert key.startswith("ERR:")
        assert "PYG:" not in key

    def test_pyg_warning_only_renders_yellow_block(self):
        from src.webui.components.training_console import _build_error_banner_payload

        html, key = _build_error_banner_payload({}, "pyg_lib missing")
        assert html is not None
        assert "shep-banner-amber" in html  # yellow
        assert key.startswith("PYG:")

    def test_error_rendered_before_warning(self):
        from src.webui.components.training_console import _build_error_banner_payload

        html, key = _build_error_banner_payload({"error_message": "boom"}, "slow")
        assert "ERR:" in key and "PYG:" in key
        assert html.index("shep-banner-red") < html.index("shep-banner-amber")

    def test_a_failed_run_shows_its_epoch_its_exit_and_its_output_tail(self):
        from src.webui.components.training_console import (
            BANNER_TAIL_LINES,
            _build_error_banner_payload,
        )

        tail = "\n".join(f"line {i}" for i in range(30))
        html, _key = _build_error_banner_payload(
            {"error_message": f"Process exited with code 1\n\n--- Last output ---\n{tail}",
             "current_epoch": 7, "total_epochs": 60}, None, run="gat/20261019-081500_run_7b10e4")
        assert "Training failed at epoch 7 of 60: Process exited with code 1" in html
        assert "run <code>gat/20261019-081500_run_7b10e4</code>" in html
        assert "line 29" in html and f"line {29 - BANNER_TAIL_LINES}" not in html
        assert 'data-act="dismiss"' in html

    def test_html_is_escaped(self):
        from src.webui.components.training_console import _build_error_banner_payload

        html, _ = _build_error_banner_payload(
            {"error_message": "<script>alert(1)</script>"}, None
        )
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_key_is_stable_for_identical_content(self):
        from src.webui.components.training_console import _build_error_banner_payload

        _, k1 = _build_error_banner_payload({"error_message": "same"}, None)
        _, k2 = _build_error_banner_payload({"error_message": "same"}, None)
        _, k3 = _build_error_banner_payload({"error_message": "other"}, None)
        assert k1 == k2  # dismissal stickiness relies on this
        assert k1 != k3  # a different error re-shows

    def test_error_not_in_status_text(self):
        """error_message must live in the banner, not the status pill; a failed
        run's pill is red."""
        from src.webui.components.training_console import pill_html

        text = pill_html({"status": "failed", "error_message": "RuntimeError: boom"})
        assert "boom" not in text and "shep-pill-red" in text and "FAILED" in text
