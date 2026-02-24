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

        with gr.Blocks() as demo:
            create_training_tab()

        # If we get here without error, the tab was created successfully
        assert demo is not None


# ==============================================================================
# Helper Function Tests
# ==============================================================================
class TestTrainingConsoleHelpers:
    """Test helper functions in the training console module."""

    def test_format_status_idle(self):
        from src.webui.components.training_console import _format_status

        result = _format_status({"status": "idle"})
        assert "IDLE" in result

    def test_format_status_running(self):
        from src.webui.components.training_console import _format_status

        result = _format_status({
            "status": "running",
            "pid": 12345,
            "current_epoch": 5,
            "total_epochs": 100,
            "elapsed_seconds": 3661,
        })
        assert "RUNNING" in result
        assert "12345" in result
        assert "5 / 100" in result
        assert "01:01:01" in result

    def test_format_resources_no_gpu(self):
        from src.webui.components.training_console import _format_resources

        result = _format_resources({
            "gpu": {"available": False, "devices": []},
            "ram": {"total_gb": 32.0, "used_gb": 16.0, "percent": 50.0},
        })
        assert "Not available" in result
        assert "32.0" in result

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
        assert "85" in result

    def test_build_loss_df_empty(self):
        from src.webui.components.training_console import _build_loss_df

        df = _build_loss_df([], [])
        assert len(df) == 0
        assert list(df.columns) == ["epoch", "loss", "split"]

    def test_build_loss_df_with_data(self):
        from src.webui.components.training_console import _build_loss_df

        train_data = [
            {"epoch": 0, "total": 2.5},
            {"epoch": 1, "total": 2.0},
        ]
        val_data = [
            {"epoch": 0, "val_loss": 3.0},
            {"epoch": 1, "val_loss": 2.5},
        ]
        df = _build_loss_df(train_data, val_data)
        assert len(df) == 4
        assert set(df["split"].unique()) == {"train", "val"}
        # Epochs should be 1-indexed for display (0-indexed log + 1)
        assert list(sorted(df["epoch"].unique())) == [1, 2]

    def test_build_hits_df(self):
        from src.webui.components.training_console import _build_hits_df

        val_data = [
            {"epoch": 0, "val_hits@1": 0.05, "val_hits@10": 0.2},
            {"epoch": 1, "val_hits@1": 0.1, "val_hits@10": 0.3},
        ]
        df = _build_hits_df(val_data)
        assert len(df) == 4
        assert set(df["metric"].unique()) == {"Hits@1", "Hits@10"}
        # Epochs should be 1-indexed for display
        assert list(sorted(df["epoch"].unique())) == [1, 2]

    def test_collect_config(self):
        from src.webui.components.training_console import _collect_config

        config = _collect_config(
            # Paths
            data_dir="data/processed", output_dir="outputs",
            checkpoint_dir="models/checkpoints",
            # Tier 1
            num_epochs=50, learning_rate=0.001, batch_size="32",
            conv_type="gat", device="auto", seed=42,
            # Tier 2
            hidden_dim="256", num_layers=4, dropout=0.1, weight_decay=0.01,
            scheduler_type="cosine", warmup_steps=500,
            early_stopping_patience=10,
            diagnosis_weight=1.0, link_prediction_weight=0.5,
            contrastive_weight=0.3, ortholog_weight=0.2,
            # Tier 3
            gradient_accumulation_steps=1, max_grad_norm=1.0,
            num_heads="8", use_ortholog_gate=True,
            use_amp=True, amp_dtype="float16",
            temperature=0.07, label_smoothing=0.1, margin=1.0,
            num_neighbors_str="15, 10, 5", max_subgraph_nodes=5000,
        )
        assert config["num_epochs"] == 50
        assert config["learning_rate"] == 0.001
        assert config["batch_size"] == 32
        assert config["hidden_dim"] == 256
        assert config["num_neighbors"] == [15, 10, 5]
        assert config["data_dir"] == "data/processed"
        assert config["output_dir"] == "outputs"
        assert config["checkpoint_dir"] == "models/checkpoints"

    def test_collect_config_strips_prefix(self):
        """Verify that SHEPHERD-Advanced/ display prefix is stripped from paths."""
        from src.webui.components.training_console import _collect_config

        config = _collect_config(
            # Paths with display prefix
            data_dir="SHEPHERD-Advanced/data/processed",
            output_dir="SHEPHERD-Advanced/outputs",
            checkpoint_dir="SHEPHERD-Advanced/models/checkpoints",
            # Tier 1
            num_epochs=10, learning_rate=0.001, batch_size="32",
            conv_type="gat", device="auto", seed=42,
            # Tier 2
            hidden_dim="256", num_layers=4, dropout=0.1, weight_decay=0.01,
            scheduler_type="cosine", warmup_steps=500,
            early_stopping_patience=10,
            diagnosis_weight=1.0, link_prediction_weight=0.5,
            contrastive_weight=0.3, ortholog_weight=0.2,
            # Tier 3
            gradient_accumulation_steps=1, max_grad_norm=1.0,
            num_heads="8", use_ortholog_gate=True,
            use_amp=True, amp_dtype="float16",
            temperature=0.07, label_smoothing=0.1, margin=1.0,
            num_neighbors_str="15, 10, 5", max_subgraph_nodes=5000,
        )
        # Prefix should be stripped — backend uses project-root-relative paths
        assert config["data_dir"] == "data/processed"
        assert config["output_dir"] == "outputs"
        assert config["checkpoint_dir"] == "models/checkpoints"


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
