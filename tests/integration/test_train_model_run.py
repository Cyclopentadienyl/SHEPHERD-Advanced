"""
`scripts/train_model.py` run end to end, on a workspace it can actually train.

The synthetic fixtures elsewhere cannot complete a training: their generated
profiles name phenotypes `d` and `d + 100`, outside the three-phenotype graph
they are paired with, and data loading stops at an IndexError. This one builds
the same complete tripartite graph with every sample's ids inside it, and runs
the script as an operator does -- a subprocess, one short epoch on CPU -- to
check what a run leaves behind: its exit status, `config.yaml`, the
`runtime.json` recording the CUDA allocator it started under, and the
checkpoint's record of its inputs, which since contract M2.1's S8 is built only
from the reads training trained on.

**The expected allocator comes from the same settings file the script reads**,
resolved by the same rule, so the test holds whatever preset the developer
running it has saved.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tests.fixtures.trainable_workspace import write_trainable_workspace  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "train_model.py"


def test_a_short_run_completes_and_records_its_allocator(tmp_path):
    from src.config.runtime_presets import (
        ALLOC_SOURCE_ENV,
        load_runtime_settings,
        resolve_allocator,
    )

    data_dir = write_trainable_workspace(tmp_path)
    output_dir = tmp_path / "out"
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", ALLOC_SOURCE_ENV)}
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--data-dir", str(data_dir), "--device", "cpu",
         "--epochs", "1", "--seed", "42", "--conv-type", "gat", "--hidden-dim", "16",
         "--num-layers", "1", "--batch-size", "4", "--output-dir", str(output_dir)],
        capture_output=True, encoding="utf-8", env={**env, "PYTHONIOENCODING": "utf-8"},
        cwd=REPO_ROOT, timeout=600,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert (output_dir / "config.yaml").is_file()
    assert (output_dir / "final_metrics.json").is_file()

    _preset, expected = resolve_allocator(load_runtime_settings().get("allocator_preset"))
    runtime = json.loads((output_dir / "runtime.json").read_text(encoding="utf-8"))
    assert runtime == {
        "PYTORCH_ALLOC_CONF": expected,
        "PYTORCH_CUDA_ALLOC_CONF": None,
        ALLOC_SOURCE_ENV: "preset",
        "allocator_source": "saved preset",
    }
    assert f"CUDA allocator: {expected} (saved preset)" in result.stdout + result.stderr

    # The record names the bytes the run read: every role, the file's digest.
    import hashlib

    checkpoint = torch.load(data_dir / "checkpoints" / "gat" / "last.pt",
                            map_location="cpu", weights_only=False)
    files = {"train_samples": "train_samples.json", "val_samples": "val_samples.json",
             "node_features": "node_features.pt", "edge_indices": "edge_indices.pt",
             "num_nodes": "num_nodes.json", "split_manifest": "split_manifest.json",
             "kg": "kg.json"}
    assert checkpoint["training_input_digests"] == {
        role: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
        for role, name in files.items()
    }
