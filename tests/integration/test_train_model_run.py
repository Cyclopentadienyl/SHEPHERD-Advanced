"""
`scripts/train_model.py` run end to end, on a workspace it can actually train.

The synthetic fixtures elsewhere cannot complete a training: their generated
profiles name phenotypes `d` and `d + 100`, outside the three-phenotype graph
they are paired with, and data loading stops at an IndexError. This one builds
the same complete tripartite graph with every sample's ids inside it, and runs
the script as an operator does -- a subprocess, one short epoch on CPU -- to
check what a run leaves behind: its exit status, `config.yaml`, and the
`runtime.json` recording the CUDA allocator it started under.

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

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "train_model.py"


def _trainable_workspace(root: Path) -> Path:
    from tests.fixtures.generated_workspace import write_generated_workspace
    from tests.fixtures.synthetic_workspace import N_GENES, _graph

    n_phenotypes, n_diseases = 6, 6
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    x_dict, edge_index_dict, num_nodes = _graph(n_phenotypes, n_diseases)
    torch.save(x_dict, data_dir / "node_features.pt")
    torch.save(edge_index_dict, data_dir / "edge_indices.pt")
    (data_dir / "num_nodes.json").write_text(json.dumps(num_nodes))
    (data_dir / "kg.json").write_text(json.dumps({"synthetic": True}))

    profiles = {
        d: {"phenotype_ids": [d % n_phenotypes, (d + 1) % n_phenotypes],
            "gene_ids": [d % N_GENES]}
        for d in range(n_diseases)
    }

    def samples(split, ids, per_disease):
        return [
            {"patient_id": f"{split}-{d}-{i}",
             "phenotype_ids": list(profiles[d]["phenotype_ids"]),
             "disease_id": d}
            for d in ids for i in range(per_disease)
        ]

    train_ids, val_ids = [0, 1, 2, 3], [4, 5]
    write_generated_workspace(
        data_dir, train_ids=train_ids, val_ids=val_ids, profiles=profiles,
        train_samples=samples("train", train_ids, 3),
        val_samples=samples("val", val_ids, 2),
    )
    return data_dir


def test_a_short_run_completes_and_records_its_allocator(tmp_path):
    from src.config.runtime_presets import (
        ALLOC_SOURCE_ENV,
        load_runtime_settings,
        resolve_allocator,
    )

    data_dir = _trainable_workspace(tmp_path)
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
