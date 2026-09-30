"""Load one checkpoint against one workspace the way the API does, and report.

Usage: loadcheck.py <workspace> <checkpoint> nosp|sp
Exit 0 only if the GNN loaded (and, for `sp`, the SP table is ready and bound).
Warnings and recorded digests are observations, not criteria (BACKLOG §3.5).
"""
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, ".")
from src.inference.pipeline import create_diagnosis_pipeline  # noqa: E402
from src.kg.graph import KnowledgeGraph  # noqa: E402
from src.utils.fingerprint import file_sha256  # noqa: E402

workspace, checkpoint, stage = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
assert stage in ("nosp", "sp"), "third argument must be nosp or sp"
assert torch.cuda.is_available(), "CUDA is mandatory"
logging.basicConfig(level=logging.WARNING)

raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
pipeline = create_diagnosis_pipeline(
    kg=KnowledgeGraph.load_json(str(workspace / "kg.json")),
    checkpoint_path=str(checkpoint), data_dir=str(workspace),
    kg_path=str(workspace / "kg.json"), device="cuda",
)
config = pipeline.get_pipeline_config()
report = {
    "checkpoint_sha256": file_sha256(checkpoint),
    "checkpoint_bytes": checkpoint.stat().st_size,
    "epoch": raw.get("epoch"),
    "gnn_ready": config["gnn_ready"],
    "has_model": config["has_model"],
    "scoring_mode": config["scoring_mode"],
    "sp_ready": config["sp_ready"],
    "sp_kg_binding": config["sp_kg_binding"],
    "sp_max_hops": config["sp_max_hops"],
    "checkpoint_meta": config["checkpoint_meta"],
    "kg_nodes": config["kg_nodes"],
    "kg_edges": config["kg_edges"],
    "fingerprint_warning_count": len(config["fingerprint_warnings"]),
    "records_training_input_digests": bool(raw.get("training_input_digests")),
}
print(json.dumps(report, default=str, indent=1))
loaded = report["gnn_ready"] and report["has_model"]
bound = report["sp_ready"] and report["sp_kg_binding"] == "verified"
sys.exit(0 if loaded and (stage == "nosp" or bound) else 1)
