#!/usr/bin/env python3
"""
Derive the gene task's functional fixture from a workspace (README §8).
======================================================================
Reads the workspace as training does (`scripts/train_model.read_training_inputs`:
the split manifest, both cohorts and the graph tensors once each, verified
against the manifest), applies the declared rule of `src/kg/task_fixture.py`
(one causal gene from each case's disease plus distractors) and publishes
`derived/gene_fixture/` bound to that manifest by digest. The six bound
workspace files are not written.

The rows are a **functional fixture**, not the paper's simulated patients.

Usage:
    python scripts/derive_task_fixture.py --data-dir data/workspaces/demo [--seed 0]
        [--distractors 19] [--replace]

Script: scripts/derive_task_fixture.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

logger = logging.getLogger(__name__)


def derive(data_dir: Path, *, seed: int = 0, distractors: Optional[int] = None,
           replace: bool = False) -> Dict[str, Any]:
    """Read and verify the workspace, derive both splits, publish; the record."""
    from src.kg.task_fixture import (
        DEFAULT_DISTRACTORS,
        FIXTURE_SPLITS,
        derive_gene_rows,
        disease_genes,
        publish_gene_fixture,
    )
    from train_model import read_training_inputs

    distractors = DEFAULT_DISTRACTORS if distractors is None else int(distractors)
    inputs = read_training_inputs(Path(data_dir))
    graph = inputs.graph.graph_data
    genes_of = disease_genes(graph["edge_index_dict"])
    num_genes = int(graph["num_nodes_dict"].get("gene", 0))
    rows, counts = {}, {}
    for split in FIXTURE_SPLITS:
        rows[split], counts[split] = derive_gene_rows(
            inputs.samples[split].samples, genes_of, num_genes, seed=seed, split=split,
            distractors=distractors)
    if not rows["train"]:
        raise ValueError(f"no training case has a disease with a gene in the graph: {counts}")
    record = {
        "seed": int(seed), "distractors": distractors, "counts": counts,
        "workspace": {"split_manifest": inputs.manifest.identity.sha256,
                      "kg": inputs.bound["kg"],
                      "samples": {split: inputs.samples[split].identity.sha256
                                  for split in FIXTURE_SPLITS}},
    }
    return publish_gene_fixture(Path(data_dir), rows, record, replace=replace)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--data-dir", required=True, help="The workspace directory")
    parser.add_argument("--seed", type=int, default=0, help="The rule's seed (default 0)")
    parser.add_argument("--distractors", type=int, default=None,
                        help="Distractor candidates per case (default 19, provisional)")
    parser.add_argument("--replace", action="store_true",
                        help="Replace an existing fixture (runs that recorded it will refuse)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    record = derive(Path(args.data_dir), seed=args.seed, distractors=args.distractors,
                    replace=args.replace)
    print(json.dumps({"label": record["label"], "rule": record["rule"],
                      "counts": record["counts"], "files": record["files"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
