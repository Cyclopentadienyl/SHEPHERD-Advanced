"""
A registered test cohort over a workspace with real HPO and MONDO ids.
=====================================================================
No other fixture combines the three things M4's flow needs: a workspace whose
`kg.json` holds real `HP:`/`MONDO:` ids (so a source can be mapped against it),
a checkpoint Mode C can score on that workspace, and a cohort version committed
into a registry outside the repository. This builds all three with the
production functions, never a copy of them:

- the workspace is `scripts/setup_demo.build_demo_kg()` written by
  `src/kg/workspace.write_workspace` (manifest, tensors, synthetic dataset
  record, generated train and val cohorts);
- the checkpoint is a `ShepherdGNN` built on that workspace's own tensors and
  written by `write_checkpoint`, carrying the `training_input_digests` training
  records, built by `scripts/train_model`'s `read_training_inputs` and
  `training_input_digests` (`train_samples`, `val_samples`, the graph roles,
  `split_manifest`, `kg`); untrained, so **no rank from it is evidence about
  the scorer**;
- the version is imported through `prepare_import` and `commit_import`.

Options cover the R10 cases a record can present: no validation recorded
(`training_record="no_validation"`), a graph-only record that is not one the
pipeline writes (`"graph_only"`), and a source built from the workspace's own
val cases (`source="val_cases"`), so its phenotype-set overlap with validation
is every case by construction.

The graph source has deliberate exclusions and one repeated term; its expected
counts are `GRAPH_SOURCE_COUNTS`. CPU, a few seconds, deterministic.

For tests that change files, `copy_registered_cohort` copies a built tree, so a
module builds once and each test works on its own copy; `write_demo_workspace`
with another `sample_seed` gives a re-sampled workspace over the same graph.

Module: tests/fixtures/registered_cohort.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional

from src.kg.graph import DEFAULT_FEATURE_SEED

#: Recorded as the version's `imported.at`, so two builds differ only where the
#: code revision differs.
IMPORTED_AT = datetime(2026, 10, 16, 12, 0, 0, tzinfo=timezone.utc)

LABEL = "demo-test-1"
ROLE = "demo"
SOURCE_FILENAME = "cases.json"

TRAINING_RECORDS = ("pipeline", "no_validation", "graph_only")
SOURCES = ("graph", "val_cases")

#: Well-formed ids that are not nodes of the demo graph.
UNKNOWN_HPO = "HP:9999999"
UNKNOWN_HPO_2 = "HP:9999998"
UNKNOWN_MONDO = "MONDO:9999999"

#: The checkpoint's architecture and seed, as `synthetic_workspace` builds one.
HIDDEN_DIM = 8
MODEL_SEED = 20260818

#: What the graph source maps to under each partial-phenotype policy.
GRAPH_SOURCE_COUNTS: Dict[str, Dict[str, Any]] = {
    "keep": {
        "source_cases": 13, "ranked_eligible": 10,
        "excluded": {"ground_truth_absent": 2, "phenotype_unmappable": 1}, "n_excluded": 3,
        "cases_with_lost_terms": 1, "terms_unmappable": 2,
        "repeats_removed": 1, "cases_with_repeats": 1,
    },
    "exclude": {
        "source_cases": 13, "ranked_eligible": 9,
        "excluded": {"ground_truth_absent": 2, "phenotype_unmappable": 2}, "n_excluded": 4,
        "cases_with_lost_terms": 0, "terms_unmappable": 2,
        "repeats_removed": 1, "cases_with_repeats": 1,
    },
}
GRAPH_SOURCE_UNMAPPABLE_TERMS = {UNKNOWN_HPO_2: 1, UNKNOWN_HPO: 2}


class RegisteredCohort(NamedTuple):
    """Where everything is. `version` is `commit_import`'s result."""

    data_dir: Path
    checkpoint: Path
    checkpoint_sha256: str
    registry: Path
    source: Path
    version: Any


def write_demo_workspace(data_dir: Path, *, feature_seed: int = DEFAULT_FEATURE_SEED,
                         sample_seed: Optional[int] = None) -> Path:
    """The demo graph as a complete workspace. Another `feature_seed` gives the
    same `kg.json` with different `node_features.pt`: another graph export, for
    a binding that must refuse. Another `sample_seed` gives the same graph
    export and dataset record with other `train` and `val` cohorts: a workspace
    re-sampled after a checkpoint was trained on the first, for R10."""
    from scripts.setup_demo import build_demo_kg
    from src.kg.workspace import SampleBudget, write_workspace

    budget = SampleBudget(num_train=20, num_val=5, val_disease_fraction=0.2)
    if sample_seed is not None:
        budget = budget._replace(seed=sample_seed)
    write_workspace(
        build_demo_kg(), data_dir, feature_dim=8, feature_seed=feature_seed, samples=budget,
    )
    return data_dir


def _terms(data_dir: Path) -> tuple[List[str], List[str]]:
    """The workspace graph's phenotype and disease ids, in node order."""
    from src.kg.graph import KnowledgeGraph

    mapping = KnowledgeGraph.read_json(data_dir / "kg.json").kg.get_node_id_mapping()
    local = [[node.split(":", 1)[1] for node in sorted(mapping[t], key=mapping[t].get)]
             for t in ("phenotype", "disease")]
    return local[0], local[1]


def graph_source_cases(data_dir: Path) -> List[Dict[str, Any]]:
    """Cases built from the graph's own ids: eight plain ones, one listing a term
    twice, one with a term the graph lacks (kept or excluded by policy), one
    whose disease is not a node, one with no term that maps, and one with
    both."""
    phenotypes, diseases = _terms(data_dir)
    cases = [
        {"case_id": f"case-{i + 1:02d}",
         "phenotypes": [phenotypes[i % len(phenotypes)], phenotypes[(i + 3) % len(phenotypes)]],
         "disease": diseases[i % len(diseases)]}
        for i in range(8)
    ]
    cases += [
        {"case_id": "case-09", "phenotypes": [phenotypes[0], phenotypes[1], phenotypes[0]],
         "disease": diseases[0]},
        {"case_id": "case-10", "phenotypes": [phenotypes[2], UNKNOWN_HPO],
         "disease": diseases[1]},
        {"case_id": "case-11", "phenotypes": [phenotypes[3], phenotypes[4]],
         "disease": UNKNOWN_MONDO},
        {"case_id": "case-12", "phenotypes": [UNKNOWN_HPO_2], "disease": diseases[2]},
        {"case_id": "case-13", "phenotypes": [UNKNOWN_HPO], "disease": UNKNOWN_MONDO},
    ]
    return cases


def val_source_cases(data_dir: Path) -> List[Dict[str, Any]]:
    """The workspace's val cases, translated back to their HPO and MONDO ids, in
    the order the sample file lists them."""
    from src.kg.graph import KnowledgeGraph
    from src.kg.storage.file_storage import read_samples

    reverse = KnowledgeGraph.read_json(data_dir / "kg.json").kg.get_reverse_node_mapping()
    return [
        {"case_id": sample.patient_id,
         "phenotypes": [reverse["phenotype"][i].split(":", 1)[1] for i in sample.phenotype_ids],
         "disease": reverse["disease"][sample.disease_id].split(":", 1)[1]}
        for sample in read_samples(data_dir, "val").samples
    ]


def write_source(path: Path, cases: List[Dict[str, Any]]) -> Path:
    path.write_text(json.dumps({"format": "shepherd-cases/1", "cases": cases}, indent=2),
                    encoding="utf-8")
    return path


def write_fixture_checkpoint(data_dir: Path, path: Path, *,
                             training_record: str = "pipeline") -> str:
    """A checkpoint Mode C can score on `data_dir`; returns the digest written.

    The model is built on the workspace's own tensors, as training builds it
    (metadata and input widths from the graph data), so `build_shepherd_model`
    rebuilds it for measurement. Its record is training's, from training's
    reads of this workspace, then narrowed for the R10 options.
    """
    import torch

    from scripts.train_model import read_training_inputs, training_input_digests
    from src.kg.artifacts import GRAPH_ARTIFACTS
    from src.models.gnn.shepherd_gnn import ShepherdGNN, ShepherdGNNConfig
    from src.utils.checkpoint_io import write_checkpoint

    if training_record not in TRAINING_RECORDS:
        raise ValueError(f"training_record must be one of {TRAINING_RECORDS}")
    inputs = read_training_inputs(data_dir)
    digests = training_input_digests(inputs)
    if training_record == "no_validation":
        del digests["val_samples"]
    elif training_record == "graph_only":
        digests = {role: digests[role] for role in GRAPH_ARTIFACTS}

    x_dict = inputs.graph.graph_data["x_dict"]
    edge_index_dict = inputs.graph.graph_data["edge_index_dict"]
    config = {"hidden_dim": HIDDEN_DIM, "num_layers": 2, "num_heads": 2}
    torch.manual_seed(MODEL_SEED)
    model = ShepherdGNN(
        metadata=(list(x_dict), list(edge_index_dict)),
        in_channels_dict={node_type: x.size(-1) for node_type, x in x_dict.items()},
        config=ShepherdGNNConfig(**config),
    )
    return write_checkpoint(
        {
            "state_dict": model.state_dict(),
            "config": config,
            "epoch": 1,
            "logs": {"val_mrr": 0.0},
            "training_input_digests": digests,
        },
        path,
    )


def import_version(registry: Path, data_dir: Path, source: Path, *, label: str = LABEL,
                   role: str = ROLE, partial_phenotypes: str = "keep") -> Any:
    """Preview, then commit under the preview's own id: the importer's two steps."""
    from src.evaluation.cohort_registry import commit_import, prepare_import

    plan = prepare_import(registry, data_dir, source, label, role, partial_phenotypes)
    return commit_import(plan, registry, plan.preview_id, now=IMPORTED_AT)


def build_registered_cohort(
    root: Path,
    *,
    training_record: str = "pipeline",
    source: str = "graph",
    label: str = LABEL,
    role: str = ROLE,
    partial_phenotypes: str = "keep",
) -> RegisteredCohort:
    """Workspace, checkpoint, source and registry under `root`, which must be
    outside the repository (a pytest tmp directory is)."""
    from src.evaluation.cohort_registry import REPO_ROOT

    root = Path(root).resolve()
    if root == REPO_ROOT or REPO_ROOT in root.parents:
        raise ValueError(f"{root} is inside the repository; build the fixture in a tmp dir")
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")

    data_dir = write_demo_workspace(root / "workspace")
    checkpoint = root / "checkpoints" / "best.pt"
    checkpoint_sha256 = write_fixture_checkpoint(data_dir, checkpoint,
                                                 training_record=training_record)
    cases = graph_source_cases(data_dir) if source == "graph" else val_source_cases(data_dir)
    source_path = write_source(root / SOURCE_FILENAME, cases)
    registry = root / "registry"
    registry.mkdir()
    version = import_version(registry, data_dir, source_path, label=label, role=role,
                             partial_phenotypes=partial_phenotypes)
    return RegisteredCohort(data_dir, checkpoint, checkpoint_sha256, registry, source_path,
                            version)


def copy_registered_cohort(cohort: RegisteredCohort, root: Path) -> RegisteredCohort:
    """A copy of a built cohort's whole tree at `root`, for a test that changes
    files: one build per module, one copy per test.

    **No version is made here.** The copy is the bytes `commit_import` wrote,
    resolved again through `resolve_version`, which checks the copy as it checks
    any version. A version names no path (only its source's basename), so the
    copy is the same version, with the same manifest digest, somewhere else.
    `root` must not exist and must be outside the repository.
    """
    import shutil

    from src.evaluation.cohort_registry import REPO_ROOT, resolve_version

    source = cohort.data_dir.parent
    root = Path(root).resolve()
    if root == REPO_ROOT or REPO_ROOT in root.parents:
        raise ValueError(f"{root} is inside the repository; copy the fixture to a tmp dir")
    shutil.copytree(source, root)

    def moved(path: Path) -> Path:
        return root / Path(path).relative_to(source)

    registry = moved(cohort.registry)
    return RegisteredCohort(
        moved(cohort.data_dir), moved(cohort.checkpoint), cohort.checkpoint_sha256, registry,
        moved(cohort.source), resolve_version(registry, cohort.version.manifest["label"]),
    )


__all__ = [
    "GRAPH_SOURCE_COUNTS",
    "GRAPH_SOURCE_UNMAPPABLE_TERMS",
    "IMPORTED_AT",
    "LABEL",
    "ROLE",
    "RegisteredCohort",
    "SOURCES",
    "TRAINING_RECORDS",
    "UNKNOWN_HPO",
    "UNKNOWN_HPO_2",
    "UNKNOWN_MONDO",
    "build_registered_cohort",
    "copy_registered_cohort",
    "graph_source_cases",
    "import_version",
    "val_source_cases",
    "write_demo_workspace",
    "write_fixture_checkpoint",
    "write_source",
]
