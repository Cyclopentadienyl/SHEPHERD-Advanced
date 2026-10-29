"""
Every exported name resolves, and the names K7 retired are gone.
================================================================
K7 (group G1) deleted dead code and whole files: the decoders package,
FlexHeteroAttention, PhenotypeDiseaseMatcher, create_model, EdgeTypeEncoder,
the positional-encoding preprocessing with the KnowledgeGraph methods only it
called, the unwired metric classes, the core config dataclasses, the
ProgressBar and GradientClipping callbacks and a few dead helpers.

A package keeps its exports in two places that fail only when used: an
``__all__`` entry for a name that is gone raises at ``from package import *``,
and ``src.models`` resolves its exports lazily, so a stale entry in its map
raises only when that name is read. This file reads every one of them, checks
the lazy map and ``__all__`` name the same set, and checks that each retired
name is gone rather than kept as an alias.

It also pins what the deletions keep (the K7 working sheet's must-not-delete
list): HeteroGNNLayer, AdaptiveAttentionBackend, DiagnosisSample, the
subgraph loader stack and RankingMetrics, which wait for K8.

Module: tests/unit/test_package_exports.py
"""
import ast
import importlib
import importlib.util
import inspect
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The packages and modules whose exports K7 group G1 changed.
TOUCHED = (
    "src.core",
    "src.kg",
    "src.inference",
    "src.models",
    "src.models.encoders",
    "src.models.attention",
    "src.models.gnn",
    "src.training",
    "src.training.callbacks",
    "src.utils.metrics",
    "src.utils.checkpoint_paths",
    "src.api.services.training_manager",
)

#: Names retired in K7 group G1, by the module that exported them.
RETIRED = {
    "src.models": (
        "PhenotypeDiseaseMatcher", "create_model", "EdgeTypeEncoder", "FlexHeteroAttention",
        "create_ontology_score_mod", "create_species_score_mod", "DiagnosisHead",
        "LinkPredictionHead", "NodeClassificationHead", "ExplanationHead",
    ),
    "src.models.gnn": ("PhenotypeDiseaseMatcher", "create_model"),
    "src.models.gnn.shepherd_gnn": ("PhenotypeDiseaseMatcher", "create_model"),
    "src.models.encoders": ("EdgeTypeEncoder",),
    "src.models.encoders.type_encoder": ("EdgeTypeEncoder",),
    "src.models.attention": (
        "FlexHeteroAttention", "create_ontology_score_mod", "create_species_score_mod",
    ),
    "src.kg": (
        "preprocess_for_gnn", "compute_laplacian_pe", "compute_rwse", "compute_degree_features",
    ),
    "src.core": ("ModelConfig", "TrainingConfig", "DataConfig"),
    "src.core.types": ("ModelConfig", "TrainingConfig", "DataConfig"),
    "src.training": ("ProgressBar", "GradientClipping"),
    "src.training.callbacks": ("ProgressBar", "GradientClipping"),
    "src.utils.metrics": (
        "OntologyViolationMetrics", "EvidenceCoverageMetrics", "TrainingMetrics",
        "LinkPredictionMetrics", "DiagnosisMetrics", "compute_hits_at_k", "compute_mrr",
        "compute_ndcg",
    ),
    "src.utils.checkpoint_paths": ("allocate_run_dir", "ranking_score_from_logs"),
    "src.api.services.training_manager": ("TrainingStatus",),
}

#: Whole modules deleted in K7 group G1.
GONE_MODULES = (
    "src.models.decoders.heads",
    "src.models.attention.flex_attention",
    "src.kg.preprocessing",
)

#: Retired members of classes that stay: (module, class, member).
GONE_MEMBERS = (
    ("src.kg.graph", "KnowledgeGraph", "metadata"),
    ("src.kg.graph", "KnowledgeGraph", "to_pyg_hetero_data"),
    ("src.models.gnn.shepherd_gnn", "ShepherdGNN", "get_layer_outputs"),
    ("src.training.trainer", "Trainer", "get_model_info"),
    ("src.core.protocols", "KnowledgeGraphProtocol", "to_pyg_hetero_data"),
    ("src.core.protocols", "ConfigLoaderProtocol", "get_model_config"),
    ("src.core.protocols", "ConfigLoaderProtocol", "get_training_config"),
    ("src.core.protocols", "ConfigLoaderProtocol", "get_data_config"),
)

#: What the deletions keep until K8 (the working sheet's must-not-delete list).
KEPT = {
    "src.models": ("HeteroGNNLayer", "AdaptiveAttentionBackend"),
    "src.models.attention": ("AdaptiveAttentionBackend",),
    "src.models.gnn": ("HeteroGNNLayer",),
    "src.kg.data_loader": ("DiagnosisSample", "SubgraphSampler"),
    "src.utils.metrics": ("RankingMetrics",),
}

_OPTIONAL_STACK = ("torch", "torch_geometric", "torch_scatter", "torch_sparse", "pyg_lib")


def _import(name: str):
    """Import `name`; skip where the optional GNN stack it needs is absent."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] in _OPTIONAL_STACK:
            pytest.skip(f"{name} needs {exc.name}")
        raise


@pytest.mark.parametrize("name", TOUCHED)
def test_every_name_in_all_resolves(name):
    module = _import(name)
    exported = getattr(module, "__all__", None)
    assert exported, f"{name} has no __all__"
    assert len(exported) == len(set(exported)), f"{name}.__all__ repeats a name"
    missing = [item for item in exported if not hasattr(module, item)]
    assert missing == [], f"{name}.__all__ names what it does not have: {missing}"


def _lazy_map_names() -> set:
    """The keys of `src.models.__getattr__`'s map, read from the source."""
    tree = ast.parse((REPO_ROOT / "src" / "models" / "__init__.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
                and any(getattr(t, "id", None) == "_public_api" for t in node.targets)):
            return {key.value for key in node.value.keys}
    raise AssertionError("src.models has no _public_api map")


def test_the_models_lazy_map_and_all_name_the_same_names_and_each_resolves():
    module = _import("src.models")
    lazy = _lazy_map_names()
    assert lazy == set(module.__all__)
    for item in sorted(lazy):
        assert getattr(module, item) is not None, item


@pytest.mark.parametrize("name,retired", sorted(RETIRED.items()))
def test_the_retired_names_are_gone(name, retired):
    module = _import(name)
    exported = set(getattr(module, "__all__", ()))
    for item in retired:
        assert item not in exported, f"{name}.__all__ still names {item}"
        assert not hasattr(module, item), f"{name}.{item} is still there"


@pytest.mark.parametrize("name", GONE_MODULES)
def test_the_deleted_modules_cannot_be_imported(name):
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        spec = None
    assert spec is None, f"{name} is still importable from {spec.origin}"


@pytest.mark.parametrize("name,cls,member", GONE_MEMBERS)
def test_the_retired_members_are_gone(name, cls, member):
    assert not hasattr(getattr(_import(name), cls), member), f"{cls}.{member} is still there"


def test_shepherd_gnn_forward_no_longer_captures_layer_outputs():
    shepherd_gnn = _import("src.models.gnn.shepherd_gnn")
    parameters = inspect.signature(shepherd_gnn.ShepherdGNN.forward).parameters
    assert "return_all_layers" not in parameters


@pytest.mark.parametrize("name,kept", sorted(KEPT.items()))
def test_what_waits_for_k8_is_kept(name, kept):
    module = _import(name)
    for item in kept:
        assert hasattr(module, item), f"{name}.{item} was deleted; it waits for K8"
