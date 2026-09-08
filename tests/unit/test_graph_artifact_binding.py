"""The graph a run consumes must be the export its manifest names.

`build_knowledge_graph` writes `kg.json`, exports three tensors from the same
in-memory graph, and generates cohorts from the same allocation. Until now the
manifest bound only `kg.json` and the sample files -- so the artifacts a model
actually opens, `node_features.pt` and `edge_indices.pt`, were bound to nothing.

Training and measurement hash them at consumption time, which records what was
used but does not show that those bytes are this graph's export. `graph_
fingerprint` does not close it either: it is structural -- node types, counts,
feature dimensions -- so a same-shaped tensor file from another workspace shares
it and passes.

**Two contracts, and the split matters.** A supplied institutional cohort carries
no allocation and is never subject to the generated splits' disjointness, but it
is scored against those same tensors. Had graph binding lived inside the
generated-cohort verifier, generated validation would have been protected while
institutional evaluation went on consuming a mixed workspace.

Module: tests/unit/test_graph_artifact_binding.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.kg.graph import KnowledgeGraph
from src.evaluation.cohort import (
    GRAPH_ARTIFACTS,
    verify_generated_cohorts,
    verify_graph_artifacts,
)
from tests.fixtures.generated_workspace import (
    one_sample_per_disease,
    profiles_for,
    write_generated_workspace,
)


def _workspace(root: Path, *, supplied=False):
    write_generated_workspace(root, train_ids=[0, 1, 2], val_ids=[3])
    if supplied:
        (root / "institutional_acceptance_samples.json").write_text(
            json.dumps(one_sample_per_disease("inst", [0], profiles_for([0])))
        )
    return root


def test_the_recorded_digests_are_the_exact_exported_bytes(tmp_path):
    from src.utils.fingerprint import file_sha256

    root = _workspace(tmp_path / "ws")
    manifest = json.loads((root / "split_manifest.json").read_text())

    for role, filename in GRAPH_ARTIFACTS.items():
        assert manifest["artifacts"][role] == file_sha256(root / filename)
    assert verify_graph_artifacts(root) == {
        role: manifest["artifacts"][role] for role in GRAPH_ARTIFACTS
    }


@pytest.mark.parametrize("role", sorted(GRAPH_ARTIFACTS))
def test_replacing_any_single_graph_artifact_is_refused(tmp_path, role):
    """A same-shaped artifact from another workspace passes structural
    fingerprinting. Only the recorded bytes catch it."""
    root = _workspace(tmp_path / "ws")
    (root / GRAPH_ARTIFACTS[role]).write_bytes(b"same shape, another workspace")

    with pytest.raises(ValueError, match=f"is not the {role} artifact"):
        verify_graph_artifacts(root)


class TestTrainingRefusesBeforeItActs:
    """The refusal has to precede the side effects, not merely exist.

    ``training_input_roles`` used to carry the verification, and a test calling it
    directly proved only that the helper *can* reject. By the time `train` reached
    it, the run directories existed, `config.yaml` was written, the graph tensors
    were loaded, dataloaders and a model were built, a Trainer was constructed and
    a resume checkpoint may have been loaded. Whether the workspace is sound is
    knowable from the files alone, so it is knowable before any of that.
    """

    @staticmethod
    def _forbid(monkeypatch, names):
        """Replace each stage with a sentinel that fails if it is reached."""
        import scripts.train_model as train_model

        for name in names:
            def _sentinel(*args, _name=name, **kwargs):
                raise AssertionError(f"{_name} ran after a digest mismatch")

            monkeypatch.setattr(train_model, name, _sentinel)
        return train_model

    def _config(self, train_model, root, tmp_path, **overrides):
        return train_model.TrainConfig(
            data_dir=str(root),
            output_dir=str(tmp_path / "outputs"),
            log_dir=str(tmp_path / "logs"),
            **overrides,
        )

    @pytest.mark.parametrize(
        "break_it,expected",
        [
            (lambda root: (root / "node_features.pt").write_bytes(b"another ws"),
             "is not the node_features artifact"),
            (lambda root: (root / "train_samples.json").write_text("[]"),
             "is not the file"),
            (lambda root: (root / "split_manifest.json").unlink(),
             "Rebuild it with"),
        ],
        ids=["mixed-graph", "replaced-cohort", "no-manifest"],
    )
    def test_no_stage_and_no_run_artifact_is_reached(
        self, monkeypatch, tmp_path, break_it, expected
    ):
        train_model = self._forbid(
            monkeypatch,
            ["load_graph_data", "create_dataloaders", "create_model_from_config",
             "resolve_resume_checkpoint"],
        )
        root = _workspace(tmp_path / "ws")
        break_it(root)

        with pytest.raises(ValueError, match=expected):
            train_model.train(self._config(train_model, root, tmp_path))

        assert not (tmp_path / "outputs").exists(), "run directory was created"
        assert not (tmp_path / "logs").exists(), "log directory was created"
        assert not (root / "checkpoints").exists(), "checkpoint directory was created"

    def test_a_sound_workspace_reaches_the_next_stage(self, monkeypatch, tmp_path):
        """The refusal must be about the workspace, not about the ordering itself:
        an unbroken workspace passes the preflight and goes on to load the graph."""
        train_model = self._forbid(monkeypatch, ["load_graph_data"])
        root = _workspace(tmp_path / "ws")

        with pytest.raises(AssertionError, match="load_graph_data ran"):
            train_model.train(self._config(train_model, root, tmp_path))


class TestTheClinicalPathIsAGraphConsumer:
    """The costliest consumer, and the one that was outside the contract.

    `DiagnosisPipeline._initialize_gnn` called `_load_graph_data` directly. A
    same-shaped `node_features.pt` from another workspace was refused by training
    and by measurement and would still have been loaded here, paired with a
    checkpoint, precomputed into embeddings and served.
    """

    @staticmethod
    def _pipeline_module(monkeypatch, reached):
        import src.inference.pipeline as pipeline

        def _forbidden(self, *args, **kwargs):
            reached.append("_load_graph_data")
            return None

        monkeypatch.setattr(pipeline.DiagnosisPipeline, "_load_graph_data", _forbidden)
        return pipeline

    def test_a_mixed_workspace_is_refused_before_the_graph_is_loaded(
        self, monkeypatch, tmp_path
    ):
        reached: list = []
        pipeline = self._pipeline_module(monkeypatch, reached)
        root = _workspace(tmp_path / "ws")
        (root / "node_features.pt").write_bytes(b"same shape, another workspace")

        with pytest.raises(ValueError, match="is not the node_features artifact"):
            pipeline.DiagnosisPipeline(
                kg=KnowledgeGraph(), data_dir=str(root),
                checkpoint_path=str(root / "ckpt.pt"),
            )

        assert reached == [], "graph loading was reached after a digest mismatch"

    def test_a_sound_workspace_goes_on_to_load_the_graph(self, monkeypatch, tmp_path):
        """The refusal is about the workspace, not about the ordering itself."""
        reached: list = []
        pipeline = self._pipeline_module(monkeypatch, reached)
        root = _workspace(tmp_path / "ws")
        (root / "ckpt.pt").write_bytes(b"weights")

        pipeline.DiagnosisPipeline(
            kg=KnowledgeGraph(), data_dir=str(root),
            checkpoint_path=str(root / "ckpt.pt"),
        )

        assert reached == ["_load_graph_data"]

    def test_in_memory_graph_data_makes_no_workspace_claim(self, monkeypatch, tmp_path):
        """A caller supplying the graph directly is not pointing at a persisted
        workspace, so there is no manifest for it to match and nothing to verify.
        Refusing there would block a legitimate embedded use.

        **Both are passed on purpose.** With `data_dir` omitted the check is
        skipped for a second reason -- there is no directory -- and a version that
        verified regardless would pass this test anyway. Supplying a `data_dir`
        that *would* fail is what isolates the seam: the graph is already in hand,
        so the files are never read and their state is irrelevant.
        """
        reached: list = []
        pipeline = self._pipeline_module(monkeypatch, reached)
        root = _workspace(tmp_path / "ws")
        (root / "node_features.pt").write_bytes(b"would fail verification")

        pipeline.DiagnosisPipeline(
            kg=KnowledgeGraph(),
            graph_data={"x_dict": {}, "edge_index_dict": {}, "num_nodes_dict": {}},
            data_dir=str(root),
            checkpoint_path=str(root / "ckpt.pt"),
        )

        assert reached == [], "the file-backed loader is not used for in-memory data"


def test_a_supplied_cohort_measurement_also_refuses_a_mixed_graph_workspace(tmp_path):
    """The case that would have been missed by putting graph binding inside the
    generated-cohort verifier: this path never touches a generated cohort."""
    from scripts.measure_scorer import artifact_digests

    root = _workspace(tmp_path / "ws", supplied=True)
    (root / "edge_indices.pt").write_bytes(b"from another workspace")

    with pytest.raises(ValueError, match="is not the edge_indices artifact"):
        artifact_digests(root / "ckpt.pt", root, "institutional_acceptance", "supplied")


def test_a_valid_supplied_cohort_needs_no_place_in_the_split_manifest(tmp_path):
    """It was never cut from this disease universe. It is identified by its own
    digest and is not subject to the generated splits' disjointness."""
    from scripts.measure_scorer import artifact_digests

    root = _workspace(tmp_path / "ws", supplied=True)
    (root / "ckpt.pt").write_bytes(b"weights")
    digests = artifact_digests(
        root / "ckpt.pt", root, "institutional_acceptance", "supplied"
    )

    manifest = json.loads((root / "split_manifest.json").read_text())
    assert "institutional_acceptance" not in json.dumps(manifest)
    assert "split_manifest" not in digests, "a supplied cohort carries no allocation"
    assert digests["samples"] is not None


def test_a_pre_migration_manifest_is_refused_with_a_rebuild_instruction(tmp_path):
    """Schema 1 bound only kg.json and the samples. The missing digests cannot be
    recovered afterwards, because only the writer could have vouched for them, so
    there is no migration and no unbound-digest path."""
    root = _workspace(tmp_path / "ws")
    manifest = json.loads((root / "split_manifest.json").read_text())
    manifest["schema_version"] = 1
    for role in ("node_features", "edge_indices", "num_nodes"):
        manifest["artifacts"].pop(role)
    (root / "split_manifest.json").write_text(json.dumps(manifest))

    for check in (verify_graph_artifacts, verify_generated_cohorts):
        with pytest.raises(ValueError, match="Rebuild it with"):
            check(root)


def test_a_manifest_at_the_current_schema_missing_a_graph_role_is_refused(tmp_path):
    """Not a schema question: a current-schema manifest that simply lacks a role
    still cannot say the tensors beside it are this graph's."""
    root = _workspace(tmp_path / "ws")
    manifest = json.loads((root / "split_manifest.json").read_text())
    manifest["artifacts"]["node_features"] = None
    (root / "split_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="records no digest for node_features"):
        verify_graph_artifacts(root)
