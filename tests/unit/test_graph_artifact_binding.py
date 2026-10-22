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
institutional evaluation went on consuming a mixed workspace. Measurement's
supplied-cohort case is pinned at its entry point, from the reads it scores
(`test_measurement_reads_once.py`).

Module: tests/unit/test_graph_artifact_binding.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.core.types import DataSource, NodeType
from src.kg.graph import Edge, KnowledgeGraph, Node, NodeID
from src.evaluation.cohort import GRAPH_ARTIFACTS
from src.kg.artifacts import read_split_manifest, verify_graph_reads
from tests.fixtures.generated_workspace import write_generated_workspace
from tests.fixtures.workspace_reads import verify_workspace_reads


def _workspace(root: Path):
    # A real, loadable `kg.json` — the API path opens it before the pipeline is
    # built, so a stand-in byte string would fail there for the wrong reason.
    # Written before the manifest, which binds these bytes.
    root.mkdir(parents=True, exist_ok=True)
    kg = KnowledgeGraph()
    kg.add_node(Node(id=NodeID(source=DataSource.MONDO, local_id=f"MONDO:{root.name}"),
                     node_type=NodeType.DISEASE, name=root.name))
    kg.save_json(str(root / "kg.json"))
    write_generated_workspace(root, train_ids=[0, 1, 2], val_ids=[3])
    return root


def test_the_recorded_digests_are_the_exact_exported_bytes(tmp_path):
    from src.utils.fingerprint import file_sha256

    root = _workspace(tmp_path / "ws")
    manifest = json.loads((root / "split_manifest.json").read_text())

    for role, filename in GRAPH_ARTIFACTS.items():
        assert manifest["artifacts"][role] == file_sha256(root / filename)
    bound, _ = verify_workspace_reads(root)
    assert bound == {role: manifest["artifacts"][role] for role in GRAPH_ARTIFACTS}


@pytest.mark.parametrize("role", sorted(GRAPH_ARTIFACTS))
def test_replacing_any_single_graph_artifact_is_refused(tmp_path, role):
    """A same-shaped artifact from another workspace passes structural
    fingerprinting. Only the recorded bytes catch it. Another workspace's valid
    file, not junk: the checks parse before they compare (contract M2.1), so
    junk is refused by the reader instead."""
    root = _workspace(tmp_path / "ws")
    other = _workspace(tmp_path / "other")
    (root / GRAPH_ARTIFACTS[role]).write_bytes((other / GRAPH_ARTIFACTS[role]).read_bytes())

    with pytest.raises(ValueError, match="is not the graph" if role == "kg"
                       else f"is not the {role} artifact"):
        verify_workspace_reads(root)


class TestTrainingRefusesBeforeItActs:
    """The refusal has to precede the side effects, not merely exist.

    A role-map helper used to carry the verification, and a test calling it
    directly proved only that the helper *can* reject. By the time `train` reached
    it, the run directories existed, `config.yaml` was written, the graph tensors
    were loaded, dataloaders and a model were built, a Trainer was constructed and
    a resume checkpoint may have been loaded. Whether the workspace is sound is
    knowable from the files alone, so it is knowable before any of that.

    Since contract M2.1's S8 the files are read once (`read_training_inputs`):
    those reads parse the graph tensors and the samples, and the check compares
    them before the dataloaders, the model, the Trainer and any run artifact.
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
        # An explicit device: `auto` refuses on a host without CUDA (M3a)
        # before any of the checks these tests are about.
        overrides.setdefault("device", "cpu")
        return train_model.TrainConfig(
            data_dir=str(root),
            run_dir=str(tmp_path / "run"),
            **overrides,
        )

    @pytest.mark.parametrize(
        "break_it,expected",
        [
            # Another workspace's valid file: refused by digest, naming it.
            (lambda root, other: (root / "node_features.pt").write_bytes(
                (other / "node_features.pt").read_bytes()),
             "node_features.pt is not the node_features artifact"),
            # Junk: refused by the reader that could not parse it, naming it.
            (lambda root, other: (root / "node_features.pt").write_bytes(b"another ws"),
             "node_features.pt could not be parsed"),
            (lambda root, other: (root / "train_samples.json").write_text("[]"),
             "train_samples.json is not the file"),
            (lambda root, other: (root / "kg.json").write_bytes(
                (other / "kg.json").read_bytes()),
             "kg.json is not the kg artifact"),
            # The samples reader's own refusal, which lists the splits there are.
            (lambda root, other: (root / "val_samples.json").unlink(),
             "Samples file not found: .*val_samples.json — this workspace has: train"),
            (lambda root, other: (root / "split_manifest.json").unlink(),
             "(?i)rebuild it with"),
        ],
        ids=["mixed-graph", "junk-graph", "replaced-cohort", "crossed-kg",
             "missing-val", "no-manifest"],
    )
    def test_no_stage_and_no_run_artifact_is_reached(
        self, monkeypatch, tmp_path, break_it, expected
    ):
        train_model = self._forbid(
            monkeypatch,
            ["create_dataloaders", "create_model_from_config", "Trainer",
             "read_resume_parent"],
        )
        root = _workspace(tmp_path / "ws")
        other = _workspace(tmp_path / "other")
        break_it(root, other)

        # A missing file is the readers' `FileNotFoundError`, an `OSError`.
        with pytest.raises((ValueError, OSError), match=expected):
            train_model.train(self._config(train_model, root, tmp_path))

        assert not (tmp_path / "run").exists(), "run directory was created"
        assert not (root / "runs").exists(), "a run directory was allocated"

    def test_a_sound_workspace_reaches_the_next_stage(self, monkeypatch, tmp_path):
        """The refusal must be about the workspace, not about the ordering itself:
        an unbroken workspace passes the preflight and goes on to build its
        dataloaders."""
        train_model = self._forbid(monkeypatch, ["create_dataloaders"])
        root = _workspace(tmp_path / "ws")

        with pytest.raises(AssertionError, match="create_dataloaders ran"):
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
        """Records whether the model was built, and stands in for the rest of
        the model build.

        **The stand-ins are what let a build past the binding check finish.**
        These workspaces carry a placeholder checkpoint, and a one-node
        `kg.json` beside the shared fixture's tensors, which were exported from
        another graph: nothing here is a graph and a checkpoint a model can be
        built from. A requested model that cannot be built is refused
        (`PipelineBuildError`, M1), so without them every case that passes the
        binding check would end in that refusal instead of in what it is about.

        The stand-in keeps a checkpoint read's identity, as the real build does
        (contract M3c), so a pipeline built over it states a verified checkpoint
        rather than a model supplied in memory. Callers construct with
        `device="cpu"` (or set `SHEPHERD_DEVICE=cpu` for the API): with no device
        named a checkpoint asks for CUDA, and on a host without it that refusal
        comes before the workspace these tests are about.
        """
        import src.inference.pipeline as pipeline
        from src.utils.fingerprint import ReadIdentity

        def _model_built(self, checkpoint_path, device):
            reached.append("model")
            self._checkpoint_identity = ReadIdentity(
                path=Path(checkpoint_path), sha256="c" * 64)
            return object()

        monkeypatch.setattr(
            pipeline.DiagnosisPipeline, "_load_model_from_checkpoint", _model_built,
        )
        monkeypatch.setattr(
            pipeline.DiagnosisPipeline, "_precompute_node_embeddings",
            lambda self, device=None: setattr(
                self, "_node_embeddings", {"phenotype": object(), "disease": object()}
            ),
        )
        return pipeline

    def test_a_mixed_workspace_is_refused_before_the_model_is_built(
        self, monkeypatch, tmp_path
    ):
        reached: list = []
        pipeline = self._pipeline_module(monkeypatch, reached)
        root = _workspace(tmp_path / "ws")
        other = _workspace(tmp_path / "other")
        (root / "node_features.pt").write_bytes((other / "node_features.pt").read_bytes())

        with pytest.raises(ValueError, match="is not the node_features artifact"):
            pipeline.DiagnosisPipeline(
                graph=KnowledgeGraph.read_json(root / "kg.json"), data_dir=str(root),
                checkpoint_path=str(root / "ckpt.pt"), device="cpu",
            )

        assert reached == [], "the model was built after a digest mismatch"

    def test_a_sound_workspace_goes_on_to_build_the_model(self, monkeypatch, tmp_path):
        """The refusal is about the workspace, not about the ordering itself."""
        reached: list = []
        pipeline = self._pipeline_module(monkeypatch, reached)
        root = _workspace(tmp_path / "ws")
        (root / "ckpt.pt").write_bytes(b"weights")

        pipeline.DiagnosisPipeline(
            graph=KnowledgeGraph.read_json(root / "kg.json"), data_dir=str(root),
            checkpoint_path=str(root / "ckpt.pt"), device="cpu",
        )

        assert reached == ["model"]

    def test_a_file_backed_pipeline_without_a_graph_read_is_refused(
        self, monkeypatch, tmp_path
    ):
        """An optional check is no check for the caller who omits it, and that
        caller is the one it exists for.

        `verify_graph_reads` alone proves only that `data_dir` is consistent
        with itself; the `kg` object stays bound to nothing, so workspace B's
        graph can still be paired with workspace A's tensors. Nothing legitimate
        is lost by requiring it: a graph built in memory belongs on the
        `graph_data` seam, and a graph loaded from disk passes the `GraphRead`
        `KnowledgeGraph.read_json` returned.
        """
        reached: list = []
        pipeline = self._pipeline_module(monkeypatch, reached)
        root = _workspace(tmp_path / "ws")
        (root / "ckpt.pt").write_bytes(b"weights")

        with pytest.raises(ValueError, match="requires graph="):
            pipeline.DiagnosisPipeline(
                kg=KnowledgeGraph(), data_dir=str(root),
                checkpoint_path=str(root / "ckpt.pt"), device="cpu",
            )

        assert reached == [], "the model was built without a source binding"

    def test_in_memory_graph_data_makes_no_workspace_claim(self, monkeypatch, tmp_path):
        """A caller supplying the graph directly is not pointing at a persisted
        workspace, so there is no manifest for it to match and its files are not
        read -- and, for the same reason, no checkpoint is loaded over that graph
        (contract M2.4): nothing shows the checkpoint was trained on it.

        **Both are passed on purpose.** Supplying a `data_dir` that *would* fail
        the workspace check isolates the seam: the files are never read, and the
        refusal is the checkpoint's, not the workspace's. The real loader runs
        here, so the refusal is shown where the checkpoint would be read.
        """
        import src.inference.pipeline as pipeline
        from src.inference.pipeline import PipelineBuildError
        from tests.fixtures.opens import count_opens

        root = _workspace(tmp_path / "ws")
        other = _workspace(tmp_path / "other")
        (root / "node_features.pt").write_bytes((other / "node_features.pt").read_bytes())
        (root / "ckpt.pt").write_bytes(b"weights")

        with count_opens() as opens:
            with pytest.raises(PipelineBuildError, match="supplied in memory"):
                pipeline.DiagnosisPipeline(
                    kg=KnowledgeGraph(),
                    graph_data={"x_dict": {}, "edge_index_dict": {}, "num_nodes_dict": {}},
                    data_dir=str(root),
                    checkpoint_path=str(root / "ckpt.pt"),
                    device="cpu",
                )

        assert not [path for path in root.iterdir() if opens.opens(path)], (
            "a workspace file, or the checkpoint, was opened for in-memory graph data"
        )


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

    # The one place a persisted manifest is read: every consumer goes through it.
    with pytest.raises(ValueError, match="Rebuild it with"):
        read_split_manifest(root)


def test_a_manifest_at_the_current_schema_missing_a_graph_role_is_refused(tmp_path):
    """Not a schema question: a current-schema manifest that simply lacks a role
    still cannot say the tensors beside it are this graph's."""
    root = _workspace(tmp_path / "ws")
    manifest = json.loads((root / "split_manifest.json").read_text())
    manifest["artifacts"]["node_features"] = None
    (root / "split_manifest.json").write_text(json.dumps(manifest))

    from src.kg.storage.file_storage import read_graph_artifacts

    with pytest.raises(ValueError, match="records no digest for node_features"):
        verify_graph_reads(read_split_manifest(root), read_graph_artifacts(root).reads)


class TestTheCompositionMustNotCrossWorkspaces:
    """Two internally valid workspaces, paired.

    `verify_graph_reads` proves a workspace is consistent with itself. It
    cannot see a caller that takes one workspace's `kg.json` and another's
    tensors -- which is exactly what the API allowed, since `SHEPHERD_KG_PATH` and
    `SHEPHERD_DATA_DIR` are resolved independently. Embedding rows then come from
    one graph and the node-id mapping that reads them from another, and
    same-shaped workspaces pass every structural check on the way.
    """

    @staticmethod
    def _two_workspaces(tmp_path):
        a = _workspace(tmp_path / "a")
        b = _workspace(tmp_path / "b")
        for root in (a, b):
            (root / "ckpt.pt").write_bytes(b"weights")
        return a, b

    def test_a_crossed_pair_is_refused(self, monkeypatch, tmp_path):
        reached: list = []
        pipeline = TestTheClinicalPathIsAGraphConsumer._pipeline_module(
            monkeypatch, reached
        )
        a, b = self._two_workspaces(tmp_path)

        with pytest.raises(ValueError, match="is not the graph"):
            pipeline.DiagnosisPipeline(
                graph=KnowledgeGraph.read_json(b / "kg.json"), data_dir=str(a),
                checkpoint_path=str(a / "ckpt.pt"), device="cpu",
            )

        assert reached == [], "the model was built on a crossed pair"

    def test_each_workspace_is_valid_on_its_own(self, tmp_path):
        """Otherwise the test above would prove only that one of them is broken."""
        a, b = self._two_workspaces(tmp_path)
        assert set(verify_workspace_reads(a)[0]) == set(GRAPH_ARTIFACTS)
        assert set(verify_workspace_reads(b)[0]) == set(GRAPH_ARTIFACTS)

    def test_a_matching_pair_proceeds(self, monkeypatch, tmp_path):
        reached: list = []
        pipeline = TestTheClinicalPathIsAGraphConsumer._pipeline_module(
            monkeypatch, reached
        )
        a, _ = self._two_workspaces(tmp_path)

        pipeline.DiagnosisPipeline(
            graph=KnowledgeGraph.read_json(a / "kg.json"), data_dir=str(a),
            checkpoint_path=str(a / "ckpt.pt"), device="cpu",
        )

        assert reached == ["model"]

    def test_the_bound_graph_may_live_at_another_path(self, monkeypatch, tmp_path):
        """A path is not an identity in this project; the bytes are. Requiring
        the graph's file to *be* `data_dir/kg.json` would also close the
        crossing, and would break a deployment that mounts or copies the file
        elsewhere."""
        reached: list = []
        pipeline = TestTheClinicalPathIsAGraphConsumer._pipeline_module(
            monkeypatch, reached
        )
        a, _ = self._two_workspaces(tmp_path)
        elsewhere = tmp_path / "mounted_kg.json"
        elsewhere.write_bytes((a / "kg.json").read_bytes())

        pipeline.DiagnosisPipeline(
            graph=KnowledgeGraph.read_json(elsewhere), data_dir=str(a),
            checkpoint_path=str(a / "ckpt.pt"), device="cpu",
        )

        assert reached == ["model"]

    def test_a_late_initialization_failure_publishes_nothing(
        self, monkeypatch, tmp_path
    ):
        """A failure between two publications would leave a pipeline visible while
        the caller reports that initialization failed. Everything is computed
        first, then published together."""
        import src.api.main as api

        pipeline = TestTheClinicalPathIsAGraphConsumer._pipeline_module(
            monkeypatch, []
        )
        a, _ = self._two_workspaces(tmp_path)
        monkeypatch.setattr(api.app_state, "kg", None, raising=False)
        monkeypatch.setattr(api.app_state, "pipeline", None, raising=False)
        monkeypatch.setattr(
            pipeline.DiagnosisPipeline, "get_pipeline_config",
            lambda self: (_ for _ in ()).throw(RuntimeError("late failure")),
        )
        monkeypatch.setenv("SHEPHERD_KG_PATH", str(a / "kg.json"))
        monkeypatch.setenv("SHEPHERD_DATA_DIR", str(a))
        monkeypatch.setenv("SHEPHERD_CHECKPOINT_PATH", str(a / "ckpt.pt"))
        monkeypatch.setenv("SHEPHERD_DEVICE", "cpu")

        with pytest.raises(RuntimeError, match="late failure"):
            api.initialize_pipeline()

        assert api.app_state.pipeline is None, "a pipeline was published anyway"
        assert api.app_state.kg is None

    def test_the_api_commits_no_state_before_the_composition_is_checked(
        self, monkeypatch, tmp_path
    ):
        """A refused composition must not leave a graph published in app state as
        though it had been accepted."""
        import src.api.main as api

        a, b = self._two_workspaces(tmp_path)
        monkeypatch.setattr(api.app_state, "kg", None, raising=False)
        monkeypatch.setattr(api.app_state, "pipeline", None, raising=False)
        monkeypatch.setenv("SHEPHERD_KG_PATH", str(b / "kg.json"))
        monkeypatch.setenv("SHEPHERD_DATA_DIR", str(a))
        # Without a checkpoint the pipeline consumes no graph at all and verifies
        # nothing, which is correct and would make this test vacuous.
        monkeypatch.setenv("SHEPHERD_CHECKPOINT_PATH", str(a / "ckpt.pt"))
        # Without a device named the checkpoint asks for CUDA and, on a host
        # without it, is refused before the graph is read (contract M3c).
        monkeypatch.setenv("SHEPHERD_DEVICE", "cpu")

        with pytest.raises(ValueError, match="is not the graph"):
            api.initialize_pipeline()

        assert api.app_state.kg is None, "a refused graph reached app state"
        assert api.app_state.pipeline is None
