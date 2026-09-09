"""There is one way to produce a workspace, and it is enforced here.

A workspace is a single production event: a graph cut into a disease-level
allocation, exported to tensors, generated into cohorts, and bound by a manifest
that records all six artifacts by digest. Every consumer verifies that binding.

The project had three producers. `build_knowledge_graph.py` was the real one.
`train_model.py:generate_synthetic_data` was unreachable and has been removed.
`setup_demo.py` was neither: it was reachable, recommended by the deployment
script, and wrote a workspace with **its own sample generator that split
patients rather than diseases** -- the exact defect this branch exists to remove
-- and no manifest at all. It reported success, produced a plausible checkpoint,
and printed a startup command the clinical verifier could only refuse.

Deleting the unreachable copy while the operator-facing one remained is what
these tests exist to prevent recurring. The guarantee they enforce is not "the
demo happens to work today" but "the primitives have one caller", which is what
makes a fourth producer impossible to add quietly.

Module: tests/unit/test_workspace_producer.py
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytest.importorskip("torch")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# **The call that persists a workspace, and only in its persisting form.**
# `generate_training_samples` is what writes `split_manifest.json`, and it only
# writes when given an `output_dir`; called without one it returns cohorts in
# memory, which is a legitimate thing to do and is what most of the generator's
# own tests do. `export_graph_data` is deliberately NOT reserved here: it takes
# an optional `output_dir` too, tensors alone do not make a workspace, and a
# repository-wide ban on a generic method name would collide with any unrelated
# object that has one.
#
# Measured, not assumed. Injecting each form into `scripts/setup_demo.py`:
#   a direct persisting call        -> caught
#   a direct in-memory call         -> allowed, correctly
#   the same call behind an alias   -> NOT caught
# So this is a guardrail over the operator-facing persisted producer, not a
# sandbox. What it catches is the thing that actually happened three times: a
# script growing its own copy of the writing sequence by copying the production
# call. An author determined to alias around it can, and the routing tests below
# would not see that either, since such a producer would still call the writer as
# well. Both are worth having; neither is a proof.
PERSISTING_CALL = "generate_training_samples"
SOLE_CALLER = PROJECT_ROOT / "src" / "kg" / "workspace.py"


def _persisting_call_sites(path: Path):
    """Line numbers where `generate_training_samples` is called with an
    `output_dir` keyword. Definitions and in-memory calls are excluded."""
    tree = ast.parse(path.read_text())
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        called = (
            func.attr if isinstance(func, ast.Attribute)
            else func.id if isinstance(func, ast.Name)
            else None
        )
        if called != PERSISTING_CALL:
            continue
        if any(kw.arg == "output_dir" for kw in node.keywords):
            lines.append(node.lineno)
    return lines


def test_only_the_writer_persists_a_workspace():
    """Shipped code, not tests: a test may compose the primitives to build a
    fixture, but nothing an operator can run may write a manifest of its own."""
    offenders = {}
    for directory in ("src", "scripts"):
        for path in sorted((PROJECT_ROOT / directory).rglob("*.py")):
            if path == SOLE_CALLER:
                continue
            sites = _persisting_call_sites(path)
            if sites:
                offenders[str(path.relative_to(PROJECT_ROOT))] = sites

    assert offenders == {}, (
        f"a manifest is written outside src/kg/workspace.py: {offenders}. "
        "Route it through write_workspace instead of composing the ordering "
        "again -- that ordering is what binds the six artifacts into one event."
    )


def test_the_module_level_importers_hold_the_same_writer():
    """Same object, not merely the same name. Two functions that agree today
    diverge the first time one is extended.

    `test_gnn_inference` imports inside its function and so has no attribute to
    compare; the routing test below is what covers it.
    """
    import scripts.build_knowledge_graph as build
    import scripts.setup_demo as demo
    import src.kg.workspace as workspace

    assert build.write_workspace is workspace.write_workspace
    assert demo.write_workspace is workspace.write_workspace


class TestEveryProducerReachesTheWriter:
    """Identity says they hold the same function; these say they call it.

    An entry point could import the writer and still assemble a workspace
    beside it, which is what the AST check above cannot see through an alias
    and what identity alone does not exclude.
    """

    class _Reached(Exception):
        pass

    @classmethod
    def _sentinel(cls, monkeypatch, module):
        """Patched at the source *and* at the entry point.

        The three producers bind the writer differently -- two import it at
        module load, one inside the function -- so a sentinel that patched only
        one of those bindings would pass vacuously against the other style.
        """
        import src.kg.workspace as workspace

        def _stop(*args, **kwargs):
            raise cls._Reached

        monkeypatch.setattr(workspace, "write_workspace", _stop)
        if hasattr(module, "write_workspace"):
            monkeypatch.setattr(module, "write_workspace", _stop)

    def test_the_demo_reaches_it(self, monkeypatch, tmp_path):
        import scripts.setup_demo as demo

        self._sentinel(monkeypatch, demo)
        monkeypatch.setattr(
            "sys.argv", ["setup_demo.py", "--output-dir", str(tmp_path / "d")]
        )
        with pytest.raises(self._Reached):
            demo.main()

    def test_the_gnn_smoke_test_reaches_it(self, monkeypatch, tmp_path):
        import scripts.test_gnn_inference as gnn_smoke

        self._sentinel(monkeypatch, gnn_smoke)
        with pytest.raises(self._Reached):
            gnn_smoke.build_workspace(gnn_smoke.build_test_kg(), tmp_path / "w")

    def test_the_production_build_reaches_it(self, monkeypatch, tmp_path):
        from tests.unit.test_data_pipeline import TestBuildPathOrdering, _wide_kg_object

        build = TestBuildPathOrdering._stub_build_stages(monkeypatch, _wide_kg_object())
        TestBuildPathOrdering._satisfy_input_validation(tmp_path)
        self._sentinel(monkeypatch, build)

        with pytest.raises(self._Reached):
            build.build_knowledge_graph(
                external_dir=tmp_path, workspace=tmp_path / "ws",
                generate_samples=True, num_train=50, num_val=10,
            )


class TestTheDemoWorkspaceIsAProductionWorkspace:
    """Not "the demo runs" -- "what the demo writes, the clinic would accept"."""

    @staticmethod
    def _run(tmp_path, monkeypatch):
        import scripts.setup_demo as demo

        workspace = tmp_path / "demo"
        monkeypatch.setattr(
            "sys.argv", ["setup_demo.py", "--output-dir", str(workspace)]
        )
        demo.main()
        return workspace

    def test_every_verifier_accepts_it(self, tmp_path, monkeypatch):
        from src.evaluation.cohort import verify_generated_cohorts
        from src.kg.artifacts import (
            GRAPH_ARTIFACTS,
            verify_graph_artifacts,
            verify_graph_source,
        )

        workspace = self._run(tmp_path, monkeypatch)

        assert set(verify_graph_artifacts(workspace)) == set(GRAPH_ARTIFACTS)
        verify_graph_source(workspace / "kg.json", workspace)
        cohorts = verify_generated_cohorts(workspace)
        assert cohorts.verified == ("train", "val")
        assert cohorts.disjointness_measured is True
        assert cohorts.disjointness_claim_checked is True

    def test_the_cohorts_are_disjoint_at_the_disease_level(
        self, tmp_path, monkeypatch
    ):
        """The property the superseded generator could not provide: it drew a
        disease per patient for each partition independently, so a disease with
        patients on both sides was the norm rather than an accident."""
        from src.evaluation.cohort import verify_generated_cohorts

        workspace = self._run(tmp_path, monkeypatch)
        sets = verify_generated_cohorts(workspace).disease_sets

        assert sets["train"] and sets["val"], "a partition came out empty"
        assert not (sets["train"] & sets["val"])

    def test_it_writes_a_manifest_at_the_current_schema(self, tmp_path, monkeypatch):
        """Without one, `verify_graph_artifacts` refuses the directory and the
        startup command the script prints cannot work."""
        import json

        from src.kg.artifacts import (
            MANIFEST_FILENAME,
            SPLIT_MANIFEST_SCHEMA_VERSION,
        )

        workspace = self._run(tmp_path, monkeypatch)
        manifest = json.loads((workspace / MANIFEST_FILENAME).read_text())

        assert manifest["schema_version"] == SPLIT_MANIFEST_SCHEMA_VERSION
        assert manifest["disjoint"] is True

    def test_a_second_run_over_trained_checkpoints_writes_nothing(
        self, tmp_path, monkeypatch
    ):
        """Rebuilding the graph beneath checkpoints leaves them paired with a
        graph they were never trained on, and nothing in a checkpoint says so.

        **The refusal has to precede the writing, and "it raised" does not show
        that.** The generator refuses on the same grounds, so a writer that had
        dropped its own check would still raise here -- after `kg.json` and the
        three tensors had been rewritten, which is the damage. Comparing bytes
        and modification times is what distinguishes the two.
        """
        from src.utils.fingerprint import file_sha256

        workspace = self._run(tmp_path, monkeypatch)
        artifacts = sorted(
            p for p in workspace.iterdir() if p.is_file()
        )
        before = {
            p.name: (p.stat().st_mtime_ns, file_sha256(p)) for p in artifacts
        }
        # The six the manifest binds, plus the manifest itself.
        assert len(before) == 7, f"expected seven files, got {sorted(before)}"

        ckpt_dir = workspace / "checkpoints" / "gat"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "last.pt").write_bytes(b"trained on the graph that is there now")

        with pytest.raises(FileExistsError, match="already holds trained checkpoints"):
            self._run(tmp_path, monkeypatch)

        after = {
            p.name: (p.stat().st_mtime_ns, file_sha256(p))
            for p in workspace.iterdir()
            if p.is_file()
        }
        assert after == before, "a refused rebuild rewrote the workspace anyway"


class TestTheWriterCarriesItsOwnCorrectness:
    """A caller that supplies no `preflight` must still be refused before a write.

    `preflight` exists for wording, and wording is not a correctness mechanism.
    While the coverage check lived only in the callback, a direct caller passing
    `True`, a float, a negative or an undersized budget allocated, wrote
    `kg.json` and the three tensors, and was refused only inside the generator —
    reconstructing the half-written workspace this writer exists to prevent.
    Every case below is checked in library terms, with no callback in sight.
    """

    @staticmethod
    def _demo_kg():
        from scripts.setup_demo import build_demo_kg

        return build_demo_kg()

    # The demo graph allocates 3 training and 1 validation disease at f=0.2.
    @pytest.mark.parametrize(
        "num_train,num_val,message",
        [
            (True, 5, "must be an integer"),
            (5, True, "must be an integer"),
            (200.0, 5, "must be an integer"),
            (5, 50.0, "must be an integer"),
            (-1, 5, "must be >= 0"),
            (5, -1, "must be >= 0"),
            (2, 5, "cannot cover"),
            (200, 0, "cannot cover"),
        ],
        ids=["bool-train", "bool-val", "float-train", "float-val",
             "negative-train", "negative-val", "undersized-train",
             "undersized-val"],
    )
    def test_a_budget_the_writer_cannot_honour_writes_nothing(
        self, tmp_path, num_train, num_val, message
    ):
        from src.kg.workspace import SampleBudget, write_workspace

        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "kg.json").write_bytes(b'{"pre-existing": true}')
        before = {
            p.name: (p.stat().st_mtime_ns, p.read_bytes())
            for p in workspace.iterdir()
        }

        with pytest.raises(ValueError, match=message):
            write_workspace(
                self._demo_kg(),
                workspace,
                feature_dim=8,
                samples=SampleBudget(
                    num_train=num_train, num_val=num_val, val_disease_fraction=0.2
                ),
            )

        after = {
            p.name: (p.stat().st_mtime_ns, p.read_bytes())
            for p in workspace.iterdir()
        }
        assert after == before, "a refused budget wrote into the workspace"

    def test_a_refusal_does_not_even_create_the_directory(self, tmp_path):
        """The refusals need a path, not a place. An empty directory left behind
        is harmless, but it is also a lie about what happened."""
        from src.kg.workspace import SampleBudget, write_workspace

        workspace = tmp_path / "never"
        with pytest.raises(ValueError):
            write_workspace(
                self._demo_kg(), workspace, feature_dim=8,
                samples=SampleBudget(num_train=-1, num_val=5),
            )

        assert not workspace.exists()

    def test_budgets_that_cover_the_allocation_are_written(self, tmp_path):
        """Otherwise the tests above would pass against a writer that refuses
        everything."""
        from src.evaluation.cohort import verify_generated_cohorts
        from src.kg.workspace import SampleBudget, write_workspace

        workspace = tmp_path / "ws"
        written = write_workspace(
            self._demo_kg(), workspace, feature_dim=8,
            samples=SampleBudget(num_train=20, num_val=5, val_disease_fraction=0.2),
        )

        assert written.manifest is not None
        assert verify_generated_cohorts(workspace).verified == ("train", "val")


class TestAGraphOnlyBuildSaysWhatItIs:
    """`--generate-samples` is optional, and the output of omitting it is not.

    A graph export is a real mode — `compute_shortest_paths.py` reads `kg.json`,
    `build_index.py` reads the tensors, and the API serves path-reasoning from
    `kg.json` alone — but it is not trainable and not GNN-servable, because
    nothing records which export those tensors are. The build used to print
    `train_model.py` as the next step regardless, so an operator learned that at
    the end of a thirty-minute build rather than at the start.

    Cohorts also cannot be added afterwards: only the writer that exported the
    tensors can vouch for their digests, so completing one means rebuilding.
    """

    @staticmethod
    def _build(monkeypatch, tmp_path, capsys, **kwargs):
        from tests.unit.test_data_pipeline import (
            TestBuildPathOrdering,
            _wide_kg_object,
        )

        build = TestBuildPathOrdering._stub_build_stages(monkeypatch, _wide_kg_object())
        TestBuildPathOrdering._satisfy_input_validation(tmp_path)
        build.build_knowledge_graph(
            external_dir=tmp_path, workspace=tmp_path / "ws", **kwargs
        )
        return capsys.readouterr().out

    def test_it_does_not_offer_a_training_command(
        self, monkeypatch, tmp_path, capsys
    ):
        out = self._build(monkeypatch, tmp_path, capsys, generate_samples=False)

        assert "train_model.py" not in out, "offered training on an unusable workspace"
        assert "CANNOT be trained on" in out
        assert "--generate-samples" in out, "did not say how to complete it"
        assert "compute_shortest_paths.py" in out, "withheld a step that does work"

    def test_a_complete_build_still_offers_one(self, monkeypatch, tmp_path, capsys):
        """Otherwise the test above would pass against a build that never
        mentions training at all."""
        out = self._build(
            monkeypatch, tmp_path, capsys,
            generate_samples=True, num_train=60, num_val=20,
        )

        assert "train_model.py" in out
        assert "CANNOT be trained on" not in out
