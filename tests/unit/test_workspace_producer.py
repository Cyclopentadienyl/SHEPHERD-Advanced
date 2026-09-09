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
#   a direct persisting call            -> caught
#   a direct in-memory call             -> allowed, correctly
#   the same call behind an alias       -> NOT caught
#   output_dir passed positionally      -> NOT caught
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


def test_no_script_copies_the_persisting_call():
    """A copy-paste regression guard, and only that.

    It does not prove the writer is the only thing that can persist a workspace
    — an alias or a positional `output_dir` walks past it, both measured above.
    What carries that claim in practice is the routing sentinels and the
    end-to-end verifier tests below. This catches the specific thing that
    happened three times: a script acquiring its own copy of the production
    call.
    """
    offenders = {}
    for directory in ("src", "scripts"):
        for path in sorted((PROJECT_ROOT / directory).rglob("*.py")):
            if path == SOLE_CALLER:
                continue
            sites = _persisting_call_sites(path)
            if sites:
                offenders[str(path.relative_to(PROJECT_ROOT))] = sites

    assert offenders == {}, (
        f"a copy of the persisting call appears outside src/kg/workspace.py: "
        f"{offenders}. Route it through write_workspace instead of composing "
        "the ordering again -- that ordering is what binds the six artifacts "
        "into one production event."
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


class TestTheRefusalOrderHasNoHookInIt:
    """There is no callback left to run before the writer's own check.

    The reviewed shape ran an optional caller hook between the allocation and
    the writer's coverage check. Making the writer's check authoritative meant
    moving it in front of that hook -- at which point the hook could no longer
    produce the message an operator needed, because the writer would already
    have refused. A parameter with no remaining job is worse than no parameter,
    so the vocabulary moved into the writer and the hook went away.

    What replaces "the hook cannot mask the check" is stronger: nothing runs
    between the allocation and the check at all.
    """

    def test_the_writer_takes_no_callback(self):
        import inspect

        from src.kg.workspace import write_workspace

        parameters = inspect.signature(write_workspace).parameters
        assert "preflight" not in parameters
        assert {"train_label", "val_label"} <= set(parameters)

    def test_the_refusal_speaks_the_caller_s_vocabulary(self, tmp_path):
        """What the hook existed for, without the hook: an operator sees the
        flag they typed, and the check that produced it is the writer's."""
        from scripts.setup_demo import build_demo_kg
        from src.kg.workspace import WorkspaceRefusal, SampleBudget, write_workspace

        with pytest.raises(WorkspaceRefusal, match=r"--num-val=0 cannot cover"):
            write_workspace(
                build_demo_kg(), tmp_path / "ws", feature_dim=8,
                samples=SampleBudget(num_train=50, num_val=0,
                                     val_disease_fraction=0.2),
                train_label="--num-train", val_label="--num-val",
            )
        assert not (tmp_path / "ws").exists()

    def test_the_production_build_reports_it_with_its_own_flags(
        self, monkeypatch, tmp_path
    ):
        """End to end through the real entry point, which is where the previous
        shape's message actually mattered."""
        from tests.unit.test_data_pipeline import (
            TestBuildPathOrdering,
            _wide_kg_object,
        )

        build = TestBuildPathOrdering._stub_build_stages(monkeypatch, _wide_kg_object())
        TestBuildPathOrdering._satisfy_input_validation(tmp_path)

        with pytest.raises(SystemExit) as excinfo:
            build.build_knowledge_graph(
                external_dir=tmp_path, workspace=tmp_path / "ws",
                generate_samples=True, num_train=1, num_val=1,
            )
        message = str(excinfo.value)
        assert "--num-train=1 cannot cover" in message
        assert "Nothing was written" in message
        assert not (tmp_path / "ws").exists()


def test_a_failure_after_the_graph_is_written_is_not_called_unwritten(
    monkeypatch, tmp_path
):
    """"Nothing was written" has to be true when the build says it.

    `WorkspaceRefusal` is raised only before the writer touches the workspace, so
    catching it and adding that sentence is sound. Catching `ValueError` broadly
    would attach the same sentence to a generator failure, which happens with
    `kg.json` and three tensors already on disk.
    """
    import src.kg.sample_generator as generator
    from tests.unit.test_data_pipeline import TestBuildPathOrdering, _wide_kg_object

    build = TestBuildPathOrdering._stub_build_stages(monkeypatch, _wide_kg_object())
    TestBuildPathOrdering._satisfy_input_validation(tmp_path)
    workspace = tmp_path / "ws"

    def _fail_after_the_graph(*args, **kwargs):
        raise ValueError("the generator refused, and the graph is already there")

    monkeypatch.setattr(generator, "generate_training_samples", _fail_after_the_graph)

    with pytest.raises(ValueError, match="already there"):
        build.build_knowledge_graph(
            external_dir=tmp_path, workspace=workspace,
            generate_samples=True, num_train=60, num_val=20,
        )

    assert (workspace / "kg.json").exists(), "the premise of this test is gone"


class TestEveryKnowableInputIsRefusedBeforeAWrite:
    """The writer acts on its inputs in sequence, and the sequence is the trap.

    `feature_dim` is used by the export, which runs after `kg.json` is saved;
    `min_phenotypes` is used to build eligible profiles, and its domain was only
    checked later still, inside generation. Both were fully knowable before any
    of it. A value rejected at the point of use leaves a workspace half written
    by an input that was wrong from the start.

    `feature_dim=0` is the case that makes this more than tidiness: `torch.randn`
    accepts it and writes real tensors with no features in them. That workspace
    passes every digest check and every verifier, and the model built from it
    has nothing to read.
    """

    @staticmethod
    def _demo_kg():
        from scripts.setup_demo import build_demo_kg

        return build_demo_kg()

    @pytest.mark.parametrize(
        "kwargs,message",
        [
            ({"feature_dim": 0}, "feature_dim must be >= 1"),
            ({"feature_dim": -1}, "feature_dim must be >= 1"),
            ({"feature_dim": 1.5}, "feature_dim must be an integer"),
            ({"feature_dim": True}, "feature_dim must be an integer"),
            ({"min_phenotypes": True}, "min_phenotypes must be an integer"),
            ({"min_phenotypes": 1.5}, "min_phenotypes must be an integer"),
            ({"min_phenotypes": 0}, "min_phenotypes must be >= 1"),
            ({"min_phenotypes": -1}, "min_phenotypes must be >= 1"),
        ],
        ids=["dim-zero", "dim-negative", "dim-fractional", "dim-bool",
             "min-bool", "min-fractional", "min-zero", "min-negative"],
    )
    def test_it_writes_nothing(self, tmp_path, kwargs, message):
        from src.kg.workspace import (
            SampleBudget,
            WorkspaceRefusal,
            write_workspace,
        )

        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "kg.json").write_bytes(b'{"pre-existing": true}')
        before = {
            p.name: (p.stat().st_mtime_ns, p.read_bytes())
            for p in workspace.iterdir()
        }

        budget_kwargs = {k: v for k, v in kwargs.items() if k == "min_phenotypes"}
        write_kwargs = {k: v for k, v in kwargs.items() if k != "min_phenotypes"}
        with pytest.raises(WorkspaceRefusal, match=message):
            write_workspace(
                self._demo_kg(),
                workspace,
                samples=SampleBudget(
                    num_train=20, num_val=5, val_disease_fraction=0.2,
                    **budget_kwargs,
                ),
                **{"feature_dim": 8, **write_kwargs},
            )

        after = {
            p.name: (p.stat().st_mtime_ns, p.read_bytes())
            for p in workspace.iterdir()
        }
        assert after == before, "a refused input wrote into the workspace"

    def test_a_graph_only_write_is_refused_on_feature_dim_too(self, tmp_path):
        """No budgets means no phenotype floor to check, and the export still
        happens — so the width still has to be right."""
        from src.kg.workspace import WorkspaceRefusal, write_workspace

        workspace = tmp_path / "never"
        with pytest.raises(WorkspaceRefusal, match="feature_dim must be >= 1"):
            write_workspace(self._demo_kg(), workspace, feature_dim=0)

        assert not workspace.exists()

    def test_the_rules_are_the_ones_the_later_stages_enforce(self):
        """Imported, not restated. Two copies agree until one is extended."""
        import src.kg.graph as graph
        import src.kg.sample_generator as generator
        import src.kg.workspace as workspace

        assert workspace.validate_phenotype_count is generator.validate_phenotype_count
        # The export enforces the same function the writer checks ahead of it.
        source = Path(graph.__file__).read_text()
        assert "validate_feature_dim(feature_dim)" in source


class TestTheSeedIsProvenanceNotOnlyRandomness:
    """`derive_stream` stringifies the seed, so almost any value yields *a*
    stream — which is why this looked bypassable and was reported as such.

    It is not. `DiseaseAllocation` keeps the object it was handed and the
    manifest serialises it, so a non-JSON seed was found by `json.dump` with the
    graph, both cohorts and part of the manifest already on disk — and `json.dump`
    writes as it walks, so it left a manifest truncated at exactly that key: a
    file that parses as nothing and reads as a workspace that has one. An
    `object()` whose repr carries a process address also makes the derived
    stream unreproducible while looking deterministic.
    """

    @staticmethod
    def _demo_kg():
        from scripts.setup_demo import build_demo_kg

        return build_demo_kg()

    def _write(self, workspace, seed):
        from src.kg.workspace import SampleBudget, write_workspace

        return write_workspace(
            self._demo_kg(), workspace, feature_dim=8,
            samples=SampleBudget(
                num_train=20, num_val=5, val_disease_fraction=0.2, seed=seed
            ),
        )

    @pytest.mark.parametrize(
        "seed", [True, False, 1.5, "42", b"42", bytearray(b"42"), object(), None],
        ids=["true", "false", "float", "str", "bytes", "bytearray", "object", "none"],
    )
    def test_a_seed_that_is_not_an_integer_writes_nothing(self, tmp_path, seed):
        from src.kg.workspace import WorkspaceRefusal

        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "kg.json").write_bytes(b'{"pre-existing": true}')
        before = {
            p.name: (p.stat().st_mtime_ns, p.read_bytes())
            for p in workspace.iterdir()
        }

        with pytest.raises(WorkspaceRefusal, match="seed must be an integer"):
            self._write(workspace, seed)

        after = {
            p.name: (p.stat().st_mtime_ns, p.read_bytes())
            for p in workspace.iterdir()
        }
        assert after == before, "a refused seed wrote into the workspace"

    @pytest.mark.parametrize("seed", [0, -1, 2 ** 128], ids=["zero", "negative", "huge"])
    def test_any_integer_is_a_seed(self, tmp_path, seed):
        """The stream comes from the decimal form and nothing downstream packs
        it into a machine word, so a bound here would be invention. This is the
        half that keeps the validator from becoming restrictive."""
        from src.evaluation.cohort import verify_generated_cohorts

        workspace = tmp_path / f"ws{abs(seed)}"
        written = self._write(workspace, seed)

        assert written.manifest["allocation"]["seed"] == seed
        assert verify_generated_cohorts(workspace).verified == ("train", "val")

    def test_the_allocation_refuses_a_bad_seed_for_direct_callers_too(self):
        """The writer is not the only caller. `allocate_diseases` is public and
        a manifest built from its result carries the same seed."""
        from src.kg.disease_allocation import allocate_diseases
        from src.kg.sample_generator import build_eligible_disease_profiles

        eligible = build_eligible_disease_profiles(self._demo_kg(), 2)
        with pytest.raises(ValueError, match="seed must be an integer"):
            allocate_diseases(eligible, 0.2, seed=b"42")

    def test_a_manifest_is_written_whole_or_not_at_all(self, tmp_path):
        """What made a bad value corrupt a file rather than merely fail.

        Every field reaching the manifest is constrained today, but that is a
        property of the current field list — re-proved by hand each time someone
        adds one. Serialising the text first makes it structural.
        """
        from src.kg.disease_allocation import DiseaseAllocation, allocate_diseases
        from src.kg.sample_generator import (
            build_eligible_disease_profiles,
            generate_training_samples,
        )

        workspace = tmp_path / "ws"
        workspace.mkdir()
        eligible = build_eligible_disease_profiles(self._demo_kg(), 2)
        sound = allocate_diseases(eligible, 0.2, seed=42)
        # A seed no encoder can take, past the validator by construction: the
        # point is the writing, not the checking.
        smuggled = DiseaseAllocation(
            train=sound.train, val=sound.val,
            val_fraction_requested=sound.val_fraction_requested,
            seed=object(), universe_digest=sound.universe_digest,
        )

        with pytest.raises(TypeError):
            generate_training_samples(
                kg=self._demo_kg(), allocation=smuggled,
                num_train=20, num_val=5, output_dir=workspace,
                graph_digests={r: "0" * 64 for r in
                               ("kg", "node_features", "edge_indices", "num_nodes")},
            )

        assert not (workspace / "split_manifest.json").exists(), (
            "a manifest that could not be serialised was left on disk"
        )
        assert not list(workspace.glob("*.tmp")), "the temporary file was left behind"


class TestThePreWriteBoundaryRefusesEarlyAndInOneVocabulary:
    """Two properties the allocation's own checks do not give the writer.

    `allocate_diseases` validates the seed and the fraction, and it runs before
    any write — so removing the writer's copy of the seed check changes no
    outcome. What it changes is *when*: the eligibility walk over every disease
    in the graph happens first. On a real workspace that is tens of thousands of
    diseases traversed to reach a refusal that was decidable from the argument.

    And what the allocation raises is a plain `ValueError`, which the build
    script does not catch — so a fraction outside (0, 1) or a universe too small
    to split would reach an operator as a traceback rather than as the refusal
    that says nothing was written.
    """

    @staticmethod
    def _demo_kg():
        from scripts.setup_demo import build_demo_kg

        return build_demo_kg()

    def test_a_bad_seed_is_refused_before_the_graph_is_walked(
        self, monkeypatch, tmp_path
    ):
        import src.kg as kg_package
        from src.kg.workspace import SampleBudget, WorkspaceRefusal, write_workspace

        def _must_not_run(*args, **kwargs):
            raise AssertionError("eligibility was computed for a refused seed")

        monkeypatch.setattr(
            kg_package, "build_eligible_disease_profiles", _must_not_run
        )

        with pytest.raises(WorkspaceRefusal, match="seed must be an integer"):
            write_workspace(
                self._demo_kg(), tmp_path / "ws", feature_dim=8,
                samples=SampleBudget(num_train=20, num_val=5, seed="42"),
            )

    @pytest.mark.parametrize(
        "kwargs,message",
        [
            ({"val_disease_fraction": 1.5}, "val_fraction must be finite"),
            ({"val_disease_fraction": 0.0}, "val_fraction must be finite"),
            ({"val_disease_fraction": True}, "val_fraction must be a number"),
            ({"min_phenotypes": 50}, "cannot be cut into two non-empty"),
        ],
        ids=["fraction-above", "fraction-zero", "fraction-bool", "universe-too-small"],
    )
    def test_an_allocation_refusal_is_a_workspace_refusal(
        self, tmp_path, kwargs, message
    ):
        """Translated at the one call that can raise them before a write. The
        whole function is not wrapped: its later failures happen with the graph
        already saved, and would inherit a promise that nothing was written."""
        from src.kg.workspace import SampleBudget, WorkspaceRefusal, write_workspace

        workspace = tmp_path / "ws"
        with pytest.raises(WorkspaceRefusal, match=message):
            write_workspace(
                self._demo_kg(), workspace, feature_dim=8,
                samples=SampleBudget(num_train=20, num_val=5, **kwargs),
            )

        assert not workspace.exists()

    def test_the_build_reports_an_allocation_refusal_as_nothing_written(
        self, monkeypatch, tmp_path
    ):
        """End to end: the CLI catches WorkspaceRefusal, so translating at the
        writer is what turns these into operator-facing refusals."""
        from tests.unit.test_data_pipeline import (
            TestBuildPathOrdering,
            _wide_kg_object,
        )

        build = TestBuildPathOrdering._stub_build_stages(monkeypatch, _wide_kg_object())
        TestBuildPathOrdering._satisfy_input_validation(tmp_path)

        with pytest.raises(SystemExit) as excinfo:
            build.build_knowledge_graph(
                external_dir=tmp_path, workspace=tmp_path / "ws",
                generate_samples=True, num_train=60, num_val=20,
                val_disease_fraction=1.5,
            )
        assert "val_fraction must be finite" in str(excinfo.value)
        assert "Nothing was written" in str(excinfo.value)
        assert not (tmp_path / "ws").exists()
