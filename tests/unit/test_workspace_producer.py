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

# The two calls that, taken together, make a directory a workspace: the graph
# export and the cohort generation whose manifest binds it.
PRIMITIVES = ("export_graph_data", "generate_training_samples")
SOLE_CALLER = PROJECT_ROOT / "src" / "kg" / "workspace.py"


def _call_sites(path: Path, name: str):
    """Line numbers where `name` is *called* in `path` -- definitions excluded."""
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
        if called == name:
            lines.append(node.lineno)
    return lines


@pytest.mark.parametrize("primitive", PRIMITIVES)
def test_the_workspace_primitives_have_exactly_one_caller(primitive):
    """Shipped code, not tests: a test may compose them to build a fixture, but
    nothing an operator can run may assemble a workspace of its own."""
    offenders = {}
    for directory in ("src", "scripts"):
        for path in sorted((PROJECT_ROOT / directory).rglob("*.py")):
            if path == SOLE_CALLER:
                continue
            sites = _call_sites(path, primitive)
            if sites:
                offenders[str(path.relative_to(PROJECT_ROOT))] = sites

    assert offenders == {}, (
        f"{primitive} is called outside src/kg/workspace.py: {offenders}. "
        "Route it through write_workspace instead of composing the ordering "
        "again -- that ordering is what binds the six artifacts into one event."
    )


def test_both_entry_points_call_the_same_writer():
    """Same object, not merely the same name. Two functions that agree today
    diverge the first time one is extended."""
    import scripts.build_knowledge_graph as build
    import scripts.setup_demo as demo
    import src.kg.workspace as workspace

    assert build.write_workspace is workspace.write_workspace
    assert demo.write_workspace is workspace.write_workspace


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
