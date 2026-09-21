"""The shortest-path pair: does it claim itself, and which graph does it name.

Two files written by one run and read as one artifact, with nothing recording
that they belong together and nothing recording which graph the distances came
from. Both gaps are silent. A new tensor beside a previous run's sidecar is
scored against the old ceiling — a present sidecar is binding — and passes every
structural check. A table computed from another graph resolves node indices to
different nodes and produces ordinary-looking numbers.

**The expensive cases here are the ones where nothing looks wrong.** A truncated
file announces itself; a mixed pair does not.

Module: tests/unit/test_sp_artifact_binding.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.inference.sp_artifact import (  # noqa: E402
    ACCEPTED_SCHEMA_VERSIONS,
    SP_SCHEMA_VERSION,
    SPBinding,
    kg_binding_state,
    new_build_id,
    publish_sp_artifact,
    read_binding,
    read_sidecar,
    require_paired,
    sidecar_path,
)
from src.inference.sp_index import SPArtifactError  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def producer():
    """The real `compute_shortest_paths.py`, loaded as the script it is."""
    spec = importlib.util.spec_from_file_location(
        "compute_shortest_paths", REPO / "scripts" / "compute_shortest_paths.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows(n=2):
    return {
        "phenotype_idx": torch.arange(n, dtype=torch.int64),
        "target_idx": torch.arange(n, dtype=torch.int64),
        "target_type": torch.ones(n, dtype=torch.int64),
        "distance": torch.ones(n, dtype=torch.int8),
    }


def meta(**extra):
    base = {"max_hops": 5, "num_pairs": 2, "num_phenotypes": 2, "num_diseases": 2}
    base.update(extra)
    return base


def publish(tmp_path, *, kg_digest, name="shortest_paths.pt", data=None, extra=None):
    """Through the producer's own entry point, never by hand."""
    path = Path(tmp_path) / name
    producer().save_shortest_paths(
        data or rows(), path, meta(**(extra or {})), kg_digest=kg_digest
    )
    return path


DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


class TestTheProducerPublishesAPairThatClaimsItself:

    def test_both_files_carry_the_same_build_id(self, tmp_path):
        path = publish(tmp_path, kg_digest=DIGEST_A)

        sidecar = json.loads(sidecar_path(path).read_text())
        tensor = torch.load(path, map_location="cpu", weights_only=True)

        assert sidecar["schema_version"] == SP_SCHEMA_VERSION
        assert sidecar["build_id"] == tensor["build_id"]
        assert sidecar["kg_digest"] == DIGEST_A
        assert len(sidecar["build_id"]) == 32

    def test_two_publications_do_not_share_a_build_id(self, tmp_path):
        """Otherwise a stale sidecar would pair with a fresh tensor, which is
        the state the whole protocol exists to catch."""
        first = json.loads(sidecar_path(publish(tmp_path / "a", kg_digest=DIGEST_A)).read_text())
        second = json.loads(sidecar_path(publish(tmp_path / "b", kg_digest=DIGEST_A)).read_text())

        assert first["build_id"] != second["build_id"]

    def test_the_sidecar_keeps_the_fields_other_readers_require(self, tmp_path):
        """`audit_sp_reachability.py` refuses a sidecar without these. Adding
        the binding must not drop them."""
        sidecar = json.loads(sidecar_path(publish(tmp_path, kg_digest=DIGEST_A)).read_text())

        for key in ("max_hops", "num_pairs", "num_phenotypes", "num_diseases"):
            assert key in sidecar

    def test_without_a_source_digest_it_publishes_an_unrecorded_pair(self, tmp_path):
        """A caller computing from a graph it built in memory has no file to
        hash. Publishing *unrecorded* is honest; inventing a digest would be the
        claim this binding exists to check."""
        path = publish(tmp_path, kg_digest=None)

        sidecar = json.loads(sidecar_path(path).read_text())
        tensor = torch.load(path, map_location="cpu", weights_only=True)

        assert "build_id" not in sidecar and "kg_digest" not in sidecar
        assert "build_id" not in tensor
        assert read_binding(sidecar, tensor.keys()).recorded is False


class TestRuleZero:
    """**Legacy is a conclusion about both files.** A new tensor beside an old
    sidecar matches "no fields in the sidecar" *and* "build_id on one side only";
    without a precedence rule the very state this catches could resolve as
    legacy and be served."""

    def test_neither_side_declaring_is_the_only_legacy(self, tmp_path):
        path = publish(tmp_path, kg_digest=None)
        tensor = torch.load(path, map_location="cpu", weights_only=True)

        assert read_binding(read_sidecar(path), tensor.keys()) == SPBinding(recorded=False)

    def test_a_sidecar_declaring_alone_is_refused_not_legacy(self, tmp_path):
        path = publish(tmp_path, kg_digest=None)
        sidecar = json.loads(sidecar_path(path).read_text())
        sidecar.update(
            {"schema_version": SP_SCHEMA_VERSION, "build_id": new_build_id(),
             "kg_digest": DIGEST_A}
        )
        sidecar_path(path).write_text(json.dumps(sidecar))

        with pytest.raises(SPArtifactError, match="One side of a pair is not a pair"):
            read_binding(read_sidecar(path), ["phenotype_idx"])

    def test_a_tensor_declaring_alone_is_refused_not_legacy(self, tmp_path):
        """The mirror, and the one a check written against the sidecar alone
        would miss."""
        with pytest.raises(SPArtifactError, match="schema_version"):
            read_binding({"max_hops": 5}, ["phenotype_idx", "build_id"])

    @pytest.mark.parametrize("version", [0, "", False, None, 2, "1"])
    def test_an_unaccepted_schema_version_is_refused(self, version):
        """Membership, not truthiness: `0`, `""` and `False` are all falsy and
        none of them means absent."""
        assert version not in ACCEPTED_SCHEMA_VERSIONS
        with pytest.raises(SPArtifactError, match="schema_version"):
            read_binding(
                {"schema_version": version, "build_id": "a" * 32, "kg_digest": DIGEST_A},
                ["build_id"],
            )

    @pytest.mark.parametrize(
        "field,value,fragment",
        [
            ("build_id", "zz" * 16, "hexadecimal"),
            ("build_id", "a" * 31, "hexadecimal"),
            ("build_id", None, "hexadecimal"),
            ("build_id", 12345, "hexadecimal"),
            ("kg_digest", "a" * 63, "SHA-256"),
            ("kg_digest", None, "SHA-256"),
        ],
    )
    def test_a_malformed_declared_field_is_refused(self, field, value, fragment):
        sidecar = {
            "schema_version": SP_SCHEMA_VERSION,
            "build_id": "a" * 32,
            "kg_digest": DIGEST_A,
        }
        sidecar[field] = value
        with pytest.raises(SPArtifactError, match=fragment):
            read_binding(sidecar, ["build_id"])


class TestAMixedPairIsRefused:

    def test_a_new_tensor_beside_a_previous_runs_sidecar(self, tmp_path):
        """The failure the protocol exists for, built the way it happens: two
        real publications, the second's tensor kept and the first's sidecar."""
        first = publish(tmp_path / "first", kg_digest=DIGEST_A)
        second = publish(tmp_path / "second", kg_digest=DIGEST_A)

        mixed = tmp_path / "mixed"
        mixed.mkdir()
        (mixed / "shortest_paths.pt").write_bytes(second.read_bytes())
        sidecar_path(mixed / "shortest_paths.pt").write_bytes(
            sidecar_path(first).read_bytes()
        )

        path = mixed / "shortest_paths.pt"
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        binding = read_binding(read_sidecar(path), tensor.keys())

        with pytest.raises(SPArtifactError, match="published by different"):
            require_paired(binding, tensor["build_id"])

    def test_every_structural_check_passes_on_that_pair(self, tmp_path):
        """Why the pairing had to be added: nothing else can see it. Both files
        are well formed and the numbers are ordinary."""
        from src.inference.sp_index import validate_sp_artifact

        first = publish(tmp_path / "first", kg_digest=DIGEST_A)
        second = publish(tmp_path / "second", kg_digest=DIGEST_A)
        assert json.loads(sidecar_path(first).read_text())["max_hops"] == 5

        tensor = torch.load(second, map_location="cpu", weights_only=True)
        assert validate_sp_artifact(
            tensor["phenotype_idx"], tensor["target_idx"],
            tensor["target_type"], tensor["distance"], 5,
        ) == 2


class TestTheGraphItNames:

    @staticmethod
    def _binding(kg_digest=DIGEST_A):
        return SPBinding(
            recorded=True, schema_version=SP_SCHEMA_VERSION,
            build_id="a" * 32, kg_digest=kg_digest,
        )

    def test_equal_digests_verify(self):
        assert kg_binding_state(self._binding(), DIGEST_A) == "verified"

    def test_a_different_graph_is_refused(self):
        with pytest.raises(SPArtifactError, match="different graph"):
            kg_binding_state(self._binding(), DIGEST_B)

    def test_no_comparable_graph_is_returned_not_raised(self):
        """Not a verdict on the artifact: a consumer that ranks refuses, and a
        tool reading only the integer ids may proceed and must label it."""
        assert kg_binding_state(self._binding(), None) == "unverifiable"

    def test_a_legacy_pair_is_unrecorded(self):
        assert kg_binding_state(SPBinding(recorded=False), DIGEST_A) == "unrecorded"

    def test_node_and_edge_counts_cannot_tell_two_graphs_apart(self, tmp_path):
        """The sidecar has recorded `kg_total_nodes` and `kg_total_edges` all
        along, and they are counts rather than identity. This is the pair that
        proves a check built on them would pass."""
        counts = {"kg_total_nodes": 24, "kg_total_edges": 32}
        a = publish(tmp_path / "a", kg_digest=DIGEST_A, extra=counts)
        b = publish(tmp_path / "b", kg_digest=DIGEST_B, extra=counts)

        meta_a = json.loads(sidecar_path(a).read_text())
        meta_b = json.loads(sidecar_path(b).read_text())

        assert meta_a["kg_total_nodes"] == meta_b["kg_total_nodes"]
        assert meta_a["kg_total_edges"] == meta_b["kg_total_edges"]
        assert meta_a["kg_digest"] != meta_b["kg_digest"]
        with pytest.raises(SPArtifactError, match="different graph"):
            kg_binding_state(read_binding(meta_a, ["build_id"]), meta_b["kg_digest"])


class TestInterruptedPublication:
    """**A mixed pair is refused; an intact one is not condemned.**

    Under temp-and-replace a failure while writing a temporary file has not
    touched the live pair, which is exactly as valid as it was a moment before.
    An earlier draft of the plan said every interruption must refuse or switch
    SP off, which would have turned a producer crash into an outage for
    artifacts nothing happened to.
    """

    @staticmethod
    def _fail_on_nth_replace(monkeypatch, n):
        calls = {"n": 0}
        real = os.replace

        def guard(src, dst, *a, **k):
            calls["n"] += 1
            if calls["n"] == n:
                raise OSError(28, "No space left on device")
            return real(src, dst, *a, **k)

        monkeypatch.setattr(os, "replace", guard)
        return calls

    def test_a_failed_temp_write_leaves_the_live_pair_byte_identical(
        self, tmp_path, monkeypatch
    ):
        path = publish(tmp_path, kg_digest=DIGEST_A)
        before = (path.read_bytes(), sidecar_path(path).read_bytes())

        def explode(*a, **k):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(torch, "save", explode)
        with pytest.raises(OSError):
            publish(tmp_path, kg_digest=DIGEST_A)

        assert (path.read_bytes(), sidecar_path(path).read_bytes()) == before

    def test_no_temporary_file_is_left_behind(self, tmp_path, monkeypatch):
        publish(tmp_path, kg_digest=DIGEST_A)
        self._fail_on_nth_replace(monkeypatch, 2)

        with pytest.raises(OSError):
            publish(tmp_path, kg_digest=DIGEST_A)

        assert not [p for p in Path(tmp_path).iterdir() if p.name.endswith(".tmp")]

    def test_one_replace_completed_is_the_mixed_pair_and_is_refused(
        self, tmp_path, monkeypatch
    ):
        publish(tmp_path, kg_digest=DIGEST_A)
        self._fail_on_nth_replace(monkeypatch, 2)
        with pytest.raises(OSError):
            publish(tmp_path, kg_digest=DIGEST_A)

        path = Path(tmp_path) / "shortest_paths.pt"
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        binding = read_binding(read_sidecar(path), tensor.keys())

        with pytest.raises(SPArtifactError, match="published by different"):
            require_paired(binding, tensor["build_id"])

    def test_both_replaces_completed_is_accepted(self, tmp_path):
        publish(tmp_path, kg_digest=DIGEST_A)
        path = publish(tmp_path, kg_digest=DIGEST_A)

        tensor = torch.load(path, map_location="cpu", weights_only=True)
        binding = read_binding(read_sidecar(path), tensor.keys())

        require_paired(binding, tensor["build_id"])
        assert kg_binding_state(binding, DIGEST_A) == "verified"

    def test_a_first_build_that_fails_leaves_nothing_usable(self, tmp_path, monkeypatch):
        self._fail_on_nth_replace(monkeypatch, 1)
        with pytest.raises(OSError):
            publish(tmp_path, kg_digest=DIGEST_A)

        assert not (Path(tmp_path) / "shortest_paths.pt").exists()


def loader_pipeline(tmp_path, *, graph_digest=None):
    """A pipeline driven through `_load_shortest_paths` and nothing else.

    Built by `__new__` plus the attributes the loader touches, because the whole
    initialisation path needs a model, a graph and a checkpoint that have
    nothing to do with an artifact binding.
    """
    from src.inference.pipeline import DiagnosisPipeline, PipelineConfig

    pipeline = DiagnosisPipeline.__new__(DiagnosisPipeline)
    pipeline.config = PipelineConfig()
    pipeline._sp_ready = False
    pipeline._sp_lookup = None
    pipeline._sp_max_hops = 5
    pipeline._sp_hop_bound_source = None
    pipeline._sp_kg_binding = None
    pipeline._graph_kg_digest = graph_digest
    return pipeline


class TestTheLoaderEnforcesTheBinding:

    def test_a_legacy_pair_serves_and_is_reported_unrecorded(self, tmp_path):
        """Refusing these would break every deployment for a claim they never
        made. Serving them while calling the provenance *verified* would be
        worse — so it is neither."""
        publish(tmp_path, kg_digest=None)
        pipeline = loader_pipeline(tmp_path, graph_digest=DIGEST_A)

        pipeline._load_shortest_paths(Path(tmp_path))

        assert pipeline._sp_ready is True
        assert pipeline._sp_kg_binding == "unrecorded"

    def test_a_bound_pair_over_the_graph_it_names_serves(self, tmp_path):
        publish(tmp_path, kg_digest=DIGEST_A)
        pipeline = loader_pipeline(tmp_path, graph_digest=DIGEST_A)

        pipeline._load_shortest_paths(Path(tmp_path))

        assert pipeline._sp_ready is True
        assert pipeline._sp_kg_binding == "verified"

    def test_a_bound_pair_over_another_graph_is_refused(self, tmp_path):
        publish(tmp_path, kg_digest=DIGEST_A)
        pipeline = loader_pipeline(tmp_path, graph_digest=DIGEST_B)

        with pytest.raises(ValueError, match="different graph"):
            pipeline._load_shortest_paths(Path(tmp_path))

        assert pipeline._sp_ready is False
        assert pipeline._sp_lookup is None

    def test_a_binding_that_cannot_be_checked_is_refused_not_waved_through(
        self, tmp_path
    ):
        """**The row the first draft got backwards.** A caller supplying its
        graph in memory skips file-backed verification by design and still loads
        SP from `data_dir`; treating "cannot check" as unknown would let the
        binding be bypassed by deleting `kg.json`."""
        publish(tmp_path, kg_digest=DIGEST_A)
        pipeline = loader_pipeline(tmp_path, graph_digest=None)

        with pytest.raises(ValueError, match="supplied in memory"):
            pipeline._load_shortest_paths(Path(tmp_path))

        assert pipeline._sp_ready is False

    def test_a_mixed_pair_is_refused_and_nothing_is_published(self, tmp_path):
        publish(tmp_path, kg_digest=DIGEST_A)
        stale = json.loads(sidecar_path(Path(tmp_path) / "shortest_paths.pt").read_text())
        publish(tmp_path, kg_digest=DIGEST_A)
        sidecar_path(Path(tmp_path) / "shortest_paths.pt").write_text(json.dumps(stale))

        pipeline = loader_pipeline(tmp_path, graph_digest=DIGEST_A)
        with pytest.raises(ValueError, match="published by different"):
            pipeline._load_shortest_paths(Path(tmp_path))

        assert pipeline._sp_ready is False
        assert pipeline._sp_hop_bound_source is None
        assert pipeline._sp_kg_binding is None

    def test_the_refusal_does_not_reach_the_graph_verifier(self, tmp_path):
        """**Scope.** A rejected shortest-path table must not stop a consumer
        that never reads it. The check belongs to the SP loader, not to
        `verify_graph_artifacts`, which is what a graph or training consumer
        goes through — that is the provenance round's lesson, where a check put
        there blocked model loading over a note that gates nothing.
        """
        import inspect

        from src.kg import artifacts

        source = inspect.getsource(artifacts.verify_graph_artifacts)
        for name in ("build_id", "kg_binding", "sp_artifact", "shortest_paths"):
            assert name not in source, (
                f"verify_graph_artifacts mentions {name}; the SP binding has "
                "moved into the gate every graph consumer passes through"
            )


class TestTheDigestRecordedIsTheOneReadAtLoad:
    """**Which of the two readings is kept, and it is not the later one.**

    The source file is hashed before the traversal and again after it, and the
    comparison warns when they differ. Recording the *second* would say
    "whatever is at this path now" and call it the input — the failure the
    binding exists to prevent, reached from the other direction. A mutant that
    records the end-of-job digest survived every other test here, which is why
    this one exists.
    """

    def test_a_source_that_changes_mid_run_is_recorded_as_it_was(
        self, tmp_path, monkeypatch, caplog
    ):
        from scripts.setup_demo import build_demo_kg
        from src.utils.fingerprint import file_sha256

        module = producer()
        kg_path = tmp_path / "kg.json"
        build_demo_kg().save_json(str(kg_path))
        at_load = file_sha256(kg_path)

        def compute_and_meddle(kg, max_hops=5, workers=None):
            """Stands in for the traversal, which at deployment scale runs for
            hours — long enough for the file underneath it to be replaced."""
            kg_path.write_text(kg_path.read_text() + "\n")
            return rows()

        monkeypatch.setattr(module, "compute_shortest_paths", compute_and_meddle)
        monkeypatch.setattr(
            sys, "argv",
            ["compute_shortest_paths.py", "--kg-path", str(kg_path),
             "--output-dir", str(tmp_path), "--max-hops", "5"],
        )

        with caplog.at_level("WARNING"):
            assert module.main() == 0

        after = file_sha256(kg_path)
        recorded = json.loads((tmp_path / "shortest_paths.meta.json").read_text())

        assert after != at_load, "the fixture did not actually change the file"
        assert recorded["kg_digest"] == at_load, (
            "the artifact records the file as it is now rather than the graph "
            "these distances were computed from"
        )
        assert recorded["kg_digest"] != after
        assert any("changed while the shortest paths" in r.message for r in caplog.records), (
            "the operator was not told the snapshot assumption was broken"
        )

    def test_an_unchanged_source_records_itself_and_says_nothing(
        self, tmp_path, monkeypatch, caplog
    ):
        """Without this, the assertion above holds for a producer that records
        the wrong digest whenever the file is stable too."""
        from scripts.setup_demo import build_demo_kg
        from src.utils.fingerprint import file_sha256

        module = producer()
        kg_path = tmp_path / "kg.json"
        build_demo_kg().save_json(str(kg_path))

        monkeypatch.setattr(
            module, "compute_shortest_paths", lambda kg, max_hops=5, workers=None: rows()
        )
        monkeypatch.setattr(
            sys, "argv",
            ["compute_shortest_paths.py", "--kg-path", str(kg_path),
             "--output-dir", str(tmp_path), "--max-hops", "5"],
        )

        with caplog.at_level("WARNING"):
            assert module.main() == 0

        recorded = json.loads((tmp_path / "shortest_paths.meta.json").read_text())
        assert recorded["kg_digest"] == file_sha256(kg_path)
        assert not any("changed while" in r.message for r in caplog.records)


class TestItReachesTheServiceResponse:
    """**A field added to the pipeline's dict is not a field in the response.**

    `PipelineStatusResponse` lists its fields explicitly and the route maps them
    one by one, so a new key in `get_pipeline_config()` goes no further unless
    both are edited. The deployment guides tell an operator to read
    `sp_kg_binding` from `GET /api/v1/pipeline/config`; this is what makes that
    sentence true rather than aspirational.
    """

    def test_the_response_model_carries_it(self):
        from src.api.routes.pipeline import PipelineStatusResponse

        assert "sp_kg_binding" in PipelineStatusResponse.model_fields

    def test_the_route_maps_it_from_the_pipeline(self, tmp_path):
        import inspect

        from src.api.routes import pipeline as route

        source = inspect.getsource(route)
        assert 'sp_kg_binding=config.get("sp_kg_binding")' in source, (
            "the field exists on the model and nothing fills it"
        )

    def test_the_pipeline_reports_it_beside_the_hop_bound(self, tmp_path):
        publish(tmp_path, kg_digest=DIGEST_A)
        pipeline = loader_pipeline(tmp_path, graph_digest=DIGEST_A)
        pipeline._load_shortest_paths(Path(tmp_path))

        assert pipeline._sp_kg_binding == "verified"
