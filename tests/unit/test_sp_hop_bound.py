"""The hop bound is provenance, not a default.

`max_hops` sets the unreachable sentinel every shortest-path score is measured
against, so a wrong one does not merely shift a number — it reorders candidates.
The loader used to read its sidecar inside `except Exception: pass`, which meant
a missing file, malformed JSON, an absent key, a string, a boolean and a value
the producer would never write all arrived at the same silent 5.

**What the fix is and is not.** The defect was the silence, not the number: every
operator-facing build path in this repository uses the producer's default of 5,
and the workspace inventories in `docs/` do not list the sidecar, so a real
5-hop table with no sidecar beside it is a shape this project's own
documentation produces. Refusing it would replace a correct score with a
different one. So a missing sidecar still assumes 5 — and says that it did.

Module: tests/unit/test_sp_hop_bound.py
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")


def _loader(tmp_path, *, distances, sidecar):
    """A pipeline instance driven through `_load_shortest_paths` and nothing else.

    Built by `__new__` plus the attributes the loader touches, because the whole
    initialisation path needs a model, a graph and a checkpoint that have nothing
    to do with the hop bound. `sidecar` is `None` for "write no file", otherwise
    the object to serialise.
    """
    from src.inference.pipeline import DiagnosisPipeline

    data_dir = tmp_path / "ws"
    data_dir.mkdir(parents=True, exist_ok=True)
    n = len(distances)
    torch.save(
        {
            "phenotype_idx": torch.arange(n, dtype=torch.int64),
            "target_idx": torch.arange(n, dtype=torch.int64),
            "target_type": torch.zeros(n, dtype=torch.int64),
            "distance": torch.tensor(distances, dtype=torch.int8),
        },
        data_dir / "shortest_paths.pt",
    )
    if sidecar is not None:
        path = data_dir / "shortest_paths.meta.json"
        path.write_text(sidecar if isinstance(sidecar, str) else json.dumps(sidecar))

    pipeline = DiagnosisPipeline.__new__(DiagnosisPipeline)
    pipeline._sp_ready = False
    pipeline._sp_lookup = None
    pipeline._sp_max_hops = 5
    pipeline._sp_hop_bound_source = None
    return pipeline, data_dir


class TestAPresentSidecarIsBinding:
    """Here and unusable is a broken artifact, not an absent one."""

    @pytest.mark.parametrize("sidecar,fragment", [
        ("{not json", "not readable JSON"),
        ("[1, 2, 3]", "not a JSON object"),
        ({"num_pairs": 3}, "max_hops=None"),
        ({"max_hops": None}, "max_hops=None"),
        ({"max_hops": "5"}, "not an integer"),
        ({"max_hops": 1.0}, "not an integer"),
        # `isinstance(True, int)` is True and `true` is valid JSON, so a bound of
        # `true` would otherwise become 1 — a sentinel of 2.0, below real
        # distances.
        ({"max_hops": True}, "not an integer"),
        ({"max_hops": 0}, r"outside the \[1, 127\]"),
        ({"max_hops": 128}, r"outside the \[1, 127\]"),
        ({"max_hops": -1}, r"outside the \[1, 127\]"),
    ], ids=["unreadable", "not-an-object", "key-absent", "null", "string",
            "float", "bool", "zero", "above-producer-range", "negative"])
    def test_it_is_refused_rather_than_defaulted(self, tmp_path, sidecar, fragment):
        pipeline, data_dir = _loader(tmp_path, distances=[1, 2, 3], sidecar=sidecar)

        with pytest.raises(ValueError, match=fragment):
            pipeline._load_shortest_paths(data_dir)

    @pytest.mark.parametrize("sidecar", [
        "{not json", {"max_hops": True}, {"max_hops": 128},
    ], ids=["unreadable", "bool", "above-producer-range"])
    def test_nothing_is_published_by_a_refused_load(self, tmp_path, sidecar):
        """`_sp_ready` was set before the sidecar was ever read, so a refusal
        landed on a pipeline already advertising a usable lookup."""
        pipeline, data_dir = _loader(tmp_path, distances=[1, 2, 3], sidecar=sidecar)

        with pytest.raises(ValueError):
            pipeline._load_shortest_paths(data_dir)

        assert pipeline._sp_ready is False
        assert pipeline._sp_lookup is None

    def test_a_sound_sidecar_is_read_and_recorded_as_read(self, tmp_path):
        """The control. Without it every refusal above holds for a loader that
        refuses every sidecar."""
        pipeline, data_dir = _loader(
            tmp_path, distances=[1, 2, 3], sidecar={"max_hops": 3, "num_pairs": 3}
        )

        pipeline._load_shortest_paths(data_dir)

        assert pipeline._sp_ready is True
        assert pipeline._sp_max_hops == 3
        assert pipeline._sp_hop_bound_source == "sidecar"
        assert pipeline._sp_lookup.unreachable_distance == 4.0


class TestAMissingSidecarAssumesAndSaysSo:
    """The defect was the silence, not the number."""

    def test_the_producer_default_is_used(self, tmp_path):
        from src.inference.scoring import ASSUMED_HOP_BOUND

        pipeline, data_dir = _loader(tmp_path, distances=[1, 2, 3], sidecar=None)

        pipeline._load_shortest_paths(data_dir)

        assert pipeline._sp_ready is True
        assert pipeline._sp_max_hops == ASSUMED_HOP_BOUND
        assert pipeline._sp_lookup.unreachable_distance == ASSUMED_HOP_BOUND + 1

    def test_the_assumption_is_recorded_rather_than_silent(self, tmp_path):
        """A log line is not a surface anyone watches. Two tables scoring
        differently must be distinguishable by something a caller can read."""
        pipeline, data_dir = _loader(tmp_path, distances=[1, 2, 3], sidecar=None)

        pipeline._load_shortest_paths(data_dir)

        assert pipeline._sp_hop_bound_source == "assumed"


class TestTheFloorCheckCatchesTheDirectionItCan:
    """`max(distance)` proves the bound is not too LOW and never that it is not
    too high — a 3-hop table is consistent with a declared 5. The one direction
    it does cover is the one a stale sidecar carried over from a smaller run
    fails in, and there the sentinel lands *below* distances really in the table,
    so an unreachable phenotype outranks a connected one."""

    def test_a_declared_bound_below_the_table_is_refused(self, tmp_path):
        pipeline, data_dir = _loader(
            tmp_path, distances=[1, 2, 5], sidecar={"max_hops": 3}
        )

        with pytest.raises(ValueError, match="records a distance of 5"):
            pipeline._load_shortest_paths(data_dir)

        assert pipeline._sp_ready is False

    def test_an_assumed_bound_below_the_table_is_refused_too(self, tmp_path):
        """The assumption is not exempt from the one check that can contradict it."""
        pipeline, data_dir = _loader(
            tmp_path, distances=[1, 7], sidecar=None
        )

        with pytest.raises(ValueError, match="records a distance of 7"):
            pipeline._load_shortest_paths(data_dir)

    def test_a_bound_exactly_one_below_the_table_is_refused(self, tmp_path):
        """The boundary, and the realistic case.

        The two tests above use a gap of two or more, which a check written as
        `observed > max_hops + 1` would still refuse — so they prove a check
        exists without pinning where it fires. A sidecar carried over from a run
        one hop smaller is the shape this actually takes, and it is the smallest
        error that still puts the sentinel on top of a real distance.
        """
        pipeline, data_dir = _loader(
            tmp_path, distances=[1, 2, 4], sidecar={"max_hops": 3}
        )

        with pytest.raises(ValueError, match="records a distance of 4"):
            pipeline._load_shortest_paths(data_dir)

    def test_a_bound_at_the_observed_maximum_is_accepted(self, tmp_path):
        """The boundary, so the check is not off by one against a real artifact
        where some pair does sit exactly at the ceiling."""
        pipeline, data_dir = _loader(
            tmp_path, distances=[1, 3], sidecar={"max_hops": 3}
        )

        pipeline._load_shortest_paths(data_dir)

        assert pipeline._sp_max_hops == 3

    def test_a_bound_above_the_observed_maximum_is_accepted(self, tmp_path):
        """And the direction the check cannot see, asserted so the limitation is
        recorded as behaviour rather than left in a comment."""
        pipeline, data_dir = _loader(
            tmp_path, distances=[1, 2], sidecar={"max_hops": 5}
        )

        pipeline._load_shortest_paths(data_dir)

        assert pipeline._sp_max_hops == 5


class TestASecondLoadCannotInheritTheFirst:
    """The loader is re-callable and overwrites its tensors before the ceiling is
    known, so a refusal on the second call would otherwise leave the first call's
    `_sp_ready` and a lookup over bytes that are no longer there."""

    def test_a_refused_reload_leaves_nothing_published(self, tmp_path):
        pipeline, first = _loader(
            tmp_path, distances=[1, 2, 3], sidecar={"max_hops": 5}
        )
        pipeline._load_shortest_paths(first)
        assert pipeline._sp_ready is True

        _, second = _loader(
            tmp_path / "again", distances=[1, 2, 3], sidecar={"max_hops": 128}
        )
        with pytest.raises(ValueError):
            pipeline._load_shortest_paths(second)

        assert pipeline._sp_ready is False, "the first load's flag survived the second"
        assert pipeline._sp_lookup is None, "a lookup over the previous table survived"
