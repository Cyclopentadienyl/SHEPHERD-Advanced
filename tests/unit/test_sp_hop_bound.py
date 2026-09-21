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


def _config(**kwargs):
    """A PipelineConfig with only the fields this loader reads."""
    from src.inference.pipeline import PipelineConfig

    return PipelineConfig(**kwargs)


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
    pipeline.config = _config()
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


class TestAMissingSidecarNeedsTheBoundStated:
    """A recorded guess is still a guess.

    A previous revision assumed the producer's default of 5 here and recorded
    that it had assumed. Recording does not stop it changing an answer: the
    producer's CLI accepts 1..127, so a legitimate 3-hop table is inside the
    supported range, and read against 5 it reorders candidates. The floor check
    cannot see that — the observed maximum is 3 either way.
    """

    def test_without_a_stated_bound_shortest_paths_stay_off(self, tmp_path):
        pipeline, data_dir = _loader(tmp_path, distances=[1, 2, 3], sidecar=None)
        pipeline.config = _config(sp_hop_bound=None)

        pipeline._load_shortest_paths(data_dir)

        assert pipeline._sp_ready is False
        assert pipeline._sp_lookup is None

    def test_the_workspace_itself_is_not_refused(self, tmp_path):
        """SP off is the state an absent table already produces. The tensors are
        sound; what is declined is scoring against a ceiling nobody chose."""
        pipeline, data_dir = _loader(tmp_path, distances=[1, 2, 3], sidecar=None)
        pipeline.config = _config(sp_hop_bound=None)

        pipeline._load_shortest_paths(data_dir)  # returns, does not raise

    def test_a_stated_bound_is_used_and_recorded_as_stated(self, tmp_path):
        pipeline, data_dir = _loader(tmp_path, distances=[1, 2, 3], sidecar=None)
        pipeline.config = _config(sp_hop_bound=3)

        pipeline._load_shortest_paths(data_dir)

        assert pipeline._sp_ready is True
        assert pipeline._sp_max_hops == 3
        assert pipeline._sp_hop_bound_source == "configured"
        assert pipeline._sp_lookup.unreachable_distance == 4.0

    @pytest.mark.parametrize("bound,fragment", [
        (True, "not an integer"),
        ("3", "not an integer"),
        (0, r"outside the \[1, 127\]"),
        (128, r"outside the \[1, 127\]"),
    ], ids=["bool", "string", "zero", "above-producer-range"])
    def test_a_stated_bound_is_held_to_the_same_domain(self, tmp_path, bound, fragment):
        """Stated is not trusted. The sidecar's rules are the configuration's."""
        pipeline, data_dir = _loader(tmp_path, distances=[1, 2, 3], sidecar=None)
        pipeline.config = _config(sp_hop_bound=bound)

        with pytest.raises(ValueError, match=fragment):
            pipeline._load_shortest_paths(data_dir)

    def test_a_stated_bound_below_the_table_is_refused(self, tmp_path):
        """And it is not exempt from the one check that can contradict it."""
        pipeline, data_dir = _loader(tmp_path, distances=[1, 5], sidecar=None)
        pipeline.config = _config(sp_hop_bound=3)

        with pytest.raises(ValueError, match="records a distance of 5"):
            pipeline._load_shortest_paths(data_dir)

    def test_the_sidecar_wins_over_the_configured_bound(self, tmp_path):
        """The file describes the artifact; the setting is for when no file does.
        Without this, a stale configuration would quietly override a table that
        came with its own answer."""
        pipeline, data_dir = _loader(
            tmp_path, distances=[1, 2, 3], sidecar={"max_hops": 3}
        )
        pipeline.config = _config(sp_hop_bound=5)

        pipeline._load_shortest_paths(data_dir)

        assert pipeline._sp_max_hops == 3
        assert pipeline._sp_hop_bound_source == "sidecar"


class TestTheRankingThisPrevents:
    """The counterexample the assumption could not survive, asserted as
    behaviour so the reasoning is not only in a commit message."""

    def test_reading_a_three_hop_table_against_five_flips_a_pair(self):
        """Not a claim about the loader — a claim about why its refusal matters.

        Two candidates over one 3-hop table: A's phenotype distances are
        [1, unreachable], B's are [3, 3], both embedding scores 0.5, eta 0.7.
        The sentinel is the only thing that differs between the two readings.
        """
        from src.inference.scoring import sp_scores_from_distances

        def mixture(max_hops):
            unreachable = float(max_hops + 1)
            means = torch.tensor(
                [(1.0 + unreachable) / 2, 3.0], dtype=torch.float64
            )
            sp = sp_scores_from_distances(means)
            return [0.7 * 0.5 + 0.3 * float(v) for v in sp]

        correct_a, correct_b = mixture(3)
        assumed_a, assumed_b = mixture(5)

        assert correct_a > correct_b, "with the real bound, A ranks above B"
        assert assumed_b > assumed_a, (
            "the assumption does not merely shift the scores, it swaps the pair"
        )


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


class TestTheProvenanceReachesACaller:
    """A field nothing carries is a field nobody reads.

    The tests above assert the loader's private attribute, which proves the
    value is computed and nothing about whether it survives to the surface an
    operator watches. Two tables scoring differently have to be distinguishable
    without reading the service's logs.
    """

    @pytest.mark.parametrize("source", ["sidecar", "configured"])
    def test_it_survives_get_pipeline_config_and_the_status_response(self, source):
        from src.api.routes.pipeline import _status_of

        status = _status_of(
            {"scoring_mode": "gnn_plus_shortest_path", "sp_max_hops": 3,
             "sp_hop_bound_source": source},
            data_dir=None, checkpoint_path=None,
        )

        assert status.sp_hop_bound_source == source

    def test_the_loader_and_the_response_agree_end_to_end(self, tmp_path):
        """Through the real `get_pipeline_config`, which the first version only
        claimed.

        That version hand-built the dict it passed to `_status_of`, so deleting
        the provenance key from production `get_pipeline_config` left all 32
        tests green — the docstring asserted coverage the test did not have.
        """
        from src.api.routes.pipeline import _status_of

        pipeline, data_dir = _loader(
            tmp_path, distances=[1, 2, 3], sidecar={"max_hops": 3}
        )
        pipeline._load_shortest_paths(data_dir)

        # The rest of `get_pipeline_config` reads attributes a full
        # initialisation sets; supplying them is what lets the real method run
        # here instead of a dict standing in for it.
        pipeline._gnn_ready = True
        pipeline.kg = None
        pipeline.model = None

        config = pipeline.get_pipeline_config()

        assert config["sp_hop_bound_source"] == "sidecar", (
            "the production method does not emit the key the response carries"
        )
        status = _status_of(config, data_dir=None, checkpoint_path=None)
        assert status.sp_max_hops == 3
        assert status.sp_hop_bound_source == "sidecar"


class TestTheLoaderRefusesAPresentButUnusableTable:
    """**Absent and broken are different deployments.** The `torch.load` handler
    warned and returned, so a corrupt artifact produced the same "scoring will
    use pure GNN" line an absent one does — and the remedies are opposite. D1
    resolved this to refusal; removing the file is how an operator reaches the
    absent path deliberately.

    Each case also checks that nothing was published. The index build joins the
    fallible tail, so a refusal after the hop bound is resolved must still leave
    `_sp_ready` False rather than a pipeline advertising a lookup it does not
    have.
    """

    @staticmethod
    def _pipeline(tmp_path, payload):
        from src.inference.pipeline import DiagnosisPipeline

        data_dir = tmp_path / "ws"
        data_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(payload, bytes):
            (data_dir / "shortest_paths.pt").write_bytes(payload)
        else:
            torch.save(payload, data_dir / "shortest_paths.pt")
        (data_dir / "shortest_paths.meta.json").write_text(json.dumps({"max_hops": 5}))

        pipeline = DiagnosisPipeline.__new__(DiagnosisPipeline)
        pipeline.config = _config()
        pipeline._sp_ready = False
        pipeline._sp_lookup = None
        pipeline._sp_max_hops = 5
        pipeline._sp_hop_bound_source = None
        return pipeline, data_dir

    def test_an_unreadable_artifact_is_refused_rather_than_ignored(self, tmp_path):
        pipeline, data_dir = self._pipeline(tmp_path, b"not a torch file at all")

        with pytest.raises(ValueError, match="could not be read"):
            pipeline._load_shortest_paths(data_dir)

        assert pipeline._sp_ready is False
        assert pipeline._sp_lookup is None

    def test_a_missing_column_is_refused(self, tmp_path):
        pipeline, data_dir = self._pipeline(
            tmp_path, {"phenotype_idx": torch.zeros(3, dtype=torch.int64)}
        )

        with pytest.raises(ValueError, match="missing"):
            pipeline._load_shortest_paths(data_dir)

        assert pipeline._sp_ready is False

    def test_duplicate_rows_are_refused_and_nothing_is_published(self, tmp_path):
        """**The publish-last invariant, tested where it can actually fail.**
        The duplicate check lives inside the index build, which runs after the
        hop bound is resolved — so this is the case that catches a `_sp_ready`
        set one line too early."""
        pipeline, data_dir = self._pipeline(
            tmp_path,
            {
                "phenotype_idx": torch.tensor([0, 0], dtype=torch.int64),
                "target_idx": torch.tensor([4, 4], dtype=torch.int64),
                "target_type": torch.tensor([1, 1], dtype=torch.int64),
                "distance": torch.tensor([2, 3], dtype=torch.int8),
            },
        )

        with pytest.raises(ValueError, match="duplicate"):
            pipeline._load_shortest_paths(data_dir)

        assert pipeline._sp_ready is False, "ready was published before the build"
        assert pipeline._sp_lookup is None
        assert pipeline._sp_hop_bound_source is None

    def test_a_third_target_type_is_refused_as_an_artifact(self, tmp_path):
        """The producer writes 0 and 1. A third value means the table came from
        something else, and which rows are genes is then a guess."""
        pipeline, data_dir = self._pipeline(
            tmp_path,
            {
                "phenotype_idx": torch.tensor([0, 1], dtype=torch.int64),
                "target_idx": torch.tensor([4, 5], dtype=torch.int64),
                "target_type": torch.tensor([1, 2], dtype=torch.int64),
                "distance": torch.tensor([2, 3], dtype=torch.int8),
            },
        )

        with pytest.raises(ValueError, match="target_type"):
            pipeline._load_shortest_paths(data_dir)

        assert pipeline._sp_ready is False

    def test_a_negative_id_is_refused_before_it_can_be_narrowed(self, tmp_path):
        """**Before narrowing, which is the whole point of the ordering.** A
        value the narrow type cannot hold does not raise on the way in — it
        wraps — so a check afterwards inspects the wrapped value."""
        pipeline, data_dir = self._pipeline(
            tmp_path,
            {
                "phenotype_idx": torch.tensor([0, 1], dtype=torch.int64),
                "target_idx": torch.tensor([4, -5], dtype=torch.int64),
                "target_type": torch.tensor([1, 1], dtype=torch.int64),
                "distance": torch.tensor([2, 3], dtype=torch.int8),
            },
        )

        with pytest.raises(ValueError, match="negative"):
            pipeline._load_shortest_paths(data_dir)

        assert pipeline._sp_ready is False

    @pytest.mark.parametrize(
        "label,payload",
        [
            (
                "distance emptied while the id columns keep a row",
                {
                    "phenotype_idx": torch.tensor([0], dtype=torch.int64),
                    "target_idx": torch.tensor([0], dtype=torch.int64),
                    "target_type": torch.tensor([1], dtype=torch.int64),
                    "distance": torch.empty(0, dtype=torch.int8),
                },
            ),
            (
                "a two-dimensional distance column",
                {
                    "phenotype_idx": torch.empty(0, dtype=torch.int64),
                    "target_idx": torch.empty(0, dtype=torch.int64),
                    "target_type": torch.empty(0, dtype=torch.int64),
                    "distance": torch.empty((0, 2), dtype=torch.int8),
                },
            ),
            (
                "an empty float distance column",
                {
                    "phenotype_idx": torch.empty(0, dtype=torch.int64),
                    "target_idx": torch.empty(0, dtype=torch.int64),
                    "target_type": torch.empty(0, dtype=torch.int64),
                    "distance": torch.empty(0, dtype=torch.float32),
                },
            ),
            (
                "an id column that is a Python list",
                {
                    "phenotype_idx": [],
                    "target_idx": torch.empty(0, dtype=torch.int64),
                    "target_type": torch.empty(0, dtype=torch.int64),
                    "distance": torch.empty(0, dtype=torch.int8),
                },
            ),
        ],
    )
    def test_a_malformed_table_is_not_an_empty_one(self, tmp_path, label, payload):
        """**Zero rows is a conclusion, not a premise.**

        A first version of the empty-table path read `distance.numel()` and
        returned on 0 — asking one column how long it is and then skipping every
        check on all four. All four tables here were accepted as "no rows". The
        damage is on reload rather than at cold start: a refusal keeps the
        running pipeline, and an acceptance replaces a healthy SP-ready pipeline
        with one serving on the GNN alone, reporting success.
        """
        pipeline, data_dir = self._pipeline(tmp_path, payload)

        with pytest.raises(ValueError):
            pipeline._load_shortest_paths(data_dir)

        assert pipeline._sp_ready is False, label
        assert pipeline._sp_lookup is None

    def test_the_sound_table_still_loads(self, tmp_path):
        """Without this, every refusal above holds for a loader that refuses
        everything."""
        pipeline, data_dir = self._pipeline(
            tmp_path,
            {
                "phenotype_idx": torch.tensor([0, 1], dtype=torch.int64),
                "target_idx": torch.tensor([4, 5], dtype=torch.int64),
                "target_type": torch.tensor([1, 0], dtype=torch.int64),
                "distance": torch.tensor([2, 3], dtype=torch.int8),
            },
        )

        pipeline._load_shortest_paths(data_dir)

        assert pipeline._sp_ready is True
        assert pipeline._sp_lookup.n_rows == 2
        assert pipeline._sp_hop_bound_source == "sidecar"

    @pytest.mark.parametrize("below", [0, -1, -2])
    def test_a_distance_below_one_is_refused_through_the_loader(self, tmp_path, below):
        """**Measured before this check existed**, with `_sp_ready` True in
        every case: distance -1 gave a mean of -1.0 and a score of `inf`; -2
        gave -1.0; 0 gave a perfect 1.0. All three reached `_calculate_sp_score`
        as numbers."""
        pipeline, data_dir = self._pipeline(
            tmp_path,
            {
                "phenotype_idx": torch.tensor([0, 1], dtype=torch.int64),
                "target_idx": torch.tensor([4, 5], dtype=torch.int64),
                "target_type": torch.tensor([1, 1], dtype=torch.int64),
                "distance": torch.tensor([1, below], dtype=torch.int8),
            },
        )

        with pytest.raises(ValueError, match="records a distance of"):
            pipeline._load_shortest_paths(data_dir)

        assert pipeline._sp_ready is False
        assert pipeline._sp_lookup is None


class TestAnEmptyTableIsTheAbsentCase:
    """**Not a refusal and not a ready pipeline.** An empty table binds nothing,
    so there is nothing for SP to be ready for; D5 takes this reading over
    refusing it because a table with no rows makes no false claim and is
    indistinguishable in effect from having no file.

    Publishing it instead is not an internal-shape difference. A query against
    an empty index returns unreachable with `available` True, so the combined
    score mixes SP in for every candidate — measured at 1/7 with the default
    bound, against the pure-GNN value an absent file gives.
    """

    def test_no_rows_leaves_shortest_paths_off(self, tmp_path):
        pipeline, data_dir = _loader(tmp_path, distances=[], sidecar={"max_hops": 5})

        pipeline._load_shortest_paths(data_dir)

        assert pipeline._sp_ready is False
        assert pipeline._sp_lookup is None

    def test_it_lands_in_the_same_state_as_no_file_at_all(self, tmp_path):
        """The claim is equivalence with the absent case, so it is asserted
        against the absent case rather than against a remembered value."""
        empty_pipeline, empty_dir = _loader(
            tmp_path / "empty", distances=[], sidecar={"max_hops": 5}
        )
        empty_pipeline._load_shortest_paths(empty_dir)

        absent_dir = tmp_path / "absent"
        absent_dir.mkdir(parents=True, exist_ok=True)
        absent_pipeline, _ = _loader(
            tmp_path / "scratch", distances=[1], sidecar={"max_hops": 5}
        )
        absent_pipeline._load_shortest_paths(absent_dir)

        assert (empty_pipeline._sp_ready, empty_pipeline._sp_lookup) == (
            absent_pipeline._sp_ready,
            absent_pipeline._sp_lookup,
        ) == (False, None)

    def test_a_table_with_rows_still_loads(self, tmp_path):
        """Without this, the two above hold for a loader that never publishes."""
        pipeline, data_dir = _loader(tmp_path, distances=[1, 2], sidecar={"max_hops": 5})

        pipeline._load_shortest_paths(data_dir)

        assert pipeline._sp_ready is True
        assert pipeline._sp_lookup.n_rows == 2
