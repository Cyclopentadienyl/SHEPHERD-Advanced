"""EVALUATION_COHORTS §6.6 step 4 — what the current generator actually produced.

§1.5 established the qualitative gap to the upstream simulator by reading its
source. These tests cover the three statements it left **unmeasured**, and the
fourth question — whether a frequency signal exists to weight by at all.

**Every fixture here is built so the answer is knowable by hand.** A capacity of
`C(2, 2) = 1` drawn five times is four excess samples, and no arithmetic in the
script gets to decide otherwise. The KG-scale correctness of the same code is not
something a unit test can establish; what it can establish is that the code
computes the quantity its report claims to.

Module: tests/unit/test_audit_generator_fidelity.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.core.types import DataSource, EdgeType, NodeType
from src.kg.graph import Edge, KnowledgeGraph, Node, NodeID

import scripts.audit_generator_fidelity as fidelity


def _hp(i: int) -> NodeID:
    return NodeID(source=DataSource.HPO, local_id=f"HP:{i:07d}")


def _mondo(i: int) -> NodeID:
    return NodeID(source=DataSource.MONDO, local_id=f"MONDO:{i:07d}")


def _workspace(
    root: Path,
    diseases,
    train,
    val,
    *,
    weights=None,
    manifest_config=None,
    manifest=True,
):
    """A workspace with exactly the content a test needs.

    ``diseases`` is a list of phenotype-index lists, one per disease, in the order
    that fixes their indices. ``train`` and ``val`` are lists of
    ``(disease_index, [phenotype indices])``. The manifest is built through the
    shared fixture, so it is bound to the sample bytes the way a real one is.
    """
    from tests.fixtures.generated_workspace import write_generated_workspace

    root.mkdir(parents=True, exist_ok=True)
    kg = KnowledgeGraph()
    n_phenotypes = 1 + max((p for ps in diseases for p in ps), default=0)
    for i in range(n_phenotypes):
        kg.add_node(Node(id=_hp(i), node_type=NodeType.PHENOTYPE, name=f"p{i}"))
    for d, phenotypes in enumerate(diseases):
        kg.add_node(Node(id=_mondo(d), node_type=NodeType.DISEASE, name=f"d{d}"))
        for p in phenotypes:
            kg.add_edge(Edge(
                source_id=_hp(p), target_id=_mondo(d),
                edge_type=EdgeType.PHENOTYPE_OF_DISEASE,
                weight=1.0 if weights is None else weights[(d, p)],
            ))
    kg.save_json(str(root / "kg.json"))

    rows = {
        split: [
            {"patient_id": f"SECRET-{split}-{i}", "disease_id": d, "phenotype_ids": ps}
            for i, (d, ps) in enumerate(pairs)
        ]
        for split, pairs in (("train", train), ("val", val))
    }
    if not manifest:
        for split, samples in rows.items():
            (root / f"{split}_samples.json").write_text(json.dumps(samples))
        return root

    profiles = {
        d: {"phenotype_ids": list(ps), "gene_ids": []}
        for d, ps in enumerate(diseases)
    }
    write_generated_workspace(
        root,
        train_ids=sorted({d for d, _ in train}),
        val_ids=sorted({d for d, _ in val}),
        profiles=profiles,
        train_samples=rows["train"], val_samples=rows["val"],
        config=manifest_config,
    )
    return root


def _run(root: Path, out: Path, **kwargs):
    args = ["--kg-path", str(root / "kg.json"), "--data-dir", str(root),
            "--output", str(out)]
    for key, value in kwargs.items():
        args += [f"--{key.replace('_', '-')}", str(value)]
    fidelity.main(args)
    return json.loads(out.read_text())


DEFAULTS = dict(min_phenotypes=2, max_phenotypes=15, phenotype_drop_rate=0.3)


# ---------------------------------------------------------------------------
# 1. Combinatorial capacity
# ---------------------------------------------------------------------------
def test_a_disease_drawn_past_its_capacity_is_counted_with_its_excess(tmp_path):
    """P = 2 gives k = 2 and C(2, 2) = 1: every draw after the first repeats."""
    root = _workspace(
        tmp_path / "ws",
        diseases=[[0, 1], [2, 3, 4, 5]],
        train=[(0, [0, 1])] * 5,
        val=[(1, [2, 3])],
        manifest_config=DEFAULTS,
    )
    report = _run(root, tmp_path / "f.json")
    capacity = report["combinatorial_capacity"]["per_split"]["train"]

    assert capacity["diseases_drawn"] == 1
    assert capacity["diseases_drawn_past_capacity"] == 1
    assert capacity["samples_in_excess_of_capacity"] == 4
    assert capacity["diseases_without_a_known_capacity"] == 0


def test_capacity_is_computed_at_the_generation_k_not_at_a_guessed_one(tmp_path):
    """A drop rate of 0.0 keeps every phenotype, so C(4, 4) = 1 and four draws
    are three in excess. At 0.5 it keeps two, C(4, 2) = 6, and nothing exceeds."""
    rows = [(0, [0, 1, 2, 3])] * 4
    counts = {}
    for rate in (0.0, 0.5):
        root = _workspace(
            tmp_path / f"ws{rate}", diseases=[[0, 1, 2, 3], [4, 5]],
            train=rows, val=[(1, [4, 5])],
            manifest_config={**DEFAULTS, "phenotype_drop_rate": rate},
        )
        report = _run(root, tmp_path / f"f{rate}.json")
        counts[rate] = report["combinatorial_capacity"]["per_split"]["train"]

    assert counts[0.0]["samples_in_excess_of_capacity"] == 3
    assert counts[0.5]["samples_in_excess_of_capacity"] == 0


def test_the_manifest_is_the_only_source_of_the_generation_rule(tmp_path):
    """`k` is not recoverable from the samples, so there is nothing for an
    operator flag to be checked against. There is no flag."""
    root = _workspace(tmp_path / "ws", diseases=[[0, 1], [2, 3]],
                      train=[(0, [0, 1])], val=[(1, [2, 3])],
                      manifest_config=DEFAULTS)
    report = _run(root, tmp_path / "f.json")

    assert report["generation_config_source"] == "split_manifest.json"
    with pytest.raises(SystemExit):
        _run(root, tmp_path / "g.json", phenotype_drop_rate=0.9)


def test_a_workspace_without_a_manifest_is_refused(tmp_path):
    """It was built before the allocation step, its cohorts overlap, and nothing
    reads such a workspace any more."""
    root = _workspace(tmp_path / "ws", diseases=[[0, 1], [2, 3]],
                      train=[(0, [0, 1])], val=[(1, [2, 3])], manifest=False)
    with pytest.raises(SystemExit, match="generated before the disease allocation"):
        _run(root, tmp_path / "f.json")


def test_a_manifest_missing_the_generation_rule_is_refused(tmp_path):
    root = _workspace(tmp_path / "ws", diseases=[[0, 1], [2, 3]],
                      train=[(0, [0, 1])], val=[(1, [2, 3])],
                      manifest_config={"min_phenotypes": 2})
    with pytest.raises(SystemExit, match="max_phenotypes"):
        _run(root, tmp_path / "f.json")


# ---------------------------------------------------------------------------
# 2. Realised redundancy
# ---------------------------------------------------------------------------
def test_redundancy_counts_content_not_records(tmp_path):
    """Three records with one content are one training signal shown three times."""
    root = _workspace(
        tmp_path / "ws", diseases=[[0, 1, 2], [3, 4]],
        train=[(0, [0, 1]), (0, [0, 1]), (0, [0, 1]), (0, [1, 2])],
        val=[(1, [3, 4])], manifest_config=DEFAULTS,
    )
    train = _run(root, tmp_path / "f.json")["realised_redundancy"]["train"]

    assert train["samples"] == 4
    assert train["distinct_signatures"] == 2
    assert train["duplicate_samples"] == 2
    assert train["distinct_over_samples"] == 0.5


def test_the_phenotype_order_does_not_make_two_samples_distinct(tmp_path):
    """The model sees a set. A signature that respected order would report a
    cohort as more varied than it is."""
    root = _workspace(
        tmp_path / "ws", diseases=[[0, 1, 2], [3, 4]],
        train=[(0, [0, 1]), (0, [1, 0])], val=[(1, [3, 4])],
        manifest_config=DEFAULTS,
    )
    train = _run(root, tmp_path / "f.json")["realised_redundancy"]["train"]

    assert train["distinct_signatures"] == 1


def test_the_same_phenotypes_under_two_diseases_are_two_signatures(tmp_path):
    """The disease id is part of what the model is asked to predict."""
    root = _workspace(
        tmp_path / "ws", diseases=[[0, 1], [0, 1], [0, 1]],
        train=[(0, [0, 1]), (1, [0, 1])], val=[(2, [0, 1])],
        manifest_config=DEFAULTS,
    )
    train = _run(root, tmp_path / "f.json")["realised_redundancy"]["train"]

    assert train["distinct_signatures"] == 2


# ---------------------------------------------------------------------------
# 3. The channel a disease-level cut does not close
# ---------------------------------------------------------------------------
def test_a_phenotype_set_appearing_under_two_labels_across_the_cut_is_counted(tmp_path):
    """Disease ids are disjoint by construction; phenotype sets are not. This is
    a validation input the model saw in training with a different answer."""
    root = _workspace(
        tmp_path / "ws", diseases=[[0, 1], [0, 1], [2, 3]],
        train=[(0, [0, 1]), (2, [2, 3])],
        val=[(1, [0, 1])], manifest_config=DEFAULTS,
    )
    drawn = _run(root, tmp_path / "f.json")["cross_split_phenotype_sets"]["drawn"]

    assert drawn["distinct_phenotype_sets_in_both_splits"] == 1
    assert drawn["train_samples_touched"] == 1
    assert drawn["val_samples_touched"] == 1


def test_disjoint_phenotype_content_reports_no_shared_sets(tmp_path):
    root = _workspace(
        tmp_path / "ws", diseases=[[0, 1], [2, 3]],
        train=[(0, [0, 1])], val=[(1, [2, 3])], manifest_config=DEFAULTS,
    )
    drawn = _run(root, tmp_path / "f.json")["cross_split_phenotype_sets"]["drawn"]

    assert drawn["distinct_phenotype_sets_in_both_splits"] == 0
    assert drawn["val_samples_touched"] == 0


def test_the_structural_measure_does_not_depend_on_what_was_drawn(tmp_path):
    """Two diseases with identical knowledge-graph profiles, split across the cut,
    and cohorts that happen to share no drawn subset. The drawn measure says zero
    and is right; the structural measure says one and is the figure that survives
    regeneration under a different seed."""
    root = _workspace(
        tmp_path / "ws", diseases=[[0, 1, 2, 3], [0, 1, 2, 3]],
        train=[(0, [0, 1])], val=[(1, [2, 3])], manifest_config=DEFAULTS,
    )
    section = _run(root, tmp_path / "f.json")["cross_split_phenotype_sets"]

    assert section["drawn"]["distinct_phenotype_sets_in_both_splits"] == 0
    assert section["structural"]["val_diseases_with_an_identical_profile_in_train"] == 1
    assert section["structural"]["val_diseases_that_could_share_an_input_with_train"] == 1
    assert section["structural"]["val_diseases_measured"] == 1


def test_unequal_profiles_that_can_emit_the_same_input_are_counted(tmp_path):
    """The case restricting the measure to identical profiles misses.

    A = {0,1,2,3}, B = {0,1,2,4}, and k = 2 for a four-phenotype profile at the
    default drop rate. Neither profile duplicates the other, and both can emit
    {0,1}. The condition is `k_A == k_B == k and |A ∩ B| >= k`, not `A == B`.
    """
    root = _workspace(
        tmp_path / "ws", diseases=[[0, 1, 2, 3], [0, 1, 2, 4]],
        train=[(0, [0, 1])], val=[(1, [2, 4])], manifest_config=DEFAULTS,
    )
    section = _run(root, tmp_path / "f.json")["cross_split_phenotype_sets"]

    assert section["structural"]["val_diseases_with_an_identical_profile_in_train"] == 0
    assert section["structural"]["val_diseases_that_could_share_an_input_with_train"] == 1
    assert section["structural"]["condition"] == "k_A == k_B == k and |A ∩ B| >= k"


def test_profiles_sharing_too_few_phenotypes_cannot_collide(tmp_path):
    """|A ∩ B| = 1 < k = 2: no k-subset is emittable by both."""
    root = _workspace(
        tmp_path / "ws", diseases=[[0, 1, 2, 3], [0, 4, 5, 6]],
        train=[(0, [0, 1])], val=[(1, [4, 5])], manifest_config=DEFAULTS,
    )
    section = _run(root, tmp_path / "f.json")["cross_split_phenotype_sets"]

    assert section["structural"]["val_diseases_that_could_share_an_input_with_train"] == 0


def test_profiles_with_different_k_cannot_collide_however_much_they_share(tmp_path):
    """B is a strict superset of A, so they share every one of A's phenotypes --
    and still cannot emit the same set, because they retain different numbers."""
    root = _workspace(
        tmp_path / "ws",
        diseases=[[0, 1, 2], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
        train=[(0, [0, 1])], val=[(1, [0, 1, 2, 3, 4, 5, 6])],
        manifest_config=DEFAULTS,
    )
    section = _run(root, tmp_path / "f.json")["cross_split_phenotype_sets"]

    assert section["structural"]["val_diseases_that_could_share_an_input_with_train"] == 0


def test_the_index_agrees_with_an_exhaustive_pairwise_oracle():
    """The early-exit inverted index is an optimisation, so it is checked against
    the definition it optimises rather than against itself."""
    import random

    from scripts.audit_generator_fidelity import _shared_input_is_possible

    def oracle(val_profiles, train_profiles, k_of):
        return sum(
            1 for v, a in val_profiles.items()
            if any(k_of[v] == k_of[t] and len(a & b) >= k_of[v]
                   for t, b in train_profiles.items())
        )

    rng = random.Random("shepherd|structural-collision")
    for _ in range(200):
        n_train, n_val = rng.randint(0, 6), rng.randint(0, 6)
        profiles, k_of = {}, {}
        for disease in range(n_train + n_val):
            size = rng.randint(1, 5)
            profiles[disease] = frozenset(rng.sample(range(8), size))
            k_of[disease] = rng.randint(1, 3)
        train = {d: profiles[d] for d in range(n_train)}
        val = {d: profiles[d] for d in range(n_train, n_train + n_val)}

        assert _shared_input_is_possible(val, train, k_of) == oracle(val, train, k_of)


# ---------------------------------------------------------------------------
# 4. Is there a frequency signal to weight by?
# ---------------------------------------------------------------------------
def test_the_graph_supports_only_a_lower_bound_and_says_so(tmp_path):
    """`_parse_frequency` returns 1.0 both for a missing annotation and for a real
    one, so the 1.0 bucket cannot be read as "no frequency"."""
    root = _workspace(
        tmp_path / "ws", diseases=[[0, 1], [2, 3]],
        train=[(0, [0, 1])], val=[(1, [2, 3])],
        weights={(0, 0): 0.55, (0, 1): 1.0, (1, 2): 0.1, (1, 3): 1.0},
        manifest_config=DEFAULTS,
    )
    section = _run(root, tmp_path / "f.json")["frequency_signal"]
    bound = section["from_the_graph"]

    assert bound["phenotype_disease_edges"] == 4
    assert bound["edges_with_a_non_default_weight"] == 2
    assert bound["fraction"] == 0.5
    assert "conflates" in bound["why_only_a_lower_bound"]
    assert section["from_the_source"]["measured"] is False


def test_the_source_column_gives_the_exact_figure(tmp_path):
    """Four comparable rows, two carrying a frequency. The header, a comment, a
    NOT-qualified row and a non-HP row are all excluded, as the parser excludes
    them — a looser denominator would not be comparable to the graph's edges."""
    external = tmp_path / "ext"
    external.mkdir()
    rows = [
        "#description: a comment line",
        "database_id\tdisease_name\tqualifier\thpo_id\treference\tevidence\tonset\tfrequency\tsex",
        "OMIM:1\tA\t\tHP:0000001\tR\tE\t\tHP:0040281\t",
        "OMIM:1\tA\t\tHP:0000002\tR\tE\t\t3/12\t",
        # Trailing empty columns, and a row truncated before the frequency column
        # at all. Both are unannotated rows that must stay in the denominator:
        # dropping them is how a coverage fraction comes out near 1.0 while
        # measuring almost nothing.
        "OMIM:1\tA\t\tHP:0000003\tR\tE\t\t\t",
        "OMIM:1\tA\t\tHP:0000004\tR\tE",
        "OMIM:1\tA\tNOT\tHP:0000005\tR\tE\t\t90%\t",
    ]
    (external / "phenotype.hpoa").write_text("\n".join(rows) + "\n")
    root = _workspace(tmp_path / "ws", diseases=[[0, 1], [2, 3]],
                      train=[(0, [0, 1])], val=[(1, [2, 3])],
                      manifest_config=DEFAULTS)

    import hashlib

    report = _run(root, tmp_path / "f.json", external_dir=external)
    exact = report["frequency_signal"]["from_the_source"]

    assert exact["measured"] is True
    assert exact["rows"] == 4
    # HP:0040281 parses to 0.90 and 3/12 to 0.25 -- both certainly usable. The
    # two empty-token rows are certainly not. Nothing is ambiguous here.
    assert exact["rows_whose_token_parses_below_one"] == 2
    assert exact["rows_with_no_frequency_token"] == 2
    assert exact["rows_whose_token_parses_to_one"] == 0
    assert exact["usable_frequency_fraction_lower_bound"] == 0.5
    assert exact["usable_frequency_fraction_upper_bound"] == 0.5

    # A basename is not identity: the report says which bytes it read.
    expected = hashlib.sha256((external / "phenotype.hpoa").read_bytes()).hexdigest()
    assert exact["digest"] == expected
    assert report["artifacts"]["phenotype_hpoa"] == expected


def test_an_obligate_token_is_ambiguous_and_widens_the_bounds(tmp_path):
    """`HP:0040280`, `100%` and `12/12` all parse to 1.0, which is also what an
    unparseable token returns. Reporting either bound alone would overstate."""
    external = tmp_path / "ext"
    external.mkdir()
    (external / "phenotype.hpoa").write_text("\n".join([
        "OMIM:1\tA\t\tHP:0000001\tR\tE\t\tHP:0040280\t",
        "OMIM:1\tA\t\tHP:0000002\tR\tE\t\t100%\t",
        "OMIM:1\tA\t\tHP:0000003\tR\tE\t\tnonsense\t",
        "OMIM:1\tA\t\tHP:0000004\tR\tE\t\t3/12\t",
    ]) + "\n")
    root = _workspace(tmp_path / "ws", diseases=[[0, 1], [2, 3]],
                      train=[(0, [0, 1])], val=[(1, [2, 3])],
                      manifest_config=DEFAULTS)

    exact = _run(root, tmp_path / "f.json",
                 external_dir=external)["frequency_signal"]["from_the_source"]

    assert exact["rows"] == 4
    assert exact["rows_whose_token_parses_below_one"] == 1
    assert exact["rows_whose_token_parses_to_one"] == 3
    assert exact["usable_frequency_fraction_lower_bound"] == 0.25
    assert exact["usable_frequency_fraction_upper_bound"] == 1.0


# ---------------------------------------------------------------------------
# Boundaries
# ---------------------------------------------------------------------------
def test_an_emptied_cohort_is_refused_by_the_manifest_binding(tmp_path):
    """Over an empty cohort every section here would be vacuously clean. It cannot
    arise from a verified workspace: the coverage contract puts a sample in every
    allocated disease, so emptying the file breaks the binding first."""
    root = _workspace(tmp_path / "ws", diseases=[[0, 1], [2, 3]],
                      train=[(0, [0, 1])], val=[(1, [2, 3])],
                      manifest_config=DEFAULTS)
    (root / "val_samples.json").write_text("[]")

    with pytest.raises(SystemExit, match="is not the file"):
        _run(root, tmp_path / "f.json")


def test_an_existing_output_is_not_overwritten_by_default(tmp_path):
    root = _workspace(tmp_path / "ws", diseases=[[0, 1], [2, 3]],
                      train=[(0, [0, 1])], val=[(1, [2, 3])],
                      manifest_config=DEFAULTS)
    out = tmp_path / "f.json"
    out.write_text("{}")
    with pytest.raises(SystemExit, match="Pass --overwrite"):
        _run(root, out)


def test_the_artifact_carries_no_identifiers(tmp_path):
    """BACKLOG §5.2: no patient ids, no sample ids, no per-disease lists."""
    root = _workspace(
        tmp_path / "ws", diseases=[[0, 1, 2], [3, 4]],
        train=[(0, [0, 1]), (0, [1, 2])], val=[(1, [3, 4])],
        manifest_config=DEFAULTS,
    )
    text = (tmp_path / "f.json")
    _run(root, text)
    body = text.read_text()

    assert "SECRET" not in body
    assert "MONDO:" not in body and "HP:0000" not in body
    # `patient_id` appears once, in prose stating that the signature excludes it.
    # The rule §5.2 sets is about identifier *values*, so the check is that no
    # JSON key by that name exists rather than that the token never occurs.
    assert '"patient_id"' not in body


def test_the_report_states_that_it_is_not_a_verdict(tmp_path):
    """Whether a redundancy figure is acceptable is not an engineering call, and
    an artifact that omitted this invites being read as one."""
    root = _workspace(tmp_path / "ws", diseases=[[0, 1], [2, 3]],
                      train=[(0, [0, 1])], val=[(1, [2, 3])],
                      manifest_config=DEFAULTS)
    report = _run(root, tmp_path / "f.json")

    assert "no threshold is applied" in report["not_a_verdict"]
    assert report["schema_version"] == 1
