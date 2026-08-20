"""
The SP lookup benchmark's artifact mode must time the artifact.
===============================================================
These tests exist because an earlier version of `scripts/benchmark_sp_lookup.py`
read a real artifact, extracted only its **mean** slice length, and then timed a
*synthetic* table parameterised by that mean — while reporting "benchmark
complete on a measured artifact distribution". The artifact's tail,
phenotype-to-slice mapping and target layout never entered the timed workload.

The defect was in the wiring rather than in any one function, so the tests here
check the wiring: that artifact mode builds its lookup from the file's own rows,
and that a synthetic run cannot describe itself as an artifact measurement.

Module: tests/integration/test_benchmark_sp_lookup.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.benchmark_sp_lookup import build_artifact_lookup, main


def write_artifact(path: Path, n_phenotypes: int = 25, target_space: int = 200) -> dict:
    """A file in `save_shortest_paths`' on-disk format (`compute_shortest_paths.py`)."""
    generator = torch.Generator().manual_seed(7)
    phenotype, target, target_type, distance = [], [], [], []
    expected = {}
    for p in range(n_phenotypes):
        length = int(torch.randint(5, 30, (1,), generator=generator))
        picked = torch.randperm(target_space, generator=generator)[:length]
        expected[p] = sorted(picked.tolist())
        for t in picked.tolist():
            phenotype.append(p)
            target.append(t)
            target_type.append(int(torch.randint(0, 2, (1,), generator=generator)))
            distance.append(int(torch.randint(1, 6, (1,), generator=generator)))
    torch.save(
        {
            "phenotype_idx": torch.tensor(phenotype, dtype=torch.int64),
            "target_idx": torch.tensor(target, dtype=torch.int64),
            "target_type": torch.tensor(target_type, dtype=torch.int64),
            "distance": torch.tensor(distance, dtype=torch.int8),
        },
        path,
    )
    return expected


def test_artifact_lookup_slices_are_the_files_own_rows(tmp_path):
    """Not a table shaped like the artifact — the artifact's rows."""
    artifact = tmp_path / "shortest_paths.pt"
    expected = write_artifact(artifact)

    lookup, lengths, disease_targets, provenance = build_artifact_lookup(artifact, 5)

    assert set(lookup.offsets) == set(expected)
    for phenotype, targets in expected.items():
        start, end = lookup.offsets[phenotype]
        assert sorted(lookup.target[start:end].tolist()) == targets
        assert end - start == len(targets)
    assert lengths == [
        lookup.offsets[p][1] - lookup.offsets[p][0] for p in sorted(lookup.offsets)
    ]
    assert provenance["n_pairs"] == sum(len(v) for v in expected.values())
    assert len(provenance["sha256"]) == 64


def test_candidates_are_drawn_from_the_real_disease_target_space(tmp_path):
    """A candidate the artifact has no disease row for is not a realistic query."""
    artifact = tmp_path / "shortest_paths.pt"
    write_artifact(artifact)

    lookup, _, disease_targets, _ = build_artifact_lookup(artifact, 5)

    assert disease_targets, "fixture produced no disease targets"
    present = set(lookup.target[lookup.target_type == 1].tolist())
    assert set(disease_targets) == present


def test_missing_keys_are_fatal_rather_than_silently_partial(tmp_path):
    artifact = tmp_path / "shortest_paths.pt"
    torch.save({"phenotype_idx": torch.zeros(3, dtype=torch.int64)}, artifact)

    with pytest.raises(SystemExit, match="missing required keys"):
        build_artifact_lookup(artifact, 5)


def test_artifact_run_records_the_artifact_as_its_slice_source(tmp_path):
    artifact = tmp_path / "shortest_paths.pt"
    write_artifact(artifact)
    output = tmp_path / "run.json"

    main(["--artifact", str(artifact), "--output", str(output)])
    report = json.loads(output.read_text())

    assert report["slice_source"]["source"] == "artifact"
    assert report["slice_source"]["sha256"]
    assert report["provenance"]["mode"] == "artifact"
    assert report["rows"]
    # A real artifact is one of PLAN_B04 §3.1's two requirements; a
    # deployment-equivalent CPU is the other, and this script cannot self-attest
    # it. Asserting the exact verdict rather than the absence of a phrase, so
    # that "not the synthetic wording" is never mistaken for "gate cleared".
    assert report["verdict"] == (
        "artifact slices timed; baseline acceptance remains subject to the "
        "deployment-equivalent CPU gate"
    )
    assert report["provenance"]["deployment_equivalent_cpu"] is False


def test_measurement_order_actually_alternates(tmp_path, monkeypatch):
    """The order was claimed, not performed.

    The first version alternated on `len(rows) % 2`. Each timed cell appends two
    rows, so `len(rows)` is even at every cell boundary and the branch never
    fired: all 240 rows of the committed evidence recorded
    `measured_first="singleton"` while §8.1 said the order was alternated.
    """
    import scripts.benchmark_sp_lookup as bench

    monkeypatch.setattr(bench, "SYNTHETIC_MEAN_SLICE_LENGTHS", (20,))
    monkeypatch.setattr(bench, "SYNTHETIC_DISTRIBUTIONS", ("representative",))
    monkeypatch.setattr(bench, "CANDIDATE_COUNTS", (5, 10))
    monkeypatch.setattr(bench, "PHENOTYPE_COUNTS", (1, 2))
    monkeypatch.setattr(bench, "SYNTHETIC_TARGET_SPACE", 200)
    output = tmp_path / "run.json"

    bench.main(["--output", str(output)])
    rows = json.loads(output.read_text())["rows"]

    cell_key = lambda r: (  # noqa: E731 - a local grouping key, not a policy
        r["candidates"], r["phenotypes"], r["phenotype_selection"],
        r["distribution"], r["mean_slice_length"],
    )
    cells: dict = {}
    for row in rows:
        cells.setdefault(cell_key(row), []).append(row)

    assert len(cells) > 1, "fixture must produce more than one timed cell"
    for key, pair in cells.items():
        assert {r["caller_shape"] for r in pair} == {"singleton", "batched"}, key
        assert len({r["measured_first"] for r in pair}) == 1, (
            f"both rows of cell {key} must record the same measurement order"
        )

    # `cells` is insertion-ordered, and rows are appended in timing order, so
    # this list is the timed-cell sequence.
    ordered = [pair[0]["measured_first"] for pair in cells.values()]

    assert set(ordered) == {"singleton", "batched"}, (
        f"measurement order never alternated across cells; saw only {set(ordered)}"
    )
    # Both orders occurring is not alternation: singleton, singleton, batched,
    # batched would satisfy the assertion above and still leave a run of cells
    # measured the same way round.
    assert all(a != b for a, b in zip(ordered, ordered[1:])), (
        f"adjacent timed cells did not alternate: {ordered}"
    )


def test_synthetic_run_never_claims_an_artifact_measurement(tmp_path, monkeypatch):
    """The regression this file exists for, stated as an assertion."""
    import scripts.benchmark_sp_lookup as bench

    monkeypatch.setattr(bench, "SYNTHETIC_MEAN_SLICE_LENGTHS", (20,))
    monkeypatch.setattr(bench, "SYNTHETIC_DISTRIBUTIONS", ("representative",))
    monkeypatch.setattr(bench, "CANDIDATE_COUNTS", (5,))
    monkeypatch.setattr(bench, "PHENOTYPE_COUNTS", (1,))
    monkeypatch.setattr(bench, "SYNTHETIC_TARGET_SPACE", 200)
    output = tmp_path / "run.json"

    bench.main(["--output", str(output)])
    report = json.loads(output.read_text())

    assert report["slice_source"]["source"] == "synthetic"
    assert report["provenance"]["mode"] == "synthetic"
    assert "artifact" not in report["verdict"]
    assert "pending institutional run" in report["verdict"]


# =============================================================================
# Prototype selection (B-0.4 prototype phase)
# =============================================================================
def test_unknown_implementation_is_rejected():
    """A typo must not silently benchmark fewer implementations than asked for."""
    from scripts.benchmark_sp_lookup import main

    with pytest.raises(SystemExit):
        main(["--implementations", "current,globl"])


def test_memory_attribution_flag_is_honest(tmp_path, monkeypatch):
    """`ru_maxrss` is a process high-water mark, so two prototypes in one process
    cannot both be attributed. The report must say so rather than imply isolation.

    Asserted in both directions: one prototype is isolated, two are not.
    """
    import scripts.benchmark_sp_lookup as bench

    monkeypatch.setattr(bench, "CANDIDATE_COUNTS", (10,))
    monkeypatch.setattr(bench, "PHENOTYPE_COUNTS", (1,))
    monkeypatch.setattr(bench, "SYNTHETIC_MEAN_SLICE_LENGTHS", (100,))
    monkeypatch.setattr(bench, "SYNTHETIC_DISTRIBUTIONS", ("representative",))
    monkeypatch.setattr(bench, "MIN_REPEATS", 1)
    monkeypatch.setattr(bench, "MAX_REPEATS", 1)
    monkeypatch.setattr(bench, "TARGET_MEASURE_SECONDS", 0.0)

    one = tmp_path / "one.json"
    assert bench.main(["--implementations", "global", "--output", str(one)]) == 0
    single = json.loads(one.read_text())
    assert single["memory_attribution_isolated"] is True
    assert single["stage"] == "B-0.4 prototype"
    assert [b["implementation"] for b in single["index_builds"]] == ["global"]

    both = tmp_path / "both.json"
    assert bench.main(
        ["--implementations", "current,global,slices", "--output", str(both)]
    ) == 0
    pair = json.loads(both.read_text())
    assert pair["memory_attribution_isolated"] is False
    assert pair["implementations"] == ["current", "global", "slices"]


def test_every_implementation_sees_the_same_cells(tmp_path, monkeypatch):
    """A timing difference must be the implementation, not a different workload."""
    import scripts.benchmark_sp_lookup as bench

    monkeypatch.setattr(bench, "CANDIDATE_COUNTS", (10, 50))
    monkeypatch.setattr(bench, "PHENOTYPE_COUNTS", (1, 20))
    monkeypatch.setattr(bench, "SYNTHETIC_MEAN_SLICE_LENGTHS", (100,))
    monkeypatch.setattr(bench, "SYNTHETIC_DISTRIBUTIONS", ("representative",))
    monkeypatch.setattr(bench, "MIN_REPEATS", 1)
    monkeypatch.setattr(bench, "MAX_REPEATS", 1)
    monkeypatch.setattr(bench, "TARGET_MEASURE_SECONDS", 0.0)

    output = tmp_path / "cells.json"
    assert bench.main(
        ["--implementations", "current,global,slices", "--output", str(output)]
    ) == 0
    rows = json.loads(output.read_text())["rows"]

    def cells(name):
        return sorted(
            (r["candidates"], r["phenotypes"], r["phenotype_selection"],
             r["caller_shape"], r["queried_slice_total"])
            for r in rows if r["implementation"] == name
        )

    assert cells("current") == cells("global") == cells("slices")
    assert cells("current"), "the matrix produced no rows to compare"


def test_implementation_order_rotates_and_workload_stays_identical(tmp_path, monkeypatch):
    """BLOCKING 3: `current` must not be measured first in every cell.

    Iterating the implementations in insertion order put `current` first in every
    cell of every documented command, so any warm-up or cache advantage accrued
    to the same implementation throughout. Both halves are asserted together
    because either alone is satisfiable by a broken benchmark: rotating the order
    while varying the workload would be worse than not rotating at all.
    """
    import scripts.benchmark_sp_lookup as bench

    monkeypatch.setattr(bench, "CANDIDATE_COUNTS", (10, 20))
    monkeypatch.setattr(bench, "PHENOTYPE_COUNTS", (1, 20))
    monkeypatch.setattr(bench, "SYNTHETIC_MEAN_SLICE_LENGTHS", (100,))
    monkeypatch.setattr(bench, "SYNTHETIC_DISTRIBUTIONS", ("representative",))
    monkeypatch.setattr(bench, "MIN_REPEATS", 1)
    monkeypatch.setattr(bench, "MAX_REPEATS", 1)
    monkeypatch.setattr(bench, "TARGET_MEASURE_SECONDS", 0.0)

    output = tmp_path / "rotation.json"
    assert bench.main(
        ["--implementations", "current,global,slices", "--output", str(output)]
    ) == 0
    rows = json.loads(output.read_text())["rows"]

    first_per_cell = {}
    for row in rows:
        if row["implementation_position"] != 0:
            continue
        key = (row["candidates"], row["phenotypes"], row["phenotype_selection"],
               row["distribution"], row["mean_slice_length"])
        first_per_cell[key] = row["implementation"]

    assert len(first_per_cell) > 1, "fixture must produce more than one timed cell"
    assert "current" in first_per_cell.values(), "no cell measured current first"
    assert {v for v in first_per_cell.values()} - {"current"}, (
        f"no cell measured a prototype first: {first_per_cell}"
    )

    def workload(name):
        return sorted(
            (r["candidates"], r["phenotypes"], r["phenotype_selection"],
             r["caller_shape"], r["queried_slice_total"])
            for r in rows if r["implementation"] == name
        )

    assert workload("current") == workload("global") == workload("slices")


def test_candidates_are_sampled_without_replacement(tmp_path, monkeypatch):
    """MAJOR 2: a repeated candidate is not a workload production can present.

    The real disease candidate list is a set, and duplicates would also flatter a
    binary search whose repeated probes hit the same cache lines.
    """
    import scripts.benchmark_sp_lookup as bench

    seen = []
    original = bench._repeat

    def capture(fn, table, phenotypes, candidates):
        seen.append(list(candidates))
        return original(fn, table, phenotypes, candidates)

    monkeypatch.setattr(bench, "_repeat", capture)
    monkeypatch.setattr(bench, "CANDIDATE_COUNTS", (25,))
    monkeypatch.setattr(bench, "PHENOTYPE_COUNTS", (1,))
    monkeypatch.setattr(bench, "SYNTHETIC_MEAN_SLICE_LENGTHS", (100,))
    monkeypatch.setattr(bench, "SYNTHETIC_DISTRIBUTIONS", ("representative",))
    monkeypatch.setattr(bench, "SYNTHETIC_TARGET_SPACE", 60)
    monkeypatch.setattr(bench, "MIN_REPEATS", 1)
    monkeypatch.setattr(bench, "MAX_REPEATS", 1)
    monkeypatch.setattr(bench, "TARGET_MEASURE_SECONDS", 0.0)

    assert bench.main(["--output", str(tmp_path / "unique.json")]) == 0

    assert seen, "no cell was timed"
    for candidates in seen:
        assert len(set(candidates)) == len(candidates), (
            f"candidates repeated within one cell: {candidates}"
        )


def test_a_cell_wanting_more_candidates_than_exist_is_reported_skipped(tmp_path, monkeypatch):
    """Sampling without replacement cannot invent candidates, and must not cap
    silently — a silent cap reads as "covered everything" when it did not."""
    import scripts.benchmark_sp_lookup as bench

    monkeypatch.setattr(bench, "CANDIDATE_COUNTS", (5_000,))
    monkeypatch.setattr(bench, "PHENOTYPE_COUNTS", (1,))
    monkeypatch.setattr(bench, "SYNTHETIC_MEAN_SLICE_LENGTHS", (50,))
    monkeypatch.setattr(bench, "SYNTHETIC_DISTRIBUTIONS", ("representative",))
    monkeypatch.setattr(bench, "SYNTHETIC_TARGET_SPACE", 60)
    monkeypatch.setattr(bench, "MIN_REPEATS", 1)
    monkeypatch.setattr(bench, "MAX_REPEATS", 1)
    monkeypatch.setattr(bench, "TARGET_MEASURE_SECONDS", 0.0)

    output = tmp_path / "skipped.json"
    assert bench.main(["--output", str(output)]) == 0
    report = json.loads(output.read_text())

    assert report["rows"] == []
    reasons = {s["reason"] for s in report["skipped"]}
    assert any("unique candidates" in r for r in reasons), reasons


@pytest.mark.parametrize("prototype", ["global", "slices"])
def test_shape_order_alternates_for_every_implementation(tmp_path, monkeypatch, prototype):
    """BLOCKING regression: run the **documented two-implementation configs**.

    An earlier version chose the shape order with `(cell_index + position) % 2`,
    meaning to decorrelate it from the rotated implementation order. With two
    implementations the rotation moves `position` in lockstep with `cell_index`,
    so the sum was constant per implementation *identity*: on the real artifact
    `current` came out singleton-first in 60/60 rows and the prototype
    batched-first in 60/60. Any warm-up or cache asymmetry between the shapes
    then attached permanently to one implementation.

    A single-implementation run cannot see this, and the previous test used one.
    These use `current,global` and `current,slices` — the two commands PLAN_B04
    §11.3 actually tells the operator to run.
    """
    import scripts.benchmark_sp_lookup as bench

    monkeypatch.setattr(bench, "CANDIDATE_COUNTS", (10, 20))
    monkeypatch.setattr(bench, "PHENOTYPE_COUNTS", (1, 20))
    monkeypatch.setattr(bench, "SYNTHETIC_MEAN_SLICE_LENGTHS", (100,))
    monkeypatch.setattr(bench, "SYNTHETIC_DISTRIBUTIONS", ("representative",))
    monkeypatch.setattr(bench, "MIN_REPEATS", 1)
    monkeypatch.setattr(bench, "MAX_REPEATS", 1)
    monkeypatch.setattr(bench, "TARGET_MEASURE_SECONDS", 0.0)

    output = tmp_path / f"order_{prototype}.json"
    assert bench.main(
        ["--implementations", f"current,{prototype}", "--output", str(output)]
    ) == 0
    rows = json.loads(output.read_text())["rows"]

    for implementation in ("current", prototype):
        orders = {r["measured_first"] for r in rows if r["implementation"] == implementation}
        assert orders == {"singleton", "batched"}, (
            f"{implementation} was always measured {orders} first across every "
            "cell; shape order is coupled to implementation identity"
        )

    # Both implementation orders still occur, and both still see one workload.
    positions = {
        (r["implementation"], r["implementation_position"]) for r in rows
    }
    for implementation in ("current", prototype):
        assert {p for i, p in positions if i == implementation} == {0, 1}, implementation

    def workload(name):
        return sorted(
            (r["candidates"], r["phenotypes"], r["phenotype_selection"],
             r["caller_shape"], r["queried_slice_total"])
            for r in rows if r["implementation"] == name
        )

    assert workload("current") == workload(prototype)

    # Within one cell every implementation must share the shape order, which is
    # what keeps the two comparable at all.
    per_cell = {}
    for r in rows:
        key = (r["candidates"], r["phenotypes"], r["phenotype_selection"],
               r["distribution"], r["mean_slice_length"])
        per_cell.setdefault(key, set()).add(r["measured_first"])
    for key, orders in per_cell.items():
        assert len(orders) == 1, f"cell {key} measured two shape orders: {orders}"
