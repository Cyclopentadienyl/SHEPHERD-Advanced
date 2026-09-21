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
import sys
from pathlib import Path

import pytest
import torch

from scripts.benchmark_sp_lookup import (
    build_artifact_lookup,
    build_index as benchmark_build_index,
    main,
)


#: The benchmark itself is Linux-only by design — see
#: `require_linux_memory_accounting`. Its *execution* tests are therefore skipped
#: elsewhere, with the reason stated rather than the suite going red for a
#: platform decision it is meant to record. The refusal contract below and the
#: pure workload helpers above stay unconditional: those are what a non-Linux
#: runner can and should still check.
linux_only = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="benchmark_sp_lookup is Linux-only by design (Linux RSS accounting)",
)


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

    columns, max_hops, lengths, disease_targets, provenance = build_artifact_lookup(
        artifact, 5
    )
    assert max_hops == 5

    phenotype, target, _, _ = columns
    by_phenotype = {}
    for p, t in zip(phenotype.tolist(), target.tolist()):
        by_phenotype.setdefault(p, []).append(t)

    assert set(by_phenotype) == set(expected)
    for p, targets in expected.items():
        assert sorted(by_phenotype[p]) == targets
    assert lengths == [len(by_phenotype[p]) for p in sorted(by_phenotype)]
    assert provenance["n_pairs"] == sum(len(v) for v in expected.values())
    assert len(provenance["sha256"]) == 64


def test_candidates_are_drawn_from_the_real_disease_target_space(tmp_path):
    """A candidate the artifact has no disease row for is not a realistic query."""
    artifact = tmp_path / "shortest_paths.pt"
    write_artifact(artifact)

    columns, _, _, disease_targets, _ = build_artifact_lookup(artifact, 5)

    _, target, target_type, _ = columns
    assert disease_targets, "fixture produced no disease targets"
    present = set(target[target_type == 1].tolist())
    assert set(disease_targets) == present


def test_missing_keys_are_fatal_rather_than_silently_partial(tmp_path):
    artifact = tmp_path / "shortest_paths.pt"
    torch.save({"phenotype_idx": torch.zeros(3, dtype=torch.int64)}, artifact)

    with pytest.raises(SystemExit, match="missing required keys"):
        build_artifact_lookup(artifact, 5)


@linux_only
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


@linux_only
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


@linux_only
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
# One implementation, and the bound it measures against
# =============================================================================
def test_the_benchmark_measures_against_the_declared_bound_not_the_observed_one(
    tmp_path, monkeypatch
):
    """**The defect §8.1 removed from the loader, found back in its consumer.**

    `max(distance)` bounds the hop limit from below only: a legitimate 5-hop
    table need contain no 5-hop row. The builder re-derived it from the column,
    so a table of 1-hop rows was read as `max_hops=1` and every miss scored
    against a sentinel of 2 where production uses 6. Measured before the fix:

        production pipeline : lookup.max_hops 5, miss 6.0
        benchmark           : lookup.max_hops 1, miss 2.0

    Two implementations agreeing with each other could never have caught it —
    both received the same corrupted bound.

    **The RSS reading is substituted, and nothing else is.** `build_index` takes
    one so its cost is reported, and `_rss_bytes` imports `resource`, which does
    not exist on Windows — so this test, which is about a hop bound and not
    about memory, failed at collection-adjacent import on any non-Linux host
    before the substitution. `@linux_only` would also have fixed it and would
    have cost the coverage everywhere else. The artifact reader, the builder and
    the query are the real ones; only the meter is stubbed, and it returns
    `None` rather than a number, so nothing here can be mistaken for a
    measurement.
    """
    import scripts.benchmark_sp_lookup as bench
    from src.inference.pipeline import DiagnosisPipeline, PipelineConfig
    from src.inference.sp_index import sp_mean_distances

    monkeypatch.setattr(
        bench, "_rss_bytes",
        lambda: {"peak_rss_bytes": None, "current_rss_bytes": None},
    )

    data_dir = tmp_path / "ws"
    data_dir.mkdir(parents=True, exist_ok=True)
    artifact = data_dir / "shortest_paths.pt"
    torch.save(
        {
            "phenotype_idx": torch.tensor([0, 1], dtype=torch.int64),
            "target_idx": torch.tensor([4, 5], dtype=torch.int64),
            "target_type": torch.tensor([1, 1], dtype=torch.int64),
            # a 5-hop table whose longest recorded path is 1
            "distance": torch.tensor([1, 1], dtype=torch.int8),
        },
        artifact,
    )
    (data_dir / "shortest_paths.meta.json").write_text(json.dumps({"max_hops": 5}))

    pipeline = DiagnosisPipeline.__new__(DiagnosisPipeline)
    pipeline.config = PipelineConfig()
    pipeline._sp_ready = False
    pipeline._sp_lookup = None
    pipeline._sp_max_hops = 5
    pipeline._sp_hop_bound_source = None
    pipeline._load_shortest_paths(data_dir)

    columns, resolved, _, _, _ = build_artifact_lookup(artifact, 99)
    table, _ = benchmark_build_index(columns, resolved)

    assert resolved == pipeline._sp_lookup.max_hops == 5, (
        "the benchmark resolved a different bound from the production loader"
    )
    served = sp_mean_distances(pipeline._sp_lookup, [0], [999], 1)[0]
    measured = sp_mean_distances(table, [0], [999], 1)[0]
    assert served.tolist() == measured.tolist() == [6.0]


@linux_only
def test_memory_attribution_flag_is_honest(tmp_path, monkeypatch):
    """`ru_maxrss` is a process high-water mark, so two tables built in one
    process cannot both be attributed. One is built per run now, by
    construction — the field stays because consumers read it, and because a
    guarantee is worth stating rather than inferring from a missing flag.
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
    assert bench.main(["--output", str(one)]) == 0
    single = json.loads(one.read_text())
    assert single["memory_attribution_isolated"] is True
    assert single["stage"] == "5a served primitive"
    assert [b["implementation"] for b in single["index_builds"]] == ["indexed"]
    assert single["implementations"] == ["indexed"]


@linux_only
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


@linux_only
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


@linux_only
def test_shape_order_alternates_across_cells(tmp_path, monkeypatch):
    """BLOCKING regression, kept because the two caller shapes are still compared.

    An earlier version chose the shape order with `(cell_index + position) % 2`,
    where `position` was an implementation's place in a rotated order, meaning
    to decorrelate the two. With two implementations the rotation moved
    `position` in lockstep with `cell_index`, so the sum was constant per
    implementation *identity*: on the real artifact one came out singleton-first
    in 60/60 rows and the other batched-first in 60/60. Any warm-up or cache
    asymmetry between the shapes then attached permanently to one of them.

    **The rotation is gone with the second implementation, and this test is
    not.** Singleton against batched is still a comparison this benchmark
    makes, and a fixed measurement order would still confound it. The keying is
    now on the cell alone, which is what it should always have been.
    """
    import scripts.benchmark_sp_lookup as bench

    monkeypatch.setattr(bench, "CANDIDATE_COUNTS", (10, 20))
    monkeypatch.setattr(bench, "PHENOTYPE_COUNTS", (1, 20))
    monkeypatch.setattr(bench, "SYNTHETIC_MEAN_SLICE_LENGTHS", (100,))
    monkeypatch.setattr(bench, "SYNTHETIC_DISTRIBUTIONS", ("representative",))
    monkeypatch.setattr(bench, "MIN_REPEATS", 1)
    monkeypatch.setattr(bench, "MAX_REPEATS", 1)
    monkeypatch.setattr(bench, "TARGET_MEASURE_SECONDS", 0.0)

    output = tmp_path / "order.json"
    assert bench.main(["--output", str(output)]) == 0
    rows = json.loads(output.read_text())["rows"]

    assert {r["implementation"] for r in rows} == {"indexed"}
    orders = {r["measured_first"] for r in rows}
    assert orders == {"singleton", "batched"}, (
        f"every row was measured {orders} first; shape order is not alternating"
    )

    # Within one cell both rows must share the shape order, which is what makes
    # the singleton and batched numbers in that cell comparable at all.
    per_cell = {}
    for r in rows:
        key = (r["candidates"], r["phenotypes"], r["phenotype_selection"],
               r["distribution"], r["mean_slice_length"])
        per_cell.setdefault(key, set()).add(r["measured_first"])
    for key, seen in per_cell.items():
        assert len(seen) == 1, f"cell {key} measured two shape orders: {seen}"

    # Both shapes still see the same workload in the same cell.
    def workload(shape):
        return sorted(
            (r["candidates"], r["phenotypes"], r["phenotype_selection"],
             r["queried_slice_total"])
            for r in rows if r["caller_shape"] == shape
        )

    assert workload("singleton") == workload("batched")


@linux_only
def test_an_existing_output_is_not_silently_replaced(tmp_path, monkeypatch):
    """A measurement artifact is cited by digest; a repeat run must not clobber it.

    This is a regression: the repeat of the B-0.4 artifact run was handed the
    first run's exact output paths, and the shell's `2>` redirection truncated
    the `time_*.txt` companions at launch. The JSONs survived only because they
    were already committed.
    """
    import scripts.benchmark_sp_lookup as bench

    for name, value in (
        ("CANDIDATE_COUNTS", (10,)), ("PHENOTYPE_COUNTS", (1,)),
        ("SYNTHETIC_MEAN_SLICE_LENGTHS", (50,)),
        ("SYNTHETIC_DISTRIBUTIONS", ("representative",)),
        ("MIN_REPEATS", 1), ("MAX_REPEATS", 1), ("TARGET_MEASURE_SECONDS", 0.0),
    ):
        monkeypatch.setattr(bench, name, value)

    output = tmp_path / "evidence.json"
    assert bench.main(["--output", str(output)]) == 0
    first = output.read_text()

    with pytest.raises(SystemExit):
        bench.main(["--output", str(output)])
    assert output.read_text() == first, "the refused run still modified the file"

    assert bench.main(["--output", str(output), "--overwrite"]) == 0
    assert output.exists()


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_a_non_linux_host_is_refused(monkeypatch, platform):
    """A **positive Linux gate**, not a Windows blocklist, and the difference is
    what this test exists for.

    The first version tested `sys.platform == "win32"`, copied from
    `src/retrieval/backends/cuvs_backend.py` — but only its first line. There the
    win32 test is a shortcut and the real decision is `import cuvs`, so a non-Linux
    POSIX host lands correctly on the `ImportError`. Copied without that second
    half, the shortcut became the whole check and `darwin` fell through into
    accounting that assumes Linux semantics: `/proc/self/status`, and `ru_maxrss`
    in kilobytes, which `_rss_bytes` multiplies by 1024.

    `darwin` is the case that would have passed the old gate and produced a number
    that looked valid. It is parametrized beside `win32` so neither can be fixed
    without the other.
    """
    import scripts.benchmark_sp_lookup as benchmark

    monkeypatch.setattr(benchmark.sys, "platform", platform)

    with pytest.raises(SystemExit) as raised:
        benchmark.main(["--output", "unused.json"])

    assert f"cannot run on {platform}" in str(raised.value)
    assert "by design" in str(raised.value)


def test_the_gate_lands_before_any_artifact_is_loaded(tmp_path, monkeypatch):
    """Ordering, asserted rather than described.

    `shortest_paths.pt` is gigabytes and takes minutes. A host that cannot account
    for memory must be told so before it pays that, not from inside a timing loop
    afterwards. Pointing `--artifact` at a path that does not exist is the proof:
    the platform refusal must arrive instead of a file error.
    """
    import scripts.benchmark_sp_lookup as benchmark

    monkeypatch.setattr(benchmark.sys, "platform", "darwin")

    with pytest.raises(SystemExit) as raised:
        benchmark.main([
            "--artifact", str(tmp_path / "does_not_exist.pt"),
            "--output", str(tmp_path / "out.json"),
        ])

    assert "cannot run on darwin" in str(raised.value)


def test_help_still_works_on_an_unsupported_host(monkeypatch, capsys):
    """The gate runs **after** argument parsing, so a reader on any platform can
    still see what the tool takes. Refusing `--help` bought nothing: it is the
    artifact load and the timed workload that are worth protecting, and argparse
    exits before either."""
    import scripts.benchmark_sp_lookup as benchmark

    monkeypatch.setattr(benchmark.sys, "platform", "win32")

    with pytest.raises(SystemExit) as raised:
        benchmark.main(["--help"])

    assert raised.value.code == 0
    assert "--artifact" in capsys.readouterr().out


def test_the_benchmark_runs_where_linux_memory_accounting_exists():
    """The other half: the gate must not fire on the platform it is written for.

    Linux ARM and Linux x86 share `/proc` and kilobyte `ru_maxrss`, so one gate
    covers both deployment architectures.
    """
    import scripts.benchmark_sp_lookup as benchmark

    if not sys.platform.startswith("linux"):
        pytest.skip("this assertion is about Linux hosts")

    benchmark.require_linux_memory_accounting()
