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
    # The claim a synthetic run may not make.
    assert "pending institutional run" not in report["verdict"]


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
