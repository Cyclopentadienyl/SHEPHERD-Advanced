"""Backlog item 10 — the three scripts that make M1-M5 reproducible.

Those facts were established by reading artifacts in a review thread and have
existed as pasted text since. M1-M3 are what established that the frozen
evaluator cannot be the calibration oracle, and M4 bounds every number the
project reports on `val` — they are the facts most in need of being checkable,
and the ones a reviewer has had to take on trust.

**What these tests are for.** The institutional run happens on the deployment's
machine and cannot happen here. What can be checked here is that each script
emits the schema BACKLOG §5.2 fixes, **omits what §5.2 forbids**, and survives
the inputs an institutional workspace will actually present — an unreadable
checkpoint, a phenotype that reaches nothing, an artifact from another graph.

Module: tests/unit/test_evidence_scripts.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _sp_workspace(root: Path, *, n_phenotypes=3, n_diseases=10):
    root.mkdir(parents=True, exist_ok=True)
    (root / "num_nodes.json").write_text(
        json.dumps({"phenotype": n_phenotypes, "gene": 2, "disease": n_diseases}))
    # phenotype 0 reaches three diseases, phenotype 1 reaches one, phenotype 2
    # reaches none and therefore appears nowhere in the table.
    torch.save({
        "phenotype_idx": torch.tensor([0, 0, 0, 1, 0]),
        "target_idx": torch.tensor([0, 1, 2, 0, 5]),
        "target_type": torch.tensor([1, 1, 1, 1, 0]),
        "distance": torch.tensor([1, 2, 3, 4, 2], dtype=torch.int8),
    }, root / "sp.pt")
    return root


# ---------------------------------------------------------------------------
# M1-M3
# ---------------------------------------------------------------------------
def test_an_unreadable_checkpoint_is_recorded_rather_than_fatal(tmp_path):
    """M1 exists because the frozen evaluator's loader fails on this family, so a
    scan that stopped at the first unreadable file would destroy the evidence it
    was run to collect."""
    from scripts.audit_checkpoint_family import main

    ckdir = tmp_path / "ck"
    ckdir.mkdir()
    torch.save({"state_dict": {}, "logs": {"val_mrr": 0.5}}, ckdir / "model-1-0.5000.pt")
    (ckdir / "corrupt.pt").write_bytes(b"not a checkpoint")
    out = tmp_path / "m1.json"

    main(["--checkpoint-dir", str(ckdir), "--output", str(out)])
    report = json.loads(out.read_text())

    assert report["summary"]["n_checkpoints_found"] == 2
    assert report["summary"]["n_loaded"] == 1
    assert report["summary"]["n_unreadable"] == 1
    assert any(r["load_error"] for r in report["per_checkpoint"])


def test_the_checkpoint_evidence_records_no_absolute_path(tmp_path):
    """§5.2 forbids absolute paths, operator and host names. Basenames stay,
    because the filename is what M3 is about."""
    from scripts.audit_checkpoint_family import main

    ckdir = tmp_path / "deeply" / "nested" / "ck"
    ckdir.mkdir(parents=True)
    torch.save({"state_dict": {}, "logs": {"val_mrr": 0.5}}, ckdir / "model-1-0.5000.pt")
    out = tmp_path / "m1.json"

    main(["--checkpoint-dir", str(ckdir), "--output", str(out)])
    text = out.read_text()

    assert "deeply" not in text and str(tmp_path) not in text
    assert "model-1-0.5000.pt" in text


def test_the_filename_number_is_compared_with_the_logs_metric(tmp_path):
    """M3. A filename carries a rounded rendering, so the comparison is made at
    the precision the filename was written to."""
    from scripts.audit_checkpoint_family import main

    ckdir = tmp_path / "ck"
    ckdir.mkdir()
    torch.save({"state_dict": {}, "logs": {"val_mrr": 0.697543}}, ckdir / "model-45-0.6975.pt")
    torch.save({"state_dict": {}, "logs": {"val_mrr": 0.1}}, ckdir / "model-46-0.9999.pt")
    out = tmp_path / "m3.json"

    main(["--checkpoint-dir", str(ckdir), "--output", str(out)])
    report = json.loads(out.read_text())

    assert report["summary"]["filename_vs_logs"] == {
        "agree": 1, "disagree": 1, "uncomparable": 0}
    assert report["summary"]["logs_ranking_metric_counts"] == {"val_mrr": 2}


# ---------------------------------------------------------------------------
# M4
# ---------------------------------------------------------------------------
def test_the_overlap_evidence_records_sizes_and_no_identifiers(tmp_path):
    """§5.2 forbids patient ids, sample ids and per-disease lists; the claim is
    two set sizes and the size of their intersection."""
    from scripts.audit_split_overlap import main

    data_dir = tmp_path / "ws"
    data_dir.mkdir()
    (data_dir / "train_samples.json").write_text(json.dumps([
        {"patient_id": "SECRET-1", "phenotype_ids": [0], "disease_id": 0},
        {"patient_id": "SECRET-2", "phenotype_ids": [1], "disease_id": 1},
    ]))
    (data_dir / "val_samples.json").write_text(json.dumps([
        {"patient_id": "SECRET-3", "phenotype_ids": [0], "disease_id": 1},
    ]))
    out = tmp_path / "m4.json"

    main(["--data-dir", str(data_dir), "--output", str(out)])
    text = out.read_text()
    report = json.loads(text)

    assert report["counts"]["train_diseases"] == 2
    assert report["counts"]["val_diseases"] == 1
    assert report["counts"]["shared_diseases"] == 1
    assert report["overlap"]["shared_over_evaluation"] == 1.0
    assert "SECRET" not in text, "no patient identifier may reach the artifact"


def test_the_overlap_denominator_is_recorded_beside_the_ratio(tmp_path):
    """A percentage alone loses the denominator, and the denominator is half the
    claim: 100% of 7,970 and 100% of 2 are not the same finding."""
    from scripts.audit_split_overlap import main

    data_dir = tmp_path / "ws"
    data_dir.mkdir()
    (data_dir / "train_samples.json").write_text(json.dumps(
        [{"patient_id": "p", "phenotype_ids": [0], "disease_id": 0}]))
    (data_dir / "val_samples.json").write_text(json.dumps(
        [{"patient_id": "q", "phenotype_ids": [0], "disease_id": 0}]))
    out = tmp_path / "m4.json"

    main(["--data-dir", str(data_dir), "--output", str(out)])

    assert json.loads(out.read_text())["overlap"]["as_written"] == "1 of 1"


# ---------------------------------------------------------------------------
# M5
# ---------------------------------------------------------------------------
def test_phenotypes_reaching_no_disease_are_counted(tmp_path):
    """**The bias the recorded figure is most vulnerable to.**

    A phenotype that reaches nothing has no rows in the artifact at all, so a
    count taken over what appears there silently drops exactly the zeroes and
    reports a distribution shifted upward. Here phenotype 2 reaches nothing: the
    minimum must be 0 and the median must fall accordingly.
    """
    from scripts.audit_sp_reachability import main

    root = _sp_workspace(tmp_path / "ws")
    out = tmp_path / "m5.json"

    main(["--artifact", str(root / "sp.pt"), "--data-dir", str(root), "--output", str(out)])
    report = json.loads(out.read_text())
    spread = report["reachable_diseases_per_phenotype"]

    assert spread["n_phenotypes"] == 3, "every phenotype in the graph, not in the table"
    assert spread["min"]["diseases"] == 0.0
    assert spread["median"]["diseases"] == 1.0
    assert report["phenotype_coverage"] == {
        "in_graph": 3, "with_at_least_one_reachable_disease": 2, "reaching_none": 1}


def test_the_reachability_evidence_states_its_selection_rule_and_omits_rows(tmp_path):
    """§5.2 requires the selection rule and forbids per-phenotype rows. The rule
    here is that there is no selection — which is why the spread is reported."""
    from scripts.audit_sp_reachability import main

    root = _sp_workspace(tmp_path / "ws")
    out = tmp_path / "m5.json"

    main(["--artifact", str(root / "sp.pt"), "--data-dir", str(root), "--output", str(out)])
    report = json.loads(out.read_text())

    assert "no phenotype is selected as typical" in report["selection_rule"].lower()
    assert report["artifact_digest"]
    assert report["hop_bound_observed"] == 4

    # The property, not the key name: `reachable_diseases_per_phenotype` is the
    # *summary* and may legitimately say so. What must not be here is a vector
    # with one entry per phenotype.
    n_phenotypes = report["phenotype_coverage"]["in_graph"]

    def no_vector(node):
        if isinstance(node, list):
            assert len(node) < n_phenotypes, f"a per-phenotype vector reached the artifact: {node}"
        elif isinstance(node, dict):
            for value in node.values():
                no_vector(value)

    no_vector(report)


def test_an_artifact_from_a_different_graph_is_refused(tmp_path):
    """An artifact referencing a phenotype the workspace does not have describes a
    different graph, and no denominator here would be right."""
    from scripts.audit_sp_reachability import main

    root = _sp_workspace(tmp_path / "ws", n_phenotypes=1)
    out = tmp_path / "m5.json"

    with pytest.raises(SystemExit, match="different graphs"):
        main(["--artifact", str(root / "sp.pt"), "--data-dir", str(root),
              "--output", str(out)])


# ---------------------------------------------------------------------------
# Shared refusals
# ---------------------------------------------------------------------------
#: Each script's required arguments beside `--output`, so the shared refusal
#: below can be driven without a chain of conditionals in the test body.
_REQUIRED_ARGS = {
    "audit_checkpoint_family": ["--checkpoint-dir", "."],
    "audit_split_overlap": ["--data-dir", "."],
    "audit_sp_reachability": ["--data-dir", ".", "--artifact", "sp.pt"],
}


@pytest.mark.parametrize("script", sorted(_REQUIRED_ARGS))
def test_no_evidence_file_is_replaced_silently(tmp_path, script):
    """Evidence is cited by digest. A file quietly overwritten by a later run is
    not evidence — the same guard `benchmark_sp_lookup.py` already carries.

    Asserted for all three, because a refusal that only two of them carry is the
    kind of gap nobody notices until an artifact has already been lost.
    """
    import importlib

    module = importlib.import_module(f"scripts.{script}")
    out = tmp_path / "existing.json"
    out.write_text("{}")
    argv = ["--output", str(out)]
    for token in _REQUIRED_ARGS[script]:
        argv.append(str(tmp_path / token) if not token.startswith("--") else token)

    with pytest.raises(SystemExit, match="exists"):
        module.main(argv)
