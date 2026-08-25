"""Backlog item 10 — the three scripts that make M1-M5 reproducible.

Those facts were established by reading artifacts in a review thread and have
existed as pasted text since. M1-M3 are what established that the frozen
evaluator cannot be the calibration oracle, and M4 bounds every number the
project reports on `val`.

**What these tests are for.** The institutional run happens on the deployment's
machine and cannot happen here. What can be checked here is that each script
emits the schema BACKLOG §5.2 fixes, **omits what §5.2 forbids**, refuses inputs
it cannot honestly summarise, and survives what an institutional workspace will
actually present — an unreadable checkpoint, a phenotype reaching nothing, an
artifact from another graph.

Module: tests/unit/test_evidence_scripts.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

#: A checkpoint's projection weights are `(hidden_dim, in_channels)`, so M2's
#: input width is the last dimension. 128 is the width M2 records.
IN_CHANNELS = 128
HIDDEN = 256


def _state_dict(in_channels: int = IN_CHANNELS) -> dict:
    return {
        f"feature_encoder.projections.{node_type}.weight":
            torch.zeros(HIDDEN, in_channels)
        for node_type in ("phenotype", "gene", "disease")
    }


def _checkpoint(tmp_path: Path, name: str, *, val_mrr=None, state=True, extra=None) -> Path:
    payload = {}
    if state:
        payload["state_dict"] = _state_dict()
    if val_mrr is not None:
        payload["logs"] = {"val_mrr": val_mrr}
    payload.update(extra or {})
    path = tmp_path / name
    torch.save(payload, path)
    return path


def _sp_workspace(root: Path, *, n_phenotypes=3, n_diseases=10, max_hops=5,
                  duplicate=False, bad_target=False):
    """An artifact plus the sidecar the producer writes beside it.

    Phenotype 0 reaches three diseases, phenotype 1 reaches one, phenotype 2
    reaches none and therefore appears nowhere in the table.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "num_nodes.json").write_text(
        json.dumps({"phenotype": n_phenotypes, "gene": 2, "disease": n_diseases}))

    pheno = [0, 0, 0, 1, 0]
    target = [0, 1, 2, 0, 5]
    ttype = [1, 1, 1, 1, 0]
    dist = [1, 2, 3, 4, 2]
    if duplicate:                      # the same (phenotype, disease) pair twice
        pheno, target, ttype, dist = pheno + [0], target + [1], ttype + [1], dist + [2]
    if bad_target:                     # a disease index the workspace does not have
        pheno, target, ttype, dist = pheno + [0], target + [n_diseases], ttype + [1], dist + [1]

    torch.save({
        "phenotype_idx": torch.tensor(pheno),
        "target_idx": torch.tensor(target),
        "target_type": torch.tensor(ttype),
        "distance": torch.tensor(dist, dtype=torch.int8),
    }, root / "sp.pt")
    (root / "sp.meta.json").write_text(json.dumps({
        "max_hops": max_hops,
        "num_phenotypes": n_phenotypes,
        "num_diseases": n_diseases,
    }))
    return root


def _run(script: str, argv: list):
    import importlib

    return importlib.import_module(f"scripts.{script}").main(argv)


# ---------------------------------------------------------------------------
# M1-M3 — the family scan
# ---------------------------------------------------------------------------
def test_an_unreadable_checkpoint_is_recorded_rather_than_fatal(tmp_path):
    """M1 exists because the frozen evaluator's loader fails on this family, so a
    scan that stopped at the first unreadable file would destroy the evidence it
    was run to collect."""
    ck = tmp_path / "ck"
    ck.mkdir()
    _checkpoint(ck, "model-1-0.5000.pt", val_mrr=0.5)
    (ck / "corrupt.pt").write_bytes(b"not a checkpoint")
    out = tmp_path / "m1.json"

    _run("audit_checkpoint_family", ["--checkpoint-dir", str(ck), "--output", str(out)])
    summary = json.loads(out.read_text())["summary"]

    assert summary["n_checkpoints_found"] == 2
    assert summary["n_loaded"] == 1
    assert summary["unreadable_filenames"] == ["corrupt.pt"]
    assert summary["load_error_categories"], "the failure category must be recorded"


def test_a_load_failure_records_a_category_and_not_its_message(tmp_path):
    """Torch and filesystem errors interpolate the path they failed on, and §5.2
    forbids absolute paths here. A stable class name says what a reader needs and
    cannot carry a path, a username or a mount point out with it."""
    ck = tmp_path / "secretdir"
    ck.mkdir()
    _checkpoint(ck, "ok-1-0.5000.pt", val_mrr=0.5)
    (ck / "corrupt.pt").write_bytes(b"nope")
    out = tmp_path / "m1.json"

    _run("audit_checkpoint_family", ["--checkpoint-dir", str(ck), "--output", str(out)])
    text = out.read_text()

    assert "secretdir" not in text and str(tmp_path) not in text
    for category in json.loads(text)["summary"]["load_error_categories"]:
        assert category.isidentifier(), f"{category!r} is a message, not a category"


@pytest.mark.parametrize("name,logged,expected", [
    ("model-45-0.6975.pt", 0.697543, "agree"),        # rounded rendering
    ("model-1-0.7000.pt", 0.7, "agree"),              # trailing zeroes preserved
    ("model-1-0.7000.pt", 0.70004, "agree"),          # inside the written precision
    ("model-1-0.7000.pt", 0.7005, "disagree"),        # outside it
    ("model-46-0.9999.pt", 0.1, "disagree"),
    ("model-45.pt", 0.5, "uncomparable"),             # an epoch, not a score
    ("last.pt", 0.5, "uncomparable"),
])
def test_the_filename_score_is_compared_at_the_precision_it_was_written(
        tmp_path, name, logged, expected):
    """M3. An earlier version parsed the token to `float` and recovered its
    precision from `repr`, so `0.7000` became one decimal place and the tolerance
    widened by three orders of magnitude — and `model-45.pt` was read as a score
    of 45."""
    ck = tmp_path / "ck"
    ck.mkdir()
    _checkpoint(ck, name, val_mrr=logged)
    out = tmp_path / "m3.json"

    _run("audit_checkpoint_family", ["--checkpoint-dir", str(ck), "--output", str(out)])
    comparison = json.loads(out.read_text())["summary"]["filename_vs_logs"]

    assert comparison[expected] == 1
    assert sum(comparison[k] for k in ("agree", "disagree", "uncomparable")) == 1


def test_the_input_width_is_read_from_the_projection_weights(tmp_path):
    """M2. Read from the weight rather than a config field, because a config
    records what was asked for and this records what was built."""
    ck = tmp_path / "ck"
    ck.mkdir()
    _checkpoint(ck, "model-1-0.5000.pt", val_mrr=0.5)
    _checkpoint(ck, "model-2-0.6000.pt", val_mrr=0.6)
    out = tmp_path / "m2.json"

    _run("audit_checkpoint_family", ["--checkpoint-dir", str(ck), "--output", str(out)])
    widths = json.loads(out.read_text())["summary"]["in_channels"]

    assert widths["value_counts"] == {str(IN_CHANNELS): 6}  # 3 node types x 2 files
    assert widths["n_loaded_exposing_projection_widths"] == 2
    assert widths["established"] is True


def test_a_family_exposing_no_projection_widths_says_so(tmp_path):
    """A family whose projection weights are named differently would otherwise
    produce an empty summary that reads like a normal result — M2 unestablished,
    reported as M2 measured."""
    ck = tmp_path / "ck"
    ck.mkdir()
    torch.save({"state_dict": {"other.weight": torch.zeros(4, 4)},
                "logs": {"val_mrr": 0.5}}, ck / "model-1-0.5000.pt")
    out = tmp_path / "m2.json"

    _run("audit_checkpoint_family", ["--checkpoint-dir", str(ck), "--output", str(out)])
    widths = json.loads(out.read_text())["summary"]["in_channels"]

    assert widths["established"] is False
    assert widths["n_loaded_without_projection_widths"] == 1


def test_an_empty_checkpoint_directory_is_refused(tmp_path):
    """Zero checkpoints reported as a finding would read as a fact about the
    family rather than about the directory that was pointed at."""
    ck = tmp_path / "ck"
    ck.mkdir()

    with pytest.raises(SystemExit, match="no \\*.pt files"):
        _run("audit_checkpoint_family",
             ["--checkpoint-dir", str(ck), "--output", str(tmp_path / "x.json")])


def test_a_family_where_nothing_loads_is_refused(tmp_path):
    """Partial failure is evidence; total failure says only that this reader could
    not open anything."""
    ck = tmp_path / "ck"
    ck.mkdir()
    (ck / "a.pt").write_bytes(b"nope")
    (ck / "b.pt").write_bytes(b"nope")

    with pytest.raises(SystemExit, match="could be\\s+loaded|none of the"):
        _run("audit_checkpoint_family",
             ["--checkpoint-dir", str(ck), "--output", str(tmp_path / "x.json")])


# ---------------------------------------------------------------------------
# M4 — split overlap
# ---------------------------------------------------------------------------
def _splits(data_dir: Path, train_ids, val_ids):
    data_dir.mkdir(parents=True, exist_ok=True)
    for split, ids in (("train", train_ids), ("val", val_ids)):
        (data_dir / f"{split}_samples.json").write_text(json.dumps([
            {"patient_id": f"SECRET-{split}-{i}", "phenotype_ids": [0], "disease_id": d}
            for i, d in enumerate(ids)
        ]))
    return data_dir


def test_the_overlap_evidence_records_sizes_and_no_identifiers(tmp_path):
    """§5.2 forbids patient ids, sample ids and per-disease lists; the claim is
    two set sizes and the size of their intersection."""
    data_dir = _splits(tmp_path / "ws", [0, 1], [1])
    out = tmp_path / "m4.json"

    _run("audit_split_overlap", ["--data-dir", str(data_dir), "--output", str(out)])
    text = out.read_text()
    report = json.loads(text)

    assert report["counts"]["train_diseases"] == 2
    assert report["counts"]["val_diseases"] == 1
    assert report["counts"]["shared_diseases"] == 1
    assert report["overlap"]["shared_over_evaluation"] == 1.0
    assert "SECRET" not in text, "no patient identifier may reach the artifact"


def test_the_overlap_denominator_is_recorded_beside_the_ratio(tmp_path):
    """A percentage alone loses the denominator, and the denominator is half the
    claim: 100% of 7,970 and 100% of 1 are not the same finding."""
    data_dir = _splits(tmp_path / "ws", [0], [0])
    out = tmp_path / "m4.json"

    _run("audit_split_overlap", ["--data-dir", str(data_dir), "--output", str(out)])

    assert json.loads(out.read_text())["overlap"]["as_written"] == "1 of 1"


# ---------------------------------------------------------------------------
# M5 — reachability
# ---------------------------------------------------------------------------
def _m5(tmp_path, **kwargs):
    root = _sp_workspace(tmp_path / "ws", **kwargs)
    out = tmp_path / "m5.json"
    _run("audit_sp_reachability",
         ["--artifact", str(root / "sp.pt"), "--data-dir", str(root), "--output", str(out)])
    return json.loads(out.read_text())


@pytest.mark.parametrize("empty,sidecar_key", [("disease", "num_diseases"),
                                               ("phenotype", "num_phenotypes")])
def test_an_empty_universe_is_refused_rather_than_reported(tmp_path, empty, sidecar_key):
    """M5 is "a phenotype reaches X% of diseases". Over an empty universe every such
    statement is vacuously true, and a file reporting median 0.0 against a null
    fraction reads as a finding about reachability rather than about the workspace.

    The sidecar is emptied to match, deliberately: leaving it at its original count
    makes the sidecar-versus-workspace cross-check fire first, and an earlier version
    of this test passed against a build with the guard deleted for exactly that
    reason. The assertion names a phrase only this guard produces, for the same
    reason — the other refusals here also mention diseases and phenotypes."""
    root = _sp_workspace(tmp_path / "ws")
    num_nodes = json.loads((root / "num_nodes.json").read_text())
    num_nodes[empty] = 0
    (root / "num_nodes.json").write_text(json.dumps(num_nodes))
    sidecar = json.loads((root / "sp.meta.json").read_text())
    sidecar[sidecar_key] = 0
    (root / "sp.meta.json").write_text(json.dumps(sidecar))

    with pytest.raises(SystemExit) as caught:
        _run("audit_sp_reachability",
             ["--artifact", str(root / "sp.pt"), "--data-dir", str(root),
              "--output", str(tmp_path / "m5.json")])
    assert "no distribution over an empty universe" in str(caught.value)
    assert f"0 {empty} nodes" in str(caught.value)
    assert not (tmp_path / "m5.json").exists()


def test_phenotypes_reaching_no_disease_are_counted(tmp_path):
    """**The bias the recorded figure is most vulnerable to.** A phenotype that
    reaches nothing has no rows at all, so a count over what appears in the table
    drops exactly the zeroes and reports a distribution shifted upward."""
    spread = _m5(tmp_path)["reachable_diseases_per_phenotype"]

    assert spread["n_phenotypes"] == 3, "every phenotype in the graph, not in the table"
    assert spread["min"]["diseases"] == 0.0
    assert spread["median"]["diseases"] == 1.0


def test_the_configured_hop_bound_is_read_from_the_sidecar(tmp_path):
    """The largest distance present is a property of the data; the bound is a
    property of what was built. An artifact configured to 5 hops with nothing at
    exactly 5 is still a 5-hop artifact, and every percentage in it must be read
    against 5."""
    report = _m5(tmp_path, max_hops=5)

    assert report["hop_bound_configured"] == 5
    assert report["hop_bound_observed"] == 4
    assert report["sidecar_digest"]


def test_a_missing_sidecar_is_refused(tmp_path):
    root = _sp_workspace(tmp_path / "ws")
    (root / "sp.meta.json").unlink()

    with pytest.raises(SystemExit, match="missing"):
        _run("audit_sp_reachability",
             ["--artifact", str(root / "sp.pt"), "--data-dir", str(root),
              "--output", str(tmp_path / "x.json")])


def test_duplicate_pairs_are_collapsed_rather_than_counted_twice(tmp_path):
    """The report calls these distinct diseases. An earlier version counted rows
    and justified it as M7 "enforced at load time" — untrue for this path, which
    calls `torch.load` directly and runs no such assertion."""
    plain = _m5(tmp_path / "a")
    duped = _m5(tmp_path / "b", duplicate=True)

    assert duped["rows"]["duplicate_pairs_collapsed"] == 1
    assert plain["rows"]["duplicate_pairs_collapsed"] == 0
    assert (duped["reachable_diseases_per_phenotype"]["max"]["diseases"]
            == plain["reachable_diseases_per_phenotype"]["max"]["diseases"])


def test_a_disease_index_outside_the_workspace_is_refused(tmp_path):
    """An artifact from another graph must be refused, not counted into a
    plausible percentage."""
    with pytest.raises(SystemExit, match="different graphs"):
        _m5(tmp_path, bad_target=True)


def test_a_sidecar_disagreeing_with_the_workspace_is_refused(tmp_path):
    """Two independent records of the same graph. A disagreement means the
    artifact was built from something other than what is being audited."""
    root = _sp_workspace(tmp_path / "ws", n_diseases=10)
    (root / "sp.meta.json").write_text(json.dumps(
        {"max_hops": 5, "num_phenotypes": 3, "num_diseases": 99}))

    with pytest.raises(SystemExit, match="different graphs"):
        _run("audit_sp_reachability",
             ["--artifact", str(root / "sp.pt"), "--data-dir", str(root),
              "--output", str(tmp_path / "x.json")])


def test_the_reachability_evidence_states_its_selection_rule(tmp_path):
    """§5.2 requires the selection rule. Here the rule is that there is no
    selection — which is why the spread is reported."""
    report = _m5(tmp_path)

    assert "no phenotype is selected as typical" in report["selection_rule"].lower()
    assert report["artifact_digest"]


# ---------------------------------------------------------------------------
# The publishable schema, pinned
# ---------------------------------------------------------------------------
#: Keys each artifact must carry. Required contract, **not** exact equality: a
#: later additive aggregate must not have to edit a test to be allowed.
_REQUIRED = {
    "m1": ("fact", "checkpoint_digests", "summary", "runtime", "deployment_relationship"),
    "m4": ("fact", "digests", "counts", "overlap", "deployment_relationship"),
    "m5": ("fact", "artifact_digest", "sidecar_digest", "hop_bound_configured",
           "reachable_diseases_per_phenotype", "selection_rule", "deployment_relationship"),
}

#: Substrings that must not appear anywhere in a published artifact, whatever
#: shape a future field takes.
_FORBIDDEN_SUBSTRINGS = ("patient_id", "sample_id", "SECRET", "/home/", "/tmp/", "\\\\Users\\\\")


def test_every_artifact_carries_its_required_contract(tmp_path):
    ck = tmp_path / "ck"
    ck.mkdir()
    _checkpoint(ck, "model-1-0.5000.pt", val_mrr=0.5)
    _run("audit_checkpoint_family",
         ["--checkpoint-dir", str(ck), "--output", str(tmp_path / "m1.json")])
    _splits(tmp_path / "ws4", [0], [0])
    _run("audit_split_overlap",
         ["--data-dir", str(tmp_path / "ws4"), "--output", str(tmp_path / "m4.json")])
    root = _sp_workspace(tmp_path / "ws5")
    _run("audit_sp_reachability",
         ["--artifact", str(root / "sp.pt"), "--data-dir", str(root),
          "--output", str(tmp_path / "m5.json")])

    for name, required in _REQUIRED.items():
        report = json.loads((tmp_path / f"{name}.json").read_text())
        missing = [key for key in required if key not in report]
        assert not missing, f"{name} is missing {missing}"
        assert report["deployment_relationship"] == "unstated"


def test_no_artifact_carries_a_forbidden_identifier_or_path(tmp_path):
    """Checked as substrings over the serialised file, so a future field cannot
    reintroduce one under a new key."""
    ck = tmp_path / "ck"
    ck.mkdir()
    _checkpoint(ck, "model-1-0.5000.pt", val_mrr=0.5)
    (ck / "corrupt.pt").write_bytes(b"nope")
    _run("audit_checkpoint_family",
         ["--checkpoint-dir", str(ck), "--output", str(tmp_path / "m1.json")])
    _splits(tmp_path / "ws4", [0, 1], [1])
    _run("audit_split_overlap",
         ["--data-dir", str(tmp_path / "ws4"), "--output", str(tmp_path / "m4.json")])
    root = _sp_workspace(tmp_path / "ws5")
    _run("audit_sp_reachability",
         ["--artifact", str(root / "sp.pt"), "--data-dir", str(root),
          "--output", str(tmp_path / "m5.json")])

    for name in _REQUIRED:
        text = (tmp_path / f"{name}.json").read_text()
        for forbidden in _FORBIDDEN_SUBSTRINGS:
            assert forbidden not in text, f"{name} carries {forbidden!r}"


def test_the_deployment_relationship_is_a_bounded_vocabulary(tmp_path):
    """§5.2 forbids operator and host names, and a schema that forbids them cannot
    then accept an arbitrary string — the first person in a hurry writes the
    hostname into it."""
    from src.utils.provenance import DEPLOYMENT_RELATIONSHIPS

    data_dir = _splits(tmp_path / "ws", [0], [0])
    out = tmp_path / "m4.json"

    with pytest.raises(SystemExit):
        _run("audit_split_overlap", ["--data-dir", str(data_dir), "--output", str(out),
                                     "--deployment-relationship", "lab-machine-hostname-42"])

    _run("audit_split_overlap", ["--data-dir", str(data_dir), "--output", str(out),
                                 "--deployment-relationship", "identical-sibling"])
    assert json.loads(out.read_text())["deployment_relationship"] == "identical-sibling"
    assert "identical-sibling" in DEPLOYMENT_RELATIONSHIPS


def test_all_three_scripts_share_one_vocabulary_object(tmp_path):
    """An institutional reader joins the three reports by machine. If M1 accepted a
    spelling M4 rejected, that join would break silently — and a per-script copy of
    the tuple is exactly how that happens. Asserted by identity, not equality: two
    equal-but-separate tuples are the state this test exists to forbid."""
    import importlib

    from src.utils.provenance import DEPLOYMENT_RELATIONSHIPS, UNSTATED_RELATIONSHIP

    modules = [importlib.import_module(f"scripts.{name}") for name in
               ("audit_checkpoint_family", "audit_split_overlap", "audit_sp_reachability")]
    for module in modules:
        assert module.DEPLOYMENT_RELATIONSHIPS is DEPLOYMENT_RELATIONSHIPS, module.__name__

    # ...and every parser agrees on what silence looks like, so three reports from an
    # operator who said nothing are joinable rather than three different blanks.
    for module, args in zip(modules, (["--checkpoint-dir", "."],
                                      ["--data-dir", "."],
                                      ["--data-dir", ".", "--artifact", "sp.pt"])):
        namespace = module.parse_args([*args, "--output", str(tmp_path / "unused.json")])
        assert namespace.deployment_relationship == UNSTATED_RELATIONSHIP, module.__name__


@pytest.mark.parametrize("script,args", [
    ("audit_checkpoint_family", ["--checkpoint-dir", "."]),
    ("audit_split_overlap", ["--data-dir", "."]),
    ("audit_sp_reachability", ["--data-dir", ".", "--artifact", "sp.pt"]),
])
def test_no_evidence_file_is_replaced_silently(tmp_path, script, args):
    """Evidence is cited by digest. A file quietly overwritten by a later run is
    not evidence — the guard `benchmark_sp_lookup.py` already carries. Asserted
    for all three rather than the two someone remembered."""
    out = tmp_path / "existing.json"
    out.write_text("{}")
    argv = ["--output", str(out)]
    for token in args:
        argv.append(token if token.startswith("--") else str(tmp_path / token))

    with pytest.raises(SystemExit, match="exists"):
        _run(script, argv)
