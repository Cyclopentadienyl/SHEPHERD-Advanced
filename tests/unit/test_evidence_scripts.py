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

import hashlib
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
    reaches none and therefore appears nowhere in the table. One row targets a
    gene, so the disease mask has something to exclude.

    Column dtypes match `compute_shortest_paths.py`: int64 indices and an int8
    distance.
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
        "phenotype_idx": torch.tensor(pheno, dtype=torch.int64),
        "target_idx": torch.tensor(target, dtype=torch.int64),
        "target_type": torch.tensor(ttype, dtype=torch.int64),
        "distance": torch.tensor(dist, dtype=torch.int8),
    }, root / "sp.pt")
    # The sidecar mirrors what `save_shortest_paths` writes, `num_pairs` included:
    # a fixture that omits a field the producer always writes would let an audit
    # that cannot read real output still pass here.
    (root / "sp.meta.json").write_text(json.dumps({
        "max_hops": max_hops,
        "num_pairs": len(pheno),
        "num_phenotypes": n_phenotypes,
        "num_genes": 2,
        "num_diseases": n_diseases,
        "kg_total_nodes": n_phenotypes + n_diseases + 2,
        "kg_total_edges": len(pheno),
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


@pytest.mark.parametrize("train,val", [([], [1]), ([0, 1], []), ([], [])])
def test_an_empty_split_is_refused_rather_than_reported_as_no_overlap(tmp_path, train, val):
    """With no evaluation diseases the ratio has no denominator; with no training
    diseases the overlap is zero for a reason that is about the workspace. Either
    would be cited later as "no contamination"."""
    data_dir = _splits(tmp_path / "ws", train, val)
    out = tmp_path / "m4.json"

    with pytest.raises(SystemExit, match="holds no samples"):
        _run("audit_split_overlap", ["--data-dir", str(data_dir), "--output", str(out)])
    assert not out.exists()


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


@pytest.mark.parametrize("field", ["num_pairs", "num_phenotypes", "num_diseases"])
def test_a_sidecar_that_does_not_bind_to_the_artifact_is_refused(tmp_path, field):
    """The digest names which sidecar was read; it does not prove the sidecar
    belongs to the tensors beside it. `num_pairs` is what ties it to this table —
    the node counts would match any artifact built from the same graph."""
    root = _sp_workspace(tmp_path / "ws")
    sidecar = json.loads((root / "sp.meta.json").read_text())
    sidecar[field] = sidecar[field] + 7
    (root / "sp.meta.json").write_text(json.dumps(sidecar))

    with pytest.raises(SystemExit, match="does not describe this artifact"):
        _run("audit_sp_reachability",
             ["--artifact", str(root / "sp.pt"), "--data-dir", str(root),
              "--output", str(tmp_path / "x.json")])
    assert not (tmp_path / "x.json").exists()


@pytest.mark.parametrize("field,value", [
    ("max_hops", "5"),          # a string where the producer writes an int
    ("num_pairs", None),        # absent
    ("max_hops", True),         # bool is an int in Python and valid JSON
    ("max_hops", 0),            # outside the [1, 127] the producer validates
    ("max_hops", 128),
])
def test_a_sidecar_outside_the_producers_schema_is_refused(tmp_path, field, value):
    """Checked before the values are used, because a sidecar that fails any of this
    did not come from `compute_shortest_paths.py` and describes nothing here."""
    root = _sp_workspace(tmp_path / "ws")
    sidecar = json.loads((root / "sp.meta.json").read_text())
    if value is None:
        del sidecar[field]
    else:
        sidecar[field] = value
    (root / "sp.meta.json").write_text(json.dumps(sidecar))

    with pytest.raises(SystemExit):
        _run("audit_sp_reachability",
             ["--artifact", str(root / "sp.pt"), "--data-dir", str(root),
              "--output", str(tmp_path / "x.json")])
    assert not (tmp_path / "x.json").exists()


def test_the_reachability_evidence_states_its_selection_rule(tmp_path):
    """§5.2 requires the selection rule. Here the rule is that there is no
    selection — which is why the spread is reported."""
    report = _m5(tmp_path)

    assert "no phenotype is selected as typical" in report["selection_rule"].lower()
    assert report["artifact_digest"]


def test_the_scan_is_independent_of_how_it_is_chunked(tmp_path):
    """The chunk size is a memory knob and nothing about the answer may depend on
    it. Run at one row per chunk against a table whose disease rows straddle every
    boundary, and compared against a single-chunk pass over the same table."""
    from scripts.audit_sp_reachability import scan_table

    root = _sp_workspace(tmp_path / "ws", duplicate=True)
    table = torch.load(root / "sp.pt", weights_only=True)
    n_rows = int(table["phenotype_idx"].numel())

    whole = scan_table(table, n_rows, 3, 10, 5, chunk_rows=n_rows + 5)
    for chunk in (1, 2, 3, n_rows - 1):
        part = scan_table(table, n_rows, 3, 10, 5, chunk_rows=chunk)
        assert torch.equal(part["counts"], whole["counts"]), chunk
        assert part["n_disease_rows"] == whole["n_disease_rows"], chunk
        assert part["n_distinct"] == whole["n_distinct"], chunk
        assert part["observed_hops"] == whole["observed_hops"], chunk


def test_the_scan_allocates_by_graph_size_and_not_by_row_count(tmp_path):
    """The property the rewrite exists for: the working set is a function of the
    graph, so a table with far more rows than cells costs no more memory. Asserted
    against the presence matrix's own ceiling rather than by measuring RSS, which
    would be a benchmark rather than a contract."""
    from scripts.audit_sp_reachability import MAX_PRESENCE_CELLS, scan_table

    root = _sp_workspace(tmp_path / "ws")
    table = torch.load(root / "sp.pt", weights_only=True)

    # 3 x 10 cells is far below the ceiling; a graph past it is refused rather
    # than attempted, because being killed part-way through a one-shot run is the
    # outcome this guard exists to prevent.
    assert 3 * 10 < MAX_PRESENCE_CELLS
    with pytest.raises(SystemExit, match="one-pass counter allocates"):
        scan_table(table, 5, MAX_PRESENCE_CELLS, 2, 5)


class _RecordingMatrix:
    """A presence matrix that remembers how large each slice reduced was."""

    def __init__(self, real):
        self._real = real
        self.reduced = []

    @property
    def shape(self):
        return self._real.shape

    def __getitem__(self, key):
        band = self._real[key]
        self.reduced.append(band.numel())
        return band


def test_the_row_reduction_never_materialises_a_whole_matrix_intermediate():
    """`presence.sum(dim=1)` on a bool matrix allocates eight bytes per cell —
    measured at +551 MiB over a 69 MiB matrix, and `dtype=torch.int64` does not
    avoid it. On the institutional graph that intermediate would be several
    gigabytes: larger than the matrix `MAX_PRESENCE_CELLS` exists to bound, and
    invisible to it.

    Pinned structurally rather than by measuring RSS, which would be a flaky
    benchmark. A whole-matrix reduction slices nothing and fails the coverage
    assertion; a band larger than the budget fails the first."""
    from scripts.audit_sp_reachability import REDUCTION_BUDGET_BYTES, _row_counts

    n_diseases = 24_000
    presence = torch.zeros((500, n_diseases), dtype=torch.bool)
    presence[::3, ::7] = True

    spy = _RecordingMatrix(presence)
    counts = _row_counts(spy, n_diseases)

    assert torch.equal(counts, presence.sum(dim=1)), "banding must not change the answer"
    assert spy.reduced, "the reduction was not banded at all"
    assert max(spy.reduced) * 8 <= REDUCTION_BUDGET_BYTES, spy.reduced
    assert sum(spy.reduced) == presence.numel(), "every row must be reduced exactly once"


@pytest.mark.parametrize("rows,n_diseases", [(1, 8), (7, 3), (129, 24_000)])
def test_the_row_reduction_agrees_with_a_whole_matrix_sum(rows, n_diseases):
    """Including shapes where the band does not divide the row count evenly."""
    from scripts.audit_sp_reachability import _row_counts

    presence = torch.zeros((rows, n_diseases), dtype=torch.bool)
    presence[::2, ::5] = True
    assert torch.equal(_row_counts(presence, n_diseases), presence.sum(dim=1))


def test_a_float_index_column_is_refused_rather_than_truncated(tmp_path):
    """0.5 converted to an index is phenotype 0, counted, and invisible in the
    result. The refusal has to come before the cast."""
    root = _sp_workspace(tmp_path / "ws")
    table = torch.load(root / "sp.pt", weights_only=True)
    table["target_idx"] = table["target_idx"].to(torch.float32) + 0.5
    torch.save(table, root / "sp.pt")

    with pytest.raises(SystemExit, match="would truncate"):
        _run("audit_sp_reachability",
             ["--artifact", str(root / "sp.pt"), "--data-dir", str(root),
              "--output", str(tmp_path / "x.json")])


@pytest.mark.parametrize("column,value,why", [
    ("target_type", -1, "an unrecognised code below the domain"),
    ("target_type", 2, "an unrecognised code above the domain"),
    ("distance", -1, "a negative hop count"),
    ("distance", 0, "a zero hop count"),
    ("distance", 6, "a distance beyond the sidecar's configured 5"),
])
def test_a_value_outside_the_producers_domain_is_refused(tmp_path, column, value, why):
    """dtype and shape do not constrain values.

    A `target_type` of 2 or -1 is not a disease, so the mask sends it to the gene
    branch and it *lowers a reachability figure invisibly* — the failure mode this
    whole file exists to prevent. A `distance` of 0 or -1 is not a hop count, and
    comparing only the observed maximum against the configured bound admits both.
    """
    root = _sp_workspace(tmp_path / "ws", max_hops=5)
    table = torch.load(root / "sp.pt", weights_only=True)
    table[column] = table[column].clone()
    table[column][0] = value
    torch.save(table, root / "sp.pt")

    with pytest.raises(SystemExit) as caught:
        _run("audit_sp_reachability",
             ["--artifact", str(root / "sp.pt"), "--data-dir", str(root),
              "--output", str(tmp_path / "x.json")])
    assert column in str(caught.value), why
    assert not (tmp_path / "x.json").exists()


def test_a_pt_file_that_is_not_a_table_is_refused_rather_than_traced(tmp_path):
    """`"phenotype_idx" not in <tensor>` is a `Tensor.__contains__` and raises a
    RuntimeError about container types — a stack trace where the operator needs to
    be told they pointed at the wrong `.pt`."""
    root = _sp_workspace(tmp_path / "ws")
    torch.save(torch.zeros(4), root / "sp.pt")

    with pytest.raises(SystemExit, match="not a table of named columns"):
        _run("audit_sp_reachability",
             ["--artifact", str(root / "sp.pt"), "--data-dir", str(root),
              "--output", str(tmp_path / "x.json")])


@pytest.mark.parametrize("column,replacement", [
    ("phenotype_idx", "two_dimensional"),
    ("target_type", "short"),
    ("distance", "not_a_tensor"),
])
def test_a_column_outside_the_producers_shape_is_refused(tmp_path, column, replacement):
    """The scan pairs row i of one column with row i of another. Columns that are
    ragged, multi-dimensional or not tensors make that pairing arbitrary."""
    root = _sp_workspace(tmp_path / "ws")
    table = torch.load(root / "sp.pt", weights_only=True)
    if replacement == "two_dimensional":
        table[column] = table[column].reshape(-1, 1)
    elif replacement == "short":
        table[column] = table[column][:-1]
    else:
        table[column] = table[column].tolist()
    torch.save(table, root / "sp.pt")

    with pytest.raises(SystemExit):
        _run("audit_sp_reachability",
             ["--artifact", str(root / "sp.pt"), "--data-dir", str(root),
              "--output", str(tmp_path / "x.json")])
    assert not (tmp_path / "x.json").exists()


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

#: Which backlog fact each report claims to be. Pinned by value: three files whose
#: `fact` keys merely exist are not distinguishable from each other, and the whole
#: point of the field is that a reader can tell which claim they are holding.
_FACT = {"m1": "M1-M3", "m4": "M4", "m5": "M5"}

#: Substrings that must not appear anywhere in a published artifact, whatever
#: shape a future field takes.
#:
#: **A regression sentinel, not a proof.** Passing this does not establish that no
#: absolute path or identifier can ever reach an artifact — it establishes that the
#: ones these scripts were shown to leak, and the categories §5.2 names, do not
#: reach one now. The refusals and the omissions in the scripts are what make the
#: property true; this is defence in depth behind them.
_FORBIDDEN_SUBSTRINGS = ("patient_id", "sample_id", "SECRET", "/home/", "/tmp/", "\\\\Users\\\\")

#: Nested fields whose **type** is part of the contract, as `(path, predicate)`.
#: Types and bounds only — no exact object equality, and no rule against a future
#: additive aggregate, which would make the schema harder to extend than to freeze.
_NESTED = {
    "m1": [
        (("summary", "n_loaded"), lambda v: _is_int(v) and v >= 1),
        (("summary", "in_channels", "established"), lambda v: isinstance(v, bool)),
        (("summary", "in_channels", "value_counts"), lambda v: isinstance(v, dict)),
        (("summary", "filename_vs_logs", "agree"), lambda v: _is_int(v)),
        (("runtime", "torch"), lambda v: isinstance(v, str) and v),
    ],
    "m4": [
        (("counts", "train_diseases"), lambda v: _is_int(v) and v >= 1),
        (("counts", "val_diseases"), lambda v: _is_int(v) and v >= 1),
        (("counts", "shared_diseases"), lambda v: _is_int(v) and v >= 0),
        (("overlap", "shared_over_evaluation"), lambda v: type(v) is float and 0 <= v <= 1),
        (("overlap", "as_written"), lambda v: isinstance(v, str) and " of " in v),
    ],
    "m5": [
        (("hop_bound_configured",), lambda v: _is_int(v) and 1 <= v <= 127),
        (("hop_bound_observed",), lambda v: v is None or _is_int(v)),
        (("reachable_diseases_per_phenotype", "denominator_diseases"),
         lambda v: _is_int(v) and v >= 1),
        (("rows", "distinct_phenotype_disease_pairs"), lambda v: _is_int(v) and v >= 0),
        (("rows", "duplicate_pairs_collapsed"), lambda v: _is_int(v) and v >= 0),
        (("phenotype_coverage", "in_graph"), lambda v: _is_int(v) and v >= 1),
    ],
}

#: Where each report records a SHA-256 of a file, as `(path-to-the-digest, filename)`.
#: The expected value is recomputed in the test from the bytes on disk through a
#: different hashlib entry point than `file_sha256` uses, so a digest of the wrong
#: file — or of nothing — cannot pass.
_DIGEST_SITES = {
    "m4": [(("digests", "train_samples"), "train_samples.json"),
           (("digests", "val_samples"), "val_samples.json")],
    "m5": [(("artifact_digest",), "sp.pt"),
           (("sidecar_digest",), "sp.meta.json")],
}


def _is_int(value) -> bool:
    """A JSON integer, and not a JSON `true`.

    `isinstance(True, int)` is True, and `true` is valid JSON, so a boolean where
    the schema promises a count would pass an `isinstance` check and read as 1.
    """
    return type(value) is int


def _at(report, path):
    for key in path:
        report = report[key]
    return report


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
        assert report["fact"] == _FACT[name], f"{name} claims {report['fact']!r}"
        assert report["deployment_relationship"] == "unstated"

        for path, ok in _NESTED[name]:
            assert ok(_at(report, path)), f"{name}{list(path)} = {_at(report, path)!r}"

    # Digests are what the reports are cited by, so each is checked against the
    # bytes it claims to describe rather than merely for being a hex string.
    sources = {"m4": tmp_path / "ws4", "m5": root}
    for name, sites in _DIGEST_SITES.items():
        report = json.loads((tmp_path / f"{name}.json").read_text())
        for path, filename in sites:
            expected = hashlib.sha256((sources[name] / filename).read_bytes()).hexdigest()
            assert _at(report, path) == expected, f"{name}{list(path)} is not {filename}"

    m1 = json.loads((tmp_path / "m1.json").read_text())
    assert m1["checkpoint_digests"] == {
        "model-1-0.5000.pt": hashlib.sha256((ck / "model-1-0.5000.pt").read_bytes()).hexdigest()
    }


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


def test_the_three_scripts_offer_one_vocabulary(tmp_path):
    """An institutional reader reads all three reports for one machine. If M1
    accepted a spelling M4 rejected, or the two disagreed on what silence looks
    like, the claims could not be compared.

    Asserted through the parsers — the published behaviour — rather than through
    object identity. A shared constant is how this is implemented today, and the
    equality check below catches a re-duplication the moment it drifts, but a
    faithful re-export or an enum wrapper would be a legitimate refactor and this
    contract is about what the scripts accept, not about how they store it."""
    import importlib

    from src.utils.provenance import DEPLOYMENT_RELATIONSHIPS, UNSTATED_RELATIONSHIP

    scripts = {
        "audit_checkpoint_family": ["--checkpoint-dir", "."],
        "audit_split_overlap": ["--data-dir", "."],
        "audit_sp_reachability": ["--data-dir", ".", "--artifact", "sp.pt"],
    }
    out = ["--output", str(tmp_path / "unused.json")]

    for name, args in scripts.items():
        module = importlib.import_module(f"scripts.{name}")

        # Every valid value is accepted by every script...
        for value in DEPLOYMENT_RELATIONSHIPS:
            chosen = module.parse_args([*args, *out, "--deployment-relationship", value])
            assert chosen.deployment_relationship == value, name

        # ...an invalid one by none...
        with pytest.raises(SystemExit):
            module.parse_args([*args, *out, "--deployment-relationship", "lab-host-42"])

        # ...and silence means the same thing in all three.
        assert module.parse_args([*args, *out]).deployment_relationship == UNSTATED_RELATIONSHIP

        assert tuple(module.DEPLOYMENT_RELATIONSHIPS) == tuple(DEPLOYMENT_RELATIONSHIPS), name


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
