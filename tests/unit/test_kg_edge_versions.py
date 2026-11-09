"""
KG edge sets: a part of a checkpoint's recorded edge split, kept as a registry version.
======================================================================================
Research comparison §5 Stage 4 (owner's O7; the plan review's F2, §5 items 1-2,
E3). A ``kg_edges`` version is imported through the registry's one
preview/commit flow from a checkpoint's model record, and read back by its one
reader. What is pinned here, on a base and a disease child trained for real
on the demo workspace (`trained_disease_child`):

- **the vocabulary**: one name per kind, shared with the measurement
  manifest's ``cohort_kind``; the roles are the split's parts;
- **every role imports** (``train``, ``val``, ``test``, none refused for not
  being test) as schema 2 with its kind, binding the graph, the dataset
  record, the view, the split, the message-passing graph (the train part,
  saying when the measured edges are in it) and a pairs file of the exact
  bytes made, read back through `read_kg_edge_pairs`;
- **a ``val`` set's pairs are base validation's**, exactly, and its counts the
  ones the base's own training recorded;
- **the flow**: the preview id is the commit id (through the command line,
  the printed command run as printed); a change of seed, view or graph
  between preview and commit is refused, each by the part of the manifest it
  changes; a copy or link of the same checkpoint makes the same version (the
  checkpoint is named by its digest, never its file name); a taken label is
  refused; a checkpoint whose record does not rebuild, contradicts itself or
  is of another graph is refused before anything is written;
- **the reader**: a tampered, truncated or missing pairs file is refused on
  read; a hand-edited manifest is refused naming the field; the pairs each
  batch kept are recorded and checked;
- **every listing** shows a KG edge set's kind, role and counts, never an edge
  or a case id; a task protocol refuses it with its reason; the Diagnosis
  route refuses it as a reference cohort, and its card lists it marked,
  never chosen by default; the overlap audit refuses it with one message;
  the supplied import form never offers its role;
- **a supplied version** still reads in schema 1, and a schema-2 manifest of
  kind supplied, which no revision writes, is refused.

Module: tests/unit/test_kg_edge_versions.py
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import shlex
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.evaluation import cohort_registry as cr  # noqa: E402
from src.evaluation.cohort_registry import (  # noqa: E402
    CohortVersionRefused,
    commit_import,
    encode_json,
    prepare_kg_edges_import,
    read_kg_edge_pairs,
    resolve_version,
)

ROLES = ("train", "val", "test")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tree(root: Path) -> dict:
    return {str(p.relative_to(root)): (p.read_bytes() if p.is_file() else None)
            for p in sorted(root.rglob("*"))}


@pytest.fixture
def registry(tmp_path):
    path = tmp_path / "registry"
    path.mkdir()
    return path


@pytest.fixture(scope="module")
def trained(trained_disease_child):
    """The session's demo base and disease child, read only."""
    return trained_disease_child


def _payload(checkpoint: Path) -> dict:
    return torch.load(checkpoint, weights_only=True, map_location="cpu")


def _bound(data_dir: Path) -> dict:
    from src.kg.artifacts import read_split_manifest, verify_graph_reads
    from src.kg.storage.file_storage import read_graph_artifacts

    manifest = read_split_manifest(data_dir)
    return dict(verify_graph_reads(manifest, read_graph_artifacts(data_dir).reads))


def _view_and_split(data_dir: Path, rule: dict, seed: int):
    from src.kg.graph_view import build_graph_view, split_facts
    from src.kg.storage.file_storage import read_graph_artifacts

    view = build_graph_view(read_graph_artifacts(data_dir).graph_data, rule=dict(rule))
    return view, split_facts(view, seed)


def _rewritten(checkpoint: Path, out: Path, edit) -> Path:
    """A copy of `checkpoint` whose payload `edit` changed (in place)."""
    payload = _payload(checkpoint)
    edit(payload)
    torch.save(payload, out)
    return out


def _import(registry: Path, data_dir: Path, checkpoint: Path, role: str, label: str = None):
    plan = prepare_kg_edges_import(registry, data_dir, checkpoint, label or f"kg-{role}", role)
    return plan, commit_import(plan, registry, plan.preview_id)


# ---------------------------------------------------------------------------
# The vocabulary
# ---------------------------------------------------------------------------
def test_one_name_per_kind_shared_with_the_measurement_manifest():
    from src.evaluation import protocols
    from src.evaluation.cohort import COHORT_KINDS
    from src.kg.graph_view import SPLIT_NAMES

    assert len(set(COHORT_KINDS)) == len(COHORT_KINDS)
    # Every kind is a registered version's since Stage 4b kept generated
    # splits in the registry (tests/unit/test_generated_versions.py).
    assert set(cr.VERSION_KINDS) == set(COHORT_KINDS)
    assert (cr.SUPPLIED, cr.KG_EDGES) == (protocols.SUPPLIED, protocols.KG_EDGES)
    assert cr.KG_EDGES_ROLES == SPLIT_NAMES
    assert cr.VERSION_SCHEMAS == (1, 2) and cr.VERSION_SCHEMA_VERSION == 1


# ---------------------------------------------------------------------------
# Every role imports, and reads back
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("role", ROLES)
def test_every_role_imports_as_recorded_and_binds_what_it_was_made_from(trained, registry,
                                                                        role):
    from src.config.recipe import BASE_FIXED
    from src.evaluation.cohort_registry import verify_cohort_binding
    from src.models.bundle.lp_head import LP_PAIR_RULE, fixed_lp_pairs, lp_pairs_table

    data_dir, base = trained["data_dir"], trained["base"]
    plan, version = _import(registry, data_dir, base, role)
    manifest = version.manifest
    roles = _payload(base)["data_roles"]

    assert (version.kind, manifest["role"], manifest["schema_version"]) == ("kg_edges", role, 2)
    assert plan.kind == "kg_edges" and plan.source_bytes is None and plan.samples_bytes is None
    assert manifest["graph"] == {r: _bound(data_dir)[r] for r in manifest["graph"]}
    verify_cohort_binding(manifest, _bound(data_dir), manifest["dataset_record"]["sha256"])
    assert manifest["graph_view"] == roles["graph_view"]
    assert manifest["split"] == roles["split"]
    edges = roles["split"]["edges"]
    counts = manifest["counts"]
    assert counts["positive_edges"] == edges[role]
    assert counts["pairs"] + counts["edges_without_negative"] == edges[role]
    assert counts["message_passing_edges"] == edges["train"]
    assert set(manifest["count_units"]) == set(counts)
    assert manifest["message_passing"] == {"part": "train", "edges": edges["train"],
                                           "includes_measured_edges": role == "train"}
    pairs = manifest["pairs"]
    assert (pairs["rule"], pairs["part"], pairs["seed"], pairs["batch_size"]) == (
        LP_PAIR_RULE, role, roles["split"]["seed"], BASE_FIXED["validation_batch_size"])
    assert pairs["batches"] == -(-edges[role] // pairs["batch_size"])
    assert len(pairs["batch_pairs"]) == pairs["batches"]
    assert sum(pairs["batch_pairs"]) == pairs["rows"] == counts["pairs"]
    assert manifest["taken_from"] == {"checkpoint_sha256": _sha(base.read_bytes()),
                                      "record_fields": ["data_roles.graph_view",
                                                        "data_roles.split"]}

    # The version holds its manifest and its pairs, nothing else; no source is
    # kept for a KG edge set.
    assert sorted(p.name for p in version.directory.iterdir()) == ["cohort.json", "pairs.bin"]
    assert not (registry / "sources").exists() and version.samples_path is None
    data = (version.directory / "pairs.bin").read_bytes()
    assert _sha(data) == pairs["sha256"] and len(data) == 32 * counts["pairs"]

    # The pairs read back are the ones the rule makes, in its order.
    view, split = _view_and_split(data_dir, roles["graph_view"]["rule"], roles["split"]["seed"])
    expected = fixed_lp_pairs(view.edge_index(split.mask(role)), view.rel[split.mask(role)],
                              seed=split.seed, batch_size=BASE_FIXED["validation_batch_size"])
    read = read_kg_edge_pairs(version)
    assert data == lp_pairs_table(expected.head, expected.relation, expected.tail,
                                  expected.negative_tail)
    for column in ("head", "relation", "tail", "negative_tail"):
        assert torch.equal(getattr(read.pairs, column), getattr(expected, column)), column
    assert read.pairs.counts == expected.counts and read.pairs.digest == expected.digest
    assert read.pairs.batch_pairs == expected.batch_pairs == tuple(pairs["batch_pairs"])
    assert read.pairs.digest == pairs["lp_pairs_digest"]
    assert read.identity.sha256 == pairs["sha256"]


def test_a_child_s_record_names_its_base_s_split_and_gives_the_same_edge_set(trained,
                                                                            registry):
    """A finetune's record carries its base's view and split, so its test part
    is the base's: the same pairs, whichever of the two records it is taken from."""
    _, from_base = _import(registry, trained["data_dir"], trained["base"], "test", "kg-b")
    _, from_child = _import(registry, trained["data_dir"], trained["disease"], "test", "kg-c")
    assert from_base.manifest["pairs"]["sha256"] == from_child.manifest["pairs"]["sha256"]
    assert from_base.manifest["split"] == from_child.manifest["split"]
    assert from_base.manifest["taken_from"] != from_child.manifest["taken_from"]


def test_val_pairs_are_base_validation_s_pairs_exactly(trained, registry):
    """The ``val`` set's pairs are the ones `BaseLPTask` builds as training
    builds it (the split's val edges, its seed, `BASE_FIXED`'s batch size),
    and its counts the ones the base's own validation recorded."""
    from src.config.recipe import BASE_FIXED
    from src.training.lp_task import BaseLPTask

    data_dir, base = trained["data_dir"], trained["base"]
    _, version = _import(registry, data_dir, base, "val")
    payload = _payload(base)
    roles = payload["data_roles"]
    view, split = _view_and_split(data_dir, roles["graph_view"]["rule"], roles["split"]["seed"])
    train, val = split.mask("train"), split.mask("val")
    task = BaseLPTask(num_nodes=view.num_nodes, train_edges=view.edge_index(train),
                      train_relations=view.rel[train], val_edges=view.edge_index(val),
                      val_relations=view.rel[val], seed=split.seed,
                      val_batch_size=BASE_FIXED["validation_batch_size"],
                      node_type=view.node_type)
    pairs = read_kg_edge_pairs(version).pairs
    for column in ("head", "relation", "tail", "negative_tail"):
        assert torch.equal(getattr(pairs, column), getattr(task.val_pairs, column)), column
    assert pairs.digest == task.val_pairs.digest
    assert pairs.batch_pairs == task.val_pairs.batch_pairs
    logs = payload["logs"]
    assert (logs["val_val_edges"], logs["val_val_without_negative"], logs["val_lp_positives"]) \
        == (version.manifest["counts"]["positive_edges"],
            version.manifest["counts"]["edges_without_negative"],
            version.manifest["counts"]["pairs"])


# ---------------------------------------------------------------------------
# The command line: preview id is the commit id
# ---------------------------------------------------------------------------
def _cli(registry, data_dir, checkpoint, role="test", label="kg-test", *extra):
    return ["--kind", "kg_edges", "--registry", str(registry), "--data-dir", str(data_dir),
            "--from-checkpoint", str(checkpoint), "--label", label, "--role", role, *extra]


def test_the_cli_previews_then_commits_under_the_id_it_showed(trained, registry, capsys):
    from scripts import import_cohort

    data_dir, base = trained["data_dir"], trained["base"]
    assert import_cohort.main(_cli(registry, data_dir, base)) == 0
    out = capsys.readouterr().out
    assert list(registry.iterdir()) == []
    assert out.startswith("KG edge set preview: label kg-test, role test")
    assert "cohort" not in out.split("To publish")[0].lower()
    assert "the measured edges are not in it" in out
    preview = re.search(r"^Preview id: ([0-9a-f]{16})$", out, re.MULTILINE).group(1)
    command = out.strip().splitlines()[-1].strip()
    argv = shlex.split(command)
    assert argv[:2] == ["python", "scripts/import_cohort.py"] and argv[-1] == preview

    assert import_cohort.main(argv[2:]) == 0
    committed = capsys.readouterr().out
    version = resolve_version(registry, "kg-test")
    counts = version.manifest["counts"]
    assert committed.startswith(f"Committed KG edge set kg-test (test): {counts['pairs']} pairs "
                                f"of {counts['positive_edges']} positive edges")
    assert version.identity.sha256 in committed


def test_the_train_preview_says_the_measured_edges_are_in_the_message_passing_graph(
        trained, registry, capsys):
    from scripts import import_cohort

    assert import_cohort.main(_cli(registry, trained["data_dir"], trained["base"], "train",
                                   "kg-train")) == 0
    out = capsys.readouterr().out
    assert "which include the" in out and "role train: nothing is removed" in out


@pytest.mark.parametrize("argv, said", [
    (["--source", "cases.json"], "--source applies to --kind supplied only"),
    (["--partial-phenotypes", "keep"], "--partial-phenotypes applies to --kind supplied only"),
])
def test_the_cli_refuses_the_other_kind_s_arguments(trained, registry, argv, said, capsys):
    from scripts import import_cohort

    with pytest.raises(SystemExit) as exited:
        import_cohort.parse_args(_cli(registry, trained["data_dir"], trained["base"]) + argv)
    assert exited.value.code == 2 and said in capsys.readouterr().err


def test_the_cli_needs_each_kind_s_input(registry, capsys):
    from scripts import import_cohort

    common = ["--registry", str(registry), "--data-dir", "ws", "--label", "x", "--role", "test"]
    for argv, said in ((["--kind", "kg_edges", *common], "--from-checkpoint is required"),
                       (common, "--source is required for --kind supplied"),
                       ([*common, "--source", "s", "--from-checkpoint", "c"],
                        "--from-checkpoint applies to --kind kg_edges only")):
        with pytest.raises(SystemExit) as exited:
            import_cohort.parse_args(argv)
        assert exited.value.code == 2 and said in capsys.readouterr().err


def test_a_role_that_is_not_a_part_is_refused_with_nothing_written(trained, registry, capsys):
    from scripts import import_cohort

    assert import_cohort.main(_cli(registry, trained["data_dir"], trained["base"],
                                   "holdout")) == 1
    err = capsys.readouterr().err
    assert "a KG edge set's role is the part of the edge split it holds, one of train, val, " \
           "test; got 'holdout'" in err and "Nothing was written" in err
    assert list(registry.iterdir()) == []


# ---------------------------------------------------------------------------
# Changed inputs, a taken label, a record that does not rebuild
# ---------------------------------------------------------------------------
def _other_seed(payload, data_dir):
    roles = payload["data_roles"]
    _, split = _view_and_split(data_dir, roles["graph_view"]["rule"], roles["split"]["seed"] + 1)
    roles["split"] = split.record()


def _other_view(payload, data_dir):
    """The record of a model of another view (one relation and its twin
    excluded), consistent in every place it names the view."""
    roles = payload["data_roles"]
    rule = {**roles["graph_view"]["rule"],
            "exclude_relations": ["gene|human_mouse_ortholog|mouse_gene"]}
    view, split = _view_and_split(data_dir, rule, roles["split"]["seed"])
    roles["graph_view"] = {"digest": view.digest, "rule": dict(view.rule)}
    roles["split"] = split.record()
    payload["graph_view"] = view.record()
    payload["model_spec"]["graph_view"]["exclude_relations"] = list(rule["exclude_relations"])


@pytest.mark.parametrize("change, parts", [
    (_other_seed, ("split", "pairs")),
    (_other_view, ("graph_view", "split", "pairs")),
])
def test_a_change_of_seed_or_view_after_the_preview_is_refused(trained, registry, tmp_path,
                                                              change, parts):
    """The seed or the view is bound by the preview id itself, not only by the
    checkpoint's digest: the manifest parts the change touches differ, and
    with the checkpoint's reference held equal the ids still differ."""
    data_dir, base = trained["data_dir"], trained["base"]
    shown = prepare_kg_edges_import(registry, data_dir, base, "kg-test", "test")
    other = _rewritten(base, tmp_path / "other.pt", lambda p: change(p, data_dir))
    now = prepare_kg_edges_import(registry, data_dir, other, "kg-test", "test")
    for part in parts:
        assert now.manifest[part] != shown.manifest[part], part
    assert (now.manifest["graph"], now.manifest["dataset_record"]) == (
        shown.manifest["graph"], shown.manifest["dataset_record"])
    same_checkpoint = {**now.manifest, "taken_from": shown.manifest["taken_from"]}
    assert cr.preview_id(same_checkpoint) != shown.preview_id
    assert now.preview_id != shown.preview_id
    with pytest.raises(CohortVersionRefused, match="inputs changed since the preview"):
        commit_import(now, registry, shown.preview_id)
    assert list(registry.iterdir()) == []


def test_a_copy_or_link_of_the_same_checkpoint_makes_the_same_version(trained, registry,
                                                                     tmp_path):
    """Identity by content (research comparison §4.8, E3): the checkpoint is
    named by its digest, and its file name or place is a location, so a copy
    under another name, or a link to it, previews the same id and commits
    under it."""
    import shutil

    data_dir, base = trained["data_dir"], trained["base"]
    shown = prepare_kg_edges_import(registry, data_dir, base, "kg-test", "test")
    copied = tmp_path / "renamed-copy.pt"
    shutil.copyfile(base, copied)
    linked = tmp_path / "linked.pt"
    linked.symlink_to(base)
    for checkpoint in (copied, linked):
        again = prepare_kg_edges_import(registry, data_dir, checkpoint, "kg-test", "test")
        assert again.manifest == shown.manifest and again.preview_id == shown.preview_id
    assert "renamed-copy" not in json.dumps(shown.manifest) and base.name not in json.dumps(
        shown.manifest)
    version = commit_import(prepare_kg_edges_import(registry, data_dir, copied, "kg-test",
                                                    "test"), registry, shown.preview_id)
    assert version.manifest["taken_from"]["checkpoint_sha256"] == _sha(base.read_bytes())


def test_a_change_of_graph_after_the_preview_is_refused(trained, registry, tmp_path, capsys):
    """The workspace's tensors replaced between preview and commit: the commit
    reads them again, and the checkpoint was not trained on them."""
    import shutil

    from scripts import import_cohort
    from tests.fixtures.registered_cohort import write_demo_workspace

    workspace = tmp_path / "ws"
    shutil.copytree(trained["data_dir"], workspace, ignore=shutil.ignore_patterns("runs"))
    assert import_cohort.main(_cli(registry, workspace, trained["base"])) == 0
    preview = re.search(r"^Preview id: ([0-9a-f]{16})$", capsys.readouterr().out,
                        re.MULTILINE).group(1)
    shutil.rmtree(workspace)
    write_demo_workspace(workspace, feature_seed=7)
    assert import_cohort.main(_cli(registry, workspace, trained["base"], "test", "kg-test",
                                   "--commit", preview)) == 1
    err = capsys.readouterr().err
    assert "does not match the workspace" in err and "Nothing was written" in err
    assert list(registry.iterdir()) == []


def test_a_taken_label_is_refused(trained, registry):
    _import(registry, trained["data_dir"], trained["base"], "test", "kg-1")
    with pytest.raises(CohortVersionRefused, match="'kg-1' already exists"):
        prepare_kg_edges_import(registry, trained["data_dir"], trained["base"], "kg-1", "val")
    plan = prepare_kg_edges_import(registry, trained["data_dir"], trained["base"], "kg-2", "val")
    _import(registry, trained["data_dir"], trained["base"], "val", "kg-2")
    with pytest.raises(CohortVersionRefused, match="already exists.*Nothing was written"):
        commit_import(plan, registry, plan.preview_id)


def _seed_kept_digest(payload, data_dir):
    payload["data_roles"]["split"]["seed"] += 1


def _part_counts(payload, data_dir):
    edges = payload["data_roles"]["split"]["edges"]
    edges["train"], edges["test"] = edges["train"] - 1, edges["test"] + 1


def _view_digest_only(payload, data_dir):
    """The view digest changed in every place the record names it, its rule not."""
    digest = "0" * 64
    payload["data_roles"]["graph_view"]["digest"] = digest
    payload["data_roles"]["split"]["view_digest"] = digest
    payload["graph_view"]["digest"] = digest


def _split_view_digest(payload, data_dir):
    payload["data_roles"]["split"]["view_digest"] = "0" * 64


def _no_split(payload, data_dir):
    del payload["data_roles"]["split"]


@pytest.mark.parametrize("change, said", [
    (_seed_kept_digest, "recorded edge split does not rebuild from this workspace's view: the "
                        "split's recorded digest"),
    (_part_counts, "the split's recorded edges"),
    (_view_digest_only, "but its rule builds view"),
    (_split_view_digest, "contradicts itself"),
    (_no_split, "names no graph view rule or edge split"),
])
def test_a_checkpoint_whose_record_does_not_rebuild_is_refused_before_anything_is_written(
        trained, registry, tmp_path, change, said, capsys):
    from scripts import import_cohort

    data_dir = trained["data_dir"]
    bad = _rewritten(trained["base"], tmp_path / "bad.pt", lambda p: change(p, data_dir))
    assert import_cohort.main(_cli(registry, data_dir, bad)) == 1
    err = capsys.readouterr().err
    assert said in err and "Nothing was written" in err, err
    assert "Traceback" not in err
    assert list(registry.iterdir()) == []


def test_a_checkpoint_without_a_record_or_of_another_graph_is_refused(trained, registry,
                                                                      tmp_path):
    from tests.fixtures.registered_cohort import (
        write_checkpoint_without_model_spec,
        write_demo_workspace,
    )

    data_dir = trained["data_dir"]
    old = tmp_path / "old.pt"
    write_checkpoint_without_model_spec(old, data_dir=data_dir)
    with pytest.raises(CohortVersionRefused, match="has no model record this build reads"):
        prepare_kg_edges_import(registry, data_dir, old, "kg", "test")
    other = write_demo_workspace(tmp_path / "other", feature_seed=7)
    with pytest.raises(ValueError, match="does not match the workspace"):
        prepare_kg_edges_import(registry, other, trained["base"], "kg", "test")
    junk = tmp_path / "junk.pt"
    junk.write_bytes(b"not a checkpoint")
    with pytest.raises(CohortVersionRefused, match="could not be read as a checkpoint"):
        prepare_kg_edges_import(registry, data_dir, junk, "kg", "test")
    assert list(registry.iterdir()) == []


# ---------------------------------------------------------------------------
# The reader: the pairs file, and the manifest
# ---------------------------------------------------------------------------
@pytest.fixture
def committed(trained, registry):
    _, version = _import(registry, trained["data_dir"], trained["base"], "test")
    return registry, version


def test_a_tampered_truncated_or_missing_pairs_file_is_refused_on_read(committed):
    registry, version = committed
    path = version.directory / "pairs.bin"
    data = path.read_bytes()
    path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    listed = resolve_version(registry, "kg-test")
    with pytest.raises(CohortVersionRefused, match="is not the pairs file .* binds"):
        read_kg_edge_pairs(listed)
    path.write_bytes(data[:-8])
    with pytest.raises(CohortVersionRefused, match="changed after the import"):
        read_kg_edge_pairs(listed)
    path.unlink()
    with pytest.raises(CohortVersionRefused, match="is missing, though"):
        read_kg_edge_pairs(listed)


def test_pairs_bytes_that_match_an_edited_digest_still_must_be_the_recorded_pairs(committed):
    """A file and its manifest digest both edited: the parse is checked against
    the rows and the pairs digest the version records, so reordered pairs or a
    torn row are refused, never read as the version's."""
    registry, version = committed
    path = version.directory / "pairs.bin"
    data = path.read_bytes()
    rows = [data[i:i + 32] for i in range(0, len(data), 32)]
    for edited, said in ((b"".join(reversed(rows)), "parses to pairs"),
                         (data + b"\x00" * 8, "is not a pairs table")):
        path.write_bytes(edited)
        manifest = copy.deepcopy(version.manifest)
        manifest["pairs"]["sha256"] = _sha(edited)
        (version.directory / "cohort.json").write_bytes(encode_json(manifest))
        with pytest.raises(CohortVersionRefused, match=said):
            read_kg_edge_pairs(resolve_version(registry, "kg-test"))


def _without(block, key):
    return {k: v for k, v in block.items() if k != key}


def _set(manifest, dotted, value):
    out = copy.deepcopy(manifest)
    *parents, last = dotted.split(".")
    target = out
    for key in parents:
        target = target[key]
    target[last] = value
    return out


@pytest.mark.parametrize("edit, said", [
    (lambda m: _without(m, "pairs"), "missing pairs"),
    (lambda m: _without(m, "kind"), "which must state its kind (missing kind)"),
    (lambda m: _set(m, "kind", "edges"), "kind 'edges'; this revision reads kinds"),
    (lambda m: _set(m, "role", "holdout"), "role is the part of the edge split"),
    (lambda m: _set(m, "counts.pairs", m["counts"]["pairs"] + 1), "counts.pairs ("),
    (lambda m: _set(m, "counts.positive_edges", m["counts"]["positive_edges"] + 1),
     "counts.positive_edges"),
    (lambda m: _set(m, "counts", _without(m["counts"], "edges_without_negative")),
     "counts must hold exactly"),
    (lambda m: _set(m, "counts.message_passing_edges", True),
     "counts.message_passing_edges must be a whole number"),
    (lambda m: _set(m, "split.edges.test", m["split"]["edges"]["test"] + 1),
     "is not split.edges.test"),
    (lambda m: _set(m, "split.view_digest", "0" * 64), "split.view_digest is not graph_view"),
    (lambda m: _set(m, "message_passing.includes_measured_edges", True),
     "message_passing.includes_measured_edges must be false for role test"),
    (lambda m: _set(m, "message_passing.part", "val"), "message_passing must be the split's"),
    (lambda m: _set(m, "pairs.rows", m["pairs"]["rows"] + 1), "pairs.rows ("),
    (lambda m: _set(m, "pairs.seed", m["pairs"]["seed"] + 1), "pairs.seed ("),
    (lambda m: _set(m, "pairs.batches", 7), "pairs.batches (7)"),
    (lambda m: _set(m, "pairs", _without(m["pairs"], "batch_pairs")),
     "pairs.batch_pairs must hold a whole number"),
    (lambda m: _set(m, "pairs.batch_pairs", m["pairs"]["batch_pairs"] + [0]),
     "pairs.batch_pairs must hold a whole number of at least 0 for each of the"),
    (lambda m: _set(m, "pairs.batch_pairs", [m["pairs"]["batch_pairs"][0] - 1,
                                              *m["pairs"]["batch_pairs"][1:]]),
     "pairs.batch_pairs sum to"),
    (lambda m: _set(m, "pairs.batch_pairs", [m["pairs"]["batch_size"] + 1,
                                              *m["pairs"]["batch_pairs"][1:]]),
     "pairs.batch_pairs[0] ("),
    (lambda m: _set(m, "pairs.part", "val"), "pairs.part ('val') is not the role (test)"),
    (lambda m: _set(m, "pairs.sha256", "x"), "pairs.sha256 must be a SHA-256"),
    (lambda m: _set(m, "count_units", {}), "count_units must give a unit"),
    (lambda m: _set(m, "graph", _without(m["graph"], "kg")), "graph must hold"),
    (lambda m: _set(m, "taken_from", {}), "taken_from must hold a checkpoint_sha256"),
    (lambda m: {**m, "schema_version": 1}, "is schema 1, which records no kind, yet states"),
])
def test_a_hand_edited_manifest_is_refused_naming_the_field(committed, edit, said):
    registry, version = committed
    (version.directory / "cohort.json").write_bytes(encode_json(edit(version.manifest)))
    with pytest.raises(CohortVersionRefused) as refused:
        resolve_version(registry, "kg-test")
    assert said in str(refused.value), str(refused.value)


def test_the_schema_check_refuses_a_schema_it_does_not_read_before_any_field(committed):
    """The current schema check, with schema 1 alone readable: a KG edge set
    (schema 2) is refused by its schema, with its reason, before any field or
    kind is read, so it is never read by the supplied rules. This pins the
    current check only; that the revision before schema 2 refuses one the
    same way ("is schema 2; this revision reads schema 1 only") was checked
    out of tree, on that revision's own reader, and is not shown here."""
    registry, version = committed
    assert version.manifest["schema_version"] == 2
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cr, "VERSION_SCHEMAS", (1,))
        with pytest.raises(CohortVersionRefused,
                           match="is schema 2; this revision reads schemas 1 only"):
            resolve_version(registry, "kg-test")


# ---------------------------------------------------------------------------
# Listings and other readers
# ---------------------------------------------------------------------------
@pytest.fixture
def both(trained, registry, tmp_path):
    """A registry holding a supplied cohort and a KG edge set."""
    from tests.fixtures.registered_cohort import (
        graph_source_cases,
        import_version,
        write_demo_workspace,
        write_source,
    )

    workspace = write_demo_workspace(tmp_path / "supplied-ws")
    source = write_source(tmp_path / "cases.json", graph_source_cases(workspace))
    supplied = import_version(registry, workspace, source, label="cases-1")
    _, edges = _import(registry, trained["data_dir"], trained["base"], "test")
    return registry, supplied, edges


def test_every_listing_shows_kind_role_and_counts_never_an_edge_or_a_case_id(both):
    from src.api.services import server_lists
    from src.api.services.record_states import CohortVersions
    from src.config import server_settings as ss
    from src.evaluation.cohort_registry import cohort_version_block, list_versions

    registry, supplied, edges = both
    environ = {ss.ENV_REGISTRY: str(registry)}
    listed = {entry.label: entry for entry in list_versions(registry)}
    assert {label: entry.version.kind for label, entry in listed.items()} == {
        "cases-1": "supplied", "kg-test": "kg_edges"}

    listing = server_lists.list_cohort_versions(
        environ=environ, report_state=server_lists.stored_report_state)
    items = {item.label: item for item in listing.items}
    item = items["kg-test"]
    assert (item.kind, item.role, item.problem) == ("kg_edges", "test", None)
    assert item.counts == edges.manifest["counts"]
    assert item.manifest_sha256 == edges.identity.sha256
    assert (item.source_format, item.source_sha256) == (None, None)
    assert items["cases-1"].kind == "supplied"

    dumped = json.dumps(listing.as_dict("versions"))
    for exclusion in supplied.manifest["exclusions"]:
        assert exclusion["case_id"] not in dumped
    assert "exclusions" not in dumped and str(registry) not in dumped
    # Aggregates only: no pairs block, view, split or provenance block, and no
    # node id list, so no edge.
    for key in ("lp_pairs_digest", "pairs.bin", "view_digest", "graph_view", "taken_from",
                '"message_passing"'):
        assert key not in dumped, key
    pairs = read_kg_edge_pairs(edges).pairs
    heads = json.dumps(pairs.head.tolist())
    assert heads not in dumped

    found = CohortVersions.listed(environ)
    entry = found.listed[edges.identity.sha256]
    assert (entry["kind"], entry["role"], entry["label"]) == ("kg_edges", "test", "kg-test")
    assert found.by_digest(edges.identity.sha256)[0].kind == "kg_edges"

    block = cohort_version_block(edges)
    assert tuple(block) == cr.KG_EDGES_BLOCK_FIELDS
    assert (block["kind"], block["role"], block["counts"]) == (
        "kg_edges", "test", edges.manifest["counts"])
    assert heads not in json.dumps(block)


def test_the_test_page_lists_a_kg_edge_set_and_its_start_says_why_it_is_not_measured(both):
    """The Test page's reader of the listing (its rows, options and summary)
    takes a KG edge set without failing, by label, kind and role, never in a
    supplied cohort's words; its listing by kind with each kind's counts is
    research comparison Stage 7's. Start's check refuses it with the
    protocol's own reason."""
    pytest.importorskip("gradio")
    from src.api.services import server_lists
    from src.config import server_settings as ss
    from src.evaluation import protocols
    from src.webui.components import test_page as tp
    from src.webui.components.workspaces import version_row
    from tests.fixtures.capabilities import described

    registry, _, edges = both
    listing = server_lists.list_cohort_versions(environ={ss.ENV_REGISTRY: str(registry)})
    page = {"versions": [version_row(item) for item in listing.items]}
    options = {value: label for label, value in tp.version_choices(page, None)}
    assert options["kg-test"] == "kg-test · a KG edge set (test)"
    shown = tp.version_html(page, "kg-test", None)
    assert "A KG edge set (test)" in shown
    assert "source cases" not in shown and "ranked-eligible" not in shown
    (row,) = [v for v in page["versions"] if v["label"] == "kg-test"]
    model = {"record": "model", "learning_stage": "finetune", "task": "disease",
             "capabilities": described("finetune", "disease")}
    assert tp.measure_refusal(model, row, "disease") == protocols.inputs_refusal(
        "disease", "kg_edges", edges.manifest["counts"])
    assert "does not measure a KG edge set" in tp.measure_refusal(model, row, "disease")


def test_the_cohorts_route_lists_a_kg_edge_set(both, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.api.routes import lists
    from src.config import server_settings as ss

    registry, _, edges = both
    monkeypatch.setenv(ss.ENV_REGISTRY, str(registry))
    app = FastAPI()
    app.include_router(lists.router, prefix="/api/v1")
    body = TestClient(app).get("/api/v1/cohorts").json()
    (item,) = [v for v in body["versions"] if v["label"] == "kg-test"]
    assert (item["kind"], item["role"], item["counts"]) == (
        "kg_edges", "test", edges.manifest["counts"])


def test_a_task_protocol_refuses_a_kg_edge_set_before_anything_is_written(trained, registry):
    from scripts import measure_scorer

    _import(registry, trained["data_dir"], trained["base"], "test")
    before = _tree(registry)
    with pytest.raises(ValueError) as refused:
        measure_scorer.read_measurement_inputs(
            trained["disease"], trained["data_dir"], torch.device("cpu"), cohort="kg-test",
            registry=registry, protocol="disease")
    assert str(refused.value) == (
        "the disease measurement does not measure a KG edge set; it measures a generated "
        "split or a supplied cohort version. Nothing was written.")
    assert _tree(registry) == before


def test_the_diagnosis_route_refuses_a_kg_edge_set_as_a_reference_cohort(trained, registry,
                                                                         monkeypatch):
    from fastapi import HTTPException

    from src.api.routes import diagnose
    from src.config import server_settings as ss

    _import(registry, trained["data_dir"], trained["base"], "test")
    monkeypatch.setenv(ss.ENV_REGISTRY, str(registry))
    with pytest.raises(HTTPException) as refused:
        diagnose._reference_cohort("kg-test", "patients_like_me", pipeline=None)
    assert refused.value.status_code == 422
    assert ("reference cohort 'kg-test': it is a KG edge set (test), not a supplied cohort "
            "version") in refused.value.detail
    assert cr.not_a_supplied_cohort("kg_edges", "test") in refused.value.detail


def _serving(state_cohorts):
    """A Diagnosis card state serving a patients-like-me model, its facts
    holding `state_cohorts` (`diagnosis_card.reference_cohorts`)."""
    identity = {"scoring_mode": "patients_like_me"}
    return {"status": {"initialized": True, "served_identity": identity},
            "facts": {"cohorts": state_cohorts}}


def test_the_diagnosis_card_lists_a_kg_edge_set_marked_and_never_chooses_it(both, trained,
                                                                             tmp_path):
    """The Reference cohort select lists every registry version: a KG edge set
    with its kind and the server's reason, never bound and never the default,
    even when it was taken from the serving graph and sorts first; a supplied
    cohort is chosen. With KG edge sets alone, Run is off with its reason."""
    pytest.importorskip("gradio")
    from src.config import server_settings as ss
    from src.webui.components import diagnosis_card as card
    from src.webui.components import diagnosis_panel as dp

    registry, supplied, edges = both
    _import(registry, trained["data_dir"], trained["base"], "train", "a-edges")
    environ = {ss.ENV_REGISTRY: str(registry)}
    record = edges.manifest["dataset_record"]["sha256"]
    rows = [{"name": "ws", "dataset_record": {"binding": "verified", "sha256": record}}]
    cohorts = card.reference_cohorts("ws", rows, {"ws": trained["data_dir"]}, environ=environ)
    by_label = {v["label"]: v for v in cohorts["versions"]}
    assert list(by_label) == ["a-edges", "cases-1", "kg-test"]
    for label, role in (("a-edges", "train"), ("kg-test", "test")):
        row = by_label[label]
        assert (row["kind"], row["binding"], row["problem"]) == ("kg_edges", card.NOT_A_COHORT,
                                                                 None)
        assert row["binding_words"] == f"a KG edge set ({role}), not a supplied cohort version"
    assert by_label["cases-1"]["binding"] != card.NOT_A_COHORT

    options, chosen = dp.cohort_choices(cohorts, None)
    texts = {value: text for text, value in options}
    assert (texts["a-edges"], texts["kg-test"]) == ("a-edges · KG edge set",
                                                    "kg-test · KG edge set")
    assert "KG edge set" not in texts["cases-1"]
    assert chosen == "cases-1"
    assert "a KG edge set (test), not a supplied cohort version" in dp.cohort_info(cohorts,
                                                                                   "kg-test")
    assert dp.cohort_refusal(_serving(cohorts)) is None

    only_edges = {**cohorts, "versions": [by_label["a-edges"], by_label["kg-test"]]}
    assert dp.cohort_choices(only_edges, None)[1] is None
    assert dp.cohort_refusal(_serving(only_edges)) == (
        f"{dp.RUN_DISABLED_NO_COHORT} {dp.NO_REFERENCE_VERSION}")
    assert "operator" in dp.NO_REFERENCE_VERSION


def test_the_overlap_audit_refuses_a_kg_edge_set_with_one_message(both, trained, tmp_path):
    """The registered overlap audit compares a supplied cohort's diseases; a
    KG edge set of any role (``train`` included, which would collide with
    the generated split's side) is refused with one message and exit 1,
    before anything is read past its manifest, and nothing is written."""
    from scripts import audit_split_overlap

    registry, _, _ = both
    _import(registry, trained["data_dir"], trained["base"], "train", "kg-train")
    for label, role in (("kg-test", "test"), ("kg-train", "train")):
        output = tmp_path / f"{label}.json"
        with pytest.raises(SystemExit) as exited:
            audit_split_overlap.main(["--data-dir", str(trained["data_dir"]), "--output",
                                      str(output), "--registry", str(registry),
                                      "--cohort", label])
        said = str(exited.value.code)
        assert said == (f"the version {label!r} is a KG edge set ({role}), not a supplied "
                        "cohort version: this audit compares the diseases of a supplied "
                        "cohort's cases with train's. Nothing was written."), said
        assert not output.exists()


# ---------------------------------------------------------------------------
# Supplied versions
# ---------------------------------------------------------------------------
def test_a_schema_2_supplied_manifest_which_no_revision_writes_is_refused(both):
    """Nothing is guessed (research comparison §2): a supplied version is
    written in schema 1, so a schema-2 manifest of kind ``supplied`` is a shape
    no revision writes. It is refused with its reason, listed as unreadable,
    never read by the schema-1 rules; the schema-1 version it was made from
    still reads, and a KG edge set stays schema 2."""
    from src.evaluation.cohort_registry import list_versions

    registry, supplied, edges = both
    assert cr.KIND_SCHEMAS == {"supplied": 1, "kg_edges": 2, "generated": 2}
    assert (supplied.manifest["schema_version"], edges.manifest["schema_version"]) == (1, 2)
    manifest = {**supplied.manifest, "schema_version": 2, "kind": "supplied"}
    (supplied.directory / "cohort.json").write_bytes(encode_json(manifest))
    said = ("is schema 2 of kind 'supplied', which no revision writes: a supplied version is "
            "schema 1.")
    with pytest.raises(CohortVersionRefused, match=re.escape(said)):
        resolve_version(registry, "cases-1")
    listed = {entry.label: entry for entry in list_versions(registry)}
    assert listed["cases-1"].version is None and said in listed["cases-1"].problem
    assert listed["kg-test"].version.kind == "kg_edges"
    (supplied.directory / "cohort.json").write_bytes(encode_json(supplied.manifest))
    assert resolve_version(registry, "cases-1").kind == "supplied"
