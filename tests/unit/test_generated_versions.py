"""
Generated splits as registry versions; content references and source roles.
==========================================================================
Research comparison §5 Stage 4 (the owner's added item A, a limited N2; the
plan review's E3 and E4). A workspace's own generated patient split is kept as
a ``generated`` version through the registry's one preview/commit flow and
read back by its one reader. What is pinned here, on the demo workspace and
on a base and disease child trained for real on it (`trained_disease_child`):

- **the vocabulary**: ``generated`` is a registered kind, schema 2, one of the
  measurement manifest's words; its roles are train, val and test, recorded as
  given;
- **each generated split imports** (``train``, ``val``) with its data identity:
  the samples' digest (the workspace's file, kept byte for byte), the split
  manifest (kept byte for byte) and the split's files by digest, the
  generation definition and the allocation as the manifest records them, what
  was verified, and counts with units, counted from the samples; no path.
  ``test`` is a role, and is refused for what the workspace lacks, never for
  the role, with nothing written;
- **the flow**: the CLI's preview id is the commit id; a workspace re-sampled
  between preview and commit, one whose split is not the file its manifest
  binds, a file changed while it was read, and a split manifest a version
  cannot record as it is (a realised count its samples contradict, a split
  file named by a path, no generation definition) are refused before
  anything is written; the CLI refuses another kind's arguments;
- **immutability**: the version reads and verifies after its workspace is
  re-sampled, its generation definition checked against the split manifest it
  keeps, the change showing as reference states;
- **the reader**: a hand-edited manifest is refused naming the field, a split
  file named by a path among them;
- **every listing** shows its kind, role and counts, never a sample; it is
  never presented as a supplied cohort (the Diagnosis route and card, the
  overlap audit, the Test page's words) or as test; a measurement of it is refused before anything is
  read with the stated gap (measurement is Stage 5), the CLI, the runner and
  the Test page alike, while the workspace's own split is measured as before;
- **no identity is made up**: a report measured with ``--split`` stays as it
  is and is still refused for registration, with a version of its very split
  in the registry;
- **E3, the move**: a registry holding a KG edge set and a generated split,
  copied and moved to another authorised root, resolves to the same versions
  by the same digests; each reference is checked at a location by its digest,
  its reason naming the location's kind, never a path, and a missing
  dependency is a state, never an exception and never a pass;
- **E4**: a version carries named, digested source roles beyond the KG
  builder's four, whose own rules are unchanged; species and namespaces are
  the graph's, by reference: the view a KG edge set binds keeps the demo's
  mouse gene and ortholog relation.

Module: tests/unit/test_generated_versions.py
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shlex
import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.evaluation import cohort_registry as cr  # noqa: E402
from src.evaluation.cohort_registry import (  # noqa: E402
    CohortVersionRefused,
    commit_import,
    encode_json,
    prepare_generated_import,
    resolve_version,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tree(root: Path) -> dict:
    return {str(p.relative_to(root)): (p.read_bytes() if p.is_file() else None)
            for p in sorted(root.rglob("*"))}


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """The demo workspace (generated train and val splits), read only."""
    from tests.fixtures.registered_cohort import write_demo_workspace

    return write_demo_workspace(tmp_path_factory.mktemp("generated") / "ws")


@pytest.fixture(scope="module")
def resampled(tmp_path_factory):
    """The same graph, re-sampled: other train and val splits and manifest."""
    from tests.fixtures.registered_cohort import write_demo_workspace

    return write_demo_workspace(tmp_path_factory.mktemp("resampled") / "ws", sample_seed=7)


@pytest.fixture
def registry(tmp_path):
    path = tmp_path / "registry"
    path.mkdir()
    return path


def _import(registry: Path, data_dir: Path, role: str, label: str = None):
    plan = prepare_generated_import(registry, data_dir, label or f"gen-{role}", role)
    return plan, commit_import(plan, registry, plan.preview_id)


def _copy_workspace(source: Path, target: Path) -> Path:
    shutil.copytree(source, target)
    return target


def _resample(data_dir: Path, resampled: Path) -> None:
    """Replace `data_dir`'s generated splits and split manifest with
    `resampled`'s (the same graph), as a re-run of the generator would."""
    for name in ("train_samples.json", "val_samples.json", "split_manifest.json"):
        shutil.copyfile(resampled / name, data_dir / name)


# ---------------------------------------------------------------------------
# The vocabulary
# ---------------------------------------------------------------------------
def test_generated_is_a_registered_kind_in_schema_2_with_the_split_roles():
    from src.evaluation import protocols
    from src.evaluation.cohort import COHORT_KINDS, GENERATED_SPLITS

    assert cr.GENERATED == protocols.GENERATED == "generated" and cr.GENERATED in COHORT_KINDS
    assert set(cr.VERSION_KINDS) == set(COHORT_KINDS)
    assert cr.KIND_SCHEMAS[cr.GENERATED] == cr.KINDED_SCHEMA_VERSION == 2
    assert cr.GENERATED_ROLES == ("train", "val", "test")
    assert set(GENERATED_SPLITS) <= set(cr.GENERATED_ROLES)
    for role in cr.GENERATED_ROLES:
        assert cr.validate_generated_role(role) == role
    with pytest.raises(CohortVersionRefused, match="the split it is"):
        cr.validate_generated_role("holdout")
    assert set(cr.GENERATED_COUNT_UNITS) == {"samples", "diseases"}


# ---------------------------------------------------------------------------
# Each split imports, with its data identity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("role", ("train", "val"))
def test_each_generated_split_imports_with_its_data_identity(workspace, registry, role):
    from src.kg.artifacts import read_split_manifest, verify_graph_reads
    from src.kg.storage.file_storage import read_graph_artifacts, read_samples_file

    plan, version = _import(registry, workspace, role)
    manifest = version.manifest
    split_bytes = (workspace / "split_manifest.json").read_bytes()
    split = json.loads(split_bytes)
    samples_bytes = (workspace / f"{role}_samples.json").read_bytes()

    assert (version.kind, manifest["schema_version"], manifest["role"]) == ("generated", 2, role)
    assert plan.kind == "generated" and plan.source_bytes is None and plan.pairs_bytes is None
    bound = verify_graph_reads(read_split_manifest(workspace),
                               read_graph_artifacts(workspace).reads)
    assert manifest["graph"] == {r: bound[r] for r in manifest["graph"]}
    assert manifest["dataset_record"] == {
        "sha256": _sha((workspace / "kg.provenance.json").read_bytes()),
        "origin": "synthetic", "sources": []}
    assert manifest["split_manifest"] == {"file": "split_manifest.json",
                                          "sha256": _sha(split_bytes),
                                          "schema_version": split["schema_version"]}
    assert manifest["split_artifacts"] == {
        name: split["artifacts"][name] for name in ("train_samples", "val_samples")}
    for block in ("generation", "allocation", "realised"):
        assert manifest[block] == split[block], block
    assert manifest["generation"]["algorithm_version"] == 1
    assert manifest["verified"] == {"splits": ["train", "val"], "disjointness_measured": True}
    n = len(json.loads(samples_bytes))
    assert manifest["samples"] == {"file": "samples.json", "sha256": _sha(samples_bytes), "n": n}
    assert manifest["samples"]["sha256"] == split["artifacts"][f"{role}_samples"]
    # Counted from the samples, which the split manifest's own count agrees with.
    diseases = {row["disease_id"] for row in json.loads(samples_bytes)}
    assert manifest["counts"] == {"samples": n, "diseases": len(diseases)}
    assert len(diseases) == split["realised"][f"{role}_diseases"]
    assert manifest["count_units"] == cr.GENERATED_COUNT_UNITS

    # The version holds its manifest, the workspace's samples and its split
    # manifest, byte for byte, the blocks it copies being that file's; no
    # source is kept and no path is recorded.
    assert sorted(p.name for p in version.directory.iterdir()) == [
        "cohort.json", "samples.json", "split_manifest.json"]
    assert (version.directory / "samples.json").read_bytes() == samples_bytes
    kept = (version.directory / "split_manifest.json").read_bytes()
    assert kept == split_bytes and plan.split_manifest_bytes == split_bytes
    assert version.samples_path == version.directory / "samples.json"
    assert not (registry / "sources").exists()
    written = (version.directory / "cohort.json").read_text()
    assert str(workspace) not in written and "filename" not in written
    read = read_samples_file(version.samples_path, expected_sha256=manifest["samples"]["sha256"])
    assert len(read.samples) == n

    block = cr.cohort_version_block(version)
    assert tuple(block) == cr.GENERATED_BLOCK_FIELDS
    assert (block["kind"], block["role"], block["manifest_sha256"]) == (
        "generated", role, version.identity.sha256)
    assert "sim_" not in json.dumps(block)


def test_a_test_role_is_a_role_and_is_refused_for_the_split_the_workspace_lacks(workspace,
                                                                                registry):
    """The generator writes train and val; ``test`` is accepted as a role and
    refused because this workspace binds no generated test split, with
    nothing written. Never refused for being test, or for not being it."""
    before = _tree(registry)
    with pytest.raises(CohortVersionRefused) as refused:
        prepare_generated_import(registry, workspace, "gen-test", "test")
    # The workspace's path is a location the message names; the rest is
    # the refusal's own words, which never refuse the role.
    said = str(refused.value).replace(str(workspace), "<workspace>")
    assert "holds no generated test split" in said and "binds the splits train, val" in said
    assert "role" not in said and "the split it is" not in said
    assert _tree(registry) == before


# ---------------------------------------------------------------------------
# The flow
# ---------------------------------------------------------------------------
def _cli(registry, data_dir, role="val", label="gen-val", *extra):
    return ["--kind", "generated", "--registry", str(registry), "--data-dir", str(data_dir),
            "--label", label, "--role", role, *extra]


def test_the_cli_previews_then_commits_under_the_id_it_showed(workspace, registry, capsys):
    from scripts import import_cohort

    assert import_cohort.main(_cli(registry, workspace)) == 0
    out = capsys.readouterr().out
    preview = re.search(r"Preview id: ([0-9a-f]{16})", out).group(1)
    assert "Generated split preview: label gen-val, role val" in out
    assert "sim_" not in out and "Nothing was written." in out
    assert not (registry / "versions").exists()
    command = out.strip().splitlines()[-1].strip()
    argv = shlex.split(command)
    assert argv[:2] == ["python", "scripts/import_cohort.py"] and argv[-1] == preview
    assert import_cohort.main(argv[2:]) == 0
    out = capsys.readouterr().out
    assert "Committed generated split gen-val (val): 5 generated patients over 1 disease" in out
    version = resolve_version(registry, "gen-val")
    assert cr.preview_id(version.manifest) == preview and version.kind == "generated"


@pytest.mark.parametrize("extra, said", [
    (("--source", "cases.json"), "--source does not apply to --kind generated"),
    (("--partial-phenotypes", "keep"), "--partial-phenotypes does not apply"),
    (("--from-checkpoint", "best.pt"), "--from-checkpoint does not apply"),
])
def test_the_cli_refuses_another_kind_s_arguments(registry, tmp_path, extra, said, capsys):
    from scripts import import_cohort

    with pytest.raises(SystemExit) as exited:
        import_cohort.parse_args(_cli(registry, tmp_path, "val", "gen-val", *extra))
    assert exited.value.code == 2
    assert said in capsys.readouterr().err


def test_a_workspace_re_sampled_after_the_preview_is_refused(workspace, resampled, registry,
                                                            tmp_path, capsys):
    from scripts import import_cohort

    data_dir = _copy_workspace(workspace, tmp_path / "ws")
    assert import_cohort.main(_cli(registry, data_dir)) == 0
    preview = re.search(r"Preview id: ([0-9a-f]{16})", capsys.readouterr().out).group(1)
    _resample(data_dir, resampled)
    assert import_cohort.main(_cli(registry, data_dir, "val", "gen-val", "--commit",
                                   preview)) == 1
    assert "the inputs changed since the preview" in capsys.readouterr().err
    assert not (registry / "versions").exists()


def test_a_split_that_is_not_the_file_its_manifest_binds_is_refused(workspace, registry,
                                                                    tmp_path):
    data_dir = _copy_workspace(workspace, tmp_path / "ws")
    samples = data_dir / "val_samples.json"
    samples.write_text(json.dumps(json.loads(samples.read_text()), indent=1))
    before = _tree(registry)
    with pytest.raises(ValueError, match="is not the file"):
        prepare_generated_import(registry, data_dir, "gen-val", "val")
    assert _tree(registry) == before


def test_a_taken_label_is_refused(workspace, registry):
    _import(registry, workspace, "val")
    with pytest.raises(CohortVersionRefused, match="already exists"):
        prepare_generated_import(registry, workspace, "gen-val", "train")


def _edit_split_manifest(data_dir: Path, edit) -> None:
    """Edit `data_dir`'s split manifest in place, its artefact digests intact."""
    path = data_dir / "split_manifest.json"
    manifest = json.loads(path.read_text())
    edit(manifest)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


@pytest.mark.parametrize("edit, said", [
    # The count is the samples', never the manifest's word: a manifest whose
    # realised count is not what its (digest-verified) split holds is refused.
    (lambda m: _set(m, "realised.val_diseases", 99),
     "counts.diseases (1, the distinct diseases among the samples) is not "
     "realised.val_diseases (99)"),
    # A split file named by a path is refused, never kept as a reference.
    (lambda m: _set(m, "artifacts", {**m["artifacts"], "../outside/secret_samples": "0" * 64}),
     "split_artifacts must name each split's samples file (<split>_samples, the split an "
     "identifier)"),
    # No generation definition to record.
    (lambda m: m["generation"].pop("algorithm_version"),
     "generation must state the generator's algorithm and algorithm_version"),
])
def test_a_split_manifest_a_version_cannot_record_as_it_is_is_refused_naming_it(
        workspace, registry, tmp_path, edit, said):
    """The workspace's readers pass each of these split manifests (their
    samples are the files they bind, and their disease sets are the ones
    recorded); the version's own rules refuse them before anything is
    written, naming the split manifest and the field."""
    data_dir = _copy_workspace(workspace, tmp_path / "ws")
    _edit_split_manifest(data_dir, edit)
    before = _tree(registry)
    with pytest.raises(CohortVersionRefused) as refused:
        prepare_generated_import(registry, data_dir, "gen-val", "val")
    message = str(refused.value)
    assert message.startswith(f"the workspace {data_dir}'s split_manifest.json does not "
                              "describe its val split as a generated split version records it")
    assert said in message
    assert _tree(registry) == before


@pytest.mark.parametrize("name", ("val_samples.json", "split_manifest.json"))
def test_a_file_changed_while_it_was_read_is_refused(workspace, registry, monkeypatch, name):
    """The bytes kept are a second read of each file; that read is kept only
    when it hashes to the digest of the read that was verified and parsed."""
    import sys

    from src.utils import fingerprint

    real = fingerprint.read_once

    def read_once(path, **kwargs):
        read = real(path, **kwargs)
        if (Path(path).name == name
                and sys._getframe(1).f_code.co_name == "prepare_generated_import"):
            other = read.data + b"\n"
            return fingerprint.FileRead(read.path, other, _sha(other), read.opened)
        return read

    monkeypatch.setattr(fingerprint, "read_once", read_once)
    before = _tree(registry)
    with pytest.raises(CohortVersionRefused, match=re.escape(
            f"{workspace / name} changed while it was read")):
        prepare_generated_import(registry, workspace, "gen-val", "val")
    assert _tree(registry) == before


def test_a_plan_without_its_split_manifest_is_not_committed(workspace, registry):
    plan = prepare_generated_import(registry, workspace, "gen-val", "val")
    for broken in (plan._replace(split_manifest_bytes=None),
                   plan._replace(split_manifest_bytes=plan.split_manifest_bytes + b" ")):
        with pytest.raises(CohortVersionRefused, match="split manifest or samples its manifest"):
            commit_import(broken, registry, plan.preview_id)
    assert not (registry / "versions").exists()


# ---------------------------------------------------------------------------
# Immutability, and the reader
# ---------------------------------------------------------------------------
def test_the_version_reads_and_verifies_after_its_workspace_is_re_sampled(workspace, resampled,
                                                                         registry, tmp_path):
    """The samples are copied, so a re-sampled workspace changes nothing the
    version holds; what changed shows as reference states, named."""
    from src.kg.storage.file_storage import read_samples_file

    data_dir = _copy_workspace(workspace, tmp_path / "ws")
    _, version = _import(registry, data_dir, "val")
    _resample(data_dir, resampled)
    again = resolve_version(registry, "gen-val")
    assert again.identity.sha256 == version.identity.sha256
    read_samples_file(again.samples_path, expected_sha256=again.manifest["samples"]["sha256"])
    states = {s.reference.name: s.state for s in cr.check_references(again, workspace=data_dir)}
    # The version keeps its split manifest, so its generation definition is
    # still checked against bytes; the workspace's split files are other bytes.
    assert states["split.train_samples"] == states["split.val_samples"] == cr.DIFFERS
    assert states["split_manifest"] == states["samples"] == cr.VERIFIED
    assert states["graph.kg"] == states["dataset_record"] == cr.VERIFIED
    assert (again.directory / "split_manifest.json").read_bytes() != (
        data_dir / "split_manifest.json").read_bytes()


def _set(manifest, dotted, value):
    *path, last = dotted.split(".")
    block = manifest
    for key in path:
        block = block[key]
    block[last] = value
    return manifest


def _drop(manifest, key):
    del manifest[key]
    return manifest


_SOURCE = {"role": "mondo", "digest": "a" * 64, "declared_version": None}


@pytest.mark.parametrize("edit, said", [
    (lambda m: _drop(m, "kind"), "must state its kind"),
    (lambda m: _set(m, "role", "holdout"), "a generated split's role is the split it is"),
    (lambda m: _set(m, "samples.sha256", "b" * 64),
     "samples.sha256 is not split_artifacts.val_samples"),
    (lambda m: _set(m, "samples.file", "rows.json"), "samples must hold the file samples.json"),
    (lambda m: _set(m, "counts.samples", 99), "counts.samples (99) is not samples.n"),
    (lambda m: _set(m, "counts.diseases", 7),
     "counts.diseases (7, the distinct diseases among the samples) is not realised.val_diseases"),
    (lambda m: _set(m, "count_units", {}), "count_units must give a unit"),
    (lambda m: _set(m, "generation.algorithm_version", "1"), "generation must state"),
    (lambda m: _set(m, "split_manifest.sha256", "x"),
     "split_manifest must hold the file split_manifest.json and a sha256"),
    (lambda m: _set(m, "split_manifest.file", "../split_manifest.json"),
     "split_manifest must hold the file split_manifest.json and a sha256"),
    (lambda m: _set(m, "split_artifacts", {}), "split_artifacts must name each split"),
    # A split file is named by its split's name, never a path: neither an
    # absolute path nor a climb out of the workspace is a reference.
    (lambda m: _set(m, "split_artifacts", {**m["split_artifacts"], "/abs/x_samples": "0" * 64}),
     "split_artifacts must name each split's samples file (<split>_samples, the split an "
     "identifier)"),
    (lambda m: _set(m, "split_artifacts", {**m["split_artifacts"], "../x_samples": "0" * 64}),
     "split_artifacts must name each split's samples file"),
    (lambda m: _set(m, "verified.splits", []), "verified must list the splits verified"),
    (lambda m: _drop(m, "allocation"), "missing allocation"),
    (lambda m: _set(m, "dataset_record.sources", [{**_SOURCE, "role": "Mondo"}]),
     "dataset_record.sources[0].role ('Mondo') is not an identifier"),
    (lambda m: _set(m, "dataset_record.sources", [_SOURCE, _SOURCE]),
     "names the role 'mondo' a second time"),
    (lambda m: _set(m, "dataset_record.sources", [{**_SOURCE, "digest": "short"}]),
     "dataset_record.sources[0].digest (role mondo) must be a SHA-256"),
    (lambda m: _set(m, "dataset_record.sources", [{**_SOURCE, "path": "/srv/mondo.obo"}]),
     "must hold exactly role, digest and declared_version"),
    (lambda m: _set(m, "schema_version", 1), "which records no kind, yet states kind"),
])
def test_a_hand_edited_generated_manifest_is_refused_naming_the_field(workspace, registry,
                                                                      edit, said):
    _, version = _import(registry, workspace, "val")
    path = version.directory / "cohort.json"
    path.write_bytes(encode_json(edit(copy.deepcopy(version.manifest))))
    with pytest.raises(CohortVersionRefused, match=re.escape(said)):
        resolve_version(registry, "gen-val")
    listed = {entry.label: entry for entry in cr.list_versions(registry)}
    assert listed["gen-val"].version is None and said in listed["gen-val"].problem


def test_the_generation_definition_is_checked_against_the_split_manifest_kept(workspace,
                                                                              registry):
    """The version's generation definition, allocation and realised sets are
    its kept split manifest's blocks: a copy edited after the import (a shape
    every reader accepts) shows as differs, naming the block, and so does a
    kept split manifest whose bytes changed."""
    _, version = _import(registry, workspace, "val")
    kept = version.directory / "split_manifest.json"
    states = _states(version)
    assert states["split_manifest"].state == cr.VERIFIED
    assert states["split_manifest"].reason == (
        "the version's split_manifest.json holds the bytes recorded")

    manifest = copy.deepcopy(version.manifest)
    manifest["generation"]["num_val"] += 1
    manifest["allocation"]["seed"] += 1
    (version.directory / "cohort.json").write_bytes(encode_json(manifest))
    edited = resolve_version(registry, "gen-val")
    state = _states(edited)["split_manifest"]
    assert (state.state, state.found) == (cr.DIFFERS, version.manifest["split_manifest"]["sha256"])
    assert state.reason == ("the version's split_manifest.json holds the bytes recorded, but the "
                            "version's generation, allocation are not that file's")

    kept.write_text("{}")
    state = _states(edited)["split_manifest"]
    assert state.state == cr.DIFFERS and state.found == _sha(b"{}")


def test_samples_changed_after_the_import_are_refused_and_show_as_differs(workspace, registry):
    from src.kg.storage.file_storage import SamplesDigestMismatch, read_samples_file

    _, version = _import(registry, workspace, "val")
    version.samples_path.write_text("[]")
    with pytest.raises(SamplesDigestMismatch):
        read_samples_file(version.samples_path,
                          expected_sha256=version.manifest["samples"]["sha256"])
    (state,) = [s for s in cr.check_references(version) if s.reference.name == "samples"]
    assert state.state == cr.DIFFERS and state.found == _sha(b"[]")


# ---------------------------------------------------------------------------
# Listings and other readers
# ---------------------------------------------------------------------------
@pytest.fixture
def listed(workspace, registry, tmp_path):
    """A registry holding a supplied cohort and a generated val split."""
    from tests.fixtures.registered_cohort import graph_source_cases, import_version, write_source

    source = write_source(tmp_path / "cases.json", graph_source_cases(workspace))
    supplied = import_version(registry, workspace, source, label="cases-1")
    _, generated = _import(registry, workspace, "val")
    return registry, supplied, generated


def test_every_listing_shows_kind_role_and_counts_never_a_sample(listed):
    from src.api.services import server_lists
    from src.api.services.record_states import CohortVersions
    from src.config import server_settings as ss

    registry, _, generated = listed
    environ = {ss.ENV_REGISTRY: str(registry)}
    kinds = {entry.label: entry.version.kind for entry in cr.list_versions(registry)}
    assert kinds == {"cases-1": "supplied", "gen-val": "generated"}

    listing = server_lists.list_cohort_versions(
        environ=environ, report_state=server_lists.stored_report_state)
    item = {item.label: item for item in listing.items}["gen-val"]
    assert (item.kind, item.role, item.problem) == ("generated", "val", None)
    assert item.counts == generated.manifest["counts"]
    assert (item.source_format, item.source_sha256) == (None, None)
    dumped = json.dumps(listing.as_dict("versions"))
    assert "sim_" not in dumped and str(registry) not in dumped
    assert "phenotype_ids" not in dumped

    found = CohortVersions.listed(environ)
    entry = found.listed[generated.identity.sha256]
    assert (entry["kind"], entry["role"], entry["label"]) == ("generated", "val", "gen-val")


def test_the_test_page_names_it_by_kind_and_role_never_in_a_cohort_s_words(listed):
    """The Test page's options and summary name a generated split as the
    registry does, never with a supplied cohort's source cases or
    ranked-eligible counts; a supplied cohort reads as before."""
    pytest.importorskip("gradio")
    from src.api.services import server_lists
    from src.config import server_settings as ss
    from src.webui.components import test_page as tp
    from src.webui.components.workspaces import version_row

    registry, supplied, _ = listed
    listing = server_lists.list_cohort_versions(environ={ss.ENV_REGISTRY: str(registry)})
    page = {"versions": [version_row(item) for item in listing.items]}
    options = {value: label for label, value in tp.version_choices(page, None)}
    assert options["gen-val"] == "gen-val · a generated split (val)"
    counts = supplied.manifest["counts"]
    assert options["cases-1"] == (f"cases-1 · role {supplied.manifest['role']} · "
                                  f"{counts['ranked_eligible']} of {counts['source_cases']} "
                                  "ranked-eligible")
    shown = re.sub(r"<[^>]+>", " ", tp.version_html(page, "gen-val", None))
    assert "A generated split (val)" in shown
    for words in ("source cases", "ranked-eligible", "test", "cohort"):
        assert words not in shown, words
    assert "source cases" in tp.version_html(page, "cases-1", None)


def test_the_cohorts_route_lists_a_generated_split(listed, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.api.routes import lists
    from src.config import server_settings as ss

    registry, _, generated = listed
    monkeypatch.setenv(ss.ENV_REGISTRY, str(registry))
    app = FastAPI()
    app.include_router(lists.router, prefix="/api/v1")
    body = TestClient(app).get("/api/v1/cohorts").json()
    (item,) = [v for v in body["versions"] if v["label"] == "gen-val"]
    assert (item["kind"], item["role"], item["counts"]) == (
        "generated", "val", generated.manifest["counts"])


def test_it_is_never_presented_as_a_supplied_cohort(listed, workspace, monkeypatch, tmp_path):
    """The Diagnosis route refuses it as a reference cohort, its card marks
    it and never chooses it, and the overlap audit refuses it, each with the
    registry's one sentence; it is named by its role, never as test."""
    pytest.importorskip("gradio")
    from fastapi import HTTPException

    from scripts import audit_split_overlap
    from src.api.routes import diagnose
    from src.config import server_settings as ss
    from src.webui.components import diagnosis_card as card
    from src.webui.components import diagnosis_panel as dp

    registry, _, generated = listed
    words = cr.not_a_supplied_cohort("generated", "val")
    assert words == "a generated split (val), not a supplied cohort version"
    assert "test" not in words

    monkeypatch.setenv(ss.ENV_REGISTRY, str(registry))
    with pytest.raises(HTTPException) as refused:
        diagnose._reference_cohort("gen-val", "patients_like_me", pipeline=None)
    assert refused.value.status_code == 422 and words in refused.value.detail

    environ = {ss.ENV_REGISTRY: str(registry)}
    record = generated.manifest["dataset_record"]["sha256"]
    rows = [{"name": "ws", "dataset_record": {"binding": "verified", "sha256": record}}]
    cohorts = card.reference_cohorts("ws", rows, {"ws": workspace}, environ=environ)
    by_label = {v["label"]: v for v in cohorts["versions"]}
    assert (by_label["gen-val"]["binding"], by_label["gen-val"]["binding_words"]) == (
        card.NOT_A_COHORT, words)
    options, chosen = dp.cohort_choices(cohorts, None)
    assert {v: t for t, v in options}["gen-val"] == "gen-val · generated split"
    assert chosen == "cases-1"
    only = {**cohorts, "versions": [by_label["gen-val"]]}
    assert dp.cohort_choices(only, None)[1] is None
    assert "a generated split" in dp.NO_REFERENCE_VERSION

    output = tmp_path / "audit.json"
    with pytest.raises(SystemExit) as exited:
        audit_split_overlap.main(["--data-dir", str(workspace), "--output",
                                  str(output), "--registry", str(registry),
                                  "--cohort", "gen-val"])
    assert str(exited.value.code).startswith(f"the version 'gen-val' is {words}: ")
    assert not output.exists()


def test_a_measurement_of_it_is_refused_before_anything_is_read(listed, tmp_path):
    """Measuring a registered generated split is the round's Stage 5: the CLI,
    the runner and the Test page say the one stated gap before anything is
    read (the checkpoint here does not even exist), and the workspace's own
    split is judged as before."""
    pytest.importorskip("gradio")
    from scripts import measure_scorer
    from src.api.services.training_manager import TestRunInputs
    from src.evaluation import protocols
    from src.webui.components import test_page as tp
    from tests.fixtures.capabilities import described

    registry, _, generated = listed
    before = _tree(registry)
    absent = tmp_path / "absent.pt"
    with pytest.raises(ValueError) as refused:
        measure_scorer.read_measurement_inputs(
            absent, tmp_path / "no-workspace", torch.device("cpu"), cohort="gen-val",
            registry=registry, protocol="disease")
    assert str(refused.value) == f"{protocols.GENERATED_VERSION_GAP} Nothing was written."
    assert _tree(registry) == before

    model = {"record": "model", "learning_stage": "finetune", "task": "disease",
             "capabilities": described("finetune", "disease")}
    row = {"label": "gen-val", "kind": "generated", "counts": generated.manifest["counts"]}
    for protocol in ("disease", "gene", "patients_like_me"):
        model = {**model, "capabilities": described("finetune", protocol)}
        assert tp.measure_refusal(model, row, protocol) == protocols.GENERATED_VERSION_GAP
    inputs = TestRunInputs(
        registry=registry, version_label="gen-val", version_sha256=generated.identity.sha256,
        workspace=tmp_path, checkpoint=absent, checkpoint_sha256="c" * 64, protocol="disease",
        learning_stage="finetune", task="disease",
        capabilities=described("finetune", "disease"), version_kind="generated",
        version_counts=generated.manifest["counts"])
    with pytest.raises(ValueError, match=re.escape(protocols.GENERATED_VERSION_GAP)):
        inputs.reports_dir()
    # The workspace's own generated split: no counts, judged as before.
    assert protocols.inputs_refusal("disease", "generated", None) is None


def test_no_identity_is_made_up_for_a_report_measured_with_split(trained_disease_child,
                                                                 registry, tmp_path):
    """A report measured with --split is kept where it was written, as it was
    written, and is still refused for registration with the ledger's own
    sentence, though a version of its very split is now in the registry."""
    from scripts import measure_scorer
    from src.evaluation.sidecar import build_record

    data_dir = trained_disease_child["data_dir"]
    output = tmp_path / "val-report.json"
    assert measure_scorer.main([
        "--mode", "disease", "--checkpoint", str(trained_disease_child["disease"]),
        "--data-dir", str(data_dir), "--split", "val", "--output", str(output),
        "--device", "cpu"]) == 0
    kept = output.read_bytes()
    _, version = _import(registry, data_dir, "val")
    report = json.loads(kept)
    assert report["manifest"]["cohort_version"] is None
    assert report["manifest"]["cohort_kind"] == "generated"
    # The report and the version name the same split by its bytes, and
    # nothing writes one into the other.
    assert report["manifest"]["artifact_digests"]["samples"] == version.manifest["samples"][
        "sha256"]
    with pytest.raises(ValueError) as refused:
        build_record(report, _sha(kept))
    said = str(refused.value)
    assert "a generated split's report is not registered" in said
    assert ("the ledger does not yet support a generated split's data identity (it has no "
            "registered cohort version)") in said
    assert output.read_bytes() == kept


# ---------------------------------------------------------------------------
# E3: content references, and the move
# ---------------------------------------------------------------------------
def test_references_name_dependencies_by_content_never_by_path(workspace, registry):
    _, version = _import(registry, workspace, "val")
    refs = {ref.name: ref for ref in cr.version_references(version)}
    assert list(refs) == [
        "graph.kg", "graph.node_features", "graph.edge_indices", "graph.num_nodes",
        "dataset_record", "split_manifest", "split.train_samples", "split.val_samples",
        "samples"]
    manifest = version.manifest
    assert refs["graph.kg"].sha256 == manifest["graph"]["kg"]
    assert "species, namespaces and external ids" in refs["graph.kg"].what
    assert refs["split_manifest"].sha256 == manifest["split_manifest"]["sha256"]
    assert (refs["split_manifest"].kept, refs["split_manifest"].file) == (
        cr.IN_VERSION, "split_manifest.json")
    assert refs["split.val_samples"].kept == cr.IN_WORKSPACE
    assert refs["samples"].sha256 == refs["split.val_samples"].sha256
    assert {ref.kept for ref in refs.values()} == {cr.IN_WORKSPACE, cr.IN_VERSION}
    for ref in refs.values():
        assert re.fullmatch(r"[0-9a-f]{64}", ref.sha256), ref
        assert ref.file is None or "/" not in ref.file, ref
    assert cr.version_references(version.manifest) == cr.version_references(version)


def _states(version, **where):
    return {s.reference.name: s for s in cr.check_references(version, **where)}


def test_a_supplied_version_s_references_are_its_source_samples_and_graph(listed, workspace):
    """The one reader of references covers every kind: a supplied cohort's
    source is the registry's copy, named by its digest, and its HPO and MONDO
    entries are the dataset record's named sources."""
    _, supplied, _ = listed
    states = _states(supplied, workspace=workspace)
    assert list(states) == ["graph.kg", "graph.node_features", "graph.edge_indices",
                            "graph.num_nodes", "dataset_record", "source", "samples"]
    assert states["source"].reference.kept == cr.IN_REGISTRY
    assert states["source"].reference.sha256 == supplied.manifest["source"]["sha256"]
    assert all(state.state == cr.VERIFIED for state in states.values())


def test_a_registry_moved_to_another_authorised_root_resolves_the_same_versions(
        trained_disease_child, registry, tmp_path):
    """E3's move acceptance: a registry holding a KG edge set and a generated
    split is copied to a new root, and the original moved away; both versions
    resolve there with the same identities and references, the service lists
    them by the same digests, and each reference is verified at a location by
    its digest. A missing dependency is a state: never an exception, never a
    pass."""
    from src.api.services import server_lists
    from src.config import server_settings as ss
    from src.evaluation.cohort_registry import prepare_kg_edges_import

    data_dir, base = trained_disease_child["data_dir"], trained_disease_child["base"]
    edges_plan = prepare_kg_edges_import(registry, data_dir, base, "kg-test", "test")
    edges = commit_import(edges_plan, registry, edges_plan.preview_id)
    _, generated = _import(registry, data_dir, "val")
    before = {v.manifest["label"]: (v.identity.sha256, v.manifest, cr.version_references(v),
                                    cr.cohort_version_block(v))
              for v in (edges, generated)}

    copied = tmp_path / "elsewhere" / "registry"
    shutil.copytree(registry, copied)
    os.rename(registry, tmp_path / "moved-away")
    for label, (sha, manifest, refs, block) in before.items():
        moved = resolve_version(copied, label)
        assert moved.directory == copied.resolve() / "versions" / label
        assert (moved.identity.sha256, moved.manifest) == (sha, manifest)
        assert cr.version_references(moved) == refs
        assert cr.cohort_version_block(moved) == block
    listing = server_lists.list_cohort_versions(environ={ss.ENV_REGISTRY: str(copied)})
    assert {i.label: i.manifest_sha256 for i in listing.items} == {
        label: sha for label, (sha, *_) in before.items()}

    # Authorisation is the root's, apart from content: the same bytes under a
    # root the rule refuses are not read.
    with pytest.raises(cr.RegistryRefused):
        resolve_version(os.path.relpath(copied), "kg-test")

    moved_edges = resolve_version(copied, "kg-test")
    moved_generated = resolve_version(copied, "gen-val")
    found = _states(moved_edges, workspace=data_dir, checkpoint=base)
    for name in ("graph.kg", "graph.node_features", "graph.edge_indices", "graph.num_nodes",
                 "dataset_record", "pairs", "checkpoint"):
        assert found[name].state == cr.VERIFIED, (name, found[name])
    assert found["graph_view"].state == found["split"].state == cr.NOT_CHECKED
    assert found["graph_view"].reason == ("rebuilt from the graph by its recorded rule, and "
                                          "checked when the import rebuilds it, not by a file")
    assert all(s.state == cr.VERIFIED for s in _states(moved_generated,
                                                       workspace=data_dir).values())

    # The referenced workspace graph is absent: every workspace reference is
    # missing, named, and the version's own files still verify.
    empty = tmp_path / "empty-workspace"
    empty.mkdir()
    absent = _states(moved_generated, workspace=empty)
    for name, state in absent.items():
        want = cr.VERIFIED if name in ("samples", "split_manifest") else cr.MISSING
        assert state.state == want, (name, state)
    # The reason names the location by its kind and the file's name, never by
    # a path (D7); the path looked at is the state's location alone.
    assert absent["graph.kg"].reason == "the workspace's kg.json is not there"
    assert absent["graph.kg"].location == empty / "kg.json"
    assert absent["samples"].reason == "the version's samples.json holds the bytes recorded"
    everything = [*absent.values(), *_states(moved_edges, workspace=empty,
                                            checkpoint=tmp_path).values()]
    for state in everything:
        assert str(tmp_path) not in state.reason, state
    # No location given: not checked, with the reason, never verified.
    unlocated = _states(moved_edges)
    assert unlocated["graph.kg"].state == unlocated["checkpoint"].state == cr.NOT_CHECKED
    assert unlocated["checkpoint"].reason == "no checkpoint location was given"
    assert unlocated["pairs"].state == cr.VERIFIED
    # A location that cannot be read is a state too.
    unreadable = _states(moved_edges, checkpoint=tmp_path)["checkpoint"]
    assert unreadable.state == cr.UNREADABLE and unreadable.location == tmp_path
    assert unreadable.reason.startswith("the given checkpoint cannot be read")
    assert _states(moved_edges, checkpoint=tmp_path / "gone.pt")["checkpoint"].state == (
        cr.MISSING)
    assert set(cr.REFERENCE_STATES) == {cr.VERIFIED, cr.DIFFERS, cr.MISSING, cr.UNREADABLE,
                                        cr.NOT_CHECKED}


# ---------------------------------------------------------------------------
# E4: source roles beyond the KG builder's four
# ---------------------------------------------------------------------------
_EXTRA_SOURCES = [
    {"role": "mondo", "digest": "1" * 64, "declared_version": "2026-10-01"},
    {"role": "hpo", "digest": "2" * 64, "declared_version": "2026-09-15"},
    {"role": "phenotype_hpoa", "digest": "3" * 64, "declared_version": None},
    {"role": "genes_to_phenotype", "digest": "4" * 64, "declared_version": None},
    {"role": "mgi_orthologs", "digest": "5" * 64, "declared_version": "2026-08"},
    {"role": "hp_mp_mapping", "digest": "6" * 64, "declared_version": None},
]


def _with_sources(plan, label):
    """`plan`'s version under `label`, its dataset record naming
    `_EXTRA_SOURCES`: what a version made from a record of a future KG build
    would hold."""
    manifest = {**copy.deepcopy(plan.manifest), "label": label}
    manifest["dataset_record"]["sources"] = copy.deepcopy(_EXTRA_SOURCES)
    return plan._replace(label=label, manifest=manifest, preview_id=cr.preview_id(manifest))


def test_a_version_carries_named_source_roles_beyond_the_kg_builder_s_four(
        trained_disease_child, registry):
    from src.evaluation.cohort_registry import prepare_kg_edges_import
    from src.kg.provenance import SOURCE_ROLES, ProvenanceError, source_entry

    data_dir, base = trained_disease_child["data_dir"], trained_disease_child["base"]
    plans = (prepare_generated_import(registry, data_dir, "gen-val", "val"),
             prepare_kg_edges_import(registry, data_dir, base, "kg-test", "test"))
    for plan in plans:
        label = f"{plan.label}-sources"
        version = commit_import(_with_sources(plan, label), registry,
                                _with_sources(plan, label).preview_id)
        again = resolve_version(registry, label)
        assert again.manifest["dataset_record"]["sources"] == _EXTRA_SOURCES
        refs = {ref.name: ref for ref in cr.version_references(again)}
        for entry in _EXTRA_SOURCES:
            ref = refs[f"source.{entry['role']}"]
            assert (ref.sha256, ref.kept) == (entry["digest"], cr.NAMED)
        states = _states(version, workspace=data_dir)
        assert states["source.mgi_orthologs"].state == cr.NOT_CHECKED
        assert "input of the KG build" in states["source.mgi_orthologs"].reason

    # The KG builder's own provenance rules are unchanged: a real new source
    # needs the builder's extension first.
    assert SOURCE_ROLES == ("mondo", "hpo", "phenotype_hpoa", "genes_to_phenotype")
    with pytest.raises(ProvenanceError, match="not one of the roles a build consumes"):
        source_entry("mgi_orthologs", None, "5" * 64)


def test_species_and_namespaces_are_the_graph_s_by_reference(trained_disease_child, registry):
    """The demo graph holds a mouse gene and a human-mouse ortholog relation:
    the view a KG edge set binds (rebuilt here by its recorded rule, to its
    recorded digest) keeps both, and the version records no species of its
    own; the graph (and the view, by its digest) is what says them."""
    from src.evaluation.cohort_registry import prepare_kg_edges_import
    from src.kg.graph_view import build_graph_view
    from src.kg.storage.file_storage import read_graph_artifacts

    data_dir, base = trained_disease_child["data_dir"], trained_disease_child["base"]
    plan = prepare_kg_edges_import(registry, data_dir, base, "kg-train", "train")
    manifest = plan.manifest
    view = build_graph_view(read_graph_artifacts(data_dir).graph_data,
                            rule=copy.deepcopy(manifest["graph_view"]["rule"]))
    assert view.digest == manifest["graph_view"]["digest"]
    assert view.type_nodes("mouse_gene").numel() == 1
    assert "gene|human_mouse_ortholog|mouse_gene" in view.relations
    assert "species" not in json.dumps(manifest)
    refs = {ref.name for ref in cr.version_references(manifest)}
    assert {"graph.kg", "graph_view"} <= refs
