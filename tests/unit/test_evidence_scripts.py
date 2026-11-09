"""Backlog item 10 — the evidence scripts: M4 reproducible, and the M2.6 inventory.

Those facts were established by reading artifacts in a review thread and had
existed as pasted text. M4 bounds every number the project reports on `val`.
The M1-M3 family scan (`audit_checkpoint_family.py --checkpoint-dir`) retired
with the checkpoints it read, files written before the model spec (K8-7b);
its recorded evidence stays as it was, and the script keeps its `--inventory`,
whose vocabulary, refusal to overwrite and report contract are checked here and
in test_m2_inventory.py.

**What these tests are for.** The institutional run happens on the deployment's
machine and cannot happen here. What can be checked here is that each script
emits the schema BACKLOG §5.2 fixes, **omits what §5.2 forbids**, refuses inputs
it cannot honestly summarise, and survives what an institutional workspace will
actually present — an unreadable checkpoint, an artifact from another graph.

Module: tests/unit/test_evidence_scripts.py
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _run(script: str, argv: list):
    import importlib

    return importlib.import_module(f"scripts.{script}").main(argv)


# ---------------------------------------------------------------------------
# M4 — split overlap
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def registered(tmp_path_factory):
    """A workspace with a committed cohort version mapped against its graph
    (`tests/fixtures/registered_cohort`), built once.

    **The shape a real deployment has**, and the only shape M4's overlap question
    is still open in. `train` and `val` come from the allocation step, are
    disjoint, and their manifest is bound to their exact bytes. The test cohort
    was imported from a source that was never cut from this disease universe, so
    how much of it the model has labelled examples for is exactly what this audit
    measures -- the upstream team reported 109 of 319 for theirs. Tests read it
    and never write into it; a test that changes a file works on `_copy`.
    """
    from tests.fixtures.registered_cohort import build_registered_cohort

    return build_registered_cohort(tmp_path_factory.mktemp("registered"))


def _registered(registered, out: Path, *extra, data_dir=None, registry=None, label=None):
    """Run the overlap audit over train against the registered cohort version."""
    from tests.fixtures.registered_cohort import LABEL

    return _run("audit_split_overlap", [
        "--data-dir", str(data_dir or registered.data_dir), "--output", str(out),
        "--registry", str(registry or registered.registry), "--cohort", label or LABEL,
        *extra,
    ])


def _copy(registered, tmp_path: Path):
    """A writable copy of the workspace and the registry. Same bytes, so the
    version still binds to the copied workspace."""
    import shutil

    shutil.copytree(registered.data_dir, tmp_path / "workspace")
    shutil.copytree(registered.registry, tmp_path / "registry")
    return tmp_path / "workspace", tmp_path / "registry"


def _version_dir(registry: Path) -> Path:
    from tests.fixtures.registered_cohort import LABEL

    return registry / "versions" / LABEL


def _edit_version(registry: Path, edit) -> None:
    """Rewrite the version manifest as the registry encodes it, after `edit`."""
    from src.evaluation.cohort_registry import encode_json

    path = _version_dir(registry) / "cohort.json"
    manifest = json.loads(path.read_text())
    edit(manifest)
    path.write_bytes(encode_json(manifest))


def _disease_sets(data_dir: Path, registry: Path):
    """Train's and the version's disease sets, read here with `json` rather than
    through the audit's reader."""
    def diseases(path: Path):
        return {row["disease_id"] for row in json.loads(path.read_text())}

    return (diseases(data_dir / "train_samples.json"),
            diseases(_version_dir(registry) / "samples.json"))


def test_the_overlap_evidence_records_sizes_and_no_identifiers(registered, tmp_path):
    """§5.2 forbids patient ids, sample ids and per-disease lists; the claim is
    two set sizes and the size of their intersection."""
    from tests.fixtures.registered_cohort import ROLE

    out = tmp_path / "m4.json"
    _registered(registered, out)
    text = out.read_text()
    report = json.loads(text)
    train, cohort = _disease_sets(registered.data_dir, registered.registry)

    assert report["splits"] == {"training": "train", "evaluation": ROLE}
    assert report["counts"]["train_diseases"] == len(train)
    assert report["counts"][f"{ROLE}_diseases"] == len(cohort)
    assert report["counts"]["shared_diseases"] == len(train & cohort)
    assert report["counts"][f"{ROLE}_diseases_absent_from_train"] == len(cohort - train)
    assert report["counts"][f"{ROLE}_samples"] == registered.version.manifest["samples"]["n"]
    assert report["schema_version"] == 6
    for row in json.loads((_version_dir(registered.registry) / "samples.json").read_text()):
        assert row["patient_id"] not in text, "no case id may reach the artifact"
    assert "sim_train" not in text, "no training patient id may reach the artifact"


def test_the_report_records_the_cohort_version_it_read(registered, tmp_path):
    """The version is what a reader cites: its label and role, and the digests
    of the manifest and the samples the audit read. It replaces the bare
    `evaluation_cohort_kind: supplied` a loose file used to be recorded as."""
    from tests.fixtures.registered_cohort import LABEL, ROLE

    out = tmp_path / "m4.json"
    _registered(registered, out)
    report = json.loads(out.read_text())
    version_dir = _version_dir(registered.registry)

    block = report["cohort_version"]
    assert {key: block[key] for key in ("label", "role", "manifest_sha256", "samples_sha256")} == {
        "label": LABEL,
        "role": ROLE,
        "manifest_sha256": hashlib.sha256((version_dir / "cohort.json").read_bytes()).hexdigest(),
        "samples_sha256": hashlib.sha256((version_dir / "samples.json").read_bytes()).hexdigest(),
    }
    assert any("graph digests" in line for line in block["verified_against"])
    assert "evaluation_cohort_kind" not in report


def test_the_overlap_denominator_is_recorded_beside_the_ratio(registered, tmp_path):
    """A percentage alone loses the denominator, and the denominator is half the
    claim: 100% of 319 and 100% of 1 are not the same finding."""
    out = tmp_path / "m4.json"
    _registered(registered, out)
    train, cohort = _disease_sets(registered.data_dir, registered.registry)

    overlap = json.loads(out.read_text())["overlap"]
    assert overlap["as_written"] == f"{len(train & cohort)} of {len(cohort)}"
    assert overlap["shared_over_evaluation"] == len(train & cohort) / len(cohort)


def test_an_empty_cohort_version_is_refused_rather_than_reported_as_no_overlap(
    registered, tmp_path
):
    """With no evaluation diseases the ratio has no denominator, and a report of
    zero overlap would be cited later as "no contamination".

    The importer never commits such a version, so this one is forged
    consistently: `samples.json` is an empty list, `cohort.json` binds it, and
    its counts say what the importer would for a source whose every case was
    excluded (the version reader refuses counts that disagree with the
    samples). An emptied *generated* split fails the manifest binding first,
    because the file no longer hashes to what the manifest records — a stricter
    and earlier refusal for the same workspace.
    """
    data_dir, registry = _copy(registered, tmp_path)
    (_version_dir(registry) / "samples.json").write_bytes(b"[]")

    def empty(manifest):
        manifest["samples"] = {"sha256": hashlib.sha256(b"[]").hexdigest(), "n": 0}
        counts = manifest["counts"]
        counts["ranked_eligible"] = 0
        counts["source_cases"] = manifest["source"]["n_cases"] = counts["n_excluded"]

    _edit_version(registry, empty)
    out = tmp_path / "m4.json"

    with pytest.raises(SystemExit, match="holds no samples.*Nothing was written"):
        _registered(registered, out, data_dir=data_dir, registry=registry)
    assert not out.exists()


def test_an_emptied_generated_split_is_caught_by_the_binding_first(registered, tmp_path):
    data_dir, _ = _copy(registered, tmp_path)
    (data_dir / "train_samples.json").write_text("[]")

    with pytest.raises(SystemExit, match="is not the file"):
        _registered(registered, tmp_path / "m4.json", data_dir=data_dir)


# ---------------------------------------------------------------------------
# M4 — the registered cohort version's own refusals
# ---------------------------------------------------------------------------
def test_a_version_mapped_against_another_graph_is_refused_before_its_samples_are_read(
    registered, tmp_path
):
    """Its indices name nodes of the graph it was mapped against; counted
    against another workspace's train, an equal index can be a different
    disease. The other workspace here differs only in `node_features.pt`, so
    the comparison is of all four recorded digests, not of `kg.json` alone."""
    from tests.fixtures.opens import count_opens
    from tests.fixtures.registered_cohort import write_demo_workspace

    other = write_demo_workspace(tmp_path / "other", feature_seed=7)
    out = tmp_path / "m4.json"

    with count_opens() as opens, pytest.raises(SystemExit) as refused:
        _registered(registered, out, data_dir=other)
    message = str(refused.value)
    assert "mapped against another graph" in message and "node_features" in message
    assert "Nothing was written" in message
    assert not out.exists()
    assert opens.opens(_version_dir(registered.registry) / "samples.json") == 0
    assert opens.opens(other / "train_samples.json") == 0


def test_a_version_bound_to_another_dataset_record_is_refused(registered, tmp_path):
    """The binding is the measurement's: the four graph digests and the dataset
    record the version was mapped against, compared with the ones this
    workspace's manifest records."""
    data_dir, registry = _copy(registered, tmp_path)
    _edit_version(registry, lambda manifest: manifest["dataset_record"].update(sha256="0" * 64))
    out = tmp_path / "m4.json"

    with pytest.raises(SystemExit, match="dataset_record") as refused:
        _registered(registered, out, data_dir=data_dir, registry=registry)
    assert "mapped against another graph" in str(refused.value)
    assert not out.exists()


def test_a_version_whose_samples_were_replaced_is_refused_unparsed(registered, tmp_path):
    """The samples are refused unless they are the bytes `cohort.json` binds,
    before they are parsed. The replacement is a valid samples file, so only
    the digest comparison can refuse it."""
    data_dir, registry = _copy(registered, tmp_path)
    samples = _version_dir(registry) / "samples.json"
    samples.write_text(json.dumps(json.loads(samples.read_text())[1:]))
    out = tmp_path / "m4.json"

    with pytest.raises(SystemExit, match="neither decoded nor parsed.*Nothing was written"):
        _registered(registered, out, data_dir=data_dir, registry=registry)
    assert not out.exists()


def test_a_version_without_its_samples_file_is_refused(registered, tmp_path):
    data_dir, registry = _copy(registered, tmp_path)
    (_version_dir(registry) / "samples.json").unlink()
    out = tmp_path / "m4.json"

    with pytest.raises(SystemExit, match="Samples file not found.*Nothing was written"):
        _registered(registered, out, data_dir=data_dir, registry=registry)
    assert not out.exists()


@pytest.mark.parametrize("given", ["--registry", "--cohort"])
def test_registry_and_cohort_are_given_together(registered, tmp_path, given, capsys):
    """An incomplete command line, not a refused workspace: exit 2 with the usage
    line, as the guide's exit-code table and measure_scorer have it."""
    value = {"--registry": str(registered.registry), "--cohort": "demo-test-1"}[given]
    out = tmp_path / "m4.json"

    with pytest.raises(SystemExit) as refused:
        _run("audit_split_overlap", ["--data-dir", str(registered.data_dir),
                                     "--output", str(out), given, value])

    assert refused.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("usage:")
    assert re.search(r"go together.*Nothing was written", err, re.DOTALL)
    assert not out.exists()


def test_splits_are_refused_beside_a_cohort(registered, tmp_path, capsys):
    """--splits names two generated splits; beside --cohort one of them would be
    silently ignored. Exit 2: the command line alone is wrong."""
    out = tmp_path / "m4.json"

    with pytest.raises(SystemExit) as refused:
        _registered(registered, out, "--splits", "train", "val")

    assert refused.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("usage:")
    assert "Pass one or the other" in err
    assert not out.exists()


def test_a_relative_registry_is_refused(registered, tmp_path):
    out = tmp_path / "m4.json"

    with pytest.raises(SystemExit, match="not an absolute path.*Nothing was written"):
        _registered(registered, out, registry=Path("registry"))
    assert not out.exists()


def test_an_unknown_label_is_refused(registered, tmp_path):
    out = tmp_path / "m4.json"

    with pytest.raises(SystemExit, match="holds no version 'no-such-version'"):
        _registered(registered, out, label="no-such-version")
    assert not out.exists()


# ---------------------------------------------------------------------------
# The publishable schema, pinned
# ---------------------------------------------------------------------------
#: Keys each artifact must carry. Required contract, **not** exact equality: a
#: later additive aggregate must not have to edit a test to be allowed.
_REQUIRED = {
    "m4": ("fact", "digests", "counts", "overlap", "deployment_relationship"),
}

#: Which backlog fact each report claims to be. Pinned by value: two files whose
#: `fact` keys merely exist are not distinguishable from each other, and the whole
#: point of the field is that a reader can tell which claim they are holding.
_FACT = {"m4": "M4"}

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
    "m4": [
        (("counts", "train_diseases"), lambda v: _is_int(v) and v >= 1),
        (("counts", "demo_diseases"), lambda v: _is_int(v) and v >= 1),
        (("counts", "shared_diseases"), lambda v: _is_int(v) and v >= 0),
        (("overlap", "shared_over_evaluation"), lambda v: type(v) is float and 0 <= v <= 1),
        (("overlap", "as_written"), lambda v: isinstance(v, str) and " of " in v),
    ],
}

#: Where each report records a SHA-256 of a file, as `(path-to-the-digest, filename)`.
#: The expected value is recomputed in the test from the bytes on disk through a
#: different hashlib entry point than `file_sha256` uses, so a digest of the wrong
#: file — or of nothing — cannot pass. M4's files are named from the root the
#: registered-cohort fixture builds: the workspace's train and the version's samples.
_DIGEST_SITES = {
    "m4": [(("digests", "train_samples"), "workspace/train_samples.json"),
           (("digests", "demo_samples"), "registry/versions/demo-test-1/samples.json")],
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


def test_every_artifact_carries_its_required_contract(tmp_path, registered):
    _registered(registered, tmp_path / "m4.json")

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
    sources = {"m4": registered.data_dir.parent}
    for name, sites in _DIGEST_SITES.items():
        report = json.loads((tmp_path / f"{name}.json").read_text())
        for path, filename in sites:
            expected = hashlib.sha256((sources[name] / filename).read_bytes()).hexdigest()
            assert _at(report, path) == expected, f"{name}{list(path)} is not {filename}"


def test_no_artifact_carries_a_forbidden_identifier_or_path(tmp_path, registered):
    """Checked as substrings over the serialised file, so a future field cannot
    reintroduce one under a new key. The registered workspace and registry live
    under the pytest tmp directory, so a path to either would carry `/tmp/`."""
    _registered(registered, tmp_path / "m4.json")

    for name in _REQUIRED:
        text = (tmp_path / f"{name}.json").read_text()
        for forbidden in _FORBIDDEN_SUBSTRINGS:
            assert forbidden not in text, f"{name} carries {forbidden!r}"


def test_the_deployment_relationship_is_a_bounded_vocabulary(tmp_path, registered):
    """§5.2 forbids operator and host names, and a schema that forbids them cannot
    then accept an arbitrary string — the first person in a hurry writes the
    hostname into it."""
    from src.utils.provenance import DEPLOYMENT_RELATIONSHIPS

    out = tmp_path / "m4.json"

    with pytest.raises(SystemExit):
        _registered(registered, out, "--deployment-relationship", "lab-machine-hostname-42")

    _registered(registered, out, "--deployment-relationship", "identical-sibling")
    assert json.loads(out.read_text())["deployment_relationship"] == "identical-sibling"
    assert "identical-sibling" in DEPLOYMENT_RELATIONSHIPS


def test_the_three_scripts_offer_one_vocabulary(tmp_path):
    """An institutional reader reads both reports for one machine. If the
    inventory accepted a spelling M4 rejected, or the two disagreed on what
    silence looks like, the claims could not be compared.

    Asserted through the parsers — the published behaviour — rather than through
    object identity. A shared constant is how this is implemented today, and the
    equality check below catches a re-duplication the moment it drifts, but a
    faithful re-export or an enum wrapper would be a legitimate refactor and this
    contract is about what the scripts accept, not about how they store it."""
    import importlib

    from src.utils.provenance import DEPLOYMENT_RELATIONSHIPS, UNSTATED_RELATIONSHIP

    scripts = {
        "audit_checkpoint_family": ["--inventory", "."],
        "audit_split_overlap": ["--data-dir", "."],
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

        # ...and silence means the same thing in both.
        assert module.parse_args([*args, *out]).deployment_relationship == UNSTATED_RELATIONSHIP

        assert tuple(module.DEPLOYMENT_RELATIONSHIPS) == tuple(DEPLOYMENT_RELATIONSHIPS), name


@pytest.mark.parametrize("script,args", [
    ("audit_checkpoint_family", ["--inventory", "."]),
    ("audit_split_overlap", ["--data-dir", "."]),
])
def test_no_evidence_file_is_replaced_silently(tmp_path, script, args):
    """Evidence is cited by digest. A file quietly overwritten by a later run is
    not evidence. Asserted for both scripts."""
    out = tmp_path / "existing.json"
    out.write_text("{}")
    argv = ["--output", str(out)]
    for token in args:
        argv.append(token if token.startswith("--") else str(tmp_path / token))

    with pytest.raises(SystemExit, match="exists"):
        _run(script, argv)


# ---------------------------------------------------------------------------
# M4 — the manifest cross-check, and the disjointness gate
# ---------------------------------------------------------------------------
def _allocated_workspace(root: Path, n_diseases: int = 8, n_val: int = 2):
    """A workspace as the current pipeline leaves it: two cohorts and a manifest.

    Built through the shared fixture, which builds it through
    ``build_split_manifest`` -- so it satisfies ``verify_cohort_reads``
    without this file encoding what that checks.
    """
    from tests.fixtures.generated_workspace import write_generated_workspace

    ids = list(range(n_diseases))
    return write_generated_workspace(
        root, train_ids=ids[n_val:], val_ids=ids[:n_val]
    )


def test_a_verified_workspace_records_what_was_bound(tmp_path):
    data_dir, _ = _allocated_workspace(tmp_path / "ws")
    out = tmp_path / "m4.json"

    _run("audit_split_overlap", ["--data-dir", str(data_dir), "--output", str(out)])
    report = json.loads(out.read_text())

    assert report["schema_version"] == 6
    assert report["counts"]["shared_diseases"] == 0
    assert report["evaluation_cohort_kind"] == "generated"
    assert "cohort_version" not in report
    verification = report["manifest_verification"]
    assert verification["verified_splits"] == ["train", "val"]
    assert verification["manifest_disjointness_claim"] is True
    assert verification["measured_disjoint"] is True
    assert verification["manifest_disjointness_claim_checked"] is True
    assert any("SHA-256" in line for line in verification["verified_against"])
    assert any("measured between" in line for line in verification["verified_against"])


@pytest.mark.parametrize("split", ["train", "val"])
def test_content_changed_under_a_preserved_disease_set_is_caught(tmp_path, split):
    """A disease-set digest cannot see this, and that is the point of hashing the
    file. Phenotypes, genes, patient ids, order and multiplicity can all move
    while the disease set is preserved exactly."""
    data_dir, _ = _allocated_workspace(tmp_path / "ws")
    rows = json.loads((data_dir / f"{split}_samples.json").read_text())
    rows[0]["phenotype_ids"] = [999, 998, 997]
    (data_dir / f"{split}_samples.json").write_text(json.dumps(rows))

    with pytest.raises(SystemExit, match="is not the file"):
        _run("audit_split_overlap",
             ["--data-dir", str(data_dir), "--output", str(tmp_path / "m4.json")])
    assert not (tmp_path / "m4.json").exists()


def test_a_manifest_from_another_workspace_is_caught(tmp_path):
    """The generator writes the manifest in the same process as the samples, so a
    disagreement means the files changed afterwards or the manifest is foreign."""
    first, _ = _allocated_workspace(tmp_path / "a")
    second, _ = _allocated_workspace(tmp_path / "b", n_diseases=6)
    (first / "split_manifest.json").write_text(
        (second / "split_manifest.json").read_text()
    )

    with pytest.raises(SystemExit, match="is not the file"):
        _run("audit_split_overlap",
             ["--data-dir", str(first), "--output", str(tmp_path / "m4.json")])


def test_a_manifest_recording_disease_sets_the_files_do_not_hold_is_caught(tmp_path):
    """The recomputation's own test, isolated from the byte check above it.

    Every other way of breaking this also changes the sample bytes, so the digest
    comparison fires first and the recomputation is never reached — removing it
    entirely passed a green run. Here the byte digests are correct and the
    manifest agrees with itself; only its recorded disease sets are another
    workspace's. That is the shape a hand-repaired legacy manifest takes: file
    digests are easy to compute, so the tempting fix is to copy a consistent
    realised/allocated pair from somewhere the tooling accepted.
    """
    data_dir, manifest = _allocated_workspace(tmp_path / "ws")
    _, donor = _allocated_workspace(tmp_path / "donor", n_diseases=6)

    for section in ("realised",):
        for split in ("train", "val"):
            manifest[section][f"{split}_digest"] = donor[section][f"{split}_digest"]
            manifest["allocation"]["allocated"][f"{split}_digest"] = (
                donor["allocation"]["allocated"][f"{split}_digest"]
            )
    (data_dir / "split_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(SystemExit) as excinfo:
        _run("audit_split_overlap",
             ["--data-dir", str(data_dir), "--output", str(tmp_path / "m4.json")])
    message = str(excinfo.value)
    assert "is not the one" in message and "records as realised" in message
    assert "is not the file" not in message, (
        "the byte digests are correct here, or this is testing the wrong check"
    )


def test_a_manifest_whose_realised_contradicts_its_allocation_is_caught(tmp_path):
    """Internal to the file, and still a workspace nothing may be trained on."""
    data_dir, manifest = _allocated_workspace(tmp_path / "ws")
    manifest["allocation"]["allocated"]["val_digest"] = "0" * 64
    (data_dir / "split_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(SystemExit, match="contradicts itself"):
        _run("audit_split_overlap",
             ["--data-dir", str(data_dir), "--output", str(tmp_path / "m4.json")])


def test_a_workspace_without_a_manifest_is_refused(tmp_path):
    """The M4 baseline was measured on such a workspace. Nothing reads one now:
    it was cut before the allocation step, and `EVIDENCE_M4.json` stands as the
    record of that regime rather than as a shape this code still accepts."""
    data_dir = tmp_path / "ws"
    data_dir.mkdir()
    for split in ("train", "val"):
        (data_dir / f"{split}_samples.json").write_text(json.dumps(
            [{"patient_id": "p", "phenotype_ids": [0], "disease_id": 0}]
        ))

    with pytest.raises(SystemExit, match="generated before the disease allocation"):
        _run("audit_split_overlap",
             ["--data-dir", str(data_dir), "--output", str(tmp_path / "m4.json")])


def test_a_manifest_present_but_unbound_does_not_let_a_legacy_workspace_through(tmp_path):
    """Existence is not binding: dropping any manifest beside overlapping cohorts
    would otherwise satisfy the presence check, and training would proceed under a
    manifest describing a different cut entirely."""
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    for split in ("train", "val"):
        (legacy / f"{split}_samples.json").write_text(json.dumps(
            [{"patient_id": "p", "phenotype_ids": [0, 100], "disease_id": 0}]
        ))
    donor, _ = _allocated_workspace(tmp_path / "donor")
    (legacy / "split_manifest.json").write_text(
        (donor / "split_manifest.json").read_text()
    )

    with pytest.raises(SystemExit, match="is not the file"):
        _run("audit_split_overlap",
             ["--data-dir", str(legacy), "--output", str(tmp_path / "m4.json")])


def test_non_disjoint_generated_cohorts_are_refused_by_the_verifier(tmp_path):
    """Disjointness is a contract of the allocation step, so this is a broken
    workspace rather than a measurement.

    The manifest here is forged consistently -- correct byte digests, correct
    realised sets for the overlapping files, realised matching allocated, and
    `disjoint: true` -- so every other check passes and the measured comparison
    is the only thing left."""
    from tests.fixtures.generated_workspace import write_generated_workspace

    data_dir, _ = write_generated_workspace(
        tmp_path / "ws", train_ids=[0, 1, 2], val_ids=[3]
    )
    rows = json.loads((data_dir / "val_samples.json").read_text())
    rows[0]["disease_id"] = 0
    (data_dir / "val_samples.json").write_text(json.dumps(rows))
    # Rebuild the manifest so it describes the overlapping files honestly; only
    # `build_split_manifest`'s own assertion stands between here and that, which
    # is why it is bypassed rather than reused.
    from src.kg.disease_allocation import disease_set_digest
    from src.utils.fingerprint import file_sha256

    manifest = json.loads((data_dir / "split_manifest.json").read_text())
    digest = disease_set_digest(sorted({row["disease_id"] for row in rows}))
    manifest["realised"]["val_digest"] = digest
    manifest["allocation"]["allocated"]["val_digest"] = digest
    manifest["artifacts"]["val_samples"] = file_sha256(data_dir / "val_samples.json")
    # `disjoint` stays True on purpose. A manifest admitting the overlap is caught
    # by the claim check above this one, which would leave the *measured*
    # disjointness check untested -- and it is the only thing standing between a
    # consistently forged manifest and a cohort pair that overlaps.
    (data_dir / "split_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(SystemExit, match="does not hold disease-disjoint cohorts"):
        _run("audit_split_overlap",
             ["--data-dir", str(data_dir), "--output", str(tmp_path / "m4.json")])


def test_a_registered_cohort_audit_does_not_depend_on_generated_val(registered, tmp_path):
    """Generated val is not an input to a train-versus-test measurement, so its
    state must neither block nor support one.

    A corrupt or missing val_samples.json is a real defect in that workspace and
    every path that *reads* val still refuses it. This path does not read it, and
    refusing here would block an institutional measurement for a reason unrelated
    to it.
    """
    data_dir, _ = _copy(registered, tmp_path)
    (data_dir / "val_samples.json").write_text("corrupted, not even json")
    out = tmp_path / "m4.json"

    _registered(registered, out, data_dir=data_dir)
    verification = json.loads(out.read_text())["manifest_verification"]

    assert verification["verified_splits"] == ["train"]
    assert verification["disjointness_was_measured"] is False


def test_a_generated_audit_over_the_same_workspace_still_refuses(tmp_path):
    """The scope narrows what is read, not what soundness means. The same corrupt
    val that a registered audit ignores stops a generated one dead.

    Since contract M2.1's S9 the audit parses each split from its one read before
    it compares digests, so a file that is not JSON is refused by the shared
    reader, naming it, as training and measurement refuse it."""
    data_dir, _ = _allocated_workspace(tmp_path / "ws")
    (data_dir / "val_samples.json").write_text("corrupted, not even json")

    with pytest.raises(SystemExit, match="val_samples.json could not be parsed"):
        _run("audit_split_overlap",
             ["--data-dir", str(data_dir), "--output", str(tmp_path / "m4.json")])


def test_the_report_names_the_scope_the_verifier_actually_ran(tmp_path):
    """The section is built from the verifier's own return value. Deriving it
    independently would let the artifact claim a gate that did not run."""
    data_dir, _ = _allocated_workspace(tmp_path / "ws")
    out = tmp_path / "m4.json"

    _run("audit_split_overlap", ["--data-dir", str(data_dir), "--output", str(out)])
    verification = json.loads(out.read_text())["manifest_verification"]

    assert verification["verified_splits"] == ["train", "val"]
    assert verification["disjointness_was_measured"] is True


def test_a_registered_audit_is_not_blocked_by_the_generated_disjointness_claim(
    registered, tmp_path
):
    """`disjoint` describes the generated train/val relationship, which a
    train-versus-test overlap does not consume.

    Refusing here would block an institutional measurement on the state of a claim
    about an input the run never reads — the same over-validation the scoping was
    introduced to remove, arriving through a claim instead of a file. The report
    says the claim was not checked rather than silently ignoring it.
    """
    data_dir, _ = _copy(registered, tmp_path)
    manifest = json.loads((data_dir / "split_manifest.json").read_text())
    manifest["disjoint"] = False
    (data_dir / "split_manifest.json").write_text(json.dumps(manifest))
    out = tmp_path / "m4.json"

    _registered(registered, out, data_dir=data_dir)
    verification = json.loads(out.read_text())["manifest_verification"]

    assert verification["manifest_disjointness_claim_checked"] is False
    assert verification["manifest_disjointness_claim"] is False, (
        "recorded as the unverified claim it is, not omitted"
    )
    assert not any("disjoint" in line for line in verification["verified_against"])


def test_a_generated_audit_is_blocked_by_that_same_claim(tmp_path):
    """Both generated cohorts are in scope there, so the claim is about an input
    the run does consume."""
    data_dir, manifest = _allocated_workspace(tmp_path / "ws")
    manifest["disjoint"] = False
    (data_dir / "split_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(SystemExit, match="claims disjoint=False"):
        _run("audit_split_overlap",
             ["--data-dir", str(data_dir), "--output", str(tmp_path / "m4.json")])


def test_overlap_with_a_registered_cohort_is_reported_not_refused(registered, tmp_path):
    """It is the finding. Refusing on it would refuse the institutional
    measurement this audit exists to produce."""
    out = tmp_path / "m4.json"

    _registered(registered, out)
    report = json.loads(out.read_text())
    train, cohort = _disease_sets(registered.data_dir, registered.registry)

    assert train & cohort, "the fixture's source shares diseases with train"
    assert report["counts"]["shared_diseases"] == len(train & cohort)
    verification = report["manifest_verification"]
    assert verification["verified_splits"] == ["train"], (
        "the manifest describes the generated splits and says nothing about a "
        "registered cohort version"
    )
    assert "measured_disjoint" not in verification
    assert ("registered cohort version 'demo-test-1'"
            in verification["why_the_evaluation_split_is_not_verified"])


def test_the_retired_cohort_kind_flag_is_refused(tmp_path, capsys):
    """`--evaluation-cohort-kind supplied` read a loose `<split>_samples.json`
    from the workspace. A test cohort is a registered version now, and the flag
    is gone rather than kept as a second way in."""
    data_dir, _ = _allocated_workspace(tmp_path / "ws")
    out = tmp_path / "m4.json"

    with pytest.raises(SystemExit):
        _run("audit_split_overlap", [
            "--data-dir", str(data_dir), "--output", str(out),
            "--splits", "train", "val", "--evaluation-cohort-kind", "supplied",
        ])
    assert "unrecognized arguments: --evaluation-cohort-kind" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize("side", ["training", "evaluation"])
@pytest.mark.parametrize(
    "escape",
    ["/tmp/elsewhere", "../outside", "a/b", "C:\\windows", "..", ".", ".hidden"],
)
def test_a_split_name_that_is_a_path_is_refused_not_resolved(tmp_path, escape, side):
    """Lifting the three-value `choices` list turned `--split` into free text that
    is interpolated into `data_dir / f"{split}_samples.json"`.

    `pathlib` lets an absolute value replace the whole path, so `/tmp/x` leaves
    the workspace entirely and `../x` traverses out of it — and the same string
    reaches the measurement manifest and the ledger's `cohort_role`, so a path
    could re-enter an evidence artifact as an identity. **Both sides:** the
    training split once reached the reader without being resolved.
    """
    data_dir, _ = _allocated_workspace(tmp_path / "ws")
    splits = ["train", escape] if side == "evaluation" else [escape, "val"]

    with pytest.raises(SystemExit, match="must be an identifier"):
        _run("audit_split_overlap", [
            "--data-dir", str(data_dir), "--output", str(tmp_path / "m4.json"),
            "--splits", *splits,
        ])


@pytest.mark.parametrize(
    "escape",
    ["/tmp/x", "../x", "a/b", "C:\\x", "\\\\host\\share", "..", ".", ".hidden",
     "-leading", "x" * 65, "with space", "", "tab\tname", 5, None,
     "TRAIN", "Val", "MyGene2", "mygene2\n", "mygene2\ntrain"],
)
def test_every_non_identifier_form_is_refused_at_the_boundary(escape):
    """The CLI cannot reach some of these — argparse eats a leading hyphen as an
    option — so the alphabet is checked where it is enforced rather than only
    through the one entry point that happens to pass a value through.

    Two families beyond the paths. **Uppercase**: on a case-insensitive
    filesystem `TRAIN_samples.json` *is* `train_samples.json`, which defeats the
    reservation outright, and even where the filesystem keeps them apart two roles
    differing only in case read as one identity to a person. **A trailing
    newline**: `re.match` with `$` accepts one, so `"mygene2\n"` passed an
    alphabet that contains no newline — `fullmatch` is what closes it.
    """
    from src.evaluation.cohort import validate_split_name

    with pytest.raises(ValueError, match="must be an identifier"):
        validate_split_name(escape)


def test_a_generated_split_cannot_be_aliased_by_case(tmp_path):
    """On a case-insensitive filesystem `TRAIN_samples.json` *is*
    `train_samples.json`; the identifier rule refuses the uppercase name before a
    path is built, whatever file is there."""
    from src.evaluation.cohort import GENERATED_SPLITS, resolve_cohort

    data_dir, _ = _allocated_workspace(tmp_path / "ws")
    for reserved in GENERATED_SPLITS:
        (data_dir / f"{reserved.upper()}_samples.json").write_text("[]")
        with pytest.raises(ValueError, match="must be an identifier"):
            resolve_cohort(data_dir, reserved.upper())


@pytest.mark.parametrize(
    "role", ["mygene2", "institutional_acceptance", "udn.v2", "ddd-2019", "test"]
)
def test_arbitrary_institutional_roles_remain_reachable(tmp_path, role):
    """The alphabet is bounded rather than the values enumerated, which is what
    keeps `mygene2` and `institutional_acceptance` distinct rather than collapsing
    both into `test`."""
    from src.evaluation.cohort import validate_split_name

    assert validate_split_name(role) == role


def test_a_loose_cohort_file_in_the_workspace_is_refused_and_never_read(tmp_path):
    """A `test_samples.json` dropped beside the generated splits is what the
    retired supplied mode read: integer indices with nothing saying which graph
    they index. It is not a split of the workspace, so naming it is refused
    before anything is read, with a pointer to the registry."""
    from tests.fixtures.generated_workspace import one_sample_per_disease, profiles_for
    from tests.fixtures.opens import count_opens

    data_dir, _ = _allocated_workspace(tmp_path / "ws")
    loose = data_dir / "test_samples.json"
    loose.write_text(json.dumps(one_sample_per_disease("test", [1], profiles_for([1]))))

    with count_opens() as opens, pytest.raises(SystemExit) as refused:
        _run("audit_split_overlap", [
            "--data-dir", str(data_dir), "--output", str(tmp_path / "m4.json"),
            "--splits", "train", "test",
        ])
    assert "not one of the generator's splits" in str(refused.value)
    assert "--registry" in str(refused.value)
    assert opens.opens(loose) == 0
    assert not (tmp_path / "m4.json").exists()


@pytest.mark.parametrize("kind", ["generated", "registered"])
def test_the_overlap_report_records_the_digests_of_the_bytes_it_read(
    tmp_path, monkeypatch, kind, registered
):
    """S9's dynamic pin for the overlap audit: every path hash is poisoned, and
    the report still records the digest of each split's one read, from which the
    verifier compared and the counts were taken."""
    from tests.fixtures.path_forms import POISON, poison_path_forms
    from tests.fixtures.registered_cohort import ROLE

    out = tmp_path / "m4.json"
    if kind == "registered":
        data_dir = registered.data_dir
        files = {"train_samples": data_dir / "train_samples.json",
                 f"{ROLE}_samples": _version_dir(registered.registry) / "samples.json"}
        poison_path_forms(monkeypatch)
        _registered(registered, out)
    else:
        data_dir, _ = _allocated_workspace(tmp_path / "ws")
        files = {f"{split}_samples": data_dir / f"{split}_samples.json"
                 for split in ("train", "val")}
        poison_path_forms(monkeypatch)
        _run("audit_split_overlap", ["--data-dir", str(data_dir), "--output", str(out)])

    report = json.loads(out.read_text())
    digests = report["digests"]
    assert POISON not in digests.values()
    assert digests == {
        key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in files.items()
    }
    if kind == "registered":
        assert POISON not in report["cohort_version"].values()


def test_the_overlap_audit_opens_each_file_once(tmp_path):
    from tests.fixtures.opens import count_opens

    data_dir, _ = _allocated_workspace(tmp_path / "ws")

    with count_opens() as opens:
        _run("audit_split_overlap", ["--data-dir", str(data_dir),
                                     "--output", str(tmp_path / "m4.json")])

    for name in ("train_samples.json", "val_samples.json", "split_manifest.json"):
        assert opens.opens(data_dir / name) == 1, name


def test_a_registered_audit_opens_each_file_once_and_val_never(registered, tmp_path):
    """The version manifest, the version's samples, the workspace manifest and
    train are each opened once; generated val is not an input."""
    from tests.fixtures.opens import count_opens

    data_dir = registered.data_dir
    version_dir = _version_dir(registered.registry)

    with count_opens() as opens:
        _registered(registered, tmp_path / "m4.json")

    for path in (version_dir / "cohort.json", version_dir / "samples.json",
                 data_dir / "split_manifest.json", data_dir / "train_samples.json"):
        assert opens.opens(path) == 1, path.name
    assert opens.opens(data_dir / "val_samples.json") == 0


def test_an_existing_report_is_replaced_whole_under_overwrite(tmp_path):
    data_dir, _ = _allocated_workspace(tmp_path / "ws")
    out = tmp_path / "m4.json"
    out.write_text("{}")

    _run("audit_split_overlap",
         ["--data-dir", str(data_dir), "--output", str(out), "--overwrite"])

    assert json.loads(out.read_text())["fact"] == "M4"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m4.json", "ws"]


def test_a_report_that_cannot_be_published_leaves_the_old_one_whole(tmp_path, monkeypatch):
    """Published through a staged file renamed over the target, so a failure at
    the rename leaves the report that was there, and no staged file beside it.
    A direct write would have truncated it first."""
    import os

    data_dir, _ = _allocated_workspace(tmp_path / "ws")
    out = tmp_path / "m4.json"
    out.write_text("{}")

    def refuse(src, dst):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(SystemExit, match="could not be written.*left as it was"):
        _run("audit_split_overlap",
             ["--data-dir", str(data_dir), "--output", str(out), "--overwrite"])

    assert out.read_text() == "{}"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m4.json", "ws"]


def test_the_manifest_section_carries_no_identifiers(tmp_path):
    """§5.2 applies to every section of the artifact, including the new one."""
    data_dir, _ = _allocated_workspace(tmp_path / "ws")
    out = tmp_path / "m4.json"

    _run("audit_split_overlap", ["--data-dir", str(data_dir), "--output", str(out)])
    text = out.read_text()

    assert "SECRET" not in text
    assert "patient_id" not in text
    section = json.dumps(json.loads(text)["manifest_verification"])
    assert "phenotype_ids" not in section and "disease_id" not in section
