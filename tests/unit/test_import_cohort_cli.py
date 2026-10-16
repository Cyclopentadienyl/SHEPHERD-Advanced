"""
The importer's command line: a preview that writes nothing, a commit that repeats it.
====================================================================================
`scripts/import_cohort.py` (contract M4, §2.2). What is pinned here:

- the preview prints every count with its denominator, each exclusion by case
  id and reasons, the unmappable terms, what the version binds and the policy,
  then a preview id and the exact commit command -- and writes nothing;
- the printed command, run as printed, publishes the version and prints its
  directory and manifest digest;
- every refusal exits non-zero with one message and no traceback, says nothing
  was written, and leaves the registry as it was;
- no case's phenotypes or disease are printed: case ids and aggregate counts
  only (D7).

Module: tests/unit/test_import_cohort_cli.py
"""
from __future__ import annotations

import hashlib
import json
import re
import shlex
from pathlib import Path

import pytest

from scripts import import_cohort
from src.evaluation.cohort_registry import REPO_ROOT
from tests.fixtures.registered_cohort import (
    UNKNOWN_HPO,
    UNKNOWN_HPO_2,
    UNKNOWN_MONDO,
    graph_source_cases,
    write_demo_workspace,
    write_source,
)


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("import_cli")
    data_dir = write_demo_workspace(root / "workspace")
    source = write_source(root / "cases.json", graph_source_cases(data_dir))
    return data_dir, source


@pytest.fixture
def registry(tmp_path):
    path = tmp_path / "registry"
    path.mkdir()
    return path


def _args(registry, data_dir, source, label="v1", role="demo", *extra):
    return ["--registry", str(registry), "--data-dir", str(data_dir), "--source", str(source),
            "--label", label, "--role", role, *extra]


def _tree(root: Path) -> dict:
    return {str(p.relative_to(root)): (p.read_bytes() if p.is_file() else None)
            for p in sorted(root.rglob("*"))}


def _preview_id(out: str) -> str:
    return re.search(r"^Preview id: ([0-9a-f]{16})$", out, re.MULTILINE).group(1)


def _assert_one_refusal(err: str) -> None:
    assert err.count("import_cohort:") == 1, err
    assert "Traceback" not in err
    assert err.startswith("import_cohort: refused:"), err


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------
def test_the_preview_prints_what_the_version_would_hold_and_writes_nothing(
        workspace, registry, capsys):
    data_dir, source = workspace
    before = _tree(data_dir)
    assert import_cohort.main(_args(registry, data_dir, source)) == 0
    out = capsys.readouterr().out

    assert "source cases      13" in out
    assert "ranked-eligible   10 of 13" in out
    assert "excluded          3 of 13" in out
    assert "ground_truth_absent: 2" in out and "phenotype_unmappable: 1" in out
    assert "case-11: ground_truth_absent (0 of 2 terms unmappable)" in out
    assert "case-13: ground_truth_absent, phenotype_unmappable (1 of 1 terms unmappable)" in out
    assert "case-12: phenotype_unmappable (1 of 1 terms unmappable)" in out
    assert "terms unmappable  2 distinct" in out
    assert f"{UNKNOWN_HPO}: listed by 2 case(s)" in out
    assert f"{UNKNOWN_HPO_2}: listed by 1 case(s)" in out
    assert "cases with lost terms  1 of 10 ranked" in out
    assert "repeats removed   1 in 1 case(s)" in out
    assert "partial phenotypes  keep" in out
    for role in ("kg", "node_features", "edge_indices", "num_nodes"):
        assert re.search(rf"^    {role}\s+[0-9a-f]{{64}}$", out, re.MULTILINE), role
    assert re.search(r"dataset record    origin synthetic, sha256 [0-9a-f]{64}", out)
    assert "HPO             not recorded" in out and "MONDO           not recorded" in out
    assert "SYNTHETIC WORKSPACE" in out

    pid = _preview_id(out)
    assert "Nothing was written" in out
    assert f"--partial-phenotypes keep --commit {pid}" in out
    assert list(registry.iterdir()) == []
    assert _tree(data_dir) == before


def test_the_preview_prints_case_ids_and_no_case_data(workspace, registry, capsys):
    data_dir, source = workspace
    import_cohort.main(_args(registry, data_dir, source))
    out = capsys.readouterr().out
    cases = json.loads(source.read_text())["cases"]
    mapped_terms = {term for case in cases for term in case["phenotypes"]} - {UNKNOWN_HPO,
                                                                               UNKNOWN_HPO_2}
    diseases = {case["disease"] for case in cases}
    assert not [term for term in mapped_terms if term in out]
    assert not [disease for disease in diseases if disease in out]
    assert UNKNOWN_MONDO not in out
    # Ranked case ids are not listed either; only the exclusions are, by id.
    assert "case-01" not in out and "case-11" in out


def test_the_exclude_policy_is_previewed_and_recorded_in_the_command(workspace, registry,
                                                                     capsys):
    data_dir, source = workspace
    assert import_cohort.main(_args(registry, data_dir, source, "v1", "demo",
                                    "--partial-phenotypes", "exclude")) == 0
    out = capsys.readouterr().out
    assert "ranked-eligible   9 of 13" in out
    assert "case-10: phenotype_unmappable (1 of 2 terms unmappable)" in out
    assert "--partial-phenotypes exclude --commit" in out


# ---------------------------------------------------------------------------
# Commit
# ---------------------------------------------------------------------------
def test_the_printed_command_commits_the_version(workspace, registry, capsys, monkeypatch):
    data_dir, source = workspace
    import_cohort.main(_args(registry, data_dir, source))
    out = capsys.readouterr().out
    command = shlex.split(out.strip().splitlines()[-1])
    assert command[:2] == ["python", "scripts/import_cohort.py"]

    assert import_cohort.main(command[2:]) == 0
    out = capsys.readouterr().out
    manifest_path = registry / "versions" / "v1" / "cohort.json"
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert "Committed cohort version v1 (role demo): 10 ranked of 13 source cases" in out
    assert f"directory        {manifest_path.parent.resolve()}" in out
    assert f"manifest sha256  {digest}" in out
    assert json.loads(manifest_path.read_text())["imported"]["at"].endswith("Z")


def test_a_commit_whose_inputs_changed_is_refused(workspace, registry, capsys, tmp_path):
    data_dir, source = workspace
    import_cohort.main(_args(registry, data_dir, source))
    pid = _preview_id(capsys.readouterr().out)
    changed = tmp_path / "cases.json"
    cases = json.loads(source.read_text())["cases"]
    write_source(changed, cases[:-1])

    assert import_cohort.main(_args(registry, data_dir, changed, "v1", "demo",
                                    "--commit", pid)) == 1
    captured = capsys.readouterr()
    _assert_one_refusal(captured.err)
    assert "inputs changed since the preview" in captured.err
    assert "Nothing was written" in captured.err
    assert captured.out == ""
    assert list(registry.iterdir()) == []


def test_a_commit_with_a_different_policy_than_previewed_is_refused(workspace, registry,
                                                                    capsys):
    data_dir, source = workspace
    import_cohort.main(_args(registry, data_dir, source))
    pid = _preview_id(capsys.readouterr().out)
    assert import_cohort.main(_args(registry, data_dir, source, "v1", "demo",
                                    "--partial-phenotypes", "exclude", "--commit", pid)) == 1
    assert "inputs changed since the preview" in capsys.readouterr().err
    assert list(registry.iterdir()) == []


def test_a_taken_label_is_refused_naming_the_version(workspace, registry, capsys):
    data_dir, source = workspace
    import_cohort.main(_args(registry, data_dir, source))
    pid = _preview_id(capsys.readouterr().out)
    assert import_cohort.main(_args(registry, data_dir, source, "v1", "demo",
                                    "--commit", pid)) == 0
    capsys.readouterr()
    before = _tree(registry)
    for argv in (_args(registry, data_dir, source),
                 _args(registry, data_dir, source, "v1", "demo", "--commit", pid)):
        assert import_cohort.main(argv) == 1
        err = capsys.readouterr().err
        _assert_one_refusal(err)
        assert "already exists" in err and str(registry.resolve() / "versions" / "v1") in err
    assert _tree(registry) == before


def test_an_interrupted_import_is_named_for_the_operator_to_remove(workspace, registry,
                                                                   capsys):
    data_dir, source = workspace
    (registry / "versions" / "v1").mkdir(parents=True)
    assert import_cohort.main(_args(registry, data_dir, source)) == 1
    err = capsys.readouterr().err
    _assert_one_refusal(err)
    assert "interrupted import" in err and "remove the directory" in err
    assert (registry / "versions" / "v1").is_dir()


def test_a_failed_write_is_reported_without_a_traceback(workspace, registry, capsys,
                                                        monkeypatch):
    from src.utils import file_modes

    data_dir, source = workspace
    import_cohort.main(_args(registry, data_dir, source))
    pid = _preview_id(capsys.readouterr().out)
    real = file_modes.publish_bytes

    def fail_on_manifest(target, data):
        if Path(target).name == "cohort.json":
            raise OSError(28, "No space left on device")
        return real(target, data)

    monkeypatch.setattr(file_modes, "publish_bytes", fail_on_manifest)
    assert import_cohort.main(_args(registry, data_dir, source, "v1", "demo",
                                    "--commit", pid)) == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err and "No space left on device" in err
    assert "was not committed" in err and "interrupted import" in err
    assert not (registry / "versions" / "v1" / "cohort.json").exists()


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------
def _malformed(tmp_path):
    return write_source(tmp_path / "bad.json", [{"case_id": "c1", "phenotypes": ["HP:1"],
                                                 "disease": "MONDO:0011073"}])


def _omim(tmp_path):
    return write_source(tmp_path / "omim.json", [{"case_id": "c1",
                                                  "phenotypes": ["HP:0001250"],
                                                  "disease": "OMIM:607208"}])


@pytest.mark.parametrize("change, expected", [
    (lambda reg, ws, src, tmp: {"registry": "relative/registry"}, "not an absolute path"),
    (lambda reg, ws, src, tmp: {"registry": REPO_ROOT / "data" / "no-such-registry"},
     "inside the repository"),
    (lambda reg, ws, src, tmp: {"registry": tmp / "missing"}, "does not exist"),
    (lambda reg, ws, src, tmp: {"source": _malformed(tmp)}, "phenotypes[0] 'HP:1'"),
    (lambda reg, ws, src, tmp: {"source": _omim(tmp)}, "OMIM: 1"),
    (lambda reg, ws, src, tmp: {"source": tmp / "absent.json"}, "No such file"),
    (lambda reg, ws, src, tmp: {"role": "train"}, "generator's own split names"),
    (lambda reg, ws, src, tmp: {"label": "../escape"}, "label is refused"),
    (lambda reg, ws, src, tmp: {"data_dir": tmp}, "split_manifest.json"),
])
def test_a_refusal_is_one_message_and_writes_nothing(change, expected, workspace, registry,
                                                     tmp_path, capsys):
    data_dir, source = workspace
    given = {"registry": registry, "data_dir": data_dir, "source": source, "label": "v1",
             "role": "demo"}
    given.update(change(registry, data_dir, source, tmp_path))
    argv = _args(given["registry"], given["data_dir"], given["source"], given["label"],
                 given["role"])
    assert import_cohort.main(argv) == 1
    captured = capsys.readouterr()
    _assert_one_refusal(captured.err)
    assert expected in captured.err
    assert "Nothing was written" in captured.err
    assert "Preview id" not in captured.out
    assert list(registry.iterdir()) == []
    assert not (REPO_ROOT / "data" / "no-such-registry").exists()


def test_a_source_nothing_of_which_maps_previews_but_offers_no_commit(workspace, registry,
                                                                      tmp_path, capsys):
    data_dir, _ = workspace
    source = write_source(tmp_path / "none.json", [
        {"case_id": "x1", "phenotypes": [UNKNOWN_HPO], "disease": "MONDO:0011073"},
    ])
    assert import_cohort.main(_args(registry, data_dir, source)) == 1
    captured = capsys.readouterr()
    assert "ranked-eligible   0 of 1" in captured.out
    assert "--commit" not in captured.out
    _assert_one_refusal(captured.err)
    assert "nothing to commit" in captured.err
    assert list(registry.iterdir()) == []
