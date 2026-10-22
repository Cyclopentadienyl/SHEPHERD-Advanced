"""The deployment probe's M2 acceptance (phases M and R), on CPU.

On the GB10 these run with `--m2-acceptance --device cuda`; here the phase
functions run on CPU over the demo graph, so the production entry points they
drive -- the trainer, the measurement CLI, the serving build, the shortest-path
producer and the inventory -- are exercised end to end, and every probe in each
phase must pass. Also: the acceptance is complete only when every required probe
of both parts ran and passed, the CLI cannot turn the acceptance into a CPU run,
and a snapshot that includes directories sees an empty one left behind.

Contract M4's flow rides on the same phases (T1-T6 after M11, RT1-RT3 before
R8): the importer, the measurement on a registered version with --record,
record_evaluation, the reload route and the read-only view, and the overlap
audit, each through its entry point. The tests check the probes' evidence
against what those entry points left on disk, that the report carries no path,
and that a registry inside the repository fails T1 in the probe's own words.
The flow uses the probe's registry as the deployment's whatever the shell
names: T5 and RT3 see their record uncontested (D5) from a fresh shell, and
phase R also from the operator's shell, whose directories stay untouched.

M5's acceptance rides on them too (W1-W11 after T6, RW1-RW3 before R8): the
service started as the launcher starts it, in M1 and read-only, the job runner
running the real CLIs on CPU, the GPU lock. The phase tests require every one to
pass and check its evidence against the disk; the helpers' own checks are shown
to fire in tests/unit/test_probe_m5_acceptance.py.

Module: tests/unit/test_probe_m2_phase.py
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _four_sources(directory):
    """Stand-ins for the four files a real build consumes, written and digested.

    **A stand-in, said as one.** The probe's real-data phase requires a record
    from files (R1); this test's graph is the in-memory demo, so no real build
    produced it. The files named here exist and their digests are theirs; that
    they made the graph is the claim a stand-in cannot make.
    """
    import hashlib

    from src.kg.provenance import SOURCE_ROLES, source_entry

    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for role in SOURCE_ROLES:
        path = directory / f"{role}.stand-in"
        path.write_text(f"stand-in for the {role} input\n")
        entries.append(source_entry(role, path, hashlib.sha256(path.read_bytes()).hexdigest()))
    return entries


def _probe():
    spec = importlib.util.spec_from_file_location(
        "probe_deployment", REPO / "scripts" / "probe_deployment.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _statuses(report, prefix):
    return {p.probe_id: (p.status, p.detail) for p in report.probes
            if p.probe_id.startswith(prefix)}


M4_DEMO_PROBES = ["T1", "T2", "T3", "T4", "T5", "T6"]
M4_REAL_PROBES = ["RT1", "RT2", "RT3"]


@pytest.fixture
def served_state(monkeypatch):
    """T5 and RT3 reload into the service's shared state and read it through the
    application; this puts the served fields and the recorded server address
    back after the test (the flags are the conftest's)."""
    import src.api.main as api
    import src.utils.server_address as server_address

    for name in ("pipeline", "kg", "_current_data_dir", "_current_checkpoint_path"):
        monkeypatch.setattr(api.app_state, name, getattr(api.app_state, name, None),
                            raising=False)
    monkeypatch.setattr(server_address, "_address", server_address._address)
    return api


def _ledger_records(directory):
    import json

    return json.loads((directory / "evaluations.json").read_text())["records"]


def test_the_probes_reload_is_bound_to_the_digest_it_expects(tmp_path, served_state,
                                                            monkeypatch):
    """The verification of EB-1 (R-6): the acceptance's reloads (T5, RT3) go
    through the precondition the dashboard's Load uses, so a GB10 run exercises
    it. The digest expected is served (the positive control); another is refused
    by the reload itself, before anything is published -- the pipeline served
    before stays the same object -- never published and then caught by the
    status's digest afterwards."""
    import hashlib

    from tests.fixtures.synthetic_workspace import build_workspace

    probe = _probe()
    for name in ("SHEPHERD_KG_PATH", "SHEPHERD_DATA_DIR", "SHEPHERD_CHECKPOINT_PATH"):
        monkeypatch.delenv(name, raising=False)
    data_dir, checkpoint = build_workspace(tmp_path / "ws")
    expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    served, answer = probe._serve_and_view(data_dir, checkpoint, expected, "cpu")
    assert served == expected and answer["checkpoint_sha256"] == expected
    serving = served_state.app_state.pipeline

    with pytest.raises(AssertionError, match="the reload naming the checkpoint was refused: "
                                             ".* is not the model that was shown"):
        probe._serve_and_view(data_dir, checkpoint, "a" * 64, "cpu")
    assert served_state.app_state.pipeline is serving


def test_the_m4_probes_are_required_in_both_parts():
    probe = _probe()
    assert set(M4_DEMO_PROBES) <= set(probe.M2_DEMO_REQUIRED)
    assert set(M4_REAL_PROBES) <= set(probe.M2_REAL_REQUIRED)
    assert len(set(probe.M2_DEMO_REQUIRED)) == len(probe.M2_DEMO_REQUIRED)
    assert len(set(probe.M2_REAL_REQUIRED)) == len(probe.M2_REAL_REQUIRED)
    report = probe.Report()
    for probe_id in probe.M2_DEMO_REQUIRED:
        report.probes.append(probe.Probe(probe_id, "m2", "x", status="passed"))
    report.probes = [p for p in report.probes if p.probe_id != "T5"]
    assert probe.m2_acceptance(report, False)["demo"]["not_passed"] == {"T5": "not run"}
    assert "M4" in probe.m2_acceptance(report, False)["scope"]


def test_the_demo_phase_passes_and_every_required_probe_runs(tmp_path, served_state):
    probe = _probe()
    report = probe.Report()
    sound = probe.phase_writer(report, tmp_path)
    checkpoint = probe.phase_training(report, tmp_path, sound, device="cpu", epochs=1)
    assert checkpoint is not None

    probe.phase_m2_demo(report, tmp_path, sound, checkpoint, device="cpu")

    phase = _statuses(report, "M")
    assert set(phase) == {p for p in probe.M2_DEMO_REQUIRED if p.startswith("M")}
    assert {k: v[0] for k, v in phase.items()} == dict.fromkeys(phase, "passed"), (
        {k: v[1] for k, v in phase.items() if v[0] != "passed"}
    )
    refused = tmp_path / "m2" / "refused"
    assert not refused.exists() or not any(refused.iterdir()), "a refused run left a directory"
    # The review's P2-2: M3 measures the checkpoint M2 resumed into, never its
    # parent in its place -- read from the measurement's own manifest.
    import json

    facts = {p.probe_id: p.facts for p in report.probes}
    measured = json.loads((tmp_path / "m2" / "measured" / "measurement.json").read_text())
    child = facts["M2"]["child_sha256"]
    assert measured["manifest"]["artifact_digests"]["checkpoint"] == child
    assert facts["M3"]["checkpoint_sha256"] == child != facts["M2"]["parent_sha256"]
    # On the CPU no mixed precision takes effect, and the report says so for
    # each continued run; on CUDA the restore is checked (scaler_restore).
    for probe_id in ("M2", "M9"):
        assert facts[probe_id]["scaler_restore"].startswith("no gradient scaler")
    # Contract M3c: M4's diagnosis names the resumed checkpoint by the digest
    # M2 recorded, the table verified, the synthetic record, the device asked
    # for -- and its evidence holds no path. These are the probe's own facts, so
    # they hold with its checks removed; that the checks in `_serve_and_diagnose`
    # (M4, R6) fire is shown in test_probe_serving_phase.py.
    identity = facts["M4"]["served_identity"]
    assert identity["checkpoint"] == {"state": "verified", "sha256": child}
    assert identity["scoring_mode"] == "gnn_plus_shortest_path"
    assert identity["shortest_paths"]["state"] == "verified"
    assert identity["dataset_record"]["origin"] == "synthetic"
    assert (identity["settings"]["device"], identity["settings"]["device_requested"]) == (
        "cpu", "cpu")
    assert str(tmp_path) not in json.dumps(identity)

    # ---- Contract M4's flow (T1-T6), after M11, every probe passed.
    m4 = _statuses(report, "T")
    assert list(m4) == M4_DEMO_PROBES
    assert {k: v[0] for k, v in m4.items()} == dict.fromkeys(m4, "passed"), (
        {k: v[1] for k, v in m4.items() if v[0] != "passed"}
    )
    # The evidence holds no path (the work directory, the workspace, a report).
    assert str(tmp_path) not in json.dumps({k: facts[k] for k in M4_DEMO_PROBES})
    # T1: the counts the probe's source was built to give, and the version on
    # disk under the work directory is the one the facts name.
    import hashlib

    registry = tmp_path / "m4" / "registry"
    version = registry / "versions" / probe.M4_DEMO_LABEL
    assert facts["T1"]["counts"] == probe.M4_DEMO_COUNTS
    assert facts["T1"]["manifest_sha256"] == hashlib.sha256(
        (version / "cohort.json").read_bytes()).hexdigest()
    # T2: the one report kept for D1's weights is what the one record names,
    # beside the copy of D1's checkpoint (the bytes D1 trained), and R10's
    # evidence was found. (W8's run keeps another, for M2's resumed weights;
    # a run directory is named for the checkpoint it measured.)
    d1_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (kept,) = [path for path in (version / "reports").glob("*/report.json")
               if f"-{d1_sha[:12]}-" in path.parent.name]
    checkpoints = tmp_path / "m4" / "checkpoints"
    (record,) = _ledger_records(checkpoints)
    assert record["source_artifact_digest"] == facts["T2"]["report_sha256"] == hashlib.sha256(
        kept.read_bytes()).hexdigest()
    assert record["checkpoint_digest"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert (checkpoints / "measured.pt").read_bytes() == checkpoint.read_bytes()
    for scope in ("training", "validation"):
        assert facts["T2"]["r10"][scope]["evidence"] == "found"
        assert facts["T2"]["r10"][scope]["disease_overlap"]["k"] is not None
    # T3 and T4 wrote nothing: still the one record, and no report for the
    # version mapped against M1's graph (its import itself is on disk).
    assert facts["T3"]["same_report_again"] == "already_registered"
    assert len(_ledger_records(checkpoints)) == 1
    assert (registry / "versions" / probe.M4_OTHER_GRAPH_LABEL / "cohort.json").is_file()
    assert not (registry / "versions" / probe.M4_OTHER_GRAPH_LABEL / "reports").exists()
    assert "kg" in facts["T4"]["differing_roles"]
    # T5: the record for the measured checkpoint, uncontested in the probe's
    # registry (this shell names none: root-cause round, INT-1), none for M2's
    # beside it.
    assert facts["T5"]["measured"]["state"] == "records"
    assert facts["T5"]["measured"]["contest_state"] == "uncontested"
    assert facts["T5"]["measured"]["checkpoint_sha256"] == record["checkpoint_digest"]
    assert facts["T5"]["other"]["state"] == "no_records"
    assert facts["T5"]["other"]["checkpoint_sha256"] == child
    # T6: the audit recorded the version the import committed, and its shared
    # diseases are R10's training disease overlap: two production computations
    # over the same train bytes and the same version samples.
    assert facts["T6"]["cohort_version"]["manifest_sha256"] == facts["T1"]["manifest_sha256"]
    training = facts["T2"]["r10"]["training"]["disease_overlap"]
    assert facts["T6"]["overlap"] == f"{training['k']} of {training['n']}"

    # ---- M5's acceptance (W1-W11), after T6, every probe passed, and its
    # evidence holds no path.
    m5 = _statuses(report, "W")
    assert list(m5) == list(probe.M5_DEMO_PROBES)
    assert {k: v[0] for k, v in m5.items()} == dict.fromkeys(m5, "passed"), (
        {k: v[1] for k, v in m5.items() if v[0] != "passed"}
    )
    assert str(tmp_path) not in json.dumps({k: facts[k] for k in probe.M5_DEMO_PROBES})
    # W1: the probe's own deployment; the listed checkpoint is a copy of M2's
    # resumed weights, named by the digest of its bytes.
    workspace = tmp_path / "m5" / "workspaces" / probe.M5_WORKSPACE
    listed = (workspace / "runs" / "gat" / probe.M5_LISTED_RUN / "checkpoints"
              / probe.M5_CHECKPOINT)
    assert hashlib.sha256(listed.read_bytes()).hexdigest() == child
    assert facts["W1"]["checkpoints"][0] == {"run": probe.M5_LISTED_RUN, "filename": "last.pt",
                                             "sha256": child, "listed_with_a_problem": False}
    assert facts["W1"]["workspaces"][0]["dataset_fingerprint"] == hashlib.sha256(
        (workspace / "split_manifest.json").read_bytes()).hexdigest()
    # W2 and W5: every request shape was decided by the condition named for it,
    # on every gated API route (D8's six) and the three gated dashboard events.
    assert sorted({seen["condition"] for seen in facts["W2"]["requests"].values()}) == [
        "cross_site", "enabled", "forwarding_header", "host_not_loopback", "origin_mismatch"]
    operator = sorted(f"{method} {path}" for method, path, _ in probe.OPERATOR_ROUTES)
    assert facts["W2"]["gated"]["api_routes"] == operator
    assert len(facts["W2"]["gated"]["gradio_events"]) == 3
    for seen in (*facts["W2"]["requests"].values(), facts["W5"]["loopback_request"]):
        expected = 403 if seen["condition"] != "enabled" else None
        assert sorted(seen["routes"]) == operator
        assert expected is None or set(seen["routes"].values()) == {403}
        assert len(seen["gradio_events"]) == 3
    assert facts["W5"]["loopback_request"]["condition"] == "deployment_mode"
    # W4: no file= URL request reached Gradio's URL fetch; the control past the
    # guard did, so the count is where a fetch happens.
    assert (facts["W4"]["url_fetches_through_the_guard"],
            facts["W4"]["url_fetch_control_past_the_guard"]) == (0, 1)
    # W1: the deployment's own lock (none configured here) was read and stated.
    assert facts["W1"]["deployment_gpu_lock"]["held"] is False
    # W8: a training stop while the run ran was refused, and the run completed.
    assert facts["W8"]["training_stop_while_it_ran"].startswith("refused")
    # Both services were stopped (uvicorn's own last words in each log).
    for mode in ("m1", "read-only"):
        assert "Finished server process" in (
            tmp_path / "m5" / f"service-{mode}" / "server.log").read_text()
    # W8 and W9: the run's report is kept in T1's version, and the one record
    # beside the listed checkpoint is built from those bytes; the usage history
    # holds the run and its registration. W10 left no third report.
    reports = sorted((version / "reports").glob("*/report.json"))
    assert len(reports) == 2
    (w8_report,) = [path for path in reports if f"-{child[:12]}-" in path.parent.name]
    assert hashlib.sha256(w8_report.read_bytes()).hexdigest() == facts["W8"]["report_sha256"]
    (w9_record,) = _ledger_records(listed.parent)
    assert w9_record["source_artifact_digest"] == facts["W8"]["report_sha256"]
    usage = [json.loads(line) for line in
             (version / "reports" / "usage.jsonl").read_text().splitlines()]
    assert [entry["kind"] for entry in usage
            if entry.get("run_id") == w8_report.parent.name] == ["test_run", "registered"]
    # W11: stopped, never completed; the lock is the probe's, under its root.
    assert facts["W11"]["stop"]["status"] == "stopped"
    assert (tmp_path / "m5" / "workspaces" / ".shepherd-gpu.lock").is_file()


def _operator_shell(monkeypatch, base):
    """The deployment's settings as the shell the guide prepares holds them
    (§4.11): its registry and its workspace root, which names the lock, each
    an empty directory under `base`."""
    (base / "registry").mkdir(parents=True)
    (base / "workspaces").mkdir()
    monkeypatch.setenv("SHEPHERD_REGISTRY", str(base / "registry"))
    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(base / "workspaces"))


@pytest.mark.parametrize("shell", ["fresh", "operator"])
def test_the_real_phase_passes_and_leaves_the_workspace_as_it_was(
        tmp_path, served_state, monkeypatch, tmp_path_factory, shell):
    """Phase R end to end, from a fresh shell (no registry named) and from the
    operator's shell (the deployment's registry, root and lock named, BACKLOG
    37 and the guide's §4.11). Either way M4's flow uses the probe's registry
    as the deployment's, so RT3 sees its record uncontested, and the
    deployment's directories are not touched (root-cause round, INT-1)."""
    import shutil

    import yaml

    if shell == "operator":
        operator = tmp_path_factory.mktemp("operator")
        _operator_shell(monkeypatch, operator)
    probe = _probe()
    from src.kg.workspace import write_workspace
    from src.utils.checkpoint_io import read_checkpoint

    root = tmp_path / "workspaces"
    real = root / "ws"
    # A real workspace's record is from files, every input identified (R1,
    # contract M3b); the stand-in names four files, as a real build would.
    write_workspace(probe._demo_kg(), real, feature_dim=probe.FEATURE_DIM,
                    samples=probe._budget(), sources=_four_sources(tmp_path / "inputs"))
    produced = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "compute_shortest_paths.py"),
         "--kg-path", str(real / "kg.json"), "--output-dir", str(real),
         "--max-hops", "5", "--workers", "1"],
        cwd=REPO, capture_output=True, text=True, timeout=600,
    )
    assert produced.returncode == 0, produced.stderr[-2000:]
    # The reference: a model of this architecture that records nothing, as the
    # models trained before the contract are.
    trained = probe._last(probe._train_run(real, tmp_path / "reference_run", "cpu", 1,
                                           settings=probe._DEMO_TRAINING))
    contents = read_checkpoint(trained, map_location="cpu", weights_only=True).checkpoint
    del contents["training_input_digests"]
    reference = real / "checkpoints" / "gat" / "model-ref.pt"
    reference.parent.mkdir(parents=True)
    import torch

    torch.save(contents, reference)
    shutil.rmtree(tmp_path / "reference_run")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(dict(probe._DEMO_TRAINING)))

    report = probe.Report()
    work = tmp_path / "work"
    probe.phase_m2_real(report, work, real, config, reference, root, device="cpu")

    phase = _statuses(report, "R")
    assert set(phase) == set(probe.M2_REAL_REQUIRED)
    assert {k: v[0] for k, v in phase.items()} == dict.fromkeys(phase, "passed"), (
        {k: v[1] for k, v in phase.items() if v[0] != "passed"}
    )
    facts = {p.probe_id: p.facts for p in report.probes}
    assert facts["R6"]["sp_kg_binding"] == "verified"
    # Contract M3c: R6's diagnosis names the checkpoint R4 trained and the
    # record from files R1 verified.
    identity = facts["R6"]["served_identity"]
    assert identity["checkpoint"]["sha256"] == facts["R4"]["child_sha256"]
    assert identity["graph"]["kg_sha256"] == facts["R1"]["kg_digest"]
    assert identity["dataset_record"] == {
        "state": "verified", "origin": "files",
        "sha256": facts["R1"]["dataset_record"]["sha256"]}
    assert facts["R7"]["reference_refusal"] == "no record"
    assert facts["R4"]["child_optimizer_step"] > facts["R4"]["parent_optimizer_step"]

    # ---- Contract M4's flow on the (stand-in) real workspace: RT1-RT3 run
    # after R7 and before R8, whose comparison therefore covers them.
    order = [p.probe_id for p in report.probes]
    assert order[order.index("R7") + 1:] == [*M4_REAL_PROBES, *probe.M5_REAL_PROBES, "R8"]
    import json

    assert str(tmp_path) not in json.dumps({k: facts[k] for k in M4_REAL_PROBES})
    n = facts["RT1"]["val_cases_written"]
    assert n == min(probe.M4_REAL_CASES, probe.NUM_VAL)        # the demo names every node
    assert facts["RT1"]["counts"]["n_excluded"] == 0
    r10 = facts["RT2"]["r10"]
    assert r10["training"]["disease_overlap"]["state"] == "none"
    assert r10["validation"]["phenotype_set_overlap"] == {"state": "overlap", "k": n, "n": n}
    # The ledger is beside R4's checkpoint, under the work directory; the
    # registry and its report too. Nothing of M4's is in the workspace.
    resumed = work / "real" / "resumed" / "checkpoints"
    (record,) = _ledger_records(resumed)
    assert record["checkpoint_digest"] == facts["R4"]["child_sha256"]
    assert record["source_artifact_digest"] == facts["RT2"]["report_sha256"]
    assert facts["RT3"]["state"] == "records"
    assert facts["RT3"]["checkpoint_sha256"] == facts["R4"]["child_sha256"]
    assert facts["RT3"]["contest_state"] == "uncontested"
    assert not list(real.rglob("evaluations.json")) and not (real / "registry").exists()

    # ---- M5's acceptance on the (stand-in) real workspace, before R8: listed
    # by its manifest's digest; RW2's record beside R3's checkpoint, under the
    # work directory; RW3 stopped. The lock is the probe's own, under a
    # workspaces root of its own in the work directory, never the one under
    # the deployment's root.
    import hashlib

    assert str(tmp_path) not in json.dumps({k: facts[k] for k in probe.M5_REAL_PROBES})
    assert facts["RW1"]["dataset_fingerprint"] == hashlib.sha256(
        (real / "split_manifest.json").read_bytes()).hexdigest()
    assert facts["RW1"]["dataset_record"] == {"binding": "verified", "origin": "files"}
    (rw2_record,) = _ledger_records(work / "real" / "initial" / "checkpoints")
    assert rw2_record["source_artifact_digest"] == facts["RW2"]["report_sha256"]
    assert facts["RW2"]["training_stop_while_it_ran"].startswith("refused")
    assert facts["RW1"]["deployment_gpu_lock"]["held"] is False
    assert facts["RW3"]["stop"]["status"] == "stopped"
    assert (work / "real" / "m5" / "gpu-lock-root" / ".shepherd-gpu.lock").is_file()
    assert not list(root.glob(".shepherd-gpu.lock*"))
    if shell == "operator":
        # Nothing of the deployment's was written or created, and its settings
        # are as the shell had them.
        assert sorted(path.relative_to(operator).as_posix()
                      for path in operator.rglob("*")) == ["registry", "workspaces"]
        assert os.environ["SHEPHERD_REGISTRY"] == str(operator / "registry")


def test_an_architecture_other_than_the_reference_s_is_refused(tmp_path):
    """R2 compares the settings in effect, defaults included, with the reference."""
    import yaml

    probe = _probe()
    from src.kg.workspace import write_workspace

    root = tmp_path / "workspaces"
    real = root / "ws"
    write_workspace(probe._demo_kg(), real, feature_dim=probe.FEATURE_DIM,
                    samples=probe._budget())
    reference = probe._last(probe._train_run(real, real / "run", "cpu", 1,
                                             settings=probe._DEMO_TRAINING))
    settings = dict(probe._DEMO_TRAINING)
    del settings["num_heads"]                    # the default, 8, is not the reference's 4
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(settings))

    report = probe.Report()
    probe.phase_m2_real(report, tmp_path / "work", real, config, reference, root, device="cpu")

    r2 = {p.probe_id: p for p in report.probes}["R2"]
    assert r2.status == "failed" and "num_heads" in r2.detail
    assert {p.probe_id: p.status for p in report.probes}["R3"] == "failed"


def test_the_acceptance_is_complete_only_when_every_required_probe_passed():
    probe = _probe()
    report = probe.Report()
    for probe_id in probe.M2_DEMO_REQUIRED:
        report.probes.append(probe.Probe(probe_id, "m2", "x", status="passed"))
    demo_only = probe.m2_acceptance(report, real_requested=False)
    assert demo_only["demo"]["complete"] and not demo_only["complete"]

    for probe_id in probe.M2_REAL_REQUIRED[:-1]:
        report.probes.append(probe.Probe(probe_id, "real", "x", status="passed"))
    report.probes.append(probe.Probe(probe.M2_REAL_REQUIRED[-1], "real", "x", status="skipped"))
    partial = probe.m2_acceptance(report, real_requested=True)
    assert partial["real_data"]["not_passed"] == {probe.M2_REAL_REQUIRED[-1]: "skipped"}
    assert not partial["complete"]

    report.probes = [p for p in report.probes if p.probe_id != "M3"]
    assert probe.m2_acceptance(report, True)["demo"]["not_passed"]["M3"] == "not run"


def test_the_cli_cannot_make_the_acceptance_a_cpu_run_or_write_the_workspace(tmp_path):
    probe = _probe()
    real = tmp_path / "workspaces" / "ws"
    (real / "checkpoints").mkdir(parents=True)
    reference = real / "checkpoints" / "model.pt"
    reference.write_bytes(b"x")
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    work = ["--work-dir", str(tmp_path / "work")]
    real_args = ["--real-workspace", str(real), "--real-config", str(config),
                 "--real-reference-checkpoint", str(reference)]

    for argv in (
        [*work, "--m2-acceptance"],                                    # device auto
        [*work, "--m2-acceptance", "--device", "cpu"],
        [*work, "--device", "cuda", *real_args],                       # not an acceptance
        [*work, "--m2-acceptance", "--device", "cuda", "--real-workspace", str(real)],
        ["--work-dir", str(real / "w"), "--m2-acceptance", "--device", "cuda", *real_args],
        [*work, "--m2-acceptance", "--device", "cuda", *real_args,
         "--workspaces-root", str(tmp_path / "workspaces" / "ws" / "checkpoints" / "x")],
    ):
        with pytest.raises(SystemExit):
            probe.parse_args(argv)

    args = probe.parse_args([*work, "--m2-acceptance", "--device", "cuda", *real_args])
    assert args.workspaces_root == real.parent


def test_a_snapshot_with_directories_sees_an_empty_one(tmp_path):
    probe = _probe()
    (tmp_path / "a.txt").write_text("a")
    files_before = probe.snapshot(tmp_path)
    before = probe.snapshot(tmp_path, include_dirs=True)
    (tmp_path / "run" / "logs").mkdir(parents=True)

    assert probe.snapshot(tmp_path) == files_before                # files alone: blind
    with pytest.raises(AssertionError, match="created"):
        probe.assert_untouched(before, tmp_path, "x", include_dirs=True)


def test_a_registry_inside_the_repository_fails_the_import_in_the_probe_s_words(
    tmp_path, capsys
):
    """A --work-dir inside the repository puts M4's registry there: the importer
    refuses it, and T1's failure says so without the paths its message names.
    Nothing is created (the registry's location is refused before it exists)."""
    probe = _probe()
    registry = REPO / "tests" / "probe-registry-never-created"
    source = probe._write_cohort_source(tmp_path / "cases.json", [
        {"case_id": "c1", "phenotypes": ["HP:0001250"], "disease": "MONDO:0011073"}])

    with pytest.raises(AssertionError, match="inside the repository") as failed:
        probe._import_version(registry, tmp_path / "workspace", source, probe.M4_DEMO_LABEL)

    assert "--work-dir outside the repository" in str(failed.value)
    assert str(REPO) not in str(failed.value), "the failure carries a path"
    assert "inside the repository" in capsys.readouterr().err   # the importer's own message
    assert not registry.exists()


def test_a_relative_work_dir_outside_the_repository_reaches_the_importer_absolute(
    tmp_path, monkeypatch
):
    """A --work-dir given relative but outside the repository (`../m4-acceptance/
    work` from the checkout) is resolved in `main`, so what the phases derive from
    it is absolute -- M4's registry included, which the importer refuses as a
    relative path. Without that, T1 and RT1 failed after the training runs had
    been spent, and the report named no cause. Through `main`, on the CPU: the
    phases before T1 are stood in for, the workspace is written under the work
    directory where phase B writes it, and T1's import runs through
    `_import_version` on the registry path phase_m2_demo derives."""
    from tests.fixtures.registered_cohort import write_demo_workspace

    probe = _probe()
    monkeypatch.chdir(tmp_path)                 # outside the repository
    given = Path("acceptance") / "work"
    seen = {}

    def writer(report, work):
        seen["work"] = work
        return write_demo_workspace(work / "sound_workspace")

    def t1_import(report, work, sound, checkpoint, device):
        m4_dir = (work / "m2").parent / "m4"   # phase_m2_demo's derivation
        registry = m4_dir / "registry"
        registry.mkdir(parents=True)
        source = probe._write_cohort_source(m4_dir / "cases.json", probe._m4_demo_cases(sound))
        seen["version"], _ = probe._import_version(registry, sound, source,
                                                   probe.M4_DEMO_LABEL)

    monkeypatch.setattr(probe, "phase_environment",
                        lambda report, device, **kwargs: {"resolved_device": "cuda"})
    monkeypatch.setattr(probe, "phase_writer", writer)
    monkeypatch.setattr(probe, "phase_workspace", lambda *args: None)
    monkeypatch.setattr(probe, "phase_training", lambda *args: Path("not-trained.pt"))
    monkeypatch.setattr(probe, "phase_serving", lambda *args: None)
    monkeypatch.setattr(probe, "phase_m2_demo", t1_import)

    # Incomplete, as it must be: only T1's import ran.
    assert probe.main(["--work-dir", str(given), "--m2-acceptance", "--device", "cuda"]) == 1

    assert seen["work"] == tmp_path.resolve() / given
    assert seen["version"].manifest["counts"]["ranked_eligible"] > 0
    assert (given / "m4" / "registry" / "versions" / probe.M4_DEMO_LABEL).is_dir()
    assert (given / "probe_report.json").is_file()


def test_a_preview_that_writes_or_prints_no_id_is_caught(tmp_path, monkeypatch):
    """`_import_version`'s two checks fire: a preview that left something in the
    registry, and a preview that printed no id."""
    import scripts.import_cohort as import_cohort

    probe = _probe()
    registry = tmp_path / "registry"
    registry.mkdir()

    def writes(argv):
        (registry / "left-behind").write_text("x")
        print("Preview id: 0123456789abcdef")
        return 0

    monkeypatch.setattr(import_cohort, "main", writes)
    with pytest.raises(AssertionError, match="the import preview created"):
        probe._import_version(registry, tmp_path, tmp_path / "cases.json", "x")

    monkeypatch.setattr(import_cohort, "main", lambda argv: 0)
    with pytest.raises(AssertionError, match="printed no preview id"):
        probe._import_version(tmp_path / "empty", tmp_path, tmp_path / "cases.json", "x")


def test_the_m4_evidence_checks_fire(tmp_path):
    """The checks T2, T5, RT2 and RT3 rest on, each shown to fail: ranks that
    are not the bytes the report names, a ledger record from another report or
    a second record, and a view that shows another report's record or none, or
    the record with a D5 state other than uncontested (root-cause round,
    INT-1: before it, "cannot_be_checked" passed)."""
    import json
    from types import SimpleNamespace

    probe = _probe()
    run = tmp_path / "registry" / "versions" / "v" / "reports" / "run"
    run.mkdir(parents=True)
    (run / "ranks.json").write_text(json.dumps([{"sample_id": "a", "rank": 1}]))
    (run / "report.json").write_text(json.dumps(
        {"n_ranked": 1, "ranks": {"file": "ranks.json", "sha256": "0" * 64}}))
    with pytest.raises(AssertionError, match="not the bytes the report names"):
        probe._kept_report(tmp_path / "registry", "v")

    record = {"checkpoint_digest": "c", "cohort_digest": "v", "source_artifact_digest": "r",
              "corroborating_reports": [], "registration": {"sequence": 1},
              "cohort": {"label": "l", "role": "o"}, "n_ranked": 1}
    version = SimpleNamespace(identity=SimpleNamespace(sha256="v"))

    def ledger(*records):
        (tmp_path / "evaluations.json").write_text(
            json.dumps({"schema_version": 3, "records": list(records)}))

    ledger(record)
    assert probe._the_record(tmp_path, SimpleNamespace(sha256="r"), "c", version)[0] == record
    with pytest.raises(AssertionError, match="not built from the kept report's bytes"):
        probe._the_record(tmp_path, SimpleNamespace(sha256="another"), "c", version)
    ledger(record, dict(record, cohort_digest="w"))
    with pytest.raises(AssertionError, match="2 records, not one"):
        probe._the_record(tmp_path, SimpleNamespace(sha256="r"), "c", version)

    view = {"source_artifact_digest": "r", "registration": {"sequence": 1},
            "cohort": {"label": "l", "role": "o"}, "n_ranked": 1,
            "r10": {"validation": {"evidence": {"state": "found"}}},
            "environment_class": "CPU",
            "contest": {"state": "uncontested", "differing": [], "runs": []}}
    answer = {"state": "records", "checkpoint_sha256": "c", "ledger": {"sha256": "L"},
              "records": [view], "records_not_shown": 0}
    shown = probe._the_view_shows(answer, record, "L", ("validation",))
    assert (shown["records"], shown["contest_state"]) == (1, "uncontested")
    for broken, message in (
        (dict(answer, records=[dict(view, source_artifact_digest="x")]), "another report"),
        (dict(answer, state="no_records", records=[]), "state is no_records"),
        (dict(answer, ledger={"sha256": "M"}), "another ledger"),
        (dict(answer, records=[dict(view, r10={"validation": {"evidence": {"state":
                                                                         "not_found"}}})]),
         "as not found"),
        (dict(answer, records=[dict(view, contest={"state": "cannot_be_checked",
                                                   "differing": [], "runs": []})]),
         "D5 state as cannot_be_checked"),
        (dict(answer, records=[dict(view, contest={"state": "contested", "differing": ["mrr"],
                                                   "runs": []})]),
         "D5 state as contested"),
        (dict(answer, records=[{k: v for k, v in view.items() if k != "contest"}]),
         "D5 state as None"),
    ):
        with pytest.raises(AssertionError, match=message):
            probe._the_view_shows(broken, record, "L", ("validation",))


def test_a_checkpoint_read_is_observed_and_the_reader_put_back(tmp_path):
    """T4's observation: a read through the one checkpoint reader, imported when
    it runs as the measurement imports it, is noted; afterwards the reader is
    the original again."""
    import torch

    import src.utils.checkpoint_io as checkpoint_io

    probe = _probe()
    path = tmp_path / "tiny.pt"
    torch.save({"x": torch.zeros(1)}, path)
    original, reads = checkpoint_io.read_checkpoint, []
    with probe._observing_checkpoint_reads(reads):
        from src.utils.checkpoint_io import read_checkpoint

        read_checkpoint(path, map_location="cpu", weights_only=True)
    assert reads == ["tiny.pt"]
    assert checkpoint_io.read_checkpoint is original


@pytest.mark.parametrize("node, node_type, term", [
    ("hpo:HP:0001250", "phenotype", "HP:0001250"),
    ("mondo:MONDO:0011073", "disease", "MONDO:0011073"),
    ("mondo:MONDO:0011073", "phenotype", None),      # not the node an HP term maps to
    ("hpo:HP:12345", "phenotype", None),             # not a term the importer accepts
    ("orphanet:ORPHA:558", "disease", None),
])
def test_a_node_is_named_by_the_term_the_importer_maps_back_to_it(node, node_type, term):
    assert _probe()._node_term(node, node_type) == term


def test_without_cuda_the_acceptance_stops_after_the_device_check(tmp_path):
    """The entry point itself: `--device cuda` on a host with no CUDA fails A2,
    runs nothing after it on CPU, and reports every required probe as not run."""
    import json

    import torch

    if torch.cuda.is_available():
        pytest.skip("this host has CUDA; the case is a host without it")
    probe = _probe()
    work = tmp_path / "work"
    report_path = work / "m2_report.json"

    assert probe.main(["--work-dir", str(work), "--report", str(report_path),
                       "--m2-acceptance", "--device", "cuda"]) == 1

    report = json.loads(report_path.read_text())
    ran = {p["id"] for p in report["probes"]}
    assert not {p for p in ran if p[0] in "BCDEMRT"}, sorted(ran)
    assert {p["id"]: p["status"] for p in report["probes"]}["A2"] == "failed"
    acceptance = report["m2_acceptance"]
    assert acceptance["complete"] is False
    assert acceptance["demo"]["not_passed"]["D1"] == "not run"
    assert acceptance["real_data"] == {"requested": False, "complete": False}
    assert report["tested_commit"]["commit"]
    assert work.is_dir(), "the acceptance keeps its work directory"


def test_the_report_cannot_be_an_input_an_earlier_result_or_outside_the_work_dir(tmp_path):
    """The review's P2-1, before any work: the report of an acceptance stays
    inside its work directory, is no input and replaces no existing file."""
    probe = _probe()
    real = tmp_path / "workspaces" / "ws"
    (real / "checkpoints").mkdir(parents=True)
    reference = real / "checkpoints" / "model.pt"
    reference.write_bytes(b"reference")
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    work = tmp_path / "work"
    earlier = tmp_path / "earlier"
    earlier.mkdir()
    (earlier / "report.json").write_text("earlier")
    acceptance = ["--m2-acceptance", "--device", "cuda", "--real-workspace", str(real),
                  "--real-config", str(config), "--real-reference-checkpoint", str(reference)]

    for report, work_dir in (
        (reference, work), (config, work), (real / "report.json", work),
        (tmp_path / "beside.json", work),                      # outside the work dir
        (earlier / "report.json", earlier),                    # an earlier result
    ):
        with pytest.raises(SystemExit):
            probe.parse_args(["--work-dir", str(work_dir), "--report", str(report),
                              *acceptance])
    assert reference.read_bytes() == b"reference" and config.read_text() == "{}"
    assert (earlier / "report.json").read_text() == "earlier"

    args = probe.parse_args(["--work-dir", str(work), "--report", str(work / "m2.json"),
                             *acceptance])
    assert args.report == work / "m2.json"


def test_a_report_target_that_appears_during_the_run_is_not_overwritten(
    tmp_path, monkeypatch, capsys
):
    """The final write creates and never replaces: a file at the report's path
    when the run ends -- here put there while it ran -- keeps its bytes, the run
    fails, and the report goes to standard output instead."""
    import json

    probe = _probe()
    work = tmp_path / "work"
    report_path = work / "m2_report.json"
    real_environment = probe.phase_environment

    def environment_that_also_writes_the_target(report, device, **kwargs):
        report_path.write_text("written during the run")
        facts = real_environment(report, device, **kwargs)
        facts.pop("resolved_device", None)       # no CUDA: the acceptance stops after A
        return facts

    monkeypatch.setattr(probe, "phase_environment", environment_that_also_writes_the_target)

    assert probe.main(["--work-dir", str(work), "--report", str(report_path),
                       "--m2-acceptance", "--device", "cuda"]) == 3

    assert report_path.read_text() == "written during the run"
    out = capsys.readouterr()
    assert "is not replaced" in out.err
    start = out.out.index("\n{\n") + 1           # the payload's opening line
    assert json.loads(out.out[start:])["m2_acceptance"]["complete"] is False


def _scaler_step(scaler, parameter, optimizer, overflow=False):
    import torch

    optimizer.zero_grad()
    loss = parameter.square() * (float("inf") if overflow else 1.0)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()


def _side(scaler_state, updates):
    amp = {"enabled": True, "dtype": "float16", "grad_scaler": True}
    return {"amp": amp, "progress": {"optimizer_updates": updates},
            "scaler": {k: v for k, v in scaler_state.items() if isinstance(v, (int, float))}}


def test_a_correct_restore_is_not_refused_when_the_scale_falls_and_rises_again():
    """The reviewer's counterexample (review of 7e1e373), with real CPU
    GradScalers: restored correctly, then an overflow halves the scale and a
    full growth interval doubles it back, the tracker reset. The end values
    equal the parent's scale; they are recorded, not judged, and the restore,
    observed before the first update, is what is checked."""
    import torch

    probe = _probe()
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.01)
    parent = torch.amp.GradScaler("cpu", init_scale=8.0, growth_interval=4)
    _scaler_step(parent, parameter, optimizer)
    parent_state = parent.state_dict()

    child = torch.amp.GradScaler("cpu")
    child.load_state_dict(parent_state)
    observed = [{"parent": probe._scaler_numbers(parent_state),
                 "restored": probe._scaler_numbers(child.state_dict())}]
    _scaler_step(child, parameter, optimizer, overflow=True)
    for _ in range(4):
        _scaler_step(child, parameter, optimizer)
    end = child.state_dict()
    assert end["scale"] == parent_state["scale"] and end["_growth_tracker"] == 0

    facts = probe._scaler_restore_facts(_side(parent_state, 1), _side(end, 6), observed)
    assert facts["scaler_restore"].startswith("verified at restore")
    assert facts["end_values"] == "recorded, not judged"


def test_a_scaler_not_restored_is_caught_at_the_restore():
    import torch

    probe = _probe()
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.01)
    parent = torch.amp.GradScaler("cpu", init_scale=8.0)
    _scaler_step(parent, parameter, optimizer)
    parent_state = parent.state_dict()
    fresh = torch.amp.GradScaler("cpu", init_scale=8.0)    # the same scale, never loaded

    observed = [{"parent": probe._scaler_numbers(parent_state),
                 "restored": probe._scaler_numbers(fresh.state_dict())}]
    with pytest.raises(AssertionError, match="not restored from the parent's state"):
        probe._scaler_restore_facts(_side(parent_state, 1), _side(parent_state, 2), observed)
    with pytest.raises(AssertionError, match="observed 0 times"):
        probe._scaler_restore_facts(_side(parent_state, 1), _side(parent_state, 2), [])
    off = {"amp": {"enabled": False, "grad_scaler": False}, "progress": {}, "scaler": None}
    assert probe._scaler_restore_facts(off, off, [{"parent": None, "restored": None}])[
        "scaler_restore"].startswith("no gradient scaler")


def test_the_probe_observes_the_production_restore_and_catches_an_omitted_one(tmp_path):
    """End to end on the CPU: a real GradScaler is given to the Trainer (only
    in this test; on CPU mixed precision takes no effect), the parent pauses,
    and the probe's own `_train_run` continues it through `Trainer.continue_from`.
    The restore is verified; a restore that leaves the scaler fresh is caught;
    and the probe leaves `Trainer.continue_from` as it found it."""
    from unittest.mock import patch

    import torch

    from src.training.trainer import Trainer
    from tests.fixtures.trainable_workspace import write_trainable_workspace

    probe = _probe()
    settings = dict(hidden_dim=16, num_layers=1, num_heads=2, num_workers=0, use_amp=False,
                    batch_size=5, warmup_steps=1, seed=7)

    def cpu_scaler(self):
        self.use_amp, self.amp_dtype = False, torch.float32
        self.scaler = torch.amp.GradScaler("cpu", init_scale=8.0)

    original = Trainer.continue_from
    with patch.object(Trainer, "_setup_amp", cpu_scaler):
        ws = write_trainable_workspace(tmp_path / "ws")
        parent = probe._last(probe._train_run(ws, tmp_path / "parent", "cpu", 2,
                                              settings=settings, pause_after=1))
        observed = []
        child = probe._last(probe._train_run(ws, tmp_path / "child", "cpu", 2, settings=settings,
                                             resume_from=parent, observe=observed))
        assert Trainer.continue_from is original, "the probe left its observer installed"
        facts = probe._assert_resumed(parent, child, observed)
        assert facts["grad_scaler"] is True
        assert facts["scaler_restore"].startswith("verified at restore")

        def restore_then_forget(self, checkpoint, plan, best_weights=None):
            original(self, checkpoint, plan, best_weights)
            self.scaler = torch.amp.GradScaler("cpu", init_scale=8.0)

        with patch.object(Trainer, "continue_from", restore_then_forget):
            observed = []
            omitted = probe._last(probe._train_run(ws, tmp_path / "omitted", "cpu", 2,
                                                   settings=settings, resume_from=parent,
                                                   observe=observed))
        with pytest.raises(AssertionError, match="not restored from the parent's state"):
            probe._assert_resumed(parent, omitted, observed)


# ---------------------------------------------------------------------------
# "Only the remaining epoch" is counted by attempts (the GB10 acceptance of
# 4b1571e: a skipped AMP update in the parent's epoch, none in the child's)
# ---------------------------------------------------------------------------
def _record(completed, attempts, updates, steps, end, batches=8, accumulation=1):
    stage = {"kind": "original", "start_epoch": 0, "target_epochs": 2,
             "schedule": {"type": "cosine", "steps_per_epoch": batches,
                          "gradient_accumulation_steps": accumulation}}
    return {"progress": {"completed_epochs": completed, "optimizer_attempts": attempts,
                         "optimizer_updates": updates, "scheduler_steps": steps,
                         "stage": stage, "end": end},
            "loaders": {"train": {"num_batches": batches}}}


@pytest.mark.parametrize("parent,child", [
    ((1, 8, 7, 7), (2, 16, 15, 15)),     # the GB10 run: the parent's epoch skipped one
    ((1, 8, 8, 8), (2, 16, 15, 15)),     # the reverse: the child's epoch skipped one
    ((1, 8, 8, 8), (2, 16, 16, 16)),     # no skip
])
def test_a_skipped_amp_update_in_either_epoch_is_not_an_extra_epoch(parent, child):
    probe = _probe()
    facts = probe._assert_only_the_remaining_epochs(_record(*parent, "paused"),
                                                    _record(*child, "finished"))
    assert facts["child_run_attempts"] == 8 and facts["epochs_run"] == 1


@pytest.mark.parametrize("child,message", [
    ((2, 24, 23, 23, "finished"), "made 16 optimizer attempts"),   # the curve replayed
    ((2, 16, 15, 14, "finished"), "scheduler stepped 7 times"),
    ((2, 16, 7, 7, "finished"), "grew by 0"),
    ((1, 8, 7, 7, "paused"), "did not finish its stage"),
])
def test_more_work_than_the_remaining_epoch_or_inconsistent_counts_fail(child, message):
    probe = _probe()
    with pytest.raises(AssertionError, match=message):
        probe._assert_only_the_remaining_epochs(_record(1, 8, 7, 7, "paused"), _record(*child))


def test_the_remaining_epoch_is_counted_with_accumulation():
    """Seven batches with an accumulation of 3 take three attempts an epoch,
    the tail included."""
    probe = _probe()
    facts = probe._assert_only_the_remaining_epochs(
        _record(1, 3, 3, 3, "paused", batches=7, accumulation=3),
        _record(2, 6, 5, 5, "finished", batches=7, accumulation=3))
    assert facts["attempts_per_epoch"] == 3


@pytest.mark.parametrize("skip_in", ["parent", "child"])
def test_a_real_skipped_update_through_the_trainer_passes_and_a_replay_fails(tmp_path, skip_in):
    """End to end on the CPU with a real CPU GradScaler: a non-finite gradient
    at one optimizer attempt makes the scaler skip that update, in the parent's
    epoch or in the child's. The probe's check passes either way; a restore that
    replays the original curve from epoch 0 is caught."""
    from unittest.mock import patch

    import torch

    from src.training.trainer import Trainer
    from tests.fixtures.trainable_workspace import write_trainable_workspace

    probe = _probe()
    settings = dict(hidden_dim=16, num_layers=1, num_heads=2, num_workers=0, use_amp=False,
                    batch_size=5, warmup_steps=1, seed=7)

    def cpu_scaler(self):
        self.use_amp, self.amp_dtype = False, torch.float32
        self.scaler = torch.amp.GradScaler("cpu", init_scale=8.0)

    original_update = Trainer._optimizer_update
    skip_attempt = {"parent": 1, "child": 4}[skip_in]    # 3 attempts an epoch here

    def update_with_one_skip(self):
        if self.state.global_step == skip_attempt:
            parameter = next(p for p in self.model.parameters() if p.grad is not None)
            parameter.grad = torch.full_like(parameter, float("inf"))
        return original_update(self)

    with patch.object(Trainer, "_setup_amp", cpu_scaler), \
            patch.object(Trainer, "_optimizer_update", update_with_one_skip):
        ws = write_trainable_workspace(tmp_path / "ws")
        parent = probe._last(probe._train_run(ws, tmp_path / "parent", "cpu", 2,
                                              settings=settings, pause_after=1))
        child = probe._last(probe._train_run(ws, tmp_path / "child", "cpu", 2, settings=settings,
                                             resume_from=parent, observe=[]))
        before, after = probe._read_run_checkpoint(parent), probe._read_run_checkpoint(child)
        skipped = (before if skip_in == "parent" else after)["progress"]
        assert skipped["optimizer_attempts"] - skipped["optimizer_updates"] == 1, (
            "the injected non-finite gradient did not make the scaler skip an update")
        facts = probe._assert_only_the_remaining_epochs(before, after)
        assert facts["child_run_skipped_updates"] == (1 if skip_in == "child" else 0)

        original_continue = Trainer.continue_from

        def replay_from_the_start(self, checkpoint, plan, best_weights=None):
            original_continue(self, checkpoint, plan, best_weights)
            self.start_epoch = 0

        with patch.object(Trainer, "continue_from", replay_from_the_start):
            replayed = probe._last(probe._train_run(ws, tmp_path / "replayed", "cpu", 2,
                                                    settings=settings, resume_from=parent,
                                                    observe=[]))
        with pytest.raises(AssertionError, match="optimizer attempts"):
            probe._assert_only_the_remaining_epochs(before, probe._read_run_checkpoint(replayed))
