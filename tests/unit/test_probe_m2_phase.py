"""The deployment probe's M2 acceptance (phases M and R), on CPU.

On the GB10 these run with `--m2-acceptance --device cuda`; here the phase
functions run on CPU over the demo graph, so the production entry points they
drive -- the trainer on the model-bundle path (bases and disease children,
paused, resumed and extended), the measurement CLI by task, the disease-task
serving build and the inventory -- are exercised end to end, and every probe
in each phase must pass. Also: the acceptance is complete only when every
required probe of both parts ran and passed, the CLI cannot turn the
acceptance into a CPU run, and a snapshot that includes directories sees an
empty one left behind. Each continued run is judged against the precision it
asked for (`_scaler_restore_facts`), and the runs pass the settings they state
to the training entry, so a setting stated at its default value is recorded
as stated (M2's `use_amp`, R3's `batch_size`).

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
running the real CLIs on CPU, the GPU lock. W7-W11's test runs measure W1's
listed copy of M2's resumed disease child; RW2's a disease child finetuned
from R4's resumed base into a run directory of its own. The phase tests
require every one to pass and check its evidence against the disk; the
helpers' own checks are shown to fire in tests/unit/test_probe_m5_acceptance.py.

Module: tests/unit/test_probe_m2_phase.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
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
    it, naming the path, the digest and the task as the Load sends them. The
    digest expected is served (the positive control); another is refused by the
    reload itself, before anything is published -- the pipeline served before
    stays the same object -- never published and then caught by the status's
    digest afterwards."""
    import hashlib

    from tests.fixtures.registered_cohort import write_demo_workspace
    from tests.fixtures.trained_children import train_children

    probe = _probe()
    for name in ("SHEPHERD_KG_PATH", "SHEPHERD_DATA_DIR", "SHEPHERD_CHECKPOINT_PATH"):
        monkeypatch.delenv(name, raising=False)
    trained = train_children(write_demo_workspace(tmp_path / "ws"), ("disease",))
    data_dir, checkpoint = trained["data_dir"], trained["disease"]
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
    # Mode D's probe (T7) went with Mode D (K7): required, it could never pass.
    assert "T7" not in probe.M2_DEMO_REQUIRED
    assert set(M4_REAL_PROBES) <= set(probe.M2_REAL_REQUIRED)
    # K8-1 (Q9 (3)): a parent without a model spec is refused (M5), and the
    # real part's disease child is finetuned from R4's resumed base, each
    # required under its own ID, no ID renamed.
    assert "M5.no_model_spec" in probe.M2_DEMO_REQUIRED
    assert "R4.finetune" in probe.M2_REAL_REQUIRED
    assert (len(probe.M2_DEMO_REQUIRED), len(probe.M2_REAL_REQUIRED)) == (52, 15)
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
    models = probe.phase_training(report, tmp_path, sound, device="cpu", epochs=1)
    assert models is not None
    checkpoint = models.child

    probe.phase_m2_demo(report, tmp_path, sound, checkpoint, models.base, device="cpu")

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
    # M2 states mixed precision float16 (Q13 (b)); on the CPU it takes no
    # effect, and the report says why for each continued run, judged against
    # that request; on CUDA the scaler's restore is checked (scaler_restore).
    for probe_id in ("M2", "M9"):
        assert facts[probe_id]["scaler_restore"] == "no gradient scaler (CPU)"
        assert (facts[probe_id]["amp_requested"], facts[probe_id]["amp_enabled"]) == (True, False)
    # **The stated setting reaches the record** (the reviewer's P2-1 (a)): the
    # paused run, M2's parent, records `use_amp` True as the operator's, never
    # filled from the finetune recipe's fp32; the resumed child inherits it.
    from src.utils.checkpoint_io import read_checkpoint

    def recorded(run):
        config = read_checkpoint(tmp_path / "m2" / run / "checkpoints" / "last.pt",
                                 map_location="cpu", weights_only=True).checkpoint["config"]
        return config["requested"], config["effective"]["loaders"]["train"]["recipe"]

    requested, recipe = recorded("paused")
    assert requested["use_amp"] is True and requested["amp_dtype"] == "float16"
    assert "use_amp" in recipe["operator_set"] and "use_amp" not in recipe["reference_filled"]
    requested, recipe = recorded("resume_ok")
    assert requested["use_amp"] is True
    assert "use_amp" in recipe["inherited_from_parent"]
    # M2 finetuned D1's base; its child names it, by the digest of its bytes.
    assert facts["D2"]["parent_sha256"] == hashlib.sha256(models.base.read_bytes()).hexdigest()
    # Contract M3c: M4's diagnosis names the resumed child by the digest M2
    # recorded, the disease task with no shortest-path term, the synthetic
    # record, the device asked for -- and its evidence holds no path. These are
    # the probe's own facts, so they hold with its checks removed; that the
    # checks in `_serve_and_diagnose` (M4, R6) fire is shown in
    # test_probe_serving_phase.py.
    identity = facts["M4"]["served_identity"]
    assert identity["checkpoint"] == {"state": "verified", "sha256": child}
    assert identity["scoring_mode"] == "disease_task"
    assert identity["shortest_paths"]["state"] == "absent"
    assert facts["M4"]["result_kind"] == "disease_task/v1"
    # M3 measured the resumed child by its task.
    assert measured["manifest"]["mode"] == facts["M3"]["mode"] == "disease"
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
    registry = tmp_path / "m4" / "registry"
    version = registry / "versions" / probe.M4_DEMO_LABEL
    assert facts["T1"]["counts"] == probe.M4_DEMO_COUNTS
    assert facts["T1"]["manifest_sha256"] == hashlib.sha256(
        (version / "cohort.json").read_bytes()).hexdigest()
    # T2: the one disease report kept for D1's child is what the record names,
    # beside the copy of D1's child (the bytes D1 trained), and R10's evidence
    # was found. (W8's run keeps another, for M2's child listed by W1; a run
    # directory is named for the checkpoint it measured.)
    d1_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    d1_reports = {json.loads(path.read_text())["manifest"]["mode"]: path
                  for path in (version / "reports").glob("*/report.json")
                  if f"-{d1_sha[:12]}-" in path.parent.name}
    assert sorted(d1_reports) == ["disease"]
    kept = d1_reports["disease"]
    checkpoints = tmp_path / "m4" / "checkpoints"
    (record,) = _ledger_records(checkpoints)
    assert record["source_artifact_digest"] == facts["T2"]["report_sha256"] == hashlib.sha256(
        kept.read_bytes()).hexdigest()
    assert record["checkpoint_digest"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert (checkpoints / "measured.pt").read_bytes() == checkpoint.read_bytes()
    for scope in ("training", "validation"):
        assert facts["T2"]["r10"][scope]["evidence"] == "found"
        assert facts["T2"]["r10"][scope]["disease_overlap"]["k"] is not None
    # T3 and T4 wrote nothing: T2's record only, and no report for the version
    # mapped against M1's graph (its import itself is on disk).
    assert facts["T3"]["same_report_again"] == "already_registered"
    assert len(_ledger_records(checkpoints)) == 1
    assert (registry / "versions" / probe.M4_OTHER_GRAPH_LABEL / "cohort.json").is_file()
    assert not (registry / "versions" / probe.M4_OTHER_GRAPH_LABEL / "reports").exists()
    assert "kg" in facts["T4"]["differing_roles"]
    # T5: the record for the measured child, uncontested in the probe's
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
    # resumed disease child, filed under its encoder kind and named by the
    # digest of its bytes.
    workspace = tmp_path / "m5" / "workspaces" / probe.M5_WORKSPACE
    listed = (workspace / "runs" / "gatv2" / probe.M5_LISTED_RUN / "checkpoints"
              / probe.M5_CHECKPOINT)
    assert hashlib.sha256(listed.read_bytes()).hexdigest() == child
    assert facts["W1"]["checkpoints"][0] == {"run": probe.M5_LISTED_RUN, "filename": "last.pt",
                                             "sha256": child, "listed_with_a_problem": False}
    assert facts["W1"]["workspaces"][0]["dataset_fingerprint"] == hashlib.sha256(
        (workspace / "split_manifest.json").read_bytes()).hexdigest()
    # W2 and W5: every request shape was decided by the condition named for it,
    # on every gated API route (D8's eight) and the three gated dashboard events.
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
    # W8 and W9: a test run measures a finetuned model by its task (K6), so
    # W7-W11 name W1's listed copy of M2's disease child (NEW-probe-1): no
    # model is trained for them, and runs/gatv2 holds the listed run and the
    # link out only. W8's report is kept in T1's version, measured as the
    # disease task, and the one record beside the listed child's checkpoint,
    # in its run, is built from those bytes and scored as that task; the usage
    # history holds the run and its registration. W10 left no third report
    # (T2's and W8's are the two).
    task_runs = workspace / "runs" / "gatv2"
    assert sorted(path.name for path in task_runs.iterdir()) == sorted(
        [probe.M5_LISTED_RUN, probe.M5_LINKED_RUN])
    task_child = listed
    task_sha = hashlib.sha256(task_child.read_bytes()).hexdigest()
    assert task_sha == child
    assert facts["W8"]["measured"]["sha256"] == task_sha
    assert facts["W8"]["measured"]["task"] == "disease"
    assert facts["W8"]["measured"]["run"] == probe.M5_LISTED_RUN
    reports = sorted((version / "reports").glob("*/report.json"))
    assert len(reports) == 2
    (w8_report,) = [path for path in reports if f"-{task_sha[:12]}-" in path.parent.name]
    assert hashlib.sha256(w8_report.read_bytes()).hexdigest() == facts["W8"]["report_sha256"]
    assert json.loads(w8_report.read_text())["manifest"]["mode"] == "disease"
    (w9_record,) = _ledger_records(task_child.parent)
    assert w9_record["source_artifact_digest"] == facts["W8"]["report_sha256"]
    assert w9_record["checkpoint_digest"] == task_sha
    assert w9_record["scoring"]["mode"] == facts["W9"]["scoring_mode"] == "disease"
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


def _history_file(path):
    """A history file: a checkpoint without a model spec or a training record,
    written as plain tensors and data, without any model class, as the
    deployment's older models are to this build (the probe reads nothing from
    it to configure a run; R2 and R7 refuse it, R8 checks it unchanged)."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": 2, "state_dict": {"encoder.weight": torch.zeros(4, 4)},
                "config": {"model_config": {"conv_type": "hgt"}}}, path)
    return path


@pytest.mark.parametrize("shell, history", [("fresh", False), ("operator", True)])
def test_the_real_phase_passes_and_leaves_the_workspace_as_it_was(
        tmp_path, served_state, monkeypatch, tmp_path_factory, shell, history):
    """Phase R end to end, from a fresh shell (no registry named) and without a
    history file, and from the operator's shell (the deployment's registry,
    root and lock named, BACKLOG 37 and the guide's §4.11) with one. Either way
    M4's flow uses the probe's registry as the deployment's, so RT3 sees its
    record uncontested, and the deployment's directories are not touched
    (root-cause round, INT-1). R1 needs no shortest-path table: the fresh
    run's workspace holds none, and the operator's holds a leftover pair, of
    another graph and with a tensor no reader could parse: nothing in the
    phase loads the table, R7's inventory still reads the sidecar and reports
    it refused until K8-5, and R8 covers the pair unchanged."""
    import json

    if shell == "operator":
        operator = tmp_path_factory.mktemp("operator")
        _operator_shell(monkeypatch, operator)
    probe = _probe()
    from scripts.setup_demo import demo_task_training
    from src.kg.workspace import write_workspace
    from src.utils.checkpoint_io import TRAINING_RULES_VERSION, read_checkpoint

    root = tmp_path / "workspaces"
    real = root / "ws"
    # A real workspace's record is from files, every input identified (R1,
    # contract M3b); the stand-in names four files, as a real build would.
    write_workspace(probe._demo_kg(), real, feature_dim=probe.FEATURE_DIM,
                    samples=probe._budget(), sources=_four_sources(tmp_path / "inputs"))
    assert not (real / "shortest_paths.pt").exists()
    reference = None
    if history:
        reference = _history_file(real / "checkpoints" / "hgt" / "model-ref.pt")
        (real / "shortest_paths.pt").write_bytes(b"a leftover table: never read")
        (real / "shortest_paths.meta.json").write_text(json.dumps(
            {"schema_version": 1, "build_id": "c" * 32, "kg_digest": "d" * 64,
             "max_hops": 5}))

    report = probe.Report()
    work = tmp_path / "work"
    # The models at the demo's sizes over the reference recipe: the
    # deployment's run trains them by the reference recipe, whose checkpoints
    # are hundreds of megabytes.
    sizes = demo_task_training()
    # Every directory the phase reads by content (`snapshot` digests each file
    # under it) is under the work directory: R2 checks the history file by
    # size and modification time, never by digesting the directory it is in,
    # which can be the real workspace or the root (before the review fix it
    # digested workspaces/ws/checkpoints/hgt here, twice).
    digested = []
    snapshot = probe.snapshot
    monkeypatch.setattr(probe, "snapshot", lambda directory, **kwargs: (
        digested.append(Path(directory)), snapshot(directory, **kwargs))[1])
    probe.phase_m2_real(report, work, real, reference, root, device="cpu", task_sizes=sizes)

    phase = _statuses(report, "R")
    assert set(phase) == set(probe.M2_REAL_REQUIRED)
    assert {k: v[0] for k, v in phase.items()} == dict.fromkeys(phase, "passed"), (
        {k: v[1] for k, v in phase.items() if v[0] != "passed"}
    )
    facts = {p.probe_id: p.facts for p in report.probes}
    order = [p.probe_id for p in report.probes]
    # R2 strips R4.finetune's child, so it runs after it.
    assert order.index("R2") == order.index("R4.finetune") + 1
    # R3: a base by the reference recipe, the build's training rules (the live
    # constant, never a literal), its graph roles and no sample role.
    assert facts["R3"]["learning_stage"] == "base"
    assert facts["R3"]["training_rules"] == TRAINING_RULES_VERSION
    assert "train_samples" not in facts["R3"]["recorded_roles"]
    # **The regression for the explicit-set inference** (the reviewer's P2-1;
    # independent of Q13): R3 states batch_size 32, which equals TrainConfig's
    # default. Passed as the explicit set it is recorded as the operator's;
    # inferred from the defaults it would read as unset and the base recipe
    # would train at its 512.
    assert sizes["base"]["batch_size"] == 32
    paused = read_checkpoint(work / "real" / "initial" / "checkpoints" / "last.pt",
                             map_location="cpu", weights_only=True).checkpoint["config"]
    recipe = paused["effective"]["loaders"]["train"]["recipe"]
    assert paused["requested"]["batch_size"] == 32
    assert paused["effective"]["loaders"]["train"]["batch_size"] == 32
    assert "batch_size" in recipe["operator_set"]
    assert "batch_size" not in recipe["reference_filled"]
    # R4: judged against R3's request, the base recipe's fp32.
    assert facts["R4"]["scaler_restore"] == "no gradient scaler (fp32 recipe)"
    assert facts["R4"]["child_optimizer_step"] > facts["R4"]["parent_optimizer_step"]
    # R4.finetune's child names R4's actual resumed file as its parent.
    resumed_base = work / "real" / "resumed" / "checkpoints" / "last.pt"
    child = work / "real" / "finetune" / "checkpoints" / "last.pt"
    child_sha = hashlib.sha256(child.read_bytes()).hexdigest()
    assert facts["R4.finetune"]["parent_sha256"] == facts["R4"]["child_sha256"] == (
        hashlib.sha256(resumed_base.read_bytes()).hexdigest())
    assert (facts["R4.finetune"]["sha256"], facts["R4.finetune"]["task"]) == (child_sha,
                                                                             "disease")
    # R2: a file without a model spec refused by a finetune and a resume: a
    # stripped copy of R4.finetune's child always, the history file when given.
    refused = {"finetune": "refused before anything was written",
               "resume": "refused before anything was written"}
    subjects = facts["R2"]["subjects"]
    assert subjects["stripped copy of R4.finetune's child"] == refused
    if history:
        assert subjects["history file given"] == refused
    else:
        assert subjects["history file"] == "not given"
    assert not (work / "real" / "history" / "refused").exists()
    assert digested and all(path.is_relative_to(work) for path in digested), (
        f"the phase read by content outside the work directory: {digested}")
    # R5 and R6: the child measured by its task and served for it.
    assert facts["R5"]["mode"] == "disease" and facts["R5"]["checkpoint_sha256"] == child_sha
    # Contract M3c: R6's diagnosis names the child R4.finetune trained and the
    # record from files R1 verified.
    identity = facts["R6"]["served_identity"]
    assert identity["checkpoint"]["sha256"] == child_sha
    assert identity["scoring_mode"] == "disease_task"
    assert identity["graph"]["kg_sha256"] == facts["R1"]["kg_digest"]
    assert identity["dataset_record"] == {
        "state": "verified", "origin": "files",
        "sha256": facts["R1"]["dataset_record"]["sha256"]}
    assert facts["R7"]["reference_refusal"] == ("no record" if history else "not given")
    assert facts["R8"]["history_file"] == ("unchanged" if history else "not given")

    # ---- Contract M4's flow on the (stand-in) real workspace: RT1-RT3 run
    # after R7 and before R8, whose comparison therefore covers them.
    assert order[order.index("R7") + 1:] == [*M4_REAL_PROBES, *probe.M5_REAL_PROBES, "R8"]
    assert str(tmp_path) not in json.dumps({k: facts[k] for k in M4_REAL_PROBES})
    n = facts["RT1"]["val_cases_written"]
    assert n == min(probe.M4_REAL_CASES, probe.NUM_VAL)        # the demo names every node
    assert facts["RT1"]["counts"]["n_excluded"] == 0
    r10 = facts["RT2"]["r10"]
    assert r10["training"]["disease_overlap"]["state"] == "none"
    assert r10["validation"]["phenotype_set_overlap"] == {"state": "overlap", "k": n, "n": n}
    # The ledger is beside R4.finetune's child, under the work directory; the
    # registry and its report too. Nothing of M4's is in the workspace.
    (record,) = _ledger_records(child.parent)
    assert record["checkpoint_digest"] == child_sha
    assert record["source_artifact_digest"] == facts["RT2"]["report_sha256"]
    assert facts["RT3"]["state"] == "records"
    assert facts["RT3"]["checkpoint_sha256"] == child_sha
    assert facts["RT3"]["contest_state"] == "uncontested"
    assert not list(real.rglob("evaluations.json")) and not (real / "registry").exists()

    # ---- M5's acceptance on the (stand-in) real workspace, before R8: listed
    # by its manifest's digest; RW2 measured a disease child of its own,
    # finetuned from R4's resumed base into its own run directory (never
    # R4.finetune's child, beside which RT2 registered), its record beside the
    # child's checkpoint and scored as the disease task, none beside R3's; RW3
    # stopped. The lock is the probe's own, under a workspaces root of its own
    # in the work directory, never the one under the deployment's root.
    assert str(tmp_path) not in json.dumps({k: facts[k] for k in probe.M5_REAL_PROBES})
    assert facts["RW1"]["dataset_fingerprint"] == hashlib.sha256(
        (real / "split_manifest.json").read_bytes()).hexdigest()
    assert facts["RW1"]["dataset_record"] == {"binding": "verified", "origin": "files"}
    disease_child = work / "real" / "m5" / "disease-child" / "checkpoints" / "last.pt"
    assert not (work / "real" / "m5" / "disease-base").exists()
    rw2_sha = hashlib.sha256(disease_child.read_bytes()).hexdigest()
    assert rw2_sha != child_sha
    rw2_parent = read_checkpoint(disease_child, map_location="cpu",
                                 weights_only=True).checkpoint["parent"]
    assert rw2_parent["sha256"] == facts["R4"]["child_sha256"]
    assert (facts["RW2"]["measured"]["sha256"], facts["RW2"]["measured"]["task"]) == (
        rw2_sha, "disease")
    (rw2_record,) = _ledger_records(disease_child.parent)
    assert rw2_record["source_artifact_digest"] == facts["RW2"]["report_sha256"]
    assert rw2_record["checkpoint_digest"] == rw2_sha
    assert rw2_record["scoring"]["mode"] == "disease"
    assert not (work / "real" / "initial" / "checkpoints" / "evaluations.json").exists()
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
    acceptance = [*work, "--m2-acceptance", "--device", "cuda"]
    real_args = ["--real-workspace", str(real), "--real-reference-checkpoint", str(reference)]

    for argv in (
        [*work, "--m2-acceptance"],                                    # device auto
        [*work, "--m2-acceptance", "--device", "cpu"],
        [*work, "--device", "cuda", *real_args],                       # not an acceptance
        ["--work-dir", str(real / "w"), "--m2-acceptance", "--device", "cuda", *real_args],
        [*acceptance, *real_args,
         "--workspaces-root", str(tmp_path / "workspaces" / "ws" / "checkpoints" / "x")],
        # A history file is a file inside --workspaces-root, where the
        # inventory lists it.
        [*acceptance, "--real-workspace", str(real),
         "--real-reference-checkpoint", str(real / "checkpoints")],
        [*acceptance, "--real-workspace", str(real), "--real-reference-checkpoint",
         str(config)],
        [*work, "--real-reference-checkpoint", str(reference)],        # without --real-workspace
        # Q9 (1) A: the settings file is retired with its reader; argparse
        # refuses the flag, with no alias.
        [*acceptance, *real_args, "--real-config", str(config)],
    ):
        with pytest.raises(SystemExit):
            probe.parse_args(argv)

    args = probe.parse_args([*acceptance, *real_args])
    assert args.workspaces_root == real.parent
    assert args.real_reference_checkpoint == reference
    assert not hasattr(args, "real_config")
    # Q9 (2): the history file is optional; the real workspace alone runs.
    alone = probe.parse_args([*acceptance, "--real-workspace", str(real)])
    assert (alone.real_reference_checkpoint, alone.workspaces_root) == (None, real.parent)


def _real_inputs(base):
    """A real workspace and a history file in it under `base`, laid out as the
    probe's usage text spells them from the repository root."""
    real = base / "data" / "workspaces" / "ws"
    (real / "checkpoints").mkdir(parents=True)
    (real / "checkpoints" / "model.pt").write_bytes(b"reference")
    return real


RELATIVE_ACCEPTANCE = [
    "--work-dir", "work", "--report", "work/m2.json", "--m2-acceptance", "--device", "cuda",
    "--real-workspace", "data/workspaces/ws",
    "--real-reference-checkpoint", "data/workspaces/ws/checkpoints/model.pt",
]


def test_relative_paths_are_made_absolute_once_from_the_starting_directory(
        tmp_path, monkeypatch):
    """BACKLOG 49: the usage text's relative `--real-workspace` (and the default
    root, its parent) reached RW1 as given, and the server's directory check
    refused a relative root only after the real training. Every path argument
    is now absolute from parse_args on, the same as its absolute spelling, and
    RW1's check takes the root as parsed."""
    from src.api.services import server_lists

    probe = _probe()
    base = tmp_path / "start"
    real = _real_inputs(base)
    monkeypatch.chdir(base)

    relative = probe.parse_args(RELATIVE_ACCEPTANCE)
    absolute = probe.parse_args([
        "--work-dir", str(base / "work"), "--report", str(base / "work" / "m2.json"),
        "--m2-acceptance", "--device", "cuda", "--real-workspace", str(real),
        "--real-reference-checkpoint", str(real / "checkpoints" / "model.pt")])

    listed = server_lists.resolve_workspaces(          # refused "not an absolute path"
        environ={"SHEPHERD_WORKSPACES_ROOT": str(relative.workspaces_root)})
    assert listed == {"ws": real.resolve()}
    assert relative.workspaces_root == real.parent
    for name in probe.PATH_ARGUMENTS:
        value = getattr(relative, name)
        assert value == getattr(absolute, name), name
        assert value is None or value.is_absolute(), name


def test_a_link_followed_by_dotdot_keeps_the_file_the_file_system_opens(
        tmp_path, monkeypatch):
    """The review of b14a6e9: `link/../source.obo` is the file beside the
    link's target, which the file system opens; collapsing `..` by its spelling
    named `source.obo` in the starting directory instead, another file, with no
    error. The starting directory is put in front and nothing else changes."""
    probe = _probe()
    (tmp_path / "real" / "sub").mkdir(parents=True)
    (tmp_path / "real" / "source.obo").write_text("A, the one meant")
    (tmp_path / "source.obo").write_text("B, a different file")
    (tmp_path / "link").symlink_to(tmp_path / "real" / "sub", target_is_directory=True)
    monkeypatch.chdir(tmp_path)

    args = probe.parse_args(["--work-dir", "w", "--mondo-path", "link/../source.obo"])

    assert args.mondo_path.is_absolute()
    assert args.mondo_path.read_text() == "A, the one meant"
    assert args.mondo_path.resolve() == (tmp_path / "real" / "source.obo").resolve()


def test_a_relative_path_from_another_directory_is_refused_before_any_work(
        tmp_path, monkeypatch, capsys):
    """The base is the directory the probe is started in, not the repository:
    the same relative arguments from elsewhere name paths that do not exist
    there, and the run stops in parse_args, naming the absolute path it looked
    for, before the work directory exists or any phase runs."""
    probe = _probe()
    _real_inputs(tmp_path / "start")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    def never(*args, **kwargs):
        raise AssertionError("a phase ran before the arguments were checked")

    monkeypatch.setattr(probe, "phase_environment", never)
    with pytest.raises(SystemExit) as stopped:
        probe.main(RELATIVE_ACCEPTANCE)
    assert stopped.value.code == 2
    assert str(elsewhere / "data" / "workspaces" / "ws") in capsys.readouterr().err
    assert not (elsewhere / "work").exists()

    for argv in (["--external-dir", "missing"], ["--mondo-path", "missing.obo"],
                 ["--hpo-path", "missing.obo"]):
        with pytest.raises(SystemExit):
            probe.main(["--work-dir", "work2", *argv])
        assert str(elsewhere / argv[1]) in capsys.readouterr().err
    assert not (elsewhere / "work2").exists()


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

    def t1_import(report, work, sound, checkpoint, base, device):
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
    monkeypatch.setattr(probe, "phase_training", lambda *args: probe.DemoModels(
        child=Path("not-trained.pt"), base=Path("not-trained-base.pt")))
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


def test_t2_and_rt2_rank_the_cases_the_probe_binds_itself():
    """T2's and RT2's coverage is checked against a binding the probe computes
    itself (`_own_binding`, the child's graph view and the version's cases),
    not only against the report's own numbers: a report whose binding and
    ranked count shrank together (a view or binding regression that ranks 1 of
    10 cases) agrees with itself, and before the review fix it passed; now it
    fails, as does a report that excludes for another reason."""
    probe = _probe()
    own = {"cases": 10, "bound": 8, "excluded": 2, "phenotypes_outside_view": 5,
           "excluded_no_phenotype_in_view": 2}

    def kept(binding):
        return {"n_ranked": binding["bound"], "sampler_evidence": {"binding": binding}}

    assert probe._ranked_cases(kept(dict(own)), 10, 10, own) == own
    altered = dict(own, bound=1, excluded=9, excluded_no_phenotype_in_view=9)
    with pytest.raises(AssertionError, match="the child's graph view binds 8 and excludes 2"):
        probe._ranked_cases(kept(altered), 10, 10, own)
    other_reason = {**{k: v for k, v in own.items() if k != "excluded_no_phenotype_in_view"},
                    "excluded_truth_outside_view": 2}
    with pytest.raises(AssertionError, match="the child's graph view binds"):
        probe._ranked_cases(kept(other_reason), 10, 10, own)
    # The checks on the report alone stay: one row per case, and the ranked
    # count is the report's bound count.
    with pytest.raises(AssertionError, match="not one per case"):
        probe._ranked_cases(kept(dict(own)), 9, 10, own)
    with pytest.raises(AssertionError, match="the view binds 8"):
        probe._ranked_cases(dict(kept(dict(own)), n_ranked=7), 10, 10, own)


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
    work = tmp_path / "work"
    earlier = tmp_path / "earlier"
    earlier.mkdir()
    (earlier / "report.json").write_text("earlier")
    acceptance = ["--m2-acceptance", "--device", "cuda", "--real-workspace", str(real),
                  "--real-reference-checkpoint", str(reference)]

    for report, work_dir in (
        (reference, work), (real / "report.json", work),
        (tmp_path / "beside.json", work),                      # outside the work dir
        (earlier / "report.json", earlier),                    # an earlier result
    ):
        with pytest.raises(SystemExit):
            probe.parse_args(["--work-dir", str(work_dir), "--report", str(report),
                              *acceptance])
    assert reference.read_bytes() == b"reference"
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


#: A CUDA float16 request, as M2 states it on the GB10 (Q13 (b)).
CUDA_FP16 = {"use_amp": True, "amp_dtype": "float16", "device": "cuda",
             "source": "stated setting"}
#: An fp32 request, as R3 asks for it by the base recipe (R4's shape).
FP32_RECIPE = {"use_amp": False, "amp_dtype": None, "device": "cuda", "source": "base recipe"}


def _amp(requested=True, enabled=True, dtype="float16", grad_scaler=True,
         dtype_requested="float16"):
    """A mixed precision record with all five keys `Trainer.amp_record` writes."""
    return {"requested": requested, "dtype_requested": dtype_requested, "enabled": enabled,
            "dtype": dtype, "grad_scaler": grad_scaler}


def _side(scaler_state, updates, amp=None):
    return {"amp": amp or _amp(), "progress": {"optimizer_updates": updates},
            "scaler": ({k: v for k, v in scaler_state.items() if isinstance(v, (int, float))}
                       if scaler_state else None)}


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

    facts = probe._scaler_restore_facts(_side(parent_state, 1), _side(end, 6), observed,
                                        requested=CUDA_FP16)
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
        probe._scaler_restore_facts(_side(parent_state, 1), _side(parent_state, 2), observed,
                                    requested=CUDA_FP16)
    with pytest.raises(AssertionError, match="observed 0 times"):
        probe._scaler_restore_facts(_side(parent_state, 1), _side(parent_state, 2), [],
                                    requested=CUDA_FP16)
    off = _side(None, 0, _amp(requested=False, enabled=False, dtype="float32",
                              grad_scaler=False))
    facts = probe._scaler_restore_facts(off, off, [{"parent": None, "restored": None}],
                                        requested=FP32_RECIPE)
    assert facts["scaler_restore"] == "no gradient scaler (fp32 recipe)"


# ---------------------------------------------------------------------------
# The scaler check judges a run against what it asked for (the reviewer's
# review of K8 revision 1, P2-1 (b)); six negative cases, on the CPU with
# synthetic records
# ---------------------------------------------------------------------------
def _restored(parent_state):
    numbers = {k: v for k, v in parent_state.items() if isinstance(v, (int, float))}
    return [{"parent": numbers, "restored": dict(numbers)}]


SCALER = {"scale": 8.0, "growth_factor": 2.0, "backoff_factor": 0.5, "growth_interval": 2000,
          "_growth_tracker": 0}


def test_a_cuda_float16_request_without_a_scaler_fails_naming_it():
    """(1) A CUDA float16 request whose records show float16 in effect and no
    gradient scaler: at HEAD this passed as "no gradient scaler"."""
    probe = _probe()
    no_scaler = _side(None, 1, _amp(grad_scaler=False))
    with pytest.raises(AssertionError, match="ran float16 without a gradient scaler"):
        probe._scaler_restore_facts(no_scaler, no_scaler, [{"parent": None, "restored": None}],
                                    requested=CUDA_FP16)


def test_a_cuda_float16_request_with_amp_recorded_off_fails():
    """(2) The same request with mixed precision recorded off: a run that
    silently fell back to fp32 is not the run asked for."""
    probe = _probe()
    off = _side(None, 1, _amp(enabled=False, dtype="float32", grad_scaler=False))
    with pytest.raises(AssertionError, match="which the run did not ask for: on in float16"):
        probe._scaler_restore_facts(off, off, [{"parent": None, "restored": None}],
                                    requested=CUDA_FP16)


def test_a_request_recorded_as_not_requested_fails():
    """(3) A record whose `requested` is False against a request of True: a
    stated setting lost on the way in (the P2-1 (a) shape), on any device."""
    probe = _probe()
    lost = _side(None, 1, _amp(requested=False, enabled=False, dtype="float32",
                               grad_scaler=False))
    for device in ("cuda", "cpu"):
        with pytest.raises(AssertionError, match="records mixed precision requested False"):
            probe._scaler_restore_facts(lost, lost, [{"parent": None, "restored": None}],
                                        requested={**CUDA_FP16, "device": device})


def test_the_reviewer_s_counterexample_leaves_the_demo_part_incomplete():
    """(4) At acceptance level: case (2)'s records fed through M2's check in a
    report where every other required ID passed. M2 fails, so `m2_acceptance`
    reports the demo part incomplete (it counts passed IDs)."""
    probe = _probe()
    off = _side(None, 1, _amp(enabled=False, dtype="float32", grad_scaler=False))
    report = probe.Report()
    for probe_id in probe.M2_DEMO_REQUIRED:
        if probe_id == "M2":
            report.run("M2", "m2", "x", lambda: ("", probe._scaler_restore_facts(
                off, off, [{"parent": None, "restored": None}], requested=CUDA_FP16)))
        else:
            report.probes.append(probe.Probe(probe_id, "m2", "x", status="passed"))
    assert probe.m2_acceptance(report, False)["demo"] == {
        "requested": True, "required": list(probe.M2_DEMO_REQUIRED),
        "not_passed": {"M2": "failed"}, "complete": False}


def test_an_fp32_request_with_float16_in_effect_fails_naming_it():
    """(5) An fp32 request (R4's: the base recipe's) against records showing
    float16 in effect with a scaler and a verified restore: a precision the run
    did not ask for, refused by name."""
    probe = _probe()
    side = _side(SCALER, 1, _amp(requested=False))
    with pytest.raises(AssertionError, match="which the run did not ask for: off in float32"):
        probe._scaler_restore_facts(side, side, _restored(SCALER), requested=FP32_RECIPE)


@pytest.mark.parametrize("amp", [{}, {k: v for k, v in _amp(requested=False, enabled=False,
                                                            dtype="float32",
                                                            grad_scaler=False).items()
                                     if k != "requested"}])
def test_an_fp32_request_against_an_incomplete_record_fails(amp):
    """(6) An fp32 request against an empty amp block (what `_read_run_
    checkpoint` gives for a record without one) or one without `requested`:
    every key is read, none defaulted."""
    probe = _probe()
    side = _side(None, 1, amp or {})
    side["amp"] = amp
    with pytest.raises(AssertionError, match="mixed precision record has no"):
        probe._scaler_restore_facts(side, side, [{"parent": None, "restored": None}],
                                    requested=FP32_RECIPE)


def test_a_stated_precision_is_the_request_and_an_unstated_one_the_recipe_s():
    """`_precision_request` reads the request from what a run states, else from
    the stage recipe (`src.config.recipe`), never a copied value; a run asking
    for mixed precision names its dtype."""
    from src.config.recipe import reference_settings

    probe = _probe()
    assert probe._precision_request(probe.M2_PRECISION, "cuda", "finetune", "disease") == {
        "use_amp": True, "amp_dtype": "float16", "device": "cuda", "source": "stated setting"}
    recipe = probe._precision_request({}, "cuda", "base", None)
    assert recipe["use_amp"] is reference_settings("base")[0]["use_amp"]
    assert (recipe["amp_dtype"], recipe["source"]) == (None, "base recipe")
    with pytest.raises(AssertionError, match="states its dtype"):
        probe._precision_request({"use_amp": True}, "cuda", "base", None)
    # The acceptance's scope says the precision it promises, from M2's request
    # and R3's (`R3_PRECISION`, the one statement R3 is given and R4 judged by).
    scope = probe.m2_acceptance(probe.Report(), False)["scope"]
    assert "M2 and M9 run in mixed precision float16" in scope
    assert "R3 and R4 run in fp32, as the base recipe asks" in scope
    assert probe.R3_PRECISION == {}
    stated = {"use_amp": True, "amp_dtype": "bfloat16"}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(probe, "R3_PRECISION", stated)
        assert "R3 and R4 run in mixed precision bfloat16, as stated" in (
            probe.m2_acceptance(probe.Report(), False)["scope"])


def test_the_scope_names_the_encoder_kind_the_runs_train():
    """The acceptance's scope names the encoder kind it covers, read from the
    specs the probe trains (the demo's and the reference recipe's base), and
    says other kinds are refused by the builder and outside the acceptance
    until the multi-architecture milestone (Q13 (b), X10 (b))."""
    from scripts.setup_demo import demo_task_training
    from src.models.bundle.spec import ModelSpec, reference_spec

    probe = _probe()
    demo = ModelSpec.from_dict(demo_task_training()["base"]["model_spec"]).encoder.kind
    assert probe._demo_encoder_kind() == demo
    scope = probe.m2_acceptance(probe.Report(), False)["scope"]
    assert f"the demo part trains the {demo} encoder" in scope
    assert f"the real part the {reference_spec('base').encoder.kind} encoder" in scope
    assert ("Other encoder kinds are refused by the model-bundle builder and are outside this "
            "acceptance") in scope
    assert "until the multi-architecture milestone" in scope


def _base_settings(**extra):
    """A model-bundle base in the demo's sizes, mixed precision stated off."""
    from scripts.setup_demo import demo_task_training

    return {**demo_task_training()["base"], "use_amp": False, **extra}


def _demo_workspace(probe, directory):
    from src.kg.workspace import write_workspace

    write_workspace(probe._demo_kg(), directory, feature_dim=probe.FEATURE_DIM,
                    samples=probe._budget())
    return directory


def test_the_probe_observes_the_production_restore_and_catches_an_omitted_one(tmp_path):
    """End to end on the CPU: a real GradScaler is given to the Trainer (only
    in this test; on CPU mixed precision takes no effect), a model-bundle base
    pauses, and the probe's own `_train_run` continues it through
    `Trainer.continue_from`. The restore is verified (a scaler recorded while
    float16 is not in effect is verified, not refused); a restore that leaves
    the scaler fresh is caught; and the probe leaves `Trainer.continue_from` as
    it found it."""
    from unittest.mock import patch

    import torch

    from src.training.trainer import Trainer

    probe = _probe()
    settings = _base_settings()
    requested = probe._precision_request(settings, "cpu", "base", None)

    def cpu_scaler(self):
        self.use_amp, self.amp_dtype = False, torch.float32
        self.scaler = torch.amp.GradScaler("cpu", init_scale=8.0)

    original = Trainer.continue_from
    with patch.object(Trainer, "_setup_amp", cpu_scaler):
        ws = _demo_workspace(probe, tmp_path / "ws")
        parent = probe._last(probe._train_run(ws, tmp_path / "parent", "cpu", stage="base",
                                              epochs=2, settings=settings, pause_after=1))
        observed = []
        child = probe._last(probe._train_run(ws, tmp_path / "child", "cpu", stage="base",
                                             resume_from=parent, observe=observed))
        assert Trainer.continue_from is original, "the probe left its observer installed"
        facts = probe._assert_resumed(parent, child, observed, requested=requested)
        assert facts["grad_scaler"] is True
        assert facts["scaler_restore"].startswith("verified at restore")

        def restore_then_forget(self, checkpoint, plan, best_weights=None):
            original(self, checkpoint, plan, best_weights)
            self.scaler = torch.amp.GradScaler("cpu", init_scale=8.0)

        with patch.object(Trainer, "continue_from", restore_then_forget):
            observed = []
            omitted = probe._last(probe._train_run(ws, tmp_path / "omitted", "cpu",
                                                   stage="base", resume_from=parent,
                                                   observe=observed))
        with pytest.raises(AssertionError, match="not restored from the parent's state"):
            probe._assert_resumed(parent, omitted, observed, requested=requested)


# ---------------------------------------------------------------------------
# "Only the remaining epoch" is counted by attempts (the GB10 acceptance of
# 4b1571e: a skipped AMP update in the parent's epoch, none in the child's),
# with batches per epoch from the run's recorded train loader (NEW-probe-2)
# ---------------------------------------------------------------------------
def _record(completed, attempts, updates, steps, end, batches=8, accumulation=1,
            scheduler="none", skipped=0):
    """A model-bundle run's record as `_read_run_checkpoint` reduces it: the
    train loader's batches (`config.effective.loaders.train.batches`), the
    requested accumulation, and the last epoch's count of batches its task
    found nothing to learn from (`train_steps_skipped`); a schedule records
    its own batches and accumulation only when a scheduler is built."""
    schedule = {"type": scheduler, "base_lr": 1e-5}
    if scheduler != "none":
        schedule.update(steps_per_epoch=batches, gradient_accumulation_steps=accumulation)
    stage = {"kind": "original", "start_epoch": 0, "target_epochs": 2, "schedule": schedule}
    return {"progress": {"completed_epochs": completed, "optimizer_attempts": attempts,
                         "optimizer_updates": updates, "scheduler_steps": steps,
                         "stage": stage, "end": end},
            "loaders": {"train": {"batches": batches}},
            "requested": {"gradient_accumulation_steps": accumulation},
            "logs": {"train_loss": 0.5, "train_steps_skipped": skipped}}


@pytest.mark.parametrize("parent,child", [
    ((1, 8, 7, 0), (2, 16, 15, 0)),      # the GB10 run: the parent's epoch skipped one
    ((1, 8, 8, 0), (2, 16, 15, 0)),      # the reverse: the child's epoch skipped one
    ((1, 8, 8, 0), (2, 16, 16, 0)),      # no skip
])
def test_a_skipped_amp_update_in_either_epoch_is_not_an_extra_epoch(parent, child):
    probe = _probe()
    facts = probe._assert_only_the_remaining_epochs(_record(*parent, "paused"),
                                                    _record(*child, "finished"))
    assert facts["child_run_attempts"] == 8 and facts["epochs_run"] == 1
    assert facts["child_run_scheduler_steps"] == 0


def test_a_built_scheduler_is_cross_checked_and_steps_with_the_updates():
    """A run whose operator set a scheduler records its own batches per epoch,
    which must agree with the train loader's, and its steps follow the
    successful updates."""
    probe = _probe()
    facts = probe._assert_only_the_remaining_epochs(
        _record(1, 8, 7, 7, "paused", scheduler="cosine"),
        _record(2, 16, 15, 15, "finished", scheduler="cosine"))
    assert facts["child_run_scheduler_steps"] == 8
    parent = _record(1, 8, 7, 7, "paused", scheduler="cosine")
    parent["loaders"]["train"]["batches"] = 9
    with pytest.raises(AssertionError, match="record different batches per epoch"):
        probe._assert_only_the_remaining_epochs(parent, _record(2, 16, 15, 15, "finished",
                                                                scheduler="cosine"))


def test_a_run_whose_loader_records_no_batches_fails():
    probe = _probe()
    parent = _record(1, 8, 7, 0, "paused")
    parent["loaders"] = {}
    with pytest.raises(AssertionError, match="records no batches per epoch"):
        probe._assert_only_the_remaining_epochs(parent, _record(2, 16, 15, 0, "finished"))


def test_a_batch_with_nothing_to_learn_takes_no_attempt():
    """A model-bundle epoch takes no attempt for a batch its task finds nothing
    to learn from, and counts it (`train_steps_skipped`): eight batches with
    one skipped take seven attempts, and eight would be an attempt too many."""
    probe = _probe()
    facts = probe._assert_only_the_remaining_epochs(
        _record(1, 7, 7, 0, "paused", skipped=1), _record(2, 14, 14, 0, "finished", skipped=1))
    assert (facts["attempts_per_epoch"], facts["batches_skipped"]) == (7, 1)
    with pytest.raises(AssertionError, match="made 8 optimizer attempts"):
        probe._assert_only_the_remaining_epochs(
            _record(1, 7, 7, 0, "paused", skipped=1),
            _record(2, 15, 15, 0, "finished", skipped=1))


def test_a_resume_the_record_cannot_count_fails():
    """The checkpoint holds its last epoch's metrics only, so a child without
    the skipped-batch count, or one that ran more than one remaining epoch, is
    not counted from a guess."""
    probe = _probe()
    child = _record(2, 16, 16, 0, "finished")
    del child["logs"]["train_steps_skipped"]
    with pytest.raises(AssertionError, match="records no count of the batches it skipped"):
        probe._assert_only_the_remaining_epochs(_record(1, 8, 8, 0, "paused"), child)
    with pytest.raises(AssertionError, match="only a resume of one remaining epoch"):
        probe._assert_only_the_remaining_epochs(_record(0, 0, 0, 0, "paused"),
                                                _record(2, 16, 16, 0, "finished"))


@pytest.mark.parametrize("child,message", [
    ((2, 24, 23, 0, "finished"), "made 16 optimizer attempts"),   # the curve replayed
    ((2, 16, 15, 1, "finished"), "scheduler stepped 1 times"),
    ((2, 16, 7, 0, "finished"), "grew by 0"),
    ((1, 8, 7, 0, "paused"), "did not finish its stage"),
])
def test_more_work_than_the_remaining_epoch_or_inconsistent_counts_fail(child, message):
    probe = _probe()
    with pytest.raises(AssertionError, match=message):
        probe._assert_only_the_remaining_epochs(_record(1, 8, 7, 0, "paused"), _record(*child))


@pytest.mark.parametrize("scheduler", ["none", "cosine"])
def test_the_remaining_epoch_is_counted_with_accumulation(scheduler):
    """Seven batches with an accumulation of 3 take three attempts an epoch,
    the tail included; the accumulation is the schedule's where it records
    one, else the run's recorded setting."""
    probe = _probe()
    steps = (0, 0) if scheduler == "none" else (3, 5)
    facts = probe._assert_only_the_remaining_epochs(
        _record(1, 3, 3, steps[0], "paused", batches=7, accumulation=3, scheduler=scheduler),
        _record(2, 6, 5, steps[1], "finished", batches=7, accumulation=3, scheduler=scheduler))
    assert facts["attempts_per_epoch"] == 3


@pytest.mark.parametrize("skip_in", ["parent", "child"])
def test_a_real_skipped_update_through_the_trainer_passes_and_a_replay_fails(tmp_path, skip_in):
    """End to end on the CPU with a real CPU GradScaler, on a model-bundle base
    at the recipe's scheduler "none" (NEW-probe-2: its schedule records no
    batches per epoch, its train loader does, and the epoch its skipped
    batches): a non-finite gradient at one optimizer attempt makes the scaler
    skip that update, in the parent's epoch or in the child's. The probe's
    check passes either way; a restore that replays the original curve from
    epoch 0 is caught."""
    from unittest.mock import patch

    import torch

    from src.training.trainer import Trainer

    probe = _probe()
    # Small batches, so each epoch takes several attempts.
    settings = _base_settings(batch_size=8)

    def cpu_scaler(self):
        self.use_amp, self.amp_dtype = False, torch.float32
        self.scaler = torch.amp.GradScaler("cpu", init_scale=8.0)

    original_update = Trainer._optimizer_update
    # The attempt that skips: the parent's second, or the child's second (the
    # parent's attempts, counted once it paused, plus one).
    skip = {"attempt": 1 if skip_in == "parent" else None}

    def update_with_one_skip(self):
        if self.state.global_step == skip["attempt"]:
            parameter = next(p for p in self.model.parameters() if p.grad is not None)
            parameter.grad = torch.full_like(parameter, float("inf"))
        return original_update(self)

    with patch.object(Trainer, "_setup_amp", cpu_scaler), \
            patch.object(Trainer, "_optimizer_update", update_with_one_skip):
        ws = _demo_workspace(probe, tmp_path / "ws")
        parent = probe._last(probe._train_run(ws, tmp_path / "parent", "cpu", stage="base",
                                              epochs=2, settings=settings, pause_after=1))
        before = probe._read_run_checkpoint(parent)
        assert before["progress"]["optimizer_attempts"] >= 2, "an epoch took one attempt"
        assert not before["progress"]["stage"]["schedule"].get("steps_per_epoch")
        if skip_in == "child":
            skip["attempt"] = before["progress"]["optimizer_attempts"] + 1
        child = probe._last(probe._train_run(ws, tmp_path / "child", "cpu", stage="base",
                                             resume_from=parent, observe=[]))
        after = probe._read_run_checkpoint(child)
        skipped = (before if skip_in == "parent" else after)["progress"]
        assert skipped["optimizer_attempts"] - skipped["optimizer_updates"] == 1, (
            "the injected non-finite gradient did not make the scaler skip an update")
        facts = probe._assert_only_the_remaining_epochs(before, after)
        assert facts["child_run_skipped_updates"] == (1 if skip_in == "child" else 0)

        original_continue = Trainer.continue_from

        def replay_from_the_start(self, checkpoint, plan, best_weights=None):
            original_continue(self, checkpoint, plan, best_weights)
            self.start_epoch = 0

        skip["attempt"] = None
        with patch.object(Trainer, "continue_from", replay_from_the_start):
            replayed = probe._last(probe._train_run(ws, tmp_path / "replayed", "cpu",
                                                    stage="base", resume_from=parent,
                                                    observe=[]))
        with pytest.raises(AssertionError, match="optimizer attempts"):
            probe._assert_only_the_remaining_epochs(before, probe._read_run_checkpoint(replayed))
