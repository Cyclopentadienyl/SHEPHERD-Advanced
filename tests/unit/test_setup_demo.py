"""The demo setup's model and printed steps.

`--train-model` trains a base and one disease child of it through the
production trainer (K7, TM3: it used to train a ShepherdGNN of its own), and
the command it prints serves that child. Serving refuses `auto` without CUDA,
as training does (contract M3c), so the command names the device the demo was
trained on. These tests hold the command to that, and run it the way the
service would: a demo trained on the CPU here must be served by the very
environment the command sets, on this host without CUDA, and answer a disease
query. Without a model no serving command is printed: since K7 nothing is
served without one, and the training it teaches is the two stages (a base,
then its disease model), never a stage-less run or a shortest-path step (D8).

Module: tests/unit/test_setup_demo.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import setup_demo  # noqa: E402


def _printed(monkeypatch, capsys, tmp_path, *args):
    output = tmp_path / "demo"
    monkeypatch.setattr(sys, "argv", ["setup_demo.py", "--output-dir", str(output), *args])
    setup_demo.main()
    return capsys.readouterr().out, output


def _environment(printed: str) -> dict:
    """The `NAME=value \\` lines of the printed command."""
    return dict(re.findall(r"^\s+(SHEPHERD_[A-Z_]+)=(\S+) \\$", printed, flags=re.M))


def _refusals(caplog) -> list[str]:
    """The ERROR lines logged, each as one line (as a terminal shows it)."""
    import logging

    return [" ".join(record.getMessage().split()) for record in caplog.records
            if record.levelno >= logging.ERROR]


@pytest.mark.parametrize("trained_on, named", [
    ("auto", "cuda"),         # training's auto ran on CUDA, or it refused
    ("cuda", "cuda"),
    ("cuda:1", "cuda:1"),
    ("cpu", "cpu"),
    (" cpu ", "cpu"),
    ("", "cuda"),
])
def test_the_command_names_the_device_the_demo_trained_on(trained_on, named):
    assert setup_demo.serving_device(trained_on) == named


@pytest.mark.parametrize("device, named", [("cpu", "cpu"), ("auto", "cuda")])
def test_the_printed_command_carries_shepherd_device(monkeypatch, capsys, tmp_path, device,
                                                     named):
    checkpoint = tmp_path / "demo" / "runs" / "gatv2" / "demo-disease" / "checkpoints" / "last.pt"
    seen = {}

    def _trained(workspace, num_epochs, device):
        seen["device"] = device
        return checkpoint

    monkeypatch.setattr(setup_demo, "train_demo_model", _trained)
    printed, output = _printed(monkeypatch, capsys, tmp_path, "--train-model",
                               "--device", device, "--epochs", "1")
    assert seen["device"] == device
    assert _environment(printed) == {
        "SHEPHERD_KG_PATH": str(output / "kg.json"),
        "SHEPHERD_CHECKPOINT_PATH": str(checkpoint),
        "SHEPHERD_DATA_DIR": str(output),
        "SHEPHERD_DEVICE": named,
    }


def test_without_a_model_no_serving_command_is_printed(monkeypatch, capsys, tmp_path):
    """K7 (PATH1): a graph is never served without a model, so a workspace
    with no model gets no command to start the API, only the reason and how
    to train one; the hint says what a host without CUDA must add. The
    training it teaches is a base, then its disease model fine-tuned from it
    (K7, D8): no stage-less run and no shortest-path step."""
    printed, output = _printed(monkeypatch, capsys, tmp_path)
    assert _environment(printed) == {}
    assert "uvicorn" not in printed and "path-reasoning" not in printed
    assert "this workspace has nothing to serve yet" in printed
    assert "refuses them (503) until one\nis loaded" in printed
    assert "add --device cpu" in printed
    trains = [line.strip() for line in printed.splitlines() if "train_model.py" in line]
    assert trains == [
        f"python scripts/train_model.py --data-dir {output} --stage base",
        f"python scripts/train_model.py --data-dir {output} --stage finetune \\",
    ]
    assert "--task disease --finetune-from <base run>/checkpoints/last.pt" in printed
    assert "compute_shortest_paths" not in printed and "--epochs" not in printed


def test_a_demo_trained_on_the_cpu_is_a_disease_child_the_printed_environment_serves(
        monkeypatch, capsys, tmp_path):
    """K7 (TM3): `--train-model --device cpu` trains a base and one disease
    child of it through the production trainer, and the command printed
    names the child: a finetuned disease model with a model spec, whose
    parent is the base this run trained, and no ShepherdGNN run. The service
    builds it from the environment the command sets (`build_pipeline`, the
    serving builder), on this host without CUDA, and answers a disease query
    with it."""
    pytest.importorskip("torch")
    import torch

    if torch.cuda.is_available():
        pytest.skip("shows the CPU-only host's case; this host has CUDA")
    printed, output = _printed(monkeypatch, capsys, tmp_path, "--train-model", "--device", "cpu",
                               "--epochs", "1")
    environment = _environment(printed)
    assert environment["SHEPHERD_DEVICE"] == "cpu"

    from src.models.bundle.record import has_model_spec
    from src.utils.checkpoint_io import read_checkpoint

    child = Path(environment["SHEPHERD_CHECKPOINT_PATH"])
    (base,) = (output / "runs" / "gatv2").glob("*_demo-base_*/checkpoints/last.pt")
    assert child.parent.parent.parent == base.parent.parent.parent
    assert child.parent.parent.name.split("_")[1] == "demo-disease" and child.name == "last.pt"
    records = {name: read_checkpoint(path, map_location="cpu", weights_only=True)
               for name, path in (("base", base), ("child", child))}
    child_record = records["child"].checkpoint
    assert has_model_spec(child_record) and has_model_spec(records["base"].checkpoint)
    assert (child_record["learning_stage"], child_record["task"]) == ("finetune", "disease")
    assert records["base"].checkpoint["learning_stage"] == "base"
    assert child_record["parent"]["sha256"] == records["base"].identity.sha256
    assert sorted(p.name for p in (output / "runs").iterdir()) == ["gatv2"], "no ShepherdGNN run"

    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    import src.api.main as api
    from src.core.types import DISEASE_TASK, NodeType, PatientPhenotypes

    bundle = api.build_pipeline()
    identity = bundle.config["served_identity"]
    assert bundle.config["gnn_ready"]
    assert identity["checkpoint"]["state"] == "verified"
    assert identity["checkpoint"]["sha256"] == records["child"].identity.sha256
    assert (identity["settings"]["device"], identity["settings"]["device_requested"]) == (
        "cpu", "cpu")
    result = bundle.pipeline.run(
        patient_input=PatientPhenotypes(patient_id="demo", phenotypes=["HP:0001250",
                                                                        "HP:0001263"]),
        top_k=3, include_explanations=False)
    diseases = {node.id for node in bundle.kg.get_nodes_by_type(NodeType.DISEASE)}
    assert result.result_kind == DISEASE_TASK and len(result.candidates) == 3
    assert {candidate.disease_id for candidate in result.candidates} <= diseases

    # The same command without the device is what serving refuses here.
    from src.inference.pipeline import ServingDeviceUnavailable

    monkeypatch.delenv("SHEPHERD_DEVICE")
    with pytest.raises(ServingDeviceUnavailable):
        api.build_pipeline()


# --------------------------------------------------------------------------- the GPU lock
def test_the_printed_commands_do_not_reload(monkeypatch, capsys, tmp_path):
    """uvicorn's reloader restarts the server on every saved file, and the
    server's shutdown stops the GPU job it runs (the owner's decision of
    2026-10-22), so the printed command has no --reload (without a model none is
    printed). The training hint says the lock must be configured, since
    training refuses without it."""
    checkpoint = tmp_path / "demo" / "runs" / "gatv2" / "demo-disease" / "checkpoints" / "last.pt"
    monkeypatch.setattr(setup_demo, "train_demo_model", lambda *a: checkpoint)
    trained, _ = _printed(monkeypatch, capsys, tmp_path, "--train-model")
    plain, output = _printed(monkeypatch, capsys, tmp_path / "plain")
    assert "python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000\n" in trained
    assert "uvicorn" not in plain
    for printed in (trained, plain):
        assert "--reload" not in printed
    assert "§4.11" in plain
    # The training step's lock is the rule's answer for this shell
    # (`gpu_lock.training_step_lines`, as build_knowledge_graph.py prints it).
    # With no folder chosen: Settings › Data locations first, the variable
    # only as the one-shell alternative (T06; it was the only way it named).
    from src.utils.gpu_lock import WORKSPACES_ROOT_ENV

    monkeypatch.delenv(WORKSPACES_ROOT_ENV, raising=False)
    unset, output = _printed(monkeypatch, capsys, tmp_path / "unset")
    root = output.resolve().parent
    flat = " ".join(line.strip().lstrip("#").strip() for line in unset.splitlines())
    assert "choose it in Settings › Data locations" in flat
    assert f"export {WORKSPACES_ROOT_ENV}={root}" in unset
    assert flat.index("Settings › Data locations") < flat.index(f"export {WORKSPACES_ROOT_ENV}")
    monkeypatch.setenv(WORKSPACES_ROOT_ENV, str(root))
    configured, _ = _printed(monkeypatch, capsys, tmp_path / "unset")
    flat = " ".join(line.strip().lstrip("#").strip() for line in configured.splitlines())
    assert f"takes the GPU lock {root / '.shepherd-gpu.lock'}" in flat
    assert "export" not in configured


def test_train_model_takes_the_gpu_lock_for_the_whole_training(monkeypatch, capsys, tmp_path):
    """Review of 8242b47, P2-4: the demo trains through the real trainer on the
    GPU, so it is a managed GPU job: the lock is held while it trains, and
    released after."""
    from src.utils.gpu_lock import LOCK_FILE_NAME, WORKSPACES_ROOT_ENV, probe

    lock = tmp_path / LOCK_FILE_NAME
    monkeypatch.setenv(WORKSPACES_ROOT_ENV, str(tmp_path))
    seen = {}

    def _trained(workspace, num_epochs, device):
        seen["held_by"] = probe(lock)
        return tmp_path / "demo" / "best.pt"

    monkeypatch.setattr(setup_demo, "train_demo_model", _trained)
    _printed(monkeypatch, capsys, tmp_path, "--train-model")
    assert "a training job (setup_demo.py --train-model" in seen["held_by"]
    assert probe(lock) is None


@pytest.mark.parametrize("case", ["not configured", "busy"])
def test_train_model_is_refused_before_anything_is_written(monkeypatch, tmp_path, caplog, case):
    """Without a configured lock, or with the lock held by another job, the
    demo is refused with exit 1 before the workspace is written or anything
    is trained."""
    import logging

    from src.utils.gpu_lock import LOCK_FILE_NAME, WORKSPACES_ROOT_ENV, GpuLock

    output = tmp_path / "demo"
    monkeypatch.setattr(sys, "argv", ["setup_demo.py", "--output-dir", str(output),
                                      "--train-model"])
    monkeypatch.setattr(setup_demo, "train_demo_model",
                        lambda *a: pytest.fail("trained without the lock"))
    monkeypatch.setattr(setup_demo, "write_workspace",
                        lambda *a, **k: pytest.fail("wrote without the lock"))
    with caplog.at_level(logging.ERROR):
        if case == "not configured":
            monkeypatch.delenv(WORKSPACES_ROOT_ENV, raising=False)
            assert setup_demo.main() == 1
            assert "setup_demo: refused: No workspaces folder is set, so the GPU " \
                "lock is not configured" in caplog.text
            # Review of b1a193f: the lock's words end in their "Ref:", which
            # ran into "Nothing was written."; the Ref now ends the line.
            [said] = _refusals(caplog)
            assert said.endswith("(deployment guide §4.11). Nothing was written. Ref: "
                                 "SHEPHERD_WORKSPACES_ROOT"), said
        else:
            lock = tmp_path / LOCK_FILE_NAME
            monkeypatch.setenv(WORKSPACES_ROOT_ENV, str(tmp_path))
            with GpuLock(lock, job_type="test", command="measure_scorer.py").acquire():
                assert setup_demo.main() == 1
            assert "is held by a test job (measure_scorer.py" in caplog.text
    assert "Nothing was written." in caplog.text and not output.exists()


# --------------------------------------------------------------------------- where it goes
def _save_root(root: Path) -> None:
    """Choose `root` in Settings › Data locations: saved and applied by the
    store the panel's Save writes through (the test's own settings file)."""
    from src.config.deployment_settings import KEY_WORKSPACES_ROOT
    from src.utils import settings_store

    root.mkdir(parents=True, exist_ok=True)
    assert settings_store.save({KEY_WORKSPACES_ROOT: str(root)}).saved
    assert settings_store.apply_saved().applied


@pytest.mark.parametrize("chosen", ["saved in Settings", "set for this command"])
def test_the_default_output_is_demo_in_the_workspaces_folder_in_effect(
        monkeypatch, tmp_path, capsys, chosen):
    """T08 (first run): with the workspaces folder saved in Settings and
    nothing exported, `--train-model` took the GPU lock in that folder but
    wrote the workspace to the repository's `data/workspaces/demo`, so
    Datasets listed nothing. The default is now `demo` in the folder in
    effect, by the one resolver (a variable set for the command wins), and
    the training it runs takes the lock in that same folder. Run from an
    empty directory so the old relative default lands there, not in the
    checkout."""
    from src.utils.gpu_lock import LOCK_FILE_NAME, WORKSPACES_ROOT_ENV, probe

    root = tmp_path / "SHEPHERD-data" / "workspaces"
    if chosen == "saved in Settings":
        monkeypatch.delenv(WORKSPACES_ROOT_ENV, raising=False)
        _save_root(root)
    else:
        root.mkdir(parents=True)
        monkeypatch.setenv(WORKSPACES_ROOT_ENV, str(root))
    started = tmp_path / "cwd"
    started.mkdir()
    monkeypatch.chdir(started)
    seen = {}

    def _trained(workspace, num_epochs, device):
        seen["workspace"], seen["held_by"] = Path(workspace), probe(root / LOCK_FILE_NAME)
        return Path(workspace) / "runs" / "gatv2" / "demo-disease" / "checkpoints" / "last.pt"

    monkeypatch.setattr(setup_demo, "train_demo_model", _trained)
    monkeypatch.setattr(sys, "argv", ["setup_demo.py", "--train-model", "--device", "cpu"])
    assert setup_demo.main() == 0
    assert seen["workspace"] == root.resolve() / "demo"
    assert "a training job (setup_demo.py --train-model" in seen["held_by"]
    assert (root / "demo" / "split_manifest.json").is_file()
    assert list(started.iterdir()) == [], "nothing is written beside where it was started"
    assert f"Workspace: {root.resolve() / 'demo'}" in capsys.readouterr().out
    # The server's own list of the folder in effect now names it.
    from src.api.services.server_lists import list_workspaces

    assert "demo" in [item.name for item in list_workspaces().items]


@pytest.mark.parametrize("folder", ["unset", "refused"])
@pytest.mark.parametrize("train", [True, False], ids=["train-model", "workspace-only"])
def test_with_no_workspaces_folder_in_effect_the_default_is_refused(monkeypatch, tmp_path,
                                                                     caplog, train, folder):
    """With no folder in effect and no `--output-dir` there is nowhere the
    dashboard reads, so the demo is refused before anything is written (and
    before the lock), naming why in the setting's own words; nothing falls
    back to the repository's `data/workspaces/demo`. `--help` says the rule.

    Review of b1a193f: the line ran the setting's trailing "Ref:" into the
    next sentence ("Ref: SHEPHERD_WORKSPACES_ROOT Or name the folder ..."),
    and said "and there is none" of a folder that was set and refused. The
    whole printed line is pinned: the setting's words (which say unset or
    refused), --output-dir, "Nothing was written.", then the one Ref."""
    import logging

    from src.utils.gpu_lock import WORKSPACES_ROOT_ENV

    missing = tmp_path / "no-such-folder"
    if folder == "unset":
        monkeypatch.delenv(WORKSPACES_ROOT_ENV, raising=False)
    else:
        monkeypatch.setenv(WORKSPACES_ROOT_ENV, str(missing))
    started = tmp_path / "cwd"
    started.mkdir()
    monkeypatch.chdir(started)
    monkeypatch.setattr(setup_demo, "write_workspace",
                        lambda *a, **k: pytest.fail("wrote with no folder in effect"))
    monkeypatch.setattr(sys, "argv", ["setup_demo.py", *(["--train-model"] if train else [])])
    with caplog.at_level(logging.ERROR):
        assert setup_demo.main() == 1
    lead = ("setup_demo: refused: the demo is written to the workspaces folder in effect, the "
            "one the dashboard lists.")
    if folder == "unset":
        said = (" No workspaces folder is set, so the server lists no workspace. Choose the "
                "folder whose subfolders are the workspaces in Settings › Data locations; its "
                "Create suggested folder makes SHEPHERD-data/workspaces in the service "
                "account's home folder. No default is assumed. Or name the folder for this "
                "demo with --output-dir. Nothing was written. Ref: SHEPHERD_WORKSPACES_ROOT")
    else:
        said = (" The workspaces folder set at start (Ref: SHEPHERD_WORKSPACES_ROOT) is refused: "
                f"the workspaces root {missing} does not exist or is not a directory. The "
                "operator creates it; nothing creates one, so a mistyped path cannot become a "
                "directory nobody chose. The server lists no workspace until it is corrected. "
                "Or name the folder for this demo with --output-dir. Nothing was written.")
    assert _refusals(caplog) == [lead + said]
    assert list(started.iterdir()) == [] and not missing.exists()


def test_help_says_where_the_demo_goes(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["setup_demo.py", "--help"])
    with pytest.raises(SystemExit):
        setup_demo.main()
    flat = " ".join(capsys.readouterr().out.split())
    assert ("default: demo in the workspaces folder in effect, the one the dashboard lists: "
            "chosen in Settings › Data locations, or SHEPHERD_WORKSPACES_ROOT set for this "
            "command; with neither, the demo is refused unless this is given") in flat
    assert "data/workspaces/demo" not in flat


# --------------------------------------------------------------------------- what it says next
def _trained_demo(monkeypatch, capsys, tmp_path, *args):
    """setup_demo.py --train-model with `args`, the training stubbed, run from
    an empty directory of the test's: what it printed, flattened (its lines
    are wrapped), and the printed text."""
    started = tmp_path / "cwd"
    started.mkdir(exist_ok=True)
    monkeypatch.chdir(started)

    def _trained(workspace, num_epochs, device):
        return Path(workspace) / "runs" / "gatv2" / "x_demo-disease_y" / "checkpoints" / "last.pt"

    monkeypatch.setattr(setup_demo, "train_demo_model", _trained)
    monkeypatch.setattr(sys, "argv", ["setup_demo.py", "--train-model", *args])
    assert setup_demo.main() == 0
    printed = capsys.readouterr().out
    return " ".join(printed.split()), printed


@pytest.mark.parametrize("device", ["cpu", "auto"])
def test_a_trained_demo_leads_with_the_dashboard_and_keeps_uvicorn_as_a_ref(
        monkeypatch, capsys, tmp_path, device):
    """T08 (first run): the last lines printed a `SHEPHERD_KG_PATH=… uvicorn
    … --port 8000` command, a bare API, after the folder was chosen in
    Settings for the dashboard. They now lead with the dashboard (its
    desktop entry or launch_shepherd.sh), which lists the demo from the saved
    folder, and Diagnosis › Load a model…; the uvicorn command stays only as
    a Ref, after them. A CPU-trained demo says the dashboard needs
    SHEPHERD_DEVICE=cpu on a host without CUDA; one trained on CUDA does not."""
    from src.utils.gpu_lock import WORKSPACES_ROOT_ENV

    monkeypatch.delenv(WORKSPACES_ROOT_ENV, raising=False)
    root = tmp_path / "SHEPHERD-data" / "workspaces"
    _save_root(root)
    flat, printed = _trained_demo(monkeypatch, capsys, tmp_path, "--device", device)
    assert ("Open the demo in the dashboard and load its disease model: 1. Start the "
            "dashboard: its desktop entry, or bash launch_shepherd.sh (launch_shepherd.cmd on "
            "Windows). It lists the workspace demo: it is in the workspaces folder saved in "
            "Settings › Data locations.") in flat
    assert ("2. Diagnosis › Load a model…: the run demo-disease of the workspace demo (the "
            "disease model; its base is the run demo-base).") in flat
    cpu = "It was trained on the CPU: on a host without CUDA its load is refused"
    assert (cpu in flat) is (device == "cpu")
    lead, ref = flat.index("Open the demo in the dashboard"), flat.index("Ref: the API alone")
    assert lead < ref < flat.index("python -m uvicorn src.api.main:app")
    assert "Start the API serving" not in printed
    # The Ref keeps the serving environment, one variable a line.
    assert _environment(printed)["SHEPHERD_DATA_DIR"] == str(root.resolve() / "demo")


def test_the_dashboard_steps_say_which_folder_to_choose_when_it_is_not_saved(
        monkeypatch, capsys, tmp_path):
    """A folder given for this command only (the variable), or a demo
    written elsewhere with --output-dir, is not what a dashboard started by
    double click reads: the steps say which folder to choose in Settings,
    never that it is listed."""
    from src.utils.gpu_lock import WORKSPACES_ROOT_ENV

    root = tmp_path / "given"
    root.mkdir()
    monkeypatch.setenv(WORKSPACES_ROOT_ENV, str(root))
    flat, _ = _trained_demo(monkeypatch, capsys, tmp_path, "--device", "cpu")
    assert (f"It lists the workspace demo once its workspaces folder is {root.resolve()}: this "
            "command was given that folder by SHEPHERD_WORKSPACES_ROOT, so choose the same "
            "folder in Settings › Data locations if it is not saved there.") in flat
    elsewhere = tmp_path / "elsewhere" / "my-demo"
    flat, _ = _trained_demo(monkeypatch, capsys, tmp_path, "--device", "cpu", "--output-dir",
                            str(elsewhere))
    assert (f"It lists the workspace my-demo once {elsewhere.parent.resolve()} is its "
            "workspaces folder: choose that folder in Settings › Data locations.") in flat
    assert "the run demo-disease of the workspace my-demo" in flat


@pytest.mark.parametrize("where", ["saved in Settings", "set for this command", "elsewhere"])
def test_a_cpu_demo_says_to_stop_the_running_dashboard_before_its_cpu_start(
        monkeypatch, capsys, tmp_path, where):
    """Review of b39bd3f: the CPU line said "start it as SHEPHERD_DEVICE=cpu
    bash launch_shepherd.sh, or the load is refused". With the folder saved
    through the dashboard, that dashboard normally still runs, and a second
    start of the same configuration only opens it, keeping its device
    (`test_service_unit.py::test_a_second_start_of_a_running_configuration_opens_it`
    starts it with SHEPHERD_DEVICE=cpu), so Load was still refused. In every
    case the steps now say to stop the running one first (Settings › Stop
    service), then start it with SHEPHERD_DEVICE=cpu, before the Load of step 2."""
    from src.utils.gpu_lock import WORKSPACES_ROOT_ENV

    root = tmp_path / "SHEPHERD-data" / "workspaces"
    args = ["--device", "cpu"]
    if where == "saved in Settings":
        monkeypatch.delenv(WORKSPACES_ROOT_ENV, raising=False)
        _save_root(root)
    else:  # the lock's folder given for this command; the demo there, or elsewhere
        root.mkdir(parents=True)
        monkeypatch.setenv(WORKSPACES_ROOT_ENV, str(root))
        if where == "elsewhere":
            args += ["--output-dir", str(tmp_path / "elsewhere" / "demo")]
    flat, _ = _trained_demo(monkeypatch, capsys, tmp_path, *args)
    said = ("It was trained on the CPU: on a host without CUDA its load is refused unless the "
            "dashboard runs with SHEPHERD_DEVICE=cpu (development hosts only; deployment guide "
            "§4.8). A dashboard already running keeps the device it was started with, and a "
            "second start only opens it. So stop it first (Settings › Stop service), then "
            "start it as SHEPHERD_DEVICE=cpu bash launch_shepherd.sh.")
    assert said in flat
    assert "start it as SHEPHERD_DEVICE=cpu bash launch_shepherd.sh, or the load" not in flat
    start, load = flat.index("1. Start the dashboard"), flat.index("2. Diagnosis › Load a model…")
    assert start < flat.index(said) < load
