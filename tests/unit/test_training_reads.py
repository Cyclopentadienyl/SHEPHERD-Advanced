"""
Training reads each recorded input once, and records those reads.
=================================================================
Contract M2.1, S8. Before the run directories, `config.yaml` and `runtime.json`,
`train` reads the manifest, the graph tensors and both sample files once each
(`read_training_inputs`, `training_fields=True`) and verifies from those reads,
with the `kg.json` identification check (decision 1). The resume parent is read
once with `read_checkpoint`, at the top since M2.3 (`test_resume_parent.py`),
and the trainer restores from the dict that read parsed. `training_input_digests`
is built only from the reads. The acceptance shown here, at the real entry point
(`train`, with `Trainer.train` stubbed), on a disease finetune of a base trained
on the demo workspace (`train_children`): the run that reads every recorded
input, the samples among them, since a base reads the graph alone:

- each input is opened once, and the record equals the digests of the files;
- a graph tensor, a sample file or the parent replaced after its read, by
  atomic rename or in-place rewrite, is the one trained on and recorded (the
  graph tensor is `edge_indices.pt`, which forms the graph view a bundle trains
  on; its input is a learned node table, so `node_features.pt` is read, checked
  and recorded but never trained on); a file
  replaced after the manifest read is refused, the refusal names it, and no run
  artifact and no later stage is reached;
- a republished manifest is refused before the file it covers is read, and is
  irrelevant after both reads;
- a file removed after its read is still recorded by the digest of what was
  read, never as `None`;
- with every path hash poisoned except the one the `kg.json` identification
  check takes (it records nothing), the run completes and no poison reaches the
  record;
- a workspace with no validation samples is refused: a disease finetune
  selects on validation, so it is refused after the file is read and verified,
  once, before anything is recorded; an unrelated split is never opened; a missing
  split is the samples reader's own refusal, before the graph is read; an empty
  training split is refused before anything is written, and so is a split with
  a case not in phenotype normal form (N2);
- training trains on both optional sample fields (`training_fields=True`), on
  the ShepherdGNN path (left for K8-7, which retires it);
- a parent that does not parse leaves no run artifact (it is read at the top
  since M2.3), after a line that names it;
- the path forms are gone, not kept beside the read forms.

Module: tests/unit/test_training_reads.py
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tests.fixtures.generated_workspace import (  # noqa: E402
    recut_cohorts,
    write_generated_workspace,
)
from tests.fixtures.opens import count_opens  # noqa: E402
from tests.fixtures.path_forms import POISON, poison_path_forms  # noqa: E402
from tests.fixtures.replacement import HOWS, hook_read_once, replace_after_read  # noqa: E402
from tests.fixtures.trained_children import (  # noqa: E402,F401 - the demo and workspace fixtures
    demo_fixture,
    finetune_demo_base,
    workspace_fixture,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RECORDED = {
    "train_samples": "train_samples.json",
    "val_samples": "val_samples.json",
    "node_features": "node_features.pt",
    "edge_indices": "edge_indices.pt",
    "num_nodes": "num_nodes.json",
    "split_manifest": "split_manifest.json",
    # contract M2.2: the manifest-bound source graph, which the run verified
    # kg.json against on disk, so its digest equals the file's
    "kg": "kg.json",
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def train_model():
    import scripts.train_model as module

    return module


@pytest.fixture
def captured(monkeypatch):
    """`Trainer.train` replaced by a recorder: everything up to training runs."""
    from src.training import trainer as trainer_module

    seen = {}

    def train(self):
        seen["trainer"] = self
        return {"val_loss": 1.0}

    monkeypatch.setattr(trainer_module.Trainer, "train", train)
    return seen


def _run_artifacts(run):
    """What a run left in its run directory (contract M3a-2); a refused run
    creates none."""
    return sorted(p.name for p in run.iterdir()) if run.exists() else []


@pytest.fixture(scope="module")
def other(demo, tmp_path_factory):
    """Other bytes for every input, each still a file of its kind: the features
    shifted, the last edge of every relation dropped, the cohorts' patients
    renamed, a key added to `kg.json`. Only read, as what a replacement writes."""
    root = tmp_path_factory.mktemp("other")
    features = torch.load(demo / "node_features.pt", weights_only=True)
    torch.save({k: v + 1.0 for k, v in features.items()}, root / "node_features.pt")
    edges = torch.load(demo / "edge_indices.pt", weights_only=True)
    torch.save({k: v[:, :-1].contiguous() for k, v in edges.items()}, root / "edge_indices.pt")
    for split in ("train", "val"):
        rows = json.loads((demo / f"{split}_samples.json").read_text(encoding="utf-8"))
        (root / f"{split}_samples.json").write_text(json.dumps(
            [{**row, "patient_id": f"other-{row['patient_id']}"} for row in rows]))
    kg = json.loads((demo / "kg.json").read_text(encoding="utf-8"))
    (root / "kg.json").write_text(json.dumps({**kg, "workspace": "other"}))
    return root


@pytest.fixture
def parents(train_model, captured, workspace, tmp_path):
    """Two checkpoints of a disease finetune over this workspace: A at epoch 7,
    B at epoch 9 with every weight shifted."""
    finetune_demo_base(train_model, workspace, tmp_path / "seed")
    trainer = captured.pop("trainer")
    a, b = tmp_path / "a.pt", tmp_path / "b.pt"
    trainer.state.epoch = 7
    trainer.save_checkpoint(a)
    with torch.no_grad():
        for parameter in trainer.model.parameters():
            parameter.add_(1.0)
    trainer.state.epoch = 9
    trainer.save_checkpoint(b)
    return a, b


def _forbid_stages(monkeypatch, train_model):
    """Each stage after the preflight fails if it is reached: the resume read
    that follows `read_training_inputs` at once, the finetune parent's read, the
    task data's binding and the Trainer."""
    for name in ("read_resume_checkpoint", "read_finetune_parent", "bind_task_data",
                 "Trainer"):
        def sentinel(*args, _name=name, **kwargs):
            raise AssertionError(f"{_name} ran after a refusal")

        monkeypatch.setattr(train_model, name, sentinel)


# ---------------------------------------------------------------------------
# One read each, and the record is those reads
# ---------------------------------------------------------------------------
def test_each_input_is_opened_once(train_model, captured, workspace, parents, tmp_path):
    parent, _ = parents

    with count_opens() as opens:
        finetune_demo_base(train_model, workspace, tmp_path / "run", resume_from=str(parent))

    for name in (*RECORDED.values(), "kg.json"):
        assert opens.opens(workspace / name) == 1, name
    assert opens.opens(parent) == 1


def test_the_record_is_the_digests_of_the_bytes_read(
    train_model, captured, workspace, parents, tmp_path
):
    parent, _ = parents

    finetune_demo_base(train_model, workspace, tmp_path / "run", resume_from=str(parent))

    record = captured["trainer"].training_input_digests
    expected = {role: _sha(workspace / name) for role, name in RECORDED.items()}
    expected["resume_checkpoint"] = _sha(parent)
    assert record == expected


# ---------------------------------------------------------------------------
# Replacements: used and recorded, or refused
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("how", HOWS)
def test_a_graph_tensor_replaced_after_its_read_is_trained_on_and_recorded(
    train_model, captured, workspace, other, monkeypatch, tmp_path, how
):
    """`edge_indices.pt`: a bundle trains on the graph view formed from it (its
    input is a learned node table, so the features file is not trained on). The
    view trained on is the one the bytes read form, not the replacement's."""
    from src.kg.graph_view import build_graph_view

    target = workspace / "edge_indices.pt"
    read = _sha(target)
    num_nodes = json.loads((workspace / "num_nodes.json").read_text(encoding="utf-8"))
    tensors = torch.load(target, weights_only=True)
    replacement = (other / "edge_indices.pt").read_bytes()
    record = replace_after_read(monkeypatch, after=target, target=target, data=replacement,
                                how=how)

    finetune_demo_base(train_model, workspace, tmp_path / "run")

    assert record.fired == 1
    trainer = captured["trainer"]
    assert trainer.training_input_digests["edge_indices"] == read != _sha(target)
    rule = trainer.model_record.data_roles["graph_view"]["rule"]

    def view_of(edge_index_dict):
        return build_graph_view({"num_nodes_dict": num_nodes, "edge_index_dict": edge_index_dict},
                                rule=rule).digest

    assert trainer.model_record.graph_view.digest == view_of(tensors) \
        != view_of(torch.load(target, weights_only=True))


@pytest.mark.parametrize("how", HOWS)
def test_a_sample_file_replaced_after_its_read_is_trained_on_and_recorded(
    train_model, captured, workspace, other, monkeypatch, tmp_path, how
):
    target = workspace / "train_samples.json"
    read = _sha(target)
    patients = {row["patient_id"] for row in json.loads(target.read_text())}
    record = replace_after_read(monkeypatch, after=target, target=target,
                                data=(other / "train_samples.json").read_bytes(), how=how)

    finetune_demo_base(train_model, workspace, tmp_path / "run")

    assert record.fired == 1
    trainer = captured["trainer"]
    assert trainer.training_input_digests["train_samples"] == read != _sha(target)
    # The disease task's sampler trains on the cases bound from the rows read
    # (every demo training case binds to the view).
    trained_on = trainer.train_dataloader.cases
    assert {case.case_id for case in trained_on} == patients
    assert len(trained_on) == len(patients)


@pytest.mark.parametrize("how", HOWS)
def test_the_parent_replaced_after_its_read_is_restored_and_recorded(
    train_model, captured, workspace, parents, monkeypatch, tmp_path, how
):
    parent, other_parent = parents
    read = _sha(parent)
    record = replace_after_read(monkeypatch, after=parent, target=parent,
                                data=other_parent.read_bytes(), how=how)

    finetune_demo_base(train_model, workspace, tmp_path / "run", resume_from=str(parent))

    assert record.fired == 1
    trainer = captured["trainer"]
    assert trainer.state.epoch == 7, "the state restored is A's, not B's"
    assert trainer.training_input_digests["resume_checkpoint"] == read != _sha(parent)


@pytest.mark.parametrize("how", HOWS)
@pytest.mark.parametrize("name", ["node_features.pt", "edge_indices.pt", "train_samples.json",
                                  "val_samples.json", "kg.json"])
def test_a_file_replaced_after_the_manifest_read_is_refused_naming_it(
    train_model, workspace, other, monkeypatch, tmp_path, name, how
):
    _forbid_stages(monkeypatch, train_model)
    target = workspace / name
    run = tmp_path / "run"
    record = replace_after_read(monkeypatch, after=workspace / "split_manifest.json",
                                target=target, data=(other / name).read_bytes(), how=how)

    with pytest.raises(ValueError) as refused:
        finetune_demo_base(train_model, workspace, run)

    assert record.fired == 1
    assert str(target) in str(refused.value)
    assert "split_manifest.json" in str(refused.value)
    assert _run_artifacts(run) == [], "a run artifact was written before the refusal"


@pytest.mark.parametrize("how", HOWS)
def test_json_that_is_not_samples_replacing_a_split_is_refused_naming_it(
    train_model, workspace, monkeypatch, tmp_path, how
):
    """Training parses before it compares, so a replacement that is JSON but not
    sample rows fails in the parse; the reader names the file there."""
    _forbid_stages(monkeypatch, train_model)
    target = workspace / "val_samples.json"
    run = tmp_path / "run"
    record = replace_after_read(monkeypatch, after=workspace / "split_manifest.json",
                                target=target, data=b'[{"tampered": true}]', how=how)

    with pytest.raises(ValueError, match="could not be parsed as a samples file") as refused:
        finetune_demo_base(train_model, workspace, run)

    assert record.fired == 1
    assert str(target) in str(refused.value)
    assert _run_artifacts(run) == []


def test_a_split_not_in_phenotype_normal_form_is_refused_before_anything_is_written(
    train_model, workspace, monkeypatch, tmp_path
):
    """N2: a case that lists a phenotype twice fails training's parse in the
    shared reader, before any run artifact, naming the file and the producer that
    rebuilds it. The repeat is never removed on the way."""
    _forbid_stages(monkeypatch, train_model)
    target = workspace / "train_samples.json"
    rows = json.loads(target.read_text(encoding="utf-8"))
    rows[0]["phenotype_ids"].append(rows[0]["phenotype_ids"][0])
    target.write_text(json.dumps(rows), encoding="utf-8")
    run = tmp_path / "run"

    with pytest.raises(ValueError, match="not in phenotype normal form") as refused:
        finetune_demo_base(train_model, workspace, run)

    assert str(target) in str(refused.value)
    assert "scripts/build_knowledge_graph.py --generate-samples" in str(refused.value)
    assert _run_artifacts(run) == [], "a run artifact was written before the refusal"


@pytest.mark.parametrize("how", HOWS)
def test_a_manifest_republished_before_the_tensor_read_is_refused(
    train_model, workspace, other, monkeypatch, tmp_path, how
):
    """The run's one reading of the manifest is A; the tensor read after the
    republication is B; the comparison is between those reads."""
    _forbid_stages(monkeypatch, train_model)
    target = workspace / "node_features.pt"
    record = replace_after_read(monkeypatch, after=workspace / "split_manifest.json",
                                target=target, data=(other / "node_features.pt").read_bytes(),
                                how=how, republish="node_features")

    with pytest.raises(ValueError, match="is not the node_features artifact"):
        finetune_demo_base(train_model, workspace, tmp_path / "run")

    assert record.fired == 1


@pytest.mark.parametrize("how", HOWS)
def test_a_manifest_republished_after_both_reads_changes_nothing_recorded(
    train_model, captured, workspace, other, monkeypatch, tmp_path, how
):
    manifest = workspace / "split_manifest.json"
    target = workspace / "node_features.pt"
    read_manifest, read_tensor = _sha(manifest), _sha(target)
    record = replace_after_read(monkeypatch, after=target, target=target,
                                data=(other / "node_features.pt").read_bytes(), how=how,
                                republish="node_features")

    finetune_demo_base(train_model, workspace, tmp_path / "run")

    assert record.fired == 1
    digests = captured["trainer"].training_input_digests
    assert digests["node_features"] == read_tensor
    assert digests["split_manifest"] == read_manifest != _sha(manifest)


@pytest.mark.parametrize("name", ["val_samples.json", "split_manifest.json", "parent"])
def test_a_file_removed_after_its_read_is_still_recorded_never_as_none(
    train_model, captured, workspace, parents, monkeypatch, tmp_path, name
):
    parent, _ = parents
    target = parent if name == "parent" else workspace / name
    read = _sha(target)

    def remove(path):
        if Path(path).resolve() == target.resolve():
            target.unlink()

    hook_read_once(monkeypatch, on_return=remove)

    finetune_demo_base(train_model, workspace, tmp_path / "run", resume_from=str(parent))

    assert not target.exists(), "the hook did not fire"
    digests = captured["trainer"].training_input_digests
    assert None not in digests.values()
    role = {"parent": "resume_checkpoint", "val_samples.json": "val_samples",
            "split_manifest.json": "split_manifest"}[name]
    assert digests[role] == read


# ---------------------------------------------------------------------------
# Only from the readers
# ---------------------------------------------------------------------------
def test_no_path_form_is_reached_and_no_path_digest_is_recorded(
    train_model, captured, workspace, parents, monkeypatch, tmp_path
):
    parent, _ = parents
    poison_path_forms(monkeypatch, true_digest_for=[workspace / "kg.json"])

    finetune_demo_base(train_model, workspace, tmp_path / "run", resume_from=str(parent))

    digests = captured["trainer"].training_input_digests
    assert POISON not in digests.values()
    assert digests["resume_checkpoint"] == _sha(parent)
    for role, name in RECORDED.items():
        assert digests[role] == _sha(workspace / name), role


def test_a_workspace_without_validation_samples_is_refused_after_one_read(
    train_model, captured, workspace, tmp_path
):
    """A workspace whose validation split is empty is refused: the split is
    read and verified, once, and a disease finetune selects its checkpoint on
    validation, so the run is refused for that before the Trainer, the record
    or any run artifact exists (the ShepherdGNN path trained on without
    validation and recorded none instead)."""
    root = recut_cohorts(workspace, val=[])
    run = tmp_path / "run"

    with count_opens() as opens:
        with pytest.raises(ValueError, match="no bound validation case"):
            finetune_demo_base(train_model, root, run)

    assert "trainer" not in captured, "training started without validation"
    assert _run_artifacts(run) == []
    assert opens.opens(root / "val_samples.json") == 1, "read and verified, once"


def test_an_empty_training_split_is_refused_before_anything_is_written(
    train_model, workspace, monkeypatch, tmp_path
):
    """The empty-train refusal runs after verification and before the run
    directories, `config.yaml` and `runtime.json`."""
    _forbid_stages(monkeypatch, train_model)
    root = recut_cohorts(workspace, train=[])
    run = tmp_path / "run"

    with pytest.raises(ValueError, match="No training samples found") as refused:
        finetune_demo_base(train_model, root, run)

    assert str(root / "train_samples.json") in str(refused.value)
    assert _run_artifacts(run) == [], "a run artifact was written before the refusal"


def test_training_trains_on_both_optional_sample_fields(
    train_model, captured, tmp_path
):
    """`training_fields=True` at the entry point: every row's `gene_ids` and
    `candidate_disease_ids` reach the samples trained on. Dropping the flag would
    leave both `None` and change the subgraph seeds silently.

    Left for K8-7: a ShepherdGNN subject. Only its data loader trains on the two
    fields (its subgraph seeds); a bundle finetune binds a case's phenotypes and
    disease alone, and the gene task reads its own fixture. K8-7 retires this
    stage-less run with the path."""
    workspace = write_generated_workspace(tmp_path / "ws", train_ids=[0, 1, 2], val_ids=[3],
                                          training_fields=True)[0]
    train_model.train(train_model.TrainConfig(
        data_dir=str(workspace), run_dir=str(tmp_path / "run"), device="cpu", hidden_dim=16,
        num_layers=1, num_heads=2, num_workers=0, use_amp=False))

    trainer = captured["trainer"]
    for split, loader in (("train", trainer.train_dataloader),
                          ("val", trainer.val_dataloader)):
        rows = json.loads((workspace / f"{split}_samples.json").read_text(encoding="utf-8"))
        expected = {row["patient_id"]: (row["gene_ids"], row["candidate_disease_ids"])
                    for row in rows}
        trained_on = {sample.patient_id: (sample.gene_ids, sample.candidate_disease_ids)
                      for sample in loader.dataset.samples}
        assert trained_on == expected, split
        assert all(genes and candidates for genes, candidates in trained_on.values()), split


def test_an_unrelated_split_is_never_opened(train_model, captured, workspace, tmp_path):
    (workspace / "test_samples.json").write_text("[]")

    with count_opens() as opens:
        finetune_demo_base(train_model, workspace, tmp_path / "run")

    assert opens.opens(workspace / "test_samples.json") == 0
    assert "test_samples" not in captured["trainer"].training_input_digests


def test_a_missing_split_is_the_samples_readers_refusal_before_the_graph_is_read(
    train_model, workspace, monkeypatch, tmp_path
):
    """One locator of a workspace split (verification of the root-cause round,
    P3-2): training stats no sample file itself; `read_samples` locates each
    and refuses a missing one, naming the splits there are, and the samples
    are read before the graph tensors, so that refusal comes before the graph
    is loaded."""
    import src.kg.storage.file_storage as storage

    _forbid_stages(monkeypatch, train_model)
    (workspace / "val_samples.json").unlink()
    monkeypatch.setattr(storage, "read_graph_artifacts",
                        lambda *a, **k: pytest.fail("the graph was read"))
    run = tmp_path / "run"

    with pytest.raises(FileNotFoundError, match=r"Samples file not found: .*val_samples\.json"
                                                r" — this workspace has: train$"):
        finetune_demo_base(train_model, workspace, run)

    assert _run_artifacts(run) == []


def test_a_parent_that_does_not_parse_leaves_no_checkpoint_and_no_metrics(
    train_model, captured, workspace, tmp_path, caplog
):
    """The parent is read at the top of the run (contract M2.3), so a failed read
    leaves no run directory, `config.yaml` or checkpoint, let alone anything that
    looks like a result. The failure follows a line that names the file, since
    torch's own message does not."""
    import logging

    junk = tmp_path / "junk.pt"
    junk.write_bytes(b"not a checkpoint")
    run = tmp_path / "run"

    with caplog.at_level(logging.INFO, logger=train_model.logger.name):
        with pytest.raises(Exception):  # noqa: B017 — torch's own unpickling error
            finetune_demo_base(train_model, workspace, run, resume_from=str(junk))

    assert f"Reading the resume parent: {junk}" in caplog.text

    assert "trainer" not in captured, "training started after a failed parent read"
    assert _run_artifacts(run) == []


# ---------------------------------------------------------------------------
# The path forms are gone, not kept beside the read forms
# ---------------------------------------------------------------------------
def test_the_path_forms_are_gone_from_training():
    from src.core.protocols import TrainerProtocol
    from src.training.trainer import Trainer

    assert not hasattr(Trainer, "load_checkpoint")
    # M3a-3: one restore path, which continues as the run plan decided.
    assert not hasattr(Trainer, "restore_checkpoint")
    assert hasattr(TrainerProtocol, "continue_from")
    assert not hasattr(TrainerProtocol, "load_checkpoint")
    assert not hasattr(TrainerProtocol, "restore_checkpoint")

    source = (REPO_ROOT / "scripts" / "train_model.py").read_text(encoding="utf-8")
    names = {node.id for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Name)}
    names |= {node.attr for node in ast.walk(ast.parse(source))
              if isinstance(node, ast.Attribute)}
    names |= {alias.name for node in ast.walk(ast.parse(source))
              if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    gone = {"verify_graph_artifacts", "verify_generated_cohorts", "verify_graph_source",
            "compute_input_digests", "file_sha256", "load_graph_data", "load_samples",
            "training_input_roles", "load_checkpoint"}
    assert not names & gone, f"train_model.py still names {sorted(names & gone)}"
    assert "torch.load" not in source


def test_continue_from_reads_nothing(train_model, captured, workspace, tmp_path):
    """The parent is read once, by the entry point; continuing restores from
    that read's dict and opens no file (contract M2.1, S8; M3a-3)."""
    from pathlib import Path

    from src.models.bundle.record import MODEL_RECORD_VERSION
    from src.utils.run_plan import plan_continuation

    finetune_demo_base(train_model, workspace, tmp_path / "seed")
    trainer = captured.pop("trainer")
    path = tmp_path / "parent.pt"
    trainer.state.epoch = 4
    trainer.save_checkpoint(path)
    saved = torch.load(path, weights_only=True)
    weights = {name: value.clone() for name, value in saved["state_dict"].items()}
    # A bundle parent is continued under the model record's version, as the
    # entry point plans it (`plan_from_parent(record_version=)`).
    plan = plan_continuation(saved, Path(path), dict(saved["config"]["requested"]), set(),
                             device="cpu", record_version=MODEL_RECORD_VERSION)

    with torch.no_grad():
        for parameter in trainer.model.parameters():
            parameter.add_(1.0)
    trainer.state.epoch = 0

    with count_opens() as opens:
        trainer.continue_from(saved, plan)

    assert sum(opens.counts.values()) == 0, "continue_from opened a file"
    assert trainer.state.epoch == 4
    for name, value in trainer.model.state_dict().items():
        assert torch.equal(value, weights[name])
