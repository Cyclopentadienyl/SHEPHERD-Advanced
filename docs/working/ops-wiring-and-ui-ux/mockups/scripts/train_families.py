"""Train extra demo runs on the snapshot's production trainer (as the probe's
_train_run does): a second base family, frozen and full fine-tunes, a gene
fine-tune, a branch and an extension. Run with cwd = snapshot, sys.path[0] = snapshot.

usage: python train_families.py <snapshot> <workspace>
"""
import sys
from pathlib import Path

snap, ws = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(snap))

from scripts.setup_demo import demo_task_training  # noqa: E402
from scripts.train_model import TrainConfig, train  # noqa: E402
from src.models.bundle.spec import ModelSpec  # noqa: E402
from src.utils.checkpoint_paths import propose_run_dir, run_checkpoint_dir  # noqa: E402

sizes = demo_task_training()
encoder = ModelSpec.from_dict(sizes["base"]["model_spec"]).encoder.kind
runs = ws / "runs" / encoder


def find(name: str) -> Path:
    hits = sorted(p for p in runs.iterdir() if f"_{name}_" in p.name)
    assert hits, name
    return run_checkpoint_dir(hits[-1]) / "last.pt"


def new(stage: str, name: str, size_key: str, **fields):
    run_dir = propose_run_dir(ws, encoder, name)
    given = {"data_dir": str(ws), "run_dir": str(run_dir), "device": "cpu",
             "learning_stage": stage, **sizes[size_key], **fields}
    given = {k: v for k, v in given.items() if v is not None}
    train(TrainConfig(**given), set(given) - {"data_dir", "run_dir"})
    return run_checkpoint_dir(run_dir) / "last.pt"


def cont(stage: str, name: str, parent: Path, task=None, **fields):
    run_dir = propose_run_dir(ws, encoder, name)
    given = {"data_dir": str(ws), "run_dir": str(run_dir), "device": "cpu",
             "learning_stage": stage, "task": task, "resume_from": str(parent), **fields}
    given = {k: v for k, v in given.items() if v is not None}
    train(TrainConfig(**given), set(given) - {"data_dir", "run_dir"})
    return run_checkpoint_dir(run_dir) / "last.pt"


def branch(stage: str, name: str, parent: Path, task=None, **fields):
    run_dir = propose_run_dir(ws, encoder, name)
    given = {"data_dir": str(ws), "run_dir": str(run_dir), "device": "cpu",
             "learning_stage": stage, "task": task, "branch_from": str(parent), **fields}
    given = {k: v for k, v in given.items() if v is not None}
    train(TrainConfig(**given), set(given) - {"data_dir", "run_dir"})
    return run_checkpoint_dir(run_dir) / "last.pt"


base = find("demo-base")
disease = find("demo-disease")
step = sys.argv[3] if len(sys.argv) > 3 else "all"
done = {}
if step in ("all", "frozen"):
    done["frozen"] = new("finetune", "demo-disease-frozen", "disease", task="disease",
                         finetune_from=str(base), num_epochs=1, frozen_parts="graph_side",
                         use_amp=False)
if step in ("all", "gene"):
    done["gene"] = new("finetune", "demo-gene", "gene", task="gene",
                       finetune_from=str(base), num_epochs=1)
if step in ("all", "gene_frozen"):
    done["gene_frozen"] = new("finetune", "demo-gene-frozen", "gene", task="gene",
                              finetune_from=str(base), num_epochs=1, frozen_parts="graph_side",
                              use_amp=False)
if step in ("all", "branch"):
    done["branch"] = branch("finetune", "demo-disease-branch", disease, task="disease",
                            learning_rate=3e-5)
if step in ("all", "extend"):
    done["extend"] = cont("finetune", "demo-disease-extend", disease, task="disease",
                          extend_epochs=1)
if step in ("all", "base2"):
    done["base2"] = new("base", "demo-base-b", "base", num_epochs=1, seed=7)
if step in ("all", "base2_disease"):
    b2 = find("demo-base-b")
    done["base2_disease"] = new("finetune", "demo-b-disease", "disease", task="disease",
                                finetune_from=str(b2), num_epochs=1)
if step in ("all", "base_branch"):
    done["base_branch"] = branch("base", "demo-base-branch", base, learning_rate=2e-4)
for k, v in done.items():
    print("TRAINED", k, v)
