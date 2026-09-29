"""
`scripts/migrate_checkpoints.py`, run the way an operator runs it.

Its per-file handler reports any exception as an unreadable checkpoint and
moves on, so when `_infer_conv_type_from_keys` left `src.inference.pipeline`
the script kept exiting 0 while skipping every file with an ImportError. These
tests run the script as a subprocess, so a broken import fails them rather
than hiding inside the handler.
"""
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "migrate_checkpoints.py"

# HGTConv parameter names; `_infer_conv_type_from_keys` reads these as hgt.
HGT_KEYS = (
    "gnn_layers.0.conv.kqv_lin.lins.disease.weight",
    "gnn_layers.0.conv.k_rel.weight",
)


def _flat_workspace(root: Path) -> Path:
    ckpt_dir = root / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    torch.save(
        {"config": {"model_config": {"conv_type": "hgt"}}, "model_state_dict": {}},
        ckpt_dir / "configured.pt",
    )
    torch.save(
        {"config": {}, "model_state_dict": {k: torch.zeros(1) for k in HGT_KEYS}},
        ckpt_dir / "weights_only.pt",
    )
    torch.save(
        {"config": {}, "model_state_dict": {"embedding.weight": torch.zeros(1)}},
        ckpt_dir / "unknown.pt",
    )
    return root


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=300,
    )


def test_dry_run_reads_every_checkpoint_and_moves_none(tmp_path):
    ws = _flat_workspace(tmp_path / "ws")
    result = _run(str(ws))

    assert result.returncode == 0, result.stderr
    assert "could not read" not in result.stdout
    assert "configured.pt  ->  hgt/configured.pt" in result.stdout
    assert "weights_only.pt  ->  hgt/weights_only.pt" in result.stdout
    assert "unknown.pt: could not determine architecture" in result.stdout
    assert "would move 2, skipped 1." in result.stdout
    assert sorted(p.name for p in (ws / "checkpoints").glob("*.pt")) == [
        "configured.pt", "unknown.pt", "weights_only.pt",
    ]


def test_apply_moves_by_config_and_by_weight_names(tmp_path):
    ws = _flat_workspace(tmp_path / "ws")
    result = _run(str(ws), "--apply")

    assert result.returncode == 0, result.stderr
    assert "moved 2, skipped 1." in result.stdout
    ckpt_dir = ws / "checkpoints"
    assert (ckpt_dir / "hgt" / "configured.pt").is_file()
    assert (ckpt_dir / "hgt" / "weights_only.pt").is_file()
    assert [p.name for p in ckpt_dir.glob("*.pt")] == ["unknown.pt"]
