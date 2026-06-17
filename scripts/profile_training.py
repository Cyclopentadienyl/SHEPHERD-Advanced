#!/usr/bin/env python3
"""
Standalone training-throughput profiler (observation-only).

Script: scripts/profile_training.py

WHY THIS EXISTS
    The HGT workload on GB10 is launch-bound. nsys gives GPU-kernel and CUDA-API
    summaries, but to attribute time to *Python lines* (which call forces a sync,
    which call allocates/zeroes buffers, where the copies come from) and to get a
    kernel-time/wall proxy, you need the CPU/operator view. This script provides
    that WITHOUT modifying the production training loop.

WHAT IT REUSES (and what it intentionally disables)
    It builds the production model / dataloader / loss / optimizer / scaler path
    (via scripts/train_model.py builders + the real Trainer), while INTENTIONALLY
    disabling scheduler / callbacks / validation / checkpoint side effects. So the
    per-step compute mirrors production, but it is NOT a full training run.

MEASUREMENT DISCIPLINE
    - gradient_accumulation_steps is FORCED to 1 (this is a per-step profiler; it
      does not emulate production accumulation cadence).
    - optimizer zero_grad() matches production by default (set_to_none=False), so
      Fill/memset/zeroing behaviour we are investigating is preserved. Opt into
      set_to_none with --zero-grad-set-to-none.
    - Clean throughput is measured WITHOUT torch.profiler. Profiler attribution is
      a SEPARATE pass (--profiler); its wall time is instrumented and must NOT be
      read as clean throughput.
    - Attribution modes (--profiler / --sync-debug / --phase-timing) perturb timing
      and/or print volume; run them separately from the clean A/B/C throughput.
    - Emits OBSERVATIONS only. It never claims a speedup.

EXAMPLES
    # clean steady-state throughput (use this for the fp16/bf16/fp32 A/B/C) -- NO profiler
    .venv/bin/python scripts/profile_training.py --data-dir data/workspaces/hpo_2026-06 \
        --conv-type hgt --warmup 10 --measure 50 --amp-dtype float16
    ...                                                                       --amp-dtype bfloat16
    ...                                                                       --amp-dtype fp32

    # operator/stack attribution (separate pass; Chrome trace + self-time tables)
    ... --warmup 10 --measure 20 --amp-dtype float16 --profiler --trace-out profiles/hgt_fp16.json

    # pinpoint implicit syncs to Python lines (short, verbose burst)
    ... --warmup 5 --measure 10 --sync-debug

    # coarse phase attribution (fenced; perturbs overlap -- attribution only)
    ... --warmup 5 --measure 10 --phase-timing
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.amp import autocast

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Reuse the EXACT production builders so data/model construction cannot drift.
import train_model as tm  # noqa: E402  (scripts/train_model.py)
from src.training.trainer import Trainer, TrainerConfig  # noqa: E402
from src.training.loss_functions import LossConfig  # noqa: E402
from src.training.callbacks import Callback  # noqa: E402


# =============================================================================
# Setup — build the production path, with side effects intentionally disabled
# =============================================================================
def build_trainer(config: "tm.TrainConfig", device: str) -> Trainer:
    """Construct the Trainer for its model/optimizer/scaler/loss/amp components."""
    graph_data = tm.load_graph_data(Path(config.data_dir))
    train_loader, _ = tm.create_dataloaders(config, graph_data)
    model = tm.create_model_from_config(config, graph_data)

    trainer_config = TrainerConfig(
        # gradient accumulation FORCED to 1 — this profiler measures per-step work.
        gradient_accumulation_steps=1,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        scheduler_type="none",        # no LR schedule needed for profiling
        use_amp=config.use_amp,
        amp_dtype=config.amp_dtype,
        device=device,
        seed=config.seed,
        checkpoint_dir=str(PROJECT_ROOT / "outputs" / "_profile_throwaway_ckpt"),
        loss_config=LossConfig(
            diagnosis_weight=config.diagnosis_weight,
            link_prediction_weight=config.link_prediction_weight,
            contrastive_weight=config.contrastive_weight,
            ortholog_weight=config.ortholog_weight,
        ),
    )
    # callbacks=[Callback()] (a no-op) suppresses the default EarlyStopping/
    # ModelCheckpoint/etc. NOTE: passing [] would NOT — Trainer uses
    # `callbacks or self._create_default_callbacks()` (trainer.py:214) and an empty
    # list is falsy, so it would still build the defaults.
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=None,
        config=trainer_config,
        callbacks=[Callback()],
    )
    if config.gradient_accumulation_steps != 1:
        print(f"  [note] gradient_accumulation_steps forced to 1 for profiling "
              f"(config had {config.gradient_accumulation_steps}).")
    trainer.model.train()
    return trainer


# =============================================================================
# Per-step compute — single source of truth, mirrors Trainer._train_epoch
# (trainer.py:480-538) minus scheduler/callbacks/metrics/logging.
# =============================================================================
def optimizer_step(trainer: Trainer, set_to_none: bool) -> None:
    """Unscale + clip + step + zero_grad, exactly like production."""
    if trainer.config.max_grad_norm > 0:
        if trainer.scaler is not None:
            trainer.scaler.unscale_(trainer.optimizer)
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.config.max_grad_norm)
    if trainer.scaler is not None:
        trainer.scaler.step(trainer.optimizer)
        trainer.scaler.update()
    else:
        trainer.optimizer.step()
    # Production uses optimizer.zero_grad() (set_to_none=False). set_to_none changes
    # gradient-buffer zero-fill behaviour, which is exactly what we are profiling.
    trainer.optimizer.zero_grad(set_to_none=set_to_none)


def train_step(trainer: Trainer, batch_data: Dict[str, Any], set_to_none: bool) -> None:
    # The harness itself does not call loss.item(); any .item()/sync from production
    # code paths (e.g. MultiTaskLoss building its logging dict) is intentionally
    # PRESERVED so --sync-debug attributes it correctly.
    batch = trainer._move_to_device(batch_data["batch"])
    subgraph_x = trainer._move_to_device(batch_data["subgraph_x_dict"])
    subgraph_edges = trainer._move_to_device(batch_data["subgraph_edge_index_dict"])

    with autocast(trainer.device.type, dtype=trainer.amp_dtype, enabled=trainer.use_amp):
        node_embeddings = trainer.model(subgraph_x, subgraph_edges)
        model_outputs = trainer._compute_model_outputs(
            node_embeddings, batch, subgraph_x, subgraph_edges
        )
        loss, _ = trainer.loss_fn(batch, model_outputs)

    if trainer.scaler is not None:
        trainer.scaler.scale(loss).backward()
    else:
        loss.backward()
    optimizer_step(trainer, set_to_none)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


# =============================================================================
# Pass 1 — clean throughput (NEVER under torch.profiler)
# =============================================================================
def measure_clean_throughput(trainer: Trainer, warmup: int, measure: int, set_to_none: bool) -> None:
    device = trainer.device
    it = iter(trainer.train_dataloader)

    print(f"\n[warmup] {warmup} steps (untimed) ...")
    for _ in range(warmup):
        train_step(trainer, next(it), set_to_none)
    _sync(device)

    print(f"[measure] {measure} steps (clean, no profiler) ...")
    cpu_ms = []  # host time inside the step (NO fence -> does not perturb overlap)
    _sync(device)
    window_start = time.perf_counter()
    for _ in range(measure):
        t0 = time.perf_counter()
        train_step(trainer, next(it), set_to_none)
        cpu_ms.append((time.perf_counter() - t0) * 1e3)
    _sync(device)
    wall = time.perf_counter() - window_start

    print("\n==================== CLEAN THROUGHPUT (observed) ====================")
    print(f"  measured steps      : {measure}")
    print(f"  wall time (fenced)  : {wall:.3f} s   <- true steady-state window")
    print(f"  throughput          : {measure / wall:.3f} steps/s")
    print(f"  host time/step mean : {statistics.mean(cpu_ms):.2f} ms   (no fence)")
    print(f"                median: {statistics.median(cpu_ms):.2f} ms")
    print("  NOTE: window is fenced only at the two boundaries, so 'throughput'")
    print("        reflects real CPU/GPU overlap. If host time/step approaches")
    print("        wall/step, the run is launch-bound.")


# =============================================================================
# Pass 2 — profiler attribution (SEPARATE; instrumented wall != clean throughput)
# =============================================================================
def run_profiler_attribution(trainer: Trainer, warmup: int, active: int,
                             set_to_none: bool, profile_memory: bool,
                             trace_out: Optional[str]) -> None:
    device = trainer.device
    it = iter(trainer.train_dataloader)
    for _ in range(warmup):
        train_step(trainer, next(it), set_to_none)
    _sync(device)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    print(f"\n==================== PROFILER ATTRIBUTION ({active} steps) ====================")
    print("  WARNING: wall time here is profiler-instrumented; do NOT read it as")
    print("           clean throughput. Use Pass 1 for throughput.")
    with torch.profiler.profile(activities=activities, record_shapes=True,
                                with_stack=True, profile_memory=profile_memory) as prof:
        _sync(device)
        w0 = time.perf_counter()
        for _ in range(active):
            train_step(trainer, next(it), set_to_none)
        _sync(device)
        wall = time.perf_counter() - w0

    ka = prof.key_averages()

    # kernel-time / wall PROXY (not Nsight GPU Active%).
    total_cuda_us = sum(
        getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0)) for e in ka
    )
    if total_cuda_us > 0:
        proxy = (total_cuda_us / 1e6) / wall
        print("\n  ---- kernel-time / wall proxy ----")
        print(f"  sum self CUDA/device time : {total_cuda_us / 1e6:.3f} s")
        print(f"  proxy (/ instrumented wall): {proxy * 100:.1f} %")
        print("  NOTE: profiler-derived proxy, NOT Nsight GPU Active%. It can over/under-")
        print("        estimate if kernels overlap or profiler overhead is high. Lower")
        print("        values SUGGEST possible idle headroom; confirm with the Nsight")
        print("        timeline / GPU Active before trusting any recoverable-headroom claim.")

    def _table(prefer):
        for key in prefer:
            try:
                return ka.table(sort_by=key, row_limit=15)
            except Exception:
                continue
        return ka.table(row_limit=15)

    print("\n  ---- top ops by self CUDA/device time ----")
    print(_table(["self_device_time_total", "self_cuda_time_total"]))
    print("\n  ---- top ops by self CPU time (launch/host overhead) ----")
    print(_table(["self_cpu_time_total"]))
    if trace_out:
        Path(trace_out).parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(trace_out)
        print(f"\n  Chrome trace written: {trace_out}")


# =============================================================================
# Attribution — pinpoint implicit syncs to Python lines (short, verbose)
# =============================================================================
def run_sync_debug(trainer: Trainer, warmup: int, steps: int, set_to_none: bool) -> None:
    if trainer.device.type != "cuda":
        print("sync-debug requires CUDA; skipping.")
        return
    if not hasattr(torch.cuda, "set_sync_debug_mode"):
        print("torch.cuda.set_sync_debug_mode unavailable in this torch build; skipping.")
        return
    it = iter(trainer.train_dataloader)
    for _ in range(warmup):
        train_step(trainer, next(it), set_to_none)
    _sync(trainer.device)  # drain BEFORE enabling, so this fence is not warned on

    print(f"\n==================== SYNC-DEBUG ({steps} steps; warns on each implicit sync) ====================")
    torch.cuda.set_sync_debug_mode("warn")
    try:
        for _ in range(steps):
            train_step(trainer, next(it), set_to_none)
        # NOTE: no harness _sync() inside the warn window — we only want warnings
        # from production code paths, not from the harness's own fence.
    finally:
        torch.cuda.set_sync_debug_mode("default")
    _sync(trainer.device)
    print("  (Each UserWarning above carries the Python stack of an implicit sync in")
    print("   production code: loss logging / GradScaler / optimizer / copies / etc.)")


# =============================================================================
# Attribution — coarse phase timing (FENCED; perturbs overlap)
# =============================================================================
def run_phase_timing(trainer: Trainer, warmup: int, steps: int, set_to_none: bool) -> None:
    device = trainer.device
    it = iter(trainer.train_dataloader)
    for _ in range(warmup):
        train_step(trainer, next(it), set_to_none)
    _sync(device)

    acc = {"data": [], "h2d": [], "fwd": [], "bwd": [], "opt": []}
    print(f"\n==================== PHASE TIMING ({steps} steps; FENCED, perturbs overlap) ====================")
    for _ in range(steps):
        t = time.perf_counter(); batch_data = next(it); acc["data"].append((time.perf_counter() - t) * 1e3)

        t = time.perf_counter()
        batch = trainer._move_to_device(batch_data["batch"])
        sx = trainer._move_to_device(batch_data["subgraph_x_dict"])
        se = trainer._move_to_device(batch_data["subgraph_edge_index_dict"])
        _sync(device); acc["h2d"].append((time.perf_counter() - t) * 1e3)

        t = time.perf_counter()
        with autocast(device.type, dtype=trainer.amp_dtype, enabled=trainer.use_amp):
            ne = trainer.model(sx, se)
            mo = trainer._compute_model_outputs(ne, batch, sx, se)
            loss, _ = trainer.loss_fn(batch, mo)
        _sync(device); acc["fwd"].append((time.perf_counter() - t) * 1e3)

        t = time.perf_counter()
        (trainer.scaler.scale(loss) if trainer.scaler is not None else loss).backward()
        _sync(device); acc["bwd"].append((time.perf_counter() - t) * 1e3)

        # mirror production optimizer phase exactly (unscale + clip + step)
        t = time.perf_counter()
        optimizer_step(trainer, set_to_none)
        _sync(device); acc["opt"].append((time.perf_counter() - t) * 1e3)

    total = sum(statistics.mean(v) for v in acc.values())
    print(f"  {'phase':<8}{'mean ms':>10}{'% of step':>12}")
    for k, v in acc.items():
        m = statistics.mean(v)
        print(f"  {k:<8}{m:>10.2f}{(m / total * 100):>11.1f}%")
    print("  NOTE: fences serialize CPU/GPU, so these are ATTRIBUTION shares, not a")
    print("        throughput figure. Read alongside the clean throughput run.")
    print("  NOTE: 'data' is the CONSUMER wait for the next prefetched batch, not the")
    print("        full producer-side subgraph-sampling cost (DiagnosisDataLoader uses")
    print("        a background prefetch thread).")


# =============================================================================
# CLI
# =============================================================================
def main() -> int:
    p = argparse.ArgumentParser(description="Observation-only training-throughput profiler")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--config", type=str, default=None, help="optional YAML (same fields as train_model.py)")
    p.add_argument("--conv-type", type=str, default=None, choices=["hgt", "gat", "sage"])
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--amp-dtype", type=str, default=None, choices=["float16", "bfloat16", "fp32"],
                   help="fp32 == disable AMP; float16 keeps GradScaler; bfloat16 drops GradScaler")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--measure", type=int, default=50)
    p.add_argument("--zero-grad-set-to-none", action="store_true",
                   help="use optimizer.zero_grad(set_to_none=True) (default False = production)")
    p.add_argument("--profiler", action="store_true", help="run a SEPARATE torch.profiler attribution pass")
    p.add_argument("--profile-memory", action="store_true", help="enable profile_memory in the profiler pass")
    p.add_argument("--trace-out", type=str, default=None)
    p.add_argument("--sync-debug", action="store_true", help="short burst with set_sync_debug_mode('warn')")
    p.add_argument("--phase-timing", action="store_true", help="fenced per-phase attribution (perturbs)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Build TrainConfig: YAML (if given) then explicit CLI overrides.
    config = tm.TrainConfig()
    if args.config:
        import yaml
        with open(args.config) as f:
            for k, v in (yaml.safe_load(f) or {}).items():
                if hasattr(config, k):
                    setattr(config, k, v)
    if args.data_dir is not None:
        config.data_dir = args.data_dir
    if args.conv_type is not None:
        config.conv_type = args.conv_type
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.amp_dtype == "fp32":
        config.use_amp = False
    elif args.amp_dtype in ("float16", "bfloat16"):
        config.use_amp = True
        config.amp_dtype = args.amp_dtype
    config.seed = args.seed

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    print("=" * 68)
    print("profile_training.py — OBSERVATION ONLY (no checkpoint, no validation)")
    print("=" * 68)
    print(f"  data_dir   : {config.data_dir}")
    print(f"  conv_type  : {config.conv_type}")
    print(f"  device     : {device}")
    print(f"  use_amp    : {config.use_amp}   amp_dtype: {config.amp_dtype}")
    print(f"  zero_grad  : set_to_none={args.zero_grad_set_to_none}")
    if device == "cuda":
        cap = torch.cuda.get_device_capability(0)
        print(f"  gpu        : {torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]})")

    trainer = build_trainer(config, device)
    print(f"  scaler     : {'GradScaler (fp16)' if trainer.scaler is not None else 'none (bf16/fp32)'}")

    # Pass 1 (always): clean throughput — the number for the AMP A/B/C.
    measure_clean_throughput(trainer, args.warmup, args.measure, args.zero_grad_set_to_none)

    # Separate attribution passes (opt-in).
    if args.profiler:
        run_profiler_attribution(trainer, args.warmup, max(10, args.measure // 2),
                                 args.zero_grad_set_to_none, args.profile_memory, args.trace_out)
    if args.sync_debug:
        run_sync_debug(trainer, args.warmup, min(args.measure, 5), args.zero_grad_set_to_none)
    if args.phase_timing:
        run_phase_timing(trainer, args.warmup, min(args.measure, 20), args.zero_grad_set_to_none)
    return 0


if __name__ == "__main__":
    sys.exit(main())
