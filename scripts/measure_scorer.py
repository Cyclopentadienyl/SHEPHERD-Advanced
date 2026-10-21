#!/usr/bin/env python
r"""
Measure the disease scorer — Mode C.
====================================
Every patient in a cohort scored against every disease in the knowledge graph,
with the model the deployed pipeline builds and raw cosine. This is the formal
Test's measurement (`docs/working/PLAN_TEST_RESULTS.md`, "This is Mode C").

**A thin CLI.** Argument parsing, one read of each input, manifest assembly, one
call into `src.evaluation.measurement.run_mode_c`, and publishing what it
returns. No measurement logic lives here, so the harness stays testable without
a subprocess.

**Two kinds of cohort, named by the argument that names them** (contract M4,
decision 8): exactly one of `--split` and `--cohort`.

    # A registered test cohort version: the formal measurement. The report is
    # kept in the registry's report store; --record registers it in the ledger
    # beside the checkpoint.
    python scripts/measure_scorer.py \
        --checkpoint <workspace>/runs/<arch>/<run>/checkpoints/best.pt \
        --data-dir <workspace> \
        --cohort mygene2-2026-10 --registry /srv/shepherd/registry \
        [--training-evidence DIR ...] [--record] [--deterministic] \
        [--expect-checkpoint-sha256 HEX] [--expect-version-sha256 HEX] [--outcome FILE]

    # A generated split. --split val is the checkpoint-selection split, not
    # held-out data, and its report is not registrable (decision 9).
    python scripts/measure_scorer.py \
        --checkpoint checkpoints/best.pt \
        --data-dir data/processed \
        --split val \
        --output measurement.json

**A registered version's report goes to the report store**,
`<registry>/versions/<label>/reports/<run id>/`: `ranks.json` (the per-case
ranks, patient-level, so kept in the registry outside the repository) and then
`report.json`, which names the ranks by the digest written and is the run's
commit point. Each file is staged and renamed into place (`publish_bytes`), and
the run directory is new for every run, so no report is ever overwritten.
`--record` registers it right away through the one registration function the
later `scripts/record_evaluation.py --report` uses; a refused registration exits
3, and the report is kept unless it is no longer the one the run published (see
the exit status below). Each run is appended to the version's usage history
right after its report is published (`src/evaluation/report_store.py`), so a
run from the command line counts as one started from the Test tab.

**For a job that chose its inputs beforehand** (M5's job runner):
`--expect-checkpoint-sha256` and `--expect-version-sha256` refuse the run before
anything is measured when the checkpoint or the version's `cohort.json` read now
is not the one chosen; `--outcome FILE` writes a small JSON once the report is
published and again at the end (`_write_outcome`): the report's path, run id and
digest, and whether it was registered and why. FILE must not exist yet, so a
file an earlier run left is never read as this one's. The file says what the
run did; the report store and the ledger stay the record, and
`src/evaluation/registration.report_state` reads kept and registered from them.

**`--deterministic`** turns on PyTorch's deterministic algorithms (never
warn-only), makes cuDNN deterministic and turns its benchmark off, and sets
`CUBLAS_WORKSPACE_CONFIG` itself, all before CUDA is initialised
(`_execution_settings`). An operation with no deterministic implementation
refuses the run (`DeterministicExecutionUnsupported`); it never falls back. The
manifest observes the settings in effect, and they are in the semantics digest,
so a deterministic run is another key (D5).

**A generated split's report goes to `--output`**, with `<output stem>_ranks.json`
beside it, both staged and renamed into place; an existing `--output` is
replaced, as it always was.

Exit status, run as a program (`_exit_as_program`): 0 measured (and, with
`--record`, registered or already held); 1 refused before anything was written,
with one message and no traceback, or measured but the report could not be
written, with a message saying what may be on disk (or the run failed after
writing it, with a message naming it); 2 the command line was rejected; 3
measured, but not registered: the registration was refused and the report is
kept, or the report was no longer the one the run published
(`ReportNotAsPublished`: it may be gone, so it is not called kept); the message
says which, and `registration.report_state` reads whether it is kept. **A
cancel** (SIGINT: the job runner's stop on POSIX, or Ctrl+C) prints one line,
`measure_scorer: cancelled ...`, and no traceback, then ends the process by
SIGINT on POSIX (the signal's default action, raised again once the line is
printed), so a parent reads "ended by SIGINT" as before; elsewhere it exits
130 (on Windows the runner's stop is CTRL_BREAK_EVENT, which this program does
not catch). The run unwinds first: a staged file is removed, the GPU lock
released, nothing more is written, and what it had published stays as it is.
A cancel while the program's modules still load (`_cancelled_while_loading`)
prints the same line and ends the same way. **Only the program ends itself**
(`_exit_as_program`, the `__main__` entry): an in-process caller of
`_run_as_program` (the tests) gets the line and then the KeyboardInterrupt
back, so its own cleanup runs. `main` itself raises its refusals, typed, for
the tests and the deployment probe that call it.

**One managed GPU job at a time** (M5): `main` takes the deployment's GPU lock
(`src/utils/gpu_lock.py`) before the device is resolved or anything is read and
holds it until it returns; a busy lock is a refusal naming its holder (exit 1,
nothing written). With no lock configured the log says so and the run goes on.

Module: scripts/measure_scorer.py
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import platform
import re
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

#: What a cancelled run prints, once, instead of a KeyboardInterrupt traceback.
CANCELLED_LINE = (
    "measure_scorer: cancelled by an interrupt (SIGINT); the run stopped where it was and "
    "nothing more was written. The report store and the ledger say whether a report was "
    "kept and registered."
)


def _end_as_cancelled() -> int:
    """End the process as the interrupt would have, without its traceback: on
    POSIX, SIGINT again with the default action, so the parent (the job
    runner) reads the exit status as "ended by SIGINT"; elsewhere, 130
    (128 + SIGINT, the shells' convention). **Called only where this file runs
    as the program** (`_exit_as_program`, `_cancelled_while_loading`), never
    from a function an in-process caller runs: it would end that caller's
    process, its cleanup and `finally` blocks unrun."""
    if os.name == "posix":
        with contextlib.suppress(OSError, ValueError, AttributeError):
            sys.stderr.flush()
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGINT)
    return 130


def _cancelled_while_loading(signum: int, frame: Any) -> None:
    """SIGINT while this program's modules load, before `_run_as_program`
    (importing torch takes seconds, and a Cancel right after Start lands
    there): the same line and ending as a cancel during the run, not an
    import's traceback. Nothing has been read or written yet."""
    with contextlib.suppress(OSError, ValueError, AttributeError):
        print(CANCELLED_LINE, file=sys.stderr)
    raise SystemExit(_end_as_cancelled())


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _cancelled_while_loading)

import torch  # noqa: E402 - after the loading guard, which must cover this import

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.caveats import COHORT_ARGUMENT_HELP, SPLIT_ARGUMENT_HELP
from src.evaluation.cohort import GENERATED_SPLITS, CohortProvenance, resolve_cohort
from src.evaluation.cohort_registry import RANKS_FILE, REPORT_FILE, cohort_version_block
from src.utils.gpu_lock import take_gpu_lock
from src.utils.interrupts import interrupt_deferred

if TYPE_CHECKING:
    from src.evaluation.cohort_registry import CohortVersionRead
    from src.evaluation.measurement import ModeResult
    from src.kg.artifacts import DatasetRecord, ManifestRead
    from src.kg.storage.file_storage import GraphArtifactsRead, SamplesRead
    from src.utils.checkpoint_io import CheckpointRead

logger = logging.getLogger(__name__)

#: What Mode C ranks each patient against, as its manifest records it. There is
#: no expansion to describe: the candidates are every disease in the graph.
CANDIDATE_CONSTRUCTION = "every disease in the knowledge graph"

#: How the scored model was built: through `build_shepherd_model`, the builder
#: the deployed pipeline uses.
MODEL_CONSTRUCTION = "production (build_shepherd_model)"

#: A run's two files in the report store, in the order they are published (the
#: ranks, then the report that names them by digest: the run's commit point),
#: and the version as a report records it (`cohort_version_block`) are the
#: registry's own definitions, imported above: the registration compares a kept
#: report with the same block of the version the store holds it under.

#: The exit status of a run that measured and published its report but whose
#: `--record` registration was refused (the report kept), or found the report
#: no longer as published (it may be gone): not a refusal before writing
#: (exit 1, one message), and not a success, since the ledger holds nothing new.
EXIT_NOT_REGISTERED = 3

#: The cuBLAS workspace settings PyTorch documents as deterministic (CUDA 10.2
#: and later); `--deterministic` sets the first when the variable is unset.
DETERMINISTIC_CUBLAS_CONFIGS: Tuple[str, ...] = (":4096:8", ":16:8")

#: The version of the `--outcome` file's shape.
OUTCOME_SCHEMA_VERSION = 1

#: How a refusal says nothing was written, in the words this codebase uses
#: ("Nothing was written.", R10's "nothing was read or written"), so the
#: program entry adds the sentence only to a message that does not say it.
_SAYS_NOTHING_WRITTEN = re.compile(r"nothing was (?:read or )?written", re.IGNORECASE)


class UnparseableCheckpoint(ValueError):
    """A checkpoint that was read but does not load as a weights-only
    checkpoint, refused under the name it was given (`path`); torch's error is
    chained, and its text kept on one line."""

    def __init__(self, path: Path, exc: Exception) -> None:
        reason = " ".join(str(exc).split())
        cause = f"{type(exc).__name__}: {reason}" if reason else type(exc).__name__
        super().__init__(f"{path} could not be parsed as a weights-only checkpoint ({cause})")
        self.path = path


class DeterministicExecutionUnsupported(ValueError):
    """`--deterministic` was asked for and this run cannot honour it: an
    operation it needs has no deterministic implementation on the device, or
    the process is in a state where deterministic execution cannot be set up.
    Raised before anything is written; the run does not fall back to
    non-deterministic execution or to another device (D5)."""


class ExpectedDigestMismatch(ValueError):
    """The checkpoint or the cohort version read is not the one the run was
    started for (`--expect-checkpoint-sha256`, `--expect-version-sha256`):
    refused before anything is measured or written."""


class ReportNotWritten(Exception):
    """The measurement ran, and publishing its report failed.

    **Neither a `ValueError` nor an `OSError`, on purpose.** Those are the
    refusals `main` raises before anything is written, which the program entry
    prints as such, with "Nothing was written." A publication failure comes
    after the first write may have happened, so its message says what may be on
    disk instead (`_publication_failure`), and the cause is chained.
    """


class FailedAfterPublishing(Exception):
    """The report was published, and what followed it failed: the console, or a
    registration that raised rather than refused. Not a refusal either, for the
    same reason as `ReportNotWritten`; its message says where the report is."""


class MeasurementInputs(NamedTuple):
    """What a run read once each: the workspace files it checked, the cohort, and
    the checkpoint, whose recorded graph roles were compared with `bound`
    (contract M2.1, S7; M2.4).

    Carried whole from `read_measurement_inputs` to `measurement_input_digests`
    and the manifest, so the tensors and samples Mode C scores, the checkpoint
    its model is built from and the digests its manifest records come from the
    same reads and cannot be paired with another read of a file that changed in
    between.

    Exactly one of `cohort` (a generated split) and `version` (a registered
    cohort version) is set. The other generated split, read for its binding and
    the disjointness check, is not kept: it is checked, not consumed, and no
    role records it; nor are R10's evidence files, which are reduced to counts.
    """

    #: The generated split, resolved; `None` for a registered version.
    cohort: Optional[CohortProvenance]
    #: The registered version as resolved (its one read of `cohort.json`); `None`
    #: for a generated split.
    version: Optional[CohortVersionRead]
    #: The version as its report records it (`cohort_version_block`), built from
    #: that one read; `None` for a generated split.
    cohort_version: Optional[Dict[str, Any]]
    manifest: ManifestRead
    graph: GraphArtifactsRead
    samples: SamplesRead
    checkpoint: CheckpointRead
    #: The digest the run's one manifest reading binds for each of the four
    #: graph roles, `kg` included, as `verify_graph_reads` returned it after
    #: comparing the three tensors read with it. Kept, not discarded: it is the
    #: `kg` role (M2.1 decision 2), what M2.4 compares the checkpoint with, and
    #: what a registered version's graph is compared with (M4 decision 7).
    bound: Dict[str, str]
    #: The workspace's dataset record, read once and verified against this
    #: manifest reading and `bound["kg"]` (contract M3b), for both kinds of
    #: cohort. Its summary goes on the manifest, so a number measured over a
    #: synthetic workspace says so.
    dataset_record: DatasetRecord
    #: The R10 block (contract §5.4) for a registered version; `None` for a
    #: generated split, whose relation to training is the allocation's.
    r10: Optional[Dict[str, Any]]


def read_measurement_inputs(
    checkpoint: Path,
    data_dir: Path,
    device: torch.device,
    *,
    split: Optional[str] = None,
    cohort: Optional[str] = None,
    registry: Optional[Path] = None,
    training_evidence: Sequence[Path] = (),
    expect_checkpoint_sha256: Optional[str] = None,
    expect_version_sha256: Optional[str] = None,
) -> MeasurementInputs:
    """Read the manifest, the graph tensors, the cohort and the checkpoint once
    each, and verify from those reads (contract M2.1, S7; M4 §6).

    Exactly one of `split` (a generated split) and `cohort` (a registered
    version's label, with `registry`) is given; `training_evidence` applies to a
    registered version only (R10).

    **What the run was started for** (M5's job runner): with
    `expect_version_sha256`, the version's `cohort.json` read must have that
    digest, compared right after `resolve_version` reads it; with
    `expect_checkpoint_sha256`, the checkpoint read must, compared right after
    it is read and before R3. Either refuses (`ExpectedDigestMismatch`) before
    anything is measured or written, so a file replaced after it was chosen is
    never measured under the choice. No file is read again for it: each
    comparison is with the digest of the read the run makes anyway.

    **Before any model is built or any output written.** `main` calls this
    right after the device is resolved, and every refusal below raises before
    the first byte is written.

    **A generated split is resolved first, and that only stats.** A split that
    is not an identifier, or not one of the generator's, or a workspace with no
    manifest, is refused before any file is opened. It then reads its sibling
    split too, with the default fields, so `verify_cohort_reads` checks both
    splits' bindings (R2), the disease sets, and the manifest's disjointness
    claim and the measured disjointness.

    **A registered version, in this order** (M4 §6): the evidence directories
    are checked (listed, nothing read); `resolve_version` reads `cohort.json`
    once; the workspace manifest and tensors are read; the version's
    `samples.json` is read once and refused **before it is parsed** unless its
    digest is the one `cohort.json` binds; R1, the `kg.json` identification and
    R5 as for a split; then the version's four graph digests and dataset record
    must be this run's (`verify_cohort_binding`), so a version mapped against
    another graph is refused before the checkpoint is read; then the checkpoint
    and R3; then R10 over the evidence candidates, each read at most once and
    compared before it is parsed. The workspace's `train_samples.json` and
    `val_samples.json` are read only as R10 candidates.

    **The digests compared are those of the bytes parsed.** Each tensor's digest
    and, for a generated split, each sample file's digest is compared with the
    one manifest reading; a version's samples with the digest its manifest
    binds. A file replaced after its read is the one scored and recorded. A
    graph tensor, or a generated split's sample file, replaced before its read
    and after the manifest's is refused by name; so is a version's samples
    replaced before their read. The checkpoint is bound by R3 to the graph and
    recorded as read. Nothing is hashed again by path.

    `kg.json` is never parsed by measurement, so it gets the identification
    check (M2.1 decision 1): the file on disk is compared with this run's one
    manifest reading, and its digest is not recorded as a file read; the `kg`
    role is the manifest-bound source graph (`bound["kg"]`, decision 2).

    **The dataset record is required, for both kinds of cohort** (contract M3b):
    `kg.provenance.json` is read once and refused unless it is the record this
    manifest reading binds and names the graph the tensors were verified as.

    The graph tensors load on CPU (`map_location="cpu"`, the shared reader's
    default and what this run passed before); `encode_full_graph` moves them.
    The checkpoint keeps this site's options: `map_location` the run's device,
    `weights_only=True`. It is read at its resolved path, so the read's identity
    is the file measured and `--record` files the ledger beside it (M4 decision
    15) even if a link to it is retargeted during the run. It is read after the
    workspace checks, so a refused workspace reads no checkpoint, and its
    recorded graph roles are compared
    with `bound` (`verify_training_record`, contract M2.4): a checkpoint from
    before the contract, or from a same-shaped other graph, is refused here. A
    file that does not load as a weights-only checkpoint is refused naming it
    (`_read_checkpoint`).
    """
    from src.kg.artifacts import (
        GRAPH_ARTIFACTS,
        check_unparsed_kg_json,
        read_split_manifest,
        verify_dataset_record,
        verify_graph_reads,
    )
    from src.kg.storage.file_storage import read_graph_artifacts
    from src.utils.fingerprint import verify_training_record

    if (split is None) == (cohort is None):
        raise ValueError("measure exactly one of a generated split and a registered cohort")
    if split is not None and (registry is not None or training_evidence
                              or expect_version_sha256 is not None):
        raise ValueError("a registry, training evidence and an expected version digest apply "
                         "to a registered cohort only")

    if split is not None:
        resolved, siblings = _resolve_generated(data_dir, split)
        version = cohort_version = candidates = None
    else:
        resolved, siblings = None, ()
        version, cohort_version, candidates = _resolve_registered(
            data_dir, cohort, registry, training_evidence
        )
        _expect(expect_version_sha256, version.identity.sha256,
                f"the cohort version {cohort!r} ({version.identity.path})")

    manifest = read_split_manifest(data_dir)
    graph = read_graph_artifacts(data_dir, map_location="cpu")
    if version is None:
        samples, checked = _read_generated(data_dir, split, siblings)
    else:
        samples, checked = _read_version_samples(version), {}

    bound = verify_graph_reads(manifest, graph.reads)
    check_unparsed_kg_json(data_dir, manifest)
    dataset_record = verify_dataset_record(manifest, bound["kg"])
    if version is None:
        _verify_generated(manifest, split, samples, checked)
    else:
        _verify_version_binding(version, bound, dataset_record)
    # The sibling is checked, not consumed: nothing holds it past this point, so
    # it is not resident while the checkpoint is read.
    del checked

    # **Read at its resolved path**, so the read's identity names the file whose
    # bytes were measured, and `--record`'s ledger directory is fixed now: a link
    # retargeted during the run cannot move this run's record beside other weights.
    read = _read_checkpoint(checkpoint, device)
    _expect(expect_checkpoint_sha256, read.identity.sha256,
            f"the checkpoint {checkpoint} (read at {read.identity.path})")
    # **The model scored must have been trained on this graph** (contract M2.4):
    # the checkpoint's recorded graph roles against `bound`, whose `kg` is the
    # manifest-bound source graph and whose tensor digests equal this run's
    # reads. Before any model is built or any output written; a refusal raises.
    verify_training_record(
        read.checkpoint,
        bound,
        tuple(GRAPH_ARTIFACTS),
        recorded_by=f"checkpoint {checkpoint}",
        compared_with=f"the workspace {data_dir}",
    )
    r10 = None if version is None else _r10(read, samples, candidates)

    logger.info(
        "Read %d samples of %s; checkpoint %s (sha256 %s)",
        len(samples.samples),
        split if version is None else f"cohort version {cohort_version['label']}",
        read.identity.path, read.identity.sha256[:12],
    )
    logger.info(
        "Dataset record %s: binding %s, origin %s (sha256 %s)",
        dataset_record.identity.path.name, dataset_record.binding, dataset_record.origin,
        dataset_record.identity.sha256[:12],
    )
    return MeasurementInputs(resolved, version, cohort_version, manifest, graph, samples, read,
                             bound, dataset_record, r10)


def _expect(expected: Optional[str], read: str, what: str) -> None:
    """Refuse a read whose digest is not the one the run was started for."""
    if expected is not None and read != expected:
        raise ExpectedDigestMismatch(
            f"{what} holds bytes {read[:12]}..., not the {expected[:12]}... this run was "
            "started for: it changed after it was chosen, so it is not measured under that "
            "choice. Nothing was written."
        )


def _resolve_generated(data_dir: Path, split: str) -> Tuple[CohortProvenance, Tuple[str, ...]]:
    """The split and its sibling, resolved. Stat only."""
    resolved = resolve_cohort(data_dir, split)
    siblings = tuple(name for name in GENERATED_SPLITS if name != split)
    for name in siblings:
        resolve_cohort(data_dir, name)
    return resolved, siblings


def _resolve_registered(
    data_dir: Path, label: str, registry: Optional[Path], training_evidence: Sequence[Path]
) -> Tuple[CohortVersionRead, Dict[str, Any], List[Path]]:
    """The evidence candidates (listed, nothing read) and the version (one read
    of its `cohort.json`), with the block its report records.

    The evidence directories come first: they are the operator's arguments, and
    one that is not a directory is refused before anything is read.
    """
    from src.evaluation.cohort_registry import resolve_version
    from src.evaluation.r10 import evidence_candidates

    if registry is None:
        raise ValueError("a registered cohort is resolved in a registry; none was given")
    candidates = evidence_candidates(data_dir, training_evidence)
    version = resolve_version(registry, label)
    return version, cohort_version_block(version), candidates


def _read_generated(
    data_dir: Path, split: str, siblings: Tuple[str, ...]
) -> Tuple[SamplesRead, Dict[str, SamplesRead]]:
    """The split's samples and its sibling's, one read each, default fields."""
    from src.kg.storage.file_storage import read_samples

    return read_samples(data_dir, split), {name: read_samples(data_dir, name) for name in siblings}


def _read_version_samples(version: CohortVersionRead) -> SamplesRead:
    """The version's `samples.json`, one read, refused before it is parsed
    unless its digest is the one the version's `cohort.json` binds.

    A mismatch or a missing file means the registry was altered after the
    import, since a committed version's files are never rewritten; either is
    refused naming both files.

    **The samples read are the count the version records.** `cohort.json`'s
    `samples.n` and `counts.ranked_eligible` become the report's and the ledger's
    denominators, so they must be the number of samples this run read and will
    rank; a `cohort.json` edited after the import, whose samples digest still
    matches, is otherwise recorded with counts that describe no run.
    """
    from src.evaluation.cohort_registry import CohortVersionRefused
    from src.kg.storage.file_storage import SamplesDigestMismatch, read_samples_file

    bound = version.manifest["samples"]["sha256"]
    try:
        samples = read_samples_file(version.samples_path, expected_sha256=bound)
    except SamplesDigestMismatch as exc:
        raise CohortVersionRefused(
            f"{version.samples_path} is not the samples {version.identity.path} binds "
            f"(its bytes are {exc.sha256[:12]}..., the version binds {bound[:12]}...): "
            "the registry was altered after the import, and its bytes were not parsed. "
            "Nothing was written."
        ) from None
    except FileNotFoundError:
        raise CohortVersionRefused(
            f"{version.samples_path} is missing, though {version.identity.path} binds it: "
            "the registry was altered after the import. Nothing was written."
        ) from None
    n_read = len(samples.samples)
    recorded_n = version.manifest["samples"]["n"]
    ranked_eligible = version.manifest["counts"]["ranked_eligible"]
    if not n_read == recorded_n == ranked_eligible:
        raise CohortVersionRefused(
            f"{version.samples_path} holds {n_read} samples, and {version.identity.path} "
            f"records samples.n {recorded_n!r} and counts.ranked_eligible "
            f"{ranked_eligible!r}: the registry was altered after the import, and its "
            "counts would be recorded as this run's denominators. Nothing was written."
        )
    return samples


def _verify_generated(
    manifest: ManifestRead, split: str, samples: SamplesRead, checked: Mapping[str, SamplesRead]
) -> None:
    """R2: both generated splits against the one manifest reading."""
    from src.evaluation.cohort import verify_cohort_reads

    verify_cohort_reads(
        manifest,
        {name: samples if name == split else checked[name] for name in GENERATED_SPLITS},
    )


def _verify_version_binding(
    version: CohortVersionRead, bound: Mapping[str, str], dataset_record: DatasetRecord
) -> None:
    """The cohort↔graph relation (M4 decision 7): the version's graph digests
    and dataset record against this run's, before the checkpoint is read."""
    from src.evaluation.cohort_registry import verify_cohort_binding

    verify_cohort_binding(version.manifest, bound, dataset_record.identity.sha256)


def _read_checkpoint(checkpoint: Path, device: torch.device) -> CheckpointRead:
    """The checkpoint, one read at its resolved path, or a refusal naming it.

    `read_checkpoint` passes on whatever torch raises for the bytes, as its
    contract says, and for bytes that are not a weights-only checkpoint that is
    a different type for each kind of junk (`UnpicklingError`; a `KeyError` or
    an `IndexError` for text; an `EOFError` for an empty file; a `RuntimeError`
    for a truncated archive) whose message names no file. So the measurement
    names it, as the graph readers do for a tensor file (`_read_tensors`), with
    torch's error chained (`UnparseableCheckpoint`). A file that cannot be read
    keeps the read's own `OSError`, which names it.
    """
    from src.utils.checkpoint_io import read_checkpoint

    try:
        return read_checkpoint(Path(checkpoint).resolve(), map_location=device, weights_only=True)
    except (OSError, MemoryError):
        # A MemoryError is not a property of the file. Converting it would send
        # a capacity shortfall to whoever reads the message as a corrupt file.
        raise
    except RuntimeError as exc:
        # Neither is torch's own report of a failed allocation (c10's CPU
        # allocator raises a plain RuntimeError, a device `torch.OutOfMemoryError`),
        # nor, since the checkpoint is loaded onto the run's device, the
        # device's: a CUDA error (`torch.AcceleratorError`), or a CUDA device
        # on a host without CUDA, which `_resolve_device` refuses first for `main`.
        if (isinstance(exc, (torch.OutOfMemoryError, torch.AcceleratorError))
                or "can't allocate memory" in str(exc)
                or (device.type == "cuda" and not torch.cuda.is_available())):
            raise
        raise UnparseableCheckpoint(checkpoint, exc) from exc
    except Exception as exc:  # noqa: BLE001 — torch raises a different type for each kind of junk
        raise UnparseableCheckpoint(checkpoint, exc) from exc


def _r10(read: CheckpointRead, samples: SamplesRead, candidates: Sequence[Path]) -> Dict[str, Any]:
    """R10 from the checkpoint's own record (its one read) and the candidates.

    Recorded, never enforced (contract §5.4). A candidate that cannot be read
    for a reason other than its absence refuses the run rather than reading as
    "not found": it may have been the evidence.
    """
    from src.evaluation.r10 import compute_r10

    # R3 has passed, so the checkpoint is a mapping that records its inputs.
    recorded = read.checkpoint.get("training_input_digests")
    try:
        return compute_r10(recorded, samples.samples, candidates)
    except OSError as exc:
        raise ValueError(
            f"an R10 evidence candidate could not be read ({type(exc).__name__}: {exc}); it "
            "may hold the inputs the checkpoint records, so the run does not report them "
            "as not found. Nothing was written."
        ) from exc


def measurement_input_digests(inputs: MeasurementInputs) -> Dict[str, str]:
    """The record of what a run consumed, by role, built only from its reads
    (contract M2.1, S7).

    **Only from the readers' results.** Every file role is the digest of the
    bytes a reader parsed: the checkpoint the model was built from, the cohort
    scored (a generated split's file or a version's `samples.json`), the three
    graph tensors encoded, and, for a generated split, the manifest those were
    verified against. Nothing here hashes a path, so a file replaced after its
    read cannot change what is recorded, and a digest that was never taken
    cannot be recorded as `None`.

    **`kg` is not a file this run read.** Measurement never parses `kg.json`;
    its role is the digest the run's manifest binds for the source graph, the
    graph the tensors scored were verified as the export of (M2.1 decision 2).
    The identification check compared `kg.json` on disk with that digest at the
    start of the run; it records nothing itself.

    `split_manifest` is recorded for a generated split only. A registered
    version's run reads the manifest too, for the graph binding, but no
    allocation cut its cohort, so the manifest says nothing about how it was
    made; the version's own manifest is identified by `cohort_version`.

    Nothing globs the directory, so an unrelated file beside these does not
    change the record of a run that never read it.
    """
    from src.kg.artifacts import GRAPH_TENSOR_ROLES

    reads: Dict[str, Any] = {
        "checkpoint": inputs.checkpoint.identity,
        "samples": inputs.samples.identity,
        **{role: inputs.graph.reads.get(role) for role in GRAPH_TENSOR_ROLES},
    }
    if inputs.version is None:
        reads["split_manifest"] = inputs.manifest.identity

    missing = sorted(role for role, read in reads.items() if read is None)
    if missing:
        raise ValueError(
            f"no read of {', '.join(missing)}, so there is no digest to record; a "
            "measurement record names only bytes a reader parsed"
        )
    digests = {role: read.sha256 for role, read in reads.items()}
    digests["kg"] = inputs.bound["kg"]
    return digests


def _resolve_device(requested: str) -> Tuple[torch.device, bool]:
    """Return the device and whether it is CUDA.

    **CUDA is a hard project requirement**, so `auto` falling back to CPU would
    silently produce a number that looks institutional and is not: different
    kernels, different reduction orders, and no statement at all about the
    hardware the tool actually runs on. `auto` therefore fails without CUDA rather
    than substituting a device nobody asked for.

    Explicit `--device cpu` stays available, because development in a container
    without a GPU is real work — but the manifest records `cuda_executed: false`,
    so a development number cannot later be quoted as one produced on the
    deployment hardware.

    An explicit `cuda` on a host without CUDA is refused here too, before
    anything is read, rather than by torch when the checkpoint is loaded onto
    the device after every workspace check has run.

    The boolean means only "this ran on CUDA". Whether the run is acceptable to
    the institution depends on which artifacts it consumed, which is checked by a
    person against the recorded digests, not here.
    """
    if requested == "auto":
        if not torch.cuda.is_available():
            raise SystemExit(
                "--device auto requires CUDA, which is not available here.\n"
                "Hardware acceleration is a hard requirement for deployment, CUDA "
                "is currently the only supported backend, and every deployment "
                "environment has it, so a silent CPU fallback would produce a "
                "number that reads as institutional but is not.\n"
                "Pass --device cpu explicitly for development; the run is then "
                "recorded as cuda_executed=false."
            )
        return torch.device("cuda"), True
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "--device cuda requires CUDA, which is not available here.\n"
            "Pass --device cpu explicitly for development; the run is then "
            "recorded as cuda_executed=false."
        )

    device = torch.device(requested)
    return device, device.type == "cuda"


@contextlib.contextmanager
def _execution_settings(deterministic: bool):
    """The run's numerical execution settings, set for the run and put back as
    they were after it, however it ends (`main` is also called in process, by
    the tests and the deployment probe, and torch's settings are global).

    **With `--deterministic`** (D5): PyTorch's deterministic algorithms on,
    **never warn-only**; cuDNN deterministic, its benchmark off; and
    `CUBLAS_WORKSPACE_CONFIG` set to `DETERMINISTIC_CUBLAS_CONFIGS[0]` when it
    is unset, which cuBLAS reads when CUDA first creates its handle -- so this
    runs before the device is resolved. Refused (`DeterministicExecutionUnsupported`,
    nothing read or written) rather than run in a state that is not what it
    would be recorded as: the variable set to a value PyTorch does not document
    as deterministic, or CUDA already initialised in this process with the
    variable unset. The manifest observes the settings in effect
    (`build_manifest`), and all three are in the semantics digest.

    **Without it**, nothing is changed, but deterministic algorithms left on in
    warn-only mode by an in-process caller are refused: the manifest would
    record a deterministic run whose non-deterministic operations only warned.
    """
    if (not deterministic and torch.are_deterministic_algorithms_enabled()
            and torch.is_deterministic_algorithms_warn_only_enabled()):
        raise DeterministicExecutionUnsupported(
            "deterministic algorithms are on in warn-only mode in this process, so the run "
            "would be recorded as deterministic although a non-deterministic operation "
            "would only warn. Run it as its own process, or with --deterministic. Nothing "
            "was read or written."
        )
    if not deterministic:
        yield
        return
    config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if config is not None and config not in DETERMINISTIC_CUBLAS_CONFIGS:
        raise DeterministicExecutionUnsupported(
            f"--deterministic: CUBLAS_WORKSPACE_CONFIG is {config!r}, which PyTorch does not "
            f"document as deterministic ({' or '.join(DETERMINISTIC_CUBLAS_CONFIGS)}). Unset "
            "it, and the run sets the first, or set one of those. Nothing was read or written."
        )
    if config is None and torch.cuda.is_initialized():
        raise DeterministicExecutionUnsupported(
            "--deterministic: CUDA was initialised in this process before "
            "CUBLAS_WORKSPACE_CONFIG could be set, so cuBLAS may already run without it. Run "
            "the measurement as its own process. Nothing was read or written."
        )
    saved = (torch.are_deterministic_algorithms_enabled(),
             torch.is_deterministic_algorithms_warn_only_enabled(),
             torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    if config is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = DETERMINISTIC_CUBLAS_CONFIGS[0]
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        yield
    finally:
        torch.use_deterministic_algorithms(saved[0], warn_only=saved[1])
        torch.backends.cudnn.deterministic = saved[2]
        torch.backends.cudnn.benchmark = saved[3]
        if config is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)


def _is_determinism_error(exc: BaseException) -> bool:
    """Whether torch refused an operation because deterministic execution was
    asked for: its messages name it ("does not have a deterministic
    implementation", "Deterministic behavior was enabled ..."), and nothing
    else this run raises does."""
    return "deterministic" in str(exc).lower()


def build_manifest(args: argparse.Namespace, graph_data: Dict[str, Any],
                   n_samples: int, device: torch.device, *,
                   digests: Mapping[str, str],
                   dataset_record: Mapping[str, Any],
                   regime: Any,
                   cohort_version: Optional[Mapping[str, Any]] = None,
                   r10: Optional[Mapping[str, Any]] = None,
                   cuda_executed: Optional[bool] = None,
                   model: Any = None) -> Any:
    """Build the manifest of one Mode C run.

    `digests` is the record of what the run read, from
    `measurement_input_digests` (contract M2.1, S7); nothing here hashes a file.

    `dataset_record` is the verified record's summary (`DatasetRecord.summary`,
    contract M3b): its binding state and its origin, recorded apart.

    `regime` is the `AutocastRegime` `encode_full_graph` reported for the
    embeddings Mode C scores: they are computed before this manifest is built,
    so the regime observed here could be a different one. It is required, and
    `run_mode_c` refuses a manifest whose regime is not the embeddings'
    (`assert_manifest_describes_regime`).

    `model` is optional and is used only to **observe** whether what ran was a
    `torch.compile` wrapper object. Omitting it records
    `torch_compile_wrapped=None` — "not observed", which is deliberately not the
    same claim as "not compiled". Neither value says a compiled graph executed;
    see `observe_torch_compile_wrapper`.

    **The environment comes from the one collector training uses**
    (`collect_environment` for the device in effect, `code_revision` of this
    repository), and `software_revision` is that revision: the same commit
    `git rev-parse HEAD` names, so taking it from there changes no semantics
    digest. Recorded, never enforced (contract §5.3).

    **The kind of cohort follows from `cohort_version`, never from an argument
    that states it.** With a version (`cohort_version_block`) the manifest is a
    registered cohort's: `split` is the version's role, `cohort_kind` is
    `supplied`, the relations are a registered cohort's and `r10` is required.
    Without one it is the generated split `args.split`, with no R10. The
    manifest refuses any other combination when it is built.
    """
    from src.evaluation.measurement import (
        METRIC_SCHEMA_VERSION,
        MeasurementManifest,
        expected_relations,
        observe_torch_compile_wrapper,
    )
    from src.utils.fingerprint import compute_fingerprint
    from src.utils.run_environment import code_revision, collect_environment

    registered = cohort_version is not None
    code = code_revision(PROJECT_ROOT)
    return MeasurementManifest(
        mode="C",
        split=cohort_version["role"] if registered else args.split,
        metric_schema_version=METRIC_SCHEMA_VERSION,
        cohort_kind="supplied" if registered else "generated",
        n_samples=n_samples,
        candidate_construction=CANDIDATE_CONSTRUCTION,
        batch_size=args.batch_size,
        score_semantics="raw cosine, no eta mixture and no shortest-path term",
        model_construction=MODEL_CONSTRUCTION,
        canonical_tie_policy_version="score-desc-then-global-id-asc/v1",
        checkpoint_path=str(args.checkpoint),
        data_dir=str(args.data_dir),
        graph_fingerprint=compute_fingerprint(graph_data),
        artifact_digests=dict(digests),
        dataset_record=dict(dataset_record),
        cuda_executed=(
            device.type == "cuda" if cuda_executed is None else cuda_executed
        ),
        software_revision=code["revision"],
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        device=str(device),
        dtype=str(torch.get_default_dtype()),
        # Observed, not asserted. `run_mode_c` opens no autocast context, but a
        # caller can, and a literal `False` then described a run that did not
        # happen.
        amp_enabled=regime.enabled,
        amp_dtype=regime.dtype,
        torch_compile_wrapped=observe_torch_compile_wrapper(model),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic if torch.cuda.is_available() else None,
        cudnn_benchmark=torch.backends.cudnn.benchmark if torch.cuda.is_available() else None,
        cohort_version=dict(cohort_version) if registered else None,
        relations=expected_relations(registered=registered),
        r10=dict(r10) if r10 is not None else None,
        environment={"facts": collect_environment(str(device)), "code": code},
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure the disease scorer: every patient against every disease (Mode C)"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    cohort = parser.add_mutually_exclusive_group(required=True)
    cohort.add_argument("--split", metavar="{train,val}", help=SPLIT_ARGUMENT_HELP)
    cohort.add_argument("--cohort", metavar="LABEL", help=COHORT_ARGUMENT_HELP)
    parser.add_argument("--registry", type=Path, default=None, metavar="DIR",
                        help="The registry --cohort was imported into: an absolute path "
                             "outside the repository that already exists. Required with "
                             "--cohort; refused with --split")
    parser.add_argument("--training-evidence", type=Path, action="append", default=None,
                        metavar="DIR",
                        help="A directory whose *_samples.json files (not recursive) may "
                             "hold the training or validation inputs the checkpoint "
                             "records, for R10, after the workspace's own splits. Only "
                             "bytes whose digest the checkpoint records count; a scope "
                             "whose bytes are found nowhere is unverifiable, never "
                             "zero. Repeatable; --cohort only")
    parser.add_argument("--record", action="store_true",
                        help="Register the kept report in evaluations.json beside the "
                             "checkpoint, as scripts/record_evaluation.py --report "
                             "would. A refused registration exits 3; the report is kept "
                             "unless it is no longer the one the run published. --cohort "
                             "only: a generated split's report is not registered")
    parser.add_argument("--output", type=Path, default=None,
                        help="Where a generated split's measurement JSON is written; the "
                             "per-sample ranks go beside it, as <output stem>_ranks.json. "
                             "Required with --split; refused with --cohort, whose report "
                             "the registry's report store keeps")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Patients scored per chunk. Recorded on the manifest: "
                             "it can move the numbers at floating-point level")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                        help="auto requires CUDA and fails without it. Explicit cpu "
                             "is permitted for development and records "
                             "cuda_executed=false in the manifest")
    parser.add_argument("--deterministic", action="store_true",
                        help="Run with PyTorch's deterministic algorithms (never warn-only), "
                             "cuDNN deterministic and its benchmark off, and "
                             f"CUBLAS_WORKSPACE_CONFIG={DETERMINISTIC_CUBLAS_CONFIGS[0]} when "
                             "unset. An operation with no deterministic implementation "
                             "refuses the run; it never falls back. Recorded on the manifest, "
                             "so a deterministic run is a different measurement key")
    parser.add_argument("--expect-checkpoint-sha256", type=_sha256_argument, default=None,
                        metavar="HEX",
                        help="Refuse before measuring unless the checkpoint read has this "
                             "SHA-256 (the one chosen for this run)")
    parser.add_argument("--expect-version-sha256", type=_sha256_argument, default=None,
                        metavar="HEX",
                        help="Refuse before measuring unless the cohort version's "
                             "cohort.json read has this SHA-256. --cohort only")
    parser.add_argument("--outcome", type=Path, default=None, metavar="FILE",
                        help="Write the run's outcome as JSON to FILE (a new file: it must "
                             "not exist, and its directory must): the report's path, run id "
                             "and digest, and whether it "
                             "was registered and why. Written once the report is published "
                             "and again at the end; nothing is written to it when the run is "
                             "refused before its report. --cohort only")
    args = parser.parse_args(argv)
    args.training_evidence = list(args.training_evidence or [])
    _refuse_mixed_arguments(parser, args)
    return args


def _sha256_argument(value: str) -> str:
    """A SHA-256 as the lists show it: 64 lowercase hexadecimal digits."""
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a SHA-256 (64 lowercase hexadecimal digits)")
    return value


def _refuse_mixed_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Refuse an argument that belongs to the other kind of cohort (M4 decision
    8), before the device is resolved or anything is read. `parser.error` exits
    with status 2 and one message."""
    if args.split is not None:
        given = [flag for flag, value in (("--registry", args.registry),
                                          ("--training-evidence", args.training_evidence),
                                          ("--record", args.record),
                                          ("--expect-version-sha256",
                                           args.expect_version_sha256),
                                          ("--outcome", args.outcome)) if value]
        if given:
            parser.error(
                f"{', '.join(given)} apply to a registered cohort version (--cohort LABEL "
                "--registry DIR) only: a generated split has no version, its report is not "
                "registered and R10 is not computed for it. Nothing was read or written."
            )
        if args.output is None:
            parser.error(
                "--split needs --output: a generated split's report is written where you "
                "say. Nothing was read or written."
            )
        return
    if args.registry is None:
        parser.error(
            "--cohort needs --registry DIR, the registry the version was imported into. "
            "Nothing was read or written."
        )
    if args.output is not None:
        parser.error(
            "--output does not apply to --cohort: a registered version's report is kept in "
            "the registry's report store, <registry>/versions/<label>/reports/<run id>/. "
            "Nothing was read or written."
        )


def _publish_generated(result: ModeResult, output: Path) -> Tuple[Path, Path]:
    """A generated split's report at `output` and its ranks beside it.

    Both are encoded before anything is written, so a value JSON cannot hold
    refuses with nothing on disk; then the ranks and the report are each staged
    and renamed into place, the report last. An existing `output` is replaced,
    as it always was.
    """
    from src.utils.file_modes import publish_bytes

    ranks_path = output.with_name(f"{output.stem}_ranks.json")
    report_bytes = json.dumps(result.to_dict(), indent=2, allow_nan=False).encode("utf-8")
    ranks_bytes = json.dumps(result.to_ranks(), indent=2, allow_nan=False).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    publish_bytes(ranks_path, ranks_bytes)
    publish_bytes(output, report_bytes)
    return output, ranks_path


def _publish_to_store(
    result: ModeResult, inputs: MeasurementInputs
) -> Tuple[str, Path, str]:
    """A registered version's report in a new run directory of its report store.

    Encoded as every registry file is (`encode_json`) and checked to encode
    before the run directory exists. Then `ranks.json` is published, and
    `report.json` last, carrying ``"ranks": {"file", "sha256"}`` with the digest
    `publish_bytes` returned for the bytes it wrote: the report is the commit
    point, and a run directory without one is an interrupted write that nothing
    registers. Returns the run id, the report's path, and the digest
    `publish_bytes` returned for the report bytes it wrote: `--record` registers
    only a report that still holds exactly those bytes (`MeasuredRun`).
    """
    from datetime import datetime, timezone

    from src.evaluation.cohort_registry import create_report_dir, encode_json
    from src.utils.file_modes import publish_bytes

    report = result.to_dict()
    ranks_bytes = encode_json(result.to_ranks())
    encode_json(report)  # refuses a value JSON cannot hold before anything exists
    run_id, run_dir = create_report_dir(
        inputs.version, inputs.checkpoint.identity.sha256, datetime.now(timezone.utc)
    )
    ranks_sha256 = publish_bytes(run_dir / RANKS_FILE, ranks_bytes)
    report["ranks"] = {"file": RANKS_FILE, "sha256": ranks_sha256}
    report_path = run_dir / REPORT_FILE
    report_sha256 = publish_bytes(report_path, encode_json(report))
    return run_id, report_path, report_sha256


def _sentence(exc: BaseException) -> str:
    """`exc`'s message ending in a full stop, so a sentence can follow it: an
    `OSError`'s never does ("[Errno 28] No space left on device: '...'")."""
    text = str(exc).rstrip()
    return text if text.endswith(".") else f"{text}."


def _publication_failure(
    exc: Exception, inputs: MeasurementInputs, output: Optional[Path]
) -> ReportNotWritten:
    """A failed publication, saying what it may have left on disk.

    `publish_bytes` leaves its target as it was, and each kind of cohort
    publishes its report last, so a failure leaves at most what was published
    before it: for a generated split, the ranks beside `output` (whatever was
    there before, or nothing); for a registered version, a new run directory,
    empty or holding only `RANKS_FILE`, with no report to commit it. Which of
    those happened depends on the step that failed, so the message says "may".

    **`create_report_dir`'s own refusal is kept as it is.** Its text already
    says what it created ("Nothing was written." before the run directory, or
    "Only the empty run directory ... was created" after it), and is the only
    `RegistryRefused` publication raises.
    """
    from src.evaluation.cohort_registry import REPORTS_DIR, RegistryRefused

    if isinstance(exc, RegistryRefused):
        return ReportNotWritten(str(exc))
    if inputs.version is None:
        ranks = output.with_name(f"{output.stem}_ranks.json")
        return ReportNotWritten(
            f"{_sentence(exc)} {output} was left as it was (whatever was there before, or "
            f"no file), and {ranks} may already hold this run's ranks, so the two may not "
            "belong to one run. Measure again to write both."
        )
    return ReportNotWritten(
        f"{_sentence(exc)} No {REPORT_FILE} was published, so this run kept no report and "
        f"nothing can register it. A new run directory in "
        f"{inputs.version.directory / REPORTS_DIR} may exist, empty or holding only "
        f"{RANKS_FILE}: an interrupted write, which nothing registers. Measure again to "
        "keep a report."
    )


def _failure_after_publishing(
    exc: Exception, inputs: MeasurementInputs, report_path: Path
) -> FailedAfterPublishing:
    """A failure after the report was published, saying where it is.

    For a registered version, whether `--record` got as far as the ledger is
    not known here, so the message gives the command that registers the report
    later: run for a report already registered, it says so and writes nothing.
    """
    said = f"{_sentence(exc)} The report was written: {report_path}."
    if inputs.version is not None:
        measured = Path(inputs.checkpoint.identity.path)
        said += (" To register it, or to see that it already is: "
                 f"{register_later_command(report_path, measured)}")
    return FailedAfterPublishing(said)


def register_later_command(report_path: Path, checkpoint: Path) -> str:
    """The command that registers a kept report later, printed for the operator
    to paste.

    **`--checkpoint` names the file this run measured**: its read's path,
    resolved when it was read. `record_evaluation.py` then hashes that one file
    against the report, whatever its name (a directory is searched for `*.pt`
    files only), and files the ledger beside it, where the service reads it.

    **Each path is double-quoted**, so a path with a space (a Windows user
    profile, "My Models") stays one argument whether the line is pasted into a
    POSIX shell, cmd or PowerShell. `shlex.quote` would be right for POSIX
    shells only: cmd does not understand its single quotes. Double quotes
    protect ordinary paths; a path holding a double quote, `$`, a backtick, `%`
    or `!` is expanded by one shell or another and has to be quoted by hand.
    """
    return (f'python scripts/record_evaluation.py --report "{report_path}" '
            f'--checkpoint "{checkpoint}"')


def _registration_outcome(requested: bool, **fields: Any) -> Dict[str, Any]:
    """The ``registration`` block of the `--outcome` file. Before an attempt it
    says so (``registered`` is `None`: not yet established); without
    `--record` it says the report was not registered because none was asked,
    and ``why`` may say what a later registration would meet
    (`_offer_later_registration`)."""
    if not requested:
        return {"requested": False, "registered": False, "outcome": "not_requested",
                "sequence": None, "ledger": None,
                "why": fields.get("why", "--record was not given; the report is kept for a "
                                         "later registration")}
    block = {"requested": True, "registered": None, "outcome": None, "sequence": None,
             "ledger": None, "why": "registration had not been attempted when this was written"}
    block.update(fields)
    return block


def _register(
    report_path: Path, report_sha256: str, inputs: MeasurementInputs
) -> Tuple[int, Dict[str, Any]]:
    """`--record`: the report this run published into the ledger beside the
    checkpoint this run read. Returns the status and what the `--outcome` file
    says about the registration.

    Through `register_report`, the function `record_evaluation.py --report`
    uses, given what this run holds about itself (`MeasuredRun`): its checkpoint
    read's identity -- the path read at its resolved form (M4 decision 15) and
    the digest of those bytes, so the weights are not hashed again -- and the
    digest of the report bytes it published. The report is read back, and
    registered only if it still holds those bytes and names that checkpoint; a
    file replaced after publication is refused, not filed in this run's name.

    A refusal returns `EXIT_NOT_REGISTERED` with the reason. When the report is
    still the one published, it is kept and the command that registers it later
    is printed -- unless runs under its key disagree (`RunsDisagree`, D5), when
    no later registration under that key would be accepted either; when the
    file no longer holds it -- replaced, changed, damaged, removed, unreadable
    or no longer in the store (`ReportNotAsPublished`) -- the run's own report
    may be gone, and the operator is told to re-measure. A registration that
    stands although the version's usage history could not be appended says so
    on stderr.
    """
    from src.evaluation.registration import (
        MeasuredRun,
        RegistrationRefused,
        ReportNotAsPublished,
        RunsDisagree,
        corroboration_note,
        register_report,
    )

    measured = Path(inputs.checkpoint.identity.path)
    try:
        registered = register_report(
            report_path,
            measured=MeasuredRun(inputs.checkpoint.identity, report_sha256),
        )
    except ReportNotAsPublished as exc:
        print(
            f"\nNOT REGISTERED. {exc}\n"
            "The measurement itself succeeded; nothing at that path is offered for a "
            "later registration in this run's name.",
            file=sys.stderr,
        )
        return EXIT_NOT_REGISTERED, _registration_outcome(
            True, registered=False, outcome="refused", why=str(exc))
    except RunsDisagree as exc:
        print(
            f"\nNOT REGISTERED. {exc}\n"
            f"The measurement succeeded and its report is kept at {report_path}, where the "
            "investigation can read it beside the others.",
            file=sys.stderr,
        )
        return EXIT_NOT_REGISTERED, _registration_outcome(
            True, registered=False, outcome="refused", why=str(exc))
    except RegistrationRefused as exc:
        print(
            f"\nNOT REGISTERED. {exc}\n"
            f"The measurement succeeded and its report is kept at {report_path}.\n"
            "Register it later with:\n"
            f"  {register_later_command(report_path, measured)}",
            file=sys.stderr,
        )
        return EXIT_NOT_REGISTERED, _registration_outcome(
            True, registered=False, outcome="refused", why=str(exc))
    sequence = registered.record["registration"]["sequence"]
    said = {
        "registered": f"Registered as record {sequence}",
        "already_registered": f"Already held as record {sequence}; nothing written",
        "corroborated": (f"Agrees with record {sequence}; this report's digest was added to "
                         "its corroborating reports"),
    }[registered.outcome]
    print(f"\n{said} in {registered.ledger_path}")
    note = corroboration_note(registered, report_path)
    if note is not None:
        print(note)
    why = said
    if registered.usage_history_problem is not None:
        print(f"NOTE: {registered.usage_history_problem}.", file=sys.stderr)
        why = f"{said}; {registered.usage_history_problem}"
    return 0, _registration_outcome(
        True, registered=True, outcome=registered.outcome, sequence=sequence,
        ledger=str(registered.ledger_path), why=why)


def _offer_later_registration(
    inputs: MeasurementInputs, result: ModeResult, report_path: Path
) -> Dict[str, Any]:
    """Without `--record`: say the report was not registered and how to
    register it later -- unless D5 would refuse that now. Returns what the
    `--outcome` file says about the registration.

    The store is checked under this report's key (`key_refusal`, the check
    `register_report` makes), so the message stays true: **while runs under
    the key disagree no register-later command is printed**, as under
    `--record`, since none would be accepted; while that cannot be checked,
    the command is printed saying it is refused until it can be. The key is
    read from the report this run published (`report_key`, which is the
    record's key for a report `build_record` accepts).
    """
    from src.evaluation.registration import RunsDisagree, key_refusal
    from src.evaluation.sidecar import report_key

    command = register_later_command(report_path, Path(inputs.checkpoint.identity.path))
    key = report_key(result.to_dict())
    block = key_refusal(inputs.version, key) if key is not None else None
    if isinstance(block, RunsDisagree):
        print(f"\nNot registered (no --record), and {block}\n"
              f"The report is kept at {report_path}, where the investigation can read it "
              "beside the others. No register-later command is offered: none would be "
              "accepted while the runs disagree.\n")
        return _registration_outcome(False, why=(
            f"--record was not given, and {block} The report is kept for the "
            "investigation; no registration under its key is accepted while the runs "
            "disagree"))
    if block is not None:
        print(f"\nNot registered (no --record), and {block}\n"
              "A later registration is refused until that can be checked; then register it "
              f"with:\n  {command}\n")
        return _registration_outcome(False, why=(
            f"--record was not given; the report is kept for a later registration, which is "
            f"refused until this can be checked: {block}"))
    print(f"\nNot registered (no --record). Register it with:\n  {command}\n")
    return _registration_outcome(False)


def _check_outcome_target(path: Optional[Path]) -> None:
    """Refuse an `--outcome` file that cannot be written, before anything is
    read: its directory must exist (none is created), and it must not be one.

    **Nor may it exist** (a link, even a dangling one, included). A run refused
    before its report writes no outcome file, so a file already there -- an
    earlier run's, with ``"complete": true`` -- would be read as this run's
    outcome; and a path named by mistake would replace a file that is not an
    outcome at all. The run's own second write still replaces its first.
    """
    if path is None:
        return
    if path.is_dir():
        raise ValueError(f"--outcome {path} is a directory; it names the file the run writes "
                         "its outcome to. Nothing was read or written.")
    if not path.parent.is_dir():
        raise ValueError(f"--outcome {path}: its directory {path.parent} does not exist, and "
                         "none is created. Nothing was read or written.")
    if os.path.lexists(path):
        raise ValueError(
            f"--outcome {path} already exists. The run writes a new file there and never "
            "replaces one, so a file an earlier run left is not read as this run's outcome "
            "and a file named by mistake is not overwritten. Name a new file, or move that "
            "one away. Nothing was read or written."
        )


def _write_outcome(
    path: Path,
    *,
    complete: bool,
    inputs: MeasurementInputs,
    run_id: str,
    report_path: Path,
    report_sha256: str,
    deterministic: bool,
    registration: Mapping[str, Any],
) -> None:
    """The `--outcome` file: what this run did, once its report is published
    (``complete`` false) and again at the end (``complete`` true), each time
    staged and renamed into place, so a reader sees one whole version.

    It names the report the run published (path, run id, digest), the checkpoint
    it read (resolved path and digest), the version, whether the run was
    deterministic, and whether the report was registered and why. **What the
    run did, not the record:** whether the report is still kept and registered
    is read from the store and the ledger (`registration.report_state`), and
    ``complete`` false means the run stopped after publishing, so its later
    steps are unknown here. No path in it is an identity; the digests are.
    """
    from src.evaluation.cohort_registry import encode_json
    from src.utils.file_modes import publish_bytes

    block = inputs.cohort_version
    publish_bytes(path, encode_json({
        "what_this_is": ("the outcome of one measure_scorer run, written by the run; the "
                         "report store and the ledger beside the checkpoint are the record"),
        "schema_version": OUTCOME_SCHEMA_VERSION,
        "complete": complete,
        "run_id": run_id,
        "report": str(report_path),
        "report_sha256": report_sha256,
        "checkpoint": str(inputs.checkpoint.identity.path),
        "checkpoint_sha256": inputs.checkpoint.identity.sha256,
        "cohort": {"label": block["label"], "role": block["role"],
                   "manifest_sha256": block["manifest_sha256"]},
        "deterministic": deterministic,
        "registration": dict(registration),
    }))


def _record_run(inputs: MeasurementInputs, result: ModeResult, run_id: str,
                report_sha256: str, record_requested: bool) -> None:
    """This run in its version's usage history (D4), right after its report is
    published: which weights, which report, its result and whether it was
    deterministic. A failure is re-raised saying this run is not in it."""
    from src.evaluation.report_store import record_test_run

    manifest = result.manifest
    try:
        record_test_run(
            inputs.version, run_id=run_id, report_sha256=report_sha256,
            checkpoint_sha256=inputs.checkpoint.identity.sha256,
            metrics=result.authoritative_metrics, n_ranked=result.n_ranked,
            deterministic=bool(manifest.deterministic_algorithms), device=manifest.device,
            record_requested=record_requested,
        )
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"the cohort version's usage history could not be appended, so this run is not "
            f"in it ({type(exc).__name__}: {exc})"
        ) from exc


def _r10_label_text(label: str, value: Mapping[str, Any], evidence_state: str) -> str:
    """One R10 label's state in the caveats' words, with its count and
    denominator only where the state has one -- the words the Diagnosis panel
    shows for the record.

    **"unverifiable" and "no validation inputs recorded" are words, never a
    number,** and an unverifiable label gives its scope's reason
    (`R10_UNVERIFIABLE_REASON`): bytes not found, found and not readable, or a
    checkpoint that names no such file, which no `--training-evidence` can help.
    **"none" says it is not evidence about data outside the recorded inputs.**
    The raw state stays in the report, which holds no prose.
    """
    from src.evaluation.caveats import R10_STATE_TEXT, R10_UNVERIFIABLE_REASON

    state = value["state"]
    words = R10_STATE_TEXT[state]
    if state == "unverifiable":
        words = R10_UNVERIFIABLE_REASON.get(evidence_state, words)
    if state not in ("overlap", "none"):
        return words
    if label == "disease_overlap":
        return (f"{words} — {value['k']} of {value['n']} distinct test diseases; "
                f"{value['cases']} test cases with one of them")
    return f"{words} — {value['k']} of {value['n']} test cases"


def _print_registered_summary(inputs: MeasurementInputs, result: ModeResult, run_id: str,
                              report_path: Path) -> None:
    """Which version, its coverage with every denominator, and R10's labels.

    Each R10 label is shown in the caveats' words (`_r10_label_text`), then the
    four label definitions once, as the Diagnosis panel shows them: the operator
    reading the console is told what a phenotype-set overlap compares and what
    "none" does not establish, without either being written into the report.

    Aggregates and paths on the operator's machine only; no case id."""
    from src.evaluation.caveats import R10_LABEL_TEXT, R10_STATE_TEXT
    from src.evaluation.r10 import R10_LABELS, R10_SCOPES

    block = inputs.cohort_version
    counts = block["counts"]
    excluded = ", ".join(f"{name} {n}" for name, n in counts["excluded"].items())
    print(f"\nCohort version {block['label']} (role {block['role']}), manifest sha256 "
          f"{block['manifest_sha256'][:12]}...")
    print(f"  coverage: {result.n_ranked} ranked of {counts['source_cases']} source cases; "
          f"{counts['n_excluded']} excluded ({excluded})")
    print("\nR10, against the inputs this checkpoint records:")
    for scope, _ in R10_SCOPES:
        entry = inputs.r10[scope]
        evidence = entry["evidence"]
        where = f" at {evidence['path']}" if evidence["path"] else ""
        problem = f" ({evidence['problem']})" if evidence["problem"] else ""
        print(f"  {scope} evidence: {evidence['state']}{where}{problem}")
        for label in R10_LABELS:
            name = R10_LABEL_TEXT[scope][label]["name"]
            print(f"    {name:34s} {_r10_label_text(label, entry[label], evidence['state'])}")
    print("\nR10 labels:")
    for scope, _ in R10_SCOPES:
        for label in R10_LABELS:
            text = R10_LABEL_TEXT[scope][label]
            print(f"  {text['name']}: {text['definition']}.")
    print(f'  A "none" label: {R10_STATE_TEXT["none"]}.')
    print(f"\nReport: {report_path}\nRun id: {run_id}")


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    _check_outcome_target(args.outcome)
    # **One managed GPU job at a time** (M5): the deployment's GPU lock, taken
    # before the device is resolved or anything is read, and held until this
    # returns; the kernel releases it however the process ends. A busy lock is
    # `GpuLockBusy`, an OSError, so the program entry refuses with exit 1 and
    # "Nothing was written." The execution settings are set inside it, before
    # the device is resolved and CUDA initialised, and put back as they were
    # when the run ends, whatever ended it (`_execution_settings`).
    with take_gpu_lock("test" if args.cohort is not None else "split measurement",
                       "measure_scorer.py", log=logger):
        with _execution_settings(args.deterministic):
            return _run(args)


def _run(args: argparse.Namespace) -> int:
    """`main`'s run, under the GPU lock and inside its execution settings."""
    device, cuda_executed = _resolve_device(args.device)
    logger.info("Device: %s (%s)", device, platform.platform())
    if not cuda_executed:
        logger.warning(
            "Not running on CUDA (%s). The manifest will record cuda_executed=false; "
            "this is a development number, not one from the deployment hardware.", device
        )

    from src.evaluation.measurement import encode_full_graph, run_mode_c
    from src.models.gnn.shepherd_gnn import build_shepherd_model

    # Each input read once, and verified and recorded from those reads, before
    # any model is built or any output written (contract M2.1, S7; M4 §6).
    inputs = read_measurement_inputs(
        args.checkpoint, args.data_dir, device, split=args.split, cohort=args.cohort,
        registry=args.registry, training_evidence=args.training_evidence,
        expect_checkpoint_sha256=args.expect_checkpoint_sha256,
        expect_version_sha256=args.expect_version_sha256,
    )
    graph_data, samples = inputs.graph.graph_data, inputs.samples.samples
    digests = measurement_input_digests(inputs)

    try:
        model = build_shepherd_model(inputs.checkpoint.checkpoint, graph_data, device)
        # Kept **whole**: `run_mode_c` checks that its manifest describes the regime
        # these embeddings were computed under, which a bare dict could not carry.
        encoded = encode_full_graph(model, graph_data, device)

        result = run_mode_c(
            full_graph_embeddings=encoded,
            samples=samples,
            manifest=build_manifest(
                args, graph_data, len(samples), device, digests=digests,
                dataset_record=inputs.dataset_record.summary(),
                cohort_version=inputs.cohort_version, r10=inputs.r10,
                cuda_executed=cuda_executed, model=model, regime=encoded.regime,
            ),
            device=device,
            batch_size=args.batch_size,
        )
    except RuntimeError as exc:
        # **No fallback** (D5): an operation without a deterministic
        # implementation ends the run, before anything is written, saying so.
        if args.deterministic and _is_determinism_error(exc):
            raise DeterministicExecutionUnsupported(
                f"--deterministic was asked for, and this run needs an operation with no "
                f"deterministic implementation on {device} ({type(exc).__name__}: "
                f"{' '.join(str(exc).split())}). It does not fall back to non-deterministic "
                "execution or to another device, so it did not complete as requested. "
                "Nothing was written."
            ) from exc
        raise

    # **Publication, the usage history and the first outcome file are one step
    # against a cancel** (M5-3 review EVAL-1, `interrupt_deferred`): a SIGINT
    # (the job runner's Cancel) arriving among them is held until all three
    # are done, then ends the run as before, so a kept report is never left out
    # of its version's usage history by a cancel.
    with interrupt_deferred():
        # **Publication is the first write**, so a failure in it is no refusal:
        # the run measured, and what it published before failing may be on
        # disk. It is re-raised saying so (`ReportNotWritten`), never as a
        # ValueError or OSError, which the program entry prints as a refusal
        # before anything was written.
        try:
            if inputs.version is None:
                report_path, ranks_path = _publish_generated(result, args.output)
                logger.info("Mode C -> %s, %s", report_path, ranks_path)
            else:
                run_id, report_path, report_sha256 = _publish_to_store(result, inputs)
                logger.info("Mode C -> %s", report_path)
        except (ValueError, OSError) as exc:
            raise _publication_failure(exc, inputs, args.output) from exc

        # Nor is a failure after it (the console closed under the run, say): it
        # is re-raised saying the report was written (`FailedAfterPublishing`).
        try:
            if inputs.version is not None:
                # The run in its version's usage history (D4), the record, and
                # what this run did so far (`--outcome`), the job runner's view,
                # before anything else can fail. **Each is tried whatever the
                # other did**: a failed outcome write must not keep a published
                # run out of the history, nor a failed history append leave a
                # published run with no outcome file. The first failure is then
                # raised, saying both.
                deterministic = bool(result.manifest.deterministic_algorithms)
                history_failure: Optional[ValueError] = None
                try:
                    _record_run(inputs, result, run_id, report_sha256, args.record)
                except ValueError as exc:
                    history_failure = exc
                if args.outcome is not None:
                    try:
                        _write_outcome(args.outcome, complete=False, inputs=inputs,
                                       run_id=run_id, report_path=report_path,
                                       report_sha256=report_sha256,
                                       deterministic=deterministic,
                                       registration=_registration_outcome(args.record))
                    except (ValueError, OSError) as exc:
                        raise ValueError(
                            f"the --outcome file {args.outcome} could not be written "
                            f"({type(exc).__name__}: {exc}); "
                            + (f"and {history_failure}" if history_failure is not None
                               else "this run is in its version's usage history")
                        ) from exc
                if history_failure is not None:
                    raise history_failure
        except (ValueError, OSError) as exc:
            raise _failure_after_publishing(exc, inputs, report_path) from exc

    try:
        print(f"\nMode C — {result.n_ranked} ranked, {result.n_ground_truth_absent} absent")
        for name, value in result.authoritative_metrics.items():
            print(f"  {name:38s} {value:.6f}")
        print(f"  {'candidate diseases per patient':38s} "
              f"{result.sampler_evidence['candidate_columns']['max']}")

        # **The number says what its graph was built from** (contract M3b), here
        # as in the manifest: binding state and origin apart.
        record = inputs.dataset_record
        print(f"\nDataset record: binding {record.binding}, origin {record.origin}")
        if record.origin == "synthetic":
            print("SYNTHETIC WORKSPACE — this number is not over a graph built from real files.")
        if not cuda_executed:
            print("\nNOT ON CUDA — development run. Recorded as cuda_executed=false.")
        print("\nWhether this is the institution's number depends on the checkpoint and\n"
              "cohort it consumed, which a person checks against the recorded digests.")

        if inputs.version is None:
            print(f"\nReport: {report_path}\n")
            status = 0
        else:
            _print_registered_summary(inputs, result, run_id, report_path)
            if args.record:
                # **Flushed before the ledger is touched**, so a console that
                # has gone is found here whatever stdout's buffering, as an
                # unbuffered one is at its first print: the run registers
                # nothing, and the failure's line offers the registration.
                _flush_stdout()
                status, registration = _register(report_path, report_sha256, inputs)
            else:
                status, registration = 0, _offer_later_registration(inputs, result,
                                                                     report_path)
            if args.outcome is not None:
                _write_outcome(args.outcome, complete=True, inputs=inputs, run_id=run_id,
                               report_path=report_path, report_sha256=report_sha256,
                               deterministic=deterministic, registration=registration)
            if status != 0:
                # Decided, and said on stderr: a refusal writes nothing to
                # stdout, and `_run_as_program` settles it, so this status
                # stands rather than becoming a failure after the fact.
                return status
        # **Flushed here**: on a pipe or a file stdout is buffered, and a closed
        # console would otherwise fail only when the interpreter flushes at exit,
        # outside this handler.
        _flush_stdout()
        return status
    except (ValueError, OSError) as exc:
        raise _failure_after_publishing(exc, inputs, report_path) from exc


def _flush_stdout() -> None:
    """Flush stdout, if there is one: started with its descriptor closed, Python
    sets `sys.stdout` to None, and `print` then writes nothing."""
    if sys.stdout is not None:
        sys.stdout.flush()


def _settle_stdout() -> None:
    """Flush stdout, or point it at the null device when it cannot be written.

    Python flushes stdout once more at exit, outside every handler: output
    still buffered for a closed console fails there, prints "Exception ignored
    ... BrokenPipeError", and exit 120 replaces the status returned. The
    redirect is the recipe in Python's note on SIGPIPE. **Only a stdout that
    failed is redirected**: one that still works (a registration raised, stdout
    on a file) is flushed instead, so the summary buffered in it is kept, ahead
    of the failure's line. A stdout with no file descriptor (an in-process
    caller's capture), or none at all, is left as it is.
    """
    if sys.stdout is None:
        return
    try:
        sys.stdout.flush()
    except (OSError, ValueError):
        try:
            fd = sys.stdout.fileno()
        except (OSError, ValueError):  # io.UnsupportedOperation is both
            return
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, fd)
        os.close(devnull)


def _run_as_program(argv: Optional[List[str]] = None) -> int:
    """`main` as the operator runs it: a refusal is one message on stderr and
    exit 1, not a traceback (BACKLOG 26).

    **Only here.** `main` keeps raising its typed refusals, because the tests
    and the deployment probe call it in process and read them and their fields
    (`TrainingRecordMismatch`, `DatasetRecordRefused.binding`,
    `CohortBindingMismatch.roles`); a `SystemExit` raised inside `main` would
    also escape the probe's `refuses`, which catches `Exception`.

    **Exactly `ValueError` and `OSError`**, as `import_cohort.py` and
    `audit_split_overlap.py` catch them: the refusals `main` raises before
    anything is written, every check's and a file that cannot be read.
    "Nothing was written." is added unless the message already says so. A
    ValueError that is really a bug becomes one line too, without its
    traceback; nothing had been written when it was raised. A failure in or
    after publication is neither (`ReportNotWritten`, `FailedAfterPublishing`)
    and is printed in its own words, also with exit 1. A cancel
    (`KeyboardInterrupt`, from SIGINT) is `CANCELLED_LINE`, printed after
    stdout is settled, and then **the KeyboardInterrupt goes on to the
    caller**: `_exit_as_program` ends the process by SIGINT, and an in-process
    caller (the tests) unwinds as it would from any interrupt, never killed
    here. Stdout is settled first because that ending skips the
    interpreter's last flush, which would lose what a pipe's buffer still
    holds. Anything else keeps its
    traceback -- an out-of-memory error, a regime mismatch -- and `SystemExit`
    (exit 2 for the command line, 1 for the device check) and the status `main`
    returns (3 for a refused registration) pass through as they are; stdout is
    settled first, so output a closed console could not take cannot turn that
    status into 120 at the interpreter's last flush.
    """
    if signal.getsignal(signal.SIGINT) is _cancelled_while_loading:
        # Loaded: a cancel is a KeyboardInterrupt again, so the run unwinds
        # (a staged file removed, the GPU lock released) before it ends.
        signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        status = main(argv)
    except KeyboardInterrupt:
        _settle_stdout()
        print(CANCELLED_LINE, file=sys.stderr)
        raise
    except ReportNotWritten as exc:
        print(f"measure_scorer: measured, but the report was not written: {exc}",
              file=sys.stderr)
        return 1
    except FailedAfterPublishing as exc:
        _settle_stdout()
        print(f"measure_scorer: measured, but the run did not finish: {exc}", file=sys.stderr)
        return 1
    except (ValueError, OSError) as exc:
        message = _sentence(exc)
        if not _SAYS_NOTHING_WRITTEN.search(message):
            message += " Nothing was written."
        print(f"measure_scorer: refused: {message}", file=sys.stderr)
        return 1
    _settle_stdout()
    return status


def _exit_as_program(argv: Optional[List[str]] = None) -> None:
    """The program's entry (`__main__`), and the only place a cancel ends the
    process: `_run_as_program`'s status is the exit status, and its
    KeyboardInterrupt (the cancel, whose line it has printed) ends the process
    by SIGINT (`_end_as_cancelled`), so the job runner reads "ended by SIGINT"
    as the cancel's doing. Raises `SystemExit`."""
    try:
        status = _run_as_program(argv)
    except KeyboardInterrupt:
        status = _end_as_cancelled()
    sys.exit(status)


if __name__ == "__main__":
    _exit_as_program()
