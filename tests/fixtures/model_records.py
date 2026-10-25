"""
Recorded test results beside listed checkpoints, for the Models page.
=====================================================================
Research › Models (`src/webui/components/models_page.py`) reads what a listed
checkpoint's ledger holds and what the cohort version's report store says of
each record's key. This puts both where the production code writes them, with
the production functions:

- a report is built as the measurement writes one
  (`tests/unit/test_evaluation_sidecar.registered_report`) for a cohort version
  block of a given label and role, naming the checkpoint's digest, and kept in
  the registry's report store (`tests/fixtures/report_store.store_report`,
  which writes the version too);
- `register` then builds its record from the stored bytes (`build_record`) and
  appends it to the ledger beside the checkpoint (`append_record`,
  `write_ledger`): the ledger the registration function writes, without its
  checkpoint read (these checkpoints may be stand-ins that cannot be scored);
- `disagree` keeps a second report under the same key with other metrics and
  registers nothing: the key is then contested in the store (D5).

The registry must be outside the repository (a pytest tmp directory is).

Module: tests/fixtures/model_records.py
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

#: The roles the page tells apart (§4.9 question 1).
RESEARCH_ROLE = "research_selection"
ACCEPTANCE_ROLE = "institutional_acceptance"


def version_block(label: str, role: str, **overrides: Any) -> Dict[str, Any]:
    """A cohort version's report block for this label and role (aggregates only)."""
    from tests.unit.test_evaluation_sidecar import cohort_version

    source = hashlib.sha256(f"{label}/{role}".encode()).hexdigest()
    return cohort_version(label=label, role=role, source_sha256=source, **overrides)


def report_for(checkpoint_sha256: str, block: Dict[str, Any], *,
               metrics: Optional[Dict[str, float]] = None, deterministic: bool = True,
               device: str = "cuda", **overrides: Any) -> Dict[str, Any]:
    """A measurement report of this checkpoint on the version ``block`` names."""
    from tests.unit.test_evaluation_sidecar import registered_report

    report = registered_report(
        metrics=metrics, split=block["role"], cohort_version=copy.deepcopy(block),
        deterministic_algorithms=deterministic, cudnn_deterministic=deterministic,
        device=device, cuda_executed=device.startswith("cuda"),
        cuda_version=("13.0" if device.startswith("cuda") else None), **overrides)
    report["manifest"]["artifact_digests"]["checkpoint"] = checkpoint_sha256
    return report


def keep(registry: Path, report: Dict[str, Any]) -> Tuple[Path, Dict[str, Any]]:
    """Keep a report in the store; returns its path and the record built from it."""
    from src.evaluation.sidecar import build_record
    from tests.fixtures.report_store import store_report

    path = store_report(registry, report)
    data = path.read_bytes()
    return path, build_record(json.loads(data), hashlib.sha256(data).hexdigest())


def register(registry: Path, checkpoint: Path, report: Dict[str, Any], *,
             registered_at: str = "2026-10-17T15:31:00Z") -> Tuple[Path, Dict[str, Any]]:
    """Keep the report and append its record to the ledger beside the
    checkpoint; returns the report's path and the record as the ledger holds it."""
    from src.evaluation.sidecar import (
        LEDGER_FILENAME,
        append_record,
        empty_ledger,
        read_ledger,
        write_ledger,
    )

    path, record = keep(registry, report)
    ledger_path = Path(checkpoint).resolve().parent / LEDGER_FILENAME
    ledger, digest = read_ledger(ledger_path)
    outcome, updated = append_record(ledger if digest else empty_ledger(), record,
                                     registered_at=registered_at)
    assert outcome == "registered", outcome
    write_ledger(ledger_path, updated, digest)
    held = [r for r in updated["records"]
            if r["source_artifact_digest"] == record["source_artifact_digest"]]
    return path, held[0]


def disagree(registry: Path, report: Dict[str, Any],
             metrics: Dict[str, float]) -> Path:
    """Keep a second report under the same key with other metrics: runs disagree."""
    other = copy.deepcopy(report)
    other["authoritative_metrics"] = dict(metrics)
    return keep(registry, other)[0]


__all__ = ["ACCEPTANCE_ROLE", "RESEARCH_ROLE", "disagree", "keep", "register", "report_for",
           "version_block"]
