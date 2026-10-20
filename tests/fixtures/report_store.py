"""
A report store for reports built in memory: a version and its run directories.
==============================================================================
Since M5-1 a report is registered only from the registry's report store
(`PLAN_TEST_RESULTS.md` D5, 2026-10-19): `register_report` finds the version
through ``<registry>/versions/<label>/reports/<run id>/report.json``, resolves it
and requires the report to name it exactly. The unit tests build their reports
with `MeasurementManifest` (`tests/unit/test_evaluation_sidecar.registered_report`)
rather than by measuring; this puts such a report where the measurement would:

- `version_manifest_for` writes the version manifest a report's
  ``cohort_version`` block was derived from: every field the block copies, plus
  what `resolve_version` checks (counts that add up, one exclusion entry per
  excluded case with its category and no case id, an ``imported`` block);
- `store_report` keeps a report in a new run directory of that version (named
  as `create_report_dir` names one, for the report's checkpoint), with the
  block's ``manifest_sha256`` set to the digest of the version's
  ``cohort.json`` -- the only field the block does not take from the version;
- `store_bytes` keeps arbitrary bytes there, for a damaged report.

The registry must be outside the repository (a pytest tmp directory is) and
exist; it is created here when absent. A version already in the registry is
reused when the manifest it would get is the same, and refused otherwise, so a
test cannot silently pair a report with a version it does not name; another
version of one label lives in another registry, as two imports of a label do
(``imported_at`` tells them apart).

Module: tests/fixtures/report_store.py
"""
from __future__ import annotations

import copy
import json
import secrets
from pathlib import Path
from typing import Any, Dict, Optional

#: When the fixture's versions say they were imported.
IMPORTED_AT = "2026-10-16T12:00:00Z"


def version_manifest_for(block: Dict[str, Any], *,
                         imported_at: str = IMPORTED_AT) -> Dict[str, Any]:
    """The version manifest whose `cohort_version_block` is ``block`` (except
    its ``manifest_sha256``, the digest of the bytes written)."""
    from src.evaluation.cohort_registry import VERSION_SCHEMA_VERSION, WHAT_THIS_IS

    counts = block["counts"]
    return {
        "schema_version": VERSION_SCHEMA_VERSION,
        "what_this_is": WHAT_THIS_IS,
        "label": block["label"],
        "role": block["role"],
        "source": {"sha256": block["source_sha256"], "format": block["source_format"],
                   "filename": "cases.json", "n_cases": counts["source_cases"]},
        "graph": copy.deepcopy(block["graph"]),
        "dataset_record": {"sha256": block["dataset_record_sha256"], "origin": "synthetic",
                           "hpo": None, "mondo": None},
        "mapping": copy.deepcopy(block["mapping"]),
        "counts": copy.deepcopy(counts),
        "samples": {"sha256": block["samples_sha256"], "n": counts["ranked_eligible"]},
        "unmappable_terms": {},
        "exclusions": [{"index": index, "category": category, "reasons": [category]}
                       for category, n in sorted(counts["excluded"].items())
                       for index in range(n)],
        "imported": {"at": imported_at, "code_revision": {"state": "not_recorded"}},
    }


def store_version(registry: Path, block: Dict[str, Any], *,
                  imported_at: str = IMPORTED_AT) -> Any:
    """The version ``block`` names, written into ``registry`` if absent, and
    resolved through `resolve_version` (its digest is ``identity.sha256``)."""
    from src.evaluation.cohort_registry import encode_json, resolve_version

    registry = Path(registry)
    registry.mkdir(parents=True, exist_ok=True)
    data = encode_json(version_manifest_for(block, imported_at=imported_at))
    directory = registry / "versions" / block["label"]
    path = directory / "cohort.json"
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"{path} holds another version of {block['label']!r}; use another "
                             "registry for another import of one label")
    else:
        directory.mkdir(parents=True)
        path.write_bytes(data)
    return resolve_version(registry, block["label"])


def new_run_id(checkpoint_sha256: str) -> str:
    """A run directory's name as `create_report_dir` makes one."""
    return f"20261016T120000Z-{checkpoint_sha256[:12]}-{secrets.token_hex(4)}"


def store_report(registry: Path, report: Dict[str, Any], *,
                 imported_at: str = IMPORTED_AT, run_id: Optional[str] = None,
                 indent: Optional[int] = 2) -> Path:
    """Keep ``report`` (a copy, its block's ``manifest_sha256`` set to its
    version's digest) in a new run directory of the version it names; returns
    the ``report.json`` path. A generated split's report (no block) goes under
    the default version of `tests/unit/test_evaluation_sidecar.cohort_version`."""
    from tests.unit.test_evaluation_sidecar import cohort_version

    report = copy.deepcopy(report)
    manifest = report["manifest"]
    block = manifest.get("cohort_version")
    version = store_version(registry, block if block is not None else cohort_version(),
                            imported_at=imported_at)
    if block is not None:
        block["manifest_sha256"] = version.identity.sha256
    checkpoint = manifest["artifact_digests"]["checkpoint"]
    data = json.dumps(report, indent=indent, sort_keys=True, allow_nan=False).encode("utf-8")
    return store_bytes(registry, data, version=version, checkpoint_sha256=checkpoint,
                       run_id=run_id)


def store_bytes(registry: Path, data: bytes, *, version: Any = None,
                checkpoint_sha256: str = "c" * 64, run_id: Optional[str] = None) -> Path:
    """Keep ``data`` as ``report.json`` in a new run directory of ``version``
    (the default version when `None`); returns its path."""
    from tests.unit.test_evaluation_sidecar import cohort_version

    if version is None:
        version = store_version(registry, cohort_version())
    run_dir = version.directory / "reports" / (run_id or new_run_id(checkpoint_sha256))
    run_dir.mkdir(parents=True)
    path = run_dir / "report.json"
    path.write_bytes(data)
    return path


def version_of(report_path: Path) -> Any:
    """The version a stored report is kept under, resolved again."""
    from src.evaluation.cohort_registry import resolve_version

    report_path = Path(report_path)
    return resolve_version(report_path.parents[4], report_path.parents[2].name)


__all__ = ["IMPORTED_AT", "new_run_id", "store_bytes", "store_report", "store_version",
           "version_manifest_for", "version_of"]
