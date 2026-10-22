"""
D5's contested state of ledger records, for every surface that shows them.
=========================================================================
`docs/working/PLAN_TEST_RESULTS.md` D5 and D4: a record registered before its
key's runs came to disagree stays, **marked contested wherever it is shown**;
one whose state cannot be read is "cannot be checked", never uncontested.

**One implementation** (BACKLOG 36, the M5 review's P2-6), moved here from the
dashboard's `src/webui/components/model_list.py` so that the API never imports
a UI module: the read-only route ``GET /api/v1/evaluations/loaded-model``
(`src/api/routes/evaluations.py`) and the dashboard's model list, Models page
and Diagnosis card all take a record's state from `contest_states`, so an API
client and the dashboard cannot disagree about a record. The read-only Models
view shows the route's answer and computes nothing itself.

Read at each call, with no cache: the registry's version manifests, each
``cohort.json`` once (`CohortVersions.listed`, to find a record's version by the
cohort digest its key carries), and one scan of each named version's report
store (`scan_report_store`).

Module: src/api/services/record_states.py
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: A digest as the server's lists give it: 64 lowercase hexadecimal digits.
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def key_text(key: Sequence[Any]) -> str:
    """A ledger key (`record_key`'s tuple) as one comparable text."""
    return json.dumps(list(key))


class CohortVersions:
    """The registry's versions by the digest of their ``cohort.json``: the one
    place a record's cohort version is found (the route, the rows, Models and
    the Diagnosis card).

    Two ways in. `listed` reads the registry's versions now, each manifest once
    (`cohort_registry.list_versions`), and keeps each read: the digest a record
    is matched by and the version returned for it come from that one read.
    The constructor takes versions a server's list gave earlier (label and
    digest each, Models' page data); each is read again on first use and
    checked against the digest it was given."""

    def __init__(self, versions: Sequence[Mapping[str, Any]],
                 environ: Optional[Mapping[str, str]]) -> None:
        from src.config import server_settings

        self.listed = {v["sha256"]: v for v in versions if _is_digest(v.get("sha256"))}
        setting = server_settings.cohort_registry(environ)
        self.registry = setting.value if setting.available else None
        self.problem = None if setting.available else (setting.problem or "no cohort registry")
        self._resolved: Dict[str, Any] = {}

    @classmethod
    def listed(cls, environ: Optional[Mapping[str, str]] = None) -> "CohortVersions":
        """The versions the server's registry holds now, each ``cohort.json``
        read once and nothing else (`list_versions`: no report store is listed
        here), kept as read; none when they cannot be listed (every record then
        "cannot be checked")."""
        from src.evaluation.cohort_registry import RegistryRefused, list_versions

        found = cls((), environ)
        if found.registry is None:
            return found
        try:
            entries = list_versions(found.registry)
        except RegistryRefused:  # the listing's own refusal: nothing is listed
            return found
        except Exception:  # noqa: BLE001 - contested then reads "cannot be checked"
            logger.exception("The cohort versions could not be listed")
            return found
        for entry in entries:
            if entry.version is None:
                continue
            digest = entry.version.identity.sha256
            found.listed[digest] = {"label": entry.label, "sha256": digest}
            found._resolved[entry.label] = (entry.version, None)
        return found

    def by_digest(self, digest: Any) -> Tuple[Any, Optional[str]]:
        """``(version, None)`` or ``(None, why)`` for a cohort version digest."""
        from src.evaluation.cohort_registry import resolve_version

        if self.registry is None:
            return None, f"no cohort registry: {self.problem}"
        listed = self.listed.get(digest)
        if listed is None:
            shown = digest[:12] if _is_digest(digest) else "—"
            return None, (f"its cohort version ({shown}) is not one the server's registry "
                          "lists")
        label = listed["label"]
        # Read by `listed` already, or once here for a version a list gave earlier.
        if label not in self._resolved:
            try:
                version = resolve_version(self.registry, label)
            except Exception as exc:  # noqa: BLE001 - a state, never a crash of the page
                self._resolved[label] = (None, f"{label} cannot be read ({type(exc).__name__})")
            else:
                if version.identity.sha256 != digest:
                    self._resolved[label] = (None, f"{label} changed since it was listed")
                else:
                    self._resolved[label] = (version, None)
        return self._resolved[label]


def contest_states(records: Iterable[Mapping[str, Any]], versions: CohortVersions, *,
                   operator: bool = True) -> Dict[str, Dict[str, Any]]:
    """Each record key's state in its cohort version's report store (D5), by
    the key's text: one scan of each store (`scan_report_store`), then
    `StoreScan.status`. A key whose state cannot be read is ``cannot_be_checked``,
    never uncontested. For a contested key, the disagreeing runs' values of the
    metrics that differ, as their reports hold them (a hand-edited report may
    hold null, NaN or text there; the state stays contested). `operator` False
    leaves out the problems' own words, which may name a server path."""
    from src.evaluation.report_store import scan_report_store
    from src.evaluation.sidecar import record_key

    keys: Dict[str, Tuple[Any, ...]] = {}
    for record in records:
        try:
            key = record_key(dict(record))
        except ValueError:
            continue
        keys[key_text(key)] = key
    scans: Dict[str, Tuple[Any, Optional[str]]] = {}
    out: Dict[str, Dict[str, Any]] = {}
    for text, key in keys.items():
        digest = key[2]
        if digest not in scans:
            version, why = versions.by_digest(digest)
            if version is None:
                scans[digest] = (None, why)
            else:
                try:
                    scans[digest] = (scan_report_store(version), None)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("A report store could not be scanned")
                    scans[digest] = (None, f"its report store could not be read "
                                           f"({type(exc).__name__})")
        scan, why = scans[digest]
        if scan is None:
            out[text] = {"state": "cannot_be_checked", "differing": [], "runs": [],
                         "problems": [why] if operator else []}
            continue
        try:
            status = scan.status(key)
        except Exception as exc:  # noqa: BLE001
            out[text] = {"state": "cannot_be_checked", "differing": [], "runs": [],
                         "problems": [f"({type(exc).__name__})"] if operator else []}
            continue
        runs = []
        if status.state == "contested":
            for run_id, record in (scan.reports.get(tuple(key)) or {}).values():
                if run_id in status.disagreeing_runs:
                    metrics = record.get("metrics") if isinstance(record, Mapping) else None
                    runs.append({"run": run_id, "values": {
                        name: metrics.get(name) for name in status.differing
                        if isinstance(metrics, Mapping) and name in metrics}})
        out[text] = {"state": status.state, "differing": list(status.differing),
                     "runs": sorted(runs, key=lambda run: run["run"]),
                     "problems": list(status.problems) if operator else []}
    return out


__all__ = ["CohortVersions", "contest_states", "key_text"]
