"""
D5's contested state, for every surface that shows it: of ledger records and of report runs.
============================================================================================
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

**A run's state** (`run_contest`) is the same judgement for one run directory
of a store: the one run listing that ``GET /api/v1/cohorts`` and the Test page
share (`server_lists._report_items`) attaches it. **One layer turns a scan into
a shown state**, for records and runs alike: a scan that fails is caught and
worded once (`read_report_store`), and the problems' words, which may name a
server path, are left out of an open surface's answer in one place (`_shown`).

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
from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: A digest as the server's lists give it: 64 lowercase hexadecimal digits.
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def key_text(key: Sequence[Any]) -> str:
    """A ledger key (`record_key`'s tuple) as one comparable text."""
    return json.dumps(list(key))


def _shown(problems: Iterable[str], operator: bool) -> List[str]:
    """The problems' words on an operator's surface; none on an open one, since
    they may name a server path."""
    return list(problems) if operator else []


class StoreRead(NamedTuple):
    """A version's report store as one scan read it (`scan_report_store`'s
    `StoreScan`), or why it could not be scanned (`problem`)."""

    scan: Optional[Any]
    problem: Optional[str]


def read_report_store(version: Any) -> StoreRead:
    """One scan of `version`'s report store (`scan_report_store`): each kept
    report read once and built into the record the ledger would hold. **The one
    place a scan's failure is caught and worded**, for records and runs alike:
    its reason, never an exception, so every record and run is still shown, its
    key's state "cannot be checked"."""
    from src.evaluation.report_store import scan_report_store

    try:
        return StoreRead(scan_report_store(version), None)
    except Exception as exc:  # noqa: BLE001 - the keys' states then read "cannot be checked"
        logger.exception("A report store could not be scanned")
        return StoreRead(None, f"its report store could not be read ({type(exc).__name__})")


#: Why a kept report has no key state of its own: the store's scan read no
#: report in its run directory (it was written after the scan, before its state
#: was read).
_NOT_COMPARED = "the report store's comparison did not read it"


def run_contest(store: StoreRead, run_id: str, state: Mapping[str, Any], *,
                operator: bool) -> Optional[Dict[str, Any]]:
    """D5's state of the key run `run_id`'s report is under, from the store's
    one scan (`StoreScan.run_status`, which finds the run by its id), **for the
    bytes `state` read** (its ``report_sha256``, the registration's one helper's
    read): a report replaced between the scan's read and that one is
    ``cannot_be_checked``, so a run's kept/registered state and its key's state
    are always of one byte string. A report the scan read but could not place is
    ``cannot_be_checked``, and so is one it did not read, or every run of a
    store that could not be scanned: never no state while a report may be kept.
    None for a run that kept none. `operator` False leaves out the problems'
    words."""
    status = (store.scan.run_status(run_id, state.get("report_sha256"))
              if store.scan is not None else None)
    if status is None:
        if state.get("kept") == "not_kept":
            return None
        return {"state": "cannot_be_checked", "differing": [], "disagreeing_runs": [],
                "problems": _shown([store.problem or _NOT_COMPARED], operator)}
    return {"state": status.state, "differing": list(status.differing),
            "disagreeing_runs": list(status.disagreeing_runs),
            "problems": _shown(status.problems, operator)}


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
        #: Why no version can be found (`registry` None), as `by_digest` says it.
        self.problem = None if setting.available else (
            f"no cohort registry: {setting.problem or 'none is configured'}")
        self._resolved: Dict[str, Any] = {}

    @classmethod
    def listed(cls, environ: Optional[Mapping[str, str]] = None) -> "CohortVersions":
        """The versions the server's registry holds now, each ``cohort.json``
        read once and nothing else (`list_versions`: no report store is listed
        here), kept as read; none when they cannot be listed, and why (every
        record then "cannot be checked", for that reason)."""
        from src.evaluation.cohort_registry import RegistryRefused, list_versions

        found = cls((), environ)
        if found.registry is None:
            return found
        try:
            entries = list_versions(found.registry)
        except RegistryRefused as exc:  # the listing's own refusal: nothing is listed
            found.registry, found.problem = None, (
                f"the registry's cohort versions cannot be listed: {exc}")
            return found
        except Exception as exc:  # noqa: BLE001 - contested then reads "cannot be checked"
            logger.exception("The cohort versions could not be listed")
            found.registry, found.problem = None, (
                f"the registry's cohort versions cannot be listed ({type(exc).__name__})")
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
            return None, self.problem
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
    the key's text: one scan of each store (`read_report_store`), then
    `StoreScan.status`. A key whose state cannot be read is ``cannot_be_checked``,
    never uncontested. For a contested key, the disagreeing runs' values of the
    metrics that differ, as their reports hold them (a hand-edited report may
    hold null, NaN or text there; the state stays contested). `operator` False
    leaves out the problems' own words, which may name a server path."""
    from src.evaluation.sidecar import record_key

    keys: Dict[str, Tuple[Any, ...]] = {}
    for record in records:
        try:
            key = record_key(dict(record))
        except ValueError:
            continue
        keys[key_text(key)] = key
    scans: Dict[str, StoreRead] = {}
    out: Dict[str, Dict[str, Any]] = {}
    for text, key in keys.items():
        digest = key[2]
        if digest not in scans:
            version, why = versions.by_digest(digest)
            scans[digest] = (StoreRead(None, why) if version is None
                             else read_report_store(version))
        scan, why = scans[digest]
        if scan is None:
            out[text] = {"state": "cannot_be_checked", "differing": [], "runs": [],
                         "problems": _shown([why], operator)}
            continue
        try:
            status = scan.status(key)
        except Exception as exc:  # noqa: BLE001
            out[text] = {"state": "cannot_be_checked", "differing": [], "runs": [],
                         "problems": _shown([f"({type(exc).__name__})"], operator)}
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
                     "problems": _shown(status.problems, operator)}
    return out


__all__ = ["CohortVersions", "StoreRead", "contest_states", "key_text", "read_report_store",
           "run_contest"]
