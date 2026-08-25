"""
Provenance vocabulary — what a person may assert about where evidence came from.
================================================================================
The evidence scripts (``scripts/audit_*.py``) each write a JSON report that an
institutional reader joins with the others: one machine, one run, three reports.
That join only works if all three spell the machine's relationship to the
deployment the same way, so the vocabulary lives here rather than being restated
in each script.

Complement to ``src/utils/fingerprint.py``: that module answers *which inputs*
produced an artifact, from bytes it can read. This module carries the part no
code can read — whether the machine that produced the evidence is the one the
system will run on.

No torch / PyG imports, and no I/O — safe to import from any script or service.

Module: src/utils/provenance.py
"""
from __future__ import annotations

#: What an operator may assert about the machine an evidence file came from.
#:
#: **A bounded vocabulary, not free text.** BACKLOG §5.2 forbids operator and host
#: names, and a schema that forbids them cannot then accept an arbitrary string:
#: the first person in a hurry writes the hostname into it. These values carry the
#: claim an institutional reader needs — is this the deployment's machine, its
#: twin, or something else — and carry nothing else.
#:
#: Unverified by design. Whether a machine really is the deployment's sibling is
#: not a fact this code can establish, and the same split
#: `MeasurementManifest.cuda_executed` uses applies: record the narrow checkable
#: things, and let the broad claim be made by a person who can make it.
DEPLOYMENT_RELATIONSHIPS = (
    "identical-sibling",          # same hardware and software build; different serial
    "same-model-different-unit",  # same model, not a controlled twin
    "same-model-different-oem",   # same silicon, different vendor build
    "different-hardware",
    "unstated",
)

#: The value written when no one has made a claim. Named rather than repeated as a
#: literal default in three argument parsers, so "the reports agree on what silence
#: looks like" is a fact of this module and not a coincidence of three call sites.
UNSTATED_RELATIONSHIP = "unstated"

__all__ = ["DEPLOYMENT_RELATIONSHIPS", "UNSTATED_RELATIONSHIP"]
