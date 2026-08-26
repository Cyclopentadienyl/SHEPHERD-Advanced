"""
Provenance vocabulary — what a person may assert about where evidence came from.
================================================================================
The evidence scripts (``scripts/audit_*.py``) each write a JSON report carrying
one operator assertion: how the machine the report was produced on relates to the
deployment. All three are read together, so all three must spell that assertion
the same way — hence one vocabulary here rather than three copies.

**It is an assertion, not an identity and not a key.** Nothing here identifies a
machine, and nothing joins reports by one. There is no hostname, no serial, no
run id, no registry; the value says only what kind of relationship a person is
claiming, and no code verifies it.

**Why it is worth recording at all.** Not determinism — these audits read files
and count integers, so their numbers do not depend on which GPU is present. What
they depend on is *which filesystem was in front of them*: M1-M3 reports the
checkpoints a directory holds, M4 the splits a workspace holds, M5 the artifact a
workspace holds. Whether that workspace is the deployment's is the one thing a
reader cannot recover from the file and cannot check, so a person states it.

Complement to ``src/utils/fingerprint.py``: that module answers *which inputs*
were read, from bytes it can hash. This module carries the claim about *where*
they were read, which no hash can establish.

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
