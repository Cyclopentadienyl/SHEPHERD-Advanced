"""
Routes kept for a feature that is not built: a plain refusal, never made-up data
================================================================================
B56 (the WebUI ledger, T06): the HPO term search and disease detail routes
answered fixed mock tables (``Seizure``, ``Marfan syndrome``, ``SCN1A``) as if
they were the ontology's and the knowledge graph's, while nothing in the
dashboard, the scripts or the tests called them. A client could not tell those
answers from real ones. Each such route now answers **501 Not Implemented**
with one sentence saying what is not built and what to use instead
(`refuse_not_built`); its path stays, so a client that calls it learns why
rather than meeting a 404. Building the feature means reading the served
ontology or graph, not these refusals.

Module: src/api/routes/not_built.py
"""
from __future__ import annotations

from typing import Any, NoReturn

from fastapi import HTTPException, status

#: How each such route's 501 is described in the OpenAPI schema.
RESPONSES: dict[int | str, dict[str, Any]] = {
    status.HTTP_501_NOT_IMPLEMENTED: {
        "description": "Not built: nothing is answered, and never made-up data in its place.",
    },
}


def refuse_not_built(feature: str, instead: str) -> NoReturn:
    """Refuse a request for `feature` with 501: it is not built, no data is
    made up in its place, and `instead` says what to use."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=f"{feature} is not built: this service answers no made-up data in its place. "
               f"{instead}")


__all__ = ["RESPONSES", "refuse_not_built"]
