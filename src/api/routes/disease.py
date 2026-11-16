"""
SHEPHERD-Advanced disease detail routes: not built
==================================================
GET /disease/{disease_id}, /disease/{disease_id}/phenotypes and
/disease/{disease_id}/genes were placeholders answering fixed mock diseases,
phenotypes and genes as if they were the knowledge graph's (B56). Nothing
calls them: a diagnosis result carries each candidate's identifiers, score and
evidence. Each now refuses with 501 (`src.api.routes.not_built`); disease
details read from the served graph are not built.

Module: src/api/routes/disease.py
"""
from __future__ import annotations

from fastapi import APIRouter

from src.api.routes.not_built import RESPONSES, refuse_not_built

router = APIRouter()

#: What a client uses instead, said in every refusal here.
INSTEAD = ("A diagnosis result (POST /api/v1/diagnose) carries each candidate's identifiers, "
           "score and evidence.")


@router.get("/disease/{disease_id}", responses=RESPONSES)
async def get_disease(disease_id: str) -> None:
    """Disease details: not built (501)."""
    refuse_not_built("Disease details", INSTEAD)


@router.get("/disease/{disease_id}/phenotypes", responses=RESPONSES)
async def get_disease_phenotypes(disease_id: str) -> None:
    """A disease's phenotypes: not built (501)."""
    refuse_not_built("Disease phenotypes", INSTEAD)


@router.get("/disease/{disease_id}/genes", responses=RESPONSES)
async def get_disease_genes(disease_id: str) -> None:
    """A disease's genes: not built (501)."""
    refuse_not_built("Disease genes", INSTEAD)


__all__ = ["INSTEAD", "router"]
