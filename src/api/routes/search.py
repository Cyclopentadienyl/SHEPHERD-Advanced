"""
SHEPHERD-Advanced HPO term routes: not built
============================================
GET /hpo/search, /hpo/{term_id}, /hpo/{term_id}/ancestors and
/hpo/{term_id}/descendants were placeholders answering fixed mock terms as if
they were the ontology's (B56). Nothing calls them: the Diagnosis page and
POST /api/v1/diagnose read HP:####### ids as given. Each now refuses with 501
(`src.api.routes.not_built`); a term search over the served ontology is not
built.

Module: src/api/routes/search.py
"""
from __future__ import annotations

from fastapi import APIRouter

from src.api.routes.not_built import RESPONSES, refuse_not_built

router = APIRouter()

#: What a client uses instead, said in every refusal here.
INSTEAD = ("Diagnosis reads HP:####### ids as given (POST /api/v1/diagnose); look a term up in "
           "the HPO itself.")


@router.get("/hpo/search", responses=RESPONSES)
async def search_hpo() -> None:
    """HPO term search: not built (501)."""
    refuse_not_built("HPO term search", INSTEAD)


@router.get("/hpo/{term_id}", responses=RESPONSES)
async def get_hpo_term(term_id: str) -> None:
    """HPO term details: not built (501)."""
    refuse_not_built("HPO term details", INSTEAD)


@router.get("/hpo/{term_id}/ancestors", responses=RESPONSES)
async def get_hpo_ancestors(term_id: str) -> None:
    """HPO term ancestors: not built (501)."""
    refuse_not_built("HPO term ancestors", INSTEAD)


@router.get("/hpo/{term_id}/descendants", responses=RESPONSES)
async def get_hpo_descendants(term_id: str) -> None:
    """HPO term descendants: not built (501)."""
    refuse_not_built("HPO term descendants", INSTEAD)


__all__ = ["INSTEAD", "router"]
