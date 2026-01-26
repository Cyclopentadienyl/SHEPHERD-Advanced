"""
SHEPHERD-Advanced Search API Routes
===================================
REST endpoints for HPO term search.

Module: src/api/routes/search.py
Absolute Path: /home/user/SHEPHERD-Advanced/src/api/routes/search.py

Purpose:
    Provide search API endpoints:
    - GET /hpo/search: Search HPO terms by text query
    - GET /hpo/{term_id}: Get HPO term details
    - GET /hpo/{term_id}/ancestors: Get term ancestors
    - GET /hpo/{term_id}/descendants: Get term descendants

Dependencies:
    - fastapi: Router, query parameters
    - pydantic: Response models
    - src.ontology: HPO ontology access

Input:
    - Search query string
    - HPO term IDs

Output:
    - List of matching HPO terms with metadata

Version: 1.0.0
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Response Models
# =============================================================================
class HPOTerm(BaseModel):
    """HPO term model"""

    term_id: str = Field(..., description="HPO term ID (e.g., HP:0001250)")
    name: str = Field(..., description="Term name")
    definition: Optional[str] = Field(None, description="Term definition")
    synonyms: List[str] = Field(default_factory=list, description="Alternative names")
    is_obsolete: bool = Field(False, description="Whether term is obsolete")

    model_config = {"json_schema_extra": {
        "example": {
            "term_id": "HP:0001250",
            "name": "Seizure",
            "definition": "A sudden, involuntary contraction of muscles...",
            "synonyms": ["Epileptic seizure", "Convulsion"],
            "is_obsolete": False,
        }
    }}


class HPOSearchResult(BaseModel):
    """HPO search result"""

    term: HPOTerm
    score: float = Field(..., ge=0.0, le=1.0, description="Match score")
    match_type: str = Field(..., description="Type of match (exact, prefix, fuzzy)")


class HPOSearchResponse(BaseModel):
    """HPO search response"""

    query: str = Field(..., description="Original query")
    results: List[HPOSearchResult] = Field(..., description="Matching terms")
    total_count: int = Field(..., description="Total matching terms")


class HPOTermDetail(BaseModel):
    """Detailed HPO term information"""

    term: HPOTerm
    parents: List[HPOTerm] = Field(default_factory=list, description="Parent terms")
    children: List[HPOTerm] = Field(default_factory=list, description="Child terms")
    ancestor_count: int = Field(0, description="Number of ancestors")
    descendant_count: int = Field(0, description="Number of descendants")
    associated_genes: List[str] = Field(default_factory=list, description="Associated genes")
    frequency: Optional[str] = Field(None, description="Frequency in affected individuals")


# =============================================================================
# Endpoints
# =============================================================================
@router.get("/hpo/search", response_model=HPOSearchResponse)
async def search_hpo(
    q: str = Query(..., min_length=2, description="Search query"),
    limit: int = Query(10, ge=1, le=100, description="Maximum results"),
    include_obsolete: bool = Query(False, description="Include obsolete terms"),
) -> HPOSearchResponse:
    """
    Search HPO terms by text query

    Searches term names, synonyms, and definitions.
    """
    logger.info(f"HPO search: query='{q}', limit={limit}")

    try:
        from src.api.main import app_state

        if app_state.ontology is not None:
            # Use actual ontology search
            # results = app_state.ontology.search(q, limit=limit)
            pass

        # Mock search results for now
        results = _mock_hpo_search(q, limit, include_obsolete)

        return HPOSearchResponse(
            query=q,
            results=results,
            total_count=len(results),
        )

    except Exception as e:
        logger.exception(f"HPO search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )


@router.get("/hpo/{term_id}", response_model=HPOTermDetail)
async def get_hpo_term(term_id: str) -> HPOTermDetail:
    """
    Get HPO term details

    Returns term information including parents, children, and associations.
    """
    # Validate format
    if not term_id.startswith("HP:"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid HPO term format. Expected HP:XXXXXXX",
        )

    logger.info(f"HPO term lookup: {term_id}")

    try:
        from src.api.main import app_state

        if app_state.ontology is not None:
            # Use actual ontology
            # term = app_state.ontology.get_term(term_id)
            pass

        # Mock response
        result = _mock_hpo_term_detail(term_id)
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"HPO term not found: {term_id}",
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"HPO term lookup failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lookup failed: {str(e)}",
        )


@router.get("/hpo/{term_id}/ancestors", response_model=List[HPOTerm])
async def get_hpo_ancestors(
    term_id: str,
    max_depth: int = Query(None, ge=1, le=20, description="Maximum ancestor depth"),
) -> List[HPOTerm]:
    """
    Get ancestors of an HPO term

    Returns all ancestor terms up to the root or max_depth.
    """
    if not term_id.startswith("HP:"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid HPO term format",
        )

    # Mock ancestors
    return [
        HPOTerm(term_id="HP:0000118", name="Phenotypic abnormality"),
        HPOTerm(term_id="HP:0000001", name="All"),
    ]


@router.get("/hpo/{term_id}/descendants", response_model=List[HPOTerm])
async def get_hpo_descendants(
    term_id: str,
    max_depth: int = Query(1, ge=1, le=5, description="Maximum descendant depth"),
    limit: int = Query(50, ge=1, le=500, description="Maximum results"),
) -> List[HPOTerm]:
    """
    Get descendants of an HPO term

    Returns child terms up to max_depth.
    """
    if not term_id.startswith("HP:"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid HPO term format",
        )

    # Mock descendants (empty for now)
    return []


# =============================================================================
# Mock Data
# =============================================================================
# Common HPO terms for mock search
_MOCK_HPO_TERMS = [
    HPOTerm(
        term_id="HP:0001250",
        name="Seizure",
        definition="A sudden, involuntary contraction of muscles",
        synonyms=["Epileptic seizure", "Convulsion", "Seizures"],
    ),
    HPOTerm(
        term_id="HP:0001252",
        name="Hypotonia",
        definition="Decreased muscle tone",
        synonyms=["Low muscle tone", "Muscle hypotonia"],
    ),
    HPOTerm(
        term_id="HP:0001263",
        name="Global developmental delay",
        definition="Significant delay in reaching developmental milestones",
        synonyms=["Developmental delay", "GDD"],
    ),
    HPOTerm(
        term_id="HP:0002311",
        name="Ataxia",
        definition="Lack of muscle coordination",
        synonyms=["Cerebellar ataxia", "Incoordination"],
    ),
    HPOTerm(
        term_id="HP:0000252",
        name="Microcephaly",
        definition="Abnormally small head circumference",
        synonyms=["Small head", "Decreased head circumference"],
    ),
    HPOTerm(
        term_id="HP:0000256",
        name="Macrocephaly",
        definition="Abnormally large head circumference",
        synonyms=["Large head", "Increased head circumference"],
    ),
    HPOTerm(
        term_id="HP:0001249",
        name="Intellectual disability",
        definition="Significant limitation in intellectual functioning",
        synonyms=["Mental retardation", "Cognitive impairment"],
    ),
    HPOTerm(
        term_id="HP:0000718",
        name="Autism",
        definition="A neurodevelopmental disorder characterized by impaired social interaction",
        synonyms=["Autistic behavior", "ASD"],
    ),
    HPOTerm(
        term_id="HP:0000729",
        name="Autistic behavior",
        definition="Behavior characteristic of autism spectrum disorder",
        synonyms=["Autism spectrum disorder behavior"],
    ),
    HPOTerm(
        term_id="HP:0001290",
        name="Generalized hypotonia",
        definition="Decreased muscle tone affecting the whole body",
        synonyms=["Global hypotonia"],
    ),
]


def _mock_hpo_search(
    query: str,
    limit: int,
    include_obsolete: bool,
) -> List[HPOSearchResult]:
    """Generate mock HPO search results"""
    query_lower = query.lower()
    results = []

    for term in _MOCK_HPO_TERMS:
        # Check for matches
        score = 0.0
        match_type = ""

        name_lower = term.name.lower()
        if query_lower == name_lower:
            score = 1.0
            match_type = "exact"
        elif name_lower.startswith(query_lower):
            score = 0.9
            match_type = "prefix"
        elif query_lower in name_lower:
            score = 0.7
            match_type = "contains"
        else:
            # Check synonyms
            for syn in term.synonyms:
                syn_lower = syn.lower()
                if query_lower in syn_lower:
                    score = 0.6
                    match_type = "synonym"
                    break

        if score > 0:
            results.append(HPOSearchResult(
                term=term,
                score=score,
                match_type=match_type,
            ))

    # Sort by score and limit
    results.sort(key=lambda x: x.score, reverse=True)
    return results[:limit]


def _mock_hpo_term_detail(term_id: str) -> Optional[HPOTermDetail]:
    """Generate mock HPO term detail"""
    for term in _MOCK_HPO_TERMS:
        if term.term_id == term_id:
            return HPOTermDetail(
                term=term,
                parents=[HPOTerm(term_id="HP:0000118", name="Phenotypic abnormality")],
                children=[],
                ancestor_count=3,
                descendant_count=10,
                associated_genes=["SCN1A", "SCN2A"] if "seizure" in term.name.lower() else [],
            )
    return None
