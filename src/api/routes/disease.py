"""
SHEPHERD-Advanced Disease API Routes
====================================
REST endpoints for disease information.

Module: src/api/routes/disease.py
Absolute Path: /home/user/SHEPHERD-Advanced/src/api/routes/disease.py

Purpose:
    Provide disease information API endpoints:
    - GET /disease/{disease_id}: Get disease details
    - GET /disease/{disease_id}/phenotypes: Get associated phenotypes
    - GET /disease/{disease_id}/genes: Get associated genes

Dependencies:
    - fastapi: Router, path parameters
    - pydantic: Response models
    - src.kg: Knowledge graph access

Input:
    - Disease identifiers (MONDO, OMIM, Orphanet)

Output:
    - Disease information with associated data

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
class GeneAssociation(BaseModel):
    """Gene-disease association"""

    gene_id: str = Field(..., description="Gene identifier")
    gene_symbol: str = Field(..., description="Gene symbol")
    association_type: str = Field(..., description="Type of association")
    evidence_level: str = Field(..., description="Evidence strength")
    sources: List[str] = Field(default_factory=list, description="Evidence sources")


class PhenotypeAssociation(BaseModel):
    """Phenotype-disease association"""

    phenotype_id: str = Field(..., description="HPO term ID")
    phenotype_name: str = Field(..., description="Phenotype name")
    frequency: Optional[str] = Field(None, description="Frequency in patients")
    frequency_value: Optional[float] = Field(None, description="Numeric frequency (0-1)")


class DiseaseInfo(BaseModel):
    """Disease information model"""

    disease_id: str = Field(..., description="Disease identifier (MONDO/OMIM/Orphanet)")
    name: str = Field(..., description="Disease name")
    description: Optional[str] = Field(None, description="Disease description")
    synonyms: List[str] = Field(default_factory=list, description="Alternative names")
    external_ids: Dict[str, str] = Field(default_factory=dict, description="External database IDs")
    inheritance_pattern: Optional[str] = Field(None, description="Mode of inheritance")
    prevalence: Optional[str] = Field(None, description="Disease prevalence")
    age_of_onset: Optional[str] = Field(None, description="Typical age of onset")

    model_config = {"json_schema_extra": {
        "example": {
            "disease_id": "MONDO:0007947",
            "name": "Marfan syndrome",
            "description": "A connective tissue disorder...",
            "synonyms": ["MFS"],
            "external_ids": {"OMIM": "154700", "Orphanet": "558"},
            "inheritance_pattern": "Autosomal dominant",
        }
    }}


class DiseaseDetailResponse(BaseModel):
    """Detailed disease response"""

    disease: DiseaseInfo
    phenotypes: List[PhenotypeAssociation] = Field(
        default_factory=list,
        description="Associated phenotypes",
    )
    genes: List[GeneAssociation] = Field(
        default_factory=list,
        description="Associated genes",
    )
    phenotype_count: int = Field(0, description="Total phenotype associations")
    gene_count: int = Field(0, description="Total gene associations")


# =============================================================================
# Endpoints
# =============================================================================
@router.get("/disease/{disease_id}", response_model=DiseaseDetailResponse)
async def get_disease(
    disease_id: str,
    include_phenotypes: bool = Query(True, description="Include associated phenotypes"),
    include_genes: bool = Query(True, description="Include associated genes"),
    phenotype_limit: int = Query(50, ge=1, le=500, description="Max phenotypes"),
    gene_limit: int = Query(20, ge=1, le=100, description="Max genes"),
) -> DiseaseDetailResponse:
    """
    Get disease information

    Returns disease details with associated phenotypes and genes.
    """
    logger.info(f"Disease lookup: {disease_id}")

    # Validate format (basic check)
    valid_prefixes = ["MONDO:", "OMIM:", "Orphanet:", "ORPHA:"]
    if not any(disease_id.startswith(p) for p in valid_prefixes):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid disease ID format. Expected prefixes: {valid_prefixes}",
        )

    try:
        from src.api.main import app_state

        if app_state.kg is not None:
            # Use actual knowledge graph
            # disease_node = app_state.kg.get_node(disease_id)
            pass

        # Mock response
        result = _mock_disease_detail(
            disease_id,
            include_phenotypes,
            include_genes,
            phenotype_limit,
            gene_limit,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Disease not found: {disease_id}",
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Disease lookup failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lookup failed: {str(e)}",
        )


@router.get("/disease/{disease_id}/phenotypes", response_model=List[PhenotypeAssociation])
async def get_disease_phenotypes(
    disease_id: str,
    limit: int = Query(100, ge=1, le=500, description="Maximum results"),
    min_frequency: float = Query(0.0, ge=0.0, le=1.0, description="Minimum frequency"),
) -> List[PhenotypeAssociation]:
    """
    Get phenotypes associated with a disease

    Returns HPO terms associated with the disease.
    """
    valid_prefixes = ["MONDO:", "OMIM:", "Orphanet:", "ORPHA:"]
    if not any(disease_id.startswith(p) for p in valid_prefixes):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid disease ID format",
        )

    # Mock phenotypes
    return _mock_disease_phenotypes(disease_id, limit, min_frequency)


@router.get("/disease/{disease_id}/genes", response_model=List[GeneAssociation])
async def get_disease_genes(
    disease_id: str,
    limit: int = Query(50, ge=1, le=100, description="Maximum results"),
) -> List[GeneAssociation]:
    """
    Get genes associated with a disease

    Returns genes with evidence of disease association.
    """
    valid_prefixes = ["MONDO:", "OMIM:", "Orphanet:", "ORPHA:"]
    if not any(disease_id.startswith(p) for p in valid_prefixes):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid disease ID format",
        )

    # Mock genes
    return _mock_disease_genes(disease_id, limit)


# =============================================================================
# Mock Data
# =============================================================================
_MOCK_DISEASES = {
    "MONDO:0007947": DiseaseInfo(
        disease_id="MONDO:0007947",
        name="Marfan syndrome",
        description="A connective tissue disorder characterized by tall stature, "
                   "long limbs, lens dislocation, and aortic root dilatation.",
        synonyms=["MFS", "Marfan's syndrome"],
        external_ids={"OMIM": "154700", "Orphanet": "558"},
        inheritance_pattern="Autosomal dominant",
        prevalence="1-5 / 10 000",
        age_of_onset="Childhood",
    ),
    "MONDO:0008029": DiseaseInfo(
        disease_id="MONDO:0008029",
        name="Neurofibromatosis type 1",
        description="A genetic condition characterized by changes in skin coloring "
                   "and the growth of tumors along nerves.",
        synonyms=["NF1", "von Recklinghausen disease"],
        external_ids={"OMIM": "162200", "Orphanet": "636"},
        inheritance_pattern="Autosomal dominant",
        prevalence="1-5 / 10 000",
        age_of_onset="Infancy",
    ),
    "MONDO:0007915": DiseaseInfo(
        disease_id="MONDO:0007915",
        name="Loeys-Dietz syndrome",
        description="A connective tissue disorder characterized by arterial aneurysms, "
                   "hypertelorism, and cleft palate.",
        synonyms=["LDS"],
        external_ids={"OMIM": "609192", "Orphanet": "60030"},
        inheritance_pattern="Autosomal dominant",
        prevalence="<1 / 1 000 000",
        age_of_onset="Childhood",
    ),
}

_MOCK_PHENOTYPES = [
    PhenotypeAssociation(
        phenotype_id="HP:0001166",
        phenotype_name="Arachnodactyly",
        frequency="Very frequent (99-80%)",
        frequency_value=0.9,
    ),
    PhenotypeAssociation(
        phenotype_id="HP:0000501",
        phenotype_name="Glaucoma",
        frequency="Frequent (79-30%)",
        frequency_value=0.5,
    ),
    PhenotypeAssociation(
        phenotype_id="HP:0001519",
        phenotype_name="Disproportionate tall stature",
        frequency="Very frequent (99-80%)",
        frequency_value=0.85,
    ),
    PhenotypeAssociation(
        phenotype_id="HP:0001382",
        phenotype_name="Joint hypermobility",
        frequency="Frequent (79-30%)",
        frequency_value=0.6,
    ),
    PhenotypeAssociation(
        phenotype_id="HP:0001083",
        phenotype_name="Ectopia lentis",
        frequency="Frequent (79-30%)",
        frequency_value=0.55,
    ),
]

_MOCK_GENES = [
    GeneAssociation(
        gene_id="HGNC:3603",
        gene_symbol="FBN1",
        association_type="Causative",
        evidence_level="Definitive",
        sources=["ClinGen", "OMIM"],
    ),
    GeneAssociation(
        gene_id="HGNC:11772",
        gene_symbol="TGFBR1",
        association_type="Causative",
        evidence_level="Definitive",
        sources=["ClinGen"],
    ),
    GeneAssociation(
        gene_id="HGNC:11773",
        gene_symbol="TGFBR2",
        association_type="Causative",
        evidence_level="Definitive",
        sources=["ClinGen", "OMIM"],
    ),
]


def _mock_disease_detail(
    disease_id: str,
    include_phenotypes: bool,
    include_genes: bool,
    phenotype_limit: int,
    gene_limit: int,
) -> Optional[DiseaseDetailResponse]:
    """Generate mock disease detail"""
    disease = _MOCK_DISEASES.get(disease_id)
    if disease is None:
        # Try without prefix normalization
        for did, d in _MOCK_DISEASES.items():
            if disease_id in did or did in disease_id:
                disease = d
                break

    if disease is None:
        return None

    phenotypes = _MOCK_PHENOTYPES[:phenotype_limit] if include_phenotypes else []
    genes = _MOCK_GENES[:gene_limit] if include_genes else []

    return DiseaseDetailResponse(
        disease=disease,
        phenotypes=phenotypes,
        genes=genes,
        phenotype_count=len(phenotypes),
        gene_count=len(genes),
    )


def _mock_disease_phenotypes(
    disease_id: str,
    limit: int,
    min_frequency: float,
) -> List[PhenotypeAssociation]:
    """Generate mock disease phenotypes"""
    results = []
    for p in _MOCK_PHENOTYPES:
        if p.frequency_value and p.frequency_value >= min_frequency:
            results.append(p)
        if len(results) >= limit:
            break
    return results


def _mock_disease_genes(disease_id: str, limit: int) -> List[GeneAssociation]:
    """Generate mock disease genes"""
    return _MOCK_GENES[:limit]
