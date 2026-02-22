"""
SHEPHERD-Advanced Diagnosis API Routes
======================================
REST endpoints for disease diagnosis.

Module: src/api/routes/diagnose.py
Absolute Path: /home/user/SHEPHERD-Advanced/src/api/routes/diagnose.py

Purpose:
    Provide diagnosis API endpoints:
    - POST /diagnose: Run diagnosis on patient phenotypes
    - GET /diagnose/{session_id}: Get diagnosis result by session

Dependencies:
    - fastapi: Router, request/response models
    - pydantic: Request validation
    - src.inference.pipeline: DiagnosisPipeline
    - src.core.types: PatientPhenotypes, InferenceResult

Input:
    - DiagnoseRequest: patient_id, phenotypes (HPO IDs), options

Output:
    - DiagnoseResponse: candidates with scores, explanations, metadata

Version: 1.0.0
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================
class DiagnoseRequest(BaseModel):
    """Diagnosis request model"""

    patient_id: Optional[str] = Field(
        default=None,
        description="Patient identifier (auto-generated if not provided)",
    )
    phenotypes: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of HPO term IDs (e.g., ['HP:0001250', 'HP:0002311'])",
    )
    phenotype_confidences: Optional[List[float]] = Field(
        default=None,
        description="Confidence scores for each phenotype (0.0-1.0)",
    )
    candidate_genes: Optional[List[str]] = Field(
        default=None,
        description="Pre-selected candidate genes to consider",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of top diagnoses to return",
    )
    include_explanations: bool = Field(
        default=True,
        description="Include human-readable explanations",
    )
    include_paths: bool = Field(
        default=True,
        description="Include reasoning paths",
    )

    @field_validator("phenotypes")
    @classmethod
    def validate_phenotypes(cls, v: List[str]) -> List[str]:
        """Validate HPO term format"""
        for term in v:
            if not term.startswith("HP:") or len(term) != 10:
                # Relaxed validation - warn but allow
                logger.warning(f"Non-standard HPO format: {term}")
        return v

    @field_validator("phenotype_confidences")
    @classmethod
    def validate_confidences(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        """Validate confidence scores"""
        if v is not None:
            for conf in v:
                if not 0.0 <= conf <= 1.0:
                    raise ValueError(f"Confidence must be 0.0-1.0, got {conf}")
        return v

    model_config = {"json_schema_extra": {
        "example": {
            "patient_id": "patient_001",
            "phenotypes": ["HP:0001250", "HP:0002311", "HP:0001263"],
            "top_k": 10,
            "include_explanations": True,
        }
    }}


class DiagnosisCandidate(BaseModel):
    """Single diagnosis candidate"""

    rank: int = Field(..., description="Ranking position (1-based)")
    disease_id: str = Field(..., description="Disease identifier (MONDO/OMIM)")
    disease_name: str = Field(..., description="Disease name")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    gnn_score: Optional[float] = Field(None, description="GNN model score")
    reasoning_score: Optional[float] = Field(None, description="Path reasoning score")
    matching_phenotypes: List[str] = Field(default_factory=list, description="Matched HPO terms")
    supporting_genes: List[str] = Field(default_factory=list, description="Supporting gene evidence")
    explanation: Optional[str] = Field(None, description="Human-readable explanation")
    reasoning_paths: Optional[List[List[str]]] = Field(None, description="Reasoning paths")


class DiagnoseResponse(BaseModel):
    """Diagnosis response model"""

    session_id: str = Field(..., description="Unique session identifier")
    patient_id: str = Field(..., description="Patient identifier")
    timestamp: str = Field(..., description="ISO timestamp")
    candidates: List[DiagnosisCandidate] = Field(..., description="Ranked diagnosis candidates")
    summary: Optional[str] = Field(None, description="Summary explanation")
    inference_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_version: str = Field(..., description="Model version used")
    warnings: List[str] = Field(default_factory=list, description="Any warnings")

    model_config = {"json_schema_extra": {
        "example": {
            "session_id": "sess_abc123",
            "patient_id": "patient_001",
            "timestamp": "2026-01-25T10:30:00Z",
            "candidates": [
                {
                    "rank": 1,
                    "disease_id": "MONDO:0007947",
                    "disease_name": "Marfan syndrome",
                    "confidence_score": 0.85,
                    "explanation": "High match based on skeletal and cardiovascular phenotypes",
                }
            ],
            "inference_time_ms": 150.5,
            "model_version": "1.0.0",
        }
    }}


# =============================================================================
# Endpoints
# =============================================================================
@router.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(request: DiagnoseRequest) -> DiagnoseResponse:
    """
    Run diagnosis on patient phenotypes

    Analyzes the provided HPO phenotypes and returns ranked disease candidates
    with explanations and supporting evidence.
    """
    import time
    start_time = time.time()

    # Generate session ID
    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    patient_id = request.patient_id or f"patient_{uuid.uuid4().hex[:8]}"

    logger.info(f"Diagnosis request: session={session_id}, phenotypes={len(request.phenotypes)}")

    warnings = []

    try:
        # Get application state
        from src.api.main import app_state, initialize_pipeline

        # Attempt lazy pipeline initialization if not yet loaded
        if app_state.pipeline is None:
            try:
                initialize_pipeline()
            except Exception as e:
                logger.warning(f"Lazy pipeline init failed: {e}")

        if app_state.pipeline is None:
            # Fallback to mock response when pipeline is unavailable
            logger.warning("Pipeline not initialized - returning mock response")
            warnings.append("Pipeline not fully initialized - using mock data")

            candidates = _generate_mock_candidates(
                request.phenotypes,
                request.top_k,
                request.include_explanations,
            )
        else:
            # Run actual diagnosis through the pipeline
            from src.core.types import PatientPhenotypes

            patient_input = PatientPhenotypes(
                patient_id=patient_id,
                phenotypes=request.phenotypes,
                phenotype_confidences=request.phenotype_confidences,
                candidate_genes=request.candidate_genes,
            )

            result = app_state.pipeline.run(
                patient_input=patient_input,
                top_k=request.top_k,
                include_explanations=request.include_explanations,
            )

            candidates = [
                DiagnosisCandidate(
                    rank=c.rank,
                    disease_id=str(c.disease_id),
                    disease_name=c.disease_name,
                    confidence_score=c.confidence_score,
                    gnn_score=c.gnn_score,
                    reasoning_score=c.reasoning_score,
                    supporting_genes=c.supporting_genes,
                    explanation=c.explanation if request.include_explanations else None,
                    reasoning_paths=(
                        [[str(n) for n in path] for path in c.reasoning_paths]
                        if request.include_paths and c.reasoning_paths else None
                    ),
                )
                for c in result.candidates
            ]

            if result.warnings:
                warnings.extend(result.warnings)

    except Exception as e:
        logger.exception(f"Diagnosis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Diagnosis failed: {str(e)}",
        )

    inference_time_ms = (time.time() - start_time) * 1000

    return DiagnoseResponse(
        session_id=session_id,
        patient_id=patient_id,
        timestamp=datetime.now().isoformat(),
        candidates=candidates,
        summary=f"Found {len(candidates)} candidate diagnoses for {len(request.phenotypes)} phenotypes",
        inference_time_ms=inference_time_ms,
        model_version="1.0.0",
        warnings=warnings,
    )


@router.get("/diagnose/{session_id}")
async def get_diagnosis_result(session_id: str) -> Dict[str, Any]:
    """
    Get diagnosis result by session ID

    Retrieves a previously computed diagnosis result.
    (Placeholder for session caching)
    """
    # TODO: Implement session storage and retrieval
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Session retrieval not yet implemented",
    )


# =============================================================================
# Mock Data (for testing without full pipeline)
# =============================================================================
def _generate_mock_candidates(
    phenotypes: List[str],
    top_k: int,
    include_explanations: bool,
) -> List[DiagnosisCandidate]:
    """Generate mock diagnosis candidates for testing"""
    mock_diseases = [
        ("MONDO:0007947", "Marfan syndrome", ["FBN1"]),
        ("MONDO:0008029", "Neurofibromatosis type 1", ["NF1"]),
        ("MONDO:0007915", "Loeys-Dietz syndrome", ["TGFBR1", "TGFBR2"]),
        ("MONDO:0008066", "Noonan syndrome", ["PTPN11", "SOS1"]),
        ("MONDO:0007669", "Ehlers-Danlos syndrome", ["COL5A1", "COL3A1"]),
        ("MONDO:0010785", "Tuberous sclerosis", ["TSC1", "TSC2"]),
        ("MONDO:0008426", "Stickler syndrome", ["COL2A1"]),
        ("MONDO:0007037", "Achondroplasia", ["FGFR3"]),
        ("MONDO:0019391", "Fanconi anemia", ["FANCA", "FANCC"]),
        ("MONDO:0009131", "Cystic fibrosis", ["CFTR"]),
    ]

    candidates = []
    for i, (disease_id, disease_name, genes) in enumerate(mock_diseases[:top_k]):
        score = 0.95 - (i * 0.08)  # Decreasing scores
        candidates.append(DiagnosisCandidate(
            rank=i + 1,
            disease_id=disease_id,
            disease_name=disease_name,
            confidence_score=max(0.1, score),
            gnn_score=max(0.1, score + 0.02),
            reasoning_score=max(0.1, score - 0.02),
            matching_phenotypes=phenotypes[:min(3, len(phenotypes))],
            supporting_genes=genes,
            explanation=f"Candidate diagnosis based on {len(phenotypes)} phenotypes. "
                        f"Key supporting genes: {', '.join(genes)}."
                        if include_explanations else None,
        ))

    return candidates
