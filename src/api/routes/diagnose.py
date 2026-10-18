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
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================
#: Route-local request-safety bound on `candidate_genes`, **not** a semantic
#: selection limit for the future gene scorer.
#:
#: Deliberately not `phenotypes`' 100: the reference paper's variant-filtered
#: candidate-gene lists average **244.3 genes with SD 244.0**
#: (`docs/SP_SCORE_GUIDE.md` §3), so a 100-item cap would reject ordinary real
#: lists. Whatever the gene scorer eventually accepts is its own decision, taken
#: with that interface's design.
CANDIDATE_GENES_MAX = 1000

#: Emitted whenever `candidate_genes` is supplied. A test proving the field is
#: inert tells nobody but us; the caller who sent it learns nothing unless the
#: response says so.
CANDIDATE_GENES_IGNORED_WARNING = (
    "candidate_genes was supplied but is not used by the current disease scorer."
)


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
        max_length=CANDIDATE_GENES_MAX,
        description=(
            "RESERVED — accepted and validated, but currently ignored by the "
            "disease scorer. It does not affect the disease candidates, their "
            "scores or their ranks. Supplying it adds a warning to the response. "
            "Reserved at the deploying institution's request for future "
            "causal-gene prioritisation."
        ),
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

    @field_validator("candidate_genes")
    @classmethod
    def validate_candidate_genes(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Strip entries and reject blanks. **No identifier ontology.**

        Which gene identifier namespace this field accepts belongs to the
        interface's real design, not to a reservation. This checks only that a
        caller sent something rather than whitespace.
        """
        if v is None:
            return v
        stripped = [item.strip() for item in v]
        if any(not item for item in stripped):
            raise ValueError("candidate_genes entries must not be blank")
        return stripped

    @field_validator("phenotype_confidences")
    @classmethod
    def validate_confidences(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        """Validate confidence scores"""
        if v is not None:
            for conf in v:
                if not 0.0 <= conf <= 1.0:
                    raise ValueError(f"Confidence must be 0.0-1.0, got {conf}")
        return v

    @model_validator(mode="after")
    def confidences_match_phenotypes(self) -> DiagnoseRequest:
        """One confidence per phenotype, or none at all.

        **Checked here, before anything is dropped.** The pipeline removes
        unknown ids and repeats, and reduces this list to the positions that
        survive. A list of another length cannot be reduced: indexing it would
        fail into a 500, and zipping it would shorten it silently. `None` means
        not supplied; an empty list for a non-empty request is a mismatch like
        any other. The pipeline checks the same for a direct caller.
        """
        confidences = self.phenotype_confidences
        if confidences is not None and len(confidences) != len(self.phenotypes):
            raise ValueError(
                f"phenotype_confidences has {len(confidences)} entries for "
                f"{len(self.phenotypes)} phenotypes; it needs one per phenotype, "
                "or none at all"
            )
        return self

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
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence (eta * gnn + (1-eta) * sp)")
    gnn_score: Optional[float] = Field(None, description="GNN embedding similarity score")
    reasoning_score: Optional[float] = Field(None, description="Path reasoning score (fallback only)")
    sp_score: Optional[float] = Field(None, description="Shortest-path similarity score")
    matching_phenotypes: List[str] = Field(
        default_factory=list,
        description=(
            "The request's phenotypes as used (known to the graph, repeats "
            "removed, in the request's spelling) that the knowledge graph "
            "annotates directly to this disease by a phenotype_of_disease edge "
            "on the exact term; not ontology ancestors or descendants, not "
            "gene-mediated. An empty list does not mean no clinical overlap. "
            "Filled whether or not explanations are requested; empty in the "
            "demo answer given when no pipeline is configured."
        ),
    )
    supporting_genes: List[str] = Field(default_factory=list, description="Supporting gene evidence")
    explanation: Optional[str] = Field(None, description="Human-readable explanation")
    reasoning_paths: Optional[List[List[str]]] = Field(None, description="Reasoning paths through KG")
    evidence_package: Optional[Dict[str, Any]] = Field(None, description="Step C evidence: mode (direct_path/analogy_based/insufficient), confidence_label, paths or analogies")
    confidence_label: Optional[str] = Field(None, description="Clinician-facing label: Strong path support / Weak path support / Analogy-based / Insufficient evidence")


# -----------------------------------------------------------------------------
# The served identity (contract M3c, R8)
# -----------------------------------------------------------------------------
# The response's form of `src.core.types.ServedIdentity`, defined here rather
# than imported: this module is loaded by file, without the application, by
# `scripts/measure_served_pipeline.py`, which validates every response it counts
# against it. `tests/unit/test_served_identity.py` pins the two to one shape and
# one set of states. Every key is required, and its value may be null where the
# state says there is nothing to identify; an unknown key is refused, so the two
# definitions cannot drift apart unnoticed.
_SHA256 = r"^[0-9a-f]{64}$"


class _Strict(BaseModel):
    """An unknown key is refused: the shape is the contract."""

    model_config = ConfigDict(extra="forbid")


class ServedCheckpoint(_Strict):
    """The model's checkpoint, or the stated absence of one."""

    state: Literal["verified", "supplied_in_memory", "absent"] = Field(
        ..., description="verified: built from one read of a checkpoint trained on "
        "the served graph; supplied_in_memory: a model object, no checkpoint bytes; "
        "absent: no model, path reasoning ranks the candidates")
    sha256: Optional[str] = Field(
        ..., pattern=_SHA256, description="SHA-256 of the checkpoint bytes read; "
        "null unless verified")


class ServedGraph(_Strict):
    """The graph served, and the workspace manifest it was verified against."""

    state: Literal["verified", "unverified", "in_memory"] = Field(
        ..., description="verified: kg.json checked against the workspace's "
        "split_manifest.json; unverified: read from a file, checked against no "
        "manifest; in_memory: no file")
    kg_sha256: Optional[str] = Field(
        ..., pattern=_SHA256, description="SHA-256 of the kg.json bytes read; null "
        "for a graph supplied in memory")
    split_manifest_sha256: Optional[str] = Field(
        ..., pattern=_SHA256, description="SHA-256 of the split_manifest.json bytes "
        "read; null unless verified")


class ServedShortestPaths(_Strict):
    """The shortest-path table, or the stated absence of one."""

    state: Literal["verified", "absent"] = Field(
        ..., description="verified: the table names the served graph; absent: no "
        "table contributes to the score")
    build_id: Optional[str] = Field(
        ..., description="The build_id the table and its sidecar share; null when absent")
    max_hops: Optional[int] = Field(
        ..., description="The sidecar's hop bound; null when absent")


class ServedDatasetRecord(_Strict):
    """The served workspace's dataset record, or its absence."""

    state: Literal["verified", "absent"] = Field(
        ..., description="verified: the record the workspace manifest binds, naming "
        "the served graph; absent: no workspace was verified")
    origin: Optional[Literal["files", "synthetic"]] = Field(
        ..., description="files: built from identified source files; synthetic: a "
        "generated stand-in, not a clinical knowledge graph")
    sha256: Optional[str] = Field(
        ..., pattern=_SHA256, description="SHA-256 of the kg.provenance.json bytes read")


class ServedSettings(_Strict):
    """The serving settings that change scores, as in effect."""

    device: Optional[str] = Field(
        ..., description="Where the model ran (torch's name, e.g. cuda:0, cpu); null "
        "when no model is served")
    device_requested: Optional[str] = Field(
        ..., description="What was asked for: auto (CUDA, refused without it), cuda, "
        "cuda:N or cpu; null when no model is served")
    eta: Optional[float] = Field(
        ..., description="Mixing weight in effect between the GNN and shortest-path "
        "terms (1.0 with no table); null when no model is served")
    max_path_length: int
    path_length_penalty: float
    aggregation_method: str
    max_paths_per_candidate: int
    use_direct_path_optimization: bool
    phenotype_normalisation_version: int


class ServedIdentity(_Strict):
    """What produced a result, part by part (contract M3c)."""

    scoring_mode: Literal["gnn_plus_shortest_path", "gnn_only", "path_reasoning_fallback"]
    checkpoint: ServedCheckpoint
    graph: ServedGraph
    shortest_paths: ServedShortestPaths
    dataset_record: ServedDatasetRecord
    settings: ServedSettings


class DiagnoseResponse(BaseModel):
    """Diagnosis response model"""

    session_id: str = Field(..., description="Unique session identifier")
    patient_id: str = Field(..., description="Patient identifier")
    timestamp: str = Field(..., description="ISO timestamp")
    candidates: List[DiagnosisCandidate] = Field(..., description="Ranked diagnosis candidates")
    summary: Optional[str] = Field(None, description="Summary explanation")
    inference_time_ms: float = Field(..., description="Processing time in milliseconds")
    served_identity: Optional[ServedIdentity] = Field(
        ...,
        description="What produced this result: the checkpoint, graph, workspace "
        "manifest, shortest-path table and dataset record by digest or identifier, "
        "the scoring mode and the settings in effect, each part stating its state "
        "(contract M3c). Taken from the result this response answers with. Null "
        "only for the demo answer given when no pipeline is configured: no model, "
        "graph or table produced those candidates.",
    )
    warnings: List[str] = Field(default_factory=list, description="Any warnings")

    model_config = {"json_schema_extra": {
        "example": {
            "session_id": "sess_a1b2c3d4e5f6",
            "patient_id": "demo_001",
            "timestamp": "2026-04-28T10:30:00Z",
            "candidates": [
                {
                    "rank": 1,
                    "disease_id": "mondo:MONDO:0011073",
                    "disease_name": "Dravet syndrome",
                    "confidence_score": 0.72,
                    "gnn_score": 0.65,
                    "sp_score": 0.88,
                    "confidence_label": "Strong path support",
                    "explanation": "Direct KG path: Seizure → SCN1A → Dravet syndrome",
                }
            ],
            "inference_time_ms": 2.5,
            "served_identity": {
                "scoring_mode": "gnn_plus_shortest_path",
                "checkpoint": {"state": "verified", "sha256": "4f80" + "0" * 60},
                "graph": {"state": "verified", "kg_sha256": "9a1c" + "0" * 60,
                          "split_manifest_sha256": "131c" + "0" * 60},
                "shortest_paths": {"state": "verified",
                                   "build_id": "0123456789abcdef0123456789abcdef",
                                   "max_hops": 5},
                "dataset_record": {"state": "verified", "origin": "files",
                                   "sha256": "2c7a" + "0" * 60},
                "settings": {"device": "cuda:0", "device_requested": "auto",
                             "eta": 0.7, "max_path_length": 4,
                             "path_length_penalty": 0.9,
                             "aggregation_method": "weighted_sum",
                             "max_paths_per_candidate": 10,
                             "use_direct_path_optimization": True,
                             "phenotype_normalisation_version": 1},
            },
        }
    }}


def _summary(n_candidates: int, n_received: int, phenotype_input: Any) -> str:
    """The response's one-line summary, counting what was scored.

    **The count comes from the result, not the request.** The pipeline drops
    unknown ids and removes repeats, so the request's length counted entries that
    were never scored: `[A, A, B]` read "for 3 phenotypes" after the pipeline had
    scored two. The received count is shown only when it differs, and labelled as
    received. The mock path has no result, so its count is labelled as received.
    """
    if phenotype_input is None:
        return (
            f"Found {n_candidates} candidate diagnoses for {n_received} "
            "phenotypes received"
        )
    used = len(phenotype_input.used)
    summary = f"Found {n_candidates} candidate diagnoses for {used} phenotypes"
    if phenotype_input.received != used:
        summary += f" ({phenotype_input.received} received)"
    return summary


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
    # What the pipeline actually scored. Stays None on the mock path, which has
    # no pipeline to say so.
    phenotype_input = None
    # What produced the answer (contract M3c), copied from the result below and
    # never read from shared state a reload could swap. Stays None on the mock
    # path, where nothing was served.
    served_identity = None

    # Reserved, accepted, ignored — and the caller is told so on every path
    # below, because the field is equally inert in the mock fallback.
    #
    # `is not None`, not truthiness. An explicitly supplied `[]` **was** supplied,
    # and the settled contract and the field's own description both say supplying
    # the field warns. A truthiness test would silently encode a "non-empty"
    # contract that nobody agreed to, and would leave a caller who sent `[]`
    # believing it had an effect.
    if request.candidate_genes is not None:
        warnings.append(CANDIDATE_GENES_IGNORED_WARNING)

    try:
        # Get application state
        from src.api.main import app_state, initialize_pipeline

        # Attempt lazy pipeline initialization if not yet loaded, **and not
        # after a build has already been asked for.** `real_pipeline_requested`
        # is set by every build that names a workspace or a checkpoint, so it is
        # True after a startup or an earlier lazy build that failed, and after a
        # refused reload. Retrying here repeated the graph load, the workspace
        # digests and the tensor reads on every request -- synchronously, in an
        # async route, so `/health` and the WebUI waited behind it -- only to
        # answer the same 503. **A retry is asked for, not implied:** the files
        # or the mount behind a failed build can be repaired while the service
        # runs, and the way to try again is a reload (or a restart), not
        # expensive I/O on every diagnosis.
        if app_state.pipeline is None and not app_state.real_pipeline_requested:
            try:
                initialize_pipeline()
            except Exception as e:
                logger.warning(f"Lazy pipeline init failed: {e}")

        if app_state.pipeline is None:
            # **A configured pipeline that failed is not a demo.**
            #
            # `_generate_mock_candidates` returns real MONDO identifiers, real
            # disease names, real gene symbols and confidence scores from 0.95
            # down, over HTTP 200. That is the intended answer when no pipeline
            # was ever configured — someone trying the service out. It is the
            # wrong answer when a deployment *was* configured and its pipeline
            # could not be built: the caller asked a clinical question of a
            # system that cannot answer it, and a warning string beside a
            # well-formed ranked list is not a refusal.
            #
            # **What separates the two is whether a workspace was ever named**,
            # by any route. The first version of this keyed on
            # `SHEPHERD_KG_PATH`, which covered startup and missed the reload
            # API — a caller could point `/pipeline/reload` at a real workspace,
            # have the candidate refused, and still be handed invented
            # candidates here. `build_pipeline` records the request where both
            # routes resolve their paths. `/ready` reads the same definition.
            if app_state.requested_pipeline_missing:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=(
                        "A pipeline is configured for this deployment but could "
                        "not be initialized, so no diagnosis can be produced. "
                        "See the service logs for the condition that stopped it; "
                        "load a pipeline (POST /api/v1/pipeline/reload) or "
                        "restart the service to try again."
                    ),
                )

            # Fallback to mock response when no pipeline is configured at all
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
                    sp_score=c.sp_score,
                    matching_phenotypes=c.matching_phenotypes,
                    supporting_genes=c.supporting_genes,
                    explanation=c.explanation if request.include_explanations else None,
                    reasoning_paths=(
                        [[str(n) for n in path] for path in c.reasoning_paths]
                        if request.include_paths and c.reasoning_paths else None
                    ),
                    evidence_package=c.evidence_package if request.include_explanations else None,
                    confidence_label=c.confidence_label if request.include_explanations else None,
                )
                for c in result.candidates
            ]

            if result.warnings:
                warnings.extend(result.warnings)
            phenotype_input = result.phenotype_input
            served_identity = result.served_identity.to_dict()

    except HTTPException:
        # **Deliberate statuses pass through.** `HTTPException` is an
        # `Exception`, so the handler below would relabel a considered 503 as an
        # unexpected 500 — which reads to a caller as a bug in the service rather
        # than as the service declining to answer.
        raise
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
        summary=_summary(len(candidates), len(request.phenotypes), phenotype_input),
        inference_time_ms=inference_time_ms,
        served_identity=served_identity,
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
        # Build mock reasoning paths: phenotype → gene → disease
        mock_paths = [
            [p, genes[0], disease_id]
            for p in phenotypes[:min(2, len(phenotypes))]
        ] if genes else []
        candidates.append(DiagnosisCandidate(
            rank=i + 1,
            disease_id=disease_id,
            disease_name=disease_name,
            confidence_score=max(0.1, score),
            gnn_score=max(0.1, score + 0.02),
            reasoning_score=max(0.1, score - 0.02),
            # No graph was consulted, so nothing is annotated to these invented
            # diseases. This used to echo the first three request phenotypes,
            # which read as matches and meant nothing.
            matching_phenotypes=[],
            supporting_genes=genes,
            explanation=f"Candidate diagnosis based on {len(phenotypes)} phenotypes. "
                        f"Key supporting genes: {', '.join(genes)}."
                        if include_explanations else None,
            reasoning_paths=mock_paths if mock_paths else None,
        ))

    return candidates
