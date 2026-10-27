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
    - DiagnoseResponse: one kind of the result union (Spec 1 §2.1), discriminated
      by `result_kind`: `served_mixture/v1`, `path_reasoning/v1` or `demo/v1`,
      each with its own candidate fields and no other kind's

Version: 1.0.0
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, ConfigDict, Field, RootModel, field_validator, model_validator

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


# -----------------------------------------------------------------------------
# The result union (Spec 1 §2.1, Batch 4)
# -----------------------------------------------------------------------------
# The response's form of `src.core.types`' result union, defined here for the
# same reason as the identity above: this module is loaded by file, without the
# application. `tests/unit/test_served_identity.py` pins each mirror to its core
# type (field names, the kinds, the SP statuses). **Every kind refuses another
# kind's fields** (`_Strict`), so a removed field cannot come back unnoticed, and
# a number is never written where its kind has none.
RESULT_KINDS = ("served_mixture/v1", "path_reasoning/v1", "demo/v1")
SP_NUMERIC_STATUSES = ("COMPUTED", "COMPUTED_PARTIAL")
SP_NON_NUMERIC_STATUSES = ("NO_TABLE", "TARGET_UNMAPPED", "NO_PHENOTYPE_MAPPED")
SERVED_TIE_POLICY = "served-stable-discovery-order/v1"

#: A number that is a number: NaN and infinity are refused at the schema, so a
#: non-finite value can never reach a caller as a score.
_Finite = Annotated[float, Field(allow_inf_nan=False)]

_MATCHING_PHENOTYPES_DESCRIPTION = (
    "The request's phenotypes as used (known to the graph, repeats "
    "removed, in the request's spelling) that the knowledge graph "
    "annotates directly to this disease by a phenotype_of_disease edge "
    "on the exact term; not ontology ancestors or descendants, not "
    "gene-mediated. An empty list does not mean no clinical overlap. "
    "Filled whether or not explanations are requested; empty in the "
    "demo answer given when no pipeline is configured."
)


class SPTerm(_Strict):
    """A candidate's shortest-path record (Spec 1 §3.3–§3.4): its status, and
    numbers only on a numeric status. Every key is required; a value is null
    where the status says there is no number."""

    status: Literal["COMPUTED", "COMPUTED_PARTIAL", "NO_TABLE", "TARGET_UNMAPPED",
                    "NO_PHENOTYPE_MAPPED"]
    phenotypes_submitted: int = Field(..., ge=0)
    phenotypes_mapped: int = Field(..., ge=0)
    unmapped_ids: List[str]
    max_hops: Optional[int] = Field(
        ..., description="The loaded table's hop bound; null when no table is loaded")
    mean_hop_distance: Optional[_Finite] = Field(
        ..., description="Mean over the mapped phenotypes of the hop distance to the "
        "candidate, an unreachable pair counting max_hops + 1; numeric statuses only")
    score: Optional[_Finite] = Field(
        ..., description="1 / (1 + mean_hop_distance); numeric statuses only")
    unreachable_pair_count: Optional[int] = Field(..., ge=0)
    denominator_policy: Optional[Literal["MAPPED_ONLY_DENOMINATOR"]]
    mixed_as: Optional[float] = Field(
        ..., description="0.0 where a non-numeric status still entered the legacy "
        "mixture as 0, so its ranking did not change; null otherwise")

    @model_validator(mode="after")
    def _numbers_follow_the_status(self) -> "SPTerm":
        numbers = (self.mean_hop_distance, self.score, self.unreachable_pair_count,
                   self.denominator_policy)
        if self.status in SP_NUMERIC_STATUSES:
            if any(value is None for value in numbers) or self.max_hops is None:
                raise ValueError(f"a {self.status} record carries its numbers")
            if self.mixed_as is not None:
                raise ValueError("a numeric record is mixed as its own value")
            if (self.status == "COMPUTED_PARTIAL") != bool(self.unmapped_ids):
                raise ValueError("COMPUTED_PARTIAL means some submitted phenotypes "
                                 "were not mapped")
        else:
            if any(value is not None for value in numbers):
                raise ValueError(f"a {self.status} record has no number")
            if (self.status == "NO_TABLE") != (self.max_hops is None):
                raise ValueError("max_hops is null exactly when no table is loaded")
            if self.mixed_as is not None and self.mixed_as != 0.0:
                raise ValueError("a non-numeric status is mixed as 0.0")
        if self.phenotypes_submitted != self.phenotypes_mapped + len(self.unmapped_ids):
            raise ValueError("the submitted phenotypes are the mapped ones and the unmapped ids")
        return self


class PhenotypeInput(_Strict):
    """What happened to the request's phenotype list before it was scored
    (`src.core.types.PhenotypeInputAccount`)."""

    received: int = Field(..., ge=0, description="Entries the request carried")
    unknown: List[str] = Field(..., description="Ids with no node in the graph, dropped")
    repeats_removed: int = Field(..., ge=0)
    used: List[str] = Field(..., description="The phenotypes scored, in first-occurrence "
                            "order; empty when the request was refused")


class _Candidate(_Strict):
    """What a candidate of every kind carries."""

    rank: int = Field(..., ge=1, description="Ranking position (1-based)")
    disease_id: str = Field(..., description="Disease identifier (MONDO/OMIM)")
    disease_name: str = Field(..., description="Disease name")
    matching_phenotypes: List[str] = Field(
        default_factory=list, description=_MATCHING_PHENOTYPES_DESCRIPTION)


class _ServedCandidate(_Candidate):
    """The evidence today's two served kinds build from the discovery paths."""

    supporting_genes: List[str] = Field(
        default_factory=list, description="Genes on this disease's discovery paths")
    explanation: Optional[str] = Field(None, description="Human-readable explanation")
    reasoning_paths: Optional[List[List[str]]] = Field(
        None, description="Reasoning paths through KG")
    evidence_package: Optional[Dict[str, Any]] = Field(
        None, description="Step C evidence: mode (direct_path/analogy_based/"
        "insufficient), evidence_label, paths or analogies")
    evidence_label: Optional[str] = Field(
        None, description="Grades the evidence, not a confidence in the diagnosis: "
        "Strong path support / Weak path support / Analogy-based / Insufficient evidence")


class ServedMixtureCandidate(_ServedCandidate):
    """A disease reached by path discovery, ranked by the legacy mixture."""

    served_score: _Finite = Field(
        ..., description="The value ranked by: eta * model_term + (1 - eta) * SP, or "
        "model_term alone with no shortest-path table")
    # No [0, 1] bound: the cosine is float32 and can round an ulp past +-1 for
    # a patient parallel to a disease, so a valid term can sit just outside.
    model_term: _Finite = Field(..., description="(cos + 1) / 2, computed in float32; "
                                "it can lie an ulp outside [0, 1]")
    sp_term: SPTerm


class PathReasoningCandidate(_ServedCandidate):
    """No model is loaded: ranked by its paths, with no model or SP value."""

    path_score: _Finite = Field(..., ge=0.0, description="The path aggregate ranked by")


class DemoCandidate(_Candidate):
    """The demo answer's candidate: no score, explanation, path or gene of any kind."""


class _Response(_Strict):
    """What a response of every kind carries."""

    session_id: str = Field(..., description="Unique session identifier")
    patient_id: str = Field(..., description="Patient identifier")
    timestamp: str = Field(..., description="ISO timestamp")
    summary: Optional[str] = Field(None, description="Summary explanation")
    inference_time_ms: float = Field(..., description="Processing time in milliseconds")
    warnings: List[str] = Field(default_factory=list, description="Any warnings")
    result_schema_version: Literal[2]
    phenotype_input: Optional[PhenotypeInput] = Field(
        ..., description="The phenotype list as received and as used; null in the demo "
        "answer and when the request was refused before its phenotypes were mapped")


def _ranked(candidates: List[Any], score: str) -> None:
    ranks = [c.rank for c in candidates]
    if ranks != list(range(1, len(ranks) + 1)):
        raise ValueError(f"ranks must be exactly 1..{len(ranks)}")
    if score:
        values = [getattr(c, score) for c in candidates]
        if any(later > earlier for earlier, later in zip(values, values[1:])):
            raise ValueError(f"candidates are not in non-increasing order of {score}")


_IDENTITY_DESCRIPTION = (
    "What produced this result: the checkpoint, graph, workspace manifest, "
    "shortest-path table and dataset record by digest or identifier, the scoring "
    "mode and the settings in effect, each part stating its state (contract M3c). "
    "Taken from the result this response answers with."
)


class ServedMixtureResponse(_Response):
    """`served_mixture/v1`: today's served ranking with a model."""

    result_kind: Literal["served_mixture/v1"]
    served_identity: ServedIdentity = Field(..., description=_IDENTITY_DESCRIPTION)
    candidates: List[ServedMixtureCandidate]
    eta: _Finite = Field(..., description="The mixing weight in effect, the served "
                         "identity's (1.0 with no table, where nothing is mixed)")
    candidates_reached: int = Field(
        ..., ge=0, description="How many diseases discovery reached before the cut")
    tie_policy: Literal["served-stable-discovery-order/v1"]

    @model_validator(mode="after")
    def _a_mixture(self) -> "ServedMixtureResponse":
        _ranked(self.candidates, "served_score")
        mode = self.served_identity.scoring_mode
        if mode not in ("gnn_plus_shortest_path", "gnn_only"):
            raise ValueError(f"a served_mixture result is not produced in {mode!r}")
        if self.eta != self.served_identity.settings.eta:
            raise ValueError("eta is the served identity's")
        if self.candidates_reached < len(self.candidates):
            raise ValueError("candidates_reached counts at least the candidates shown")
        mixes = mode == "gnn_plus_shortest_path"
        for candidate in self.candidates:
            record = candidate.sp_term
            if (record.status == "NO_TABLE") == mixes:
                raise ValueError("an SP record is NO_TABLE exactly when no table is served")
            numeric = record.status in SP_NUMERIC_STATUSES
            if (mixes and not numeric) != (record.mixed_as is not None):
                raise ValueError("mixed_as is 0.0 exactly on a non-numeric status the "
                                 "mixture mixed")
        return self


class PathReasoningResponse(_Response):
    """`path_reasoning/v1`: no model loaded, ranked by path reasoning."""

    result_kind: Literal["path_reasoning/v1"]
    served_identity: ServedIdentity = Field(..., description=_IDENTITY_DESCRIPTION)
    candidates: List[PathReasoningCandidate]
    path_score_aggregation: Literal["weighted_sum/v1", "max/v1", "mean/v1", "sum/v1"]
    tie_policy: Literal["served-stable-discovery-order/v1"]

    @model_validator(mode="after")
    def _path_reasoning(self) -> "PathReasoningResponse":
        _ranked(self.candidates, "path_score")
        if self.served_identity.scoring_mode != "path_reasoning_fallback":
            raise ValueError("a path_reasoning result is served only with no model")
        return self


class DemoResponse(_Response):
    """`demo/v1`: no pipeline was ever configured. Not a diagnosis, and it
    carries no score, explanation, path or gene."""

    result_kind: Literal["demo/v1"]
    served_identity: None = Field(
        ..., description="Null: no model, graph or table produced the demo answer")
    candidates: List[DemoCandidate]

    @model_validator(mode="after")
    def _a_demo(self) -> "DemoResponse":
        _ranked(self.candidates, "")
        return self


class DiagnoseResponse(RootModel[Annotated[
        Union[ServedMixtureResponse, PathReasoningResponse, DemoResponse],
        Field(discriminator="result_kind")]]):
    """Diagnosis response: one kind of the result union, chosen by `result_kind`."""

    model_config = {"json_schema_extra": {
        "example": {
            "result_kind": "served_mixture/v1",
            "result_schema_version": 2,
            "session_id": "sess_a1b2c3d4e5f6",
            "patient_id": "demo_001",
            "timestamp": "2026-04-28T10:30:00Z",
            "eta": 0.7,
            "candidates_reached": 42,
            "tie_policy": "served-stable-discovery-order/v1",
            "phenotype_input": {"received": 2, "unknown": [], "repeats_removed": 0,
                                "used": ["HP:0001250", "HP:0001263"]},
            "candidates": [
                {
                    "rank": 1,
                    "disease_id": "mondo:MONDO:0011073",
                    "disease_name": "Dravet syndrome",
                    "served_score": 0.605,
                    "model_term": 0.65,
                    "sp_term": {"status": "COMPUTED", "phenotypes_submitted": 2,
                                "phenotypes_mapped": 2, "unmapped_ids": [],
                                "max_hops": 5, "mean_hop_distance": 1.0,
                                "score": 0.5, "unreachable_pair_count": 0,
                                "denominator_policy": "MAPPED_ONLY_DENOMINATOR",
                                "mixed_as": None},
                    "evidence_label": "Strong path support",
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


def _candidate_body(candidate: Any, include_explanations: bool,
                    include_paths: bool) -> Dict[str, Any]:
    """A pipeline candidate as its kind's response candidate: the kind's own
    fields copied, nothing coerced (an SP record keeps its nulls)."""
    from dataclasses import asdict

    from src.core.types import ServedMixtureCandidate as CoreMixtureCandidate

    body: Dict[str, Any] = {
        "rank": candidate.rank,
        "disease_id": str(candidate.disease_id),
        "disease_name": candidate.disease_name,
        "matching_phenotypes": list(candidate.matching_phenotypes),
        "supporting_genes": list(candidate.supporting_genes),
        "explanation": candidate.explanation if include_explanations else None,
        "reasoning_paths": (
            [[str(n) for n in path] for path in candidate.reasoning_paths]
            if include_paths and candidate.reasoning_paths else None
        ),
        "evidence_package": candidate.evidence_package if include_explanations else None,
        "evidence_label": candidate.evidence_label if include_explanations else None,
    }
    if isinstance(candidate, CoreMixtureCandidate):
        body.update(served_score=candidate.served_score, model_term=candidate.model_term,
                    sp_term=asdict(candidate.sp_term))
    else:
        body["path_score"] = candidate.path_score
    return body


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

    from dataclasses import asdict

    from src.core.types import ResultRefused, result_kind_for

    warnings = []
    # **Everything kind-specific comes from the result** (`result_kind`, η,
    # `candidates_reached`, the identity), never from shared state a reload
    # could swap. The mock path has no result: its kind is the demo's, and it
    # has no account of what was scored.
    account = None

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
            # `_generate_mock_candidates` returns real MONDO identifiers and
            # real disease names, over HTTP 200, with no score of any kind
            # (`demo/v1`). That is the intended answer when no pipeline
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

            candidates = _generate_mock_candidates(request.phenotypes, request.top_k)
            body: Dict[str, Any] = {
                "result_kind": result_kind_for(None),
                "served_identity": None,
                "phenotype_input": None,
                "candidates": [c.model_dump() for c in candidates],
            }
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

            if result.warnings:
                warnings.extend(result.warnings)
            account = result.phenotype_input
            body = {
                "result_kind": result.result_kind,
                "served_identity": result.served_identity.to_dict(),
                "phenotype_input": (asdict(result.phenotype_input)
                                    if result.phenotype_input is not None else None),
                "candidates": [
                    _candidate_body(c, request.include_explanations, request.include_paths)
                    for c in result.candidates
                ],
            }
            if result.result_kind == "served_mixture/v1":
                body.update(eta=result.eta, candidates_reached=result.candidates_reached,
                            tie_policy=result.tie_policy)
            elif result.result_kind == "path_reasoning/v1":
                body.update(path_score_aggregation=result.path_score_aggregation,
                            tie_policy=result.tie_policy)

        body.update(
            result_schema_version=2,
            session_id=session_id,
            patient_id=patient_id,
            timestamp=datetime.now().isoformat(),
            summary=_summary(len(body["candidates"]), len(request.phenotypes), account),
            inference_time_ms=(time.time() - start_time) * 1000,
            warnings=warnings,
        )
        # Validated here, inside the handler, so a response that breaks the
        # union's rules is refused with its reason, never sent.
        response = DiagnoseResponse.model_validate(body)

    except HTTPException:
        # **Deliberate statuses pass through.** `HTTPException` is an
        # `Exception`, so the handler below would relabel a considered 503 as an
        # unexpected 500 — which reads to a caller as a bug in the service rather
        # than as the service declining to answer.
        raise
    except ResultRefused as e:
        # **The pipeline refused to write a number it does not have** (Spec 1
        # §2.1): a model term it cannot compute, a row outside its table, a
        # score that is not finite. A state of the served model and graph, not
        # of the request, so a server status; the detail starts with the code.
        logger.error(f"Diagnosis refused: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
    except Exception as e:
        logger.exception(f"Diagnosis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Diagnosis failed: {str(e)}",
        )

    return response


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
def _generate_mock_candidates(phenotypes: List[str], top_k: int) -> List[DemoCandidate]:
    """The demo answer's candidates (`demo/v1`): illustrative diseases with no
    score, explanation, path or gene of any kind, for a service tried with
    nothing configured. Nothing here was computed, so nothing pretends it was."""
    mock_diseases = [
        ("MONDO:0007947", "Marfan syndrome"),
        ("MONDO:0008029", "Neurofibromatosis type 1"),
        ("MONDO:0007915", "Loeys-Dietz syndrome"),
        ("MONDO:0008066", "Noonan syndrome"),
        ("MONDO:0007669", "Ehlers-Danlos syndrome"),
        ("MONDO:0010785", "Tuberous sclerosis"),
        ("MONDO:0008426", "Stickler syndrome"),
        ("MONDO:0007037", "Achondroplasia"),
        ("MONDO:0019391", "Fanconi anemia"),
        ("MONDO:0009131", "Cystic fibrosis"),
    ]
    # No graph was consulted, so nothing is annotated to these diseases: the
    # matching phenotypes are empty, never an echo of the request.
    return [
        DemoCandidate(rank=i + 1, disease_id=disease_id, disease_name=disease_name,
                      matching_phenotypes=[])
        for i, (disease_id, disease_name) in enumerate(mock_diseases[:top_k])
    ]
