"""
SHEPHERD-Advanced Core Types
============================
統一的資料類型定義，所有模組共享

版本: 1.0.0
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    TypeVar,
    Generic,
    Sequence,
)
from datetime import datetime
import numpy as np
from numpy.typing import NDArray


# =============================================================================
# Type Variables
# =============================================================================
T = TypeVar("T")
EmbeddingType = NDArray[np.float32]  # (dim,) or (batch, dim)
TensorType = TypeVar("TensorType")  # For torch.Tensor compatibility


# =============================================================================
# Enums - Node Types
# =============================================================================
class NodeType(str, Enum):
    """
    知識圖譜節點類型

    設計原則:
    - 使用 str 繼承以支援 JSON 序列化
    - 預留擴展空間
    """
    # === Core Medical Entities ===
    GENE = "gene"
    DISEASE = "disease"
    PHENOTYPE = "phenotype"  # HPO terms
    PATHWAY = "pathway"      # GO/Reactome pathways
    DRUG = "drug"
    PROTEIN = "protein"
    VARIANT = "variant"      # Genetic variants

    # === Ontology Nodes ===
    HPO_TERM = "hpo_term"
    MONDO_TERM = "mondo_term"
    GO_TERM = "go_term"

    # === Cross-Species (同源基因比對) ===
    ORTHOLOG_GROUP = "ortholog_group"  # 同源基因群組
    MOUSE_GENE = "mouse_gene"          # 小鼠基因 (MGI)
    MOUSE_PHENOTYPE = "mouse_phenotype"  # 小鼠表型 (MP)
    ZEBRAFISH_GENE = "zebrafish_gene"  # 斑馬魚基因 (ZFIN)

    # === Literature (PubMed) ===
    PUBLICATION = "publication"  # PubMed articles
    AUTHOR = "author"            # Publication authors (optional)

    # === Clinical ===
    PATIENT = "patient"          # Anonymized patient node
    CLINICAL_FINDING = "clinical_finding"


class EdgeType(str, Enum):
    """
    知識圖譜邊類型

    命名慣例: SOURCE_RELATION_TARGET
    """
    # === Gene-Disease Associations ===
    GENE_ASSOCIATED_WITH_DISEASE = "gene_associated_with_disease"
    GENE_CAUSES_DISEASE = "gene_causes_disease"

    # === Phenotype Relations ===
    PHENOTYPE_OF_DISEASE = "phenotype_of_disease"
    GENE_HAS_PHENOTYPE = "gene_has_phenotype"

    # === Ontology Hierarchy ===
    IS_A = "is_a"                    # Ontology subsumption
    PART_OF = "part_of"              # Parthood relation
    HAS_SUBCLASS = "has_subclass"    # Inverse of IS_A

    # === Cross-Species (同源基因比對) ===
    ORTHOLOG_OF = "ortholog_of"              # Human gene <-> Mouse gene
    HUMAN_MOUSE_ORTHOLOG = "human_mouse_ortholog"
    HUMAN_ZEBRAFISH_ORTHOLOG = "human_zebrafish_ortholog"
    ORTHOLOG_IN_GROUP = "ortholog_in_group"  # Gene -> OrthologGroup
    MOUSE_GENE_HAS_PHENOTYPE = "mouse_gene_has_phenotype"

    # === Pathway Relations ===
    GENE_IN_PATHWAY = "gene_in_pathway"
    PATHWAY_INVOLVES_GENE = "pathway_involves_gene"
    PROTEIN_INTERACTS_WITH = "protein_interacts_with"

    # === Drug Relations ===
    DRUG_TARGETS_GENE = "drug_targets_gene"
    DRUG_TREATS_DISEASE = "drug_treats_disease"

    # === Literature (PubMed) ===
    PUBLICATION_MENTIONS_GENE = "publication_mentions_gene"
    PUBLICATION_MENTIONS_DISEASE = "publication_mentions_disease"
    PUBLICATION_SUPPORTS_ASSOCIATION = "publication_supports_association"

    # === Clinical ===
    PATIENT_HAS_PHENOTYPE = "patient_has_phenotype"
    PATIENT_DIAGNOSED_WITH = "patient_diagnosed_with"


class Species(str, Enum):
    """支援的物種 (用於同源基因比對)"""
    HUMAN = "human"           # Homo sapiens (NCBI: 9606)
    MOUSE = "mouse"           # Mus musculus (NCBI: 10090)
    ZEBRAFISH = "zebrafish"   # Danio rerio (NCBI: 7955)
    RAT = "rat"               # Rattus norvegicus (NCBI: 10116)


class EvidenceLevel(str, Enum):
    """證據等級 (用於文獻可信度評估)"""
    META_ANALYSIS = "meta_analysis"      # 最高: 統合分析
    SYSTEMATIC_REVIEW = "systematic_review"
    RCT = "rct"                          # 隨機對照試驗
    COHORT_STUDY = "cohort_study"
    CASE_CONTROL = "case_control"
    CASE_SERIES = "case_series"
    CASE_REPORT = "case_report"          # 最低: 病例報告
    EXPERT_OPINION = "expert_opinion"
    IN_VITRO = "in_vitro"
    IN_SILICO = "in_silico"
    UNKNOWN = "unknown"


class DataSource(str, Enum):
    """資料來源"""
    # === Ontologies ===
    HPO = "hpo"
    MONDO = "mondo"
    GO = "go"
    REACTOME = "reactome"

    # === Gene-Disease Databases ===
    DISGENET = "disgenet"
    CLINVAR = "clinvar"
    OMIM = "omim"
    ORPHANET = "orphanet"

    # === Literature ===
    PUBMED = "pubmed"
    PUBTATOR = "pubtator"

    # === Cross-Species ===
    MGI = "mgi"              # Mouse Genome Informatics
    ZFIN = "zfin"            # Zebrafish Information Network
    ENSEMBL = "ensembl"
    ORTHODB = "orthodb"
    PANTHER = "panther"

    # === Drug ===
    DRUGBANK = "drugbank"
    CHEMBL = "chembl"

    # === Clinical ===
    FHIR = "fhir"
    HISS = "hiss"


# =============================================================================
# Core Data Classes
# =============================================================================
@dataclass(frozen=True)
class NodeID:
    """
    節點唯一識別碼

    格式: {source}:{id}
    例如: hpo:HP:0001250, gene:BRCA1, pubmed:12345678
    """
    source: DataSource
    local_id: str

    def __str__(self) -> str:
        return f"{self.source.value}:{self.local_id}"

    @classmethod
    def from_string(cls, s: str) -> "NodeID":
        """從字串解析 NodeID"""
        parts = s.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid NodeID format: {s}")
        return cls(DataSource(parts[0]), parts[1])

    def __hash__(self) -> int:
        return hash((self.source, self.local_id))


@dataclass
class Node:
    """
    知識圖譜節點
    """
    id: NodeID
    node_type: NodeType
    name: str

    # Optional metadata
    aliases: List[str] = field(default_factory=list)
    description: Optional[str] = None
    species: Optional[Species] = None  # For cross-species nodes

    # Embedding (populated after encoding)
    embedding: Optional[EmbeddingType] = None

    # Provenance
    data_sources: Set[DataSource] = field(default_factory=set)
    last_updated: Optional[datetime] = None

    # Extra attributes (flexible)
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    """
    知識圖譜邊
    """
    source_id: NodeID
    target_id: NodeID
    edge_type: EdgeType

    # Edge attributes
    weight: float = 1.0
    confidence: float = 1.0

    # Evidence
    evidence_sources: List["EvidenceSource"] = field(default_factory=list)
    publication_count: int = 0  # For literature-backed edges

    # Provenance
    data_source: Optional[DataSource] = None
    last_updated: Optional[datetime] = None

    # Extra attributes
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceSource:
    """
    證據來源 (用於邊的可解釋性)
    """
    source_type: DataSource
    source_id: str  # e.g., PMID, ClinVar ID
    evidence_level: EvidenceLevel = EvidenceLevel.UNKNOWN
    confidence_score: float = 1.0

    # For PubMed sources
    publication_year: Optional[int] = None
    journal_impact_factor: Optional[float] = None
    citation_count: Optional[int] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Ortholog (同源基因) Types
# =============================================================================
@dataclass
class OrthologMapping:
    """
    同源基因映射
    用於跨物種基因比對
    """
    human_gene_id: NodeID
    ortholog_gene_id: NodeID
    ortholog_species: Species

    # Orthology confidence
    ortholog_type: str = "one2one"  # one2one, one2many, many2many
    confidence_score: float = 1.0

    # Source database
    source: DataSource = DataSource.ENSEMBL

    # Associated phenotypes in model organism
    model_phenotypes: List[NodeID] = field(default_factory=list)

    # Evidence
    evidence_sources: List[EvidenceSource] = field(default_factory=list)


@dataclass
class OrthologGroup:
    """
    同源基因群組
    包含跨多物種的同源基因
    """
    group_id: str  # e.g., OrthoGroup:12345
    group_name: Optional[str] = None

    # Member genes by species
    human_genes: List[NodeID] = field(default_factory=list)
    mouse_genes: List[NodeID] = field(default_factory=list)
    zebrafish_genes: List[NodeID] = field(default_factory=list)

    # Group-level annotations
    go_terms: List[NodeID] = field(default_factory=list)

    # Source
    source: DataSource = DataSource.ORTHODB


# =============================================================================
# Literature (PubMed) Types
# =============================================================================
@dataclass
class Publication:
    """
    PubMed 文獻記錄
    """
    pmid: str
    title: str
    abstract: Optional[str] = None

    # Journal info
    journal: Optional[str] = None
    publication_year: Optional[int] = None
    impact_factor: Optional[float] = None

    # Citation metrics
    citation_count: int = 0

    # Evidence level
    evidence_level: EvidenceLevel = EvidenceLevel.UNKNOWN

    # Extracted entities (from Pubtator)
    mentioned_genes: List[str] = field(default_factory=list)
    mentioned_diseases: List[str] = field(default_factory=list)
    mentioned_phenotypes: List[str] = field(default_factory=list)

    # Computed credibility score
    credibility_score: Optional[float] = None


@dataclass
class LiteratureEvidence:
    """
    文獻證據 (用於知識圖譜邊的文獻支持)
    """
    publications: List[Publication]

    # Aggregated metrics
    total_publications: int = 0
    avg_credibility: float = 0.0
    max_evidence_level: EvidenceLevel = EvidenceLevel.UNKNOWN

    # Confidence based on literature
    literature_confidence: float = 0.0


# =============================================================================
# Inference Types
# =============================================================================
@dataclass
class PatientPhenotypes:
    """
    患者表型輸入
    """
    patient_id: str

    # HPO phenotypes
    phenotypes: List[str]  # HPO IDs: HP:XXXXXXX
    phenotype_confidences: Optional[List[float]] = None

    # Optional: genetic data
    candidate_genes: Optional[List[str]] = None
    variants: Optional[List[str]] = None

    # Optional: demographics
    age: Optional[int] = None
    sex: Optional[str] = None


# =============================================================================
# The result union (Spec 1 §2.1, Batch 4)
# =============================================================================
# **One discriminated union, the kind on the result, never on a candidate**
# (`docs/working/results-review/SPEC_1_RESULTS_REVIEW.md` §2.1). Every candidate
# of a result has the same score semantics, so a number never changes meaning
# from row to row. Kept here, in the lowest layer, because the explanation text
# (`src.reasoning`), the API, the page and the exports all name the kinds and
# their labels, and `src.reasoning` cannot import `src.inference`.
RESULT_SCHEMA_VERSION = 2
SERVED_MIXTURE = "served_mixture/v1"
#: A finetuned model bundle's disease task (docs/working/scorer-retraining
#: README §3, K3): every disease of the model's graph view, ranked by the
#: disease head's score; path discovery only explains the rows shown.
DISEASE_TASK = "disease_task/v1"
#: A finetuned gene task model (README A8, §6; PLAN_TEST_RESULTS §8.3): the
#: request's own candidate genes, ranked by η·EMBSIM + (1 − η)·SPLSIM.
GENE_TASK = "gene_task/v1"
#: A finetuned patients-like-me model (README A9; PLAN_TEST_RESULTS §8.4): the
#: patients of a registered reference cohort version, ranked by similarity.
PATIENTS_LIKE_ME = "patients_like_me/v1"
RESULT_KINDS = (SERVED_MIXTURE, DISEASE_TASK, GENE_TASK, PATIENTS_LIKE_ME)
#: The mixture ranks by a stable sort, so equal scores keep the order path
#: discovery reached the diseases in.
SERVED_TIE_POLICY = "served-stable-discovery-order/v1"
#: The disease task orders equal scores by the disease's graph-view id, so a
#: tie is shown in a fixed order (README E3: the screen may order ties by a
#: stable id; a metric uses average ranks).
DISEASE_TASK_TIE_POLICY = "score-desc-then-view-id-asc/v1"
#: The gene task orders equal scores by the gene's graph-view id.
GENE_TASK_TIE_POLICY = DISEASE_TASK_TIE_POLICY
#: Patients-like-me orders equal similarities by the reference case id.
PATIENTS_LIKE_ME_TIE_POLICY = "similarity-desc-then-case-id-asc/v1"

#: Spec 1 §3.3's statuses. **Predicates use these explicit sets**: `status !=
#: "COMPUTED"` is not a test for unavailability, since it also catches the
#: numeric `COMPUTED_PARTIAL`.
SP_NUMERIC_STATUSES = ("COMPUTED", "COMPUTED_PARTIAL")
SP_NON_NUMERIC_STATUSES = ("NO_TABLE", "TARGET_UNMAPPED", "NO_PHENOTYPE_MAPPED")
SP_STATUSES = SP_NUMERIC_STATUSES + SP_NON_NUMERIC_STATUSES
#: Unmapped phenotypes never enter the mean, so it divides by the mapped count
#: (Spec 1 §3.4): structural today, not a choice.
MAPPED_ONLY_DENOMINATOR = "MAPPED_ONLY_DENOMINATOR"
#: Each status in words, for a surface that shows one. A numeric status shows
#: its number instead (and a partial one its coverage beside it).
SP_STATUS_WORDS = {
    "COMPUTED": "Computed",
    "COMPUTED_PARTIAL": "Computed on part of the phenotypes",
    "NO_TABLE": "No SP table",
    "TARGET_UNMAPPED": "Disease not mapped",
    "NO_PHENOTYPE_MAPPED": "No phenotype mapped",
}
#: How a surface says that a non-numeric SP status still entered the legacy
#: mixture as 0 (Spec 1 §2.1), so the ranking it shows did not change.
MIXED_AS_WORDS = "mixed as 0 (legacy)"


class ResultRefused(RuntimeError):
    """The pipeline refused to produce a result, rather than write a number it
    does not have (Spec 1 §2.1: the model term fails closed, a non-finite score
    refuses the result).

    These are states of the served model and graph, not of the request: a model
    term that cannot be computed for a node, a row outside its table, a score
    that is not finite. `code` names the class (`MODEL_TERM_UNAVAILABLE`,
    `NON_FINITE_SCORE`); the message names the node or the value. A request the
    validation refuses is not one of these: it is answered with no candidates
    and its reasons in `warnings`, as before.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def _require_finite(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what} must be a number, got {value!r}")
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{what} is not finite: {value!r}")
    return value


@dataclass(frozen=True)
class SPAnalysisRecord:
    """One candidate's shortest-path quantities, or the reason there are none
    (Spec 1 §3.1–§3.4), built by `src.inference.sp_index.sp_analysis_records`.

    **An absent number and a zero cannot be the same thing here**: the numbers
    are required on the two numeric statuses and forbidden on the other three.
    `mean_hop_distance` is the mean over the mapped phenotypes of the hop
    distance to the candidate, an unreachable pair counting `max_hops + 1`;
    `score` is `1 / (1 + mean_hop_distance)`, shown beside it for continuity.

    `mixed_as` belongs to the legacy mixture alone: 0.0 on a non-numeric status
    the mixture still mixed as 0, so its ranking did not change (Spec 1 §2.1).
    It is absent everywhere else, and never set on a numeric status.
    """

    status: str
    #: The phenotypes the request carried that the SP side counts: the mapped
    #: ones and the distinct ids it could not map (`unmapped_ids`).
    phenotypes_submitted: int
    phenotypes_mapped: int
    unmapped_ids: Tuple[str, ...] = ()
    #: The table's hop bound; None when no table is loaded.
    max_hops: Optional[int] = None
    mean_hop_distance: Optional[float] = None
    score: Optional[float] = None
    unreachable_pair_count: Optional[int] = None
    denominator_policy: Optional[str] = None
    mixed_as: Optional[float] = None

    def __post_init__(self) -> None:
        if self.status not in SP_STATUSES:
            raise ValueError(f"sp status {self.status!r} is not one of {SP_STATUSES}")
        numbers = ("mean_hop_distance", "score", "unreachable_pair_count",
                   "denominator_policy")
        if self.numeric:
            missing = [name for name in numbers if getattr(self, name) is None]
            if missing or self.max_hops is None:
                raise ValueError(
                    f"a {self.status} record needs {', '.join(missing or ['max_hops'])}")
            _require_finite(self.mean_hop_distance, "mean_hop_distance")
            _require_finite(self.score, "the SP score")
            if self.mixed_as is not None:
                raise ValueError("a numeric SP record is mixed as its own value, "
                                 "never as a stand-in")
        else:
            present = [name for name in numbers if getattr(self, name) is not None]
            if present:
                raise ValueError(
                    f"a {self.status} record has no number, but carries "
                    f"{', '.join(present)}")
            if (self.status == "NO_TABLE") != (self.max_hops is None):
                raise ValueError("max_hops is the loaded table's bound: absent exactly "
                                 "when there is no table")
            if self.mixed_as is not None and self.mixed_as != 0.0:
                raise ValueError(f"a non-numeric status is mixed as 0.0, not {self.mixed_as!r}")
        if self.numeric and (self.status == "COMPUTED_PARTIAL") != bool(self.unmapped_ids):
            raise ValueError("COMPUTED_PARTIAL means some submitted phenotypes were "
                             "not mapped, COMPUTED that every one was")
        if self.phenotypes_submitted != self.phenotypes_mapped + len(self.unmapped_ids):
            raise ValueError("the submitted phenotypes are the mapped ones and the "
                             "unmapped ids")

    @property
    def numeric(self) -> bool:
        return self.status in SP_NUMERIC_STATUSES


@dataclass(frozen=True)
class _CandidateBase:
    """What every candidate of every kind carries (Spec 1 §2.1)."""

    #: 1..N in the result's order.
    rank: int
    disease_id: NodeID
    disease_name: str


@dataclass(frozen=True)
class _ServedEvidence:
    """The evidence a disease row carries from the discovery paths (Spec 1
    §2.1). None of it changes a score or the order."""

    supporting_genes: List[str] = field(default_factory=list)
    reasoning_paths: List[List[NodeID]] = field(default_factory=list)
    evidence_sources: List[EvidenceSource] = field(default_factory=list)
    # The request's phenotypes as used that the graph annotates directly to this
    # disease, on the exact term: no ontology ancestors or descendants, no
    # gene-mediated links, so an empty list does not mean no clinical overlap.
    # Defined by `src.reasoning.explanation_generator.find_matching_phenotypes`.
    matching_phenotypes: List[str] = field(default_factory=list)
    # Step C evidence: `EvidencePackage.to_dict()` (src/reasoning/evidence_panel.py),
    # stored as a dict to avoid an import cycle with the reasoning module.
    evidence_package: Optional[Dict[str, Any]] = None
    #: Grades the evidence for the candidate ("Strong path support", …), never
    #: a confidence in the diagnosis: the field `confidence_label` was, and went.
    evidence_label: Optional[str] = None
    explanation: Optional[str] = None


@dataclass(frozen=True, kw_only=True)
class ServedMixtureCandidate(_ServedEvidence, _CandidateBase):
    """A candidate of `served_mixture/v1`: today's served ranking, a disease
    reached by path discovery and scored by the legacy mixture."""

    #: The value it is ranked by: η·model_term + (1−η)·SP in
    #: `gnn_plus_shortest_path`, `model_term` alone in `gnn_only`.
    served_score: float
    #: (cos+1)/2 of the pooled patient phenotypes and the disease, in float32:
    #: in [0, 1] up to an ulp either side (a patient parallel or anti-parallel
    #: to a disease rounds the cosine past +-1).
    model_term: float
    sp_term: SPAnalysisRecord

    def __post_init__(self) -> None:
        _require_finite(self.served_score, "served_score")
        _require_finite(self.model_term, "model_term")
        if not isinstance(self.sp_term, SPAnalysisRecord):
            raise ValueError("a served_mixture candidate carries an SP record (sp_term)")


@dataclass(frozen=True, kw_only=True)
class DiseaseTaskCandidate(_ServedEvidence, _CandidateBase):
    """A candidate of `disease_task/v1`: a disease of the model's graph view,
    ranked by the finetuned disease head. The evidence fields explain the row
    from path discovery and change neither the score nor the order."""

    #: The disease head's score, −‖ẑ_d − ẑ_P‖² on normalised vectors, in
    #: [−4, 0]; uncalibrated, a ranking value and never a probability.
    score: float

    def __post_init__(self) -> None:
        _require_finite(self.score, "score")


@dataclass(frozen=True, kw_only=True)
class GeneTaskCandidate:
    """A candidate of `gene_task/v1`: one of the request's candidate genes,
    ranked by the finetuned gene head's score."""

    rank: int
    gene_id: NodeID
    gene_name: str
    #: η·EMBSIM + (1 − η)·SPLSIM, uncalibrated (README A8, U12).
    score: float
    #: Its two parts: the head's tanh similarity, and the fixed-bound SP
    #: similarity of the exact mean hop distance (README §6), unclipped.
    embsim: float
    sp_similarity: float
    mean_distance: float

    def __post_init__(self) -> None:
        for name in ("score", "embsim", "sp_similarity", "mean_distance"):
            _require_finite(getattr(self, name), name)


@dataclass(frozen=True, kw_only=True)
class PatientCandidate:
    """A candidate of `patients_like_me/v1`: a patient of the reference cohort
    version, by research id, with what the cohort recorded for it. No name or
    contact field exists (README §8)."""

    rank: int
    case_id: str
    #: −‖ẑ_q − ẑ_r‖² of the patient tower's normalised vectors, in [−4, 0];
    #: uncalibrated.
    similarity: float
    #: The recorded HPO terms, the recorded diagnosis and causal genes.
    phenotypes: Tuple[str, ...] = ()
    disease_id: Optional[NodeID] = None
    disease_name: Optional[str] = None
    causal_genes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_finite(self.similarity, "similarity")


#: A candidate of any kind the pipeline produces.
Candidate = Union[ServedMixtureCandidate, DiseaseTaskCandidate, GeneTaskCandidate,
                  PatientCandidate]

#: Each kind's candidate type.
CANDIDATE_TYPES = {SERVED_MIXTURE: ServedMixtureCandidate,
                   DISEASE_TASK: DiseaseTaskCandidate,
                   GENE_TASK: GeneTaskCandidate,
                   PATIENTS_LIKE_ME: PatientCandidate}


def candidate_score(candidate: Candidate) -> float:
    """The value `candidate`'s kind ranks by."""
    if isinstance(candidate, ServedMixtureCandidate):
        return candidate.served_score
    if isinstance(candidate, (DiseaseTaskCandidate, GeneTaskCandidate)):
        return candidate.score
    return candidate.similarity


@dataclass(frozen=True)
class PhenotypeInputAccount:
    """What happened to a request's phenotype list before it was scored.

    **The result says what was used, so nobody has to infer it from the request.**
    The pipeline drops unknown ids and removes repeats by graph node
    (`src/kg/phenotype_normalisation.py`); a caller that counted its own request
    afterwards would report repeats and unknown ids as though they were scored.
    """
    #: Entries the request carried, repeats and unknown ids included.
    received: int
    #: Ids the served model has no node for, in request order: none in the
    #: graph, or (a model bundle served) none in the model's graph view.
    #: Dropped, not scored.
    unknown: Tuple[str, ...]
    #: Entries that repeated a phenotype already kept.
    repeats_removed: int
    #: The phenotypes scored, in first-occurrence order. Empty when the request
    #: was refused, because then nothing was scored.
    used: Tuple[str, ...]


# =============================================================================
# Served identity (contract M3c, R8)
# =============================================================================
#: The states each part of a served identity can be in. Every part says which
#: one it is in words, so an absence is stated and never left as a blank or an
#: "unknown" for a reader to interpret. **A served model is always read from a
#: checkpoint over a workspace verified with its dataset record**, so those
#: three parts have one state; only the shortest-path term can be computed or
#: absent.
SERVED_CHECKPOINT_STATES = ("verified",)
SERVED_GRAPH_STATES = ("verified",)
SERVED_SHORTEST_PATHS_STATES = ("verified", "computed", "absent")
SERVED_DATASET_RECORD_STATES = ("verified",)
#: The dataset record's origins (`src.kg.provenance.ORIGINS`, which this layer
#: cannot import; `tests/unit/test_served_identity.py` pins the two equal).
SERVED_DATASET_ORIGINS = ("files", "synthetic")
SERVED_SCORING_MODES = ("gnn_plus_shortest_path", "gnn_only", "disease_task", "gene_task",
                        "patients_like_me")
#: **The one table of the scoring mode each finetuned task is served under**
#: (README §1: switching task means loading another model), and its inverse.
#: The pipeline names the mode of the model it serves from it, and every
#: reader of a served identity takes the task back from it; nobody keeps a copy.
#: Keyed by the model spec's tasks (`src.models.bundle.spec.TASKS`, which this
#: layer cannot import; `tests/unit/test_task_first_selection.py` pins the two
#: equal).
TASK_SCORING_MODES: Dict[str, str] = {"disease": "disease_task", "gene": "gene_task",
                                      "patients_like_me": "patients_like_me"}
SCORING_MODE_TASKS: Dict[str, str] = {mode: task for task, mode in TASK_SCORING_MODES.items()}
#: **The shortest-path state each scoring mode is served with**, the one rule:
#: `ServedIdentity` refuses any other, the pipeline builds its identity from it
#: and the WebUI's reload confirmation says it before a load. A verified table
#: for the GNN with its SP term; the gene task's exact hop distances computed
#: from the model's own graph view; no table for every other mode.
SCORING_MODE_SP_STATES: Dict[str, str] = {
    "gnn_plus_shortest_path": "verified", "gnn_only": "absent", "disease_task": "absent",
    "gene_task": "computed", "patients_like_me": "absent"}

_HEX = frozenset("0123456789abcdef")


def _require_sha256(value: Any, what: str) -> None:
    if not (isinstance(value, str) and len(value) == 64 and set(value) <= _HEX):
        raise ValueError(f"{what} must be a lowercase hexadecimal SHA-256, not {value!r}")


def _require_state(state: Any, allowed: Tuple[str, ...], what: str) -> None:
    if state not in allowed:
        raise ValueError(f"{what} state {state!r} is not one of {allowed}")


@dataclass(frozen=True)
class ServedCheckpoint:
    """The model's checkpoint."""

    #: "verified": the model was built from one read of a checkpoint whose
    #: record matched the served workspace's graph (contract M2.4); `sha256` is
    #: that read's digest.
    state: str
    sha256: str

    def __post_init__(self) -> None:
        _require_state(self.state, SERVED_CHECKPOINT_STATES, "checkpoint")
        _require_sha256(self.sha256, "a verified checkpoint's sha256")


@dataclass(frozen=True)
class ServedGraph:
    """The graph served, and the workspace manifest it was verified against."""

    #: "verified": `kg.json` and `split_manifest.json` were each read once and
    #: the graph's digest matched the manifest's binding; both digests are of
    #: those reads.
    state: str
    kg_sha256: str
    split_manifest_sha256: str

    def __post_init__(self) -> None:
        _require_state(self.state, SERVED_GRAPH_STATES, "graph")
        _require_sha256(self.kg_sha256, "a verified graph's kg_sha256")
        _require_sha256(self.split_manifest_sha256, "a verified graph's split_manifest_sha256")


@dataclass(frozen=True)
class ServedShortestPaths:
    """The shortest-path table, or the stated absence of one."""

    #: "verified": the table and its sidecar carry one `build_id` and name the
    #: served graph (contract M2.5); `max_hops` is the sidecar's hop bound.
    #: "computed": the gene task's exact hop distances, computed in process
    #: from the served graph view (README §6); no table, so no `build_id` or
    #: `max_hops`. "absent": no shortest-path term contributes to the score.
    state: str
    build_id: Optional[str] = None
    max_hops: Optional[int] = None

    def __post_init__(self) -> None:
        _require_state(self.state, SERVED_SHORTEST_PATHS_STATES, "shortest_paths")
        if self.state == "verified":
            if not (isinstance(self.build_id, str) and self.build_id):
                raise ValueError("a verified shortest-path table carries its build_id")
            if not (isinstance(self.max_hops, int) and not isinstance(self.max_hops, bool)):
                raise ValueError("a verified shortest-path table carries its max_hops")
        elif self.build_id is not None or self.max_hops is not None:
            raise ValueError(f"a shortest-path term in state {self.state!r} carries no "
                             "build_id or max_hops")


@dataclass(frozen=True)
class ServedDatasetRecord:
    """The served workspace's dataset record (contract M3b)."""

    #: "verified": the record bound by the workspace manifest, naming the graph
    #: served; `origin` is "files" or "synthetic".
    state: str
    origin: str
    sha256: str

    def __post_init__(self) -> None:
        _require_state(self.state, SERVED_DATASET_RECORD_STATES, "dataset_record")
        if self.origin not in SERVED_DATASET_ORIGINS:
            raise ValueError(f"a verified dataset record's origin is {self.origin!r}")
        _require_sha256(self.sha256, "a verified dataset record's sha256")


@dataclass(frozen=True)
class ServedSettings:
    """The serving settings that change scores, as in effect."""

    #: Where the model ran: torch's name for the device its parameters are on
    #: ("cuda:0", "cpu").
    device: str
    #: What was asked for: "auto" (CUDA, refused without it), "cuda", "cuda:N"
    #: or "cpu"; "auto" when a checkpoint was loaded with none named.
    device_requested: str
    #: The mixing weight in effect between the model and shortest-path terms:
    #: 1.0 with no table; the gene task's η (EMBSIM against SPLSIM, from its
    #: spec); None for the disease and patients-like-me tasks, which mix
    #: nothing.
    eta: Optional[float]
    #: Path discovery and path scoring: the diseases the mixture ranks are
    #: those reached by these paths, and a disease row's evidence paths are
    #: found with them.
    max_path_length: int
    path_length_penalty: float
    max_paths_per_candidate: int
    use_direct_path_optimization: bool
    #: The phenotype-list rule applied before scoring
    #: (`src/kg/phenotype_normalisation.py`).
    phenotype_normalisation_version: int


@dataclass(frozen=True)
class ServedIdentity:
    """What produced a diagnosis result: one structured value (contract M3c, R8).

    **It replaces the two version strings** `InferenceResult` used to carry,
    which were filled with a class constant and "unknown". Each part names what
    was served by the digest or identifier of the bytes read, or says in its
    `state` why there is none; nothing here is a version string to be parsed.

    Built by the pipeline from its own state when the result is produced, and
    reported by `get_pipeline_config()` from the same builder; the API copies
    it from the result it answers with.
    """

    scoring_mode: str
    checkpoint: ServedCheckpoint
    graph: ServedGraph
    shortest_paths: ServedShortestPaths
    dataset_record: ServedDatasetRecord
    settings: ServedSettings

    def __post_init__(self) -> None:
        _require_state(self.scoring_mode, SERVED_SCORING_MODES, "scoring_mode")
        if self.shortest_paths.state != SCORING_MODE_SP_STATES[self.scoring_mode]:
            raise ValueError(
                f"scoring_mode {self.scoring_mode!r} contradicts a shortest-path "
                f"table in state {self.shortest_paths.state!r}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """JSON primitives only, nested by part."""
        from dataclasses import asdict

        return asdict(self)


def result_kind_for(scoring_mode: str) -> str:
    """**The one mapping from what produced a result to its kind** (Spec 1 §2.1).

    `scoring_mode` is the served identity's (`ServedIdentity.scoring_mode`). A
    caller never chooses a kind: it is read off the producer."""
    if scoring_mode in ("gnn_plus_shortest_path", "gnn_only"):
        return SERVED_MIXTURE
    if scoring_mode == "disease_task":
        return DISEASE_TASK
    if scoring_mode == "gene_task":
        return GENE_TASK
    if scoring_mode == "patients_like_me":
        return PATIENTS_LIKE_ME
    raise ValueError(f"no result kind for scoring mode {scoring_mode!r}")


def result_label(result_kind: str, scoring_mode: Optional[str] = None,
                 eta: Optional[float] = None) -> str:
    """The kind's fixed label (Spec 1 §2.1), the one text every surface shows.

    For `served_mixture/v1`, η is the result's own, read from its served
    identity (`ServedSettings.eta`), never a constant; in `gnn_only` nothing is
    mixed, and the label says the model term ranks alone."""
    if result_kind == SERVED_MIXTURE:
        head = "Served ranking today: diseases reached by path discovery, "
        if scoring_mode == "gnn_only":
            return head + "scored by the model term alone"
        if eta is None:
            raise ValueError("a mixture's label names its η, and none was given")
        return head + f"scored as η × model term + (1 − η) × SP, η = {float(eta):g}"
    if result_kind == DISEASE_TASK:
        return ("Disease task model: every disease of the model's graph ranked by its "
                "Score (uncalibrated); paths only explain the rows shown")
    if result_kind == GENE_TASK:
        return ("Gene task model: your candidate genes ranked by their Score "
                "(uncalibrated), the model's similarity mixed with shortest-path proximity")
    if result_kind == PATIENTS_LIKE_ME:
        return ("Patients-like-me model: the reference cohort's patients ranked by "
                "Similarity (uncalibrated); a shared causal gene is not implied")
    raise ValueError(f"unknown result kind {result_kind!r}")


@dataclass
class InferenceResult:
    """A diagnosis result: one kind of the result union (Spec 1 §2.1).

    **The kind is read off what produced the result**, never passed in:
    `result_kind` follows `served_identity.scoring_mode` (`result_kind_for`), so
    a caller cannot attach one kind's label to another's numbers. The kind's
    result-level fields are derived the same way (`eta`, `tie_policy`), except
    `candidates_reached`, which only the ranking knows.

    **Its invariants are checked when it is built** (`__post_init__`): every
    candidate of the kind's type, ranks exactly 1..N, every score finite, the
    order non-increasing by the kind's score, and the kind's fields present and
    the others absent. Ties keep discovery order by construction (the stable
    sort of `DiagnosisPipeline._score_and_rank_candidates`); the order a tie came
    in is internal metadata (`ServedRanking.discovery_ordinals`), not on the
    result, so only the non-increasing order is checked here.
    """
    patient_id: str
    timestamp: datetime

    # Candidates in rank order, every one of the result's kind.
    candidates: List[Candidate]

    #: What produced this result (contract M3c). Required: every result the
    #: pipeline returns, a refused request's included, says what answered it.
    served_identity: ServedIdentity

    # Overall explanation
    summary_explanation: Optional[str] = None

    # Metadata
    inference_time_ms: float = 0.0

    # Warnings
    warnings: List[str] = field(default_factory=list)

    # The phenotype list as received and as used. None when the request was
    # refused before its phenotypes were mapped.
    phenotype_input: Optional[PhenotypeInputAccount] = None

    #: `served_mixture/v1` only: how many diseases discovery reached before the
    #: cut at `top_k` (0 for a refused or empty result). None for other kinds.
    candidates_reached: Optional[int] = None

    #: `patients_like_me/v1` only: the reference cohort version ranked against
    #: (`label`, `version_sha256`, `samples_sha256`, `patients`). None otherwise.
    reference_cohort: Optional[Dict[str, Any]] = None

    #: `gene_task/v1` only: the SP similarity's fixed bounds (lower, upper),
    #: the model spec's (README §6), so `sp_similarity` can be recomputed from
    #: `mean_distance`. None otherwise.
    sp_bounds: Optional[Tuple[float, float]] = None

    def __post_init__(self) -> None:
        kind = self.result_kind
        expected = CANDIDATE_TYPES[kind]
        for candidate in self.candidates:
            if type(candidate) is not expected:
                raise ValueError(
                    f"a {kind} result holds a {type(candidate).__name__}; every "
                    "candidate of a result has the result's kind")
        ranks = [candidate.rank for candidate in self.candidates]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError(f"ranks must be exactly 1..{len(ranks)}, got {ranks}")
        scores = [candidate_score(candidate) for candidate in self.candidates]
        if any(later > earlier for earlier, later in zip(scores, scores[1:])):
            raise ValueError("candidates are not in non-increasing order of their score")
        if kind == SERVED_MIXTURE:
            if isinstance(self.candidates_reached, bool) or not isinstance(
                    self.candidates_reached, int) or self.candidates_reached < len(ranks):
                raise ValueError(
                    "a served_mixture result says how many diseases discovery reached "
                    f"(at least the {len(ranks)} it holds), got {self.candidates_reached!r}")
            mixes = self.served_identity.scoring_mode == "gnn_plus_shortest_path"
            for candidate in self.candidates:
                record = candidate.sp_term
                if not mixes and record.status != "NO_TABLE":
                    raise ValueError("with no SP table served, every SP record is NO_TABLE")
                if mixes and record.status == "NO_TABLE":
                    raise ValueError("an SP table is served, so no record can be NO_TABLE")
                if (mixes and not record.numeric) != (record.mixed_as is not None):
                    raise ValueError(
                        "mixed_as is 0.0 exactly on a non-numeric status the mixture "
                        "mixed, and absent otherwise")
        elif self.candidates_reached is not None:
            raise ValueError(f"candidates_reached belongs to {SERVED_MIXTURE} alone")
        if (kind == PATIENTS_LIKE_ME) != (self.reference_cohort is not None):
            raise ValueError(f"a {PATIENTS_LIKE_ME} result names its reference cohort version, "
                             "and only such a result does")
        if (kind == GENE_TASK) != (self.sp_bounds is not None):
            raise ValueError(f"a {GENE_TASK} result names its SP bounds, and only such a "
                             "result does")

    @property
    def result_kind(self) -> str:
        return result_kind_for(self.served_identity.scoring_mode)

    @property
    def result_schema_version(self) -> int:
        return RESULT_SCHEMA_VERSION

    @property
    def eta(self) -> Optional[float]:
        """`served_mixture/v1`: the mixing weight in effect, the served
        identity's (1.0 in `gnn_only`, where nothing is mixed). None otherwise."""
        if self.result_kind != SERVED_MIXTURE:
            return None
        return self.served_identity.settings.eta

    @property
    def tie_policy(self) -> str:
        return {DISEASE_TASK: DISEASE_TASK_TIE_POLICY, GENE_TASK: GENE_TASK_TIE_POLICY,
                PATIENTS_LIKE_ME: PATIENTS_LIKE_ME_TIE_POLICY}.get(self.result_kind,
                                                                   SERVED_TIE_POLICY)

    @property
    def label(self) -> str:
        """The kind's fixed label, with this result's own η (`result_label`)."""
        return result_label(self.result_kind, self.served_identity.scoring_mode, self.eta)


# =============================================================================
# Result Wrapper
# =============================================================================
@dataclass
class Result(Generic[T]):
    """
    通用結果包裝器
    用於錯誤處理和狀態傳遞
    """
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, data: T, **metadata) -> "Result[T]":
        return cls(success=True, data=data, metadata=metadata)

    @classmethod
    def fail(cls, error: str, **metadata) -> "Result[T]":
        return cls(success=False, error=error, metadata=metadata)
