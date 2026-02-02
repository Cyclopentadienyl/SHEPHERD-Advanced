"""
SHEPHERD-Advanced Protocol Definitions
======================================
所有模組的接口契約 (Protocol/ABC)

設計原則:
1. 所有模組必須實現對應的 Protocol
2. Protocol 定義輸入/輸出類型，確保模組間相容
3. 使用 typing.Protocol 實現結構性子類型 (structural subtyping)

版本: 1.0.0
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)
from pathlib import Path
from dataclasses import dataclass

# Import core types
from src.core.types import (
    Node,
    Edge,
    NodeID,
    NodeType,
    EdgeType,
    EmbeddingType,
    PatientPhenotypes,
    DiagnosisCandidate,
    InferenceResult,
    Publication,
    LiteratureEvidence,
    OrthologMapping,
    OrthologGroup,
    EvidenceSource,
    DataSource,
    Species,
    ModelConfig,
    TrainingConfig,
    DataConfig,
    Result,
)

# Type variables
T = TypeVar("T")
GraphType = TypeVar("GraphType")  # For torch_geometric.data.HeteroData


# =============================================================================
# Ontology Module Protocols
# =============================================================================
@runtime_checkable
class OntologyLoaderProtocol(Protocol):
    """
    本體載入器協議

    實現模組: src/ontology/loader.py
    """

    def load(self, path: Path) -> "OntologyProtocol":
        """載入本體檔案 (OBO/OWL)"""
        ...

    def load_hpo(self, version: str = "latest") -> "OntologyProtocol":
        """載入 HPO (Human Phenotype Ontology)"""
        ...

    def load_mondo(self, version: str = "latest") -> "OntologyProtocol":
        """載入 MONDO (Disease Ontology)"""
        ...

    def load_go(self, version: str = "latest") -> "OntologyProtocol":
        """載入 GO (Gene Ontology)"""
        ...

    def load_mp(self, version: str = "latest") -> "OntologyProtocol":
        """載入 MP (Mammalian Phenotype Ontology) - 用於同源基因"""
        ...


@runtime_checkable
class OntologyProtocol(Protocol):
    """
    本體操作協議

    實現模組: src/ontology/hierarchy.py
    """

    @property
    def name(self) -> str:
        """本體名稱"""
        ...

    @property
    def version(self) -> str:
        """本體版本"""
        ...

    def get_term(self, term_id: str) -> Optional[Dict[str, Any]]:
        """獲取單個術語"""
        ...

    def get_ancestors(self, term_id: str, include_self: bool = False) -> Set[str]:
        """獲取所有祖先節點 (IS_A 關係)"""
        ...

    def get_descendants(self, term_id: str, include_self: bool = False) -> Set[str]:
        """獲取所有後代節點"""
        ...

    def get_parents(self, term_id: str) -> Set[str]:
        """獲取直接父節點"""
        ...

    def get_children(self, term_id: str) -> Set[str]:
        """獲取直接子節點"""
        ...

    def compute_similarity(
        self,
        term1: str,
        term2: str,
        method: str = "resnik",
    ) -> float:
        """
        計算兩個術語的語義相似度

        Args:
            method: resnik, lin, jiang, wang
        """
        ...

    def get_information_content(self, term_id: str) -> float:
        """獲取術語的 Information Content (IC)"""
        ...

    def search(
        self,
        query: str,
        max_results: int = 10,
        include_synonyms: bool = True,
    ) -> List[Tuple[str, str, float]]:
        """
        搜尋術語

        Returns:
            List of (term_id, term_name, score)
        """
        ...


@runtime_checkable
class OntologyConstraintProtocol(Protocol):
    """
    本體約束協議

    用於約束推理輸出，防止本體違規

    實現模組: src/ontology/constraints.py
    """

    def validate_phenotype_set(
        self,
        phenotypes: List[str],
    ) -> Tuple[bool, List[str]]:
        """
        驗證表型集合是否符合本體約束

        Returns:
            (is_valid, list of violations)
        """
        ...

    def remove_redundant_ancestors(
        self,
        phenotypes: List[str],
    ) -> List[str]:
        """移除冗餘的祖先表型"""
        ...

    def get_implied_phenotypes(
        self,
        phenotypes: List[str],
    ) -> Set[str]:
        """獲取隱含的表型 (基於本體推理)"""
        ...


# =============================================================================
# Knowledge Graph Module Protocols
# =============================================================================
@runtime_checkable
class KnowledgeGraphProtocol(Protocol):
    """
    知識圖譜操作協議

    實現模組: src/kg/
    """

    @property
    def num_nodes(self) -> Dict[NodeType, int]:
        """各類型節點數量"""
        ...

    @property
    def num_edges(self) -> Dict[EdgeType, int]:
        """各類型邊數量"""
        ...

    def add_node(self, node: Node) -> None:
        """添加節點"""
        ...

    def add_edge(self, edge: Edge) -> None:
        """添加邊"""
        ...

    def get_node(self, node_id: NodeID) -> Optional[Node]:
        """獲取節點"""
        ...

    def get_edges(
        self,
        source_id: Optional[NodeID] = None,
        target_id: Optional[NodeID] = None,
        edge_type: Optional[EdgeType] = None,
    ) -> List[Edge]:
        """查詢邊"""
        ...

    def get_neighbors(
        self,
        node_id: NodeID,
        edge_types: Optional[List[EdgeType]] = None,
        direction: str = "both",  # "in", "out", "both"
    ) -> List[Tuple[NodeID, EdgeType]]:
        """獲取鄰居節點"""
        ...

    def get_subgraph(
        self,
        node_ids: List[NodeID],
        num_hops: int = 1,
    ) -> "KnowledgeGraphProtocol":
        """提取子圖"""
        ...

    def to_pyg_hetero_data(self) -> GraphType:
        """轉換為 PyTorch Geometric HeteroData"""
        ...


@runtime_checkable
class KnowledgeGraphBuilderProtocol(Protocol):
    """
    知識圖譜建構協議

    實現模組: src/kg/builder.py
    """

    def add_ontology(
        self,
        ontology: OntologyProtocol,
        node_type: NodeType,
    ) -> None:
        """添加本體到知識圖譜"""
        ...

    def add_associations(
        self,
        associations: List[Tuple[str, str, EdgeType, Dict]],
        source: DataSource,
    ) -> None:
        """
        添加關聯邊

        Args:
            associations: List of (source_id, target_id, edge_type, attributes)
        """
        ...

    def add_orthologs(
        self,
        ortholog_mappings: List[OrthologMapping],
    ) -> None:
        """添加同源基因映射"""
        ...

    def add_literature_edges(
        self,
        publications: List[Publication],
    ) -> None:
        """添加文獻支持的邊"""
        ...

    def build(self) -> KnowledgeGraphProtocol:
        """建構知識圖譜"""
        ...

    def save(self, path: Path) -> None:
        """保存知識圖譜"""
        ...

    def load(self, path: Path) -> KnowledgeGraphProtocol:
        """載入知識圖譜"""
        ...


# =============================================================================
# Data Sources Module Protocols
# =============================================================================
@runtime_checkable
class DataSourceProtocol(Protocol):
    """
    資料來源協議 (基類)

    實現模組: src/data_sources/
    """

    @property
    def source_name(self) -> DataSource:
        """資料來源名稱"""
        ...

    @property
    def version(self) -> str:
        """資料版本"""
        ...

    def download(self, output_dir: Path) -> Path:
        """下載原始資料"""
        ...

    def parse(self, input_path: Path) -> Iterator[Dict[str, Any]]:
        """解析資料"""
        ...

    def get_gene_disease_associations(self) -> Iterator[Tuple[str, str, float, Dict]]:
        """
        獲取基因-疾病關聯

        Yields:
            (gene_id, disease_id, confidence, metadata)
        """
        ...


@runtime_checkable
class PubMedDataSourceProtocol(Protocol):
    """
    PubMed 資料來源協議

    用於文獻資料整合

    實現模組: src/data_sources/pubmed.py
    """

    def search(
        self,
        query: str,
        max_results: int = 1000,
    ) -> List[str]:
        """
        搜尋 PubMed (返回 PMID 列表)

        僅在 online 模式可用
        """
        ...

    def fetch_article(self, pmid: str) -> Optional[Publication]:
        """獲取文章詳情"""
        ...

    def batch_fetch_articles(self, pmids: List[str]) -> List[Publication]:
        """批量獲取文章"""
        ...

    def get_pubtator_annotations(
        self,
        pmid: str,
    ) -> Dict[str, List[str]]:
        """
        獲取 Pubtator NER 標註

        Returns:
            {"genes": [...], "diseases": [...], "chemicals": [...]}
        """
        ...

    def compute_credibility_score(self, publication: Publication) -> float:
        """
        計算文獻可信度評分

        考慮因素:
        - 期刊影響因子 (0-40%)
        - 證據等級 (0-30%)
        - 引用次數 (0-20%)
        - 作者機構 (0-10%)
        """
        ...

    def get_gene_disease_literature(
        self,
        gene_id: str,
        disease_id: str,
        min_credibility: float = 0.5,
    ) -> LiteratureEvidence:
        """獲取支持特定基因-疾病關聯的文獻證據"""
        ...


@runtime_checkable
class PubtatorLocalDBProtocol(Protocol):
    """
    Pubtator 本地資料庫協議 (離線模式)

    實現模組: src/data_sources/pubtator_local.py
    """

    def load_database(self, db_path: Path) -> None:
        """載入預下載的 Pubtator 資料庫"""
        ...

    def query_by_gene(self, gene_id: str) -> List[Publication]:
        """按基因查詢相關文獻"""
        ...

    def query_by_disease(self, disease_id: str) -> List[Publication]:
        """按疾病查詢相關文獻"""
        ...

    def query_gene_disease_pair(
        self,
        gene_id: str,
        disease_id: str,
    ) -> List[Publication]:
        """查詢同時提及基因和疾病的文獻"""
        ...


@runtime_checkable
class OrthologDataSourceProtocol(Protocol):
    """
    同源基因資料來源協議

    實現模組: src/data_sources/ortholog.py
    """

    def get_orthologs(
        self,
        human_gene_id: str,
        species: Species = Species.MOUSE,
    ) -> List[OrthologMapping]:
        """
        獲取人類基因的同源基因

        Args:
            human_gene_id: Human gene (HGNC symbol or Ensembl ID)
            species: Target species
        """
        ...

    def get_ortholog_groups(
        self,
        gene_ids: List[str],
    ) -> List[OrthologGroup]:
        """獲取基因所屬的同源基因群組"""
        ...

    def get_model_organism_phenotypes(
        self,
        ortholog_gene_id: str,
        species: Species,
    ) -> List[NodeID]:
        """
        獲取模式生物基因的表型

        例如: 小鼠基因 knockout 導致的 MP 表型
        """
        ...

    def map_model_phenotype_to_human(
        self,
        model_phenotype_id: str,
        source_species: Species,
    ) -> List[Tuple[str, float]]:
        """
        將模式生物表型映射到人類 HPO

        Returns:
            List of (hpo_id, confidence)
        """
        ...


# =============================================================================
# Model Module Protocols
# =============================================================================
@runtime_checkable
class NodeEncoderProtocol(Protocol):
    """
    節點編碼器協議

    實現模組: src/models/encoders/
    """

    def encode(
        self,
        node_features: Dict[NodeType, Any],
    ) -> Dict[NodeType, EmbeddingType]:
        """
        編碼節點特徵

        Args:
            node_features: {NodeType: tensor/array}

        Returns:
            {NodeType: embeddings}
        """
        ...

    @property
    def output_dim(self) -> int:
        """輸出嵌入維度"""
        ...


@runtime_checkable
class GNNProtocol(Protocol):
    """
    圖神經網路協議

    實現模組: src/models/gnn/
    """

    def forward(
        self,
        x_dict: Dict[str, Any],
        edge_index_dict: Dict[Tuple[str, str, str], Any],
        edge_attr_dict: Optional[Dict[Tuple[str, str, str], Any]] = None,
    ) -> Dict[str, Any]:
        """
        GNN 前向傳播

        Args:
            x_dict: Node features by type
            edge_index_dict: Edge indices by type
            edge_attr_dict: Edge attributes by type

        Returns:
            Updated node embeddings by type
        """
        ...

    @property
    def num_layers(self) -> int:
        """GNN 層數"""
        ...

    @property
    def hidden_dim(self) -> int:
        """隱藏層維度"""
        ...


@runtime_checkable
class AttentionBackendProtocol(Protocol):
    """
    注意力機制後端協議

    實現模組: src/models/attention/adaptive_backend.py
    """

    def compute_attention(
        self,
        query: Any,  # torch.Tensor
        key: Any,
        value: Any,
        attn_mask: Optional[Any] = None,
    ) -> Any:
        """計算注意力"""
        ...

    @property
    def backend_name(self) -> str:
        """當前使用的後端名稱"""
        ...

    def available_backends(self) -> List[str]:
        """可用的後端列表"""
        ...


@runtime_checkable
class DecoderProtocol(Protocol):
    """
    解碼器協議

    實現模組: src/models/decoders/
    """

    def decode(
        self,
        patient_embedding: EmbeddingType,
        disease_embeddings: Dict[str, EmbeddingType],
    ) -> Dict[str, float]:
        """
        解碼預測疾病分數

        Returns:
            {disease_id: score}
        """
        ...


@runtime_checkable
class DiagnosisModelProtocol(Protocol):
    """
    完整診斷模型協議

    整合 Encoder + GNN + Decoder

    實現模組: src/models/tasks/diagnosis.py
    """

    def predict(
        self,
        patient_phenotypes: PatientPhenotypes,
        kg: KnowledgeGraphProtocol,
        top_k: int = 10,
    ) -> List[DiagnosisCandidate]:
        """
        預測疾病候選

        Args:
            patient_phenotypes: 患者表型輸入
            kg: 知識圖譜
            top_k: 返回前 k 個候選

        Returns:
            排序的疾病候選列表
        """
        ...

    def get_patient_embedding(
        self,
        patient_phenotypes: PatientPhenotypes,
        kg: KnowledgeGraphProtocol,
    ) -> EmbeddingType:
        """獲取患者表型的嵌入表示"""
        ...

    def save(self, path: Path) -> None:
        """保存模型"""
        ...

    def load(self, path: Path) -> None:
        """載入模型"""
        ...


# =============================================================================
# Reasoning Module Protocols
# =============================================================================
@runtime_checkable
class PathReasonerProtocol(Protocol):
    """
    路徑推理協議 (DR.KNOWS style)

    實現模組: src/reasoning/path_reasoning.py
    """

    def find_paths(
        self,
        source_ids: List[NodeID],
        target_type: NodeType,
        kg: KnowledgeGraphProtocol,
        max_length: int = 4,
        top_k: int = 100,
    ) -> List[List[NodeID]]:
        """
        尋找從源節點到目標類型的路徑

        Args:
            source_ids: 源節點 (e.g., patient phenotypes)
            target_type: 目標節點類型 (e.g., DISEASE)
            max_length: 最大路徑長度

        Returns:
            找到的路徑列表
        """
        ...

    def score_paths(
        self,
        paths: List[List[NodeID]],
        kg: KnowledgeGraphProtocol,
    ) -> List[Tuple[List[NodeID], float]]:
        """
        對路徑評分

        Returns:
            (path, score) pairs
        """
        ...

    def aggregate_path_scores(
        self,
        scored_paths: List[Tuple[List[NodeID], float]],
        target_ids: List[NodeID],
    ) -> Dict[NodeID, float]:
        """
        聚合到同一目標的路徑分數

        Returns:
            {target_id: aggregated_score}
        """
        ...


@runtime_checkable
class OrthologReasonerProtocol(Protocol):
    """
    同源基因推理協議

    用於跨物種的疾病關聯推理

    實現模組: src/reasoning/ortholog_reasoning.py
    """

    def infer_from_orthologs(
        self,
        human_gene_id: str,
        kg: KnowledgeGraphProtocol,
        ortholog_source: OrthologDataSourceProtocol,
    ) -> List[Tuple[NodeID, float, List[EvidenceSource]]]:
        """
        基於同源基因推理可能的疾病關聯

        Returns:
            List of (disease_id, confidence, evidence)
        """
        ...

    def get_ortholog_evidence_chain(
        self,
        human_gene_id: str,
        disease_id: str,
        kg: KnowledgeGraphProtocol,
    ) -> Optional[Dict[str, Any]]:
        """
        獲取同源基因證據鏈

        Returns:
            {
                "human_gene": ...,
                "ortholog_gene": ...,
                "ortholog_species": ...,
                "model_phenotypes": [...],
                "mapped_human_phenotypes": [...],
                "confidence": ...
            }
        """
        ...


@runtime_checkable
class ConstraintCheckerProtocol(Protocol):
    """
    約束檢查協議

    用於驗證推理結果是否符合本體約束

    實現模組: src/reasoning/constraint_checker.py
    """

    def check_prediction(
        self,
        patient_phenotypes: List[str],
        predicted_disease: str,
        kg: KnowledgeGraphProtocol,
        ontology: OntologyProtocol,
    ) -> Tuple[bool, List[str], float]:
        """
        檢查預測是否符合約束

        Returns:
            (is_valid, violations, penalty_score)
        """
        ...

    def apply_constraints(
        self,
        candidates: List[DiagnosisCandidate],
        patient_phenotypes: List[str],
        kg: KnowledgeGraphProtocol,
        ontology: OntologyProtocol,
    ) -> List[DiagnosisCandidate]:
        """應用約束調整候選分數"""
        ...


@runtime_checkable
class ExplanationGeneratorProtocol(Protocol):
    """
    解釋生成協議

    實現模組: src/reasoning/explanation_generator.py
    """

    def generate_explanation(
        self,
        candidate: DiagnosisCandidate,
        patient_phenotypes: PatientPhenotypes,
        kg: KnowledgeGraphProtocol,
        include_ortholog_evidence: bool = True,
        include_literature_evidence: bool = True,
    ) -> str:
        """
        生成診斷解釋

        Returns:
            Human-readable explanation
        """
        ...

    def generate_evidence_summary(
        self,
        candidate: DiagnosisCandidate,
        kg: KnowledgeGraphProtocol,
    ) -> Dict[str, Any]:
        """
        生成證據摘要

        Returns:
            {
                "direct_evidence": [...],
                "pathway_evidence": [...],
                "ortholog_evidence": [...],
                "literature_evidence": [...],
            }
        """
        ...


# =============================================================================
# Retrieval Module Protocols
# =============================================================================
@runtime_checkable
class VectorIndexProtocol(Protocol):
    """
    向量索引協議

    實現模組: src/retrieval/vector_index.py
    """

    def build_index(
        self,
        embeddings: Dict[str, EmbeddingType],
    ) -> None:
        """建立索引"""
        ...

    def search(
        self,
        query: EmbeddingType,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        向量搜尋

        Returns:
            List of (id, distance)
        """
        ...

    def batch_search(
        self,
        queries: List[EmbeddingType],
        top_k: int = 10,
    ) -> List[List[Tuple[str, float]]]:
        """批量搜尋"""
        ...

    def save(self, path: Path) -> None:
        """保存索引"""
        ...

    def load(self, path: Path) -> None:
        """載入索引"""
        ...

    @property
    def backend_name(self) -> str:
        """Index backend name (cuvs/voyager)"""
        ...


@runtime_checkable
class SubgraphSamplerProtocol(Protocol):
    """
    子圖採樣協議

    實現模組: src/retrieval/subgraph_sampler.py
    """

    def sample_subgraph(
        self,
        seed_nodes: List[NodeID],
        kg: KnowledgeGraphProtocol,
        num_hops: int = 2,
        max_nodes_per_hop: int = 50,
    ) -> KnowledgeGraphProtocol:
        """採樣子圖用於推理"""
        ...


# =============================================================================
# LLM Module Protocols
# =============================================================================
@runtime_checkable
class LLMProtocol(Protocol):
    """
    LLM 協議

    實現模組: src/llm/
    """

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """生成文本"""
        ...

    def batch_generate(
        self,
        prompts: List[str],
        **kwargs,
    ) -> List[str]:
        """批量生成"""
        ...

    @property
    def model_name(self) -> str:
        """模型名稱"""
        ...

    @property
    def backend(self) -> str:
        """後端類型 (vllm/llama_cpp/transformers)"""
        ...


@runtime_checkable
class MedicalLLMProtocol(LLMProtocol, Protocol):
    """
    醫療 LLM 協議 (擴展基礎 LLM)

    實現模組: src/llm/medical_llm.py
    """

    def generate_clinical_explanation(
        self,
        candidate: DiagnosisCandidate,
        patient_context: str,
    ) -> str:
        """生成臨床解釋"""
        ...

    def summarize_evidence(
        self,
        evidence_sources: List[EvidenceSource],
    ) -> str:
        """總結證據"""
        ...

    def extract_clinical_features(
        self,
        free_text: str,
    ) -> List[Dict[str, Any]]:
        """從自由文字提取臨床特徵"""
        ...


# =============================================================================
# NLP Module Protocols
# =============================================================================
@runtime_checkable
class SymptomExtractorProtocol(Protocol):
    """
    症狀提取協議

    實現模組: src/nlp/symptom_extractor.py
    """

    def extract(
        self,
        text: str,
        confidence_threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        從文字提取症狀

        Returns:
            List of {"text": str, "hpo_id": str, "confidence": float}
        """
        ...

    def batch_extract(
        self,
        texts: List[str],
        **kwargs,
    ) -> List[List[Dict[str, Any]]]:
        """批量提取"""
        ...


@runtime_checkable
class HPOMatcherProtocol(Protocol):
    """
    HPO 術語匹配協議

    實現模組: src/nlp/hpo_matcher.py
    """

    def match(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Tuple[str, str, float]]:
        """
        模糊匹配 HPO 術語

        Returns:
            List of (hpo_id, hpo_name, score)
        """
        ...

    def batch_match(
        self,
        queries: List[str],
        **kwargs,
    ) -> List[List[Tuple[str, str, float]]]:
        """批量匹配"""
        ...


# =============================================================================
# Inference Module Protocols
# =============================================================================
@runtime_checkable
class InferencePipelineProtocol(Protocol):
    """
    推理管線協議

    整合所有模組的完整推理流程

    實現模組: src/inference/pipeline.py
    """

    def run(
        self,
        patient_input: PatientPhenotypes,
        top_k: int = 10,
        include_explanations: bool = True,
        include_ortholog_evidence: bool = True,
        include_literature_evidence: bool = True,
    ) -> InferenceResult:
        """
        執行完整推理

        Args:
            patient_input: 患者輸入
            top_k: 返回前 k 個候選
            include_explanations: 是否生成解釋
            include_ortholog_evidence: 是否包含同源基因證據
            include_literature_evidence: 是否包含文獻證據

        Returns:
            完整推理結果
        """
        ...

    def validate_input(
        self,
        patient_input: PatientPhenotypes,
    ) -> Result[PatientPhenotypes]:
        """驗證輸入"""
        ...

    def get_pipeline_config(self) -> Dict[str, Any]:
        """獲取管線配置"""
        ...


@runtime_checkable
class InputValidatorProtocol(Protocol):
    """
    輸入驗證協議

    實現模組: src/inference/input_validator.py
    """

    def validate_phenotypes(
        self,
        phenotypes: List[str],
        ontology: OntologyProtocol,
    ) -> Result[List[str]]:
        """驗證 HPO 表型"""
        ...

    def validate_patient_input(
        self,
        patient_input: Dict[str, Any],
    ) -> Result[PatientPhenotypes]:
        """驗證完整患者輸入"""
        ...


@runtime_checkable
class OutputFormatterProtocol(Protocol):
    """
    輸出格式化協議

    實現模組: src/inference/output_formatter.py
    """

    def format_result(
        self,
        result: InferenceResult,
        format: str = "json",  # json, fhir, html
    ) -> str:
        """格式化輸出"""
        ...

    def to_fhir_diagnostic_report(
        self,
        result: InferenceResult,
    ) -> Dict[str, Any]:
        """轉換為 FHIR DiagnosticReport"""
        ...


# =============================================================================
# Training Module Protocols
# =============================================================================
@runtime_checkable
class TrainerProtocol(Protocol):
    """
    訓練器協議

    實現模組: src/training/trainer.py
    """

    def train(
        self,
        model: DiagnosisModelProtocol,
        train_data: Any,
        val_data: Optional[Any] = None,
        config: Optional[TrainingConfig] = None,
    ) -> Dict[str, Any]:
        """
        訓練模型

        Returns:
            Training history/metrics
        """
        ...

    def evaluate(
        self,
        model: DiagnosisModelProtocol,
        test_data: Any,
    ) -> Dict[str, float]:
        """
        評估模型

        Returns:
            Evaluation metrics
        """
        ...

    def save_checkpoint(
        self,
        model: DiagnosisModelProtocol,
        path: Path,
        metadata: Optional[Dict] = None,
    ) -> None:
        """保存檢查點"""
        ...

    def load_checkpoint(
        self,
        path: Path,
    ) -> Tuple[DiagnosisModelProtocol, Dict]:
        """載入檢查點"""
        ...


# =============================================================================
# API Module Protocols
# =============================================================================
@runtime_checkable
class APIServiceProtocol(Protocol):
    """
    API 服務協議

    實現模組: src/api/
    """

    async def diagnose(
        self,
        patient_input: Dict[str, Any],
    ) -> Dict[str, Any]:
        """診斷 API"""
        ...

    async def search_hpo(
        self,
        query: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """HPO 搜尋 API"""
        ...

    async def get_disease_info(
        self,
        disease_id: str,
    ) -> Dict[str, Any]:
        """疾病資訊 API"""
        ...

    async def health_check(self) -> Dict[str, Any]:
        """健康檢查"""
        ...


# =============================================================================
# Medical Standards Module Protocols
# =============================================================================
@runtime_checkable
class FHIRAdapterProtocol(Protocol):
    """
    FHIR 適配器協議

    實現模組: src/medical_standards/fhir_adapter.py
    """

    def parse_bundle(
        self,
        fhir_bundle: Dict[str, Any],
    ) -> PatientPhenotypes:
        """解析 FHIR Bundle"""
        ...

    def export_to_fhir(
        self,
        result: InferenceResult,
    ) -> Dict[str, Any]:
        """導出為 FHIR 格式"""
        ...


@runtime_checkable
class MedicalCodeMapperProtocol(Protocol):
    """
    醫療編碼映射協議

    實現模組: src/medical_standards/icd_mapper.py
    """

    def map_to_hpo(
        self,
        code: str,
        system: str,
    ) -> List[Tuple[str, float]]:
        """
        映射到 HPO

        Args:
            code: Source code (e.g., "G71.0")
            system: Code system (e.g., "icd10")

        Returns:
            List of (hpo_id, confidence)
        """
        ...

    def map_to_mondo(
        self,
        code: str,
        system: str,
    ) -> List[Tuple[str, float]]:
        """映射到 MONDO"""
        ...

    def reverse_map(
        self,
        hpo_id: str,
        target_system: str,
    ) -> List[Tuple[str, float]]:
        """反向映射"""
        ...


# =============================================================================
# Config Module Protocols
# =============================================================================
@runtime_checkable
class ConfigLoaderProtocol(Protocol):
    """
    配置載入協議

    實現模組: src/config/
    """

    def load(self, config_path: Path) -> Dict[str, Any]:
        """載入配置"""
        ...

    def validate(self, config: Dict[str, Any]) -> Result[Dict[str, Any]]:
        """驗證配置"""
        ...

    def get_model_config(self) -> ModelConfig:
        """獲取模型配置"""
        ...

    def get_training_config(self) -> TrainingConfig:
        """獲取訓練配置"""
        ...

    def get_data_config(self) -> DataConfig:
        """獲取資料配置"""
        ...
