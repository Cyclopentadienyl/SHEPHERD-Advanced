# 醫療知識圖譜診斷推理引擎 - 工程藍圖 v2.0（升級版）

**版本**: v2.0 - 整合2024-2025前沿技術  
**最後更新**: 2025-10-07  
**重大變更**: ✅ 分層本體架構 | ✅ 知識超圖 | ✅ 跨平台兼容

---

## 專案概述

### 核心目標
構建**臨床級精準度**的醫療知識圖譜與罕見疾病診斷推理引擎，整合最新的分層本體感知技術、知識超圖和路徑推理，實現：
- **精準度**: Hits@10 ≥ 80% (vs SHEPHERD 60%)
- **可靠性**: 幻覺率 < 5% (vs 傳統20%)
- **可解釋性**: 完整的推理路徑與證據鏈
- **跨平台**: Windows x86 + ARM (DGX Spark) 統一支持

### 技術棧升級

#### 核心框架
- **PyTorch**: 2.8+ (ARM + CUDA 12.8 原生支持)
- **PyTorch Geometric**: 2.6+ (異質圖處理)
- **注意力加速**: FlashAttention-2 (x86) / xformers (ARM) / PyTorch SDPA (通用)

#### 創新技術（2024-2025）
- **分層本體感知 GNN**: 整合HPO/MONDO/GO層次結構
- **知識超圖**: 處理多元高階關係（≥3實體）
- **DR.KNOWS路徑推理**: 知識圖譜路徑檢索 + LLM推理
- **Neural ODE**: 連續時間患者狀態建模
- **本體約束推理**: 防止不合理預測

#### 檢索與生成
- **向量索引**: cuVS (Linux GPU) / Voyager (跨平台 CPU)
- **GraphRAG**: 知識圖譜增強檢索生成
- **LLM整合**: GPT-4/Claude (可選) 用於證據解釋

### 運行環境

#### 環境一：開發訓練環境
```yaml
硬體:
  OS: Windows 11 (支持WSL2)
  CPU: x86-64
  GPU: NVIDIA Blackwell @ 16GB VRAM
  RAM: 32GB+ (建議64GB)
  儲存: 1TB NVMe SSD

軟體:
  Python: 3.12
  CUDA: 12.8
  PyTorch: 2.8.0+cu128
  注意力: FlashAttention-2 ✅
```

#### 環境二：邊緣推理環境
```yaml
硬體:
  系統: NVIDIA DGX Spark (GB10 SoC)
  CPU: 20核 ARM v9.2
    - 10x Cortex-X925 (高性能)
    - 10x Cortex-A725 (效能)
  GPU: Blackwell (集成)
    - 6144 CUDA核心
    - 1 PFLOPS @ FP4
  記憶體: 128GB LPDDR5X-9400 (統一)
  網絡: 2x 200GbE ConnectX-7
  功耗: 140W

軟體:
  OS: NVIDIA DGX OS (Ubuntu-based)
  Python: 3.12
  CUDA: 12.8
  PyTorch: 2.8.0+cu128 (ARM build)
  注意力: PyTorch SDPA / xformers (降級)
```

**關鍵差異：**
- ✅ DGX Spark: 128GB統一記憶體（優勢）
- ⚠️ DGX Spark: 部分套件需特殊處理（挑戰）
- ⚠️ Windows: 16GB VRAM限制（需子圖採樣）

---

## 系統架構（5+1層設計）

### 第0層：本體知識層 🧬 **【新增】**

**核心創新：將Ontology從配角提升為核心**

```
┌──────────────────────────────────────────┐
│  分層醫療本體知識庫                       │
├──────────────────────────────────────────┤
│  - 疾病本體: MONDO + Orphanet (IS-A層次) │
│  - 表型本體: HPO (症狀分類樹)             │
│  - 功能本體: GO + Reactome (生物通路)     │
│  - 疾病共現圖: 從EHR統計構建              │
├──────────────────────────────────────────┤
│  本體推理引擎:                            │
│  - 層次約束 (父子關係)                    │
│  - 互斥規則 (不可共存症狀)                │
│  - 蘊含規則 (必然伴隨症狀)                │
│  - 相似度計算 (本體距離)                  │
└──────────────────────────────────────────┘
```

**實現：**
```python
class OntologyKnowledgeBase:
    """
    統一的本體知識管理
    """
    def __init__(self):
        # 載入本體
        self.hpo = load_ontology('HPO')      # 130,000+ 表型
        self.mondo = load_ontology('MONDO')  # 60,000+ 疾病
        self.go = load_ontology('GO')        # 44,000+ 功能
        
        # 構建層次索引
        self.hierarchy_index = HierarchyIndex([self.hpo, self.mondo, self.go])
        
        # 疾病共現圖
        self.cooccurrence = DiseaseCooccurrenceGraph.from_ehr(
            sources=['MIMIC-III', 'UK-Biobank', 'ChinaMap'],
            min_support=10
        )
        
        # 約束規則
        self.constraints = OntologyConstraints(
            mutex_rules=self.extract_mutex_rules(),
            implication_rules=self.extract_implication_rules()
        )
    
    def validate_phenotype_set(self, phenotypes):
        """驗證症狀集合的本體一致性"""
        return self.constraints.check(phenotypes)
```

### 第1層：資料層 (Data Layer)

#### 1.1 多源異質資料整合

**資料來源（優先級排序）：**

| 優先級 | 資料源 | 更新頻率 | 用途 | 規模 |
|--------|--------|----------|------|------|
| 🔴 P0 | **HPO** | 季度 | 表型本體 | 130K+ terms |
| 🔴 P0 | **MONDO** | 季度 | 疾病本體 | 60K+ diseases |
| 🔴 P0 | **OMIM** | 月度 | 基因-疾病 | 26K+ entries |
| 🔴 P0 | **DisGeNET** | 季度 | 基因-疾病關聯 | 1.1M+ associations |
| 🟠 P1 | **Orphanet** | 年度 | 罕見疾病 | 6K+ rare diseases |
| 🟠 P1 | **ClinVar** | 週度 | 變異-疾病 | 2M+ variants |
| 🟠 P1 | **GO** | 月度 | 基因功能 | 44K+ terms |
| 🟠 P1 | **Reactome** | 季度 | 生物通路 | 2.5K+ pathways |
| 🟡 P2 | **Pubtator 3.0** | 即時 | 文獻證據 | 30M+ relations |
| 🟡 P2 | **NCBI Gene** | 週度 | 基因資訊 | 60K+ genes |

#### 1.2 知識圖譜構建（異質+超圖）

**傳統異質圖：**
```python
HeteroData = {
    'gene': {nodes: 20,000},
    'disease': {nodes: 15,000},
    'phenotype': {nodes: 50,000},
    'pathway': {nodes: 2,500},
    
    ('gene', 'associated_with', 'disease'): {edges: 120,000},
    ('phenotype', 'indicates', 'disease'): {edges: 200,000},
    ('gene', 'in_pathway', 'pathway'): {edges: 80,000}
}
```

**新增：知識超圖** 🆕
```python
class MedicalKnowledgeHypergraph:
    """
    處理高階關係（>2個實體）
    """
    def __init__(self):
        self.hyperedges = {
            # 超邊ID: {節點集合, 關係類型, 置信度, 證據}
            'he_001': {
                'nodes': ['BRCA1', 'BRCA2', 'family_history'],
                'relation': 'breast_cancer_risk',
                'confidence': 0.92,
                'evidence': ['PMID:12345', 'ClinVar:678']
            },
            'he_002': {
                'nodes': ['obesity', 'hypertension', 'diabetes'],
                'relation': 'metabolic_syndrome',
                'confidence': 0.88,
                'evidence': ['PMID:98765']
            }
        }
    
    def detect_hyperedges_from_ehr(self, ehr_data):
        """
        從電子病歷自動發現高階關係
        使用頻繁模式挖掘
        """
        # Apriori算法尋找頻繁症狀組合
        frequent_sets = apriori(
            ehr_data,
            min_support=0.01,
            max_len=5  # 最多5個實體
        )
        
        # 轉換為超邊
        for item_set in frequent_sets:
            if len(item_set) >= 3:  # 至少3個實體
                self.add_hyperedge(
                    nodes=list(item_set),
                    confidence=self.compute_confidence(item_set)
                )
```

#### 1.3 知識圖譜存儲（混合策略）

```python
class HybridKGStorage:
    """
    混合存儲策略：文件 + 圖資料庫 + 向量索引
    """
    def __init__(self):
        # 1. 訓練用：PyG Data對象（快速載入）
        self.pyg_data = HeteroData()
        
        # 2. 推理用：Neo4j（複雜查詢）
        self.graph_db = Neo4jInterface(
            uri="bolt://localhost:7687"
        )
        
        # 3. 檢索用：向量索引 (auto-select: cuVS/Voyager)
        from src.retrieval import create_index
        self.vector_index = create_index(
            backend='auto',
            dim=512
        )
    
    def query_subgraph(self, patient_phenotypes, k_hop=2):
        """
        為患者檢索相關子圖
        """
        # 1. 向量相似度快速篩選
        similar_nodes = self.vector_index.search(
            patient_phenotypes,
            k=100
        )
        
        # 2. 圖資料庫擴展鄰居
        subgraph = self.graph_db.expand_neighbors(
            seed_nodes=similar_nodes,
            max_hops=k_hop,
            max_nodes=10000
        )
        
        # 3. 轉換為PyG格式
        return self.to_pyg_data(subgraph)
```

### 第2層：模型層 (Model Layer) - 重大升級 🚀

#### 2.1 分層本體感知圖編碼器 **【核心創新】**

**架構設計：**

```
輸入節點
    ↓
┌─────────────────────────────────────┐
│ 第1階段：本體層次編碼               │
│ - 祖先節點嵌入                      │
│ - 兄弟節點嵌入                      │
│ - 層次位置編碼                      │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 第2階段：疾病共現增強               │
│ - 統計共現權重                      │
│ - 臨床語義相似度                    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 第3階段：圖結構學習                 │
│ - GraphGPS / Graph Transformer      │
│ - 本體引導注意力                    │
│ - 超圖卷積（處理高階關係）          │
└─────────────────────────────────────┘
    ↓
增強節點嵌入
```

**實現：**
```python
class HierarchicalOntologyAwareGNN(nn.Module):
    """
    整合本體知識的圖神經網路
    """
    def __init__(self, ontology_kb, hidden_dim=512):
        super().__init__()
        self.ontology_kb = ontology_kb
        
        # 階段1：本體編碼器
        self.ontology_encoder = OntologyHierarchyEncoder(
            ontologies=[ontology_kb.hpo, ontology_kb.mondo, ontology_kb.go],
            embedding_dim=hidden_dim,
            encode_ancestors=True,
            encode_siblings=True,
            max_depth=10
        )
        
        # 階段2：共現增強
        self.cooccurrence_layer = CooccurrenceAggregation(
            cooccurrence_graph=ontology_kb.cooccurrence,
            hidden_dim=hidden_dim
        )
        
        # 階段3：圖結構學習
        self.gnn_layers = nn.ModuleList([
            OntologyGuidedGraphGPS(
                hidden_dim=hidden_dim,
                num_heads=8,
                ontology_tree=ontology_kb.hierarchy_index
            )
            for _ in range(6)
        ])
        
        # 超圖卷積（處理高階關係）
        self.hypergraph_conv = HypergraphConv(hidden_dim)
    
    def forward(self, hetero_data, hypergraph_data):
        # 階段1：本體編碼
        x = self.ontology_encoder(hetero_data.x, hetero_data.node_type)
        
        # 階段2：共現增強
        x = self.cooccurrence_layer(x, hetero_data.node_id)
        
        # 階段3：圖學習
        for gnn in self.gnn_layers:
            x = gnn(x, hetero_data.edge_index, hetero_data.edge_type)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
        
        # 超圖卷積
        if hypergraph_data is not None:
            x = self.hypergraph_conv(x, hypergraph_data.hyperedges)
        
        return x
```

**關鍵組件：本體引導注意力**
```python
class OntologyGuidedGraphGPS(nn.Module):
    """
    GraphGPS + 本體引導注意力
    """
    def __init__(self, hidden_dim, num_heads, ontology_tree):
        super().__init__()
        self.ontology_tree = ontology_tree
        
        # 本地消息傳遞（MPNN）
        self.local_mpnn = GATConv(hidden_dim, hidden_dim, heads=num_heads)
        
        # 全局注意力（Transformer）
        self.global_attn = OntologyGuidedAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            ontology_tree=ontology_tree
        )
        
        # 注意力加速
        self.attn_backend = AdaptiveAttentionBackend()
    
    def forward(self, x, edge_index, edge_type):
        # 本地：鄰居聚合
        local_out = self.local_mpnn(x, edge_index)
        
        # 全局：本體引導的注意力
        global_out = self.global_attn(x, use_backend=self.attn_backend)
        
        # 融合
        return local_out + global_out
```

#### 2.2 DR.KNOWS式路徑推理模組 **【新增】**

**核心思想：從知識圖譜提取推理路徑作為證據**

```python
class DRKNOWSPathReasoning(nn.Module):
    """
    知識圖譜路徑檢索 + 排序 + LLM解釋
    """
    def __init__(self, kg, llm_model=None):
        super().__init__()
        self.kg = kg
        self.llm = llm_model  # 可選：用於生成解釋
        
        # 路徑檢索器
        self.path_retriever = MultiHopPathRetriever(
            kg=kg,
            max_hops=3,
            max_paths=50
        )
        
        # 路徑評分器
        self.path_scorer = PathScorer(
            scoring_method='structural_semantic',
            use_attention=True
        )
        
        # 路徑到文本
        self.path_to_text = PathToTextConverter()
    
    def retrieve_and_explain(self, patient_phenotypes, top_k=10):
        """
        檢索推理路徑並生成解釋
        """
        # 1. 檢索路徑
        paths = self.path_retriever.retrieve(
            start_nodes=patient_phenotypes,
            target_types=['gene', 'disease']
        )
        
        # 2. 評分與排序
        scored_paths = self.path_scorer.score(paths)
        top_paths = sorted(scored_paths, key=lambda x: x['score'], reverse=True)[:top_k]
        
        # 3. 多樣化（避免冗餘）
        diverse_paths = self.diversify_paths(top_paths, threshold=0.7)
        
        # 4. 轉換為文本（可選：用於LLM）
        path_texts = [self.path_to_text(p) for p in diverse_paths]
        
        # 5. LLM生成解釋（可選）
        if self.llm:
            explanation = self.llm.generate(
                prompt=f"Based on these diagnostic paths, explain the diagnosis:\n{path_texts}",
                max_tokens=500
            )
        else:
            explanation = self.generate_template_explanation(diverse_paths)
        
        return {
            'paths': diverse_paths,
            'explanation': explanation,
            'evidence': self.extract_evidence(diverse_paths)
        }
```

**路徑範例：**
```
症狀: 肌肉無力 (HP:0001324)
  ↓ phenotype_to_gene (confidence: 0.85)
基因: DMD (突變)
  ↓ gene_to_disease (confidence: 0.92)
疾病: 杜氏肌肉營養不良症 (MONDO:0010679)

證據鏈:
- ClinVar: rs123456 (致病性: Pathogenic)
- OMIM: #310200
- 文獻: PMID:12345678 (150例病例研究)
```

#### 2.3 患者嵌入與時序建模

**傳統方法的問題：**
```python
# 簡單聚合 - 丟失時序信息
patient_embedding = mean(symptom_embeddings)
```

**升級方案：Neural ODE + Transformer**
```python
class TemporalPatientEncoder(nn.Module):
    """
    整合時序信息的患者編碼器
    """
    def __init__(self):
        # 單次就診編碼器
        self.visit_encoder = VisitEncoder(
            input_dim=512,
            hidden_dim=256
        )
        
        # Neural ODE（處理不規則時間間隔）
        self.ode_func = ODEFunc(hidden_dim=256)
        
        # Time-aware Transformer
        self.temporal_transformer = TimeAwareTransformer(
            hidden_dim=256,
            num_layers=4,
            num_heads=8
        )
    
    def forward(self, patient_history):
        """
        patient_history: [
            {'time': t1, 'phenotypes': [...]},
            {'time': t2, 'phenotypes': [...]},
            ...
        ]
        """
        # 1. 編碼每次就診
        visit_embeddings = [
            self.visit_encoder(visit['phenotypes'])
            for visit in patient_history
        ]
        
        # 2. Neural ODE模擬連續演化
        time_stamps = [visit['time'] for visit in patient_history]
        ode_states = odeint(
            self.ode_func,
            visit_embeddings[0],
            time_stamps,
            method='dopri5'
        )
        
        # 3. Transformer捕捉長期依賴
        final_state = self.temporal_transformer(
            ode_states,
            time_stamps
        )
        
        return final_state
```

#### 2.4 本體約束解碼器 **【新增】**

**傳統解碼器的問題：**
```python
# DistMult: 可能產生不合理預測
score = (h * r * t).sum()  # 無約束
```

**升級：本體約束解碼**
```python
class OntologyConstrainedDecoder(nn.Module):
    """
    使用本體知識約束預測空間
    """
    def __init__(self, ontology_kb):
        super().__init__()
        self.ontology_kb = ontology_kb
        
        # 基礎解碼器
        self.base_decoder = DistMult(hidden_dim=512)
        
        # 約束檢查器
        self.constraint_checker = OntologyConstraintChecker(
            ontology_kb=ontology_kb
        )
    
    def forward(self, patient_emb, candidate_diseases):
        # 1. 基礎評分
        base_scores = self.base_decoder(patient_emb, candidate_diseases)
        
        # 2. 本體約束檢查
        valid_mask = self.constraint_checker.validate_batch(
            patient_phenotypes=patient_emb.phenotypes,
            candidate_diseases=candidate_diseases
        )
        
        # 3. 懲罰不合理預測
        constrained_scores = base_scores * valid_mask.float()
        
        # 4. 本體相似度增強
        ontology_scores = self.compute_ontology_similarity(
            patient_emb.phenotypes,
            candidate_diseases
        )
        
        # 5. 融合
        final_scores = (
            0.6 * constrained_scores + 
            0.4 * ontology_scores
        )
        
        return final_scores
```

### 第3層：檢索層 (Retrieval Layer)

#### 3.1 混合檢索策略

```python
class HybridRetrievalEngine:
    """
    向量檢索 + 圖檢索 + 路徑檢索
    """
    def __init__(self, kg, vector_index, path_retriever):
        self.kg = kg
        self.vector_index = vector_index
        self.path_retriever = path_retriever
    
    def retrieve(self, patient_phenotypes, top_k=20):
        # 1. 向量檢索（快速篩選）
        vector_results = self.vector_index.search(
            patient_phenotypes,
            k=100
        )
        
        # 2. 圖擴展（找到相關子圖）
        subgraph = self.kg.expand_subgraph(
            seed_nodes=vector_results,
            max_nodes=10000
        )
        
        # 3. 路徑檢索（推理路徑）
        diagnostic_paths = self.path_retriever.retrieve(
            patient_phenotypes,
            subgraph=subgraph
        )
        
        # 4. 融合排序
        final_ranking = self.fuse_rankings([
            vector_results,
            subgraph.node_scores,
            diagnostic_paths
        ], weights=[0.3, 0.3, 0.4])
        
        return final_ranking[:top_k]
```

### 第4層：服務層 (Service Layer)

**API設計（RESTful）：**

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Medical KG Diagnosis API")

class DiagnosisRequest(BaseModel):
    patient_id: str
    phenotypes: List[str]  # HPO IDs
    include_explanation: bool = True
    include_paths: bool = True

class DiagnosisResponse(BaseModel):
    patient_id: str
    top_diseases: List[Dict]  # [{disease_id, name, confidence, evidence}]
    top_genes: List[Dict]
    explanation: Optional[str]
    diagnostic_paths: Optional[List[Dict]]
    warnings: List[str]  # 本體約束警告

@app.post("/api/v2/diagnose")
async def diagnose(request: DiagnosisRequest):
    """
    診斷接口（v2 - 本體增強版）
    """
    # 1. 驗證輸入（本體檢查）
    validation = ontology_kb.validate_phenotype_set(request.phenotypes)
    if not validation['valid']:
        return {
            'error': 'Invalid phenotype combination',
            'details': validation['reason']
        }
    
    # 2. 檢索與推理
    results = model.predict(
        phenotypes=request.phenotypes,
        include_paths=request.include_paths
    )
    
    # 3. 生成解釋
    if request.include_explanation:
        explanation = path_reasoning.generate_explanation(
            results['paths']
        )
    else:
        explanation = None
    
    return DiagnosisResponse(
        patient_id=request.patient_id,
        top_diseases=results['diseases'],
        top_genes=results['genes'],
        explanation=explanation,
        diagnostic_paths=results['paths'] if request.include_paths else None,
        warnings=validation.get('warnings', [])
    )
```

### 第5層：部署層 (Deployment Layer)

#### 5.1 跨平台自適應部署

```python
# deploy/adaptive_deployment.py

class AdaptiveDeployment:
    """
    根據平台自動調整配置
    """
    def __init__(self):
        self.platform = detect_platform()
        self.config = self.get_platform_config()
    
    def get_platform_config(self):
        if self.platform == 'windows_x86_blackwell':
            return {
                'model': 'GraphGPS',
                'layers': 6,
                'hidden_dim': 512,
                'attention': 'flash_attention_2',
                'batch_size': 32,
                'fp16': True
            }
        elif self.platform == 'dgx_spark_arm_blackwell':
            return {
                'model': 'GraphGPS',  # 同樣的模型
                'layers': 4,  # 層數減少（記憶體考量）
                'hidden_dim': 256,  # 維度降低
                'attention': 'pytorch_sdpa',  # 降級注意力
                'batch_size': 64,  # 利用大記憶體
                'fp16': True
            }
        else:
            raise ValueError(f"Unsupported platform: {self.platform}")
```

---

## 跨平台兼容性策略

### 套件安裝矩陣

| 套件 | Windows x86 安裝 | DGX Spark ARM 安裝 | 備註 |
|------|------------------|---------------------|------|
| PyTorch 2.9 | `pip install torch==2.9.0 --index-url .../cu130` | 同左 | ✅ 官方支持 |
| PyG 2.6 | `pip install torch-geometric pyg-lib ...` | 同左或從源碼 | ⚠️ 需測試 |
| FlashAttn-2 | `pip install flash-attn --no-build-isolation` | ❌ 跳過 | 使用備案 |
| xformers | `pip install xformers` | 嘗試安裝 | ARM支持有限 |
| Voyager | `pip install voyager>=2.0` | ✅ | 跨平台 CPU |
| cuVS | ❌ (Linux only) | `pip install cuvs-cu13` | Linux GPU 加速 |

### 自適應注意力實現

```python
# src/models/attention/adaptive_backend.py

class AdaptiveAttentionBackend:
    """
    自動選擇最佳注意力實現
    優先級: FlashAttn-2 > xformers > PyTorch SDPA > 手動實現
    """
    def __init__(self):
        self.backend = self._detect_backend()
        logger.info(f"Using attention backend: {self.backend}")
    
    def _detect_backend(self):
        # 1. 嘗試FlashAttention-2
        try:
            import flash_attn
            if torch.backends.cuda.flash_sdp_enabled():
                return 'flash_attention_2'
        except ImportError:
            pass
        
        # 2. 嘗試xformers
        try:
            import xformers.ops
            return 'xformers'
        except ImportError:
            pass
        
        # 3. PyTorch原生SDPA
        if hasattr(F, 'scaled_dot_product_attention'):
            return 'pytorch_sdpa'
        
        # 4. 手動實現（最慢）
        return 'manual'
    
    def compute(self, q, k, v, attn_mask=None):
        if self.backend == 'flash_attention_2':
            from flash_attn import flash_attn_func
            return flash_attn_func(q, k, v, causal=False)
        
        elif self.backend == 'xformers':
            from xformers.ops import memory_efficient_attention
            return memory_efficient_attention(q, k, v, attn_bias=attn_mask)
        
        elif self.backend == 'pytorch_sdpa':
            return F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=attn_mask,
                enable_gqa=True  # Grouped Query Attention
            )
        
        else:  # manual
            return self._manual_attention(q, k, v, attn_mask)
```

---

## 效能預期與基準

### 模型效能（與SHEPHERD對比）

| 指標 | SHEPHERD | 本系統（基線） | 本系統（完整） |
|------|----------|----------------|----------------|
| Hits@1 | 42% | 45% | **58%** |
| Hits@5 | 54% | 58% | **74%** |
| Hits@10 | 60% | 65% | **82%** |
| Hits@20 | 68% | 72% | **88%** |
| MRR | 0.49 | 0.52 | **0.67** |
| 幻覺率 | ~20% | ~12% | **<5%** |
| 可解釋性 | ❌ | ⚠️ | ✅ |

### 推理延遲

| 平台 | 單患者延遲 | 吞吐量 (QPS) | 備註 |
|------|------------|--------------|------|
| Windows x86 (16GB) | 1.2s | 25 | FlashAttn-2 |
| DGX Spark (128GB) | 1.8s | 35 | 大batch優勢 |

### 訓練時間（完整PrimeKG）

| 階段 | Windows x86 | DGX Spark | 備註 |
|------|-------------|-----------|------|
| 資料預處理 | 2h | 1.5h | CPU密集 |
| 圖預訓練 | 48h | 36h | FP16混合精度 |
| 患者微調 | 12h | 8h | - |
| **總計** | **~62h** | **~45h** | 約2-3天 |

---

## 成功標準（臨床級）

### Phase 1: 技術驗證（MVP+）
- ✅ Hits@10 ≥ 70%
- ✅ 幻覺率 ≤ 10%
- ✅ 兩個平台都能運行
- ✅ 基本可解釋性

### Phase 2: 臨床原型
- ✅ Hits@10 ≥ 80%
- ✅ 幻覺率 ≤ 5%
- ✅ 完整推理路徑
- ✅ 本體約束驗證
- ✅ 推理延遲 < 2s

### Phase 3: 生產就緒
- ✅ Hits@10 ≥ 85%
- ✅ 幻覺率 ≤ 3%
- ✅ 與人類專家對比
- ✅ 臨床試驗驗證
- ✅ 監管合規（FDA/NMPA）

---

## 關鍵文獻

1. **SHEPHERD** (2025): Few-shot learning for phenotype-driven diagnosis
2. **DORI** (2025): Dual ontology + relational graph + Neural ODE
3. **DR.KNOWS** (2025): Knowledge graph paths for diagnosis prediction
4. **Knowledge Hypergraph** (2024): Higher-order medical relationships
5. **Foundation Models in Medicine** (2025): Comprehensive survey

---

**版本**: v2.0  
**最後更新**: 2025-10-07  
**升級重點**: 本體深度整合、知識超圖、跨平台兼容