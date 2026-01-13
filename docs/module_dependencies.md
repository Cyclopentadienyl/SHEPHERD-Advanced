# SHEPHERD-Advanced 模組依賴關係

**版本**: 1.0.0
**日期**: 2026-01-13
**狀態**: 框架設計階段

---

## 模組總覽

```
src/
├── core/               # 核心模組 (Protocol + Types + Schema)
├── config/             # 配置管理
├── utils/              # 工具函數
├── ontology/           # 本體處理
├── data_sources/       # 資料來源 (PubMed, Ortholog, etc.)
├── kg/                 # 知識圖譜
├── nlp/                # 自然語言處理
├── medical_standards/  # 醫療標準
├── models/             # 神經網路模型
├── retrieval/          # 向量檢索
├── reasoning/          # 推理引擎
├── llm/                # LLM 整合
├── training/           # 訓練模組
├── inference/          # 推理管線
├── api/                # REST API
└── webui/              # Web UI
```

---

## 依賴關係圖

```
                    ┌─────────────────────────────────────────────────────┐
                    │                     core/                            │
                    │  (types.py, schema.py, protocols.py)                 │
                    │                                                      │
                    │  所有模組的基礎依賴                                   │
                    └───────────────────────┬─────────────────────────────┘
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    │                       │                       │
                    ▼                       ▼                       ▼
            ┌───────────┐           ┌───────────┐           ┌───────────┐
            │  config/  │           │  utils/   │           │ ontology/ │
            │           │           │           │           │           │
            │ 配置載入   │           │ 工具函數  │           │ 本體載入   │
            └─────┬─────┘           └─────┬─────┘           └─────┬─────┘
                  │                       │                       │
                  └───────────────────────┼───────────────────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
                    ▼                     ▼                     ▼
          ┌─────────────────┐    ┌───────────────┐    ┌───────────────────┐
          │  data_sources/  │    │      kg/      │    │ medical_standards/│
          │                 │    │               │    │                   │
          │ PubMed          │───▶│ 知識圖譜建構  │◀───│ FHIR/ICD 映射     │
          │ Ortholog        │    │ 存儲/查詢     │    │                   │
          └────────┬────────┘    └───────┬───────┘    └─────────┬─────────┘
                   │                     │                      │
                   └─────────────────────┼──────────────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
                    ▼                    ▼                    ▼
            ┌───────────┐        ┌───────────┐        ┌───────────┐
            │  models/  │        │ retrieval/│        │    nlp/   │
            │           │        │           │        │           │
            │ GNN       │        │ 向量索引  │        │ 症狀提取  │
            │ Encoder   │        │ 子圖採樣  │        │ HPO 匹配  │
            │ Decoder   │        │           │        │           │
            └─────┬─────┘        └─────┬─────┘        └─────┬─────┘
                  │                    │                    │
                  └────────────────────┼────────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
                    ▼                  ▼                  ▼
            ┌───────────┐      ┌───────────────┐   ┌───────────┐
            │ reasoning/│      │    llm/       │   │ training/ │
            │           │      │               │   │           │
            │ 路徑推理  │◀────▶│ 解釋生成     │   │ 模型訓練  │
            │ 約束檢查  │      │ 臨床建議     │   │           │
            └─────┬─────┘      └───────┬───────┘   └─────┬─────┘
                  │                    │                 │
                  └────────────────────┼─────────────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │     inference/      │
                            │                     │
                            │ 整合推理管線        │
                            │ 輸入驗證/輸出格式化 │
                            └──────────┬──────────┘
                                       │
                          ┌────────────┼────────────┐
                          │                         │
                          ▼                         ▼
                  ┌───────────────┐         ┌───────────────┐
                  │     api/      │         │    webui/     │
                  │               │         │               │
                  │ REST API      │         │ Gradio UI     │
                  │ FastAPI       │         │               │
                  └───────────────┘         └───────────────┘
```

---

## 模組詳細說明

### 1. `core/` - 核心模組 (無依賴)

**檔案結構**:
```
core/
├── __init__.py      # 統一導出
├── types.py         # 核心類型定義 (NodeType, EdgeType, etc.)
├── schema.py        # KG Schema 定義 (預留同源基因)
└── protocols.py     # 所有 Protocol 接口
```

**提供**:
- 所有 Enum 類型 (NodeType, EdgeType, Species, etc.)
- 資料結構 (Node, Edge, Publication, OrthologMapping, etc.)
- 配置類型 (ModelConfig, TrainingConfig, DataConfig)
- 所有模組的 Protocol 接口

**依賴**: 僅依賴標準庫和 numpy

---

### 2. `config/` - 配置管理

**Protocol**: `ConfigLoaderProtocol`

**依賴**:
- `core/` (types, protocols)

**提供**:
- 配置載入和驗證
- JSON Schema 驗證
- 平台檢測

---

### 3. `utils/` - 工具函數

**依賴**:
- `core/` (types)
- `config/` (配置)

**提供**:
- 日誌設置
- 平台檢測
- 版本檢查
- 雜湊生成
- 評估指標

---

### 4. `ontology/` - 本體處理

**Protocol**: `OntologyLoaderProtocol`, `OntologyProtocol`, `OntologyConstraintProtocol`

**依賴**:
- `core/` (types, protocols)
- `config/` (配置)
- `utils/` (工具)

**提供**:
- HPO/MONDO/GO/MP 本體載入
- 本體階層查詢
- 語義相似度計算
- 本體約束驗證

---

### 5. `data_sources/` - 資料來源 (NEW)

**Protocol**: `DataSourceProtocol`, `PubMedDataSourceProtocol`, `OrthologDataSourceProtocol`

**依賴**:
- `core/` (types, protocols)
- `config/` (配置)

**提供**:
- DisGeNET/ClinVar/OMIM 資料載入
- **PubMed/Pubtator 文獻資料** (Phase 2)
- **同源基因資料 (Ensembl/OrthoDB/MGI)** (Phase 2)
- 文獻可信度評分

**重要檔案**:
- `pubmed.py` - PubMed 資料來源
- `ortholog.py` - 同源基因資料來源

---

### 6. `kg/` - 知識圖譜

**Protocol**: `KnowledgeGraphProtocol`, `KnowledgeGraphBuilderProtocol`

**依賴**:
- `core/` (types, schema, protocols)
- `ontology/` (本體)
- `data_sources/` (資料來源)

**提供**:
- 知識圖譜建構
- 圖存儲和查詢
- PyG HeteroData 轉換
- 子圖提取

---

### 7. `nlp/` - 自然語言處理

**Protocol**: `SymptomExtractorProtocol`, `HPOMatcherProtocol`

**依賴**:
- `core/` (types, protocols)
- `ontology/` (HPO 本體)

**提供**:
- 症狀提取 (Free text → HPO)
- HPO 術語模糊匹配
- 醫療實體識別

---

### 8. `medical_standards/` - 醫療標準

**Protocol**: `FHIRAdapterProtocol`, `MedicalCodeMapperProtocol`

**依賴**:
- `core/` (types, protocols)
- `ontology/` (本體映射)

**提供**:
- FHIR 資料解析和導出
- ICD-10/11 → HPO/MONDO 映射
- SNOMED CT 映射

---

### 9. `models/` - 神經網路模型

**Protocol**: `NodeEncoderProtocol`, `GNNProtocol`, `AttentionBackendProtocol`, `DecoderProtocol`, `DiagnosisModelProtocol`

**依賴**:
- `core/` (types, protocols)
- `config/` (模型配置)

**提供**:
- 節點編碼器
- 異質圖神經網路 (HeteroGNN)
- 自適應注意力後端
- 診斷解碼器

---

### 10. `retrieval/` - 向量檢索

**Protocol**: `VectorIndexProtocol`, `SubgraphSamplerProtocol`

**依賴**:
- `core/` (types, protocols)
- `kg/` (知識圖譜)

**提供**:
- FAISS/hnswlib 向量索引
- 子圖採樣

---

### 11. `reasoning/` - 推理引擎

**Protocol**: `PathReasonerProtocol`, `OrthologReasonerProtocol`, `ConstraintCheckerProtocol`, `ExplanationGeneratorProtocol`

**依賴**:
- `core/` (types, protocols)
- `kg/` (知識圖譜)
- `ontology/` (約束)
- `data_sources/` (同源基因)

**提供**:
- 路徑推理 (DR.KNOWS style)
- **同源基因推理** (Phase 2)
- 約束檢查
- 解釋生成

---

### 12. `llm/` - LLM 整合

**Protocol**: `LLMProtocol`, `MedicalLLMProtocol`

**依賴**:
- `core/` (types, protocols)
- `config/` (LLM 配置)

**提供**:
- vLLM/llama.cpp 後端
- 臨床解釋生成
- 證據總結

---

### 13. `training/` - 訓練模組

**Protocol**: `TrainerProtocol`

**依賴**:
- `core/` (types, protocols)
- `models/` (模型)
- `kg/` (訓練資料)

**提供**:
- 訓練循環
- 損失函數
- 檢查點管理

---

### 14. `inference/` - 推理管線

**Protocol**: `InferencePipelineProtocol`, `InputValidatorProtocol`, `OutputFormatterProtocol`

**依賴**:
- `core/` (所有類型)
- `models/` (診斷模型)
- `kg/` (知識圖譜)
- `reasoning/` (推理)
- `llm/` (解釋生成)
- `nlp/` (輸入處理)

**提供**:
- 完整推理管線
- 輸入驗證
- 輸出格式化

---

### 15. `api/` - REST API

**Protocol**: `APIServiceProtocol`

**依賴**:
- `inference/` (推理管線)
- `medical_standards/` (FHIR)

**提供**:
- FastAPI 路由
- 診斷 API
- 健康檢查

---

### 16. `webui/` - Web UI

**依賴**:
- `inference/` (推理管線)
- `nlp/` (HPO 搜尋)

**提供**:
- Gradio 界面
- 智能輸入表單
- 結果可視化

---

## 新增功能的接口預留

### PubMed 資料整合

**相關模組**: `data_sources/pubmed.py`

**Protocol**: `PubMedDataSourceProtocol`

**主要方法**:
```python
def compute_credibility_score(publication: Publication) -> float
def get_gene_disease_literature(gene_id, disease_id) -> LiteratureEvidence
def get_pubtator_annotations(pmid) -> Dict[str, List[str]]
```

**整合點**:
1. `kg/builder.py` - 添加文獻支持的邊
2. `reasoning/explanation_generator.py` - 生成文獻證據解釋

---

### 同源基因比對 (深度整合到 GNN)

**相關模組**: `data_sources/ortholog.py`, `reasoning/ortholog_reasoning.py`

**Protocol**: `OrthologDataSourceProtocol`, `OrthologReasonerProtocol`

**KG Schema 預留**:
```python
# Node Types
NodeType.ORTHOLOG_GROUP      # 同源基因群組
NodeType.MOUSE_GENE          # 小鼠基因
NodeType.MOUSE_PHENOTYPE     # 小鼠表型

# Edge Types
EdgeType.ORTHOLOG_OF                # 同源關係
EdgeType.HUMAN_MOUSE_ORTHOLOG       # 人-鼠同源
EdgeType.ORTHOLOG_IN_GROUP          # 基因 -> 群組
EdgeType.MOUSE_GENE_HAS_PHENOTYPE   # 小鼠基因 -> 表型
```

**Metapath 預留**:
```python
# 同源基因推理路徑
"ortholog_phenotype_inference":
    Gene -> HUMAN_MOUSE_ORTHOLOG -> MouseGene -> MOUSE_GENE_HAS_PHENOTYPE -> MousePhenotype

"ortholog_group_reasoning":
    Gene -> ORTHOLOG_IN_GROUP -> OrthologGroup -> ORTHOLOG_IN_GROUP -> MouseGene -> ...
```

**整合點**:
1. `kg/builder.py` - 添加同源基因節點和邊
2. `models/gnn/` - 處理跨物種邊類型
3. `reasoning/ortholog_reasoning.py` - 同源基因推理邏輯
4. `reasoning/explanation_generator.py` - 生成同源基因證據解釋

---

## Import 規則 (.import-linter.ini)

```ini
[importlinter]
root_package = src

[importlinter:contract:layers]
name = Enforce layered architecture
type = layers
layers =
    src.core          # Layer 0: 基礎類型 (無依賴)
    src.config        # Layer 1: 配置
    src.utils         # Layer 1: 工具
    src.ontology      # Layer 2: 本體
    src.data_sources  # Layer 2: 資料來源
    src.kg            # Layer 3: 知識圖譜
    src.nlp           # Layer 3: NLP
    src.medical_standards  # Layer 3: 醫療標準
    src.models        # Layer 4: 模型
    src.retrieval     # Layer 4: 檢索
    src.reasoning     # Layer 5: 推理
    src.llm           # Layer 5: LLM
    src.training      # Layer 6: 訓練
    src.inference     # Layer 7: 推理管線
    src.api           # Layer 8: API
    src.webui         # Layer 8: Web UI
```

---

## 開發建議

### 新增模組步驟

1. **定義 Protocol** (在 `core/protocols.py`)
2. **定義類型** (在 `core/types.py` 如需要)
3. **創建模組目錄和 `__init__.py`**
4. **實現 Skeleton (stub)** - 先讓接口可用
5. **填充實現**
6. **添加測試**

### 模組開發優先級

1. **Phase 1** (基礎): `core`, `config`, `utils`, `ontology`
2. **Phase 1** (KG): `kg`, `data_sources` (基礎)
3. **Phase 2** (模型): `models`, `retrieval`
4. **Phase 2** (推理): `reasoning`, `llm`
5. **Phase 2** (高級): `data_sources` (PubMed, Ortholog)
6. **Phase 3** (整合): `inference`, `api`, `webui`

---

## 版本歷史

| 版本 | 日期 | 變更 |
|------|------|------|
| 1.0.0 | 2026-01-13 | 初始版本，包含完整模組設計和 Protocol 定義 |

---

**文檔結束**
