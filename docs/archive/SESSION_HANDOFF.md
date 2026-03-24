# SHEPHERD-Advanced Session Handoff

**最後更新**: 2026-01-20
**用途**: 新對話起始prompt

---

## 起始 Prompt（複製到新對話）

```
我們正在開發 SHEPHERD-Advanced 醫療知識圖譜診斷推理引擎。

## 專案概述
這是一個基於異質圖神經網路的罕見疾病診斷系統：
- **P0 核心流程**: Phenotype → Gene → Disease 路徑推理
- **P1 附加功能**: Ortholog 跨物種推理（接口已預留）
- **目標**: Hits@10 ≥ 80%，幻覺率 < 5%

## 關鍵約束
1. **高精度要求**: 醫療系統需極高精度和可解釋性，不接受任何會大幅犧牲精度或可解釋性的做法
2. **16GB VRAM限制**: Windows環境需子圖採樣策略
3. **跨平台**: x86 + ARM (DGX Spark)
4. **協議合規**: 所有模組應符合 `src/core/protocols.py` 定義

## 已完成模組
- `src/core/`: types.py, protocols.py, schema.py
- `src/kg/`: graph.py, builder.py, preprocessing.py
- `src/reasoning/`: path_reasoning.py, explanation_generator.py
- `src/inference/`: pipeline.py, input_validator.py
- `src/models/`: gnn/, encoders/, decoders/, attention/ (框架完成)
- `src/ontology/`: hierarchy.py, loader.py, constraints.py
- 測試: 130 tests passing, ~52% coverage

## 當前進度
Phase 1 核心模組開發中，下一步需實現：
1. `scripts/train_model.py` - 訓練流程
2. `src/utils/metrics.py` - 評估指標 (Hits@k, MRR)
3. 完善資料載入器（子圖採樣）

## 文件位置
- 工程藍圖: `medical-kg-blueprint.md`
- TODO清單: `medical-kg-todo.md`
- 進度摘要: `docs/PROGRESS_2026-01-20.md`

## 開發規範
1. 所有模組都應有標準文件頭（功能、依賴、輸入輸出、路徑）
2. 使用 Result[T] 類型進行錯誤處理
3. 保持模組高度內聚，避免重複功能
4. P1 Ortholog 接口必須保留，但無需立即實現

請閱讀 `docs/PROGRESS_2026-01-20.md` 和 `medical-kg-todo.md` 了解詳細進度，然後繼續下一步開發。
```

---

## 快速參考

### 核心類型
```python
from src.core.types import (
    NodeID, Node, Edge, EdgeType, NodeType, DataSource,
    PatientPhenotypes, DiagnosisCandidate, InferenceResult, Result
)
```

### 推理流程
```python
from src.inference import DiagnosisPipeline, create_diagnosis_pipeline
from src.core.types import PatientPhenotypes

pipeline = create_diagnosis_pipeline(kg=knowledge_graph)
result = pipeline.run(
    patient_input=PatientPhenotypes(
        patient_id="patient_001",
        phenotypes=["HP:0001234", "HP:0002345"],
    ),
    top_k=10,
)
```

### 路徑推理
```python
from src.reasoning import PathReasoner, create_path_reasoner

reasoner = create_path_reasoner()
paths = reasoner.find_paths(source_ids, NodeType.DISEASE, kg)
scored_paths = reasoner.score_paths(paths, kg)
```

### 測試命令
```bash
python -m pytest tests/unit/ -v --tb=short
python -m pytest tests/unit/test_inference.py -v  # 推理測試
python -m pytest tests/unit/test_reasoning.py -v  # 推理模組測試
```

---

## 架構圖

```
PatientPhenotypes (HPO IDs)
        │
        ▼
┌─────────────────┐
│ InputValidator  │ ← 驗證HPO格式，支援自定義鉤子
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PathReasoner   │ ← BFS路徑搜索: Phenotype→Gene→Disease
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│DiagnosisPipeline│ ← 評分、排名、整合GNN（可選）
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ExplanationGenerator │ ← 人類可讀解釋
└─────────┬───────────┘
          │
          ▼
    InferenceResult
```

---

## P1 Ortholog 接口預留

```python
# PipelineConfig 已包含
ortholog_weight: float = 0.3
ortholog_species: List[str] = ["mouse", "zebrafish", "rat"]
min_ortholog_confidence: float = 0.5

# OrthologGate 已在 models/gnn/layers.py
# OrthologMapping 已在 core/types.py
# 待實現: src/reasoning/ortholog_reasoning.py
```

---

**建議**: 新session開始時，先執行 `python -m pytest tests/unit/ -v --tb=short` 確認環境正常。
