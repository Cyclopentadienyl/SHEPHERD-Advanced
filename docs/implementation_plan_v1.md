# SHEPHERD-Advanced 實施計劃 - 資料結構與醫療整合

**版本**: 1.0  
**日期**: 2025-11-04  
**參考文檔**: `data_structure_and_validation_v3.md`

---

## 🎯 實施目標

1. **建立完整的資料結構與校驗系統** (v3.0 架構)
2. **整合醫生團隊建議** (NLP + FHIR + 多模態輸入)
3. **預留擴充接口** (確保未來可擴展性)

---

## 📅 實施時程

```
Week 1-2: Phase 1 - 核心架構與驗證系統 (🔴 P0)
Week 3-6: Phase 2 - 醫療功能整合 (🟡 P1)  
Week 7-8: Phase 3 - 擴充接口與文檔 (🟢 P2)
```

**總預計時間**: 8 週  
**最快完成**: 6 週 (如果並行作業)  
**建議完成**: 8 週 (穩健開發)

---

## 📋 Phase 1: 核心架構 (Week 1-2)

### Day 1-2: 專案配置與工具鏈

#### 任務 1.1: 建立 Python 專案配置
```bash
# 1. 創建 pyproject.toml
cd /path/to/shepherd-advanced
cat > pyproject.toml << 'EOF'
[project]
name = "shepherd-advanced"
version = "1.0.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.5.0",
    "torch-geometric>=2.5.0",
    "pronto>=2.5.0",
    "pydantic>=2.0.0",
    "fastapi>=0.100.0",
    "gradio>=4.0.0",
    "jsonschema>=4.0.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
nlp = [
    "transformers>=4.30.0",
    "scispacy>=0.5.0",
]
medical = [
    "fhir.resources>=7.0.0",
    "python-hl7>=0.4.0",
]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
    "import-linter>=1.12.0",
]

[tool.black]
line-length = 100

[tool.ruff]
line-length = 100
select = ["E", "F", "I", "N", "W"]

[tool.mypy]
python_version = "3.10"
strict = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --cov=src"
EOF

# 2. 安裝依賴
pip install -e ".[dev,nlp,medical]"
```

**檢查點**:
- [ ] pyproject.toml 創建完成
- [ ] 所有核心依賴安裝成功
- [ ] 可選依賴 (nlp, medical) 安裝成功
- [ ] black/ruff 可正常執行

---

#### 任務 1.2: 配置依賴規則檢查
```bash
# 創建 .import-linter.ini
cat > .import-linter.ini << 'EOF'
[importlinter]
root_package = src

[importlinter:contract:layers]
name = Enforce layered architecture
type = layers
layers =
    src.utils
    src.config
    src.ontology
    src.kg
    src.nlp
    src.medical_standards
    src.models
    src.retrieval
    src.reasoning
    src.llm
    src.training
    src.inference
    src.api
    src.webui

[importlinter:contract:forbidden]
name = Forbidden imports
type = forbidden
source_modules =
    src.models
    src.inference
forbidden_modules =
    src.api
    src.webui
    src.training
EOF

# 測試依賴檢查
lint-imports
```

**檢查點**:
- [ ] .import-linter.ini 創建完成
- [ ] 依賴檢查通過 (初次可能有警告)
- [ ] 理解分層架構規則

---

#### 任務 1.3: 配置 Git Hooks
```bash
# 創建 .pre-commit-config.yaml
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.0
    hooks:
      - id: black
        language_version: python3.12

  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.1.9
    hooks:
      - id: ruff
        args: [--fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-PyYAML, types-requests]
EOF

# 安裝 pre-commit hooks
pre-commit install
```

**檢查點**:
- [ ] .pre-commit-config.yaml 創建完成
- [ ] pre-commit hooks 安裝成功
- [ ] 測試 commit 時自動格式化

---

### Day 3-4: JSON Schema 與配置驗證

#### 任務 2.1: 創建 JSON Schema 檔案
```bash
# 創建 schemas 目錄
mkdir -p configs/schemas

# 1. 患者輸入 Schema
cat > configs/schemas/patient_input.schema.json << 'EOF'
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Patient Input Schema",
  "type": "object",
  "required": ["patient_id", "phenotypes"],
  "properties": {
    "patient_id": {
      "type": "string",
      "pattern": "^P[0-9]{5,10}$"
    },
    "phenotypes": {
      "type": "array",
      "minItems": 1,
      "items": {"type": "string", "pattern": "^HP:[0-9]{7}$"}
    },
    "demographics": {
      "type": "object",
      "properties": {
        "age": {"type": "integer", "minimum": 0, "maximum": 150},
        "gender": {"type": "string", "enum": ["male", "female", "other"]}
      }
    }
  }
}
EOF

# 2. 推理輸出 Schema
cat > configs/schemas/inference_output.schema.json << 'EOF'
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Inference Output Schema",
  "type": "object",
  "required": ["patient_id", "timestamp", "top_candidates"],
  "properties": {
    "patient_id": {"type": "string"},
    "timestamp": {"type": "string", "format": "date-time"},
    "top_candidates": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["disease", "confidence"],
        "properties": {
          "disease": {
            "type": "object",
            "properties": {
              "mondo_id": {"type": "string", "pattern": "^MONDO:[0-9]{7}$"},
              "name": {"type": "string"}
            }
          },
          "confidence": {"type": "number", "minimum": 0, "maximum": 1}
        }
      }
    }
  }
}
EOF

# 3-5. 其他配置 Schema (base, model, data)
# ... (類似格式)
```

**檢查點**:
- [ ] 所有 5 個 schema 檔案創建完成
- [ ] JSON 格式驗證通過 (使用 jsonlint)
- [ ] Schema 邏輯正確 (手動測試)

---

#### 任務 2.2: 實現配置驗證器
```bash
# 創建目錄
mkdir -p src/config

# 創建驗證器
cat > src/config/config_validator.py << 'EOF'
"""配置驗證器實現"""
# (參考完整設計文檔中的程式碼)
EOF

# 創建測試
cat > tests/unit/test_config_validator.py << 'EOF'
import pytest
from src.config.config_validator import ConfigValidator

def test_validate_patient_input():
    validator = ConfigValidator(...)
    # 測試有效輸入
    valid_input = {
        "patient_id": "P12345",
        "phenotypes": ["HP:0003324"]
    }
    assert validator.validate(valid_input) is True
    
    # 測試無效輸入
    invalid_input = {"patient_id": "INVALID"}
    with pytest.raises(ValidationError):
        validator.validate(invalid_input)
EOF

# 執行測試
pytest tests/unit/test_config_validator.py -v
```

**檢查點**:
- [ ] `ConfigValidator` 類實現完成
- [ ] 可載入所有 JSON Schema
- [ ] 單元測試通過
- [ ] CLI 命令可用: `python -m src.config.config_validator`

---

### Day 5-7: 版本管理與 Metadata

#### 任務 3.1: 實現版本檢查器
```bash
# 創建版本檢查器
cat > src/utils/version_checker.py << 'EOF'
"""版本兼容性檢查器"""
# (參考完整設計文檔)
EOF

# 創建哈希生成器
cat > src/utils/hash_generator.py << 'EOF'
"""資料哈希生成工具"""
import hashlib
from pathlib import Path

def compute_file_hash(file_path: Path) -> str:
    """計算 SHA256"""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"
EOF

# 測試
python -c "
from src.utils.hash_generator import compute_file_hash
from pathlib import Path
print(compute_file_hash(Path('README.md')))
"
```

**檢查點**:
- [ ] `VersionChecker` 實現完成
- [ ] `hash_generator` 實現完成
- [ ] 可計算任意檔案哈希
- [ ] 版本兼容性檢查邏輯正確

---

#### 任務 3.2: 創建 Metadata 模板
```bash
# 1. 知識圖譜 Metadata 模板
cat > data/processed/knowledge_graph/metadata.json << 'EOF'
{
  "schema_version": "3.0",
  "data_version": "YYYY.MM.DD",
  "creation_timestamp": "ISO-8601",
  "generator": {
    "script": "scripts/build_knowledge_graph.py",
    "commit_sha": "COMMIT_SHA",
    "git_branch": "main"
  },
  "data_sources": {
    "hpo": {
      "version": "YYYY-MM-DD",
      "url": "http://purl.obolibrary.org/obo/hp.obo",
      "sha256": "TO_BE_FILLED"
    }
  },
  "statistics": {
    "num_nodes": 0,
    "num_edges": 0
  },
  "data_hash": {
    "graph_structure": "sha256:TO_BE_FILLED"
  }
}
EOF

# 2. 模型註冊表模板
mkdir -p models/production
cat > models/production/registry.json << 'EOF'
{
  "registry_version": "1.0",
  "models": [],
  "current_production": null
}
EOF
```

**檢查點**:
- [ ] metadata.json 模板創建
- [ ] registry.json 模板創建
- [ ] 理解 metadata 結構與用途

---

#### 任務 3.3: 更新 KG Builder 自動生成 Metadata
```bash
# 編輯 src/kg/builder.py
# 添加 metadata 生成邏輯

cat >> src/kg/builder.py << 'EOF'

def generate_metadata(self, graph, output_path: Path):
    """生成知識圖譜 metadata"""
    import json
    from datetime import datetime
    from src.utils.hash_generator import compute_file_hash
    import subprocess
    
    # 獲取 Git 資訊
    commit_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"]
    ).decode().strip()
    
    metadata = {
        "schema_version": "3.0",
        "data_version": datetime.now().strftime("%Y.%m.%d"),
        "creation_timestamp": datetime.now().isoformat(),
        "generator": {
            "script": "scripts/build_knowledge_graph.py",
            "commit_sha": commit_sha,
            "git_branch": "main"
        },
        "statistics": {
            "num_nodes": graph.num_nodes,
            "num_edges": graph.num_edges
        },
        "data_hash": {
            "graph_structure": compute_file_hash(output_path / "hetero_graph.pt")
        }
    }
    
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
EOF
```

**檢查點**:
- [ ] KG builder 可自動生成 metadata
- [ ] Metadata 包含所有必要欄位
- [ ] 資料哈希計算正確

---

### Day 8-10: 平台特定測試

#### 任務 4.1: 創建測試框架
```bash
# 創建測試目錄
mkdir -p tests/benchmarks/platform_specific

# x86 注意力測試
cat > tests/benchmarks/platform_specific/test_attention_x86.py << 'EOF'
import torch
import pytest

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires CUDA"
)
@pytest.mark.x86_64
def test_flash_attention_x86():
    """測試 FlashAttention-2 在 x86 上是否可用"""
    device = torch.device("cuda")
    
    # 嘗試使用 FlashAttention
    try:
        from flash_attn import flash_attn_func
        
        batch_size, seq_len, d_model = 4, 128, 512
        q = torch.randn(batch_size, seq_len, 8, 64, device=device)
        k = torch.randn(batch_size, seq_len, 8, 64, device=device)
        v = torch.randn(batch_size, seq_len, 8, 64, device=device)
        
        output = flash_attn_func(q, k, v)
        assert output.shape[0] == batch_size
        
    except ImportError:
        pytest.skip("FlashAttention not available")
EOF

# ARM 注意力測試
cat > tests/benchmarks/platform_specific/test_attention_arm.py << 'EOF'
import torch
import pytest

@pytest.mark.arm64
def test_cudnn_sdpa_arm():
    """測試 cuDNN SDPA 在 ARM 上是否可用"""
    device = torch.device("cuda")
    
    batch_size, seq_len, d_model = 4, 128, 512
    q = torch.randn(batch_size, seq_len, d_model, device=device)
    k = torch.randn(batch_size, seq_len, d_model, device=device)
    v = torch.randn(batch_size, seq_len, d_model, device=device)
    
    # 使用 PyTorch 內建 SDPA
    with torch.backends.cuda.sdp_kernel(
        enable_flash=False,
        enable_math=False,
        enable_mem_efficient=True
    ):
        output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    
    assert torch.isfinite(output).all()
EOF

# 執行測試
pytest tests/benchmarks/platform_specific/ -v -m x86_64
```

**檢查點**:
- [ ] x86 測試框架建立
- [ ] ARM 測試框架建立
- [ ] 測試可正確標記平台
- [ ] CI 可分別執行不同平台測試

---

### Phase 1 完成檢查 ✅

**必須達成的里程碑**:
- [ ] ✅ pyproject.toml + 工具鏈正常運作
- [ ] ✅ import-linter 檢查通過
- [ ] ✅ 所有 JSON Schema 創建並驗證通過
- [ ] ✅ ConfigValidator 實現並測試通過
- [ ] ✅ VersionChecker 實現並測試通過
- [ ] ✅ Metadata 模板創建
- [ ] ✅ 平台特定測試框架建立
- [ ] ✅ 文檔更新: 新增實施紀錄

**預計工作量**: 16-20 小時  
**實際工作量**: _____ 小時 (待填寫)

---

## 📋 Phase 2: 醫療功能整合 (Week 3-6)

### Week 3: NLP 模塊基礎

#### 任務 5.1: 建立 NLP 模塊結構
```bash
# 創建目錄
mkdir -p src/nlp
touch src/nlp/__init__.py

# 創建佔位檔案 (Phase 2 實現)
for file in symptom_extractor entity_recognizer clinical_bert hpo_matcher; do
    cat > src/nlp/${file}.py << 'EOF'
"""
${file} - Phase 2 實現
TODO: 整合 SciBERT/ClinicalBERT
"""
from typing import List, Dict

class PlaceholderClass:
    """佔位類 - Phase 2 實現"""
    
    def __init__(self):
        self._initialized = False
    
    def initialize(self):
        """延遲初始化"""
        raise NotImplementedError("Phase 2 implementation")
EOF
done
```

**檢查點**:
- [ ] src/nlp/ 目錄結構創建
- [ ] 佔位類定義完成
- [ ] import 路徑正確

---

#### 任務 5.2: 下載 NLP 預訓練模型
```bash
# 下載 SciBERT
mkdir -p models/pretrained/scibert
python -c "
from transformers import AutoTokenizer, AutoModel

model_name = 'allenai/scibert_scivocab_uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

tokenizer.save_pretrained('models/pretrained/scibert')
model.save_pretrained('models/pretrained/scibert')
print('SciBERT downloaded successfully')
"

# 安裝 scispacy
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_sm-0.5.0.tar.gz
```

**檢查點**:
- [ ] SciBERT 模型下載完成
- [ ] scispacy 安裝成功
- [ ] 可成功載入模型

---

#### 任務 5.3: 實現 HPO 術語匹配器
```bash
# 實現 HPOMatcher (簡化版)
cat > src/nlp/hpo_matcher.py << 'EOF'
"""HPO 術語匹配器 - 模糊搜尋"""
from typing import List, Tuple
from pronto import Ontology
import re

class HPOMatcher:
    """HPO 術語模糊匹配"""
    
    def __init__(self, hpo_path: str):
        self.hpo = Ontology(hpo_path)
        self._build_index()
    
    def _build_index(self):
        """建立搜尋索引"""
        self.term_index = {}
        for term in self.hpo.terms():
            # 主名稱
            self.term_index[term.name.lower()] = term.id
            # 同義詞
            for syn in term.synonyms:
                self.term_index[syn.description.lower()] = term.id
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """
        模糊搜尋 HPO 術語
        
        Returns:
            List of (hpo_id, hpo_name, similarity_score)
        """
        query = query.lower()
        results = []
        
        for term_name, hpo_id in self.term_index.items():
            # 簡單的字串相似度
            if query in term_name or term_name in query:
                score = len(query) / max(len(term_name), len(query))
                results.append((hpo_id, term_name, score))
        
        # 排序並返回 top_k
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]
EOF

# 測試
python -c "
from src.nlp.hpo_matcher import HPOMatcher
matcher = HPOMatcher('data/raw/ontologies/hpo.obo')
results = matcher.search('muscle weakness')
print(results)
"
```

**檢查點**:
- [ ] HPOMatcher 實現完成
- [ ] 可搜尋 HPO 術語
- [ ] 搜尋結果合理

---

### Week 4: FHIR/HISS 適配器

#### 任務 6.1: 建立醫療標準模塊
```bash
# 創建目錄
mkdir -p src/medical_standards
touch src/medical_standards/__init__.py

# 安裝 FHIR 庫
pip install fhir.resources python-hl7
```

---

#### 任務 6.2: 實現 FHIR 適配器 (基礎版)
```bash
cat > src/medical_standards/fhir_adapter.py << 'EOF'
"""FHIR 適配器 - 基礎實現"""
from typing import Dict, Any
from fhir.resources.bundle import Bundle
import logging

logger = logging.getLogger(__name__)

class FHIRAdapter:
    """FHIR 資料適配器"""
    
    def parse_bundle(self, fhir_json: Dict[str, Any]) -> Dict[str, Any]:
        """解析 FHIR Bundle"""
        try:
            bundle = Bundle.parse_obj(fhir_json)
        except Exception as e:
            logger.error(f"Failed to parse FHIR Bundle: {e}")
            raise
        
        # 提取患者資料
        patient_data = {
            "patient_id": None,
            "phenotypes": [],
            "diagnoses": {"icd10": []},
            "demographics": {}
        }
        
        for entry in bundle.entry or []:
            resource = entry.resource
            
            if resource.resource_type == "Patient":
                patient_data["patient_id"] = f"P{resource.id}"
                if resource.birthDate:
                    from datetime import datetime
                    age = (datetime.now() - datetime.fromisoformat(str(resource.birthDate))).days // 365
                    patient_data["demographics"]["age"] = age
                patient_data["demographics"]["gender"] = resource.gender
            
            elif resource.resource_type == "Condition":
                for coding in resource.code.coding or []:
                    if "icd" in coding.system.lower():
                        patient_data["diagnoses"]["icd10"].append(coding.code)
        
        return patient_data
EOF

# 測試
python -c "
from src.medical_standards.fhir_adapter import FHIRAdapter

# 測試資料
fhir_json = {
    'resourceType': 'Bundle',
    'entry': [
        {
            'resource': {
                'resourceType': 'Patient',
                'id': '12345',
                'birthDate': '2010-01-01',
                'gender': 'female'
            }
        }
    ]
}

adapter = FHIRAdapter()
result = adapter.parse_bundle(fhir_json)
print(result)
"
```

**檢查點**:
- [ ] FHIRAdapter 基礎實現完成
- [ ] 可解析 FHIR Bundle
- [ ] 單元測試通過

---

### Week 5-6: WebUI 增強

#### 任務 7.1: 實現智能輸入表單
```bash
# 創建 WebUI 組件
mkdir -p src/webui/components
touch src/webui/components/__init__.py

# 實現 HPO 搜尋組件
cat > src/webui/components/hpo_search.py << 'EOF'
"""HPO 搜尋組件"""
import gradio as gr
from src.nlp.hpo_matcher import HPOMatcher

class HPOSearchComponent:
    """HPO 術語搜尋 UI 組件"""
    
    def __init__(self, hpo_path: str):
        self.matcher = HPOMatcher(hpo_path)
    
    def search(self, query: str) -> list:
        """搜尋 HPO 術語"""
        if not query or len(query) < 2:
            return []
        
        results = self.matcher.search(query, top_k=10)
        return [f"{hpo_id} - {name}" for hpo_id, name, score in results]
    
    def create_ui(self):
        """創建 UI"""
        with gr.Row():
            search_box = gr.Textbox(
                label="搜尋 HPO 術語",
                placeholder="輸入症狀關鍵字..."
            )
            results = gr.Dropdown(
                label="搜尋結果",
                choices=[],
                multiselect=False
            )
        
        search_box.change(
            fn=self.search,
            inputs=[search_box],
            outputs=[results]
        )
        
        return search_box, results
EOF
```

---

#### 任務 7.2: 整合到主界面
```bash
# 更新 src/webui/app.py
# 添加 HPO 搜尋功能
```

**檢查點**:
- [ ] HPO 搜尋組件實現
- [ ] 整合到 Gradio 界面
- [ ] UI 功能測試通過

---

### Phase 2 完成檢查 ✅

**必須達成的里程碑**:
- [ ] ✅ NLP 模塊結構建立
- [ ] ✅ SciBERT 模型下載
- [ ] ✅ HPOMatcher 實現並測試
- [ ] ✅ FHIRAdapter 基礎實現
- [ ] ✅ WebUI 增強 (HPO 搜尋)
- [ ] ✅ 整合測試通過

**預計工作量**: 40-50 小時  
**實際工作量**: _____ 小時 (待填寫)

---

## 📋 Phase 3: 擴充接口與文檔 (Week 7-8)

### Week 7: LLM 接口與最終整合

#### 任務 8.1: 實現 LLM 接口
```bash
# 實現 LLM 接口 (參考完整設計文檔)
cat > src/llm/interface.py << 'EOF'
# (實現內容見設計文檔)
EOF

# 實現 vLLM 後端
cat > src/llm/vllm_backend.py << 'EOF'
# (實現內容見設計文檔)
EOF
```

---

#### 任務 8.2: 端到端測試
```bash
# 創建端到端測試
cat > tests/integration/test_full_pipeline.py << 'EOF'
"""端到端測試 - 完整推理流程"""
import pytest

def test_full_pipeline_structured_input():
    """測試結構化輸入完整流程"""
    # 1. 準備患者資料
    patient_data = {
        "patient_id": "P12345",
        "phenotypes": ["HP:0003324", "HP:0011675"],
        "demographics": {"age": 8, "gender": "female"}
    }
    
    # 2. 驗證輸入
    from src.inference.input_validator import InputValidator
    validator = InputValidator(schema_path="configs/schemas/patient_input.schema.json")
    validator.validate(patient_data)
    
    # 3. 執行推理
    from src.inference.pipeline import DiagnosticPipeline
    pipeline = DiagnosticPipeline()
    results = pipeline.predict(patient_data)
    
    # 4. 驗證輸出
    assert "top_candidates" in results
    assert len(results["top_candidates"]) > 0

def test_full_pipeline_fhir_input():
    """測試 FHIR 輸入完整流程"""
    # 1. 準備 FHIR 資料
    fhir_bundle = {...}
    
    # 2. FHIR 轉換
    from src.medical_standards.fhir_adapter import FHIRAdapter
    adapter = FHIRAdapter()
    patient_data = adapter.parse_bundle(fhir_bundle)
    
    # 3. 執行推理
    # ...
EOF

# 執行測試
pytest tests/integration/test_full_pipeline.py -v
```

**檢查點**:
- [ ] LLM 接口實現完成
- [ ] 端到端測試通過
- [ ] 所有模塊整合正常

---

### Week 8: 文檔與部署準備

#### 任務 9.1: 更新文檔
```bash
# 更新架構文檔
# 撰寫醫療整合指南
# 更新 API 參考
```

---

#### 任務 9.2: 最終驗證
```bash
# 執行完整測試套件
pytest tests/ -v --cov=src --cov-report=html

# 檢查程式碼品質
ruff check src/
mypy src/
lint-imports

# 生成覆蓋率報告
open htmlcov/index.html
```

**檢查點**:
- [ ] 測試覆蓋率 > 80%
- [ ] 所有檢查通過
- [ ] 文檔完整

---

### Phase 3 完成檢查 ✅

**必須達成的里程碑**:
- [ ] ✅ LLM 接口實現
- [ ] ✅ 端到端測試通過
- [ ] ✅ 文檔完整更新
- [ ] ✅ 程式碼品質檢查通過
- [ ] ✅ 部署準備就緒

**預計工作量**: 20-24 小時  
**實際工作量**: _____ 小時 (待填寫)

---

## 📊 總進度追蹤

### 整體進度

| Phase | 任務數 | 完成數 | 進度 | 狀態 |
|-------|--------|--------|------|------|
| Phase 1 | 12 | 0 | 0% | ⏸️ 待開始 |
| Phase 2 | 15 | 0 | 0% | ⏸️ 待開始 |
| Phase 3 | 8 | 0 | 0% | ⏸️ 待開始 |
| **總計** | **35** | **0** | **0%** | ⏸️ 待開始 |

---

## 🎯 關鍵里程碑

- [ ] **Milestone 1**: 核心架構完成 (Week 2 結束)
- [ ] **Milestone 2**: NLP 模塊可用 (Week 4 結束)
- [ ] **Milestone 3**: FHIR 整合完成 (Week 5 結束)
- [ ] **Milestone 4**: 全功能可用 (Week 8 結束)

---

## 📝 實施日誌

### 2025-11-04
- ✅ 創建實施計劃文檔
- ✅ 定義 Phase 1-3 詳細任務
- ⏸️ 等待開始實施

### YYYY-MM-DD
- (待填寫實施紀錄)

---

## 🚨 風險與應對

| 風險 | 機率 | 影響 | 應對策略 |
|------|------|------|----------|
| NLP 模型性能不足 | 中 | 高 | 準備降級方案 (關鍵字匹配) |
| FHIR 整合複雜度高 | 高 | 中 | 分階段實現,先支援核心資源 |
| 時間超支 | 中 | 中 | 調整 Phase 2 範圍,部分功能 Phase 3 實現 |

---

**文檔結束**
