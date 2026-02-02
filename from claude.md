非常感謝這三點重要補充！特別是**離線推理要求**是一個關鍵的架構性變更。讓我重新調整整個方案：

---

## 🔴 關鍵變更：離線推理架構

### **原設計的問題：**
```python
# ❌ 原設計依賴外部 LLM API
from openai import OpenAI
client = OpenAI(api_key="...")
response = client.chat.completions.create(...)
```

### **新的離線方案：**
```python
# ✅ 本地部署 LLM
from vllm import LLM
model = LLM(model="meta-llama/Llama-3.1-8B-Instruct", 
            tensor_parallel_size=1)
```

---

## 📋 更新後的技術棧（高可靠性版本）

### **核心原則：**
- ✅ GitHub Stars > 1000
- ✅ 論文引用 > 50（如適用）
- ✅ 最近6個月有活躍更新
- ✅ 完整文檔與測試覆蓋

### **1. 本地 LLM 方案（離線推理）**

| 模型 | 規模 | Stars | 優勢 | 醫療適用性 |
|------|------|-------|------|-----------|
| **Llama 3.1** | 8B/70B | 26k+ | Meta官方，社群最活躍 | ⭐⭐⭐⭐⭐ |
| **Mistral 7B** | 7B | 10k+ | 效能優秀，低資源 | ⭐⭐⭐⭐ |
| **Qwen 2.5** | 7B/14B/72B | 8k+ | 多語言，中文優秀 | ⭐⭐⭐⭐⭐ |
| **BioMistral** | 7B | 500+ | 醫療領域微調 | ⭐⭐⭐⭐⭐ |

**推薦配置：**
```yaml
# 離線推理配置
offline_llm:
  primary: "meta-llama/Llama-3.1-8B-Instruct"  # 主力模型
  medical: "BioMistral/BioMistral-7B"          # 醫療專用
  framework: "vllm"                             # 推理加速
  quantization: "GPTQ-4bit"                     # 降低記憶體
  max_memory:
    windows: "12GB"  # 預留4GB給圖模型
    dgx_spark: "64GB"  # 預留64GB給圖模型
```

**推理框架選擇：**

| 框架 | Stars | 特點 | Windows | ARM |
|------|-------|------|---------|-----|
| **vLLM** | 28k+ | 極快，PagedAttention | ✅ | ✅ |
| **llama.cpp** | 67k+ | CPU友好，GGUF量化 | ✅ | ✅ |
| **Transformers** | 134k+ | 穩定，官方支持 | ✅ | ✅ |

**最終選擇：vLLM（主） + llama.cpp（備用）**

---

### **2. 圖神經網路（可靠性優先）**

| 技術 | 論文/項目 | Stars/引用 | 狀態 | 選擇 |
|------|-----------|------------|------|------|
| **PyTorch Geometric** | - | 21k+ | 🟢 活躍 | ✅ 主力 |
| **DGL** | KDD'19 | 13k+ | 🟢 活躍 | ✅ 備用 |
| ~~Graphormer~~ | ICLR'22 | 2k+ | 🔴 維護模式 | ❌ 棄用 |
| **Graph Transformer (DIY)** | 多篇綜述 | - | - | ✅ 自實現 |

**決策：**
- ✅ 使用 **PyG** 作為基礎框架（成熟穩定）
- ✅ **自實現** Graph Transformer 層（參考 GPS/Graphormer 論文）
- ❌ **放棄** Graphormer 官方代碼（已停止維護）

---

### **3. 注意力機制（跨平台兼容）**

| 實現 | Stars | 跨平台 | 速度 | 可靠性 |
|------|-------|--------|------|--------|
| **FlashAttention-2** | 13k+ | ⚠️ x86 only | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **xformers** | 8k+ | ⚠️ 編譯複雜 | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **PyTorch SDPA** | - | ✅ 原生 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

**策略：自適應三層降級**
```python
# 優先級：FlashAttn-2 > xformers > PyTorch SDPA
class AdaptiveAttentionBackend:
    def __init__(self):
        self.backend = self._detect()  # 自動選擇最佳
```

---

### **4. 本體處理（醫療核心）**

| 技術 | Stars | 維護 | 醫療適用 |
|------|-------|------|----------|
| **owlready2** | 300+ | 🟢 活躍 | ⭐⭐⭐⭐⭐ |
| **pronto** | 200+ | 🟢 活躍 | ⭐⭐⭐⭐ |

**選擇：owlready2（功能更完整）**

---

### **5. 向量檢索（跨平台）**

| 實現 | Stars | x86 GPU | ARM | 選擇 |
|------|-------|---------|-----|------|
| **Voyager** | 2k+ | ✅ (CPU) | ✅ | ✅ 跨平台 |
| **cuVS** | RAPIDS | ✅ (GPU) | ✅ | ✅ Linux GPU |
| ~~FAISS~~ | 31k+ | ✅ | ❌ | ❌ 已棄用 |
| ~~hnswlib~~ | 4k+ | ✅ | ✅ | ❌ 已棄用 |

**策略 (v3.2)：cuVS (Linux GPU) + Voyager (跨平台 fallback)**

---

## 🏗️ 模塊化架構設計（詳細版）

### **命名規範標準：**

```yaml
# 模組命名規範
modules:
  - 格式: "{category}_{function}.py"
  - 範例: "kg_builder.py", "model_gnn.py"
  
classes:
  - 格式: "PascalCase"
  - 範例: "OntologyKnowledgeBase", "GraphTransformerEncoder"
  
functions:
  - 格式: "snake_case"
  - 範例: "load_ontology", "compute_embeddings"
  
constants:
  - 格式: "UPPER_SNAKE_CASE"
  - 範例: "DEFAULT_HIDDEN_DIM", "MAX_SEQUENCE_LENGTH"
```

### **目錄結構（高度模塊化）：**

```
src/
├── config/
│   ├── __init__.py
│   ├── base_config.py          # 基礎配置類
│   ├── model_config.py         # 模型配置
│   ├── data_config.py          # 數據配置
│   └── deployment_config.py    # 部署配置
│
├── ontology/                   # 本體處理模組
│   ├── __init__.py
│   ├── loader.py               # 本體載入
│   ├── hierarchy.py            # 層次結構處理
│   ├── constraints.py          # 約束規則
│   ├── similarity.py           # 相似度計算
│   └── validator.py            # 驗證器
│
├── kg/                         # 知識圖譜模組
│   ├── __init__.py
│   ├── builder.py              # 圖構建器
│   ├── data_loader.py          # 資料源載入
│   ├── preprocessor.py         # 預處理
│   ├── hypergraph.py           # 超圖處理
│   └── storage/                # 存儲子模組
│       ├── file_storage.py     # 文件存儲
│       └── graph_db.py         # 圖資料庫
│
├── models/                     # 模型模組
│   ├── __init__.py
│   ├── gnn/                    # GNN子模組
│   │   ├── gat_layer.py        # GAT層
│   │   ├── graph_transformer.py # Graph Transformer
│   │   ├── hypergraph_conv.py  # 超圖卷積
│   │   └── message_passing.py  # 通用消息傳遞
│   │
│   ├── attention/              # 注意力機制
│   │   ├── adaptive_backend.py # 自適應後端
│   │   ├── flash_attention.py  # FlashAttn包裝
│   │   └── sparse_attention.py # 稀疏注意力
│   │
│   ├── encoders/               # 編碼器
│   │   ├── ontology_encoder.py # 本體編碼器
│   │   ├── patient_encoder.py  # 患者編碼器
│   │   └── temporal_encoder.py # 時序編碼器
│   │
│   ├── decoders/               # 解碼器
│   │   ├── distmult.py         # DistMult
│   │   ├── rotate.py           # RotatE
│   │   └── constrained_decoder.py # 約束解碼器
│   │
│   └── tasks/                  # 任務頭
│       ├── gene_ranking.py     # 基因排序
│       ├── disease_prediction.py # 疾病預測
│       └── patient_similarity.py # 患者相似度
│
├── retrieval/                  # 檢索模組
│   ├── __init__.py
│   ├── vector_index.py         # 向量索引（cuVS/Voyager auto-select）
│   ├── path_retriever.py       # 路徑檢索器
│   ├── path_scorer.py          # 路徑評分
│   └── subgraph_sampler.py     # 子圖採樣
│
├── llm/                        # 本地LLM模組 🆕
│   ├── __init__.py
│   ├── model_loader.py         # 模型載入器
│   ├── inference_engine.py     # 推理引擎（vLLM/llama.cpp）
│   ├── prompt_templates.py     # Prompt模板
│   └── graph_rag.py            # GraphRAG實現
│
├── reasoning/                  # 推理模組
│   ├── __init__.py
│   ├── path_reasoning.py       # 路徑推理
│   ├── evidence_extractor.py   # 證據提取
│   └── explanation_generator.py # 解釋生成（本地LLM）
│
├── training/                   # 訓練模組
│   ├── __init__.py
│   ├── trainer.py              # 訓練器
│   ├── loss_functions.py       # 損失函數
│   ├── metrics.py              # 評估指標
│   └── callbacks.py            # 訓練回調
│
├── inference/                  # 推理模組
│   ├── __init__.py
│   ├── pipeline.py             # 推理流程
│   ├── batch_processor.py      # 批次處理
│   └── result_formatter.py     # 結果格式化
│
└── utils/                      # 工具模組
    ├── __init__.py
    ├── logging.py              # 日誌系統
    ├── platform_detector.py    # 平台檢測
    ├── device_manager.py       # 設備管理
    └── data_structures.py      # 通用數據結構
```

---

## 🔧 模組接口設計（統一標準）

### **1. 基礎接口類：**

```python
# src/core/interfaces.py

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import torch

class BaseModule(ABC):
    """所有模組的基礎類"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = self._detect_device()
        self.logger = self._setup_logger()
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        """前向傳播（必須實現）"""
        pass
    
    def _detect_device(self) -> torch.device:
        """檢測設備"""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _setup_logger(self):
        """設置日誌"""
        from src.utils.logging import get_logger
        return get_logger(self.__class__.__name__)

class OntologyProcessor(ABC):
    """本體處理器接口"""
    
    @abstractmethod
    def load(self, path: str):
        """載入本體"""
        pass
    
    @abstractmethod
    def query(self, node_id: str) -> Dict:
        """查詢節點信息"""
        pass

class GraphBuilder(ABC):
    """圖構建器接口"""
    
    @abstractmethod
    def build(self, data_sources: List[str]) -> Any:
        """構建圖"""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """保存圖"""
        pass

class Retriever(ABC):
    """檢索器接口"""
    
    @abstractmethod
    def index(self, embeddings: torch.Tensor):
        """建立索引"""
        pass
    
    @abstractmethod
    def search(self, query: torch.Tensor, k: int) -> tuple:
        """搜索"""
        pass
```

### **2. 配置管理系統：**

```python
# src/config/base_config.py

from dataclasses import dataclass, field
from typing import Optional, List
import yaml

@dataclass
class ModelConfig:
    """模型配置"""
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    attention_backend: str = "auto"  # "flash", "xformers", "sdpa", "auto"
    
@dataclass
class DataConfig:
    """數據配置"""
    data_root: str = "data/"
    kg_path: str = "data/processed/kg.pt"
    ontology_path: str = "data/raw/ontologies/"
    batch_size: int = 32
    num_workers: int = 4

@dataclass
class LLMConfig:
    """本地LLM配置 🆕"""
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    framework: str = "vllm"  # "vllm", "llama.cpp", "transformers"
    quantization: Optional[str] = "GPTQ-4bit"  # None, "GPTQ-4bit", "GGUF-Q4"
    max_tokens: int = 512
    temperature: float = 0.7
    offline_mode: bool = True  # 🔴 強制離線

@dataclass
class DeploymentConfig:
    """部署配置"""
    platform: str = "auto"  # "windows_x86", "linux_arm", "auto"
    use_gpu: bool = True
    max_memory_gb: Optional[int] = None
    log_level: str = "INFO"

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: str = "config/default.yaml"):
        self.config_path = config_path
        self.model = ModelConfig()
        self.data = DataConfig()
        self.llm = LLMConfig()
        self.deployment = DeploymentConfig()
        
        if config_path:
            self.load(config_path)
    
    def load(self, path: str):
        """從YAML載入配置"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        if 'model' in config_dict:
            self.model = ModelConfig(**config_dict['model'])
        if 'data' in config_dict:
            self.data = DataConfig(**config_dict['data'])
        if 'llm' in config_dict:
            self.llm = LLMConfig(**config_dict['llm'])
        if 'deployment' in config_dict:
            self.deployment = DeploymentConfig(**config_dict['deployment'])
    
    def save(self, path: str):
        """保存配置到YAML"""
        config_dict = {
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'llm': self.llm.__dict__,
            'deployment': self.deployment.__dict__
        }
        with open(path, 'w') as f:
            yaml.dump(config_dict, f)
```

### **3. 統一的數據結構：**

```python
# src/utils/data_structures.py

from dataclasses import dataclass
from typing import List, Dict, Optional
import torch

@dataclass
class PatientData:
    """患者數據結構"""
    patient_id: str
    phenotypes: List[str]  # HPO IDs
    age: Optional[int] = None
    sex: Optional[str] = None
    medical_history: Optional[List[str]] = None
    
    def to_dict(self) -> Dict:
        return {
            'patient_id': self.patient_id,
            'phenotypes': self.phenotypes,
            'age': self.age,
            'sex': self.sex,
            'medical_history': self.medical_history
        }

@dataclass
class DiagnosticPath:
    """診斷路徑結構"""
    nodes: List[str]  # 節點序列
    relations: List[str]  # 關係類型
    confidence: float
    evidence: List[str]  # 證據來源（PMID等）
    
@dataclass
class DiagnosisResult:
    """診斷結果結構"""
    patient_id: str
    candidate_genes: List[Dict]  # [{'gene_id': ..., 'score': ..., 'rank': ...}]
    candidate_diseases: List[Dict]
    diagnostic_paths: List[DiagnosticPath]
    explanation: Optional[str] = None  # 本地LLM生成
    confidence: float = 0.0
```

---

## 🚀 本地LLM整合方案（離線推理）

### **架構設計：**

```python
# src/llm/offline_llm_engine.py

from typing import List, Optional
import torch
from dataclasses import dataclass

@dataclass
class LLMConfig:
    model_path: str
    framework: str = "vllm"  # "vllm", "llama.cpp"
    quantization: Optional[str] = "GPTQ-4bit"
    max_memory_gb: int = 12
    tensor_parallel_size: int = 1

class OfflineLLMEngine:
    """離線LLM推理引擎"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """載入本地模型"""
        if self.config.framework == "vllm":
            from vllm import LLM, SamplingParams
            self.model = LLM(
                model=self.config.model_path,
                tensor_parallel_size=self.config.tensor_parallel_size,
                quantization=self.config.quantization,
                max_model_len=4096,
                gpu_memory_utilization=0.8
            )
            self.sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=512
            )
        
        elif self.config.framework == "llama.cpp":
            from llama_cpp import Llama
            self.model = Llama(
                model_path=self.config.model_path,
                n_gpu_layers=-1,  # 全部加載到GPU
                n_ctx=4096
            )
        
        else:  # transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
    
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """生成文本"""
        if self.config.framework == "vllm":
            outputs = self.model.generate([prompt], self.sampling_params)
            return outputs[0].outputs[0].text
        
        elif self.config.framework == "llama.cpp":
            output = self.model(prompt, max_tokens=max_tokens)
            return output['choices'][0]['text']
        
        else:  # transformers
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class GraphRAGOffline:
    """離線GraphRAG系統"""
    
    def __init__(self, llm_engine: OfflineLLMEngine, kg, path_retriever):
        self.llm = llm_engine
        self.kg = kg
        self.path_retriever = path_retriever
    
    def diagnose_with_explanation(
        self,
        patient_phenotypes: List[str],
        top_k_paths: int = 5
    ) -> Dict:
        """使用本地LLM生成診斷解釋"""
        
        # 1. 檢索診斷路徑
        paths = self.path_retriever.retrieve(patient_phenotypes, k=top_k_paths)
        
        # 2. 構建Prompt
        prompt = self._build_diagnostic_prompt(patient_phenotypes, paths)
        
        # 3. 本地LLM生成解釋
        explanation = self.llm.generate(prompt, max_tokens=512)
        
        return {
            'paths': paths,
            'explanation': explanation,
            'model': self.llm.config.model_path
        }
    
    def _build_diagnostic_prompt(self, phenotypes, paths):
        """構建診斷Prompt"""
        prompt = f"""你是一位醫療診斷專家。基於以下患者症狀和知識圖譜路徑，生成診斷解釋。

患者症狀：
{self._format_phenotypes(phenotypes)}

診斷路徑：
{self._format_paths(paths)}

請提供：
1. 最可能的診斷
2. 推理過程
3. 支持證據

診斷解釋："""
        return prompt
```

### **本地LLM部署配置：**

```yaml
# config/llm_offline.yaml

llm:
  # Windows 環境（16GB VRAM，需留4GB給圖模型）
  windows:
    model_name: "meta-llama/Llama-3.1-8B-Instruct"
    framework: "vllm"
    quantization: "GPTQ-4bit"  # 約佔6-8GB
    max_memory_gb: 12
    tensor_parallel_size: 1
  
  # DGX Spark 環境（128GB統一記憶體，可用更大模型）
  dgx_spark:
    model_name: "meta-llama/Llama-3.1-70B-Instruct"
    framework: "vllm"
    quantization: "GPTQ-4bit"  # 約佔35-40GB
    max_memory_gb: 64
    tensor_parallel_size: 1
  
  # 醫療專用模型（備選）
  medical:
    model_name: "BioMistral/BioMistral-7B"
    framework: "vllm"
    quantization: "GPTQ-4bit"

# 離線推理強制設置
offline_mode: true
allow_external_api: false  # 🔴 禁止外部API
```

---

## 📊 更新後的技術選型總結

### **最終技術棧（高可靠性 + 離線 + 模塊化）：**

| 組件 | 技術選型 | Stars | 狀態 | 理由 |
|------|----------|-------|------|------|
| **深度學習框架** | PyTorch 2.8 | 82k+ | 🟢 | 官方，穩定 |
| **圖學習** | PyTorch Geometric | 21k+ | 🟢 | 成熟，社群活躍 |
| **圖學習備用** | DGL | 13k+ | 🟢 | 備用方案 |
| **注意力** | 自適應三層 | - | 🟢 | 跨平台兼容 |
| **本體** | owlready2 | 300+ | 🟢 | 醫療標準 |
| **向量檢索** | cuVS/Voyager | RAPIDS/2k+ | 🟢 | GPU加速+跨平台 |
| **本地LLM** | vLLM + Llama 3.1 | 28k+/26k+ | 🟢 | 離線推理 |
| **推理加速** | vLLM/llama.cpp | 28k+/67k+ | 🟢 | 高效 |

### **放棄的技術（不穩定/已停止維護）：**

| 技術 | 原因 | 替代方案 |
|------|------|----------|
| ❌ Graphormer官方代碼 | 維護模式 | 自實現Graph Transformer |
| ❌ 外部LLM API | 離線限制 | 本地vLLM |
| ❌ 在線Pubtator API | 離線限制 | 預下載數據 |

---

## ✅ 行動計劃更新

### **Week 1: 基礎設施 + 本地LLM測試**
```bash
# 1. 環境設置
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# 2. 安裝核心依賴
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install torch-geometric vllm transformers

# 3. 測試本地LLM
python tests/test_offline_llm.py
```

### **Week 2-3: 模塊化架構實現**
- ✅ 實現所有基礎接口類
- ✅ 配置管理系統
- ✅ 本體處理模組
- ✅ 開始知識圖譜構建

---

## 🎯 關鍵決策確認

請確認以下決策：

1. ✅ **本地LLM方案**：使用 vLLM + Llama 3.1 8B（Windows）/ 70B（DGX Spark）
2. ✅ **模塊化設計**：採用上述詳細的目錄結構和接口規範
3. ✅ **技術選型**：放棄Graphormer官方代碼，自實現Graph Transformer
4. ⚠️ **需要確認**：DGX Spark的128GB記憶體是否足夠運行70B模型 + 圖模型？

**你同意這個更新後的方案嗎？** 有任何需要調整的地方請告訴我！