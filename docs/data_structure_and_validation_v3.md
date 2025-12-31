# SHEPHERD-Advanced 資料結構與項目校驗設計 v3.0

**版本**: 3.0  
**日期**: 2025-11-04  
**狀態**: 整合醫生團隊建議 + 原有架構優化  
**變更重點**: 
1. 新增 NLP/FHIR 模塊架構
2. 強化 metadata 版本追溯
3. 統一 schema 驗證系統
4. 預留醫療標準接口

---

## 📋 目錄

1. [核心原則](#核心原則)
2. [完整目錄結構](#完整目錄結構)
3. [資料結構設計](#資料結構設計)
4. [項目校驗系統](#項目校驗系統)
5. [醫生團隊功能整合](#醫生團隊功能整合)
6. [擴充接口設計](#擴充接口設計)
7. [實施檢查清單](#實施檢查清單)

---

## 🎯 核心原則

### 1. 模塊化與可維護性
- ✅ **單一職責**: 每個模塊只負責一件事
- ✅ **接口分離**: 清晰的模塊邊界與依賴方向
- ✅ **統一命名**: 遵循 PEP 8 + 醫療領域慣例
- ✅ **版本追溯**: 所有資料和模型可完整追溯

### 2. 醫療合規性
- ✅ **資料血統**: 完整記錄資料來源和處理過程
- ✅ **版本管理**: semantic versioning + SHA256 校驗
- ✅ **安全隔離**: 敏感資料加密存儲
- ✅ **審計日誌**: 所有推理過程可審查

### 3. 跨平台一致性
- ✅ **配置統一**: 單一真實來源 (SSOT)
- ✅ **平台檢測**: 自動適配 x86/ARM 差異
- ✅ **降級方案**: 優雅處理環境限制

---

## 📂 完整目錄結構

```
shepherd-advanced/
├── .github/                        # CI/CD 配置
│   └── workflows/
│       ├── test-x86.yml
│       ├── test-arm.yml
│       └── deploy.yml
│
├── configs/                        # ✨ 配置文件 (統一管理)
│   ├── schemas/                    # 🆕 JSON Schema 驗證規則
│   │   ├── base_config.schema.json
│   │   ├── model_config.schema.json
│   │   ├── data_config.schema.json
│   │   ├── patient_input.schema.json      # 🆕 患者輸入格式
│   │   └── inference_output.schema.json   # 🆕 推理輸出格式
│   │
│   ├── base_config.yaml            # 基礎配置
│   ├── model_config.yaml           # 模型配置
│   ├── data_config.yaml            # 資料配置
│   ├── deployment_config.yaml      # 部署配置
│   └── medical_standards.yaml      # 🆕 醫療標準映射 (HPO/ICD/FHIR)
│
├── data/                           # ✨ 資料目錄
│   ├── raw/                        # 原始資料
│   │   ├── ontologies/
│   │   │   ├── hpo.obo
│   │   │   ├── mondo.owl
│   │   │   └── go.obo
│   │   ├── kg_sources/
│   │   │   ├── disgenet/
│   │   │   ├── clinvar/
│   │   │   └── omim/
│   │   └── patient_records/        # 🆕 患者原始資料
│   │       ├── fhir/                # 🆕 FHIR 格式資料
│   │       └── hiss/                # 🆕 HISS 格式資料
│   │
│   ├── processed/                  # 處理後資料
│   │   ├── knowledge_graph/
│   │   │   ├── metadata.json       # ✨ 增強版 metadata (下詳)
│   │   │   ├── VERSION             # 語義化版本號
│   │   │   ├── hetero_graph.pt
│   │   │   ├── hypergraph.pt
│   │   │   └── embeddings/
│   │   │
│   │   ├── ontology_cache/         # 本體快取
│   │   │   ├── hpo_hierarchy.pkl
│   │   │   ├── mondo_hierarchy.pkl
│   │   │   └── constraints.json
│   │   │
│   │   └── nlp_extractions/        # 🆕 NLP 提取結果快取
│   │       ├── symptom_cache.db    # 症狀提取快取
│   │       └── entity_mappings.json
│   │
│   └── external/                   # 大型外部資料 (不納入版控)
│       └── .gitignore
│
├── models/                         # ✨ 模型目錄
│   ├── production/
│   │   ├── registry.json           # 🆕 模型註冊表
│   │   ├── checkpoint_v1.0.0/
│   │   │   ├── model.pt
│   │   │   ├── metadata.json       # 模型元資料
│   │   │   └── config.yaml
│   │   └── current -> checkpoint_v1.0.0  # 符號連結
│   │
│   ├── checkpoints/                # 訓練檢查點
│   ├── pretrained/                 # 預訓練模型
│   │   └── scibertuncased/        # 🆕 NLP 預訓練模型
│   └── experiments/                # 實驗模型
│
├── src/                            # ✨ 源代碼
│   ├── config/                     # 配置模塊
│   │   ├── __init__.py
│   │   ├── base_config.py
│   │   ├── config_validator.py     # 🆕 配置驗證器
│   │   └── schema_loader.py        # 🆕 Schema 載入器
│   │
│   ├── ontology/                   # 本體處理
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── hierarchy.py
│   │   ├── constraints.py
│   │   ├── similarity.py
│   │   └── validator.py
│   │
│   ├── kg/                         # 知識圖譜
│   │   ├── __init__.py
│   │   ├── builder.py
│   │   ├── data_loader.py
│   │   ├── preprocessor.py
│   │   ├── hypergraph.py
│   │   ├── entity_linker.py        # 🆕 實體連結
│   │   └── storage/
│   │       ├── file_storage.py
│   │       └── graph_db.py
│   │
│   ├── nlp/                        # 🆕 自然語言處理模塊
│   │   ├── __init__.py
│   │   ├── symptom_extractor.py   # Free text → HPO terms
│   │   ├── entity_recognizer.py   # 醫療實體識別
│   │   ├── clinical_bert.py       # ClinicalBERT 包裝
│   │   └── hpo_matcher.py         # HPO 術語匹配
│   │
│   ├── medical_standards/          # 🆕 醫療標準接口
│   │   ├── __init__.py
│   │   ├── fhir_adapter.py        # FHIR 適配器
│   │   ├── hiss_adapter.py        # HISS 適配器
│   │   ├── icd_mapper.py          # ICD-10/11 映射
│   │   └── snomed_mapper.py       # SNOMED CT 映射
│   │
│   ├── models/                     # 模型架構
│   │   ├── __init__.py
│   │   ├── gnn/
│   │   ├── attention/
│   │   ├── encoders/
│   │   ├── decoders/
│   │   └── tasks/
│   │
│   ├── retrieval/                  # 檢索模塊
│   │   ├── __init__.py
│   │   ├── vector_index.py
│   │   ├── path_retriever.py
│   │   ├── path_scorer.py
│   │   └── subgraph_sampler.py
│   │
│   ├── reasoning/                  # 推理模塊
│   │   ├── __init__.py
│   │   ├── path_reasoning.py
│   │   ├── constraint_checker.py
│   │   └── explanation_generator.py
│   │
│   ├── llm/                        # 本地 LLM
│   │   ├── __init__.py
│   │   ├── interface.py           # 🆕 LLM 接口定義
│   │   ├── model_loader.py
│   │   ├── inference_engine.py
│   │   └── prompt_templates.py
│   │
│   ├── inference/                  # 推理管道
│   │   ├── __init__.py
│   │   ├── pipeline.py
│   │   ├── schemas.py              # 🆕 共享資料結構 (SSOT)
│   │   ├── input_validator.py      # 🆕 輸入驗證
│   │   └── output_formatter.py     # 🆕 輸出格式化
│   │
│   ├── training/                   # 訓練模塊
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── loss_functions.py
│   │   └── callbacks.py
│   │
│   ├── utils/                      # 工具函數
│   │   ├── __init__.py
│   │   ├── platform_detector.py
│   │   ├── version_checker.py      # 🆕 版本兼容性檢查
│   │   ├── hash_generator.py       # 🆕 資料哈希生成
│   │   ├── logger.py
│   │   └── metrics.py
│   │
│   ├── api/                        # FastAPI 後端
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── routes/
│   │   │   ├── inference.py
│   │   │   ├── training.py
│   │   │   └── health.py
│   │   └── middleware/
│   │       ├── auth.py
│   │       └── logging.py
│   │
│   └── webui/                      # Gradio 前端
│       ├── __init__.py
│       ├── app.py
│       └── components/
│           ├── input_form.py       # 🆕 智能表單
│           ├── hpo_search.py       # 🆕 HPO 搜尋
│           └── result_viewer.py
│
├── tests/                          # ✨ 測試
│   ├── unit/                       # 單元測試
│   │   ├── test_ontology.py
│   │   ├── test_kg.py
│   │   ├── test_nlp.py             # 🆕
│   │   └── test_medical_standards.py # 🆕
│   │
│   ├── integration/                # 整合測試
│   │   ├── test_pipeline.py
│   │   └── test_fhir_workflow.py   # 🆕
│   │
│   ├── benchmarks/                 # 基準測試
│   │   └── platform_specific/
│   │       ├── test_attention_x86.py
│   │       ├── test_attention_arm.py
│   │       ├── test_vector_index_x86.py
│   │       └── test_vector_index_arm.py
│   │
│   └── fixtures/                   # 測試資料
│       ├── sample_patients.json    # 🆕 範例患者資料
│       └── sample_fhir.json        # 🆕 範例 FHIR 資料
│
├── scripts/                        # ✨ 腳本
│   ├── download_data.py
│   ├── preprocess_data.py
│   ├── build_knowledge_graph.py
│   ├── train_model.py
│   ├── validate_installation.py    # 🆕 安裝驗證
│   └── medical_standards/          # 🆕 醫療標準工具
│       ├── convert_fhir_to_internal.py
│       └── export_to_fhir.py
│
├── docs/                           # ✨ 文檔
│   ├── architecture_v3.md          # 🆕 更新架構文檔
│   ├── medical_integration.md      # 🆕 醫療系統整合指南
│   ├── api_reference.md
│   └── developer_guide.md
│
├── logs/                           # 日誌
├── reports/                        # 報告輸出
│
├── pyproject.toml                  # 🆕 Python 專案配置
├── .import-linter.ini              # 🆕 依賴規則約束
├── .pre-commit-config.yaml         # 🆕 Git hooks
├── .gitignore
├── README.md
└── LICENSE
```

---

## 📊 資料結構設計

### 1. 增強版知識圖譜 Metadata

**位置**: `data/processed/knowledge_graph/metadata.json`

```json
{
  "schema_version": "3.0",
  "data_version": "2025.11.04",
  "creation_timestamp": "2025-11-04T10:30:00Z",
  
  "generator": {
    "script": "scripts/build_knowledge_graph.py",
    "commit_sha": "a3f5b2c1234567890abcdef",
    "git_branch": "main",
    "python_version": "3.12.0",
    "torch_version": "2.8.0",
    "pyg_version": "2.6.0"
  },
  
  "data_sources": {
    "hpo": {
      "version": "2025-09-01",
      "url": "http://purl.obolibrary.org/obo/hp.obo",
      "download_date": "2025-11-01",
      "sha256": "abc123..."
    },
    "mondo": {
      "version": "2025-08-15",
      "url": "http://purl.obolibrary.org/obo/mondo.owl",
      "download_date": "2025-11-01",
      "sha256": "def456..."
    },
    "disgenet": {
      "version": "v7.0",
      "url": "https://www.disgenet.org/downloads",
      "download_date": "2025-11-01",
      "sha256": "ghi789..."
    }
  },
  
  "preprocessing": {
    "steps": [
      "ontology_alignment",
      "entity_deduplication",
      "edge_normalization",
      "hypergraph_construction"
    ],
    "parameters": {
      "min_edge_confidence": 0.5,
      "k_hop_subgraph": 3,
      "hyperedge_min_support": 10
    }
  },
  
  "statistics": {
    "num_nodes": 500000,
    "num_edges": 2000000,
    "node_types": {
      "gene": 25000,
      "disease": 15000,
      "phenotype": 130000,
      "pathway": 2500
    },
    "edge_types": {
      "gene_disease": 450000,
      "phenotype_disease": 800000,
      "gene_pathway": 120000
    },
    "hyperedges": 5000,
    "avg_degree": 8.5,
    "connected_components": 1
  },
  
  "validation": {
    "ontology_constraints_passed": true,
    "no_self_loops": true,
    "no_duplicate_edges": true,
    "all_nodes_reachable": true
  },
  
  "data_hash": {
    "graph_structure": "sha256:abcd1234...",
    "node_features": "sha256:efgh5678...",
    "edge_features": "sha256:ijkl9012..."
  },
  
  "compatibility": {
    "min_python": "3.10",
    "min_torch": "2.5.0",
    "min_pyg": "2.5.0",
    "platforms": ["x86_64", "aarch64"]
  }
}
```

### 2. 模型註冊表

**位置**: `models/production/registry.json`

```json
{
  "registry_version": "1.0",
  "models": [
    {
      "model_id": "shepherd-v1.0.0",
      "type": "ontology_aware_gnn",
      "status": "production",
      "created_at": "2025-11-04T10:00:00Z",
      
      "training_data": {
        "kg_version": "2025.11.04",
        "kg_hash": "sha256:abcd1234...",
        "patient_dataset": "synthetic_1000",
        "split": {
          "train": 800,
          "val": 100,
          "test": 100
        }
      },
      
      "hyperparameters": {
        "hidden_dim": 512,
        "num_layers": 4,
        "attention_heads": 8,
        "dropout": 0.1,
        "learning_rate": 0.0001,
        "batch_size": 32
      },
      
      "performance": {
        "val_mrr": 0.856,
        "val_hits@10": 0.923,
        "test_mrr": 0.832,
        "test_hits@10": 0.911,
        "inference_time_ms": 150
      },
      
      "compatible_data_versions": ["2025.11.04", "2025.10.*"],
      "platforms": ["x86_cuda", "arm_cuda"],
      
      "files": {
        "model_weights": "models/production/checkpoint_v1.0.0/model.pt",
        "config": "models/production/checkpoint_v1.0.0/config.yaml",
        "metadata": "models/production/checkpoint_v1.0.0/metadata.json"
      }
    }
  ],
  "current_production": "shepherd-v1.0.0"
}
```

### 3. 患者輸入 Schema

**位置**: `configs/schemas/patient_input.schema.json`

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Patient Input Schema",
  "description": "標準化患者輸入格式,支援多種資料類型",
  "type": "object",
  "required": ["patient_id", "phenotypes"],
  
  "properties": {
    "patient_id": {
      "type": "string",
      "pattern": "^P[0-9]{5,10}$",
      "description": "患者唯一識別碼"
    },
    
    "input_type": {
      "type": "string",
      "enum": ["structured", "free_text", "fhir", "hiss"],
      "default": "structured",
      "description": "輸入資料類型"
    },
    
    "phenotypes": {
      "type": "array",
      "minItems": 1,
      "items": {
        "oneOf": [
          {
            "type": "string",
            "pattern": "^HP:[0-9]{7}$",
            "description": "標準 HPO term (HP:1234567)"
          },
          {
            "type": "object",
            "properties": {
              "text": {"type": "string"},
              "hpo_id": {"type": "string", "pattern": "^HP:[0-9]{7}$"},
              "confidence": {"type": "number", "minimum": 0, "maximum": 1}
            },
            "required": ["text", "hpo_id"]
          }
        ]
      }
    },
    
    "free_text_symptoms": {
      "type": "string",
      "description": "自由文字症狀描述 (需 NLP 處理)"
    },
    
    "genetic_data": {
      "type": "object",
      "properties": {
        "variants": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "gene": {"type": "string"},
              "variant": {"type": "string"},
              "clinvar_id": {"type": "string"},
              "pathogenicity": {
                "type": "string",
                "enum": ["pathogenic", "likely_pathogenic", "uncertain", "likely_benign", "benign"]
              }
            }
          }
        },
        "gene_panel": {
          "type": "array",
          "items": {"type": "string"},
          "description": "已檢測基因清單"
        },
        "wgs_available": {"type": "boolean"}
      }
    },
    
    "diagnoses": {
      "type": "object",
      "properties": {
        "icd10": {
          "type": "array",
          "items": {"type": "string", "pattern": "^[A-Z][0-9]{2}\\.[0-9]{1,2}$"}
        },
        "icd11": {
          "type": "array",
          "items": {"type": "string"}
        },
        "snomed": {
          "type": "array",
          "items": {"type": "string", "pattern": "^[0-9]+$"}
        },
        "mondo": {
          "type": "array",
          "items": {"type": "string", "pattern": "^MONDO:[0-9]{7}$"}
        }
      }
    },
    
    "medical_history": {
      "type": "object",
      "properties": {
        "family_history": {
          "type": "array",
          "items": {"type": "string"}
        },
        "lab_results": {
          "type": "object",
          "patternProperties": {
            ".*": {
              "type": "object",
              "properties": {
                "value": {"type": "number"},
                "unit": {"type": "string"},
                "normal_range": {
                  "type": "array",
                  "items": {"type": "number"},
                  "minItems": 2,
                  "maxItems": 2
                },
                "date": {"type": "string", "format": "date-time"}
              }
            }
          }
        },
        "medications": {
          "type": "array",
          "items": {"type": "string"}
        }
      }
    },
    
    "visit_history": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "date": {"type": "string", "format": "date"},
          "phenotypes": {
            "type": "array",
            "items": {"type": "string", "pattern": "^HP:[0-9]{7}$"}
          },
          "severity": {
            "type": "string",
            "enum": ["mild", "moderate", "severe"]
          }
        },
        "required": ["date", "phenotypes"]
      }
    },
    
    "fhir_bundle": {
      "type": "object",
      "description": "完整的 FHIR Bundle (若使用 FHIR 輸入)"
    },
    
    "demographics": {
      "type": "object",
      "properties": {
        "age": {"type": "integer", "minimum": 0, "maximum": 150},
        "gender": {"type": "string", "enum": ["male", "female", "other", "unknown"]},
        "ethnicity": {"type": "string"}
      }
    }
  }
}
```

### 4. 推理輸出 Schema

**位置**: `configs/schemas/inference_output.schema.json`

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Inference Output Schema",
  "description": "標準化推理輸出格式",
  "type": "object",
  "required": ["patient_id", "timestamp", "top_candidates", "metadata"],
  
  "properties": {
    "patient_id": {"type": "string"},
    "timestamp": {"type": "string", "format": "date-time"},
    "inference_time_ms": {"type": "number"},
    
    "top_candidates": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "required": ["disease", "confidence", "reasoning_path"],
        "properties": {
          "rank": {"type": "integer", "minimum": 1},
          "disease": {
            "type": "object",
            "properties": {
              "mondo_id": {"type": "string", "pattern": "^MONDO:[0-9]{7}$"},
              "name": {"type": "string"},
              "orphanet_id": {"type": "string"},
              "omim_id": {"type": "string"}
            },
            "required": ["mondo_id", "name"]
          },
          "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1
          },
          "supporting_genes": {
            "type": "array",
            "items": {"type": "string"}
          },
          "reasoning_path": {
            "type": "array",
            "items": {"type": "string"},
            "description": "可解釋的推理路徑"
          },
          "evidence": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "source": {"type": "string"},
                "reference": {"type": "string"},
                "confidence": {"type": "number"}
              }
            }
          },
          "ontology_validation": {
            "type": "object",
            "properties": {
              "passed": {"type": "boolean"},
              "violations": {"type": "array", "items": {"type": "string"}}
            }
          },
          "clinical_notes": {
            "type": "string",
            "description": "臨床建議 (由 LLM 生成)"
          }
        }
      }
    },
    
    "explanation": {
      "type": "string",
      "description": "整體推理解釋 (自然語言)"
    },
    
    "metadata": {
      "type": "object",
      "properties": {
        "model_version": {"type": "string"},
        "kg_version": {"type": "string"},
        "platform": {"type": "string"},
        "gpu_type": {"type": "string"}
      }
    },
    
    "warnings": {
      "type": "array",
      "items": {"type": "string"},
      "description": "推理過程中的警告訊息"
    }
  }
}
```

---

## 🔒 項目校驗系統

### 1. 工程化工具鏈

**位置**: `pyproject.toml`

```toml
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
]

[project.optional-dependencies]
nlp = [
    "transformers>=4.30.0",
    "scispacy>=0.5.0",
    "en-core-sci-sm @ https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_sm-0.5.0.tar.gz"
]
medical = [
    "fhir.resources>=7.0.0",
    "python-hl7>=0.4.0"
]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
    "import-linter>=1.12.0"
]

[tool.black]
line-length = 100
target-version = ['py310', 'py311', 'py312']

[tool.ruff]
line-length = 100
select = ["E", "F", "I", "N", "W"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.10"
strict = true
warn_return_any = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --cov=src --cov-report=html"
```

### 2. 依賴規則約束

**位置**: `.import-linter.ini`

```ini
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
    src.nlp                   # 新增層
    src.medical_standards     # 新增層
    src.models
    src.retrieval
    src.reasoning
    src.llm
    src.training
    src.inference
    src.api
    src.webui

[importlinter:contract:independence]
name = Keep modules independent
type = independence
modules =
    src.nlp
    src.medical_standards
    src.ontology
    src.kg

ignore_imports =
    src.models.* -> src.training.*
    src.retrieval.* -> src.reasoning.*
    src.api.* -> src.training.*
    src.webui.* -> src.training.*

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
```

### 3. 配置驗證器

**位置**: `src/config/config_validator.py`

```python
"""
配置驗證器 - 啟動前驗證所有配置檔案
"""
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any
from jsonschema import validate, ValidationError
import logging

logger = logging.getLogger(__name__)


class ConfigValidator:
    """配置文件驗證器"""
    
    def __init__(self, config_dir: Path, schema_dir: Path):
        self.config_dir = config_dir
        self.schema_dir = schema_dir
        self.schemas: Dict[str, Any] = {}
        self._load_schemas()
    
    def _load_schemas(self):
        """載入所有 JSON Schema"""
        for schema_file in self.schema_dir.glob("*.schema.json"):
            schema_name = schema_file.stem.replace(".schema", "")
            with open(schema_file) as f:
                self.schemas[schema_name] = json.load(f)
            logger.info(f"Loaded schema: {schema_name}")
    
    def validate_yaml_config(self, config_name: str) -> bool:
        """驗證 YAML 配置文件"""
        config_file = self.config_dir / f"{config_name}.yaml"
        
        if not config_file.exists():
            logger.error(f"Config file not found: {config_file}")
            return False
        
        # 載入 YAML
        with open(config_file) as f:
            config_data = yaml.safe_load(f)
        
        # 獲取對應 schema
        if config_name not in self.schemas:
            logger.warning(f"No schema found for {config_name}, skipping validation")
            return True
        
        # 驗證
        try:
            validate(instance=config_data, schema=self.schemas[config_name])
            logger.info(f"✅ {config_name}.yaml validation passed")
            return True
        except ValidationError as e:
            logger.error(f"❌ {config_name}.yaml validation failed: {e.message}")
            return False
    
    def validate_all(self) -> bool:
        """驗證所有配置檔案"""
        logger.info("Starting configuration validation...")
        
        configs_to_validate = [
            "base_config",
            "model_config",
            "data_config",
            "deployment_config"
        ]
        
        results = []
        for config_name in configs_to_validate:
            result = self.validate_yaml_config(config_name)
            results.append(result)
        
        if all(results):
            logger.info("✅ All configuration files are valid")
            return True
        else:
            logger.error("❌ Some configuration files have errors")
            return False


def main():
    """CLI 入口"""
    from src.config.base_config import Config
    
    config = Config()
    validator = ConfigValidator(
        config_dir=Path("configs"),
        schema_dir=Path("configs/schemas")
    )
    
    success = validator.validate_all()
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
```

### 4. 版本兼容性檢查器

**位置**: `src/utils/version_checker.py`

```python
"""
版本兼容性檢查器 - 確保模型與資料版本匹配
"""
import json
import hashlib
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class IncompatibleVersionError(Exception):
    """版本不兼容異常"""
    pass


class VersionChecker:
    """版本兼容性檢查器"""
    
    def __init__(self, models_dir: Path, data_dir: Path):
        self.models_dir = models_dir
        self.data_dir = data_dir
    
    def load_metadata(self, file_path: Path) -> Dict[str, Any]:
        """載入 metadata.json"""
        with open(file_path) as f:
            return json.load(f)
    
    def check_model_data_compatibility(
        self,
        model_version: str,
        data_version: str
    ) -> bool:
        """檢查模型與資料版本是否兼容"""
        
        # 載入模型 metadata
        model_meta_path = self.models_dir / model_version / "metadata.json"
        if not model_meta_path.exists():
            raise FileNotFoundError(f"Model metadata not found: {model_meta_path}")
        
        model_meta = self.load_metadata(model_meta_path)
        
        # 載入資料 metadata
        data_meta_path = self.data_dir / "processed" / "knowledge_graph" / "metadata.json"
        if not data_meta_path.exists():
            raise FileNotFoundError(f"Data metadata not found: {data_meta_path}")
        
        data_meta = self.load_metadata(data_meta_path)
        
        # 檢查資料哈希
        expected_hash = model_meta["training_data"]["kg_hash"]
        actual_hash = data_meta["data_hash"]["graph_structure"]
        
        if expected_hash != actual_hash:
            logger.error(
                f"Data hash mismatch!\n"
                f"  Model expects: {expected_hash}\n"
                f"  Current data:  {actual_hash}"
            )
            raise IncompatibleVersionError(
                f"Model {model_version} is not compatible with current data version"
            )
        
        logger.info(f"✅ Model {model_version} is compatible with data version {data_version}")
        return True
    
    def verify_installation(self) -> Dict[str, bool]:
        """驗證安裝完整性"""
        checks = {}
        
        # 檢查必要檔案
        required_files = [
            self.data_dir / "processed" / "knowledge_graph" / "metadata.json",
            self.models_dir / "production" / "registry.json",
        ]
        
        for file_path in required_files:
            checks[str(file_path)] = file_path.exists()
        
        return checks


def compute_file_hash(file_path: Path) -> str:
    """計算檔案 SHA256 哈希"""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"
```

### 5. 輸入驗證器

**位置**: `src/inference/input_validator.py`

```python
"""
輸入驗證器 - 驗證患者輸入格式
"""
import json
from pathlib import Path
from typing import Dict, Any, List
from jsonschema import validate, ValidationError
from pydantic import BaseModel, Field, validator
import logging

logger = logging.getLogger(__name__)


class PatientInput(BaseModel):
    """患者輸入資料模型 (使用 Pydantic)"""
    
    patient_id: str = Field(..., regex=r"^P[0-9]{5,10}$")
    input_type: str = Field("structured", regex=r"^(structured|free_text|fhir|hiss)$")
    phenotypes: List[str] = Field(..., min_items=1)
    free_text_symptoms: str = Field(None)
    genetic_data: Dict[str, Any] = Field(None)
    diagnoses: Dict[str, List[str]] = Field(None)
    medical_history: Dict[str, Any] = Field(None)
    visit_history: List[Dict[str, Any]] = Field(None)
    fhir_bundle: Dict[str, Any] = Field(None)
    demographics: Dict[str, Any] = Field(None)
    
    @validator('phenotypes', each_item=True)
    def validate_hpo_term(cls, v):
        """驗證 HPO term 格式"""
        if not v.startswith("HP:") or len(v) != 10:
            raise ValueError(f"Invalid HPO term format: {v}")
        return v
    
    class Config:
        extra = "forbid"  # 禁止額外欄位


class InputValidator:
    """輸入驗證器"""
    
    def __init__(self, schema_path: Path):
        with open(schema_path) as f:
            self.schema = json.load(f)
    
    def validate(self, patient_data: Dict[str, Any]) -> bool:
        """使用 JSON Schema 驗證"""
        try:
            validate(instance=patient_data, schema=self.schema)
            logger.info("✅ Patient input validation passed")
            return True
        except ValidationError as e:
            logger.error(f"❌ Patient input validation failed: {e.message}")
            raise
    
    def validate_pydantic(self, patient_data: Dict[str, Any]) -> PatientInput:
        """使用 Pydantic 驗證 (更嚴格)"""
        try:
            validated = PatientInput(**patient_data)
            logger.info("✅ Patient input Pydantic validation passed")
            return validated
        except Exception as e:
            logger.error(f"❌ Patient input Pydantic validation failed: {e}")
            raise
```

### 6. 平台特定測試

**位置**: `tests/benchmarks/platform_specific/test_attention_arm.py`

```python
"""
ARM 平台注意力機制回歸測試
"""
import torch
import pytest
from src.models.attention.adaptive_backend import AdaptiveAttentionBackend


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="Requires Blackwell GPU"
)
@pytest.mark.arm64
def test_cudnn_sdpa_smoke_arm():
    """ARM 平台 cuDNN SDPA 快速驗證"""
    device = torch.device("cuda")
    backend = AdaptiveAttentionBackend()
    
    # 建立測試張量
    batch_size, seq_len, d_model = 4, 128, 512
    q = torch.randn(batch_size, seq_len, d_model, device=device)
    k = torch.randn(batch_size, seq_len, d_model, device=device)
    v = torch.randn(batch_size, seq_len, d_model, device=device)
    
    # 執行 100 次推理
    for _ in range(100):
        output = backend.compute_attention(q, k, v)
    
    # 驗證結果
    assert torch.isfinite(output).all(), "Output contains NaN or Inf"
    assert output.shape == q.shape, "Output shape mismatch"


@pytest.mark.arm64
def test_platform_detection_arm():
    """驗證 ARM 平台檢測"""
    from src.utils.platform_detector import PlatformDetector
    
    detector = PlatformDetector()
    info = detector.get_platform_info()
    
    assert info["cpu_arch"] == "aarch64"
    assert "ARM" in info["cpu_model"]
```

---

## 🏥 醫生團隊功能整合

### 1. NLP 症狀提取模塊

**位置**: `src/nlp/symptom_extractor.py`

```python
"""
症狀提取器 - 從自由文字提取 HPO terms
狀態: 🟡 Phase 2 實現 (預留接口)
"""
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import logging

logger = logging.getLogger(__name__)


class SymptomExtractor:
    """從自由文字提取症狀並映射到 HPO"""
    
    def __init__(self, model_name: str = "allenai/scibert_scivocab_uncased"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._initialized = False
    
    def initialize(self):
        """延遲初始化 (避免不必要的模型載入)"""
        if self._initialized:
            return
        
        logger.info(f"Loading NLP model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
        self._initialized = True
        logger.info("NLP model loaded successfully")
    
    def extract_symptoms(
        self,
        free_text: str,
        confidence_threshold: float = 0.7
    ) -> List[Dict[str, any]]:
        """
        從自由文字提取症狀
        
        Args:
            free_text: 自由文字症狀描述
            confidence_threshold: 最低信心閾值
        
        Returns:
            List of {'text': str, 'hpo_id': str, 'confidence': float}
        
        Example:
            >>> extractor = SymptomExtractor()
            >>> result = extractor.extract_symptoms(
            ...     "患者8歲女孩,逐漸出現四肢無力,運動後心跳加速"
            ... )
            >>> print(result)
            [
                {'text': '四肢無力', 'hpo_id': 'HP:0003324', 'confidence': 0.89},
                {'text': '心跳加速', 'hpo_id': 'HP:0001649', 'confidence': 0.85}
            ]
        """
        if not self._initialized:
            self.initialize()
        
        # TODO: 實際 NER 推理 (Phase 2)
        logger.warning("SymptomExtractor is not fully implemented yet (Phase 2)")
        
        # 臨時返回空列表
        return []
    
    def batch_extract(
        self,
        texts: List[str],
        confidence_threshold: float = 0.7
    ) -> List[List[Dict[str, any]]]:
        """批量提取症狀"""
        return [self.extract_symptoms(text, confidence_threshold) for text in texts]


class HPOMatcher:
    """HPO 術語匹配器 (模糊匹配)"""
    
    def __init__(self, hpo_index_path: str):
        self.hpo_index_path = hpo_index_path
        self.hpo_index = None
    
    def build_index(self):
        """建立 HPO 搜尋索引 (使用 FAISS 或 hnswlib)"""
        # TODO: Phase 2 實現
        pass
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """
        模糊搜尋 HPO terms
        
        Returns:
            List of (hpo_id, hpo_name, similarity_score)
        """
        # TODO: Phase 2 實現
        return []
```

### 2. FHIR 適配器

**位置**: `src/medical_standards/fhir_adapter.py`

```python
"""
FHIR 適配器 - 整合 HL7 FHIR 病歷資料
狀態: 🟡 Phase 2 實現 (預留接口)
"""
from typing import Dict, List, Any
from fhir.resources.bundle import Bundle
from fhir.resources.patient import Patient
from fhir.resources.condition import Condition
from fhir.resources.observation import Observation
import logging

logger = logging.getLogger(__name__)


class FHIRAdapter:
    """FHIR 資料適配器"""
    
    def __init__(self):
        self.supported_resources = [
            "Patient",
            "Condition",
            "Observation",
            "DiagnosticReport",
            "FamilyMemberHistory"
        ]
    
    def parse_bundle(self, fhir_bundle: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析 FHIR Bundle 並轉換為內部格式
        
        Args:
            fhir_bundle: FHIR Bundle JSON
        
        Returns:
            Internal patient data format
        
        Example:
            >>> adapter = FHIRAdapter()
            >>> internal_data = adapter.parse_bundle(fhir_bundle)
            >>> print(internal_data['phenotypes'])
            ['HP:0003324', 'HP:0011675']
        """
        try:
            bundle = Bundle.parse_obj(fhir_bundle)
        except Exception as e:
            logger.error(f"Failed to parse FHIR Bundle: {e}")
            raise
        
        # 初始化內部資料結構
        internal_data = {
            "patient_id": None,
            "phenotypes": [],
            "diagnoses": {"icd10": [], "snomed": []},
            "medical_history": {
                "family_history": [],
                "lab_results": {},
                "medications": []
            },
            "demographics": {}
        }
        
        # 提取 Patient 資源
        for entry in bundle.entry or []:
            resource = entry.resource
            
            if resource.resource_type == "Patient":
                internal_data["patient_id"] = f"P{resource.id}"
                internal_data["demographics"] = self._extract_demographics(resource)
            
            elif resource.resource_type == "Condition":
                internal_data["diagnoses"] = self._extract_conditions(resource)
            
            elif resource.resource_type == "Observation":
                phenotypes, lab_results = self._extract_observations(resource)
                internal_data["phenotypes"].extend(phenotypes)
                internal_data["medical_history"]["lab_results"].update(lab_results)
            
            # TODO: 處理其他資源類型
        
        return internal_data
    
    def _extract_demographics(self, patient: Patient) -> Dict[str, Any]:
        """提取人口統計資料"""
        return {
            "age": self._calculate_age(patient.birthDate) if patient.birthDate else None,
            "gender": patient.gender,
        }
    
    def _extract_conditions(self, condition: Condition) -> Dict[str, List[str]]:
        """提取診斷編碼"""
        diagnoses = {"icd10": [], "snomed": []}
        
        for coding in condition.code.coding or []:
            if coding.system == "http://hl7.org/fhir/sid/icd-10":
                diagnoses["icd10"].append(coding.code)
            elif coding.system == "http://snomed.info/sct":
                diagnoses["snomed"].append(coding.code)
        
        return diagnoses
    
    def _extract_observations(
        self,
        observation: Observation
    ) -> Tuple[List[str], Dict[str, Any]]:
        """提取觀察結果 (症狀 + 檢驗數據)"""
        phenotypes = []
        lab_results = {}
        
        # TODO: 實際映射邏輯
        
        return phenotypes, lab_results
    
    @staticmethod
    def _calculate_age(birth_date: str) -> int:
        """計算年齡"""
        from datetime import datetime
        birth = datetime.fromisoformat(birth_date)
        today = datetime.now()
        return (today - birth).days // 365
    
    def export_to_fhir(self, internal_data: Dict[str, Any]) -> Dict[str, Any]:
        """將內部格式轉換為 FHIR Bundle (用於資料匯出)"""
        # TODO: Phase 2 實現
        pass
```

### 3. 智能輸入表單 (Gradio Component)

**位置**: `src/webui/components/input_form.py`

```python
"""
智能輸入表單 - Gradio UI 組件
"""
import gradio as gr
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class SmartInputForm:
    """智能患者資料輸入表單"""
    
    def __init__(self):
        self.hpo_search_enabled = True
        self.nlp_extraction_enabled = False  # Phase 2 啟用
    
    def create_interface(self) -> gr.Blocks:
        """建立 Gradio 界面"""
        
        with gr.Blocks() as interface:
            gr.Markdown("# 🏥 SHEPHERD 診斷推理系統")
            
            with gr.Tab("結構化輸入"):
                patient_id = gr.Textbox(
                    label="患者 ID",
                    placeholder="P12345",
                    info="格式: P + 5-10位數字"
                )
                
                with gr.Row():
                    age = gr.Number(label="年齡", value=None)
                    gender = gr.Radio(
                        label="性別",
                        choices=["male", "female", "other"],
                        value="female"
                    )
                
                # HPO 症狀輸入 (帶搜尋)
                gr.Markdown("### 症狀 (HPO Terms)")
                with gr.Row():
                    hpo_search = gr.Textbox(
                        label="搜尋 HPO 術語",
                        placeholder="輸入症狀關鍵字...",
                        interactive=True
                    )
                    hpo_results = gr.Dropdown(
                        label="搜尋結果",
                        choices=[],
                        multiselect=False,
                        interactive=True
                    )
                
                selected_phenotypes = gr.Dataframe(
                    headers=["HPO ID", "名稱", "信心分數"],
                    datatype=["str", "str", "number"],
                    label="已選擇症狀",
                    interactive=True
                )
                
                # 基因資料 (可選)
                with gr.Accordion("基因資料 (選填)", open=False):
                    genes = gr.Textbox(
                        label="基因清單",
                        placeholder="DMD, BRCA1, TP53",
                        info="逗號分隔"
                    )
                    variants = gr.Textbox(
                        label="變異位點",
                        placeholder="c.123G>A",
                        lines=3
                    )
                
                # ICD 診斷碼 (可選)
                with gr.Accordion("診斷編碼 (選填)", open=False):
                    icd10_codes = gr.Textbox(
                        label="ICD-10 編碼",
                        placeholder="G71.0, I47.2",
                        info="逗號分隔"
                    )
                
                submit_btn = gr.Button("開始推理", variant="primary")
            
            with gr.Tab("自由文字輸入"):
                gr.Markdown("### 📝 症狀描述 (自然語言)")
                free_text = gr.Textbox(
                    label="症狀描述",
                    placeholder="例如: 患者8歲女孩,逐漸出現四肢無力,運動後心跳加速...",
                    lines=10
                )
                
                extract_btn = gr.Button("提取症狀", variant="secondary")
                extracted_symptoms = gr.Dataframe(
                    headers=["症狀文字", "HPO ID", "信心分數"],
                    label="提取結果"
                )
                
                confirm_btn = gr.Button("確認並推理", variant="primary")
            
            with gr.Tab("FHIR 匯入"):
                fhir_upload = gr.File(
                    label="上傳 FHIR Bundle (JSON)",
                    file_types=[".json"]
                )
                fhir_preview = gr.JSON(label="預覽")
                fhir_submit_btn = gr.Button("使用 FHIR 資料推理", variant="primary")
            
            # 結果顯示
            gr.Markdown("---")
            gr.Markdown("## 📊 推理結果")
            
            with gr.Row():
                with gr.Column(scale=2):
                    result_table = gr.Dataframe(
                        headers=["排名", "疾病名稱", "MONDO ID", "信心分數"],
                        label="候選疾病"
                    )
                
                with gr.Column(scale=1):
                    reasoning_path = gr.JSON(label="推理路徑")
            
            explanation = gr.Textbox(
                label="詳細解釋",
                lines=10,
                interactive=False
            )
            
            # 事件處理
            hpo_search.change(
                fn=self._search_hpo_terms,
                inputs=[hpo_search],
                outputs=[hpo_results]
            )
            
            extract_btn.click(
                fn=self._extract_symptoms_from_text,
                inputs=[free_text],
                outputs=[extracted_symptoms]
            )
            
            submit_btn.click(
                fn=self._run_inference,
                inputs=[patient_id, age, gender, selected_phenotypes, genes, icd10_codes],
                outputs=[result_table, reasoning_path, explanation]
            )
        
        return interface
    
    def _search_hpo_terms(self, query: str) -> List[str]:
        """搜尋 HPO 術語 (模糊匹配)"""
        if not query or len(query) < 2:
            return []
        
        # TODO: 實際 HPO 搜尋邏輯
        # 暫時返回範例
        return [
            "HP:0003324 - 肌肉無力 (Muscle weakness)",
            "HP:0011675 - 心律不整 (Arrhythmia)",
            "HP:0000365 - 聽力喪失 (Hearing loss)"
        ]
    
    def _extract_symptoms_from_text(self, free_text: str) -> List[List[str]]:
        """從自由文字提取症狀"""
        # TODO: 調用 NLP 模塊
        logger.warning("NLP extraction not implemented yet (Phase 2)")
        return []
    
    def _run_inference(self, *args) -> tuple:
        """執行推理"""
        # TODO: 調用推理管道
        return [], {}, "推理功能尚未完全實現"
```

---

## 🔌 擴充接口設計

### 1. LLM 接口 (策略模式)

**位置**: `src/llm/interface.py`

```python
"""
LLM 接口定義 - 支援多種後端實現
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class LLMConfig:
    """LLM 配置"""
    model_name: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9


class LLMInterface(ABC):
    """LLM 抽象接口 (僅負責文本生成)"""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        生成文本 (純函數,無副作用)
        
        Args:
            prompt: 輸入提示
            max_tokens: 最大生成長度
            temperature: 溫度參數
        
        Returns:
            生成的文本
        """
        pass
    
    @abstractmethod
    def batch_generate(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[str]:
        """批量生成文本"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """獲取模型資訊"""
        pass


class VLLMBackend(LLMInterface):
    """vLLM 後端實現 (離線推理)"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.engine = None
    
    def initialize(self):
        """初始化 vLLM 引擎"""
        from vllm import LLM
        
        self.engine = LLM(
            model=self.config.model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9
        )
    
    def generate(self, prompt: str, **kwargs) -> str:
        if not self.engine:
            self.initialize()
        
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
            temperature=kwargs.get('temperature', self.config.temperature),
            top_p=self.config.top_p
        )
        
        outputs = self.engine.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        # TODO: 實現批量生成
        return [self.generate(p, **kwargs) for p in prompts]
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "backend": "vLLM",
            "model_name": self.config.model_name,
            "max_tokens": self.config.max_tokens
        }


class LlamaCppBackend(LLMInterface):
    """llama.cpp 後端實現 (ARM 優化)"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.model = None
    
    def initialize(self):
        """初始化 llama.cpp"""
        from llama_cpp import Llama
        
        self.model = Llama(
            model_path=self.config.model_name,
            n_ctx=2048,
            n_gpu_layers=-1  # 全部 GPU 加速
        )
    
    def generate(self, prompt: str, **kwargs) -> str:
        if not self.model:
            self.initialize()
        
        output = self.model(
            prompt,
            max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
            temperature=kwargs.get('temperature', self.config.temperature),
            top_p=self.config.top_p
        )
        
        return output['choices'][0]['text']
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        return [self.generate(p, **kwargs) for p in prompts]
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "backend": "llama.cpp",
            "model_name": self.config.model_name
        }
```

### 2. 醫療標準映射接口

**位置**: `src/medical_standards/mapper_interface.py`

```python
"""
醫療標準映射接口 - 統一不同編碼體系
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class MappingResult:
    """映射結果"""
    source_code: str
    target_code: str
    confidence: float
    mapping_type: str  # "exact", "broad", "narrow", "related"


class MedicalCodeMapper(ABC):
    """醫療編碼映射器抽象接口"""
    
    @abstractmethod
    def map_to_hpo(self, code: str, system: str) -> List[MappingResult]:
        """
        將其他編碼系統映射到 HPO
        
        Args:
            code: 源編碼
            system: 編碼系統 (icd10, icd11, snomed)
        
        Returns:
            映射結果列表
        """
        pass
    
    @abstractmethod
    def map_to_mondo(self, code: str, system: str) -> List[MappingResult]:
        """映射到 MONDO 疾病本體"""
        pass
    
    @abstractmethod
    def reverse_map(self, hpo_id: str, target_system: str) -> List[MappingResult]:
        """反向映射: HPO → 其他系統"""
        pass


class ICDMapper(MedicalCodeMapper):
    """ICD-10/11 映射器"""
    
    def __init__(self, mapping_file: str):
        self.mapping_file = mapping_file
        self.mappings = {}
    
    def load_mappings(self):
        """載入映射表"""
        # TODO: 載入預建映射表
        pass
    
    def map_to_hpo(self, code: str, system: str) -> List[MappingResult]:
        # TODO: 實現 ICD → HPO 映射
        return []
    
    def map_to_mondo(self, code: str, system: str) -> List[MappingResult]:
        # TODO: 實現 ICD → MONDO 映射
        return []
    
    def reverse_map(self, hpo_id: str, target_system: str) -> List[MappingResult]:
        # TODO: 實現反向映射
        return []
```

---

## ✅ 實施檢查清單

### Phase 1: 核心架構 (Week 1-2)

#### 配置與驗證系統
- [ ] 創建 `pyproject.toml` + 工具鏈配置
- [ ] 創建 `.import-linter.ini` 依賴規則
- [ ] 創建 `.pre-commit-config.yaml` Git hooks
- [ ] 實現 `ConfigValidator` (配置驗證器)
- [ ] 創建所有 JSON Schema 檔案
  - [ ] `patient_input.schema.json`
  - [ ] `inference_output.schema.json`
  - [ ] `base_config.schema.json`
  - [ ] `model_config.schema.json`
  - [ ] `data_config.schema.json`

#### 版本管理系統
- [ ] 實現 `VersionChecker` (版本兼容性檢查)
- [ ] 實現 `hash_generator.py` (資料哈希生成)
- [ ] 創建增強版 KG metadata.json 模板
- [ ] 創建模型 registry.json 模板
- [ ] 更新 `builder.py` 自動生成 metadata

#### 測試基礎設施
- [ ] 創建平台特定測試框架
  - [ ] `test_attention_x86.py`
  - [ ] `test_attention_arm.py`
  - [ ] `test_vector_index_x86.py`
  - [ ] `test_vector_index_arm.py`
- [ ] 創建測試資料 fixtures
  - [ ] `sample_patients.json`
  - [ ] `sample_fhir.json`

**預計工作量**: 16-20 小時

---

### Phase 2: 醫療功能整合 (Week 3-6)

#### NLP 模塊 (🟡 中優先級)
- [ ] 實現 `SymptomExtractor` 基礎類
- [ ] 實現 `HPOMatcher` (模糊匹配)
- [ ] 下載並整合 SciBERT/ClinicalBERT
- [ ] 建立 HPO 術語搜尋索引
- [ ] 實現批量提取接口
- [ ] 編寫 NLP 單元測試

#### FHIR/HISS 適配器 (🟡 中優先級)
- [ ] 實現 `FHIRAdapter` 基礎類
- [ ] 實現 `HI SSAdapter` 基礎類
- [ ] 支援 Patient, Condition, Observation 資源
- [ ] 實現 FHIR → 內部格式轉換
- [ ] 實現內部格式 → FHIR 匯出
- [ ] 編寫 FHIR 整合測試

#### 醫療標準映射 (🟡 中優先級)
- [ ] 實現 `ICDMapper` (ICD-10/11 → HPO/MONDO)
- [ ] 實現 `SNOMEDMapper`
- [ ] 下載並整合映射表 (UMLS, BioPortal)
- [ ] 實現反向映射功能
- [ ] 編寫映射單元測試

#### WebUI 增強 (🟢 低優先級)
- [ ] 實現 `SmartInputForm` (智能表單)
- [ ] 實現 HPO 搜尋組件
- [ ] 實現自由文字輸入 UI
- [ ] 實現 FHIR 上傳組件
- [ ] 實現結果可視化組件

**預計工作量**: 40-50 小時

---

### Phase 3: 擴充接口 (Week 7-8)

#### LLM 接口標準化
- [ ] 實現 `LLMInterface` 抽象類
- [ ] 實現 `VLLMBackend`
- [ ] 實現 `LlamaCppBackend`
- [ ] 測試多後端切換
- [ ] 性能基準測試

#### 輸入/輸出驗證
- [ ] 實現 `InputValidator` (JSON Schema + Pydantic)
- [ ] 實現 `OutputFormatter`
- [ ] 整合到推理管道
- [ ] 編寫端到端測試

#### 文檔更新
- [ ] 更新 `architecture_v3.md`
- [ ] 撰寫 `medical_integration.md`
- [ ] 更新 API 參考文檔
- [ ] 撰寫部署指南

**預計工作量**: 20-24 小時

---

## 📊 總時間估算

| 階段 | 工作內容 | 預計時間 | 優先級 |
|------|---------|----------|--------|
| Phase 1 | 核心架構 + 驗證系統 | 16-20h | 🔴 P0 |
| Phase 2 | 醫療功能整合 | 40-50h | 🟡 P1 |
| Phase 3 | 擴充接口 + 文檔 | 20-24h | 🟢 P2 |
| **總計** | | **76-94h** | **約2-3週** |

---

## 🎯 關鍵成功指標

### 技術指標
- ✅ 所有配置檔案通過 JSON Schema 驗證
- ✅ import-linter 檢查通過 (無循環依賴)
- ✅ 平台特定測試覆蓋率 > 80%
- ✅ 版本兼容性檢查自動化

### 醫療功能指標
- ✅ NLP 症狀提取準確率 > 75% (Phase 2)
- ✅ FHIR 資料解析成功率 > 95%
- ✅ ICD/SNOMED 映射覆蓋率 > 80%

### 可維護性指標
- ✅ 程式碼註解覆蓋率 > 60%
- ✅ 文檔完整性評分 > 90%
- ✅ CI/CD 流程自動化率 100%

---

## 🔄 版本歷史

| 版本 | 日期 | 變更內容 |
|------|------|----------|
| v1.0 | 2025-10-07 | 初始版本 (基礎目錄結構) |
| v2.0 | 2025-10-22 | 整合 ChatGPT 建議 (工具鏈 + metadata) |
| v3.0 | 2025-11-04 | **整合醫生團隊建議 (NLP + FHIR + 智能UI)** |

---

## 📞 聯絡資訊

**項目負責人**: [待填寫]  
**技術審核**: [待填寫]  
**醫療顧問**: [待填寫]  

---

**文檔結束**
