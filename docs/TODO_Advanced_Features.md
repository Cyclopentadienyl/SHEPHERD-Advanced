# 醫療知識圖譜診斷引擎 - 進階功能 TODO

**版本**: v1.0  
**建議優先級**: Phase 2-3  
**預計時間**: 4-6 週  
**狀態**: 🔴 未開始

---

## 🎯 功能概述

### 功能 1: 症狀/變異關聯分析
- **目標**: 分析兩個症狀或變異之間的關聯性
- **應用**: 輔助醫生理解症狀組合的生物學機制
- **技術**: 圖路徑分析 + 統計共現 + 本體推理

### 功能 2: 基因型-表型關聯排序增強
- **目標**: 提供詳細的基因-症狀關聯強度、外顯率、文獻支持
- **應用**: 精準醫學，基因檢測結果解讀
- **技術**: 整合 ClinVar + Pubtator + 知識圖譜

### 功能 3: 藥物/治療建議（研究參考）
- **目標**: 基於診斷結果提供藥物治療思路（僅供參考）
- **應用**: 給醫生提供治療方向的初步線索
- **技術**: 擴展知識圖譜整合 DrugBank
- **⚠️ 重要**: 附帶法律免責聲明，僅作研究參考

### 功能 4: 離線推理 + 文獻檢索與排序
- **目標**: 診斷後自動檢索相關文獻並按可信度排序
- **應用**: 提供診斷的文獻支持，輔助臨床決策
- **技術**: Pubtator 預下載 + 可選 PubMed API

---

## 📋 Phase 2.3: 症狀/變異關聯分析 (Week 1-1.5)

### 🟠 P1 - 基礎架構 (Day 1-2)

#### 1.1 創建關聯分析模組

- [ ] 創建 `src/analysis/__init__.py` 🕐 5min
  ```python
  """
  分析模組：症狀關聯、基因-表型關聯等
  """
  from .phenotype_correlation import PhenotypeCorrelationAnalyzer
  from .genotype_phenotype_ranker import GenotypePhenotypeRanker
  
  __all__ = [
      'PhenotypeCorrelationAnalyzer',
      'GenotypePhenotypeRanker'
  ]
  ```

- [ ] 創建 `src/analysis/phenotype_correlation.py` 📅 1 天
  ```python
  """
  症狀關聯分析器
  
  依賴:
      - src/kg/builder.py (知識圖譜)
      - src/retrieval/path_retriever.py (路徑檢索)
      - src/ontology/similarity.py (本體相似度)
  """
  from typing import Dict, List, Tuple
  from src.kg.builder import KnowledgeGraphBuilder
  from src.retrieval.path_retriever import PathRetriever
  from src.ontology.similarity import OntologySimilarity
  
  class PhenotypeCorrelationAnalyzer:
      """分析兩個症狀之間的多維度關聯"""
      
      def __init__(self, kg: KnowledgeGraphBuilder):
          self.kg = kg
          self.path_retriever = PathRetriever(kg)
          self.ontology_sim = OntologySimilarity()
          self.cooccurrence = self._load_cooccurrence_graph()
      
      def analyze_correlation(
          self,
          phenotype_a: str,  # HPO ID
          phenotype_b: str   # HPO ID
      ) -> Dict:
          """
          分析兩個症狀的關聯性
          
          Returns:
              {
                  'correlation_type': str,       # 'synergistic', 'antagonistic', 'independent'
                  'strength': float,             # [0, 1]
                  'confidence': float,           # 統計顯著性
                  'mechanisms': List[Dict],      # 生物學機制
                  'shared_diseases': List[str],  # 共同關聯疾病
                  'shared_genes': List[str],     # 共同關聯基因
                  'connecting_paths': List[Dict],# 連接路徑
                  'evidence': List[Dict]         # 支持證據
              }
          """
          # 1. 路徑分析
          paths = self._find_connecting_paths(phenotype_a, phenotype_b)
          
          # 2. 共現分析
          cooccurrence_stats = self._compute_cooccurrence(phenotype_a, phenotype_b)
          
          # 3. 本體關係
          ontology_relation = self._check_ontology_relation(phenotype_a, phenotype_b)
          
          # 4. 共享實體
          shared_entities = self._find_shared_entities(phenotype_a, phenotype_b)
          
          # 5. 生物學機制推斷
          mechanisms = self._infer_mechanisms(paths, shared_entities)
          
          # 6. 綜合評分
          correlation_type, strength = self._classify_and_score(
              paths, cooccurrence_stats, ontology_relation
          )
          
          return {
              'correlation_type': correlation_type,
              'strength': strength,
              'confidence': cooccurrence_stats.get('p_value', 0.0),
              'mechanisms': mechanisms,
              'shared_diseases': shared_entities['diseases'],
              'shared_genes': shared_entities['genes'],
              'connecting_paths': paths[:5],  # 前5條路徑
              'evidence': self._collect_evidence(paths)
          }
      
      def _find_connecting_paths(self, phenotype_a, phenotype_b):
          """找到連接兩症狀的所有路徑"""
          return self.path_retriever.find_paths_between(
              source=phenotype_a,
              target=phenotype_b,
              max_hops=4,
              top_k=20
          )
      
      def _compute_cooccurrence(self, phenotype_a, phenotype_b):
          """
          計算統計共現指標
          
          Returns:
              {
                  'frequency': float,      # 共現頻率
                  'pmi': float,            # 點互信息
                  'chi_square': float,     # 卡方統計量
                  'p_value': float,        # 顯著性
                  'odds_ratio': float      # 優勢比
              }
          """
          # 從疾病共現圖查詢
          return self.cooccurrence.compute_metrics(phenotype_a, phenotype_b)
      
      def _check_ontology_relation(self, phenotype_a, phenotype_b):
          """檢查本體層次關係"""
          return {
              'is_parent_child': self.ontology_sim.is_ancestor(phenotype_a, phenotype_b),
              'is_sibling': self.ontology_sim.are_siblings(phenotype_a, phenotype_b),
              'semantic_similarity': self.ontology_sim.compute_similarity(
                  phenotype_a, phenotype_b
              ),
              'common_ancestor': self.ontology_sim.lowest_common_ancestor(
                  phenotype_a, phenotype_b
              )
          }
      
      def _find_shared_entities(self, phenotype_a, phenotype_b):
          """找到兩症狀共享的疾病和基因"""
          # 查詢知識圖譜
          diseases_a = self.kg.query_connected_entities(phenotype_a, 'disease')
          diseases_b = self.kg.query_connected_entities(phenotype_b, 'disease')
          
          genes_a = self.kg.query_connected_entities(phenotype_a, 'gene')
          genes_b = self.kg.query_connected_entities(phenotype_b, 'gene')
          
          return {
              'diseases': list(set(diseases_a) & set(diseases_b)),
              'genes': list(set(genes_a) & set(genes_b))
          }
      
      def _infer_mechanisms(self, paths, shared_entities):
          """
          推斷生物學機制
          
          機制類型:
              1. 共享基因突變
              2. 共享生物通路
              3. 級聯效應（一個症狀導致另一個）
              4. 平行效應（共同原因）
          """
          mechanisms = []
          
          # 機制1：共享基因
          if shared_entities['genes']:
              mechanisms.append({
                  'type': 'shared_genetic_basis',
                  'genes': shared_entities['genes'][:5],
                  'description': f"兩症狀由 {len(shared_entities['genes'])} 個共同基因關聯"
              })
          
          # 機制2：路徑推斷
          for path in paths[:3]:
              if len(path['nodes']) == 3:  # A → X → B
                  intermediate = path['nodes'][1]
                  mechanisms.append({
                      'type': 'cascade_effect',
                      'mediator': intermediate,
                      'description': f"症狀A通過 {intermediate} 導致症狀B"
                  })
          
          return mechanisms
      
      def _classify_and_score(self, paths, cooccurrence, ontology_relation):
          """
          分類關聯類型並評分
          
          類型:
              - synergistic: 協同（共現頻率高）
              - antagonistic: 拮抗（很少共現）
              - independent: 獨立（無顯著關聯）
          """
          # 基於共現頻率和p值
          if cooccurrence['p_value'] < 0.05:
              if cooccurrence['odds_ratio'] > 1.5:
                  correlation_type = 'synergistic'
                  strength = min(cooccurrence['odds_ratio'] / 10.0, 1.0)
              elif cooccurrence['odds_ratio'] < 0.5:
                  correlation_type = 'antagonistic'
                  strength = max(1.0 - cooccurrence['odds_ratio'], 0.5)
              else:
                  correlation_type = 'independent'
                  strength = 0.3
          else:
              correlation_type = 'independent'
              strength = 0.1
          
          # 路徑存在性增強評分
          if paths:
              strength = min(strength + 0.2, 1.0)
          
          # 本體相似度增強評分
          if ontology_relation['semantic_similarity'] > 0.7:
              strength = min(strength + 0.15, 1.0)
          
          return correlation_type, strength
  ```

#### 1.2 API 端點整合

- [ ] 更新 `src/api/routes/analysis.py` 📅 0.5 天
  ```python
  """
  分析相關 API 端點
  """
  from fastapi import APIRouter, HTTPException
  from pydantic import BaseModel
  from typing import List
  from src.analysis.phenotype_correlation import PhenotypeCorrelationAnalyzer
  
  router = APIRouter(prefix="/api/v2/analysis", tags=["analysis"])
  
  class CorrelationRequest(BaseModel):
      phenotype_a: str  # HPO ID
      phenotype_b: str  # HPO ID
  
  @router.post("/phenotype-correlation")
  async def analyze_phenotype_correlation(request: CorrelationRequest):
      """
      分析兩個症狀的關聯性
      
      Example:
          POST /api/v2/analysis/phenotype-correlation
          {
              "phenotype_a": "HP:0001324",  // 肌肉無力
              "phenotype_b": "HP:0001649"   // 心律不整
          }
      """
      try:
          analyzer = PhenotypeCorrelationAnalyzer(kg=global_kg)
          result = analyzer.analyze_correlation(
              request.phenotype_a,
              request.phenotype_b
          )
          return result
      except Exception as e:
          raise HTTPException(status_code=500, detail=str(e))
  ```

#### 1.3 WebUI 整合

- [ ] 更新 `webui/components/analysis_tab.py` 📅 0.5 天
  ```python
  # 新增 Analysis Tab
  
  with gr.Tab("🔬 關聯分析"):
      gr.Markdown("## 症狀關聯分析")
      
      with gr.Row():
          phenotype_a_input = gr.Textbox(
              label="症狀 A (HPO ID)",
              placeholder="例如: HP:0001324"
          )
          phenotype_b_input = gr.Textbox(
              label="症狀 B (HPO ID)",
              placeholder="例如: HP:0001649"
          )
      
      analyze_btn = gr.Button("分析關聯", variant="primary")
      
      # 結果顯示
      with gr.Column():
          correlation_type = gr.Textbox(label="關聯類型")
          strength_slider = gr.Slider(
              minimum=0, maximum=1,
              label="關聯強度",
              interactive=False
          )
          mechanisms_display = gr.JSON(label="生物學機制")
          shared_diseases = gr.DataFrame(label="共同關聯疾病")
          paths_viz = gr.HTML(label="連接路徑可視化")
  ```

**小計**: 📆 3-4 天

---

## 📋 Phase 2.4: 基因型-表型關聯排序增強 (Week 1.5-2)

### 🟠 P1 - 增強排序模組

- [ ] 擴展 `src/models/tasks/gene_scoring.py` 📅 2 天
  ```python
  """
  增強版基因-表型關聯評分
  
  新增功能:
      - 外顯率計算
      - ClinVar 變異資訊整合
      - 遺傳模式推斷
      - 文獻支持評分
  """
  from typing import Dict, List
  import requests
  
  class EnhancedGeneScoring:
      """增強版基因評分系統"""
      
      def __init__(self, kg, clinvar_api, pubtator_db):
          self.kg = kg
          self.clinvar_api = clinvar_api
          self.pubtator = pubtator_db
          self.base_scorer = GeneScoring(kg)  # 現有模組
      
      def rank_genotype_phenotype_correlations(
          self,
          genotypes: List[str],  # 基因清單或變異ID
          phenotypes: List[str]  # HPO IDs
      ) -> List[Dict]:
          """
          詳細的基因-表型關聯排序
          
          Returns:
              List[{
                  'gene': str,
                  'phenotype': str,
                  'correlation_score': float,       # 整體分數
                  'penetrance': float,              # 外顯率
                  'pathogenicity': str,             # 致病性
                  'mode_of_inheritance': str,       # 遺傳模式
                  'allele_frequency': float,        # 等位基因頻率
                  'clinical_significance': str,
                  'evidence_strength': str,         # 'strong', 'moderate', 'weak'
                  'literature_count': int,
                  'top_papers': List[Dict]
              }]
          """
          results = []
          
          for gene in genotypes:
              for phenotype in phenotypes:
                  # 1. 基礎 GNN 評分
                  gnn_score = self.base_scorer.score_gene_phenotype(gene, phenotype)
                  
                  # 2. ClinVar 變異資訊
                  clinvar_data = self._fetch_clinvar_data(gene, phenotype)
                  
                  # 3. 外顯率估算
                  penetrance = self._estimate_penetrance(gene, phenotype, clinvar_data)
                  
                  # 4. 遺傳模式
                  inheritance_mode = self._infer_inheritance_mode(gene, clinvar_data)
                  
                  # 5. 文獻支持
                  literature = self._search_literature(gene, phenotype)
                  
                  # 6. 綜合評分
                  final_score = self._compute_final_score(
                      gnn_score,
                      clinvar_data,
                      penetrance,
                      len(literature)
                  )
                  
                  results.append({
                      'gene': gene,
                      'phenotype': phenotype,
                      'correlation_score': final_score,
                      'penetrance': penetrance,
                      'pathogenicity': clinvar_data.get('significance', 'VUS'),
                      'mode_of_inheritance': inheritance_mode,
                      'allele_frequency': clinvar_data.get('frequency', 0.0),
                      'clinical_significance': clinvar_data.get('description', ''),
                      'evidence_strength': self._classify_evidence(clinvar_data, literature),
                      'literature_count': len(literature),
                      'top_papers': literature[:5]
                  })
          
          # 排序
          return sorted(results, key=lambda x: x['correlation_score'], reverse=True)
      
      def _fetch_clinvar_data(self, gene, phenotype):
          """從 ClinVar 獲取變異資訊"""
          # 查詢 ClinVar API
          response = self.clinvar_api.search(
              gene=gene,
              phenotype=phenotype
          )
          return self._parse_clinvar_response(response)
      
      def _estimate_penetrance(self, gene, phenotype, clinvar_data):
          """
          估算外顯率
          
          方法:
              1. 從 ClinVar 變異資料推斷
              2. 從文獻中提取
              3. 基於知識圖譜統計
          """
          # 簡化版：基於 ClinVar 致病性
          pathogenicity_penetrance = {
              'Pathogenic': 0.8,
              'Likely pathogenic': 0.6,
              'VUS': 0.3,
              'Likely benign': 0.1,
              'Benign': 0.05
          }
          return pathogenicity_penetrance.get(
              clinvar_data.get('significance', 'VUS'),
              0.5
          )
      
      def _infer_inheritance_mode(self, gene, clinvar_data):
          """推斷遺傳模式"""
          # 從 ClinVar 或知識圖譜查詢
          kg_mode = self.kg.query_gene_attribute(gene, 'inheritance_mode')
          clinvar_mode = clinvar_data.get('inheritance', '')
          
          # 優先使用 ClinVar
          if clinvar_mode:
              return clinvar_mode
          elif kg_mode:
              return kg_mode
          else:
              return 'unknown'
      
      def _search_literature(self, gene, phenotype):
          """搜尋相關文獻"""
          # 從 Pubtator 本地資料庫查詢
          papers = self.pubtator.search(
              entities=[gene, phenotype],
              relation_type='gene_phenotype',
              limit=20
          )
          return papers
      
      def _compute_final_score(self, gnn_score, clinvar_data, penetrance, lit_count):
          """綜合評分"""
          # 加權平均
          weights = {
              'gnn': 0.4,
              'clinvar': 0.3,
              'penetrance': 0.2,
              'literature': 0.1
          }
          
          clinvar_score = self._clinvar_to_score(clinvar_data)
          lit_score = min(lit_count / 20.0, 1.0)
          
          final = (
              weights['gnn'] * gnn_score +
              weights['clinvar'] * clinvar_score +
              weights['penetrance'] * penetrance +
              weights['literature'] * lit_score
          )
          
          return final
      
      def _clinvar_to_score(self, clinvar_data):
          """ClinVar 致病性轉評分"""
          pathogenicity_scores = {
              'Pathogenic': 1.0,
              'Likely pathogenic': 0.8,
              'VUS': 0.5,
              'Likely benign': 0.2,
              'Benign': 0.0
          }
          return pathogenicity_scores.get(
              clinvar_data.get('significance', 'VUS'),
              0.5
          )
      
      def _classify_evidence(self, clinvar_data, literature):
          """分類證據強度"""
          if clinvar_data.get('significance') == 'Pathogenic' and len(literature) >= 10:
              return 'strong'
          elif len(literature) >= 5:
              return 'moderate'
          else:
              return 'weak'
  ```

- [ ] 創建 ClinVar API 包裝器 📅 0.5 天
  ```python
  # src/data/integrations/clinvar_api.py
  
  """
  ClinVar API 包裝器
  """
  import requests
  from typing import Dict, List
  
  class ClinVarAPI:
      """ClinVar 變異資料庫 API"""
      
      BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
      
      def search(self, gene: str, phenotype: str = None) -> List[Dict]:
          """
          搜尋 ClinVar 變異
          
          API: E-utilities
          """
          # 構建查詢
          query = f"{gene}[gene]"
          if phenotype:
              query += f" AND {phenotype}[phenotype]"
          
          # 呼叫 API
          response = requests.get(
              f"{self.BASE_URL}/esearch.fcgi",
              params={
                  'db': 'clinvar',
                  'term': query,
                  'retmode': 'json',
                  'retmax': 100
              }
          )
          
          # 解析結果
          return self._parse_response(response.json())
  ```

**小計**: 📆 2-3 天

---

## 📋 Phase 2.5: 藥物/治療建議（研究參考）(Week 2.5-3.5) ⚠️

### 🔴 重要聲明

```
⚠️⚠️⚠️ 法律與倫理聲明 ⚠️⚠️⚠️

此功能僅供研究與教學用途，提供的藥物建議：
1. 不構成醫療建議或處方
2. 必須由專業醫師審核確認
3. 不得直接用於臨床決策
4. 系統開發者不承擔任何法律責任

使用前必須：
- 獲得醫院倫理委員會批准
- 在 UI 顯著位置標示免責聲明
- 記錄所有查詢日誌供審計
- 定期由醫學專家審核建議品質
```

### 🟡 P2 - 藥物知識圖譜擴展

#### 3.1 資料整合

- [ ] 下載 DrugBank 資料 📅 0.5 天
  ```bash
  # 需要註冊 DrugBank 帳號
  # https://www.drugbank.ca/
  
  # 下載位置
  data/raw/drugbank/
  ├── drugbank_all_full_database.xml
  ├── drug_links.csv
  └── README.txt
  ```

- [ ] 創建 DrugBank 解析器 📅 1 天
  ```python
  # src/data/parsers/drugbank_parser.py
  
  """
  DrugBank XML 解析器
  """
  import xml.etree.ElementTree as ET
  from typing import Dict, List
  
  class DrugBankParser:
      """解析 DrugBank 資料庫"""
      
      def parse(self, xml_file: str) -> List[Dict]:
          """
          解析 DrugBank XML
          
          提取資訊:
              - 藥物ID, 名稱
              - 適應症 (indications)
              - 藥物-疾病關聯
              - 藥物-基因交互作用
              - 副作用
          """
          tree = ET.parse(xml_file)
          root = tree.getroot()
          
          drugs = []
          for drug_element in root.findall('.//drug'):
              drug_info = self._extract_drug_info(drug_element)
              drugs.append(drug_info)
          
          return drugs
      
      def _extract_drug_info(self, drug_element):
          """提取單個藥物資訊"""
          return {
              'drugbank_id': drug_element.find('drugbank-id').text,
              'name': drug_element.find('name').text,
              'indications': self._extract_indications(drug_element),
              'targets': self._extract_targets(drug_element),
              'side_effects': self._extract_side_effects(drug_element)
          }
  ```

- [ ] 構建藥物知識圖譜 📅 1 天
  ```python
  # src/kg/drug_kg_builder.py
  
  """
  藥物知識圖譜構建器
  """
  from typing import Dict, List
  import torch
  from torch_geometric.data import HeteroData
  
  class DrugKnowledgeGraphBuilder:
      """構建包含藥物的知識圖譜"""
      
      def build(self, drugbank_data, base_kg):
          """
          擴展現有知識圖譜加入藥物節點
          
          新增節點類型:
              - drug
          
          新增邊類型:
              - (drug, treats, disease)
              - (drug, targets, gene)
              - (drug, causes, phenotype)  # 副作用
          """
          # 添加藥物節點
          drug_nodes = [d['drugbank_id'] for d in drugbank_data]
          
          # 添加藥物-疾病邊
          drug_disease_edges = self._create_drug_disease_edges(drugbank_data)
          
          # 添加藥物-基因邊
          drug_gene_edges = self._create_drug_gene_edges(drugbank_data)
          
          # 合併到現有圖
          extended_kg = self._merge_with_base_kg(
              base_kg,
              drug_nodes,
              drug_disease_edges,
              drug_gene_edges
          )
          
          return extended_kg
  ```

#### 3.2 藥物建議引擎

- [ ] 創建 `src/treatment/drug_recommender.py` 📅 2 天
  ```python
  """
  藥物建議引擎（研究參考）
  
  ⚠️ 警告：此模組僅供研究參考，不得用於臨床決策
  """
  from typing import Dict, List
  import logging
  
  logger = logging.getLogger(__name__)
  
  class DrugRecommendationEngine:
      """
      藥物建議引擎
      
      ⚠️ 重要：所有輸出必須包含免責聲明
      """
      
      DISCLAIMER = """
      ⚠️⚠️⚠️ 免責聲明 ⚠️⚠️⚠️
      
      以下藥物建議僅供研究參考，不構成醫療建議。
      所有治療決策必須由專業醫師基於患者具體情況做出。
      請勿在未經醫師確認的情況下使用任何藥物。
      
      本系統開發者不對藥物使用後果承擔任何責任。
      """
      
      def __init__(self, drug_kg, confidence_threshold=0.6):
          self.drug_kg = drug_kg
          self.confidence_threshold = confidence_threshold
          
          # 記錄所有查詢（審計用）
          self.audit_log = []
      
      def suggest_treatments(
          self,
          diagnosis_results: Dict,
          patient_genotype: List[str] = None,
          patient_allergies: List[str] = None
      ) -> Dict:
          """
          基於診斷結果建議治療方案
          
          Args:
              diagnosis_results: 診斷結果（含疾病、基因）
              patient_genotype: 患者基因型（可選）
              patient_allergies: 藥物過敏史（可選）
          
          Returns:
              {
                  'disclaimer': str,              # ⚠️ 必須包含
                  'suggestions': List[Dict],      # 藥物建議
                  'confidence': str,              # 'low', 'medium', 'high'
                  'warnings': List[str],          # 警告資訊
                  'references': List[str]         # 文獻參考
              }
          """
          # 記錄查詢（審計）
          self._log_query(diagnosis_results, patient_genotype)
          
          # 提取疾病
          diseases = diagnosis_results.get('top_diseases', [])
          genes = diagnosis_results.get('top_genes', [])
          
          # 查詢藥物
          drug_candidates = self._query_drugs_for_diseases(diseases)
          
          # 基因型過濾（藥物代謝）
          if patient_genotype:
              drug_candidates = self._filter_by_genotype(
                  drug_candidates,
                  patient_genotype
              )
          
          # 過敏史過濾
          if patient_allergies:
              drug_candidates = self._filter_by_allergies(
                  drug_candidates,
                  patient_allergies
              )
          
          # 評分與排序
          ranked_drugs = self._rank_drugs(drug_candidates, diseases, genes)
          
          # 置信度評估
          confidence = self._assess_confidence(ranked_drugs)
          
          # 生成警告
          warnings = self._generate_warnings(ranked_drugs, diseases)
          
          return {
              'disclaimer': self.DISCLAIMER,  # ⚠️ 強制包含
              'suggestions': ranked_drugs[:10],  # 前10個
              'confidence': confidence,
              'warnings': warnings,
              'references': self._collect_references(ranked_drugs)
          }
      
      def _query_drugs_for_diseases(self, diseases):
          """查詢治療這些疾病的藥物"""
          drugs = []
          for disease in diseases:
              disease_drugs = self.drug_kg.query_edges(
                  source_type='drug',
                  relation='treats',
                  target=disease['id']
              )
              drugs.extend(disease_drugs)
          return drugs
      
      def _filter_by_genotype(self, drugs, genotype):
          """
          基於基因型過濾藥物
          
          考慮因素:
              - 藥物代謝酶基因型（CYP450家族）
              - 藥物轉運體基因型
              - 藥物靶點基因型
          """
          # 檢查 CYP450 代謝
          metabolizer_status = self._predict_metabolizer_status(genotype)
          
          filtered_drugs = []
          for drug in drugs:
              # 檢查是否需要特定代謝能力
              if self._is_compatible_with_metabolism(drug, metabolizer_status):
                  filtered_drugs.append(drug)
          
          return filtered_drugs
      
      def _predict_metabolizer_status(self, genotype):
          """
          預測藥物代謝能力
          
          分類:
              - ultra_rapid_metabolizer
              - extensive_metabolizer (正常)
              - intermediate_metabolizer
              - poor_metabolizer
          """
          # 簡化版：檢查 CYP2D6, CYP2C19 等
          # 實際應使用 PharmGKB 資料庫
          return 'extensive_metabolizer'  # 預設
      
      def _rank_drugs(self, drugs, diseases, genes):
          """
          藥物排序
          
          評分因素:
              1. 疾病適應症匹配度 (40%)
              2. 藥物-基因交互作用 (20%)
              3. 文獻支持強度 (20%)
              4. 副作用嚴重程度 (10%)
              5. 臨床使用頻率 (10%)
          """
          scored_drugs = []
          
          for drug in drugs:
              indication_score = self._score_indication_match(drug, diseases)
              gene_interaction_score = self._score_gene_interaction(drug, genes)
              literature_score = self._score_literature_support(drug, diseases)
              safety_score = self._score_safety_profile(drug)
              usage_score = self._score_clinical_usage(drug)
              
              total_score = (
                  0.4 * indication_score +
                  0.2 * gene_interaction_score +
                  0.2 * literature_score +
                  0.1 * safety_score +
                  0.1 * usage_score
              )
              
              scored_drugs.append({
                  'drug_id': drug['drugbank_id'],
                  'drug_name': drug['name'],
                  'score': total_score,
                  'indication': drug.get('indication', ''),
                  'mechanism': drug.get('mechanism_of_action', ''),
                  'side_effects': drug.get('side_effects', [])[:5],
                  'contraindications': drug.get('contraindications', []),
                  'evidence_level': self._classify_evidence(literature_score)
              })
          
          return sorted(scored_drugs, key=lambda x: x['score'], reverse=True)
      
      def _assess_confidence(self, ranked_drugs):
          """
          評估建議的置信度
          
          規則:
              - 前3個藥物分數 > 0.8 → high
              - 前3個藥物分數 > 0.6 → medium
              - 其他 → low
          """
          if not ranked_drugs:
              return 'low'
          
          top_3_scores = [d['score'] for d in ranked_drugs[:3]]
          avg_score = sum(top_3_scores) / len(top_3_scores)
          
          if avg_score > 0.8:
              return 'high'
          elif avg_score > 0.6:
              return 'medium'
          else:
              return 'low'
      
      def _generate_warnings(self, drugs, diseases):
          """
          生成警告資訊
          
          警告類型:
              - 藥物交互作用
              - 嚴重副作用
              - 禁忌症
              - 特殊人群注意事項
          """
          warnings = []
          
          # 檢查禁忌症
          for drug in drugs[:5]:
              if drug.get('contraindications'):
                  warnings.append(
                      f"⚠️ {drug['drug_name']}: 禁忌症包括 {drug['contraindications']}"
                  )
          
          # 檢查嚴重副作用
          for drug in drugs[:5]:
              serious_effects = [
                  se for se in drug.get('side_effects', [])
                  if 'serious' in se.lower() or 'severe' in se.lower()
              ]
              if serious_effects:
                  warnings.append(
                      f"⚠️ {drug['drug_name']}: 可能有嚴重副作用"
                  )
          
          return warnings
      
      def _log_query(self, diagnosis, genotype):
          """記錄查詢（審計用）"""
          import datetime
          log_entry = {
              'timestamp': datetime.datetime.now().isoformat(),
              'diagnosis': diagnosis,
              'genotype': genotype
          }
          self.audit_log.append(log_entry)
          logger.info(f"Drug recommendation query: {log_entry}")
  ```

#### 3.3 API 與 WebUI 整合

- [ ] 創建藥物建議 API 端點 📅 0.5 天
  ```python
  # src/api/routes/treatment.py
  
  from fastapi import APIRouter, HTTPException, Depends
  from pydantic import BaseModel
  
  router = APIRouter(prefix="/api/v2/treatment", tags=["treatment"])
  
  class TreatmentRequest(BaseModel):
      diagnosis_results: dict
      patient_genotype: list = None
      patient_allergies: list = None
      acknowledge_disclaimer: bool  # ⚠️ 必須確認
  
  @router.post("/suggest-drugs")
  async def suggest_drug_treatments(request: TreatmentRequest):
      """
      藥物治療建議（⚠️ 研究參考）
      
      ⚠️ 警告：必須在請求中確認免責聲明
      """
      # 強制確認免責聲明
      if not request.acknowledge_disclaimer:
          raise HTTPException(
              status_code=400,
              detail="必須確認免責聲明才能使用此功能"
          )
      
      # 生成建議
      recommender = DrugRecommendationEngine(drug_kg=global_drug_kg)
      result = recommender.suggest_treatments(
          request.diagnosis_results,
          request.patient_genotype,
          request.patient_allergies
      )
      
      return result
  ```

- [ ] WebUI 藥物建議介面 📅 1 天
  ```python
  # webui/components/treatment_tab.py
  
  with gr.Tab("💊 治療建議（研究參考）"):
      # ⚠️ 顯著的免責聲明
      gr.Markdown("""
      ## ⚠️⚠️⚠️ 重要警告 ⚠️⚠️⚠️
      
      此功能提供的藥物建議僅供研究與教學參考：
      - ❌ 不構成醫療建議或處方
      - ❌ 不得直接用於臨床決策
      - ✅ 必須由專業醫師審核確認
      - ✅ 僅作為治療思路的初步參考
      
      **使用此功能即表示您已理解並同意上述聲明**
      """, elem_classes=["warning-box"])
      
      acknowledge_checkbox = gr.Checkbox(
          label="✅ 我已閱讀並理解上述免責聲明",
          value=False
      )
      
      diagnosis_input = gr.JSON(label="診斷結果（從推理頁面複製）")
      
      genotype_input = gr.Textbox(
          label="患者基因型（可選）",
          placeholder="例如: CYP2D6*1/*4"
      )
      
      allergies_input = gr.Textbox(
          label="藥物過敏史（可選）",
          placeholder="例如: Penicillin, Aspirin"
      )
      
      suggest_btn = gr.Button(
          "生成藥物建議",
          variant="primary",
          interactive=False  # 預設不可點擊
      )
      
      # 結果顯示
      disclaimer_display = gr.Markdown()
      confidence_display = gr.Textbox(label="置信度")
      warnings_display = gr.Markdown(label="⚠️ 警告")
      suggestions_table = gr.DataFrame(
          headers=["藥物", "評分", "適應症", "證據等級"],
          label="建議藥物"
      )
      references_display = gr.Markdown(label="參考文獻")
      
      # 確認免責聲明後才能點擊
      def enable_button(acknowledged):
          return gr.Button.update(interactive=acknowledged)
      
      acknowledge_checkbox.change(
          fn=enable_button,
          inputs=[acknowledge_checkbox],
          outputs=[suggest_btn]
      )
  ```

**小計**: 📆 5-6 天

---

## 📋 Phase 2.6: 文獻檢索與可信度排序 (Week 3.5-4.5)

### 🟠 P1 - 文獻檢索引擎

#### 4.1 混合式檢索系統

- [ ] 創建 `src/literature/__init__.py` 🕐 5min

- [ ] 創建 `src/literature/hybrid_retrieval.py` 📅 2 天
  ```python
  """
  混合式文獻檢索系統
  
  模式:
      - offline: 僅使用預下載的 Pubtator 資料
      - online: 允許即時查詢 PubMed API（需醫院批准）
  """
  from typing import Dict, List
  import requests
  from src.data.parsers.pubtator_parser import PubtatorParser
  
  class HybridLiteratureRetrieval:
      """混合式文獻檢索與排序"""
      
      def __init__(self, mode='offline', pubmed_api_key=None):
          """
          Args:
              mode: 'offline' 或 'online'
              pubmed_api_key: PubMed API 金鑰（線上模式必需）
          """
          self.mode = mode
          
          # 離線資料庫（Pubtator 預下載）
          self.offline_db = self._load_pubtator_database()
          
          # 線上 API（可選）
          if mode == 'online':
              if not pubmed_api_key:
                  raise ValueError("線上模式需要 PubMed API 金鑰")
              self.pubmed_api = PubMedAPI(api_key=pubmed_api_key)
          else:
              self.pubmed_api = None
      
      def retrieve_and_rank(
          self,
          diagnosis_results: Dict,
          max_results: int = 10
      ) -> List[Dict]:
          """
          檢索並排序相關文獻
          
          Returns:
              List[{
                  'pmid': str,
                  'title': str,
                  'abstract': str,
                  'authors': List[str],
                  'journal': str,
                  'publication_date': str,
                  'relevance_score': float,       # 相關性
                  'credibility_score': float,     # 可信度
                  'combined_score': float,        # 綜合評分
                  'citation_count': int,
                  'journal_impact_factor': float,
                  'evidence_level': str,          # 證據等級
                  'study_type': str               # 研究類型
              }]
          """
          # 提取關鍵實體
          diseases = [d['id'] for d in diagnosis_results.get('top_diseases', [])]
          genes = [g['id'] for g in diagnosis_results.get('top_genes', [])]
          
          # 1. 離線檢索
          offline_papers = self._search_offline(diseases, genes, limit=50)
          
          # 2. 線上補充（如果允許）
          online_papers = []
          if self.mode == 'online' and self.pubmed_api:
              online_papers = self._search_online(diseases, genes, limit=20)
          
          # 3. 合併去重
          all_papers = self._merge_papers(offline_papers, online_papers)
          
          # 4. 多維度評分
          scored_papers = [
              self._score_paper(paper, diseases, genes)
              for paper in all_papers
          ]
          
          # 5. 排序
          ranked_papers = sorted(
              scored_papers,
              key=lambda x: x['combined_score'],
              reverse=True
          )
          
          return ranked_papers[:max_results]
      
      def _search_offline(self, diseases, genes, limit):
          """從 Pubtator 本地資料庫搜尋"""
          papers = []
          
          # 查詢疾病相關文獻
          for disease in diseases:
              disease_papers = self.offline_db.search_by_entity(
                  entity_id=disease,
                  entity_type='disease',
                  limit=limit
              )
              papers.extend(disease_papers)
          
          # 查詢基因相關文獻
          for gene in genes:
              gene_papers = self.offline_db.search_by_entity(
                  entity_id=gene,
                  entity_type='gene',
                  limit=limit
              )
              papers.extend(gene_papers)
          
          return papers
      
      def _search_online(self, diseases, genes, limit):
          """從 PubMed API 搜尋"""
          if not self.pubmed_api:
              return []
          
          # 構建查詢字串
          query_terms = diseases + genes + ['rare disease']
          query = ' AND '.join(query_terms)
          
          # 呼叫 API
          papers = self.pubmed_api.search(
              query=query,
              max_results=limit,
              sort='relevance'
          )
          
          return papers
      
      def _merge_papers(self, offline_papers, online_papers):
          """合併並去重"""
          # 使用 PMID 去重
          seen_pmids = set()
          merged = []
          
          for paper in offline_papers + online_papers:
              pmid = paper.get('pmid')
              if pmid and pmid not in seen_pmids:
                  seen_pmids.add(pmid)
                  merged.append(paper)
          
          return merged
      
      def _score_paper(self, paper, diseases, genes):
          """
          多維度評分
          
          維度:
              1. 相關性 (40%): 與診斷實體的相關程度
              2. 可信度 (30%): 期刊、證據等級
              3. 時效性 (20%): 發表時間
              4. 影響力 (10%): 引用次數
          """
          # 1. 相關性評分
          relevance_score = self._compute_relevance(paper, diseases, genes)
          
          # 2. 可信度評分
          credibility_score = self._compute_credibility(paper)
          
          # 3. 時效性評分
          recency_score = self._compute_recency(paper)
          
          # 4. 影響力評分
          impact_score = self._compute_impact(paper)
          
          # 綜合評分
          combined_score = (
              0.4 * relevance_score +
              0.3 * credibility_score +
              0.2 * recency_score +
              0.1 * impact_score
          )
          
          # 添加評分到論文資訊
          paper['relevance_score'] = relevance_score
          paper['credibility_score'] = credibility_score
          paper['combined_score'] = combined_score
          
          return paper
      
      def _compute_relevance(self, paper, diseases, genes):
          """
          計算相關性
          
          方法:
              - 實體共現：論文中提及的疾病/基因數量
              - 標題匹配：關鍵詞在標題中出現
              - 摘要匹配：關鍵詞在摘要中出現
          """
          score = 0.0
          
          # 檢查標題
          title = paper.get('title', '').lower()
          for entity in diseases + genes:
              if entity.lower() in title:
                  score += 0.3
          
          # 檢查摘要
          abstract = paper.get('abstract', '').lower()
          for entity in diseases + genes:
              if entity.lower() in abstract:
                  score += 0.1
          
          # 歸一化
          return min(score, 1.0)
      
      def _compute_credibility(self, paper):
          """
          計算可信度
          
          因素:
              - 期刊影響因子 (40%)
              - 證據等級 (30%): Meta-analysis > RCT > Cohort > Case
              - 作者機構 (20%): 頂級醫學中心加分
              - 同行評審 (10%)
          """
          score = 0.0
          
          # 期刊影響因子（假設已有資料）
          if 'journal_impact_factor' in paper:
              if_score = min(paper['journal_impact_factor'] / 50.0, 1.0)
              score += if_score * 0.4
          
          # 證據等級
          evidence_weights = {
              'meta-analysis': 1.0,
              'systematic_review': 0.9,
              'randomized_controlled_trial': 0.8,
              'cohort_study': 0.6,
              'case_control_study': 0.5,
              'case_report': 0.3,
              'review': 0.4
          }
          evidence_level = paper.get('evidence_level', 'case_report')
          score += evidence_weights.get(evidence_level, 0.3) * 0.3
          
          # 作者機構（簡化版）
          affiliations = paper.get('affiliations', [])
          top_institutions = [
              'Harvard', 'Stanford', 'Mayo Clinic', 'Johns Hopkins',
              'NIH', 'Cambridge', 'Oxford'
          ]
          if any(inst in str(affiliations) for inst in top_institutions):
              score += 0.2
          
          # 同行評審（預設為是）
          score += 0.1
          
          return min(score, 1.0)
      
      def _compute_recency(self, paper):
          """
          計算時效性
          
          規則:
              - 5年內: 1.0
              - 10年內: 0.7
              - 15年內: 0.5
              - 更早: 0.3
          """
          import datetime
          
          pub_date = paper.get('publication_date')
          if not pub_date:
              return 0.5
          
          try:
              pub_year = int(pub_date.split('-')[0])
              current_year = datetime.datetime.now().year
              years_ago = current_year - pub_year
              
              if years_ago <= 5:
                  return 1.0
              elif years_ago <= 10:
                  return 0.7
              elif years_ago <= 15:
                  return 0.5
              else:
                  return 0.3
          except:
              return 0.5
      
      def _compute_impact(self, paper):
          """
          計算影響力（引用次數）
          
          歸一化：引用次數 / 1000
          """
          citations = paper.get('citation_count', 0)
          return min(citations / 1000.0, 1.0)
  ```

#### 4.2 PubMed API 包裝器

- [ ] 創建 `src/literature/pubmed_api.py` 📅 0.5 天
  ```python
  """
  PubMed E-utilities API 包裝器
  """
  import requests
  import time
  from typing import List, Dict
  
  class PubMedAPI:
      """PubMed API 客戶端"""
      
      BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
      
      def __init__(self, api_key=None, rate_limit=3):
          """
          Args:
              api_key: NCBI API 金鑰
              rate_limit: 每秒請求數限制
          """
          self.api_key = api_key
          self.rate_limit = rate_limit
          self.last_request_time = 0
      
      def search(self, query: str, max_results: int = 20, sort='relevance'):
          """
          搜尋 PubMed
          
          API: esearch + efetch
          """
          # 1. 搜尋獲取 PMID 列表
          pmids = self._esearch(query, max_results, sort)
          
          # 2. 獲取詳細資訊
          papers = self._efetch(pmids)
          
          return papers
      
      def _esearch(self, query, max_results, sort):
          """搜尋 PMID"""
          self._wait_for_rate_limit()
          
          params = {
              'db': 'pubmed',
              'term': query,
              'retmax': max_results,
              'retmode': 'json',
              'sort': sort
          }
          
          if self.api_key:
              params['api_key'] = self.api_key
          
          response = requests.get(f"{self.BASE_URL}esearch.fcgi", params=params)
          data = response.json()
          
          pmids = data.get('esearchresult', {}).get('idlist', [])
          return pmids
      
      def _efetch(self, pmids):
          """獲取論文詳細資訊"""
          if not pmids:
              return []
          
          self._wait_for_rate_limit()
          
          params = {
              'db': 'pubmed',
              'id': ','.join(pmids),
              'retmode': 'xml'
          }
          
          if self.api_key:
              params['api_key'] = self.api_key
          
          response = requests.get(f"{self.BASE_URL}efetch.fcgi", params=params)
          
          # 解析 XML（簡化版）
          papers = self._parse_pubmed_xml(response.text)
          return papers
      
      def _wait_for_rate_limit(self):
          """遵守速率限制"""
          elapsed = time.time() - self.last_request_time
          wait_time = 1.0 / self.rate_limit
          
          if elapsed < wait_time:
              time.sleep(wait_time - elapsed)
          
          self.last_request_time = time.time()
      
      def _parse_pubmed_xml(self, xml_text):
          """解析 PubMed XML（簡化版）"""
          # TODO: 完整的 XML 解析
          # 這裡僅示意
          return []
  ```

#### 4.3 API 與 WebUI 整合

- [ ] 創建文獻檢索 API 端點 📅 0.5 天
  ```python
  # src/api/routes/literature.py
  
  from fastapi import APIRouter
  from pydantic import BaseModel
  
  router = APIRouter(prefix="/api/v2/literature", tags=["literature"])
  
  class LiteratureRequest(BaseModel):
      diagnosis_results: dict
      mode: str = 'offline'  # 'offline' 或 'online'
      max_results: int = 10
  
  @router.post("/search")
  async def search_literature(request: LiteratureRequest):
      """
      檢索相關文獻
      
      模式:
          - offline: 僅本地 Pubtator 資料
          - online: 允許 PubMed API（需設定）
      """
      retriever = HybridLiteratureRetrieval(mode=request.mode)
      papers = retriever.retrieve_and_rank(
          request.diagnosis_results,
          max_results=request.max_results
      )
      return {'papers': papers}
  ```

- [ ] WebUI 文獻檢索介面 📅 1 天
  ```python
  # webui/components/literature_tab.py
  
  with gr.Tab("📚 文獻檢索"):
      gr.Markdown("## 相關文獻檢索與排序")
      
      diagnosis_input = gr.JSON(label="診斷結果（從推理頁面複製）")
      
      mode_radio = gr.Radio(
          choices=["offline", "online"],
          value="offline",
          label="檢索模式",
          info="offline: 僅本地資料 | online: 包含即時 PubMed 查詢"
      )
      
      max_results_slider = gr.Slider(
          minimum=5,
          maximum=50,
          value=10,
          step=5,
          label="最大結果數"
      )
      
      search_btn = gr.Button("檢索文獻", variant="primary")
      
      # 結果顯示
      papers_table = gr.DataFrame(
          headers=[
              "PMID", "標題", "期刊", "年份",
              "相關性", "可信度", "綜合評分"
          ],
          label="檢索結果"
      )
      
      # 選中論文的詳細資訊
      paper_detail = gr.HTML(label="論文詳情")
  ```

**小計**: 📆 4-5 天

---

## 📊 驗收標準

### 功能 1: 症狀關聯分析 ✅
- [ ] 能分析兩症狀的多維度關聯
- [ ] 提供生物學機制推斷
- [ ] API 可正常運作
- [ ] WebUI 可視化展示

### 功能 2: 基因型-表型排序 ✅
- [ ] 整合 ClinVar 變異資訊
- [ ] 提供外顯率估算
- [ ] 包含文獻支持評分
- [ ] 詳細的證據強度分類

### 功能 3: 藥物建議 ✅ （⚠️ 附帶警告）
- [ ] **免責聲明在所有輸出中顯著標示**
- [ ] 藥物知識圖譜構建完成
- [ ] 基因型藥物代謝過濾
- [ ] 置信度評估與警告生成
- [ ] 所有查詢記錄審計日誌

### 功能 4: 文獻檢索 ✅
- [ ] 離線模式完全可用
- [ ] 線上模式（可選）實現
- [ ] 多維度可信度評分
- [ ] 排序結果準確

---

## 🔧 依賴與注意事項

### 外部套件
```bash
# 功能 1-2: 分析模組
# （使用現有套件，無新增）

# 功能 3: 藥物建議
pip install drugbank-parser==0.1.0  # 或手動解析

# 功能 4: 文獻檢索
pip install biopython==1.81  # PubMed API
pip install xmltodict==0.13.0  # XML 解析
```

### 資料需求
- [ ] 下載 DrugBank 完整資料庫（需註冊）
- [ ] 下載 Pubtator 3.0 預處理資料
- [ ] 準備 ClinVar 變異資料（可透過 API）

### 呼叫現有模組
```python
# 功能 1: 症狀關聯
from src.kg.builder import KnowledgeGraphBuilder  # ✅
from src.retrieval.path_retriever import PathRetriever  # ✅
from src.ontology.similarity import OntologySimilarity  # ✅

# 功能 2: 基因排序
from src.models.tasks.gene_scoring import GeneScoring  # ✅

# 功能 4: 文獻檢索
# （新模組，無依賴衝突）
```

---

## 🎯 優先級建議

| 功能 | Phase 2 | Phase 3 | 備註 |
|------|---------|---------|------|
| **症狀關聯分析** | ✅ 實現 | - | 核心功能擴展 |
| **基因排序增強** | ✅ 實現 | - | 提升診斷價值 |
| **藥物建議** | 🟡 可選 | ✅ 完整 | ⚠️ 需審慎評估 |
| **文獻檢索** | ✅ 離線 | ✅ 線上 | 實用性高 |

---

## 📅 時間規劃

```
Week 1 (Day 1-5): 症狀關聯分析
├── Day 1-2: 核心分析器實現
├── Day 3: API 與 WebUI 整合
└── Day 4-5: 測試與調優

Week 2 (Day 1-5): 基因排序增強 + 藥物建議（起步）
├── Day 1-2: 增強排序模組
├── Day 3: DrugBank 整合
├── Day 4-5: 藥物知識圖譜構建

Week 3 (Day 1-5): 藥物建議（完成）
├── Day 1-2: 藥物建議引擎
├── Day 3: API 整合 + 免責聲明
├── Day 4: WebUI 整合
└── Day 5: 測試與審核

Week 4 (Day 1-5): 文獻檢索
├── Day 1-2: 混合檢索引擎
├── Day 3: 可信度評分系統
├── Day 4: API 與 WebUI
└── Day 5: 端到端測試
```

**總計**: 📆 4-5 週

---

## ⚠️ 特別注意事項

### 藥物建議功能
```
🔴 法律風險評估清單：

1. ✅ 免責聲明在所有介面顯著標示
2. ✅ 使用者必須明確確認免責聲明
3. ✅ 所有查詢記錄審計日誌
4. ✅ 定期由醫學專家審核建議品質
5. ✅ 獲得醫院倫理委員會批准
6. ✅ 明確標示為"研究參考"而非"臨床建議"
7. ✅ 不提供劑量資訊
8. ✅ 不提供用藥時間表

建議：
- 藥物建議功能預設關閉
- 需管理員權限啟用
- 每次使用需重新確認免責聲明
```

---

## ✅ 檢查清單

### Phase 2.3 完成確認
- [ ] 症狀關聯分析 API 可用
- [ ] WebUI 可視化完整
- [ ] 生物學機制推斷準確
- [ ] 單元測試覆蓋率 > 70%

### Phase 2.4 完成確認
- [ ] ClinVar API 整合完成
- [ ] 外顯率估算實現
- [ ] 文獻支持評分準確
- [ ] API 端點測試通過

### Phase 2.5 完成確認（藥物建議）
- [ ] **免責聲明審核通過**
- [ ] **醫學專家審核通過**
- [ ] **倫理委員會批准文件**
- [ ] DrugBank 知識圖譜構建完成
- [ ] 審計日誌系統運作
- [ ] 安全性測試通過

### Phase 2.6 完成確認
- [ ] 離線檢索完全可用
- [ ] 線上檢索（可選）實現
- [ ] 可信度評分驗證
- [ ] 排序結果與醫學專家評估一致

---

**版本**: v1.0  
**創建日期**: 2025-11-04  
**負責人**: TBD  
**醫學顧問**: TBD（藥物建議功能必需）  
**法律顧問**: TBD（藥物建議功能必需）
