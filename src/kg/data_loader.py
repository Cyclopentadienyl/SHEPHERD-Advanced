"""
SHEPHERD-Advanced Knowledge Graph Data Loader
=============================================
Mini-batch data loading with subgraph sampling for GNN training under VRAM constraints.

Module: src/kg/data_loader.py
Absolute Path: /home/user/SHEPHERD-Advanced/src/kg/data_loader.py

Purpose:
    Provide efficient data loading for heterogeneous GNN training on large-scale
    knowledge graphs. Implements subgraph sampling to enable training within
    16GB VRAM limit (Windows constraint).

Components:
    - DataLoaderConfig: Configuration for batch size, sampling, workers
    - SubgraphSampler: Mini-batch subgraph extraction (neighbor/random_walk/khop)
    - NegativeSampler: Negative sample generation for contrastive learning
    - DiagnosisSample: Single patient-disease training sample
    - DiagnosisDataset: PyTorch Dataset for diagnosis task
    - DiagnosisDataLoader: Iterator yielding batches with subgraphs

Dependencies:
    - torch: Tensor operations and DataLoader
    - torch.nn.functional: Padding operations
    - numpy: Frequency-based sampling
    - random: Uniform sampling
    - collections.defaultdict: Adjacency list construction

Input:
    - samples: List[DiagnosisSample] - Training samples (patient phenotypes -> disease)
    - graph_data: Dict containing:
        - x_dict: {node_type: Tensor} - Node features
        - edge_index_dict: {(src, rel, dst): Tensor} - Edge indices
        - num_nodes_dict: {node_type: int} - Node counts
    - config: DataLoaderConfig - Sampling and batching parameters

Output:
    - Iterator[Dict] yielding per batch:
        - batch: Collated sample data (phenotype_ids, disease_ids, masks)
        - subgraph_x_dict: Subgraph node features
        - subgraph_edge_index_dict: Subgraph edges (remapped indices)
        - node_mapping: Original -> subgraph index mapping

Called by:
    - src/training/trainer.py (training loop)
    - scripts/train_model.py (data pipeline setup)

Note:
    This module handles TRAINING-TIME mini-batch sampling. For INFERENCE-TIME
    subgraph extraction, see src/retrieval/subgraph_sampler.py which implements
    the SubgraphSamplerProtocol from src/core/protocols.py.

Version: 1.0.0
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

# Type aliases
EdgeType = Tuple[str, str, str]


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class DataLoaderConfig:
    """資料載入器配置"""
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True

    # 子圖採樣設定
    num_neighbors: List[int] = field(default_factory=lambda: [15, 10, 5])
    max_subgraph_nodes: int = 5000  # 限制子圖大小以適應 16GB VRAM
    sampling_strategy: str = "neighbor"  # "neighbor", "random_walk", "khop"

    # 負樣本設定
    num_negative_samples: int = 5
    negative_sampling_strategy: str = "uniform"  # "uniform", "frequency", "hard"

    # 預取設定
    prefetch_factor: int = 2


# =============================================================================
# Subgraph Sampler
# =============================================================================
class SubgraphSampler:
    """
    子圖採樣器

    針對大型知識圖譜進行子圖採樣，確保能在有限 VRAM 中訓練
    支援多種採樣策略：
    - neighbor: 鄰居採樣 (類似 GraphSAGE)
    - random_walk: 隨機遊走採樣
    - khop: K-hop 子圖
    """

    def __init__(
        self,
        edge_index_dict: Dict[EdgeType, Tensor],
        num_nodes_dict: Dict[str, int],
        config: Optional[DataLoaderConfig] = None,
    ):
        """
        Args:
            edge_index_dict: {(src_type, rel, dst_type): (2, num_edges)}
            num_nodes_dict: {node_type: num_nodes}
            config: 資料載入器配置
        """
        self.edge_index_dict = edge_index_dict
        self.num_nodes_dict = num_nodes_dict
        self.config = config or DataLoaderConfig()

        # 建立鄰接表以加速採樣
        self._build_adjacency_lists()

    def _build_adjacency_lists(self) -> None:
        """建立鄰接表"""
        self.adj_lists: Dict[str, Dict[int, List[Tuple[str, int, EdgeType]]]] = {
            node_type: defaultdict(list)
            for node_type in self.num_nodes_dict.keys()
        }

        for edge_type, edge_index in self.edge_index_dict.items():
            src_type, rel, dst_type = edge_type
            src_nodes = edge_index[0].tolist()
            dst_nodes = edge_index[1].tolist()

            for src, dst in zip(src_nodes, dst_nodes):
                # 出邊
                self.adj_lists[src_type][src].append((dst_type, dst, edge_type))
                # 入邊 (反向)
                self.adj_lists[dst_type][dst].append((src_type, src, edge_type))

    def sample_subgraph(
        self,
        seed_nodes: Dict[str, Tensor],
        num_hops: int = 2,
    ) -> Tuple[Dict[str, Tensor], Dict[EdgeType, Tensor], Dict[str, Tensor]]:
        """
        從種子節點採樣子圖

        Args:
            seed_nodes: {node_type: seed_node_indices}
            num_hops: 採樣跳數

        Returns:
            (subgraph_x_dict, subgraph_edge_index_dict, node_mapping_dict)
            - subgraph_x_dict: {node_type: node_indices_in_original_graph}
            - subgraph_edge_index_dict: {edge_type: edge_index_in_subgraph}
            - node_mapping_dict: {node_type: original_to_subgraph_mapping}
        """
        if self.config.sampling_strategy == "neighbor":
            return self._neighbor_sampling(seed_nodes, num_hops)
        elif self.config.sampling_strategy == "random_walk":
            return self._random_walk_sampling(seed_nodes, num_hops)
        else:
            return self._khop_sampling(seed_nodes, num_hops)

    def _neighbor_sampling(
        self,
        seed_nodes: Dict[str, Tensor],
        num_hops: int,
    ) -> Tuple[Dict[str, Tensor], Dict[EdgeType, Tensor], Dict[str, Tensor]]:
        """鄰居採樣 (GraphSAGE 風格)"""
        # 收集所有層的節點
        sampled_nodes: Dict[str, Set[int]] = {
            node_type: set(nodes.tolist())
            for node_type, nodes in seed_nodes.items()
        }

        # 對於未指定的節點類型，初始化空集合
        for node_type in self.num_nodes_dict.keys():
            if node_type not in sampled_nodes:
                sampled_nodes[node_type] = set()

        num_neighbors = self.config.num_neighbors[:num_hops]

        for hop, n_neighbors in enumerate(num_neighbors):
            # 收集當前層需要擴展的節點
            frontier: Dict[str, Set[int]] = {
                nt: set(nodes) for nt, nodes in sampled_nodes.items()
            }

            for node_type, nodes in frontier.items():
                for node in nodes:
                    neighbors = self.adj_lists[node_type].get(node, [])

                    # 採樣鄰居
                    if len(neighbors) > n_neighbors:
                        sampled = random.sample(neighbors, n_neighbors)
                    else:
                        sampled = neighbors

                    for neighbor_type, neighbor_node, _ in sampled:
                        sampled_nodes[neighbor_type].add(neighbor_node)

            # 檢查節點數限制
            total_nodes = sum(len(nodes) for nodes in sampled_nodes.values())
            if total_nodes > self.config.max_subgraph_nodes:
                break

        return self._build_subgraph(sampled_nodes)

    def _random_walk_sampling(
        self,
        seed_nodes: Dict[str, Tensor],
        num_hops: int,
    ) -> Tuple[Dict[str, Tensor], Dict[EdgeType, Tensor], Dict[str, Tensor]]:
        """隨機遊走採樣"""
        sampled_nodes: Dict[str, Set[int]] = {
            node_type: set()
            for node_type in self.num_nodes_dict.keys()
        }

        walk_length = num_hops * 2
        num_walks = self.config.num_neighbors[0] if self.config.num_neighbors else 10

        for node_type, nodes in seed_nodes.items():
            for seed in nodes.tolist():
                sampled_nodes[node_type].add(seed)

                # 執行多次隨機遊走
                for _ in range(num_walks):
                    current_node = seed
                    current_type = node_type

                    for _ in range(walk_length):
                        neighbors = self.adj_lists[current_type].get(current_node, [])
                        if not neighbors:
                            break

                        # 隨機選擇下一個節點
                        next_type, next_node, _ = random.choice(neighbors)
                        sampled_nodes[next_type].add(next_node)
                        current_node = next_node
                        current_type = next_type

                        # 檢查節點數限制
                        total = sum(len(n) for n in sampled_nodes.values())
                        if total > self.config.max_subgraph_nodes:
                            return self._build_subgraph(sampled_nodes)

        return self._build_subgraph(sampled_nodes)

    def _khop_sampling(
        self,
        seed_nodes: Dict[str, Tensor],
        num_hops: int,
    ) -> Tuple[Dict[str, Tensor], Dict[EdgeType, Tensor], Dict[str, Tensor]]:
        """K-hop 子圖採樣"""
        sampled_nodes: Dict[str, Set[int]] = {
            node_type: set(nodes.tolist()) if node_type in seed_nodes else set()
            for node_type in self.num_nodes_dict.keys()
        }

        for _ in range(num_hops):
            new_nodes: Dict[str, Set[int]] = {nt: set() for nt in self.num_nodes_dict}

            for node_type, nodes in sampled_nodes.items():
                for node in nodes:
                    neighbors = self.adj_lists[node_type].get(node, [])
                    for neighbor_type, neighbor_node, _ in neighbors:
                        if neighbor_node not in sampled_nodes[neighbor_type]:
                            new_nodes[neighbor_type].add(neighbor_node)

            # 合併新節點
            for node_type, nodes in new_nodes.items():
                sampled_nodes[node_type].update(nodes)

            # 檢查節點數限制
            total = sum(len(n) for n in sampled_nodes.values())
            if total > self.config.max_subgraph_nodes:
                break

        return self._build_subgraph(sampled_nodes)

    def _build_subgraph(
        self,
        sampled_nodes: Dict[str, Set[int]],
    ) -> Tuple[Dict[str, Tensor], Dict[EdgeType, Tensor], Dict[str, Tensor]]:
        """從採樣節點建立子圖"""
        # 建立節點映射 (原始索引 -> 子圖索引)
        node_mapping: Dict[str, Dict[int, int]] = {}
        subgraph_nodes: Dict[str, Tensor] = {}

        for node_type, nodes in sampled_nodes.items():
            if nodes:
                sorted_nodes = sorted(nodes)
                node_mapping[node_type] = {
                    orig: new for new, orig in enumerate(sorted_nodes)
                }
                subgraph_nodes[node_type] = torch.tensor(sorted_nodes, dtype=torch.long)
            else:
                node_mapping[node_type] = {}
                subgraph_nodes[node_type] = torch.tensor([], dtype=torch.long)

        # 建立子圖邊
        subgraph_edges: Dict[EdgeType, Tensor] = {}

        for edge_type, edge_index in self.edge_index_dict.items():
            src_type, _, dst_type = edge_type
            src_mapping = node_mapping[src_type]
            dst_mapping = node_mapping[dst_type]

            if not src_mapping or not dst_mapping:
                subgraph_edges[edge_type] = torch.tensor([[], []], dtype=torch.long)
                continue

            # 篩選在子圖中的邊
            new_src = []
            new_dst = []

            src_nodes = edge_index[0].tolist()
            dst_nodes = edge_index[1].tolist()

            for src, dst in zip(src_nodes, dst_nodes):
                if src in src_mapping and dst in dst_mapping:
                    new_src.append(src_mapping[src])
                    new_dst.append(dst_mapping[dst])

            subgraph_edges[edge_type] = torch.tensor(
                [new_src, new_dst], dtype=torch.long
            )

        # 建立映射張量
        mapping_tensors: Dict[str, Tensor] = {}
        for node_type, mapping in node_mapping.items():
            if mapping:
                max_idx = max(mapping.keys()) + 1
                tensor = torch.full((max_idx,), -1, dtype=torch.long)
                for orig, new in mapping.items():
                    tensor[orig] = new
                mapping_tensors[node_type] = tensor

        return subgraph_nodes, subgraph_edges, mapping_tensors


# =============================================================================
# Negative Sampler
# =============================================================================
class NegativeSampler:
    """
    負樣本採樣器

    用於對比學習和連結預測的負樣本生成
    """

    def __init__(
        self,
        num_nodes_dict: Dict[str, int],
        edge_index_dict: Dict[EdgeType, Tensor],
        strategy: str = "uniform",
        num_negative: int = 5,
    ):
        """
        Args:
            num_nodes_dict: {node_type: num_nodes}
            edge_index_dict: {edge_type: edge_index}
            strategy: 採樣策略 ("uniform", "frequency", "hard")
            num_negative: 每個正樣本的負樣本數
        """
        self.num_nodes_dict = num_nodes_dict
        self.edge_index_dict = edge_index_dict
        self.strategy = strategy
        self.num_negative = num_negative

        # 建立正樣本集合 (用於排除)
        self._build_positive_sets()

        # 計算節點頻率 (用於 frequency-based 採樣)
        if strategy == "frequency":
            self._compute_node_frequencies()

    def _build_positive_sets(self) -> None:
        """建立正樣本集合"""
        self.positive_edges: Dict[EdgeType, Set[Tuple[int, int]]] = {}

        for edge_type, edge_index in self.edge_index_dict.items():
            edges = set()
            src_nodes = edge_index[0].tolist()
            dst_nodes = edge_index[1].tolist()
            for src, dst in zip(src_nodes, dst_nodes):
                edges.add((src, dst))
            self.positive_edges[edge_type] = edges

    def _compute_node_frequencies(self) -> None:
        """計算節點頻率"""
        self.node_frequencies: Dict[str, np.ndarray] = {}

        for node_type, num_nodes in self.num_nodes_dict.items():
            freq = np.zeros(num_nodes)

            for edge_type, edge_index in self.edge_index_dict.items():
                src_type, _, dst_type = edge_type

                if src_type == node_type:
                    src_nodes = edge_index[0].numpy()
                    np.add.at(freq, src_nodes, 1)

                if dst_type == node_type:
                    dst_nodes = edge_index[1].numpy()
                    np.add.at(freq, dst_nodes, 1)

            # 轉換為機率分佈 (加平滑)
            freq = freq + 1
            self.node_frequencies[node_type] = freq / freq.sum()

    def sample_negative_edges(
        self,
        edge_type: EdgeType,
        positive_edges: Tensor,
    ) -> Tensor:
        """
        為給定的正邊採樣負邊

        Args:
            edge_type: 邊類型
            positive_edges: (2, num_positive) 正邊

        Returns:
            (num_positive, num_negative, 2) 負邊
        """
        src_type, _, dst_type = edge_type
        num_dst = self.num_nodes_dict[dst_type]
        num_positive = positive_edges.shape[1]
        positive_set = self.positive_edges.get(edge_type, set())

        negative_edges = []

        for i in range(num_positive):
            src = positive_edges[0, i].item()
            neg_dsts = []

            attempts = 0
            max_attempts = self.num_negative * 10

            while len(neg_dsts) < self.num_negative and attempts < max_attempts:
                if self.strategy == "frequency" and dst_type in self.node_frequencies:
                    neg_dst = np.random.choice(
                        num_dst, p=self.node_frequencies[dst_type]
                    )
                else:
                    neg_dst = random.randint(0, num_dst - 1)

                # 排除正樣本
                if (src, neg_dst) not in positive_set:
                    neg_dsts.append(neg_dst)

                attempts += 1

            # 如果採樣不足，用隨機填充
            while len(neg_dsts) < self.num_negative:
                neg_dsts.append(random.randint(0, num_dst - 1))

            negative_edges.append([[src, dst] for dst in neg_dsts])

        return torch.tensor(negative_edges, dtype=torch.long)

    def sample_negative_nodes(
        self,
        node_type: str,
        positive_nodes: Tensor,
        exclude_nodes: Optional[Set[int]] = None,
    ) -> Tensor:
        """
        採樣負節點

        Args:
            node_type: 節點類型
            positive_nodes: (num_positive,) 正節點
            exclude_nodes: 需要排除的節點

        Returns:
            (num_positive, num_negative) 負節點
        """
        num_nodes = self.num_nodes_dict[node_type]
        num_positive = positive_nodes.shape[0]
        exclude = exclude_nodes or set()
        exclude = exclude.union(set(positive_nodes.tolist()))

        negative_nodes = []

        for _ in range(num_positive):
            neg_samples = []
            attempts = 0
            max_attempts = self.num_negative * 10

            while len(neg_samples) < self.num_negative and attempts < max_attempts:
                if self.strategy == "frequency" and node_type in self.node_frequencies:
                    neg = np.random.choice(
                        num_nodes, p=self.node_frequencies[node_type]
                    )
                else:
                    neg = random.randint(0, num_nodes - 1)

                if neg not in exclude:
                    neg_samples.append(neg)

                attempts += 1

            while len(neg_samples) < self.num_negative:
                neg_samples.append(random.randint(0, num_nodes - 1))

            negative_nodes.append(neg_samples)

        return torch.tensor(negative_nodes, dtype=torch.long)


# =============================================================================
# Diagnosis Dataset
# =============================================================================
@dataclass
class DiagnosisSample:
    """單個診斷樣本"""
    patient_id: str
    phenotype_ids: List[int]  # 在圖中的索引
    disease_id: int  # 正確疾病在圖中的索引
    candidate_disease_ids: Optional[List[int]] = None  # 候選疾病
    gene_ids: Optional[List[int]] = None  # 候選基因


class DiagnosisDataset(Dataset):
    """
    診斷任務資料集

    每個樣本包含：患者表型 -> 正確疾病
    """

    def __init__(
        self,
        samples: List[DiagnosisSample],
        num_diseases: int,
        num_negative_diseases: int = 10,
        include_all_candidates: bool = False,
    ):
        """
        Args:
            samples: 診斷樣本列表
            num_diseases: 疾病節點總數
            num_negative_diseases: 負樣本疾病數
            include_all_candidates: 是否包含所有候選疾病
        """
        self.samples = samples
        self.num_diseases = num_diseases
        self.num_negative_diseases = num_negative_diseases
        self.include_all_candidates = include_all_candidates

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        item = {
            "patient_id": sample.patient_id,
            "phenotype_ids": torch.tensor(sample.phenotype_ids, dtype=torch.long),
            "disease_id": torch.tensor(sample.disease_id, dtype=torch.long),
        }

        # 採樣負樣本疾病
        if self.include_all_candidates and sample.candidate_disease_ids:
            item["candidate_ids"] = torch.tensor(
                sample.candidate_disease_ids, dtype=torch.long
            )
        else:
            # 隨機採樣負樣本
            negative_diseases = []
            positive = sample.disease_id
            candidates = set(sample.candidate_disease_ids or [])

            while len(negative_diseases) < self.num_negative_diseases:
                neg = random.randint(0, self.num_diseases - 1)
                if neg != positive and neg not in candidates:
                    negative_diseases.append(neg)

            item["negative_disease_ids"] = torch.tensor(
                negative_diseases, dtype=torch.long
            )

        if sample.gene_ids:
            item["gene_ids"] = torch.tensor(sample.gene_ids, dtype=torch.long)

        return item


# =============================================================================
# Custom Collate Function
# =============================================================================
def diagnosis_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    自定義批次整理函數

    處理變長序列和異質數據
    """
    collated = {
        "patient_ids": [item["patient_id"] for item in batch],
        "disease_ids": torch.stack([item["disease_id"] for item in batch]),
    }

    # 填充表型序列
    max_phenotypes = max(item["phenotype_ids"].shape[0] for item in batch)
    phenotype_ids = []
    phenotype_mask = []

    for item in batch:
        pids = item["phenotype_ids"]
        pad_len = max_phenotypes - pids.shape[0]
        padded = F.pad(pids, (0, pad_len), value=-1)
        mask = torch.cat([
            torch.ones(pids.shape[0], dtype=torch.bool),
            torch.zeros(pad_len, dtype=torch.bool),
        ])
        phenotype_ids.append(padded)
        phenotype_mask.append(mask)

    collated["phenotype_ids"] = torch.stack(phenotype_ids)
    collated["phenotype_mask"] = torch.stack(phenotype_mask)

    # 處理負樣本疾病
    if "negative_disease_ids" in batch[0]:
        collated["negative_disease_ids"] = torch.stack([
            item["negative_disease_ids"] for item in batch
        ])

    # 處理候選疾病
    if "candidate_ids" in batch[0]:
        max_candidates = max(item["candidate_ids"].shape[0] for item in batch)
        candidate_ids = []
        candidate_mask = []

        for item in batch:
            cids = item["candidate_ids"]
            pad_len = max_candidates - cids.shape[0]
            padded = F.pad(cids, (0, pad_len), value=-1)
            mask = torch.cat([
                torch.ones(cids.shape[0], dtype=torch.bool),
                torch.zeros(pad_len, dtype=torch.bool),
            ])
            candidate_ids.append(padded)
            candidate_mask.append(mask)

        collated["candidate_ids"] = torch.stack(candidate_ids)
        collated["candidate_mask"] = torch.stack(candidate_mask)

    # 處理基因
    if "gene_ids" in batch[0] and batch[0]["gene_ids"] is not None:
        max_genes = max(
            item["gene_ids"].shape[0] for item in batch
            if "gene_ids" in item and item["gene_ids"] is not None
        )
        gene_ids = []
        gene_mask = []

        for item in batch:
            if "gene_ids" in item and item["gene_ids"] is not None:
                gids = item["gene_ids"]
                pad_len = max_genes - gids.shape[0]
                padded = F.pad(gids, (0, pad_len), value=-1)
                mask = torch.cat([
                    torch.ones(gids.shape[0], dtype=torch.bool),
                    torch.zeros(pad_len, dtype=torch.bool),
                ])
            else:
                padded = torch.full((max_genes,), -1, dtype=torch.long)
                mask = torch.zeros(max_genes, dtype=torch.bool)

            gene_ids.append(padded)
            gene_mask.append(mask)

        collated["gene_ids"] = torch.stack(gene_ids)
        collated["gene_mask"] = torch.stack(gene_mask)

    return collated


# =============================================================================
# Main Data Loader
# =============================================================================
class DiagnosisDataLoader:
    """
    診斷任務資料載入器

    整合子圖採樣和批次載入
    """

    def __init__(
        self,
        dataset: DiagnosisDataset,
        graph_data: Dict[str, Any],
        config: Optional[DataLoaderConfig] = None,
    ):
        """
        Args:
            dataset: 診斷資料集
            graph_data: 知識圖譜數據 {
                "x_dict": {node_type: features},
                "edge_index_dict": {edge_type: edge_index},
                "num_nodes_dict": {node_type: num_nodes},
            }
            config: 資料載入器配置
        """
        self.dataset = dataset
        self.graph_data = graph_data
        self.config = config or DataLoaderConfig()

        # 初始化子圖採樣器
        self.subgraph_sampler = SubgraphSampler(
            edge_index_dict=graph_data["edge_index_dict"],
            num_nodes_dict=graph_data["num_nodes_dict"],
            config=self.config,
        )

        # 初始化負樣本採樣器
        self.negative_sampler = NegativeSampler(
            num_nodes_dict=graph_data["num_nodes_dict"],
            edge_index_dict=graph_data["edge_index_dict"],
            strategy=self.config.negative_sampling_strategy,
            num_negative=self.config.num_negative_samples,
        )

        # 建立底層 DataLoader
        self._dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=diagnosis_collate_fn,
            prefetch_factor=self.config.prefetch_factor if self.config.num_workers > 0 else None,
        )

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """迭代批次"""
        for batch in self._dataloader:
            # 採樣子圖
            seed_nodes = self._get_seed_nodes(batch)
            subgraph_nodes, subgraph_edges, node_mapping = \
                self.subgraph_sampler.sample_subgraph(seed_nodes, num_hops=2)

            # 準備子圖特徵
            subgraph_x_dict = {}
            for node_type, indices in subgraph_nodes.items():
                if node_type in self.graph_data.get("x_dict", {}):
                    subgraph_x_dict[node_type] = self.graph_data["x_dict"][node_type][indices]

            # 更新批次中的索引到子圖索引
            batch = self._remap_indices(batch, node_mapping)

            yield {
                "batch": batch,
                "subgraph_x_dict": subgraph_x_dict,
                "subgraph_edge_index_dict": subgraph_edges,
                "node_mapping": node_mapping,
                "original_indices": subgraph_nodes,
            }

    def __len__(self) -> int:
        return len(self._dataloader)

    def _get_seed_nodes(self, batch: Dict[str, Any]) -> Dict[str, Tensor]:
        """從批次中提取種子節點"""
        seed_nodes = {}

        # 表型節點
        phenotype_ids = batch["phenotype_ids"]
        valid_phenotypes = phenotype_ids[batch["phenotype_mask"]]
        if len(valid_phenotypes) > 0:
            seed_nodes["phenotype"] = valid_phenotypes.unique()

        # 疾病節點
        disease_ids = batch["disease_ids"]
        all_diseases = [disease_ids]

        if "negative_disease_ids" in batch:
            all_diseases.append(batch["negative_disease_ids"].flatten())
        if "candidate_ids" in batch:
            valid_candidates = batch["candidate_ids"][batch["candidate_mask"]]
            all_diseases.append(valid_candidates)

        seed_nodes["disease"] = torch.cat(all_diseases).unique()

        # 基因節點 (如果有)
        if "gene_ids" in batch and batch.get("gene_mask") is not None:
            valid_genes = batch["gene_ids"][batch["gene_mask"]]
            if len(valid_genes) > 0:
                seed_nodes["gene"] = valid_genes.unique()

        return seed_nodes

    def _remap_indices(
        self,
        batch: Dict[str, Any],
        node_mapping: Dict[str, Tensor],
    ) -> Dict[str, Any]:
        """將批次中的索引重映射到子圖索引"""
        remapped = dict(batch)

        # 重映射表型索引
        if "phenotype" in node_mapping:
            pheno_map = node_mapping["phenotype"]
            old_ids = batch["phenotype_ids"].clone()
            mask = batch["phenotype_mask"]

            # 只重映射有效索引
            valid_mask = mask & (old_ids >= 0) & (old_ids < len(pheno_map))
            new_ids = old_ids.clone()
            new_ids[valid_mask] = pheno_map[old_ids[valid_mask]]
            remapped["phenotype_ids"] = new_ids

        # 重映射疾病索引
        if "disease" in node_mapping:
            disease_map = node_mapping["disease"]

            old_ids = batch["disease_ids"]
            valid_mask = (old_ids >= 0) & (old_ids < len(disease_map))
            new_ids = old_ids.clone()
            new_ids[valid_mask] = disease_map[old_ids[valid_mask]]
            remapped["disease_ids"] = new_ids

            if "negative_disease_ids" in batch:
                old_neg = batch["negative_disease_ids"]
                valid_mask = (old_neg >= 0) & (old_neg < len(disease_map))
                new_neg = old_neg.clone()
                new_neg[valid_mask] = disease_map[old_neg[valid_mask]]
                remapped["negative_disease_ids"] = new_neg

        return remapped


# =============================================================================
# Helper Functions
# =============================================================================
def create_diagnosis_dataloader(
    samples: List[DiagnosisSample],
    graph_data: Dict[str, Any],
    config: Optional[DataLoaderConfig] = None,
) -> DiagnosisDataLoader:
    """
    便捷函數：創建診斷資料載入器

    Args:
        samples: 診斷樣本列表
        graph_data: 知識圖譜數據
        config: 資料載入器配置

    Returns:
        DiagnosisDataLoader
    """
    config = config or DataLoaderConfig()

    dataset = DiagnosisDataset(
        samples=samples,
        num_diseases=graph_data["num_nodes_dict"].get("disease", 0),
        num_negative_diseases=config.num_negative_samples,
    )

    return DiagnosisDataLoader(
        dataset=dataset,
        graph_data=graph_data,
        config=config,
    )


# =============================================================================
# Module Exports
# =============================================================================
__all__ = [
    "DataLoaderConfig",
    "SubgraphSampler",
    "NegativeSampler",
    "DiagnosisSample",
    "DiagnosisDataset",
    "DiagnosisDataLoader",
    "diagnosis_collate_fn",
    "create_diagnosis_dataloader",
]
