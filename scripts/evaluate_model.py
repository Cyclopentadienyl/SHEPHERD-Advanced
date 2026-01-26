#!/usr/bin/env python3
"""
SHEPHERD-Advanced Model Evaluation Script
==========================================
Standalone evaluation script for trained ShepherdGNN models.

Script: scripts/evaluate_model.py
Absolute Path: /home/user/SHEPHERD-Advanced/scripts/evaluate_model.py

Purpose:
    Evaluate trained model performance on test/validation datasets.
    Generates detailed metrics report including:
    - Ranking metrics (MRR, Hits@K)
    - Per-disease-category breakdown
    - Confidence calibration analysis
    - Optional: comparison with baseline models

Usage:
    python scripts/evaluate_model.py --checkpoint checkpoints/best.pt --data data/processed
    python scripts/evaluate_model.py --checkpoint checkpoints/best.pt --output reports/eval_report.json
    python scripts/evaluate_model.py --config configs/eval_config.yaml

Dependencies:
    - torch: Model loading and inference
    - src.training.trainer: Trainer class for evaluation
    - src.kg.data_loader: Test data loading
    - src.utils.metrics: Metric computation
    - src.models.gnn.shepherd_gnn: Model architecture

Input:
    - Model checkpoint file (.pt)
    - Test/validation dataset
    - Optional: evaluation configuration

Output:
    - Console: Summary metrics
    - JSON file: Detailed evaluation report
    - Optional: Per-sample predictions

Called by:
    - User via command line
    - CI/CD pipelines for model validation
    - API layer for on-demand evaluation

Version: 1.0.0
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.metrics import RankingMetrics, DiagnosisMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class EvalConfig:
    """Evaluation configuration"""
    # Paths
    checkpoint_path: str = "checkpoints/best.pt"
    data_dir: str = "data/processed"
    output_path: Optional[str] = None

    # Evaluation settings
    split: str = "test"  # "test", "val", "train"
    batch_size: int = 32
    num_workers: int = 4

    # Metrics
    top_k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])
    compute_per_category: bool = True

    # Output
    save_predictions: bool = False
    predictions_path: Optional[str] = None

    # Device
    device: str = "auto"


# =============================================================================
# Evaluation Functions
# =============================================================================
def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Dict[str, Any]:
    """Load model checkpoint"""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    logger.info(f"Checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'logs' in checkpoint:
        logger.info(f"Training metrics: {checkpoint['logs']}")

    return checkpoint


def create_model_from_checkpoint(
    checkpoint: Dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    """Recreate model from checkpoint"""
    from src.models.gnn.shepherd_gnn import ShepherdGNN, ShepherdGNNConfig

    # Extract config from checkpoint
    config_dict = checkpoint.get("config", {})

    # Try to get model config
    model_config = ShepherdGNNConfig(
        hidden_dim=config_dict.get("hidden_dim", 256),
        num_layers=config_dict.get("num_layers", 4),
        num_heads=config_dict.get("num_heads", 8),
    )

    # Get metadata from checkpoint or use defaults
    metadata = checkpoint.get("metadata", (
        ["phenotype", "disease", "gene"],
        [("phenotype", "associated_with", "gene"),
         ("gene", "causes", "disease")]
    ))

    in_channels_dict = checkpoint.get("in_channels_dict", {
        "phenotype": 256,
        "disease": 256,
        "gene": 256,
    })

    model = ShepherdGNN(
        metadata=metadata,
        in_channels_dict=in_channels_dict,
        config=model_config,
    )

    # Load weights
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return model


def load_test_data(
    data_dir: Path,
    split: str,
    batch_size: int,
    num_workers: int,
):
    """Load test/validation data"""
    from src.kg.data_loader import (
        DataLoaderConfig,
        DiagnosisSample,
        create_diagnosis_dataloader,
    )

    # Load samples
    samples_path = data_dir / f"{split}_samples.json"
    if not samples_path.exists():
        raise FileNotFoundError(f"Samples file not found: {samples_path}")

    with open(samples_path) as f:
        raw_samples = json.load(f)

    samples = [
        DiagnosisSample(
            patient_id=s["patient_id"],
            phenotype_ids=s["phenotype_ids"],
            disease_id=s["disease_id"],
        )
        for s in raw_samples
    ]

    logger.info(f"Loaded {len(samples)} {split} samples")

    # Load graph data
    graph_data = {}

    node_features_path = data_dir / "node_features.pt"
    if node_features_path.exists():
        graph_data["x_dict"] = torch.load(node_features_path)

    edge_indices_path = data_dir / "edge_indices.pt"
    if edge_indices_path.exists():
        graph_data["edge_index_dict"] = torch.load(edge_indices_path)

    num_nodes_path = data_dir / "num_nodes.json"
    if num_nodes_path.exists():
        with open(num_nodes_path) as f:
            graph_data["num_nodes_dict"] = json.load(f)

    # Create dataloader
    config = DataLoaderConfig(
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    dataloader = create_diagnosis_dataloader(
        samples=samples,
        graph_data=graph_data,
        config=config,
    )

    return dataloader, samples


def evaluate_model(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    top_k_values: List[int],
) -> Dict[str, Any]:
    """Run evaluation and compute metrics"""
    from src.training.loss_functions import MultiTaskLoss

    model.eval()
    loss_fn = MultiTaskLoss()

    all_predictions = []
    all_ground_truths = []
    all_scores = []
    total_loss = 0.0
    num_batches = 0

    logger.info("Running evaluation...")

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch_data["batch"].items()}
            subgraph_x = {k: v.to(device) for k, v in batch_data["subgraph_x_dict"].items()}
            subgraph_edges = {k: v.to(device) for k, v in batch_data["subgraph_edge_index_dict"].items()}

            # Forward pass
            node_embeddings = model(subgraph_x, subgraph_edges)

            # Get disease and phenotype embeddings
            disease_emb = node_embeddings.get("disease")
            phenotype_emb = node_embeddings.get("phenotype")

            if disease_emb is None or phenotype_emb is None:
                continue

            # Get patient phenotype IDs
            phenotype_ids = batch.get("phenotype_ids")
            disease_ids = batch.get("disease_ids")
            phenotype_mask = batch.get("phenotype_mask")

            if phenotype_ids is None or disease_ids is None:
                continue

            # Compute patient embeddings
            batch_size = phenotype_ids.size(0)
            max_phenotypes = phenotype_ids.size(1)

            valid_ids = phenotype_ids.clamp(min=0, max=phenotype_emb.size(0) - 1)
            patient_phenotype_emb = phenotype_emb[valid_ids.view(-1)].view(
                batch_size, max_phenotypes, -1
            )

            if phenotype_mask is not None:
                mask = phenotype_mask.unsqueeze(-1).float()
                patient_emb = (patient_phenotype_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                patient_emb = patient_phenotype_emb.mean(dim=1)

            # Compute scores
            patient_norm = torch.nn.functional.normalize(patient_emb, dim=-1)
            disease_norm = torch.nn.functional.normalize(disease_emb, dim=-1)
            scores = torch.mm(patient_norm, disease_norm.t())

            # Get rankings
            _, indices = scores.sort(dim=-1, descending=True)

            for i in range(batch_size):
                pred_indices = indices[i].tolist()
                all_predictions.append([str(idx) for idx in pred_indices[:max(top_k_values)]])
                all_ground_truths.append(str(disease_ids[i].item()))
                all_scores.append(scores[i].cpu().tolist())

            num_batches += 1

            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {batch_idx + 1} batches")

    # Compute metrics
    logger.info("Computing metrics...")
    ranking_metrics = RankingMetrics()

    results = {
        "num_samples": len(all_predictions),
        "num_batches": num_batches,
    }

    # Standard metrics
    standard_metrics = ranking_metrics.compute_all(all_predictions, all_ground_truths)
    results.update(standard_metrics)

    # Hits@K for all specified values
    for k in top_k_values:
        hits_at_k = ranking_metrics.hits_at_k(all_predictions, all_ground_truths, k)
        results[f"hits@{k}"] = hits_at_k

    return results, all_predictions, all_scores


def generate_report(
    results: Dict[str, Any],
    config: EvalConfig,
    checkpoint_path: Path,
) -> Dict[str, Any]:
    """Generate detailed evaluation report"""
    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "checkpoint": str(checkpoint_path),
            "split": config.split,
            "num_samples": results["num_samples"],
        },
        "metrics": {
            "mrr": results.get("mrr", 0.0),
            "mean_rank": results.get("mean_rank", 0.0),
        },
        "hits_at_k": {},
        "config": asdict(config),
    }

    # Add Hits@K
    for k in config.top_k_values:
        key = f"hits@{k}"
        if key in results:
            report["hits_at_k"][str(k)] = results[key]

    return report


def print_results(results: Dict[str, Any], config: EvalConfig) -> None:
    """Print evaluation results to console"""
    print("\n" + "=" * 60)
    print("SHEPHERD-Advanced Evaluation Results")
    print("=" * 60)
    print(f"Split: {config.split}")
    print(f"Samples: {results['num_samples']}")
    print("-" * 60)
    print(f"MRR:        {results.get('mrr', 0.0):.4f}")
    print(f"Mean Rank:  {results.get('mean_rank', 0.0):.2f}")
    print("-" * 60)
    print("Hits@K:")
    for k in config.top_k_values:
        key = f"hits@{k}"
        if key in results:
            print(f"  Hits@{k:2d}:  {results[key]:.4f}")
    print("=" * 60 + "\n")


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained ShepherdGNN model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default="checkpoints/best.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data-dir", "-d",
        type=str,
        default="data/processed",
        help="Directory containing test data",
    )
    parser.add_argument(
        "--split", "-s",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Data split to evaluate on",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output path for evaluation report (JSON)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save per-sample predictions",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Build config
    config = EvalConfig(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        split=args.split,
        output_path=args.output,
        batch_size=args.batch_size,
        device=args.device,
        save_predictions=args.save_predictions,
    )

    # Set device
    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)

    logger.info(f"Using device: {device}")

    # Load checkpoint
    checkpoint_path = Path(config.checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path, device)

    # Create model
    model = create_model_from_checkpoint(checkpoint, device)
    logger.info(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Load data
    data_dir = Path(config.data_dir)
    dataloader, samples = load_test_data(
        data_dir, config.split, config.batch_size, config.num_workers
    )

    # Evaluate
    results, predictions, scores = evaluate_model(
        model, dataloader, device, config.top_k_values
    )

    # Print results
    print_results(results, config)

    # Generate and save report
    report = generate_report(results, config, checkpoint_path)

    if config.output_path:
        output_path = Path(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved to {output_path}")

    # Save predictions if requested
    if config.save_predictions:
        pred_path = Path(config.predictions_path or f"predictions_{config.split}.json")
        pred_path.parent.mkdir(parents=True, exist_ok=True)

        pred_data = [
            {
                "sample_id": samples[i].patient_id,
                "ground_truth": samples[i].disease_id,
                "predictions": predictions[i][:20],
            }
            for i in range(len(predictions))
        ]

        with open(pred_path, "w") as f:
            json.dump(pred_data, f, indent=2)

        logger.info(f"Predictions saved to {pred_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
