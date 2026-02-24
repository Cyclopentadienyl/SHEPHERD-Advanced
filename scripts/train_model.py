#!/usr/bin/env python3
"""
SHEPHERD-Advanced Model Training Script
========================================
Main entry point for training the ShepherdGNN model.

Script: scripts/train_model.py
Absolute Path: /home/user/SHEPHERD-Advanced/scripts/train_model.py

Purpose:
    Command-line interface for training ShepherdGNN on rare disease diagnosis.
    Handles configuration loading, data preparation, model initialization,
    and training loop execution.

Usage:
    python scripts/train_model.py --config configs/train_config.yaml
    python scripts/train_model.py --data-dir data/processed --epochs 100
    python scripts/train_model.py --resume checkpoints/last.pt

Dependencies:
    - argparse: CLI argument parsing
    - yaml: Configuration file loading
    - torch: Model training
    - src.training.trainer: Trainer, TrainerConfig
    - src.kg.data_loader: DiagnosisDataLoader, DataLoaderConfig
    - src.kg.graph: KnowledgeGraph
    - src.models.gnn.shepherd_gnn: ShepherdGNN, ShepherdGNNConfig

Input:
    - Configuration file (YAML) or CLI arguments
    - Processed knowledge graph data
    - Training samples (patient phenotypes -> disease)

Output:
    - Trained model checkpoints
    - Training logs and metrics
    - Final evaluation report

Called by:
    - User via command line
    - CI/CD pipelines
    - Automated training scripts

Version: 1.0.0
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.trainer import Trainer, TrainerConfig
from src.training.loss_functions import LossConfig
from src.kg.data_loader import (
    DataLoaderConfig,
    DiagnosisDataLoader,
    DiagnosisDataset,
    DiagnosisSample,
    create_diagnosis_dataloader,
)
from src.models.gnn.shepherd_gnn import ShepherdGNN, ShepherdGNNConfig, create_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class TrainConfig:
    """Complete training configuration"""
    # Paths
    data_dir: str = "data/processed"
    output_dir: str = "outputs"
    checkpoint_dir: str = "models/checkpoints"
    log_dir: str = "logs"
    config_file: Optional[str] = None

    # Model configuration
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    conv_type: str = "gat"
    dropout: float = 0.1
    use_ortholog_gate: bool = True

    # Training configuration
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Learning rate scheduling
    scheduler_type: str = "cosine"
    warmup_steps: int = 500
    min_lr_ratio: float = 0.01

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "float16"

    # Data loading
    num_workers: int = 4
    num_neighbors: List[int] = field(default_factory=lambda: [15, 10, 5])
    max_subgraph_nodes: int = 5000
    num_negative_samples: int = 5

    # Validation
    eval_every_n_epochs: int = 1
    early_stopping_patience: int = 10
    save_top_k: int = 3

    # Loss weights
    diagnosis_weight: float = 1.0
    link_prediction_weight: float = 0.5
    contrastive_weight: float = 0.3
    ortholog_weight: float = 0.2

    # Device
    device: str = "auto"  # "auto", "cuda", "cpu"

    # Reproducibility
    seed: int = 42

    # Resume training
    resume_from: Optional[str] = None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Train ShepherdGNN for rare disease diagnosis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Configuration file
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to YAML configuration file",
    )

    # Paths
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing processed data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for training outputs",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory for model checkpoints",
    )

    # Model
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Hidden dimension size",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of GNN layers",
    )
    parser.add_argument(
        "--conv-type",
        type=str,
        choices=["gat", "hgt", "sage"],
        default=None,
        help="GNN convolution type",
    )

    # Training
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr", "--learning-rate",
        type=float,
        default=None,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=None,
        help="Gradient accumulation steps",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cuda", "cpu"],
        help="Device to use for training",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision",
    )

    # Resume
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume from",
    )

    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (verbose logging)",
    )

    return parser.parse_args()


def load_config(args: argparse.Namespace) -> TrainConfig:
    """Load configuration from file and command-line arguments"""
    config = TrainConfig()

    # Load from YAML if provided
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            yaml_config = yaml.safe_load(f)
        for key, value in yaml_config.items():
            if hasattr(config, key):
                setattr(config, key, value)

    # Override with command-line arguments (only if explicitly provided)
    if args.data_dir is not None:
        config.data_dir = args.data_dir
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir
    if args.hidden_dim is not None:
        config.hidden_dim = args.hidden_dim
    if args.num_layers is not None:
        config.num_layers = args.num_layers
    if args.conv_type is not None:
        config.conv_type = args.conv_type
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.grad_accum is not None:
        config.gradient_accumulation_steps = args.grad_accum
    if args.device is not None:
        config.device = args.device
    if args.no_amp:
        config.use_amp = False
    if args.resume:
        config.resume_from = args.resume
    if args.seed is not None:
        config.seed = args.seed

    return config


# =============================================================================
# Data Loading
# =============================================================================
def load_graph_data(data_dir: Path) -> Dict[str, Any]:
    """
    Load knowledge graph data from disk

    Expected files:
    - node_features.pt: {node_type: tensor}
    - edge_indices.pt: {edge_type: tensor}
    - num_nodes.json: {node_type: count}
    """
    logger.info(f"Loading graph data from {data_dir}")

    graph_data = {
        "x_dict": {},
        "edge_index_dict": {},
        "num_nodes_dict": {},
    }

    # Load node features
    node_features_path = data_dir / "node_features.pt"
    if node_features_path.exists():
        graph_data["x_dict"] = torch.load(node_features_path, weights_only=True)
        logger.info(f"Loaded node features: {list(graph_data['x_dict'].keys())}")

    # Load edge indices
    edge_indices_path = data_dir / "edge_indices.pt"
    if edge_indices_path.exists():
        graph_data["edge_index_dict"] = torch.load(edge_indices_path, weights_only=True)
        logger.info(f"Loaded edge indices: {len(graph_data['edge_index_dict'])} edge types")

    # Load node counts
    num_nodes_path = data_dir / "num_nodes.json"
    if num_nodes_path.exists():
        with open(num_nodes_path) as f:
            graph_data["num_nodes_dict"] = json.load(f)
    else:
        # Infer from features
        for node_type, features in graph_data["x_dict"].items():
            graph_data["num_nodes_dict"][node_type] = features.size(0)

    logger.info(f"Node counts: {graph_data['num_nodes_dict']}")

    return graph_data


def load_samples(data_dir: Path, split: str = "train") -> List[DiagnosisSample]:
    """
    Load training/validation samples

    Expected file: {split}_samples.json
    Format: [{"patient_id": str, "phenotype_ids": [int], "disease_id": int}, ...]
    """
    samples_path = data_dir / f"{split}_samples.json"

    if not samples_path.exists():
        logger.warning(f"Samples file not found: {samples_path}")
        return []

    with open(samples_path) as f:
        raw_samples = json.load(f)

    samples = [
        DiagnosisSample(
            patient_id=s["patient_id"],
            phenotype_ids=s["phenotype_ids"],
            disease_id=s["disease_id"],
            candidate_disease_ids=s.get("candidate_disease_ids"),
            gene_ids=s.get("gene_ids"),
        )
        for s in raw_samples
    ]

    logger.info(f"Loaded {len(samples)} {split} samples")
    return samples


def create_dataloaders(
    config: TrainConfig,
    graph_data: Dict[str, Any],
) -> Tuple[DiagnosisDataLoader, Optional[DiagnosisDataLoader]]:
    """Create training and validation data loaders"""
    data_dir = Path(config.data_dir)

    # Data loader config
    dl_config = DataLoaderConfig(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        num_neighbors=config.num_neighbors,
        max_subgraph_nodes=config.max_subgraph_nodes,
        num_negative_samples=config.num_negative_samples,
    )

    # Training samples
    train_samples = load_samples(data_dir, "train")
    if not train_samples:
        raise ValueError("No training samples found")

    train_loader = create_diagnosis_dataloader(
        samples=train_samples,
        graph_data=graph_data,
        config=dl_config,
    )

    # Validation samples
    val_samples = load_samples(data_dir, "val")
    if val_samples:
        val_config = DataLoaderConfig(
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
            num_neighbors=config.num_neighbors,
            max_subgraph_nodes=config.max_subgraph_nodes,
        )
        val_loader = create_diagnosis_dataloader(
            samples=val_samples,
            graph_data=graph_data,
            config=val_config,
        )
    else:
        val_loader = None
        logger.warning("No validation samples found, skipping validation")

    return train_loader, val_loader


# =============================================================================
# Model Creation
# =============================================================================
def create_model_from_config(
    config: TrainConfig,
    graph_data: Dict[str, Any],
) -> ShepherdGNN:
    """Create ShepherdGNN model from configuration"""
    # Extract metadata
    node_types = list(graph_data["num_nodes_dict"].keys())
    edge_types = list(graph_data["edge_index_dict"].keys())
    metadata = (node_types, edge_types)

    # Infer input channels from features
    in_channels_dict = {}
    for node_type, features in graph_data["x_dict"].items():
        if features.dim() >= 2:
            in_channels_dict[node_type] = features.size(-1)
        else:
            in_channels_dict[node_type] = config.hidden_dim

    # Model config
    model_config = ShepherdGNNConfig(
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        conv_type=config.conv_type,
        dropout=config.dropout,
        use_ortholog_gate=config.use_ortholog_gate,
    )

    model = ShepherdGNN(
        metadata=metadata,
        in_channels_dict=in_channels_dict,
        config=model_config,
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,} total, {num_trainable:,} trainable")

    return model


# =============================================================================
# Main Training Function
# =============================================================================
def train(config: TrainConfig) -> Dict[str, float]:
    """
    Main training function

    Returns:
        Final evaluation metrics
    """
    logger.info("=" * 60)
    logger.info("SHEPHERD-Advanced Training")
    logger.info("=" * 60)
    logger.info(f"Configuration: {asdict(config)}")

    # Set device
    if config.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.device

    logger.info(f"Using device: {device}")

    if device == "cuda":
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create output directories
    output_dir = Path(config.output_dir)
    checkpoint_dir = Path(config.checkpoint_dir)
    log_dir = Path(config.log_dir)

    for d in [output_dir, checkpoint_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(asdict(config), f)
    logger.info(f"Configuration saved to {config_path}")

    # Load data
    data_dir = Path(config.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.info("Please run data preprocessing first:")
        logger.info("  python scripts/preprocess_data.py")
        return {}

    graph_data = load_graph_data(data_dir)
    train_loader, val_loader = create_dataloaders(config, graph_data)

    # Create model
    model = create_model_from_config(config, graph_data)

    # Training configuration
    trainer_config = TrainerConfig(
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_grad_norm=config.max_grad_norm,
        scheduler_type=config.scheduler_type,
        warmup_steps=config.warmup_steps,
        min_lr_ratio=config.min_lr_ratio,
        use_amp=config.use_amp,
        amp_dtype=config.amp_dtype,
        eval_every_n_epochs=config.eval_every_n_epochs,
        early_stopping_patience=config.early_stopping_patience,
        checkpoint_dir=str(checkpoint_dir),
        save_top_k=config.save_top_k,
        log_dir=str(log_dir),
        device=device,
        seed=config.seed,
        loss_config=LossConfig(
            diagnosis_weight=config.diagnosis_weight,
            link_prediction_weight=config.link_prediction_weight,
            contrastive_weight=config.contrastive_weight,
            ortholog_weight=config.ortholog_weight,
        ),
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=trainer_config,
    )

    # Resume if specified
    if config.resume_from:
        resume_path = Path(config.resume_from)
        if resume_path.exists():
            logger.info(f"Resuming from checkpoint: {resume_path}")
            trainer.load_checkpoint(resume_path)
        else:
            logger.warning(f"Checkpoint not found: {resume_path}")

    # Train
    logger.info("Starting training...")
    final_metrics = trainer.train()

    # Log final results
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)

    for name, value in final_metrics.items():
        logger.info(f"  {name}: {value:.4f}")

    # Save final metrics
    metrics_path = output_dir / "final_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    logger.info(f"Final metrics saved to {metrics_path}")

    return final_metrics


# =============================================================================
# Synthetic Data Generation (for testing)
# =============================================================================
def generate_synthetic_data(
    data_dir: Path,
    num_phenotypes: int = 5000,
    num_diseases: int = 1000,
    num_genes: int = 10000,
    num_train_samples: int = 1000,
    num_val_samples: int = 200,
    hidden_dim: int = 256,
) -> None:
    """
    Generate synthetic data for testing the training pipeline

    This creates placeholder data with correct structure for testing.
    """
    logger.info("Generating synthetic data for testing...")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Node features (random embeddings)
    x_dict = {
        "phenotype": torch.randn(num_phenotypes, hidden_dim),
        "disease": torch.randn(num_diseases, hidden_dim),
        "gene": torch.randn(num_genes, hidden_dim),
    }
    torch.save(x_dict, data_dir / "node_features.pt")

    # Edge indices (random connections)
    def random_edges(n_src, n_dst, n_edges):
        src = torch.randint(0, n_src, (n_edges,))
        dst = torch.randint(0, n_dst, (n_edges,))
        return torch.stack([src, dst])

    # Forward edges
    phenotype_gene_edges = random_edges(num_phenotypes, num_genes, 20000)
    gene_disease_edges = random_edges(num_genes, num_diseases, 15000)
    phenotype_disease_edges = random_edges(num_phenotypes, num_diseases, 10000)

    edge_index_dict = {
        # Forward edges
        ("phenotype", "associated_with", "gene"): phenotype_gene_edges,
        ("gene", "causes", "disease"): gene_disease_edges,
        ("phenotype", "observed_in", "disease"): phenotype_disease_edges,
        # Reverse edges (for bidirectional message passing)
        ("gene", "rev_associated_with", "phenotype"): phenotype_gene_edges.flip(0),
        ("disease", "rev_causes", "gene"): gene_disease_edges.flip(0),
        ("disease", "rev_observed_in", "phenotype"): phenotype_disease_edges.flip(0),
    }
    torch.save(edge_index_dict, data_dir / "edge_indices.pt")

    # Node counts
    num_nodes = {
        "phenotype": num_phenotypes,
        "disease": num_diseases,
        "gene": num_genes,
    }
    with open(data_dir / "num_nodes.json", "w") as f:
        json.dump(num_nodes, f)

    # Training samples
    import random

    def generate_samples(n_samples):
        samples = []
        for i in range(n_samples):
            n_phenotypes = random.randint(3, 10)
            samples.append({
                "patient_id": f"patient_{i:05d}",
                "phenotype_ids": random.sample(range(num_phenotypes), n_phenotypes),
                "disease_id": random.randint(0, num_diseases - 1),
            })
        return samples

    train_samples = generate_samples(num_train_samples)
    with open(data_dir / "train_samples.json", "w") as f:
        json.dump(train_samples, f)

    val_samples = generate_samples(num_val_samples)
    with open(data_dir / "val_samples.json", "w") as f:
        json.dump(val_samples, f)

    logger.info(f"Synthetic data generated in {data_dir}")
    logger.info(f"  Nodes: {num_nodes}")
    logger.info(f"  Edges: {sum(e.size(1) for e in edge_index_dict.values())}")
    logger.info(f"  Train samples: {num_train_samples}")
    logger.info(f"  Val samples: {num_val_samples}")


# =============================================================================
# Entry Point
# =============================================================================
def main():
    """Main entry point"""
    args = parse_args()

    # Set debug logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    config = load_config(args)

    # Check if data exists, generate synthetic if not
    data_dir = Path(config.data_dir)
    if not data_dir.exists() or not (data_dir / "train_samples.json").exists():
        logger.warning(f"Data not found in {data_dir}")
        logger.info("Generating synthetic data for testing...")
        generate_synthetic_data(data_dir)

    # Run training
    try:
        metrics = train(config)
        return 0 if metrics else 1
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
