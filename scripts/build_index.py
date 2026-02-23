"""
SHEPHERD-Advanced Vector Index Builder
======================================
Functionality:
  - Build vector index from GNN node embeddings
  - Supports two modes:
    1. From checkpoint: load model + graph data, run forward pass, extract embeddings
    2. From pre-exported embeddings: load .npz file directly
  - Auto-select optimal backend based on platform
  - Support configuration from deployment.yaml

Path:
  - Relative: scripts/build_index.py
  - Absolute: SHEPHERD-Advanced/scripts/build_index.py

Input:
  - --checkpoint: Path to GNN model checkpoint (.pt)
  - --data-dir: Path to processed graph data directory
  - --embeddings: Path to pre-exported embeddings (.npz) (alternative to checkpoint)
  - --config: Path to deployment config (YAML)
  - --output: Path to save index
  - --node-types: Which node types to index (default: disease)

Output:
  - Vector index saved to specified path (.voyager / .cuvs + .ids.json)
  - Optionally export raw embeddings to .npz for reuse

Backend Selection:
  - Linux (x86/ARM): cuVS (GPU) -> Voyager (CPU fallback)
  - Windows: Voyager (CPU only)

Usage:
  # From checkpoint (most common):
  python scripts/build_index.py \\
      --checkpoint data/demo/model_checkpoint.pt \\
      --data-dir data/demo \\
      --output data/demo/vector_index

  # From pre-exported embeddings:
  python scripts/build_index.py \\
      --embeddings data/demo/embeddings.npz \\
      --output data/demo/vector_index

  # Export embeddings only (no index build):
  python scripts/build_index.py \\
      --checkpoint data/demo/model_checkpoint.pt \\
      --data-dir data/demo \\
      --export-embeddings data/demo/embeddings.npz
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.retrieval import create_index, resolve_backend, list_available_backends

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_embeddings_from_checkpoint(
    checkpoint_path: str,
    data_dir: str,
    device: Optional[str] = None,
    node_types: Optional[list] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, int]], int]:
    """
    Load GNN model, run forward pass, and extract node embeddings.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        data_dir: Path to directory containing graph data files
        device: Compute device ("cpu", "cuda", or None for auto)
        node_types: Which node types to extract (None = all)

    Returns:
        Tuple of:
        - embeddings: {node_type: ndarray of shape (N, hidden_dim)}
        - id_mappings: {node_type: {node_id_str: int_idx}}
        - hidden_dim: embedding dimension
    """
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Extract config and state dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        ckpt_config = checkpoint.get("config", {})
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        ckpt_config = checkpoint.get("config", {})
    else:
        state_dict = checkpoint
        ckpt_config = {}

    hidden_dim = ckpt_config.get("hidden_dim", 256)

    # Load graph data
    data_path = Path(data_dir)
    graph_data: Dict[str, Any] = {"x_dict": {}, "edge_index_dict": {}, "num_nodes_dict": {}}

    node_features_path = data_path / "node_features.pt"
    if node_features_path.exists():
        graph_data["x_dict"] = torch.load(node_features_path, map_location="cpu", weights_only=True)
        logger.info(f"Loaded node features: {list(graph_data['x_dict'].keys())}")

    edge_indices_path = data_path / "edge_indices.pt"
    if edge_indices_path.exists():
        graph_data["edge_index_dict"] = torch.load(edge_indices_path, map_location="cpu", weights_only=True)
        logger.info(f"Loaded edge indices: {len(graph_data['edge_index_dict'])} edge types")

    num_nodes_path = data_path / "num_nodes.json"
    if num_nodes_path.exists():
        with open(num_nodes_path, "r") as f:
            graph_data["num_nodes_dict"] = json.load(f)

    if not graph_data["x_dict"]:
        raise FileNotFoundError(f"No node features found in {data_dir}")

    # Load node ID mapping
    node_mapping_path = data_path / "node_id_mapping.json"
    if node_mapping_path.exists():
        with open(node_mapping_path, "r") as f:
            id_mappings = json.load(f)
        logger.info(f"Loaded node ID mapping: {list(id_mappings.keys())}")
    else:
        # Try loading from KG
        id_mappings = _load_id_mapping_from_kg(data_path)

    # Build model
    metadata = _infer_metadata(graph_data)
    in_channels_dict = {k: v.shape[1] for k, v in graph_data["x_dict"].items()}

    # Detect model features from state dict
    state_keys = set(state_dict.keys())
    has_pos_encoder = any(k.startswith("pos_encoder.") for k in state_keys)
    has_ortholog_gate = any(k.startswith("ortholog_gate.") for k in state_keys)

    from src.models.gnn.shepherd_gnn import ShepherdGNN, ShepherdGNNConfig

    model_config = ShepherdGNNConfig(
        hidden_dim=hidden_dim,
        num_layers=ckpt_config.get("num_layers", 4),
        num_heads=ckpt_config.get("num_heads", 8),
        conv_type=ckpt_config.get("conv_type", "gat"),
        dropout=0.0,  # No dropout at inference
        use_positional_encoding=ckpt_config.get("use_positional_encoding", has_pos_encoder),
        use_ortholog_gate=ckpt_config.get("use_ortholog_gate", has_ortholog_gate),
    )

    model = ShepherdGNN(
        metadata=metadata,
        in_channels_dict=in_channels_dict,
        config=model_config,
    )
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)

    # Forward pass
    logger.info(f"Running GNN forward pass on device={device}...")
    x_dict = {k: v.to(device) for k, v in graph_data["x_dict"].items()}
    edge_index_dict = {k: v.to(device) for k, v in graph_data["edge_index_dict"].items()}

    with torch.no_grad():
        embeddings_torch = model(x_dict, edge_index_dict)

    # Convert to numpy
    embeddings = {}
    for ntype, emb_tensor in embeddings_torch.items():
        if node_types is None or ntype in node_types:
            embeddings[ntype] = emb_tensor.cpu().numpy().astype(np.float32)
            logger.info(f"  {ntype}: {embeddings[ntype].shape}")

    return embeddings, id_mappings, hidden_dim


def _infer_metadata(graph_data: Dict[str, Any]):
    """Infer PyG metadata (node_types, edge_types) from graph data."""
    node_types = sorted(graph_data["x_dict"].keys())
    edge_types = []
    for key in graph_data["edge_index_dict"].keys():
        if isinstance(key, tuple) and len(key) == 3:
            edge_types.append(key)
    return node_types, edge_types


def _load_id_mapping_from_kg(data_path: Path) -> Dict[str, Dict[str, int]]:
    """Try to load node ID mappings from the KG JSON file."""
    kg_path = data_path / "kg.json"
    if not kg_path.exists():
        # Look for it one level up
        kg_path = data_path.parent / "kg.json"
    if not kg_path.exists():
        logger.warning("No node_id_mapping.json or kg.json found; ID mapping will be index-based")
        return {}

    logger.info(f"Building ID mapping from KG: {kg_path}")
    from src.kg import KnowledgeGraph
    kg = KnowledgeGraph.load_json(str(kg_path))
    return kg.get_node_id_mapping()


def load_embeddings_from_file(
    embeddings_path: str,
    node_types: Optional[list] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, int]], int]:
    """
    Load pre-exported embeddings from .npz file.

    The .npz file should contain:
    - Arrays named by node type (e.g., "disease", "phenotype")
    - A JSON string under key "id_mappings" with the node ID mappings

    Returns:
        Same format as extract_embeddings_from_checkpoint.
    """
    logger.info(f"Loading embeddings from: {embeddings_path}")
    data = np.load(embeddings_path, allow_pickle=True)

    # Load ID mappings
    if "id_mappings" in data:
        id_mappings = json.loads(str(data["id_mappings"]))
    else:
        id_mappings = {}

    # Load embeddings
    embeddings = {}
    hidden_dim = 0
    for key in data.files:
        if key == "id_mappings":
            continue
        if node_types is None or key in node_types:
            arr = data[key].astype(np.float32)
            embeddings[key] = arr
            hidden_dim = arr.shape[1]
            logger.info(f"  {key}: {arr.shape}")

    return embeddings, id_mappings, hidden_dim


def export_embeddings(
    embeddings: Dict[str, np.ndarray],
    id_mappings: Dict[str, Dict[str, int]],
    output_path: str,
) -> None:
    """Export embeddings and ID mappings to .npz file."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    save_dict = dict(embeddings)
    save_dict["id_mappings"] = np.array(json.dumps(id_mappings))

    np.savez(str(output), **save_dict)
    logger.info(f"Embeddings exported to: {output}")


def build_and_save_index(
    embeddings: Dict[str, np.ndarray],
    id_mappings: Dict[str, Dict[str, int]],
    hidden_dim: int,
    output_path: str,
    node_types: list,
    backend: str = "auto",
    config_path: Optional[str] = None,
) -> None:
    """
    Build vector index from embeddings and save to disk.

    For each indexed node type, creates:
    - {output_path}_{node_type}.voyager (or .cuvs)
    - {output_path}_{node_type}.ids.json

    Args:
        embeddings: {node_type: ndarray(N, hidden_dim)}
        id_mappings: {node_type: {node_id_str: int_idx}}
        hidden_dim: Embedding dimension
        output_path: Base output path (without extension)
        node_types: Which node types to index
        backend: Backend name or "auto"
        config_path: Optional deployment.yaml path for backend config
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Load backend config if available
    backend_config = {}
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            deploy_config = yaml.safe_load(f)
        indexing_cfg = deploy_config.get("indexing", {})
        resolved = resolve_backend(backend)
        backend_config = indexing_cfg.get(resolved, {})

    for ntype in node_types:
        if ntype not in embeddings:
            logger.warning(f"Node type '{ntype}' not found in embeddings, skipping")
            continue

        emb_array = embeddings[ntype]
        mapping = id_mappings.get(ntype, {})

        if not mapping:
            # Generate index-based mapping as fallback
            logger.warning(f"No ID mapping for '{ntype}', using index-based IDs")
            mapping = {str(i): i for i in range(emb_array.shape[0])}

        # Invert mapping: {node_id_str: int_idx} -> {int_idx: node_id_str}
        # Then build the dict format that VectorIndexBase.build_index expects:
        # {entity_id: embedding_vector}
        idx_to_id = {idx: nid for nid, idx in mapping.items()}

        entity_embeddings = {}
        for idx in range(emb_array.shape[0]):
            entity_id = idx_to_id.get(idx, str(idx))
            entity_embeddings[entity_id] = emb_array[idx]

        # Create and build index
        logger.info(f"Building index for '{ntype}': {len(entity_embeddings)} entities, dim={hidden_dim}")
        index = create_index(backend=backend, dim=hidden_dim, **backend_config)
        index.build_index(entity_embeddings)

        # Save
        index_path = output.parent / f"{output.name}_{ntype}"
        index.save(index_path)
        logger.info(f"Saved {ntype} index to: {index_path}")

    logger.info("Index build complete")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build vector index for SHEPHERD knowledge graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From checkpoint:
  python scripts/build_index.py --checkpoint data/demo/model_checkpoint.pt --data-dir data/demo --output data/demo/vector_index

  # From pre-exported embeddings:
  python scripts/build_index.py --embeddings data/demo/embeddings.npz --output data/demo/vector_index

  # Export embeddings only:
  python scripts/build_index.py --checkpoint data/demo/model_checkpoint.pt --data-dir data/demo --export-embeddings data/demo/embeddings.npz
        """,
    )

    # Input sources (mutually exclusive)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to GNN model checkpoint (.pt)",
    )
    source.add_argument(
        "--embeddings",
        type=Path,
        help="Path to pre-exported embeddings file (.npz)",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Path to processed graph data directory (required with --checkpoint)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "deployment.yaml",
        help="Path to deployment config (default: configs/deployment.yaml)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Base path to save the built index (e.g., data/demo/vector_index)",
    )
    parser.add_argument(
        "--export-embeddings",
        type=Path,
        help="Export embeddings to .npz file (can be used with --output for both)",
    )
    parser.add_argument(
        "--node-types",
        nargs="+",
        default=["disease"],
        help="Node types to index (default: disease)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "cuvs", "voyager"],
        help="Force specific backend (default: auto)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device for GNN forward pass (default: auto)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration and exit without building",
    )
    args = parser.parse_args()

    # Validation
    if args.checkpoint and not args.data_dir:
        parser.error("--data-dir is required when using --checkpoint")

    if not args.checkpoint and not args.embeddings:
        parser.error("Either --checkpoint or --embeddings is required")

    if not args.output and not args.export_embeddings:
        parser.error("At least one of --output or --export-embeddings is required")

    # Print config
    available = list_available_backends()
    resolved = resolve_backend(args.backend)
    logger.info(f"Config: {args.config}")
    logger.info(f"Available backends: {available}")
    logger.info(f"Selected backend: {resolved}")
    logger.info(f"Node types to index: {args.node_types}")

    if args.dry_run:
        logger.info("[dry-run] Exiting without building index")
        return 0

    # Step 1: Get embeddings
    if args.checkpoint:
        embeddings, id_mappings, hidden_dim = extract_embeddings_from_checkpoint(
            checkpoint_path=str(args.checkpoint),
            data_dir=str(args.data_dir),
            device=args.device,
            node_types=args.node_types,
        )
    else:
        embeddings, id_mappings, hidden_dim = load_embeddings_from_file(
            embeddings_path=str(args.embeddings),
            node_types=args.node_types,
        )

    if not embeddings:
        logger.error("No embeddings extracted")
        return 1

    logger.info(f"Embeddings ready: {list(embeddings.keys())}, dim={hidden_dim}")

    # Step 2: Export embeddings if requested
    if args.export_embeddings:
        export_embeddings(embeddings, id_mappings, str(args.export_embeddings))

    # Step 3: Build and save index
    if args.output:
        config_path = str(args.config) if args.config.exists() else None
        build_and_save_index(
            embeddings=embeddings,
            id_mappings=id_mappings,
            hidden_dim=hidden_dim,
            output_path=str(args.output),
            node_types=args.node_types,
            backend=args.backend,
            config_path=config_path,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
