"""
SHEPHERD-Advanced Vector Index Builder
======================================
Functionality:
  - Build a vector index from pre-exported node embeddings (an .npz file)
  - Auto-select optimal backend based on platform
  - Support configuration from deployment.yaml

**It builds no model** (K7, TM10): it used to rebuild a ShepherdGNN from a
checkpoint (`--checkpoint`), guessing its structure from key names, a second
builder beside the one model builder. That input is gone and nothing replaces
it here: which vectors a task model would index is not decided, so the vectors
come from a file whoever decides that writes.

Path:
  - Relative: scripts/build_index.py
  - Absolute: SHEPHERD-Advanced/scripts/build_index.py

Input:
  - --embeddings: Path to the embeddings (.npz): one 2-D array per node type,
    all of one width, and optionally ``id_mappings`` (a JSON string of
    ``{node_type: {node_id: row}}``). A file that cannot be read this way, an
    array that is not 2-D, and arrays of different widths are refused.
  - --config: Path to deployment config (YAML)
  - --output: Path to save index
  - --node-types: Which node types to index (default: disease)

Output:
  - Vector index saved to specified path (.voyager / .cuvs + .ids.json)
  - Optionally the embeddings read, for the node types chosen, to another .npz

Backend Selection:
  - Linux (x86/ARM): cuVS (GPU) -> Voyager (CPU fallback)
  - Windows: Voyager (CPU only)

Usage:
  python scripts/build_index.py \\
      --embeddings data/demo/embeddings.npz \\
      --output data/demo/vector_index
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.retrieval import create_index, resolve_backend, list_available_backends

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class EmbeddingsRefused(ValueError):
    """The embeddings file cannot be indexed; the message says why."""


def load_embeddings_from_file(
    embeddings_path: str,
    node_types: Optional[list] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, int]], int]:
    """
    Load pre-exported embeddings from .npz file.

    The .npz file should contain:
    - Arrays named by node type (e.g., "disease", "phenotype"), each 2-D
      (one row per node) and all of one width: the index's dimension
    - A JSON string under key "id_mappings" with the node ID mappings

    It is read without pickle (an object array is refused, never unpickled).

    Returns:
        ``(embeddings by node type, id mappings, dimension)`` for the node
        types chosen.

    Raises:
        EmbeddingsRefused: the file cannot be read as such an .npz, an array
            chosen is not 2-D or is empty, or the arrays chosen differ in width.
    """
    logger.info(f"Loading embeddings from: {embeddings_path}")
    unreadable = f"{embeddings_path} cannot be read as an embeddings file (.npz of node-type arrays)"
    try:
        loaded = np.load(embeddings_path, allow_pickle=False)
        if not isinstance(loaded, np.lib.npyio.NpzFile):
            raise EmbeddingsRefused(f"{unreadable}: it holds one array, not one per node type")
        with loaded as data:
            arrays = {key: data[key] for key in data.files}
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        if isinstance(exc, EmbeddingsRefused):
            raise
        raise EmbeddingsRefused(f"{unreadable}: {type(exc).__name__}: {exc}") from exc

    # Load ID mappings
    id_mappings: Dict[str, Dict[str, int]] = {}
    if "id_mappings" in arrays:
        try:
            id_mappings = json.loads(str(arrays.pop("id_mappings")))
        except ValueError as exc:
            raise EmbeddingsRefused(
                f"{embeddings_path}: id_mappings is not a JSON string ({exc})") from exc

    # Load embeddings
    embeddings = {}
    for key, arr in arrays.items():
        if node_types is not None and key not in node_types:
            continue
        if arr.ndim != 2 or 0 in arr.shape:
            raise EmbeddingsRefused(
                f"{embeddings_path}: '{key}' has shape {arr.shape}; an embeddings array is "
                "2-D, one non-empty row per node")
        try:
            embeddings[key] = arr.astype(np.float32)
        except (TypeError, ValueError) as exc:
            raise EmbeddingsRefused(
                f"{embeddings_path}: '{key}' is not numeric ({arr.dtype})") from exc
        logger.info(f"  {key}: {arr.shape}")

    widths = {key: arr.shape[1] for key, arr in embeddings.items()}
    if len(set(widths.values())) > 1:
        raise EmbeddingsRefused(
            f"{embeddings_path}: the arrays differ in width ({widths}); one index "
            "dimension needs one width")
    return embeddings, id_mappings, next(iter(widths.values()), 0)


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
        description="Build a vector index from pre-exported node embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python scripts/build_index.py --embeddings data/demo/embeddings.npz --output data/demo/vector_index

No model is read: a checkpoint is not an input (--checkpoint was removed in K7).
        """,
    )

    parser.add_argument(
        "--embeddings",
        type=Path,
        required=True,
        help="Path to the embeddings file (.npz): one 2-D array per node type, one width",
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
        help="Also write the embeddings read, for the node types chosen, to this .npz "
             "(can be used with --output for both)",
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
        "--dry-run",
        action="store_true",
        help="Print configuration and exit without building",
    )
    args = parser.parse_args()

    # Validation
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
    try:
        embeddings, id_mappings, hidden_dim = load_embeddings_from_file(
            embeddings_path=str(args.embeddings),
            node_types=args.node_types,
        )
    except EmbeddingsRefused as exc:
        logger.error(f"Refused: {exc}. Nothing was written.")
        return 1

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
