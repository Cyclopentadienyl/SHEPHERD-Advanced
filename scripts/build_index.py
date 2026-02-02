"""
SHEPHERD-Advanced Vector Index Builder
======================================
Functionality:
  - Build vector index from entity embeddings
  - Auto-select optimal backend based on platform
  - Support configuration from deployment.yaml

Path:
  - Relative: scripts/build_index.py
  - Absolute: SHEPHERD-Advanced/scripts/build_index.py

Input:
  - --config: Path to deployment config (YAML)
  - --embeddings: Path to embeddings file (optional)
  - --output: Path to save index (optional)

Output:
  - Vector index saved to specified path
  - Console log with backend selection info

Backend Selection:
  - Linux (x86/ARM): cuVS (GPU) -> Voyager (CPU fallback)
  - Windows: Voyager (CPU only)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval import create_index_from_config, resolve_backend, list_available_backends

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build vector index for SHEPHERD knowledge graph"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "deployment.yaml",
        help="Path to deployment config (default: configs/deployment.yaml)",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        help="Path to embeddings file (.npy or .npz)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save the built index",
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

    # Load config
    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        return 1

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Resolve backend
    backend = resolve_backend(args.backend)
    available = list_available_backends()

    logger.info(f"Config: {args.config}")
    logger.info(f"Available backends: {available}")
    logger.info(f"Selected backend: {backend}")

    if args.dry_run:
        logger.info("[dry-run] Exiting without building index")
        return 0

    # Create index from config
    try:
        index = create_index_from_config(config)
        logger.info(f"Index created: {index.backend_name}, dim={index.dim}")
    except Exception as e:
        logger.error(f"Failed to create index: {e}")
        return 1

    # TODO: Load embeddings and build index
    if args.embeddings:
        logger.info(f"Embeddings file: {args.embeddings}")
        # embeddings = np.load(args.embeddings)
        # index.build_index(embeddings)
        logger.warning("Embeddings loading not yet implemented")

    # TODO: Save index
    if args.output:
        logger.info(f"Output path: {args.output}")
        # index.save(args.output)
        logger.warning("Index saving not yet implemented")

    logger.info("Index build completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
