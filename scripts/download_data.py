#!/usr/bin/env python3
"""
Download External Data for SHEPHERD-Advanced
=============================================
Downloads annotation files needed for KG construction.

Ontology files (HPO, MONDO, GO) are handled separately by OntologyLoader.
This script downloads *annotation* files that link ontology terms together:
  - phenotype.hpoa: phenotype-disease annotations from HPO project
  - genes_to_phenotype.txt: gene-phenotype and gene-disease links

Usage:
    # Download all sources
    python scripts/download_data.py

    # Download specific source
    python scripts/download_data.py --sources hpo_annotations

    # Custom output directory
    python scripts/download_data.py --output-dir data/external/

    # Force re-download
    python scripts/download_data.py --force
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


DATA_SOURCES = {
    "hpo_annotations": {
        "description": "HPO phenotype-disease and gene-phenotype annotations",
        "files": [
            {
                "url": "http://purl.obolibrary.org/obo/hp/hpoa/phenotype.hpoa",
                "filename": "phenotype.hpoa",
                "description": "Phenotype-disease annotations (~250K rows)",
                "expected_header": "#description:",
            },
            {
                "url": "http://purl.obolibrary.org/obo/hp/hpoa/genes_to_phenotype.txt",
                "filename": "genes_to_phenotype.txt",
                "description": "Gene-phenotype associations (~200K rows)",
                "expected_header": "gene_id",
            },
        ],
    },
}


def download_file(url: str, dest: Path, expected_header: str | None = None) -> bool:
    """Download a file and verify it's non-empty and has the expected header."""
    logger.info(f"Downloading {url} -> {dest}")
    try:
        urlretrieve(url, dest)
    except URLError as e:
        logger.error(f"Failed to download {url}: {e}")
        return False

    if not dest.exists() or dest.stat().st_size == 0:
        logger.error(f"Downloaded file is empty: {dest}")
        return False

    if expected_header:
        with open(dest, "r", encoding="utf-8") as f:
            first_line = f.readline()
        if not first_line.startswith(expected_header):
            logger.warning(
                f"Header mismatch in {dest.name}: "
                f"expected '{expected_header}...', got '{first_line[:60].strip()}...'"
            )

    size_mb = dest.stat().st_size / (1024 * 1024)
    logger.info(f"Downloaded {dest.name} ({size_mb:.1f} MB)")
    return True


def download_source(source_name: str, output_dir: Path, force: bool = False) -> bool:
    """Download all files for a given data source."""
    if source_name not in DATA_SOURCES:
        logger.error(f"Unknown source: {source_name}. Available: {list(DATA_SOURCES)}")
        return False

    source = DATA_SOURCES[source_name]
    logger.info(f"Downloading source: {source_name} ({source['description']})")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_ok = True
    for file_info in source["files"]:
        dest = output_dir / file_info["filename"]

        if dest.exists() and not force:
            logger.info(f"Skipping {file_info['filename']} (already exists, use --force to re-download)")
            continue

        ok = download_file(
            url=file_info["url"],
            dest=dest,
            expected_header=file_info.get("expected_header"),
        )
        if not ok:
            all_ok = False

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Download external data files for SHEPHERD-Advanced KG construction"
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=list(DATA_SOURCES.keys()),
        help=f"Data sources to download (default: all). Available: {list(DATA_SOURCES.keys())}",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/external",
        help="Output directory (default: data/external/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_sources",
        help="List available data sources and exit",
    )
    args = parser.parse_args()

    if args.list_sources:
        print("Available data sources:")
        for name, info in DATA_SOURCES.items():
            print(f"  {name}: {info['description']}")
            for f in info["files"]:
                print(f"    - {f['filename']}: {f['description']}")
        return

    output_dir = Path(args.output_dir)
    failed = []

    for source_name in args.sources:
        ok = download_source(source_name, output_dir, force=args.force)
        if not ok:
            failed.append(source_name)

    print()
    if failed:
        print(f"WARNING: {len(failed)} source(s) had errors: {failed}")
        sys.exit(1)
    else:
        print(f"All downloads complete. Files saved to {output_dir}/")
        print("\nNext step: build the knowledge graph:")
        print(f"  python scripts/build_knowledge_graph.py --external-dir {output_dir}")


if __name__ == "__main__":
    main()
