"""
HPO Annotation File Parser
===========================
Parses HPO annotation TSV files into the dict format expected by
KnowledgeGraphBuilder.add_*() methods.

Supports:
  - phenotype.hpoa: phenotype-disease annotations
  - genes_to_phenotype.txt: gene-phenotype and gene-disease links

Data files available from: http://purl.obolibrary.org/obo/hp/hpoa/
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class HPOAnnotationParser:
    """Parses HPO annotation files into KnowledgeGraphBuilder-compatible dicts."""

    def __init__(self, mondo_ontology=None):
        """
        Args:
            mondo_ontology: Optional Ontology instance for OMIM->MONDO ID mapping.
                           If provided, OMIM disease IDs will be translated to MONDO.
        """
        self._omim_to_mondo: Dict[str, str] = {}
        self._orpha_to_mondo: Dict[str, str] = {}
        if mondo_ontology is not None:
            self._omim_to_mondo = self.build_omim_to_mondo_map(mondo_ontology)
            self._orpha_to_mondo = self.build_orpha_to_mondo_map(mondo_ontology)
            logger.info(
                f"Built OMIM->MONDO map: {len(self._omim_to_mondo)} mappings; "
                f"ORPHA->MONDO map: {len(self._orpha_to_mondo)} mappings"
            )

    @staticmethod
    def build_omim_to_mondo_map(mondo_ontology) -> Dict[str, str]:
        """
        Build OMIM -> MONDO ID mapping from MONDO ontology xrefs.

        Args:
            mondo_ontology: Ontology instance with MONDO loaded via pronto.

        Returns:
            Dict mapping "OMIM:123456" -> "MONDO:0001234"
        """
        omim_to_mondo: Dict[str, str] = {}

        for term_id in mondo_ontology.get_all_terms(include_obsolete=False):
            if not term_id.startswith("MONDO:"):
                continue
            term_info = mondo_ontology.get_term(term_id)
            if term_info is None:
                continue

            for xref in term_info.get("xrefs", []):
                xref_str = str(xref)
                if xref_str.startswith("OMIM:"):
                    omim_to_mondo[xref_str] = term_id

        return omim_to_mondo

    @staticmethod
    def build_orpha_to_mondo_map(mondo_ontology) -> Dict[str, str]:
        """
        Build ORPHA -> MONDO ID mapping from MONDO ontology xrefs.

        Note the prefix mismatch: phenotype.hpoa (and genes_to_phenotype.txt)
        use "ORPHA:<n>", while MONDO xrefs use "Orphanet:<n>". The returned
        dict is keyed by the annotation-file form ("ORPHA:<n>") so that
        _resolve_disease_id can look it up directly.

        Returns:
            Dict mapping "ORPHA:558" -> "MONDO:0001234"
        """
        orpha_to_mondo: Dict[str, str] = {}

        for term_id in mondo_ontology.get_all_terms(include_obsolete=False):
            if not term_id.startswith("MONDO:"):
                continue
            term_info = mondo_ontology.get_term(term_id)
            if term_info is None:
                continue

            for xref in term_info.get("xrefs", []):
                xref_str = str(xref)
                if xref_str.startswith("Orphanet:"):
                    orpha_id = "ORPHA:" + xref_str.split(":", 1)[1]
                    orpha_to_mondo[orpha_id] = term_id

        return orpha_to_mondo

    def _resolve_disease_id(self, raw_id: str) -> Optional[str]:
        """
        Resolve a disease ID to a MONDO ID if possible.

        Handles: MONDO:*, OMIM:*, ORPHA:*, DECIPHER:*
        Returns None for IDs that can't be resolved.
        """
        if raw_id.startswith("MONDO:"):
            return raw_id
        if raw_id.startswith("OMIM:"):
            return self._omim_to_mondo.get(raw_id)
        if raw_id.startswith("ORPHA:"):
            return self._orpha_to_mondo.get(raw_id)
        # DECIPHER is not mapped for now (only 296 annotations / 47 diseases)
        return None

    def parse_phenotype_hpoa(
        self, path: Path | str
    ) -> List[Dict[str, Any]]:
        """
        Parse phenotype.hpoa into phenotype-disease annotations.

        The file uses '#' comment lines and a tab-separated format.
        Columns (as of 2024 format):
          database_id, disease_name, qualifier, hpo_id, reference,
          evidence, onset, frequency, sex, modifier, aspect, biocuration

        Returns:
            List of {"phenotype_id": "HP:...", "disease_id": "MONDO:...", "frequency": float}
        """
        path = Path(path)
        logger.info(f"Parsing phenotype.hpoa from {path}")

        annotations: List[Dict[str, Any]] = []
        skipped_not = 0
        skipped_unmapped = 0
        seen: Set[Tuple[str, str]] = set()

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                line = line.strip()
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) < 4:
                    continue

                database_id = parts[0]     # e.g. OMIM:101200 or MONDO:0007037
                qualifier = parts[2] if len(parts) > 2 else ""
                hpo_id = parts[3]          # e.g. HP:0001250

                # Skip negated annotations
                if qualifier == "NOT":
                    skipped_not += 1
                    continue

                if not hpo_id.startswith("HP:"):
                    continue

                # Resolve disease ID to MONDO
                disease_id = self._resolve_disease_id(database_id)
                if disease_id is None:
                    skipped_unmapped += 1
                    continue

                # Deduplicate (same phenotype-disease pair)
                pair = (hpo_id, disease_id)
                if pair in seen:
                    continue
                seen.add(pair)

                # Parse frequency if available (column 7)
                frequency = self._parse_frequency(parts[7] if len(parts) > 7 else "")

                annotations.append({
                    "phenotype_id": hpo_id,
                    "disease_id": disease_id,
                    "frequency": frequency,
                })

        logger.info(
            f"Parsed {len(annotations)} phenotype-disease annotations "
            f"(skipped {skipped_not} NOT, {skipped_unmapped} unmapped disease IDs)"
        )
        return annotations

    def parse_genes_to_phenotype(
        self, path: Path | str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Parse genes_to_phenotype.txt into gene-phenotype and gene-disease links.

        Tab-separated columns:
          gene_id (entrez), gene_symbol, hpo_id, hpo_name,
          frequency, disease_id

        Returns:
            (gene_phenotype_list, gene_disease_list) where:
            - gene_phenotype_list: [{"gene_id": "CLN3", "phenotype_id": "HP:..."}]
            - gene_disease_list: [{"gene_id": "CLN3", "disease_id": "MONDO:...", "score": 1.0}]
        """
        path = Path(path)
        logger.info(f"Parsing genes_to_phenotype.txt from {path}")

        gene_pheno: List[Dict[str, Any]] = []
        gene_disease: List[Dict[str, Any]] = []
        seen_gp: Set[Tuple[str, str]] = set()
        seen_gd: Set[Tuple[str, str]] = set()

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#") or line.startswith("gene_id"):
                    continue
                line = line.strip()
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) < 4:
                    continue

                gene_symbol = parts[1]     # HGNC symbol
                hpo_id = parts[2]          # HP:XXXXXXX

                if not gene_symbol or not hpo_id.startswith("HP:"):
                    continue

                # Gene-phenotype link
                gp_key = (gene_symbol, hpo_id)
                if gp_key not in seen_gp:
                    seen_gp.add(gp_key)
                    gene_pheno.append({
                        "gene_id": gene_symbol,
                        "phenotype_id": hpo_id,
                    })

                # Gene-disease link (from disease_id column, index 5)
                if len(parts) > 5:
                    raw_disease_id = parts[5]
                    disease_id = self._resolve_disease_id(raw_disease_id)
                    if disease_id:
                        gd_key = (gene_symbol, disease_id)
                        if gd_key not in seen_gd:
                            seen_gd.add(gd_key)
                            gene_disease.append({
                                "gene_id": gene_symbol,
                                "gene_symbol": gene_symbol,
                                "disease_id": disease_id,
                                "score": 1.0,
                            })

        logger.info(
            f"Parsed {len(gene_pheno)} gene-phenotype and "
            f"{len(gene_disease)} gene-disease links"
        )
        return gene_pheno, gene_disease

    @staticmethod
    def _parse_frequency(freq_str: str) -> float:
        """Parse HPO frequency annotation to a float in [0, 1]."""
        freq_str = freq_str.strip()
        if not freq_str:
            return 1.0

        # HPO frequency terms
        hpo_frequencies = {
            "HP:0040280": 1.0,    # Obligate (100%)
            "HP:0040281": 0.90,   # Very frequent (80-99%)
            "HP:0040282": 0.55,   # Frequent (30-79%)
            "HP:0040283": 0.10,   # Occasional (5-29%)
            "HP:0040284": 0.02,   # Very rare (1-4%)
            "HP:0040285": 0.005,  # Excluded (0%)
        }

        if freq_str in hpo_frequencies:
            return hpo_frequencies[freq_str]

        # Percentage: "45%"
        if freq_str.endswith("%"):
            try:
                return float(freq_str[:-1]) / 100.0
            except ValueError:
                pass

        # Fraction: "3/12"
        if "/" in freq_str:
            try:
                num, den = freq_str.split("/")
                return float(num) / float(den)
            except (ValueError, ZeroDivisionError):
                pass

        return 1.0
