"""
SHEPHERD-Advanced Ontology Loader
=================================
本體載入器，支援 OBO 和 OWL 格式

支援的本體:
- HPO (Human Phenotype Ontology)
- MONDO (Disease Ontology)
- GO (Gene Ontology)
- MP (Mammalian Phenotype Ontology) - 用於同源基因

版本: 1.0.0
"""
from __future__ import annotations

import gzip
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple
from urllib.request import urlretrieve
from urllib.error import URLError

from src.core.types import DataSource

logger = logging.getLogger(__name__)


# =============================================================================
# OBO Term Data Structure
# =============================================================================
@dataclass
class OBOTerm:
    """
    OBO 格式的術語結構
    """
    id: str
    name: str
    namespace: Optional[str] = None
    definition: Optional[str] = None

    # Relationships
    is_a: List[str] = field(default_factory=list)  # Parent terms
    part_of: List[str] = field(default_factory=list)

    # Alternative identifiers
    alt_ids: List[str] = field(default_factory=list)

    # Synonyms
    synonyms: List[Tuple[str, str]] = field(default_factory=list)  # (text, type)

    # Cross-references
    xrefs: List[str] = field(default_factory=list)

    # Status
    is_obsolete: bool = False
    replaced_by: Optional[str] = None
    consider: List[str] = field(default_factory=list)

    # Additional properties
    properties: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class OBOHeader:
    """
    OBO 檔案 header 資訊
    """
    format_version: Optional[str] = None
    data_version: Optional[str] = None
    ontology: Optional[str] = None
    date: Optional[str] = None
    saved_by: Optional[str] = None
    subsetdef: List[str] = field(default_factory=list)
    default_namespace: Optional[str] = None
    remark: Optional[str] = None

    # Additional properties
    properties: Dict[str, str] = field(default_factory=dict)


# =============================================================================
# OBO Parser
# =============================================================================
class OBOParser:
    """
    OBO 格式解析器

    支援 OBO 1.2 和 1.4 格式
    """

    # Regex patterns
    TAG_VALUE_PATTERN = re.compile(r'^(\S+):\s*(.*)$')
    XREF_PATTERN = re.compile(r'\[([^\]]+)\]')
    SYNONYM_PATTERN = re.compile(r'"([^"]+)"\s+(\w+)')
    DEF_PATTERN = re.compile(r'"([^"]+)"')

    def __init__(self):
        self.header: Optional[OBOHeader] = None
        self.terms: Dict[str, OBOTerm] = {}
        self.typedefs: Dict[str, Dict[str, Any]] = {}

    def parse_file(self, file_path: Path) -> Tuple[OBOHeader, Dict[str, OBOTerm]]:
        """
        解析 OBO 檔案

        Args:
            file_path: OBO 檔案路徑 (支援 .obo 和 .obo.gz)

        Returns:
            (header, terms_dict)
        """
        logger.info(f"Parsing OBO file: {file_path}")

        self.header = OBOHeader()
        self.terms = {}
        self.typedefs = {}

        # Handle gzipped files
        if str(file_path).endswith('.gz'):
            open_func = lambda p: gzip.open(p, 'rt', encoding='utf-8')
        else:
            open_func = lambda p: open(p, 'r', encoding='utf-8')

        with open_func(file_path) as f:
            current_stanza = None
            current_data: Dict[str, Any] = {}

            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith('!'):
                    continue

                # Check for stanza start
                if line.startswith('[') and line.endswith(']'):
                    # Save previous stanza
                    if current_stanza:
                        self._save_stanza(current_stanza, current_data)

                    # Start new stanza
                    current_stanza = line[1:-1]
                    current_data = {}
                    continue

                # Parse tag-value pair
                match = self.TAG_VALUE_PATTERN.match(line)
                if match:
                    tag, value = match.groups()
                    value = value.strip()

                    # Remove trailing comments
                    if ' !' in value:
                        value = value.split(' !')[0].strip()

                    if current_stanza is None:
                        # Header
                        self._parse_header_tag(tag, value)
                    else:
                        # Stanza content
                        if tag not in current_data:
                            current_data[tag] = []
                        current_data[tag].append(value)

            # Save last stanza
            if current_stanza:
                self._save_stanza(current_stanza, current_data)

        logger.info(f"Parsed {len(self.terms)} terms from {file_path.name}")
        return self.header, self.terms

    def _parse_header_tag(self, tag: str, value: str) -> None:
        """解析 header tag"""
        if tag == 'format-version':
            self.header.format_version = value
        elif tag == 'data-version':
            self.header.data_version = value
        elif tag == 'ontology':
            self.header.ontology = value
        elif tag == 'date':
            self.header.date = value
        elif tag == 'saved-by':
            self.header.saved_by = value
        elif tag == 'default-namespace':
            self.header.default_namespace = value
        elif tag == 'subsetdef':
            self.header.subsetdef.append(value)
        elif tag == 'remark':
            self.header.remark = value
        else:
            self.header.properties[tag] = value

    def _save_stanza(self, stanza_type: str, data: Dict[str, List[str]]) -> None:
        """保存解析的 stanza"""
        if stanza_type == 'Term':
            term = self._parse_term(data)
            if term and term.id:
                self.terms[term.id] = term
                # Also index by alt_ids
                for alt_id in term.alt_ids:
                    if alt_id not in self.terms:
                        self.terms[alt_id] = term
        elif stanza_type == 'Typedef':
            typedef_id = data.get('id', [''])[0]
            if typedef_id:
                self.typedefs[typedef_id] = data

    def _parse_term(self, data: Dict[str, List[str]]) -> Optional[OBOTerm]:
        """解析 Term stanza"""
        term_id = data.get('id', [''])[0]
        if not term_id:
            return None

        term = OBOTerm(
            id=term_id,
            name=data.get('name', [''])[0],
            namespace=data.get('namespace', [self.header.default_namespace])[0],
        )

        # Definition
        if 'def' in data:
            def_match = self.DEF_PATTERN.search(data['def'][0])
            if def_match:
                term.definition = def_match.group(1)

        # IS_A relationships
        for is_a in data.get('is_a', []):
            # Format: HP:0000001 ! Term name
            parent_id = is_a.split()[0]
            term.is_a.append(parent_id)

        # Part_of relationships (from relationship tag)
        for rel in data.get('relationship', []):
            parts = rel.split()
            if len(parts) >= 2:
                rel_type, target = parts[0], parts[1]
                if rel_type == 'part_of':
                    term.part_of.append(target)
                # Store other relationships in properties
                if rel_type not in term.properties:
                    term.properties[rel_type] = []
                term.properties[rel_type].append(target)

        # Alternative IDs
        term.alt_ids = data.get('alt_id', [])

        # Synonyms
        for syn in data.get('synonym', []):
            syn_match = self.SYNONYM_PATTERN.search(syn)
            if syn_match:
                term.synonyms.append((syn_match.group(1), syn_match.group(2)))

        # Cross-references
        term.xrefs = data.get('xref', [])

        # Obsolete status
        if 'is_obsolete' in data and data['is_obsolete'][0].lower() == 'true':
            term.is_obsolete = True

        # Replaced by
        if 'replaced_by' in data:
            term.replaced_by = data['replaced_by'][0]

        # Consider
        term.consider = data.get('consider', [])

        return term


# =============================================================================
# Ontology Loader
# =============================================================================
class OntologyLoader:
    """
    本體載入器

    支援從檔案或 URL 載入本體
    """

    # Known ontology URLs
    ONTOLOGY_URLS = {
        'hpo': 'https://raw.githubusercontent.com/obophenotype/human-phenotype-ontology/master/hp.obo',
        'mondo': 'https://raw.githubusercontent.com/monarch-initiative/mondo/master/mondo.obo',
        'go': 'http://purl.obolibrary.org/obo/go.obo',
        'mp': 'http://purl.obolibrary.org/obo/mp.obo',
    }

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Args:
            cache_dir: 快取目錄，用於存放下載的本體檔案
        """
        self.cache_dir = cache_dir or Path.home() / '.shepherd' / 'ontologies'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.parser = OBOParser()
        self._loaded_ontologies: Dict[str, 'Ontology'] = {}

    def load(self, path: Path) -> 'Ontology':
        """
        載入本體檔案

        Args:
            path: OBO/OWL 檔案路徑

        Returns:
            Ontology 物件
        """
        if not path.exists():
            raise FileNotFoundError(f"Ontology file not found: {path}")

        if str(path).endswith('.obo') or str(path).endswith('.obo.gz'):
            header, terms = self.parser.parse_file(path)
            ontology = Ontology(header, terms)
            return ontology
        elif str(path).endswith('.owl') or str(path).endswith('.owl.gz'):
            raise NotImplementedError("OWL format not yet supported. Please use OBO format.")
        else:
            raise ValueError(f"Unsupported ontology format: {path.suffix}")

    def load_hpo(self, version: str = "latest", force_download: bool = False) -> 'Ontology':
        """
        載入 HPO (Human Phenotype Ontology)

        Args:
            version: 版本號或 "latest"
            force_download: 是否強制重新下載
        """
        return self._load_known_ontology('hpo', version, force_download)

    def load_mondo(self, version: str = "latest", force_download: bool = False) -> 'Ontology':
        """載入 MONDO (Disease Ontology)"""
        return self._load_known_ontology('mondo', version, force_download)

    def load_go(self, version: str = "latest", force_download: bool = False) -> 'Ontology':
        """載入 GO (Gene Ontology)"""
        return self._load_known_ontology('go', version, force_download)

    def load_mp(self, version: str = "latest", force_download: bool = False) -> 'Ontology':
        """載入 MP (Mammalian Phenotype Ontology) - 用於同源基因"""
        return self._load_known_ontology('mp', version, force_download)

    def _load_known_ontology(
        self,
        ontology_name: str,
        version: str,
        force_download: bool
    ) -> 'Ontology':
        """載入已知本體"""
        cache_key = f"{ontology_name}_{version}"

        # Check memory cache
        if cache_key in self._loaded_ontologies and not force_download:
            logger.info(f"Using cached {ontology_name} ontology")
            return self._loaded_ontologies[cache_key]

        # Check file cache
        cache_file = self.cache_dir / f"{ontology_name}.obo"

        if not cache_file.exists() or force_download:
            # Download
            url = self.ONTOLOGY_URLS.get(ontology_name)
            if not url:
                raise ValueError(f"Unknown ontology: {ontology_name}")

            logger.info(f"Downloading {ontology_name} ontology from {url}")
            try:
                urlretrieve(url, cache_file)
                logger.info(f"Downloaded {ontology_name} to {cache_file}")
            except URLError as e:
                if cache_file.exists():
                    logger.warning(f"Download failed, using cached file: {e}")
                else:
                    raise RuntimeError(f"Failed to download {ontology_name}: {e}")

        # Load from file
        ontology = self.load(cache_file)
        ontology._source = DataSource[ontology_name.upper()] if ontology_name.upper() in DataSource.__members__ else None

        # Cache in memory
        self._loaded_ontologies[cache_key] = ontology

        return ontology


# =============================================================================
# Ontology Class (imported from hierarchy.py)
# =============================================================================
# Forward declaration - actual implementation in hierarchy.py
from src.ontology.hierarchy import Ontology


# =============================================================================
# Factory Function
# =============================================================================
def create_ontology_loader(cache_dir: Optional[Path] = None) -> OntologyLoader:
    """
    工廠函數: 創建本體載入器

    Args:
        cache_dir: 快取目錄

    Returns:
        OntologyLoader 實例
    """
    return OntologyLoader(cache_dir)
