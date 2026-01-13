"""
Ontology Loader
===============
本體載入器實現

支援格式:
- OBO (Open Biomedical Ontologies) - 主要格式
- OWL (Web Ontology Language) - 通過 pronto 轉換

支援來源:
- 本地檔案
- URL 下載
- 預設版本 (自動下載)

版本: 1.0.0
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
from urllib.request import urlretrieve
from urllib.error import URLError
import tempfile
import hashlib

from src.core import OntologyLoaderProtocol, DataSource
from src.ontology.base import Ontology, OntologyTerm

logger = logging.getLogger(__name__)


# =============================================================================
# Ontology URLs (Official Sources)
# =============================================================================
ONTOLOGY_URLS: Dict[str, Dict[str, str]] = {
    "hpo": {
        "name": "Human Phenotype Ontology",
        "url": "https://raw.githubusercontent.com/obophenotype/human-phenotype-ontology/master/hp.obo",
        "format": "obo",
    },
    "mondo": {
        "name": "Mondo Disease Ontology",
        "url": "https://purl.obolibrary.org/obo/mondo.obo",
        "format": "obo",
    },
    "go": {
        "name": "Gene Ontology",
        "url": "https://purl.obolibrary.org/obo/go.obo",
        "format": "obo",
    },
    "mp": {
        "name": "Mammalian Phenotype Ontology",
        "url": "https://purl.obolibrary.org/obo/mp.obo",
        "format": "obo",
    },
}

# Data source mapping
ONTOLOGY_DATA_SOURCES: Dict[str, DataSource] = {
    "hpo": DataSource.HPO,
    "mondo": DataSource.MONDO,
    "go": DataSource.GO,
    "mp": DataSource.MGI,  # MP is from MGI
}


# =============================================================================
# Ontology Loader
# =============================================================================
class OntologyLoader(OntologyLoaderProtocol):
    """
    本體載入器

    使用方式:
        loader = OntologyLoader(cache_dir="/path/to/cache")

        # 載入本地檔案
        hpo = loader.load(Path("hp.obo"))

        # 載入預設版本 (自動下載)
        hpo = loader.load_hpo()
        mondo = loader.load_mondo()
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        use_pronto: bool = True,
    ):
        """
        初始化載入器

        Args:
            cache_dir: 快取目錄 (用於下載的本體)
            use_pronto: 是否使用 pronto 庫 (推薦)
        """
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "shepherd_ontology"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.use_pronto = use_pronto

        # Check pronto availability
        self._pronto_available = False
        if use_pronto:
            try:
                import pronto
                self._pronto_available = True
                logger.debug("pronto library available")
            except ImportError:
                logger.warning(
                    "pronto library not installed. "
                    "Using fallback OBO parser (limited functionality)."
                )

        logger.info(f"OntologyLoader initialized (cache: {self.cache_dir})")

    # =========================================================================
    # Main Loading Methods
    # =========================================================================
    def load(self, path: Path) -> Ontology:
        """
        載入本體檔案

        Args:
            path: 本體檔案路徑 (OBO 或 OWL)

        Returns:
            Ontology 實例
        """
        if not path.exists():
            raise FileNotFoundError(f"Ontology file not found: {path}")

        logger.info(f"Loading ontology from: {path}")

        # Determine format from extension
        suffix = path.suffix.lower()
        if suffix == ".obo":
            return self._load_obo(path)
        elif suffix in (".owl", ".xml"):
            return self._load_owl(path)
        else:
            # Try OBO format by default
            logger.warning(f"Unknown file extension '{suffix}', trying OBO format")
            return self._load_obo(path)

    def load_from_url(self, url: str, name: str) -> Ontology:
        """
        從 URL 載入本體

        Args:
            url: 本體 URL
            name: 本體名稱

        Returns:
            Ontology 實例
        """
        # Generate cache filename
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        cache_file = self.cache_dir / f"{name}_{url_hash}.obo"

        # Download if not cached
        if not cache_file.exists():
            logger.info(f"Downloading {name} from {url}...")
            try:
                urlretrieve(url, cache_file)
                logger.info(f"Downloaded to {cache_file}")
            except URLError as e:
                raise RuntimeError(f"Failed to download ontology: {e}")

        return self.load(cache_file)

    # =========================================================================
    # Specific Ontology Loaders
    # =========================================================================
    def load_hpo(self, version: str = "latest") -> Ontology:
        """
        載入 HPO (Human Phenotype Ontology)

        Args:
            version: 版本 ("latest" 使用最新版)

        Returns:
            HPO Ontology 實例
        """
        info = ONTOLOGY_URLS["hpo"]
        ontology = self.load_from_url(info["url"], "hpo")
        ontology._name = "HPO"
        ontology._source = DataSource.HPO
        return ontology

    def load_mondo(self, version: str = "latest") -> Ontology:
        """
        載入 MONDO (Disease Ontology)

        Args:
            version: 版本 ("latest" 使用最新版)

        Returns:
            MONDO Ontology 實例
        """
        info = ONTOLOGY_URLS["mondo"]
        ontology = self.load_from_url(info["url"], "mondo")
        ontology._name = "MONDO"
        ontology._source = DataSource.MONDO
        return ontology

    def load_go(self, version: str = "latest") -> Ontology:
        """
        載入 GO (Gene Ontology)

        Args:
            version: 版本 ("latest" 使用最新版)

        Returns:
            GO Ontology 實例
        """
        info = ONTOLOGY_URLS["go"]
        ontology = self.load_from_url(info["url"], "go")
        ontology._name = "GO"
        ontology._source = DataSource.GO
        return ontology

    def load_mp(self, version: str = "latest") -> Ontology:
        """
        載入 MP (Mammalian Phenotype Ontology)

        用於同源基因的小鼠表型

        Args:
            version: 版本 ("latest" 使用最新版)

        Returns:
            MP Ontology 實例
        """
        info = ONTOLOGY_URLS["mp"]
        ontology = self.load_from_url(info["url"], "mp")
        ontology._name = "MP"
        ontology._source = DataSource.MGI
        return ontology

    # =========================================================================
    # Format-Specific Loaders
    # =========================================================================
    def _load_obo(self, path: Path) -> Ontology:
        """載入 OBO 格式"""
        if self._pronto_available:
            return self._load_with_pronto(path)
        else:
            return self._load_with_fallback_parser(path)

    def _load_owl(self, path: Path) -> Ontology:
        """載入 OWL 格式"""
        if not self._pronto_available:
            raise RuntimeError(
                "OWL format requires pronto library. "
                "Install with: pip install pronto"
            )
        return self._load_with_pronto(path)

    def _load_with_pronto(self, path: Path) -> Ontology:
        """使用 pronto 庫載入"""
        import pronto

        logger.info(f"Loading with pronto: {path}")

        # Load ontology
        ponto = pronto.Ontology(str(path))

        # Extract metadata
        name = ponto.metadata.ontology or path.stem.upper()
        version = ponto.metadata.data_version or "unknown"

        # Create ontology instance
        ontology = Ontology(name=name, version=version)

        # Load terms
        term_count = 0
        for term in ponto.terms():
            # Skip obsolete by default (but still add them marked)
            onto_term = OntologyTerm(
                id=term.id,
                name=term.name or term.id,
                definition=str(term.definition) if term.definition else None,
                synonyms=[syn.description for syn in term.synonyms],
                xrefs=[str(xref) for xref in term.xrefs],
                is_obsolete=term.obsolete,
                namespace=term.namespace,
                alt_ids=list(term.alternate_ids),
            )

            # Add parent relationships (IS_A)
            for parent in term.superclasses(distance=1, with_self=False):
                if parent.id != term.id:  # Avoid self-reference
                    onto_term.parents.add(parent.id)

            ontology._add_term(onto_term)
            term_count += 1

        # Build children index
        ontology._build_children_index()

        logger.info(
            f"Loaded {term_count} terms from {name} "
            f"(version: {version})"
        )

        return ontology

    def _load_with_fallback_parser(self, path: Path) -> Ontology:
        """
        Fallback OBO 解析器 (不依賴 pronto)

        功能有限，僅解析基本結構
        """
        logger.warning("Using fallback OBO parser (limited functionality)")

        ontology = Ontology(name=path.stem.upper(), version="unknown")

        current_term: Optional[Dict[str, Any]] = None
        term_count = 0

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                # Start of term
                if line == "[Term]":
                    # Save previous term
                    if current_term and current_term.get("id"):
                        self._save_term_from_dict(ontology, current_term)
                        term_count += 1
                    current_term = {}
                    continue

                # Skip other stanzas
                if line.startswith("[") and line.endswith("]"):
                    if current_term and current_term.get("id"):
                        self._save_term_from_dict(ontology, current_term)
                        term_count += 1
                    current_term = None
                    continue

                # Parse term fields
                if current_term is not None and ":" in line:
                    key, _, value = line.partition(":")
                    value = value.strip()

                    if key == "id":
                        current_term["id"] = value
                    elif key == "name":
                        current_term["name"] = value
                    elif key == "def":
                        # Extract definition text
                        if value.startswith('"'):
                            end_quote = value.find('"', 1)
                            if end_quote > 0:
                                current_term["definition"] = value[1:end_quote]
                    elif key == "synonym":
                        if "synonyms" not in current_term:
                            current_term["synonyms"] = []
                        # Extract synonym text
                        if value.startswith('"'):
                            end_quote = value.find('"', 1)
                            if end_quote > 0:
                                current_term["synonyms"].append(value[1:end_quote])
                    elif key == "is_a":
                        if "parents" not in current_term:
                            current_term["parents"] = []
                        # Extract parent ID
                        parent_id = value.split("!")[0].strip()
                        current_term["parents"].append(parent_id)
                    elif key == "is_obsolete":
                        current_term["is_obsolete"] = value.lower() == "true"
                    elif key == "namespace":
                        current_term["namespace"] = value

        # Save last term
        if current_term and current_term.get("id"):
            self._save_term_from_dict(ontology, current_term)
            term_count += 1

        # Build children index
        ontology._build_children_index()

        logger.info(f"Loaded {term_count} terms (fallback parser)")

        return ontology

    def _save_term_from_dict(
        self,
        ontology: Ontology,
        term_dict: Dict[str, Any],
    ) -> None:
        """從字典建立術語並添加到本體"""
        term = OntologyTerm(
            id=term_dict["id"],
            name=term_dict.get("name", term_dict["id"]),
            definition=term_dict.get("definition"),
            synonyms=term_dict.get("synonyms", []),
            is_obsolete=term_dict.get("is_obsolete", False),
            namespace=term_dict.get("namespace"),
            parents=set(term_dict.get("parents", [])),
        )
        ontology._add_term(term)

    # =========================================================================
    # Cache Management
    # =========================================================================
    def clear_cache(self) -> None:
        """清除所有快取"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Ontology cache cleared")

    def get_cached_files(self) -> List[Path]:
        """獲取所有快取檔案"""
        return list(self.cache_dir.glob("*.obo"))


# =============================================================================
# Convenience Functions
# =============================================================================
def load_hpo(
    cache_dir: Optional[Path] = None,
    version: str = "latest",
) -> Ontology:
    """
    便捷函數: 載入 HPO

    Example:
        hpo = load_hpo()
        term = hpo.get_term("HP:0001250")
    """
    loader = OntologyLoader(cache_dir=cache_dir)
    return loader.load_hpo(version)


def load_mondo(
    cache_dir: Optional[Path] = None,
    version: str = "latest",
) -> Ontology:
    """便捷函數: 載入 MONDO"""
    loader = OntologyLoader(cache_dir=cache_dir)
    return loader.load_mondo(version)


def load_go(
    cache_dir: Optional[Path] = None,
    version: str = "latest",
) -> Ontology:
    """便捷函數: 載入 GO"""
    loader = OntologyLoader(cache_dir=cache_dir)
    return loader.load_go(version)


def load_mp(
    cache_dir: Optional[Path] = None,
    version: str = "latest",
) -> Ontology:
    """便捷函數: 載入 MP (用於同源基因)"""
    loader = OntologyLoader(cache_dir=cache_dir)
    return loader.load_mp(version)
