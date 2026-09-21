"""
SHEPHERD-Advanced Ontology Loader
=================================
本體載入器，使用 pronto 作為後端

pronto 是專為生物醫學本體設計的 Python 庫，支援:
- OBO 1.4 格式
- OWL 格式
- OBO Graphs (JSON)

支援的本體:
- HPO (Human Phenotype Ontology)
- MONDO (Disease Ontology)
- GO (Gene Ontology)
- MP (Mammalian Phenotype Ontology) - 用於同源基因

版本: 1.1.0
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional
from urllib.request import urlretrieve
from urllib.error import URLError

import pronto

from src.core.types import DataSource
from src.ontology.roles import check_ontology_role

logger = logging.getLogger(__name__)


# =============================================================================
# Ontology Loader using Pronto
# =============================================================================
class OntologyImportError(ValueError):
    """The file declares imports, which this project does not resolve.

    Its own type so a caller can tell "this artifact needs a dependency story"
    from "this file is the wrong ontology" — different remedies, and a single
    `ValueError` would make them one message to an operator.
    """


class OntologyLoader:
    """
    本體載入器 (使用 pronto 後端)

    pronto 是專為生物醫學本體設計的庫，比手寫解析器更可靠
    """

    # Known ontology URLs (OBO Foundry PURLs — stable permanent redirects)
    ONTOLOGY_URLS = {
        'hpo': 'http://purl.obolibrary.org/obo/hp.obo',
        'mondo': 'http://purl.obolibrary.org/obo/mondo.obo',
        'go': 'http://purl.obolibrary.org/obo/go.obo',
        'mp': 'http://purl.obolibrary.org/obo/mp.obo',
    }

    # Alternative OWL URLs (if OBO fails)
    ONTOLOGY_OWL_URLS = {
        'hpo': 'http://purl.obolibrary.org/obo/hp.owl',
        'mondo': 'http://purl.obolibrary.org/obo/mondo.owl',
    }

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Args:
            cache_dir: 快取目錄，用於存放下載的本體檔案
        """
        self.cache_dir = cache_dir or Path.home() / '.shepherd' / 'ontologies'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._loaded_ontologies: Dict[str, 'Ontology'] = {}

    #: The only value `version` can honour. Anything else names a release this
    #: loader has no way to fetch or select, and saying so is the whole point of
    #: `_require_supported_version`.
    SUPPORTED_VERSION = "latest"

    @classmethod
    def _require_supported_version(cls, ontology_name: str, version: str) -> None:
        """Refuse a version this loader cannot honour, instead of ignoring it.

        **The argument used to read as version selection and do nothing of the
        kind.** `_load_known_ontology` folds `version` into an in-memory cache
        key and then opens `<cache_dir>/<name>.obo` regardless, so
        `load_mondo(version="2026-01-01")` returned whatever that file happened
        to be — a different release, silently, under the name of the one asked
        for. Two different version strings in one process even produced two
        cache entries over the same bytes.

        Selecting a release is Phase 2's work (a path per ontology). Until it
        exists, the honest behaviour is to refuse: a caller that asked for a
        specific vintage and received another has no way to find out, and a
        knowledge graph carries the consequence into every index it assigns.

        Raises:
            ValueError: naming what was asked for and what to do instead.
        """
        if version != cls.SUPPORTED_VERSION:
            raise ValueError(
                f"load of {ontology_name!r} asked for version {version!r}, and "
                f"this loader can only honour {cls.SUPPORTED_VERSION!r}. It "
                "reads one file per ontology from its cache directory and has "
                "no way to fetch or select a release. Point `cache_dir` at a "
                "directory holding the file you mean, or rebuild with the "
                "ontology you want in place — do not rely on this argument to "
                "choose one."
            )

    #: Imports are never resolved. See `load` — this is the whole of the
    #: imports policy's first step, and it is a constant so that no call site
    #: can reintroduce the default by omitting the argument.
    IMPORT_DEPTH = 0

    def load(self, path: Path, expect: Optional[str] = None) -> 'Ontology':
        """Parse one ontology file. **Nothing is fetched and nothing is guessed.**

        Args:
            path: the OBO/OWL file to read.
            expect: the canonical ontology this file is being used as, e.g.
                `"hpo"`. When given, the role check of `PLAN_ONTOLOGY_PHASE2.md`
                §3.4 runs before this returns.

        Returns:
            The loaded `Ontology`.

        Raises:
            FileNotFoundError: the path is not there.
            OntologyImportError: the file declares an import (see below).
            OntologyRoleError: `expect` was given and the file is not it.

        **The imports policy, and why `import_depth=0` alone is not it.**
        `pronto 2.7.3` defaults `import_depth` to `-1`: a root file carrying
        `import:` lines resolves them without bound, over the network, at parse
        time — and the root file's digest does not cover what they contributed.
        Setting the depth to 0 stops the fetching and **introduces a worse
        defect**, measured rather than assumed:

            import_depth=0   loads silently, one term, no warning, imports == {}
            import_depth=1   raises URLError
            import_depth=-1  raises URLError

        A build would then be missing whatever the import carried, with nothing
        saying so — the silent suppression `PLAN_ONTOLOGY_PROVENANCE.md` §4.2
        rules out. `metadata.imports` keeps the declared set at depth 0, which
        is what makes an honest refusal possible: **parse without fetching, then
        refuse on the declaration.**

        Every declared import is refused, including one pointing at a local
        file. Resolving those is a dependency story with its own digests and its
        own provenance schema, and nothing measured says an artifact in use
        needs it.
        """
        if not path.exists():
            raise FileNotFoundError(f"Ontology file not found: {path}")

        logger.info(f"Loading ontology from {path}")

        # pronto 自動偵測格式 (OBO, OWL, JSON)
        pronto_ont = pronto.Ontology(str(path), import_depth=self.IMPORT_DEPTH)

        declared_imports = tuple(getattr(pronto_ont.metadata, "imports", ()) or ())
        if declared_imports:
            listed = "\n".join(f"  {item}" for item in sorted(declared_imports))
            raise OntologyImportError(
                f"{path} declares {len(declared_imports)} import(s) and this "
                f"project requires a self-contained ontology file:\n{listed}\n"
                "They were not fetched. Resolving them would pull content this "
                "file's digest does not cover, and ignoring them would build a "
                "graph missing whatever they carry while looking complete. "
                "Supply a file that declares no imports."
            )

        # 包裝成我們的 Ontology 類
        ontology = Ontology(pronto_ont, source_path=path)

        if expect is not None:
            # **Before this returns, so no caller can reach a build without
            # it.** A check the caller has to remember is a check that is
            # missing wherever somebody forgot.
            check_ontology_role(ontology, expect, source=path)

        logger.info(f"Loaded {ontology.num_terms} terms from {path.name}")
        return ontology

    def load_hpo(self, version: str = "latest", force_download: bool = False) -> 'Ontology':
        """
        載入 HPO (Human Phenotype Ontology)

        Args:
            version: must be "latest"; a specific release is refused rather
                than silently ignored -- see `_require_supported_version`
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
        self._require_supported_version(ontology_name, version)
        cache_key = f"{ontology_name}_{version}"

        # Check memory cache
        if cache_key in self._loaded_ontologies and not force_download:
            logger.info(f"Using cached {ontology_name} ontology")
            return self._loaded_ontologies[cache_key]

        # Check file cache (try OBO first, then OWL)
        cache_file_obo = self.cache_dir / f"{ontology_name}.obo"
        cache_file_owl = self.cache_dir / f"{ontology_name}.owl"

        cache_file = None
        if cache_file_obo.exists() and not force_download:
            cache_file = cache_file_obo
        elif cache_file_owl.exists() and not force_download:
            cache_file = cache_file_owl

        if cache_file is None or force_download:
            # Download
            cache_file = self._download_ontology(ontology_name, force_download)

        # Load from file
        ontology = self.load(cache_file)

        # Set source info
        ontology._source = DataSource[ontology_name.upper()] if ontology_name.upper() in DataSource.__members__ else None
        ontology._ontology_name = ontology_name

        # Cache in memory
        self._loaded_ontologies[cache_key] = ontology

        return ontology

    # Manual download instructions per ontology
    ONTOLOGY_MANUAL_INSTRUCTIONS = {
        'hpo': 'https://hpo.jax.org/data/ontology  (download hp.obo)',
        'mondo': 'https://mondo.monarchinitiative.org/pages/download/  (download mondo.obo)',
        'go': 'https://geneontology.org/docs/download-ontology/  (download go.obo)',
        'mp': 'https://www.informatics.jax.org/vocabulary/mp_ontology  (download mp.obo)',
    }

    def _download_ontology(self, ontology_name: str, force_download: bool) -> Path:
        """Download ontology file, with manual download instructions on failure."""
        # Try OBO first
        url = self.ONTOLOGY_URLS.get(ontology_name)
        cache_file = self.cache_dir / f"{ontology_name}.obo"

        if url:
            logger.info(f"Downloading {ontology_name} ontology from {url}")
            try:
                urlretrieve(url, cache_file)
                logger.info(f"Downloaded {ontology_name} to {cache_file}")
                return cache_file
            except URLError as e:
                logger.warning(f"Failed to download OBO: {e}")

        # Try OWL as fallback
        owl_url = self.ONTOLOGY_OWL_URLS.get(ontology_name)
        cache_file_owl = self.cache_dir / f"{ontology_name}.owl"

        if owl_url:
            logger.info(f"Trying OWL format from {owl_url}")
            try:
                urlretrieve(owl_url, cache_file_owl)
                logger.info(f"Downloaded {ontology_name} OWL to {cache_file_owl}")
                return cache_file_owl
            except URLError as e:
                logger.error(f"Failed to download OWL: {e}")

        # Check if we have a cached version
        if cache_file.exists():
            logger.warning(f"Download failed, using existing cache: {cache_file}")
            return cache_file
        if cache_file_owl.exists():
            logger.warning(f"Download failed, using existing cache: {cache_file_owl}")
            return cache_file_owl

        # All attempts failed — give clear manual download instructions
        manual_hint = self.ONTOLOGY_MANUAL_INSTRUCTIONS.get(ontology_name, "")
        raise RuntimeError(
            f"Failed to download {ontology_name} ontology.\n"
            f"\n"
            f"  Auto-download URLs may be outdated. Please download manually:\n"
            f"    {manual_hint}\n"
            f"\n"
            f"  Then place the file at:\n"
            f"    {cache_file}\n"
            f"\n"
            f"  The file will be cached and reused on subsequent runs."
        )


# =============================================================================
# Ontology Class (wraps pronto.Ontology)
# =============================================================================
from src.ontology.hierarchy import Ontology


# =============================================================================
# Legacy OBO Parser (kept for compatibility with test fixtures)
# =============================================================================
from dataclasses import dataclass, field
from typing import Any, List, Tuple
import re


@dataclass
class OBOTerm:
    """OBO 格式的術語結構 (legacy, for test fixtures)"""
    id: str
    name: str
    namespace: Optional[str] = None
    definition: Optional[str] = None
    is_a: List[str] = field(default_factory=list)
    part_of: List[str] = field(default_factory=list)
    alt_ids: List[str] = field(default_factory=list)
    synonyms: List[Tuple[str, str]] = field(default_factory=list)
    xrefs: List[str] = field(default_factory=list)
    is_obsolete: bool = False
    replaced_by: Optional[str] = None
    consider: List[str] = field(default_factory=list)
    properties: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class OBOHeader:
    """OBO 檔案 header 資訊 (legacy, for test fixtures)"""
    format_version: Optional[str] = None
    data_version: Optional[str] = None
    ontology: Optional[str] = None
    date: Optional[str] = None
    saved_by: Optional[str] = None
    subsetdef: List[str] = field(default_factory=list)
    default_namespace: Optional[str] = None
    remark: Optional[str] = None
    properties: Dict[str, str] = field(default_factory=dict)


class OBOParser:
    """
    OBO 格式解析器 (legacy, for test fixtures)

    Note: 對於生產環境，建議使用 OntologyLoader (基於 pronto)
    這個解析器主要用於測試 fixtures
    """

    TAG_VALUE_PATTERN = re.compile(r'^(\S+):\s*(.*)$')
    SYNONYM_PATTERN = re.compile(r'"([^"]+)"\s+(\w+)')
    DEF_PATTERN = re.compile(r'"([^"]+)"')

    def __init__(self):
        self.header: Optional[OBOHeader] = None
        self.terms: Dict[str, OBOTerm] = {}

    def parse_file(self, file_path: Path) -> Tuple[OBOHeader, Dict[str, OBOTerm]]:
        """解析 OBO 檔案"""
        import gzip

        logger.info(f"Parsing OBO file (legacy parser): {file_path}")

        self.header = OBOHeader()
        self.terms = {}

        if str(file_path).endswith('.gz'):
            open_func = lambda p: gzip.open(p, 'rt', encoding='utf-8')
        else:
            open_func = lambda p: open(p, 'r', encoding='utf-8')

        with open_func(file_path) as f:
            current_stanza = None
            current_data: Dict[str, Any] = {}

            for line in f:
                line = line.strip()

                if not line or line.startswith('!'):
                    continue

                if line.startswith('[') and line.endswith(']'):
                    if current_stanza:
                        self._save_stanza(current_stanza, current_data)
                    current_stanza = line[1:-1]
                    current_data = {}
                    continue

                match = self.TAG_VALUE_PATTERN.match(line)
                if match:
                    tag, value = match.groups()
                    value = value.strip()
                    if ' !' in value:
                        value = value.split(' !')[0].strip()

                    if current_stanza is None:
                        self._parse_header_tag(tag, value)
                    else:
                        if tag not in current_data:
                            current_data[tag] = []
                        current_data[tag].append(value)

            if current_stanza:
                self._save_stanza(current_stanza, current_data)

        logger.info(f"Parsed {len(self.terms)} terms")
        return self.header, self.terms

    def _parse_header_tag(self, tag: str, value: str) -> None:
        if tag == 'format-version':
            self.header.format_version = value
        elif tag == 'data-version':
            self.header.data_version = value
        elif tag == 'ontology':
            self.header.ontology = value
        elif tag == 'default-namespace':
            self.header.default_namespace = value
        else:
            self.header.properties[tag] = value

    def _save_stanza(self, stanza_type: str, data: Dict[str, List[str]]) -> None:
        if stanza_type == 'Term':
            term = self._parse_term(data)
            if term and term.id:
                self.terms[term.id] = term

    def _parse_term(self, data: Dict[str, List[str]]) -> Optional[OBOTerm]:
        term_id = data.get('id', [''])[0]
        if not term_id:
            return None

        term = OBOTerm(
            id=term_id,
            name=data.get('name', [''])[0],
            namespace=data.get('namespace', [self.header.default_namespace])[0] if self.header else None,
        )

        if 'def' in data:
            def_match = self.DEF_PATTERN.search(data['def'][0])
            if def_match:
                term.definition = def_match.group(1)

        for is_a in data.get('is_a', []):
            parent_id = is_a.split()[0]
            term.is_a.append(parent_id)

        for syn in data.get('synonym', []):
            syn_match = self.SYNONYM_PATTERN.search(syn)
            if syn_match:
                term.synonyms.append((syn_match.group(1), syn_match.group(2)))

        term.xrefs = data.get('xref', [])

        if 'is_obsolete' in data and data['is_obsolete'][0].lower() == 'true':
            term.is_obsolete = True

        if 'replaced_by' in data:
            term.replaced_by = data['replaced_by'][0]

        return term


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
