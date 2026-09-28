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

import hashlib
import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
import pronto

from src.core.types import DataSource
from src.ontology.roles import check_ontology_role

logger = logging.getLogger(__name__)


# =============================================================================
# Ontology Loader using Pronto
# =============================================================================
class OntologyFetchError(RuntimeError):
    """No acceptable copy of an ontology could be obtained.

    A `RuntimeError` so existing callers that caught the old untyped failure
    still do; its own type so the build's entry point can turn it into an exit
    status without swallowing unrelated runtime errors.
    """


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

    # The download URLs live in `src/ontology/settings.py`
    # (`DEFAULT_ONTOLOGY_SOURCES`), overridable from `configs/deployment.yaml`.
    # `ONTOLOGY_URLS` and `ONTOLOGY_OWL_URLS` used to sit here; after the move
    # nothing read them, they disagreed with the settings defaults (no OWL
    # entries for GO or MP), and a docstring called them "one list and not
    # two". They are gone rather than left to be edited by someone who
    # believes they still do something.

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Args:
            cache_dir: 快取目錄，用於存放下載的本體檔案
        """
        self.cache_dir = cache_dir or Path.home() / '.shepherd' / 'ontologies'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._loaded_ontologies: Dict[str, 'Ontology'] = {}
        self._ontology_settings = None

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

    def _settings(self):
        """The deployment's ontology settings, read once per loader.

        Read lazily rather than in `__init__`, so constructing a loader for a
        directory of files does not require a configuration file to exist.
        """
        if self._ontology_settings is None:
            from src.ontology.settings import load_ontology_settings

            self._ontology_settings = load_ontology_settings()
        return self._ontology_settings

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

        from src.ontology.resolver import IMPORT_WITHOUT_TARGET
        from src.ontology.resolver import declared_imports as _listed_imports

        # **One open, three readings of the same bytes.** The digest, the parse
        # and the import scan all go through this handle, so they describe one
        # file even if the path is replaced while this runs — by another build
        # publishing a fresh download to the same cache name, for one. Hashing
        # the path afterwards (as the build used to) recorded whatever the path
        # named by then; a digest is only a claim about what was built from if
        # it is taken from what was parsed.
        with open(path, "rb") as handle:
            digest = hashlib.sha256()
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
            handle.seek(0)
            # **pronto gets a descriptor of its own for the same open file.**
            # When chardet does not call a file UTF-8 — the real 2026-09-01
            # `hp.obo` has 22 non-ASCII bytes in 10.9 MB and is guessed
            # ISO-8859-1 — pronto wraps the handle in an `EncodedFile` whose
            # collection closes it, and the import scan below then read a
            # closed file: that release failed to load with "seek of closed
            # file", while ASCII fixtures and MONDO passed. A duplicate
            # descriptor is the same open file (so the same bytes, and on
            # Windows the same refusal to be replaced), and pronto closing it
            # closes only the duplicate.
            # It carries the path as its name because pronto reads `.name` as
            # the file's location, and a descriptor's own name is an integer.
            duplicate = io.FileIO(os.dup(handle.fileno()), "rb")
            duplicate.name = str(path)
            with io.BufferedReader(duplicate) as for_pronto:
                # pronto 自動偵測格式 (OBO, OWL, JSON)
                pronto_ont = pronto.Ontology(for_pronto, import_depth=self.IMPORT_DEPTH)
            handle.seek(0)
            listed = _listed_imports(handle)

        # **The union of two readings**, so the loader and the listing cannot
        # disagree. pronto records only `owl:imports` directly under the first
        # `owl:Ontology`; RDF/XML may state the same triple inside an
        # `rdf:Description` about the ontology, which the listing saw and pronto
        # did not — so a file listed as "will be refused" loaded with its import
        # dropped. An import declared with no target (pronto records `None`) is
        # still a declaration and still refused.
        pronto_imports = set(getattr(pronto_ont.metadata, "imports", ()) or ())
        declared_imports = tuple(sorted(
            {str(item) for item in pronto_imports if item}
            | set(listed)
            | ({IMPORT_WITHOUT_TARGET} if None in pronto_imports else set())
        ))
        if declared_imports:
            listed = "\n".join(f"  {item}" for item in declared_imports)
            raise OntologyImportError(
                f"{path} declares {len(declared_imports)} import(s) and this "
                f"project requires a self-contained ontology file:\n{listed}\n"
                "They were not fetched. Resolving them would pull content this "
                "file's digest does not cover, and ignoring them would build a "
                "graph missing whatever they carry while looking complete. "
                "Supply a file that declares no imports."
            )

        # 包裝成我們的 Ontology 類
        ontology = Ontology(pronto_ont, source_path=path, source_digest=digest.hexdigest())

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
        """Load a named ontology from the cache, or fetch it.

        **Selection here is the same selection the build uses**, and it used to
        be a second one. This method opened `<cache_dir>/<name>.obo` and fell
        back to `<name>.owl`, so a cache holding both took the OBO silently —
        exactly the implicit precedence `PLAN_ONTOLOGY_PHASE2.md` §3.1.2
        declines to reinstate, still live on the library path while the CLI
        refused. Two semantics for one question is the parallel pipeline this
        phase exists to avoid, so this defers to `select_ontology_file` over
        the cache directory.

        **And the role is passed.** `load_hpo` knows it wants HPO; not saying
        so left the §3.4 check reachable only from the build script, so
        `load_hpo()` on a file declaring `mondo` returned a MONDO ontology
        under the HPO name while the same file through the CLI was refused.
        """
        from src.ontology.resolver import (
            NoOntologyCandidateError,
            select_ontology_file,
        )

        self._require_supported_version(ontology_name, version)
        cache_key = f"{ontology_name}_{version}"

        # Check memory cache
        if cache_key in self._loaded_ontologies and not force_download:
            logger.info(f"Using cached {ontology_name} ontology")
            return self._loaded_ontologies[cache_key]

        # **The same roots the build searches.** This used to look only in the
        # cache directory, so a library caller on a deployment that configured
        # `paths.ontology_roots` fetched over the network while the configured
        # file sat there — and the fetched copy then made the next build refuse
        # as ambiguous.
        roots = list(self._settings().roots) + [self.cache_dir]

        if force_download:
            ontology = self._fetch_ontology(ontology_name, True, roots)
        else:
            try:
                chosen = select_ontology_file(ontology_name, roots=roots).path
            except NoOntologyCandidateError:
                ontology = self._fetch_ontology(ontology_name, False, roots)
            else:
                # With the role, so this entry point is a gate too.
                ontology = self.load(chosen, expect=ontology_name)

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

    #: A download sits under a name ending in this until it has passed the
    #: imports policy and the role check. Hidden and not `.obo`/`.owl`, so the
    #: resolver never lists a half-verified file as a candidate.
    _STAGED = ".staged"

    def _download_ontology(
        self,
        ontology_name: str,
        force_download: bool,
        roots: Sequence[Path] = (),
    ) -> Tuple[Path, Path]:
        """Fetch an ontology to a **staging** path, through the one guarded path.

        Returns `(staged, final)`: where the bytes are, and the cache name they
        may be published under once verified — see `_fetch_ontology`, which is
        the only caller that should use this.

        **The staging file belongs to this call alone.** It used to be a fixed
        name, `.<name>.<ext>.staged`, shared by every fetch of that ontology
        into that cache. Two builds fetching at once then published each
        other's bytes: one verified MONDO, the other's HPO landed under the
        same staging name, and the first renamed it into `mondo.obo` — having
        verified something else. A name made with `mkstemp` cannot be written
        by another call, and a failed call removes only its own.

        **Every attempt uses `download_ontology`, the OWL fallback included.**
        A fallback with its own fetch is the gate not existing, and the
        fallback is the path a rotted PURL leads to.

        **Nothing already on disk is overwritten before it has been judged.**
        The first version wrote straight to `<cache>/<name>.obo`. When that file
        existed and declared another ontology — the misfiled case §3.4 is about
        — the resolver passed it over for this slot and counted it for the other
        one, and this download then replaced it: the other slot's input was
        destroyed and its provenance recorded the digest of the replacement. So
        a target that exists and is not this ontology's own file refuses, before
        any fetch; and even this ontology's own file is replaced only after the
        new copy has been verified.

        **No stale fallback.** A transfer that fails has nothing valid to fall
        back to: an acceptable file for this slot would have been selected
        without a download, and the files that were not selected were not
        selected for a reason. A refusal says which of the two it was — a
        policy decision the configuration must change, or a transfer that may
        succeed on another attempt — because the remedies differ.
        """
        from urllib.parse import urlsplit

        from src.ontology.download import (
            DestinationPolicy,
            OntologyDestinationRefused,
            OntologyDownloadError,
            download_ontology,
        )
        from src.ontology.resolver import enumerate_candidates, scan_identity

        settings = self._settings()
        urls = settings.urls_for(ontology_name)
        searched = ", ".join(str(root) for root in roots) or "(no roots)"
        manual_hint = self.ONTOLOGY_MANUAL_INSTRUCTIONS.get(ontology_name, "")

        if not urls:
            raise OntologyFetchError(
                f"no {ontology_name} ontology file was found under {searched}, "
                f"and no download source is configured for {ontology_name}, so "
                "nothing was attempted.\n"
                + (f"  Download it manually from {manual_hint}\n" if manual_hint else "")
                + "  and place it under one of the roots above, or name it "
                "explicitly."
            )

        def target_for(url: str) -> Path:
            suffix = ".owl" if urlsplit(url).path.lower().endswith(".owl") else ".obo"
            return self.cache_dir / f"{ontology_name}{suffix}"

        own = {c.path.resolve() for c in enumerate_candidates([self.cache_dir], ontology=ontology_name)}
        for target in {target_for(url) for url in urls}:
            if target.exists() and target.resolve() not in own:
                try:
                    declared = scan_identity(target)["declared_ontology"]
                except Exception:
                    declared = "(unidentifiable)"
                raise OntologyFetchError(
                    f"{target} is already there and is not a {ontology_name} "
                    f"file (it declares {declared!r}). Downloading "
                    f"{ontology_name} would overwrite it, and it may be another "
                    "slot's input. It has been left untouched and nothing was "
                    "fetched: move or rename it, or name the file for each "
                    "ontology explicitly."
                )

        policy = DestinationPolicy(allowed_hosts=settings.allowed_hosts)
        refused: list = []
        failed: list = []
        for url in urls:
            target = target_for(url)
            target.parent.mkdir(parents=True, exist_ok=True)
            handle, name = tempfile.mkstemp(
                dir=target.parent, prefix=f".{target.name}.", suffix=self._STAGED
            )
            os.close(handle)
            staged = Path(name)
            logger.info(f"Downloading {ontology_name} ontology from {url}")
            try:
                return Path(download_ontology(url, staged, policy=policy)), target
            except OntologyDestinationRefused as exc:
                staged.unlink(missing_ok=True)
                refused.append(f"  {url}\n    {exc}")
                logger.error("policy refused %s (%s)", url, exc)
            except OntologyDownloadError as exc:
                staged.unlink(missing_ok=True)
                failed.append(f"  {url}: {exc}")
                logger.warning("could not fetch %s (%s)", url, exc)
            except BaseException:
                staged.unlink(missing_ok=True)
                raise

        if refused and not failed:
            raise OntologyFetchError(
                f"every configured source for {ontology_name} was refused by "
                "the destination policy, so nothing was fetched:\n"
                + "\n".join(refused)
                + "\n\nThis is a configuration decision, not a transient "
                "failure: fix the sources or add the host to "
                "`ontology.allowed_hosts` in configs/deployment.yaml."
            )

        what = (
            "a fresh copy was requested and no source delivered one"
            if force_download else
            f"no {ontology_name} file was found under {searched} and no source "
            "delivered one"
        )
        raise OntologyFetchError(
            f"{what}:\n" + "\n".join(refused + failed)
            + "\n\n  The sources may be unreachable from this machine, or out "
            "of date. Download the file manually"
            + (f" from {manual_hint}" if manual_hint else "")
            + f",\n  place it under one of the roots ({searched}), or name it "
            "explicitly. Nothing already on disk has been used in its place."
        )

    def _fetch_ontology(
        self,
        ontology_name: str,
        force_download: bool,
        roots: Sequence[Path] = (),
    ) -> 'Ontology':
        """Download, **verify, then publish** — in that order.

        The download lands under a staging name this call alone owns; it is
        parsed with the imports policy and the role check; only if both pass
        does it replace the file at its cache name. Publishing first and
        checking afterwards is the defect this project keeps finding in other
        clothes: a fresh copy that turned out to declare an import, or to be
        the wrong ontology, used to have already replaced a working cached
        release by the time it was refused.

        **The digest travels with the verified bytes.** `load` hashes what it
        parses; renaming the file does not change them, so the ontology that
        comes back carries the digest of exactly what was checked, whatever
        another build publishes to the same name a moment later.
        """
        staged, final = self._download_ontology(ontology_name, force_download, roots=roots)
        staged, final = Path(staged), Path(final)
        if staged == final:
            # Nothing to publish: already in place (a caller that fetched
            # straight to the cache name — the tests' stand-ins do).
            return self.load(final, expect=ontology_name)
        try:
            ontology = self.load(staged, expect=ontology_name)
            os.replace(staged, final)
        except BaseException:
            staged.unlink(missing_ok=True)
            raise
        ontology._source_path = final
        return ontology


# =============================================================================
# Ontology Class (wraps pronto.Ontology)
# =============================================================================
from src.ontology.hierarchy import Ontology


# =============================================================================
# Legacy OBO Parser (kept for compatibility with test fixtures)
# =============================================================================
from dataclasses import dataclass, field
from typing import Any, List
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
