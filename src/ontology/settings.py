"""
Where the ontology roots and sources are configured, and what reads them.
=========================================================================
`PLAN_ONTOLOGY_PHASE2.md` §3.2 and §3.6. Two settings move out of code:

- **`ontology_roots`** — the directories the resolver searches. The parent
  plan's §3.1 put them in `configs/deployment.yaml`'s `paths:` block, beside
  `workspaces_root` and `cache_root`, rather than in a new configuration file.
- **`ontology_sources`** — the curated download URLs, which were a class
  constant at `loader.py:45`. A PURL that stops resolving was therefore a code
  change; now it is an edit.

**Why there is a reader here at all.** `src/config/config_validator.py` is an
intentionally empty reserved module — its own docstring says so, and nothing on
any runtime path imports it. So there was nowhere obvious to put this, and the
choice is deliberate: one small reader with a stated contract. It does **not**
revive a global configuration manager, which `docs/CONFIG_AUTHORITY.md` removed
on purpose, and it does not fill in the reserved validator.

**The contract, because a relative path means nothing without a base.**

- The file is `configs/deployment.yaml` at the repository root, unless a caller
  names another.
- **Relative paths in it resolve against the repository root**, not the current
  working directory and not the config file's directory. `paths:` already reads
  `data/workspaces/`, which only means anything under that rule, and a build
  run from a subdirectory has to find the same files as one run from the root.
- A missing file, a missing section or a missing key is **absence, not an
  error**: this configures optional behaviour, and a deployment that has not
  set roots simply has none.
- A key that is *present and malformed* **raises**. Absent and wrong are
  different states and only one of them is a deployment as intended.

Module: src/ontology/settings.py

Dependencies: the standard library and PyYAML.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_ONTOLOGY_SOURCES",
    "OntologySettings",
    "OntologySettingsError",
    "REPO_ROOT",
    "load_ontology_settings",
]

#: The repository root. Relative paths in the configuration resolve against it.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: The committed deployment configuration.
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "deployment.yaml"

#: What `ONTOLOGY_URLS` and `ONTOLOGY_OWL_URLS` held at `loader.py:45` and
#: `:53`. Kept as the fallback for a deployment that configures nothing, so
#: moving the list into configuration does not make an unconfigured install
#: stop working — and kept **here**, where the reader is, rather than back in
#: the loader, so there is one table and not two.
DEFAULT_ONTOLOGY_SOURCES: Dict[str, Tuple[str, ...]] = {
    "hpo": (
        "http://purl.obolibrary.org/obo/hp.obo",
        "http://purl.obolibrary.org/obo/hp.owl",
    ),
    "mondo": (
        "http://purl.obolibrary.org/obo/mondo.obo",
        "http://purl.obolibrary.org/obo/mondo.owl",
    ),
    "go": (
        "http://purl.obolibrary.org/obo/go.obo",
        "http://purl.obolibrary.org/obo/go.owl",
    ),
    "mp": (
        "http://purl.obolibrary.org/obo/mp.obo",
        "http://purl.obolibrary.org/obo/mp.owl",
    ),
}


class OntologySettingsError(ValueError):
    """The configuration is present and does not say what it appears to say."""


@dataclass(frozen=True)
class OntologySettings:
    """What the deployment says about ontologies.

    `sources` maps an ontology name to the URLs to try, **in the order given**
    — which is an order of *attempts*, not of preference between files already
    on disk. Nothing about §3.1's refusal to rank candidates is weakened by a
    list of places to fetch from when there are no candidates at all.
    """

    roots: Tuple[Path, ...] = ()
    sources: Dict[str, Tuple[str, ...]] = None  # type: ignore[assignment]
    allowed_hosts: Tuple[str, ...] = ()
    config_path: Optional[Path] = None

    def __post_init__(self):
        if self.sources is None:
            object.__setattr__(self, "sources", dict(DEFAULT_ONTOLOGY_SOURCES))

    def urls_for(self, ontology: str) -> Tuple[str, ...]:
        return tuple(self.sources.get(str(ontology).lower(), ()))


def _resolve(value: Any, *, key: str) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path)


def _string_list(value: Any, *, key: str) -> Tuple[str, ...]:
    """A YAML list of strings, or a refusal naming the key.

    A bare string is **not** silently wrapped into a one-element list: that
    would make `allowed_hosts: mirror.example` and
    `allowed_hosts: [mirror.example]` mean the same thing, and then a typo that
    produced a string where a list was meant would configure something rather
    than being reported.
    """
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise OntologySettingsError(
            f"{key} must be a list; got {type(value).__name__}. A single entry "
            "is a list of one."
        )
    out: List[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise OntologySettingsError(f"{key} contains {item!r}, which is not a name")
        out.append(item.strip())
    return tuple(out)


def _require_allowed_scheme(url: str, *, key: str) -> None:
    """The scheme rule, asked of the shared module so there is one answer."""
    from urllib.parse import urlsplit

    from src.ontology.download import ALLOWED_SCHEMES

    scheme = (urlsplit(url).scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise OntologySettingsError(
            f"{key} lists {url!r}, whose scheme is {scheme or '(none)'!r}; only "
            f"{', '.join(ALLOWED_SCHEMES)} are permitted as ontology sources."
        )


def load_ontology_settings(config_path: Any = None) -> OntologySettings:
    """Read the ontology settings, or return the defaults.

    Args:
        config_path: the deployment configuration. Defaults to
            `configs/deployment.yaml` at the repository root.

    Raises:
        OntologySettingsError: the file is present and a key in it is malformed.
    """
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH

    if not path.exists():
        logger.debug("no deployment configuration at %s; using defaults", path)
        return OntologySettings(config_path=None)

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - PyYAML is a hard dependency
        raise OntologySettingsError(
            f"{path} exists and PyYAML is not installed, so what it configures "
            "cannot be read — which is not the same as its configuring nothing"
        ) from exc

    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise OntologySettingsError(
            f"{path} is present and could not be read ({type(exc).__name__}: {exc})"
        ) from exc

    if document is None:
        return OntologySettings(config_path=path)
    if not isinstance(document, dict):
        raise OntologySettingsError(f"{path} is not a YAML mapping")

    paths_block = document.get("paths") or {}
    if not isinstance(paths_block, dict):
        raise OntologySettingsError(f"{path}: `paths` is not a mapping")

    roots = tuple(
        _resolve(item, key="paths.ontology_roots")
        for item in _string_list(paths_block.get("ontology_roots"),
                                 key="paths.ontology_roots")
    )

    ontology_block = document.get("ontology") or {}
    if not isinstance(ontology_block, dict):
        raise OntologySettingsError(f"{path}: `ontology` is not a mapping")

    sources = dict(DEFAULT_ONTOLOGY_SOURCES)
    configured = ontology_block.get("sources")
    if configured is not None:
        if not isinstance(configured, dict):
            raise OntologySettingsError(
                f"{path}: `ontology.sources` must be a mapping of ontology name "
                "to a list of URLs"
            )
        for name, urls in configured.items():
            key = f"ontology.sources.{name}"
            entries = _string_list(urls, key=key)
            # **Acceptance 21 — refused when the list is read, naming the
            # entry.** Necessary and not sufficient: §3.5 enforces the same rule
            # at the request boundary because a redirect arrives after this has
            # already passed. Both, because a bad entry should be reported when
            # someone edits the file, not at the first build that needs it.
            for entry in entries:
                _require_allowed_scheme(entry, key=key)
            sources[str(name).lower()] = entries

    allowed_hosts = _string_list(
        ontology_block.get("allowed_hosts"), key="ontology.allowed_hosts"
    )

    return OntologySettings(
        roots=roots,
        sources=sources,
        allowed_hosts=allowed_hosts,
        config_path=path,
    )
