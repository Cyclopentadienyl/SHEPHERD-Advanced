"""
The gene task's functional fixture: a derived, digest-bound artifact.
====================================================================
`docs/working/scorer-retraining/README.md` §8 ("Data in this round"), F9;
`docs/working/PLAN_PROVENANCE_CONTRACT.md` §9.3 ("Reference cohorts and derived
artifacts"); `docs/working/PLAN_TEST_RESULTS.md` §8.3–8.4.

**What it is.** Gene and patients-like-me training rows made from a
workspace's own disease cohorts by one declared rule: every case keeps its
patient id, phenotypes and disease, and gets **one causal gene drawn from its
disease's genes** plus distractor candidates drawn from the genes not
associated with that disease. Labelled "functional fixture", never the paper's
simulated patients. A case whose disease has no gene in the graph is left out
and counted.

**Bound, not written into the workspace.** The six files the split manifest
binds are never rewritten. The fixture lives in its own directory
(`derived/gene_fixture/`), its two sample files beside a record that names the
workspace's split manifest by digest and each file by the digest of the bytes
written, so a reader refuses a fixture from another workspace build or one
whose files changed (the SP table's sidecar pattern, with digests in place of
a build id: the files are small).

**Produced by an explicit CLI** (`scripts/derive_task_fixture.py`), which reads
and verifies the workspace first; this module holds the rule, the publish and
the bound read.

Module: src/kg/task_fixture.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

FIXTURE_DIR = Path("derived") / "gene_fixture"
FIXTURE_RECORD = "task_fixture.json"
FIXTURE_SCHEMA_VERSION = 1
FIXTURE_KIND = "gene_fixture"
FIXTURE_LABEL = "functional fixture"
FIXTURE_RULE = "one-causal-gene-from-the-disease-plus-distractors/v1"
FIXTURE_SPLITS = ("train", "val")
#: Distractor candidates per case (provisional default, README §0.3: not a
#: value of the paper or the source, whose patients carry real candidate lists).
DEFAULT_DISTRACTORS = 19
#: The relation whose edges say which genes a disease has.
GENE_DISEASE_RELATION = ("gene", "gene_associated_with_disease", "disease")


#: The tasks that train on the gene task's functional fixture (README §8): the
#: training process reads it for them, and a Fine-tune start checks it first.
FIXTURE_TASKS = ("gene", "patients_like_me")

#: What a fixture refused for is (`TaskFixtureError.state`): no record, a
#: record bound to another build of the workspace, or one that cannot be read
#: (malformed, of another kind or schema, or files not the ones it names).
FIXTURE_STATES = ("absent", "another_build", "unreadable")


class TaskFixtureError(ValueError):
    """A fixture that is absent, malformed, from another workspace build, or
    whose files are not the ones its record names. `state` says which, as
    `FIXTURE_STATES` names it, so a caller that shows it without the message
    (which may name a server path) can still say why."""

    def __init__(self, message: str, state: str = "unreadable") -> None:
        if state not in FIXTURE_STATES:
            raise ValueError(f"unknown fixture state {state!r}")
        super().__init__(message)
        self.state = state


@dataclass
class GeneFixtureRead:
    #: Split -> the samples read (`SamplesRead`), each checked against its digest.
    samples: Dict[str, Any]
    record: Dict[str, Any]
    #: The identity of the record's bytes.
    identity: Any


def fixture_dir(data_dir: Path) -> Path:
    return Path(data_dir) / FIXTURE_DIR


def disease_genes(edge_index_dict: Mapping[Tuple[str, str, str], Any]) -> Dict[int, List[int]]:
    """Each disease's genes (type-local ids, ascending) from the graph's
    gene–disease relation."""
    edges = edge_index_dict.get(GENE_DISEASE_RELATION)
    genes: Dict[int, set] = {}
    if edges is not None:
        for gene, disease in zip(edges[0].tolist(), edges[1].tolist(), strict=True):
            genes.setdefault(int(disease), set()).add(int(gene))
    return {d: sorted(g) for d, g in genes.items()}


def derive_gene_rows(samples: Sequence[Any], genes_of: Mapping[int, Sequence[int]],
                     num_genes: int, *, seed: int, split: str,
                     distractors: int = DEFAULT_DISTRACTORS
                     ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """The rule, applied to one split's samples in file order: one causal gene
    from the disease's genes, `distractors` candidates from the other genes,
    candidates ascending by gene id. One named random stream per split
    (`derive_stream`), so the train rows do not move when val's do."""
    from src.kg.disease_allocation import derive_stream

    if distractors < 0:
        raise ValueError(f"distractors must be non-negative, got {distractors}")
    rng = derive_stream(seed, f"gene-fixture/{split}")
    rows: List[Dict[str, Any]] = []
    counts = {"cases": len(samples), "derived": 0, "excluded_no_gene_for_disease": 0,
              "short_of_distractors": 0}
    for sample in samples:
        own = list(genes_of.get(int(sample.disease_id), ()))
        if not own:
            counts["excluded_no_gene_for_disease"] += 1
            continue
        causal = rng.choice(own)
        others = sorted(set(range(num_genes)) - set(own))
        if len(others) < distractors:
            counts["short_of_distractors"] += 1
        picked = rng.sample(others, min(distractors, len(others)))
        rows.append({"patient_id": sample.patient_id,
                     "phenotype_ids": list(sample.phenotype_ids),
                     "disease_id": int(sample.disease_id),
                     "causal_gene_ids": [causal],
                     "gene_ids": sorted([causal, *picked])})
        counts["derived"] += 1
    return rows, counts


def publish_gene_fixture(data_dir: Path, rows: Mapping[str, List[Dict[str, Any]]],
                         record: Dict[str, Any], *, replace: bool = False) -> Dict[str, Any]:
    """Write the two sample files, then the record naming their digests. A
    fixture already there is refused unless `replace`: a run that recorded its
    digests would no longer find its data."""
    from src.utils.file_modes import publish_bytes

    out = fixture_dir(data_dir)
    if (out / FIXTURE_RECORD).exists() and not replace:
        raise TaskFixtureError(f"{out / FIXTURE_RECORD} exists; derive with replace to write "
                               "another fixture (runs that recorded the old one will refuse it)")
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for split in FIXTURE_SPLITS:
        name = f"{split}_samples.json"
        data = (json.dumps(rows[split], indent=2) + "\n").encode("utf-8")
        files[split] = {"name": name, "sha256": publish_bytes(out / name, data),
                        "rows": len(rows[split])}
    full = {"schema_version": FIXTURE_SCHEMA_VERSION, "kind": FIXTURE_KIND,
            "label": FIXTURE_LABEL, "rule": FIXTURE_RULE, **record, "files": files}
    publish_bytes(out / FIXTURE_RECORD,
                  (json.dumps(full, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return full


def read_fixture_record(data_dir: Path, *, manifest_sha256: str) -> Tuple[Dict[str, Any], Any]:
    """The fixture record of this workspace build and the identity of its
    bytes: read once, refused unless it is a gene fixture record of this schema
    that names `manifest_sha256` (the caller's one manifest reading). The
    sample files are not read here: `read_gene_fixture` reads each against the
    digest the record names, and a Fine-tune start or a list that only asks
    whether the fixture was derived for this build stops at the record.

    Raises:
        TaskFixtureError: with `state` "absent", "another_build" or "unreadable".
    """
    from src.utils.fingerprint import read_once

    out = fixture_dir(data_dir)
    try:
        raw = read_once(out / FIXTURE_RECORD)
    except FileNotFoundError:
        raise TaskFixtureError(
            f"No gene task fixture at {out}; derive one with scripts/derive_task_fixture.py "
            f"--data-dir {data_dir}", "absent") from None
    except OSError as exc:  # a record this account may not read
        raise TaskFixtureError(f"{out / FIXTURE_RECORD} cannot be read ({type(exc).__name__}: "
                               f"{exc.strerror or exc})") from None
    identity = raw.identity
    try:
        record = json.loads(raw.data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TaskFixtureError(f"{out / FIXTURE_RECORD} is not a readable record: {exc}") from None
    if not isinstance(record, dict) or record.get("kind") != FIXTURE_KIND \
            or record.get("schema_version") != FIXTURE_SCHEMA_VERSION:
        raise TaskFixtureError(f"{out / FIXTURE_RECORD} is not a gene fixture record of schema "
                               f"{FIXTURE_SCHEMA_VERSION}")
    bound = (record.get("workspace") or {}).get("split_manifest")
    if bound != manifest_sha256:
        raise TaskFixtureError(
            f"The gene fixture was derived from another build of this workspace (split "
            f"manifest {str(bound)[:12]}, this one {manifest_sha256[:12]}); derive it again",
            "another_build")
    return record, identity


def read_gene_fixture(data_dir: Path, *, manifest_sha256: str) -> GeneFixtureRead:
    """The fixture of this workspace build: its record (`read_fixture_record`),
    and each sample file read once against the digest the record names."""
    from src.kg.storage.file_storage import SamplesDigestMismatch, read_samples_file

    record, identity = read_fixture_record(data_dir, manifest_sha256=manifest_sha256)
    out = fixture_dir(data_dir)
    samples = {}
    for split in FIXTURE_SPLITS:
        entry = (record.get("files") or {}).get(split) or {}
        try:
            samples[split] = read_samples_file(out / str(entry.get("name")), gene_fields=True,
                                               expected_sha256=str(entry.get("sha256")))
        except SamplesDigestMismatch as exc:
            raise TaskFixtureError(f"The gene fixture's {split} file is not the one its record "
                                   f"names: {exc}") from None
        except FileNotFoundError as exc:
            raise TaskFixtureError(str(exc)) from None
    return GeneFixtureRead(samples, record, identity)


__all__ = [
    "DEFAULT_DISTRACTORS",
    "FIXTURE_DIR",
    "FIXTURE_LABEL",
    "FIXTURE_RECORD",
    "FIXTURE_RULE",
    "FIXTURE_STATES",
    "FIXTURE_TASKS",
    "GeneFixtureRead",
    "TaskFixtureError",
    "derive_gene_rows",
    "disease_genes",
    "fixture_dir",
    "publish_gene_fixture",
    "read_fixture_record",
    "read_gene_fixture",
]
