#!/usr/bin/env python
"""
M5 evidence — how much of the disease space a phenotype reaches.
================================================================
Backlog item 10. M5 was recorded as *"a typical phenotype reaches 71.3% of
diseases within 5 hops (16,845 of 23,640)"*, and BACKLOG §5.2 says of it:

    "A typical phenotype reaches 71.3%" is not reproducible as written, and
    putting it in JSON would not make it so. The selection rule has to be
    operational — which phenotype or phenotypes, chosen how, and whether 71.3%
    is one phenotype's value, a median or a mean.

**This script resolves that by removing the choice rather than making one.** It
reports the reachable-disease count for **every** phenotype in the artifact and
summarises the distribution — count, min, quartiles, median, mean, max. There is
then no "typical" phenotype to have picked, no rule to disagree about, and
whatever 71.3% was can be located in the distribution or corrected against it.
Reporting one selected phenotype would have needed a rule that nothing in the
project justifies, and would have thrown away the spread, which is the part that
says whether "dense" is a property of the graph or of one lucky node.

**Aggregates only.** §5.2 forbids per-phenotype rows, and the distribution is
what the claim is about; the per-phenotype vector is computed and summarised, not
written out.

**Distinct pairs, counted here rather than assumed elsewhere.** An earlier
version took a `bincount` over the disease-typed rows and said the uniqueness it
needed was M7, "enforced at load time". That was false for this path: this script
calls `torch.load` directly and no load-time assertion runs, so a duplicated
`(phenotype, disease)` pair would have inflated a number the report calls a count
of *distinct* diseases.

**Counted in memory that does not grow with the table.** The institutional
artifact is tens of gigabytes and this script runs once, so a working set
proportional to the row count is not a performance question but a risk of losing
the run. Rows are scanned in chunks into a phenotype-by-disease presence matrix,
whose size depends only on the graph — for a graph of ~30k phenotypes and ~24k
diseases that is well under a gigabyte, against a row count near a billion.
Duplicate pairs collapse into the same cell by construction, and the number
collapsed is still reported, so duplication remains a visible fact.

**The hop bound comes from the sidecar, not from the data.**
`scripts/compute_shortest_paths.py` writes the **configured** `max_hops` to
`<artifact>.meta.json`. The largest distance present is not the same number: an
artifact built to 5 hops in which no pair happens to sit at exactly 5 would
otherwise be reported as a 4-hop artifact, and every percentage in it read against
the wrong bound.

Usage:
    python scripts/audit_sp_reachability.py \\
        --artifact data/workspaces/<ws>/shortest_paths.pt \\
        --data-dir data/workspaces/<ws> \\
        --output docs/working/EVIDENCE_M5_sp_reachability.json

Module: scripts/audit_sp_reachability.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Shared with the other evidence scripts rather than restated here: all three
# reports are read together, and a claim spelled differently in each cannot be
# compared across them.
from src.utils.provenance import DEPLOYMENT_RELATIONSHIPS, UNSTATED_RELATIONSHIP  # noqa: E402

logger = logging.getLogger(__name__)

#: `scripts/compute_shortest_paths.py` writes `target_type` as 0 = gene, 1 = disease.
DISEASE_TARGET_TYPE = 1

REQUIRED_COLUMNS = ("phenotype_idx", "target_idx", "target_type", "distance")

#: `scripts/compute_shortest_paths.py` validates `max_hops` into this range before
#: it writes anything. A sidecar outside it did not come from that producer.
PRODUCER_HOP_RANGE = (1, 127)

#: Integer fields the sidecar must carry. Checked against history rather than
#: assumed: every version of `save_shortest_paths` since the artifact format was
#: introduced (`eadb839`) has written all four, so requiring them cannot refuse an
#: artifact the project itself produced.
REQUIRED_SIDECAR_INTS = ("max_hops", "num_pairs", "num_phenotypes", "num_diseases")

#: Rows per chunk of the scan. Sized so the transient int64 index copies stay in
#: the tens of megabytes; nothing about the result depends on it, and the tests
#: run a chunk size of 1 over a table of several rows to prove that.
SCAN_CHUNK_ROWS = 4_000_000

#: Ceiling on the presence matrix, in cells — one byte each, so 2**32 is 4 GiB.
#: The institutional graph needs roughly 0.7 GiB of it. Beyond this the one-pass
#: counter is the wrong tool and says so rather than trying and being killed.
MAX_PRESENCE_CELLS = 2 ** 32

#: Budget for the row reduction at the end of the scan, in bytes.
#:
#: **Measured, not guessed.** `presence.sum(dim=1)` on a bool matrix allocates a
#: full int64 copy of it — eight bytes per cell, verified as +551 MiB over a
#: 69 MiB matrix, and `dtype=torch.int64` does not avoid it. On a graph of ~30k
#: phenotypes that intermediate would be several gigabytes, dwarfing the matrix
#: the ceiling above is written to bound. Reducing in bands of rows holds it here
#: instead; the same measurement puts the banded cost at +49 MiB.
REDUCTION_BUDGET_BYTES = 64 * 2 ** 20


def summarise_distribution(counts: Any, denominator: int) -> Dict[str, Any]:
    """The spread over every phenotype, as counts and as fractions.

    Both, because a fraction alone loses the denominator and the denominator is
    half the claim — 71% of 23,640 and 71% of 12 are not the same finding.
    """
    import torch

    if counts.numel() == 0:
        return {"n_phenotypes": 0}

    as_float = counts.to(torch.float64)
    quantiles = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
    q = torch.quantile(as_float, quantiles).tolist()

    def both(value: float) -> Dict[str, float]:
        return {
            "diseases": value,
            "fraction_of_all_diseases": value / denominator if denominator else None,
        }

    return {
        "n_phenotypes": int(counts.numel()),
        "denominator_diseases": denominator,
        "min": both(q[0]),
        "q1": both(q[1]),
        "median": both(q[2]),
        "q3": both(q[3]),
        "max": both(q[4]),
        "mean": both(float(as_float.mean())),
    }


def _read_sidecar(artifact: Path) -> Dict[str, int]:
    """The producer's own record of what it built, or a refusal.

    `<artifact>.meta.json` carries the **configured** `max_hops` and the node
    counts the graph had when the table was computed. Both are claims this script
    cannot reconstruct from the tensors, and a run without them would have to
    guess — so a missing or malformed sidecar is refused rather than worked around.

    **The digest identifies the sidecar; it does not bind it to the tensors.** Two
    files sitting next to each other prove nothing about each other, so every
    integer here is checked for type and range, and `build_report` then binds them
    to what the artifact and the workspace actually contain. A sidecar that
    survives all of that is describing this table.
    """
    import json as _json

    sidecar = artifact.with_suffix(".meta.json")
    if not sidecar.exists():
        raise SystemExit(
            f"{sidecar} is missing. It carries the configured hop bound, which the "
            "tensors do not: the largest distance present is a property of the data, "
            "not of what the artifact was built to. Without it every percentage here "
            "would be read against a bound nobody chose."
        )
    try:
        meta = _json.loads(sidecar.read_text())
    except Exception as exc:  # noqa: BLE001 - a malformed sidecar is a refusal
        raise SystemExit(f"{sidecar} is not readable JSON: {type(exc).__name__}") from exc
    if not isinstance(meta, dict):
        raise SystemExit(f"{sidecar} is not a JSON object; it does not describe an artifact")

    validated: Dict[str, int] = {}
    for key in REQUIRED_SIDECAR_INTS:
        value = meta.get(key)
        # `bool` is an `int` in Python and `true` is valid JSON, so it is excluded
        # by name rather than trusted to fail the range checks below.
        if not isinstance(value, int) or isinstance(value, bool):
            raise SystemExit(
                f"{sidecar} carries {key}={value!r}, which is not an integer. Every "
                "version of the producer writes all of "
                f"{list(REQUIRED_SIDECAR_INTS)}; a file without them is not its output."
            )
        validated[key] = value

    low, high = PRODUCER_HOP_RANGE
    if not low <= validated["max_hops"] <= high:
        raise SystemExit(
            f"{sidecar} declares max_hops={validated['max_hops']}, outside the "
            f"[{low}, {high}] the producer validates before writing. This sidecar did "
            "not come from it."
        )
    for key in ("num_pairs", "num_phenotypes", "num_diseases"):
        if validated[key] < 0:
            raise SystemExit(f"{sidecar} declares a negative {key}={validated[key]}")
    return validated


def _validate_columns(table: Any, artifact: Path) -> int:
    """Shape and dtype, **before** any value is read as an index.

    A float column silently truncates on conversion — `0.5` becomes phenotype `0`
    and is counted — so the refusal has to come before the cast rather than after
    it. Ragged columns are refused for the same reason: the scan pairs row *i* of
    one column with row *i* of another, and columns of different lengths make that
    pairing arbitrary rather than wrong in a visible way.

    Returns the row count the four columns agree on.
    """
    import torch

    missing = [c for c in REQUIRED_COLUMNS if c not in table]
    if missing:
        raise SystemExit(
            f"{artifact} is missing {missing}. This reads the format "
            "`scripts/compute_shortest_paths.py` writes; a different one would be "
            "measured wrongly rather than refused."
        )

    lengths = {}
    for name in REQUIRED_COLUMNS:
        column = table[name]
        if not isinstance(column, torch.Tensor):
            raise SystemExit(f"{artifact}: {name} is {type(column).__name__}, not a tensor")
        if column.dim() != 1:
            raise SystemExit(
                f"{artifact}: {name} has shape {tuple(column.shape)}; the producer "
                "writes parallel one-dimensional columns"
            )
        if column.is_floating_point() or column.is_complex() or column.dtype is torch.bool:
            raise SystemExit(
                f"{artifact}: {name} has dtype {column.dtype}. Converting it to an "
                "index would truncate — 0.5 would be counted as 0 — so it is refused "
                "rather than converted."
            )
        lengths[name] = int(column.numel())

    if len(set(lengths.values())) != 1:
        raise SystemExit(
            f"{artifact}: the columns disagree on length ({lengths}). They are read "
            "as parallel rows, and that pairing means nothing here."
        )
    return next(iter(lengths.values()))


def scan_table(table: Any, n_rows: int, n_phenotypes: int, n_diseases: int,
               chunk_rows: int = SCAN_CHUNK_ROWS) -> Dict[str, Any]:
    """One pass over the table, in memory bounded by the graph rather than the rows.

    Every per-row quantity M5 needs is collected here — the reachable set, the
    disease-row count and the largest distance — because each extra pass over a
    table this size costs minutes and buys nothing.

    **The presence matrix is what makes duplicates free.** Writing `True` into
    `(phenotype, disease)` twice leaves one cell set, so distinctness is a property
    of the container rather than of a sort, and the working set stops depending on
    how many rows there are. The alternative this replaced built an int64 key per
    row and sorted it, which on the institutional artifact would have been several
    copies of several gigabytes each, discovered during a one-shot run.

    Index ranges are checked here rather than up front, for the same reason: a
    bounds check over the whole column needs the whole column.
    """
    import torch

    cells = n_phenotypes * n_diseases
    if cells > MAX_PRESENCE_CELLS:
        raise SystemExit(
            f"a presence matrix over {n_phenotypes} phenotypes and {n_diseases} "
            f"diseases needs {cells} bytes, beyond the {MAX_PRESENCE_CELLS} this "
            "one-pass counter allocates. The graph is larger than this script was "
            "built for, and it says so rather than being killed part-way through."
        )
    if chunk_rows < 1:
        raise ValueError(f"chunk_rows must be positive; got {chunk_rows}")

    presence = torch.zeros((n_phenotypes, n_diseases), dtype=torch.bool)
    n_disease_rows = 0
    observed_hops = None

    for start in range(0, n_rows, chunk_rows):
        stop = min(start + chunk_rows, n_rows)
        phenotype_chunk = table["phenotype_idx"][start:stop]
        _refuse_out_of_range("phenotype_idx", phenotype_chunk, n_phenotypes)

        distance_chunk = table["distance"][start:stop]
        if distance_chunk.numel():
            chunk_max = int(distance_chunk.max().item())
            observed_hops = chunk_max if observed_hops is None else max(observed_hops, chunk_max)

        is_disease = table["target_type"][start:stop] == DISEASE_TARGET_TYPE
        # Gene rows index the gene universe, so their target_idx is deliberately
        # not checked against the disease count and deliberately not counted.
        target_chunk = table["target_idx"][start:stop][is_disease]
        _refuse_out_of_range("target_idx", target_chunk, n_diseases)

        n_disease_rows += int(is_disease.sum().item())
        if target_chunk.numel():
            presence[phenotype_chunk[is_disease].long(), target_chunk.long()] = True

    counts = _row_counts(presence, n_diseases)
    return {
        "counts": counts,
        "n_disease_rows": n_disease_rows,
        "n_distinct": int(counts.sum().item()),
        "observed_hops": observed_hops,
    }


def _row_counts(presence: Any, n_diseases: int) -> Any:
    """Set cells per row, without materialising an int64 copy of the matrix.

    A whole-matrix `sum(dim=1)` allocates eight bytes per cell — measured at
    +551 MiB over a 69 MiB bool matrix, and unchanged by passing `dtype`. That is
    the largest allocation in this script and it grows with the graph, so it is
    the one thing `MAX_PRESENCE_CELLS` was written to bound and the one thing it
    does not see. Reducing a band of rows at a time keeps the intermediate at
    `REDUCTION_BUDGET_BYTES` regardless of how large the graph is.
    """
    import torch

    band = max(1, REDUCTION_BUDGET_BYTES // max(1, n_diseases * 8))
    counts = torch.empty(presence.shape[0], dtype=torch.int64)
    for start in range(0, presence.shape[0], band):
        stop = min(start + band, presence.shape[0])
        counts[start:stop] = presence[start:stop].sum(dim=1, dtype=torch.int64)
    return counts


def _refuse_out_of_range(name: str, values: Any, ceiling: int) -> None:
    """An index outside its universe is an artifact from another graph, and counting
    it would land inside a plausible percentage rather than raise."""
    if values.numel() == 0:
        return
    low, high = int(values.min().item()), int(values.max().item())
    if low < 0 or high >= ceiling:
        raise SystemExit(
            f"{name} spans [{low}, {high}] against a universe of {ceiling}. "
            "The artifact and the workspace describe different graphs."
        )


def _load_table(artifact: Path) -> Any:
    """The artifact, memory-mapped where the format allows it.

    `torch.save`'s zipfile format lets `mmap=True` leave the columns on disk and
    page them in as the scan reaches them, which keeps a tens-of-gigabytes artifact
    out of resident memory. An artifact written some other way simply loads the
    ordinary way — the numbers are identical either path, so this is allowed to
    fall back rather than refuse.
    """
    import torch

    try:
        table = torch.load(artifact, map_location="cpu", weights_only=True, mmap=True)
        logger.info("read %s memory-mapped", artifact.name)
        return table
    except (RuntimeError, TypeError, ValueError) as exc:
        logger.info("%s is not mappable (%s); reading it into memory",
                    artifact.name, type(exc).__name__)
        return torch.load(artifact, map_location="cpu", weights_only=True)


def build_report(artifact: Path, data_dir: Path, relationship: str) -> Dict[str, Any]:
    import json as _json

    from src.utils.fingerprint import file_sha256

    table = _load_table(artifact)
    n_rows = _validate_columns(table, artifact)
    meta = _read_sidecar(artifact)
    configured_hops = meta["max_hops"]

    num_nodes = _json.loads((data_dir / "num_nodes.json").read_text())
    n_diseases = int(num_nodes["disease"])
    n_phenotypes = int(num_nodes["phenotype"])

    # Both universes must be non-empty. M5 is a statement of the form "a phenotype
    # reaches X% of diseases", and over an empty universe every such statement is
    # vacuously true — an evidence file reporting median 0.0 against a null fraction
    # would be read as a finding about reachability rather than about the workspace.
    for label, size in (("disease", n_diseases), ("phenotype", n_phenotypes)):
        if size <= 0:
            raise SystemExit(
                f"num_nodes.json reports {size} {label} nodes. There is no distribution "
                "over an empty universe, and a percentage against one is not evidence."
            )

    # **Binding the sidecar to the tensors and to the workspace.** The digest names
    # which sidecar was read; these three checks are what make it this artifact's.
    # `num_pairs` is the one that ties it to the tensors — the node counts would
    # match any table built from the same graph, but the row count would not.
    for key, observed, subject in (
        ("num_pairs", n_rows, "the table's row count"),
        ("num_phenotypes", n_phenotypes, "num_nodes.json"),
        ("num_diseases", n_diseases, "num_nodes.json"),
    ):
        if meta[key] != observed:
            raise SystemExit(
                f"the sidecar declares {key}={meta[key]} while {subject} has {observed}. "
                "The sidecar does not describe this artifact, and no number below "
                "would be about what it claims to be about."
            )

    scan = scan_table(table, n_rows, n_phenotypes, n_diseases)
    counts = scan["counts"]
    n_disease_rows = scan["n_disease_rows"]
    n_distinct = scan["n_distinct"]

    # Over the whole table, gene targets included: this is the artifact's own bound,
    # cross-checked against the sidecar, not a statistic about the disease rows.
    observed_hops = scan["observed_hops"]
    if observed_hops is not None and observed_hops > configured_hops:
        raise SystemExit(
            f"the table holds a distance of {observed_hops} while the sidecar "
            f"configured {configured_hops}. The artifact does not match its own record."
        )

    n_with_any = int((counts > 0).sum().item())

    return {
        "fact": "M5",
        "what_this_shows": (
            "how many distinct diseases each phenotype reaches within the artifact's "
            "configured hop bound, summarised over every phenotype in the graph"
        ),
        "selection_rule": (
            "none — every phenotype in the graph is counted, including those that "
            "reach no disease and therefore appear nowhere in the artifact, and the "
            "distribution is reported. No phenotype is selected as typical, because "
            "no rule for selecting one is justified and the spread is what says "
            "whether reachability is dense in the graph or in one node."
        ),
        "artifact_digest": file_sha256(artifact),
        "sidecar_digest": file_sha256(artifact.with_suffix(".meta.json")),
        "hop_bound_configured": configured_hops,
        "hop_bound_observed": observed_hops,
        "reachable_diseases_per_phenotype": summarise_distribution(counts, n_diseases),
        "rows": {
            "total": n_rows,
            "disease_targets": n_disease_rows,
            "distinct_phenotype_disease_pairs": n_distinct,
            "duplicate_pairs_collapsed": n_disease_rows - n_distinct,
        },
        "phenotype_coverage": {
            "in_graph": n_phenotypes,
            "with_at_least_one_reachable_disease": n_with_any,
            "reaching_none": n_phenotypes - n_with_any,
        },
        "deployment_relationship": relationship,
        "excluded_by_design": ["per-phenotype rows (only the distribution is recorded)"],
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M5 shortest-path reachability evidence")
    parser.add_argument("--artifact", type=Path, required=True,
                        help="shortest_paths.pt")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Workspace holding num_nodes.json, which carries the "
                             "disease denominator. The artifact knows only which pairs "
                             "are reachable, never how many diseases exist.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing --output. Off by default.")
    parser.add_argument("--deployment-relationship", default=UNSTATED_RELATIONSHIP,
                        choices=DEPLOYMENT_RELATIONSHIPS,
                        help="How this machine relates to the deployment. A bounded "
                             "vocabulary rather than free text: the schema forbids "
                             "operator and host names, and cannot then accept an "
                             "arbitrary string. Unverified by design.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)

    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"{args.output} exists. Pass --overwrite or write elsewhere.")

    report = build_report(args.artifact, args.data_dir, args.deployment_relationship)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))

    spread = report["reachable_diseases_per_phenotype"]
    logger.info(
        "%s phenotypes, median %s of %s diseases -> %s",
        spread.get("n_phenotypes"),
        spread.get("median", {}).get("diseases"),
        spread.get("denominator_diseases"), args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
