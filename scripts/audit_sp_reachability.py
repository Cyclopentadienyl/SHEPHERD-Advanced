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
of *distinct* diseases. Pairs are now reduced with `torch.unique` over a composite
key before counting, and the number collapsed is reported — so duplication becomes
a visible fact rather than a silent overcount.

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

# The vocabulary is shared with the other evidence scripts rather than restated
# here: an institutional reader joins these reports by machine, and that join
# breaks the moment two scripts spell the same claim differently.
from src.utils.provenance import DEPLOYMENT_RELATIONSHIPS, UNSTATED_RELATIONSHIP  # noqa: E402

logger = logging.getLogger(__name__)

#: `scripts/compute_shortest_paths.py` writes `target_type` as 0 = gene, 1 = disease.
DISEASE_TARGET_TYPE = 1

REQUIRED_COLUMNS = ("phenotype_idx", "target_idx", "target_type", "distance")


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


def _read_sidecar(artifact: Path) -> Dict[str, Any]:
    """The producer's own record of what it built, or a refusal.

    `<artifact>.meta.json` carries the **configured** `max_hops` and the node
    counts the graph had when the table was computed. Both are claims this script
    cannot reconstruct from the tensors, and a run without them would have to
    guess — so a missing or malformed sidecar is refused rather than worked around.
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
    if not isinstance(meta, dict) or "max_hops" not in meta:
        raise SystemExit(f"{sidecar} carries no max_hops; it does not describe this artifact")
    return meta


def _distinct_pair_counts(phenotypes: Any, targets: Any, n_phenotypes: int, n_diseases: int):
    """Reachable **diseases** per phenotype, with duplicate pairs collapsed.

    A composite key `phenotype * n_diseases + target` makes a pair one integer, so
    `torch.unique` reduces the table to distinct pairs in one pass. The domain is
    checked in Python **before** any int64 tensor exists, because an overflow
    inside the tensor would wrap silently and produce a plausible wrong answer —
    the same guard `scripts/sp_index_prototypes.py` uses for the same reason.

    Returns `(counts, n_rows, n_distinct)` so the number collapsed is reportable.
    """
    import torch

    largest = (n_phenotypes - 1) * n_diseases + (n_diseases - 1) if n_phenotypes else 0
    if largest >= 2 ** 63 - 1:
        raise SystemExit(
            f"a composite key over {n_phenotypes} phenotypes and {n_diseases} diseases "
            "would overflow int64; this script cannot count distinct pairs safely here"
        )

    keys = phenotypes.to(torch.int64) * n_diseases + targets.to(torch.int64)
    distinct = torch.unique(keys)
    counts = torch.bincount(distinct // n_diseases, minlength=n_phenotypes)
    return counts, int(keys.numel()), int(distinct.numel())


def build_report(artifact: Path, data_dir: Path, relationship: str) -> Dict[str, Any]:
    import json as _json

    import torch

    from src.utils.fingerprint import file_sha256

    table = torch.load(artifact, map_location="cpu", weights_only=True)
    missing = [c for c in REQUIRED_COLUMNS if c not in table]
    if missing:
        raise SystemExit(
            f"{artifact} is missing {missing}. This reads the format "
            "`scripts/compute_shortest_paths.py` writes; a different one would be "
            "measured wrongly rather than refused."
        )

    meta = _read_sidecar(artifact)
    configured_hops = int(meta["max_hops"])

    num_nodes = _json.loads((data_dir / "num_nodes.json").read_text())
    n_diseases = int(num_nodes["disease"])
    n_phenotypes = int(num_nodes["phenotype"])

    # Both universes must be non-empty. M5 is a statement of the form "a phenotype
    # reaches X% of diseases", and over an empty universe every such statement is
    # vacuously true — an evidence file reporting median 0.0 against a null fraction
    # would be read as a finding about reachability rather than about the workspace.
    # This also keeps the composite key below away from a zero modulus.
    for label, size in (("disease", n_diseases), ("phenotype", n_phenotypes)):
        if size <= 0:
            raise SystemExit(
                f"num_nodes.json reports {size} {label} nodes. There is no distribution "
                "over an empty universe, and a percentage against one is not evidence."
            )

    # The sidecar and the workspace must describe the same graph. They are two
    # independent records of it, so a disagreement means the artifact was built
    # from something other than what is being audited — and every percentage below
    # would be a real number over the wrong denominator.
    for key, observed in (("num_diseases", n_diseases), ("num_phenotypes", n_phenotypes)):
        declared = meta.get(key)
        if declared is not None and int(declared) != observed:
            raise SystemExit(
                f"the artifact's sidecar declares {key}={declared} while num_nodes.json "
                f"has {observed}. They describe different graphs, and no denominator "
                "here would be right."
            )

    is_disease = table["target_type"] == DISEASE_TARGET_TYPE
    phenotypes = table["phenotype_idx"][is_disease]
    targets = table["target_idx"][is_disease]
    distances = table["distance"]

    # Both axes, both bounds. An index outside its universe is an artifact from
    # another graph, and counting it would land inside a plausible percentage.
    for name, values, ceiling in (
        ("phenotype_idx", phenotypes, n_phenotypes),
        ("target_idx", targets, n_diseases),
    ):
        if values.numel() == 0:
            continue
        low, high = int(values.min().item()), int(values.max().item())
        if low < 0 or high >= ceiling:
            raise SystemExit(
                f"{name} spans [{low}, {high}] against a universe of {ceiling}. "
                "The artifact and the workspace describe different graphs."
            )

    # Over the whole table, gene targets included: this is the artifact's own bound,
    # cross-checked against the sidecar, not a statistic about the disease rows.
    observed_hops = int(distances.max().item()) if distances.numel() else None
    if observed_hops is not None and observed_hops > configured_hops:
        raise SystemExit(
            f"the table holds a distance of {observed_hops} while the sidecar "
            f"configured {configured_hops}. The artifact does not match its own record."
        )

    counts, n_rows, n_distinct = _distinct_pair_counts(
        phenotypes, targets, n_phenotypes, n_diseases)
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
            "total": int(table["phenotype_idx"].numel()),
            "disease_targets": n_rows,
            "distinct_phenotype_disease_pairs": n_distinct,
            "duplicate_pairs_collapsed": n_rows - n_distinct,
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
