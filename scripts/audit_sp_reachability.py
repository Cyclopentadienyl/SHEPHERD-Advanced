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

**One recorded assumption.** The per-phenotype count is a `bincount` over the
disease-typed rows, which is the reachable-disease count only if the artifact has
no duplicate `(phenotype, target)` pairs. That is M7, and it is enforced at load
time rather than re-derived here — checking it would need a key column over the
whole table, and the artifact is gigabytes.

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


def build_report(artifact: Path, data_dir: Path, platform_note: Optional[str]) -> Dict[str, Any]:
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

    num_nodes = _json.loads((data_dir / "num_nodes.json").read_text())
    denominator = int(num_nodes["disease"])

    is_disease = table["target_type"] == DISEASE_TARGET_TYPE
    phenotypes = table["phenotype_idx"]

    # **Over every phenotype in the graph, not every phenotype in the table.**
    # A phenotype that reaches no disease has no rows at all, so counting only
    # what appears would silently drop exactly the zeroes and report a
    # distribution shifted upward — the bias a claim like "a typical phenotype
    # reaches 71.3%" is most vulnerable to. Both figures are recorded below so
    # the gap between them is visible rather than assumed to be nil.
    n_phenotypes_total = int(num_nodes["phenotype"])
    counts = torch.bincount(phenotypes[is_disease], minlength=n_phenotypes_total)
    if counts.numel() > n_phenotypes_total:
        raise SystemExit(
            f"the artifact references phenotype index {counts.numel() - 1}, beyond the "
            f"{n_phenotypes_total} in num_nodes.json. The artifact and the workspace "
            "describe different graphs and no denominator here would be right."
        )
    n_phenotypes_in_table = int(torch.unique(phenotypes[is_disease]).numel())

    hop_bound = int(table["distance"].max().item()) if table["distance"].numel() else None

    return {
        "fact": "M5",
        "what_this_shows": (
            "how many distinct diseases each phenotype reaches within the artifact's "
            "hop bound, summarised over every phenotype present"
        ),
        "selection_rule": (
            "none — every phenotype in the graph is counted, including those that "
            "reach no disease and therefore appear nowhere in the artifact, and the "
            "distribution is reported. No phenotype is selected as typical, because "
            "no rule for selecting one is justified and the spread is what says "
            "whether reachability is dense in the graph or in one node."
        ),
        "artifact_digest": file_sha256(artifact),
        "hop_bound_observed": hop_bound,
        "reachable_diseases_per_phenotype": summarise_distribution(counts, denominator),
        "rows": {
            "total": int(table["phenotype_idx"].numel()),
            "disease_targets": int(is_disease.sum().item()),
        },
        "phenotype_coverage": {
            "in_graph": n_phenotypes_total,
            "with_at_least_one_reachable_disease": n_phenotypes_in_table,
            "reaching_none": n_phenotypes_total - n_phenotypes_in_table,
        },
        "assumptions": [
            "no duplicate (phenotype, target) pairs — M7, enforced at load time "
            "rather than re-derived here",
        ],
        "platform_note": platform_note,
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
    parser.add_argument("--platform-note", default=None,
                        help="An operator's statement about this machine's relationship "
                             "to the deployment. Recorded verbatim and not verified.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)

    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"{args.output} exists. Pass --overwrite or write elsewhere.")

    report = build_report(args.artifact, args.data_dir, args.platform_note)
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
