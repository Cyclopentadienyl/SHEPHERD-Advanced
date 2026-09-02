#!/usr/bin/env python
"""
Split feasibility audit — what cutting the disease universe would cost.
======================================================================
Backlog item 11, and the one engineering step cleared to proceed by
``docs/working/EVALUATION_COHORTS.md`` §6.8.

**It decides nothing.** It selects no withheld fraction, modifies no generator
behaviour, writes no allocation and builds no UI. Its output is an input to the
institutional questions in §5.1 — how many diseases to withhold from patient
supervision, and whether the draw should be uniform or stratified — which are
currently unanswerable because nobody has measured what either would cost.

**Counts, bands and closed-form probabilities only.** BACKLOG §5.2 forbids
patient ids, sample ids, per-disease lists, absolute paths, host names and
operator names in evidence artifacts, and nothing here needs them: every figure
is a set size, a band population, or a probability derived from the two.

Two things this script must get right, both learned the hard way:

1. **Eligibility comes from the generator's own helper**
   (``build_eligible_disease_profiles``). An audit measuring a different disease
   universe from the one generation partitions would report the cost of a split
   nobody runs.
2. **The withheld count is an integer ``W``, fixed before anything else.**
   ``f · N`` is generally not an integer, and every quantity here is defined
   against ``W``, never against ``f`` directly.

Usage:
    python scripts/audit_split_feasibility.py \\
        --kg-path data/workspaces/<workspace>/kg.json \\
        --train-budget 100000 --val-budget 15000 \\
        --output docs/working/EVIDENCE_split_feasibility.json

Module: scripts/audit_split_feasibility.py
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.provenance import DEPLOYMENT_RELATIONSHIPS, UNSTATED_RELATIONSHIP  # noqa: E402

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

#: Withheld fractions the sensitivity curve is reported at. 0.15 is the upstream
#: value (EVALUATION_COHORTS §1.6); the rest bound it on both sides.
DEFAULT_FRACTIONS: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)

#: Samples-per-disease values the coverage-budget column is reported at. Named
#: `samples_per_disease` and never `k`: `k` is the retained-phenotype count in
#: C(P, k) and means nothing else anywhere in this file.
DEFAULT_SAMPLES_PER_DISEASE: Tuple[int, ...] = (1, 5, 10, 20)

#: Band lower bounds. A band runs from its own bound up to the next one, and the
#: last runs to infinity. Ordering everywhere is by ascending lower bound, which
#: is what makes the largest-remainder tie-break deterministic.
PHENOTYPE_COUNT_BANDS: Tuple[int, ...] = (0, 2, 4, 6, 11, 21, 51)
GENE_COUNT_BANDS: Tuple[int, ...] = (0, 1, 2, 4, 11)
PROFILE_SUPPORT_BANDS: Tuple[int, ...] = (0, 1, 2, 5, 11, 26, 51)
CAPACITY_BANDS: Tuple[int, ...] = (1, 2, 6, 21, 101, 1001)

#: Where a value that could not be computed is placed. Sorted **after** every
#: numeric band, ahead of any label tie-break, so bucket order is total.
MISSING_LABEL = "missing"


# --------------------------------------------------------------------------
# The input boundary
# --------------------------------------------------------------------------


class AuditSettings(NamedTuple):
    """Validated, normalised audit inputs — the single source for every caller.

    Returned frozen so nothing downstream re-derives a value differently from
    what the report echoes under ``assumptions``.
    """

    min_phenotypes: int
    max_phenotypes: int
    phenotype_drop_rate: float
    train_budget: int
    val_budget: int
    fractions: Tuple[float, ...]
    samples_per_disease: Tuple[int, ...]


def _integer(name: str, value: Any, minimum: int) -> int:
    """An ``int`` that is not a ``bool``. ``True`` would otherwise pass as 1."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _dedupe(values: Sequence[Any]) -> List[Any]:
    """Preserve first-seen order. Repeats would only duplicate report rows.

    **Only ever called on already-validated, canonical values.** Deduplicating
    first would hide an illegal element behind a legal one, because Python's
    equality and hashing make ``True == 1 == 1.0``: ``[1, True]`` collapsed to
    ``[1]`` and the ``True`` was never checked, while ``[True, 1]`` collapsed to
    ``[True]`` and was correctly refused. The same multiset, opposite outcomes,
    decided by input order — which is how the ordering bug was found.
    """
    seen, unique = set(), []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def _fraction(value: Any) -> float:
    """A finite fraction strictly inside ``(0, 1)``, canonicalised to ``float``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"fraction must be a number, got {value!r}")
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(
            f"fraction must be finite and strictly between 0 and 1, got {value}"
        )
    return float(value)


def validate_relationship(relationship: Any) -> str:
    """Membership of the bounded vocabulary, checked at the report boundary too.

    ``argparse`` constrains the flag, but a programmatic caller reaches
    ``build_report`` directly, and this string is written verbatim into the
    evidence artifact. An unchecked value is a hole in the schema that forbids
    host and operator names — the check belongs where the artifact is produced,
    not only where the command line is parsed.
    """
    if relationship not in DEPLOYMENT_RELATIONSHIPS:
        raise ValueError(
            f"deployment_relationship must be one of {DEPLOYMENT_RELATIONSHIPS}, "
            f"got {relationship!r}"
        )
    return relationship


def validate_settings(
    *,
    min_phenotypes: int,
    max_phenotypes: int,
    phenotype_drop_rate: float,
    train_budget: int,
    val_budget: int,
    fractions: Sequence[float],
    samples_per_disease: Sequence[int],
) -> AuditSettings:
    """Refuse inputs that would produce plausible-looking but meaningless evidence.

    **This runs before the knowledge graph is loaded and before anything is
    written.** An evidence artifact that silently clamped a fraction of ``-1`` to
    a legal value, or reported a budget as sufficient because
    ``samples_per_disease`` was zero, would be worse than no artifact: the
    numbers would look reportable. Every domain below is checked here rather than
    at the argparse layer, so the same guarantees hold for an API caller.
    """
    min_phenotypes = _integer("min_phenotypes", min_phenotypes, 1)
    max_phenotypes = _integer("max_phenotypes", max_phenotypes, min_phenotypes)
    train_budget = _integer("train_budget", train_budget, 0)
    val_budget = _integer("val_budget", val_budget, 0)

    if isinstance(phenotype_drop_rate, bool) or not isinstance(
        phenotype_drop_rate, (int, float)
    ):
        raise ValueError(f"phenotype_drop_rate must be a number, got {phenotype_drop_rate!r}")
    if not math.isfinite(phenotype_drop_rate) or not 0.0 <= phenotype_drop_rate <= 1.0:
        raise ValueError(
            f"phenotype_drop_rate must be finite and in [0, 1], got {phenotype_drop_rate}"
        )

    # **Validate every original element, then canonicalise, then deduplicate.**
    # The order is load-bearing; see `_dedupe`.
    if not list(fractions):
        raise ValueError("at least one fraction is required")
    checked_fractions = [_fraction(value) for value in fractions]

    if not list(samples_per_disease):
        raise ValueError("at least one samples_per_disease value is required")
    checked_per_disease = [
        _integer("samples_per_disease", value, 1) for value in samples_per_disease
    ]

    return AuditSettings(
        min_phenotypes=min_phenotypes,
        max_phenotypes=max_phenotypes,
        phenotype_drop_rate=float(phenotype_drop_rate),
        train_budget=train_budget,
        val_budget=val_budget,
        fractions=tuple(_dedupe(checked_fractions)),
        samples_per_disease=tuple(_dedupe(checked_per_disease)),
    )


# --------------------------------------------------------------------------
# The arithmetic
# --------------------------------------------------------------------------


def withheld_count(n_eligible: int, fraction: float) -> int:
    """The integer withheld count ``W``, fixed before anything else uses it.

    Round half up, then clamp so neither side of the cut can be empty — the
    clamp is §6.2's degenerate-partition guard expressed in arithmetic. A
    universe too small to supply both sides is refused by the caller rather than
    clamped into a shape it cannot hold.
    """
    if n_eligible < 2:
        raise ValueError(
            f"a disease universe of {n_eligible} cannot be cut into two non-empty "
            "partitions; at least 2 eligible diseases are required"
        )
    return min(max(math.floor(fraction * n_eligible + 0.5), 1), n_eligible - 1)


def p_no_validation_representation(n_total: int, n_stratum: int, withheld: int) -> float:
    """``P(X_s = 0)`` — the stratum contributes **nothing to the withheld side**.

    Under a uniform draw of ``withheld`` diseases without replacement, this is
    ``C(N − n_s, W) / C(N, W)``, which telescopes to

        ∏_{i=0}^{n_s−1} (N − W − i) / (N − i)

    **The product form is used rather than `math.comb`, and it is not merely an
    optimisation.** At deployment scale ``C(10576, 1586)`` is a ~1,900-digit
    integer; this is a product of ``n_s`` factors in ``[0, 1]``, evaluated in
    microseconds. It is also *domain-correct by construction*: when
    ``W > N − n_s`` the factor at ``i = N − W`` is exactly zero and the product
    collapses, which is the right answer and needs no guard. `math.comb` would
    have to be wrapped to avoid raising on a negative argument.

    An empty stratum returns 1.0 — the vacuous truth that a stratum with no
    members contributes none.
    """
    result = 1.0
    for i in range(n_stratum):
        numerator = n_total - withheld - i
        if numerator <= 0:
            return 0.0
        result *= numerator / (n_total - i)
    return result


def p_no_training_representation(n_total: int, n_stratum: int, withheld: int) -> float:
    """``P(X_s = n_s)`` — **every** disease in the stratum is withheld.

    ``C(N − n_s, W − n_s) / C(N, W)``, which telescopes to

        ∏_{i=0}^{n_s−1} (W − i) / (N − i)

    **This is a different event from the one above, and an earlier revision of
    the specification confused it with a third.** It was given as
    ``C(n_s, W) / C(N, W)``, which is the probability that the *whole withheld
    set came from this stratum* — the event ``X_s = W``. The three coincide only
    when ``W = n_s``, and that was the single case the brute-force check used, so
    the check confirmed a coincidence. The tests here enumerate ``W < n_s``,
    ``W = n_s`` and ``W > n_s``.

    Domain-correct by construction for the same reason as above: when
    ``W < n_s`` the factor at ``i = W`` is zero.
    """
    result = 1.0
    for i in range(n_stratum):
        numerator = withheld - i
        if numerator <= 0:
            return 0.0
        result *= numerator / (n_total - i)
    return result


def hypergeometric_mean(n_total: int, n_stratum: int, withheld: int) -> float:
    """``E[X_s] = W · n_s / N``. Generally fractional, and **not** the quota."""
    return withheld * n_stratum / n_total


def hypergeometric_sd(n_total: int, n_stratum: int, withheld: int) -> float:
    """Standard deviation of ``X_s``, in disease-count units.

    Reported as a standard deviation rather than a variance so it is directly
    comparable to the mean beside it. One form only: the document, this key and
    the tests would otherwise be free to drift.
    """
    if n_total < 2:
        return 0.0
    p = n_stratum / n_total
    variance = withheld * p * (1.0 - p) * (n_total - withheld) / (n_total - 1)
    return math.sqrt(max(variance, 0.0))


def largest_remainder_quotas(
    sizes: Sequence[int], withheld: int, keys: Sequence[Tuple[Any, ...]]
) -> List[int]:
    """Integer per-bucket quotas summing exactly to ``withheld``.

    **Quotas are assigned to buckets, not to diseases**, so equal fractional
    remainders break on a canonical *bucket* key — ascending band lower bound,
    the missing bucket last, then the label — and never on an ordering of the
    disease universe, which is the wrong object entirely.

    **No cap is applied, because under the stated precondition none is
    reachable.** ``withheld <= total`` makes every ``exact[i] <= sizes[i]``, so a
    floor cannot reach its bucket's size, and the shortfall is strictly smaller
    than the number of buckets, so each bucket is raised at most once. Earlier
    versions carried ``min(..., size)`` on the floor pass and a
    ``quotas[i] < sizes[i]`` test in the distribution pass; both were dead code,
    which is how they were found — removing either broke no test. The
    precondition is checked instead, where it *can* fail.
    """
    total = sum(sizes)
    if total == 0:
        return [0] * len(sizes)
    if withheld > total:
        raise ValueError(
            f"cannot allocate {withheld} across buckets holding {total} in total"
        )

    exact = [withheld * size / total for size in sizes]
    quotas = [int(math.floor(value)) for value in exact]
    order = sorted(
        range(len(sizes)),
        key=lambda i: (-(exact[i] - math.floor(exact[i])), keys[i]),
    )
    for i in order[: withheld - sum(quotas)]:
        quotas[i] += 1
    return quotas


# --------------------------------------------------------------------------
# Banding
# --------------------------------------------------------------------------


def band_label(value: Optional[int], bounds: Sequence[int]) -> str:
    """Which band a value falls in, as a stable label.

    ``None`` goes to the explicit missing bucket. It is never imputed and never
    silently dropped — and for the strata computed here it is structurally
    unreachable, since every eligible disease has both counts. The bucket exists
    so that the ordering rule is total rather than conditional on the data.
    """
    if value is None:
        return MISSING_LABEL
    for lower, upper in zip(bounds, list(bounds[1:]) + [None]):
        if value >= lower and (upper is None or value < upper):
            return f"{lower}+" if upper is None else (
                str(lower) if upper == lower + 1 else f"{lower}-{upper - 1}"
            )
    return MISSING_LABEL


def band_sort_key(label: str, bounds: Sequence[int]) -> Tuple[int, int, str]:
    """Canonical bucket order: ascending lower bound, missing last, then label.

    The first element separates numeric bands (0) from the missing bucket (1),
    which is what puts missing after every band regardless of its lower bound.
    The label is the final tie-break so the ordering is total even if two bands
    were ever given the same bound.
    """
    if label == MISSING_LABEL:
        return (1, 0, label)
    head = label.rstrip("+").split("-")[0]
    try:
        return (0, int(head), label)
    except ValueError:  # pragma: no cover - labels are generated, not parsed
        return (1, 0, label)


def bucket(values: Sequence[Optional[int]], bounds: Sequence[int]) -> List[Tuple[str, int]]:
    """Band populations, in canonical bucket order, empty bands included."""
    counts: Dict[str, int] = {}
    for value in values:
        label = band_label(value, bounds)
        counts[label] = counts.get(label, 0) + 1
    for lower in bounds:
        counts.setdefault(band_label(lower, bounds), 0)
    counts.setdefault(MISSING_LABEL, 0)
    return sorted(counts.items(), key=lambda kv: band_sort_key(kv[0], bounds))


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------


def stratification_report(
    bands: List[Tuple[str, int]], n_total: int, withheld: int, bounds: Sequence[int]
) -> List[Dict[str, Any]]:
    """One row per band, at one withheld count.

    ``quota`` and ``expected_withheld`` are reported side by side and are not the
    same object: the first is an integer allocation, the second a fractional
    expectation. Their difference has exactly one source — the per-bucket integer
    rounding the quota performs. (The separate rounding of ``f · N`` to ``W``
    explains something else: the gap between an ideal ``f · n_s`` and the
    realised-draw expectation ``W · n_s / N``.)
    """
    labels = [label for label, _ in bands]
    sizes = [size for _, size in bands]
    keys = [band_sort_key(label, bounds) for label in labels]
    quotas = largest_remainder_quotas(sizes, withheld, keys)

    return [
        {
            "band": label,
            "diseases": size,
            "quota": quota,
            "expected_withheld": hypergeometric_mean(n_total, size, withheld),
            "sd_withheld": hypergeometric_sd(n_total, size, withheld),
            "p_no_validation_representation": p_no_validation_representation(
                n_total, size, withheld
            ),
            "p_no_training_representation": p_no_training_representation(
                n_total, size, withheld
            ),
        }
        for label, size, quota in zip(labels, sizes, quotas)
    ]


def build_report(kg_path: Path, settings: AuditSettings, relationship: str) -> Dict[str, Any]:
    """The report, and a validating boundary in its own right.

    **Immutability is not validity.** ``AuditSettings`` is a public
    ``NamedTuple``, so any caller can construct one holding a negative
    ``min_phenotypes`` or a ``samples_per_disease`` of zero and hand it over. The
    settings and the deployment relationship are therefore re-checked here,
    before the knowledge graph is loaded and long before anything is written —
    validation is pure arithmetic on a handful of scalars, and it is idempotent,
    so paying for it twice on the command-line path costs nothing worth counting.
    """
    settings = validate_settings(**settings._asdict())
    relationship = validate_relationship(relationship)

    from src.kg import build_eligible_disease_profiles, retained_phenotype_count
    from src.kg.graph import KnowledgeGraph
    from src.utils.fingerprint import file_sha256

    kg = KnowledgeGraph.load_json(str(kg_path))
    disease_nodes = len(kg.get_node_id_mapping().get("disease", {}))
    eligible = build_eligible_disease_profiles(kg, settings.min_phenotypes)
    n_eligible = len(eligible)

    if n_eligible < 2:
        raise SystemExit(
            f"only {n_eligible} of {disease_nodes} disease nodes are eligible at "
            f"min_phenotypes={settings.min_phenotypes}; a split needs at least 2. "
            "Nothing was written."
        )

    phenotype_counts = [len(profile["phenotype_ids"]) for _, profile in eligible]
    gene_counts = [len(profile["gene_ids"]) for _, profile in eligible]
    support_sizes = [p + g for p, g in zip(phenotype_counts, gene_counts)]
    capacities = [
        math.comb(
            n_phen,
            retained_phenotype_count(
                n_phen,
                settings.min_phenotypes,
                settings.max_phenotypes,
                settings.phenotype_drop_rate,
            ),
        )
        for n_phen in phenotype_counts
    ]

    strata = {
        "phenotype_count": (
            bucket(phenotype_counts, PHENOTYPE_COUNT_BANDS), PHENOTYPE_COUNT_BANDS
        ),
        "gene_count": (bucket(gene_counts, GENE_COUNT_BANDS), GENE_COUNT_BANDS),
        "profile_support_size": (
            bucket(support_sizes, PROFILE_SUPPORT_BANDS), PROFILE_SUPPORT_BANDS
        ),
        "generator_capacity": (bucket(capacities, CAPACITY_BANDS), CAPACITY_BANDS),
    }

    sensitivity = []
    for fraction in settings.fractions:
        withheld = withheld_count(n_eligible, fraction)
        sensitivity.append(
            {
                "fraction_requested": fraction,
                "withheld_diseases": withheld,
                "retained_diseases": n_eligible - withheld,
                "realised_fraction": withheld / n_eligible,
                "strata": {
                    name: stratification_report(bands, n_eligible, withheld, bounds)
                    for name, (bands, bounds) in strata.items()
                },
            }
        )

    coverage = []
    for per_disease in settings.samples_per_disease:
        for fraction in settings.fractions:
            withheld = withheld_count(n_eligible, fraction)
            train_required = (n_eligible - withheld) * per_disease
            val_required = withheld * per_disease
            coverage.append(
                {
                    "samples_per_disease": per_disease,
                    "fraction_requested": fraction,
                    "train_required": train_required,
                    "val_required": val_required,
                    "train_budget_sufficient": settings.train_budget >= train_required,
                    "val_budget_sufficient": settings.val_budget >= val_required,
                }
            )

    return {
        "audit": "split_feasibility",
        "schema_version": SCHEMA_VERSION,
        "deployment_relationship": relationship,
        "assumptions": {
            "note": (
                "Supplied to this audit, not recovered from the workspace. No "
                "generation manifest exists yet, so the configuration an existing "
                "workspace was built under is not readable from it."
            ),
            "min_phenotypes": settings.min_phenotypes,
            "max_phenotypes": settings.max_phenotypes,
            "phenotype_drop_rate": settings.phenotype_drop_rate,
            "current_train_budget": settings.train_budget,
            "current_val_budget": settings.val_budget,
        },
        "inputs": {"kg": file_sha256(kg_path)},
        "universe": {
            "disease_nodes": disease_nodes,
            "eligible_diseases": n_eligible,
            "eligibility_rule": (
                f"profile phenotype support >= {settings.min_phenotypes}"
            ),
            "exclusions": {
                "below_min_phenotypes": disease_nodes - n_eligible,
                "note": (
                    "The only exclusion reason observable from a materialised "
                    "kg.json. Identifier-mapping outcomes happen during KG "
                    "construction and leave no trace in the artifact, so no "
                    "mapping success rate is reported here."
                ),
            },
        },
        "distributions": {
            name: [{"band": label, "diseases": size} for label, size in bands]
            for name, (bands, _) in strata.items()
        },
        "axis_definitions": {
            "phenotype_count": (
                "Phenotypes in the disease's generator profile. Includes phenotypes "
                "reached through an associated gene, not only those directly linked "
                "to the disease."
            ),
            "gene_count": "Genes associated with the disease.",
            "profile_support_size": (
                "phenotype_count + gene_count. **Not the disease node's direct KG "
                "degree**, because phenotype_count is propagated through genes. "
                "Direct degree is a different quantity and is not measured here; a "
                "disease with no incident edges at all has an empty profile and is "
                "already counted under exclusions."
            ),
            "generator_capacity": (
                "C(P, k) distinct phenotype subsets the generator can draw, with k "
                "from the production rule in src.kg.retained_phenotype_count."
            ),
        },
        "coverage_budgets": coverage,
        "sensitivity": sensitivity,
        "excluded_by_design": [
            "disease ids",
            "patient ids",
            "sample ids",
            "per-disease rows (only band populations are recorded)",
            "absolute paths, host names, operator names",
        ],
        "decides": "nothing — no fraction is selected and no allocation is written",
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split feasibility audit — aggregate-only, decides nothing"
    )
    parser.add_argument("--kg-path", type=Path, required=True,
                        help="kg.json for the workspace to audit.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing --output. Off by default.")
    parser.add_argument("--min-phenotypes", type=int, default=2,
                        help="Generator assumption, echoed into the report. Not "
                             "readable from a workspace: no generation manifest "
                             "exists yet.")
    parser.add_argument("--max-phenotypes", type=int, default=15,
                        help="Generator assumption, echoed into the report.")
    parser.add_argument("--phenotype-drop-rate", type=float, default=0.3,
                        help="Generator assumption, echoed into the report.")
    parser.add_argument("--train-budget", type=int, required=True,
                        help="Current training sample budget, for the coverage "
                             "column. An assumption, not an observation.")
    parser.add_argument("--val-budget", type=int, required=True,
                        help="Current validation sample budget. Same standing.")
    parser.add_argument("--fractions", type=float, nargs="+", default=list(DEFAULT_FRACTIONS),
                        help="Withheld fractions to report. 0.15 is the upstream value.")
    parser.add_argument("--samples-per-disease", type=int, nargs="+",
                        default=list(DEFAULT_SAMPLES_PER_DISEASE),
                        help="Budget dimension for the coverage column. Never "
                             "called k, which is the retained-phenotype count.")
    parser.add_argument("--deployment-relationship", default=UNSTATED_RELATIONSHIP,
                        choices=DEPLOYMENT_RELATIONSHIPS,
                        help="How this machine relates to the deployment. A bounded "
                             "vocabulary rather than free text, for the reason the "
                             "schema forbids host and operator names.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)

    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"{args.output} exists. Pass --overwrite or write elsewhere.")

    # Before the knowledge graph is loaded and before anything is written.
    try:
        settings = validate_settings(
            min_phenotypes=args.min_phenotypes,
            max_phenotypes=args.max_phenotypes,
            phenotype_drop_rate=args.phenotype_drop_rate,
            train_budget=args.train_budget,
            val_budget=args.val_budget,
            fractions=args.fractions,
            samples_per_disease=args.samples_per_disease,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    report = build_report(args.kg_path, settings, args.deployment_relationship)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    logger.info(
        "%s eligible of %s disease nodes -> %s",
        report["universe"]["eligible_diseases"],
        report["universe"]["disease_nodes"],
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
