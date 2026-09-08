#!/usr/bin/env python
"""
How the current generator's output differs from the upstream simulator's design.
================================================================================
`EVALUATION_COHORTS.md` §6.6 step 4. §1.5 already established the qualitative gap
by reading the upstream source: frequency-weighted initialisation, hierarchy
corruption, noise phenotypes, negative phenotypes and ~13 distractor genes per
patient, none of which `src/kg/sample_generator.py` has. That comparison needed
no measurement and this script does not repeat it.

What §1.5 could **not** establish, and named as unmeasured, is what our generator
actually produced. Three of its open statements are measurable from a workspace
alone, and this script closes them:

1. **Combinatorial capacity.** A disease with `P` phenotypes admits exactly
   `C(P, k)` distinct phenotype subsets, where `k` is `retained_phenotype_count`.
   Any disease drawn more often than that must repeat a subset. §1.5: *"how many
   diseases cross it here is unmeasured"*.
2. **Realised redundancy.** The bound above says repeats are unavoidable past a
   threshold. It does not say how many there are. Distinct model-visible
   signatures over emitted samples says exactly that.
3. **The leakage channel a disease-level cut does not close.** Disease-disjoint
   partitions cannot share a `(phenotypes, disease)` signature. They can share a
   *phenotype set* under two different disease labels, and then the model sees
   the same input with two answers, one of which it was never trained on. §1.5's
   cross-split question, restated for the regime that now exists.

A fourth section prices the largest single divergence. Frequency-weighted
initialisation is the upstream simulator's first stage, and whether we could
adopt it depends on whether the annotation source carries a frequency at all.

**Aggregate only.** BACKLOG §5.2 forbids patient ids, sample ids and per-disease
lists; every population here is reported as band counts or totals. The bands come
from `src.utils.banding`, shared with the feasibility audit so that a capacity
band in one report means the same thing in the other.

**This script measures. It decides nothing.** No threshold is applied and no
verdict is emitted; whether a redundancy figure is acceptable is a question for
the institution, informed by what the upstream cohort looks like.

Usage:
    python scripts/audit_generator_fidelity.py \\
        --data-dir data/workspaces/<ws> \\
        --external-dir data/external \\
        --output docs/working/EVIDENCE_generator_fidelity.json

Module: scripts/audit_generator_fidelity.py
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.kg.artifacts import GRAPH_ARTIFACTS  # noqa: E402
from src.evaluation.cohort import (  # noqa: E402
    MANIFEST_FILENAME,
    verify_generated_cohorts,
    verify_graph_artifacts,
)
from src.utils.banding import CAPACITY_BANDS, bucket  # noqa: E402
from src.utils.provenance import DEPLOYMENT_RELATIONSHIPS, UNSTATED_RELATIONSHIP  # noqa: E402

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

#: How many times a disease was drawn. Deliberately not `CAPACITY_BANDS`: draw
#: counts are a budget divided by a universe, an order of magnitude or two, while
#: capacities span thirty. Reusing capacity's bounds would put almost every
#: disease in the first band and report nothing.
DRAW_COUNT_BANDS: Tuple[int, ...] = (0, 1, 2, 3, 6, 11, 21, 51)

#: Columns of `phenotype.hpoa`, zero-indexed. Named rather than written as
#: literals at the point of use, which is where an off-by-one in a file-format
#: constant hides.
HPOA_QUALIFIER_COLUMN = 2
HPOA_PHENOTYPE_COLUMN = 3
HPOA_FREQUENCY_COLUMN = 7


def generation_config(data_dir: Path) -> Dict[str, Any]:
    """The three parameters that determine `k`.

    **`k` is not observable from the samples.** A sample records the phenotypes it
    kept, not the rule that chose how many, so an audit that guessed the rule
    would report the capacity of a generator nobody ran. `split_manifest.json`
    records the parameters the generator was actually given, and it is the only
    source accepted — there is no operator override, because the two could
    disagree and neither the audit nor the operator could tell which was right.

    A workspace without a manifest is refused upstream by ``resolve_cohort``.
    """
    manifest = json.loads((data_dir / MANIFEST_FILENAME).read_text())
    generation = manifest.get("generation", {})
    required = ("min_phenotypes", "max_phenotypes", "phenotype_drop_rate")
    missing = [key for key in required if key not in generation]
    if missing:
        raise SystemExit(
            f"{data_dir / MANIFEST_FILENAME} has no {', '.join(missing)} in its "
            "generation section, so the retained-phenotype rule this workspace was "
            "built under is unknown"
        )
    return {key: generation[key] for key in required}


def capacity_section(
    profiles: Dict[int, int], draws: Dict[str, Dict[int, int]], config: Dict[str, Any]
) -> Dict[str, Any]:
    """`C(P, k)` against how often each disease was actually drawn.

    ``profiles`` maps disease index to its phenotype count; ``draws`` maps a split
    name to disease index to sample count. A disease drawn more often than its
    capacity has repeated a phenotype subset — the bound is exact, and the excess
    is the number of samples that cannot be distinct.
    """
    from src.kg.sample_generator import retained_phenotype_count

    capacities = {
        disease: math.comb(
            n_phenotypes,
            retained_phenotype_count(
                n_phenotypes, config["min_phenotypes"], config["max_phenotypes"],
                config["phenotype_drop_rate"],
            ),
        )
        for disease, n_phenotypes in profiles.items()
    }

    per_split = {}
    for split, counts in draws.items():
        # A drawn disease absent from the profiles cannot be priced: its capacity
        # is unknown, not zero. Counted separately rather than folded into either
        # side of the comparison.
        unpriceable = [d for d in counts if d not in capacities]
        exceeded = [
            d for d, n in counts.items() if d in capacities and n > capacities[d]
        ]
        per_split[split] = {
            "diseases_drawn": len(counts),
            "diseases_without_a_known_capacity": len(unpriceable),
            "diseases_drawn_past_capacity": len(exceeded),
            "samples_in_excess_of_capacity": sum(
                counts[d] - capacities[d] for d in exceeded
            ),
            "draw_count_bands": bucket(list(counts.values()), DRAW_COUNT_BANDS),
        }

    return {
        "what_this_shows": (
            "a disease with P phenotypes admits exactly C(P, k) distinct phenotype "
            "subsets, so any disease drawn more often than that has repeated one. "
            "The bound is exact; the excess is how many samples cannot be distinct"
        ),
        "k_rule": "retained_phenotype_count(P, min_phenotypes, max_phenotypes, drop_rate)",
        "diseases_priced": len(capacities),
        "capacity_bands": bucket(
            # Banded, never listed: a per-disease capacity list is a per-disease
            # list, which §5.2 forbids in an evidence artifact.
            [min(c, CAPACITY_BANDS[-1]) for c in capacities.values()], CAPACITY_BANDS
        ),
        "per_split": per_split,
    }


def redundancy_section(cohorts: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Distinct model-visible signatures over emitted samples.

    The signature is the phenotype **set** and the disease id — what the model
    sees. `patient_id` is not part of it: two records with different patient ids
    and identical content are one training signal presented twice, and counting
    them as two is the mistake this section exists to avoid making.
    """
    section = {
        "what_this_shows": (
            "how much of each cohort is distinct as the model sees it. The "
            "capacity bound says repeats become unavoidable past a threshold; this "
            "says how many there are"
        ),
        "signature": "(frozenset(phenotype_ids), disease_id) — patient_id excluded",
    }
    for split, samples in cohorts.items():
        signatures = {
            (frozenset(int(p) for p in s["phenotype_ids"]), int(s["disease_id"]))
            for s in samples
        }
        section[split] = {
            "samples": len(samples),
            "distinct_signatures": len(signatures),
            "distinct_over_samples": (
                len(signatures) / len(samples) if samples else None
            ),
            "duplicate_samples": len(samples) - len(signatures),
        }
    return section


def _shared_input_is_possible(
    val_profiles: Dict[int, frozenset],
    train_profiles: Dict[int, frozenset],
    k_of: Dict[int, int],
) -> int:
    """Validation diseases that could emit an input a training disease could too.

    **The condition, exactly.** The generator emits a `k`-subset of a disease's
    phenotype profile. Two diseases can therefore emit the *same* set iff they
    retain the same number of phenotypes and share at least that many:

        k_A == k_B == k   and   |A ∩ B| >= k

    Identical full profiles are the special case `A == B`. Restricting to it — as
    the first version of this section did — undercounts: with `A = {1,2,3}`,
    `B = {1,2,4}` and `k = 2`, both can emit `{1,2}` while no profile is a
    duplicate of any other.

    **Why an inverted index rather than a pairwise scan.** Comparing every
    (val, train) pair is |val| x |train| intersections — tens of millions on the
    audited universe, each over sets of up to a hundred ids. Instead: within each
    `k` group, walk the val disease's phenotypes, count how many of them each
    training disease shares, and stop at the first disease reaching `k`. Diseases
    that share no phenotype at all are never touched, which is the overwhelming
    majority of pairs, and the early exit ends most val diseases after a handful
    of increments. Memory is one counter dict per val disease, discarded
    immediately.

    Returns the count of affected validation diseases, not the number of pairs:
    the question is how much of the validation cohort is compromised, and one
    disease with forty possible partners is still one disease.
    """
    by_k: Dict[int, Dict[int, List[int]]] = {}
    for disease, profile in train_profiles.items():
        index = by_k.setdefault(k_of[disease], {})
        for phenotype in profile:
            index.setdefault(phenotype, []).append(disease)

    affected = 0
    for disease, profile in val_profiles.items():
        k = k_of[disease]
        index = by_k.get(k)
        if not index:
            continue
        shared: Dict[int, int] = {}
        for phenotype in profile:
            for partner in index.get(phenotype, ()):
                count = shared.get(partner, 0) + 1
                if count >= k:
                    break
                shared[partner] = count
            else:
                continue
            break
        else:
            continue
        affected += 1
    return affected


def cross_split_section(
    cohorts: Dict[str, List[Dict[str, Any]]],
    profiles: Dict[int, frozenset],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """The channel a disease-level cut leaves open.

    Disease-disjoint partitions cannot share a `(phenotypes, disease)` signature —
    the disease ids are disjoint by construction. They can share a **phenotype
    set** under two different labels, and then a validation input is one the model
    has already seen in training with a different answer.

    Two measurements, because they answer different questions. The *drawn* one
    says what this cohort actually contains. The *structural* one says what the
    knowledge graph makes possible regardless of what was drawn, and is therefore
    the figure that survives a regeneration with a different seed.
    """
    from src.kg.sample_generator import retained_phenotype_count

    train_sets = {
        frozenset(int(p) for p in s["phenotype_ids"]) for s in cohorts["train"]
    }
    val_sets = {frozenset(int(p) for p in s["phenotype_ids"]) for s in cohorts["val"]}
    shared = train_sets & val_sets

    touched = {
        split: sum(
            1
            for s in samples
            if frozenset(int(p) for p in s["phenotype_ids"]) in shared
        )
        for split, samples in cohorts.items()
    }

    train_diseases = {int(s["disease_id"]) for s in cohorts["train"]} & set(profiles)
    val_diseases = {int(s["disease_id"]) for s in cohorts["val"]} & set(profiles)
    k_of = {
        disease: retained_phenotype_count(
            len(profiles[disease]), config["min_phenotypes"],
            config["max_phenotypes"], config["phenotype_drop_rate"],
        )
        for disease in train_diseases | val_diseases
    }
    train_profile_set = {profiles[d] for d in train_diseases}
    exact_duplicates = sum(
        1 for d in val_diseases if profiles[d] in train_profile_set
    )
    possible = _shared_input_is_possible(
        {d: profiles[d] for d in val_diseases},
        {d: profiles[d] for d in train_diseases},
        k_of,
    )

    return {
        "what_this_shows": (
            "a disease-level cut cannot leak a (phenotypes, disease) signature, "
            "because the disease ids are disjoint. It can leak a phenotype set "
            "under two labels, which is a validation input the model has already "
            "seen in training with a different answer"
        ),
        "drawn": {
            "distinct_phenotype_sets_in_both_splits": len(shared),
            "train_samples_touched": touched["train"],
            "val_samples_touched": touched["val"],
            "val_samples": len(cohorts["val"]),
        },
        "structural": {
            "what_this_shows": (
                "what the knowledge graph makes possible regardless of what was "
                "drawn, so it survives regeneration under a different seed"
            ),
            "condition": "k_A == k_B == k and |A ∩ B| >= k",
            "val_diseases_measured": len(val_diseases),
            "val_diseases_that_could_share_an_input_with_train": possible,
            "val_diseases_with_an_identical_profile_in_train": exact_duplicates,
            "note": (
                "the identical-profile count is the special case A == B and is "
                "therefore a subset of the first figure, not an alternative to it"
            ),
        },
    }


def frequency_section(
    kg_edge_weights: List[float], hpoa_path: Optional[Path], hpoa_digest: Optional[str]
) -> Dict[str, Any]:
    """Is there a frequency signal to weight by at all?

    The upstream simulator's first stage draws each phenotype with
    `np.random.binomial(1, freq)` on its Orphanet frequency band. Adopting that
    needs a frequency per phenotype-disease edge, and this measures whether one
    exists.

    **The ambiguity is the finding, and it is reported as bounds rather than
    resolved by guessing.** `HPOAnnotationParser.parse_frequency` returns `1.0`
    both for an absent or unparseable annotation *and* for a real one --
    `HP:0040280` (Obligate), `"100%"`, `"12/12"`. So neither the graph nor a token
    count can give a single number, and the measurement is stated as a lower and
    an upper bound with the ambiguous mass named between them:

    - **lower** — rows whose token the shared parser turns into something other
      than `1.0`. Certainly a usable frequency.
    - **ambiguous** — rows with a non-empty token that parses to `1.0`. Obligate,
      a literal 100%, an `n/n` fraction, or unparseable; this measurement cannot
      separate them without duplicating the parser's rules.
    - **upper** — lower + ambiguous.

    Counted **through the parser the graph was built with**, not through a private
    reimplementation of its rules, so the figure describes the parser that runs.
    """
    informative = sum(1 for w in kg_edge_weights if w != 1.0)
    section = {
        "what_this_shows": (
            "whether the annotation source carries the frequency the upstream "
            "simulator's first stage requires"
        ),
        "from_the_graph": {
            "phenotype_disease_edges": len(kg_edge_weights),
            "edges_with_a_non_default_weight": informative,
            "fraction": informative / len(kg_edge_weights) if kg_edge_weights else None,
            "why_only_a_lower_bound": (
                "parse_frequency returns 1.0 both for a missing annotation and for "
                "a real one (HP:0040280 Obligate, '100%', '12/12'), so the 1.0 "
                "bucket conflates the two"
            ),
        },
    }
    if hpoa_path is None:
        section["from_the_source"] = {
            "measured": False,
            "why": "--external-dir was not supplied, so phenotype.hpoa was not read",
        }
        return section

    from src.data_sources.hpo_annotations import HPOAnnotationParser

    parse = HPOAnnotationParser.parse_frequency
    total = absent = parsed_below_one = ambiguous = 0
    with open(hpoa_path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            # Only the line terminator is removed. `line.strip()` would also eat
            # a leading tab, shifting every column left by one and reading the
            # disease name as the qualifier. What keeps the *denominator* honest
            # is the short-row rule below, not this.
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) <= HPOA_PHENOTYPE_COLUMN:
                continue
            # The same row filter `parse_phenotype_hpoa` applies -- NOT-qualified
            # rows out, a real `HP:` term required, which is also what drops the
            # header. A denominator counted under looser rules would not be
            # comparable to the annotation count the graph was built from.
            if parts[HPOA_QUALIFIER_COLUMN] == "NOT":
                continue
            if not parts[HPOA_PHENOTYPE_COLUMN].startswith("HP:"):
                continue
            total += 1
            # A short row is unannotated, not uncountable -- the same reading
            # `parse_phenotype_hpoa` gives it.
            token = (
                parts[HPOA_FREQUENCY_COLUMN]
                if len(parts) > HPOA_FREQUENCY_COLUMN else ""
            ).strip()
            if not token:
                absent += 1
            elif parse(token) != 1.0:
                parsed_below_one += 1
            else:
                ambiguous += 1

    section["from_the_source"] = {
        "measured": True,
        "source": hpoa_path.name,
        "digest": hpoa_digest,
        "rows": total,
        "rows_with_no_frequency_token": absent,
        "rows_whose_token_parses_below_one": parsed_below_one,
        "rows_whose_token_parses_to_one": ambiguous,
        "usable_frequency_fraction_lower_bound": (
            parsed_below_one / total if total else None
        ),
        "usable_frequency_fraction_upper_bound": (
            (parsed_below_one + ambiguous) / total if total else None
        ),
        "why_two_bounds": (
            "a token parsing to 1.0 is Obligate, a literal 100%, an n/n fraction, "
            "or unparseable, and separating those would mean reimplementing "
            "parse_frequency's rules here"
        ),
        "note": (
            "rows of phenotype.hpoa that survive the parser's own filters (comment, "
            "NOT-qualified and non-HP rows dropped) but before MONDO resolution and "
            "de-duplication, so this is the source's coverage and not the graph's "
            "edge count"
        ),
    }
    return section


def build_report(
    data_dir: Path, external_dir: Optional[Path], relationship: str,
) -> Dict[str, Any]:
    """The workspace is the unit, so the graph is the one this workspace holds.

    ``--kg-path`` used to be separate, which let a graph from one workspace be
    characterised against cohorts from another — the very mixing the artifact
    binding exists to refuse.
    """
    kg_path = data_dir / GRAPH_ARTIFACTS["kg"]

    from src.core.types import EdgeType
    from src.kg.graph import KnowledgeGraph
    from src.kg.sample_generator import build_eligible_disease_profiles
    from src.kg.storage.file_storage import read_samples
    from src.utils.fingerprint import file_sha256

    if relationship not in DEPLOYMENT_RELATIONSHIPS:
        raise SystemExit(
            f"deployment_relationship must be one of {DEPLOYMENT_RELATIONSHIPS}, "
            f"got {relationship!r}"
        )

    # **Both cohorts must be this project's own.** This audit characterises *our*
    # generator, so a supplied cohort has nothing here to be measured against —
    # its samples were not produced by the rule whose capacity is being priced.
    graph_digests = verify_graph_artifacts(data_dir)
    verify_generated_cohorts(data_dir)
    config = generation_config(data_dir)
    kg = KnowledgeGraph.load_json(str(kg_path))

    # Eligibility at the *generation* threshold, so the universe priced here is
    # the one the generator drew from rather than a wider one this audit chose.
    eligible = build_eligible_disease_profiles(kg, config["min_phenotypes"])
    phenotype_counts = {d: len(p["phenotype_ids"]) for d, p in eligible}
    phenotype_profiles = {
        d: frozenset(int(x) for x in p["phenotype_ids"]) for d, p in eligible
    }

    cohorts: Dict[str, List[Dict[str, Any]]] = {}
    for split in ("train", "val"):
        cohorts[split] = [
            {"phenotype_ids": list(s.phenotype_ids), "disease_id": int(s.disease_id)}
            for s in read_samples(data_dir, split)
        ]
        if not cohorts[split]:
            raise SystemExit(
                f"the {split} split holds no samples; every section here is a "
                "statement about a generated cohort, and over an empty one each "
                "would be vacuously clean"
            )

    draws: Dict[str, Dict[int, int]] = {}
    for split, samples in cohorts.items():
        counts: Dict[int, int] = {}
        for sample in samples:
            counts[sample["disease_id"]] = counts.get(sample["disease_id"], 0) + 1
        draws[split] = counts

    weights = [
        float(edge.weight)
        for edge in kg._edges
        if edge.edge_type == EdgeType.PHENOTYPE_OF_DISEASE
    ]
    hpoa = (external_dir / "phenotype.hpoa") if external_dir is not None else None
    if hpoa is not None and not hpoa.exists():
        raise SystemExit(f"{hpoa} does not exist")
    # **A basename is not identity.** The frequency figures are read from this
    # file, so a report citing them has to say which bytes it read.
    hpoa_digest = file_sha256(hpoa) if hpoa is not None else None

    artifacts = dict(graph_digests)
    artifacts["train_samples"] = file_sha256(data_dir / "train_samples.json")
    artifacts["val_samples"] = file_sha256(data_dir / "val_samples.json")
    artifacts["split_manifest"] = file_sha256(data_dir / MANIFEST_FILENAME)
    if hpoa_digest is not None:
        artifacts["phenotype_hpoa"] = hpoa_digest

    return {
        "schema_version": SCHEMA_VERSION,
        "subject": (
            "what the current generator produced, on the three points "
            "EVALUATION_COHORTS §1.5 named as unmeasured, plus whether the "
            "frequency signal its first upstream stage needs exists"
        ),
        "not_a_verdict": (
            "no threshold is applied and no cohort is accepted or rejected here"
        ),
        "generation_config": config,
        "generation_config_source": MANIFEST_FILENAME,
        "artifacts": artifacts,
        "combinatorial_capacity": capacity_section(phenotype_counts, draws, config),
        "realised_redundancy": redundancy_section(cohorts),
        "cross_split_phenotype_sets": cross_split_section(
            cohorts, phenotype_profiles, config
        ),
        "frequency_signal": frequency_section(weights, hpoa, hpoa_digest),
        "deployment_relationship": relationship,
        "excluded_by_design": [
            "patient ids",
            "sample ids",
            "per-disease lists (populations are reported as band counts)",
            "phenotype and disease identifiers",
        ],
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generator fidelity — EVALUATION_COHORTS §6.6 step 4"
    )
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="The workspace. Its kg.json, graph tensors, cohorts "
                             "and manifest are one production event and are read "
                             "as one; there is no separate --kg-path, because a "
                             "graph from elsewhere is exactly what the artifact "
                             "binding refuses.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--external-dir", type=Path, default=None,
                        help="Directory holding phenotype.hpoa. Optional: without it "
                             "the frequency section reports only the lower bound the "
                             "graph supports, and says so.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing --output. Off by default.")
    parser.add_argument("--deployment-relationship", default=UNSTATED_RELATIONSHIP,
                        choices=DEPLOYMENT_RELATIONSHIPS,
                        help="How this machine relates to the deployment. A bounded "
                             "vocabulary rather than free text. Unverified by design.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)

    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"{args.output} exists. Pass --overwrite or write elsewhere.")

    try:
        report = build_report(
            args.data_dir, args.external_dir, args.deployment_relationship
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    logger.info("Generator fidelity -> %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
