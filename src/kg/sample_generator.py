"""
Training Sample Generator
==========================
Generates simulated patient training samples from a KnowledgeGraph.

Each sample represents a "patient" with:
  - A set of observed phenotypes (from the KG)
  - A correct disease diagnosis (ground truth)
  - Optional: candidate genes

The generator traverses disease->phenotype and disease->gene edges
in the KG to create realistic training data for the diagnosis model.

Output format matches what scripts/train_model.py expects:
  [{"patient_id": str, "phenotype_ids": [int], "disease_id": int}, ...]
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from src.kg.artifacts import GRAPH_ARTIFACTS, SPLIT_MANIFEST_SCHEMA_VERSION
from src.kg.disease_allocation import (
    DiseaseAllocation,
    derive_stream,
    disease_set_digest,
    require_eligible,
    universe_digest,
    validate_allocation,
)
from src.kg.graph import KnowledgeGraph
from src.utils.fingerprint import file_sha256

logger = logging.getLogger(__name__)

#: Bumped when generation changes in a way that makes two cohorts from the same
#: allocation, budgets and seed differ. Recorded in the split manifest: the
#: coverage-first pass changes the sample distribution relative to pure
#: replacement sampling, and a reader comparing two workspaces has to be able to
#: see that they were produced under different rules.
GENERATION_ALGORITHM = "coverage-first-then-replacement"
GENERATION_ALGORITHM_VERSION = 1



def retained_phenotype_count(
    n_phenotypes: int,
    min_phenotypes: int,
    max_phenotypes: int,
    phenotype_drop_rate: float,
) -> int:
    """How many phenotypes a generated sample keeps — the ``k`` in ``C(P, k)``.

    **The one definition, called rather than restated.** ``_generate_samples``
    uses it to build every sample, and the split feasibility audit uses it to
    compute generator-capacity bands. A second copy of this arithmetic would let
    the audit report the capacity of a generator nobody runs, and the copy would
    stay green while doing so — which is exactly the failure mode a shared
    ``build_eligible_disease_profiles`` already rules out for eligibility.

    The floor is ``min_phenotypes`` and the ceilings are ``max_phenotypes`` and
    the disease's own phenotype count, so the result is always in
    ``[min(min_phenotypes, n_phenotypes), n_phenotypes]``.
    """
    keep = max(min_phenotypes, int(n_phenotypes * (1.0 - phenotype_drop_rate)))
    return min(keep, max_phenotypes, n_phenotypes)


def generate_training_samples(
    kg: KnowledgeGraph,
    allocation: DiseaseAllocation,
    num_train: int,
    num_val: int,
    min_phenotypes: int = 2,
    max_phenotypes: int = 15,
    phenotype_drop_rate: float = 0.3,
    output_dir: Optional[Path] = None,
    graph_digests: Optional[Dict[str, str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Generate simulated patients from a **disease allocation**.

    **The allocation is supplied, not decided here** (§6.2). Each partition is
    generated separately from its own diseases and its own random stream, so the
    two cohorts are disease-disjoint by construction rather than by luck. The
    superseded version drew one pooled set and sliced it by index, which splits
    patients and leaves every multi-sample disease on both sides.

    ``kg`` is still taken because the manifest records the graph the allocation
    was cut from; the samples themselves come from the allocation's profiles.

    Args:
        kg: the KnowledgeGraph the allocation was built from.
        allocation: from ``allocate_diseases`` or ``train_only_allocation``.
        num_train: training sample budget. Must reach every allocated training
            disease — see ``_generate_partition``.
        num_val: validation sample budget. ``0`` means no validation cohort, and
            then the allocation must have no validation partition either.
        min_phenotypes / max_phenotypes / phenotype_drop_rate: generation config.
        output_dir: when given, writes ``train_samples.json``,
            ``val_samples.json`` and ``split_manifest.json``.
        graph_digests: the digests of the graph artifacts, **computed by whoever
            wrote them**, keyed by manifest role. Required when writing a
            manifest. This function does not hash them itself and must not: it
            would be digesting whatever files happen to sit in ``output_dir``,
            which is a statement about the directory rather than about the export
            this allocation was cut from. Only the writer can make that binding.

    Returns:
        ``(train_samples, val_samples, manifest)``.
    """
    _validate_generation_inputs(
        num_train, num_val, min_phenotypes, max_phenotypes, phenotype_drop_rate
    )
    validate_allocation(allocation)
    require_eligible(allocation, min_phenotypes)

    # **The allocation must have been cut from the graph now being generated
    # from.** ``kg`` was previously accepted and never read, so an allocation
    # built from one knowledge graph could be generated against another: same
    # disease indices, different phenotype content, and samples drawn from
    # profiles the allocation never saw. Recomputing eligibility here and
    # comparing content digests is what makes the docstring's claim true.
    observed = universe_digest(build_eligible_disease_profiles(kg, min_phenotypes))
    if observed != allocation.universe_digest:
        raise ValueError(
            "this allocation was cut from a different disease universe than the "
            f"graph supplied ({allocation.universe_digest[:12]}... vs "
            f"{observed[:12]}...). Re-allocate from this graph, or generate "
            "against the graph the allocation was cut from."
        )

    if num_val == 0 and allocation.val:
        raise ValueError(
            f"num_val is 0 but the allocation withholds {len(allocation.val)} "
            "diseases; use train_only_allocation, or ask for validation samples"
        )
    if num_val > 0 and not allocation.val:
        raise ValueError(
            f"num_val is {num_val} but the allocation has no validation partition"
        )

    config = dict(
        min_phenotypes=min_phenotypes,
        max_phenotypes=max_phenotypes,
        phenotype_drop_rate=phenotype_drop_rate,
    )

    train_samples = _generate_partition(
        allocation.train, num_train, "train", allocation.seed, **config
    )
    val_samples = _generate_partition(
        allocation.val, num_val, "val", allocation.seed, **config
    )

    logger.info(
        "Generated %d train and %d val samples", len(train_samples), len(val_samples)
    )

    # **Write the sample files first, then digest the bytes that actually
    # landed.** Hashing a separately reconstructed representation would record a
    # digest of something no reader can obtain — the manifest has to describe the
    # files on disk, not an equivalent-looking serialisation of the same objects.
    artifacts: Dict[str, Optional[str]] = {"train_samples": None, "val_samples": None}
    if output_dir is not None:
        output_dir = Path(output_dir)
        refuse_if_checkpoints_exist(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, payload in (
            ("train_samples.json", train_samples),
            ("val_samples.json", val_samples),
        ):
            with open(output_dir / name, "w") as handle:
                json.dump(payload, handle)
        artifacts["train_samples"] = file_sha256(output_dir / "train_samples.json")
        artifacts["val_samples"] = file_sha256(output_dir / "val_samples.json")

    # **The graph digests are supplied by whoever wrote those files, never taken
    # here.** Hashing `output_dir/*.pt` would digest whatever happens to sit
    # there, which need not be the export of the graph this allocation was cut
    # from: a caller can pass graph A with its allocation while the directory
    # holds graph B's tensors, and the manifest would record A's universe digest
    # beside B's file digests as one provenance chain. Only the writer of an
    # export can vouch that these are the artifacts it just produced.
    #
    # All four are required together. They are one call to `export_graph_data`
    # from one in-memory graph, so a manifest binding some of them describes half
    # a production event — and the half it omits is the half a model consumes.
    # **A written workspace must bind its whole graph export.** All four roles are
    # one call to `export_graph_data` from one in-memory graph, so a manifest
    # binding some of them describes half a production event — and the half it
    # omits is the half a model consumes. Only checked when a workspace is being
    # written: an in-memory manifest describes a cut, not a directory.
    artifacts.update(graph_digests or {})
    if output_dir is not None:
        unbound = [role for role in GRAPH_ARTIFACTS if artifacts.get(role) is None]
        if unbound:
            raise ValueError(
                f"this workspace would leave {', '.join(sorted(unbound))} unbound. "
                "Only the writer of the graph export can vouch for those digests, "
                "so they are supplied rather than recomputed here — and a manifest "
                "without them cannot show that the tensors a model consumes are "
                "this graph's."
            )

    manifest = build_split_manifest(
        allocation=allocation,
        train_samples=train_samples,
        val_samples=val_samples,
        config=config,
        num_train=num_train,
        num_val=num_val,
        artifacts=artifacts,
    )

    if output_dir is not None:
        with open(output_dir / "split_manifest.json", "w") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        logger.info("Samples and split manifest saved to %s", output_dir)

    return train_samples, val_samples, manifest


def validate_sample_budgets(num_train: Any, num_val: Any) -> None:
    """The sample budgets, checked against their whole domain — the one copy.

    **Phase one of a two-phase preflight, and the reason it is here rather than
    in the build script.** Whether a budget is a usable number is knowable from
    the budget alone, so it is knowable before an ontology is read; whether it is
    *large enough* needs the allocation, so it cannot be. Splitting the check on
    that line is what lets the cheap half run first.

    A second copy in the script would be the failure this exists to prevent. The
    build path used to check only "not ``None``" and "not smaller than the
    partition", and ``50000.0`` and ``True`` slip through both: a float is never
    less than a disease count, and ``True`` covers a one-disease partition. The
    graph was then written and the generator refused afterwards — the half-built
    workspace the preflight is for. One validator called from both places cannot
    drift into that shape again.

    ``None`` is refused here rather than by a separate "explicit budgets" check,
    because a missing budget and a nonsensical one are the same event to every
    caller: no usable number was supplied.
    """
    for name, value in (("num_train", num_train), ("num_val", num_val)):
        if value is None:
            raise ValueError(
                f"{name} is required and has no default; every allocated disease "
                "must receive at least one sample, so any fixed default would "
                "fail on a real disease universe"
            )
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer, got {value!r}")
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")


def _validate_generation_inputs(
    num_train: Any,
    num_val: Any,
    min_phenotypes: Any,
    max_phenotypes: Any,
    phenotype_drop_rate: Any,
) -> None:
    """Narrow domain checks, before the graph is walked or anything is written.

    The same reason the audit validates at its API rather than only at argparse:
    this function is importable, and a budget of ``True`` would otherwise pass as
    ``1`` while a non-finite drop rate would surface as an obscure failure deep
    inside sampling.
    """
    validate_sample_budgets(num_train, num_val)
    for name, value, minimum in (
        ("min_phenotypes", min_phenotypes, 1),
        ("max_phenotypes", max_phenotypes, min_phenotypes),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer, got {value!r}")
        if value < minimum:
            raise ValueError(f"{name} must be >= {minimum}, got {value}")
    if isinstance(phenotype_drop_rate, bool) or not isinstance(
        phenotype_drop_rate, (int, float)
    ):
        raise ValueError(
            f"phenotype_drop_rate must be a number, got {phenotype_drop_rate!r}"
        )
    if not math.isfinite(phenotype_drop_rate) or not 0.0 <= phenotype_drop_rate <= 1.0:
        raise ValueError(
            f"phenotype_drop_rate must be finite and in [0, 1], got {phenotype_drop_rate}"
        )


def refuse_if_checkpoints_exist(workspace: Path) -> None:
    """Refuse to regenerate samples where trained checkpoints already live.

    **A split regime is not a label, and this makes that an enforced fact rather
    than a policy.** Checkpoints trained on a sample-level split carry a
    ``val_mrr`` measuring recognition of new phenotype subsets of diseases they
    have labelled examples of. Checkpoints trained on a disease-disjoint split
    carry a ``val_mrr`` measuring generalisation to diseases they have none for.
    Two different quantities under one name.

    They meet in ``src/utils/checkpoint_paths.py``, whose selector reads the raw
    ranking metric out of each checkpoint's own logs and has no way to tell the
    regimes apart. Regenerating samples in place would leave both kinds in one
    directory for it to compare.

    There is no override flag, and none is wanted: a fresh workspace is cheap.
    The graph artifacts can simply be copied, which is better than rebuilding
    them — an identical ``kg.json`` digest proves both workspaces were cut from
    the same graph.
    """
    checkpoints = workspace / "checkpoints"
    if not checkpoints.is_dir():
        return
    existing = sorted(path.name for path in checkpoints.iterdir())
    if not existing:
        return
    raise FileExistsError(
        f"{workspace} already holds trained checkpoints ({', '.join(existing[:3])}"
        f"{'...' if len(existing) > 3 else ''}). Regenerating samples here would "
        "put two split regimes in one checkpoint directory, where the ranking-"
        "metric selector cannot tell them apart. Generate into a fresh workspace "
        "and copy the graph artifacts across."
    )


def build_split_manifest(
    *,
    allocation: DiseaseAllocation,
    train_samples: Sequence[Dict[str, Any]],
    val_samples: Sequence[Dict[str, Any]],
    config: Dict[str, Any],
    num_train: int,
    num_val: int,
    artifacts: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    """What the sample digests cannot say: how this workspace was cut.

    Two byte-different sample files could have been cut the same way or
    differently, and their digests cannot tell them apart. The manifest can.

    **The realised sets are derived from the emitted records, never copied from
    the allocation.** A realised field populated from allocation metadata would
    restate the allocation rather than evidence it, and the coverage assertion
    below would be checking a value against itself.

    ``artifacts`` is recorded as given. Whether it binds the whole graph export is
    checked where a **workspace** is produced — ``generate_training_samples`` with
    an ``output_dir`` — because that is where the claim is made. Building a
    manifest in memory to inspect a cut produces no graph export to bind.
    """

    realised_train = {int(sample["disease_id"]) for sample in train_samples}
    realised_val = {int(sample["disease_id"]) for sample in val_samples}

    if realised_train != set(allocation.train_ids):
        raise AssertionError(
            f"training coverage broken: {len(set(allocation.train_ids) - realised_train)} "
            "allocated diseases received no sample"
        )
    if realised_val != set(allocation.val_ids):
        raise AssertionError(
            f"validation coverage broken: {len(set(allocation.val_ids) - realised_val)} "
            "allocated diseases received no sample"
        )

    manifest = {
        "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
        "generation": {
            "algorithm": GENERATION_ALGORITHM,
            "algorithm_version": GENERATION_ALGORITHM_VERSION,
            "num_train": num_train,
            "num_val": num_val,
            **config,
        },
        "allocation": allocation.provenance(),
        "realised": {
            "train_diseases": len(realised_train),
            "val_diseases": len(realised_val),
            "train_digest": disease_set_digest(sorted(realised_train)),
            "val_digest": disease_set_digest(sorted(realised_val)),
            "derived_from": "the emitted sample records, not the allocation",
        },
        "artifacts": artifacts,
        "disjoint": not (realised_train & realised_val),
    }
    if not manifest["disjoint"]:  # pragma: no cover - impossible from one allocation
        raise AssertionError("train and validation partitions overlap at the disease level")
    return manifest


def build_eligible_disease_profiles(
    kg: KnowledgeGraph,
    min_phenotypes: int = 2,
) -> List[Tuple[int, Dict[str, Any]]]:
    """Disease profiles filtered to those a sample can actually be generated from.

    **The one definition of "eligible", shared rather than restated.** Sample
    generation and the split feasibility audit must describe the *same* disease
    universe: an audit that measured a different universe from the one the
    generator partitions would report costs for a split nobody runs. Two
    implementations of one filter can disagree; one cannot.

    Returned as a list of ``(disease_index, profile)`` pairs, in dictionary order.
    A profile is ``{"phenotype_ids": [...], "gene_ids": [...]}``.

    Args:
        kg: KnowledgeGraph with nodes and edges loaded.
        min_phenotypes: A disease needs at least this many phenotypes to be
            eligible, because a sample keeps at least this many of them.

    Returns:
        Eligible ``(disease_index, profile)`` pairs; empty when none qualify.
    """
    node_mapping = kg.get_node_id_mapping()
    disease_profiles = _build_disease_profiles(kg, node_mapping)
    return [
        (disease_idx, profile)
        for disease_idx, profile in disease_profiles.items()
        if len(profile["phenotype_ids"]) >= min_phenotypes
    ]


def _build_disease_profiles(
    kg: KnowledgeGraph,
    node_mapping: Dict[str, Dict[str, int]],
) -> Dict[int, Dict[str, Any]]:
    """
    Build a profile for each disease: its associated phenotypes and genes.

    Traverses edges to find:
      - Phenotypes linked to each disease (PHENOTYPE_OF_DISEASE, GENE_HAS_PHENOTYPE via shared gene)
      - Genes linked to each disease (GENE_ASSOCIATED_WITH_DISEASE)
    """
    disease_mapping = node_mapping.get("disease", {})
    phenotype_mapping = node_mapping.get("phenotype", {})
    gene_mapping = node_mapping.get("gene", {})

    # disease_idx -> {phenotype_ids: set, gene_ids: set}
    profiles: Dict[int, Dict[str, Set[int]]] = {}
    for d_idx in disease_mapping.values():
        profiles[d_idx] = {"phenotype_ids": set(), "gene_ids": set()}

    # Two-pass edge traversal:
    # Pass 1: collect direct edges (phenotype-disease, gene-disease)
    gene_phenotype_edges: List[Tuple[int, int]] = []

    for edge in kg._edges:
        src_str = str(edge.source_id)
        tgt_str = str(edge.target_id)
        et = edge.edge_type.value

        if et == "phenotype_of_disease":
            pheno_idx = phenotype_mapping.get(src_str)
            disease_idx = disease_mapping.get(tgt_str)
            if pheno_idx is not None and disease_idx is not None:
                profiles[disease_idx]["phenotype_ids"].add(pheno_idx)

        elif et == "gene_associated_with_disease":
            gene_idx = gene_mapping.get(src_str)
            disease_idx = disease_mapping.get(tgt_str)
            if gene_idx is not None and disease_idx is not None:
                profiles[disease_idx]["gene_ids"].add(gene_idx)

        elif et == "gene_has_phenotype":
            gene_idx = gene_mapping.get(src_str)
            pheno_idx = phenotype_mapping.get(tgt_str)
            if gene_idx is not None and pheno_idx is not None:
                gene_phenotype_edges.append((gene_idx, pheno_idx))

    # Pass 2: propagate gene-phenotype edges to diseases via reverse index
    gene_to_diseases: Dict[int, Set[int]] = {}
    for d_idx, prof in profiles.items():
        for g_idx in prof["gene_ids"]:
            gene_to_diseases.setdefault(g_idx, set()).add(d_idx)

    for gene_idx, pheno_idx in gene_phenotype_edges:
        for d_idx in gene_to_diseases.get(gene_idx, ()):
            profiles[d_idx]["phenotype_ids"].add(pheno_idx)

    return {
        d_idx: {
            # **Sorted, so sampling cannot depend on set iteration order.** For
            # the integer ids used today CPython's order is already stable across
            # processes, so this is canonicalisation and defence in depth rather
            # than a live bug fix: it makes order-independence a structural fact
            # instead of an implementation detail, keeps serialised profiles
            # byte-stable, and survives a future move to string identifiers.
            "phenotype_ids": sorted(prof["phenotype_ids"]),
            "gene_ids": sorted(prof["gene_ids"]),
        }
        for d_idx, prof in profiles.items()
    }


def _generate_partition(
    diseases: Sequence[Tuple[int, Dict[str, Any]]],
    count: int,
    id_prefix: str,
    seed: int,
    *,
    min_phenotypes: int,
    max_phenotypes: int,
    phenotype_drop_rate: float,
) -> List[Dict[str, Any]]:
    """One cohort, from one partition, with coverage made true by construction.

    **A budget alone does not guarantee coverage.** Drawing ``count`` diseases
    with replacement can miss one at any budget, so every allocated disease is
    emitted once first and only the remainder is drawn. Refusing a budget smaller
    than the partition is the other half: without it the guarantee is unachievable
    rather than merely unmet.

    **Each partition gets its own patient-id namespace.** A shared
    ``sim_patient_%06d`` counter restarts per call, so two partitions generated
    separately would both begin at zero and collide.

    The order is shuffled before ids are assigned, so file order carries no
    signal about which pass produced a record.
    """
    if not diseases:
        if count:
            raise ValueError(
                f"cannot generate {count} {id_prefix} samples from no diseases"
            )
        return []
    # A budget of zero against a non-empty partition falls through to the coverage
    # check below and is refused there. An early return on ``count == 0`` used to
    # skip it, so a partition with diseases and no budget produced nothing while
    # the run still looked successful.
    if count < len(diseases):
        raise ValueError(
            f"{id_prefix} budget of {count} cannot cover {len(diseases)} allocated "
            "diseases; every allocated disease must receive at least one sample"
        )

    rng = derive_stream(seed, id_prefix)
    picks = list(diseases) + [
        rng.choice(diseases) for _ in range(count - len(diseases))
    ]
    rng.shuffle(picks)

    samples: List[Dict[str, Any]] = []
    for index, (disease_idx, profile) in enumerate(picks):
        all_phenos = profile["phenotype_ids"]
        n_keep = retained_phenotype_count(
            len(all_phenos), min_phenotypes, max_phenotypes, phenotype_drop_rate
        )
        sample: Dict[str, Any] = {
            "patient_id": f"sim_{id_prefix}_{index:06d}",
            "phenotype_ids": rng.sample(all_phenos, n_keep),
            "disease_id": disease_idx,
        }
        if profile["gene_ids"]:
            sample["gene_ids"] = profile["gene_ids"]
        samples.append(sample)
    return samples


