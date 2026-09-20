"""Producing a workspace: the one sequence that produces one.

A workspace is not a directory that happens to contain the right filenames. It
is a **single production event** — a graph cut into a disease-level allocation,
exported to tensors, and generated into cohorts — recorded by a manifest that
binds all six artifacts to each other by digest. Every consumer verifies that
binding: training, measurement, and the clinical inference path all refuse a
directory whose parts came from different events.

That makes the ordering load-bearing, and the ordering is easy to get wrong in a
way nothing detects until much later:

  1. the allocation is cut from the in-memory graph, and every refusal that can
     be decided from it happens **before the first byte is written**;
  2. `kg.json`, then the three tensors, are written;
  3. their digests are taken **by the writer**, from the files it just wrote;
  4. the cohorts are generated and the manifest records all six together.

Steps 3 and 4 are why this cannot be assembled by a caller from the outside: a
generator that hashed the artifacts itself would be digesting whatever files
happen to sit in the directory, and recording them beside this allocation's
universe digest as though they were one chain.

**This module exists so there is exactly one implementation of that sequence.**
It used to live inside `build_knowledge_graph.py`, which made "the production
path" a property of one script rather than of the project: `setup_demo.py` wrote
a workspace too, with its own sample generator that split patients rather than
diseases — the very defect this branch exists to remove — and no manifest at
all. It produced a directory that looked right, trained, and then could not be
served, because the verifier correctly refused it. Deleting that second producer
was not the fix; having one producer is.

Module: src/kg/workspace.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional

from src.kg.graph import DEFAULT_FEATURE_SEED
from src.kg.sample_generator import (
    validate_phenotype_count,
    validate_sample_budgets,
)

logger = logging.getLogger(__name__)


class SampleBudget(NamedTuple):
    """How many patients to simulate, and how to cut the diseases they come from.

    ``val_disease_fraction`` is a *disease* fraction, not a patient fraction:
    the partitions are disjoint at the disease level by construction, which is
    the property a patient-level slice cannot provide (EVALUATION_COHORTS §6.2).
    """

    num_train: int
    num_val: int
    val_disease_fraction: float = 0.15
    seed: int = 42
    min_phenotypes: int = 2


class WorkspaceRefusal(ValueError):
    """An input this writer will not act on, raised before it writes anything.

    Distinct from a plain ``ValueError`` for one reason: a caller can say
    "nothing was written" and be right. Every other failure inside
    ``write_workspace`` can happen after the graph is on disk, so a caller that
    caught them all would be attaching a true sentence to a false claim.

    It covers every input whose domain is knowable up front — the two budgets,
    their coverage of the allocation, the phenotype floor that decides
    eligibility, and the feature width the export uses.
    """


def require_budget_coverage(
    num_train: int,
    num_val: int,
    allocation: Any,
    *,
    train_label: str = "num_train",
    val_label: str = "num_val",
) -> None:
    """Every allocated disease must receive at least one sample.

    **One implementation, two vocabularies.** A command-line tool wants to name
    its flags and a library caller wants to name its arguments, but the
    comparison is the same one and a second copy of it would be a second thing
    to keep true. The labels are the only part the caller supplies.
    """
    for budget, partition, label in (
        (num_train, allocation.train, train_label),
        (num_val, allocation.val, val_label),
    ):
        if budget < len(partition):
            raise WorkspaceRefusal(
                f"{label}={budget} cannot cover {len(partition)} allocated "
                "diseases; every allocated disease must receive at least one "
                "sample."
            )


class WorkspaceWrite(NamedTuple):
    """What one production event produced."""

    graph_digests: Dict[str, str]
    allocation: Optional[Any]
    train_samples: List[Dict[str, Any]]
    val_samples: List[Dict[str, Any]]
    manifest: Optional[Dict[str, Any]]
    #: SHA-256 of the `kg.provenance.json` this build wrote. Always produced,
    #: including for a graph-only write, because the graph is what the sources
    #: made and it exists on both paths.
    provenance_digest: Optional[str] = None


def write_workspace(
    kg: Any,
    workspace: Path,
    *,
    feature_dim: int = 128,
    feature_seed: int = DEFAULT_FEATURE_SEED,
    samples: Optional[SampleBudget] = None,
    train_label: str = "num_train",
    val_label: str = "num_val",
    sources: Optional[List[Dict[str, Any]]] = None,
    source_counters: Optional[Dict[str, Any]] = None,
) -> WorkspaceWrite:
    """Write `kg.json`, the graph tensors, and — when asked — bound cohorts.

    Args:
        kg: the finished in-memory KnowledgeGraph. Nothing here builds or edits
            it; the caller owns what goes into the graph.
        workspace: the directory to write. Created if absent.
        feature_dim: node feature width for the export.
        feature_seed: the seed the node features are drawn from. Fixed by
            default so a rebuild reproduces the workspace; name another to get a
            different initialisation. Unrelated to `samples.seed`, which cuts the
            allocation — independent streams, so changing one cannot move the
            other.
        samples: when given, cut an allocation and generate cohorts with a
            `split_manifest.json`. When omitted, only the graph is written.
            That is a real mode, not a broken one — `compute_shortest_paths.py`
            reads `kg.json`, `build_index.py` reads the tensors, and the API
            serves path-reasoning from `kg.json` alone — but it is **not a
            trainable or GNN-servable workspace**, because nothing records which
            export those tensors are. Completing it means running this function
            again with budgets, not adding a manifest to what is already there:
            only the writer that exported the tensors can vouch for their
            digests.
        train_label / val_label: what to call the two budgets when refusing
            them. **The vocabulary is the only part a caller supplies**; the
            checks themselves are unconditional and belong here. This replaced a
            `preflight` callback that carried the refusal for its caller: once
            the writer's own check became authoritative and therefore had to run
            first, the callback could no longer produce the message an operator
            needed, and a hook with no remaining job is worse than none.

    Returns:
        The digests, the allocation, both cohorts and the manifest.
    """
    from src.kg.artifacts import GRAPH_ARTIFACTS
    from src.kg.sample_generator import refuse_if_checkpoints_exist
    from src.kg.provenance import (
        ProvenanceError,
        build_provenance,
        encode_provenance,
        write_provenance,
    )
    from src.utils.fingerprint import file_sha256

    workspace = Path(workspace)

    # **Every input whose domain is knowable now is checked now.** Each of these
    # is acted on before the one below it, and the last of them runs after
    # `kg.json` is on disk -- so a value rejected at the point of use leaves a
    # workspace half written by an input that was wrong from the start.
    # `feature_dim` decides the export's tensor width; `min_phenotypes` decides
    # which diseases are eligible and so the shape of the allocation itself. The
    # rules are imported, not restated: `validate_feature_dim` is the one the
    # export enforces and `validate_phenotype_count` is the one generation
    # enforces, so neither can drift from what actually runs later.
    from src.kg.disease_allocation import validate_allocation_seed
    from src.kg.graph import validate_feature_dim, validate_feature_seed

    try:
        validate_feature_dim(feature_dim)
        validate_feature_seed(feature_seed)
        if samples is not None:
            validate_phenotype_count("min_phenotypes", samples.min_phenotypes, 1)
            # **The seed is provenance, not only randomness.** `derive_stream`
            # stringifies it so almost anything yields a stream, but the
            # allocation keeps the object and the manifest serialises it -- so a
            # non-JSON seed was discovered by `json.dump` with the graph, both
            # cohorts and half the manifest already written.
            validate_allocation_seed(samples.seed)
    except ValueError as exc:
        raise WorkspaceRefusal(str(exc)) from exc

    # **A workspace under trained checkpoints is not rewritable.** Rebuilding
    # the graph beneath them leaves those checkpoints paired with a graph they
    # were never trained on, and nothing in the checkpoint says so. Callers that
    # can check earlier still should — an ontology load is minutes — but the
    # authoritative refusal belongs to the writer, because it is the writer that
    # would do the damage.
    refuse_if_checkpoints_exist(workspace)

    allocation = None
    if samples is not None:
        from src.kg import allocate_diseases, build_eligible_disease_profiles

        # **The budgets' own domain, before an allocation is even cut.** The
        # same validator the generator uses, so the writer and the library
        # cannot disagree about what a budget is. A `True` or a large float used
        # to pass every comparison below it and be caught only by the generator,
        # with the graph already on disk.
        try:
            validate_sample_budgets(samples.num_train, samples.num_val)
        except ValueError as exc:
            raise WorkspaceRefusal(str(exc)) from exc

        eligible = build_eligible_disease_profiles(
            kg, min_phenotypes=samples.min_phenotypes
        )
        # **Only this call is translated.** Everything it refuses -- a fraction
        # outside (0, 1), a universe too small to cut in two -- it refuses
        # before returning, so nothing has been written and `WorkspaceRefusal`
        # is telling the truth. Wrapping the whole function instead would
        # attach that promise to failures that happen after the graph is saved.
        try:
            allocation = allocate_diseases(
                eligible, samples.val_disease_fraction, seed=samples.seed
            )
        except ValueError as exc:
            raise WorkspaceRefusal(str(exc)) from exc
        # Coverage needs the partition sizes, so it cannot run earlier than
        # this -- but it still runs before the first byte, and nothing runs
        # between it and the allocation it checks.
        require_budget_coverage(
            samples.num_train, samples.num_val, allocation,
            train_label=train_label, val_label=val_label,
        )

    # **The new inputs join the old refusals rather than trailing them.**
    # `sources` and `source_counters` come from a caller and are knowable before
    # anything is written — so a malformed entry or a counter no encoder takes
    # must refuse here, not after `kg.json` and three tensors have been
    # overwritten. Measured on the shape that got this wrong: an unserialisable
    # counter raised `TypeError` with four graph artifacts already written, and
    # over an existing workspace it left new tensors beside the previous
    # manifest.
    #
    # The record is assembled against a placeholder digest purely to prove it
    # encodes; the real one is not known until `kg.json` exists, and the record
    # written below is built again with it. Encoding twice costs nothing and is
    # what makes the refusal structural rather than a list of fields someone
    # re-checks by hand.
    #
    # **`encode_provenance`, not `json.dumps` -- the gate must run the writer's
    # encoder, not an encoder.** These were two calls with different arguments,
    # and they disagreed: a counter keyed `{1: 2, "OMIM": 3}` encodes under the
    # default and fails `sort_keys=True`, so the gate passed and the writer then
    # raised `TypeError` with `kg.json` and three tensors already on disk --
    # precisely the failure the gate exists to prevent, reintroduced by the gate
    # itself. One function now answers for both.
    if sources is not None or source_counters is not None:
        try:
            encode_provenance(build_provenance(
                kg_digest="0" * 64,
                sources=sources,
                counters=source_counters,
                origin="files" if sources is not None else "synthetic",
            ))
        except ProvenanceError as exc:
            raise WorkspaceRefusal(f"the provenance this build would record is unusable: {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise WorkspaceRefusal(
                "the provenance this build would record cannot be serialised "
                f"({type(exc).__name__}: {exc}). Nothing has been written."
            ) from exc

    # Every refusal is behind us; this is the first thing that exists afterwards.
    workspace.mkdir(parents=True, exist_ok=True)
    kg_path = workspace / GRAPH_ARTIFACTS["kg"]
    kg.save_json(str(kg_path))
    logger.info("KG saved to %s", kg_path)

    kg.export_graph_data(
        output_dir=workspace, feature_dim=feature_dim, feature_seed=feature_seed
    )
    logger.info("Graph data exported to %s (feature_dim=%d)", workspace, feature_dim)

    graph_digests = {
        role: file_sha256(workspace / filename)
        for role, filename in GRAPH_ARTIFACTS.items()
    }

    # **Written here, between the graph and the manifest, and on both paths.**
    # The record carries `kg.json`'s digest, so it can only be assembled once
    # that file is final — and it must exist before the manifest, which binds
    # it. A graph-only write returns below without a manifest, which is exactly
    # why the record cannot live in one: the build whose inputs most need naming
    # is the one that produces no manifest at all.
    #
    # **Sources are supplied, never discovered.** Only the caller that opened
    # the ontology and annotation files knows what this graph was made from;
    # scanning a cache afterwards would record whatever is there now, which is
    # a statement about the machine rather than about the build. A caller with
    # no real inputs — a demo, a test, the probe — passes none and the record
    # says `synthetic` rather than inventing digests.
    provenance_digest = write_provenance(
        workspace,
        build_provenance(
            kg_digest=graph_digests["kg"],
            sources=sources,
            counters=source_counters,
            origin="files" if sources is not None else "synthetic",
        ),
    )

    if samples is None:
        return WorkspaceWrite(graph_digests, None, [], [], None, provenance_digest)

    from src.kg.sample_generator import generate_training_samples

    from src.kg.graph import (
        FEATURE_INITIALISATION,
        FEATURE_INITIALISATION_VERSION,
    )

    train_samples, val_samples, manifest = generate_training_samples(
        kg=kg,
        allocation=allocation,
        num_train=samples.num_train,
        num_val=samples.num_val,
        min_phenotypes=samples.min_phenotypes,
        output_dir=workspace,
        graph_digests={**graph_digests, "provenance": provenance_digest},
        # **What the digests cannot say.** They prove these are the bytes this
        # writer exported; they do not say how to make them again. The recipe
        # is what turns "rebuild this workspace" into an instruction.
        graph_export={
            "feature_dim": feature_dim,
            "feature_seed": feature_seed,
            "initialisation": FEATURE_INITIALISATION,
            "initialisation_version": FEATURE_INITIALISATION_VERSION,
        },
    )
    logger.info(
        "Generated %d train samples over %d diseases, %d val over %d — disjoint: %s",
        len(train_samples), manifest["realised"]["train_diseases"],
        len(val_samples), manifest["realised"]["val_diseases"], manifest["disjoint"],
    )
    return WorkspaceWrite(
        graph_digests, allocation, train_samples, val_samples, manifest,
        provenance_digest
    )


__all__ = [
    "WorkspaceRefusal",
    "SampleBudget",
    "WorkspaceWrite",
    "require_budget_coverage",
    "write_workspace",
]
