#!/usr/bin/env python
"""
Import a data version: preview what it would hold, then commit it to the registry.
==================================================================================
Contract M4 (`docs/working/PLAN_TEST_RESULTS.md` D6, D7). Three kinds
(`--kind`), each published as an immutable version in a registry the operator
keeps outside the repository (`src/evaluation/cohort_registry.py`):

- ``supplied`` (the default): a source case file (`shepherd-cases/1`) mapped
  against the graph a workspace manifest binds, a test cohort version;
- ``kg_edges`` (research comparison §5 Stage 4): a KG edge set, one part
  (``--role train|val|test``) of the edge split a checkpoint's model record
  names (``--from-checkpoint``), rebuilt and checked against the workspace's
  graph, with its fixed link-prediction pairs. Its role is recorded as given;
- ``generated`` (research comparison §3 (A)): one of the workspace's own
  generated patient splits (``--role train|val|test``, the split it is),
  verified against the workspace's split manifest as ``measure_scorer --split``
  verifies it, its samples and that split manifest kept as written, and the
  workspace's functional gene fixture of that split with them when it holds
  one of its build (research comparison Stage 5b). Its role is recorded as
  given; the generator writes ``train`` and ``val``, so ``test`` is refused
  where the workspace holds no generated test split.

**Two steps, and the second repeats the first.** Without ``--commit`` this
prints what the import would produce -- for a supplied cohort every exclusion
with its case id and reasons, the terms that did not map, the graph and
dataset record it binds, the policy; for a KG edge set the view, the split,
the pairs and their counts, the message-passing graph and what it binds; for a
generated split the generation definition, the split manifest, the split's
files and the samples by digest, their counts, and the functional gene
fixture kept or why none is -- and a preview id, and writes nothing.
``--commit PREVIEW_ID`` with the same arguments reads every input again and
publishes only if they still give that id, so what is committed is what the
operator saw.

**What it prints is for the machine it runs on** (D7): case ids appear, as the
operator needs them to check the exclusions; no case's phenotypes or disease are
printed, only aggregate counts and the terms that did not map. A refusal of a
malformed source names each problem by case index, case id, field and position
and says what type the value was or which rule it broke, never the value.

Usage:
    python scripts/import_cohort.py \\
        --registry /srv/shepherd/cohorts --data-dir data/workspaces/<ws> \\
        --source /srv/intake/cases.json --label mygene2-2026-10 --role mygene2

    # then, with the same arguments, the id the preview printed:
    python scripts/import_cohort.py ... --commit 0123456789abcdef

    # a KG edge set: the test part of the split a base was trained on
    python scripts/import_cohort.py --kind kg_edges \\
        --registry /srv/shepherd/cohorts --data-dir data/workspaces/<ws> \\
        --from-checkpoint data/workspaces/<ws>/runs/gatv2/<run>/checkpoints/best.pt \\
        --label kg-test-2026-11 --role test

    # a generated split: the workspace's own val split, kept as written
    python scripts/import_cohort.py --kind generated \\
        --registry /srv/shepherd/cohorts --data-dir data/workspaces/<ws> \\
        --label gen-val-2026-11 --role val

Module: scripts/import_cohort.py
"""
from __future__ import annotations

import argparse
import logging
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.cohort_registry import (  # noqa: E402
    DEFAULT_PARTIAL_PHENOTYPES,
    EXCLUSION_CATEGORIES,
    GENERATED,
    GENERATED_FIXTURE_COUNT,
    GENERATED_ROLES,
    KG_EDGES,
    KG_EDGES_ROLES,
    PARTIAL_PHENOTYPE_POLICIES,
    SOURCE_FORMAT,
    SUPPLIED,
    VERSION_KINDS,
    ImportPlan,
    commit_import,
    prepare_generated_import,
    prepare_import,
    prepare_kg_edges_import,
)

#: How many of the unmappable terms the preview lists by name, most frequent
#: first. All of them are recorded in the version manifest.
TOP_UNMAPPABLE_TERMS = 10


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview, then commit, a data version: a test cohort mapped to a workspace "
                    "graph, a KG edge set taken from a checkpoint's edge split, or one of a "
                    "workspace's generated splits"
    )
    parser.add_argument("--kind", choices=VERSION_KINDS, default=SUPPLIED,
                        help="supplied (the default): a case file mapped to the workspace's "
                             "graph. kg_edges: a KG edge set, one part of the edge split a "
                             "checkpoint was trained on, with its fixed link-prediction "
                             "pairs. generated: one of the workspace's own generated splits, "
                             "its samples kept as written.")
    parser.add_argument("--registry", type=Path, required=True,
                        help="The cohort registry: an existing directory, given as an "
                             "absolute path outside the repository. It holds "
                             "patient-level data.")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="The workspace whose graph the cases are mapped to, whose "
                             "graph the KG edge set is taken from, or whose generated split "
                             "is kept.")
    parser.add_argument("--source", type=Path, default=None,
                        help=f"--kind supplied: the case file, in the {SOURCE_FORMAT} format. "
                             "Its bytes are kept verbatim in the registry.")
    parser.add_argument("--from-checkpoint", type=Path, default=None,
                        help="--kind kg_edges: the checkpoint whose model record names the "
                             "graph view rule and edge split to take the part from, trained "
                             "on this workspace's graph. Nothing of it is copied.")
    parser.add_argument("--label", required=True,
                        help="The version's label: a lowercase identifier, unique in the "
                             "registry, never reused.")
    parser.add_argument("--role", required=True,
                        help="--kind supplied: the cohort's role, such as mygene2, a "
                             "lowercase identifier, not train or val. --kind kg_edges: the "
                             f"part of the split, {', '.join(KG_EDGES_ROLES)}, recorded as "
                             "given. --kind generated: the generated split, "
                             f"{', '.join(GENERATED_ROLES)}, recorded as given.")
    parser.add_argument("--partial-phenotypes", choices=PARTIAL_PHENOTYPE_POLICIES,
                        default=None,
                        help="--kind supplied: a case some of whose terms do not map: keep "
                             "it, ranked on the terms that map, or exclude it (default "
                             f"{DEFAULT_PARTIAL_PHENOTYPES}). Recorded in the version.")
    parser.add_argument("--commit", metavar="PREVIEW_ID", default=None,
                        help="Publish the version, if the inputs still give this preview "
                             "id. Without it nothing is written.")
    args = parser.parse_args(argv)
    _check_kind_arguments(parser, args)
    return args


def _check_kind_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Each kind's arguments, and none of another's (argparse's exit 2 with
    the reason). A supplied import needs ``--source``; a KG edge set needs
    ``--from-checkpoint`` and takes no source or phenotype policy; a generated
    split takes none of the three: it is the workspace's own. The role's rule
    is the registry's, said by its refusal."""
    if args.kind == SUPPLIED:
        if args.from_checkpoint is not None:
            parser.error("--from-checkpoint applies to --kind kg_edges only")
        if args.source is None:
            parser.error("--source is required for --kind supplied (the default)")
        if args.partial_phenotypes is None:
            args.partial_phenotypes = DEFAULT_PARTIAL_PHENOTYPES
        return
    given = [flag for flag, value in (("--source", args.source),
                                      ("--partial-phenotypes", args.partial_phenotypes))
             if value is not None]
    if args.kind == GENERATED:
        if args.from_checkpoint is not None:
            given.append("--from-checkpoint")
        if given:
            parser.error(f"{' and '.join(given)} {'do' if len(given) > 1 else 'does'} not apply "
                         "to --kind generated: a generated split is the workspace's own")
        return
    if given:
        parser.error(f"{' and '.join(given)} {'apply' if len(given) > 1 else 'applies'} to "
                     "--kind supplied only; a KG edge set is taken from --from-checkpoint")
    if args.from_checkpoint is None:
        parser.error("--from-checkpoint is required for --kind kg_edges")


def _commit_command(args: argparse.Namespace, preview: str) -> str:
    if args.kind == KG_EDGES:
        parts = [
            "python", "scripts/import_cohort.py", "--kind", KG_EDGES,
            "--registry", str(args.registry), "--data-dir", str(args.data_dir),
            "--from-checkpoint", str(args.from_checkpoint), "--label", args.label,
            "--role", args.role, "--commit", preview,
        ]
    elif args.kind == GENERATED:
        parts = [
            "python", "scripts/import_cohort.py", "--kind", GENERATED,
            "--registry", str(args.registry), "--data-dir", str(args.data_dir),
            "--label", args.label, "--role", args.role, "--commit", preview,
        ]
    else:
        parts = [
            "python", "scripts/import_cohort.py",
            "--registry", str(args.registry), "--data-dir", str(args.data_dir),
            "--source", str(args.source), "--label", args.label, "--role", args.role,
            "--partial-phenotypes", args.partial_phenotypes, "--commit", preview,
        ]
    return " ".join(shlex.quote(part) for part in parts)


def _entry(entry: Optional[Dict[str, Any]]) -> str:
    if entry is None:
        return "not recorded"
    return f"digest {entry['digest']}, declared version {entry['declared_version'] or 'none'}"


def _shown_id(case_id: str) -> str:
    """A case id as the preview prints it: as it is when it is printable ASCII,
    escaped (`ascii`) otherwise.

    `parse_source` already refuses an id that is not printable text; this is
    the second line, so an id with a newline can never forge a preview line and
    one the console cannot encode never ends the preview in a traceback."""
    return case_id if case_id.isascii() and case_id.isprintable() else ascii(case_id)


def format_preview(plan: ImportPlan) -> List[str]:
    """The preview, as lines: counts with their denominators, each exclusion by
    case id with its reasons, the unmappable terms, and what the version binds.
    A KG edge set's is `format_kg_edges_preview`, a generated split's
    `format_generated_preview`."""
    if plan.kind == KG_EDGES:
        return format_kg_edges_preview(plan)
    if plan.kind == GENERATED:
        return format_generated_preview(plan)
    manifest = plan.manifest
    counts, source, record = manifest["counts"], manifest["source"], manifest["dataset_record"]
    lines = [
        f"Cohort version preview: label {plan.label}, role {plan.role}",
        f"  source            {source['filename']} ({source['format']}), "
        f"sha256 {source['sha256']}",
        f"  source cases      {counts['source_cases']}",
        f"  ranked-eligible   {counts['ranked_eligible']} of {counts['source_cases']}",
        f"  excluded          {counts['n_excluded']} of {counts['source_cases']}",
    ]
    for category in EXCLUSION_CATEGORIES:
        entries = [e for e in manifest["exclusions"] if e["category"] == category]
        lines.append(f"    {category}: {counts['excluded'][category]}")
        for entry in entries:
            lines.append(
                f"      {_shown_id(entry['case_id'])}: {', '.join(entry['reasons'])} "
                f"({entry['terms_unmappable']} of {entry['terms_total']} terms unmappable)"
            )
    terms = manifest["unmappable_terms"]
    lines.append(f"  terms unmappable  {counts['terms_unmappable']} distinct")
    top = sorted(terms.items(), key=lambda item: (-item[1], item[0]))[:TOP_UNMAPPABLE_TERMS]
    for term, n in top:
        lines.append(f"    {term}: listed by {n} case(s)")
    if len(terms) > len(top):
        lines.append(f"    ... {len(terms) - len(top)} more, all recorded in the version")
    lines += [
        f"  cases with lost terms  {counts['cases_with_lost_terms']} of "
        f"{counts['ranked_eligible']} ranked (scored on fewer terms than listed)",
        f"  repeats removed   {counts['repeats_removed']} in {counts['cases_with_repeats']} "
        f"case(s) (phenotype normalisation v"
        f"{manifest['mapping']['phenotype_normalisation_version']})",
        f"  partial phenotypes  {manifest['mapping']['partial_phenotypes']}",
    ]
    genes = counts.get("genes")
    if genes:
        lines += [
            f"  gene identifiers  {genes['identifiers_listed']} listed, "
            f"{genes['identifiers_unmappable']} name no gene of the graph, "
            f"{genes['identifiers_repeated']} repeat an earlier one",
            f"  cases with genes  {genes['cases_with_causal_genes']} with causal genes "
            f"({genes['cases_with_unmapped_causal_genes']} with some unmapped), "
            f"{genes['cases_with_candidate_genes']} with candidate genes",
        ]
    lines += [
        "  graph (the workspace manifest's digests):",
    ]
    lines += [f"    {role:14s} {digest}" for role, digest in manifest["graph"].items()]
    lines += [
        f"  dataset record    origin {record['origin']}, sha256 {record['sha256']}",
        f"    HPO             {_entry(record['hpo'])}",
        f"    MONDO           {_entry(record['mondo'])}",
    ]
    if record["origin"] == "synthetic":
        lines.append("  SYNTHETIC WORKSPACE: this graph was not built from real files.")
    return lines


def _graph_lines(manifest: Dict[str, Any]) -> List[str]:
    lines = ["  graph (the workspace manifest's digests):"]
    return lines + [f"    {role:14s} {digest}" for role, digest in manifest["graph"].items()]


def format_kg_edges_preview(plan: ImportPlan) -> List[str]:
    """A KG edge set's preview, as lines: where its view and split were taken
    from, the view, the split's parts, the counts with their denominators and
    units, the pair rule, the pairs file, the message-passing graph (and, for
    role train, that the measured edges are in it), and what the version binds.
    Aggregates only: no edge is printed."""
    manifest = plan.manifest
    counts, pairs, split = manifest["counts"], manifest["pairs"], manifest["split"]
    view, taken, record = manifest["graph_view"], manifest["taken_from"], manifest["dataset_record"]
    positive = counts["positive_edges"]
    parts = ", ".join(f"{part} {n}" for part, n in split["edges"].items())
    batches = f"{pairs['batches']} batch{'es' if pairs['batches'] != 1 else ''}"
    passing = manifest["message_passing"]
    inside = (f"which include the {positive} measured edges (role train: nothing is removed)"
              if passing["includes_measured_edges"] else "the measured edges are not in it")
    lines = [
        f"KG edge set preview: label {plan.label}, role {plan.role}",
        f"  taken from        checkpoint sha256 {taken['checkpoint_sha256']} (its "
        f"model record's {' and '.join(taken['record_fields'])})",
        f"  graph view        rule {view['rule']['rule']}, digest {view['digest']}",
        f"  edge split        rule {split['rule']}, seed {split['seed']}, digest {split['digest']}",
        f"    edges by part   {parts} (directed graph-view edges)",
        f"  positive edges    {positive} (the {plan.role} part)",
        f"  pairs             {counts['pairs']} of {positive} positive edges, each with one "
        "negative tail",
        f"  without negative  {counts['edges_without_negative']} of {positive} (no other tail "
        "of their relation in their batch; left out, counted)",
        f"  pair rule         {pairs['rule']}, seed {pairs['seed']}, batches of "
        f"{pairs['batch_size']} ({batches}): base validation's rule",
        f"  pairs file        {pairs['file']}, {pairs['rows']} rows, sha256 {pairs['sha256']}",
        f"  message passing   the split's train part: {passing['edges']} directed edges; {inside}",
        *_graph_lines(manifest),
    ]
    return lines + _record_lines(record)


def _record_lines(record: Dict[str, Any]) -> List[str]:
    """The dataset record a KG edge set or a generated split binds, with every
    source it names, by role."""
    lines = [f"  dataset record    origin {record['origin']}, sha256 {record['sha256']}"]
    sources = record["sources"]
    if not sources:
        lines.append("    sources         none named")
    for entry in sources:
        lines.append(f"    {entry['role']:15s} {_entry(entry)}")
    if record["origin"] == "synthetic":
        lines.append("  SYNTHETIC WORKSPACE: this graph was not built from real files.")
    return lines


def format_generated_preview(plan: ImportPlan) -> List[str]:
    """A generated split's preview, as lines: the split and its counts with
    their units (counted from the samples verified), the generation
    definition and the allocation it cut, the split manifest kept and the
    split's files by digest, the samples kept, what was verified, and what
    the version binds. Aggregates only: no sample or
    patient id is printed."""
    manifest = plan.manifest
    counts, generation = manifest["counts"], manifest["generation"]
    allocation, verified = manifest["allocation"], manifest["verified"]
    settings = ", ".join(f"{key} {value}" for key, value in sorted(generation.items())
                         if key not in ("algorithm", "algorithm_version"))
    lines = [
        f"Generated split preview: label {plan.label}, role {plan.role}",
        f"  samples           {counts['samples']} generated patients of the {plan.role} split",
        f"  diseases          {counts['diseases']} distinct among them (the split's realised "
        "set)",
        f"  generation        {generation['algorithm']} v{generation['algorithm_version']}"
        + (f" ({settings})" if settings else ""),
        f"  allocation        {allocation.get('algorithm')} "
        f"v{allocation.get('algorithm_version')}, seed {allocation.get('seed')}",
        f"  split manifest    {manifest['split_manifest']['file']}, sha256 "
        f"{manifest['split_manifest']['sha256']} (the workspace's file, as written)",
    ]
    lines += [f"    {name:15s} {digest}" for name, digest in manifest["split_artifacts"].items()]
    lines += [
        f"  samples kept      {manifest['samples']['file']}, {manifest['samples']['n']} rows, "
        f"sha256 {manifest['samples']['sha256']} (the workspace's file, as written)",
        f"  verified          {' and '.join(verified['splits'])} against the split manifest"
        + ("; disjoint diseases measured" if verified["disjointness_measured"] else ""),
        *_fixture_lines(plan),
        *_graph_lines(manifest),
    ]
    return lines + _record_lines(manifest["dataset_record"])


def _fixture_lines(plan: ImportPlan) -> List[str]:
    """A generated split's functional gene fixture in its preview: kept, its
    rule, seed and derivation counts and the two files kept by digest; or not
    kept, why (the plan's note), and what that leaves unmeasured. Aggregates
    only."""
    fixture = plan.manifest.get("task_fixture")
    if fixture is None:
        return [f"  gene fixture      none kept ({plan.task_fixture_note}); the gene and "
                "patients-like-me measurements are refused on this version for that data gap"]
    derivation = ", ".join(f"{key} {value}" for key, value in
                           sorted((fixture["derivation"] or {}).items()))
    return [
        f"  gene fixture      the workspace's {fixture['label']}, rule {fixture['rule']}, seed "
        f"{fixture['seed']}, {fixture['distractors']} distractors"
        + (f" ({derivation})" if derivation else ""),
        f"    record kept     {fixture['record']['file']}, sha256 {fixture['record']['sha256']} "
        "(the workspace's file, as written)",
        f"    rows kept       {fixture['samples']['file']}, {fixture['samples']['n']} rows, "
        f"sha256 {fixture['samples']['sha256']} (the workspace's {plan.role} file, as written)",
    ]


def _nothing_to_commit(plan: ImportPlan) -> Optional[str]:
    """Why the preview offers no commit, or None: nothing would be measured."""
    manifest = plan.manifest
    if plan.kind == KG_EDGES:
        if manifest["counts"]["pairs"]:
            return None
        return (f"no edge of the {plan.role} part has a negative tail, so there is nothing to "
                "commit.")
    if plan.kind == GENERATED:
        if manifest["samples"]["n"]:
            return None
        return f"the {plan.role} split holds no sample, so there is nothing to commit."
    if manifest["samples"]["n"]:
        return None
    return "no case is eligible for ranking against this graph, so there is nothing to commit."


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    try:
        if args.kind == KG_EDGES:
            plan = prepare_kg_edges_import(args.registry, args.data_dir, args.from_checkpoint,
                                           args.label, args.role)
        elif args.kind == GENERATED:
            plan = prepare_generated_import(args.registry, args.data_dir, args.label, args.role)
        else:
            plan = prepare_import(args.registry, args.data_dir, args.source, args.label,
                                  args.role, args.partial_phenotypes)
    except OSError as exc:
        print(f"import_cohort: refused: {exc}. Nothing was written.", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"import_cohort: refused: {exc} Nothing was written.", file=sys.stderr)
        return 1

    if args.commit is None:
        print("\n".join(format_preview(plan)))
        print(f"\nPreview id: {plan.preview_id}")
        why = _nothing_to_commit(plan)
        if why is not None:
            print(f"import_cohort: refused: {why} Nothing was written.", file=sys.stderr)
            return 1
        print("Nothing was written. To publish this version, run:")
        print(f"  {_commit_command(args, plan.preview_id)}")
        return 0

    try:
        version = commit_import(plan, args.registry, args.commit)
    except OSError as exc:
        print(f"import_cohort: failed while writing: {exc}. The version was not committed "
              "(cohort.json is written last); a version directory left without it is an "
              "interrupted import, refused by every reader. Remove it before reusing the "
              "label.", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"import_cohort: refused: {exc}", file=sys.stderr)
        return 1
    if version.kind == KG_EDGES:
        counts = version.manifest["counts"]
        print(f"Committed KG edge set {plan.label} ({plan.role}): {counts['pairs']} pairs of "
              f"{counts['positive_edges']} positive edges")
    elif version.kind == GENERATED:
        counts = version.manifest["counts"]
        diseases = f"{counts['diseases']} disease{'s' if counts['diseases'] != 1 else ''}"
        fixture = counts.get(GENERATED_FIXTURE_COUNT)
        kept = ("no functional gene fixture" if fixture is None
                else f"its functional gene fixture ({fixture} rows)")
        print(f"Committed generated split {plan.label} ({plan.role}): {counts['samples']} "
              f"generated patients over {diseases}, with {kept}")
    else:
        print(f"Committed cohort version {plan.label} (role {plan.role}): "
              f"{version.manifest['samples']['n']} ranked of "
              f"{version.manifest['counts']['source_cases']} source cases")
    print(f"  directory        {version.directory}")
    print(f"  manifest sha256  {version.identity.sha256}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
