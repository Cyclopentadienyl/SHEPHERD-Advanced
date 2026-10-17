#!/usr/bin/env python
"""
Import a test cohort: preview what a source maps to, then commit it as a version.
================================================================================
Contract M4 (`docs/working/PLAN_TEST_RESULTS.md` D6, D7). A source case file
(`shepherd-cases/1`) is mapped against the graph a workspace manifest binds and
published as an immutable cohort version in a registry the operator keeps
outside the repository (`src/evaluation/cohort_registry.py`).

**Two steps, and the second repeats the first.** Without ``--commit`` this
prints what the import would produce -- every exclusion with its case id and
reasons, the terms that did not map, the graph and dataset record it binds, the
policy -- and a preview id, and writes nothing. ``--commit PREVIEW_ID`` with the
same arguments reads every input again and publishes only if they still give
that id, so what is committed is what the operator saw.

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
    PARTIAL_PHENOTYPE_POLICIES,
    SOURCE_FORMAT,
    ImportPlan,
    commit_import,
    prepare_import,
)

#: How many of the unmappable terms the preview lists by name, most frequent
#: first. All of them are recorded in the version manifest.
TOP_UNMAPPABLE_TERMS = 10


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview, then commit, a test cohort version mapped to a workspace graph"
    )
    parser.add_argument("--registry", type=Path, required=True,
                        help="The cohort registry: an existing directory, given as an "
                             "absolute path outside the repository. It holds "
                             "patient-level data.")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="The workspace whose graph the cases are mapped to.")
    parser.add_argument("--source", type=Path, required=True,
                        help=f"The case file, in the {SOURCE_FORMAT} format. Its bytes are "
                             "kept verbatim in the registry.")
    parser.add_argument("--label", required=True,
                        help="The version's label: a lowercase identifier, unique in the "
                             "registry, never reused.")
    parser.add_argument("--role", required=True,
                        help="The cohort's role, such as mygene2: a lowercase identifier, "
                             "not train or val.")
    parser.add_argument("--partial-phenotypes", choices=PARTIAL_PHENOTYPE_POLICIES,
                        default=DEFAULT_PARTIAL_PHENOTYPES,
                        help="A case some of whose terms do not map: keep it, ranked on "
                             "the terms that map, or exclude it. Recorded in the version.")
    parser.add_argument("--commit", metavar="PREVIEW_ID", default=None,
                        help="Publish the version, if the inputs still give this preview "
                             "id. Without it nothing is written.")
    return parser.parse_args(argv)


def _commit_command(args: argparse.Namespace, preview: str) -> str:
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
    case id with its reasons, the unmappable terms, and what the version binds."""
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    try:
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
        if not plan.manifest["samples"]["n"]:
            print("import_cohort: refused: no case is eligible for ranking against this "
                  "graph, so there is nothing to commit. Nothing was written.",
                  file=sys.stderr)
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
    print(f"Committed cohort version {plan.label} (role {plan.role}): "
          f"{version.manifest['samples']['n']} ranked of "
          f"{version.manifest['counts']['source_cases']} source cases")
    print(f"  directory        {version.directory}")
    print(f"  manifest sha256  {version.identity.sha256}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
