#!/usr/bin/env python
"""
Export every kept and recorded result as CSV or JSON: the results service's command line.
=========================================================================================
Research comparison §5 Stage 8 (owner's O6; plan review F3, E3). A thin client
of the one read-only results service (`src/api/services/results.py`): it names
the sources, the service reads them, and this writes the service's long form
(one row per result and metric) as CSV or JSON. Nothing is computed here, and
nothing is written but the export.

**Run as the operator on this machine.** It reads the directories you name
with this account's rights, never through the server: there is no request, so
D8 does not apply, and nothing here is offered over the network. Name each
source explicitly:

- ``--workspaces-root [NAME=]DIR``, as many as needed: a directory whose
  subdirectories are workspaces (what ``SHEPHERD_WORKSPACES_ROOT`` names for the
  service). Every checkpoint of every workspace is listed, and every ledger of
  every workspace read, whatever became of the weights it names (an entry
  that is not a workspace the server can use is a row with its reason);
- ``--registry [NAME=]DIR``, as many as needed: a cohort registry (what
  ``SHEPHERD_REGISTRY`` names). Every version is listed and every report its
  store keeps read.

Each is read with the rule the service reads its own setting by (absolute,
existing; a registry outside the repository), as a source of its own; this
process's ``SHEPHERD_*`` settings are not read or changed for them. ``NAME``
is what rows show for the source (never its path or the directory's own name,
which may say more than an export should); by default a fixed word and the
source's place among its kind (``workspaces-root-1``, ``registry-1``, ...), and
two sources may not share one.

**What it holds.** Every result: a kept report and a ledger record are one row
only when the record names that report's run and bytes; a ledger record whose
report is in no named store reads "report not available"; a record merged
before each run had its own record is its own row, never counted as a run; an
item that cannot be read is a row with its reason, and hides nothing. Each
row lists what its records say (the model's identity, lineage, spec summary,
structure and training settings, the data's identity and role, the
measurement's settings and protocol, the environment, the data's relation to
the model's training: a supplied cohort's R10, a KG edge set's exposure, the
states and content references), and ``unknown`` names what has no source,
with why. Measurement
aggregates only: no case id, no ranks, no server path. No row is capped and
none is ordered by a metric; filters keep the rows that match (an unreadable
item is always kept).

Usage:
    python scripts/export_results.py --workspaces-root /srv/shepherd/workspaces \\
        --registry /srv/shepherd/registry --format csv --output results.csv

    python scripts/export_results.py --workspaces-root lab=/data/lab/workspaces \\
        --workspaces-root archive=/mnt/archive/workspaces --registry /srv/registry \\
        --protocol kg_link_prediction --role test --format json

Exit status: 0 when the export is written (an item that could not be read is a
row of it, and is counted on stderr); 2 for a usage error, a source name given
twice, or an output file that already exists (it is never overwritten). The
file is written only once the whole export is encoded, and appears whole or
not at all.

Module: scripts/export_results.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.services import results  # noqa: E402

#: ``NAME=DIR``: a name that is no path (letters, digits, ``_ . -``), then the
#: directory.
_NAMED = re.compile(r"([A-Za-z0-9][A-Za-z0-9_.-]*)=(.+)")


def _named(texts: Sequence[str], word: str) -> List[Tuple[str, str]]:
    """``(name, directory)`` for each ``[NAME=]DIR`` of one kind. A directory
    given alone is named by `word` and its place among them (``registry-2``),
    never by its own name: a directory's name may say more than an export
    passed around should (D7)."""
    out = []
    for place, text in enumerate(texts, start=1):
        found = _NAMED.fullmatch(text)
        out.append((found.group(1), found.group(2)) if found is not None
                   else (f"{word}-{place}", text))
    return out


#: A filter's help where "keep the results whose <name> is VALUE" would say
#: less than it does.
_FILTER_HELP = {
    "workspace": ("keep the results found in a workspace named VALUE (a listed checkpoint "
                  "of their model or a ledger holding their record); the name is matched "
                  "in every named workspaces root, and each row's locations name the root "
                  "it was found in"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Export every kept and recorded result of the named workspaces roots and "
                     "registries as CSV or JSON, one row per result and metric. Runs as the "
                     "operator on this machine, reading the named directories with this "
                     "account's rights; nothing is offered over the network and nothing but "
                     "the export is written."),
        epilog=("Measurement aggregates only: no case id, rank or server path. An item that "
                "cannot be read is a row with its reason; unknown names each field without a "
                "source. No row is capped or ordered by a metric."))
    parser.add_argument("--workspaces-root", action="append", default=[], metavar="[NAME=]DIR",
                        help="a directory whose subdirectories are workspaces, as "
                             "SHEPHERD_WORKSPACES_ROOT names one for the service (repeatable); "
                             "NAME is what rows show for it, never its path; by default "
                             "workspaces-root-1, workspaces-root-2, ... in the order given")
    parser.add_argument("--registry", action="append", default=[], metavar="[NAME=]DIR",
                        help="a cohort registry, as SHEPHERD_REGISTRY names one for the service "
                             "(repeatable); NAME as above, by default registry-1, registry-2, ...")
    for name in results.FILTERS:
        parser.add_argument(f"--{name.replace('_', '-')}", action="append", default=[],
                            dest=name, metavar="VALUE",
                            help=_FILTER_HELP.get(name, f"keep the results whose "
                                                        f"{name.replace('_', ' ')} is VALUE")
                            + " (repeatable: any of the values)")
    parser.add_argument("--format", choices=("csv", "json"), required=True,
                        help="the long form as CSV (one header row) or JSON")
    parser.add_argument("--output", type=Path, default=None,
                        help="the file to write; it must not exist (never overwritten). "
                             "Default: standard output")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.workspaces_root and not args.registry:
        parser.error("name at least one --workspaces-root or --registry")
    try:
        sources = results.explicit_sources(
            workspaces_roots=_named(args.workspaces_root, "workspaces-root"),
            registries=_named(args.registry, "registry"))
    except ValueError as exc:
        parser.error(f"{exc}; give each a NAME= of its own")
    found = results.filtered(results.query(sources),
                             {name: getattr(args, name) for name in results.FILTERS})
    text = results.to_csv(found) if args.format == "csv" else results.to_json(found)
    # The whole export is encoded before anything is written, and the file is
    # published whole (`publish_bytes`, never replacing one): a failure leaves
    # no file, never a part of one (review fix RC-R1).
    data = text.encode("utf-8")
    if args.output is None:
        sys.stdout.write(text)
    else:
        from src.utils.file_modes import publish_bytes

        try:
            publish_bytes(args.output, data, replace=False)
        except FileExistsError:
            print(f"{args.output} exists; it is never overwritten. Nothing was written.",
                  file=sys.stderr)
            return 2
    unreadable: List[dict] = [row for row in found if row["result"] == results.UNREADABLE]
    print(f"{len(found) - len(unreadable)} result(s) exported; {len(unreadable)} item(s) could "
          "not be read and are rows with their reasons.", file=sys.stderr)
    for row in unreadable:
        print(f"  {row['where'].get('source')}: {row['item']}: {row['reason']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
