"""
Write the Stage 1a goldens from the code under test, and build the P0-1 corpus.
===============================================================================
Run from the repository root with the project's interpreter:

    .venv/bin/python -m tests.fixtures.goldens.generate           # every golden
    .venv/bin/python -m tests.fixtures.goldens.generate p0_2      # one
    .venv/bin/python -m tests.fixtures.goldens.generate --corpus  # the corpus, once

Each golden file is `encode` of what `values.py` computes; the tests compute
the same values and compare them with the committed bytes. A golden is
rewritten only to record a change that names it, and a strict one
(`goldens.STRICT`: the P0-1 records and keys, base validation, the supplied
importer's `cohort.json` bytes) never this round: every golden asked for is
computed first, and when one strict golden's bytes would differ, the command
says which and why it is strict, writes nothing and exits 1.

**The P0-1 corpus** (`build_corpus`, written once and never rewritten: a kept
artefact is read as it is) is a registry with two schema-1 versions and the
reports their stores keep, and three ledgers, made with the existing
fixtures and production writers:

- ``mygene2-2026-10`` (role ``mygene2``): the version
  `tests/fixtures/report_store.store_version` writes for the suite's default
  report block, keeping the suite's in-memory reports as the store fixture
  keeps them (`store_report`): a disease task report (`registered_report`, a
  CUDA run with deterministic algorithms on), a second run of it whose
  report differs only in the checkpoint's location, and the kept Mode C (a
  CUDA run with deterministic algorithms off) and Mode D reports
  (`mode_c_report`, `mode_d_report`);
- ``demo-genes-1`` (role ``demo``): a version the supplied importer writes on
  the demo workspace from a source listing genes (`values.gene_source_cases`),
  keeping the reports `scripts/measure_scorer.py` writes when it measures a
  disease, a gene and a patients-like-me child of one base, each trained for
  one epoch on CPU (`trained_children.train_children`), by its protocol;
- ``ledgers/registered.json``: schema 4, every report above registered as its
  own run by the ledger's writer (`append_record`, `write_ledger`);
  ``ledgers/schema3_merged.json``: schema 3 as the writer before K8-6r.2 left
  it, the disease report's record with its second run merged in as a
  corroborating report; ``ledgers/schema3_appended.json``: that file after
  today's writer registered the second run (schema 4 from its boundary).

**Normalised when the corpus was built:** the measured reports' paths (the
checkpoint, the workspace and R10's evidence) under the build's temporary
root are rewritten to `KEPT_ROOT`, and the reports re-encoded as the store
encodes them (`encode_json`); each run directory gets a fixed run id (its
checkpoint's prefix kept); every code revision recorded is
`values.FIXED_CODE_REVISION`; import and registration times are fixed. The
usage history (`usage.jsonl`) and the kept source copies (`sources/`) are not
kept: P0-1 reads neither. Everything else is as written: the environment
facts, checkpoint digests and metrics of the generating machine.

Module: tests/fixtures/goldens/generate.py
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tests.fixtures.goldens import CORPUS, GOLDENS, STRICT, encode, sha256

#: Where the measured reports' paths are said to be, in place of the build's
#: temporary root.
KEPT_ROOT = "/kept/p0-1"

#: The three tasks measured into ``demo-genes-1``, in run order.
TASKS = ("disease", "gene", "patients_like_me")


def _run_id(time: str, checkpoint: str, n: int) -> str:
    """A run directory's name as `create_report_dir` makes one, fixed."""
    return f"{time}-{checkpoint[:12]}-{n:08x}"


def _relocated(value: Any, root: str) -> Any:
    """`value` with every string's `root` prefix replaced by `KEPT_ROOT`."""
    if isinstance(value, dict):
        return {key: _relocated(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_relocated(item, root) for item in value]
    if isinstance(value, str) and value.startswith(root):
        return KEPT_ROOT + value[len(root):]
    return value


def _builder_reports() -> list[tuple]:
    """The suite's in-memory reports kept under ``mygene2-2026-10``: (name,
    report), in run order."""
    from tests.unit.test_evaluation_sidecar import (
        mode_c_report,
        mode_d_report,
        registered_report,
    )

    return [
        ("disease", registered_report()),
        ("disease_second_run",
         registered_report(checkpoint_path="/srv/ws-moved/runs/shepherd/r1/checkpoints/best.pt")),
        ("mode_c", mode_c_report(deterministic_algorithms=False, cudnn_deterministic=False)),
        ("mode_d", mode_d_report()),
    ]


def build_corpus(out: Path) -> None:
    """Build the P0-1 corpus (module docstring) into `out`, which must not
    exist."""
    from src.config.server_settings import ENV_REGISTRY
    from src.evaluation.cohort_registry import encode_json
    from src.evaluation.sidecar import append_record, build_record, empty_ledger, write_ledger
    from src.utils.gpu_lock import WORKSPACES_ROOT_ENV
    from tests.fixtures.goldens.values import GENE_LABEL, fixed_code_revision, gene_source_cases
    from tests.fixtures.registered_cohort import (
        ROLE,
        SOURCE_FILENAME,
        import_version,
        write_demo_workspace,
        write_source,
    )
    from tests.fixtures.report_store import store_report
    from tests.fixtures.trained_children import train_children

    out = Path(out)
    if out.exists():
        raise SystemExit(f"{out} exists: the corpus is kept, never rewritten")
    with tempfile.TemporaryDirectory() as tmp, fixed_code_revision():
        root = Path(tmp).resolve()
        registry = root / "registry"
        registry.mkdir()
        kept: list[tuple] = []  # (name, report path)

        for n, (name, report) in enumerate(_builder_reports(), start=1):
            checkpoint = report["manifest"]["artifact_digests"]["checkpoint"]
            path = store_report(registry, report,
                                run_id=_run_id(f"20261016T12000{n}Z", checkpoint, n))
            kept.append((name, path))

        workspace = write_demo_workspace(root / "workspace")
        trained = train_children(workspace, TASKS)
        source = write_source(root / SOURCE_FILENAME, gene_source_cases(workspace))
        import_version(registry, workspace, source, label=GENE_LABEL, role=ROLE)
        import scripts.measure_scorer as measure_scorer

        lock_root = root / "lock"
        lock_root.mkdir()
        reports = registry / "versions" / GENE_LABEL / "reports"
        saved = {key: os.environ.get(key) for key in (ENV_REGISTRY, WORKSPACES_ROOT_ENV)}
        os.environ.update({ENV_REGISTRY: str(registry), WORKSPACES_ROOT_ENV: str(lock_root)})
        try:
            for n, task in enumerate(TASKS, start=1):
                before = set(reports.iterdir()) if reports.exists() else set()
                with contextlib.redirect_stdout(io.StringIO()):
                    code = measure_scorer.main([
                        "--mode", task, "--checkpoint", str(trained[task]),
                        "--data-dir", str(workspace), "--cohort", GENE_LABEL,
                        "--registry", str(registry), "--device", "cpu"])
                if code != 0:
                    raise SystemExit(f"measuring the {task} child exited {code}")
                (run_dir,) = [p for p in set(reports.iterdir()) - before if p.is_dir()]
                report = _relocated(json.loads((run_dir / "report.json").read_bytes()), str(root))
                (run_dir / "report.json").write_bytes(encode_json(report))
                checkpoint = report["manifest"]["artifact_digests"]["checkpoint"]
                fixed = run_dir.with_name(_run_id(f"20261110T00000{n}Z", checkpoint, n))
                run_dir.rename(fixed)
                kept.append((f"measured_{task}", fixed / "report.json"))
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        ledgers = root / "ledgers"
        ledgers.mkdir()
        records = {}
        ledger = empty_ledger()
        for n, (name, path) in enumerate(kept, start=1):
            data = path.read_bytes()
            records[name] = (build_record(json.loads(data), sha256(data)), path.parent.name)
            outcome, ledger = append_record(ledger, records[name][0], run_id=records[name][1],
                                            registered_at=f"2026-11-10T01:{n:02d}:00Z")
            assert outcome == "registered", (name, outcome)
        write_ledger(ledgers / "registered.json", ledger, None)
        assert len(records) == len(kept), "two kept reports share a name"

        # The second run's report merged into the first's record, as the
        # writer before K8-6r.2 merged equal values: one condition group.
        assert (records["disease"][0]["measurement_semantics_digest"]
                == records["disease_second_run"][0]["measurement_semantics_digest"])
        merged = copy.deepcopy(records["disease"][0])
        merged["corroborating_reports"] = [records["disease_second_run"][0][
            "source_artifact_digest"]]
        merged["registration"] = {"sequence": 1, "registered_at": "2026-10-16T12:30:00Z"}
        schema_3 = {**empty_ledger(), "schema_version": 3, "records": [merged]}
        write_ledger(ledgers / "schema3_merged.json", schema_3, None)
        second, second_run = records["disease_second_run"]
        outcome, appended = append_record(copy.deepcopy(schema_3), second, run_id=second_run,
                                          registered_at="2026-10-18T09:00:00Z")
        assert outcome == "registered", outcome
        write_ledger(ledgers / "schema3_appended.json", appended, None)

        for version in sorted((registry / "versions").iterdir()):
            target = out / "registry" / "versions" / version.name
            target.mkdir(parents=True)
            for name in ("cohort.json", "samples.json"):
                if (version / name).exists():
                    shutil.copyfile(version / name, target / name)
            for run_dir in sorted(p for p in (version / "reports").iterdir() if p.is_dir()):
                shutil.copytree(run_dir, target / "reports" / run_dir.name)
        shutil.copytree(ledgers, out / "ledgers")


def _p0_1(scratch: Path) -> dict[str, bytes]:
    from tests.fixtures.goldens.values import p0_1_values

    records, views = p0_1_values(scratch)
    return {"p0_1_records.json": encode(records), "p0_1_views.json": encode(views)}


def _p0_2(scratch: Path) -> dict[str, bytes]:
    from tests.fixtures.goldens.values import p0_2_values

    return {"p0_2_protocols.json": encode(p0_2_values())}


def _p0_3(scratch: Path) -> dict[str, bytes]:
    from tests.fixtures.goldens.values import p0_3_values

    return {"p0_3_base_validation.json": encode(p0_3_values(scratch))}


def _p0_4(scratch: Path) -> dict[str, bytes]:
    from tests.fixtures.goldens.values import p0_4_values

    values, written = p0_4_values(scratch)
    return {"p0_4_registry.json": encode(values),
            **{f"p0_4_cohort_{name}.json": data for name, data in written.items()}}


#: Each golden's writer, by name.
GOLDEN_WRITERS: dict[str, Callable[[Path], dict[str, bytes]]] = {
    "p0_1": _p0_1, "p0_2": _p0_2, "p0_3": _p0_3, "p0_4": _p0_4}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("goldens", nargs="*", choices=sorted(GOLDEN_WRITERS),
                        help="the goldens to write (default: all)")
    parser.add_argument("--corpus", action="store_true",
                        help="build the P0-1 corpus (refused when it exists)")
    args = parser.parse_args(argv)
    if args.corpus:
        build_corpus(CORPUS)
        print(f"wrote {CORPUS}")
        return 0
    computed: dict[str, bytes] = {}
    for name in args.goldens or sorted(GOLDEN_WRITERS):
        with tempfile.TemporaryDirectory() as tmp:
            computed.update(GOLDEN_WRITERS[name](Path(tmp).resolve()))
    committed = {filename: (GOLDENS / filename).read_bytes()
                 if (GOLDENS / filename).exists() else None for filename in computed}
    refused = sorted(filename for filename in computed
                     if filename in STRICT and committed[filename] != computed[filename])
    if refused:
        for filename in refused:
            what = "is missing" if committed[filename] is None else "would change"
            print(f"refused: {filename} {what}; it is strict ({STRICT[filename]})",
                  file=sys.stderr)
        print("nothing was written: a strict golden's difference is a change of the code it "
              "pins, found by its test (tests/unit/test_golden_p0_*.py), which names what "
              "differs", file=sys.stderr)
        return 1
    for filename, data in computed.items():
        if committed[filename] == data:
            print(f"unchanged {filename}")
            continue
        (GOLDENS / filename).write_bytes(data)
        print(f"wrote {filename}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
