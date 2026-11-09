"""
The values the Stage 1a goldens pin, computed from the code under test.
======================================================================
`docs/working/research-comparison/README.md` §5 Stage 1. Every function here
runs the production functions and returns what one golden file holds; the
tests compute it again and compare it with the committed file, and
`generate.py` writes the file from the same function. Nothing here restates
what a production function computes.

**What is normalised, and how** (nothing else varies between runs):

- **Paths** are never pinned: a registry or workspace is built or copied
  under the caller's scratch directory, and a path is pinned only relative to
  its registry. The P0-1 corpus's own paths were fixed when it was built
  (`generate.build_corpus`).
- **Times** are fixed by the fixtures (`registered_cohort.IMPORTED_AT`, the
  corpus's run ids and registration times); the P0-4 ``resolve_version``
  identity's ``opened`` stat (device, inode, size, times) is left out.
- **The code revision** the importer records (`imported.code_revision`) is
  fixed to main 2da28cf, clean (`fixed_code_revision`): the checkout taking a
  golden is otherwise whatever it is.
- **Weights** for P0-3 are filled from one seeded generator in state-dict key
  order (`fixed_weights`), so the golden pins validation, not the random
  initialisation a bundle's construction draws.
- **The environment** of P0-3's floats is recorded beside them
  (`cpu_environment`): its scores are pinned exactly as this CPU produced them.

Module: tests/fixtures/goldens/values.py
"""
from __future__ import annotations

import contextlib
import dataclasses
import platform
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from tests.fixtures.goldens import CORPUS, sha256

#: The revision the goldens were taken on: main 2da28cf, whose `src/` and
#: `scripts/` the round starts from (the commit adding them adds tests only).
TAKEN_ON = "2da28cf89d6eb7e31ad28d11f2f03024aa8f03da"

#: The code revision the importer records while a golden is taken.
FIXED_CODE_REVISION: dict[str, Any] = {
    "state": "recorded", "revision": TAKEN_ON, "dirty": False, "modified_tracked": 0,
    "untracked": 0,
}


@contextlib.contextmanager
def fixed_code_revision() -> Iterator[None]:
    """`run_environment.code_revision` answering `FIXED_CODE_REVISION` (a copy)
    for every caller that reads it at call time (the importer, training,
    measurement), restored on exit."""
    from unittest import mock

    import src.utils.run_environment as run_environment

    with mock.patch.object(run_environment, "code_revision",
                           lambda root: dict(FIXED_CODE_REVISION)):
        yield


# ---------------------------------------------------------------------------
# P0-1: kept reports' records, store-scan keys, ledger readings and views
# ---------------------------------------------------------------------------
def _group_items(mapping: dict[tuple[Any, ...], Any]) -> list[dict[str, Any]]:
    return [{"key": list(key), "items": [list(item) if isinstance(item, tuple) else item
                                         for item in items]}
            for key, items in sorted(mapping.items())]


def p0_1_values(scratch: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """P0-1 over the kept corpus (`CORPUS`): the strict values and the views.

    **Strict** (`p0_1_records.json`): for each kept report, in label and run
    order, the record `build_record` makes of the bytes the store's reader
    parsed (`report_store.read_report`), its `record_key`, and where the
    store's one scan places it (`scan_report_store`, `StoreScan.run_key`) with
    the digest of the record the scan built; for each version, its
    ``cohort.json`` digest and the scan's groups, unbuilt and unplaced lists;
    for each ledger, read by the one reader (`read_ledger`), each record's key,
    its run identity (`record_identities`) and its possible overlaps
    (`possible_overlaps`).

    **Views** (`p0_1_views.json`): for each ledger and each checkpoint digest
    it holds records for, in first-appearance order, the API's
    `read_records` listing: each readable record's `RecordView` as JSON and
    each unreadable one's `UnreadableRecord`.

    The registry is read from a copy under `scratch`, since a registry inside
    the repository is refused; the ledgers are read where they are."""
    from src.api.routes.evaluations import read_records
    from src.evaluation.cohort_registry import resolve_version
    from src.evaluation.report_store import read_report, scan_report_store
    from src.evaluation.sidecar import (
        build_record,
        possible_overlaps,
        read_ledger,
        record_identities,
        record_key,
    )
    from tests.fixtures.goldens import encode

    registry = Path(shutil.copytree(CORPUS / "registry", Path(scratch) / "registry"))
    versions: dict[str, Any] = {}
    reports: list[dict[str, Any]] = []
    for directory in sorted((registry / "versions").iterdir()):
        version = resolve_version(registry, directory.name)
        scan = scan_report_store(version)
        versions[directory.name] = {
            "cohort_json_sha256": version.identity.sha256,
            "scan_groups": [{"key": list(key), "runs": list(runs)}
                            for key, runs in sorted(scan.runs.items())],
            "scan_unbuilt": _group_items(scan.unbuilt),
            "scan_unplaced": [list(item) for item in scan.unplaced],
        }
        for run_dir in sorted(p for p in (directory / "reports").iterdir() if p.is_dir()):
            report, digest = read_report(run_dir / "report.json")
            record = build_record(report, digest)
            placed = scan.run_key(run_dir.name, digest)
            key, problem = placed if placed is not None else (None, "not placed")
            scanned = scan.reports[key][run_dir.name][1] if key in scan.reports else None
            reports.append({
                "label": directory.name,
                "run_id": run_dir.name,
                "report_sha256": digest,
                "record": record,
                "record_sha256": sha256(encode(record)),
                "record_key": list(record_key(record)),
                "scan_key": list(key) if key is not None else None,
                "scan_problem": problem,
                "scan_record_sha256": sha256(encode(scanned)) if scanned is not None else None,
            })

    ledgers: dict[str, Any] = {}
    views: dict[str, Any] = {}
    for path in sorted((CORPUS / "ledgers").glob("*.json")):
        ledger, digest = read_ledger(path)
        rows = []
        for index, (record, identity, overlaps) in enumerate(zip(
                ledger["records"], record_identities(ledger), possible_overlaps(ledger),
                strict=True)):
            rows.append({"index": index, "record_key": list(record_key(record)),
                         "run_id": identity.run_id, "historical": identity.historical,
                         "problem": identity.problem, "overlaps": list(overlaps)})
        ledgers[path.name] = {
            "sha256": digest,
            "schema_version": ledger["schema_version"],
            "run_identity_from_sequence": ledger.get("run_identity_from_sequence"),
            "records": rows,
        }
        listed = []
        for checkpoint in dict.fromkeys(r["checkpoint_digest"] for r in ledger["records"]):
            viewed, unreadable = read_records(ledger, checkpoint)
            listed.append({
                "checkpoint_digest": checkpoint,
                "records": [view.model_dump(mode="json") for _record, view in viewed],
                "unreadable": [item.model_dump(mode="json") for item in unreadable],
            })
        views[path.name] = listed
    return ({"versions": versions, "reports": reports, "ledgers": ledgers},
            {"ledgers": views})


# ---------------------------------------------------------------------------
# P0-2: the protocol judgments and their sentences
# ---------------------------------------------------------------------------
#: The records the model spec can write (`spec.LEARNING_STAGES`, `spec.TASKS`:
#: a base has no task, a finetune one of the three), and no record read.
P0_2_RECORDS: tuple[tuple[str | None, str | None], ...] = (
    ("base", None),
    ("finetune", "disease"),
    ("finetune", "gene"),
    ("finetune", "patients_like_me"),
    (None, None),
)

#: Records the model spec cannot write (`ModelSpec.validate` refuses a
#: finetune without a known task head and a base with one), judged beside
#: `P0_2_RECORDS` because only they reach `available`'s sentence for a record
#: no protocol takes and `model_refusal`'s sentence without "It can be
#: measured by": a finetune with no head, one with an unknown head, and a base
#: with a head.
P0_2_UNWRITABLE_RECORDS: tuple[tuple[str | None, str | None], ...] = (
    ("finetune", None),
    ("finetune", "unknown"),
    ("base", "disease"),
)

#: Names that are no protocol of this build: retired modes, a wrong case, an
#: empty name and none at all.
P0_2_UNKNOWN_NAMES: tuple[Any, ...] = ("C", "D", "Disease", "", None)

#: A cohort version's counts, by case: none, the importer's without genes
#: (`registered_cohort.GRAPH_SOURCE_COUNTS`'s shape), each gene need met or
#: missing, and counts of the wrong type.
P0_2_COUNTS: dict[str, Any] = {
    "no_counts": None,
    "empty": {},
    "no_genes_block": {"source_cases": 13, "ranked_eligible": 10},
    "genes_not_an_object": {"genes": [1, 2]},
    "genes_empty": {"genes": {}},
    "genes_zero": {"genes": {"cases_with_causal_genes": 0, "cases_with_candidate_genes": 0}},
    "causal_only": {"genes": {"cases_with_causal_genes": 4, "cases_with_candidate_genes": 0}},
    "candidate_only": {"genes": {"cases_with_causal_genes": 0,
                                 "cases_with_candidate_genes": 4}},
    "both": {"genes": {"cases_with_causal_genes": 4, "cases_with_candidate_genes": 3}},
    "booleans": {"genes": {"cases_with_causal_genes": True,
                           "cases_with_candidate_genes": True}},
    "text": {"genes": {"cases_with_causal_genes": "4", "cases_with_candidate_genes": "3"}},
    "negative": {"genes": {"cases_with_causal_genes": -1, "cases_with_candidate_genes": -1}},
}


def p0_2_values() -> dict[str, Any]:
    """P0-2 (`p0_2_protocols.json`): `names()`, the two standing sentences,
    and every result of `available`, `model_refusal`, `inputs_refusal` and
    `refusal` for each protocol and unknown name, each record of
    `P0_2_RECORDS` and then of `P0_2_UNWRITABLE_RECORDS`, and each counts
    case of `P0_2_COUNTS`, with its sentence."""
    from src.evaluation import protocols

    names = (*protocols.names(), *P0_2_UNKNOWN_NAMES)
    records = (*P0_2_RECORDS, *P0_2_UNWRITABLE_RECORDS)
    return {
        "names": list(protocols.names()),
        "sentences": {"BASE_MESSAGE": protocols.BASE_MESSAGE,
                      "NO_RECORD": protocols.NO_RECORD},
        "counts_cases": P0_2_COUNTS,
        "available": [
            {"learning_stage": stage, "task": task,
             "protocols": list(found), "reason": reason}
            for stage, task in records
            for found, reason in [protocols.available(stage, task)]],
        "model_refusal": [
            {"name": name, "learning_stage": stage, "task": task,
             "refusal": protocols.model_refusal(name, stage, task)}
            for name in names for stage, task in records],
        "inputs_refusal": [
            {"name": name, "counts": case, "refusal": protocols.inputs_refusal(name, counts)}
            for name in names for case, counts in P0_2_COUNTS.items()],
        "refusal": [
            {"name": name, "learning_stage": stage, "task": task, "counts": case,
             "refusal": protocols.refusal(name, stage, task, counts)}
            for name in names for stage, task in records
            for case, counts in P0_2_COUNTS.items()],
    }


# ---------------------------------------------------------------------------
# P0-3: base validation's pairs, scores and output on CPU
# ---------------------------------------------------------------------------
#: The generator `fixed_weights` draws every weight from.
WEIGHT_SEED = 20261110


def cpu_environment() -> dict[str, Any]:
    """What decides a CPU float here: the torch build, the machine and the
    CPU kernels torch dispatches to."""
    import torch

    return {"torch": torch.__version__, "machine": platform.machine(),
            "cpu_capability": torch.backends.cpu.get_cpu_capability()}


def _tensor(t: Any) -> dict[str, Any]:
    """A tensor as its dtype, shape, values and the digest of its bytes."""
    t = t.detach().cpu().contiguous()
    return {"dtype": str(t.dtype), "shape": list(t.shape), "values": t.tolist(),
            "sha256": sha256(t.numpy().tobytes())}


def state_digest(module: Any) -> str:
    """SHA-256 over a module's state, key by key: name, dtype, shape, bytes."""
    import hashlib

    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        t = tensor.detach().cpu().contiguous()
        h.update(f"{name}|{t.dtype}|{list(t.shape)}|".encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def fixed_weights(bundle: Any) -> str:
    """Fill every floating tensor of `bundle`'s state, in key order and each
    shared tensor once, from one generator seeded `WEIGHT_SEED`: uniform in
    [-0.5, 0.5), a running variance in [0.5, 1.5). Integer buffers are left.
    Returns `state_digest`."""
    import torch

    generator = torch.Generator().manual_seed(WEIGHT_SEED)
    seen = set()
    with torch.no_grad():
        for name, tensor in sorted(bundle.state_dict(keep_vars=True).items()):
            if id(tensor) in seen or not tensor.is_floating_point():
                continue
            seen.add(id(tensor))
            drawn = torch.rand(tuple(tensor.shape), generator=generator, dtype=torch.float64)
            drawn = drawn + 0.5 if name.endswith("running_var") else drawn - 0.5
            tensor.copy_(drawn.to(tensor.dtype))
    return state_digest(bundle)


def toy_view() -> tuple[Any, Any]:
    """The toy view and split of tests/unit/test_base_lp_task.py (`_toy_view`)
    at 2da28cf, restated as data so this golden does not move with that module:
    40 genes on a ring with chords, each relation with its `rev_` twin, split
    with seed 1."""
    import torch

    from src.kg.graph_view import build_graph_view, split_facts

    def ei(pairs):
        return torch.tensor(pairs, dtype=torch.long).T.reshape(2, -1)

    n = 40
    pairs = [(i, (i + 1) % n) for i in range(n)] + [(i, (i + 7) % n) for i in range(0, n, 2)]
    view = build_graph_view({
        "num_nodes_dict": {"gene": n},
        "edge_index_dict": {("gene", "link", "gene"): ei(pairs),
                            ("gene", "rev_link", "gene"): ei([(b, a) for a, b in pairs])}})
    return view, split_facts(view, seed=1)


def toy_spec() -> Any:
    """The toy spec of tests/unit/test_base_lp_task.py (`_spec`) at 2da28cf:
    the reference base spec with an 8-wide input and a 2-layer, 4-wide,
    2-head encoder."""
    from src.models.bundle import reference_spec
    from src.models.bundle.spec import InputSpec

    spec = reference_spec("base")
    return dataclasses.replace(
        spec, input=InputSpec(dim=8),
        encoder=dataclasses.replace(spec.encoder, layers=2, hidden=4, output=4, heads=2)
    ).validate()


def _validated(task: Any, bundle: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """`task.validate(bundle)` on CPU, and the raw scores it computed, as
    validation itself computed them: every output of the LP head's
    `raw_score` during it, concatenated in call order, the first
    `task.val_head.numel()` being the positives' and the rest the negatives'.

    Only the scores are pinned, not how many calls made them: one call per
    side today, and one call over both sides or one per chunk score the same
    pairs. What this assumes is the order: every positive, in pair order,
    before every negative, in pair order. A validation that interleaved them
    would fail P0-3 with no score changed, which its author then states."""
    import torch

    scores: list[Any] = []
    raw_score = bundle.lp_head.raw_score

    def recording(*args: Any, **kwargs: Any) -> Any:
        out = raw_score(*args, **kwargs)
        scores.append(out.detach().clone())
        return out

    bundle.lp_head.raw_score = recording
    try:
        metrics = task.validate(bundle, torch.device("cpu"))
    finally:
        del bundle.lp_head.raw_score
    scored = torch.cat(scores)
    positives = task.val_head.numel()
    return metrics, {"positive": _tensor(scored[:positives]),
                     "negative": _tensor(scored[positives:])}


def _validation_case(view: Any, split: Any, spec: Any, *, seed: int,
                     val_batch_size: int) -> dict[str, Any]:
    from src.models.bundle import build_bundle
    from src.training.lp_task import BaseLPTask

    train, val = split.mask("train"), split.mask("val")
    task = BaseLPTask(num_nodes=view.num_nodes, train_edges=view.edge_index(train),
                      train_relations=view.rel[train], val_edges=view.edge_index(val),
                      val_relations=view.rel[val], seed=seed, val_batch_size=val_batch_size)
    bundle = build_bundle(spec, view)
    weights = fixed_weights(bundle)
    metrics, scores = _validated(task, bundle)
    return {
        "view_digest": view.digest,
        "split": split.record(),
        "task": {"seed": seed, "val_batch_size": val_batch_size,
                 "val_counts": dict(task.val_counts)},
        "pairs": {name: _tensor(getattr(task, name))
                  for name in ("val_head", "val_rel", "val_tail", "val_neg_tail")},
        "weights": {"seed": WEIGHT_SEED, "state_sha256": weights},
        "scores": scores,
        "metrics": metrics,
    }


def p0_3_values(scratch: Path) -> dict[str, Any]:
    """P0-3 (`p0_3_base_validation.json`): `BaseLPTask`'s fixed validation
    pairs, the scores its `validate` computes and the dict it returns, on CPU,
    for three cases:

    - ``toy_view``: `toy_view` and `toy_spec`, the task as
      tests/unit/test_base_lp_task.py builds it (seed 0, batches of 4);
    - ``base_workspace`` and ``demo_workspace``: the base-trainable synthetic
      workspace (tests/fixtures/base_workspace.py) and the demo workspace
      (tests/fixtures/registered_cohort.write_demo_workspace), each with the
      demo's base spec (`trained_children.small_base_spec`), its view by the
      spec's rule and its split and task as `scripts/train_model.py` builds a
      base's: the training default split seed, `BASE_FIXED`'s batch size.

    Weights are `fixed_weights`'; the CPU is `cpu_environment`'s."""
    from scripts.train_model import TrainConfig
    from src.config.recipe import BASE_FIXED
    from src.kg.graph_view import build_graph_view, split_facts
    from src.kg.storage.file_storage import read_graph_artifacts
    from tests.fixtures.base_workspace import write_base_workspace
    from tests.fixtures.registered_cohort import write_demo_workspace
    from tests.fixtures.trained_children import small_base_spec

    scratch = Path(scratch)
    split_seed = {f.name: f.default for f in dataclasses.fields(TrainConfig)}["split_seed"]
    cases: dict[str, Any] = {}
    view, split = toy_view()
    cases["toy_view"] = _validation_case(view, split, toy_spec(), seed=0, val_batch_size=4)
    spec = small_base_spec()
    rule = dataclasses.asdict(spec.graph_view)
    rule["exclude_relations"] = list(rule["exclude_relations"])
    for name, data_dir in (
            ("base_workspace", write_base_workspace(scratch / "base_workspace")),
            ("demo_workspace", write_demo_workspace(scratch / "demo_workspace" / "data"))):
        view = build_graph_view(read_graph_artifacts(data_dir, map_location="cpu").graph_data,
                                rule=rule)
        split = split_facts(view, split_seed)
        cases[name] = _validation_case(view, split, spec, seed=int(split_seed),
                                       val_batch_size=BASE_FIXED["validation_batch_size"])
    return {"environment": cpu_environment(), "cases": cases}


# ---------------------------------------------------------------------------
# P0-4: the registry's schema-1 reads and the supplied importer's bytes
# ---------------------------------------------------------------------------
#: The label of the version whose source lists genes.
GENE_LABEL = "demo-genes-1"


def gene_source_cases(data_dir: Path) -> list[dict[str, Any]]:
    """`registered_cohort.graph_source_cases` with genes: each case names one
    causal gene of the graph and four candidates, in the graph's gene order;
    case-02 also names a causal gene the graph lacks."""
    from src.kg.graph import KnowledgeGraph
    from tests.fixtures.registered_cohort import graph_source_cases

    mapping = KnowledgeGraph.read_json(Path(data_dir) / "kg.json").kg.get_node_id_mapping()
    genes = [node.split(":", 1)[1] for node in sorted(mapping["gene"], key=mapping["gene"].get)]
    cases = graph_source_cases(data_dir)
    for i, case in enumerate(cases):
        case["causal_genes"] = [genes[i % len(genes)]]
        case["candidate_genes"] = [genes[(i + j) % len(genes)] for j in range(4)]
    cases[1]["causal_genes"].append("NOT-A-GENE-1")
    return cases


def p0_4_values(scratch: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    """P0-4 (`p0_4_registry.json` and the `cohort.json` bytes by source).

    The demo workspace (`write_demo_workspace`) is written once, its split
    manifest's artifact digests recorded first (a workspace of other bytes is
    then named as the cause, not the importer). For each fixed synthetic
    source -- ``graph`` (`graph_source_cases`, label `registered_cohort.LABEL`)
    and ``genes`` (`gene_source_cases`, `GENE_LABEL`) -- the supplied importer's
    preview and commit (`registered_cohort.import_version`, under
    `fixed_code_revision`) writes a version into a registry of its own; the
    exact `cohort.json` bytes it wrote are returned, and the golden holds the
    source's and the version files' digests, `resolve_version`'s result
    (manifest, digest of the bytes read, directory and samples path relative to
    the registry; the read's stat left out) and `cohort_version_block`'s block
    with the digest of its `encode_json` bytes."""
    from src.evaluation.cohort_registry import cohort_version_block, encode_json, resolve_version
    from src.kg.artifacts import read_split_manifest
    from tests.fixtures.registered_cohort import (
        LABEL,
        ROLE,
        SOURCE_FILENAME,
        graph_source_cases,
        import_version,
        write_demo_workspace,
        write_source,
    )

    scratch = Path(scratch)
    workspace = write_demo_workspace(scratch / "workspace")
    split_manifest = read_split_manifest(workspace)
    out: dict[str, Any] = {
        "workspace": {"artifacts": dict(split_manifest.manifest["artifacts"]),
                      "split_manifest_sha256": split_manifest.identity.sha256},
        "versions": {},
    }
    written: dict[str, bytes] = {}
    for name, label, cases in (("graph", LABEL, graph_source_cases(workspace)),
                               ("genes", GENE_LABEL, gene_source_cases(workspace))):
        root = scratch / name
        root.mkdir()
        source = write_source(root / SOURCE_FILENAME, cases)
        registry = root / "registry"
        registry.mkdir()
        with fixed_code_revision():
            import_version(registry, workspace, source, label=label, role=ROLE)
        version = resolve_version(registry, label)
        cohort_bytes = (version.directory / "cohort.json").read_bytes()
        block = cohort_version_block(version)
        out["versions"][name] = {
            "source_sha256": sha256(source.read_bytes()),
            "cohort_json_sha256": sha256(cohort_bytes),
            "samples_json_sha256": sha256(version.samples_path.read_bytes()),
            "resolve_version": {
                "manifest": version.manifest,
                "identity_sha256": version.identity.sha256,
                "directory": version.directory.relative_to(registry.resolve()).as_posix(),
                "samples_path": version.samples_path.relative_to(
                    registry.resolve()).as_posix(),
            },
            "cohort_version_block": block,
            "cohort_version_block_sha256": sha256(encode_json(block)),
        }
        written[name] = cohort_bytes
    return out, written


__all__ = ["FIXED_CODE_REVISION", "GENE_LABEL", "P0_2_COUNTS", "P0_2_RECORDS",
           "P0_2_UNKNOWN_NAMES", "P0_2_UNWRITABLE_RECORDS", "TAKEN_ON", "WEIGHT_SEED",
           "cpu_environment", "fixed_code_revision", "fixed_weights", "gene_source_cases",
           "p0_1_values", "p0_2_values", "p0_3_values", "p0_4_values", "state_digest",
           "toy_spec", "toy_view"]
