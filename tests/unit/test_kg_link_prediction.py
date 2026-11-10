"""
The common KG link-prediction measurement: one protocol, one evaluator, its kept outputs.
=========================================================================================
Research comparison §5 Stage 5 (owner's O3; the plan review's F4 and its §5
items 7-9). `scripts/measure_scorer.py --mode kg_link_prediction --cohort
<KG edge set>` scores a KG edge set's stored pairs with the model's LP head
through the one LP core, over the version's message-passing graph (its split's
train part, with the view's node types), for a base and a finetune alike.
What is pinned here:

- **a hand oracle on a toy view**: the raw scores, metric schema 3 (LP
  accuracy at raw > 0, ROC-AUC by average ranks, AP, lp_loss) with units and
  denominators, and the message passing over the train part only;
- **a metric that cannot be computed is a status**, never a number, and
  `lp_metrics`' AP of 1.0 over positives alone is never reached: a stored pair
  always holds a positive and a negative;
- **the scores table** round-trips chunk by chunk, exactly;
- **on a base and a disease child trained for real** (`trained_disease_child`)
  with KG edge sets of every role imported from the base: the report's
  counts with their units (n_samples the positive edges, n_ranked the pairs,
  no R10); measured edges never in message passing for val and test, all of
  them for train, and the version and exposure saying so; the scores file,
  the pairs file and the summary verifying each other (the metrics recomputed
  from the scores file, the raw scores recomputed from the pairs);
- **every exposure case**: held out (a base on its own test set), k of n (its
  val and train sets), after full-view exposure (a finetune), another split of
  the same view (k of n, counted independently), and exposure not recorded (a
  record this build's writer never wrote, or whose split does not rebuild),
  never "unseen", and never a gate;
- **the same weights twice** on CPU give identical outputs, and the
  measurement draws from no torch random generator; two runs with equal and
  with different results are both kept, each in its own run directory;
- **every refusal writes nothing**: a tampered pairs file, a record that
  contradicts its pairs, another graph view (a capability gap, said as one),
  a file without a model spec, a protocol and a data kind that do not match,
  `--split` with the LP protocol, `--training-evidence` with a KG edge set, a
  version with no pair, a version whose split does not rebuild or whose pairs
  are not its role's edges; and a model of another split is measured, not
  refused;
- **a cancelled or failed run** leaves no successful report, and its true
  state (a failure among the scores' chunks, or the report's publication
  failing after the scores);
- **registration by kind** (research comparison Stage 6): `--record`
  registers a KG edge set's run in the ledger, its record keeping its
  exposure, counts and metric statuses; the register-later command is
  offered as for any kept report; the store scan builds and groups the runs;
- **the shared words**: the vocabularies restated across modules pinned
  equal, and the result's own label and note.

Module: tests/unit/test_kg_link_prediction.py
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tests.fixtures.opens import count_opens  # noqa: E402

ROLES = ("train", "val", "test")
LP = "kg_link_prediction"
LEDGER = "evaluations.json"
SUPPLIED_LABEL = "demo-test-1"
#: The other base's split seed (the session base's is 0).
OTHER_SEED = 2


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tree(root: Path) -> dict[Path, bytes]:
    return {p: p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


class Site(NamedTuple):
    root: Path
    data_dir: Path
    base: Path
    disease: Path
    other_base: Path
    registry: Path


def _train_other_base(data_dir: Path, run: Path) -> Path:
    """A second base of the same workspace and view, on another edge split
    (split seed 2, one whose val part has pairs on the demo graph): a model
    of another split of the same view."""
    from tests.fixtures.trained_children import _demo_training, _train

    _train(data_dir=str(data_dir), run_dir=str(run), learning_stage="base", num_epochs=1,
           device="cpu", save_top_k=1, split_seed=OTHER_SEED, **_demo_training()["base"])
    return run / "checkpoints" / "last.pt"


@pytest.fixture(scope="module")
def built(trained_disease_child, tmp_path_factory):
    """The session's demo workspace, base and disease child copied under one
    root (the session's files stay read-only), a second base on another split
    seed (`OTHER_SEED`), and a registry holding a KG edge set of each role
    taken from the base and a supplied cohort version of the graph source.
    Nothing is measured here."""
    from src.evaluation.cohort_registry import commit_import, prepare_kg_edges_import
    from tests.fixtures.registered_cohort import graph_source_cases, import_version, write_source

    root = (tmp_path_factory.mktemp("kg_lp") / "root").resolve()
    data_dir = Path(shutil.copytree(trained_disease_child["data_dir"], root / "workspace"))
    checkpoints = {}
    for name in ("base", "disease"):
        target = root / name / "last.pt"
        target.parent.mkdir(parents=True)
        shutil.copyfile(trained_disease_child[name], target)
        checkpoints[name] = target
    other = _train_other_base(data_dir, root / "runs" / "other_base")
    target = root / "other_base" / "last.pt"
    target.parent.mkdir()
    shutil.copyfile(other, target)
    shutil.rmtree(root / "runs")
    registry = root / "registry"
    registry.mkdir()
    for role in ROLES:
        plan = prepare_kg_edges_import(registry, data_dir, checkpoints["base"], f"kg-{role}",
                                       role)
        commit_import(plan, registry, plan.preview_id)
    source = write_source(root / "cases.json", graph_source_cases(data_dir))
    import_version(registry, data_dir, source, label=SUPPLIED_LABEL)
    return Site(root, data_dir, checkpoints["base"], checkpoints["disease"], target, registry)


@pytest.fixture
def site(built, tmp_path, monkeypatch):
    """A copy of the built tree for one test, its registry the deployment's
    (`SHEPHERD_REGISTRY`): a ``--cohort`` run measures only there."""
    root = Path(shutil.copytree(built.root, tmp_path / "root"))

    def moved(path: Path) -> Path:
        return root / path.relative_to(built.root)

    copied = Site(root, moved(built.data_dir), moved(built.base), moved(built.disease),
                  moved(built.other_base), moved(built.registry))
    monkeypatch.setenv("SHEPHERD_REGISTRY", str(copied.registry))
    return copied


@pytest.fixture
def cli():
    import scripts.measure_scorer as module

    return module


def _argv(site: Site, label: str, checkpoint: Path = None, *extra: str, mode: str = LP):
    return ["--mode", mode, "--checkpoint", str(checkpoint or site.base),
            "--data-dir", str(site.data_dir), "--cohort", label,
            "--registry", str(site.registry), "--device", "cpu", "--batch-size", "7", *extra]


def _version(site: Site, label: str):
    from src.evaluation.cohort_registry import resolve_version

    return resolve_version(site.registry, label)


def _runs(site: Site, label: str):
    reports = site.registry / "versions" / label / "reports"
    if not reports.is_dir():
        return []
    from src.evaluation.cohort_registry import is_run_id

    return sorted(p for p in reports.iterdir() if is_run_id(p.name))


def _only_run(site: Site, label: str, *earlier: Path) -> Path:
    (run,) = set(_runs(site, label)) - set(earlier)
    return run


def _report(run: Path) -> dict[str, Any]:
    return json.loads((run / "report.json").read_text())


def _scores(run: Path, report: dict[str, Any]) -> np.ndarray:
    from src.models.bundle.lp_head import lp_scores_from_table

    data = (run / report["scores"]["file"]).read_bytes()
    assert _sha(data) == report["scores"]["sha256"] and len(data) == report["scores"]["bytes"]
    return lp_scores_from_table(data, report["scores"]["layout"])


def _nothing_written(site: Site, before: dict[Path, bytes]) -> None:
    assert _tree(site.registry) == before
    for checkpoint in (site.base, site.disease, site.other_base):
        assert not (checkpoint.parent / LEDGER).exists()


# ---------------------------------------------------------------------------
# A toy view: the hand oracle
# ---------------------------------------------------------------------------
#: A toy graph of two node types and two relations, each with its reverse:
#: 50 facts in one connected component, so each part of a split has pairs.
_TOY_NODES = {"gene": 10, "disease": 8}
_TOY_EDGES = {
    ("gene", "assoc", "disease"): [[g for g in range(10) for _k in range(3)],
                                   [(g * 3 + k) % 8 for g in range(10) for k in range(3)]],
    ("gene", "interacts", "gene"): [[g for g in range(10) for _k in range(2)],
                                    [(g + 1 + k * 3) % 10 for g in range(10) for k in range(2)]],
}
_TOY_SEED = 1


def _toy_view():
    from src.kg.graph_view import build_graph_view

    edges = {}
    for (s, r, d), (src, dst) in _TOY_EDGES.items():
        edges[(s, r, d)] = torch.tensor([src, dst])
        edges[(d, f"rev_{r}", s)] = torch.tensor([dst, src])
    return build_graph_view({"num_nodes_dict": dict(_TOY_NODES), "edge_index_dict": edges})


class _ToyBundle:
    """A stand-in for a bundle: fixed node vectors and the real DistMult head
    with fixed relation weights, recording the graph input it was given."""

    def __init__(self, view, width: int = 3):
        from src.models.bundle.lp_head import DistMultHead
        from src.models.bundle.spec import LPHeadSpec

        n, r = int(view.num_nodes), len(view.relations)
        self.z = torch.tensor([[((i * 7 + j * 3) % 5) - 2.0 for j in range(width)]
                               for i in range(n)], dtype=torch.float32)
        self.lp_head = DistMultHead(LPHeadSpec(), r, width)
        with torch.no_grad():
            self.lp_head.relation_weights.copy_(torch.tensor(
                [[((k * 5 + j) % 3) - 1.0 for j in range(width)] for k in range(r)]))
        self.seen = []

    def eval(self):
        return self

    def node_vectors_full(self, edges, *, edge_type, node_type, chunk_size):
        self.seen.append((edges.clone(), edge_type.clone(), node_type.clone()))
        return self.z


def _toy_block(view, split, pairs, role: str, label: str = "toy") -> dict[str, Any]:
    """The report block of a toy KG edge set (the fields the evaluator reads)."""
    from src.models.bundle.lp_head import LP_PAIR_RULE

    record = split.record()
    return {"label": label, "role": role, "kind": "kg_edges", "manifest_sha256": "b" * 64,
            "split": record,
            "message_passing": {"part": "train", "edges": record["edges"]["train"],
                                "includes_measured_edges": role == "train"},
            "pairs": {"rule": LP_PAIR_RULE, "file": "pairs.bin", "sha256": "c" * 64,
                      "lp_pairs_digest": pairs.digest},
            "counts": {"positive_edges": pairs.counts["edges"], "pairs": pairs.counts["pairs"],
                       "edges_without_negative": pairs.counts["without_negative"],
                       "message_passing_edges": record["edges"]["train"]}}


def _toy_record(split, *, stage: str = "base", exposure: Any = None):
    from src.models.bundle import record as rec

    if exposure is None:
        exposure = {"base": {
            "training": {"message_passing": rec.EXPOSURE_TRAIN_EDGES_DRAWN_PAIRS_REMOVED,
                         "supervision": rec.EXPOSURE_TRAIN_EDGES},
            "validation": {"message_passing": rec.EXPOSURE_TRAIN_EDGES,
                           "supervision": rec.EXPOSURE_VAL_EDGES}}}
    return SimpleNamespace(data_roles={"split": split.record()}, learning_stage=stage,
                           operation="train", graph_exposure=exposure)


def _toy_manifest(block: dict[str, Any]):
    return SimpleNamespace(cohort_kind="kg_edges", cohort_version=block,
                           artifact_digests={"checkpoint": "a" * 64}, amp_enabled=False,
                           amp_dtype=None)


def _toy(role: str = "test", seed: int = _TOY_SEED):
    from src.kg.graph_view import split_facts
    from src.models.bundle.lp_head import fixed_lp_pairs

    view = _toy_view()
    split = split_facts(view, seed)
    mask = split.mask(role)
    pairs = fixed_lp_pairs(view.edge_index(mask), view.rel[mask], seed=seed, batch_size=64)
    return view, split, pairs


def test_the_hand_oracle_on_a_toy_view():
    """Every number of the report computed by hand from the toy vectors: each
    raw score sum(h * w_r * t), LP accuracy over both sides at raw > 0 /
    raw <= 0, ROC-AUC by pairwise comparison (a tie half), AP by its step sum,
    and lp_loss the mean hinge over the tanh scores; message passing over the
    train part only, with the view's node types."""
    from src.evaluation.task_measurement import LP_METRICS, run_kg_link_prediction

    view, split, pairs = _toy("train")
    assert pairs.counts["pairs"] >= 40
    bundle = _ToyBundle(view)
    block = _toy_block(view, split, pairs, "train")
    result = run_kg_link_prediction(
        bundle=bundle, view=view, pairs=pairs, record=_toy_record(split),
        manifest=_toy_manifest(block), device=torch.device("cpu"), batch_size=3)

    z, w = bundle.z.double(), bundle.lp_head.relation_weights.detach().double()

    def raw(h, r, t):
        return float((z[h] * w[r] * z[t]).sum())

    pos = [raw(h, r, t) for h, r, t in zip(pairs.head.tolist(), pairs.relation.tolist(),
                                           pairs.tail.tolist(), strict=True)]
    neg = [raw(h, r, t) for h, r, t in zip(pairs.head.tolist(), pairs.relation.tolist(),
                                           pairs.negative_tail.tolist(), strict=True)]
    # Small whole numbers: the float32 scores are exact.
    assert result.positive.tolist() == pos and result.negative.tolist() == neg
    assert result.positive.dtype == torch.float32 and len(set(pos + neg)) < 2 * len(pos)
    n = len(pos)
    accuracy = (sum(p > 0 for p in pos) + sum(q <= 0 for q in neg)) / (2 * n)
    auc = sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg) / (n * n)
    scored = sorted([(p, 1) for p in pos] + [(q, 0) for q in neg], key=lambda x: -x[0])
    ap, hits, seen, before, k = 0.0, 0, 0, 0, 0
    while k < len(scored):  # one step per distinct threshold
        j = k
        while j < len(scored) and scored[j][0] == scored[k][0]:
            hits, seen, j = hits + scored[j][1], seen + 1, j + 1
        ap += ((hits - before) / n) * (hits / seen)
        before, k = hits, j
    loss = sum(max(0.0, 1.0 - (math.tanh(p) - math.tanh(q))) for p, q in
               zip(pos, neg, strict=True)) / n
    metrics = result.authoritative_metrics
    assert set(metrics) == set(LP_METRICS)
    assert metrics["lp_accuracy"] == pytest.approx(accuracy, abs=1e-12)
    assert metrics["lp_roc_auc"] == pytest.approx(auc, abs=1e-12)
    assert metrics["lp_average_precision"] == pytest.approx(ap, abs=1e-12)
    assert metrics["lp_loss"] == pytest.approx(loss, abs=1e-6)

    # Each metric's status, unit and denominator; the counts with their units.
    statuses = result.sampler_evidence["metrics"]
    assert {name: s["status"] for name, s in statuses.items()} == dict.fromkeys(LP_METRICS,
                                                                                "computed")
    assert {name: s["denominator"]["count"] for name, s in statuses.items()} == {
        "lp_accuracy": 2 * n, "lp_roc_auc": n * n, "lp_average_precision": n, "lp_loss": n}
    assert all(s["unit"] and s["denominator"]["unit"] for s in statuses.values())
    counts = result.sampler_evidence["counts"]
    assert counts == {"positive_edges": pairs.counts["edges"], "pairs": n,
                      "edges_without_negative": pairs.counts["without_negative"],
                      "pairs_scored": n, "pairs_not_finite": 0,
                      "message_passing_edges": int(split.mask("train").sum())}
    assert set(result.sampler_evidence["count_units"]) == set(counts)
    assert (result.n_ranked, result.n_ground_truth_absent) == (n, 0)

    # Message passing: exactly the train part, with the view's node types.
    (edges, relations, node_type), = bundle.seen
    train = split.mask("train")
    assert torch.equal(edges, view.edge_index(train)) and torch.equal(relations, view.rel[train])
    assert torch.equal(node_type, view.node_type)
    passing = result.sampler_evidence["message_passing"]
    assert passing["measured_edges_in_it"] == n and passing["includes_measured_edges"] is True


@pytest.mark.parametrize("role", ["val", "test"])
def test_measured_edges_are_never_in_message_passing_for_val_and_test(role):
    """The pairs' positives (a val or test part's edges) are not among the edges
    the bundle passes messages over, and the evidence and exposure say so; a
    test set of a base is held out relative to its recorded graph use only
    for test (val is in its validation supervision)."""
    from src.evaluation.exposure import HELD_OUT, IN_RECORDED_GRAPH_USE, edge_keys
    from src.evaluation.task_measurement import run_kg_link_prediction

    view, split, pairs = _toy(role)
    bundle = _ToyBundle(view)
    result = run_kg_link_prediction(
        bundle=bundle, view=view, pairs=pairs, record=_toy_record(split),
        manifest=_toy_manifest(_toy_block(view, split, pairs, role)),
        device=torch.device("cpu"), batch_size=64)
    (edges, relations, _types), = bundle.seen
    passed = set(edge_keys(view, edges[0], relations, edges[1]).tolist())
    measured = set(edge_keys(view, pairs.head, pairs.relation, pairs.tail).tolist())
    assert measured and not measured & passed
    assert result.sampler_evidence["message_passing"]["measured_edges_in_it"] == 0
    assert result.exposure["measurement_message_passing"]["measured_edges_in_it"] == 0
    assert result.exposure["measurement_message_passing"]["includes_measured_edges"] is False
    n = pairs.counts["pairs"]
    if role == "test":
        assert (result.exposure["state"], result.exposure["reading"]) == (
            HELD_OUT, "held out relative to the recorded training and validation graph use")
    else:
        assert (result.exposure["state"], result.exposure["reading"]) == (
            IN_RECORDED_GRAPH_USE,
            f"{n} of {n} measured edges were in the recorded training or validation graph use")


def test_a_version_whose_record_contradicts_the_view_is_refused_before_scoring():
    """Two refusals of the evaluator, before anything is scored: a recorded
    split that does not rebuild from the view, and pairs whose positives are
    not edges of the role's part. Each says the version's record contradicts
    its bytes, and that nothing was written."""
    from src.evaluation.task_measurement import run_kg_link_prediction

    view, split, pairs = _toy("test")
    bundle = _ToyBundle(view)
    block = _toy_block(view, split, pairs, "test")
    block["split"] = {**block["split"], "seed": block["split"]["seed"] + 1}
    with pytest.raises(ValueError, match="records an edge split that does not rebuild") as info:
        run_kg_link_prediction(bundle=bundle, view=view, pairs=pairs, record=_toy_record(split),
                               manifest=_toy_manifest(block), device=torch.device("cpu"),
                               batch_size=8)
    assert str(info.value).endswith("Nothing was written.")

    _view, _split, val_pairs = _toy("val")
    block = _toy_block(view, split, val_pairs, "test")
    with pytest.raises(ValueError, match="not edges of its split's test part") as info:
        run_kg_link_prediction(bundle=bundle, view=view, pairs=val_pairs,
                               record=_toy_record(split), manifest=_toy_manifest(block),
                               device=torch.device("cpu"), batch_size=8)
    assert str(info.value).endswith("Nothing was written.")
    assert bundle.seen == []


def test_a_metric_that_cannot_be_computed_is_a_status_never_a_number():
    """With a raw score that is not finite in the set no whole-set metric is a
    number; each says why. `lp_metrics` itself gives AP 1.0 over positives
    alone: the statuses never rest on it, since a stored pair always holds a
    positive and a negative (the pairs file's four columns, one row each)."""
    from src.evaluation.task_measurement import LP_METRICS, lp_metric_statuses
    from src.models.bundle.lp_head import LP_PAIR_COLUMNS
    from src.models.bundle.metrics import lp_metrics

    # The pin: AP 1.0 and no ROC-AUC over positives alone.
    alone = lp_metrics([0.5, 1.5], [])
    assert alone["lp_average_precision"] == 1.0 and alone["lp_roc_auc"] is None
    assert LP_PAIR_COLUMNS == ("head", "relation", "tail", "negative_tail")

    values, statuses = lp_metric_statuses(lp_metrics([0.5, float("nan")], [0.1, -0.2]),
                                          pairs=2, not_finite=1)
    assert values == {}
    for name in LP_METRICS:
        assert statuses[name]["status"] == "not_computable"
        assert statuses[name]["reason"] == ("1 of 2 pairs have a raw score that is not finite, "
                                            "so no whole-set value is defined")
    values, statuses = lp_metric_statuses({"lp_accuracy": 1.0, "lp_roc_auc": None,
                                           "lp_average_precision": 1.0, "lp_loss": 0.0},
                                          pairs=0, not_finite=0)
    assert set(values) == {"lp_accuracy", "lp_loss"}
    for name in ("lp_roc_auc", "lp_average_precision"):
        assert statuses[name] == {**statuses[name], "status": "not_computable",
                                  "reason": "it needs at least one positive and one negative"}
    values, statuses = lp_metric_statuses({"lp_accuracy": 0.5, "lp_roc_auc": True,
                                           "lp_average_precision": 0.5, "lp_loss": 0.1},
                                          pairs=1, not_finite=0)
    assert "lp_roc_auc" not in values
    assert statuses["lp_roc_auc"]["reason"] == "its value True is not a finite number"


def test_a_not_finite_raw_score_is_kept_with_its_state_and_no_metric():
    """A pair whose raw score is not finite is kept in the scores table as
    computed, with its state; the report then gives no metric as a number."""
    from src.evaluation.task_measurement import run_kg_link_prediction
    from src.models.bundle.lp_head import LP_SCORE_STATES, lp_scores_from_table

    view, split, pairs = _toy("train")
    bundle = _ToyBundle(view)
    bundle.z[int(pairs.tail[0])] = float("inf")
    result = run_kg_link_prediction(
        bundle=bundle, view=view, pairs=pairs, record=_toy_record(split),
        manifest=_toy_manifest(_toy_block(view, split, pairs, "train")),
        device=torch.device("cpu"), batch_size=2)
    assert result.authoritative_metrics == {}
    assert {s["status"] for s in result.sampler_evidence["metrics"].values()} == {
        "not_computable"}
    table = lp_scores_from_table(b"".join(result.scores_chunks(rows=3)), result.layout)
    states = [LP_SCORE_STATES[s] for s in table["state"]]
    assert "not_finite" in states and states[0] == "not_finite"
    assert result.sampler_evidence["counts"]["pairs_not_finite"] == states.count("not_finite")


def test_the_scores_table_streams_in_chunks_and_round_trips_exactly():
    """Chunks of any size make one table: each row its pairs-table row index,
    the raw scores exactly as computed (float32 stored as float32, float64 as
    float64, half precision widened exactly) and its state; a layout this build
    does not read, or a ragged table, is refused."""
    from src.models.bundle import lp_head as lp

    positive = torch.tensor([0.25, -1.5, float("nan"), 3.0, 1e-8], dtype=torch.float32)
    negative = torch.tensor([-0.5, 2.0, 0.0, float("-inf"), -1e-8], dtype=torch.float32)
    state = lp.lp_score_states(positive, negative)
    assert state.tolist() == [0, 0, 1, 1, 0]
    layout = lp.lp_scores_layout(torch.float32)
    whole = lp.lp_scores_table(0, positive, negative, state, layout)
    for rows in (1, 2, 5, 9):
        assert b"".join(lp.lp_scores_chunks(positive, negative, state, layout,
                                            rows=rows)) == whole
    table = lp.lp_scores_from_table(whole, layout)
    assert table["pair_row"].tolist() == [0, 1, 2, 3, 4]
    assert np.array_equal(table["positive_raw"], positive.numpy(), equal_nan=True)
    assert np.array_equal(table["negative_raw"], negative.numpy(), equal_nan=True)
    assert len(whole) == 5 * (8 + 4 + 4 + 1)
    for dtype, width in ((torch.float64, "<f8"), (torch.float16, "<f4"),
                         (torch.bfloat16, "<f4")):
        layout = lp.lp_scores_layout(dtype)
        assert layout["dtypes"] == ["<i8", width, width, "|u1"]
        p, q = positive.to(dtype), negative.to(dtype)
        table = lp.lp_scores_from_table(
            lp.lp_scores_table(0, p, q, lp.lp_score_states(p, q), layout), layout)
        assert np.array_equal(table["positive_raw"], p.double().numpy(), equal_nan=True)
    with pytest.raises(ValueError, match="not stored by a scores table"):
        lp.lp_scores_layout(torch.int64)
    with pytest.raises(ValueError, match="not a scores table layout"):
        lp.lp_scores_from_table(whole, {**lp.lp_scores_layout(torch.float32), "order": "cols"})
    with pytest.raises(ValueError, match="not a whole number"):
        lp.lp_scores_from_table(whole[:-1], lp.lp_scores_layout(torch.float32))


def test_a_chunked_publication_is_whole_or_absent(tmp_path):
    """`publish_chunks` writes what the chunks yield, hashed as written, and a
    failure or an interrupt among the chunks leaves no file and no staged
    file; never replacing, it refuses a target that exists."""
    from src.utils.file_modes import publish_chunks

    target = tmp_path / "scores.bin"
    assert publish_chunks(target, (b"ab", b"", b"cd"), replace=False) == (_sha(b"abcd"), 4)
    assert target.read_bytes() == b"abcd"
    with pytest.raises(FileExistsError):
        publish_chunks(target, (b"x",), replace=False)
    assert target.read_bytes() == b"abcd"

    def broken(error):
        yield b"part"
        raise error

    for error in (OSError("disk full"), KeyboardInterrupt()):
        other = tmp_path / f"other-{type(error).__name__}.bin"
        with pytest.raises(type(error)):
            publish_chunks(other, broken(error), replace=False)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["scores.bin"]
    with pytest.raises(TypeError, match="chunks of bytes"):
        publish_chunks(tmp_path / "x.bin", ("text",))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["scores.bin"]


# ---------------------------------------------------------------------------
# The recorded graph exposure, in its writer's words
# ---------------------------------------------------------------------------
def test_the_shared_vocabularies_are_one():
    """The words restated across modules that cannot import each other are
    pinned equal: the edge split's parts a full-view word puts in graph use
    (`record.EXPOSURE_ALL_PARTS`, restated: the model record imports no graph
    code) are the split's parts, in the split's order (`exposure` orders by
    one and compares with the other); the data kind the evaluator and the
    ledger's record builder name is the protocols' word, and the ledger
    records every kind the registry holds (research comparison Stage 6); each
    protocol has the words a page offers it under, the LP metrics' words are
    the evaluator's metrics, and the kind words a protocol's refusal restates
    (it imports the standard library only) are the readers'."""
    from src.evaluation import cohort_registry, exposure, protocols, sidecar, task_measurement
    from src.evaluation.caveats import (
        DATA_KIND_WORDS,
        LP_METRIC_WORDS,
        MEASUREMENT_NAMES,
        TASK_NAMES,
        version_words,
    )
    from src.kg.graph_view import SPLIT_NAMES
    from src.models.bundle.record import EXPOSURE_ALL_PARTS

    assert EXPOSURE_ALL_PARTS == SPLIT_NAMES
    assert task_measurement.KG_EDGES is protocols.KG_EDGES
    assert set(sidecar.VERSION_BLOCK_FIELDS) == set(protocols.REGISTERED_KINDS)
    assert set(sidecar.RECORD_COHORT_FIELDS) == set(protocols.REGISTERED_KINDS)
    assert set(sidecar.KIND_RECORD_FIELDS) == set(protocols.REGISTERED_KINDS)
    assert cohort_registry.KG_EDGES_BLOCK_FIELDS is sidecar.VERSION_BLOCK_FIELDS["kg_edges"]
    assert cohort_registry.GENERATED_BLOCK_FIELDS is sidecar.VERSION_BLOCK_FIELDS["generated"]
    assert cohort_registry.version_words is version_words
    assert protocols._KIND_WORDS == DATA_KIND_WORDS
    assert set(LP_METRIC_WORDS) == set(task_measurement.LP_METRICS)
    assert exposure.EXPOSURE_READINGS is __import__(
        "src.evaluation.caveats", fromlist=["EXPOSURE_READINGS"]).EXPOSURE_READINGS
    assert set(MEASUREMENT_NAMES) == set(protocols.names())
    assert {k: MEASUREMENT_NAMES[k] for k in TASK_NAMES} == TASK_NAMES
    assert MEASUREMENT_NAMES[LP] == "KG link prediction"


def test_a_kg_link_prediction_result_is_named_by_its_own_words():
    """The read-side words a `kg_link_prediction` result is shown under, added
    with the protocol (research comparison Stage 5a): its own label and note
    (`caveats.MODE_RESULT_LABELS`, `MODE_RESULT_NOTES`), never "a measurement
    mode this version does not describe". Its metrics have their own words
    under the measurement's name since the one reader by data kind (research
    comparison Stage 6): "KG link prediction ROC-AUC"; a name the measurement
    does not write keeps its recorded name after the measurement's."""
    from src.evaluation.caveats import (
        KG_LINK_PREDICTION_NOTE,
        KG_LINK_PREDICTION_RESULT_LABEL,
        metric_text,
        mode_label,
        mode_note,
    )

    assert mode_label(LP) == KG_LINK_PREDICTION_RESULT_LABEL
    assert KG_LINK_PREDICTION_RESULT_LABEL.startswith("KG link prediction (model bundle): ")
    assert "does not describe" not in mode_label(LP)
    assert mode_note(LP) == KG_LINK_PREDICTION_NOTE
    assert "rule-generated, never proven non-relations" in KG_LINK_PREDICTION_NOTE
    assert "never called held out" in KG_LINK_PREDICTION_NOTE
    assert metric_text(LP, "lp_accuracy") == "KG link prediction accuracy (raw > 0)"
    assert metric_text(LP, "lp_roc_auc") == "KG link prediction ROC-AUC"
    assert metric_text(LP, "lp_average_precision") == "KG link prediction average precision"
    assert metric_text(LP, "lp_loss") == "KG link prediction loss (the LP head's hinge)"
    assert metric_text(LP, "lp_other") == "KG link prediction lp_other"
    assert metric_text("C", "hits@10") == "Mode C hits@10"


def test_the_writer_and_the_reader_share_the_graph_exposure_words(built):
    """`scripts/train_model.py` writes `graph_exposure` only through the names
    `src.models.bundle.record` defines (no literal of those words in the
    writer), and the records it wrote read back through them; a word no build
    wrote is not read as any part."""
    import ast

    from src.models.bundle import record as rec
    from src.models.bundle.record import read_model_record, recorded_graph_use

    words = {getattr(rec, name) for name in rec.__all__ if name.startswith("EXPOSURE_")
             and isinstance(getattr(rec, name), str)}
    assert words == set(rec.EXPOSURE_EDGE_PARTS)
    writer = Path(rec.__file__).resolve().parents[3] / "scripts" / "train_model.py"
    literals = {node.value for node in ast.walk(ast.parse(writer.read_text()))
                if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert not words & literals
    for checkpoint, stage in ((built.base, "base"), (built.disease, "finetune")):
        record = read_model_record(torch.load(checkpoint, weights_only=True,
                                              map_location="cpu"))
        uses = recorded_graph_use(record.graph_exposure)
        assert uses and all(at.startswith(f"{stage}.") for at in uses)
    for bad, says in (
            ({"base": {"training": {"message_passing": "unseen"}}}, "no build's writer wrote"),
            ({"base": {"training": "val edges", "x": 3}}, "is not a use"),
            ({"stage": {"training": {"message_passing": "train edges"}}}, "names the stage"),
            ({}, "not one learning stage's block"), ("train edges", "not one learning stage")):
        with pytest.raises(ValueError, match=says):
            recorded_graph_use(bad)


# ---------------------------------------------------------------------------
# On a base and a disease child trained for real, through the command line
# ---------------------------------------------------------------------------
def _model(checkpoint: Path):
    """The bundle `checkpoint` holds, on the CPU, and its view."""
    from dataclasses import asdict

    from src.kg.graph_view import build_graph_view
    from src.kg.storage.file_storage import read_graph_artifacts
    from src.models.bundle.record import load_bundle, read_model_record

    payload = torch.load(checkpoint, weights_only=True, map_location="cpu")
    record = read_model_record(payload)
    rule = asdict(record.model_spec.graph_view)
    rule["exclude_relations"] = list(rule["exclude_relations"])
    data_dir = checkpoint.parents[1] / "workspace"
    view = build_graph_view(read_graph_artifacts(data_dir).graph_data, rule=rule)
    bundle, record = load_bundle(payload, view_digest=view.digest)
    return bundle, record, view


def _check_scores_against_pairs_and_summary(site: Site, run: Path, checkpoint: Path,
                                            label: str) -> np.ndarray:
    """The scores file, the version's pairs file and the report verify each
    other: the file is the one the report names (digest, bytes, rows), keyed to
    the version's pairs file; each row's raw scores are the model's scores of
    that pairs row (recomputed through the LP core over the version's
    message-passing graph); and the report's metrics are recomputed from the
    scores file alone."""
    from src.evaluation.cohort_registry import read_kg_edge_pairs
    from src.evaluation.task_measurement import LP_METRICS
    from src.kg.graph_view import rebuild_split
    from src.models.bundle.lp_head import LP_SCORE_STATES, score_lp_pairs
    from src.models.bundle.metrics import lp_metrics

    report = _report(run)
    version = _version(site, label)
    pairs = read_kg_edge_pairs(version).pairs
    block = version.manifest["pairs"]
    scores = report["scores"]
    assert scores["keyed_to"] == {"file": "pairs.bin", "sha256": block["sha256"],
                                  "lp_pairs_digest": block["lp_pairs_digest"]}
    table = _scores(run, report)
    assert scores["rows"] == len(table) == block["rows"] == int(pairs.head.numel())
    assert table["pair_row"].tolist() == list(range(len(table)))
    assert [LP_SCORE_STATES[s] for s in set(table["state"].tolist())] == ["scored"]

    # The raw scores are the model's, row by row, over the train part.
    bundle, _record, view = _model(checkpoint)
    split = rebuild_split(view, version.manifest["split"])
    train = split.mask("train")
    again = score_lp_pairs(bundle, pairs, view.edge_index(train), view.rel[train],
                           torch.device("cpu"), node_type=view.node_type, pair_chunk=7)
    assert np.array_equal(table["positive_raw"], again.positive.numpy())
    assert np.array_equal(table["negative_raw"], again.negative.numpy())

    # The summary, recomputed from the scores file alone.
    positive = torch.from_numpy(table["positive_raw"].copy())
    negative = torch.from_numpy(table["negative_raw"].copy())
    recomputed = lp_metrics(positive.tolist(), negative.tolist())
    recomputed["lp_loss"] = float(bundle.lp_head.hinge(torch.tanh(positive),
                                                       torch.tanh(negative)))
    assert report["authoritative_metrics"] == {name: recomputed[name] for name in LP_METRICS}
    return table


@pytest.mark.parametrize("role", ROLES)
def test_a_base_is_measured_on_each_role_and_its_report_says_what_was_measured(
        cli, site, role, capsys):
    """A base on a KG edge set of each role, through the command line: the
    report's manifest is a registered KG edge set's (no R10, R2 not
    applicable), its counts carry their units (n_samples the version's
    positive edges, n_ranked the pairs, nothing absent), metric schema 3 with
    every metric's unit and denominator, the message passing the split's
    train part (the measured edges in it exactly for role train, as the
    version says), and the exposure scoped to the base's record. The run
    directory holds the scores file and then the report."""
    from src.evaluation.caveats import KG_LINK_PREDICTION_RESULT_LABEL
    from src.evaluation.exposure import (
        EXPOSURE_CAVEAT,
        EXPOSURE_DEFINITION_VERSION,
        HELD_OUT,
        IN_RECORDED_GRAPH_USE,
    )
    from src.evaluation.measurement import expected_relations
    from src.evaluation.task_measurement import LP_METRICS

    label = f"kg-{role}"
    assert cli.main(_argv(site, label)) == 0
    run = _only_run(site, label)
    assert sorted(p.name for p in run.iterdir()) == ["report.json", "scores.bin"]
    report = _report(run)
    version = _version(site, label)
    counts = version.manifest["counts"]
    manifest = report["manifest"]

    assert (manifest["mode"], manifest["cohort_kind"], manifest["split"]) == (
        LP, "kg_edges", role)
    assert manifest["cohort_version"] == cli.cohort_version_block(version)
    assert manifest["r10"] is None
    assert manifest["relations"] == expected_relations(registered=True)
    assert manifest["metric_schema_version"] == 3 and manifest["batch_size"] == 7
    assert "never proven non-relations" in manifest["candidate_construction"]
    assert set(manifest["artifact_digests"]) >= {"checkpoint", "pairs"}
    assert "samples" not in manifest["artifact_digests"]
    assert manifest["artifact_digests"]["pairs"] == version.manifest["pairs"]["sha256"]
    assert manifest["n_samples"] == counts["positive_edges"]
    assert (report["n_ranked"], report["n_ground_truth_absent"]) == (counts["pairs"], 0)
    assert set(report["authoritative_metrics"]) == set(LP_METRICS)

    evidence = report["sampler_evidence"]
    assert evidence["counts"] == {
        "positive_edges": counts["positive_edges"], "pairs": counts["pairs"],
        "edges_without_negative": counts["edges_without_negative"],
        "pairs_scored": counts["pairs"], "pairs_not_finite": 0,
        "message_passing_edges": counts["message_passing_edges"]}
    assert set(evidence["count_units"]) == set(evidence["counts"])
    assert {n: s["status"] for n, s in evidence["metrics"].items()} == dict.fromkeys(
        LP_METRICS, "computed")
    passing = evidence["message_passing"]
    in_it = counts["pairs"] if role == "train" else 0
    assert passing == {"part": "train", "split_digest": version.manifest["split"]["digest"],
                       "edges": counts["message_passing_edges"],
                       "measured_edges_in_it": in_it,
                       "includes_measured_edges": role == "train"}
    assert version.manifest["message_passing"]["includes_measured_edges"] is (role == "train")

    exposure = report["exposure"]
    n = counts["pairs"]
    assert exposure["definition_version"] == EXPOSURE_DEFINITION_VERSION
    assert exposure["sources"]["model_record"]["checkpoint_sha256"] == _sha(
        site.base.read_bytes())
    assert exposure["sources"]["model_record"]["learning_stage"] == "base"
    assert exposure["sources"]["version"] == {
        "label": label, "kind": "kg_edges", "role": role,
        "manifest_sha256": version.identity.sha256,
        "split_digest": version.manifest["split"]["digest"]}
    assert exposure["split"] == "the version's split"
    assert exposure["measurement_message_passing"]["measured_edges_in_it"] == in_it
    assert exposure["caveat"] == EXPOSURE_CAVEAT and "rule-generated" in EXPOSURE_CAVEAT
    if role == "test":
        assert (exposure["state"], exposure["reading"]) == (
            HELD_OUT, "held out relative to the recorded training and validation graph use")
        assert exposure["measured_edges"]["in_recorded_graph_use"] == 0
    else:
        assert exposure["state"] == IN_RECORDED_GRAPH_USE
        assert exposure["reading"] == (f"{n} of {n} measured edges were in the recorded "
                                       "training or validation graph use")
        assert exposure["measured_edges"]["in_model_parts"][role] == n
    assert "unseen" not in json.dumps(exposure)

    out = capsys.readouterr().out
    assert f"Exposure ({EXPOSURE_DEFINITION_VERSION}): {exposure['reading']}" in out
    # The console's result header is the protocol's own label.
    assert f"\n{KG_LINK_PREDICTION_RESULT_LABEL}\n  {counts['pairs']} pairs scored\n" in out
    assert "lp_accuracy" in out
    # The register-later command, as for any kept report (research
    # comparison Stage 6: the ledger records a KG edge set's run).
    assert "Not registered (no --record). Register it with:" in out
    assert (f'record_evaluation.py --report "{run / "report.json"}" --checkpoint '
            f'"{site.base.resolve()}"') in out
    _check_scores_against_pairs_and_summary(site, run, site.base, label)


def test_a_finetune_is_measured_on_the_same_pairs_and_reads_after_full_view_exposure(
        cli, site):
    """The disease child, taken by the same protocol, is measured on the same
    pairs over the same message-passing graph; its record passed messages over
    the full view in validation, so its test-set figure reads "after full-view
    exposure", never held out. Its numbers differ from the base's: two runs
    with different results, each kept in its own run directory."""
    from src.evaluation.exposure import AFTER_FULL_VIEW_EXPOSURE

    assert cli.main(_argv(site, "kg-test")) == 0
    base_run = _only_run(site, "kg-test")
    assert cli.main(_argv(site, "kg-test", site.disease)) == 0
    child_run = _only_run(site, "kg-test", base_run)
    base, child = _report(base_run), _report(child_run)

    exposure = child["exposure"]
    assert (exposure["state"], exposure["reading"]) == (AFTER_FULL_VIEW_EXPOSURE,
                                                        "after full-view exposure")
    assert exposure["reason"] == ("the model record's finetune.validation.message_passing "
                                  "covered the full view, so every measured edge was in that "
                                  "graph use")
    assert exposure["sources"]["model_record"]["learning_stage"] == "finetune"
    assert "held out" not in exposure["reading"]
    for key in ("cohort_version", "metric_schema_version", "candidate_construction"):
        assert child["manifest"][key] == base["manifest"][key]
    assert child["sampler_evidence"]["message_passing"] == base["sampler_evidence"][
        "message_passing"]
    assert child["scores"]["keyed_to"] == base["scores"]["keyed_to"]
    assert child["authoritative_metrics"] != base["authoritative_metrics"]
    assert child_run.name.split("-")[1] == _sha(site.disease.read_bytes())[:12]
    _check_scores_against_pairs_and_summary(site, child_run, site.disease, "kg-test")


def test_the_same_weights_measured_twice_give_identical_outputs_both_kept(cli, site,
                                                                          monkeypatch):
    """On CPU, a deterministic small fixture: the same base on the same pairs
    twice gives byte-identical scores and equal metrics, and both runs are
    kept, each in its own run directory, both in the version's usage history.
    The evaluator draws from no torch random generator (nor NumPy's global
    one): the generators' states are the same after it as before. NumPy's
    state is compared whole, its position and cached Gaussian included: a
    draw short of a twist boundary moves the position and leaves the keys."""
    from src.evaluation import task_measurement
    from src.evaluation.report_store import read_usage_history

    original = task_measurement.TASK_RUNS[LP]
    drawn = []

    def numpy_state():
        name, keys, *rest = np.random.get_state()
        return name, keys.copy(), tuple(rest)

    def same_numpy_state(one, two):
        return one[0] == two[0] and np.array_equal(one[1], two[1]) and one[2] == two[2]

    # The comparison sees a draw that leaves the keys as they were (past the
    # first twist after seeding); the global state is put back as it was.
    saved = np.random.get_state()
    try:
        np.random.seed(0)
        np.random.rand()
        probe = numpy_state()
        np.random.rand()
        assert np.array_equal(probe[1], np.random.get_state()[1])
        assert not same_numpy_state(probe, numpy_state())
    finally:
        np.random.set_state(saved)

    def watched(**kwargs):
        before = (torch.get_rng_state(), numpy_state())
        result = original(**kwargs)
        drawn.append(not (torch.equal(before[0], torch.get_rng_state())
                          and same_numpy_state(before[1], numpy_state())))
        return result

    monkeypatch.setitem(task_measurement.TASK_RUNS, LP, watched)
    assert cli.main(_argv(site, "kg-val")) == 0
    first = _only_run(site, "kg-val")
    assert cli.main(_argv(site, "kg-val")) == 0
    second = _only_run(site, "kg-val", first)
    assert drawn == [False, False]

    one, two = _report(first), _report(second)
    assert first != second
    assert (first / "scores.bin").read_bytes() == (second / "scores.bin").read_bytes()
    assert one["scores"]["sha256"] == two["scores"]["sha256"]
    assert one["authoritative_metrics"] == two["authoritative_metrics"]
    assert one["exposure"] == two["exposure"]
    entries, problems = read_usage_history(_version(site, "kg-val"))
    runs = [e for e in entries if e.get("kind") == "test_run"]
    assert problems == []
    assert sorted(e["run_id"] for e in runs) == sorted([first.name, second.name])
    assert {e["mode"] for e in runs} == {LP}


# ---------------------------------------------------------------------------
# Every exposure case; never a gate
# ---------------------------------------------------------------------------
def test_a_model_of_another_split_of_the_same_view_is_measured_and_says_k_of_n(cli, site):
    """Not a refusal: a base whose recorded split (seed `OTHER_SEED`) differs
    from the KG edge set's (the session base's, seed 0) is measured on it, on
    the version's message passing; its exposure says how many of the measured
    edges were in that model's recorded training or validation graph use,
    counted here independently from the model's own split."""
    from src.evaluation.cohort_registry import read_kg_edge_pairs
    from src.evaluation.exposure import ANOTHER_SPLIT, edge_keys
    from src.kg.graph_view import split_facts

    assert cli.main(_argv(site, "kg-test", site.other_base)) == 0
    report = _report(_only_run(site, "kg-test"))
    exposure = report["exposure"]

    _bundle, record, view = _model(site.other_base)
    other = split_facts(view, OTHER_SEED)
    assert record.data_roles["split"]["digest"] == other.digest
    version = _version(site, "kg-test")
    assert other.digest != version.manifest["split"]["digest"]
    pairs = read_kg_edge_pairs(version).pairs
    measured = edge_keys(view, pairs.head, pairs.relation, pairs.tail)
    used = other.mask("train") | other.mask("val")
    k = int(np.isin(measured, edge_keys(view, view.src[used], view.rel[used],
                                        view.dst[used])).sum())
    n = int(pairs.head.numel())
    assert 0 < k <= n
    assert (exposure["state"], exposure["split"]) == (ANOTHER_SPLIT,
                                                      "another split of the same view")
    assert exposure["reading"] == (f"{k} of {n} measured edges were in this model's recorded "
                                   "training or validation graph use, on another split of the "
                                   "same view")
    assert exposure["measured_edges"]["in_recorded_graph_use"] == k
    assert exposure["sources"]["model_record"]["split_digest"] == other.digest
    # The run's own message passing is still the version's.
    assert report["sampler_evidence"]["message_passing"]["split_digest"] == (
        version.manifest["split"]["digest"])
    assert report["sampler_evidence"]["message_passing"]["measured_edges_in_it"] == 0


def test_a_record_whose_exposure_no_build_wrote_is_measured_and_reads_not_recorded(
        cli, site, tmp_path):
    """A checkpoint whose record says something in `graph_exposure` this
    build's writer never wrote is still measured (exposure is never a gate);
    its exposure reads "exposure not recorded" with the reason, never
    "unseen", and the record's words are not kept as if read."""
    from src.evaluation.exposure import NOT_RECORDED

    payload = torch.load(site.base, weights_only=True, map_location="cpu")
    payload["graph_exposure"] = {"base": {"training": {"message_passing": "unseen edges"}}}
    edited = site.base.parent / "edited.pt"
    torch.save(payload, edited)

    assert cli.main(_argv(site, "kg-test", edited)) == 0
    exposure = _report(_only_run(site, "kg-test"))["exposure"]
    assert (exposure["state"], exposure["reading"]) == (NOT_RECORDED, "exposure not recorded")
    assert exposure["reason"] == ("the record's graph_exposure.base.training.message_passing "
                                  "says 'unseen edges', which no build's writer wrote")
    assert exposure["sources"]["model_record"]["graph_exposure"] is None
    assert exposure["measured_edges"]["in_recorded_graph_use"] is None
    assert exposure["sources"]["model_record"]["checkpoint_sha256"] == _sha(edited.read_bytes())


@pytest.mark.parametrize("changed", [{"digest": "0" * 64}, {"seed": -1}, {"seed": 2**32},
                                     {"seed": 10**30}, {"seed": "x"}, {"seed": True}],
                         ids=["digest", "seed-negative", "seed-2**32", "seed-huge", "seed-text",
                              "seed-bool"])
def test_a_record_whose_split_does_not_rebuild_reads_not_recorded(changed):
    """The model's recorded split, rebuilt from the view, is what the counts
    are taken from; one that does not rebuild gives exposure not recorded,
    with the reason, and the block still names its sources. A recorded seed
    NumPy's `RandomState` does not take (below 0, 2**32 or above) is one that
    does not rebuild, said by name: exposure never raises for what a record
    says, and is never a gate."""
    from src.evaluation.exposure import (
        EXPOSURE_DEFINITION_VERSION,
        NOT_RECORDED,
        kg_edges_exposure,
        measured_in_parts,
    )

    view, split, pairs = _toy("test")
    record = _toy_record(split)
    record.data_roles["split"] = {**record.data_roles["split"], **changed}
    block = kg_edges_exposure(record=record, checkpoint_sha256="a" * 64,
                              version={**_toy_block(view, split, pairs, "test"),
                                       "manifest_sha256": "b" * 64},
                              view=view, version_split=split, pairs=pairs,
                              in_version=measured_in_parts(view, split, pairs))
    assert (block["definition_version"], block["state"], block["reading"]) == (
        EXPOSURE_DEFINITION_VERSION, NOT_RECORDED, "exposure not recorded")
    assert block["reason"].startswith("the model record's edge split does not rebuild from "
                                      "this view")
    field = next(iter(changed))
    assert f"the split's recorded {field} " in block["reason"]
    assert block["sources"]["model_record"]["split_digest"] == record.data_roles["split"][
        "digest"]
    assert block["split"] is None


# ---------------------------------------------------------------------------
# Refusals: nothing written
# ---------------------------------------------------------------------------
def _refused(cli, site: Site, argv, error, says: str, *, unopened=(), capsys=None) -> str:
    """`main(argv)` refused with `error` naming `says`, nothing written in the
    registry or beside any checkpoint and none of `unopened` opened (a file, or
    any file in a directory); and, run
    as the program, the operator's one line saying it and that nothing was
    written, exit 1."""
    before = _tree(site.registry)
    with count_opens() as opens:
        with pytest.raises(error) as info:
            cli.main(argv)
    message = str(info.value)
    assert says in message, message
    for path in unopened:
        if path.is_dir():
            inside = [p for p, n in opens.counts.items()
                      if n and Path(p).is_relative_to(path.resolve())]
            assert inside == [], inside
        else:
            assert opens.opens(path) == 0, path
    _nothing_written(site, before)
    if capsys is not None:
        capsys.readouterr()
        assert cli._run_as_program(argv) == 1
        err = capsys.readouterr().err
        assert err.startswith("measure_scorer: refused: ") and says in err, err
        assert "Nothing was written." in err or "Nothing was read or written." in err, err
        _nothing_written(site, before)
    return message


def test_a_tampered_pairs_file_is_refused_before_it_is_parsed(cli, site, capsys):
    from src.evaluation.cohort_registry import CohortVersionRefused

    pairs = site.registry / "versions" / "kg-test" / "pairs.bin"
    data = bytearray(pairs.read_bytes())
    data[3] ^= 1
    pairs.write_bytes(bytes(data))
    _refused(cli, site, _argv(site, "kg-test"), CohortVersionRefused,
             "is not the pairs file", unopened=(site.base,), capsys=capsys)


def test_a_record_that_contradicts_its_pairs_is_refused(cli, site, capsys):
    """The pairs file is the bytes the version binds (its digest rebound in a
    hand-edited cohort.json), but they parse to other pairs than the version
    records: the version's own record contradicts its bytes."""
    from src.models.bundle.lp_head import lp_pairs_from_table, lp_pairs_table

    directory = site.registry / "versions" / "kg-test"
    head, relation, tail, negative = lp_pairs_from_table((directory / "pairs.bin").read_bytes())
    swapped = lp_pairs_table(head, relation, negative, tail)
    (directory / "pairs.bin").write_bytes(swapped)
    manifest = json.loads((directory / "cohort.json").read_text())
    manifest["pairs"]["sha256"] = _sha(swapped)
    (directory / "cohort.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    message = _refused(cli, site, _argv(site, "kg-test"), ValueError, "parses to pairs",
                       unopened=(site.base,), capsys=capsys)
    assert re.search(r"pairs\.bin parses to pairs [0-9a-f]{12}\.\.\., not the [0-9a-f]{12}\.\.\. "
                     r"\S+cohort\.json records\. Nothing was written\.$", message), message


def test_a_pairs_refusal_names_its_own_cause_and_that_nothing_was_written(cli, site,
                                                                          monkeypatch):
    """A refusal of the pairs file is `read_kg_edge_pairs`' own sentence,
    which names its cause, and says nothing was written; it never claims a
    contradiction that is not there (a missing file, a layout this revision
    does not read)."""
    from src.evaluation import cohort_registry
    from src.evaluation.cohort_registry import CohortVersionRefused

    version = _version(site, "kg-test")
    (site.registry / "versions" / "kg-test" / "pairs.bin").unlink()
    with pytest.raises(CohortVersionRefused) as refused:
        cli._read_version_pairs(version)
    assert str(refused.value).endswith(
        "binds it: the KG edge set cannot be read. Nothing was written.")
    assert "contradicts" not in str(refused.value)

    def unread_layout(_version):
        raise CohortVersionRefused("x/cohort.json lays its pairs out as 'other'; this revision "
                                   "reads 'lp-pairs/v1' only. Nothing was read.")

    monkeypatch.setattr(cohort_registry, "read_kg_edge_pairs", unread_layout)
    with pytest.raises(CohortVersionRefused) as refused:
        cli._read_version_pairs(version)
    assert str(refused.value) == ("x/cohort.json lays its pairs out as 'other'; this revision "
                                  "reads 'lp-pairs/v1' only. Nothing was read. Nothing was "
                                  "written.")


def test_a_version_of_another_graph_view_is_refused_as_a_capability_gap(cli, site, capsys):
    """A KG edge set recorded for another graph view than the model's: its
    edges are named by another view's node ids, and this build has no node
    mapping; said as a capability this build lacks, never as a research
    rule, before anything is built or written."""
    directory = site.registry / "versions" / "kg-test"
    manifest = json.loads((directory / "cohort.json").read_text())
    manifest["graph_view"]["digest"] = manifest["split"]["view_digest"] = "e" * 64
    (directory / "cohort.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    message = _refused(cli, site, _argv(site, "kg-test"), ValueError,
                       "this build has no node mapping between graph views", capsys=capsys)
    assert "That is a capability this build lacks, not a research rule" in message


def test_a_file_without_a_model_spec_is_refused(cli, site, capsys):
    from src.models.bundle.record import ModelRecordError
    from tests.fixtures.registered_cohort import write_checkpoint_without_model_spec

    history = site.root / "history" / "best.pt"
    history.parent.mkdir()
    write_checkpoint_without_model_spec(history, data_dir=site.data_dir)
    _refused(cli, site, _argv(site, "kg-test", history), ModelRecordError,
             "written before the model spec", capsys=capsys)


def test_a_protocol_and_a_data_kind_that_do_not_match_are_refused_before_the_checkpoint(
        cli, site, tmp_path, capsys):
    """The LP protocol on a supplied cohort version or with --split, and a task
    protocol on a KG edge set: refused by the protocol's data kinds right
    after the version is read (or the split resolved), before any file of the
    workspace or the checkpoint is opened."""
    workspace = site.data_dir
    _refused(cli, site, _argv(site, SUPPLIED_LABEL), ValueError,
             "the kg_link_prediction measurement does not measure a supplied cohort version; "
             "it measures a KG edge set.", unopened=(site.base, workspace), capsys=capsys)
    _refused(cli, site, _argv(site, "kg-test", site.disease, mode="disease"), ValueError,
             "the disease measurement does not measure a KG edge set; it measures a generated "
             "split or a supplied cohort version.", unopened=(site.disease, workspace),
             capsys=capsys)
    _refused(cli, site, ["--mode", LP, "--checkpoint", str(site.base), "--data-dir",
                         str(site.data_dir), "--split", "val", "--output",
                         str(tmp_path / "out.json"), "--device", "cpu"], ValueError,
             "the kg_link_prediction measurement does not measure a generated split; it "
             "measures a KG edge set.", unopened=(site.base, workspace), capsys=capsys)
    assert not (tmp_path / "out.json").exists()


def test_training_evidence_is_refused_for_a_kg_edge_set(cli, site, tmp_path, capsys):
    """R10 is a supplied cohort version's: `--training-evidence` with a KG edge
    set is refused right after its cohort.json is judged, its pairs, the
    workspace and the checkpoint unread."""
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    message = _refused(cli, site, _argv(site, "kg-test", None, "--training-evidence",
                                        str(evidence)), ValueError,
                       "--training-evidence applies to a supplied cohort version only",
                       unopened=(site.base, site.data_dir,
                                 site.registry / "versions" / "kg-test" / "pairs.bin"),
                       capsys=capsys)
    assert "'kg-test' is a KG edge set (test)" in message


def test_a_version_with_no_pair_is_refused_by_its_counts(cli, site, capsys):
    """A KG edge set whose part left every edge without a negative (a
    consistent, hand-made version: no pair, an empty pairs file) is refused
    by the protocol's need before anything else is read."""
    from src.models.bundle.lp_head import lp_pairs_digest

    directory = site.registry / "versions" / "kg-test"
    manifest = json.loads((directory / "cohort.json").read_text())
    positive = manifest["counts"]["positive_edges"]
    manifest["counts"].update(pairs=0, edges_without_negative=positive)
    empty = torch.empty(0, dtype=torch.long)
    manifest["pairs"].update(rows=0, sha256=_sha(b""),
                             lp_pairs_digest=lp_pairs_digest(empty, empty, empty, empty),
                             batch_pairs=[0] * len(manifest["pairs"]["batch_pairs"]))
    (directory / "pairs.bin").write_bytes(b"")
    (directory / "cohort.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    assert _version(site, "kg-test").manifest["counts"]["pairs"] == 0
    _refused(cli, site, _argv(site, "kg-test"), ValueError,
             "this KG edge set holds no (positive, negative) pair",
             unopened=(site.base, directory / "pairs.bin", site.data_dir),
             capsys=capsys)


# ---------------------------------------------------------------------------
# A cancelled or failed run, and --record
# ---------------------------------------------------------------------------
def test_a_run_cancelled_while_it_scores_leaves_nothing(cli, site, monkeypatch):
    """A cancel (an interrupt) while the pairs are scored ends the run before
    its run directory exists: no report, no scores, no usage entry."""
    from src.models.bundle import lp_head

    def cancelled(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(lp_head, "score_lp_pairs", cancelled)
    before = _tree(site.registry)
    with pytest.raises(KeyboardInterrupt):
        cli.main(_argv(site, "kg-test"))
    _nothing_written(site, before)


@pytest.mark.parametrize("error", [OSError(28, "No space left on device"), KeyboardInterrupt()],
                         ids=["disk-full", "interrupted"])
def test_a_run_whose_scores_fail_midway_leaves_no_successful_report(cli, site, monkeypatch,
                                                                    error):
    """The scores are streamed into the run's new directory chunk by chunk and
    placed whole, and the report is published only after them: a failure (or
    an interrupt) among the chunks leaves a run directory with neither file,
    an interrupted write that no listing reads as a report, and the next run
    is kept beside it."""
    from src.evaluation import task_measurement
    from src.evaluation.cohort_registry import list_report_runs

    original = task_measurement.LinkPredictionResult.scores_chunks

    def failing(self, rows=65536):
        chunks = original(self, rows=2)
        yield next(chunks)
        raise error

    monkeypatch.setattr(task_measurement.LinkPredictionResult, "scores_chunks", failing)
    with pytest.raises((cli.ReportNotWritten, KeyboardInterrupt)) as info:
        cli.main(_argv(site, "kg-test"))
    if isinstance(error, OSError):
        assert "may exist, empty or holding only scores.bin" in str(info.value)
    run = _only_run(site, "kg-test")
    assert list(run.iterdir()) == []
    (listed,) = list_report_runs(_version(site, "kg-test"))
    assert listed.directory == run
    assert not (site.base.parent / LEDGER).exists()

    monkeypatch.setattr(task_measurement.LinkPredictionResult, "scores_chunks", original)
    assert cli.main(_argv(site, "kg-test")) == 0
    kept = _only_run(site, "kg-test", run)
    assert sorted(p.name for p in kept.iterdir()) == ["report.json", "scores.bin"]
    assert list(run.iterdir()) == []


def test_a_run_whose_report_fails_after_its_scores_leaves_an_interrupted_write(
        cli, site, monkeypatch):
    """The other half of the commit point: the scores file is published and
    the report's publication then fails. The run directory holds exactly
    `scores.bin` and no report: an interrupted write, which the store's
    listing names by its run and its scan never reads as a report; the
    failure names what may be there; nothing is in the usage history or a
    ledger; and the next run is kept beside it."""
    import errno

    import src.utils.file_modes as file_modes
    from src.evaluation.cohort_registry import list_report_runs
    from src.evaluation.report_store import read_usage_history, scan_report_store

    real = file_modes.publish_bytes

    def publish(target, data, **kwargs):
        if Path(target).name == "report.json":
            raise OSError(errno.ENOSPC, "No space left on device", str(target))
        return real(target, data, **kwargs)

    monkeypatch.setattr(file_modes, "publish_bytes", publish)
    with pytest.raises(cli.ReportNotWritten) as info:
        cli.main(_argv(site, "kg-test"))
    said = str(info.value)
    assert "No report.json was published" in said
    assert "may exist, empty or holding only scores.bin: an interrupted write" in said
    run = _only_run(site, "kg-test")
    assert [p.name for p in run.iterdir()] == ["scores.bin"]
    (listed,) = list_report_runs(_version(site, "kg-test"))
    assert (listed.directory, listed.problem) == (run, None)
    # Read as no report: the store's scan passes it over, as it passes over
    # every run directory without a report.json.
    scan = scan_report_store(_version(site, "kg-test"))
    assert (scan.reports, scan.unplaced, scan.digests) == ({}, (), {})
    entries, _problems = read_usage_history(_version(site, "kg-test"))
    assert [e for e in entries if e.get("kind") == "test_run"] == []
    assert not (site.base.parent / LEDGER).exists()

    monkeypatch.setattr(file_modes, "publish_bytes", real)
    assert cli.main(_argv(site, "kg-test")) == 0
    kept = _only_run(site, "kg-test", run)
    assert sorted(p.name for p in kept.iterdir()) == ["report.json", "scores.bin"]
    assert [p.name for p in run.iterdir()] == ["scores.bin"]


def test_record_registers_the_run_by_its_kind_with_its_exposure(cli, site, tmp_path, capsys):
    """`--record` on a KG edge set (research comparison Stage 6): the run is
    measured, its report kept and registered in the ledger beside the
    checkpoint, its record a KG edge set's: the key fields keep their meanings
    (the version's role, its manifest's digest), no R10, the run's counts with
    their units and each metric's status (``measured``), and its exposure block
    as the report wrote it, definition version and sources included. The
    `--outcome` file says registered. The one reader shows it as a KG edge
    set's view; the same run's same bytes again write nothing, and its bytes
    replaced are refused."""
    from src.api.routes.evaluations import KgEdgesRecordView, read_records
    from src.evaluation.registration import RegistrationRefused, register_report
    from src.evaluation.sidecar import read_ledger

    outcome = tmp_path / "outcome.json"
    assert cli.main(_argv(site, "kg-test", None, "--record", "--outcome", str(outcome))) == 0
    out = capsys.readouterr().out
    run = _only_run(site, "kg-test")
    report = _report(run)
    ledger_path = site.base.parent / LEDGER
    assert f"Registered as record 1 in {ledger_path}" in out
    ledger, _ = read_ledger(ledger_path)
    (record,) = ledger["records"]
    version = _version(site, "kg-test")
    assert (record["checkpoint_digest"], record["cohort_role"], record["cohort_digest"]) == (
        _sha(site.base.read_bytes()), "test", version.identity.sha256)
    assert record["run_id"] == run.name and record["r10"] is None
    assert record["cohort"]["kind"] == "kg_edges"
    assert record["cohort"]["counts"] == version.manifest["counts"]
    assert record["cohort"]["count_units"] == version.manifest["count_units"]
    assert record["cohort"]["pairs"] == version.manifest["pairs"]
    assert record["exposure"] == report["exposure"]
    assert record["exposure"]["definition_version"] == "kg-lp-exposure/v1"
    assert record["measured"] == {key: report["sampler_evidence"][key]
                                  for key in ("counts", "count_units", "metrics")}
    assert record["metrics"] == report["authoritative_metrics"]
    registration = json.loads(outcome.read_text())["registration"]
    assert (registration["registered"], registration["outcome"], registration["sequence"]) == (
        True, "registered", 1)

    ((_record, view),), unreadable = read_records(ledger, record["checkpoint_digest"])
    assert isinstance(view, KgEdgesRecordView) and unreadable == []
    assert (view.exposure.state, view.exposure.measured_edges.n) == (
        report["exposure"]["state"], report["exposure"]["measured_edges"]["n"])

    written = ledger_path.read_bytes()
    again = register_report(run / "report.json", checkpoint=site.base)
    assert again.outcome == "already_registered" and ledger_path.read_bytes() == written
    replaced = dict(report, n_ground_truth_absent=1)
    (run / "report.json").write_text(json.dumps(replaced))
    with pytest.raises(RegistrationRefused, match="is other bytes"):
        register_report(run / "report.json", checkpoint=site.base)
    assert ledger_path.read_bytes() == written


def test_without_record_the_report_is_kept_for_a_later_registration(cli, site, tmp_path,
                                                                   monkeypatch):
    """Without `--record`, a KG edge set's run is offered the register-later
    command as any kept report is (research comparison Stage 6), and both
    writes of the `--outcome` file say it was kept for a later
    registration."""
    written = []
    original = cli._write_outcome

    def watched(path, **kwargs):
        written.append((kwargs["complete"], kwargs["registration"]))
        return original(path, **kwargs)

    monkeypatch.setattr(cli, "_write_outcome", watched)
    outcome = tmp_path / "outcome.json"
    assert cli.main(_argv(site, "kg-test", None, "--outcome", str(outcome))) == 0
    why = "--record was not given; the report is kept for a later registration"
    assert [complete for complete, _ in written] == [False, True]
    for _complete, registration in written:
        assert registration == {"requested": False, "registered": False,
                                "outcome": "not_requested", "sequence": None, "ledger": None,
                                "why": why}
    assert json.loads(outcome.read_text())["registration"]["why"] == why


def test_a_failure_after_publishing_offers_the_register_later_command(cli, site):
    """A failure after a KG edge set's report was published says where the
    report is and gives the command that registers it, or says it already
    is, as for any registered version."""
    version = _version(site, "kg-test")
    inputs = SimpleNamespace(version=version, data_kind="kg_edges",
                             checkpoint=SimpleNamespace(identity=SimpleNamespace(
                                 path=str(site.base))))
    report = version.directory / "reports" / "run" / "report.json"
    said = str(cli._failure_after_publishing(OSError(5, "Input/output error"), inputs, report))
    assert said == (f"[Errno 5] Input/output error. The report was written: {report}. To "
                    "register it, or to see that it already is: "
                    f"{cli.register_later_command(report, site.base)}")


def test_the_store_scan_builds_and_groups_kg_edge_set_runs(cli, site):
    """The report store's one scan builds a KG edge set's reports into records
    and groups them by condition group (research comparison Stage 6): two
    runs of the same weights on the same pairs are one group with both runs
    named, their figures equal (CPU, no generator drawn); another model's run
    is a group of its own. Nothing is left unbuilt or unplaced."""
    from src.evaluation.report_store import scan_report_store
    from src.evaluation.sidecar import record_key

    assert cli.main(_argv(site, "kg-test")) == 0
    first = _only_run(site, "kg-test")
    assert cli.main(_argv(site, "kg-test")) == 0
    second = _only_run(site, "kg-test", first)
    assert cli.main(_argv(site, "kg-test", site.disease)) == 0
    child = _only_run(site, "kg-test", first, second)

    scan = scan_report_store(_version(site, "kg-test"))
    assert scan.unbuilt == {} and scan.unplaced == ()
    groups = {runs: key for key, runs in scan.runs.items()}
    assert set(groups) == {tuple(sorted((first.name, second.name))), (child.name,)}
    pair = groups[tuple(sorted((first.name, second.name)))]
    assert scan.state(pair) == "uncontested"
    record = scan.reports[pair][first.name][1]
    assert record["cohort"]["kind"] == "kg_edges" and record_key(record) == pair


# ---------------------------------------------------------------------------
# The manifest's rules, per kind of data
# ---------------------------------------------------------------------------
def test_the_manifest_s_rules_are_per_kind_without_a_new_field():
    """No field is added (`build_record` reads the exact field set); the
    rules become per kind: a version is recorded exactly for a registered
    version of any kind, and its kind is the one its block states (a block
    stating none is a supplied cohort's); R10 exactly for a supplied version;
    the relations fixed by whether a version is recorded. What a supplied
    version's and a generated split's report requires is unchanged."""
    from dataclasses import fields as dataclass_fields

    from src.evaluation import protocols
    from src.evaluation.measurement import MeasurementManifest, expected_relations
    from src.evaluation.sidecar import NON_SEMANTIC_FIELDS, SEMANTIC_MANIFEST_FIELDS
    from tests.unit.test_evaluation_sidecar import cohort_version, registered_manifest

    names = {f.name for f in dataclass_fields(MeasurementManifest)}
    assert names == set(SEMANTIC_MANIFEST_FIELDS) | set(NON_SEMANTIC_FIELDS)
    assert set(protocols.REGISTERED_KINDS) == {"supplied", "kg_edges", "generated"}

    kg = {"label": "kg-test", "role": "test", "kind": "kg_edges"}
    generated = {"label": "gen-val", "role": "val", "kind": "generated"}
    # Accepted: a KG edge set's run (no R10), a registered generated split's
    # (no R10), a supplied version's (R10), a generated split by --split.
    registered_manifest(cohort_kind="kg_edges", split="test", cohort_version=kg, r10=None)
    registered_manifest(cohort_kind="generated", split="val", cohort_version=generated,
                        r10=None)
    registered_manifest()
    registered_manifest(cohort_kind="generated", split="val", cohort_version=None, r10=None,
                        relations=expected_relations(registered=False))

    refused = [
        # R10 is a supplied version's only.
        ({"cohort_kind": "kg_edges", "cohort_version": kg},
         "R10 is recorded for a registered cohort version of kind supplied"),
        ({"r10": None}, "R10 is recorded for a registered cohort version of kind supplied"),
        # The kind is the one the block states; a block stating none is supplied.
        ({"cohort_kind": "kg_edges", "r10": None, "cohort_version": cohort_version()},
         "records cohort_kind 'kg_edges' beside one of kind 'supplied'"),
        ({"cohort_kind": "supplied", "cohort_version": kg}, "beside one of kind 'kg_edges'"),
        ({"cohort_kind": "cases", "r10": None, "cohort_version": {**kg, "kind": "cases"}},
         "a version's kind is one of the registry's (supplied, kg_edges, generated)"),
        # No version: a generated split only.
        ({"cohort_kind": "kg_edges", "cohort_version": None, "r10": None,
          "relations": expected_relations(registered=False)},
         "'kg_edges' cohort with no registered cohort version"),
        # The relations are fixed by whether a version is recorded.
        ({"cohort_kind": "kg_edges", "cohort_version": kg, "r10": None,
          "relations": expected_relations(registered=False)}, "relation states"),
    ]
    for overrides, says in refused:
        with pytest.raises(ValueError, match=re.escape(says)):
            registered_manifest(**overrides)
