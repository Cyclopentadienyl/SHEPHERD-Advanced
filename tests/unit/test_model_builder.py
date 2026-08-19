"""
`build_shepherd_model` — the one construction path, and the guards that keep it one.
====================================================================================
The deployed pipeline and the measurement harness must build the model the same
way, or a measurement stops describing the thing it claims to measure — silently,
because a model built from slightly different metadata still loads, still runs,
and still produces numbers. That is why construction is one function with two
callers rather than two functions that agree today.

What is tested here is therefore not "does it return a model" but the properties
that make sharing it worth anything: architecture comes from `graph_data` and not
from the checkpoint's own idea of the graph, both checkpoint layouts are
recognised, and every way of getting an unusable model out of it raises instead.
"""
import pytest

torch = pytest.importorskip("torch")

from src.models.gnn.shepherd_gnn import ShepherdGNN, build_shepherd_model  # noqa: E402
from tests.fixtures.synthetic_workspace import build_workspace  # noqa: E402


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("builder")
    data_dir, checkpoint_path = build_workspace(root)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    graph_data = {
        "x_dict": torch.load(data_dir / "node_features.pt", weights_only=True),
        "edge_index_dict": torch.load(data_dir / "edge_indices.pt", weights_only=True),
    }
    return checkpoint, graph_data


# ---------------------------------------------------------------------------
# What it builds
# ---------------------------------------------------------------------------
def test_builds_a_model_in_eval_mode(workspace):
    """Eval mode is part of the contract, not the caller's afterthought: a model
    left in train mode has dropout active, which makes scores non-deterministic
    for no reason a reader of the calling code would suspect."""
    checkpoint, graph_data = workspace

    model = build_shepherd_model(checkpoint, graph_data)

    assert isinstance(model, ShepherdGNN)
    assert model.training is False


def test_architecture_comes_from_graph_data_not_the_checkpoint(workspace):
    """The reason this function exists in one copy. The trainer builds from
    `graph_data` keys, which carry the reverse edges added for bidirectional
    message passing; a knowledge-graph object has only the forward ones. Building
    from the wrong source yields a model with fewer conv layers than the
    checkpoint expects.

    **The checkpoint's own `metadata` is corrupted here on purpose.** In the
    fixture the two sources agree, so a test that merely compared the built model
    against `graph_data` would pass just as happily if the builder read the
    checkpoint. Wrecking the field the builder must ignore is the only way to show
    that it ignores it.
    """
    checkpoint, graph_data = workspace
    misleading = {**checkpoint, "metadata": (["disease"], [("disease", "nonsense", "disease")])}

    model = build_shepherd_model(misleading, graph_data)

    assert set(model.metadata[1]) == set(graph_data["edge_index_dict"].keys())
    assert set(model.metadata[0]) == set(graph_data["x_dict"].keys())


def test_both_checkpoint_layouts_are_recognised(workspace):
    """Trainer format writes `model_state_dict`; the callback writes `state_dict`.
    Both are real and in the wild."""
    checkpoint, graph_data = workspace
    trainer_format = {**checkpoint, "model_state_dict": checkpoint["state_dict"]}
    del trainer_format["state_dict"]

    from_callback = build_shepherd_model(checkpoint, graph_data)
    from_trainer = build_shepherd_model(trainer_format, graph_data)

    for left, right in zip(from_callback.state_dict().values(),
                           from_trainer.state_dict().values()):
        assert torch.equal(left, right)


def test_device_placement_is_opt_in(workspace):
    """`None` leaves the model where it was built, so a caller that has already
    decided on a device is not overruled by a default."""
    checkpoint, graph_data = workspace

    placed = build_shepherd_model(checkpoint, graph_data, device=torch.device("cpu"))

    assert next(placed.parameters()).device.type == "cpu"


# ---------------------------------------------------------------------------
# Every way to get an unusable model out of it
# ---------------------------------------------------------------------------
def test_a_checkpoint_with_no_state_dict_raises(workspace):
    """Rather than returning an untrained model, which would score, rank, and
    report — plausibly and wrongly."""
    _, graph_data = workspace

    with pytest.raises(KeyError, match="model_state_dict"):
        build_shepherd_model({"config": {}, "epoch": 1}, graph_data)


@pytest.mark.parametrize("missing", ["x_dict", "edge_index_dict"])
def test_graph_data_missing_either_dict_raises(workspace, missing):
    """Both are load-bearing: one supplies the node types and input widths, the
    other the edge types. An absent one does not fail loudly on its own — it
    builds a different model."""
    checkpoint, graph_data = workspace
    incomplete = {k: v for k, v in graph_data.items() if k != missing}

    with pytest.raises(ValueError, match="x_dict"):
        build_shepherd_model(checkpoint, incomplete)


def test_a_state_dict_that_does_not_fit_raises_with_the_diagnosis_attached(workspace):
    """A bare `RuntimeError` from `load_state_dict` sends an operator looking for
    a bug. The common cause is a metadata-source mismatch, so the message says
    so and reports the edge-type count the model was built for."""
    checkpoint, graph_data = workspace
    mangled = {**checkpoint, "state_dict": {"gnn_layers.0.nonsense": torch.zeros(3)}}

    with pytest.raises(RuntimeError, match="trainer and inference derived metadata"):
        build_shepherd_model(mangled, graph_data)
