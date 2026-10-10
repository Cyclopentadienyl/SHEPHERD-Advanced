"""
The shared readers parse each file from one read and return its identity.
=========================================================================
Contract M2.1, S4: `read_graph_artifacts`, `read_samples`, `read_split_manifest`
and `read_checkpoint` each parse the bytes `read_once` returned and return what
they parsed with a `ReadIdentity`, never the bytes. For each:

- the identity is the digest of the bytes parsed, and each file is opened once;
- a replacement after the read, by atomic rename or in-place rewrite, changes
  neither what was parsed nor the identity;
- the release check passes: no raw buffer, and no `BytesIO` over one, survives
  into the reader's next read or past its return;
- the result holds identities, never a `FileRead` or bytes.

Module: tests/unit/test_shared_readers.py
"""
from __future__ import annotations

import hashlib
import json

import pytest

from src.utils.fingerprint import FileRead, ReadIdentity
from tests.fixtures.generated_workspace import write_generated_workspace
from tests.fixtures.opens import count_opens
from tests.fixtures.release import check_release
from tests.fixtures.replacement import replace_after_read

torch = pytest.importorskip("torch")

from src.kg.storage import file_storage  # noqa: E402
from src.kg.storage.file_storage import (  # noqa: E402
    GraphArtifactsRead,
    SamplesRead,
    read_graph_artifacts,
    read_samples,
)
from src.utils import checkpoint_io  # noqa: E402
from src.utils.checkpoint_io import CheckpointRead, read_checkpoint  # noqa: E402

TENSOR_FILES = {"node_features": "node_features.pt", "edge_indices": "edge_indices.pt",
                "num_nodes": "num_nodes.json"}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _holds_no_bytes(value):
    """Walks a reader's result: identities only, never a FileRead or a buffer."""
    if isinstance(value, (FileRead, bytes, bytearray, memoryview)):
        return False
    if isinstance(value, dict):
        return all(_holds_no_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_holds_no_bytes(v) for v in value)
    return True


@pytest.fixture
def workspace(tmp_path):
    root, _ = write_generated_workspace(tmp_path / "ws", train_ids=[0, 1, 2], val_ids=[3],
                                        training_fields=True)
    return root


@pytest.fixture
def other(tmp_path):
    root, _ = write_generated_workspace(tmp_path / "other", train_ids=[5, 6], val_ids=[7])
    return root


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "model.pt"
    torch.save({"model_state_dict": {"w": torch.arange(4096, dtype=torch.float32)},
                "epoch": 3}, path)
    return path


# ---------------------------------------------------------------------------
# read_graph_artifacts
# ---------------------------------------------------------------------------
class TestReadGraphArtifacts:
    def test_each_identity_is_the_digest_of_the_file_parsed(self, workspace):
        result = read_graph_artifacts(workspace)

        assert isinstance(result, GraphArtifactsRead)
        assert set(result.graph_data) == {"x_dict", "edge_index_dict", "num_nodes_dict"}
        assert result.reads == {
            role: ReadIdentity(path=workspace / name, sha256=_sha(workspace / name))
            for role, name in TENSOR_FILES.items()
        }
        expected = torch.load(workspace / "node_features.pt", weights_only=True)
        assert all(torch.equal(result.graph_data["x_dict"][k], expected[k]) for k in expected)
        assert result.graph_data["num_nodes_dict"] == json.loads(
            (workspace / "num_nodes.json").read_text(encoding="utf-8"))
        assert _holds_no_bytes(result.reads)

    def test_each_file_is_opened_once(self, workspace):
        with count_opens() as opens:
            read_graph_artifacts(workspace)

        assert {name: opens.opens(workspace / name) for name in TENSOR_FILES.values()} == (
            dict.fromkeys(TENSOR_FILES.values(), 1)
        )

    def test_tensors_are_loaded_onto_the_cpu(self, workspace):
        x_dict = read_graph_artifacts(workspace).graph_data["x_dict"]
        assert {tensor.device.type for tensor in x_dict.values()} == {"cpu"}

    def test_the_map_location_given_is_the_one_used(self, workspace):
        """The CPU test above cannot fail on a CPU-saved export; this one can."""
        graph = read_graph_artifacts(workspace, map_location="meta").graph_data

        assert {t.device.type for t in graph["x_dict"].values()} == {"meta"}
        assert {t.device.type for t in graph["edge_index_dict"].values()} == {"meta"}

    def test_a_graph_tensor_file_that_is_not_weights_is_refused(self, workspace):
        """`weights_only=True`: a graph file that would execute code is refused."""
        import argparse

        torch.save({"object": argparse.Namespace(arbitrary=True)},
                   workspace / "node_features.pt")

        with pytest.raises(Exception, match="Weights only load failed"):
            read_graph_artifacts(workspace)

    def test_an_absent_file_has_neither_data_nor_identity(self, workspace):
        (workspace / "num_nodes.json").unlink()

        result = read_graph_artifacts(workspace)

        assert "num_nodes_dict" not in result.graph_data
        assert set(result.reads) == {"node_features", "edge_indices"}

    @pytest.mark.parametrize("how", ["rename", "rewrite"])
    def test_a_replacement_after_the_read_changes_nothing_returned(
        self, monkeypatch, workspace, other, how
    ):
        target = workspace / "node_features.pt"
        original_digest = _sha(target)
        original = torch.load(target, weights_only=True)
        record = replace_after_read(monkeypatch, after=target, target=target,
                                    data=(other / "node_features.pt").read_bytes(), how=how)

        result = read_graph_artifacts(workspace)

        assert record.fired == 1
        assert result.reads["node_features"].sha256 == original_digest != _sha(target)
        assert all(torch.equal(result.graph_data["x_dict"][k], original[k]) for k in original)

    def test_every_buffer_is_released(self, monkeypatch, workspace):
        result, reads = check_release(monkeypatch, lambda: read_graph_artifacts(workspace),
                                      bytesio_modules=[file_storage])

        assert reads == tuple(workspace / name for name in TENSOR_FILES.values())
        assert set(result.reads) == set(TENSOR_FILES)


# ---------------------------------------------------------------------------
# read_samples
# ---------------------------------------------------------------------------
class TestReadSamples:
    def test_the_identity_is_the_digest_of_the_file_parsed(self, workspace):
        result = read_samples(workspace, "train")

        path = workspace / "train_samples.json"
        assert isinstance(result, SamplesRead)
        assert result.identity == ReadIdentity(path=path, sha256=_sha(path))
        assert [s.disease_id for s in result.samples] == [
            row["disease_id"] for row in json.loads(path.read_text(encoding="utf-8"))
        ]
        assert _holds_no_bytes(tuple(result))

    def test_three_fields_by_default(self, workspace):
        """Measurement reads three fields; the file carries five here."""
        samples = read_samples(workspace, "train").samples

        assert samples
        assert all(s.candidate_disease_ids is None and s.gene_ids is None for s in samples)

    def test_training_fields_carry_both_optional_fields(self, workspace):
        rows = json.loads((workspace / "train_samples.json").read_text(encoding="utf-8"))

        samples = read_samples(workspace, "train", training_fields=True).samples

        assert [(s.candidate_disease_ids, s.gene_ids) for s in samples] == [
            (row["candidate_disease_ids"], row["gene_ids"]) for row in rows
        ]
        assert all(s.candidate_disease_ids and s.gene_ids for s in samples)

    def test_the_file_is_opened_once(self, workspace):
        with count_opens() as opens:
            read_samples(workspace, "train")

        assert opens.opens(workspace / "train_samples.json") == 1

    def test_a_missing_split_is_refused_with_no_identity(self, workspace):
        with pytest.raises(FileNotFoundError, match="this workspace has: train, val"):
            read_samples(workspace, "test")

    @pytest.mark.parametrize("how", ["rename", "rewrite"])
    def test_a_replacement_after_the_read_changes_nothing_returned(
        self, monkeypatch, workspace, other, how
    ):
        target = workspace / "train_samples.json"
        original_digest = _sha(target)
        original_ids = [row["disease_id"] for row in json.loads(target.read_text())]
        record = replace_after_read(monkeypatch, after=target, target=target,
                                    data=(other / "train_samples.json").read_bytes(), how=how)

        result = read_samples(workspace, "train")

        assert record.fired == 1
        assert result.identity.sha256 == original_digest != _sha(target)
        assert [s.disease_id for s in result.samples] == original_ids

    def test_the_buffer_is_released(self, monkeypatch, workspace):
        _, reads = check_release(
            monkeypatch, lambda: read_samples(workspace, "val", training_fields=True)
        )

        assert reads == (workspace / "val_samples.json",)


# ---------------------------------------------------------------------------
# read_split_manifest
# ---------------------------------------------------------------------------
class TestReadSplitManifest:
    def test_the_identity_is_the_digest_of_the_manifest_parsed(self, workspace):
        from src.kg.artifacts import ManifestRead, read_split_manifest

        path = workspace / "split_manifest.json"
        result = read_split_manifest(workspace)

        assert isinstance(result, ManifestRead)
        assert result.identity == ReadIdentity(path=path, sha256=_sha(path))
        assert result.manifest == json.loads(path.read_text(encoding="utf-8"))
        assert _holds_no_bytes(tuple(result))

    def test_the_file_is_opened_once(self, workspace):
        from src.kg.artifacts import read_split_manifest

        with count_opens() as opens:
            read_split_manifest(workspace)

        assert opens.opens(workspace / "split_manifest.json") == 1

    def test_an_absent_manifest_is_refused(self, tmp_path):
        from src.kg.artifacts import read_split_manifest

        with pytest.raises(ValueError, match="has no split_manifest.json.*Rebuild it with"):
            read_split_manifest(tmp_path)

    def test_an_old_schema_is_refused(self, workspace):
        from src.kg.artifacts import read_split_manifest

        path = workspace / "split_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["schema_version"] = 2
        path.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(ValueError, match="schema 2.*recorded no export recipe"):
            read_split_manifest(workspace)

    @pytest.mark.parametrize("payload", [b"{ not json", b"\xff\xfe{}", b"[]"])
    def test_a_manifest_that_is_not_a_utf8_json_object_is_refused_by_name(
        self, workspace, payload
    ):
        from src.kg.artifacts import read_split_manifest

        path = workspace / "split_manifest.json"
        path.write_bytes(payload)

        with pytest.raises(ValueError, match="split_manifest.json is not") as refused:
            read_split_manifest(workspace)
        assert str(path) in str(refused.value)

    @pytest.mark.parametrize("how", ["rename", "rewrite"])
    def test_a_replacement_after_the_read_changes_nothing_returned(
        self, monkeypatch, workspace, other, how
    ):
        from src.kg.artifacts import read_split_manifest

        target = workspace / "split_manifest.json"
        original = target.read_bytes()
        record = replace_after_read(monkeypatch, after=target, target=target,
                                    data=(other / "split_manifest.json").read_bytes(), how=how)

        result = read_split_manifest(workspace)

        assert record.fired == 1
        assert result.identity.sha256 == hashlib.sha256(original).hexdigest() != _sha(target)
        assert result.manifest == json.loads(original)

    def test_the_buffer_is_released(self, monkeypatch, workspace):
        from src.kg.artifacts import read_split_manifest

        _, reads = check_release(monkeypatch, lambda: read_split_manifest(workspace))

        assert reads == (workspace / "split_manifest.json",)


# ---------------------------------------------------------------------------
# read_checkpoint
# ---------------------------------------------------------------------------
class TestReadCheckpoint:
    def test_the_identity_is_the_digest_of_the_checkpoint_loaded(self, checkpoint):
        result = read_checkpoint(checkpoint, map_location="cpu", weights_only=True)

        assert isinstance(result, CheckpointRead)
        assert result.identity == ReadIdentity(path=checkpoint, sha256=_sha(checkpoint))
        assert result.checkpoint["epoch"] == 3
        assert torch.equal(result.checkpoint["model_state_dict"]["w"],
                           torch.arange(4096, dtype=torch.float32))
        assert _holds_no_bytes([result.identity])

    def test_each_site_states_its_own_options(self, checkpoint):
        """No defaults: a caller cannot inherit a `weights_only` it did not choose."""
        with pytest.raises(TypeError):
            read_checkpoint(checkpoint)  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            read_checkpoint(checkpoint, map_location="cpu")  # type: ignore[call-arg]

    def test_the_options_given_are_the_ones_used(self, tmp_path, checkpoint):
        """Each site keeps its own: serving's `weights_only=False` loads what the safe
        loader refuses, and the device given is the one used."""
        import argparse

        pickled = tmp_path / "pickled.pt"
        torch.save({"object": argparse.Namespace(arbitrary=True)}, pickled)

        loaded = read_checkpoint(pickled, map_location="cpu", weights_only=False)
        on_meta = read_checkpoint(checkpoint, map_location="meta", weights_only=True)

        assert loaded.checkpoint["object"].arbitrary is True
        assert on_meta.checkpoint["model_state_dict"]["w"].device.type == "meta"

    def test_the_safe_loader_refuses_what_it_refuses_with_no_fallback(self, tmp_path):
        import argparse

        path = tmp_path / "pickled.pt"
        torch.save({"object": argparse.Namespace(arbitrary=True)}, path)

        with pytest.raises(Exception, match="Weights only load failed"):
            read_checkpoint(path, map_location="cpu", weights_only=True)

    def test_the_file_is_opened_once(self, checkpoint):
        with count_opens() as opens:
            read_checkpoint(checkpoint, map_location="cpu", weights_only=True)

        assert opens.opens(checkpoint) == 1

    def test_a_missing_checkpoint_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_checkpoint(tmp_path / "absent.pt", map_location="cpu", weights_only=True)

    @pytest.mark.parametrize("how", ["rename", "rewrite"])
    def test_a_replacement_after_the_read_changes_nothing_returned(
        self, monkeypatch, tmp_path, checkpoint, how
    ):
        original_digest = _sha(checkpoint)
        replacement = tmp_path / "replacement.pt"
        torch.save({"model_state_dict": {"w": torch.zeros(3)}, "epoch": 9}, replacement)
        record = replace_after_read(monkeypatch, after=checkpoint, target=checkpoint,
                                    data=replacement.read_bytes(), how=how)

        result = read_checkpoint(checkpoint, map_location="cpu", weights_only=True)

        assert record.fired == 1
        assert result.identity.sha256 == original_digest != _sha(checkpoint)
        assert result.checkpoint["epoch"] == 3

    def test_the_buffer_is_released(self, monkeypatch, checkpoint):
        _, reads = check_release(
            monkeypatch,
            lambda: read_checkpoint(checkpoint, map_location="cpu", weights_only=True),
            bytesio_modules=[checkpoint_io],
        )

        assert reads == (checkpoint,)
