"""
`scripts/build_index.py` — the index builder's CLI contract.
============================================================
Nothing exercised this script before. That mattered once the vector-index
subsystem was detached from the diagnosis pipeline but deliberately retained as
an operable standalone tool: "retained and operable" is a claim, and a script no
test has ever run is not evidence for it. (The `make index` target that was
supposed to be its entry point had, in fact, never worked — it passed only
`--config` and failed argument validation on every invocation.)

The tests that previously lived in `TestVectorIndexE2E` were *not* builder
coverage. They imported `create_index` and called `build_index()` in-process,
never invoking this script, so they verified none of its CLI contract,
embedding loading, ID mapping, node-type selection, or artifact naming.

This exercises the script the way an operator does: a subprocess, real
arguments, real files on disk, then load the artifacts back and search them.

The `--embeddings` (pre-exported NPZ) path is used rather than `--checkpoint`
because `build_index.py` imports torch lazily inside a function, so this path
needs neither torch nor PyG and runs wherever numpy and the backend are present.
`--backend voyager` is explicit: `auto` would resolve differently per platform
and turn this into a test of the resolver rather than of the builder.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("voyager", reason="Voyager backend not installed")

from src.retrieval import create_index  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_INDEX = REPO_ROOT / "scripts" / "build_index.py"

N_ENTITIES = 12
DIM = 16


@pytest.fixture
def npz_embeddings(tmp_path):
    """A pre-exported embeddings file in the layout the builder documents.

    Arrays keyed by node type, plus an ``id_mappings`` entry holding a JSON
    string of ``{node_type: {entity_id: row_index}}``.
    """
    rng = np.random.default_rng(20260811)
    vectors = rng.standard_normal((N_ENTITIES, DIM)).astype(np.float32)
    id_map = {f"MONDO:{i:07d}": i for i in range(N_ENTITIES)}

    path = tmp_path / "embeddings.npz"
    np.savez(
        path,
        disease=vectors,
        id_mappings=np.array(json.dumps({"disease": id_map})),
    )
    return path, id_map, vectors


def _run_builder(*args):
    """Invoke the script as an operator would, on the interpreter running the tests.

    sys.executable rather than "python": a bare name resolves against PATH, which
    is only this environment when a venv happens to be active. Otherwise the
    subprocess runs an interpreter without the project's dependencies and fails
    for a reason that has nothing to do with what is being tested.
    """
    return subprocess.run(
        [sys.executable, str(BUILD_INDEX), *map(str, args)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_builder_writes_loadable_searchable_artifacts(npz_embeddings, tmp_path):
    """The whole operator path: build from NPZ, then load and search the result."""
    npz_path, id_map, vectors = npz_embeddings
    out_base = tmp_path / "workspace" / "vector_index"

    result = _run_builder(
        "--embeddings", npz_path,
        "--output", out_base,
        "--node-types", "disease",
        "--backend", "voyager",
    )
    assert result.returncode == 0, (
        f"builder exited {result.returncode}\n--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )

    # Artifact naming is part of the contract: the builder appends the node type,
    # and the backend appends its own suffixes. An operator pointing the pipeline
    # at the wrong name is a real failure mode.
    artifact_base = out_base.parent / f"{out_base.name}_disease"
    index_file = artifact_base.with_suffix(".voyager")
    ids_file = artifact_base.with_suffix(".ids.json")
    assert index_file.exists(), f"missing index artifact {index_file}"
    assert ids_file.exists(), f"missing id-map artifact {ids_file}"

    # The original entity IDs must survive the round trip — an index that returns
    # row numbers instead of MONDO ids is useless to every caller.
    index = create_index(backend="voyager", dim=DIM)
    index.load(artifact_base)
    assert len(index) == N_ENTITIES

    query = vectors[3]
    results = index.search(query, top_k=5)
    assert results, "search returned nothing from a freshly built index"

    returned_ids = {entity_id for entity_id, _ in results}
    assert returned_ids <= set(id_map), (
        f"search returned ids absent from the source mapping: {returned_ids - set(id_map)}"
    )
    assert "MONDO:0000003" in returned_ids, (
        "querying with an indexed vector did not retrieve its own entity"
    )


def test_builder_honours_node_type_selection(npz_embeddings, tmp_path):
    """A node type absent from the NPZ must not silently produce an artifact."""
    npz_path, _, _ = npz_embeddings
    out_base = tmp_path / "vector_index"

    result = _run_builder(
        "--embeddings", npz_path,
        "--output", out_base,
        "--node-types", "gene",
        "--backend", "voyager",
    )

    gene_artifact = (out_base.parent / f"{out_base.name}_gene").with_suffix(".voyager")
    assert not gene_artifact.exists(), (
        "builder wrote an artifact for a node type that has no embeddings"
    )


def test_builder_rejects_a_missing_embeddings_file(tmp_path):
    """A wrong path must fail loudly rather than produce an empty index."""
    result = _run_builder(
        "--embeddings", tmp_path / "does-not-exist.npz",
        "--output", tmp_path / "vector_index",
        "--node-types", "disease",
        "--backend", "voyager",
    )
    assert result.returncode != 0, (
        f"builder reported success for a nonexistent input\nstdout:\n{result.stdout}"
    )


def test_builder_requires_a_source():
    """Regression guard for the defect that made `make index` useless.

    The old Makefile target passed only `--config`, so the builder always exited
    on argument validation. Whatever entry point exists must supply a source.
    """
    result = _run_builder("--config", "configs/deployment.yaml")
    assert result.returncode != 0
    assert "--checkpoint" in result.stderr and "--embeddings" in result.stderr, (
        f"expected a message naming the required source arguments, got:\n{result.stderr}"
    )
