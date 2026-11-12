"""
P0-4: the registry's schema-1 reads and the supplied importer's bytes.
=====================================================================
`docs/working/research-comparison/README.md` §5 Stage 1 (golden P0-4) and
Stage 4: the registry gains schema 2 with an explicit kind through the same
parser and validator, "schema 1 reads as `supplied`, and supplied imports keep
writing schema 1 byte-identically". This pins, for two fixed synthetic
sources over the demo workspace (one without genes, one listing causal and
candidate genes; `tests/fixtures/goldens/values.p0_4_values`):

- the exact `cohort.json` bytes the importer's preview and commit write
  (`p0_4_cohort_graph.json`, `p0_4_cohort_genes.json`), with the code revision
  it records fixed;
- `resolve_version`'s result for each version (manifest, digest of the bytes
  read, directory and samples path relative to the registry) and
  `cohort_version_block`'s block with the digest of its bytes, and the
  digests of the source and samples files (`p0_4_registry.json`).

The demo workspace's own digests are compared first: when they differ, the
importer was given other inputs (another torch build can write other tensor
bytes), and the test says so rather than blaming the importer.

Module: tests/unit/test_golden_p0_4_registry.py
"""
from __future__ import annotations

import pytest

pytest.importorskip("torch")

from tests.fixtures.goldens import encode, golden, golden_bytes  # noqa: E402
from tests.fixtures.goldens.values import p0_4_values  # noqa: E402


def test_p0_4_the_supplied_importer_writes_and_reads_schema_1_as_before(tmp_path):
    """P0-4: the workspace the sources are imported over is the golden's; then
    each version's `cohort.json` bytes, its schema-1 `resolve_version` result,
    its report block and the files' digests equal the golden, and the
    golden's bytes are reproduced."""
    values, written = p0_4_values(tmp_path)
    expected = golden("p0_4_registry.json")
    assert values["workspace"] == expected["workspace"], (
        "the demo workspace's bytes differ from the golden's in this environment, so the "
        "importer was given other inputs; the importer is not what differs")
    assert sorted(written) == sorted(expected["versions"]) == ["genes", "graph"]
    for name, want in expected["versions"].items():
        got = values["versions"][name]
        assert got["resolve_version"]["manifest"]["schema_version"] == 1, name
        assert written[name] == golden_bytes(f"p0_4_cohort_{name}.json"), name
        for key in want:
            assert got[key] == want[key], (name, key)
    assert encode(values) == golden_bytes("p0_4_registry.json")
