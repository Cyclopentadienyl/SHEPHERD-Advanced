"""
P0-1: kept reports and ledgers read exactly as on the unchanged tree.
=====================================================================
`docs/working/research-comparison/README.md` §4 invariant 1 and §5 Stage 1
(golden P0-1). The corpus `tests/fixtures/goldens/p0_1_corpus/` is kept
artefacts of every kind that exists at 2da28cf, built once by
`tests/fixtures/goldens/generate.py` (its docstring says from what, and what
was normalised):

- two schema-1 cohort versions and the reports their stores keep: disease,
  gene and patients-like-me task reports as `scripts/measure_scorer.py`
  writes them, the suite's in-memory disease report and a second run of it,
  and a kept Mode C and a kept Mode D report;
- a schema-4 ledger registering every one of them, a schema-3 ledger whose one
  record has the second run merged in, and that file after today's writer
  appended the second run (schema 4 from its run-identity boundary).

**Strict** (`p0_1_records.json`, byte-identical for the whole round): for
each kept report the record `build_record` makes and its key, where the
store's one scan (`scan_report_store`) places it, each version's scan
listing, and for each ledger record its key, run identity and possible
overlaps (`tests/fixtures/goldens/values.p0_1_values`).

**Views** (`p0_1_views.json`): the API's `RecordView` of each ledger record
(`read_records`). A later stage may change a view only by a named addition,
updating this golden in the commit that adds it.

Module: tests/unit/test_golden_p0_1_kept_reports.py
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")  # `read_records` is the API route's

from tests.fixtures.goldens import CORPUS, encode, golden, golden_bytes  # noqa: E402
from tests.fixtures.goldens.values import p0_1_values  # noqa: E402


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory):
    """The P0-1 values computed again from the committed corpus."""
    return p0_1_values(tmp_path_factory.mktemp("p0_1"))


def test_p0_1_the_corpus_holds_every_kind_kept_today():
    """P0-1's corpus is representative, and every kept report in it is
    pinned: the three task protocols as measured, the kept Mode C and Mode D
    reports, a CPU run and CUDA runs with deterministic algorithms on and off,
    a schema-3 merged record (a corroborating report, no run), and a schema-4
    file whose historical record and named run may overlap."""
    records = golden("p0_1_records.json")
    kept = sorted(f"{p.parents[2].name}/{p.parent.name}"
                  for p in (CORPUS / "registry").glob("versions/*/reports/*/report.json"))
    assert sorted(f"{r['label']}/{r['run_id']}" for r in records["reports"]) == kept
    assert {r["record"]["scoring"]["mode"] for r in records["reports"]} == {
        "disease", "gene", "patients_like_me", "C", "D"}
    views = golden("p0_1_views.json")["ledgers"]
    shown = [v for listing in views.values() for c in listing for v in c["records"]]
    assert {v["environment_class"] for v in shown} == {
        "CUDA, deterministic algorithms on", "CUDA, deterministic algorithms off",
        "CPU (development; a different measurement from a CUDA run)"}
    merged = records["ledgers"]["schema3_merged.json"]
    assert merged["schema_version"] == 3
    assert [(r["run_id"], r["historical"]) for r in merged["records"]] == [(None, True)]
    assert [v["n_corroborating"] for c in views["schema3_merged.json"] for v in c["records"]] == [1]
    appended = records["ledgers"]["schema3_appended.json"]
    assert (appended["schema_version"], appended["run_identity_from_sequence"]) == (4, 2)
    assert [r["overlaps"] for r in appended["records"]] == [[2], [1]]


def test_p0_1_each_kept_report_builds_the_same_record_and_store_key(regenerated):
    """P0-1 strict: `build_record`'s record (and so the measurement semantics
    and scoring semantics digests in it), its key, the store scan's key and
    record, each version's scan listing and each ledger record's key, run
    identity and possible overlaps equal the golden, and its bytes are the
    golden's bytes. Compared report by report first, so a failure names the
    report."""
    strict, _views = regenerated
    expected = golden("p0_1_records.json")
    assert ([(r["label"], r["run_id"]) for r in strict["reports"]]
            == [(r["label"], r["run_id"]) for r in expected["reports"]])
    for got, want in zip(strict["reports"], expected["reports"], strict=True):
        assert got == want, f"{want['label']}/{want['run_id']}"
        assert got["scan_key"] == got["record_key"] and got["scan_problem"] is None
        assert got["scan_record_sha256"] == got["record_sha256"]
    assert strict["versions"] == expected["versions"]
    for name, want in expected["ledgers"].items():
        assert strict["ledgers"][name] == want, name
    assert encode(strict) == golden_bytes("p0_1_records.json")


def test_p0_1_the_api_views_of_the_kept_ledgers(regenerated):
    """P0-1 views: every ledger record's `RecordView` (and every unreadable
    one, of which the corpus has none) equals the golden, ledger by ledger and
    checkpoint by checkpoint, and its bytes are the golden's bytes."""
    _strict, views = regenerated
    expected = golden("p0_1_views.json")
    assert sorted(views["ledgers"]) == sorted(expected["ledgers"])
    for name, listing in expected["ledgers"].items():
        for got, want in zip(views["ledgers"][name], listing, strict=True):
            assert got == want, f"{name}: {want['checkpoint_digest'][:12]}"
    assert encode(views) == golden_bytes("p0_1_views.json")
