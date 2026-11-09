"""
P0-2: the protocol judgments and their sentences, as on the unchanged tree.
==========================================================================
`docs/working/research-comparison/README.md` §5 Stage 1 (golden P0-2) and
Stage 3: the protocols are re-expressed by capability "with the same
acceptance and sentences". This pins, for every record the model spec can
write (a base with no task; a finetune with the disease, gene or
patients-like-me head), for no record read and for three records the spec
refuses (a finetune with no head or an unknown head, a base with a head:
only they reach `available`'s sentence for a record no protocol takes and
`model_refusal`'s without "It can be measured by"), every protocol name and
five names that are none (retired modes, a wrong case, an empty name, None),
and twelve cohort-version counts cases (none; without genes; each gene need
met or missing; counts of the wrong type), what
`src/evaluation/protocols.py`'s `available`, `model_refusal`,
`inputs_refusal` and `refusal` return, with their sentences, and `names()`
and the two standing sentences (`tests/fixtures/goldens/values.p0_2_values`).
That is every sentence the module can say for these inputs; the one it has
for a need without words (`inputs_refusal`) no protocol of the table has.

A later stage that adds a protocol (Stage 5) changes this golden by name in
its own commit; the three existing rows' acceptance and sentences do not
change.

Module: tests/unit/test_golden_p0_2_protocols.py
"""
from __future__ import annotations

from tests.fixtures.goldens import encode, golden, golden_bytes
from tests.fixtures.goldens.values import P0_2_RECORDS, P0_2_UNWRITABLE_RECORDS, p0_2_values


def test_p0_2_the_records_judged_are_every_one_the_spec_can_write():
    """P0-2's coverage: each (learning stage, task) `src.models.bundle.spec`
    can write, a base having no task, and the record of no record read."""
    from src.models.bundle.spec import LEARNING_STAGES, TASKS

    writable = {("base", None)} | {("finetune", task) for task in TASKS}
    assert {stage for stage, _task in writable} == set(LEARNING_STAGES)
    assert set(P0_2_RECORDS) == writable | {(None, None)}


def test_p0_2_the_records_the_spec_cannot_write_reach_what_no_written_one_does():
    """P0-2's coverage of the sentences for a record no protocol takes: none
    of `P0_2_UNWRITABLE_RECORDS` is a record the spec can write or the record
    of none read, and between them they reach an `available` reason and a
    `model_refusal` sentence that no record of `P0_2_RECORDS` reaches, which
    the golden holds."""
    from src.models.bundle.spec import TASKS

    writable = {("base", None)} | {("finetune", task) for task in TASKS}
    unwritable = set(P0_2_UNWRITABLE_RECORDS)
    assert len(unwritable) == len(P0_2_UNWRITABLE_RECORDS)
    assert not unwritable & (writable | {(None, None)})
    expected = golden("p0_2_protocols.json")
    for section, key in (("available", "reason"), ("model_refusal", "refusal")):
        said = {True: set(), False: set()}
        for entry in expected[section]:
            said[(entry["learning_stage"], entry["task"]) in unwritable].add(entry[key])
        assert said[True] - said[False], section


def test_p0_2_every_protocol_judgment_and_its_sentence():
    """P0-2: each judgment equals the golden, section by section (so a failure
    names the judgment), and the bytes are the golden's bytes."""
    values = p0_2_values()
    expected = golden("p0_2_protocols.json")
    assert sorted(values) == sorted(expected)
    for section in ("names", "sentences", "counts_cases", "available"):
        assert values[section] == expected[section], section
    for section in ("model_refusal", "inputs_refusal", "refusal"):
        assert len(values[section]) == len(expected[section]), section
        for got, want in zip(values[section], expected[section], strict=True):
            assert got == want, section
    assert encode(values) == golden_bytes("p0_2_protocols.json")
