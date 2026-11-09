"""
Goldens taken on the unchanged tree (research comparison, Stage 1a).
====================================================================
`docs/working/research-comparison/README.md` §4 invariant 1 and §5 Stage 1
("Goldens on the unchanged tree"). Each file here pins what the code of
`main` 2da28cf computes, so a later stage that changes any of it is caught
by `tests/unit/test_golden_p0_*.py` and `tests/unit/test_pin_p0_5_*.py`:

- **P0-1** (`p0_1_records.json`, strict; `p0_1_views.json`, separate): for
  each kept report of the corpus `p0_1_corpus/` and each record of its
  ledgers, the record `build_record` makes, the report store's scan key and
  the ledger reading (strict, byte-identical over the round), and the API's
  `RecordView` (which a later stage may change by a named addition only, in
  the commit that adds it);
- **P0-2** (`p0_2_protocols.json`): every judgment of
  `src/evaluation/protocols.py` with its sentence;
- **P0-3** (`p0_3_base_validation.json`): base validation's fixed pairs,
  scores and returned dict on CPU;
- **P0-4** (`p0_4_registry.json`, `p0_4_cohort_graph.json`,
  `p0_4_cohort_genes.json`): the registry's schema-1 reads, the report block
  and the exact `cohort.json` bytes the supplied importer writes.

**How they were made.** `values.py` computes every pinned value from the
production functions; `generate.py` writes the files from those same
functions (``python -m tests.fixtures.goldens.generate`` from the repository
root) and built the P0-1 corpus once (``--corpus``). The tests compute the
values again and compare them with the committed bytes. A golden is updated
only by a change that names what it changes, and the strict ones not at all
this round.

**One encoding** (`encode`): ASCII JSON, keys sorted, two-space indent, no NaN
or infinity, one final newline. It is the ledger writer's and the registry's
encoding (`sidecar.write_ledger`, `cohort_registry.encode_json`), so a pinned
record has the bytes a ledger holds for it.

Module: tests/fixtures/goldens/__init__.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

#: Where the golden files are.
GOLDENS = Path(__file__).resolve().parent

#: The P0-1 corpus: a kept registry (two versions and their report stores)
#: and three ledgers, as `generate.build_corpus` wrote them. Never rewritten:
#: kept artefacts are read as they are.
CORPUS = GOLDENS / "p0_1_corpus"


def encode(value: Any) -> bytes:
    """The one encoding of every golden file (module docstring)."""
    return (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False)
            + "\n").encode("ascii")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def golden_bytes(name: str) -> bytes:
    """The committed bytes of the golden file `name`."""
    return (GOLDENS / name).read_bytes()


def golden(name: str) -> Any:
    """The committed golden file `name`, decoded."""
    return json.loads(golden_bytes(name))


__all__ = ["CORPUS", "GOLDENS", "encode", "golden", "golden_bytes", "sha256"]
