"""
Path-form verifiers that fail when reached, and path hashes that lie.
====================================================================
Contract M2.1's dynamic pin (S9), brought forward for each entry point as it
moves (S6, S8): run the entry point with every path-form verifier raising and
every path hash returning a poison digest, then assert the run completes and no
poison reaches its record. A record built only from the readers' results passes;
one that took a digest from a path at any recording site fails.

`true_digest_for` names files whose path hash must stay true: the `kg.json`
identification check (`check_unparsed_kg_json`, M2.1 decision 1) hashes the file
by path on purpose, records nothing, and would otherwise refuse the run.

Module: tests/fixtures/path_forms.py
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

POISON = "0" * 64


def poison_path_forms(monkeypatch, *, true_digest_for: Iterable[Path] = ()) -> None:
    """Make every path-form verifier raise and every other path hash return
    `POISON`, for the rest of the test."""
    import src.evaluation.cohort as cohort
    import src.kg.artifacts as artifacts
    import src.utils.fingerprint as fingerprint

    real_file_sha256 = fingerprint.file_sha256
    keep = {os.path.realpath(path) for path in true_digest_for}

    def file_sha256(path):
        if os.path.realpath(path) in keep:
            return real_file_sha256(path)
        return POISON

    def reached(*args, **kwargs):
        raise AssertionError("a path form was reached")

    # `artifacts` and `cohort` import `file_sha256` inside their functions, so
    # patching it at its source reaches every path hash there.
    monkeypatch.setattr(fingerprint, "file_sha256", file_sha256)
    monkeypatch.setattr(fingerprint, "compute_input_digests", reached)
    for module, name in (
        (artifacts, "verify_graph_artifacts"),
        (artifacts, "verify_graph_source"),
        (cohort, "verify_graph_artifacts"),
        (cohort, "verify_generated_cohorts"),
    ):
        monkeypatch.setattr(module, name, reached, raising=False)
