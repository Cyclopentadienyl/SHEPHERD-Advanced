"""A workspace in the shape the generator leaves one, for tests that consume it.

**Built through `build_split_manifest`, not by writing a plausible manifest.**
Every entry point now verifies the manifest against the exact sample bytes and
the disease sets they hold, so a hand-written manifest in a fixture would encode
this file's belief about that shape and keep passing after the real shape moved --
which is the class of defect the verification exists to catch, reproduced inside
its own tests.

Module: tests/fixtures/generated_workspace.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_CONFIG = {"min_phenotypes": 2, "max_phenotypes": 15, "phenotype_drop_rate": 0.3}


def profiles_for(disease_ids: Iterable[int]) -> Dict[int, Dict[str, List[int]]]:
    """A profile per disease, with phenotypes derived from the id."""
    return {
        int(d): {"phenotype_ids": [int(d), int(d) + 100], "gene_ids": [int(d)]}
        for d in disease_ids
    }


def one_sample_per_disease(split: str, ids: Sequence[int], profiles) -> List[Dict[str, Any]]:
    return [
        {"patient_id": f"SECRET-{split}-{i}",
         "phenotype_ids": list(profiles[d]["phenotype_ids"]),
         "disease_id": int(d)}
        for i, d in enumerate(ids)
    ]


def write_generated_workspace(
    root: Path,
    *,
    train_ids: Sequence[int],
    val_ids: Sequence[int],
    profiles: Optional[Dict[int, Dict[str, Any]]] = None,
    train_samples: Optional[List[Dict[str, Any]]] = None,
    val_samples: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
    kg_digest: Optional[str] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Write `train_samples.json`, `val_samples.json` and a real manifest.

    The manifest's artifact digests are taken from the files after they are
    written, exactly as `generate_training_samples` takes them, so the workspace
    satisfies `verify_generated_cohorts` without the fixture knowing what that
    checks.
    """
    from src.kg.disease_allocation import DiseaseAllocation, universe_digest
    from src.kg.sample_generator import build_split_manifest
    from src.utils.fingerprint import file_sha256

    root.mkdir(parents=True, exist_ok=True)
    train_ids, val_ids = [int(d) for d in train_ids], [int(d) for d in val_ids]
    profiles = profiles or profiles_for(train_ids + val_ids)
    rows = {
        "train": train_samples if train_samples is not None
        else one_sample_per_disease("train", train_ids, profiles),
        "val": val_samples if val_samples is not None
        else one_sample_per_disease("val", val_ids, profiles),
    }
    for split, samples in rows.items():
        (root / f"{split}_samples.json").write_text(json.dumps(samples))

    ordered = sorted(train_ids + val_ids)
    allocation = DiseaseAllocation(
        train=tuple((d, profiles[d]) for d in sorted(train_ids)),
        val=tuple((d, profiles[d]) for d in sorted(val_ids)),
        val_fraction_requested=len(val_ids) / max(len(ordered), 1),
        seed=0,
        universe_digest=universe_digest([(d, profiles[d]) for d in ordered]),
    )
    manifest = build_split_manifest(
        allocation=allocation,
        train_samples=rows["train"], val_samples=rows["val"],
        config=dict(config or DEFAULT_CONFIG),
        num_train=len(rows["train"]), num_val=len(rows["val"]),
        artifacts={
            "kg": kg_digest,
            "train_samples": file_sha256(root / "train_samples.json"),
            "val_samples": file_sha256(root / "val_samples.json"),
        },
    )
    (root / "split_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return root, manifest
