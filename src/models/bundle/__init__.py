"""
The model bundle: spec, builder and the shared graph-side modules.
=================================================================
`docs/working/scorer-retraining/README.md` §3. A checkpoint's model is one
`ModelBundle` built by `build_bundle(spec, view)` from its versioned
`ModelSpec`; the patient encoder and task heads live in `src.models.tasks`.

The spec and the model record are read without torch (the API, the WebUI and
the inventory check a checkpoint's stage and task before anything is built);
the builder and the modules load torch when first used.

Module: src/models/bundle/__init__.py
"""
from src.models.bundle.spec import (
    SPEC_VERSION,
    ModelSpec,
    SpecError,
    ViewShape,
    finetune_spec_from_parent,
    reference_spec,
)


def __getattr__(name):
    if name in ("ModelBundle", "build_bundle"):
        from src.models.bundle import builder

        return getattr(builder, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ModelBundle",
    "ModelSpec",
    "SPEC_VERSION",
    "SpecError",
    "ViewShape",
    "build_bundle",
    "finetune_spec_from_parent",
    "reference_spec",
]
