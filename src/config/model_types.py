"""
Model-type constants — dependency-free.
=======================================
The supported GNN conv types, kept in a torch-free module so training scripts,
API services, and path helpers can import them without pulling the model stack
(``src/models/gnn`` eagerly imports torch via its package __init__).

``src/models/gnn/layers.py`` (the GNN factory) imports these.

Module: src/config/model_types.py
"""
from __future__ import annotations

# Conv types the GNN factory (src/models/gnn/layers.py) can build. Adding a new
# architecture = implement its branch in HeteroGNNLayer AND add its name here.
SUPPORTED_CONV_TYPES = ("hgt", "gat", "sage")

# Encoder kinds a model bundle's spec can name (src/models/bundle/spec.py,
# README §3.3): the reference homogeneous GATv2 and the non-reference
# heterogeneous encoders behind the same contract. Here, torch-free, so the run
# layout (src/utils/checkpoint_paths.py) files each new run under its spec's
# kind; the listing reads every directory under runs/, whatever its name.
ENCODER_KINDS = ("gatv2", "hetero_gat", "hgt", "sage")

# The top-level keys a model bundle checkpoint's record owns (provenance §9.1),
# here, torch-free, so the run listing can carry them from its one read of a
# file; src/models/bundle/record.py reads and checks them (its RECORD_KEYS).
MODEL_RECORD_KEYS = ("model_spec", "graph_view", "learning_stage", "task", "operation",
                     "parent", "family_id", "weight_transfer", "data_roles",
                     "graph_exposure", "loss_identity", "scoring_identity", "selection")

# Every top-level key the one reader of records (`read_model_record`) reads
# besides `record_version`: the record's keys and the training record's
# `training_input_digests`, which names some of the record's inputs a second
# time and is compared with the record's own claims ("one input, one claim").
# The run listing carries exactly these from its one read of a file, so a list
# judges a file as a load of it does.
MODEL_RECORD_READ_KEYS = MODEL_RECORD_KEYS + ("training_input_digests",)

# What every reader says of a checkpoint without a model spec: the record
# reader (`read_model_record`, and so every list, load, measurement and
# training read of one) and the run plan's record-version check
# (`src.utils.run_plan`), in one phrase. Nothing is guessed from such a file's
# weights; it stays listed, with this reason.
WRITTEN_BEFORE_SPEC = ("written before the model spec: not readable by this build; "
                       "start a new run")


__all__ = ["SUPPORTED_CONV_TYPES", "ENCODER_KINDS", "MODEL_RECORD_KEYS",
           "MODEL_RECORD_READ_KEYS", "WRITTEN_BEFORE_SPEC"]
