"""
Model-type constants — dependency-free.
=======================================
The supported GNN conv types, kept in a torch-free module so training scripts,
API services, and path helpers can import them without pulling the model stack
(``src/models/gnn`` eagerly imports torch via its package __init__).

``src/models/gnn/layers.py`` (the GNN factory) imports these.

It also holds `resolve_arch_params`, which recovers a checkpoint's architecture
by precedence. That lives here for the same reason the constants do: it is
deliberately free of torch and model imports so it can be unit tested in
isolation, and its only data dependency — the supported conv types — is already
here. `src/models/gnn/shepherd_gnn.py:build_shepherd_model` is its caller.

Module: src/config/model_types.py
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Conv types the GNN factory (src/models/gnn/layers.py) can build. Adding a new
# architecture = implement its branch in HeteroGNNLayer AND add its name here.
SUPPORTED_CONV_TYPES = ("hgt", "gat", "sage")

# Default when a conv type is missing or unrecognised.
DEFAULT_CONV_TYPE = "gat"


# ==============================================================================
# Checkpoint architecture inference
# ==============================================================================
def _infer_conv_type_from_keys(state_keys) -> Optional[str]:
    """Infer the GNN conv type from a checkpoint's parameter names.

    Older checkpoints don't record ``conv_type`` in their config (the trainer
    serialized only training hyperparameters), so we recover it from the weight
    names — otherwise an HGT/SAGE checkpoint is silently rebuilt as the GAT
    default and the state_dict load fails.

    Distinguishing markers (params are under ``gnn_layers.<n>.conv...``):
      - HGT (``HGTConv``): ``kqv_lin`` / ``k_rel`` / ``v_rel`` / ``p_rel`` / ``skip``
      - GAT (``GATConv`` in ``HeteroConv``): ``att_src`` / ``att_dst``
      - SAGE (``SAGEConv`` in ``HeteroConv``): ``lin_l`` / ``lin_r`` (no attention)
    """
    keys = state_keys if isinstance(state_keys, (set, frozenset)) else set(state_keys)
    if any(
        (".conv.kqv_lin." in k or ".conv.k_rel" in k or ".conv.v_rel" in k
         or ".conv.p_rel." in k or ".conv.skip." in k)
        for k in keys
    ):
        return "hgt"
    if any(k.endswith(".att_src") or k.endswith(".att_dst") for k in keys):
        return "gat"
    if any((".conv.convs." in k) and (".lin_l." in k or ".lin_r." in k) for k in keys):
        return "sage"
    return None


def _infer_num_layers_from_keys(state_keys) -> Optional[int]:
    """Infer the number of GNN layers from the highest ``gnn_layers.<n>`` index."""
    import re

    indices = set()
    for k in state_keys:
        m = re.match(r"gnn_layers\.(\d+)\.", k)
        if m:
            indices.add(int(m.group(1)))
    return (max(indices) + 1) if indices else None


def resolve_arch_params(
    ckpt_config: dict,
    state_keys,
    *,
    valid_fields,
    supported_conv,
    has_pos_encoder: bool = False,
    has_ortholog_gate: bool = False,
) -> dict:
    """Resolve model-architecture kwargs from a checkpoint, by precedence.

    This is the schema resolution the diagnosis pipeline uses to reconstruct a
    model. It is deliberately free of torch/model imports so it can be unit
    tested in isolation: the caller passes ``valid_fields`` (the current
    ``ShepherdGNNConfig`` field names) and ``supported_conv`` (the conv types the
    factory can build).

    Precedence (highest first):
      1. ``ckpt_config["model_config"]`` — the full, self-describing sub-dict
         written by current trainers.
      2. Legacy flat arch fields at the top level of ``ckpt_config`` (written by
         the interim fix before ``model_config`` existed).
      3. Inference from the parameter names (``conv_type`` / ``num_layers``) for
         checkpoints that carry no architecture metadata at all.
      4. ``ShepherdGNNConfig`` defaults (applied by the caller when a key is
         simply absent from the returned dict).

    Unknown keys are filtered against ``valid_fields`` to tolerate version drift
    (and are logged). ``conv_type`` handling depends on where it came from, which
    is tracked explicitly:

      - ``model_config`` (tier 1) is the trainer's authoritative self-description
        and is TRUSTED over the weight-key heuristic (only warned on conflict).
        The heuristic is a legacy fallback that is not future-proof — a new
        architecture may reuse PyG key patterns (e.g. ``att_src`` / ``lin_l``),
        so it must not override an explicit model_config value.
      - ``legacy_flat`` (tier 2) predates ``model_config``; here the weights are
        treated as structural ground truth and override a conflicting value.
      - ``inferred`` (tier 3) is derived from the weights, so it cannot conflict.

    In all cases an explicit-but-unsupported ``conv_type`` raises rather than
    silently degrading to GAT; only a truly absent/undetectable one defaults.
    """
    params: dict = {}
    conv_source = None  # "model_config" | "legacy_flat" | "inferred"

    # Tier 1: full self-describing model_config sub-dict.
    model_config = ckpt_config.get("model_config")
    if isinstance(model_config, dict):
        ignored = []
        for k, v in model_config.items():
            if k in valid_fields:
                params[k] = v
            else:
                ignored.append(k)
        if ignored:
            logger.warning(
                "Ignoring unknown model_config field(s) not in the current "
                "ShepherdGNNConfig schema: %s",
                sorted(ignored),
            )
        if "conv_type" in params:
            conv_source = "model_config"

    # Tier 2: legacy flat arch fields (fill only what tier 1 didn't provide).
    for key in (
        "conv_type",
        "hidden_dim",
        "num_layers",
        "num_heads",
        "use_positional_encoding",
        "use_ortholog_gate",
    ):
        if key not in params and key in valid_fields and ckpt_config.get(key) is not None:
            params[key] = ckpt_config[key]
            if key == "conv_type":
                conv_source = "legacy_flat"

    # Tier 3: infer structural fields from the parameter names when still absent.
    detected_conv = _infer_conv_type_from_keys(state_keys)
    if "conv_type" not in params and detected_conv:
        params["conv_type"] = detected_conv
        conv_source = "inferred"
        logger.info(
            "conv_type not in checkpoint config; detected %r from weights.",
            detected_conv,
        )
    if "num_layers" not in params:
        detected_layers = _infer_num_layers_from_keys(state_keys)
        if detected_layers:
            params["num_layers"] = detected_layers
    if "use_positional_encoding" in valid_fields:
        params.setdefault("use_positional_encoding", has_pos_encoder)
    if "use_ortholog_gate" in valid_fields:
        params.setdefault("use_ortholog_gate", has_ortholog_gate)

    # Conflict handling depends on the SOURCE of conv_type:
    #   - legacy_flat: weights are ground truth -> override on conflict.
    #   - model_config: authoritative -> keep it, only warn (the key heuristic is
    #     a legacy fallback that may misclassify future architectures).
    conv_type = params.get("conv_type")
    if conv_type is not None and detected_conv and conv_type != detected_conv:
        if conv_source == "legacy_flat":
            logger.warning(
                "Legacy flat conv_type=%r disagrees with parameter names "
                "(detected %r); trusting the weights.",
                conv_type, detected_conv,
            )
            params["conv_type"] = detected_conv
        else:  # model_config
            logger.warning(
                "model_config conv_type=%r disagrees with the weight-key "
                "heuristic (detected %r); trusting model_config, as the heuristic "
                "is a legacy fallback that may misclassify newer architectures.",
                conv_type, detected_conv,
            )

    # Validate: an explicit-but-unsupported conv_type must fail loudly. Only a
    # truly absent/undetectable conv_type falls back to the GAT default.
    conv_type = params.get("conv_type")
    if conv_type is None:
        params["conv_type"] = "gat"
    elif conv_type not in supported_conv:
        raise ValueError(
            f"Checkpoint specifies unsupported conv_type={conv_type!r}; "
            f"supported types are {tuple(supported_conv)}."
        )

    # Inference-time override (never carry training dropout into eval).
    if "dropout" in valid_fields:
        params["dropout"] = 0.0

    return params


__all__ = ["SUPPORTED_CONV_TYPES", "DEFAULT_CONV_TYPE", "resolve_arch_params"]
