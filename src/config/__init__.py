"""
SHEPHERD-Advanced Configuration Package
=======================================
Config-related constants and (reserved) deployment-config helpers.

Config authority (decision record: docs/CONFIG_AUTHORITY.md):
    Runtime configuration is owned by the dataclasses closest to each concern —
    TrainerConfig (src/training/trainer.py), PipelineConfig (src/inference/pipeline.py),
    and ShepherdGNNConfig (src/models/gnn/shepherd_gnn.py). There is intentionally NO
    central HyperparameterManager: a single global config object would couple the
    training, inference, model, API and UI layers together. The one genuinely shared
    config concern — training-parameter metadata shared by the API request model and
    the WebUI — will live in a small field-spec module (docs/CONFIG_AUTHORITY.md, B-lite).

Contents:
    - model_types.py       : torch-free conv-type constants
                             (SUPPORTED_CONV_TYPES, DEFAULT_CONV_TYPE).
    - config_validator.py,
      schema_loader.py     : RESERVED homes for committed-config validation
                             (configs/deployment.yaml + configs/schemas/*.json).
                             Empty by design until deployment validation is built.

Module: src/config/__init__.py
"""

# No package-level re-exports: import config submodules directly, e.g.
#   from src.config.model_types import SUPPORTED_CONV_TYPES, DEFAULT_CONV_TYPE
__all__: list[str] = []
