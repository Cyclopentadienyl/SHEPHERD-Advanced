"""
The deployment's device setting
===============================
`SHEPHERD_DEVICE`, read where every consumer can reach it: the served pipeline's
build and reload (`src/api/main.py`, `src/api/routes/pipeline.py`) and the job
runner, which gives a test run the deployment's device
(`src/api/services/training_manager.py`, D5). Kept out of the composition root
so that a service reading it does not build the application to do so.

Module: src/config/serving_device.py
"""
from __future__ import annotations

import os
from typing import Optional

#: The deployment's device: unset or `auto` for CUDA, `cpu` for development.
DEVICE_ENV = "SHEPHERD_DEVICE"


def requested_serving_device(device: Optional[str]) -> Optional[str]:
    """The device a build is asked for: the request's, else `SHEPHERD_DEVICE`,
    else None (`auto`).

    **Blank is unset at both steps**, so `""` and `"  "` defer to the
    deployment's setting alike, as `resolve_serving_device` treats them alike
    (contract M3c). The value is passed on as given, not resolved: the served
    identity records what was asked for beside what took effect.
    """
    for value in (device, os.environ.get(DEVICE_ENV)):
        if value is not None and value.strip():
            return value.strip()
    return None


__all__ = ["DEVICE_ENV", "requested_serving_device"]
