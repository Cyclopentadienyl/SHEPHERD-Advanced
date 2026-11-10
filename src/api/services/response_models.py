"""
The base every strict API answer shape is built on, for routes and services alike.
==================================================================================
An answer's shape is its contract: a model built on `StrictModel` refuses a key
it does not declare, and a digest field takes `SHA256_PATTERN`. Defined here,
in the service layer, so that a service that builds or reads such a shape (the
one reader of recorded results, `record_views`, and through it the results
service and its export) never imports a route (ARCHITECTURE §12). The routes
that answer with these shapes (`src/api/routes/diagnose.py`,
`src/api/routes/evaluations.py`) import them from here. Moved here from the
Diagnosis route in research comparison Stage 8; unchanged.

Module: src/api/services/response_models.py
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

#: A SHA-256 as an answer gives it: 64 lowercase hexadecimal digits.
SHA256_PATTERN = r"^[0-9a-f]{64}$"


class StrictModel(BaseModel):
    """An unknown key is refused: the shape is the contract."""

    model_config = ConfigDict(extra="forbid")


__all__ = ["SHA256_PATTERN", "StrictModel"]
