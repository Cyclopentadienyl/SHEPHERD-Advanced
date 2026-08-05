"""
API middleware — RESERVED package (no implementation yet).
==========================================================
Reserved home for middleware currently defined inline in ``src/api/main.py``.

The concern is live, not hypothetical: ``src.api.main`` defines the request-logging
middleware ``log_requests`` inline via ``@app.middleware("http")`` (together with
its ``_QUIET_PREFIXES`` filter), and that module's own docstring lists "CORS and
security middleware" as part of the API service. Extracting the locally-defined
middleware here is part of decomposing ``main.py``'s bootstrap / app-state /
middleware concerns.

Scope note: the ``CORSMiddleware`` registration in ``src.api.main`` is
third-party (``fastapi.middleware.cors``) and would stay a registration call in
``main.py``. Only middleware this project defines belongs here.

Status: intentionally empty until that extraction is done. Nothing imports this
package.

Module: src/api/middleware/__init__.py
"""
