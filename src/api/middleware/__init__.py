"""
API middleware — RESERVED package (no implementation yet).
==========================================================
Reserved home for middleware currently defined inline in ``src/api/main.py``.

The concern is live, not hypothetical: ``main.py:180`` defines a request-logging
middleware via ``@app.middleware("http")``, and the module docstring
(``main.py:13``) lists "CORS and security middleware" as part of the API
service. Extracting the locally-defined middleware here is part of decomposing
``main.py``'s bootstrap / app-state / middleware concerns.

Scope note: the CORS middleware registered at ``main.py:161`` is third-party
(``fastapi.middleware.cors.CORSMiddleware``) and would stay a registration call
in ``main.py``. Only middleware this project defines belongs here.

Status: intentionally empty until that extraction is done. Nothing imports this
package.

Module: src/api/middleware/__init__.py
"""
