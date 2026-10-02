"""
API middleware defined by this project.
=======================================
Home for middleware this project defines, registered by ``src/api/main.py``.

- ``server_address.RecordServerAddress`` records the socket each request arrived
  on, so the in-process WebUI calls the API at the address the server actually
  bound (`src/utils/server_address.py`).

Still inline, and still to move here: ``src.api.main`` defines the
request-logging middleware ``log_requests`` via ``@app.middleware("http")``
(together with its ``_QUIET_PREFIXES`` filter), and that module's own docstring
lists "CORS and security middleware" as part of the API service. Extracting it is
part of decomposing ``main.py``'s bootstrap / app-state / middleware concerns.

Scope note: the ``CORSMiddleware`` registration in ``src.api.main`` is
third-party (``fastapi.middleware.cors``) and would stay a registration call in
``main.py``. Only middleware this project defines belongs here.

Module: src/api/middleware/__init__.py
"""
