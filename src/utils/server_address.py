"""
The address this process's HTTP server accepted its requests on.
================================================================
The WebUI is mounted inside the API's own process (`src/api/main.py` mounts it at
`/ui`), and the Diagnosis tab reaches the API over HTTP. It used to do so at a
hard-coded `http://127.0.0.1:8000`, which is only uvicorn's default: the systemd
unit passes `--port 8264`, uvicorn takes the last `--port` it is given, and the
tab then called a port nothing was listening on.

**The address comes from the server, not from configuration.** Every HTTP request
carries the ASGI ``server`` field — the local (host, port) of the socket that
accepted it — and `src/api/middleware/server_address.py` records it here. A
Diagnosis callback only ever runs because a request reached this server, so by
the time one asks, the address is known. There is no second copy of the port to
keep in step with the launcher, the unit, `python -m src.api.main` or a bare
`uvicorn` command line, and no list of ports to try.

**Any recorded address will do.** Under a wildcard bind, connections arrive on
different local addresses — loopback through an SSH forward, the LAN address
directly — and each is this server's own socket, reachable from this process. The
latest one is kept.

The scheme stays ``http``, as before. The address is the server's own socket,
reached from inside the same process, not the URL a browser used.

Lives in `src.utils` because both sides need it without importing each other's
package: `src/api/__init__.py` imports `src.api.main`, which builds the WebUI, so
the WebUI cannot import anything under `src.api` without building the app.

Module: src/utils/server_address.py
"""
from typing import Any, Optional, Tuple

_address: Optional[Tuple[str, int]] = None


class ServerAddressUnknownError(RuntimeError):
    """No request has reached this server yet, so its address is not known."""


def record_server_address(server: Any) -> None:
    """Remember a request's ASGI ``scope["server"]`` value.

    A value without a host and port is ignored rather than recorded: uvicorn
    reports ``None`` for a Unix-domain socket, and there is no http URL to build
    from it. Ignoring it keeps a TCP address recorded earlier, which is still this
    server's.
    """
    global _address
    if not isinstance(server, (tuple, list)) or len(server) != 2:
        return
    host, port = server
    if not host or port is None:
        return
    _address = (str(host), int(port))


def api_base_url() -> str:
    """``http://<host>:<port>`` of this server, from the last request it accepted.

    Raises:
        ServerAddressUnknownError: no request has been recorded. In a running
            server this cannot happen inside a request handler; it means the
            caller ran before the server served anything, for example while the
            app was being built.
    """
    if _address is None:
        raise ServerAddressUnknownError(
            "the server's address is not known yet: no request has reached it. "
            "It is recorded from each incoming request, so this call ran before "
            "the server served one."
        )
    host, port = _address
    if ":" in host:  # an IPv6 literal is bracketed in a URL
        host = f"[{host}]"
    return f"http://{host}:{port}"


__all__ = ["ServerAddressUnknownError", "api_base_url", "record_server_address"]
