"""
D8's check: may this request use an operator control?
======================================================
`docs/working/PLAN_TEST_RESULTS.md` D8 and §6 ("M5's boundaries"); SPEC_4 §3
(deployment mode M1). **The one check**: the FastAPI dependency
(`src/api/operator_access.py`) and the Gradio helper
(`src/webui/operator_access.py`) both call `check_operator_access`, each on the
request it was given, and nothing else decides.

An operator control (a reload, a configuration save, a restart, a training or
test job, a registration, the checkpoint list) is allowed only when **all** of
these hold, checked in this order; the first that fails is the one reported:

1. the deployment setting says `M1` (`server_settings.deployment_mode`, read
   now, never inferred);
2. the request can be read: an HTTP or WebSocket scope whose ``server`` is a
   (host, port) pair and whose headers are a list of byte pairs;
3. it arrived on a loopback socket: the scope's own ``server``, the local
   address of the socket that accepted it (127.0.0.0/8, ``::1``, or an
   IPv4-mapped form of one);
4. it carries no forwarding header (`FORWARDING_HEADERS`): M1 excludes a
   reverse proxy, behind which every request arrives on loopback;
5. its ``Host`` header, exactly one, names ``localhost``, ``127.0.0.1`` or
   ``[::1]``, with any port (an SSH forward's local port may differ). A
   browser that reached this server under another name -- DNS rebinding --
   sends that name;
6. its ``Origin`` header, when present (at most one), is ``http://<Host>`` or
   ``https://<Host>``: a browser puts the sending page's origin there on a
   POST and on any script's cross-origin request. A request with no Origin (a
   command-line client, a page loaded from the address bar) passes;
7. its ``Sec-Fetch-Site`` header, when present (at most one), is
   ``same-origin`` or ``none`` (a request the user started outside any web
   page: the address bar, a bookmark, a link opened from another program). A
   browser sends **no** Origin on a GET that another site's page makes with an
   image, a script, a stylesheet or a link, so condition 6 cannot see one;
   current browsers (Chrome and Edge 76, Firefox 90, Safari 16.4 and later)
   say where it came from here: ``cross-site``, or ``same-site`` for another
   port or subdomain of the same host. Command-line clients send none and
   pass. An older browser that sends neither header is not caught on such a
   GET.

**Never** ``scope["client"]``: uvicorn rewrites it from ``X-Forwarded-For``
when it trusts proxy headers, which is its default for a peer on 127.0.0.1.
The ``server`` field and the headers are the triggering request's own; the
address F1 records (`server_address.py`) is the latest of anyone's and is
never read here.

**Fails closed.** Anything that cannot be read -- no scope, a server field of
another shape, headers that are not byte pairs, any exception while reading --
is a refusal, never a pass.

**Refusals name the failed condition in words an operator can act on, and never
repeat a header's value**: a page may show the reason, and a header's value is
whatever the client sent. A forwarding header is named by its (fixed) name.

**What it is not.** Loopback is not authorisation (SPEC_4 §3): several SSH
forwards all arrive on loopback, which is why M1 names a verified
single-workspace deployment operated by its owner or a designated technician.
A silent same-host proxy, one that adds no forwarding header, is excluded by
the deployment guide, not detected. Nothing here authenticates anyone.

Pure: it reads the scope and the environment, writes nothing and keeps no
state. Platform: standard library only.

Module: src/utils/operator_access.py
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Optional, Tuple

from src.config import server_settings

#: The headers a proxy adds to say a request was forwarded (D8's list, with the
#: extensions the D8 map found in use). Compared by lowercased name.
FORWARDING_HEADERS: Tuple[str, ...] = (
    "forwarded",
    "x-forwarded-for",
    "x-forwarded-host",
    "x-forwarded-proto",
    "x-real-ip",
    "x-original-forwarded-for",
    "cf-connecting-ip",
    "true-client-ip",
)

#: The host names a loopback Host header may carry, before its port.
LOOPBACK_HOST_NAMES: Tuple[str, ...] = ("localhost", "127.0.0.1", "[::1]")

_HOST = re.compile(r"^(localhost|127\.0\.0\.1|\[::1\])(?::([0-9]{1,5}))?$", re.IGNORECASE)

#: Each condition a decision can report, by the code `OperatorAccess.condition`
#: carries (the UI's chip and the tests read the code, people read the reason).
CONDITIONS: Tuple[str, ...] = (
    "enabled",
    "deployment_mode",
    "unreadable_request",
    "not_loopback",
    "forwarding_header",
    "host_not_loopback",
    "origin_mismatch",
    "cross_site",
)

#: What every operator control is called in the reasons below.
_CONTROLS = "operator controls"

_UNREADABLE = (
    "This request's arrival could not be read, so whether it may use the operator controls "
    "cannot be checked; it is refused (D8 fails closed)."
)
_NOT_LOOPBACK = (
    "This request did not arrive on a loopback address of this machine. The operator controls "
    "work only on the machine itself, or through an SSH forward to localhost (for example "
    "ssh -L 8264:localhost:8264 <server>, then http://localhost:8264/ui)."
)
_HOST_NOT_LOOPBACK = (
    "This request was not addressed to localhost, 127.0.0.1 or [::1] (its Host header names "
    "another host, or there is not exactly one). Open the dashboard as http://localhost:<port>/ui "
    "on the machine or through an SSH forward; a page reached under another name could be a "
    "DNS-rebinding attack, so the operator controls are refused there."
)
_ORIGIN_MISMATCH = (
    "This request was sent by a page whose origin is not this dashboard's address (its Origin "
    "header does not match its Host), as a request made by another website would be. Use the "
    "operator controls from the dashboard itself."
)
_CROSS_SITE = (
    "This request was made by a page that is not this dashboard: the browser reports it as sent "
    "from another site or another address of this machine (its Sec-Fetch-Site header is not "
    "same-origin), as an image, script or link on another website's page would send it. Use the "
    "operator controls from the dashboard itself, or open the address directly."
)

#: The ``Sec-Fetch-Site`` values condition 7 allows: the dashboard's own page,
#: and a request the user started outside any web page (the address bar, a
#: bookmark, a link opened from another program).
_OWN_SITE = ("same-origin", "none")


class OperatorAccess(NamedTuple):
    """D8's answer for one request: `allowed`, the condition that decided it
    (`CONDITIONS`; ``"enabled"`` when allowed) and `reason`, a sentence an
    operator can act on that repeats no header value."""

    allowed: bool
    condition: str
    reason: str

    @property
    def refusal(self) -> Optional[str]:
        """The message a refused control shows, saying nothing was done;
        `None` when allowed."""
        if self.allowed:
            return None
        return f"Refused: {self.reason} Nothing was done."

    def as_status(self) -> Dict[str, Any]:
        """The decision as JSON-ready data, for a read-only status (the UI's
        chip): ``enabled``, ``condition`` and ``reason``."""
        return {"enabled": self.allowed, "condition": self.condition, "reason": self.reason}


ENABLED = OperatorAccess(True, "enabled", (
    f"The {_CONTROLS} are enabled for this request: deployment mode M1, a loopback arrival "
    "with no forwarding header, a loopback Host, no foreign Origin and not sent from another "
    "site."))


def _refused(condition: str, reason: str) -> OperatorAccess:
    return OperatorAccess(False, condition, reason)


def _header_pairs(raw: Any) -> List[Tuple[str, str]]:
    """The scope's headers as (lowercased name, value) text pairs, in order.

    ASGI gives them as an iterable of two-item byte sequences, latin-1 by the
    HTTP rules. Anything else raises ``ValueError``, which the caller turns
    into a refusal.
    """
    if raw is None or isinstance(raw, (str, bytes, Mapping)):
        raise ValueError("the scope's headers are not a list of name and value pairs")
    pairs = []
    for item in raw:
        name, value = item  # a pair, or ValueError / TypeError
        if not isinstance(name, (bytes, bytearray)) or not isinstance(value, (bytes, bytearray)):
            raise ValueError("a header is not a pair of byte strings")
        pairs.append((bytes(name).decode("latin-1").lower(), bytes(value).decode("latin-1")))
    return pairs


def _values(pairs: Iterable[Tuple[str, str]], name: str) -> List[str]:
    return [value for key, value in pairs if key == name]


def check_operator_access(
    scope: Any, *, environ: Optional[Mapping[str, str]] = None
) -> OperatorAccess:
    """D8's decision for the request whose ASGI scope this is.

    `scope` is the triggering request's own (``request.scope`` in a route,
    ``request.request.scope`` in a Gradio handler); its ``headers`` are the
    raw pairs the server received, so a repeated header is seen as repeated.
    `environ` is the environment to read the deployment setting from (the
    process's when `None`). Never raises: whatever cannot be read is a
    refusal (`CONDITIONS`).
    """
    try:
        mode = server_settings.deployment_mode(environ)
        if not mode.available:
            return _refused("deployment_mode", mode.problem)
        return _check_request(scope)
    except Exception:  # noqa: BLE001 - D8 fails closed, whatever went wrong
        return _refused("unreadable_request", _UNREADABLE)


def _check_request(scope: Any) -> OperatorAccess:
    """Conditions 2 to 7, for a deployment in mode M1."""
    if not isinstance(scope, Mapping) or scope.get("type") not in ("http", "websocket"):
        return _refused("unreadable_request", _UNREADABLE)
    server = scope.get("server")
    if not (isinstance(server, (tuple, list)) and len(server) == 2
            and isinstance(server[0], str)):
        return _refused("unreadable_request", _UNREADABLE)
    try:
        headers = _header_pairs(scope.get("headers"))
    except (TypeError, ValueError):
        return _refused("unreadable_request", _UNREADABLE)

    if not server_settings.is_loopback_ip(server[0]):
        return _refused("not_loopback", _NOT_LOOPBACK)

    present = [name for name in FORWARDING_HEADERS if _values(headers, name)]
    if present:
        return _refused("forwarding_header", (
            f"This request carries a forwarding header ({present[0]}), which a reverse proxy "
            "adds. Deployment mode M1 excludes a reverse proxy, behind which every request "
            "arrives on loopback, so the operator controls are refused through one; use them "
            "on the machine or through an SSH forward to localhost."))

    hosts = _values(headers, "host")
    host = hosts[0].strip() if len(hosts) == 1 else None
    match = _HOST.match(host) if host is not None else None
    if match is None or (match.group(2) is not None and int(match.group(2)) > 65535):
        return _refused("host_not_loopback", _HOST_NOT_LOOPBACK)

    origins = _values(headers, "origin")
    if origins:
        allowed = {f"http://{host.lower()}", f"https://{host.lower()}"}
        if len(origins) != 1 or origins[0].strip().lower() not in allowed:
            return _refused("origin_mismatch", _ORIGIN_MISMATCH)

    sites = _values(headers, "sec-fetch-site")
    if sites and (len(sites) != 1 or sites[0].strip().lower() not in _OWN_SITE):
        return _refused("cross_site", _CROSS_SITE)
    return ENABLED


__all__ = ["CONDITIONS", "ENABLED", "FORWARDING_HEADERS", "LOOPBACK_HOST_NAMES",
           "OperatorAccess", "check_operator_access"]
