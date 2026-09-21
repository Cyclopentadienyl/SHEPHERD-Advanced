"""
Where an ontology may be fetched from, and the one place that decides it.
=========================================================================
`PLAN_ONTOLOGY_PHASE2.md` §3.5 and §3.5.1. Two rules, both enforced **at the
request boundary** rather than where a setting is read, because every source
this project configures is a PURL and a PURL is a redirect:

1. **Scheme.** `http` and `https` only.
2. **Destination.** Every address the host resolves to must be globally
   routable, unless an administrator put that host on the allow list.

**Why the boundary and not the setting.** A check performed when the list is
read has already passed by the time a 302 arrives. Measured on CPython 3.13.9,
`HTTPRedirectHandler.http_error_302` permits redirect targets whose scheme is in
``('http', 'https', 'ftp', '')`` — so an `https` source that redirects to `ftp`
is followed, and a config-time allowlist never sees it.

**Why addresses and not hostnames.** A literal private IP in a URL has no
hostname to match against a list, and a name that resolves into the hospital's
network is inside it whatever it is called. `is_global` is one predicate over
both address families and covers loopback, private, link-local, reserved and
unspecified — so the rule is a fact about where a request would go rather than a
pattern over how it was spelled.

**The residual, stated rather than implied away.** This is check-then-connect:
the resolution this module sees is not the one the socket uses, so a name whose
answer changes between them is not covered. Closing that means connecting to a
pinned address while preserving the `Host` header, which is a larger change than
Phase 2 and **is not claimed here**. What this does cover is the ordinary
cases — a misconfigured URL, a redirect into the internal network, a literal
internal address — which is what a build tool fetching four known artifacts is
actually exposed to.

**What this is not.** Not a judgement that a destination is trustworthy: neither
`https` nor `is_global` says anything about what arrives. The imports policy
(§3.3) and the role check (§3.4) are what look at the content, and a permitted
destination does not excuse either.

Module: src/ontology/download.py

Dependencies: the standard library only. No torch, no pronto.
"""
from __future__ import annotations

import ipaddress
import logging
import os
import socket
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

__all__ = [
    "ALLOWED_SCHEMES",
    "DestinationPolicy",
    "OntologyDownloadError",
    "check_destination",
    "download_ontology",
]

#: The only schemes a source may use. `urlretrieve` itself would accept
#: `file://`, `ftp://` and `data:` — its opener carries `FileHandler`,
#: `FTPHandler` and `DataHandler` — so a field named "download URL" has to say
#: what it means rather than inherit whatever the library supports.
ALLOWED_SCHEMES = ("http", "https")

#: How many redirects to follow before giving up. Bounded because each hop is a
#: fresh destination decision and an unbounded chain is an unbounded number of
#: them.
MAX_REDIRECTS = 5


class OntologyDownloadError(RuntimeError):
    """A fetch was refused or failed. Never a silent fallback to something else."""


@dataclass(frozen=True)
class DestinationPolicy:
    """What this server may fetch from.

    `allowed_hosts` is the administrator's exemption and the reason the default
    is not a blanket block on the hospital's own network: an in-house mirror is
    a legitimate and probably desirable source, reached by saying so on purpose.

    `resolver` exists so the rule can be tested without DNS. It is not a hook
    for relaxing the policy — it answers "what does this name resolve to", and
    everything the policy decides is decided from its answer.
    """

    allowed_hosts: Tuple[str, ...] = ()
    resolver: Optional[Callable[[str], Sequence[str]]] = field(default=None, compare=False)

    def permits_host(self, host: str) -> bool:
        return host.lower() in {item.lower() for item in self.allowed_hosts}

    def resolve(self, host: str) -> List[str]:
        if self.resolver is not None:
            return list(self.resolver(host))
        try:
            infos = socket.getaddrinfo(host, None)
        except OSError as exc:
            raise OntologyDownloadError(
                f"{host} could not be resolved ({type(exc).__name__}: {exc}), so "
                "where a request to it would go is unknown — which is not the "
                "same as its being acceptable"
            ) from exc
        return [info[4][0] for info in infos]


def check_destination(url: str, policy: DestinationPolicy, *, why: str = "source") -> None:
    """Refuse a URL this server may not fetch. **Both rules, in one place.**

    Called for the initial URL and again for every redirect target, which is the
    whole point: a gate that runs once has already been passed when the redirect
    arrives.

    Raises:
        OntologyDownloadError: naming the URL and which rule refused it.
    """
    parts = urlsplit(url)
    scheme = (parts.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise OntologyDownloadError(
            f"the {why} {url!r} uses the scheme {scheme or '(none)'!r}; only "
            f"{', '.join(ALLOWED_SCHEMES)} are permitted. urlretrieve would "
            "otherwise accept file://, ftp:// and data: URLs, which is more "
            "than a download URL should mean."
        )

    host = parts.hostname
    if not host:
        raise OntologyDownloadError(f"the {why} {url!r} names no host")

    if policy.permits_host(host):
        logger.info("%s: %s is on the configured allow list", why, host)
        return

    # **A literal address is already the answer, and is not looked up.**
    # `socket.getaddrinfo` happens to echo a literal back, so resolving one
    # works by accident — and a resolver that caches, rewrites or is supplied
    # for a test does not owe that behaviour. Relying on it made
    # `http://169.254.169.254/` pass whenever the resolver did not
    # special-case literals, which is precisely the address this rule exists
    # to refuse.
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        _require_global(literal, host, url, why)
        return

    addresses = policy.resolve(host)
    if not addresses:
        raise OntologyDownloadError(
            f"the {why} {url!r} resolved to no address at all, so where a "
            "request to it would go is unknown"
        )

    for raw in addresses:
        try:
            address = ipaddress.ip_address(raw)
        except ValueError:
            raise OntologyDownloadError(
                f"the {why} {url!r} resolved to {raw!r}, which is not an "
                "address this policy can classify"
            ) from None
        # **Every answer, not one of them.** A name resolving to a public
        # address and a private one is a name that can reach the private one,
        # and admitting it because the first answer was acceptable is the whole
        # gap in one line.
        _require_global(address, host, url, why)


def _require_global(address, host: str, url: str, why: str) -> None:
    """One classification, so a literal and a resolved answer are judged alike."""
    if address.is_global:
        return
    raise OntologyDownloadError(
        f"the {why} {url!r} resolves to {address}, which is not a globally "
        "routable address (loopback, private, link-local or reserved). This "
        "server does not fetch from its own network unless an administrator "
        f"lists the host: add {host!r} to the ontology sources' allowed hosts "
        "if that is an in-house mirror."
    )


class _GuardedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Every redirect target goes through the same gate as the first URL.

    The stdlib handler permits `ftp` targets, so an `https` source can leave the
    allowed schemes without anything noticing. Subclassing rather than
    post-checking, because by the time a caller sees the final URL the requests
    have already been made.
    """

    def __init__(self, policy: DestinationPolicy):
        self._policy = policy
        self.hops: List[str] = []

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_destination(newurl, self._policy, why="redirect target")
        self.hops.append(newurl)
        if len(self.hops) > MAX_REDIRECTS:
            raise OntologyDownloadError(
                f"more than {MAX_REDIRECTS} redirects while fetching an "
                "ontology; each hop is a separate destination decision and the "
                "chain is bounded on purpose"
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_ontology(
    url: str,
    destination: Any,
    *,
    policy: Optional[DestinationPolicy] = None,
    opener_factory: Optional[Callable[..., Any]] = None,
    timeout: float = 60.0,
) -> Path:
    """Fetch one ontology file through the one guarded path.

    **Every fetch goes through here**, the OBO attempt and the OWL fallback
    alike. A fallback with its own fetch is the gate not existing: the rules
    would hold for the path that is usually taken and not for the one taken when
    something has already gone wrong.

    Written whole to a temporary file beside the destination and moved into
    place, so a failed or refused fetch leaves whatever was there untouched
    rather than truncated.

    Returns:
        The path written.

    Raises:
        OntologyDownloadError: refused by policy, or the transfer failed.
    """
    policy = policy or DestinationPolicy()
    destination = Path(destination)

    check_destination(url, policy, why="source")

    handler = _GuardedRedirectHandler(policy)
    factory = opener_factory or urllib.request.build_opener
    opener = factory(handler)

    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "wb", dir=str(destination.parent), prefix=destination.name,
        suffix=".part", delete=False,
    )
    staged = Path(handle.name)
    try:
        with handle:
            with opener.open(url, timeout=timeout) as response:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staged, destination)
    except OntologyDownloadError:
        staged.unlink(missing_ok=True)
        raise
    except Exception as exc:
        staged.unlink(missing_ok=True)
        raise OntologyDownloadError(
            f"fetching {url} failed ({type(exc).__name__}: {exc})"
        ) from exc

    logger.info("fetched %s to %s", url, destination)
    return destination
